//! Physics world interface, backend abstraction, and the primary `PhysicsWorld` struct.
//!
//! `PhysicsWorld` owns all rigid bodies, colliders, and constraints. It provides
//! the main API for stepping the simulation, managing bodies, and performing queries.

use std::collections::HashMap;

use glam::{Quat, Vec3};
use thiserror::Error;

use crate::collision::{
    ray_vs_shape, ColliderDesc, CollisionEvent, CollisionShape,
    ContactManifold, NarrowPhase, SpatialHashGrid, layers_interact,
};
use crate::dynamics::{
    BallJoint, Constraint, ConstraintHandle, ContactConstraint, FixedJoint, ForceMode, HingeJoint,
    JointDesc, RigidBody, SpringJoint,
};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors produced by physics operations.
#[derive(Debug, Error)]
pub enum PhysicsError {
    #[error("rigid body handle {0:?} is invalid or has been removed")]
    InvalidBodyHandle(RigidBodyHandle),

    #[error("collider handle {0:?} is invalid or has been removed")]
    InvalidColliderHandle(ColliderHandle),

    #[error("backend error: {0}")]
    BackendError(String),

    #[error("physics world has not been initialized")]
    NotInitialized,
}

pub type PhysicsResult<T> = Result<T, PhysicsError>;

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

/// Opaque handle to a rigid body within the physics world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RigidBodyHandle(pub u64);

/// Opaque handle to a collider attached to a rigid body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColliderHandle(pub u64);

// ---------------------------------------------------------------------------
// Body type
// ---------------------------------------------------------------------------

/// Classification of rigid body simulation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    /// Does not move; infinite mass. Used for terrain, walls, etc.
    Static,
    /// Fully simulated with forces and collisions.
    Dynamic,
    /// Moved programmatically; affects dynamic bodies but is not affected by them.
    Kinematic,
}

impl Default for BodyType {
    fn default() -> Self {
        Self::Dynamic
    }
}

// ---------------------------------------------------------------------------
// Descriptors and data types
// ---------------------------------------------------------------------------

/// Descriptor used to create a new rigid body.
#[derive(Debug, Clone)]
pub struct RigidBodyDesc {
    pub body_type: BodyType,
    /// Mass in kilograms. Ignored for `Static` bodies.
    pub mass: f32,
    /// Coefficient of friction [0, 1].
    pub friction: f32,
    /// Coefficient of restitution (bounciness) [0, 1].
    pub restitution: f32,
    /// Linear velocity damping factor.
    pub linear_damping: f32,
    /// Angular velocity damping factor.
    pub angular_damping: f32,
    /// Initial position in world space.
    pub position: Vec3,
    /// Initial rotation in world space.
    pub rotation: Quat,
}

impl Default for RigidBodyDesc {
    fn default() -> Self {
        Self {
            body_type: BodyType::Dynamic,
            mass: 1.0,
            friction: 0.5,
            restitution: 0.3,
            linear_damping: 0.01,
            angular_damping: 0.05,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

/// Describes a physics material with surface interaction properties.
#[derive(Debug, Clone, Copy)]
pub struct PhysicsMaterial {
    /// Coefficient of friction [0, 1].
    pub friction: f32,
    /// Coefficient of restitution (bounciness) [0, 1].
    pub restitution: f32,
    /// Density in kg/m^3. Used for automatic mass calculation from collider volume.
    pub density: f32,
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self {
            friction: 0.5,
            restitution: 0.3,
            density: 1000.0,
        }
    }
}

/// Result of a raycast query against the physics world.
#[derive(Debug, Clone)]
pub struct RaycastHit {
    /// Handle of the body that was hit.
    pub body: RigidBodyHandle,
    /// Collider that was hit.
    pub collider: ColliderHandle,
    /// World-space point of intersection.
    pub point: Vec3,
    /// Surface normal at the hit point.
    pub normal: Vec3,
    /// Distance from ray origin to the hit point.
    pub distance: f32,
}

// ---------------------------------------------------------------------------
// Internal collider storage
// ---------------------------------------------------------------------------

/// Internal representation of a collider within the physics world.
#[derive(Debug, Clone)]
struct ColliderData {
    pub handle: ColliderHandle,
    pub body: RigidBodyHandle,
    pub desc: ColliderDesc,
}

// ---------------------------------------------------------------------------
// PhysicsWorld struct
// ---------------------------------------------------------------------------

/// The primary physics simulation. Owns all rigid bodies, colliders, constraints,
/// and the collision detection pipeline.
///
/// Call `step(dt)` each physics tick to advance the simulation.
pub struct PhysicsWorld {
    /// Global gravity vector.
    gravity: Vec3,

    /// All rigid bodies, stored contiguously for cache-friendly iteration.
    bodies: Vec<RigidBody>,
    /// Map from handle to index in the `bodies` vec.
    body_index_map: HashMap<RigidBodyHandle, usize>,
    /// Next body handle ID.
    next_body_id: u64,

    /// All colliders.
    colliders: HashMap<ColliderHandle, ColliderData>,
    /// Map from body handle to its collider handles.
    body_colliders: HashMap<RigidBodyHandle, Vec<ColliderHandle>>,
    /// Next collider handle ID.
    next_collider_id: u64,

    /// Broad phase spatial hash grid.
    broad_phase: SpatialHashGrid,

    /// Joint constraints.
    joints: Vec<Box<dyn Constraint>>,
    /// Map from constraint handle to index in the joints vec.
    joint_handle_map: HashMap<ConstraintHandle, usize>,
    /// Next joint handle ID.
    next_joint_id: u64,

    /// Collision events from the last step.
    collision_events: Vec<CollisionEvent>,

    /// Warm start data: cached contact constraints from the previous frame.
    warm_start_cache: Vec<ContactConstraint>,

    /// Set of pairs that were colliding in the previous frame (for begin/end events).
    previous_colliding_pairs: std::collections::HashSet<(ColliderHandle, ColliderHandle)>,

    /// Per-body friction/restitution for combining with collider materials.
    body_friction: HashMap<RigidBodyHandle, f32>,
    body_restitution: HashMap<RigidBodyHandle, f32>,
}

impl PhysicsWorld {
    /// Create a new physics world with the given gravity.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            gravity,
            bodies: Vec::new(),
            body_index_map: HashMap::new(),
            next_body_id: 1,
            colliders: HashMap::new(),
            body_colliders: HashMap::new(),
            next_collider_id: 1,
            broad_phase: SpatialHashGrid::new(4.0),
            joints: Vec::new(),
            joint_handle_map: HashMap::new(),
            next_joint_id: 1,
            collision_events: Vec::new(),
            warm_start_cache: Vec::new(),
            previous_colliding_pairs: std::collections::HashSet::new(),
            body_friction: HashMap::new(),
            body_restitution: HashMap::new(),
        }
    }

    /// Set the global gravity vector.
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.gravity = gravity;
    }

    /// Get the current global gravity vector.
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    // -----------------------------------------------------------------------
    // Body management
    // -----------------------------------------------------------------------

    /// Create a new rigid body from the given descriptor and return its handle.
    pub fn add_body(&mut self, desc: &RigidBodyDesc) -> PhysicsResult<RigidBodyHandle> {
        let handle = RigidBodyHandle(self.next_body_id);
        self.next_body_id += 1;

        let (mass, inv_mass, is_static, is_kinematic) = match desc.body_type {
            BodyType::Static => (0.0, 0.0, true, false),
            BodyType::Kinematic => (desc.mass, 0.0, false, true),
            BodyType::Dynamic => {
                let m = desc.mass.max(0.001);
                (m, 1.0 / m, false, false)
            }
        };

        let body = RigidBody {
            handle,
            position: desc.position,
            rotation: desc.rotation,
            linear_velocity: Vec3::ZERO,
            accumulated_force: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            accumulated_torque: Vec3::ZERO,
            mass,
            inv_mass,
            inertia_tensor: glam::Mat3::IDENTITY,
            inv_inertia_tensor: if is_static || is_kinematic {
                glam::Mat3::ZERO
            } else {
                glam::Mat3::IDENTITY
            },
            linear_damping: desc.linear_damping,
            angular_damping: desc.angular_damping,
            is_sleeping: false,
            sleep_counter: 0,
            gravity_scale: 1.0,
            is_static,
            is_kinematic,
        };

        let idx = self.bodies.len();
        self.bodies.push(body);
        self.body_index_map.insert(handle, idx);
        self.body_colliders.insert(handle, Vec::new());
        self.body_friction.insert(handle, desc.friction);
        self.body_restitution.insert(handle, desc.restitution);

        Ok(handle)
    }

    /// Remove a rigid body and all attached colliders from the world.
    pub fn remove_body(&mut self, handle: RigidBodyHandle) -> PhysicsResult<()> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;

        // Remove all attached colliders
        if let Some(collider_handles) = self.body_colliders.remove(&handle) {
            for ch in collider_handles {
                self.broad_phase.remove(ch);
                self.colliders.remove(&ch);
            }
        }

        // Remove joints referencing this body
        self.joints.retain(|j| {
            let (a, b) = j.bodies();
            a != handle && b != handle
        });
        // Rebuild joint handle map
        self.joint_handle_map.clear();

        // Remove the body (swap-remove for O(1))
        self.bodies.swap_remove(idx);
        self.body_index_map.remove(&handle);

        // Update the index of the body that was swapped in
        if idx < self.bodies.len() {
            let swapped_handle = self.bodies[idx].handle;
            self.body_index_map.insert(swapped_handle, idx);
        }

        self.body_friction.remove(&handle);
        self.body_restitution.remove(&handle);

        Ok(())
    }

    /// Get a reference to a body by handle.
    pub fn get_body(&self, handle: RigidBodyHandle) -> PhysicsResult<&RigidBody> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        Ok(&self.bodies[idx])
    }

    /// Get a mutable reference to a body by handle.
    pub fn get_body_mut(&mut self, handle: RigidBodyHandle) -> PhysicsResult<&mut RigidBody> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        Ok(&mut self.bodies[idx])
    }

    // -----------------------------------------------------------------------
    // Collider management
    // -----------------------------------------------------------------------

    /// Attach a collider to an existing rigid body. Returns the collider handle.
    pub fn add_collider(
        &mut self,
        body: RigidBodyHandle,
        desc: &ColliderDesc,
    ) -> PhysicsResult<ColliderHandle> {
        if !self.body_index_map.contains_key(&body) {
            return Err(PhysicsError::InvalidBodyHandle(body));
        }

        let handle = ColliderHandle(self.next_collider_id);
        self.next_collider_id += 1;

        let collider = ColliderData {
            handle,
            body,
            desc: desc.clone(),
        };

        self.colliders.insert(handle, collider);
        self.body_colliders
            .entry(body)
            .or_default()
            .push(handle);

        // Recompute inertia tensor from shape
        let body_idx = self.body_index_map[&body];
        let rb = &mut self.bodies[body_idx];
        if !rb.is_static && !rb.is_kinematic {
            let inertia = desc.shape.compute_inertia_tensor(rb.mass);
            rb.inertia_tensor = inertia;
            let det = inertia.determinant();
            rb.inv_inertia_tensor = if det.abs() > 1e-10 {
                inertia.inverse()
            } else {
                glam::Mat3::IDENTITY
            };
        }

        // Insert into broad phase
        let rb = &self.bodies[body_idx];
        let aabb = desc
            .shape
            .compute_aabb(rb.position + desc.offset, rb.rotation);
        self.broad_phase.update(handle, aabb);

        Ok(handle)
    }

    /// Remove a collider from the world.
    pub fn remove_collider(&mut self, handle: ColliderHandle) -> PhysicsResult<()> {
        let collider = self
            .colliders
            .remove(&handle)
            .ok_or(PhysicsError::InvalidColliderHandle(handle))?;

        self.broad_phase.remove(handle);

        if let Some(collider_list) = self.body_colliders.get_mut(&collider.body) {
            collider_list.retain(|h| *h != handle);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Force/torque application
    // -----------------------------------------------------------------------

    /// Apply a force or impulse to a rigid body.
    pub fn apply_force(
        &mut self,
        handle: RigidBodyHandle,
        force: Vec3,
        mode: ForceMode,
    ) -> PhysicsResult<()> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        self.bodies[idx].apply_force(force, mode);
        Ok(())
    }

    /// Apply a torque to a rigid body.
    pub fn apply_torque(
        &mut self,
        handle: RigidBodyHandle,
        torque: Vec3,
        mode: ForceMode,
    ) -> PhysicsResult<()> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        self.bodies[idx].apply_torque(torque, mode);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Transform queries
    // -----------------------------------------------------------------------

    /// Get the current position of a rigid body.
    pub fn get_position(&self, handle: RigidBodyHandle) -> PhysicsResult<Vec3> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        Ok(self.bodies[idx].position)
    }

    /// Get the current rotation of a rigid body.
    pub fn get_rotation(&self, handle: RigidBodyHandle) -> PhysicsResult<Quat> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        Ok(self.bodies[idx].rotation)
    }

    /// Get the linear velocity of a rigid body.
    pub fn get_linear_velocity(&self, handle: RigidBodyHandle) -> PhysicsResult<Vec3> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        Ok(self.bodies[idx].linear_velocity)
    }

    /// Get the angular velocity of a rigid body.
    pub fn get_angular_velocity(&self, handle: RigidBodyHandle) -> PhysicsResult<Vec3> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        Ok(self.bodies[idx].angular_velocity)
    }

    /// Set the world-space position of a rigid body.
    pub fn set_position(
        &mut self,
        handle: RigidBodyHandle,
        position: Vec3,
    ) -> PhysicsResult<()> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        self.bodies[idx].position = position;
        self.bodies[idx].wake_up();
        self.update_collider_aabbs(handle);
        Ok(())
    }

    /// Set the world-space rotation of a rigid body.
    pub fn set_rotation(&mut self, handle: RigidBodyHandle, rotation: Quat) -> PhysicsResult<()> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        self.bodies[idx].rotation = rotation;
        self.bodies[idx].wake_up();
        self.update_collider_aabbs(handle);
        Ok(())
    }

    /// Set the linear velocity of a rigid body.
    pub fn set_linear_velocity(
        &mut self,
        handle: RigidBodyHandle,
        vel: Vec3,
    ) -> PhysicsResult<()> {
        let idx = self
            .body_index_map
            .get(&handle)
            .copied()
            .ok_or(PhysicsError::InvalidBodyHandle(handle))?;
        self.bodies[idx].linear_velocity = vel;
        self.bodies[idx].wake_up();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Joint management
    // -----------------------------------------------------------------------

    /// Add a joint constraint between two bodies.
    pub fn add_joint(
        &mut self,
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        desc: &JointDesc,
    ) -> PhysicsResult<ConstraintHandle> {
        if !self.body_index_map.contains_key(&body_a) {
            return Err(PhysicsError::InvalidBodyHandle(body_a));
        }
        if !self.body_index_map.contains_key(&body_b) {
            return Err(PhysicsError::InvalidBodyHandle(body_b));
        }

        let handle = ConstraintHandle(self.next_joint_id);
        self.next_joint_id += 1;

        let joint: Box<dyn Constraint> = match desc {
            JointDesc::Fixed {
                anchor_a,
                anchor_b,
            } => Box::new(FixedJoint::new(body_a, body_b, *anchor_a, *anchor_b)),
            JointDesc::Ball {
                anchor_a,
                anchor_b,
                cone_limit,
            } => Box::new(BallJoint::new(
                body_a, body_b, *anchor_a, *anchor_b, *cone_limit,
            )),
            JointDesc::Hinge {
                anchor_a,
                anchor_b,
                axis,
                limits,
            } => Box::new(HingeJoint::new(
                body_a, body_b, *anchor_a, *anchor_b, *axis, *limits,
            )),
            JointDesc::Spring {
                anchor_a,
                anchor_b,
                rest_length,
                stiffness,
                damping,
            } => Box::new(SpringJoint::new(
                body_a,
                body_b,
                *anchor_a,
                *anchor_b,
                *rest_length,
                *stiffness,
                *damping,
            )),
            JointDesc::Slider {
                anchor_a,
                anchor_b,
                ..
            } => {
                // Slider not fully implemented -- fall back to fixed joint
                Box::new(FixedJoint::new(body_a, body_b, *anchor_a, *anchor_b))
            }
        };

        let idx = self.joints.len();
        self.joints.push(joint);
        self.joint_handle_map.insert(handle, idx);

        Ok(handle)
    }

    /// Remove a joint constraint.
    pub fn remove_joint(&mut self, handle: ConstraintHandle) -> PhysicsResult<()> {
        if let Some(idx) = self.joint_handle_map.remove(&handle) {
            if idx < self.joints.len() {
                self.joints.swap_remove(idx);
                // Update the map for the swapped element
                if idx < self.joints.len() {
                    // Find which handle maps to the old last index
                    let old_last = self.joints.len(); // This was the last index before swap_remove
                    for (_h, i) in self.joint_handle_map.iter_mut() {
                        if *i == old_last {
                            *i = idx;
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Raycasting
    // -----------------------------------------------------------------------

    /// Cast a ray and return all hits sorted by distance.
    pub fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
    ) -> Vec<RaycastHit> {
        self.raycast_filtered(origin, direction, max_distance, u32::MAX)
    }

    /// Cast a ray with layer mask filtering. Returns all hits sorted by distance.
    pub fn raycast_filtered(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        layer_mask: u32,
    ) -> Vec<RaycastHit> {
        let dir = direction.normalize_or_zero();
        if dir.length_squared() < 0.5 {
            return Vec::new();
        }

        let mut hits = Vec::new();

        for (_, collider) in &self.colliders {
            // Layer filtering
            if (collider.desc.collision_layer.bits() & layer_mask) == 0 {
                continue;
            }

            // Get body transform
            let body_idx = match self.body_index_map.get(&collider.body) {
                Some(&idx) => idx,
                None => continue,
            };
            let body = &self.bodies[body_idx];
            let shape_pos = body.position + body.rotation * collider.desc.offset;

            // Quick AABB check first
            let aabb = collider
                .desc
                .shape
                .compute_aabb(shape_pos, body.rotation);
            if aabb.ray_intersect(origin, dir, max_distance).is_none() {
                continue;
            }

            // Exact shape test
            if let Some((t, point, normal)) =
                ray_vs_shape(origin, dir, max_distance, &collider.desc.shape, shape_pos, body.rotation)
            {
                hits.push(RaycastHit {
                    body: collider.body,
                    collider: collider.handle,
                    point,
                    normal,
                    distance: t,
                });
            }
        }

        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }

    /// Cast a ray and return the closest hit, if any.
    pub fn raycast_closest(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        layer_mask: u32,
    ) -> Option<RaycastHit> {
        let hits = self.raycast_filtered(origin, direction, max_distance, layer_mask);
        hits.into_iter().next()
    }

    /// Test whether any collider overlaps the given shape at the specified position.
    pub fn overlap_test(
        &self,
        shape: &CollisionShape,
        position: Vec3,
        rotation: Quat,
        layer_mask: u32,
    ) -> Vec<ColliderHandle> {
        let test_aabb = shape.compute_aabb(position, rotation);
        let candidates = self.broad_phase.query_aabb(&test_aabb);

        let mut results = Vec::new();
        for ch in candidates {
            if let Some(collider) = self.colliders.get(&ch) {
                if (collider.desc.collision_layer.bits() & layer_mask) == 0 {
                    continue;
                }
                let body_idx = match self.body_index_map.get(&collider.body) {
                    Some(&idx) => idx,
                    None => continue,
                };
                let body = &self.bodies[body_idx];
                let shape_pos = body.position + body.rotation * collider.desc.offset;

                if NarrowPhase::test_pair(shape, position, rotation, &collider.desc.shape, shape_pos, body.rotation)
                    .is_some()
                {
                    results.push(ch);
                }
            }
        }
        results
    }

    // -----------------------------------------------------------------------
    // Simulation step
    // -----------------------------------------------------------------------

    /// Advance the simulation by `dt` seconds.
    ///
    /// Pipeline:
    /// 1. Update broad-phase AABBs
    /// 2. Broad phase: find candidate pairs via spatial hash
    /// 3. Narrow phase: generate contact manifolds
    /// 4. Integrate forces -> velocities (semi-implicit Euler)
    /// 5. Solve constraints (contacts + joints) with sequential impulses
    /// 6. Integrate velocities -> positions
    /// 7. Update sleep states
    /// 8. Generate collision events
    pub fn step(&mut self, dt: f32) -> PhysicsResult<()> {
        if dt <= 0.0 {
            return Ok(());
        }

        profiling::scope!("PhysicsWorld::step");

        // 1. Update broad-phase AABBs for all colliders
        self.update_all_aabbs();

        // 2. Broad phase
        let broad_pairs = self.broad_phase.query_pairs();

        // 3. Narrow phase -- generate contact manifolds
        let manifolds = self.narrow_phase(&broad_pairs);

        // 4. Integrate forces -> velocities (gravity, external forces)
        //    We do this BEFORE solving so the solver can correct velocities.
        self.apply_gravity_and_forces(dt);

        // 5. Solve contact constraints with sequential impulse solver
        crate::dynamics::solve_contacts(
            &manifolds,
            &mut self.bodies,
            &self.body_index_map,
            dt,
            &mut self.warm_start_cache,
        );

        // Solve joint constraints
        self.solve_joints(dt);

        // 6. Integrate velocities -> positions
        self.integrate_positions(dt);

        // 7. Update sleep states
        for body in &mut self.bodies {
            body.update_sleep();
        }

        // 8. Generate collision events
        self.generate_collision_events(&manifolds);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn update_all_aabbs(&mut self) {
        for (_, collider) in &self.colliders {
            if let Some(&body_idx) = self.body_index_map.get(&collider.body) {
                let body = &self.bodies[body_idx];
                if body.is_sleeping {
                    continue;
                }
                let shape_pos = body.position + body.rotation * collider.desc.offset;
                let aabb = collider.desc.shape.compute_aabb(shape_pos, body.rotation);
                // Expand AABB slightly for movement prediction
                let expanded = aabb.expanded(0.05);
                self.broad_phase.update(collider.handle, expanded);
            }
        }
    }

    fn update_collider_aabbs(&mut self, body_handle: RigidBodyHandle) {
        if let Some(collider_handles) = self.body_colliders.get(&body_handle) {
            let handles: Vec<ColliderHandle> = collider_handles.clone();
            for ch in handles {
                if let Some(collider) = self.colliders.get(&ch) {
                    if let Some(&body_idx) = self.body_index_map.get(&collider.body) {
                        let body = &self.bodies[body_idx];
                        let shape_pos = body.position + body.rotation * collider.desc.offset;
                        let aabb = collider.desc.shape.compute_aabb(shape_pos, body.rotation);
                        self.broad_phase.update(ch, aabb);
                    }
                }
            }
        }
    }

    fn narrow_phase(
        &self,
        broad_pairs: &[crate::collision::BroadPhasePair],
    ) -> Vec<ContactManifold> {
        let mut manifolds = Vec::new();

        for pair in broad_pairs {
            let col_a = match self.colliders.get(&pair.collider_a) {
                Some(c) => c,
                None => continue,
            };
            let col_b = match self.colliders.get(&pair.collider_b) {
                Some(c) => c,
                None => continue,
            };

            // Skip if same body
            if col_a.body == col_b.body {
                continue;
            }

            // Layer filtering
            if !layers_interact(
                col_a.desc.collision_layer,
                col_a.desc.collision_mask,
                col_b.desc.collision_layer,
                col_b.desc.collision_mask,
            ) {
                continue;
            }

            // Skip sensors for contact generation (they only generate events)
            if col_a.desc.is_sensor || col_b.desc.is_sensor {
                // Still record as collision event but don't generate contacts
                continue;
            }

            let body_a_idx = match self.body_index_map.get(&col_a.body) {
                Some(&idx) => idx,
                None => continue,
            };
            let body_b_idx = match self.body_index_map.get(&col_b.body) {
                Some(&idx) => idx,
                None => continue,
            };

            let body_a = &self.bodies[body_a_idx];
            let body_b = &self.bodies[body_b_idx];

            // Skip if both bodies are static or sleeping
            if body_a.is_static && body_b.is_static {
                continue;
            }
            if body_a.is_sleeping && body_b.is_sleeping {
                continue;
            }

            let pos_a = body_a.position + body_a.rotation * col_a.desc.offset;
            let pos_b = body_b.position + body_b.rotation * col_b.desc.offset;

            if let Some(contacts) = NarrowPhase::test_pair(
                &col_a.desc.shape,
                pos_a,
                body_a.rotation,
                &col_b.desc.shape,
                pos_b,
                body_b.rotation,
            ) {
                if !contacts.is_empty() {
                    // Wake both bodies on contact
                    // (We'll do this after the borrow of self is released)

                    // Combine materials: geometric mean for friction, max for restitution
                    let friction_a = col_a.desc.material.friction;
                    let friction_b = col_b.desc.material.friction;
                    let restitution_a = col_a.desc.material.restitution;
                    let restitution_b = col_b.desc.material.restitution;

                    let combined_friction = (friction_a * friction_b).sqrt();
                    let combined_restitution = restitution_a.max(restitution_b);

                    manifolds.push(ContactManifold {
                        collider_a: pair.collider_a,
                        collider_b: pair.collider_b,
                        body_a: col_a.body,
                        body_b: col_b.body,
                        contacts,
                        friction: combined_friction,
                        restitution: combined_restitution,
                    });
                }
            }
        }

        // Wake bodies that are in contact
        for manifold in &manifolds {
            if let Some(&idx_a) = self.body_index_map.get(&manifold.body_a) {
                if let Some(&idx_b) = self.body_index_map.get(&manifold.body_b) {
                    // We can't mutably borrow here since we borrowed self immutably for narrow_phase
                    // This will be handled after the method returns
                    let _ = (idx_a, idx_b);
                }
            }
        }

        manifolds
    }

    fn apply_gravity_and_forces(&mut self, dt: f32) {
        for body in &mut self.bodies {
            if body.is_static || body.is_kinematic || body.is_sleeping {
                body.clear_forces();
                continue;
            }

            // Apply gravity
            let gravity_force = self.gravity * body.mass * body.gravity_scale;
            body.accumulated_force += gravity_force;

            // Integrate forces -> velocity
            let linear_accel = body.accumulated_force * body.inv_mass;
            body.linear_velocity += linear_accel * dt;

            let world_inv_inertia = body.world_inv_inertia();
            let angular_accel = world_inv_inertia * body.accumulated_torque;
            body.angular_velocity += angular_accel * dt;

            // Apply damping
            body.linear_velocity *= (1.0 - body.linear_damping).max(0.0);
            body.angular_velocity *= (1.0 - body.angular_damping).max(0.0);

            body.clear_forces();
        }
    }

    fn integrate_positions(&mut self, dt: f32) {
        for body in &mut self.bodies {
            if body.is_static || body.is_sleeping {
                continue;
            }

            // For kinematic bodies, position is set externally, but we still update
            if body.is_kinematic {
                continue;
            }

            // Integrate velocity -> position
            body.position += body.linear_velocity * dt;

            // Integrate angular velocity -> orientation
            let omega = body.angular_velocity;
            if omega.length_squared() > 1e-12 {
                let omega_quat = Quat::from_xyzw(omega.x, omega.y, omega.z, 0.0);
                let dq = omega_quat * body.rotation * 0.5;
                body.rotation = Quat::from_xyzw(
                    body.rotation.x + dq.x * dt,
                    body.rotation.y + dq.y * dt,
                    body.rotation.z + dq.z * dt,
                    body.rotation.w + dq.w * dt,
                )
                .normalize();
            }
        }
    }

    fn solve_joints(&mut self, dt: f32) {
        // Pre-solve all joints
        for joint in &mut self.joints {
            let (ha, hb) = joint.bodies();
            if let (Some(&idx_a), Some(&idx_b)) = (
                self.body_index_map.get(&ha),
                self.body_index_map.get(&hb),
            ) {
                // Clone bodies for pre_solve (to avoid multiple mutable borrows)
                let body_a = self.bodies[idx_a].clone();
                let body_b = self.bodies[idx_b].clone();
                joint.pre_solve(&body_a, &body_b, dt);
            }
        }

        // Solve joints for multiple iterations
        for _ in 0..crate::dynamics::SOLVER_ITERATIONS {
            for joint in &mut self.joints {
                let (ha, hb) = joint.bodies();
                if let (Some(&idx_a), Some(&idx_b)) = (
                    self.body_index_map.get(&ha),
                    self.body_index_map.get(&hb),
                ) {
                    if idx_a == idx_b {
                        continue;
                    }
                    // Split borrow: use unsafe to get two mutable references to different indices
                    // This is safe because idx_a != idx_b.
                    let (body_a, body_b) = if idx_a < idx_b {
                        let (left, right) = self.bodies.split_at_mut(idx_b);
                        (&mut left[idx_a], &mut right[0])
                    } else {
                        let (left, right) = self.bodies.split_at_mut(idx_a);
                        (&mut right[0], &mut left[idx_b])
                    };
                    joint.solve(body_a, body_b);
                }
            }
        }

        // Remove broken joints
        self.joints.retain(|j| !j.is_broken());
    }

    fn generate_collision_events(&mut self, manifolds: &[ContactManifold]) {
        self.collision_events.clear();

        let mut current_pairs = std::collections::HashSet::new();

        for manifold in manifolds {
            let pair = if manifold.collider_a.0 < manifold.collider_b.0 {
                (manifold.collider_a, manifold.collider_b)
            } else {
                (manifold.collider_b, manifold.collider_a)
            };
            current_pairs.insert(pair);

            if self.previous_colliding_pairs.contains(&pair) {
                // Stay event
                self.collision_events.push(CollisionEvent::Stay {
                    collider_a: manifold.collider_a,
                    collider_b: manifold.collider_b,
                    body_a: manifold.body_a,
                    body_b: manifold.body_b,
                    contacts: manifold.contacts.clone(),
                });
            } else {
                // Begin event
                self.collision_events.push(CollisionEvent::Begin {
                    collider_a: manifold.collider_a,
                    collider_b: manifold.collider_b,
                    body_a: manifold.body_a,
                    body_b: manifold.body_b,
                });
            }

            // Wake bodies on collision
            if let Some(&idx_a) = self.body_index_map.get(&manifold.body_a) {
                self.bodies[idx_a].wake_up();
            }
            if let Some(&idx_b) = self.body_index_map.get(&manifold.body_b) {
                self.bodies[idx_b].wake_up();
            }
        }

        // End events for pairs that were colliding last frame but not this frame
        for &pair in &self.previous_colliding_pairs {
            if !current_pairs.contains(&pair) {
                // We need the body handles for these colliders
                let body_a = self
                    .colliders
                    .get(&pair.0)
                    .map(|c| c.body)
                    .unwrap_or(RigidBodyHandle(0));
                let body_b = self
                    .colliders
                    .get(&pair.1)
                    .map(|c| c.body)
                    .unwrap_or(RigidBodyHandle(0));

                self.collision_events.push(CollisionEvent::End {
                    collider_a: pair.0,
                    collider_b: pair.1,
                    body_a,
                    body_b,
                });
            }
        }

        self.previous_colliding_pairs = current_pairs;
    }

    /// Drain collision events from the last step.
    pub fn drain_collision_events(&mut self) -> Vec<CollisionEvent> {
        std::mem::take(&mut self.collision_events)
    }

    /// Get the number of rigid bodies in the world.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Get the number of colliders in the world.
    pub fn collider_count(&self) -> usize {
        self.colliders.len()
    }

    /// Get the number of active (non-sleeping) bodies.
    pub fn active_body_count(&self) -> usize {
        self.bodies.iter().filter(|b| !b.is_sleeping && !b.is_static).count()
    }

    /// Get all body handles.
    pub fn body_handles(&self) -> Vec<RigidBodyHandle> {
        self.bodies.iter().map(|b| b.handle).collect()
    }
}

// ---------------------------------------------------------------------------
// PhysicsBackend trait
// ---------------------------------------------------------------------------

/// Backend abstraction for pluggable physics engine implementations.
pub trait PhysicsBackend: Send + Sync {
    /// Human-readable name of the backend.
    fn name(&self) -> &str;

    /// Create a new physics world using this backend.
    fn create_world(&self, gravity: Vec3) -> PhysicsResult<PhysicsWorld>;

    /// Check whether this backend is available on the current platform.
    fn is_available(&self) -> bool;
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::CollisionShape;

    #[test]
    fn test_add_and_remove_body() {
        let mut world = PhysicsWorld::new(Vec3::new(0.0, -9.81, 0.0));

        let h = world.add_body(&RigidBodyDesc::default()).unwrap();
        assert_eq!(world.body_count(), 1);

        world.remove_body(h).unwrap();
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn test_add_and_remove_collider() {
        let mut world = PhysicsWorld::new(Vec3::new(0.0, -9.81, 0.0));

        let body = world.add_body(&RigidBodyDesc::default()).unwrap();
        let col = world
            .add_collider(
                body,
                &ColliderDesc {
                    shape: CollisionShape::Sphere { radius: 1.0 },
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(world.collider_count(), 1);
        world.remove_collider(col).unwrap();
        assert_eq!(world.collider_count(), 0);
    }

    #[test]
    fn test_gravity_step() {
        let mut world = PhysicsWorld::new(Vec3::new(0.0, -9.81, 0.0));

        let body = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Dynamic,
                mass: 1.0,
                position: Vec3::new(0.0, 10.0, 0.0),
                ..Default::default()
            })
            .unwrap();

        world
            .add_collider(
                body,
                &ColliderDesc {
                    shape: CollisionShape::Sphere { radius: 0.5 },
                    ..Default::default()
                },
            )
            .unwrap();

        // Step the simulation
        world.step(1.0 / 60.0).unwrap();

        let pos = world.get_position(body).unwrap();
        // Body should have moved downward
        assert!(pos.y < 10.0, "position y = {}", pos.y);
    }

    #[test]
    fn test_static_body_doesnt_move() {
        let mut world = PhysicsWorld::new(Vec3::new(0.0, -9.81, 0.0));

        let body = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Static,
                position: Vec3::new(0.0, 0.0, 0.0),
                ..Default::default()
            })
            .unwrap();

        world.step(1.0 / 60.0).unwrap();

        let pos = world.get_position(body).unwrap();
        assert_eq!(pos, Vec3::ZERO);
    }

    #[test]
    fn test_collision_detection_spheres() {
        let mut world = PhysicsWorld::new(Vec3::ZERO);

        let body_a = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Dynamic,
                mass: 1.0,
                position: Vec3::new(0.0, 0.0, 0.0),
                ..Default::default()
            })
            .unwrap();
        world
            .add_collider(
                body_a,
                &ColliderDesc {
                    shape: CollisionShape::Sphere { radius: 1.0 },
                    ..Default::default()
                },
            )
            .unwrap();

        let body_b = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Dynamic,
                mass: 1.0,
                position: Vec3::new(1.5, 0.0, 0.0),
                ..Default::default()
            })
            .unwrap();
        world
            .add_collider(
                body_b,
                &ColliderDesc {
                    shape: CollisionShape::Sphere { radius: 1.0 },
                    ..Default::default()
                },
            )
            .unwrap();

        // Step -- the spheres overlap and should be pushed apart
        for _ in 0..10 {
            world.step(1.0 / 60.0).unwrap();
        }

        let pos_a = world.get_position(body_a).unwrap();
        let pos_b = world.get_position(body_b).unwrap();
        let dist = (pos_b - pos_a).length();

        // After several steps of collision resolution, they should be further apart
        assert!(dist > 1.5, "distance = {}", dist);
    }

    #[test]
    fn test_raycast() {
        let mut world = PhysicsWorld::new(Vec3::ZERO);

        let body = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Static,
                position: Vec3::new(5.0, 0.0, 0.0),
                ..Default::default()
            })
            .unwrap();
        world
            .add_collider(
                body,
                &ColliderDesc {
                    shape: CollisionShape::Sphere { radius: 1.0 },
                    ..Default::default()
                },
            )
            .unwrap();

        let hits = world.raycast(Vec3::ZERO, Vec3::X, 100.0);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].distance - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_raycast_miss() {
        let mut world = PhysicsWorld::new(Vec3::ZERO);

        let body = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Static,
                position: Vec3::new(5.0, 5.0, 0.0),
                ..Default::default()
            })
            .unwrap();
        world
            .add_collider(
                body,
                &ColliderDesc {
                    shape: CollisionShape::Sphere { radius: 1.0 },
                    ..Default::default()
                },
            )
            .unwrap();

        let hits = world.raycast(Vec3::ZERO, Vec3::X, 100.0);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_set_gravity() {
        let mut world = PhysicsWorld::new(Vec3::new(0.0, -9.81, 0.0));
        world.set_gravity(Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(world.gravity(), Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn test_apply_force() {
        let mut world = PhysicsWorld::new(Vec3::ZERO);

        let body = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Dynamic,
                mass: 1.0,
                ..Default::default()
            })
            .unwrap();
        world
            .add_collider(
                body,
                &ColliderDesc {
                    shape: CollisionShape::Sphere { radius: 0.5 },
                    ..Default::default()
                },
            )
            .unwrap();

        world
            .apply_force(body, Vec3::new(10.0, 0.0, 0.0), ForceMode::Impulse)
            .unwrap();

        let vel = world.get_linear_velocity(body).unwrap();
        assert!((vel.x - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_invalid_handle() {
        let world = PhysicsWorld::new(Vec3::ZERO);
        let result = world.get_position(RigidBodyHandle(999));
        assert!(result.is_err());
    }

    #[test]
    fn test_collision_events() {
        let mut world = PhysicsWorld::new(Vec3::ZERO);

        let body_a = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Dynamic,
                mass: 1.0,
                position: Vec3::ZERO,
                ..Default::default()
            })
            .unwrap();
        world
            .add_collider(
                body_a,
                &ColliderDesc {
                    shape: CollisionShape::Sphere { radius: 1.0 },
                    ..Default::default()
                },
            )
            .unwrap();

        let body_b = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Dynamic,
                mass: 1.0,
                position: Vec3::new(1.5, 0.0, 0.0),
                ..Default::default()
            })
            .unwrap();
        world
            .add_collider(
                body_b,
                &ColliderDesc {
                    shape: CollisionShape::Sphere { radius: 1.0 },
                    ..Default::default()
                },
            )
            .unwrap();

        world.step(1.0 / 60.0).unwrap();

        let events = world.drain_collision_events();
        // Should have at least one Begin event
        assert!(
            events.iter().any(|e| matches!(e, CollisionEvent::Begin { .. })),
            "Expected Begin event, got {:?}",
            events
        );
    }

    #[test]
    fn test_ball_on_floor() {
        let mut world = PhysicsWorld::new(Vec3::new(0.0, -9.81, 0.0));

        // Floor (static)
        let floor = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Static,
                position: Vec3::new(0.0, -1.0, 0.0),
                ..Default::default()
            })
            .unwrap();
        world
            .add_collider(
                floor,
                &ColliderDesc {
                    shape: CollisionShape::Box {
                        half_extents: Vec3::new(10.0, 1.0, 10.0),
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        // Ball (dynamic)
        let ball = world
            .add_body(&RigidBodyDesc {
                body_type: BodyType::Dynamic,
                mass: 1.0,
                position: Vec3::new(0.0, 5.0, 0.0),
                restitution: 0.5,
                ..Default::default()
            })
            .unwrap();
        world
            .add_collider(
                ball,
                &ColliderDesc {
                    shape: CollisionShape::Sphere { radius: 0.5 },
                    material: PhysicsMaterial {
                        restitution: 0.5,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        // Simulate 2 seconds
        for _ in 0..120 {
            world.step(1.0 / 60.0).unwrap();
        }

        let pos = world.get_position(ball).unwrap();
        // Ball should have fallen and come to rest near y=0.5 (sphere radius above floor surface)
        // The floor surface is at y=0 (center at -1, half_extent.y = 1)
        assert!(pos.y > -1.0, "Ball fell through floor: y = {}", pos.y);
        assert!(pos.y < 5.0, "Ball didn't fall: y = {}", pos.y);
    }

    #[test]
    fn test_add_joint() {
        let mut world = PhysicsWorld::new(Vec3::ZERO);

        let body_a = world.add_body(&RigidBodyDesc::default()).unwrap();
        let body_b = world
            .add_body(&RigidBodyDesc {
                position: Vec3::new(2.0, 0.0, 0.0),
                ..Default::default()
            })
            .unwrap();

        let joint = world
            .add_joint(
                body_a,
                body_b,
                &JointDesc::Fixed {
                    anchor_a: Vec3::new(1.0, 0.0, 0.0),
                    anchor_b: Vec3::new(-1.0, 0.0, 0.0),
                },
            )
            .unwrap();

        // Step should not panic
        world.step(1.0 / 60.0).unwrap();
    }

    #[test]
    fn test_multiple_bodies_no_panic() {
        let mut world = PhysicsWorld::new(Vec3::new(0.0, -9.81, 0.0));

        for i in 0..20 {
            let body = world
                .add_body(&RigidBodyDesc {
                    body_type: BodyType::Dynamic,
                    mass: 1.0,
                    position: Vec3::new(i as f32 * 0.5, i as f32 * 2.0, 0.0),
                    ..Default::default()
                })
                .unwrap();
            world
                .add_collider(
                    body,
                    &ColliderDesc {
                        shape: CollisionShape::Sphere { radius: 0.5 },
                        ..Default::default()
                    },
                )
                .unwrap();
        }

        for _ in 0..60 {
            world.step(1.0 / 60.0).unwrap();
        }

        assert_eq!(world.body_count(), 20);
    }
}
