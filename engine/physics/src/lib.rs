//! # Genovo Physics
//!
//! Physics simulation module for the Genovo game engine.
//! Provides rigid body dynamics, collision detection, constraint solving,
//! and a complete physics pipeline including:
//!
//! - **Spatial hash broad phase** for efficient pair detection
//! - **Analytical narrow phase** with sphere/box/capsule intersection tests
//! - **SAT (Separating Axis Theorem)** for OBB-OBB collisions
//! - **Sequential impulse constraint solver** with warm starting
//! - **Semi-implicit Euler integration** with damping
//! - **Joint constraints**: Fixed, Hinge, Ball, Spring
//! - **Body sleeping** for performance
//! - **Collision layers and masks** for filtering
//! - **ECS integration** via components and systems
//!
//! ## Advanced Physics Systems
//!
//! - **Cloth simulation** (`cloth`): mass-spring system with Verlet integration,
//!   Jakobsen constraint relaxation, self-collision, wind, and tearing
//! - **Soft body physics** (`softbody`): FEM with co-rotational linear elasticity,
//!   shape matching (Mueller et al.), volume preservation
//! - **Fluid simulation** (`fluid`): SPH with Poly6/Spiky/Viscosity kernels,
//!   surface tension, marching cubes mesh extraction
//! - **Vehicle physics** (`vehicle`): Pacejka tire model, spring-damper suspension,
//!   drivetrain with engine torque curve and gears, Ackermann steering, ABS/TCS
//! - **Ragdoll system** (`ragdoll`): skeleton-to-physics mapping with joint limits,
//!   humanoid presets, partial ragdoll blending
//! - **Destruction system** (`destruction`): Voronoi fracture, connectivity graphs,
//!   cascading destruction, debris cleanup
//! - **Rope/chain physics** (`rope`): Verlet particle ropes with Jakobsen constraints,
//!   wind, self-collision, Catmull-Rom rendering, rigid-link chains
//! - **Buoyancy** (`buoyancy`): Archimedes-principle buoyancy, partial submersion,
//!   water volumes with current and waves, splash detection
//! - **Extended constraints** (`constraints_extended`): SliderJoint, ConeTwistJoint,
//!   GearJoint, PulleyJoint, WeldJoint, DistanceJoint, MouseJoint
//! - **Continuous collision detection** (`continuous_collision`): sphere/AABB/plane/triangle
//!   sweep tests, conservative advancement, CCD settings
//! - **Character physics** (`character_physics`): capsule-based character controller
//!   with sweep tests, iterative collide-and-slide, step detection, ground queries,
//!   slope handling, and moving platform tracking

pub mod backends;
pub mod buoyancy;
pub mod character_physics;
pub mod cloth;
pub mod collision;
pub mod components;
pub mod constraints_extended;
pub mod continuous_collision;
pub mod destruction;
pub mod dynamics;
pub mod fluid;
pub mod interface;
pub mod motor_joint;
pub mod ragdoll;
pub mod rope;
pub mod softbody;
pub mod vehicle;

// Re-exports for ergonomic top-level access.
pub use collision::{
    ColliderDesc, CollisionEvent, CollisionLayer, CollisionMask, CollisionShape, ContactManifold,
    ContactPoint,
};
pub use components::{ColliderComponent, PhysicsSystem, RigidBodyComponent};
pub use dynamics::{Constraint, ConstraintHandle, ForceMode, JointDesc, RigidBody};
pub use interface::{
    BodyType, ColliderHandle, PhysicsBackend, PhysicsError, PhysicsMaterial, PhysicsResult,
    PhysicsWorld, RaycastHit, RigidBodyDesc, RigidBodyHandle,
};

// Re-exports for advanced physics systems.
pub use buoyancy::{BuoyancyBody, BuoyancyComponent, BuoyancyShape, BuoyancySystem, WaterVolume};
pub use cloth::{ClothComponent, ClothMesh, ClothSettings, ClothSystem};
pub use constraints_extended::{
    ConeTwistJoint, DistanceJoint, GearJoint, MouseJoint, PulleyJoint, SliderJoint, WeldJoint,
};
pub use continuous_collision::{CCDSettings, SweepResult};
pub use destruction::{DestructibleComponent, DestructibleMesh, DestructionManager, DestructionSystem};
pub use fluid::{FluidComponent, FluidSettings, FluidSystem, SPHFluid};
pub use ragdoll::{RagdollComponent, RagdollDefinition, RagdollInstance, RagdollSystem};
pub use rope::{ChainSimulation, RopeComponent, RopeSimulation, RopeSystem};
pub use softbody::{SoftBody, SoftBodyComponent, SoftBodySettings, SoftBodySystem};
pub use vehicle::{VehicleComponent, VehicleConfig, VehicleController, VehicleSystem};
pub use motor_joint::{
    BodyState, GenericJoint, JointAxis, JointAxisConfig, MotorJoint, MotorMode,
    PdController, PrismaticJoint, RopeJoint, SuspensionSettings, WheelJoint,
    WheelMotorSettings,
};
pub use character_physics::{
    CharacterCapsule, CharacterCollision, CharacterControllerManager, CharacterHandle,
    CharacterMoveResult, GroundInfo, GroundQuery, MovingPlatformTracker, PhysicsCharacter,
    PlatformHandle, StaticCollider, SurfaceMaterial, SweepHit,
};
