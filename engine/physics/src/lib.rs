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
//! - **Spatial queries** (`spatial_query`): BVH-accelerated raycast, sphere cast,
//!   box cast, overlap, and point queries with configurable filters
//! - **Trigger volumes** (`trigger_volumes`): sensor shapes with enter/stay/exit
//!   event tracking, callbacks, one-shot triggers, cooldowns, and layer filtering
//!
//! ## Gravity, Joints & Layers
//!
//! - **Gravity fields** (`gravity_field`): custom gravity zones (spherical planets,
//!   directional overrides), gravity wells, anti-gravity volumes, smooth zone
//!   transitions, radial gravity for spherical worlds
//! - **Breakable joints** (`breakable_joints`): joints that break under force/torque,
//!   partial degradation, chain-breaking propagation, fatigue stress accumulation
//! - **Physics layers** (`physics_layers`): 32-layer collision filtering, collision
//!   matrix, per-body layer assignment, raycast/trigger layer filtering, presets
//! - **PBD Fluid v2** (`particles_v2`): position-based fluid dynamics with XSPH
//!   viscosity, incompressibility constraint, vorticity confinement, surface
//!   reconstruction, boundary particle handling
//!
//! ## Extended Physics Subsystems
//!
//! - **Particle physics** (`particle_physics`): PBD particle simulation with
//!   mass-spring networks, distance/bending/volume constraints, strain limiting,
//!   self-collision via spatial hashing, emitters, and groups
//! - **Magnetic fields** (`magnetic_field`): electromagnetic dipole fields, Lorentz
//!   force on charged particles, field superposition, field line visualization
//! - **Aerodynamics** (`aerodynamics`): lift/drag computation, NACA airfoil profiles,
//!   stall modeling, parachute drag, glider physics, paper airplane tumble
//! - **Wind system** (`wind_system`): directional wind, periodic/random gusts,
//!   turbulence zones, Beaufort scale presets, spatial wind field sampling
//! - **Advanced fracture** (`fracture_v2`): runtime mesh fracture, stress propagation,
//!   crack initiation/propagation, fragment mass/inertia, multi-material support
//! - **Physics debug** (`physics_debug`): collision shape wireframes, contact points/
//!   normals, joint axes/limits, velocity arrows, broadphase grid, constraint errors
//! - **Physics materials** (`physics_materials`): named material database (ice, rubber,
//!   wood, metal, glass, concrete, sand), combination rules, contact sound hints
//! - **Physics profiler** (`physics_profiler`): per-phase timing, pair/island/sleeping
//!   counts, solver convergence, memory usage, worst-frame tracking

pub mod aerodynamics;
pub mod backends;
pub mod breakable_joints;
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
pub mod fracture_v2;
pub mod gravity_field;
pub mod interface;
pub mod magnetic_field;
pub mod motor_joint;
pub mod particle_physics;
pub mod particles_v2;
pub mod physics_debug;
pub mod physics_materials;
pub mod physics_layers;
pub mod physics_profiler;
pub mod ragdoll;
pub mod rope;
pub mod softbody;
pub mod spatial_query;
pub mod trigger_volumes;
pub mod vehicle;
pub mod wind_system;

// Collision event system: begin/stay/end events, trigger enter/exit, event filtering,
// callbacks, contact info (points, normals, impulses), event history buffer.
pub mod collision_events;

// Enhanced physics world: sub-worlds, physics islands, sleeping island optimization,
// broad-phase switching (SAP/grid/BVH), narrow-phase cache, constraint groups.
pub mod physics_world_v2;

// Shape cast queries: convex shape sweep, box sweep, capsule sweep, sphere sweep,
// layer filter, contact point generation, time of impact.
pub mod shape_casting;

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
pub use spatial_query::{
    Aabb, BodyHandle, Bvh, BvhStats, ColliderProxy, MaterialId, OverlapHit, PhysicsQuery,
    PointHit, QueryFilter, QueryShape, RayHit, ShapecastBuffer,
    SweepHit as SpatialSweepHit,
};
pub use trigger_volumes::{
    TriggerComponent, TriggerEntity, TriggerEvent, TriggerEventRecord, TriggerShape,
    TriggerSystem, TriggerVolume,
};

// Re-exports for extended physics subsystems.
pub use particle_physics::{
    BendingConstraint as PbdBendingConstraint, ConstraintHandle as PbdConstraintHandle,
    DistanceConstraint as PbdDistanceConstraint, MassSpringNetworkBuilder, Particle,
    ParticleEmitter, ParticleEmitterConfig, ParticleGroup, ParticleId,
    ParticlePhysicsComponent, ParticlePhysicsSystem, ParticleSimulation,
    ParticleSimulationSettings, ParticleSimulationStats, SpatialHash,
    StrainLimiter, VolumeConstraint as PbdVolumeConstraint,
};
pub use magnetic_field::{
    ChargedBodyComponent, ChargedParticle, FieldFalloff, FieldLine, FieldLineData,
    MagneticDipole, MagneticFieldComponent, MagneticFieldSystem, MagneticSourceId,
    UniformField as MagneticUniformField,
};
pub use aerodynamics::{
    AerodynamicBody, AerodynamicsComponent, AerodynamicsSystem, AeroCoefficients,
    AeroModel, GliderModel, NacaAirfoil, PaperAirplane, PaperFlightMode,
    Parachute, ParachuteState, SurfaceDrag,
};
pub use wind_system::{
    BeaufortScale, GustPattern, TurbulenceZone, WindField, WindReceiverComponent,
    WindSample, WindSettings, WindSource, WindSourceComponent, WindSourceId,
};
pub use fracture_v2::{
    CrackNetwork, CrackPattern, CrackSegment, FracturableComponent, FractureConfig,
    FractureEvent, FractureManager, FractureMaterial, FractureMeshV2, FractureSound,
    FractureSystem, Fragment, StressField,
};
pub use physics_debug::{
    DebugBodyInfo, DebugBroadphaseGrid, DebugCollisionShape, DebugConstraintError,
    DebugContactPoint, DebugJointInfo, DebugPrimitive, DebugRenderStats,
    PhysicsDebugRenderer, PhysicsDebugSettings,
};
pub use physics_materials::{
    CombinedMaterial, CombineRule, ContactSoundHint, ImpactEffect, MaterialDatabase,
    MaterialPairOverride, PhysMaterial, PhysMaterialId, SurfaceType,
};
pub use physics_profiler::{
    CollisionPairCounts, FrameRecord, IslandInfo, MemoryUsage, PhaseTimings,
    PhysicsObjectCounts, PhysicsProfiler, SolverStats, WorstFrameRecord,
};
pub use gravity_field::{
    GravityBlendMode, GravityFalloff, GravityFieldComponent, GravityFieldManager,
    GravityFieldSystem, GravityMode, GravityPulsation, GravityQueryResult,
    GravitySourceComponent, GravityTransition, GravityWell, GravityWellId,
    GravityZone, GravityZoneId, GravityZoneShape, TransitionEasing,
};
pub use breakable_joints::{
    BreakCondition, BreakEffectHint, BreakEvent, BreakMode, BreakableJoint,
    BreakableJointComponent, BreakableJointId, BreakableJointSystem,
    ChainBreakConfig, DegradationState, DegradeEvent,
};
pub use physics_layers::{
    BuiltinLayer, CollisionMatrix, LayerFilter, LayerGroup, LayerGroupManager,
    PhysicsLayerComponent, PhysicsLayerSystem, TriggerLayerFilter,
};
pub use particles_v2::{
    BoundaryParticle, FluidParticle, FluidParticleComponent as FluidParticleComponentV2,
    FluidSettingsV2, FluidSimulationV2, FluidSimStats,
    SpatialHashGrid as FluidSpatialHash, SurfaceInfo,
};
