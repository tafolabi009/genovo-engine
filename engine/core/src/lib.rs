//! # Genovo Core
//!
//! The foundational crate for the Genovo game engine. Provides core primitives
//! used across all engine subsystems: math types, memory allocators, a job system
//! for multi-threaded workloads, runtime reflection, generational handles, error
//! types, and time management utilities.

pub mod collections;
pub mod color;
pub mod compression;
pub mod crypto;
pub mod curve;
pub mod error;
pub mod event_bus;
pub mod geometry;
pub mod handle;
pub mod intersect;
pub mod math;
pub mod memory;
pub mod noise;
pub mod plugin;
pub mod random;
pub mod reflection;
pub mod serialization;
pub mod type_registry;
pub mod simd;
pub mod spatial;
pub mod threading;
pub mod time;

// Re-export the most commonly used items at the crate root for ergonomic access.
pub use color::{Color, ColorStop, Gradient};
pub use curve::{AnimationCurve, BezierCurve, BSpline, CatmullRomSpline, CurveKeyframe, CurveMode};
pub use error::{EngineError, EngineResult};
pub use handle::{Handle, HandlePool};
pub use math::{Transform, AABB, Frustum, Plane, Ray, Rect};
pub use geometry::{
    ConvexHull2D, ConvexHull3D, DelaunayTriangulation, EarClipTriangulation,
    OBB, Circle, Sphere as BoundingSphere,
};
pub use intersect::{Capsule, RayHit};
pub use spatial::{BVH, BVHItem, KDTree, Octree, RTree, SpatialHashGrid};
pub use noise::{CurlNoise, PerlinNoise, SimplexNoise, ValueNoise, WorleyNoise};
pub use random::{Halton, PoissonDisk, Rng};
pub use plugin::{
    ClosurePlugin, Plugin, PluginContext, PluginError, PluginManager, PluginState, PluginVersion,
};
pub use time::Clock;
pub use collections::{
    BitSet, FreeList, FreeListHandle, InternedString, ObjectPool, PoolHandle,
    PriorityQueue, RingBuffer, SmallVec, SparseArray, StringInterner,
};
pub use compression::{
    DeltaEncoder, HuffmanCoder, HuffmanTable, LZ4Compressor, RunLengthEncoder,
};
pub use crypto::{Aes128, Crc32, Crc32Hasher, Hmac, Sha256, Sha256Hasher};
pub use event_bus::{
    AppLifecycleEvent, CollisionEvent, EntityDespawned, EntitySpawned, Event, EventBus,
    EventBusStats, EventId, KeyModifiers, KeyPressed, KeyReleased, PhysicsStep, SceneLoaded,
    SceneUnloading, SubscriberId, WindowResized,
};

/// Unique identifier for engine objects.
pub type ObjectId = uuid::Uuid;
