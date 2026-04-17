//! # Genovo Core
//!
//! The foundational crate for the Genovo game engine. Provides core primitives
//! used across all engine subsystems: math types, memory allocators, a job system
//! for multi-threaded workloads, runtime reflection, generational handles, error
//! types, and time management utilities.

pub mod collections;
pub mod color;
pub mod compression;

// Centralized input: keyboard/mouse/gamepad polling, action mapping,
// axis mapping with dead zones, input consumed flag, input recording.
pub mod input_system;

// Engine runtime: subsystem lifecycle, fixed timestep + variable render,
// spiral-of-death protection, shutdown sequence, frame timing.
pub mod engine_runtime;
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
pub mod pool_allocator;
pub mod string_utils;
pub mod task_scheduler;
pub mod threading;
pub mod time;
pub mod async_runtime;
pub mod config_system;
pub mod math_extended;
pub mod object_model;
pub mod signal_slot;
pub mod state_machine;

// Thread pool: configurable worker count, task queue with priority, work stealing,
// thread affinity, thread naming, idle callbacks, graceful shutdown, pool statistics.
pub mod thread_pool;

// Arena allocator: bump allocation, frame arena (reset each frame), typed arena,
// arena scope with RAII, alignment handling, memory tracking.
pub mod memory_arena;

// Enhanced profiling: GPU timestamp queries, nested scope tree, flame graph data
// export, per-system averages over N frames, automatic hotspot alerts, memory
// allocation tracking per scope.
pub mod profiling;

// Additional data structures: skip list, B-tree, trie (prefix tree), bloom filter,
// count-min sketch, disjoint set (union-find with path compression), LRU cache,
// concurrent queue.
pub mod data_structures;

// Job graph system: jobs with typed inputs/outputs, automatic dependency from
// data flow, fan-out/fan-in, job stealing, job priorities, visualization.
pub mod job_graph;

// Binary serialization: compact binary format, schema versioning, forward/backward
// compatibility, field tags, nested structs, arrays, maps, CRC32 checksum.
pub mod binary_serializer;

// Debug drawing API: draw_line, draw_sphere, draw_box, draw_text_3d, draw_arrow,
// draw_frustum, persistent vs single-frame, depth-tested vs overlay.
pub mod debug_draw;

// Engine versioning: semantic version struct, version comparison, build info
// (commit hash, build date, platform), feature flags, version compatibility checking.
pub mod version;

// Re-exports for new profiling types.
pub use profiling::{
    AlertSeverity as ProfileAlertSeverity, FlameGraphEntry, FrameProfile,
    HotspotAlert, ProfileScope, ProfilerV2, ScopeAverage, ScopeId,
};

// Re-exports for data structures.
pub use data_structures::{
    BloomFilter, ConcurrentQueue, CountMinSketch, DisjointSet, LruCache, Trie,
};

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
pub use pool_allocator::{
    AllocError, AllocHandle, AllocResult, AllocationCategory, AllocationRecord,
    AllocatorStats, CategoryBudget, DefragMove, DefragPlan, LeakDetector,
    PoolAllocatorV2, ScopedAllocator, SourceLocation,
};
pub use string_utils::{
    StringBuilder, StringPool, StringId, NamedArgs,
    city_hash_64, city_hash_str, fnv1a_32, fnv1a_64, fnv1a_str, fnv1a_str_64,
    format_named, fuzzy_find, levenshtein_distance, levenshtein_similarity,
    murmur3_32, murmur3_str, path_extension, path_filename, path_join,
    path_normalize, path_parent, path_stem, wildcard_match,
};
pub use task_scheduler::{
    GroupStats, SchedulerConfig, SchedulerStats, TaskDescriptor, TaskError,
    TaskFuture, TaskId, TaskOutcome, TaskPriority, TaskProfileEntry,
    TaskProfiler, TaskResult, TaskScheduler, TaskState, WaitGroup, WorkerStats,
};
pub use async_runtime::{
    BoxFuture, ChannelClosedError, Executor, ExecutorConfig, ExecutorStats,
    LocalBoxFuture, MpscReceiver, MpscSender, OneshotReceiver, OneshotSender,
    RuntimeWaker, TaskHandle, TimerFuture,
};
pub use config_system::{
    CommandLineArgs, ConfigChange, ConfigError, ConfigParser, ConfigResult,
    ConfigSchema, ConfigSystem, ConfigValue, ConfigWatcher, EnvResolver,
    ValidationRule,
};
pub use math_extended::{
    Complex, Cylindrical, Dual, Fixed16, Fixed24, Half, Spherical,
    differentiate, fast_cos, fast_inv_sqrt, fast_sin, fast_sqrt,
    octahedron_decode, octahedron_encode, octahedron_pack_u32, octahedron_unpack_u32,
};
pub use object_model::{
    EngineObject, ObjectData, ObjectFactory, ObjectId as EngineObjectId, ObjectRef,
    ObjectStore, PropertyChangeEvent, PropertyDescriptor, PropertyValue,
    TypeInfo, TypeRegistry, WeakObjectRef,
};
pub use signal_slot::{
    ConnectionGuard, ConnectionId, ConnectionType, SharedSignal, Signal, SignalMap,
};
pub use state_machine::{
    ClosureState, ContextValue, HistoryEntry, ParallelStateMachine, State,
    StateContext, StateData, StateId, StateMachine, TimedState, Transition,
};

// 64-bit large world coordinates: floating origin, world cell grid, streaming
// volumes, world bounds, multi-system LOD, sector-based simulation rates.
pub mod large_world;

pub use large_world::{
    AudioLODQuality, CellCoord3D, CellLoadState, LODEntity, LODLevel, LODSystem,
    LODSystemStats, LODThresholds, LargeWorldSystem, LargeWorldUpdateResult,
    RebaseEvent, Sector, SectorManager, SectorManagerConfig, SectorManagerStats,
    SectorTier, StreamingPriority, StreamingVolume, StreamingVolumeEvent,
    StreamingVolumeManager, StreamingVolumeShape, WorldBounds, WorldCell,
    WorldCellConfig, WorldCellGrid, WorldCellGridStats, WorldOrigin,
    WorldPosition,
};

// Structured logging: log levels (Trace/Debug/Info/Warn/Error), log targets
// (console/file/callback), log formatting with timestamp/module/level, log
// filtering per module, ring buffer for recent logs, log file rotation,
// colored console output.
pub mod logging;

// Resource lifecycle management: reference-counted resources, dependency
// tracking, unload when refcount=0, resource reload, resource events
// (loaded/unloaded/error), resource statistics.
// Asset references: typed handles, weak refs, load-on-demand,
// dependency tracking, hot-reload notification.
pub mod asset_ref;

// Main engine loop: fixed timestep, variable render, frame pacing,
// vsync, smoothing, spiral-of-death protection.
pub mod engine_loop;

pub mod resource_manager;

// Platform detection: OS name/version, CPU model/cores/features, GPU info
// (from adapter), RAM total/available, display info (resolution/DPI/refresh
// rate), storage info, build configuration (debug/release).
pub mod platform_info;

// Per-frame allocator: bump allocator reset each frame, typed allocation,
// arena with destructor, frame-scoped collections (Vec, scratch buffer),
// multi-frame ring buffer allocator.
pub mod frame_allocator;

// Engine-optimized HashMap: open addressing, Robin Hood hashing, power-of-2
// sizing, FxHash (Fibonacci hashing), backward-shift deletion, FastHashSet.
pub mod hash_map_fast;

// Engine console commands: cvar system (typed console variables), command
// registration with help text, argument parsing, autocomplete, cvar
// persistence to config file, cvar change callbacks, built-in engine cvars.
pub mod console_commands;

// Engine configuration: rendering settings, physics settings, audio settings,
// network settings, quality presets, per-platform defaults.
pub mod engine_config;

// Performance tracking: CPU/GPU frame time, per-system timing, memory usage,
// allocation rate, GC pressure, performance budget system.
pub mod performance_counters;

// Command buffer pattern: deferred commands, command recording, command replay,
// command serialization for networking, command undo.
pub mod command_buffer;

/// Unique identifier for engine objects.
pub type ObjectId = uuid::Uuid;

// Re-exports for input system.
pub use input_system::{
    AxisMapping, GamepadAxis, GamepadButton, InputSystem, KeyBinding, KeyCode,
    MouseButton, RecordedEvent, RecordedInput,
};

// Re-exports for engine runtime.
pub use engine_runtime::{
    EngineRuntime, FrameTiming, RuntimeConfig, RuntimeStats, SubsystemId,
    SubsystemRecord, SubsystemState,
};
