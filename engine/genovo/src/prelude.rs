//! Convenience re-exports for common engine types.
//!
//! ```rust
//! use genovo::prelude::*;
//! ```

// Engine facade
pub use crate::config::{EngineConfig, LogLevel, WindowConfig};
pub use crate::engine::{Engine, EngineError};

// Core math
pub use genovo_core::math::{Mat4, Quat, Transform, Vec2, Vec3, Vec4};
pub use genovo_core::Clock;

// ECS
pub use genovo_ecs::{Component, Entity, World};

// Rendering
pub use genovo_render::RenderBackend;

// Physics
pub use genovo_physics::{
    BodyType, ColliderDesc, CollisionShape, PhysicsMaterial, PhysicsWorld, RigidBodyDesc,
    RigidBodyHandle,
};

// Audio
pub use genovo_audio::{AudioBus, AudioClip, AudioMixer, AudioSource, SoftwareMixer};

// Scene / transforms
pub use genovo_scene::{GlobalTransform, SceneGraph, TransformComponent};

// Assets
pub use genovo_assets::{AssetHandle, AssetServer};

// Debug / Profiling
pub use genovo_debug::{Console, DebugRenderer, MemoryProfiler, Profiler};

// Terrain
pub use genovo_terrain::{Heightmap, TerrainComponent, TerrainMesh, TerrainSystem};

// Cinematics
pub use genovo_cinematics::{CinematicCamera, Sequence, SequencePlayer, Timeline};

// Localization
pub use genovo_localization::{LocaleId, LocaleManager, StringTable};

// Procedural Generation
pub use genovo_procgen::{DungeonMap, LSystem, Maze, NameGenerator, WFCSolver};

// UI
pub use genovo_ui::{DrawList, Style, UIContext, UITree};

// Save System
pub use genovo_save::{SaveGame, SaveManager, WorldState};

// Editor
pub use genovo_editor::{EditorViewport, Inspector, NodeGraph, Project};
