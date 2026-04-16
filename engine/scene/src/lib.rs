//! # genovo-scene
//!
//! Scene management module for the Genovo game engine.
//!
//! This crate provides:
//!
//! - **Scene nodes** -- an OOP-style scene graph with hierarchical transforms.
//! - **Transform components** -- ECS components for position, rotation, and
//!   scale with automatic parent-to-child propagation.
//! - **ECS bridge** -- bidirectional sync between the scene graph and the ECS
//!   world.
//! - **Prefabs** -- serializable entity templates with per-instance overrides.

pub mod ecs_bridge;
pub mod node;
pub mod prefab;
pub mod scene_serializer;
pub mod transform;

// Re-exports for ergonomic use.
pub use ecs_bridge::{NodeEvent, SceneGraph, SceneNodeComponent, SyncDirection};
pub use node::{
    AllNodesIter, AncestorIter, BreadthFirstIter, DepthFirstIter, DescendantIter, NodeId,
    SceneNode, SceneNodeTree,
};
pub use prefab::{
    ComponentOverride, Prefab, PrefabError, PrefabId, PrefabInstance, PrefabNodeDescriptor,
    PrefabRegistry,
};
pub use scene_serializer::{
    DiffChange, SceneDiff, SceneSerializer, SerializedComponent, SerializedEntity,
    SerializedScene, SerializedValue,
};
pub use transform::{GlobalTransform, TransformComponent, TransformHierarchy, TransformSystem};
