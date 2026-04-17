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
pub mod hierarchy_system;
pub mod light_system;
pub mod node;
pub mod prefab;
pub mod scene_manager;
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

// Scene manager re-exports.
pub use scene_manager::{
    EntityId, LoadProgress, LoadSceneMode, SceneDefinition, SceneDependency, SceneEvent,
    SceneHandle, SceneId, SceneInstance, SceneLoadRequest, SceneManager, SceneStackEntry,
    SceneState, TransitionEffect, TransitionPhase,
};

// Light system re-exports.
pub use light_system::{
    AmbientMode, CameraFrustum, CulledLight, LightComponent, LightEnvironment, LightIntensity,
    LightLayerMask, LightSystem, LightType, LinearColor, ShadowConfig, ShadowFilter,
    ShadowResolution, color_temperature_to_rgb,
};

// Visibility determination: frustum culling integration, occlusion query integration,
// portal-based indoor visibility, PVS query, distance culling, small-object culling,
// shadow caster visibility, per-camera visibility lists.
pub mod visibility_system;

// Scene queries: find entities by name/tag/component, spatial queries (entities in
// radius/box), raycast through scene, nearest entity, entity iteration with filters.
pub mod scene_queries;

// Enhanced scene graph: transformation cache, dirty bit propagation, spatial
// index integration, scene queries, scene comparison, scene merging.
pub mod scene_graph;

// Component serialization: serialize/deserialize components by name, component
// factories, component cloning, component diffing.
pub mod component_system;

// Level streaming: async level load, level visibility, level transform offset,
// level LOD, streaming volume triggers.
pub mod level_streaming;

// Scene templates/prefabs: template definition, template instantiation,
// override tracking, template inheritance, template variables.
pub mod scene_templates;

// Hierarchy system re-exports.
    ancestor_path, collect_descendants, despawn_recursive, detach_children, hierarchy_depth,
    is_ancestor_of, lowest_common_ancestor, root_ancestor, set_parent, remove_parent, siblings,
    subtree_count, Children, DirtyTransform, HierarchyPlugin, Parent, PropagateTransforms,
};
