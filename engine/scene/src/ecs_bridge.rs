//! ECS / scene graph bridge.
//!
//! This module provides bidirectional synchronization between the OOP-style
//! [`SceneNodeTree`](crate::node::SceneNodeTree) and the ECS
//! [`World`](genovo_ecs::World). Scene graph mutations automatically propagate
//! to ECS entities and vice versa.
//!
//! # Architecture
//!
//! The [`SceneGraph`] struct owns a [`SceneNodeTree`] and maintains bidirectional
//! maps between [`NodeId`]s and [`Entity`]s. When a node is created through
//! `SceneGraph`, the corresponding entity is automatically spawned with
//! [`TransformComponent`] and [`GlobalTransform`]. When a node is destroyed,
//! the entity is despawned.
//!
//! Synchronization between the two representations happens during `sync()`:
//! - **SceneToEcs**: Local transforms from scene nodes are pushed into ECS
//!   TransformComponents.
//! - **EcsToScene**: ECS TransformComponents are pulled into scene node local
//!   transforms.
//! - **Bidirectional**: Dirty flags determine which direction to sync per-node.
//!
//! After syncing, [`SceneNodeTree::propagate_transforms`] is called so that
//! world transforms are up-to-date for rendering.

use std::collections::HashMap;

use genovo_ecs::{Entity, World};
use glam::{Mat4, Vec3};

use crate::node::{NodeId, SceneNodeTree};
use crate::transform::{GlobalTransform, TransformComponent, TransformHierarchy};

// ---------------------------------------------------------------------------
// SceneNodeComponent trait
// ---------------------------------------------------------------------------

/// Marker trait for ECS components that should be visible in the scene graph
/// inspector and synchronized with scene node properties.
///
/// Implementors can provide hooks that run when the scene graph or ECS side
/// changes.
pub trait SceneNodeComponent: genovo_ecs::Component {
    /// Human-readable name shown in the scene graph inspector.
    fn display_name() -> &'static str;

    /// Called when the scene graph node's transform changes.
    /// Allows the component to react (e.g., update physics body).
    fn on_transform_changed(&mut self) {
        // Default: no-op.
    }

    /// Called when the component is first attached to a scene node.
    fn on_attached(&mut self, _node_id: NodeId) {
        // Default: no-op.
    }

    /// Called when the component is detached from a scene node.
    fn on_detached(&mut self) {
        // Default: no-op.
    }

    /// Called when the node's visibility changes.
    fn on_visibility_changed(&mut self, _visible: bool) {
        // Default: no-op.
    }

    /// Called once per frame during sync, allowing the component to update
    /// itself based on the current node state.
    fn on_sync(&mut self, _node_id: NodeId) {
        // Default: no-op.
    }
}

// ---------------------------------------------------------------------------
// SyncDirection
// ---------------------------------------------------------------------------

/// Describes which direction a sync operation should flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    /// Scene graph is authoritative; push changes to ECS.
    SceneToEcs,
    /// ECS is authoritative; push changes to scene graph.
    EcsToScene,
    /// Both directions; dirty flags determine flow per-entity.
    Bidirectional,
}

// ---------------------------------------------------------------------------
// NodeEvent
// ---------------------------------------------------------------------------

/// Events emitted by the scene graph when nodes are created/destroyed/reparented.
#[derive(Debug, Clone)]
pub enum NodeEvent {
    /// A node was created.
    Created {
        node_id: NodeId,
        entity: Entity,
        name: String,
    },
    /// A node was destroyed.
    Destroyed { node_id: NodeId, entity: Entity },
    /// A node was reparented.
    Reparented {
        node_id: NodeId,
        old_parent: Option<NodeId>,
        new_parent: Option<NodeId>,
    },
    /// A node's transform was modified.
    TransformChanged { node_id: NodeId },
    /// A node's visibility changed.
    VisibilityChanged { node_id: NodeId, visible: bool },
}

// ---------------------------------------------------------------------------
// SceneGraph
// ---------------------------------------------------------------------------

/// The bridge struct that owns both an OOP scene node tree and mappings
/// between scene nodes and ECS entities.
///
/// This is the primary API for managing scene content. It ensures that every
/// scene node has a corresponding ECS entity with transform components, and
/// every ECS entity spawned through this API has a corresponding scene node.
pub struct SceneGraph {
    /// The OOP scene node tree.
    tree: SceneNodeTree,

    /// Mapping: scene node id -> ECS entity.
    node_to_entity: HashMap<NodeId, Entity>,
    /// Mapping: ECS entity -> scene node id.
    entity_to_node: HashMap<Entity, NodeId>,

    /// Sync direction policy.
    sync_direction: SyncDirection,

    /// Nodes that have been modified since the last sync.
    dirty_nodes: Vec<NodeId>,
    /// Entities that have been modified since the last sync.
    dirty_entities: Vec<Entity>,

    /// Pending events (drained by the caller after sync).
    pending_events: Vec<NodeEvent>,

    /// Counter for generating unique default names.
    name_counter: u64,
}

impl SceneGraph {
    /// Create a new, empty scene graph with bidirectional sync.
    pub fn new() -> Self {
        Self {
            tree: SceneNodeTree::new(),
            node_to_entity: HashMap::new(),
            entity_to_node: HashMap::new(),
            sync_direction: SyncDirection::Bidirectional,
            dirty_nodes: Vec::new(),
            dirty_entities: Vec::new(),
            pending_events: Vec::new(),
            name_counter: 0,
        }
    }

    /// Create a scene graph with a specific sync direction policy.
    pub fn with_sync_direction(direction: SyncDirection) -> Self {
        Self {
            sync_direction: direction,
            ..Self::new()
        }
    }

    // -- Node management (integrated with ECS) ------------------------------

    /// Create a root scene node and spawn a corresponding ECS entity with
    /// [`TransformComponent`] and [`GlobalTransform`].
    pub fn create_root(
        &mut self,
        world: &mut World,
        name: impl Into<String>,
    ) -> (NodeId, Entity) {
        let name_str: String = name.into();
        let node_id = self.tree.add_root(&name_str);
        let entity = world
            .spawn_entity()
            .with(TransformComponent::default())
            .with(GlobalTransform::IDENTITY)
            .build();

        self.node_to_entity.insert(node_id, entity);
        self.entity_to_node.insert(entity, node_id);

        // Link entity to node.
        if let Some(node) = self.tree.get_mut(node_id) {
            node.entity = Some(entity);
        }

        self.pending_events.push(NodeEvent::Created {
            node_id,
            entity,
            name: name_str,
        });

        log::trace!(
            "SceneGraph: created root node {:?} <-> {:?}",
            node_id,
            entity
        );
        (node_id, entity)
    }

    /// Create a child scene node under `parent` and spawn a corresponding
    /// ECS entity with transform components.
    ///
    /// Also sets up the parent-child relationship in the
    /// [`TransformHierarchy`] resource if present.
    pub fn create_child(
        &mut self,
        world: &mut World,
        parent: NodeId,
        name: impl Into<String>,
    ) -> (NodeId, Entity) {
        let name_str: String = name.into();
        let node_id = self.tree.add_child(parent, &name_str);
        let entity = world
            .spawn_entity()
            .with(TransformComponent::default())
            .with(GlobalTransform::IDENTITY)
            .build();

        self.node_to_entity.insert(node_id, entity);
        self.entity_to_node.insert(entity, node_id);

        // Link entity to node.
        if let Some(node) = self.tree.get_mut(node_id) {
            node.entity = Some(entity);
        }

        // Set up transform hierarchy in ECS.
        if let Some(parent_entity) = self.node_to_entity.get(&parent).copied() {
            if let Some(hierarchy) = world.get_resource_mut::<TransformHierarchy>() {
                hierarchy.set_parent(entity, parent_entity);
            }
        }

        self.pending_events.push(NodeEvent::Created {
            node_id,
            entity,
            name: name_str,
        });

        log::trace!(
            "SceneGraph: created child node {:?} (parent {:?}) <-> {:?}",
            node_id,
            parent,
            entity,
        );
        (node_id, entity)
    }

    /// Combined: create an entity and a scene node together.
    /// Returns `(Entity, NodeId)`.
    pub fn create_entity_with_node(
        &mut self,
        world: &mut World,
        name: &str,
    ) -> (Entity, NodeId) {
        let (node_id, entity) = self.create_root(world, name);
        (entity, node_id)
    }

    /// Combined: create a child entity and scene node.
    pub fn create_child_entity_with_node(
        &mut self,
        world: &mut World,
        parent_entity: Entity,
        name: &str,
    ) -> Option<(Entity, NodeId)> {
        let parent_node = self.entity_to_node.get(&parent_entity).copied()?;
        let (node_id, entity) = self.create_child(world, parent_node, name);
        Some((entity, node_id))
    }

    /// Generate a unique name for auto-created nodes.
    pub fn generate_name(&mut self, prefix: &str) -> String {
        self.name_counter += 1;
        format!("{}_{}", prefix, self.name_counter)
    }

    /// Remove a scene node and despawn its ECS entity. Also removes all
    /// descendants and their entities.
    pub fn remove_node(&mut self, world: &mut World, node_id: NodeId) {
        // Collect all descendants first.
        let descendants: Vec<NodeId> = self.tree.iter_descendants(node_id).map(|n| n.id).collect();

        // Despawn descendant entities.
        for &desc_id in &descendants {
            if let Some(entity) = self.node_to_entity.remove(&desc_id) {
                self.entity_to_node.remove(&entity);
                self.pending_events.push(NodeEvent::Destroyed {
                    node_id: desc_id,
                    entity,
                });
                // Remove from hierarchy resource.
                if let Some(hierarchy) = world.get_resource_mut::<TransformHierarchy>() {
                    hierarchy.remove_entity(entity);
                }
                world.despawn(entity);
            }
        }

        // Despawn this node's entity.
        if let Some(entity) = self.node_to_entity.remove(&node_id) {
            self.entity_to_node.remove(&entity);
            self.pending_events.push(NodeEvent::Destroyed {
                node_id,
                entity,
            });
            if let Some(hierarchy) = world.get_resource_mut::<TransformHierarchy>() {
                hierarchy.remove_entity(entity);
            }
            world.despawn(entity);
        }

        // Destroy in the tree (which also destroys descendants).
        self.tree.destroy_node(node_id);
    }

    /// Destroy the entity and its associated scene node.
    pub fn destroy_entity_and_node(&mut self, world: &mut World, entity: Entity) {
        if let Some(node_id) = self.entity_to_node.get(&entity).copied() {
            self.remove_node(world, node_id);
        } else {
            // No scene node associated; just despawn.
            world.despawn(entity);
        }
    }

    /// Reparent a node within the scene graph. Returns `true` on success.
    pub fn reparent_node(
        &mut self,
        world: &mut World,
        node_id: NodeId,
        new_parent: Option<NodeId>,
    ) -> bool {
        let old_parent = self.tree.get(node_id).and_then(|n| n.parent);

        if !self.tree.set_parent(node_id, new_parent) {
            return false;
        }

        // Update transform hierarchy in ECS.
        if let Some(entity) = self.node_to_entity.get(&node_id).copied() {
            match new_parent {
                Some(parent_nid) => {
                    if let Some(parent_entity) = self.node_to_entity.get(&parent_nid).copied() {
                        if let Some(hierarchy) = world.get_resource_mut::<TransformHierarchy>() {
                            hierarchy.set_parent(entity, parent_entity);
                        }
                    }
                }
                None => {
                    if let Some(hierarchy) = world.get_resource_mut::<TransformHierarchy>() {
                        hierarchy.remove_parent(entity);
                    }
                }
            }
        }

        self.pending_events.push(NodeEvent::Reparented {
            node_id,
            old_parent,
            new_parent,
        });

        true
    }

    /// Set the visibility of a node and emit an event.
    pub fn set_visibility(&mut self, node_id: NodeId, visible: bool) {
        if let Some(node) = self.tree.get_mut(node_id) {
            if node.visible != visible {
                node.visible = visible;
                self.pending_events.push(NodeEvent::VisibilityChanged {
                    node_id,
                    visible,
                });
            }
        }
    }

    /// Set the local transform of a node via the scene graph.
    pub fn set_node_transform(
        &mut self,
        node_id: NodeId,
        position: Vec3,
        rotation: glam::Quat,
        scale: Vec3,
    ) {
        if let Some(node) = self.tree.get_mut(node_id) {
            node.set_position(position);
            node.set_rotation(rotation);
            node.set_scale(scale);
        }
        self.dirty_nodes.push(node_id);
        self.pending_events.push(NodeEvent::TransformChanged { node_id });
    }

    // -- Lookup -------------------------------------------------------------

    /// Get the ECS entity associated with a scene node.
    #[inline]
    pub fn node_to_entity(&self, node_id: NodeId) -> Option<Entity> {
        self.node_to_entity.get(&node_id).copied()
    }

    /// Get the scene node associated with an ECS entity.
    #[inline]
    pub fn entity_to_node(&self, entity: Entity) -> Option<NodeId> {
        self.entity_to_node.get(&entity).copied()
    }

    /// Get an immutable reference to the underlying scene node tree.
    #[inline]
    pub fn tree(&self) -> &SceneNodeTree {
        &self.tree
    }

    /// Get a mutable reference to the underlying scene node tree.
    /// Changes made directly to the tree must be followed by a call to
    /// [`sync`](Self::sync).
    #[inline]
    pub fn tree_mut(&mut self) -> &mut SceneNodeTree {
        &mut self.tree
    }

    /// Number of managed nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_to_entity.len()
    }

    /// Check if a node id has a corresponding entity.
    #[inline]
    pub fn is_managed(&self, node_id: NodeId) -> bool {
        self.node_to_entity.contains_key(&node_id)
    }

    /// Check if an entity has a corresponding scene node.
    #[inline]
    pub fn is_entity_managed(&self, entity: Entity) -> bool {
        self.entity_to_node.contains_key(&entity)
    }

    // -- Dirty tracking & sync ----------------------------------------------

    /// Mark a scene node as dirty (transform or property changed).
    pub fn mark_node_dirty(&mut self, node_id: NodeId) {
        self.dirty_nodes.push(node_id);
    }

    /// Mark an ECS entity as dirty (component changed outside scene graph).
    pub fn mark_entity_dirty(&mut self, entity: Entity) {
        self.dirty_entities.push(entity);
    }

    /// Synchronize scene graph and ECS based on the configured
    /// [`SyncDirection`] and dirty flags.
    ///
    /// Call this once per frame, after systems have run but before rendering.
    pub fn sync(&mut self, world: &mut World) {
        match self.sync_direction {
            SyncDirection::SceneToEcs => {
                self.sync_scene_to_ecs(world);
            }
            SyncDirection::EcsToScene => {
                self.sync_ecs_to_scene(world);
            }
            SyncDirection::Bidirectional => {
                // Scene-side dirty nodes push to ECS.
                self.sync_scene_to_ecs(world);
                // ECS-side dirty entities push to scene.
                self.sync_ecs_to_scene(world);
            }
        }

        self.dirty_nodes.clear();
        self.dirty_entities.clear();

        // Propagate world transforms through the tree.
        self.tree.propagate_transforms();

        // After propagation, push world transforms to GlobalTransform components.
        self.push_world_transforms_to_ecs(world);
    }

    /// Copy scene node local transforms to ECS TransformComponents.
    pub fn sync_scene_to_ecs(&self, world: &mut World) {
        for &node_id in &self.dirty_nodes {
            if let (Some(node), Some(&entity)) =
                (self.tree.get(node_id), self.node_to_entity.get(&node_id))
            {
                let tc = TransformComponent::new(
                    node.local_position,
                    node.local_rotation,
                    node.local_scale,
                );
                world.add_component(entity, tc);
                log::trace!(
                    "SceneGraph: synced node {:?} -> entity {:?} (scene->ecs)",
                    node_id,
                    entity,
                );
            }
        }
    }

    /// Copy ECS TransformComponents to scene node local transforms.
    pub fn sync_ecs_to_scene(&mut self, world: &World) {
        for &entity in &self.dirty_entities {
            if let Some(&node_id) = self.entity_to_node.get(&entity) {
                if let Some(tc) = world.get_component::<TransformComponent>(entity) {
                    if let Some(node) = self.tree.get_mut(node_id) {
                        node.local_position = tc.position;
                        node.local_rotation = tc.rotation;
                        node.local_scale = tc.scale;
                        node.dirty = true;
                        log::trace!(
                            "SceneGraph: synced entity {:?} -> node {:?} (ecs->scene)",
                            entity,
                            node_id,
                        );
                    }
                }
            }
        }
    }

    /// After transform propagation, push the computed world transforms from
    /// scene nodes to ECS GlobalTransform components.
    fn push_world_transforms_to_ecs(&self, world: &mut World) {
        for (&node_id, &entity) in &self.node_to_entity {
            if let Some(node) = self.tree.get(node_id) {
                let gt = GlobalTransform {
                    matrix: node.world_transform,
                    updated_this_frame: true,
                };
                world.add_component(entity, gt);
            }
        }
    }

    /// Full sync: pull ALL ECS TransformComponents into scene nodes, regardless
    /// of dirty flags. Useful for initial setup or after bulk modifications.
    pub fn full_sync_ecs_to_scene(&mut self, world: &World) {
        for (&entity, &node_id) in &self.entity_to_node {
            if let Some(tc) = world.get_component::<TransformComponent>(entity) {
                if let Some(node) = self.tree.get_mut(node_id) {
                    node.local_position = tc.position;
                    node.local_rotation = tc.rotation;
                    node.local_scale = tc.scale;
                    node.dirty = true;
                }
            }
        }
    }

    /// Full sync: push ALL scene node local transforms to ECS TransformComponents.
    pub fn full_sync_scene_to_ecs(&self, world: &mut World) {
        for (&node_id, &entity) in &self.node_to_entity {
            if let Some(node) = self.tree.get(node_id) {
                let tc = TransformComponent::new(
                    node.local_position,
                    node.local_rotation,
                    node.local_scale,
                );
                world.add_component(entity, tc);
            }
        }
    }

    // -- Events -------------------------------------------------------------

    /// Drain all pending events since the last drain.
    pub fn drain_events(&mut self) -> Vec<NodeEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Return a reference to pending events without draining.
    pub fn pending_events(&self) -> &[NodeEvent] {
        &self.pending_events
    }

    /// Clear all pending events.
    pub fn clear_events(&mut self) {
        self.pending_events.clear();
    }

    // -- Policy -------------------------------------------------------------

    /// Current sync direction policy.
    #[inline]
    pub fn sync_direction(&self) -> SyncDirection {
        self.sync_direction
    }

    /// Change the sync direction policy.
    pub fn set_sync_direction(&mut self, direction: SyncDirection) {
        self.sync_direction = direction;
    }

    // -- Iteration / queries ------------------------------------------------

    /// Iterate over all `(NodeId, Entity)` pairs.
    pub fn iter_pairs(&self) -> impl Iterator<Item = (NodeId, Entity)> + '_ {
        self.node_to_entity.iter().map(|(&n, &e)| (n, e))
    }

    /// Find a node by name through the scene graph.
    pub fn find_node_by_name(&self, name: &str) -> Option<(NodeId, Entity)> {
        let node_id = self.tree.find_by_name(name)?;
        let entity = self.node_to_entity.get(&node_id).copied()?;
        Some((node_id, entity))
    }

    /// Find a node by path through the scene graph.
    pub fn find_node_by_path(&self, path: &str) -> Option<(NodeId, Entity)> {
        let node_id = self.tree.find_by_path(path)?;
        let entity = self.node_to_entity.get(&node_id).copied()?;
        Some((node_id, entity))
    }

    /// Get the world position of a node (from the cached world transform).
    pub fn world_position(&self, node_id: NodeId) -> Option<Vec3> {
        self.tree.get(node_id).map(|n| n.world_position())
    }

    /// Get the world transform matrix of a node.
    pub fn world_transform(&self, node_id: NodeId) -> Option<Mat4> {
        self.tree.get(node_id).map(|n| n.world_transform)
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SceneGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneGraph")
            .field("node_count", &self.node_to_entity.len())
            .field("sync_direction", &self.sync_direction)
            .field("dirty_nodes", &self.dirty_nodes.len())
            .field("dirty_entities", &self.dirty_entities.len())
            .field("pending_events", &self.pending_events.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn create_root_and_lookup() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (node_id, entity) = sg.create_root(&mut world, "Root");

        assert_eq!(sg.node_to_entity(node_id), Some(entity));
        assert_eq!(sg.entity_to_node(entity), Some(node_id));
        assert!(sg.is_managed(node_id));
        assert!(sg.is_entity_managed(entity));
        assert_eq!(sg.node_count(), 1);
    }

    #[test]
    fn create_child_and_lookup() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (root_id, _root_entity) = sg.create_root(&mut world, "Root");
        let (child_id, child_entity) = sg.create_child(&mut world, root_id, "Child");

        assert_eq!(sg.node_to_entity(child_id), Some(child_entity));
        assert_eq!(sg.node_count(), 2);

        // Scene tree should show the parent-child relationship.
        let child_node = sg.tree().get(child_id).unwrap();
        assert_eq!(child_node.parent, Some(root_id));
    }

    #[test]
    fn create_entity_with_node() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (entity, node_id) = sg.create_entity_with_node(&mut world, "Player");

        assert_eq!(sg.node_to_entity(node_id), Some(entity));
        assert_eq!(sg.entity_to_node(entity), Some(node_id));
    }

    #[test]
    fn remove_node_despawns_entity() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (node_id, entity) = sg.create_root(&mut world, "Temp");
        sg.remove_node(&mut world, node_id);

        assert!(!world.is_alive(entity));
        assert!(sg.node_to_entity(node_id).is_none());
        assert_eq!(sg.node_count(), 0);
    }

    #[test]
    fn remove_node_cascades_to_children() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (root_id, root_entity) = sg.create_root(&mut world, "Root");
        let (child_id, child_entity) = sg.create_child(&mut world, root_id, "Child");
        let (grandchild_id, grandchild_entity) =
            sg.create_child(&mut world, child_id, "Grandchild");

        sg.remove_node(&mut world, root_id);

        assert!(!world.is_alive(root_entity));
        assert!(!world.is_alive(child_entity));
        assert!(!world.is_alive(grandchild_entity));
        assert_eq!(sg.node_count(), 0);
    }

    #[test]
    fn destroy_entity_and_node() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (entity, node_id) = sg.create_entity_with_node(&mut world, "Doomed");

        sg.destroy_entity_and_node(&mut world, entity);
        assert!(!world.is_alive(entity));
        assert!(sg.node_to_entity(node_id).is_none());
    }

    #[test]
    fn reparent_node() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (a, _) = sg.create_root(&mut world, "A");
        let (b, _) = sg.create_root(&mut world, "B");
        let (c, _) = sg.create_child(&mut world, a, "C");

        assert!(sg.reparent_node(&mut world, c, Some(b)));
        let c_node = sg.tree().get(c).unwrap();
        assert_eq!(c_node.parent, Some(b));
    }

    #[test]
    fn reparent_to_none_makes_root() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (a, _) = sg.create_root(&mut world, "A");
        let (c, _) = sg.create_child(&mut world, a, "C");

        assert!(sg.reparent_node(&mut world, c, None));
        assert!(sg.tree().get(c).unwrap().is_root());
    }

    #[test]
    fn set_visibility() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (node, _) = sg.create_root(&mut world, "Node");

        assert!(sg.tree().get(node).unwrap().visible);
        sg.set_visibility(node, false);
        assert!(!sg.tree().get(node).unwrap().visible);
    }

    #[test]
    fn set_node_transform() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (node, _) = sg.create_root(&mut world, "Node");

        sg.set_node_transform(node, Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);

        let n = sg.tree().get(node).unwrap();
        assert!((n.local_position - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn sync_scene_to_ecs() {
        let mut world = World::new();
        let mut sg = SceneGraph::with_sync_direction(SyncDirection::SceneToEcs);
        let (node, entity) = sg.create_root(&mut world, "Synced");

        sg.set_node_transform(node, Vec3::new(42.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);
        sg.sync(&mut world);

        let tc = world.get_component::<TransformComponent>(entity).unwrap();
        assert!((tc.position - Vec3::new(42.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn sync_ecs_to_scene() {
        let mut world = World::new();
        let mut sg = SceneGraph::with_sync_direction(SyncDirection::EcsToScene);
        let (node, entity) = sg.create_root(&mut world, "Synced");

        // Modify ECS side directly.
        world.add_component(
            entity,
            TransformComponent::from_position(Vec3::new(99.0, 0.0, 0.0)),
        );
        sg.mark_entity_dirty(entity);
        sg.sync(&mut world);

        let n = sg.tree().get(node).unwrap();
        assert!((n.local_position - Vec3::new(99.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn full_sync_ecs_to_scene() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (node, entity) = sg.create_root(&mut world, "Full");

        world.add_component(
            entity,
            TransformComponent::from_position(Vec3::new(7.0, 8.0, 9.0)),
        );
        sg.full_sync_ecs_to_scene(&world);

        let n = sg.tree().get(node).unwrap();
        assert!((n.local_position - Vec3::new(7.0, 8.0, 9.0)).length() < 1e-5);
    }

    #[test]
    fn events_are_emitted() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();

        let (_node, _entity) = sg.create_root(&mut world, "Evented");
        let events = sg.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            NodeEvent::Created { name, .. } => assert_eq!(name, "Evented"),
            _ => panic!("Expected Created event"),
        }
    }

    #[test]
    fn find_node_by_name() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (node, entity) = sg.create_root(&mut world, "Findable");

        let result = sg.find_node_by_name("Findable");
        assert_eq!(result, Some((node, entity)));
        assert_eq!(sg.find_node_by_name("Nope"), None);
    }

    #[test]
    fn find_node_by_path() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        let (root, _) = sg.create_root(&mut world, "Root");
        let (child, child_entity) = sg.create_child(&mut world, root, "Body");

        let result = sg.find_node_by_path("Root/Body");
        assert_eq!(result, Some((child, child_entity)));
    }

    #[test]
    fn generate_name() {
        let mut sg = SceneGraph::new();
        let n1 = sg.generate_name("Node");
        let n2 = sg.generate_name("Node");
        assert_ne!(n1, n2);
        assert!(n1.starts_with("Node_"));
    }

    #[test]
    fn iter_pairs() {
        let mut world = World::new();
        let mut sg = SceneGraph::new();
        sg.create_root(&mut world, "A");
        sg.create_root(&mut world, "B");

        let pairs: Vec<_> = sg.iter_pairs().collect();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn world_position_after_sync() {
        let mut world = World::new();
        let mut sg = SceneGraph::with_sync_direction(SyncDirection::SceneToEcs);
        let (root, _) = sg.create_root(&mut world, "Root");
        sg.set_node_transform(root, Vec3::new(3.0, 4.0, 5.0), Quat::IDENTITY, Vec3::ONE);
        sg.sync(&mut world);

        let pos = sg.world_position(root).unwrap();
        assert!((pos - Vec3::new(3.0, 4.0, 5.0)).length() < 1e-5);
    }

    #[test]
    fn sync_direction_change() {
        let mut sg = SceneGraph::new();
        assert_eq!(sg.sync_direction(), SyncDirection::Bidirectional);
        sg.set_sync_direction(SyncDirection::SceneToEcs);
        assert_eq!(sg.sync_direction(), SyncDirection::SceneToEcs);
    }
}
