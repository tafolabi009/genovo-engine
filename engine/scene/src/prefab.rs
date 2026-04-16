//! Prefab / template system for the Genovo scene module.
//!
//! A [`Prefab`] is a serializable template describing an entity hierarchy with
//! default component values. [`PrefabInstance`]s are instantiated copies of a
//! prefab that can override individual component values while sharing the
//! template's structure.
//!
//! # Serialization
//!
//! Prefabs serialize to JSON or RON via serde. Component data is stored as
//! type-name-keyed JSON values so the exact component types need not be known
//! at deserialization time.
//!
//! # Instantiation
//!
//! `Prefab::instantiate()` spawns one ECS entity per node, wires up the
//! [`TransformHierarchy`], attaches [`TransformComponent`] and
//! [`GlobalTransform`], and optionally applies per-instance overrides.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use genovo_ecs::Entity;
use glam::{Quat, Vec3};

use crate::ecs_bridge::SceneGraph;
use crate::node::NodeId;
use crate::transform::{GlobalTransform, TransformComponent, TransformHierarchy};

// ---------------------------------------------------------------------------
// PrefabId
// ---------------------------------------------------------------------------

/// Unique identifier for a prefab template (typically an asset hash or UUID).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefabId(pub u64);

impl PrefabId {
    /// Create a new prefab id.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for PrefabId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PrefabId({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// PrefabNodeDescriptor
// ---------------------------------------------------------------------------

/// Describes a single node in a prefab template hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefabNodeDescriptor {
    /// Human-readable name for this node.
    pub name: String,
    /// Index of the parent node within the prefab, or `None` for roots.
    pub parent: Option<usize>,
    /// Local position relative to parent.
    pub position: [f32; 3],
    /// Local rotation as a quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    /// Local scale.
    pub scale: [f32; 3],
    /// Serialized component data keyed by type name. The values are opaque
    /// JSON objects that the component registry can deserialize.
    pub components: HashMap<String, serde_json::Value>,
    /// Tags for this node.
    pub tags: Vec<String>,
    /// Whether this node should be visible by default.
    pub visible: bool,
}

impl PrefabNodeDescriptor {
    /// Create a new descriptor with default values.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parent: None,
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            components: HashMap::new(),
            tags: Vec::new(),
            visible: true,
        }
    }

    /// Set the local position.
    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z];
        self
    }

    /// Set the local rotation (quaternion xyzw).
    pub fn with_rotation(mut self, x: f32, y: f32, z: f32, w: f32) -> Self {
        self.rotation = [x, y, z, w];
        self
    }

    /// Set the local scale.
    pub fn with_scale(mut self, x: f32, y: f32, z: f32) -> Self {
        self.scale = [x, y, z];
        self
    }

    /// Set the parent index.
    pub fn with_parent(mut self, parent: usize) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Add a component.
    pub fn with_component(
        mut self,
        type_name: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.components.insert(type_name.into(), value);
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Convert to a `TransformComponent`.
    pub fn to_transform_component(&self) -> TransformComponent {
        TransformComponent::new(
            Vec3::new(self.position[0], self.position[1], self.position[2]),
            Quat::from_xyzw(
                self.rotation[0],
                self.rotation[1],
                self.rotation[2],
                self.rotation[3],
            ),
            Vec3::new(self.scale[0], self.scale[1], self.scale[2]),
        )
    }

    /// Populate from a `TransformComponent`.
    pub fn from_transform_component(name: impl Into<String>, tc: &TransformComponent) -> Self {
        Self {
            name: name.into(),
            parent: None,
            position: [tc.position.x, tc.position.y, tc.position.z],
            rotation: [tc.rotation.x, tc.rotation.y, tc.rotation.z, tc.rotation.w],
            scale: [tc.scale.x, tc.scale.y, tc.scale.z],
            components: HashMap::new(),
            tags: Vec::new(),
            visible: true,
        }
    }
}

impl Default for PrefabNodeDescriptor {
    fn default() -> Self {
        Self::new("")
    }
}

// ---------------------------------------------------------------------------
// Prefab
// ---------------------------------------------------------------------------

/// A serializable entity template that describes a hierarchy of nodes with
/// default component values.
///
/// Prefabs are typically authored in the editor and stored as assets. At
/// runtime they can be instantiated multiple times; each instance may apply
/// per-instance overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prefab {
    /// Unique identifier.
    pub id: PrefabId,
    /// Human-readable name (e.g. "Enemy_Goblin").
    pub name: String,
    /// Ordered list of node descriptors. Index 0 is typically the root.
    pub nodes: Vec<PrefabNodeDescriptor>,
    /// Metadata / tags for editor classification.
    pub tags: Vec<String>,
}

impl Prefab {
    /// Create a new prefab with a single root node.
    pub fn new(id: PrefabId, name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            id,
            name: name.clone(),
            nodes: vec![PrefabNodeDescriptor::new(&name)],
            tags: Vec::new(),
        }
    }

    /// Create an empty prefab (no nodes).
    pub fn empty(id: PrefabId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            nodes: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Add a child node to the prefab hierarchy. Returns the index of the new node.
    pub fn add_node(
        &mut self,
        parent_index: usize,
        name: impl Into<String>,
    ) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(PrefabNodeDescriptor {
            name: name.into(),
            parent: Some(parent_index),
            ..Default::default()
        });
        idx
    }

    /// Add a root node (no parent). Returns the index.
    pub fn add_root_node(&mut self, name: impl Into<String>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(PrefabNodeDescriptor::new(name));
        idx
    }

    /// Add a fully specified node descriptor. Returns the index.
    pub fn add_descriptor(&mut self, descriptor: PrefabNodeDescriptor) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(descriptor);
        idx
    }

    /// Set a serialized component value on a node.
    pub fn set_component(
        &mut self,
        node_index: usize,
        type_name: impl Into<String>,
        value: serde_json::Value,
    ) {
        if let Some(node) = self.nodes.get_mut(node_index) {
            node.components.insert(type_name.into(), value);
        }
    }

    /// Remove a component from a node.
    pub fn remove_component(
        &mut self,
        node_index: usize,
        type_name: &str,
    ) -> Option<serde_json::Value> {
        self.nodes
            .get_mut(node_index)
            .and_then(|n| n.components.remove(type_name))
    }

    /// Set the transform for a node.
    pub fn set_transform(
        &mut self,
        node_index: usize,
        position: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
    ) {
        if let Some(node) = self.nodes.get_mut(node_index) {
            node.position = position;
            node.rotation = rotation;
            node.scale = scale;
        }
    }

    /// Number of nodes in the prefab hierarchy.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get a reference to a node descriptor by index.
    pub fn get_node(&self, index: usize) -> Option<&PrefabNodeDescriptor> {
        self.nodes.get(index)
    }

    /// Get a mutable reference to a node descriptor by index.
    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut PrefabNodeDescriptor> {
        self.nodes.get_mut(index)
    }

    /// Validate the prefab structure: check that all parent indices are valid
    /// and there are no cycles.
    pub fn validate(&self) -> Result<(), PrefabError> {
        for (i, node) in self.nodes.iter().enumerate() {
            if let Some(parent_idx) = node.parent {
                if parent_idx >= self.nodes.len() {
                    return Err(PrefabError::InvalidParentIndex {
                        node_index: i,
                        parent_index: parent_idx,
                    });
                }
                if parent_idx == i {
                    return Err(PrefabError::SelfParent { node_index: i });
                }
                // Check for cycles: walk up the parent chain.
                let mut visited = vec![false; self.nodes.len()];
                let mut current = Some(i);
                while let Some(idx) = current {
                    if visited[idx] {
                        return Err(PrefabError::CycleDetected { node_index: i });
                    }
                    visited[idx] = true;
                    current = self.nodes[idx].parent;
                }
            }
        }
        Ok(())
    }

    /// Instantiate this prefab into the given world, creating entities for
    /// each node and wiring up the transform hierarchy.
    ///
    /// Returns a [`PrefabInstance`] tracking the spawned entities.
    pub fn instantiate(&self, world: &mut genovo_ecs::World) -> PrefabInstance {
        log::trace!("Instantiating prefab '{}' (id={:?})", self.name, self.id);

        let mut entities: Vec<Entity> = Vec::with_capacity(self.nodes.len());

        // Phase 1: Spawn entities with TransformComponent and GlobalTransform.
        for node_desc in &self.nodes {
            let tc = node_desc.to_transform_component();
            let entity = world
                .spawn_entity()
                .with(tc)
                .with(GlobalTransform::IDENTITY)
                .build();
            entities.push(entity);
        }

        // Phase 2: Set up parent-child relationships in TransformHierarchy.
        // We create a new hierarchy if one doesn't exist yet.
        let has_hierarchy = world.has_resource::<TransformHierarchy>();
        if !has_hierarchy {
            world.add_resource(TransformHierarchy::new());
        }

        for (i, node_desc) in self.nodes.iter().enumerate() {
            if let Some(parent_idx) = node_desc.parent {
                if parent_idx < entities.len() {
                    let child = entities[i];
                    let parent = entities[parent_idx];
                    if let Some(hierarchy) = world.get_resource_mut::<TransformHierarchy>() {
                        hierarchy.set_parent(child, parent);
                    }
                }
            }
        }

        let root = if entities.is_empty() {
            Entity::PLACEHOLDER
        } else {
            // Find the first node without a parent.
            let root_idx = self
                .nodes
                .iter()
                .position(|n| n.parent.is_none())
                .unwrap_or(0);
            entities[root_idx]
        };

        PrefabInstance {
            prefab_id: self.id,
            root,
            entities,
            overrides: HashMap::new(),
        }
    }

    /// Instantiate this prefab into a SceneGraph (creates both entities and
    /// scene nodes).
    pub fn instantiate_into_scene(
        &self,
        world: &mut genovo_ecs::World,
        scene: &mut SceneGraph,
    ) -> PrefabInstance {
        log::trace!(
            "Instantiating prefab '{}' into scene graph",
            self.name,
        );

        let mut entities: Vec<Entity> = Vec::with_capacity(self.nodes.len());
        let mut node_ids: Vec<NodeId> = Vec::with_capacity(self.nodes.len());

        for (_i, node_desc) in self.nodes.iter().enumerate() {
            let (node_id, entity) = match node_desc.parent {
                Some(parent_idx) if parent_idx < node_ids.len() => {
                    let parent_nid = node_ids[parent_idx];
                    scene.create_child(world, parent_nid, &node_desc.name)
                }
                _ => scene.create_root(world, &node_desc.name),
            };

            // Set the transform on the scene node.
            let pos = Vec3::new(
                node_desc.position[0],
                node_desc.position[1],
                node_desc.position[2],
            );
            let rot = Quat::from_xyzw(
                node_desc.rotation[0],
                node_desc.rotation[1],
                node_desc.rotation[2],
                node_desc.rotation[3],
            );
            let scl = Vec3::new(node_desc.scale[0], node_desc.scale[1], node_desc.scale[2]);

            scene.set_node_transform(node_id, pos, rot, scl);

            // Apply tags.
            if let Some(node) = scene.tree_mut().get_mut(node_id) {
                for tag in &node_desc.tags {
                    node.add_tag(tag);
                }
                node.visible = node_desc.visible;
            }

            // Write the TransformComponent to the entity.
            let tc = node_desc.to_transform_component();
            world.add_component(entity, tc);

            entities.push(entity);
            node_ids.push(node_id);
        }

        let root = if entities.is_empty() {
            Entity::PLACEHOLDER
        } else {
            let root_idx = self
                .nodes
                .iter()
                .position(|n| n.parent.is_none())
                .unwrap_or(0);
            entities[root_idx]
        };

        PrefabInstance {
            prefab_id: self.id,
            root,
            entities,
            overrides: HashMap::new(),
        }
    }

    /// Serialize the prefab to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a prefab from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize the prefab to RON format.
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }

    /// Deserialize a prefab from RON format.
    pub fn from_ron(ron_str: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(ron_str)
    }

    /// Add a tag to the prefab metadata.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Check if the prefab has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Get the root node indices (nodes with no parent).
    pub fn root_indices(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent.is_none())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get the children indices of a given node index.
    pub fn children_of(&self, index: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent == Some(index))
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// PrefabError
// ---------------------------------------------------------------------------

/// Errors that can occur during prefab validation or instantiation.
#[derive(Debug, Clone)]
pub enum PrefabError {
    /// A node references a parent index that doesn't exist.
    InvalidParentIndex {
        node_index: usize,
        parent_index: usize,
    },
    /// A node is its own parent.
    SelfParent { node_index: usize },
    /// A cycle was detected in the parent chain.
    CycleDetected { node_index: usize },
}

impl std::fmt::Display for PrefabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrefabError::InvalidParentIndex {
                node_index,
                parent_index,
            } => write!(
                f,
                "Node {} references invalid parent index {}",
                node_index, parent_index
            ),
            PrefabError::SelfParent { node_index } => {
                write!(f, "Node {} is its own parent", node_index)
            }
            PrefabError::CycleDetected { node_index } => {
                write!(f, "Cycle detected starting at node {}", node_index)
            }
        }
    }
}

impl std::error::Error for PrefabError {}

// ---------------------------------------------------------------------------
// ComponentOverride
// ---------------------------------------------------------------------------

/// A per-instance override for a single component field on a specific node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentOverride {
    /// Index of the node in the prefab hierarchy.
    pub node_index: usize,
    /// Type name of the component being overridden.
    pub component_type: String,
    /// The field name being overridden.
    pub field: String,
    /// The overridden value (replaces the prefab's default).
    pub value: serde_json::Value,
}

impl ComponentOverride {
    /// Create a new override.
    pub fn new(
        node_index: usize,
        component_type: impl Into<String>,
        field: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        Self {
            node_index,
            component_type: component_type.into(),
            field: field.into(),
            value,
        }
    }
}

// ---------------------------------------------------------------------------
// PrefabInstance
// ---------------------------------------------------------------------------

/// A live instance of a [`Prefab`] in the world.
///
/// Each instance tracks which entities were spawned and any per-instance
/// overrides applied on top of the template defaults. Overrides are keyed by
/// `(node_index, component_type_name)`.
pub struct PrefabInstance {
    /// The id of the prefab template this instance was created from.
    pub prefab_id: PrefabId,
    /// The root entity of the instantiated hierarchy.
    pub root: Entity,
    /// All entities spawned for this instance, in the same order as the
    /// prefab's node list.
    pub entities: Vec<Entity>,
    /// Per-instance overrides keyed by `(node_index, component_type_name)`.
    pub overrides: HashMap<(usize, String), serde_json::Value>,
}

impl PrefabInstance {
    /// Apply an override for a specific component on a specific node.
    pub fn set_override(
        &mut self,
        node_index: usize,
        component_type: impl Into<String>,
        value: serde_json::Value,
    ) {
        self.overrides
            .insert((node_index, component_type.into()), value);
    }

    /// Apply an override using path and field (for the specification's
    /// `apply_override(path, field, value)` API). The path is mapped to a
    /// node index by name matching.
    pub fn apply_override_by_path(
        &mut self,
        _path: &str,
        field: &str,
        value: serde_json::Value,
        prefab: &Prefab,
    ) {
        // Find node index by matching the last path segment to node names.
        let target_name = _path.rsplit('/').next().unwrap_or(_path);
        for (i, node) in prefab.nodes.iter().enumerate() {
            if node.name == target_name {
                let key = format!("{}:{}", field, target_name);
                self.overrides.insert((i, key), value);
                return;
            }
        }
        log::warn!(
            "PrefabInstance::apply_override: node not found for path '{}'",
            _path
        );
    }

    /// Remove an override, reverting to the prefab template's default value.
    pub fn clear_override(
        &mut self,
        node_index: usize,
        component_type: &str,
    ) -> Option<serde_json::Value> {
        self.overrides
            .remove(&(node_index, component_type.to_string()))
    }

    /// Revert an override by path and field.
    pub fn revert_to_prefab(&mut self, path: &str, field: &str, prefab: &Prefab) {
        let target_name = path.rsplit('/').next().unwrap_or(path);
        for (i, node) in prefab.nodes.iter().enumerate() {
            if node.name == target_name {
                let key = format!("{}:{}", field, target_name);
                self.overrides.remove(&(i, key));
                return;
            }
        }
    }

    /// Apply all pending overrides to the ECS entities.
    ///
    /// Currently handles `TransformComponent` overrides for position, rotation,
    /// and scale. Custom component overrides require a component registry with
    /// deserializers.
    pub fn apply_overrides(&self, world: &mut genovo_ecs::World) {
        for ((node_index, component_type), value) in &self.overrides {
            if *node_index >= self.entities.len() {
                continue;
            }
            let entity = self.entities[*node_index];

            // Handle known transform overrides.
            if component_type == "TransformComponent" || component_type.starts_with("position") {
                if let Some(tc) = world.get_component::<TransformComponent>(entity) {
                    let mut tc_copy = tc.clone();
                    // Try to apply as position override.
                    if let Ok(pos) = serde_json::from_value::<[f32; 3]>(value.clone()) {
                        tc_copy.set_position(Vec3::new(pos[0], pos[1], pos[2]));
                    }
                    world.add_component(entity, tc_copy);
                }
            }

            log::trace!(
                "Applied override: node={}, component={}",
                node_index,
                component_type,
            );
        }
    }

    /// Despawn all entities belonging to this instance.
    pub fn despawn(&self, world: &mut genovo_ecs::World) {
        for &entity in &self.entities {
            world.despawn(entity);
        }
    }

    /// Despawn all entities and also remove from a SceneGraph.
    pub fn despawn_from_scene(
        &self,
        world: &mut genovo_ecs::World,
        scene: &mut SceneGraph,
    ) {
        for &entity in self.entities.iter().rev() {
            scene.destroy_entity_and_node(world, entity);
        }
    }

    /// Returns `true` if any overrides have been set.
    #[inline]
    pub fn has_overrides(&self) -> bool {
        !self.overrides.is_empty()
    }

    /// Number of overrides.
    #[inline]
    pub fn override_count(&self) -> usize {
        self.overrides.len()
    }

    /// Number of entities in this instance.
    #[inline]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get the entity at a given node index.
    pub fn entity_at(&self, node_index: usize) -> Option<Entity> {
        self.entities.get(node_index).copied()
    }

    /// Returns `true` if all entities in this instance are still alive.
    pub fn is_alive(&self, world: &genovo_ecs::World) -> bool {
        self.entities.iter().all(|e| world.is_alive(*e))
    }

    /// Get all override keys.
    pub fn override_keys(&self) -> Vec<(usize, String)> {
        self.overrides.keys().cloned().collect()
    }

    /// Serialize overrides to JSON (for saving instance state).
    pub fn overrides_to_json(&self) -> Result<String, serde_json::Error> {
        // Convert HashMap keys to a serializable format.
        let serializable: Vec<((usize, &str), &serde_json::Value)> = self
            .overrides
            .iter()
            .map(|((idx, name), val)| ((*idx, name.as_str()), val))
            .collect();
        serde_json::to_string_pretty(&serializable)
    }
}

impl std::fmt::Debug for PrefabInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefabInstance")
            .field("prefab_id", &self.prefab_id)
            .field("root", &self.root)
            .field("entity_count", &self.entities.len())
            .field("override_count", &self.overrides.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PrefabRegistry
// ---------------------------------------------------------------------------

/// Asset-style registry that stores all loaded prefab templates for fast
/// lookup by [`PrefabId`] or by name.
pub struct PrefabRegistry {
    /// Prefabs keyed by id.
    prefabs: HashMap<PrefabId, Prefab>,
    /// Name -> id mapping for name-based lookup.
    name_index: HashMap<String, PrefabId>,
    /// Tag -> ids mapping for tag-based lookup.
    tag_index: HashMap<String, Vec<PrefabId>>,
}

impl PrefabRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            prefabs: HashMap::new(),
            name_index: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Register a prefab template. If a prefab with the same id already
    /// exists, it is replaced.
    pub fn register(&mut self, prefab: Prefab) {
        let id = prefab.id;
        let name = prefab.name.clone();
        let tags = prefab.tags.clone();

        // Remove old name/tag index entries if replacing.
        if let Some(old) = self.prefabs.get(&id) {
            self.name_index.remove(&old.name);
            for tag in &old.tags {
                if let Some(ids) = self.tag_index.get_mut(tag) {
                    ids.retain(|&x| x != id);
                }
            }
        }

        self.prefabs.insert(id, prefab);
        self.name_index.insert(name, id);
        for tag in tags {
            self.tag_index.entry(tag).or_default().push(id);
        }
    }

    /// Look up a prefab by id.
    pub fn get(&self, id: PrefabId) -> Option<&Prefab> {
        self.prefabs.get(&id)
    }

    /// Look up a prefab by name.
    pub fn get_by_name(&self, name: &str) -> Option<&Prefab> {
        let id = self.name_index.get(name)?;
        self.prefabs.get(id)
    }

    /// Look up prefab ids by tag.
    pub fn get_by_tag(&self, tag: &str) -> &[PrefabId] {
        self.tag_index
            .get(tag)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Remove a prefab from the registry.
    pub fn remove(&mut self, id: PrefabId) -> Option<Prefab> {
        if let Some(prefab) = self.prefabs.remove(&id) {
            self.name_index.remove(&prefab.name);
            for tag in &prefab.tags {
                if let Some(ids) = self.tag_index.get_mut(tag) {
                    ids.retain(|&x| x != id);
                }
            }
            Some(prefab)
        } else {
            None
        }
    }

    /// Number of registered prefabs.
    #[inline]
    pub fn len(&self) -> usize {
        self.prefabs.len()
    }

    /// Whether the registry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.prefabs.is_empty()
    }

    /// Iterate over all registered prefabs.
    pub fn iter(&self) -> impl Iterator<Item = (&PrefabId, &Prefab)> {
        self.prefabs.iter()
    }

    /// Get all registered prefab names.
    pub fn names(&self) -> Vec<&str> {
        self.name_index.keys().map(|s| s.as_str()).collect()
    }

    /// Get all registered prefab ids.
    pub fn ids(&self) -> Vec<PrefabId> {
        self.prefabs.keys().copied().collect()
    }

    /// Instantiate a prefab by id into the world.
    pub fn instantiate(
        &self,
        id: PrefabId,
        world: &mut genovo_ecs::World,
    ) -> Option<PrefabInstance> {
        let prefab = self.prefabs.get(&id)?;
        Some(prefab.instantiate(world))
    }

    /// Instantiate a prefab by name into the world.
    pub fn instantiate_by_name(
        &self,
        name: &str,
        world: &mut genovo_ecs::World,
    ) -> Option<PrefabInstance> {
        let id = *self.name_index.get(name)?;
        self.instantiate(id, world)
    }

    /// Clear all registered prefabs.
    pub fn clear(&mut self) {
        self.prefabs.clear();
        self.name_index.clear();
        self.tag_index.clear();
    }
}

impl Default for PrefabRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PrefabRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefabRegistry")
            .field("count", &self.prefabs.len())
            .field("names", &self.names())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    // -- PrefabNodeDescriptor tests -----------------------------------------

    #[test]
    fn descriptor_default() {
        let d = PrefabNodeDescriptor::default();
        assert_eq!(d.position, [0.0, 0.0, 0.0]);
        assert_eq!(d.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(d.scale, [1.0, 1.0, 1.0]);
        assert!(d.visible);
    }

    #[test]
    fn descriptor_builder() {
        let d = PrefabNodeDescriptor::new("Arm")
            .with_position(1.0, 2.0, 3.0)
            .with_scale(2.0, 2.0, 2.0)
            .with_parent(0)
            .with_tag("limb");

        assert_eq!(d.name, "Arm");
        assert_eq!(d.position, [1.0, 2.0, 3.0]);
        assert_eq!(d.parent, Some(0));
        assert!(d.tags.contains(&"limb".to_string()));
    }

    #[test]
    fn descriptor_to_transform_component() {
        let d = PrefabNodeDescriptor::new("Test").with_position(5.0, 10.0, 15.0);
        let tc = d.to_transform_component();
        assert!((tc.position - Vec3::new(5.0, 10.0, 15.0)).length() < 1e-5);
    }

    #[test]
    fn descriptor_from_transform_component() {
        let tc = TransformComponent::from_position(Vec3::new(7.0, 8.0, 9.0));
        let d = PrefabNodeDescriptor::from_transform_component("FromTC", &tc);
        assert_eq!(d.name, "FromTC");
        assert!((d.position[0] - 7.0).abs() < 1e-5);
    }

    // -- Prefab tests -------------------------------------------------------

    #[test]
    fn prefab_add_nodes() {
        let mut prefab = Prefab::new(PrefabId(1), "TestPrefab");
        assert_eq!(prefab.node_count(), 1);
        let child = prefab.add_node(0, "Child");
        assert_eq!(child, 1);
        assert_eq!(prefab.node_count(), 2);
        assert_eq!(prefab.nodes[child].parent, Some(0));
    }

    #[test]
    fn prefab_add_root_node() {
        let mut prefab = Prefab::new(PrefabId(1), "Multi");
        let r2 = prefab.add_root_node("SecondRoot");
        assert_eq!(prefab.node_count(), 2);
        assert_eq!(prefab.nodes[r2].parent, None);
    }

    #[test]
    fn prefab_set_component() {
        let mut prefab = Prefab::new(PrefabId(1), "WithComp");
        prefab.set_component(0, "Health", serde_json::json!(100));
        assert_eq!(
            prefab.nodes[0].components.get("Health"),
            Some(&serde_json::json!(100))
        );
    }

    #[test]
    fn prefab_remove_component() {
        let mut prefab = Prefab::new(PrefabId(1), "WithComp");
        prefab.set_component(0, "Health", serde_json::json!(100));
        let removed = prefab.remove_component(0, "Health");
        assert_eq!(removed, Some(serde_json::json!(100)));
        assert!(prefab.nodes[0].components.is_empty());
    }

    #[test]
    fn prefab_set_transform() {
        let mut prefab = Prefab::new(PrefabId(1), "Transformed");
        prefab.set_transform(0, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0]);
        assert_eq!(prefab.nodes[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(prefab.nodes[0].scale, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn prefab_validate_ok() {
        let mut prefab = Prefab::new(PrefabId(1), "Valid");
        prefab.add_node(0, "Child");
        assert!(prefab.validate().is_ok());
    }

    #[test]
    fn prefab_validate_invalid_parent() {
        let mut prefab = Prefab::new(PrefabId(1), "Invalid");
        prefab.nodes.push(PrefabNodeDescriptor {
            name: "Bad".to_string(),
            parent: Some(999),
            ..Default::default()
        });
        assert!(prefab.validate().is_err());
    }

    #[test]
    fn prefab_validate_self_parent() {
        let mut prefab = Prefab::empty(PrefabId(1), "SelfParent");
        prefab.nodes.push(PrefabNodeDescriptor {
            name: "Self".to_string(),
            parent: Some(0),
            ..Default::default()
        });
        assert!(prefab.validate().is_err());
    }

    #[test]
    fn prefab_tags() {
        let mut prefab = Prefab::new(PrefabId(1), "Tagged");
        prefab.add_tag("enemy");
        prefab.add_tag("boss");
        assert!(prefab.has_tag("enemy"));
        assert!(prefab.has_tag("boss"));
        assert!(!prefab.has_tag("player"));
        // Duplicate add.
        prefab.add_tag("enemy");
        assert_eq!(prefab.tags.len(), 2);
    }

    #[test]
    fn prefab_root_indices() {
        let mut prefab = Prefab::new(PrefabId(1), "Multi");
        prefab.add_root_node("R2");
        prefab.add_node(0, "C1");
        let roots = prefab.root_indices();
        assert_eq!(roots, vec![0, 1]);
    }

    #[test]
    fn prefab_children_of() {
        let mut prefab = Prefab::new(PrefabId(1), "Parent");
        prefab.add_node(0, "C1");
        prefab.add_node(0, "C2");
        let children = prefab.children_of(0);
        assert_eq!(children, vec![1, 2]);
    }

    // -- Serialization tests ------------------------------------------------

    #[test]
    fn prefab_json_roundtrip() {
        let mut prefab = Prefab::new(PrefabId(42), "Goblin");
        prefab.add_node(0, "Sword");
        prefab.set_component(0, "Health", serde_json::json!(100));
        prefab.set_transform(0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]);

        let json = prefab.to_json().unwrap();
        let loaded = Prefab::from_json(&json).unwrap();

        assert_eq!(loaded.id, PrefabId(42));
        assert_eq!(loaded.name, "Goblin");
        assert_eq!(loaded.node_count(), 2);
        assert_eq!(loaded.nodes[1].name, "Sword");
    }

    #[test]
    fn prefab_ron_roundtrip() {
        let mut prefab = Prefab::new(PrefabId(7), "Knight");
        prefab.add_node(0, "Shield");

        let ron_str = prefab.to_ron().unwrap();
        let loaded = Prefab::from_ron(&ron_str).unwrap();

        assert_eq!(loaded.id, PrefabId(7));
        assert_eq!(loaded.name, "Knight");
        assert_eq!(loaded.node_count(), 2);
    }

    // -- Instance tests -----------------------------------------------------

    #[test]
    fn instance_overrides() {
        let prefab = Prefab::new(PrefabId(2), "OverridePrefab");
        let mut world = genovo_ecs::World::new();
        let mut instance = prefab.instantiate(&mut world);

        instance.set_override(0, "Health", serde_json::json!(50));
        assert!(instance.has_overrides());
        assert_eq!(instance.override_count(), 1);

        instance.clear_override(0, "Health");
        assert!(!instance.has_overrides());
    }

    #[test]
    fn instance_entity_count() {
        let mut prefab = Prefab::new(PrefabId(3), "Multi");
        prefab.add_node(0, "A");
        prefab.add_node(0, "B");

        let mut world = genovo_ecs::World::new();
        let instance = prefab.instantiate(&mut world);

        assert_eq!(instance.entity_count(), 3);
        assert!(!instance.root.is_placeholder());
    }

    #[test]
    fn instance_despawn() {
        let prefab = Prefab::new(PrefabId(4), "Despawnable");
        let mut world = genovo_ecs::World::new();
        let instance = prefab.instantiate(&mut world);

        assert!(instance.is_alive(&world));
        instance.despawn(&mut world);
        assert!(!instance.is_alive(&world));
    }

    #[test]
    fn instance_entity_at() {
        let mut prefab = Prefab::new(PrefabId(5), "Indexed");
        prefab.add_node(0, "Child");

        let mut world = genovo_ecs::World::new();
        let instance = prefab.instantiate(&mut world);

        assert!(instance.entity_at(0).is_some());
        assert!(instance.entity_at(1).is_some());
        assert!(instance.entity_at(99).is_none());
    }

    #[test]
    fn instance_override_keys() {
        let prefab = Prefab::new(PrefabId(6), "Keys");
        let mut world = genovo_ecs::World::new();
        let mut instance = prefab.instantiate(&mut world);

        instance.set_override(0, "A", serde_json::json!(1));
        instance.set_override(0, "B", serde_json::json!(2));

        let keys = instance.override_keys();
        assert_eq!(keys.len(), 2);
    }

    // -- PrefabRegistry tests -----------------------------------------------

    #[test]
    fn registry_crud() {
        let mut reg = PrefabRegistry::new();
        let p = Prefab::new(PrefabId(10), "Goblin");
        reg.register(p);
        assert_eq!(reg.len(), 1);
        assert!(reg.get(PrefabId(10)).is_some());
        reg.remove(PrefabId(10));
        assert!(reg.is_empty());
    }

    #[test]
    fn registry_get_by_name() {
        let mut reg = PrefabRegistry::new();
        reg.register(Prefab::new(PrefabId(1), "Goblin"));
        reg.register(Prefab::new(PrefabId(2), "Troll"));

        assert!(reg.get_by_name("Goblin").is_some());
        assert!(reg.get_by_name("Troll").is_some());
        assert!(reg.get_by_name("Dragon").is_none());
    }

    #[test]
    fn registry_get_by_tag() {
        let mut reg = PrefabRegistry::new();
        let mut p1 = Prefab::new(PrefabId(1), "Goblin");
        p1.add_tag("enemy");
        let mut p2 = Prefab::new(PrefabId(2), "Troll");
        p2.add_tag("enemy");
        p2.add_tag("boss");

        reg.register(p1);
        reg.register(p2);

        assert_eq!(reg.get_by_tag("enemy").len(), 2);
        assert_eq!(reg.get_by_tag("boss").len(), 1);
        assert_eq!(reg.get_by_tag("friendly").len(), 0);
    }

    #[test]
    fn registry_replace() {
        let mut reg = PrefabRegistry::new();
        reg.register(Prefab::new(PrefabId(1), "OldName"));
        reg.register(Prefab::new(PrefabId(1), "NewName"));

        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get(PrefabId(1)).unwrap().name, "NewName");
        assert!(reg.get_by_name("NewName").is_some());
        assert!(reg.get_by_name("OldName").is_none());
    }

    #[test]
    fn registry_instantiate_by_name() {
        let mut reg = PrefabRegistry::new();
        reg.register(Prefab::new(PrefabId(1), "Bat"));

        let mut world = genovo_ecs::World::new();
        let instance = reg.instantiate_by_name("Bat", &mut world);
        assert!(instance.is_some());
        assert_eq!(instance.unwrap().entity_count(), 1);
    }

    #[test]
    fn registry_names_and_ids() {
        let mut reg = PrefabRegistry::new();
        reg.register(Prefab::new(PrefabId(1), "A"));
        reg.register(Prefab::new(PrefabId(2), "B"));

        let names = reg.names();
        assert_eq!(names.len(), 2);

        let ids = reg.ids();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn registry_clear() {
        let mut reg = PrefabRegistry::new();
        reg.register(Prefab::new(PrefabId(1), "A"));
        reg.register(Prefab::new(PrefabId(2), "B"));

        reg.clear();
        assert!(reg.is_empty());
    }

    // -- Integration test: instantiate into scene ---------------------------

    #[test]
    fn instantiate_into_scene() {
        let mut prefab = Prefab::new(PrefabId(100), "Robot");
        prefab.set_transform(0, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]);
        let body = prefab.add_node(0, "Body");
        let _left_arm = prefab.add_node(body, "LeftArm");
        let _right_arm = prefab.add_node(body, "RightArm");

        let mut world = genovo_ecs::World::new();
        let mut scene = SceneGraph::new();

        let instance = prefab.instantiate_into_scene(&mut world, &mut scene);
        assert_eq!(instance.entity_count(), 4);
        assert_eq!(scene.node_count(), 4);

        // Check that the hierarchy is correct in the scene tree.
        let root_path = scene.find_node_by_path("Robot");
        assert!(root_path.is_some());

        let body_path = scene.find_node_by_path("Robot/Body");
        assert!(body_path.is_some());
    }

    #[test]
    fn prefab_id_display() {
        let id = PrefabId(42);
        assert_eq!(format!("{}", id), "PrefabId(42)");
    }

    #[test]
    fn descriptor_with_component() {
        let d = PrefabNodeDescriptor::new("Node")
            .with_component("Mesh", serde_json::json!({"path": "mesh.glb"}));
        assert!(d.components.contains_key("Mesh"));
    }
}
