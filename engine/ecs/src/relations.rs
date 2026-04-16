//! Entity relations for the Genovo ECS.
//!
//! Relations model directed connections between entities -- parent-child
//! hierarchies, custom graph edges, and typed links. Unlike regular
//! components which are self-contained data on a single entity, relations
//! express *connections* between two entities.
//!
//! # Built-in relations
//!
//! - [`ChildOf`] -- the fundamental parent-child relation. When a parent
//!   entity is despawned, all children are recursively despawned too
//!   (cascading delete).
//!
//! # Custom relations
//!
//! ```ignore
//! // Define a custom relation marker.
//! struct LikesColor;
//! impl RelationKind for LikesColor {}
//!
//! // Add a relation between entities.
//! relations.add::<LikesColor>(entity_player, entity_red);
//!
//! // Query all entities related to a given entity.
//! for target in relations.targets::<LikesColor>(entity_player) {
//!     // entity_player -[LikesColor]-> target
//! }
//! ```
//!
//! # Cascading delete
//!
//! When an entity is despawned through the [`RelationManager`], all entities
//! that are children (`ChildOf`) of the despawned entity are also despawned
//! recursively. This ensures hierarchical consistency.

use std::any::TypeId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

// ---------------------------------------------------------------------------
// Relation traits and markers
// ---------------------------------------------------------------------------

/// Marker trait for relation types. Implement this to define a custom
/// relation kind.
pub trait RelationKind: 'static + Send + Sync {}

/// The built-in parent-child relation. An entity with `ChildOf` is a child
/// of the target entity.
///
/// ```ignore
/// relations.add_child_of(child, parent);
/// // equivalent to:
/// relations.add::<ChildOf>(child, parent);
/// ```
pub struct ChildOf;
impl RelationKind for ChildOf {}

/// A generic relation component that can be stored on entities.
///
/// This is useful when you want the relation to also be queryable as a
/// component (e.g., for finding all entities that have a parent).
#[derive(Debug, Clone)]
pub struct Relation<R: RelationKind> {
    /// The target entity of the relation.
    pub target: Entity,
    _marker: PhantomData<R>,
}

impl<R: RelationKind> Relation<R> {
    /// Create a new relation pointing to `target`.
    pub fn new(target: Entity) -> Self {
        Self {
            target,
            _marker: PhantomData,
        }
    }
}

impl<R: RelationKind> Component for Relation<R> {}

// ---------------------------------------------------------------------------
// RelationId -- identifies a specific relation kind
// ---------------------------------------------------------------------------

/// Runtime identifier for a relation kind, derived from `TypeId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelationId(TypeId);

impl RelationId {
    /// Get the relation id for a specific kind.
    #[inline]
    pub fn of<R: RelationKind>() -> Self {
        Self(TypeId::of::<R>())
    }
}

// ---------------------------------------------------------------------------
// RelationEdge -- a single directed edge
// ---------------------------------------------------------------------------

/// A directed edge from `source` to `target` with a specific relation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelationEdge {
    /// The source entity.
    pub source: Entity,
    /// The target entity.
    pub target: Entity,
    /// Which relation kind this edge represents.
    pub relation: RelationId,
}

// ---------------------------------------------------------------------------
// RelationStorage -- internal storage for all edges of one kind
// ---------------------------------------------------------------------------

/// Storage for all edges of a single relation kind.
///
/// Maintains two indices: source-to-targets and target-to-sources.
#[derive(Debug, Clone, Default)]
struct RelationStorage {
    /// For each source entity, the set of target entities.
    source_to_targets: HashMap<Entity, Vec<Entity>>,
    /// For each target entity, the set of source entities.
    target_to_sources: HashMap<Entity, Vec<Entity>>,
    /// Total edge count.
    edge_count: usize,
}

impl RelationStorage {
    fn new() -> Self {
        Self::default()
    }

    /// Add an edge from source to target. Returns `true` if newly added.
    fn add(&mut self, source: Entity, target: Entity) -> bool {
        let targets = self.source_to_targets.entry(source).or_default();
        if targets.contains(&target) {
            return false;
        }
        targets.push(target);
        self.target_to_sources
            .entry(target)
            .or_default()
            .push(source);
        self.edge_count += 1;
        true
    }

    /// Remove an edge from source to target. Returns `true` if found.
    fn remove(&mut self, source: Entity, target: Entity) -> bool {
        let removed_forward = if let Some(targets) = self.source_to_targets.get_mut(&source) {
            if let Some(pos) = targets.iter().position(|&t| t == target) {
                targets.swap_remove(pos);
                if targets.is_empty() {
                    self.source_to_targets.remove(&source);
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        if removed_forward {
            if let Some(sources) = self.target_to_sources.get_mut(&target) {
                if let Some(pos) = sources.iter().position(|&s| s == source) {
                    sources.swap_remove(pos);
                    if sources.is_empty() {
                        self.target_to_sources.remove(&target);
                    }
                }
            }
            self.edge_count -= 1;
        }

        removed_forward
    }

    /// Remove all edges involving the given entity (as source or target).
    fn remove_entity(&mut self, entity: Entity) {
        // Remove as source.
        if let Some(targets) = self.source_to_targets.remove(&entity) {
            for target in &targets {
                if let Some(sources) = self.target_to_sources.get_mut(target) {
                    sources.retain(|&s| s != entity);
                    if sources.is_empty() {
                        self.target_to_sources.remove(target);
                    }
                }
            }
            self.edge_count -= targets.len();
        }

        // Remove as target.
        if let Some(sources) = self.target_to_sources.remove(&entity) {
            for source in &sources {
                if let Some(targets) = self.source_to_targets.get_mut(source) {
                    let before = targets.len();
                    targets.retain(|&t| t != entity);
                    let removed = before - targets.len();
                    self.edge_count -= removed;
                    if targets.is_empty() {
                        self.source_to_targets.remove(source);
                    }
                }
            }
        }
    }

    /// Get all targets for a given source entity.
    fn targets(&self, source: Entity) -> &[Entity] {
        self.source_to_targets
            .get(&source)
            .map_or(&[], |v| v.as_slice())
    }

    /// Get all sources for a given target entity.
    fn sources(&self, target: Entity) -> &[Entity] {
        self.target_to_sources
            .get(&target)
            .map_or(&[], |v| v.as_slice())
    }

    /// Check if an edge exists.
    fn has(&self, source: Entity, target: Entity) -> bool {
        self.source_to_targets
            .get(&source)
            .map_or(false, |targets| targets.contains(&target))
    }

    /// Total number of edges.
    fn len(&self) -> usize {
        self.edge_count
    }
}

// ---------------------------------------------------------------------------
// RelationManager
// ---------------------------------------------------------------------------

/// Central manager for all entity relations.
///
/// Stores edges for all relation kinds and provides methods for adding,
/// removing, querying, and cascading-deleting relations.
pub struct RelationManager {
    /// Per-relation-kind storage.
    storages: HashMap<RelationId, RelationStorage>,
}

impl RelationManager {
    /// Create a new, empty relation manager.
    pub fn new() -> Self {
        Self {
            storages: HashMap::new(),
        }
    }

    // -- Edge management ------------------------------------------------------

    /// Add a relation from `source` to `target`.
    ///
    /// Returns `true` if the edge was newly created, `false` if it already
    /// existed.
    pub fn add<R: RelationKind>(&mut self, source: Entity, target: Entity) -> bool {
        let rid = RelationId::of::<R>();
        self.storages.entry(rid).or_insert_with(RelationStorage::new).add(source, target)
    }

    /// Remove a relation from `source` to `target`.
    ///
    /// Returns `true` if the edge was found and removed.
    pub fn remove<R: RelationKind>(&mut self, source: Entity, target: Entity) -> bool {
        let rid = RelationId::of::<R>();
        if let Some(storage) = self.storages.get_mut(&rid) {
            storage.remove(source, target)
        } else {
            false
        }
    }

    /// Check if a relation exists from `source` to `target`.
    pub fn has<R: RelationKind>(&self, source: Entity, target: Entity) -> bool {
        let rid = RelationId::of::<R>();
        self.storages
            .get(&rid)
            .map_or(false, |s| s.has(source, target))
    }

    /// Get all targets of `source` for relation `R`.
    pub fn targets<R: RelationKind>(&self, source: Entity) -> &[Entity] {
        let rid = RelationId::of::<R>();
        self.storages
            .get(&rid)
            .map_or(&[], |s| s.targets(source))
    }

    /// Get all sources pointing to `target` for relation `R`.
    pub fn sources<R: RelationKind>(&self, target: Entity) -> &[Entity] {
        let rid = RelationId::of::<R>();
        self.storages
            .get(&rid)
            .map_or(&[], |s| s.sources(target))
    }

    // -- Parent-child convenience methods ------------------------------------

    /// Add a parent-child relationship: `child` is a child of `parent`.
    pub fn add_child_of(&mut self, child: Entity, parent: Entity) -> bool {
        self.add::<ChildOf>(child, parent)
    }

    /// Remove a parent-child relationship.
    pub fn remove_child_of(&mut self, child: Entity, parent: Entity) -> bool {
        self.remove::<ChildOf>(child, parent)
    }

    /// Get the parent of `child`, if any.
    ///
    /// A child can have multiple ChildOf targets (multiple parents), but
    /// this returns the first one found. For strict single-parent
    /// hierarchies, ensure you only add one ChildOf per entity.
    pub fn parent(&self, child: Entity) -> Option<Entity> {
        let targets = self.targets::<ChildOf>(child);
        targets.first().copied()
    }

    /// Get all children of `parent`.
    ///
    /// These are all entities that have a `ChildOf` edge pointing to
    /// `parent` (i.e., `child -[ChildOf]-> parent`).
    pub fn children(&self, parent: Entity) -> &[Entity] {
        self.sources::<ChildOf>(parent)
    }

    /// Iterate over all descendants (children, grandchildren, etc.) of
    /// `ancestor` via breadth-first traversal.
    pub fn descendants(&self, ancestor: Entity) -> Vec<Entity> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(ancestor);

        while let Some(current) = queue.pop_front() {
            let children = self.children(current);
            for &child in children {
                result.push(child);
                queue.push_back(child);
            }
        }

        result
    }

    /// Iterate over all ancestors (parent, grandparent, etc.) of `entity`.
    pub fn ancestors(&self, entity: Entity) -> Vec<Entity> {
        let mut result = Vec::new();
        let mut current = entity;

        while let Some(parent) = self.parent(current) {
            if result.contains(&parent) {
                // Cycle detection -- should not happen in a well-formed
                // hierarchy but prevents infinite loops.
                break;
            }
            result.push(parent);
            current = parent;
        }

        result
    }

    /// Check if `ancestor` is an ancestor of `entity`.
    pub fn is_ancestor_of(&self, ancestor: Entity, entity: Entity) -> bool {
        self.ancestors(entity).contains(&ancestor)
    }

    /// Check if `descendant` is a descendant of `entity`.
    pub fn is_descendant_of(&self, descendant: Entity, entity: Entity) -> bool {
        self.descendants(entity).contains(&descendant)
    }

    /// Get the root of the hierarchy containing `entity`.
    pub fn root(&self, entity: Entity) -> Entity {
        let ancestors = self.ancestors(entity);
        ancestors.last().copied().unwrap_or(entity)
    }

    /// Get the depth of `entity` in the hierarchy (root = 0).
    pub fn depth(&self, entity: Entity) -> usize {
        self.ancestors(entity).len()
    }

    // -- Cascading delete -----------------------------------------------------

    /// Despawn an entity and all its descendants (cascading delete).
    ///
    /// This performs a breadth-first traversal of the `ChildOf` hierarchy,
    /// collecting all descendants, then despawns them all from the world.
    /// Relations involving any of the despawned entities are also cleaned up.
    pub fn despawn_recursive(&mut self, entity: Entity, world: &mut World) {
        // Collect all entities to despawn.
        let mut to_despawn = Vec::new();
        to_despawn.push(entity);

        let descendants = self.descendants(entity);
        to_despawn.extend(descendants);

        // Remove all relations involving these entities.
        for &e in &to_despawn {
            self.remove_all_edges(e);
        }

        // Despawn from the world (in reverse order so children go first).
        for &e in to_despawn.iter().rev() {
            world.despawn(e);
        }
    }

    /// Remove all edges (of all relation kinds) involving the given entity.
    pub fn remove_all_edges(&mut self, entity: Entity) {
        for storage in self.storages.values_mut() {
            storage.remove_entity(entity);
        }
    }

    // -- Relation queries -----------------------------------------------------

    /// Get the total number of edges of a specific relation kind.
    pub fn edge_count<R: RelationKind>(&self) -> usize {
        let rid = RelationId::of::<R>();
        self.storages.get(&rid).map_or(0, |s| s.len())
    }

    /// Get all edges of a specific relation kind.
    pub fn all_edges<R: RelationKind>(&self) -> Vec<RelationEdge> {
        let rid = RelationId::of::<R>();
        if let Some(storage) = self.storages.get(&rid) {
            let mut edges = Vec::new();
            for (&source, targets) in &storage.source_to_targets {
                for &target in targets {
                    edges.push(RelationEdge {
                        source,
                        target,
                        relation: rid,
                    });
                }
            }
            edges
        } else {
            Vec::new()
        }
    }

    /// Get all entities that are sources in the given relation kind.
    pub fn all_sources<R: RelationKind>(&self) -> Vec<Entity> {
        let rid = RelationId::of::<R>();
        self.storages
            .get(&rid)
            .map_or(Vec::new(), |s| s.source_to_targets.keys().copied().collect())
    }

    /// Get all entities that are targets in the given relation kind.
    pub fn all_targets<R: RelationKind>(&self) -> Vec<Entity> {
        let rid = RelationId::of::<R>();
        self.storages
            .get(&rid)
            .map_or(Vec::new(), |s| s.target_to_sources.keys().copied().collect())
    }

    /// Check if the entity has any outgoing edges of any relation kind.
    pub fn has_any_relations(&self, entity: Entity) -> bool {
        for storage in self.storages.values() {
            if !storage.targets(entity).is_empty() {
                return true;
            }
            if !storage.sources(entity).is_empty() {
                return true;
            }
        }
        false
    }

    /// Total number of relation kinds with at least one edge.
    pub fn active_relation_count(&self) -> usize {
        self.storages.values().filter(|s| s.len() > 0).count()
    }

    /// Clear all relations.
    pub fn clear(&mut self) {
        self.storages.clear();
    }
}

impl Default for RelationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// World extension methods
// ---------------------------------------------------------------------------

impl World {
    /// Convenience method: add a parent-child relation and store the relation
    /// manager as a resource.
    pub fn add_relation<R: RelationKind>(&mut self, source: Entity, target: Entity) {
        if !self.has_resource::<RelationManager>() {
            self.add_resource(RelationManager::new());
        }
        if let Some(rm) = self.get_resource_mut::<RelationManager>() {
            rm.add::<R>(source, target);
        }
    }

    /// Convenience method: get children of a parent entity.
    pub fn children_of(&self, parent: Entity) -> Vec<Entity> {
        self.get_resource::<RelationManager>()
            .map_or(Vec::new(), |rm| rm.children(parent).to_vec())
    }

    /// Convenience method: get parent of a child entity.
    pub fn parent_of(&self, child: Entity) -> Option<Entity> {
        self.get_resource::<RelationManager>()
            .and_then(|rm| rm.parent(child))
    }

    /// Despawn entity and all children recursively via the relation manager.
    pub fn despawn_recursive(&mut self, entity: Entity) {
        if let Some(mut rm) = self.remove_resource::<RelationManager>() {
            rm.despawn_recursive(entity, self);
            self.add_resource(rm);
        } else {
            self.despawn(entity);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Position { x: f32, y: f32 }
    impl Component for Position {}

    struct LikesColor;
    impl RelationKind for LikesColor {}

    struct Targets;
    impl RelationKind for Targets {}

    #[test]
    fn add_and_remove_relation() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let e1 = world.spawn_entity().build();
        let e2 = world.spawn_entity().build();

        assert!(rm.add::<LikesColor>(e1, e2));
        assert!(rm.has::<LikesColor>(e1, e2));
        assert!(!rm.has::<LikesColor>(e2, e1)); // directed

        // Adding again returns false.
        assert!(!rm.add::<LikesColor>(e1, e2));

        assert!(rm.remove::<LikesColor>(e1, e2));
        assert!(!rm.has::<LikesColor>(e1, e2));
    }

    #[test]
    fn targets_and_sources() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let e1 = world.spawn_entity().build();
        let e2 = world.spawn_entity().build();
        let e3 = world.spawn_entity().build();

        rm.add::<LikesColor>(e1, e2);
        rm.add::<LikesColor>(e1, e3);

        assert_eq!(rm.targets::<LikesColor>(e1).len(), 2);
        assert!(rm.targets::<LikesColor>(e1).contains(&e2));
        assert!(rm.targets::<LikesColor>(e1).contains(&e3));

        assert_eq!(rm.sources::<LikesColor>(e2).len(), 1);
        assert!(rm.sources::<LikesColor>(e2).contains(&e1));
    }

    #[test]
    fn parent_child_basic() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let parent = world.spawn_entity().build();
        let child1 = world.spawn_entity().build();
        let child2 = world.spawn_entity().build();

        rm.add_child_of(child1, parent);
        rm.add_child_of(child2, parent);

        assert_eq!(rm.parent(child1), Some(parent));
        assert_eq!(rm.parent(child2), Some(parent));
        assert_eq!(rm.parent(parent), None);

        let children = rm.children(parent);
        assert_eq!(children.len(), 2);
        assert!(children.contains(&child1));
        assert!(children.contains(&child2));
    }

    #[test]
    fn descendants() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let root = world.spawn_entity().build();
        let child1 = world.spawn_entity().build();
        let child2 = world.spawn_entity().build();
        let grandchild = world.spawn_entity().build();

        rm.add_child_of(child1, root);
        rm.add_child_of(child2, root);
        rm.add_child_of(grandchild, child1);

        let desc = rm.descendants(root);
        assert_eq!(desc.len(), 3);
        assert!(desc.contains(&child1));
        assert!(desc.contains(&child2));
        assert!(desc.contains(&grandchild));
    }

    #[test]
    fn ancestors() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let root = world.spawn_entity().build();
        let child = world.spawn_entity().build();
        let grandchild = world.spawn_entity().build();

        rm.add_child_of(child, root);
        rm.add_child_of(grandchild, child);

        let anc = rm.ancestors(grandchild);
        assert_eq!(anc.len(), 2);
        assert_eq!(anc[0], child);
        assert_eq!(anc[1], root);
    }

    #[test]
    fn cascading_delete() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let root = world.spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .build();
        let child1 = world.spawn_entity()
            .with(Position { x: 1.0, y: 0.0 })
            .build();
        let child2 = world.spawn_entity()
            .with(Position { x: 2.0, y: 0.0 })
            .build();
        let grandchild = world.spawn_entity()
            .with(Position { x: 3.0, y: 0.0 })
            .build();
        let unrelated = world.spawn_entity()
            .with(Position { x: 4.0, y: 0.0 })
            .build();

        rm.add_child_of(child1, root);
        rm.add_child_of(child2, root);
        rm.add_child_of(grandchild, child1);

        rm.despawn_recursive(root, &mut world);

        assert!(!world.is_alive(root));
        assert!(!world.is_alive(child1));
        assert!(!world.is_alive(child2));
        assert!(!world.is_alive(grandchild));
        assert!(world.is_alive(unrelated));
    }

    #[test]
    fn root_and_depth() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let root = world.spawn_entity().build();
        let child = world.spawn_entity().build();
        let grandchild = world.spawn_entity().build();

        rm.add_child_of(child, root);
        rm.add_child_of(grandchild, child);

        assert_eq!(rm.root(grandchild), root);
        assert_eq!(rm.root(child), root);
        assert_eq!(rm.root(root), root);

        assert_eq!(rm.depth(root), 0);
        assert_eq!(rm.depth(child), 1);
        assert_eq!(rm.depth(grandchild), 2);
    }

    #[test]
    fn is_ancestor_descendant() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let root = world.spawn_entity().build();
        let child = world.spawn_entity().build();
        let grandchild = world.spawn_entity().build();

        rm.add_child_of(child, root);
        rm.add_child_of(grandchild, child);

        assert!(rm.is_ancestor_of(root, grandchild));
        assert!(rm.is_ancestor_of(child, grandchild));
        assert!(!rm.is_ancestor_of(grandchild, root));

        assert!(rm.is_descendant_of(grandchild, root));
        assert!(!rm.is_descendant_of(root, grandchild));
    }

    #[test]
    fn remove_child_of() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let parent = world.spawn_entity().build();
        let child = world.spawn_entity().build();

        rm.add_child_of(child, parent);
        assert_eq!(rm.children(parent).len(), 1);

        rm.remove_child_of(child, parent);
        assert_eq!(rm.children(parent).len(), 0);
        assert_eq!(rm.parent(child), None);
    }

    #[test]
    fn all_edges() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let e1 = world.spawn_entity().build();
        let e2 = world.spawn_entity().build();
        let e3 = world.spawn_entity().build();

        rm.add::<LikesColor>(e1, e2);
        rm.add::<LikesColor>(e1, e3);
        rm.add::<LikesColor>(e2, e3);

        let edges = rm.all_edges::<LikesColor>();
        assert_eq!(edges.len(), 3);
        assert_eq!(rm.edge_count::<LikesColor>(), 3);
    }

    #[test]
    fn remove_all_edges_for_entity() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let e1 = world.spawn_entity().build();
        let e2 = world.spawn_entity().build();
        let e3 = world.spawn_entity().build();

        rm.add::<LikesColor>(e1, e2);
        rm.add::<LikesColor>(e2, e3);
        rm.add::<LikesColor>(e3, e1);
        rm.add::<Targets>(e1, e3);

        rm.remove_all_edges(e1);

        // All edges involving e1 should be gone.
        assert!(!rm.has::<LikesColor>(e1, e2));
        assert!(!rm.has::<LikesColor>(e3, e1));
        assert!(!rm.has::<Targets>(e1, e3));

        // Edge not involving e1 should remain.
        assert!(rm.has::<LikesColor>(e2, e3));
    }

    #[test]
    fn has_any_relations() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let e1 = world.spawn_entity().build();
        let e2 = world.spawn_entity().build();
        let e3 = world.spawn_entity().build();

        assert!(!rm.has_any_relations(e1));
        rm.add::<LikesColor>(e1, e2);
        assert!(rm.has_any_relations(e1));
        assert!(rm.has_any_relations(e2)); // e2 is a target
        assert!(!rm.has_any_relations(e3));
    }

    #[test]
    fn world_convenience_methods() {
        let mut world = World::new();
        let parent = world.spawn_entity().build();
        let child1 = world.spawn_entity().build();
        let child2 = world.spawn_entity().build();

        world.add_relation::<ChildOf>(child1, parent);
        world.add_relation::<ChildOf>(child2, parent);

        let children = world.children_of(parent);
        assert_eq!(children.len(), 2);

        assert_eq!(world.parent_of(child1), Some(parent));
    }

    #[test]
    fn world_despawn_recursive() {
        let mut world = World::new();
        let root = world.spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .build();
        let child = world.spawn_entity()
            .with(Position { x: 1.0, y: 0.0 })
            .build();

        world.add_relation::<ChildOf>(child, root);

        world.despawn_recursive(root);

        assert!(!world.is_alive(root));
        assert!(!world.is_alive(child));
    }

    #[test]
    fn multiple_relation_kinds() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let e1 = world.spawn_entity().build();
        let e2 = world.spawn_entity().build();

        rm.add::<LikesColor>(e1, e2);
        rm.add::<Targets>(e1, e2);

        assert!(rm.has::<LikesColor>(e1, e2));
        assert!(rm.has::<Targets>(e1, e2));

        rm.remove::<LikesColor>(e1, e2);
        assert!(!rm.has::<LikesColor>(e1, e2));
        assert!(rm.has::<Targets>(e1, e2)); // other relation intact
    }

    #[test]
    fn clear_all_relations() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let e1 = world.spawn_entity().build();
        let e2 = world.spawn_entity().build();

        rm.add::<LikesColor>(e1, e2);
        rm.add::<Targets>(e1, e2);
        assert_eq!(rm.active_relation_count(), 2);

        rm.clear();
        assert_eq!(rm.active_relation_count(), 0);
    }

    #[test]
    fn all_sources_and_targets() {
        let mut rm = RelationManager::new();
        let mut world = World::new();
        let e1 = world.spawn_entity().build();
        let e2 = world.spawn_entity().build();
        let e3 = world.spawn_entity().build();

        rm.add::<LikesColor>(e1, e3);
        rm.add::<LikesColor>(e2, e3);

        let sources = rm.all_sources::<LikesColor>();
        assert_eq!(sources.len(), 2);

        let targets = rm.all_targets::<LikesColor>();
        assert_eq!(targets.len(), 1);
        assert!(targets.contains(&e3));
    }

    #[test]
    fn cascading_delete_deep_hierarchy() {
        let mut rm = RelationManager::new();
        let mut world = World::new();

        // Build a 5-level hierarchy.
        let mut entities = Vec::new();
        let root = world.spawn_entity().build();
        entities.push(root);

        let mut parent = root;
        for _ in 0..4 {
            let child = world.spawn_entity().build();
            rm.add_child_of(child, parent);
            entities.push(child);
            parent = child;
        }

        // Despawn from root -- all 5 entities should be gone.
        rm.despawn_recursive(root, &mut world);

        for &e in &entities {
            assert!(!world.is_alive(e));
        }
    }

    #[test]
    fn relation_component() {
        let mut world = World::new();
        let parent = world.spawn_entity().build();
        let child = world.spawn_entity().build();

        let rel = Relation::<ChildOf>::new(parent);
        world.add_component(child, rel);

        let stored = world.get_component::<Relation<ChildOf>>(child).unwrap();
        assert_eq!(stored.target, parent);
    }
}
