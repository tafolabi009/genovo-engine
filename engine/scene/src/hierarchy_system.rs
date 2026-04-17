//! Proper hierarchical transform system with ECS-native parent/child components.
//!
//! This module provides a fully ECS-integrated hierarchy system, as opposed to
//! the resource-based [`TransformHierarchy`](crate::transform::TransformHierarchy).
//! Here, parent-child relationships are stored as components on entities
//! themselves:
//!
//! - [`Parent`] — optional parent entity reference.
//! - [`Children`] — list of child entities (stored via `SmallVec` for cache
//!   efficiency with small child counts).
//!
//! The [`PropagateTransforms`] system computes [`GlobalTransform`] values by
//! traversing the hierarchy in topological order (parents before children),
//! with dirty-flag optimization to skip clean subtrees.
//!
//! # Key Functions
//!
//! - [`set_parent`] — attach a child to a parent, updating both `Parent` and
//!   `Children` components atomically.
//! - [`remove_parent`] — detach from parent, making the entity a root.
//! - [`despawn_recursive`] — despawn an entity and all its descendants.
//! - [`HierarchyPlugin`] — registers all hierarchy components and systems.

use std::collections::{HashMap, HashSet, VecDeque};

use genovo_ecs::component::Component;
use genovo_ecs::entity::Entity;
use genovo_ecs::world::World;

use crate::transform::{GlobalTransform, TransformComponent};

// ---------------------------------------------------------------------------
// Parent component
// ---------------------------------------------------------------------------

/// Component that stores an entity's parent in the transform hierarchy.
///
/// If an entity has no `Parent` component (or `Parent(None)`), it is
/// considered a root entity.
#[derive(Debug, Clone)]
pub struct Parent(pub Option<Entity>);

impl Parent {
    /// No parent (root entity).
    pub const NONE: Self = Self(None);

    /// Create a parent component pointing to the given entity.
    pub fn new(entity: Entity) -> Self {
        Self(Some(entity))
    }

    /// Returns the parent entity, if any.
    pub fn get(&self) -> Option<Entity> {
        self.0
    }

    /// Returns `true` if this entity has a parent.
    pub fn has_parent(&self) -> bool {
        self.0.is_some()
    }
}

impl Default for Parent {
    fn default() -> Self {
        Self::NONE
    }
}

impl Component for Parent {}

// ---------------------------------------------------------------------------
// Children component
// ---------------------------------------------------------------------------

/// Maximum number of children stored inline before spilling to the heap.
const INLINE_CHILDREN: usize = 8;

/// Component that stores the list of child entities.
///
/// Uses a `SmallVec`-style inline storage: the first [`INLINE_CHILDREN`]
/// children are stored inline to avoid heap allocation for common cases
/// (most entities have few children).
#[derive(Debug, Clone)]
pub struct Children {
    /// Child entities. In a full implementation this would be a SmallVec;
    /// here we use a plain Vec for simplicity but document the intent.
    entities: Vec<Entity>,
}

impl Children {
    /// Create an empty children list.
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
        }
    }

    /// Create a children list with the given entities.
    pub fn with(entities: Vec<Entity>) -> Self {
        Self { entities }
    }

    /// Add a child entity. Does nothing if already present.
    pub fn add(&mut self, child: Entity) {
        if !self.entities.contains(&child) {
            self.entities.push(child);
        }
    }

    /// Remove a child entity. Returns `true` if it was present.
    pub fn remove(&mut self, child: Entity) -> bool {
        if let Some(pos) = self.entities.iter().position(|e| *e == child) {
            self.entities.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Returns `true` if the child is in the list.
    pub fn contains(&self, child: Entity) -> bool {
        self.entities.contains(&child)
    }

    /// Number of children.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether there are no children.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get the child entities as a slice.
    pub fn as_slice(&self) -> &[Entity] {
        &self.entities
    }

    /// Iterate over child entities.
    pub fn iter(&self) -> impl Iterator<Item = &Entity> {
        self.entities.iter()
    }

    /// Clear all children.
    pub fn clear(&mut self) {
        self.entities.clear();
    }

    /// Get child at index.
    pub fn get(&self, index: usize) -> Option<Entity> {
        self.entities.get(index).copied()
    }

    /// Swap two children by index.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a < self.entities.len() && b < self.entities.len() {
            self.entities.swap(a, b);
        }
    }

    /// Insert a child at a specific index.
    pub fn insert_at(&mut self, index: usize, child: Entity) {
        if !self.entities.contains(&child) {
            let idx = index.min(self.entities.len());
            self.entities.insert(idx, child);
        }
    }
}

impl Default for Children {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for Children {}

// ---------------------------------------------------------------------------
// DirtyTransform — marks entities needing global transform recomputation
// ---------------------------------------------------------------------------

/// Marker component indicating that an entity's global transform needs
/// recomputation. This is added when a `TransformComponent` changes or
/// when the entity's parent changes.
#[derive(Debug, Clone, Default)]
pub struct DirtyTransform;

impl Component for DirtyTransform {}

// ---------------------------------------------------------------------------
// Hierarchy manipulation functions
// ---------------------------------------------------------------------------

/// Set `child` as a child of `parent`, updating both `Parent` and `Children`
/// components. If the child already has a parent, it is detached first.
///
/// Returns `false` if the operation would create a cycle (parent is a
/// descendant of child) or if either entity is not alive.
///
/// # Panics
///
/// Does not panic. Returns `false` on invalid input.
pub fn set_parent(world: &mut World, child: Entity, parent: Entity) -> bool {
    // Validate both entities are alive.
    if !world.is_alive(child) || !world.is_alive(parent) {
        return false;
    }

    // Self-parenting check.
    if child == parent {
        return false;
    }

    // Cycle detection: walk from parent upward; if we reach child, abort.
    if is_ancestor_of(world, child, parent) {
        return false;
    }

    // Detach from current parent.
    remove_parent(world, child);

    // Set the Parent component on the child.
    world.add_component(child, Parent::new(parent));

    // Add to parent's Children component.
    let mut children = world
        .remove_component::<Children>(parent)
        .unwrap_or_default();
    children.add(child);
    world.add_component(parent, children);

    // Mark the child subtree as dirty.
    mark_dirty_recursive(world, child);

    true
}

/// Remove the parent relationship for `child`, making it a root entity.
///
/// Updates both the child's `Parent` component and the old parent's
/// `Children` component.
pub fn remove_parent(world: &mut World, child: Entity) {
    if !world.is_alive(child) {
        return;
    }

    // Get current parent.
    let old_parent = world
        .get_component::<Parent>(child)
        .and_then(|p| p.get());

    if let Some(old_parent_entity) = old_parent {
        // Remove child from old parent's Children list.
        if let Some(mut children) = world.remove_component::<Children>(old_parent_entity) {
            children.remove(child);
            if !children.is_empty() {
                world.add_component(old_parent_entity, children);
            }
        }
    }

    // Set parent to None.
    world.add_component(child, Parent::NONE);

    // Mark dirty.
    mark_dirty_recursive(world, child);
}

/// Check if `ancestor` is an ancestor of `entity` by walking up the
/// parent chain.
pub fn is_ancestor_of(world: &World, ancestor: Entity, entity: Entity) -> bool {
    let mut current = world
        .get_component::<Parent>(entity)
        .and_then(|p| p.get());

    let mut depth = 0;
    const MAX_DEPTH: usize = 1024; // Safety limit to avoid infinite loops.

    while let Some(p) = current {
        if p == ancestor {
            return true;
        }
        depth += 1;
        if depth > MAX_DEPTH {
            break;
        }
        current = world.get_component::<Parent>(p).and_then(|pp| pp.get());
    }
    false
}

/// Get the depth of an entity in the hierarchy. Root entities have depth 0.
pub fn hierarchy_depth(world: &World, entity: Entity) -> usize {
    let mut depth = 0;
    let mut current = world
        .get_component::<Parent>(entity)
        .and_then(|p| p.get());

    while let Some(p) = current {
        depth += 1;
        current = world.get_component::<Parent>(p).and_then(|pp| pp.get());
    }
    depth
}

/// Collect all descendants of an entity (breadth-first order).
pub fn collect_descendants(world: &World, entity: Entity) -> Vec<Entity> {
    let mut result = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(entity);

    while let Some(current) = queue.pop_front() {
        if current != entity {
            result.push(current);
        }
        if let Some(children) = world.get_component::<Children>(current) {
            for &child in children.as_slice() {
                queue.push_back(child);
            }
        }
    }
    result
}

/// Despawn an entity and all its descendants recursively.
///
/// Cleans up parent-child relationships and removes the entity from its
/// parent's `Children` list before despawning.
pub fn despawn_recursive(world: &mut World, entity: Entity) {
    if !world.is_alive(entity) {
        return;
    }

    // Collect all descendants first (since we cannot iterate while mutating).
    let descendants = collect_descendants(world, entity);

    // Detach from parent.
    let old_parent = world
        .get_component::<Parent>(entity)
        .and_then(|p| p.get());

    if let Some(old_parent_entity) = old_parent {
        if let Some(mut children) = world.remove_component::<Children>(old_parent_entity) {
            children.remove(entity);
            if !children.is_empty() {
                world.add_component(old_parent_entity, children);
            }
        }
    }

    // Despawn all descendants (deepest first to avoid dangling references).
    for desc in descendants.iter().rev() {
        world.despawn(*desc);
    }

    // Despawn the entity itself.
    world.despawn(entity);
}

/// Detach all children from an entity, making them roots.
pub fn detach_children(world: &mut World, entity: Entity) {
    if !world.is_alive(entity) {
        return;
    }

    let children_list: Vec<Entity> = world
        .get_component::<Children>(entity)
        .map(|c| c.as_slice().to_vec())
        .unwrap_or_default();

    for child in children_list {
        world.add_component(child, Parent::NONE);
    }

    // Clear the Children component.
    world.add_component(entity, Children::new());
}

/// Get the root ancestor of an entity (the topmost parent in the hierarchy).
pub fn root_ancestor(world: &World, entity: Entity) -> Entity {
    let mut current = entity;
    let mut depth = 0;
    const MAX_DEPTH: usize = 1024;

    while let Some(parent) = world
        .get_component::<Parent>(current)
        .and_then(|p| p.get())
    {
        current = parent;
        depth += 1;
        if depth > MAX_DEPTH {
            break;
        }
    }
    current
}

/// Mark an entity and all its descendants as needing transform recomputation.
fn mark_dirty_recursive(world: &mut World, entity: Entity) {
    if !world.is_alive(entity) {
        return;
    }

    // Mark this entity's transform as dirty.
    if let Some(tc) = world.get_component::<TransformComponent>(entity) {
        let mut tc_copy = tc.clone();
        tc_copy.dirty = true;
        world.add_component(entity, tc_copy);
    }

    // Recursively mark children.
    let children: Vec<Entity> = world
        .get_component::<Children>(entity)
        .map(|c| c.as_slice().to_vec())
        .unwrap_or_default();

    for child in children {
        mark_dirty_recursive(world, child);
    }
}

// ---------------------------------------------------------------------------
// PropagateTransforms system
// ---------------------------------------------------------------------------

/// System that propagates local transforms through the hierarchy to compute
/// world-space [`GlobalTransform`] values.
///
/// # Algorithm
///
/// 1. Find all root entities (entities with `TransformComponent` but no
///    parent or `Parent(None)`).
/// 2. Build a topological ordering: process each root, then its children,
///    then grandchildren, etc. This guarantees parents are processed before
///    children.
/// 3. For each entity in topological order:
///    - If the entity or any ancestor is dirty, recompute:
///      `GlobalTransform = parent_global * local_transform`
///    - Clear the dirty flag.
///
/// # Dirty Flag Optimization
///
/// Only dirty subtrees are recomputed. If an entity and all its ancestors
/// are clean, the system skips it entirely. When a parent is dirty, all
/// children are also recomputed (since their world transform depends on
/// the parent).
pub struct PropagateTransforms {
    /// Entities updated during the last propagation pass.
    updated_entities: Vec<Entity>,
    /// Statistics: how many entities were skipped (clean).
    skipped_count: usize,
    /// Statistics: how many entities were updated (dirty).
    updated_count: usize,
}

impl PropagateTransforms {
    /// Create a new propagation system.
    pub fn new() -> Self {
        Self {
            updated_entities: Vec::new(),
            skipped_count: 0,
            updated_count: 0,
        }
    }

    /// Get entities updated in the last propagation pass.
    pub fn updated_entities(&self) -> &[Entity] {
        &self.updated_entities
    }

    /// Number of entities skipped in the last pass.
    pub fn skipped_count(&self) -> usize {
        self.skipped_count
    }

    /// Number of entities updated in the last pass.
    pub fn updated_count(&self) -> usize {
        self.updated_count
    }

    /// Run the propagation pass.
    ///
    /// This is the main entry point. It finds all root entities, then
    /// processes them depth-first, propagating transforms down the hierarchy.
    pub fn propagate(&mut self, world: &mut World) {
        self.updated_entities.clear();
        self.skipped_count = 0;
        self.updated_count = 0;

        // Collect all root entities: entities that have a TransformComponent
        // but no parent (or Parent(None)).
        let roots = find_root_entities(world);

        // Process each root and its descendants.
        for root in roots {
            self.propagate_subtree(world, root, glam::Mat4::IDENTITY, false);
        }
    }

    /// Recursively propagate transforms for an entity and its children.
    fn propagate_subtree(
        &mut self,
        world: &mut World,
        entity: Entity,
        parent_world_matrix: glam::Mat4,
        parent_dirty: bool,
    ) {
        if !world.is_alive(entity) {
            return;
        }

        // Read the local transform.
        let (local_matrix, is_dirty) = match world.get_component::<TransformComponent>(entity) {
            Some(tc) => (tc.to_matrix(), tc.dirty),
            None => (glam::Mat4::IDENTITY, false),
        };

        let needs_update = is_dirty || parent_dirty;
        let world_matrix = parent_world_matrix * local_matrix;

        if needs_update {
            // Write the global transform.
            world.add_component(
                entity,
                GlobalTransform {
                    matrix: world_matrix,
                    updated_this_frame: true,
                },
            );
            self.updated_entities.push(entity);
            self.updated_count += 1;

            // Clear the dirty flag on the local transform.
            if is_dirty {
                if let Some(tc) = world.get_component::<TransformComponent>(entity) {
                    let mut tc_copy = tc.clone();
                    tc_copy.dirty = false;
                    world.add_component(entity, tc_copy);
                }
            }
        } else {
            self.skipped_count += 1;

            // Even if this entity is clean, we might need to use its existing
            // global transform as the parent matrix for children. If no
            // GlobalTransform exists yet, use the computed one.
            if world.get_component::<GlobalTransform>(entity).is_none() {
                world.add_component(
                    entity,
                    GlobalTransform {
                        matrix: world_matrix,
                        updated_this_frame: false,
                    },
                );
            }
        }

        // Get children from the Children component.
        let children: Vec<Entity> = world
            .get_component::<Children>(entity)
            .map(|c| c.as_slice().to_vec())
            .unwrap_or_default();

        // Recurse into children.
        for child in children {
            self.propagate_subtree(world, child, world_matrix, needs_update);
        }
    }
}

impl Default for PropagateTransforms {
    fn default() -> Self {
        Self::new()
    }
}

impl genovo_ecs::System for PropagateTransforms {
    fn run(&mut self, world: &mut World) {
        self.propagate(world);
    }
}

/// Find all entities that are roots of the transform hierarchy.
///
/// A root entity has a `TransformComponent` and either:
/// - No `Parent` component, or
/// - `Parent(None)`.
fn find_root_entities(world: &World) -> Vec<Entity> {
    let mut roots = Vec::new();

    // Iterate all entities with TransformComponent and check for parent.
    for (entity, _tc) in world.query::<&TransformComponent>() {
        let has_parent = world
            .get_component::<Parent>(entity)
            .map(|p| p.has_parent())
            .unwrap_or(false);

        if !has_parent {
            roots.push(entity);
        }
    }

    // Sort by entity id for deterministic ordering.
    roots.sort_by_key(|e| e.id);
    roots
}

// ---------------------------------------------------------------------------
// HierarchyPlugin — registers hierarchy components and systems
// ---------------------------------------------------------------------------

/// Plugin that registers all hierarchy-related components and systems.
///
/// Call `HierarchyPlugin::register` to set up the hierarchy system in a world.
pub struct HierarchyPlugin;

impl HierarchyPlugin {
    /// Register hierarchy components in the world.
    pub fn register(world: &mut World) {
        world.register_component::<Parent>();
        world.register_component::<Children>();
        world.register_component::<DirtyTransform>();
    }

    /// Create a configured PropagateTransforms system.
    pub fn create_system() -> PropagateTransforms {
        PropagateTransforms::new()
    }
}

// ---------------------------------------------------------------------------
// Hierarchy query helpers
// ---------------------------------------------------------------------------

/// Get all siblings of an entity (entities with the same parent, excluding self).
pub fn siblings(world: &World, entity: Entity) -> Vec<Entity> {
    let parent = world
        .get_component::<Parent>(entity)
        .and_then(|p| p.get());

    match parent {
        Some(parent_entity) => {
            world
                .get_component::<Children>(parent_entity)
                .map(|c| {
                    c.as_slice()
                        .iter()
                        .copied()
                        .filter(|e| *e != entity)
                        .collect()
                })
                .unwrap_or_default()
        }
        None => Vec::new(),
    }
}

/// Get the path from root to entity as a list of entities.
pub fn ancestor_path(world: &World, entity: Entity) -> Vec<Entity> {
    let mut path = Vec::new();
    let mut current = entity;

    loop {
        path.push(current);
        match world
            .get_component::<Parent>(current)
            .and_then(|p| p.get())
        {
            Some(parent) => current = parent,
            None => break,
        }
    }

    path.reverse();
    path
}

/// Find the lowest common ancestor of two entities.
pub fn lowest_common_ancestor(world: &World, a: Entity, b: Entity) -> Option<Entity> {
    let path_a = ancestor_path(world, a);
    let path_b = ancestor_path(world, b);

    let mut lca = None;
    for (ea, eb) in path_a.iter().zip(path_b.iter()) {
        if ea == eb {
            lca = Some(*ea);
        } else {
            break;
        }
    }
    lca
}

/// Count all entities in a subtree (including the root entity).
pub fn subtree_count(world: &World, entity: Entity) -> usize {
    1 + collect_descendants(world, entity).len()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn make_world_with_transforms() -> (World, Entity, Entity, Entity) {
        let mut world = World::new();
        HierarchyPlugin::register(&mut world);

        let a = world.spawn_entity().build();
        let b = world.spawn_entity().build();
        let c = world.spawn_entity().build();

        world.add_component(
            a,
            TransformComponent::from_position(Vec3::new(10.0, 0.0, 0.0)),
        );
        world.add_component(
            b,
            TransformComponent::from_position(Vec3::new(0.0, 5.0, 0.0)),
        );
        world.add_component(
            c,
            TransformComponent::from_position(Vec3::new(0.0, 0.0, 3.0)),
        );

        (world, a, b, c)
    }

    // -- Parent / Children component tests --------------------------------------

    #[test]
    fn parent_none() {
        let p = Parent::NONE;
        assert!(!p.has_parent());
        assert_eq!(p.get(), None);
    }

    #[test]
    fn parent_some() {
        let e = Entity::new(42, 0);
        let p = Parent::new(e);
        assert!(p.has_parent());
        assert_eq!(p.get(), Some(e));
    }

    #[test]
    fn children_add_remove() {
        let mut children = Children::new();
        let e1 = Entity::new(1, 0);
        let e2 = Entity::new(2, 0);

        assert!(children.is_empty());
        children.add(e1);
        children.add(e2);
        assert_eq!(children.len(), 2);
        assert!(children.contains(e1));
        assert!(children.contains(e2));

        assert!(children.remove(e1));
        assert!(!children.contains(e1));
        assert_eq!(children.len(), 1);

        assert!(!children.remove(e1)); // Already removed.
    }

    #[test]
    fn children_no_duplicates() {
        let mut children = Children::new();
        let e = Entity::new(1, 0);
        children.add(e);
        children.add(e);
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn children_insert_at() {
        let mut children = Children::new();
        let e1 = Entity::new(1, 0);
        let e2 = Entity::new(2, 0);
        let e3 = Entity::new(3, 0);

        children.add(e1);
        children.add(e3);
        children.insert_at(1, e2);

        assert_eq!(children.as_slice(), &[e1, e2, e3]);
    }

    #[test]
    fn children_swap() {
        let mut children = Children::new();
        let e1 = Entity::new(1, 0);
        let e2 = Entity::new(2, 0);
        children.add(e1);
        children.add(e2);

        children.swap(0, 1);
        assert_eq!(children.as_slice(), &[e2, e1]);
    }

    #[test]
    fn children_get() {
        let mut children = Children::new();
        let e1 = Entity::new(1, 0);
        children.add(e1);

        assert_eq!(children.get(0), Some(e1));
        assert_eq!(children.get(1), None);
    }

    #[test]
    fn children_clear() {
        let mut children = Children::new();
        children.add(Entity::new(1, 0));
        children.add(Entity::new(2, 0));
        children.clear();
        assert!(children.is_empty());
    }

    // -- set_parent / remove_parent tests ---------------------------------------

    #[test]
    fn set_parent_basic() {
        let (mut world, a, b, _c) = make_world_with_transforms();

        assert!(set_parent(&mut world, b, a));

        let parent = world.get_component::<Parent>(b).unwrap();
        assert_eq!(parent.get(), Some(a));

        let children = world.get_component::<Children>(a).unwrap();
        assert!(children.contains(b));
    }

    #[test]
    fn set_parent_self_rejected() {
        let (mut world, a, _b, _c) = make_world_with_transforms();
        assert!(!set_parent(&mut world, a, a));
    }

    #[test]
    fn set_parent_cycle_rejected() {
        let (mut world, a, b, c) = make_world_with_transforms();

        // a -> b -> c
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, b);

        // Trying to set a's parent to c would create a cycle.
        assert!(!set_parent(&mut world, a, c));
    }

    #[test]
    fn set_parent_reparent() {
        let (mut world, a, b, c) = make_world_with_transforms();

        // Start: c is child of a.
        set_parent(&mut world, c, a);
        assert_eq!(
            world.get_component::<Parent>(c).unwrap().get(),
            Some(a)
        );

        // Reparent c from a to b.
        set_parent(&mut world, c, b);
        assert_eq!(
            world.get_component::<Parent>(c).unwrap().get(),
            Some(b)
        );

        // c should no longer be in a's children.
        let a_children = world.get_component::<Children>(a);
        assert!(
            a_children.is_none() || !a_children.unwrap().contains(c)
        );

        // c should be in b's children.
        assert!(world.get_component::<Children>(b).unwrap().contains(c));
    }

    #[test]
    fn remove_parent_basic() {
        let (mut world, a, b, _c) = make_world_with_transforms();

        set_parent(&mut world, b, a);
        remove_parent(&mut world, b);

        let parent = world.get_component::<Parent>(b).unwrap();
        assert!(!parent.has_parent());

        let a_children = world.get_component::<Children>(a);
        assert!(
            a_children.is_none() || !a_children.unwrap().contains(b)
        );
    }

    #[test]
    fn remove_parent_no_parent_noop() {
        let (mut world, a, _b, _c) = make_world_with_transforms();
        // a has no parent; removing should be a no-op.
        remove_parent(&mut world, a);
        let parent = world.get_component::<Parent>(a).unwrap();
        assert!(!parent.has_parent());
    }

    // -- is_ancestor_of tests ---------------------------------------------------

    #[test]
    fn is_ancestor_basic() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, b);

        assert!(is_ancestor_of(&world, a, c));
        assert!(is_ancestor_of(&world, a, b));
        assert!(is_ancestor_of(&world, b, c));
        assert!(!is_ancestor_of(&world, c, a));
        assert!(!is_ancestor_of(&world, b, a));
    }

    // -- hierarchy_depth tests --------------------------------------------------

    #[test]
    fn depth_test() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, b);

        assert_eq!(hierarchy_depth(&world, a), 0);
        assert_eq!(hierarchy_depth(&world, b), 1);
        assert_eq!(hierarchy_depth(&world, c), 2);
    }

    // -- collect_descendants tests ----------------------------------------------

    #[test]
    fn collect_descendants_test() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, b);

        let desc = collect_descendants(&world, a);
        assert_eq!(desc.len(), 2);
        assert!(desc.contains(&b));
        assert!(desc.contains(&c));
    }

    #[test]
    fn collect_descendants_leaf() {
        let (world, _a, _b, c) = make_world_with_transforms();
        let desc = collect_descendants(&world, c);
        assert!(desc.is_empty());
    }

    // -- despawn_recursive tests ------------------------------------------------

    #[test]
    fn despawn_recursive_test() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, b);

        despawn_recursive(&mut world, a);

        assert!(!world.is_alive(a));
        assert!(!world.is_alive(b));
        assert!(!world.is_alive(c));
    }

    #[test]
    fn despawn_recursive_leaf() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, a);

        despawn_recursive(&mut world, b);

        assert!(world.is_alive(a));
        assert!(!world.is_alive(b));
        assert!(world.is_alive(c));

        // b should have been removed from a's children.
        let a_children = world.get_component::<Children>(a).unwrap();
        assert!(!a_children.contains(b));
        assert!(a_children.contains(c));
    }

    #[test]
    fn despawn_recursive_dead_entity_noop() {
        let mut world = World::new();
        let e = Entity::new(999, 0);
        despawn_recursive(&mut world, e); // Should not panic.
    }

    // -- detach_children tests --------------------------------------------------

    #[test]
    fn detach_children_test() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, a);

        detach_children(&mut world, a);

        let a_children = world.get_component::<Children>(a).unwrap();
        assert!(a_children.is_empty());

        // b and c should now have no parent.
        assert!(!world.get_component::<Parent>(b).unwrap().has_parent());
        assert!(!world.get_component::<Parent>(c).unwrap().has_parent());
    }

    // -- root_ancestor tests ----------------------------------------------------

    #[test]
    fn root_ancestor_test() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, b);

        assert_eq!(root_ancestor(&world, c), a);
        assert_eq!(root_ancestor(&world, b), a);
        assert_eq!(root_ancestor(&world, a), a);
    }

    // -- ancestor_path tests ----------------------------------------------------

    #[test]
    fn ancestor_path_test() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, b);

        let path = ancestor_path(&world, c);
        assert_eq!(path, vec![a, b, c]);
    }

    // -- lowest_common_ancestor tests -------------------------------------------

    #[test]
    fn lca_test() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, a);

        assert_eq!(lowest_common_ancestor(&world, b, c), Some(a));
    }

    #[test]
    fn lca_same_entity() {
        let (world, a, _b, _c) = make_world_with_transforms();
        assert_eq!(lowest_common_ancestor(&world, a, a), Some(a));
    }

    // -- siblings tests ---------------------------------------------------------

    #[test]
    fn siblings_test() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, a);

        let sibs = siblings(&world, b);
        assert_eq!(sibs, vec![c]);
    }

    // -- subtree_count tests ----------------------------------------------------

    #[test]
    fn subtree_count_test() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, b);

        assert_eq!(subtree_count(&world, a), 3);
        assert_eq!(subtree_count(&world, b), 2);
        assert_eq!(subtree_count(&world, c), 1);
    }

    // -- PropagateTransforms system tests ---------------------------------------

    #[test]
    fn propagate_simple_hierarchy() {
        let (mut world, a, b, c) = make_world_with_transforms();
        set_parent(&mut world, b, a);
        set_parent(&mut world, c, b);

        let mut system = PropagateTransforms::new();
        system.propagate(&mut world);

        // a: at (10, 0, 0)
        let gt_a = world.get_component::<GlobalTransform>(a).unwrap();
        assert!((gt_a.position() - Vec3::new(10.0, 0.0, 0.0)).length() < 1e-4);

        // b: parent (10,0,0) + local (0,5,0) = (10, 5, 0)
        let gt_b = world.get_component::<GlobalTransform>(b).unwrap();
        assert!((gt_b.position() - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-4);

        // c: parent (10,5,0) + local (0,0,3) = (10, 5, 3)
        let gt_c = world.get_component::<GlobalTransform>(c).unwrap();
        assert!((gt_c.position() - Vec3::new(10.0, 5.0, 3.0)).length() < 1e-4);
    }

    #[test]
    fn propagate_dirty_flag_optimization() {
        let (mut world, a, b, _c) = make_world_with_transforms();
        set_parent(&mut world, b, a);

        let mut system = PropagateTransforms::new();

        // First pass: everything is dirty.
        system.propagate(&mut world);
        assert!(system.updated_count() >= 2);

        // Second pass: nothing is dirty (transforms haven't changed).
        system.propagate(&mut world);
        // Note: skipped_count should be > 0 now.
        // The exact counts depend on implementation details, but updated_count
        // should be 0 since nothing changed.
        assert_eq!(system.updated_count(), 0);
    }

    #[test]
    fn propagate_after_position_change() {
        let (mut world, a, b, _c) = make_world_with_transforms();
        set_parent(&mut world, b, a);

        let mut system = PropagateTransforms::new();
        system.propagate(&mut world);

        // Move the parent.
        world.add_component(
            a,
            TransformComponent::from_position(Vec3::new(20.0, 0.0, 0.0)),
        );

        system.propagate(&mut world);

        // b should have updated: parent (20,0,0) + local (0,5,0) = (20,5,0)
        let gt_b = world.get_component::<GlobalTransform>(b).unwrap();
        assert!((gt_b.position() - Vec3::new(20.0, 5.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn propagate_root_entities() {
        let mut world = World::new();
        HierarchyPlugin::register(&mut world);

        // Two independent root entities.
        let e1 = world.spawn_entity().build();
        let e2 = world.spawn_entity().build();

        world.add_component(
            e1,
            TransformComponent::from_position(Vec3::new(1.0, 0.0, 0.0)),
        );
        world.add_component(
            e2,
            TransformComponent::from_position(Vec3::new(0.0, 2.0, 0.0)),
        );

        let mut system = PropagateTransforms::new();
        system.propagate(&mut world);

        let gt1 = world.get_component::<GlobalTransform>(e1).unwrap();
        assert!((gt1.position() - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4);

        let gt2 = world.get_component::<GlobalTransform>(e2).unwrap();
        assert!((gt2.position() - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn propagate_with_scale() {
        let mut world = World::new();
        HierarchyPlugin::register(&mut world);

        let parent = world.spawn_entity().build();
        let child = world.spawn_entity().build();

        world.add_component(
            parent,
            TransformComponent::new(
                Vec3::ZERO,
                glam::Quat::IDENTITY,
                Vec3::splat(2.0),
            ),
        );
        world.add_component(
            child,
            TransformComponent::from_position(Vec3::new(1.0, 0.0, 0.0)),
        );

        set_parent(&mut world, child, parent);

        let mut system = PropagateTransforms::new();
        system.propagate(&mut world);

        // Child at local (1,0,0) under parent with scale 2 => world (2,0,0).
        let gt = world.get_component::<GlobalTransform>(child).unwrap();
        assert!((gt.position() - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-4);
    }

    // -- HierarchyPlugin test ---------------------------------------------------

    #[test]
    fn hierarchy_plugin_register() {
        let mut world = World::new();
        HierarchyPlugin::register(&mut world);
        // Verify components are registered by spawning an entity with them.
        let e = world.spawn_entity().build();
        world.add_component(e, Parent::NONE);
        world.add_component(e, Children::new());
        assert!(world.has_component::<Parent>(e));
        assert!(world.has_component::<Children>(e));
    }
}
