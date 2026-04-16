//! Scene graph node hierarchy.
//!
//! A scene graph is a tree of [`SceneNode`]s managed by a [`SceneNodeTree`].
//! Each node carries a local transform relative to its parent, visibility flags,
//! optional ECS entity linkage, and a set of user-defined tags. The tree uses
//! arena-style slot allocation with a free-list for O(1) node creation/recycling
//! and supports depth-first, breadth-first, ancestor, and descendant traversals.

use std::collections::VecDeque;
use std::fmt;

use genovo_ecs::Entity;
use glam::{Mat4, Quat, Vec3};
// smallvec available but not currently needed for the public API.
// use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// NodeId
// ---------------------------------------------------------------------------

/// Opaque handle to a node in the scene graph. Internally an index into the
/// arena-allocated node pool stored in [`SceneNodeTree`].
///
/// A `NodeId` is only valid for the tree that created it. Using it with a
/// different tree is a logic error (but will not cause undefined behavior).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    /// Sentinel value meaning "no node".
    pub const NONE: Self = Self(u32::MAX);

    /// Whether this is the sentinel value.
    #[inline]
    pub fn is_none(&self) -> bool {
        self.0 == u32::MAX
    }

    /// Whether this is a valid (non-sentinel) value.
    #[inline]
    pub fn is_some(&self) -> bool {
        self.0 != u32::MAX
    }

    /// Return the raw index. Useful for external data structures that need to
    /// key on node identity.
    #[inline]
    pub fn index(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            write!(f, "NodeId(NONE)")
        } else {
            write!(f, "NodeId({})", self.0)
        }
    }
}

// ---------------------------------------------------------------------------
// SceneNode
// ---------------------------------------------------------------------------

/// A single node in the scene graph tree.
///
/// Each node has:
/// - A human-readable **name** (not required to be unique).
/// - A **local transform** (position, rotation, scale) relative to its parent.
/// - A cached **world transform** matrix recomputed during
///   [`SceneNodeTree::propagate_transforms`].
/// - An optional link to an ECS [`Entity`].
/// - A set of string **tags** for grouping/filtering.
/// - Visibility and dirty flags.
pub struct SceneNode {
    /// Unique identifier within the owning tree.
    pub id: NodeId,
    /// Human-readable name.
    pub name: String,

    // -- Hierarchy ----------------------------------------------------------
    /// Parent node, or `None` for root nodes.
    pub parent: Option<NodeId>,
    /// Ordered list of child node ids.
    pub children: Vec<NodeId>,

    // -- Transform ----------------------------------------------------------
    /// Local-space position relative to parent.
    pub local_position: Vec3,
    /// Local-space rotation relative to parent.
    pub local_rotation: Quat,
    /// Local-space scale relative to parent.
    pub local_scale: Vec3,

    /// Cached world-space transform matrix. Recomputed during
    /// [`SceneNodeTree::propagate_transforms`].
    pub world_transform: Mat4,

    // -- Flags --------------------------------------------------------------
    /// Whether this node (and its subtree) is visible.
    pub visible: bool,
    /// Whether the world transform is out of date.
    pub dirty: bool,

    // -- ECS linkage --------------------------------------------------------
    /// Optional ECS entity linked to this node.
    pub entity: Option<Entity>,

    // -- Tags ---------------------------------------------------------------
    /// User-defined tags for grouping and filtering.
    pub tags: Vec<String>,
}

impl SceneNode {
    /// Create a new node with default identity transform.
    pub fn new(id: NodeId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            parent: None,
            children: Vec::new(),
            local_position: Vec3::ZERO,
            local_rotation: Quat::IDENTITY,
            local_scale: Vec3::ONE,
            world_transform: Mat4::IDENTITY,
            visible: true,
            dirty: true,
            entity: None,
            tags: Vec::new(),
        }
    }

    /// Compute the local 4x4 transform matrix from position, rotation, and scale.
    #[inline]
    pub fn local_transform(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.local_scale,
            self.local_rotation,
            self.local_position,
        )
    }

    /// Set position, rotation, and scale from a `genovo_core::Transform`.
    pub fn set_local_transform(&mut self, t: &genovo_core::Transform) {
        self.local_position = t.position;
        self.local_rotation = t.rotation;
        self.local_scale = t.scale;
        self.dirty = true;
    }

    /// Extract the current local transform as a `genovo_core::Transform`.
    pub fn get_local_transform(&self) -> genovo_core::Transform {
        genovo_core::Transform::new(self.local_position, self.local_rotation, self.local_scale)
    }

    /// Set the local position and mark the node dirty.
    pub fn set_position(&mut self, position: Vec3) {
        self.local_position = position;
        self.dirty = true;
    }

    /// Set the local rotation and mark the node dirty.
    pub fn set_rotation(&mut self, rotation: Quat) {
        self.local_rotation = rotation;
        self.dirty = true;
    }

    /// Set the local scale and mark the node dirty.
    pub fn set_scale(&mut self, scale: Vec3) {
        self.local_scale = scale;
        self.dirty = true;
    }

    /// Returns `true` if this is a root node (no parent).
    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Returns `true` if this node has children.
    #[inline]
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Returns the number of direct children.
    #[inline]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Add a tag. Does nothing if the tag already exists.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Remove a tag. Returns `true` if the tag was present.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        if let Some(pos) = self.tags.iter().position(|t| t == tag) {
            self.tags.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Returns `true` if the node has the given tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Extract the world-space position from the cached world transform.
    #[inline]
    pub fn world_position(&self) -> Vec3 {
        self.world_transform.w_axis.truncate()
    }

    /// Extract the world-space rotation from the cached world transform
    /// (assumes no skew/shear).
    pub fn world_rotation(&self) -> Quat {
        let (_, rotation, _) = self.world_transform.to_scale_rotation_translation();
        rotation
    }

    /// Extract the world-space scale from the cached world transform
    /// (assumes no skew/shear).
    pub fn world_scale(&self) -> Vec3 {
        let (scale, _, _) = self.world_transform.to_scale_rotation_translation();
        scale
    }

    /// Returns the depth of this node in the tree. Root nodes have depth 0.
    /// This is a property stored externally and computed during traversal;
    /// here we provide it as a helper that requires the tree reference.
    pub fn depth_in(&self, tree: &SceneNodeTree) -> usize {
        let mut depth = 0;
        let mut current = self.parent;
        while let Some(pid) = current {
            depth += 1;
            current = tree.get(pid).and_then(|n| n.parent);
        }
        depth
    }
}

impl fmt::Debug for SceneNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SceneNode")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("parent", &self.parent)
            .field("children", &self.children.len())
            .field("visible", &self.visible)
            .field("dirty", &self.dirty)
            .field("entity", &self.entity)
            .field("tags", &self.tags)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SceneNodeTree
// ---------------------------------------------------------------------------

/// Arena-allocated scene node tree with O(1) node access by [`NodeId`].
///
/// Nodes are stored in a slot-based `Vec<Option<SceneNode>>` with a free-list
/// for recycling destroyed slots. Multiple root nodes are supported.
pub struct SceneNodeTree {
    /// Slot-based node storage. `None` means the slot is free.
    nodes: Vec<Option<SceneNode>>,
    /// Root node ids (nodes with no parent).
    root_nodes: Vec<NodeId>,
    /// Free list of recyclable slot indices for O(1) allocation.
    free_list: Vec<NodeId>,
    /// Total number of alive nodes.
    alive_count: usize,
}

impl SceneNodeTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root_nodes: Vec::new(),
            free_list: Vec::new(),
            alive_count: 0,
        }
    }

    /// Create an empty tree with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            root_nodes: Vec::new(),
            free_list: Vec::new(),
            alive_count: 0,
        }
    }

    // -- Node creation ------------------------------------------------------

    /// Create a new node with the given name. The node starts as a root.
    /// Returns the node's id.
    pub fn create_node(&mut self, name: &str) -> NodeId {
        let id = self.allocate_slot(name);
        self.root_nodes.push(id);
        id
    }

    /// Add a new root node and return its id. Alias for `create_node`.
    pub fn add_root(&mut self, name: impl Into<String>) -> NodeId {
        let name_str: String = name.into();
        self.create_node(&name_str)
    }

    /// Add a child node to an existing parent. The child is created and
    /// appended to the parent's children list.
    pub fn add_child(&mut self, parent: NodeId, name: impl Into<String>) -> NodeId {
        let name_str: String = name.into();
        let child_id = self.allocate_slot(&name_str);

        // Link child to parent.
        if let Some(child_node) = self.get_mut(child_id) {
            child_node.parent = Some(parent);
        }
        if let Some(parent_node) = self.get_mut(parent) {
            parent_node.children.push(child_id);
        }

        child_id
    }

    // -- Node destruction ---------------------------------------------------

    /// Destroy a node and **all of its descendants**. Removes the node from
    /// its parent's children list and from the root list if applicable.
    pub fn destroy_node(&mut self, id: NodeId) {
        if id.is_none() || !self.is_alive(id) {
            return;
        }

        // Collect all descendants depth-first.
        let descendants = self.collect_descendants(id);

        // Remove from parent's child list.
        if let Some(parent_id) = self.get(id).and_then(|n| n.parent) {
            if let Some(parent_node) = self.get_mut(parent_id) {
                parent_node.children.retain(|c| *c != id);
            }
        }

        // Remove from root list if applicable.
        self.root_nodes.retain(|r| *r != id);

        // Destroy the node itself and all descendants.
        self.free_slot(id);
        for desc_id in descendants {
            // Also remove from root list (shouldn't be there, but be safe).
            self.root_nodes.retain(|r| *r != desc_id);
            self.free_slot(desc_id);
        }
    }

    /// Remove a node but **reparent its children** to the node's parent (or
    /// make them roots). Returns `true` if the node existed.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        if id.is_none() || !self.is_alive(id) {
            return false;
        }

        let parent = self.get(id).and_then(|n| n.parent);
        let children: Vec<NodeId> = self
            .get(id)
            .map(|n| n.children.clone())
            .unwrap_or_default();

        // Reparent children.
        for &child in &children {
            if let Some(child_node) = self.get_mut(child) {
                child_node.parent = parent;
            }
            if let Some(parent_id) = parent {
                if let Some(parent_node) = self.get_mut(parent_id) {
                    parent_node.children.push(child);
                }
            } else {
                // Child becomes a root.
                if !self.root_nodes.contains(&child) {
                    self.root_nodes.push(child);
                }
            }
        }

        // Remove from parent's child list.
        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.get_mut(parent_id) {
                parent_node.children.retain(|c| *c != id);
            }
        }

        // Remove from roots if applicable.
        self.root_nodes.retain(|r| *r != id);

        // Recycle slot.
        self.free_slot(id);
        true
    }

    // -- Hierarchy manipulation ---------------------------------------------

    /// Set the parent of `child` to `new_parent`. Pass `None` to make the
    /// child a root node.
    ///
    /// Performs cycle detection: returns `false` if setting this parent would
    /// create a cycle (i.e., `new_parent` is a descendant of `child`).
    pub fn set_parent(&mut self, child: NodeId, new_parent: Option<NodeId>) -> bool {
        if child.is_none() || !self.is_alive(child) {
            return false;
        }

        // If new_parent is Some, validate it exists and check for cycles.
        if let Some(parent_id) = new_parent {
            if parent_id.is_none() || !self.is_alive(parent_id) {
                return false;
            }
            // Don't parent to self.
            if parent_id == child {
                return false;
            }
            // Check if parent_id is a descendant of child (would create cycle).
            if self.is_ancestor_of(child, parent_id) {
                return false;
            }
        }

        // Remove from current parent's child list.
        let old_parent = self.get(child).and_then(|n| n.parent);
        if let Some(old_pid) = old_parent {
            if let Some(old_parent_node) = self.get_mut(old_pid) {
                old_parent_node.children.retain(|c| *c != child);
            }
        }

        // Remove from root list if it was a root.
        if old_parent.is_none() {
            self.root_nodes.retain(|r| *r != child);
        }

        // Set new parent.
        if let Some(child_node) = self.get_mut(child) {
            child_node.parent = new_parent;
            child_node.dirty = true;
        }

        match new_parent {
            Some(parent_id) => {
                if let Some(parent_node) = self.get_mut(parent_id) {
                    parent_node.children.push(child);
                }
            }
            None => {
                // Becomes a root.
                if !self.root_nodes.contains(&child) {
                    self.root_nodes.push(child);
                }
            }
        }

        // Mark the entire subtree as dirty.
        self.mark_subtree_dirty(child);

        true
    }

    /// Convenience: add `child` as a child of `parent`. Equivalent to
    /// `set_parent(child, Some(parent))`.
    pub fn reparent_child(&mut self, parent: NodeId, child: NodeId) -> bool {
        self.set_parent(child, Some(parent))
    }

    /// Remove `child` from `parent`'s children. The child becomes a root node.
    pub fn remove_child(&mut self, parent: NodeId, child: NodeId) -> bool {
        if !self.is_alive(parent) || !self.is_alive(child) {
            return false;
        }

        // Verify child is indeed a child of parent.
        let is_child = self
            .get(child)
            .and_then(|n| n.parent)
            .map(|p| p == parent)
            .unwrap_or(false);

        if !is_child {
            return false;
        }

        self.set_parent(child, None)
    }

    // -- Lookup / Search ----------------------------------------------------

    /// Get an immutable reference to a node by id.
    #[inline]
    pub fn get(&self, id: NodeId) -> Option<&SceneNode> {
        if id.is_none() {
            return None;
        }
        self.nodes
            .get(id.0 as usize)
            .and_then(|slot| slot.as_ref())
    }

    /// Get a mutable reference to a node by id.
    #[inline]
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        if id.is_none() {
            return None;
        }
        self.nodes
            .get_mut(id.0 as usize)
            .and_then(|slot| slot.as_mut())
    }

    /// Returns `true` if the node id refers to a living node.
    #[inline]
    pub fn is_alive(&self, id: NodeId) -> bool {
        if id.is_none() {
            return false;
        }
        self.nodes
            .get(id.0 as usize)
            .map(|slot| slot.is_some())
            .unwrap_or(false)
    }

    /// Find the first node with the given name using breadth-first search
    /// across all roots.
    pub fn find_by_name(&self, name: &str) -> Option<NodeId> {
        let mut queue = VecDeque::new();
        for &root in &self.root_nodes {
            queue.push_back(root);
        }
        while let Some(id) = queue.pop_front() {
            if let Some(node) = self.get(id) {
                if node.name == name {
                    return Some(id);
                }
                for &child in &node.children {
                    queue.push_back(child);
                }
            }
        }
        None
    }

    /// Find all nodes with the given name (breadth-first).
    pub fn find_all_by_name(&self, name: &str) -> Vec<NodeId> {
        let mut results = Vec::new();
        let mut queue = VecDeque::new();
        for &root in &self.root_nodes {
            queue.push_back(root);
        }
        while let Some(id) = queue.pop_front() {
            if let Some(node) = self.get(id) {
                if node.name == name {
                    results.push(id);
                }
                for &child in &node.children {
                    queue.push_back(child);
                }
            }
        }
        results
    }

    /// Find a node by path, e.g. `"Root/Body/LeftArm"`. Each segment is
    /// matched against child names starting from the root nodes.
    pub fn find_by_path(&self, path: &str) -> Option<NodeId> {
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if segments.is_empty() {
            return None;
        }

        // Find the root matching the first segment.
        let mut current: Option<NodeId> = None;
        for &root_id in &self.root_nodes {
            if let Some(root) = self.get(root_id) {
                if root.name == segments[0] {
                    current = Some(root_id);
                    break;
                }
            }
        }

        let mut current = current?;

        // Walk subsequent segments.
        for &segment in &segments[1..] {
            let node = self.get(current)?;
            let mut found = false;
            for &child_id in &node.children {
                if let Some(child) = self.get(child_id) {
                    if child.name == segment {
                        current = child_id;
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                return None;
            }
        }

        Some(current)
    }

    /// Build the path string for a node (e.g. "Root/Body/LeftArm").
    pub fn node_path(&self, id: NodeId) -> Option<String> {
        if !self.is_alive(id) {
            return None;
        }

        let mut segments = Vec::new();
        let mut current = Some(id);
        while let Some(cid) = current {
            if let Some(node) = self.get(cid) {
                segments.push(node.name.clone());
                current = node.parent;
            } else {
                break;
            }
        }
        segments.reverse();
        Some(segments.join("/"))
    }

    /// Find all nodes with a given tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<NodeId> {
        let mut results = Vec::new();
        for slot in &self.nodes {
            if let Some(node) = slot {
                if node.has_tag(tag) {
                    results.push(node.id);
                }
            }
        }
        results
    }

    /// Find the node linked to the given ECS entity.
    pub fn find_by_entity(&self, entity: Entity) -> Option<NodeId> {
        for slot in &self.nodes {
            if let Some(node) = slot {
                if node.entity == Some(entity) {
                    return Some(node.id);
                }
            }
        }
        None
    }

    // -- Traversal iterators ------------------------------------------------

    /// Pre-order depth-first iterator starting from the given node.
    pub fn iter_depth_first(&self, root: NodeId) -> DepthFirstIter<'_> {
        let mut stack = Vec::new();
        if root.is_some() && self.is_alive(root) {
            stack.push(root);
        }
        DepthFirstIter { tree: self, stack }
    }

    /// Level-order breadth-first iterator starting from the given node.
    pub fn iter_breadth_first(&self, root: NodeId) -> BreadthFirstIter<'_> {
        let mut queue = VecDeque::new();
        if root.is_some() && self.is_alive(root) {
            queue.push_back(root);
        }
        BreadthFirstIter { tree: self, queue }
    }

    /// Iterator that walks from the given node up to the root (ancestors).
    pub fn iter_ancestors(&self, node: NodeId) -> AncestorIter<'_> {
        let current = if node.is_some() && self.is_alive(node) {
            // Start from the node's parent.
            self.get(node).and_then(|n| n.parent)
        } else {
            None
        };
        AncestorIter {
            tree: self,
            current,
        }
    }

    /// Iterator over all descendants of a node (not including the node itself),
    /// in depth-first order.
    pub fn iter_descendants(&self, node: NodeId) -> DescendantIter<'_> {
        let mut stack = Vec::new();
        if node.is_some() && self.is_alive(node) {
            if let Some(n) = self.get(node) {
                // Push children in reverse for correct DFS order.
                for &child in n.children.iter().rev() {
                    stack.push(child);
                }
            }
        }
        DescendantIter { tree: self, stack }
    }

    /// Depth-first iterator over ALL nodes in the tree (starting from all roots).
    pub fn iter_all(&self) -> AllNodesIter<'_> {
        let mut stack = Vec::new();
        for &root_id in self.root_nodes.iter().rev() {
            stack.push(root_id);
        }
        AllNodesIter { tree: self, stack }
    }

    // -- Transform propagation ----------------------------------------------

    /// Propagate transforms top-down: walk the tree from all roots, computing
    /// `world_transform = parent_world * local_transform` for every node.
    ///
    /// This clears the `dirty` flag on every visited node.
    pub fn propagate_transforms(&mut self) {
        // Clone root list to avoid borrow conflict.
        let roots: Vec<NodeId> = self.root_nodes.clone();
        for root_id in roots {
            self.propagate_recursive(root_id, Mat4::IDENTITY);
        }
    }

    /// Propagate transforms for only the dirty subtrees. Walks from roots and
    /// skips clean subtrees entirely.
    pub fn propagate_transforms_incremental(&mut self) {
        let roots: Vec<NodeId> = self.root_nodes.clone();
        for root_id in roots {
            self.propagate_incremental_recursive(root_id, Mat4::IDENTITY, false);
        }
    }

    // -- Query helpers ------------------------------------------------------

    /// Total number of alive nodes in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.alive_count
    }

    /// Whether the tree has no alive nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.alive_count == 0
    }

    /// Total number of allocated slots (including free ones).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.nodes.len()
    }

    /// Number of root nodes.
    #[inline]
    pub fn root_count(&self) -> usize {
        self.root_nodes.len()
    }

    /// Slice of all root node ids.
    #[inline]
    pub fn roots(&self) -> &[NodeId] {
        &self.root_nodes
    }

    /// Count all descendants of a node (not including the node itself).
    pub fn descendant_count(&self, id: NodeId) -> usize {
        self.iter_descendants(id).count()
    }

    /// Returns the sibling index of a node within its parent's children list,
    /// or `None` if the node is not alive.
    pub fn sibling_index(&self, id: NodeId) -> Option<usize> {
        let node = self.get(id)?;
        if let Some(parent_id) = node.parent {
            let parent = self.get(parent_id)?;
            parent.children.iter().position(|c| *c == id)
        } else {
            // Root node: index in root list.
            self.root_nodes.iter().position(|r| *r == id)
        }
    }

    /// Check if `ancestor_id` is an ancestor of `descendant_id`.
    pub fn is_ancestor_of(&self, ancestor_id: NodeId, descendant_id: NodeId) -> bool {
        let mut current = self.get(descendant_id).and_then(|n| n.parent);
        while let Some(cid) = current {
            if cid == ancestor_id {
                return true;
            }
            current = self.get(cid).and_then(|n| n.parent);
        }
        false
    }

    /// Debug print the tree structure to a string.
    pub fn debug_print(&self) -> String {
        let mut out = String::new();
        for &root_id in &self.root_nodes {
            self.debug_print_recursive(&mut out, root_id, 0);
        }
        out
    }

    // -- Internal helpers ---------------------------------------------------

    fn allocate_slot(&mut self, name: &str) -> NodeId {
        let id = if let Some(free_id) = self.free_list.pop() {
            let idx = free_id.0 as usize;
            let node = SceneNode::new(free_id, name);
            self.nodes[idx] = Some(node);
            free_id
        } else {
            let idx = self.nodes.len() as u32;
            let id = NodeId(idx);
            self.nodes.push(Some(SceneNode::new(id, name)));
            id
        };
        self.alive_count += 1;
        id
    }

    fn free_slot(&mut self, id: NodeId) {
        if id.is_none() {
            return;
        }
        let idx = id.0 as usize;
        if idx < self.nodes.len() && self.nodes[idx].is_some() {
            self.nodes[idx] = None;
            self.free_list.push(id);
            self.alive_count -= 1;
        }
    }

    fn collect_descendants(&self, id: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut stack = Vec::new();
        if let Some(node) = self.get(id) {
            for &child in node.children.iter().rev() {
                stack.push(child);
            }
        }
        while let Some(cid) = stack.pop() {
            result.push(cid);
            if let Some(node) = self.get(cid) {
                for &child in node.children.iter().rev() {
                    stack.push(child);
                }
            }
        }
        result
    }

    fn mark_subtree_dirty(&mut self, id: NodeId) {
        if let Some(node) = self.get_mut(id) {
            node.dirty = true;
        }
        let children: Vec<NodeId> = self
            .get(id)
            .map(|n| n.children.clone())
            .unwrap_or_default();
        for child in children {
            self.mark_subtree_dirty(child);
        }
    }

    fn propagate_recursive(&mut self, node_id: NodeId, parent_world: Mat4) {
        if !self.is_alive(node_id) {
            return;
        }

        let local = self.nodes[node_id.0 as usize]
            .as_ref()
            .unwrap()
            .local_transform();
        let world = parent_world * local;

        let node = self.nodes[node_id.0 as usize].as_mut().unwrap();
        node.world_transform = world;
        node.dirty = false;

        let children: Vec<NodeId> = node.children.clone();
        for child in children {
            self.propagate_recursive(child, world);
        }
    }

    fn propagate_incremental_recursive(
        &mut self,
        node_id: NodeId,
        parent_world: Mat4,
        parent_was_dirty: bool,
    ) {
        if !self.is_alive(node_id) {
            return;
        }

        let node_ref = self.nodes[node_id.0 as usize].as_ref().unwrap();
        let is_dirty = node_ref.dirty || parent_was_dirty;

        if !is_dirty {
            // Node and ancestors are clean. But we still need to check children
            // in case they themselves are dirty.
            let children: Vec<NodeId> = node_ref.children.clone();
            let current_world = node_ref.world_transform;
            for child in children {
                self.propagate_incremental_recursive(child, current_world, false);
            }
            return;
        }

        // Recompute.
        let local = node_ref.local_transform();
        let world = parent_world * local;

        let node = self.nodes[node_id.0 as usize].as_mut().unwrap();
        node.world_transform = world;
        node.dirty = false;

        let children: Vec<NodeId> = node.children.clone();
        for child in children {
            self.propagate_incremental_recursive(child, world, true);
        }
    }

    fn debug_print_recursive(&self, out: &mut String, id: NodeId, depth: usize) {
        if let Some(node) = self.get(id) {
            let indent = "  ".repeat(depth);
            let vis = if node.visible { "V" } else { "H" };
            let dirty = if node.dirty { "D" } else { "C" };
            out.push_str(&format!(
                "{}{} [{}] ({}, {}) pos=({:.1}, {:.1}, {:.1})\n",
                indent,
                node.name,
                node.id,
                vis,
                dirty,
                node.local_position.x,
                node.local_position.y,
                node.local_position.z,
            ));
            for &child in &node.children {
                self.debug_print_recursive(out, child, depth + 1);
            }
        }
    }
}

impl Default for SceneNodeTree {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SceneNodeTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SceneNodeTree")
            .field("alive_count", &self.alive_count)
            .field("capacity", &self.nodes.len())
            .field("root_count", &self.root_nodes.len())
            .field("free_list_len", &self.free_list.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Free function for transform propagation
// ---------------------------------------------------------------------------

/// Walk the entire scene node tree and compute world transforms.
///
/// This is the same as `tree.propagate_transforms()` but provided as a free
/// function matching the specification signature.
pub fn propagate_transforms(tree: &mut SceneNodeTree) {
    tree.propagate_transforms();
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// Pre-order depth-first iterator over scene nodes.
pub struct DepthFirstIter<'a> {
    tree: &'a SceneNodeTree,
    stack: Vec<NodeId>,
}

impl<'a> Iterator for DepthFirstIter<'a> {
    type Item = &'a SceneNode;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let id = self.stack.pop()?;
            if let Some(node) = self.tree.get(id) {
                // Push children in reverse so the first child is visited first.
                for &child in node.children.iter().rev() {
                    self.stack.push(child);
                }
                return Some(node);
            }
            // Dead slot, skip it.
        }
    }
}

impl<'a> DepthFirstIter<'a> {
    /// Returns a size hint. The upper bound is unknown without walking.
    pub fn size_hint_lower(&self) -> usize {
        self.stack.len()
    }
}

/// Level-order breadth-first iterator over scene nodes.
pub struct BreadthFirstIter<'a> {
    tree: &'a SceneNodeTree,
    queue: VecDeque<NodeId>,
}

impl<'a> Iterator for BreadthFirstIter<'a> {
    type Item = &'a SceneNode;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let id = self.queue.pop_front()?;
            if let Some(node) = self.tree.get(id) {
                for &child in &node.children {
                    self.queue.push_back(child);
                }
                return Some(node);
            }
        }
    }
}

/// Iterator that walks from a node up to the root, yielding ancestors.
/// Does NOT yield the starting node itself.
pub struct AncestorIter<'a> {
    tree: &'a SceneNodeTree,
    current: Option<NodeId>,
}

impl<'a> Iterator for AncestorIter<'a> {
    type Item = &'a SceneNode;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.current?;
        let node = self.tree.get(id)?;
        self.current = node.parent;
        Some(node)
    }
}

/// Iterator over all descendants of a node (not including the node itself),
/// in depth-first pre-order.
pub struct DescendantIter<'a> {
    tree: &'a SceneNodeTree,
    stack: Vec<NodeId>,
}

impl<'a> Iterator for DescendantIter<'a> {
    type Item = &'a SceneNode;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let id = self.stack.pop()?;
            if let Some(node) = self.tree.get(id) {
                for &child in node.children.iter().rev() {
                    self.stack.push(child);
                }
                return Some(node);
            }
        }
    }
}

/// Depth-first iterator over all nodes in the tree (from all roots).
pub struct AllNodesIter<'a> {
    tree: &'a SceneNodeTree,
    stack: Vec<NodeId>,
}

impl<'a> Iterator for AllNodesIter<'a> {
    type Item = &'a SceneNode;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let id = self.stack.pop()?;
            if let Some(node) = self.tree.get(id) {
                for &child in node.children.iter().rev() {
                    self.stack.push(child);
                }
                return Some(node);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn node_id_sentinel() {
        let id = NodeId::NONE;
        assert!(id.is_none());
        assert!(!id.is_some());
        assert_eq!(format!("{}", id), "NodeId(NONE)");
    }

    #[test]
    fn node_id_display() {
        let id = NodeId(42);
        assert!(!id.is_none());
        assert!(id.is_some());
        assert_eq!(format!("{}", id), "NodeId(42)");
        assert_eq!(id.index(), 42);
    }

    #[test]
    fn create_node_and_access() {
        let mut tree = SceneNodeTree::new();
        let id = tree.create_node("Root");
        assert!(tree.is_alive(id));
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.root_count(), 1);

        let node = tree.get(id).unwrap();
        assert_eq!(node.name, "Root");
        assert!(node.is_root());
        assert!(!node.has_children());
    }

    #[test]
    fn add_root_and_child() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let child = tree.add_child(root, "Child");

        assert_eq!(tree.len(), 2);
        assert_eq!(tree.get(root).unwrap().children.len(), 1);
        assert_eq!(tree.get(child).unwrap().parent, Some(root));
        assert!(!tree.get(child).unwrap().is_root());
    }

    #[test]
    fn destroy_node_removes_descendants() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let a = tree.add_child(root, "A");
        let b = tree.add_child(a, "B");
        let _c = tree.add_child(b, "C");

        assert_eq!(tree.len(), 4);
        tree.destroy_node(a);
        // A, B, C all destroyed; root remains.
        assert_eq!(tree.len(), 1);
        assert!(tree.get(root).unwrap().children.is_empty());
        assert!(!tree.is_alive(a));
        assert!(!tree.is_alive(b));
    }

    #[test]
    fn remove_node_reparents_children() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let a = tree.add_child(root, "A");
        let b = tree.add_child(a, "B");
        let c = tree.add_child(a, "C");

        tree.remove_node(a);
        assert_eq!(tree.len(), 3);
        // B and C should now be children of Root.
        let root_node = tree.get(root).unwrap();
        assert!(root_node.children.contains(&b));
        assert!(root_node.children.contains(&c));
        assert_eq!(tree.get(b).unwrap().parent, Some(root));
        assert_eq!(tree.get(c).unwrap().parent, Some(root));
    }

    #[test]
    fn remove_root_node_children_become_roots() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let a = tree.add_child(root, "A");
        let b = tree.add_child(root, "B");

        tree.remove_node(root);
        assert_eq!(tree.len(), 2);
        assert!(tree.root_nodes.contains(&a));
        assert!(tree.root_nodes.contains(&b));
        assert!(tree.get(a).unwrap().is_root());
        assert!(tree.get(b).unwrap().is_root());
    }

    #[test]
    fn set_parent_basic() {
        let mut tree = SceneNodeTree::new();
        let a = tree.create_node("A");
        let b = tree.create_node("B");

        assert!(tree.set_parent(b, Some(a)));
        assert_eq!(tree.get(b).unwrap().parent, Some(a));
        assert!(tree.get(a).unwrap().children.contains(&b));
        assert_eq!(tree.root_count(), 1); // Only A is a root now.
    }

    #[test]
    fn set_parent_detects_cycle() {
        let mut tree = SceneNodeTree::new();
        let a = tree.create_node("A");
        let b = tree.add_child(a, "B");
        let c = tree.add_child(b, "C");

        // Trying to set A's parent to C would create a cycle.
        assert!(!tree.set_parent(a, Some(c)));
        // A should still be a root.
        assert!(tree.get(a).unwrap().is_root());
    }

    #[test]
    fn set_parent_self_rejected() {
        let mut tree = SceneNodeTree::new();
        let a = tree.create_node("A");
        assert!(!tree.set_parent(a, Some(a)));
    }

    #[test]
    fn set_parent_to_none_makes_root() {
        let mut tree = SceneNodeTree::new();
        let a = tree.create_node("A");
        let b = tree.add_child(a, "B");

        assert!(!tree.get(b).unwrap().is_root());
        assert!(tree.set_parent(b, None));
        assert!(tree.get(b).unwrap().is_root());
        assert!(tree.root_nodes.contains(&b));
    }

    #[test]
    fn find_by_name_bfs() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("World");
        let _a = tree.add_child(root, "Player");
        let _b = tree.add_child(root, "Enemy");

        assert_eq!(tree.find_by_name("Player"), Some(_a));
        assert_eq!(tree.find_by_name("Enemy"), Some(_b));
        assert_eq!(tree.find_by_name("NotFound"), None);
    }

    #[test]
    fn find_all_by_name() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let a = tree.add_child(root, "Sword");
        let b = tree.add_child(root, "Shield");
        let c = tree.add_child(b, "Sword");

        let swords = tree.find_all_by_name("Sword");
        assert_eq!(swords.len(), 2);
        assert!(swords.contains(&a));
        assert!(swords.contains(&c));
    }

    #[test]
    fn find_by_path() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let body = tree.add_child(root, "Body");
        let left_arm = tree.add_child(body, "LeftArm");
        let _hand = tree.add_child(left_arm, "Hand");

        assert_eq!(tree.find_by_path("Root/Body/LeftArm"), Some(left_arm));
        assert_eq!(tree.find_by_path("Root/Body/LeftArm/Hand"), Some(_hand));
        assert_eq!(tree.find_by_path("Root"), Some(root));
        assert_eq!(tree.find_by_path("Root/Body/RightArm"), None);
        assert_eq!(tree.find_by_path(""), None);
        assert_eq!(tree.find_by_path("NotARoot"), None);
    }

    #[test]
    fn node_path_builds_correctly() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let body = tree.add_child(root, "Body");
        let arm = tree.add_child(body, "LeftArm");

        assert_eq!(tree.node_path(arm).unwrap(), "Root/Body/LeftArm");
        assert_eq!(tree.node_path(root).unwrap(), "Root");
    }

    #[test]
    fn find_by_tag() {
        let mut tree = SceneNodeTree::new();
        let a = tree.create_node("A");
        let b = tree.create_node("B");

        tree.get_mut(a).unwrap().add_tag("enemy");
        tree.get_mut(b).unwrap().add_tag("enemy");
        tree.get_mut(b).unwrap().add_tag("boss");

        let enemies = tree.find_by_tag("enemy");
        assert_eq!(enemies.len(), 2);

        let bosses = tree.find_by_tag("boss");
        assert_eq!(bosses.len(), 1);
        assert_eq!(bosses[0], b);
    }

    #[test]
    fn node_tags() {
        let mut node = SceneNode::new(NodeId(0), "Test");
        node.add_tag("visible");
        assert!(node.has_tag("visible"));
        assert!(!node.has_tag("hidden"));

        // Duplicate add does nothing.
        node.add_tag("visible");
        assert_eq!(node.tags.len(), 1);

        assert!(node.remove_tag("visible"));
        assert!(!node.has_tag("visible"));
        assert!(!node.remove_tag("nonexistent"));
    }

    #[test]
    fn depth_first_order() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("A");
        let b = tree.add_child(root, "B");
        let _c = tree.add_child(root, "C");
        let _d = tree.add_child(b, "D");

        let names: Vec<&str> = tree
            .iter_depth_first(root)
            .map(|n| n.name.as_str())
            .collect();
        assert_eq!(names, vec!["A", "B", "D", "C"]);
    }

    #[test]
    fn breadth_first_order() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("A");
        let b = tree.add_child(root, "B");
        let _c = tree.add_child(root, "C");
        let _d = tree.add_child(b, "D");

        let names: Vec<&str> = tree
            .iter_breadth_first(root)
            .map(|n| n.name.as_str())
            .collect();
        assert_eq!(names, vec!["A", "B", "C", "D"]);
    }

    #[test]
    fn ancestor_iter() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let a = tree.add_child(root, "A");
        let b = tree.add_child(a, "B");
        let c = tree.add_child(b, "C");

        let ancestors: Vec<&str> = tree
            .iter_ancestors(c)
            .map(|n| n.name.as_str())
            .collect();
        assert_eq!(ancestors, vec!["B", "A", "Root"]);
    }

    #[test]
    fn ancestor_iter_from_root_is_empty() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");

        let ancestors: Vec<&str> = tree
            .iter_ancestors(root)
            .map(|n| n.name.as_str())
            .collect();
        assert!(ancestors.is_empty());
    }

    #[test]
    fn descendant_iter() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let a = tree.add_child(root, "A");
        let b = tree.add_child(root, "B");
        let c = tree.add_child(a, "C");
        let _d = tree.add_child(a, "D");

        let descendants: Vec<&str> = tree
            .iter_descendants(root)
            .map(|n| n.name.as_str())
            .collect();
        assert_eq!(descendants, vec!["A", "C", "D", "B"]);
    }

    #[test]
    fn iter_all_nodes() {
        let mut tree = SceneNodeTree::new();
        let r1 = tree.add_root("R1");
        let r2 = tree.add_root("R2");
        let _a = tree.add_child(r1, "A");
        let _b = tree.add_child(r2, "B");

        let names: Vec<&str> = tree.iter_all().map(|n| n.name.as_str()).collect();
        assert_eq!(names.len(), 4);
        assert_eq!(names[0], "R1");
    }

    #[test]
    fn propagate_transforms_identity() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        tree.propagate_transforms();

        let node = tree.get(root).unwrap();
        assert!(!node.dirty);
        assert_eq!(node.world_transform, Mat4::IDENTITY);
    }

    #[test]
    fn propagate_transforms_translation() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        tree.get_mut(root).unwrap().set_position(Vec3::new(10.0, 0.0, 0.0));

        let child = tree.add_child(root, "Child");
        tree.get_mut(child).unwrap().set_position(Vec3::new(0.0, 5.0, 0.0));

        tree.propagate_transforms();

        let root_pos = tree.get(root).unwrap().world_position();
        assert!((root_pos - Vec3::new(10.0, 0.0, 0.0)).length() < 1e-5);

        let child_pos = tree.get(child).unwrap().world_position();
        assert!((child_pos - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn propagate_transforms_scale() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        tree.get_mut(root).unwrap().set_scale(Vec3::splat(2.0));

        let child = tree.add_child(root, "Child");
        tree.get_mut(child)
            .unwrap()
            .set_position(Vec3::new(1.0, 0.0, 0.0));

        tree.propagate_transforms();

        let child_pos = tree.get(child).unwrap().world_position();
        // Child at local (1,0,0) under parent with scale 2 => world (2,0,0).
        assert!((child_pos - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn propagate_incremental_only_dirty() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        tree.get_mut(root)
            .unwrap()
            .set_position(Vec3::new(1.0, 0.0, 0.0));
        let child = tree.add_child(root, "Child");
        tree.get_mut(child)
            .unwrap()
            .set_position(Vec3::new(0.0, 1.0, 0.0));

        // Full propagation first.
        tree.propagate_transforms();

        // Modify only the child.
        tree.get_mut(child)
            .unwrap()
            .set_position(Vec3::new(0.0, 2.0, 0.0));

        tree.propagate_transforms_incremental();

        let child_pos = tree.get(child).unwrap().world_position();
        assert!((child_pos - Vec3::new(1.0, 2.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn free_list_reuse() {
        let mut tree = SceneNodeTree::new();
        let a = tree.create_node("A");
        let b = tree.create_node("B");
        assert_eq!(tree.capacity(), 2);

        tree.destroy_node(a);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.free_list.len(), 1);

        let c = tree.create_node("C");
        // Should reuse slot.
        assert_eq!(c.0, a.0);
        assert_eq!(tree.capacity(), 2);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn is_ancestor_of() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let a = tree.add_child(root, "A");
        let b = tree.add_child(a, "B");

        assert!(tree.is_ancestor_of(root, b));
        assert!(tree.is_ancestor_of(a, b));
        assert!(!tree.is_ancestor_of(b, root));
        assert!(!tree.is_ancestor_of(b, a));
    }

    #[test]
    fn descendant_count() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let a = tree.add_child(root, "A");
        let _b = tree.add_child(a, "B");
        let _c = tree.add_child(root, "C");

        assert_eq!(tree.descendant_count(root), 3);
        assert_eq!(tree.descendant_count(a), 1);
    }

    #[test]
    fn sibling_index() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let a = tree.add_child(root, "A");
        let b = tree.add_child(root, "B");
        let c = tree.add_child(root, "C");

        assert_eq!(tree.sibling_index(a), Some(0));
        assert_eq!(tree.sibling_index(b), Some(1));
        assert_eq!(tree.sibling_index(c), Some(2));
    }

    #[test]
    fn entity_linkage() {
        let mut tree = SceneNodeTree::new();
        let id = tree.create_node("Linked");
        let entity = Entity::new(42, 0);
        tree.get_mut(id).unwrap().entity = Some(entity);

        assert_eq!(tree.find_by_entity(entity), Some(id));
        assert_eq!(tree.find_by_entity(Entity::new(99, 0)), None);
    }

    #[test]
    fn debug_print() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let _a = tree.add_child(root, "A");
        let _b = tree.add_child(root, "B");

        let output = tree.debug_print();
        assert!(output.contains("Root"));
        assert!(output.contains("A"));
        assert!(output.contains("B"));
    }

    #[test]
    fn node_set_and_get_local_transform() {
        let mut node = SceneNode::new(NodeId(0), "Test");
        let t = genovo_core::Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::IDENTITY,
            Vec3::ONE,
        );
        node.set_local_transform(&t);
        assert!(node.dirty);

        let t2 = node.get_local_transform();
        assert!((t.position - t2.position).length() < 1e-5);
    }

    #[test]
    fn node_world_accessors() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        tree.get_mut(root)
            .unwrap()
            .set_position(Vec3::new(5.0, 10.0, 15.0));
        tree.propagate_transforms();

        let node = tree.get(root).unwrap();
        let pos = node.world_position();
        assert!((pos - Vec3::new(5.0, 10.0, 15.0)).length() < 1e-5);
    }

    #[test]
    fn depth_in_tree() {
        let mut tree = SceneNodeTree::new();
        let root = tree.add_root("Root");
        let a = tree.add_child(root, "A");
        let b = tree.add_child(a, "B");

        assert_eq!(tree.get(root).unwrap().depth_in(&tree), 0);
        assert_eq!(tree.get(a).unwrap().depth_in(&tree), 1);
        assert_eq!(tree.get(b).unwrap().depth_in(&tree), 2);
    }

    #[test]
    fn reparent_child() {
        let mut tree = SceneNodeTree::new();
        let a = tree.create_node("A");
        let b = tree.create_node("B");
        let c = tree.add_child(a, "C");

        // Move C from A to B.
        assert!(tree.reparent_child(b, c));
        assert_eq!(tree.get(c).unwrap().parent, Some(b));
        assert!(tree.get(a).unwrap().children.is_empty());
        assert!(tree.get(b).unwrap().children.contains(&c));
    }

    #[test]
    fn remove_child() {
        let mut tree = SceneNodeTree::new();
        let a = tree.create_node("A");
        let b = tree.add_child(a, "B");

        assert!(tree.remove_child(a, b));
        assert!(tree.get(b).unwrap().is_root());
        assert!(tree.get(a).unwrap().children.is_empty());
    }

    #[test]
    fn remove_child_wrong_parent_fails() {
        let mut tree = SceneNodeTree::new();
        let a = tree.create_node("A");
        let b = tree.create_node("B");
        let c = tree.add_child(a, "C");

        assert!(!tree.remove_child(b, c));
    }

    #[test]
    fn multiple_roots() {
        let mut tree = SceneNodeTree::new();
        let r1 = tree.create_node("R1");
        let r2 = tree.create_node("R2");
        let r3 = tree.create_node("R3");

        assert_eq!(tree.root_count(), 3);
        assert_eq!(tree.roots(), &[r1, r2, r3]);
    }

    #[test]
    fn with_capacity() {
        let tree = SceneNodeTree::with_capacity(100);
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn destroy_root_node() {
        let mut tree = SceneNodeTree::new();
        let root = tree.create_node("Root");
        assert_eq!(tree.root_count(), 1);

        tree.destroy_node(root);
        assert_eq!(tree.root_count(), 0);
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn destroy_nonexistent_noop() {
        let mut tree = SceneNodeTree::new();
        tree.destroy_node(NodeId::NONE);
        tree.destroy_node(NodeId(999));
        assert_eq!(tree.len(), 0);
    }
}
