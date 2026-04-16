//! # Scene Hierarchy Panel
//!
//! Displays the scene's entity tree in a collapsible tree view. Supports
//! entity creation, deletion, reparenting via drag-and-drop, multi-select
//! with shift/ctrl, context menu operations, and name-based search/filter.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// HierarchyNode
// ---------------------------------------------------------------------------

/// A node in the hierarchy tree, representing an entity and its children.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyNode {
    /// Entity UUID.
    pub entity_id: Uuid,
    /// Display name shown in the hierarchy.
    pub name: String,
    /// Whether this node is expanded in the tree view.
    pub expanded: bool,
    /// Whether this node is visible (eye icon).
    pub visible: bool,
    /// Whether this node is locked (lock icon -- locked entities cannot be
    /// selected or moved in the viewport).
    pub locked: bool,
    /// Child nodes.
    pub children: Vec<HierarchyNode>,
    /// Depth in the tree (0 = root level). Cached for rendering.
    pub depth: u32,
}

impl HierarchyNode {
    /// Create a new root-level node.
    pub fn new(entity_id: Uuid, name: impl Into<String>) -> Self {
        Self {
            entity_id,
            name: name.into(),
            expanded: true,
            visible: true,
            locked: false,
            children: Vec::new(),
            depth: 0,
        }
    }

    /// Create a node at a specific depth.
    pub fn with_depth(entity_id: Uuid, name: impl Into<String>, depth: u32) -> Self {
        Self {
            entity_id,
            name: name.into(),
            expanded: true,
            visible: true,
            locked: false,
            children: Vec::new(),
            depth,
        }
    }

    /// Total number of nodes in this subtree (including self).
    pub fn count(&self) -> usize {
        1 + self.children.iter().map(|c| c.count()).sum::<usize>()
    }

    /// Flatten the tree into a list of (depth, node_ref) for rendering.
    pub fn flatten(&self) -> Vec<(u32, &HierarchyNode)> {
        let mut result = Vec::new();
        self.flatten_inner(self.depth, &mut result);
        result
    }

    fn flatten_inner<'a>(&'a self, depth: u32, out: &mut Vec<(u32, &'a HierarchyNode)>) {
        out.push((depth, self));
        if self.expanded {
            for child in &self.children {
                child.flatten_inner(depth + 1, out);
            }
        }
    }

    /// Check if this subtree contains an entity with the given id.
    pub fn contains(&self, entity_id: Uuid) -> bool {
        if self.entity_id == entity_id {
            return true;
        }
        self.children.iter().any(|c| c.contains(entity_id))
    }

    /// Collect all entity IDs in this subtree.
    pub fn all_entity_ids(&self) -> Vec<Uuid> {
        let mut ids = vec![self.entity_id];
        for child in &self.children {
            ids.extend(child.all_entity_ids());
        }
        ids
    }

    /// Set depth recursively for all children.
    fn update_depths(&mut self, depth: u32) {
        self.depth = depth;
        for child in &mut self.children {
            child.update_depths(depth + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Drag state
// ---------------------------------------------------------------------------

/// State tracking an in-progress drag for reparenting.
#[derive(Debug, Clone)]
pub struct HierarchyDragState {
    /// Entity being dragged.
    pub dragged_entity: Uuid,
    /// Entity currently under the cursor (drop target).
    pub drop_target: Option<Uuid>,
    /// Drop position relative to the target.
    pub drop_position: DropPosition,
}

/// Where to drop an entity relative to the target node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPosition {
    /// Insert as a child of the target.
    Inside,
    /// Insert above the target (same parent).
    Above,
    /// Insert below the target (same parent).
    Below,
}

// ---------------------------------------------------------------------------
// Context menu
// ---------------------------------------------------------------------------

/// Actions available in the hierarchy right-click context menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HierarchyContextAction {
    CreateEmpty,
    CreateFromPrefab,
    Delete,
    Duplicate,
    Rename,
    CopyEntity,
    PasteEntity,
    SelectChildren,
    ExpandAll,
    CollapseAll,
    ToggleVisibility,
    ToggleLock,
}

// ---------------------------------------------------------------------------
// HierarchyPanel
// ---------------------------------------------------------------------------

/// The scene hierarchy panel.
#[derive(Debug)]
pub struct HierarchyPanel {
    /// Root nodes of the hierarchy (top-level entities).
    pub roots: Vec<HierarchyNode>,
    /// Whether the hierarchy panel is visible.
    pub visible: bool,
    /// Current search/filter string.
    pub search_query: String,
    /// Entity ID currently being renamed (if any).
    pub renaming: Option<Uuid>,
    /// The rename text buffer.
    pub rename_buffer: String,
    /// Currently selected entity IDs.
    selected: Vec<Uuid>,
    /// Clipboard for copy/paste.
    clipboard: Option<HierarchyNode>,
    /// Active drag state.
    drag_state: Option<HierarchyDragState>,
    /// Whether the context menu is open.
    pub context_menu_open: bool,
    /// Entity for which the context menu was opened.
    pub context_menu_entity: Option<Uuid>,
    /// Next entity ID counter (used for entity creation without a world).
    next_entity_counter: u64,
}

impl Default for HierarchyPanel {
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            visible: true,
            search_query: String::new(),
            renaming: None,
            rename_buffer: String::new(),
            selected: Vec::new(),
            clipboard: None,
            drag_state: None,
            context_menu_open: false,
            context_menu_entity: None,
            next_entity_counter: 0,
        }
    }
}

impl HierarchyPanel {
    /// Create a new hierarchy panel.
    pub fn new() -> Self {
        Self::default()
    }

    // --- Entity CRUD --------------------------------------------------------

    /// Create a new empty entity as a child of the given parent (or root).
    pub fn create_entity(&mut self, parent: Option<Uuid>, name: &str) -> Uuid {
        let id = Uuid::new_v4();

        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.find_node_mut(parent_id) {
                let depth = parent_node.depth + 1;
                parent_node
                    .children
                    .push(HierarchyNode::with_depth(id, name, depth));
                parent_node.expanded = true;
            }
        } else {
            self.roots.push(HierarchyNode::new(id, name));
        }

        id
    }

    /// Delete an entity and all its descendants from the hierarchy.
    pub fn delete_entity(&mut self, entity_id: Uuid) {
        // Remove from selected.
        self.selected.retain(|&id| id != entity_id);

        // Remove from roots.
        if self.roots.iter().any(|r| r.entity_id == entity_id) {
            self.roots.retain(|r| r.entity_id != entity_id);
            return;
        }

        // Remove from children recursively.
        for root in &mut self.roots {
            if remove_child_recursive(root, entity_id) {
                return;
            }
        }
    }

    /// Delete all selected entities.
    pub fn delete_selected(&mut self) {
        let to_delete: Vec<Uuid> = self.selected.clone();
        for id in to_delete {
            self.delete_entity(id);
        }
    }

    /// Reparent an entity under a new parent (or to root level if `None`).
    pub fn reparent_entity(&mut self, entity_id: Uuid, new_parent: Option<Uuid>) {
        // Prevent reparenting under self or own descendant.
        if let Some(np) = new_parent {
            if np == entity_id {
                return;
            }
            if let Some(node) = self.find_node(entity_id) {
                if node.contains(np) {
                    return; // Would create cycle.
                }
            }
        }

        // Extract the node.
        let node = self.extract_node(entity_id);
        let mut node = match node {
            Some(n) => n,
            None => return,
        };

        // Re-insert.
        if let Some(parent_id) = new_parent {
            if let Some(parent_node) = self.find_node_mut(parent_id) {
                node.update_depths(parent_node.depth + 1);
                parent_node.children.push(node);
                parent_node.expanded = true;
            } else {
                // Parent not found; add to root.
                node.update_depths(0);
                self.roots.push(node);
            }
        } else {
            node.update_depths(0);
            self.roots.push(node);
        }
    }

    /// Duplicate an entity and all its components/children.
    pub fn duplicate_entity(&mut self, entity_id: Uuid) -> Option<Uuid> {
        let original = self.find_node(entity_id)?;
        let cloned = deep_clone_node(original);
        let new_id = cloned.entity_id;

        // Find the parent of the original and insert the clone as a sibling.
        let parent_id = self.find_parent(entity_id);
        if let Some(pid) = parent_id {
            if let Some(parent_node) = self.find_node_mut(pid) {
                parent_node.children.push(cloned);
            }
        } else {
            self.roots.push(cloned);
        }

        Some(new_id)
    }

    /// Rename an entity.
    pub fn rename_entity(&mut self, entity_id: Uuid, new_name: &str) {
        if let Some(node) = self.find_node_mut(entity_id) {
            node.name = new_name.to_string();
        }
        if self.renaming == Some(entity_id) {
            self.renaming = None;
            self.rename_buffer.clear();
        }
    }

    /// Start renaming an entity.
    pub fn begin_rename(&mut self, entity_id: Uuid) {
        if let Some(node) = self.find_node(entity_id) {
            self.rename_buffer = node.name.clone();
            self.renaming = Some(entity_id);
        }
    }

    /// Commit the current rename operation.
    pub fn commit_rename(&mut self) {
        if let Some(id) = self.renaming {
            let name = self.rename_buffer.clone();
            if !name.is_empty() {
                self.rename_entity(id, &name);
            }
        }
        self.renaming = None;
        self.rename_buffer.clear();
    }

    /// Cancel the current rename operation.
    pub fn cancel_rename(&mut self) {
        self.renaming = None;
        self.rename_buffer.clear();
    }

    /// Toggle visibility of an entity and its descendants.
    pub fn toggle_visibility(&mut self, entity_id: Uuid) {
        if let Some(node) = self.find_node_mut(entity_id) {
            let new_vis = !node.visible;
            set_visibility_recursive(node, new_vis);
        }
    }

    /// Toggle lock state of an entity.
    pub fn toggle_lock(&mut self, entity_id: Uuid) {
        if let Some(node) = self.find_node_mut(entity_id) {
            node.locked = !node.locked;
        }
    }

    // --- Selection ----------------------------------------------------------

    /// Select a single entity (deselecting all others).
    pub fn select(&mut self, entity_id: Uuid) {
        self.selected.clear();
        self.selected.push(entity_id);
    }

    /// Add an entity to the selection.
    pub fn add_to_selection(&mut self, entity_id: Uuid) {
        if !self.selected.contains(&entity_id) {
            self.selected.push(entity_id);
        }
    }

    /// Toggle an entity's selection state.
    pub fn toggle_selection(&mut self, entity_id: Uuid) {
        if let Some(pos) = self.selected.iter().position(|&id| id == entity_id) {
            self.selected.remove(pos);
        } else {
            self.selected.push(entity_id);
        }
    }

    /// Range select: select all entities between the last-selected and the
    /// given entity (in flattened tree order).
    pub fn range_select(&mut self, entity_id: Uuid) {
        let last = self.selected.last().copied();

        let Some(anchor) = last else {
            self.select(entity_id);
            return;
        };

        // Collect entity IDs from flattened tree to avoid borrow conflict.
        let flat_ids: Vec<Uuid> = self.flatten().iter().map(|(_, node)| node.entity_id).collect();

        let mut anchor_idx = None;
        let mut target_idx = None;

        for (i, id) in flat_ids.iter().enumerate() {
            if *id == anchor {
                anchor_idx = Some(i);
            }
            if *id == entity_id {
                target_idx = Some(i);
            }
        }

        let (Some(a), Some(b)) = (anchor_idx, target_idx) else {
            self.select(entity_id);
            return;
        };

        let (start, end) = if a <= b { (a, b) } else { (b, a) };

        for i in start..=end {
            let id = flat_ids[i];
            if !self.selected.contains(&id) {
                self.selected.push(id);
            }
        }
    }

    /// Clear the selection.
    pub fn clear_selection(&mut self) {
        self.selected.clear();
    }

    /// Get the selected entity IDs.
    pub fn selected(&self) -> &[Uuid] {
        &self.selected
    }

    /// Whether the given entity is selected.
    pub fn is_selected(&self, entity_id: Uuid) -> bool {
        self.selected.contains(&entity_id)
    }

    /// Select all children of the given entity.
    pub fn select_children(&mut self, entity_id: Uuid) {
        if let Some(node) = self.find_node(entity_id) {
            let ids = node.all_entity_ids();
            for id in ids {
                if !self.selected.contains(&id) {
                    self.selected.push(id);
                }
            }
        }
    }

    // --- Drag & Drop --------------------------------------------------------

    /// Begin a drag operation.
    pub fn begin_drag(&mut self, entity_id: Uuid) {
        self.drag_state = Some(HierarchyDragState {
            dragged_entity: entity_id,
            drop_target: None,
            drop_position: DropPosition::Inside,
        });
    }

    /// Update the drag target.
    pub fn update_drag(&mut self, target: Option<Uuid>, position: DropPosition) {
        if let Some(ref mut state) = self.drag_state {
            state.drop_target = target;
            state.drop_position = position;
        }
    }

    /// Complete the drag and perform reparenting.
    pub fn end_drag(&mut self) -> Option<(Uuid, Option<Uuid>)> {
        let state = self.drag_state.take()?;
        let target = state.drop_target?;

        match state.drop_position {
            DropPosition::Inside => {
                self.reparent_entity(state.dragged_entity, Some(target));
                Some((state.dragged_entity, Some(target)))
            }
            DropPosition::Above | DropPosition::Below => {
                // Reparent under the target's parent.
                let target_parent = self.find_parent(target);
                self.reparent_entity(state.dragged_entity, target_parent);
                Some((state.dragged_entity, target_parent))
            }
        }
    }

    /// Cancel the drag.
    pub fn cancel_drag(&mut self) {
        self.drag_state = None;
    }

    /// Whether a drag is in progress.
    pub fn is_dragging(&self) -> bool {
        self.drag_state.is_some()
    }

    // --- Context Menu -------------------------------------------------------

    /// Open the context menu for an entity.
    pub fn open_context_menu(&mut self, entity_id: Option<Uuid>) {
        self.context_menu_open = true;
        self.context_menu_entity = entity_id;
    }

    /// Close the context menu.
    pub fn close_context_menu(&mut self) {
        self.context_menu_open = false;
        self.context_menu_entity = None;
    }

    /// Execute a context menu action.
    pub fn execute_context_action(&mut self, action: HierarchyContextAction) -> Option<Uuid> {
        let entity = self.context_menu_entity;
        self.close_context_menu();

        match action {
            HierarchyContextAction::CreateEmpty => {
                let id = self.create_entity(entity, "New Entity");
                Some(id)
            }
            HierarchyContextAction::Delete => {
                if let Some(id) = entity {
                    self.delete_entity(id);
                }
                None
            }
            HierarchyContextAction::Duplicate => {
                entity.and_then(|id| self.duplicate_entity(id))
            }
            HierarchyContextAction::Rename => {
                if let Some(id) = entity {
                    self.begin_rename(id);
                }
                None
            }
            HierarchyContextAction::CopyEntity => {
                if let Some(id) = entity {
                    if let Some(node) = self.find_node(id) {
                        self.clipboard = Some(deep_clone_node(node));
                    }
                }
                None
            }
            HierarchyContextAction::PasteEntity => {
                if let Some(clip) = self.clipboard.clone() {
                    let id = clip.entity_id;
                    if let Some(parent_id) = entity {
                        if let Some(parent) = self.find_node_mut(parent_id) {
                            parent.children.push(clip);
                        }
                    } else {
                        self.roots.push(clip);
                    }
                    Some(id)
                } else {
                    None
                }
            }
            HierarchyContextAction::SelectChildren => {
                if let Some(id) = entity {
                    self.select_children(id);
                }
                None
            }
            HierarchyContextAction::ExpandAll => {
                if let Some(id) = entity {
                    if let Some(node) = self.find_node_mut(id) {
                        set_expanded_recursive(node, true);
                    }
                } else {
                    for root in &mut self.roots {
                        set_expanded_recursive(root, true);
                    }
                }
                None
            }
            HierarchyContextAction::CollapseAll => {
                if let Some(id) = entity {
                    if let Some(node) = self.find_node_mut(id) {
                        set_expanded_recursive(node, false);
                    }
                } else {
                    for root in &mut self.roots {
                        set_expanded_recursive(root, false);
                    }
                }
                None
            }
            HierarchyContextAction::ToggleVisibility => {
                if let Some(id) = entity {
                    self.toggle_visibility(id);
                }
                None
            }
            HierarchyContextAction::ToggleLock => {
                if let Some(id) = entity {
                    self.toggle_lock(id);
                }
                None
            }
            HierarchyContextAction::CreateFromPrefab => {
                // Prefab instantiation requires asset browser integration.
                // Create a placeholder entity.
                let id = self.create_entity(entity, "Prefab Instance");
                Some(id)
            }
        }
    }

    // --- Search / Filter ----------------------------------------------------

    /// Set the search filter. Returns matching entity IDs.
    pub fn set_search(&mut self, query: &str) -> Vec<Uuid> {
        self.search_query = query.to_string();
        if query.is_empty() {
            return Vec::new();
        }
        self.search_results()
    }

    /// Get entity IDs matching the current search query.
    pub fn search_results(&self) -> Vec<Uuid> {
        if self.search_query.is_empty() {
            return Vec::new();
        }
        let query = self.search_query.to_lowercase();
        let mut results = Vec::new();
        for root in &self.roots {
            search_recursive(root, &query, &mut results);
        }
        results
    }

    // --- Tree queries -------------------------------------------------------

    /// Find a node by entity ID in the hierarchy tree.
    pub fn find_node(&self, entity_id: Uuid) -> Option<&HierarchyNode> {
        fn search(nodes: &[HierarchyNode], id: Uuid) -> Option<&HierarchyNode> {
            for node in nodes {
                if node.entity_id == id {
                    return Some(node);
                }
                if let Some(found) = search(&node.children, id) {
                    return Some(found);
                }
            }
            None
        }
        search(&self.roots, entity_id)
    }

    /// Find a mutable node by entity ID.
    pub fn find_node_mut(&mut self, entity_id: Uuid) -> Option<&mut HierarchyNode> {
        fn search(nodes: &mut [HierarchyNode], id: Uuid) -> Option<&mut HierarchyNode> {
            for node in nodes {
                if node.entity_id == id {
                    return Some(node);
                }
                if let Some(found) = search(&mut node.children, id) {
                    return Some(found);
                }
            }
            None
        }
        search(&mut self.roots, entity_id)
    }

    /// Find the parent entity ID of a given entity.
    pub fn find_parent(&self, entity_id: Uuid) -> Option<Uuid> {
        fn search(nodes: &[HierarchyNode], target: Uuid) -> Option<Uuid> {
            for node in nodes {
                for child in &node.children {
                    if child.entity_id == target {
                        return Some(node.entity_id);
                    }
                    if let Some(parent) = search(&node.children, target) {
                        return Some(parent);
                    }
                }
            }
            None
        }
        search(&self.roots, entity_id)
    }

    /// Extract (remove) a node from the tree and return it.
    fn extract_node(&mut self, entity_id: Uuid) -> Option<HierarchyNode> {
        // Check roots.
        if let Some(idx) = self.roots.iter().position(|r| r.entity_id == entity_id) {
            return Some(self.roots.remove(idx));
        }
        // Check children recursively.
        for root in &mut self.roots {
            if let Some(node) = extract_child_recursive(root, entity_id) {
                return Some(node);
            }
        }
        None
    }

    /// Flatten the entire tree into rendering order.
    pub fn flatten(&self) -> Vec<(u32, &HierarchyNode)> {
        let mut result = Vec::new();
        for root in &self.roots {
            result.extend(root.flatten());
        }
        result
    }

    /// Get the total number of entities in the hierarchy.
    pub fn entity_count(&self) -> usize {
        self.roots.iter().map(|r| r.count()).sum()
    }

    /// Render the hierarchy panel, producing render items for the UI layer.
    pub fn render(&self) -> Vec<HierarchyRenderItem> {
        let mut items = Vec::new();
        let flat = self.flatten();
        let search_active = !self.search_query.is_empty();
        let search_results = if search_active {
            self.search_results()
        } else {
            Vec::new()
        };

        for (depth, node) in &flat {
            // Filter by search.
            if search_active && !search_results.contains(&node.entity_id) {
                continue;
            }

            items.push(HierarchyRenderItem {
                entity_id: node.entity_id,
                name: node.name.clone(),
                depth: *depth,
                expanded: node.expanded,
                has_children: !node.children.is_empty(),
                visible: node.visible,
                locked: node.locked,
                selected: self.is_selected(node.entity_id),
                renaming: self.renaming == Some(node.entity_id),
                drag_target: self
                    .drag_state
                    .as_ref()
                    .map(|d| d.drop_target == Some(node.entity_id))
                    .unwrap_or(false),
            });
        }

        items
    }
}

/// A single item emitted for rendering by the hierarchy panel.
#[derive(Debug, Clone)]
pub struct HierarchyRenderItem {
    pub entity_id: Uuid,
    pub name: String,
    pub depth: u32,
    pub expanded: bool,
    pub has_children: bool,
    pub visible: bool,
    pub locked: bool,
    pub selected: bool,
    pub renaming: bool,
    pub drag_target: bool,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn remove_child_recursive(parent: &mut HierarchyNode, entity_id: Uuid) -> bool {
    if let Some(idx) = parent.children.iter().position(|c| c.entity_id == entity_id) {
        parent.children.remove(idx);
        return true;
    }
    for child in &mut parent.children {
        if remove_child_recursive(child, entity_id) {
            return true;
        }
    }
    false
}

fn extract_child_recursive(parent: &mut HierarchyNode, entity_id: Uuid) -> Option<HierarchyNode> {
    if let Some(idx) = parent.children.iter().position(|c| c.entity_id == entity_id) {
        return Some(parent.children.remove(idx));
    }
    for child in &mut parent.children {
        if let Some(found) = extract_child_recursive(child, entity_id) {
            return Some(found);
        }
    }
    None
}

fn deep_clone_node(node: &HierarchyNode) -> HierarchyNode {
    let mut clone = HierarchyNode {
        entity_id: Uuid::new_v4(),
        name: format!("{} (Copy)", node.name),
        expanded: node.expanded,
        visible: node.visible,
        locked: node.locked,
        children: Vec::new(),
        depth: node.depth,
    };
    for child in &node.children {
        clone.children.push(deep_clone_node(child));
    }
    clone
}

fn set_visibility_recursive(node: &mut HierarchyNode, visible: bool) {
    node.visible = visible;
    for child in &mut node.children {
        set_visibility_recursive(child, visible);
    }
}

fn set_expanded_recursive(node: &mut HierarchyNode, expanded: bool) {
    node.expanded = expanded;
    for child in &mut node.children {
        set_expanded_recursive(child, expanded);
    }
}

fn search_recursive(node: &HierarchyNode, query: &str, results: &mut Vec<Uuid>) {
    if node.name.to_lowercase().contains(query) {
        results.push(node.entity_id);
    }
    for child in &node.children {
        search_recursive(child, query, results);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_hierarchy() -> HierarchyPanel {
        let mut panel = HierarchyPanel::new();
        let root_id = panel.create_entity(None, "Root");
        let child1_id = panel.create_entity(Some(root_id), "Child1");
        let _child2_id = panel.create_entity(Some(root_id), "Child2");
        let _grandchild_id = panel.create_entity(Some(child1_id), "Grandchild");
        panel
    }

    #[test]
    fn create_entity_root() {
        let mut panel = HierarchyPanel::new();
        let id = panel.create_entity(None, "TestEntity");
        assert_eq!(panel.roots.len(), 1);
        assert_eq!(panel.find_node(id).unwrap().name, "TestEntity");
    }

    #[test]
    fn create_entity_child() {
        let mut panel = HierarchyPanel::new();
        let parent_id = panel.create_entity(None, "Parent");
        let child_id = panel.create_entity(Some(parent_id), "Child");

        let parent = panel.find_node(parent_id).unwrap();
        assert_eq!(parent.children.len(), 1);
        assert_eq!(parent.children[0].entity_id, child_id);
    }

    #[test]
    fn delete_entity() {
        let mut panel = build_test_hierarchy();
        assert_eq!(panel.entity_count(), 4);

        let root_id = panel.roots[0].entity_id;
        let child1_id = panel.roots[0].children[0].entity_id;

        panel.delete_entity(child1_id);
        assert_eq!(panel.find_node(root_id).unwrap().children.len(), 1);
    }

    #[test]
    fn delete_root_entity() {
        let mut panel = HierarchyPanel::new();
        let id = panel.create_entity(None, "Root");
        panel.delete_entity(id);
        assert!(panel.roots.is_empty());
    }

    #[test]
    fn reparent_entity() {
        let mut panel = HierarchyPanel::new();
        let a = panel.create_entity(None, "A");
        let b = panel.create_entity(None, "B");

        assert_eq!(panel.roots.len(), 2);

        panel.reparent_entity(b, Some(a));
        assert_eq!(panel.roots.len(), 1);
        assert_eq!(panel.find_node(a).unwrap().children.len(), 1);
        assert_eq!(panel.find_parent(b), Some(a));
    }

    #[test]
    fn reparent_prevents_cycle() {
        let mut panel = HierarchyPanel::new();
        let parent = panel.create_entity(None, "Parent");
        let child = panel.create_entity(Some(parent), "Child");

        // Trying to reparent parent under child should be a no-op.
        panel.reparent_entity(parent, Some(child));
        // Parent should still be a root.
        assert!(panel.roots.iter().any(|r| r.entity_id == parent));
    }

    #[test]
    fn reparent_to_root() {
        let mut panel = HierarchyPanel::new();
        let parent = panel.create_entity(None, "Parent");
        let child = panel.create_entity(Some(parent), "Child");

        panel.reparent_entity(child, None);
        assert_eq!(panel.roots.len(), 2);
    }

    #[test]
    fn duplicate_entity() {
        let mut panel = build_test_hierarchy();
        let root_id = panel.roots[0].entity_id;
        let child1_id = panel.roots[0].children[0].entity_id;

        let dup_id = panel.duplicate_entity(child1_id);
        assert!(dup_id.is_some());
        let dup_id = dup_id.unwrap();

        // The duplicate should be a sibling.
        let root = panel.find_node(root_id).unwrap();
        assert_eq!(root.children.len(), 3); // Child1, Child2, Child1 (Copy)

        let dup = panel.find_node(dup_id).unwrap();
        assert!(dup.name.contains("Copy"));
    }

    #[test]
    fn rename_entity() {
        let mut panel = HierarchyPanel::new();
        let id = panel.create_entity(None, "OldName");
        panel.rename_entity(id, "NewName");
        assert_eq!(panel.find_node(id).unwrap().name, "NewName");
    }

    #[test]
    fn rename_workflow() {
        let mut panel = HierarchyPanel::new();
        let id = panel.create_entity(None, "Original");
        panel.begin_rename(id);
        assert_eq!(panel.renaming, Some(id));
        assert_eq!(panel.rename_buffer, "Original");

        panel.rename_buffer = "Renamed".to_string();
        panel.commit_rename();
        assert_eq!(panel.find_node(id).unwrap().name, "Renamed");
        assert!(panel.renaming.is_none());
    }

    #[test]
    fn toggle_visibility() {
        let mut panel = HierarchyPanel::new();
        let parent = panel.create_entity(None, "Parent");
        let child = panel.create_entity(Some(parent), "Child");

        assert!(panel.find_node(parent).unwrap().visible);
        assert!(panel.find_node(child).unwrap().visible);

        panel.toggle_visibility(parent);
        assert!(!panel.find_node(parent).unwrap().visible);
        assert!(!panel.find_node(child).unwrap().visible);
    }

    #[test]
    fn toggle_lock() {
        let mut panel = HierarchyPanel::new();
        let id = panel.create_entity(None, "Entity");
        assert!(!panel.find_node(id).unwrap().locked);
        panel.toggle_lock(id);
        assert!(panel.find_node(id).unwrap().locked);
    }

    #[test]
    fn selection_single() {
        let mut panel = HierarchyPanel::new();
        let a = panel.create_entity(None, "A");
        let b = panel.create_entity(None, "B");

        panel.select(a);
        assert!(panel.is_selected(a));
        assert!(!panel.is_selected(b));

        panel.select(b);
        assert!(!panel.is_selected(a));
        assert!(panel.is_selected(b));
    }

    #[test]
    fn selection_multi() {
        let mut panel = HierarchyPanel::new();
        let a = panel.create_entity(None, "A");
        let b = panel.create_entity(None, "B");

        panel.select(a);
        panel.add_to_selection(b);
        assert_eq!(panel.selected().len(), 2);

        panel.toggle_selection(a);
        assert_eq!(panel.selected().len(), 1);
        assert!(panel.is_selected(b));
    }

    #[test]
    fn selection_range() {
        let mut panel = HierarchyPanel::new();
        let a = panel.create_entity(None, "A");
        let b = panel.create_entity(None, "B");
        let c = panel.create_entity(None, "C");
        let d = panel.create_entity(None, "D");

        panel.select(a);
        panel.range_select(c);
        assert_eq!(panel.selected().len(), 3);
        assert!(panel.is_selected(a));
        assert!(panel.is_selected(b));
        assert!(panel.is_selected(c));
        assert!(!panel.is_selected(d));
    }

    #[test]
    fn search_filter() {
        let mut panel = build_test_hierarchy();
        let results = panel.set_search("child");
        assert_eq!(results.len(), 3); // Child1, Child2, Grandchild
    }

    #[test]
    fn search_no_results() {
        let mut panel = build_test_hierarchy();
        let results = panel.set_search("xyz");
        assert!(results.is_empty());
    }

    #[test]
    fn flatten_respects_expanded() {
        let mut panel = HierarchyPanel::new();
        let root = panel.create_entity(None, "Root");
        let _child = panel.create_entity(Some(root), "Child");

        let flat_expanded = panel.flatten();
        assert_eq!(flat_expanded.len(), 2);

        panel.find_node_mut(root).unwrap().expanded = false;
        let flat_collapsed = panel.flatten();
        assert_eq!(flat_collapsed.len(), 1);
    }

    #[test]
    fn context_menu_create() {
        let mut panel = HierarchyPanel::new();
        panel.open_context_menu(None);
        assert!(panel.context_menu_open);

        let result = panel.execute_context_action(HierarchyContextAction::CreateEmpty);
        assert!(result.is_some());
        assert_eq!(panel.roots.len(), 1);
        assert!(!panel.context_menu_open);
    }

    #[test]
    fn context_menu_copy_paste() {
        let mut panel = HierarchyPanel::new();
        let id = panel.create_entity(None, "Original");

        panel.open_context_menu(Some(id));
        panel.execute_context_action(HierarchyContextAction::CopyEntity);
        assert!(panel.clipboard.is_some());

        panel.open_context_menu(None);
        let pasted = panel.execute_context_action(HierarchyContextAction::PasteEntity);
        assert!(pasted.is_some());
        assert_eq!(panel.roots.len(), 2);
    }

    #[test]
    fn drag_reparent() {
        let mut panel = HierarchyPanel::new();
        let a = panel.create_entity(None, "A");
        let b = panel.create_entity(None, "B");

        panel.begin_drag(b);
        assert!(panel.is_dragging());

        panel.update_drag(Some(a), DropPosition::Inside);
        let result = panel.end_drag();
        assert!(result.is_some());
        assert_eq!(panel.roots.len(), 1);
        assert_eq!(panel.find_parent(b), Some(a));
    }

    #[test]
    fn delete_selected() {
        let mut panel = HierarchyPanel::new();
        let a = panel.create_entity(None, "A");
        let b = panel.create_entity(None, "B");
        let c = panel.create_entity(None, "C");

        panel.add_to_selection(a);
        panel.add_to_selection(b);
        panel.delete_selected();
        assert_eq!(panel.roots.len(), 1);
        assert_eq!(panel.roots[0].entity_id, c);
    }

    #[test]
    fn render_produces_items() {
        let panel = build_test_hierarchy();
        let items = panel.render();
        assert_eq!(items.len(), 4);
        assert_eq!(items[0].depth, 0);
        assert_eq!(items[1].depth, 1);
    }

    #[test]
    fn entity_count() {
        let panel = build_test_hierarchy();
        assert_eq!(panel.entity_count(), 4);
    }

    #[test]
    fn find_parent() {
        let panel = build_test_hierarchy();
        let root_id = panel.roots[0].entity_id;
        let child1_id = panel.roots[0].children[0].entity_id;
        let gc_id = panel.roots[0].children[0].children[0].entity_id;

        assert_eq!(panel.find_parent(child1_id), Some(root_id));
        assert_eq!(panel.find_parent(gc_id), Some(child1_id));
        assert_eq!(panel.find_parent(root_id), None);
    }

    #[test]
    fn select_children() {
        let mut panel = build_test_hierarchy();
        let root_id = panel.roots[0].entity_id;
        panel.select_children(root_id);
        assert_eq!(panel.selected().len(), 4);
    }

    #[test]
    fn expand_collapse_all() {
        let mut panel = build_test_hierarchy();
        let root_id = panel.roots[0].entity_id;

        panel.open_context_menu(Some(root_id));
        panel.execute_context_action(HierarchyContextAction::CollapseAll);
        assert!(!panel.find_node(root_id).unwrap().expanded);

        panel.open_context_menu(Some(root_id));
        panel.execute_context_action(HierarchyContextAction::ExpandAll);
        assert!(panel.find_node(root_id).unwrap().expanded);
    }
}
