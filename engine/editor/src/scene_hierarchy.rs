// engine/editor/src/scene_hierarchy_v2.rs
//
// Enhanced scene hierarchy panel: drag-drop reparenting, multi-select,
// search/filter, visibility/lock toggles, context menu, inline rename,
// expand/collapse, and tree navigation.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Entity handle
// ---------------------------------------------------------------------------

/// Entity handle for the hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

impl EntityId {
    pub const ROOT: Self = Self(0);
    pub const INVALID: Self = Self(u64::MAX);

    pub fn is_root(&self) -> bool { *self == Self::ROOT }
    pub fn is_valid(&self) -> bool { *self != Self::INVALID }
}

// ---------------------------------------------------------------------------
// Hierarchy node
// ---------------------------------------------------------------------------

/// Data for one entity in the hierarchy.
#[derive(Debug, Clone)]
pub struct HierarchyNode {
    pub entity: EntityId,
    pub name: String,
    pub parent: EntityId,
    pub children: Vec<EntityId>,
    pub visible: bool,
    pub locked: bool,
    pub expanded: bool,
    pub depth: u32,
    pub icon: EntityIcon,
    pub tags: Vec<String>,
    /// User-defined type hint (e.g. "Camera", "Light", "Mesh").
    pub type_hint: String,
    /// Whether this entity is a prefab instance root.
    pub is_prefab: bool,
    /// Sort order within siblings.
    pub sibling_index: u32,
}

/// Icon for different entity types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityIcon {
    Default,
    Camera,
    Light,
    Mesh,
    Sprite,
    Audio,
    Particle,
    UI,
    Script,
    Prefab,
    Folder,
    Terrain,
}

impl HierarchyNode {
    pub fn new(entity: EntityId, name: &str) -> Self {
        Self {
            entity,
            name: name.to_string(),
            parent: EntityId::ROOT,
            children: Vec::new(),
            visible: true,
            locked: false,
            expanded: true,
            depth: 0,
            icon: EntityIcon::Default,
            tags: Vec::new(),
            type_hint: String::new(),
            is_prefab: false,
            sibling_index: 0,
        }
    }

    pub fn with_parent(mut self, parent: EntityId) -> Self {
        self.parent = parent;
        self
    }

    pub fn with_icon(mut self, icon: EntityIcon) -> Self {
        self.icon = icon;
        self
    }

    pub fn with_type_hint(mut self, hint: &str) -> Self {
        self.type_hint = hint.to_string();
        self
    }

    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// Multi-select state.
#[derive(Debug, Clone, Default)]
pub struct HierarchySelection {
    selected: Vec<EntityId>,
    primary: Option<EntityId>,
    anchor: Option<EntityId>,
}

impl HierarchySelection {
    pub fn new() -> Self { Self::default() }

    /// Select a single entity (clear existing selection).
    pub fn select(&mut self, entity: EntityId) {
        self.selected.clear();
        self.selected.push(entity);
        self.primary = Some(entity);
        self.anchor = Some(entity);
    }

    /// Toggle selection of an entity (Ctrl+click).
    pub fn toggle(&mut self, entity: EntityId) {
        if let Some(pos) = self.selected.iter().position(|&e| e == entity) {
            self.selected.remove(pos);
            if self.primary == Some(entity) {
                self.primary = self.selected.last().copied();
            }
        } else {
            self.selected.push(entity);
            self.primary = Some(entity);
        }
    }

    /// Range select (Shift+click): select everything between anchor and entity.
    pub fn range_select(&mut self, entity: EntityId, flat_order: &[EntityId]) {
        let anchor = self.anchor.unwrap_or(entity);
        let anchor_pos = flat_order.iter().position(|&e| e == anchor).unwrap_or(0);
        let entity_pos = flat_order.iter().position(|&e| e == entity).unwrap_or(0);

        let (start, end) = if anchor_pos <= entity_pos {
            (anchor_pos, entity_pos)
        } else {
            (entity_pos, anchor_pos)
        };

        self.selected.clear();
        for i in start..=end {
            self.selected.push(flat_order[i]);
        }
        self.primary = Some(entity);
    }

    /// Select all entities.
    pub fn select_all(&mut self, entities: &[EntityId]) {
        self.selected = entities.to_vec();
        self.primary = entities.last().copied();
    }

    /// Clear selection.
    pub fn clear(&mut self) {
        self.selected.clear();
        self.primary = None;
        self.anchor = None;
    }

    /// Is this entity selected?
    pub fn is_selected(&self, entity: EntityId) -> bool {
        self.selected.contains(&entity)
    }

    /// Get all selected entities.
    pub fn selected(&self) -> &[EntityId] {
        &self.selected
    }

    /// Get the primary (most recently clicked) selected entity.
    pub fn primary(&self) -> Option<EntityId> {
        self.primary
    }

    /// Number of selected entities.
    pub fn count(&self) -> usize {
        self.selected.len()
    }

    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

/// Filter configuration for the hierarchy.
#[derive(Debug, Clone, Default)]
pub struct HierarchyFilter {
    pub search_text: String,
    pub type_filter: Option<String>,
    pub tag_filter: Option<String>,
    pub show_hidden: bool,
    pub show_locked_only: bool,
}

impl HierarchyFilter {
    pub fn is_active(&self) -> bool {
        !self.search_text.is_empty()
            || self.type_filter.is_some()
            || self.tag_filter.is_some()
            || self.show_locked_only
    }

    pub fn matches(&self, node: &HierarchyNode) -> bool {
        if !self.show_hidden && !node.visible {
            return false;
        }
        if self.show_locked_only && !node.locked {
            return false;
        }
        if !self.search_text.is_empty() {
            let lower_search = self.search_text.to_lowercase();
            let lower_name = node.name.to_lowercase();
            if !lower_name.contains(&lower_search) {
                return false;
            }
        }
        if let Some(ref type_filter) = self.type_filter {
            if !node.type_hint.eq_ignore_ascii_case(type_filter) {
                return false;
            }
        }
        if let Some(ref tag_filter) = self.tag_filter {
            if !node.tags.iter().any(|t| t.eq_ignore_ascii_case(tag_filter)) {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Context menu
// ---------------------------------------------------------------------------

/// Context menu actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextMenuAction {
    CreateEmpty,
    CreateChild,
    Duplicate,
    Delete,
    Rename,
    CopyPath,
    ToggleVisible,
    ToggleLock,
    ExpandAll,
    CollapseAll,
    SelectChildren,
    FocusInViewport,
    CreatePrefab,
    UnpackPrefab,
    MoveUp,
    MoveDown,
    MoveToTop,
    MoveToBottom,
    Group,
    Ungroup,
}

/// A context menu item.
#[derive(Debug, Clone)]
pub struct ContextMenuItem {
    pub action: ContextMenuAction,
    pub label: String,
    pub shortcut: Option<String>,
    pub enabled: bool,
    pub separator_before: bool,
}

impl ContextMenuItem {
    pub fn new(action: ContextMenuAction, label: &str) -> Self {
        Self {
            action,
            label: label.to_string(),
            shortcut: None,
            enabled: true,
            separator_before: false,
        }
    }

    pub fn with_shortcut(mut self, shortcut: &str) -> Self {
        self.shortcut = Some(shortcut.to_string());
        self
    }

    pub fn with_separator(mut self) -> Self {
        self.separator_before = true;
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

// ---------------------------------------------------------------------------
// Drag-drop
// ---------------------------------------------------------------------------

/// Drag-drop state.
#[derive(Debug, Clone)]
pub struct DragDropState {
    pub dragging: Vec<EntityId>,
    pub drop_target: Option<EntityId>,
    pub drop_position: DropPosition,
    pub is_active: bool,
}

/// Where to drop relative to the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPosition {
    /// Drop as a child of the target.
    Inside,
    /// Drop above the target (as a sibling).
    Above,
    /// Drop below the target (as a sibling).
    Below,
}

impl DragDropState {
    pub fn new() -> Self {
        Self {
            dragging: Vec::new(),
            drop_target: None,
            drop_position: DropPosition::Inside,
            is_active: false,
        }
    }

    pub fn start_drag(&mut self, entities: Vec<EntityId>) {
        self.dragging = entities;
        self.is_active = true;
    }

    pub fn update_target(&mut self, target: EntityId, position: DropPosition) {
        self.drop_target = Some(target);
        self.drop_position = position;
    }

    pub fn cancel(&mut self) {
        self.dragging.clear();
        self.drop_target = None;
        self.is_active = false;
    }

    /// Check if dropping would create a cycle (child becoming parent of ancestor).
    pub fn would_create_cycle(&self, hierarchy: &SceneHierarchy) -> bool {
        if let Some(target) = self.drop_target {
            for &dragged in &self.dragging {
                if hierarchy.is_ancestor_of(dragged, target) {
                    return true;
                }
            }
        }
        false
    }
}

impl Default for DragDropState {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Rename state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RenameState {
    pub entity: EntityId,
    pub current_text: String,
    pub is_active: bool,
}

impl RenameState {
    pub fn new() -> Self {
        Self {
            entity: EntityId::INVALID,
            current_text: String::new(),
            is_active: false,
        }
    }

    pub fn start(&mut self, entity: EntityId, current_name: &str) {
        self.entity = entity;
        self.current_text = current_name.to_string();
        self.is_active = true;
    }

    pub fn cancel(&mut self) {
        self.is_active = false;
        self.entity = EntityId::INVALID;
    }

    pub fn confirm(&mut self) -> Option<(EntityId, String)> {
        if self.is_active {
            self.is_active = false;
            let entity = self.entity;
            self.entity = EntityId::INVALID;
            Some((entity, std::mem::take(&mut self.current_text)))
        } else {
            None
        }
    }
}

impl Default for RenameState {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Scene hierarchy events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum HierarchyEvent {
    EntitySelected { entity: EntityId },
    EntityDeselected { entity: EntityId },
    EntityRenamed { entity: EntityId, old_name: String, new_name: String },
    EntityReparented { entity: EntityId, old_parent: EntityId, new_parent: EntityId },
    EntityCreated { entity: EntityId, parent: EntityId },
    EntityDeleted { entity: EntityId },
    EntityDuplicated { source: EntityId, duplicate: EntityId },
    VisibilityChanged { entity: EntityId, visible: bool },
    LockChanged { entity: EntityId, locked: bool },
    SiblingOrderChanged { entity: EntityId, new_index: u32 },
}

// ---------------------------------------------------------------------------
// SceneHierarchy
// ---------------------------------------------------------------------------

/// The main scene hierarchy data structure.
pub struct SceneHierarchy {
    nodes: HashMap<u64, HierarchyNode>,
    root_children: Vec<EntityId>,
    selection: HierarchySelection,
    filter: HierarchyFilter,
    drag_drop: DragDropState,
    rename: RenameState,
    events: Vec<HierarchyEvent>,
    next_entity_id: u64,
    flat_cache: Vec<EntityId>,
    flat_cache_dirty: bool,
}

impl SceneHierarchy {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            root_children: Vec::new(),
            selection: HierarchySelection::new(),
            filter: HierarchyFilter::default(),
            drag_drop: DragDropState::new(),
            rename: RenameState::new(),
            events: Vec::new(),
            next_entity_id: 1,
            flat_cache: Vec::new(),
            flat_cache_dirty: true,
        }
    }

    // -----------------------------------------------------------------------
    // Node operations
    // -----------------------------------------------------------------------

    /// Add an entity to the hierarchy.
    pub fn add_entity(&mut self, node: HierarchyNode) {
        let entity = node.entity;
        let parent = node.parent;

        self.nodes.insert(entity.0, node);

        if parent.is_root() {
            self.root_children.push(entity);
        } else {
            if let Some(parent_node) = self.nodes.get_mut(&parent.0) {
                parent_node.children.push(entity);
            }
        }

        self.flat_cache_dirty = true;
        self.events.push(HierarchyEvent::EntityCreated { entity, parent });
    }

    /// Create a new entity with auto-generated ID.
    pub fn create_entity(&mut self, name: &str, parent: EntityId) -> EntityId {
        let id = EntityId(self.next_entity_id);
        self.next_entity_id += 1;

        let depth = if parent.is_root() {
            0
        } else {
            self.nodes.get(&parent.0).map(|n| n.depth + 1).unwrap_or(0)
        };

        let mut node = HierarchyNode::new(id, name).with_parent(parent);
        node.depth = depth;
        self.add_entity(node);
        id
    }

    /// Remove an entity and all its descendants.
    pub fn remove_entity(&mut self, entity: EntityId) {
        // Collect all descendants first.
        let descendants = self.collect_descendants(entity);

        // Remove from parent's children list.
        if let Some(node) = self.nodes.get(&entity.0) {
            let parent = node.parent;
            if parent.is_root() {
                self.root_children.retain(|&e| e != entity);
            } else if let Some(parent_node) = self.nodes.get_mut(&parent.0) {
                parent_node.children.retain(|&e| e != entity);
            }
        }

        // Remove all descendants.
        for &desc in &descendants {
            self.nodes.remove(&desc.0);
            self.selection.selected.retain(|&e| e != desc);
        }
        self.nodes.remove(&entity.0);
        self.selection.selected.retain(|&e| e != entity);

        self.flat_cache_dirty = true;
        self.events.push(HierarchyEvent::EntityDeleted { entity });
    }

    /// Reparent an entity.
    pub fn reparent(&mut self, entity: EntityId, new_parent: EntityId) {
        if entity == new_parent { return; }
        if self.is_ancestor_of(entity, new_parent) { return; }

        let old_parent = self.nodes.get(&entity.0).map(|n| n.parent).unwrap_or(EntityId::ROOT);

        // Remove from old parent.
        if old_parent.is_root() {
            self.root_children.retain(|&e| e != entity);
        } else if let Some(old_p) = self.nodes.get_mut(&old_parent.0) {
            old_p.children.retain(|&e| e != entity);
        }

        // Add to new parent.
        if new_parent.is_root() {
            self.root_children.push(entity);
        } else if let Some(new_p) = self.nodes.get_mut(&new_parent.0) {
            new_p.children.push(entity);
        }

        // Update node's parent.
        if let Some(node) = self.nodes.get_mut(&entity.0) {
            node.parent = new_parent;
        }

        // Update depths.
        self.update_depths(entity);

        self.flat_cache_dirty = true;
        self.events.push(HierarchyEvent::EntityReparented {
            entity,
            old_parent,
            new_parent,
        });
    }

    /// Rename an entity.
    pub fn rename_entity(&mut self, entity: EntityId, new_name: &str) {
        if let Some(node) = self.nodes.get_mut(&entity.0) {
            let old_name = std::mem::replace(&mut node.name, new_name.to_string());
            self.events.push(HierarchyEvent::EntityRenamed {
                entity,
                old_name,
                new_name: new_name.to_string(),
            });
        }
    }

    /// Toggle visibility.
    pub fn toggle_visibility(&mut self, entity: EntityId) {
        if let Some(node) = self.nodes.get_mut(&entity.0) {
            node.visible = !node.visible;
            let visible = node.visible;
            self.events.push(HierarchyEvent::VisibilityChanged { entity, visible });
        }
    }

    /// Toggle lock.
    pub fn toggle_lock(&mut self, entity: EntityId) {
        if let Some(node) = self.nodes.get_mut(&entity.0) {
            node.locked = !node.locked;
            let locked = node.locked;
            self.events.push(HierarchyEvent::LockChanged { entity, locked });
        }
    }

    /// Toggle expand/collapse.
    pub fn toggle_expand(&mut self, entity: EntityId) {
        if let Some(node) = self.nodes.get_mut(&entity.0) {
            node.expanded = !node.expanded;
            self.flat_cache_dirty = true;
        }
    }

    /// Expand all nodes.
    pub fn expand_all(&mut self) {
        for node in self.nodes.values_mut() {
            node.expanded = true;
        }
        self.flat_cache_dirty = true;
    }

    /// Collapse all nodes.
    pub fn collapse_all(&mut self) {
        for node in self.nodes.values_mut() {
            node.expanded = false;
        }
        self.flat_cache_dirty = true;
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    pub fn get_node(&self, entity: EntityId) -> Option<&HierarchyNode> {
        self.nodes.get(&entity.0)
    }

    pub fn get_node_mut(&mut self, entity: EntityId) -> Option<&mut HierarchyNode> {
        self.nodes.get_mut(&entity.0)
    }

    pub fn entity_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_children(&self) -> &[EntityId] {
        &self.root_children
    }

    pub fn selection(&self) -> &HierarchySelection {
        &self.selection
    }

    pub fn selection_mut(&mut self) -> &mut HierarchySelection {
        &mut self.selection
    }

    pub fn filter(&self) -> &HierarchyFilter {
        &self.filter
    }

    pub fn filter_mut(&mut self) -> &mut HierarchyFilter {
        &mut self.filter
    }

    pub fn drag_drop(&self) -> &DragDropState {
        &self.drag_drop
    }

    pub fn drag_drop_mut(&mut self) -> &mut DragDropState {
        &mut self.drag_drop
    }

    pub fn rename_state(&self) -> &RenameState {
        &self.rename
    }

    pub fn rename_state_mut(&mut self) -> &mut RenameState {
        &mut self.rename
    }

    /// Check if `ancestor` is an ancestor of `descendant`.
    pub fn is_ancestor_of(&self, ancestor: EntityId, descendant: EntityId) -> bool {
        let mut current = descendant;
        loop {
            if current == ancestor { return true; }
            match self.nodes.get(&current.0) {
                Some(node) => {
                    if node.parent.is_root() { return false; }
                    current = node.parent;
                }
                None => return false,
            }
        }
    }

    /// Get the full path of an entity (e.g. "Root/Parent/Child").
    pub fn entity_path(&self, entity: EntityId) -> String {
        let mut parts = Vec::new();
        let mut current = entity;
        loop {
            match self.nodes.get(&current.0) {
                Some(node) => {
                    parts.push(node.name.clone());
                    if node.parent.is_root() { break; }
                    current = node.parent;
                }
                None => break,
            }
        }
        parts.reverse();
        parts.join("/")
    }

    /// Find entity by name (first match).
    pub fn find_by_name(&self, name: &str) -> Option<EntityId> {
        self.nodes.values()
            .find(|n| n.name == name)
            .map(|n| n.entity)
    }

    /// Collect all descendants of an entity.
    pub fn collect_descendants(&self, entity: EntityId) -> Vec<EntityId> {
        let mut result = Vec::new();
        let mut stack = VecDeque::new();
        if let Some(node) = self.nodes.get(&entity.0) {
            for &child in &node.children {
                stack.push_back(child);
            }
        }
        while let Some(current) = stack.pop_front() {
            result.push(current);
            if let Some(node) = self.nodes.get(&current.0) {
                for &child in &node.children {
                    stack.push_back(child);
                }
            }
        }
        result
    }

    /// Get the flattened visible list (for rendering in the panel).
    pub fn flat_list(&mut self) -> &[EntityId] {
        if self.flat_cache_dirty {
            self.rebuild_flat_cache();
            self.flat_cache_dirty = false;
        }
        &self.flat_cache
    }

    fn rebuild_flat_cache(&mut self) {
        self.flat_cache.clear();
        let filter_active = self.filter.is_active();
        for &root_child in &self.root_children.clone() {
            self.flatten_node(root_child, filter_active);
        }
    }

    fn flatten_node(&mut self, entity: EntityId, filter_active: bool) {
        let node = match self.nodes.get(&entity.0) {
            Some(n) => n.clone(),
            None => return,
        };

        if filter_active && !self.filter.matches(&node) {
            // If filter is active but doesn't match, still check children.
            for &child in &node.children {
                self.flatten_node(child, filter_active);
            }
            return;
        }

        self.flat_cache.push(entity);

        if node.expanded || filter_active {
            for &child in &node.children {
                self.flatten_node(child, filter_active);
            }
        }
    }

    fn update_depths(&mut self, entity: EntityId) {
        let depth = if let Some(node) = self.nodes.get(&entity.0) {
            if node.parent.is_root() {
                0
            } else {
                self.nodes.get(&node.parent.0).map(|p| p.depth + 1).unwrap_or(0)
            }
        } else {
            return;
        };

        if let Some(node) = self.nodes.get_mut(&entity.0) {
            node.depth = depth;
        }

        let children: Vec<EntityId> = self.nodes.get(&entity.0)
            .map(|n| n.children.clone())
            .unwrap_or_default();
        for child in children {
            self.update_depths(child);
        }
    }

    // -----------------------------------------------------------------------
    // Context menu
    // -----------------------------------------------------------------------

    /// Build context menu items for the current selection.
    pub fn build_context_menu(&self) -> Vec<ContextMenuItem> {
        let has_selection = !self.selection.is_empty();
        let single_selection = self.selection.count() == 1;

        let mut items = vec![
            ContextMenuItem::new(ContextMenuAction::CreateEmpty, "Create Empty"),
            ContextMenuItem::new(ContextMenuAction::CreateChild, "Create Child")
                .with_shortcut("Ctrl+Shift+N"),
        ];

        if has_selection {
            items.push(ContextMenuItem::new(ContextMenuAction::Duplicate, "Duplicate")
                .with_shortcut("Ctrl+D")
                .with_separator());
            items.push(ContextMenuItem::new(ContextMenuAction::Delete, "Delete")
                .with_shortcut("Delete"));

            if single_selection {
                items.push(ContextMenuItem::new(ContextMenuAction::Rename, "Rename")
                    .with_shortcut("F2")
                    .with_separator());
                items.push(ContextMenuItem::new(ContextMenuAction::CopyPath, "Copy Path"));
                items.push(ContextMenuItem::new(ContextMenuAction::FocusInViewport, "Focus")
                    .with_shortcut("F"));
            }

            items.push(ContextMenuItem::new(ContextMenuAction::ToggleVisible, "Toggle Visible")
                .with_shortcut("H")
                .with_separator());
            items.push(ContextMenuItem::new(ContextMenuAction::ToggleLock, "Toggle Lock")
                .with_shortcut("L"));

            items.push(ContextMenuItem::new(ContextMenuAction::MoveUp, "Move Up")
                .with_separator());
            items.push(ContextMenuItem::new(ContextMenuAction::MoveDown, "Move Down"));

            if self.selection.count() > 1 {
                items.push(ContextMenuItem::new(ContextMenuAction::Group, "Group Selection")
                    .with_shortcut("Ctrl+G")
                    .with_separator());
            }
        }

        items.push(ContextMenuItem::new(ContextMenuAction::ExpandAll, "Expand All")
            .with_separator());
        items.push(ContextMenuItem::new(ContextMenuAction::CollapseAll, "Collapse All"));

        items
    }

    /// Execute a context menu action.
    pub fn execute_action(&mut self, action: ContextMenuAction) {
        match action {
            ContextMenuAction::CreateEmpty => {
                self.create_entity("New Entity", EntityId::ROOT);
            }
            ContextMenuAction::CreateChild => {
                if let Some(parent) = self.selection.primary() {
                    self.create_entity("New Child", parent);
                }
            }
            ContextMenuAction::Delete => {
                let to_delete: Vec<EntityId> = self.selection.selected().to_vec();
                for entity in to_delete {
                    self.remove_entity(entity);
                }
            }
            ContextMenuAction::Rename => {
                if let Some(entity) = self.selection.primary() {
                    let name = self.nodes.get(&entity.0)
                        .map(|n| n.name.clone())
                        .unwrap_or_default();
                    self.rename.start(entity, &name);
                }
            }
            ContextMenuAction::ToggleVisible => {
                let selected: Vec<EntityId> = self.selection.selected().to_vec();
                for entity in selected {
                    self.toggle_visibility(entity);
                }
            }
            ContextMenuAction::ToggleLock => {
                let selected: Vec<EntityId> = self.selection.selected().to_vec();
                for entity in selected {
                    self.toggle_lock(entity);
                }
            }
            ContextMenuAction::ExpandAll => self.expand_all(),
            ContextMenuAction::CollapseAll => self.collapse_all(),
            ContextMenuAction::SelectChildren => {
                if let Some(entity) = self.selection.primary() {
                    let children = self.collect_descendants(entity);
                    self.selection.select_all(&children);
                }
            }
            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // Drag-drop execution
    // -----------------------------------------------------------------------

    /// Execute the current drag-drop operation.
    pub fn execute_drag_drop(&mut self) {
        if !self.drag_drop.is_active { return; }
        if self.drag_drop.would_create_cycle(self) { return; }

        let target = match self.drag_drop.drop_target {
            Some(t) => t,
            None => { self.drag_drop.cancel(); return; }
        };

        let entities: Vec<EntityId> = self.drag_drop.dragging.clone();
        let position = self.drag_drop.drop_position;

        match position {
            DropPosition::Inside => {
                for &entity in &entities {
                    self.reparent(entity, target);
                }
            }
            DropPosition::Above | DropPosition::Below => {
                let parent = self.nodes.get(&target.0)
                    .map(|n| n.parent)
                    .unwrap_or(EntityId::ROOT);
                for &entity in &entities {
                    self.reparent(entity, parent);
                }
            }
        }

        self.drag_drop.cancel();
    }

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    pub fn drain_events(&mut self) -> Vec<HierarchyEvent> {
        std::mem::take(&mut self.events)
    }
}

impl Default for SceneHierarchy {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_remove() {
        let mut h = SceneHierarchy::new();
        let e1 = h.create_entity("Player", EntityId::ROOT);
        let e2 = h.create_entity("Enemy", EntityId::ROOT);
        assert_eq!(h.entity_count(), 2);

        h.remove_entity(e1);
        assert_eq!(h.entity_count(), 1);
        assert!(h.get_node(e2).is_some());
    }

    #[test]
    fn test_reparent() {
        let mut h = SceneHierarchy::new();
        let parent = h.create_entity("Parent", EntityId::ROOT);
        let child = h.create_entity("Child", EntityId::ROOT);
        h.reparent(child, parent);

        let parent_node = h.get_node(parent).unwrap();
        assert!(parent_node.children.contains(&child));
        assert_eq!(h.get_node(child).unwrap().parent, parent);
    }

    #[test]
    fn test_cycle_prevention() {
        let mut h = SceneHierarchy::new();
        let a = h.create_entity("A", EntityId::ROOT);
        let b = h.create_entity("B", a);
        // Trying to make A a child of B would create a cycle.
        assert!(h.is_ancestor_of(a, b));
    }

    #[test]
    fn test_selection() {
        let mut sel = HierarchySelection::new();
        sel.select(EntityId(1));
        assert!(sel.is_selected(EntityId(1)));
        assert_eq!(sel.count(), 1);

        sel.toggle(EntityId(2));
        assert_eq!(sel.count(), 2);

        sel.toggle(EntityId(1));
        assert_eq!(sel.count(), 1);
        assert!(sel.is_selected(EntityId(2)));
    }

    #[test]
    fn test_filter() {
        let mut h = SceneHierarchy::new();
        h.create_entity("Player", EntityId::ROOT);
        h.create_entity("Enemy", EntityId::ROOT);
        h.create_entity("PlayerCamera", EntityId::ROOT);

        h.filter_mut().search_text = "Player".to_string();
        let flat = h.flat_list().to_vec();
        assert_eq!(flat.len(), 2); // Player and PlayerCamera
    }

    #[test]
    fn test_entity_path() {
        let mut h = SceneHierarchy::new();
        let a = h.create_entity("Root", EntityId::ROOT);
        let b = h.create_entity("Child", a);
        let c = h.create_entity("Grandchild", b);
        assert_eq!(h.entity_path(c), "Root/Child/Grandchild");
    }

    #[test]
    fn test_rename() {
        let mut h = SceneHierarchy::new();
        let e = h.create_entity("OldName", EntityId::ROOT);
        h.rename_entity(e, "NewName");
        assert_eq!(h.get_node(e).unwrap().name, "NewName");
    }

    #[test]
    fn test_visibility_lock() {
        let mut h = SceneHierarchy::new();
        let e = h.create_entity("Test", EntityId::ROOT);
        assert!(h.get_node(e).unwrap().visible);
        h.toggle_visibility(e);
        assert!(!h.get_node(e).unwrap().visible);
        h.toggle_lock(e);
        assert!(h.get_node(e).unwrap().locked);
    }
}
