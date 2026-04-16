//! UI core types: the node tree, event system, and fundamental data structures.
//!
//! The [`UITree`] is an arena-allocated tree of [`UINode`]s, conceptually
//! similar to a scene graph but specialised for 2-D user interfaces. Each node
//! carries layout parameters, visibility flags, and a z-order that drives
//! rendering and hit-testing.

use std::collections::HashMap;

use glam::Vec2;
use serde::{Deserialize, Serialize};

use genovo_core::Rect;

// ---------------------------------------------------------------------------
// UIId
// ---------------------------------------------------------------------------

/// Unique identifier for a UI node within a [`UITree`].
///
/// Internally a generational index so that stale references are detected
/// cheaply at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UIId {
    /// Slot index in the arena.
    pub index: u32,
    /// Generation counter.
    pub generation: u32,
}

impl UIId {
    /// Creates a new identifier.
    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Sentinel value that does not refer to any valid node.
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: u32::MAX,
    };

    /// Returns `true` if this is the [`INVALID`](Self::INVALID) sentinel.
    pub fn is_invalid(&self) -> bool {
        *self == Self::INVALID
    }
}

impl Default for UIId {
    fn default() -> Self {
        Self::INVALID
    }
}

impl std::fmt::Display for UIId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UIId({}/{})", self.index, self.generation)
    }
}

// ---------------------------------------------------------------------------
// Anchor / Margin / Padding
// ---------------------------------------------------------------------------

/// Describes how a node is anchored relative to its parent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    /// Anchor point on the parent (0.0 = left/top, 1.0 = right/bottom).
    pub min: Vec2,
    /// Anchor point on the parent (0.0 = left/top, 1.0 = right/bottom).
    pub max: Vec2,
}

impl Anchor {
    pub const TOP_LEFT: Self = Self {
        min: Vec2::ZERO,
        max: Vec2::ZERO,
    };
    pub const TOP_RIGHT: Self = Self {
        min: Vec2::new(1.0, 0.0),
        max: Vec2::new(1.0, 0.0),
    };
    pub const BOTTOM_LEFT: Self = Self {
        min: Vec2::new(0.0, 1.0),
        max: Vec2::new(0.0, 1.0),
    };
    pub const BOTTOM_RIGHT: Self = Self {
        min: Vec2::ONE,
        max: Vec2::ONE,
    };
    pub const CENTER: Self = Self {
        min: Vec2::new(0.5, 0.5),
        max: Vec2::new(0.5, 0.5),
    };
    pub const STRETCH: Self = Self {
        min: Vec2::ZERO,
        max: Vec2::ONE,
    };
    pub const TOP_STRETCH: Self = Self {
        min: Vec2::ZERO,
        max: Vec2::new(1.0, 0.0),
    };
    pub const BOTTOM_STRETCH: Self = Self {
        min: Vec2::new(0.0, 1.0),
        max: Vec2::ONE,
    };
    pub const LEFT_STRETCH: Self = Self {
        min: Vec2::ZERO,
        max: Vec2::new(0.0, 1.0),
    };
    pub const RIGHT_STRETCH: Self = Self {
        min: Vec2::new(1.0, 0.0),
        max: Vec2::ONE,
    };
}

impl Default for Anchor {
    fn default() -> Self {
        Self::TOP_LEFT
    }
}

/// Edge offsets (left, top, right, bottom) used for margins.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Margin {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl Margin {
    pub const ZERO: Self = Self {
        left: 0.0,
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
    };

    pub fn all(value: f32) -> Self {
        Self {
            left: value,
            top: value,
            right: value,
            bottom: value,
        }
    }

    pub fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Self {
            left: horizontal,
            top: vertical,
            right: horizontal,
            bottom: vertical,
        }
    }

    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// Total horizontal edge size.
    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    /// Total vertical edge size.
    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

impl Default for Margin {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Edge offsets used for padding (same structure as Margin, separate type for
/// clarity).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Padding {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl Padding {
    pub const ZERO: Self = Self {
        left: 0.0,
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
    };

    pub fn all(value: f32) -> Self {
        Self {
            left: value,
            top: value,
            right: value,
            bottom: value,
        }
    }

    pub fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Self {
            left: horizontal,
            top: vertical,
            right: horizontal,
            bottom: vertical,
        }
    }

    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

impl Default for Padding {
    fn default() -> Self {
        Self::ZERO
    }
}

// ---------------------------------------------------------------------------
// UIEvent
// ---------------------------------------------------------------------------

/// Key modifier flags for input events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct KeyModifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

/// Mouse button identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
}

/// Virtual key code for keyboard events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyCode {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Key0, Key1, Key2, Key3, Key4, Key5, Key6, Key7, Key8, Key9,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    Escape, Tab, CapsLock, Shift, Control, Alt, Meta,
    Space, Enter, Backspace, Delete,
    ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
    Home, End, PageUp, PageDown,
    Insert, PrintScreen, ScrollLock, Pause,
    NumLock, NumpadAdd, NumpadSubtract, NumpadMultiply, NumpadDivide, NumpadEnter,
    Numpad0, Numpad1, Numpad2, Numpad3, Numpad4,
    Numpad5, Numpad6, Numpad7, Numpad8, Numpad9,
    Minus, Equals, LeftBracket, RightBracket,
    Semicolon, Apostrophe, Backslash, Comma, Period, Slash, Grave,
}

/// Events that the UI system produces and routes to nodes.
#[derive(Debug, Clone)]
pub enum UIEvent {
    /// Mouse button pressed down over a node.
    Click {
        position: Vec2,
        button: MouseButton,
        modifiers: KeyModifiers,
    },
    /// Mouse cursor entered a node's bounds.
    Hover {
        position: Vec2,
    },
    /// Mouse cursor left a node's bounds.
    HoverEnd,
    /// Node received keyboard focus.
    Focus,
    /// Node lost keyboard focus.
    Blur,
    /// A key was pressed while a node had focus.
    KeyInput {
        key: KeyCode,
        pressed: bool,
        modifiers: KeyModifiers,
    },
    /// Text input received (for text fields).
    TextInput {
        character: char,
    },
    /// Scroll wheel moved over a node.
    Scroll {
        delta: Vec2,
        modifiers: KeyModifiers,
    },
    /// Drag started on a node.
    DragStart {
        position: Vec2,
        button: MouseButton,
    },
    /// Drag moved (cursor position updated while dragging).
    DragMove {
        position: Vec2,
        delta: Vec2,
    },
    /// Drag ended (mouse button released).
    DragEnd {
        position: Vec2,
    },
    /// An item was dropped onto this node.
    DragDrop {
        position: Vec2,
        /// Opaque payload identifier.
        payload: u64,
    },
    /// Mouse button released.
    MouseUp {
        position: Vec2,
        button: MouseButton,
    },
    /// Double click.
    DoubleClick {
        position: Vec2,
        button: MouseButton,
    },
}

// ---------------------------------------------------------------------------
// UIEventHandler
// ---------------------------------------------------------------------------

/// Trait for objects that can handle UI events.
///
/// Returns `true` if the event was consumed and should not propagate further
/// up the tree.
pub trait UIEventHandler {
    fn handle(&mut self, event: &UIEvent) -> bool;
}

// ---------------------------------------------------------------------------
// UINode
// ---------------------------------------------------------------------------

/// Layout parameters stored on every UI node. These are inputs to the layout
/// engine; the resulting computed position lands in [`UINode::computed_rect`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutParams {
    /// Preferred width.
    pub width: crate::layout::Size,
    /// Preferred height.
    pub height: crate::layout::Size,
    /// Minimum width constraint.
    pub min_width: Option<f32>,
    /// Maximum width constraint.
    pub max_width: Option<f32>,
    /// Minimum height constraint.
    pub min_height: Option<f32>,
    /// Maximum height constraint.
    pub max_height: Option<f32>,
    /// Anchor relative to parent.
    pub anchor: Anchor,
    /// Margin (outer spacing).
    pub margin: Margin,
    /// Padding (inner spacing).
    pub padding: Padding,
    /// Flex grow factor (used by FlexLayout).
    pub flex_grow: f32,
    /// Flex shrink factor.
    pub flex_shrink: f32,
    /// Flex basis (initial size before grow/shrink).
    pub flex_basis: Option<f32>,
    /// Alignment override for this child within its parent's cross axis.
    pub align_self: Option<crate::layout::LayoutAlign>,
    /// Grid column span.
    pub grid_column: Option<GridPlacement>,
    /// Grid row span.
    pub grid_row: Option<GridPlacement>,
}

/// Grid placement for a single axis (column or row).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridPlacement {
    /// Starting line (0-based).
    pub start: u16,
    /// Number of tracks to span.
    pub span: u16,
}

impl GridPlacement {
    pub fn new(start: u16, span: u16) -> Self {
        Self { start, span }
    }

    pub fn single(index: u16) -> Self {
        Self {
            start: index,
            span: 1,
        }
    }
}

impl Default for LayoutParams {
    fn default() -> Self {
        Self {
            width: crate::layout::Size::Auto,
            height: crate::layout::Size::Auto,
            min_width: None,
            max_width: None,
            min_height: None,
            max_height: None,
            anchor: Anchor::default(),
            margin: Margin::default(),
            padding: Padding::default(),
            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: None,
            align_self: None,
            grid_column: None,
            grid_row: None,
        }
    }
}

/// A single node in the UI tree.
#[derive(Debug, Clone)]
pub struct UINode {
    /// Unique identifier (arena index + generation).
    pub id: UIId,
    /// Human-readable name for debugging.
    pub name: String,
    /// Layout input parameters.
    pub layout_params: LayoutParams,
    /// Computed bounding rectangle after layout (in screen-space pixels).
    pub computed_rect: Rect,
    /// Whether this node and its subtree are visible.
    pub visible: bool,
    /// Whether this node accepts input events.
    pub enabled: bool,
    /// Rendering order within the same parent; higher values draw on top.
    pub z_order: i32,
    /// Parent node (INVALID if root).
    pub parent: UIId,
    /// Ordered list of child node ids.
    pub children: Vec<UIId>,
    /// Whether this node clips its children to its bounds.
    pub clips_children: bool,
    /// Opacity (0.0 = fully transparent, 1.0 = fully opaque).
    pub opacity: f32,
    /// Optional style class names applied to this node.
    pub style_classes: Vec<String>,
    /// Whether this node can receive keyboard focus.
    pub focusable: bool,
    /// Arbitrary string tag for application use.
    pub tag: String,
}

impl UINode {
    /// Creates a new node with default parameters.
    pub fn new(id: UIId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            layout_params: LayoutParams::default(),
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
            visible: true,
            enabled: true,
            z_order: 0,
            parent: UIId::INVALID,
            children: Vec::new(),
            clips_children: false,
            opacity: 1.0,
            style_classes: Vec::new(),
            focusable: false,
            tag: String::new(),
        }
    }

    /// Returns `true` if `point` is inside the computed bounds.
    pub fn hit_test(&self, point: Vec2) -> bool {
        self.visible && self.computed_rect.contains(point)
    }

    /// Computed width.
    pub fn width(&self) -> f32 {
        self.computed_rect.width()
    }

    /// Computed height.
    pub fn height(&self) -> f32 {
        self.computed_rect.height()
    }

    /// Computed size as Vec2.
    pub fn size(&self) -> Vec2 {
        Vec2::new(self.width(), self.height())
    }
}

// ---------------------------------------------------------------------------
// UITree — arena-allocated node tree
// ---------------------------------------------------------------------------

/// Internal arena entry for a UI node slot.
struct ArenaEntry {
    generation: u32,
    node: Option<UINode>,
}

/// Arena-allocated tree of UI nodes.
///
/// Nodes are stored in a flat `Vec` with generational indices so that node
/// removal does not invalidate sibling references. The design mirrors
/// [`genovo_core::HandlePool`] but is specialised for the UI domain.
pub struct UITree {
    entries: Vec<ArenaEntry>,
    free_list: Vec<u32>,
    root: UIId,
    /// Number of live nodes.
    count: usize,
}

impl UITree {
    /// Creates an empty tree with a root node.
    pub fn new() -> Self {
        let mut tree = Self {
            entries: Vec::with_capacity(256),
            free_list: Vec::new(),
            root: UIId::INVALID,
            count: 0,
        };
        let root_id = tree.allocate_node(UINode::new(UIId::INVALID, "root"));
        tree.root = root_id;
        // Patch the root's own id.
        if let Some(node) = tree.get_mut(root_id) {
            node.id = root_id;
        }
        tree
    }

    /// Returns the root node id.
    pub fn root(&self) -> UIId {
        self.root
    }

    /// Number of live nodes (including root).
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Allocate a new node in the arena and return its id.
    fn allocate_node(&mut self, mut node: UINode) -> UIId {
        let id = if let Some(free_index) = self.free_list.pop() {
            let entry = &mut self.entries[free_index as usize];
            let id = UIId::new(free_index, entry.generation);
            node.id = id;
            entry.node = Some(node);
            id
        } else {
            let index = self.entries.len() as u32;
            let id = UIId::new(index, 0);
            node.id = id;
            self.entries.push(ArenaEntry {
                generation: 0,
                node: Some(node),
            });
            id
        };
        self.count += 1;
        id
    }

    /// Adds a child node under `parent` and returns the child's id.
    pub fn add_child(&mut self, parent: UIId, mut node: UINode) -> UIId {
        node.parent = parent;
        let child_id = self.allocate_node(node);
        if let Some(parent_node) = self.get_mut(parent) {
            parent_node.children.push(child_id);
        }
        child_id
    }

    /// Adds a child node under the root.
    pub fn add_root_child(&mut self, node: UINode) -> UIId {
        self.add_child(self.root, node)
    }

    /// Removes a node and all its descendants. Returns `true` if the node
    /// existed.
    pub fn remove(&mut self, id: UIId) -> bool {
        if id == self.root {
            log::warn!("Cannot remove the root node");
            return false;
        }

        // Collect descendant ids first (breadth-first).
        let mut to_remove = vec![id];
        let mut i = 0;
        while i < to_remove.len() {
            let current = to_remove[i];
            if let Some(node) = self.get(current) {
                let children = node.children.clone();
                to_remove.extend(children);
            }
            i += 1;
        }

        // Remove from parent's child list.
        if let Some(node) = self.get(id) {
            let parent = node.parent;
            if let Some(parent_node) = self.get_mut(parent) {
                parent_node.children.retain(|c| *c != id);
            }
        }

        // Free all collected nodes.
        for remove_id in to_remove {
            if let Some(entry) = self.entries.get_mut(remove_id.index as usize) {
                if entry.generation == remove_id.generation && entry.node.is_some() {
                    entry.node = None;
                    entry.generation = entry.generation.wrapping_add(1);
                    self.free_list.push(remove_id.index);
                    self.count -= 1;
                }
            }
        }
        true
    }

    /// Returns a shared reference to a node.
    pub fn get(&self, id: UIId) -> Option<&UINode> {
        let entry = self.entries.get(id.index as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.node.as_ref()
    }

    /// Returns a mutable reference to a node.
    pub fn get_mut(&mut self, id: UIId) -> Option<&mut UINode> {
        let entry = self.entries.get_mut(id.index as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        entry.node.as_mut()
    }

    /// Returns all children of a node.
    pub fn children(&self, id: UIId) -> Vec<UIId> {
        self.get(id).map(|n| n.children.clone()).unwrap_or_default()
    }

    /// Returns the parent of a node.
    pub fn parent(&self, id: UIId) -> Option<UIId> {
        self.get(id).map(|n| n.parent)
    }

    /// Iterates all live nodes in arena order.
    pub fn iter(&self) -> impl Iterator<Item = &UINode> {
        self.entries
            .iter()
            .filter_map(|entry| entry.node.as_ref())
    }

    /// Iterates all live nodes mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut UINode> {
        self.entries
            .iter_mut()
            .filter_map(|entry| entry.node.as_mut())
    }

    /// Performs a depth-first hit test at `point`, returning the deepest
    /// (topmost) visible+enabled node that contains the point.
    pub fn hit_test(&self, point: Vec2) -> Option<UIId> {
        self.hit_test_recursive(self.root, point)
    }

    fn hit_test_recursive(&self, id: UIId, point: Vec2) -> Option<UIId> {
        let node = self.get(id)?;
        if !node.visible || !node.enabled {
            return None;
        }
        if node.clips_children && !node.computed_rect.contains(point) {
            return None;
        }

        // Check children in reverse z-order (highest first).
        let mut sorted_children = node.children.clone();
        sorted_children.sort_by(|a, b| {
            let za = self.get(*a).map(|n| n.z_order).unwrap_or(0);
            let zb = self.get(*b).map(|n| n.z_order).unwrap_or(0);
            zb.cmp(&za) // highest z-order first
        });

        for child_id in sorted_children {
            if let Some(hit) = self.hit_test_recursive(child_id, point) {
                return Some(hit);
            }
        }

        // If no child was hit, test self.
        if node.hit_test(point) {
            Some(id)
        } else {
            None
        }
    }

    /// Moves a node to a new parent. Returns false if either node doesn't
    /// exist.
    pub fn reparent(&mut self, node_id: UIId, new_parent: UIId) -> bool {
        if node_id == self.root {
            return false;
        }
        let old_parent = match self.get(node_id) {
            Some(n) => n.parent,
            None => return false,
        };
        if self.get(new_parent).is_none() {
            return false;
        }

        // Remove from old parent.
        if let Some(old) = self.get_mut(old_parent) {
            old.children.retain(|c| *c != node_id);
        }
        // Add to new parent.
        if let Some(new) = self.get_mut(new_parent) {
            new.children.push(node_id);
        }
        if let Some(node) = self.get_mut(node_id) {
            node.parent = new_parent;
        }
        true
    }

    /// Collects all ancestor ids from `id` up to (but not including) root.
    pub fn ancestors(&self, id: UIId) -> Vec<UIId> {
        let mut result = Vec::new();
        let mut current = id;
        while let Some(node) = self.get(current) {
            if node.parent == UIId::INVALID || node.parent == current {
                break;
            }
            result.push(node.parent);
            current = node.parent;
        }
        result
    }
}

impl Default for UITree {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// UIContext — per-frame UI state
// ---------------------------------------------------------------------------

/// Transient state for the current UI frame.
///
/// Tracks which node is hovered, focused, and being dragged so that widgets
/// can query interaction state without maintaining their own bookkeeping.
pub struct UIContext {
    /// The node currently under the mouse cursor.
    pub hovered: UIId,
    /// The node that has keyboard focus.
    pub focused: UIId,
    /// The node being dragged.
    pub dragging: UIId,
    /// Current mouse position in logical pixels.
    pub mouse_position: Vec2,
    /// Whether the left mouse button is held.
    pub mouse_down: bool,
    /// Delta time since last frame (seconds).
    pub delta_time: f32,
    /// Total elapsed time (seconds).
    pub total_time: f32,
    /// Pending events to dispatch after the current frame.
    pub pending_events: Vec<(UIId, UIEvent)>,
    /// Map of node id to a set of string tags for transient per-frame data.
    pub frame_data: HashMap<UIId, HashMap<String, String>>,
    /// Clipboard contents (for copy/paste in text fields).
    pub clipboard: String,
}

impl UIContext {
    pub fn new() -> Self {
        Self {
            hovered: UIId::INVALID,
            focused: UIId::INVALID,
            dragging: UIId::INVALID,
            mouse_position: Vec2::ZERO,
            mouse_down: false,
            delta_time: 0.0,
            total_time: 0.0,
            pending_events: Vec::new(),
            frame_data: HashMap::new(),
            clipboard: String::new(),
        }
    }

    /// Queue an event to be dispatched to a specific node.
    pub fn send_event(&mut self, target: UIId, event: UIEvent) {
        self.pending_events.push((target, event));
    }

    /// Sets the focused node, blurring the previous one.
    pub fn set_focus(&mut self, id: UIId) {
        if self.focused != id {
            if !self.focused.is_invalid() {
                self.pending_events.push((self.focused, UIEvent::Blur));
            }
            self.focused = id;
            if !id.is_invalid() {
                self.pending_events.push((id, UIEvent::Focus));
            }
        }
    }

    /// Updates hovered node, sending Hover/HoverEnd events as needed.
    pub fn set_hovered(&mut self, id: UIId) {
        if self.hovered != id {
            if !self.hovered.is_invalid() {
                self.pending_events.push((self.hovered, UIEvent::HoverEnd));
            }
            self.hovered = id;
            if !id.is_invalid() {
                self.pending_events.push((
                    id,
                    UIEvent::Hover {
                        position: self.mouse_position,
                    },
                ));
            }
        }
    }

    /// Advance time for the frame.
    pub fn begin_frame(&mut self, delta_time: f32) {
        self.delta_time = delta_time;
        self.total_time += delta_time;
        self.pending_events.clear();
        self.frame_data.clear();
    }
}

impl Default for UIContext {
    fn default() -> Self {
        Self::new()
    }
}
