//! Docking system for the Genovo editor.
//!
//! Provides a tree-based layout of dockable tab panels that can be split
//! horizontally or vertically, with tabs that can be dragged between panels
//! or floated into separate windows. This is the #1 feature that makes an
//! editor look professional.
//!
//! # Architecture
//!
//! The dock layout is represented as a binary tree of [`DockNode`]s. Leaf
//! nodes contain a tab well with one or more [`DockTab`]s; inner nodes are
//! either horizontal or vertical splits with a configurable ratio. The
//! complete layout is stored in a [`DockState`] which also tracks floating
//! windows and provides mutation helpers.
//!
//! ```text
//!  ┌──────────────────────────────────────────────┐
//!  │  HorizontalSplit (ratio 0.25)                │
//!  │  ┌────────┐  ┌────────────────────────────┐  │
//!  │  │ Leaf   │  │ VerticalSplit (ratio 0.7)  │  │
//!  │  │ (tabs) │  │ ┌────────────────────────┐ │  │
//!  │  │        │  │ │ Leaf (tabs)            │ │  │
//!  │  │        │  │ ├────────────────────────┤ │  │
//!  │  │        │  │ │ Leaf (tabs)            │ │  │
//!  │  │        │  │ └────────────────────────┘ │  │
//!  │  └────────┘  └────────────────────────────┘  │
//!  └──────────────────────────────────────────────┘
//! ```

use glam::Vec2;
use serde::{Deserialize, Serialize};

use genovo_core::Rect;

use crate::render_commands::{
    Border, Color, CornerRadii, DrawList, TextureId,
};

// ---------------------------------------------------------------------------
// Tab Identifier
// ---------------------------------------------------------------------------

/// Unique identifier for a dock tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DockTabId(pub u64);

impl DockTabId {
    /// Sentinel value for no tab.
    pub const INVALID: Self = Self(u64::MAX);

    /// Create a new tab id.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns `true` if this is the invalid sentinel.
    pub fn is_invalid(&self) -> bool {
        *self == Self::INVALID
    }
}

impl Default for DockTabId {
    fn default() -> Self {
        Self::INVALID
    }
}

impl std::fmt::Display for DockTabId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DockTab({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Node Identifier
// ---------------------------------------------------------------------------

/// Unique identifier for a node in the dock tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DockNodeId(pub u64);

impl DockNodeId {
    pub const INVALID: Self = Self(u64::MAX);

    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn is_invalid(&self) -> bool {
        *self == Self::INVALID
    }
}

impl Default for DockNodeId {
    fn default() -> Self {
        Self::INVALID
    }
}

impl std::fmt::Display for DockNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DockNode({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// DockTab — metadata for a single dockable tab
// ---------------------------------------------------------------------------

/// Metadata for a single tab within a dockable panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockTab {
    /// Unique identifier for this tab.
    pub id: DockTabId,
    /// Display title.
    pub title: String,
    /// Optional icon texture.
    pub icon: Option<TextureId>,
    /// Whether the tab can be closed by the user.
    pub closable: bool,
    /// Whether to show a dirty (unsaved) indicator dot.
    pub dirty_indicator: bool,
    /// Optional tooltip text.
    pub tooltip: String,
    /// Arbitrary user data tag.
    pub user_data: u64,
    /// Computed rect of the tab header (set during layout).
    #[serde(skip)]
    pub header_rect: Rect,
    /// Whether this tab is being dragged.
    #[serde(skip)]
    pub dragging: bool,
}

impl DockTab {
    /// Creates a new dock tab.
    pub fn new(id: DockTabId, title: impl Into<String>) -> Self {
        Self {
            id,
            title: title.into(),
            icon: None,
            closable: true,
            dirty_indicator: false,
            tooltip: String::new(),
            user_data: 0,
            header_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
            dragging: false,
        }
    }

    /// Builder: set icon.
    pub fn with_icon(mut self, icon: TextureId) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Builder: set closable.
    pub fn with_closable(mut self, closable: bool) -> Self {
        self.closable = closable;
        self
    }

    /// Builder: set dirty indicator.
    pub fn with_dirty(mut self, dirty: bool) -> Self {
        self.dirty_indicator = dirty;
        self
    }

    /// Builder: set tooltip.
    pub fn with_tooltip(mut self, tooltip: impl Into<String>) -> Self {
        self.tooltip = tooltip.into();
        self
    }

    /// Builder: set user data.
    pub fn with_user_data(mut self, data: u64) -> Self {
        self.user_data = data;
        self
    }

    /// Compute the width needed for this tab header.
    pub fn compute_header_width(&self, font_size: f32) -> f32 {
        let text_width = self.title.len() as f32 * font_size * 0.55;
        let icon_width = if self.icon.is_some() { font_size + 4.0 } else { 0.0 };
        let close_width = if self.closable { font_size + 2.0 } else { 0.0 };
        let dirty_width = if self.dirty_indicator { 8.0 } else { 0.0 };
        let padding = 16.0;
        text_width + icon_width + close_width + dirty_width + padding
    }
}

// ---------------------------------------------------------------------------
// DockTarget — where a dragged tab can be docked
// ---------------------------------------------------------------------------

/// Describes where a dragged tab can be docked relative to an existing node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DockTarget {
    /// Dock to the left, creating a horizontal split.
    Left,
    /// Dock to the right, creating a horizontal split.
    Right,
    /// Dock to the top, creating a vertical split.
    Top,
    /// Dock to the bottom, creating a vertical split.
    Bottom,
    /// Dock as a new tab in the existing tab well.
    Center,
    /// Float as a separate window.
    Float,
}

impl DockTarget {
    /// Returns `true` if this target creates a split.
    pub fn is_split(&self) -> bool {
        matches!(self, Self::Left | Self::Right | Self::Top | Self::Bottom)
    }

    /// Returns all directional targets (excluding Center and Float).
    pub fn directional_targets() -> &'static [DockTarget] {
        &[
            DockTarget::Left,
            DockTarget::Right,
            DockTarget::Top,
            DockTarget::Bottom,
        ]
    }

    /// Returns the split direction that this target implies.
    pub fn split_direction(&self) -> Option<SplitDirection> {
        match self {
            Self::Left | Self::Right => Some(SplitDirection::Horizontal),
            Self::Top | Self::Bottom => Some(SplitDirection::Vertical),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SplitDirection
// ---------------------------------------------------------------------------

/// Direction of a split in the dock tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SplitDirection {
    /// Left | Right split.
    Horizontal,
    /// Top | Bottom split.
    Vertical,
}

// ---------------------------------------------------------------------------
// DockNode — recursive tree node
// ---------------------------------------------------------------------------

/// A node in the docking tree. This is a recursive structure: leaves hold
/// tabs, inner nodes hold two children separated by a split divider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DockNode {
    /// A leaf node containing a tab well with one or more tabs.
    Leaf {
        /// Unique identifier for this node.
        id: DockNodeId,
        /// The tabs in this panel.
        tabs: Vec<DockTab>,
        /// Index of the currently active (selected) tab.
        active_tab: usize,
        /// Computed bounding rect for this leaf (set during layout).
        #[serde(skip)]
        rect: Rect,
        /// Height of the tab bar header area.
        #[serde(skip)]
        tab_bar_height: f32,
    },
    /// A horizontal split: left child | right child.
    HorizontalSplit {
        /// Unique identifier for this node.
        id: DockNodeId,
        /// Left child subtree.
        left: Box<DockNode>,
        /// Right child subtree.
        right: Box<DockNode>,
        /// Ratio of left vs right (0.0 to 1.0). 0.5 means equal split.
        ratio: f32,
        /// Computed bounding rect (set during layout).
        #[serde(skip)]
        rect: Rect,
    },
    /// A vertical split: top child / bottom child.
    VerticalSplit {
        /// Unique identifier for this node.
        id: DockNodeId,
        /// Top child subtree.
        top: Box<DockNode>,
        /// Bottom child subtree.
        bottom: Box<DockNode>,
        /// Ratio of top vs bottom (0.0 to 1.0). 0.5 means equal split.
        ratio: f32,
        /// Computed bounding rect (set during layout).
        #[serde(skip)]
        rect: Rect,
    },
}

impl DockNode {
    // -- Constructors -------------------------------------------------------

    /// Creates a new leaf node with the given tabs.
    pub fn leaf(id: DockNodeId, tabs: Vec<DockTab>) -> Self {
        Self::Leaf {
            id,
            active_tab: 0,
            tabs,
            rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
            tab_bar_height: 28.0,
        }
    }

    /// Creates a new leaf node with a single tab.
    pub fn single_tab(id: DockNodeId, tab: DockTab) -> Self {
        Self::leaf(id, vec![tab])
    }

    /// Creates a new empty leaf node (no tabs).
    pub fn empty_leaf(id: DockNodeId) -> Self {
        Self::leaf(id, Vec::new())
    }

    /// Creates a horizontal split.
    pub fn horizontal_split(
        id: DockNodeId,
        left: DockNode,
        right: DockNode,
        ratio: f32,
    ) -> Self {
        Self::HorizontalSplit {
            id,
            left: Box::new(left),
            right: Box::new(right),
            ratio: ratio.clamp(0.05, 0.95),
            rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    /// Creates a vertical split.
    pub fn vertical_split(
        id: DockNodeId,
        top: DockNode,
        bottom: DockNode,
        ratio: f32,
    ) -> Self {
        Self::VerticalSplit {
            id,
            top: Box::new(top),
            bottom: Box::new(bottom),
            ratio: ratio.clamp(0.05, 0.95),
            rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    // -- Accessors ----------------------------------------------------------

    /// Returns the node id.
    pub fn id(&self) -> DockNodeId {
        match self {
            Self::Leaf { id, .. }
            | Self::HorizontalSplit { id, .. }
            | Self::VerticalSplit { id, .. } => *id,
        }
    }

    /// Returns the computed bounding rect.
    pub fn rect(&self) -> Rect {
        match self {
            Self::Leaf { rect, .. }
            | Self::HorizontalSplit { rect, .. }
            | Self::VerticalSplit { rect, .. } => *rect,
        }
    }

    /// Returns `true` if this is a leaf node.
    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }

    /// Returns `true` if this is a split node.
    pub fn is_split(&self) -> bool {
        !self.is_leaf()
    }

    /// Returns `true` if this is a leaf node with no tabs.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Leaf { tabs, .. } => tabs.is_empty(),
            _ => false,
        }
    }

    /// Returns the number of tabs in this node (leaf only, 0 for splits).
    pub fn tab_count(&self) -> usize {
        match self {
            Self::Leaf { tabs, .. } => tabs.len(),
            _ => 0,
        }
    }

    /// Returns all tabs in this subtree.
    pub fn all_tabs(&self) -> Vec<&DockTab> {
        let mut result = Vec::new();
        self.collect_tabs(&mut result);
        result
    }

    fn collect_tabs<'a>(&'a self, out: &mut Vec<&'a DockTab>) {
        match self {
            Self::Leaf { tabs, .. } => {
                for tab in tabs {
                    out.push(tab);
                }
            }
            Self::HorizontalSplit { left, right, .. } => {
                left.collect_tabs(out);
                right.collect_tabs(out);
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.collect_tabs(out);
                bottom.collect_tabs(out);
            }
        }
    }

    /// Returns all tabs in this subtree (mutable).
    pub fn all_tabs_mut(&mut self) -> Vec<&mut DockTab> {
        let mut result = Vec::new();
        self.collect_tabs_mut(&mut result);
        result
    }

    fn collect_tabs_mut<'a>(&'a mut self, out: &mut Vec<&'a mut DockTab>) {
        match self {
            Self::Leaf { tabs, .. } => {
                for tab in tabs {
                    out.push(tab);
                }
            }
            Self::HorizontalSplit { left, right, .. } => {
                left.collect_tabs_mut(out);
                right.collect_tabs_mut(out);
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.collect_tabs_mut(out);
                bottom.collect_tabs_mut(out);
            }
        }
    }

    /// Counts total nodes in this subtree.
    pub fn node_count(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::HorizontalSplit { left, right, .. } => {
                1 + left.node_count() + right.node_count()
            }
            Self::VerticalSplit { top, bottom, .. } => {
                1 + top.node_count() + bottom.node_count()
            }
        }
    }

    /// Collects all node ids in this subtree.
    pub fn all_node_ids(&self) -> Vec<DockNodeId> {
        let mut result = Vec::new();
        self.collect_node_ids(&mut result);
        result
    }

    fn collect_node_ids(&self, out: &mut Vec<DockNodeId>) {
        out.push(self.id());
        match self {
            Self::Leaf { .. } => {}
            Self::HorizontalSplit { left, right, .. } => {
                left.collect_node_ids(out);
                right.collect_node_ids(out);
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.collect_node_ids(out);
                bottom.collect_node_ids(out);
            }
        }
    }

    /// Collects all leaf node ids in this subtree.
    pub fn all_leaf_ids(&self) -> Vec<DockNodeId> {
        let mut result = Vec::new();
        self.collect_leaf_ids(&mut result);
        result
    }

    fn collect_leaf_ids(&self, out: &mut Vec<DockNodeId>) {
        match self {
            Self::Leaf { id, .. } => out.push(*id),
            Self::HorizontalSplit { left, right, .. } => {
                left.collect_leaf_ids(out);
                right.collect_leaf_ids(out);
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.collect_leaf_ids(out);
                bottom.collect_leaf_ids(out);
            }
        }
    }

    // -- Tab operations -----------------------------------------------------

    /// Find a tab by id in this subtree. Returns the node id and tab index.
    pub fn find_tab(&self, tab_id: DockTabId) -> Option<(DockNodeId, usize)> {
        match self {
            Self::Leaf { id, tabs, .. } => {
                for (i, tab) in tabs.iter().enumerate() {
                    if tab.id == tab_id {
                        return Some((*id, i));
                    }
                }
                None
            }
            Self::HorizontalSplit { left, right, .. } => {
                left.find_tab(tab_id).or_else(|| right.find_tab(tab_id))
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.find_tab(tab_id).or_else(|| bottom.find_tab(tab_id))
            }
        }
    }

    /// Get a reference to a tab by id.
    pub fn get_tab(&self, tab_id: DockTabId) -> Option<&DockTab> {
        match self {
            Self::Leaf { tabs, .. } => tabs.iter().find(|t| t.id == tab_id),
            Self::HorizontalSplit { left, right, .. } => {
                left.get_tab(tab_id).or_else(|| right.get_tab(tab_id))
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.get_tab(tab_id).or_else(|| bottom.get_tab(tab_id))
            }
        }
    }

    /// Get a mutable reference to a tab by id.
    pub fn get_tab_mut(&mut self, tab_id: DockTabId) -> Option<&mut DockTab> {
        match self {
            Self::Leaf { tabs, .. } => tabs.iter_mut().find(|t| t.id == tab_id),
            Self::HorizontalSplit { left, right, .. } => {
                if let Some(tab) = left.get_tab_mut(tab_id) {
                    Some(tab)
                } else {
                    right.get_tab_mut(tab_id)
                }
            }
            Self::VerticalSplit { top, bottom, .. } => {
                if let Some(tab) = top.get_tab_mut(tab_id) {
                    Some(tab)
                } else {
                    bottom.get_tab_mut(tab_id)
                }
            }
        }
    }

    /// Find a node by id in this subtree.
    pub fn find_node(&self, node_id: DockNodeId) -> Option<&DockNode> {
        if self.id() == node_id {
            return Some(self);
        }
        match self {
            Self::Leaf { .. } => None,
            Self::HorizontalSplit { left, right, .. } => {
                left.find_node(node_id).or_else(|| right.find_node(node_id))
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.find_node(node_id).or_else(|| bottom.find_node(node_id))
            }
        }
    }

    /// Find a mutable node by id in this subtree.
    pub fn find_node_mut(&mut self, node_id: DockNodeId) -> Option<&mut DockNode> {
        if self.id() == node_id {
            return Some(self);
        }
        match self {
            Self::Leaf { .. } => None,
            Self::HorizontalSplit { left, right, .. } => {
                if let Some(n) = left.find_node_mut(node_id) {
                    Some(n)
                } else {
                    right.find_node_mut(node_id)
                }
            }
            Self::VerticalSplit { top, bottom, .. } => {
                if let Some(n) = top.find_node_mut(node_id) {
                    Some(n)
                } else {
                    bottom.find_node_mut(node_id)
                }
            }
        }
    }

    /// Adds a tab to a leaf node with the given id. Returns `true` on success.
    pub fn add_tab(&mut self, node_id: DockNodeId, tab: DockTab) -> bool {
        if let Some(node) = self.find_node_mut(node_id) {
            match node {
                Self::Leaf { tabs, active_tab, .. } => {
                    tabs.push(tab);
                    *active_tab = tabs.len() - 1;
                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Removes a tab from this subtree. Returns the removed tab if found.
    pub fn remove_tab(&mut self, tab_id: DockTabId) -> Option<DockTab> {
        match self {
            Self::Leaf { tabs, active_tab, .. } => {
                if let Some(idx) = tabs.iter().position(|t| t.id == tab_id) {
                    let tab = tabs.remove(idx);
                    if *active_tab >= tabs.len() && !tabs.is_empty() {
                        *active_tab = tabs.len() - 1;
                    }
                    Some(tab)
                } else {
                    None
                }
            }
            Self::HorizontalSplit { left, right, .. } => {
                if let Some(tab) = left.remove_tab(tab_id) {
                    Some(tab)
                } else {
                    right.remove_tab(tab_id)
                }
            }
            Self::VerticalSplit { top, bottom, .. } => {
                if let Some(tab) = top.remove_tab(tab_id) {
                    Some(tab)
                } else {
                    bottom.remove_tab(tab_id)
                }
            }
        }
    }

    /// Sets the active tab index for a leaf node.
    pub fn set_active_tab(&mut self, node_id: DockNodeId, idx: usize) -> bool {
        if let Some(node) = self.find_node_mut(node_id) {
            match node {
                Self::Leaf { tabs, active_tab, .. } => {
                    if idx < tabs.len() {
                        *active_tab = idx;
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Activates a tab by its id.
    pub fn activate_tab(&mut self, tab_id: DockTabId) -> bool {
        match self {
            Self::Leaf { tabs, active_tab, .. } => {
                if let Some(idx) = tabs.iter().position(|t| t.id == tab_id) {
                    *active_tab = idx;
                    true
                } else {
                    false
                }
            }
            Self::HorizontalSplit { left, right, .. } => {
                left.activate_tab(tab_id) || right.activate_tab(tab_id)
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.activate_tab(tab_id) || bottom.activate_tab(tab_id)
            }
        }
    }

    /// Reorder a tab within a leaf node. Moves tab at `from` to `to`.
    pub fn reorder_tab(&mut self, node_id: DockNodeId, from: usize, to: usize) -> bool {
        if let Some(node) = self.find_node_mut(node_id) {
            match node {
                Self::Leaf { tabs, active_tab, .. } => {
                    if from < tabs.len() && to < tabs.len() && from != to {
                        let tab = tabs.remove(from);
                        tabs.insert(to, tab);
                        // Adjust active tab index
                        if *active_tab == from {
                            *active_tab = to;
                        } else if from < *active_tab && to >= *active_tab {
                            *active_tab = active_tab.saturating_sub(1);
                        } else if from > *active_tab && to <= *active_tab {
                            *active_tab += 1;
                        }
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            }
        } else {
            false
        }
    }

    // -- Layout computation --------------------------------------------------

    /// Compute layout for this subtree within the given rect.
    pub fn compute_layout(&mut self, available_rect: Rect) {
        self.compute_layout_internal(available_rect, DIVIDER_THICKNESS);
    }

    fn compute_layout_internal(&mut self, available_rect: Rect, divider_px: f32) {
        match self {
            Self::Leaf { rect, .. } => {
                *rect = available_rect;
            }
            Self::HorizontalSplit {
                left,
                right,
                ratio,
                rect,
                ..
            } => {
                *rect = available_rect;
                let total_w = available_rect.width();
                let half_div = divider_px * 0.5;
                let split_x =
                    available_rect.min.x + (total_w * (*ratio)).max(MIN_PANEL_SIZE);
                let split_x = split_x.min(
                    available_rect.max.x - MIN_PANEL_SIZE,
                );

                let left_rect = Rect::new(
                    available_rect.min,
                    Vec2::new(split_x - half_div, available_rect.max.y),
                );
                let right_rect = Rect::new(
                    Vec2::new(split_x + half_div, available_rect.min.y),
                    available_rect.max,
                );

                left.compute_layout_internal(left_rect, divider_px);
                right.compute_layout_internal(right_rect, divider_px);
            }
            Self::VerticalSplit {
                top,
                bottom,
                ratio,
                rect,
                ..
            } => {
                *rect = available_rect;
                let total_h = available_rect.height();
                let half_div = divider_px * 0.5;
                let split_y =
                    available_rect.min.y + (total_h * (*ratio)).max(MIN_PANEL_SIZE);
                let split_y = split_y.min(
                    available_rect.max.y - MIN_PANEL_SIZE,
                );

                let top_rect = Rect::new(
                    available_rect.min,
                    Vec2::new(available_rect.max.x, split_y - half_div),
                );
                let bottom_rect = Rect::new(
                    Vec2::new(available_rect.min.x, split_y + half_div),
                    available_rect.max,
                );

                top.compute_layout_internal(top_rect, divider_px);
                bottom.compute_layout_internal(bottom_rect, divider_px);
            }
        }
    }

    /// Compute tab header rects within a leaf node.
    pub fn compute_tab_headers(&mut self, font_size: f32) {
        match self {
            Self::Leaf {
                tabs,
                rect,
                tab_bar_height,
                ..
            } => {
                let mut x = rect.min.x + TAB_PADDING;
                let y = rect.min.y;
                let height = *tab_bar_height;

                for tab in tabs.iter_mut() {
                    let w = tab.compute_header_width(font_size);
                    tab.header_rect = Rect::new(
                        Vec2::new(x, y),
                        Vec2::new(x + w, y + height),
                    );
                    x += w + TAB_GAP;
                }
            }
            Self::HorizontalSplit { left, right, .. } => {
                left.compute_tab_headers(font_size);
                right.compute_tab_headers(font_size);
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.compute_tab_headers(font_size);
                bottom.compute_tab_headers(font_size);
            }
        }
    }

    /// Returns the content rect (below the tab bar) for a leaf node.
    pub fn content_rect(&self) -> Option<Rect> {
        match self {
            Self::Leaf {
                rect,
                tab_bar_height,
                ..
            } => Some(Rect::new(
                Vec2::new(rect.min.x, rect.min.y + tab_bar_height),
                rect.max,
            )),
            _ => None,
        }
    }

    // -- Split divider hit-testing -------------------------------------------

    /// Returns the divider rect for a split node.
    pub fn divider_rect(&self) -> Option<Rect> {
        match self {
            Self::HorizontalSplit {
                left, rect, ..
            } => {
                let left_rect = left.rect();
                let split_x = left_rect.max.x;
                Some(Rect::new(
                    Vec2::new(split_x, rect.min.y),
                    Vec2::new(split_x + DIVIDER_THICKNESS, rect.max.y),
                ))
            }
            Self::VerticalSplit {
                top, rect, ..
            } => {
                let top_rect = top.rect();
                let split_y = top_rect.max.y;
                Some(Rect::new(
                    Vec2::new(rect.min.x, split_y),
                    Vec2::new(rect.max.x, split_y + DIVIDER_THICKNESS),
                ))
            }
            _ => None,
        }
    }

    /// Hit-test for the divider. Returns the node id of the split whose
    /// divider contains the point, along with the direction.
    pub fn hit_test_divider(&self, point: Vec2) -> Option<(DockNodeId, SplitDirection)> {
        match self {
            Self::HorizontalSplit {
                id, left, right, ..
            } => {
                if let Some(div_rect) = self.divider_rect() {
                    if div_rect.contains(point) {
                        return Some((*id, SplitDirection::Horizontal));
                    }
                }
                left.hit_test_divider(point)
                    .or_else(|| right.hit_test_divider(point))
            }
            Self::VerticalSplit {
                id, top, bottom, ..
            } => {
                if let Some(div_rect) = self.divider_rect() {
                    if div_rect.contains(point) {
                        return Some((*id, SplitDirection::Vertical));
                    }
                }
                top.hit_test_divider(point)
                    .or_else(|| bottom.hit_test_divider(point))
            }
            Self::Leaf { .. } => None,
        }
    }

    /// Hit-test for a tab header. Returns the node id and tab index.
    pub fn hit_test_tab(&self, point: Vec2) -> Option<(DockNodeId, usize)> {
        match self {
            Self::Leaf { id, tabs, .. } => {
                for (i, tab) in tabs.iter().enumerate() {
                    if tab.header_rect.contains(point) {
                        return Some((*id, i));
                    }
                }
                None
            }
            Self::HorizontalSplit { left, right, .. } => {
                left.hit_test_tab(point)
                    .or_else(|| right.hit_test_tab(point))
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.hit_test_tab(point)
                    .or_else(|| bottom.hit_test_tab(point))
            }
        }
    }

    /// Hit-test for a tab close button. Returns the tab id if the point is
    /// over a close button on a closable tab.
    pub fn hit_test_close_button(&self, point: Vec2, font_size: f32) -> Option<DockTabId> {
        match self {
            Self::Leaf { tabs, .. } => {
                for tab in tabs {
                    if !tab.closable {
                        continue;
                    }
                    let close_rect = compute_close_button_rect(&tab.header_rect, font_size);
                    if close_rect.contains(point) {
                        return Some(tab.id);
                    }
                }
                None
            }
            Self::HorizontalSplit { left, right, .. } => {
                left.hit_test_close_button(point, font_size)
                    .or_else(|| right.hit_test_close_button(point, font_size))
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.hit_test_close_button(point, font_size)
                    .or_else(|| bottom.hit_test_close_button(point, font_size))
            }
        }
    }

    /// Hit-test for a leaf node's content area.
    pub fn hit_test_content(&self, point: Vec2) -> Option<DockNodeId> {
        match self {
            Self::Leaf { id, .. } => {
                if let Some(cr) = self.content_rect() {
                    if cr.contains(point) {
                        return Some(*id);
                    }
                }
                None
            }
            Self::HorizontalSplit { left, right, .. } => {
                left.hit_test_content(point)
                    .or_else(|| right.hit_test_content(point))
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.hit_test_content(point)
                    .or_else(|| bottom.hit_test_content(point))
            }
        }
    }

    /// Determine the dock target zone for a point within a leaf node's rect.
    /// The outer 25% of each edge is a directional dock target; the center is
    /// Center.
    pub fn compute_dock_target(&self, point: Vec2) -> Option<(DockNodeId, DockTarget)> {
        match self {
            Self::Leaf { id, rect, .. } => {
                if !rect.contains(point) {
                    return None;
                }
                let w = rect.width();
                let h = rect.height();
                let rel_x = (point.x - rect.min.x) / w;
                let rel_y = (point.y - rect.min.y) / h;

                // Edge zones are the outer 25%
                let target = if rel_x < DOCK_ZONE_FRACTION {
                    DockTarget::Left
                } else if rel_x > (1.0 - DOCK_ZONE_FRACTION) {
                    DockTarget::Right
                } else if rel_y < DOCK_ZONE_FRACTION {
                    DockTarget::Top
                } else if rel_y > (1.0 - DOCK_ZONE_FRACTION) {
                    DockTarget::Bottom
                } else {
                    DockTarget::Center
                };
                Some((*id, target))
            }
            Self::HorizontalSplit { left, right, .. } => {
                left.compute_dock_target(point)
                    .or_else(|| right.compute_dock_target(point))
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.compute_dock_target(point)
                    .or_else(|| bottom.compute_dock_target(point))
            }
        }
    }

    /// Updates the split ratio of this node (if it is a split).
    pub fn set_ratio(&mut self, new_ratio: f32) {
        let clamped = new_ratio.clamp(0.05, 0.95);
        match self {
            Self::HorizontalSplit { ratio, .. } | Self::VerticalSplit { ratio, .. } => {
                *ratio = clamped;
            }
            _ => {}
        }
    }

    /// Clean up empty leaf nodes by collapsing them. If a split has an empty
    /// leaf child, it is replaced by the other child. Returns `true` if this
    /// node itself became empty and should be removed.
    pub fn collapse_empty(&mut self) -> bool {
        match self {
            Self::Leaf { tabs, .. } => tabs.is_empty(),
            Self::HorizontalSplit { left, right, .. } => {
                let left_empty = left.collapse_empty();
                let right_empty = right.collapse_empty();

                if left_empty && right_empty {
                    // Both empty — this whole subtree is empty. Replace self
                    // with an empty leaf.
                    let id = self.id();
                    *self = Self::empty_leaf(id);
                    true
                } else if left_empty {
                    let replacement = std::mem::replace(
                        right.as_mut(),
                        Self::empty_leaf(DockNodeId::INVALID),
                    );
                    *self = replacement;
                    false
                } else if right_empty {
                    let replacement = std::mem::replace(
                        left.as_mut(),
                        Self::empty_leaf(DockNodeId::INVALID),
                    );
                    *self = replacement;
                    false
                } else {
                    false
                }
            }
            Self::VerticalSplit { top, bottom, .. } => {
                let top_empty = top.collapse_empty();
                let bottom_empty = bottom.collapse_empty();

                if top_empty && bottom_empty {
                    let id = self.id();
                    *self = Self::empty_leaf(id);
                    true
                } else if top_empty {
                    let replacement = std::mem::replace(
                        bottom.as_mut(),
                        Self::empty_leaf(DockNodeId::INVALID),
                    );
                    *self = replacement;
                    false
                } else if bottom_empty {
                    let replacement = std::mem::replace(
                        top.as_mut(),
                        Self::empty_leaf(DockNodeId::INVALID),
                    );
                    *self = replacement;
                    false
                } else {
                    false
                }
            }
        }
    }

    // -- Rendering ----------------------------------------------------------

    /// Render the dock tree into a draw list using the provided style.
    pub fn render(&self, draw_list: &mut DrawList, style: &DockStyle, font_size: f32) {
        match self {
            Self::Leaf {
                tabs,
                active_tab,
                rect,
                tab_bar_height,
                ..
            } => {
                // Draw the panel background
                draw_list.draw_rect(*rect, style.panel_background);

                // Draw the tab bar background
                let tab_bar_rect = Rect::new(
                    rect.min,
                    Vec2::new(rect.max.x, rect.min.y + tab_bar_height),
                );
                draw_list.draw_rect(tab_bar_rect, style.tab_bar_background);

                // Draw each tab header
                for (i, tab) in tabs.iter().enumerate() {
                    let is_active = i == *active_tab;
                    let tab_bg = if is_active {
                        style.active_tab_background
                    } else {
                        style.inactive_tab_background
                    };

                    // Tab background
                    draw_list.draw_rounded_rect(
                        tab.header_rect,
                        tab_bg,
                        CornerRadii::new(4.0, 4.0, 0.0, 0.0),
                        Border::default(),
                    );

                    // Tab title
                    let text_x = tab.header_rect.min.x + 8.0
                        + if tab.icon.is_some() { font_size + 4.0 } else { 0.0 };
                    let text_y = tab.header_rect.min.y
                        + (*tab_bar_height - font_size) * 0.5;

                    let text_color = if is_active {
                        style.active_tab_text
                    } else {
                        style.inactive_tab_text
                    };

                    // Dirty indicator dot
                    if tab.dirty_indicator {
                        let dot_x = tab.header_rect.min.x + 4.0;
                        let dot_y = tab.header_rect.center().y;
                        draw_list.draw_circle(
                            Vec2::new(dot_x, dot_y),
                            3.0,
                            style.dirty_indicator_color,
                        );
                    }

                    draw_list.draw_text(
                        &tab.title,
                        Vec2::new(text_x, text_y),
                        font_size,
                        text_color,
                    );

                    // Close button
                    if tab.closable {
                        let close_rect = compute_close_button_rect(
                            &tab.header_rect,
                            font_size,
                        );
                        draw_list.draw_text(
                            "\u{00D7}", // multiplication sign as close icon
                            Vec2::new(close_rect.min.x + 1.0, close_rect.min.y),
                            font_size * 0.8,
                            style.close_button_color,
                        );
                    }
                }

                // Bottom border of tab bar
                draw_list.draw_line(
                    Vec2::new(rect.min.x, rect.min.y + tab_bar_height),
                    Vec2::new(rect.max.x, rect.min.y + tab_bar_height),
                    style.border_color,
                    1.0,
                );

                // Content area border
                let content = Rect::new(
                    Vec2::new(rect.min.x, rect.min.y + tab_bar_height),
                    rect.max,
                );
                draw_list.draw_rounded_rect(
                    content,
                    style.content_background,
                    CornerRadii::ZERO,
                    Border::new(style.border_color, 1.0),
                );
            }
            Self::HorizontalSplit { left, right, .. } => {
                left.render(draw_list, style, font_size);
                right.render(draw_list, style, font_size);

                // Draw the divider
                if let Some(div_rect) = self.divider_rect() {
                    draw_list.draw_rect(div_rect, style.divider_color);
                }
            }
            Self::VerticalSplit { top, bottom, .. } => {
                top.render(draw_list, style, font_size);
                bottom.render(draw_list, style, font_size);

                if let Some(div_rect) = self.divider_rect() {
                    draw_list.draw_rect(div_rect, style.divider_color);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Floating Window
// ---------------------------------------------------------------------------

/// A floating (detached) window that holds a dock node outside the main tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatingWindow {
    /// Unique window identifier.
    pub id: u64,
    /// Title displayed in the floating window title bar.
    pub title: String,
    /// Position of the window (top-left corner).
    pub position: Vec2,
    /// Size of the window.
    pub size: Vec2,
    /// The dock node tree inside this floating window.
    pub root: DockNode,
    /// Whether the window is minimized.
    pub minimized: bool,
    /// Whether the window is maximized.
    pub maximized: bool,
    /// Whether the window can be resized.
    pub resizable: bool,
    /// Z-order for floating window stacking.
    pub z_order: i32,
    /// Whether the title bar is being dragged.
    #[serde(skip)]
    pub dragging: bool,
    /// Whether the window is being resized (and which edge).
    #[serde(skip)]
    pub resize_edge: Option<ResizeEdge>,
    /// Saved position/size before maximizing.
    #[serde(skip)]
    pub restore_rect: Option<(Vec2, Vec2)>,
}

/// Which edge of a floating window is being resized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeEdge {
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl ResizeEdge {
    /// Determine the resize edge for a point near the window border.
    pub fn from_point(point: Vec2, window_pos: Vec2, window_size: Vec2) -> Option<Self> {
        let border = RESIZE_BORDER_SIZE;
        let rect = Rect::new(window_pos, window_pos + window_size);
        if !rect.contains(point) {
            return None;
        }

        let on_left = point.x - rect.min.x < border;
        let on_right = rect.max.x - point.x < border;
        let on_top = point.y - rect.min.y < border;
        let on_bottom = rect.max.y - point.y < border;

        match (on_left, on_right, on_top, on_bottom) {
            (true, _, true, _) => Some(Self::TopLeft),
            (true, _, _, true) => Some(Self::BottomLeft),
            (_, true, true, _) => Some(Self::TopRight),
            (_, true, _, true) => Some(Self::BottomRight),
            (true, _, _, _) => Some(Self::Left),
            (_, true, _, _) => Some(Self::Right),
            (_, _, true, _) => Some(Self::Top),
            (_, _, _, true) => Some(Self::Bottom),
            _ => None,
        }
    }

    /// Returns the cursor resize direction for the given edge.
    pub fn is_horizontal(&self) -> bool {
        matches!(self, Self::Left | Self::Right)
    }

    pub fn is_vertical(&self) -> bool {
        matches!(self, Self::Top | Self::Bottom)
    }

    pub fn is_corner(&self) -> bool {
        matches!(
            self,
            Self::TopLeft | Self::TopRight | Self::BottomLeft | Self::BottomRight
        )
    }
}

impl FloatingWindow {
    /// Creates a new floating window.
    pub fn new(id: u64, title: impl Into<String>, position: Vec2, size: Vec2, root: DockNode) -> Self {
        Self {
            id,
            title: title.into(),
            position,
            size,
            root,
            minimized: false,
            maximized: false,
            resizable: true,
            z_order: 0,
            dragging: false,
            resize_edge: None,
            restore_rect: None,
        }
    }

    /// Returns the bounding rect of this window.
    pub fn rect(&self) -> Rect {
        Rect::new(self.position, self.position + self.size)
    }

    /// Returns the title bar rect.
    pub fn title_bar_rect(&self) -> Rect {
        Rect::new(
            self.position,
            Vec2::new(
                self.position.x + self.size.x,
                self.position.y + FLOATING_TITLE_BAR_HEIGHT,
            ),
        )
    }

    /// Returns the content rect (below the title bar).
    pub fn content_rect(&self) -> Rect {
        Rect::new(
            Vec2::new(
                self.position.x,
                self.position.y + FLOATING_TITLE_BAR_HEIGHT,
            ),
            self.position + self.size,
        )
    }

    /// Hit-test for the title bar.
    pub fn hit_test_title_bar(&self, point: Vec2) -> bool {
        self.title_bar_rect().contains(point) && !self.minimized
    }

    /// Hit-test for the entire window.
    pub fn hit_test(&self, point: Vec2) -> bool {
        self.rect().contains(point)
    }

    /// Toggle maximize/restore.
    pub fn toggle_maximize(&mut self, screen_size: Vec2) {
        if self.maximized {
            if let Some((pos, size)) = self.restore_rect.take() {
                self.position = pos;
                self.size = size;
            }
            self.maximized = false;
        } else {
            self.restore_rect = Some((self.position, self.size));
            self.position = Vec2::ZERO;
            self.size = screen_size;
            self.maximized = true;
        }
    }

    /// Apply resize delta based on the current resize edge.
    pub fn apply_resize(&mut self, delta: Vec2) {
        let min_size = Vec2::new(MIN_FLOATING_WIDTH, MIN_FLOATING_HEIGHT);
        if let Some(edge) = self.resize_edge {
            match edge {
                ResizeEdge::Left => {
                    let new_w = (self.size.x - delta.x).max(min_size.x);
                    let diff = self.size.x - new_w;
                    self.position.x += diff;
                    self.size.x = new_w;
                }
                ResizeEdge::Right => {
                    self.size.x = (self.size.x + delta.x).max(min_size.x);
                }
                ResizeEdge::Top => {
                    let new_h = (self.size.y - delta.y).max(min_size.y);
                    let diff = self.size.y - new_h;
                    self.position.y += diff;
                    self.size.y = new_h;
                }
                ResizeEdge::Bottom => {
                    self.size.y = (self.size.y + delta.y).max(min_size.y);
                }
                ResizeEdge::TopLeft => {
                    let new_w = (self.size.x - delta.x).max(min_size.x);
                    let new_h = (self.size.y - delta.y).max(min_size.y);
                    let diff_x = self.size.x - new_w;
                    let diff_y = self.size.y - new_h;
                    self.position.x += diff_x;
                    self.position.y += diff_y;
                    self.size.x = new_w;
                    self.size.y = new_h;
                }
                ResizeEdge::TopRight => {
                    self.size.x = (self.size.x + delta.x).max(min_size.x);
                    let new_h = (self.size.y - delta.y).max(min_size.y);
                    let diff_y = self.size.y - new_h;
                    self.position.y += diff_y;
                    self.size.y = new_h;
                }
                ResizeEdge::BottomLeft => {
                    let new_w = (self.size.x - delta.x).max(min_size.x);
                    let diff_x = self.size.x - new_w;
                    self.position.x += diff_x;
                    self.size.x = new_w;
                    self.size.y = (self.size.y + delta.y).max(min_size.y);
                }
                ResizeEdge::BottomRight => {
                    self.size.x = (self.size.x + delta.x).max(min_size.x);
                    self.size.y = (self.size.y + delta.y).max(min_size.y);
                }
            }
        }
    }

    /// Render the floating window.
    pub fn render(&self, draw_list: &mut DrawList, style: &DockStyle, font_size: f32) {
        if self.minimized {
            return;
        }

        // Window shadow
        let shadow_rect = Rect::new(
            self.position - Vec2::splat(4.0),
            self.position + self.size + Vec2::splat(4.0),
        );
        draw_list.draw_rect(shadow_rect, style.floating_shadow_color);

        // Window background
        let win_rect = self.rect();
        draw_list.draw_rounded_rect(
            win_rect,
            style.floating_background,
            CornerRadii::all(6.0),
            Border::new(style.floating_border_color, 1.0),
        );

        // Title bar
        let tb_rect = self.title_bar_rect();
        draw_list.draw_rounded_rect(
            tb_rect,
            style.floating_title_bar,
            CornerRadii::new(6.0, 6.0, 0.0, 0.0),
            Border::default(),
        );

        // Title text
        draw_list.draw_text(
            &self.title,
            Vec2::new(tb_rect.min.x + 10.0, tb_rect.min.y + 4.0),
            font_size,
            style.floating_title_text,
        );

        // Content area — render the inner dock tree
        let content = self.content_rect();
        draw_list.push_clip(content);
        self.root.render(draw_list, style, font_size);
        draw_list.pop_clip();
    }
}

// ---------------------------------------------------------------------------
// DockStyle — visual style configuration
// ---------------------------------------------------------------------------

/// Visual style for the docking system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockStyle {
    pub panel_background: Color,
    pub content_background: Color,
    pub tab_bar_background: Color,
    pub active_tab_background: Color,
    pub inactive_tab_background: Color,
    pub hovered_tab_background: Color,
    pub active_tab_text: Color,
    pub inactive_tab_text: Color,
    pub close_button_color: Color,
    pub close_button_hover_color: Color,
    pub dirty_indicator_color: Color,
    pub border_color: Color,
    pub divider_color: Color,
    pub divider_hover_color: Color,
    pub dock_preview_color: Color,
    pub floating_background: Color,
    pub floating_border_color: Color,
    pub floating_title_bar: Color,
    pub floating_title_text: Color,
    pub floating_shadow_color: Color,
}

impl DockStyle {
    /// Dark theme preset.
    pub fn dark_theme() -> Self {
        Self {
            panel_background: Color::from_hex("#1E1E2E"),
            content_background: Color::from_hex("#181825"),
            tab_bar_background: Color::from_hex("#11111B"),
            active_tab_background: Color::from_hex("#1E1E2E"),
            inactive_tab_background: Color::from_hex("#14141F"),
            hovered_tab_background: Color::from_hex("#252535"),
            active_tab_text: Color::from_hex("#CDD6F4"),
            inactive_tab_text: Color::from_hex("#6C7086"),
            close_button_color: Color::from_hex("#6C7086"),
            close_button_hover_color: Color::from_hex("#F38BA8"),
            dirty_indicator_color: Color::from_hex("#F9E2AF"),
            border_color: Color::from_hex("#313244"),
            divider_color: Color::from_hex("#313244"),
            divider_hover_color: Color::from_hex("#585B70"),
            dock_preview_color: Color::new(0.35, 0.55, 0.85, 0.3),
            floating_background: Color::from_hex("#1E1E2E"),
            floating_border_color: Color::from_hex("#45475A"),
            floating_title_bar: Color::from_hex("#181825"),
            floating_title_text: Color::from_hex("#CDD6F4"),
            floating_shadow_color: Color::new(0.0, 0.0, 0.0, 0.35),
        }
    }

    /// Light theme preset.
    pub fn light_theme() -> Self {
        Self {
            panel_background: Color::from_hex("#EFF1F5"),
            content_background: Color::from_hex("#E6E9EF"),
            tab_bar_background: Color::from_hex("#DCE0E8"),
            active_tab_background: Color::from_hex("#EFF1F5"),
            inactive_tab_background: Color::from_hex("#CCD0DA"),
            hovered_tab_background: Color::from_hex("#E6E9EF"),
            active_tab_text: Color::from_hex("#4C4F69"),
            inactive_tab_text: Color::from_hex("#8C8FA1"),
            close_button_color: Color::from_hex("#8C8FA1"),
            close_button_hover_color: Color::from_hex("#D20F39"),
            dirty_indicator_color: Color::from_hex("#DF8E1D"),
            border_color: Color::from_hex("#9CA0B0"),
            divider_color: Color::from_hex("#BCC0CC"),
            divider_hover_color: Color::from_hex("#8C8FA1"),
            dock_preview_color: Color::new(0.25, 0.45, 0.75, 0.3),
            floating_background: Color::from_hex("#EFF1F5"),
            floating_border_color: Color::from_hex("#9CA0B0"),
            floating_title_bar: Color::from_hex("#DCE0E8"),
            floating_title_text: Color::from_hex("#4C4F69"),
            floating_shadow_color: Color::new(0.0, 0.0, 0.0, 0.15),
        }
    }
}

impl Default for DockStyle {
    fn default() -> Self {
        Self::dark_theme()
    }
}

// ---------------------------------------------------------------------------
// DockDragState — tracks tab drag operations
// ---------------------------------------------------------------------------

/// State for an in-progress tab drag operation.
#[derive(Debug, Clone)]
pub struct DockDragState {
    /// The tab being dragged.
    pub tab_id: DockTabId,
    /// Node the tab was dragged from.
    pub source_node: DockNodeId,
    /// Tab index within the source node.
    pub source_index: usize,
    /// Current mouse position.
    pub mouse_pos: Vec2,
    /// Position where dragging started.
    pub start_pos: Vec2,
    /// Whether the drag has exceeded the threshold to start.
    pub active: bool,
    /// Current dock target preview.
    pub preview_target: Option<(DockNodeId, DockTarget)>,
}

impl DockDragState {
    pub fn new(tab_id: DockTabId, source_node: DockNodeId, source_index: usize, pos: Vec2) -> Self {
        Self {
            tab_id,
            source_node,
            source_index,
            mouse_pos: pos,
            start_pos: pos,
            active: false,
            preview_target: None,
        }
    }

    /// Returns `true` if the drag has exceeded the activation threshold.
    pub fn should_activate(&self) -> bool {
        let delta = self.mouse_pos - self.start_pos;
        delta.length() > DRAG_THRESHOLD
    }

    /// Update mouse position during drag.
    pub fn update_position(&mut self, pos: Vec2) {
        self.mouse_pos = pos;
        if !self.active && self.should_activate() {
            self.active = true;
        }
    }
}

/// State for an in-progress divider drag.
#[derive(Debug, Clone)]
pub struct DividerDragState {
    /// The split node whose divider is being dragged.
    pub node_id: DockNodeId,
    /// Direction of the split.
    pub direction: SplitDirection,
    /// Starting mouse position when the drag began.
    pub start_pos: Vec2,
    /// Starting ratio when the drag began.
    pub start_ratio: f32,
}

impl DividerDragState {
    pub fn new(
        node_id: DockNodeId,
        direction: SplitDirection,
        start_pos: Vec2,
        start_ratio: f32,
    ) -> Self {
        Self {
            node_id,
            direction,
            start_pos,
            start_ratio,
        }
    }

    /// Compute the new ratio based on a mouse position delta and the total
    /// size of the split's rect.
    pub fn compute_ratio(&self, current_pos: Vec2, total_size: f32) -> f32 {
        if total_size < 1.0 {
            return self.start_ratio;
        }
        let delta = match self.direction {
            SplitDirection::Horizontal => current_pos.x - self.start_pos.x,
            SplitDirection::Vertical => current_pos.y - self.start_pos.y,
        };
        let ratio_delta = delta / total_size;
        (self.start_ratio + ratio_delta).clamp(0.05, 0.95)
    }
}

// ---------------------------------------------------------------------------
// DockState — complete docking layout state
// ---------------------------------------------------------------------------

/// Complete docking layout state, including the main tree, floating windows,
/// and interaction tracking.
pub struct DockState {
    /// Root of the dock tree.
    pub root: DockNode,
    /// Floating (detached) windows.
    pub floating_windows: Vec<FloatingWindow>,
    /// Next unique id for nodes.
    next_node_id: u64,
    /// Next unique id for tabs.
    next_tab_id: u64,
    /// Next unique id for floating windows.
    next_window_id: u64,
    /// Current tab drag state, if any.
    pub tab_drag: Option<DockDragState>,
    /// Current divider drag state, if any.
    pub divider_drag: Option<DividerDragState>,
    /// The currently hovered tab (for highlighting).
    pub hovered_tab: Option<DockTabId>,
    /// The currently hovered divider node.
    pub hovered_divider: Option<DockNodeId>,
    /// Visual style.
    pub style: DockStyle,
    /// Font size for tab headers.
    pub font_size: f32,
    /// The focused/active leaf node.
    pub focused_leaf: Option<DockNodeId>,
}

impl DockState {
    /// Creates a new empty dock state with a single empty root leaf.
    pub fn new() -> Self {
        Self {
            root: DockNode::empty_leaf(DockNodeId::new(0)),
            floating_windows: Vec::new(),
            next_node_id: 1,
            next_tab_id: 0,
            next_window_id: 0,
            tab_drag: None,
            divider_drag: None,
            hovered_tab: None,
            hovered_divider: None,
            style: DockStyle::dark_theme(),
            font_size: 13.0,
            focused_leaf: None,
        }
    }

    /// Allocate a new unique node id.
    pub fn next_node_id(&mut self) -> DockNodeId {
        let id = DockNodeId::new(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// Allocate a new unique tab id.
    pub fn next_tab_id(&mut self) -> DockTabId {
        let id = DockTabId::new(self.next_tab_id);
        self.next_tab_id += 1;
        id
    }

    /// Allocate a new unique window id.
    pub fn next_window_id(&mut self) -> u64 {
        let id = self.next_window_id;
        self.next_window_id += 1;
        id
    }

    // -- Tab operations -----------------------------------------------------

    /// Find a tab anywhere in the dock state (main tree + floating windows).
    pub fn find_tab(&self, tab_id: DockTabId) -> Option<&DockTab> {
        if let Some(tab) = self.root.get_tab(tab_id) {
            return Some(tab);
        }
        for win in &self.floating_windows {
            if let Some(tab) = win.root.get_tab(tab_id) {
                return Some(tab);
            }
        }
        None
    }

    /// Find a tab mutably.
    pub fn find_tab_mut(&mut self, tab_id: DockTabId) -> Option<&mut DockTab> {
        if let Some(tab) = self.root.get_tab_mut(tab_id) {
            return Some(tab);
        }
        for win in &mut self.floating_windows {
            if let Some(tab) = win.root.get_tab_mut(tab_id) {
                return Some(tab);
            }
        }
        None
    }

    /// Locate which node contains a tab. Returns (node_id, tab_index).
    pub fn locate_tab(&self, tab_id: DockTabId) -> Option<(DockNodeId, usize)> {
        if let Some(result) = self.root.find_tab(tab_id) {
            return Some(result);
        }
        for win in &self.floating_windows {
            if let Some(result) = win.root.find_tab(tab_id) {
                return Some(result);
            }
        }
        None
    }

    /// Add a tab to a specific node.
    pub fn add_tab_to_node(&mut self, node_id: DockNodeId, tab: DockTab) -> bool {
        if self.root.add_tab(node_id, tab.clone()) {
            return true;
        }
        for win in &mut self.floating_windows {
            if win.root.add_tab(node_id, tab.clone()) {
                return true;
            }
        }
        false
    }

    /// Close (remove) a tab by id. Returns the removed tab.
    pub fn close_tab(&mut self, tab_id: DockTabId) -> Option<DockTab> {
        if let Some(tab) = self.root.remove_tab(tab_id) {
            self.root.collapse_empty();
            return Some(tab);
        }
        for win in &mut self.floating_windows {
            if let Some(tab) = win.root.remove_tab(tab_id) {
                win.root.collapse_empty();
                return Some(tab);
            }
        }
        // Clean up empty floating windows
        self.floating_windows.retain(|w| !w.root.is_empty());
        None
    }

    /// Move a tab from one node to another. If `target` is Center, the tab is
    /// added to the target node's tab well. If it is a directional target, the
    /// target node is split and the tab is placed in the new panel.
    pub fn move_tab(
        &mut self,
        tab_id: DockTabId,
        target_node_id: DockNodeId,
        target: DockTarget,
    ) -> bool {
        // Remove the tab from wherever it currently lives.
        let tab = if let Some(t) = self.root.remove_tab(tab_id) {
            t
        } else {
            let mut found = None;
            for win in &mut self.floating_windows {
                if let Some(t) = win.root.remove_tab(tab_id) {
                    found = Some(t);
                    break;
                }
            }
            match found {
                Some(t) => t,
                None => return false,
            }
        };

        match target {
            DockTarget::Center => {
                // Add to target node's tab well.
                let result = self.add_tab_to_node(target_node_id, tab);
                self.root.collapse_empty();
                self.floating_windows.retain(|w| !w.root.is_empty());
                result
            }
            DockTarget::Float => {
                // Create a new floating window.
                let win_id = self.next_window_id();
                let node_id = self.next_node_id();
                let node = DockNode::single_tab(node_id, tab.clone());
                let window = FloatingWindow::new(
                    win_id,
                    tab.title.clone(),
                    Vec2::new(200.0, 200.0),
                    Vec2::new(400.0, 300.0),
                    node,
                );
                self.floating_windows.push(window);
                self.root.collapse_empty();
                true
            }
            DockTarget::Left | DockTarget::Right | DockTarget::Top | DockTarget::Bottom => {
                self.split_and_dock(tab, target_node_id, target)
            }
        }
    }

    /// Split a target node and place a tab in the new panel.
    fn split_and_dock(
        &mut self,
        tab: DockTab,
        target_node_id: DockNodeId,
        target: DockTarget,
    ) -> bool {
        let new_leaf_id = self.next_node_id();
        let split_id = self.next_node_id();
        let new_leaf = DockNode::single_tab(new_leaf_id, tab);

        // Try to split in the main tree
        if self.split_node_in_tree(&mut self.root.clone(), target_node_id, split_id, new_leaf.clone(), target) {
            // Need to actually apply the split. We do this by finding and
            // replacing the target node in the real tree.
            self.apply_split_at_node(target_node_id, split_id, new_leaf, target);
            self.root.collapse_empty();
            return true;
        }

        // Try floating windows
        for win in &mut self.floating_windows {
            if win.root.find_node(target_node_id).is_some() {
                Self::apply_split_at_node_in_tree(
                    &mut win.root,
                    target_node_id,
                    split_id,
                    new_leaf,
                    target,
                );
                win.root.collapse_empty();
                return true;
            }
        }

        false
    }

    fn split_node_in_tree(
        &self,
        tree: &DockNode,
        target_id: DockNodeId,
        _split_id: DockNodeId,
        _new_leaf: DockNode,
        _target: DockTarget,
    ) -> bool {
        tree.find_node(target_id).is_some()
    }

    fn apply_split_at_node(
        &mut self,
        target_id: DockNodeId,
        split_id: DockNodeId,
        new_leaf: DockNode,
        target: DockTarget,
    ) {
        Self::apply_split_at_node_in_tree(
            &mut self.root,
            target_id,
            split_id,
            new_leaf,
            target,
        );
    }

    fn apply_split_at_node_in_tree(
        tree: &mut DockNode,
        target_id: DockNodeId,
        split_id: DockNodeId,
        new_leaf: DockNode,
        target: DockTarget,
    ) {
        // If this node is the target, replace it with a split containing
        // itself and the new leaf.
        if tree.id() == target_id {
            let old_node = std::mem::replace(tree, DockNode::empty_leaf(DockNodeId::INVALID));

            *tree = match target {
                DockTarget::Left => DockNode::horizontal_split(
                    split_id,
                    new_leaf,
                    old_node,
                    0.3,
                ),
                DockTarget::Right => DockNode::horizontal_split(
                    split_id,
                    old_node,
                    new_leaf,
                    0.7,
                ),
                DockTarget::Top => DockNode::vertical_split(
                    split_id,
                    new_leaf,
                    old_node,
                    0.3,
                ),
                DockTarget::Bottom => DockNode::vertical_split(
                    split_id,
                    old_node,
                    new_leaf,
                    0.7,
                ),
                _ => {
                    // Should not happen — Center and Float are handled elsewhere.
                    old_node
                }
            };
            return;
        }

        // Otherwise recurse into children.
        match tree {
            DockNode::Leaf { .. } => {}
            DockNode::HorizontalSplit { left, right, .. } => {
                Self::apply_split_at_node_in_tree(left, target_id, split_id, new_leaf.clone(), target);
                Self::apply_split_at_node_in_tree(right, target_id, split_id, new_leaf, target);
            }
            DockNode::VerticalSplit { top, bottom, .. } => {
                Self::apply_split_at_node_in_tree(top, target_id, split_id, new_leaf.clone(), target);
                Self::apply_split_at_node_in_tree(bottom, target_id, split_id, new_leaf, target);
            }
        }
    }

    /// Split a node in the given direction with the given ratio. The existing
    /// content stays in the "first" position and a new empty leaf is created
    /// in the "second" position.
    pub fn split_node(
        &mut self,
        node_id: DockNodeId,
        direction: SplitDirection,
        ratio: f32,
    ) -> Option<DockNodeId> {
        let new_leaf_id = self.next_node_id();
        let split_id = self.next_node_id();
        let new_leaf = DockNode::empty_leaf(new_leaf_id);

        let target = match direction {
            SplitDirection::Horizontal => DockTarget::Right,
            SplitDirection::Vertical => DockTarget::Bottom,
        };

        // Apply to main tree
        if self.root.find_node(node_id).is_some() {
            Self::apply_split_at_node_in_tree(
                &mut self.root,
                node_id,
                split_id,
                new_leaf,
                target,
            );
            // Update the ratio
            if let Some(split_node) = self.root.find_node_mut(split_id) {
                split_node.set_ratio(ratio);
            }
            return Some(new_leaf_id);
        }

        // Try floating windows
        for win in &mut self.floating_windows {
            if win.root.find_node(node_id).is_some() {
                Self::apply_split_at_node_in_tree(
                    &mut win.root,
                    node_id,
                    split_id,
                    new_leaf,
                    target,
                );
                if let Some(split_node) = win.root.find_node_mut(split_id) {
                    split_node.set_ratio(ratio);
                }
                return Some(new_leaf_id);
            }
        }

        None
    }

    // -- Layout and rendering -----------------------------------------------

    /// Compute layout for the entire dock state within the given viewport.
    pub fn compute_layout(&mut self, viewport: Rect) {
        self.root.compute_layout(viewport);
        self.root.compute_tab_headers(self.font_size);

        for win in &mut self.floating_windows {
            if !win.minimized {
                let content = win.content_rect();
                win.root.compute_layout(content);
                win.root.compute_tab_headers(self.font_size);
            }
        }
    }

    /// Render the entire dock state.
    pub fn render(&self, draw_list: &mut DrawList) {
        // Render main tree
        self.root.render(draw_list, &self.style, self.font_size);

        // Render dock preview overlay (if dragging a tab)
        if let Some(ref drag) = self.tab_drag {
            if drag.active {
                if let Some((node_id, target)) = drag.preview_target {
                    self.render_dock_preview(draw_list, node_id, target);
                }

                // Render the floating tab preview
                self.render_dragged_tab(draw_list, drag);
            }
        }

        // Render divider hover highlight
        if let Some(node_id) = self.hovered_divider {
            if let Some(node) = self.root.find_node(node_id) {
                if let Some(div_rect) = node.divider_rect() {
                    draw_list.draw_rect(div_rect, self.style.divider_hover_color);
                }
            }
        }

        // Render floating windows (back to front)
        let mut sorted_indices: Vec<usize> = (0..self.floating_windows.len()).collect();
        sorted_indices.sort_by_key(|i| self.floating_windows[*i].z_order);
        for i in sorted_indices {
            self.floating_windows[i].render(draw_list, &self.style, self.font_size);
        }
    }

    /// Render the dock preview overlay when dragging a tab.
    fn render_dock_preview(
        &self,
        draw_list: &mut DrawList,
        node_id: DockNodeId,
        target: DockTarget,
    ) {
        let node = if let Some(n) = self.root.find_node(node_id) {
            n
        } else {
            return;
        };

        let node_rect = node.rect();
        let preview_rect = match target {
            DockTarget::Left => Rect::new(
                node_rect.min,
                Vec2::new(
                    node_rect.min.x + node_rect.width() * 0.3,
                    node_rect.max.y,
                ),
            ),
            DockTarget::Right => Rect::new(
                Vec2::new(
                    node_rect.max.x - node_rect.width() * 0.3,
                    node_rect.min.y,
                ),
                node_rect.max,
            ),
            DockTarget::Top => Rect::new(
                node_rect.min,
                Vec2::new(
                    node_rect.max.x,
                    node_rect.min.y + node_rect.height() * 0.3,
                ),
            ),
            DockTarget::Bottom => Rect::new(
                Vec2::new(
                    node_rect.min.x,
                    node_rect.max.y - node_rect.height() * 0.3,
                ),
                node_rect.max,
            ),
            DockTarget::Center => node_rect,
            DockTarget::Float => return,
        };

        draw_list.draw_rounded_rect(
            preview_rect,
            self.style.dock_preview_color,
            CornerRadii::all(4.0),
            Border::new(
                Color::new(
                    self.style.dock_preview_color.r,
                    self.style.dock_preview_color.g,
                    self.style.dock_preview_color.b,
                    0.6,
                ),
                2.0,
            ),
        );
    }

    /// Render the tab currently being dragged.
    fn render_dragged_tab(&self, draw_list: &mut DrawList, drag: &DockDragState) {
        let tab = match self.find_tab(drag.tab_id) {
            Some(t) => t,
            None => return,
        };

        let tab_w = tab.compute_header_width(self.font_size);
        let tab_h = 28.0;
        let half_w = tab_w * 0.5;
        let half_h = tab_h * 0.5;

        let drag_rect = Rect::new(
            Vec2::new(drag.mouse_pos.x - half_w, drag.mouse_pos.y - half_h),
            Vec2::new(drag.mouse_pos.x + half_w, drag.mouse_pos.y + half_h),
        );

        // Semi-transparent dragged tab
        draw_list.draw_rounded_rect(
            drag_rect,
            self.style.active_tab_background.with_alpha(0.8),
            CornerRadii::all(4.0),
            Border::new(self.style.border_color, 1.0),
        );
        draw_list.draw_text(
            &tab.title,
            Vec2::new(drag_rect.min.x + 8.0, drag_rect.min.y + 6.0),
            self.font_size,
            self.style.active_tab_text,
        );
    }

    // -- Input handling -----------------------------------------------------

    /// Handle mouse down at the given position. Returns `true` if the event
    /// was consumed.
    pub fn on_mouse_down(&mut self, pos: Vec2) -> bool {
        // Check floating windows first (front to back)
        for i in (0..self.floating_windows.len()).rev() {
            let win = &self.floating_windows[i];
            if win.hit_test(pos) {
                // Bring to front
                let max_z = self
                    .floating_windows
                    .iter()
                    .map(|w| w.z_order)
                    .max()
                    .unwrap_or(0);
                self.floating_windows[i].z_order = max_z + 1;

                // Check title bar drag
                if win.hit_test_title_bar(pos) {
                    self.floating_windows[i].dragging = true;
                    return true;
                }

                // Check resize edge
                if win.resizable {
                    if let Some(edge) = ResizeEdge::from_point(pos, win.position, win.size) {
                        self.floating_windows[i].resize_edge = Some(edge);
                        return true;
                    }
                }

                // Check tabs in floating window
                if let Some((node_id, tab_idx)) = win.root.hit_test_tab(pos) {
                    let tab_id = if let Some(node) = win.root.find_node(node_id) {
                        match node {
                            DockNode::Leaf { tabs, .. } => tabs.get(tab_idx).map(|t| t.id),
                            _ => None,
                        }
                    } else {
                        None
                    };

                    if let Some(tid) = tab_id {
                        // Check close button
                        if let Some(close_id) = win.root.hit_test_close_button(pos, self.font_size)
                        {
                            self.close_tab(close_id);
                            return true;
                        }

                        // Start tab drag
                        self.tab_drag = Some(DockDragState::new(tid, node_id, tab_idx, pos));

                        // Activate the tab
                        self.floating_windows[i]
                            .root
                            .set_active_tab(node_id, tab_idx);
                        return true;
                    }
                }

                return true;
            }
        }

        // Check divider hit in main tree
        if let Some((node_id, direction)) = self.root.hit_test_divider(pos) {
            let ratio = if let Some(node) = self.root.find_node(node_id) {
                match node {
                    DockNode::HorizontalSplit { ratio, .. }
                    | DockNode::VerticalSplit { ratio, .. } => *ratio,
                    _ => 0.5,
                }
            } else {
                0.5
            };
            self.divider_drag = Some(DividerDragState::new(node_id, direction, pos, ratio));
            return true;
        }

        // Check tab hit in main tree
        if let Some((node_id, tab_idx)) = self.root.hit_test_tab(pos) {
            let tab_id = if let Some(node) = self.root.find_node(node_id) {
                match node {
                    DockNode::Leaf { tabs, .. } => tabs.get(tab_idx).map(|t| t.id),
                    _ => None,
                }
            } else {
                None
            };

            if let Some(tid) = tab_id {
                // Check close button first
                if let Some(close_id) = self.root.hit_test_close_button(pos, self.font_size) {
                    self.close_tab(close_id);
                    return true;
                }

                // Start tab drag
                self.tab_drag = Some(DockDragState::new(tid, node_id, tab_idx, pos));

                // Activate the tab
                self.root.set_active_tab(node_id, tab_idx);
                self.focused_leaf = Some(node_id);
                return true;
            }
        }

        // Check content area click (to set focus)
        if let Some(node_id) = self.root.hit_test_content(pos) {
            self.focused_leaf = Some(node_id);
            return true;
        }

        false
    }

    /// Handle mouse move at the given position. Returns `true` if a drag is
    /// active.
    pub fn on_mouse_move(&mut self, pos: Vec2) -> bool {
        // Handle floating window dragging
        for win in &mut self.floating_windows {
            if win.dragging {
                let delta = pos - win.position - Vec2::new(win.size.x * 0.5, 10.0);
                win.position += delta;
                return true;
            }
            if win.resize_edge.is_some() {
                let prev_pos = win.position + win.size * 0.5;
                let delta = pos - prev_pos;
                win.apply_resize(delta);
                return true;
            }
        }

        // Handle divider drag
        if let Some(ref drag) = self.divider_drag {
            let node_id = drag.node_id;
            let total_size = if let Some(node) = self.root.find_node(node_id) {
                match drag.direction {
                    SplitDirection::Horizontal => node.rect().width(),
                    SplitDirection::Vertical => node.rect().height(),
                }
            } else {
                1.0
            };
            let new_ratio = drag.compute_ratio(pos, total_size);
            if let Some(node) = self.root.find_node_mut(node_id) {
                node.set_ratio(new_ratio);
            }
            return true;
        }

        // Handle tab drag
        if let Some(ref mut drag) = self.tab_drag {
            drag.update_position(pos);
            if drag.active {
                // Update dock preview target
                drag.preview_target = self.root.compute_dock_target(pos);
            }
            return drag.active;
        }

        // Hover detection for dividers
        if let Some((node_id, _)) = self.root.hit_test_divider(pos) {
            self.hovered_divider = Some(node_id);
        } else {
            self.hovered_divider = None;
        }

        // Hover detection for tabs
        if let Some((node_id, tab_idx)) = self.root.hit_test_tab(pos) {
            if let Some(node) = self.root.find_node(node_id) {
                match node {
                    DockNode::Leaf { tabs, .. } => {
                        self.hovered_tab = tabs.get(tab_idx).map(|t| t.id);
                    }
                    _ => {
                        self.hovered_tab = None;
                    }
                }
            }
        } else {
            self.hovered_tab = None;
        }

        false
    }

    /// Handle mouse up. Returns `true` if a drag operation was completed.
    pub fn on_mouse_up(&mut self, pos: Vec2) -> bool {
        // End floating window operations
        for win in &mut self.floating_windows {
            win.dragging = false;
            win.resize_edge = None;
        }

        // End divider drag
        if self.divider_drag.take().is_some() {
            return true;
        }

        // End tab drag
        if let Some(drag) = self.tab_drag.take() {
            if drag.active {
                // Dock the tab at the preview target
                if let Some((_node_id, target)) = drag.preview_target {
                    self.move_tab(drag.tab_id, _node_id, target);
                } else {
                    // Float the tab
                    self.move_tab(drag.tab_id, DockNodeId::INVALID, DockTarget::Float);
                }
                return true;
            }
        }

        false
    }

    /// Handle a double-click on the dock system. Used for toggling floating
    /// window maximize.
    pub fn on_double_click(&mut self, pos: Vec2) -> bool {
        for win in &mut self.floating_windows {
            if win.hit_test_title_bar(pos) {
                let screen_size = Vec2::new(1920.0, 1080.0); // TODO: get from context
                win.toggle_maximize(screen_size);
                return true;
            }
        }
        false
    }
}

impl Default for DockState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DockLayout — serializable layout presets
// ---------------------------------------------------------------------------

/// A serializable dock layout that can be saved/loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockLayout {
    /// The root dock node tree.
    pub root: DockNode,
    /// Floating window definitions.
    pub floating_windows: Vec<FloatingWindowDef>,
}

/// Serializable floating window definition (without runtime state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatingWindowDef {
    pub title: String,
    pub position: (f32, f32),
    pub size: (f32, f32),
    pub root: DockNode,
}

impl DockLayout {
    /// Serialize the layout to a JSON string.
    pub fn serialize(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Deserialize a layout from a JSON string.
    pub fn deserialize(json: &str) -> Option<Self> {
        serde_json::from_str(json).ok()
    }

    /// Apply this layout to a dock state.
    pub fn apply_to(&self, state: &mut DockState) {
        state.root = self.root.clone();
        state.floating_windows.clear();
        for (i, def) in self.floating_windows.iter().enumerate() {
            let win = FloatingWindow::new(
                i as u64,
                def.title.clone(),
                Vec2::new(def.position.0, def.position.1),
                Vec2::new(def.size.0, def.size.1),
                def.root.clone(),
            );
            state.floating_windows.push(win);
        }
    }

    /// Create a layout from the current dock state.
    pub fn from_state(state: &DockState) -> Self {
        let floating_windows = state
            .floating_windows
            .iter()
            .map(|w| FloatingWindowDef {
                title: w.title.clone(),
                position: (w.position.x, w.position.y),
                size: (w.size.x, w.size.y),
                root: w.root.clone(),
            })
            .collect();

        Self {
            root: state.root.clone(),
            floating_windows,
        }
    }

    /// Default editor layout preset.
    ///
    /// ```text
    ///  ┌──────────┬─────────────────────┬──────────┐
    ///  │          │                     │          │
    ///  │ Outliner │     Viewport        │ Details  │
    ///  │          │                     │          │
    ///  │          ├─────────────────────┤          │
    ///  │          │ Content Browser     │          │
    ///  └──────────┴─────────────────────┴──────────┘
    /// ```
    pub fn preset_default() -> Self {
        let mut id_counter: u64 = 0;
        let mut next_id = || {
            id_counter += 1;
            id_counter
        };

        let outliner_tab = DockTab::new(DockTabId::new(next_id()), "Outliner")
            .with_closable(false);
        let viewport_tab = DockTab::new(DockTabId::new(next_id()), "Viewport")
            .with_closable(false);
        let content_tab = DockTab::new(DockTabId::new(next_id()), "Content Browser")
            .with_closable(false);
        let details_tab = DockTab::new(DockTabId::new(next_id()), "Details")
            .with_closable(false);

        let outliner = DockNode::single_tab(DockNodeId::new(next_id()), outliner_tab);
        let viewport = DockNode::single_tab(DockNodeId::new(next_id()), viewport_tab);
        let content = DockNode::single_tab(DockNodeId::new(next_id()), content_tab);
        let details = DockNode::single_tab(DockNodeId::new(next_id()), details_tab);

        // Viewport + Content Browser (vertical split, 70/30)
        let center = DockNode::vertical_split(
            DockNodeId::new(next_id()),
            viewport,
            content,
            0.7,
        );

        // Outliner | Center (horizontal split, 20/80)
        let left_center = DockNode::horizontal_split(
            DockNodeId::new(next_id()),
            outliner,
            center,
            0.2,
        );

        // LeftCenter | Details (horizontal split, 75/25)
        let root = DockNode::horizontal_split(
            DockNodeId::new(next_id()),
            left_center,
            details,
            0.75,
        );

        Self {
            root,
            floating_windows: Vec::new(),
        }
    }

    /// Debug-focused layout preset.
    ///
    /// ```text
    ///  ┌──────────────────────────────────────────┐
    ///  │                Viewport                  │
    ///  ├──────────┬──────────────┬────────────────┤
    ///  │ Console  │ Breakpoints  │ Watch / Locals │
    ///  └──────────┴──────────────┴────────────────┘
    /// ```
    pub fn preset_debug() -> Self {
        let mut id_counter: u64 = 100;
        let mut next_id = || {
            id_counter += 1;
            id_counter
        };

        let viewport_tab = DockTab::new(DockTabId::new(next_id()), "Viewport");
        let console_tab = DockTab::new(DockTabId::new(next_id()), "Console");
        let breakpoints_tab = DockTab::new(DockTabId::new(next_id()), "Breakpoints");
        let watch_tab = DockTab::new(DockTabId::new(next_id()), "Watch");
        let locals_tab = DockTab::new(DockTabId::new(next_id()), "Locals");
        let callstack_tab = DockTab::new(DockTabId::new(next_id()), "Call Stack");

        let viewport = DockNode::single_tab(DockNodeId::new(next_id()), viewport_tab);
        let console = DockNode::single_tab(DockNodeId::new(next_id()), console_tab);
        let breakpoints = DockNode::leaf(
            DockNodeId::new(next_id()),
            vec![breakpoints_tab, callstack_tab],
        );
        let watch = DockNode::leaf(
            DockNodeId::new(next_id()),
            vec![watch_tab, locals_tab],
        );

        // Console | Breakpoints | Watch (horizontal splits)
        let bp_watch = DockNode::horizontal_split(
            DockNodeId::new(next_id()),
            breakpoints,
            watch,
            0.5,
        );
        let bottom = DockNode::horizontal_split(
            DockNodeId::new(next_id()),
            console,
            bp_watch,
            0.33,
        );

        let root = DockNode::vertical_split(
            DockNodeId::new(next_id()),
            viewport,
            bottom,
            0.65,
        );

        Self {
            root,
            floating_windows: Vec::new(),
        }
    }

    /// Animation-focused layout preset.
    ///
    /// ```text
    ///  ┌──────────┬─────────────────────┬──────────┐
    ///  │          │                     │          │
    ///  │ Skeleton │     Viewport        │ Anim     │
    ///  │ Tree     │                     │ Details  │
    ///  │          ├─────────────────────┤          │
    ///  │          │     Timeline        │          │
    ///  └──────────┴─────────────────────┴──────────┘
    /// ```
    pub fn preset_animation() -> Self {
        let mut id_counter: u64 = 200;
        let mut next_id = || {
            id_counter += 1;
            id_counter
        };

        let skeleton_tab = DockTab::new(DockTabId::new(next_id()), "Skeleton Tree");
        let viewport_tab = DockTab::new(DockTabId::new(next_id()), "Viewport");
        let timeline_tab = DockTab::new(DockTabId::new(next_id()), "Timeline");
        let curves_tab = DockTab::new(DockTabId::new(next_id()), "Curve Editor");
        let details_tab = DockTab::new(DockTabId::new(next_id()), "Anim Details");
        let notifies_tab = DockTab::new(DockTabId::new(next_id()), "Notifies");

        let skeleton = DockNode::single_tab(DockNodeId::new(next_id()), skeleton_tab);
        let viewport = DockNode::single_tab(DockNodeId::new(next_id()), viewport_tab);
        let timeline = DockNode::leaf(
            DockNodeId::new(next_id()),
            vec![timeline_tab, curves_tab, notifies_tab],
        );
        let details = DockNode::single_tab(DockNodeId::new(next_id()), details_tab);

        // Viewport + Timeline (vertical split, 60/40)
        let center = DockNode::vertical_split(
            DockNodeId::new(next_id()),
            viewport,
            timeline,
            0.6,
        );

        // Skeleton | Center (horizontal split, 18/82)
        let left_center = DockNode::horizontal_split(
            DockNodeId::new(next_id()),
            skeleton,
            center,
            0.18,
        );

        // LeftCenter | Details (horizontal split, 78/22)
        let root = DockNode::horizontal_split(
            DockNodeId::new(next_id()),
            left_center,
            details,
            0.78,
        );

        Self {
            root,
            floating_windows: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// DockEvent — events emitted by the dock system
// ---------------------------------------------------------------------------

/// Events emitted by the dock system that the host application should handle.
#[derive(Debug, Clone)]
pub enum DockEvent {
    /// A tab was activated (selected).
    TabActivated {
        tab_id: DockTabId,
        node_id: DockNodeId,
    },
    /// A tab was closed.
    TabClosed {
        tab_id: DockTabId,
    },
    /// A tab was moved to a new location.
    TabMoved {
        tab_id: DockTabId,
        from_node: DockNodeId,
        to_node: DockNodeId,
        target: DockTarget,
    },
    /// A tab was floated into a new window.
    TabFloated {
        tab_id: DockTabId,
        window_id: u64,
    },
    /// A tab was docked from a floating window back into the tree.
    TabDocked {
        tab_id: DockTabId,
        from_window: u64,
        to_node: DockNodeId,
    },
    /// A node was split.
    NodeSplit {
        node_id: DockNodeId,
        new_node_id: DockNodeId,
        direction: SplitDirection,
    },
    /// The divider ratio was changed.
    DividerMoved {
        node_id: DockNodeId,
        new_ratio: f32,
    },
    /// A floating window was moved.
    WindowMoved {
        window_id: u64,
        position: Vec2,
    },
    /// A floating window was resized.
    WindowResized {
        window_id: u64,
        size: Vec2,
    },
    /// The focused leaf changed.
    FocusChanged {
        node_id: Option<DockNodeId>,
    },
    /// The layout was changed (any structural modification).
    LayoutChanged,
}

// ---------------------------------------------------------------------------
// DockArea — top-level container widget
// ---------------------------------------------------------------------------

/// Top-level container that manages the entire docking area. This is the
/// widget that an editor embeds to get the full docking experience.
pub struct DockArea {
    /// The dock state.
    pub state: DockState,
    /// Pending events to be polled by the host.
    pub events: Vec<DockEvent>,
    /// The viewport rect.
    pub viewport: Rect,
}

impl DockArea {
    /// Creates a new dock area with the default layout.
    pub fn new() -> Self {
        let layout = DockLayout::preset_default();
        let mut state = DockState::new();
        layout.apply_to(&mut state);

        Self {
            state,
            events: Vec::new(),
            viewport: Rect::new(Vec2::ZERO, Vec2::new(1920.0, 1080.0)),
        }
    }

    /// Creates a dock area with a specific layout.
    pub fn with_layout(layout: DockLayout) -> Self {
        let mut state = DockState::new();
        layout.apply_to(&mut state);

        Self {
            state,
            events: Vec::new(),
            viewport: Rect::new(Vec2::ZERO, Vec2::new(1920.0, 1080.0)),
        }
    }

    /// Set the viewport rect.
    pub fn set_viewport(&mut self, viewport: Rect) {
        self.viewport = viewport;
    }

    /// Update layout and process pending state.
    pub fn update(&mut self) {
        self.state.compute_layout(self.viewport);
    }

    /// Render to a draw list.
    pub fn render(&self, draw_list: &mut DrawList) {
        self.state.render(draw_list);
    }

    /// Take all pending events.
    pub fn take_events(&mut self) -> Vec<DockEvent> {
        std::mem::take(&mut self.events)
    }

    /// Save the current layout.
    pub fn save_layout(&self) -> DockLayout {
        DockLayout::from_state(&self.state)
    }

    /// Load a layout.
    pub fn load_layout(&mut self, layout: DockLayout) {
        layout.apply_to(&mut self.state);
        self.events.push(DockEvent::LayoutChanged);
    }

    /// Add a new tab to the focused leaf (or the first leaf if none focused).
    pub fn add_tab(&mut self, tab: DockTab) -> bool {
        let target = self
            .state
            .focused_leaf
            .unwrap_or_else(|| {
                self.state
                    .root
                    .all_leaf_ids()
                    .first()
                    .copied()
                    .unwrap_or(DockNodeId::INVALID)
            });

        if target.is_invalid() {
            return false;
        }

        self.state.add_tab_to_node(target, tab)
    }

    /// Close a tab.
    pub fn close_tab(&mut self, tab_id: DockTabId) -> Option<DockTab> {
        let tab = self.state.close_tab(tab_id);
        if tab.is_some() {
            self.events.push(DockEvent::TabClosed { tab_id });
        }
        tab
    }

    /// Handle mouse down.
    pub fn on_mouse_down(&mut self, pos: Vec2) -> bool {
        self.state.on_mouse_down(pos)
    }

    /// Handle mouse move.
    pub fn on_mouse_move(&mut self, pos: Vec2) -> bool {
        self.state.on_mouse_move(pos)
    }

    /// Handle mouse up.
    pub fn on_mouse_up(&mut self, pos: Vec2) -> bool {
        self.state.on_mouse_up(pos)
    }
}

impl Default for DockArea {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum size for any panel in pixels.
const MIN_PANEL_SIZE: f32 = 50.0;

/// Thickness of the divider between split panes in pixels.
const DIVIDER_THICKNESS: f32 = 4.0;

/// Padding around the tab bar.
const TAB_PADDING: f32 = 4.0;

/// Gap between tabs.
const TAB_GAP: f32 = 1.0;

/// Fraction of a leaf's edge that counts as a directional dock zone.
const DOCK_ZONE_FRACTION: f32 = 0.25;

/// Distance threshold before a tab drag becomes active.
const DRAG_THRESHOLD: f32 = 5.0;

/// Height of the floating window title bar.
const FLOATING_TITLE_BAR_HEIGHT: f32 = 28.0;

/// Minimum floating window width.
const MIN_FLOATING_WIDTH: f32 = 150.0;

/// Minimum floating window height.
const MIN_FLOATING_HEIGHT: f32 = 100.0;

/// Resize border detection size.
const RESIZE_BORDER_SIZE: f32 = 6.0;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute the close button rect within a tab header.
fn compute_close_button_rect(tab_rect: &Rect, font_size: f32) -> Rect {
    let size = font_size * 0.8;
    let margin = 4.0;
    Rect::new(
        Vec2::new(
            tab_rect.max.x - size - margin,
            tab_rect.min.y + (tab_rect.height() - size) * 0.5,
        ),
        Vec2::new(
            tab_rect.max.x - margin,
            tab_rect.min.y + (tab_rect.height() + size) * 0.5,
        ),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tab(id: u64, title: &str) -> DockTab {
        DockTab::new(DockTabId::new(id), title)
    }

    #[test]
    fn test_dock_node_leaf_creation() {
        let tab = make_tab(1, "Test");
        let node = DockNode::single_tab(DockNodeId::new(0), tab);
        assert!(node.is_leaf());
        assert_eq!(node.tab_count(), 1);
        assert!(!node.is_empty());
    }

    #[test]
    fn test_dock_node_split() {
        let tab1 = make_tab(1, "Left");
        let tab2 = make_tab(2, "Right");
        let left = DockNode::single_tab(DockNodeId::new(0), tab1);
        let right = DockNode::single_tab(DockNodeId::new(1), tab2);
        let split = DockNode::horizontal_split(DockNodeId::new(2), left, right, 0.5);

        assert!(split.is_split());
        assert_eq!(split.node_count(), 3);
        assert_eq!(split.all_tabs().len(), 2);
    }

    #[test]
    fn test_find_tab() {
        let tab1 = make_tab(1, "A");
        let tab2 = make_tab(2, "B");
        let tab3 = make_tab(3, "C");

        let left = DockNode::single_tab(DockNodeId::new(0), tab1);
        let right = DockNode::leaf(DockNodeId::new(1), vec![tab2, tab3]);
        let root = DockNode::horizontal_split(DockNodeId::new(2), left, right, 0.5);

        assert!(root.find_tab(DockTabId::new(1)).is_some());
        assert!(root.find_tab(DockTabId::new(2)).is_some());
        assert!(root.find_tab(DockTabId::new(3)).is_some());
        assert!(root.find_tab(DockTabId::new(99)).is_none());
    }

    #[test]
    fn test_remove_tab() {
        let tab1 = make_tab(1, "A");
        let tab2 = make_tab(2, "B");
        let mut node = DockNode::leaf(DockNodeId::new(0), vec![tab1, tab2]);

        let removed = node.remove_tab(DockTabId::new(1));
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().title, "A");
        assert_eq!(node.tab_count(), 1);
    }

    #[test]
    fn test_collapse_empty() {
        let tab1 = make_tab(1, "A");
        let left = DockNode::single_tab(DockNodeId::new(0), tab1);
        let right = DockNode::empty_leaf(DockNodeId::new(1));
        let mut root = DockNode::horizontal_split(DockNodeId::new(2), left, right, 0.5);

        root.collapse_empty();
        assert!(root.is_leaf());
        assert_eq!(root.tab_count(), 1);
    }

    #[test]
    fn test_layout_computation() {
        let tab1 = make_tab(1, "Left");
        let tab2 = make_tab(2, "Right");
        let left = DockNode::single_tab(DockNodeId::new(0), tab1);
        let right = DockNode::single_tab(DockNodeId::new(1), tab2);
        let mut root = DockNode::horizontal_split(DockNodeId::new(2), left, right, 0.5);

        let viewport = Rect::new(Vec2::ZERO, Vec2::new(1000.0, 800.0));
        root.compute_layout(viewport);

        // After layout, both children should have computed rects
        match &root {
            DockNode::HorizontalSplit { left, right, .. } => {
                assert!(left.rect().width() > 0.0);
                assert!(right.rect().width() > 0.0);
                // Left + right + divider should roughly equal total width
                let total = left.rect().width() + right.rect().width() + DIVIDER_THICKNESS;
                assert!((total - 1000.0).abs() < 2.0);
            }
            _ => panic!("Expected horizontal split"),
        }
    }

    #[test]
    fn test_dock_state_operations() {
        let mut state = DockState::new();
        let node_id = state.root.id();

        // Add tabs
        let tab_id1 = state.next_tab_id();
        let tab_id2 = state.next_tab_id();
        let tab1 = DockTab::new(tab_id1, "First");
        let tab2 = DockTab::new(tab_id2, "Second");

        assert!(state.add_tab_to_node(node_id, tab1));
        assert!(state.add_tab_to_node(node_id, tab2));

        assert!(state.find_tab(tab_id1).is_some());
        assert!(state.find_tab(tab_id2).is_some());

        // Close a tab
        let closed = state.close_tab(tab_id1);
        assert!(closed.is_some());
        assert!(state.find_tab(tab_id1).is_none());
    }

    #[test]
    fn test_dock_layout_preset() {
        let layout = DockLayout::preset_default();
        assert!(!layout.root.is_empty());

        let all_tabs = layout.root.all_tabs();
        assert_eq!(all_tabs.len(), 4);
    }

    #[test]
    fn test_dock_layout_serialization() {
        let layout = DockLayout::preset_default();
        let json = layout.serialize();
        assert!(!json.is_empty());

        let restored = DockLayout::deserialize(&json);
        assert!(restored.is_some());
        let restored = restored.unwrap();
        assert_eq!(restored.root.all_tabs().len(), layout.root.all_tabs().len());
    }

    #[test]
    fn test_dock_target_zones() {
        let tab = make_tab(1, "Test");
        let mut node = DockNode::single_tab(DockNodeId::new(0), tab);

        let viewport = Rect::new(Vec2::ZERO, Vec2::new(400.0, 300.0));
        node.compute_layout(viewport);

        // Left edge
        let result = node.compute_dock_target(Vec2::new(10.0, 150.0));
        assert_eq!(result, Some((DockNodeId::new(0), DockTarget::Left)));

        // Right edge
        let result = node.compute_dock_target(Vec2::new(390.0, 150.0));
        assert_eq!(result, Some((DockNodeId::new(0), DockTarget::Right)));

        // Center
        let result = node.compute_dock_target(Vec2::new(200.0, 150.0));
        assert_eq!(result, Some((DockNodeId::new(0), DockTarget::Center)));
    }

    #[test]
    fn test_floating_window() {
        let tab = make_tab(1, "Float");
        let node = DockNode::single_tab(DockNodeId::new(0), tab);
        let win = FloatingWindow::new(0, "Window", Vec2::new(100.0, 100.0), Vec2::new(300.0, 200.0), node);

        assert!(win.hit_test(Vec2::new(200.0, 200.0)));
        assert!(!win.hit_test(Vec2::new(50.0, 50.0)));
        assert!(win.hit_test_title_bar(Vec2::new(200.0, 110.0)));
    }

    #[test]
    fn test_divider_hit_test() {
        let tab1 = make_tab(1, "Left");
        let tab2 = make_tab(2, "Right");
        let left = DockNode::single_tab(DockNodeId::new(0), tab1);
        let right = DockNode::single_tab(DockNodeId::new(1), tab2);
        let mut root = DockNode::horizontal_split(DockNodeId::new(2), left, right, 0.5);

        let viewport = Rect::new(Vec2::ZERO, Vec2::new(1000.0, 800.0));
        root.compute_layout(viewport);

        // Hit the divider near the center
        let split_x = 500.0;
        let result = root.hit_test_divider(Vec2::new(split_x, 400.0));
        assert!(result.is_some());
    }

    #[test]
    fn test_resize_edge_detection() {
        let pos = Vec2::new(100.0, 100.0);
        let size = Vec2::new(300.0, 200.0);

        // Near left edge
        let edge = ResizeEdge::from_point(Vec2::new(102.0, 200.0), pos, size);
        assert_eq!(edge, Some(ResizeEdge::Left));

        // Near top-left corner
        let edge = ResizeEdge::from_point(Vec2::new(102.0, 102.0), pos, size);
        assert_eq!(edge, Some(ResizeEdge::TopLeft));

        // Center (no resize)
        let edge = ResizeEdge::from_point(Vec2::new(250.0, 200.0), pos, size);
        assert_eq!(edge, None);
    }
}
