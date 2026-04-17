//! Slate Debug and Development Tools
//!
//! Provides comprehensive debugging and development tools for the Genovo UI
//! framework, including a real-time widget tree inspector, performance overlays,
//! automated UI testing, and hot style reloading.
//!
//! # Architecture
//!
//! ```text
//!  WidgetReflector ──> DebugOverlays ──> DebugRenderer
//!       │                   │
//!  SlateStats         UITestDriver
//!       │                   │
//!  PerformanceGraph   HotStyleReload
//! ```
//!
//! # Usage
//!
//! All debug tools are gated behind a `debug_enabled` flag and have zero cost
//! when disabled. They are designed to be used during development and removed
//! (or left disabled) in shipping builds.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

use glam::Vec2;

use crate::core::UIId;
use crate::render_commands::Color;

// ---------------------------------------------------------------------------
// WidgetInfo
// ---------------------------------------------------------------------------

/// Detailed information about a single widget, collected by the reflector.
#[derive(Debug, Clone)]
pub struct WidgetInfo {
    /// The widget's unique identifier.
    pub id: UIId,
    /// The widget type name (e.g., "Button", "TextInput", "Panel").
    pub type_name: String,
    /// Optional string identifier for debugging.
    pub debug_id: String,
    /// Position in screen space (top-left corner).
    pub position: Vec2,
    /// Size in logical pixels.
    pub size: Vec2,
    /// Whether the widget is currently visible.
    pub visible: bool,
    /// Whether the widget is enabled (interactive).
    pub enabled: bool,
    /// Parent widget ID, if any.
    pub parent: Option<UIId>,
    /// Child widget IDs.
    pub children: Vec<UIId>,
    /// Depth in the widget tree (root = 0).
    pub depth: u32,
    /// Z-order for rendering.
    pub z_order: i32,
    /// Clip rectangle applied to this widget.
    pub clip_rect: Option<[f32; 4]>,
    /// Whether this widget is hit-testable.
    pub hit_testable: bool,
    /// Last compute time in microseconds (layout).
    pub compute_time_us: u64,
    /// Last paint time in microseconds.
    pub paint_time_us: u64,
    /// Number of draw elements produced by this widget.
    pub draw_element_count: u32,
    /// Number of vertices produced.
    pub vertex_count: u32,
    /// Whether this widget is currently invalidated.
    pub invalidated: bool,
    /// Whether this widget is cached.
    pub cached: bool,
    /// Layout constraints applied to this widget.
    pub layout_constraints: Option<Vec2>,
    /// Desired size computed by this widget.
    pub desired_size: Option<Vec2>,
    /// Actual allocated size.
    pub allocated_size: Option<Vec2>,
    /// Margin.
    pub margin: [f32; 4],
    /// Padding.
    pub padding: [f32; 4],
    /// Content bounds (inside padding).
    pub content_bounds: Option<[f32; 4]>,
    /// Opacity.
    pub opacity: f32,
    /// Render transform (if any).
    pub has_render_transform: bool,
    /// Custom properties (for widget-specific debugging).
    pub custom_properties: HashMap<String, String>,
}

impl WidgetInfo {
    /// Creates a new widget info with minimal required fields.
    pub fn new(id: UIId, type_name: &str) -> Self {
        Self {
            id,
            type_name: type_name.to_string(),
            debug_id: String::new(),
            position: Vec2::ZERO,
            size: Vec2::ZERO,
            visible: true,
            enabled: true,
            parent: None,
            children: Vec::new(),
            depth: 0,
            z_order: 0,
            clip_rect: None,
            hit_testable: true,
            compute_time_us: 0,
            paint_time_us: 0,
            draw_element_count: 0,
            vertex_count: 0,
            invalidated: false,
            cached: false,
            layout_constraints: None,
            desired_size: None,
            allocated_size: None,
            margin: [0.0; 4],
            padding: [0.0; 4],
            content_bounds: None,
            opacity: 1.0,
            has_render_transform: false,
            custom_properties: HashMap::new(),
        }
    }

    /// Returns the bounding rectangle [x, y, width, height].
    pub fn bounds(&self) -> [f32; 4] {
        [self.position.x, self.position.y, self.size.x, self.size.y]
    }

    /// Returns the margin rectangle (outer bounds).
    pub fn margin_bounds(&self) -> [f32; 4] {
        [
            self.position.x - self.margin[0],
            self.position.y - self.margin[1],
            self.size.x + self.margin[0] + self.margin[2],
            self.size.y + self.margin[1] + self.margin[3],
        ]
    }

    /// Returns the content rectangle (inner bounds, inside padding).
    pub fn content_rect(&self) -> [f32; 4] {
        [
            self.position.x + self.padding[0],
            self.position.y + self.padding[1],
            self.size.x - self.padding[0] - self.padding[2],
            self.size.y - self.padding[1] - self.padding[3],
        ]
    }

    /// Returns true if the given screen point is inside this widget's bounds.
    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.position.x
            && point.x < self.position.x + self.size.x
            && point.y >= self.position.y
            && point.y < self.position.y + self.size.y
    }

    /// Returns the total time (compute + paint) in microseconds.
    pub fn total_time_us(&self) -> u64 {
        self.compute_time_us + self.paint_time_us
    }

    /// Returns the parent chain from this widget to the root.
    pub fn parent_chain(&self) -> Vec<UIId> {
        // This only holds the direct parent; the full chain is constructed
        // by the reflector using the tree.
        let mut chain = Vec::new();
        if let Some(parent) = self.parent {
            chain.push(parent);
        }
        chain
    }

    /// Returns a summary string for display.
    pub fn summary(&self) -> String {
        format!(
            "{} [{}] @ ({:.0},{:.0}) {}x{} | z:{} | vis:{} | {}us",
            self.type_name,
            if self.debug_id.is_empty() {
                self.id.to_string()
            } else {
                self.debug_id.clone()
            },
            self.position.x,
            self.position.y,
            self.size.x,
            self.size.y,
            self.z_order,
            self.visible,
            self.total_time_us(),
        )
    }
}

// ---------------------------------------------------------------------------
// PickResult
// ---------------------------------------------------------------------------

/// Result of a pick (hit-test) operation on the widget tree.
#[derive(Debug, Clone)]
pub struct PickResult {
    /// The widget at the top of the hit-test stack (front-most).
    pub hit_widget: Option<UIId>,
    /// All widgets under the cursor, from front to back.
    pub all_hits: Vec<UIId>,
    /// The screen position that was tested.
    pub screen_pos: Vec2,
    /// Whether the pick found any hit-testable widget.
    pub has_hit: bool,
}

impl PickResult {
    /// Creates an empty pick result (no hit).
    pub fn none(pos: Vec2) -> Self {
        Self {
            hit_widget: None,
            all_hits: Vec::new(),
            screen_pos: pos,
            has_hit: false,
        }
    }

    /// Creates a pick result with hits.
    pub fn new(pos: Vec2, hits: Vec<UIId>) -> Self {
        Self {
            hit_widget: hits.first().copied(),
            has_hit: !hits.is_empty(),
            all_hits: hits,
            screen_pos: pos,
        }
    }
}

// ---------------------------------------------------------------------------
// GeometryOverlay
// ---------------------------------------------------------------------------

/// Configuration for the geometry overlay visualization.
#[derive(Debug, Clone)]
pub struct GeometryOverlay {
    /// Whether the overlay is enabled.
    pub enabled: bool,
    /// Color for widget bounds rectangles.
    pub bounds_color: Color,
    /// Color for margin rectangles.
    pub margin_color: Color,
    /// Color for padding rectangles.
    pub padding_color: Color,
    /// Color for content rectangles.
    pub content_color: Color,
    /// Line thickness for the overlays.
    pub line_thickness: f32,
    /// Whether to show margin bounds.
    pub show_margins: bool,
    /// Whether to show padding bounds.
    pub show_padding: bool,
    /// Whether to show content bounds.
    pub show_content: bool,
    /// Whether to show widget type labels.
    pub show_labels: bool,
    /// Whether to show size labels.
    pub show_sizes: bool,
    /// Font size for labels.
    pub label_font_size: f32,
    /// Label background color.
    pub label_background: Color,
    /// Label text color.
    pub label_text_color: Color,
    /// Whether to only show overlay for the picked widget.
    pub picked_only: bool,
    /// Maximum depth to show (0 = all).
    pub max_depth: u32,
}

impl GeometryOverlay {
    /// Creates a new geometry overlay with default settings.
    pub fn new() -> Self {
        Self {
            enabled: false,
            bounds_color: Color::new(0.2, 0.6, 1.0, 0.5),
            margin_color: Color::new(1.0, 0.6, 0.2, 0.3),
            padding_color: Color::new(0.2, 1.0, 0.4, 0.3),
            content_color: Color::new(0.8, 0.2, 0.8, 0.3),
            line_thickness: 1.0,
            show_margins: true,
            show_padding: true,
            show_content: false,
            show_labels: true,
            show_sizes: true,
            label_font_size: 10.0,
            label_background: Color::new(0.0, 0.0, 0.0, 0.8),
            label_text_color: Color::WHITE,
            picked_only: false,
            max_depth: 0,
        }
    }

    /// Returns the draw commands for a single widget's geometry overlay.
    pub fn overlay_rects(&self, info: &WidgetInfo) -> Vec<OverlayRect> {
        let mut rects = Vec::new();

        if self.show_margins {
            let m = info.margin_bounds();
            rects.push(OverlayRect {
                rect: m,
                color: self.margin_color,
                filled: false,
                thickness: self.line_thickness,
                label: None,
            });
        }

        // Widget bounds.
        let b = info.bounds();
        rects.push(OverlayRect {
            rect: b,
            color: self.bounds_color,
            filled: false,
            thickness: self.line_thickness,
            label: if self.show_labels {
                let size_text = if self.show_sizes {
                    format!(" {:.0}x{:.0}", info.size.x, info.size.y)
                } else {
                    String::new()
                };
                Some(format!("{}{}", info.type_name, size_text))
            } else {
                None
            },
        });

        if self.show_padding {
            let p = info.content_rect();
            if p[2] > 0.0 && p[3] > 0.0 {
                rects.push(OverlayRect {
                    rect: p,
                    color: self.padding_color,
                    filled: false,
                    thickness: self.line_thickness,
                    label: None,
                });
            }
        }

        if self.show_content {
            if let Some(c) = info.content_bounds {
                rects.push(OverlayRect {
                    rect: c,
                    color: self.content_color,
                    filled: false,
                    thickness: self.line_thickness,
                    label: None,
                });
            }
        }

        rects
    }
}

impl Default for GeometryOverlay {
    fn default() -> Self {
        Self::new()
    }
}

/// A single overlay rectangle to be drawn.
#[derive(Debug, Clone)]
pub struct OverlayRect {
    /// Rectangle [x, y, width, height].
    pub rect: [f32; 4],
    /// Color of the overlay.
    pub color: Color,
    /// Whether to fill the rectangle (vs outline only).
    pub filled: bool,
    /// Line thickness for outlines.
    pub thickness: f32,
    /// Optional label text to draw at the top-left of the rect.
    pub label: Option<String>,
}

// ---------------------------------------------------------------------------
// WidgetReflector
// ---------------------------------------------------------------------------

/// Real-time widget tree inspector for development and debugging.
///
/// Provides a pick mode where hovering over the UI highlights and identifies
/// widgets, a tree view of the widget hierarchy, and per-widget performance
/// information with geometry overlays.
///
/// # Pick Mode
///
/// When pick mode is active, moving the mouse highlights the widget under the
/// cursor and displays its type, ID, position, size, and parent chain. Clicking
/// selects the widget and shows detailed information in the inspector panel.
///
/// # Geometry Overlay
///
/// Draws rectangle outlines for each widget showing margin, bounds, padding,
/// and content areas in different colors. This is invaluable for debugging
/// layout issues.
///
/// # Performance
///
/// Shows per-widget compute and paint times, allowing identification of
/// expensive widgets that may need optimization.
#[derive(Debug, Clone)]
pub struct WidgetReflector {
    /// Whether the reflector is enabled.
    pub enabled: bool,
    /// Whether pick mode is active (hover to identify).
    pub pick_mode: bool,
    /// The widget currently under the cursor (in pick mode).
    pub hovered_widget: Option<UIId>,
    /// The widget currently selected for inspection.
    pub selected_widget: Option<UIId>,
    /// All collected widget info for the current frame.
    pub widgets: HashMap<u32, WidgetInfo>,
    /// Geometry overlay configuration.
    pub geometry_overlay: GeometryOverlay,
    /// Search filter string.
    pub search_filter: String,
    /// Whether to search by type name.
    pub search_by_type: bool,
    /// Whether to search by debug ID.
    pub search_by_id: bool,
    /// Search results (widget IDs matching the filter).
    pub search_results: Vec<UIId>,
    /// History of selected widgets (for back navigation).
    pub selection_history: VecDeque<UIId>,
    /// Maximum history length.
    pub max_history: usize,
    /// Whether to auto-select hovered widget on click.
    pub auto_select_on_click: bool,
    /// Whether to show the tree view panel.
    pub show_tree_view: bool,
    /// Whether to show the property inspector panel.
    pub show_properties: bool,
    /// Whether to show performance info.
    pub show_performance: bool,
    /// Total widget count.
    pub total_widget_count: u32,
    /// Visible widget count.
    pub visible_widget_count: u32,
    /// Highlight color for picked widgets.
    pub highlight_color: Color,
    /// Highlight color for selected widgets.
    pub selected_color: Color,
    /// Tooltip text for the hovered widget.
    pub tooltip_text: String,
    /// Whether to show the tooltip near the cursor.
    pub show_tooltip: bool,
    /// Root widget IDs (top-level widgets with no parent).
    pub root_widgets: Vec<UIId>,
    /// Expanded node IDs in the tree view.
    pub expanded_nodes: HashSet<u32>,
    /// Frame number for staleness detection.
    pub current_frame: u64,
}

impl WidgetReflector {
    /// Creates a new widget reflector.
    pub fn new() -> Self {
        Self {
            enabled: false,
            pick_mode: false,
            hovered_widget: None,
            selected_widget: None,
            widgets: HashMap::with_capacity(512),
            geometry_overlay: GeometryOverlay::new(),
            search_filter: String::new(),
            search_by_type: true,
            search_by_id: true,
            search_results: Vec::new(),
            selection_history: VecDeque::with_capacity(32),
            max_history: 32,
            auto_select_on_click: true,
            show_tree_view: true,
            show_properties: true,
            show_performance: false,
            total_widget_count: 0,
            visible_widget_count: 0,
            highlight_color: Color::new(0.0, 0.8, 1.0, 0.3),
            selected_color: Color::new(1.0, 0.5, 0.0, 0.4),
            tooltip_text: String::new(),
            show_tooltip: true,
            root_widgets: Vec::new(),
            expanded_nodes: HashSet::new(),
            current_frame: 0,
        }
    }

    /// Begins a new frame of widget collection.
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
        self.widgets.clear();
        self.root_widgets.clear();
        self.total_widget_count = 0;
        self.visible_widget_count = 0;
    }

    /// Registers a widget for the current frame.
    pub fn register_widget(&mut self, info: WidgetInfo) {
        self.total_widget_count += 1;
        if info.visible {
            self.visible_widget_count += 1;
        }
        if info.parent.is_none() {
            self.root_widgets.push(info.id);
        }
        self.widgets.insert(info.id.index, info);
    }

    /// Performs a pick (hit-test) at the given screen position.
    pub fn pick(&self, screen_pos: Vec2) -> PickResult {
        let mut hits: Vec<(UIId, i32, u32)> = self
            .widgets
            .values()
            .filter(|w| w.visible && w.hit_testable && w.contains_point(screen_pos))
            .map(|w| (w.id, w.z_order, w.depth))
            .collect();

        // Sort by z-order (descending) then depth (descending) to get front-most first.
        hits.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        let hit_ids: Vec<UIId> = hits.into_iter().map(|(id, _, _)| id).collect();

        PickResult::new(screen_pos, hit_ids)
    }

    /// Updates the reflector for mouse hover (pick mode).
    pub fn update_hover(&mut self, screen_pos: Vec2) {
        if !self.enabled || !self.pick_mode {
            return;
        }

        let result = self.pick(screen_pos);
        self.hovered_widget = result.hit_widget;

        if let Some(id) = self.hovered_widget {
            if let Some(info) = self.widgets.get(&id.index) {
                self.tooltip_text = info.summary();
            }
        } else {
            self.tooltip_text.clear();
        }
    }

    /// Handles a click in pick mode (selects the hovered widget).
    pub fn on_click(&mut self, screen_pos: Vec2) {
        if !self.enabled || !self.pick_mode || !self.auto_select_on_click {
            return;
        }

        let result = self.pick(screen_pos);
        if let Some(id) = result.hit_widget {
            self.select_widget(id);
        }
    }

    /// Selects a widget for detailed inspection.
    pub fn select_widget(&mut self, widget_id: UIId) {
        // Push to history.
        if let Some(prev) = self.selected_widget {
            self.selection_history.push_back(prev);
            if self.selection_history.len() > self.max_history {
                self.selection_history.pop_front();
            }
        }
        self.selected_widget = Some(widget_id);

        // Auto-expand parent chain in tree view.
        self.expand_to_widget(widget_id);
    }

    /// Expands the tree view to show the given widget.
    fn expand_to_widget(&mut self, widget_id: UIId) {
        let mut current = Some(widget_id);
        while let Some(id) = current {
            self.expanded_nodes.insert(id.index);
            current = self.widgets.get(&id.index).and_then(|w| w.parent);
        }
    }

    /// Goes back to the previously selected widget.
    pub fn go_back(&mut self) {
        if let Some(prev) = self.selection_history.pop_back() {
            self.selected_widget = Some(prev);
        }
    }

    /// Returns the selected widget's info.
    pub fn selected_info(&self) -> Option<&WidgetInfo> {
        self.selected_widget
            .and_then(|id| self.widgets.get(&id.index))
    }

    /// Returns the parent chain for a widget (from widget to root).
    pub fn parent_chain(&self, widget_id: UIId) -> Vec<UIId> {
        let mut chain = Vec::new();
        let mut current = Some(widget_id);
        while let Some(id) = current {
            chain.push(id);
            current = self.widgets.get(&id.index).and_then(|w| w.parent);
        }
        chain
    }

    /// Returns the children of a widget.
    pub fn children(&self, widget_id: UIId) -> Vec<UIId> {
        self.widgets
            .get(&widget_id.index)
            .map(|w| w.children.clone())
            .unwrap_or_default()
    }

    /// Applies the search filter and updates results.
    pub fn apply_search(&mut self) {
        self.search_results.clear();
        if self.search_filter.is_empty() {
            return;
        }

        let filter_lower = self.search_filter.to_lowercase();
        for info in self.widgets.values() {
            let mut matches = false;
            if self.search_by_type && info.type_name.to_lowercase().contains(&filter_lower) {
                matches = true;
            }
            if self.search_by_id && info.debug_id.to_lowercase().contains(&filter_lower) {
                matches = true;
            }
            if matches {
                self.search_results.push(info.id);
            }
        }
    }

    /// Returns the overlay draw commands for the current state.
    pub fn get_overlays(&self) -> Vec<OverlayRect> {
        let mut overlays = Vec::new();

        if !self.enabled {
            return overlays;
        }

        // Geometry overlay for all widgets.
        if self.geometry_overlay.enabled && !self.geometry_overlay.picked_only {
            for info in self.widgets.values() {
                if !info.visible {
                    continue;
                }
                if self.geometry_overlay.max_depth > 0
                    && info.depth > self.geometry_overlay.max_depth
                {
                    continue;
                }
                overlays.extend(self.geometry_overlay.overlay_rects(info));
            }
        }

        // Hovered widget highlight.
        if let Some(id) = self.hovered_widget {
            if let Some(info) = self.widgets.get(&id.index) {
                overlays.push(OverlayRect {
                    rect: info.bounds(),
                    color: self.highlight_color,
                    filled: true,
                    thickness: 2.0,
                    label: Some(info.type_name.clone()),
                });

                if self.geometry_overlay.enabled && self.geometry_overlay.picked_only {
                    overlays.extend(self.geometry_overlay.overlay_rects(info));
                }
            }
        }

        // Selected widget highlight.
        if let Some(id) = self.selected_widget {
            if let Some(info) = self.widgets.get(&id.index) {
                overlays.push(OverlayRect {
                    rect: info.bounds(),
                    color: self.selected_color,
                    filled: true,
                    thickness: 3.0,
                    label: Some(format!("SELECTED: {}", info.type_name)),
                });
            }
        }

        // Search result highlights.
        for id in &self.search_results {
            if let Some(info) = self.widgets.get(&id.index) {
                overlays.push(OverlayRect {
                    rect: info.bounds(),
                    color: Color::new(0.0, 1.0, 0.0, 0.2),
                    filled: true,
                    thickness: 1.0,
                    label: None,
                });
            }
        }

        overlays
    }

    /// Returns the tree view data starting from root widgets.
    pub fn tree_data(&self) -> Vec<TreeViewNode> {
        let mut nodes = Vec::new();
        for root_id in &self.root_widgets {
            self.build_tree_recursive(*root_id, &mut nodes, 0);
        }
        nodes
    }

    /// Recursively builds tree view data.
    fn build_tree_recursive(
        &self,
        widget_id: UIId,
        nodes: &mut Vec<TreeViewNode>,
        depth: u32,
    ) {
        let info = match self.widgets.get(&widget_id.index) {
            Some(i) => i,
            None => return,
        };

        let expanded = self.expanded_nodes.contains(&widget_id.index);
        let selected = self.selected_widget == Some(widget_id);
        let hovered = self.hovered_widget == Some(widget_id);

        nodes.push(TreeViewNode {
            widget_id,
            type_name: info.type_name.clone(),
            debug_id: info.debug_id.clone(),
            depth,
            expanded,
            selected,
            hovered,
            has_children: !info.children.is_empty(),
            visible: info.visible,
            child_count: info.children.len() as u32,
        });

        if expanded {
            for child_id in &info.children {
                self.build_tree_recursive(*child_id, nodes, depth + 1);
            }
        }
    }

    /// Toggles a node's expanded state in the tree view.
    pub fn toggle_expanded(&mut self, widget_id: UIId) {
        if self.expanded_nodes.contains(&widget_id.index) {
            self.expanded_nodes.remove(&widget_id.index);
        } else {
            self.expanded_nodes.insert(widget_id.index);
        }
    }

    /// Expands all nodes in the tree view.
    pub fn expand_all(&mut self) {
        for id in self.widgets.keys() {
            self.expanded_nodes.insert(*id);
        }
    }

    /// Collapses all nodes in the tree view.
    pub fn collapse_all(&mut self) {
        self.expanded_nodes.clear();
    }
}

impl Default for WidgetReflector {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the tree view representation.
#[derive(Debug, Clone)]
pub struct TreeViewNode {
    /// Widget ID.
    pub widget_id: UIId,
    /// Widget type name.
    pub type_name: String,
    /// Debug ID string.
    pub debug_id: String,
    /// Depth in the tree.
    pub depth: u32,
    /// Whether this node is expanded.
    pub expanded: bool,
    /// Whether this node is selected.
    pub selected: bool,
    /// Whether this node is hovered.
    pub hovered: bool,
    /// Whether this node has children.
    pub has_children: bool,
    /// Whether this widget is visible.
    pub visible: bool,
    /// Number of children.
    pub child_count: u32,
}

impl TreeViewNode {
    /// Returns the indentation string for tree display.
    pub fn indent_string(&self) -> String {
        "  ".repeat(self.depth as usize)
    }

    /// Returns a display string for the tree line.
    pub fn display_line(&self) -> String {
        let arrow = if self.has_children {
            if self.expanded {
                "v "
            } else {
                "> "
            }
        } else {
            "  "
        };
        let vis = if self.visible { "" } else { " [hidden]" };
        let id_str = if self.debug_id.is_empty() {
            String::new()
        } else {
            format!(" #{}", self.debug_id)
        };
        format!(
            "{}{}{}{}{} ({})",
            self.indent_string(),
            arrow,
            self.type_name,
            id_str,
            vis,
            self.child_count,
        )
    }
}

// ---------------------------------------------------------------------------
// FrameStats
// ---------------------------------------------------------------------------

/// Detailed statistics for a single frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Frame number.
    pub frame_number: u64,
    /// Total tick time in microseconds.
    pub tick_time_us: u64,
    /// Layout time in microseconds.
    pub layout_time_us: u64,
    /// Paint time in microseconds.
    pub paint_time_us: u64,
    /// Event dispatch time in microseconds.
    pub event_time_us: u64,
    /// Total widget count.
    pub widget_count: u32,
    /// Visible widget count.
    pub visible_count: u32,
    /// Invalidated widget count.
    pub invalidated_count: u32,
    /// Cached widget count (layout cache hits).
    pub cached_count: u32,
    /// Draw calls (batched).
    pub batched_draw_calls: u32,
    /// Draw calls (unbatched / raw).
    pub unbatched_draw_calls: u32,
    /// GPU state changes.
    pub state_changes: u32,
    /// Total elements.
    pub element_count: u32,
    /// Total vertices.
    pub vertex_count: u32,
    /// Total indices.
    pub index_count: u32,
    /// Widget memory estimate in bytes.
    pub widget_memory: u64,
    /// Cache memory in bytes.
    pub cache_memory: u64,
    /// Dirty screen fraction.
    pub dirty_fraction: f32,
}

impl FrameStats {
    /// Returns the frame time in milliseconds.
    pub fn frame_time_ms(&self) -> f32 {
        self.tick_time_us as f32 / 1000.0
    }

    /// Returns the estimated FPS based on tick time.
    pub fn estimated_fps(&self) -> f32 {
        if self.tick_time_us == 0 {
            0.0
        } else {
            1_000_000.0 / self.tick_time_us as f32
        }
    }

    /// Returns the batch efficiency.
    pub fn batch_efficiency(&self) -> f32 {
        if self.unbatched_draw_calls == 0 {
            1.0
        } else {
            1.0 - (self.batched_draw_calls as f32 / self.unbatched_draw_calls as f32)
        }
    }
}

// ---------------------------------------------------------------------------
// SlateStats
// ---------------------------------------------------------------------------

/// Collects and maintains rolling frame-by-frame statistics for the UI system.
///
/// Maintains a history of the last 120 frames for sparkline visualization
/// and trend analysis. Provides aggregate statistics (min, max, average)
/// over the history window.
#[derive(Debug, Clone)]
pub struct SlateStats {
    /// Rolling history of frame statistics.
    pub history: VecDeque<FrameStats>,
    /// Maximum history length.
    pub max_history: usize,
    /// Whether statistics collection is enabled.
    pub enabled: bool,
    /// Current frame number.
    pub current_frame: u64,
    /// Peak frame time seen.
    pub peak_frame_time_us: u64,
    /// Peak widget count seen.
    pub peak_widget_count: u32,
    /// Peak draw call count seen.
    pub peak_draw_calls: u32,
    /// Peak vertex count seen.
    pub peak_vertex_count: u32,
    /// Total frames recorded.
    pub total_frames: u64,
    /// Total time recorded in microseconds.
    pub total_time_us: u64,
    /// Whether to pause recording.
    pub paused: bool,
    /// Budget target in microseconds (e.g., 16666 for 60fps).
    pub frame_budget_us: u64,
    /// Number of frames that exceeded the budget.
    pub over_budget_count: u64,
}

impl SlateStats {
    /// Creates a new statistics collector.
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(120),
            max_history: 120,
            enabled: true,
            current_frame: 0,
            peak_frame_time_us: 0,
            peak_widget_count: 0,
            peak_draw_calls: 0,
            peak_vertex_count: 0,
            total_frames: 0,
            total_time_us: 0,
            paused: false,
            frame_budget_us: 16666,
            over_budget_count: 0,
        }
    }

    /// Records a new frame's statistics.
    pub fn record(&mut self, stats: FrameStats) {
        if !self.enabled || self.paused {
            return;
        }

        self.current_frame = stats.frame_number;
        self.total_frames += 1;
        self.total_time_us += stats.tick_time_us;

        // Update peaks.
        if stats.tick_time_us > self.peak_frame_time_us {
            self.peak_frame_time_us = stats.tick_time_us;
        }
        if stats.widget_count > self.peak_widget_count {
            self.peak_widget_count = stats.widget_count;
        }
        if stats.batched_draw_calls > self.peak_draw_calls {
            self.peak_draw_calls = stats.batched_draw_calls;
        }
        if stats.vertex_count > self.peak_vertex_count {
            self.peak_vertex_count = stats.vertex_count;
        }

        if stats.tick_time_us > self.frame_budget_us {
            self.over_budget_count += 1;
        }

        self.history.push_back(stats);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Returns the average frame time over the history window.
    pub fn avg_frame_time_us(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.history.iter().map(|s| s.tick_time_us).sum();
        sum as f64 / self.history.len() as f64
    }

    /// Returns the average layout time over the history window.
    pub fn avg_layout_time_us(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.history.iter().map(|s| s.layout_time_us).sum();
        sum as f64 / self.history.len() as f64
    }

    /// Returns the average paint time over the history window.
    pub fn avg_paint_time_us(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.history.iter().map(|s| s.paint_time_us).sum();
        sum as f64 / self.history.len() as f64
    }

    /// Returns the maximum frame time in the history window.
    pub fn max_frame_time_us(&self) -> u64 {
        self.history.iter().map(|s| s.tick_time_us).max().unwrap_or(0)
    }

    /// Returns the minimum frame time in the history window.
    pub fn min_frame_time_us(&self) -> u64 {
        self.history.iter().map(|s| s.tick_time_us).min().unwrap_or(0)
    }

    /// Returns the average widget count.
    pub fn avg_widget_count(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.history.iter().map(|s| s.widget_count as u64).sum();
        sum as f64 / self.history.len() as f64
    }

    /// Returns the frame-time values as a slice for sparkline rendering.
    pub fn frame_time_series(&self) -> Vec<f32> {
        self.history
            .iter()
            .map(|s| s.tick_time_us as f32 / 1000.0)
            .collect()
    }

    /// Returns the fraction of frames that exceeded the budget.
    pub fn over_budget_fraction(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            self.over_budget_count as f64 / self.total_frames as f64
        }
    }

    /// Returns a summary string.
    pub fn summary(&self) -> String {
        format!(
            "Frame: {:.2}ms avg ({:.2}ms peak) | Budget: {:.1}% over | \
             Widgets: {:.0} avg ({} peak) | Draws: {} peak | Verts: {} peak",
            self.avg_frame_time_us() / 1000.0,
            self.peak_frame_time_us as f64 / 1000.0,
            self.over_budget_fraction() * 100.0,
            self.avg_widget_count(),
            self.peak_widget_count,
            self.peak_draw_calls,
            self.peak_vertex_count,
        )
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.history.clear();
        self.peak_frame_time_us = 0;
        self.peak_widget_count = 0;
        self.peak_draw_calls = 0;
        self.peak_vertex_count = 0;
        self.total_frames = 0;
        self.total_time_us = 0;
        self.over_budget_count = 0;
    }
}

impl Default for SlateStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DebugOverlayKind
// ---------------------------------------------------------------------------

/// Types of debug overlays that can be toggled on/off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DebugOverlayKind {
    /// Show dirty rects as colored flashing rectangles.
    DirtyRects,
    /// Show clip rects as colored outlines.
    ClipRects,
    /// Show the focus path (currently focused widget chain).
    FocusPath,
    /// Show event propagation path.
    EventPath,
    /// Show layout bounds (margin, padding, content).
    LayoutBounds,
    /// Show invalidation reasons on widgets.
    InvalidationReasons,
    /// Show widget type labels on each widget.
    WidgetLabels,
    /// Show FPS counter.
    FpsCounter,
    /// Show draw call count.
    DrawCallCount,
    /// Show memory usage.
    MemoryUsage,
    /// Show batch statistics.
    BatchStats,
}

// ---------------------------------------------------------------------------
// DebugOverlays
// ---------------------------------------------------------------------------

/// Manages debug overlay visualization for the UI system.
///
/// Each overlay type can be independently toggled. Overlays are drawn on top
/// of the regular UI as a separate pass, so they don't affect layout or
/// interaction.
#[derive(Debug, Clone)]
pub struct DebugOverlays {
    /// Which overlays are currently enabled.
    pub enabled_overlays: HashSet<DebugOverlayKind>,
    /// Whether all overlays are globally enabled.
    pub global_enabled: bool,
    /// Dirty rect overlay: flash duration in frames.
    pub dirty_rect_flash_frames: u32,
    /// Dirty rect overlay: accumulated rects with frame counter.
    pub dirty_rect_flashes: Vec<(DirtyRectFlash, u32)>,
    /// Clip rect overlay color.
    pub clip_rect_color: Color,
    /// Focus path overlay color.
    pub focus_path_color: Color,
    /// Event path overlay: last event propagation path.
    pub event_path: Vec<UIId>,
    /// Event path overlay color.
    pub event_path_color: Color,
    /// Layout bounds overlay: current data.
    pub layout_bounds: Vec<LayoutBoundsOverlay>,
    /// Invalidation reason overlay: current data.
    pub invalidation_data: Vec<InvalidationOverlay>,
    /// FPS counter position.
    pub fps_position: Vec2,
    /// FPS counter font size.
    pub fps_font_size: f32,
    /// Current FPS value.
    pub current_fps: f32,
    /// Current draw call count.
    pub current_draw_calls: u32,
    /// Current memory usage.
    pub current_memory_mb: f32,
    /// Overlay opacity.
    pub overlay_opacity: f32,
}

impl DebugOverlays {
    /// Creates a new debug overlays manager.
    pub fn new() -> Self {
        Self {
            enabled_overlays: HashSet::new(),
            global_enabled: false,
            dirty_rect_flash_frames: 10,
            dirty_rect_flashes: Vec::new(),
            clip_rect_color: Color::new(0.0, 1.0, 1.0, 0.3),
            focus_path_color: Color::new(1.0, 0.8, 0.0, 0.5),
            event_path: Vec::new(),
            event_path_color: Color::new(0.0, 1.0, 0.0, 0.4),
            layout_bounds: Vec::new(),
            invalidation_data: Vec::new(),
            fps_position: Vec2::new(10.0, 10.0),
            fps_font_size: 14.0,
            current_fps: 0.0,
            current_draw_calls: 0,
            current_memory_mb: 0.0,
            overlay_opacity: 0.7,
        }
    }

    /// Enables a specific overlay.
    pub fn enable(&mut self, kind: DebugOverlayKind) {
        self.enabled_overlays.insert(kind);
    }

    /// Disables a specific overlay.
    pub fn disable(&mut self, kind: DebugOverlayKind) {
        self.enabled_overlays.remove(&kind);
    }

    /// Toggles a specific overlay.
    pub fn toggle(&mut self, kind: DebugOverlayKind) {
        if self.enabled_overlays.contains(&kind) {
            self.enabled_overlays.remove(&kind);
        } else {
            self.enabled_overlays.insert(kind);
        }
    }

    /// Returns true if a specific overlay is enabled.
    pub fn is_enabled(&self, kind: DebugOverlayKind) -> bool {
        self.global_enabled && self.enabled_overlays.contains(&kind)
    }

    /// Adds a dirty rect flash.
    pub fn add_dirty_rect_flash(&mut self, rect: [f32; 4], color: Color) {
        self.dirty_rect_flashes.push((
            DirtyRectFlash { rect, color },
            self.dirty_rect_flash_frames,
        ));
    }

    /// Updates flash timers and removes expired flashes.
    pub fn update_flashes(&mut self) {
        self.dirty_rect_flashes.retain_mut(|(_, frames_left)| {
            *frames_left = frames_left.saturating_sub(1);
            *frames_left > 0
        });
    }

    /// Sets the event propagation path for visualization.
    pub fn set_event_path(&mut self, path: Vec<UIId>) {
        self.event_path = path;
    }

    /// Adds layout bounds data for a widget.
    pub fn add_layout_bounds(&mut self, data: LayoutBoundsOverlay) {
        self.layout_bounds.push(data);
    }

    /// Adds invalidation data for a widget.
    pub fn add_invalidation(&mut self, data: InvalidationOverlay) {
        self.invalidation_data.push(data);
    }

    /// Clears per-frame overlay data.
    pub fn begin_frame(&mut self) {
        self.layout_bounds.clear();
        self.invalidation_data.clear();
        self.update_flashes();
    }

    /// Enables all overlays.
    pub fn enable_all(&mut self) {
        self.global_enabled = true;
        self.enabled_overlays.insert(DebugOverlayKind::DirtyRects);
        self.enabled_overlays.insert(DebugOverlayKind::ClipRects);
        self.enabled_overlays.insert(DebugOverlayKind::FocusPath);
        self.enabled_overlays.insert(DebugOverlayKind::LayoutBounds);
        self.enabled_overlays.insert(DebugOverlayKind::FpsCounter);
    }

    /// Disables all overlays.
    pub fn disable_all(&mut self) {
        self.enabled_overlays.clear();
        self.global_enabled = false;
    }

    /// Returns all overlay draw commands for the current frame.
    pub fn get_overlay_rects(&self) -> Vec<OverlayRect> {
        let mut overlays = Vec::new();

        if !self.global_enabled {
            return overlays;
        }

        // Dirty rect flashes.
        if self.is_enabled(DebugOverlayKind::DirtyRects) {
            for (flash, frames_left) in &self.dirty_rect_flashes {
                let alpha = *frames_left as f32 / self.dirty_rect_flash_frames as f32;
                overlays.push(OverlayRect {
                    rect: flash.rect,
                    color: flash.color.with_alpha(alpha * self.overlay_opacity),
                    filled: true,
                    thickness: 2.0,
                    label: None,
                });
            }
        }

        // Layout bounds.
        if self.is_enabled(DebugOverlayKind::LayoutBounds) {
            for lb in &self.layout_bounds {
                overlays.push(OverlayRect {
                    rect: lb.margin_rect,
                    color: Color::new(1.0, 0.5, 0.0, 0.2 * self.overlay_opacity),
                    filled: false,
                    thickness: 1.0,
                    label: None,
                });
                overlays.push(OverlayRect {
                    rect: lb.padding_rect,
                    color: Color::new(0.0, 0.5, 1.0, 0.2 * self.overlay_opacity),
                    filled: false,
                    thickness: 1.0,
                    label: None,
                });
                overlays.push(OverlayRect {
                    rect: lb.content_rect,
                    color: Color::new(0.0, 1.0, 0.0, 0.2 * self.overlay_opacity),
                    filled: false,
                    thickness: 1.0,
                    label: None,
                });
            }
        }

        overlays
    }
}

impl Default for DebugOverlays {
    fn default() -> Self {
        Self::new()
    }
}

/// A dirty rect flash for visualization.
#[derive(Debug, Clone)]
pub struct DirtyRectFlash {
    /// Rectangle [x, y, width, height].
    pub rect: [f32; 4],
    /// Flash color.
    pub color: Color,
}

/// Layout bounds overlay data for a single widget.
#[derive(Debug, Clone)]
pub struct LayoutBoundsOverlay {
    /// Widget ID.
    pub widget_id: UIId,
    /// Margin rectangle.
    pub margin_rect: [f32; 4],
    /// Padding rectangle.
    pub padding_rect: [f32; 4],
    /// Content rectangle.
    pub content_rect: [f32; 4],
}

/// Invalidation overlay data for a single widget.
#[derive(Debug, Clone)]
pub struct InvalidationOverlay {
    /// Widget ID.
    pub widget_id: UIId,
    /// Widget bounds.
    pub bounds: [f32; 4],
    /// Invalidation reason text.
    pub reason: String,
    /// Color based on reason.
    pub color: Color,
}

// ---------------------------------------------------------------------------
// UITestAction
// ---------------------------------------------------------------------------

/// An action that can be performed by the UI test driver.
#[derive(Debug, Clone)]
pub enum UITestAction {
    /// Click on a widget.
    Click { widget_id: UIId },
    /// Double-click on a widget.
    DoubleClick { widget_id: UIId },
    /// Right-click on a widget.
    RightClick { widget_id: UIId },
    /// Type text into a widget.
    TypeText { widget_id: UIId, text: String },
    /// Press a key.
    PressKey { key_code: u32, modifiers: KeyModifiers },
    /// Drag from one point to another.
    Drag { from: Vec2, to: Vec2, duration_ms: u32 },
    /// Scroll a widget.
    Scroll { widget_id: UIId, delta: Vec2 },
    /// Wait for a condition.
    Wait { condition: WaitCondition, timeout_ms: u32 },
    /// Assert a condition is true.
    Assert { assertion: TestAssertion },
    /// Move the mouse to a position.
    MoveMouse { position: Vec2 },
    /// Hover over a widget.
    Hover { widget_id: UIId, duration_ms: u32 },
    /// Focus a widget.
    Focus { widget_id: UIId },
    /// Set a widget's value.
    SetValue { widget_id: UIId, value: String },
}

/// Key modifiers for test actions.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyModifiers {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

/// Conditions the test driver can wait for.
#[derive(Debug, Clone)]
pub enum WaitCondition {
    /// Wait for a widget to become visible.
    WidgetVisible(UIId),
    /// Wait for a widget to have specific text.
    WidgetText(UIId, String),
    /// Wait for a widget to be enabled.
    WidgetEnabled(UIId),
    /// Wait for no animations to be running.
    NoAnimations,
    /// Wait for a specific number of frames.
    Frames(u32),
    /// Wait for the UI to be idle (sleeping).
    Idle,
}

/// Assertions for test verification.
#[derive(Debug, Clone)]
pub enum TestAssertion {
    /// Assert a widget is visible.
    Visible(UIId),
    /// Assert a widget is hidden.
    Hidden(UIId),
    /// Assert a widget has specific text.
    Text(UIId, String),
    /// Assert a widget has specific text (contains).
    TextContains(UIId, String),
    /// Assert a widget is enabled.
    Enabled(UIId),
    /// Assert a widget is disabled.
    Disabled(UIId),
    /// Assert a widget is focused.
    Focused(UIId),
    /// Assert a widget is selected.
    Selected(UIId),
    /// Assert a widget is checked.
    Checked(UIId),
    /// Assert a widget exists.
    Exists(UIId),
    /// Assert a widget does not exist.
    NotExists(UIId),
    /// Assert widget count matches.
    WidgetCount(u32),
    /// Custom assertion with a message.
    Custom(String, bool),
}

/// Result of a test action execution.
#[derive(Debug, Clone)]
pub struct TestActionResult {
    /// The action that was executed.
    pub action_index: usize,
    /// Whether the action succeeded.
    pub success: bool,
    /// Error message if the action failed.
    pub error: Option<String>,
    /// Execution time in microseconds.
    pub duration_us: u64,
    /// Frame number when the action was executed.
    pub frame: u64,
}

// ---------------------------------------------------------------------------
// UITestDriver
// ---------------------------------------------------------------------------

/// Automated UI testing driver.
///
/// Allows programmatic interaction with the UI for automated testing:
/// - Click, type, drag, scroll on widgets by ID.
/// - Find widgets by predicate.
/// - Assert widget visibility, text, state.
/// - Wait for conditions (widget visible, animations done, etc.).
/// - Record and replay interaction sequences.
///
/// # Usage
///
/// ```ignore
/// let mut driver = UITestDriver::new();
/// driver.click(button_id);
/// driver.type_text(input_id, "Hello");
/// driver.assert_text(label_id, "Hello");
/// driver.wait_for_no_animations(1000);
/// ```
#[derive(Debug, Clone)]
pub struct UITestDriver {
    /// Whether the test driver is active.
    pub active: bool,
    /// Queue of actions to execute.
    pub action_queue: VecDeque<UITestAction>,
    /// Results of executed actions.
    pub results: Vec<TestActionResult>,
    /// Currently executing action index.
    pub current_action: usize,
    /// Whether to stop on first failure.
    pub stop_on_failure: bool,
    /// Widget finder results (from last find operation).
    pub found_widgets: Vec<UIId>,
    /// Recording: list of recorded actions.
    pub recorded_actions: Vec<UITestAction>,
    /// Whether recording mode is active.
    pub recording: bool,
    /// Whether replay mode is active.
    pub replaying: bool,
    /// Replay speed multiplier (1.0 = normal).
    pub replay_speed: f32,
    /// Current frame number.
    pub current_frame: u64,
    /// Wait state: remaining frames to wait.
    pub wait_frames: u32,
    /// Wait state: condition being waited for.
    pub wait_condition: Option<WaitCondition>,
    /// Total tests run.
    pub total_tests: u32,
    /// Total tests passed.
    pub passed_tests: u32,
    /// Total tests failed.
    pub failed_tests: u32,
    /// Test timeout in frames.
    pub timeout_frames: u32,
    /// Frames elapsed for current wait.
    pub wait_elapsed: u32,
}

impl UITestDriver {
    /// Creates a new test driver.
    pub fn new() -> Self {
        Self {
            active: false,
            action_queue: VecDeque::new(),
            results: Vec::new(),
            current_action: 0,
            stop_on_failure: true,
            found_widgets: Vec::new(),
            recorded_actions: Vec::new(),
            recording: false,
            replaying: false,
            replay_speed: 1.0,
            current_frame: 0,
            wait_frames: 0,
            wait_condition: None,
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            timeout_frames: 600,
            wait_elapsed: 0,
        }
    }

    /// Queues a click action on a widget.
    pub fn click(&mut self, widget_id: UIId) {
        self.action_queue
            .push_back(UITestAction::Click { widget_id });
    }

    /// Queues a double-click action on a widget.
    pub fn double_click(&mut self, widget_id: UIId) {
        self.action_queue
            .push_back(UITestAction::DoubleClick { widget_id });
    }

    /// Queues a type-text action on a widget.
    pub fn type_text(&mut self, widget_id: UIId, text: &str) {
        self.action_queue
            .push_back(UITestAction::TypeText {
                widget_id,
                text: text.to_string(),
            });
    }

    /// Queues a key press action.
    pub fn press_key(&mut self, key_code: u32) {
        self.action_queue
            .push_back(UITestAction::PressKey {
                key_code,
                modifiers: KeyModifiers::default(),
            });
    }

    /// Queues a key press with modifiers.
    pub fn press_key_with_modifiers(&mut self, key_code: u32, modifiers: KeyModifiers) {
        self.action_queue
            .push_back(UITestAction::PressKey {
                key_code,
                modifiers,
            });
    }

    /// Queues a drag action.
    pub fn drag(&mut self, from: Vec2, to: Vec2, duration_ms: u32) {
        self.action_queue
            .push_back(UITestAction::Drag { from, to, duration_ms });
    }

    /// Queues a scroll action on a widget.
    pub fn scroll(&mut self, widget_id: UIId, delta: Vec2) {
        self.action_queue
            .push_back(UITestAction::Scroll { widget_id, delta });
    }

    /// Queues a wait-for-visible action.
    pub fn wait_for_visible(&mut self, widget_id: UIId, timeout_ms: u32) {
        self.action_queue
            .push_back(UITestAction::Wait {
                condition: WaitCondition::WidgetVisible(widget_id),
                timeout_ms,
            });
    }

    /// Queues a wait-for-no-animations action.
    pub fn wait_for_no_animations(&mut self, timeout_ms: u32) {
        self.action_queue
            .push_back(UITestAction::Wait {
                condition: WaitCondition::NoAnimations,
                timeout_ms,
            });
    }

    /// Queues an assert-visible action.
    pub fn assert_visible(&mut self, widget_id: UIId) {
        self.action_queue
            .push_back(UITestAction::Assert {
                assertion: TestAssertion::Visible(widget_id),
            });
    }

    /// Queues an assert-text action.
    pub fn assert_text(&mut self, widget_id: UIId, expected: &str) {
        self.action_queue
            .push_back(UITestAction::Assert {
                assertion: TestAssertion::Text(widget_id, expected.to_string()),
            });
    }

    /// Queues an assert-enabled action.
    pub fn assert_enabled(&mut self, widget_id: UIId) {
        self.action_queue
            .push_back(UITestAction::Assert {
                assertion: TestAssertion::Enabled(widget_id),
            });
    }

    /// Queues an assert-focused action.
    pub fn assert_focused(&mut self, widget_id: UIId) {
        self.action_queue
            .push_back(UITestAction::Assert {
                assertion: TestAssertion::Focused(widget_id),
            });
    }

    /// Starts recording user interactions.
    pub fn start_recording(&mut self) {
        self.recording = true;
        self.recorded_actions.clear();
    }

    /// Stops recording and returns the recorded actions.
    pub fn stop_recording(&mut self) -> Vec<UITestAction> {
        self.recording = false;
        self.recorded_actions.clone()
    }

    /// Replays a sequence of recorded actions.
    pub fn replay(&mut self, actions: Vec<UITestAction>) {
        self.action_queue.clear();
        for action in actions {
            self.action_queue.push_back(action);
        }
        self.replaying = true;
        self.active = true;
    }

    /// Records an action (called when recording is active).
    pub fn record_action(&mut self, action: UITestAction) {
        if self.recording {
            self.recorded_actions.push(action);
        }
    }

    /// Executes the next action in the queue, if ready.
    pub fn tick(&mut self) -> Option<UITestAction> {
        self.current_frame += 1;

        // Handle waiting.
        if self.wait_frames > 0 {
            self.wait_frames -= 1;
            return None;
        }

        if !self.active || self.action_queue.is_empty() {
            return None;
        }

        let action = self.action_queue.pop_front();
        if let Some(ref a) = action {
            self.current_action += 1;
        }

        action
    }

    /// Records the result of an action execution.
    pub fn record_result(&mut self, success: bool, error: Option<String>) {
        self.total_tests += 1;
        if success {
            self.passed_tests += 1;
        } else {
            self.failed_tests += 1;
            if self.stop_on_failure {
                self.active = false;
            }
        }

        self.results.push(TestActionResult {
            action_index: self.current_action,
            success,
            error,
            duration_us: 0,
            frame: self.current_frame,
        });
    }

    /// Returns a test summary.
    pub fn summary(&self) -> String {
        format!(
            "Tests: {} total, {} passed, {} failed ({:.0}% pass rate)",
            self.total_tests,
            self.passed_tests,
            self.failed_tests,
            if self.total_tests > 0 {
                self.passed_tests as f64 / self.total_tests as f64 * 100.0
            } else {
                0.0
            },
        )
    }

    /// Resets all test state.
    pub fn reset(&mut self) {
        self.action_queue.clear();
        self.results.clear();
        self.current_action = 0;
        self.total_tests = 0;
        self.passed_tests = 0;
        self.failed_tests = 0;
        self.wait_frames = 0;
        self.wait_condition = None;
    }

    /// Returns true if all queued actions have been executed.
    pub fn is_complete(&self) -> bool {
        self.action_queue.is_empty() && self.wait_frames == 0
    }

    /// Returns true if all tests passed.
    pub fn all_passed(&self) -> bool {
        self.failed_tests == 0 && self.total_tests > 0
    }
}

impl Default for UITestDriver {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WatchedStyleFile
// ---------------------------------------------------------------------------

/// A style file being watched for hot reload.
#[derive(Debug, Clone)]
pub struct WatchedStyleFile {
    /// File path.
    pub path: String,
    /// Last known modification timestamp (as a duration since epoch).
    pub last_modified: u64,
    /// Content hash for change detection.
    pub content_hash: u64,
    /// Number of times this file has been reloaded.
    pub reload_count: u32,
    /// Whether the file has pending changes.
    pub dirty: bool,
    /// List of properties that changed in the last reload.
    pub changed_properties: Vec<String>,
    /// Whether loading this file failed.
    pub load_error: Option<String>,
}

impl WatchedStyleFile {
    /// Creates a new watched file entry.
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            last_modified: 0,
            content_hash: 0,
            reload_count: 0,
            dirty: false,
            changed_properties: Vec::new(),
            load_error: None,
        }
    }

    /// Computes a simple hash of a string content for change detection.
    pub fn compute_hash(content: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in content.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }

    /// Checks if the file content has changed.
    pub fn check_changed(&mut self, content: &str) -> bool {
        let new_hash = Self::compute_hash(content);
        if new_hash != self.content_hash {
            self.content_hash = new_hash;
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Marks the file as successfully reloaded.
    pub fn mark_reloaded(&mut self, changed_props: Vec<String>) {
        self.reload_count += 1;
        self.dirty = false;
        self.changed_properties = changed_props;
        self.load_error = None;
    }

    /// Marks the file as having a load error.
    pub fn mark_error(&mut self, error: &str) {
        self.load_error = Some(error.to_string());
        self.dirty = false;
    }
}

// ---------------------------------------------------------------------------
// HotStyleReload
// ---------------------------------------------------------------------------

/// Watches style definition files and reloads them without restarting.
///
/// Monitors one or more style files for changes (based on content hash),
/// and when changes are detected, parses and applies the new styles. Shows
/// which properties changed and provides live preview of style changes.
///
/// # Usage
///
/// ```ignore
/// let mut hot_reload = HotStyleReload::new();
/// hot_reload.watch("styles/editor.json");
/// hot_reload.watch("styles/theme.json");
///
/// // Each frame:
/// if hot_reload.check_for_changes() {
///     // Apply the new styles.
///     let changes = hot_reload.get_changes();
/// }
/// ```
#[derive(Debug, Clone)]
pub struct HotStyleReload {
    /// Files being watched.
    pub watched_files: Vec<WatchedStyleFile>,
    /// Whether hot reload is enabled.
    pub enabled: bool,
    /// Interval between checks in frames.
    pub check_interval: u32,
    /// Frame counter for interval checks.
    pub frame_counter: u32,
    /// Whether changes are pending application.
    pub has_pending_changes: bool,
    /// Total reloads performed.
    pub total_reloads: u32,
    /// Total reload errors.
    pub total_errors: u32,
    /// Whether live preview is active.
    pub live_preview: bool,
    /// Notification text for the last reload.
    pub notification: Option<String>,
    /// Notification timeout in frames.
    pub notification_timeout: u32,
    /// Notification frame counter.
    pub notification_frames: u32,
}

impl HotStyleReload {
    /// Creates a new hot style reload manager.
    pub fn new() -> Self {
        Self {
            watched_files: Vec::new(),
            enabled: true,
            check_interval: 30,
            frame_counter: 0,
            has_pending_changes: false,
            total_reloads: 0,
            total_errors: 0,
            live_preview: true,
            notification: None,
            notification_timeout: 120,
            notification_frames: 0,
        }
    }

    /// Adds a file to the watch list.
    pub fn watch(&mut self, path: &str) {
        if !self.watched_files.iter().any(|f| f.path == path) {
            self.watched_files.push(WatchedStyleFile::new(path));
        }
    }

    /// Removes a file from the watch list.
    pub fn unwatch(&mut self, path: &str) {
        self.watched_files.retain(|f| f.path != path);
    }

    /// Checks if it's time to poll for changes.
    pub fn should_check(&mut self) -> bool {
        if !self.enabled {
            return false;
        }
        self.frame_counter += 1;
        if self.frame_counter >= self.check_interval {
            self.frame_counter = 0;
            true
        } else {
            false
        }
    }

    /// Reports that a file has been reloaded successfully.
    pub fn report_reload(&mut self, path: &str, changed_props: Vec<String>) {
        self.total_reloads += 1;
        let prop_count = changed_props.len();
        if let Some(file) = self.watched_files.iter_mut().find(|f| f.path == path) {
            file.mark_reloaded(changed_props);
        }
        self.notification = Some(format!(
            "Style reloaded: {} ({} properties changed)",
            path, prop_count
        ));
        self.notification_frames = 0;
    }

    /// Reports a reload error.
    pub fn report_error(&mut self, path: &str, error: &str) {
        self.total_errors += 1;
        if let Some(file) = self.watched_files.iter_mut().find(|f| f.path == path) {
            file.mark_error(error);
        }
        self.notification = Some(format!("Style reload error: {}: {}", path, error));
        self.notification_frames = 0;
    }

    /// Updates the notification timer.
    pub fn update_notification(&mut self) {
        if self.notification.is_some() {
            self.notification_frames += 1;
            if self.notification_frames >= self.notification_timeout {
                self.notification = None;
            }
        }
    }

    /// Returns the currently pending notification text.
    pub fn get_notification(&self) -> Option<&str> {
        self.notification.as_deref()
    }

    /// Returns a summary of watched files.
    pub fn summary(&self) -> String {
        format!(
            "Hot reload: {} files | {} reloads | {} errors",
            self.watched_files.len(),
            self.total_reloads,
            self.total_errors,
        )
    }

    /// Returns the list of recently changed properties.
    pub fn recent_changes(&self) -> Vec<(String, Vec<String>)> {
        self.watched_files
            .iter()
            .filter(|f| !f.changed_properties.is_empty())
            .map(|f| (f.path.clone(), f.changed_properties.clone()))
            .collect()
    }
}

impl Default for HotStyleReload {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_info_contains_point() {
        let mut info = WidgetInfo::new(UIId::new(1, 0), "Button");
        info.position = Vec2::new(100.0, 50.0);
        info.size = Vec2::new(80.0, 30.0);

        assert!(info.contains_point(Vec2::new(120.0, 60.0)));
        assert!(!info.contains_point(Vec2::new(50.0, 60.0)));
        assert!(!info.contains_point(Vec2::new(200.0, 60.0)));
    }

    #[test]
    fn test_pick_result() {
        let ids = vec![UIId::new(3, 0), UIId::new(1, 0)];
        let result = PickResult::new(Vec2::new(100.0, 50.0), ids);
        assert!(result.has_hit);
        assert_eq!(result.hit_widget, Some(UIId::new(3, 0)));
        assert_eq!(result.all_hits.len(), 2);
    }

    #[test]
    fn test_widget_reflector_pick() {
        let mut reflector = WidgetReflector::new();
        reflector.enabled = true;

        let mut info_a = WidgetInfo::new(UIId::new(1, 0), "Button");
        info_a.position = Vec2::new(10.0, 10.0);
        info_a.size = Vec2::new(100.0, 30.0);
        info_a.z_order = 1;

        let mut info_b = WidgetInfo::new(UIId::new(2, 0), "Panel");
        info_b.position = Vec2::new(0.0, 0.0);
        info_b.size = Vec2::new(200.0, 200.0);
        info_b.z_order = 0;

        reflector.register_widget(info_a);
        reflector.register_widget(info_b);

        let result = reflector.pick(Vec2::new(50.0, 20.0));
        // Button is z_order 1, Panel is z_order 0 => Button on top.
        assert_eq!(result.hit_widget, Some(UIId::new(1, 0)));
    }

    #[test]
    fn test_slate_stats_averages() {
        let mut stats = SlateStats::new();
        for i in 0..10 {
            stats.record(FrameStats {
                frame_number: i,
                tick_time_us: 16000,
                layout_time_us: 3000,
                paint_time_us: 8000,
                ..Default::default()
            });
        }
        assert!((stats.avg_frame_time_us() - 16000.0).abs() < 1.0);
        assert!((stats.avg_layout_time_us() - 3000.0).abs() < 1.0);
        assert!((stats.avg_paint_time_us() - 8000.0).abs() < 1.0);
    }

    #[test]
    fn test_debug_overlays_toggle() {
        let mut overlays = DebugOverlays::new();
        overlays.global_enabled = true;
        overlays.toggle(DebugOverlayKind::FpsCounter);
        assert!(overlays.is_enabled(DebugOverlayKind::FpsCounter));
        overlays.toggle(DebugOverlayKind::FpsCounter);
        assert!(!overlays.is_enabled(DebugOverlayKind::FpsCounter));
    }

    #[test]
    fn test_ui_test_driver() {
        let mut driver = UITestDriver::new();
        driver.active = true;
        driver.click(UIId::new(1, 0));
        driver.assert_visible(UIId::new(1, 0));
        assert_eq!(driver.action_queue.len(), 2);

        let action = driver.tick().unwrap();
        assert!(matches!(action, UITestAction::Click { .. }));
        assert_eq!(driver.action_queue.len(), 1);
    }

    #[test]
    fn test_hot_style_reload() {
        let mut reload = HotStyleReload::new();
        reload.watch("styles/editor.json");
        assert_eq!(reload.watched_files.len(), 1);

        reload.report_reload("styles/editor.json", vec!["color".to_string()]);
        assert_eq!(reload.total_reloads, 1);
        assert!(reload.notification.is_some());
    }

    #[test]
    fn test_watched_style_file_hash() {
        let mut file = WatchedStyleFile::new("test.json");
        assert!(file.check_changed("hello world"));
        assert!(!file.check_changed("hello world"));
        assert!(file.check_changed("hello world!"));
    }

    #[test]
    fn test_tree_view_node_display() {
        let node = TreeViewNode {
            widget_id: UIId::new(1, 0),
            type_name: "Button".to_string(),
            debug_id: "save_btn".to_string(),
            depth: 2,
            expanded: false,
            selected: false,
            hovered: false,
            has_children: true,
            visible: true,
            child_count: 3,
        };
        let line = node.display_line();
        assert!(line.contains("Button"));
        assert!(line.contains("save_btn"));
        assert!(line.contains("(3)"));
    }
}
