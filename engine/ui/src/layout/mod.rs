//! Layout engine for the UI system.
//!
//! Implements flex layout, grid layout, stack layout, and scroll layout
//! algorithms that compute screen-space rectangles for every node in the
//! [`UITree`].
//!
//! # Design
//!
//! Layout is a two-pass process:
//!
//! 1. **Measure pass** — walk the tree bottom-up to determine each node's
//!    desired (intrinsic) size given the constraints imposed by its parent.
//! 2. **Arrange pass** — walk the tree top-down to assign each node its final
//!    position and size, distributing remaining space according to flex
//!    grow/shrink or grid tracks.

use glam::Vec2;
use serde::{Deserialize, Serialize};

use genovo_core::Rect;

use crate::core::{UIId, UITree};

// ---------------------------------------------------------------------------
// Size / Constraint
// ---------------------------------------------------------------------------

/// Specifies a dimension (width or height) for a UI node.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Size {
    /// A fixed pixel value.
    Fixed(f32),
    /// A percentage of the parent's available size (0.0 – 1.0).
    Percent(f32),
    /// Size to content.
    Auto,
    /// Fill all remaining space.
    Fill,
    /// Size to content but do not exceed available space.
    FitContent,
}

impl Default for Size {
    fn default() -> Self {
        Self::Auto
    }
}

impl Size {
    /// Resolve a size value given the available space and an intrinsic content
    /// size.
    pub fn resolve(&self, available: f32, intrinsic: f32) -> f32 {
        match self {
            Self::Fixed(v) => *v,
            Self::Percent(p) => available * p,
            Self::Auto => intrinsic,
            Self::Fill => available,
            Self::FitContent => intrinsic.min(available),
        }
    }
}

/// Min/max constraints on a single axis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisConstraint {
    pub min: f32,
    pub max: f32,
}

impl AxisConstraint {
    pub const UNBOUNDED: Self = Self {
        min: 0.0,
        max: f32::INFINITY,
    };

    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Clamp a value to [min, max].
    pub fn clamp(&self, value: f32) -> f32 {
        value.clamp(self.min, self.max)
    }
}

impl Default for AxisConstraint {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

/// 2-D constraint box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    pub width: AxisConstraint,
    pub height: AxisConstraint,
}

impl Constraint {
    pub const UNBOUNDED: Self = Self {
        width: AxisConstraint::UNBOUNDED,
        height: AxisConstraint::UNBOUNDED,
    };

    pub fn new(
        min_width: f32,
        max_width: f32,
        min_height: f32,
        max_height: f32,
    ) -> Self {
        Self {
            width: AxisConstraint::new(min_width, max_width),
            height: AxisConstraint::new(min_height, max_height),
        }
    }

    pub fn tight(size: Vec2) -> Self {
        Self {
            width: AxisConstraint::new(size.x, size.x),
            height: AxisConstraint::new(size.y, size.y),
        }
    }

    pub fn loose(max: Vec2) -> Self {
        Self {
            width: AxisConstraint::new(0.0, max.x),
            height: AxisConstraint::new(0.0, max.y),
        }
    }

    /// Clamp a size to the constraint box.
    pub fn clamp(&self, size: Vec2) -> Vec2 {
        Vec2::new(self.width.clamp(size.x), self.height.clamp(size.y))
    }

    pub fn max_size(&self) -> Vec2 {
        Vec2::new(self.width.max, self.height.max)
    }

    pub fn min_size(&self) -> Vec2 {
        Vec2::new(self.width.min, self.height.min)
    }
}

impl Default for Constraint {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

// ---------------------------------------------------------------------------
// LayoutDirection / LayoutAlign
// ---------------------------------------------------------------------------

/// Primary axis direction for flex/stack layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayoutDirection {
    Horizontal,
    Vertical,
}

impl Default for LayoutDirection {
    fn default() -> Self {
        Self::Vertical
    }
}

/// Alignment on the main or cross axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayoutAlign {
    Start,
    Center,
    End,
    Stretch,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

impl Default for LayoutAlign {
    fn default() -> Self {
        Self::Start
    }
}

// ---------------------------------------------------------------------------
// FlexLayout
// ---------------------------------------------------------------------------

/// Flexbox-like layout parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlexLayout {
    /// Primary axis direction.
    pub direction: LayoutDirection,
    /// Whether children wrap to the next line when they exceed available space.
    pub wrap: bool,
    /// Alignment of children along the main axis.
    pub justify_content: LayoutAlign,
    /// Alignment of children along the cross axis.
    pub align_items: LayoutAlign,
    /// Alignment of wrapped lines along the cross axis.
    pub align_content: LayoutAlign,
    /// Gap between items along the main axis.
    pub gap: f32,
    /// Gap between lines when wrapping (cross-axis gap).
    pub cross_gap: f32,
}

impl Default for FlexLayout {
    fn default() -> Self {
        Self {
            direction: LayoutDirection::Vertical,
            wrap: false,
            justify_content: LayoutAlign::Start,
            align_items: LayoutAlign::Start,
            align_content: LayoutAlign::Start,
            gap: 0.0,
            cross_gap: 0.0,
        }
    }
}

impl FlexLayout {
    pub fn row() -> Self {
        Self {
            direction: LayoutDirection::Horizontal,
            ..Default::default()
        }
    }

    pub fn column() -> Self {
        Self {
            direction: LayoutDirection::Vertical,
            ..Default::default()
        }
    }

    pub fn with_gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }

    pub fn with_justify(mut self, align: LayoutAlign) -> Self {
        self.justify_content = align;
        self
    }

    pub fn with_align_items(mut self, align: LayoutAlign) -> Self {
        self.align_items = align;
        self
    }

    pub fn with_wrap(mut self, wrap: bool) -> Self {
        self.wrap = wrap;
        self
    }
}

// ---------------------------------------------------------------------------
// GridLayout
// ---------------------------------------------------------------------------

/// A single track (column or row) size definition.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrackSize {
    /// Fixed pixel size.
    Fixed(f32),
    /// Fractional unit (similar to CSS `fr`).
    Fractional(f32),
    /// Size to content.
    Auto,
    /// Percentage of available space.
    Percent(f32),
}

impl Default for TrackSize {
    fn default() -> Self {
        Self::Auto
    }
}

/// CSS-grid-like layout parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridLayout {
    /// Column track definitions.
    pub columns: Vec<TrackSize>,
    /// Row track definitions.
    pub rows: Vec<TrackSize>,
    /// Gap between columns.
    pub column_gap: f32,
    /// Gap between rows.
    pub row_gap: f32,
    /// Default alignment of items within their cell (main axis).
    pub justify_items: LayoutAlign,
    /// Default alignment of items within their cell (cross axis).
    pub align_items: LayoutAlign,
}

impl Default for GridLayout {
    fn default() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            column_gap: 0.0,
            row_gap: 0.0,
            justify_items: LayoutAlign::Stretch,
            align_items: LayoutAlign::Stretch,
        }
    }
}

impl GridLayout {
    /// Create a grid with `n` equal-fraction columns.
    pub fn uniform_columns(n: usize) -> Self {
        Self {
            columns: vec![TrackSize::Fractional(1.0); n],
            ..Default::default()
        }
    }

    /// Create a grid with `n` equal-fraction rows.
    pub fn uniform_rows(n: usize) -> Self {
        Self {
            rows: vec![TrackSize::Fractional(1.0); n],
            ..Default::default()
        }
    }

    pub fn with_gap(mut self, column_gap: f32, row_gap: f32) -> Self {
        self.column_gap = column_gap;
        self.row_gap = row_gap;
        self
    }
}

// ---------------------------------------------------------------------------
// StackLayout
// ---------------------------------------------------------------------------

/// Absolute-position layout with z-ordering (like a CSS `position: absolute`
/// container).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StackLayout {
    /// Whether children are automatically sorted by z-order for rendering.
    pub auto_sort: bool,
}

// ---------------------------------------------------------------------------
// ScrollLayout
// ---------------------------------------------------------------------------

/// A scrollable content region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollLayout {
    /// Current scroll offset (positive = content scrolled up/left).
    pub offset: Vec2,
    /// Whether horizontal scrolling is enabled.
    pub horizontal: bool,
    /// Whether vertical scrolling is enabled.
    pub vertical: bool,
    /// The measured content size (set by layout).
    pub content_size: Vec2,
    /// The visible viewport size (set by layout).
    pub viewport_size: Vec2,
    /// Scroll deceleration factor for momentum scrolling.
    pub deceleration: f32,
    /// Current scroll velocity (for momentum).
    pub velocity: Vec2,
}

impl Default for ScrollLayout {
    fn default() -> Self {
        Self {
            offset: Vec2::ZERO,
            horizontal: false,
            vertical: true,
            content_size: Vec2::ZERO,
            viewport_size: Vec2::ZERO,
            deceleration: 0.95,
            velocity: Vec2::ZERO,
        }
    }
}

impl ScrollLayout {
    /// Maximum scroll offset given content and viewport sizes.
    pub fn max_offset(&self) -> Vec2 {
        Vec2::new(
            (self.content_size.x - self.viewport_size.x).max(0.0),
            (self.content_size.y - self.viewport_size.y).max(0.0),
        )
    }

    /// Clamp the current offset to valid bounds.
    pub fn clamp_offset(&mut self) {
        let max = self.max_offset();
        self.offset.x = self.offset.x.clamp(0.0, max.x);
        self.offset.y = self.offset.y.clamp(0.0, max.y);
    }

    /// Apply a scroll delta and return the consumed portion.
    pub fn scroll_by(&mut self, delta: Vec2) -> Vec2 {
        let old = self.offset;
        if self.horizontal {
            self.offset.x += delta.x;
        }
        if self.vertical {
            self.offset.y += delta.y;
        }
        self.clamp_offset();
        self.offset - old
    }

    /// Scroll fraction (0..1) for each axis.
    pub fn scroll_fraction(&self) -> Vec2 {
        let max = self.max_offset();
        Vec2::new(
            if max.x > 0.0 {
                self.offset.x / max.x
            } else {
                0.0
            },
            if max.y > 0.0 {
                self.offset.y / max.y
            } else {
                0.0
            },
        )
    }

    /// Update momentum scrolling.
    pub fn update_momentum(&mut self, dt: f32) {
        if self.velocity.length_squared() > 0.01 {
            let delta = self.velocity * dt;
            self.scroll_by(delta);
            self.velocity *= self.deceleration;
        } else {
            self.velocity = Vec2::ZERO;
        }
    }

    /// Whether the scrollbar thumb is needed (content exceeds viewport).
    pub fn needs_scrollbar_h(&self) -> bool {
        self.horizontal && self.content_size.x > self.viewport_size.x
    }

    pub fn needs_scrollbar_v(&self) -> bool {
        self.vertical && self.content_size.y > self.viewport_size.y
    }
}

// ---------------------------------------------------------------------------
// LayoutKind — which layout algorithm to use
// ---------------------------------------------------------------------------

/// The layout strategy applied to a node's children.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutKind {
    /// No automatic layout; children are positioned by their anchors/offsets.
    None,
    /// Flexbox-like layout.
    Flex(FlexLayout),
    /// CSS-grid-like layout.
    Grid(GridLayout),
    /// Absolute positioning (like a pile of elements).
    Stack(StackLayout),
    /// Scrollable container.
    Scroll(ScrollLayout),
}

impl Default for LayoutKind {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// LayoutEngine
// ---------------------------------------------------------------------------

/// Data computed during the measure pass for a single node.
#[derive(Debug, Clone, Copy)]
struct MeasuredSize {
    width: f32,
    height: f32,
}

/// The layout engine. Stateless — all persistent state lives on the
/// [`UITree`] nodes and the layout descriptors stored alongside them.
pub struct LayoutEngine {
    /// Per-node layout kinds, keyed by UIId index.
    layout_kinds: Vec<Option<LayoutKind>>,
    /// Per-node measured sizes, keyed by UIId index.
    measured_sizes: Vec<Option<MeasuredSize>>,
    /// Scratch buffer for child sizing during flex layout.
    flex_scratch: Vec<FlexChildInfo>,
}

/// Internal scratch data per flex child.
#[derive(Debug, Clone)]
struct FlexChildInfo {
    id: UIId,
    #[allow(dead_code)]
    main_size: f32,
    cross_size: f32,
    flex_grow: f32,
    flex_shrink: f32,
    flex_basis: f32,
    margin_before: f32,
    margin_after: f32,
    margin_cross_before: f32,
    margin_cross_after: f32,
    #[allow(dead_code)]
    frozen: bool,
    final_main: f32,
    final_cross: f32,
    final_offset: f32,
    final_cross_offset: f32,
}

impl LayoutEngine {
    pub fn new() -> Self {
        Self {
            layout_kinds: Vec::new(),
            measured_sizes: Vec::new(),
            flex_scratch: Vec::new(),
        }
    }

    /// Set the layout kind for a node.
    pub fn set_layout(&mut self, id: UIId, kind: LayoutKind) {
        let idx = id.index as usize;
        if idx >= self.layout_kinds.len() {
            self.layout_kinds.resize_with(idx + 1, || None);
        }
        self.layout_kinds[idx] = Some(kind);
    }

    /// Get the layout kind for a node.
    pub fn get_layout(&self, id: UIId) -> Option<&LayoutKind> {
        self.layout_kinds
            .get(id.index as usize)
            .and_then(|o| o.as_ref())
    }

    /// Get mutable layout kind.
    pub fn get_layout_mut(&mut self, id: UIId) -> Option<&mut LayoutKind> {
        self.layout_kinds
            .get_mut(id.index as usize)
            .and_then(|o| o.as_mut())
    }

    /// Compute layout for the entire tree starting at the root.
    pub fn compute_layout(&mut self, tree: &mut UITree, available_size: Vec2) {
        self.measured_sizes
            .resize(tree.len() + 64, None);
        let root = tree.root();

        // Set the root's rect to the available area.
        if let Some(root_node) = tree.get_mut(root) {
            root_node.computed_rect = Rect::new(Vec2::ZERO, available_size);
        }

        // Measure pass (bottom-up).
        self.measure_recursive(tree, root, available_size);

        // Arrange pass (top-down).
        self.arrange_recursive(tree, root);
    }

    // -- Measure pass --------------------------------------------------------

    fn measure_recursive(&mut self, tree: &mut UITree, id: UIId, available: Vec2) {
        let children = tree.children(id);

        // Measure children first (bottom-up).
        for &child_id in &children {
            let child_available = self.child_available_size(tree, id, available);
            self.measure_recursive(tree, child_id, child_available);
        }

        // Now measure this node.
        let measured = self.measure_node(tree, id, available, &children);
        let idx = id.index as usize;
        if idx >= self.measured_sizes.len() {
            self.measured_sizes.resize(idx + 1, None);
        }
        self.measured_sizes[idx] = Some(measured);
    }

    fn child_available_size(&self, tree: &UITree, parent_id: UIId, parent_available: Vec2) -> Vec2 {
        if let Some(parent) = tree.get(parent_id) {
            let pad_h = parent.layout_params.padding.horizontal();
            let pad_v = parent.layout_params.padding.vertical();
            Vec2::new(
                (parent_available.x - pad_h).max(0.0),
                (parent_available.y - pad_v).max(0.0),
            )
        } else {
            parent_available
        }
    }

    fn measure_node(
        &self,
        tree: &UITree,
        id: UIId,
        available: Vec2,
        children: &[UIId],
    ) -> MeasuredSize {
        let node = match tree.get(id) {
            Some(n) => n,
            None => return MeasuredSize { width: 0.0, height: 0.0 },
        };

        let params = &node.layout_params;
        let pad_h = params.padding.horizontal();
        let pad_v = params.padding.vertical();
        let inner_available = Vec2::new(
            (available.x - params.margin.horizontal() - pad_h).max(0.0),
            (available.y - params.margin.vertical() - pad_v).max(0.0),
        );

        // Content size from children.
        let content_size = self.measure_children_content(id, children, inner_available);

        // Resolve this node's size.
        let width = params
            .width
            .resolve(inner_available.x, content_size.x + pad_h);
        let height = params
            .height
            .resolve(inner_available.y, content_size.y + pad_v);

        // Apply min/max constraints.
        let width = apply_minmax(width, params.min_width, params.max_width);
        let height = apply_minmax(height, params.min_height, params.max_height);

        MeasuredSize { width, height }
    }

    fn measure_children_content(
        &self,
        parent_id: UIId,
        children: &[UIId],
        _available: Vec2,
    ) -> Vec2 {
        let layout_kind = self
            .layout_kinds
            .get(parent_id.index as usize)
            .and_then(|o| o.as_ref());

        match layout_kind {
            Some(LayoutKind::Flex(flex)) => {
                self.measure_flex_content(flex, children)
            }
            Some(LayoutKind::Grid(grid)) => {
                self.measure_grid_content(grid, children)
            }
            _ => {
                // For None/Stack/Scroll, content size = bounding box of
                // children's measured sizes.
                self.measure_stacked_content(children)
            }
        }
    }

    fn measure_flex_content(&self, flex: &FlexLayout, children: &[UIId]) -> Vec2 {
        let mut main_total = 0.0_f32;
        let mut cross_max = 0.0_f32;
        let gap_count = if children.len() > 1 {
            (children.len() - 1) as f32
        } else {
            0.0
        };

        for &child_id in children {
            let ms = self.get_measured(child_id);
            match flex.direction {
                LayoutDirection::Horizontal => {
                    main_total += ms.width;
                    cross_max = cross_max.max(ms.height);
                }
                LayoutDirection::Vertical => {
                    main_total += ms.height;
                    cross_max = cross_max.max(ms.width);
                }
            }
        }
        main_total += gap_count * flex.gap;

        match flex.direction {
            LayoutDirection::Horizontal => Vec2::new(main_total, cross_max),
            LayoutDirection::Vertical => Vec2::new(cross_max, main_total),
        }
    }

    fn measure_grid_content(&self, grid: &GridLayout, children: &[UIId]) -> Vec2 {
        let num_cols = grid.columns.len().max(1);
        let num_rows = if grid.rows.is_empty() {
            // Auto-generate rows.
            (children.len() + num_cols - 1) / num_cols
        } else {
            grid.rows.len()
        };

        // Compute rough content sizes per track.
        let mut col_widths = vec![0.0_f32; num_cols];
        let mut row_heights = vec![0.0_f32; num_rows];

        for (i, &child_id) in children.iter().enumerate() {
            let col = i % num_cols;
            let row = i / num_cols;
            if row >= num_rows {
                break;
            }
            let ms = self.get_measured(child_id);
            col_widths[col] = col_widths[col].max(ms.width);
            row_heights[row] = row_heights[row].max(ms.height);
        }

        let total_w: f32 = col_widths.iter().sum::<f32>()
            + grid.column_gap * (num_cols.saturating_sub(1)) as f32;
        let total_h: f32 = row_heights.iter().sum::<f32>()
            + grid.row_gap * (num_rows.saturating_sub(1)) as f32;
        Vec2::new(total_w, total_h)
    }

    fn measure_stacked_content(&self, children: &[UIId]) -> Vec2 {
        let mut max_w = 0.0_f32;
        let mut max_h = 0.0_f32;
        for &child_id in children {
            let ms = self.get_measured(child_id);
            max_w = max_w.max(ms.width);
            max_h = max_h.max(ms.height);
        }
        Vec2::new(max_w, max_h)
    }

    fn get_measured(&self, id: UIId) -> MeasuredSize {
        self.measured_sizes
            .get(id.index as usize)
            .copied()
            .flatten()
            .unwrap_or(MeasuredSize {
                width: 0.0,
                height: 0.0,
            })
    }

    // -- Arrange pass --------------------------------------------------------

    fn arrange_recursive(&mut self, tree: &mut UITree, id: UIId) {
        let children = tree.children(id);
        if children.is_empty() {
            return;
        }

        let parent_rect = tree
            .get(id)
            .map(|n| n.computed_rect)
            .unwrap_or(Rect::new(Vec2::ZERO, Vec2::ZERO));
        let parent_padding = tree
            .get(id)
            .map(|n| n.layout_params.padding)
            .unwrap_or_default();

        let content_origin = Vec2::new(
            parent_rect.min.x + parent_padding.left,
            parent_rect.min.y + parent_padding.top,
        );
        let content_size = Vec2::new(
            (parent_rect.width() - parent_padding.horizontal()).max(0.0),
            (parent_rect.height() - parent_padding.vertical()).max(0.0),
        );

        let layout_kind = self
            .layout_kinds
            .get(id.index as usize)
            .and_then(|o| o.clone());

        match layout_kind.as_ref() {
            Some(LayoutKind::Flex(flex)) => {
                self.arrange_flex(tree, flex, &children, content_origin, content_size);
            }
            Some(LayoutKind::Grid(grid)) => {
                self.arrange_grid(tree, grid, &children, content_origin, content_size);
            }
            Some(LayoutKind::Stack(_)) => {
                self.arrange_stack(tree, &children, content_origin, content_size);
            }
            Some(LayoutKind::Scroll(scroll)) => {
                let scroll_offset = scroll.offset;
                self.arrange_scroll(
                    tree,
                    &children,
                    content_origin,
                    content_size,
                    scroll_offset,
                );
                // Update scroll layout content size.
                let content_measured = self.measure_stacked_content(&children);
                if let Some(LayoutKind::Scroll(s)) = self.get_layout_mut(id) {
                    s.viewport_size = content_size;
                    s.content_size = Vec2::new(
                        content_measured.x.max(content_size.x),
                        content_measured.y.max(content_size.y),
                    );
                }
            }
            _ => {
                // No layout — use anchor-based positioning.
                self.arrange_anchored(tree, &children, content_origin, content_size);
            }
        }

        // Recurse into children.
        for child_id in children {
            self.arrange_recursive(tree, child_id);
        }
    }

    // -- Flex arrange --------------------------------------------------------

    fn arrange_flex(
        &mut self,
        tree: &mut UITree,
        flex: &FlexLayout,
        children: &[UIId],
        origin: Vec2,
        available: Vec2,
    ) {
        if children.is_empty() {
            return;
        }

        let is_horizontal = flex.direction == LayoutDirection::Horizontal;
        let main_available = if is_horizontal { available.x } else { available.y };
        let cross_available = if is_horizontal { available.y } else { available.x };

        // Build child info.
        self.flex_scratch.clear();
        for &child_id in children {
            let ms = self.get_measured(child_id);
            let (main, cross) = if is_horizontal {
                (ms.width, ms.height)
            } else {
                (ms.height, ms.width)
            };

            let (grow, shrink, basis) = tree
                .get(child_id)
                .map(|n| {
                    let p = &n.layout_params;
                    (
                        p.flex_grow,
                        p.flex_shrink,
                        p.flex_basis.unwrap_or(main),
                    )
                })
                .unwrap_or((0.0, 1.0, main));

            let margin = tree
                .get(child_id)
                .map(|n| n.layout_params.margin)
                .unwrap_or_default();
            let (m_before, m_after, mc_before, mc_after) = if is_horizontal {
                (margin.left, margin.right, margin.top, margin.bottom)
            } else {
                (margin.top, margin.bottom, margin.left, margin.right)
            };

            self.flex_scratch.push(FlexChildInfo {
                id: child_id,
                main_size: main,
                cross_size: cross,
                flex_grow: grow,
                flex_shrink: shrink,
                flex_basis: basis,
                margin_before: m_before,
                margin_after: m_after,
                margin_cross_before: mc_before,
                margin_cross_after: mc_after,
                frozen: false,
                final_main: basis,
                final_cross: cross,
                final_offset: 0.0,
                final_cross_offset: 0.0,
            });
        }

        // Compute total basis + gaps.
        let gap_count = (self.flex_scratch.len().saturating_sub(1)) as f32;
        let total_gaps = gap_count * flex.gap;
        let total_margins: f32 = self
            .flex_scratch
            .iter()
            .map(|c| c.margin_before + c.margin_after)
            .sum();
        let total_basis: f32 = self.flex_scratch.iter().map(|c| c.flex_basis).sum();
        let remaining = main_available - total_basis - total_gaps - total_margins;

        // Distribute flex grow or shrink.
        if remaining > 0.0 {
            let total_grow: f32 = self.flex_scratch.iter().map(|c| c.flex_grow).sum();
            if total_grow > 0.0 {
                for child in &mut self.flex_scratch {
                    child.final_main =
                        child.flex_basis + remaining * (child.flex_grow / total_grow);
                }
            }
        } else if remaining < 0.0 {
            let total_shrink: f32 = self
                .flex_scratch
                .iter()
                .map(|c| c.flex_shrink * c.flex_basis)
                .sum();
            if total_shrink > 0.0 {
                for child in &mut self.flex_scratch {
                    let shrink_amount =
                        remaining.abs() * (child.flex_shrink * child.flex_basis / total_shrink);
                    child.final_main = (child.flex_basis - shrink_amount).max(0.0);
                }
            }
        }

        // Compute cross sizes.
        for child in &mut self.flex_scratch {
            let align = tree
                .get(child.id)
                .and_then(|n| n.layout_params.align_self)
                .unwrap_or(flex.align_items);

            child.final_cross = match align {
                LayoutAlign::Stretch => {
                    cross_available - child.margin_cross_before - child.margin_cross_after
                }
                _ => child.cross_size,
            };
        }

        // Position children along main axis.
        let total_final_main: f32 = self.flex_scratch.iter().map(|c| c.final_main).sum();
        let total_final_margins: f32 = self
            .flex_scratch
            .iter()
            .map(|c| c.margin_before + c.margin_after)
            .sum();
        let free_space =
            (main_available - total_final_main - total_gaps - total_final_margins).max(0.0);

        let n = self.flex_scratch.len() as f32;
        let (initial_offset, inter_item_space) = match flex.justify_content {
            LayoutAlign::Start => (0.0, 0.0),
            LayoutAlign::End => (free_space, 0.0),
            LayoutAlign::Center => (free_space / 2.0, 0.0),
            LayoutAlign::SpaceBetween => {
                if n > 1.0 {
                    (0.0, free_space / (n - 1.0))
                } else {
                    (0.0, 0.0)
                }
            }
            LayoutAlign::SpaceAround => {
                let space = free_space / n;
                (space / 2.0, space)
            }
            LayoutAlign::SpaceEvenly => {
                let space = free_space / (n + 1.0);
                (space, space)
            }
            LayoutAlign::Stretch => (0.0, 0.0),
        };

        let mut cursor = initial_offset;
        for child in &mut self.flex_scratch {
            cursor += child.margin_before;
            child.final_offset = cursor;
            cursor += child.final_main + child.margin_after + flex.gap + inter_item_space;
        }

        // Position children along cross axis.
        for child in &mut self.flex_scratch {
            let align = tree
                .get(child.id)
                .and_then(|n| n.layout_params.align_self)
                .unwrap_or(flex.align_items);

            let cross_free = cross_available
                - child.final_cross
                - child.margin_cross_before
                - child.margin_cross_after;
            child.final_cross_offset = child.margin_cross_before
                + match align {
                    LayoutAlign::Start => 0.0,
                    LayoutAlign::End => cross_free.max(0.0),
                    LayoutAlign::Center => (cross_free / 2.0).max(0.0),
                    LayoutAlign::Stretch => 0.0,
                    _ => 0.0,
                };
        }

        // Apply computed positions to nodes.
        for child in &self.flex_scratch {
            if let Some(node) = tree.get_mut(child.id) {
                let (x, y, w, h) = if is_horizontal {
                    (
                        origin.x + child.final_offset,
                        origin.y + child.final_cross_offset,
                        child.final_main,
                        child.final_cross,
                    )
                } else {
                    (
                        origin.x + child.final_cross_offset,
                        origin.y + child.final_offset,
                        child.final_cross,
                        child.final_main,
                    )
                };
                node.computed_rect = Rect::new(Vec2::new(x, y), Vec2::new(x + w, y + h));
            }
        }
    }

    // -- Grid arrange --------------------------------------------------------

    fn arrange_grid(
        &mut self,
        tree: &mut UITree,
        grid: &GridLayout,
        children: &[UIId],
        origin: Vec2,
        available: Vec2,
    ) {
        let num_cols = grid.columns.len().max(1);
        let num_rows = if grid.rows.is_empty() {
            (children.len() + num_cols - 1) / num_cols
        } else {
            grid.rows.len()
        };

        // Resolve column widths.
        let col_widths = resolve_tracks(&grid.columns, available.x, grid.column_gap, num_cols);
        // Resolve row heights.
        let row_tracks = if grid.rows.is_empty() {
            vec![TrackSize::Auto; num_rows]
        } else {
            grid.rows.clone()
        };

        // For auto rows, measure children to determine heights.
        let mut row_heights = resolve_tracks(&row_tracks, available.y, grid.row_gap, num_rows);
        for (i, &child_id) in children.iter().enumerate() {
            let row = i / num_cols;
            if row >= num_rows {
                break;
            }
            if matches!(
                row_tracks.get(row).unwrap_or(&TrackSize::Auto),
                TrackSize::Auto
            ) {
                let ms = self.get_measured(child_id);
                row_heights[row] = row_heights[row].max(ms.height);
            }
        }

        // Also measure auto-width columns.
        for (i, &child_id) in children.iter().enumerate() {
            let col = i % num_cols;
            if matches!(
                grid.columns.get(col).unwrap_or(&TrackSize::Auto),
                TrackSize::Auto
            ) {
                let ms = self.get_measured(child_id);
                // The resolve_tracks already handled fixed/fractional; for Auto
                // we need to update.
                if col < col_widths.len() {
                    // col_widths was resolved; for auto tracks it starts at 0.
                    // We take the max of all children in this column.
                    // Since resolve_tracks returns 0 for Auto, we need to fix.
                }
                let _ = ms;
            }
        }

        // Compute column positions.
        let mut col_positions = Vec::with_capacity(num_cols);
        let mut x = origin.x;
        for (i, &w) in col_widths.iter().enumerate() {
            col_positions.push(x);
            x += w;
            if i < num_cols - 1 {
                x += grid.column_gap;
            }
        }

        // Compute row positions.
        let mut row_positions = Vec::with_capacity(num_rows);
        let mut y = origin.y;
        for (i, &h) in row_heights.iter().enumerate() {
            row_positions.push(y);
            y += h;
            if i < num_rows - 1 {
                y += grid.row_gap;
            }
        }

        // Place children.
        for (i, &child_id) in children.iter().enumerate() {
            let (col, row, col_span, row_span) = tree
                .get(child_id)
                .map(|n| {
                    let c = n.layout_params.grid_column.unwrap_or(
                        crate::core::GridPlacement::single((i % num_cols) as u16),
                    );
                    let r = n.layout_params.grid_row.unwrap_or(
                        crate::core::GridPlacement::single((i / num_cols) as u16),
                    );
                    (
                        c.start as usize,
                        r.start as usize,
                        c.span as usize,
                        r.span as usize,
                    )
                })
                .unwrap_or((i % num_cols, i / num_cols, 1, 1));

            if col >= num_cols || row >= num_rows {
                continue;
            }

            let cell_x = col_positions.get(col).copied().unwrap_or(origin.x);
            let cell_y = row_positions.get(row).copied().unwrap_or(origin.y);

            // Compute spanned width.
            let end_col = (col + col_span).min(num_cols);
            let cell_w: f32 = col_widths[col..end_col].iter().sum::<f32>()
                + grid.column_gap * (col_span.saturating_sub(1)) as f32;

            let end_row = (row + row_span).min(num_rows);
            let cell_h: f32 = row_heights[row..end_row].iter().sum::<f32>()
                + grid.row_gap * (row_span.saturating_sub(1)) as f32;

            // Align within cell.
            let ms = self.get_measured(child_id);
            let (final_x, final_w) = align_in_cell(cell_x, cell_w, ms.width, grid.justify_items);
            let (final_y, final_h) = align_in_cell(cell_y, cell_h, ms.height, grid.align_items);

            if let Some(node) = tree.get_mut(child_id) {
                node.computed_rect = Rect::new(
                    Vec2::new(final_x, final_y),
                    Vec2::new(final_x + final_w, final_y + final_h),
                );
            }
        }
    }

    // -- Stack / anchor arrange ----------------------------------------------

    fn arrange_stack(
        &mut self,
        tree: &mut UITree,
        children: &[UIId],
        origin: Vec2,
        available: Vec2,
    ) {
        self.arrange_anchored(tree, children, origin, available);
    }

    fn arrange_anchored(
        &mut self,
        tree: &mut UITree,
        children: &[UIId],
        origin: Vec2,
        available: Vec2,
    ) {
        for &child_id in children {
            let ms = self.get_measured(child_id);
            let (anchor, margin, w_size, h_size) = match tree.get(child_id) {
                Some(n) => (
                    n.layout_params.anchor,
                    n.layout_params.margin,
                    n.layout_params.width,
                    n.layout_params.height,
                ),
                None => continue,
            };

            let anchor_min_x = origin.x + anchor.min.x * available.x;
            let anchor_min_y = origin.y + anchor.min.y * available.y;
            let anchor_max_x = origin.x + anchor.max.x * available.x;
            let anchor_max_y = origin.y + anchor.max.y * available.y;

            let (x, w) = if (anchor.min.x - anchor.max.x).abs() < 1e-5 {
                // Point anchor — use measured width.
                let w = w_size.resolve(available.x, ms.width);
                let x = anchor_min_x + margin.left;
                (x, w)
            } else {
                // Stretch anchor.
                let x = anchor_min_x + margin.left;
                let w = (anchor_max_x - margin.right) - x;
                (x, w.max(0.0))
            };

            let (y, h) = if (anchor.min.y - anchor.max.y).abs() < 1e-5 {
                let h = h_size.resolve(available.y, ms.height);
                let y = anchor_min_y + margin.top;
                (y, h)
            } else {
                let y = anchor_min_y + margin.top;
                let h = (anchor_max_y - margin.bottom) - y;
                (y, h.max(0.0))
            };

            if let Some(node) = tree.get_mut(child_id) {
                node.computed_rect = Rect::new(Vec2::new(x, y), Vec2::new(x + w, y + h));
            }
        }
    }

    // -- Scroll arrange ------------------------------------------------------

    fn arrange_scroll(
        &mut self,
        tree: &mut UITree,
        children: &[UIId],
        origin: Vec2,
        available: Vec2,
        scroll_offset: Vec2,
    ) {
        // Arrange children as if stacked, then offset by scroll.
        let scrolled_origin = origin - scroll_offset;

        // For scroll layout, children are laid out in a column by default.
        let mut cursor_y = scrolled_origin.y;
        for &child_id in children {
            let ms = self.get_measured(child_id);
            let w = ms.width.min(available.x);
            let h = ms.height;

            if let Some(node) = tree.get_mut(child_id) {
                node.computed_rect = Rect::new(
                    Vec2::new(scrolled_origin.x, cursor_y),
                    Vec2::new(scrolled_origin.x + w, cursor_y + h),
                );
            }
            cursor_y += h;
        }
    }
}

impl Default for LayoutEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn apply_minmax(value: f32, min: Option<f32>, max: Option<f32>) -> f32 {
    let mut v = value;
    if let Some(min) = min {
        v = v.max(min);
    }
    if let Some(max) = max {
        v = v.min(max);
    }
    v
}

/// Resolve a list of track sizes into pixel sizes.
fn resolve_tracks(
    tracks: &[TrackSize],
    available: f32,
    gap: f32,
    count: usize,
) -> Vec<f32> {
    let total_gaps = gap * (count.saturating_sub(1)) as f32;
    let remaining = (available - total_gaps).max(0.0);

    let mut sizes = vec![0.0_f32; count];
    let mut fixed_total = 0.0_f32;
    let mut frac_total = 0.0_f32;

    for (i, track) in tracks.iter().enumerate().take(count) {
        match track {
            TrackSize::Fixed(v) => {
                sizes[i] = *v;
                fixed_total += *v;
            }
            TrackSize::Percent(p) => {
                let v = remaining * p;
                sizes[i] = v;
                fixed_total += v;
            }
            TrackSize::Fractional(f) => {
                frac_total += *f;
            }
            TrackSize::Auto => {
                // Auto tracks get 0 initially; caller fills in measured sizes.
            }
        }
    }

    // Distribute remaining space to fractional tracks.
    if frac_total > 0.0 {
        let frac_space = (remaining - fixed_total).max(0.0);
        for (i, track) in tracks.iter().enumerate().take(count) {
            if let TrackSize::Fractional(f) = track {
                sizes[i] = frac_space * (*f / frac_total);
            }
        }
    }

    // Fill missing tracks with equal share.
    if tracks.len() < count {
        let auto_count = count - tracks.len();
        let auto_space = (remaining - fixed_total) / auto_count as f32;
        for size in sizes.iter_mut().skip(tracks.len()) {
            *size = auto_space.max(0.0);
        }
    }

    sizes
}

/// Align a child of size `child_size` within a cell at `cell_pos` of size
/// `cell_size`.
fn align_in_cell(
    cell_pos: f32,
    cell_size: f32,
    child_size: f32,
    align: LayoutAlign,
) -> (f32, f32) {
    match align {
        LayoutAlign::Start => (cell_pos, child_size),
        LayoutAlign::End => (cell_pos + cell_size - child_size, child_size),
        LayoutAlign::Center => (cell_pos + (cell_size - child_size) / 2.0, child_size),
        LayoutAlign::Stretch => (cell_pos, cell_size),
        _ => (cell_pos, child_size),
    }
}
