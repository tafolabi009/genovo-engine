//! Slate layout containers for the Genovo UI framework.
//!
//! Provides a comprehensive set of layout containers that position and size
//! their children according to different strategies: horizontal/vertical boxes,
//! overlays, canvases, grids, scroll boxes, splitters, and more.
//!
//! Each container implements a two-pass layout algorithm:
//! 1. **Measure pass** -- compute desired size bottom-up
//! 2. **Arrange pass** -- distribute space top-down

use glam::Vec2;
use serde::{Deserialize, Serialize};

use genovo_core::Rect;

use crate::render_commands::Color;

// Re-export types used heavily in this module.
pub use crate::brush_system::BrushMargin;

// ---------------------------------------------------------------------------
// SizeRule
// ---------------------------------------------------------------------------

/// Determines how a slot in a box layout is sized.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SizeRule {
    /// Size to content (automatic sizing).
    Auto,
    /// Fill remaining space with the given weight.
    Fill(f32),
    /// Fixed pixel size.
    Fixed(f32),
}

impl Default for SizeRule {
    fn default() -> Self {
        Self::Auto
    }
}

impl SizeRule {
    /// Returns true if this rule contributes to the fill pool.
    pub fn is_fill(&self) -> bool {
        matches!(self, Self::Fill(_))
    }

    /// Returns the fill weight, or 0 if not a fill rule.
    pub fn fill_weight(&self) -> f32 {
        match self {
            Self::Fill(w) => *w,
            _ => 0.0,
        }
    }

    /// Returns the fixed size, or 0 if not fixed.
    pub fn fixed_size(&self) -> f32 {
        match self {
            Self::Fixed(s) => *s,
            _ => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// HAlign / VAlign
// ---------------------------------------------------------------------------

/// Horizontal alignment within a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HAlign {
    /// Align to the left edge.
    Start,
    /// Center horizontally.
    Center,
    /// Align to the right edge.
    End,
    /// Stretch to fill the available width.
    Fill,
}

impl Default for HAlign {
    fn default() -> Self {
        Self::Fill
    }
}

impl HAlign {
    /// Compute the X offset for a child within a slot.
    pub fn align_offset(&self, child_width: f32, slot_width: f32) -> f32 {
        match self {
            Self::Start | Self::Fill => 0.0,
            Self::Center => (slot_width - child_width).max(0.0) * 0.5,
            Self::End => (slot_width - child_width).max(0.0),
        }
    }

    /// Compute the width for a child given the slot width.
    pub fn resolve_width(
        &self,
        desired_width: f32,
        slot_width: f32,
    ) -> f32 {
        match self {
            Self::Fill => slot_width,
            _ => desired_width.min(slot_width),
        }
    }
}

/// Vertical alignment within a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VAlign {
    /// Align to the top edge.
    Start,
    /// Center vertically.
    Center,
    /// Align to the bottom edge.
    End,
    /// Stretch to fill the available height.
    Fill,
}

impl Default for VAlign {
    fn default() -> Self {
        Self::Fill
    }
}

impl VAlign {
    /// Compute the Y offset for a child within a slot.
    pub fn align_offset(
        &self,
        child_height: f32,
        slot_height: f32,
    ) -> f32 {
        match self {
            Self::Start | Self::Fill => 0.0,
            Self::Center => (slot_height - child_height).max(0.0) * 0.5,
            Self::End => (slot_height - child_height).max(0.0),
        }
    }

    /// Compute the height for a child given the slot height.
    pub fn resolve_height(
        &self,
        desired_height: f32,
        slot_height: f32,
    ) -> f32 {
        match self {
            Self::Fill => slot_height,
            _ => desired_height.min(slot_height),
        }
    }
}

// ---------------------------------------------------------------------------
// SlateMargin
// ---------------------------------------------------------------------------

/// Edge offsets (left, top, right, bottom) used for layout margins and
/// padding. Named `SlateMargin` to avoid conflicts with the core `Margin`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SlateMargin {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl SlateMargin {
    pub const ZERO: Self = Self {
        left: 0.0,
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
    };

    /// All sides equal.
    pub fn all(v: f32) -> Self {
        Self {
            left: v,
            top: v,
            right: v,
            bottom: v,
        }
    }

    /// Symmetric horizontal and vertical.
    pub fn symmetric(h: f32, v: f32) -> Self {
        Self {
            left: h,
            top: v,
            right: h,
            bottom: v,
        }
    }

    /// Explicit per-side.
    pub fn new(l: f32, t: f32, r: f32, b: f32) -> Self {
        Self {
            left: l,
            top: t,
            right: r,
            bottom: b,
        }
    }

    /// Total horizontal edge.
    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    /// Total vertical edge.
    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }

    /// Shrink a rectangle by this margin.
    pub fn shrink_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.min.x + self.left, rect.min.y + self.top),
            Vec2::new(
                rect.max.x - self.right,
                rect.max.y - self.bottom,
            ),
        )
    }

    /// Expand a rectangle by this margin.
    pub fn expand_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.min.x - self.left, rect.min.y - self.top),
            Vec2::new(
                rect.max.x + self.right,
                rect.max.y + self.bottom,
            ),
        )
    }
}

impl Default for SlateMargin {
    fn default() -> Self {
        Self::ZERO
    }
}

// ---------------------------------------------------------------------------
// BoxSlot
// ---------------------------------------------------------------------------

/// Configuration for a single child slot in a HorizontalBox or VerticalBox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxSlot {
    /// How this slot is sized along the primary axis.
    pub size_rule: SizeRule,
    /// Horizontal alignment of the child within the slot.
    pub h_align: HAlign,
    /// Vertical alignment of the child within the slot.
    pub v_align: VAlign,
    /// Padding around the child within the slot.
    pub padding: SlateMargin,
    /// Maximum size constraint (0 = unconstrained).
    pub max_size: f32,
    /// Desired size reported by the child during the measure pass.
    pub desired_size: Vec2,
    /// Final computed rect after arrange.
    pub computed_rect: Rect,
}

impl BoxSlot {
    /// Creates a new auto-sized slot.
    pub fn auto() -> Self {
        Self {
            size_rule: SizeRule::Auto,
            h_align: HAlign::Fill,
            v_align: VAlign::Fill,
            padding: SlateMargin::ZERO,
            max_size: 0.0,
            desired_size: Vec2::ZERO,
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    /// Creates a fill slot with the given weight.
    pub fn fill(weight: f32) -> Self {
        Self {
            size_rule: SizeRule::Fill(weight),
            ..Self::auto()
        }
    }

    /// Creates a fixed-size slot.
    pub fn fixed(pixels: f32) -> Self {
        Self {
            size_rule: SizeRule::Fixed(pixels),
            ..Self::auto()
        }
    }

    /// Builder: set horizontal alignment.
    pub fn with_h_align(mut self, align: HAlign) -> Self {
        self.h_align = align;
        self
    }

    /// Builder: set vertical alignment.
    pub fn with_v_align(mut self, align: VAlign) -> Self {
        self.v_align = align;
        self
    }

    /// Builder: set padding.
    pub fn with_padding(mut self, padding: SlateMargin) -> Self {
        self.padding = padding;
        self
    }

    /// Builder: set max size.
    pub fn with_max_size(mut self, max: f32) -> Self {
        self.max_size = max;
        self
    }
}

impl Default for BoxSlot {
    fn default() -> Self {
        Self::auto()
    }
}

// ---------------------------------------------------------------------------
// HorizontalBox
// ---------------------------------------------------------------------------

/// Lays out children left-to-right in a horizontal row.
///
/// Each child occupies a slot with independent sizing rules, alignment,
/// and padding. Auto-sized slots take their child's desired width; Fill
/// slots share remaining space by weight; Fixed slots are exactly sized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizontalBox {
    /// Slots for each child.
    pub slots: Vec<BoxSlot>,
    /// Spacing between adjacent children.
    pub spacing: f32,
    /// Computed desired size of the entire box.
    pub desired_size: Vec2,
    /// Computed layout rect.
    pub computed_rect: Rect,
}

impl HorizontalBox {
    /// Creates an empty horizontal box.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            spacing: 0.0,
            desired_size: Vec2::ZERO,
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    /// Builder: set spacing.
    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    /// Add a slot.
    pub fn add_slot(&mut self, slot: BoxSlot) {
        self.slots.push(slot);
    }

    /// Set the desired size for a child (called during measure pass).
    pub fn set_child_desired_size(
        &mut self,
        index: usize,
        size: Vec2,
    ) {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.desired_size = size;
        }
    }

    /// Measure pass: compute the desired size of the box from children.
    pub fn compute_desired_size(&mut self) -> Vec2 {
        let mut total_width: f32 = 0.0;
        let mut max_height: f32 = 0.0;

        for (i, slot) in self.slots.iter().enumerate() {
            let child_width = match slot.size_rule {
                SizeRule::Auto => {
                    slot.desired_size.x + slot.padding.horizontal()
                }
                SizeRule::Fixed(px) => px + slot.padding.horizontal(),
                SizeRule::Fill(_) => slot.padding.horizontal(),
            };
            total_width += child_width;
            let child_height =
                slot.desired_size.y + slot.padding.vertical();
            max_height = max_height.max(child_height);
            if i > 0 {
                total_width += self.spacing;
            }
        }

        self.desired_size = Vec2::new(total_width, max_height);
        self.desired_size
    }

    /// Arrange pass: distribute space to children within the given rect.
    pub fn arrange_children(&mut self, available: Rect) {
        self.computed_rect = available;
        let total_width = available.width();
        let total_height = available.height();

        let mut used_width: f32 = 0.0;
        let mut total_fill_weight: f32 = 0.0;
        let num_gaps = self.slots.len().saturating_sub(1);
        let total_spacing = num_gaps as f32 * self.spacing;

        for slot in &self.slots {
            match slot.size_rule {
                SizeRule::Auto => {
                    used_width +=
                        slot.desired_size.x + slot.padding.horizontal();
                }
                SizeRule::Fixed(px) => {
                    used_width += px + slot.padding.horizontal();
                }
                SizeRule::Fill(w) => {
                    used_width += slot.padding.horizontal();
                    total_fill_weight += w;
                }
            }
        }

        let remaining =
            (total_width - used_width - total_spacing).max(0.0);

        let mut x = available.min.x;
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if i > 0 {
                x += self.spacing;
            }

            let slot_width = match slot.size_rule {
                SizeRule::Auto => slot.desired_size.x,
                SizeRule::Fixed(px) => px,
                SizeRule::Fill(w) => {
                    if total_fill_weight > 0.0 {
                        remaining * (w / total_fill_weight)
                    } else {
                        0.0
                    }
                }
            };

            let padded_x = x + slot.padding.left;
            let padded_width = slot_width;
            let slot_total_width =
                padded_width + slot.padding.horizontal();

            let child_height = slot.v_align.resolve_height(
                slot.desired_size.y,
                total_height - slot.padding.vertical(),
            );
            let child_width = slot
                .h_align
                .resolve_width(slot.desired_size.x, padded_width);
            let y_offset = slot.v_align.align_offset(
                child_height,
                total_height - slot.padding.vertical(),
            );
            let x_offset =
                slot.h_align.align_offset(child_width, padded_width);

            slot.computed_rect = Rect::new(
                Vec2::new(
                    padded_x + x_offset,
                    available.min.y + slot.padding.top + y_offset,
                ),
                Vec2::new(
                    padded_x + x_offset + child_width,
                    available.min.y
                        + slot.padding.top
                        + y_offset
                        + child_height,
                ),
            );

            if slot.max_size > 0.0 {
                let w = slot.computed_rect.width().min(slot.max_size);
                slot.computed_rect.max.x =
                    slot.computed_rect.min.x + w;
            }

            x += slot_total_width;
        }
    }

    /// Get the computed rect for a child.
    pub fn child_rect(&self, index: usize) -> Option<Rect> {
        self.slots.get(index).map(|s| s.computed_rect)
    }

    /// Number of slots.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

impl Default for HorizontalBox {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VerticalBox
// ---------------------------------------------------------------------------

/// Lays out children top-to-bottom in a vertical column.
///
/// Same slot system as `HorizontalBox` but the primary axis is vertical.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerticalBox {
    pub slots: Vec<BoxSlot>,
    pub spacing: f32,
    pub desired_size: Vec2,
    pub computed_rect: Rect,
}

impl VerticalBox {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            spacing: 0.0,
            desired_size: Vec2::ZERO,
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    pub fn add_slot(&mut self, slot: BoxSlot) {
        self.slots.push(slot);
    }

    pub fn set_child_desired_size(
        &mut self,
        index: usize,
        size: Vec2,
    ) {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.desired_size = size;
        }
    }

    pub fn compute_desired_size(&mut self) -> Vec2 {
        let mut max_width: f32 = 0.0;
        let mut total_height: f32 = 0.0;

        for (i, slot) in self.slots.iter().enumerate() {
            let child_height = match slot.size_rule {
                SizeRule::Auto => {
                    slot.desired_size.y + slot.padding.vertical()
                }
                SizeRule::Fixed(px) => px + slot.padding.vertical(),
                SizeRule::Fill(_) => slot.padding.vertical(),
            };
            total_height += child_height;
            let child_width =
                slot.desired_size.x + slot.padding.horizontal();
            max_width = max_width.max(child_width);
            if i > 0 {
                total_height += self.spacing;
            }
        }

        self.desired_size = Vec2::new(max_width, total_height);
        self.desired_size
    }

    pub fn arrange_children(&mut self, available: Rect) {
        self.computed_rect = available;
        let total_width = available.width();
        let total_height = available.height();

        let mut used_height: f32 = 0.0;
        let mut total_fill_weight: f32 = 0.0;
        let num_gaps = self.slots.len().saturating_sub(1);
        let total_spacing = num_gaps as f32 * self.spacing;

        for slot in &self.slots {
            match slot.size_rule {
                SizeRule::Auto => {
                    used_height +=
                        slot.desired_size.y + slot.padding.vertical();
                }
                SizeRule::Fixed(px) => {
                    used_height += px + slot.padding.vertical();
                }
                SizeRule::Fill(w) => {
                    used_height += slot.padding.vertical();
                    total_fill_weight += w;
                }
            }
        }

        let remaining =
            (total_height - used_height - total_spacing).max(0.0);

        let mut y = available.min.y;
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if i > 0 {
                y += self.spacing;
            }

            let slot_height = match slot.size_rule {
                SizeRule::Auto => slot.desired_size.y,
                SizeRule::Fixed(px) => px,
                SizeRule::Fill(w) => {
                    if total_fill_weight > 0.0 {
                        remaining * (w / total_fill_weight)
                    } else {
                        0.0
                    }
                }
            };

            let padded_y = y + slot.padding.top;
            let padded_height = slot_height;
            let slot_total_height =
                padded_height + slot.padding.vertical();

            let child_width = slot.h_align.resolve_width(
                slot.desired_size.x,
                total_width - slot.padding.horizontal(),
            );
            let child_height = slot
                .v_align
                .resolve_height(slot.desired_size.y, padded_height);
            let x_offset = slot.h_align.align_offset(
                child_width,
                total_width - slot.padding.horizontal(),
            );
            let y_offset = slot
                .v_align
                .align_offset(child_height, padded_height);

            slot.computed_rect = Rect::new(
                Vec2::new(
                    available.min.x + slot.padding.left + x_offset,
                    padded_y + y_offset,
                ),
                Vec2::new(
                    available.min.x
                        + slot.padding.left
                        + x_offset
                        + child_width,
                    padded_y + y_offset + child_height,
                ),
            );

            if slot.max_size > 0.0 {
                let h = slot.computed_rect.height().min(slot.max_size);
                slot.computed_rect.max.y =
                    slot.computed_rect.min.y + h;
            }

            y += slot_total_height;
        }
    }

    pub fn child_rect(&self, index: usize) -> Option<Rect> {
        self.slots.get(index).map(|s| s.computed_rect)
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

impl Default for VerticalBox {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------

/// Stacks all children on top of each other in the same rectangle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Overlay {
    pub slots: Vec<OverlaySlot>,
    pub desired_size: Vec2,
    pub computed_rect: Rect,
}

/// Configuration for a child in an overlay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlaySlot {
    pub h_align: HAlign,
    pub v_align: VAlign,
    pub padding: SlateMargin,
    pub desired_size: Vec2,
    pub computed_rect: Rect,
}

impl OverlaySlot {
    pub fn new() -> Self {
        Self {
            h_align: HAlign::Fill,
            v_align: VAlign::Fill,
            padding: SlateMargin::ZERO,
            desired_size: Vec2::ZERO,
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }
}

impl Default for OverlaySlot {
    fn default() -> Self {
        Self::new()
    }
}

impl Overlay {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            desired_size: Vec2::ZERO,
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    pub fn add_slot(&mut self, slot: OverlaySlot) {
        self.slots.push(slot);
    }

    pub fn compute_desired_size(&mut self) -> Vec2 {
        let mut mw: f32 = 0.0;
        let mut mh: f32 = 0.0;
        for s in &self.slots {
            mw = mw.max(s.desired_size.x + s.padding.horizontal());
            mh = mh.max(s.desired_size.y + s.padding.vertical());
        }
        self.desired_size = Vec2::new(mw, mh);
        self.desired_size
    }

    pub fn arrange_children(&mut self, available: Rect) {
        self.computed_rect = available;
        let w = available.width();
        let h = available.height();
        for slot in &mut self.slots {
            let iw = w - slot.padding.horizontal();
            let ih = h - slot.padding.vertical();
            let cw =
                slot.h_align.resolve_width(slot.desired_size.x, iw);
            let ch =
                slot.v_align.resolve_height(slot.desired_size.y, ih);
            let xo = slot.h_align.align_offset(cw, iw);
            let yo = slot.v_align.align_offset(ch, ih);
            slot.computed_rect = Rect::new(
                Vec2::new(
                    available.min.x + slot.padding.left + xo,
                    available.min.y + slot.padding.top + yo,
                ),
                Vec2::new(
                    available.min.x + slot.padding.left + xo + cw,
                    available.min.y + slot.padding.top + yo + ch,
                ),
            );
        }
    }

    pub fn child_rect(&self, index: usize) -> Option<Rect> {
        self.slots.get(index).map(|s| s.computed_rect)
    }
}

impl Default for Overlay {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Canvas
// ---------------------------------------------------------------------------

/// Absolute-positioned children with anchor + offset positioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Canvas {
    pub slots: Vec<CanvasSlot>,
    pub desired_size: Vec2,
    pub computed_rect: Rect,
}

/// Positioning data for a child in a Canvas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanvasSlot {
    pub anchor: Vec2,
    pub offset: Vec2,
    pub size: Vec2,
    pub alignment: Vec2,
    pub auto_size: bool,
    pub desired_size: Vec2,
    pub computed_rect: Rect,
    pub z_order: i32,
}

impl CanvasSlot {
    pub fn new(anchor: Vec2, offset: Vec2) -> Self {
        Self {
            anchor,
            offset,
            size: Vec2::ZERO,
            alignment: Vec2::ZERO,
            auto_size: true,
            desired_size: Vec2::ZERO,
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
            z_order: 0,
        }
    }

    pub fn with_size(mut self, size: Vec2) -> Self {
        self.size = size;
        self.auto_size = false;
        self
    }

    pub fn with_alignment(mut self, alignment: Vec2) -> Self {
        self.alignment = alignment;
        self
    }

    pub fn at_position(x: f32, y: f32) -> Self {
        Self::new(Vec2::ZERO, Vec2::new(x, y))
    }

    pub fn centered(offset: Vec2) -> Self {
        Self::new(Vec2::new(0.5, 0.5), offset)
            .with_alignment(Vec2::new(0.5, 0.5))
    }
}

impl Canvas {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            desired_size: Vec2::ZERO,
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    pub fn add_slot(&mut self, slot: CanvasSlot) {
        self.slots.push(slot);
    }

    pub fn arrange_children(&mut self, available: Rect) {
        self.computed_rect = available;
        let w = available.width();
        let h = available.height();
        for slot in &mut self.slots {
            let cs = if slot.auto_size {
                slot.desired_size
            } else {
                slot.size
            };
            let ax = available.min.x + w * slot.anchor.x;
            let ay = available.min.y + h * slot.anchor.y;
            let alx = cs.x * slot.alignment.x;
            let aly = cs.y * slot.alignment.y;
            let x = ax + slot.offset.x - alx;
            let y = ay + slot.offset.y - aly;
            slot.computed_rect = Rect::new(
                Vec2::new(x, y),
                Vec2::new(x + cs.x, y + cs.y),
            );
        }
    }

    pub fn child_rect(&self, index: usize) -> Option<Rect> {
        self.slots.get(index).map(|s| s.computed_rect)
    }
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GridPanel
// ---------------------------------------------------------------------------

/// A grid layout with row/column spans and fill weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPanel {
    pub slots: Vec<GridSlot>,
    pub num_columns: usize,
    pub num_rows: usize,
    pub column_fills: Vec<f32>,
    pub row_fills: Vec<f32>,
    pub column_spacing: f32,
    pub row_spacing: f32,
    pub column_widths: Vec<f32>,
    pub row_heights: Vec<f32>,
    pub desired_size: Vec2,
    pub computed_rect: Rect,
}

/// A child slot in a grid panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSlot {
    pub column: usize,
    pub row: usize,
    pub column_span: usize,
    pub row_span: usize,
    pub h_align: HAlign,
    pub v_align: VAlign,
    pub padding: SlateMargin,
    pub desired_size: Vec2,
    pub computed_rect: Rect,
}

impl GridSlot {
    pub fn new(column: usize, row: usize) -> Self {
        Self {
            column,
            row,
            column_span: 1,
            row_span: 1,
            h_align: HAlign::Fill,
            v_align: VAlign::Fill,
            padding: SlateMargin::ZERO,
            desired_size: Vec2::ZERO,
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    pub fn with_span(mut self, cs: usize, rs: usize) -> Self {
        self.column_span = cs.max(1);
        self.row_span = rs.max(1);
        self
    }
}

impl GridPanel {
    pub fn new(columns: usize, rows: usize) -> Self {
        Self {
            slots: Vec::new(),
            num_columns: columns,
            num_rows: rows,
            column_fills: Vec::new(),
            row_fills: Vec::new(),
            column_spacing: 0.0,
            row_spacing: 0.0,
            column_widths: vec![0.0; columns],
            row_heights: vec![0.0; rows],
            desired_size: Vec2::ZERO,
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    pub fn set_column_fill(&mut self, col: usize, fill: f32) {
        if self.column_fills.len() <= col {
            self.column_fills.resize(col + 1, 0.0);
        }
        self.column_fills[col] = fill;
    }

    pub fn set_row_fill(&mut self, row: usize, fill: f32) {
        if self.row_fills.len() <= row {
            self.row_fills.resize(row + 1, 0.0);
        }
        self.row_fills[row] = fill;
    }

    pub fn add_slot(&mut self, slot: GridSlot) {
        self.slots.push(slot);
    }

    pub fn set_child_desired_size(
        &mut self,
        index: usize,
        size: Vec2,
    ) {
        if let Some(s) = self.slots.get_mut(index) {
            s.desired_size = size;
        }
    }

    pub fn compute_desired_size(&mut self) -> Vec2 {
        self.column_widths = vec![0.0; self.num_columns];
        self.row_heights = vec![0.0; self.num_rows];
        for s in &self.slots {
            if s.column_span == 1 && s.column < self.num_columns {
                let w =
                    s.desired_size.x + s.padding.horizontal();
                self.column_widths[s.column] =
                    self.column_widths[s.column].max(w);
            }
            if s.row_span == 1 && s.row < self.num_rows {
                let h = s.desired_size.y + s.padding.vertical();
                self.row_heights[s.row] =
                    self.row_heights[s.row].max(h);
            }
        }
        let tw: f32 = self.column_widths.iter().sum::<f32>()
            + (self.num_columns.saturating_sub(1)) as f32
                * self.column_spacing;
        let th: f32 = self.row_heights.iter().sum::<f32>()
            + (self.num_rows.saturating_sub(1)) as f32
                * self.row_spacing;
        self.desired_size = Vec2::new(tw, th);
        self.desired_size
    }

    pub fn arrange_children(&mut self, available: Rect) {
        self.computed_rect = available;

        // Distribute extra space to fill columns.
        let dw: f32 = self.column_widths.iter().sum::<f32>()
            + (self.num_columns.saturating_sub(1)) as f32
                * self.column_spacing;
        let ew = (available.width() - dw).max(0.0);
        let tcf: f32 = self.column_fills.iter().sum();
        if tcf > 0.0 && ew > 0.0 {
            for (i, w) in self.column_widths.iter_mut().enumerate() {
                let f = self
                    .column_fills
                    .get(i)
                    .copied()
                    .unwrap_or(0.0);
                if f > 0.0 {
                    *w += ew * (f / tcf);
                }
            }
        }

        // Distribute extra space to fill rows.
        let dh: f32 = self.row_heights.iter().sum::<f32>()
            + (self.num_rows.saturating_sub(1)) as f32
                * self.row_spacing;
        let eh = (available.height() - dh).max(0.0);
        let trf: f32 = self.row_fills.iter().sum();
        if trf > 0.0 && eh > 0.0 {
            for (i, h) in self.row_heights.iter_mut().enumerate() {
                let f =
                    self.row_fills.get(i).copied().unwrap_or(0.0);
                if f > 0.0 {
                    *h += eh * (f / trf);
                }
            }
        }

        // Compute positions.
        let mut cpos = vec![0.0_f32; self.num_columns + 1];
        cpos[0] = available.min.x;
        for i in 0..self.num_columns {
            cpos[i + 1] = cpos[i] + self.column_widths[i];
            if i < self.num_columns - 1 {
                cpos[i + 1] += self.column_spacing;
            }
        }

        let mut rpos = vec![0.0_f32; self.num_rows + 1];
        rpos[0] = available.min.y;
        for i in 0..self.num_rows {
            rpos[i + 1] = rpos[i] + self.row_heights[i];
            if i < self.num_rows - 1 {
                rpos[i + 1] += self.row_spacing;
            }
        }

        // Position each slot.
        for slot in &mut self.slots {
            if slot.column >= self.num_columns
                || slot.row >= self.num_rows
            {
                continue;
            }
            let ec = (slot.column + slot.column_span)
                .min(self.num_columns);
            let er =
                (slot.row + slot.row_span).min(self.num_rows);
            let cx = cpos[slot.column];
            let cy = rpos[slot.row];
            let cw = cpos[ec] - cx;
            let ch = rpos[er] - cy;
            let iw = cw - slot.padding.horizontal();
            let ih = ch - slot.padding.vertical();
            let cwidth =
                slot.h_align.resolve_width(slot.desired_size.x, iw);
            let cheight = slot
                .v_align
                .resolve_height(slot.desired_size.y, ih);
            let xo = slot.h_align.align_offset(cwidth, iw);
            let yo = slot.v_align.align_offset(cheight, ih);
            slot.computed_rect = Rect::new(
                Vec2::new(
                    cx + slot.padding.left + xo,
                    cy + slot.padding.top + yo,
                ),
                Vec2::new(
                    cx + slot.padding.left + xo + cwidth,
                    cy + slot.padding.top + yo + cheight,
                ),
            );
        }
    }

    pub fn child_rect(&self, index: usize) -> Option<Rect> {
        self.slots.get(index).map(|s| s.computed_rect)
    }
}

// ---------------------------------------------------------------------------
// UniformGridPanel
// ---------------------------------------------------------------------------

/// A grid where all cells are the same size. Row-major order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformGridPanel {
    pub num_columns: usize,
    pub child_count: usize,
    pub column_spacing: f32,
    pub row_spacing: f32,
    pub computed_rects: Vec<Rect>,
    pub computed_rect: Rect,
}

impl UniformGridPanel {
    pub fn new(columns: usize) -> Self {
        Self {
            num_columns: columns.max(1),
            child_count: 0,
            column_spacing: 0.0,
            row_spacing: 0.0,
            computed_rects: Vec::new(),
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    pub fn set_child_count(&mut self, count: usize) {
        self.child_count = count;
        self.computed_rects
            .resize(count, Rect::new(Vec2::ZERO, Vec2::ZERO));
    }

    fn effective_rows(&self) -> usize {
        (self.child_count + self.num_columns - 1)
            / self.num_columns.max(1)
    }

    pub fn arrange_children(&mut self, available: Rect) {
        self.computed_rect = available;
        let rows = self.effective_rows();
        let cw = if self.num_columns > 0 {
            (available.width()
                - (self.num_columns.saturating_sub(1)) as f32
                    * self.column_spacing)
                / self.num_columns as f32
        } else {
            0.0
        };
        let ch = if rows > 0 {
            (available.height()
                - rows.saturating_sub(1) as f32 * self.row_spacing)
                / rows as f32
        } else {
            0.0
        };
        for i in 0..self.child_count {
            let col = i % self.num_columns;
            let row = i / self.num_columns;
            let x = available.min.x
                + col as f32 * (cw + self.column_spacing);
            let y = available.min.y
                + row as f32 * (ch + self.row_spacing);
            if let Some(r) = self.computed_rects.get_mut(i) {
                *r = Rect::new(
                    Vec2::new(x, y),
                    Vec2::new(x + cw, y + ch),
                );
            }
        }
    }

    pub fn child_rect(&self, index: usize) -> Option<Rect> {
        self.computed_rects.get(index).copied()
    }
}

// ---------------------------------------------------------------------------
// WrapBox
// ---------------------------------------------------------------------------

/// Children wrap to next line when exceeding width (flexbox wrap).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrapBox {
    pub h_spacing: f32,
    pub v_spacing: f32,
    pub child_desired_sizes: Vec<Vec2>,
    pub child_rects: Vec<Rect>,
    pub desired_size: Vec2,
    pub computed_rect: Rect,
}

impl WrapBox {
    pub fn new() -> Self {
        Self {
            h_spacing: 4.0,
            v_spacing: 4.0,
            child_desired_sizes: Vec::new(),
            child_rects: Vec::new(),
            desired_size: Vec2::ZERO,
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    pub fn with_spacing(mut self, h: f32, v: f32) -> Self {
        self.h_spacing = h;
        self.v_spacing = v;
        self
    }

    pub fn set_child_count(&mut self, count: usize) {
        self.child_desired_sizes.resize(count, Vec2::ZERO);
        self.child_rects
            .resize(count, Rect::new(Vec2::ZERO, Vec2::ZERO));
    }

    pub fn set_child_desired_size(
        &mut self,
        index: usize,
        size: Vec2,
    ) {
        if let Some(s) = self.child_desired_sizes.get_mut(index) {
            *s = size;
        }
    }

    pub fn compute_desired_size(
        &mut self,
        available_width: f32,
    ) -> Vec2 {
        let mut x: f32 = 0.0;
        let mut y: f32 = 0.0;
        let mut rh: f32 = 0.0;
        let mut mw: f32 = 0.0;

        for (i, size) in self.child_desired_sizes.iter().enumerate() {
            if i > 0 && x + size.x > available_width && x > 0.0 {
                y += rh + self.v_spacing;
                x = 0.0;
                rh = 0.0;
            }
            x += size.x;
            mw = mw.max(x);
            rh = rh.max(size.y);
            if i < self.child_desired_sizes.len() - 1 {
                x += self.h_spacing;
            }
        }

        self.desired_size = Vec2::new(mw, y + rh);
        self.desired_size
    }

    pub fn arrange_children(&mut self, available: Rect) {
        self.computed_rect = available;
        let mw = available.width();
        let mut x: f32 = 0.0;
        let mut y: f32 = 0.0;
        let mut rh: f32 = 0.0;

        for (i, size) in self.child_desired_sizes.iter().enumerate() {
            if i > 0 && x + size.x > mw && x > 0.0 {
                y += rh + self.v_spacing;
                x = 0.0;
                rh = 0.0;
            }
            if let Some(r) = self.child_rects.get_mut(i) {
                *r = Rect::new(
                    Vec2::new(
                        available.min.x + x,
                        available.min.y + y,
                    ),
                    Vec2::new(
                        available.min.x + x + size.x,
                        available.min.y + y + size.y,
                    ),
                );
            }
            rh = rh.max(size.y);
            x += size.x + self.h_spacing;
        }
    }

    pub fn child_rect(&self, index: usize) -> Option<Rect> {
        self.child_rects.get(index).copied()
    }
}

impl Default for WrapBox {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WidgetSwitcher
// ---------------------------------------------------------------------------

/// Shows one child at a time by index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetSwitcher {
    pub active_index: usize,
    pub child_count: usize,
    pub desired_sizes: Vec<Vec2>,
    pub computed_rect: Rect,
    pub desired_size: Vec2,
}

impl WidgetSwitcher {
    pub fn new(count: usize) -> Self {
        Self {
            active_index: 0,
            child_count: count,
            desired_sizes: vec![Vec2::ZERO; count],
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
            desired_size: Vec2::ZERO,
        }
    }

    pub fn set_active(&mut self, i: usize) {
        self.active_index =
            i.min(self.child_count.saturating_sub(1));
    }

    pub fn set_child_desired_size(
        &mut self,
        i: usize,
        s: Vec2,
    ) {
        if let Some(d) = self.desired_sizes.get_mut(i) {
            *d = s;
        }
    }

    pub fn compute_desired_size(&mut self) -> Vec2 {
        let mut mw: f32 = 0.0;
        let mut mh: f32 = 0.0;
        for d in &self.desired_sizes {
            mw = mw.max(d.x);
            mh = mh.max(d.y);
        }
        self.desired_size = Vec2::new(mw, mh);
        self.desired_size
    }

    pub fn arrange(&mut self, available: Rect) {
        self.computed_rect = available;
    }
}

// ---------------------------------------------------------------------------
// ScaleBox
// ---------------------------------------------------------------------------

/// How the content is scaled within a ScaleBox.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize,
)]
pub enum StretchMode {
    None,
    Fill,
    Fit,
    FitWidth,
    FitHeight,
    UserSpecified,
}

impl Default for StretchMode {
    fn default() -> Self {
        Self::Fit
    }
}

/// Scales child to fit/fill with multiple stretch modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleBox {
    pub stretch: StretchMode,
    pub user_scale: f32,
    pub h_align: HAlign,
    pub v_align: VAlign,
    pub child_desired_size: Vec2,
    pub computed_scale: f32,
    pub child_rect: Rect,
    pub computed_rect: Rect,
}

impl ScaleBox {
    pub fn new(stretch: StretchMode) -> Self {
        Self {
            stretch,
            user_scale: 1.0,
            h_align: HAlign::Center,
            v_align: VAlign::Center,
            child_desired_size: Vec2::ZERO,
            computed_scale: 1.0,
            child_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    pub fn set_child_desired_size(&mut self, size: Vec2) {
        self.child_desired_size = size;
    }

    pub fn arrange(&mut self, available: Rect) {
        self.computed_rect = available;
        let aw = available.width();
        let ah = available.height();
        let cw = self.child_desired_size.x.max(1.0);
        let ch = self.child_desired_size.y.max(1.0);
        let scale = match self.stretch {
            StretchMode::None => 1.0,
            StretchMode::Fill => (aw / cw).max(ah / ch),
            StretchMode::Fit => (aw / cw).min(ah / ch),
            StretchMode::FitWidth => aw / cw,
            StretchMode::FitHeight => ah / ch,
            StretchMode::UserSpecified => self.user_scale,
        };
        self.computed_scale = scale;
        let sw = cw * scale;
        let sh = ch * scale;
        let xo = self.h_align.align_offset(sw, aw);
        let yo = self.v_align.align_offset(sh, ah);
        self.child_rect = Rect::new(
            Vec2::new(available.min.x + xo, available.min.y + yo),
            Vec2::new(
                available.min.x + xo + sw,
                available.min.y + yo + sh,
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// SizeBox
// ---------------------------------------------------------------------------

/// Single child with size constraints + padding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeBox {
    pub width_override: Option<f32>,
    pub height_override: Option<f32>,
    pub min_width: f32,
    pub max_width: f32,
    pub min_height: f32,
    pub max_height: f32,
    pub padding: SlateMargin,
    pub h_align: HAlign,
    pub v_align: VAlign,
    pub child_desired_size: Vec2,
    pub child_rect: Rect,
    pub computed_rect: Rect,
    pub desired_size: Vec2,
}

impl SizeBox {
    pub fn new() -> Self {
        Self {
            width_override: None,
            height_override: None,
            min_width: 0.0,
            max_width: 0.0,
            min_height: 0.0,
            max_height: 0.0,
            padding: SlateMargin::ZERO,
            h_align: HAlign::Fill,
            v_align: VAlign::Fill,
            child_desired_size: Vec2::ZERO,
            child_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
            computed_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
            desired_size: Vec2::ZERO,
        }
    }

    pub fn with_width(mut self, w: f32) -> Self {
        self.width_override = Some(w);
        self
    }

    pub fn with_height(mut self, h: f32) -> Self {
        self.height_override = Some(h);
        self
    }

    pub fn with_size(mut self, w: f32, h: f32) -> Self {
        self.width_override = Some(w);
        self.height_override = Some(h);
        self
    }

    pub fn with_min_size(mut self, w: f32, h: f32) -> Self {
        self.min_width = w;
        self.min_height = h;
        self
    }

    pub fn with_max_size(mut self, w: f32, h: f32) -> Self {
        self.max_width = w;
        self.max_height = h;
        self
    }

    pub fn with_padding(mut self, p: SlateMargin) -> Self {
        self.padding = p;
        self
    }

    pub fn set_child_desired_size(&mut self, size: Vec2) {
        self.child_desired_size = size;
    }

    pub fn compute_desired_size(&mut self) -> Vec2 {
        let mut w = self
            .width_override
            .unwrap_or(self.child_desired_size.x);
        let mut h = self
            .height_override
            .unwrap_or(self.child_desired_size.y);
        w = w.max(self.min_width);
        h = h.max(self.min_height);
        if self.max_width > 0.0 {
            w = w.min(self.max_width);
        }
        if self.max_height > 0.0 {
            h = h.min(self.max_height);
        }
        self.desired_size = Vec2::new(
            w + self.padding.horizontal(),
            h + self.padding.vertical(),
        );
        self.desired_size
    }

    pub fn arrange(&mut self, available: Rect) {
        self.computed_rect = available;
        let inner = self.padding.shrink_rect(available);
        let mut w = self
            .width_override
            .unwrap_or(inner.width());
        let mut h = self
            .height_override
            .unwrap_or(inner.height());
        w = w.max(self.min_width).min(inner.width());
        h = h.max(self.min_height).min(inner.height());
        if self.max_width > 0.0 {
            w = w.min(self.max_width);
        }
        if self.max_height > 0.0 {
            h = h.min(self.max_height);
        }
        let cw = self.h_align.resolve_width(w, inner.width());
        let ch = self.v_align.resolve_height(h, inner.height());
        let xo = self.h_align.align_offset(cw, inner.width());
        let yo = self.v_align.align_offset(ch, inner.height());
        self.child_rect = Rect::new(
            Vec2::new(inner.min.x + xo, inner.min.y + yo),
            Vec2::new(
                inner.min.x + xo + cw,
                inner.min.y + yo + ch,
            ),
        );
    }
}

impl Default for SizeBox {
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
    fn test_halign_offset() {
        assert_eq!(HAlign::Start.align_offset(100.0, 200.0), 0.0);
        assert_eq!(HAlign::Center.align_offset(100.0, 200.0), 50.0);
        assert_eq!(HAlign::End.align_offset(100.0, 200.0), 100.0);
    }

    #[test]
    fn test_valign_offset() {
        assert_eq!(VAlign::Start.align_offset(50.0, 100.0), 0.0);
        assert_eq!(VAlign::Center.align_offset(50.0, 100.0), 25.0);
        assert_eq!(VAlign::End.align_offset(50.0, 100.0), 50.0);
    }

    #[test]
    fn test_margin() {
        let m = SlateMargin::all(10.0);
        let r = Rect::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        let s = m.shrink_rect(r);
        assert_eq!(s.min.x, 10.0);
        assert_eq!(s.max.x, 90.0);
    }

    #[test]
    fn test_hbox() {
        let mut hbox = HorizontalBox::new();
        hbox.add_slot(BoxSlot::fixed(100.0));
        hbox.add_slot(BoxSlot::fill(1.0));
        hbox.add_slot(BoxSlot::fixed(50.0));
        hbox.set_child_desired_size(0, Vec2::new(100.0, 30.0));
        hbox.set_child_desired_size(1, Vec2::new(200.0, 30.0));
        hbox.set_child_desired_size(2, Vec2::new(50.0, 30.0));
        hbox.compute_desired_size();
        hbox.arrange_children(Rect::new(
            Vec2::ZERO,
            Vec2::new(500.0, 100.0),
        ));
        let r0 = hbox.child_rect(0).unwrap();
        let r1 = hbox.child_rect(1).unwrap();
        let r2 = hbox.child_rect(2).unwrap();
        assert!((r0.width() - 100.0).abs() < 1.0);
        assert!((r2.width() - 50.0).abs() < 1.0);
        assert!((r1.width() - 350.0).abs() < 1.0);
    }

    #[test]
    fn test_vbox() {
        let mut vbox = VerticalBox::new();
        vbox.add_slot(BoxSlot::auto());
        vbox.add_slot(BoxSlot::fill(1.0));
        vbox.set_child_desired_size(0, Vec2::new(100.0, 50.0));
        vbox.set_child_desired_size(1, Vec2::new(100.0, 100.0));
        vbox.compute_desired_size();
        vbox.arrange_children(Rect::new(
            Vec2::ZERO,
            Vec2::new(200.0, 400.0),
        ));
        let r0 = vbox.child_rect(0).unwrap();
        let r1 = vbox.child_rect(1).unwrap();
        assert!((r0.height() - 50.0).abs() < 1.0);
        assert!((r1.height() - 350.0).abs() < 1.0);
    }

    #[test]
    fn test_canvas() {
        let mut c = Canvas::new();
        c.add_slot(
            CanvasSlot::at_position(10.0, 20.0)
                .with_size(Vec2::new(100.0, 50.0)),
        );
        c.add_slot(
            CanvasSlot::centered(Vec2::ZERO)
                .with_size(Vec2::new(80.0, 40.0)),
        );
        c.arrange_children(Rect::new(
            Vec2::ZERO,
            Vec2::new(400.0, 300.0),
        ));
        let r0 = c.child_rect(0).unwrap();
        assert!((r0.min.x - 10.0).abs() < 0.01);
        let r1 = c.child_rect(1).unwrap();
        assert!((r1.min.x - 160.0).abs() < 0.01);
    }

    #[test]
    fn test_scale_box() {
        let mut sb = ScaleBox::new(StretchMode::Fit);
        sb.set_child_desired_size(Vec2::new(400.0, 200.0));
        sb.arrange(Rect::new(
            Vec2::ZERO,
            Vec2::new(200.0, 200.0),
        ));
        assert!((sb.computed_scale - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_size_box() {
        let mut sb = SizeBox::new()
            .with_min_size(100.0, 50.0)
            .with_max_size(300.0, 200.0);
        sb.set_child_desired_size(Vec2::new(50.0, 30.0));
        let ds = sb.compute_desired_size();
        assert!(ds.x >= 100.0);
        assert!(ds.y >= 50.0);
    }

    #[test]
    fn test_widget_switcher() {
        let mut sw = WidgetSwitcher::new(3);
        sw.set_child_desired_size(0, Vec2::new(100.0, 100.0));
        sw.set_child_desired_size(1, Vec2::new(200.0, 150.0));
        sw.set_child_desired_size(2, Vec2::new(50.0, 50.0));
        let ds = sw.compute_desired_size();
        assert!((ds.x - 200.0).abs() < 0.01);
        assert!((ds.y - 150.0).abs() < 0.01);
    }
}
