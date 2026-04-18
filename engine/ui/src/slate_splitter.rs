//! Resizable splitter for the Genovo Slate UI.
//!
//! Provides:
//! - `Splitter`: horizontal or vertical divider between panels
//! - Drag-to-resize with min-size constraints
//! - Double-click to reset to default ratio
//! - Visual: thin line, cursor changes on hover
//! - Multi-split: more than 2 children

use glam::Vec2;
use genovo_core::Rect;

use crate::core::{MouseButton, UIEvent, UIId};
use crate::render_commands::{
    Border as BorderSpec, Color, CornerRadii, DrawCommand, DrawList,
};
use crate::slate_widgets::EventReply;

// =========================================================================
// SplitDirection
// =========================================================================

/// Direction of the splitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SplitterDirection {
    /// Horizontal split: panels are side-by-side (divider is vertical line).
    Horizontal,
    /// Vertical split: panels are stacked (divider is horizontal line).
    Vertical,
}

impl Default for SplitterDirection {
    fn default() -> Self {
        SplitterDirection::Horizontal
    }
}

// =========================================================================
// SplitterStyle
// =========================================================================

/// Visual style for the splitter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitterStyle {
    /// Divider thickness (pixels).
    pub thickness: f32,
    /// Hit-test area thickness (wider than visual for easy grabbing).
    pub hit_thickness: f32,
    /// Divider colour (normal).
    pub color: Color,
    /// Divider colour (hovered).
    pub hover_color: Color,
    /// Divider colour (dragging).
    pub drag_color: Color,
    /// Background colour (area between panels including the divider).
    pub background: Color,
    /// Whether to show a grip indicator (dots or lines) on the divider.
    pub show_grip: bool,
    /// Grip colour.
    pub grip_color: Color,
    /// Number of grip marks.
    pub grip_count: u32,
}

impl Default for SplitterStyle {
    fn default() -> Self {
        Self {
            thickness: 4.0,
            hit_thickness: 8.0,
            color: Color::from_hex("#333333"),
            hover_color: Color::from_hex("#007ACC"),
            drag_color: Color::from_hex("#007ACC"),
            background: Color::TRANSPARENT,
            show_grip: true,
            grip_color: Color::from_hex("#555555"),
            grip_count: 5,
        }
    }
}

impl SplitterStyle {
    /// Thin style.
    pub fn thin() -> Self {
        Self {
            thickness: 1.0,
            hit_thickness: 6.0,
            show_grip: false,
            ..Default::default()
        }
    }

    /// Wide style with visible grip.
    pub fn wide() -> Self {
        Self {
            thickness: 6.0,
            hit_thickness: 10.0,
            grip_count: 7,
            ..Default::default()
        }
    }
}

// =========================================================================
// PanelConstraints
// =========================================================================

/// Constraints on a panel's size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelConstraints {
    /// Minimum size in pixels.
    pub min_size: f32,
    /// Maximum size in pixels (f32::MAX = no limit).
    pub max_size: f32,
    /// Whether this panel can be collapsed (dragged to zero).
    pub collapsible: bool,
    /// Collapse threshold (if dragged below this, snap to zero).
    pub collapse_threshold: f32,
}

impl Default for PanelConstraints {
    fn default() -> Self {
        Self {
            min_size: 50.0,
            max_size: f32::MAX,
            collapsible: false,
            collapse_threshold: 30.0,
        }
    }
}

// =========================================================================
// SplitPanel
// =========================================================================

/// A panel in a multi-split layout.
#[derive(Debug, Clone)]
pub struct SplitPanel {
    /// Panel identifier.
    pub id: UIId,
    /// Current size ratio (0.0 - 1.0, sum of all panels = 1.0).
    pub ratio: f32,
    /// Default ratio (for double-click reset).
    pub default_ratio: f32,
    /// Constraints.
    pub constraints: PanelConstraints,
    /// Whether this panel is collapsed.
    pub collapsed: bool,
    /// Computed rect (set during layout).
    pub rect: Rect,
}

impl SplitPanel {
    pub fn new(ratio: f32) -> Self {
        Self {
            id: UIId::INVALID,
            ratio,
            default_ratio: ratio,
            constraints: PanelConstraints::default(),
            collapsed: false,
            rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
        }
    }

    pub fn with_constraints(mut self, constraints: PanelConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn with_min_size(mut self, min: f32) -> Self {
        self.constraints.min_size = min;
        self
    }

    pub fn with_max_size(mut self, max: f32) -> Self {
        self.constraints.max_size = max;
        self
    }

    pub fn collapsible(mut self) -> Self {
        self.constraints.collapsible = true;
        self
    }
}

// =========================================================================
// DividerState
// =========================================================================

/// State for a single divider between panels.
#[derive(Debug, Clone, Copy)]
struct DividerState {
    /// Whether the divider is hovered.
    hovered: bool,
    /// Whether the divider is being dragged.
    dragging: bool,
    /// Mouse position at drag start.
    drag_start_pos: f32,
    /// Left/top panel ratio at drag start.
    drag_start_ratio_left: f32,
    /// Right/bottom panel ratio at drag start.
    drag_start_ratio_right: f32,
    /// Last click time (for double-click detection).
    last_click_time: f32,
}

impl Default for DividerState {
    fn default() -> Self {
        Self {
            hovered: false,
            dragging: false,
            drag_start_pos: 0.0,
            drag_start_ratio_left: 0.0,
            drag_start_ratio_right: 0.0,
            last_click_time: -1.0,
        }
    }
}

// =========================================================================
// Splitter
// =========================================================================

/// A resizable splitter that divides space between multiple panels.
///
/// Supports 2 or more panels with draggable dividers between them.
#[derive(Debug, Clone)]
pub struct Splitter {
    /// Unique widget ID.
    pub id: UIId,
    /// Split direction.
    pub direction: SplitterDirection,
    /// Panels.
    pub panels: Vec<SplitPanel>,
    /// Divider states (panels.len() - 1).
    dividers: Vec<DividerState>,
    /// Visual style.
    pub style: SplitterStyle,
    /// Whether the splitter is enabled.
    pub enabled: bool,
    /// Whether the splitter is visible.
    pub visible: bool,
    /// Current time (for double-click detection).
    current_time: f32,
    /// Double-click interval (seconds).
    double_click_interval: f32,
    /// Whether any divider was resized this frame.
    pub resized: bool,
}

impl Splitter {
    /// Create a splitter with two panels at 50/50.
    pub fn two_panel(direction: SplitterDirection) -> Self {
        Self {
            id: UIId::INVALID,
            direction,
            panels: vec![SplitPanel::new(0.5), SplitPanel::new(0.5)],
            dividers: vec![DividerState::default()],
            style: SplitterStyle::default(),
            enabled: true,
            visible: true,
            current_time: 0.0,
            double_click_interval: 0.3,
            resized: false,
        }
    }

    /// Create a splitter with the given ratios.
    pub fn with_ratios(direction: SplitterDirection, ratios: &[f32]) -> Self {
        assert!(ratios.len() >= 2, "Splitter needs at least 2 panels");
        let total: f32 = ratios.iter().sum();
        let panels: Vec<SplitPanel> = ratios
            .iter()
            .map(|&r| SplitPanel::new(r / total))
            .collect();
        let divider_count = panels.len() - 1;
        Self {
            id: UIId::INVALID,
            direction,
            panels,
            dividers: vec![DividerState::default(); divider_count],
            style: SplitterStyle::default(),
            enabled: true,
            visible: true,
            current_time: 0.0,
            double_click_interval: 0.3,
            resized: false,
        }
    }

    /// Create a three-panel splitter.
    pub fn three_panel(direction: SplitterDirection, r1: f32, r2: f32, r3: f32) -> Self {
        Self::with_ratios(direction, &[r1, r2, r3])
    }

    /// Builder: set style.
    pub fn with_style(mut self, style: SplitterStyle) -> Self {
        self.style = style;
        self
    }

    /// Builder: set constraints on a panel.
    pub fn with_panel_constraints(mut self, index: usize, constraints: PanelConstraints) -> Self {
        if index < self.panels.len() {
            self.panels[index].constraints = constraints;
        }
        self
    }

    /// Builder: set min size on all panels.
    pub fn with_min_sizes(mut self, min: f32) -> Self {
        for panel in &mut self.panels {
            panel.constraints.min_size = min;
        }
        self
    }

    /// Get the panel rectangles within the given total rect.
    pub fn panel_rects(&self, rect: Rect) -> Vec<Rect> {
        let total_size = match self.direction {
            SplitterDirection::Horizontal => rect.width(),
            SplitterDirection::Vertical => rect.height(),
        };
        let divider_total = self.dividers.len() as f32 * self.style.thickness;
        let available = (total_size - divider_total).max(0.0);

        let mut rects = Vec::new();
        let mut pos = match self.direction {
            SplitterDirection::Horizontal => rect.min.x,
            SplitterDirection::Vertical => rect.min.y,
        };

        for (i, panel) in self.panels.iter().enumerate() {
            let panel_size = if panel.collapsed {
                0.0
            } else {
                available * panel.ratio
            };

            let panel_rect = match self.direction {
                SplitterDirection::Horizontal => Rect::new(
                    Vec2::new(pos, rect.min.y),
                    Vec2::new(pos + panel_size, rect.max.y),
                ),
                SplitterDirection::Vertical => Rect::new(
                    Vec2::new(rect.min.x, pos),
                    Vec2::new(rect.max.x, pos + panel_size),
                ),
            };

            rects.push(panel_rect);
            pos += panel_size;

            // Add divider space.
            if i < self.panels.len() - 1 {
                pos += self.style.thickness;
            }
        }

        rects
    }

    /// Get divider rectangles.
    fn divider_rects(&self, rect: Rect) -> Vec<Rect> {
        let panel_rects = self.panel_rects(rect);
        let mut rects = Vec::new();

        for i in 0..self.dividers.len() {
            if i < panel_rects.len() {
                let panel_end = match self.direction {
                    SplitterDirection::Horizontal => panel_rects[i].max.x,
                    SplitterDirection::Vertical => panel_rects[i].max.y,
                };

                let divider_rect = match self.direction {
                    SplitterDirection::Horizontal => Rect::new(
                        Vec2::new(panel_end, rect.min.y),
                        Vec2::new(panel_end + self.style.thickness, rect.max.y),
                    ),
                    SplitterDirection::Vertical => Rect::new(
                        Vec2::new(rect.min.x, panel_end),
                        Vec2::new(rect.max.x, panel_end + self.style.thickness),
                    ),
                };
                rects.push(divider_rect);
            }
        }

        rects
    }

    /// Get hit-test rects (wider than visual).
    fn divider_hit_rects(&self, rect: Rect) -> Vec<Rect> {
        let visual_rects = self.divider_rects(rect);
        let expand = (self.style.hit_thickness - self.style.thickness) * 0.5;

        visual_rects
            .iter()
            .map(|r| match self.direction {
                SplitterDirection::Horizontal => Rect::new(
                    Vec2::new(r.min.x - expand, r.min.y),
                    Vec2::new(r.max.x + expand, r.max.y),
                ),
                SplitterDirection::Vertical => Rect::new(
                    Vec2::new(r.min.x, r.min.y - expand),
                    Vec2::new(r.max.x, r.max.y + expand),
                ),
            })
            .collect()
    }

    /// Normalize ratios so they sum to 1.0.
    fn normalize_ratios(&mut self) {
        let total: f32 = self.panels.iter().filter(|p| !p.collapsed).map(|p| p.ratio).sum();
        if total > 0.0 {
            for panel in &mut self.panels {
                if !panel.collapsed {
                    panel.ratio /= total;
                }
            }
        }
    }

    /// Reset all panels to their default ratios.
    pub fn reset_to_default(&mut self) {
        for panel in &mut self.panels {
            panel.ratio = panel.default_ratio;
            panel.collapsed = false;
        }
        self.normalize_ratios();
    }

    /// Set a specific panel ratio (re-normalizes others).
    pub fn set_panel_ratio(&mut self, index: usize, ratio: f32) {
        if index < self.panels.len() {
            let old_ratio = self.panels[index].ratio;
            let delta = ratio - old_ratio;
            self.panels[index].ratio = ratio;

            // Distribute delta among other panels.
            let others: Vec<usize> = (0..self.panels.len())
                .filter(|&i| i != index && !self.panels[i].collapsed)
                .collect();
            if !others.is_empty() {
                let share = delta / others.len() as f32;
                for &i in &others {
                    self.panels[i].ratio -= share;
                    self.panels[i].ratio = self.panels[i].ratio.max(0.01);
                }
            }
            self.normalize_ratios();
        }
    }

    /// Collapse a panel.
    pub fn collapse_panel(&mut self, index: usize) {
        if index < self.panels.len() && self.panels[index].constraints.collapsible {
            self.panels[index].collapsed = true;
            self.normalize_ratios();
        }
    }

    /// Expand a collapsed panel.
    pub fn expand_panel(&mut self, index: usize) {
        if index < self.panels.len() && self.panels[index].collapsed {
            self.panels[index].collapsed = false;
            self.panels[index].ratio = self.panels[index].default_ratio;
            self.normalize_ratios();
        }
    }

    /// Update (advance time for double-click detection).
    pub fn update(&mut self, dt: f32) {
        self.current_time += dt;
        self.resized = false;
    }

    /// Paint the splitter dividers.
    pub fn paint(&self, rect: Rect, draw: &mut DrawList) {
        if !self.visible {
            return;
        }

        // Background.
        if self.style.background.a > 0.0 {
            draw.commands.push(DrawCommand::Rect {
                rect,
                color: self.style.background,
                corner_radii: CornerRadii::ZERO,
                border: BorderSpec::default(),
                shadow: None,
            });
        }

        let divider_rects = self.divider_rects(rect);

        for (i, dr) in divider_rects.iter().enumerate() {
            let divider = &self.dividers[i];
            let color = if divider.dragging {
                self.style.drag_color
            } else if divider.hovered {
                self.style.hover_color
            } else {
                self.style.color
            };

            draw.commands.push(DrawCommand::Rect {
                rect: *dr,
                color,
                corner_radii: CornerRadii::ZERO,
                border: BorderSpec::default(),
                shadow: None,
            });

            // Grip dots/lines.
            if self.style.show_grip {
                self.paint_grip(*dr, color, draw);
            }
        }
    }

    fn paint_grip(&self, divider_rect: Rect, _color: Color, draw: &mut DrawList) {
        let grip_color = self.style.grip_color;
        let count = self.style.grip_count;

        match self.direction {
            SplitterDirection::Horizontal => {
                let cx = (divider_rect.min.x + divider_rect.max.x) * 0.5;
                let total_h = count as f32 * 4.0;
                let start_y = (divider_rect.min.y + divider_rect.max.y - total_h) * 0.5;
                for i in 0..count {
                    let y = start_y + i as f32 * 4.0;
                    draw.commands.push(DrawCommand::Circle {
                        center: Vec2::new(cx, y),
                        radius: 1.0,
                        color: grip_color,
                        border: BorderSpec::default(),
                    });
                }
            }
            SplitterDirection::Vertical => {
                let cy = (divider_rect.min.y + divider_rect.max.y) * 0.5;
                let total_w = count as f32 * 4.0;
                let start_x = (divider_rect.min.x + divider_rect.max.x - total_w) * 0.5;
                for i in 0..count {
                    let x = start_x + i as f32 * 4.0;
                    draw.commands.push(DrawCommand::Circle {
                        center: Vec2::new(x, cy),
                        radius: 1.0,
                        color: grip_color,
                        border: BorderSpec::default(),
                    });
                }
            }
        }
    }

    /// Handle events.
    pub fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled || !self.visible {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } | UIEvent::DragMove { position, .. } => {
                let pos = *position;

                // Handle dragging -- find the active divider index first.
                let dragging_idx = self.dividers.iter().position(|d| d.dragging);
                if let Some(i) = dragging_idx {
                    let mouse_pos = match self.direction {
                        SplitterDirection::Horizontal => pos.x,
                        SplitterDirection::Vertical => pos.y,
                    };
                    let delta_mouse = mouse_pos - self.dividers[i].drag_start_pos;

                    let total_size = match self.direction {
                        SplitterDirection::Horizontal => rect.width(),
                        SplitterDirection::Vertical => rect.height(),
                    };
                    let divider_count = self.dividers.len();
                    let divider_total = divider_count as f32 * self.style.thickness;
                    let available = (total_size - divider_total).max(1.0);
                    let delta_ratio = delta_mouse / available;

                    let new_left = (self.dividers[i].drag_start_ratio_left + delta_ratio).max(0.01);
                    let new_right = (self.dividers[i].drag_start_ratio_right - delta_ratio).max(0.01);

                    // Apply constraints.
                    let left_min = self.panels[i].constraints.min_size / available;
                    let right_min = self.panels[i + 1].constraints.min_size / available;
                    let left_max = self.panels[i].constraints.max_size / available;
                    let right_max = self.panels[i + 1].constraints.max_size / available;

                    let clamped_left = new_left.clamp(left_min, left_max.min(1.0));
                    let clamped_right = new_right.clamp(right_min, right_max.min(1.0));

                    // Check collapsible.
                    if self.panels[i].constraints.collapsible
                        && new_left * available < self.panels[i].constraints.collapse_threshold
                    {
                        self.panels[i].collapsed = true;
                        self.panels[i].ratio = 0.01;
                        self.panels[i + 1].ratio = self.dividers[i].drag_start_ratio_left
                            + self.dividers[i].drag_start_ratio_right
                            - 0.01;
                    } else if self.panels[i + 1].constraints.collapsible
                        && new_right * available
                            < self.panels[i + 1].constraints.collapse_threshold
                    {
                        self.panels[i + 1].collapsed = true;
                        self.panels[i + 1].ratio = 0.01;
                        self.panels[i].ratio = self.dividers[i].drag_start_ratio_left
                            + self.dividers[i].drag_start_ratio_right
                            - 0.01;
                    } else {
                        self.panels[i].collapsed = false;
                        self.panels[i + 1].collapsed = false;
                        self.panels[i].ratio = clamped_left;
                        self.panels[i + 1].ratio = clamped_right;
                    }

                    self.normalize_ratios();
                    self.resized = true;
                    return EventReply::Handled;
                }

                // Update hover states.
                let hit_rects = self.divider_hit_rects(rect);
                let mut any_hovered = false;
                for (i, divider) in self.dividers.iter_mut().enumerate() {
                    divider.hovered = i < hit_rects.len() && hit_rects[i].contains(pos);
                    if divider.hovered {
                        any_hovered = true;
                    }
                }

                if any_hovered {
                    return EventReply::Handled;
                }
            }

            UIEvent::Click { position, button, .. } => {
                if *button != MouseButton::Left {
                    return EventReply::Unhandled;
                }

                let pos = *position;
                let hit_rects = self.divider_hit_rects(rect);

                for (i, hr) in hit_rects.iter().enumerate() {
                    if hr.contains(pos) && i < self.dividers.len() {
                        // Double-click detection.
                        let time_since_last = self.current_time - self.dividers[i].last_click_time;
                        if time_since_last < self.double_click_interval
                            && time_since_last > 0.0
                        {
                            // Reset to default.
                            self.panels[i].ratio = self.panels[i].default_ratio;
                            self.panels[i + 1].ratio = self.panels[i + 1].default_ratio;
                            self.panels[i].collapsed = false;
                            self.panels[i + 1].collapsed = false;
                            self.normalize_ratios();
                            self.resized = true;
                            self.dividers[i].last_click_time = -1.0;
                            return EventReply::Handled;
                        }

                        self.dividers[i].last_click_time = self.current_time;
                        self.dividers[i].dragging = true;
                        self.dividers[i].drag_start_pos = match self.direction {
                            SplitterDirection::Horizontal => pos.x,
                            SplitterDirection::Vertical => pos.y,
                        };
                        self.dividers[i].drag_start_ratio_left = self.panels[i].ratio;
                        self.dividers[i].drag_start_ratio_right = self.panels[i + 1].ratio;
                        return EventReply::CaptureMouse;
                    }
                }
            }

            UIEvent::MouseUp { .. } | UIEvent::DragEnd { .. } => {
                let is_left = match event {
                    UIEvent::MouseUp { button, .. } => *button == MouseButton::Left,
                    _ => true,
                };
                if is_left {
                    for divider in &mut self.dividers {
                        if divider.dragging {
                            divider.dragging = false;
                            return EventReply::ReleaseMouse;
                        }
                    }
                }
            }

            _ => {}
        }

        EventReply::Unhandled
    }

    /// Whether any divider is being dragged.
    pub fn is_dragging(&self) -> bool {
        self.dividers.iter().any(|d| d.dragging)
    }

    /// Whether any divider is hovered.
    pub fn is_hovered(&self) -> bool {
        self.dividers.iter().any(|d| d.hovered)
    }

    /// Get the cursor type that should be shown.
    pub fn cursor_type(&self) -> Option<SplitterCursor> {
        if self.is_dragging() || self.is_hovered() {
            Some(match self.direction {
                SplitterDirection::Horizontal => SplitterCursor::ResizeHorizontal,
                SplitterDirection::Vertical => SplitterCursor::ResizeVertical,
            })
        } else {
            None
        }
    }
}

/// Cursor type hint for the splitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitterCursor {
    ResizeHorizontal,
    ResizeVertical,
}

impl Default for Splitter {
    fn default() -> Self {
        Self::two_panel(SplitterDirection::Horizontal)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_panel_rects() {
        let s = Splitter::two_panel(SplitterDirection::Horizontal);
        let rect = Rect::new(Vec2::ZERO, Vec2::new(800.0, 600.0));
        let rects = s.panel_rects(rect);
        assert_eq!(rects.len(), 2);
        // Each panel ~50% of (800 - divider thickness).
        assert!((rects[0].width() - rects[1].width()).abs() < 1.0);
    }

    #[test]
    fn test_three_panel() {
        let s = Splitter::three_panel(SplitterDirection::Vertical, 1.0, 2.0, 1.0);
        let rect = Rect::new(Vec2::ZERO, Vec2::new(800.0, 600.0));
        let rects = s.panel_rects(rect);
        assert_eq!(rects.len(), 3);
    }

    #[test]
    fn test_normalize_ratios() {
        let mut s = Splitter::with_ratios(SplitterDirection::Horizontal, &[1.0, 3.0]);
        s.normalize_ratios();
        let total: f32 = s.panels.iter().map(|p| p.ratio).sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_reset_to_default() {
        let mut s = Splitter::two_panel(SplitterDirection::Horizontal);
        s.panels[0].ratio = 0.8;
        s.panels[1].ratio = 0.2;
        s.reset_to_default();
        assert!((s.panels[0].ratio - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_divider_rects() {
        let s = Splitter::two_panel(SplitterDirection::Horizontal);
        let rect = Rect::new(Vec2::ZERO, Vec2::new(800.0, 600.0));
        let dividers = s.divider_rects(rect);
        assert_eq!(dividers.len(), 1);
        assert!(dividers[0].width() > 0.0);
    }
}
