//! Advanced / complex Slate widgets for the Genovo engine editor.
//!
//! These widgets build on the primitive types in [`slate_widgets`] and provide
//! higher-level functionality such as virtualised lists, tree views, colour
//! pickers, curve editors, and node-graph canvases.

use std::collections::HashMap;

use glam::Vec2;
use genovo_core::Rect;

use crate::core::{KeyCode, KeyModifiers, MouseButton, Padding, UIEvent, UIId};
use crate::render_commands::{
    Border as BorderSpec, Color, CornerRadii, DrawCommand, DrawList, GradientStop, Gradient,
    ImageScaleMode, Shadow, TextAlign, TextVerticalAlign, TextureId,
};
use crate::slate_widgets::{
    Brush, EventReply, HAlign, Orientation, SlateWidget, TextWrapping, VAlign,
};

// =========================================================================
// Selection helper
// =========================================================================

/// Selection model for list/tree/tile widgets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionMode {
    /// No selection.
    None,
    /// Single item selection.
    Single,
    /// Multiple items via Ctrl-click / Shift-click.
    Multi,
}

impl Default for SelectionMode {
    fn default() -> Self {
        SelectionMode::Single
    }
}

/// Tracks selected indices.
#[derive(Debug, Clone)]
pub struct SelectionState {
    pub selected: Vec<usize>,
    pub anchor: Option<usize>,
    pub mode: SelectionMode,
    pub selection_changed: bool,
}

impl SelectionState {
    pub fn new(mode: SelectionMode) -> Self {
        Self {
            selected: Vec::new(),
            anchor: None,
            mode,
            selection_changed: false,
        }
    }

    pub fn is_selected(&self, index: usize) -> bool {
        self.selected.contains(&index)
    }

    pub fn select(&mut self, index: usize, ctrl: bool, shift: bool) {
        match self.mode {
            SelectionMode::None => {}
            SelectionMode::Single => {
                self.selected.clear();
                self.selected.push(index);
                self.anchor = Some(index);
                self.selection_changed = true;
            }
            SelectionMode::Multi => {
                if shift {
                    if let Some(anchor) = self.anchor {
                        let start = anchor.min(index);
                        let end = anchor.max(index);
                        if !ctrl {
                            self.selected.clear();
                        }
                        for i in start..=end {
                            if !self.selected.contains(&i) {
                                self.selected.push(i);
                            }
                        }
                    } else {
                        self.selected.clear();
                        self.selected.push(index);
                        self.anchor = Some(index);
                    }
                } else if ctrl {
                    if let Some(pos) = self.selected.iter().position(|&i| i == index) {
                        self.selected.remove(pos);
                    } else {
                        self.selected.push(index);
                    }
                    self.anchor = Some(index);
                } else {
                    self.selected.clear();
                    self.selected.push(index);
                    self.anchor = Some(index);
                }
                self.selection_changed = true;
            }
        }
    }

    pub fn clear(&mut self) {
        self.selected.clear();
        self.anchor = None;
        self.selection_changed = true;
    }

    pub fn select_all(&mut self, count: usize) {
        self.selected.clear();
        for i in 0..count {
            self.selected.push(i);
        }
        self.selection_changed = true;
    }

    pub fn take_selection_changed(&mut self) -> bool {
        let c = self.selection_changed;
        self.selection_changed = false;
        c
    }
}

// =========================================================================
// 1. ListView<T>
// =========================================================================

/// Item data for a list view row.
#[derive(Debug, Clone)]
pub struct ListViewItem {
    pub label: String,
    pub icon: Option<TextureId>,
    pub data: u64,
    pub enabled: bool,
}

impl ListViewItem {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            icon: None,
            data: 0,
            enabled: true,
        }
    }

    pub fn with_icon(mut self, icon: TextureId) -> Self {
        self.icon = Some(icon);
        self
    }

    pub fn with_data(mut self, data: u64) -> Self {
        self.data = data;
        self
    }
}

/// Virtualized scrolling list -- only creates widgets for visible rows.
/// Handles 100K+ items efficiently.
#[derive(Debug, Clone)]
pub struct ListView {
    pub id: UIId,
    pub items: Vec<ListViewItem>,
    pub selection: SelectionState,
    pub row_height: f32,
    pub font_size: f32,
    pub font_id: u32,
    pub text_color: Color,
    pub background_color: Color,
    pub row_bg: Color,
    pub alt_row_bg: Color,
    pub selected_bg: Color,
    pub hovered_bg: Color,
    pub border_color: Color,
    pub corner_radii: CornerRadii,
    pub scroll_offset: f32,
    pub hovered_index: Option<usize>,
    pub desired_size: Vec2,
    pub show_scrollbar: bool,
    pub scrollbar_width: f32,
    pub scrollbar_color: Color,
    pub scrollbar_track_color: Color,
    pub scrollbar_hovered: bool,
    pub scrollbar_dragging: bool,
    pub scrollbar_drag_start_offset: f32,
    pub scrollbar_drag_start_y: f32,
    pub visible: bool,
    pub enabled: bool,
    pub icon_size: f32,
    pub padding: Padding,
}

impl ListView {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            items: Vec::new(),
            selection: SelectionState::new(SelectionMode::Single),
            row_height: 24.0,
            font_size: 13.0,
            font_id: 0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#1E1E1E"),
            row_bg: Color::from_hex("#1E1E1E"),
            alt_row_bg: Color::from_hex("#222222"),
            selected_bg: Color::from_hex("#2266CC"),
            hovered_bg: Color::from_hex("#333333"),
            border_color: Color::from_hex("#444444"),
            corner_radii: CornerRadii::all(3.0),
            scroll_offset: 0.0,
            hovered_index: None,
            desired_size: Vec2::new(250.0, 300.0),
            show_scrollbar: true,
            scrollbar_width: 10.0,
            scrollbar_color: Color::from_hex("#666666"),
            scrollbar_track_color: Color::from_hex("#2A2A2A"),
            scrollbar_hovered: false,
            scrollbar_dragging: false,
            scrollbar_drag_start_offset: 0.0,
            scrollbar_drag_start_y: 0.0,
            visible: true,
            enabled: true,
            icon_size: 16.0,
            padding: Padding::new(2.0, 2.0, 2.0, 2.0),
        }
    }

    pub fn with_items(mut self, items: Vec<ListViewItem>) -> Self {
        self.items = items;
        self
    }

    pub fn with_selection_mode(mut self, mode: SelectionMode) -> Self {
        self.selection.mode = mode;
        self
    }

    pub fn total_content_height(&self) -> f32 {
        self.items.len() as f32 * self.row_height
    }

    pub fn max_scroll(&self, view_height: f32) -> f32 {
        (self.total_content_height() - view_height).max(0.0)
    }

    pub fn scroll_to_item(&mut self, index: usize, view_height: f32) {
        let item_top = index as f32 * self.row_height;
        let item_bottom = item_top + self.row_height;
        if item_top < self.scroll_offset {
            self.scroll_offset = item_top;
        } else if item_bottom > self.scroll_offset + view_height {
            self.scroll_offset = item_bottom - view_height;
        }
    }

    fn content_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.min.x + self.padding.left, rect.min.y + self.padding.top),
            Vec2::new(
                rect.max.x - self.padding.right - if self.show_scrollbar { self.scrollbar_width } else { 0.0 },
                rect.max.y - self.padding.bottom,
            ),
        )
    }

    fn scrollbar_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.max.x - self.scrollbar_width, rect.min.y),
            Vec2::new(rect.max.x, rect.max.y),
        )
    }

    fn scrollbar_thumb_rect(&self, rect: Rect) -> Rect {
        let sb = self.scrollbar_rect(rect);
        let view_h = rect.height();
        let total_h = self.total_content_height().max(1.0);
        let thumb_h = (view_h / total_h * sb.height()).clamp(20.0, sb.height());
        let scroll_range = sb.height() - thumb_h;
        let max_scroll = self.max_scroll(view_h);
        let thumb_y = if max_scroll > 0.0 {
            sb.min.y + (self.scroll_offset / max_scroll) * scroll_range
        } else {
            sb.min.y
        };
        Rect::new(
            Vec2::new(sb.min.x, thumb_y),
            Vec2::new(sb.max.x, thumb_y + thumb_h),
        )
    }

    fn index_at_y(&self, y: f32, rect: Rect) -> Option<usize> {
        let content = self.content_rect(rect);
        let local_y = y - content.min.y + self.scroll_offset;
        if local_y < 0.0 {
            return None;
        }
        let idx = (local_y / self.row_height) as usize;
        if idx < self.items.len() {
            Some(idx)
        } else {
            None
        }
    }
}

impl SlateWidget for ListView {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        self.desired_size
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        // Background.
        draw_list.draw_rounded_rect(
            rect,
            self.background_color,
            self.corner_radii,
            BorderSpec::new(self.border_color, 1.0),
        );

        let content = self.content_rect(rect);
        draw_list.push_clip(content);

        let first_visible = (self.scroll_offset / self.row_height).floor() as usize;
        let visible_count =
            ((content.height() / self.row_height).ceil() as usize + 1).min(
                self.items.len().saturating_sub(first_visible),
            );

        for i in 0..visible_count {
            let item_idx = first_visible + i;
            if item_idx >= self.items.len() {
                break;
            }

            let y = content.min.y + i as f32 * self.row_height
                - (self.scroll_offset % self.row_height);
            let row_rect = Rect::new(
                Vec2::new(content.min.x, y),
                Vec2::new(content.max.x, y + self.row_height),
            );

            // Row background.
            let bg = if self.selection.is_selected(item_idx) {
                self.selected_bg
            } else if self.hovered_index == Some(item_idx) {
                self.hovered_bg
            } else if item_idx % 2 == 1 {
                self.alt_row_bg
            } else {
                self.row_bg
            };
            draw_list.draw_rect(row_rect, bg);

            let item = &self.items[item_idx];
            let mut text_x = content.min.x + 4.0;

            // Icon.
            if let Some(icon) = item.icon {
                let icon_y = y + (self.row_height - self.icon_size) * 0.5;
                draw_list.push(DrawCommand::Image {
                    rect: Rect::new(
                        Vec2::new(text_x, icon_y),
                        Vec2::new(text_x + self.icon_size, icon_y + self.icon_size),
                    ),
                    texture: icon,
                    tint: Color::WHITE,
                    corner_radii: CornerRadii::ZERO,
                    scale_mode: ImageScaleMode::Fit,
                    uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
                });
                text_x += self.icon_size + 4.0;
            }

            // Text.
            let ty = y + (self.row_height - self.font_size) * 0.5;
            let tc = if item.enabled {
                self.text_color
            } else {
                Color::from_rgba8(128, 128, 128, 180)
            };
            draw_list.push(DrawCommand::Text {
                text: item.label.clone(),
                position: Vec2::new(text_x, ty),
                font_size: self.font_size,
                color: tc,
                font_id: self.font_id,
                max_width: Some(content.max.x - text_x - 4.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }

        draw_list.pop_clip();

        // Scrollbar.
        if self.show_scrollbar && self.total_content_height() > content.height() {
            let sb = self.scrollbar_rect(rect);
            draw_list.draw_rect(sb, self.scrollbar_track_color);
            let thumb = self.scrollbar_thumb_rect(rect);
            let thumb_color = if self.scrollbar_dragging || self.scrollbar_hovered {
                self.scrollbar_color.lighten(0.2)
            } else {
                self.scrollbar_color
            };
            draw_list.draw_rounded_rect(
                thumb,
                thumb_color,
                CornerRadii::all(self.scrollbar_width * 0.5),
                BorderSpec::default(),
            );
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                if rect.contains(*position) {
                    self.hovered_index = self.index_at_y(position.y, rect);
                    let sb = self.scrollbar_thumb_rect(rect);
                    self.scrollbar_hovered = sb.contains(*position);
                    EventReply::Handled
                } else {
                    self.hovered_index = None;
                    self.scrollbar_hovered = false;
                    EventReply::Unhandled
                }
            }
            UIEvent::HoverEnd => {
                self.hovered_index = None;
                self.scrollbar_hovered = false;
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                modifiers,
            } => {
                if !rect.contains(*position) {
                    return EventReply::Unhandled;
                }

                // Scrollbar click.
                let sb_rect = self.scrollbar_rect(rect);
                if self.show_scrollbar && sb_rect.contains(*position) {
                    let thumb = self.scrollbar_thumb_rect(rect);
                    if thumb.contains(*position) {
                        self.scrollbar_dragging = true;
                        self.scrollbar_drag_start_offset = self.scroll_offset;
                        self.scrollbar_drag_start_y = position.y;
                        return EventReply::Handled.then(EventReply::CaptureMouse);
                    } else {
                        // Click on track -- jump.
                        let view_h = rect.height();
                        let total_h = self.total_content_height();
                        let ratio = (position.y - sb_rect.min.y) / sb_rect.height();
                        self.scroll_offset = (ratio * total_h - view_h * 0.5)
                            .clamp(0.0, self.max_scroll(view_h));
                        return EventReply::Handled;
                    }
                }

                // Item click.
                if let Some(idx) = self.index_at_y(position.y, rect) {
                    self.selection.select(idx, modifiers.ctrl, modifiers.shift);
                    return EventReply::Handled;
                }

                EventReply::Handled
            }
            UIEvent::DragMove { position, delta } => {
                if self.scrollbar_dragging {
                    let sb = self.scrollbar_rect(rect);
                    let view_h = rect.height();
                    let total_h = self.total_content_height();
                    let thumb_h = (view_h / total_h * sb.height()).clamp(20.0, sb.height());
                    let scroll_range = sb.height() - thumb_h;
                    let dy = position.y - self.scrollbar_drag_start_y;
                    let max_sc = self.max_scroll(view_h);
                    if scroll_range > 0.0 {
                        self.scroll_offset = (self.scrollbar_drag_start_offset
                            + dy / scroll_range * max_sc)
                            .clamp(0.0, max_sc);
                    }
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::MouseUp {
                button: MouseButton::Left,
                ..
            }
            | UIEvent::DragEnd { .. } => {
                if self.scrollbar_dragging {
                    self.scrollbar_dragging = false;
                    return EventReply::Handled.then(EventReply::ReleaseMouse);
                }
                EventReply::Unhandled
            }
            UIEvent::Scroll { delta, .. } => {
                if rect.contains(Vec2::ZERO) || true {
                    // Always handle scroll if the widget is receiving events.
                    let view_h = rect.height();
                    self.scroll_offset = (self.scroll_offset - delta.y * self.row_height * 3.0)
                        .clamp(0.0, self.max_scroll(view_h));
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::KeyInput {
                key,
                pressed: true,
                modifiers,
            } => {
                match key {
                    KeyCode::ArrowUp => {
                        if let Some(&current) = self.selection.selected.last() {
                            if current > 0 {
                                self.selection
                                    .select(current - 1, modifiers.ctrl, modifiers.shift);
                                let view_h = self.content_rect(rect).height();
                                self.scroll_to_item(current - 1, view_h);
                            }
                        }
                        EventReply::Handled
                    }
                    KeyCode::ArrowDown => {
                        if let Some(&current) = self.selection.selected.last() {
                            if current + 1 < self.items.len() {
                                self.selection
                                    .select(current + 1, modifiers.ctrl, modifiers.shift);
                                let view_h = self.content_rect(rect).height();
                                self.scroll_to_item(current + 1, view_h);
                            }
                        } else if !self.items.is_empty() {
                            self.selection.select(0, false, false);
                        }
                        EventReply::Handled
                    }
                    KeyCode::Home => {
                        if !self.items.is_empty() {
                            self.selection.select(0, false, false);
                            self.scroll_offset = 0.0;
                        }
                        EventReply::Handled
                    }
                    KeyCode::End => {
                        if !self.items.is_empty() {
                            let last = self.items.len() - 1;
                            self.selection.select(last, false, false);
                            let view_h = self.content_rect(rect).height();
                            self.scroll_to_item(last, view_h);
                        }
                        EventReply::Handled
                    }
                    KeyCode::PageUp => {
                        let page = (rect.height() / self.row_height) as usize;
                        if let Some(&current) = self.selection.selected.last() {
                            let target = current.saturating_sub(page);
                            self.selection.select(target, false, false);
                            let view_h = self.content_rect(rect).height();
                            self.scroll_to_item(target, view_h);
                        }
                        EventReply::Handled
                    }
                    KeyCode::PageDown => {
                        let page = (rect.height() / self.row_height) as usize;
                        if let Some(&current) = self.selection.selected.last() {
                            let target = (current + page).min(self.items.len().saturating_sub(1));
                            self.selection.select(target, false, false);
                            let view_h = self.content_rect(rect).height();
                            self.scroll_to_item(target, view_h);
                        }
                        EventReply::Handled
                    }
                    KeyCode::A if modifiers.ctrl => {
                        self.selection.select_all(self.items.len());
                        EventReply::Handled
                    }
                    _ => EventReply::Unhandled,
                }
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 2. TreeView
// =========================================================================

/// A tree node in the tree view.
#[derive(Debug, Clone)]
pub struct TreeViewNode {
    pub label: String,
    pub icon: Option<TextureId>,
    pub data: u64,
    pub expanded: bool,
    pub children: Vec<TreeViewNode>,
    pub depth: usize,
    pub enabled: bool,
    pub visible: bool,
}

impl TreeViewNode {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            icon: None,
            data: 0,
            expanded: false,
            children: Vec::new(),
            depth: 0,
            enabled: true,
            visible: true,
        }
    }

    pub fn with_children(mut self, children: Vec<TreeViewNode>) -> Self {
        self.children = children;
        self
    }

    pub fn add_child(&mut self, child: TreeViewNode) {
        self.children.push(child);
    }

    /// Flatten the tree into a list of visible rows with depth information.
    pub fn flatten_visible(&self) -> Vec<FlattenedTreeRow> {
        let mut rows = Vec::new();
        self.flatten_recursive(&mut rows, 0);
        rows
    }

    fn flatten_recursive(&self, rows: &mut Vec<FlattenedTreeRow>, depth: usize) {
        rows.push(FlattenedTreeRow {
            label: self.label.clone(),
            icon: self.icon,
            data: self.data,
            depth,
            has_children: !self.children.is_empty(),
            expanded: self.expanded,
            enabled: self.enabled,
        });
        if self.expanded {
            for child in &self.children {
                child.flatten_recursive(rows, depth + 1);
            }
        }
    }

    /// Find a node by data value and toggle its expanded state. Returns true if found.
    pub fn toggle_expand(&mut self, data: u64) -> bool {
        if self.data == data {
            self.expanded = !self.expanded;
            return true;
        }
        for child in &mut self.children {
            if child.toggle_expand(data) {
                return true;
            }
        }
        false
    }

    /// Set expanded state for all nodes recursively.
    pub fn set_expanded_recursive(&mut self, expanded: bool) {
        self.expanded = expanded;
        for child in &mut self.children {
            child.set_expanded_recursive(expanded);
        }
    }
}

/// A flattened row from the tree for rendering.
#[derive(Debug, Clone)]
pub struct FlattenedTreeRow {
    pub label: String,
    pub icon: Option<TextureId>,
    pub data: u64,
    pub depth: usize,
    pub has_children: bool,
    pub expanded: bool,
    pub enabled: bool,
}

/// Virtualized hierarchical tree with expand/collapse.
#[derive(Debug, Clone)]
pub struct TreeView {
    pub id: UIId,
    pub roots: Vec<TreeViewNode>,
    pub selection: SelectionState,
    pub row_height: f32,
    pub indent_width: f32,
    pub font_size: f32,
    pub font_id: u32,
    pub text_color: Color,
    pub background_color: Color,
    pub selected_bg: Color,
    pub hovered_bg: Color,
    pub border_color: Color,
    pub expand_icon_color: Color,
    pub corner_radii: CornerRadii,
    pub scroll_offset: f32,
    pub hovered_index: Option<usize>,
    pub desired_size: Vec2,
    pub show_scrollbar: bool,
    pub scrollbar_width: f32,
    pub scrollbar_color: Color,
    pub scrollbar_dragging: bool,
    pub scrollbar_drag_start_offset: f32,
    pub scrollbar_drag_start_y: f32,
    pub visible: bool,
    pub enabled: bool,
    pub icon_size: f32,
    /// Cached flattened rows (rebuild when tree changes).
    flattened_cache: Vec<FlattenedTreeRow>,
    cache_dirty: bool,
}

impl TreeView {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            roots: Vec::new(),
            selection: SelectionState::new(SelectionMode::Single),
            row_height: 22.0,
            indent_width: 18.0,
            font_size: 13.0,
            font_id: 0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#1E1E1E"),
            selected_bg: Color::from_hex("#2266CC"),
            hovered_bg: Color::from_hex("#333333"),
            border_color: Color::from_hex("#444444"),
            expand_icon_color: Color::from_hex("#CCCCCC"),
            corner_radii: CornerRadii::all(3.0),
            scroll_offset: 0.0,
            hovered_index: None,
            desired_size: Vec2::new(250.0, 300.0),
            show_scrollbar: true,
            scrollbar_width: 10.0,
            scrollbar_color: Color::from_hex("#666666"),
            scrollbar_dragging: false,
            scrollbar_drag_start_offset: 0.0,
            scrollbar_drag_start_y: 0.0,
            visible: true,
            enabled: true,
            icon_size: 16.0,
            flattened_cache: Vec::new(),
            cache_dirty: true,
        }
    }

    pub fn with_roots(mut self, roots: Vec<TreeViewNode>) -> Self {
        self.roots = roots;
        self.cache_dirty = true;
        self
    }

    pub fn invalidate_cache(&mut self) {
        self.cache_dirty = true;
    }

    fn rebuild_cache(&mut self) {
        self.flattened_cache.clear();
        for root in &self.roots {
            let rows = root.flatten_visible();
            self.flattened_cache.extend(rows);
        }
        self.cache_dirty = false;
    }

    fn flattened(&mut self) -> &[FlattenedTreeRow] {
        if self.cache_dirty {
            self.rebuild_cache();
        }
        &self.flattened_cache
    }

    fn total_content_height(&mut self) -> f32 {
        if self.cache_dirty {
            self.rebuild_cache();
        }
        self.flattened_cache.len() as f32 * self.row_height
    }

    fn max_scroll(&mut self, view_height: f32) -> f32 {
        (self.total_content_height() - view_height).max(0.0)
    }

    fn toggle_node(&mut self, data: u64) {
        for root in &mut self.roots {
            if root.toggle_expand(data) {
                self.cache_dirty = true;
                return;
            }
        }
    }
}

impl SlateWidget for TreeView {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        self.desired_size
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        draw_list.draw_rounded_rect(
            rect,
            self.background_color,
            self.corner_radii,
            BorderSpec::new(self.border_color, 1.0),
        );

        let content = Rect::new(
            Vec2::new(rect.min.x + 2.0, rect.min.y + 2.0),
            Vec2::new(
                rect.max.x - 2.0 - if self.show_scrollbar { self.scrollbar_width } else { 0.0 },
                rect.max.y - 2.0,
            ),
        );
        draw_list.push_clip(content);

        let rows = &self.flattened_cache;
        let first_visible = (self.scroll_offset / self.row_height).floor() as usize;
        let visible_count =
            ((content.height() / self.row_height).ceil() as usize + 1)
                .min(rows.len().saturating_sub(first_visible));

        for i in 0..visible_count {
            let row_idx = first_visible + i;
            if row_idx >= rows.len() {
                break;
            }
            let row = &rows[row_idx];
            let y = content.min.y + i as f32 * self.row_height
                - (self.scroll_offset % self.row_height);
            let row_rect = Rect::new(
                Vec2::new(content.min.x, y),
                Vec2::new(content.max.x, y + self.row_height),
            );

            // Row background.
            let bg = if self.selection.is_selected(row_idx) {
                self.selected_bg
            } else if self.hovered_index == Some(row_idx) {
                self.hovered_bg
            } else {
                Color::TRANSPARENT
            };
            if bg.a > 0.0 {
                draw_list.draw_rect(row_rect, bg);
            }

            let indent = row.depth as f32 * self.indent_width;
            let mut text_x = content.min.x + indent + 4.0;

            // Expand/collapse arrow.
            if row.has_children {
                let arrow_x = text_x + 4.0;
                let arrow_y = y + self.row_height * 0.5;
                let s = 4.0;
                if row.expanded {
                    draw_list.push(DrawCommand::Triangle {
                        p0: Vec2::new(arrow_x - s, arrow_y - s * 0.5),
                        p1: Vec2::new(arrow_x + s, arrow_y - s * 0.5),
                        p2: Vec2::new(arrow_x, arrow_y + s * 0.5),
                        color: self.expand_icon_color,
                    });
                } else {
                    draw_list.push(DrawCommand::Triangle {
                        p0: Vec2::new(arrow_x - s * 0.5, arrow_y - s),
                        p1: Vec2::new(arrow_x + s * 0.5, arrow_y),
                        p2: Vec2::new(arrow_x - s * 0.5, arrow_y + s),
                        color: self.expand_icon_color,
                    });
                }
                text_x += 14.0;
            } else {
                text_x += 14.0;
            }

            // Icon.
            if let Some(icon) = row.icon {
                let icon_y = y + (self.row_height - self.icon_size) * 0.5;
                draw_list.push(DrawCommand::Image {
                    rect: Rect::new(
                        Vec2::new(text_x, icon_y),
                        Vec2::new(text_x + self.icon_size, icon_y + self.icon_size),
                    ),
                    texture: icon,
                    tint: Color::WHITE,
                    corner_radii: CornerRadii::ZERO,
                    scale_mode: ImageScaleMode::Fit,
                    uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
                });
                text_x += self.icon_size + 4.0;
            }

            // Label.
            let ty = y + (self.row_height - self.font_size) * 0.5;
            draw_list.push(DrawCommand::Text {
                text: row.label.clone(),
                position: Vec2::new(text_x, ty),
                font_size: self.font_size,
                color: if row.enabled {
                    self.text_color
                } else {
                    Color::from_rgba8(128, 128, 128, 180)
                },
                font_id: self.font_id,
                max_width: Some(content.max.x - text_x - 4.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }

        draw_list.pop_clip();

        // Scrollbar.
        if self.show_scrollbar && !rows.is_empty() {
            let total_h = rows.len() as f32 * self.row_height;
            if total_h > rect.height() {
                let sb = Rect::new(
                    Vec2::new(rect.max.x - self.scrollbar_width, rect.min.y),
                    Vec2::new(rect.max.x, rect.max.y),
                );
                draw_list.draw_rect(sb, Color::from_hex("#2A2A2A"));
                let view_h = rect.height();
                let thumb_h = (view_h / total_h * sb.height()).clamp(20.0, sb.height());
                let max_sc = (total_h - view_h).max(0.0);
                let scroll_range = sb.height() - thumb_h;
                let thumb_y = if max_sc > 0.0 {
                    sb.min.y + (self.scroll_offset / max_sc) * scroll_range
                } else {
                    sb.min.y
                };
                draw_list.draw_rounded_rect(
                    Rect::new(
                        Vec2::new(sb.min.x, thumb_y),
                        Vec2::new(sb.max.x, thumb_y + thumb_h),
                    ),
                    self.scrollbar_color,
                    CornerRadii::all(self.scrollbar_width * 0.5),
                    BorderSpec::default(),
                );
            }
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                if rect.contains(*position) {
                    let content_y = position.y - rect.min.y - 2.0 + self.scroll_offset;
                    let idx = (content_y / self.row_height).floor() as usize;
                    if self.cache_dirty {
                        self.rebuild_cache();
                    }
                    self.hovered_index = if idx < self.flattened_cache.len() {
                        Some(idx)
                    } else {
                        None
                    };
                    EventReply::Handled
                } else {
                    self.hovered_index = None;
                    EventReply::Unhandled
                }
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                modifiers,
            } => {
                if !rect.contains(*position) {
                    return EventReply::Unhandled;
                }

                if self.cache_dirty {
                    self.rebuild_cache();
                }

                let content_y = position.y - rect.min.y - 2.0 + self.scroll_offset;
                let idx = (content_y / self.row_height).floor() as usize;
                if idx < self.flattened_cache.len() {
                    let row = &self.flattened_cache[idx];
                    let indent = row.depth as f32 * self.indent_width;
                    let arrow_region_x = rect.min.x + 2.0 + indent;

                    // Check if click is on the expand arrow.
                    if row.has_children
                        && position.x >= arrow_region_x
                        && position.x < arrow_region_x + 18.0
                    {
                        let data = row.data;
                        self.toggle_node(data);
                        return EventReply::Handled;
                    }

                    self.selection.select(idx, modifiers.ctrl, modifiers.shift);
                    return EventReply::Handled;
                }
                EventReply::Handled
            }
            UIEvent::DoubleClick {
                position,
                button: MouseButton::Left,
            } => {
                if rect.contains(*position) {
                    if self.cache_dirty {
                        self.rebuild_cache();
                    }
                    let content_y = position.y - rect.min.y - 2.0 + self.scroll_offset;
                    let idx = (content_y / self.row_height).floor() as usize;
                    if idx < self.flattened_cache.len() {
                        let data = self.flattened_cache[idx].data;
                        if self.flattened_cache[idx].has_children {
                            self.toggle_node(data);
                        }
                    }
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::Scroll { delta, .. } => {
                if self.cache_dirty {
                    self.rebuild_cache();
                }
                let view_h = rect.height();
                let max_sc = (self.flattened_cache.len() as f32 * self.row_height - view_h).max(0.0);
                self.scroll_offset =
                    (self.scroll_offset - delta.y * self.row_height * 3.0).clamp(0.0, max_sc);
                EventReply::Handled
            }
            UIEvent::KeyInput {
                key: KeyCode::ArrowRight,
                pressed: true,
                ..
            } => {
                if let Some(&idx) = self.selection.selected.last() {
                    if self.cache_dirty {
                        self.rebuild_cache();
                    }
                    if idx < self.flattened_cache.len() {
                        let row = &self.flattened_cache[idx];
                        if row.has_children && !row.expanded {
                            let data = row.data;
                            self.toggle_node(data);
                        }
                    }
                }
                EventReply::Handled
            }
            UIEvent::KeyInput {
                key: KeyCode::ArrowLeft,
                pressed: true,
                ..
            } => {
                if let Some(&idx) = self.selection.selected.last() {
                    if self.cache_dirty {
                        self.rebuild_cache();
                    }
                    if idx < self.flattened_cache.len() {
                        let row = &self.flattened_cache[idx];
                        if row.has_children && row.expanded {
                            let data = row.data;
                            self.toggle_node(data);
                        }
                    }
                }
                EventReply::Handled
            }
            UIEvent::KeyInput {
                key: KeyCode::ArrowUp,
                pressed: true,
                ..
            } => {
                if let Some(&current) = self.selection.selected.last() {
                    if current > 0 {
                        self.selection.select(current - 1, false, false);
                    }
                }
                EventReply::Handled
            }
            UIEvent::KeyInput {
                key: KeyCode::ArrowDown,
                pressed: true,
                ..
            } => {
                if self.cache_dirty {
                    self.rebuild_cache();
                }
                if let Some(&current) = self.selection.selected.last() {
                    if current + 1 < self.flattened_cache.len() {
                        self.selection.select(current + 1, false, false);
                    }
                } else if !self.flattened_cache.is_empty() {
                    self.selection.select(0, false, false);
                }
                EventReply::Handled
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 3. TileView
// =========================================================================

/// A tile (grid cell) item.
#[derive(Debug, Clone)]
pub struct TileViewItem {
    pub label: String,
    pub icon: Option<TextureId>,
    pub data: u64,
    pub thumbnail_color: Color,
}

impl TileViewItem {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            icon: None,
            data: 0,
            thumbnail_color: Color::from_hex("#444444"),
        }
    }
}

/// Virtualized grid of tiles.
#[derive(Debug, Clone)]
pub struct TileView {
    pub id: UIId,
    pub items: Vec<TileViewItem>,
    pub selection: SelectionState,
    pub tile_size: Vec2,
    pub tile_spacing: f32,
    pub font_size: f32,
    pub font_id: u32,
    pub text_color: Color,
    pub background_color: Color,
    pub selected_border: Color,
    pub hovered_bg: Color,
    pub border_color: Color,
    pub corner_radii: CornerRadii,
    pub scroll_offset: f32,
    pub hovered_index: Option<usize>,
    pub desired_size: Vec2,
    pub visible: bool,
    pub enabled: bool,
    pub label_height: f32,
}

impl TileView {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            items: Vec::new(),
            selection: SelectionState::new(SelectionMode::Single),
            tile_size: Vec2::new(100.0, 120.0),
            tile_spacing: 8.0,
            font_size: 11.0,
            font_id: 0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#1E1E1E"),
            selected_border: Color::from_hex("#4488FF"),
            hovered_bg: Color::from_hex("#333333"),
            border_color: Color::from_hex("#444444"),
            corner_radii: CornerRadii::all(3.0),
            scroll_offset: 0.0,
            hovered_index: None,
            desired_size: Vec2::new(400.0, 300.0),
            visible: true,
            enabled: true,
            label_height: 20.0,
        }
    }

    fn columns(&self, width: f32) -> usize {
        ((width + self.tile_spacing) / (self.tile_size.x + self.tile_spacing)).max(1.0) as usize
    }

    fn rows(&self, cols: usize) -> usize {
        if cols == 0 {
            return 0;
        }
        (self.items.len() + cols - 1) / cols
    }

    fn total_content_height(&self, cols: usize) -> f32 {
        self.rows(cols) as f32 * (self.tile_size.y + self.tile_spacing) + self.tile_spacing
    }

    fn tile_rect_for_index(&self, index: usize, cols: usize, rect: Rect) -> Rect {
        let col = index % cols;
        let row = index / cols;
        let x = rect.min.x + self.tile_spacing + col as f32 * (self.tile_size.x + self.tile_spacing);
        let y = rect.min.y + self.tile_spacing + row as f32 * (self.tile_size.y + self.tile_spacing)
            - self.scroll_offset;
        Rect::new(
            Vec2::new(x, y),
            Vec2::new(x + self.tile_size.x, y + self.tile_size.y),
        )
    }

    fn index_at_position(&self, pos: Vec2, rect: Rect) -> Option<usize> {
        let cols = self.columns(rect.width());
        for i in 0..self.items.len() {
            let tile_rect = self.tile_rect_for_index(i, cols, rect);
            if tile_rect.contains(pos) {
                return Some(i);
            }
        }
        None
    }
}

impl SlateWidget for TileView {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        self.desired_size
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        draw_list.draw_rounded_rect(
            rect,
            self.background_color,
            self.corner_radii,
            BorderSpec::new(self.border_color, 1.0),
        );

        draw_list.push_clip(rect);

        let cols = self.columns(rect.width());
        for (i, item) in self.items.iter().enumerate() {
            let tile_rect = self.tile_rect_for_index(i, cols, rect);

            // Skip tiles outside visible area.
            if tile_rect.max.y < rect.min.y || tile_rect.min.y > rect.max.y {
                continue;
            }

            // Thumbnail area.
            let thumb_rect = Rect::new(
                tile_rect.min,
                Vec2::new(tile_rect.max.x, tile_rect.max.y - self.label_height),
            );

            let bg = if self.hovered_index == Some(i) {
                self.hovered_bg
            } else {
                item.thumbnail_color
            };
            draw_list.draw_rounded_rect(
                thumb_rect,
                bg,
                CornerRadii::new(4.0, 4.0, 0.0, 0.0),
                BorderSpec::default(),
            );

            // Icon / thumbnail.
            if let Some(icon) = item.icon {
                draw_list.push(DrawCommand::Image {
                    rect: thumb_rect,
                    texture: icon,
                    tint: Color::WHITE,
                    corner_radii: CornerRadii::new(4.0, 4.0, 0.0, 0.0),
                    scale_mode: ImageScaleMode::Fit,
                    uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
                });
            }

            // Label area.
            let label_rect = Rect::new(
                Vec2::new(tile_rect.min.x, tile_rect.max.y - self.label_height),
                tile_rect.max,
            );
            draw_list.draw_rounded_rect(
                label_rect,
                Color::from_hex("#2A2A2A"),
                CornerRadii::new(0.0, 0.0, 4.0, 4.0),
                BorderSpec::default(),
            );
            let ty = label_rect.min.y + (self.label_height - self.font_size) * 0.5;
            draw_list.push(DrawCommand::Text {
                text: item.label.clone(),
                position: Vec2::new(label_rect.min.x + 4.0, ty),
                font_size: self.font_size,
                color: self.text_color,
                font_id: self.font_id,
                max_width: Some(self.tile_size.x - 8.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            // Selection border.
            if self.selection.is_selected(i) {
                draw_list.draw_rounded_rect(
                    tile_rect,
                    Color::TRANSPARENT,
                    CornerRadii::all(4.0),
                    BorderSpec::new(self.selected_border, 2.0),
                );
            }
        }

        draw_list.pop_clip();
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled {
            return EventReply::Unhandled;
        }

        match event {
            UIEvent::Hover { position } => {
                self.hovered_index = self.index_at_position(*position, rect);
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                modifiers,
            } => {
                if let Some(idx) = self.index_at_position(*position, rect) {
                    self.selection.select(idx, modifiers.ctrl, modifiers.shift);
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::Scroll { delta, .. } => {
                let cols = self.columns(rect.width());
                let max_scroll = (self.total_content_height(cols) - rect.height()).max(0.0);
                self.scroll_offset =
                    (self.scroll_offset - delta.y * 40.0).clamp(0.0, max_scroll);
                EventReply::Handled
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 4. HeaderRow
// =========================================================================

/// Sort direction indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    None,
    Ascending,
    Descending,
}

impl Default for SortDirection {
    fn default() -> Self {
        SortDirection::None
    }
}

/// A single column header.
#[derive(Debug, Clone)]
pub struct ColumnHeader {
    pub label: String,
    pub width: f32,
    pub min_width: f32,
    pub max_width: f32,
    pub sort_direction: SortDirection,
    pub resizable: bool,
    pub reorderable: bool,
    pub hovered: bool,
    pub resize_hovered: bool,
}

impl ColumnHeader {
    pub fn new(label: &str, width: f32) -> Self {
        Self {
            label: label.to_string(),
            width,
            min_width: 30.0,
            max_width: 600.0,
            sort_direction: SortDirection::None,
            resizable: true,
            reorderable: true,
            hovered: false,
            resize_hovered: false,
        }
    }
}

/// Column headers with resize handles, reorder via drag, and sort indicators.
#[derive(Debug, Clone)]
pub struct HeaderRow {
    pub id: UIId,
    pub columns: Vec<ColumnHeader>,
    pub height: f32,
    pub background_color: Color,
    pub text_color: Color,
    pub border_color: Color,
    pub hovered_bg: Color,
    pub font_size: f32,
    pub font_id: u32,
    pub sort_changed: Option<usize>,
    pub resizing_column: Option<usize>,
    pub resizing_start_width: f32,
    pub resizing_start_x: f32,
    pub dragging_column: Option<usize>,
    pub drag_insert_index: Option<usize>,
    pub visible: bool,
    pub resize_handle_width: f32,
}

impl HeaderRow {
    pub fn new(columns: Vec<ColumnHeader>) -> Self {
        Self {
            id: UIId::INVALID,
            columns,
            height: 26.0,
            background_color: Color::from_hex("#2D2D2D"),
            text_color: Color::from_hex("#CCCCCC"),
            border_color: Color::from_hex("#444444"),
            hovered_bg: Color::from_hex("#383838"),
            font_size: 12.0,
            font_id: 0,
            sort_changed: None,
            resizing_column: None,
            resizing_start_width: 0.0,
            resizing_start_x: 0.0,
            dragging_column: None,
            drag_insert_index: None,
            visible: true,
            resize_handle_width: 6.0,
        }
    }

    pub fn total_width(&self) -> f32 {
        self.columns.iter().map(|c| c.width).sum()
    }

    fn column_x_offset(&self, index: usize) -> f32 {
        self.columns.iter().take(index).map(|c| c.width).sum()
    }

    fn column_at_x(&self, x: f32, rect: Rect) -> Option<usize> {
        let mut acc = rect.min.x;
        for (i, col) in self.columns.iter().enumerate() {
            if x >= acc && x < acc + col.width {
                return Some(i);
            }
            acc += col.width;
        }
        None
    }

    fn resize_handle_at_x(&self, x: f32, rect: Rect) -> Option<usize> {
        let hw = self.resize_handle_width * 0.5;
        let mut acc = rect.min.x;
        for (i, col) in self.columns.iter().enumerate() {
            acc += col.width;
            if x >= acc - hw && x <= acc + hw && col.resizable {
                return Some(i);
            }
        }
        None
    }

    pub fn take_sort_changed(&mut self) -> Option<usize> {
        self.sort_changed.take()
    }
}

impl SlateWidget for HeaderRow {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        Vec2::new(self.total_width(), self.height)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        draw_list.draw_rect(rect, self.background_color);

        let mut x = rect.min.x;
        for (i, col) in self.columns.iter().enumerate() {
            let col_rect = Rect::new(
                Vec2::new(x, rect.min.y),
                Vec2::new(x + col.width, rect.max.y),
            );

            if col.hovered {
                draw_list.draw_rect(col_rect, self.hovered_bg);
            }

            // Label.
            let ty = rect.min.y + (self.height - self.font_size) * 0.5;
            draw_list.push(DrawCommand::Text {
                text: col.label.clone(),
                position: Vec2::new(x + 6.0, ty),
                font_size: self.font_size,
                color: self.text_color,
                font_id: self.font_id,
                max_width: Some(col.width - 20.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            // Sort indicator.
            match col.sort_direction {
                SortDirection::Ascending => {
                    let sx = x + col.width - 14.0;
                    let sy = rect.min.y + self.height * 0.5;
                    draw_list.push(DrawCommand::Triangle {
                        p0: Vec2::new(sx, sy - 3.0),
                        p1: Vec2::new(sx - 4.0, sy + 3.0),
                        p2: Vec2::new(sx + 4.0, sy + 3.0),
                        color: self.text_color,
                    });
                }
                SortDirection::Descending => {
                    let sx = x + col.width - 14.0;
                    let sy = rect.min.y + self.height * 0.5;
                    draw_list.push(DrawCommand::Triangle {
                        p0: Vec2::new(sx, sy + 3.0),
                        p1: Vec2::new(sx - 4.0, sy - 3.0),
                        p2: Vec2::new(sx + 4.0, sy - 3.0),
                        color: self.text_color,
                    });
                }
                SortDirection::None => {}
            }

            // Column separator.
            draw_list.draw_line(
                Vec2::new(x + col.width, rect.min.y + 2.0),
                Vec2::new(x + col.width, rect.max.y - 2.0),
                self.border_color,
                1.0,
            );

            // Resize handle hover indicator.
            if col.resize_hovered || self.resizing_column == Some(i) {
                draw_list.draw_line(
                    Vec2::new(x + col.width, rect.min.y),
                    Vec2::new(x + col.width, rect.max.y),
                    Color::from_hex("#4488FF"),
                    2.0,
                );
            }

            x += col.width;
        }

        // Bottom border.
        draw_list.draw_line(
            Vec2::new(rect.min.x, rect.max.y),
            Vec2::new(rect.max.x, rect.max.y),
            self.border_color,
            1.0,
        );

        // Drag insertion indicator.
        if let Some(insert_idx) = self.drag_insert_index {
            let ix = rect.min.x + self.column_x_offset(insert_idx);
            draw_list.draw_line(
                Vec2::new(ix, rect.min.y),
                Vec2::new(ix, rect.max.y),
                Color::from_hex("#FF8800"),
                3.0,
            );
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        match event {
            UIEvent::Hover { position } => {
                if !rect.contains(*position) {
                    for col in &mut self.columns {
                        col.hovered = false;
                        col.resize_hovered = false;
                    }
                    return EventReply::Unhandled;
                }

                // Check resize handles.
                let resize_col = self.resize_handle_at_x(position.x, rect);
                for (i, col) in self.columns.iter_mut().enumerate() {
                    col.resize_hovered = resize_col == Some(i);
                    col.hovered = false;
                }

                if resize_col.is_none() {
                    if let Some(idx) = self.column_at_x(position.x, rect) {
                        if idx < self.columns.len() {
                            self.columns[idx].hovered = true;
                        }
                    }
                }
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if !rect.contains(*position) {
                    return EventReply::Unhandled;
                }

                // Start resize?
                if let Some(col_idx) = self.resize_handle_at_x(position.x, rect) {
                    self.resizing_column = Some(col_idx);
                    self.resizing_start_width = self.columns[col_idx].width;
                    self.resizing_start_x = position.x;
                    return EventReply::Handled.then(EventReply::CaptureMouse);
                }

                // Column click (sort).
                if let Some(idx) = self.column_at_x(position.x, rect) {
                    if idx < self.columns.len() {
                        let new_dir = match self.columns[idx].sort_direction {
                            SortDirection::None => SortDirection::Ascending,
                            SortDirection::Ascending => SortDirection::Descending,
                            SortDirection::Descending => SortDirection::None,
                        };
                        // Clear all sort indicators.
                        for col in &mut self.columns {
                            col.sort_direction = SortDirection::None;
                        }
                        self.columns[idx].sort_direction = new_dir;
                        self.sort_changed = Some(idx);
                    }
                }

                EventReply::Handled
            }
            UIEvent::DragMove { position, delta } => {
                if let Some(col_idx) = self.resizing_column {
                    let dx = position.x - self.resizing_start_x;
                    let new_width = (self.resizing_start_width + dx)
                        .clamp(self.columns[col_idx].min_width, self.columns[col_idx].max_width);
                    self.columns[col_idx].width = new_width;
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::MouseUp {
                button: MouseButton::Left,
                ..
            }
            | UIEvent::DragEnd { .. } => {
                if self.resizing_column.is_some() {
                    self.resizing_column = None;
                    return EventReply::Handled.then(EventReply::ReleaseMouse);
                }
                if self.dragging_column.is_some() {
                    // Reorder.
                    if let (Some(src), Some(dst)) =
                        (self.dragging_column, self.drag_insert_index)
                    {
                        if src != dst && src < self.columns.len() && dst <= self.columns.len() {
                            let col = self.columns.remove(src);
                            let insert = if dst > src { dst - 1 } else { dst };
                            self.columns.insert(insert, col);
                        }
                    }
                    self.dragging_column = None;
                    self.drag_insert_index = None;
                    return EventReply::Handled.then(EventReply::ReleaseMouse);
                }
                EventReply::Unhandled
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 5. ColorPicker
// =========================================================================

/// Full color selection widget with HSV, RGB, hex, and alpha.
#[derive(Debug, Clone)]
pub struct ColorPicker {
    pub id: UIId,
    pub color: Color,
    pub hue: f32,
    pub saturation: f32,
    pub value: f32,
    pub alpha: f32,
    pub hex_text: String,
    pub r_text: String,
    pub g_text: String,
    pub b_text: String,
    pub show_alpha: bool,
    pub sv_rect_size: f32,
    pub hue_bar_width: f32,
    pub alpha_bar_width: f32,
    pub recent_colors: Vec<Color>,
    pub max_recent: usize,
    pub srgb_mode: bool,
    pub eye_dropper_mode: bool,
    pub dragging_sv: bool,
    pub dragging_hue: bool,
    pub dragging_alpha: bool,
    pub editing_hex: bool,
    pub color_changed: bool,
    pub desired_size: Vec2,
    pub font_size: f32,
    pub font_id: u32,
    pub visible: bool,
    pub background_color: Color,
    pub border_color: Color,
}

impl ColorPicker {
    pub fn new(initial: Color) -> Self {
        let (h, s, v) = initial.to_hsv();
        Self {
            id: UIId::INVALID,
            color: initial,
            hue: h,
            saturation: s,
            value: v,
            alpha: initial.a,
            hex_text: Self::color_to_hex(initial),
            r_text: format!("{}", (initial.r * 255.0) as u8),
            g_text: format!("{}", (initial.g * 255.0) as u8),
            b_text: format!("{}", (initial.b * 255.0) as u8),
            show_alpha: true,
            sv_rect_size: 180.0,
            hue_bar_width: 20.0,
            alpha_bar_width: 20.0,
            recent_colors: Vec::new(),
            max_recent: 16,
            srgb_mode: true,
            eye_dropper_mode: false,
            dragging_sv: false,
            dragging_hue: false,
            dragging_alpha: false,
            editing_hex: false,
            color_changed: false,
            desired_size: Vec2::new(260.0, 320.0),
            font_size: 12.0,
            font_id: 0,
            visible: true,
            background_color: Color::from_hex("#2A2A2A"),
            border_color: Color::from_hex("#555555"),
        }
    }

    fn update_from_hsv(&mut self) {
        self.color = Color::from_hsv(self.hue, self.saturation, self.value);
        self.color.a = self.alpha;
        self.hex_text = Self::color_to_hex(self.color);
        self.r_text = format!("{}", (self.color.r * 255.0) as u8);
        self.g_text = format!("{}", (self.color.g * 255.0) as u8);
        self.b_text = format!("{}", (self.color.b * 255.0) as u8);
        self.color_changed = true;
    }

    fn update_from_rgb(&mut self) {
        let (h, s, v) = self.color.to_hsv();
        self.hue = h;
        self.saturation = s;
        self.value = v;
        self.hex_text = Self::color_to_hex(self.color);
        self.r_text = format!("{}", (self.color.r * 255.0) as u8);
        self.g_text = format!("{}", (self.color.g * 255.0) as u8);
        self.b_text = format!("{}", (self.color.b * 255.0) as u8);
        self.color_changed = true;
    }

    pub fn set_color(&mut self, c: Color) {
        self.color = c;
        self.alpha = c.a;
        self.update_from_rgb();
    }

    pub fn add_to_recent(&mut self) {
        if self.recent_colors.len() >= self.max_recent {
            self.recent_colors.remove(0);
        }
        self.recent_colors.push(self.color);
    }

    pub fn take_color_changed(&mut self) -> bool {
        let c = self.color_changed;
        self.color_changed = false;
        c
    }

    fn color_to_hex(c: Color) -> String {
        format!(
            "#{:02X}{:02X}{:02X}",
            (c.r * 255.0) as u8,
            (c.g * 255.0) as u8,
            (c.b * 255.0) as u8
        )
    }

    fn sv_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.min.x + 8.0, rect.min.y + 8.0),
            Vec2::new(
                rect.min.x + 8.0 + self.sv_rect_size,
                rect.min.y + 8.0 + self.sv_rect_size,
            ),
        )
    }

    fn hue_bar_rect(&self, rect: Rect) -> Rect {
        let sv = self.sv_rect(rect);
        Rect::new(
            Vec2::new(sv.max.x + 8.0, sv.min.y),
            Vec2::new(sv.max.x + 8.0 + self.hue_bar_width, sv.max.y),
        )
    }

    fn alpha_bar_rect(&self, rect: Rect) -> Rect {
        let hue = self.hue_bar_rect(rect);
        Rect::new(
            Vec2::new(hue.max.x + 8.0, hue.min.y),
            Vec2::new(hue.max.x + 8.0 + self.alpha_bar_width, hue.max.y),
        )
    }
}

impl SlateWidget for ColorPicker {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        self.desired_size
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        draw_list.draw_rounded_rect(
            rect,
            self.background_color,
            CornerRadii::all(6.0),
            BorderSpec::new(self.border_color, 1.0),
        );

        let sv = self.sv_rect(rect);
        let hue_bar = self.hue_bar_rect(rect);

        // SV square -- simplified rendering with colour blocks.
        let steps = 16;
        let step_w = sv.width() / steps as f32;
        let step_h = sv.height() / steps as f32;
        for sy in 0..steps {
            for sx in 0..steps {
                let s = (sx as f32 + 0.5) / steps as f32;
                let v = 1.0 - (sy as f32 + 0.5) / steps as f32;
                let c = Color::from_hsv(self.hue, s, v);
                let block = Rect::new(
                    Vec2::new(sv.min.x + sx as f32 * step_w, sv.min.y + sy as f32 * step_h),
                    Vec2::new(
                        sv.min.x + (sx + 1) as f32 * step_w,
                        sv.min.y + (sy + 1) as f32 * step_h,
                    ),
                );
                draw_list.draw_rect(block, c);
            }
        }

        // SV cursor.
        let sv_cx = sv.min.x + self.saturation * sv.width();
        let sv_cy = sv.min.y + (1.0 - self.value) * sv.height();
        draw_list.push(DrawCommand::Circle {
            center: Vec2::new(sv_cx, sv_cy),
            radius: 6.0,
            color: Color::TRANSPARENT,
            border: BorderSpec::new(Color::WHITE, 2.0),
        });
        draw_list.push(DrawCommand::Circle {
            center: Vec2::new(sv_cx, sv_cy),
            radius: 5.0,
            color: Color::TRANSPARENT,
            border: BorderSpec::new(Color::BLACK, 1.0),
        });

        // Hue bar.
        let hue_steps = 24;
        let hue_step_h = hue_bar.height() / hue_steps as f32;
        for i in 0..hue_steps {
            let h = (i as f32 / hue_steps as f32) * 360.0;
            let c = Color::from_hsv(h, 1.0, 1.0);
            let block = Rect::new(
                Vec2::new(hue_bar.min.x, hue_bar.min.y + i as f32 * hue_step_h),
                Vec2::new(hue_bar.max.x, hue_bar.min.y + (i + 1) as f32 * hue_step_h),
            );
            draw_list.draw_rect(block, c);
        }

        // Hue cursor.
        let hue_y = hue_bar.min.y + (self.hue / 360.0) * hue_bar.height();
        draw_list.draw_rounded_rect(
            Rect::new(
                Vec2::new(hue_bar.min.x - 2.0, hue_y - 2.0),
                Vec2::new(hue_bar.max.x + 2.0, hue_y + 2.0),
            ),
            Color::TRANSPARENT,
            CornerRadii::all(2.0),
            BorderSpec::new(Color::WHITE, 2.0),
        );

        // Alpha bar.
        if self.show_alpha {
            let alpha_bar = self.alpha_bar_rect(rect);
            // Checkerboard background.
            let check_size = 6.0;
            let cols_check = (alpha_bar.width() / check_size).ceil() as usize;
            let rows_check = (alpha_bar.height() / check_size).ceil() as usize;
            for cy in 0..rows_check {
                for cx in 0..cols_check {
                    let c = if (cx + cy) % 2 == 0 {
                        Color::from_hex("#999999")
                    } else {
                        Color::from_hex("#666666")
                    };
                    let block = Rect::new(
                        Vec2::new(
                            alpha_bar.min.x + cx as f32 * check_size,
                            alpha_bar.min.y + cy as f32 * check_size,
                        ),
                        Vec2::new(
                            (alpha_bar.min.x + (cx + 1) as f32 * check_size).min(alpha_bar.max.x),
                            (alpha_bar.min.y + (cy + 1) as f32 * check_size).min(alpha_bar.max.y),
                        ),
                    );
                    draw_list.draw_rect(block, c);
                }
            }

            // Gradient overlay.
            let alpha_steps = 16;
            let alpha_step_h = alpha_bar.height() / alpha_steps as f32;
            for i in 0..alpha_steps {
                let a = 1.0 - i as f32 / alpha_steps as f32;
                let c = self.color.with_alpha(a);
                let block = Rect::new(
                    Vec2::new(alpha_bar.min.x, alpha_bar.min.y + i as f32 * alpha_step_h),
                    Vec2::new(alpha_bar.max.x, alpha_bar.min.y + (i + 1) as f32 * alpha_step_h),
                );
                draw_list.draw_rect(block, c);
            }

            // Alpha cursor.
            let alpha_y = alpha_bar.min.y + (1.0 - self.alpha) * alpha_bar.height();
            draw_list.draw_rounded_rect(
                Rect::new(
                    Vec2::new(alpha_bar.min.x - 2.0, alpha_y - 2.0),
                    Vec2::new(alpha_bar.max.x + 2.0, alpha_y + 2.0),
                ),
                Color::TRANSPARENT,
                CornerRadii::all(2.0),
                BorderSpec::new(Color::WHITE, 2.0),
            );
        }

        // Preview swatch.
        let preview_y = sv.max.y + 12.0;
        let preview_rect = Rect::new(
            Vec2::new(rect.min.x + 8.0, preview_y),
            Vec2::new(rect.min.x + 48.0, preview_y + 28.0),
        );
        draw_list.draw_rounded_rect(
            preview_rect,
            self.color,
            CornerRadii::all(4.0),
            BorderSpec::new(self.border_color, 1.0),
        );

        // Hex display.
        draw_list.push(DrawCommand::Text {
            text: self.hex_text.clone(),
            position: Vec2::new(preview_rect.max.x + 8.0, preview_y + 6.0),
            font_size: self.font_size,
            color: Color::WHITE,
            font_id: self.font_id,
            max_width: None,
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });

        // RGB values.
        let rgb_y = preview_y + 34.0;
        let labels = ["R:", "G:", "B:"];
        let values = [&self.r_text, &self.g_text, &self.b_text];
        for (i, (lbl, val)) in labels.iter().zip(values.iter()).enumerate() {
            let ly = rgb_y + i as f32 * 18.0;
            draw_list.push(DrawCommand::Text {
                text: format!("{} {}", lbl, val),
                position: Vec2::new(rect.min.x + 8.0, ly),
                font_size: self.font_size,
                color: Color::WHITE,
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }

        // sRGB / Linear toggle label.
        let mode_y = rgb_y + 60.0;
        let mode_text = if self.srgb_mode { "sRGB" } else { "Linear" };
        draw_list.push(DrawCommand::Text {
            text: mode_text.to_string(),
            position: Vec2::new(rect.min.x + 8.0, mode_y),
            font_size: 11.0,
            color: Color::from_hex("#888888"),
            font_id: self.font_id,
            max_width: None,
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });

        // Recent colors.
        if !self.recent_colors.is_empty() {
            let recent_y = mode_y + 20.0;
            draw_list.push(DrawCommand::Text {
                text: "Recent:".to_string(),
                position: Vec2::new(rect.min.x + 8.0, recent_y),
                font_size: 11.0,
                color: Color::from_hex("#888888"),
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
            let swatch_y = recent_y + 16.0;
            let swatch_size = 14.0;
            for (i, c) in self.recent_colors.iter().enumerate() {
                let sx = rect.min.x + 8.0 + i as f32 * (swatch_size + 3.0);
                draw_list.draw_rounded_rect(
                    Rect::new(
                        Vec2::new(sx, swatch_y),
                        Vec2::new(sx + swatch_size, swatch_y + swatch_size),
                    ),
                    *c,
                    CornerRadii::all(2.0),
                    BorderSpec::new(self.border_color, 1.0),
                );
            }
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        match event {
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                let sv = self.sv_rect(rect);
                let hue_bar = self.hue_bar_rect(rect);

                if sv.contains(*position) {
                    self.dragging_sv = true;
                    self.saturation = ((position.x - sv.min.x) / sv.width()).clamp(0.0, 1.0);
                    self.value =
                        1.0 - ((position.y - sv.min.y) / sv.height()).clamp(0.0, 1.0);
                    self.update_from_hsv();
                    return EventReply::Handled.then(EventReply::CaptureMouse);
                }

                if hue_bar.contains(*position) {
                    self.dragging_hue = true;
                    self.hue =
                        ((position.y - hue_bar.min.y) / hue_bar.height()).clamp(0.0, 1.0) * 360.0;
                    self.update_from_hsv();
                    return EventReply::Handled.then(EventReply::CaptureMouse);
                }

                if self.show_alpha {
                    let alpha_bar = self.alpha_bar_rect(rect);
                    if alpha_bar.contains(*position) {
                        self.dragging_alpha = true;
                        self.alpha = 1.0
                            - ((position.y - alpha_bar.min.y) / alpha_bar.height()).clamp(0.0, 1.0);
                        self.update_from_hsv();
                        return EventReply::Handled.then(EventReply::CaptureMouse);
                    }
                }

                // Recent color click.
                if !self.recent_colors.is_empty() {
                    let sv_rect = self.sv_rect(rect);
                    let recent_base_y = sv_rect.max.y + 12.0 + 28.0 + 34.0 + 60.0 + 16.0;
                    let swatch_size = 14.0;
                    for (i, c) in self.recent_colors.iter().enumerate() {
                        let sx = rect.min.x + 8.0 + i as f32 * (swatch_size + 3.0);
                        let swatch_rect = Rect::new(
                            Vec2::new(sx, recent_base_y),
                            Vec2::new(sx + swatch_size, recent_base_y + swatch_size),
                        );
                        if swatch_rect.contains(*position) {
                            self.color = *c;
                            self.alpha = c.a;
                            self.update_from_rgb();
                            return EventReply::Handled;
                        }
                    }
                }

                EventReply::Unhandled
            }
            UIEvent::DragMove { position, .. } => {
                if self.dragging_sv {
                    let sv = self.sv_rect(rect);
                    self.saturation = ((position.x - sv.min.x) / sv.width()).clamp(0.0, 1.0);
                    self.value =
                        1.0 - ((position.y - sv.min.y) / sv.height()).clamp(0.0, 1.0);
                    self.update_from_hsv();
                    return EventReply::Handled;
                }
                if self.dragging_hue {
                    let hue_bar = self.hue_bar_rect(rect);
                    self.hue = ((position.y - hue_bar.min.y) / hue_bar.height()).clamp(0.0, 1.0)
                        * 360.0;
                    self.update_from_hsv();
                    return EventReply::Handled;
                }
                if self.dragging_alpha {
                    let alpha_bar = self.alpha_bar_rect(rect);
                    self.alpha = 1.0
                        - ((position.y - alpha_bar.min.y) / alpha_bar.height()).clamp(0.0, 1.0);
                    self.update_from_hsv();
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::MouseUp { .. } | UIEvent::DragEnd { .. } => {
                let was_dragging =
                    self.dragging_sv || self.dragging_hue || self.dragging_alpha;
                self.dragging_sv = false;
                self.dragging_hue = false;
                self.dragging_alpha = false;
                if was_dragging {
                    EventReply::Handled.then(EventReply::ReleaseMouse)
                } else {
                    EventReply::Unhandled
                }
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 6. CurveEditor
// =========================================================================

/// A keyframe on the curve.
#[derive(Debug, Clone, Copy)]
pub struct CurveKeyframe {
    pub time: f32,
    pub value: f32,
    pub in_tangent: Vec2,
    pub out_tangent: Vec2,
    pub tangent_mode: TangentMode,
}

/// Tangent mode for keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TangentMode {
    Auto,
    Linear,
    Step,
    Free,
}

impl Default for TangentMode {
    fn default() -> Self {
        TangentMode::Auto
    }
}

/// Curve editor with keyframes, tangent handles, zoom/pan.
#[derive(Debug, Clone)]
pub struct CurveEditor {
    pub id: UIId,
    pub keyframes: Vec<CurveKeyframe>,
    pub selected_key: Option<usize>,
    pub hovered_key: Option<usize>,
    pub dragging_key: Option<usize>,
    pub dragging_tangent: Option<(usize, bool)>,
    pub view_offset: Vec2,
    pub view_scale: Vec2,
    pub grid_color: Color,
    pub curve_color: Color,
    pub key_color: Color,
    pub selected_key_color: Color,
    pub tangent_color: Color,
    pub background_color: Color,
    pub border_color: Color,
    pub font_size: f32,
    pub font_id: u32,
    pub key_radius: f32,
    pub desired_size: Vec2,
    pub visible: bool,
    pub panning: bool,
    pub pan_start: Vec2,
    pub pan_start_offset: Vec2,
    pub snap_to_grid: bool,
    pub grid_x_spacing: f32,
    pub grid_y_spacing: f32,
    pub value_changed: bool,
}

impl CurveEditor {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            keyframes: Vec::new(),
            selected_key: None,
            hovered_key: None,
            dragging_key: None,
            dragging_tangent: None,
            view_offset: Vec2::ZERO,
            view_scale: Vec2::new(100.0, 100.0),
            grid_color: Color::from_hex("#333333"),
            curve_color: Color::from_hex("#44BB44"),
            key_color: Color::WHITE,
            selected_key_color: Color::from_hex("#FFAA00"),
            tangent_color: Color::from_hex("#888888"),
            background_color: Color::from_hex("#1A1A1A"),
            border_color: Color::from_hex("#444444"),
            font_size: 10.0,
            font_id: 0,
            key_radius: 5.0,
            desired_size: Vec2::new(400.0, 200.0),
            visible: true,
            panning: false,
            pan_start: Vec2::ZERO,
            pan_start_offset: Vec2::ZERO,
            snap_to_grid: false,
            grid_x_spacing: 1.0,
            grid_y_spacing: 0.25,
            value_changed: false,
        }
    }

    pub fn add_keyframe(&mut self, time: f32, value: f32) {
        self.keyframes.push(CurveKeyframe {
            time,
            value,
            in_tangent: Vec2::new(-0.3, 0.0),
            out_tangent: Vec2::new(0.3, 0.0),
            tangent_mode: TangentMode::Auto,
        });
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        self.value_changed = true;
    }

    pub fn remove_selected(&mut self) {
        if let Some(idx) = self.selected_key {
            if idx < self.keyframes.len() {
                self.keyframes.remove(idx);
                self.selected_key = None;
                self.value_changed = true;
            }
        }
    }

    fn world_to_screen(&self, world: Vec2, rect: Rect) -> Vec2 {
        let cx = (rect.min.x + rect.max.x) * 0.5;
        let cy = (rect.min.y + rect.max.y) * 0.5;
        Vec2::new(
            cx + (world.x - self.view_offset.x) * self.view_scale.x,
            cy - (world.y - self.view_offset.y) * self.view_scale.y,
        )
    }

    fn screen_to_world(&self, screen: Vec2, rect: Rect) -> Vec2 {
        let cx = (rect.min.x + rect.max.x) * 0.5;
        let cy = (rect.min.y + rect.max.y) * 0.5;
        Vec2::new(
            (screen.x - cx) / self.view_scale.x + self.view_offset.x,
            -((screen.y - cy) / self.view_scale.y) + self.view_offset.y,
        )
    }

    fn key_screen_pos(&self, idx: usize, rect: Rect) -> Vec2 {
        let k = &self.keyframes[idx];
        self.world_to_screen(Vec2::new(k.time, k.value), rect)
    }

    /// Evaluate the curve at a given time using cubic hermite interpolation.
    pub fn evaluate(&self, time: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].value;
        }
        if time <= self.keyframes[0].time {
            return self.keyframes[0].value;
        }
        let last = self.keyframes.len() - 1;
        if time >= self.keyframes[last].time {
            return self.keyframes[last].value;
        }

        // Find segment.
        let mut seg = 0;
        for i in 0..last {
            if time >= self.keyframes[i].time && time <= self.keyframes[i + 1].time {
                seg = i;
                break;
            }
        }

        let k0 = &self.keyframes[seg];
        let k1 = &self.keyframes[seg + 1];
        let dt = k1.time - k0.time;
        if dt.abs() < 1e-10 {
            return k0.value;
        }
        let t = (time - k0.time) / dt;

        // Cubic hermite.
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        let m0 = k0.out_tangent.y * dt;
        let m1 = k1.in_tangent.y * dt;

        h00 * k0.value + h10 * m0 + h01 * k1.value + h11 * m1
    }
}

impl SlateWidget for CurveEditor {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        self.desired_size
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        draw_list.draw_rounded_rect(
            rect,
            self.background_color,
            CornerRadii::all(4.0),
            BorderSpec::new(self.border_color, 1.0),
        );

        draw_list.push_clip(rect);

        // Grid lines.
        let world_min = self.screen_to_world(rect.min, rect);
        let world_max = self.screen_to_world(rect.max, rect);

        // Vertical grid (time axis).
        let x_start = (world_min.x / self.grid_x_spacing).floor() * self.grid_x_spacing;
        let mut gx = x_start;
        while gx < world_max.x {
            let sx = self.world_to_screen(Vec2::new(gx, 0.0), rect).x;
            if sx >= rect.min.x && sx <= rect.max.x {
                let color = if (gx.abs() < 0.01) {
                    Color::from_hex("#555555")
                } else {
                    self.grid_color
                };
                draw_list.draw_line(
                    Vec2::new(sx, rect.min.y),
                    Vec2::new(sx, rect.max.y),
                    color,
                    1.0,
                );
            }
            gx += self.grid_x_spacing;
        }

        // Horizontal grid (value axis).
        let y_start = (world_max.y / self.grid_y_spacing).floor() * self.grid_y_spacing;
        let mut gy = y_start;
        while gy < world_min.y {
            let sy = self.world_to_screen(Vec2::new(0.0, gy), rect).y;
            if sy >= rect.min.y && sy <= rect.max.y {
                let color = if (gy.abs() < 0.01) {
                    Color::from_hex("#555555")
                } else {
                    self.grid_color
                };
                draw_list.draw_line(
                    Vec2::new(rect.min.x, sy),
                    Vec2::new(rect.max.x, sy),
                    color,
                    1.0,
                );
            }
            gy += self.grid_y_spacing;
        }

        // Curve.
        if self.keyframes.len() >= 2 {
            let segments = 100;
            let t_start = self.keyframes[0].time;
            let t_end = self.keyframes.last().unwrap().time;
            let dt = (t_end - t_start) / segments as f32;
            let mut points = Vec::with_capacity(segments + 1);
            for i in 0..=segments {
                let t = t_start + i as f32 * dt;
                let v = self.evaluate(t);
                points.push(self.world_to_screen(Vec2::new(t, v), rect));
            }
            draw_list.push(DrawCommand::Polyline {
                points,
                color: self.curve_color,
                thickness: 2.0,
                closed: false,
            });
        }

        // Keyframes and tangent handles.
        for (i, key) in self.keyframes.iter().enumerate() {
            let pos = self.world_to_screen(Vec2::new(key.time, key.value), rect);
            let selected = self.selected_key == Some(i);

            // Tangent handles (only for selected).
            if selected {
                let in_pos = self.world_to_screen(
                    Vec2::new(key.time + key.in_tangent.x, key.value + key.in_tangent.y),
                    rect,
                );
                let out_pos = self.world_to_screen(
                    Vec2::new(key.time + key.out_tangent.x, key.value + key.out_tangent.y),
                    rect,
                );

                draw_list.draw_line(pos, in_pos, self.tangent_color, 1.0);
                draw_list.draw_line(pos, out_pos, self.tangent_color, 1.0);

                draw_list.push(DrawCommand::Circle {
                    center: in_pos,
                    radius: 3.0,
                    color: self.tangent_color,
                    border: BorderSpec::default(),
                });
                draw_list.push(DrawCommand::Circle {
                    center: out_pos,
                    radius: 3.0,
                    color: self.tangent_color,
                    border: BorderSpec::default(),
                });
            }

            // Key point.
            let kc = if selected {
                self.selected_key_color
            } else if self.hovered_key == Some(i) {
                self.key_color.lighten(0.2)
            } else {
                self.key_color
            };

            draw_list.push(DrawCommand::Circle {
                center: pos,
                radius: self.key_radius,
                color: kc,
                border: BorderSpec::new(Color::BLACK, 1.0),
            });
        }

        draw_list.pop_clip();
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        match event {
            UIEvent::Hover { position } => {
                if !rect.contains(*position) {
                    self.hovered_key = None;
                    return EventReply::Unhandled;
                }
                self.hovered_key = None;
                for (i, _key) in self.keyframes.iter().enumerate() {
                    let kpos = self.key_screen_pos(i, rect);
                    if (*position - kpos).length() <= self.key_radius + 4.0 {
                        self.hovered_key = Some(i);
                        break;
                    }
                }
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if !rect.contains(*position) {
                    return EventReply::Unhandled;
                }

                // Key click.
                for (i, _key) in self.keyframes.iter().enumerate() {
                    let kpos = self.key_screen_pos(i, rect);
                    if (*position - kpos).length() <= self.key_radius + 4.0 {
                        self.selected_key = Some(i);
                        self.dragging_key = Some(i);
                        return EventReply::Handled.then(EventReply::CaptureMouse);
                    }
                }

                // Tangent handle click (for selected key).
                if let Some(sel) = self.selected_key {
                    let key = &self.keyframes[sel];
                    let in_pos = self.world_to_screen(
                        Vec2::new(key.time + key.in_tangent.x, key.value + key.in_tangent.y),
                        rect,
                    );
                    if (*position - in_pos).length() <= 6.0 {
                        self.dragging_tangent = Some((sel, true));
                        return EventReply::Handled.then(EventReply::CaptureMouse);
                    }
                    let out_pos = self.world_to_screen(
                        Vec2::new(key.time + key.out_tangent.x, key.value + key.out_tangent.y),
                        rect,
                    );
                    if (*position - out_pos).length() <= 6.0 {
                        self.dragging_tangent = Some((sel, false));
                        return EventReply::Handled.then(EventReply::CaptureMouse);
                    }
                }

                self.selected_key = None;
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Right,
                ..
            } => {
                if rect.contains(*position) {
                    // Right-click: add keyframe.
                    let world = self.screen_to_world(*position, rect);
                    self.add_keyframe(world.x, world.y);
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Middle,
                ..
            } => {
                if rect.contains(*position) {
                    self.panning = true;
                    self.pan_start = *position;
                    self.pan_start_offset = self.view_offset;
                    return EventReply::Handled.then(EventReply::CaptureMouse);
                }
                EventReply::Unhandled
            }
            UIEvent::DragMove { position, .. } => {
                if self.panning {
                    let dx = (position.x - self.pan_start.x) / self.view_scale.x;
                    let dy = -(position.y - self.pan_start.y) / self.view_scale.y;
                    self.view_offset =
                        Vec2::new(self.pan_start_offset.x - dx, self.pan_start_offset.y - dy);
                    return EventReply::Handled;
                }
                if let Some(idx) = self.dragging_key {
                    let world = self.screen_to_world(*position, rect);
                    let key = &mut self.keyframes[idx];
                    key.time = world.x;
                    key.value = world.y;
                    self.value_changed = true;
                    return EventReply::Handled;
                }
                if let Some((idx, is_in)) = self.dragging_tangent {
                    let world = self.screen_to_world(*position, rect);
                    let key = &mut self.keyframes[idx];
                    let key_pos = Vec2::new(key.time, key.value);
                    let tangent = world - key_pos;
                    if is_in {
                        key.in_tangent = tangent;
                    } else {
                        key.out_tangent = tangent;
                    }
                    self.value_changed = true;
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::MouseUp { .. } | UIEvent::DragEnd { .. } => {
                let was_active =
                    self.panning || self.dragging_key.is_some() || self.dragging_tangent.is_some();
                self.panning = false;
                self.dragging_key = None;
                self.dragging_tangent = None;
                if was_active {
                    // Re-sort keyframes after dragging.
                    self.keyframes
                        .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
                    EventReply::Handled.then(EventReply::ReleaseMouse)
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::Scroll { delta, .. } => {
                // Zoom.
                let factor = 1.0 + delta.y * 0.1;
                self.view_scale.x = (self.view_scale.x * factor).clamp(10.0, 1000.0);
                self.view_scale.y = (self.view_scale.y * factor).clamp(10.0, 1000.0);
                EventReply::Handled
            }
            UIEvent::KeyInput {
                key: KeyCode::Delete,
                pressed: true,
                ..
            } => {
                self.remove_selected();
                EventReply::Handled
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 7. GradientEditor
// =========================================================================

/// A gradient stop with color and position.
#[derive(Debug, Clone, Copy)]
pub struct GradientColorStop {
    pub position: f32,
    pub color: Color,
}

/// Gradient editor with draggable stops.
#[derive(Debug, Clone)]
pub struct GradientEditor {
    pub id: UIId,
    pub stops: Vec<GradientColorStop>,
    pub selected_stop: Option<usize>,
    pub hovered_stop: Option<usize>,
    pub dragging_stop: Option<usize>,
    pub bar_height: f32,
    pub stop_size: f32,
    pub desired_width: f32,
    pub background_color: Color,
    pub border_color: Color,
    pub visible: bool,
    pub value_changed: bool,
}

impl GradientEditor {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            stops: vec![
                GradientColorStop { position: 0.0, color: Color::BLACK },
                GradientColorStop { position: 1.0, color: Color::WHITE },
            ],
            selected_stop: None,
            hovered_stop: None,
            dragging_stop: None,
            bar_height: 24.0,
            stop_size: 10.0,
            desired_width: 250.0,
            background_color: Color::from_hex("#2A2A2A"),
            border_color: Color::from_hex("#555555"),
            visible: true,
            value_changed: false,
        }
    }

    pub fn add_stop(&mut self, position: f32, color: Color) {
        self.stops.push(GradientColorStop {
            position: position.clamp(0.0, 1.0),
            color,
        });
        self.stops.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
        self.value_changed = true;
    }

    pub fn remove_stop(&mut self, index: usize) {
        if self.stops.len() > 2 && index < self.stops.len() {
            self.stops.remove(index);
            self.selected_stop = None;
            self.value_changed = true;
        }
    }

    fn gradient_bar_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.min.x + self.stop_size, rect.min.y + 4.0),
            Vec2::new(rect.max.x - self.stop_size, rect.min.y + 4.0 + self.bar_height),
        )
    }

    fn stop_screen_x(&self, index: usize, rect: Rect) -> f32 {
        let bar = self.gradient_bar_rect(rect);
        bar.min.x + self.stops[index].position * bar.width()
    }

    /// Sample the gradient at position t (0..1).
    pub fn sample(&self, t: f32) -> Color {
        if self.stops.is_empty() {
            return Color::BLACK;
        }
        if t <= self.stops[0].position {
            return self.stops[0].color;
        }
        let last = self.stops.len() - 1;
        if t >= self.stops[last].position {
            return self.stops[last].color;
        }
        for i in 0..last {
            if t >= self.stops[i].position && t <= self.stops[i + 1].position {
                let range = self.stops[i + 1].position - self.stops[i].position;
                if range < 1e-6 {
                    return self.stops[i].color;
                }
                let local_t = (t - self.stops[i].position) / range;
                return self.stops[i].color.lerp(self.stops[i + 1].color, local_t);
            }
        }
        self.stops[last].color
    }
}

impl SlateWidget for GradientEditor {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        Vec2::new(
            self.desired_width,
            self.bar_height + self.stop_size * 2.0 + 12.0,
        )
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        draw_list.draw_rounded_rect(
            rect,
            self.background_color,
            CornerRadii::all(4.0),
            BorderSpec::default(),
        );

        let bar = self.gradient_bar_rect(rect);

        // Gradient bar (rendered as segments).
        let segments = 64;
        let seg_w = bar.width() / segments as f32;
        for i in 0..segments {
            let t = (i as f32 + 0.5) / segments as f32;
            let c = self.sample(t);
            draw_list.draw_rect(
                Rect::new(
                    Vec2::new(bar.min.x + i as f32 * seg_w, bar.min.y),
                    Vec2::new(bar.min.x + (i + 1) as f32 * seg_w, bar.max.y),
                ),
                c,
            );
        }
        draw_list.draw_rounded_rect(
            bar,
            Color::TRANSPARENT,
            CornerRadii::all(2.0),
            BorderSpec::new(self.border_color, 1.0),
        );

        // Stop markers.
        for (i, stop) in self.stops.iter().enumerate() {
            let sx = self.stop_screen_x(i, rect);
            let sy = bar.max.y + 4.0;
            let selected = self.selected_stop == Some(i);
            let hovered = self.hovered_stop == Some(i);

            // Triangle marker.
            draw_list.push(DrawCommand::Triangle {
                p0: Vec2::new(sx, bar.max.y),
                p1: Vec2::new(sx - self.stop_size * 0.5, sy + self.stop_size),
                p2: Vec2::new(sx + self.stop_size * 0.5, sy + self.stop_size),
                color: if selected {
                    Color::from_hex("#FFAA00")
                } else if hovered {
                    Color::WHITE
                } else {
                    Color::from_hex("#CCCCCC")
                },
            });

            // Color swatch on the marker.
            draw_list.draw_rect(
                Rect::new(
                    Vec2::new(sx - 4.0, sy + 2.0),
                    Vec2::new(sx + 4.0, sy + self.stop_size - 2.0),
                ),
                stop.color,
            );
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        match event {
            UIEvent::Hover { position } => {
                self.hovered_stop = None;
                let bar = self.gradient_bar_rect(rect);
                for (i, _stop) in self.stops.iter().enumerate() {
                    let sx = self.stop_screen_x(i, rect);
                    let sy = bar.max.y + 4.0 + self.stop_size * 0.5;
                    if (position.x - sx).abs() < self.stop_size
                        && (position.y - sy).abs() < self.stop_size
                    {
                        self.hovered_stop = Some(i);
                        break;
                    }
                }
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                // Click on stop.
                let bar = self.gradient_bar_rect(rect);
                for (i, _stop) in self.stops.iter().enumerate() {
                    let sx = self.stop_screen_x(i, rect);
                    let sy = bar.max.y + 4.0 + self.stop_size * 0.5;
                    if (position.x - sx).abs() < self.stop_size
                        && (position.y - sy).abs() < self.stop_size
                    {
                        self.selected_stop = Some(i);
                        self.dragging_stop = Some(i);
                        return EventReply::Handled.then(EventReply::CaptureMouse);
                    }
                }

                // Click on bar -- add new stop.
                if bar.contains(*position) {
                    let t = ((position.x - bar.min.x) / bar.width()).clamp(0.0, 1.0);
                    let color = self.sample(t);
                    self.add_stop(t, color);
                    return EventReply::Handled;
                }

                EventReply::Unhandled
            }
            UIEvent::DragMove { position, .. } => {
                if let Some(idx) = self.dragging_stop {
                    let bar = self.gradient_bar_rect(rect);
                    let t = ((position.x - bar.min.x) / bar.width()).clamp(0.0, 1.0);
                    self.stops[idx].position = t;
                    self.value_changed = true;
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::MouseUp { .. } | UIEvent::DragEnd { .. } => {
                if self.dragging_stop.is_some() {
                    self.dragging_stop = None;
                    self.stops
                        .sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
                    return EventReply::Handled.then(EventReply::ReleaseMouse);
                }
                EventReply::Unhandled
            }
            UIEvent::KeyInput {
                key: KeyCode::Delete,
                pressed: true,
                ..
            } => {
                if let Some(idx) = self.selected_stop {
                    self.remove_stop(idx);
                }
                EventReply::Handled
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 8. BreadcrumbTrail
// =========================================================================

/// Clickable breadcrumb path segments.
#[derive(Debug, Clone)]
pub struct BreadcrumbTrail {
    pub id: UIId,
    pub segments: Vec<String>,
    pub hovered_segment: Option<usize>,
    pub clicked_segment: Option<usize>,
    pub font_size: f32,
    pub font_id: u32,
    pub text_color: Color,
    pub hovered_color: Color,
    pub separator_color: Color,
    pub separator: String,
    pub visible: bool,
    pub height: f32,
}

impl BreadcrumbTrail {
    pub fn new(segments: Vec<String>) -> Self {
        Self {
            id: UIId::INVALID,
            segments,
            hovered_segment: None,
            clicked_segment: None,
            font_size: 13.0,
            font_id: 0,
            text_color: Color::from_hex("#CCCCCC"),
            hovered_color: Color::from_hex("#4488FF"),
            separator_color: Color::from_hex("#666666"),
            separator: " > ".to_string(),
            visible: true,
            height: 22.0,
        }
    }

    pub fn take_clicked_segment(&mut self) -> Option<usize> {
        self.clicked_segment.take()
    }

    fn char_width(&self) -> f32 {
        self.font_size * 0.6
    }
}

impl SlateWidget for BreadcrumbTrail {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        let cw = self.char_width();
        let total: f32 = self
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let seg_w = s.len() as f32 * cw;
                if i > 0 {
                    seg_w + self.separator.len() as f32 * cw
                } else {
                    seg_w
                }
            })
            .sum();
        Vec2::new(total + 8.0, self.height)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        let cw = self.char_width();
        let ty = rect.min.y + (self.height - self.font_size) * 0.5;
        let mut x = rect.min.x + 4.0;

        for (i, seg) in self.segments.iter().enumerate() {
            if i > 0 {
                draw_list.push(DrawCommand::Text {
                    text: self.separator.clone(),
                    position: Vec2::new(x, ty),
                    font_size: self.font_size,
                    color: self.separator_color,
                    font_id: self.font_id,
                    max_width: None,
                    align: TextAlign::Left,
                    vertical_align: TextVerticalAlign::Top,
                });
                x += self.separator.len() as f32 * cw;
            }

            let color = if self.hovered_segment == Some(i) {
                self.hovered_color
            } else {
                self.text_color
            };

            draw_list.push(DrawCommand::Text {
                text: seg.clone(),
                position: Vec2::new(x, ty),
                font_size: self.font_size,
                color,
                font_id: self.font_id,
                max_width: None,
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            // Underline on hover.
            if self.hovered_segment == Some(i) {
                let w = seg.len() as f32 * cw;
                draw_list.draw_line(
                    Vec2::new(x, ty + self.font_size + 1.0),
                    Vec2::new(x + w, ty + self.font_size + 1.0),
                    self.hovered_color,
                    1.0,
                );
            }

            x += seg.len() as f32 * cw;
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        match event {
            UIEvent::Hover { position } => {
                if !rect.contains(*position) {
                    self.hovered_segment = None;
                    return EventReply::Unhandled;
                }

                let cw = self.char_width();
                let mut x = rect.min.x + 4.0;
                self.hovered_segment = None;

                for (i, seg) in self.segments.iter().enumerate() {
                    if i > 0 {
                        x += self.separator.len() as f32 * cw;
                    }
                    let seg_w = seg.len() as f32 * cw;
                    if position.x >= x && position.x < x + seg_w {
                        self.hovered_segment = Some(i);
                        break;
                    }
                    x += seg_w;
                }
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if self.hovered_segment.is_some() && rect.contains(*position) {
                    self.clicked_segment = self.hovered_segment;
                    EventReply::Handled
                } else {
                    EventReply::Unhandled
                }
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 9. NotificationList + NotificationItem
// =========================================================================

/// Notification completion state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationState {
    Pending,
    Success,
    Fail,
    Info,
}

/// A single notification item.
#[derive(Debug, Clone)]
pub struct NotificationItem {
    pub id: u64,
    pub title: String,
    pub message: String,
    pub state: NotificationState,
    pub progress: Option<f32>,
    pub lifetime: f32,
    pub elapsed: f32,
    pub fade_duration: f32,
    pub visible: bool,
    pub dismissed: bool,
    pub icon_color: Color,
}

impl NotificationItem {
    pub fn new(id: u64, title: &str, message: &str, state: NotificationState) -> Self {
        Self {
            id,
            title: title.to_string(),
            message: message.to_string(),
            state,
            progress: None,
            lifetime: 5.0,
            elapsed: 0.0,
            fade_duration: 0.5,
            visible: true,
            dismissed: false,
            icon_color: match state {
                NotificationState::Pending => Color::from_hex("#FFAA00"),
                NotificationState::Success => Color::from_hex("#44BB44"),
                NotificationState::Fail => Color::from_hex("#FF4444"),
                NotificationState::Info => Color::from_hex("#4488FF"),
            },
        }
    }

    pub fn opacity(&self) -> f32 {
        let remaining = self.lifetime - self.elapsed;
        if remaining < self.fade_duration {
            (remaining / self.fade_duration).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    pub fn is_expired(&self) -> bool {
        self.elapsed >= self.lifetime || self.dismissed
    }
}

/// Stacked toast notifications.
#[derive(Debug, Clone)]
pub struct NotificationList {
    pub id: UIId,
    pub notifications: Vec<NotificationItem>,
    pub item_height: f32,
    pub item_width: f32,
    pub spacing: f32,
    pub font_size: f32,
    pub font_id: u32,
    pub anchor_corner: NotificationAnchor,
    pub visible: bool,
    next_id: u64,
}

/// Which corner notifications appear from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationAnchor {
    TopRight,
    TopLeft,
    BottomRight,
    BottomLeft,
}

impl NotificationList {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            notifications: Vec::new(),
            item_height: 60.0,
            item_width: 300.0,
            spacing: 4.0,
            font_size: 12.0,
            font_id: 0,
            anchor_corner: NotificationAnchor::TopRight,
            visible: true,
            next_id: 1,
        }
    }

    pub fn push(&mut self, title: &str, message: &str, state: NotificationState) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.notifications
            .push(NotificationItem::new(id, title, message, state));
        id
    }

    pub fn tick(&mut self, dt: f32) {
        for n in &mut self.notifications {
            n.elapsed += dt;
        }
        self.notifications.retain(|n| !n.is_expired());
    }

    pub fn dismiss(&mut self, id: u64) {
        if let Some(n) = self.notifications.iter_mut().find(|n| n.id == id) {
            n.dismissed = true;
        }
    }
}

impl SlateWidget for NotificationList {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        let h = self.notifications.len() as f32 * (self.item_height + self.spacing);
        Vec2::new(self.item_width, h)
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        for (i, notif) in self.notifications.iter().enumerate() {
            let opacity = notif.opacity();
            if opacity <= 0.0 {
                continue;
            }

            let ny = match self.anchor_corner {
                NotificationAnchor::TopRight | NotificationAnchor::TopLeft => {
                    rect.min.y + i as f32 * (self.item_height + self.spacing)
                }
                NotificationAnchor::BottomRight | NotificationAnchor::BottomLeft => {
                    rect.max.y
                        - (i + 1) as f32 * (self.item_height + self.spacing)
                }
            };
            let nx = match self.anchor_corner {
                NotificationAnchor::TopRight | NotificationAnchor::BottomRight => {
                    rect.max.x - self.item_width
                }
                NotificationAnchor::TopLeft | NotificationAnchor::BottomLeft => rect.min.x,
            };

            let item_rect = Rect::new(
                Vec2::new(nx, ny),
                Vec2::new(nx + self.item_width, ny + self.item_height),
            );

            // Background.
            draw_list.draw_rounded_rect(
                item_rect,
                Color::from_hex("#2A2A2A").with_alpha(opacity * 0.95),
                CornerRadii::all(6.0),
                BorderSpec::new(notif.icon_color.with_alpha(opacity * 0.6), 1.0),
            );

            // State icon circle.
            let icon_x = item_rect.min.x + 16.0;
            let icon_y = item_rect.min.y + self.item_height * 0.5;
            draw_list.push(DrawCommand::Circle {
                center: Vec2::new(icon_x, icon_y),
                radius: 6.0,
                color: notif.icon_color.with_alpha(opacity),
                border: BorderSpec::default(),
            });

            // Title.
            draw_list.push(DrawCommand::Text {
                text: notif.title.clone(),
                position: Vec2::new(item_rect.min.x + 30.0, item_rect.min.y + 8.0),
                font_size: self.font_size,
                color: Color::WHITE.with_alpha(opacity),
                font_id: self.font_id,
                max_width: Some(self.item_width - 50.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            // Message.
            draw_list.push(DrawCommand::Text {
                text: notif.message.clone(),
                position: Vec2::new(item_rect.min.x + 30.0, item_rect.min.y + 24.0),
                font_size: self.font_size - 1.0,
                color: Color::from_hex("#BBBBBB").with_alpha(opacity),
                font_id: self.font_id,
                max_width: Some(self.item_width - 50.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            // Progress bar.
            if let Some(progress) = notif.progress {
                let pb_y = item_rect.max.y - 8.0;
                let pb_rect = Rect::new(
                    Vec2::new(item_rect.min.x + 30.0, pb_y - 3.0),
                    Vec2::new(item_rect.max.x - 8.0, pb_y),
                );
                draw_list.draw_rounded_rect(
                    pb_rect,
                    Color::from_hex("#444444").with_alpha(opacity),
                    CornerRadii::all(1.5),
                    BorderSpec::default(),
                );
                let fill_w = pb_rect.width() * progress.clamp(0.0, 1.0);
                if fill_w > 0.5 {
                    draw_list.draw_rounded_rect(
                        Rect::new(pb_rect.min, Vec2::new(pb_rect.min.x + fill_w, pb_rect.max.y)),
                        notif.icon_color.with_alpha(opacity),
                        CornerRadii::all(1.5),
                        BorderSpec::default(),
                    );
                }
            }

            // Close button (X).
            let close_x = item_rect.max.x - 16.0;
            let close_y = item_rect.min.y + 10.0;
            let cs = 4.0;
            draw_list.draw_line(
                Vec2::new(close_x - cs, close_y - cs),
                Vec2::new(close_x + cs, close_y + cs),
                Color::from_hex("#888888").with_alpha(opacity),
                1.0,
            );
            draw_list.draw_line(
                Vec2::new(close_x + cs, close_y - cs),
                Vec2::new(close_x - cs, close_y + cs),
                Color::from_hex("#888888").with_alpha(opacity),
                1.0,
            );
        }
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if let UIEvent::Click {
            position,
            button: MouseButton::Left,
            ..
        } = event
        {
            // Check close button clicks.
            for (i, notif) in self.notifications.iter_mut().enumerate() {
                let ny = match self.anchor_corner {
                    NotificationAnchor::TopRight | NotificationAnchor::TopLeft => {
                        rect.min.y + i as f32 * (self.item_height + self.spacing)
                    }
                    NotificationAnchor::BottomRight | NotificationAnchor::BottomLeft => {
                        rect.max.y - (i + 1) as f32 * (self.item_height + self.spacing)
                    }
                };
                let nx = match self.anchor_corner {
                    NotificationAnchor::TopRight | NotificationAnchor::BottomRight => {
                        rect.max.x - self.item_width
                    }
                    NotificationAnchor::TopLeft | NotificationAnchor::BottomLeft => rect.min.x,
                };

                let close_x = nx + self.item_width - 16.0;
                let close_y = ny + 10.0;
                if (position.x - close_x).abs() < 8.0 && (position.y - close_y).abs() < 8.0 {
                    notif.dismissed = true;
                    return EventReply::Handled;
                }
            }
        }
        EventReply::Unhandled
    }
}

// =========================================================================
// 10. Tooltip
// =========================================================================

/// A tooltip popup that shows rich widget content on hover.
#[derive(Debug, Clone)]
pub struct Tooltip {
    pub id: UIId,
    pub text: String,
    pub delay: f32,
    pub timer: f32,
    pub is_showing: bool,
    pub position: Vec2,
    pub max_width: f32,
    pub font_size: f32,
    pub font_id: u32,
    pub background_color: Color,
    pub text_color: Color,
    pub border_color: Color,
    pub padding: Padding,
    pub corner_radii: CornerRadii,
}

impl Tooltip {
    pub fn new(text: &str) -> Self {
        Self {
            id: UIId::INVALID,
            text: text.to_string(),
            delay: 0.5,
            timer: 0.0,
            is_showing: false,
            position: Vec2::ZERO,
            max_width: 250.0,
            font_size: 12.0,
            font_id: 0,
            background_color: Color::from_hex("#1A1A1A"),
            text_color: Color::from_hex("#EEEEEE"),
            border_color: Color::from_hex("#555555"),
            padding: Padding::new(8.0, 6.0, 8.0, 6.0),
            corner_radii: CornerRadii::all(4.0),
        }
    }

    pub fn tick(&mut self, dt: f32, hovering: bool) {
        if hovering {
            self.timer += dt;
            if self.timer >= self.delay {
                self.is_showing = true;
            }
        } else {
            self.timer = 0.0;
            self.is_showing = false;
        }
    }

    pub fn set_position(&mut self, pos: Vec2) {
        self.position = Vec2::new(pos.x + 12.0, pos.y + 16.0);
    }
}

impl SlateWidget for Tooltip {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        let cw = self.font_size * 0.6;
        let text_w = (self.text.len() as f32 * cw).min(self.max_width);
        let text_h = self.font_size * 1.2;
        Vec2::new(
            text_w + self.padding.left + self.padding.right,
            text_h + self.padding.top + self.padding.bottom,
        )
    }

    fn paint(&self, draw_list: &mut DrawList, _rect: Rect) {
        if !self.is_showing {
            return;
        }

        let size = self.compute_desired_size(None);
        let tip_rect = Rect::new(self.position, self.position + size);

        draw_list.draw_rounded_rect_with_shadow(
            tip_rect,
            self.background_color,
            self.corner_radii,
            BorderSpec::new(self.border_color, 1.0),
            Shadow::new(
                Color::new(0.0, 0.0, 0.0, 0.3),
                Vec2::new(2.0, 2.0),
                4.0,
                0.0,
            ),
        );

        draw_list.push(DrawCommand::Text {
            text: self.text.clone(),
            position: Vec2::new(
                self.position.x + self.padding.left,
                self.position.y + self.padding.top,
            ),
            font_size: self.font_size,
            color: self.text_color,
            font_id: self.font_id,
            max_width: Some(self.max_width),
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });
    }

    fn handle_event(&mut self, _event: &UIEvent, _rect: Rect) -> EventReply {
        EventReply::Unhandled
    }
}

// =========================================================================
// 11. MenuAnchor
// =========================================================================

/// Popup menu positioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuPlacement {
    Below,
    Above,
    Right,
    Left,
}

/// Menu item.
#[derive(Debug, Clone)]
pub struct MenuItem {
    pub label: String,
    pub shortcut: String,
    pub icon: Option<TextureId>,
    pub enabled: bool,
    pub is_separator: bool,
    pub submenu: Vec<MenuItem>,
    pub checked: bool,
    pub data: u64,
}

impl MenuItem {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            shortcut: String::new(),
            icon: None,
            enabled: true,
            is_separator: false,
            submenu: Vec::new(),
            checked: false,
            data: 0,
        }
    }

    pub fn separator() -> Self {
        Self {
            label: String::new(),
            shortcut: String::new(),
            icon: None,
            enabled: false,
            is_separator: true,
            submenu: Vec::new(),
            checked: false,
            data: 0,
        }
    }

    pub fn with_shortcut(mut self, s: &str) -> Self {
        self.shortcut = s.to_string();
        self
    }

    pub fn with_submenu(mut self, items: Vec<MenuItem>) -> Self {
        self.submenu = items;
        self
    }
}

/// Popup menu anchor.
#[derive(Debug, Clone)]
pub struct MenuAnchor {
    pub id: UIId,
    pub items: Vec<MenuItem>,
    pub is_open: bool,
    pub placement: MenuPlacement,
    pub anchor_rect: Rect,
    pub hovered_index: Option<usize>,
    pub clicked_index: Option<usize>,
    pub item_height: f32,
    pub separator_height: f32,
    pub min_width: f32,
    pub font_size: f32,
    pub font_id: u32,
    pub background_color: Color,
    pub text_color: Color,
    pub disabled_text_color: Color,
    pub hovered_bg: Color,
    pub border_color: Color,
    pub shortcut_color: Color,
    pub check_color: Color,
    pub visible: bool,
}

impl MenuAnchor {
    pub fn new(items: Vec<MenuItem>) -> Self {
        Self {
            id: UIId::INVALID,
            items,
            is_open: false,
            placement: MenuPlacement::Below,
            anchor_rect: Rect::new(Vec2::ZERO, Vec2::ZERO),
            hovered_index: None,
            clicked_index: None,
            item_height: 24.0,
            separator_height: 8.0,
            min_width: 150.0,
            font_size: 13.0,
            font_id: 0,
            background_color: Color::from_hex("#2A2A2A"),
            text_color: Color::WHITE,
            disabled_text_color: Color::from_hex("#666666"),
            hovered_bg: Color::from_hex("#3A3A3A"),
            border_color: Color::from_hex("#555555"),
            shortcut_color: Color::from_hex("#888888"),
            check_color: Color::from_hex("#4488FF"),
            visible: true,
        }
    }

    pub fn open(&mut self, anchor: Rect) {
        self.anchor_rect = anchor;
        self.is_open = true;
        self.hovered_index = None;
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.hovered_index = None;
    }

    pub fn take_clicked_index(&mut self) -> Option<usize> {
        self.clicked_index.take()
    }

    fn menu_rect(&self) -> Rect {
        let total_h: f32 = self
            .items
            .iter()
            .map(|i| {
                if i.is_separator {
                    self.separator_height
                } else {
                    self.item_height
                }
            })
            .sum();

        let width = self.min_width;

        match self.placement {
            MenuPlacement::Below => Rect::new(
                Vec2::new(self.anchor_rect.min.x, self.anchor_rect.max.y),
                Vec2::new(self.anchor_rect.min.x + width, self.anchor_rect.max.y + total_h),
            ),
            MenuPlacement::Above => Rect::new(
                Vec2::new(self.anchor_rect.min.x, self.anchor_rect.min.y - total_h),
                Vec2::new(self.anchor_rect.min.x + width, self.anchor_rect.min.y),
            ),
            MenuPlacement::Right => Rect::new(
                Vec2::new(self.anchor_rect.max.x, self.anchor_rect.min.y),
                Vec2::new(self.anchor_rect.max.x + width, self.anchor_rect.min.y + total_h),
            ),
            MenuPlacement::Left => Rect::new(
                Vec2::new(self.anchor_rect.min.x - width, self.anchor_rect.min.y),
                Vec2::new(self.anchor_rect.min.x, self.anchor_rect.min.y + total_h),
            ),
        }
    }
}

impl SlateWidget for MenuAnchor {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        Vec2::ZERO
    }

    fn paint(&self, draw_list: &mut DrawList, _rect: Rect) {
        if !self.is_open || !self.visible {
            return;
        }

        let menu_rect = self.menu_rect();
        draw_list.draw_rounded_rect_with_shadow(
            menu_rect,
            self.background_color,
            CornerRadii::all(4.0),
            BorderSpec::new(self.border_color, 1.0),
            Shadow::new(
                Color::new(0.0, 0.0, 0.0, 0.4),
                Vec2::new(2.0, 4.0),
                8.0,
                0.0,
            ),
        );

        let mut y = menu_rect.min.y;
        for (i, item) in self.items.iter().enumerate() {
            if item.is_separator {
                let mid_y = y + self.separator_height * 0.5;
                draw_list.draw_line(
                    Vec2::new(menu_rect.min.x + 8.0, mid_y),
                    Vec2::new(menu_rect.max.x - 8.0, mid_y),
                    self.border_color,
                    1.0,
                );
                y += self.separator_height;
                continue;
            }

            let item_rect = Rect::new(
                Vec2::new(menu_rect.min.x, y),
                Vec2::new(menu_rect.max.x, y + self.item_height),
            );

            // Hover highlight.
            if self.hovered_index == Some(i) && item.enabled {
                draw_list.draw_rect(item_rect, self.hovered_bg);
            }

            // Check mark.
            if item.checked {
                let cx = menu_rect.min.x + 14.0;
                let cy = y + self.item_height * 0.5;
                let s = 4.0;
                draw_list.draw_line(
                    Vec2::new(cx - s, cy),
                    Vec2::new(cx - s * 0.3, cy + s * 0.7),
                    self.check_color,
                    2.0,
                );
                draw_list.draw_line(
                    Vec2::new(cx - s * 0.3, cy + s * 0.7),
                    Vec2::new(cx + s, cy - s * 0.5),
                    self.check_color,
                    2.0,
                );
            }

            // Label.
            let ty = y + (self.item_height - self.font_size) * 0.5;
            let tc = if item.enabled {
                self.text_color
            } else {
                self.disabled_text_color
            };
            draw_list.push(DrawCommand::Text {
                text: item.label.clone(),
                position: Vec2::new(menu_rect.min.x + 28.0, ty),
                font_size: self.font_size,
                color: tc,
                font_id: self.font_id,
                max_width: Some(menu_rect.width() - 80.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            // Shortcut.
            if !item.shortcut.is_empty() {
                let sw = item.shortcut.len() as f32 * self.font_size * 0.6;
                draw_list.push(DrawCommand::Text {
                    text: item.shortcut.clone(),
                    position: Vec2::new(menu_rect.max.x - sw - 8.0, ty),
                    font_size: self.font_size,
                    color: self.shortcut_color,
                    font_id: self.font_id,
                    max_width: None,
                    align: TextAlign::Right,
                    vertical_align: TextVerticalAlign::Top,
                });
            }

            // Submenu arrow.
            if !item.submenu.is_empty() {
                let ax = menu_rect.max.x - 12.0;
                let ay = y + self.item_height * 0.5;
                draw_list.push(DrawCommand::Triangle {
                    p0: Vec2::new(ax + 3.0, ay),
                    p1: Vec2::new(ax - 3.0, ay - 4.0),
                    p2: Vec2::new(ax - 3.0, ay + 4.0),
                    color: tc,
                });
            }

            y += self.item_height;
        }
    }

    fn handle_event(&mut self, event: &UIEvent, _rect: Rect) -> EventReply {
        if !self.is_open {
            return EventReply::Unhandled;
        }

        let menu_rect = self.menu_rect();

        match event {
            UIEvent::Hover { position } => {
                if !menu_rect.contains(*position) {
                    self.hovered_index = None;
                    return EventReply::Unhandled;
                }

                let mut y = menu_rect.min.y;
                self.hovered_index = None;
                for (i, item) in self.items.iter().enumerate() {
                    let h = if item.is_separator {
                        self.separator_height
                    } else {
                        self.item_height
                    };
                    if position.y >= y && position.y < y + h && !item.is_separator {
                        self.hovered_index = Some(i);
                        break;
                    }
                    y += h;
                }
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                ..
            } => {
                if menu_rect.contains(*position) {
                    if let Some(idx) = self.hovered_index {
                        if idx < self.items.len() && self.items[idx].enabled {
                            self.clicked_index = Some(idx);
                            self.is_open = false;
                            return EventReply::Handled;
                        }
                    }
                    EventReply::Handled
                } else {
                    self.close();
                    EventReply::Unhandled
                }
            }
            UIEvent::KeyInput {
                key: KeyCode::Escape,
                pressed: true,
                ..
            } => {
                self.close();
                EventReply::Handled
            }
            _ => EventReply::Unhandled,
        }
    }
}

// =========================================================================
// 12. GraphEditor (Node Graph)
// =========================================================================

/// Pin type determines colouring and connection compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PinType {
    Float,
    Vector,
    Color,
    Bool,
    Object,
    Exec,
    Custom(u32),
}

impl PinType {
    pub fn default_color(&self) -> Color {
        match self {
            PinType::Float => Color::from_hex("#00CC00"),
            PinType::Vector => Color::from_hex("#FFCC00"),
            PinType::Color => Color::from_hex("#FF4444"),
            PinType::Bool => Color::from_hex("#CC0000"),
            PinType::Object => Color::from_hex("#0088FF"),
            PinType::Exec => Color::WHITE,
            PinType::Custom(_) => Color::from_hex("#AA66CC"),
        }
    }
}

/// Direction of a pin (input or output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinDirection {
    Input,
    Output,
}

/// A connection point on a graph node.
#[derive(Debug, Clone)]
pub struct GraphPin {
    pub id: u64,
    pub name: String,
    pub pin_type: PinType,
    pub direction: PinDirection,
    pub hovered: bool,
    pub connected: bool,
    pub color_override: Option<Color>,
}

impl GraphPin {
    pub fn new(id: u64, name: &str, pin_type: PinType, direction: PinDirection) -> Self {
        Self {
            id,
            name: name.to_string(),
            pin_type,
            direction,
            hovered: false,
            connected: false,
            color_override: None,
        }
    }

    pub fn color(&self) -> Color {
        self.color_override.unwrap_or_else(|| self.pin_type.default_color())
    }
}

/// A node in the graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: u64,
    pub title: String,
    pub position: Vec2,
    pub size: Vec2,
    pub input_pins: Vec<GraphPin>,
    pub output_pins: Vec<GraphPin>,
    pub title_color: Color,
    pub body_color: Color,
    pub selected: bool,
    pub hovered: bool,
    pub collapsed: bool,
    pub comment: String,
    pub error: bool,
}

impl GraphNode {
    pub fn new(id: u64, title: &str, position: Vec2) -> Self {
        Self {
            id,
            title: title.to_string(),
            position,
            size: Vec2::new(160.0, 100.0),
            input_pins: Vec::new(),
            output_pins: Vec::new(),
            title_color: Color::from_hex("#2266CC"),
            body_color: Color::from_hex("#1E1E1E"),
            selected: false,
            hovered: false,
            collapsed: false,
            comment: String::new(),
            error: false,
        }
    }

    pub fn add_input(&mut self, pin: GraphPin) {
        self.input_pins.push(pin);
    }

    pub fn add_output(&mut self, pin: GraphPin) {
        self.output_pins.push(pin);
    }

    fn compute_size(&self, font_size: f32, pin_spacing: f32, title_height: f32) -> Vec2 {
        let pin_count = self.input_pins.len().max(self.output_pins.len());
        let body_h = pin_count as f32 * pin_spacing + 8.0;
        let width = 160.0f32;
        Vec2::new(width, title_height + body_h)
    }

    fn pin_position(
        &self,
        pin_id: u64,
        font_size: f32,
        pin_spacing: f32,
        title_height: f32,
    ) -> Option<Vec2> {
        for (i, pin) in self.input_pins.iter().enumerate() {
            if pin.id == pin_id {
                return Some(Vec2::new(
                    self.position.x,
                    self.position.y + title_height + pin_spacing * 0.5 + i as f32 * pin_spacing,
                ));
            }
        }
        for (i, pin) in self.output_pins.iter().enumerate() {
            if pin.id == pin_id {
                let w = self.compute_size(font_size, pin_spacing, title_height).x;
                return Some(Vec2::new(
                    self.position.x + w,
                    self.position.y + title_height + pin_spacing * 0.5 + i as f32 * pin_spacing,
                ));
            }
        }
        None
    }
}

/// A wire connecting two pins.
#[derive(Debug, Clone)]
pub struct GraphWire {
    pub from_node: u64,
    pub from_pin: u64,
    pub to_node: u64,
    pub to_pin: u64,
    pub color: Color,
}

/// Node graph editor canvas with zoom/pan, nodes, wires, and selection.
#[derive(Debug, Clone)]
pub struct GraphEditor {
    pub id: UIId,
    pub nodes: Vec<GraphNode>,
    pub wires: Vec<GraphWire>,
    pub view_offset: Vec2,
    pub view_zoom: f32,
    pub background_color: Color,
    pub grid_color: Color,
    pub wire_color: Color,
    pub wire_thickness: f32,
    pub pin_radius: f32,
    pub pin_spacing: f32,
    pub title_height: f32,
    pub font_size: f32,
    pub font_id: u32,
    pub desired_size: Vec2,
    pub panning: bool,
    pub pan_start: Vec2,
    pub pan_start_offset: Vec2,
    pub dragging_node: Option<u64>,
    pub drag_node_start: Vec2,
    pub creating_wire: Option<(u64, u64)>,
    pub creating_wire_end: Vec2,
    pub selection_box: Option<(Vec2, Vec2)>,
    pub border_color: Color,
    pub visible: bool,
    pub next_wire_id: u64,
}

impl GraphEditor {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            nodes: Vec::new(),
            wires: Vec::new(),
            view_offset: Vec2::ZERO,
            view_zoom: 1.0,
            background_color: Color::from_hex("#1A1A1A"),
            grid_color: Color::from_hex("#252525"),
            wire_color: Color::from_hex("#AAAAAA"),
            wire_thickness: 2.0,
            pin_radius: 5.0,
            pin_spacing: 22.0,
            title_height: 26.0,
            font_size: 12.0,
            font_id: 0,
            desired_size: Vec2::new(600.0, 400.0),
            panning: false,
            pan_start: Vec2::ZERO,
            pan_start_offset: Vec2::ZERO,
            dragging_node: None,
            drag_node_start: Vec2::ZERO,
            creating_wire: None,
            creating_wire_end: Vec2::ZERO,
            selection_box: None,
            border_color: Color::from_hex("#444444"),
            visible: true,
            next_wire_id: 1,
        }
    }

    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.push(node);
    }

    pub fn add_wire(&mut self, from_node: u64, from_pin: u64, to_node: u64, to_pin: u64) {
        let color = self
            .find_pin(from_node, from_pin)
            .map(|p| p.color())
            .unwrap_or(self.wire_color);
        self.wires.push(GraphWire {
            from_node,
            from_pin,
            to_node,
            to_pin,
            color,
        });
    }

    fn find_pin(&self, node_id: u64, pin_id: u64) -> Option<&GraphPin> {
        for node in &self.nodes {
            if node.id == node_id {
                for pin in &node.input_pins {
                    if pin.id == pin_id {
                        return Some(pin);
                    }
                }
                for pin in &node.output_pins {
                    if pin.id == pin_id {
                        return Some(pin);
                    }
                }
            }
        }
        None
    }

    fn world_to_screen(&self, world: Vec2, rect: Rect) -> Vec2 {
        Vec2::new(
            rect.min.x + (world.x - self.view_offset.x) * self.view_zoom,
            rect.min.y + (world.y - self.view_offset.y) * self.view_zoom,
        )
    }

    fn screen_to_world(&self, screen: Vec2, rect: Rect) -> Vec2 {
        Vec2::new(
            (screen.x - rect.min.x) / self.view_zoom + self.view_offset.x,
            (screen.y - rect.min.y) / self.view_zoom + self.view_offset.y,
        )
    }

    fn node_screen_rect(&self, node: &GraphNode, rect: Rect) -> Rect {
        let pos = self.world_to_screen(node.position, rect);
        let size = node.compute_size(self.font_size, self.pin_spacing, self.title_height)
            * self.view_zoom;
        Rect::new(pos, pos + size)
    }

    fn pin_screen_pos(&self, node: &GraphNode, pin_id: u64, rect: Rect) -> Option<Vec2> {
        if let Some(world) =
            node.pin_position(pin_id, self.font_size, self.pin_spacing, self.title_height)
        {
            Some(self.world_to_screen(world, rect))
        } else {
            None
        }
    }

    /// Draw a spline wire between two points.
    fn draw_wire(draw_list: &mut DrawList, start: Vec2, end: Vec2, color: Color, thickness: f32) {
        let dx = (end.x - start.x).abs().max(30.0);
        let cp1 = Vec2::new(start.x + dx * 0.5, start.y);
        let cp2 = Vec2::new(end.x - dx * 0.5, end.y);

        let segments = 32;
        let mut points = Vec::with_capacity(segments + 1);
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            let it = 1.0 - t;
            let x = it * it * it * start.x
                + 3.0 * it * it * t * cp1.x
                + 3.0 * it * t * t * cp2.x
                + t * t * t * end.x;
            let y = it * it * it * start.y
                + 3.0 * it * it * t * cp1.y
                + 3.0 * it * t * t * cp2.y
                + t * t * t * end.y;
            points.push(Vec2::new(x, y));
        }

        draw_list.push(DrawCommand::Polyline {
            points,
            color,
            thickness,
            closed: false,
        });
    }
}

impl SlateWidget for GraphEditor {
    fn compute_desired_size(&self, _max_width: Option<f32>) -> Vec2 {
        self.desired_size
    }

    fn paint(&self, draw_list: &mut DrawList, rect: Rect) {
        if !self.visible {
            return;
        }

        draw_list.draw_rounded_rect(
            rect,
            self.background_color,
            CornerRadii::all(4.0),
            BorderSpec::new(self.border_color, 1.0),
        );

        draw_list.push_clip(rect);

        // Grid.
        let grid_spacing = 40.0 * self.view_zoom;
        if grid_spacing > 4.0 {
            let start_x = rect.min.x
                - ((self.view_offset.x * self.view_zoom) % grid_spacing)
                    .rem_euclid(grid_spacing);
            let start_y = rect.min.y
                - ((self.view_offset.y * self.view_zoom) % grid_spacing)
                    .rem_euclid(grid_spacing);

            let mut gx = start_x;
            while gx < rect.max.x {
                draw_list.draw_line(
                    Vec2::new(gx, rect.min.y),
                    Vec2::new(gx, rect.max.y),
                    self.grid_color,
                    1.0,
                );
                gx += grid_spacing;
            }
            let mut gy = start_y;
            while gy < rect.max.y {
                draw_list.draw_line(
                    Vec2::new(rect.min.x, gy),
                    Vec2::new(rect.max.x, gy),
                    self.grid_color,
                    1.0,
                );
                gy += grid_spacing;
            }
        }

        // Wires.
        for wire in &self.wires {
            let from_node = self.nodes.iter().find(|n| n.id == wire.from_node);
            let to_node = self.nodes.iter().find(|n| n.id == wire.to_node);
            if let (Some(fn_node), Some(tn_node)) = (from_node, to_node) {
                if let (Some(start), Some(end)) = (
                    self.pin_screen_pos(fn_node, wire.from_pin, rect),
                    self.pin_screen_pos(tn_node, wire.to_pin, rect),
                ) {
                    Self::draw_wire(draw_list, start, end, wire.color, self.wire_thickness);
                }
            }
        }

        // Wire being created.
        if let Some((_node_id, _pin_id)) = self.creating_wire {
            // Find start position.
            for node in &self.nodes {
                if node.id == _node_id {
                    if let Some(start) = self.pin_screen_pos(node, _pin_id, rect) {
                        Self::draw_wire(
                            draw_list,
                            start,
                            self.creating_wire_end,
                            Color::from_hex("#FFAA00"),
                            self.wire_thickness,
                        );
                    }
                }
            }
        }

        // Nodes.
        for node in &self.nodes {
            let node_rect = self.node_screen_rect(node, rect);

            // Skip if not visible.
            if node_rect.max.x < rect.min.x
                || node_rect.min.x > rect.max.x
                || node_rect.max.y < rect.min.y
                || node_rect.min.y > rect.max.y
            {
                continue;
            }

            let title_rect = Rect::new(
                node_rect.min,
                Vec2::new(node_rect.max.x, node_rect.min.y + self.title_height * self.view_zoom),
            );

            // Node body.
            draw_list.draw_rounded_rect(
                node_rect,
                node.body_color,
                CornerRadii::all(6.0),
                if node.selected {
                    BorderSpec::new(Color::from_hex("#FFAA00"), 2.0)
                } else if node.error {
                    BorderSpec::new(Color::from_hex("#FF4444"), 2.0)
                } else {
                    BorderSpec::new(Color::from_hex("#444444"), 1.0)
                },
            );

            // Title bar.
            draw_list.draw_rounded_rect(
                title_rect,
                node.title_color,
                CornerRadii::new(6.0, 6.0, 0.0, 0.0),
                BorderSpec::default(),
            );

            // Title text.
            let fs = self.font_size * self.view_zoom;
            let ty = title_rect.min.y + (title_rect.height() - fs) * 0.5;
            draw_list.push(DrawCommand::Text {
                text: node.title.clone(),
                position: Vec2::new(title_rect.min.x + 8.0 * self.view_zoom, ty),
                font_size: fs,
                color: Color::WHITE,
                font_id: self.font_id,
                max_width: Some(title_rect.width() - 16.0 * self.view_zoom),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });

            // Input pins.
            let pin_fs = (self.font_size - 1.0) * self.view_zoom;
            for (i, pin) in node.input_pins.iter().enumerate() {
                let py = node_rect.min.y
                    + self.title_height * self.view_zoom
                    + self.pin_spacing * self.view_zoom * 0.5
                    + i as f32 * self.pin_spacing * self.view_zoom;
                let px = node_rect.min.x;

                let pin_c = if pin.hovered {
                    pin.color().lighten(0.3)
                } else {
                    pin.color()
                };

                draw_list.push(DrawCommand::Circle {
                    center: Vec2::new(px, py),
                    radius: self.pin_radius * self.view_zoom,
                    color: if pin.connected { pin_c } else { Color::TRANSPARENT },
                    border: BorderSpec::new(pin_c, 1.5),
                });

                draw_list.push(DrawCommand::Text {
                    text: pin.name.clone(),
                    position: Vec2::new(
                        px + (self.pin_radius + 4.0) * self.view_zoom,
                        py - pin_fs * 0.5,
                    ),
                    font_size: pin_fs,
                    color: Color::from_hex("#CCCCCC"),
                    font_id: self.font_id,
                    max_width: None,
                    align: TextAlign::Left,
                    vertical_align: TextVerticalAlign::Top,
                });
            }

            // Output pins.
            for (i, pin) in node.output_pins.iter().enumerate() {
                let py = node_rect.min.y
                    + self.title_height * self.view_zoom
                    + self.pin_spacing * self.view_zoom * 0.5
                    + i as f32 * self.pin_spacing * self.view_zoom;
                let px = node_rect.max.x;

                let pin_c = if pin.hovered {
                    pin.color().lighten(0.3)
                } else {
                    pin.color()
                };

                draw_list.push(DrawCommand::Circle {
                    center: Vec2::new(px, py),
                    radius: self.pin_radius * self.view_zoom,
                    color: if pin.connected { pin_c } else { Color::TRANSPARENT },
                    border: BorderSpec::new(pin_c, 1.5),
                });

                let name_w = pin.name.len() as f32 * pin_fs * 0.6;
                draw_list.push(DrawCommand::Text {
                    text: pin.name.clone(),
                    position: Vec2::new(
                        px - (self.pin_radius + 4.0) * self.view_zoom - name_w,
                        py - pin_fs * 0.5,
                    ),
                    font_size: pin_fs,
                    color: Color::from_hex("#CCCCCC"),
                    font_id: self.font_id,
                    max_width: None,
                    align: TextAlign::Left,
                    vertical_align: TextVerticalAlign::Top,
                });
            }
        }

        // Selection box.
        if let Some((start, end)) = self.selection_box {
            let sel_rect = Rect::new(
                Vec2::new(start.x.min(end.x), start.y.min(end.y)),
                Vec2::new(start.x.max(end.x), start.y.max(end.y)),
            );
            draw_list.draw_rounded_rect(
                sel_rect,
                Color::new(0.3, 0.5, 0.8, 0.15),
                CornerRadii::ZERO,
                BorderSpec::new(Color::new(0.3, 0.5, 0.8, 0.6), 1.0),
            );
        }

        draw_list.pop_clip();
    }

    fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        match event {
            UIEvent::Hover { position } => {
                if !rect.contains(*position) {
                    return EventReply::Unhandled;
                }
                // Update pin hover state.
                let world_pos = self.screen_to_world(*position, rect);
                for node in &mut self.nodes {
                    let nr = Rect::new(
                        node.position,
                        node.position
                            + node.compute_size(self.font_size, self.pin_spacing, self.title_height),
                    );
                    node.hovered = nr.contains(world_pos);
                    for pin in &mut node.input_pins {
                        if let Some(pp) =
                            node.pin_position(pin.id, self.font_size, self.pin_spacing, self.title_height)
                        {
                            pin.hovered = (world_pos - pp).length() < self.pin_radius + 4.0;
                        }
                    }
                    for pin in &mut node.output_pins {
                        if let Some(pp) =
                            node.pin_position(pin.id, self.font_size, self.pin_spacing, self.title_height)
                        {
                            pin.hovered = (world_pos - pp).length() < self.pin_radius + 4.0;
                        }
                    }
                }
                EventReply::Handled
            }
            UIEvent::Click {
                position,
                button: MouseButton::Left,
                modifiers,
            } => {
                if !rect.contains(*position) {
                    return EventReply::Unhandled;
                }

                let world_pos = self.screen_to_world(*position, rect);

                // Check pin clicks first (for wire creation).
                for node in &self.nodes {
                    for pin in node.output_pins.iter().chain(node.input_pins.iter()) {
                        if let Some(pp) =
                            node.pin_position(pin.id, self.font_size, self.pin_spacing, self.title_height)
                        {
                            if (world_pos - pp).length() < self.pin_radius + 4.0 {
                                self.creating_wire = Some((node.id, pin.id));
                                self.creating_wire_end = *position;
                                return EventReply::Handled.then(EventReply::CaptureMouse);
                            }
                        }
                    }
                }

                // Check node title click (for dragging).
                for node in self.nodes.iter_mut().rev() {
                    let nr = Rect::new(
                        node.position,
                        node.position
                            + node.compute_size(
                                self.font_size,
                                self.pin_spacing,
                                self.title_height,
                            ),
                    );
                    if nr.contains(world_pos) {
                        if !modifiers.ctrl {
                            for n in &mut self.nodes {
                                n.selected = false;
                            }
                        }
                        // Must break and re-find because of borrow checker.
                        let node_id = node.id;
                        let node_pos = node.position;
                        drop(node);
                        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == node_id) {
                            n.selected = true;
                        }
                        self.dragging_node = Some(node_id);
                        self.drag_node_start = world_pos - node_pos;
                        return EventReply::Handled.then(EventReply::CaptureMouse);
                    }
                }

                // Background click -- start selection box.
                if !modifiers.ctrl {
                    for node in &mut self.nodes {
                        node.selected = false;
                    }
                }
                self.selection_box = Some((*position, *position));
                EventReply::Handled.then(EventReply::CaptureMouse)
            }
            UIEvent::Click {
                position,
                button: MouseButton::Middle,
                ..
            } => {
                if rect.contains(*position) {
                    self.panning = true;
                    self.pan_start = *position;
                    self.pan_start_offset = self.view_offset;
                    EventReply::Handled.then(EventReply::CaptureMouse)
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::DragMove { position, .. } => {
                if self.panning {
                    let dx = (position.x - self.pan_start.x) / self.view_zoom;
                    let dy = (position.y - self.pan_start.y) / self.view_zoom;
                    self.view_offset =
                        Vec2::new(self.pan_start_offset.x - dx, self.pan_start_offset.y - dy);
                    return EventReply::Handled;
                }
                if let Some(node_id) = self.dragging_node {
                    let world_pos = self.screen_to_world(*position, rect);
                    if let Some(node) = self.nodes.iter_mut().find(|n| n.id == node_id) {
                        node.position = world_pos - self.drag_node_start;
                    }
                    return EventReply::Handled;
                }
                if self.creating_wire.is_some() {
                    self.creating_wire_end = *position;
                    return EventReply::Handled;
                }
                if let Some((start, ref mut end)) = self.selection_box {
                    *end = *position;
                    return EventReply::Handled;
                }
                EventReply::Unhandled
            }
            UIEvent::MouseUp { .. } | UIEvent::DragEnd { .. } => {
                let mut was_active = false;

                if self.panning {
                    self.panning = false;
                    was_active = true;
                }

                if let Some((_from_node, _from_pin)) = self.creating_wire.take() {
                    // Check if we landed on a compatible pin.
                    if let UIEvent::MouseUp { position, .. } | UIEvent::DragEnd { position } =
                        event
                    {
                        let world_pos = self.screen_to_world(*position, rect);
                        for node in &self.nodes {
                            for pin in node.input_pins.iter().chain(node.output_pins.iter()) {
                                if let Some(pp) = node.pin_position(
                                    pin.id,
                                    self.font_size,
                                    self.pin_spacing,
                                    self.title_height,
                                ) {
                                    if (world_pos - pp).length() < self.pin_radius + 4.0
                                        && node.id != _from_node
                                    {
                                        self.add_wire(_from_node, _from_pin, node.id, pin.id);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    was_active = true;
                }

                if self.dragging_node.is_some() {
                    self.dragging_node = None;
                    was_active = true;
                }

                if let Some((start, end)) = self.selection_box.take() {
                    // Select nodes within box.
                    let sel_rect = Rect::new(
                        Vec2::new(start.x.min(end.x), start.y.min(end.y)),
                        Vec2::new(start.x.max(end.x), start.y.max(end.y)),
                    );
                    for node in &mut self.nodes {
                        let nr = self.node_screen_rect(node, rect);
                        // Check intersection.
                        if nr.min.x < sel_rect.max.x
                            && nr.max.x > sel_rect.min.x
                            && nr.min.y < sel_rect.max.y
                            && nr.max.y > sel_rect.min.y
                        {
                            node.selected = true;
                        }
                    }
                    was_active = true;
                }

                if was_active {
                    EventReply::Handled.then(EventReply::ReleaseMouse)
                } else {
                    EventReply::Unhandled
                }
            }
            UIEvent::Scroll { delta, .. } => {
                let factor = 1.0 + delta.y * 0.1;
                self.view_zoom = (self.view_zoom * factor).clamp(0.1, 4.0);
                EventReply::Handled
            }
            UIEvent::KeyInput {
                key: KeyCode::Delete,
                pressed: true,
                ..
            } => {
                // Delete selected nodes.
                let selected_ids: Vec<u64> = self
                    .nodes
                    .iter()
                    .filter(|n| n.selected)
                    .map(|n| n.id)
                    .collect();
                self.nodes.retain(|n| !n.selected);
                // Remove wires connected to deleted nodes.
                self.wires.retain(|w| {
                    !selected_ids.contains(&w.from_node) && !selected_ids.contains(&w.to_node)
                });
                EventReply::Handled
            }
            _ => EventReply::Unhandled,
        }
    }
}
