//! Dropdown combo box for the Genovo Slate UI.
//!
//! Provides:
//! - `SlateComboBox`: button that opens a dropdown list
//! - Selected item display
//! - Search/filter in dropdown (type to filter)
//! - Virtualized scrolling for long lists
//! - Click outside to close
//! - Keyboard: arrow keys, enter, escape, type to search
//! - Custom item rendering (not just text)

use glam::Vec2;
use genovo_core::Rect;

use crate::core::{KeyCode, KeyModifiers, MouseButton, Padding, UIEvent, UIId};
use crate::render_commands::{
    Border as BorderSpec, Color, CornerRadii, DrawCommand, DrawList, TextAlign,
    TextVerticalAlign, TextureId,
};
use crate::slate_widgets::EventReply;

// =========================================================================
// Constants
// =========================================================================

const COMBO_HEIGHT: f32 = 28.0;
const ITEM_HEIGHT: f32 = 26.0;
const DROPDOWN_MAX_VISIBLE: usize = 8;
const DROPDOWN_PADDING: f32 = 4.0;
const ARROW_SIZE: f32 = 8.0;
const SEARCH_ICON_WIDTH: f32 = 24.0;
const SCROLLBAR_WIDTH: f32 = 8.0;
const MIN_DROPDOWN_WIDTH: f32 = 120.0;

// =========================================================================
// ComboBoxItem
// =========================================================================

/// An item in the combo box dropdown.
#[derive(Debug, Clone)]
pub struct ComboBoxItem {
    /// Display text.
    pub label: String,
    /// Optional icon.
    pub icon: Option<TextureId>,
    /// User data.
    pub data: u64,
    /// Whether this item is enabled.
    pub enabled: bool,
    /// Whether this is a separator.
    pub is_separator: bool,
    /// Optional group header text.
    pub group: Option<String>,
    /// Secondary description text.
    pub description: Option<String>,
    /// Custom color (overrides default).
    pub custom_color: Option<Color>,
}

impl ComboBoxItem {
    /// Create a simple text item.
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            icon: None,
            data: 0,
            enabled: true,
            is_separator: false,
            group: None,
            description: None,
            custom_color: None,
        }
    }

    /// Create with icon.
    pub fn with_icon(mut self, icon: TextureId) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Create with data.
    pub fn with_data(mut self, data: u64) -> Self {
        self.data = data;
        self
    }

    /// Create disabled.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Create a separator.
    pub fn separator() -> Self {
        Self {
            label: String::new(),
            icon: None,
            data: 0,
            enabled: false,
            is_separator: true,
            group: None,
            description: None,
            custom_color: None,
        }
    }

    /// Set a group header.
    pub fn with_group(mut self, group: &str) -> Self {
        self.group = Some(group.to_string());
        self
    }

    /// Set a description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set a custom color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.custom_color = Some(color);
        self
    }

    fn item_height(&self) -> f32 {
        if self.is_separator {
            8.0
        } else if self.description.is_some() {
            ITEM_HEIGHT + 14.0
        } else {
            ITEM_HEIGHT
        }
    }
}

// =========================================================================
// ComboBoxStyle
// =========================================================================

/// Visual styling for the combo box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComboBoxStyle {
    /// Button background colour.
    pub button_bg: Color,
    /// Button background (hovered).
    pub button_hover_bg: Color,
    /// Button background (pressed/open).
    pub button_pressed_bg: Color,
    /// Button border.
    pub button_border: Color,
    /// Button border (focused).
    pub button_focus_border: Color,
    /// Text colour.
    pub text_color: Color,
    /// Disabled text colour.
    pub text_disabled: Color,
    /// Arrow colour.
    pub arrow_color: Color,
    /// Dropdown background.
    pub dropdown_bg: Color,
    /// Dropdown border.
    pub dropdown_border: Color,
    /// Item hover colour.
    pub item_hover: Color,
    /// Item selected colour.
    pub item_selected: Color,
    /// Search field background.
    pub search_bg: Color,
    /// Search text colour.
    pub search_text: Color,
    /// Placeholder text colour.
    pub placeholder_color: Color,
    /// Group header colour.
    pub group_color: Color,
    /// Description colour.
    pub description_color: Color,
    /// Separator colour.
    pub separator_color: Color,
    /// Scrollbar thumb colour.
    pub scrollbar_thumb: Color,
    /// Corner radius.
    pub corner_radius: f32,
    /// Font size.
    pub font_size: f32,
    /// Font ID.
    pub font_id: u32,
}

impl Default for ComboBoxStyle {
    fn default() -> Self {
        Self {
            button_bg: Color::from_hex("#3C3C3C"),
            button_hover_bg: Color::from_hex("#454545"),
            button_pressed_bg: Color::from_hex("#2D2D30"),
            button_border: Color::from_hex("#555555"),
            button_focus_border: Color::from_hex("#007ACC"),
            text_color: Color::from_hex("#CCCCCC"),
            text_disabled: Color::from_hex("#656565"),
            arrow_color: Color::from_hex("#AAAAAA"),
            dropdown_bg: Color::from_hex("#252526"),
            dropdown_border: Color::from_hex("#454545"),
            item_hover: Color::from_hex("#094771"),
            item_selected: Color::from_hex("#094771"),
            search_bg: Color::from_hex("#1E1E1E"),
            search_text: Color::from_hex("#CCCCCC"),
            placeholder_color: Color::from_hex("#666666"),
            group_color: Color::from_hex("#888888"),
            description_color: Color::from_hex("#777777"),
            separator_color: Color::from_hex("#3F3F46"),
            scrollbar_thumb: Color::from_hex("#555555"),
            corner_radius: 3.0,
            font_size: 13.0,
            font_id: 0,
        }
    }
}

// =========================================================================
// SlateComboBox
// =========================================================================

/// A dropdown combo box widget.
#[derive(Debug, Clone)]
pub struct SlateComboBox {
    /// Widget ID.
    pub id: UIId,
    /// All items.
    pub items: Vec<ComboBoxItem>,
    /// Selected item index.
    pub selected_index: Option<usize>,
    /// Whether the dropdown is open.
    pub is_open: bool,
    /// Visual style.
    pub style: ComboBoxStyle,
    /// Placeholder text when nothing selected.
    pub placeholder: String,
    /// Whether the search/filter is enabled.
    pub searchable: bool,
    /// Current search/filter text.
    pub search_text: String,
    /// Filtered item indices.
    filtered_indices: Vec<usize>,
    /// Hovered item index (within filtered list).
    pub hovered_index: Option<usize>,
    /// Scroll offset for the dropdown.
    pub scroll_offset: f32,
    /// Whether the button is hovered.
    pub button_hovered: bool,
    /// Whether the widget is focused.
    pub focused: bool,
    /// Whether the widget is enabled.
    pub enabled: bool,
    /// Whether the widget is visible.
    pub visible: bool,
    /// The width of the combo box (0 = auto).
    pub width: f32,
    /// Max dropdown height in items (0 = default).
    pub max_visible_items: usize,
    /// Whether selection changed this frame.
    pub selection_changed: bool,
    /// Screen size (for dropdown positioning).
    pub screen_size: Vec2,
    /// Keyboard cursor index in filtered list.
    pub keyboard_index: Option<usize>,
}

impl SlateComboBox {
    /// Create a new combo box.
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            items: Vec::new(),
            selected_index: None,
            is_open: false,
            style: ComboBoxStyle::default(),
            placeholder: "Select...".to_string(),
            searchable: true,
            search_text: String::new(),
            filtered_indices: Vec::new(),
            hovered_index: None,
            scroll_offset: 0.0,
            button_hovered: false,
            focused: false,
            enabled: true,
            visible: true,
            width: 200.0,
            max_visible_items: DROPDOWN_MAX_VISIBLE,
            selection_changed: false,
            screen_size: Vec2::new(1920.0, 1080.0),
            keyboard_index: None,
        }
    }

    /// Builder: set items.
    pub fn with_items(mut self, items: Vec<ComboBoxItem>) -> Self {
        self.items = items;
        self.rebuild_filter();
        self
    }

    /// Builder: set selected index.
    pub fn with_selected(mut self, index: usize) -> Self {
        self.selected_index = Some(index);
        self
    }

    /// Builder: set placeholder.
    pub fn with_placeholder(mut self, text: &str) -> Self {
        self.placeholder = text.to_string();
        self
    }

    /// Builder: disable search.
    pub fn without_search(mut self) -> Self {
        self.searchable = false;
        self
    }

    /// Builder: set width.
    pub fn with_width(mut self, w: f32) -> Self {
        self.width = w;
        self
    }

    /// Set items (mutable).
    pub fn set_items(&mut self, items: Vec<ComboBoxItem>) {
        self.items = items;
        self.rebuild_filter();
    }

    /// Get the selected item.
    pub fn selected_item(&self) -> Option<&ComboBoxItem> {
        self.selected_index.and_then(|i| self.items.get(i))
    }

    /// Get the selected label.
    pub fn selected_label(&self) -> &str {
        self.selected_item()
            .map(|i| i.label.as_str())
            .unwrap_or(&self.placeholder)
    }

    /// Open the dropdown.
    pub fn open(&mut self) {
        self.is_open = true;
        self.search_text.clear();
        self.rebuild_filter();
        self.scroll_offset = 0.0;
        self.hovered_index = None;
        self.keyboard_index = self.selected_index.and_then(|si| {
            self.filtered_indices.iter().position(|&fi| fi == si)
        });
    }

    /// Close the dropdown.
    pub fn close(&mut self) {
        self.is_open = false;
        self.search_text.clear();
    }

    /// Toggle the dropdown.
    pub fn toggle(&mut self) {
        if self.is_open {
            self.close();
        } else {
            self.open();
        }
    }

    /// Select an item by index.
    pub fn select(&mut self, index: usize) {
        if index < self.items.len() && self.items[index].enabled && !self.items[index].is_separator
        {
            self.selected_index = Some(index);
            self.selection_changed = true;
            self.close();
        }
    }

    /// Rebuild the filtered index list based on search text.
    fn rebuild_filter(&mut self) {
        if self.search_text.is_empty() {
            self.filtered_indices = (0..self.items.len()).collect();
        } else {
            let query = self.search_text.to_lowercase();
            self.filtered_indices = self
                .items
                .iter()
                .enumerate()
                .filter(|(_, item)| {
                    item.label.to_lowercase().contains(&query)
                        || item
                            .description
                            .as_ref()
                            .map_or(false, |d| d.to_lowercase().contains(&query))
                })
                .map(|(i, _)| i)
                .collect();
        }
    }

    /// Compute the button rect.
    fn button_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            rect.min,
            Vec2::new(rect.min.x + self.width, rect.min.y + COMBO_HEIGHT),
        )
    }

    /// Compute the dropdown rect.
    fn dropdown_rect(&self, button: Rect) -> Rect {
        let max_items = self.max_visible_items.min(self.filtered_indices.len()).max(1);
        let item_heights: f32 = self.filtered_indices[..max_items.min(self.filtered_indices.len())]
            .iter()
            .map(|&i| self.items[i].item_height())
            .sum();
        let search_height = if self.searchable { COMBO_HEIGHT } else { 0.0 };
        let dropdown_h = item_heights + search_height + DROPDOWN_PADDING * 2.0;

        // Prefer below the button, but flip to above if not enough space.
        let below_y = button.max.y + 2.0;
        let above_y = button.min.y - dropdown_h - 2.0;

        let y = if below_y + dropdown_h > self.screen_size.y && above_y >= 0.0 {
            above_y
        } else {
            below_y
        };

        Rect::new(
            Vec2::new(button.min.x, y),
            Vec2::new(
                (button.min.x + self.width.max(MIN_DROPDOWN_WIDTH)).min(self.screen_size.x),
                y + dropdown_h,
            ),
        )
    }

    /// Content area of the dropdown (below search, if present).
    fn content_rect(&self, dropdown: Rect) -> Rect {
        let search_h = if self.searchable { COMBO_HEIGHT } else { 0.0 };
        Rect::new(
            Vec2::new(dropdown.min.x, dropdown.min.y + search_h + DROPDOWN_PADDING),
            Vec2::new(dropdown.max.x, dropdown.max.y - DROPDOWN_PADDING),
        )
    }

    /// Total content height.
    fn total_content_height(&self) -> f32 {
        self.filtered_indices
            .iter()
            .map(|&i| self.items[i].item_height())
            .sum()
    }

    /// Max scroll offset.
    fn max_scroll(&self, content_h: f32) -> f32 {
        (self.total_content_height() - content_h).max(0.0)
    }

    /// Needs scrollbar?
    fn needs_scroll(&self, content_h: f32) -> bool {
        self.total_content_height() > content_h
    }

    /// Find which filtered item is at position Y in the dropdown.
    fn item_at_y(&self, y: f32, content_rect: Rect) -> Option<usize> {
        let mut item_y = content_rect.min.y - self.scroll_offset;
        for (fi, &idx) in self.filtered_indices.iter().enumerate() {
            let h = self.items[idx].item_height();
            if y >= item_y && y < item_y + h {
                if self.items[idx].enabled && !self.items[idx].is_separator {
                    return Some(fi);
                }
                return None;
            }
            item_y += h;
        }
        None
    }

    /// Compute desired size.
    pub fn compute_desired_size(&self) -> Vec2 {
        Vec2::new(self.width, COMBO_HEIGHT)
    }

    /// Paint the combo box.
    pub fn paint(&self, rect: Rect, draw: &mut DrawList) {
        if !self.visible {
            return;
        }

        let button = self.button_rect(rect);
        self.paint_button(button, draw);

        if self.is_open {
            let dropdown = self.dropdown_rect(button);
            self.paint_dropdown(dropdown, draw);
        }
    }

    fn paint_button(&self, rect: Rect, draw: &mut DrawList) {
        let bg = if self.is_open {
            self.style.button_pressed_bg
        } else if self.button_hovered {
            self.style.button_hover_bg
        } else {
            self.style.button_bg
        };

        let border_color = if self.focused || self.is_open {
            self.style.button_focus_border
        } else {
            self.style.button_border
        };

        // Background.
        draw.commands.push(DrawCommand::Rect {
            rect,
            color: bg,
            corner_radii: CornerRadii::all(self.style.corner_radius),
            border: BorderSpec::new(border_color, 1.0),
            shadow: None,
        });

        // Selected text.
        let text_color = if self.enabled {
            if self.selected_index.is_some() {
                self.style.text_color
            } else {
                self.style.placeholder_color
            }
        } else {
            self.style.text_disabled
        };

        let display_text = self.selected_label().to_string();
        draw.commands.push(DrawCommand::Text {
            text: display_text,
            position: Vec2::new(rect.min.x + 8.0, rect.min.y + 6.0),
            font_size: self.style.font_size,
            color: text_color,
            font_id: self.style.font_id,
            max_width: Some(rect.width() - ARROW_SIZE - 24.0),
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });

        // Arrow.
        let arrow_x = rect.max.x - 16.0;
        let arrow_y = rect.min.y + COMBO_HEIGHT * 0.5;
        let half = ARROW_SIZE * 0.4;
        if self.is_open {
            // Up arrow.
            draw.commands.push(DrawCommand::Triangle {
                p0: Vec2::new(arrow_x - half, arrow_y + half * 0.5),
                p1: Vec2::new(arrow_x, arrow_y - half * 0.5),
                p2: Vec2::new(arrow_x + half, arrow_y + half * 0.5),
                color: self.style.arrow_color,
            });
        } else {
            // Down arrow.
            draw.commands.push(DrawCommand::Triangle {
                p0: Vec2::new(arrow_x - half, arrow_y - half * 0.5),
                p1: Vec2::new(arrow_x + half, arrow_y - half * 0.5),
                p2: Vec2::new(arrow_x, arrow_y + half * 0.5),
                color: self.style.arrow_color,
            });
        }
    }

    fn paint_dropdown(&self, dropdown: Rect, draw: &mut DrawList) {
        // Shadow.
        draw.commands.push(DrawCommand::Rect {
            rect: Rect::new(
                dropdown.min + Vec2::new(2.0, 2.0),
                dropdown.max + Vec2::new(6.0, 6.0),
            ),
            color: Color::new(0.0, 0.0, 0.0, 0.4),
            corner_radii: CornerRadii::all(self.style.corner_radius),
            border: BorderSpec::default(),
            shadow: None,
        });

        // Background.
        draw.commands.push(DrawCommand::Rect {
            rect: dropdown,
            color: self.style.dropdown_bg,
            corner_radii: CornerRadii::all(self.style.corner_radius),
            border: BorderSpec::new(self.style.dropdown_border, 1.0),
            shadow: None,
        });

        // Search field.
        if self.searchable {
            let search_rect = Rect::new(
                Vec2::new(dropdown.min.x + DROPDOWN_PADDING, dropdown.min.y + DROPDOWN_PADDING),
                Vec2::new(
                    dropdown.max.x - DROPDOWN_PADDING,
                    dropdown.min.y + COMBO_HEIGHT,
                ),
            );
            draw.commands.push(DrawCommand::Rect {
                rect: search_rect,
                color: self.style.search_bg,
                corner_radii: CornerRadii::all(2.0),
                border: BorderSpec::new(self.style.dropdown_border, 1.0),
                shadow: None,
            });

            let text = if self.search_text.is_empty() {
                "Type to filter..."
            } else {
                &self.search_text
            };
            let color = if self.search_text.is_empty() {
                self.style.placeholder_color
            } else {
                self.style.search_text
            };
            draw.commands.push(DrawCommand::Text {
                text: text.to_string(),
                position: Vec2::new(search_rect.min.x + 8.0, search_rect.min.y + 5.0),
                font_size: self.style.font_size,
                color,
                font_id: self.style.font_id,
                max_width: Some(search_rect.width() - 16.0),
                align: TextAlign::Left,
                vertical_align: TextVerticalAlign::Top,
            });
        }

        // Content area with clipping.
        let content = self.content_rect(dropdown);
        draw.commands.push(DrawCommand::PushClip { rect: content });

        let mut y = content.min.y - self.scroll_offset;
        for (fi, &idx) in self.filtered_indices.iter().enumerate() {
            let item = &self.items[idx];
            let h = item.item_height();

            // Cull items outside visible area.
            if y + h < content.min.y {
                y += h;
                continue;
            }
            if y > content.max.y {
                break;
            }

            if item.is_separator {
                let sep_y = y + h * 0.5;
                draw.commands.push(DrawCommand::Line {
                    start: Vec2::new(content.min.x + 8.0, sep_y),
                    end: Vec2::new(content.max.x - 8.0, sep_y),
                    color: self.style.separator_color,
                    thickness: 1.0,
                });
            } else {
                let is_hovered = self.hovered_index == Some(fi)
                    || self.keyboard_index == Some(fi);
                let is_selected = self.selected_index == Some(idx);

                // Highlight.
                if is_hovered || is_selected {
                    let hl_rect = Rect::new(
                        Vec2::new(content.min.x + 2.0, y),
                        Vec2::new(content.max.x - 2.0, y + ITEM_HEIGHT),
                    );
                    draw.commands.push(DrawCommand::Rect {
                        rect: hl_rect,
                        color: if is_hovered {
                            self.style.item_hover
                        } else {
                            self.style.item_selected.with_alpha(0.5)
                        },
                        corner_radii: CornerRadii::all(2.0),
                        border: BorderSpec::default(),
                        shadow: None,
                    });
                }

                // Icon.
                let mut text_x = content.min.x + 8.0;
                if let Some(tex) = item.icon {
                    let icon_size = 16.0;
                    draw.commands.push(DrawCommand::Image {
                        rect: Rect::new(
                            Vec2::new(text_x, y + (ITEM_HEIGHT - icon_size) * 0.5),
                            Vec2::new(text_x + icon_size, y + (ITEM_HEIGHT + icon_size) * 0.5),
                        ),
                        texture: tex,
                        tint: Color::WHITE,
                        corner_radii: CornerRadii::ZERO,
                        scale_mode: crate::render_commands::ImageScaleMode::Fit,
                        uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
                    });
                    text_x += icon_size + 6.0;
                }

                // Label.
                let label_color = item.custom_color.unwrap_or_else(|| {
                    if item.enabled {
                        self.style.text_color
                    } else {
                        self.style.text_disabled
                    }
                });
                draw.commands.push(DrawCommand::Text {
                    text: item.label.clone(),
                    position: Vec2::new(text_x, y + 5.0),
                    font_size: self.style.font_size,
                    color: label_color,
                    font_id: self.style.font_id,
                    max_width: Some(content.width() - text_x + content.min.x - 8.0),
                    align: TextAlign::Left,
                    vertical_align: TextVerticalAlign::Top,
                });

                // Description.
                if let Some(ref desc) = item.description {
                    draw.commands.push(DrawCommand::Text {
                        text: desc.clone(),
                        position: Vec2::new(text_x, y + ITEM_HEIGHT - 2.0),
                        font_size: self.style.font_size - 2.0,
                        color: self.style.description_color,
                        font_id: self.style.font_id,
                        max_width: Some(content.width() - text_x + content.min.x - 8.0),
                        align: TextAlign::Left,
                        vertical_align: TextVerticalAlign::Top,
                    });
                }
            }

            y += h;
        }

        draw.commands.push(DrawCommand::PopClip);

        // Scrollbar.
        if self.needs_scroll(content.height()) {
            let total_h = self.total_content_height();
            let visible_ratio = content.height() / total_h;
            let thumb_h = (content.height() * visible_ratio).max(20.0);
            let max_scroll = self.max_scroll(content.height());
            let scroll_ratio = if max_scroll > 0.0 {
                self.scroll_offset / max_scroll
            } else {
                0.0
            };
            let thumb_y = content.min.y + scroll_ratio * (content.height() - thumb_h);

            draw.commands.push(DrawCommand::Rect {
                rect: Rect::new(
                    Vec2::new(content.max.x - SCROLLBAR_WIDTH, thumb_y),
                    Vec2::new(content.max.x, thumb_y + thumb_h),
                ),
                color: self.style.scrollbar_thumb,
                corner_radii: CornerRadii::all(SCROLLBAR_WIDTH * 0.5),
                border: BorderSpec::default(),
                shadow: None,
            });
        }
    }

    /// Handle events.
    pub fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled || !self.visible {
            return EventReply::Unhandled;
        }

        self.selection_changed = false;
        let button = self.button_rect(rect);

        if self.is_open {
            let dropdown = self.dropdown_rect(button);
            let content = self.content_rect(dropdown);
            return self.handle_dropdown_event(event, button, dropdown, content);
        }

        match event {
            UIEvent::MouseMove { x, y, .. } => {
                let pos = Vec2::new(*x, *y);
                self.button_hovered = button.contains(pos);
                if self.button_hovered {
                    return EventReply::Handled;
                }
            }

            UIEvent::MouseDown { x, y, button: btn, .. } => {
                if *btn != MouseButton::Left {
                    return EventReply::Unhandled;
                }
                let pos = Vec2::new(*x, *y);
                if button.contains(pos) {
                    self.toggle();
                    return EventReply::Handled;
                }
            }

            UIEvent::KeyDown { key, .. } => {
                if self.focused {
                    match key {
                        KeyCode::Space | KeyCode::Enter => {
                            self.open();
                            return EventReply::Handled;
                        }
                        KeyCode::ArrowDown => {
                            // Select next.
                            let next = self.selected_index.map_or(0, |i| i + 1);
                            if next < self.items.len() {
                                self.select(next);
                            }
                            return EventReply::Handled;
                        }
                        KeyCode::ArrowUp => {
                            // Select previous.
                            if let Some(i) = self.selected_index {
                                if i > 0 {
                                    self.select(i - 1);
                                }
                            }
                            return EventReply::Handled;
                        }
                        _ => {}
                    }
                }
            }

            _ => {}
        }

        EventReply::Unhandled
    }

    fn handle_dropdown_event(
        &mut self,
        event: &UIEvent,
        button_rect: Rect,
        dropdown_rect: Rect,
        content_rect: Rect,
    ) -> EventReply {
        match event {
            UIEvent::MouseMove { x, y, .. } => {
                let pos = Vec2::new(*x, *y);
                self.button_hovered = button_rect.contains(pos);

                if content_rect.contains(pos) {
                    self.hovered_index = self.item_at_y(*y, content_rect);
                } else {
                    self.hovered_index = None;
                }
                EventReply::Handled
            }

            UIEvent::MouseDown { x, y, button, .. } => {
                if *button != MouseButton::Left {
                    return EventReply::Handled;
                }
                let pos = Vec2::new(*x, *y);

                if button_rect.contains(pos) {
                    self.close();
                    return EventReply::Handled;
                }

                if content_rect.contains(pos) {
                    if let Some(fi) = self.item_at_y(*y, content_rect) {
                        if fi < self.filtered_indices.len() {
                            let actual_index = self.filtered_indices[fi];
                            self.select(actual_index);
                        }
                    }
                    return EventReply::Handled;
                }

                if !dropdown_rect.contains(pos) {
                    self.close();
                    return EventReply::Handled;
                }

                EventReply::Handled
            }

            UIEvent::MouseWheel { delta, x, y, .. } => {
                let pos = Vec2::new(*x, *y);
                if dropdown_rect.contains(pos) {
                    self.scroll_offset = (self.scroll_offset - delta * 30.0)
                        .clamp(0.0, self.max_scroll(content_rect.height()));
                    return EventReply::Handled;
                }
                EventReply::Handled
            }

            UIEvent::KeyDown { key, .. } => {
                match key {
                    KeyCode::Escape => {
                        self.close();
                    }
                    KeyCode::Enter => {
                        if let Some(ki) = self.keyboard_index {
                            if ki < self.filtered_indices.len() {
                                let actual_index = self.filtered_indices[ki];
                                self.select(actual_index);
                            }
                        }
                    }
                    KeyCode::ArrowDown => {
                        let count = self.filtered_indices.len();
                        if count > 0 {
                            let next = self.keyboard_index.map_or(0, |i| (i + 1).min(count - 1));
                            // Skip separators and disabled items.
                            let mut idx = next;
                            while idx < count {
                                let actual = self.filtered_indices[idx];
                                if self.items[actual].enabled && !self.items[actual].is_separator {
                                    break;
                                }
                                idx += 1;
                            }
                            if idx < count {
                                self.keyboard_index = Some(idx);
                                self.ensure_visible(idx, content_rect.height());
                            }
                        }
                    }
                    KeyCode::ArrowUp => {
                        if let Some(ki) = self.keyboard_index {
                            if ki > 0 {
                                let mut idx = ki - 1;
                                loop {
                                    let actual = self.filtered_indices[idx];
                                    if self.items[actual].enabled
                                        && !self.items[actual].is_separator
                                    {
                                        break;
                                    }
                                    if idx == 0 {
                                        break;
                                    }
                                    idx -= 1;
                                }
                                self.keyboard_index = Some(idx);
                                self.ensure_visible(idx, content_rect.height());
                            }
                        }
                    }
                    KeyCode::Home => {
                        self.keyboard_index = Some(0);
                        self.scroll_offset = 0.0;
                    }
                    KeyCode::End => {
                        let count = self.filtered_indices.len();
                        if count > 0 {
                            self.keyboard_index = Some(count - 1);
                            self.scroll_offset = self.max_scroll(content_rect.height());
                        }
                    }
                    KeyCode::Backspace => {
                        if self.searchable && !self.search_text.is_empty() {
                            self.search_text.pop();
                            self.rebuild_filter();
                            self.keyboard_index = None;
                        }
                    }
                    _ => {}
                }
                EventReply::Handled
            }

            UIEvent::TextInput { text, .. } => {
                if self.searchable {
                    self.search_text.push_str(text);
                    self.rebuild_filter();
                    self.keyboard_index = if self.filtered_indices.is_empty() {
                        None
                    } else {
                        Some(0)
                    };
                    self.scroll_offset = 0.0;
                }
                EventReply::Handled
            }

            _ => EventReply::Handled,
        }
    }

    /// Ensure a filtered item index is visible in the scroll view.
    fn ensure_visible(&mut self, filtered_index: usize, content_h: f32) {
        let mut y: f32 = 0.0;
        for (fi, &idx) in self.filtered_indices.iter().enumerate() {
            let h = self.items[idx].item_height();
            if fi == filtered_index {
                if y < self.scroll_offset {
                    self.scroll_offset = y;
                } else if y + h > self.scroll_offset + content_h {
                    self.scroll_offset = y + h - content_h;
                }
                return;
            }
            y += h;
        }
    }

    /// Whether the combo is open and blocking.
    pub fn is_blocking(&self) -> bool {
        self.is_open
    }

    /// Take the selection-changed flag.
    pub fn take_selection_changed(&mut self) -> bool {
        let c = self.selection_changed;
        self.selection_changed = false;
        c
    }
}

impl Default for SlateComboBox {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combo_box_create() {
        let cb = SlateComboBox::new()
            .with_items(vec![
                ComboBoxItem::new("Item 1"),
                ComboBoxItem::new("Item 2"),
                ComboBoxItem::new("Item 3"),
            ])
            .with_selected(0);
        assert_eq!(cb.selected_label(), "Item 1");
    }

    #[test]
    fn test_combo_box_select() {
        let mut cb = SlateComboBox::new().with_items(vec![
            ComboBoxItem::new("A"),
            ComboBoxItem::new("B"),
        ]);
        cb.select(1);
        assert_eq!(cb.selected_index, Some(1));
        assert!(cb.selection_changed);
    }

    #[test]
    fn test_combo_box_filter() {
        let mut cb = SlateComboBox::new().with_items(vec![
            ComboBoxItem::new("Apple"),
            ComboBoxItem::new("Banana"),
            ComboBoxItem::new("Cherry"),
        ]);
        cb.open();
        cb.search_text = "an".to_string();
        cb.rebuild_filter();
        assert_eq!(cb.filtered_indices.len(), 1);
        assert_eq!(cb.filtered_indices[0], 1); // Banana
    }

    #[test]
    fn test_combo_box_toggle() {
        let mut cb = SlateComboBox::new();
        assert!(!cb.is_open);
        cb.toggle();
        assert!(cb.is_open);
        cb.toggle();
        assert!(!cb.is_open);
    }

    #[test]
    fn test_combo_box_placeholder() {
        let cb = SlateComboBox::new().with_placeholder("Pick one");
        assert_eq!(cb.selected_label(), "Pick one");
    }
}
