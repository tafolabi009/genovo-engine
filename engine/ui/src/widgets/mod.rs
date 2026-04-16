//! Widget library for the Genovo UI system.
//!
//! Every widget is a self-contained struct with:
//! - All visual and interaction state as fields.
//! - A `new()` constructor with sensible defaults.
//! - An `update()` method for input handling.
//! - A `layout()` method that returns the desired size.
//! - A `render()` method that emits draw commands.

use glam::Vec2;

use genovo_core::Rect;

use crate::core::{Padding, UIEvent, UIId, MouseButton};
use crate::render_commands::{
    Border, Color, CornerRadii, DrawList, ImageScaleMode, Shadow, TextAlign, TextureId,
};
use crate::style::PseudoState;
use crate::text::{Font, TextLayout, TextMeasurement, WrapMode};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_font() -> Font {
    Font::default_proportional()
}

fn measure_text(text: &str, font: &Font, font_size: f32, max_width: Option<f32>) -> TextMeasurement {
    let layout = TextLayout::new();
    layout.measure(text, font, font_size, max_width, WrapMode::Word)
}

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------

/// A clickable button with text and/or icon, supporting hover, press, and
/// disabled visual states.
#[derive(Debug, Clone)]
pub struct Button {
    pub id: UIId,
    pub text: String,
    pub icon: Option<TextureId>,
    pub font_size: f32,
    pub text_color: Color,
    pub background_color: Color,
    pub hover_color: Color,
    pub pressed_color: Color,
    pub disabled_color: Color,
    pub disabled_text_color: Color,
    pub border: Border,
    pub corner_radius: CornerRadii,
    pub padding: Padding,
    pub shadow: Option<Shadow>,
    pub enabled: bool,
    pub state: PseudoState,
    pub min_width: f32,
    pub min_height: f32,
    /// Ripple animation state (normalized 0..1, with center position).
    pub ripple_progress: f32,
    pub ripple_center: Vec2,
    pub ripple_active: bool,
    /// Callbacks are handled externally; this flag is set when clicked.
    pub clicked: bool,
}

impl Button {
    pub fn new(text: &str) -> Self {
        Self {
            id: UIId::INVALID,
            text: text.to_string(),
            icon: None,
            font_size: 14.0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#6200EE"),
            hover_color: Color::from_hex("#7722FF"),
            pressed_color: Color::from_hex("#4400BB"),
            disabled_color: Color::from_rgba8(128, 128, 128, 128),
            disabled_text_color: Color::from_rgba8(180, 180, 180, 200),
            border: Border::default(),
            corner_radius: CornerRadii::all(8.0),
            padding: Padding::new(12.0, 8.0, 12.0, 8.0),
            shadow: Some(Shadow::new(
                Color::from_rgba8(0, 0, 0, 40),
                Vec2::new(0.0, 2.0),
                4.0,
                0.0,
            )),
            enabled: true,
            state: PseudoState::Normal,
            min_width: 64.0,
            min_height: 36.0,
            ripple_progress: 0.0,
            ripple_center: Vec2::ZERO,
            ripple_active: false,
            clicked: false,
        }
    }

    pub fn with_icon(mut self, icon: TextureId) -> Self {
        self.icon = icon.into();
        self.icon = Some(icon);
        self
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        self.clicked = false;
        if !self.enabled {
            self.state = PseudoState::Disabled;
            return false;
        }
        match event {
            UIEvent::Hover { position } => {
                if rect.contains(*position) {
                    self.state = PseudoState::Hovered;
                } else {
                    self.state = PseudoState::Normal;
                }
                false
            }
            UIEvent::HoverEnd => {
                self.state = PseudoState::Normal;
                false
            }
            UIEvent::Click {
                position, button, ..
            } => {
                if rect.contains(*position) && *button == MouseButton::Left {
                    self.state = PseudoState::Pressed;
                    self.ripple_active = true;
                    self.ripple_progress = 0.0;
                    self.ripple_center = *position - rect.min;
                    self.clicked = true;
                    true
                } else {
                    false
                }
            }
            UIEvent::MouseUp { position, .. } => {
                if rect.contains(*position) {
                    self.state = PseudoState::Hovered;
                } else {
                    self.state = PseudoState::Normal;
                }
                false
            }
            _ => false,
        }
    }

    pub fn layout(&self, font: &Font) -> Vec2 {
        let text_size = measure_text(&self.text, font, self.font_size, None);
        let w = (text_size.width + self.padding.horizontal()).max(self.min_width);
        let h = (text_size.height + self.padding.vertical()).max(self.min_height);
        Vec2::new(w, h)
    }

    pub fn render(&mut self, draw: &mut DrawList, rect: Rect, dt: f32) {
        // Update ripple.
        if self.ripple_active {
            self.ripple_progress += dt * 3.0;
            if self.ripple_progress >= 1.0 {
                self.ripple_progress = 1.0;
                self.ripple_active = false;
            }
        }

        let bg = match self.state {
            PseudoState::Disabled => self.disabled_color,
            PseudoState::Pressed => self.pressed_color,
            PseudoState::Hovered => self.hover_color,
            _ => self.background_color,
        };

        let text_color = if self.state == PseudoState::Disabled {
            self.disabled_text_color
        } else {
            self.text_color
        };

        // Shadow.
        if let Some(shadow) = &self.shadow {
            draw.draw_rounded_rect_with_shadow(rect, bg, self.corner_radius, self.border, *shadow);
        } else {
            draw.draw_rounded_rect(rect, bg, self.corner_radius, self.border);
        }

        // Ripple effect.
        if self.ripple_progress > 0.0 && self.ripple_progress < 1.0 {
            let max_radius = rect.width().max(rect.height());
            let radius = max_radius * self.ripple_progress;
            let alpha = 0.3 * (1.0 - self.ripple_progress);
            let ripple_color = Color::new(1.0, 1.0, 1.0, alpha);
            draw.push_clip(rect);
            draw.draw_circle(rect.min + self.ripple_center, radius, ripple_color);
            draw.pop_clip();
        }

        // Icon.
        let mut text_x = rect.min.x + self.padding.left;
        if let Some(icon_tex) = self.icon {
            let icon_size = self.font_size + 4.0;
            let icon_y = rect.center().y - icon_size / 2.0;
            let icon_rect = Rect::new(
                Vec2::new(text_x, icon_y),
                Vec2::new(text_x + icon_size, icon_y + icon_size),
            );
            draw.draw_image(icon_rect, icon_tex, Color::WHITE);
            text_x += icon_size + 4.0;
        }

        // Text (centered vertically).
        let text_y = rect.center().y - self.font_size / 2.0;
        draw.draw_text(&self.text, Vec2::new(text_x, text_y), self.font_size, text_color);
    }
}

// ---------------------------------------------------------------------------
// Label
// ---------------------------------------------------------------------------

/// A text display widget with alignment, wrapping, and truncation support.
#[derive(Debug, Clone)]
pub struct Label {
    pub id: UIId,
    pub text: String,
    pub font_size: f32,
    pub color: Color,
    pub align: TextAlign,
    pub wrap: bool,
    pub max_lines: Option<u32>,
    pub ellipsis: bool,
    pub line_height: f32,
    pub font_id: u32,
}

impl Label {
    pub fn new(text: &str) -> Self {
        Self {
            id: UIId::INVALID,
            text: text.to_string(),
            font_size: 14.0,
            color: Color::WHITE,
            align: TextAlign::Left,
            wrap: true,
            max_lines: None,
            ellipsis: true,
            line_height: 1.2,
            font_id: 0,
        }
    }

    pub fn heading(text: &str) -> Self {
        Self {
            font_size: 24.0,
            ..Self::new(text)
        }
    }

    pub fn small(text: &str) -> Self {
        Self {
            font_size: 12.0,
            ..Self::new(text)
        }
    }

    pub fn layout(&self, font: &Font, max_width: Option<f32>) -> Vec2 {
        let _wrap_mode = if self.wrap { WrapMode::Word } else { WrapMode::None };
        let m = measure_text(&self.text, font, self.font_size, max_width);
        Vec2::new(m.width, m.height)
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        let mut display_text = self.text.clone();

        // Truncation with ellipsis.
        if self.ellipsis && !self.wrap {
            let font = default_font();
            let full_width = measure_text(&display_text, &font, self.font_size, None).width;
            if full_width > rect.width() {
                let ellipsis = "...";
                let ellipsis_w =
                    measure_text(ellipsis, &font, self.font_size, None).width;
                let available = rect.width() - ellipsis_w;
                let mut w = 0.0;
                let mut end_idx = 0;
                for (i, c) in display_text.char_indices() {
                    let cw = font.advance_px(c, self.font_size);
                    if w + cw > available {
                        break;
                    }
                    w += cw;
                    end_idx = i + c.len_utf8();
                }
                display_text = format!("{}...", &display_text[..end_idx]);
            }
        }

        let position = match self.align {
            TextAlign::Left => rect.min,
            TextAlign::Center => Vec2::new(rect.center().x, rect.min.y),
            TextAlign::Right => Vec2::new(rect.max.x, rect.min.y),
        };

        draw.draw_text_ex(
            &display_text,
            position,
            self.font_size,
            self.color,
            self.font_id,
            Some(rect.width()),
            self.align,
            crate::render_commands::TextVerticalAlign::Top,
        );
    }
}

// ---------------------------------------------------------------------------
// TextInput
// ---------------------------------------------------------------------------

/// Single-line text entry with cursor, selection, and placeholder.
#[derive(Debug, Clone)]
pub struct TextInput {
    pub id: UIId,
    pub text: String,
    pub placeholder: String,
    pub font_size: f32,
    pub text_color: Color,
    pub placeholder_color: Color,
    pub background_color: Color,
    pub focused_border_color: Color,
    pub border: Border,
    pub corner_radius: CornerRadii,
    pub padding: Padding,
    pub cursor_position: usize,
    pub selection_start: Option<usize>,
    pub cursor_blink_timer: f32,
    pub cursor_visible: bool,
    pub focused: bool,
    pub enabled: bool,
    pub password: bool,
    pub max_length: Option<usize>,
    pub scroll_offset: f32,
    pub state: PseudoState,
}

impl TextInput {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            text: String::new(),
            placeholder: String::new(),
            font_size: 14.0,
            text_color: Color::WHITE,
            placeholder_color: Color::GRAY,
            background_color: Color::from_hex("#2D2D2D"),
            focused_border_color: Color::from_hex("#6200EE"),
            border: Border::new(Color::from_rgba8(100, 100, 100, 200), 1.0),
            corner_radius: CornerRadii::all(4.0),
            padding: Padding::new(8.0, 6.0, 8.0, 6.0),
            cursor_position: 0,
            selection_start: None,
            cursor_blink_timer: 0.0,
            cursor_visible: true,
            focused: false,
            enabled: true,
            password: false,
            max_length: None,
            scroll_offset: 0.0,
            state: PseudoState::Normal,
        }
    }

    pub fn with_placeholder(mut self, placeholder: &str) -> Self {
        self.placeholder = placeholder.to_string();
        self
    }

    pub fn with_password(mut self) -> Self {
        self.password = true;
        self
    }

    pub fn display_text(&self) -> String {
        if self.password {
            "\u{2022}".repeat(self.text.len())
        } else {
            self.text.clone()
        }
    }

    pub fn update(&mut self, event: &UIEvent, _rect: Rect, dt: f32) -> bool {
        if !self.enabled {
            self.state = PseudoState::Disabled;
            return false;
        }

        // Cursor blink.
        if self.focused {
            self.cursor_blink_timer += dt;
            if self.cursor_blink_timer >= 0.53 {
                self.cursor_blink_timer = 0.0;
                self.cursor_visible = !self.cursor_visible;
            }
        }

        match event {
            UIEvent::Focus => {
                self.focused = true;
                self.state = PseudoState::Focused;
                self.cursor_visible = true;
                self.cursor_blink_timer = 0.0;
                true
            }
            UIEvent::Blur => {
                self.focused = false;
                self.state = PseudoState::Normal;
                self.selection_start = None;
                true
            }
            UIEvent::TextInput { character } if self.focused => {
                if character.is_control() {
                    return false;
                }
                if let Some(max) = self.max_length {
                    if self.text.len() >= max {
                        return false;
                    }
                }
                self.delete_selection();
                let idx = self.byte_index(self.cursor_position);
                self.text.insert(idx, *character);
                self.cursor_position += 1;
                self.cursor_visible = true;
                self.cursor_blink_timer = 0.0;
                true
            }
            UIEvent::KeyInput {
                key, pressed: true, modifiers, ..
            } if self.focused => {
                use crate::core::KeyCode;
                match key {
                    KeyCode::Backspace => {
                        if self.selection_start.is_some() {
                            self.delete_selection();
                        } else if self.cursor_position > 0 {
                            self.cursor_position -= 1;
                            let idx = self.byte_index(self.cursor_position);
                            let c = self.text[idx..].chars().next().unwrap();
                            self.text.remove(idx);
                            let _ = c;
                        }
                        true
                    }
                    KeyCode::Delete => {
                        if self.selection_start.is_some() {
                            self.delete_selection();
                        } else if self.cursor_position < self.text.chars().count() {
                            let idx = self.byte_index(self.cursor_position);
                            self.text.remove(idx);
                        }
                        true
                    }
                    KeyCode::ArrowLeft => {
                        if modifiers.shift {
                            if self.selection_start.is_none() {
                                self.selection_start = Some(self.cursor_position);
                            }
                        } else {
                            self.selection_start = None;
                        }
                        if self.cursor_position > 0 {
                            self.cursor_position -= 1;
                        }
                        self.cursor_visible = true;
                        self.cursor_blink_timer = 0.0;
                        true
                    }
                    KeyCode::ArrowRight => {
                        if modifiers.shift {
                            if self.selection_start.is_none() {
                                self.selection_start = Some(self.cursor_position);
                            }
                        } else {
                            self.selection_start = None;
                        }
                        let char_count = self.text.chars().count();
                        if self.cursor_position < char_count {
                            self.cursor_position += 1;
                        }
                        self.cursor_visible = true;
                        self.cursor_blink_timer = 0.0;
                        true
                    }
                    KeyCode::Home => {
                        if modifiers.shift {
                            if self.selection_start.is_none() {
                                self.selection_start = Some(self.cursor_position);
                            }
                        } else {
                            self.selection_start = None;
                        }
                        self.cursor_position = 0;
                        true
                    }
                    KeyCode::End => {
                        if modifiers.shift {
                            if self.selection_start.is_none() {
                                self.selection_start = Some(self.cursor_position);
                            }
                        } else {
                            self.selection_start = None;
                        }
                        self.cursor_position = self.text.chars().count();
                        true
                    }
                    KeyCode::A if modifiers.ctrl => {
                        // Select all.
                        self.selection_start = Some(0);
                        self.cursor_position = self.text.chars().count();
                        true
                    }
                    _ => false,
                }
            }
            UIEvent::Click { position, .. } => {
                // Place cursor at click position.
                let font = default_font();
                let display = self.display_text();
                let relative_x = position.x - _rect.min.x - self.padding.left + self.scroll_offset;

                let mut x = 0.0;
                let mut best_pos = 0;
                for (i, c) in display.chars().enumerate() {
                    let advance = font.advance_px(c, self.font_size);
                    if relative_x < x + advance / 2.0 {
                        best_pos = i;
                        break;
                    }
                    x += advance;
                    best_pos = i + 1;
                }
                self.cursor_position = best_pos;
                self.selection_start = None;
                self.cursor_visible = true;
                self.cursor_blink_timer = 0.0;
                true
            }
            _ => false,
        }
    }

    fn byte_index(&self, char_index: usize) -> usize {
        self.text
            .char_indices()
            .nth(char_index)
            .map(|(i, _)| i)
            .unwrap_or(self.text.len())
    }

    fn delete_selection(&mut self) {
        if let Some(sel_start) = self.selection_start.take() {
            let (start, end) = if sel_start < self.cursor_position {
                (sel_start, self.cursor_position)
            } else {
                (self.cursor_position, sel_start)
            };
            let byte_start = self.byte_index(start);
            let byte_end = self.byte_index(end);
            self.text.drain(byte_start..byte_end);
            self.cursor_position = start;
        }
    }

    pub fn selected_text(&self) -> Option<String> {
        self.selection_start.map(|sel| {
            let (start, end) = if sel < self.cursor_position {
                (sel, self.cursor_position)
            } else {
                (self.cursor_position, sel)
            };
            let bs = self.byte_index(start);
            let be = self.byte_index(end);
            self.text[bs..be].to_string()
        })
    }

    pub fn layout(&self, _font: &Font) -> Vec2 {
        Vec2::new(200.0, self.font_size + self.padding.vertical())
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        let border = if self.focused {
            Border::new(self.focused_border_color, 2.0)
        } else {
            self.border
        };

        draw.draw_rounded_rect(rect, self.background_color, self.corner_radius, border);

        let text_area = Rect::new(
            Vec2::new(rect.min.x + self.padding.left, rect.min.y + self.padding.top),
            Vec2::new(rect.max.x - self.padding.right, rect.max.y - self.padding.bottom),
        );
        draw.push_clip(text_area);

        let display = self.display_text();
        let text_y = text_area.center().y - self.font_size / 2.0;

        if display.is_empty() && !self.focused {
            // Placeholder.
            draw.draw_text(
                &self.placeholder,
                Vec2::new(text_area.min.x - self.scroll_offset, text_y),
                self.font_size,
                self.placeholder_color,
            );
        } else {
            // Selection highlight.
            if let Some(sel_start) = self.selection_start {
                let font = default_font();
                let (s, e) = if sel_start < self.cursor_position {
                    (sel_start, self.cursor_position)
                } else {
                    (self.cursor_position, sel_start)
                };
                let sx = cursor_x_offset(&display, s, &font, self.font_size);
                let ex = cursor_x_offset(&display, e, &font, self.font_size);
                let sel_rect = Rect::new(
                    Vec2::new(text_area.min.x + sx - self.scroll_offset, text_area.min.y),
                    Vec2::new(text_area.min.x + ex - self.scroll_offset, text_area.max.y),
                );
                draw.draw_rect(sel_rect, Color::from_rgba8(100, 100, 255, 100));
            }

            draw.draw_text(
                &display,
                Vec2::new(text_area.min.x - self.scroll_offset, text_y),
                self.font_size,
                self.text_color,
            );

            // Cursor.
            if self.focused && self.cursor_visible {
                let font = default_font();
                let cx = cursor_x_offset(&display, self.cursor_position, &font, self.font_size);
                let cursor_x = text_area.min.x + cx - self.scroll_offset;
                draw.draw_line(
                    Vec2::new(cursor_x, text_area.min.y),
                    Vec2::new(cursor_x, text_area.max.y),
                    self.text_color,
                    1.0,
                );
            }
        }

        draw.pop_clip();
    }
}

fn cursor_x_offset(text: &str, char_pos: usize, font: &Font, font_size: f32) -> f32 {
    let mut x = 0.0;
    for (i, c) in text.chars().enumerate() {
        if i >= char_pos {
            break;
        }
        x += font.advance_px(c, font_size);
    }
    x
}

// ---------------------------------------------------------------------------
// TextArea
// ---------------------------------------------------------------------------

/// Multi-line text editor with scrolling and optional line numbers.
#[derive(Debug, Clone)]
pub struct TextArea {
    pub id: UIId,
    pub text: String,
    pub font_size: f32,
    pub text_color: Color,
    pub background_color: Color,
    pub border: Border,
    pub padding: Padding,
    pub cursor_position: usize,
    pub cursor_line: usize,
    pub cursor_column: usize,
    pub scroll_offset: Vec2,
    pub focused: bool,
    pub enabled: bool,
    pub show_line_numbers: bool,
    pub line_number_width: f32,
    pub line_number_color: Color,
    pub cursor_blink_timer: f32,
    pub cursor_visible: bool,
    pub selection_start: Option<usize>,
    pub tab_size: usize,
}

impl TextArea {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            text: String::new(),
            font_size: 14.0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#1E1E1E"),
            border: Border::new(Color::from_rgba8(60, 60, 60, 200), 1.0),
            padding: Padding::all(8.0),
            cursor_position: 0,
            cursor_line: 0,
            cursor_column: 0,
            scroll_offset: Vec2::ZERO,
            focused: false,
            enabled: true,
            show_line_numbers: true,
            line_number_width: 40.0,
            line_number_color: Color::GRAY,
            cursor_blink_timer: 0.0,
            cursor_visible: true,
            selection_start: None,
            tab_size: 4,
        }
    }

    pub fn line_count(&self) -> usize {
        self.text.lines().count().max(1)
    }

    pub fn current_line(&self) -> &str {
        self.text.lines().nth(self.cursor_line).unwrap_or("")
    }

    fn update_cursor_line_col(&mut self) {
        let mut line = 0;
        let mut col = 0;
        for (i, c) in self.text.chars().enumerate() {
            if i == self.cursor_position {
                break;
            }
            if c == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        self.cursor_line = line;
        self.cursor_column = col;
    }

    pub fn update(&mut self, event: &UIEvent, _rect: Rect, dt: f32) -> bool {
        if !self.enabled {
            return false;
        }

        if self.focused {
            self.cursor_blink_timer += dt;
            if self.cursor_blink_timer >= 0.53 {
                self.cursor_blink_timer = 0.0;
                self.cursor_visible = !self.cursor_visible;
            }
        }

        match event {
            UIEvent::Focus => {
                self.focused = true;
                true
            }
            UIEvent::Blur => {
                self.focused = false;
                true
            }
            UIEvent::TextInput { character } if self.focused => {
                if *character == '\t' {
                    let spaces = " ".repeat(self.tab_size);
                    let idx = self.byte_index(self.cursor_position);
                    self.text.insert_str(idx, &spaces);
                    self.cursor_position += self.tab_size;
                } else if !character.is_control() || *character == '\n' {
                    let idx = self.byte_index(self.cursor_position);
                    self.text.insert(idx, *character);
                    self.cursor_position += 1;
                }
                self.update_cursor_line_col();
                self.cursor_visible = true;
                self.cursor_blink_timer = 0.0;
                true
            }
            UIEvent::KeyInput { key, pressed: true, modifiers, .. } if self.focused => {
                use crate::core::KeyCode;
                match key {
                    KeyCode::Enter => {
                        let idx = self.byte_index(self.cursor_position);
                        self.text.insert(idx, '\n');
                        self.cursor_position += 1;
                        self.update_cursor_line_col();
                        true
                    }
                    KeyCode::Backspace => {
                        if self.cursor_position > 0 {
                            self.cursor_position -= 1;
                            let idx = self.byte_index(self.cursor_position);
                            self.text.remove(idx);
                            self.update_cursor_line_col();
                        }
                        true
                    }
                    KeyCode::Delete => {
                        if self.cursor_position < self.text.chars().count() {
                            let idx = self.byte_index(self.cursor_position);
                            self.text.remove(idx);
                            self.update_cursor_line_col();
                        }
                        true
                    }
                    KeyCode::ArrowLeft => {
                        if self.cursor_position > 0 {
                            self.cursor_position -= 1;
                            self.update_cursor_line_col();
                        }
                        true
                    }
                    KeyCode::ArrowRight => {
                        if self.cursor_position < self.text.chars().count() {
                            self.cursor_position += 1;
                            self.update_cursor_line_col();
                        }
                        true
                    }
                    KeyCode::ArrowUp => {
                        if self.cursor_line > 0 {
                            self.move_cursor_to_line(self.cursor_line - 1);
                        }
                        true
                    }
                    KeyCode::ArrowDown => {
                        if self.cursor_line + 1 < self.line_count() {
                            self.move_cursor_to_line(self.cursor_line + 1);
                        }
                        true
                    }
                    KeyCode::Home => {
                        self.cursor_column = 0;
                        self.recalc_cursor_position();
                        true
                    }
                    KeyCode::End => {
                        let line = self.current_line().to_string();
                        self.cursor_column = line.len();
                        self.recalc_cursor_position();
                        true
                    }
                    _ => {
                        let _ = modifiers;
                        false
                    }
                }
            }
            UIEvent::Scroll { delta, .. } => {
                self.scroll_offset.y = (self.scroll_offset.y - delta.y * 20.0).max(0.0);
                true
            }
            _ => false,
        }
    }

    fn byte_index(&self, char_index: usize) -> usize {
        self.text
            .char_indices()
            .nth(char_index)
            .map(|(i, _)| i)
            .unwrap_or(self.text.len())
    }

    fn move_cursor_to_line(&mut self, target_line: usize) {
        let desired_col = self.cursor_column;
        self.cursor_line = target_line;
        let line_text: &str = self.text.lines().nth(target_line).unwrap_or("");
        let max_col = line_text.chars().count();
        self.cursor_column = desired_col.min(max_col);
        self.recalc_cursor_position();
    }

    fn recalc_cursor_position(&mut self) {
        let mut pos = 0;
        for (i, line) in self.text.lines().enumerate() {
            if i == self.cursor_line {
                pos += self.cursor_column.min(line.chars().count());
                break;
            }
            pos += line.chars().count() + 1; // +1 for newline
        }
        self.cursor_position = pos;
    }

    pub fn layout(&self, _font: &Font) -> Vec2 {
        Vec2::new(400.0, 200.0)
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        draw.draw_rounded_rect(rect, self.background_color, CornerRadii::all(4.0), self.border);

        let content_x = if self.show_line_numbers {
            rect.min.x + self.line_number_width
        } else {
            rect.min.x
        } + self.padding.left;

        let content_rect = Rect::new(
            Vec2::new(content_x, rect.min.y + self.padding.top),
            Vec2::new(rect.max.x - self.padding.right, rect.max.y - self.padding.bottom),
        );

        draw.push_clip(rect);

        let line_h = self.font_size * 1.4;

        // Line numbers.
        if self.show_line_numbers {
            let ln_bg = Rect::new(
                Vec2::new(rect.min.x, rect.min.y),
                Vec2::new(rect.min.x + self.line_number_width, rect.max.y),
            );
            draw.draw_rect(ln_bg, Color::from_rgba8(30, 30, 30, 200));

            for i in 0..self.line_count() {
                let y = rect.min.y + self.padding.top + i as f32 * line_h - self.scroll_offset.y;
                if y + line_h < rect.min.y || y > rect.max.y {
                    continue;
                }
                let num = format!("{}", i + 1);
                draw.draw_text(
                    &num,
                    Vec2::new(rect.min.x + 4.0, y),
                    self.font_size,
                    self.line_number_color,
                );
            }

            // Separator line.
            draw.draw_line(
                Vec2::new(rect.min.x + self.line_number_width, rect.min.y),
                Vec2::new(rect.min.x + self.line_number_width, rect.max.y),
                Color::from_rgba8(60, 60, 60, 200),
                1.0,
            );
        }

        // Text content.
        draw.push_clip(content_rect);
        for (i, line) in self.text.lines().enumerate() {
            let y = content_rect.min.y + i as f32 * line_h - self.scroll_offset.y;
            if y + line_h < content_rect.min.y || y > content_rect.max.y {
                continue;
            }
            draw.draw_text(
                line,
                Vec2::new(content_rect.min.x - self.scroll_offset.x, y),
                self.font_size,
                self.text_color,
            );
        }

        // Cursor.
        if self.focused && self.cursor_visible {
            let font = default_font();
            let line_text = self.text.lines().nth(self.cursor_line).unwrap_or("");
            let cx = cursor_x_offset(line_text, self.cursor_column, &font, self.font_size);
            let cy = content_rect.min.y + self.cursor_line as f32 * line_h - self.scroll_offset.y;
            draw.draw_line(
                Vec2::new(content_rect.min.x + cx, cy),
                Vec2::new(content_rect.min.x + cx, cy + line_h),
                self.text_color,
                1.0,
            );
        }

        draw.pop_clip();
        draw.pop_clip();
    }
}

// ---------------------------------------------------------------------------
// Checkbox
// ---------------------------------------------------------------------------

/// A checkbox with check, uncheck, and indeterminate states.
#[derive(Debug, Clone)]
pub struct Checkbox {
    pub id: UIId,
    pub checked: bool,
    pub indeterminate: bool,
    pub label: String,
    pub font_size: f32,
    pub text_color: Color,
    pub check_color: Color,
    pub box_color: Color,
    pub checked_bg: Color,
    pub box_size: f32,
    pub enabled: bool,
    pub state: PseudoState,
    pub toggled: bool,
}

impl Checkbox {
    pub fn new(label: &str, checked: bool) -> Self {
        Self {
            id: UIId::INVALID,
            checked,
            indeterminate: false,
            label: label.to_string(),
            font_size: 14.0,
            text_color: Color::WHITE,
            check_color: Color::WHITE,
            box_color: Color::from_rgba8(150, 150, 150, 200),
            checked_bg: Color::from_hex("#6200EE"),
            box_size: 18.0,
            enabled: true,
            state: PseudoState::Normal,
            toggled: false,
        }
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        self.toggled = false;
        if !self.enabled {
            return false;
        }
        match event {
            UIEvent::Click { position, button: MouseButton::Left, .. } => {
                if rect.contains(*position) {
                    self.checked = !self.checked;
                    self.indeterminate = false;
                    self.toggled = true;
                    true
                } else {
                    false
                }
            }
            UIEvent::Hover { position } => {
                self.state = if rect.contains(*position) {
                    PseudoState::Hovered
                } else {
                    PseudoState::Normal
                };
                false
            }
            UIEvent::HoverEnd => {
                self.state = PseudoState::Normal;
                false
            }
            _ => false,
        }
    }

    pub fn layout(&self, font: &Font) -> Vec2 {
        let text_w = measure_text(&self.label, font, self.font_size, None).width;
        Vec2::new(self.box_size + 8.0 + text_w, self.box_size.max(self.font_size))
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        let box_y = rect.center().y - self.box_size / 2.0;
        let box_rect = Rect::new(
            Vec2::new(rect.min.x, box_y),
            Vec2::new(rect.min.x + self.box_size, box_y + self.box_size),
        );

        let bg = if self.checked || self.indeterminate {
            self.checked_bg
        } else {
            Color::TRANSPARENT
        };
        let border_color = if self.checked || self.indeterminate {
            self.checked_bg
        } else {
            self.box_color
        };

        draw.draw_rounded_rect(
            box_rect,
            bg,
            CornerRadii::all(3.0),
            Border::new(border_color, 2.0),
        );

        // Check mark or indeterminate dash.
        if self.checked {
            // Simple checkmark using two lines.
            let cx = box_rect.center().x;
            let cy = box_rect.center().y;
            let s = self.box_size * 0.25;
            draw.draw_line(
                Vec2::new(cx - s, cy),
                Vec2::new(cx - s * 0.3, cy + s),
                self.check_color,
                2.0,
            );
            draw.draw_line(
                Vec2::new(cx - s * 0.3, cy + s),
                Vec2::new(cx + s, cy - s * 0.7),
                self.check_color,
                2.0,
            );
        } else if self.indeterminate {
            let cx = box_rect.center().x;
            let cy = box_rect.center().y;
            let s = self.box_size * 0.3;
            draw.draw_line(
                Vec2::new(cx - s, cy),
                Vec2::new(cx + s, cy),
                self.check_color,
                2.0,
            );
        }

        // Label text.
        let text_x = box_rect.max.x + 8.0;
        let text_y = rect.center().y - self.font_size / 2.0;
        draw.draw_text(&self.label, Vec2::new(text_x, text_y), self.font_size, self.text_color);
    }
}

// ---------------------------------------------------------------------------
// RadioButton
// ---------------------------------------------------------------------------

/// A radio button for grouped selection.
#[derive(Debug, Clone)]
pub struct RadioButton {
    pub id: UIId,
    pub selected: bool,
    pub label: String,
    pub group: String,
    pub font_size: f32,
    pub text_color: Color,
    pub selected_color: Color,
    pub ring_color: Color,
    pub circle_size: f32,
    pub enabled: bool,
    pub toggled: bool,
}

impl RadioButton {
    pub fn new(label: &str, group: &str, selected: bool) -> Self {
        Self {
            id: UIId::INVALID,
            selected,
            label: label.to_string(),
            group: group.to_string(),
            font_size: 14.0,
            text_color: Color::WHITE,
            selected_color: Color::from_hex("#6200EE"),
            ring_color: Color::from_rgba8(150, 150, 150, 200),
            circle_size: 18.0,
            enabled: true,
            toggled: false,
        }
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        self.toggled = false;
        if !self.enabled {
            return false;
        }
        if let UIEvent::Click { position, button: MouseButton::Left, .. } = event {
            if rect.contains(*position) {
                self.selected = true;
                self.toggled = true;
                return true;
            }
        }
        false
    }

    pub fn layout(&self, font: &Font) -> Vec2 {
        let text_w = measure_text(&self.label, font, self.font_size, None).width;
        Vec2::new(self.circle_size + 8.0 + text_w, self.circle_size.max(self.font_size))
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        let center = Vec2::new(
            rect.min.x + self.circle_size / 2.0,
            rect.center().y,
        );
        let radius = self.circle_size / 2.0;

        let ring = if self.selected { self.selected_color } else { self.ring_color };
        draw.draw_circle(center, radius, Color::TRANSPARENT);
        draw.push(crate::render_commands::DrawCommand::Circle {
            center,
            radius,
            color: Color::TRANSPARENT,
            border: Border::new(ring, 2.0),
        });

        if self.selected {
            draw.draw_circle(center, radius * 0.5, self.selected_color);
        }

        let text_x = rect.min.x + self.circle_size + 8.0;
        let text_y = rect.center().y - self.font_size / 2.0;
        draw.draw_text(&self.label, Vec2::new(text_x, text_y), self.font_size, self.text_color);
    }
}

// ---------------------------------------------------------------------------
// Slider
// ---------------------------------------------------------------------------

/// A horizontal or vertical slider with range, step, and drag interaction.
#[derive(Debug, Clone)]
pub struct Slider {
    pub id: UIId,
    pub value: f32,
    pub min: f32,
    pub max: f32,
    pub step: Option<f32>,
    pub vertical: bool,
    pub track_color: Color,
    pub fill_color: Color,
    pub thumb_color: Color,
    pub thumb_radius: f32,
    pub track_height: f32,
    pub enabled: bool,
    pub dragging: bool,
    pub state: PseudoState,
    pub changed: bool,
    pub show_value: bool,
}

impl Slider {
    pub fn new(min: f32, max: f32, value: f32) -> Self {
        Self {
            id: UIId::INVALID,
            value: value.clamp(min, max),
            min,
            max,
            step: None,
            vertical: false,
            track_color: Color::from_rgba8(60, 60, 60, 200),
            fill_color: Color::from_hex("#6200EE"),
            thumb_color: Color::WHITE,
            thumb_radius: 8.0,
            track_height: 4.0,
            enabled: true,
            dragging: false,
            state: PseudoState::Normal,
            changed: false,
            show_value: false,
        }
    }

    pub fn with_step(mut self, step: f32) -> Self {
        self.step = Some(step);
        self
    }

    pub fn normalized(&self) -> f32 {
        if (self.max - self.min).abs() < 1e-8 {
            0.0
        } else {
            (self.value - self.min) / (self.max - self.min)
        }
    }

    fn apply_step(&mut self) {
        if let Some(step) = self.step {
            if step > 0.0 {
                self.value = ((self.value - self.min) / step).round() * step + self.min;
            }
        }
        self.value = self.value.clamp(self.min, self.max);
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        self.changed = false;
        if !self.enabled {
            return false;
        }
        match event {
            UIEvent::Click { position, button: MouseButton::Left, .. }
            | UIEvent::DragStart { position, button: MouseButton::Left } => {
                if rect.contains(*position) {
                    self.dragging = true;
                    self.set_value_from_position(*position, rect);
                    return true;
                }
                false
            }
            UIEvent::DragMove { position, .. } if self.dragging => {
                self.set_value_from_position(*position, rect);
                true
            }
            UIEvent::DragEnd { .. } | UIEvent::MouseUp { .. } => {
                self.dragging = false;
                false
            }
            UIEvent::Hover { position } => {
                self.state = if rect.contains(*position) {
                    PseudoState::Hovered
                } else {
                    PseudoState::Normal
                };
                false
            }
            _ => false,
        }
    }

    fn set_value_from_position(&mut self, pos: Vec2, rect: Rect) {
        let t = if self.vertical {
            1.0 - (pos.y - rect.min.y) / rect.height()
        } else {
            (pos.x - rect.min.x) / rect.width()
        };
        let t = t.clamp(0.0, 1.0);
        let old = self.value;
        self.value = self.min + t * (self.max - self.min);
        self.apply_step();
        self.changed = (self.value - old).abs() > 1e-8;
    }

    pub fn layout(&self) -> Vec2 {
        if self.vertical {
            Vec2::new(self.thumb_radius * 2.0 + 4.0, 150.0)
        } else {
            Vec2::new(200.0, self.thumb_radius * 2.0 + 4.0)
        }
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        let t = self.normalized();

        if self.vertical {
            let track_x = rect.center().x;
            let track_top = rect.min.y + self.thumb_radius;
            let track_bottom = rect.max.y - self.thumb_radius;

            // Track background.
            let track_rect = Rect::new(
                Vec2::new(track_x - self.track_height / 2.0, track_top),
                Vec2::new(track_x + self.track_height / 2.0, track_bottom),
            );
            draw.draw_rounded_rect(track_rect, self.track_color, CornerRadii::all(2.0), Border::default());

            // Fill.
            let fill_y = track_bottom - (track_bottom - track_top) * t;
            let fill_rect = Rect::new(
                Vec2::new(track_x - self.track_height / 2.0, fill_y),
                Vec2::new(track_x + self.track_height / 2.0, track_bottom),
            );
            draw.draw_rounded_rect(fill_rect, self.fill_color, CornerRadii::all(2.0), Border::default());

            // Thumb.
            let thumb_y = fill_y;
            draw.draw_circle(Vec2::new(track_x, thumb_y), self.thumb_radius, self.thumb_color);
        } else {
            let track_y = rect.center().y;
            let track_left = rect.min.x + self.thumb_radius;
            let track_right = rect.max.x - self.thumb_radius;

            let track_rect = Rect::new(
                Vec2::new(track_left, track_y - self.track_height / 2.0),
                Vec2::new(track_right, track_y + self.track_height / 2.0),
            );
            draw.draw_rounded_rect(track_rect, self.track_color, CornerRadii::all(2.0), Border::default());

            let fill_x = track_left + (track_right - track_left) * t;
            let fill_rect = Rect::new(
                Vec2::new(track_left, track_y - self.track_height / 2.0),
                Vec2::new(fill_x, track_y + self.track_height / 2.0),
            );
            draw.draw_rounded_rect(fill_rect, self.fill_color, CornerRadii::all(2.0), Border::default());

            let thumb_center = Vec2::new(fill_x, track_y);
            draw.draw_circle(thumb_center, self.thumb_radius, self.thumb_color);

            if self.show_value {
                let val_text = format!("{:.1}", self.value);
                draw.draw_text(
                    &val_text,
                    Vec2::new(thumb_center.x - 10.0, rect.min.y - 16.0),
                    12.0,
                    self.thumb_color,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ProgressBar
// ---------------------------------------------------------------------------

/// Determinate or indeterminate progress indicator.
#[derive(Debug, Clone)]
pub struct ProgressBar {
    pub id: UIId,
    pub progress: f32,
    pub indeterminate: bool,
    pub circular: bool,
    pub track_color: Color,
    pub fill_color: Color,
    pub height: f32,
    pub corner_radius: f32,
    pub animation_time: f32,
    pub text_visible: bool,
}

impl ProgressBar {
    pub fn new(progress: f32) -> Self {
        Self {
            id: UIId::INVALID,
            progress: progress.clamp(0.0, 1.0),
            indeterminate: false,
            circular: false,
            track_color: Color::from_rgba8(60, 60, 60, 200),
            fill_color: Color::from_hex("#6200EE"),
            height: 6.0,
            corner_radius: 3.0,
            animation_time: 0.0,
            text_visible: false,
        }
    }

    pub fn indeterminate() -> Self {
        Self {
            indeterminate: true,
            ..Self::new(0.0)
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.animation_time += dt;
    }

    pub fn layout(&self) -> Vec2 {
        if self.circular {
            Vec2::new(40.0, 40.0)
        } else {
            Vec2::new(200.0, self.height)
        }
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        if self.circular {
            // Circular progress.
            let center = rect.center();
            let radius = rect.width().min(rect.height()) / 2.0 - 2.0;

            draw.draw_circle(center, radius, Color::TRANSPARENT);
            draw.push(crate::render_commands::DrawCommand::Circle {
                center,
                radius,
                color: Color::TRANSPARENT,
                border: Border::new(self.track_color, 3.0),
            });

            // For the fill arc, we approximate with a circle for now.
            if !self.indeterminate {
                // Draw a partial circle (simplified: just draw a filled circle
                // with clipping). A real implementation would use arc drawing.
                let fill_height = rect.height() * self.progress;
                let clip = Rect::new(
                    Vec2::new(rect.min.x, rect.max.y - fill_height),
                    rect.max,
                );
                draw.push_clip(clip);
                draw.push(crate::render_commands::DrawCommand::Circle {
                    center,
                    radius,
                    color: Color::TRANSPARENT,
                    border: Border::new(self.fill_color, 3.0),
                });
                draw.pop_clip();
            }
        } else {
            // Linear progress bar.
            draw.draw_rounded_rect(
                rect,
                self.track_color,
                CornerRadii::all(self.corner_radius),
                Border::default(),
            );

            if self.indeterminate {
                // Animated sliding bar.
                let cycle = (self.animation_time * 1.5) % 2.0;
                let start = if cycle < 1.0 { cycle } else { 2.0 - cycle };
                let width = 0.3;
                let fill_start = rect.min.x + rect.width() * start.clamp(0.0, 1.0 - width);
                let fill_end = fill_start + rect.width() * width;
                let fill_rect = Rect::new(
                    Vec2::new(fill_start, rect.min.y),
                    Vec2::new(fill_end, rect.max.y),
                );
                draw.draw_rounded_rect(
                    fill_rect,
                    self.fill_color,
                    CornerRadii::all(self.corner_radius),
                    Border::default(),
                );
            } else {
                let fill_width = rect.width() * self.progress.clamp(0.0, 1.0);
                if fill_width > 0.0 {
                    let fill_rect = Rect::new(
                        rect.min,
                        Vec2::new(rect.min.x + fill_width, rect.max.y),
                    );
                    draw.draw_rounded_rect(
                        fill_rect,
                        self.fill_color,
                        CornerRadii::all(self.corner_radius),
                        Border::default(),
                    );
                }

                if self.text_visible {
                    let pct = format!("{}%", (self.progress * 100.0) as u32);
                    draw.draw_text(
                        &pct,
                        Vec2::new(rect.center().x - 12.0, rect.min.y - 16.0),
                        12.0,
                        self.fill_color,
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dropdown / ComboBox
// ---------------------------------------------------------------------------

/// Dropdown / combo box with popup list selection and optional search filter.
#[derive(Debug, Clone)]
pub struct Dropdown {
    pub id: UIId,
    pub items: Vec<String>,
    pub selected_index: Option<usize>,
    pub open: bool,
    pub search_text: String,
    pub searchable: bool,
    pub font_size: f32,
    pub text_color: Color,
    pub background_color: Color,
    pub highlight_color: Color,
    pub border: Border,
    pub corner_radius: CornerRadii,
    pub padding: Padding,
    pub item_height: f32,
    pub max_visible_items: usize,
    pub hovered_index: Option<usize>,
    pub changed: bool,
}

impl Dropdown {
    pub fn new(items: Vec<String>) -> Self {
        Self {
            id: UIId::INVALID,
            items,
            selected_index: None,
            open: false,
            search_text: String::new(),
            searchable: false,
            font_size: 14.0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#2D2D2D"),
            highlight_color: Color::from_hex("#6200EE"),
            border: Border::new(Color::from_rgba8(100, 100, 100, 200), 1.0),
            corner_radius: CornerRadii::all(4.0),
            padding: Padding::new(8.0, 6.0, 8.0, 6.0),
            item_height: 28.0,
            max_visible_items: 8,
            hovered_index: None,
            changed: false,
        }
    }

    pub fn selected_text(&self) -> &str {
        self.selected_index
            .and_then(|i| self.items.get(i))
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    pub fn filtered_items(&self) -> Vec<(usize, &str)> {
        if self.search_text.is_empty() {
            self.items.iter().enumerate().map(|(i, s)| (i, s.as_str())).collect()
        } else {
            let lower = self.search_text.to_lowercase();
            self.items
                .iter()
                .enumerate()
                .filter(|(_, s)| s.to_lowercase().contains(&lower))
                .map(|(i, s)| (i, s.as_str()))
                .collect()
        }
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        self.changed = false;
        match event {
            UIEvent::Click { position, button: MouseButton::Left, .. } => {
                if self.open {
                    // Check if clicking on a popup item.
                    let popup_top = rect.max.y;
                    let filtered = self.filtered_items();
                    let visible = filtered.len().min(self.max_visible_items);
                    let popup_rect = Rect::new(
                        Vec2::new(rect.min.x, popup_top),
                        Vec2::new(rect.max.x, popup_top + visible as f32 * self.item_height),
                    );
                    if popup_rect.contains(*position) {
                        let rel_y = position.y - popup_top;
                        let idx = (rel_y / self.item_height) as usize;
                        if let Some((original_idx, _)) = filtered.get(idx) {
                            self.selected_index = Some(*original_idx);
                            self.changed = true;
                        }
                        self.open = false;
                        self.search_text.clear();
                        return true;
                    }
                    self.open = false;
                    self.search_text.clear();
                    return true;
                } else if rect.contains(*position) {
                    self.open = true;
                    return true;
                }
                false
            }
            UIEvent::Hover { position } if self.open => {
                let popup_top = rect.max.y;
                let rel_y = position.y - popup_top;
                if rel_y >= 0.0 {
                    self.hovered_index = Some((rel_y / self.item_height) as usize);
                } else {
                    self.hovered_index = None;
                }
                false
            }
            UIEvent::TextInput { character } if self.open && self.searchable => {
                if !character.is_control() {
                    self.search_text.push(*character);
                }
                true
            }
            UIEvent::KeyInput { key, pressed: true, .. } if self.open => {
                use crate::core::KeyCode;
                match key {
                    KeyCode::Escape => {
                        self.open = false;
                        self.search_text.clear();
                        true
                    }
                    KeyCode::Backspace if self.searchable => {
                        self.search_text.pop();
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn layout(&self) -> Vec2 {
        Vec2::new(200.0, self.font_size + self.padding.vertical())
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        // Main button area.
        draw.draw_rounded_rect(rect, self.background_color, self.corner_radius, self.border);

        let display = self.selected_text();
        let text_y = rect.center().y - self.font_size / 2.0;
        draw.draw_text(
            if display.is_empty() { "Select..." } else { display },
            Vec2::new(rect.min.x + self.padding.left, text_y),
            self.font_size,
            if display.is_empty() { Color::GRAY } else { self.text_color },
        );

        // Down arrow.
        let arrow_x = rect.max.x - self.padding.right - 8.0;
        let arrow_y = rect.center().y;
        draw.draw_line(
            Vec2::new(arrow_x - 4.0, arrow_y - 2.0),
            Vec2::new(arrow_x, arrow_y + 2.0),
            self.text_color,
            1.5,
        );
        draw.draw_line(
            Vec2::new(arrow_x, arrow_y + 2.0),
            Vec2::new(arrow_x + 4.0, arrow_y - 2.0),
            self.text_color,
            1.5,
        );

        // Popup.
        if self.open {
            let filtered = self.filtered_items();
            let visible = filtered.len().min(self.max_visible_items);
            let popup_height = visible as f32 * self.item_height;
            let popup_rect = Rect::new(
                Vec2::new(rect.min.x, rect.max.y + 2.0),
                Vec2::new(rect.max.x, rect.max.y + 2.0 + popup_height),
            );

            draw.draw_rounded_rect(
                popup_rect,
                self.background_color,
                CornerRadii::all(4.0),
                self.border,
            );

            for (vi, (original_idx, text)) in filtered.iter().take(visible).enumerate() {
                let item_y = popup_rect.min.y + vi as f32 * self.item_height;
                let item_rect = Rect::new(
                    Vec2::new(popup_rect.min.x, item_y),
                    Vec2::new(popup_rect.max.x, item_y + self.item_height),
                );

                let is_selected = self.selected_index == Some(*original_idx);
                let is_hovered = self.hovered_index == Some(vi);

                if is_hovered || is_selected {
                    draw.draw_rect(item_rect, self.highlight_color.with_alpha(if is_selected { 0.3 } else { 0.15 }));
                }

                draw.draw_text(
                    text,
                    Vec2::new(item_rect.min.x + self.padding.left, item_rect.center().y - self.font_size / 2.0),
                    self.font_size,
                    self.text_color,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ListView
// ---------------------------------------------------------------------------

/// Virtualized scrolling list (only renders visible items).
#[derive(Debug, Clone)]
pub struct ListView {
    pub id: UIId,
    pub items: Vec<String>,
    pub selected_index: Option<usize>,
    pub item_height: f32,
    pub scroll_offset: f32,
    pub font_size: f32,
    pub text_color: Color,
    pub background_color: Color,
    pub selected_color: Color,
    pub hover_color: Color,
    pub hovered_index: Option<usize>,
    pub multi_select: bool,
    pub selected_indices: Vec<usize>,
    pub changed: bool,
}

impl ListView {
    pub fn new(items: Vec<String>) -> Self {
        Self {
            id: UIId::INVALID,
            items,
            selected_index: None,
            item_height: 28.0,
            scroll_offset: 0.0,
            font_size: 14.0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#1E1E1E"),
            selected_color: Color::from_hex("#6200EE"),
            hover_color: Color::from_rgba8(255, 255, 255, 20),
            hovered_index: None,
            multi_select: false,
            selected_indices: Vec::new(),
            changed: false,
        }
    }

    fn total_height(&self) -> f32 {
        self.items.len() as f32 * self.item_height
    }

    fn first_visible(&self, viewport_height: f32) -> (usize, usize) {
        let first = (self.scroll_offset / self.item_height) as usize;
        let count = (viewport_height / self.item_height).ceil() as usize + 1;
        (first, count)
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        self.changed = false;
        match event {
            UIEvent::Click { position, button: MouseButton::Left, modifiers, .. } => {
                if rect.contains(*position) {
                    let rel_y = position.y - rect.min.y + self.scroll_offset;
                    let idx = (rel_y / self.item_height) as usize;
                    if idx < self.items.len() {
                        if self.multi_select && modifiers.ctrl {
                            if let Some(pos) = self.selected_indices.iter().position(|&i| i == idx) {
                                self.selected_indices.remove(pos);
                            } else {
                                self.selected_indices.push(idx);
                            }
                        } else {
                            self.selected_index = Some(idx);
                            self.selected_indices = vec![idx];
                        }
                        self.changed = true;
                        return true;
                    }
                }
                false
            }
            UIEvent::Scroll { delta, .. } => {
                self.scroll_offset = (self.scroll_offset - delta.y * 28.0)
                    .clamp(0.0, (self.total_height() - rect.height()).max(0.0));
                true
            }
            UIEvent::Hover { position } => {
                if rect.contains(*position) {
                    let rel_y = position.y - rect.min.y + self.scroll_offset;
                    let idx = (rel_y / self.item_height) as usize;
                    self.hovered_index = if idx < self.items.len() { Some(idx) } else { None };
                } else {
                    self.hovered_index = None;
                }
                false
            }
            _ => false,
        }
    }

    pub fn layout(&self) -> Vec2 {
        Vec2::new(200.0, 200.0)
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        draw.draw_rect(rect, self.background_color);
        draw.push_clip(rect);

        let (first, count) = self.first_visible(rect.height());
        for i in first..(first + count).min(self.items.len()) {
            let y = rect.min.y + i as f32 * self.item_height - self.scroll_offset;
            let item_rect = Rect::new(
                Vec2::new(rect.min.x, y),
                Vec2::new(rect.max.x, y + self.item_height),
            );

            let is_selected = self.selected_indices.contains(&i)
                || self.selected_index == Some(i);
            let is_hovered = self.hovered_index == Some(i);

            if is_selected {
                draw.draw_rect(item_rect, self.selected_color.with_alpha(0.3));
            } else if is_hovered {
                draw.draw_rect(item_rect, self.hover_color);
            }

            draw.draw_text(
                &self.items[i],
                Vec2::new(item_rect.min.x + 8.0, item_rect.center().y - self.font_size / 2.0),
                self.font_size,
                self.text_color,
            );
        }

        // Scrollbar.
        let total = self.total_height();
        if total > rect.height() {
            let bar_h = (rect.height() / total * rect.height()).max(20.0);
            let bar_y = rect.min.y + (self.scroll_offset / total) * rect.height();
            let bar_rect = Rect::new(
                Vec2::new(rect.max.x - 6.0, bar_y),
                Vec2::new(rect.max.x - 2.0, bar_y + bar_h),
            );
            draw.draw_rounded_rect(bar_rect, Color::from_rgba8(128, 128, 128, 100), CornerRadii::all(2.0), Border::default());
        }

        draw.pop_clip();
    }
}

// ---------------------------------------------------------------------------
// TreeView
// ---------------------------------------------------------------------------

/// A single node in a tree view.
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub label: String,
    pub expanded: bool,
    pub children: Vec<TreeNode>,
    pub selected: bool,
    pub icon: Option<TextureId>,
    pub depth: u32,
}

impl TreeNode {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            expanded: false,
            children: Vec::new(),
            selected: false,
            icon: None,
            depth: 0,
        }
    }

    pub fn with_children(mut self, children: Vec<TreeNode>) -> Self {
        self.children = children;
        self
    }

    fn flatten(&self, depth: u32) -> Vec<FlatTreeItem> {
        let mut items = vec![FlatTreeItem {
            label: self.label.clone(),
            depth,
            has_children: !self.children.is_empty(),
            expanded: self.expanded,
            selected: self.selected,
        }];
        if self.expanded {
            for child in &self.children {
                items.extend(child.flatten(depth + 1));
            }
        }
        items
    }
}

#[derive(Debug, Clone)]
struct FlatTreeItem {
    label: String,
    depth: u32,
    has_children: bool,
    expanded: bool,
    selected: bool,
}

/// Expandable/collapsible tree view.
#[derive(Debug, Clone)]
pub struct TreeView {
    pub id: UIId,
    pub roots: Vec<TreeNode>,
    pub item_height: f32,
    pub indent: f32,
    pub font_size: f32,
    pub text_color: Color,
    pub selected_color: Color,
    pub scroll_offset: f32,
}

impl TreeView {
    pub fn new(roots: Vec<TreeNode>) -> Self {
        Self {
            id: UIId::INVALID,
            roots,
            item_height: 24.0,
            indent: 20.0,
            font_size: 14.0,
            text_color: Color::WHITE,
            selected_color: Color::from_hex("#6200EE"),
            scroll_offset: 0.0,
        }
    }

    fn flat_items(&self) -> Vec<FlatTreeItem> {
        let mut items = Vec::new();
        for root in &self.roots {
            items.extend(root.flatten(0));
        }
        items
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        draw.push_clip(rect);
        let items = self.flat_items();

        for (i, item) in items.iter().enumerate() {
            let y = rect.min.y + i as f32 * self.item_height - self.scroll_offset;
            if y + self.item_height < rect.min.y || y > rect.max.y {
                continue;
            }

            let x = rect.min.x + item.depth as f32 * self.indent;

            if item.selected {
                let sel_rect = Rect::new(
                    Vec2::new(rect.min.x, y),
                    Vec2::new(rect.max.x, y + self.item_height),
                );
                draw.draw_rect(sel_rect, self.selected_color.with_alpha(0.2));
            }

            // Expand/collapse arrow.
            if item.has_children {
                let arrow_x = x + 4.0;
                let arrow_y = y + self.item_height / 2.0;
                if item.expanded {
                    draw.draw_line(
                        Vec2::new(arrow_x, arrow_y - 3.0),
                        Vec2::new(arrow_x + 4.0, arrow_y + 3.0),
                        self.text_color,
                        1.5,
                    );
                    draw.draw_line(
                        Vec2::new(arrow_x + 4.0, arrow_y + 3.0),
                        Vec2::new(arrow_x + 8.0, arrow_y - 3.0),
                        self.text_color,
                        1.5,
                    );
                } else {
                    draw.draw_line(
                        Vec2::new(arrow_x, arrow_y - 4.0),
                        Vec2::new(arrow_x + 4.0, arrow_y),
                        self.text_color,
                        1.5,
                    );
                    draw.draw_line(
                        Vec2::new(arrow_x + 4.0, arrow_y),
                        Vec2::new(arrow_x, arrow_y + 4.0),
                        self.text_color,
                        1.5,
                    );
                }
            }

            let label_x = x + if item.has_children { 16.0 } else { 4.0 };
            draw.draw_text(
                &item.label,
                Vec2::new(label_x, y + (self.item_height - self.font_size) / 2.0),
                self.font_size,
                self.text_color,
            );
        }

        draw.pop_clip();
    }
}

// ---------------------------------------------------------------------------
// TabBar
// ---------------------------------------------------------------------------

/// Horizontal tab bar with selection indicator.
#[derive(Debug, Clone)]
pub struct TabBar {
    pub id: UIId,
    pub tabs: Vec<String>,
    pub selected: usize,
    pub font_size: f32,
    pub text_color: Color,
    pub selected_text_color: Color,
    pub indicator_color: Color,
    pub background_color: Color,
    pub tab_padding: f32,
    pub indicator_height: f32,
    pub changed: bool,
}

impl TabBar {
    pub fn new(tabs: Vec<String>) -> Self {
        Self {
            id: UIId::INVALID,
            tabs,
            selected: 0,
            font_size: 14.0,
            text_color: Color::GRAY,
            selected_text_color: Color::WHITE,
            indicator_color: Color::from_hex("#6200EE"),
            background_color: Color::from_hex("#1E1E1E"),
            tab_padding: 16.0,
            indicator_height: 3.0,
            changed: false,
        }
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        self.changed = false;
        if let UIEvent::Click { position, button: MouseButton::Left, .. } = event {
            if rect.contains(*position) {
                let tab_width = rect.width() / self.tabs.len() as f32;
                let idx = ((position.x - rect.min.x) / tab_width) as usize;
                if idx < self.tabs.len() && idx != self.selected {
                    self.selected = idx;
                    self.changed = true;
                    return true;
                }
            }
        }
        false
    }

    pub fn layout(&self) -> Vec2 {
        Vec2::new(
            self.tabs.len() as f32 * 100.0,
            self.font_size + self.tab_padding * 2.0 + self.indicator_height,
        )
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        draw.draw_rect(rect, self.background_color);

        let tab_width = rect.width() / self.tabs.len().max(1) as f32;

        for (i, label) in self.tabs.iter().enumerate() {
            let x = rect.min.x + i as f32 * tab_width;
            let is_sel = i == self.selected;

            let text_color = if is_sel {
                self.selected_text_color
            } else {
                self.text_color
            };

            let text_x = x + tab_width / 2.0;
            let text_y = rect.min.y + self.tab_padding;
            draw.draw_text_ex(
                label,
                Vec2::new(text_x, text_y),
                self.font_size,
                text_color,
                0,
                None,
                TextAlign::Center,
                crate::render_commands::TextVerticalAlign::Top,
            );

            if is_sel {
                let ind_rect = Rect::new(
                    Vec2::new(x, rect.max.y - self.indicator_height),
                    Vec2::new(x + tab_width, rect.max.y),
                );
                draw.draw_rect(ind_rect, self.indicator_color);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Panel / Window
// ---------------------------------------------------------------------------

/// A draggable, resizable panel with title bar and close button.
#[derive(Debug, Clone)]
pub struct Panel {
    pub id: UIId,
    pub title: String,
    pub position: Vec2,
    pub size: Vec2,
    pub min_size: Vec2,
    pub max_size: Vec2,
    pub draggable: bool,
    pub resizable: bool,
    pub closable: bool,
    pub visible: bool,
    pub title_bar_height: f32,
    pub title_font_size: f32,
    pub background_color: Color,
    pub title_bar_color: Color,
    pub title_text_color: Color,
    pub border: Border,
    pub corner_radius: CornerRadii,
    pub shadow: Option<Shadow>,
    // State
    dragging: bool,
    resizing: bool,
    drag_offset: Vec2,
    pub close_requested: bool,
}

impl Panel {
    pub fn new(title: &str, position: Vec2, size: Vec2) -> Self {
        Self {
            id: UIId::INVALID,
            title: title.to_string(),
            position,
            size,
            min_size: Vec2::new(100.0, 80.0),
            max_size: Vec2::new(2000.0, 2000.0),
            draggable: true,
            resizable: true,
            closable: true,
            visible: true,
            title_bar_height: 32.0,
            title_font_size: 14.0,
            background_color: Color::from_hex("#2D2D2D"),
            title_bar_color: Color::from_hex("#3D3D3D"),
            title_text_color: Color::WHITE,
            border: Border::new(Color::from_rgba8(80, 80, 80, 200), 1.0),
            corner_radius: CornerRadii::all(8.0),
            shadow: Some(Shadow::new(
                Color::from_rgba8(0, 0, 0, 80),
                Vec2::new(0.0, 4.0),
                12.0,
                2.0,
            )),
            dragging: false,
            resizing: false,
            drag_offset: Vec2::ZERO,
            close_requested: false,
        }
    }

    pub fn rect(&self) -> Rect {
        Rect::new(self.position, self.position + self.size)
    }

    pub fn title_bar_rect(&self) -> Rect {
        Rect::new(
            self.position,
            Vec2::new(self.position.x + self.size.x, self.position.y + self.title_bar_height),
        )
    }

    pub fn content_rect(&self) -> Rect {
        Rect::new(
            Vec2::new(self.position.x, self.position.y + self.title_bar_height),
            self.position + self.size,
        )
    }

    pub fn close_button_rect(&self) -> Rect {
        let btn_size = self.title_bar_height - 8.0;
        let x = self.position.x + self.size.x - btn_size - 4.0;
        let y = self.position.y + 4.0;
        Rect::new(Vec2::new(x, y), Vec2::new(x + btn_size, y + btn_size))
    }

    fn resize_handle_rect(&self) -> Rect {
        let handle_size = 12.0;
        Rect::new(
            Vec2::new(
                self.position.x + self.size.x - handle_size,
                self.position.y + self.size.y - handle_size,
            ),
            self.position + self.size,
        )
    }

    pub fn update(&mut self, event: &UIEvent) -> bool {
        self.close_requested = false;
        if !self.visible {
            return false;
        }

        match event {
            UIEvent::Click { position, button: MouseButton::Left, .. } => {
                // Close button.
                if self.closable && self.close_button_rect().contains(*position) {
                    self.close_requested = true;
                    return true;
                }
                // Resize handle.
                if self.resizable && self.resize_handle_rect().contains(*position) {
                    self.resizing = true;
                    self.drag_offset = *position - (self.position + self.size);
                    return true;
                }
                // Title bar drag.
                if self.draggable && self.title_bar_rect().contains(*position) {
                    self.dragging = true;
                    self.drag_offset = *position - self.position;
                    return true;
                }
                false
            }
            UIEvent::DragMove { position, .. } => {
                if self.dragging {
                    self.position = *position - self.drag_offset;
                    return true;
                }
                if self.resizing {
                    let new_br = *position - self.drag_offset;
                    let new_size = (new_br - self.position)
                        .max(self.min_size)
                        .min(self.max_size);
                    self.size = new_size;
                    return true;
                }
                false
            }
            UIEvent::DragEnd { .. } | UIEvent::MouseUp { .. } => {
                self.dragging = false;
                self.resizing = false;
                false
            }
            _ => false,
        }
    }

    pub fn render(&self, draw: &mut DrawList) {
        if !self.visible {
            return;
        }

        let rect = self.rect();

        // Shadow.
        if let Some(shadow) = &self.shadow {
            draw.draw_rounded_rect_with_shadow(rect, self.background_color, self.corner_radius, self.border, *shadow);
        } else {
            draw.draw_rounded_rect(rect, self.background_color, self.corner_radius, self.border);
        }

        // Title bar.
        let tb = self.title_bar_rect();
        let tb_radii = CornerRadii::new(
            self.corner_radius.top_left,
            self.corner_radius.top_right,
            0.0,
            0.0,
        );
        draw.draw_rounded_rect(tb, self.title_bar_color, tb_radii, Border::default());

        // Title text.
        draw.draw_text(
            &self.title,
            Vec2::new(tb.min.x + 12.0, tb.center().y - self.title_font_size / 2.0),
            self.title_font_size,
            self.title_text_color,
        );

        // Close button.
        if self.closable {
            let cb = self.close_button_rect();
            let cx = cb.center().x;
            let cy = cb.center().y;
            let s = 5.0;
            draw.draw_line(
                Vec2::new(cx - s, cy - s),
                Vec2::new(cx + s, cy + s),
                Color::from_rgba8(200, 200, 200, 200),
                1.5,
            );
            draw.draw_line(
                Vec2::new(cx + s, cy - s),
                Vec2::new(cx - s, cy + s),
                Color::from_rgba8(200, 200, 200, 200),
                1.5,
            );
        }

        // Resize handle.
        if self.resizable {
            let rh = self.resize_handle_rect();
            for i in 0..3 {
                let offset = i as f32 * 4.0;
                draw.draw_line(
                    Vec2::new(rh.max.x - 2.0 - offset, rh.max.y - 2.0),
                    Vec2::new(rh.max.x - 2.0, rh.max.y - 2.0 - offset),
                    Color::from_rgba8(100, 100, 100, 150),
                    1.0,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ScrollView
// ---------------------------------------------------------------------------

/// A scrollable container with scrollbars.
#[derive(Debug, Clone)]
pub struct ScrollView {
    pub id: UIId,
    pub scroll_offset: Vec2,
    pub content_size: Vec2,
    pub horizontal: bool,
    pub vertical: bool,
    pub scrollbar_width: f32,
    pub scrollbar_color: Color,
    pub track_color: Color,
    pub show_scrollbars: bool,
    pub auto_hide_scrollbars: bool,
    pub scrollbar_opacity: f32,
}

impl ScrollView {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            scroll_offset: Vec2::ZERO,
            content_size: Vec2::ZERO,
            horizontal: false,
            vertical: true,
            scrollbar_width: 8.0,
            scrollbar_color: Color::from_rgba8(128, 128, 128, 150),
            track_color: Color::from_rgba8(40, 40, 40, 100),
            show_scrollbars: true,
            auto_hide_scrollbars: true,
            scrollbar_opacity: 1.0,
        }
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        if let UIEvent::Scroll { delta, .. } = event {
            let max_x = (self.content_size.x - rect.width()).max(0.0);
            let max_y = (self.content_size.y - rect.height()).max(0.0);
            if self.horizontal {
                self.scroll_offset.x = (self.scroll_offset.x - delta.x * 20.0).clamp(0.0, max_x);
            }
            if self.vertical {
                self.scroll_offset.y = (self.scroll_offset.y - delta.y * 20.0).clamp(0.0, max_y);
            }
            return true;
        }
        false
    }

    pub fn render_scrollbars(&self, draw: &mut DrawList, rect: Rect) {
        if !self.show_scrollbars {
            return;
        }

        let opacity = self.scrollbar_opacity;
        let sb_color = self.scrollbar_color.with_alpha(self.scrollbar_color.a * opacity);

        // Vertical scrollbar.
        if self.vertical && self.content_size.y > rect.height() {
            let ratio = rect.height() / self.content_size.y;
            let thumb_h = (ratio * rect.height()).max(20.0);
            let max_offset = self.content_size.y - rect.height();
            let thumb_y = if max_offset > 0.0 {
                rect.min.y + (self.scroll_offset.y / max_offset) * (rect.height() - thumb_h)
            } else {
                rect.min.y
            };

            let track = Rect::new(
                Vec2::new(rect.max.x - self.scrollbar_width, rect.min.y),
                Vec2::new(rect.max.x, rect.max.y),
            );
            draw.draw_rect(track, self.track_color.with_alpha(self.track_color.a * opacity));

            let thumb = Rect::new(
                Vec2::new(rect.max.x - self.scrollbar_width, thumb_y),
                Vec2::new(rect.max.x, thumb_y + thumb_h),
            );
            draw.draw_rounded_rect(thumb, sb_color, CornerRadii::all(self.scrollbar_width / 2.0), Border::default());
        }

        // Horizontal scrollbar.
        if self.horizontal && self.content_size.x > rect.width() {
            let ratio = rect.width() / self.content_size.x;
            let thumb_w = (ratio * rect.width()).max(20.0);
            let max_offset = self.content_size.x - rect.width();
            let thumb_x = if max_offset > 0.0 {
                rect.min.x + (self.scroll_offset.x / max_offset) * (rect.width() - thumb_w)
            } else {
                rect.min.x
            };

            let track = Rect::new(
                Vec2::new(rect.min.x, rect.max.y - self.scrollbar_width),
                Vec2::new(rect.max.x, rect.max.y),
            );
            draw.draw_rect(track, self.track_color.with_alpha(self.track_color.a * opacity));

            let thumb = Rect::new(
                Vec2::new(thumb_x, rect.max.y - self.scrollbar_width),
                Vec2::new(thumb_x + thumb_w, rect.max.y),
            );
            draw.draw_rounded_rect(thumb, sb_color, CornerRadii::all(self.scrollbar_width / 2.0), Border::default());
        }
    }
}

// ---------------------------------------------------------------------------
// Image widget
// ---------------------------------------------------------------------------

/// Texture display with scaling modes.
#[derive(Debug, Clone)]
pub struct ImageWidget {
    pub id: UIId,
    pub texture: TextureId,
    pub scale_mode: ImageScaleMode,
    pub tint: Color,
    pub corner_radius: CornerRadii,
    pub source_size: Vec2,
}

impl ImageWidget {
    pub fn new(texture: TextureId, source_size: Vec2) -> Self {
        Self {
            id: UIId::INVALID,
            texture,
            scale_mode: ImageScaleMode::Fit,
            tint: Color::WHITE,
            corner_radius: CornerRadii::ZERO,
            source_size,
        }
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        let dest_rect = match self.scale_mode {
            ImageScaleMode::Stretch => rect,
            ImageScaleMode::Fit => {
                let aspect = self.source_size.x / self.source_size.y.max(0.001);
                let dest_aspect = rect.width() / rect.height().max(0.001);
                if aspect > dest_aspect {
                    let h = rect.width() / aspect;
                    let y = rect.min.y + (rect.height() - h) / 2.0;
                    Rect::new(Vec2::new(rect.min.x, y), Vec2::new(rect.max.x, y + h))
                } else {
                    let w = rect.height() * aspect;
                    let x = rect.min.x + (rect.width() - w) / 2.0;
                    Rect::new(Vec2::new(x, rect.min.y), Vec2::new(x + w, rect.max.y))
                }
            }
            ImageScaleMode::Fill => {
                let aspect = self.source_size.x / self.source_size.y.max(0.001);
                let dest_aspect = rect.width() / rect.height().max(0.001);
                if aspect < dest_aspect {
                    let h = rect.width() / aspect;
                    let y = rect.min.y + (rect.height() - h) / 2.0;
                    Rect::new(Vec2::new(rect.min.x, y), Vec2::new(rect.max.x, y + h))
                } else {
                    let w = rect.height() * aspect;
                    let x = rect.min.x + (rect.width() - w) / 2.0;
                    Rect::new(Vec2::new(x, rect.min.y), Vec2::new(x + w, rect.max.y))
                }
            }
            ImageScaleMode::Center => {
                let x = rect.min.x + (rect.width() - self.source_size.x) / 2.0;
                let y = rect.min.y + (rect.height() - self.source_size.y) / 2.0;
                Rect::new(Vec2::new(x, y), Vec2::new(x + self.source_size.x, y + self.source_size.y))
            }
            ImageScaleMode::Tile => rect, // Tiling handled by renderer.
        };

        draw.push(crate::render_commands::DrawCommand::Image {
            rect: dest_rect,
            texture: self.texture,
            tint: self.tint,
            corner_radii: self.corner_radius,
            scale_mode: self.scale_mode,
            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
        });
    }
}

// ---------------------------------------------------------------------------
// Divider / Separator
// ---------------------------------------------------------------------------

/// Horizontal or vertical divider line.
#[derive(Debug, Clone)]
pub struct Divider {
    pub id: UIId,
    pub vertical: bool,
    pub color: Color,
    pub thickness: f32,
    pub margin: f32,
}

impl Divider {
    pub fn horizontal() -> Self {
        Self {
            id: UIId::INVALID,
            vertical: false,
            color: Color::from_rgba8(80, 80, 80, 200),
            thickness: 1.0,
            margin: 8.0,
        }
    }

    pub fn vertical() -> Self {
        Self {
            id: UIId::INVALID,
            vertical: true,
            color: Color::from_rgba8(80, 80, 80, 200),
            thickness: 1.0,
            margin: 8.0,
        }
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        if self.vertical {
            let x = rect.center().x;
            draw.draw_line(
                Vec2::new(x, rect.min.y + self.margin),
                Vec2::new(x, rect.max.y - self.margin),
                self.color,
                self.thickness,
            );
        } else {
            let y = rect.center().y;
            draw.draw_line(
                Vec2::new(rect.min.x + self.margin, y),
                Vec2::new(rect.max.x - self.margin, y),
                self.color,
                self.thickness,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

/// Delayed popup on hover.
#[derive(Debug, Clone)]
pub struct Tooltip {
    pub id: UIId,
    pub text: String,
    pub font_size: f32,
    pub text_color: Color,
    pub background_color: Color,
    pub border: Border,
    pub corner_radius: CornerRadii,
    pub padding: Padding,
    pub delay: f32,
    pub visible: bool,
    pub hover_timer: f32,
    pub position: Vec2,
    pub max_width: f32,
}

impl Tooltip {
    pub fn new(text: &str) -> Self {
        Self {
            id: UIId::INVALID,
            text: text.to_string(),
            font_size: 12.0,
            text_color: Color::WHITE,
            background_color: Color::from_rgba8(50, 50, 50, 230),
            border: Border::default(),
            corner_radius: CornerRadii::all(4.0),
            padding: Padding::new(8.0, 4.0, 8.0, 4.0),
            delay: 0.5,
            visible: false,
            hover_timer: 0.0,
            position: Vec2::ZERO,
            max_width: 300.0,
        }
    }

    pub fn update(&mut self, hovering: bool, mouse_pos: Vec2, dt: f32) {
        if hovering {
            self.hover_timer += dt;
            self.position = mouse_pos + Vec2::new(12.0, 12.0);
            if self.hover_timer >= self.delay {
                self.visible = true;
            }
        } else {
            self.hover_timer = 0.0;
            self.visible = false;
        }
    }

    pub fn render(&self, draw: &mut DrawList) {
        if !self.visible {
            return;
        }
        let font = default_font();
        let m = measure_text(&self.text, &font, self.font_size, Some(self.max_width));
        let w = m.width + self.padding.horizontal();
        let h = m.height + self.padding.vertical();
        let rect = Rect::new(self.position, self.position + Vec2::new(w, h));

        draw.draw_rounded_rect(rect, self.background_color, self.corner_radius, self.border);
        draw.draw_text(
            &self.text,
            Vec2::new(rect.min.x + self.padding.left, rect.min.y + self.padding.top),
            self.font_size,
            self.text_color,
        );
    }
}

// ---------------------------------------------------------------------------
// ContextMenu
// ---------------------------------------------------------------------------

/// Right-click popup menu.
#[derive(Debug, Clone)]
pub struct ContextMenu {
    pub id: UIId,
    pub items: Vec<MenuItem>,
    pub position: Vec2,
    pub visible: bool,
    pub font_size: f32,
    pub text_color: Color,
    pub background_color: Color,
    pub highlight_color: Color,
    pub item_height: f32,
    pub padding: Padding,
    pub min_width: f32,
    pub hovered_index: Option<usize>,
    pub selected_index: Option<usize>,
}

/// A single menu item.
#[derive(Debug, Clone)]
pub struct MenuItem {
    pub label: String,
    pub shortcut: Option<String>,
    pub enabled: bool,
    pub separator: bool,
    pub icon: Option<TextureId>,
}

impl MenuItem {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            shortcut: None,
            enabled: true,
            separator: false,
            icon: None,
        }
    }

    pub fn with_shortcut(mut self, shortcut: &str) -> Self {
        self.shortcut = Some(shortcut.to_string());
        self
    }

    pub fn separator() -> Self {
        Self {
            label: String::new(),
            shortcut: None,
            enabled: false,
            separator: true,
            icon: None,
        }
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

impl ContextMenu {
    pub fn new(items: Vec<MenuItem>) -> Self {
        Self {
            id: UIId::INVALID,
            items,
            position: Vec2::ZERO,
            visible: false,
            font_size: 13.0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#2D2D2D"),
            highlight_color: Color::from_hex("#6200EE"),
            item_height: 26.0,
            padding: Padding::new(4.0, 4.0, 4.0, 4.0),
            min_width: 150.0,
            hovered_index: None,
            selected_index: None,
        }
    }

    pub fn show(&mut self, position: Vec2) {
        self.position = position;
        self.visible = true;
        self.selected_index = None;
    }

    pub fn hide(&mut self) {
        self.visible = false;
    }

    pub fn update(&mut self, event: &UIEvent) -> bool {
        if !self.visible {
            return false;
        }
        self.selected_index = None;

        match event {
            UIEvent::Click { position, button: MouseButton::Left, .. } => {
                let menu_rect = self.menu_rect();
                if menu_rect.contains(*position) {
                    let rel_y = position.y - menu_rect.min.y - self.padding.top;
                    let idx = (rel_y / self.item_height) as usize;
                    if let Some(item) = self.items.get(idx) {
                        if item.enabled && !item.separator {
                            self.selected_index = Some(idx);
                            self.visible = false;
                            return true;
                        }
                    }
                } else {
                    self.visible = false;
                }
                true
            }
            UIEvent::Hover { position } => {
                let menu_rect = self.menu_rect();
                if menu_rect.contains(*position) {
                    let rel_y = position.y - menu_rect.min.y - self.padding.top;
                    self.hovered_index = Some((rel_y / self.item_height) as usize);
                } else {
                    self.hovered_index = None;
                }
                false
            }
            UIEvent::KeyInput { key, pressed: true, .. } => {
                if *key == crate::core::KeyCode::Escape {
                    self.visible = false;
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    fn menu_rect(&self) -> Rect {
        let separator_count = self.items.iter().filter(|i| i.separator).count();
        let item_count = self.items.len() - separator_count;
        let h = item_count as f32 * self.item_height
            + separator_count as f32 * 8.0
            + self.padding.vertical();
        Rect::new(
            self.position,
            self.position + Vec2::new(self.min_width, h),
        )
    }

    pub fn render(&self, draw: &mut DrawList) {
        if !self.visible {
            return;
        }

        let rect = self.menu_rect();
        draw.draw_rounded_rect(
            rect,
            self.background_color,
            CornerRadii::all(6.0),
            Border::new(Color::from_rgba8(60, 60, 60, 200), 1.0),
        );

        let mut y = rect.min.y + self.padding.top;
        for (i, item) in self.items.iter().enumerate() {
            if item.separator {
                let sep_y = y + 4.0;
                draw.draw_line(
                    Vec2::new(rect.min.x + 8.0, sep_y),
                    Vec2::new(rect.max.x - 8.0, sep_y),
                    Color::from_rgba8(80, 80, 80, 200),
                    1.0,
                );
                y += 8.0;
                continue;
            }

            let item_rect = Rect::new(
                Vec2::new(rect.min.x + 2.0, y),
                Vec2::new(rect.max.x - 2.0, y + self.item_height),
            );

            if self.hovered_index == Some(i) && item.enabled {
                draw.draw_rounded_rect(
                    item_rect,
                    self.highlight_color.with_alpha(0.2),
                    CornerRadii::all(4.0),
                    Border::default(),
                );
            }

            let text_color = if item.enabled {
                self.text_color
            } else {
                Color::GRAY
            };

            draw.draw_text(
                &item.label,
                Vec2::new(item_rect.min.x + 8.0, item_rect.center().y - self.font_size / 2.0),
                self.font_size,
                text_color,
            );

            if let Some(shortcut) = &item.shortcut {
                draw.draw_text_ex(
                    shortcut,
                    Vec2::new(item_rect.max.x - 8.0, item_rect.center().y - self.font_size / 2.0),
                    self.font_size - 1.0,
                    Color::GRAY,
                    0,
                    None,
                    TextAlign::Right,
                    crate::render_commands::TextVerticalAlign::Top,
                );
            }

            y += self.item_height;
        }
    }
}

// ---------------------------------------------------------------------------
// Modal / Dialog
// ---------------------------------------------------------------------------

/// Centered overlay dialog with backdrop.
#[derive(Debug, Clone)]
pub struct Modal {
    pub id: UIId,
    pub title: String,
    pub content: String,
    pub visible: bool,
    pub width: f32,
    pub padding: Padding,
    pub background_color: Color,
    pub backdrop_color: Color,
    pub title_font_size: f32,
    pub content_font_size: f32,
    pub corner_radius: CornerRadii,
    pub shadow: Shadow,
    pub buttons: Vec<String>,
    pub pressed_button: Option<usize>,
}

impl Modal {
    pub fn new(title: &str, content: &str) -> Self {
        Self {
            id: UIId::INVALID,
            title: title.to_string(),
            content: content.to_string(),
            visible: false,
            width: 400.0,
            padding: Padding::all(20.0),
            background_color: Color::from_hex("#2D2D2D"),
            backdrop_color: Color::from_rgba8(0, 0, 0, 128),
            title_font_size: 18.0,
            content_font_size: 14.0,
            corner_radius: CornerRadii::all(12.0),
            shadow: Shadow::new(Color::from_rgba8(0, 0, 0, 100), Vec2::new(0.0, 8.0), 24.0, 4.0),
            buttons: vec!["OK".to_string()],
            pressed_button: None,
        }
    }

    pub fn confirm(title: &str, message: &str) -> Self {
        let mut m = Self::new(title, message);
        m.buttons = vec!["Cancel".to_string(), "OK".to_string()];
        m
    }

    pub fn show(&mut self) {
        self.visible = true;
        self.pressed_button = None;
    }

    pub fn hide(&mut self) {
        self.visible = false;
    }

    pub fn update(&mut self, event: &UIEvent, screen_size: Vec2) -> bool {
        self.pressed_button = None;
        if !self.visible {
            return false;
        }
        if let UIEvent::Click { position, button: MouseButton::Left, .. } = event {
            let dialog_rect = self.dialog_rect(screen_size);
            if !dialog_rect.contains(*position) {
                // Click on backdrop -- could close.
                return true;
            }
            // Check button clicks.
            let btn_y = dialog_rect.max.y - self.padding.bottom - 36.0;
            let btn_width = 80.0;
            let btn_gap = 8.0;
            let total_w = self.buttons.len() as f32 * btn_width
                + (self.buttons.len().saturating_sub(1)) as f32 * btn_gap;
            let start_x = dialog_rect.max.x - self.padding.right - total_w;

            for (i, _) in self.buttons.iter().enumerate() {
                let bx = start_x + i as f32 * (btn_width + btn_gap);
                let btn_rect = Rect::new(
                    Vec2::new(bx, btn_y),
                    Vec2::new(bx + btn_width, btn_y + 32.0),
                );
                if btn_rect.contains(*position) {
                    self.pressed_button = Some(i);
                    self.visible = false;
                    return true;
                }
            }
            return true;
        }
        false
    }

    fn dialog_rect(&self, screen_size: Vec2) -> Rect {
        let font = default_font();
        let content_m = measure_text(
            &self.content,
            &font,
            self.content_font_size,
            Some(self.width - self.padding.horizontal()),
        );
        let h = self.padding.vertical()
            + self.title_font_size * 1.5
            + 12.0
            + content_m.height
            + 12.0
            + 36.0;
        let x = (screen_size.x - self.width) / 2.0;
        let y = (screen_size.y - h) / 2.0;
        Rect::new(Vec2::new(x, y), Vec2::new(x + self.width, y + h))
    }

    pub fn render(&self, draw: &mut DrawList, screen_size: Vec2) {
        if !self.visible {
            return;
        }

        // Backdrop.
        draw.draw_rect(
            Rect::new(Vec2::ZERO, screen_size),
            self.backdrop_color,
        );

        let rect = self.dialog_rect(screen_size);
        draw.draw_rounded_rect_with_shadow(
            rect,
            self.background_color,
            self.corner_radius,
            Border::default(),
            self.shadow,
        );

        // Title.
        draw.draw_text(
            &self.title,
            Vec2::new(rect.min.x + self.padding.left, rect.min.y + self.padding.top),
            self.title_font_size,
            Color::WHITE,
        );

        // Content.
        let content_y = rect.min.y + self.padding.top + self.title_font_size * 1.5 + 12.0;
        draw.draw_text_ex(
            &self.content,
            Vec2::new(rect.min.x + self.padding.left, content_y),
            self.content_font_size,
            Color::from_rgba8(200, 200, 200, 255),
            0,
            Some(self.width - self.padding.horizontal()),
            TextAlign::Left,
            crate::render_commands::TextVerticalAlign::Top,
        );

        // Buttons.
        let btn_y = rect.max.y - self.padding.bottom - 36.0;
        let btn_width = 80.0;
        let btn_gap = 8.0;
        let total_w = self.buttons.len() as f32 * btn_width
            + (self.buttons.len().saturating_sub(1)) as f32 * btn_gap;
        let start_x = rect.max.x - self.padding.right - total_w;

        for (i, label) in self.buttons.iter().enumerate() {
            let bx = start_x + i as f32 * (btn_width + btn_gap);
            let btn_rect = Rect::new(
                Vec2::new(bx, btn_y),
                Vec2::new(bx + btn_width, btn_y + 32.0),
            );
            let is_primary = i == self.buttons.len() - 1;
            let bg = if is_primary {
                Color::from_hex("#6200EE")
            } else {
                Color::from_rgba8(80, 80, 80, 200)
            };
            draw.draw_rounded_rect(btn_rect, bg, CornerRadii::all(6.0), Border::default());
            draw.draw_text_ex(
                label,
                Vec2::new(btn_rect.center().x, btn_rect.center().y - 7.0),
                13.0,
                Color::WHITE,
                0,
                None,
                TextAlign::Center,
                crate::render_commands::TextVerticalAlign::Top,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// ColorPicker
// ---------------------------------------------------------------------------

/// HSV color picker with a hue wheel, saturation/value square, and hex input.
#[derive(Debug, Clone)]
pub struct ColorPicker {
    pub id: UIId,
    pub color: Color,
    pub hue: f32,
    pub saturation: f32,
    pub value: f32,
    pub alpha: f32,
    pub show_alpha: bool,
    pub show_hex: bool,
    pub hex_text: String,
    pub size: f32,
    pub changed: bool,
    dragging_hue: bool,
    dragging_sv: bool,
    dragging_alpha: bool,
}

impl ColorPicker {
    pub fn new(initial_color: Color) -> Self {
        let (h, s, v) = initial_color.to_hsv();
        Self {
            id: UIId::INVALID,
            color: initial_color,
            hue: h,
            saturation: s,
            value: v,
            alpha: initial_color.a,
            show_alpha: true,
            show_hex: true,
            hex_text: format!(
                "#{:02X}{:02X}{:02X}",
                (initial_color.r * 255.0) as u8,
                (initial_color.g * 255.0) as u8,
                (initial_color.b * 255.0) as u8
            ),
            size: 200.0,
            changed: false,
            dragging_hue: false,
            dragging_sv: false,
            dragging_alpha: false,
        }
    }

    fn update_color(&mut self) {
        self.color = Color::from_hsv(self.hue, self.saturation, self.value).with_alpha(self.alpha);
        self.hex_text = format!(
            "#{:02X}{:02X}{:02X}",
            (self.color.r * 255.0) as u8,
            (self.color.g * 255.0) as u8,
            (self.color.b * 255.0) as u8
        );
        self.changed = true;
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        self.changed = false;
        let sv_rect = Rect::new(
            rect.min,
            Vec2::new(rect.min.x + self.size, rect.min.y + self.size),
        );
        let hue_rect = Rect::new(
            Vec2::new(rect.min.x + self.size + 8.0, rect.min.y),
            Vec2::new(rect.min.x + self.size + 28.0, rect.min.y + self.size),
        );

        match event {
            UIEvent::Click { position, button: MouseButton::Left, .. }
            | UIEvent::DragStart { position, button: MouseButton::Left } => {
                if sv_rect.contains(*position) {
                    self.dragging_sv = true;
                    self.saturation = ((position.x - sv_rect.min.x) / sv_rect.width()).clamp(0.0, 1.0);
                    self.value = 1.0 - ((position.y - sv_rect.min.y) / sv_rect.height()).clamp(0.0, 1.0);
                    self.update_color();
                    return true;
                }
                if hue_rect.contains(*position) {
                    self.dragging_hue = true;
                    self.hue = ((position.y - hue_rect.min.y) / hue_rect.height()).clamp(0.0, 1.0) * 360.0;
                    self.update_color();
                    return true;
                }
                false
            }
            UIEvent::DragMove { position, .. } => {
                if self.dragging_sv {
                    self.saturation = ((position.x - sv_rect.min.x) / sv_rect.width()).clamp(0.0, 1.0);
                    self.value = 1.0 - ((position.y - sv_rect.min.y) / sv_rect.height()).clamp(0.0, 1.0);
                    self.update_color();
                    return true;
                }
                if self.dragging_hue {
                    self.hue = ((position.y - hue_rect.min.y) / hue_rect.height()).clamp(0.0, 1.0) * 360.0;
                    self.update_color();
                    return true;
                }
                false
            }
            UIEvent::DragEnd { .. } | UIEvent::MouseUp { .. } => {
                self.dragging_sv = false;
                self.dragging_hue = false;
                self.dragging_alpha = false;
                false
            }
            _ => false,
        }
    }

    pub fn layout(&self) -> Vec2 {
        let width = self.size + 36.0;
        let height = self.size + if self.show_hex { 30.0 } else { 0.0 };
        Vec2::new(width, height)
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        // SV square (simplified: draw as a gradient using the background and
        // a few overlaid rects).
        let sv_rect = Rect::new(
            rect.min,
            Vec2::new(rect.min.x + self.size, rect.min.y + self.size),
        );
        let hue_color = Color::from_hsv(self.hue, 1.0, 1.0);
        draw.draw_rect(sv_rect, hue_color);

        // White-to-transparent horizontal gradient (saturation).
        draw.push(crate::render_commands::DrawCommand::GradientRect {
            rect: sv_rect,
            gradient: crate::render_commands::Gradient::Linear {
                start: sv_rect.min,
                end: Vec2::new(sv_rect.max.x, sv_rect.min.y),
                stops: vec![
                    crate::render_commands::GradientStop { offset: 0.0, color: Color::WHITE },
                    crate::render_commands::GradientStop { offset: 1.0, color: Color::TRANSPARENT },
                ],
            },
            corner_radii: CornerRadii::ZERO,
        });
        // Black-to-transparent vertical gradient (value).
        draw.push(crate::render_commands::DrawCommand::GradientRect {
            rect: sv_rect,
            gradient: crate::render_commands::Gradient::Linear {
                start: Vec2::new(sv_rect.min.x, sv_rect.min.y),
                end: Vec2::new(sv_rect.min.x, sv_rect.max.y),
                stops: vec![
                    crate::render_commands::GradientStop { offset: 0.0, color: Color::TRANSPARENT },
                    crate::render_commands::GradientStop { offset: 1.0, color: Color::BLACK },
                ],
            },
            corner_radii: CornerRadii::ZERO,
        });

        // SV cursor.
        let cursor_x = sv_rect.min.x + self.saturation * sv_rect.width();
        let cursor_y = sv_rect.min.y + (1.0 - self.value) * sv_rect.height();
        draw.draw_circle(Vec2::new(cursor_x, cursor_y), 6.0, Color::TRANSPARENT);
        draw.push(crate::render_commands::DrawCommand::Circle {
            center: Vec2::new(cursor_x, cursor_y),
            radius: 6.0,
            color: Color::TRANSPARENT,
            border: Border::new(Color::WHITE, 2.0),
        });

        // Hue strip.
        let hue_x = rect.min.x + self.size + 8.0;
        let hue_w = 20.0;
        let hue_strip = Rect::new(
            Vec2::new(hue_x, rect.min.y),
            Vec2::new(hue_x + hue_w, rect.min.y + self.size),
        );
        let num_segments = 6;
        let segment_h = self.size / num_segments as f32;
        let hue_colors = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0];
        for i in 0..num_segments {
            let sy = hue_strip.min.y + i as f32 * segment_h;
            let ey = sy + segment_h;
            let seg = Rect::new(
                Vec2::new(hue_strip.min.x, sy),
                Vec2::new(hue_strip.max.x, ey),
            );
            let top_color = Color::from_hsv(hue_colors[i], 1.0, 1.0);
            let bot_color = Color::from_hsv(hue_colors[i + 1], 1.0, 1.0);
            draw.push(crate::render_commands::DrawCommand::GradientRect {
                rect: seg,
                gradient: crate::render_commands::Gradient::Linear {
                    start: Vec2::new(seg.min.x, sy),
                    end: Vec2::new(seg.min.x, ey),
                    stops: vec![
                        crate::render_commands::GradientStop { offset: 0.0, color: top_color },
                        crate::render_commands::GradientStop { offset: 1.0, color: bot_color },
                    ],
                },
                corner_radii: CornerRadii::ZERO,
            });
        }

        // Hue indicator.
        let hue_y = hue_strip.min.y + (self.hue / 360.0) * hue_strip.height();
        draw.draw_line(
            Vec2::new(hue_strip.min.x - 2.0, hue_y),
            Vec2::new(hue_strip.max.x + 2.0, hue_y),
            Color::WHITE,
            2.0,
        );

        // Color preview.
        let preview_rect = Rect::new(
            Vec2::new(rect.min.x, rect.min.y + self.size + 8.0),
            Vec2::new(rect.min.x + 30.0, rect.min.y + self.size + 28.0),
        );
        draw.draw_rounded_rect(
            preview_rect,
            self.color,
            CornerRadii::all(4.0),
            Border::new(Color::from_rgba8(100, 100, 100, 200), 1.0),
        );

        // Hex text.
        if self.show_hex {
            draw.draw_text(
                &self.hex_text,
                Vec2::new(rect.min.x + 38.0, rect.min.y + self.size + 12.0),
                12.0,
                Color::WHITE,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Spinner / NumberInput
// ---------------------------------------------------------------------------

/// Numeric input with increment/decrement buttons.
#[derive(Debug, Clone)]
pub struct NumberInput {
    pub id: UIId,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub precision: usize,
    pub font_size: f32,
    pub text_color: Color,
    pub background_color: Color,
    pub button_color: Color,
    pub border: Border,
    pub width: f32,
    pub enabled: bool,
    pub changed: bool,
    pub editing: bool,
    pub edit_text: String,
}

impl NumberInput {
    pub fn new(value: f64, min: f64, max: f64, step: f64) -> Self {
        Self {
            id: UIId::INVALID,
            value: value.clamp(min, max),
            min,
            max,
            step,
            precision: 2,
            font_size: 14.0,
            text_color: Color::WHITE,
            background_color: Color::from_hex("#2D2D2D"),
            button_color: Color::from_rgba8(80, 80, 80, 200),
            border: Border::new(Color::from_rgba8(100, 100, 100, 200), 1.0),
            width: 120.0,
            enabled: true,
            changed: false,
            editing: false,
            edit_text: String::new(),
        }
    }

    fn display_text(&self) -> String {
        format!("{:.prec$}", self.value, prec = self.precision)
    }

    pub fn increment(&mut self) {
        let old = self.value;
        self.value = (self.value + self.step).min(self.max);
        self.changed = (self.value - old).abs() > 1e-12;
    }

    pub fn decrement(&mut self) {
        let old = self.value;
        self.value = (self.value - self.step).max(self.min);
        self.changed = (self.value - old).abs() > 1e-12;
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        self.changed = false;
        if !self.enabled {
            return false;
        }

        let btn_w = 24.0;
        let dec_rect = Rect::new(
            rect.min,
            Vec2::new(rect.min.x + btn_w, rect.max.y),
        );
        let inc_rect = Rect::new(
            Vec2::new(rect.max.x - btn_w, rect.min.y),
            rect.max,
        );

        match event {
            UIEvent::Click { position, button: MouseButton::Left, .. } => {
                if dec_rect.contains(*position) {
                    self.decrement();
                    return true;
                }
                if inc_rect.contains(*position) {
                    self.increment();
                    return true;
                }
                false
            }
            UIEvent::Scroll { delta, .. } => {
                if rect.contains(Vec2::new(rect.center().x, rect.center().y)) {
                    if delta.y > 0.0 {
                        self.increment();
                    } else if delta.y < 0.0 {
                        self.decrement();
                    }
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    pub fn layout(&self) -> Vec2 {
        Vec2::new(self.width, self.font_size + 12.0)
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        draw.draw_rounded_rect(rect, self.background_color, CornerRadii::all(4.0), self.border);

        let btn_w = 24.0;

        // Decrement button.
        let dec_rect = Rect::new(
            rect.min,
            Vec2::new(rect.min.x + btn_w, rect.max.y),
        );
        draw.draw_rounded_rect(
            dec_rect,
            self.button_color,
            CornerRadii::new(4.0, 0.0, 0.0, 4.0),
            Border::default(),
        );
        draw.draw_text(
            "-",
            Vec2::new(dec_rect.center().x - 3.0, dec_rect.center().y - self.font_size / 2.0),
            self.font_size,
            self.text_color,
        );

        // Increment button.
        let inc_rect = Rect::new(
            Vec2::new(rect.max.x - btn_w, rect.min.y),
            rect.max,
        );
        draw.draw_rounded_rect(
            inc_rect,
            self.button_color,
            CornerRadii::new(0.0, 4.0, 4.0, 0.0),
            Border::default(),
        );
        draw.draw_text(
            "+",
            Vec2::new(inc_rect.center().x - 3.0, inc_rect.center().y - self.font_size / 2.0),
            self.font_size,
            self.text_color,
        );

        // Value text.
        let text = self.display_text();
        draw.draw_text_ex(
            &text,
            Vec2::new(rect.center().x, rect.center().y - self.font_size / 2.0),
            self.font_size,
            self.text_color,
            0,
            None,
            TextAlign::Center,
            crate::render_commands::TextVerticalAlign::Top,
        );
    }
}

// ---------------------------------------------------------------------------
// Toggle / Switch
// ---------------------------------------------------------------------------

/// Animated on/off toggle switch.
#[derive(Debug, Clone)]
pub struct Toggle {
    pub id: UIId,
    pub on: bool,
    pub enabled: bool,
    pub width: f32,
    pub height: f32,
    pub on_color: Color,
    pub off_color: Color,
    pub thumb_color: Color,
    pub disabled_color: Color,
    pub animation_progress: f32,
    pub toggled: bool,
}

impl Toggle {
    pub fn new(on: bool) -> Self {
        Self {
            id: UIId::INVALID,
            on,
            enabled: true,
            width: 48.0,
            height: 24.0,
            on_color: Color::from_hex("#6200EE"),
            off_color: Color::from_rgba8(120, 120, 120, 200),
            thumb_color: Color::WHITE,
            disabled_color: Color::from_rgba8(80, 80, 80, 128),
            animation_progress: if on { 1.0 } else { 0.0 },
            toggled: false,
        }
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect, dt: f32) -> bool {
        self.toggled = false;

        // Animate.
        let target = if self.on { 1.0 } else { 0.0 };
        let speed = 6.0;
        if (self.animation_progress - target).abs() > 0.001 {
            if self.animation_progress < target {
                self.animation_progress = (self.animation_progress + speed * dt).min(target);
            } else {
                self.animation_progress = (self.animation_progress - speed * dt).max(target);
            }
        }

        if !self.enabled {
            return false;
        }

        if let UIEvent::Click { position, button: MouseButton::Left, .. } = event {
            if rect.contains(*position) {
                self.on = !self.on;
                self.toggled = true;
                return true;
            }
        }
        false
    }

    pub fn layout(&self) -> Vec2 {
        Vec2::new(self.width, self.height)
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        let track_color = if !self.enabled {
            self.disabled_color
        } else {
            self.off_color.lerp(self.on_color, self.animation_progress)
        };

        let radius = rect.height() / 2.0;
        draw.draw_rounded_rect(rect, track_color, CornerRadii::all(radius), Border::default());

        // Thumb.
        let thumb_radius = radius - 3.0;
        let thumb_x_start = rect.min.x + radius;
        let thumb_x_end = rect.max.x - radius;
        let thumb_x = thumb_x_start + (thumb_x_end - thumb_x_start) * self.animation_progress;
        let thumb_center = Vec2::new(thumb_x, rect.center().y);

        let thumb_c = if self.enabled {
            self.thumb_color
        } else {
            Color::from_rgba8(160, 160, 160, 200)
        };

        draw.draw_circle(thumb_center, thumb_radius, thumb_c);
    }
}

// ---------------------------------------------------------------------------
// Badge
// ---------------------------------------------------------------------------

/// Small notification indicator badge.
#[derive(Debug, Clone)]
pub struct Badge {
    pub id: UIId,
    pub count: Option<u32>,
    pub color: Color,
    pub text_color: Color,
    pub font_size: f32,
    pub min_size: f32,
    pub visible: bool,
}

impl Badge {
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            count: None,
            color: Color::RED,
            text_color: Color::WHITE,
            font_size: 10.0,
            min_size: 8.0,
            visible: true,
        }
    }

    pub fn with_count(mut self, count: u32) -> Self {
        self.count = Some(count);
        self
    }

    pub fn render(&self, draw: &mut DrawList, anchor: Vec2) {
        if !self.visible {
            return;
        }

        if let Some(count) = self.count {
            let text = if count > 99 {
                "99+".to_string()
            } else {
                count.to_string()
            };
            let font = default_font();
            let m = measure_text(&text, &font, self.font_size, None);
            let w = (m.width + 6.0).max(self.min_size + 8.0);
            let h = self.min_size + 6.0;
            let badge_rect = Rect::new(
                Vec2::new(anchor.x - w / 2.0, anchor.y - h / 2.0),
                Vec2::new(anchor.x + w / 2.0, anchor.y + h / 2.0),
            );
            draw.draw_rounded_rect(badge_rect, self.color, CornerRadii::all(h / 2.0), Border::default());
            draw.draw_text_ex(
                &text,
                Vec2::new(anchor.x, anchor.y - self.font_size / 2.0),
                self.font_size,
                self.text_color,
                0,
                None,
                TextAlign::Center,
                crate::render_commands::TextVerticalAlign::Top,
            );
        } else {
            // Dot badge.
            draw.draw_circle(anchor, self.min_size / 2.0, self.color);
        }
    }
}

// ---------------------------------------------------------------------------
// Breadcrumbs
// ---------------------------------------------------------------------------

/// Navigation path display.
#[derive(Debug, Clone)]
pub struct Breadcrumbs {
    pub id: UIId,
    pub items: Vec<String>,
    pub separator: String,
    pub font_size: f32,
    pub text_color: Color,
    pub active_color: Color,
    pub separator_color: Color,
    pub clicked_index: Option<usize>,
}

impl Breadcrumbs {
    pub fn new(items: Vec<String>) -> Self {
        Self {
            id: UIId::INVALID,
            items,
            separator: "/".to_string(),
            font_size: 14.0,
            text_color: Color::from_hex("#6200EE"),
            active_color: Color::WHITE,
            separator_color: Color::GRAY,
            clicked_index: None,
        }
    }

    pub fn update(&mut self, event: &UIEvent, rect: Rect) -> bool {
        self.clicked_index = None;
        if let UIEvent::Click { position, button: MouseButton::Left, .. } = event {
            if rect.contains(*position) {
                // Determine which segment was clicked.
                let font = default_font();
                let mut x = rect.min.x;
                for (i, item) in self.items.iter().enumerate() {
                    let w = measure_text(item, &font, self.font_size, None).width;
                    if position.x >= x && position.x <= x + w {
                        self.clicked_index = Some(i);
                        return true;
                    }
                    x += w;
                    if i < self.items.len() - 1 {
                        let sep_w = measure_text(&self.separator, &font, self.font_size, None).width + 8.0;
                        x += sep_w;
                    }
                }
            }
        }
        false
    }

    pub fn layout(&self, font: &Font) -> Vec2 {
        let mut w = 0.0;
        for (i, item) in self.items.iter().enumerate() {
            w += measure_text(item, font, self.font_size, None).width;
            if i < self.items.len() - 1 {
                w += measure_text(&self.separator, font, self.font_size, None).width + 8.0;
            }
        }
        Vec2::new(w, self.font_size)
    }

    pub fn render(&self, draw: &mut DrawList, rect: Rect) {
        let font = default_font();
        let mut x = rect.min.x;
        let y = rect.center().y - self.font_size / 2.0;
        let last = self.items.len().saturating_sub(1);

        for (i, item) in self.items.iter().enumerate() {
            let color = if i == last {
                self.active_color
            } else {
                self.text_color
            };
            draw.draw_text(item, Vec2::new(x, y), self.font_size, color);
            x += measure_text(item, &font, self.font_size, None).width;

            if i < last {
                let sep_x = x + 4.0;
                draw.draw_text(&self.separator, Vec2::new(sep_x, y), self.font_size, self.separator_color);
                x = sep_x + measure_text(&self.separator, &font, self.font_size, None).width + 4.0;
            }
        }
    }
}
