//! Immediate-mode UI framework built on top of the GPU renderer.
//!
//! Provides a complete widget API (buttons, sliders, text inputs, dropdowns,
//! tree views, menus, tabs, etc.) with automatic layout, input handling, and
//! a premium dark theme suitable for game engine editors.
//!
//! # Usage
//!
//! ```ignore
//! // Each frame:
//! ui.begin_frame(input, screen_size);
//!
//! ui.begin_panel("Properties", rect);
//! ui.label("Transform");
//! if ui.button("Reset") { /* ... */ }
//! ui.slider_f32("Speed", &mut speed, 0.0, 100.0);
//! ui.end_panel();
//!
//! ui.finish_frame(encoder, target_view);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec2;
use genovo_core::Rect;

use crate::gpu_renderer::UIGpuRenderer;
use crate::render_commands::Color;

// ---------------------------------------------------------------------------
// WidgetId
// ---------------------------------------------------------------------------

/// Unique identifier for a widget, generated from the ID stack and label hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WidgetId(pub u64);

impl WidgetId {
    pub const NONE: Self = Self(0);

    pub fn is_none(&self) -> bool {
        self.0 == 0
    }
}

// ---------------------------------------------------------------------------
// UIInputState
// ---------------------------------------------------------------------------

/// Per-frame input state for the UI system. The application fills this in
/// from OS/windowing events before calling `begin_frame`.
#[derive(Debug, Clone)]
pub struct UIInputState {
    /// Current mouse position in screen pixels.
    pub mouse_pos: Vec2,
    /// Mouse movement delta since last frame.
    pub mouse_delta: Vec2,
    /// Whether the left mouse button is currently held.
    pub mouse_left_down: bool,
    /// Whether the left mouse button was just pressed this frame.
    pub mouse_left_pressed: bool,
    /// Whether the left mouse button was just released this frame.
    pub mouse_left_released: bool,
    /// Whether the right mouse button is currently held.
    pub mouse_right_down: bool,
    /// Whether the right mouse button was just pressed this frame.
    pub mouse_right_pressed: bool,
    /// Whether the right mouse button was just released this frame.
    pub mouse_right_released: bool,
    /// Scroll wheel delta (positive = up/forward).
    pub scroll_delta: Vec2,
    /// Characters typed this frame (for text input).
    pub text_input: Vec<char>,
    /// Keys pressed this frame (virtual key codes as u32).
    pub keys_pressed: Vec<u32>,
    /// Keys released this frame.
    pub keys_released: Vec<u32>,
    /// Keys currently held down.
    pub keys_down: Vec<u32>,
    /// Modifier key states.
    pub mod_ctrl: bool,
    pub mod_shift: bool,
    pub mod_alt: bool,
}

impl UIInputState {
    pub fn new() -> Self {
        Self {
            mouse_pos: Vec2::ZERO,
            mouse_delta: Vec2::ZERO,
            mouse_left_down: false,
            mouse_left_pressed: false,
            mouse_left_released: false,
            mouse_right_down: false,
            mouse_right_pressed: false,
            mouse_right_released: false,
            scroll_delta: Vec2::ZERO,
            text_input: Vec::new(),
            keys_pressed: Vec::new(),
            keys_released: Vec::new(),
            keys_down: Vec::new(),
            mod_ctrl: false,
            mod_shift: false,
            mod_alt: false,
        }
    }

    /// Returns true if the given key code was just pressed this frame.
    pub fn key_pressed(&self, key: u32) -> bool {
        self.keys_pressed.contains(&key)
    }

    /// Returns true if the given key code is currently held.
    pub fn key_down(&self, key: u32) -> bool {
        self.keys_down.contains(&key)
    }
}

impl Default for UIInputState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Well-known key codes (matching typical winit / OS values)
// ---------------------------------------------------------------------------

pub mod keys {
    pub const BACKSPACE: u32 = 8;
    pub const TAB: u32 = 9;
    pub const ENTER: u32 = 13;
    pub const ESCAPE: u32 = 27;
    pub const DELETE: u32 = 46;
    pub const LEFT: u32 = 37;
    pub const RIGHT: u32 = 39;
    pub const UP: u32 = 38;
    pub const DOWN: u32 = 40;
    pub const HOME: u32 = 36;
    pub const END: u32 = 35;
    pub const A: u32 = 65;
    pub const C: u32 = 67;
    pub const V: u32 = 86;
    pub const X: u32 = 88;
    pub const Z: u32 = 90;
}

// ---------------------------------------------------------------------------
// UIStyle — premium dark theme
// ---------------------------------------------------------------------------

/// Complete visual style configuration for the UI.
#[derive(Debug, Clone)]
pub struct UIStyle {
    pub bg_base: Color,
    pub bg_panel: Color,
    pub bg_widget: Color,
    pub bg_hover: Color,
    pub bg_active: Color,
    pub accent: Color,
    pub accent_dim: Color,
    pub text_bright: Color,
    pub text_normal: Color,
    pub text_dim: Color,
    pub border: Color,
    pub green: Color,
    pub yellow: Color,
    pub red: Color,
    pub font_size: f32,
    pub font_size_small: f32,
    pub font_size_heading: f32,
    pub corner_radius: f32,
    pub panel_padding: f32,
    pub item_spacing: f32,
    pub item_height: f32,
    pub indent_width: f32,
    pub scrollbar_width: f32,
    pub separator_thickness: f32,
}

impl UIStyle {
    /// Premium dark theme preset (Unreal Engine / Blender inspired).
    pub fn dark() -> Self {
        Self {
            bg_base: Color::from_rgba8(18, 18, 22, 255),
            bg_panel: Color::from_rgba8(24, 24, 28, 255),
            bg_widget: Color::from_rgba8(32, 32, 38, 255),
            bg_hover: Color::from_rgba8(42, 42, 50, 255),
            bg_active: Color::from_rgba8(52, 52, 62, 255),
            accent: Color::from_rgba8(56, 132, 244, 255),
            accent_dim: Color::from_rgba8(40, 100, 200, 255),
            text_bright: Color::from_rgba8(230, 230, 235, 255),
            text_normal: Color::from_rgba8(180, 180, 188, 255),
            text_dim: Color::from_rgba8(110, 110, 120, 255),
            border: Color::from_rgba8(38, 38, 44, 255),
            green: Color::from_rgba8(72, 199, 142, 255),
            yellow: Color::from_rgba8(245, 196, 80, 255),
            red: Color::from_rgba8(235, 87, 87, 255),
            font_size: 13.5,
            font_size_small: 11.0,
            font_size_heading: 15.0,
            corner_radius: 3.0,
            panel_padding: 8.0,
            item_spacing: 4.0,
            item_height: 22.0,
            indent_width: 16.0,
            scrollbar_width: 4.0,
            separator_thickness: 1.0,
        }
    }
}

impl Default for UIStyle {
    fn default() -> Self {
        Self::dark()
    }
}

// ---------------------------------------------------------------------------
// LayoutCursor — tracks where the next widget goes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LayoutCursor {
    /// Current position for the next widget.
    pos: Vec2,
    /// Available width for the current row/column.
    available_width: f32,
    /// Whether we are laying out horizontally (true) or vertically (false).
    horizontal: bool,
    /// Maximum height of items in the current horizontal row.
    row_height: f32,
    /// Starting X for the current layout scope.
    origin_x: f32,
    /// Starting Y for the current layout scope.
    origin_y: f32,
    /// Indent level.
    indent: u32,
}

// ---------------------------------------------------------------------------
// PanelState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PanelState {
    title: String,
    rect: Rect,
    scroll_y: f32,
    content_height: f32,
}

// ---------------------------------------------------------------------------
// WidgetState (per-widget persistent state)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct WidgetState {
    /// For text inputs: the current cursor position.
    cursor_pos: usize,
    /// For text inputs: selection start.
    selection_start: usize,
    /// For dropdowns: whether the popup is open.
    open: bool,
    /// For tree nodes: whether expanded.
    expanded: bool,
    /// Scroll offset.
    scroll: f32,
}

// ---------------------------------------------------------------------------
// UI — the main immediate-mode UI context
// ---------------------------------------------------------------------------

/// The main UI context. Holds the GPU renderer, input state, style, and
/// all per-frame and persistent widget state needed for immediate-mode
/// operation.
pub struct UI {
    /// The GPU renderer backend.
    pub renderer: UIGpuRenderer,
    /// Current frame input state.
    input: UIInputState,
    /// Visual style/theme.
    pub style: UIStyle,

    /// Widget currently under the mouse cursor.
    hot: WidgetId,
    /// Widget currently being interacted with (pressed/dragging).
    active: WidgetId,
    /// Widget with keyboard focus.
    focused: WidgetId,

    /// ID stack for hierarchical widget ID generation.
    id_stack: Vec<u64>,
    /// Next auto-increment ID within the current scope.
    auto_id: u64,

    /// Layout cursor stack.
    layout_stack: Vec<LayoutCursor>,

    /// Panel state stack.
    panel_stack: Vec<PanelState>,

    /// Persistent per-widget state keyed by WidgetId.
    widget_states: HashMap<u64, WidgetState>,

    /// Tooltip text to show at end of frame, if any.
    pending_tooltip: Option<(Vec2, String)>,

    /// Menu state: currently open menu label.
    open_menu: Option<String>,
    /// Menu bar Y position.
    menu_bar_y: f32,

    /// Screen size.
    screen_size: Vec2,

    /// Delta time this frame.
    delta_time: f32,
    /// Total elapsed time.
    total_time: f32,
}

impl UI {
    /// Create a new UI context with the given GPU renderer.
    pub fn new(renderer: UIGpuRenderer) -> Self {
        Self {
            renderer,
            input: UIInputState::new(),
            style: UIStyle::dark(),
            hot: WidgetId::NONE,
            active: WidgetId::NONE,
            focused: WidgetId::NONE,
            id_stack: vec![0],
            auto_id: 0,
            layout_stack: Vec::with_capacity(16),
            panel_stack: Vec::new(),
            widget_states: HashMap::new(),
            pending_tooltip: None,
            open_menu: None,
            menu_bar_y: 0.0,
            screen_size: Vec2::new(1920.0, 1080.0),
            delta_time: 1.0 / 60.0,
            total_time: 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // Frame lifecycle
    // -----------------------------------------------------------------------

    /// Start a new UI frame. Call this before issuing any widget commands.
    pub fn begin_frame(&mut self, input: UIInputState, screen_size: Vec2, delta_time: f32) {
        self.input = input;
        self.screen_size = screen_size;
        self.delta_time = delta_time;
        self.total_time += delta_time;
        self.auto_id = 0;
        self.id_stack.clear();
        self.id_stack.push(0);
        self.layout_stack.clear();
        self.pending_tooltip = None;

        // Reset hot widget each frame; it will be re-set during widget processing.
        self.hot = WidgetId::NONE;

        // Start the GPU renderer frame.
        self.renderer.begin_frame(screen_size);

        // Push a default full-screen layout cursor.
        self.layout_stack.push(LayoutCursor {
            pos: Vec2::ZERO,
            available_width: screen_size.x,
            horizontal: false,
            row_height: 0.0,
            origin_x: 0.0,
            origin_y: 0.0,
            indent: 0,
        });
    }

    /// Finish the UI frame. Renders tooltip if pending, then submits GPU work.
    pub fn finish_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
    ) {
        // Draw tooltip if pending.
        if let Some((pos, text)) = self.pending_tooltip.take() {
            self.draw_tooltip_at(pos, &text);
        }

        self.renderer.end_frame(encoder, target_view);
    }

    // -----------------------------------------------------------------------
    // ID generation
    // -----------------------------------------------------------------------

    fn gen_id(&mut self) -> WidgetId {
        self.auto_id += 1;
        let parent = self.id_stack.last().copied().unwrap_or(0);
        WidgetId(hash_combine(parent, self.auto_id))
    }

    fn gen_id_from_str(&mut self, label: &str) -> WidgetId {
        let parent = self.id_stack.last().copied().unwrap_or(0);
        WidgetId(hash_combine(parent, hash_str(label)))
    }

    fn push_id(&mut self, id: u64) {
        self.id_stack.push(id);
    }

    fn pop_id(&mut self) {
        if self.id_stack.len() > 1 {
            self.id_stack.pop();
        }
    }

    // -----------------------------------------------------------------------
    // Layout helpers
    // -----------------------------------------------------------------------

    fn cursor(&self) -> &LayoutCursor {
        self.layout_stack.last().expect("layout stack empty")
    }

    fn cursor_mut(&mut self) -> &mut LayoutCursor {
        self.layout_stack.last_mut().expect("layout stack empty")
    }

    /// Allocate space for a widget of the given size. Returns the rect.
    fn allocate_widget(&mut self, width: f32, height: f32) -> Rect {
        let spacing = self.style.item_spacing;
        let cursor = self.cursor_mut();
        let indent_offset = cursor.indent as f32 * self.style.indent_width;

        let rect = if cursor.horizontal {
            let r = Rect::new(
                cursor.pos,
                Vec2::new(cursor.pos.x + width, cursor.pos.y + height),
            );
            cursor.pos.x += width + spacing;
            cursor.row_height = cursor.row_height.max(height);
            r
        } else {
            let x = cursor.pos.x + indent_offset;
            let r = Rect::new(
                Vec2::new(x, cursor.pos.y),
                Vec2::new(x + width, cursor.pos.y + height),
            );
            cursor.pos.y += height + spacing;
            r
        };

        rect
    }

    /// Get available width for the current layout scope (minus indent).
    fn available_width(&self) -> f32 {
        let cursor = self.cursor();
        let indent_offset = cursor.indent as f32 * self.style.indent_width;
        (cursor.available_width - indent_offset).max(0.0)
    }

    // -----------------------------------------------------------------------
    // Interaction helpers
    // -----------------------------------------------------------------------

    fn is_mouse_over(&self, rect: &Rect) -> bool {
        rect.contains(self.input.mouse_pos)
    }

    /// Returns true if this widget is being hovered.
    pub fn is_hovered(&self, id: WidgetId) -> bool {
        self.hot == id
    }

    /// Returns true if this widget is being actively interacted with.
    pub fn is_active(&self, id: WidgetId) -> bool {
        self.active == id
    }

    /// Returns true if this widget has keyboard focus.
    pub fn is_focused(&self, id: WidgetId) -> bool {
        self.focused == id
    }

    fn update_interaction(&mut self, id: WidgetId, rect: &Rect) -> (bool, bool, bool) {
        let hovered = self.is_mouse_over(rect);
        let pressed;
        let released;

        if hovered {
            self.hot = id;
        }

        if hovered && self.input.mouse_left_pressed {
            self.active = id;
            self.focused = id;
            pressed = true;
            released = false;
        } else if self.active == id && self.input.mouse_left_released {
            pressed = false;
            released = hovered; // Only "click" if released while still over
            self.active = WidgetId::NONE;
        } else {
            pressed = false;
            released = false;
        }

        (hovered, pressed, released)
    }

    fn get_widget_state(&mut self, id: WidgetId) -> &mut WidgetState {
        self.widget_states.entry(id.0).or_insert_with(WidgetState::default)
    }

    // -----------------------------------------------------------------------
    // Panels
    // -----------------------------------------------------------------------

    /// Begin a panel (window-like container). Returns true if the panel is
    /// visible. Must be paired with `end_panel()`.
    pub fn begin_panel(&mut self, title: &str, rect: Rect) -> bool {
        let id = self.gen_id_from_str(title);
        self.push_id(id.0);

        // Draw panel background.
        self.renderer.draw_rect(rect, self.style.bg_panel, self.style.corner_radius);

        // Draw panel border.
        self.renderer.draw_rect_outline(
            rect,
            self.style.border,
            1.0,
            self.style.corner_radius,
        );

        // Draw title bar.
        let title_height = self.style.font_size_heading + self.style.panel_padding * 2.0;
        let title_rect = Rect::new(
            rect.min,
            Vec2::new(rect.max.x, rect.min.y + title_height),
        );
        self.renderer.draw_rect(
            title_rect,
            self.style.bg_base,
            self.style.corner_radius,
        );

        // Draw title text.
        self.renderer.draw_text(
            title,
            Vec2::new(
                rect.min.x + self.style.panel_padding,
                rect.min.y + self.style.panel_padding,
            ),
            self.style.font_size_heading,
            self.style.text_bright,
        );

        // Draw separator line.
        self.renderer.draw_line(
            Vec2::new(rect.min.x, rect.min.y + title_height),
            Vec2::new(rect.max.x, rect.min.y + title_height),
            self.style.border,
            1.0,
        );

        // Set up clipping and layout cursor for panel content.
        let content_rect = Rect::new(
            Vec2::new(
                rect.min.x + self.style.panel_padding,
                rect.min.y + title_height + self.style.panel_padding,
            ),
            Vec2::new(
                rect.max.x - self.style.panel_padding,
                rect.max.y - self.style.panel_padding,
            ),
        );

        self.renderer.push_clip(content_rect);

        self.panel_stack.push(PanelState {
            title: title.to_string(),
            rect,
            scroll_y: 0.0,
            content_height: 0.0,
        });

        self.layout_stack.push(LayoutCursor {
            pos: content_rect.min,
            available_width: content_rect.width(),
            horizontal: false,
            row_height: 0.0,
            origin_x: content_rect.min.x,
            origin_y: content_rect.min.y,
            indent: 0,
        });

        true
    }

    /// End the current panel.
    pub fn end_panel(&mut self) {
        self.layout_stack.pop();
        self.panel_stack.pop();
        self.renderer.pop_clip();
        self.pop_id();
    }

    // -----------------------------------------------------------------------
    // Layout control
    // -----------------------------------------------------------------------

    /// Begin a horizontal layout group. Widgets inside will be placed
    /// left-to-right.
    pub fn horizontal<F: FnOnce(&mut UI)>(&mut self, f: F) {
        let cursor = self.cursor().clone();
        self.layout_stack.push(LayoutCursor {
            pos: cursor.pos,
            available_width: cursor.available_width,
            horizontal: true,
            row_height: 0.0,
            origin_x: cursor.pos.x,
            origin_y: cursor.pos.y,
            indent: cursor.indent,
        });

        f(self);

        let finished = self.layout_stack.pop().unwrap();
        let parent = self.cursor_mut();
        parent.pos.y = finished.pos.y + finished.row_height + self.style.item_spacing;
    }

    /// Begin a vertical layout group.
    pub fn vertical<F: FnOnce(&mut UI)>(&mut self, f: F) {
        let cursor = self.cursor().clone();
        self.layout_stack.push(LayoutCursor {
            pos: cursor.pos,
            available_width: cursor.available_width,
            horizontal: false,
            row_height: 0.0,
            origin_x: cursor.pos.x,
            origin_y: cursor.pos.y,
            indent: cursor.indent,
        });

        f(self);

        let finished = self.layout_stack.pop().unwrap();
        let parent = self.cursor_mut();
        parent.pos.y = finished.pos.y;
    }

    /// Insert empty space.
    pub fn space(&mut self, pixels: f32) {
        let cursor = self.cursor_mut();
        if cursor.horizontal {
            cursor.pos.x += pixels;
        } else {
            cursor.pos.y += pixels;
        }
    }

    /// Draw a horizontal separator line.
    pub fn separator(&mut self) {
        let spacing = self.style.item_spacing;
        let y = self.cursor().pos.y + spacing;
        let x0 = self.cursor().origin_x;
        let x1 = x0 + self.cursor().available_width;

        self.renderer.draw_line(
            Vec2::new(x0, y),
            Vec2::new(x1, y),
            self.style.border,
            self.style.separator_thickness,
        );

        self.cursor_mut().pos.y = y + spacing + self.style.separator_thickness;
    }

    /// Indent the layout by one level.
    pub fn indent(&mut self) {
        self.cursor_mut().indent += 1;
    }

    /// Unindent the layout by one level.
    pub fn unindent(&mut self) {
        let cursor = self.cursor_mut();
        if cursor.indent > 0 {
            cursor.indent -= 1;
        }
    }

    // -----------------------------------------------------------------------
    // Widgets: Label
    // -----------------------------------------------------------------------

    /// Draw a text label.
    pub fn label(&mut self, text: &str) {
        let (tw, _th) = self.renderer.measure_text(text, self.style.font_size);
        let h = self.style.item_height;
        let w = tw.max(10.0);
        let rect = self.allocate_widget(w, h);

        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            text,
            Vec2::new(rect.min.x, text_y),
            self.style.font_size,
            self.style.text_normal,
        );
    }

    /// Draw a text label with a specific color.
    pub fn label_colored(&mut self, text: &str, color: Color) {
        let (tw, _th) = self.renderer.measure_text(text, self.style.font_size);
        let h = self.style.item_height;
        let w = tw.max(10.0);
        let rect = self.allocate_widget(w, h);

        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            text,
            Vec2::new(rect.min.x, text_y),
            self.style.font_size,
            color,
        );
    }

    /// Draw a heading label (larger font).
    pub fn heading(&mut self, text: &str) {
        let (tw, _th) = self.renderer.measure_text(text, self.style.font_size_heading);
        let h = self.style.font_size_heading + self.style.item_spacing * 2.0;
        let w = tw.max(10.0);
        let rect = self.allocate_widget(w, h);

        let text_y = rect.min.y + (h - self.style.font_size_heading) * 0.5;
        self.renderer.draw_text(
            text,
            Vec2::new(rect.min.x, text_y),
            self.style.font_size_heading,
            self.style.text_bright,
        );
    }

    // -----------------------------------------------------------------------
    // Widgets: Button
    // -----------------------------------------------------------------------

    /// Draw a clickable button. Returns true if clicked this frame.
    pub fn button(&mut self, text: &str) -> bool {
        let id = self.gen_id_from_str(text);
        let (tw, _) = self.renderer.measure_text(text, self.style.font_size);
        let padding = self.style.panel_padding;
        let w = tw + padding * 2.0;
        let h = self.style.item_height;
        let rect = self.allocate_widget(w, h);

        let (hovered, _pressed, released) = self.update_interaction(id, &rect);

        // Background color based on state.
        let bg = if self.active == id {
            self.style.accent
        } else if hovered {
            self.style.bg_hover
        } else {
            self.style.bg_widget
        };

        self.renderer.draw_rect(rect, bg, self.style.corner_radius);

        // Text centered in button.
        let text_x = rect.min.x + (rect.width() - tw) * 0.5;
        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            text,
            Vec2::new(text_x, text_y),
            self.style.font_size,
            self.style.text_bright,
        );

        released
    }

    /// Draw a small icon-style button with a tooltip.
    pub fn icon_button(&mut self, icon: &str, tooltip: &str) -> bool {
        let id = self.gen_id_from_str(icon);
        let h = self.style.item_height;
        let w = h; // Square button
        let rect = self.allocate_widget(w, h);

        let (hovered, _pressed, released) = self.update_interaction(id, &rect);

        let bg = if self.active == id {
            self.style.accent
        } else if hovered {
            self.style.bg_hover
        } else {
            self.style.bg_widget
        };

        self.renderer.draw_rect(rect, bg, self.style.corner_radius);

        // Draw icon character centered.
        let (iw, _) = self.renderer.measure_text(icon, self.style.font_size);
        let text_x = rect.min.x + (w - iw) * 0.5;
        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            icon,
            Vec2::new(text_x, text_y),
            self.style.font_size,
            self.style.text_normal,
        );

        if hovered && !tooltip.is_empty() {
            self.pending_tooltip = Some((
                Vec2::new(rect.min.x, rect.max.y + 4.0),
                tooltip.to_string(),
            ));
        }

        released
    }

    // -----------------------------------------------------------------------
    // Widgets: Checkbox
    // -----------------------------------------------------------------------

    /// Draw a checkbox with a label. Returns true if the value changed.
    pub fn checkbox(&mut self, label: &str, value: &mut bool) -> bool {
        let id = self.gen_id_from_str(label);
        let h = self.style.item_height;
        let box_size = h - 4.0;
        let (tw, _) = self.renderer.measure_text(label, self.style.font_size);
        let w = box_size + 6.0 + tw;
        let rect = self.allocate_widget(w, h);

        let box_rect = Rect::new(
            Vec2::new(rect.min.x, rect.min.y + 2.0),
            Vec2::new(rect.min.x + box_size, rect.min.y + 2.0 + box_size),
        );

        let (hovered, _pressed, released) = self.update_interaction(id, &rect);

        let mut changed = false;
        if released {
            *value = !*value;
            changed = true;
        }

        // Draw checkbox box.
        let bg = if *value {
            self.style.accent
        } else if hovered {
            self.style.bg_hover
        } else {
            self.style.bg_widget
        };
        self.renderer.draw_rect(box_rect, bg, 2.0);

        if !*value {
            self.renderer.draw_rect_outline(box_rect, self.style.border, 1.0, 2.0);
        }

        // Draw checkmark if checked.
        if *value {
            let cx = box_rect.center().x;
            let cy = box_rect.center().y;
            let s = box_size * 0.3;
            // Simple checkmark as two lines.
            self.renderer.draw_line(
                Vec2::new(cx - s, cy),
                Vec2::new(cx - s * 0.3, cy + s * 0.7),
                self.style.text_bright,
                2.0,
            );
            self.renderer.draw_line(
                Vec2::new(cx - s * 0.3, cy + s * 0.7),
                Vec2::new(cx + s, cy - s * 0.5),
                self.style.text_bright,
                2.0,
            );
        }

        // Label text.
        let text_x = rect.min.x + box_size + 6.0;
        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            label,
            Vec2::new(text_x, text_y),
            self.style.font_size,
            self.style.text_normal,
        );

        changed
    }

    // -----------------------------------------------------------------------
    // Widgets: Slider
    // -----------------------------------------------------------------------

    /// Draw a float slider. Returns true if the value changed.
    pub fn slider_f32(
        &mut self,
        label: &str,
        value: &mut f32,
        min: f32,
        max: f32,
    ) -> bool {
        let id = self.gen_id_from_str(label);
        let h = self.style.item_height;
        let available = self.available_width();
        let label_width = available * 0.35;
        let slider_width = available - label_width - self.style.item_spacing;
        let w = available;
        let rect = self.allocate_widget(w, h);

        // Draw label.
        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            label,
            Vec2::new(rect.min.x, text_y),
            self.style.font_size,
            self.style.text_dim,
        );

        // Slider track.
        let track_x = rect.min.x + label_width + self.style.item_spacing;
        let track_rect = Rect::new(
            Vec2::new(track_x, rect.min.y + h * 0.35),
            Vec2::new(track_x + slider_width, rect.min.y + h * 0.65),
        );

        let (hovered, _pressed, _released) = self.update_interaction(id, &Rect::new(
            Vec2::new(track_x, rect.min.y),
            Vec2::new(track_x + slider_width, rect.max.y),
        ));

        let mut changed = false;

        // Handle dragging.
        if self.active == id && self.input.mouse_left_down {
            let t = ((self.input.mouse_pos.x - track_x) / slider_width).clamp(0.0, 1.0);
            let new_val = min + t * (max - min);
            if (*value - new_val).abs() > f32::EPSILON {
                *value = new_val;
                changed = true;
            }
        }

        // Draw track background.
        self.renderer.draw_rect(track_rect, self.style.bg_widget, 2.0);

        // Draw filled portion.
        let t = ((*value - min) / (max - min)).clamp(0.0, 1.0);
        let fill_rect = Rect::new(
            track_rect.min,
            Vec2::new(track_rect.min.x + track_rect.width() * t, track_rect.max.y),
        );
        self.renderer.draw_rect(fill_rect, self.style.accent_dim, 2.0);

        // Draw thumb.
        let thumb_x = track_x + slider_width * t;
        let thumb_radius = 5.0;
        let thumb_color = if self.active == id {
            self.style.accent
        } else if hovered {
            self.style.text_bright
        } else {
            self.style.text_normal
        };
        self.renderer.draw_circle(
            Vec2::new(thumb_x, rect.min.y + h * 0.5),
            thumb_radius,
            thumb_color,
        );

        // Draw value text.
        let val_text = format!("{:.2}", value);
        let (vw, _) = self.renderer.measure_text(&val_text, self.style.font_size_small);
        self.renderer.draw_text(
            &val_text,
            Vec2::new(track_x + slider_width - vw, text_y),
            self.style.font_size_small,
            self.style.text_dim,
        );

        changed
    }

    // -----------------------------------------------------------------------
    // Widgets: Drag value
    // -----------------------------------------------------------------------

    /// Draw a drag-to-edit value. Click and drag horizontally to change the
    /// value. Returns true if the value changed.
    pub fn drag_value(&mut self, label: &str, value: &mut f32, speed: f32) -> bool {
        let id = self.gen_id_from_str(label);
        let h = self.style.item_height;
        let available = self.available_width();
        let label_width = available * 0.35;
        let field_width = available - label_width - self.style.item_spacing;
        let w = available;
        let rect = self.allocate_widget(w, h);

        // Label.
        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            label,
            Vec2::new(rect.min.x, text_y),
            self.style.font_size,
            self.style.text_dim,
        );

        // Value field.
        let field_x = rect.min.x + label_width + self.style.item_spacing;
        let field_rect = Rect::new(
            Vec2::new(field_x, rect.min.y),
            Vec2::new(field_x + field_width, rect.max.y),
        );

        let (hovered, _pressed, _released) = self.update_interaction(id, &field_rect);

        let mut changed = false;

        if self.active == id && self.input.mouse_left_down {
            *value += self.input.mouse_delta.x * speed;
            if self.input.mouse_delta.x.abs() > 0.001 {
                changed = true;
            }
        }

        let bg = if self.active == id {
            self.style.bg_active
        } else if hovered {
            self.style.bg_hover
        } else {
            self.style.bg_widget
        };

        self.renderer.draw_rect(field_rect, bg, self.style.corner_radius);

        let val_text = format!("{:.3}", value);
        let (vw, _) = self.renderer.measure_text(&val_text, self.style.font_size);
        let vx = field_x + (field_width - vw) * 0.5;
        self.renderer.draw_text(
            &val_text,
            Vec2::new(vx, text_y),
            self.style.font_size,
            self.style.text_normal,
        );

        changed
    }

    // -----------------------------------------------------------------------
    // Widgets: Text input
    // -----------------------------------------------------------------------

    /// Draw a single-line text input. Returns true if the text changed.
    pub fn text_input(&mut self, label: &str, text: &mut String) -> bool {
        let id = self.gen_id_from_str(label);
        let h = self.style.item_height;
        let available = self.available_width();
        let label_width = available * 0.35;
        let field_width = available - label_width - self.style.item_spacing;
        let w = available;
        let rect = self.allocate_widget(w, h);

        // Label.
        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            label,
            Vec2::new(rect.min.x, text_y),
            self.style.font_size,
            self.style.text_dim,
        );

        // Field.
        let field_x = rect.min.x + label_width + self.style.item_spacing;
        let field_rect = Rect::new(
            Vec2::new(field_x, rect.min.y),
            Vec2::new(field_x + field_width, rect.max.y),
        );

        let (_hovered, _pressed, _released) = self.update_interaction(id, &field_rect);
        let focused = self.focused == id;

        // Draw field background.
        let bg = if focused {
            self.style.bg_active
        } else {
            self.style.bg_widget
        };
        self.renderer.draw_rect(field_rect, bg, self.style.corner_radius);

        if focused {
            self.renderer.draw_rect_outline(
                field_rect,
                self.style.accent,
                1.0,
                self.style.corner_radius,
            );
        }

        let mut changed = false;

        // Handle text input when focused.
        if focused {
            let state = self.widget_states.entry(id.0).or_insert_with(|| {
                let mut s = WidgetState::default();
                s.cursor_pos = text.len();
                s
            });

            // Process typed characters.
            for &ch in &self.input.text_input {
                if ch >= ' ' && ch != '\x7f' {
                    if state.cursor_pos > text.len() {
                        state.cursor_pos = text.len();
                    }
                    text.insert(state.cursor_pos, ch);
                    state.cursor_pos += 1;
                    changed = true;
                }
            }

            // Backspace.
            if self.input.key_pressed(keys::BACKSPACE) && state.cursor_pos > 0 {
                state.cursor_pos -= 1;
                if state.cursor_pos < text.len() {
                    text.remove(state.cursor_pos);
                    changed = true;
                }
            }

            // Delete.
            if self.input.key_pressed(keys::DELETE) && state.cursor_pos < text.len() {
                text.remove(state.cursor_pos);
                changed = true;
            }

            // Arrow keys.
            if self.input.key_pressed(keys::LEFT) && state.cursor_pos > 0 {
                state.cursor_pos -= 1;
            }
            if self.input.key_pressed(keys::RIGHT) && state.cursor_pos < text.len() {
                state.cursor_pos += 1;
            }
            if self.input.key_pressed(keys::HOME) {
                state.cursor_pos = 0;
            }
            if self.input.key_pressed(keys::END) {
                state.cursor_pos = text.len();
            }

            // Draw cursor.
            let cursor_text = &text[..state.cursor_pos.min(text.len())];
            let (cx, _) = self.renderer.measure_text(cursor_text, self.style.font_size);
            let cursor_x = field_x + 4.0 + cx;
            let blink = ((self.total_time * 2.0) as u32) % 2 == 0;
            if blink {
                self.renderer.draw_line(
                    Vec2::new(cursor_x, rect.min.y + 3.0),
                    Vec2::new(cursor_x, rect.max.y - 3.0),
                    self.style.text_bright,
                    1.0,
                );
            }
        }

        // Clip text drawing.
        self.renderer.push_clip(field_rect);
        self.renderer.draw_text(
            text,
            Vec2::new(field_x + 4.0, text_y),
            self.style.font_size,
            self.style.text_normal,
        );
        self.renderer.pop_clip();

        changed
    }

    // -----------------------------------------------------------------------
    // Widgets: Color editor
    // -----------------------------------------------------------------------

    /// Draw a color edit widget (four sliders + preview swatch).
    /// Returns true if the color changed.
    pub fn color_edit(&mut self, label: &str, color: &mut [f32; 4]) -> bool {
        let id = self.gen_id_from_str(label);
        self.push_id(id.0);

        let h = self.style.item_height;
        let available = self.available_width();
        let rect = self.allocate_widget(available, h);

        // Label.
        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        let label_width = available * 0.35;
        self.renderer.draw_text(
            label,
            Vec2::new(rect.min.x, text_y),
            self.style.font_size,
            self.style.text_dim,
        );

        // Color swatch preview.
        let swatch_size = h - 4.0;
        let swatch_x = rect.min.x + label_width + self.style.item_spacing;
        let swatch_rect = Rect::new(
            Vec2::new(swatch_x, rect.min.y + 2.0),
            Vec2::new(swatch_x + swatch_size, rect.min.y + 2.0 + swatch_size),
        );
        let preview_color = Color::new(color[0], color[1], color[2], color[3]);
        self.renderer.draw_rect(swatch_rect, preview_color, 2.0);
        self.renderer.draw_rect_outline(swatch_rect, self.style.border, 1.0, 2.0);

        // RGBA text display.
        let rgba_text = format!(
            "({:.0}, {:.0}, {:.0}, {:.0})",
            color[0] * 255.0, color[1] * 255.0, color[2] * 255.0, color[3] * 255.0
        );
        let text_x = swatch_x + swatch_size + 6.0;
        self.renderer.draw_text(
            &rgba_text,
            Vec2::new(text_x, text_y),
            self.style.font_size_small,
            self.style.text_dim,
        );

        // Inline sliders for R, G, B, A.
        let mut changed = false;
        let channel_names = ["R", "G", "B", "A"];
        let channel_colors = [self.style.red, self.style.green, self.style.accent, self.style.text_normal];
        for i in 0..4 {
            let ch_id = self.gen_id();
            let ch_rect = self.allocate_widget(available, self.style.item_height * 0.8);
            let track_rect = Rect::new(
                Vec2::new(ch_rect.min.x + 20.0, ch_rect.min.y + 2.0),
                Vec2::new(ch_rect.max.x, ch_rect.max.y - 2.0),
            );

            let (_h, _p, _r) = self.update_interaction(ch_id, &track_rect);

            if self.active == ch_id && self.input.mouse_left_down {
                let t = ((self.input.mouse_pos.x - track_rect.min.x) / track_rect.width()).clamp(0.0, 1.0);
                if (color[i] - t).abs() > 0.001 {
                    color[i] = t;
                    changed = true;
                }
            }

            // Draw channel label.
            self.renderer.draw_text(
                channel_names[i],
                Vec2::new(ch_rect.min.x, ch_rect.min.y + 1.0),
                self.style.font_size_small,
                channel_colors[i],
            );

            // Draw track.
            self.renderer.draw_rect(track_rect, self.style.bg_widget, 2.0);

            // Draw fill.
            let t = color[i].clamp(0.0, 1.0);
            let fill_rect = Rect::new(
                track_rect.min,
                Vec2::new(track_rect.min.x + track_rect.width() * t, track_rect.max.y),
            );
            self.renderer.draw_rect(fill_rect, channel_colors[i].with_alpha(0.6), 2.0);
        }

        self.pop_id();
        changed
    }

    // -----------------------------------------------------------------------
    // Widgets: Dropdown
    // -----------------------------------------------------------------------

    /// Draw a dropdown selector. Returns true if the selection changed.
    pub fn dropdown(
        &mut self,
        label: &str,
        selected: &mut usize,
        options: &[&str],
    ) -> bool {
        let id = self.gen_id_from_str(label);
        let h = self.style.item_height;
        let available = self.available_width();
        let label_width = available * 0.35;
        let field_width = available - label_width - self.style.item_spacing;
        let w = available;
        let rect = self.allocate_widget(w, h);

        // Label.
        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            label,
            Vec2::new(rect.min.x, text_y),
            self.style.font_size,
            self.style.text_dim,
        );

        // Dropdown button.
        let field_x = rect.min.x + label_width + self.style.item_spacing;
        let field_rect = Rect::new(
            Vec2::new(field_x, rect.min.y),
            Vec2::new(field_x + field_width, rect.max.y),
        );

        let (hovered, _pressed, released) = self.update_interaction(id, &field_rect);
        let state = self.widget_states.entry(id.0).or_insert_with(WidgetState::default);
        let is_open = state.open;

        if released {
            state.open = !state.open;
        }

        let bg = if is_open || hovered {
            self.style.bg_hover
        } else {
            self.style.bg_widget
        };
        self.renderer.draw_rect(field_rect, bg, self.style.corner_radius);

        // Current selection text.
        let selected_text = if *selected < options.len() {
            options[*selected]
        } else {
            ""
        };
        self.renderer.draw_text(
            selected_text,
            Vec2::new(field_x + 4.0, text_y),
            self.style.font_size,
            self.style.text_normal,
        );

        // Dropdown arrow indicator.
        let arrow = if is_open { "^" } else { "v" };
        let (aw, _) = self.renderer.measure_text(arrow, self.style.font_size_small);
        self.renderer.draw_text(
            arrow,
            Vec2::new(field_rect.max.x - aw - 4.0, text_y),
            self.style.font_size_small,
            self.style.text_dim,
        );

        let mut changed = false;

        // Draw popup if open.
        if is_open {
            let popup_h = options.len() as f32 * h;
            let popup_rect = Rect::new(
                Vec2::new(field_x, rect.max.y + 2.0),
                Vec2::new(field_x + field_width, rect.max.y + 2.0 + popup_h),
            );

            // Popup background.
            self.renderer.draw_rect(popup_rect, self.style.bg_panel, self.style.corner_radius);
            self.renderer.draw_rect_outline(popup_rect, self.style.border, 1.0, self.style.corner_radius);

            for (i, option) in options.iter().enumerate() {
                let item_rect = Rect::new(
                    Vec2::new(field_x, rect.max.y + 2.0 + i as f32 * h),
                    Vec2::new(field_x + field_width, rect.max.y + 2.0 + (i + 1) as f32 * h),
                );

                let item_hovered = self.is_mouse_over(&item_rect);
                if item_hovered {
                    self.renderer.draw_rect(item_rect, self.style.bg_hover, 0.0);
                }

                let item_text_y = item_rect.min.y + (h - self.style.font_size) * 0.5;
                let text_color = if i == *selected {
                    self.style.accent
                } else {
                    self.style.text_normal
                };
                self.renderer.draw_text(
                    option,
                    Vec2::new(field_x + 4.0, item_text_y),
                    self.style.font_size,
                    text_color,
                );

                if item_hovered && self.input.mouse_left_pressed {
                    *selected = i;
                    changed = true;
                    // Close the dropdown.
                    if let Some(s) = self.widget_states.get_mut(&id.0) {
                        s.open = false;
                    }
                }
            }

            // Close on click outside.
            if self.input.mouse_left_pressed && !self.is_mouse_over(&popup_rect) && !self.is_mouse_over(&field_rect) {
                if let Some(s) = self.widget_states.get_mut(&id.0) {
                    s.open = false;
                }
            }
        }

        changed
    }

    // -----------------------------------------------------------------------
    // Widgets: Tree node
    // -----------------------------------------------------------------------

    /// Draw a collapsible tree node. Returns true if the node is open (so the
    /// caller can draw children).
    pub fn tree_node(&mut self, label: &str, open: &mut bool) -> bool {
        let id = self.gen_id_from_str(label);
        let h = self.style.item_height;
        let available = self.available_width();
        let rect = self.allocate_widget(available, h);

        let (hovered, _pressed, released) = self.update_interaction(id, &rect);

        if released {
            *open = !*open;
        }

        // Background on hover.
        if hovered {
            self.renderer.draw_rect(rect, self.style.bg_hover, 0.0);
        }

        // Expand/collapse arrow.
        let arrow = if *open { "v" } else { ">" };
        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            arrow,
            Vec2::new(rect.min.x, text_y),
            self.style.font_size_small,
            self.style.text_dim,
        );

        // Label.
        self.renderer.draw_text(
            label,
            Vec2::new(rect.min.x + 14.0, text_y),
            self.style.font_size,
            self.style.text_normal,
        );

        if *open {
            self.indent();
        }

        *open
    }

    /// End a tree node scope (call after tree_node returns true and you have
    /// drawn children).
    pub fn tree_node_end(&mut self) {
        self.unindent();
    }

    // -----------------------------------------------------------------------
    // Widgets: Selectable
    // -----------------------------------------------------------------------

    /// Draw a selectable list item. Returns true if clicked.
    pub fn selectable(&mut self, label: &str, selected: bool) -> bool {
        let id = self.gen_id_from_str(label);
        let h = self.style.item_height;
        let available = self.available_width();
        let rect = self.allocate_widget(available, h);

        let (hovered, _pressed, released) = self.update_interaction(id, &rect);

        // Background.
        if selected {
            self.renderer.draw_rect(rect, self.style.accent_dim.with_alpha(0.3), 0.0);
        } else if hovered {
            self.renderer.draw_rect(rect, self.style.bg_hover, 0.0);
        }

        // Text.
        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        let text_color = if selected {
            self.style.text_bright
        } else {
            self.style.text_normal
        };
        self.renderer.draw_text(
            label,
            Vec2::new(rect.min.x + 4.0, text_y),
            self.style.font_size,
            text_color,
        );

        released
    }

    // -----------------------------------------------------------------------
    // Widgets: Progress bar
    // -----------------------------------------------------------------------

    /// Draw a progress bar (0.0 to 1.0 fraction).
    pub fn progress_bar(&mut self, fraction: f32, text: Option<&str>) {
        let h = self.style.item_height;
        let available = self.available_width();
        let rect = self.allocate_widget(available, h);

        // Track.
        self.renderer.draw_rect(rect, self.style.bg_widget, self.style.corner_radius);

        // Fill.
        let fill_w = rect.width() * fraction.clamp(0.0, 1.0);
        let fill_rect = Rect::new(
            rect.min,
            Vec2::new(rect.min.x + fill_w, rect.max.y),
        );
        self.renderer.draw_rect(fill_rect, self.style.accent, self.style.corner_radius);

        // Text overlay.
        let display_text = text.map(|t| t.to_string())
            .unwrap_or_else(|| format!("{:.0}%", fraction * 100.0));
        let (tw, _) = self.renderer.measure_text(&display_text, self.style.font_size_small);
        let text_x = rect.min.x + (rect.width() - tw) * 0.5;
        let text_y = rect.min.y + (h - self.style.font_size_small) * 0.5;
        self.renderer.draw_text(
            &display_text,
            Vec2::new(text_x, text_y),
            self.style.font_size_small,
            self.style.text_bright,
        );
    }

    // -----------------------------------------------------------------------
    // Widgets: Tooltip
    // -----------------------------------------------------------------------

    /// Schedule a tooltip to be drawn at the end of the frame.
    pub fn tooltip(&mut self, text: &str) {
        self.pending_tooltip = Some((
            self.input.mouse_pos + Vec2::new(12.0, 12.0),
            text.to_string(),
        ));
    }

    fn draw_tooltip_at(&mut self, pos: Vec2, text: &str) {
        let padding = 6.0;
        let (tw, _) = self.renderer.measure_text(text, self.style.font_size_small);
        let th = self.style.font_size_small;
        let rect = Rect::new(
            pos,
            Vec2::new(pos.x + tw + padding * 2.0, pos.y + th + padding * 2.0),
        );

        // Shadow.
        self.renderer.draw_rect_shadow(
            rect,
            Color::new(0.0, 0.0, 0.0, 0.4),
            6.0,
            Vec2::new(2.0, 2.0),
        );

        // Background.
        self.renderer.draw_rect(rect, self.style.bg_panel, 4.0);
        self.renderer.draw_rect_outline(rect, self.style.border, 1.0, 4.0);

        // Text.
        self.renderer.draw_text(
            text,
            Vec2::new(pos.x + padding, pos.y + padding),
            self.style.font_size_small,
            self.style.text_normal,
        );
    }

    // -----------------------------------------------------------------------
    // Widgets: Menu bar
    // -----------------------------------------------------------------------

    /// Draw a menu bar. The closure receives the UI and can call `menu()`
    /// to add menus.
    pub fn menu_bar<F: FnOnce(&mut UI)>(&mut self, f: F) {
        let h = self.style.item_height + 4.0;
        let bar_rect = Rect::new(
            Vec2::ZERO,
            Vec2::new(self.screen_size.x, h),
        );

        self.renderer.draw_rect(bar_rect, self.style.bg_panel, 0.0);
        self.renderer.draw_line(
            Vec2::new(0.0, h),
            Vec2::new(self.screen_size.x, h),
            self.style.border,
            1.0,
        );

        self.menu_bar_y = h;

        // Push horizontal layout for menu items.
        self.layout_stack.push(LayoutCursor {
            pos: Vec2::new(self.style.panel_padding, 2.0),
            available_width: self.screen_size.x,
            horizontal: true,
            row_height: h - 4.0,
            origin_x: self.style.panel_padding,
            origin_y: 2.0,
            indent: 0,
        });

        f(self);

        self.layout_stack.pop();
    }

    /// Draw a menu within a menu bar. The closure can call `menu_item()`.
    pub fn menu<F: FnOnce(&mut UI)>(&mut self, label: &str, f: F) {
        let id = self.gen_id_from_str(label);
        let (tw, _) = self.renderer.measure_text(label, self.style.font_size);
        let padding = 8.0;
        let w = tw + padding * 2.0;
        let h = self.style.item_height;
        let rect = self.allocate_widget(w, h);

        let (hovered, _pressed, released) = self.update_interaction(id, &rect);
        let is_open = self.open_menu.as_deref() == Some(label);

        if released {
            if is_open {
                self.open_menu = None;
            } else {
                self.open_menu = Some(label.to_string());
            }
        }

        // Highlight if open or hovered.
        if is_open || hovered {
            self.renderer.draw_rect(rect, self.style.bg_hover, self.style.corner_radius);
        }

        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            label,
            Vec2::new(rect.min.x + padding, text_y),
            self.style.font_size,
            self.style.text_normal,
        );

        // Draw menu popup.
        if is_open {
            self.push_id(id.0);

            let menu_width = 200.0;
            // We need to draw the popup content. Use a temporary vertical layout.
            let popup_x = rect.min.x;
            let popup_y = rect.max.y + 2.0;

            // Draw popup background (we don't know the height yet, so pre-allocate).
            let popup_bg_rect = Rect::new(
                Vec2::new(popup_x, popup_y),
                Vec2::new(popup_x + menu_width, popup_y + 200.0),
            );
            self.renderer.draw_rect(popup_bg_rect, self.style.bg_panel, self.style.corner_radius);
            self.renderer.draw_rect_outline(popup_bg_rect, self.style.border, 1.0, self.style.corner_radius);

            self.layout_stack.push(LayoutCursor {
                pos: Vec2::new(popup_x + 4.0, popup_y + 4.0),
                available_width: menu_width - 8.0,
                horizontal: false,
                row_height: 0.0,
                origin_x: popup_x + 4.0,
                origin_y: popup_y + 4.0,
                indent: 0,
            });

            f(self);

            self.layout_stack.pop();
            self.pop_id();
        }
    }

    /// Draw a menu item. Returns true if clicked.
    pub fn menu_item(&mut self, label: &str, shortcut: &str) -> bool {
        let id = self.gen_id_from_str(label);
        let h = self.style.item_height;
        let available = self.available_width();
        let rect = self.allocate_widget(available, h);

        let (hovered, _pressed, released) = self.update_interaction(id, &rect);

        if hovered {
            self.renderer.draw_rect(rect, self.style.bg_hover, 0.0);
        }

        let text_y = rect.min.y + (h - self.style.font_size) * 0.5;
        self.renderer.draw_text(
            label,
            Vec2::new(rect.min.x + 4.0, text_y),
            self.style.font_size,
            self.style.text_normal,
        );

        if !shortcut.is_empty() {
            let (sw, _) = self.renderer.measure_text(shortcut, self.style.font_size_small);
            self.renderer.draw_text(
                shortcut,
                Vec2::new(rect.max.x - sw - 4.0, text_y),
                self.style.font_size_small,
                self.style.text_dim,
            );
        }

        if released {
            self.open_menu = None;
        }

        released
    }

    // -----------------------------------------------------------------------
    // Widgets: Tab bar
    // -----------------------------------------------------------------------

    /// Draw a tab bar. Modifies `selected` to the clicked tab index.
    pub fn tab_bar(&mut self, selected: &mut usize, tabs: &[&str]) {
        let h = self.style.item_height + 4.0;
        let available = self.available_width();
        let rect = self.allocate_widget(available, h);

        // Background.
        self.renderer.draw_rect(rect, self.style.bg_widget, 0.0);

        let mut tab_x = rect.min.x;
        for (i, tab_label) in tabs.iter().enumerate() {
            let id = self.gen_id();
            let (tw, _) = self.renderer.measure_text(tab_label, self.style.font_size);
            let tab_w = tw + self.style.panel_padding * 2.0;
            let tab_rect = Rect::new(
                Vec2::new(tab_x, rect.min.y),
                Vec2::new(tab_x + tab_w, rect.max.y),
            );

            let (hovered, _pressed, released) = self.update_interaction(id, &tab_rect);

            if released {
                *selected = i;
            }

            let is_selected = i == *selected;

            // Tab background.
            if is_selected {
                self.renderer.draw_rect(tab_rect, self.style.bg_panel, 0.0);
            } else if hovered {
                self.renderer.draw_rect(tab_rect, self.style.bg_hover, 0.0);
            }

            // Tab text.
            let text_y = tab_rect.min.y + (h - self.style.font_size) * 0.5;
            let text_color = if is_selected {
                self.style.text_bright
            } else {
                self.style.text_dim
            };
            self.renderer.draw_text(
                tab_label,
                Vec2::new(tab_x + self.style.panel_padding, text_y),
                self.style.font_size,
                text_color,
            );

            // Underline indicator for selected tab.
            if is_selected {
                self.renderer.draw_line(
                    Vec2::new(tab_x + 2.0, rect.max.y - 2.0),
                    Vec2::new(tab_x + tab_w - 2.0, rect.max.y - 2.0),
                    self.style.accent,
                    2.0,
                );
            }

            tab_x += tab_w;
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current input state.
    pub fn input(&self) -> &UIInputState {
        &self.input
    }

    /// Current screen size.
    pub fn screen_size(&self) -> Vec2 {
        self.screen_size
    }

    /// Delta time for the current frame.
    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }

    /// Total elapsed time.
    pub fn total_time(&self) -> f32 {
        self.total_time
    }
}

// ---------------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------------

fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn hash_combine(a: u64, b: u64) -> u64 {
    let mut h = a;
    h ^= b.wrapping_add(0x9e3779b97f4a7c15).wrapping_add(h << 6).wrapping_add(h >> 2);
    h
}
