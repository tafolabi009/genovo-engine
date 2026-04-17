//! Retained-mode widget system for the Genovo UI framework.
//!
//! Unlike the immediate-mode approach where widgets are recreated every frame,
//! retained widgets persist across frames and only repaint when their state
//! changes. This allows for efficient updates: only dirty subtrees are
//! re-laid-out and re-painted.
//!
//! # Architecture
//!
//! Widgets implement the [`Widget`] trait, which provides methods for:
//! - **Size computation** (bottom-up): each widget reports its desired size
//!   based on its content and children.
//! - **Arrangement** (top-down): each widget positions its children within
//!   the allotted rectangle.
//! - **Painting**: each widget draws itself into a [`PaintCanvas`].
//! - **Event handling**: mouse and keyboard events bubble up through the tree.
//!
//! The [`WidgetTree`] manages the widget hierarchy, tracking dirty sets for
//! both layout and paint so that only changed subtrees are processed.

use std::collections::{HashMap, HashSet, VecDeque};

use glam::{Mat3, Vec2};
use serde::{Deserialize, Serialize};

use genovo_core::Rect;

use crate::render_commands::{
    Border, Color, CornerRadii, DrawCommand, DrawList, ImageScaleMode, Shadow, TextAlign,
    TextVerticalAlign, TextureId,
};

// ---------------------------------------------------------------------------
// WidgetId
// ---------------------------------------------------------------------------

/// Unique identifier for a widget in the retained widget tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WidgetId(pub u64);

impl WidgetId {
    /// Sentinel value representing no widget.
    pub const INVALID: Self = Self(u64::MAX);

    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn is_invalid(&self) -> bool {
        *self == Self::INVALID
    }
}

impl Default for WidgetId {
    fn default() -> Self {
        Self::INVALID
    }
}

impl std::fmt::Display for WidgetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Widget({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// EventReply
// ---------------------------------------------------------------------------

/// Reply from a widget after handling an event. Controls event propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventReply {
    /// The event was handled. Stop propagation (bubbling).
    Handled,
    /// The event was not handled. Continue bubbling to parent.
    Unhandled,
}

impl EventReply {
    /// Returns `true` if the event was handled.
    pub fn is_handled(&self) -> bool {
        *self == Self::Handled
    }
}

// ---------------------------------------------------------------------------
// InvalidateReason
// ---------------------------------------------------------------------------

/// Reason for invalidating a widget, which determines what work must be
/// redone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InvalidateReason {
    /// Only the visual appearance changed; no layout recomputation needed.
    Paint,
    /// The widget's size or position may have changed; layout and paint
    /// must both be redone.
    Layout,
    /// The order or number of children changed; layout must be redone.
    ChildOrder,
    /// The visibility changed; layout and paint must be redone.
    Visibility,
}

impl InvalidateReason {
    /// Returns `true` if this reason requires a layout pass.
    pub fn needs_layout(&self) -> bool {
        matches!(
            self,
            Self::Layout | Self::ChildOrder | Self::Visibility
        )
    }
}

// ---------------------------------------------------------------------------
// MouseEvent / KeyEvent
// ---------------------------------------------------------------------------

/// A mouse event delivered to a widget.
#[derive(Debug, Clone)]
pub struct MouseEvent {
    /// Position relative to the widget's top-left corner (local space).
    pub local_position: Vec2,
    /// Position in screen space.
    pub screen_position: Vec2,
    /// Which button (if applicable).
    pub button: MouseButton,
    /// Modifier keys held during the event.
    pub modifiers: KeyModifiers,
    /// Scroll wheel delta (for scroll events).
    pub scroll_delta: Vec2,
}

impl MouseEvent {
    pub fn new(local: Vec2, screen: Vec2) -> Self {
        Self {
            local_position: local,
            screen_position: screen,
            button: MouseButton::Left,
            modifiers: KeyModifiers::default(),
            scroll_delta: Vec2::ZERO,
        }
    }

    pub fn with_button(mut self, button: MouseButton) -> Self {
        self.button = button;
        self
    }

    pub fn with_modifiers(mut self, modifiers: KeyModifiers) -> Self {
        self.modifiers = modifiers;
        self
    }

    pub fn with_scroll(mut self, delta: Vec2) -> Self {
        self.scroll_delta = delta;
        self
    }
}

/// Mouse button.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MouseButton {
    #[default]
    Left,
    Right,
    Middle,
}

/// Keyboard modifier keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KeyModifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

impl KeyModifiers {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn with_shift(mut self) -> Self {
        self.shift = true;
        self
    }

    pub fn with_ctrl(mut self) -> Self {
        self.ctrl = true;
        self
    }

    pub fn with_alt(mut self) -> Self {
        self.alt = true;
        self
    }

    pub fn any_modifier(&self) -> bool {
        self.shift || self.ctrl || self.alt || self.meta
    }
}

/// A keyboard event delivered to a widget.
#[derive(Debug, Clone)]
pub struct KeyEvent {
    /// Virtual key code.
    pub key: KeyCode,
    /// Whether the key was pressed (true) or released (false).
    pub pressed: bool,
    /// Whether this is a repeat event (key held down).
    pub is_repeat: bool,
    /// Modifier keys held during the event.
    pub modifiers: KeyModifiers,
    /// Character input (for text input events). `None` for non-character keys.
    pub character: Option<char>,
}

impl KeyEvent {
    pub fn new(key: KeyCode, pressed: bool) -> Self {
        Self {
            key,
            pressed,
            is_repeat: false,
            modifiers: KeyModifiers::default(),
            character: None,
        }
    }

    pub fn with_modifiers(mut self, modifiers: KeyModifiers) -> Self {
        self.modifiers = modifiers;
        self
    }

    pub fn with_character(mut self, ch: char) -> Self {
        self.character = Some(ch);
        self
    }
}

/// Virtual key codes (subset — enough for UI purposes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Key0, Key1, Key2, Key3, Key4, Key5, Key6, Key7, Key8, Key9,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    Escape, Tab, Enter, Space, Backspace, Delete,
    ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
    Home, End, PageUp, PageDown,
    Shift, Control, Alt, Meta,
}

// ---------------------------------------------------------------------------
// Widget trait
// ---------------------------------------------------------------------------

/// The core widget trait. All retained widgets implement this.
///
/// Widgets are stored as trait objects (`Box<dyn Widget>`) in the widget tree.
/// The tree calls these methods in a specific order:
///
/// 1. `compute_desired_size` (bottom-up pass)
/// 2. `arrange_children` (top-down pass)
/// 3. `paint` (render pass, only for dirty widgets)
///
/// Events are dispatched from leaves to root (bubbling) via the `on_*`
/// methods.
pub trait Widget: Send + Sync + 'static {
    /// Returns a human-readable type name for debugging.
    fn type_name(&self) -> &str;

    /// Upcast to `Any` for downcasting support.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Upcast to `Any` (mutable) for downcasting support.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Returns the widget's unique id.
    fn id(&self) -> WidgetId;

    /// Returns whether this widget is visible.
    fn is_visible(&self) -> bool {
        true
    }

    /// Returns whether this widget is enabled (accepts input).
    fn is_enabled(&self) -> bool {
        true
    }

    /// Returns whether this widget can receive keyboard focus.
    fn is_focusable(&self) -> bool {
        false
    }

    /// Returns whether this widget clips its children to its bounds.
    fn clips_children(&self) -> bool {
        false
    }

    /// Returns the current render opacity (0.0 to 1.0).
    fn opacity(&self) -> f32 {
        1.0
    }

    /// Returns the current render transform (identity by default).
    fn render_transform(&self) -> Mat3 {
        Mat3::IDENTITY
    }

    /// Returns the tooltip text for this widget, if any.
    fn tooltip(&self) -> Option<&str> {
        None
    }

    // -- Layout methods ------------------------------------------------------

    /// Compute the widget's desired size. This is called bottom-up: children
    /// are measured first, then the parent uses their sizes to compute its
    /// own desired size.
    ///
    /// `available_size` is a hint from the parent about how much space is
    /// available. Components should return their ideal size, clamped to any
    /// min/max constraints.
    fn compute_desired_size(&self, available_size: Vec2) -> Vec2;

    /// Arrange children within the allotted rect. This is called top-down:
    /// the parent decides how to subdivide its space among its children.
    ///
    /// The default implementation does nothing (leaf widgets have no children
    /// to arrange).
    fn arrange_children(&mut self, _allotted_rect: Rect, _children: &[WidgetId]) -> Vec<Rect> {
        Vec::new()
    }

    // -- Paint methods -------------------------------------------------------

    /// Paint this widget into the canvas. Called only when the widget is dirty.
    fn paint(&self, canvas: &mut PaintCanvas, rect: Rect);

    // -- Event handlers ------------------------------------------------------

    /// Called when the mouse cursor enters this widget's bounds.
    fn on_mouse_enter(&mut self, _event: &MouseEvent) -> EventReply {
        EventReply::Unhandled
    }

    /// Called when the mouse cursor leaves this widget's bounds.
    fn on_mouse_leave(&mut self, _event: &MouseEvent) -> EventReply {
        EventReply::Unhandled
    }

    /// Called when the mouse moves within this widget's bounds.
    fn on_mouse_move(&mut self, _event: &MouseEvent) -> EventReply {
        EventReply::Unhandled
    }

    /// Called when a mouse button is pressed within this widget's bounds.
    fn on_mouse_down(&mut self, _event: &MouseEvent) -> EventReply {
        EventReply::Unhandled
    }

    /// Called when a mouse button is released within this widget's bounds.
    fn on_mouse_up(&mut self, _event: &MouseEvent) -> EventReply {
        EventReply::Unhandled
    }

    /// Called when the mouse wheel scrolls over this widget.
    fn on_scroll(&mut self, _event: &MouseEvent) -> EventReply {
        EventReply::Unhandled
    }

    /// Called when a key is pressed while this widget has focus.
    fn on_key_down(&mut self, _event: &KeyEvent) -> EventReply {
        EventReply::Unhandled
    }

    /// Called when a key is released while this widget has focus.
    fn on_key_up(&mut self, _event: &KeyEvent) -> EventReply {
        EventReply::Unhandled
    }

    /// Called when this widget gains keyboard focus.
    fn on_focus_gained(&mut self) {}

    /// Called when this widget loses keyboard focus.
    fn on_focus_lost(&mut self) {}

    /// Invalidate this widget for the given reason. The widget should update
    /// its internal state to reflect the need for re-layout and/or re-paint.
    fn invalidate(&mut self, _reason: InvalidateReason) {}

    /// Called each frame with the delta time. Widgets can use this for
    /// internal animations or timers.
    fn tick(&mut self, _dt: f32) {}
}

// ---------------------------------------------------------------------------
// PaintCanvas — command buffer for draw calls
// ---------------------------------------------------------------------------

/// A command buffer that widgets paint into. Provides high-level drawing
/// operations that ultimately produce [`DrawCommand`]s.
///
/// The canvas maintains a clip stack and a transform stack so that nested
/// widgets can draw in local coordinates with automatic clipping.
pub struct PaintCanvas {
    /// Accumulated draw commands.
    commands: Vec<DrawCommand>,
    /// Clip rect stack. Each widget can push a clip rect that restricts
    /// drawing to a region.
    clip_stack: Vec<Rect>,
    /// Transform stack. Each widget can push a local transform.
    transform_stack: Vec<Mat3>,
    /// Current opacity (multiplicative). When a parent sets opacity 0.5 and
    /// a child sets 0.5, the effective opacity is 0.25.
    opacity_stack: Vec<f32>,
    /// Current effective opacity.
    current_opacity: f32,
    /// Whether to log paint operations for debugging.
    pub debug_paint: bool,
}

impl PaintCanvas {
    /// Creates a new empty paint canvas.
    pub fn new() -> Self {
        Self {
            commands: Vec::with_capacity(512),
            clip_stack: Vec::new(),
            transform_stack: Vec::new(),
            opacity_stack: Vec::new(),
            current_opacity: 1.0,
            debug_paint: false,
        }
    }

    /// Clear all commands for a new frame.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.clip_stack.clear();
        self.transform_stack.clear();
        self.opacity_stack.clear();
        self.current_opacity = 1.0;
    }

    /// Returns the number of draw commands accumulated.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Drain all commands into a `DrawList`.
    pub fn flush_to_draw_list(&mut self, draw_list: &mut DrawList) {
        for cmd in self.commands.drain(..) {
            draw_list.push(cmd);
        }
    }

    /// Returns the current clip rect (the intersection of all active clips).
    pub fn current_clip(&self) -> Option<&Rect> {
        self.clip_stack.last()
    }

    /// Returns the current transform.
    pub fn current_transform(&self) -> Mat3 {
        self.transform_stack.last().copied().unwrap_or(Mat3::IDENTITY)
    }

    // -- Clip stack -----------------------------------------------------------

    /// Push a clip rectangle. Subsequent drawing is restricted to the
    /// intersection of this rect and the current clip.
    pub fn push_clip(&mut self, rect: Rect) {
        let clipped = if let Some(current) = self.clip_stack.last() {
            Rect::new(
                Vec2::new(rect.min.x.max(current.min.x), rect.min.y.max(current.min.y)),
                Vec2::new(rect.max.x.min(current.max.x), rect.max.y.min(current.max.y)),
            )
        } else {
            rect
        };
        self.clip_stack.push(clipped);
        self.commands.push(DrawCommand::PushClip { rect: clipped });
    }

    /// Pop the most recent clip rectangle.
    pub fn pop_clip(&mut self) {
        self.clip_stack.pop();
        self.commands.push(DrawCommand::PopClip);
    }

    // -- Transform stack ------------------------------------------------------

    /// Push a 2-D transform matrix. Subsequent drawing is transformed by this
    /// matrix (multiplied with any existing transforms).
    pub fn push_transform(&mut self, transform: Mat3) {
        let combined = if let Some(current) = self.transform_stack.last() {
            *current * transform
        } else {
            transform
        };
        self.transform_stack.push(combined);
        self.commands
            .push(DrawCommand::PushTransform { transform: combined });
    }

    /// Pop the most recent transform.
    pub fn pop_transform(&mut self) {
        self.transform_stack.pop();
        self.commands.push(DrawCommand::PopTransform);
    }

    // -- Opacity stack --------------------------------------------------------

    /// Push an opacity level. This multiplies with the current opacity.
    pub fn push_opacity(&mut self, opacity: f32) {
        self.opacity_stack.push(self.current_opacity);
        self.current_opacity *= opacity.clamp(0.0, 1.0);
    }

    /// Pop the most recent opacity level.
    pub fn pop_opacity(&mut self) {
        if let Some(prev) = self.opacity_stack.pop() {
            self.current_opacity = prev;
        }
    }

    /// Returns the current effective opacity.
    pub fn effective_opacity(&self) -> f32 {
        self.current_opacity
    }

    /// Apply opacity to a color.
    fn apply_opacity(&self, color: Color) -> Color {
        if (self.current_opacity - 1.0).abs() < 0.001 {
            color
        } else {
            color.with_alpha(color.a * self.current_opacity)
        }
    }

    // -- Drawing methods ------------------------------------------------------

    /// Draw a filled rectangle.
    pub fn draw_rect(&mut self, rect: Rect, color: Color) {
        let color = self.apply_opacity(color);
        self.commands.push(DrawCommand::Rect {
            rect,
            color,
            corner_radii: CornerRadii::ZERO,
            border: Border::default(),
            shadow: None,
        });
    }

    /// Draw a filled rectangle with rounded corners.
    pub fn draw_rounded_rect(
        &mut self,
        rect: Rect,
        color: Color,
        radii: CornerRadii,
    ) {
        let color = self.apply_opacity(color);
        self.commands.push(DrawCommand::Rect {
            rect,
            color,
            corner_radii: radii,
            border: Border::default(),
            shadow: None,
        });
    }

    /// Draw a filled rectangle with rounded corners and a border.
    pub fn draw_rounded_rect_with_border(
        &mut self,
        rect: Rect,
        color: Color,
        radii: CornerRadii,
        border: Border,
    ) {
        let color = self.apply_opacity(color);
        let border_color = self.apply_opacity(border.color);
        self.commands.push(DrawCommand::Rect {
            rect,
            color,
            corner_radii: radii,
            border: Border::new(border_color, border.width),
            shadow: None,
        });
    }

    /// Draw a filled rectangle with rounded corners, border, and shadow.
    pub fn draw_rounded_rect_with_shadow(
        &mut self,
        rect: Rect,
        color: Color,
        radii: CornerRadii,
        border: Border,
        shadow: Shadow,
    ) {
        let color = self.apply_opacity(color);
        let border_color = self.apply_opacity(border.color);
        let shadow_color = self.apply_opacity(shadow.color);
        self.commands.push(DrawCommand::Rect {
            rect,
            color,
            corner_radii: radii,
            border: Border::new(border_color, border.width),
            shadow: Some(Shadow::new(
                shadow_color,
                shadow.offset,
                shadow.blur_radius,
                shadow.spread,
            )),
        });
    }

    /// Draw a filled circle.
    pub fn draw_circle(&mut self, center: Vec2, radius: f32, color: Color) {
        let color = self.apply_opacity(color);
        self.commands.push(DrawCommand::Circle {
            center,
            radius,
            color,
            border: Border::default(),
        });
    }

    /// Draw a circle with a border.
    pub fn draw_circle_with_border(
        &mut self,
        center: Vec2,
        radius: f32,
        color: Color,
        border: Border,
    ) {
        let color = self.apply_opacity(color);
        let border_color = self.apply_opacity(border.color);
        self.commands.push(DrawCommand::Circle {
            center,
            radius,
            color,
            border: Border::new(border_color, border.width),
        });
    }

    /// Draw a line segment.
    pub fn draw_line(&mut self, start: Vec2, end: Vec2, color: Color, thickness: f32) {
        let color = self.apply_opacity(color);
        self.commands.push(DrawCommand::Line {
            start,
            end,
            color,
            thickness,
        });
    }

    /// Draw a polyline (series of connected line segments).
    pub fn draw_polyline(&mut self, points: &[Vec2], color: Color, thickness: f32, closed: bool) {
        let color = self.apply_opacity(color);
        self.commands.push(DrawCommand::Polyline {
            points: points.to_vec(),
            color,
            thickness,
            closed,
        });
    }

    /// Draw a text string.
    pub fn draw_text(&mut self, text: &str, position: Vec2, font_size: f32, color: Color) {
        let color = self.apply_opacity(color);
        self.commands.push(DrawCommand::Text {
            text: text.to_string(),
            position,
            font_size,
            color,
            font_id: 0,
            max_width: None,
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });
    }

    /// Draw text with full parameters.
    pub fn draw_text_ex(
        &mut self,
        text: &str,
        position: Vec2,
        font_size: f32,
        color: Color,
        font_id: u32,
        max_width: Option<f32>,
        align: TextAlign,
        vertical_align: TextVerticalAlign,
    ) {
        let color = self.apply_opacity(color);
        self.commands.push(DrawCommand::Text {
            text: text.to_string(),
            position,
            font_size,
            color,
            font_id,
            max_width,
            align,
            vertical_align,
        });
    }

    /// Draw a textured image.
    pub fn draw_image(&mut self, rect: Rect, texture: TextureId, tint: Color) {
        let tint = self.apply_opacity(tint);
        self.commands.push(DrawCommand::Image {
            rect,
            texture,
            tint,
            corner_radii: CornerRadii::ZERO,
            scale_mode: ImageScaleMode::Stretch,
            uv_rect: Rect::new(Vec2::ZERO, Vec2::ONE),
        });
    }

    /// Draw a textured image with custom UV rect and scale mode.
    pub fn draw_image_ex(
        &mut self,
        rect: Rect,
        texture: TextureId,
        tint: Color,
        corner_radii: CornerRadii,
        scale_mode: ImageScaleMode,
        uv_rect: Rect,
    ) {
        let tint = self.apply_opacity(tint);
        self.commands.push(DrawCommand::Image {
            rect,
            texture,
            tint,
            corner_radii,
            scale_mode,
            uv_rect,
        });
    }

    /// Draw a filled triangle.
    pub fn draw_triangle(&mut self, p0: Vec2, p1: Vec2, p2: Vec2, color: Color) {
        let color = self.apply_opacity(color);
        self.commands.push(DrawCommand::Triangle {
            p0,
            p1,
            p2,
            color,
        });
    }

    /// Draw a horizontal separator line.
    pub fn draw_separator_h(&mut self, y: f32, x_start: f32, x_end: f32, color: Color) {
        self.draw_line(
            Vec2::new(x_start, y),
            Vec2::new(x_end, y),
            color,
            1.0,
        );
    }

    /// Draw a vertical separator line.
    pub fn draw_separator_v(&mut self, x: f32, y_start: f32, y_end: f32, color: Color) {
        self.draw_line(
            Vec2::new(x, y_start),
            Vec2::new(x, y_end),
            color,
            1.0,
        );
    }

    /// Draw a debug outline around a rect.
    pub fn draw_debug_rect(&mut self, rect: Rect, color: Color) {
        if self.debug_paint {
            self.draw_rounded_rect_with_border(
                rect,
                Color::TRANSPARENT,
                CornerRadii::ZERO,
                Border::new(color, 1.0),
            );
        }
    }
}

impl Default for PaintCanvas {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WidgetEvent — events that the tree dispatches
// ---------------------------------------------------------------------------

/// Events that can be dispatched through the widget tree.
#[derive(Debug, Clone)]
pub enum WidgetEvent {
    MouseEnter(MouseEvent),
    MouseLeave(MouseEvent),
    MouseMove(MouseEvent),
    MouseDown(MouseEvent),
    MouseUp(MouseEvent),
    Scroll(MouseEvent),
    KeyDown(KeyEvent),
    KeyUp(KeyEvent),
    FocusGained,
    FocusLost,
}

// ---------------------------------------------------------------------------
// WidgetTree — the retained widget hierarchy
// ---------------------------------------------------------------------------

/// The retained widget tree. Manages the hierarchy, dirty tracking, layout,
/// painting, and event dispatch.
pub struct WidgetTree {
    /// Root widget id.
    root: WidgetId,
    /// Child relationships: parent -> ordered children.
    children: HashMap<WidgetId, Vec<WidgetId>>,
    /// Parent relationships: child -> parent.
    parent: HashMap<WidgetId, WidgetId>,
    /// All widgets by id.
    widgets: HashMap<WidgetId, Box<dyn Widget>>,
    /// Computed rects for each widget.
    rects: HashMap<WidgetId, Rect>,
    /// Desired sizes computed during the measure pass.
    desired_sizes: HashMap<WidgetId, Vec2>,
    /// Widgets that need to be repainted.
    dirty_paint: HashSet<WidgetId>,
    /// Widgets that need to be re-laid-out.
    dirty_layout: HashSet<WidgetId>,
    /// The widget that currently has keyboard focus.
    focused: WidgetId,
    /// The widget that the mouse is currently over.
    hovered: WidgetId,
    /// The widget being pressed (mouse down not yet released).
    pressed: WidgetId,
    /// The widget capturing mouse events (during drag operations).
    mouse_capture: WidgetId,
    /// Next unique id.
    next_id: u64,
    /// Total elapsed time in seconds.
    elapsed: f32,
    /// Whether the entire tree needs a full layout pass.
    needs_full_layout: bool,
}

impl WidgetTree {
    /// Creates a new empty widget tree.
    pub fn new() -> Self {
        Self {
            root: WidgetId::INVALID,
            children: HashMap::new(),
            parent: HashMap::new(),
            widgets: HashMap::new(),
            rects: HashMap::new(),
            desired_sizes: HashMap::new(),
            dirty_paint: HashSet::new(),
            dirty_layout: HashSet::new(),
            focused: WidgetId::INVALID,
            hovered: WidgetId::INVALID,
            pressed: WidgetId::INVALID,
            mouse_capture: WidgetId::INVALID,
            next_id: 0,
            elapsed: 0.0,
            needs_full_layout: true,
        }
    }

    /// Allocate a new unique widget id.
    pub fn next_id(&mut self) -> WidgetId {
        let id = WidgetId::new(self.next_id);
        self.next_id += 1;
        id
    }

    // -- Tree operations ------------------------------------------------------

    /// Set the root widget.
    pub fn set_root(&mut self, widget: Box<dyn Widget>) {
        let id = widget.id();
        self.root = id;
        self.widgets.insert(id, widget);
        self.children.entry(id).or_default();
        self.needs_full_layout = true;
        self.dirty_paint.insert(id);
    }

    /// Add a child widget to a parent.
    pub fn add_child(&mut self, parent_id: WidgetId, widget: Box<dyn Widget>) {
        let child_id = widget.id();
        self.widgets.insert(child_id, widget);
        self.children.entry(parent_id).or_default().push(child_id);
        self.children.entry(child_id).or_default();
        self.parent.insert(child_id, parent_id);
        self.dirty_layout.insert(parent_id);
        self.dirty_paint.insert(child_id);
    }

    /// Insert a child at a specific index.
    pub fn insert_child(&mut self, parent_id: WidgetId, index: usize, widget: Box<dyn Widget>) {
        let child_id = widget.id();
        self.widgets.insert(child_id, widget);
        let children = self.children.entry(parent_id).or_default();
        let idx = index.min(children.len());
        children.insert(idx, child_id);
        self.children.entry(child_id).or_default();
        self.parent.insert(child_id, parent_id);
        self.dirty_layout.insert(parent_id);
        self.dirty_paint.insert(child_id);
    }

    /// Remove a widget and all its descendants from the tree.
    pub fn remove(&mut self, id: WidgetId) -> Option<Box<dyn Widget>> {
        if id == self.root {
            log::warn!("Cannot remove the root widget");
            return None;
        }

        // Collect all descendants
        let mut to_remove = Vec::new();
        self.collect_descendants(id, &mut to_remove);
        to_remove.push(id);

        // Remove from parent's child list
        if let Some(&parent_id) = self.parent.get(&id) {
            if let Some(siblings) = self.children.get_mut(&parent_id) {
                siblings.retain(|c| *c != id);
            }
            self.dirty_layout.insert(parent_id);
        }

        // Remove all descendants
        for &desc_id in &to_remove[..to_remove.len() - 1] {
            self.widgets.remove(&desc_id);
            self.children.remove(&desc_id);
            self.parent.remove(&desc_id);
            self.rects.remove(&desc_id);
            self.desired_sizes.remove(&desc_id);
            self.dirty_paint.remove(&desc_id);
            self.dirty_layout.remove(&desc_id);

            // Clean up focus/hover/pressed state
            if self.focused == desc_id {
                self.focused = WidgetId::INVALID;
            }
            if self.hovered == desc_id {
                self.hovered = WidgetId::INVALID;
            }
            if self.pressed == desc_id {
                self.pressed = WidgetId::INVALID;
            }
        }

        // Remove the widget itself
        self.children.remove(&id);
        self.parent.remove(&id);
        self.rects.remove(&id);
        self.desired_sizes.remove(&id);
        self.dirty_paint.remove(&id);
        self.dirty_layout.remove(&id);

        if self.focused == id {
            self.focused = WidgetId::INVALID;
        }
        if self.hovered == id {
            self.hovered = WidgetId::INVALID;
        }
        if self.pressed == id {
            self.pressed = WidgetId::INVALID;
        }

        self.widgets.remove(&id)
    }

    fn collect_descendants(&self, id: WidgetId, out: &mut Vec<WidgetId>) {
        if let Some(children) = self.children.get(&id) {
            for &child in children {
                out.push(child);
                self.collect_descendants(child, out);
            }
        }
    }

    /// Move a widget from one parent to another.
    pub fn reparent(&mut self, widget_id: WidgetId, new_parent: WidgetId) -> bool {
        if widget_id == self.root {
            return false;
        }

        // Remove from old parent
        if let Some(&old_parent) = self.parent.get(&widget_id) {
            if let Some(siblings) = self.children.get_mut(&old_parent) {
                siblings.retain(|c| *c != widget_id);
            }
            self.dirty_layout.insert(old_parent);
        }

        // Add to new parent
        self.children
            .entry(new_parent)
            .or_default()
            .push(widget_id);
        self.parent.insert(widget_id, new_parent);
        self.dirty_layout.insert(new_parent);

        true
    }

    // -- Accessors -----------------------------------------------------------

    /// Get a reference to a widget.
    pub fn get(&self, id: WidgetId) -> Option<&dyn Widget> {
        self.widgets.get(&id).map(|w| w.as_ref())
    }

    /// Get a mutable reference to a widget.
    pub fn get_mut(&mut self, id: WidgetId) -> Option<&mut dyn Widget> {
        self.widgets.get_mut(&id).map(|w| w.as_mut())
    }

    /// Downcast a widget to a concrete type.
    pub fn get_as<T: Widget + 'static>(&self, id: WidgetId) -> Option<&T> {
        self.widgets
            .get(&id)
            .and_then(|w| w.as_any().downcast_ref::<T>())
    }

    /// Downcast a widget to a concrete type (mutable).
    pub fn get_as_mut<T: Widget + 'static>(&mut self, id: WidgetId) -> Option<&mut T> {
        self.widgets
            .get_mut(&id)
            .and_then(|w| w.as_any_mut().downcast_mut::<T>())
    }

    /// Returns the root widget id.
    pub fn root(&self) -> WidgetId {
        self.root
    }

    /// Returns the children of a widget.
    pub fn children_of(&self, id: WidgetId) -> &[WidgetId] {
        self.children
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns the parent of a widget.
    pub fn parent_of(&self, id: WidgetId) -> WidgetId {
        self.parent.get(&id).copied().unwrap_or(WidgetId::INVALID)
    }

    /// Returns the computed rect of a widget.
    pub fn rect_of(&self, id: WidgetId) -> Rect {
        self.rects
            .get(&id)
            .copied()
            .unwrap_or(Rect::new(Vec2::ZERO, Vec2::ZERO))
    }

    /// Returns the desired size of a widget.
    pub fn desired_size_of(&self, id: WidgetId) -> Vec2 {
        self.desired_sizes
            .get(&id)
            .copied()
            .unwrap_or(Vec2::ZERO)
    }

    /// Returns the number of widgets in the tree.
    pub fn widget_count(&self) -> usize {
        self.widgets.len()
    }

    /// Returns the number of dirty (paint) widgets.
    pub fn dirty_paint_count(&self) -> usize {
        self.dirty_paint.len()
    }

    /// Returns the number of dirty (layout) widgets.
    pub fn dirty_layout_count(&self) -> usize {
        self.dirty_layout.len()
    }

    /// Returns the currently focused widget id.
    pub fn focused(&self) -> WidgetId {
        self.focused
    }

    /// Returns the currently hovered widget id.
    pub fn hovered(&self) -> WidgetId {
        self.hovered
    }

    /// Returns the ancestor chain from a widget up to (but not including) root.
    pub fn ancestors(&self, id: WidgetId) -> Vec<WidgetId> {
        let mut result = Vec::new();
        let mut current = id;
        loop {
            let parent = self.parent_of(current);
            if parent.is_invalid() || parent == current {
                break;
            }
            result.push(parent);
            current = parent;
        }
        result
    }

    // -- Dirty tracking ------------------------------------------------------

    /// Mark a widget as needing repaint.
    pub fn mark_dirty_paint(&mut self, id: WidgetId) {
        self.dirty_paint.insert(id);
    }

    /// Mark a widget as needing re-layout.
    pub fn mark_dirty_layout(&mut self, id: WidgetId) {
        self.dirty_layout.insert(id);
        // Layout changes always require repaint too.
        self.dirty_paint.insert(id);
    }

    /// Invalidate a widget for the given reason.
    pub fn invalidate(&mut self, id: WidgetId, reason: InvalidateReason) {
        if reason.needs_layout() {
            self.mark_dirty_layout(id);
            // Propagate layout dirtiness up to the root so that the layout
            // pass processes the entire path.
            let mut current = id;
            loop {
                let parent = self.parent_of(current);
                if parent.is_invalid() || parent == current {
                    break;
                }
                self.dirty_layout.insert(parent);
                current = parent;
            }
        } else {
            self.mark_dirty_paint(id);
        }

        if let Some(w) = self.widgets.get_mut(&id) {
            w.invalidate(reason);
        }
    }

    // -- Focus management ----------------------------------------------------

    /// Set keyboard focus to a widget. Sends focus-lost to the old widget and
    /// focus-gained to the new one.
    pub fn set_focus(&mut self, id: WidgetId) {
        if self.focused == id {
            return;
        }

        // Blur old
        let old_focused = self.focused;
        if !old_focused.is_invalid() {
            if let Some(w) = self.widgets.get_mut(&old_focused) {
                w.on_focus_lost();
            }
            self.dirty_paint.insert(old_focused);
        }

        // Focus new
        self.focused = id;
        if !id.is_invalid() {
            if let Some(w) = self.widgets.get_mut(&id) {
                w.on_focus_gained();
            }
            self.dirty_paint.insert(id);
        }
    }

    /// Clear keyboard focus.
    pub fn clear_focus(&mut self) {
        self.set_focus(WidgetId::INVALID);
    }

    /// Advance focus to the next focusable widget in tree order.
    pub fn focus_next(&mut self) {
        let focusable = self.collect_focusable_ids();
        if focusable.is_empty() {
            return;
        }

        let current_idx = focusable
            .iter()
            .position(|&id| id == self.focused)
            .unwrap_or(focusable.len() - 1);
        let next_idx = (current_idx + 1) % focusable.len();
        self.set_focus(focusable[next_idx]);
    }

    /// Advance focus to the previous focusable widget in tree order.
    pub fn focus_prev(&mut self) {
        let focusable = self.collect_focusable_ids();
        if focusable.is_empty() {
            return;
        }

        let current_idx = focusable
            .iter()
            .position(|&id| id == self.focused)
            .unwrap_or(0);
        let prev_idx = if current_idx == 0 {
            focusable.len() - 1
        } else {
            current_idx - 1
        };
        self.set_focus(focusable[prev_idx]);
    }

    fn collect_focusable_ids(&self) -> Vec<WidgetId> {
        let mut result = Vec::new();
        if !self.root.is_invalid() {
            self.collect_focusable_recursive(self.root, &mut result);
        }
        result
    }

    fn collect_focusable_recursive(&self, id: WidgetId, out: &mut Vec<WidgetId>) {
        if let Some(w) = self.widgets.get(&id) {
            if w.is_visible() && w.is_enabled() && w.is_focusable() {
                out.push(id);
            }
        }
        if let Some(children) = self.children.get(&id) {
            for &child in children {
                self.collect_focusable_recursive(child, out);
            }
        }
    }

    // -- Per-frame updates ---------------------------------------------------

    /// Tick all widgets with the delta time.
    pub fn tick(&mut self, dt: f32) {
        self.elapsed += dt;
        let widget_ids: Vec<WidgetId> = self.widgets.keys().copied().collect();
        for id in widget_ids {
            if let Some(w) = self.widgets.get_mut(&id) {
                w.tick(dt);
            }
        }
    }

    /// Perform the layout pass.
    ///
    /// This is a two-pass algorithm:
    /// 1. **Measure** (bottom-up): compute desired sizes from leaves to root.
    /// 2. **Arrange** (top-down): allocate space from root to leaves.
    ///
    /// Only dirty subtrees are processed (unless `needs_full_layout` is set).
    pub fn layout(&mut self, viewport: Rect) {
        if self.root.is_invalid() {
            return;
        }

        if self.needs_full_layout {
            // Full layout — mark everything dirty
            let all_ids: Vec<WidgetId> = self.widgets.keys().copied().collect();
            for id in all_ids {
                self.dirty_layout.insert(id);
            }
            self.needs_full_layout = false;
        }

        if self.dirty_layout.is_empty() {
            return;
        }

        // Measure pass (bottom-up)
        let viewport_size = Vec2::new(viewport.width(), viewport.height());
        self.measure_recursive(self.root, viewport_size);

        // Arrange pass (top-down)
        self.arrange_recursive(self.root, viewport);

        // Clear layout dirty set
        self.dirty_layout.clear();
    }

    fn measure_recursive(&mut self, id: WidgetId, available_size: Vec2) {
        // First measure all children
        if let Some(children) = self.children.get(&id).cloned() {
            for child in children {
                self.measure_recursive(child, available_size);
            }
        }

        // Then measure this widget
        if let Some(widget) = self.widgets.get(&id) {
            let desired = widget.compute_desired_size(available_size);
            self.desired_sizes.insert(id, desired);
        }
    }

    fn arrange_recursive(&mut self, id: WidgetId, allotted_rect: Rect) {
        // Store the rect for this widget
        self.rects.insert(id, allotted_rect);
        self.dirty_paint.insert(id);

        // Get child ids and child rects from the widget's arrange_children
        let child_ids = self
            .children
            .get(&id)
            .cloned()
            .unwrap_or_default();

        if child_ids.is_empty() {
            return;
        }

        // Call arrange_children on the widget
        let child_rects = if let Some(widget) = self.widgets.get_mut(&id) {
            widget.arrange_children(allotted_rect, &child_ids)
        } else {
            Vec::new()
        };

        // Recurse into children
        for (i, &child_id) in child_ids.iter().enumerate() {
            let child_rect = child_rects
                .get(i)
                .copied()
                .unwrap_or(allotted_rect);
            self.arrange_recursive(child_id, child_rect);
        }
    }

    /// Paint all dirty widgets into the canvas.
    ///
    /// Only widgets in the dirty set are repainted. The canvas accumulates
    /// draw commands that can then be flushed to a `DrawList`.
    pub fn paint(&mut self, canvas: &mut PaintCanvas) {
        if self.root.is_invalid() {
            return;
        }

        // For now, paint the entire tree. A more sophisticated implementation
        // would only paint dirty subtrees and use cached render targets for
        // clean subtrees.
        self.paint_recursive(self.root, canvas);
        self.dirty_paint.clear();
    }

    fn paint_recursive(&self, id: WidgetId, canvas: &mut PaintCanvas) {
        let widget = match self.widgets.get(&id) {
            Some(w) => w,
            None => return,
        };

        if !widget.is_visible() {
            return;
        }

        let rect = self.rect_of(id);
        let opacity = widget.opacity();
        let transform = widget.render_transform();
        let clips = widget.clips_children();
        let has_transform = transform != Mat3::IDENTITY;

        // Push state
        if opacity < 1.0 {
            canvas.push_opacity(opacity);
        }
        if has_transform {
            canvas.push_transform(transform);
        }
        if clips {
            canvas.push_clip(rect);
        }

        // Paint this widget
        widget.paint(canvas, rect);

        // Paint children
        if let Some(children) = self.children.get(&id) {
            for &child in children {
                self.paint_recursive(child, canvas);
            }
        }

        // Pop state (reverse order)
        if clips {
            canvas.pop_clip();
        }
        if has_transform {
            canvas.pop_transform();
        }
        if opacity < 1.0 {
            canvas.pop_opacity();
        }
    }

    // -- Event dispatch -------------------------------------------------------

    /// Dispatch a widget event through the tree. Events are routed to the
    /// deepest matching widget and then bubble up to the root.
    pub fn dispatch_event(&mut self, event: WidgetEvent) {
        match event {
            WidgetEvent::MouseMove(ref me) => {
                let screen_pos = me.screen_position;

                // If a widget has mouse capture, route to it directly
                if !self.mouse_capture.is_invalid() {
                    self.dispatch_to_widget(self.mouse_capture, event);
                    return;
                }

                // Hit test to find which widget is under the cursor
                let hit = self.hit_test(screen_pos);

                // Handle hover transitions
                if hit != self.hovered {
                    let old_hovered = self.hovered;
                    self.hovered = hit;

                    if !old_hovered.is_invalid() {
                        let leave_event = MouseEvent::new(
                            self.to_local(old_hovered, screen_pos),
                            screen_pos,
                        );
                        self.dispatch_to_widget(
                            old_hovered,
                            WidgetEvent::MouseLeave(leave_event),
                        );
                    }

                    if !hit.is_invalid() {
                        let enter_event = MouseEvent::new(
                            self.to_local(hit, screen_pos),
                            screen_pos,
                        );
                        self.dispatch_to_widget(
                            hit,
                            WidgetEvent::MouseEnter(enter_event),
                        );
                    }
                }

                if !hit.is_invalid() {
                    let move_event = MouseEvent::new(
                        self.to_local(hit, screen_pos),
                        screen_pos,
                    )
                    .with_button(me.button)
                    .with_modifiers(me.modifiers);
                    self.dispatch_to_widget(hit, WidgetEvent::MouseMove(move_event));
                }
            }
            WidgetEvent::MouseDown(ref me) => {
                let screen_pos = me.screen_position;
                let hit = self.hit_test(screen_pos);

                if !hit.is_invalid() {
                    self.pressed = hit;
                    self.mouse_capture = hit;

                    // Set focus if the widget is focusable
                    if let Some(w) = self.widgets.get(&hit) {
                        if w.is_focusable() {
                            self.set_focus(hit);
                        }
                    }

                    let local_event = MouseEvent::new(
                        self.to_local(hit, screen_pos),
                        screen_pos,
                    )
                    .with_button(me.button)
                    .with_modifiers(me.modifiers);
                    self.dispatch_to_widget(hit, WidgetEvent::MouseDown(local_event));
                } else {
                    // Clicked on empty space — clear focus
                    self.clear_focus();
                }
            }
            WidgetEvent::MouseUp(ref me) => {
                let screen_pos = me.screen_position;
                let target = if !self.mouse_capture.is_invalid() {
                    self.mouse_capture
                } else {
                    self.hit_test(screen_pos)
                };

                if !target.is_invalid() {
                    let local_event = MouseEvent::new(
                        self.to_local(target, screen_pos),
                        screen_pos,
                    )
                    .with_button(me.button)
                    .with_modifiers(me.modifiers);
                    self.dispatch_to_widget(target, WidgetEvent::MouseUp(local_event));
                }

                self.pressed = WidgetId::INVALID;
                self.mouse_capture = WidgetId::INVALID;
            }
            WidgetEvent::Scroll(ref me) => {
                let screen_pos = me.screen_position;
                let hit = if !self.mouse_capture.is_invalid() {
                    self.mouse_capture
                } else {
                    self.hit_test(screen_pos)
                };

                if !hit.is_invalid() {
                    let local_event = MouseEvent::new(
                        self.to_local(hit, screen_pos),
                        screen_pos,
                    )
                    .with_scroll(me.scroll_delta)
                    .with_modifiers(me.modifiers);
                    self.dispatch_to_widget(hit, WidgetEvent::Scroll(local_event));
                }
            }
            WidgetEvent::KeyDown(_) | WidgetEvent::KeyUp(_) => {
                // Key events go to the focused widget and bubble up.
                if !self.focused.is_invalid() {
                    self.dispatch_to_widget(self.focused, event);
                }
            }
            WidgetEvent::FocusGained | WidgetEvent::FocusLost => {
                // These are dispatched internally by set_focus / clear_focus.
            }
            WidgetEvent::MouseEnter(_) | WidgetEvent::MouseLeave(_) => {
                // These are dispatched internally by hover tracking.
            }
        }
    }

    /// Dispatch an event to a specific widget with bubbling.
    fn dispatch_to_widget(&mut self, id: WidgetId, event: WidgetEvent) {
        let reply = self.handle_event_on_widget(id, &event);
        if reply == EventReply::Handled {
            return;
        }

        // Bubble up through ancestors
        let ancestors = self.ancestors(id);
        for &ancestor_id in &ancestors {
            let reply = self.handle_event_on_widget(ancestor_id, &event);
            if reply == EventReply::Handled {
                return;
            }
        }
    }

    fn handle_event_on_widget(&mut self, id: WidgetId, event: &WidgetEvent) -> EventReply {
        let widget = match self.widgets.get_mut(&id) {
            Some(w) => w,
            None => return EventReply::Unhandled,
        };

        if !widget.is_enabled() {
            return EventReply::Unhandled;
        }

        let reply = match event {
            WidgetEvent::MouseEnter(me) => widget.on_mouse_enter(me),
            WidgetEvent::MouseLeave(me) => widget.on_mouse_leave(me),
            WidgetEvent::MouseMove(me) => widget.on_mouse_move(me),
            WidgetEvent::MouseDown(me) => widget.on_mouse_down(me),
            WidgetEvent::MouseUp(me) => widget.on_mouse_up(me),
            WidgetEvent::Scroll(me) => widget.on_scroll(me),
            WidgetEvent::KeyDown(ke) => widget.on_key_down(ke),
            WidgetEvent::KeyUp(ke) => widget.on_key_up(ke),
            WidgetEvent::FocusGained => {
                widget.on_focus_gained();
                EventReply::Handled
            }
            WidgetEvent::FocusLost => {
                widget.on_focus_lost();
                EventReply::Handled
            }
        };

        if reply == EventReply::Handled {
            self.dirty_paint.insert(id);
        }

        reply
    }

    // -- Hit testing ----------------------------------------------------------

    /// Perform a hit test to find the deepest visible+enabled widget at the
    /// given screen position.
    pub fn hit_test(&self, screen_pos: Vec2) -> WidgetId {
        if self.root.is_invalid() {
            return WidgetId::INVALID;
        }
        self.hit_test_recursive(self.root, screen_pos)
            .unwrap_or(WidgetId::INVALID)
    }

    fn hit_test_recursive(&self, id: WidgetId, screen_pos: Vec2) -> Option<WidgetId> {
        let widget = self.widgets.get(&id)?;
        if !widget.is_visible() || !widget.is_enabled() {
            return None;
        }

        let rect = self.rect_of(id);
        if !rect.contains(screen_pos) {
            return None;
        }

        // Check children in reverse order (last child = topmost)
        if let Some(children) = self.children.get(&id) {
            for &child in children.iter().rev() {
                if let Some(hit) = self.hit_test_recursive(child, screen_pos) {
                    return Some(hit);
                }
            }
        }

        // Return self if point is in our rect
        Some(id)
    }

    /// Convert a screen position to widget-local coordinates.
    fn to_local(&self, id: WidgetId, screen_pos: Vec2) -> Vec2 {
        let rect = self.rect_of(id);
        Vec2::new(screen_pos.x - rect.min.x, screen_pos.y - rect.min.y)
    }

    // -- Mouse capture --------------------------------------------------------

    /// Set mouse capture to a widget. All mouse events will be routed to this
    /// widget until capture is released.
    pub fn capture_mouse(&mut self, id: WidgetId) {
        self.mouse_capture = id;
    }

    /// Release mouse capture.
    pub fn release_mouse(&mut self) {
        self.mouse_capture = WidgetId::INVALID;
    }

    /// Returns the widget that has mouse capture, if any.
    pub fn mouse_capture(&self) -> WidgetId {
        self.mouse_capture
    }

    // -- Traversal helpers ---------------------------------------------------

    /// Visit all widgets in depth-first order.
    pub fn visit_depth_first<F: FnMut(WidgetId, &dyn Widget)>(&self, mut f: F) {
        if !self.root.is_invalid() {
            self.visit_df_recursive(self.root, &mut f);
        }
    }

    fn visit_df_recursive<F: FnMut(WidgetId, &dyn Widget)>(&self, id: WidgetId, f: &mut F) {
        if let Some(w) = self.widgets.get(&id) {
            f(id, w.as_ref());
        }
        if let Some(children) = self.children.get(&id) {
            for &child in children {
                self.visit_df_recursive(child, f);
            }
        }
    }

    /// Visit all widgets in breadth-first order.
    pub fn visit_breadth_first<F: FnMut(WidgetId, &dyn Widget)>(&self, mut f: F) {
        if self.root.is_invalid() {
            return;
        }
        let mut queue = VecDeque::new();
        queue.push_back(self.root);

        while let Some(id) = queue.pop_front() {
            if let Some(w) = self.widgets.get(&id) {
                f(id, w.as_ref());
            }
            if let Some(children) = self.children.get(&id) {
                for &child in children {
                    queue.push_back(child);
                }
            }
        }
    }

    /// Find the first widget matching a predicate.
    pub fn find<F: Fn(&dyn Widget) -> bool>(&self, predicate: F) -> Option<WidgetId> {
        for (&id, widget) in &self.widgets {
            if predicate(widget.as_ref()) {
                return Some(id);
            }
        }
        None
    }

    /// Debug print the tree structure.
    pub fn debug_print(&self) -> String {
        let mut output = String::new();
        if !self.root.is_invalid() {
            self.debug_print_recursive(self.root, 0, &mut output);
        }
        output
    }

    fn debug_print_recursive(&self, id: WidgetId, depth: usize, output: &mut String) {
        let indent = "  ".repeat(depth);
        if let Some(w) = self.widgets.get(&id) {
            let rect = self.rect_of(id);
            let focused = if id == self.focused { " [FOCUSED]" } else { "" };
            let hovered = if id == self.hovered { " [HOVERED]" } else { "" };
            output.push_str(&format!(
                "{}{} ({}) @ [{:.0},{:.0}]-[{:.0},{:.0}]{}{}\n",
                indent,
                w.type_name(),
                id,
                rect.min.x,
                rect.min.y,
                rect.max.x,
                rect.max.y,
                focused,
                hovered,
            ));
        }
        if let Some(children) = self.children.get(&id) {
            for &child in children {
                self.debug_print_recursive(child, depth + 1, output);
            }
        }
    }
}

impl Default for WidgetTree {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in widgets — Panel
// ---------------------------------------------------------------------------

/// A simple panel widget (filled rectangle with optional border).
pub struct PanelWidget {
    id: WidgetId,
    pub background: Color,
    pub border: Border,
    pub corner_radii: CornerRadii,
    pub shadow: Option<Shadow>,
    pub min_size: Vec2,
    pub max_size: Vec2,
    pub visible: bool,
    pub enabled: bool,
    pub opacity_val: f32,
}

impl PanelWidget {
    pub fn new(id: WidgetId) -> Self {
        Self {
            id,
            background: Color::from_hex("#1E1E2E"),
            border: Border::default(),
            corner_radii: CornerRadii::ZERO,
            shadow: None,
            min_size: Vec2::ZERO,
            max_size: Vec2::new(f32::MAX, f32::MAX),
            visible: true,
            enabled: true,
            opacity_val: 1.0,
        }
    }

    pub fn with_background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    pub fn with_border(mut self, border: Border) -> Self {
        self.border = border;
        self
    }

    pub fn with_corner_radii(mut self, radii: CornerRadii) -> Self {
        self.corner_radii = radii;
        self
    }

    pub fn with_min_size(mut self, size: Vec2) -> Self {
        self.min_size = size;
        self
    }
}

impl Widget for PanelWidget {
    fn type_name(&self) -> &str {
        "Panel"
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn id(&self) -> WidgetId {
        self.id
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn opacity(&self) -> f32 {
        self.opacity_val
    }

    fn compute_desired_size(&self, _available_size: Vec2) -> Vec2 {
        self.min_size
    }

    fn arrange_children(&mut self, allotted_rect: Rect, children: &[WidgetId]) -> Vec<Rect> {
        // Panel stacks all children to fill its rect
        children.iter().map(|_| allotted_rect).collect()
    }

    fn paint(&self, canvas: &mut PaintCanvas, rect: Rect) {
        if let Some(ref shadow) = self.shadow {
            canvas.draw_rounded_rect_with_shadow(
                rect,
                self.background,
                self.corner_radii,
                self.border,
                *shadow,
            );
        } else if self.border.width > 0.0 {
            canvas.draw_rounded_rect_with_border(
                rect,
                self.background,
                self.corner_radii,
                self.border,
            );
        } else if !self.corner_radii.is_zero() {
            canvas.draw_rounded_rect(rect, self.background, self.corner_radii);
        } else {
            canvas.draw_rect(rect, self.background);
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in widgets — Label
// ---------------------------------------------------------------------------

/// A text label widget.
pub struct LabelWidget {
    id: WidgetId,
    pub text: String,
    pub font_size: f32,
    pub color: Color,
    pub font_id: u32,
    pub align: TextAlign,
    pub vertical_align: TextVerticalAlign,
    pub visible: bool,
}

impl LabelWidget {
    pub fn new(id: WidgetId, text: impl Into<String>) -> Self {
        Self {
            id,
            text: text.into(),
            font_size: 14.0,
            color: Color::WHITE,
            font_id: 0,
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
            visible: true,
        }
    }

    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn with_align(mut self, align: TextAlign) -> Self {
        self.align = align;
        self
    }
}

impl Widget for LabelWidget {
    fn type_name(&self) -> &str {
        "Label"
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn id(&self) -> WidgetId {
        self.id
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn compute_desired_size(&self, _available_size: Vec2) -> Vec2 {
        // Estimate text size
        let text_width = self.text.len() as f32 * self.font_size * 0.55;
        let text_height = self.font_size * 1.2;
        Vec2::new(text_width, text_height)
    }

    fn paint(&self, canvas: &mut PaintCanvas, rect: Rect) {
        canvas.draw_text_ex(
            &self.text,
            rect.min,
            self.font_size,
            self.color,
            self.font_id,
            Some(rect.width()),
            self.align,
            self.vertical_align,
        );
    }
}

// ---------------------------------------------------------------------------
// Built-in widgets — ButtonWidget
// ---------------------------------------------------------------------------

/// A retained-mode button widget.
pub struct ButtonWidget {
    id: WidgetId,
    pub text: String,
    pub font_size: f32,
    pub text_color: Color,
    pub background: Color,
    pub hovered_background: Color,
    pub pressed_background: Color,
    pub disabled_background: Color,
    pub border: Border,
    pub corner_radii: CornerRadii,
    pub padding: Vec2,
    pub min_size: Vec2,
    pub enabled_val: bool,
    pub visible: bool,
    pub focusable: bool,
    state: ButtonState,
    pub on_click: Option<Box<dyn Fn() + Send + Sync>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ButtonState {
    Normal,
    Hovered,
    Pressed,
}

impl ButtonWidget {
    pub fn new(id: WidgetId, text: impl Into<String>) -> Self {
        Self {
            id,
            text: text.into(),
            font_size: 14.0,
            text_color: Color::WHITE,
            background: Color::from_hex("#45475A"),
            hovered_background: Color::from_hex("#585B70"),
            pressed_background: Color::from_hex("#313244"),
            disabled_background: Color::from_hex("#313244"),
            border: Border::default(),
            corner_radii: CornerRadii::all(4.0),
            padding: Vec2::new(12.0, 6.0),
            min_size: Vec2::new(60.0, 28.0),
            enabled_val: true,
            visible: true,
            focusable: true,
            state: ButtonState::Normal,
            on_click: None,
        }
    }

    pub fn with_on_click<F: Fn() + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.on_click = Some(Box::new(f));
        self
    }

    fn current_background(&self) -> Color {
        if !self.enabled_val {
            self.disabled_background
        } else {
            match self.state {
                ButtonState::Normal => self.background,
                ButtonState::Hovered => self.hovered_background,
                ButtonState::Pressed => self.pressed_background,
            }
        }
    }
}

impl Widget for ButtonWidget {
    fn type_name(&self) -> &str {
        "Button"
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn id(&self) -> WidgetId {
        self.id
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn is_enabled(&self) -> bool {
        self.enabled_val
    }

    fn is_focusable(&self) -> bool {
        self.focusable
    }

    fn compute_desired_size(&self, _available_size: Vec2) -> Vec2 {
        let text_width = self.text.len() as f32 * self.font_size * 0.55;
        let text_height = self.font_size * 1.2;
        Vec2::new(
            (text_width + self.padding.x * 2.0).max(self.min_size.x),
            (text_height + self.padding.y * 2.0).max(self.min_size.y),
        )
    }

    fn paint(&self, canvas: &mut PaintCanvas, rect: Rect) {
        let bg = self.current_background();
        canvas.draw_rounded_rect_with_border(
            rect,
            bg,
            self.corner_radii,
            self.border,
        );

        // Draw text centered
        let text_x = rect.min.x + self.padding.x;
        let text_y = rect.min.y + (rect.height() - self.font_size) * 0.5;
        canvas.draw_text(
            &self.text,
            Vec2::new(text_x, text_y),
            self.font_size,
            self.text_color,
        );
    }

    fn on_mouse_enter(&mut self, _event: &MouseEvent) -> EventReply {
        self.state = ButtonState::Hovered;
        EventReply::Handled
    }

    fn on_mouse_leave(&mut self, _event: &MouseEvent) -> EventReply {
        self.state = ButtonState::Normal;
        EventReply::Handled
    }

    fn on_mouse_down(&mut self, _event: &MouseEvent) -> EventReply {
        self.state = ButtonState::Pressed;
        EventReply::Handled
    }

    fn on_mouse_up(&mut self, _event: &MouseEvent) -> EventReply {
        if self.state == ButtonState::Pressed {
            if let Some(ref on_click) = self.on_click {
                on_click();
            }
        }
        self.state = ButtonState::Hovered;
        EventReply::Handled
    }

    fn on_key_down(&mut self, event: &KeyEvent) -> EventReply {
        if event.key == KeyCode::Enter || event.key == KeyCode::Space {
            if let Some(ref on_click) = self.on_click {
                on_click();
            }
            EventReply::Handled
        } else {
            EventReply::Unhandled
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in widgets — StackLayout
// ---------------------------------------------------------------------------

/// Layout direction for stack layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackDirection {
    Horizontal,
    Vertical,
}

/// A stack layout widget that arranges children either horizontally or
/// vertically.
pub struct StackLayoutWidget {
    id: WidgetId,
    pub direction: StackDirection,
    pub spacing: f32,
    pub padding: Vec2,
    pub visible: bool,
}

impl StackLayoutWidget {
    pub fn horizontal(id: WidgetId) -> Self {
        Self {
            id,
            direction: StackDirection::Horizontal,
            spacing: 4.0,
            padding: Vec2::ZERO,
            visible: true,
        }
    }

    pub fn vertical(id: WidgetId) -> Self {
        Self {
            id,
            direction: StackDirection::Vertical,
            spacing: 4.0,
            padding: Vec2::ZERO,
            visible: true,
        }
    }

    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    pub fn with_padding(mut self, padding: Vec2) -> Self {
        self.padding = padding;
        self
    }
}

impl Widget for StackLayoutWidget {
    fn type_name(&self) -> &str {
        "StackLayout"
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn id(&self) -> WidgetId {
        self.id
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn compute_desired_size(&self, _available_size: Vec2) -> Vec2 {
        // Size is computed from children during the measure pass; this is
        // just the minimum.
        self.padding * 2.0
    }

    fn arrange_children(&mut self, allotted_rect: Rect, children: &[WidgetId]) -> Vec<Rect> {
        let mut rects = Vec::with_capacity(children.len());
        let content_rect = Rect::new(
            allotted_rect.min + self.padding,
            allotted_rect.max - self.padding,
        );

        let mut offset = 0.0_f32;

        for (i, _child) in children.iter().enumerate() {
            let child_size = content_rect.width().min(content_rect.height());
            let _ = child_size; // child sizes are handled by the measure pass

            let rect = match self.direction {
                StackDirection::Horizontal => {
                    let w = (content_rect.width() - self.spacing * (children.len() as f32 - 1.0))
                        / children.len() as f32;
                    let r = Rect::new(
                        Vec2::new(content_rect.min.x + offset, content_rect.min.y),
                        Vec2::new(content_rect.min.x + offset + w, content_rect.max.y),
                    );
                    offset += w + self.spacing;
                    r
                }
                StackDirection::Vertical => {
                    let h = (content_rect.height() - self.spacing * (children.len() as f32 - 1.0))
                        / children.len() as f32;
                    let r = Rect::new(
                        Vec2::new(content_rect.min.x, content_rect.min.y + offset),
                        Vec2::new(content_rect.max.x, content_rect.min.y + offset + h),
                    );
                    offset += h + self.spacing;
                    r
                }
            };

            rects.push(rect);
        }

        rects
    }

    fn paint(&self, _canvas: &mut PaintCanvas, _rect: Rect) {
        // Stack layout itself is invisible; children paint themselves.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_id() {
        let id = WidgetId::new(42);
        assert!(!id.is_invalid());
        assert_eq!(format!("{}", id), "Widget(42)");

        let invalid = WidgetId::INVALID;
        assert!(invalid.is_invalid());
    }

    #[test]
    fn test_paint_canvas_basic() {
        let mut canvas = PaintCanvas::new();
        canvas.draw_rect(
            Rect::new(Vec2::ZERO, Vec2::new(100.0, 50.0)),
            Color::RED,
        );
        canvas.draw_text("Hello", Vec2::new(10.0, 10.0), 14.0, Color::WHITE);
        assert_eq!(canvas.command_count(), 2);
    }

    #[test]
    fn test_paint_canvas_opacity() {
        let mut canvas = PaintCanvas::new();
        canvas.push_opacity(0.5);
        assert!((canvas.effective_opacity() - 0.5).abs() < 0.001);

        canvas.push_opacity(0.5);
        assert!((canvas.effective_opacity() - 0.25).abs() < 0.001);

        canvas.pop_opacity();
        assert!((canvas.effective_opacity() - 0.5).abs() < 0.001);

        canvas.pop_opacity();
        assert!((canvas.effective_opacity() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_paint_canvas_clip_stack() {
        let mut canvas = PaintCanvas::new();
        let rect1 = Rect::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        let rect2 = Rect::new(Vec2::new(25.0, 25.0), Vec2::new(75.0, 75.0));

        canvas.push_clip(rect1);
        canvas.push_clip(rect2);

        let clip = canvas.current_clip().unwrap();
        assert_eq!(clip.min.x, 25.0);
        assert_eq!(clip.min.y, 25.0);
        assert_eq!(clip.max.x, 75.0);
        assert_eq!(clip.max.y, 75.0);

        canvas.pop_clip();
        let clip = canvas.current_clip().unwrap();
        assert_eq!(clip.min.x, 0.0);
    }

    #[test]
    fn test_widget_tree_basic() {
        let mut tree = WidgetTree::new();

        let root_id = tree.next_id();
        let panel = PanelWidget::new(root_id);
        tree.set_root(Box::new(panel));

        assert_eq!(tree.widget_count(), 1);
        assert_eq!(tree.root(), root_id);
    }

    #[test]
    fn test_widget_tree_add_children() {
        let mut tree = WidgetTree::new();

        let root_id = tree.next_id();
        let child1_id = tree.next_id();
        let child2_id = tree.next_id();

        tree.set_root(Box::new(PanelWidget::new(root_id)));
        tree.add_child(root_id, Box::new(LabelWidget::new(child1_id, "Hello")));
        tree.add_child(root_id, Box::new(LabelWidget::new(child2_id, "World")));

        assert_eq!(tree.widget_count(), 3);
        assert_eq!(tree.children_of(root_id).len(), 2);
        assert_eq!(tree.parent_of(child1_id), root_id);
    }

    #[test]
    fn test_widget_tree_remove() {
        let mut tree = WidgetTree::new();

        let root_id = tree.next_id();
        let child_id = tree.next_id();
        let grandchild_id = tree.next_id();

        tree.set_root(Box::new(PanelWidget::new(root_id)));
        tree.add_child(root_id, Box::new(PanelWidget::new(child_id)));
        tree.add_child(child_id, Box::new(LabelWidget::new(grandchild_id, "Test")));

        assert_eq!(tree.widget_count(), 3);
        tree.remove(child_id);
        assert_eq!(tree.widget_count(), 1);
    }

    #[test]
    fn test_widget_tree_layout() {
        let mut tree = WidgetTree::new();

        let root_id = tree.next_id();
        let child_id = tree.next_id();

        tree.set_root(Box::new(PanelWidget::new(root_id)));
        tree.add_child(root_id, Box::new(LabelWidget::new(child_id, "Test Label")));

        let viewport = Rect::new(Vec2::ZERO, Vec2::new(800.0, 600.0));
        tree.layout(viewport);

        let root_rect = tree.rect_of(root_id);
        assert_eq!(root_rect.min, Vec2::ZERO);
        assert_eq!(root_rect.max, Vec2::new(800.0, 600.0));
    }

    #[test]
    fn test_widget_tree_hit_test() {
        let mut tree = WidgetTree::new();

        let root_id = tree.next_id();
        tree.set_root(Box::new(PanelWidget::new(root_id)));

        let viewport = Rect::new(Vec2::ZERO, Vec2::new(800.0, 600.0));
        tree.layout(viewport);

        assert_eq!(tree.hit_test(Vec2::new(400.0, 300.0)), root_id);
        assert_eq!(tree.hit_test(Vec2::new(900.0, 300.0)), WidgetId::INVALID);
    }

    #[test]
    fn test_focus_navigation() {
        let mut tree = WidgetTree::new();

        let root_id = tree.next_id();
        let btn1_id = tree.next_id();
        let btn2_id = tree.next_id();
        let btn3_id = tree.next_id();

        tree.set_root(Box::new(StackLayoutWidget::vertical(root_id)));
        tree.add_child(root_id, Box::new(ButtonWidget::new(btn1_id, "Button 1")));
        tree.add_child(root_id, Box::new(ButtonWidget::new(btn2_id, "Button 2")));
        tree.add_child(root_id, Box::new(ButtonWidget::new(btn3_id, "Button 3")));

        tree.focus_next();
        assert_eq!(tree.focused(), btn1_id);

        tree.focus_next();
        assert_eq!(tree.focused(), btn2_id);

        tree.focus_next();
        assert_eq!(tree.focused(), btn3_id);

        // Wraps around
        tree.focus_next();
        assert_eq!(tree.focused(), btn1_id);

        // Previous
        tree.focus_prev();
        assert_eq!(tree.focused(), btn3_id);
    }

    #[test]
    fn test_event_reply() {
        assert!(EventReply::Handled.is_handled());
        assert!(!EventReply::Unhandled.is_handled());
    }

    #[test]
    fn test_invalidate_reason() {
        assert!(!InvalidateReason::Paint.needs_layout());
        assert!(InvalidateReason::Layout.needs_layout());
        assert!(InvalidateReason::ChildOrder.needs_layout());
        assert!(InvalidateReason::Visibility.needs_layout());
    }

    #[test]
    fn test_debug_print() {
        let mut tree = WidgetTree::new();
        let root_id = tree.next_id();
        let child_id = tree.next_id();

        tree.set_root(Box::new(PanelWidget::new(root_id)));
        tree.add_child(root_id, Box::new(LabelWidget::new(child_id, "Hi")));

        let viewport = Rect::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        tree.layout(viewport);

        let debug = tree.debug_print();
        assert!(debug.contains("Panel"));
        assert!(debug.contains("Label"));
    }
}
