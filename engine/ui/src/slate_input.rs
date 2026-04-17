//! Complete input handling system for the Slate UI.
//!
//! This module provides structured input events, a focus system, mouse
//! capture, drag-and-drop, cursor management, keyboard command binding,
//! high-precision mouse mode, and per-widget hit testing.

use std::collections::HashMap;
use std::fmt;

use glam::Vec2;
use genovo_core::Rect;

use crate::core::{KeyCode, KeyModifiers, MouseButton, UIId};
use crate::render_commands::Color;

// =========================================================================
// 1. Mouse Event
// =========================================================================

/// Fully-specified mouse event with all context needed by widgets.
#[derive(Debug, Clone)]
pub struct MouseEvent {
    /// Current cursor position in logical (DPI-independent) pixels.
    pub position: Vec2,
    /// Cursor delta since last frame.
    pub delta: Vec2,
    /// Which button triggered this event (if any).
    pub button: Option<MouseButton>,
    /// How many rapid clicks have occurred (1 = single, 2 = double, etc.).
    pub click_count: u32,
    /// Keyboard modifiers held at the time of the event.
    pub modifiers: KeyModifiers,
    /// Buttons currently held down (bit mask).
    pub buttons_down: MouseButtonMask,
    /// Screen-space position in physical pixels.
    pub screen_position: Vec2,
    /// Timestamp of the event in seconds since application start.
    pub timestamp: f64,
}

impl MouseEvent {
    pub fn new(position: Vec2) -> Self {
        Self {
            position,
            delta: Vec2::ZERO,
            button: None,
            click_count: 0,
            modifiers: KeyModifiers::default(),
            buttons_down: MouseButtonMask::empty(),
            screen_position: position,
            timestamp: 0.0,
        }
    }

    pub fn with_button(mut self, button: MouseButton) -> Self {
        self.button = Some(button);
        self
    }

    pub fn with_delta(mut self, delta: Vec2) -> Self {
        self.delta = delta;
        self
    }

    pub fn with_modifiers(mut self, mods: KeyModifiers) -> Self {
        self.modifiers = mods;
        self
    }

    pub fn with_click_count(mut self, count: u32) -> Self {
        self.click_count = count;
        self
    }

    pub fn with_timestamp(mut self, t: f64) -> Self {
        self.timestamp = t;
        self
    }

    pub fn is_left_button(&self) -> bool {
        self.button == Some(MouseButton::Left)
    }

    pub fn is_right_button(&self) -> bool {
        self.button == Some(MouseButton::Right)
    }

    pub fn is_middle_button(&self) -> bool {
        self.button == Some(MouseButton::Middle)
    }

    pub fn is_double_click(&self) -> bool {
        self.click_count == 2
    }

    pub fn is_triple_click(&self) -> bool {
        self.click_count == 3
    }
}

/// Bit mask for simultaneously-held mouse buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MouseButtonMask(u8);

impl MouseButtonMask {
    pub fn empty() -> Self {
        Self(0)
    }

    pub fn set(&mut self, button: MouseButton) {
        self.0 |= Self::bit(button);
    }

    pub fn clear(&mut self, button: MouseButton) {
        self.0 &= !Self::bit(button);
    }

    pub fn contains(&self, button: MouseButton) -> bool {
        self.0 & Self::bit(button) != 0
    }

    pub fn any_pressed(&self) -> bool {
        self.0 != 0
    }

    fn bit(button: MouseButton) -> u8 {
        match button {
            MouseButton::Left => 1,
            MouseButton::Right => 2,
            MouseButton::Middle => 4,
            MouseButton::Back => 8,
            MouseButton::Forward => 16,
        }
    }
}

// =========================================================================
// 2. Key Event
// =========================================================================

/// A keyboard key event.
#[derive(Debug, Clone)]
pub struct KeyEvent {
    /// Virtual key code.
    pub key_code: KeyCode,
    /// Whether the key is being pressed (true) or released (false).
    pub pressed: bool,
    /// Whether this is a key repeat (held down).
    pub is_repeat: bool,
    /// Keyboard modifiers held at the time.
    pub modifiers: KeyModifiers,
    /// Timestamp of the event.
    pub timestamp: f64,
    /// The native scancode (platform-specific).
    pub scan_code: u32,
}

impl KeyEvent {
    pub fn new(key: KeyCode, pressed: bool) -> Self {
        Self {
            key_code: key,
            pressed,
            is_repeat: false,
            modifiers: KeyModifiers::default(),
            timestamp: 0.0,
            scan_code: 0,
        }
    }

    pub fn with_modifiers(mut self, mods: KeyModifiers) -> Self {
        self.modifiers = mods;
        self
    }

    pub fn with_repeat(mut self, repeat: bool) -> Self {
        self.is_repeat = repeat;
        self
    }

    pub fn is_ctrl(&self) -> bool {
        self.modifiers.ctrl
    }

    pub fn is_shift(&self) -> bool {
        self.modifiers.shift
    }

    pub fn is_alt(&self) -> bool {
        self.modifiers.alt
    }

    pub fn is_super(&self) -> bool {
        self.modifiers.meta
    }

    /// Check if this key event matches a specific chord.
    pub fn matches_chord(&self, chord: &InputChord) -> bool {
        self.key_code == chord.key
            && self.modifiers.ctrl == chord.ctrl
            && self.modifiers.shift == chord.shift
            && self.modifiers.alt == chord.alt
            && self.modifiers.meta == chord.meta
    }
}

// =========================================================================
// 3. Scroll Event
// =========================================================================

/// Scroll / mouse-wheel event.
#[derive(Debug, Clone)]
pub struct ScrollEvent {
    /// Horizontal scroll delta (positive = right).
    pub delta_x: f32,
    /// Vertical scroll delta (positive = up / away from user).
    pub delta_y: f32,
    /// True if the scroll is from a high-precision device (trackpad).
    pub is_precise: bool,
    /// Cursor position at the time of the scroll.
    pub position: Vec2,
    /// Keyboard modifiers.
    pub modifiers: KeyModifiers,
    /// Timestamp.
    pub timestamp: f64,
    /// Phase of the scroll gesture (for trackpads).
    pub phase: ScrollPhase,
}

/// Trackpad scroll phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollPhase {
    /// No specific phase (regular mouse wheel).
    None,
    /// Scroll gesture started (finger down).
    Started,
    /// Scroll gesture in progress.
    Changed,
    /// Scroll gesture ended (finger lifted).
    Ended,
    /// Scroll gesture cancelled.
    Cancelled,
    /// Momentum phase (inertial scrolling after finger lift).
    MayBegin,
}

impl Default for ScrollPhase {
    fn default() -> Self {
        ScrollPhase::None
    }
}

impl ScrollEvent {
    pub fn new(delta_x: f32, delta_y: f32, position: Vec2) -> Self {
        Self {
            delta_x,
            delta_y,
            is_precise: false,
            position,
            modifiers: KeyModifiers::default(),
            timestamp: 0.0,
            phase: ScrollPhase::None,
        }
    }

    pub fn with_precise(mut self, precise: bool) -> Self {
        self.is_precise = precise;
        self
    }

    pub fn with_modifiers(mut self, mods: KeyModifiers) -> Self {
        self.modifiers = mods;
        self
    }

    pub fn with_phase(mut self, phase: ScrollPhase) -> Self {
        self.phase = phase;
        self
    }

    /// Delta as a Vec2.
    pub fn delta(&self) -> Vec2 {
        Vec2::new(self.delta_x, self.delta_y)
    }

    /// Normalized delta (for trackpad, already pixel-accurate; for mice, multiply by line height).
    pub fn effective_delta(&self, line_height: f32) -> Vec2 {
        if self.is_precise {
            Vec2::new(self.delta_x, self.delta_y)
        } else {
            Vec2::new(self.delta_x * line_height * 3.0, self.delta_y * line_height * 3.0)
        }
    }
}

// =========================================================================
// 4. Text Input Event
// =========================================================================

/// Text input event (character input and IME composition).
#[derive(Debug, Clone)]
pub struct TextInputEvent {
    /// The character entered (for committed text).
    pub character: Option<char>,
    /// IME composition string (preedit text).
    pub composition: Option<String>,
    /// Cursor position within the composition string.
    pub composition_cursor: Option<usize>,
    /// Whether the composition is being committed (finalized).
    pub committed: bool,
    /// Timestamp.
    pub timestamp: f64,
}

impl TextInputEvent {
    pub fn character(c: char) -> Self {
        Self {
            character: Some(c),
            composition: None,
            composition_cursor: None,
            committed: true,
            timestamp: 0.0,
        }
    }

    pub fn composition(text: &str, cursor: usize) -> Self {
        Self {
            character: None,
            composition: Some(text.to_string()),
            composition_cursor: Some(cursor),
            committed: false,
            timestamp: 0.0,
        }
    }

    pub fn commit(text: &str) -> Self {
        Self {
            character: None,
            composition: Some(text.to_string()),
            composition_cursor: None,
            committed: true,
            timestamp: 0.0,
        }
    }

    pub fn is_printable_char(&self) -> bool {
        self.character.map_or(false, |c| !c.is_control())
    }
}

// =========================================================================
// 5. Drag/Drop Event
// =========================================================================

/// Drag payload -- typed data that is carried during a drag operation.
#[derive(Debug, Clone)]
pub enum DragPayloadData {
    /// An asset reference (texture, mesh, sound, etc.).
    AssetId(u64),
    /// An entity reference.
    EntityId(u64),
    /// A colour value.
    Color(Color),
    /// A file path.
    FilePath(String),
    /// A plain text string.
    Text(String),
    /// Serialised binary data with a type tag.
    Custom {
        type_tag: String,
        data: Vec<u8>,
    },
    /// Multiple items bundled together.
    Multi(Vec<DragPayloadData>),
}

impl DragPayloadData {
    pub fn type_name(&self) -> &str {
        match self {
            DragPayloadData::AssetId(_) => "AssetId",
            DragPayloadData::EntityId(_) => "EntityId",
            DragPayloadData::Color(_) => "Color",
            DragPayloadData::FilePath(_) => "FilePath",
            DragPayloadData::Text(_) => "Text",
            DragPayloadData::Custom { type_tag, .. } => type_tag.as_str(),
            DragPayloadData::Multi(_) => "Multi",
        }
    }

    pub fn as_asset_id(&self) -> Option<u64> {
        if let DragPayloadData::AssetId(id) = self { Some(*id) } else { None }
    }

    pub fn as_entity_id(&self) -> Option<u64> {
        if let DragPayloadData::EntityId(id) = self { Some(*id) } else { None }
    }

    pub fn as_color(&self) -> Option<Color> {
        if let DragPayloadData::Color(c) = self { Some(*c) } else { None }
    }

    pub fn as_text(&self) -> Option<&str> {
        if let DragPayloadData::Text(s) = self { Some(s.as_str()) } else { None }
    }

    pub fn as_file_path(&self) -> Option<&str> {
        if let DragPayloadData::FilePath(s) = self { Some(s.as_str()) } else { None }
    }
}

/// Visual representation drawn at the cursor during a drag operation.
#[derive(Debug, Clone)]
pub struct DragVisual {
    /// Text label shown next to the cursor.
    pub label: String,
    /// Optional icon texture.
    pub icon: Option<crate::render_commands::TextureId>,
    /// Offset from cursor to draw the visual.
    pub offset: Vec2,
    /// Size of the visual widget.
    pub size: Vec2,
    /// Background colour.
    pub bg_color: Color,
    /// Opacity.
    pub opacity: f32,
}

impl DragVisual {
    pub fn text(label: &str) -> Self {
        Self {
            label: label.to_string(),
            icon: None,
            offset: Vec2::new(8.0, 8.0),
            size: Vec2::new(120.0, 24.0),
            bg_color: Color::from_hex("#333333"),
            opacity: 0.85,
        }
    }

    pub fn with_icon(mut self, icon: crate::render_commands::TextureId) -> Self {
        self.icon = Some(icon);
        self
    }
}

/// Result of a drop target validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropValidationResult {
    /// The drop is accepted.
    Accept,
    /// The drop is rejected.
    Reject,
    /// The drop is accepted as a copy operation.
    AcceptCopy,
    /// The drop is accepted as a move operation.
    AcceptMove,
    /// The drop is accepted as a link/reference.
    AcceptLink,
}

/// Configuration for a drop target.
#[derive(Debug, Clone)]
pub struct DropTargetConfig {
    /// Widget id of the drop target.
    pub widget_id: UIId,
    /// Which payload types this target accepts.
    pub accepted_types: Vec<String>,
    /// Highlight colour when a valid drag is over this target.
    pub highlight_color: Color,
    /// Whether to show the highlight.
    pub show_highlight: bool,
}

impl DropTargetConfig {
    pub fn new(widget_id: UIId) -> Self {
        Self {
            widget_id,
            accepted_types: Vec::new(),
            highlight_color: Color::from_hex("#2266CC40"),
            show_highlight: true,
        }
    }

    pub fn with_accepted_type(mut self, t: &str) -> Self {
        self.accepted_types.push(t.to_string());
        self
    }

    pub fn accepts(&self, payload: &DragPayloadData) -> bool {
        if self.accepted_types.is_empty() {
            return true; // Accept all.
        }
        self.accepted_types
            .iter()
            .any(|t| t == payload.type_name())
    }
}

/// Full drag-drop event passed to widgets.
#[derive(Debug, Clone)]
pub struct SlateDragDropEvent {
    /// The drag payload.
    pub payload: DragPayloadData,
    /// Source widget id.
    pub source_widget: UIId,
    /// Current cursor position.
    pub position: Vec2,
    /// Cursor delta since last frame.
    pub delta: Vec2,
    /// Keyboard modifiers.
    pub modifiers: KeyModifiers,
    /// Phase of the drag operation.
    pub phase: SlateDragPhase,
}

/// Phase of a drag operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlateDragPhase {
    /// Drag operation just started.
    Started,
    /// Drag is in progress (cursor is moving with a payload).
    Dragging,
    /// Drag entered a potential drop target.
    Entered,
    /// Drag is over a drop target (has been there for a while).
    Over,
    /// Drag left a drop target.
    Left,
    /// The item was dropped.
    Dropped,
    /// Drag was cancelled (e.g. Escape key).
    Cancelled,
}

/// Drag-and-drop manager that tracks the current drag session.
#[derive(Debug)]
pub struct SlateDragDropManager {
    /// Currently active drag session (if any).
    pub active_drag: Option<ActiveDrag>,
    /// Registered drop targets.
    pub drop_targets: Vec<DropTargetConfig>,
    /// Minimum pixel distance before a drag starts.
    pub drag_threshold: f32,
    /// Last mouse-down position (used for threshold check).
    pending_drag_start: Option<Vec2>,
    /// Pending payload to attach when threshold is exceeded.
    pending_payload: Option<DragPayloadData>,
    /// Pending source widget.
    pending_source: UIId,
    /// Pending visual.
    pending_visual: Option<DragVisual>,
}

/// State of an active drag operation.
#[derive(Debug, Clone)]
pub struct ActiveDrag {
    pub payload: DragPayloadData,
    pub source_widget: UIId,
    pub visual: DragVisual,
    pub current_position: Vec2,
    pub start_position: Vec2,
    pub current_target: Option<UIId>,
    pub validation_result: DropValidationResult,
}

impl SlateDragDropManager {
    pub fn new() -> Self {
        Self {
            active_drag: None,
            drop_targets: Vec::new(),
            drag_threshold: 5.0,
            pending_drag_start: None,
            pending_payload: None,
            pending_source: UIId::INVALID,
            pending_visual: None,
        }
    }

    /// Register a potential drag start. The drag won't actually begin until
    /// the cursor moves beyond `drag_threshold` pixels.
    pub fn begin_potential_drag(
        &mut self,
        position: Vec2,
        source: UIId,
        payload: DragPayloadData,
        visual: DragVisual,
    ) {
        self.pending_drag_start = Some(position);
        self.pending_payload = Some(payload);
        self.pending_source = source;
        self.pending_visual = Some(visual);
    }

    /// Update the manager each frame with the current mouse position.
    /// Returns the drag phase if a drag event occurred.
    pub fn update(&mut self, position: Vec2) -> Option<SlateDragPhase> {
        // Check if pending drag should start.
        if let Some(start) = self.pending_drag_start {
            let dist = (position - start).length();
            if dist >= self.drag_threshold {
                // Promote to active drag.
                let payload = self.pending_payload.take().unwrap();
                let visual = self.pending_visual.take().unwrap_or(DragVisual::text(""));
                self.active_drag = Some(ActiveDrag {
                    payload,
                    source_widget: self.pending_source,
                    visual,
                    current_position: position,
                    start_position: start,
                    current_target: None,
                    validation_result: DropValidationResult::Reject,
                });
                self.pending_drag_start = None;
                return Some(SlateDragPhase::Started);
            }
        }

        // Update active drag.
        if let Some(ref mut drag) = self.active_drag {
            drag.current_position = position;

            // Check drop targets.
            let mut found_target = false;
            for target in &self.drop_targets {
                // In a real implementation, we'd check the widget's bounding rect.
                // Here we just track the target id for the framework to use.
                if target.accepts(&drag.payload) {
                    // The actual hit test is done by the application.
                    found_target = true;
                }
            }

            return Some(SlateDragPhase::Dragging);
        }

        None
    }

    /// Complete the drag with a drop on the given target.
    pub fn drop_on_target(&mut self, target: UIId) -> Option<SlateDragDropEvent> {
        if let Some(drag) = self.active_drag.take() {
            Some(SlateDragDropEvent {
                payload: drag.payload,
                source_widget: drag.source_widget,
                position: drag.current_position,
                delta: Vec2::ZERO,
                modifiers: KeyModifiers::default(),
                phase: SlateDragPhase::Dropped,
            })
        } else {
            None
        }
    }

    /// Cancel the current drag operation.
    pub fn cancel_drag(&mut self) {
        self.active_drag = None;
        self.pending_drag_start = None;
        self.pending_payload = None;
        self.pending_visual = None;
    }

    /// Register a drop target.
    pub fn register_target(&mut self, config: DropTargetConfig) {
        self.drop_targets.push(config);
    }

    /// Unregister a drop target.
    pub fn unregister_target(&mut self, widget_id: UIId) {
        self.drop_targets.retain(|t| t.widget_id != widget_id);
    }

    /// Whether a drag is currently in progress.
    pub fn is_dragging(&self) -> bool {
        self.active_drag.is_some()
    }

    /// Get the current drag payload type (if dragging).
    pub fn current_payload_type(&self) -> Option<&str> {
        self.active_drag.as_ref().map(|d| d.payload.type_name())
    }
}

impl Default for SlateDragDropManager {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// 6. EventReply (re-export chain from slate_widgets for input layer)
// =========================================================================

// EventReply is defined in slate_widgets.rs; we reference it from there.

// =========================================================================
// 7. Focus System
// =========================================================================

/// Reason why focus was acquired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusCause {
    /// Focus set by a mouse click.
    Mouse,
    /// Focus set by keyboard navigation (Tab).
    Keyboard,
    /// Focus set by explicit navigation (arrow keys in a focus group).
    Navigation,
    /// Focus set programmatically.
    Programmatic,
    /// Focus cleared.
    Cleared,
}

impl Default for FocusCause {
    fn default() -> Self {
        FocusCause::Cleared
    }
}

/// Entry in the focus stack.
#[derive(Debug, Clone)]
pub struct FocusEntry {
    pub widget_id: UIId,
    pub cause: FocusCause,
    pub timestamp: f64,
}

/// Focus system that manages the keyboard focus stack and tab order.
#[derive(Debug)]
pub struct FocusSystem {
    /// The currently focused widget.
    pub focused: Option<FocusEntry>,
    /// Focus history stack (for restoring focus after popups).
    pub focus_stack: Vec<FocusEntry>,
    /// Tab navigation order (widget ids in order).
    pub tab_order: Vec<UIId>,
    /// Current index in the tab order.
    pub tab_index: Option<usize>,
    /// Whether focus cycling wraps around at the ends.
    pub wrap_tab_navigation: bool,
    /// Focus scope groups (e.g. for modal dialogs).
    pub scope_stack: Vec<FocusScope>,
}

/// A focus scope restricts tab navigation to a subset of widgets.
#[derive(Debug, Clone)]
pub struct FocusScope {
    pub name: String,
    pub widget_ids: Vec<UIId>,
    pub owner: UIId,
}

impl FocusScope {
    pub fn new(name: &str, owner: UIId) -> Self {
        Self {
            name: name.to_string(),
            widget_ids: Vec::new(),
            owner,
        }
    }

    pub fn with_widgets(mut self, ids: Vec<UIId>) -> Self {
        self.widget_ids = ids;
        self
    }
}

impl FocusSystem {
    pub fn new() -> Self {
        Self {
            focused: None,
            focus_stack: Vec::new(),
            tab_order: Vec::new(),
            tab_index: None,
            wrap_tab_navigation: true,
            scope_stack: Vec::new(),
        }
    }

    /// Set focus to a specific widget.
    pub fn set_focus(&mut self, widget_id: UIId, cause: FocusCause, timestamp: f64) -> Option<UIId> {
        let previous = self.focused.as_ref().map(|f| f.widget_id);
        let entry = FocusEntry {
            widget_id,
            cause,
            timestamp,
        };
        self.focused = Some(entry.clone());
        self.focus_stack.push(entry);

        // Update tab index.
        self.tab_index = self.tab_order.iter().position(|&id| id == widget_id);

        previous
    }

    /// Clear focus.
    pub fn clear_focus(&mut self) -> Option<UIId> {
        let previous = self.focused.as_ref().map(|f| f.widget_id);
        self.focused = None;
        self.tab_index = None;
        previous
    }

    /// Move focus to the next widget in the tab order.
    pub fn focus_next(&mut self, timestamp: f64) -> Option<UIId> {
        let order = self.effective_tab_order();
        if order.is_empty() {
            return None;
        }

        let next_idx = match self.tab_index {
            Some(current) => {
                let next = current + 1;
                if next >= order.len() {
                    if self.wrap_tab_navigation {
                        0
                    } else {
                        return None;
                    }
                } else {
                    next
                }
            }
            None => 0,
        };

        let widget_id = order[next_idx];
        self.set_focus(widget_id, FocusCause::Keyboard, timestamp);
        Some(widget_id)
    }

    /// Move focus to the previous widget in the tab order.
    pub fn focus_previous(&mut self, timestamp: f64) -> Option<UIId> {
        let order = self.effective_tab_order();
        if order.is_empty() {
            return None;
        }

        let prev_idx = match self.tab_index {
            Some(current) => {
                if current == 0 {
                    if self.wrap_tab_navigation {
                        order.len() - 1
                    } else {
                        return None;
                    }
                } else {
                    current - 1
                }
            }
            None => order.len() - 1,
        };

        let widget_id = order[prev_idx];
        self.set_focus(widget_id, FocusCause::Keyboard, timestamp);
        Some(widget_id)
    }

    /// Get the currently focused widget id.
    pub fn current_focus(&self) -> Option<UIId> {
        self.focused.as_ref().map(|f| f.widget_id)
    }

    /// Get the cause of the current focus.
    pub fn focus_cause(&self) -> FocusCause {
        self.focused
            .as_ref()
            .map(|f| f.cause)
            .unwrap_or(FocusCause::Cleared)
    }

    /// Check if a specific widget has focus.
    pub fn has_focus(&self, widget_id: UIId) -> bool {
        self.focused
            .as_ref()
            .map(|f| f.widget_id == widget_id)
            .unwrap_or(false)
    }

    /// Push a focus scope (e.g. when a modal opens).
    pub fn push_scope(&mut self, scope: FocusScope) {
        // Save current focus.
        self.scope_stack.push(scope);
    }

    /// Pop a focus scope and restore the previous focus.
    pub fn pop_scope(&mut self, timestamp: f64) -> Option<FocusScope> {
        let scope = self.scope_stack.pop()?;

        // Restore focus from the stack.
        if let Some(previous) = self.focus_stack.last() {
            let widget_id = previous.widget_id;
            self.set_focus(widget_id, FocusCause::Programmatic, timestamp);
        }

        Some(scope)
    }

    /// Register a widget in the tab order.
    pub fn register_tab_stop(&mut self, widget_id: UIId) {
        if !self.tab_order.contains(&widget_id) {
            self.tab_order.push(widget_id);
        }
    }

    /// Remove a widget from the tab order.
    pub fn unregister_tab_stop(&mut self, widget_id: UIId) {
        self.tab_order.retain(|&id| id != widget_id);
        if self.has_focus(widget_id) {
            self.clear_focus();
        }
    }

    /// Get the effective tab order (considering scope restrictions).
    fn effective_tab_order(&self) -> Vec<UIId> {
        if let Some(scope) = self.scope_stack.last() {
            if !scope.widget_ids.is_empty() {
                return scope.widget_ids.clone();
            }
        }
        self.tab_order.clone()
    }

    /// Handle a Tab key press. Returns the newly-focused widget id (if any).
    pub fn handle_tab(&mut self, shift: bool, timestamp: f64) -> Option<UIId> {
        if shift {
            self.focus_previous(timestamp)
        } else {
            self.focus_next(timestamp)
        }
    }
}

impl Default for FocusSystem {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// 8. Mouse Capture
// =========================================================================

/// Tracks exclusive mouse capture (one widget receives all mouse events).
#[derive(Debug)]
pub struct MouseCaptureManager {
    /// The widget that currently has mouse capture (if any).
    pub captured_by: Option<UIId>,
    /// The button that initiated the capture.
    pub capture_button: Option<MouseButton>,
    /// Position where capture started.
    pub capture_start: Vec2,
    /// Whether capture should auto-release when the button is released.
    pub auto_release: bool,
}

impl MouseCaptureManager {
    pub fn new() -> Self {
        Self {
            captured_by: None,
            capture_button: None,
            capture_start: Vec2::ZERO,
            auto_release: true,
        }
    }

    /// Capture the mouse for a specific widget.
    pub fn capture(&mut self, widget_id: UIId, button: MouseButton, position: Vec2) {
        self.captured_by = Some(widget_id);
        self.capture_button = Some(button);
        self.capture_start = position;
    }

    /// Release the mouse capture.
    pub fn release(&mut self) -> Option<UIId> {
        let prev = self.captured_by.take();
        self.capture_button = None;
        prev
    }

    /// Check if a widget has capture.
    pub fn has_capture(&self, widget_id: UIId) -> bool {
        self.captured_by == Some(widget_id)
    }

    /// Check if any widget has capture.
    pub fn is_captured(&self) -> bool {
        self.captured_by.is_some()
    }

    /// Get the capturing widget.
    pub fn capturing_widget(&self) -> Option<UIId> {
        self.captured_by
    }
}

impl Default for MouseCaptureManager {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// 9. Cursor Manager
// =========================================================================

/// Standard cursor shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CursorShape {
    Arrow,
    Hand,
    IBeam,
    ResizeH,
    ResizeV,
    ResizeNESW,
    ResizeNWSE,
    Crosshair,
    Move,
    NotAllowed,
    Wait,
    Progress,
    Help,
    Grab,
    Grabbing,
    Hidden,
    Custom(u32),
}

impl Default for CursorShape {
    fn default() -> Self {
        CursorShape::Arrow
    }
}

impl fmt::Display for CursorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            CursorShape::Arrow => "Arrow",
            CursorShape::Hand => "Hand",
            CursorShape::IBeam => "IBeam",
            CursorShape::ResizeH => "ResizeH",
            CursorShape::ResizeV => "ResizeV",
            CursorShape::ResizeNESW => "ResizeNESW",
            CursorShape::ResizeNWSE => "ResizeNWSE",
            CursorShape::Crosshair => "Crosshair",
            CursorShape::Move => "Move",
            CursorShape::NotAllowed => "NotAllowed",
            CursorShape::Wait => "Wait",
            CursorShape::Progress => "Progress",
            CursorShape::Help => "Help",
            CursorShape::Grab => "Grab",
            CursorShape::Grabbing => "Grabbing",
            CursorShape::Hidden => "Hidden",
            CursorShape::Custom(id) => return write!(f, "Custom({})", id),
        };
        write!(f, "{}", name)
    }
}

/// Manages the cursor shape based on which widget the cursor is over.
#[derive(Debug)]
pub struct CursorManager {
    /// Current cursor shape.
    pub current: CursorShape,
    /// Stack of cursor overrides (higher priority on top).
    cursor_stack: Vec<(UIId, CursorShape)>,
    /// Default cursor when nothing is hovered.
    pub default_cursor: CursorShape,
    /// Whether the cursor has changed since last frame (for platform update).
    pub changed: bool,
}

impl CursorManager {
    pub fn new() -> Self {
        Self {
            current: CursorShape::Arrow,
            cursor_stack: Vec::new(),
            default_cursor: CursorShape::Arrow,
            changed: false,
        }
    }

    /// Set the cursor for a specific widget (pushed onto the stack).
    pub fn set_cursor(&mut self, widget_id: UIId, shape: CursorShape) {
        // Remove any existing entry for this widget.
        self.cursor_stack.retain(|(id, _)| *id != widget_id);
        self.cursor_stack.push((widget_id, shape));
        self.update_current();
    }

    /// Clear the cursor override for a specific widget.
    pub fn clear_cursor(&mut self, widget_id: UIId) {
        self.cursor_stack.retain(|(id, _)| *id != widget_id);
        self.update_current();
    }

    /// Clear all cursor overrides.
    pub fn clear_all(&mut self) {
        self.cursor_stack.clear();
        self.update_current();
    }

    fn update_current(&mut self) {
        let new_cursor = self
            .cursor_stack
            .last()
            .map(|(_, shape)| *shape)
            .unwrap_or(self.default_cursor);
        if new_cursor != self.current {
            self.current = new_cursor;
            self.changed = true;
        }
    }

    /// Acknowledge the cursor change (platform code calls this after updating
    /// the OS cursor).
    pub fn acknowledge_change(&mut self) {
        self.changed = false;
    }
}

impl Default for CursorManager {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// 10. Input Chord
// =========================================================================

/// A key + modifiers combination (e.g. Ctrl+S, Ctrl+Shift+Z).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InputChord {
    pub key: KeyCode,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub meta: bool,
}

impl InputChord {
    pub fn new(key: KeyCode) -> Self {
        Self {
            key,
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        }
    }

    pub fn ctrl(mut self) -> Self {
        self.ctrl = true;
        self
    }

    pub fn shift(mut self) -> Self {
        self.shift = true;
        self
    }

    pub fn alt(mut self) -> Self {
        self.alt = true;
        self
    }

    pub fn meta(mut self) -> Self {
        self.meta = true;
        self
    }

    /// Check if the given modifiers and key match this chord.
    pub fn matches(&self, key: KeyCode, modifiers: &KeyModifiers) -> bool {
        self.key == key
            && self.ctrl == modifiers.ctrl
            && self.shift == modifiers.shift
            && self.alt == modifiers.alt
            && self.meta == modifiers.meta
    }

    /// Human-readable label for the chord (e.g. "Ctrl+S").
    pub fn display_string(&self) -> String {
        let mut parts = Vec::new();
        if self.ctrl {
            parts.push("Ctrl");
        }
        if self.shift {
            parts.push("Shift");
        }
        if self.alt {
            parts.push("Alt");
        }
        if self.meta {
            parts.push("Super");
        }
        parts.push(self.key_name());
        parts.join("+")
    }

    fn key_name(&self) -> &str {
        match self.key {
            KeyCode::A => "A", KeyCode::B => "B", KeyCode::C => "C",
            KeyCode::D => "D", KeyCode::E => "E", KeyCode::F => "F",
            KeyCode::G => "G", KeyCode::H => "H", KeyCode::I => "I",
            KeyCode::J => "J", KeyCode::K => "K", KeyCode::L => "L",
            KeyCode::M => "M", KeyCode::N => "N", KeyCode::O => "O",
            KeyCode::P => "P", KeyCode::Q => "Q", KeyCode::R => "R",
            KeyCode::S => "S", KeyCode::T => "T", KeyCode::U => "U",
            KeyCode::V => "V", KeyCode::W => "W", KeyCode::X => "X",
            KeyCode::Y => "Y", KeyCode::Z => "Z",
            KeyCode::Key0 => "0", KeyCode::Key1 => "1", KeyCode::Key2 => "2",
            KeyCode::Key3 => "3", KeyCode::Key4 => "4", KeyCode::Key5 => "5",
            KeyCode::Key6 => "6", KeyCode::Key7 => "7", KeyCode::Key8 => "8",
            KeyCode::Key9 => "9",
            KeyCode::F1 => "F1", KeyCode::F2 => "F2", KeyCode::F3 => "F3",
            KeyCode::F4 => "F4", KeyCode::F5 => "F5", KeyCode::F6 => "F6",
            KeyCode::F7 => "F7", KeyCode::F8 => "F8", KeyCode::F9 => "F9",
            KeyCode::F10 => "F10", KeyCode::F11 => "F11", KeyCode::F12 => "F12",
            KeyCode::Escape => "Esc", KeyCode::Tab => "Tab",
            KeyCode::Space => "Space", KeyCode::Enter => "Enter",
            KeyCode::Backspace => "Backspace", KeyCode::Delete => "Del",
            KeyCode::ArrowUp => "Up", KeyCode::ArrowDown => "Down",
            KeyCode::ArrowLeft => "Left", KeyCode::ArrowRight => "Right",
            KeyCode::Home => "Home", KeyCode::End => "End",
            KeyCode::PageUp => "PgUp", KeyCode::PageDown => "PgDn",
            KeyCode::Insert => "Ins",
            _ => "?",
        }
    }
}

impl fmt::Display for InputChord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_string())
    }
}

// =========================================================================
// 11. Command System
// =========================================================================

/// A UI command that can be bound to a key chord and executed.
#[derive(Debug, Clone)]
pub struct UICommand {
    /// Unique identifier for the command.
    pub name: String,
    /// Human-readable label.
    pub label: String,
    /// Description for tooltips.
    pub description: String,
    /// Keyboard shortcut.
    pub input_chord: Option<InputChord>,
    /// Whether the command is currently enabled.
    pub enabled: bool,
    /// Whether the command is currently checked (for toggle commands).
    pub checked: bool,
    /// Icon texture.
    pub icon: Option<crate::render_commands::TextureId>,
    /// Category for menu organization.
    pub category: String,
}

impl UICommand {
    pub fn new(name: &str, label: &str) -> Self {
        Self {
            name: name.to_string(),
            label: label.to_string(),
            description: String::new(),
            input_chord: None,
            enabled: true,
            checked: false,
            icon: None,
            category: String::new(),
        }
    }

    pub fn with_chord(mut self, chord: InputChord) -> Self {
        self.input_chord = Some(chord);
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_category(mut self, cat: &str) -> Self {
        self.category = cat.to_string();
        self
    }

    pub fn shortcut_text(&self) -> String {
        self.input_chord
            .as_ref()
            .map(|c| c.display_string())
            .unwrap_or_default()
    }
}

/// Tracks which commands have been executed since last check.
#[derive(Debug, Clone)]
pub struct CommandExecution {
    pub command_name: String,
    pub timestamp: f64,
}

/// Registry of commands that can be bound to key chords and menus.
#[derive(Debug)]
pub struct CommandList {
    /// All registered commands.
    pub commands: Vec<UICommand>,
    /// Index by name for fast lookup.
    name_index: HashMap<String, usize>,
    /// Pending executions.
    pub pending_executions: Vec<CommandExecution>,
}

impl CommandList {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            name_index: HashMap::new(),
            pending_executions: Vec::new(),
        }
    }

    /// Create a new command list with standard built-in commands.
    pub fn with_defaults() -> Self {
        let mut list = Self::new();

        list.register(
            UICommand::new("undo", "Undo")
                .with_chord(InputChord::new(KeyCode::Z).ctrl())
                .with_description("Undo the last action")
                .with_category("Edit"),
        );
        list.register(
            UICommand::new("redo", "Redo")
                .with_chord(InputChord::new(KeyCode::Z).ctrl().shift())
                .with_description("Redo the last undone action")
                .with_category("Edit"),
        );
        list.register(
            UICommand::new("cut", "Cut")
                .with_chord(InputChord::new(KeyCode::X).ctrl())
                .with_description("Cut the selection to clipboard")
                .with_category("Edit"),
        );
        list.register(
            UICommand::new("copy", "Copy")
                .with_chord(InputChord::new(KeyCode::C).ctrl())
                .with_description("Copy the selection to clipboard")
                .with_category("Edit"),
        );
        list.register(
            UICommand::new("paste", "Paste")
                .with_chord(InputChord::new(KeyCode::V).ctrl())
                .with_description("Paste from clipboard")
                .with_category("Edit"),
        );
        list.register(
            UICommand::new("delete", "Delete")
                .with_chord(InputChord::new(KeyCode::Delete))
                .with_description("Delete the selection")
                .with_category("Edit"),
        );
        list.register(
            UICommand::new("select_all", "Select All")
                .with_chord(InputChord::new(KeyCode::A).ctrl())
                .with_description("Select all items")
                .with_category("Edit"),
        );
        list.register(
            UICommand::new("save", "Save")
                .with_chord(InputChord::new(KeyCode::S).ctrl())
                .with_description("Save the current file")
                .with_category("File"),
        );
        list.register(
            UICommand::new("save_all", "Save All")
                .with_chord(InputChord::new(KeyCode::S).ctrl().shift())
                .with_description("Save all open files")
                .with_category("File"),
        );
        list.register(
            UICommand::new("find", "Find")
                .with_chord(InputChord::new(KeyCode::F).ctrl())
                .with_description("Open the find dialog")
                .with_category("Edit"),
        );
        list.register(
            UICommand::new("find_replace", "Find and Replace")
                .with_chord(InputChord::new(KeyCode::H).ctrl())
                .with_description("Open the find and replace dialog")
                .with_category("Edit"),
        );

        list
    }

    /// Register a new command.
    pub fn register(&mut self, command: UICommand) {
        let name = command.name.clone();
        let idx = self.commands.len();
        self.commands.push(command);
        self.name_index.insert(name, idx);
    }

    /// Find a command by name.
    pub fn find(&self, name: &str) -> Option<&UICommand> {
        self.name_index
            .get(name)
            .and_then(|&idx| self.commands.get(idx))
    }

    /// Find a command by name (mutable).
    pub fn find_mut(&mut self, name: &str) -> Option<&mut UICommand> {
        self.name_index
            .get(name)
            .copied()
            .and_then(move |idx| self.commands.get_mut(idx))
    }

    /// Execute a command by name.
    pub fn execute(&mut self, name: &str, timestamp: f64) -> bool {
        if let Some(&idx) = self.name_index.get(name) {
            if self.commands[idx].enabled {
                self.pending_executions.push(CommandExecution {
                    command_name: name.to_string(),
                    timestamp,
                });
                return true;
            }
        }
        false
    }

    /// Check if a key event matches any command and execute it.
    pub fn try_execute_key(
        &mut self,
        key: KeyCode,
        modifiers: &KeyModifiers,
        timestamp: f64,
    ) -> Option<String> {
        for cmd in &self.commands {
            if cmd.enabled {
                if let Some(ref chord) = cmd.input_chord {
                    if chord.matches(key, modifiers) {
                        let name = cmd.name.clone();
                        self.pending_executions.push(CommandExecution {
                            command_name: name.clone(),
                            timestamp,
                        });
                        return Some(name);
                    }
                }
            }
        }
        None
    }

    /// Take all pending command executions.
    pub fn take_executions(&mut self) -> Vec<CommandExecution> {
        std::mem::take(&mut self.pending_executions)
    }

    /// Get all commands in a specific category.
    pub fn commands_in_category(&self, category: &str) -> Vec<&UICommand> {
        self.commands
            .iter()
            .filter(|c| c.category == category)
            .collect()
    }

    /// Set the enabled state of a command.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) {
        if let Some(cmd) = self.find_mut(name) {
            cmd.enabled = enabled;
        }
    }

    /// Set the checked state of a command.
    pub fn set_checked(&mut self, name: &str, checked: bool) {
        if let Some(cmd) = self.find_mut(name) {
            cmd.checked = checked;
        }
    }

    /// Rebind a command to a different chord.
    pub fn rebind(&mut self, name: &str, chord: InputChord) {
        if let Some(cmd) = self.find_mut(name) {
            cmd.input_chord = Some(chord);
        }
    }

    /// Remove a key binding from a command.
    pub fn unbind(&mut self, name: &str) {
        if let Some(cmd) = self.find_mut(name) {
            cmd.input_chord = None;
        }
    }
}

impl Default for CommandList {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// 12. High-Precision Mouse Mode
// =========================================================================

/// High-precision (infinite) mouse mode for drag operations.
///
/// When active, the cursor wraps at screen edges, providing unlimited
/// drag distance. This is used for value sliders, rotation knobs, etc.
#[derive(Debug)]
pub struct HighPrecisionMouseMode {
    /// Whether high-precision mode is currently active.
    pub active: bool,
    /// The widget that activated this mode.
    pub owner: UIId,
    /// Accumulated delta since mode was activated.
    pub accumulated_delta: Vec2,
    /// Position where the mode was activated.
    pub activation_position: Vec2,
    /// Screen bounds for cursor wrapping.
    pub screen_bounds: Rect,
    /// Whether to hide the cursor in this mode.
    pub hide_cursor: bool,
    /// Sensitivity multiplier.
    pub sensitivity: f32,
    /// Current virtual position (accumulated).
    pub virtual_position: Vec2,
}

impl HighPrecisionMouseMode {
    pub fn new() -> Self {
        Self {
            active: false,
            owner: UIId::INVALID,
            accumulated_delta: Vec2::ZERO,
            activation_position: Vec2::ZERO,
            screen_bounds: Rect::new(Vec2::ZERO, Vec2::new(1920.0, 1080.0)),
            hide_cursor: true,
            sensitivity: 1.0,
            virtual_position: Vec2::ZERO,
        }
    }

    /// Activate high-precision mode.
    pub fn activate(&mut self, owner: UIId, position: Vec2, screen_bounds: Rect) {
        self.active = true;
        self.owner = owner;
        self.accumulated_delta = Vec2::ZERO;
        self.activation_position = position;
        self.virtual_position = position;
        self.screen_bounds = screen_bounds;
    }

    /// Deactivate high-precision mode.
    pub fn deactivate(&mut self) -> Vec2 {
        self.active = false;
        self.owner = UIId::INVALID;
        let result = self.accumulated_delta;
        self.accumulated_delta = Vec2::ZERO;
        result
    }

    /// Process a mouse delta. Returns the adjusted delta.
    pub fn process_delta(&mut self, delta: Vec2) -> Vec2 {
        if !self.active {
            return delta;
        }

        let scaled = delta * self.sensitivity;
        self.accumulated_delta += scaled;
        self.virtual_position += scaled;

        // Wrap the virtual position at screen edges.
        let margin = 10.0;
        if self.virtual_position.x < self.screen_bounds.min.x + margin {
            self.virtual_position.x = self.screen_bounds.max.x - margin;
        } else if self.virtual_position.x > self.screen_bounds.max.x - margin {
            self.virtual_position.x = self.screen_bounds.min.x + margin;
        }
        if self.virtual_position.y < self.screen_bounds.min.y + margin {
            self.virtual_position.y = self.screen_bounds.max.y - margin;
        } else if self.virtual_position.y > self.screen_bounds.max.y - margin {
            self.virtual_position.y = self.screen_bounds.min.y + margin;
        }

        scaled
    }

    /// Get the total accumulated movement since activation.
    pub fn total_delta(&self) -> Vec2 {
        self.accumulated_delta
    }
}

impl Default for HighPrecisionMouseMode {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// 13. Hit Testing
// =========================================================================

/// Result of a hit test on a specific widget.
#[derive(Debug, Clone)]
pub struct HitTestResult {
    /// Widget id that was hit.
    pub widget_id: UIId,
    /// The position relative to the widget's origin.
    pub local_position: Vec2,
    /// The position in screen space.
    pub screen_position: Vec2,
    /// Z-order of the hit widget.
    pub z_order: i32,
    /// Whether the hit is in a transparent region (widget may opt out).
    pub is_transparent: bool,
}

/// Per-widget hit test configuration.
#[derive(Debug, Clone)]
pub struct HitTestConfig {
    /// Widget id.
    pub widget_id: UIId,
    /// Bounding rectangle in screen space.
    pub bounds: Rect,
    /// Z-order (higher = on top).
    pub z_order: i32,
    /// Whether this widget is visible for hit testing.
    pub is_visible: bool,
    /// Whether this widget accepts mouse events.
    pub is_interactive: bool,
    /// Optional transparent regions within the bounds.
    pub transparent_regions: Vec<Rect>,
    /// Optional custom hit-test shape (circle: center + radius).
    pub custom_shape: Option<HitTestShape>,
}

/// Custom hit-test shapes.
#[derive(Debug, Clone)]
pub enum HitTestShape {
    Circle { center: Vec2, radius: f32 },
    RoundedRect { rect: Rect, corner_radius: f32 },
    Polygon { points: Vec<Vec2> },
}

impl HitTestConfig {
    pub fn new(widget_id: UIId, bounds: Rect) -> Self {
        Self {
            widget_id,
            bounds,
            z_order: 0,
            is_visible: true,
            is_interactive: true,
            transparent_regions: Vec::new(),
            custom_shape: None,
        }
    }

    pub fn with_z_order(mut self, z: i32) -> Self {
        self.z_order = z;
        self
    }

    pub fn with_transparent_region(mut self, region: Rect) -> Self {
        self.transparent_regions.push(region);
        self
    }

    pub fn with_circle_shape(mut self, center: Vec2, radius: f32) -> Self {
        self.custom_shape = Some(HitTestShape::Circle { center, radius });
        self
    }

    /// Test if a point is within this widget's hit area.
    pub fn test_point(&self, point: Vec2) -> bool {
        if !self.is_visible || !self.is_interactive {
            return false;
        }

        // Custom shape test.
        if let Some(ref shape) = self.custom_shape {
            return match shape {
                HitTestShape::Circle { center, radius } => {
                    (point - *center).length() <= *radius
                }
                HitTestShape::RoundedRect { rect, corner_radius } => {
                    if !rect.contains(point) {
                        return false;
                    }
                    // Simplified: just check bounds for now.
                    true
                }
                HitTestShape::Polygon { points } => {
                    // Point-in-polygon test (ray casting).
                    if points.len() < 3 {
                        return false;
                    }
                    let mut inside = false;
                    let n = points.len();
                    let mut j = n - 1;
                    for i in 0..n {
                        let pi = points[i];
                        let pj = points[j];
                        if ((pi.y > point.y) != (pj.y > point.y))
                            && (point.x
                                < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)
                        {
                            inside = !inside;
                        }
                        j = i;
                    }
                    inside
                }
            };
        }

        // Bounds test.
        if !self.bounds.contains(point) {
            return false;
        }

        // Check transparent regions.
        for region in &self.transparent_regions {
            if region.contains(point) {
                return false;
            }
        }

        true
    }
}

/// Hit-test manager that tests all registered widgets.
#[derive(Debug)]
pub struct HitTestManager {
    /// All registered hit-testable widgets.
    pub configs: Vec<HitTestConfig>,
}

impl HitTestManager {
    pub fn new() -> Self {
        Self {
            configs: Vec::new(),
        }
    }

    /// Register a widget for hit testing.
    pub fn register(&mut self, config: HitTestConfig) {
        // Replace existing config for the same widget.
        self.configs.retain(|c| c.widget_id != config.widget_id);
        self.configs.push(config);
    }

    /// Unregister a widget.
    pub fn unregister(&mut self, widget_id: UIId) {
        self.configs.retain(|c| c.widget_id != widget_id);
    }

    /// Clear all registrations.
    pub fn clear(&mut self) {
        self.configs.clear();
    }

    /// Perform a hit test at a screen-space position.
    /// Returns the topmost (highest z-order) widget that contains the point.
    pub fn hit_test(&self, position: Vec2) -> Option<HitTestResult> {
        let mut best: Option<(i32, usize)> = None;

        for (i, config) in self.configs.iter().enumerate() {
            if config.test_point(position) {
                match best {
                    Some((best_z, _)) if config.z_order > best_z => {
                        best = Some((config.z_order, i));
                    }
                    None => {
                        best = Some((config.z_order, i));
                    }
                    _ => {}
                }
            }
        }

        best.map(|(_, idx)| {
            let config = &self.configs[idx];
            HitTestResult {
                widget_id: config.widget_id,
                local_position: position - config.bounds.min,
                screen_position: position,
                z_order: config.z_order,
                is_transparent: false,
            }
        })
    }

    /// Test all widgets at a position, returning all hits sorted by z-order
    /// (highest first).
    pub fn hit_test_all(&self, position: Vec2) -> Vec<HitTestResult> {
        let mut results: Vec<HitTestResult> = self
            .configs
            .iter()
            .filter(|c| c.test_point(position))
            .map(|config| HitTestResult {
                widget_id: config.widget_id,
                local_position: position - config.bounds.min,
                screen_position: position,
                z_order: config.z_order,
                is_transparent: false,
            })
            .collect();

        results.sort_by(|a, b| b.z_order.cmp(&a.z_order));
        results
    }

    /// Update the bounds for a widget.
    pub fn update_bounds(&mut self, widget_id: UIId, bounds: Rect) {
        if let Some(config) = self.configs.iter_mut().find(|c| c.widget_id == widget_id) {
            config.bounds = bounds;
        }
    }

    /// Update z-order for a widget.
    pub fn update_z_order(&mut self, widget_id: UIId, z_order: i32) {
        if let Some(config) = self.configs.iter_mut().find(|c| c.widget_id == widget_id) {
            config.z_order = z_order;
        }
    }
}

impl Default for HitTestManager {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// 14. Input Processor (ties everything together)
// =========================================================================

/// The central input processor that owns all input subsystems and routes
/// events to the appropriate systems.
#[derive(Debug)]
pub struct SlateInputProcessor {
    pub focus: FocusSystem,
    pub mouse_capture: MouseCaptureManager,
    pub cursor: CursorManager,
    pub commands: CommandList,
    pub drag_drop: SlateDragDropManager,
    pub high_precision: HighPrecisionMouseMode,
    pub hit_test: HitTestManager,
    /// Current mouse position.
    pub mouse_position: Vec2,
    /// Previous mouse position.
    pub prev_mouse_position: Vec2,
    /// Currently pressed mouse buttons.
    pub buttons_down: MouseButtonMask,
    /// Current keyboard modifiers.
    pub modifiers: KeyModifiers,
    /// Time of last click (for double-click detection).
    pub last_click_time: f64,
    /// Position of last click.
    pub last_click_position: Vec2,
    /// Button of last click.
    pub last_click_button: Option<MouseButton>,
    /// Current click count (reset after timeout).
    pub click_count: u32,
    /// Double-click time threshold in seconds.
    pub double_click_time: f64,
    /// Double-click distance threshold in pixels.
    pub double_click_distance: f32,
    /// Current time.
    pub current_time: f64,
}

impl SlateInputProcessor {
    pub fn new() -> Self {
        Self {
            focus: FocusSystem::new(),
            mouse_capture: MouseCaptureManager::new(),
            cursor: CursorManager::new(),
            commands: CommandList::with_defaults(),
            drag_drop: SlateDragDropManager::new(),
            high_precision: HighPrecisionMouseMode::new(),
            hit_test: HitTestManager::new(),
            mouse_position: Vec2::ZERO,
            prev_mouse_position: Vec2::ZERO,
            buttons_down: MouseButtonMask::empty(),
            modifiers: KeyModifiers::default(),
            last_click_time: 0.0,
            last_click_position: Vec2::ZERO,
            last_click_button: None,
            click_count: 0,
            double_click_time: 0.5,
            double_click_distance: 5.0,
            current_time: 0.0,
        }
    }

    /// Begin a new input frame.
    pub fn begin_frame(&mut self, time: f64) {
        self.current_time = time;
        self.prev_mouse_position = self.mouse_position;
        self.cursor.changed = false;
    }

    /// Process a mouse move.
    pub fn on_mouse_move(&mut self, position: Vec2) -> Option<UIId> {
        self.mouse_position = position;
        let delta = position - self.prev_mouse_position;

        // High-precision mode.
        if self.high_precision.active {
            self.high_precision.process_delta(delta);
        }

        // Update drag/drop.
        self.drag_drop.update(position);

        // Hit test.
        if let Some(hit) = self.hit_test.hit_test(position) {
            return Some(hit.widget_id);
        }
        None
    }

    /// Process a mouse button press.
    pub fn on_mouse_down(&mut self, button: MouseButton, position: Vec2) -> MouseEvent {
        self.mouse_position = position;
        self.buttons_down.set(button);

        // Click count detection.
        let time_since_last = self.current_time - self.last_click_time;
        let dist = (position - self.last_click_position).length();

        if time_since_last < self.double_click_time
            && dist < self.double_click_distance
            && self.last_click_button == Some(button)
        {
            self.click_count += 1;
        } else {
            self.click_count = 1;
        }

        self.last_click_time = self.current_time;
        self.last_click_position = position;
        self.last_click_button = Some(button);

        MouseEvent {
            position,
            delta: Vec2::ZERO,
            button: Some(button),
            click_count: self.click_count,
            modifiers: self.modifiers,
            buttons_down: self.buttons_down,
            screen_position: position,
            timestamp: self.current_time,
        }
    }

    /// Process a mouse button release.
    pub fn on_mouse_up(&mut self, button: MouseButton, position: Vec2) -> MouseEvent {
        self.mouse_position = position;
        self.buttons_down.clear(button);

        MouseEvent {
            position,
            delta: Vec2::ZERO,
            button: Some(button),
            click_count: 0,
            modifiers: self.modifiers,
            buttons_down: self.buttons_down,
            screen_position: position,
            timestamp: self.current_time,
        }
    }

    /// Process a key event.
    pub fn on_key(&mut self, key: KeyCode, pressed: bool, is_repeat: bool) -> KeyEvent {
        // Update modifier state.
        match key {
            KeyCode::Shift => self.modifiers.shift = pressed,
            KeyCode::Control => self.modifiers.ctrl = pressed,
            KeyCode::Alt => self.modifiers.alt = pressed,
            KeyCode::Meta => self.modifiers.meta = pressed,
            _ => {}
        }

        let key_event = KeyEvent {
            key_code: key,
            pressed,
            is_repeat,
            modifiers: self.modifiers,
            timestamp: self.current_time,
            scan_code: 0,
        };

        // Try command execution.
        if pressed && !is_repeat {
            self.commands
                .try_execute_key(key, &self.modifiers, self.current_time);
        }

        // Tab navigation.
        if pressed && key == KeyCode::Tab {
            self.focus.handle_tab(self.modifiers.shift, self.current_time);
        }

        key_event
    }

    /// Process a scroll event.
    pub fn on_scroll(&mut self, delta_x: f32, delta_y: f32, position: Vec2) -> ScrollEvent {
        ScrollEvent::new(delta_x, delta_y, position)
            .with_modifiers(self.modifiers)
    }

    /// Process a text input event.
    pub fn on_text_input(&mut self, character: char) -> TextInputEvent {
        TextInputEvent::character(character)
    }

    /// Get the widget currently under the cursor.
    pub fn hovered_widget(&self) -> Option<UIId> {
        self.hit_test
            .hit_test(self.mouse_position)
            .map(|h| h.widget_id)
    }

    /// Whether any button is currently held.
    pub fn any_button_down(&self) -> bool {
        self.buttons_down.any_pressed()
    }

    /// Mouse delta since last frame.
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_position - self.prev_mouse_position
    }
}

impl Default for SlateInputProcessor {
    fn default() -> Self {
        Self::new()
    }
}
