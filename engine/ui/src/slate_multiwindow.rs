//! Slate Multi-Window Management
//!
//! Provides a comprehensive multi-window management system for the Genovo UI
//! framework, supporting the main editor window, floating tool windows, popup
//! menus/tooltips, modal windows, and window layout persistence.
//!
//! # Architecture
//!
//! ```text
//!  MultiWindowManager ──> FloatingWindow ──> WindowTitleBar
//!       │                      │                  │
//!  WindowFocusManager     PopupWindow        WindowControls
//!       │                      │
//!  WindowLayout           ModalWindow
//!       │
//!  MonitorInfo
//! ```
//!
//! # Window Types
//!
//! - **Main Window**: The primary editor window. Always present.
//! - **Floating Window**: Torn-off dock tabs or tool windows. Can be moved,
//!   resized, minimized, and maximized.
//! - **Popup Window**: Menus, tooltips, dropdowns. Auto-close on click-outside.
//! - **Modal Window**: Blocks input to all other windows.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec2;

use crate::core::UIId;
use crate::render_commands::Color;

// ---------------------------------------------------------------------------
// WindowId
// ---------------------------------------------------------------------------

/// Unique identifier for a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(pub u64);

impl WindowId {
    /// The main window ID.
    pub const MAIN: Self = Self(1);
    /// Invalid/null window ID.
    pub const INVALID: Self = Self(0);

    /// Returns true if this is the main window.
    pub fn is_main(&self) -> bool {
        *self == Self::MAIN
    }

    /// Returns true if this is a valid window ID.
    pub fn is_valid(&self) -> bool {
        self.0 != 0
    }
}

impl fmt::Display for WindowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window({})", self.0)
    }
}

static NEXT_WINDOW_ID: AtomicU64 = AtomicU64::new(100);

fn next_window_id() -> WindowId {
    WindowId(NEXT_WINDOW_ID.fetch_add(1, Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// WindowState
// ---------------------------------------------------------------------------

/// The visual state of a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowState {
    /// Normal (restored) state.
    Normal,
    /// Minimized to taskbar.
    Minimized,
    /// Maximized (fills the screen/monitor).
    Maximized,
    /// Fullscreen (borderless).
    Fullscreen,
    /// Snapped to left half of screen.
    SnappedLeft,
    /// Snapped to right half of screen.
    SnappedRight,
    /// Snapped to top half of screen.
    SnappedTop,
    /// Snapped to bottom half of screen.
    SnappedBottom,
}

impl WindowState {
    /// Returns true if the window is visible (not minimized).
    pub fn is_visible(&self) -> bool {
        !matches!(self, Self::Minimized)
    }

    /// Returns true if the window can be resized in this state.
    pub fn is_resizable(&self) -> bool {
        matches!(self, Self::Normal)
    }

    /// Returns true if the window can be moved in this state.
    pub fn is_movable(&self) -> bool {
        matches!(self, Self::Normal)
    }
}

// ---------------------------------------------------------------------------
// WindowEdge
// ---------------------------------------------------------------------------

/// Edge or corner of a window, used for resize handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowEdge {
    None,
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl WindowEdge {
    /// Returns the cursor style for this edge.
    pub fn cursor_style(&self) -> CursorStyle {
        match self {
            Self::None => CursorStyle::Default,
            Self::Top | Self::Bottom => CursorStyle::ResizeVertical,
            Self::Left | Self::Right => CursorStyle::ResizeHorizontal,
            Self::TopLeft | Self::BottomRight => CursorStyle::ResizeNWSE,
            Self::TopRight | Self::BottomLeft => CursorStyle::ResizeNESW,
        }
    }

    /// Returns true if this edge involves horizontal resizing.
    pub fn resizes_horizontal(&self) -> bool {
        matches!(
            self,
            Self::Left | Self::Right | Self::TopLeft | Self::TopRight | Self::BottomLeft | Self::BottomRight
        )
    }

    /// Returns true if this edge involves vertical resizing.
    pub fn resizes_vertical(&self) -> bool {
        matches!(
            self,
            Self::Top | Self::Bottom | Self::TopLeft | Self::TopRight | Self::BottomLeft | Self::BottomRight
        )
    }
}

/// Cursor style hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorStyle {
    Default,
    Hand,
    Text,
    Move,
    ResizeHorizontal,
    ResizeVertical,
    ResizeNWSE,
    ResizeNESW,
    Wait,
    NotAllowed,
    Crosshair,
}

// ---------------------------------------------------------------------------
// MonitorInfo
// ---------------------------------------------------------------------------

/// Information about a display monitor.
#[derive(Debug, Clone)]
pub struct MonitorInfo {
    /// Monitor index (0 = primary).
    pub index: u32,
    /// Monitor name.
    pub name: String,
    /// Position of the monitor in the virtual desktop space.
    pub position: Vec2,
    /// Resolution in pixels.
    pub resolution: Vec2,
    /// Usable area (excluding taskbar, system bars).
    pub work_area_position: Vec2,
    /// Usable area size.
    pub work_area_size: Vec2,
    /// DPI scale factor (1.0 = 96 DPI).
    pub dpi_scale: f32,
    /// Whether this is the primary monitor.
    pub is_primary: bool,
    /// Refresh rate in Hz.
    pub refresh_rate: u32,
}

impl MonitorInfo {
    /// Creates a new monitor info for a single monitor setup.
    pub fn primary(width: f32, height: f32) -> Self {
        Self {
            index: 0,
            name: "Primary".to_string(),
            position: Vec2::ZERO,
            resolution: Vec2::new(width, height),
            work_area_position: Vec2::new(0.0, 0.0),
            work_area_size: Vec2::new(width, height - 40.0),
            dpi_scale: 1.0,
            is_primary: true,
            refresh_rate: 60,
        }
    }

    /// Returns the bounds of this monitor [x, y, w, h].
    pub fn bounds(&self) -> [f32; 4] {
        [
            self.position.x,
            self.position.y,
            self.resolution.x,
            self.resolution.y,
        ]
    }

    /// Returns the work area bounds.
    pub fn work_area(&self) -> [f32; 4] {
        [
            self.work_area_position.x,
            self.work_area_position.y,
            self.work_area_size.x,
            self.work_area_size.y,
        ]
    }

    /// Returns true if a point is on this monitor.
    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.position.x
            && point.x < self.position.x + self.resolution.x
            && point.y >= self.position.y
            && point.y < self.position.y + self.resolution.y
    }

    /// Clamps a window position to be within this monitor's work area.
    pub fn clamp_to_work_area(&self, pos: Vec2, size: Vec2) -> Vec2 {
        let wa = self.work_area();
        Vec2::new(
            pos.x.clamp(wa[0], wa[0] + wa[2] - size.x),
            pos.y.clamp(wa[1], wa[1] + wa[3] - size.y),
        )
    }
}

// ---------------------------------------------------------------------------
// SnapZone
// ---------------------------------------------------------------------------

/// A screen edge snap zone for window positioning.
#[derive(Debug, Clone)]
pub struct SnapZone {
    /// The screen edge this zone is associated with.
    pub edge: WindowState,
    /// Detection rectangle [x, y, w, h].
    pub detection_rect: [f32; 4],
    /// Target rectangle when snapped [x, y, w, h].
    pub target_rect: [f32; 4],
    /// Whether this zone is currently active (mouse in zone during drag).
    pub active: bool,
    /// Preview opacity for the snap target.
    pub preview_opacity: f32,
}

impl SnapZone {
    /// Creates a new snap zone.
    pub fn new(edge: WindowState, detection: [f32; 4], target: [f32; 4]) -> Self {
        Self {
            edge,
            detection_rect: detection,
            target_rect: target,
            active: false,
            preview_opacity: 0.0,
        }
    }

    /// Returns true if a point is in the detection zone.
    pub fn contains_point(&self, point: Vec2) -> bool {
        let d = &self.detection_rect;
        point.x >= d[0]
            && point.x < d[0] + d[2]
            && point.y >= d[1]
            && point.y < d[1] + d[3]
    }

    /// Updates the preview animation.
    pub fn update(&mut self, dt: f32) {
        let target_opacity = if self.active { 0.3 } else { 0.0 };
        self.preview_opacity += (target_opacity - self.preview_opacity) * (dt * 10.0).min(1.0);
    }
}

// ---------------------------------------------------------------------------
// WindowDragState
// ---------------------------------------------------------------------------

/// State tracking for window drag operations.
#[derive(Debug, Clone)]
pub struct WindowDragState {
    /// Whether a drag is in progress.
    pub dragging: bool,
    /// Which edge is being dragged (None for title bar move).
    pub drag_edge: WindowEdge,
    /// Mouse position at drag start.
    pub drag_start_mouse: Vec2,
    /// Window position at drag start.
    pub drag_start_pos: Vec2,
    /// Window size at drag start.
    pub drag_start_size: Vec2,
    /// Active snap zone during drag.
    pub snap_zone: Option<WindowState>,
}

impl WindowDragState {
    /// Creates a new idle drag state.
    pub fn new() -> Self {
        Self {
            dragging: false,
            drag_edge: WindowEdge::None,
            drag_start_mouse: Vec2::ZERO,
            drag_start_pos: Vec2::ZERO,
            drag_start_size: Vec2::ZERO,
            snap_zone: None,
        }
    }

    /// Begins a move drag.
    pub fn begin_move(&mut self, mouse_pos: Vec2, window_pos: Vec2) {
        self.dragging = true;
        self.drag_edge = WindowEdge::None;
        self.drag_start_mouse = mouse_pos;
        self.drag_start_pos = window_pos;
    }

    /// Begins a resize drag.
    pub fn begin_resize(&mut self, mouse_pos: Vec2, window_pos: Vec2, window_size: Vec2, edge: WindowEdge) {
        self.dragging = true;
        self.drag_edge = edge;
        self.drag_start_mouse = mouse_pos;
        self.drag_start_pos = window_pos;
        self.drag_start_size = window_size;
    }

    /// Ends the drag.
    pub fn end_drag(&mut self) {
        self.dragging = false;
        self.snap_zone = None;
    }

    /// Computes the new position during a move drag.
    pub fn compute_move_position(&self, current_mouse: Vec2) -> Vec2 {
        let delta = current_mouse - self.drag_start_mouse;
        self.drag_start_pos + delta
    }

    /// Computes the new position and size during a resize drag.
    pub fn compute_resize(&self, current_mouse: Vec2, min_size: Vec2) -> (Vec2, Vec2) {
        let delta = current_mouse - self.drag_start_mouse;
        let mut pos = self.drag_start_pos;
        let mut size = self.drag_start_size;

        match self.drag_edge {
            WindowEdge::Right => {
                size.x = (self.drag_start_size.x + delta.x).max(min_size.x);
            }
            WindowEdge::Bottom => {
                size.y = (self.drag_start_size.y + delta.y).max(min_size.y);
            }
            WindowEdge::Left => {
                let new_width = (self.drag_start_size.x - delta.x).max(min_size.x);
                pos.x = self.drag_start_pos.x + self.drag_start_size.x - new_width;
                size.x = new_width;
            }
            WindowEdge::Top => {
                let new_height = (self.drag_start_size.y - delta.y).max(min_size.y);
                pos.y = self.drag_start_pos.y + self.drag_start_size.y - new_height;
                size.y = new_height;
            }
            WindowEdge::BottomRight => {
                size.x = (self.drag_start_size.x + delta.x).max(min_size.x);
                size.y = (self.drag_start_size.y + delta.y).max(min_size.y);
            }
            WindowEdge::TopLeft => {
                let new_width = (self.drag_start_size.x - delta.x).max(min_size.x);
                let new_height = (self.drag_start_size.y - delta.y).max(min_size.y);
                pos.x = self.drag_start_pos.x + self.drag_start_size.x - new_width;
                pos.y = self.drag_start_pos.y + self.drag_start_size.y - new_height;
                size = Vec2::new(new_width, new_height);
            }
            WindowEdge::TopRight => {
                size.x = (self.drag_start_size.x + delta.x).max(min_size.x);
                let new_height = (self.drag_start_size.y - delta.y).max(min_size.y);
                pos.y = self.drag_start_pos.y + self.drag_start_size.y - new_height;
                size.y = new_height;
            }
            WindowEdge::BottomLeft => {
                let new_width = (self.drag_start_size.x - delta.x).max(min_size.x);
                pos.x = self.drag_start_pos.x + self.drag_start_size.x - new_width;
                size.x = new_width;
                size.y = (self.drag_start_size.y + delta.y).max(min_size.y);
            }
            WindowEdge::None => {}
        }

        (pos, size)
    }
}

impl Default for WindowDragState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FloatingWindow
// ---------------------------------------------------------------------------

/// A floating tool window that can be moved, resized, minimized, and maximized.
///
/// Floating windows are created when dock tabs are torn off or when the user
/// opens a tool window. They have a custom title bar drawn by the UI system
/// (not the OS), with minimize/maximize/close buttons.
///
/// # Features
///
/// - Custom title bar with icon, title text, and window controls.
/// - Resizable with handle cursors on all edges and corners.
/// - Snap to screen edges (left half, right half, maximize).
/// - Remember position and size on close, restore on reopen.
/// - DPI-aware rendering.
#[derive(Debug, Clone)]
pub struct FloatingWindow {
    /// Unique identifier.
    pub id: WindowId,
    /// Window title.
    pub title: String,
    /// Window icon (texture ID).
    pub icon: Option<u64>,
    /// Current position in virtual desktop pixels.
    pub position: Vec2,
    /// Current size in logical pixels.
    pub size: Vec2,
    /// Minimum size.
    pub min_size: Vec2,
    /// Maximum size.
    pub max_size: Vec2,
    /// Current visual state.
    pub state: WindowState,
    /// Previous state (before minimize/maximize).
    pub previous_state: WindowState,
    /// Position before maximize/snap (for restore).
    pub restore_position: Vec2,
    /// Size before maximize/snap (for restore).
    pub restore_size: Vec2,
    /// Whether the window is currently visible.
    pub visible: bool,
    /// Whether the window is currently focused.
    pub focused: bool,
    /// Z-order (higher = on top).
    pub z_order: i32,
    /// Whether the window is resizable.
    pub resizable: bool,
    /// Whether the window has minimize button.
    pub has_minimize: bool,
    /// Whether the window has maximize button.
    pub has_maximize: bool,
    /// Whether the window has close button.
    pub has_close: bool,
    /// Drag state.
    pub drag: WindowDragState,
    /// Title bar height in logical pixels.
    pub title_bar_height: f32,
    /// Resize handle size in logical pixels.
    pub resize_handle_size: f32,
    /// DPI scale factor.
    pub dpi_scale: f32,
    /// Background color.
    pub background_color: Color,
    /// Title bar color.
    pub title_bar_color: Color,
    /// Title bar color when focused.
    pub title_bar_focused_color: Color,
    /// Title text color.
    pub title_text_color: Color,
    /// Border color.
    pub border_color: Color,
    /// Whether the close button is hovered.
    pub close_hovered: bool,
    /// Whether the minimize button is hovered.
    pub minimize_hovered: bool,
    /// Whether the maximize button is hovered.
    pub maximize_hovered: bool,
    /// Corner radius.
    pub corner_radius: f32,
    /// Shadow parameters.
    pub shadow_offset: Vec2,
    /// Shadow blur radius.
    pub shadow_blur: f32,
    /// Shadow color.
    pub shadow_color: Color,
    /// Opacity.
    pub opacity: f32,
    /// The widget ID of the content root.
    pub content_root: Option<UIId>,
    /// Monitor index this window is on.
    pub monitor_index: u32,
    /// Whether this window has been closed (for saved position tracking).
    pub was_closed: bool,
    /// Saved position for reopening.
    pub saved_position: Option<Vec2>,
    /// Saved size for reopening.
    pub saved_size: Option<Vec2>,
}

impl FloatingWindow {
    /// Creates a new floating window.
    pub fn new(title: &str) -> Self {
        let id = next_window_id();
        Self {
            id,
            title: title.to_string(),
            icon: None,
            position: Vec2::new(100.0, 100.0),
            size: Vec2::new(400.0, 300.0),
            min_size: Vec2::new(200.0, 150.0),
            max_size: Vec2::new(4096.0, 4096.0),
            state: WindowState::Normal,
            previous_state: WindowState::Normal,
            restore_position: Vec2::new(100.0, 100.0),
            restore_size: Vec2::new(400.0, 300.0),
            visible: true,
            focused: false,
            z_order: 0,
            resizable: true,
            has_minimize: true,
            has_maximize: true,
            has_close: true,
            drag: WindowDragState::new(),
            title_bar_height: 30.0,
            resize_handle_size: 5.0,
            dpi_scale: 1.0,
            background_color: Color::new(0.14, 0.14, 0.17, 1.0),
            title_bar_color: Color::new(0.12, 0.12, 0.15, 1.0),
            title_bar_focused_color: Color::new(0.15, 0.15, 0.2, 1.0),
            title_text_color: Color::new(0.85, 0.85, 0.85, 1.0),
            border_color: Color::new(0.25, 0.25, 0.3, 1.0),
            close_hovered: false,
            minimize_hovered: false,
            maximize_hovered: false,
            corner_radius: 6.0,
            shadow_offset: Vec2::new(0.0, 4.0),
            shadow_blur: 12.0,
            shadow_color: Color::new(0.0, 0.0, 0.0, 0.4),
            opacity: 1.0,
            content_root: None,
            monitor_index: 0,
            was_closed: false,
            saved_position: None,
            saved_size: None,
        }
    }

    /// Returns the title bar rectangle.
    pub fn title_bar_rect(&self) -> [f32; 4] {
        [self.position.x, self.position.y, self.size.x, self.title_bar_height]
    }

    /// Returns the content area rectangle.
    pub fn content_rect(&self) -> [f32; 4] {
        [
            self.position.x,
            self.position.y + self.title_bar_height,
            self.size.x,
            self.size.y - self.title_bar_height,
        ]
    }

    /// Returns the full window bounds [x, y, w, h].
    pub fn bounds(&self) -> [f32; 4] {
        [self.position.x, self.position.y, self.size.x, self.size.y]
    }

    /// Returns true if a point is in the window bounds.
    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.position.x
            && point.x < self.position.x + self.size.x
            && point.y >= self.position.y
            && point.y < self.position.y + self.size.y
    }

    /// Returns true if a point is in the title bar.
    pub fn is_in_title_bar(&self, point: Vec2) -> bool {
        point.x >= self.position.x
            && point.x < self.position.x + self.size.x
            && point.y >= self.position.y
            && point.y < self.position.y + self.title_bar_height
    }

    /// Determines which resize edge a point is near.
    pub fn hit_test_edge(&self, point: Vec2) -> WindowEdge {
        if !self.resizable || !self.state.is_resizable() {
            return WindowEdge::None;
        }

        let hs = self.resize_handle_size;
        let x = self.position.x;
        let y = self.position.y;
        let w = self.size.x;
        let h = self.size.y;

        let on_left = point.x >= x - hs && point.x < x + hs;
        let on_right = point.x >= x + w - hs && point.x < x + w + hs;
        let on_top = point.y >= y - hs && point.y < y + hs;
        let on_bottom = point.y >= y + h - hs && point.y < y + h + hs;

        if on_top && on_left {
            WindowEdge::TopLeft
        } else if on_top && on_right {
            WindowEdge::TopRight
        } else if on_bottom && on_left {
            WindowEdge::BottomLeft
        } else if on_bottom && on_right {
            WindowEdge::BottomRight
        } else if on_top {
            WindowEdge::Top
        } else if on_bottom {
            WindowEdge::Bottom
        } else if on_left {
            WindowEdge::Left
        } else if on_right {
            WindowEdge::Right
        } else {
            WindowEdge::None
        }
    }

    /// Minimizes the window.
    pub fn minimize(&mut self) {
        if self.state != WindowState::Minimized {
            self.previous_state = self.state;
            self.state = WindowState::Minimized;
        }
    }

    /// Maximizes the window (or restores if already maximized).
    pub fn toggle_maximize(&mut self, monitor: &MonitorInfo) {
        if self.state == WindowState::Maximized {
            self.restore();
        } else {
            self.restore_position = self.position;
            self.restore_size = self.size;
            self.previous_state = self.state;
            self.state = WindowState::Maximized;
            let wa = monitor.work_area();
            self.position = Vec2::new(wa[0], wa[1]);
            self.size = Vec2::new(wa[2], wa[3]);
        }
    }

    /// Restores the window to its previous state and position.
    pub fn restore(&mut self) {
        self.state = WindowState::Normal;
        self.position = self.restore_position;
        self.size = self.restore_size;
    }

    /// Closes the window.
    pub fn close(&mut self) {
        self.saved_position = Some(self.position);
        self.saved_size = Some(self.size);
        self.visible = false;
        self.was_closed = true;
    }

    /// Reopens a previously closed window.
    pub fn reopen(&mut self) {
        if let Some(pos) = self.saved_position {
            self.position = pos;
        }
        if let Some(size) = self.saved_size {
            self.size = size;
        }
        self.visible = true;
        self.was_closed = false;
    }

    /// Snaps the window to a screen edge.
    pub fn snap_to(&mut self, state: WindowState, monitor: &MonitorInfo) {
        if self.state == WindowState::Normal {
            self.restore_position = self.position;
            self.restore_size = self.size;
        }

        let wa = monitor.work_area();
        match state {
            WindowState::SnappedLeft => {
                self.position = Vec2::new(wa[0], wa[1]);
                self.size = Vec2::new(wa[2] * 0.5, wa[3]);
            }
            WindowState::SnappedRight => {
                self.position = Vec2::new(wa[0] + wa[2] * 0.5, wa[1]);
                self.size = Vec2::new(wa[2] * 0.5, wa[3]);
            }
            WindowState::SnappedTop => {
                self.position = Vec2::new(wa[0], wa[1]);
                self.size = Vec2::new(wa[2], wa[3] * 0.5);
            }
            WindowState::SnappedBottom => {
                self.position = Vec2::new(wa[0], wa[1] + wa[3] * 0.5);
                self.size = Vec2::new(wa[2], wa[3] * 0.5);
            }
            _ => {}
        }
        self.state = state;
    }

    /// Clamps the window to be within the given monitor's work area.
    pub fn ensure_visible(&mut self, monitor: &MonitorInfo) {
        if self.state == WindowState::Normal {
            self.position = monitor.clamp_to_work_area(self.position, self.size);
        }
    }

    /// Close button rectangle.
    pub fn close_button_rect(&self) -> [f32; 4] {
        let btn_size = self.title_bar_height - 8.0;
        [
            self.position.x + self.size.x - btn_size - 4.0,
            self.position.y + 4.0,
            btn_size,
            btn_size,
        ]
    }

    /// Maximize button rectangle.
    pub fn maximize_button_rect(&self) -> [f32; 4] {
        let btn_size = self.title_bar_height - 8.0;
        [
            self.position.x + self.size.x - (btn_size + 4.0) * 2.0,
            self.position.y + 4.0,
            btn_size,
            btn_size,
        ]
    }

    /// Minimize button rectangle.
    pub fn minimize_button_rect(&self) -> [f32; 4] {
        let btn_size = self.title_bar_height - 8.0;
        [
            self.position.x + self.size.x - (btn_size + 4.0) * 3.0,
            self.position.y + 4.0,
            btn_size,
            btn_size,
        ]
    }
}

// ---------------------------------------------------------------------------
// PopupAnchor
// ---------------------------------------------------------------------------

/// Positioning for a popup window relative to its anchor widget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopupAnchor {
    /// Below the anchor widget.
    Below,
    /// Above the anchor widget.
    Above,
    /// To the left of the anchor widget.
    Left,
    /// To the right of the anchor widget.
    Right,
    /// At the cursor position.
    AtCursor,
}

// ---------------------------------------------------------------------------
// PopupWindow
// ---------------------------------------------------------------------------

/// A popup window for menus, tooltips, and dropdowns.
///
/// Popups auto-close when clicking outside their bounds. They are positioned
/// relative to an anchor widget and automatically reposition if they would be
/// clipped by the screen edge.
#[derive(Debug, Clone)]
pub struct PopupWindow {
    /// Unique identifier.
    pub id: WindowId,
    /// Position in screen pixels.
    pub position: Vec2,
    /// Size in logical pixels.
    pub size: Vec2,
    /// Anchor widget bounds [x, y, w, h].
    pub anchor_rect: [f32; 4],
    /// Preferred anchor direction.
    pub anchor_direction: PopupAnchor,
    /// Whether the popup is currently open.
    pub open: bool,
    /// Whether clicking outside should close the popup.
    pub auto_close: bool,
    /// Whether to close when the anchor widget loses focus.
    pub close_on_blur: bool,
    /// Background color.
    pub background_color: Color,
    /// Border color.
    pub border_color: Color,
    /// Corner radius.
    pub corner_radius: f32,
    /// Shadow parameters.
    pub shadow_blur: f32,
    /// Shadow color.
    pub shadow_color: Color,
    /// Z-order.
    pub z_order: i32,
    /// Margin from the anchor widget.
    pub anchor_margin: f32,
    /// The widget root for this popup's content.
    pub content_root: Option<UIId>,
    /// Opacity.
    pub opacity: f32,
    /// Animation progress (0 to 1).
    pub open_animation: f32,
    /// Animation speed.
    pub animation_speed: f32,
    /// Screen bounds for auto-repositioning.
    pub screen_bounds: Vec2,
}

impl PopupWindow {
    /// Creates a new popup window.
    pub fn new(anchor_rect: [f32; 4], direction: PopupAnchor) -> Self {
        Self {
            id: next_window_id(),
            position: Vec2::ZERO,
            size: Vec2::new(200.0, 300.0),
            anchor_rect,
            anchor_direction: direction,
            open: true,
            auto_close: true,
            close_on_blur: true,
            background_color: Color::new(0.16, 0.16, 0.2, 0.98),
            border_color: Color::new(0.3, 0.3, 0.35, 1.0),
            corner_radius: 6.0,
            shadow_blur: 8.0,
            shadow_color: Color::new(0.0, 0.0, 0.0, 0.4),
            z_order: 1000,
            anchor_margin: 4.0,
            content_root: None,
            opacity: 0.0,
            open_animation: 0.0,
            animation_speed: 8.0,
            screen_bounds: Vec2::new(1920.0, 1080.0),
        }
    }

    /// Computes the position based on the anchor and screen bounds.
    pub fn compute_position(&mut self) {
        let ar = &self.anchor_rect;
        let mut pos = match self.anchor_direction {
            PopupAnchor::Below => {
                Vec2::new(ar[0], ar[1] + ar[3] + self.anchor_margin)
            }
            PopupAnchor::Above => {
                Vec2::new(ar[0], ar[1] - self.size.y - self.anchor_margin)
            }
            PopupAnchor::Right => {
                Vec2::new(ar[0] + ar[2] + self.anchor_margin, ar[1])
            }
            PopupAnchor::Left => {
                Vec2::new(ar[0] - self.size.x - self.anchor_margin, ar[1])
            }
            PopupAnchor::AtCursor => {
                Vec2::new(ar[0], ar[1])
            }
        };

        // Ensure visible on screen.
        if pos.x + self.size.x > self.screen_bounds.x {
            pos.x = self.screen_bounds.x - self.size.x;
        }
        if pos.y + self.size.y > self.screen_bounds.y {
            pos.y = self.screen_bounds.y - self.size.y;
        }
        if pos.x < 0.0 {
            pos.x = 0.0;
        }
        if pos.y < 0.0 {
            pos.y = 0.0;
        }

        self.position = pos;
    }

    /// Updates the popup animation.
    pub fn update(&mut self, dt: f32) {
        if self.open && self.open_animation < 1.0 {
            self.open_animation = (self.open_animation + dt * self.animation_speed).min(1.0);
            self.opacity = self.open_animation;
        }
    }

    /// Closes the popup.
    pub fn close(&mut self) {
        self.open = false;
        self.opacity = 0.0;
    }

    /// Returns true if a point is outside the popup.
    pub fn is_click_outside(&self, point: Vec2) -> bool {
        !(point.x >= self.position.x
            && point.x < self.position.x + self.size.x
            && point.y >= self.position.y
            && point.y < self.position.y + self.size.y)
    }
}

// ---------------------------------------------------------------------------
// ModalWindow
// ---------------------------------------------------------------------------

/// A modal window that blocks input to all other windows.
///
/// Centers itself on the parent window and dims the background. Input events
/// are not passed to windows behind the modal.
#[derive(Debug, Clone)]
pub struct ModalWindow {
    /// Unique identifier.
    pub id: WindowId,
    /// Title text.
    pub title: String,
    /// Position (computed, centered on parent).
    pub position: Vec2,
    /// Size.
    pub size: Vec2,
    /// Parent window ID.
    pub parent: WindowId,
    /// Parent window size.
    pub parent_size: Vec2,
    /// Whether the window is open.
    pub open: bool,
    /// Backdrop color.
    pub backdrop_color: Color,
    /// Background color.
    pub background_color: Color,
    /// Z-order.
    pub z_order: i32,
    /// Open animation progress.
    pub animation: f32,
    /// Animation speed.
    pub animation_speed: f32,
    /// Content root widget.
    pub content_root: Option<UIId>,
    /// Corner radius.
    pub corner_radius: f32,
}

impl ModalWindow {
    /// Creates a new modal window.
    pub fn new(title: &str, parent: WindowId, parent_size: Vec2) -> Self {
        Self {
            id: next_window_id(),
            title: title.to_string(),
            position: Vec2::ZERO,
            size: Vec2::new(400.0, 300.0),
            parent,
            parent_size,
            open: true,
            backdrop_color: Color::new(0.0, 0.0, 0.0, 0.5),
            background_color: Color::new(0.18, 0.18, 0.22, 1.0),
            z_order: 10000,
            animation: 0.0,
            animation_speed: 5.0,
            content_root: None,
            corner_radius: 8.0,
        }
    }

    /// Updates the modal (animation, centering).
    pub fn update(&mut self, dt: f32) {
        if self.open && self.animation < 1.0 {
            self.animation = (self.animation + dt * self.animation_speed).min(1.0);
        }
        self.position = Vec2::new(
            (self.parent_size.x - self.size.x) * 0.5,
            (self.parent_size.y - self.size.y) * 0.5,
        );
    }

    /// Closes the modal.
    pub fn close(&mut self) {
        self.open = false;
    }

    /// Returns the effective backdrop opacity.
    pub fn effective_backdrop_opacity(&self) -> f32 {
        self.backdrop_color.a * self.animation
    }
}

// ---------------------------------------------------------------------------
// WindowLayoutEntry
// ---------------------------------------------------------------------------

/// Saved position and state of a window for layout persistence.
#[derive(Debug, Clone)]
pub struct WindowLayoutEntry {
    /// Window identifier.
    pub window_id: WindowId,
    /// Saved position.
    pub position: Vec2,
    /// Saved size.
    pub size: Vec2,
    /// Saved state.
    pub state: WindowState,
    /// Monitor index.
    pub monitor_index: u32,
    /// Whether the window was visible when layout was saved.
    pub visible: bool,
    /// Window title (for matching when IDs change).
    pub title: String,
}

// ---------------------------------------------------------------------------
// WindowLayout
// ---------------------------------------------------------------------------

/// Saves and restores all window positions and states.
///
/// Multi-monitor aware: tracks which monitor each window is on and adjusts
/// positions when monitors change. DPI-aware: stores positions in logical
/// pixels and adjusts for DPI changes.
#[derive(Debug, Clone)]
pub struct WindowLayout {
    /// Saved window entries.
    pub entries: Vec<WindowLayoutEntry>,
    /// Known monitors.
    pub monitors: Vec<MonitorInfo>,
    /// Layout name (e.g., "Default", "Tall Monitor", "Dual Monitor").
    pub name: String,
    /// Whether the layout has been modified since last save.
    pub dirty: bool,
}

impl WindowLayout {
    /// Creates a new empty window layout.
    pub fn new(name: &str) -> Self {
        Self {
            entries: Vec::new(),
            monitors: Vec::new(),
            name: name.to_string(),
            dirty: false,
        }
    }

    /// Saves the current state of a window.
    pub fn save_window(&mut self, window: &FloatingWindow) {
        // Find existing entry or create new.
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| e.window_id == window.id)
        {
            entry.position = window.position;
            entry.size = window.size;
            entry.state = window.state;
            entry.visible = window.visible;
            entry.monitor_index = window.monitor_index;
        } else {
            self.entries.push(WindowLayoutEntry {
                window_id: window.id,
                position: window.position,
                size: window.size,
                state: window.state,
                monitor_index: window.monitor_index,
                visible: window.visible,
                title: window.title.clone(),
            });
        }
        self.dirty = true;
    }

    /// Restores saved state to a window.
    pub fn restore_window(&self, window: &mut FloatingWindow) {
        if let Some(entry) = self
            .entries
            .iter()
            .find(|e| e.window_id == window.id || e.title == window.title)
        {
            window.position = entry.position;
            window.size = entry.size;
            window.state = entry.state;
            window.visible = entry.visible;
            window.monitor_index = entry.monitor_index;
        }
    }

    /// Returns the number of saved windows.
    pub fn window_count(&self) -> usize {
        self.entries.len()
    }

    /// Clears all saved entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// MultiWindowManager
// ---------------------------------------------------------------------------

/// Top-level manager for all windows in the application.
///
/// Coordinates the main window, floating tool windows, popup menus/tooltips,
/// and modal windows. Handles focus management, z-ordering, and window layout
/// persistence.
#[derive(Debug, Clone)]
pub struct MultiWindowManager {
    /// All floating windows.
    pub floating_windows: HashMap<WindowId, FloatingWindow>,
    /// All popup windows.
    pub popup_windows: Vec<PopupWindow>,
    /// All modal windows.
    pub modal_windows: Vec<ModalWindow>,
    /// The currently focused window ID.
    pub focused_window: Option<WindowId>,
    /// Z-order counter (incremented on focus).
    pub z_order_counter: i32,
    /// Monitor information.
    pub monitors: Vec<MonitorInfo>,
    /// Snap zones for the primary monitor.
    pub snap_zones: Vec<SnapZone>,
    /// Saved window layouts.
    pub layouts: HashMap<String, WindowLayout>,
    /// Active layout name.
    pub active_layout: String,
    /// Whether a modal is currently open.
    pub has_modal: bool,
    /// Current frame number.
    pub current_frame: u64,
    /// Main window size.
    pub main_window_size: Vec2,
    /// Whether window snapping is enabled.
    pub snapping_enabled: bool,
    /// Snap detection distance from screen edge in pixels.
    pub snap_distance: f32,
}

impl MultiWindowManager {
    /// Creates a new multi-window manager.
    pub fn new(main_width: f32, main_height: f32) -> Self {
        let primary = MonitorInfo::primary(main_width, main_height);
        let monitors = vec![primary];

        Self {
            floating_windows: HashMap::new(),
            popup_windows: Vec::new(),
            modal_windows: Vec::new(),
            focused_window: Some(WindowId::MAIN),
            z_order_counter: 1,
            monitors,
            snap_zones: Vec::new(),
            layouts: HashMap::new(),
            active_layout: "Default".to_string(),
            has_modal: false,
            current_frame: 0,
            main_window_size: Vec2::new(main_width, main_height),
            snapping_enabled: true,
            snap_distance: 20.0,
        }
    }

    /// Creates a new floating window and returns its ID.
    pub fn create_floating(&mut self, title: &str) -> WindowId {
        let mut window = FloatingWindow::new(title);
        self.z_order_counter += 1;
        window.z_order = self.z_order_counter;
        let id = window.id;
        self.floating_windows.insert(id, window);
        id
    }

    /// Removes a floating window.
    pub fn destroy_floating(&mut self, id: WindowId) {
        if let Some(mut window) = self.floating_windows.remove(&id) {
            window.close();
        }
        if self.focused_window == Some(id) {
            self.focused_window = Some(WindowId::MAIN);
        }
    }

    /// Shows a popup window and returns its ID.
    pub fn show_popup(
        &mut self,
        anchor_rect: [f32; 4],
        direction: PopupAnchor,
        size: Vec2,
    ) -> WindowId {
        let mut popup = PopupWindow::new(anchor_rect, direction);
        popup.size = size;
        popup.screen_bounds = self.main_window_size;
        popup.compute_position();
        self.z_order_counter += 1;
        popup.z_order = self.z_order_counter;
        let id = popup.id;
        self.popup_windows.push(popup);
        id
    }

    /// Closes a popup by ID.
    pub fn close_popup(&mut self, id: WindowId) {
        if let Some(popup) = self.popup_windows.iter_mut().find(|p| p.id == id) {
            popup.close();
        }
    }

    /// Closes all popups.
    pub fn close_all_popups(&mut self) {
        for popup in &mut self.popup_windows {
            popup.close();
        }
    }

    /// Shows a modal window and returns its ID.
    pub fn show_modal(&mut self, title: &str) -> WindowId {
        let mut modal = ModalWindow::new(title, WindowId::MAIN, self.main_window_size);
        self.z_order_counter += 1;
        modal.z_order = self.z_order_counter;
        let id = modal.id;
        self.modal_windows.push(modal);
        self.has_modal = true;
        id
    }

    /// Closes a modal by ID.
    pub fn close_modal(&mut self, id: WindowId) {
        self.modal_windows.retain(|m| m.id != id);
        self.has_modal = !self.modal_windows.is_empty();
    }

    /// Focuses a window, bringing it to the front.
    pub fn focus_window(&mut self, id: WindowId) {
        self.z_order_counter += 1;
        if let Some(window) = self.floating_windows.get_mut(&id) {
            window.z_order = self.z_order_counter;
            window.focused = true;
        }
        // Unfocus previous.
        for (wid, window) in &mut self.floating_windows {
            if *wid != id {
                window.focused = false;
            }
        }
        self.focused_window = Some(id);
    }

    /// Updates all windows for one frame.
    pub fn update(&mut self, dt: f32) {
        self.current_frame += 1;

        // Update popups.
        for popup in &mut self.popup_windows {
            popup.update(dt);
        }
        self.popup_windows.retain(|p| p.open);

        // Update modals.
        for modal in &mut self.modal_windows {
            modal.update(dt);
        }
        self.modal_windows.retain(|m| m.open);
        self.has_modal = !self.modal_windows.is_empty();
    }

    /// Returns windows sorted by z-order (back to front) for rendering.
    pub fn render_order(&self) -> Vec<WindowId> {
        let mut windows: Vec<(WindowId, i32)> = self
            .floating_windows
            .values()
            .filter(|w| w.visible)
            .map(|w| (w.id, w.z_order))
            .collect();
        windows.sort_by_key(|(_, z)| *z);
        windows.into_iter().map(|(id, _)| id).collect()
    }

    /// Handles a click and determines which window was hit.
    pub fn hit_test(&self, point: Vec2) -> Option<WindowId> {
        // Check modals first.
        if let Some(modal) = self.modal_windows.last() {
            return Some(modal.id);
        }

        // Check popups.
        for popup in self.popup_windows.iter().rev() {
            if !popup.is_click_outside(point) {
                return Some(popup.id);
            }
        }

        // Check floating windows (reverse z-order).
        let mut windows: Vec<&FloatingWindow> = self
            .floating_windows
            .values()
            .filter(|w| w.visible)
            .collect();
        windows.sort_by(|a, b| b.z_order.cmp(&a.z_order));
        for window in windows {
            if window.contains_point(point) {
                return Some(window.id);
            }
        }

        Some(WindowId::MAIN)
    }

    /// Saves the current window layout.
    pub fn save_layout(&mut self, name: &str) {
        let mut layout = WindowLayout::new(name);
        layout.monitors = self.monitors.clone();
        for window in self.floating_windows.values() {
            layout.save_window(window);
        }
        self.layouts.insert(name.to_string(), layout);
    }

    /// Restores a saved window layout.
    pub fn restore_layout(&mut self, name: &str) {
        if let Some(layout) = self.layouts.get(name).cloned() {
            for window in self.floating_windows.values_mut() {
                layout.restore_window(window);
            }
        }
    }

    /// Returns the total number of open windows.
    pub fn window_count(&self) -> usize {
        1 + self.floating_windows.len() + self.popup_windows.len() + self.modal_windows.len()
    }

    /// Returns the monitor that contains the given point.
    pub fn monitor_at_point(&self, point: Vec2) -> Option<&MonitorInfo> {
        self.monitors.iter().find(|m| m.contains_point(point))
    }

    /// Gets a reference to a floating window.
    pub fn get_floating(&self, id: WindowId) -> Option<&FloatingWindow> {
        self.floating_windows.get(&id)
    }

    /// Gets a mutable reference to a floating window.
    pub fn get_floating_mut(&mut self, id: WindowId) -> Option<&mut FloatingWindow> {
        self.floating_windows.get_mut(&id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floating_window_creation() {
        let window = FloatingWindow::new("Test Window");
        assert!(window.visible);
        assert_eq!(window.state, WindowState::Normal);
        assert!(window.id.is_valid());
    }

    #[test]
    fn test_floating_window_resize_edge() {
        let window = FloatingWindow::new("Test");
        let edge = window.hit_test_edge(Vec2::new(
            window.position.x + window.size.x,
            window.position.y + window.size.y,
        ));
        assert_eq!(edge, WindowEdge::BottomRight);
    }

    #[test]
    fn test_floating_window_minimize_restore() {
        let mut window = FloatingWindow::new("Test");
        let original_pos = window.position;
        window.minimize();
        assert_eq!(window.state, WindowState::Minimized);
        window.restore();
        assert_eq!(window.state, WindowState::Normal);
        assert_eq!(window.position, original_pos);
    }

    #[test]
    fn test_floating_window_close_reopen() {
        let mut window = FloatingWindow::new("Test");
        let pos = window.position;
        window.close();
        assert!(!window.visible);
        assert!(window.was_closed);
        window.reopen();
        assert!(window.visible);
        assert_eq!(window.position, pos);
    }

    #[test]
    fn test_popup_positioning() {
        let mut popup = PopupWindow::new([100.0, 50.0, 80.0, 30.0], PopupAnchor::Below);
        popup.size = Vec2::new(200.0, 150.0);
        popup.screen_bounds = Vec2::new(1920.0, 1080.0);
        popup.compute_position();
        // Should be below anchor.
        assert!(popup.position.y > 50.0 + 30.0);
    }

    #[test]
    fn test_popup_screen_clamping() {
        let mut popup = PopupWindow::new([1800.0, 1000.0, 80.0, 30.0], PopupAnchor::Below);
        popup.size = Vec2::new(200.0, 150.0);
        popup.screen_bounds = Vec2::new(1920.0, 1080.0);
        popup.compute_position();
        // Should be clamped to not go off-screen.
        assert!(popup.position.x + popup.size.x <= 1920.0);
    }

    #[test]
    fn test_multi_window_manager_focus() {
        let mut mgr = MultiWindowManager::new(1920.0, 1080.0);
        let id1 = mgr.create_floating("Window 1");
        let id2 = mgr.create_floating("Window 2");

        mgr.focus_window(id1);
        assert_eq!(mgr.focused_window, Some(id1));

        mgr.focus_window(id2);
        assert_eq!(mgr.focused_window, Some(id2));
        assert!(mgr.floating_windows[&id2].z_order > mgr.floating_windows[&id1].z_order);
    }

    #[test]
    fn test_multi_window_manager_layout() {
        let mut mgr = MultiWindowManager::new(1920.0, 1080.0);
        let id = mgr.create_floating("Tool");
        mgr.floating_windows.get_mut(&id).unwrap().position = Vec2::new(500.0, 300.0);

        mgr.save_layout("Test Layout");
        assert!(mgr.layouts.contains_key("Test Layout"));
    }

    #[test]
    fn test_window_drag_state_move() {
        let mut drag = WindowDragState::new();
        drag.begin_move(Vec2::new(100.0, 50.0), Vec2::new(200.0, 100.0));
        let new_pos = drag.compute_move_position(Vec2::new(150.0, 80.0));
        assert_eq!(new_pos, Vec2::new(250.0, 130.0));
    }

    #[test]
    fn test_monitor_info() {
        let monitor = MonitorInfo::primary(1920.0, 1080.0);
        assert!(monitor.contains_point(Vec2::new(960.0, 540.0)));
        assert!(!monitor.contains_point(Vec2::new(2000.0, 540.0)));
    }
}
