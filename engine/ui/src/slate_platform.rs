//! Platform integration layer for the Slate UI framework.
//!
//! Provides abstractions for OS window management, clipboard, cursors,
//! DPI scaling, file dialogs, and system information queries. This layer
//! decouples the UI framework from the underlying platform so that the
//! same widget code works on Windows, macOS, and Linux.

use std::collections::HashMap;
use std::path::PathBuf;

use glam::Vec2;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// WindowType
// ---------------------------------------------------------------------------

/// The type/purpose of an OS window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WindowType {
    /// A standard application window with title bar and frame.
    Normal,
    /// A popup window (e.g., dropdown menu, combo box list).
    Popup,
    /// A tooltip window.
    Tooltip,
    /// A menu window.
    Menu,
    /// A floating tool window (e.g., detached dock panel).
    Floating,
    /// A modal dialog.
    Modal,
}

impl Default for WindowType {
    fn default() -> Self {
        Self::Normal
    }
}

// ---------------------------------------------------------------------------
// WindowProperties
// ---------------------------------------------------------------------------

/// Properties for creating or configuring an OS window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowProperties {
    /// Window title (shown in title bar and taskbar).
    pub title: String,
    /// Initial width in logical pixels.
    pub width: f32,
    /// Initial height in logical pixels.
    pub height: f32,
    /// Initial X position in screen coordinates (None = OS default).
    pub x: Option<f32>,
    /// Initial Y position in screen coordinates (None = OS default).
    pub y: Option<f32>,
    /// Minimum width.
    pub min_width: f32,
    /// Minimum height.
    pub min_height: f32,
    /// Maximum width (0 = unconstrained).
    pub max_width: f32,
    /// Maximum height (0 = unconstrained).
    pub max_height: f32,
    /// Whether the window is resizable by the user.
    pub resizable: bool,
    /// Whether the window has OS decorations (title bar, border).
    pub decorated: bool,
    /// Whether the window should stay on top of other windows.
    pub always_on_top: bool,
    /// Whether the window is initially visible.
    pub visible: bool,
    /// Whether the window is initially maximized.
    pub maximized: bool,
    /// Whether the window is initially focused.
    pub focused: bool,
    /// Whether the window should have a transparent background.
    pub transparent: bool,
    /// Window type.
    pub window_type: WindowType,
    /// Whether to use a custom (borderless) title bar rendered by the UI.
    pub custom_title_bar: bool,
    /// Custom title bar height (only used if `custom_title_bar` is true).
    pub custom_title_bar_height: f32,
    /// Icon path or resource identifier.
    pub icon: Option<String>,
}

impl Default for WindowProperties {
    fn default() -> Self {
        Self {
            title: "Genovo".to_string(),
            width: 1280.0,
            height: 720.0,
            x: None,
            y: None,
            min_width: 320.0,
            min_height: 240.0,
            max_width: 0.0,
            max_height: 0.0,
            resizable: true,
            decorated: true,
            always_on_top: false,
            visible: true,
            maximized: false,
            focused: true,
            transparent: false,
            window_type: WindowType::Normal,
            custom_title_bar: false,
            custom_title_bar_height: 30.0,
            icon: None,
        }
    }
}

impl WindowProperties {
    /// Create properties for a normal application window.
    pub fn normal(title: impl Into<String>, width: f32, height: f32) -> Self {
        Self {
            title: title.into(),
            width,
            height,
            ..Default::default()
        }
    }

    /// Create properties for a popup window.
    pub fn popup(width: f32, height: f32) -> Self {
        Self {
            title: String::new(),
            width,
            height,
            decorated: false,
            resizable: false,
            window_type: WindowType::Popup,
            ..Default::default()
        }
    }

    /// Create properties for a tooltip window.
    pub fn tooltip() -> Self {
        Self {
            title: String::new(),
            width: 200.0,
            height: 40.0,
            decorated: false,
            resizable: false,
            always_on_top: true,
            focused: false,
            window_type: WindowType::Tooltip,
            ..Default::default()
        }
    }

    /// Create properties for a floating tool window.
    pub fn floating(title: impl Into<String>, width: f32, height: f32) -> Self {
        Self {
            title: title.into(),
            width,
            height,
            window_type: WindowType::Floating,
            ..Default::default()
        }
    }

    /// Create properties for a modal dialog.
    pub fn modal(title: impl Into<String>, width: f32, height: f32) -> Self {
        Self {
            title: title.into(),
            width,
            height,
            resizable: false,
            window_type: WindowType::Modal,
            ..Default::default()
        }
    }

    /// Builder: set position.
    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.x = Some(x);
        self.y = Some(y);
        self
    }

    /// Builder: set size constraints.
    pub fn with_min_size(mut self, w: f32, h: f32) -> Self {
        self.min_width = w;
        self.min_height = h;
        self
    }

    /// Builder: set max size constraints.
    pub fn with_max_size(mut self, w: f32, h: f32) -> Self {
        self.max_width = w;
        self.max_height = h;
        self
    }

    /// Builder: custom title bar.
    pub fn with_custom_title_bar(mut self, height: f32) -> Self {
        self.custom_title_bar = true;
        self.custom_title_bar_height = height;
        self.decorated = false;
        self
    }
}

// ---------------------------------------------------------------------------
// WindowId
// ---------------------------------------------------------------------------

/// Opaque identifier for an OS window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WindowId(pub u64);

impl WindowId {
    pub const INVALID: Self = Self(u64::MAX);

    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }
}

impl Default for WindowId {
    fn default() -> Self {
        Self::INVALID
    }
}

// ---------------------------------------------------------------------------
// WindowState
// ---------------------------------------------------------------------------

/// Runtime state of a managed window.
#[derive(Debug, Clone)]
pub struct WindowState {
    /// Unique identifier.
    pub id: WindowId,
    /// Current properties.
    pub properties: WindowProperties,
    /// Current position in screen coordinates.
    pub position: Vec2,
    /// Current size in logical pixels.
    pub size: Vec2,
    /// Whether the window is currently focused.
    pub focused: bool,
    /// Whether the window is minimized.
    pub minimized: bool,
    /// Whether the window is maximized.
    pub maximized: bool,
    /// DPI scale factor for this window's monitor.
    pub dpi_scale: f32,
    /// Monitor index this window is primarily on.
    pub monitor_index: usize,
}

impl WindowState {
    /// Create a new window state from properties.
    pub fn from_properties(id: WindowId, props: &WindowProperties) -> Self {
        Self {
            id,
            properties: props.clone(),
            position: Vec2::new(
                props.x.unwrap_or(100.0),
                props.y.unwrap_or(100.0),
            ),
            size: Vec2::new(props.width, props.height),
            focused: props.focused,
            minimized: false,
            maximized: props.maximized,
            dpi_scale: 1.0,
            monitor_index: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// WindowEvent
// ---------------------------------------------------------------------------

/// Events generated by the windowing system.
#[derive(Debug, Clone)]
pub enum WindowEvent {
    /// The window was resized.
    Resized { id: WindowId, width: f32, height: f32 },
    /// The window was moved.
    Moved { id: WindowId, x: f32, y: f32 },
    /// The window received focus.
    Focused { id: WindowId },
    /// The window lost focus.
    Unfocused { id: WindowId },
    /// The window was minimized.
    Minimized { id: WindowId },
    /// The window was restored from minimized state.
    Restored { id: WindowId },
    /// The window was maximized.
    Maximized { id: WindowId },
    /// The close button was pressed.
    CloseRequested { id: WindowId },
    /// The window was destroyed.
    Destroyed { id: WindowId },
    /// DPI scale changed (e.g., moved to a different monitor).
    DpiChanged { id: WindowId, new_scale: f32 },
    /// A file was dropped onto the window.
    FileDropped { id: WindowId, path: PathBuf },
    /// Files are being hovered over the window.
    FileHovered { id: WindowId, paths: Vec<PathBuf> },
    /// File hover cancelled (dragged away).
    FileHoverCancelled { id: WindowId },
}

// ---------------------------------------------------------------------------
// WindowManager
// ---------------------------------------------------------------------------

/// Manages OS windows: creation, destruction, focus, and event routing.
///
/// This is an abstraction layer; the actual platform calls are handled by
/// a backend implementation (e.g., winit, SDL, custom Win32).
pub struct WindowManager {
    /// All managed windows.
    windows: HashMap<WindowId, WindowState>,
    /// Next window id to assign.
    next_id: u64,
    /// The currently focused window.
    focused_window: WindowId,
    /// Pending events from the OS.
    pending_events: Vec<WindowEvent>,
    /// The main/primary window id.
    main_window: WindowId,
}

impl WindowManager {
    /// Create a new window manager.
    pub fn new() -> Self {
        Self {
            windows: HashMap::new(),
            next_id: 1,
            focused_window: WindowId::INVALID,
            pending_events: Vec::new(),
            main_window: WindowId::INVALID,
        }
    }

    /// Create a new window with the given properties. Returns the window id.
    pub fn create_window(&mut self, props: WindowProperties) -> WindowId {
        let id = WindowId(self.next_id);
        self.next_id += 1;

        let state = WindowState::from_properties(id, &props);
        self.windows.insert(id, state);

        if self.main_window == WindowId::INVALID {
            self.main_window = id;
        }

        id
    }

    /// Destroy a window by id.
    pub fn destroy_window(&mut self, id: WindowId) -> bool {
        if self.windows.remove(&id).is_some() {
            self.pending_events
                .push(WindowEvent::Destroyed { id });
            if self.focused_window == id {
                self.focused_window = WindowId::INVALID;
            }
            if self.main_window == id {
                self.main_window = WindowId::INVALID;
            }
            true
        } else {
            false
        }
    }

    /// Get a reference to a window's state.
    pub fn get_window(&self, id: WindowId) -> Option<&WindowState> {
        self.windows.get(&id)
    }

    /// Get a mutable reference to a window's state.
    pub fn get_window_mut(
        &mut self,
        id: WindowId,
    ) -> Option<&mut WindowState> {
        self.windows.get_mut(&id)
    }

    /// Set the title of a window.
    pub fn set_title(&mut self, id: WindowId, title: &str) {
        if let Some(state) = self.windows.get_mut(&id) {
            state.properties.title = title.to_string();
        }
    }

    /// Set the size of a window.
    pub fn set_size(&mut self, id: WindowId, width: f32, height: f32) {
        if let Some(state) = self.windows.get_mut(&id) {
            state.size = Vec2::new(width, height);
            state.properties.width = width;
            state.properties.height = height;
        }
    }

    /// Set the position of a window.
    pub fn set_position(&mut self, id: WindowId, x: f32, y: f32) {
        if let Some(state) = self.windows.get_mut(&id) {
            state.position = Vec2::new(x, y);
        }
    }

    /// Focus a window.
    pub fn focus_window(&mut self, id: WindowId) {
        if self.windows.contains_key(&id) {
            if self.focused_window.is_valid()
                && self.focused_window != id
            {
                if let Some(old) =
                    self.windows.get_mut(&self.focused_window)
                {
                    old.focused = false;
                }
                self.pending_events.push(WindowEvent::Unfocused {
                    id: self.focused_window,
                });
            }
            self.focused_window = id;
            if let Some(state) = self.windows.get_mut(&id) {
                state.focused = true;
            }
            self.pending_events
                .push(WindowEvent::Focused { id });
        }
    }

    /// Get the focused window id.
    pub fn focused_window(&self) -> WindowId {
        self.focused_window
    }

    /// Get the main window id.
    pub fn main_window(&self) -> WindowId {
        self.main_window
    }

    /// Minimize a window.
    pub fn minimize(&mut self, id: WindowId) {
        if let Some(state) = self.windows.get_mut(&id) {
            state.minimized = true;
            state.maximized = false;
            self.pending_events
                .push(WindowEvent::Minimized { id });
        }
    }

    /// Maximize a window.
    pub fn maximize(&mut self, id: WindowId) {
        if let Some(state) = self.windows.get_mut(&id) {
            state.maximized = true;
            state.minimized = false;
            self.pending_events
                .push(WindowEvent::Maximized { id });
        }
    }

    /// Restore a window from minimized/maximized state.
    pub fn restore(&mut self, id: WindowId) {
        if let Some(state) = self.windows.get_mut(&id) {
            state.minimized = false;
            state.maximized = false;
            self.pending_events
                .push(WindowEvent::Restored { id });
        }
    }

    /// Set always-on-top for a window.
    pub fn set_always_on_top(&mut self, id: WindowId, on_top: bool) {
        if let Some(state) = self.windows.get_mut(&id) {
            state.properties.always_on_top = on_top;
        }
    }

    /// Toggle visibility of a window.
    pub fn set_visible(&mut self, id: WindowId, visible: bool) {
        if let Some(state) = self.windows.get_mut(&id) {
            state.properties.visible = visible;
        }
    }

    /// Get all window ids.
    pub fn window_ids(&self) -> Vec<WindowId> {
        self.windows.keys().copied().collect()
    }

    /// Number of managed windows.
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Take pending events (drains the queue).
    pub fn take_events(&mut self) -> Vec<WindowEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Push an external event (from the OS backend).
    pub fn push_event(&mut self, event: WindowEvent) {
        // Update internal state based on event.
        match &event {
            WindowEvent::Resized { id, width, height } => {
                if let Some(state) = self.windows.get_mut(id) {
                    state.size = Vec2::new(*width, *height);
                }
            }
            WindowEvent::Moved { id, x, y } => {
                if let Some(state) = self.windows.get_mut(id) {
                    state.position = Vec2::new(*x, *y);
                }
            }
            WindowEvent::Focused { id } => {
                self.focused_window = *id;
                if let Some(state) = self.windows.get_mut(id) {
                    state.focused = true;
                }
            }
            WindowEvent::DpiChanged { id, new_scale } => {
                if let Some(state) = self.windows.get_mut(id) {
                    state.dpi_scale = *new_scale;
                }
            }
            _ => {}
        }
        self.pending_events.push(event);
    }
}

impl Default for WindowManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CursorType
// ---------------------------------------------------------------------------

/// Mouse cursor types that can be set by the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CursorType {
    /// Default arrow cursor.
    Arrow,
    /// Hand/pointer (for clickable elements).
    Hand,
    /// Text I-beam (for text input areas).
    IBeam,
    /// Horizontal resize.
    ResizeH,
    /// Vertical resize.
    ResizeV,
    /// NW-SE diagonal resize.
    ResizeNW,
    /// NE-SW diagonal resize.
    ResizeNE,
    /// Move/drag cursor.
    Move,
    /// Crosshair.
    Crosshair,
    /// Not-allowed / forbidden.
    NotAllowed,
    /// Wait / hourglass.
    Wait,
}

impl Default for CursorType {
    fn default() -> Self {
        Self::Arrow
    }
}

// ---------------------------------------------------------------------------
// CursorManager
// ---------------------------------------------------------------------------

/// Manages the mouse cursor appearance, visibility, and position.
pub struct CursorManager {
    /// Current cursor type.
    pub current_cursor: CursorType,
    /// Whether the cursor is visible.
    pub visible: bool,
    /// Whether the cursor is locked (for infinite drag).
    pub locked: bool,
    /// Lock center position (where the cursor was locked).
    pub lock_center: Vec2,
    /// Current cursor position (in screen coordinates).
    pub position: Vec2,
    /// Stack of cursor overrides (last one wins).
    cursor_stack: Vec<CursorType>,
}

impl CursorManager {
    /// Create a new cursor manager.
    pub fn new() -> Self {
        Self {
            current_cursor: CursorType::Arrow,
            visible: true,
            locked: false,
            lock_center: Vec2::ZERO,
            position: Vec2::ZERO,
            cursor_stack: Vec::new(),
        }
    }

    /// Set the cursor type.
    pub fn set_cursor(&mut self, cursor: CursorType) {
        self.current_cursor = cursor;
    }

    /// Push a cursor override onto the stack.
    pub fn push_cursor(&mut self, cursor: CursorType) {
        self.cursor_stack.push(self.current_cursor);
        self.current_cursor = cursor;
    }

    /// Pop the last cursor override.
    pub fn pop_cursor(&mut self) {
        if let Some(prev) = self.cursor_stack.pop() {
            self.current_cursor = prev;
        }
    }

    /// Hide the cursor.
    pub fn hide_cursor(&mut self) {
        self.visible = false;
    }

    /// Show the cursor.
    pub fn show_cursor(&mut self) {
        self.visible = true;
    }

    /// Lock the cursor at the given position (for infinite drag).
    pub fn lock_cursor(&mut self, center: Vec2) {
        self.locked = true;
        self.lock_center = center;
    }

    /// Unlock the cursor.
    pub fn unlock_cursor(&mut self) {
        self.locked = false;
    }

    /// Set the cursor position (warp).
    pub fn set_cursor_position(&mut self, pos: Vec2) {
        self.position = pos;
    }

    /// Update the cursor position from the OS.
    pub fn update_position(&mut self, pos: Vec2) {
        self.position = pos;
    }
}

impl Default for CursorManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ClipboardManager
// ---------------------------------------------------------------------------

/// Manages clipboard operations (copy/paste text and images).
pub struct ClipboardManager {
    /// Current text clipboard contents (cached).
    text_contents: String,
    /// Whether the clipboard has been modified since last read.
    dirty: bool,
}

impl ClipboardManager {
    /// Create a new clipboard manager.
    pub fn new() -> Self {
        Self {
            text_contents: String::new(),
            dirty: false,
        }
    }

    /// Copy text to the clipboard.
    pub fn copy_text(&mut self, text: &str) {
        self.text_contents = text.to_string();
        self.dirty = true;
        // In a real implementation, this would call the OS clipboard API.
    }

    /// Paste text from the clipboard. Returns None if clipboard is empty
    /// or does not contain text.
    pub fn paste_text(&self) -> Option<String> {
        // In a real implementation, this would read from the OS clipboard.
        if self.text_contents.is_empty() {
            None
        } else {
            Some(self.text_contents.clone())
        }
    }

    /// Copy image data to the clipboard.
    pub fn copy_image(&mut self, _data: &ImageData) {
        // Placeholder: OS clipboard image support.
    }

    /// Paste image data from the clipboard.
    pub fn paste_image(&self) -> Option<ImageData> {
        // Placeholder: OS clipboard image support.
        None
    }

    /// Whether the clipboard has text content.
    pub fn has_text(&self) -> bool {
        !self.text_contents.is_empty()
    }

    /// Clear the clipboard.
    pub fn clear(&mut self) {
        self.text_contents.clear();
        self.dirty = true;
    }
}

impl Default for ClipboardManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Raw image data for clipboard operations.
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA pixel data.
    pub pixels: Vec<u8>,
}

impl ImageData {
    /// Create a new image data buffer.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width * height * 4) as usize],
        }
    }

    /// Create from existing pixel data.
    pub fn from_pixels(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        Self {
            width,
            height,
            pixels,
        }
    }
}

// ---------------------------------------------------------------------------
// DPIManager
// ---------------------------------------------------------------------------

/// Manages DPI scaling and coordinate conversions.
pub struct DPIManager {
    /// Current DPI scale factor (1.0 = 96 DPI, 2.0 = 192 DPI).
    scale_factor: f32,
    /// Per-monitor DPI scales.
    monitor_scales: Vec<f32>,
}

impl DPIManager {
    /// Create a new DPI manager with a default scale of 1.0.
    pub fn new() -> Self {
        Self {
            scale_factor: 1.0,
            monitor_scales: vec![1.0],
        }
    }

    /// Get the current DPI scale factor.
    pub fn get_dpi_scale(&self) -> f32 {
        self.scale_factor
    }

    /// Set the DPI scale factor.
    pub fn set_dpi_scale(&mut self, scale: f32) {
        self.scale_factor = scale.max(0.5);
    }

    /// Set per-monitor DPI scales.
    pub fn set_monitor_scales(&mut self, scales: Vec<f32>) {
        self.monitor_scales = scales;
    }

    /// Get the DPI scale for a specific monitor.
    pub fn monitor_scale(&self, monitor_index: usize) -> f32 {
        self.monitor_scales
            .get(monitor_index)
            .copied()
            .unwrap_or(1.0)
    }

    /// Convert a logical position to physical (pixel) position.
    pub fn logical_to_physical(&self, pos: Vec2) -> Vec2 {
        pos * self.scale_factor
    }

    /// Convert a physical (pixel) position to logical position.
    pub fn physical_to_logical(&self, pos: Vec2) -> Vec2 {
        pos / self.scale_factor
    }

    /// Convert a logical size to physical size.
    pub fn logical_size_to_physical(&self, size: Vec2) -> Vec2 {
        size * self.scale_factor
    }

    /// Snap a logical position to the nearest physical pixel boundary.
    /// This prevents blurry rendering of lines and text at fractional DPI.
    pub fn snap_to_pixel(&self, pos: Vec2) -> Vec2 {
        let physical = self.logical_to_physical(pos);
        let snapped = Vec2::new(physical.x.round(), physical.y.round());
        self.physical_to_logical(snapped)
    }

    /// Snap a size to the nearest pixel boundary, rounding up.
    pub fn snap_size_to_pixel(&self, size: Vec2) -> Vec2 {
        let physical = self.logical_size_to_physical(size);
        let snapped = Vec2::new(physical.x.ceil(), physical.y.ceil());
        self.physical_to_logical(snapped)
    }

    /// Scale a font size for the current DPI.
    pub fn scale_font_size(&self, logical_size: f32) -> f32 {
        (logical_size * self.scale_factor).round()
    }

    /// Number of monitors.
    pub fn monitor_count(&self) -> usize {
        self.monitor_scales.len()
    }
}

impl Default for DPIManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FileDialog
// ---------------------------------------------------------------------------

/// File type filter for file dialogs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileFilter {
    /// Display name (e.g., "Image files").
    pub name: String,
    /// File extensions without dot (e.g., ["png", "jpg", "bmp"]).
    pub extensions: Vec<String>,
}

impl FileFilter {
    /// Create a new file filter.
    pub fn new(
        name: impl Into<String>,
        extensions: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            extensions,
        }
    }

    /// Common filter: all files.
    pub fn all_files() -> Self {
        Self::new("All files", vec!["*".to_string()])
    }

    /// Common filter: image files.
    pub fn images() -> Self {
        Self::new(
            "Image files",
            vec![
                "png".to_string(),
                "jpg".to_string(),
                "jpeg".to_string(),
                "bmp".to_string(),
                "gif".to_string(),
                "tga".to_string(),
                "hdr".to_string(),
            ],
        )
    }

    /// Common filter: scene files.
    pub fn scenes() -> Self {
        Self::new(
            "Scene files",
            vec!["scene".to_string(), "json".to_string()],
        )
    }

    /// Common filter: Rust source files.
    pub fn rust_source() -> Self {
        Self::new("Rust files", vec!["rs".to_string()])
    }

    /// Common filter: text files.
    pub fn text() -> Self {
        Self::new(
            "Text files",
            vec!["txt".to_string(), "md".to_string(), "log".to_string()],
        )
    }
}

/// File dialog operations.
///
/// These are blocking calls that show native OS file/folder selection dialogs.
/// In a real implementation, these would call into the OS file dialog API.
pub struct FileDialog;

impl FileDialog {
    /// Show an "Open File" dialog. Returns the selected file path, or None
    /// if the user cancelled.
    pub fn open_file(filters: &[FileFilter]) -> Option<PathBuf> {
        // Placeholder: would show an OS file open dialog.
        let _ = filters;
        None
    }

    /// Show an "Open Files" dialog for multiple selection.
    pub fn open_files(filters: &[FileFilter]) -> Vec<PathBuf> {
        let _ = filters;
        Vec::new()
    }

    /// Show a "Save File" dialog. Returns the selected path, or None.
    pub fn save_file(
        filters: &[FileFilter],
        default_name: &str,
    ) -> Option<PathBuf> {
        let _ = (filters, default_name);
        None
    }

    /// Show a "Select Folder" dialog.
    pub fn select_folder() -> Option<PathBuf> {
        None
    }
}

// ---------------------------------------------------------------------------
// MonitorInfo
// ---------------------------------------------------------------------------

/// Information about a connected display monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorInfo {
    /// Human-readable name of the monitor.
    pub name: String,
    /// Width in physical pixels.
    pub width: u32,
    /// Height in physical pixels.
    pub height: u32,
    /// Refresh rate in Hz.
    pub refresh_rate: u32,
    /// DPI scale factor.
    pub dpi_scale: f32,
    /// Position of the monitor's top-left corner in virtual screen space.
    pub position: Vec2,
    /// Whether this is the primary monitor.
    pub is_primary: bool,
}

impl MonitorInfo {
    /// Create a default monitor info for the primary monitor.
    pub fn default_primary() -> Self {
        Self {
            name: "Primary".to_string(),
            width: 1920,
            height: 1080,
            refresh_rate: 60,
            dpi_scale: 1.0,
            position: Vec2::ZERO,
            is_primary: true,
        }
    }

    /// Logical size (physical size / DPI scale).
    pub fn logical_size(&self) -> Vec2 {
        Vec2::new(
            self.width as f32 / self.dpi_scale,
            self.height as f32 / self.dpi_scale,
        )
    }

    /// Physical size as Vec2.
    pub fn physical_size(&self) -> Vec2 {
        Vec2::new(self.width as f32, self.height as f32)
    }
}

// ---------------------------------------------------------------------------
// SystemInfo
// ---------------------------------------------------------------------------

/// System information queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system name (e.g., "Windows 11").
    pub os_name: String,
    /// OS version string.
    pub os_version: String,
    /// Number of connected displays.
    pub display_count: u32,
    /// Total system RAM in megabytes.
    pub total_ram_mb: u64,
    /// GPU name (primary adapter).
    pub gpu_name: String,
    /// Number of logical CPU cores.
    pub cpu_cores: u32,
    /// CPU name/model.
    pub cpu_name: String,
}

impl SystemInfo {
    /// Query the system for information.
    pub fn query() -> Self {
        Self {
            os_name: std::env::consts::OS.to_string(),
            os_version: "unknown".to_string(),
            display_count: 1,
            total_ram_mb: 0,
            gpu_name: "unknown".to_string(),
            cpu_cores: 1,
            cpu_name: std::env::consts::ARCH.to_string(),
        }
    }

    /// Whether the OS is Windows.
    pub fn is_windows(&self) -> bool {
        self.os_name == "windows"
    }

    /// Whether the OS is macOS.
    pub fn is_macos(&self) -> bool {
        self.os_name == "macos"
    }

    /// Whether the OS is Linux.
    pub fn is_linux(&self) -> bool {
        self.os_name == "linux"
    }
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self::query()
    }
}

// ---------------------------------------------------------------------------
// PlatformContext
// ---------------------------------------------------------------------------

/// Combined platform context providing access to all platform services.
pub struct PlatformContext {
    /// Window manager.
    pub windows: WindowManager,
    /// Cursor manager.
    pub cursors: CursorManager,
    /// Clipboard manager.
    pub clipboard: ClipboardManager,
    /// DPI manager.
    pub dpi: DPIManager,
    /// System information (cached at startup).
    pub system_info: SystemInfo,
    /// Connected monitors.
    pub monitors: Vec<MonitorInfo>,
}

impl PlatformContext {
    /// Create a new platform context with default settings.
    pub fn new() -> Self {
        Self {
            windows: WindowManager::new(),
            cursors: CursorManager::new(),
            clipboard: ClipboardManager::new(),
            dpi: DPIManager::new(),
            system_info: SystemInfo::query(),
            monitors: vec![MonitorInfo::default_primary()],
        }
    }

    /// Set the list of connected monitors.
    pub fn set_monitors(&mut self, monitors: Vec<MonitorInfo>) {
        let scales: Vec<f32> =
            monitors.iter().map(|m| m.dpi_scale).collect();
        self.dpi.set_monitor_scales(scales);
        self.monitors = monitors;
    }

    /// Get the primary monitor info.
    pub fn primary_monitor(&self) -> Option<&MonitorInfo> {
        self.monitors.iter().find(|m| m.is_primary)
    }

    /// Get a monitor by index.
    pub fn monitor(&self, index: usize) -> Option<&MonitorInfo> {
        self.monitors.get(index)
    }

    /// Number of connected monitors.
    pub fn monitor_count(&self) -> usize {
        self.monitors.len()
    }

    /// Process pending window events.
    pub fn process_events(&mut self) -> Vec<WindowEvent> {
        self.windows.take_events()
    }
}

impl Default for PlatformContext {
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
    fn test_window_properties_default() {
        let props = WindowProperties::default();
        assert_eq!(props.width, 1280.0);
        assert_eq!(props.height, 720.0);
        assert!(props.resizable);
        assert!(props.decorated);
    }

    #[test]
    fn test_window_properties_popup() {
        let props = WindowProperties::popup(200.0, 300.0);
        assert!(!props.decorated);
        assert!(!props.resizable);
        assert_eq!(props.window_type, WindowType::Popup);
    }

    #[test]
    fn test_window_manager_create_destroy() {
        let mut wm = WindowManager::new();
        let id = wm.create_window(WindowProperties::default());
        assert!(id.is_valid());
        assert_eq!(wm.window_count(), 1);

        let state = wm.get_window(id).unwrap();
        assert_eq!(state.properties.title, "Genovo");

        assert!(wm.destroy_window(id));
        assert_eq!(wm.window_count(), 0);
    }

    #[test]
    fn test_window_manager_focus() {
        let mut wm = WindowManager::new();
        let id1 = wm.create_window(WindowProperties::normal("W1", 800.0, 600.0));
        let id2 = wm.create_window(WindowProperties::normal("W2", 800.0, 600.0));

        wm.focus_window(id1);
        assert_eq!(wm.focused_window(), id1);

        wm.focus_window(id2);
        assert_eq!(wm.focused_window(), id2);
    }

    #[test]
    fn test_window_manager_main_window() {
        let mut wm = WindowManager::new();
        let id = wm.create_window(WindowProperties::default());
        assert_eq!(wm.main_window(), id);
    }

    #[test]
    fn test_cursor_manager() {
        let mut cm = CursorManager::new();
        assert_eq!(cm.current_cursor, CursorType::Arrow);
        assert!(cm.visible);

        cm.set_cursor(CursorType::Hand);
        assert_eq!(cm.current_cursor, CursorType::Hand);

        cm.push_cursor(CursorType::IBeam);
        assert_eq!(cm.current_cursor, CursorType::IBeam);

        cm.pop_cursor();
        assert_eq!(cm.current_cursor, CursorType::Hand);

        cm.hide_cursor();
        assert!(!cm.visible);

        cm.show_cursor();
        assert!(cm.visible);
    }

    #[test]
    fn test_cursor_lock() {
        let mut cm = CursorManager::new();
        cm.lock_cursor(Vec2::new(400.0, 300.0));
        assert!(cm.locked);
        assert_eq!(cm.lock_center, Vec2::new(400.0, 300.0));

        cm.unlock_cursor();
        assert!(!cm.locked);
    }

    #[test]
    fn test_clipboard_manager() {
        let mut cb = ClipboardManager::new();
        assert!(!cb.has_text());

        cb.copy_text("Hello, world!");
        assert!(cb.has_text());

        let text = cb.paste_text().unwrap();
        assert_eq!(text, "Hello, world!");

        cb.clear();
        assert!(!cb.has_text());
    }

    #[test]
    fn test_dpi_manager() {
        let mut dpi = DPIManager::new();
        assert_eq!(dpi.get_dpi_scale(), 1.0);

        dpi.set_dpi_scale(2.0);
        assert_eq!(dpi.get_dpi_scale(), 2.0);

        let logical = Vec2::new(100.0, 50.0);
        let physical = dpi.logical_to_physical(logical);
        assert_eq!(physical, Vec2::new(200.0, 100.0));

        let back = dpi.physical_to_logical(physical);
        assert!((back - logical).length() < 0.01);
    }

    #[test]
    fn test_dpi_snap_to_pixel() {
        let dpi = DPIManager::new();
        let pos = Vec2::new(10.3, 20.7);
        let snapped = dpi.snap_to_pixel(pos);
        assert_eq!(snapped, Vec2::new(10.0, 21.0));
    }

    #[test]
    fn test_dpi_scale_2x() {
        let mut dpi = DPIManager::new();
        dpi.set_dpi_scale(2.0);

        let pos = Vec2::new(10.3, 20.7);
        let snapped = dpi.snap_to_pixel(pos);
        // At 2x: 10.3*2=20.6 -> round=21 -> /2=10.5
        // At 2x: 20.7*2=41.4 -> round=41 -> /2=20.5
        assert!((snapped.x - 10.5).abs() < 0.01);
        assert!((snapped.y - 20.5).abs() < 0.01);
    }

    #[test]
    fn test_file_filter() {
        let filter = FileFilter::images();
        assert!(filter.extensions.contains(&"png".to_string()));
        assert!(filter.extensions.contains(&"jpg".to_string()));
    }

    #[test]
    fn test_monitor_info() {
        let monitor = MonitorInfo::default_primary();
        assert!(monitor.is_primary);
        assert_eq!(monitor.width, 1920);
        assert_eq!(monitor.height, 1080);

        let logical = monitor.logical_size();
        assert_eq!(logical.x, 1920.0);
        assert_eq!(logical.y, 1080.0);
    }

    #[test]
    fn test_system_info() {
        let info = SystemInfo::query();
        assert!(!info.os_name.is_empty());
    }

    #[test]
    fn test_platform_context() {
        let mut ctx = PlatformContext::new();
        assert!(ctx.monitor_count() > 0);

        let id = ctx.windows.create_window(WindowProperties::default());
        assert!(id.is_valid());

        ctx.cursors.set_cursor(CursorType::IBeam);
        assert_eq!(ctx.cursors.current_cursor, CursorType::IBeam);

        ctx.clipboard.copy_text("test");
        assert_eq!(ctx.clipboard.paste_text().unwrap(), "test");
    }

    #[test]
    fn test_window_state_from_properties() {
        let props = WindowProperties::normal("Test", 800.0, 600.0)
            .with_position(100.0, 200.0);
        let state = WindowState::from_properties(WindowId(1), &props);
        assert_eq!(state.position, Vec2::new(100.0, 200.0));
        assert_eq!(state.size, Vec2::new(800.0, 600.0));
    }

    #[test]
    fn test_window_minimize_maximize() {
        let mut wm = WindowManager::new();
        let id = wm.create_window(WindowProperties::default());

        wm.minimize(id);
        assert!(wm.get_window(id).unwrap().minimized);

        wm.restore(id);
        assert!(!wm.get_window(id).unwrap().minimized);

        wm.maximize(id);
        assert!(wm.get_window(id).unwrap().maximized);
    }

    #[test]
    fn test_window_events() {
        let mut wm = WindowManager::new();
        let id = wm.create_window(WindowProperties::default());

        wm.push_event(WindowEvent::Resized {
            id,
            width: 1024.0,
            height: 768.0,
        });

        let events = wm.take_events();
        assert_eq!(events.len(), 1);

        let state = wm.get_window(id).unwrap();
        assert_eq!(state.size, Vec2::new(1024.0, 768.0));
    }

    #[test]
    fn test_image_data() {
        let img = ImageData::new(100, 50);
        assert_eq!(img.width, 100);
        assert_eq!(img.height, 50);
        assert_eq!(img.pixels.len(), 100 * 50 * 4);
    }

    #[test]
    fn test_dpi_monitor_scales() {
        let mut dpi = DPIManager::new();
        dpi.set_monitor_scales(vec![1.0, 1.5, 2.0]);
        assert_eq!(dpi.monitor_count(), 3);
        assert_eq!(dpi.monitor_scale(0), 1.0);
        assert_eq!(dpi.monitor_scale(1), 1.5);
        assert_eq!(dpi.monitor_scale(2), 2.0);
        assert_eq!(dpi.monitor_scale(99), 1.0); // fallback
    }
}
