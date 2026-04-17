// engine/editor/src/editor_preferences.rs
//
// Editor preferences for the Genovo editor.
//
// Manages user settings, UI layout, and editor configuration:
//
// - **User settings persistence** -- Save/load editor preferences.
// - **UI layout save/restore** -- Remember panel positions and sizes.
// - **Recent projects** -- Track recently opened projects.
// - **Recently opened files** -- Track recently opened assets/scenes.
// - **Editor theme selection** -- Light/dark/custom themes.
// - **Viewport settings** -- Camera speed, grid, gizmo settings.
// - **Grid settings** -- Grid size, snap, visibility.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_RECENT_PROJECTS: usize = 20;
const MAX_RECENT_FILES: usize = 50;
const PREFS_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Editor theme
// ---------------------------------------------------------------------------

/// Editor color theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditorTheme {
    Light,
    Dark,
    HighContrast,
    Custom(u32),
}

impl Default for EditorTheme {
    fn default() -> Self {
        Self::Dark
    }
}

impl fmt::Display for EditorTheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Light => write!(f, "Light"),
            Self::Dark => write!(f, "Dark"),
            Self::HighContrast => write!(f, "High Contrast"),
            Self::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// Theme color palette.
#[derive(Debug, Clone)]
pub struct ThemeColors {
    pub background: [f32; 4],
    pub foreground: [f32; 4],
    pub accent: [f32; 4],
    pub panel_background: [f32; 4],
    pub panel_border: [f32; 4],
    pub text: [f32; 4],
    pub text_dim: [f32; 4],
    pub selection: [f32; 4],
    pub hover: [f32; 4],
    pub error: [f32; 4],
    pub warning: [f32; 4],
    pub success: [f32; 4],
}

impl ThemeColors {
    pub fn dark() -> Self {
        Self {
            background: [0.15, 0.15, 0.15, 1.0],
            foreground: [0.2, 0.2, 0.2, 1.0],
            accent: [0.3, 0.5, 0.9, 1.0],
            panel_background: [0.18, 0.18, 0.18, 1.0],
            panel_border: [0.3, 0.3, 0.3, 1.0],
            text: [0.9, 0.9, 0.9, 1.0],
            text_dim: [0.5, 0.5, 0.5, 1.0],
            selection: [0.3, 0.5, 0.9, 0.5],
            hover: [0.25, 0.25, 0.3, 1.0],
            error: [0.9, 0.2, 0.2, 1.0],
            warning: [0.9, 0.7, 0.1, 1.0],
            success: [0.2, 0.8, 0.2, 1.0],
        }
    }

    pub fn light() -> Self {
        Self {
            background: [0.92, 0.92, 0.92, 1.0],
            foreground: [0.96, 0.96, 0.96, 1.0],
            accent: [0.2, 0.4, 0.8, 1.0],
            panel_background: [0.94, 0.94, 0.94, 1.0],
            panel_border: [0.75, 0.75, 0.75, 1.0],
            text: [0.1, 0.1, 0.1, 1.0],
            text_dim: [0.5, 0.5, 0.5, 1.0],
            selection: [0.3, 0.5, 0.9, 0.3],
            hover: [0.85, 0.85, 0.9, 1.0],
            error: [0.8, 0.1, 0.1, 1.0],
            warning: [0.8, 0.6, 0.0, 1.0],
            success: [0.1, 0.7, 0.1, 1.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Viewport settings
// ---------------------------------------------------------------------------

/// Viewport camera and display settings.
#[derive(Debug, Clone)]
pub struct ViewportSettings {
    pub camera_speed: f32,
    pub camera_sensitivity: f32,
    pub camera_fov: f32,
    pub camera_near: f32,
    pub camera_far: f32,
    pub show_grid: bool,
    pub show_gizmos: bool,
    pub show_wireframe: bool,
    pub show_bounds: bool,
    pub show_fps: bool,
    pub show_stats: bool,
    pub gizmo_size: f32,
    pub anti_aliasing: bool,
    pub vsync: bool,
}

impl Default for ViewportSettings {
    fn default() -> Self {
        Self {
            camera_speed: 5.0,
            camera_sensitivity: 0.3,
            camera_fov: 60.0,
            camera_near: 0.1,
            camera_far: 1000.0,
            show_grid: true,
            show_gizmos: true,
            show_wireframe: false,
            show_bounds: false,
            show_fps: true,
            show_stats: false,
            gizmo_size: 1.0,
            anti_aliasing: true,
            vsync: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Grid settings
// ---------------------------------------------------------------------------

/// Grid display and snap settings.
#[derive(Debug, Clone)]
pub struct GridSettings {
    pub visible: bool,
    pub snap_enabled: bool,
    pub grid_size: f32,
    pub snap_size: f32,
    pub major_line_interval: u32,
    pub grid_color: [f32; 4],
    pub major_color: [f32; 4],
    pub grid_plane: GridPlane,
    pub infinite: bool,
    pub fade_distance: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridPlane {
    XZ,
    XY,
    YZ,
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            visible: true,
            snap_enabled: true,
            grid_size: 1.0,
            snap_size: 0.25,
            major_line_interval: 10,
            grid_color: [0.3, 0.3, 0.3, 0.3],
            major_color: [0.4, 0.4, 0.4, 0.5],
            grid_plane: GridPlane::XZ,
            infinite: true,
            fade_distance: 100.0,
        }
    }
}

// ---------------------------------------------------------------------------
// UI layout
// ---------------------------------------------------------------------------

/// Panel position and size.
#[derive(Debug, Clone)]
pub struct PanelLayout {
    pub name: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub visible: bool,
    pub docked: bool,
    pub dock_side: DockSide,
    pub tab_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DockSide {
    Left,
    Right,
    Top,
    Bottom,
    Center,
    Floating,
}

impl Default for DockSide {
    fn default() -> Self {
        Self::Left
    }
}

/// Complete UI layout state.
#[derive(Debug, Clone, Default)]
pub struct UILayout {
    pub name: String,
    pub panels: Vec<PanelLayout>,
    pub window_width: u32,
    pub window_height: u32,
    pub window_maximized: bool,
}

impl UILayout {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Add a panel.
    pub fn add_panel(&mut self, panel: PanelLayout) {
        self.panels.push(panel);
    }

    /// Get a panel by name.
    pub fn panel(&self, name: &str) -> Option<&PanelLayout> {
        self.panels.iter().find(|p| p.name == name)
    }
}

// ---------------------------------------------------------------------------
// Recent entry
// ---------------------------------------------------------------------------

/// A recent project or file entry.
#[derive(Debug, Clone)]
pub struct RecentEntry {
    pub path: String,
    pub name: String,
    pub last_opened: f64,
    pub pinned: bool,
}

impl RecentEntry {
    pub fn new(path: &str, name: &str, timestamp: f64) -> Self {
        Self {
            path: path.to_string(),
            name: name.to_string(),
            last_opened: timestamp,
            pinned: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Editor preferences
// ---------------------------------------------------------------------------

/// Complete editor preferences.
#[derive(Debug, Clone)]
pub struct EditorPreferences {
    pub version: u32,
    pub theme: EditorTheme,
    pub theme_colors: ThemeColors,
    pub viewport: ViewportSettings,
    pub grid: GridSettings,
    pub layouts: Vec<UILayout>,
    pub active_layout: String,
    pub recent_projects: Vec<RecentEntry>,
    pub recent_files: Vec<RecentEntry>,
    pub font_size: f32,
    pub auto_save: bool,
    pub auto_save_interval: f32,
    pub undo_history_size: u32,
    pub show_tooltips: bool,
    pub language: String,
    pub custom: HashMap<String, String>,
}

impl Default for EditorPreferences {
    fn default() -> Self {
        Self {
            version: PREFS_VERSION,
            theme: EditorTheme::Dark,
            theme_colors: ThemeColors::dark(),
            viewport: ViewportSettings::default(),
            grid: GridSettings::default(),
            layouts: vec![UILayout::new("Default")],
            active_layout: "Default".to_string(),
            recent_projects: Vec::new(),
            recent_files: Vec::new(),
            font_size: 14.0,
            auto_save: true,
            auto_save_interval: 300.0,
            undo_history_size: 100,
            show_tooltips: true,
            language: "en".to_string(),
            custom: HashMap::new(),
        }
    }
}

impl EditorPreferences {
    /// Create new preferences.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the theme.
    pub fn set_theme(&mut self, theme: EditorTheme) {
        self.theme = theme;
        self.theme_colors = match theme {
            EditorTheme::Light => ThemeColors::light(),
            EditorTheme::Dark => ThemeColors::dark(),
            _ => self.theme_colors.clone(),
        };
    }

    /// Add a recent project.
    pub fn add_recent_project(&mut self, path: &str, name: &str, timestamp: f64) {
        self.recent_projects.retain(|p| p.path != path);
        self.recent_projects.insert(0, RecentEntry::new(path, name, timestamp));
        if self.recent_projects.len() > MAX_RECENT_PROJECTS {
            self.recent_projects
                .retain(|p| p.pinned || self.recent_projects.iter().position(|x| x.path == p.path).unwrap() < MAX_RECENT_PROJECTS);
        }
    }

    /// Add a recent file.
    pub fn add_recent_file(&mut self, path: &str, name: &str, timestamp: f64) {
        self.recent_files.retain(|f| f.path != path);
        self.recent_files.insert(0, RecentEntry::new(path, name, timestamp));
        if self.recent_files.len() > MAX_RECENT_FILES {
            self.recent_files.truncate(MAX_RECENT_FILES);
        }
    }

    /// Clear recent projects.
    pub fn clear_recent_projects(&mut self) {
        self.recent_projects.retain(|p| p.pinned);
    }

    /// Clear recent files.
    pub fn clear_recent_files(&mut self) {
        self.recent_files.clear();
    }

    /// Save a layout.
    pub fn save_layout(&mut self, layout: UILayout) {
        self.layouts.retain(|l| l.name != layout.name);
        self.layouts.push(layout);
    }

    /// Load a layout by name.
    pub fn load_layout(&mut self, name: &str) -> Option<&UILayout> {
        self.active_layout = name.to_string();
        self.layouts.iter().find(|l| l.name == name)
    }

    /// Set a custom preference.
    pub fn set_custom(&mut self, key: &str, value: &str) {
        self.custom.insert(key.to_string(), value.to_string());
    }

    /// Get a custom preference.
    pub fn get_custom(&self, key: &str) -> Option<&str> {
        self.custom.get(key).map(|s| s.as_str())
    }

    /// Serialize to a simple key-value format.
    pub fn serialize_simple(&self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        pairs.push(("version".to_string(), self.version.to_string()));
        pairs.push(("theme".to_string(), format!("{:?}", self.theme)));
        pairs.push(("font_size".to_string(), self.font_size.to_string()));
        pairs.push(("auto_save".to_string(), self.auto_save.to_string()));
        pairs.push(("auto_save_interval".to_string(), self.auto_save_interval.to_string()));
        pairs.push(("language".to_string(), self.language.clone()));
        pairs.push(("show_tooltips".to_string(), self.show_tooltips.to_string()));
        pairs.push(("camera_speed".to_string(), self.viewport.camera_speed.to_string()));
        pairs.push(("camera_fov".to_string(), self.viewport.camera_fov.to_string()));
        pairs.push(("grid_visible".to_string(), self.grid.visible.to_string()));
        pairs.push(("grid_size".to_string(), self.grid.grid_size.to_string()));
        pairs.push(("snap_enabled".to_string(), self.grid.snap_enabled.to_string()));
        pairs.push(("snap_size".to_string(), self.grid.snap_size.to_string()));
        for (k, v) in &self.custom {
            pairs.push((format!("custom.{}", k), v.clone()));
        }
        pairs
    }

    /// Number of recent projects.
    pub fn recent_project_count(&self) -> usize {
        self.recent_projects.len()
    }

    /// Number of recent files.
    pub fn recent_file_count(&self) -> usize {
        self.recent_files.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let prefs = EditorPreferences::new();
        assert_eq!(prefs.theme, EditorTheme::Dark);
        assert!(prefs.viewport.show_grid);
        assert!(prefs.grid.snap_enabled);
    }

    #[test]
    fn test_recent_projects() {
        let mut prefs = EditorPreferences::new();
        prefs.add_recent_project("/path/to/project1", "Project 1", 1.0);
        prefs.add_recent_project("/path/to/project2", "Project 2", 2.0);
        assert_eq!(prefs.recent_project_count(), 2);
        assert_eq!(prefs.recent_projects[0].name, "Project 2");
    }

    #[test]
    fn test_recent_dedup() {
        let mut prefs = EditorPreferences::new();
        prefs.add_recent_project("/path", "P1", 1.0);
        prefs.add_recent_project("/path", "P1", 2.0);
        assert_eq!(prefs.recent_project_count(), 1);
    }

    #[test]
    fn test_theme_switch() {
        let mut prefs = EditorPreferences::new();
        prefs.set_theme(EditorTheme::Light);
        assert!(prefs.theme_colors.background[0] > 0.8);
    }

    #[test]
    fn test_custom_prefs() {
        let mut prefs = EditorPreferences::new();
        prefs.set_custom("my_plugin.setting", "value");
        assert_eq!(prefs.get_custom("my_plugin.setting"), Some("value"));
    }

    #[test]
    fn test_serialization() {
        let prefs = EditorPreferences::new();
        let pairs = prefs.serialize_simple();
        assert!(!pairs.is_empty());
        assert!(pairs.iter().any(|(k, _)| k == "font_size"));
    }
}
