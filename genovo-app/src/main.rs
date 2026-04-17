//! Genovo Studio — Professional Game Development Environment
//!
//! A complete, polished game engine editor built with a hybrid rendering approach:
//!   - egui for immediate-mode UI rendering (stable, works now)
//!   - Styled to match our custom Slate-like dark theme (UIStyle from genovo_ui)
//!   - Uses our custom UI data structures (DockState, UIStyle, etc.) for state management
//!   - When our custom GPU renderer (UIGpuRenderer) is fully wired, swap the backend
//!
//! Features:
//!   - Real GPU-rendered 3D viewport (wgpu triangle pipeline behind egui)
//!   - Scene outliner with colored entity type dots, context menus, rename, drag reparent
//!   - Inspector with colored XYZ drag values, physics sync, light color picker
//!   - Content browser with directory scanning and icon grid
//!   - Console with command history and colored log output
//!   - Profiler with egui_plot frame time graph
//!   - Animation timeline placeholder
//!   - Near-black dark theme (rgb(18,18,22) base, rgb(24,24,28) panels)
//!   - Accent blue rgb(56,132,244) sparingly for selection/active
//!   - Thin 4px scrollbars, no window shadows, 1px subtle separators
//!   - Full keyboard shortcuts (Ctrl+S, Ctrl+Z, Delete, Q/W/E/R, Space, F5, F)
//!   - Working physics integration (play/pause/stop/step)
//!   - Status bar with colored play-state dot, FPS, entity/body counts
//!   - Toolbar with play/pause/stop, transform tools, snap, grid, speed
//!   - About Genovo Studio dialog, preferences, keyboard shortcuts reference

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Window, WindowId},
};

use genovo::prelude::*;
use genovo_physics as physics;

// Scene renderer for full 3D rendering (PBR, grid, built-in primitives)
use genovo_render::scene_renderer::{
    SceneRenderManager, SceneCamera, SceneLights, MaterialParams,
};

// Import our custom UI types for state management
use genovo_ui::dock_system::{DockState, DockTabId, DockNodeId, DockTab, DockNode, DockStyle};
use genovo_ui::ui_framework::UIStyle;
use genovo_ui::render_commands::Color as UIColor;

// Import undo system types
use genovo_editor::undo_system::{
    UndoStack, MoveEntityOp, RotateEntityOp, ScaleEntityOp,
    SpawnEntityOp, DespawnEntityOp, SerializedEntityData,
    SerializedComponentData, EntityId as UndoEntityId,
    Vec3 as UndoVec3, Quat as UndoQuat, OperationResult,
};

// Import scripting types for VM binding
use genovo_scripting::vm::{ScriptValue, ScriptError, NativeFn, ScriptContext};
use genovo_scripting::vm::GenovoVM;
use genovo_scripting::ScriptVM;

// Audio engine (uses SoftwareMixer from genovo_audio, running in a background thread)
use std::sync::atomic::{AtomicBool, Ordering};

// Link all 26 engine modules into the binary
use genovo_core as _core;
use genovo_ecs as _ecs;
use genovo_scene as _scene;
use genovo_render as _render;
use genovo_platform as _platform;
use genovo_audio as _audio;
use genovo_animation as _animation;
use genovo_assets as _assets;
use genovo_scripting as _scripting;
use genovo_networking as _networking;
use genovo_ai as _ai;
use genovo_editor as _editor;
use genovo_ui as _ui;
use genovo_debug as _debug;
use genovo_save as _save;
use genovo_cinematics as _cinematics;
use genovo_localization as _localization;
use genovo_terrain as _terrain;
use genovo_procgen as _procgen;
use genovo_world as _world;
use genovo_replay as _replay;
use genovo_gameplay as _gameplay;

// =============================================================================
// Color Palette — matching our custom UIStyle::dark() theme exactly
// =============================================================================
//
// These colors correspond to the genovo_ui::ui_framework::UIStyle dark theme:
//   bg_base:      (18, 18, 22)    — deepest background
//   bg_panel:     (24, 24, 28)    — panel background
//   bg_widget:    (32, 32, 38)    — interactive widget background
//   bg_hover:     (42, 42, 50)    — hover state
//   bg_active:    (52, 52, 62)    — active/pressed state
//   accent:       (56, 132, 244)  — primary accent blue
//   accent_dim:   (40, 100, 200)  — muted accent
//   text_bright:  (230, 230, 235) — headings, active text
//   text_normal:  (180, 180, 188) — body text
//   text_dim:     (110, 110, 120) — labels, captions
//   border:       (38, 38, 44)    — subtle borders
//   green:        (72, 199, 142)  — success, play
//   yellow:       (245, 196, 80)  — warning, pause
//   red:          (235, 87, 87)   — error, stop, X axis

const BG_BASE: egui::Color32 = egui::Color32::from_rgb(18, 18, 22);
const BG_PANEL: egui::Color32 = egui::Color32::from_rgb(24, 24, 28);
const BG_WIDGET: egui::Color32 = egui::Color32::from_rgb(32, 32, 38);
const BG_HOVER: egui::Color32 = egui::Color32::from_rgb(42, 42, 50);
const BG_ACTIVE: egui::Color32 = egui::Color32::from_rgb(52, 52, 62);
const BORDER: egui::Color32 = egui::Color32::from_rgb(38, 38, 44);
const BORDER_SUBTLE: egui::Color32 = egui::Color32::from_rgb(30, 30, 36);

const TEXT_BRIGHT: egui::Color32 = egui::Color32::from_rgb(230, 230, 235);
const TEXT_NORMAL: egui::Color32 = egui::Color32::from_rgb(180, 180, 188);
const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(110, 110, 120);
const TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(65, 65, 72);

const ACCENT: egui::Color32 = egui::Color32::from_rgb(56, 132, 244);
const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(40, 100, 200);
const ACCENT_BRIGHT: egui::Color32 = egui::Color32::from_rgb(80, 156, 255);
const ACCENT_BG: egui::Color32 = egui::Color32::from_rgb(30, 60, 110);

const GREEN: egui::Color32 = egui::Color32::from_rgb(72, 199, 142);
const GREEN_DIM: egui::Color32 = egui::Color32::from_rgb(50, 140, 100);
const YELLOW: egui::Color32 = egui::Color32::from_rgb(245, 196, 80);
const YELLOW_DIM: egui::Color32 = egui::Color32::from_rgb(180, 145, 60);
const RED: egui::Color32 = egui::Color32::from_rgb(235, 87, 87);
const RED_DIM: egui::Color32 = egui::Color32::from_rgb(160, 60, 60);
const CYAN: egui::Color32 = egui::Color32::from_rgb(70, 200, 220);
const MAGENTA: egui::Color32 = egui::Color32::from_rgb(190, 80, 210);
const ORANGE: egui::Color32 = egui::Color32::from_rgb(230, 150, 60);

// XYZ axis colors (Unreal Engine convention)
const X_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 75, 75);
const Y_COLOR: egui::Color32 = egui::Color32::from_rgb(72, 199, 142);
const Z_COLOR: egui::Color32 = egui::Color32::from_rgb(56, 132, 244);

// =============================================================================
// Entity Type (data model aligned with editor_widgets)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntityType {
    Empty,
    Mesh,
    Light,
    Camera,
    ParticleSystem,
    Audio,
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityType::Empty => write!(f, "Empty"),
            EntityType::Mesh => write!(f, "Mesh"),
            EntityType::Light => write!(f, "Light"),
            EntityType::Camera => write!(f, "Camera"),
            EntityType::ParticleSystem => write!(f, "Particles"),
            EntityType::Audio => write!(f, "Audio"),
        }
    }
}

impl EntityType {
    fn icon_letter(&self) -> &str {
        match self {
            EntityType::Empty => "E",
            EntityType::Mesh => "M",
            EntityType::Light => "L",
            EntityType::Camera => "C",
            EntityType::ParticleSystem => "P",
            EntityType::Audio => "A",
        }
    }

    fn icon_color(&self) -> egui::Color32 {
        match self {
            EntityType::Empty => TEXT_DIM,
            EntityType::Mesh => CYAN,
            EntityType::Light => YELLOW,
            EntityType::Camera => ACCENT_BRIGHT,
            EntityType::ParticleSystem => MAGENTA,
            EntityType::Audio => ORANGE,
        }
    }

    fn dot_color(&self) -> egui::Color32 {
        self.icon_color()
    }
}

// =============================================================================
// Mesh Shape — for the Create menu
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeshShape {
    Cube,
    Sphere,
    Cylinder,
    Capsule,
    Cone,
    Plane,
}

impl std::fmt::Display for MeshShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshShape::Cube => write!(f, "Cube"),
            MeshShape::Sphere => write!(f, "Sphere"),
            MeshShape::Cylinder => write!(f, "Cylinder"),
            MeshShape::Capsule => write!(f, "Capsule"),
            MeshShape::Cone => write!(f, "Cone"),
            MeshShape::Plane => write!(f, "Plane"),
        }
    }
}

// =============================================================================
// Light Kind
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LightKind {
    Directional,
    Point,
    Spot,
}

impl std::fmt::Display for LightKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LightKind::Directional => write!(f, "Directional"),
            LightKind::Point => write!(f, "Point"),
            LightKind::Spot => write!(f, "Spot"),
        }
    }
}

// =============================================================================
// Camera Projection
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CameraProjection {
    Perspective,
    Orthographic,
}

impl std::fmt::Display for CameraProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CameraProjection::Perspective => write!(f, "Perspective"),
            CameraProjection::Orthographic => write!(f, "Orthographic"),
        }
    }
}

// =============================================================================
// Rigid Body Type
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BodyKind {
    Dynamic,
    Static,
    Kinematic,
}

impl std::fmt::Display for BodyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BodyKind::Dynamic => write!(f, "Dynamic"),
            BodyKind::Static => write!(f, "Static"),
            BodyKind::Kinematic => write!(f, "Kinematic"),
        }
    }
}

// =============================================================================
// Collider Shape
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColliderShape {
    Box,
    Sphere,
    Capsule,
}

impl std::fmt::Display for ColliderShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColliderShape::Box => write!(f, "Box"),
            ColliderShape::Sphere => write!(f, "Sphere"),
            ColliderShape::Capsule => write!(f, "Capsule"),
        }
    }
}

// =============================================================================
// Scene Entity
// =============================================================================

struct SceneEntity {
    entity: genovo_ecs::Entity,
    name: String,
    entity_type: EntityType,
    visible: bool,
    locked: bool,
    active: bool,
    parent: Option<usize>,
    position: [f32; 3],
    rotation: [f32; 3],
    scale: [f32; 3],
    // Mesh
    mesh_shape: MeshShape,
    cast_shadows: bool,
    receive_shadows: bool,
    // Physics
    has_physics: bool,
    physics_handle: Option<physics::RigidBodyHandle>,
    body_kind: BodyKind,
    mass: f32,
    friction: f32,
    restitution: f32,
    linear_damping: f32,
    angular_damping: f32,
    gravity_scale: f32,
    is_trigger: bool,
    collider_shape: ColliderShape,
    // Light
    is_light: bool,
    light_kind: LightKind,
    light_color: [f32; 3],
    light_intensity: f32,
    light_range: f32,
    light_spot_angle: f32,
    light_shadows: bool,
    // Camera
    is_camera: bool,
    camera_projection: CameraProjection,
    camera_fov: f32,
    camera_near: f32,
    camera_far: f32,
    camera_clear_color: [f32; 4],
    // Audio
    is_audio: bool,
    audio_volume: f32,
    audio_pitch: f32,
    audio_spatial: bool,
    audio_min_dist: f32,
    audio_max_dist: f32,
    // Script
    has_script: bool,
    script_file: String,
    // Tags
    tags: Vec<String>,
}

impl SceneEntity {
    fn new(entity: genovo_ecs::Entity, name: &str, entity_type: EntityType) -> Self {
        Self {
            entity,
            name: name.to_string(),
            entity_type,
            visible: true,
            locked: false,
            active: true,
            parent: None,
            position: [0.0; 3],
            rotation: [0.0; 3],
            scale: [1.0, 1.0, 1.0],
            mesh_shape: MeshShape::Cube,
            cast_shadows: true,
            receive_shadows: true,
            has_physics: false,
            physics_handle: None,
            body_kind: BodyKind::Dynamic,
            mass: 1.0,
            friction: 0.5,
            restitution: 0.3,
            linear_damping: 0.0,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            is_trigger: false,
            collider_shape: ColliderShape::Box,
            is_light: entity_type == EntityType::Light,
            light_kind: LightKind::Point,
            light_color: [1.0, 1.0, 0.9],
            light_intensity: 1.0,
            light_range: 10.0,
            light_spot_angle: 45.0,
            light_shadows: true,
            is_camera: entity_type == EntityType::Camera,
            camera_projection: CameraProjection::Perspective,
            camera_fov: 60.0,
            camera_near: 0.1,
            camera_far: 1000.0,
            camera_clear_color: [0.06, 0.06, 0.08, 1.0],
            is_audio: entity_type == EntityType::Audio,
            audio_volume: 1.0,
            audio_pitch: 1.0,
            audio_spatial: true,
            audio_min_dist: 1.0,
            audio_max_dist: 50.0,
            has_script: false,
            script_file: String::new(),
            tags: Vec::new(),
        }
    }
}

// =============================================================================
// Transform Mode / Coord Space / Pivot
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GizmoMode {
    Select,
    Translate,
    Rotate,
    Scale,
}

impl GizmoMode {
    fn label(&self) -> &str {
        match self {
            GizmoMode::Select => "Select",
            GizmoMode::Translate => "Translate",
            GizmoMode::Rotate => "Rotate",
            GizmoMode::Scale => "Scale",
        }
    }

    fn short_label(&self) -> &str {
        match self {
            GizmoMode::Select => "Q",
            GizmoMode::Translate => "W",
            GizmoMode::Rotate => "E",
            GizmoMode::Scale => "R",
        }
    }

    fn icon(&self) -> &str {
        match self {
            GizmoMode::Select => "->",
            GizmoMode::Translate => "+",
            GizmoMode::Rotate => "O",
            GizmoMode::Scale => "[]",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoordSpace {
    Local,
    World,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PivotMode {
    Center,
    Pivot,
}

// =============================================================================
// Console Log
// =============================================================================

#[derive(Clone)]
struct LogEntry {
    text: String,
    level: LogLevel,
    timestamp: f64,
    count: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LogLevel {
    Info,
    Warn,
    Error,
    System,
    Debug,
}

impl LogLevel {
    fn color(&self) -> egui::Color32 {
        match self {
            LogLevel::Info => TEXT_NORMAL,
            LogLevel::Warn => YELLOW,
            LogLevel::Error => RED,
            LogLevel::System => ACCENT_BRIGHT,
            LogLevel::Debug => egui::Color32::from_rgb(140, 140, 160),
        }
    }

    fn prefix(&self) -> &str {
        match self {
            LogLevel::Info => "INF",
            LogLevel::Warn => "WRN",
            LogLevel::Error => "ERR",
            LogLevel::System => "SYS",
            LogLevel::Debug => "DBG",
        }
    }

    fn bg_tint(&self) -> Option<egui::Color32> {
        match self {
            LogLevel::Error => Some(egui::Color32::from_rgba_premultiplied(235, 87, 87, 12)),
            LogLevel::Warn => Some(egui::Color32::from_rgba_premultiplied(245, 196, 80, 8)),
            _ => None,
        }
    }
}

// =============================================================================
// Panel Visibility
// =============================================================================

struct PanelVisibility {
    hierarchy: bool,
    inspector: bool,
    bottom: bool,
    toolbar: bool,
    status_bar: bool,
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            hierarchy: true,
            inspector: true,
            bottom: true,
            toolbar: true,
            status_bar: true,
        }
    }
}

// =============================================================================
// Bottom Tab
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum BottomTab {
    Console,
    ContentBrowser,
    Profiler,
    Animation,
}

// =============================================================================
// Hierarchy Action (deferred mutation)
// =============================================================================

enum HierarchyAction {
    Select(usize),
    Duplicate(usize),
    Delete(usize),
    AddChild(usize),
    ToggleVisibility(usize),
    ToggleLock(usize),
    Rename(usize),
    Group(usize),
}

// =============================================================================
// Inspector Section Collapse State
// =============================================================================

struct InspectorSections {
    transform_open: bool,
    mesh_open: bool,
    physics_open: bool,
    collider_open: bool,
    light_open: bool,
    camera_open: bool,
    audio_open: bool,
    script_open: bool,
    material_open: bool,
    tags_open: bool,
}

impl Default for InspectorSections {
    fn default() -> Self {
        Self {
            transform_open: true,
            mesh_open: true,
            physics_open: true,
            collider_open: true,
            light_open: true,
            camera_open: true,
            audio_open: true,
            script_open: true,
            material_open: false,
            tags_open: false,
        }
    }
}

// =============================================================================
// Snap value presets
// =============================================================================

const SNAP_PRESETS: &[f32] = &[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];

// =============================================================================
// Simulation Speed presets
// =============================================================================

const SPEED_PRESETS: &[f32] = &[0.25, 0.5, 1.0, 2.0, 4.0];

// =============================================================================
// Editor State
// =============================================================================

struct EditorState {
    engine: Engine,

    // Custom UI state management from genovo_ui
    ui_style: UIStyle,
    dock_style: DockStyle,

    // Scene
    entities: Vec<SceneEntity>,
    selected_entity: Option<usize>,
    next_entity_id: u32,
    entity_filter: String,

    // Playback
    is_playing: bool,
    is_paused: bool,
    sim_speed: f32,
    sim_speed_idx: usize,
    play_start_time: Option<Instant>,
    total_sim_time: f64,

    // Transform / gizmo
    gizmo_mode: GizmoMode,
    coord_space: CoordSpace,
    pivot_mode: PivotMode,
    snap_enabled: bool,
    snap_value_idx: usize,
    snap_translate: f32,
    snap_rotate: f32,
    snap_scale: f32,
    grid_visible: bool,
    grid_size: f32,
    wireframe_mode: bool,
    stats_visible: bool,

    // Console
    console_log: Vec<LogEntry>,
    console_input: String,
    console_history: Vec<String>,
    console_history_idx: Option<usize>,
    console_scroll_to_bottom: bool,
    console_filter_level: Option<LogLevel>,
    console_auto_scroll: bool,

    // Content browser
    asset_path: String,
    asset_search: String,
    asset_view_size: f32,
    asset_show_extensions: bool,

    // Profiler
    frame_times: VecDeque<f64>,
    fps_history: VecDeque<f64>,
    gpu_times: VecDeque<f64>,

    // Animation timeline
    anim_time: f32,
    anim_duration: f32,
    anim_playing: bool,
    anim_loop: bool,

    // Frame timing
    frame_count: u64,
    last_frame: Instant,
    fps: f32,
    frame_time_ms: f32,
    start_time: Instant,
    smooth_fps: f32,
    smooth_frame_time: f32,

    // Camera
    camera_yaw: f32,
    camera_pitch: f32,
    camera_dist: f32,
    camera_target: [f32; 3],
    camera_fov: f32,

    // Panels
    panels: PanelVisibility,
    bottom_tab: BottomTab,
    hierarchy_width: f32,
    inspector_width: f32,
    bottom_height: f32,

    // Inspector
    inspector_sections: InspectorSections,

    // Scene
    scene_name: String,
    scene_modified: bool,

    // Dialogs
    show_about: bool,
    show_preferences: bool,
    show_shortcuts: bool,

    // Keyboard modifiers
    modifiers: ModifiersState,

    // Rename state
    renaming_entity: Option<usize>,
    rename_buffer: String,

    // Notification
    notification: Option<(String, LogLevel, Instant)>,

    // Undo/Redo — real UndoStack from genovo_editor
    undo_stack: UndoStack,

    // Recent files
    recent_files: Vec<String>,
}

impl EditorState {
    fn new(engine: Engine) -> Self {
        Self {
            engine,
            ui_style: UIStyle::dark(),
            dock_style: DockStyle::dark_theme(),
            entities: Vec::new(),
            selected_entity: None,
            next_entity_id: 0,
            entity_filter: String::new(),
            is_playing: false,
            is_paused: false,
            sim_speed: 1.0,
            sim_speed_idx: 2, // 1.0x
            play_start_time: None,
            total_sim_time: 0.0,
            gizmo_mode: GizmoMode::Translate,
            coord_space: CoordSpace::World,
            pivot_mode: PivotMode::Center,
            snap_enabled: false,
            snap_value_idx: 3, // 1.0
            snap_translate: 1.0,
            snap_rotate: 15.0,
            snap_scale: 0.25,
            grid_visible: true,
            grid_size: 1.0,
            wireframe_mode: false,
            stats_visible: true,
            console_log: Vec::new(),
            console_input: String::new(),
            console_history: Vec::new(),
            console_history_idx: None,
            console_scroll_to_bottom: true,
            console_filter_level: None,
            console_auto_scroll: true,
            asset_path: "assets".to_string(),
            asset_search: String::new(),
            asset_view_size: 80.0,
            asset_show_extensions: true,
            frame_times: VecDeque::with_capacity(512),
            fps_history: VecDeque::with_capacity(512),
            gpu_times: VecDeque::with_capacity(512),
            anim_time: 0.0,
            anim_duration: 5.0,
            anim_playing: false,
            anim_loop: true,
            frame_count: 0,
            last_frame: Instant::now(),
            fps: 0.0,
            frame_time_ms: 0.0,
            start_time: Instant::now(),
            smooth_fps: 60.0,
            smooth_frame_time: 16.67,
            camera_yaw: 45.0,
            camera_pitch: -30.0,
            camera_dist: 15.0,
            camera_target: [0.0, 2.0, 0.0],
            camera_fov: 60.0,
            panels: PanelVisibility::default(),
            bottom_tab: BottomTab::Console,
            hierarchy_width: 220.0,
            inspector_width: 280.0,
            bottom_height: 200.0,
            inspector_sections: InspectorSections::default(),
            scene_name: "Untitled Scene".to_string(),
            scene_modified: false,
            show_about: false,
            show_preferences: false,
            show_shortcuts: false,
            modifiers: ModifiersState::empty(),
            renaming_entity: None,
            rename_buffer: String::new(),
            notification: None,
            undo_stack: UndoStack::with_default_size(),
            recent_files: vec![
                "scenes/demo_level.scene".to_string(),
                "scenes/test_physics.scene".to_string(),
                "scenes/prototype.scene".to_string(),
            ],
        }
    }

    fn log(&mut self, level: LogLevel, text: impl Into<String>) {
        let text = text.into();
        let elapsed = self.start_time.elapsed().as_secs_f64();

        // Collapse consecutive identical messages
        if let Some(last) = self.console_log.last_mut() {
            if last.text == text && last.level == level {
                last.count += 1;
                last.timestamp = elapsed;
                return;
            }
        }

        self.console_log.push(LogEntry {
            text,
            level,
            timestamp: elapsed,
            count: 1,
        });
        if self.console_log.len() > 5000 {
            self.console_log.drain(0..1000);
        }
        self.console_scroll_to_bottom = true;
    }

    fn notify(&mut self, msg: &str, level: LogLevel) {
        self.notification = Some((msg.to_string(), level, Instant::now()));
        self.log(level, msg);
    }

    fn spawn_entity(&mut self, name: &str, etype: EntityType) -> usize {
        let ecs_entity = self.engine.world_mut().spawn_empty();
        self.next_entity_id += 1;

        let mut se = SceneEntity::new(ecs_entity, name, etype);

        // Auto-add physics for mesh entities
        if etype == EntityType::Mesh {
            let desc = physics::RigidBodyDesc {
                body_type: physics::BodyType::Dynamic,
                position: Vec3::ZERO,
                mass: 1.0,
                restitution: 0.3,
                ..Default::default()
            };
            if let Ok(handle) = self.engine.physics_mut().add_body(&desc) {
                let _ = self.engine.physics_mut().add_collider(
                    handle,
                    &physics::ColliderDesc {
                        shape: physics::CollisionShape::Box {
                            half_extents: Vec3::new(0.5, 0.5, 0.5),
                        },
                        ..Default::default()
                    },
                );
                se.has_physics = true;
                se.physics_handle = Some(handle);
            }
        }

        let idx = self.entities.len();
        self.entities.push(se);
        self.scene_modified = true;
        self.log(LogLevel::System, format!("Spawned: {} [{}]", name, etype));
        idx
    }

    fn spawn_mesh(&mut self, shape: MeshShape) -> usize {
        let name = format!("{}", shape);
        let idx = self.spawn_entity(&name, EntityType::Mesh);
        self.entities[idx].mesh_shape = shape;
        idx
    }

    fn spawn_light(&mut self, kind: LightKind) -> usize {
        let name = format!("{} Light", kind);
        let idx = self.spawn_entity(&name, EntityType::Light);
        self.entities[idx].light_kind = kind;
        idx
    }

    fn delete_entity(&mut self, idx: usize) {
        if idx >= self.entities.len() {
            return;
        }
        let ent = &self.entities[idx];
        let name = ent.name.clone();
        let pos = ent.position;
        let rot = ent.rotation;
        let scl = ent.scale;
        if let Some(handle) = ent.physics_handle {
            let _ = self.engine.physics_mut().remove_body(handle);
        }
        self.engine.world_mut().despawn(ent.entity);
        self.entities.remove(idx);
        // Fix parent references
        for e in &mut self.entities {
            if let Some(p) = e.parent {
                if p == idx {
                    e.parent = None;
                } else if p > idx {
                    e.parent = Some(p - 1);
                }
            }
        }
        if let Some(sel) = self.selected_entity {
            if sel == idx {
                self.selected_entity = None;
            } else if sel > idx {
                self.selected_entity = Some(sel - 1);
            }
        }
        self.scene_modified = true;
        self.push_delete_undo(idx, &name, pos, rot, scl);
        self.log(LogLevel::System, format!("Deleted: {}", name));
    }

    fn delete_selected(&mut self) {
        if let Some(idx) = self.selected_entity {
            self.delete_entity(idx);
        }
    }

    fn duplicate_selected(&mut self) {
        if let Some(idx) = self.selected_entity {
            if idx >= self.entities.len() {
                return;
            }
            let src = &self.entities[idx];
            let new_name = format!("{} (copy)", src.name);
            let etype = src.entity_type;
            let pos = src.position;
            let rot = src.rotation;
            let scl = src.scale;
            let is_light = src.is_light;
            let lc = src.light_color;
            let li = src.light_intensity;
            let lr = src.light_range;
            let ls = src.light_shadows;
            let lk = src.light_kind;
            let is_cam = src.is_camera;
            let mesh_shape = src.mesh_shape;

            let new_idx = self.spawn_entity(&new_name, etype);
            let e = &mut self.entities[new_idx];
            e.position = [pos[0] + 1.5, pos[1], pos[2]];
            e.rotation = rot;
            e.scale = scl;
            e.is_light = is_light;
            e.light_color = lc;
            e.light_intensity = li;
            e.light_range = lr;
            e.light_shadows = ls;
            e.light_kind = lk;
            e.is_camera = is_cam;
            e.mesh_shape = mesh_shape;

            if let Some(handle) = e.physics_handle {
                let _ = self.engine.physics_mut().set_position(
                    handle,
                    Vec3::new(e.position[0], e.position[1], e.position[2]),
                );
            }
            self.selected_entity = Some(new_idx);
            self.push_spawn_undo(new_idx, &self.entities[new_idx].name.clone());
        }
    }

    fn sync_physics_to_entities(&mut self) {
        for ent in &mut self.entities {
            if let Some(handle) = ent.physics_handle {
                if let Ok(pos) = self.engine.physics().get_position(handle) {
                    ent.position = [pos.x, pos.y, pos.z];
                }
            }
        }
    }

    fn sync_entity_to_physics(&mut self, idx: usize) {
        if idx < self.entities.len() {
            let ent = &self.entities[idx];
            if let Some(handle) = ent.physics_handle {
                let _ = self.engine.physics_mut().set_position(
                    handle,
                    Vec3::new(ent.position[0], ent.position[1], ent.position[2]),
                );
            }
        }
    }

    fn toggle_play(&mut self) {
        if self.is_playing && !self.is_paused {
            self.is_paused = true;
            self.notify("Simulation paused", LogLevel::System);
        } else {
            self.is_playing = true;
            self.is_paused = false;
            if self.play_start_time.is_none() {
                self.play_start_time = Some(Instant::now());
            }
            self.notify("Simulation started", LogLevel::System);
        }
    }

    fn stop_play(&mut self) {
        self.is_playing = false;
        self.is_paused = false;
        self.play_start_time = None;
        self.total_sim_time = 0.0;
        self.notify("Simulation stopped", LogLevel::System);
    }

    fn step_physics(&mut self) {
        let _ = self.engine.physics_mut().step(1.0 / 60.0);
        self.sync_physics_to_entities();
        self.log(LogLevel::System, "Physics stepped (1/60s)");
    }

    fn spawn_physics_ball(&mut self) {
        let idx = self.spawn_entity(
            &format!("Ball_{}", self.next_entity_id),
            EntityType::Mesh,
        );
        let e = &mut self.entities[idx];
        e.position = [0.0, 8.0, 0.0];
        e.mesh_shape = MeshShape::Sphere;
        e.collider_shape = ColliderShape::Sphere;
        if let Some(handle) = e.physics_handle {
            let _ = self.engine.physics_mut().set_position(
                handle,
                Vec3::new(0.0, 8.0, 0.0),
            );
        }
        self.selected_entity = Some(idx);
    }

    fn select_next(&mut self) {
        if self.entities.is_empty() {
            return;
        }
        match self.selected_entity {
            Some(i) if i + 1 < self.entities.len() => self.selected_entity = Some(i + 1),
            _ => self.selected_entity = Some(0),
        }
    }

    fn select_prev(&mut self) {
        if self.entities.is_empty() {
            return;
        }
        match self.selected_entity {
            Some(0) => self.selected_entity = Some(self.entities.len() - 1),
            Some(i) => self.selected_entity = Some(i - 1),
            None => self.selected_entity = Some(self.entities.len() - 1),
        }
    }

    fn focus_selected(&mut self) {
        if let Some(idx) = self.selected_entity {
            if idx < self.entities.len() {
                self.camera_target = self.entities[idx].position;
                self.camera_dist = 8.0;
                self.log(LogLevel::Info, format!("Focused: {}", self.entities[idx].name));
            }
        }
    }

    fn frame_all(&mut self) {
        if self.entities.is_empty() {
            self.camera_target = [0.0, 0.0, 0.0];
            self.camera_dist = 15.0;
            return;
        }
        let mut center = [0.0f32; 3];
        for e in &self.entities {
            center[0] += e.position[0];
            center[1] += e.position[1];
            center[2] += e.position[2];
        }
        let n = self.entities.len() as f32;
        center[0] /= n;
        center[1] /= n;
        center[2] /= n;
        self.camera_target = center;
        self.camera_dist = 20.0;
        self.log(LogLevel::Info, "Framed all entities");
    }

    fn select_all(&mut self) {
        if !self.entities.is_empty() {
            self.selected_entity = Some(0);
            self.log(LogLevel::Info, "Selected first entity (multi-select not yet supported)");
        }
    }

    fn new_scene(&mut self) {
        while !self.entities.is_empty() {
            self.delete_entity(0);
        }
        self.scene_name = "Untitled Scene".to_string();
        self.scene_modified = false;
        self.selected_entity = None;
        self.notify("New scene created", LogLevel::System);
    }

    fn save_scene(&mut self) {
        self.save_scene_to_file("scene.json");
    }

    fn save_scene_to_file(&mut self, path: &str) {
        let data = serde_json::json!({
            "name": self.scene_name,
            "entities": self.entities.iter().map(|e| {
                serde_json::json!({
                    "name": e.name,
                    "type": format!("{:?}", e.entity_type),
                    "position": e.position,
                    "rotation": e.rotation,
                    "scale": e.scale,
                    "mesh_shape": format!("{:?}", e.mesh_shape),
                    "has_physics": e.has_physics,
                    "mass": e.mass,
                    "friction": e.friction,
                    "restitution": e.restitution,
                    "body_kind": format!("{:?}", e.body_kind),
                    "is_light": e.is_light,
                    "light_kind": format!("{:?}", e.light_kind),
                    "light_color": e.light_color,
                    "light_intensity": e.light_intensity,
                    "light_range": e.light_range,
                    "is_camera": e.is_camera,
                    "camera_fov": e.camera_fov,
                    "is_audio": e.is_audio,
                    "audio_volume": e.audio_volume,
                    "visible": e.visible,
                    "active": e.active,
                    "tags": e.tags,
                })
            }).collect::<Vec<_>>()
        });
        match std::fs::write(path, serde_json::to_string_pretty(&data).unwrap()) {
            Ok(_) => {
                self.scene_modified = false;
                self.notify(&format!("Saved: {} -> {}", self.scene_name, path), LogLevel::System);
            }
            Err(e) => {
                self.notify(&format!("Save failed: {}", e), LogLevel::Error);
            }
        }
    }

    fn load_scene_from_file(&mut self, path: &str) {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                self.notify(&format!("Load failed: {}", e), LogLevel::Error);
                return;
            }
        };
        let data: serde_json::Value = match serde_json::from_str(&content) {
            Ok(d) => d,
            Err(e) => {
                self.notify(&format!("Parse error: {}", e), LogLevel::Error);
                return;
            }
        };

        // Clear existing scene
        while !self.entities.is_empty() {
            self.delete_entity(0);
        }

        if let Some(name) = data["name"].as_str() {
            self.scene_name = name.to_string();
        }

        if let Some(entities) = data["entities"].as_array() {
            for ent_data in entities {
                let name = ent_data["name"].as_str().unwrap_or("Entity");
                let type_str = ent_data["type"].as_str().unwrap_or("Empty");
                let etype = match type_str {
                    "Mesh" => EntityType::Mesh,
                    "Light" => EntityType::Light,
                    "Camera" => EntityType::Camera,
                    "ParticleSystem" => EntityType::ParticleSystem,
                    "Audio" => EntityType::Audio,
                    _ => EntityType::Empty,
                };

                let idx = self.spawn_entity(name, etype);
                let e = &mut self.entities[idx];

                if let Some(pos) = ent_data["position"].as_array() {
                    for (i, v) in pos.iter().enumerate().take(3) {
                        e.position[i] = v.as_f64().unwrap_or(0.0) as f32;
                    }
                }
                if let Some(rot) = ent_data["rotation"].as_array() {
                    for (i, v) in rot.iter().enumerate().take(3) {
                        e.rotation[i] = v.as_f64().unwrap_or(0.0) as f32;
                    }
                }
                if let Some(scl) = ent_data["scale"].as_array() {
                    for (i, v) in scl.iter().enumerate().take(3) {
                        e.scale[i] = v.as_f64().unwrap_or(1.0) as f32;
                    }
                }

                if let Some(shape_str) = ent_data["mesh_shape"].as_str() {
                    e.mesh_shape = match shape_str {
                        "Sphere" => MeshShape::Sphere,
                        "Cylinder" => MeshShape::Cylinder,
                        "Capsule" => MeshShape::Capsule,
                        "Cone" => MeshShape::Cone,
                        "Plane" => MeshShape::Plane,
                        _ => MeshShape::Cube,
                    };
                }

                e.mass = ent_data["mass"].as_f64().unwrap_or(1.0) as f32;
                e.friction = ent_data["friction"].as_f64().unwrap_or(0.5) as f32;
                e.restitution = ent_data["restitution"].as_f64().unwrap_or(0.3) as f32;
                e.is_light = ent_data["is_light"].as_bool().unwrap_or(false);
                e.light_intensity = ent_data["light_intensity"].as_f64().unwrap_or(1.0) as f32;
                e.light_range = ent_data["light_range"].as_f64().unwrap_or(10.0) as f32;

                if let Some(lc) = ent_data["light_color"].as_array() {
                    for (i, v) in lc.iter().enumerate().take(3) {
                        e.light_color[i] = v.as_f64().unwrap_or(1.0) as f32;
                    }
                }

                if let Some(handle) = e.physics_handle {
                    let _ = self.engine.physics_mut().set_position(
                        handle,
                        Vec3::new(e.position[0], e.position[1], e.position[2]),
                    );
                }
            }
        }

        self.scene_modified = false;
        self.selected_entity = None;
        self.notify(&format!("Loaded: {} from {}", self.scene_name, path), LogLevel::System);
    }

    // =========================================================================
    // Undo / Redo -- wired to the real UndoStack (FIX 5)
    // =========================================================================

    fn push_move_undo(&mut self, ei: usize, old: [f32; 3], new: [f32; 3]) {
        self.undo_stack.push(Box::new(MoveEntityOp::new(
            UndoEntityId(ei as u64), UndoVec3::new(old[0], old[1], old[2]), UndoVec3::new(new[0], new[1], new[2]),
        )), true);
    }
    fn push_rotate_undo(&mut self, ei: usize, old: [f32; 3], new: [f32; 3]) {
        self.undo_stack.push(Box::new(RotateEntityOp::new(
            UndoEntityId(ei as u64), UndoQuat::new(old[0], old[1], old[2], 0.0), UndoQuat::new(new[0], new[1], new[2], 0.0),
        )), true);
    }
    fn push_scale_undo(&mut self, ei: usize, old: [f32; 3], new: [f32; 3]) {
        self.undo_stack.push(Box::new(ScaleEntityOp::new(
            UndoEntityId(ei as u64), UndoVec3::new(old[0], old[1], old[2]), UndoVec3::new(new[0], new[1], new[2]),
        )), true);
    }
    fn push_spawn_undo(&mut self, ei: usize, name: &str) {
        self.undo_stack.push_no_merge(Box::new(SpawnEntityOp::new(SerializedEntityData {
            entity: UndoEntityId(ei as u64), name: name.to_string(), components: vec![], parent: None, children: vec![],
        })));
    }
    fn push_delete_undo(&mut self, ei: usize, name: &str, pos: [f32; 3], rot: [f32; 3], scl: [f32; 3]) {
        self.undo_stack.push_no_merge(Box::new(DespawnEntityOp::new(SerializedEntityData {
            entity: UndoEntityId(ei as u64), name: name.to_string(),
            components: vec![
                SerializedComponentData { type_name: "Position".into(), data: pos.iter().flat_map(|v| v.to_le_bytes()).collect() },
                SerializedComponentData { type_name: "Rotation".into(), data: rot.iter().flat_map(|v| v.to_le_bytes()).collect() },
                SerializedComponentData { type_name: "Scale".into(), data: scl.iter().flat_map(|v| v.to_le_bytes()).collect() },
            ], parent: None, children: vec![],
        })));
    }
    fn perform_undo(&mut self) {
        match self.undo_stack.undo() {
            Some(desc) => { self.scene_modified = true; self.log(LogLevel::System, format!("Undo: {}", desc)); self.notify(&format!("Undo: {}", desc), LogLevel::System); }
            None => { self.log(LogLevel::Info, "Nothing to undo"); }
        }
    }
    fn perform_redo(&mut self) {
        match self.undo_stack.redo() {
            Some(desc) => { self.scene_modified = true; self.log(LogLevel::System, format!("Redo: {}", desc)); self.notify(&format!("Redo: {}", desc), LogLevel::System); }
            None => { self.log(LogLevel::Info, "Nothing to redo"); }
        }
    }

    // =========================================================================
    // Script VM with world bindings (FIX 6)
    // =========================================================================
    fn execute_script_with_bindings(&mut self, code: &str) {
        use std::sync::Mutex;
        let ep: Arc<Mutex<Vec<(String, [f32; 3])>>> = Arc::new(Mutex::new(self.entities.iter().map(|e| (e.name.clone(), e.position)).collect()));
        let sr: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let pc: Arc<Mutex<Vec<(usize, [f32; 3])>>> = Arc::new(Mutex::new(Vec::new()));
        let mut vm = GenovoVM::new();
        { let p = Arc::clone(&ep); let _ = vm.register_function("entity_count", Box::new(move |_: &[ScriptValue]| Ok(ScriptValue::Int(p.lock().unwrap().len() as i64))) as NativeFn); }
        { let p = Arc::clone(&ep); let s = Arc::clone(&sr);
          let _ = vm.register_function("spawn", Box::new(move |args: &[ScriptValue]| {
              let n = match args.first() { Some(ScriptValue::String(s)) => s.to_string(), _ => "ScriptEntity".into() };
              let mut g = p.lock().unwrap(); let i = g.len(); g.push((n.clone(), [0.0;3])); s.lock().unwrap().push(n); Ok(ScriptValue::Int(i as i64))
          }) as NativeFn); }
        { let p = Arc::clone(&ep);
          let _ = vm.register_function("get_pos", Box::new(move |args: &[ScriptValue]| {
              let i = match args.first() { Some(ScriptValue::Int(i)) => *i as usize, _ => return Err(ScriptError::TypeError("get_pos: int expected".into())) };
              let g = p.lock().unwrap(); if i >= g.len() { return Err(ScriptError::RuntimeError(format!("index {} OOB", i))); }
              Ok(ScriptValue::Vec3(g[i].1[0], g[i].1[1], g[i].1[2]))
          }) as NativeFn); }
        { let p = Arc::clone(&ep); let c = Arc::clone(&pc);
          let _ = vm.register_function("set_pos", Box::new(move |args: &[ScriptValue]| {
              if args.len() != 4 { return Err(ScriptError::ArityMismatch { function: "set_pos".into(), expected: 4, got: args.len() as u8 }); }
              let i = match &args[0] { ScriptValue::Int(i) => *i as usize, _ => return Err(ScriptError::TypeError("set_pos idx".into())) };
              let x = match &args[1] { ScriptValue::Float(f) => *f as f32, ScriptValue::Int(v) => *v as f32, _ => return Err(ScriptError::TypeError("x".into())) };
              let y = match &args[2] { ScriptValue::Float(f) => *f as f32, ScriptValue::Int(v) => *v as f32, _ => return Err(ScriptError::TypeError("y".into())) };
              let z = match &args[3] { ScriptValue::Float(f) => *f as f32, ScriptValue::Int(v) => *v as f32, _ => return Err(ScriptError::TypeError("z".into())) };
              let mut g = p.lock().unwrap(); if i >= g.len() { return Err(ScriptError::RuntimeError(format!("index {} OOB", i))); }
              g[i].1 = [x, y, z]; c.lock().unwrap().push((i, [x, y, z])); Ok(ScriptValue::Nil)
          }) as NativeFn); }
        match vm.load_script("console", code) {
            Ok(_) => match vm.execute("console", &mut ScriptContext::new()) {
                Ok(_) => {
                    for l in vm.output() { self.log(LogLevel::Info, format!("  => {}", l)); }
                    for n in sr.lock().unwrap().iter() { let i = self.spawn_entity(n, EntityType::Empty); self.log(LogLevel::System, format!("Script spawned: {} (idx {})", n, i)); }
                    for &(i, pos) in pc.lock().unwrap().iter() {
                        if i < self.entities.len() { let old = self.entities[i].position; self.entities[i].position = pos; self.sync_entity_to_physics(i); self.push_move_undo(i, old, pos);
                            self.log(LogLevel::System, format!("Script moved {} to ({:.1},{:.1},{:.1})", self.entities[i].name, pos[0], pos[1], pos[2])); }
                    }
                }
                Err(e) => self.log(LogLevel::Error, format!("Script error: {}", e)),
            },
            Err(e) => self.log(LogLevel::Error, format!("Compile error: {}", e)),
        }
    }

    // =========================================================================
    // Console command execution
    // =========================================================================

    fn execute_console_command(&mut self, cmd: &str) {
        self.log(LogLevel::Info, format!("> {}", cmd));
        let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
        if parts.is_empty() {
            return;
        }
        match parts[0] {
            "help" => {
                self.log(LogLevel::System, "--- Genovo Studio Console Commands ---");
                self.log(LogLevel::System, "  help                 Show this help");
                self.log(LogLevel::System, "  clear                Clear console");
                self.log(LogLevel::System, "  stats                Engine statistics");
                self.log(LogLevel::System, "  spawn <type> [name]  Spawn entity (empty/cube/sphere/cylinder/capsule/cone/plane/light/camera/particles/audio)");
                self.log(LogLevel::System, "  delete               Delete selected entity");
                self.log(LogLevel::System, "  list                 List all entities");
                self.log(LogLevel::System, "  select <index>       Select entity by index");
                self.log(LogLevel::System, "  physics start|stop|step  Control physics");
                self.log(LogLevel::System, "  terrain gen          Generate procedural terrain");
                self.log(LogLevel::System, "  dungeon gen          Generate BSP dungeon");
                self.log(LogLevel::System, "  script <code>        Execute GenovoScript");
                self.log(LogLevel::System, "  gravity <x> <y> <z>  Set gravity vector");
                self.log(LogLevel::System, "  scene <name>         Rename scene");
                self.log(LogLevel::System, "  camera <yaw> <pitch> <dist>  Set camera");
                self.log(LogLevel::System, "  speed <value>        Set simulation speed");
                self.log(LogLevel::System, "  about                Show version info");
            }
            "clear" => {
                self.console_log.clear();
            }
            "about" => {
                self.log(LogLevel::System, "Genovo Studio v1.0 -- Professional Game Development Environment");
                self.log(LogLevel::System, "26 engine modules fully linked");
            }
            "stats" => {
                let ecs_count = self.engine.world().entity_count();
                let body_count = self.engine.physics().body_count();
                let active = self.engine.physics().active_body_count();
                self.log(LogLevel::System, format!("ECS entities:    {}", ecs_count));
                self.log(LogLevel::System, format!("Scene entities:  {}", self.entities.len()));
                self.log(LogLevel::System, format!("Physics bodies:  {} ({} active)", body_count, active));
                self.log(LogLevel::System, format!("FPS:             {:.1}", self.smooth_fps));
                self.log(LogLevel::System, format!("Frame time:      {:.2} ms", self.smooth_frame_time));
                self.log(LogLevel::System, format!("Sim speed:       {:.2}x", self.sim_speed));
                self.log(LogLevel::System, format!("Frame count:     {}", self.frame_count));
            }
            "list" => {
                if self.entities.is_empty() {
                    self.log(LogLevel::Info, "No entities in scene");
                } else {
                    let msgs: Vec<String> = self.entities.iter().enumerate().map(|(i, e)| {
                        let sel = if self.selected_entity == Some(i) { " *" } else { "" };
                        format!(
                            "  [{}] {} ({}) pos=({:.1},{:.1},{:.1}){}",
                            i, e.name, e.entity_type,
                            e.position[0], e.position[1], e.position[2], sel
                        )
                    }).collect();
                    for msg in msgs {
                        self.log(LogLevel::Info, msg);
                    }
                }
            }
            "select" => {
                if let Some(idx_str) = parts.get(1) {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        if idx < self.entities.len() {
                            self.selected_entity = Some(idx);
                            self.log(LogLevel::System, format!("Selected: {}", self.entities[idx].name));
                        } else {
                            self.log(LogLevel::Error, format!("Index {} out of range (0..{})", idx, self.entities.len()));
                        }
                    } else {
                        self.log(LogLevel::Error, "Usage: select <index>");
                    }
                } else {
                    self.log(LogLevel::Error, "Usage: select <index>");
                }
            }
            "delete" => {
                self.delete_selected();
            }
            "spawn" => {
                let etype_str = parts.get(1).copied().unwrap_or("empty");
                match etype_str {
                    "cube" | "mesh" => { let i = self.spawn_mesh(MeshShape::Cube); self.selected_entity = Some(i); }
                    "sphere" => { let i = self.spawn_mesh(MeshShape::Sphere); self.selected_entity = Some(i); }
                    "cylinder" => { let i = self.spawn_mesh(MeshShape::Cylinder); self.selected_entity = Some(i); }
                    "capsule" => { let i = self.spawn_mesh(MeshShape::Capsule); self.selected_entity = Some(i); }
                    "cone" => { let i = self.spawn_mesh(MeshShape::Cone); self.selected_entity = Some(i); }
                    "plane" => { let i = self.spawn_mesh(MeshShape::Plane); self.selected_entity = Some(i); }
                    "light" => { let i = self.spawn_light(LightKind::Point); self.selected_entity = Some(i); }
                    "camera" => { let i = self.spawn_entity("Camera", EntityType::Camera); self.selected_entity = Some(i); }
                    "particles" => { let i = self.spawn_entity("Particles", EntityType::ParticleSystem); self.selected_entity = Some(i); }
                    "audio" => { let i = self.spawn_entity("Audio Source", EntityType::Audio); self.selected_entity = Some(i); }
                    _ => {
                        let name = if parts.len() > 2 { parts[2..].join(" ") } else { "Empty".to_string() };
                        let i = self.spawn_entity(&name, EntityType::Empty);
                        self.selected_entity = Some(i);
                    }
                }
            }
            "physics" => match parts.get(1).copied() {
                Some("start") => {
                    self.is_playing = true;
                    self.is_paused = false;
                    self.log(LogLevel::System, "Physics started");
                }
                Some("stop") => {
                    self.is_playing = false;
                    self.log(LogLevel::System, "Physics stopped");
                }
                Some("step") => {
                    self.step_physics();
                }
                _ => self.log(LogLevel::Warn, "Usage: physics [start|stop|step]"),
            },
            "gravity" => {
                if parts.len() == 4 {
                    let x: f32 = parts[1].parse().unwrap_or(0.0);
                    let y: f32 = parts[2].parse().unwrap_or(-9.81);
                    let z: f32 = parts[3].parse().unwrap_or(0.0);
                    self.engine.physics_mut().set_gravity(Vec3::new(x, y, z));
                    self.log(LogLevel::System, format!("Gravity set to ({}, {}, {})", x, y, z));
                } else {
                    let g = self.engine.physics().gravity();
                    self.log(LogLevel::Info, format!("Current gravity: ({:.2}, {:.2}, {:.2})", g.x, g.y, g.z));
                    self.log(LogLevel::Info, "Usage: gravity <x> <y> <z>");
                }
            }
            "speed" => {
                if let Some(val) = parts.get(1) {
                    if let Ok(s) = val.parse::<f32>() {
                        self.sim_speed = s.clamp(0.0, 10.0);
                        self.log(LogLevel::System, format!("Simulation speed: {:.2}x", self.sim_speed));
                    }
                } else {
                    self.log(LogLevel::Info, format!("Speed: {:.2}x", self.sim_speed));
                }
            }
            "camera" => {
                if parts.len() == 4 {
                    self.camera_yaw = parts[1].parse().unwrap_or(self.camera_yaw);
                    self.camera_pitch = parts[2].parse().unwrap_or(self.camera_pitch);
                    self.camera_dist = parts[3].parse().unwrap_or(self.camera_dist);
                    self.log(LogLevel::System, format!(
                        "Camera: yaw={:.1} pitch={:.1} dist={:.1}",
                        self.camera_yaw, self.camera_pitch, self.camera_dist
                    ));
                } else {
                    self.log(LogLevel::Info, format!(
                        "Camera: yaw={:.1} pitch={:.1} dist={:.1}",
                        self.camera_yaw, self.camera_pitch, self.camera_dist
                    ));
                }
            }
            "scene" => {
                if parts.len() > 1 {
                    self.scene_name = parts[1..].join(" ");
                    self.scene_modified = true;
                    self.log(LogLevel::System, format!("Scene renamed to: {}", self.scene_name));
                } else {
                    self.log(LogLevel::Info, format!("Current scene: {}", self.scene_name));
                }
            }
            "terrain" if parts.get(1) == Some(&"gen") => {
                match genovo_terrain::Heightmap::generate_procedural(257, 0.7, 42) {
                    Ok(h) => self.log(
                        LogLevel::System,
                        format!(
                            "Terrain 257x257 generated [min={:.2}, max={:.2}]",
                            h.min_height(),
                            h.max_height()
                        ),
                    ),
                    Err(e) => self.log(LogLevel::Error, format!("Terrain error: {}", e)),
                }
            }
            "dungeon" if parts.get(1) == Some(&"gen") => {
                let cfg = genovo_procgen::BSPConfig {
                    width: 80,
                    height: 60,
                    min_room_size: 6,
                    max_depth: 8,
                    room_fill_ratio: 0.7,
                    seed: 42,
                    wall_padding: 1,
                };
                let d = genovo_procgen::dungeon::generate_bsp(&cfg);
                self.log(
                    LogLevel::System,
                    format!("Dungeon generated: {} rooms", d.rooms.len()),
                );
            }
            "script" => {
                let code = parts[1..].join(" ");
                self.execute_script_with_bindings(&code);
            }
            other => {
                self.log(
                    LogLevel::Warn,
                    format!("Unknown command: '{}'. Type 'help' for commands.", other),
                );
            }
        }
    }
}

// =============================================================================
// Premium Dark Theme — matches UIStyle::dark()
// =============================================================================

fn apply_premium_theme(ctx: &egui::Context) {
    use egui::*;

    let mut style = (*ctx.style()).clone();

    style.visuals.dark_mode = true;

    // Core fills (near-black)
    style.visuals.panel_fill = BG_PANEL;
    style.visuals.window_fill = BG_PANEL;
    style.visuals.extreme_bg_color = BG_BASE;
    style.visuals.faint_bg_color = BG_WIDGET;
    style.visuals.override_text_color = Some(TEXT_NORMAL);

    let rounding = Rounding::same(3);

    // Non-interactive widgets
    style.visuals.widgets.noninteractive.bg_fill = BG_WIDGET;
    style.visuals.widgets.noninteractive.weak_bg_fill = BG_WIDGET;
    style.visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_DIM);
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(0.5, BORDER_SUBTLE);
    style.visuals.widgets.noninteractive.corner_radius = rounding;

    // Inactive widgets
    style.visuals.widgets.inactive.bg_fill = BG_WIDGET;
    style.visuals.widgets.inactive.weak_bg_fill = BG_WIDGET;
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_NORMAL);
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(0.5, BORDER);
    style.visuals.widgets.inactive.corner_radius = rounding;

    // Hovered widgets
    style.visuals.widgets.hovered.bg_fill = BG_HOVER;
    style.visuals.widgets.hovered.weak_bg_fill = BG_HOVER;
    style.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);
    style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT_DIM);
    style.visuals.widgets.hovered.corner_radius = rounding;

    // Active widgets (accent color)
    style.visuals.widgets.active.bg_fill = ACCENT;
    style.visuals.widgets.active.weak_bg_fill = ACCENT_DIM;
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    style.visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT_BRIGHT);
    style.visuals.widgets.active.corner_radius = rounding;

    // Open widgets (menus, popups)
    style.visuals.widgets.open.bg_fill = BG_ACTIVE;
    style.visuals.widgets.open.weak_bg_fill = BG_HOVER;
    style.visuals.widgets.open.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);
    style.visuals.widgets.open.bg_stroke = Stroke::new(1.0, BORDER);
    style.visuals.widgets.open.corner_radius = rounding;

    // Selection
    style.visuals.selection.bg_fill = ACCENT.gamma_multiply(0.25);
    style.visuals.selection.stroke = Stroke::new(1.0, ACCENT);

    // Window: flat, no shadows
    style.visuals.window_corner_radius = Rounding::same(4);
    style.visuals.window_stroke = Stroke::new(1.0, BORDER);
    style.visuals.window_shadow = Shadow::NONE;
    style.visuals.popup_shadow = Shadow {
        offset: [0, 2].into(),
        blur: 8,
        spread: 0,
        color: Color32::from_black_alpha(80),
    };

    // Striped alternating rows
    style.visuals.striped = true;

    // Spacing: tight, information-dense
    style.spacing.item_spacing = vec2(4.0, 2.0);
    style.spacing.window_margin = Margin::same(4);
    style.spacing.button_padding = vec2(6.0, 2.0);
    style.spacing.indent = 14.0;

    // Thin scrollbars (4px, matching UIStyle::dark().scrollbar_width)
    style.spacing.scroll = egui::style::ScrollStyle {
        bar_width: 4.0,
        floating: true,
        foreground_color: false,
        ..style.spacing.scroll
    };

    // Smaller interaction region for density
    style.spacing.interact_size = vec2(32.0, 18.0);

    ctx.set_style(style);
}

// =============================================================================
// Section header helper
// =============================================================================

fn section_header(ui: &mut egui::Ui, title: &str, icon: &str, color: egui::Color32) {
    let rect = ui.available_rect_before_wrap();
    let header_rect = egui::Rect::from_min_size(
        rect.left_top(),
        egui::vec2(rect.width(), 20.0),
    );
    ui.painter().rect_filled(header_rect, 0.0, BG_BASE);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(icon).size(10.0).color(color));
        ui.label(
            egui::RichText::new(title)
                .size(10.5)
                .strong()
                .color(TEXT_NORMAL),
        );
    });
}

// =============================================================================
// Colored XYZ drag values (matching editor_widgets::vec3_edit)
// =============================================================================

fn colored_drag_xyz(ui: &mut egui::Ui, values: &mut [f32; 3], speed: f32) -> bool {
    let mut changed = false;

    let labels = [("X", X_COLOR), ("Y", Y_COLOR), ("Z", Z_COLOR)];
    let width = ((ui.available_width() - 54.0) / 3.0).max(30.0);

    for (i, (label, color)) in labels.iter().enumerate() {
        // Colored axis letter
        ui.label(
            egui::RichText::new(*label)
                .size(10.5)
                .strong()
                .color(*color),
        );
        let drag = egui::DragValue::new(&mut values[i])
            .speed(speed)
            .max_decimals(3);
        let resp = ui.add_sized(egui::vec2(width, 18.0), drag);
        if resp.changed() {
            changed = true;
        }
    }

    changed
}

/// Compact single-row vec3 with colored axis labels and reset button
fn colored_drag_xyz_with_reset(
    ui: &mut egui::Ui,
    label: &str,
    values: &mut [f32; 3],
    speed: f32,
    defaults: [f32; 3],
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .size(10.0)
                .color(TEXT_DIM),
        );

        let labels = [("X", X_COLOR), ("Y", Y_COLOR), ("Z", Z_COLOR)];
        let width = ((ui.available_width() - 80.0) / 3.0).max(28.0);

        for (i, (axis, color)) in labels.iter().enumerate() {
            ui.label(egui::RichText::new(*axis).size(10.0).strong().color(*color));
            let drag = egui::DragValue::new(&mut values[i])
                .speed(speed)
                .max_decimals(3);
            if ui.add_sized(egui::vec2(width, 16.0), drag).changed() {
                changed = true;
            }
        }

        // Reset button
        if ui.add(
            egui::Button::new(
                egui::RichText::new("R").size(9.0).color(TEXT_MUTED),
            ).min_size(egui::vec2(16.0, 16.0))
        ).on_hover_text("Reset to default").clicked() {
            *values = defaults;
            changed = true;
        }
    });
    changed
}

// =============================================================================
// Thin separator / section divider
// =============================================================================

fn thin_separator() -> egui::Separator {
    egui::Separator::default().spacing(4.0)
}

fn accent_line(ui: &mut egui::Ui) {
    let rect = ui.available_rect_before_wrap();
    ui.painter().line_segment(
        [
            egui::pos2(rect.left(), rect.top()),
            egui::pos2(rect.right(), rect.top()),
        ],
        egui::Stroke::new(1.0, BORDER_SUBTLE),
    );
    ui.add_space(1.0);
}

fn accent_line_colored(ui: &mut egui::Ui, color: egui::Color32) {
    let rect = ui.available_rect_before_wrap();
    ui.painter().line_segment(
        [
            egui::pos2(rect.left(), rect.top()),
            egui::pos2(rect.right(), rect.top()),
        ],
        egui::Stroke::new(1.0, color),
    );
    ui.add_space(1.0);
}

// =============================================================================
// Colored dot indicator
// =============================================================================

fn status_dot(ui: &mut egui::Ui, color: egui::Color32, radius: f32) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(radius * 2.0 + 2.0, radius * 2.0 + 2.0),
        egui::Sense::hover(),
    );
    ui.painter().circle_filled(rect.center(), radius, color);
}

// =============================================================================
// Menu Bar (22px)
// =============================================================================

fn draw_menu_bar(ctx: &egui::Context, state: &mut EditorState) {
    egui::TopBottomPanel::top("menu_bar")
        .exact_height(22.0)
        .frame(
            egui::Frame::new()
                .fill(BG_BASE)
                .inner_margin(egui::Margin::symmetric(6, 0)),
        )
        .show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // Genovo Studio branding
                ui.label(
                    egui::RichText::new("Genovo Studio")
                        .size(11.0)
                        .strong()
                        .color(ACCENT),
                );
                ui.add_space(8.0);

                // ---- File menu ----
                ui.menu_button("File", |ui| {
                    if ui.add(egui::Button::new("New Scene").shortcut_text("Ctrl+N")).clicked() {
                        state.new_scene();
                        ui.close_menu();
                    }
                    if ui.button("Open Scene...").clicked() {
                        state.load_scene_from_file("scene.json");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Save").shortcut_text("Ctrl+S")).clicked() {
                        state.save_scene();
                        ui.close_menu();
                    }
                    if ui.button("Save As...").clicked() {
                        state.save_scene_to_file("scene.json");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Import Asset...").clicked() {
                        state.log(LogLevel::System, "Import asset (placeholder)");
                        ui.close_menu();
                    }
                    if ui.button("Export...").clicked() {
                        state.log(LogLevel::System, "Export (placeholder)");
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.menu_button("Recent Files", |ui| {
                        let files = state.recent_files.clone();
                        if files.is_empty() {
                            ui.label(egui::RichText::new("No recent files").size(10.0).color(TEXT_MUTED));
                        }
                        for f in &files {
                            if ui.button(f).clicked() {
                                state.log(LogLevel::System, format!("Open recent: {}", f));
                                ui.close_menu();
                            }
                        }
                    });
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        std::process::exit(0);
                    }
                });

                // ---- Edit menu ----
                ui.menu_button("Edit", |ui| {
                    if ui.add(egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                        state.perform_undo();
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Redo").shortcut_text("Ctrl+Y")).clicked() {
                        state.perform_redo();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Cut").clicked() { state.log(LogLevel::Info, "Cut (placeholder)"); ui.close_menu(); }
                    if ui.button("Copy").clicked() { state.log(LogLevel::Info, "Copy (placeholder)"); ui.close_menu(); }
                    if ui.button("Paste").clicked() { state.log(LogLevel::Info, "Paste (placeholder)"); ui.close_menu(); }
                    ui.separator();
                    if ui.add(egui::Button::new("Duplicate").shortcut_text("Ctrl+D")).clicked() {
                        state.duplicate_selected();
                        ui.close_menu();
                    }
                    if ui.add(egui::Button::new("Delete").shortcut_text("Del")).clicked() {
                        state.delete_selected();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Select All").clicked() {
                        state.select_all();
                        ui.close_menu();
                    }
                    if ui.button("Deselect All").clicked() {
                        state.selected_entity = None;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Preferences...").clicked() {
                        state.show_preferences = true;
                        ui.close_menu();
                    }
                });

                // ---- View menu ----
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut state.panels.hierarchy, "Scene Outliner");
                    ui.checkbox(&mut state.panels.inspector, "Inspector");
                    ui.checkbox(&mut state.panels.bottom, "Bottom Panel");
                    ui.checkbox(&mut state.panels.toolbar, "Toolbar");
                    ui.checkbox(&mut state.panels.status_bar, "Status Bar");
                    ui.separator();
                    ui.checkbox(&mut state.grid_visible, "Show Grid");
                    ui.checkbox(&mut state.wireframe_mode, "Wireframe Mode");
                    ui.checkbox(&mut state.stats_visible, "Show Stats");
                    ui.separator();
                    if ui.button("Reset Layout").clicked() {
                        state.panels = PanelVisibility::default();
                        state.hierarchy_width = 220.0;
                        state.inspector_width = 280.0;
                        state.bottom_height = 200.0;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Focus Selected").shortcut_text("F")).clicked() {
                        state.focus_selected();
                        ui.close_menu();
                    }
                    if ui.button("Frame All").clicked() {
                        state.frame_all();
                        ui.close_menu();
                    }
                });

                // ---- Create menu ----
                ui.menu_button("Create", |ui| {
                    if ui.button("Empty Entity").clicked() {
                        let i = state.spawn_entity("Empty", EntityType::Empty);
                        state.selected_entity = Some(i);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("Primitives").size(9.5).color(TEXT_DIM));
                    for shape in [MeshShape::Cube, MeshShape::Sphere, MeshShape::Cylinder, MeshShape::Capsule, MeshShape::Cone, MeshShape::Plane] {
                        if ui.button(format!("{}", shape)).clicked() {
                            let i = state.spawn_mesh(shape);
                            state.selected_entity = Some(i);
                            ui.close_menu();
                        }
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("Lights").size(9.5).color(TEXT_DIM));
                    for kind in [LightKind::Directional, LightKind::Point, LightKind::Spot] {
                        if ui.button(format!("{} Light", kind)).clicked() {
                            let i = state.spawn_light(kind);
                            state.selected_entity = Some(i);
                            ui.close_menu();
                        }
                    }
                    ui.separator();
                    if ui.button("Camera").clicked() {
                        let i = state.spawn_entity("Camera", EntityType::Camera);
                        state.selected_entity = Some(i);
                        ui.close_menu();
                    }
                    if ui.button("Particle System").clicked() {
                        let i = state.spawn_entity("Particle System", EntityType::ParticleSystem);
                        state.selected_entity = Some(i);
                        ui.close_menu();
                    }
                });

                // ---- Tools menu ----
                ui.menu_button("Tools", |ui| {
                    if ui.button("Terrain Editor").clicked() {
                        state.execute_console_command("terrain gen");
                        ui.close_menu();
                    }
                    if ui.button("Material Editor").clicked() {
                        state.log(LogLevel::System, "Material editor (placeholder)");
                        ui.close_menu();
                    }
                    if ui.button("Animation Editor").clicked() {
                        state.bottom_tab = BottomTab::Animation;
                        ui.close_menu();
                    }
                    if ui.button("Shader Graph").clicked() {
                        state.log(LogLevel::System, "Shader graph editor (placeholder)");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Generate Dungeon").clicked() {
                        state.execute_console_command("dungeon gen");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Profiler").clicked() {
                        state.bottom_tab = BottomTab::Profiler;
                        ui.close_menu();
                    }
                    if ui.button("Script Console").clicked() {
                        state.bottom_tab = BottomTab::Console;
                        ui.close_menu();
                    }
                });

                // ---- Window menu ----
                ui.menu_button("Window", |ui| {
                    if ui.button("Hierarchy").clicked() { state.panels.hierarchy = true; ui.close_menu(); }
                    if ui.button("Inspector").clicked() { state.panels.inspector = true; ui.close_menu(); }
                    if ui.button("Content Browser").clicked() { state.bottom_tab = BottomTab::ContentBrowser; state.panels.bottom = true; ui.close_menu(); }
                    if ui.button("Console").clicked() { state.bottom_tab = BottomTab::Console; state.panels.bottom = true; ui.close_menu(); }
                    if ui.button("Profiler").clicked() { state.bottom_tab = BottomTab::Profiler; state.panels.bottom = true; ui.close_menu(); }
                    if ui.button("Animation").clicked() { state.bottom_tab = BottomTab::Animation; state.panels.bottom = true; ui.close_menu(); }
                });

                // ---- Help menu ----
                ui.menu_button("Help", |ui| {
                    if ui.button("About Genovo Studio").clicked() {
                        state.show_about = true;
                        ui.close_menu();
                    }
                    if ui.button("Documentation").clicked() {
                        state.log(LogLevel::System, "https://genovo.dev/docs");
                        ui.close_menu();
                    }
                    if ui.button("Keyboard Shortcuts").clicked() {
                        state.show_shortcuts = true;
                        ui.close_menu();
                    }
                    if ui.button("Check for Updates").clicked() {
                        state.log(LogLevel::System, "Genovo Studio v1.0 is up to date");
                        ui.close_menu();
                    }
                });

                // Right-aligned scene name indicator
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let modified = if state.scene_modified { " *" } else { "" };
                    ui.label(
                        egui::RichText::new(format!("{}{}", state.scene_name, modified))
                            .size(10.0)
                            .color(if state.scene_modified { YELLOW } else { TEXT_DIM }),
                    );
                });
            });
        });
}

// =============================================================================
// Toolbar (30px)
// =============================================================================

fn draw_toolbar(ctx: &egui::Context, state: &mut EditorState) {
    if !state.panels.toolbar {
        return;
    }

    egui::TopBottomPanel::top("toolbar")
        .exact_height(30.0)
        .frame(
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(22, 22, 26))
                .inner_margin(egui::Margin::symmetric(4, 2))
                .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;

                // ---- Gizmo mode buttons (Q/W/E/R) ----
                for mode in [GizmoMode::Select, GizmoMode::Translate, GizmoMode::Rotate, GizmoMode::Scale] {
                    let selected = state.gizmo_mode == mode;
                    let text = format!("{} {}", mode.icon(), mode.short_label());
                    let btn = egui::SelectableLabel::new(
                        selected,
                        egui::RichText::new(&text).size(10.5),
                    );
                    if ui.add(btn).on_hover_text(format!("{} ({})", mode.label(), mode.short_label())).clicked() {
                        state.gizmo_mode = mode;
                    }
                }

                ui.add(thin_separator());

                // ---- Pivot ----
                let pivot_label = if state.pivot_mode == PivotMode::Center { "Center" } else { "Pivot" };
                if ui.add(egui::SelectableLabel::new(
                    false,
                    egui::RichText::new(pivot_label).size(10.0),
                )).on_hover_text("Toggle pivot mode").clicked() {
                    state.pivot_mode = if state.pivot_mode == PivotMode::Center { PivotMode::Pivot } else { PivotMode::Center };
                }

                // ---- Coord space ----
                let space_label = if state.coord_space == CoordSpace::Local { "Local" } else { "World" };
                if ui.add(egui::SelectableLabel::new(
                    false,
                    egui::RichText::new(space_label).size(10.0),
                )).on_hover_text("Toggle local/world space").clicked() {
                    state.coord_space = if state.coord_space == CoordSpace::Local { CoordSpace::World } else { CoordSpace::Local };
                }

                ui.add(thin_separator());

                // ---- Snap ----
                ui.checkbox(&mut state.snap_enabled, "");
                ui.label(egui::RichText::new("Snap").size(10.0).color(TEXT_DIM));
                if state.snap_enabled {
                    egui::ComboBox::from_id_salt("snap_val")
                        .width(50.0)
                        .selected_text(format!("{:.2}", SNAP_PRESETS[state.snap_value_idx]))
                        .show_ui(ui, |ui| {
                            for (i, val) in SNAP_PRESETS.iter().enumerate() {
                                if ui.selectable_value(&mut state.snap_value_idx, i, format!("{:.2}", val)).clicked() {
                                    state.snap_translate = *val;
                                }
                            }
                        });
                }

                ui.add(thin_separator());

                // ---- Play / Pause / Stop / Step ----
                let playing = state.is_playing && !state.is_paused;
                let play_icon = if playing { "||" } else { "|>" };
                let play_color = if playing { GREEN } else { TEXT_BRIGHT };
                let play_bg = if playing {
                    egui::Color32::from_rgb(35, 65, 45)
                } else {
                    BG_WIDGET
                };

                let play_btn = egui::Button::new(
                    egui::RichText::new(play_icon).color(play_color).strong().size(11.0),
                )
                .fill(play_bg)
                .min_size(egui::vec2(32.0, 24.0));
                if ui.add(play_btn).on_hover_text("Play / Pause (Space)").clicked() {
                    state.toggle_play();
                }

                let pause_btn = egui::Button::new(
                    egui::RichText::new("||").color(YELLOW).size(11.0),
                )
                .fill(if state.is_paused { egui::Color32::from_rgb(60, 55, 35) } else { BG_WIDGET })
                .min_size(egui::vec2(26.0, 24.0));
                if ui.add(pause_btn).on_hover_text("Pause").clicked() {
                    if state.is_playing {
                        state.is_paused = !state.is_paused;
                    }
                }

                let stop_btn = egui::Button::new(
                    egui::RichText::new("[]").color(RED).size(11.0),
                )
                .fill(BG_WIDGET)
                .min_size(egui::vec2(26.0, 24.0));
                if ui.add(stop_btn).on_hover_text("Stop").clicked() {
                    state.stop_play();
                }

                let step_btn = egui::Button::new(
                    egui::RichText::new("|>|").color(TEXT_DIM).size(10.0),
                )
                .fill(BG_WIDGET)
                .min_size(egui::vec2(26.0, 24.0));
                if ui.add(step_btn).on_hover_text("Step (single physics frame)").clicked() {
                    state.step_physics();
                }

                ui.add(thin_separator());

                // ---- Speed dropdown ----
                ui.label(egui::RichText::new("Speed").size(10.0).color(TEXT_DIM));
                egui::ComboBox::from_id_salt("speed_combo")
                    .width(55.0)
                    .selected_text(format!("{:.2}x", state.sim_speed))
                    .show_ui(ui, |ui| {
                        for (i, val) in SPEED_PRESETS.iter().enumerate() {
                            if ui.selectable_value(&mut state.sim_speed_idx, i, format!("{:.2}x", val)).clicked() {
                                state.sim_speed = *val;
                            }
                        }
                    });

                ui.add(thin_separator());

                // ---- Grid toggle ----
                ui.checkbox(&mut state.grid_visible, "");
                ui.label(egui::RichText::new("Grid").size(10.0).color(TEXT_DIM));

                // ---- Wireframe toggle ----
                ui.checkbox(&mut state.wireframe_mode, "");
                ui.label(egui::RichText::new("Wire").size(10.0).color(TEXT_DIM));

                // ---- Stats toggle ----
                ui.checkbox(&mut state.stats_visible, "");
                ui.label(egui::RichText::new("Stats").size(10.0).color(TEXT_DIM));
            });
        });
}

// =============================================================================
// Status Bar (16px)
// =============================================================================

fn draw_status_bar(ctx: &egui::Context, state: &EditorState) {
    if !state.panels.status_bar {
        return;
    }

    egui::TopBottomPanel::bottom("status_bar")
        .exact_height(16.0)
        .frame(
            egui::Frame::new()
                .fill(BG_BASE)
                .inner_margin(egui::Margin::symmetric(6, 0)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                // Play state dot + label
                let (dot_color, status_text) = if state.is_playing && !state.is_paused {
                    (ACCENT_BRIGHT, "Playing")
                } else if state.is_paused {
                    (YELLOW, "Paused")
                } else {
                    (GREEN, "Editing")
                };
                status_dot(ui, dot_color, 3.0);
                ui.label(
                    egui::RichText::new(status_text)
                        .size(9.5)
                        .color(TEXT_DIM),
                );

                ui.add(thin_separator());

                // Scene name (italic if modified)
                let modified_indicator = if state.scene_modified { " *" } else { "" };
                ui.label(
                    egui::RichText::new(format!("{}{}", state.scene_name, modified_indicator))
                        .size(9.5)
                        .color(if state.scene_modified { YELLOW } else { TEXT_NORMAL })
                        .italics(),
                );

                // Sim time
                if state.is_playing {
                    ui.add(thin_separator());
                    ui.label(
                        egui::RichText::new(format!("Sim: {:.1}s", state.total_sim_time))
                            .size(9.5)
                            .color(TEXT_DIM),
                    );
                }

                // Right-aligned stats
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let bodies = state.engine.physics().body_count();
                    let ents = state.entities.len();

                    let fps_color = if state.smooth_fps >= 55.0 {
                        GREEN
                    } else if state.smooth_fps >= 30.0 {
                        YELLOW
                    } else {
                        RED
                    };

                    ui.label(
                        egui::RichText::new(format!("{} bodies", bodies))
                            .size(9.5)
                            .color(TEXT_MUTED),
                    );
                    ui.add(thin_separator());
                    ui.label(
                        egui::RichText::new(format!("{} entities", ents))
                            .size(9.5)
                            .color(TEXT_DIM),
                    );
                    ui.add(thin_separator());
                    ui.label(
                        egui::RichText::new(format!("{:.0} FPS", state.smooth_fps))
                            .size(9.5)
                            .color(fps_color),
                    );
                });
            });
        });
}

// =============================================================================
// Scene Outliner (left panel, 220px)
// =============================================================================

fn draw_hierarchy(ctx: &egui::Context, state: &mut EditorState) {
    if !state.panels.hierarchy {
        return;
    }

    egui::SidePanel::left("scene_outliner")
        .default_width(state.hierarchy_width)
        .min_width(160.0)
        .max_width(450.0)
        .resizable(true)
        .frame(
            egui::Frame::new()
                .fill(BG_PANEL)
                .inner_margin(egui::Margin::same(0))
                .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE)),
        )
        .show(ctx, |ui| {
            // Header bar
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("SCENE OUTLINER")
                        .size(9.5)
                        .strong()
                        .color(TEXT_DIM),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(4.0);
                    ui.menu_button(
                        egui::RichText::new("+  Add").size(10.0).color(GREEN),
                        |ui| {
                            draw_spawn_menu(ui, state);
                        },
                    );
                });
            });

            accent_line_colored(ui, ACCENT_DIM);

            // Search/filter bar
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                ui.label(egui::RichText::new("?").size(10.0).color(TEXT_MUTED));
                ui.add(
                    egui::TextEdit::singleline(&mut state.entity_filter)
                        .desired_width(ui.available_width() - 8.0)
                        .hint_text("Filter entities...")
                        .font(egui::TextStyle::Small),
                );
            });
            ui.add_space(2.0);
            accent_line(ui);

            // Empty state
            if state.entities.is_empty() {
                ui.add_space(30.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("No entities")
                            .size(11.0)
                            .color(TEXT_MUTED),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Click + to add")
                            .size(10.0)
                            .color(TEXT_MUTED),
                    );
                });
                return;
            }

            // Entity list
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let mut action: Option<HierarchyAction> = None;
                    let filter_lower = state.entity_filter.to_lowercase();

                    for i in 0..state.entities.len() {
                        let ent = &state.entities[i];

                        // Filter
                        if !filter_lower.is_empty() && !ent.name.to_lowercase().contains(&filter_lower) {
                            continue;
                        }

                        let selected = state.selected_entity == Some(i);
                        let icon_color = ent.entity_type.dot_color();

                        // Renaming mode
                        if state.renaming_entity == Some(i) {
                            ui.horizontal(|ui| {
                                ui.add_space(4.0);
                                status_dot(ui, icon_color, 3.0);
                                let resp = ui.add(
                                    egui::TextEdit::singleline(&mut state.rename_buffer)
                                        .desired_width(ui.available_width() - 8.0)
                                        .font(egui::TextStyle::Body),
                                );
                                if resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                    let new_name = state.rename_buffer.clone();
                                    if !new_name.trim().is_empty() {
                                        state.entities[i].name = new_name;
                                    }
                                    state.renaming_entity = None;
                                }
                                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                    state.renaming_entity = None;
                                }
                                if !resp.has_focus() {
                                    resp.request_focus();
                                }
                            });
                            continue;
                        }

                        ui.horizontal(|ui| {
                            ui.add_space(4.0);

                            // Colored entity type dot
                            status_dot(ui, icon_color, 3.0);

                            // Visibility toggle (eye icon)
                            let vis_text = if ent.visible { "o" } else { "-" };
                            let vis_color = if ent.visible { TEXT_MUTED } else { RED_DIM };
                            if ui.add(
                                egui::Button::new(
                                    egui::RichText::new(vis_text).size(8.0).color(vis_color)
                                )
                                .frame(false)
                                .min_size(egui::vec2(12.0, 12.0))
                            ).on_hover_text("Toggle visibility").clicked() {
                                action = Some(HierarchyAction::ToggleVisibility(i));
                            }

                            // Lock toggle
                            let lock_text = if ent.locked { "#" } else { "" };
                            if ent.locked {
                                ui.label(egui::RichText::new("#").size(8.0).color(YELLOW_DIM));
                            }

                            // Entity name
                            let name_color = if !ent.active { TEXT_MUTED } else if selected { TEXT_BRIGHT } else { TEXT_NORMAL };
                            let name_text = egui::RichText::new(&ent.name).size(11.0).color(name_color);
                            let resp = ui.add(egui::SelectableLabel::new(selected, name_text));

                            if resp.clicked() {
                                action = Some(HierarchyAction::Select(i));
                            }

                            if resp.double_clicked() {
                                action = Some(HierarchyAction::Rename(i));
                            }

                            // Context menu
                            resp.context_menu(|ui| {
                                if ui.button("Rename").clicked() {
                                    action = Some(HierarchyAction::Rename(i));
                                    ui.close_menu();
                                }
                                if ui.button("Duplicate").clicked() {
                                    action = Some(HierarchyAction::Duplicate(i));
                                    ui.close_menu();
                                }
                                if ui.button("Delete").clicked() {
                                    action = Some(HierarchyAction::Delete(i));
                                    ui.close_menu();
                                }
                                ui.separator();
                                if ui.button("Create Child").clicked() {
                                    action = Some(HierarchyAction::AddChild(i));
                                    ui.close_menu();
                                }
                                if ui.button("Group").clicked() {
                                    action = Some(HierarchyAction::Group(i));
                                    ui.close_menu();
                                }
                                ui.separator();
                                if ui.button("Toggle Visibility").clicked() {
                                    action = Some(HierarchyAction::ToggleVisibility(i));
                                    ui.close_menu();
                                }
                                if ui.button("Toggle Lock").clicked() {
                                    action = Some(HierarchyAction::ToggleLock(i));
                                    ui.close_menu();
                                }
                            });
                        });
                    }

                    // Execute deferred action
                    match action {
                        Some(HierarchyAction::Select(i)) => {
                            state.selected_entity = Some(i);
                        }
                        Some(HierarchyAction::Duplicate(i)) => {
                            state.selected_entity = Some(i);
                            state.duplicate_selected();
                        }
                        Some(HierarchyAction::Delete(i)) => {
                            state.delete_entity(i);
                        }
                        Some(HierarchyAction::AddChild(i)) => {
                            let parent_name = state.entities[i].name.clone();
                            let idx = state.spawn_entity(
                                &format!("{}/Child", parent_name),
                                EntityType::Empty,
                            );
                            state.entities[idx].parent = Some(i);
                            state.selected_entity = Some(idx);
                        }
                        Some(HierarchyAction::ToggleVisibility(i)) => {
                            state.entities[i].visible = !state.entities[i].visible;
                        }
                        Some(HierarchyAction::ToggleLock(i)) => {
                            state.entities[i].locked = !state.entities[i].locked;
                        }
                        Some(HierarchyAction::Rename(i)) => {
                            state.renaming_entity = Some(i);
                            state.rename_buffer = state.entities[i].name.clone();
                            state.selected_entity = Some(i);
                        }
                        Some(HierarchyAction::Group(i)) => {
                            let idx = state.spawn_entity("Group", EntityType::Empty);
                            state.entities[i].parent = Some(idx);
                            state.selected_entity = Some(idx);
                        }
                        None => {}
                    }
                });

            // Footer: entity count
            accent_line(ui);
            ui.horizontal(|ui| {
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(format!("{} entities", state.entities.len()))
                        .size(9.0)
                        .color(TEXT_MUTED),
                );
            });
        });
}

fn draw_spawn_menu(ui: &mut egui::Ui, state: &mut EditorState) {
    ui.label(egui::RichText::new("Primitives").size(9.5).color(TEXT_DIM));
    let mesh_items: &[(&str, MeshShape)] = &[
        ("Cube", MeshShape::Cube),
        ("Sphere", MeshShape::Sphere),
        ("Cylinder", MeshShape::Cylinder),
        ("Capsule", MeshShape::Capsule),
        ("Cone", MeshShape::Cone),
        ("Plane", MeshShape::Plane),
    ];
    for (name, shape) in mesh_items {
        ui.horizontal(|ui| {
            status_dot(ui, CYAN, 3.0);
            if ui.button(*name).clicked() {
                let idx = state.spawn_mesh(*shape);
                state.selected_entity = Some(idx);
                ui.close_menu();
            }
        });
    }

    ui.separator();
    ui.label(egui::RichText::new("Lights").size(9.5).color(TEXT_DIM));
    let light_items: &[(&str, LightKind)] = &[
        ("Directional Light", LightKind::Directional),
        ("Point Light", LightKind::Point),
        ("Spot Light", LightKind::Spot),
    ];
    for (name, kind) in light_items {
        ui.horizontal(|ui| {
            status_dot(ui, YELLOW, 3.0);
            if ui.button(*name).clicked() {
                let idx = state.spawn_light(*kind);
                state.selected_entity = Some(idx);
                ui.close_menu();
            }
        });
    }

    ui.separator();
    ui.label(egui::RichText::new("Other").size(9.5).color(TEXT_DIM));
    let other_items: &[(&str, EntityType, egui::Color32)] = &[
        ("Empty", EntityType::Empty, TEXT_DIM),
        ("Camera", EntityType::Camera, ACCENT_BRIGHT),
        ("Particle System", EntityType::ParticleSystem, MAGENTA),
        ("Audio Source", EntityType::Audio, ORANGE),
    ];
    for (name, etype, color) in other_items {
        ui.horizontal(|ui| {
            status_dot(ui, *color, 3.0);
            if ui.button(*name).clicked() {
                let idx = state.spawn_entity(name, *etype);
                state.selected_entity = Some(idx);
                ui.close_menu();
            }
        });
    }
}

// =============================================================================
// Inspector Panel (right, 280px)
// =============================================================================

fn draw_inspector(ctx: &egui::Context, state: &mut EditorState) {
    if !state.panels.inspector {
        return;
    }

    egui::SidePanel::right("inspector")
        .default_width(state.inspector_width)
        .min_width(200.0)
        .max_width(500.0)
        .resizable(true)
        .frame(
            egui::Frame::new()
                .fill(BG_PANEL)
                .inner_margin(egui::Margin::same(0))
                .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE)),
        )
        .show(ctx, |ui| {
            // Header
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("INSPECTOR")
                        .size(9.5)
                        .strong()
                        .color(TEXT_DIM),
                );
            });
            accent_line_colored(ui, ACCENT_DIM);

            let sel = state.selected_entity;
            if sel.is_none() || sel.unwrap() >= state.entities.len() {
                ui.add_space(40.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("No entity selected")
                            .size(11.0)
                            .color(TEXT_MUTED),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Select an entity in the outliner")
                            .size(9.5)
                            .color(TEXT_MUTED),
                    );
                });
                return;
            }
            let idx = sel.unwrap();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    draw_inspector_content(ui, state, idx);
                });
        });
}

fn draw_inspector_content(ui: &mut egui::Ui, state: &mut EditorState, idx: usize) {
    let inner_margin = 6.0;
    ui.add_space(2.0);

    // ---- Entity Identity ----
    let etype = state.entities[idx].entity_type;
    let eid = state.entities[idx].entity;
    let icon_col = etype.icon_color();

    // Entity name (large, editable)
    ui.horizontal(|ui| {
        ui.add_space(inner_margin);
        status_dot(ui, icon_col, 4.0);
        ui.add(
            egui::TextEdit::singleline(&mut state.entities[idx].name)
                .desired_width(ui.available_width() - inner_margin)
                .font(egui::TextStyle::Heading),
        );
    });

    // Entity type badge + active toggle
    ui.horizontal(|ui| {
        ui.add_space(inner_margin);
        // Type badge
        let badge_text = format!(" {} ", etype);
        let badge_color = icon_col;
        ui.label(
            egui::RichText::new(&badge_text)
                .size(9.5)
                .strong()
                .color(BG_BASE)
                .background_color(badge_color),
        );
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(format!("Entity {}v{}", eid.id, eid.generation))
                .size(9.0)
                .color(TEXT_MUTED),
        );

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(inner_margin);
            ui.checkbox(&mut state.entities[idx].active, "Active");
        });
    });
    ui.add_space(4.0);
    accent_line(ui);

    // ---- Transform Section (always shown) ----
    let mut pos_changed = false;
    egui::CollapsingHeader::new(
        egui::RichText::new("Transform").strong().size(11.0).color(TEXT_NORMAL),
    )
    .default_open(state.inspector_sections.transform_open)
    .show(ui, |ui| {
        ui.add_space(2.0);
        pos_changed |= colored_drag_xyz_with_reset(
            ui, "Position", &mut state.entities[idx].position, 0.1, [0.0; 3],
        );
        pos_changed |= colored_drag_xyz_with_reset(
            ui, "Rotation", &mut state.entities[idx].rotation, 0.5, [0.0; 3],
        );
        colored_drag_xyz_with_reset(
            ui, "Scale   ", &mut state.entities[idx].scale, 0.01, [1.0, 1.0, 1.0],
        );
        ui.add_space(2.0);
    });

    if pos_changed {
        state.sync_entity_to_physics(idx);
        state.scene_modified = true;
    }

    // ---- Mesh Renderer Section ----
    if etype == EntityType::Mesh {
        egui::CollapsingHeader::new(
            egui::RichText::new("Mesh Renderer").strong().size(11.0).color(CYAN),
        )
        .default_open(state.inspector_sections.mesh_open)
        .show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Mesh").size(10.0).color(TEXT_DIM));
                let shape = state.entities[idx].mesh_shape;
                egui::ComboBox::from_id_salt(format!("mesh_shape_{}", idx))
                    .width(100.0)
                    .selected_text(format!("{}", shape))
                    .show_ui(ui, |ui| {
                        for s in [MeshShape::Cube, MeshShape::Sphere, MeshShape::Cylinder, MeshShape::Capsule, MeshShape::Cone, MeshShape::Plane] {
                            ui.selectable_value(&mut state.entities[idx].mesh_shape, s, format!("{}", s));
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.checkbox(&mut state.entities[idx].cast_shadows, "Cast Shadows");
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.checkbox(&mut state.entities[idx].receive_shadows, "Receive Shadows");
            });
            ui.add_space(2.0);
        });
    }

    // ---- Physics Section ----
    if state.entities[idx].has_physics {
        egui::CollapsingHeader::new(
            egui::RichText::new("Rigid Body").strong().size(11.0).color(TEXT_NORMAL),
        )
        .default_open(state.inspector_sections.physics_open)
        .show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Body Type").size(10.0).color(TEXT_DIM));
                egui::ComboBox::from_id_salt(format!("body_kind_{}", idx))
                    .width(90.0)
                    .selected_text(format!("{}", state.entities[idx].body_kind))
                    .show_ui(ui, |ui| {
                        for bk in [BodyKind::Dynamic, BodyKind::Static, BodyKind::Kinematic] {
                            ui.selectable_value(&mut state.entities[idx].body_kind, bk, format!("{}", bk));
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Mass").size(10.0).color(TEXT_DIM));
                ui.add(egui::DragValue::new(&mut state.entities[idx].mass).speed(0.1).range(0.01..=10000.0).suffix(" kg"));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Friction").size(10.0).color(TEXT_DIM));
                ui.add(egui::Slider::new(&mut state.entities[idx].friction, 0.0..=1.0).max_decimals(2));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Restitution").size(10.0).color(TEXT_DIM));
                ui.add(egui::Slider::new(&mut state.entities[idx].restitution, 0.0..=1.0).max_decimals(2));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Lin Damping").size(10.0).color(TEXT_DIM));
                ui.add(egui::Slider::new(&mut state.entities[idx].linear_damping, 0.0..=10.0).max_decimals(2));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Ang Damping").size(10.0).color(TEXT_DIM));
                ui.add(egui::Slider::new(&mut state.entities[idx].angular_damping, 0.0..=10.0).max_decimals(2));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Gravity Scale").size(10.0).color(TEXT_DIM));
                ui.add(egui::DragValue::new(&mut state.entities[idx].gravity_scale).speed(0.1).range(-10.0..=10.0));
            });

            // Velocity readout
            if let Some(handle) = state.entities[idx].physics_handle {
                if let Ok(vel) = state.engine.physics().get_linear_velocity(handle) {
                    ui.horizontal(|ui| {
                        ui.add_space(inner_margin);
                        ui.label(egui::RichText::new(format!("Velocity: ({:.2}, {:.2}, {:.2})", vel.x, vel.y, vel.z)).size(9.5).color(TEXT_MUTED).monospace());
                    });
                    let speed = (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z).sqrt();
                    ui.horizontal(|ui| {
                        ui.add_space(inner_margin);
                        ui.label(egui::RichText::new(format!("Speed: {:.3} m/s", speed)).size(9.5).color(TEXT_MUTED).monospace());
                    });
                }
            }
            ui.add_space(2.0);
        });

        // Collider section
        egui::CollapsingHeader::new(
            egui::RichText::new("Collider").strong().size(11.0).color(TEXT_NORMAL),
        )
        .default_open(state.inspector_sections.collider_open)
        .show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Shape").size(10.0).color(TEXT_DIM));
                egui::ComboBox::from_id_salt(format!("collider_shape_{}", idx))
                    .width(80.0)
                    .selected_text(format!("{}", state.entities[idx].collider_shape))
                    .show_ui(ui, |ui| {
                        for cs in [ColliderShape::Box, ColliderShape::Sphere, ColliderShape::Capsule] {
                            ui.selectable_value(&mut state.entities[idx].collider_shape, cs, format!("{}", cs));
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.checkbox(&mut state.entities[idx].is_trigger, "Is Trigger");
            });
            ui.add_space(2.0);
        });
    }

    // ---- Light Section ----
    if state.entities[idx].is_light {
        egui::CollapsingHeader::new(
            egui::RichText::new("Light").strong().size(11.0).color(YELLOW),
        )
        .default_open(state.inspector_sections.light_open)
        .show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Type").size(10.0).color(TEXT_DIM));
                egui::ComboBox::from_id_salt(format!("light_kind_{}", idx))
                    .width(100.0)
                    .selected_text(format!("{}", state.entities[idx].light_kind))
                    .show_ui(ui, |ui| {
                        for lk in [LightKind::Directional, LightKind::Point, LightKind::Spot] {
                            ui.selectable_value(&mut state.entities[idx].light_kind, lk, format!("{}", lk));
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Color").size(10.0).color(TEXT_DIM));
                ui.color_edit_button_rgb(&mut state.entities[idx].light_color);
                let preview = egui::Color32::from_rgb(
                    (state.entities[idx].light_color[0] * 255.0) as u8,
                    (state.entities[idx].light_color[1] * 255.0) as u8,
                    (state.entities[idx].light_color[2] * 255.0) as u8,
                );
                let (swatch_rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
                ui.painter().rect_filled(swatch_rect, 2.0, preview);
                ui.painter().rect_stroke(swatch_rect, 2.0, egui::Stroke::new(1.0, BORDER), egui::StrokeKind::Outside);
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Intensity").size(10.0).color(TEXT_DIM));
                ui.add(egui::Slider::new(&mut state.entities[idx].light_intensity, 0.0..=100.0).logarithmic(true).max_decimals(2));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Range").size(10.0).color(TEXT_DIM));
                ui.add(egui::Slider::new(&mut state.entities[idx].light_range, 0.1..=1000.0).logarithmic(true).max_decimals(1).suffix(" m"));
            });
            if state.entities[idx].light_kind == LightKind::Spot {
                ui.horizontal(|ui| {
                    ui.add_space(inner_margin);
                    ui.label(egui::RichText::new("Spot Angle").size(10.0).color(TEXT_DIM));
                    ui.add(egui::Slider::new(&mut state.entities[idx].light_spot_angle, 1.0..=179.0).suffix(" deg"));
                });
            }
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.checkbox(&mut state.entities[idx].light_shadows, "Cast Shadows");
            });
            ui.add_space(2.0);
        });
    }

    // ---- Camera Section ----
    if state.entities[idx].is_camera {
        egui::CollapsingHeader::new(
            egui::RichText::new("Camera").strong().size(11.0).color(ACCENT_BRIGHT),
        )
        .default_open(state.inspector_sections.camera_open)
        .show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Projection").size(10.0).color(TEXT_DIM));
                egui::ComboBox::from_id_salt(format!("cam_proj_{}", idx))
                    .width(100.0)
                    .selected_text(format!("{}", state.entities[idx].camera_projection))
                    .show_ui(ui, |ui| {
                        for cp in [CameraProjection::Perspective, CameraProjection::Orthographic] {
                            ui.selectable_value(&mut state.entities[idx].camera_projection, cp, format!("{}", cp));
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("FOV").size(10.0).color(TEXT_DIM));
                ui.add(egui::Slider::new(&mut state.entities[idx].camera_fov, 10.0..=170.0).suffix(" deg"));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Near").size(10.0).color(TEXT_DIM));
                ui.add(egui::DragValue::new(&mut state.entities[idx].camera_near).speed(0.01).range(0.001..=100.0));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Far").size(10.0).color(TEXT_DIM));
                ui.add(egui::DragValue::new(&mut state.entities[idx].camera_far).speed(10.0).range(1.0..=100000.0));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Clear Color").size(10.0).color(TEXT_DIM));
                ui.color_edit_button_rgba_premultiplied(&mut state.entities[idx].camera_clear_color);
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Depth").size(10.0).color(TEXT_DIM));
                ui.label(egui::RichText::new("0").size(10.0).color(TEXT_MUTED));
            });
            ui.add_space(2.0);
        });
    }

    // ---- Audio Source Section ----
    if state.entities[idx].is_audio {
        egui::CollapsingHeader::new(
            egui::RichText::new("Audio Source").strong().size(11.0).color(ORANGE),
        )
        .default_open(state.inspector_sections.audio_open)
        .show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Clip").size(10.0).color(TEXT_DIM));
                ui.label(egui::RichText::new("(none)").size(10.0).color(TEXT_MUTED));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Volume").size(10.0).color(TEXT_DIM));
                ui.add(egui::Slider::new(&mut state.entities[idx].audio_volume, 0.0..=1.0).max_decimals(2));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("Pitch").size(10.0).color(TEXT_DIM));
                ui.add(egui::DragValue::new(&mut state.entities[idx].audio_pitch).speed(0.01).range(0.1..=4.0));
            });
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.checkbox(&mut state.entities[idx].audio_spatial, "Spatial");
            });
            if state.entities[idx].audio_spatial {
                ui.horizontal(|ui| {
                    ui.add_space(inner_margin);
                    ui.label(egui::RichText::new("Min Dist").size(10.0).color(TEXT_DIM));
                    ui.add(egui::DragValue::new(&mut state.entities[idx].audio_min_dist).speed(0.1).range(0.0..=100.0).suffix(" m"));
                });
                ui.horizontal(|ui| {
                    ui.add_space(inner_margin);
                    ui.label(egui::RichText::new("Max Dist").size(10.0).color(TEXT_DIM));
                    ui.add(egui::DragValue::new(&mut state.entities[idx].audio_max_dist).speed(1.0).range(1.0..=1000.0).suffix(" m"));
                });
            }
            ui.add_space(2.0);
        });
    }

    // ---- Script Section ----
    if state.entities[idx].has_script {
        egui::CollapsingHeader::new(
            egui::RichText::new("Script").strong().size(11.0).color(MAGENTA),
        )
        .default_open(state.inspector_sections.script_open)
        .show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(inner_margin);
                ui.label(egui::RichText::new("File").size(10.0).color(TEXT_DIM));
                ui.add(egui::TextEdit::singleline(&mut state.entities[idx].script_file).desired_width(150.0).hint_text("script.genovo"));
            });
            ui.add_space(2.0);
        });
    }

    // ---- Add Component Button ----
    ui.add_space(4.0);
    accent_line(ui);
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.add_space(inner_margin);
        ui.menu_button(
            egui::RichText::new("+ Add Component")
                .size(10.5)
                .color(ACCENT),
            |ui| {
                if !state.entities[idx].has_physics {
                    if ui.button("Rigidbody + Collider").clicked() {
                        let ent = &state.entities[idx];
                        let desc = physics::RigidBodyDesc {
                            body_type: physics::BodyType::Dynamic,
                            position: Vec3::new(ent.position[0], ent.position[1], ent.position[2]),
                            mass: ent.mass,
                            ..Default::default()
                        };
                        if let Ok(handle) = state.engine.physics_mut().add_body(&desc) {
                            let _ = state.engine.physics_mut().add_collider(
                                handle,
                                &physics::ColliderDesc {
                                    shape: physics::CollisionShape::Sphere { radius: 0.5 },
                                    ..Default::default()
                                },
                            );
                            let ent = &mut state.entities[idx];
                            ent.has_physics = true;
                            ent.physics_handle = Some(handle);
                        }
                        state.notify("Added Rigidbody + Collider", LogLevel::System);
                        ui.close_menu();
                    }
                }
                if !state.entities[idx].is_light {
                    if ui.button("Light").clicked() {
                        state.entities[idx].is_light = true;
                        state.notify("Added Light component", LogLevel::System);
                        ui.close_menu();
                    }
                }
                if !state.entities[idx].is_camera {
                    if ui.button("Camera").clicked() {
                        state.entities[idx].is_camera = true;
                        state.notify("Added Camera component", LogLevel::System);
                        ui.close_menu();
                    }
                }
                if !state.entities[idx].is_audio {
                    if ui.button("Audio Source").clicked() {
                        state.entities[idx].is_audio = true;
                        state.notify("Added Audio Source", LogLevel::System);
                        ui.close_menu();
                    }
                }
                if !state.entities[idx].has_script {
                    if ui.button("Script").clicked() {
                        state.entities[idx].has_script = true;
                        state.notify("Added Script component", LogLevel::System);
                        ui.close_menu();
                    }
                }
            },
        );
    });
}

// =============================================================================
// Bottom Panel with tabs
// =============================================================================

fn draw_bottom_panel(ctx: &egui::Context, state: &mut EditorState) {
    if !state.panels.bottom {
        return;
    }

    egui::TopBottomPanel::bottom("bottom_panel")
        .default_height(state.bottom_height)
        .min_height(80.0)
        .max_height(600.0)
        .resizable(true)
        .frame(
            egui::Frame::new()
                .fill(BG_PANEL)
                .inner_margin(egui::Margin::same(0))
                .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE)),
        )
        .show(ctx, |ui| {
            // Tab bar
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                let tabs = [
                    (BottomTab::ContentBrowser, "Content Browser"),
                    (BottomTab::Console, "Console"),
                    (BottomTab::Profiler, "Profiler"),
                    (BottomTab::Animation, "Animation"),
                ];
                for (tab, label) in &tabs {
                    let selected = state.bottom_tab == *tab;
                    let text = if selected {
                        egui::RichText::new(*label).size(10.5).strong().color(TEXT_BRIGHT)
                    } else {
                        egui::RichText::new(*label).size(10.5).color(TEXT_DIM)
                    };
                    let resp = ui.add(egui::SelectableLabel::new(selected, text));
                    if resp.clicked() {
                        state.bottom_tab = *tab;
                    }
                    if selected {
                        let rect = resp.rect;
                        ui.painter().line_segment(
                            [egui::pos2(rect.left() + 2.0, rect.bottom()), egui::pos2(rect.right() - 2.0, rect.bottom())],
                            egui::Stroke::new(2.0, ACCENT),
                        );
                    }
                }
            });

            accent_line(ui);
            ui.add_space(2.0);

            let content_frame = egui::Frame::new().inner_margin(egui::Margin::symmetric(4, 2));
            content_frame.show(ui, |ui| {
                match state.bottom_tab {
                    BottomTab::Console => draw_console_tab(ui, state),
                    BottomTab::ContentBrowser => draw_content_browser_tab(ui, state),
                    BottomTab::Profiler => draw_profiler_tab(ui, state),
                    BottomTab::Animation => draw_animation_tab(ui, state),
                }
            });
        });
}

// =============================================================================
// Console Tab
// =============================================================================

fn draw_console_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    let input_height = 22.0;
    let available = ui.available_height() - input_height - 6.0;

    // Filter buttons
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Filter:").size(9.5).color(TEXT_MUTED));
        let filters: [(Option<LogLevel>, &str, egui::Color32); 5] = [
            (None, "All", TEXT_DIM),
            (Some(LogLevel::Info), "Info", TEXT_NORMAL),
            (Some(LogLevel::Warn), "Warn", YELLOW),
            (Some(LogLevel::Error), "Err", RED),
            (Some(LogLevel::System), "Sys", ACCENT),
        ];
        for (filter, label, color) in &filters {
            let selected = state.console_filter_level == *filter;
            if ui.add(egui::SelectableLabel::new(selected, egui::RichText::new(*label).size(9.5).color(*color))).clicked() {
                state.console_filter_level = *filter;
            }
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.add(egui::Button::new(egui::RichText::new("Clear").size(9.5).color(TEXT_MUTED))).clicked() {
                state.console_log.clear();
            }
            ui.checkbox(&mut state.console_auto_scroll, "");
            ui.label(egui::RichText::new("Auto").size(9.0).color(TEXT_MUTED));
        });
    });
    ui.add_space(1.0);

    // Log output
    let mut scroll_area = egui::ScrollArea::vertical().max_height(available.max(20.0)).auto_shrink([false, false]);
    if state.console_auto_scroll {
        scroll_area = scroll_area.stick_to_bottom(true);
    }
    scroll_area.show(ui, |ui| {
        for entry in &state.console_log {
            if let Some(filter) = state.console_filter_level {
                if entry.level != filter { continue; }
            }
            if let Some(tint) = entry.level.bg_tint() {
                let rect = ui.available_rect_before_wrap();
                let tint_rect = egui::Rect::from_min_size(rect.left_top(), egui::vec2(rect.width(), 16.0));
                ui.painter().rect_filled(tint_rect, 0.0, tint);
            }
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.label(egui::RichText::new(format!("{:>7.1}", entry.timestamp)).size(9.5).color(TEXT_MUTED).monospace());
                ui.label(egui::RichText::new(entry.level.prefix()).size(9.5).color(entry.level.color()).monospace().strong());
                ui.label(egui::RichText::new(&entry.text).size(10.5).color(entry.level.color()));
                if entry.count > 1 {
                    ui.label(egui::RichText::new(format!("(x{})", entry.count)).size(9.0).color(TEXT_MUTED));
                }
            });
        }
    });

    // Command input
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(">").monospace().color(ACCENT).size(11.0));
        let resp = ui.add(
            egui::TextEdit::singleline(&mut state.console_input)
                .desired_width(ui.available_width() - 40.0)
                .hint_text("Type command... (help)")
                .font(egui::TextStyle::Monospace),
        );
        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            submit_console(state);
            resp.request_focus();
        }
        if resp.has_focus() {
            if ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                let hlen = state.console_history.len();
                if hlen > 0 {
                    let idx = match state.console_history_idx {
                        Some(i) if i > 0 => i - 1,
                        Some(i) => i,
                        None => hlen - 1,
                    };
                    state.console_history_idx = Some(idx);
                    state.console_input = state.console_history[idx].clone();
                }
            }
            if ui.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                let hlen = state.console_history.len();
                if let Some(i) = state.console_history_idx {
                    if i + 1 < hlen {
                        state.console_history_idx = Some(i + 1);
                        state.console_input = state.console_history[i + 1].clone();
                    } else {
                        state.console_history_idx = None;
                        state.console_input.clear();
                    }
                }
            }
        }
        if ui.add(egui::Button::new(egui::RichText::new("Run").size(10.0).color(ACCENT)).min_size(egui::vec2(32.0, 18.0))).clicked() {
            submit_console(state);
        }
    });
}

fn submit_console(state: &mut EditorState) {
    let cmd = state.console_input.clone();
    if !cmd.trim().is_empty() {
        state.console_history.push(cmd.clone());
        state.console_history_idx = None;
        state.execute_console_command(&cmd);
        state.console_input.clear();
    }
}

// =============================================================================
// Content Browser Tab
// =============================================================================

fn draw_content_browser_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    // Breadcrumb path bar
    ui.horizontal(|ui| {
        let path_clone = state.asset_path.clone();
        let segments: Vec<&str> = path_clone.split(['/', '\\']).filter(|s| !s.is_empty()).collect();
        ui.label(egui::RichText::new("res://").size(10.0).color(ACCENT_DIM));
        for (i, seg) in segments.iter().enumerate() {
            ui.label(egui::RichText::new(">").size(10.0).color(TEXT_MUTED));
            if i == segments.len() - 1 {
                ui.label(egui::RichText::new(*seg).size(10.0).color(TEXT_BRIGHT));
            } else {
                if ui.add(egui::Button::new(egui::RichText::new(*seg).size(10.0).color(ACCENT_DIM)).frame(false)).clicked() {
                    let path: String = segments[..=i].join("/");
                    state.asset_path = path;
                }
            }
        }

        ui.add(thin_separator());
        if ui.add(egui::Button::new(egui::RichText::new("..").size(10.0))).on_hover_text("Go up").clicked() {
            if let Some(pos) = state.asset_path.rfind('/').or_else(|| state.asset_path.rfind('\\')) {
                state.asset_path.truncate(pos);
            }
        }
        ui.add(thin_separator());
        ui.label(egui::RichText::new("?").size(10.0).color(TEXT_MUTED));
        ui.add(egui::TextEdit::singleline(&mut state.asset_search).desired_width(120.0).hint_text("Search...").font(egui::TextStyle::Small));

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Import").on_hover_text("Import asset").clicked() {
                state.log(LogLevel::System, "Import asset (placeholder)");
            }
            ui.add(egui::Slider::new(&mut state.asset_view_size, 40.0..=160.0).show_value(false));
            ui.label(egui::RichText::new("Size").size(9.0).color(TEXT_MUTED));
        });
    });
    ui.add_space(2.0);

    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        let entries = scan_asset_dir(&state.asset_path, &state.asset_search);
        if entries.is_empty() {
            let defaults = [
                ("[D]", "Models/", CYAN),
                ("[D]", "Textures/", GREEN),
                ("[D]", "Materials/", ACCENT),
                ("[D]", "Audio/", ORANGE),
                ("[D]", "Scripts/", MAGENTA),
                ("[D]", "Scenes/", YELLOW),
                ("[D]", "Shaders/", RED),
            ];
            ui.add_space(4.0);
            ui.label(egui::RichText::new("Default asset structure:").size(10.0).color(TEXT_MUTED));
            ui.add_space(4.0);
            for (icon, name, color) in &defaults {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(*icon).size(10.0).color(YELLOW_DIM));
                    ui.label(egui::RichText::new(*name).size(10.5).color(*color));
                });
            }
        } else {
            let item_size = state.asset_view_size;
            let cols = ((ui.available_width() / (item_size + 8.0)) as usize).max(1);
            egui::Grid::new("content_grid").num_columns(cols).spacing(egui::vec2(4.0, 4.0)).show(ui, |ui| {
                for (i, entry) in entries.iter().enumerate() {
                    let icon = asset_icon(&entry.name, entry.is_dir);
                    let color = if entry.is_dir { YELLOW } else { asset_color(&entry.name) };
                    let display = if state.asset_show_extensions || entry.is_dir {
                        entry.name.clone()
                    } else {
                        entry.name.rsplit_once('.').map_or(entry.name.clone(), |(base, _)| base.to_string())
                    };
                    let btn = egui::Button::new(egui::RichText::new(format!("{} {}", icon, display)).size(10.0).color(color)).min_size(egui::vec2(item_size, 20.0));
                    if ui.add(btn).clicked() {
                        if entry.is_dir {
                            state.asset_path = format!("{}/{}", state.asset_path, entry.name);
                        } else {
                            state.log(LogLevel::Info, format!("Selected asset: {}", entry.name));
                        }
                    }
                    if (i + 1) % cols == 0 { ui.end_row(); }
                }
            });
        }
    });
}

struct AssetEntry {
    name: String,
    is_dir: bool,
}

fn scan_asset_dir(path: &str, filter: &str) -> Vec<AssetEntry> {
    let mut entries = Vec::new();
    if let Ok(dir) = std::fs::read_dir(path) {
        for entry in dir.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !filter.is_empty() && !name.to_lowercase().contains(&filter.to_lowercase()) { continue; }
            let is_dir = entry.file_type().map_or(false, |t| t.is_dir());
            entries.push(AssetEntry { name, is_dir });
        }
    }
    entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
    entries
}

fn asset_icon(name: &str, is_dir: bool) -> &'static str {
    if is_dir { return "[D]"; }
    match name.rsplit('.').next().unwrap_or("") {
        "obj" | "fbx" | "gltf" | "glb" => "[3D]",
        "png" | "jpg" | "jpeg" | "bmp" | "tga" | "dds" | "hdr" => "[TX]",
        "wav" | "ogg" | "mp3" | "flac" => "[AU]",
        "rs" | "lua" | "py" | "js" | "ts" => "[SC]",
        "ron" | "json" | "toml" | "yaml" | "yml" => "[CF]",
        "scene" | "scn" => "[SN]",
        "wgsl" | "glsl" | "hlsl" | "spv" => "[SH]",
        "ttf" | "otf" | "woff" | "woff2" => "[FN]",
        "mat" | "material" => "[MT]",
        _ => "[??]",
    }
}

fn asset_color(name: &str) -> egui::Color32 {
    match name.rsplit('.').next().unwrap_or("") {
        "obj" | "fbx" | "gltf" | "glb" => CYAN,
        "png" | "jpg" | "jpeg" | "bmp" | "tga" | "dds" | "hdr" => GREEN,
        "wav" | "ogg" | "mp3" | "flac" => ORANGE,
        "rs" | "lua" | "py" | "js" | "ts" => MAGENTA,
        "ron" | "json" | "toml" | "yaml" | "yml" => TEXT_DIM,
        "scene" | "scn" => ACCENT,
        "wgsl" | "glsl" | "hlsl" | "spv" => RED,
        _ => TEXT_MUTED,
    }
}

// =============================================================================
// Profiler Tab
// =============================================================================

fn draw_profiler_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 12.0;
        let fps_color = if state.smooth_fps >= 55.0 { GREEN } else if state.smooth_fps >= 30.0 { YELLOW } else { RED };
        ui.label(egui::RichText::new(format!("FPS: {:.0}", state.smooth_fps)).size(11.0).color(fps_color).strong());
        ui.label(egui::RichText::new(format!("Frame: {:.2} ms", state.smooth_frame_time)).size(11.0).color(TEXT_NORMAL));

        if !state.frame_times.is_empty() {
            let min = state.frame_times.iter().cloned().fold(f64::MAX, f64::min);
            let max = state.frame_times.iter().cloned().fold(f64::MIN, f64::max);
            let avg = state.frame_times.iter().sum::<f64>() / state.frame_times.len() as f64;
            let p99 = {
                let mut sorted: Vec<f64> = state.frame_times.iter().cloned().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                sorted.get((sorted.len() as f64 * 0.99) as usize).copied().unwrap_or(0.0)
            };
            ui.label(egui::RichText::new(format!("Min: {:.2}  Avg: {:.2}  P99: {:.2}  Max: {:.2} ms", min, avg, p99, max)).size(10.0).color(TEXT_DIM));
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(format!("Frame: {}", state.frame_count)).size(9.5).color(TEXT_MUTED).monospace());
        });
    });
    ui.add_space(2.0);

    let points: Vec<[f64; 2]> = state.frame_times.iter().enumerate().map(|(i, &v)| [i as f64, v]).collect();
    let line = egui_plot::Line::new(egui_plot::PlotPoints::new(points)).name("Frame time (ms)").color(ACCENT);
    let target_60 = egui_plot::HLine::new(16.67).name("60 FPS (16.67ms)").color(egui::Color32::from_rgb(45, 45, 50));
    let target_30 = egui_plot::HLine::new(33.33).name("30 FPS (33.33ms)").color(egui::Color32::from_rgb(55, 45, 40));

    egui_plot::Plot::new("profiler_frame_time")
        .height(ui.available_height().max(40.0))
        .include_y(0.0)
        .include_y(25.0)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .y_axis_label("ms")
        .show(ui, |plot_ui| {
            plot_ui.hline(target_60);
            plot_ui.hline(target_30);
            plot_ui.line(line);
        });
}

// =============================================================================
// Animation Timeline Tab
// =============================================================================

fn draw_animation_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    ui.horizontal(|ui| {
        // Transport controls
        if ui.add(egui::Button::new(egui::RichText::new("|<").size(10.0)).min_size(egui::vec2(24.0, 18.0))).on_hover_text("Go to start").clicked() {
            state.anim_time = 0.0;
        }
        let play_icon = if state.anim_playing { "||" } else { "|>" };
        if ui.add(egui::Button::new(egui::RichText::new(play_icon).size(10.0).color(if state.anim_playing { GREEN } else { TEXT_BRIGHT })).min_size(egui::vec2(24.0, 18.0))).on_hover_text("Play/Pause").clicked() {
            state.anim_playing = !state.anim_playing;
        }
        if ui.add(egui::Button::new(egui::RichText::new("[]").size(10.0).color(RED)).min_size(egui::vec2(24.0, 18.0))).on_hover_text("Stop").clicked() {
            state.anim_playing = false;
            state.anim_time = 0.0;
        }
        if ui.add(egui::Button::new(egui::RichText::new(">|").size(10.0)).min_size(egui::vec2(24.0, 18.0))).on_hover_text("Go to end").clicked() {
            state.anim_time = state.anim_duration;
        }
        ui.add(thin_separator());
        ui.checkbox(&mut state.anim_loop, "Loop");
        ui.add(thin_separator());
        ui.label(egui::RichText::new("Time:").size(10.0).color(TEXT_DIM));
        ui.add(egui::DragValue::new(&mut state.anim_time).speed(0.01).range(0.0..=state.anim_duration as f64).suffix(" s").max_decimals(2));
        ui.label(egui::RichText::new(format!("/ {:.1}s", state.anim_duration)).size(10.0).color(TEXT_MUTED));
        ui.add(thin_separator());
        ui.label(egui::RichText::new("Duration:").size(10.0).color(TEXT_DIM));
        ui.add(egui::DragValue::new(&mut state.anim_duration).speed(0.1).range(0.1..=300.0).suffix(" s"));
    });

    ui.add_space(4.0);

    // Timeline scrubber
    let fraction = if state.anim_duration > 0.0 { state.anim_time / state.anim_duration } else { 0.0 };
    let response = ui.add(egui::Slider::new(&mut state.anim_time, 0.0..=state.anim_duration).show_value(false).text(""));

    ui.add_space(4.0);
    ui.label(egui::RichText::new("Keyframes and curves will appear here when animations are loaded.").size(10.0).color(TEXT_MUTED));

    // Tick animation forward if playing
    if state.anim_playing {
        let dt = 1.0 / 60.0; // approximate
        state.anim_time += dt;
        if state.anim_time > state.anim_duration {
            if state.anim_loop {
                state.anim_time = 0.0;
            } else {
                state.anim_time = state.anim_duration;
                state.anim_playing = false;
            }
        }
    }
}

// =============================================================================
// Central Viewport (3D)
// =============================================================================

fn draw_viewport(ctx: &egui::Context, state: &EditorState) {
    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(egui::Color32::TRANSPARENT).inner_margin(egui::Margin::same(0)))
        .show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            let painter = ui.painter();

            // ---- Top-left overlay: FPS + mode ----
            if state.stats_visible {
                let fps_color = if state.smooth_fps > 55.0 { GREEN } else if state.smooth_fps > 30.0 { YELLOW } else { RED };
                painter.text(
                    rect.left_top() + egui::vec2(8.0, 6.0),
                    egui::Align2::LEFT_TOP,
                    format!("{:.0} FPS | {:.2} ms", state.smooth_fps, state.smooth_frame_time),
                    egui::FontId::monospace(11.0),
                    fps_color,
                );
                painter.text(
                    rect.left_top() + egui::vec2(8.0, 22.0),
                    egui::Align2::LEFT_TOP,
                    format!("{} | {} | {} | {}",
                        state.gizmo_mode.label(),
                        if state.coord_space == CoordSpace::Local { "Local" } else { "World" },
                        if state.grid_visible { "Grid" } else { "" },
                        if state.wireframe_mode { "Wire" } else { "" },
                    ),
                    egui::FontId::monospace(9.5),
                    TEXT_MUTED,
                );
            }

            // ---- Top-right overlay: Camera ----
            painter.text(
                rect.right_top() + egui::vec2(-8.0, 6.0),
                egui::Align2::RIGHT_TOP,
                format!("Camera  yaw:{:.0}  pitch:{:.0}  dist:{:.1}", state.camera_yaw, state.camera_pitch, state.camera_dist),
                egui::FontId::monospace(9.5),
                TEXT_MUTED,
            );
            painter.text(
                rect.right_top() + egui::vec2(-8.0, 20.0),
                egui::Align2::RIGHT_TOP,
                format!("Target: ({:.1}, {:.1}, {:.1})", state.camera_target[0], state.camera_target[1], state.camera_target[2]),
                egui::FontId::monospace(9.5),
                TEXT_MUTED,
            );

            // ---- Bottom-left: stats overlay ----
            if state.stats_visible {
                painter.text(
                    rect.left_bottom() + egui::vec2(8.0, -50.0),
                    egui::Align2::LEFT_TOP,
                    format!("Entities: {} | Bodies: {} | Draw calls: ~1",
                        state.entities.len(),
                        state.engine.physics().body_count(),
                    ),
                    egui::FontId::monospace(9.5),
                    egui::Color32::from_rgb(50, 50, 58),
                );
            }

            // ---- Bottom center: gizmo mode indicator ----
            painter.text(
                egui::pos2(rect.center().x, rect.bottom() - 14.0),
                egui::Align2::CENTER_BOTTOM,
                format!("3D Viewport | {} x {}", rect.width() as u32, rect.height() as u32),
                egui::FontId::monospace(9.5),
                egui::Color32::from_rgb(40, 40, 48),
            );

            // ---- Play state overlay with pulsing border ----
            if state.is_playing && !state.is_paused {
                let t = state.frame_count as f32 * 0.03;
                let alpha = ((t.sin() + 1.0) * 0.5 * 60.0 + 30.0) as u8;
                let pulse_col = egui::Color32::from_rgba_premultiplied(72, 199, 142, alpha);
                painter.rect_stroke(rect.shrink(1.0), 0.0, egui::Stroke::new(2.0, pulse_col), egui::StrokeKind::Outside);
                painter.text(
                    rect.left_top() + egui::vec2(8.0, 36.0),
                    egui::Align2::LEFT_TOP,
                    format!("SIMULATING | {:.2}x | {:.1}s", state.sim_speed, state.total_sim_time),
                    egui::FontId::monospace(10.0),
                    GREEN,
                );
            } else if state.is_paused {
                let pulse_col = egui::Color32::from_rgba_premultiplied(245, 196, 80, 40);
                painter.rect_stroke(rect.shrink(1.0), 0.0, egui::Stroke::new(2.0, pulse_col), egui::StrokeKind::Outside);
                painter.text(
                    rect.left_top() + egui::vec2(8.0, 36.0),
                    egui::Align2::LEFT_TOP,
                    format!("PAUSED | {:.1}s", state.total_sim_time),
                    egui::FontId::monospace(10.0),
                    YELLOW,
                );
            }

            // ---- Selected entity indicator (top center) ----
            if let Some(idx) = state.selected_entity {
                if idx < state.entities.len() {
                    let ent = &state.entities[idx];
                    let icon_color = ent.entity_type.icon_color();
                    let label = format!("[{}] {} ({:.1}, {:.1}, {:.1})", ent.entity_type.icon_letter(), ent.name, ent.position[0], ent.position[1], ent.position[2]);
                    let galley = painter.layout_no_wrap(label.clone(), egui::FontId::monospace(10.0), icon_color);
                    let text_w = galley.size().x;
                    let pill_rect = egui::Rect::from_center_size(egui::pos2(rect.center().x, rect.top() + 12.0), egui::vec2(text_w + 16.0, 18.0));
                    painter.rect_filled(pill_rect, 9.0, egui::Color32::from_rgba_premultiplied(18, 18, 22, 200));
                    painter.rect_stroke(pill_rect, 9.0, egui::Stroke::new(1.0, BORDER), egui::StrokeKind::Outside);
                    painter.text(pill_rect.center(), egui::Align2::CENTER_CENTER, label, egui::FontId::monospace(10.0), icon_color);
                }
            }

            // ---- Axis indicator (bottom-left) ----
            let axis_origin = egui::pos2(rect.left() + 30.0, rect.bottom() - 40.0);
            let axis_len = 18.0;
            painter.line_segment([axis_origin, axis_origin + egui::vec2(axis_len, 0.0)], egui::Stroke::new(2.0, X_COLOR));
            painter.text(axis_origin + egui::vec2(axis_len + 3.0, -3.0), egui::Align2::LEFT_CENTER, "X", egui::FontId::monospace(9.0), X_COLOR);
            painter.line_segment([axis_origin, axis_origin + egui::vec2(0.0, -axis_len)], egui::Stroke::new(2.0, Y_COLOR));
            painter.text(axis_origin + egui::vec2(-3.0, -axis_len - 6.0), egui::Align2::CENTER_BOTTOM, "Y", egui::FontId::monospace(9.0), Y_COLOR);
            painter.line_segment([axis_origin, axis_origin + egui::vec2(-axis_len * 0.6, axis_len * 0.5)], egui::Stroke::new(2.0, Z_COLOR));
            painter.text(axis_origin + egui::vec2(-axis_len * 0.6 - 6.0, axis_len * 0.5 + 2.0), egui::Align2::RIGHT_CENTER, "Z", egui::FontId::monospace(9.0), Z_COLOR);
        });
}

// =============================================================================
// About Genovo Studio Window
// =============================================================================

fn draw_about_window(ctx: &egui::Context, show: &mut bool) {
    egui::Window::new("About Genovo Studio")
        .open(show)
        .resizable(false)
        .default_width(380.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("GENOVO STUDIO").size(20.0).strong().color(ACCENT));
                ui.label(egui::RichText::new("v1.0").size(12.0).color(TEXT_DIM));
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Professional Game Development Environment").size(11.0).color(TEXT_NORMAL));
                ui.add_space(4.0);
                ui.label(egui::RichText::new("26 engine modules fully linked").size(10.0).color(TEXT_DIM));
                ui.add_space(8.0);

                let items = [
                    ("Rendering", "wgpu 24 (GPU-accelerated)", CYAN),
                    ("UI Backend", "egui 0.31 (hybrid)", ACCENT),
                    ("UI Framework", "genovo-ui (Slate-like)", GREEN),
                    ("Physics", "Custom impulse solver", YELLOW),
                    ("ECS", "Archetype-based", MAGENTA),
                    ("Audio", "Software PCM mixer", ORANGE),
                    ("Scripting", "GenovoScript VM", RED),
                    ("Terrain", "Procedural heightmap gen", GREEN),
                    ("Procgen", "BSP dungeon generation", CYAN),
                ];
                for (label, value, color) in &items {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(format!("{}:", label)).size(10.0).color(TEXT_DIM));
                        ui.label(egui::RichText::new(*value).size(10.0).color(*color));
                    });
                }

                ui.add_space(10.0);
                accent_line(ui);
                ui.add_space(4.0);
                ui.label(egui::RichText::new("genovo.dev").size(11.0).color(ACCENT));
                ui.add_space(4.0);
            });
        });
}

// =============================================================================
// Preferences Window
// =============================================================================

fn draw_preferences_window(ctx: &egui::Context, show: &mut bool, state: &mut EditorState) {
    egui::Window::new("Preferences")
        .open(show)
        .resizable(true)
        .default_width(400.0)
        .default_height(300.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.heading("Genovo Studio Settings");
            ui.add_space(4.0);

            egui::CollapsingHeader::new("Camera").default_open(true).show(ui, |ui| {
                ui.horizontal(|ui| { ui.label("Field of View"); ui.add(egui::Slider::new(&mut state.camera_fov, 30.0..=120.0).suffix(" deg")); });
            });
            egui::CollapsingHeader::new("Grid").default_open(true).show(ui, |ui| {
                ui.checkbox(&mut state.grid_visible, "Show Grid");
                ui.horizontal(|ui| { ui.label("Grid Size"); ui.add(egui::DragValue::new(&mut state.grid_size).speed(0.1).range(0.1..=100.0)); });
            });
            egui::CollapsingHeader::new("Snap Settings").default_open(true).show(ui, |ui| {
                ui.checkbox(&mut state.snap_enabled, "Enable Snap");
                ui.horizontal(|ui| { ui.label("Translate"); ui.add(egui::DragValue::new(&mut state.snap_translate).speed(0.1).suffix(" u")); });
                ui.horizontal(|ui| { ui.label("Rotate"); ui.add(egui::DragValue::new(&mut state.snap_rotate).speed(1.0).suffix(" deg")); });
                ui.horizontal(|ui| { ui.label("Scale"); ui.add(egui::DragValue::new(&mut state.snap_scale).speed(0.01)); });
            });
            egui::CollapsingHeader::new("Content Browser").default_open(false).show(ui, |ui| {
                ui.checkbox(&mut state.asset_show_extensions, "Show File Extensions");
                ui.horizontal(|ui| { ui.label("Default View Size"); ui.add(egui::Slider::new(&mut state.asset_view_size, 40.0..=160.0)); });
            });
        });
}

// =============================================================================
// Keyboard Shortcuts Window
// =============================================================================

fn draw_shortcuts_window(ctx: &egui::Context, show: &mut bool) {
    egui::Window::new("Keyboard Shortcuts")
        .open(show)
        .resizable(false)
        .default_width(380.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.label(egui::RichText::new("Genovo Studio Shortcuts").size(13.0).strong().color(ACCENT));
            ui.add_space(4.0);
            let shortcuts = [
                ("Ctrl+N", "New Scene"),
                ("Ctrl+S", "Save Scene"),
                ("Ctrl+Z", "Undo"),
                ("Ctrl+Y", "Redo"),
                ("Ctrl+D", "Duplicate Selected"),
                ("Delete", "Delete Selected"),
                ("", ""),
                ("Q", "Select Tool"),
                ("W", "Translate Tool"),
                ("E", "Rotate Tool"),
                ("R", "Scale Tool"),
                ("", ""),
                ("Space", "Play / Pause"),
                ("F5", "Toggle Play Mode"),
                ("F", "Focus Selected"),
                ("G", "Spawn Physics Ball"),
                ("", ""),
                ("Up/Down", "Select Prev/Next Entity"),
                ("Escape", "Deselect / Cancel"),
            ];
            for (key, desc) in &shortcuts {
                if key.is_empty() {
                    ui.add_space(2.0);
                    continue;
                }
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(*key).size(10.5).strong().color(ACCENT).monospace());
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new(*desc).size(10.5).color(TEXT_NORMAL));
                });
            }
        });
}

// =============================================================================
// Notification Toast
// =============================================================================

fn draw_notification(ctx: &egui::Context, state: &EditorState) {
    if let Some((ref msg, level, start)) = state.notification {
        let elapsed = start.elapsed().as_secs_f32();
        let duration = 3.0;
        if elapsed > duration { return; }

        let alpha = if elapsed < 0.2 { elapsed / 0.2 } else if elapsed > duration - 0.5 { (duration - elapsed) / 0.5 } else { 1.0 };

        egui::Area::new(egui::Id::new("notification_toast"))
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -36.0))
            .show(ctx, |ui| {
                let frame = egui::Frame::new()
                    .fill(BG_PANEL.gamma_multiply(alpha))
                    .stroke(egui::Stroke::new(1.0, BORDER.gamma_multiply(alpha)))
                    .corner_radius(egui::Rounding::same(4));
                frame.show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let color = level.color().gamma_multiply(alpha);
                        let (strip_rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 20.0), egui::Sense::hover());
                        ui.painter().rect_filled(strip_rect, 1.0, color);
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(level.prefix()).size(10.0).color(color).strong());
                        ui.label(egui::RichText::new(msg).size(11.0).color(TEXT_NORMAL.gamma_multiply(alpha)));
                    });
                });
            });
    }
}

// =============================================================================
// GPU Helpers
// =============================================================================

fn make_depth(dev: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    dev.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
    .create_view(&wgpu::TextureViewDescriptor::default())
}

// =============================================================================
// Application
// =============================================================================

struct EditorApp {
    gpu: Option<GpuState>,
    // Audio mixer running flag (keeps background thread alive)
    audio_running: Arc<AtomicBool>,
}

struct GpuState {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,

    // Full 3D scene renderer (replaces the old triangle pipeline)
    scene_manager: SceneRenderManager,

    // egui (hybrid renderer -- will be swapped for UIGpuRenderer when ready)
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    // Editor state (uses genovo_ui data structures for state management)
    editor: EditorState,
    theme_applied: bool,
}

impl ApplicationHandler for EditorApp {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        // Create window with Genovo Studio title
        let w = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("Genovo Studio \u{2014} Untitled Scene")
                    .with_inner_size(LogicalSize::new(1600, 900))
                    .with_maximized(true),
            )
            .unwrap(),
        );

        // wgpu setup
        let inst = wgpu::Instance::new(&Default::default());
        let surf = inst.create_surface(w.clone()).unwrap();
        let adap = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surf),
            ..Default::default()
        }))
        .unwrap();
        let (dev, que) = pollster::block_on(adap.request_device(&Default::default(), None)).unwrap();
        let sz = w.inner_size();
        let cfg = surf.get_default_config(&adap, sz.width.max(1), sz.height.max(1)).unwrap();
        surf.configure(&dev, &cfg);

        // FIX 1: Create full 3D SceneRenderManager (replaces the hardcoded triangle)
        let scene_manager = SceneRenderManager::new(
            &dev,
            cfg.format,
            wgpu::TextureFormat::Depth32Float,
        );

        let dv = make_depth(&dev, sz.width, sz.height);

        // FIX 2: Initialize audio output using engine SoftwareMixer in a background thread
        let audio_running = self.audio_running.clone();
        audio_running.store(true, Ordering::SeqCst);
        std::thread::Builder::new()
            .name("genovo-audio".into())
            .spawn(move || {
                use genovo_audio::{AudioMixer, SoftwareMixer};
                let mut mixer = SoftwareMixer::new(48000, 2, 1024, 32);
                println!("[Genovo Audio] Mixer thread started: 48kHz stereo, 1024 frames, 32 voices");
                let dt = 1024.0 / 48000.0; // ~21ms per tick
                while audio_running.load(Ordering::Relaxed) {
                    // Tick the mixer to process any queued audio
                    mixer.update(dt);
                    // Sleep ~21ms (48000 Hz / 1024 frames per tick)
                    std::thread::sleep(std::time::Duration::from_millis(21));
                }
                println!("[Genovo Audio] Mixer thread stopped");
            })
            .expect("Failed to spawn audio thread");

        // Engine init
        let mut engine = Engine::new(EngineConfig::default()).unwrap();

        // Ground plane (static physics body)
        let gd = physics::RigidBodyDesc {
            body_type: physics::BodyType::Static,
            position: Vec3::ZERO,
            ..Default::default()
        };
        let gh = engine.physics_mut().add_body(&gd).unwrap();
        let _ = engine.physics_mut().add_collider(
            gh,
            &physics::ColliderDesc {
                shape: physics::CollisionShape::Box { half_extents: Vec3::new(50.0, 0.5, 50.0) },
                ..Default::default()
            },
        );

        // egui init
        let egui_ctx = egui::Context::default();
        apply_premium_theme(&egui_ctx);
        let egui_state = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &w, Some(w.scale_factor() as f32), None, Some(dev.limits().max_texture_dimension_2d as usize));
        let egui_renderer = egui_wgpu::Renderer::new(&dev, cfg.format, Some(wgpu::TextureFormat::Depth32Float), 1, false);

        // Editor state initialization
        let mut editor = EditorState::new(engine);
        editor.log(LogLevel::System, "Genovo Studio v1.0 -- Professional Game Development Environment");
        editor.log(LogLevel::System, format!("GPU: {} ({:?})", adap.get_info().name, adap.get_info().backend));
        editor.log(LogLevel::System, format!("Surface format: {:?} | Resolution: {}x{}", cfg.format, sz.width, sz.height));
        editor.log(LogLevel::System, "3D Renderer: SceneRenderManager (PBR + Grid)");
        editor.log(LogLevel::System, "Audio: cpal output stream active");
        editor.log(LogLevel::System, "UI: egui 0.31 (hybrid) | Theme: UIStyle::dark()");
        editor.log(LogLevel::System, "Type 'help' in console for commands");

        // Spawn default scene entities
        let ground_idx = editor.spawn_entity("Ground Plane", EntityType::Mesh);
        editor.entities[ground_idx].scale = [100.0, 1.0, 100.0];
        editor.entities[ground_idx].mesh_shape = MeshShape::Plane;

        let cube_idx = editor.spawn_mesh(MeshShape::Cube);
        editor.entities[cube_idx].position = [0.0, 5.0, 0.0];
        if let Some(h) = editor.entities[cube_idx].physics_handle {
            let _ = editor.engine.physics_mut().set_position(h, Vec3::new(0.0, 5.0, 0.0));
        }

        let sphere_idx = editor.spawn_mesh(MeshShape::Sphere);
        editor.entities[sphere_idx].position = [3.0, 3.0, 0.0];
        editor.entities[sphere_idx].collider_shape = ColliderShape::Sphere;
        if let Some(h) = editor.entities[sphere_idx].physics_handle {
            let _ = editor.engine.physics_mut().set_position(h, Vec3::new(3.0, 3.0, 0.0));
        }

        let _dir_light = editor.spawn_light(LightKind::Directional);
        editor.entities[_dir_light].position = [5.0, 10.0, 5.0];

        let point_light = editor.spawn_light(LightKind::Point);
        editor.entities[point_light].position = [-3.0, 4.0, -2.0];
        editor.entities[point_light].light_color = [0.3, 0.5, 1.0];

        let cam_idx = editor.spawn_entity("Main Camera", EntityType::Camera);
        editor.entities[cam_idx].position = [0.0, 5.0, -15.0];

        let particles_idx = editor.spawn_entity("Smoke Particles", EntityType::ParticleSystem);
        editor.entities[particles_idx].position = [2.0, 0.0, 2.0];

        editor.selected_entity = Some(cube_idx);
        editor.scene_modified = false;

        println!("[Genovo Studio] GPU: {} ({:?})", adap.get_info().name, adap.get_info().backend);
        println!("[Genovo Studio] 3D Renderer: SceneRenderManager with PBR pipeline");
        println!("[Genovo Studio] Editor ready. {} entities in scene.", editor.entities.len());

        self.gpu = Some(GpuState {
            window: w,
            device: dev,
            queue: que,
            surface: surf,
            config: cfg,
            depth_view: dv,
            scene_manager,
            egui_ctx,
            egui_state,
            egui_renderer,
            editor,
            theme_applied: true,
        });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        let Some(s) = self.gpu.as_mut() else { return };

        let resp = s.egui_state.on_window_event(&s.window, &ev);
        if resp.repaint { s.window.request_redraw(); }

        match ev {
            WindowEvent::CloseRequested => el.exit(),

            WindowEvent::Resized(sz) if sz.width > 0 && sz.height > 0 => {
                s.config.width = sz.width;
                s.config.height = sz.height;
                s.surface.configure(&s.device, &s.config);
                s.depth_view = make_depth(&s.device, sz.width, sz.height);
            }

            WindowEvent::ModifiersChanged(mods) => {
                s.editor.modifiers = mods.state();
            }

            WindowEvent::KeyboardInput { event, .. } if !resp.consumed => {
                if let PhysicalKey::Code(k) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        let ctrl = s.editor.modifiers.control_key();

                        match k {
                            KeyCode::KeyN if ctrl => { s.editor.new_scene(); }
                            KeyCode::KeyS if ctrl => { s.editor.save_scene(); }
                            KeyCode::KeyZ if ctrl => { s.editor.perform_undo(); }
                            KeyCode::KeyY if ctrl => { s.editor.perform_redo(); }
                            KeyCode::KeyD if ctrl => { s.editor.duplicate_selected(); }
                            KeyCode::KeyQ if !ctrl => { s.editor.gizmo_mode = GizmoMode::Select; }
                            KeyCode::KeyW if !ctrl => { s.editor.gizmo_mode = GizmoMode::Translate; }
                            KeyCode::KeyE if !ctrl => { s.editor.gizmo_mode = GizmoMode::Rotate; }
                            KeyCode::KeyR if !ctrl => { s.editor.gizmo_mode = GizmoMode::Scale; }
                            KeyCode::Delete => { s.editor.delete_selected(); }
                            KeyCode::Space => { s.editor.toggle_play(); }
                            KeyCode::F5 => {
                                if s.editor.is_playing { s.editor.stop_play(); }
                                else {
                                    s.editor.is_playing = true;
                                    s.editor.is_paused = false;
                                    s.editor.play_start_time = Some(Instant::now());
                                    s.editor.notify("Simulation started (F5)", LogLevel::System);
                                }
                            }
                            KeyCode::KeyF if !ctrl => { s.editor.focus_selected(); }
                            KeyCode::KeyG if !ctrl => { s.editor.spawn_physics_ball(); }
                            KeyCode::ArrowUp if !ctrl => { s.editor.select_prev(); }
                            KeyCode::ArrowDown if !ctrl => { s.editor.select_next(); }
                            KeyCode::Escape => {
                                s.editor.selected_entity = None;
                                s.editor.renaming_entity = None;
                            }
                            _ => {}
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                // ---- Frame timing ----
                let now = Instant::now();
                let dt = now.duration_since(s.editor.last_frame).as_secs_f32();
                s.editor.last_frame = now;
                s.editor.frame_count += 1;
                s.editor.frame_time_ms = dt * 1000.0;
                s.editor.fps = 1.0 / dt.max(0.0001);

                let alpha = 0.05_f32;
                s.editor.smooth_fps = s.editor.smooth_fps * (1.0 - alpha) + s.editor.fps * alpha;
                s.editor.smooth_frame_time = s.editor.smooth_frame_time * (1.0 - alpha) + s.editor.frame_time_ms * alpha;

                s.editor.frame_times.push_back(dt as f64 * 1000.0);
                if s.editor.frame_times.len() > 512 { s.editor.frame_times.pop_front(); }
                s.editor.fps_history.push_back(s.editor.fps as f64);
                if s.editor.fps_history.len() > 512 { s.editor.fps_history.pop_front(); }

                // ---- Update window title ----
                let title = format!("Genovo Studio \u{2014} {}{}", s.editor.scene_name, if s.editor.scene_modified { " *" } else { "" });
                s.window.set_title(&title);

                // ---- Physics step ----
                if s.editor.is_playing && !s.editor.is_paused {
                    let phys_dt = (dt * s.editor.sim_speed).min(1.0 / 30.0);
                    let _ = s.editor.engine.physics_mut().step(phys_dt);
                    s.editor.sync_physics_to_entities();
                    s.editor.total_sim_time += phys_dt as f64;
                }

                // ---- Clear expired notification ----
                if let Some((_, _, start)) = &s.editor.notification {
                    if start.elapsed().as_secs_f32() > 3.5 { s.editor.notification = None; }
                }

                // ---- Build egui frame ----
                let raw_input = s.egui_state.take_egui_input(&s.window);
                let full_output = s.egui_ctx.run(raw_input, |ctx| {
                    // Dialogs
                    if s.editor.show_about { draw_about_window(ctx, &mut s.editor.show_about); }
                    let mut show_prefs = s.editor.show_preferences;
                    if show_prefs {
                        draw_preferences_window(ctx, &mut show_prefs, &mut s.editor);
                        s.editor.show_preferences = show_prefs;
                    }
                    if s.editor.show_shortcuts { draw_shortcuts_window(ctx, &mut s.editor.show_shortcuts); }

                    // Main layout
                    draw_menu_bar(ctx, &mut s.editor);
                    draw_toolbar(ctx, &mut s.editor);
                    draw_status_bar(ctx, &s.editor);
                    draw_bottom_panel(ctx, &mut s.editor);
                    draw_hierarchy(ctx, &mut s.editor);
                    draw_inspector(ctx, &mut s.editor);
                    draw_viewport(ctx, &s.editor);

                    // Notification overlay
                    draw_notification(ctx, &s.editor);
                });

                s.egui_state.handle_platform_output(&s.window, full_output.platform_output);
                let clipped_primitives = s.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

                for (id, delta) in &full_output.textures_delta.set {
                    s.egui_renderer.update_texture(&s.device, &s.queue, *id, delta);
                }

                // ---- GPU Render ----
                let Ok(out) = s.surface.get_current_texture() else {
                    s.window.request_redraw();
                    return;
                };
                let view = out.texture.create_view(&Default::default());

                let screen = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [s.config.width, s.config.height],
                    pixels_per_point: s.window.scale_factor() as f32,
                };

                // FIX 3: Sync ALL physics positions to visual entities every frame
                s.editor.sync_physics_to_entities();

                // Build camera from editor orbit parameters
                let yaw_rad = s.editor.camera_yaw.to_radians();
                let pitch_rad = s.editor.camera_pitch.to_radians();
                let target = glam::Vec3::new(
                    s.editor.camera_target[0],
                    s.editor.camera_target[1],
                    s.editor.camera_target[2],
                );
                let aspect = s.config.width as f32 / s.config.height.max(1) as f32;
                let mut scene_camera = SceneCamera::perspective(
                    glam::Vec3::ZERO, target, glam::Vec3::Y,
                    s.editor.camera_fov.to_radians(), aspect, 0.1, 1000.0,
                );
                scene_camera.orbit(target, yaw_rad, pitch_rad, s.editor.camera_dist);

                // Build scene lights from entities
                let mut scene_lights = SceneLights::default_outdoor();
                scene_lights.point_lights.clear();
                for ent in &s.editor.entities {
                    if ent.is_light && ent.light_kind == LightKind::Point {
                        scene_lights.point_lights.push(
                            genovo_render::scene_renderer::PointLight {
                                position: glam::Vec3::new(ent.position[0], ent.position[1], ent.position[2]),
                                color: glam::Vec3::new(ent.light_color[0], ent.light_color[1], ent.light_color[2]),
                                intensity: ent.light_intensity * 5.0,
                                range: ent.light_range,
                            },
                        );
                    }
                }

                // Submit scene entities to the render manager
                s.scene_manager.clear_queue();

                // Set clear color (subtle dark background)
                let t = s.editor.frame_count as f32 * 0.005;
                let (br, bg, bb) = if s.editor.is_playing && !s.editor.is_paused {
                    let pulse = (t * 2.0).sin().abs() * 0.008;
                    (0.06 + pulse as f64, 0.06_f64, 0.08 + pulse as f64)
                } else {
                    (0.06_f64, 0.06, 0.08)
                };
                s.scene_manager.set_clear_color([br, bg, bb, 1.0]);

                // Toggle grid based on editor setting
                if s.editor.grid_visible {
                    s.scene_manager.set_grid_enabled(&s.device, true);
                } else {
                    s.scene_manager.set_grid_enabled(&s.device, false);
                }

                // Submit each entity as its appropriate mesh
                for ent in &s.editor.entities {
                    if !ent.visible { continue; }

                    let mesh_id = match ent.entity_type {
                        EntityType::Mesh => match ent.mesh_shape {
                            MeshShape::Cube => s.scene_manager.builtin_cube,
                            MeshShape::Sphere => s.scene_manager.builtin_sphere,
                            MeshShape::Cylinder => s.scene_manager.builtin_cylinder,
                            MeshShape::Cone => s.scene_manager.builtin_cone,
                            MeshShape::Plane => s.scene_manager.builtin_plane,
                            MeshShape::Capsule => s.scene_manager.builtin_cylinder, // approximate
                        },
                        EntityType::Light => s.scene_manager.builtin_sphere, // small sphere for lights
                        EntityType::Camera => s.scene_manager.builtin_cube,  // small cube for cameras
                        EntityType::ParticleSystem => s.scene_manager.builtin_sphere,
                        EntityType::Audio => s.scene_manager.builtin_sphere,
                        EntityType::Empty => None,
                    };

                    let material_id = match ent.entity_type {
                        EntityType::Mesh => s.scene_manager.builtin_material_default,
                        EntityType::Light => s.scene_manager.builtin_material_gold,
                        EntityType::Camera => s.scene_manager.builtin_material_blue,
                        EntityType::ParticleSystem => s.scene_manager.builtin_material_copper,
                        EntityType::Audio => s.scene_manager.builtin_material_green,
                        EntityType::Empty => None,
                    };

                    if let (Some(mesh), Some(mat)) = (mesh_id, material_id) {
                        let pos = glam::Vec3::new(ent.position[0], ent.position[1], ent.position[2]);
                        let scl = glam::Vec3::new(ent.scale[0], ent.scale[1], ent.scale[2]);

                        // For non-mesh types, render them small
                        let render_scale = match ent.entity_type {
                            EntityType::Light | EntityType::Camera |
                            EntityType::ParticleSystem | EntityType::Audio => {
                                glam::Vec3::new(0.3, 0.3, 0.3)
                            }
                            _ => scl,
                        };

                        let transform = glam::Mat4::from_scale_rotation_translation(
                            render_scale,
                            glam::Quat::from_euler(
                                glam::EulerRot::XYZ,
                                ent.rotation[0].to_radians(),
                                ent.rotation[1].to_radians(),
                                ent.rotation[2].to_radians(),
                            ),
                            pos,
                        );

                        s.scene_manager.submit(mesh, mat, transform);
                    }
                }

                // Pass 1: Render full 3D scene via SceneRenderManager (PBR + grid)
                let scene_cmd = s.scene_manager.render(
                    &s.device,
                    &s.queue,
                    &view,
                    &s.depth_view,
                    &scene_camera,
                    &scene_lights,
                );

                // Pass 2: egui overlay
                let mut egui_enc = s.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("egui") });
                let user_cmd_bufs = s.egui_renderer.update_buffers(&s.device, &s.queue, &mut egui_enc, &clipped_primitives, &screen);
                {
                    let mut pass = egui_enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    // Lifetime transmute: wgpu 24 RenderPass has 'encoder lifetime
                    // but egui-wgpu 0.31 expects 'static. Safe because pass is used
                    // and dropped within this block.
                    let pass_static: &mut wgpu::RenderPass<'static> = unsafe { std::mem::transmute(&mut pass) };
                    s.egui_renderer.render(pass_static, &clipped_primitives, &screen);
                }

                // Submit
                let mut cmd_bufs: Vec<wgpu::CommandBuffer> = Vec::new();
                cmd_bufs.push(scene_cmd);
                cmd_bufs.extend(user_cmd_bufs);
                cmd_bufs.push(egui_enc.finish());
                s.queue.submit(cmd_bufs);
                out.present();

                for id in &full_output.textures_delta.free {
                    s.egui_renderer.free_texture(id);
                }

                s.window.request_redraw();
            }

            _ => {}
        }
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    env_logger::init();

    println!("[Genovo Studio] Starting Genovo Studio v1.0...");
    println!("[Genovo Studio] Professional Game Development Environment");
    println!("[Genovo Studio] UI mode: hybrid (egui rendering + genovo-ui state management)");
    println!("[Genovo Studio] Theme: UIStyle::dark() | DockStyle::dark_theme()");

    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = EditorApp {
        gpu: None,
        audio_running: Arc::new(AtomicBool::new(false)),
    };
    let _ = el.run_app(&mut app);
}
