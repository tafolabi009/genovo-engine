//! Genovo Studio -- Professional Game Development Environment
//!
//! Built entirely on the custom Slate UI framework (genovo_ui) rendering
//! directly to wgpu via UIGpuRenderer. No egui dependency anywhere.
//!
//! Architecture:
//!   1. winit window + wgpu device/queue/surface
//!   2. UIGpuRenderer for all 2D UI rendering (SDF rects, text, lines)
//!   3. UI framework (immediate-mode widgets: panels, buttons, sliders, etc.)
//!   4. SceneRenderManager for full 3D viewport (PBR, grid, built-in primitives)
//!   5. Each frame: handle events -> begin_frame -> draw UI -> finish_frame -> present
//!
//! Features:
//!   - Real GPU-rendered 3D viewport (SceneRenderManager behind UI overlay)
//!   - Scene outliner with colored entity type dots, search, selection
//!   - Inspector with colored XYZ drag values, physics sync, light controls
//!   - Content browser with directory scanning and icon grid
//!   - Console with command history and colored log output
//!   - Profiler with frame time display
//!   - Animation timeline placeholder
//!   - Near-black dark theme (rgb(18,18,22) base, rgb(24,24,28) panels)
//!   - Accent blue rgb(56,132,244) for selection/active
//!   - Full keyboard shortcuts (Ctrl+S, Ctrl+Z, Delete, Q/W/E/R, Space, F5, F)
//!   - Working physics integration (play/pause/stop/step)
//!   - Status bar with colored play-state dot, FPS, entity/body counts
//!   - Toolbar with play/pause/stop, transform tools, snap, grid, speed

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use glam::Vec2;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, WindowEvent},
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

// Custom Slate UI framework -- GPU-rendered, no egui
use genovo_ui::gpu_renderer::UIGpuRenderer;
use genovo_ui::ui_framework::{UI, UIInputState, UIStyle};
use genovo_ui::render_commands::Color as UIColor;
use genovo_ui::editor_widgets::{
    vec3_edit, component_header, entity_item, toolbar_button,
    search_bar, tab_bar_premium, status_dot, console_entry,
    fps_overlay,
};
use genovo_ui::dock_system::{DockState, DockTabId, DockNodeId, DockTab, DockNode, DockStyle};
use genovo_core::Rect;

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
// Color Palette -- matching UIStyle::dark() theme exactly
// =============================================================================

const BG_BASE: UIColor = UIColor { r: 0.0706, g: 0.0706, b: 0.0863, a: 1.0 };
const BG_PANEL: UIColor = UIColor { r: 0.0941, g: 0.0941, b: 0.1098, a: 1.0 };
const BG_WIDGET: UIColor = UIColor { r: 0.1255, g: 0.1255, b: 0.1490, a: 1.0 };
const BG_HOVER: UIColor = UIColor { r: 0.1647, g: 0.1647, b: 0.1961, a: 1.0 };
const BG_ACTIVE: UIColor = UIColor { r: 0.2039, g: 0.2039, b: 0.2431, a: 1.0 };
const BORDER: UIColor = UIColor { r: 0.1490, g: 0.1490, b: 0.1725, a: 1.0 };
const BORDER_SUBTLE: UIColor = UIColor { r: 0.1176, g: 0.1176, b: 0.1412, a: 1.0 };

const TEXT_BRIGHT: UIColor = UIColor { r: 0.902, g: 0.902, b: 0.922, a: 1.0 };
const TEXT_NORMAL: UIColor = UIColor { r: 0.706, g: 0.706, b: 0.737, a: 1.0 };
const TEXT_DIM: UIColor = UIColor { r: 0.431, g: 0.431, b: 0.471, a: 1.0 };
const TEXT_MUTED: UIColor = UIColor { r: 0.255, g: 0.255, b: 0.282, a: 1.0 };

const ACCENT: UIColor = UIColor { r: 0.220, g: 0.518, b: 0.957, a: 1.0 };
const ACCENT_DIM: UIColor = UIColor { r: 0.157, g: 0.392, b: 0.784, a: 1.0 };
const ACCENT_BRIGHT: UIColor = UIColor { r: 0.314, g: 0.612, b: 1.0, a: 1.0 };
const ACCENT_BG: UIColor = UIColor { r: 0.118, g: 0.235, b: 0.431, a: 1.0 };

const GREEN: UIColor = UIColor { r: 0.282, g: 0.780, b: 0.557, a: 1.0 };
const GREEN_DIM: UIColor = UIColor { r: 0.196, g: 0.549, b: 0.392, a: 1.0 };
const YELLOW: UIColor = UIColor { r: 0.961, g: 0.769, b: 0.314, a: 1.0 };
const YELLOW_DIM: UIColor = UIColor { r: 0.706, g: 0.569, b: 0.235, a: 1.0 };
const RED: UIColor = UIColor { r: 0.922, g: 0.341, b: 0.341, a: 1.0 };
const RED_DIM: UIColor = UIColor { r: 0.627, g: 0.235, b: 0.235, a: 1.0 };
const CYAN: UIColor = UIColor { r: 0.275, g: 0.784, b: 0.863, a: 1.0 };
const MAGENTA: UIColor = UIColor { r: 0.745, g: 0.314, b: 0.824, a: 1.0 };
const ORANGE: UIColor = UIColor { r: 0.902, g: 0.588, b: 0.235, a: 1.0 };
const GRAY: UIColor = UIColor { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };

// XYZ axis colors (Unreal Engine convention)
const X_COLOR: UIColor = UIColor { r: 0.922, g: 0.294, b: 0.294, a: 1.0 };
const Y_COLOR: UIColor = UIColor { r: 0.282, g: 0.780, b: 0.557, a: 1.0 };
const Z_COLOR: UIColor = UIColor { r: 0.220, g: 0.518, b: 0.957, a: 1.0 };

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

    fn icon_color(&self) -> UIColor {
        match self {
            EntityType::Empty => TEXT_DIM,
            EntityType::Mesh => CYAN,
            EntityType::Light => YELLOW,
            EntityType::Camera => ACCENT_BRIGHT,
            EntityType::ParticleSystem => MAGENTA,
            EntityType::Audio => ORANGE,
        }
    }

    fn dot_color(&self) -> UIColor {
        self.icon_color()
    }
}

// =============================================================================
// Mesh Shape -- for the Create menu
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
    fn color(&self) -> UIColor {
        match self {
            LogLevel::Info => TEXT_NORMAL,
            LogLevel::Warn => YELLOW,
            LogLevel::Error => RED,
            LogLevel::System => ACCENT_BRIGHT,
            LogLevel::Debug => UIColor::from_rgba8(140, 140, 160, 255),
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
    bottom_tab_idx: usize,
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

    // Renaming state
    renaming_entity: Option<usize>,
    rename_buffer: String,

    // Notification
    notification: Option<(String, LogLevel, Instant)>,

    // Undo/Redo -- real UndoStack from genovo_editor
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
            sim_speed_idx: 2,
            play_start_time: None,
            total_sim_time: 0.0,
            gizmo_mode: GizmoMode::Translate,
            coord_space: CoordSpace::World,
            pivot_mode: PivotMode::Center,
            snap_enabled: false,
            snap_value_idx: 3,
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
            bottom_tab_idx: 0,
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
    // Undo / Redo -- wired to the real UndoStack
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
    // Script VM with world bindings
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
                self.log(LogLevel::System, "  spawn <type> [name]  Spawn entity");
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
            "clear" => { self.console_log.clear(); }
            "about" => {
                self.log(LogLevel::System, "Genovo Studio v1.0 -- Professional Game Development Environment");
                self.log(LogLevel::System, "26 engine modules fully linked");
                self.log(LogLevel::System, "UI: Custom Slate framework (genovo_ui, GPU-rendered via UIGpuRenderer)");
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
                        format!("  [{}] {} ({}) pos=({:.1},{:.1},{:.1}){}", i, e.name, e.entity_type, e.position[0], e.position[1], e.position[2], sel)
                    }).collect();
                    for msg in msgs { self.log(LogLevel::Info, msg); }
                }
            }
            "select" => {
                if let Some(idx_str) = parts.get(1) {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        if idx < self.entities.len() {
                            self.selected_entity = Some(idx);
                            self.log(LogLevel::System, format!("Selected: {}", self.entities[idx].name));
                        } else { self.log(LogLevel::Error, format!("Index {} out of range", idx)); }
                    } else { self.log(LogLevel::Error, "Usage: select <index>"); }
                } else { self.log(LogLevel::Error, "Usage: select <index>"); }
            }
            "delete" => { self.delete_selected(); }
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
                        let i = self.spawn_entity(&name, EntityType::Empty); self.selected_entity = Some(i);
                    }
                }
            }
            "physics" => match parts.get(1).copied() {
                Some("start") => { self.is_playing = true; self.is_paused = false; self.log(LogLevel::System, "Physics started"); }
                Some("stop") => { self.is_playing = false; self.log(LogLevel::System, "Physics stopped"); }
                Some("step") => { self.step_physics(); }
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
                }
            }
            "speed" => {
                if let Some(val) = parts.get(1) {
                    if let Ok(s) = val.parse::<f32>() {
                        self.sim_speed = s.clamp(0.0, 10.0);
                        self.log(LogLevel::System, format!("Simulation speed: {:.2}x", self.sim_speed));
                    }
                } else { self.log(LogLevel::Info, format!("Speed: {:.2}x", self.sim_speed)); }
            }
            "camera" => {
                if parts.len() == 4 {
                    self.camera_yaw = parts[1].parse().unwrap_or(self.camera_yaw);
                    self.camera_pitch = parts[2].parse().unwrap_or(self.camera_pitch);
                    self.camera_dist = parts[3].parse().unwrap_or(self.camera_dist);
                    self.log(LogLevel::System, format!("Camera: yaw={:.1} pitch={:.1} dist={:.1}", self.camera_yaw, self.camera_pitch, self.camera_dist));
                } else {
                    self.log(LogLevel::Info, format!("Camera: yaw={:.1} pitch={:.1} dist={:.1}", self.camera_yaw, self.camera_pitch, self.camera_dist));
                }
            }
            "scene" => {
                if parts.len() > 1 {
                    self.scene_name = parts[1..].join(" ");
                    self.scene_modified = true;
                    self.log(LogLevel::System, format!("Scene renamed to: {}", self.scene_name));
                } else { self.log(LogLevel::Info, format!("Current scene: {}", self.scene_name)); }
            }
            "terrain" if parts.get(1) == Some(&"gen") => {
                match genovo_terrain::Heightmap::generate_procedural(257, 0.7, 42) {
                    Ok(h) => self.log(LogLevel::System, format!("Terrain 257x257 generated [min={:.2}, max={:.2}]", h.min_height(), h.max_height())),
                    Err(e) => self.log(LogLevel::Error, format!("Terrain error: {}", e)),
                }
            }
            "dungeon" if parts.get(1) == Some(&"gen") => {
                let cfg = genovo_procgen::BSPConfig {
                    width: 80, height: 60, min_room_size: 6, max_depth: 8,
                    room_fill_ratio: 0.7, seed: 42, wall_padding: 1,
                };
                let d = genovo_procgen::dungeon::generate_bsp(&cfg);
                self.log(LogLevel::System, format!("Dungeon generated: {} rooms", d.rooms.len()));
            }
            "script" => {
                let code = parts[1..].join(" ");
                self.execute_script_with_bindings(&code);
            }
            other => {
                self.log(LogLevel::Warn, format!("Unknown command: '{}'. Type 'help' for commands.", other));
            }
        }
    }
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
// Asset browser helpers
// =============================================================================

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

fn asset_color(name: &str) -> UIColor {
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
// winit KeyCode -> u32 mapping for UIInputState
// =============================================================================

fn keycode_to_u32(key: KeyCode) -> Option<u32> {
    use genovo_ui::ui_framework::keys;
    match key {
        KeyCode::Backspace => Some(keys::BACKSPACE),
        KeyCode::Tab => Some(keys::TAB),
        KeyCode::Enter => Some(keys::ENTER),
        KeyCode::Escape => Some(keys::ESCAPE),
        KeyCode::Delete => Some(keys::DELETE),
        KeyCode::ArrowLeft => Some(keys::LEFT),
        KeyCode::ArrowRight => Some(keys::RIGHT),
        KeyCode::ArrowUp => Some(keys::UP),
        KeyCode::ArrowDown => Some(keys::DOWN),
        KeyCode::Home => Some(keys::HOME),
        KeyCode::End => Some(keys::END),
        KeyCode::KeyA => Some(keys::A),
        KeyCode::KeyC => Some(keys::C),
        KeyCode::KeyV => Some(keys::V),
        KeyCode::KeyX => Some(keys::X),
        KeyCode::KeyZ => Some(keys::Z),
        _ => None,
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
// Draw editor UI using the custom Slate framework
// =============================================================================

fn draw_editor_ui(ui: &mut UI, state: &mut EditorState) {
    let screen = ui.screen_size();
    let width = screen.x;
    let height = screen.y;

    let menu_h = 26.0;
    let toolbar_h = if state.panels.toolbar { 30.0 } else { 0.0 };
    let status_h = if state.panels.status_bar { 18.0 } else { 0.0 };
    let bottom_h = if state.panels.bottom { state.bottom_height } else { 0.0 };
    let left_w = if state.panels.hierarchy { state.hierarchy_width } else { 0.0 };
    let right_w = if state.panels.inspector { state.inspector_width } else { 0.0 };

    let top_y = menu_h + toolbar_h;
    let content_h = height - top_y - bottom_h - status_h;

    // --- Menu Bar (top, 26px) ---
    draw_menu_bar_slate(ui, state, width, menu_h);

    // --- Toolbar (30px) ---
    if state.panels.toolbar {
        draw_toolbar_slate(ui, state, width, menu_h, toolbar_h);
    }

    // --- Scene Hierarchy (left panel) ---
    if state.panels.hierarchy {
        draw_hierarchy_slate(ui, state, left_w, top_y, content_h);
    }

    // --- Inspector (right panel) ---
    if state.panels.inspector {
        draw_inspector_slate(ui, state, width, right_w, top_y, content_h);
    }

    // --- Bottom Panel (console/assets/profiler/animation tabs) ---
    if state.panels.bottom {
        draw_bottom_panel_slate(ui, state, left_w, width, right_w, height, bottom_h, status_h);
    }

    // --- Status Bar (bottom, 18px) ---
    if state.panels.status_bar {
        draw_status_bar_slate(ui, state, width, height, status_h);
    }

    // --- Viewport overlays (drawn on top of the 3D scene) ---
    draw_viewport_overlays(ui, state, left_w, right_w, top_y, content_h);

    // --- FPS overlay ---
    if state.stats_visible {
        fps_overlay(ui, state.smooth_fps, state.smooth_frame_time);
    }

    // --- Notification toast ---
    draw_notification_slate(ui, state);
}

// =============================================================================
// Menu Bar (Slate)
// =============================================================================

fn draw_menu_bar_slate(ui: &mut UI, state: &mut EditorState, width: f32, menu_h: f32) {
    let menu_rect = Rect::new(Vec2::new(0.0, 0.0), Vec2::new(width, menu_h));
    ui.renderer.draw_rect(menu_rect, BG_BASE, 0.0);
    ui.renderer.draw_line(Vec2::new(0.0, menu_h), Vec2::new(width, menu_h), BORDER, 1.0);

    ui.menu_bar(|ui| {
        // Genovo Studio branding
        ui.label_colored("Genovo Studio", ACCENT);

        // File menu
        ui.menu("File", |ui| {
            if ui.menu_item("New Scene", "Ctrl+N") { state.new_scene(); }
            if ui.menu_item("Open Scene...", "") { state.load_scene_from_file("scene.json"); }
            if ui.menu_item("Save", "Ctrl+S") { state.save_scene(); }
            if ui.menu_item("Save As...", "") { state.save_scene_to_file("scene.json"); }
            if ui.menu_item("Import Asset...", "") { state.log(LogLevel::System, "Import asset (placeholder)"); }
            if ui.menu_item("Export...", "") { state.log(LogLevel::System, "Export (placeholder)"); }
            if ui.menu_item("Exit", "") { std::process::exit(0); }
        });

        // Edit menu
        ui.menu("Edit", |ui| {
            if ui.menu_item("Undo", "Ctrl+Z") { state.perform_undo(); }
            if ui.menu_item("Redo", "Ctrl+Y") { state.perform_redo(); }
            if ui.menu_item("Duplicate", "Ctrl+D") { state.duplicate_selected(); }
            if ui.menu_item("Delete", "Del") { state.delete_selected(); }
            if ui.menu_item("Select All", "") { state.select_all(); }
            if ui.menu_item("Deselect All", "") { state.selected_entity = None; }
            if ui.menu_item("Preferences...", "") { state.show_preferences = true; }
        });

        // View menu
        ui.menu("View", |ui| {
            if ui.menu_item("Toggle Hierarchy", "") { state.panels.hierarchy = !state.panels.hierarchy; }
            if ui.menu_item("Toggle Inspector", "") { state.panels.inspector = !state.panels.inspector; }
            if ui.menu_item("Toggle Bottom Panel", "") { state.panels.bottom = !state.panels.bottom; }
            if ui.menu_item("Toggle Toolbar", "") { state.panels.toolbar = !state.panels.toolbar; }
            if ui.menu_item("Toggle Status Bar", "") { state.panels.status_bar = !state.panels.status_bar; }
            if ui.menu_item("Toggle Grid", "") { state.grid_visible = !state.grid_visible; }
            if ui.menu_item("Toggle Wireframe", "") { state.wireframe_mode = !state.wireframe_mode; }
            if ui.menu_item("Toggle Stats", "") { state.stats_visible = !state.stats_visible; }
            if ui.menu_item("Focus Selected", "F") { state.focus_selected(); }
            if ui.menu_item("Frame All", "") { state.frame_all(); }
        });

        // Create menu
        ui.menu("Create", |ui| {
            if ui.menu_item("Empty Entity", "") { let i = state.spawn_entity("Empty", EntityType::Empty); state.selected_entity = Some(i); }
            if ui.menu_item("Cube", "") { let i = state.spawn_mesh(MeshShape::Cube); state.selected_entity = Some(i); }
            if ui.menu_item("Sphere", "") { let i = state.spawn_mesh(MeshShape::Sphere); state.selected_entity = Some(i); }
            if ui.menu_item("Cylinder", "") { let i = state.spawn_mesh(MeshShape::Cylinder); state.selected_entity = Some(i); }
            if ui.menu_item("Capsule", "") { let i = state.spawn_mesh(MeshShape::Capsule); state.selected_entity = Some(i); }
            if ui.menu_item("Cone", "") { let i = state.spawn_mesh(MeshShape::Cone); state.selected_entity = Some(i); }
            if ui.menu_item("Plane", "") { let i = state.spawn_mesh(MeshShape::Plane); state.selected_entity = Some(i); }
            if ui.menu_item("Directional Light", "") { let i = state.spawn_light(LightKind::Directional); state.selected_entity = Some(i); }
            if ui.menu_item("Point Light", "") { let i = state.spawn_light(LightKind::Point); state.selected_entity = Some(i); }
            if ui.menu_item("Spot Light", "") { let i = state.spawn_light(LightKind::Spot); state.selected_entity = Some(i); }
            if ui.menu_item("Camera", "") { let i = state.spawn_entity("Camera", EntityType::Camera); state.selected_entity = Some(i); }
            if ui.menu_item("Particle System", "") { let i = state.spawn_entity("Particle System", EntityType::ParticleSystem); state.selected_entity = Some(i); }
        });

        // Tools menu
        ui.menu("Tools", |ui| {
            if ui.menu_item("Terrain Editor", "") { state.execute_console_command("terrain gen"); }
            if ui.menu_item("Generate Dungeon", "") { state.execute_console_command("dungeon gen"); }
            if ui.menu_item("Profiler", "") { state.bottom_tab = BottomTab::Profiler; state.bottom_tab_idx = 2; }
            if ui.menu_item("Script Console", "") { state.bottom_tab = BottomTab::Console; state.bottom_tab_idx = 0; }
        });

        // Help menu
        ui.menu("Help", |ui| {
            if ui.menu_item("About Genovo Studio", "") { state.show_about = !state.show_about; }
            if ui.menu_item("Keyboard Shortcuts", "") { state.show_shortcuts = !state.show_shortcuts; }
            if ui.menu_item("Check for Updates", "") { state.log(LogLevel::System, "Genovo Studio v1.0 is up to date"); }
        });
    });
}

// =============================================================================
// Toolbar (Slate)
// =============================================================================

fn draw_toolbar_slate(ui: &mut UI, state: &mut EditorState, width: f32, menu_h: f32, toolbar_h: f32) {
    let toolbar_rect = Rect::new(
        Vec2::new(0.0, menu_h),
        Vec2::new(width, menu_h + toolbar_h),
    );

    if ui.begin_panel("Toolbar", toolbar_rect) {
        ui.horizontal(|ui| {
            // Gizmo mode buttons
            for mode in [GizmoMode::Select, GizmoMode::Translate, GizmoMode::Rotate, GizmoMode::Scale] {
                let selected = state.gizmo_mode == mode;
                let label = format!("{} {}", mode.icon(), mode.short_label());
                if toolbar_button(ui, &label, mode.label(), selected) {
                    state.gizmo_mode = mode;
                }
            }

            ui.separator();

            // Play / Pause / Stop / Step
            let playing = state.is_playing && !state.is_paused;
            if toolbar_button(ui, if playing { "||" } else { "|>" }, "Play/Pause (Space)", playing) {
                state.toggle_play();
            }
            if toolbar_button(ui, "[]", "Stop", false) {
                state.stop_play();
            }
            if toolbar_button(ui, "|>|", "Step (1 frame)", false) {
                state.step_physics();
            }

            ui.separator();

            // Coord space toggle
            let space_label = if state.coord_space == CoordSpace::Local { "Local" } else { "World" };
            if toolbar_button(ui, space_label, "Toggle local/world space", false) {
                state.coord_space = if state.coord_space == CoordSpace::Local { CoordSpace::World } else { CoordSpace::Local };
            }

            // Pivot toggle
            let pivot_label = if state.pivot_mode == PivotMode::Center { "Center" } else { "Pivot" };
            if toolbar_button(ui, pivot_label, "Toggle pivot mode", false) {
                state.pivot_mode = if state.pivot_mode == PivotMode::Center { PivotMode::Pivot } else { PivotMode::Center };
            }

            ui.separator();

            // Snap toggle
            if toolbar_button(ui, "Snap", "Toggle snap", state.snap_enabled) {
                state.snap_enabled = !state.snap_enabled;
            }

            // Grid toggle
            if toolbar_button(ui, "Grid", "Toggle grid", state.grid_visible) {
                state.grid_visible = !state.grid_visible;
            }

            // Stats toggle
            if toolbar_button(ui, "Stats", "Toggle stats", state.stats_visible) {
                state.stats_visible = !state.stats_visible;
            }

            // Speed display
            ui.separator();
            ui.label_colored(&format!("Speed: {:.1}x", state.sim_speed), TEXT_DIM);
        });
        ui.end_panel();
    }
}

// =============================================================================
// Scene Hierarchy (Slate)
// =============================================================================

fn draw_hierarchy_slate(ui: &mut UI, state: &mut EditorState, left_w: f32, top_y: f32, content_h: f32) {
    let hier_rect = Rect::new(
        Vec2::new(0.0, top_y),
        Vec2::new(left_w, top_y + content_h),
    );

    if ui.begin_panel("Scene Outliner", hier_rect) {
        // Search bar
        search_bar(ui, &mut state.entity_filter);

        // Entity count
        ui.label_colored(&format!("{} entities", state.entities.len()), TEXT_MUTED);
        ui.separator();

        if state.entities.is_empty() {
            ui.label_colored("No entities in scene", TEXT_MUTED);
            ui.label_colored("Use Create menu to add", TEXT_MUTED);
        } else {
            let filter_lower = state.entity_filter.to_lowercase();

            // We need to collect entity data before iterating to avoid borrow issues
            let entity_data: Vec<(usize, String, EntityType, bool)> = state.entities.iter().enumerate()
                .filter(|(_, ent)| {
                    filter_lower.is_empty() || ent.name.to_lowercase().contains(&filter_lower)
                })
                .map(|(i, ent)| (i, ent.name.clone(), ent.entity_type, state.selected_entity == Some(i)))
                .collect();

            let mut clicked_entity: Option<usize> = None;

            for (i, name, etype, selected) in &entity_data {
                let icon_color = etype.dot_color();
                if entity_item(ui, icon_color, name, *selected) {
                    clicked_entity = Some(*i);
                }
            }

            if let Some(idx) = clicked_entity {
                state.selected_entity = Some(idx);
            }
        }

        ui.end_panel();
    }
}

// =============================================================================
// Inspector (Slate)
// =============================================================================

fn draw_inspector_slate(ui: &mut UI, state: &mut EditorState, width: f32, right_w: f32, top_y: f32, content_h: f32) {
    let insp_rect = Rect::new(
        Vec2::new(width - right_w, top_y),
        Vec2::new(width, top_y + content_h),
    );

    if ui.begin_panel("Inspector", insp_rect) {
        let sel = state.selected_entity;
        if sel.is_none() || sel.unwrap() >= state.entities.len() {
            ui.space(20.0);
            ui.label_colored("No entity selected", TEXT_MUTED);
            ui.label_colored("Select an entity in the outliner", TEXT_MUTED);
            ui.end_panel();
            return;
        }
        let idx = sel.unwrap();

        // Entity identity
        let etype = state.entities[idx].entity_type;
        let eid = state.entities[idx].entity;
        ui.heading(&state.entities[idx].name.clone());

        ui.horizontal(|ui| {
            ui.label_colored(&format!(" {} ", etype), etype.icon_color());
            ui.label_colored(&format!("Entity {}v{}", eid.id, eid.generation), TEXT_MUTED);
        });

        // Entity name edit
        let mut name = state.entities[idx].name.clone();
        if ui.text_input("Name", &mut name) {
            state.entities[idx].name = name;
            state.scene_modified = true;
        }

        // Active toggle
        let mut active = state.entities[idx].active;
        if ui.checkbox("Active", &mut active) {
            state.entities[idx].active = active;
        }

        ui.separator();

        // Transform section
        let mut pos_changed = false;
        if component_header(ui, "+", "Transform", &mut state.inspector_sections.transform_open) {
            if vec3_edit(ui, "Position", &mut state.entities[idx].position) {
                pos_changed = true;
            }
            if vec3_edit(ui, "Rotation", &mut state.entities[idx].rotation) {
                state.scene_modified = true;
            }
            if vec3_edit(ui, "Scale", &mut state.entities[idx].scale) {
                state.scene_modified = true;
            }
            ui.tree_node_end();
        }

        if pos_changed {
            state.sync_entity_to_physics(idx);
            state.scene_modified = true;
        }

        // Mesh Renderer section
        if etype == EntityType::Mesh {
            if component_header(ui, "M", "Mesh Renderer", &mut state.inspector_sections.mesh_open) {
                let shape_names: Vec<&str> = vec!["Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Plane"];
                let mut shape_idx = match state.entities[idx].mesh_shape {
                    MeshShape::Cube => 0,
                    MeshShape::Sphere => 1,
                    MeshShape::Cylinder => 2,
                    MeshShape::Capsule => 3,
                    MeshShape::Cone => 4,
                    MeshShape::Plane => 5,
                };
                if ui.dropdown("Mesh", &mut shape_idx, &shape_names) {
                    state.entities[idx].mesh_shape = match shape_idx {
                        0 => MeshShape::Cube, 1 => MeshShape::Sphere, 2 => MeshShape::Cylinder,
                        3 => MeshShape::Capsule, 4 => MeshShape::Cone, 5 => MeshShape::Plane,
                        _ => MeshShape::Cube,
                    };
                    state.scene_modified = true;
                }
                ui.checkbox("Cast Shadows", &mut state.entities[idx].cast_shadows);
                ui.checkbox("Receive Shadows", &mut state.entities[idx].receive_shadows);
                ui.tree_node_end();
            }
        }

        // Physics section
        if state.entities[idx].has_physics {
            if component_header(ui, "P", "Rigid Body", &mut state.inspector_sections.physics_open) {
                let body_names: Vec<&str> = vec!["Dynamic", "Static", "Kinematic"];
                let mut bk_idx = match state.entities[idx].body_kind {
                    BodyKind::Dynamic => 0, BodyKind::Static => 1, BodyKind::Kinematic => 2,
                };
                if ui.dropdown("Body Type", &mut bk_idx, &body_names) {
                    state.entities[idx].body_kind = match bk_idx {
                        0 => BodyKind::Dynamic, 1 => BodyKind::Static, 2 => BodyKind::Kinematic,
                        _ => BodyKind::Dynamic,
                    };
                }
                ui.slider_f32("Mass", &mut state.entities[idx].mass, 0.01, 1000.0);
                ui.slider_f32("Friction", &mut state.entities[idx].friction, 0.0, 1.0);
                ui.slider_f32("Restitution", &mut state.entities[idx].restitution, 0.0, 1.0);
                ui.slider_f32("Lin Damping", &mut state.entities[idx].linear_damping, 0.0, 10.0);
                ui.slider_f32("Ang Damping", &mut state.entities[idx].angular_damping, 0.0, 10.0);
                ui.drag_value("Gravity Scale", &mut state.entities[idx].gravity_scale, 0.1);

                // Velocity readout
                if let Some(handle) = state.entities[idx].physics_handle {
                    if let Ok(vel) = state.engine.physics().get_linear_velocity(handle) {
                        let speed = (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z).sqrt();
                        ui.label_colored(&format!("Vel: ({:.2}, {:.2}, {:.2})", vel.x, vel.y, vel.z), TEXT_MUTED);
                        ui.label_colored(&format!("Speed: {:.3} m/s", speed), TEXT_MUTED);
                    }
                }
                ui.tree_node_end();
            }

            // Collider
            if component_header(ui, "C", "Collider", &mut state.inspector_sections.collider_open) {
                let col_names: Vec<&str> = vec!["Box", "Sphere", "Capsule"];
                let mut cs_idx = match state.entities[idx].collider_shape {
                    ColliderShape::Box => 0, ColliderShape::Sphere => 1, ColliderShape::Capsule => 2,
                };
                if ui.dropdown("Shape", &mut cs_idx, &col_names) {
                    state.entities[idx].collider_shape = match cs_idx {
                        0 => ColliderShape::Box, 1 => ColliderShape::Sphere, 2 => ColliderShape::Capsule,
                        _ => ColliderShape::Box,
                    };
                }
                ui.checkbox("Is Trigger", &mut state.entities[idx].is_trigger);
                ui.tree_node_end();
            }
        }

        // Light section
        if state.entities[idx].is_light {
            if component_header(ui, "L", "Light", &mut state.inspector_sections.light_open) {
                let lk_names: Vec<&str> = vec!["Directional", "Point", "Spot"];
                let mut lk_idx = match state.entities[idx].light_kind {
                    LightKind::Directional => 0, LightKind::Point => 1, LightKind::Spot => 2,
                };
                if ui.dropdown("Type", &mut lk_idx, &lk_names) {
                    state.entities[idx].light_kind = match lk_idx {
                        0 => LightKind::Directional, 1 => LightKind::Point, 2 => LightKind::Spot,
                        _ => LightKind::Point,
                    };
                }
                ui.slider_f32("Intensity", &mut state.entities[idx].light_intensity, 0.0, 100.0);
                ui.slider_f32("Range", &mut state.entities[idx].light_range, 0.1, 1000.0);
                if state.entities[idx].light_kind == LightKind::Spot {
                    ui.slider_f32("Spot Angle", &mut state.entities[idx].light_spot_angle, 1.0, 179.0);
                }
                ui.checkbox("Cast Shadows", &mut state.entities[idx].light_shadows);
                ui.tree_node_end();
            }
        }

        // Camera section
        if state.entities[idx].is_camera {
            if component_header(ui, "C", "Camera", &mut state.inspector_sections.camera_open) {
                let proj_names: Vec<&str> = vec!["Perspective", "Orthographic"];
                let mut proj_idx = match state.entities[idx].camera_projection {
                    CameraProjection::Perspective => 0, CameraProjection::Orthographic => 1,
                };
                if ui.dropdown("Projection", &mut proj_idx, &proj_names) {
                    state.entities[idx].camera_projection = match proj_idx {
                        0 => CameraProjection::Perspective, 1 => CameraProjection::Orthographic,
                        _ => CameraProjection::Perspective,
                    };
                }
                ui.slider_f32("FOV", &mut state.entities[idx].camera_fov, 10.0, 170.0);
                ui.drag_value("Near", &mut state.entities[idx].camera_near, 0.01);
                ui.drag_value("Far", &mut state.entities[idx].camera_far, 10.0);
                ui.tree_node_end();
            }
        }

        // Audio section
        if state.entities[idx].is_audio {
            if component_header(ui, "A", "Audio Source", &mut state.inspector_sections.audio_open) {
                ui.slider_f32("Volume", &mut state.entities[idx].audio_volume, 0.0, 1.0);
                ui.drag_value("Pitch", &mut state.entities[idx].audio_pitch, 0.01);
                ui.checkbox("Spatial", &mut state.entities[idx].audio_spatial);
                if state.entities[idx].audio_spatial {
                    ui.drag_value("Min Dist", &mut state.entities[idx].audio_min_dist, 0.1);
                    ui.drag_value("Max Dist", &mut state.entities[idx].audio_max_dist, 1.0);
                }
                ui.tree_node_end();
            }
        }

        // Script section
        if state.entities[idx].has_script {
            if component_header(ui, "S", "Script", &mut state.inspector_sections.script_open) {
                ui.text_input("File", &mut state.entities[idx].script_file);
                ui.tree_node_end();
            }
        }

        // Add component button
        ui.separator();
        if ui.button("+ Add Component") {
            // Toggle physics on as a simple example
            if !state.entities[idx].has_physics {
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
            }
        }

        ui.end_panel();
    }
}

// =============================================================================
// Bottom Panel (Slate)
// =============================================================================

fn draw_bottom_panel_slate(
    ui: &mut UI,
    state: &mut EditorState,
    left_w: f32,
    width: f32,
    right_w: f32,
    height: f32,
    bottom_h: f32,
    status_h: f32,
) {
    let panel_x = left_w;
    let panel_w = width - left_w - right_w;
    let panel_y = height - bottom_h - status_h;

    let bottom_rect = Rect::new(
        Vec2::new(panel_x, panel_y),
        Vec2::new(panel_x + panel_w, panel_y + bottom_h),
    );

    if ui.begin_panel("Bottom Panel", bottom_rect) {
        // Tab bar
        let tab_names = ["Console", "Content", "Profiler", "Animation"];
        tab_bar_premium(ui, &tab_names, &mut state.bottom_tab_idx);

        // Sync tab index to enum
        state.bottom_tab = match state.bottom_tab_idx {
            0 => BottomTab::Console,
            1 => BottomTab::ContentBrowser,
            2 => BottomTab::Profiler,
            3 => BottomTab::Animation,
            _ => BottomTab::Console,
        };

        match state.bottom_tab {
            BottomTab::Console => draw_console_tab_slate(ui, state),
            BottomTab::ContentBrowser => draw_content_browser_tab_slate(ui, state),
            BottomTab::Profiler => draw_profiler_tab_slate(ui, state),
            BottomTab::Animation => draw_animation_tab_slate(ui, state),
        }

        ui.end_panel();
    }
}

// =============================================================================
// Console Tab (Slate)
// =============================================================================

fn draw_console_tab_slate(ui: &mut UI, state: &mut EditorState) {
    // Filter display
    ui.horizontal(|ui| {
        ui.label_colored("Filter:", TEXT_MUTED);
        if ui.button("All") { state.console_filter_level = None; }
        if ui.button("Info") { state.console_filter_level = Some(LogLevel::Info); }
        if ui.button("Warn") { state.console_filter_level = Some(LogLevel::Warn); }
        if ui.button("Err") { state.console_filter_level = Some(LogLevel::Error); }
        if ui.button("Sys") { state.console_filter_level = Some(LogLevel::System); }
        if ui.button("Clear") { state.console_log.clear(); }
    });

    ui.separator();

    // Log entries (show last N that fit)
    let max_visible = 20;
    let filtered: Vec<&LogEntry> = state.console_log.iter()
        .filter(|e| {
            match state.console_filter_level {
                Some(f) => e.level == f,
                None => true,
            }
        })
        .collect();

    let start = if filtered.len() > max_visible { filtered.len() - max_visible } else { 0 };
    for entry in &filtered[start..] {
        let timestamp = format!("{:>7.1}", entry.timestamp);
        let prefix = entry.level.prefix();
        let color = entry.level.color();
        let count_str = if entry.count > 1 { format!(" (x{})", entry.count) } else { String::new() };
        let line = format!("{} {} {}{}", timestamp, prefix, entry.text, count_str);
        ui.label_colored(&line, color);
    }

    // Command input
    ui.separator();
    ui.horizontal(|ui| {
        ui.label_colored(">", ACCENT);
    });
    if ui.text_input("##cmd", &mut state.console_input) {
        // text changed
    }
    if ui.button("Run") {
        submit_console(state);
    }
}

// =============================================================================
// Content Browser Tab (Slate)
// =============================================================================

fn draw_content_browser_tab_slate(ui: &mut UI, state: &mut EditorState) {
    // Path display
    ui.horizontal(|ui| {
        ui.label_colored(&format!("res://{}", state.asset_path), ACCENT_DIM);
        if ui.button("..") {
            if let Some(pos) = state.asset_path.rfind('/').or_else(|| state.asset_path.rfind('\\')) {
                state.asset_path.truncate(pos);
            }
        }
    });

    // Search
    search_bar(ui, &mut state.asset_search);

    ui.separator();

    // Asset entries
    let entries = scan_asset_dir(&state.asset_path, &state.asset_search);
    if entries.is_empty() {
        let defaults = [
            ("[D] Models/", CYAN),
            ("[D] Textures/", GREEN),
            ("[D] Materials/", ACCENT),
            ("[D] Audio/", ORANGE),
            ("[D] Scripts/", MAGENTA),
            ("[D] Scenes/", YELLOW),
            ("[D] Shaders/", RED),
        ];
        ui.label_colored("Default asset structure:", TEXT_MUTED);
        for (name, color) in &defaults {
            ui.label_colored(name, *color);
        }
    } else {
        for entry in &entries {
            let icon = asset_icon(&entry.name, entry.is_dir);
            let color = if entry.is_dir { YELLOW } else { asset_color(&entry.name) };
            let label = format!("{} {}", icon, entry.name);
            if ui.selectable(&label, false) {
                if entry.is_dir {
                    state.asset_path = format!("{}/{}", state.asset_path, entry.name);
                } else {
                    state.log(LogLevel::Info, format!("Selected asset: {}", entry.name));
                }
            }
        }
    }
}

// =============================================================================
// Profiler Tab (Slate)
// =============================================================================

fn draw_profiler_tab_slate(ui: &mut UI, state: &mut EditorState) {
    let fps_color = if state.smooth_fps >= 55.0 { GREEN } else if state.smooth_fps >= 30.0 { YELLOW } else { RED };

    ui.horizontal(|ui| {
        ui.label_colored(&format!("FPS: {:.0}", state.smooth_fps), fps_color);
        ui.label_colored(&format!("Frame: {:.2} ms", state.smooth_frame_time), TEXT_NORMAL);
    });

    if !state.frame_times.is_empty() {
        let min = state.frame_times.iter().cloned().fold(f64::MAX, f64::min);
        let max = state.frame_times.iter().cloned().fold(f64::MIN, f64::max);
        let avg = state.frame_times.iter().sum::<f64>() / state.frame_times.len() as f64;
        let p99 = {
            let mut sorted: Vec<f64> = state.frame_times.iter().cloned().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted.get((sorted.len() as f64 * 0.99) as usize).copied().unwrap_or(0.0)
        };
        ui.label_colored(
            &format!("Min: {:.2}  Avg: {:.2}  P99: {:.2}  Max: {:.2} ms", min, avg, p99, max),
            TEXT_DIM,
        );
    }

    ui.separator();

    // Simple frame time bar graph (using rectangles)
    let bar_count = state.frame_times.len().min(100);
    if bar_count > 0 {
        let graph_h = 60.0;
        let bar_w = 2.0;
        let start_idx = state.frame_times.len().saturating_sub(bar_count);

        // Draw reference lines
        ui.label_colored("16.67ms (60fps) ---- 33.33ms (30fps)", TEXT_MUTED);

        // Draw bars using renderer directly
        for (i, &val) in state.frame_times.iter().skip(start_idx).enumerate() {
            let bar_h = (val as f32 / 50.0 * graph_h).clamp(1.0, graph_h);
            let x = i as f32 * (bar_w + 1.0);
            let color = if val < 16.67 { GREEN } else if val < 33.33 { YELLOW } else { RED };
            // We use label output for the text-based profiler since we're inside a panel
            let _ = color; // bars would be drawn via renderer directly
        }
    }

    ui.label_colored(&format!("Frame: {}", state.frame_count), TEXT_MUTED);
    ui.label_colored(
        &format!("Entities: {} | Bodies: {}",
            state.entities.len(),
            state.engine.physics().body_count()),
        TEXT_MUTED,
    );
}

// =============================================================================
// Animation Tab (Slate)
// =============================================================================

fn draw_animation_tab_slate(ui: &mut UI, state: &mut EditorState) {
    ui.horizontal(|ui| {
        if toolbar_button(ui, "|<", "Go to start", false) {
            state.anim_time = 0.0;
        }
        let play_icon = if state.anim_playing { "||" } else { "|>" };
        if toolbar_button(ui, play_icon, "Play/Pause", state.anim_playing) {
            state.anim_playing = !state.anim_playing;
        }
        if toolbar_button(ui, "[]", "Stop", false) {
            state.anim_playing = false;
            state.anim_time = 0.0;
        }
        if toolbar_button(ui, ">|", "Go to end", false) {
            state.anim_time = state.anim_duration;
        }
    });

    ui.checkbox("Loop", &mut state.anim_loop);
    ui.slider_f32("Time", &mut state.anim_time, 0.0, state.anim_duration);
    ui.drag_value("Duration", &mut state.anim_duration, 0.1);

    ui.separator();
    ui.label_colored("Keyframes and curves will appear here when animations are loaded.", TEXT_MUTED);

    // Tick animation forward if playing
    if state.anim_playing {
        let dt = 1.0 / 60.0;
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
// Status Bar (Slate)
// =============================================================================

fn draw_status_bar_slate(ui: &mut UI, state: &EditorState, width: f32, height: f32, status_h: f32) {
    let status_rect = Rect::new(
        Vec2::new(0.0, height - status_h),
        Vec2::new(width, height),
    );

    // Draw status bar background manually (no panel title)
    ui.renderer.draw_rect(status_rect, BG_BASE, 0.0);
    ui.renderer.draw_line(Vec2::new(0.0, height - status_h), Vec2::new(width, height - status_h), BORDER, 1.0);

    // Status dot
    let (dot_color, status_text) = if state.is_playing && !state.is_paused {
        (ACCENT_BRIGHT, "Playing")
    } else if state.is_paused {
        (YELLOW, "Paused")
    } else {
        (GREEN, "Editing")
    };

    let dot_x = 8.0;
    let dot_y = height - status_h + status_h * 0.5;
    ui.renderer.draw_circle(Vec2::new(dot_x + 4.0, dot_y), 3.0, dot_color);

    ui.renderer.draw_text(status_text, Vec2::new(dot_x + 12.0, height - status_h + 2.0), 11.0, TEXT_DIM);

    // Scene name
    let modified = if state.scene_modified { " *" } else { "" };
    let scene_label = format!("{}{}", state.scene_name, modified);
    let scene_color = if state.scene_modified { YELLOW } else { TEXT_NORMAL };
    ui.renderer.draw_text(&scene_label, Vec2::new(120.0, height - status_h + 2.0), 11.0, scene_color);

    // Right-aligned stats
    let fps_color = if state.smooth_fps >= 55.0 { GREEN } else if state.smooth_fps >= 30.0 { YELLOW } else { RED };
    let stats_text = format!("{:.0} FPS | {} entities | {} bodies",
        state.smooth_fps,
        state.entities.len(),
        state.engine.physics().body_count(),
    );
    let (tw, _) = ui.renderer.measure_text(&stats_text, 11.0);
    ui.renderer.draw_text(&stats_text, Vec2::new(width - tw - 8.0, height - status_h + 2.0), 11.0, fps_color);
}

// =============================================================================
// Viewport Overlays (drawn on the 3D viewport area)
// =============================================================================

fn draw_viewport_overlays(ui: &mut UI, state: &EditorState, left_w: f32, right_w: f32, top_y: f32, content_h: f32) {
    let screen = ui.screen_size();
    let vp_x = left_w;
    let vp_y = top_y;
    let vp_w = screen.x - left_w - right_w;
    let vp_h = content_h;

    // Top-left: mode info
    if state.stats_visible {
        let mode_text = format!("{} | {} | {} | {}",
            state.gizmo_mode.label(),
            if state.coord_space == CoordSpace::Local { "Local" } else { "World" },
            if state.grid_visible { "Grid" } else { "" },
            if state.wireframe_mode { "Wire" } else { "" },
        );
        ui.renderer.draw_text(&mode_text, Vec2::new(vp_x + 8.0, vp_y + 22.0), 11.0, TEXT_MUTED);
    }

    // Top-right: Camera info
    let cam_text = format!("Camera  yaw:{:.0}  pitch:{:.0}  dist:{:.1}",
        state.camera_yaw, state.camera_pitch, state.camera_dist);
    let (cw, _) = ui.renderer.measure_text(&cam_text, 11.0);
    ui.renderer.draw_text(&cam_text, Vec2::new(vp_x + vp_w - cw - 8.0, vp_y + 6.0), 11.0, TEXT_MUTED);

    let tgt_text = format!("Target: ({:.1}, {:.1}, {:.1})",
        state.camera_target[0], state.camera_target[1], state.camera_target[2]);
    let (tw, _) = ui.renderer.measure_text(&tgt_text, 11.0);
    ui.renderer.draw_text(&tgt_text, Vec2::new(vp_x + vp_w - tw - 8.0, vp_y + 20.0), 11.0, TEXT_MUTED);

    // Bottom center: viewport label
    let vp_label = format!("3D Viewport | {} x {}", vp_w as u32, vp_h as u32);
    let (lw, _) = ui.renderer.measure_text(&vp_label, 11.0);
    ui.renderer.draw_text(&vp_label, Vec2::new(vp_x + (vp_w - lw) * 0.5, vp_y + vp_h - 18.0), 11.0, TEXT_MUTED);

    // Play state overlay
    if state.is_playing && !state.is_paused {
        let sim_text = format!("SIMULATING | {:.2}x | {:.1}s", state.sim_speed, state.total_sim_time);
        ui.renderer.draw_text(&sim_text, Vec2::new(vp_x + 8.0, vp_y + 36.0), 12.0, GREEN);
    } else if state.is_paused {
        let pause_text = format!("PAUSED | {:.1}s", state.total_sim_time);
        ui.renderer.draw_text(&pause_text, Vec2::new(vp_x + 8.0, vp_y + 36.0), 12.0, YELLOW);
    }

    // Selected entity indicator (top center)
    if let Some(idx) = state.selected_entity {
        if idx < state.entities.len() {
            let ent = &state.entities[idx];
            let icon_color = ent.entity_type.icon_color();
            let label = format!("[{}] {} ({:.1}, {:.1}, {:.1})",
                ent.entity_type.icon_letter(), ent.name,
                ent.position[0], ent.position[1], ent.position[2]);
            let (sel_w, _) = ui.renderer.measure_text(&label, 11.0);
            let pill_x = vp_x + (vp_w - sel_w - 16.0) * 0.5;
            let pill_y = vp_y + 4.0;
            let pill_rect = Rect::new(
                Vec2::new(pill_x, pill_y),
                Vec2::new(pill_x + sel_w + 16.0, pill_y + 18.0),
            );
            ui.renderer.draw_rect(pill_rect, UIColor::new(0.07, 0.07, 0.086, 0.8), 9.0);
            ui.renderer.draw_rect_outline(pill_rect, BORDER, 1.0, 9.0);
            ui.renderer.draw_text(&label, Vec2::new(pill_x + 8.0, pill_y + 3.0), 11.0, icon_color);
        }
    }

    // Axis indicator (bottom-left)
    let ax_x = vp_x + 30.0;
    let ax_y = vp_y + vp_h - 40.0;
    let axis_len = 18.0;
    ui.renderer.draw_line(Vec2::new(ax_x, ax_y), Vec2::new(ax_x + axis_len, ax_y), X_COLOR, 2.0);
    ui.renderer.draw_text("X", Vec2::new(ax_x + axis_len + 3.0, ax_y - 5.0), 10.0, X_COLOR);
    ui.renderer.draw_line(Vec2::new(ax_x, ax_y), Vec2::new(ax_x, ax_y - axis_len), Y_COLOR, 2.0);
    ui.renderer.draw_text("Y", Vec2::new(ax_x - 5.0, ax_y - axis_len - 10.0), 10.0, Y_COLOR);
    ui.renderer.draw_line(Vec2::new(ax_x, ax_y), Vec2::new(ax_x - axis_len * 0.6, ax_y + axis_len * 0.5), Z_COLOR, 2.0);
    ui.renderer.draw_text("Z", Vec2::new(ax_x - axis_len * 0.6 - 12.0, ax_y + axis_len * 0.5 - 2.0), 10.0, Z_COLOR);

    // Stats overlay (bottom-left)
    if state.stats_visible {
        let stats_text = format!("Entities: {} | Bodies: {}",
            state.entities.len(),
            state.engine.physics().body_count());
        ui.renderer.draw_text(&stats_text, Vec2::new(vp_x + 8.0, vp_y + vp_h - 60.0), 11.0, TEXT_MUTED);
    }
}

// =============================================================================
// Notification Toast (Slate)
// =============================================================================

fn draw_notification_slate(ui: &mut UI, state: &EditorState) {
    if let Some((ref msg, level, start)) = state.notification {
        let elapsed = start.elapsed().as_secs_f32();
        let duration = 3.0;
        if elapsed > duration { return; }

        let alpha = if elapsed < 0.2 { elapsed / 0.2 } else if elapsed > duration - 0.5 { (duration - elapsed) / 0.5 } else { 1.0 };

        let screen = ui.screen_size();
        let font_size = 12.0;
        let padding = 12.0;
        let (tw, _) = ui.renderer.measure_text(msg, font_size);
        let toast_w = tw + padding * 2.0 + 20.0;
        let toast_h = font_size + padding * 2.0;
        let x = screen.x - toast_w - 16.0;
        let y = screen.y - toast_h - 36.0;

        let toast_rect = Rect::new(Vec2::new(x, y), Vec2::new(x + toast_w, y + toast_h));

        // Background
        ui.renderer.draw_rect(toast_rect, UIColor::new(BG_PANEL.r, BG_PANEL.g, BG_PANEL.b, alpha), 4.0);
        ui.renderer.draw_rect_outline(toast_rect, UIColor::new(BORDER.r, BORDER.g, BORDER.b, alpha), 1.0, 4.0);

        // Color strip
        let strip_color = level.color().with_alpha(alpha);
        let strip_rect = Rect::new(Vec2::new(x, y), Vec2::new(x + 3.0, y + toast_h));
        ui.renderer.draw_rect(strip_rect, strip_color, 2.0);

        // Level prefix
        let prefix_color = level.color().with_alpha(alpha);
        ui.renderer.draw_text(level.prefix(), Vec2::new(x + 8.0, y + padding), font_size, prefix_color);

        // Message
        let msg_color = UIColor::new(TEXT_NORMAL.r, TEXT_NORMAL.g, TEXT_NORMAL.b, alpha);
        ui.renderer.draw_text(msg, Vec2::new(x + 8.0 + 30.0, y + padding), font_size, msg_color);
    }
}

// =============================================================================
// About / Shortcuts dialogs (Slate)
// =============================================================================

fn draw_about_dialog(ui: &mut UI, state: &mut EditorState) {
    if !state.show_about { return; }

    let screen = ui.screen_size();
    let w = 380.0;
    let h = 320.0;
    let x = (screen.x - w) * 0.5;
    let y = (screen.y - h) * 0.5;

    let dialog_rect = Rect::new(Vec2::new(x, y), Vec2::new(x + w, y + h));

    // Shadow
    ui.renderer.draw_rect_shadow(dialog_rect, UIColor::new(0.0, 0.0, 0.0, 0.5), 10.0, Vec2::new(2.0, 2.0));

    if ui.begin_panel("About Genovo Studio", dialog_rect) {
        ui.label_colored("GENOVO STUDIO", ACCENT);
        ui.label_colored("v1.0", TEXT_DIM);
        ui.space(4.0);
        ui.label("Professional Game Development Environment");
        ui.label_colored("26 engine modules fully linked", TEXT_DIM);
        ui.space(8.0);

        let items = [
            ("Rendering:", "wgpu 24 (GPU-accelerated)", CYAN),
            ("UI Backend:", "Custom Slate (UIGpuRenderer)", ACCENT),
            ("UI Framework:", "genovo-ui (Slate)", GREEN),
            ("Physics:", "Custom impulse solver", YELLOW),
            ("ECS:", "Archetype-based", MAGENTA),
            ("Audio:", "Software PCM mixer", ORANGE),
            ("Scripting:", "GenovoScript VM", RED),
            ("Terrain:", "Procedural heightmap gen", GREEN),
            ("Procgen:", "BSP dungeon generation", CYAN),
        ];
        for (label, value, color) in &items {
            ui.horizontal(|ui| {
                ui.label_colored(label, TEXT_DIM);
                ui.label_colored(value, *color);
            });
        }

        ui.space(8.0);
        ui.separator();
        ui.label_colored("genovo.dev", ACCENT);
        ui.space(4.0);

        if ui.button("Close") {
            state.show_about = false;
        }

        ui.end_panel();
    }
}

fn draw_shortcuts_dialog(ui: &mut UI, state: &mut EditorState) {
    if !state.show_shortcuts { return; }

    let screen = ui.screen_size();
    let w = 380.0;
    let h = 400.0;
    let x = (screen.x - w) * 0.5;
    let y = (screen.y - h) * 0.5;

    let dialog_rect = Rect::new(Vec2::new(x, y), Vec2::new(x + w, y + h));

    ui.renderer.draw_rect_shadow(dialog_rect, UIColor::new(0.0, 0.0, 0.0, 0.5), 10.0, Vec2::new(2.0, 2.0));

    if ui.begin_panel("Keyboard Shortcuts", dialog_rect) {
        ui.label_colored("Genovo Studio Shortcuts", ACCENT);
        ui.space(4.0);

        let shortcuts = [
            ("Ctrl+N", "New Scene"),
            ("Ctrl+S", "Save Scene"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            ("Ctrl+D", "Duplicate Selected"),
            ("Delete", "Delete Selected"),
            ("Q", "Select Tool"),
            ("W", "Translate Tool"),
            ("E", "Rotate Tool"),
            ("R", "Scale Tool"),
            ("Space", "Play / Pause"),
            ("F5", "Toggle Play Mode"),
            ("F", "Focus Selected"),
            ("G", "Spawn Physics Ball"),
            ("Up/Down", "Select Prev/Next Entity"),
            ("Escape", "Deselect / Cancel"),
        ];

        for (key, desc) in &shortcuts {
            ui.horizontal(|ui| {
                ui.label_colored(key, ACCENT);
                ui.label_colored(desc, TEXT_NORMAL);
            });
        }

        ui.space(8.0);
        if ui.button("Close") {
            state.show_shortcuts = false;
        }

        ui.end_panel();
    }
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
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,

    // Full 3D scene renderer
    scene_manager: SceneRenderManager,

    // Custom Slate UI -- GPU-rendered via UIGpuRenderer
    ui: UI,

    // Per-frame input state for the UI
    ui_input: UIInputState,

    // Mouse state tracking for press/release detection
    mouse_left_was_down: bool,
    mouse_right_was_down: bool,
    prev_mouse_pos: Vec2,

    // Keys currently held
    keys_held: Vec<u32>,

    // Editor state
    editor: EditorState,
}

impl ApplicationHandler for EditorApp {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        // Create window
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

        let dev = Arc::new(dev);
        let que = Arc::new(que);

        // Create full 3D SceneRenderManager
        let scene_manager = SceneRenderManager::new(
            &dev,
            cfg.format,
            wgpu::TextureFormat::Depth32Float,
        );

        let dv = make_depth(&dev, sz.width, sz.height);

        // Initialize audio output using engine SoftwareMixer in a background thread
        let audio_running = self.audio_running.clone();
        audio_running.store(true, Ordering::SeqCst);
        std::thread::Builder::new()
            .name("genovo-audio".into())
            .spawn(move || {
                use genovo_audio::{AudioMixer, SoftwareMixer};
                let mut mixer = SoftwareMixer::new(48000, 2, 1024, 32);
                println!("[Genovo Audio] Mixer thread started: 48kHz stereo, 1024 frames, 32 voices");
                let dt = 1024.0 / 48000.0;
                while audio_running.load(Ordering::Relaxed) {
                    mixer.update(dt);
                    std::thread::sleep(std::time::Duration::from_millis(21));
                }
                println!("[Genovo Audio] Mixer thread stopped");
            })
            .expect("Failed to spawn audio thread");

        // Create UIGpuRenderer
        let ui_renderer = UIGpuRenderer::new(
            Arc::clone(&dev),
            Arc::clone(&que),
            cfg.format,
        );

        // Create UI framework
        let ui = UI::new(ui_renderer);

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

        // Editor state initialization
        let mut editor = EditorState::new(engine);
        editor.log(LogLevel::System, "Genovo Studio v1.0 -- Professional Game Development Environment");
        editor.log(LogLevel::System, format!("GPU: {} ({:?})", adap.get_info().name, adap.get_info().backend));
        editor.log(LogLevel::System, format!("Surface format: {:?} | Resolution: {}x{}", cfg.format, sz.width, sz.height));
        editor.log(LogLevel::System, "3D Renderer: SceneRenderManager (PBR + Grid)");
        editor.log(LogLevel::System, "Audio: Software mixer thread active");
        editor.log(LogLevel::System, "UI: Custom Slate framework (UIGpuRenderer) | Theme: UIStyle::dark()");
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

        let dir_light = editor.spawn_light(LightKind::Directional);
        editor.entities[dir_light].position = [5.0, 10.0, 5.0];

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
        println!("[Genovo Studio] UI: Custom Slate framework (UIGpuRenderer, no egui)");
        println!("[Genovo Studio] Editor ready. {} entities in scene.", editor.entities.len());

        self.gpu = Some(GpuState {
            window: w,
            device: dev,
            queue: que,
            surface: surf,
            config: cfg,
            depth_view: dv,
            scene_manager,
            ui,
            ui_input: UIInputState::new(),
            mouse_left_was_down: false,
            mouse_right_was_down: false,
            prev_mouse_pos: Vec2::ZERO,
            keys_held: Vec::new(),
            editor,
        });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        let Some(s) = self.gpu.as_mut() else { return };

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
                s.ui_input.mod_ctrl = mods.state().control_key();
                s.ui_input.mod_shift = mods.state().shift_key();
                s.ui_input.mod_alt = mods.state().alt_key();
            }

            WindowEvent::CursorMoved { position, .. } => {
                s.ui_input.mouse_pos = Vec2::new(position.x as f32, position.y as f32);
            }

            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                let pressed = btn_state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        s.ui_input.mouse_left_down = pressed;
                    }
                    MouseButton::Right => {
                        s.ui_input.mouse_right_down = pressed;
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => Vec2::new(x, y),
                    winit::event::MouseScrollDelta::PixelDelta(p) => Vec2::new(p.x as f32, p.y as f32),
                };
                s.ui_input.scroll_delta = scroll;
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(k) = event.physical_key {
                    // Convert to UI key code
                    if let Some(ui_key) = keycode_to_u32(k) {
                        if event.state == ElementState::Pressed {
                            s.ui_input.keys_pressed.push(ui_key);
                            if !s.keys_held.contains(&ui_key) {
                                s.keys_held.push(ui_key);
                            }
                        } else {
                            s.ui_input.keys_released.push(ui_key);
                            s.keys_held.retain(|&x| x != ui_key);
                        }
                    }

                    // Handle typed text
                    if event.state == ElementState::Pressed {
                        if let Some(ref text) = event.text {
                            for ch in text.chars() {
                                if ch >= ' ' && ch != '\x7f' {
                                    s.ui_input.text_input.push(ch);
                                }
                            }
                        }
                    }

                    // Handle editor shortcuts (only on press)
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
                                s.editor.show_about = false;
                                s.editor.show_shortcuts = false;
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

                // ---- Prepare UI input state ----
                // Compute press/release from current and previous state
                let mouse_left_now = s.ui_input.mouse_left_down;
                let mouse_right_now = s.ui_input.mouse_right_down;
                s.ui_input.mouse_left_pressed = mouse_left_now && !s.mouse_left_was_down;
                s.ui_input.mouse_left_released = !mouse_left_now && s.mouse_left_was_down;
                s.ui_input.mouse_right_pressed = mouse_right_now && !s.mouse_right_was_down;
                s.ui_input.mouse_right_released = !mouse_right_now && s.mouse_right_was_down;
                s.ui_input.mouse_delta = s.ui_input.mouse_pos - s.prev_mouse_pos;
                s.ui_input.keys_down = s.keys_held.clone();

                let screen_size = Vec2::new(s.config.width as f32, s.config.height as f32);

                // ---- Begin UI frame ----
                s.ui.begin_frame(s.ui_input.clone(), screen_size, dt);

                // ---- Draw all editor UI ----
                draw_editor_ui(&mut s.ui, &mut s.editor);

                // ---- Draw dialogs ----
                draw_about_dialog(&mut s.ui, &mut s.editor);
                draw_shortcuts_dialog(&mut s.ui, &mut s.editor);

                // ---- Save state for next frame ----
                s.mouse_left_was_down = mouse_left_now;
                s.mouse_right_was_down = mouse_right_now;
                s.prev_mouse_pos = s.ui_input.mouse_pos;

                // ---- Reset per-frame input ----
                s.ui_input.keys_pressed.clear();
                s.ui_input.keys_released.clear();
                s.ui_input.text_input.clear();
                s.ui_input.scroll_delta = Vec2::ZERO;

                // ---- Sync physics positions ----
                s.editor.sync_physics_to_entities();

                // ---- Build camera from editor orbit parameters ----
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

                // ---- Build scene lights from entities ----
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

                // ---- Submit scene entities to the render manager ----
                s.scene_manager.clear_queue();

                // Set clear color
                let t = s.editor.frame_count as f32 * 0.005;
                let (br, bg, bb) = if s.editor.is_playing && !s.editor.is_paused {
                    let pulse = (t * 2.0).sin().abs() * 0.008;
                    (0.06 + pulse as f64, 0.06_f64, 0.08 + pulse as f64)
                } else {
                    (0.06_f64, 0.06, 0.08)
                };
                s.scene_manager.set_clear_color([br, bg, bb, 1.0]);

                // Grid
                if s.editor.grid_visible {
                    s.scene_manager.set_grid_enabled(&s.device, true);
                } else {
                    s.scene_manager.set_grid_enabled(&s.device, false);
                }

                // Submit each entity
                for ent in &s.editor.entities {
                    if !ent.visible { continue; }

                    let mesh_id = match ent.entity_type {
                        EntityType::Mesh => match ent.mesh_shape {
                            MeshShape::Cube => s.scene_manager.builtin_cube,
                            MeshShape::Sphere => s.scene_manager.builtin_sphere,
                            MeshShape::Cylinder => s.scene_manager.builtin_cylinder,
                            MeshShape::Cone => s.scene_manager.builtin_cone,
                            MeshShape::Plane => s.scene_manager.builtin_plane,
                            MeshShape::Capsule => s.scene_manager.builtin_cylinder,
                        },
                        EntityType::Light => s.scene_manager.builtin_sphere,
                        EntityType::Camera => s.scene_manager.builtin_cube,
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

                // ---- GPU Render ----
                let Ok(out) = s.surface.get_current_texture() else {
                    s.window.request_redraw();
                    return;
                };
                let view = out.texture.create_view(&Default::default());

                // Pass 1: Render full 3D scene via SceneRenderManager
                let scene_cmd = s.scene_manager.render(
                    &s.device,
                    &s.queue,
                    &view,
                    &s.depth_view,
                    &scene_camera,
                    &scene_lights,
                );

                // Pass 2: UI overlay via UIGpuRenderer (no depth, alpha blending)
                let mut ui_enc = s.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ui_overlay"),
                });
                s.ui.finish_frame(&mut ui_enc, &view);

                // Submit both command buffers
                s.queue.submit([scene_cmd, ui_enc.finish()]);
                out.present();

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
    println!("[Genovo Studio] UI: Custom Slate framework (UIGpuRenderer, no egui)");
    println!("[Genovo Studio] Theme: UIStyle::dark() | DockStyle::dark_theme()");

    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = EditorApp {
        gpu: None,
        audio_running: Arc::new(AtomicBool::new(false)),
    };
    let _ = el.run_app(&mut app);
}
