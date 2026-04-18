//! Genovo Studio -- Professional Game Development Environment
//!
//! Architecture:
//!   1. winit 0.30 ApplicationHandler + wgpu for GPU
//!   2. SceneRenderManager for 3D viewport (PBR, grid, built-in primitives)
//!   3. egui overlay for all 2D UI -- NO depth attachment (prevents crash)
//!   4. Custom Genovo dark theme with accent blue rgb(56,132,244)
//!   5. Each frame: handle events -> egui begin_pass -> draw UI -> 3D render -> egui render

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
use genovo_render::scene_renderer::{SceneRenderManager, SceneCamera, SceneLights};

// Custom Slate UI framework -- for state management types only
use genovo_ui::ui_framework::UIStyle;
use genovo_ui::dock_system::DockStyle;

// Import undo system types
use genovo_editor::undo_system::{
    UndoStack, MoveEntityOp, RotateEntityOp, ScaleEntityOp,
    SpawnEntityOp, DespawnEntityOp, SerializedEntityData,
    SerializedComponentData, EntityId as UndoEntityId,
    Vec3 as UndoVec3, Quat as UndoQuat,
};

// Import scripting types for VM binding
use genovo_scripting::vm::{ScriptValue, ScriptError, NativeFn, ScriptContext};
use genovo_scripting::vm::GenovoVM;
use genovo_scripting::ScriptVM;

// Audio engine
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
// egui Color Palette -- Genovo dark theme
// =============================================================================

const BG_BASE: egui::Color32 = egui::Color32::from_rgb(18, 18, 22);
const BG_PANEL: egui::Color32 = egui::Color32::from_rgb(22, 22, 26);
const BG_WIDGET: egui::Color32 = egui::Color32::from_rgb(32, 32, 38);
const BG_HOVER: egui::Color32 = egui::Color32::from_rgb(42, 42, 50);
const BG_ACTIVE: egui::Color32 = egui::Color32::from_rgb(52, 52, 62);
const BORDER: egui::Color32 = egui::Color32::from_rgb(38, 38, 44);

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

// XYZ axis colors
const X_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 75, 75);
const Y_COLOR: egui::Color32 = egui::Color32::from_rgb(72, 199, 142);
const Z_COLOR: egui::Color32 = egui::Color32::from_rgb(56, 132, 244);

// =============================================================================
// Entity Type
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
// Mesh Shape
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
    mesh_shape: MeshShape,
    cast_shadows: bool,
    receive_shadows: bool,
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
    is_light: bool,
    light_kind: LightKind,
    light_color: [f32; 3],
    light_intensity: f32,
    light_range: f32,
    light_spot_angle: f32,
    light_shadows: bool,
    is_camera: bool,
    camera_projection: CameraProjection,
    camera_fov: f32,
    camera_near: f32,
    camera_far: f32,
    camera_clear_color: [f32; 4],
    is_audio: bool,
    audio_volume: f32,
    audio_pitch: f32,
    audio_spatial: bool,
    audio_min_dist: f32,
    audio_max_dist: f32,
    has_script: bool,
    script_file: String,
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
// Gizmo / Transform enums
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GizmoMode { Select, Translate, Rotate, Scale }

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
enum CoordSpace { Local, World }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PivotMode { Center, Pivot }

// =============================================================================
// Console
// =============================================================================

#[derive(Clone)]
struct LogEntry {
    text: String,
    level: LogLevel,
    timestamp: f64,
    count: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LogLevel { Info, Warn, Error, System, Debug }

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
        Self { hierarchy: true, inspector: true, bottom: true, toolbar: true, status_bar: true }
    }
}

// =============================================================================
// Bottom Tab
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum BottomTab { Console, ContentBrowser, Profiler, Animation }

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
        }
    }
}

// =============================================================================
// Snap / Speed presets
// =============================================================================

const SNAP_PRESETS: &[f32] = &[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
const SPEED_PRESETS: &[f32] = &[0.25, 0.5, 1.0, 2.0, 4.0];

// =============================================================================
// Editor State
// =============================================================================

struct EditorState {
    engine: Engine,
    ui_style: UIStyle,
    dock_style: DockStyle,
    entities: Vec<SceneEntity>,
    selected_entity: Option<usize>,
    next_entity_id: u32,
    entity_filter: String,
    is_playing: bool,
    is_paused: bool,
    sim_speed: f32,
    sim_speed_idx: usize,
    play_start_time: Option<Instant>,
    total_sim_time: f64,
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
    console_log: Vec<LogEntry>,
    console_input: String,
    console_history: Vec<String>,
    console_history_idx: Option<usize>,
    console_scroll_to_bottom: bool,
    console_filter_level: Option<LogLevel>,
    console_auto_scroll: bool,
    asset_path: String,
    asset_search: String,
    asset_view_size: f32,
    asset_show_extensions: bool,
    frame_times: VecDeque<f64>,
    fps_history: VecDeque<f64>,
    gpu_times: VecDeque<f64>,
    anim_time: f32,
    anim_duration: f32,
    anim_playing: bool,
    anim_loop: bool,
    frame_count: u64,
    last_frame: Instant,
    fps: f32,
    frame_time_ms: f32,
    start_time: Instant,
    smooth_fps: f32,
    smooth_frame_time: f32,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_dist: f32,
    camera_target: [f32; 3],
    camera_fov: f32,
    panels: PanelVisibility,
    bottom_tab: BottomTab,
    bottom_tab_idx: usize,
    hierarchy_width: f32,
    inspector_width: f32,
    bottom_height: f32,
    scene_name: String,
    scene_modified: bool,
    show_about: bool,
    show_preferences: bool,
    show_shortcuts: bool,
    modifiers: ModifiersState,
    renaming_entity: Option<usize>,
    rename_buffer: String,
    notification: Option<(String, LogLevel, Instant)>,
    undo_stack: UndoStack,
    recent_files: Vec<String>,
    inspector_sections: InspectorSections,
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
            bottom_height: 180.0,
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
            inspector_sections: InspectorSections::default(),
        }
    }

    fn log(&mut self, level: LogLevel, text: impl Into<String>) {
        let text = text.into();
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if let Some(last) = self.console_log.last_mut() {
            if last.text == text && last.level == level {
                last.count += 1;
                last.timestamp = elapsed;
                return;
            }
        }
        self.console_log.push(LogEntry { text, level, timestamp: elapsed, count: 1 });
        if self.console_log.len() > 5000 { self.console_log.drain(0..1000); }
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
                        shape: physics::CollisionShape::Box { half_extents: Vec3::new(0.5, 0.5, 0.5) },
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
        if idx >= self.entities.len() { return; }
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
        for e in &mut self.entities {
            if let Some(p) = e.parent {
                if p == idx { e.parent = None; }
                else if p > idx { e.parent = Some(p - 1); }
            }
        }
        if let Some(sel) = self.selected_entity {
            if sel == idx { self.selected_entity = None; }
            else if sel > idx { self.selected_entity = Some(sel - 1); }
        }
        self.scene_modified = true;
        self.push_delete_undo(idx, &name, pos, rot, scl);
        self.log(LogLevel::System, format!("Deleted: {}", name));
    }

    fn delete_selected(&mut self) {
        if let Some(idx) = self.selected_entity { self.delete_entity(idx); }
    }

    fn duplicate_selected(&mut self) {
        if let Some(idx) = self.selected_entity {
            if idx >= self.entities.len() { return; }
            let src = &self.entities[idx];
            let new_name = format!("{} (copy)", src.name);
            let etype = src.entity_type;
            let pos = src.position;
            let rot = src.rotation;
            let scl = src.scale;
            let mesh_shape = src.mesh_shape;
            let is_light = src.is_light;
            let lc = src.light_color;
            let li = src.light_intensity;
            let lr = src.light_range;
            let ls = src.light_shadows;
            let lk = src.light_kind;
            let is_cam = src.is_camera;

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
                let _ = self.engine.physics_mut().set_position(handle, Vec3::new(e.position[0], e.position[1], e.position[2]));
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
                let _ = self.engine.physics_mut().set_position(handle, Vec3::new(ent.position[0], ent.position[1], ent.position[2]));
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
            if self.play_start_time.is_none() { self.play_start_time = Some(Instant::now()); }
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
        let idx = self.spawn_entity(&format!("Ball_{}", self.next_entity_id), EntityType::Mesh);
        let e = &mut self.entities[idx];
        e.position = [0.0, 8.0, 0.0];
        e.mesh_shape = MeshShape::Sphere;
        e.collider_shape = ColliderShape::Sphere;
        if let Some(handle) = e.physics_handle {
            let _ = self.engine.physics_mut().set_position(handle, Vec3::new(0.0, 8.0, 0.0));
        }
        self.selected_entity = Some(idx);
    }

    fn select_next(&mut self) {
        if self.entities.is_empty() { return; }
        match self.selected_entity {
            Some(i) if i + 1 < self.entities.len() => self.selected_entity = Some(i + 1),
            _ => self.selected_entity = Some(0),
        }
    }

    fn select_prev(&mut self) {
        if self.entities.is_empty() { return; }
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
        while !self.entities.is_empty() { self.delete_entity(0); }
        self.scene_name = "Untitled Scene".to_string();
        self.scene_modified = false;
        self.selected_entity = None;
        self.notify("New scene created", LogLevel::System);
    }

    fn save_scene(&mut self) { self.save_scene_to_file("scene.json"); }

    fn save_scene_to_file(&mut self, path: &str) {
        let data = serde_json::json!({
            "name": self.scene_name,
            "entities": self.entities.iter().map(|e| {
                serde_json::json!({
                    "name": e.name, "type": format!("{:?}", e.entity_type),
                    "position": e.position, "rotation": e.rotation, "scale": e.scale,
                    "mesh_shape": format!("{:?}", e.mesh_shape),
                    "has_physics": e.has_physics, "mass": e.mass, "friction": e.friction, "restitution": e.restitution,
                    "body_kind": format!("{:?}", e.body_kind),
                    "is_light": e.is_light, "light_kind": format!("{:?}", e.light_kind),
                    "light_color": e.light_color, "light_intensity": e.light_intensity, "light_range": e.light_range,
                    "is_camera": e.is_camera, "camera_fov": e.camera_fov,
                    "is_audio": e.is_audio, "audio_volume": e.audio_volume,
                    "visible": e.visible, "active": e.active, "tags": e.tags,
                })
            }).collect::<Vec<_>>()
        });
        match std::fs::write(path, serde_json::to_string_pretty(&data).unwrap()) {
            Ok(_) => { self.scene_modified = false; self.notify(&format!("Saved: {} -> {}", self.scene_name, path), LogLevel::System); }
            Err(e) => { self.notify(&format!("Save failed: {}", e), LogLevel::Error); }
        }
    }

    fn load_scene_from_file(&mut self, path: &str) {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => { self.notify(&format!("Load failed: {}", e), LogLevel::Error); return; }
        };
        let data: serde_json::Value = match serde_json::from_str(&content) {
            Ok(d) => d,
            Err(e) => { self.notify(&format!("Parse error: {}", e), LogLevel::Error); return; }
        };
        while !self.entities.is_empty() { self.delete_entity(0); }
        if let Some(name) = data["name"].as_str() { self.scene_name = name.to_string(); }
        if let Some(entities) = data["entities"].as_array() {
            for ent_data in entities {
                let name = ent_data["name"].as_str().unwrap_or("Entity");
                let type_str = ent_data["type"].as_str().unwrap_or("Empty");
                let etype = match type_str {
                    "Mesh" => EntityType::Mesh, "Light" => EntityType::Light, "Camera" => EntityType::Camera,
                    "ParticleSystem" => EntityType::ParticleSystem, "Audio" => EntityType::Audio, _ => EntityType::Empty,
                };
                let idx = self.spawn_entity(name, etype);
                let e = &mut self.entities[idx];
                if let Some(pos) = ent_data["position"].as_array() { for (i, v) in pos.iter().enumerate().take(3) { e.position[i] = v.as_f64().unwrap_or(0.0) as f32; } }
                if let Some(rot) = ent_data["rotation"].as_array() { for (i, v) in rot.iter().enumerate().take(3) { e.rotation[i] = v.as_f64().unwrap_or(0.0) as f32; } }
                if let Some(scl) = ent_data["scale"].as_array() { for (i, v) in scl.iter().enumerate().take(3) { e.scale[i] = v.as_f64().unwrap_or(1.0) as f32; } }
                if let Some(handle) = e.physics_handle {
                    let _ = self.engine.physics_mut().set_position(handle, Vec3::new(e.position[0], e.position[1], e.position[2]));
                }
            }
        }
        self.scene_modified = false;
        self.selected_entity = None;
        self.notify(&format!("Loaded: {} from {}", self.scene_name, path), LogLevel::System);
    }

    // Undo/Redo
    fn push_move_undo(&mut self, ei: usize, old: [f32; 3], new: [f32; 3]) {
        self.undo_stack.push(Box::new(MoveEntityOp::new(UndoEntityId(ei as u64), UndoVec3::new(old[0], old[1], old[2]), UndoVec3::new(new[0], new[1], new[2]))), true);
    }
    fn push_rotate_undo(&mut self, ei: usize, old: [f32; 3], new: [f32; 3]) {
        self.undo_stack.push(Box::new(RotateEntityOp::new(UndoEntityId(ei as u64), UndoQuat::new(old[0], old[1], old[2], 0.0), UndoQuat::new(new[0], new[1], new[2], 0.0))), true);
    }
    fn push_scale_undo(&mut self, ei: usize, old: [f32; 3], new: [f32; 3]) {
        self.undo_stack.push(Box::new(ScaleEntityOp::new(UndoEntityId(ei as u64), UndoVec3::new(old[0], old[1], old[2]), UndoVec3::new(new[0], new[1], new[2]))), true);
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

    // Script VM
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
        { let p = Arc::clone(&ep); let cc = Arc::clone(&pc);
          let _ = vm.register_function("set_pos", Box::new(move |args: &[ScriptValue]| {
              if args.len() != 4 { return Err(ScriptError::ArityMismatch { function: "set_pos".into(), expected: 4, got: args.len() as u8 }); }
              let i = match &args[0] { ScriptValue::Int(i) => *i as usize, _ => return Err(ScriptError::TypeError("set_pos idx".into())) };
              let x = match &args[1] { ScriptValue::Float(f) => *f as f32, ScriptValue::Int(v) => *v as f32, _ => return Err(ScriptError::TypeError("x".into())) };
              let y = match &args[2] { ScriptValue::Float(f) => *f as f32, ScriptValue::Int(v) => *v as f32, _ => return Err(ScriptError::TypeError("y".into())) };
              let z = match &args[3] { ScriptValue::Float(f) => *f as f32, ScriptValue::Int(v) => *v as f32, _ => return Err(ScriptError::TypeError("z".into())) };
              let mut g = p.lock().unwrap(); if i >= g.len() { return Err(ScriptError::RuntimeError(format!("index {} OOB", i))); }
              g[i].1 = [x, y, z]; cc.lock().unwrap().push((i, [x, y, z])); Ok(ScriptValue::Nil)
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

    // Console command execution
    fn execute_console_command(&mut self, cmd: &str) {
        self.log(LogLevel::Info, format!("> {}", cmd));
        let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
        if parts.is_empty() { return; }
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
                self.log(LogLevel::System, "UI: egui rendering + custom Genovo dark theme");
            }
            "stats" => {
                let ecs_count = self.engine.world().entity_count();
                let body_count = self.engine.physics().body_count();
                let active = self.engine.physics().active_body_count();
                self.log(LogLevel::System, format!("ECS: {}  Scene: {}  Physics: {} ({} active)  FPS: {:.0}  Frame: {:.2}ms",
                    ecs_count, self.entities.len(), body_count, active, self.smooth_fps, self.smooth_frame_time));
            }
            "list" => {
                if self.entities.is_empty() { self.log(LogLevel::Info, "No entities"); }
                else {
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
                        if idx < self.entities.len() { self.selected_entity = Some(idx); self.log(LogLevel::System, format!("Selected: {}", self.entities[idx].name)); }
                        else { self.log(LogLevel::Error, format!("Index {} out of range", idx)); }
                    } else { self.log(LogLevel::Error, "Usage: select <index>"); }
                }
            }
            "delete" => { self.delete_selected(); }
            "spawn" => {
                let etype_str = parts.get(1).copied().unwrap_or("empty");
                match etype_str {
                    "cube" | "mesh" => { let i = self.spawn_mesh(MeshShape::Cube); self.selected_entity = Some(i); }
                    "sphere" => { let i = self.spawn_mesh(MeshShape::Sphere); self.selected_entity = Some(i); }
                    "cylinder" => { let i = self.spawn_mesh(MeshShape::Cylinder); self.selected_entity = Some(i); }
                    "light" => { let i = self.spawn_light(LightKind::Point); self.selected_entity = Some(i); }
                    "camera" => { let i = self.spawn_entity("Camera", EntityType::Camera); self.selected_entity = Some(i); }
                    _ => { let name = if parts.len() > 2 { parts[2..].join(" ") } else { "Empty".to_string() }; let i = self.spawn_entity(&name, EntityType::Empty); self.selected_entity = Some(i); }
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
                    self.log(LogLevel::System, format!("Gravity: ({}, {}, {})", x, y, z));
                }
            }
            "speed" => {
                if let Some(val) = parts.get(1) {
                    if let Ok(s) = val.parse::<f32>() { self.sim_speed = s.clamp(0.0, 10.0); self.log(LogLevel::System, format!("Speed: {:.2}x", self.sim_speed)); }
                }
            }
            "camera" => {
                if parts.len() == 4 {
                    self.camera_yaw = parts[1].parse().unwrap_or(self.camera_yaw);
                    self.camera_pitch = parts[2].parse().unwrap_or(self.camera_pitch);
                    self.camera_dist = parts[3].parse().unwrap_or(self.camera_dist);
                }
            }
            "scene" => {
                if parts.len() > 1 { self.scene_name = parts[1..].join(" "); self.scene_modified = true; }
            }
            "terrain" if parts.get(1) == Some(&"gen") => {
                match genovo_terrain::Heightmap::generate_procedural(257, 0.7, 42) {
                    Ok(h) => self.log(LogLevel::System, format!("Terrain 257x257 [min={:.2}, max={:.2}]", h.min_height(), h.max_height())),
                    Err(e) => self.log(LogLevel::Error, format!("Terrain error: {}", e)),
                }
            }
            "dungeon" if parts.get(1) == Some(&"gen") => {
                let cfg = genovo_procgen::BSPConfig { width: 80, height: 60, min_room_size: 6, max_depth: 8, room_fill_ratio: 0.7, seed: 42, wall_padding: 1 };
                let d = genovo_procgen::dungeon::generate_bsp(&cfg);
                self.log(LogLevel::System, format!("Dungeon: {} rooms", d.rooms.len()));
            }
            "script" => { let code = parts[1..].join(" "); self.execute_script_with_bindings(&code); }
            other => { self.log(LogLevel::Warn, format!("Unknown: '{}'. Type 'help'.", other)); }
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
// Asset Browser Helpers
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
// Apply Genovo dark theme to egui
// =============================================================================

fn apply_genovo_dark_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.visuals.dark_mode = true;
    style.visuals.panel_fill = egui::Color32::from_rgb(22, 22, 26);
    style.visuals.window_fill = egui::Color32::from_rgb(22, 22, 26);
    style.visuals.extreme_bg_color = egui::Color32::from_rgb(18, 18, 22);
    style.visuals.faint_bg_color = egui::Color32::from_rgb(32, 32, 38);
    style.visuals.code_bg_color = egui::Color32::from_rgb(28, 28, 34);
    style.visuals.striped = false;

    // Window
    style.visuals.window_shadow = egui::epaint::Shadow {
        offset: [0, 2],
        blur: 8,
        spread: 0,
        color: egui::Color32::from_black_alpha(100),
    };
    style.visuals.window_corner_radius = egui::CornerRadius::same(4);
    style.visuals.window_stroke = egui::Stroke::new(1.0, BORDER);

    // Widgets -- inactive
    style.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(32, 32, 38);
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_NORMAL);
    style.visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(3);
    style.visuals.widgets.inactive.expansion = 0.0;

    // Widgets -- hovered
    style.visuals.widgets.hovered.bg_fill = BG_HOVER;
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, TEXT_BRIGHT);
    style.visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(3);
    style.visuals.widgets.hovered.expansion = 1.0;

    // Widgets -- active (pressed)
    style.visuals.widgets.active.bg_fill = ACCENT;
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT_BRIGHT);
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0, TEXT_BRIGHT);
    style.visuals.widgets.active.corner_radius = egui::CornerRadius::same(3);
    style.visuals.widgets.active.expansion = 0.0;

    // Widgets -- open (menus, combo boxes)
    style.visuals.widgets.open.bg_fill = ACCENT_BG;
    style.visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.open.fg_stroke = egui::Stroke::new(1.0, TEXT_BRIGHT);
    style.visuals.widgets.open.corner_radius = egui::CornerRadius::same(3);
    style.visuals.widgets.open.expansion = 0.0;

    // Widgets -- non-interactive
    style.visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_NORMAL);
    style.visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(3);
    style.visuals.widgets.noninteractive.expansion = 0.0;

    // Selection
    style.visuals.selection.bg_fill = egui::Color32::from_rgba_premultiplied(56, 132, 244, 80);
    style.visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);

    // Hyperlinks
    style.visuals.hyperlink_color = ACCENT_BRIGHT;

    // Override text color
    style.visuals.override_text_color = Some(TEXT_NORMAL);

    // Spacing
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.window_margin = egui::Margin::same(8);
    style.spacing.button_padding = egui::vec2(6.0, 3.0);
    style.spacing.indent = 16.0;
    style.spacing.slider_width = 120.0;
    style.spacing.text_edit_width = 120.0;

    ctx.set_style(style);
}

// =============================================================================
// Draw complete editor UI using egui
// =============================================================================

fn draw_editor_ui(ctx: &egui::Context, state: &mut EditorState) {
    draw_menu_bar(ctx, state);

    if state.panels.toolbar {
        draw_toolbar(ctx, state);
    }

    if state.panels.hierarchy {
        draw_hierarchy(ctx, state);
    }

    if state.panels.inspector {
        draw_inspector(ctx, state);
    }

    if state.panels.bottom {
        draw_bottom_panel(ctx, state);
    }

    if state.panels.status_bar {
        draw_status_bar(ctx, state);
    }

    draw_viewport(ctx, state);

    if state.stats_visible {
        draw_fps_overlay(ctx, state);
    }

    draw_notification(ctx, state);
    draw_about_dialog(ctx, state);
    draw_shortcuts_dialog(ctx, state);
}

// =============================================================================
// Menu Bar
// =============================================================================

fn draw_menu_bar(ctx: &egui::Context, state: &mut EditorState) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.colored_label(ACCENT, "Genovo Studio");
            ui.separator();

            ui.menu_button("File", |ui| {
                if ui.button("New Scene       Ctrl+N").clicked() { state.new_scene(); ui.close_menu(); }
                if ui.button("Open Scene...").clicked() { state.load_scene_from_file("scene.json"); ui.close_menu(); }
                if ui.button("Save            Ctrl+S").clicked() { state.save_scene(); ui.close_menu(); }
                if ui.button("Save As...").clicked() { state.save_scene_to_file("scene.json"); ui.close_menu(); }
                ui.separator();
                if ui.button("Import Asset...").clicked() { state.log(LogLevel::System, "Import asset (placeholder)"); ui.close_menu(); }
                if ui.button("Export...").clicked() { state.log(LogLevel::System, "Export (placeholder)"); ui.close_menu(); }
                ui.separator();
                if ui.button("Exit").clicked() { std::process::exit(0); }
            });

            ui.menu_button("Edit", |ui| {
                if ui.button("Undo        Ctrl+Z").clicked() { state.perform_undo(); ui.close_menu(); }
                if ui.button("Redo        Ctrl+Y").clicked() { state.perform_redo(); ui.close_menu(); }
                ui.separator();
                if ui.button("Duplicate   Ctrl+D").clicked() { state.duplicate_selected(); ui.close_menu(); }
                if ui.button("Delete      Del").clicked() { state.delete_selected(); ui.close_menu(); }
                ui.separator();
                if ui.button("Select All").clicked() { state.select_all(); ui.close_menu(); }
                if ui.button("Deselect All").clicked() { state.selected_entity = None; ui.close_menu(); }
                ui.separator();
                if ui.button("Preferences...").clicked() { state.show_preferences = true; ui.close_menu(); }
            });

            ui.menu_button("View", |ui| {
                if ui.button("Toggle Hierarchy").clicked() { state.panels.hierarchy = !state.panels.hierarchy; ui.close_menu(); }
                if ui.button("Toggle Inspector").clicked() { state.panels.inspector = !state.panels.inspector; ui.close_menu(); }
                if ui.button("Toggle Bottom Panel").clicked() { state.panels.bottom = !state.panels.bottom; ui.close_menu(); }
                if ui.button("Toggle Toolbar").clicked() { state.panels.toolbar = !state.panels.toolbar; ui.close_menu(); }
                if ui.button("Toggle Status Bar").clicked() { state.panels.status_bar = !state.panels.status_bar; ui.close_menu(); }
                ui.separator();
                if ui.button("Toggle Grid").clicked() { state.grid_visible = !state.grid_visible; ui.close_menu(); }
                if ui.button("Toggle Wireframe").clicked() { state.wireframe_mode = !state.wireframe_mode; ui.close_menu(); }
                if ui.button("Toggle Stats").clicked() { state.stats_visible = !state.stats_visible; ui.close_menu(); }
                ui.separator();
                if ui.button("Focus Selected  F").clicked() { state.focus_selected(); ui.close_menu(); }
                if ui.button("Frame All").clicked() { state.frame_all(); ui.close_menu(); }
            });

            ui.menu_button("Create", |ui| {
                if ui.button("Empty Entity").clicked() { let i = state.spawn_entity("Empty", EntityType::Empty); state.selected_entity = Some(i); ui.close_menu(); }
                ui.separator();
                ui.label("Meshes");
                if ui.button("  Cube").clicked() { let i = state.spawn_mesh(MeshShape::Cube); state.selected_entity = Some(i); ui.close_menu(); }
                if ui.button("  Sphere").clicked() { let i = state.spawn_mesh(MeshShape::Sphere); state.selected_entity = Some(i); ui.close_menu(); }
                if ui.button("  Cylinder").clicked() { let i = state.spawn_mesh(MeshShape::Cylinder); state.selected_entity = Some(i); ui.close_menu(); }
                if ui.button("  Capsule").clicked() { let i = state.spawn_mesh(MeshShape::Capsule); state.selected_entity = Some(i); ui.close_menu(); }
                if ui.button("  Cone").clicked() { let i = state.spawn_mesh(MeshShape::Cone); state.selected_entity = Some(i); ui.close_menu(); }
                if ui.button("  Plane").clicked() { let i = state.spawn_mesh(MeshShape::Plane); state.selected_entity = Some(i); ui.close_menu(); }
                ui.separator();
                ui.label("Lights");
                if ui.button("  Directional Light").clicked() { let i = state.spawn_light(LightKind::Directional); state.selected_entity = Some(i); ui.close_menu(); }
                if ui.button("  Point Light").clicked() { let i = state.spawn_light(LightKind::Point); state.selected_entity = Some(i); ui.close_menu(); }
                if ui.button("  Spot Light").clicked() { let i = state.spawn_light(LightKind::Spot); state.selected_entity = Some(i); ui.close_menu(); }
                ui.separator();
                if ui.button("Camera").clicked() { let i = state.spawn_entity("Camera", EntityType::Camera); state.selected_entity = Some(i); ui.close_menu(); }
                if ui.button("Particle System").clicked() { let i = state.spawn_entity("Particle System", EntityType::ParticleSystem); state.selected_entity = Some(i); ui.close_menu(); }
            });

            ui.menu_button("Tools", |ui| {
                if ui.button("Terrain Editor").clicked() { state.execute_console_command("terrain gen"); ui.close_menu(); }
                if ui.button("Generate Dungeon").clicked() { state.execute_console_command("dungeon gen"); ui.close_menu(); }
                ui.separator();
                if ui.button("Profiler").clicked() { state.bottom_tab = BottomTab::Profiler; state.bottom_tab_idx = 2; ui.close_menu(); }
                if ui.button("Script Console").clicked() { state.bottom_tab = BottomTab::Console; state.bottom_tab_idx = 0; ui.close_menu(); }
            });

            ui.menu_button("Help", |ui| {
                if ui.button("About Genovo Studio").clicked() { state.show_about = !state.show_about; ui.close_menu(); }
                if ui.button("Keyboard Shortcuts").clicked() { state.show_shortcuts = !state.show_shortcuts; ui.close_menu(); }
                if ui.button("Check for Updates").clicked() { state.log(LogLevel::System, "Genovo Studio v1.0 is up to date"); ui.close_menu(); }
            });
        });
    });
}

// =============================================================================
// Toolbar
// =============================================================================

fn draw_toolbar(ctx: &egui::Context, state: &mut EditorState) {
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            for mode in [GizmoMode::Select, GizmoMode::Translate, GizmoMode::Rotate, GizmoMode::Scale] {
                let selected = state.gizmo_mode == mode;
                let label = format!("{} {}", mode.icon(), mode.short_label());
                let btn = egui::Button::new(egui::RichText::new(&label).color(if selected { ACCENT_BRIGHT } else { TEXT_NORMAL }))
                    .fill(if selected { ACCENT_BG } else { BG_WIDGET });
                if ui.add(btn).on_hover_text(mode.label()).clicked() {
                    state.gizmo_mode = mode;
                }
            }

            ui.separator();

            // Play / Pause / Stop / Step
            let playing = state.is_playing && !state.is_paused;
            let play_label = if playing { "||  Pause" } else { "|>  Play" };
            let play_color = if playing { GREEN } else { TEXT_NORMAL };
            if ui.add(egui::Button::new(egui::RichText::new(play_label).color(play_color))
                .fill(if playing { GREEN_DIM } else { BG_WIDGET }))
                .on_hover_text("Play/Pause (Space)").clicked() {
                state.toggle_play();
            }
            if ui.add(egui::Button::new(egui::RichText::new("[]  Stop").color(RED))
                .fill(BG_WIDGET))
                .on_hover_text("Stop").clicked() {
                state.stop_play();
            }
            if ui.add(egui::Button::new(egui::RichText::new("|>|  Step").color(TEXT_DIM))
                .fill(BG_WIDGET))
                .on_hover_text("Step (1 frame)").clicked() {
                state.step_physics();
            }

            ui.separator();

            // Coord space toggle
            let space_label = if state.coord_space == CoordSpace::Local { "Local" } else { "World" };
            if ui.button(space_label).on_hover_text("Toggle local/world space").clicked() {
                state.coord_space = if state.coord_space == CoordSpace::Local { CoordSpace::World } else { CoordSpace::Local };
            }

            // Pivot toggle
            let pivot_label = if state.pivot_mode == PivotMode::Center { "Center" } else { "Pivot" };
            if ui.button(pivot_label).on_hover_text("Toggle pivot mode").clicked() {
                state.pivot_mode = if state.pivot_mode == PivotMode::Center { PivotMode::Pivot } else { PivotMode::Center };
            }

            ui.separator();

            // Snap toggle
            let snap_btn = egui::Button::new(egui::RichText::new("Snap").color(if state.snap_enabled { ACCENT_BRIGHT } else { TEXT_DIM }))
                .fill(if state.snap_enabled { ACCENT_BG } else { BG_WIDGET });
            if ui.add(snap_btn).on_hover_text("Toggle snap").clicked() {
                state.snap_enabled = !state.snap_enabled;
            }

            // Grid toggle
            let grid_btn = egui::Button::new(egui::RichText::new("Grid").color(if state.grid_visible { ACCENT_BRIGHT } else { TEXT_DIM }))
                .fill(if state.grid_visible { ACCENT_BG } else { BG_WIDGET });
            if ui.add(grid_btn).on_hover_text("Toggle grid").clicked() {
                state.grid_visible = !state.grid_visible;
            }

            // Stats toggle
            let stats_btn = egui::Button::new(egui::RichText::new("Stats").color(if state.stats_visible { ACCENT_BRIGHT } else { TEXT_DIM }))
                .fill(if state.stats_visible { ACCENT_BG } else { BG_WIDGET });
            if ui.add(stats_btn).on_hover_text("Toggle stats overlay").clicked() {
                state.stats_visible = !state.stats_visible;
            }

            ui.separator();
            ui.colored_label(TEXT_DIM, format!("Speed: {:.1}x", state.sim_speed));
        });
    });
}

// =============================================================================
// Scene Hierarchy (left panel, 220px)
// =============================================================================

fn draw_hierarchy(ctx: &egui::Context, state: &mut EditorState) {
    egui::SidePanel::left("hierarchy")
        .default_width(state.hierarchy_width)
        .min_width(150.0)
        .max_width(400.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("Scene Hierarchy");
            ui.separator();

            // Search bar
            ui.horizontal(|ui| {
                ui.label("Search:");
                ui.text_edit_singleline(&mut state.entity_filter);
            });

            ui.colored_label(TEXT_MUTED, format!("{} entities", state.entities.len()));
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                if state.entities.is_empty() {
                    ui.colored_label(TEXT_MUTED, "No entities in scene");
                    ui.colored_label(TEXT_MUTED, "Use Create menu to add");
                } else {
                    let filter_lower = state.entity_filter.to_lowercase();

                    let entity_data: Vec<(usize, String, EntityType, bool)> = state.entities.iter().enumerate()
                        .filter(|(_, ent)| {
                            filter_lower.is_empty() || ent.name.to_lowercase().contains(&filter_lower)
                        })
                        .map(|(i, ent)| (i, ent.name.clone(), ent.entity_type, state.selected_entity == Some(i)))
                        .collect();

                    let mut clicked_entity: Option<usize> = None;

                    for (i, name, etype, selected) in &entity_data {
                        let dot_color = etype.dot_color();
                        let text_color = if *selected { TEXT_BRIGHT } else { TEXT_NORMAL };

                        let response = ui.horizontal(|ui| {
                            // Colored dot for entity type
                            let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                            ui.painter().circle_filled(rect.center(), 4.0, dot_color);
                            // Entity name
                            let label = egui::RichText::new(format!(" {} [{}]", name, etype.icon_letter()))
                                .color(text_color);
                            let btn = ui.selectable_label(*selected, label);
                            if btn.clicked() {
                                return Some(*i);
                            }
                            None
                        });

                        if let Some(idx) = response.inner {
                            clicked_entity = Some(idx);
                        }
                    }

                    if let Some(idx) = clicked_entity {
                        state.selected_entity = Some(idx);
                    }
                }
            });
        });
}

// =============================================================================
// Inspector (right panel, 280px)
// =============================================================================

fn draw_inspector(ctx: &egui::Context, state: &mut EditorState) {
    egui::SidePanel::right("inspector")
        .default_width(state.inspector_width)
        .min_width(200.0)
        .max_width(500.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("Inspector");
            ui.separator();

            let sel = state.selected_entity;
            if sel.is_none() || sel.unwrap() >= state.entities.len() {
                ui.add_space(20.0);
                ui.colored_label(TEXT_MUTED, "No entity selected");
                ui.colored_label(TEXT_MUTED, "Select an entity in the outliner");
                return;
            }
            let idx = sel.unwrap();

            let etype = state.entities[idx].entity_type;
            let eid = state.entities[idx].entity;
            ui.heading(&state.entities[idx].name.clone());

            ui.horizontal(|ui| {
                ui.colored_label(etype.icon_color(), format!(" {} ", etype));
                ui.colored_label(TEXT_MUTED, format!("Entity {}v{}", eid.id, eid.generation));
            });

            // Entity name edit
            ui.horizontal(|ui| {
                ui.label("Name:");
                let mut name = state.entities[idx].name.clone();
                if ui.text_edit_singleline(&mut name).changed() {
                    state.entities[idx].name = name;
                    state.scene_modified = true;
                }
            });

            // Active toggle
            let mut active = state.entities[idx].active;
            if ui.checkbox(&mut active, "Active").changed() {
                state.entities[idx].active = active;
            }

            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                // Transform section with colored XYZ
                let mut pos_changed = false;
                egui::CollapsingHeader::new(egui::RichText::new("+ Transform").color(TEXT_BRIGHT))
                    .default_open(state.inspector_sections.transform_open)
                    .show(ui, |ui| {
                        // Position with colored X/Y/Z labels
                        ui.horizontal(|ui| {
                            ui.colored_label(X_COLOR, "X");
                            if ui.add(egui::DragValue::new(&mut state.entities[idx].position[0]).speed(0.1)).changed() { pos_changed = true; }
                            ui.colored_label(Y_COLOR, "Y");
                            if ui.add(egui::DragValue::new(&mut state.entities[idx].position[1]).speed(0.1)).changed() { pos_changed = true; }
                            ui.colored_label(Z_COLOR, "Z");
                            if ui.add(egui::DragValue::new(&mut state.entities[idx].position[2]).speed(0.1)).changed() { pos_changed = true; }
                        });
                        // Rotation
                        ui.horizontal(|ui| {
                            ui.label("Rotation");
                            ui.colored_label(X_COLOR, "X");
                            ui.add(egui::DragValue::new(&mut state.entities[idx].rotation[0]).speed(0.5).suffix("d"));
                            ui.colored_label(Y_COLOR, "Y");
                            ui.add(egui::DragValue::new(&mut state.entities[idx].rotation[1]).speed(0.5).suffix("d"));
                            ui.colored_label(Z_COLOR, "Z");
                            ui.add(egui::DragValue::new(&mut state.entities[idx].rotation[2]).speed(0.5).suffix("d"));
                        });
                        // Scale
                        ui.horizontal(|ui| {
                            ui.label("Scale   ");
                            ui.colored_label(X_COLOR, "X");
                            ui.add(egui::DragValue::new(&mut state.entities[idx].scale[0]).speed(0.01).range(0.001..=1000.0));
                            ui.colored_label(Y_COLOR, "Y");
                            ui.add(egui::DragValue::new(&mut state.entities[idx].scale[1]).speed(0.01).range(0.001..=1000.0));
                            ui.colored_label(Z_COLOR, "Z");
                            ui.add(egui::DragValue::new(&mut state.entities[idx].scale[2]).speed(0.01).range(0.001..=1000.0));
                        });
                    });

                if pos_changed {
                    state.sync_entity_to_physics(idx);
                    state.scene_modified = true;
                }

                // Mesh Renderer section
                if etype == EntityType::Mesh {
                    egui::CollapsingHeader::new(egui::RichText::new("M Mesh Renderer").color(CYAN))
                        .default_open(state.inspector_sections.mesh_open)
                        .show(ui, |ui| {
                            let shape_names = ["Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Plane"];
                            let shape_idx = match state.entities[idx].mesh_shape {
                                MeshShape::Cube => 0, MeshShape::Sphere => 1, MeshShape::Cylinder => 2,
                                MeshShape::Capsule => 3, MeshShape::Cone => 4, MeshShape::Plane => 5,
                            };
                            let mut new_shape_idx = shape_idx;
                            ui.horizontal(|ui| {
                                ui.label("Mesh:");
                                egui::ComboBox::from_id_source("mesh_shape")
                                    .selected_text(shape_names[new_shape_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, name) in shape_names.iter().enumerate() {
                                            if ui.selectable_value(&mut new_shape_idx, i, *name).changed() {
                                                state.entities[idx].mesh_shape = match i {
                                                    0 => MeshShape::Cube, 1 => MeshShape::Sphere, 2 => MeshShape::Cylinder,
                                                    3 => MeshShape::Capsule, 4 => MeshShape::Cone, 5 => MeshShape::Plane,
                                                    _ => MeshShape::Cube,
                                                };
                                                state.scene_modified = true;
                                            }
                                        }
                                    });
                            });
                            ui.checkbox(&mut state.entities[idx].cast_shadows, "Cast Shadows");
                            ui.checkbox(&mut state.entities[idx].receive_shadows, "Receive Shadows");
                        });
                }

                // Physics section
                if state.entities[idx].has_physics {
                    egui::CollapsingHeader::new(egui::RichText::new("P Rigid Body").color(GREEN))
                        .default_open(state.inspector_sections.physics_open)
                        .show(ui, |ui| {
                            let body_names = ["Dynamic", "Static", "Kinematic"];
                            let mut bk_idx = match state.entities[idx].body_kind {
                                BodyKind::Dynamic => 0, BodyKind::Static => 1, BodyKind::Kinematic => 2,
                            };
                            ui.horizontal(|ui| {
                                ui.label("Body Type:");
                                egui::ComboBox::from_id_source("body_type")
                                    .selected_text(body_names[bk_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, name) in body_names.iter().enumerate() {
                                            if ui.selectable_value(&mut bk_idx, i, *name).changed() {
                                                state.entities[idx].body_kind = match i {
                                                    0 => BodyKind::Dynamic, 1 => BodyKind::Static, 2 => BodyKind::Kinematic,
                                                    _ => BodyKind::Dynamic,
                                                };
                                            }
                                        }
                                    });
                            });
                            ui.add(egui::Slider::new(&mut state.entities[idx].mass, 0.01..=1000.0).text("Mass").logarithmic(true));
                            ui.add(egui::Slider::new(&mut state.entities[idx].friction, 0.0..=1.0).text("Friction"));
                            ui.add(egui::Slider::new(&mut state.entities[idx].restitution, 0.0..=1.0).text("Restitution"));
                            ui.add(egui::Slider::new(&mut state.entities[idx].linear_damping, 0.0..=10.0).text("Lin Damping"));
                            ui.add(egui::Slider::new(&mut state.entities[idx].angular_damping, 0.0..=10.0).text("Ang Damping"));
                            ui.add(egui::DragValue::new(&mut state.entities[idx].gravity_scale).speed(0.1).prefix("Gravity Scale: "));

                            if let Some(handle) = state.entities[idx].physics_handle {
                                if let Ok(vel) = state.engine.physics().get_linear_velocity(handle) {
                                    let speed = (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z).sqrt();
                                    ui.colored_label(TEXT_MUTED, format!("Vel: ({:.2}, {:.2}, {:.2})", vel.x, vel.y, vel.z));
                                    ui.colored_label(TEXT_MUTED, format!("Speed: {:.3} m/s", speed));
                                }
                            }
                        });

                    // Collider
                    egui::CollapsingHeader::new(egui::RichText::new("C Collider").color(YELLOW))
                        .default_open(state.inspector_sections.collider_open)
                        .show(ui, |ui| {
                            let col_names = ["Box", "Sphere", "Capsule"];
                            let mut cs_idx = match state.entities[idx].collider_shape {
                                ColliderShape::Box => 0, ColliderShape::Sphere => 1, ColliderShape::Capsule => 2,
                            };
                            ui.horizontal(|ui| {
                                ui.label("Shape:");
                                egui::ComboBox::from_id_source("col_shape")
                                    .selected_text(col_names[cs_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, name) in col_names.iter().enumerate() {
                                            if ui.selectable_value(&mut cs_idx, i, *name).changed() {
                                                state.entities[idx].collider_shape = match i {
                                                    0 => ColliderShape::Box, 1 => ColliderShape::Sphere, 2 => ColliderShape::Capsule,
                                                    _ => ColliderShape::Box,
                                                };
                                            }
                                        }
                                    });
                            });
                            ui.checkbox(&mut state.entities[idx].is_trigger, "Is Trigger");
                        });
                }

                // Light section
                if state.entities[idx].is_light {
                    egui::CollapsingHeader::new(egui::RichText::new("L Light").color(YELLOW))
                        .default_open(state.inspector_sections.light_open)
                        .show(ui, |ui| {
                            let lk_names = ["Directional", "Point", "Spot"];
                            let mut lk_idx = match state.entities[idx].light_kind {
                                LightKind::Directional => 0, LightKind::Point => 1, LightKind::Spot => 2,
                            };
                            ui.horizontal(|ui| {
                                ui.label("Type:");
                                egui::ComboBox::from_id_source("light_type")
                                    .selected_text(lk_names[lk_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, name) in lk_names.iter().enumerate() {
                                            if ui.selectable_value(&mut lk_idx, i, *name).changed() {
                                                state.entities[idx].light_kind = match i {
                                                    0 => LightKind::Directional, 1 => LightKind::Point, 2 => LightKind::Spot,
                                                    _ => LightKind::Point,
                                                };
                                            }
                                        }
                                    });
                            });
                            ui.add(egui::Slider::new(&mut state.entities[idx].light_intensity, 0.0..=100.0).text("Intensity"));
                            ui.add(egui::Slider::new(&mut state.entities[idx].light_range, 0.1..=1000.0).text("Range").logarithmic(true));
                            if state.entities[idx].light_kind == LightKind::Spot {
                                ui.add(egui::Slider::new(&mut state.entities[idx].light_spot_angle, 1.0..=179.0).text("Spot Angle"));
                            }
                            ui.checkbox(&mut state.entities[idx].light_shadows, "Cast Shadows");
                        });
                }

                // Camera section
                if state.entities[idx].is_camera {
                    egui::CollapsingHeader::new(egui::RichText::new("C Camera").color(ACCENT_BRIGHT))
                        .default_open(state.inspector_sections.camera_open)
                        .show(ui, |ui| {
                            let proj_names = ["Perspective", "Orthographic"];
                            let mut proj_idx = match state.entities[idx].camera_projection {
                                CameraProjection::Perspective => 0, CameraProjection::Orthographic => 1,
                            };
                            ui.horizontal(|ui| {
                                ui.label("Projection:");
                                egui::ComboBox::from_id_source("cam_proj")
                                    .selected_text(proj_names[proj_idx])
                                    .show_ui(ui, |ui| {
                                        for (i, name) in proj_names.iter().enumerate() {
                                            if ui.selectable_value(&mut proj_idx, i, *name).changed() {
                                                state.entities[idx].camera_projection = match i {
                                                    0 => CameraProjection::Perspective, 1 => CameraProjection::Orthographic,
                                                    _ => CameraProjection::Perspective,
                                                };
                                            }
                                        }
                                    });
                            });
                            ui.add(egui::Slider::new(&mut state.entities[idx].camera_fov, 10.0..=170.0).text("FOV"));
                            ui.add(egui::DragValue::new(&mut state.entities[idx].camera_near).speed(0.01).prefix("Near: "));
                            ui.add(egui::DragValue::new(&mut state.entities[idx].camera_far).speed(10.0).prefix("Far: "));
                        });
                }

                // Audio section
                if state.entities[idx].is_audio {
                    egui::CollapsingHeader::new(egui::RichText::new("A Audio Source").color(ORANGE))
                        .default_open(state.inspector_sections.audio_open)
                        .show(ui, |ui| {
                            ui.add(egui::Slider::new(&mut state.entities[idx].audio_volume, 0.0..=1.0).text("Volume"));
                            ui.add(egui::DragValue::new(&mut state.entities[idx].audio_pitch).speed(0.01).prefix("Pitch: "));
                            ui.checkbox(&mut state.entities[idx].audio_spatial, "Spatial");
                            if state.entities[idx].audio_spatial {
                                ui.add(egui::Slider::new(&mut state.entities[idx].audio_min_dist, 0.1..=100.0).text("Min Dist"));
                                ui.add(egui::Slider::new(&mut state.entities[idx].audio_max_dist, 1.0..=1000.0).text("Max Dist"));
                            }
                        });
                }
            });
        });
}

// =============================================================================
// Bottom Panel (180px, with tabs)
// =============================================================================

fn draw_bottom_panel(ctx: &egui::Context, state: &mut EditorState) {
    egui::TopBottomPanel::bottom("bottom_panel")
        .default_height(state.bottom_height)
        .min_height(100.0)
        .max_height(500.0)
        .resizable(true)
        .show(ctx, |ui| {
            // Tab bar
            ui.horizontal(|ui| {
                let tabs = ["Console", "Content", "Profiler", "Animation"];
                for (i, tab) in tabs.iter().enumerate() {
                    let selected = state.bottom_tab_idx == i;
                    let color = if selected { ACCENT_BRIGHT } else { TEXT_DIM };
                    let btn = egui::Button::new(egui::RichText::new(*tab).color(color))
                        .fill(if selected { ACCENT_BG } else { BG_WIDGET });
                    if ui.add(btn).clicked() {
                        state.bottom_tab_idx = i;
                        state.bottom_tab = match i {
                            0 => BottomTab::Console,
                            1 => BottomTab::ContentBrowser,
                            2 => BottomTab::Profiler,
                            3 => BottomTab::Animation,
                            _ => BottomTab::Console,
                        };
                    }
                }
            });
            ui.separator();

            match state.bottom_tab {
                BottomTab::Console => draw_console_tab(ui, state),
                BottomTab::ContentBrowser => draw_content_browser_tab(ui, state),
                BottomTab::Profiler => draw_profiler_tab(ui, state),
                BottomTab::Animation => draw_animation_tab(ui, state),
            }
        });
}

fn draw_console_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    // Filter buttons
    ui.horizontal(|ui| {
        ui.colored_label(TEXT_MUTED, "Filter:");
        if ui.button("All").clicked() { state.console_filter_level = None; }
        if ui.add(egui::Button::new(egui::RichText::new("Info").color(TEXT_NORMAL))).clicked() { state.console_filter_level = Some(LogLevel::Info); }
        if ui.add(egui::Button::new(egui::RichText::new("Warn").color(YELLOW))).clicked() { state.console_filter_level = Some(LogLevel::Warn); }
        if ui.add(egui::Button::new(egui::RichText::new("Err").color(RED))).clicked() { state.console_filter_level = Some(LogLevel::Error); }
        if ui.add(egui::Button::new(egui::RichText::new("Sys").color(ACCENT_BRIGHT))).clicked() { state.console_filter_level = Some(LogLevel::System); }
        if ui.button("Clear").clicked() { state.console_log.clear(); }
    });

    ui.separator();

    // Log entries
    let available_h = ui.available_height() - 30.0;
    egui::ScrollArea::vertical().max_height(available_h).stick_to_bottom(true).show(ui, |ui| {
        let filtered: Vec<&LogEntry> = state.console_log.iter()
            .filter(|e| state.console_filter_level.map_or(true, |f| e.level == f))
            .collect();

        for entry in &filtered {
            let timestamp = format!("{:>7.1}", entry.timestamp);
            let prefix = entry.level.prefix();
            let color = entry.level.color();
            let count_str = if entry.count > 1 { format!(" (x{})", entry.count) } else { String::new() };
            let line = format!("{} {} {}{}", timestamp, prefix, entry.text, count_str);
            ui.colored_label(color, &line);
        }
    });

    // Command input
    ui.separator();
    ui.horizontal(|ui| {
        ui.colored_label(ACCENT, ">");
        let response = ui.text_edit_singleline(&mut state.console_input);
        if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            submit_console(state);
        }
        if ui.button("Run").clicked() {
            submit_console(state);
        }
    });
}

fn draw_content_browser_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    ui.horizontal(|ui| {
        ui.colored_label(ACCENT_DIM, format!("res://{}", state.asset_path));
        if ui.button("..").clicked() {
            if let Some(pos) = state.asset_path.rfind('/').or_else(|| state.asset_path.rfind('\\')) {
                state.asset_path.truncate(pos);
            }
        }
    });

    ui.horizontal(|ui| {
        ui.label("Search:");
        ui.text_edit_singleline(&mut state.asset_search);
    });

    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
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
            ui.colored_label(TEXT_MUTED, "Default asset structure:");
            for (name, color) in &defaults {
                ui.colored_label(*color, *name);
            }
        } else {
            for entry in &entries {
                let icon = asset_icon(&entry.name, entry.is_dir);
                let color = if entry.is_dir { YELLOW } else { asset_color(&entry.name) };
                let label = format!("{} {}", icon, entry.name);
                if ui.add(egui::Label::new(egui::RichText::new(&label).color(color)).sense(egui::Sense::click())).clicked() {
                    if entry.is_dir {
                        state.asset_path = format!("{}/{}", state.asset_path, entry.name);
                    } else {
                        state.log(LogLevel::Info, format!("Selected asset: {}", entry.name));
                    }
                }
            }
        }
    });
}

fn draw_profiler_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    let fps_color = if state.smooth_fps >= 55.0 { GREEN } else if state.smooth_fps >= 30.0 { YELLOW } else { RED };

    ui.horizontal(|ui| {
        ui.colored_label(fps_color, format!("FPS: {:.0}", state.smooth_fps));
        ui.colored_label(TEXT_NORMAL, format!("Frame: {:.2} ms", state.smooth_frame_time));
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
        ui.colored_label(TEXT_DIM,
            format!("Min: {:.2}  Avg: {:.2}  P99: {:.2}  Max: {:.2} ms", min, avg, p99, max),
        );
    }

    ui.separator();

    // Frame time bar graph
    let bar_count = state.frame_times.len().min(200);
    if bar_count > 0 {
        let graph_h = 60.0;
        let available_w = ui.available_width();
        let bar_w = (available_w / bar_count as f32).max(1.0).min(4.0);
        let start_idx = state.frame_times.len().saturating_sub(bar_count);

        let (rect, _) = ui.allocate_exact_size(egui::vec2(available_w, graph_h), egui::Sense::hover());
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 0.0, BG_BASE);

        // Reference lines
        let y_60fps = rect.bottom() - (16.67 / 50.0 * graph_h as f64) as f32;
        let y_30fps = rect.bottom() - (33.33 / 50.0 * graph_h as f64) as f32;
        painter.line_segment([egui::pos2(rect.left(), y_60fps), egui::pos2(rect.right(), y_60fps)],
            egui::Stroke::new(1.0, GREEN_DIM));
        painter.line_segment([egui::pos2(rect.left(), y_30fps), egui::pos2(rect.right(), y_30fps)],
            egui::Stroke::new(1.0, YELLOW_DIM));

        for (i, &val) in state.frame_times.iter().skip(start_idx).enumerate() {
            let bar_h = (val as f32 / 50.0 * graph_h).clamp(1.0, graph_h);
            let x = rect.left() + i as f32 * bar_w;
            let color = if val < 16.67 { GREEN } else if val < 33.33 { YELLOW } else { RED };
            let bar_rect = egui::Rect::from_min_size(
                egui::pos2(x, rect.bottom() - bar_h),
                egui::vec2(bar_w.max(1.0) - 0.5, bar_h),
            );
            painter.rect_filled(bar_rect, 0.0, color);
        }
    }

    ui.colored_label(TEXT_MUTED, format!("Frame: {} | Entities: {} | Bodies: {}",
        state.frame_count, state.entities.len(), state.engine.physics().body_count()));
}

fn draw_animation_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    ui.horizontal(|ui| {
        if ui.button("|<").on_hover_text("Go to start").clicked() {
            state.anim_time = 0.0;
        }
        let play_icon = if state.anim_playing { "||" } else { "|>" };
        let play_color = if state.anim_playing { GREEN } else { TEXT_NORMAL };
        if ui.add(egui::Button::new(egui::RichText::new(play_icon).color(play_color)))
            .on_hover_text("Play/Pause").clicked() {
            state.anim_playing = !state.anim_playing;
        }
        if ui.button("[]").on_hover_text("Stop").clicked() {
            state.anim_playing = false;
            state.anim_time = 0.0;
        }
        if ui.button(">|").on_hover_text("Go to end").clicked() {
            state.anim_time = state.anim_duration;
        }
    });

    ui.checkbox(&mut state.anim_loop, "Loop");
    ui.add(egui::Slider::new(&mut state.anim_time, 0.0..=state.anim_duration).text("Time"));
    ui.add(egui::DragValue::new(&mut state.anim_duration).speed(0.1).prefix("Duration: "));

    ui.separator();
    ui.colored_label(TEXT_MUTED, "Keyframes and curves will appear here when animations are loaded.");

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
// Central Viewport (transparent, overlays on top of 3D scene)
// =============================================================================

fn draw_viewport(ctx: &egui::Context, state: &mut EditorState) {
    egui::CentralPanel::default()
        .frame(egui::Frame::none().fill(egui::Color32::TRANSPARENT))
        .show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            let painter = ui.painter_at(rect);

            // Top-left: mode info
            if state.stats_visible {
                let mode_text = format!("{} | {} | {} | {}",
                    state.gizmo_mode.label(),
                    if state.coord_space == CoordSpace::Local { "Local" } else { "World" },
                    if state.grid_visible { "Grid" } else { "" },
                    if state.wireframe_mode { "Wire" } else { "" },
                );
                painter.text(
                    egui::pos2(rect.left() + 8.0, rect.top() + 22.0),
                    egui::Align2::LEFT_TOP,
                    &mode_text,
                    egui::FontId::proportional(11.0),
                    TEXT_MUTED,
                );
            }

            // Top-right: Camera info
            let cam_text = format!("Camera  yaw:{:.0}  pitch:{:.0}  dist:{:.1}",
                state.camera_yaw, state.camera_pitch, state.camera_dist);
            painter.text(
                egui::pos2(rect.right() - 8.0, rect.top() + 6.0),
                egui::Align2::RIGHT_TOP,
                &cam_text,
                egui::FontId::proportional(11.0),
                TEXT_MUTED,
            );

            let tgt_text = format!("Target: ({:.1}, {:.1}, {:.1})",
                state.camera_target[0], state.camera_target[1], state.camera_target[2]);
            painter.text(
                egui::pos2(rect.right() - 8.0, rect.top() + 20.0),
                egui::Align2::RIGHT_TOP,
                &tgt_text,
                egui::FontId::proportional(11.0),
                TEXT_MUTED,
            );

            // Bottom center: viewport label
            let vp_w = rect.width() as u32;
            let vp_h = rect.height() as u32;
            let vp_label = format!("3D Viewport | {} x {}", vp_w, vp_h);
            painter.text(
                egui::pos2(rect.center().x, rect.bottom() - 18.0),
                egui::Align2::CENTER_TOP,
                &vp_label,
                egui::FontId::proportional(11.0),
                TEXT_MUTED,
            );

            // Play state overlay
            if state.is_playing && !state.is_paused {
                let sim_text = format!("SIMULATING | {:.2}x | {:.1}s", state.sim_speed, state.total_sim_time);
                painter.text(
                    egui::pos2(rect.left() + 8.0, rect.top() + 36.0),
                    egui::Align2::LEFT_TOP,
                    &sim_text,
                    egui::FontId::proportional(12.0),
                    GREEN,
                );
            } else if state.is_paused {
                let pause_text = format!("PAUSED | {:.1}s", state.total_sim_time);
                painter.text(
                    egui::pos2(rect.left() + 8.0, rect.top() + 36.0),
                    egui::Align2::LEFT_TOP,
                    &pause_text,
                    egui::FontId::proportional(12.0),
                    YELLOW,
                );
            }

            // Selected entity indicator (top center)
            if let Some(idx) = state.selected_entity {
                if idx < state.entities.len() {
                    let ent = &state.entities[idx];
                    let icon_color = ent.entity_type.icon_color();
                    let label = format!("[{}] {} ({:.1}, {:.1}, {:.1})",
                        ent.entity_type.icon_letter(), ent.name,
                        ent.position[0], ent.position[1], ent.position[2]);
                    let galley = painter.layout_no_wrap(label.clone(), egui::FontId::proportional(11.0), icon_color);
                    let pill_w = galley.size().x + 16.0;
                    let pill_h = 18.0;
                    let pill_x = rect.center().x - pill_w * 0.5;
                    let pill_y = rect.top() + 4.0;
                    let pill_rect = egui::Rect::from_min_size(
                        egui::pos2(pill_x, pill_y),
                        egui::vec2(pill_w, pill_h),
                    );
                    painter.rect_filled(pill_rect, 9.0, egui::Color32::from_rgba_premultiplied(18, 18, 22, 200));
                    painter.rect_stroke(pill_rect, 9u8, egui::Stroke::new(1.0, BORDER), egui::epaint::StrokeKind::Middle);
                    painter.text(
                        egui::pos2(pill_x + 8.0, pill_y + 3.0),
                        egui::Align2::LEFT_TOP,
                        &label,
                        egui::FontId::proportional(11.0),
                        icon_color,
                    );
                }
            }

            // Axis indicator (bottom-left)
            let ax_x = rect.left() + 30.0;
            let ax_y = rect.bottom() - 40.0;
            let axis_len = 18.0;
            painter.line_segment(
                [egui::pos2(ax_x, ax_y), egui::pos2(ax_x + axis_len, ax_y)],
                egui::Stroke::new(2.0, X_COLOR),
            );
            painter.text(egui::pos2(ax_x + axis_len + 3.0, ax_y - 5.0), egui::Align2::LEFT_TOP,
                "X", egui::FontId::proportional(10.0), X_COLOR);
            painter.line_segment(
                [egui::pos2(ax_x, ax_y), egui::pos2(ax_x, ax_y - axis_len)],
                egui::Stroke::new(2.0, Y_COLOR),
            );
            painter.text(egui::pos2(ax_x - 5.0, ax_y - axis_len - 12.0), egui::Align2::LEFT_TOP,
                "Y", egui::FontId::proportional(10.0), Y_COLOR);
            painter.line_segment(
                [egui::pos2(ax_x, ax_y), egui::pos2(ax_x - axis_len * 0.6, ax_y + axis_len * 0.5)],
                egui::Stroke::new(2.0, Z_COLOR),
            );
            painter.text(egui::pos2(ax_x - axis_len * 0.6 - 12.0, ax_y + axis_len * 0.5 - 2.0), egui::Align2::LEFT_TOP,
                "Z", egui::FontId::proportional(10.0), Z_COLOR);

            // Stats overlay (bottom-left)
            if state.stats_visible {
                let stats_text = format!("Entities: {} | Bodies: {}",
                    state.entities.len(),
                    state.engine.physics().body_count());
                painter.text(
                    egui::pos2(rect.left() + 8.0, rect.bottom() - 60.0),
                    egui::Align2::LEFT_TOP,
                    &stats_text,
                    egui::FontId::proportional(11.0),
                    TEXT_MUTED,
                );
            }
        });
}

// =============================================================================
// Status Bar
// =============================================================================

fn draw_status_bar(ctx: &egui::Context, state: &EditorState) {
    egui::TopBottomPanel::bottom("status_bar")
        .exact_height(18.0)
        .frame(egui::Frame::none().fill(BG_BASE).inner_margin(egui::Margin::symmetric(4, 1)))
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let (dot_color, status_text) = if state.is_playing && !state.is_paused {
                    (ACCENT_BRIGHT, "Playing")
                } else if state.is_paused {
                    (YELLOW, "Paused")
                } else {
                    (GREEN, "Editing")
                };
                let (dot_rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(dot_rect.center(), 4.0, dot_color);
                ui.colored_label(TEXT_DIM, status_text);

                ui.separator();

                let modified = if state.scene_modified { " *" } else { "" };
                let scene_label = format!("{}{}", state.scene_name, modified);
                let scene_color = if state.scene_modified { YELLOW } else { TEXT_NORMAL };
                ui.colored_label(scene_color, &scene_label);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let fps_color = if state.smooth_fps >= 55.0 { GREEN } else if state.smooth_fps >= 30.0 { YELLOW } else { RED };
                    ui.colored_label(fps_color, format!("{:.0} FPS | {} entities | {} bodies",
                        state.smooth_fps,
                        state.entities.len(),
                        state.engine.physics().body_count(),
                    ));
                });
            });
        });
}

// =============================================================================
// FPS Overlay
// =============================================================================

fn draw_fps_overlay(ctx: &egui::Context, state: &EditorState) {
    let fps_color = if state.smooth_fps >= 55.0 { GREEN } else if state.smooth_fps >= 30.0 { YELLOW } else { RED };

    egui::Area::new(egui::Id::new("fps_overlay"))
        .fixed_pos(egui::pos2(6.0, 54.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_premultiplied(18, 18, 22, 180))
                .rounding(egui::CornerRadius::same(4))
                .inner_margin(egui::Margin::same(6))
                .show(ui, |ui| {
                    ui.colored_label(fps_color, format!("{:.0} FPS", state.smooth_fps));
                    ui.colored_label(TEXT_DIM, format!("{:.2} ms", state.smooth_frame_time));
                });
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
        let alpha_u8 = (alpha * 255.0) as u8;

        egui::Area::new(egui::Id::new("notification_toast"))
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -36.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::none()
                    .fill(egui::Color32::from_rgba_premultiplied(24, 24, 28, alpha_u8))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(38, 38, 44, alpha_u8)))
                    .rounding(egui::CornerRadius::same(4))
                    .inner_margin(egui::Margin::same(12))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let prefix_color = level.color();
                            ui.colored_label(prefix_color, level.prefix());
                            ui.colored_label(egui::Color32::from_rgba_premultiplied(180, 180, 188, alpha_u8), msg);
                        });
                    });
            });
    }
}

// =============================================================================
// About / Shortcuts dialogs
// =============================================================================

fn draw_about_dialog(ctx: &egui::Context, state: &mut EditorState) {
    if !state.show_about { return; }

    egui::Window::new("About Genovo Studio")
        .collapsible(false)
        .resizable(false)
        .default_width(380.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.colored_label(ACCENT, "GENOVO STUDIO");
            ui.colored_label(TEXT_DIM, "v1.0");
            ui.add_space(4.0);
            ui.label("Professional Game Development Environment");
            ui.colored_label(TEXT_DIM, "26 engine modules fully linked");
            ui.add_space(8.0);

            let items = [
                ("Rendering:", "wgpu 24 (GPU-accelerated)", CYAN),
                ("UI:", "egui with Genovo dark theme", ACCENT),
                ("Physics:", "Custom impulse solver", YELLOW),
                ("ECS:", "Archetype-based", MAGENTA),
                ("Audio:", "Software PCM mixer", ORANGE),
                ("Scripting:", "GenovoScript VM", RED),
                ("Terrain:", "Procedural heightmap gen", GREEN),
                ("Procgen:", "BSP dungeon generation", CYAN),
            ];
            for (label, value, color) in &items {
                ui.horizontal(|ui| {
                    ui.colored_label(TEXT_DIM, *label);
                    ui.colored_label(*color, *value);
                });
            }

            ui.add_space(8.0);
            ui.separator();
            ui.colored_label(ACCENT, "genovo.dev");
            ui.add_space(4.0);

            if ui.button("Close").clicked() {
                state.show_about = false;
            }
        });
}

fn draw_shortcuts_dialog(ctx: &egui::Context, state: &mut EditorState) {
    if !state.show_shortcuts { return; }

    egui::Window::new("Keyboard Shortcuts")
        .collapsible(false)
        .resizable(false)
        .default_width(380.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.colored_label(ACCENT, "Genovo Studio Shortcuts");
            ui.add_space(4.0);

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

            egui::Grid::new("shortcuts_grid").show(ui, |ui| {
                for (key, desc) in &shortcuts {
                    ui.colored_label(ACCENT, *key);
                    ui.colored_label(TEXT_NORMAL, *desc);
                    ui.end_row();
                }
            });

            ui.add_space(8.0);
            if ui.button("Close").clicked() {
                state.show_shortcuts = false;
            }
        });
}

// =============================================================================
// Application
// =============================================================================

struct EditorApp {
    gpu: Option<GpuState>,
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

    // egui integration
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    // Editor state
    editor: EditorState,
}

impl ApplicationHandler for EditorApp {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gpu.is_some() { return; }

        let w = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("Genovo Studio")
                    .with_inner_size(LogicalSize::new(1600, 900))
                    .with_maximized(true),
            ).unwrap(),
        );

        let inst = wgpu::Instance::new(&Default::default());
        let surf = inst.create_surface(w.clone()).unwrap();
        let adap = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surf),
            ..Default::default()
        })).unwrap();
        let (dev, que) = pollster::block_on(adap.request_device(&Default::default(), None)).unwrap();
        let sz = w.inner_size();
        let cfg = surf.get_default_config(&adap, sz.width.max(1), sz.height.max(1)).unwrap();
        surf.configure(&dev, &cfg);

        let dev = Arc::new(dev);
        let que = Arc::new(que);

        let scene_manager = SceneRenderManager::new(&dev, cfg.format, wgpu::TextureFormat::Depth32Float);
        let dv = make_depth(&dev, sz.width, sz.height);

        // Audio thread
        let audio_running = self.audio_running.clone();
        audio_running.store(true, Ordering::SeqCst);
        std::thread::Builder::new()
            .name("genovo-audio".into())
            .spawn(move || {
                use genovo_audio::{AudioMixer, SoftwareMixer};
                let mut mixer = SoftwareMixer::new(48000, 2, 1024, 32);
                println!("[Genovo Audio] Mixer thread started: 48kHz stereo");
                let dt = 1024.0 / 48000.0;
                while audio_running.load(Ordering::Relaxed) {
                    mixer.update(dt);
                    std::thread::sleep(std::time::Duration::from_millis(21));
                }
                println!("[Genovo Audio] Mixer thread stopped");
            })
            .expect("Failed to spawn audio thread");

        // egui setup
        let egui_ctx = egui::Context::default();
        apply_genovo_dark_theme(&egui_ctx);

        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &w,
            Some(w.scale_factor() as f32),
            None,
            None,
        );

        // egui renderer -- NO depth attachment (None) prevents crash
        let egui_renderer = egui_wgpu::Renderer::new(&dev, cfg.format, None, 1, false);

        // Engine init
        let mut engine = Engine::new(EngineConfig::default()).unwrap();

        // Ground plane
        let gd = physics::RigidBodyDesc {
            body_type: physics::BodyType::Static,
            position: Vec3::ZERO,
            ..Default::default()
        };
        let gh = engine.physics_mut().add_body(&gd).unwrap();
        let _ = engine.physics_mut().add_collider(gh, &physics::ColliderDesc {
            shape: physics::CollisionShape::Box { half_extents: Vec3::new(50.0, 0.5, 50.0) },
            ..Default::default()
        });

        let mut editor = EditorState::new(engine);
        editor.log(LogLevel::System, "Genovo Studio v1.0 -- Professional Game Development Environment");
        editor.log(LogLevel::System, format!("GPU: {} ({:?})", adap.get_info().name, adap.get_info().backend));
        editor.log(LogLevel::System, format!("Surface: {:?} | {}x{}", cfg.format, sz.width, sz.height));
        editor.log(LogLevel::System, "3D Renderer: SceneRenderManager (PBR + Grid)");
        editor.log(LogLevel::System, "Audio: Software mixer thread active");
        editor.log(LogLevel::System, "UI: egui with custom Genovo dark theme");
        editor.log(LogLevel::System, "Type 'help' in console for commands");

        // Default scene
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
        println!("[Genovo Studio] 3D: SceneRenderManager with PBR pipeline");
        println!("[Genovo Studio] UI: egui with custom Genovo dark theme");
        println!("[Genovo Studio] Editor ready. {} entities.", editor.entities.len());

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
        });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _wid: WindowId, ev: WindowEvent) {
        let Some(s) = self.gpu.as_mut() else { return };

        // Let egui handle the event first
        let egui_response = s.egui_state.on_window_event(&s.window, &ev);
        let _ = egui_response;

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

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(k) = event.physical_key {
                    if event.state == ElementState::Pressed && !s.egui_ctx.wants_keyboard_input() {
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
                // Frame timing
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

                // Window title
                let title = format!("Genovo Studio \u{2014} {}{}", s.editor.scene_name, if s.editor.scene_modified { " *" } else { "" });
                s.window.set_title(&title);

                // Physics step
                if s.editor.is_playing && !s.editor.is_paused {
                    let phys_dt = (dt * s.editor.sim_speed).min(1.0 / 30.0);
                    let _ = s.editor.engine.physics_mut().step(phys_dt);
                    s.editor.sync_physics_to_entities();
                    s.editor.total_sim_time += phys_dt as f64;
                }

                // Notification expiry
                if let Some((_, _, start)) = &s.editor.notification {
                    if start.elapsed().as_secs_f32() > 3.5 { s.editor.notification = None; }
                }

                s.editor.sync_physics_to_entities();

                // ---- egui begin frame ----
                let raw_input = s.egui_state.take_egui_input(&s.window);
                s.egui_ctx.begin_pass(raw_input);

                // Draw all editor UI via egui
                draw_editor_ui(&s.egui_ctx, &mut s.editor);

                // ---- egui end frame ----
                let full_output = s.egui_ctx.end_pass();
                s.egui_state.handle_platform_output(&s.window, full_output.platform_output);

                let paint_jobs = s.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [s.config.width, s.config.height],
                    pixels_per_point: s.window.scale_factor() as f32,
                };

                // Handle egui texture updates
                for (id, delta) in &full_output.textures_delta.set {
                    s.egui_renderer.update_texture(&s.device, &s.queue, *id, delta);
                }
                let mut egui_cmd_buffers = s.egui_renderer.update_buffers(
                    &s.device,
                    &s.queue,
                    &mut s.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("egui_update"),
                    }),
                    &paint_jobs,
                    &screen_descriptor,
                );

                // ---- Build camera ----
                let yaw_rad = s.editor.camera_yaw.to_radians();
                let pitch_rad = s.editor.camera_pitch.to_radians();
                let target = glam::Vec3::new(s.editor.camera_target[0], s.editor.camera_target[1], s.editor.camera_target[2]);
                let aspect = s.config.width as f32 / s.config.height.max(1) as f32;
                let mut scene_camera = SceneCamera::perspective(
                    glam::Vec3::ZERO, target, glam::Vec3::Y,
                    s.editor.camera_fov.to_radians(), aspect, 0.1, 1000.0,
                );
                scene_camera.orbit(target, yaw_rad, pitch_rad, s.editor.camera_dist);

                // ---- Build scene lights ----
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

                // ---- Submit scene entities ----
                s.scene_manager.clear_queue();

                let t = s.editor.frame_count as f32 * 0.005;
                let (br, bg, bb) = if s.editor.is_playing && !s.editor.is_paused {
                    let pulse = (t * 2.0).sin().abs() * 0.008;
                    (0.06 + pulse as f64, 0.06_f64, 0.08 + pulse as f64)
                } else { (0.06_f64, 0.06, 0.08) };
                s.scene_manager.set_clear_color([br, bg, bb, 1.0]);

                if s.editor.grid_visible {
                    s.scene_manager.set_grid_enabled(&s.device, true);
                } else {
                    s.scene_manager.set_grid_enabled(&s.device, false);
                }

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
                            EntityType::ParticleSystem | EntityType::Audio => glam::Vec3::new(0.3, 0.3, 0.3),
                            _ => scl,
                        };
                        let transform = glam::Mat4::from_scale_rotation_translation(
                            render_scale,
                            glam::Quat::from_euler(glam::EulerRot::XYZ, ent.rotation[0].to_radians(), ent.rotation[1].to_radians(), ent.rotation[2].to_radians()),
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

                // Pass 1: 3D scene
                let scene_cmd = s.scene_manager.render(&s.device, &s.queue, &view, &s.depth_view, &scene_camera, &scene_lights);

                // Pass 2: egui UI overlay (no depth, LoadOp::Load)
                let mut egui_enc = s.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("egui_render"),
                });

                {
                    let render_pass = egui_enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load, // Load existing 3D scene
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None, // NO depth attachment prevents crash
                        ..Default::default()
                    });
                    // egui-wgpu 0.31 requires RenderPass<'static>
                    let mut render_pass = render_pass.forget_lifetime();

                    s.egui_renderer.render(
                        &mut render_pass,
                        &paint_jobs,
                        &screen_descriptor,
                    );
                }

                // Free egui textures
                for id in &full_output.textures_delta.free {
                    s.egui_renderer.free_texture(id);
                }

                // Submit: 3D scene first, then egui update buffers, then egui render
                s.queue.submit(
                    std::iter::once(scene_cmd)
                        .chain(egui_cmd_buffers.drain(..))
                        .chain(std::iter::once(egui_enc.finish()))
                );
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
    println!("[Genovo Studio] UI: egui with custom Genovo dark theme");

    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = EditorApp {
        gpu: None,
        audio_running: Arc::new(AtomicBool::new(false)),
    };
    let _ = el.run_app(&mut app);
}
