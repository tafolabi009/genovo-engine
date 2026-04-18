//! Genovo Studio -- Professional Game Development Environment
//!
//! Custom Slate UI rendered entirely through UIGpuRenderer (genovo-ui) with
//! bitmap font text rendering. No egui dependency.
//!
//! Architecture:
//!   1. winit window + wgpu device/queue/surface
//!   2. UIGpuRenderer for ALL 2D UI (text, rectangles, lines)
//!   3. EditorState for state management, layout, panels
//!   4. SceneRenderManager for full 3D viewport (PBR, grid, built-in primitives)
//!   5. Each frame: handle events -> draw UI via UIGpuRenderer -> render

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

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
    SceneRenderManager, SceneCamera, SceneLights,
};

// Custom Slate UI framework
use genovo_ui::ui_framework::UIStyle;
use genovo_ui::dock_system::DockStyle;
use genovo_ui::gpu_renderer::UIGpuRenderer;
use genovo_ui::render_commands::Color;
use genovo_core::Rect;

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
// Color constants (linear float RGBA)
// =============================================================================

fn c(r: u8, g: u8, b: u8) -> Color {
    Color::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}
fn ca(r: u8, g: u8, b: u8, a: u8) -> Color {
    Color::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, a as f32 / 255.0)
}

fn bg_base() -> Color { c(18, 18, 22) }
fn bg_panel() -> Color { c(24, 24, 28) }
fn bg_widget() -> Color { c(32, 32, 38) }
fn bg_hover() -> Color { c(42, 42, 50) }
fn border_color() -> Color { c(38, 38, 44) }

fn text_bright() -> Color { c(230, 230, 235) }
fn text_normal() -> Color { c(180, 180, 188) }
fn text_dim() -> Color { c(110, 110, 120) }
fn text_muted() -> Color { c(65, 65, 72) }

fn accent() -> Color { c(56, 132, 244) }
fn accent_dim() -> Color { c(40, 100, 200) }
fn accent_bright() -> Color { c(80, 156, 255) }
fn accent_bg() -> Color { c(30, 60, 110) }

fn green() -> Color { c(72, 199, 142) }
fn yellow() -> Color { c(245, 196, 80) }
fn red() -> Color { c(235, 87, 87) }
fn cyan() -> Color { c(70, 200, 220) }
fn magenta() -> Color { c(190, 80, 210) }
fn orange() -> Color { c(230, 150, 60) }

fn x_color() -> Color { c(235, 75, 75) }
fn y_color() -> Color { c(72, 199, 142) }
fn z_color() -> Color { c(56, 132, 244) }

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

    fn icon_color(&self) -> Color {
        match self {
            EntityType::Empty => text_dim(),
            EntityType::Mesh => cyan(),
            EntityType::Light => yellow(),
            EntityType::Camera => accent_bright(),
            EntityType::ParticleSystem => magenta(),
            EntityType::Audio => orange(),
        }
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
    fn color(&self) -> Color {
        match self {
            LogLevel::Info => text_normal(),
            LogLevel::Warn => yellow(),
            LogLevel::Error => red(),
            LogLevel::System => accent_bright(),
            LogLevel::Debug => c(140, 140, 160),
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
                self.log(LogLevel::System, "  help / clear / stats / list / about");
                self.log(LogLevel::System, "  spawn <type> [name] / delete / select <idx>");
                self.log(LogLevel::System, "  physics start|stop|step / gravity <x> <y> <z>");
                self.log(LogLevel::System, "  terrain gen / dungeon gen / script <code>");
                self.log(LogLevel::System, "  camera <yaw> <pitch> <dist> / speed <val>");
                self.log(LogLevel::System, "  scene <name>");
            }
            "clear" => { self.console_log.clear(); }
            "about" => {
                self.log(LogLevel::System, "Genovo Studio v1.0 -- Professional Game Development Environment");
                self.log(LogLevel::System, "26 engine modules fully linked");
                self.log(LogLevel::System, "UI: Custom Slate (UIGpuRenderer) with bitmap font");
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
// UI Drawing functions using UIGpuRenderer
// =============================================================================

/// Draw the complete editor UI. Called each frame with the renderer.
fn draw_editor_ui(r: &mut UIGpuRenderer, state: &EditorState) {
    let screen = r.screen_size();
    let sw = screen.x;
    let sh = screen.y;

    // Layout parameters
    let menu_h = 24.0;
    let toolbar_h = if state.panels.toolbar { 30.0 } else { 0.0 };
    let status_h = if state.panels.status_bar { 22.0 } else { 0.0 };
    let hier_w = if state.panels.hierarchy { state.hierarchy_width } else { 0.0 };
    let insp_w = if state.panels.inspector { state.inspector_width } else { 0.0 };
    let bottom_h = if state.panels.bottom { state.bottom_height } else { 0.0 };
    let top_y = menu_h + toolbar_h;

    // Menu bar background
    r.draw_rect(Rect::new(glam::Vec2::ZERO, glam::Vec2::new(sw, menu_h)), bg_base(), 0.0);
    draw_menu_bar(r, state, sw, menu_h);

    // Toolbar
    if state.panels.toolbar {
        let toolbar_rect = Rect::new(glam::Vec2::new(0.0, menu_h), glam::Vec2::new(sw, menu_h + toolbar_h));
        r.draw_rect(toolbar_rect, bg_panel(), 0.0);
        draw_toolbar(r, state, sw, menu_h, toolbar_h);
    }

    // Hierarchy panel (left)
    if state.panels.hierarchy {
        let hier_rect = Rect::new(
            glam::Vec2::new(0.0, top_y),
            glam::Vec2::new(hier_w, sh - bottom_h - status_h),
        );
        r.draw_rect(hier_rect, bg_panel(), 0.0);
        // Border
        r.draw_rect(Rect::new(glam::Vec2::new(hier_w - 1.0, top_y), glam::Vec2::new(hier_w, sh - bottom_h - status_h)), border_color(), 0.0);

        r.push_clip(hier_rect);
        draw_hierarchy(r, state, hier_w, top_y, sh - bottom_h - status_h);
        r.pop_clip();
    }

    // Inspector panel (right)
    if state.panels.inspector {
        let insp_x = sw - insp_w;
        let insp_rect = Rect::new(
            glam::Vec2::new(insp_x, top_y),
            glam::Vec2::new(sw, sh - bottom_h - status_h),
        );
        r.draw_rect(insp_rect, bg_panel(), 0.0);
        r.draw_rect(Rect::new(glam::Vec2::new(insp_x, top_y), glam::Vec2::new(insp_x + 1.0, sh - bottom_h - status_h)), border_color(), 0.0);

        r.push_clip(insp_rect);
        draw_inspector(r, state, insp_x, top_y, sw, sh - bottom_h - status_h);
        r.pop_clip();
    }

    // Bottom panel
    if state.panels.bottom {
        let bottom_top = sh - bottom_h - status_h;
        let bottom_rect = Rect::new(
            glam::Vec2::new(0.0, bottom_top),
            glam::Vec2::new(sw, sh - status_h),
        );
        r.draw_rect(bottom_rect, bg_panel(), 0.0);
        r.draw_rect(Rect::new(glam::Vec2::new(0.0, bottom_top), glam::Vec2::new(sw, bottom_top + 1.0)), border_color(), 0.0);

        r.push_clip(bottom_rect);
        draw_bottom_panel(r, state, sw, bottom_top, sh - status_h);
        r.pop_clip();
    }

    // Viewport area -- just draw an overlay label
    {
        let vp_x = hier_w;
        let vp_y = top_y;
        let vp_w = sw - hier_w - insp_w;
        let vp_h = sh - top_y - bottom_h - status_h;
        if vp_w > 0.0 && vp_h > 0.0 {
            draw_viewport_overlay(r, state, vp_x, vp_y, vp_w, vp_h);
        }
    }

    // Status bar
    if state.panels.status_bar {
        let sb_rect = Rect::new(glam::Vec2::new(0.0, sh - status_h), glam::Vec2::new(sw, sh));
        r.draw_rect(sb_rect, bg_base(), 0.0);
        r.draw_rect(Rect::new(glam::Vec2::new(0.0, sh - status_h), glam::Vec2::new(sw, sh - status_h + 1.0)), border_color(), 0.0);
        draw_status_bar(r, state, sw, sh, status_h);
    }

    // FPS overlay
    if state.stats_visible {
        draw_fps_overlay(r, state);
    }

    // Notification toast
    draw_notification(r, state, sw, sh);
}

fn draw_menu_bar(r: &mut UIGpuRenderer, state: &EditorState, sw: f32, h: f32) {
    let fs = 14.0;
    let mut x = 8.0;
    let y = (h - fs) * 0.5;

    r.draw_text("Genovo Studio", glam::Vec2::new(x, y), fs, accent());
    x += r.text_width("Genovo Studio", fs) + 12.0;

    r.draw_text("|", glam::Vec2::new(x, y), fs, border_color());
    x += 12.0;

    let menus = ["File", "Edit", "Create", "View", "Window", "Help"];
    for label in &menus {
        r.draw_text(label, glam::Vec2::new(x, y), fs, text_normal());
        x += r.text_width(label, fs) + 14.0;
    }
}

fn draw_toolbar(r: &mut UIGpuRenderer, state: &EditorState, sw: f32, top: f32, h: f32) {
    let fs = 12.0;
    let mut x = 8.0;
    let y = top + (h - fs) * 0.5;

    // Gizmo mode buttons
    for mode in [GizmoMode::Select, GizmoMode::Translate, GizmoMode::Rotate, GizmoMode::Scale] {
        let label = format!("{} {}", mode.icon(), mode.label());
        let color = if state.gizmo_mode == mode { accent_bright() } else { text_dim() };
        if state.gizmo_mode == mode {
            let w = r.text_width(&label, fs) + 8.0;
            r.draw_rect(Rect::new(glam::Vec2::new(x - 4.0, y - 2.0), glam::Vec2::new(x + w - 4.0, y + fs + 2.0)), bg_widget(), 3.0);
        }
        r.draw_text(&label, glam::Vec2::new(x, y), fs, color);
        x += r.text_width(&label, fs) + 14.0;
    }

    x += 10.0;
    r.draw_text("|", glam::Vec2::new(x, y), fs, border_color());
    x += 14.0;

    // Play/Pause/Stop
    let play_label = if state.is_playing && !state.is_paused { "|| Pause" } else { "> Play" };
    let play_color = if state.is_playing && !state.is_paused { green() } else { text_normal() };
    r.draw_text(play_label, glam::Vec2::new(x, y), fs, play_color);
    x += r.text_width(play_label, fs) + 10.0;

    r.draw_text("[] Stop", glam::Vec2::new(x, y), fs, red());
    x += r.text_width("[] Stop", fs) + 10.0;

    r.draw_text("|> Step", glam::Vec2::new(x, y), fs, text_dim());
    x += r.text_width("|> Step", fs) + 14.0;

    r.draw_text("|", glam::Vec2::new(x, y), fs, border_color());
    x += 14.0;

    // Snap/Grid/Stats
    let snap_col = if state.snap_enabled { accent_bright() } else { text_dim() };
    r.draw_text("Snap", glam::Vec2::new(x, y), fs, snap_col);
    x += r.text_width("Snap", fs) + 10.0;

    let grid_col = if state.grid_visible { accent_bright() } else { text_dim() };
    r.draw_text("Grid", glam::Vec2::new(x, y), fs, grid_col);
    x += r.text_width("Grid", fs) + 10.0;

    let stats_col = if state.stats_visible { accent_bright() } else { text_dim() };
    r.draw_text("Stats", glam::Vec2::new(x, y), fs, stats_col);
    x += r.text_width("Stats", fs) + 10.0;

    // Speed indicator on right
    let speed_text = format!("{:.1}x", state.sim_speed);
    let stw = r.text_width(&speed_text, fs);
    r.draw_text(&speed_text, glam::Vec2::new(sw - stw - 10.0, y), fs, text_dim());
}

fn draw_hierarchy(r: &mut UIGpuRenderer, state: &EditorState, width: f32, top: f32, bottom: f32) {
    let fs = 13.0;
    let line_h = 20.0;
    let pad = 6.0;
    let mut y = top + 4.0;

    // Header
    r.draw_text("Scene Hierarchy", glam::Vec2::new(pad, y), fs, text_bright());
    y += line_h;

    // Separator
    r.draw_rect(Rect::new(glam::Vec2::new(pad, y), glam::Vec2::new(width - pad, y + 1.0)), border_color(), 0.0);
    y += 4.0;

    // Entity list
    let fs_item = 12.0;
    for (i, ent) in state.entities.iter().enumerate() {
        if y + line_h > bottom { break; }

        let selected = state.selected_entity == Some(i);

        // Selection highlight
        if selected {
            r.draw_rect(
                Rect::new(glam::Vec2::new(0.0, y), glam::Vec2::new(width, y + line_h)),
                accent_bg(), 0.0,
            );
        }

        // Entity type dot
        let dot_x = pad + 4.0;
        let dot_y = y + (line_h - 6.0) * 0.5;
        r.draw_circle(glam::Vec2::new(dot_x + 3.0, dot_y + 3.0), 3.0, ent.entity_type.icon_color());

        // Entity name
        let label = format!("{} [{}]", ent.name, ent.entity_type.icon_letter());
        let name_color = if selected { text_bright() } else { text_normal() };
        r.draw_text(&label, glam::Vec2::new(pad + 16.0, y + (line_h - fs_item) * 0.5), fs_item, name_color);

        // Position on right side
        let pos_text = format!("({:.1},{:.1},{:.1})", ent.position[0], ent.position[1], ent.position[2]);
        let ptw = r.text_width(&pos_text, 10.0);
        if pad + 16.0 + r.text_width(&label, fs_item) + ptw + 10.0 < width {
            r.draw_text(&pos_text, glam::Vec2::new(width - ptw - pad, y + (line_h - 10.0) * 0.5), 10.0, text_muted());
        }

        y += line_h;
    }

    // Entity count at bottom
    if state.entities.is_empty() {
        r.draw_text("(empty scene)", glam::Vec2::new(pad, y + 4.0), 11.0, text_muted());
    }
}

fn draw_inspector(r: &mut UIGpuRenderer, state: &EditorState, x: f32, top: f32, right: f32, bottom: f32) {
    let fs = 13.0;
    let fs_small = 11.0;
    let line_h = 18.0;
    let pad = x + 8.0;
    let mut y = top + 4.0;

    r.draw_text("Inspector", glam::Vec2::new(pad, y), fs, text_bright());
    y += 20.0;

    r.draw_rect(Rect::new(glam::Vec2::new(x + 6.0, y), glam::Vec2::new(right - 6.0, y + 1.0)), border_color(), 0.0);
    y += 4.0;

    let idx = match state.selected_entity {
        Some(i) if i < state.entities.len() => i,
        _ => {
            r.draw_text("No entity selected", glam::Vec2::new(pad, y + 4.0), 12.0, text_muted());
            return;
        }
    };

    let ent = &state.entities[idx];

    // Entity name
    r.draw_text(&ent.name, glam::Vec2::new(pad, y), fs, accent_bright());
    y += line_h;

    r.draw_text(&format!("Type: {}", ent.entity_type), glam::Vec2::new(pad, y), fs_small, text_dim());
    y += line_h + 4.0;

    // Transform section
    r.draw_rect(Rect::new(glam::Vec2::new(x + 6.0, y), glam::Vec2::new(right - 6.0, y + 1.0)), border_color(), 0.0);
    y += 4.0;
    r.draw_text("+ Transform", glam::Vec2::new(pad, y), fs, text_bright());
    y += line_h;

    // Position
    r.draw_text("Position", glam::Vec2::new(pad, y), fs_small, text_dim());
    y += 14.0;
    let pos_text = format!("X: {:.2}  Y: {:.2}  Z: {:.2}", ent.position[0], ent.position[1], ent.position[2]);
    r.draw_text("X:", glam::Vec2::new(pad, y), fs_small, x_color());
    r.draw_text(&format!("{:.2}", ent.position[0]), glam::Vec2::new(pad + 18.0, y), fs_small, text_normal());
    let rx = pad + 80.0;
    r.draw_text("Y:", glam::Vec2::new(rx, y), fs_small, y_color());
    r.draw_text(&format!("{:.2}", ent.position[1]), glam::Vec2::new(rx + 18.0, y), fs_small, text_normal());
    let rz = pad + 160.0;
    r.draw_text("Z:", glam::Vec2::new(rz, y), fs_small, z_color());
    r.draw_text(&format!("{:.2}", ent.position[2]), glam::Vec2::new(rz + 18.0, y), fs_small, text_normal());
    y += line_h;

    // Rotation
    r.draw_text("Rotation", glam::Vec2::new(pad, y), fs_small, text_dim());
    y += 14.0;
    r.draw_text("X:", glam::Vec2::new(pad, y), fs_small, x_color());
    r.draw_text(&format!("{:.1}d", ent.rotation[0]), glam::Vec2::new(pad + 18.0, y), fs_small, text_normal());
    r.draw_text("Y:", glam::Vec2::new(rx, y), fs_small, y_color());
    r.draw_text(&format!("{:.1}d", ent.rotation[1]), glam::Vec2::new(rx + 18.0, y), fs_small, text_normal());
    r.draw_text("Z:", glam::Vec2::new(rz, y), fs_small, z_color());
    r.draw_text(&format!("{:.1}d", ent.rotation[2]), glam::Vec2::new(rz + 18.0, y), fs_small, text_normal());
    y += line_h;

    // Scale
    r.draw_text("Scale", glam::Vec2::new(pad, y), fs_small, text_dim());
    y += 14.0;
    r.draw_text("X:", glam::Vec2::new(pad, y), fs_small, x_color());
    r.draw_text(&format!("{:.2}", ent.scale[0]), glam::Vec2::new(pad + 18.0, y), fs_small, text_normal());
    r.draw_text("Y:", glam::Vec2::new(rx, y), fs_small, y_color());
    r.draw_text(&format!("{:.2}", ent.scale[1]), glam::Vec2::new(rx + 18.0, y), fs_small, text_normal());
    r.draw_text("Z:", glam::Vec2::new(rz, y), fs_small, z_color());
    r.draw_text(&format!("{:.2}", ent.scale[2]), glam::Vec2::new(rz + 18.0, y), fs_small, text_normal());
    y += line_h + 6.0;

    // Component sections based on entity type
    if ent.entity_type == EntityType::Mesh {
        r.draw_rect(Rect::new(glam::Vec2::new(x + 6.0, y), glam::Vec2::new(right - 6.0, y + 1.0)), border_color(), 0.0);
        y += 4.0;
        r.draw_text("M Mesh Renderer", glam::Vec2::new(pad, y), fs, cyan());
        y += line_h;
        r.draw_text(&format!("Shape: {}", ent.mesh_shape), glam::Vec2::new(pad, y), fs_small, text_normal());
        y += line_h;
        r.draw_text(&format!("Shadows: {} / {}", if ent.cast_shadows {"cast"} else {"no cast"}, if ent.receive_shadows {"recv"} else {"no recv"}), glam::Vec2::new(pad, y), fs_small, text_dim());
        y += line_h + 4.0;
    }

    if ent.has_physics {
        r.draw_rect(Rect::new(glam::Vec2::new(x + 6.0, y), glam::Vec2::new(right - 6.0, y + 1.0)), border_color(), 0.0);
        y += 4.0;
        r.draw_text("P Rigid Body", glam::Vec2::new(pad, y), fs, green());
        y += line_h;
        r.draw_text(&format!("Type: {}  Mass: {:.1}", ent.body_kind, ent.mass), glam::Vec2::new(pad, y), fs_small, text_normal());
        y += line_h;
        r.draw_text(&format!("Friction: {:.2}  Restitution: {:.2}", ent.friction, ent.restitution), glam::Vec2::new(pad, y), fs_small, text_dim());
        y += line_h;
        r.draw_text(&format!("Damping: lin={:.2} ang={:.2}", ent.linear_damping, ent.angular_damping), glam::Vec2::new(pad, y), fs_small, text_dim());
        y += line_h + 4.0;
    }

    if ent.is_light {
        r.draw_rect(Rect::new(glam::Vec2::new(x + 6.0, y), glam::Vec2::new(right - 6.0, y + 1.0)), border_color(), 0.0);
        y += 4.0;
        r.draw_text("L Light", glam::Vec2::new(pad, y), fs, yellow());
        y += line_h;
        r.draw_text(&format!("{} | Int: {:.1} | Range: {:.1}", ent.light_kind, ent.light_intensity, ent.light_range), glam::Vec2::new(pad, y), fs_small, text_normal());
        y += line_h;
        r.draw_text(&format!("Color: ({:.2},{:.2},{:.2})", ent.light_color[0], ent.light_color[1], ent.light_color[2]), glam::Vec2::new(pad, y), fs_small, text_dim());
        y += line_h + 4.0;
    }

    if ent.is_camera {
        r.draw_rect(Rect::new(glam::Vec2::new(x + 6.0, y), glam::Vec2::new(right - 6.0, y + 1.0)), border_color(), 0.0);
        y += 4.0;
        r.draw_text("C Camera", glam::Vec2::new(pad, y), fs, accent_bright());
        y += line_h;
        r.draw_text(&format!("{} | FOV: {:.0} | Near: {:.2} Far: {:.0}", ent.camera_projection, ent.camera_fov, ent.camera_near, ent.camera_far), glam::Vec2::new(pad, y), fs_small, text_normal());
        y += line_h + 4.0;
    }

    if ent.is_audio {
        r.draw_rect(Rect::new(glam::Vec2::new(x + 6.0, y), glam::Vec2::new(right - 6.0, y + 1.0)), border_color(), 0.0);
        y += 4.0;
        r.draw_text("A Audio Source", glam::Vec2::new(pad, y), fs, orange());
        y += line_h;
        r.draw_text(&format!("Vol: {:.2} | Pitch: {:.2} | Spatial: {}", ent.audio_volume, ent.audio_pitch, ent.audio_spatial), glam::Vec2::new(pad, y), fs_small, text_normal());
        y += line_h + 4.0;
    }
}

fn draw_bottom_panel(r: &mut UIGpuRenderer, state: &EditorState, sw: f32, top: f32, bottom: f32) {
    let fs = 12.0;
    let tab_h = 22.0;
    let pad = 8.0;

    // Tab bar
    let tabs = ["Console", "Assets", "Profiler", "Animation"];
    let tab_indices = [BottomTab::Console, BottomTab::ContentBrowser, BottomTab::Profiler, BottomTab::Animation];
    let mut tx = pad;
    for (i, tab) in tabs.iter().enumerate() {
        let active = state.bottom_tab == tab_indices[i];
        let col = if active { accent_bright() } else { text_dim() };
        if active {
            let tw = r.text_width(tab, fs) + 12.0;
            r.draw_rect(Rect::new(glam::Vec2::new(tx - 4.0, top + 2.0), glam::Vec2::new(tx + tw - 4.0, top + tab_h)), bg_widget(), 3.0);
        }
        r.draw_text(tab, glam::Vec2::new(tx, top + (tab_h - fs) * 0.5), fs, col);
        tx += r.text_width(tab, fs) + 20.0;
    }

    // Content area
    let content_top = top + tab_h + 2.0;
    r.draw_rect(Rect::new(glam::Vec2::new(0.0, content_top - 1.0), glam::Vec2::new(sw, content_top)), border_color(), 0.0);

    match state.bottom_tab {
        BottomTab::Console => draw_console(r, state, sw, content_top, bottom),
        BottomTab::ContentBrowser => {
            r.draw_text("Content Browser", glam::Vec2::new(pad, content_top + 4.0), fs, text_dim());
            r.draw_text(&format!("Path: {}", state.asset_path), glam::Vec2::new(pad, content_top + 20.0), 11.0, text_muted());
        }
        BottomTab::Profiler => draw_profiler(r, state, sw, content_top, bottom),
        BottomTab::Animation => {
            r.draw_text("Animation Timeline", glam::Vec2::new(pad, content_top + 4.0), fs, text_dim());
            r.draw_text(&format!("Time: {:.2}s / {:.1}s", state.anim_time, state.anim_duration), glam::Vec2::new(pad, content_top + 20.0), 11.0, text_muted());
        }
    }
}

fn draw_console(r: &mut UIGpuRenderer, state: &EditorState, sw: f32, top: f32, bottom: f32) {
    let fs = 11.0;
    let line_h = 15.0;
    let pad = 8.0;
    let max_lines = ((bottom - top - 20.0) / line_h) as usize;

    let filtered: Vec<&LogEntry> = state.console_log.iter()
        .filter(|e| state.console_filter_level.map_or(true, |f| e.level == f))
        .collect();

    let start = if filtered.len() > max_lines { filtered.len() - max_lines } else { 0 };
    let mut y = top + 4.0;

    for entry in &filtered[start..] {
        if y + line_h > bottom - 16.0 { break; }

        let prefix = format!("[{:.1}] {} ", entry.timestamp, entry.level.prefix());
        r.draw_text(&prefix, glam::Vec2::new(pad, y), fs, text_muted());
        let prefix_w = r.text_width(&prefix, fs);

        let msg = if entry.count > 1 {
            format!("{} (x{})", entry.text, entry.count)
        } else {
            entry.text.clone()
        };
        r.draw_text(&msg, glam::Vec2::new(pad + prefix_w, y), fs, entry.level.color());
        y += line_h;
    }

    // Input prompt at bottom
    let input_y = bottom - 16.0;
    r.draw_rect(Rect::new(glam::Vec2::new(0.0, input_y - 1.0), glam::Vec2::new(sw, input_y)), border_color(), 0.0);
    r.draw_text("> ", glam::Vec2::new(pad, input_y + 1.0), fs, accent());
    r.draw_text(&state.console_input, glam::Vec2::new(pad + 14.0, input_y + 1.0), fs, text_normal());
    // Blinking cursor
    if state.frame_count % 60 < 30 {
        let cursor_x = pad + 14.0 + r.text_width(&state.console_input, fs);
        r.draw_text("_", glam::Vec2::new(cursor_x, input_y + 1.0), fs, text_bright());
    }
}

fn draw_profiler(r: &mut UIGpuRenderer, state: &EditorState, sw: f32, top: f32, bottom: f32) {
    let fs = 11.0;
    let pad = 8.0;
    let mut y = top + 4.0;

    r.draw_text(&format!("FPS: {:.0}  Frame: {:.2}ms", state.smooth_fps, state.smooth_frame_time), glam::Vec2::new(pad, y), fs, text_bright());
    y += 16.0;

    let avg_ft = if !state.frame_times.is_empty() {
        state.frame_times.iter().sum::<f64>() / state.frame_times.len() as f64
    } else { 16.67 };
    let max_ft = state.frame_times.iter().copied().fold(0.0f64, f64::max);
    let min_ft = state.frame_times.iter().copied().fold(f64::MAX, f64::min);

    r.draw_text(&format!("Avg: {:.2}ms  Min: {:.2}ms  Max: {:.2}ms", avg_ft, min_ft, max_ft), glam::Vec2::new(pad, y), fs, text_dim());
    y += 16.0;

    // Draw frame time bar graph
    let graph_h = (bottom - y - 8.0).max(20.0);
    let graph_w = sw - pad * 2.0;
    let bar_count = state.frame_times.len();
    if bar_count > 0 {
        let bar_w = (graph_w / bar_count as f32).max(1.0);
        for (i, &ft) in state.frame_times.iter().enumerate() {
            let bar_h = ((ft / 33.33) * graph_h as f64).min(graph_h as f64) as f32;
            let bx = pad + i as f32 * bar_w;
            let by = y + graph_h - bar_h;
            let bar_color = if ft < 16.67 { green() } else if ft < 33.33 { yellow() } else { red() };
            r.draw_rect(Rect::new(glam::Vec2::new(bx, by), glam::Vec2::new(bx + (bar_w - 0.5).max(0.5), y + graph_h)), bar_color, 0.0);
        }
    }

    // 60fps / 30fps lines
    let y_60 = y + graph_h * (1.0 - (16.67 / 33.33) as f32);
    let y_30 = y + graph_h * (1.0 - 1.0);
    r.draw_line(glam::Vec2::new(pad, y_60), glam::Vec2::new(sw - pad, y_60), ca(72, 199, 142, 100), 1.0);
    r.draw_line(glam::Vec2::new(pad, y_30), glam::Vec2::new(sw - pad, y_30), ca(245, 196, 80, 100), 1.0);
}

fn draw_viewport_overlay(r: &mut UIGpuRenderer, state: &EditorState, vx: f32, vy: f32, vw: f32, vh: f32) {
    let fs = 11.0;

    // Camera info (top-left of viewport)
    let cam_text = format!("Yaw: {:.0}  Pitch: {:.0}  Dist: {:.1}", state.camera_yaw, state.camera_pitch, state.camera_dist);
    r.draw_text(&cam_text, glam::Vec2::new(vx + 8.0, vy + 6.0), fs, text_dim());

    // Gizmo mode (top-right of viewport)
    let mode_text = format!("{} | {}", state.gizmo_mode.label(), if state.coord_space == CoordSpace::World { "World" } else { "Local" });
    let mtw = r.text_width(&mode_text, fs);
    r.draw_text(&mode_text, glam::Vec2::new(vx + vw - mtw - 8.0, vy + 6.0), fs, text_dim());

    // Play state indicator (bottom-center)
    if state.is_playing {
        let play_text = if state.is_paused { "PAUSED" } else { "PLAYING" };
        let play_col = if state.is_paused { yellow() } else { green() };
        let pw = r.text_width(play_text, 14.0);
        r.draw_text(play_text, glam::Vec2::new(vx + (vw - pw) * 0.5, vy + vh - 24.0), 14.0, play_col);
    }

    // Axis gizmo (bottom-left)
    let ax_x = vx + 30.0;
    let ax_y = vy + vh - 30.0;
    let axis_len = 22.0;
    r.draw_line(glam::Vec2::new(ax_x, ax_y), glam::Vec2::new(ax_x + axis_len, ax_y), x_color(), 2.0);
    r.draw_text("X", glam::Vec2::new(ax_x + axis_len + 3.0, ax_y - 5.0), 10.0, x_color());
    r.draw_line(glam::Vec2::new(ax_x, ax_y), glam::Vec2::new(ax_x, ax_y - axis_len), y_color(), 2.0);
    r.draw_text("Y", glam::Vec2::new(ax_x - 5.0, ax_y - axis_len - 12.0), 10.0, y_color());
    r.draw_line(glam::Vec2::new(ax_x, ax_y), glam::Vec2::new(ax_x - axis_len * 0.6, ax_y + axis_len * 0.5), z_color(), 2.0);
    r.draw_text("Z", glam::Vec2::new(ax_x - axis_len * 0.6 - 12.0, ax_y + axis_len * 0.5 - 2.0), 10.0, z_color());

    // Selected entity info
    if let Some(idx) = state.selected_entity {
        if idx < state.entities.len() {
            let ent = &state.entities[idx];
            let sel_text = format!("[{}] {} ({:.1},{:.1},{:.1})", ent.entity_type.icon_letter(), ent.name, ent.position[0], ent.position[1], ent.position[2]);
            r.draw_text(&sel_text, glam::Vec2::new(vx + 8.0, vy + 20.0), fs, text_dim());
        }
    }

    // Entity count pills at top
    let entity_labels: Vec<(String, Color)> = vec![
        (format!("{} entities", state.entities.len()), text_dim()),
    ];
    let mut pill_x = vx + vw - 8.0;
    for (label, col) in entity_labels.iter().rev() {
        let tw = r.text_width(label, 10.0);
        let pw = tw + 12.0;
        pill_x -= pw + 4.0;
        r.draw_rect(Rect::new(glam::Vec2::new(pill_x, vy + 20.0), glam::Vec2::new(pill_x + pw, vy + 35.0)), ca(18, 18, 22, 200), 7.0);
        r.draw_text(label, glam::Vec2::new(pill_x + 6.0, vy + 22.0), 10.0, *col);
    }
}

fn draw_status_bar(r: &mut UIGpuRenderer, state: &EditorState, sw: f32, sh: f32, h: f32) {
    let fs = 11.0;
    let y = sh - h + (h - fs) * 0.5;
    let pad = 8.0;

    // Play state dot
    let dot_color = if state.is_playing && !state.is_paused { green() }
                    else if state.is_paused { yellow() }
                    else { text_muted() };
    r.draw_circle(glam::Vec2::new(pad + 4.0, sh - h * 0.5), 4.0, dot_color);

    let status_text = if state.is_playing {
        if state.is_paused { "Paused" } else { "Playing" }
    } else { "Stopped" };
    r.draw_text(status_text, glam::Vec2::new(pad + 14.0, y), fs, text_dim());

    // Scene name
    r.draw_text(&format!(" | {}", state.scene_name), glam::Vec2::new(pad + 14.0 + r.text_width(status_text, fs), y), fs, text_muted());

    // Right side: FPS, entity count, body count
    let ecs_count = state.engine.world().entity_count();
    let body_count = state.engine.physics().body_count();
    let active = state.engine.physics().active_body_count();

    let right_text = format!("{:.0} FPS | {} entities | {} bodies ({} active)", state.smooth_fps, state.entities.len(), body_count, active);
    let rtw = r.text_width(&right_text, fs);
    r.draw_text(&right_text, glam::Vec2::new(sw - rtw - pad, y), fs, text_dim());
}

fn draw_fps_overlay(r: &mut UIGpuRenderer, state: &EditorState) {
    let fs = 11.0;
    let x = 6.0;
    let y = 58.0;
    let w = 140.0;
    let h = 36.0;

    r.draw_rect(Rect::new(glam::Vec2::new(x, y), glam::Vec2::new(x + w, y + h)), ca(18, 18, 22, 180), 4.0);
    r.draw_text(&format!("FPS: {:.0}", state.smooth_fps), glam::Vec2::new(x + 6.0, y + 4.0), fs, green());
    r.draw_text(&format!("Frame: {:.2}ms", state.smooth_frame_time), glam::Vec2::new(x + 6.0, y + 18.0), fs, text_dim());
}

fn draw_notification(r: &mut UIGpuRenderer, state: &EditorState, sw: f32, sh: f32) {
    if let Some((ref msg, level, ref start)) = state.notification {
        let age = start.elapsed().as_secs_f32();
        if age > 3.5 { return; }

        let alpha = if age > 2.5 { ((3.5 - age) / 1.0).clamp(0.0, 1.0) } else { 1.0 };
        let a = (alpha * 255.0) as u8;

        let tw = r.text_width(msg, 13.0);
        let w = tw + 24.0;
        let h = 32.0;
        let x = sw - w - 16.0;
        let y = sh - 52.0;

        r.draw_rect(Rect::new(glam::Vec2::new(x, y), glam::Vec2::new(x + w, y + h)), ca(24, 24, 28, a), 4.0);
        r.draw_rect_outline(Rect::new(glam::Vec2::new(x, y), glam::Vec2::new(x + w, y + h)), ca(38, 38, 44, a), 1.0, 4.0);

        let col = level.color();
        let text_col = Color::new(col.r, col.g, col.b, alpha);
        r.draw_text(msg, glam::Vec2::new(x + 12.0, y + (h - 13.0) * 0.5), 13.0, text_col);
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

    // Custom Slate UI renderer (replaces egui)
    ui_renderer: UIGpuRenderer,

    // Editor state
    editor: EditorState,
}

impl ApplicationHandler for EditorApp {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gpu.is_some() { return; }

        let w = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("Genovo Studio \u{2014} Untitled Scene")
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

        // UIGpuRenderer (replaces egui)
        let ui_renderer = UIGpuRenderer::new(dev.clone(), que.clone(), cfg.format);

        // Editor state
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
        editor.log(LogLevel::System, "UI: Custom Slate (UIGpuRenderer + bitmap font)");
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
        println!("[Genovo Studio] UI: Custom Slate (UIGpuRenderer + bitmap font)");
        println!("[Genovo Studio] Editor ready. {} entities.", editor.entities.len());

        self.gpu = Some(GpuState {
            window: w,
            device: dev,
            queue: que,
            surface: surf,
            config: cfg,
            depth_view: dv,
            scene_manager,
            ui_renderer,
            editor,
        });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _wid: WindowId, ev: WindowEvent) {
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
            }

            WindowEvent::KeyboardInput { event, .. } => {
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
                            KeyCode::Enter => { submit_console(&mut s.editor); }
                            KeyCode::Escape => {
                                s.editor.selected_entity = None;
                                s.editor.renaming_entity = None;
                                s.editor.show_about = false;
                                s.editor.show_shortcuts = false;
                            }
                            KeyCode::Backspace => {
                                s.editor.console_input.pop();
                            }
                            _ => {
                                // Basic text input for console
                                if let Some(text) = &event.text {
                                    let t = text.as_str();
                                    if !ctrl && t.len() == 1 {
                                        let ch = t.chars().next().unwrap();
                                        if ch >= ' ' && ch <= '~' {
                                            s.editor.console_input.push(ch);
                                        }
                                    }
                                }
                            }
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

                // --- UIGpuRenderer begin frame ---
                let screen_size = glam::Vec2::new(s.config.width as f32, s.config.height as f32);
                s.ui_renderer.begin_frame(screen_size);

                // Draw all editor UI
                draw_editor_ui(&mut s.ui_renderer, &s.editor);

                // --- Build camera ---
                let yaw_rad = s.editor.camera_yaw.to_radians();
                let pitch_rad = s.editor.camera_pitch.to_radians();
                let target = glam::Vec3::new(s.editor.camera_target[0], s.editor.camera_target[1], s.editor.camera_target[2]);
                let aspect = s.config.width as f32 / s.config.height.max(1) as f32;
                let mut scene_camera = SceneCamera::perspective(
                    glam::Vec3::ZERO, target, glam::Vec3::Y,
                    s.editor.camera_fov.to_radians(), aspect, 0.1, 1000.0,
                );
                scene_camera.orbit(target, yaw_rad, pitch_rad, s.editor.camera_dist);

                // --- Build scene lights ---
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

                // --- Submit scene entities ---
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

                // --- GPU Render ---
                let Ok(out) = s.surface.get_current_texture() else {
                    s.window.request_redraw();
                    return;
                };
                let view = out.texture.create_view(&Default::default());

                // Pass 1: 3D scene
                let scene_cmd = s.scene_manager.render(&s.device, &s.queue, &view, &s.depth_view, &scene_camera, &scene_lights);

                // Pass 2: UI overlay via UIGpuRenderer
                let mut ui_enc = s.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ui_render") });
                s.ui_renderer.end_frame(&mut ui_enc, &view);

                // Submit: 3D first, then UI
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
    println!("[Genovo Studio] UI: Custom Slate (UIGpuRenderer + bitmap font)");

    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = EditorApp {
        gpu: None,
        audio_running: Arc::new(AtomicBool::new(false)),
    };
    let _ = el.run_app(&mut app);
}
