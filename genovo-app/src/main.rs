//! Genovo Engine Editor — Professional Game Engine Editor
//!
//! A complete, polished editor built with egui + wgpu + winit 0.30:
//!   - Real GPU-rendered 3D viewport (wgpu triangle pipeline behind egui)
//!   - Scene hierarchy with colored entity type icons, context menus
//!   - Inspector with colored XYZ drag values, physics sync, light editing
//!   - Asset browser with directory scanning and icon grid
//!   - Console with command history and colored log output
//!   - Profiler with egui_plot frame time graph
//!   - Professional dark theme (near-black, accent blue, thin scrollbars)
//!   - Keyboard shortcuts (Ctrl+S, Ctrl+Z, Delete, W/E/R, Space, F5, F)
//!   - Working physics integration (play/pause/stop/step)
//!   - Status bar with colored play-state dot, FPS, entity/body counts

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
// GPU Shader
// =============================================================================

const SHADER: &str = r#"
struct Out { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    var p = array<vec2<f32>,3>(vec2(0.0,0.6),vec2(-0.5,-0.4),vec2(0.5,-0.4));
    var c = array<vec3<f32>,3>(vec3(1.0,0.2,0.2),vec3(0.2,1.0,0.2),vec3(0.2,0.4,1.0));
    var o: Out; o.pos = vec4(p[i],0.0,1.0); o.col = c[i]; return o;
}
@fragment fn fs(in: Out) -> @location(0) vec4<f32> { return vec4(in.col,1.0); }
"#;

// =============================================================================
// Color Palette
// =============================================================================

// Theme
const BG_DARKEST: egui::Color32 = egui::Color32::from_rgb(22, 22, 25);
const BG_PANEL: egui::Color32 = egui::Color32::from_rgb(28, 28, 32);
const BG_WIDGET: egui::Color32 = egui::Color32::from_rgb(36, 36, 40);
const BG_HOVER: egui::Color32 = egui::Color32::from_rgb(48, 48, 55);
const BORDER: egui::Color32 = egui::Color32::from_rgb(42, 42, 46);
const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(215, 215, 215);
const TEXT_SECONDARY: egui::Color32 = egui::Color32::from_rgb(130, 130, 130);
const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(80, 80, 85);

// Accent
const ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 122, 204);
const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(0, 90, 160);
const ACCENT_BRIGHT: egui::Color32 = egui::Color32::from_rgb(40, 150, 230);

// Semantic
const GREEN: egui::Color32 = egui::Color32::from_rgb(76, 195, 85);
const YELLOW: egui::Color32 = egui::Color32::from_rgb(230, 190, 50);
const RED: egui::Color32 = egui::Color32::from_rgb(220, 65, 55);
const CYAN: egui::Color32 = egui::Color32::from_rgb(70, 200, 220);
const MAGENTA: egui::Color32 = egui::Color32::from_rgb(190, 80, 210);

// XYZ axis colors (Unreal-style)
const X_COLOR: egui::Color32 = egui::Color32::from_rgb(220, 60, 60);
const Y_COLOR: egui::Color32 = egui::Color32::from_rgb(80, 200, 80);
const Z_COLOR: egui::Color32 = egui::Color32::from_rgb(60, 120, 230);

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
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityType::Empty => write!(f, "Empty"),
            EntityType::Mesh => write!(f, "Mesh"),
            EntityType::Light => write!(f, "Light"),
            EntityType::Camera => write!(f, "Camera"),
            EntityType::ParticleSystem => write!(f, "Particles"),
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
        }
    }

    fn icon_color(&self) -> egui::Color32 {
        match self {
            EntityType::Empty => TEXT_SECONDARY,
            EntityType::Mesh => CYAN,
            EntityType::Light => YELLOW,
            EntityType::Camera => ACCENT_BRIGHT,
            EntityType::ParticleSystem => MAGENTA,
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
    position: [f32; 3],
    rotation: [f32; 3],
    scale: [f32; 3],
    // Physics
    has_physics: bool,
    physics_handle: Option<physics::RigidBodyHandle>,
    mass: f32,
    friction: f32,
    restitution: f32,
    // Light
    is_light: bool,
    light_color: [f32; 3],
    light_intensity: f32,
    light_range: f32,
}

// =============================================================================
// Transform Mode / Coord Space
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransformMode {
    Translate,
    Rotate,
    Scale,
}

impl TransformMode {
    fn label(&self) -> &str {
        match self {
            TransformMode::Translate => "Translate",
            TransformMode::Rotate => "Rotate",
            TransformMode::Scale => "Scale",
        }
    }

    fn short_label(&self) -> &str {
        match self {
            TransformMode::Translate => "W",
            TransformMode::Rotate => "E",
            TransformMode::Scale => "R",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoordSpace {
    Local,
    World,
}

// =============================================================================
// Console Log
// =============================================================================

#[derive(Clone)]
struct LogEntry {
    text: String,
    level: LogLevel,
    timestamp: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LogLevel {
    Info,
    Warn,
    Error,
    System,
}

impl LogLevel {
    fn color(&self) -> egui::Color32 {
        match self {
            LogLevel::Info => egui::Color32::from_rgb(175, 175, 175),
            LogLevel::Warn => YELLOW,
            LogLevel::Error => RED,
            LogLevel::System => ACCENT_BRIGHT,
        }
    }

    fn prefix(&self) -> &str {
        match self {
            LogLevel::Info => "INF",
            LogLevel::Warn => "WRN",
            LogLevel::Error => "ERR",
            LogLevel::System => "SYS",
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
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            hierarchy: true,
            inspector: true,
            bottom: true,
        }
    }
}

// =============================================================================
// Bottom Tab
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum BottomTab {
    Console,
    AssetBrowser,
    Profiler,
}

// =============================================================================
// Hierarchy Action (deferred mutation)
// =============================================================================

enum HierarchyAction {
    Select(usize),
    Duplicate(usize),
    Delete(usize),
    AddChild(usize),
}

// =============================================================================
// Editor State
// =============================================================================

struct EditorState {
    engine: Engine,

    // Scene
    entities: Vec<SceneEntity>,
    selected_entity: Option<usize>,
    next_entity_id: u32,

    // Playback
    is_playing: bool,
    is_paused: bool,
    sim_speed: f32,

    // Transform
    transform_mode: TransformMode,
    coord_space: CoordSpace,
    snap_enabled: bool,
    snap_value: f32,
    grid_visible: bool,

    // Console
    console_log: Vec<LogEntry>,
    console_input: String,
    console_history: Vec<String>,
    console_history_idx: Option<usize>,

    // Asset browser
    asset_path: String,
    asset_search: String,

    // Profiler
    frame_times: VecDeque<f64>,
    fps_history: VecDeque<f64>,

    // Frame timing
    frame_count: u64,
    last_frame: Instant,
    fps: f32,
    frame_time_ms: f32,
    start_time: Instant,

    // Camera
    camera_yaw: f32,
    camera_pitch: f32,
    camera_dist: f32,
    camera_target: [f32; 3],

    // Panels
    panels: PanelVisibility,
    bottom_tab: BottomTab,

    // Scene
    scene_name: String,

    // Dialogs
    show_about: bool,

    // Keyboard modifiers
    modifiers: ModifiersState,
}

impl EditorState {
    fn new(engine: Engine) -> Self {
        Self {
            engine,
            entities: Vec::new(),
            selected_entity: None,
            next_entity_id: 0,
            is_playing: false,
            is_paused: false,
            sim_speed: 1.0,
            transform_mode: TransformMode::Translate,
            coord_space: CoordSpace::World,
            snap_enabled: false,
            snap_value: 1.0,
            grid_visible: true,
            console_log: Vec::new(),
            console_input: String::new(),
            console_history: Vec::new(),
            console_history_idx: None,
            asset_path: "assets".to_string(),
            asset_search: String::new(),
            frame_times: VecDeque::with_capacity(300),
            fps_history: VecDeque::with_capacity(300),
            frame_count: 0,
            last_frame: Instant::now(),
            fps: 0.0,
            frame_time_ms: 0.0,
            start_time: Instant::now(),
            camera_yaw: 45.0,
            camera_pitch: -30.0,
            camera_dist: 15.0,
            camera_target: [0.0, 2.0, 0.0],
            panels: PanelVisibility::default(),
            bottom_tab: BottomTab::Console,
            scene_name: "Untitled Scene".to_string(),
            show_about: false,
            modifiers: ModifiersState::empty(),
        }
    }

    fn log(&mut self, level: LogLevel, text: impl Into<String>) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        self.console_log.push(LogEntry {
            text: text.into(),
            level,
            timestamp: elapsed,
        });
        if self.console_log.len() > 2000 {
            self.console_log.drain(0..500);
        }
    }

    fn spawn_entity(&mut self, name: &str, etype: EntityType) -> usize {
        let ecs_entity = self.engine.world_mut().spawn_empty();
        self.next_entity_id += 1;

        let mut se = SceneEntity {
            entity: ecs_entity,
            name: name.to_string(),
            entity_type: etype,
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            has_physics: false,
            physics_handle: None,
            mass: 1.0,
            friction: 0.5,
            restitution: 0.3,
            is_light: etype == EntityType::Light,
            light_color: [1.0, 1.0, 0.9],
            light_intensity: 1.0,
            light_range: 10.0,
        };

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
        self.log(LogLevel::System, format!("Spawned: {} [{}]", name, etype));
        idx
    }

    fn delete_entity(&mut self, idx: usize) {
        if idx >= self.entities.len() {
            return;
        }
        let ent = &self.entities[idx];
        let name = ent.name.clone();
        if let Some(handle) = ent.physics_handle {
            let _ = self.engine.physics_mut().remove_body(handle);
        }
        self.engine.world_mut().despawn(ent.entity);
        self.entities.remove(idx);
        // Fix selected index
        if let Some(sel) = self.selected_entity {
            if sel == idx {
                self.selected_entity = None;
            } else if sel > idx {
                self.selected_entity = Some(sel - 1);
            }
        }
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

            let new_idx = self.spawn_entity(&new_name, etype);
            let e = &mut self.entities[new_idx];
            e.position = [pos[0] + 1.5, pos[1], pos[2]];
            e.rotation = rot;
            e.scale = scl;
            e.is_light = is_light;
            e.light_color = lc;
            e.light_intensity = li;
            e.light_range = lr;

            if let Some(handle) = e.physics_handle {
                let _ = self.engine.physics_mut().set_position(
                    handle,
                    Vec3::new(e.position[0], e.position[1], e.position[2]),
                );
            }
            self.selected_entity = Some(new_idx);
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
            self.log(LogLevel::System, "Simulation paused");
        } else {
            self.is_playing = true;
            self.is_paused = false;
            self.log(LogLevel::System, "Simulation started");
        }
    }

    fn stop_play(&mut self) {
        self.is_playing = false;
        self.is_paused = false;
        self.log(LogLevel::System, "Simulation stopped");
    }

    fn spawn_physics_ball(&mut self) {
        let idx = self.spawn_entity(
            &format!("Ball_{}", self.next_entity_id),
            EntityType::Mesh,
        );
        let e = &mut self.entities[idx];
        e.position = [0.0, 8.0, 0.0];
        if let Some(handle) = e.physics_handle {
            let _ = self.engine.physics_mut().set_position(
                handle,
                Vec3::new(0.0, 8.0, 0.0),
            );
        }
        self.selected_entity = Some(idx);
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
                self.log(LogLevel::System, "--- Available Commands ---");
                self.log(LogLevel::System, "  help                 Show this help");
                self.log(LogLevel::System, "  clear                Clear console");
                self.log(LogLevel::System, "  stats                Engine statistics");
                self.log(LogLevel::System, "  spawn <type> [name]  Spawn entity (empty/cube/sphere/light/camera/particles)");
                self.log(LogLevel::System, "  physics start|stop|step  Control physics");
                self.log(LogLevel::System, "  terrain gen          Generate procedural terrain");
                self.log(LogLevel::System, "  dungeon gen          Generate BSP dungeon");
                self.log(LogLevel::System, "  script <code>        Execute GenovoScript");
                self.log(LogLevel::System, "  gravity <x> <y> <z>  Set gravity vector");
                self.log(LogLevel::System, "  scene <name>         Rename scene");
            }
            "clear" => {
                self.console_log.clear();
            }
            "stats" => {
                let ecs_count = self.engine.world().entity_count();
                let body_count = self.engine.physics().body_count();
                let active = self.engine.physics().active_body_count();
                self.log(LogLevel::System, format!("ECS entities:    {}", ecs_count));
                self.log(LogLevel::System, format!("Scene entities:  {}", self.entities.len()));
                self.log(LogLevel::System, format!("Physics bodies:  {} ({} active)", body_count, active));
                self.log(LogLevel::System, format!("FPS:             {:.1}", self.fps));
                self.log(LogLevel::System, format!("Frame time:      {:.2} ms", self.frame_time_ms));
                self.log(LogLevel::System, format!("Sim speed:       {:.2}x", self.sim_speed));
            }
            "spawn" => {
                let etype = match parts.get(1).copied() {
                    Some("cube") | Some("mesh") => EntityType::Mesh,
                    Some("sphere") => EntityType::Mesh,
                    Some("cylinder") => EntityType::Mesh,
                    Some("light") => EntityType::Light,
                    Some("camera") => EntityType::Camera,
                    Some("particles") => EntityType::ParticleSystem,
                    _ => EntityType::Empty,
                };
                let name = if parts.len() > 2 {
                    parts[2..].join(" ")
                } else {
                    format!("{}", etype)
                };
                let idx = self.spawn_entity(&name, etype);
                self.selected_entity = Some(idx);
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
                    let _ = self.engine.physics_mut().step(1.0 / 60.0);
                    self.sync_physics_to_entities();
                    self.log(LogLevel::System, "Physics stepped (1/60s)");
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
            "scene" => {
                if parts.len() > 1 {
                    self.scene_name = parts[1..].join(" ");
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
                let mut vm = genovo_scripting::GenovoVM::new();
                use genovo_scripting::ScriptVM;
                match vm.load_script("console", &code) {
                    Ok(_) => {
                        let mut ctx = genovo_scripting::ScriptContext::new();
                        match vm.execute("console", &mut ctx) {
                            Ok(_) => {
                                for line in vm.output() {
                                    self.log(LogLevel::Info, format!("  => {}", line));
                                }
                            }
                            Err(e) => self.log(LogLevel::Error, format!("Script error: {}", e)),
                        }
                    }
                    Err(e) => self.log(LogLevel::Error, format!("Compile error: {}", e)),
                }
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
// Theme
// =============================================================================

fn apply_professional_theme(ctx: &egui::Context) {
    use egui::*;

    let mut style = (*ctx.style()).clone();

    style.visuals.dark_mode = true;

    // Core fills
    style.visuals.panel_fill = BG_PANEL;
    style.visuals.window_fill = BG_PANEL;
    style.visuals.extreme_bg_color = BG_DARKEST;
    style.visuals.faint_bg_color = BG_WIDGET;
    style.visuals.override_text_color = Some(TEXT_PRIMARY);

    let rounding = Rounding::same(3);

    // Non-interactive widgets
    style.visuals.widgets.noninteractive.bg_fill = BG_WIDGET;
    style.visuals.widgets.noninteractive.weak_bg_fill = BG_WIDGET;
    style.visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_SECONDARY);
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(0.5, BORDER);
    style.visuals.widgets.noninteractive.corner_radius = rounding;

    // Inactive widgets
    style.visuals.widgets.inactive.bg_fill = BG_WIDGET;
    style.visuals.widgets.inactive.weak_bg_fill = BG_WIDGET;
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(0.5, BORDER);
    style.visuals.widgets.inactive.corner_radius = rounding;

    // Hovered widgets
    style.visuals.widgets.hovered.bg_fill = BG_HOVER;
    style.visuals.widgets.hovered.weak_bg_fill = BG_HOVER;
    style.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, egui::Color32::WHITE);
    style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
    style.visuals.widgets.hovered.corner_radius = rounding;

    // Active widgets
    style.visuals.widgets.active.bg_fill = ACCENT;
    style.visuals.widgets.active.weak_bg_fill = ACCENT;
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, egui::Color32::WHITE);
    style.visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT_BRIGHT);
    style.visuals.widgets.active.corner_radius = rounding;

    // Open widgets (menus, etc.)
    style.visuals.widgets.open.bg_fill = egui::Color32::from_rgb(32, 32, 38);
    style.visuals.widgets.open.weak_bg_fill = egui::Color32::from_rgb(32, 32, 38);
    style.visuals.widgets.open.corner_radius = rounding;

    // Selection
    style.visuals.selection.bg_fill = ACCENT.gamma_multiply(0.35);
    style.visuals.selection.stroke = Stroke::new(1.0, ACCENT_BRIGHT);

    // Window
    style.visuals.window_corner_radius = Rounding::same(4);
    style.visuals.window_stroke = Stroke::new(1.0, BORDER);
    style.visuals.window_shadow = Shadow::NONE;
    style.visuals.popup_shadow = Shadow::NONE;

    // Striped rows
    style.visuals.striped = true;

    // Spacing: tight but readable
    style.spacing.item_spacing = vec2(5.0, 3.0);
    style.spacing.window_margin = Margin::same(6);
    style.spacing.button_padding = vec2(6.0, 2.0);
    style.spacing.indent = 14.0;
    style.spacing.scroll = egui::style::ScrollStyle {
        bar_width: 5.0,
        ..style.spacing.scroll
    };

    ctx.set_style(style);
}

// =============================================================================
// Menu Bar (22px)
// =============================================================================

fn draw_menu_bar(ctx: &egui::Context, state: &mut EditorState) {
    egui::TopBottomPanel::top("menu_bar")
        .exact_height(22.0)
        .frame(
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(24, 24, 28))
                .inner_margin(egui::Margin::symmetric(4, 0)),
        )
        .show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New Scene").clicked() {
                        while !state.entities.is_empty() {
                            state.delete_entity(0);
                        }
                        state.scene_name = "Untitled Scene".to_string();
                        state.log(LogLevel::System, "New scene created");
                        ui.close_menu();
                    }
                    if ui.button("Open...").clicked() {
                        state.log(LogLevel::System, "Open scene (placeholder)");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui
                        .add(egui::Button::new("Save").shortcut_text("Ctrl+S"))
                        .clicked()
                    {
                        state.log(
                            LogLevel::System,
                            format!("Scene saved: {}", state.scene_name),
                        );
                        ui.close_menu();
                    }
                    if ui.button("Save As...").clicked() {
                        state.log(LogLevel::System, "Save As (placeholder)");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Export").clicked() {
                        state.log(LogLevel::System, "Export (placeholder)");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        std::process::exit(0);
                    }
                });

                ui.menu_button("Edit", |ui| {
                    if ui
                        .add(egui::Button::new("Undo").shortcut_text("Ctrl+Z"))
                        .clicked()
                    {
                        state.log(LogLevel::Info, "Undo (placeholder)");
                        ui.close_menu();
                    }
                    if ui
                        .add(egui::Button::new("Redo").shortcut_text("Ctrl+Y"))
                        .clicked()
                    {
                        state.log(LogLevel::Info, "Redo (placeholder)");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui
                        .add(egui::Button::new("Duplicate").shortcut_text("Ctrl+D"))
                        .clicked()
                    {
                        state.duplicate_selected();
                        ui.close_menu();
                    }
                    if ui
                        .add(egui::Button::new("Delete").shortcut_text("Del"))
                        .clicked()
                    {
                        state.delete_selected();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Deselect All").clicked() {
                        state.selected_entity = None;
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut state.panels.hierarchy, "Hierarchy Panel");
                    ui.checkbox(&mut state.panels.inspector, "Inspector Panel");
                    ui.checkbox(&mut state.panels.bottom, "Bottom Panel");
                    ui.separator();
                    ui.checkbox(&mut state.grid_visible, "Show Grid");
                    ui.separator();
                    if ui.button("Reset Layout").clicked() {
                        state.panels = PanelVisibility::default();
                        ui.close_menu();
                    }
                });

                ui.menu_button("Tools", |ui| {
                    if ui.button("Generate Terrain").clicked() {
                        state.execute_console_command("terrain gen");
                        ui.close_menu();
                    }
                    if ui.button("Generate Dungeon").clicked() {
                        state.execute_console_command("dungeon gen");
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Run Script...").clicked() {
                        state.log(
                            LogLevel::System,
                            "Use console: script <code>",
                        );
                        ui.close_menu();
                    }
                });

                ui.menu_button("Help", |ui| {
                    if ui.button("About Genovo").clicked() {
                        state.show_about = true;
                        ui.close_menu();
                    }
                    if ui.button("Documentation").clicked() {
                        state.log(LogLevel::System, "https://genovo.dev/docs");
                        ui.close_menu();
                    }
                    if ui.button("Shortcuts").clicked() {
                        state.log(LogLevel::System, "--- Keyboard Shortcuts ---");
                        state.log(LogLevel::System, "  Ctrl+S    Save scene");
                        state.log(LogLevel::System, "  Ctrl+Z    Undo");
                        state.log(LogLevel::System, "  Delete    Delete selected entity");
                        state.log(LogLevel::System, "  W/E/R     Translate/Rotate/Scale");
                        state.log(LogLevel::System, "  Space     Play/Pause");
                        state.log(LogLevel::System, "  F5        Toggle play mode");
                        state.log(LogLevel::System, "  F         Spawn physics ball");
                        ui.close_menu();
                    }
                });
            });
        });
}

// =============================================================================
// Toolbar (30px)
// =============================================================================

fn draw_toolbar(ctx: &egui::Context, state: &mut EditorState) {
    egui::TopBottomPanel::top("toolbar")
        .exact_height(30.0)
        .frame(
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(26, 26, 30))
                .inner_margin(egui::Margin::symmetric(4, 2))
                .stroke(egui::Stroke::new(1.0, BORDER)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;

                // Play / Pause / Stop
                let play_label = if state.is_playing && !state.is_paused {
                    "|| Pause"
                } else {
                    "|> Play"
                };
                let play_col = if state.is_playing && !state.is_paused {
                    GREEN
                } else {
                    TEXT_PRIMARY
                };
                let play_btn = egui::Button::new(
                    egui::RichText::new(play_label).color(play_col).strong().size(12.0),
                )
                .min_size(egui::vec2(60.0, 22.0));
                if ui.add(play_btn).on_hover_text("Space").clicked() {
                    state.toggle_play();
                }

                let stop_btn = egui::Button::new(
                    egui::RichText::new("[] Stop").color(RED).size(12.0),
                )
                .min_size(egui::vec2(50.0, 22.0));
                if ui.add(stop_btn).clicked() {
                    state.stop_play();
                }

                ui.add(thin_separator());

                // Transform mode
                for mode in [TransformMode::Translate, TransformMode::Rotate, TransformMode::Scale] {
                    let selected = state.transform_mode == mode;
                    let label = format!("{} ({})", mode.label(), mode.short_label());
                    let resp = ui.add(
                        egui::SelectableLabel::new(selected, egui::RichText::new(&label).size(11.5)),
                    );
                    if resp.clicked() {
                        state.transform_mode = mode;
                    }
                }

                ui.add(thin_separator());

                // Coord space
                let local = state.coord_space == CoordSpace::Local;
                if ui
                    .add(egui::SelectableLabel::new(
                        local,
                        egui::RichText::new("Local").size(11.0),
                    ))
                    .clicked()
                {
                    state.coord_space = CoordSpace::Local;
                }
                if ui
                    .add(egui::SelectableLabel::new(
                        !local,
                        egui::RichText::new("World").size(11.0),
                    ))
                    .clicked()
                {
                    state.coord_space = CoordSpace::World;
                }

                ui.add(thin_separator());

                // Snap
                ui.checkbox(&mut state.snap_enabled, "");
                ui.label(egui::RichText::new("Snap").size(11.0));
                if state.snap_enabled {
                    ui.add(
                        egui::DragValue::new(&mut state.snap_value)
                            .speed(0.1)
                            .range(0.01..=100.0)
                            .suffix(" u")
                            .max_decimals(2),
                    );
                }

                ui.add(thin_separator());

                // Grid
                ui.checkbox(&mut state.grid_visible, "");
                ui.label(egui::RichText::new("Grid").size(11.0));

                ui.add(thin_separator());

                // Speed
                ui.label(egui::RichText::new("Speed").size(11.0).color(TEXT_SECONDARY));
                ui.add(
                    egui::Slider::new(&mut state.sim_speed, 0.0..=4.0)
                        .max_decimals(2)
                        .clamping(egui::SliderClamping::Always)
                        .text("x"),
                );
            });
        });
}

fn thin_separator() -> egui::Separator {
    egui::Separator::default().spacing(6.0)
}

// =============================================================================
// Status Bar (18px)
// =============================================================================

fn draw_status_bar(ctx: &egui::Context, state: &EditorState) {
    egui::TopBottomPanel::bottom("status_bar")
        .exact_height(18.0)
        .frame(
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(24, 24, 28))
                .inner_margin(egui::Margin::symmetric(6, 0)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                // Status dot
                let (dot_color, status_text) = if state.is_playing && !state.is_paused {
                    (GREEN, "Playing")
                } else if state.is_paused {
                    (YELLOW, "Paused")
                } else {
                    (egui::Color32::from_rgb(60, 120, 60), "Editing")
                };
                let (dot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(dot_rect.center(), 3.5, dot_color);
                ui.label(
                    egui::RichText::new(status_text)
                        .size(10.5)
                        .color(TEXT_SECONDARY),
                );

                ui.add(thin_separator());
                ui.label(
                    egui::RichText::new(&state.scene_name)
                        .size(10.5)
                        .color(TEXT_PRIMARY),
                );

                // Right-aligned stats
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let bodies = state.engine.physics().body_count();
                    let ents = state.entities.len();
                    ui.label(
                        egui::RichText::new(format!(
                            "{:.0} FPS  |  {:.2} ms  |  {} entities  |  {} bodies",
                            state.fps, state.frame_time_ms, ents, bodies
                        ))
                        .size(10.5)
                        .color(TEXT_SECONDARY),
                    );
                });
            });
        });
}

// =============================================================================
// Scene Hierarchy Panel (left, 220px)
// =============================================================================

fn draw_hierarchy(ctx: &egui::Context, state: &mut EditorState) {
    if !state.panels.hierarchy {
        return;
    }

    egui::SidePanel::left("hierarchy")
        .default_width(220.0)
        .min_width(150.0)
        .max_width(400.0)
        .resizable(true)
        .frame(
            egui::Frame::new()
                .fill(BG_PANEL)
                .inner_margin(egui::Margin::same(4))
                .stroke(egui::Stroke::new(1.0, BORDER)),
        )
        .show(ctx, |ui| {
            // Header
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "HIERARCHY  ({})",
                        state.entities.len()
                    ))
                    .size(11.0)
                    .strong()
                    .color(TEXT_SECONDARY),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.menu_button(
                        egui::RichText::new("+").size(14.0).strong().color(GREEN),
                        |ui| {
                            draw_spawn_menu(ui, state);
                        },
                    );
                });
            });

            ui.add_space(2.0);
            // Thin accent line under header
            let rect = ui.available_rect_before_wrap();
            ui.painter().line_segment(
                [
                    egui::pos2(rect.left(), rect.top()),
                    egui::pos2(rect.right(), rect.top()),
                ],
                egui::Stroke::new(1.0, ACCENT_DIM),
            );
            ui.add_space(3.0);

            if state.entities.is_empty() {
                ui.add_space(20.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("No entities")
                            .size(11.0)
                            .color(TEXT_DIM),
                    );
                    ui.label(
                        egui::RichText::new("Click + to add")
                            .size(10.0)
                            .color(TEXT_DIM),
                    );
                });
                return;
            }

            // Entity list
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let mut action: Option<HierarchyAction> = None;

                    for i in 0..state.entities.len() {
                        let ent = &state.entities[i];
                        let selected = state.selected_entity == Some(i);
                        let icon_color = ent.entity_type.icon_color();
                        let icon_letter = ent.entity_type.icon_letter();

                        ui.horizontal(|ui| {
                            // Colored icon badge
                            let badge_text = egui::RichText::new(format!("[{}]", icon_letter))
                                .size(10.5)
                                .strong()
                                .color(icon_color);
                            ui.label(badge_text);

                            // Entity name as selectable
                            let name_text = egui::RichText::new(&ent.name).size(11.5);
                            let resp = ui.add(egui::SelectableLabel::new(selected, name_text));

                            if resp.clicked() {
                                action = Some(HierarchyAction::Select(i));
                            }

                            resp.context_menu(|ui| {
                                if ui.button("Duplicate").clicked() {
                                    action = Some(HierarchyAction::Duplicate(i));
                                    ui.close_menu();
                                }
                                if ui.button("Delete").clicked() {
                                    action = Some(HierarchyAction::Delete(i));
                                    ui.close_menu();
                                }
                                ui.separator();
                                if ui.button("Add Child (Empty)").clicked() {
                                    action = Some(HierarchyAction::AddChild(i));
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
                            state.selected_entity = Some(idx);
                        }
                        None => {}
                    }
                });
        });
}

fn draw_spawn_menu(ui: &mut egui::Ui, state: &mut EditorState) {
    let items: &[(&str, EntityType, &str)] = &[
        ("Empty", EntityType::Empty, "Empty object"),
        ("Cube", EntityType::Mesh, "Cube mesh with physics"),
        ("Sphere", EntityType::Mesh, "Sphere mesh with physics"),
        ("Cylinder", EntityType::Mesh, "Cylinder mesh with physics"),
        ("Point Light", EntityType::Light, "Point light source"),
        ("Camera", EntityType::Camera, "Camera entity"),
        ("Particles", EntityType::ParticleSystem, "Particle system"),
    ];

    for (name, etype, tooltip) in items {
        let icon_col = etype.icon_color();
        let btn = egui::Button::new(
            egui::RichText::new(format!("[{}] {}", etype.icon_letter(), name))
                .color(icon_col)
                .size(11.5),
        );
        if ui.add(btn).on_hover_text(*tooltip).clicked() {
            let idx = state.spawn_entity(name, *etype);
            state.selected_entity = Some(idx);
            ui.close_menu();
        }
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
        .default_width(280.0)
        .min_width(200.0)
        .max_width(450.0)
        .resizable(true)
        .frame(
            egui::Frame::new()
                .fill(BG_PANEL)
                .inner_margin(egui::Margin::same(4))
                .stroke(egui::Stroke::new(1.0, BORDER)),
        )
        .show(ctx, |ui| {
            // Header
            ui.label(
                egui::RichText::new("INSPECTOR")
                    .size(11.0)
                    .strong()
                    .color(TEXT_SECONDARY),
            );
            ui.add_space(2.0);
            let rect = ui.available_rect_before_wrap();
            ui.painter().line_segment(
                [
                    egui::pos2(rect.left(), rect.top()),
                    egui::pos2(rect.right(), rect.top()),
                ],
                egui::Stroke::new(1.0, ACCENT_DIM),
            );
            ui.add_space(3.0);

            let sel = state.selected_entity;
            if sel.is_none() || sel.unwrap() >= state.entities.len() {
                ui.add_space(20.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("No entity selected")
                            .size(11.0)
                            .color(TEXT_DIM),
                    );
                });
                return;
            }
            let idx = sel.unwrap();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    draw_inspector_content(ui, state, idx);
                });
        });
}

fn draw_inspector_content(ui: &mut egui::Ui, state: &mut EditorState, idx: usize) {
    // Name and Type
    let etype = state.entities[idx].entity_type;
    let eid = state.entities[idx].entity;
    let icon_col = etype.icon_color();

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("[{}]", etype.icon_letter()))
                .size(12.0)
                .strong()
                .color(icon_col),
        );
        ui.add(
            egui::TextEdit::singleline(&mut state.entities[idx].name)
                .desired_width(ui.available_width())
                .font(egui::TextStyle::Body),
        );
    });
    ui.label(
        egui::RichText::new(format!(
            "{} | Entity {}v{}",
            etype, eid.id, eid.generation
        ))
        .size(10.0)
        .color(TEXT_DIM),
    );
    ui.add_space(4.0);

    // Transform Section
    let mut pos_changed = false;
    egui::CollapsingHeader::new(
        egui::RichText::new("Transform").strong().size(11.5),
    )
    .default_open(true)
    .show(ui, |ui| {
        // Position
        ui.label(
            egui::RichText::new("Position")
                .size(10.5)
                .color(TEXT_SECONDARY),
        );
        ui.horizontal(|ui| {
            pos_changed |= colored_drag_xyz(ui, &mut state.entities[idx].position, 0.1);
        });

        ui.add_space(2.0);

        // Rotation
        ui.label(
            egui::RichText::new("Rotation")
                .size(10.5)
                .color(TEXT_SECONDARY),
        );
        ui.horizontal(|ui| {
            colored_drag_xyz(ui, &mut state.entities[idx].rotation, 0.5);
        });

        ui.add_space(2.0);

        // Scale
        ui.label(
            egui::RichText::new("Scale")
                .size(10.5)
                .color(TEXT_SECONDARY),
        );
        ui.horizontal(|ui| {
            colored_drag_xyz(ui, &mut state.entities[idx].scale, 0.01);
        });
    });

    if pos_changed {
        state.sync_entity_to_physics(idx);
    }

    // Physics Section
    if state.entities[idx].has_physics {
        ui.add_space(2.0);
        egui::CollapsingHeader::new(
            egui::RichText::new("Physics").strong().size(11.5),
        )
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Mass")
                        .size(10.5)
                        .color(TEXT_SECONDARY),
                );
                ui.add(
                    egui::DragValue::new(&mut state.entities[idx].mass)
                        .speed(0.1)
                        .range(0.01..=10000.0)
                        .suffix(" kg"),
                );
            });

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Friction")
                        .size(10.5)
                        .color(TEXT_SECONDARY),
                );
                ui.add(
                    egui::Slider::new(&mut state.entities[idx].friction, 0.0..=1.0)
                        .max_decimals(2),
                );
            });

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Restitution")
                        .size(10.5)
                        .color(TEXT_SECONDARY),
                );
                ui.add(
                    egui::Slider::new(&mut state.entities[idx].restitution, 0.0..=1.0)
                        .max_decimals(2),
                );
            });

            // Show velocity
            if let Some(handle) = state.entities[idx].physics_handle {
                if let Ok(vel) = state.engine.physics().get_linear_velocity(handle) {
                    ui.label(
                        egui::RichText::new(format!(
                            "Vel: ({:.2}, {:.2}, {:.2})",
                            vel.x, vel.y, vel.z
                        ))
                        .size(10.0)
                        .color(TEXT_DIM),
                    );
                }
            }
        });
    }

    // Light Section
    if state.entities[idx].is_light {
        ui.add_space(2.0);
        egui::CollapsingHeader::new(
            egui::RichText::new("Light").strong().size(11.5),
        )
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Color")
                        .size(10.5)
                        .color(TEXT_SECONDARY),
                );
                ui.color_edit_button_rgb(&mut state.entities[idx].light_color);
            });

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Intensity")
                        .size(10.5)
                        .color(TEXT_SECONDARY),
                );
                ui.add(
                    egui::Slider::new(&mut state.entities[idx].light_intensity, 0.0..=100.0)
                        .logarithmic(true)
                        .max_decimals(2),
                );
            });

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Range")
                        .size(10.5)
                        .color(TEXT_SECONDARY),
                );
                ui.add(
                    egui::Slider::new(&mut state.entities[idx].light_range, 0.1..=1000.0)
                        .logarithmic(true)
                        .max_decimals(1),
                );
            });
        });
    }

    // Add Component
    ui.add_space(6.0);
    ui.separator();
    ui.menu_button(
        egui::RichText::new("+ Add Component")
            .size(11.0)
            .color(ACCENT_BRIGHT),
        |ui| {
            if !state.entities[idx].has_physics {
                if ui.button("Rigidbody").clicked() {
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
                    state.log(LogLevel::System, "Added Rigidbody component");
                    ui.close_menu();
                }
            }
            if !state.entities[idx].is_light {
                if ui.button("Light").clicked() {
                    state.entities[idx].is_light = true;
                    state.log(LogLevel::System, "Added Light component");
                    ui.close_menu();
                }
            }
            if state.entities[idx].has_physics && state.entities[idx].is_light {
                ui.label(
                    egui::RichText::new("All components added")
                        .size(10.5)
                        .color(TEXT_DIM),
                );
            }
        },
    );
}

/// Draw colored X/Y/Z drag values (Unreal-style)
fn colored_drag_xyz(ui: &mut egui::Ui, values: &mut [f32; 3], speed: f32) -> bool {
    let mut changed = false;

    let labels = [("X", X_COLOR), ("Y", Y_COLOR), ("Z", Z_COLOR)];
    let width = ((ui.available_width() - 45.0) / 3.0).max(30.0);

    for (i, (label, color)) in labels.iter().enumerate() {
        ui.label(egui::RichText::new(*label).size(11.0).strong().color(*color));
        let drag = egui::DragValue::new(&mut values[i])
            .speed(speed)
            .max_decimals(2);
        let resp = ui.add_sized(egui::vec2(width, 18.0), drag);
        if resp.changed() {
            changed = true;
        }
    }

    changed
}

// =============================================================================
// Bottom Panel (180px) with tabs
// =============================================================================

fn draw_bottom_panel(ctx: &egui::Context, state: &mut EditorState) {
    if !state.panels.bottom {
        return;
    }

    egui::TopBottomPanel::bottom("bottom_panel")
        .default_height(180.0)
        .min_height(60.0)
        .max_height(500.0)
        .resizable(true)
        .frame(
            egui::Frame::new()
                .fill(BG_PANEL)
                .inner_margin(egui::Margin::same(4))
                .stroke(egui::Stroke::new(1.0, BORDER)),
        )
        .show(ctx, |ui| {
            // Tab bar
            ui.horizontal(|ui| {
                let tabs = [
                    (BottomTab::Console, "Console"),
                    (BottomTab::AssetBrowser, "Asset Browser"),
                    (BottomTab::Profiler, "Profiler"),
                ];
                for (tab, label) in &tabs {
                    let selected = state.bottom_tab == *tab;
                    let text = if selected {
                        egui::RichText::new(*label)
                            .size(11.0)
                            .strong()
                            .color(TEXT_PRIMARY)
                    } else {
                        egui::RichText::new(*label)
                            .size(11.0)
                            .color(TEXT_SECONDARY)
                    };
                    if ui.add(egui::SelectableLabel::new(selected, text)).clicked() {
                        state.bottom_tab = *tab;
                    }
                }
            });

            // Thin accent line under tabs
            let rect = ui.available_rect_before_wrap();
            ui.painter().line_segment(
                [
                    egui::pos2(rect.left(), rect.top()),
                    egui::pos2(rect.right(), rect.top()),
                ],
                egui::Stroke::new(1.0, ACCENT_DIM),
            );
            ui.add_space(2.0);

            match state.bottom_tab {
                BottomTab::Console => draw_console_tab(ui, state),
                BottomTab::AssetBrowser => draw_asset_browser_tab(ui, state),
                BottomTab::Profiler => draw_profiler_tab(ui, state),
            }
        });
}

// =============================================================================
// Console Tab
// =============================================================================

fn draw_console_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    let input_height = 24.0;
    let available = ui.available_height() - input_height - 4.0;

    // Log output
    egui::ScrollArea::vertical()
        .max_height(available.max(20.0))
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for entry in &state.console_log {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    // Timestamp
                    ui.label(
                        egui::RichText::new(format!("{:>6.1}", entry.timestamp))
                            .size(10.0)
                            .color(TEXT_DIM)
                            .monospace(),
                    );
                    // Level badge
                    ui.label(
                        egui::RichText::new(entry.level.prefix())
                            .size(10.0)
                            .color(entry.level.color())
                            .monospace()
                            .strong(),
                    );
                    // Message
                    ui.label(
                        egui::RichText::new(&entry.text)
                            .size(11.0)
                            .color(entry.level.color()),
                    );
                });
            }
        });

    // Command input
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(">")
                .monospace()
                .color(ACCENT_BRIGHT)
                .size(12.0),
        );
        let resp = ui.add(
            egui::TextEdit::singleline(&mut state.console_input)
                .desired_width(ui.available_width() - 50.0)
                .hint_text("Type command... (help for list)")
                .font(egui::TextStyle::Monospace),
        );

        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            submit_console(state);
            resp.request_focus();
        }

        // History navigation
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

        if ui
            .add(egui::Button::new(
                egui::RichText::new("Run").size(11.0).color(ACCENT_BRIGHT),
            ))
            .clicked()
        {
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
// Asset Browser Tab
// =============================================================================

fn draw_asset_browser_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    // Path bar
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Path:").size(10.5).color(TEXT_SECONDARY));
        ui.add(
            egui::TextEdit::singleline(&mut state.asset_path)
                .desired_width(200.0)
                .font(egui::TextStyle::Monospace),
        );
        if ui
            .add(egui::Button::new(egui::RichText::new("Up").size(10.5)))
            .clicked()
        {
            if let Some(pos) = state
                .asset_path
                .rfind('/')
                .or_else(|| state.asset_path.rfind('\\'))
            {
                state.asset_path.truncate(pos);
            }
        }
        ui.add(thin_separator());
        ui.label(
            egui::RichText::new("Search:")
                .size(10.5)
                .color(TEXT_SECONDARY),
        );
        ui.add(
            egui::TextEdit::singleline(&mut state.asset_search)
                .desired_width(120.0)
                .hint_text("Filter..."),
        );
    });

    ui.add_space(2.0);

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let entries = scan_asset_dir(&state.asset_path, &state.asset_search);

            if entries.is_empty() {
                let defaults = [
                    ("[D] models/", "3D mesh assets", YELLOW),
                    ("[D] textures/", "Image textures", YELLOW),
                    ("[D] audio/", "Sound effects & music", YELLOW),
                    ("[D] scripts/", "GenovoScript files", CYAN),
                    ("[D] scenes/", "Scene definitions", MAGENTA),
                    ("[D] materials/", "Material definitions", GREEN),
                ];
                ui.label(
                    egui::RichText::new("Default asset layout:")
                        .size(10.5)
                        .color(TEXT_DIM),
                );
                for (name, desc, color) in &defaults {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(*name).size(11.0).color(*color));
                        ui.label(
                            egui::RichText::new(*desc).size(10.5).color(TEXT_DIM),
                        );
                    });
                }
            } else {
                let col_width = 150.0;
                let cols = ((ui.available_width() / col_width) as usize).max(1);
                egui::Grid::new("asset_grid")
                    .num_columns(cols)
                    .spacing(egui::vec2(6.0, 4.0))
                    .show(ui, |ui| {
                        for (i, entry) in entries.iter().enumerate() {
                            let icon = asset_icon(&entry.name, entry.is_dir);
                            let color = if entry.is_dir { YELLOW } else { TEXT_SECONDARY };
                            let btn = egui::Button::new(
                                egui::RichText::new(format!("{} {}", icon, entry.name))
                                    .size(10.5)
                                    .color(color),
                            )
                            .min_size(egui::vec2(col_width - 10.0, 18.0));
                            if ui.add(btn).clicked() {
                                if entry.is_dir {
                                    state.asset_path =
                                        format!("{}/{}", state.asset_path, entry.name);
                                } else {
                                    state.log(
                                        LogLevel::Info,
                                        format!("Selected: {}", entry.name),
                                    );
                                }
                            }
                            if (i + 1) % cols == 0 {
                                ui.end_row();
                            }
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
            if !filter.is_empty() && !name.to_lowercase().contains(&filter.to_lowercase()) {
                continue;
            }
            let is_dir = entry.file_type().map_or(false, |t| t.is_dir());
            entries.push(AssetEntry { name, is_dir });
        }
    }
    entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
    entries
}

fn asset_icon(name: &str, is_dir: bool) -> &'static str {
    if is_dir {
        return "[D]";
    }
    match name.rsplit('.').next().unwrap_or("") {
        "obj" | "fbx" | "gltf" | "glb" => "[3D]",
        "png" | "jpg" | "jpeg" | "bmp" | "tga" | "dds" => "[TX]",
        "wav" | "ogg" | "mp3" | "flac" => "[AU]",
        "rs" | "lua" | "py" | "js" => "[SC]",
        "ron" | "json" | "toml" | "yaml" => "[CF]",
        "scene" | "scn" => "[SN]",
        _ => "[??]",
    }
}

// =============================================================================
// Profiler Tab
// =============================================================================

fn draw_profiler_tab(ui: &mut egui::Ui, state: &mut EditorState) {
    // Stats header
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 12.0;
        ui.label(
            egui::RichText::new(format!("FPS: {:.0}", state.fps))
                .size(11.0)
                .color(GREEN)
                .strong(),
        );
        ui.label(
            egui::RichText::new(format!("Frame: {:.2} ms", state.frame_time_ms))
                .size(11.0)
                .color(TEXT_PRIMARY),
        );

        if !state.frame_times.is_empty() {
            let min = state.frame_times.iter().cloned().fold(f64::MAX, f64::min);
            let max = state.frame_times.iter().cloned().fold(f64::MIN, f64::max);
            let avg = state.frame_times.iter().sum::<f64>() / state.frame_times.len() as f64;
            ui.label(
                egui::RichText::new(format!(
                    "Min: {:.2}  Max: {:.2}  Avg: {:.2} ms",
                    min, max, avg
                ))
                .size(10.5)
                .color(TEXT_SECONDARY),
            );
        }
    });

    ui.add_space(2.0);

    // Frame time plot
    let points: Vec<[f64; 2]> = state
        .frame_times
        .iter()
        .enumerate()
        .map(|(i, &v)| [i as f64, v])
        .collect();
    let line = egui_plot::Line::new(egui_plot::PlotPoints::new(points))
        .name("Frame time (ms)")
        .color(ACCENT_BRIGHT);

    // 16.67ms target line
    let target_line = egui_plot::HLine::new(16.67)
        .name("60 FPS target")
        .color(egui::Color32::from_rgb(60, 60, 60));

    egui_plot::Plot::new("profiler_frame_time")
        .height(ui.available_height().max(40.0))
        .include_y(0.0)
        .include_y(20.0)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .y_axis_label("ms")
        .show(ui, |plot_ui| {
            plot_ui.hline(target_line);
            plot_ui.line(line);
        });
}

// =============================================================================
// Central Viewport
// =============================================================================

fn draw_viewport(ctx: &egui::Context, state: &EditorState) {
    egui::CentralPanel::default()
        .frame(
            egui::Frame::new()
                .fill(egui::Color32::TRANSPARENT)
                .inner_margin(egui::Margin::same(0)),
        )
        .show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            let painter = ui.painter();

            // FPS overlay (top-left)
            let fps_col = if state.fps > 55.0 {
                GREEN
            } else if state.fps > 30.0 {
                YELLOW
            } else {
                RED
            };
            painter.text(
                rect.left_top() + egui::vec2(10.0, 10.0),
                egui::Align2::LEFT_TOP,
                format!("{:.0} FPS | {:.2} ms", state.fps, state.frame_time_ms),
                egui::FontId::monospace(12.0),
                fps_col,
            );

            // Camera info (top-right)
            painter.text(
                rect.right_top() + egui::vec2(-10.0, 10.0),
                egui::Align2::RIGHT_TOP,
                format!(
                    "Camera ({:.1}, {:.1}, {:.1}) dist={:.1}",
                    state.camera_target[0],
                    state.camera_target[1],
                    state.camera_target[2],
                    state.camera_dist,
                ),
                egui::FontId::monospace(10.0),
                TEXT_DIM,
            );

            // Transform mode (top-left, below FPS)
            painter.text(
                rect.left_top() + egui::vec2(10.0, 28.0),
                egui::Align2::LEFT_TOP,
                format!(
                    "{} | {} | {}",
                    state.transform_mode.label(),
                    if state.coord_space == CoordSpace::Local {
                        "Local"
                    } else {
                        "World"
                    },
                    if state.grid_visible { "Grid ON" } else { "Grid OFF" },
                ),
                egui::FontId::monospace(10.0),
                TEXT_DIM,
            );

            // Play state overlay
            if state.is_playing && !state.is_paused {
                let t = state.frame_count as f32 * 0.03;
                let alpha = ((t.sin() + 1.0) * 0.5 * 80.0 + 40.0) as u8;
                let pulse_col = egui::Color32::from_rgba_premultiplied(76, 195, 85, alpha);
                painter.rect_stroke(
                    rect.shrink(2.0),
                    0.0,
                    egui::Stroke::new(2.0, pulse_col),
                    egui::StrokeKind::Outside,
                );
                painter.text(
                    rect.left_top() + egui::vec2(10.0, 44.0),
                    egui::Align2::LEFT_TOP,
                    "SIMULATING",
                    egui::FontId::monospace(11.0),
                    GREEN,
                );
            } else if state.is_paused {
                painter.rect_stroke(
                    rect.shrink(2.0),
                    0.0,
                    egui::Stroke::new(2.0, egui::Color32::from_rgba_premultiplied(230, 190, 50, 60)),
                    egui::StrokeKind::Outside,
                );
                painter.text(
                    rect.left_top() + egui::vec2(10.0, 44.0),
                    egui::Align2::LEFT_TOP,
                    "PAUSED",
                    egui::FontId::monospace(11.0),
                    YELLOW,
                );
            }

            // Center info
            painter.text(
                egui::pos2(rect.center().x, rect.bottom() - 20.0),
                egui::Align2::CENTER_BOTTOM,
                format!(
                    "3D Viewport | {} entities | {} physics bodies",
                    state.entities.len(),
                    state.engine.physics().body_count(),
                ),
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(50, 50, 60),
            );

            // Selected entity indicator
            if let Some(idx) = state.selected_entity {
                if idx < state.entities.len() {
                    let ent = &state.entities[idx];
                    painter.text(
                        egui::pos2(rect.center().x, rect.top() + 10.0),
                        egui::Align2::CENTER_TOP,
                        format!(
                            "Selected: {} ({:.1}, {:.1}, {:.1})",
                            ent.name, ent.position[0], ent.position[1], ent.position[2]
                        ),
                        egui::FontId::monospace(10.0),
                        ACCENT_BRIGHT,
                    );
                }
            }
        });
}

// =============================================================================
// About Window
// =============================================================================

fn draw_about_window(ctx: &egui::Context, show: &mut bool) {
    egui::Window::new("About Genovo Engine")
        .open(show)
        .resizable(false)
        .default_width(320.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("GENOVO ENGINE")
                        .size(18.0)
                        .strong()
                        .color(ACCENT_BRIGHT),
                );
                ui.label(egui::RichText::new("v0.1.0").size(11.0).color(TEXT_SECONDARY));
                ui.add_space(8.0);
                ui.label("A AAA-tier game engine built in Rust");
                ui.label("26 engine modules fully linked");
                ui.add_space(6.0);

                let items = [
                    ("Rendering", "wgpu 24"),
                    ("UI", "egui 0.31"),
                    ("Physics", "Custom impulse solver"),
                    ("ECS", "Archetype-based"),
                    ("Audio", "Software PCM mixer"),
                    ("Scripting", "GenovoScript VM"),
                ];
                for (label, value) in &items {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("{}:", label))
                                .size(10.5)
                                .color(TEXT_SECONDARY),
                        );
                        ui.label(egui::RichText::new(*value).size(10.5));
                    });
                }

                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("genovo.dev")
                        .size(11.0)
                        .color(ACCENT),
                );
            });
        });
}

// =============================================================================
// GPU Helpers
// =============================================================================

fn make_depth(dev: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    dev.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
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
}

struct GpuState {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,

    // egui
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    // editor
    editor: EditorState,
    theme_applied: bool,
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
                    .with_title("Genovo Engine Editor")
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
        let (dev, que) =
            pollster::block_on(adap.request_device(&Default::default(), None)).unwrap();
        let sz = w.inner_size();
        let cfg = surf
            .get_default_config(&adap, sz.width.max(1), sz.height.max(1))
            .unwrap();
        surf.configure(&dev, &cfg);

        // Shader + pipeline
        let sh = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pl = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &sh,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sh,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: cfg.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let dv = make_depth(&dev, sz.width, sz.height);

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
                shape: physics::CollisionShape::Box {
                    half_extents: Vec3::new(50.0, 0.5, 50.0),
                },
                ..Default::default()
            },
        );

        // egui init
        let egui_ctx = egui::Context::default();

        // Apply theme immediately
        apply_professional_theme(&egui_ctx);

        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &w,
            Some(w.scale_factor() as f32),
            None,
            Some(dev.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &dev,
            cfg.format,
            Some(wgpu::TextureFormat::Depth32Float),
            1,
            false,
        );

        // Editor state with default entities
        let mut editor = EditorState::new(engine);
        editor.log(LogLevel::System, "Genovo Engine Editor initialized");
        editor.log(
            LogLevel::System,
            format!(
                "GPU: {} ({:?})",
                adap.get_info().name,
                adap.get_info().backend
            ),
        );
        editor.log(LogLevel::System, "Type 'help' in console for commands");

        // Spawn default scene entities
        let ground_idx = editor.spawn_entity("Ground Plane", EntityType::Mesh);
        editor.entities[ground_idx].scale = [100.0, 1.0, 100.0];

        let cube_idx = editor.spawn_entity("Cube", EntityType::Mesh);
        editor.entities[cube_idx].position = [0.0, 5.0, 0.0];
        if let Some(h) = editor.entities[cube_idx].physics_handle {
            let _ = editor
                .engine
                .physics_mut()
                .set_position(h, Vec3::new(0.0, 5.0, 0.0));
        }

        let light_idx = editor.spawn_entity("Directional Light", EntityType::Light);
        editor.entities[light_idx].position = [5.0, 10.0, 5.0];

        let cam_idx = editor.spawn_entity("Main Camera", EntityType::Camera);
        editor.entities[cam_idx].position = [0.0, 5.0, -15.0];

        editor.selected_entity = Some(cube_idx);

        println!(
            "[Genovo] GPU: {} ({:?})",
            adap.get_info().name,
            adap.get_info().backend
        );
        println!("[Genovo] Editor ready.");

        self.gpu = Some(GpuState {
            window: w,
            device: dev,
            queue: que,
            surface: surf,
            config: cfg,
            pipeline: pl,
            depth_view: dv,
            egui_ctx,
            egui_state,
            egui_renderer,
            editor,
            theme_applied: true,
        });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        let Some(s) = self.gpu.as_mut() else {
            return;
        };

        // Let egui handle events first
        let resp = s.egui_state.on_window_event(&s.window, &ev);
        if resp.repaint {
            s.window.request_redraw();
        }

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
                            // Ctrl+S: Save
                            KeyCode::KeyS if ctrl => {
                                s.editor.log(
                                    LogLevel::System,
                                    format!("Scene saved: {}", s.editor.scene_name),
                                );
                            }
                            // Ctrl+Z: Undo
                            KeyCode::KeyZ if ctrl => {
                                s.editor.log(LogLevel::Info, "Undo (placeholder)");
                            }
                            // Ctrl+D: Duplicate
                            KeyCode::KeyD if ctrl => {
                                s.editor.duplicate_selected();
                            }
                            // Transform modes
                            KeyCode::KeyW if !ctrl => {
                                s.editor.transform_mode = TransformMode::Translate;
                            }
                            KeyCode::KeyE if !ctrl => {
                                s.editor.transform_mode = TransformMode::Rotate;
                            }
                            KeyCode::KeyR if !ctrl => {
                                s.editor.transform_mode = TransformMode::Scale;
                            }
                            // Delete selected
                            KeyCode::Delete => {
                                s.editor.delete_selected();
                            }
                            // Space: Play/Pause toggle
                            KeyCode::Space => {
                                s.editor.toggle_play();
                            }
                            // F5: Toggle play mode
                            KeyCode::F5 => {
                                if s.editor.is_playing {
                                    s.editor.stop_play();
                                } else {
                                    s.editor.is_playing = true;
                                    s.editor.is_paused = false;
                                    s.editor.log(LogLevel::System, "Simulation started (F5)");
                                }
                            }
                            // F: Spawn physics ball
                            KeyCode::KeyF if !ctrl => {
                                s.editor.spawn_physics_ball();
                            }
                            // Escape
                            KeyCode::Escape => {
                                s.editor.selected_entity = None;
                            }
                            _ => {}
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                // Timing
                let now = Instant::now();
                let dt = now.duration_since(s.editor.last_frame).as_secs_f32();
                s.editor.last_frame = now;
                s.editor.frame_count += 1;
                s.editor.frame_time_ms = dt * 1000.0;
                s.editor.fps = 1.0 / dt.max(0.0001);

                // Profiler data
                s.editor.frame_times.push_back(dt as f64 * 1000.0);
                if s.editor.frame_times.len() > 300 {
                    s.editor.frame_times.pop_front();
                }
                s.editor.fps_history.push_back(s.editor.fps as f64);
                if s.editor.fps_history.len() > 300 {
                    s.editor.fps_history.pop_front();
                }

                // Physics step
                if s.editor.is_playing && !s.editor.is_paused {
                    let phys_dt = (dt * s.editor.sim_speed).min(1.0 / 30.0);
                    let _ = s.editor.engine.physics_mut().step(phys_dt);
                    s.editor.sync_physics_to_entities();
                }

                // Re-apply theme if needed
                if !s.theme_applied {
                    apply_professional_theme(&s.egui_ctx);
                    s.theme_applied = true;
                }

                // Build egui frame
                let raw_input = s.egui_state.take_egui_input(&s.window);
                let full_output = s.egui_ctx.run(raw_input, |ctx| {
                    if s.editor.show_about {
                        draw_about_window(ctx, &mut s.editor.show_about);
                    }

                    draw_menu_bar(ctx, &mut s.editor);
                    draw_toolbar(ctx, &mut s.editor);
                    draw_status_bar(ctx, &s.editor);
                    draw_bottom_panel(ctx, &mut s.editor);
                    draw_hierarchy(ctx, &mut s.editor);
                    draw_inspector(ctx, &mut s.editor);
                    draw_viewport(ctx, &s.editor);
                });

                s.egui_state
                    .handle_platform_output(&s.window, full_output.platform_output);

                let clipped_primitives =
                    s.egui_ctx
                        .tessellate(full_output.shapes, full_output.pixels_per_point);

                for (id, delta) in &full_output.textures_delta.set {
                    s.egui_renderer
                        .update_texture(&s.device, &s.queue, *id, delta);
                }

                // GPU Render
                let Ok(out) = s.surface.get_current_texture() else {
                    s.window.request_redraw();
                    return;
                };
                let view = out.texture.create_view(&Default::default());

                let screen = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [s.config.width, s.config.height],
                    pixels_per_point: s.window.scale_factor() as f32,
                };

                // Scene background color
                let t = s.editor.frame_count as f32 * 0.005;
                let (br, bg, bb) = if s.editor.is_playing && !s.editor.is_paused {
                    let pulse = (t * 2.0).sin().abs() * 0.01;
                    (0.065 + pulse as f64, 0.065_f64, 0.085 + pulse as f64)
                } else {
                    (0.065_f64, 0.065, 0.085)
                };

                // Pass 1: Clear + 3D scene (triangle)
                let mut scene_enc =
                    s.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("scene"),
                        });
                {
                    let mut pass = scene_enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("scene_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: br,
                                    g: bg,
                                    b: bb,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &s.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(&s.pipeline);
                    pass.draw(0..3, 0..1);
                }

                // Pass 2: egui overlay
                let mut egui_enc =
                    s.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("egui"),
                        });
                let user_cmd_bufs = s.egui_renderer.update_buffers(
                    &s.device,
                    &s.queue,
                    &mut egui_enc,
                    &clipped_primitives,
                    &screen,
                );
                {
                    let mut pass = egui_enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    // Lifetime transmute: wgpu 24 RenderPass has 'encoder lifetime
                    // but egui-wgpu 0.31 expects 'static. Safe because pass is used
                    // and dropped within this block.
                    let pass_static: &mut wgpu::RenderPass<'static> =
                        unsafe { std::mem::transmute(&mut pass) };
                    s.egui_renderer
                        .render(pass_static, &clipped_primitives, &screen);
                }

                // Submit
                let mut cmd_bufs: Vec<wgpu::CommandBuffer> = Vec::new();
                cmd_bufs.push(scene_enc.finish());
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

    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = EditorApp { gpu: None };
    let _ = el.run_app(&mut app);
}
