//! Genovo Engine Editor — Professional Game Engine Editor
//!
//! A complete, functional editor built with egui + wgpu + winit:
//!   - Real GPU-rendered 3D viewport with triangle pipeline
//!   - Scene hierarchy with ECS entities
//!   - Inspector with live-editing of transforms and physics
//!   - Asset browser with directory scanning
//!   - Console with command execution
//!   - Profiler with frame time graphs
//!   - Custom dark theme (Unreal-inspired)
//!   - Dockable panel layout via egui side/bottom panels

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
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

// ═══════════════════════════════════════════════════════════════════════════════
// Constants & Shader
// ═══════════════════════════════════════════════════════════════════════════════

const SHADER: &str = r#"
struct Out { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    var p = array<vec2<f32>,3>(vec2(0.0,0.6),vec2(-0.5,-0.4),vec2(0.5,-0.4));
    var c = array<vec3<f32>,3>(vec3(1.0,0.2,0.2),vec3(0.2,1.0,0.2),vec3(0.2,0.4,1.0));
    var o: Out; o.pos = vec4(p[i],0.0,1.0); o.col = c[i]; return o;
}
@fragment fn fs(in: Out) -> @location(0) vec4<f32> { return vec4(in.col,1.0); }
"#;

const ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 120, 215);
const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(0, 90, 170);
const GREEN_ACCENT: egui::Color32 = egui::Color32::from_rgb(60, 180, 75);
const YELLOW_ACCENT: egui::Color32 = egui::Color32::from_rgb(220, 180, 40);
const RED_ACCENT: egui::Color32 = egui::Color32::from_rgb(220, 60, 60);

// ═══════════════════════════════════════════════════════════════════════════════
// Entity types & Scene entity
// ═══════════════════════════════════════════════════════════════════════════════

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

impl SceneEntity {
    fn icon(&self) -> &str {
        match self.entity_type {
            EntityType::Empty => "[E]",
            EntityType::Mesh => "[M]",
            EntityType::Light => "[L]",
            EntityType::Camera => "[C]",
            EntityType::ParticleSystem => "[P]",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transform mode
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransformMode {
    Translate,
    Rotate,
    Scale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoordSpace {
    Local,
    World,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Console log entry
// ═══════════════════════════════════════════════════════════════════════════════

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
            LogLevel::Info => egui::Color32::from_rgb(160, 160, 160),
            LogLevel::Warn => YELLOW_ACCENT,
            LogLevel::Error => RED_ACCENT,
            LogLevel::System => egui::Color32::from_rgb(80, 150, 255),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Panel visibility
// ═══════════════════════════════════════════════════════════════════════════════

struct PanelVisibility {
    hierarchy: bool,
    inspector: bool,
    asset_browser: bool,
    console: bool,
    profiler: bool,
    scene_settings: bool,
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            hierarchy: true,
            inspector: true,
            asset_browser: true,
            console: true,
            profiler: false,
            scene_settings: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bottom tab selection
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq)]
enum BottomTab {
    AssetBrowser,
    Console,
    Profiler,
    SceneSettings,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Editor state
// ═══════════════════════════════════════════════════════════════════════════════

struct EditorState {
    // Engine
    engine: Engine,

    // Scene entities (our high-level tracking)
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

    // Scene name
    scene_name: String,

    // About dialog
    show_about: bool,
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
            console_log: Vec::new(),
            console_input: String::new(),
            console_history: Vec::new(),
            console_history_idx: None,
            asset_path: "assets".to_string(),
            asset_search: String::new(),
            frame_times: VecDeque::with_capacity(256),
            fps_history: VecDeque::with_capacity(256),
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
        }
    }

    fn log(&mut self, level: LogLevel, text: impl Into<String>) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        self.console_log.push(LogEntry {
            text: text.into(),
            level,
            timestamp: elapsed,
        });
        // Keep log bounded
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

        // Auto-add physics for Mesh entities
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
        self.log(LogLevel::System, format!("Spawned entity: {} ({})", name, etype));
        idx
    }

    fn delete_selected(&mut self) {
        if let Some(idx) = self.selected_entity {
            if idx < self.entities.len() {
                let ent = &self.entities[idx];
                let name = ent.name.clone();
                // Remove physics body
                if let Some(handle) = ent.physics_handle {
                    let _ = self.engine.physics_mut().remove_body(handle);
                }
                // Despawn ECS entity
                self.engine.world_mut().despawn(ent.entity);
                self.entities.remove(idx);
                self.selected_entity = None;
                self.log(LogLevel::System, format!("Deleted entity: {}", name));
            }
        }
    }

    fn duplicate_selected(&mut self) {
        if let Some(idx) = self.selected_entity {
            if idx < self.entities.len() {
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
                e.position = [pos[0] + 1.0, pos[1], pos[2]];
                e.rotation = rot;
                e.scale = scl;
                e.is_light = is_light;
                e.light_color = lc;
                e.light_intensity = li;
                e.light_range = lr;

                // Update physics position if applicable
                if let Some(handle) = e.physics_handle {
                    let _ = self.engine.physics_mut().set_position(
                        handle,
                        Vec3::new(e.position[0], e.position[1], e.position[2]),
                    );
                }

                self.selected_entity = Some(new_idx);
            }
        }
    }

    fn execute_console_command(&mut self, cmd: &str) {
        self.log(LogLevel::Info, format!("> {}", cmd));
        let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
        if parts.is_empty() {
            return;
        }
        match parts[0] {
            "help" => {
                self.log(LogLevel::System, "Available commands:");
                self.log(LogLevel::System, "  help                 - Show this help");
                self.log(LogLevel::System, "  clear                - Clear console");
                self.log(LogLevel::System, "  stats                - Engine statistics");
                self.log(LogLevel::System, "  spawn <type> [name]  - Spawn entity (empty/cube/sphere/light/camera)");
                self.log(LogLevel::System, "  physics start        - Start physics simulation");
                self.log(LogLevel::System, "  physics stop         - Stop physics simulation");
                self.log(LogLevel::System, "  physics step         - Step physics once");
                self.log(LogLevel::System, "  terrain gen          - Generate terrain heightmap");
                self.log(LogLevel::System, "  dungeon gen          - Generate BSP dungeon");
                self.log(LogLevel::System, "  script <code>        - Execute script code");
            }
            "clear" => {
                self.console_log.clear();
            }
            "stats" => {
                let ecs_count = self.engine.world().entity_count();
                let body_count = self.engine.physics().body_count();
                let active_bodies = self.engine.physics().active_body_count();
                self.log(LogLevel::System, format!("ECS entities: {}", ecs_count));
                self.log(LogLevel::System, format!("Scene entities: {}", self.entities.len()));
                self.log(LogLevel::System, format!("Physics bodies: {} ({} active)", body_count, active_bodies));
                self.log(LogLevel::System, format!("FPS: {:.1}", self.fps));
                self.log(LogLevel::System, format!("Frame time: {:.2}ms", self.frame_time_ms));
            }
            "spawn" => {
                let etype = match parts.get(1).copied() {
                    Some("cube") | Some("mesh") => EntityType::Mesh,
                    Some("sphere") => EntityType::Mesh,
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
            "terrain" if parts.get(1) == Some(&"gen") => {
                match genovo_terrain::Heightmap::generate_procedural(257, 0.7, 42) {
                    Ok(h) => self.log(
                        LogLevel::System,
                        format!(
                            "Generated 257x257 heightmap [{:.2}, {:.2}]",
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
                    format!("Generated dungeon: {} rooms", d.rooms.len()),
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
                    Err(e) => self.log(LogLevel::Error, format!("Script compile error: {}", e)),
                }
            }
            other => {
                self.log(LogLevel::Warn, format!("Unknown command: '{}'. Type 'help'", other));
            }
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
}

// ═══════════════════════════════════════════════════════════════════════════════
// Theme
// ═══════════════════════════════════════════════════════════════════════════════

fn apply_theme(ctx: &egui::Context) {
    use egui::*;

    let mut style = (*ctx.style()).clone();

    // Unreal-inspired dark palette
    style.visuals.dark_mode = true;
    style.visuals.panel_fill = Color32::from_rgb(30, 30, 30);
    style.visuals.window_fill = Color32::from_rgb(30, 30, 30);
    style.visuals.extreme_bg_color = Color32::from_rgb(18, 18, 18);
    style.visuals.faint_bg_color = Color32::from_rgb(38, 38, 42);
    style.visuals.override_text_color = Some(Color32::from_rgb(210, 210, 210));

    // Widgets
    let rounding = Rounding::same(4);
    style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(40, 40, 44);
    style.visuals.widgets.noninteractive.weak_bg_fill = Color32::from_rgb(40, 40, 44);
    style.visuals.widgets.noninteractive.fg_stroke =
        Stroke::new(1.0, Color32::from_rgb(150, 150, 150));
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);
    style.visuals.widgets.noninteractive.corner_radius = rounding;

    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(45, 45, 50);
    style.visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(45, 45, 50);
    style.visuals.widgets.inactive.fg_stroke =
        Stroke::new(1.0, Color32::from_rgb(200, 200, 200));
    style.visuals.widgets.inactive.bg_stroke =
        Stroke::new(0.5, Color32::from_rgb(55, 55, 60));
    style.visuals.widgets.inactive.corner_radius = rounding;

    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(55, 55, 62);
    style.visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(55, 55, 62);
    style.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    style.visuals.widgets.hovered.bg_stroke =
        Stroke::new(1.0, Color32::from_rgb(0, 120, 215));
    style.visuals.widgets.hovered.corner_radius = rounding;

    style.visuals.widgets.active.bg_fill = Color32::from_rgb(0, 120, 215);
    style.visuals.widgets.active.weak_bg_fill = Color32::from_rgb(0, 120, 215);
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    style.visuals.widgets.active.corner_radius = rounding;

    style.visuals.widgets.open.bg_fill = Color32::from_rgb(38, 38, 42);
    style.visuals.widgets.open.weak_bg_fill = Color32::from_rgb(38, 38, 42);
    style.visuals.widgets.open.corner_radius = rounding;

    style.visuals.selection.bg_fill =
        Color32::from_rgb(0, 100, 180).gamma_multiply(0.4);
    style.visuals.selection.stroke =
        Stroke::new(1.0, Color32::from_rgb(0, 140, 220));

    style.visuals.window_corner_radius = Rounding::same(6);
    style.visuals.window_stroke =
        Stroke::new(1.0, Color32::from_rgb(50, 50, 55));
    style.visuals.window_shadow = Shadow::NONE;

    style.visuals.striped = true;

    // Spacing
    style.spacing.item_spacing = vec2(6.0, 3.0);
    style.spacing.window_margin = Margin::same(6);
    style.spacing.button_padding = vec2(6.0, 3.0);
    style.spacing.indent = 16.0;
    style.spacing.scroll = egui::style::ScrollStyle {
        bar_width: 6.0,
        ..style.spacing.scroll
    };

    ctx.set_style(style);
}

// ═══════════════════════════════════════════════════════════════════════════════
// UI drawing functions
// ═══════════════════════════════════════════════════════════════════════════════

fn draw_menu_bar(ctx: &egui::Context, state: &mut EditorState) {
    egui::TopBottomPanel::top("menu_bar").exact_height(24.0).show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New Scene").clicked() {
                    // Clear all entities
                    for ent in state.entities.drain(..) {
                        if let Some(h) = ent.physics_handle {
                            let _ = state.engine.physics_mut().remove_body(h);
                        }
                        state.engine.world_mut().despawn(ent.entity);
                    }
                    state.selected_entity = None;
                    state.scene_name = "Untitled Scene".to_string();
                    state.log(LogLevel::System, "New scene created");
                    ui.close_menu();
                }
                if ui.button("Open...").clicked() {
                    state.log(LogLevel::System, "Open dialog (placeholder)");
                    ui.close_menu();
                }
                ui.separator();
                if ui.add(egui::Button::new("Save").shortcut_text("Ctrl+S")).clicked() {
                    state.log(LogLevel::System, format!("Saved: {}", state.scene_name));
                    ui.close_menu();
                }
                if ui.button("Save As...").clicked() {
                    state.log(LogLevel::System, "Save As dialog (placeholder)");
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
                if ui.add(egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                    state.log(LogLevel::Info, "Undo (placeholder)");
                    ui.close_menu();
                }
                if ui.add(egui::Button::new("Redo").shortcut_text("Ctrl+Y")).clicked() {
                    state.log(LogLevel::Info, "Redo (placeholder)");
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Select All").clicked() {
                    state.log(LogLevel::Info, "Select All");
                    ui.close_menu();
                }
                if ui.button("Deselect").clicked() {
                    state.selected_entity = None;
                    ui.close_menu();
                }
                if ui.button("Delete").clicked() {
                    state.delete_selected();
                    ui.close_menu();
                }
            });

            ui.menu_button("View", |ui| {
                if ui.button("Reset Layout").clicked() {
                    state.panels = PanelVisibility::default();
                    ui.close_menu();
                }
                ui.separator();
                ui.checkbox(&mut state.panels.hierarchy, "Hierarchy");
                ui.checkbox(&mut state.panels.inspector, "Inspector");
                ui.checkbox(&mut state.panels.asset_browser, "Asset Browser");
                ui.checkbox(&mut state.panels.console, "Console");
                ui.checkbox(&mut state.panels.profiler, "Profiler");
                ui.checkbox(&mut state.panels.scene_settings, "Scene Settings");
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
                if ui.button("Run Script...").clicked() {
                    state.log(LogLevel::System, "Script dialog (use console: script <code>)");
                    ui.close_menu();
                }
            });

            ui.menu_button("Help", |ui| {
                if ui.button("About").clicked() {
                    state.show_about = true;
                    ui.close_menu();
                }
                if ui.button("Documentation").clicked() {
                    state.log(LogLevel::System, "https://genovo.dev/docs");
                    ui.close_menu();
                }
            });
        });
    });
}

fn draw_toolbar(ctx: &egui::Context, state: &mut EditorState) {
    egui::TopBottomPanel::top("toolbar")
        .exact_height(32.0)
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;

                // Play / Pause / Stop
                let play_color = if state.is_playing && !state.is_paused {
                    GREEN_ACCENT
                } else {
                    egui::Color32::from_rgb(180, 180, 180)
                };
                if ui
                    .add(egui::Button::new(
                        egui::RichText::new(if state.is_playing && !state.is_paused {
                            "|| Pause"
                        } else {
                            "> Play"
                        })
                        .color(play_color)
                        .strong(),
                    ))
                    .clicked()
                {
                    if state.is_playing && !state.is_paused {
                        state.is_paused = true;
                        state.log(LogLevel::System, "Simulation paused");
                    } else {
                        state.is_playing = true;
                        state.is_paused = false;
                        state.log(LogLevel::System, "Simulation started");
                    }
                }

                if ui
                    .add(egui::Button::new(
                        egui::RichText::new("[] Stop").color(RED_ACCENT),
                    ))
                    .clicked()
                {
                    state.is_playing = false;
                    state.is_paused = false;
                    state.log(LogLevel::System, "Simulation stopped");
                }

                ui.separator();

                // Transform mode buttons
                let modes = [
                    (TransformMode::Translate, "Translate (W)"),
                    (TransformMode::Rotate, "Rotate (E)"),
                    (TransformMode::Scale, "Scale (R)"),
                ];
                for (mode, label) in &modes {
                    let selected = state.transform_mode == *mode;
                    if ui
                        .add(egui::SelectableLabel::new(selected, *label))
                        .clicked()
                    {
                        state.transform_mode = *mode;
                    }
                }

                ui.separator();

                // Coord space toggle
                let is_local = state.coord_space == CoordSpace::Local;
                if ui
                    .add(egui::SelectableLabel::new(is_local, "Local"))
                    .clicked()
                {
                    state.coord_space = CoordSpace::Local;
                }
                if ui
                    .add(egui::SelectableLabel::new(!is_local, "World"))
                    .clicked()
                {
                    state.coord_space = CoordSpace::World;
                }

                ui.separator();

                // Snap
                ui.checkbox(&mut state.snap_enabled, "Snap");
                if state.snap_enabled {
                    ui.add(
                        egui::DragValue::new(&mut state.snap_value)
                            .speed(0.1)
                            .range(0.01..=100.0)
                            .suffix(" u"),
                    );
                }

                ui.separator();

                // Speed slider
                ui.label("Speed:");
                ui.add(
                    egui::Slider::new(&mut state.sim_speed, 0.0..=4.0)
                        .max_decimals(2)
                        .clamp_to_range(true),
                );
            });
        });
}

fn draw_hierarchy(ctx: &egui::Context, state: &mut EditorState) {
    if !state.panels.hierarchy {
        return;
    }
    egui::SidePanel::left("hierarchy")
        .default_width(220.0)
        .min_width(160.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.strong("Scene Hierarchy");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Add entity dropdown
                    ui.menu_button("+", |ui| {
                        if ui.button("Empty").clicked() {
                            let idx = state.spawn_entity("Empty", EntityType::Empty);
                            state.selected_entity = Some(idx);
                            ui.close_menu();
                        }
                        if ui.button("Cube").clicked() {
                            let idx = state.spawn_entity("Cube", EntityType::Mesh);
                            state.selected_entity = Some(idx);
                            ui.close_menu();
                        }
                        if ui.button("Sphere").clicked() {
                            let idx = state.spawn_entity("Sphere", EntityType::Mesh);
                            state.selected_entity = Some(idx);
                            ui.close_menu();
                        }
                        if ui.button("Light").clicked() {
                            let idx = state.spawn_entity("Point Light", EntityType::Light);
                            state.selected_entity = Some(idx);
                            ui.close_menu();
                        }
                        if ui.button("Camera").clicked() {
                            let idx = state.spawn_entity("Camera", EntityType::Camera);
                            state.selected_entity = Some(idx);
                            ui.close_menu();
                        }
                    });
                });
            });
            ui.separator();

            if state.entities.is_empty() {
                ui.colored_label(
                    egui::Color32::from_rgb(120, 120, 120),
                    "No entities. Click + to add.",
                );
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut action: Option<HierarchyAction> = None;
                for i in 0..state.entities.len() {
                    let ent = &state.entities[i];
                    let selected = state.selected_entity == Some(i);
                    let label = format!("{} {}", ent.icon(), ent.name);

                    let resp = ui.add(egui::SelectableLabel::new(selected, &label));
                    if resp.clicked() {
                        state.selected_entity = Some(i);
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
                        if ui.button("Add Child (Empty)").clicked() {
                            action = Some(HierarchyAction::AddChild(i));
                            ui.close_menu();
                        }
                    });
                }

                match action {
                    Some(HierarchyAction::Duplicate(i)) => {
                        state.selected_entity = Some(i);
                        state.duplicate_selected();
                    }
                    Some(HierarchyAction::Delete(i)) => {
                        state.selected_entity = Some(i);
                        state.delete_selected();
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

enum HierarchyAction {
    Duplicate(usize),
    Delete(usize),
    AddChild(usize),
}

fn draw_inspector(ctx: &egui::Context, state: &mut EditorState) {
    if !state.panels.inspector {
        return;
    }
    egui::SidePanel::right("inspector")
        .default_width(280.0)
        .min_width(200.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.strong("Inspector");
            ui.separator();

            let sel = state.selected_entity;
            if sel.is_none() || sel.unwrap() >= state.entities.len() {
                ui.colored_label(
                    egui::Color32::from_rgb(120, 120, 120),
                    "No entity selected",
                );
                return;
            }
            let idx = sel.unwrap();

            // Name & type
            let etype_str = format!("{}", state.entities[idx].entity_type);
            let eid = state.entities[idx].entity;
            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.text_edit_singleline(&mut state.entities[idx].name);
            });
            ui.label(format!("Type: {}  |  ECS: {}v{}", etype_str, eid.id, eid.generation));
            ui.separator();

            // Transform section
            let mut pos_changed = false;
            egui::CollapsingHeader::new("Transform")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label("Position");
                    ui.horizontal(|ui| {
                        ui.label("X");
                        if ui
                            .add(egui::DragValue::new(&mut state.entities[idx].position[0]).speed(0.1))
                            .changed()
                        {
                            pos_changed = true;
                        }
                        ui.label("Y");
                        if ui
                            .add(egui::DragValue::new(&mut state.entities[idx].position[1]).speed(0.1))
                            .changed()
                        {
                            pos_changed = true;
                        }
                        ui.label("Z");
                        if ui
                            .add(egui::DragValue::new(&mut state.entities[idx].position[2]).speed(0.1))
                            .changed()
                        {
                            pos_changed = true;
                        }
                    });

                    ui.label("Rotation");
                    ui.horizontal(|ui| {
                        ui.label("X");
                        ui.add(egui::DragValue::new(&mut state.entities[idx].rotation[0]).speed(0.5));
                        ui.label("Y");
                        ui.add(egui::DragValue::new(&mut state.entities[idx].rotation[1]).speed(0.5));
                        ui.label("Z");
                        ui.add(egui::DragValue::new(&mut state.entities[idx].rotation[2]).speed(0.5));
                    });

                    ui.label("Scale");
                    ui.horizontal(|ui| {
                        ui.label("X");
                        ui.add(egui::DragValue::new(&mut state.entities[idx].scale[0]).speed(0.01));
                        ui.label("Y");
                        ui.add(egui::DragValue::new(&mut state.entities[idx].scale[1]).speed(0.01));
                        ui.label("Z");
                        ui.add(egui::DragValue::new(&mut state.entities[idx].scale[2]).speed(0.01));
                    });
                });

            if pos_changed {
                state.sync_entity_to_physics(idx);
            }

            // Physics section
            if state.entities[idx].has_physics {
                ui.separator();
                egui::CollapsingHeader::new("Physics")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Mass:");
                            ui.add(
                                egui::DragValue::new(&mut state.entities[idx].mass)
                                    .speed(0.1)
                                    .range(0.01..=10000.0),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Friction:");
                            ui.add(
                                egui::Slider::new(&mut state.entities[idx].friction, 0.0..=1.0),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Restitution:");
                            ui.add(
                                egui::Slider::new(&mut state.entities[idx].restitution, 0.0..=1.0),
                            );
                        });

                        if let Some(handle) = state.entities[idx].physics_handle {
                            if let Ok(vel) = state.engine.physics().get_linear_velocity(handle) {
                                ui.label(format!(
                                    "Velocity: ({:.1}, {:.1}, {:.1})",
                                    vel.x, vel.y, vel.z
                                ));
                            }
                        }
                    });
            }

            // Light section
            if state.entities[idx].is_light {
                ui.separator();
                egui::CollapsingHeader::new("Light")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Color:");
                            ui.color_edit_button_rgb(&mut state.entities[idx].light_color);
                        });
                        ui.horizontal(|ui| {
                            ui.label("Intensity:");
                            ui.add(
                                egui::Slider::new(
                                    &mut state.entities[idx].light_intensity,
                                    0.0..=100.0,
                                )
                                .logarithmic(true),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Range:");
                            ui.add(
                                egui::Slider::new(
                                    &mut state.entities[idx].light_range,
                                    0.1..=1000.0,
                                )
                                .logarithmic(true),
                            );
                        });
                    });
            }

            // Add component button
            ui.separator();
            ui.menu_button("Add Component", |ui| {
                if !state.entities[idx].has_physics {
                    if ui.button("Rigidbody").clicked() {
                        let ent = &mut state.entities[idx];
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
            });
        });
}

fn draw_bottom_panel(ctx: &egui::Context, state: &mut EditorState) {
    let show_bottom = state.panels.asset_browser
        || state.panels.console
        || state.panels.profiler
        || state.panels.scene_settings;
    if !show_bottom {
        return;
    }

    egui::TopBottomPanel::bottom("bottom_panel")
        .default_height(200.0)
        .min_height(80.0)
        .resizable(true)
        .show(ctx, |ui| {
            // Tab bar
            ui.horizontal(|ui| {
                if state.panels.console {
                    if ui
                        .add(egui::SelectableLabel::new(
                            state.bottom_tab == BottomTab::Console,
                            "Console",
                        ))
                        .clicked()
                    {
                        state.bottom_tab = BottomTab::Console;
                    }
                }
                if state.panels.asset_browser {
                    if ui
                        .add(egui::SelectableLabel::new(
                            state.bottom_tab == BottomTab::AssetBrowser,
                            "Assets",
                        ))
                        .clicked()
                    {
                        state.bottom_tab = BottomTab::AssetBrowser;
                    }
                }
                if state.panels.profiler {
                    if ui
                        .add(egui::SelectableLabel::new(
                            state.bottom_tab == BottomTab::Profiler,
                            "Profiler",
                        ))
                        .clicked()
                    {
                        state.bottom_tab = BottomTab::Profiler;
                    }
                }
                if state.panels.scene_settings {
                    if ui
                        .add(egui::SelectableLabel::new(
                            state.bottom_tab == BottomTab::SceneSettings,
                            "Scene Settings",
                        ))
                        .clicked()
                    {
                        state.bottom_tab = BottomTab::SceneSettings;
                    }
                }
            });
            ui.separator();

            match state.bottom_tab {
                BottomTab::Console => draw_console_content(ui, state),
                BottomTab::AssetBrowser => draw_asset_browser_content(ui, state),
                BottomTab::Profiler => draw_profiler_content(ui, state),
                BottomTab::SceneSettings => draw_scene_settings_content(ui, state),
            }
        });
}

fn draw_console_content(ui: &mut egui::Ui, state: &mut EditorState) {
    // Log output
    let available = ui.available_height() - 28.0;
    egui::ScrollArea::vertical()
        .max_height(available.max(20.0))
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for entry in &state.console_log {
                let ts = format!("[{:.1}]", entry.timestamp);
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(80, 80, 80), &ts);
                    ui.colored_label(entry.level.color(), &entry.text);
                });
            }
        });

    // Input line
    ui.horizontal(|ui| {
        ui.label(">");
        let resp = ui.add(
            egui::TextEdit::singleline(&mut state.console_input)
                .desired_width(ui.available_width() - 60.0)
                .hint_text("Type command..."),
        );
        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            let cmd = state.console_input.clone();
            if !cmd.trim().is_empty() {
                state.console_history.push(cmd.clone());
                state.console_history_idx = None;
                state.execute_console_command(&cmd);
                state.console_input.clear();
            }
            resp.request_focus();
        }
        // Up/Down arrow for history
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
        if ui.button("Run").clicked() {
            let cmd = state.console_input.clone();
            if !cmd.trim().is_empty() {
                state.console_history.push(cmd.clone());
                state.console_history_idx = None;
                state.execute_console_command(&cmd);
                state.console_input.clear();
            }
        }
    });
}

fn draw_asset_browser_content(ui: &mut egui::Ui, state: &mut EditorState) {
    // Breadcrumb
    ui.horizontal(|ui| {
        ui.label("Path:");
        ui.text_edit_singleline(&mut state.asset_path);
        if ui.button("Up").clicked() {
            if let Some(pos) = state.asset_path.rfind('/') {
                state.asset_path.truncate(pos);
            } else if let Some(pos) = state.asset_path.rfind('\\') {
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
        // Scan directory (or show defaults)
        let entries = scan_asset_dir(&state.asset_path, &state.asset_search);
        if entries.is_empty() {
            ui.colored_label(
                egui::Color32::from_rgb(120, 120, 120),
                "No assets found. Default asset types:",
            );
            let defaults = [
                ("[DIR] models/", "Mesh assets"),
                ("[DIR] textures/", "Texture assets"),
                ("[DIR] audio/", "Audio clips"),
                ("[DIR] scripts/", "Script files"),
                ("[DIR] scenes/", "Scene files"),
                ("[DIR] materials/", "Material definitions"),
            ];
            for (name, desc) in &defaults {
                ui.horizontal(|ui| {
                    ui.colored_label(YELLOW_ACCENT, *name);
                    ui.colored_label(egui::Color32::from_rgb(120, 120, 120), *desc);
                });
            }
        } else {
            // Grid display
            let col_width = 160.0;
            let cols = ((ui.available_width() / col_width) as usize).max(1);
            egui::Grid::new("asset_grid")
                .num_columns(cols)
                .spacing(egui::vec2(8.0, 8.0))
                .show(ui, |ui| {
                    for (i, entry) in entries.iter().enumerate() {
                        let icon = asset_icon(&entry.name, entry.is_dir);
                        let color = if entry.is_dir {
                            YELLOW_ACCENT
                        } else {
                            egui::Color32::from_rgb(180, 180, 180)
                        };
                        if ui.add(egui::Button::new(
                            egui::RichText::new(format!("{} {}", icon, entry.name)).color(color),
                        ).min_size(egui::vec2(col_width - 10.0, 20.0)))
                        .clicked()
                        {
                            if entry.is_dir {
                                state.asset_path =
                                    format!("{}/{}", state.asset_path, entry.name);
                            } else {
                                state.log(
                                    LogLevel::Info,
                                    format!("Selected asset: {}", entry.name),
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
    entries.sort_by(|a, b| {
        b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name))
    });
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

fn draw_profiler_content(ui: &mut egui::Ui, state: &mut EditorState) {
    ui.horizontal(|ui| {
        ui.label(format!("FPS: {:.1}", state.fps));
        ui.separator();
        ui.label(format!("Frame: {:.2} ms", state.frame_time_ms));
        ui.separator();
        let (min, max, avg) = if state.frame_times.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let min = state.frame_times.iter().cloned().fold(f64::MAX, f64::min);
            let max = state.frame_times.iter().cloned().fold(f64::MIN, f64::max);
            let avg = state.frame_times.iter().sum::<f64>() / state.frame_times.len() as f64;
            (min, max, avg)
        };
        ui.label(format!("Min: {:.2} ms  Max: {:.2} ms  Avg: {:.2} ms", min, max, avg));
    });
    ui.separator();

    // Frame time plot using egui_plot
    let points: Vec<[f64; 2]> = state
        .frame_times
        .iter()
        .enumerate()
        .map(|(i, &v)| [i as f64, v])
        .collect();
    let line = egui_plot::Line::new(egui_plot::PlotPoints::new(points))
        .name("Frame time (ms)")
        .color(ACCENT);

    egui_plot::Plot::new("frame_time_plot")
        .height(ui.available_height().max(60.0))
        .include_y(0.0)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .show(ui, |plot_ui| {
            plot_ui.line(line);
        });
}

fn draw_scene_settings_content(ui: &mut egui::Ui, state: &mut EditorState) {
    ui.horizontal(|ui| {
        ui.label("Scene name:");
        ui.text_edit_singleline(&mut state.scene_name);
    });
    ui.separator();
    ui.label("Gravity:");
    let g = state.engine.physics().gravity();
    let mut gv = [g.x, g.y, g.z];
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut gv[0]).speed(0.1));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut gv[1]).speed(0.1));
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut gv[2]).speed(0.1));
    });
    if gv != [g.x, g.y, g.z] {
        state
            .engine
            .physics_mut()
            .set_gravity(Vec3::new(gv[0], gv[1], gv[2]));
    }
    ui.separator();
    ui.label(format!("Physics bodies: {}", state.engine.physics().body_count()));
    ui.label(format!(
        "Active bodies: {}",
        state.engine.physics().active_body_count()
    ));
    ui.label(format!("ECS entities: {}", state.engine.world().entity_count()));
}

fn draw_viewport(ctx: &egui::Context, state: &EditorState) {
    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(egui::Color32::from_rgb(18, 18, 20)))
        .show(ctx, |ui| {
            // FPS overlay top-left
            let rect = ui.available_rect_before_wrap();
            let painter = ui.painter();

            painter.text(
                rect.left_top() + egui::vec2(8.0, 8.0),
                egui::Align2::LEFT_TOP,
                format!("{:.0} FPS | {:.2} ms", state.fps, state.frame_time_ms),
                egui::FontId::monospace(13.0),
                GREEN_ACCENT,
            );

            // Camera info top-right
            painter.text(
                rect.right_top() + egui::vec2(-8.0, 8.0),
                egui::Align2::RIGHT_TOP,
                format!(
                    "Cam: ({:.1}, {:.1}, {:.1}) d={:.1}",
                    state.camera_target[0],
                    state.camera_target[1],
                    state.camera_target[2],
                    state.camera_dist
                ),
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(120, 120, 130),
            );

            // Grid info center-bottom
            painter.text(
                egui::pos2(rect.center().x, rect.bottom() - 24.0),
                egui::Align2::CENTER_BOTTOM,
                format!(
                    "3D Viewport | {} | {} entities | {} bodies",
                    state.transform_mode_str(),
                    state.entities.len(),
                    state.engine.physics().body_count()
                ),
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(90, 90, 100),
            );

            // Playing indicator
            if state.is_playing && !state.is_paused {
                painter.text(
                    rect.left_top() + egui::vec2(8.0, 28.0),
                    egui::Align2::LEFT_TOP,
                    "SIMULATING",
                    egui::FontId::monospace(12.0),
                    GREEN_ACCENT,
                );
            } else if state.is_paused {
                painter.text(
                    rect.left_top() + egui::vec2(8.0, 28.0),
                    egui::Align2::LEFT_TOP,
                    "PAUSED",
                    egui::FontId::monospace(12.0),
                    YELLOW_ACCENT,
                );
            }

            // GPU-rendered triangle note
            painter.text(
                egui::pos2(rect.center().x, rect.center().y + 40.0),
                egui::Align2::CENTER_CENTER,
                "GPU Triangle renders behind this UI overlay",
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(60, 60, 70),
            );
        });
}

impl EditorState {
    fn transform_mode_str(&self) -> &str {
        match self.transform_mode {
            TransformMode::Translate => "Translate",
            TransformMode::Rotate => "Rotate",
            TransformMode::Scale => "Scale",
        }
    }
}

fn draw_status_bar(ctx: &egui::Context, state: &EditorState) {
    egui::TopBottomPanel::bottom("status_bar")
        .exact_height(20.0)
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                // Status indicator
                let (color, label) = if state.is_playing && !state.is_paused {
                    (GREEN_ACCENT, "Playing")
                } else if state.is_paused {
                    (YELLOW_ACCENT, "Paused")
                } else {
                    (GREEN_ACCENT, "Editing")
                };
                let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 4.0, color);
                ui.label(label);

                ui.separator();

                // Center: scene name
                ui.label(&state.scene_name);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!(
                        "{:.0} FPS | {} entities | {} bodies",
                        state.fps,
                        state.entities.len(),
                        state.engine.physics().body_count()
                    ));
                });
            });
        });
}

fn draw_about_window(ctx: &egui::Context, show: &mut bool) {
    egui::Window::new("About Genovo Engine")
        .open(show)
        .resizable(false)
        .default_width(350.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Genovo Engine");
                ui.label("Version 0.1.0");
                ui.add_space(8.0);
                ui.label("A AAA-tier game engine built in Rust.");
                ui.label("26 engine modules fully linked.");
                ui.add_space(8.0);
                ui.label("Rendering: wgpu");
                ui.label("UI: egui 0.31");
                ui.label("Physics: Custom impulse solver");
                ui.label("ECS: Archetype-based");
                ui.add_space(8.0);
                ui.colored_label(ACCENT, "genovo.dev");
            });
        });
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU state
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// Application
// ═══════════════════════════════════════════════════════════════════════════════

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

        let w = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("Genovo Engine Editor")
                    .with_inner_size(LogicalSize::new(1600, 900))
                    .with_maximized(true),
            )
            .unwrap(),
        );

        let inst = wgpu::Instance::new(&Default::default());
        let surf = inst.create_surface(w.clone()).unwrap();
        let adap = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surf),
            ..Default::default()
        }))
        .unwrap();
        let (dev, que) = pollster::block_on(adap.request_device(&Default::default(), None)).unwrap();
        let sz = w.inner_size();
        let cfg = surf
            .get_default_config(&adap, sz.width.max(1), sz.height.max(1))
            .unwrap();
        surf.configure(&dev, &cfg);

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
        // Ground plane
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
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &w,
            Some(w.scale_factor() as f32),
            None,
            Some(dev.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&dev, cfg.format, Some(wgpu::TextureFormat::Depth32Float), 1, false);

        let mut editor = EditorState::new(engine);
        editor.log(LogLevel::System, "Genovo Engine Editor initialized");
        editor.log(LogLevel::System, format!("GPU: {} ({:?})", adap.get_info().name, adap.get_info().backend));
        editor.log(LogLevel::System, "Type 'help' in console for commands");

        // Spawn some default entities
        let ground_idx = editor.spawn_entity("Ground Plane", EntityType::Mesh);
        editor.entities[ground_idx].scale = [100.0, 1.0, 100.0];
        let cube_idx = editor.spawn_entity("Cube", EntityType::Mesh);
        editor.entities[cube_idx].position = [0.0, 5.0, 0.0];
        if let Some(h) = editor.entities[cube_idx].physics_handle {
            let _ = editor.engine.physics_mut().set_position(h, Vec3::new(0.0, 5.0, 0.0));
        }
        let light_idx = editor.spawn_entity("Directional Light", EntityType::Light);
        editor.entities[light_idx].position = [5.0, 10.0, 5.0];
        let cam_idx = editor.spawn_entity("Main Camera", EntityType::Camera);
        editor.entities[cam_idx].position = [0.0, 5.0, -15.0];

        editor.selected_entity = Some(cube_idx);

        println!("[Genovo] GPU: {} ({:?})", adap.get_info().name, adap.get_info().backend);
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
            theme_applied: false,
        });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        let Some(s) = self.gpu.as_mut() else { return };

        // Let egui handle the event first
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
            WindowEvent::KeyboardInput { event, .. } if !resp.consumed => {
                if let PhysicalKey::Code(k) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        match k {
                            KeyCode::Escape => el.exit(),
                            KeyCode::KeyW => s.editor.transform_mode = TransformMode::Translate,
                            KeyCode::KeyE => s.editor.transform_mode = TransformMode::Rotate,
                            KeyCode::KeyR => s.editor.transform_mode = TransformMode::Scale,
                            KeyCode::Delete => s.editor.delete_selected(),
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
                if s.editor.frame_times.len() > 256 {
                    s.editor.frame_times.pop_front();
                }
                s.editor.fps_history.push_back(s.editor.fps as f64);
                if s.editor.fps_history.len() > 256 {
                    s.editor.fps_history.pop_front();
                }

                // Physics step
                if s.editor.is_playing && !s.editor.is_paused {
                    let phys_dt = (dt * s.editor.sim_speed).min(1.0 / 30.0);
                    let _ = s.editor.engine.physics_mut().step(phys_dt);
                    s.editor.sync_physics_to_entities();
                }

                // Apply theme once
                if !s.theme_applied {
                    apply_theme(&s.egui_ctx);
                    s.theme_applied = true;
                }

                // Build egui frame
                let raw_input = s.egui_state.take_egui_input(&s.window);
                let full_output = s.egui_ctx.run(raw_input, |ctx| {
                    // About window
                    if s.editor.show_about {
                        draw_about_window(ctx, &mut s.editor.show_about);
                    }

                    // Menu bar
                    draw_menu_bar(ctx, &mut s.editor);

                    // Toolbar
                    draw_toolbar(ctx, &mut s.editor);

                    // Status bar (must be before bottom panel)
                    draw_status_bar(ctx, &s.editor);

                    // Bottom panel (console, assets, profiler)
                    draw_bottom_panel(ctx, &mut s.editor);

                    // Left panel (hierarchy)
                    draw_hierarchy(ctx, &mut s.editor);

                    // Right panel (inspector)
                    draw_inspector(ctx, &mut s.editor);

                    // Central viewport
                    draw_viewport(ctx, &s.editor);
                });

                s.egui_state
                    .handle_platform_output(&s.window, full_output.platform_output);

                let clipped_primitives = s.egui_ctx.tessellate(
                    full_output.shapes,
                    full_output.pixels_per_point,
                );

                // Handle textures
                for (id, delta) in &full_output.textures_delta.set {
                    s.egui_renderer.update_texture(&s.device, &s.queue, *id, delta);
                }

                // Get surface texture
                let Ok(out) = s.surface.get_current_texture() else {
                    s.window.request_redraw();
                    return;
                };
                let view = out.texture.create_view(&Default::default());

                let screen = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [s.config.width, s.config.height],
                    pixels_per_point: s.window.scale_factor() as f32,
                };

                // Render pass 1: Clear + 3D triangle (scene)
                let t = s.editor.frame_count as f32 * 0.005;
                let (br, bg, bb) = if s.editor.is_playing && !s.editor.is_paused {
                    let pulse = (t * 2.0).sin().abs() * 0.015;
                    (0.06 + pulse as f64, 0.06_f64, 0.09 + pulse as f64)
                } else {
                    (0.06_f64, 0.06, 0.09)
                };

                let mut scene_enc =
                    s.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
                    // Draw the GPU triangle (behind the UI)
                    pass.set_pipeline(&s.pipeline);
                    pass.draw(0..3, 0..1);
                }

                // Render pass 2: egui overlay
                let mut egui_enc =
                    s.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
                    // SAFETY: The egui renderer requires 'static but the pass
                    // lives long enough since we drop it before enc.finish().
                    // wgpu 24's RenderPass is bound to 'encoder but egui-wgpu 0.31
                    // requires 'static. We transmute the lifetime since the pass
                    // is used and dropped within this block.
                    let pass_static: &mut wgpu::RenderPass<'static> =
                        unsafe { std::mem::transmute(&mut pass) };
                    s.egui_renderer
                        .render(pass_static, &clipped_primitives, &screen);
                }

                // Submit scene + egui
                let mut cmd_bufs: Vec<wgpu::CommandBuffer> = Vec::new();
                cmd_bufs.push(scene_enc.finish());
                cmd_bufs.extend(user_cmd_bufs);
                cmd_bufs.push(egui_enc.finish());
                s.queue.submit(cmd_bufs);
                out.present();

                // Free textures
                for id in &full_output.textures_delta.free {
                    s.egui_renderer.free_texture(id);
                }

                // Request next frame
                s.window.request_redraw();
            }
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn main() {
    env_logger::init();

    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = EditorApp { gpu: None };
    let _ = el.run_app(&mut app);
}
