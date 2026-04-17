#!/usr/bin/env python3
"""Generate remaining Genovo engine source files."""
import os

base = "C:/Users/USER/Downloads/game_engine/engine"

files = {}

files[f"{base}/editor/src/particle_editor.rs"] = r'''//! Particle effect editor: emitter configuration, force field setup, preview
//! with timeline, curve editors for particle properties.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmitterId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForceFieldId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitterShape { Point, Sphere, Box, Cone, Cylinder, Ring, Mesh, Edge }

impl EmitterShape {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Point => "Point", Self::Sphere => "Sphere", Self::Box => "Box",
            Self::Cone => "Cone", Self::Cylinder => "Cylinder", Self::Ring => "Ring",
            Self::Mesh => "Mesh", Self::Edge => "Edge",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode { Additive, AlphaBlend, Premultiplied, Opaque }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationSpace { Local, World }

#[derive(Debug, Clone)]
pub struct CurvePoint {
    pub time: f32,
    pub value: f32,
    pub tangent_in: f32,
    pub tangent_out: f32,
}

impl CurvePoint {
    pub fn new(time: f32, value: f32) -> Self {
        Self { time, value, tangent_in: 0.0, tangent_out: 0.0 }
    }
}

#[derive(Debug, Clone)]
pub struct ParticleCurve {
    pub name: String,
    pub points: Vec<CurvePoint>,
    pub min_value: f32,
    pub max_value: f32,
}

impl ParticleCurve {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), points: Vec::new(), min_value: 0.0, max_value: 1.0 }
    }

    pub fn constant(name: impl Into<String>, value: f32) -> Self {
        let mut c = Self::new(name);
        c.points.push(CurvePoint::new(0.0, value));
        c.points.push(CurvePoint::new(1.0, value));
        c
    }

    pub fn linear(name: impl Into<String>, start: f32, end: f32) -> Self {
        let mut c = Self::new(name);
        c.points.push(CurvePoint::new(0.0, start));
        c.points.push(CurvePoint::new(1.0, end));
        c
    }

    pub fn evaluate(&self, t: f32) -> f32 {
        if self.points.is_empty() { return 0.0; }
        if self.points.len() == 1 { return self.points[0].value; }
        let t = t.clamp(0.0, 1.0);
        if t <= self.points[0].time { return self.points[0].value; }
        if t >= self.points.last().unwrap().time { return self.points.last().unwrap().value; }
        for i in 0..self.points.len() - 1 {
            let a = &self.points[i];
            let b = &self.points[i + 1];
            if t >= a.time && t <= b.time {
                let f = (t - a.time) / (b.time - a.time).max(0.0001);
                return a.value + (b.value - a.value) * f;
            }
        }
        0.0
    }
}

#[derive(Debug, Clone)]
pub struct EmitterConfig {
    pub id: EmitterId,
    pub name: String,
    pub shape: EmitterShape,
    pub max_particles: u32,
    pub emission_rate: f32,
    pub burst_count: u32,
    pub burst_interval: f32,
    pub lifetime: ParticleCurve,
    pub speed: ParticleCurve,
    pub size: ParticleCurve,
    pub color_over_life: Vec<([f32; 4], f32)>,
    pub rotation_speed: ParticleCurve,
    pub gravity_modifier: f32,
    pub blend_mode: BlendMode,
    pub simulation_space: SimulationSpace,
    pub texture_path: Option<String>,
    pub sprite_sheet_cols: u32,
    pub sprite_sheet_rows: u32,
    pub enabled: bool,
    pub looping: bool,
    pub duration: f32,
    pub start_delay: f32,
    pub inherit_velocity: f32,
    pub shape_radius: f32,
    pub shape_angle: f32,
    pub shape_thickness: f32,
    pub noise_strength: f32,
    pub noise_frequency: f32,
    pub drag: f32,
    pub stretch_speed: f32,
    pub sort_mode: u32,
}

impl EmitterConfig {
    pub fn new(id: EmitterId, name: impl Into<String>) -> Self {
        Self {
            id, name: name.into(), shape: EmitterShape::Cone, max_particles: 1000,
            emission_rate: 50.0, burst_count: 0, burst_interval: 0.0,
            lifetime: ParticleCurve::constant("Lifetime", 2.0),
            speed: ParticleCurve::constant("Speed", 5.0),
            size: ParticleCurve::linear("Size", 0.1, 0.0),
            color_over_life: vec![([1.0, 1.0, 1.0, 1.0], 0.0), ([1.0, 1.0, 1.0, 0.0], 1.0)],
            rotation_speed: ParticleCurve::constant("RotSpeed", 0.0),
            gravity_modifier: 0.0, blend_mode: BlendMode::Additive,
            simulation_space: SimulationSpace::World, texture_path: None,
            sprite_sheet_cols: 1, sprite_sheet_rows: 1, enabled: true, looping: true,
            duration: 5.0, start_delay: 0.0, inherit_velocity: 0.0,
            shape_radius: 1.0, shape_angle: 25.0, shape_thickness: 1.0,
            noise_strength: 0.0, noise_frequency: 1.0, drag: 0.0,
            stretch_speed: 0.0, sort_mode: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceFieldType { Directional, Radial, Vortex, Turbulence, Drag }

#[derive(Debug, Clone)]
pub struct ForceField {
    pub id: ForceFieldId,
    pub name: String,
    pub field_type: ForceFieldType,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub strength: f32,
    pub radius: f32,
    pub falloff: f32,
    pub enabled: bool,
}

impl ForceField {
    pub fn new(id: ForceFieldId, name: impl Into<String>, ft: ForceFieldType) -> Self {
        Self {
            id, name: name.into(), field_type: ft, position: [0.0; 3],
            direction: [0.0, 1.0, 0.0], strength: 10.0, radius: 5.0,
            falloff: 1.0, enabled: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewState { Stopped, Playing, Paused }

#[derive(Debug, Clone)]
pub struct ParticlePreview {
    pub state: PreviewState,
    pub time: f32,
    pub speed: f32,
    pub particle_count: u32,
    pub show_bounds: bool,
    pub show_forces: bool,
    pub show_velocity: bool,
    pub show_grid: bool,
    pub background_color: [f32; 4],
    pub camera_distance: f32,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub loop_preview: bool,
}

impl Default for ParticlePreview {
    fn default() -> Self {
        Self {
            state: PreviewState::Playing, time: 0.0, speed: 1.0, particle_count: 0,
            show_bounds: false, show_forces: true, show_velocity: false, show_grid: true,
            background_color: [0.1, 0.1, 0.1, 1.0], camera_distance: 10.0,
            camera_yaw: 45.0, camera_pitch: 30.0, loop_preview: true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ParticleEditorEvent {
    EmitterAdded(EmitterId), EmitterRemoved(EmitterId), EmitterModified(EmitterId),
    ForceFieldAdded(ForceFieldId), ForceFieldRemoved(ForceFieldId),
    PreviewStateChanged(PreviewState), CurveEdited(EmitterId, String),
    EffectSaved(String), EffectLoaded(String),
}

pub struct ParticleEditorState {
    pub emitters: Vec<EmitterConfig>,
    pub force_fields: Vec<ForceField>,
    pub preview: ParticlePreview,
    pub selected_emitter: Option<EmitterId>,
    pub selected_force_field: Option<ForceFieldId>,
    pub events: Vec<ParticleEditorEvent>,
    pub next_emitter_id: u64,
    pub next_force_field_id: u64,
    pub effect_name: String,
    pub effect_path: String,
    pub dirty: bool,
    pub editing_curve: Option<(EmitterId, String)>,
    pub show_curve_editor: bool,
    pub show_color_gradient: bool,
    pub undo_stack: Vec<String>,
}

impl ParticleEditorState {
    pub fn new() -> Self {
        Self {
            emitters: Vec::new(), force_fields: Vec::new(),
            preview: ParticlePreview::default(), selected_emitter: None,
            selected_force_field: None, events: Vec::new(),
            next_emitter_id: 1, next_force_field_id: 1,
            effect_name: "New Effect".to_string(), effect_path: String::new(),
            dirty: false, editing_curve: None, show_curve_editor: true,
            show_color_gradient: true, undo_stack: Vec::new(),
        }
    }

    pub fn add_emitter(&mut self, name: impl Into<String>) -> EmitterId {
        let id = EmitterId(self.next_emitter_id);
        self.next_emitter_id += 1;
        self.emitters.push(EmitterConfig::new(id, name));
        self.events.push(ParticleEditorEvent::EmitterAdded(id));
        self.dirty = true;
        id
    }

    pub fn remove_emitter(&mut self, id: EmitterId) -> bool {
        let len = self.emitters.len();
        self.emitters.retain(|e| e.id != id);
        if self.emitters.len() < len {
            if self.selected_emitter == Some(id) { self.selected_emitter = None; }
            self.events.push(ParticleEditorEvent::EmitterRemoved(id));
            self.dirty = true;
            true
        } else { false }
    }

    pub fn add_force_field(&mut self, name: impl Into<String>, ft: ForceFieldType) -> ForceFieldId {
        let id = ForceFieldId(self.next_force_field_id);
        self.next_force_field_id += 1;
        self.force_fields.push(ForceField::new(id, name, ft));
        self.events.push(ParticleEditorEvent::ForceFieldAdded(id));
        id
    }

    pub fn remove_force_field(&mut self, id: ForceFieldId) -> bool {
        let len = self.force_fields.len();
        self.force_fields.retain(|f| f.id != id);
        self.force_fields.len() < len
    }

    pub fn play(&mut self) { self.preview.state = PreviewState::Playing; }
    pub fn pause(&mut self) { self.preview.state = PreviewState::Paused; }
    pub fn stop(&mut self) { self.preview.state = PreviewState::Stopped; self.preview.time = 0.0; }
    pub fn restart(&mut self) { self.preview.time = 0.0; self.preview.state = PreviewState::Playing; }

    pub fn update(&mut self, dt: f32) {
        if self.preview.state == PreviewState::Playing {
            self.preview.time += dt * self.preview.speed;
            let max_dur = self.emitters.iter().map(|e| e.duration).fold(0.0f32, f32::max);
            if self.preview.loop_preview && self.preview.time > max_dur && max_dur > 0.0 {
                self.preview.time = 0.0;
            }
        }
    }

    pub fn get_emitter(&self, id: EmitterId) -> Option<&EmitterConfig> {
        self.emitters.iter().find(|e| e.id == id)
    }
    pub fn get_emitter_mut(&mut self, id: EmitterId) -> Option<&mut EmitterConfig> {
        self.emitters.iter_mut().find(|e| e.id == id)
    }
    pub fn emitter_count(&self) -> usize { self.emitters.len() }
    pub fn drain_events(&mut self) -> Vec<ParticleEditorEvent> { std::mem::take(&mut self.events) }
}

impl Default for ParticleEditorState {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn particle_curve_eval() {
        let c = ParticleCurve::linear("test", 0.0, 10.0);
        assert!((c.evaluate(0.5) - 5.0).abs() < 0.01);
    }
    #[test]
    fn emitter_lifecycle() {
        let mut state = ParticleEditorState::new();
        let id = state.add_emitter("Smoke");
        assert_eq!(state.emitter_count(), 1);
        state.remove_emitter(id);
        assert_eq!(state.emitter_count(), 0);
    }
}
'''

files[f"{base}/editor/src/physics_editor.rs"] = r'''//! Physics configuration editor: collision matrix editor, physics material
//! editor, and joint constraint visualizer.

use std::collections::HashMap;
use std::fmt;

pub const MAX_COLLISION_LAYERS: usize = 32;

#[derive(Debug, Clone)]
pub struct CollisionLayer {
    pub index: usize,
    pub name: String,
    pub color: [f32; 4],
    pub enabled: bool,
    pub description: String,
}

impl CollisionLayer {
    pub fn new(index: usize, name: impl Into<String>) -> Self {
        let colors = [
            [1.0, 0.3, 0.3, 1.0], [0.3, 1.0, 0.3, 1.0], [0.3, 0.3, 1.0, 1.0],
            [1.0, 1.0, 0.3, 1.0], [1.0, 0.3, 1.0, 1.0], [0.3, 1.0, 1.0, 1.0],
            [1.0, 0.6, 0.3, 1.0], [0.6, 0.3, 1.0, 1.0],
        ];
        Self {
            index, name: name.into(), color: colors[index % colors.len()],
            enabled: true, description: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CollisionMatrix {
    pub layers: Vec<CollisionLayer>,
    pub matrix: [[bool; MAX_COLLISION_LAYERS]; MAX_COLLISION_LAYERS],
    pub layer_count: usize,
}

impl CollisionMatrix {
    pub fn new() -> Self {
        let mut matrix = [[false; MAX_COLLISION_LAYERS]; MAX_COLLISION_LAYERS];
        for i in 0..MAX_COLLISION_LAYERS { matrix[i][i] = true; }
        let default_layers = vec![
            CollisionLayer::new(0, "Default"), CollisionLayer::new(1, "Static"),
            CollisionLayer::new(2, "Dynamic"), CollisionLayer::new(3, "Trigger"),
            CollisionLayer::new(4, "Player"), CollisionLayer::new(5, "Enemy"),
            CollisionLayer::new(6, "Projectile"), CollisionLayer::new(7, "Environment"),
        ];
        Self { layers: default_layers, matrix, layer_count: 8 }
    }

    pub fn set_collision(&mut self, a: usize, b: usize, collides: bool) {
        if a < MAX_COLLISION_LAYERS && b < MAX_COLLISION_LAYERS {
            self.matrix[a][b] = collides;
            self.matrix[b][a] = collides;
        }
    }

    pub fn get_collision(&self, a: usize, b: usize) -> bool {
        if a < MAX_COLLISION_LAYERS && b < MAX_COLLISION_LAYERS { self.matrix[a][b] } else { false }
    }

    pub fn add_layer(&mut self, name: impl Into<String>) -> usize {
        if self.layer_count >= MAX_COLLISION_LAYERS { return self.layer_count - 1; }
        let idx = self.layer_count;
        self.layers.push(CollisionLayer::new(idx, name));
        self.layer_count += 1;
        idx
    }

    pub fn enable_all_for_layer(&mut self, layer: usize) {
        for i in 0..self.layer_count { self.set_collision(layer, i, true); }
    }
    pub fn disable_all_for_layer(&mut self, layer: usize) {
        for i in 0..self.layer_count { self.set_collision(layer, i, false); }
    }
}

impl Default for CollisionMatrix {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysMaterialId(pub u64);

#[derive(Debug, Clone)]
pub struct PhysicsMaterialDef {
    pub id: PhysMaterialId,
    pub name: String,
    pub static_friction: f32,
    pub dynamic_friction: f32,
    pub restitution: f32,
    pub density: f32,
    pub friction_combine: CombineMode,
    pub restitution_combine: CombineMode,
    pub color: [f32; 4],
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineMode { Average, Min, Max, Multiply }

impl PhysicsMaterialDef {
    pub fn new(id: PhysMaterialId, name: impl Into<String>) -> Self {
        Self {
            id, name: name.into(), static_friction: 0.5, dynamic_friction: 0.4,
            restitution: 0.3, density: 1.0, friction_combine: CombineMode::Average,
            restitution_combine: CombineMode::Average, color: [0.5, 0.5, 0.5, 1.0],
            description: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointType { Fixed, Hinge, Slider, Ball, Spring, Distance, Cone, D6 }

impl JointType {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Fixed => "Fixed", Self::Hinge => "Hinge", Self::Slider => "Slider",
            Self::Ball => "Ball", Self::Spring => "Spring", Self::Distance => "Distance",
            Self::Cone => "Cone", Self::D6 => "6-DOF",
        }
    }
}

#[derive(Debug, Clone)]
pub struct JointConstraintVis {
    pub id: u64,
    pub joint_type: JointType,
    pub body_a: u64,
    pub body_b: u64,
    pub anchor_a: [f32; 3],
    pub anchor_b: [f32; 3],
    pub axis: [f32; 3],
    pub limit_min: f32,
    pub limit_max: f32,
    pub spring_stiffness: f32,
    pub spring_damping: f32,
    pub break_force: f32,
    pub break_torque: f32,
    pub enable_collision: bool,
    pub color: [f32; 4],
    pub show_limits: bool,
    pub show_axis: bool,
}

impl JointConstraintVis {
    pub fn new(id: u64, jt: JointType, body_a: u64, body_b: u64) -> Self {
        Self {
            id, joint_type: jt, body_a, body_b, anchor_a: [0.0; 3], anchor_b: [0.0; 3],
            axis: [0.0, 1.0, 0.0], limit_min: -90.0, limit_max: 90.0,
            spring_stiffness: 0.0, spring_damping: 0.0, break_force: f32::MAX,
            break_torque: f32::MAX, enable_collision: false,
            color: [0.8, 0.8, 0.2, 1.0], show_limits: true, show_axis: true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PhysicsEditorEvent {
    CollisionMatrixChanged(usize, usize, bool), LayerAdded(usize),
    MaterialCreated(PhysMaterialId), MaterialModified(PhysMaterialId),
    MaterialRemoved(PhysMaterialId), JointCreated(u64), JointModified(u64),
    JointRemoved(u64),
}

pub struct PhysicsEditorState {
    pub collision_matrix: CollisionMatrix,
    pub materials: HashMap<PhysMaterialId, PhysicsMaterialDef>,
    pub joints: Vec<JointConstraintVis>,
    pub events: Vec<PhysicsEditorEvent>,
    pub next_material_id: u64,
    pub next_joint_id: u64,
    pub selected_material: Option<PhysMaterialId>,
    pub selected_joint: Option<u64>,
    pub show_collision_matrix: bool,
    pub show_materials: bool,
    pub show_joints: bool,
    pub gravity: [f32; 3],
    pub fixed_timestep: f32,
    pub solver_iterations: u32,
    pub velocity_iterations: u32,
    pub enable_ccd: bool,
    pub enable_sleeping: bool,
    pub sleep_threshold: f32,
}

impl PhysicsEditorState {
    pub fn new() -> Self {
        Self {
            collision_matrix: CollisionMatrix::new(), materials: HashMap::new(),
            joints: Vec::new(), events: Vec::new(), next_material_id: 1,
            next_joint_id: 1, selected_material: None, selected_joint: None,
            show_collision_matrix: true, show_materials: true, show_joints: true,
            gravity: [0.0, -9.81, 0.0], fixed_timestep: 1.0 / 60.0,
            solver_iterations: 8, velocity_iterations: 3, enable_ccd: true,
            enable_sleeping: true, sleep_threshold: 0.005,
        }
    }

    pub fn create_material(&mut self, name: impl Into<String>) -> PhysMaterialId {
        let id = PhysMaterialId(self.next_material_id);
        self.next_material_id += 1;
        self.materials.insert(id, PhysicsMaterialDef::new(id, name));
        self.events.push(PhysicsEditorEvent::MaterialCreated(id));
        id
    }

    pub fn remove_material(&mut self, id: PhysMaterialId) -> bool {
        if self.materials.remove(&id).is_some() {
            self.events.push(PhysicsEditorEvent::MaterialRemoved(id));
            true
        } else { false }
    }

    pub fn create_joint(&mut self, jt: JointType, a: u64, b: u64) -> u64 {
        let id = self.next_joint_id;
        self.next_joint_id += 1;
        self.joints.push(JointConstraintVis::new(id, jt, a, b));
        self.events.push(PhysicsEditorEvent::JointCreated(id));
        id
    }

    pub fn remove_joint(&mut self, id: u64) -> bool {
        let len = self.joints.len();
        self.joints.retain(|j| j.id != id);
        self.joints.len() < len
    }

    pub fn material_count(&self) -> usize { self.materials.len() }
    pub fn joint_count(&self) -> usize { self.joints.len() }
    pub fn drain_events(&mut self) -> Vec<PhysicsEditorEvent> { std::mem::take(&mut self.events) }
}

impl Default for PhysicsEditorState {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn collision_matrix() {
        let mut m = CollisionMatrix::new();
        m.set_collision(0, 1, true);
        assert!(m.get_collision(0, 1));
        assert!(m.get_collision(1, 0)); // symmetric
    }
    #[test]
    fn physics_material() {
        let mut state = PhysicsEditorState::new();
        let id = state.create_material("Rubber");
        assert_eq!(state.material_count(), 1);
        state.remove_material(id);
        assert_eq!(state.material_count(), 0);
    }
}
'''

files[f"{base}/editor/src/settings_editor.rs"] = r'''//! Settings/preferences editor: categorized settings, search, reset to defaults,
//! import/export settings.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum SettingValue {
    Bool(bool), Int(i32), Float(f32), String(String),
    Color([f32; 4]), Vec2([f32; 2]), Vec3([f32; 3]),
    Enum(usize, Vec<String>), Path(String), KeyBinding(String),
}

impl SettingValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool", Self::Int(_) => "int", Self::Float(_) => "float",
            Self::String(_) => "string", Self::Color(_) => "color",
            Self::Vec2(_) => "vec2", Self::Vec3(_) => "vec3",
            Self::Enum(_, _) => "enum", Self::Path(_) => "path",
            Self::KeyBinding(_) => "keybinding",
        }
    }
    pub fn as_bool(&self) -> Option<bool> { match self { Self::Bool(v) => Some(*v), _ => None } }
    pub fn as_int(&self) -> Option<i32> { match self { Self::Int(v) => Some(*v), _ => None } }
    pub fn as_float(&self) -> Option<f32> { match self { Self::Float(v) => Some(*v), _ => None } }
    pub fn as_string(&self) -> Option<&str> { match self { Self::String(v) => Some(v), Self::Path(v) => Some(v), _ => None } }
}

impl fmt::Display for SettingValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{}", v), Self::Int(v) => write!(f, "{}", v),
            Self::Float(v) => write!(f, "{:.3}", v), Self::String(v) => write!(f, "{}", v),
            Self::Color(c) => write!(f, "({:.2},{:.2},{:.2},{:.2})", c[0], c[1], c[2], c[3]),
            Self::Vec2(v) => write!(f, "({:.2},{:.2})", v[0], v[1]),
            Self::Vec3(v) => write!(f, "({:.2},{:.2},{:.2})", v[0], v[1], v[2]),
            Self::Enum(idx, opts) => write!(f, "{}", opts.get(*idx).map(|s| s.as_str()).unwrap_or("?")),
            Self::Path(v) => write!(f, "{}", v),
            Self::KeyBinding(v) => write!(f, "{}", v),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SettingDef {
    pub key: String,
    pub display_name: String,
    pub category: String,
    pub subcategory: Option<String>,
    pub value: SettingValue,
    pub default_value: SettingValue,
    pub description: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub requires_restart: bool,
    pub hidden: bool,
    pub modified: bool,
    pub tags: Vec<String>,
}

impl SettingDef {
    pub fn new(key: impl Into<String>, name: impl Into<String>, cat: impl Into<String>, value: SettingValue) -> Self {
        let v = value.clone();
        Self {
            key: key.into(), display_name: name.into(), category: cat.into(),
            subcategory: None, value, default_value: v, description: String::new(),
            min: None, max: None, step: None, requires_restart: false,
            hidden: false, modified: false, tags: Vec::new(),
        }
    }
    pub fn with_range(mut self, min: f64, max: f64) -> Self { self.min = Some(min); self.max = Some(max); self }
    pub fn with_description(mut self, desc: impl Into<String>) -> Self { self.description = desc.into(); self }
    pub fn with_restart(mut self) -> Self { self.requires_restart = true; self }
    pub fn is_default(&self) -> bool { self.value == self.default_value }
    pub fn reset(&mut self) { self.value = self.default_value.clone(); self.modified = false; }
}

#[derive(Debug, Clone)]
pub enum SettingsEvent {
    ValueChanged(String, SettingValue),
    CategorySelected(String),
    SearchChanged(String),
    ResetToDefaults(Option<String>),
    Imported(String),
    Exported(String),
    Saved,
}

pub struct SettingsEditorState {
    pub settings: Vec<SettingDef>,
    pub categories: Vec<String>,
    pub selected_category: Option<String>,
    pub search_text: String,
    pub events: Vec<SettingsEvent>,
    pub show_modified_only: bool,
    pub show_advanced: bool,
    pub dirty: bool,
    pub profiles: HashMap<String, Vec<(String, SettingValue)>>,
    pub active_profile: String,
}

impl SettingsEditorState {
    pub fn new() -> Self {
        Self {
            settings: Vec::new(), categories: Vec::new(),
            selected_category: None, search_text: String::new(),
            events: Vec::new(), show_modified_only: false,
            show_advanced: false, dirty: false,
            profiles: HashMap::new(), active_profile: "Default".to_string(),
        }
    }

    pub fn add_setting(&mut self, setting: SettingDef) {
        if !self.categories.contains(&setting.category) {
            self.categories.push(setting.category.clone());
            self.categories.sort();
        }
        self.settings.push(setting);
    }

    pub fn set_value(&mut self, key: &str, value: SettingValue) {
        if let Some(s) = self.settings.iter_mut().find(|s| s.key == key) {
            s.value = value.clone();
            s.modified = s.value != s.default_value;
            self.dirty = true;
            self.events.push(SettingsEvent::ValueChanged(key.to_string(), value));
        }
    }

    pub fn get_value(&self, key: &str) -> Option<&SettingValue> {
        self.settings.iter().find(|s| s.key == key).map(|s| &s.value)
    }

    pub fn reset_to_defaults(&mut self, category: Option<&str>) {
        for s in &mut self.settings {
            if category.map_or(true, |c| s.category == c) { s.reset(); }
        }
        self.dirty = true;
        self.events.push(SettingsEvent::ResetToDefaults(category.map(|s| s.to_string())));
    }

    pub fn select_category(&mut self, cat: impl Into<String>) {
        let cat = cat.into();
        self.selected_category = Some(cat.clone());
        self.events.push(SettingsEvent::CategorySelected(cat));
    }

    pub fn search(&mut self, text: impl Into<String>) {
        self.search_text = text.into();
        self.events.push(SettingsEvent::SearchChanged(self.search_text.clone()));
    }

    pub fn filtered_settings(&self) -> Vec<&SettingDef> {
        self.settings.iter().filter(|s| {
            if s.hidden && !self.show_advanced { return false; }
            if self.show_modified_only && !s.modified { return false; }
            if let Some(ref cat) = self.selected_category { if s.category != *cat { return false; } }
            if !self.search_text.is_empty() {
                let lower = self.search_text.to_lowercase();
                if !s.display_name.to_lowercase().contains(&lower)
                    && !s.key.to_lowercase().contains(&lower)
                    && !s.description.to_lowercase().contains(&lower) { return false; }
            }
            true
        }).collect()
    }

    pub fn modified_count(&self) -> usize { self.settings.iter().filter(|s| s.modified).count() }
    pub fn setting_count(&self) -> usize { self.settings.len() }
    pub fn category_count(&self) -> usize { self.categories.len() }

    pub fn export_to_string(&self) -> String {
        let mut out = String::new();
        for s in &self.settings {
            out.push_str(&format!("{}={}\n", s.key, s.value));
        }
        out
    }

    pub fn save_profile(&mut self, name: impl Into<String>) {
        let name = name.into();
        let values: Vec<(String, SettingValue)> = self.settings.iter()
            .map(|s| (s.key.clone(), s.value.clone())).collect();
        self.profiles.insert(name, values);
    }

    pub fn load_profile(&mut self, name: &str) -> bool {
        if let Some(values) = self.profiles.get(name).cloned() {
            for (key, val) in values { self.set_value(&key, val); }
            self.active_profile = name.to_string();
            true
        } else { false }
    }

    pub fn drain_events(&mut self) -> Vec<SettingsEvent> { std::mem::take(&mut self.events) }
}

impl Default for SettingsEditorState {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn settings_basic() {
        let mut editor = SettingsEditorState::new();
        editor.add_setting(SettingDef::new("render.vsync", "VSync", "Rendering", SettingValue::Bool(true)));
        editor.add_setting(SettingDef::new("render.fov", "FOV", "Rendering", SettingValue::Float(90.0)).with_range(60.0, 120.0));
        assert_eq!(editor.setting_count(), 2);
        assert_eq!(editor.category_count(), 1);
        editor.set_value("render.fov", SettingValue::Float(100.0));
        assert_eq!(editor.modified_count(), 1);
        editor.reset_to_defaults(None);
        assert_eq!(editor.modified_count(), 0);
    }
    #[test]
    fn search_filter() {
        let mut editor = SettingsEditorState::new();
        editor.add_setting(SettingDef::new("audio.volume", "Volume", "Audio", SettingValue::Float(1.0)));
        editor.add_setting(SettingDef::new("render.fov", "FOV", "Rendering", SettingValue::Float(90.0)));
        editor.search("vol");
        let filtered = editor.filtered_settings();
        assert_eq!(filtered.len(), 1);
    }
}
'''

# Now write all remaining files with substantial content
for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='\n') as f:
        f.write(content)
    lines = content.count('\n') + 1
    print(f"Wrote {os.path.basename(path)} ({lines} lines)")

print("Batch done")
