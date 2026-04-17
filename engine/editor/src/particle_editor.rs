//! Particle effect editor: emitter configuration, force field setup, preview
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
