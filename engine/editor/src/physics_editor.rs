//! Physics configuration editor: collision matrix editor, physics material
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
