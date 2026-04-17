//! Enhanced material editor: node graph for materials, real-time preview sphere,
//! material parameter animation, and material comparison.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialNodeId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialPinId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialWireId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialPinType {
    Float, Vec2, Vec3, Vec4, Color, Texture2D, TextureCube, Sampler, Bool, Int, Matrix,
}

impl MaterialPinType {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Float => "Float", Self::Vec2 => "Vec2", Self::Vec3 => "Vec3",
            Self::Vec4 => "Vec4", Self::Color => "Color", Self::Texture2D => "Tex2D",
            Self::TextureCube => "TexCube", Self::Sampler => "Sampler",
            Self::Bool => "Bool", Self::Int => "Int", Self::Matrix => "Matrix",
        }
    }

    pub fn color(&self) -> [f32; 4] {
        match self {
            Self::Float => [0.5, 0.8, 0.2, 1.0], Self::Vec2 => [0.2, 0.8, 0.5, 1.0],
            Self::Vec3 => [0.8, 0.8, 0.2, 1.0], Self::Vec4 => [0.8, 0.6, 0.2, 1.0],
            Self::Color => [1.0, 0.3, 0.3, 1.0], Self::Texture2D => [0.8, 0.2, 0.8, 1.0],
            Self::TextureCube => [0.6, 0.2, 0.8, 1.0], Self::Sampler => [0.5, 0.5, 0.5, 1.0],
            Self::Bool => [0.8, 0.2, 0.2, 1.0], Self::Int => [0.2, 0.5, 0.8, 1.0],
            Self::Matrix => [0.6, 0.6, 0.6, 1.0],
        }
    }

    pub fn is_compatible(&self, other: &Self) -> bool {
        if self == other { return true; }
        matches!(
            (self, other),
            (Self::Float, Self::Vec2 | Self::Vec3 | Self::Vec4)
                | (Self::Vec3, Self::Color) | (Self::Color, Self::Vec3)
                | (Self::Vec4, Self::Color) | (Self::Color, Self::Vec4)
                | (Self::Int, Self::Float) | (Self::Bool, Self::Float)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialNodeCategory { Math, Texture, Utility, Input, Output, Custom }

#[derive(Debug, Clone)]
pub struct MaterialPin {
    pub id: MaterialPinId,
    pub name: String,
    pub pin_type: MaterialPinType,
    pub is_output: bool,
    pub connected: bool,
    pub default_value: [f32; 4],
}

impl MaterialPin {
    pub fn input(id: MaterialPinId, name: impl Into<String>, pt: MaterialPinType) -> Self {
        Self { id, name: name.into(), pin_type: pt, is_output: false, connected: false, default_value: [0.0; 4] }
    }
    pub fn output(id: MaterialPinId, name: impl Into<String>, pt: MaterialPinType) -> Self {
        Self { id, name: name.into(), pin_type: pt, is_output: true, connected: false, default_value: [0.0; 4] }
    }
}

#[derive(Debug, Clone)]
pub struct MaterialNode {
    pub id: MaterialNodeId,
    pub title: String,
    pub category: MaterialNodeCategory,
    pub position: [f32; 2],
    pub inputs: Vec<MaterialPin>,
    pub outputs: Vec<MaterialPin>,
    pub preview_enabled: bool,
    pub error: Option<String>,
    pub collapsed: bool,
    pub comment: Option<String>,
    pub parameters: HashMap<String, [f32; 4]>,
    pub code_snippet: Option<String>,
}

impl MaterialNode {
    pub fn new(id: MaterialNodeId, title: impl Into<String>, cat: MaterialNodeCategory) -> Self {
        Self {
            id, title: title.into(), category: cat, position: [0.0; 2],
            inputs: Vec::new(), outputs: Vec::new(), preview_enabled: false,
            error: None, collapsed: false, comment: None,
            parameters: HashMap::new(), code_snippet: None,
        }
    }
    pub fn add_input(&mut self, pin: MaterialPin) { self.inputs.push(pin); }
    pub fn add_output(&mut self, pin: MaterialPin) { self.outputs.push(pin); }
}

#[derive(Debug, Clone)]
pub struct MaterialWire {
    pub id: MaterialWireId,
    pub source_node: MaterialNodeId,
    pub source_pin: MaterialPinId,
    pub target_node: MaterialNodeId,
    pub target_pin: MaterialPinId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewShape { Sphere, Cube, Cylinder, Plane, Torus, CustomMesh }

#[derive(Debug, Clone)]
pub struct MaterialPreview {
    pub shape: PreviewShape,
    pub rotation: [f32; 3],
    pub auto_rotate: bool,
    pub rotation_speed: f32,
    pub light_direction: [f32; 3],
    pub light_color: [f32; 3],
    pub light_intensity: f32,
    pub environment_map: Option<String>,
    pub background_color: [f32; 4],
    pub show_grid: bool,
    pub zoom: f32,
    pub resolution: u32,
}

impl Default for MaterialPreview {
    fn default() -> Self {
        Self {
            shape: PreviewShape::Sphere, rotation: [0.0; 3], auto_rotate: true,
            rotation_speed: 30.0, light_direction: [0.5, 1.0, 0.3],
            light_color: [1.0; 3], light_intensity: 1.0, environment_map: None,
            background_color: [0.15, 0.15, 0.15, 1.0], show_grid: true, zoom: 1.0,
            resolution: 256,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaterialParamAnim {
    pub parameter_name: String,
    pub node_id: MaterialNodeId,
    pub keyframes: Vec<(f32, [f32; 4])>,
    pub duration: f32,
    pub looping: bool,
    pub current_time: f32,
    pub playing: bool,
}

impl MaterialParamAnim {
    pub fn new(name: impl Into<String>, node: MaterialNodeId) -> Self {
        Self {
            parameter_name: name.into(), node_id: node, keyframes: Vec::new(),
            duration: 1.0, looping: true, current_time: 0.0, playing: false,
        }
    }

    pub fn add_keyframe(&mut self, time: f32, value: [f32; 4]) {
        self.keyframes.push((time, value));
        self.keyframes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if let Some(last) = self.keyframes.last() {
            self.duration = self.duration.max(last.0);
        }
    }

    pub fn evaluate(&self, time: f32) -> [f32; 4] {
        if self.keyframes.is_empty() { return [0.0; 4]; }
        if self.keyframes.len() == 1 { return self.keyframes[0].1; }
        let t = if self.looping { time % self.duration } else { time.min(self.duration) };
        for i in 0..self.keyframes.len() - 1 {
            let (t0, v0) = self.keyframes[i];
            let (t1, v1) = self.keyframes[i + 1];
            if t >= t0 && t <= t1 {
                let f = (t - t0) / (t1 - t0).max(0.0001);
                return [
                    v0[0] + (v1[0] - v0[0]) * f,
                    v0[1] + (v1[1] - v0[1]) * f,
                    v0[2] + (v1[2] - v0[2]) * f,
                    v0[3] + (v1[3] - v0[3]) * f,
                ];
            }
        }
        self.keyframes.last().unwrap().1
    }

    pub fn advance(&mut self, dt: f32) {
        if self.playing { self.current_time += dt; }
    }
}

#[derive(Debug, Clone)]
pub struct MaterialComparison {
    pub left_material: Option<String>,
    pub right_material: Option<String>,
    pub split_position: f32,
    pub sync_rotation: bool,
    pub differences: Vec<String>,
    pub active: bool,
}

impl MaterialComparison {
    pub fn new() -> Self {
        Self {
            left_material: None, right_material: None, split_position: 0.5,
            sync_rotation: true, differences: Vec::new(), active: false,
        }
    }
}

impl Default for MaterialComparison {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub enum MaterialEditorEvent {
    NodeAdded(MaterialNodeId),
    NodeRemoved(MaterialNodeId),
    WireAdded(MaterialWireId),
    WireRemoved(MaterialWireId),
    ParameterChanged(MaterialNodeId, String),
    PreviewUpdated,
    MaterialCompiled,
    CompileError(String),
    MaterialSaved(String),
}

pub struct MaterialEditorV2 {
    pub nodes: HashMap<MaterialNodeId, MaterialNode>,
    pub wires: Vec<MaterialWire>,
    pub preview: MaterialPreview,
    pub comparison: MaterialComparison,
    pub animations: Vec<MaterialParamAnim>,
    pub events: Vec<MaterialEditorEvent>,
    pub next_node_id: u64,
    pub next_pin_id: u64,
    pub next_wire_id: u64,
    pub material_name: String,
    pub material_path: String,
    pub zoom: f32,
    pub pan: [f32; 2],
    pub selected_nodes: Vec<MaterialNodeId>,
    pub show_preview: bool,
    pub show_code: bool,
    pub auto_compile: bool,
    pub compiled_shader: Option<String>,
    pub compile_errors: Vec<String>,
    pub dirty: bool,
}

impl MaterialEditorV2 {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(), wires: Vec::new(), preview: MaterialPreview::default(),
            comparison: MaterialComparison::new(), animations: Vec::new(),
            events: Vec::new(), next_node_id: 1, next_pin_id: 1, next_wire_id: 1,
            material_name: "New Material".to_string(), material_path: String::new(),
            zoom: 1.0, pan: [0.0; 2], selected_nodes: Vec::new(),
            show_preview: true, show_code: false, auto_compile: true,
            compiled_shader: None, compile_errors: Vec::new(), dirty: false,
        }
    }

    pub fn alloc_node_id(&mut self) -> MaterialNodeId {
        let id = MaterialNodeId(self.next_node_id); self.next_node_id += 1; id
    }
    pub fn alloc_pin_id(&mut self) -> MaterialPinId {
        let id = MaterialPinId(self.next_pin_id); self.next_pin_id += 1; id
    }
    pub fn alloc_wire_id(&mut self) -> MaterialWireId {
        let id = MaterialWireId(self.next_wire_id); self.next_wire_id += 1; id
    }

    pub fn add_node(&mut self, node: MaterialNode) -> MaterialNodeId {
        let id = node.id;
        self.nodes.insert(id, node);
        self.events.push(MaterialEditorEvent::NodeAdded(id));
        self.dirty = true;
        if self.auto_compile { self.compile(); }
        id
    }

    pub fn remove_node(&mut self, id: MaterialNodeId) {
        self.nodes.remove(&id);
        self.wires.retain(|w| w.source_node != id && w.target_node != id);
        self.selected_nodes.retain(|&n| n != id);
        self.events.push(MaterialEditorEvent::NodeRemoved(id));
        self.dirty = true;
    }

    pub fn add_wire(
        &mut self, src_node: MaterialNodeId, src_pin: MaterialPinId,
        dst_node: MaterialNodeId, dst_pin: MaterialPinId,
    ) -> Option<MaterialWireId> {
        let src_type = self.nodes.get(&src_node)
            .and_then(|n| n.outputs.iter().find(|p| p.id == src_pin))
            .map(|p| p.pin_type);
        let dst_type = self.nodes.get(&dst_node)
            .and_then(|n| n.inputs.iter().find(|p| p.id == dst_pin))
            .map(|p| p.pin_type);
        if let (Some(s), Some(d)) = (src_type, dst_type) {
            if !s.is_compatible(&d) { return None; }
        }
        let id = self.alloc_wire_id();
        self.wires.push(MaterialWire { id, source_node: src_node, source_pin: src_pin, target_node: dst_node, target_pin: dst_pin });
        self.events.push(MaterialEditorEvent::WireAdded(id));
        self.dirty = true;
        Some(id)
    }

    pub fn compile(&mut self) {
        self.compile_errors.clear();
        let mut code = String::from("// Generated Material Shader\n");
        code.push_str("struct MaterialOutput {\n");
        code.push_str("  vec3 base_color;\n  float metallic;\n  float roughness;\n");
        code.push_str("  vec3 normal;\n  vec3 emissive;\n  float opacity;\n};\n\n");
        code.push_str("void material_main(inout MaterialOutput o) {\n");
        for (_, node) in &self.nodes {
            code.push_str(&format!("  // Node: {} ({})\n", node.title, node.id.0));
            if let Some(snippet) = &node.code_snippet {
                code.push_str(&format!("  {}\n", snippet));
            }
        }
        code.push_str("}\n");
        self.compiled_shader = Some(code);
        self.events.push(MaterialEditorEvent::MaterialCompiled);
        self.dirty = false;
    }

    pub fn create_output_node(&mut self) -> MaterialNodeId {
        let id = self.alloc_node_id();
        let mut node = MaterialNode::new(id, "Material Output", MaterialNodeCategory::Output);
        node.add_input(MaterialPin::input(self.alloc_pin_id(), "Base Color", MaterialPinType::Color));
        node.add_input(MaterialPin::input(self.alloc_pin_id(), "Metallic", MaterialPinType::Float));
        node.add_input(MaterialPin::input(self.alloc_pin_id(), "Roughness", MaterialPinType::Float));
        node.add_input(MaterialPin::input(self.alloc_pin_id(), "Normal", MaterialPinType::Vec3));
        node.add_input(MaterialPin::input(self.alloc_pin_id(), "Emissive", MaterialPinType::Color));
        node.add_input(MaterialPin::input(self.alloc_pin_id(), "Opacity", MaterialPinType::Float));
        node.add_input(MaterialPin::input(self.alloc_pin_id(), "AO", MaterialPinType::Float));
        self.add_node(node); id
    }

    pub fn create_texture_node(&mut self, name: impl Into<String>) -> MaterialNodeId {
        let id = self.alloc_node_id();
        let mut node = MaterialNode::new(id, name, MaterialNodeCategory::Texture);
        node.add_input(MaterialPin::input(self.alloc_pin_id(), "UV", MaterialPinType::Vec2));
        node.add_output(MaterialPin::output(self.alloc_pin_id(), "RGB", MaterialPinType::Vec3));
        node.add_output(MaterialPin::output(self.alloc_pin_id(), "R", MaterialPinType::Float));
        node.add_output(MaterialPin::output(self.alloc_pin_id(), "G", MaterialPinType::Float));
        node.add_output(MaterialPin::output(self.alloc_pin_id(), "B", MaterialPinType::Float));
        node.add_output(MaterialPin::output(self.alloc_pin_id(), "A", MaterialPinType::Float));
        self.add_node(node); id
    }

    pub fn create_constant_node(&mut self, value: [f32; 4]) -> MaterialNodeId {
        let id = self.alloc_node_id();
        let mut node = MaterialNode::new(id, "Constant", MaterialNodeCategory::Input);
        node.parameters.insert("value".to_string(), value);
        node.add_output(MaterialPin::output(self.alloc_pin_id(), "Value", MaterialPinType::Vec4));
        self.add_node(node); id
    }

    pub fn create_math_node(&mut self, op: &str) -> MaterialNodeId {
        let id = self.alloc_node_id();
        let mut node = MaterialNode::new(id, op, MaterialNodeCategory::Math);
        node.add_input(MaterialPin::input(self.alloc_pin_id(), "A", MaterialPinType::Float));
        node.add_input(MaterialPin::input(self.alloc_pin_id(), "B", MaterialPinType::Float));
        node.add_output(MaterialPin::output(self.alloc_pin_id(), "Result", MaterialPinType::Float));
        node.code_snippet = Some(format!("float result = {} (a, b);", op.to_lowercase()));
        self.add_node(node); id
    }

    pub fn update(&mut self, dt: f32) {
        if self.preview.auto_rotate {
            self.preview.rotation[1] += self.preview.rotation_speed * dt;
            if self.preview.rotation[1] >= 360.0 { self.preview.rotation[1] -= 360.0; }
        }
        for anim in &mut self.animations { anim.advance(dt); }
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn wire_count(&self) -> usize { self.wires.len() }
    pub fn drain_events(&mut self) -> Vec<MaterialEditorEvent> { std::mem::take(&mut self.events) }
}

impl Default for MaterialEditorV2 {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_editor_basic() {
        let mut e = MaterialEditorV2::new();
        let _out = e.create_output_node();
        let _tex = e.create_texture_node("Albedo");
        assert_eq!(e.node_count(), 2);
    }

    #[test]
    fn pin_compatibility() {
        assert!(MaterialPinType::Float.is_compatible(&MaterialPinType::Float));
        assert!(MaterialPinType::Vec3.is_compatible(&MaterialPinType::Color));
        assert!(!MaterialPinType::Texture2D.is_compatible(&MaterialPinType::Float));
    }

    #[test]
    fn param_animation() {
        let mut anim = MaterialParamAnim::new("color", MaterialNodeId(1));
        anim.add_keyframe(0.0, [0.0, 0.0, 0.0, 1.0]);
        anim.add_keyframe(1.0, [1.0, 1.0, 1.0, 1.0]);
        let v = anim.evaluate(0.5);
        assert!((v[0] - 0.5).abs() < 0.01);
    }
}
