// =============================================================================
// Genovo Engine - Material Editor
// =============================================================================
//
// A node-graph-based material creation tool. Uses the `NodeGraph` system
// internally, providing material-specific node types, live preview, and
// WGSL shader code generation from the graph.

use crate::node_graph::{
    built_in_node_library, GraphError, GraphNode, NodeGraph, NodeId, NodeTypeDescriptor,
    PinDataType, PinId, Value,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Material Shader Code
// ---------------------------------------------------------------------------

/// Generated shader code from a material graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialShaderCode {
    /// WGSL vertex shader source.
    pub vertex_shader: String,
    /// WGSL fragment shader source.
    pub fragment_shader: String,
    /// List of texture bindings required.
    pub texture_bindings: Vec<TextureBinding>,
    /// List of uniform parameters.
    pub uniform_parameters: Vec<UniformParameter>,
    /// Compilation errors, if any.
    pub errors: Vec<String>,
    /// Whether the compilation was successful.
    pub success: bool,
}

/// A texture binding required by the material.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextureBinding {
    /// Binding group index.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Parameter name.
    pub name: String,
    /// Default texture asset UUID.
    pub default_texture: Option<Uuid>,
}

/// A uniform parameter exposed by the material.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformParameter {
    /// Parameter name.
    pub name: String,
    /// Data type.
    pub data_type: PinDataType,
    /// Default value.
    pub default_value: Value,
}

// ---------------------------------------------------------------------------
// Material Preview
// ---------------------------------------------------------------------------

/// Settings for the material preview display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialPreviewSettings {
    /// Preview mesh shape.
    pub mesh: PreviewMesh,
    /// Environment lighting mode.
    pub environment: PreviewEnvironment,
    /// Whether to rotate the preview.
    pub auto_rotate: bool,
    /// Rotation speed (degrees per second).
    pub rotation_speed: f32,
    /// Preview resolution.
    pub resolution: u32,
    /// Background color (RGBA).
    pub background_color: [f32; 4],
}

impl Default for MaterialPreviewSettings {
    fn default() -> Self {
        Self {
            mesh: PreviewMesh::Sphere,
            environment: PreviewEnvironment::StudioLighting,
            auto_rotate: true,
            rotation_speed: 30.0,
            resolution: 256,
            background_color: [0.15, 0.15, 0.15, 1.0],
        }
    }
}

/// Mesh used for material preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreviewMesh {
    Sphere,
    Cube,
    Cylinder,
    Torus,
    Plane,
    Custom,
}

/// Environment for material preview lighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreviewEnvironment {
    StudioLighting,
    OutdoorHDRI,
    IndoorHDRI,
    SingleDirectional,
    Unlit,
}

// ---------------------------------------------------------------------------
// Material Editor
// ---------------------------------------------------------------------------

/// The material editor panel, backed by a node graph.
#[derive(Debug, Clone)]
pub struct MaterialEditor {
    /// The underlying node graph.
    pub graph: NodeGraph,
    /// Material name.
    pub name: String,
    /// Material asset UUID.
    pub asset_id: Uuid,
    /// Preview settings.
    pub preview: MaterialPreviewSettings,
    /// Whether the material has unsaved changes.
    pub dirty: bool,
    /// Currently selected nodes.
    pub selected_nodes: Vec<NodeId>,
    /// Node library filtered for material nodes.
    pub node_palette: Vec<NodeTypeDescriptor>,
    /// The output node ID (the material output is special).
    pub output_node: Option<NodeId>,
    /// Search query for the node palette.
    pub palette_search: String,
    /// Compiled shader code (cached from last compile).
    compiled_shader: Option<MaterialShaderCode>,
    /// Whether auto-compile is enabled.
    pub auto_compile: bool,
    /// Current preview rotation angle.
    pub preview_rotation: f32,
}

impl MaterialEditor {
    /// Create a new material editor with a default material output node.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let mut graph = NodeGraph::new(format!("{} Graph", name));

        // Create the material output node.
        let library = built_in_node_library();
        let output_desc = library
            .iter()
            .find(|d| d.type_id == "material_output")
            .expect("Material output node must exist in library");

        let output = GraphNode::from_descriptor(output_desc, [600.0, 200.0]);
        let output_id = graph.add_node(output);

        // Filter library to material-relevant categories.
        let material_categories = [
            "Material", "Texture", "Constant", "Math", "Vector", "Color",
            "UV", "Shading",
        ];
        let node_palette: Vec<NodeTypeDescriptor> = library
            .into_iter()
            .filter(|d| material_categories.contains(&d.category.as_str()))
            .collect();

        Self {
            graph,
            name,
            asset_id: Uuid::new_v4(),
            preview: MaterialPreviewSettings::default(),
            dirty: false,
            selected_nodes: Vec::new(),
            node_palette,
            output_node: Some(output_id),
            palette_search: String::new(),
            compiled_shader: None,
            auto_compile: true,
            preview_rotation: 0.0,
        }
    }

    // --- Node operations ---

    /// Add a node from the palette by type_id.
    pub fn add_node(&mut self, type_id: &str, position: [f32; 2]) -> Option<NodeId> {
        let desc = self.node_palette.iter().find(|d| d.type_id == type_id)?;
        let node = GraphNode::from_descriptor(desc, position);
        let id = self.graph.add_node(node);
        self.dirty = true;
        if self.auto_compile {
            self.compile();
        }
        Some(id)
    }

    /// Remove a node (cannot remove the output node).
    pub fn remove_node(&mut self, node_id: NodeId) -> bool {
        if self.output_node == Some(node_id) {
            return false;
        }
        self.graph.remove_node(node_id);
        self.selected_nodes.retain(|id| *id != node_id);
        self.dirty = true;
        if self.auto_compile {
            self.compile();
        }
        true
    }

    /// Connect two pins.
    pub fn connect(
        &mut self,
        from_node: NodeId,
        from_pin: PinId,
        to_node: NodeId,
        to_pin: PinId,
    ) -> Result<(), GraphError> {
        self.graph.connect(from_node, from_pin, to_node, to_pin)?;
        self.dirty = true;
        if self.auto_compile {
            self.compile();
        }
        Ok(())
    }

    /// Disconnect a pin.
    pub fn disconnect_pin(&mut self, pin_id: PinId) {
        self.graph.disconnect_pin(pin_id);
        self.dirty = true;
        if self.auto_compile {
            self.compile();
        }
    }

    // --- Selection ---

    /// Select a node.
    pub fn select_node(&mut self, node_id: NodeId) {
        self.selected_nodes.clear();
        self.selected_nodes.push(node_id);
    }

    /// Add a node to the selection.
    pub fn add_to_selection(&mut self, node_id: NodeId) {
        if !self.selected_nodes.contains(&node_id) {
            self.selected_nodes.push(node_id);
        }
    }

    /// Clear node selection.
    pub fn clear_selection(&mut self) {
        self.selected_nodes.clear();
    }

    /// Delete selected nodes.
    pub fn delete_selected(&mut self) {
        let to_delete: Vec<NodeId> = self
            .selected_nodes
            .iter()
            .filter(|id| self.output_node != Some(**id))
            .copied()
            .collect();
        for id in to_delete {
            self.graph.remove_node(id);
        }
        self.selected_nodes.clear();
        self.dirty = true;
        if self.auto_compile {
            self.compile();
        }
    }

    /// Duplicate selected nodes.
    pub fn duplicate_selected(&mut self) {
        let ids: Vec<NodeId> = self
            .selected_nodes
            .iter()
            .filter(|id| self.output_node != Some(**id))
            .copied()
            .collect();
        let new_ids = self.graph.duplicate_nodes(&ids);
        self.selected_nodes = new_ids;
        self.dirty = true;
    }

    // --- Node palette ---

    /// Get filtered node palette entries.
    pub fn filtered_palette(&self) -> Vec<&NodeTypeDescriptor> {
        if self.palette_search.is_empty() {
            return self.node_palette.iter().collect();
        }
        let query = self.palette_search.to_lowercase();
        self.node_palette
            .iter()
            .filter(|d| {
                d.display_name.to_lowercase().contains(&query)
                    || d.category.to_lowercase().contains(&query)
                    || d.type_id.to_lowercase().contains(&query)
            })
            .collect()
    }

    /// Get palette entries grouped by category.
    pub fn palette_by_category(&self) -> Vec<(String, Vec<&NodeTypeDescriptor>)> {
        let filtered = self.filtered_palette();
        let mut categories: Vec<(String, Vec<&NodeTypeDescriptor>)> = Vec::new();

        for desc in filtered {
            if let Some(cat) = categories.iter_mut().find(|(c, _)| *c == desc.category) {
                cat.1.push(desc);
            } else {
                categories.push((desc.category.clone(), vec![desc]));
            }
        }

        categories.sort_by(|a, b| a.0.cmp(&b.0));
        categories
    }

    // --- Compilation ---

    /// Compile the material graph into shader code.
    pub fn compile(&mut self) -> &MaterialShaderCode {
        let code = compile_material(&self.graph, self.output_node);
        self.compiled_shader = Some(code);
        self.compiled_shader.as_ref().unwrap()
    }

    /// Get the last compiled shader code.
    pub fn compiled_shader(&self) -> Option<&MaterialShaderCode> {
        self.compiled_shader.as_ref()
    }

    /// Whether the compiled shader has errors.
    pub fn has_errors(&self) -> bool {
        self.compiled_shader
            .as_ref()
            .map(|s| !s.success)
            .unwrap_or(false)
    }

    // --- Preview ---

    /// Update the preview rotation.
    pub fn update_preview(&mut self, dt: f32) {
        if self.preview.auto_rotate {
            self.preview_rotation += self.preview.rotation_speed * dt;
            if self.preview_rotation > 360.0 {
                self.preview_rotation -= 360.0;
            }
        }
    }

    // --- Serialization ---

    /// Serialize the material editor state to JSON.
    pub fn to_json(&self) -> Result<String, GraphError> {
        let data = MaterialEditorData {
            name: self.name.clone(),
            asset_id: self.asset_id,
            graph_json: self.graph.to_json()?,
            preview: self.preview.clone(),
        };
        serde_json::to_string_pretty(&data)
            .map_err(|e| GraphError::SerializationError(e.to_string()))
    }

    /// Deserialize material editor state from JSON.
    pub fn from_json(json: &str) -> Result<Self, GraphError> {
        let data: MaterialEditorData = serde_json::from_str(json)
            .map_err(|e| GraphError::SerializationError(e.to_string()))?;

        let graph = NodeGraph::from_json(&data.graph_json)?;

        // Find the output node.
        let output_node = graph
            .nodes
            .iter()
            .find(|n| n.type_id == "material_output")
            .map(|n| n.id);

        let library = built_in_node_library();
        let material_categories = [
            "Material", "Texture", "Constant", "Math", "Vector", "Color",
            "UV", "Shading",
        ];
        let node_palette: Vec<NodeTypeDescriptor> = library
            .into_iter()
            .filter(|d| material_categories.contains(&d.category.as_str()))
            .collect();

        Ok(Self {
            graph,
            name: data.name,
            asset_id: data.asset_id,
            preview: data.preview,
            dirty: false,
            selected_nodes: Vec::new(),
            node_palette,
            output_node,
            palette_search: String::new(),
            compiled_shader: None,
            auto_compile: true,
            preview_rotation: 0.0,
        })
    }

    /// Get the material output node.
    pub fn output_node(&self) -> Option<&GraphNode> {
        self.output_node.and_then(|id| self.graph.find_node(id))
    }
}

/// Serialization-only data structure for the material editor.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MaterialEditorData {
    name: String,
    asset_id: Uuid,
    graph_json: String,
    preview: MaterialPreviewSettings,
}

// ---------------------------------------------------------------------------
// Material Compilation (Graph -> WGSL)
// ---------------------------------------------------------------------------

/// Compile a material node graph into WGSL shader code.
pub fn compile_material(graph: &NodeGraph, output_node: Option<NodeId>) -> MaterialShaderCode {
    let mut errors = Vec::new();
    let mut texture_bindings = Vec::new();
    let mut uniform_parameters = Vec::new();

    let output_id = match output_node {
        Some(id) => id,
        None => {
            errors.push("No material output node found".into());
            return MaterialShaderCode {
                vertex_shader: String::new(),
                fragment_shader: String::new(),
                texture_bindings,
                uniform_parameters,
                errors,
                success: false,
            };
        }
    };

    let output = match graph.find_node(output_id) {
        Some(n) => n,
        None => {
            errors.push("Material output node not found in graph".into());
            return MaterialShaderCode {
                vertex_shader: String::new(),
                fragment_shader: String::new(),
                texture_bindings,
                uniform_parameters,
                errors,
                success: false,
            };
        }
    };

    // Topological sort.
    let sorted = match graph.topological_sort() {
        Ok(s) => s,
        Err(e) => {
            errors.push(format!("Graph sort failed: {}", e));
            return MaterialShaderCode {
                vertex_shader: String::new(),
                fragment_shader: String::new(),
                texture_bindings,
                uniform_parameters,
                errors,
                success: false,
            };
        }
    };

    // Evaluate the graph for default values.
    let values = match graph.evaluate() {
        Ok(v) => v,
        Err(e) => {
            errors.push(format!("Graph evaluation failed: {}", e));
            HashMap::new()
        }
    };

    // Generate WGSL fragment shader body.
    let mut fragment_body = String::new();
    let mut var_counter = 0_u32;
    let mut pin_to_var: HashMap<PinId, String> = HashMap::new();
    let mut texture_count = 0_u32;

    for node_id in &sorted {
        let node = match graph.find_node(*node_id) {
            Some(n) => n,
            None => continue,
        };

        let var_prefix = format!("n{}", var_counter);
        var_counter += 1;

        // Generate code for each node type.
        match node.type_id.as_str() {
            "float_constant" => {
                let val = node
                    .parameters
                    .get("value")
                    .cloned()
                    .unwrap_or(Value::Float(0.0))
                    .as_float();
                let var_name = format!("{}_out", var_prefix);
                fragment_body.push_str(&format!("    let {} = {:.6};\n", var_name, val));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "color_constant" => {
                let val = node
                    .parameters
                    .get("value")
                    .cloned()
                    .unwrap_or(Value::Color([1.0, 1.0, 1.0, 1.0]))
                    .as_vec4();
                let var_name = format!("{}_out", var_prefix);
                fragment_body.push_str(&format!(
                    "    let {} = vec4<f32>({:.6}, {:.6}, {:.6}, {:.6});\n",
                    var_name, val[0], val[1], val[2], val[3]
                ));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "texture_sample" => {
                let tex_binding = texture_count;
                texture_count += 1;
                let tex_name = format!("t_texture_{}", tex_binding);
                let sampler_name = format!("s_sampler_{}", tex_binding);
                let uv_var = resolve_input(node, 1, &pin_to_var, graph, "input.uv");
                let var_name = format!("{}_rgba", var_prefix);
                fragment_body.push_str(&format!(
                    "    let {} = textureSample({}, {}, {});\n",
                    var_name, tex_name, sampler_name, uv_var
                ));

                texture_bindings.push(TextureBinding {
                    group: 1,
                    binding: tex_binding,
                    name: tex_name.clone(),
                    default_texture: None,
                });

                if node.outputs.len() >= 5 {
                    pin_to_var.insert(node.outputs[0].id, var_name.clone()); // RGBA
                    pin_to_var.insert(node.outputs[1].id, format!("{}.r", var_name)); // R
                    pin_to_var.insert(node.outputs[2].id, format!("{}.g", var_name)); // G
                    pin_to_var.insert(node.outputs[3].id, format!("{}.b", var_name)); // B
                    pin_to_var.insert(node.outputs[4].id, format!("{}.a", var_name)); // A
                }
            }
            "math_add" => {
                let a = resolve_input(node, 0, &pin_to_var, graph, "0.0");
                let b = resolve_input(node, 1, &pin_to_var, graph, "0.0");
                let var_name = format!("{}_out", var_prefix);
                fragment_body.push_str(&format!("    let {} = {} + {};\n", var_name, a, b));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "math_subtract" => {
                let a = resolve_input(node, 0, &pin_to_var, graph, "0.0");
                let b = resolve_input(node, 1, &pin_to_var, graph, "0.0");
                let var_name = format!("{}_out", var_prefix);
                fragment_body.push_str(&format!("    let {} = {} - {};\n", var_name, a, b));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "math_multiply" => {
                let a = resolve_input(node, 0, &pin_to_var, graph, "1.0");
                let b = resolve_input(node, 1, &pin_to_var, graph, "1.0");
                let var_name = format!("{}_out", var_prefix);
                fragment_body.push_str(&format!("    let {} = {} * {};\n", var_name, a, b));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "math_lerp" => {
                let a = resolve_input(node, 0, &pin_to_var, graph, "0.0");
                let b = resolve_input(node, 1, &pin_to_var, graph, "1.0");
                let t = resolve_input(node, 2, &pin_to_var, graph, "0.5");
                let var_name = format!("{}_out", var_prefix);
                fragment_body.push_str(&format!(
                    "    let {} = mix({}, {}, {});\n",
                    var_name, a, b, t
                ));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "math_clamp" => {
                let v = resolve_input(node, 0, &pin_to_var, graph, "0.0");
                let min = resolve_input(node, 1, &pin_to_var, graph, "0.0");
                let max = resolve_input(node, 2, &pin_to_var, graph, "1.0");
                let var_name = format!("{}_out", var_prefix);
                fragment_body.push_str(&format!(
                    "    let {} = clamp({}, {}, {});\n",
                    var_name, v, min, max
                ));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "math_saturate" => {
                let v = resolve_input(node, 0, &pin_to_var, graph, "0.0");
                let var_name = format!("{}_out", var_prefix);
                fragment_body.push_str(&format!(
                    "    let {} = saturate({});\n",
                    var_name, v
                ));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "math_power" => {
                let base = resolve_input(node, 0, &pin_to_var, graph, "2.0");
                let exp = resolve_input(node, 1, &pin_to_var, graph, "2.0");
                let var_name = format!("{}_out", var_prefix);
                fragment_body.push_str(&format!(
                    "    let {} = pow({}, {});\n",
                    var_name, base, exp
                ));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "fresnel" => {
                let power = resolve_input(node, 1, &pin_to_var, graph, "5.0");
                let var_name = format!("{}_out", var_prefix);
                fragment_body.push_str(&format!(
                    "    let {} = pow(1.0 - max(dot(input.normal, input.view_dir), 0.0), {});\n",
                    var_name, power
                ));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "uv_coords" => {
                let var_name = format!("{}_uv", var_prefix);
                fragment_body.push_str(&format!("    let {} = input.uv;\n", var_name));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "view_direction" => {
                let var_name = format!("{}_dir", var_prefix);
                fragment_body.push_str(&format!("    let {} = input.view_dir;\n", var_name));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "world_position" => {
                let var_name = format!("{}_pos", var_prefix);
                fragment_body.push_str(&format!("    let {} = input.world_pos;\n", var_name));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "world_normal" => {
                let var_name = format!("{}_normal", var_prefix);
                fragment_body.push_str(&format!("    let {} = input.normal;\n", var_name));
                if let Some(pin) = node.outputs.first() {
                    pin_to_var.insert(pin.id, var_name);
                }
            }
            "vec_combine" => {
                let x = resolve_input(node, 0, &pin_to_var, graph, "0.0");
                let y = resolve_input(node, 1, &pin_to_var, graph, "0.0");
                let z = resolve_input(node, 2, &pin_to_var, graph, "0.0");
                let w = resolve_input(node, 3, &pin_to_var, graph, "1.0");
                if node.outputs.len() >= 3 {
                    let v4 = format!("{}_v4", var_prefix);
                    let v3 = format!("{}_v3", var_prefix);
                    let v2 = format!("{}_v2", var_prefix);
                    fragment_body.push_str(&format!(
                        "    let {} = vec4<f32>({}, {}, {}, {});\n",
                        v4, x, y, z, w
                    ));
                    fragment_body.push_str(&format!(
                        "    let {} = vec3<f32>({}, {}, {});\n",
                        v3, x, y, z
                    ));
                    fragment_body.push_str(&format!(
                        "    let {} = vec2<f32>({}, {});\n",
                        v2, x, y
                    ));
                    pin_to_var.insert(node.outputs[0].id, v4);
                    pin_to_var.insert(node.outputs[1].id, v3);
                    pin_to_var.insert(node.outputs[2].id, v2);
                }
            }
            "material_output" => {
                // Handled at the end.
            }
            _ => {
                // Unknown node type: skip.
                for out_pin in &node.outputs {
                    pin_to_var.insert(out_pin.id, "0.0".to_string());
                }
            }
        }
    }

    // Resolve material output inputs.
    let output = graph.find_node(output_id).unwrap();
    let albedo = resolve_input(output, 0, &pin_to_var, graph, "vec4<f32>(0.8, 0.8, 0.8, 1.0)");
    let metallic = resolve_input(output, 1, &pin_to_var, graph, "0.0");
    let roughness = resolve_input(output, 2, &pin_to_var, graph, "0.5");
    let normal = resolve_input(output, 3, &pin_to_var, graph, "vec3<f32>(0.0, 0.0, 1.0)");
    let emissive = resolve_input(output, 4, &pin_to_var, graph, "vec4<f32>(0.0, 0.0, 0.0, 1.0)");
    let opacity = resolve_input(output, 5, &pin_to_var, graph, "1.0");
    let ao = resolve_input(output, 6, &pin_to_var, graph, "1.0");

    // Build the complete fragment shader.
    let mut fragment_shader = String::new();
    fragment_shader.push_str("// Auto-generated by Genovo Material Editor\n\n");

    // Struct definitions.
    fragment_shader.push_str("struct FragmentInput {\n");
    fragment_shader.push_str("    @location(0) uv: vec2<f32>,\n");
    fragment_shader.push_str("    @location(1) normal: vec3<f32>,\n");
    fragment_shader.push_str("    @location(2) world_pos: vec3<f32>,\n");
    fragment_shader.push_str("    @location(3) view_dir: vec3<f32>,\n");
    fragment_shader.push_str("};\n\n");

    fragment_shader.push_str("struct MaterialOutput {\n");
    fragment_shader.push_str("    albedo: vec4<f32>,\n");
    fragment_shader.push_str("    metallic: f32,\n");
    fragment_shader.push_str("    roughness: f32,\n");
    fragment_shader.push_str("    normal: vec3<f32>,\n");
    fragment_shader.push_str("    emissive: vec4<f32>,\n");
    fragment_shader.push_str("    opacity: f32,\n");
    fragment_shader.push_str("    ao: f32,\n");
    fragment_shader.push_str("};\n\n");

    // Texture/sampler bindings.
    for tb in &texture_bindings {
        fragment_shader.push_str(&format!(
            "@group({}) @binding({}) var {}: texture_2d<f32>;\n",
            tb.group, tb.binding, tb.name
        ));
        fragment_shader.push_str(&format!(
            "@group({}) @binding({}) var s_sampler_{}: sampler;\n",
            tb.group, tb.binding + 100, tb.binding
        ));
    }
    if !texture_bindings.is_empty() {
        fragment_shader.push('\n');
    }

    // Main function.
    fragment_shader.push_str("fn evaluate_material(input: FragmentInput) -> MaterialOutput {\n");
    fragment_shader.push_str(&fragment_body);
    fragment_shader.push_str("\n    var output: MaterialOutput;\n");
    fragment_shader.push_str(&format!("    output.albedo = {};\n", albedo));
    fragment_shader.push_str(&format!("    output.metallic = {};\n", metallic));
    fragment_shader.push_str(&format!("    output.roughness = {};\n", roughness));
    fragment_shader.push_str(&format!("    output.normal = {};\n", normal));
    fragment_shader.push_str(&format!("    output.emissive = {};\n", emissive));
    fragment_shader.push_str(&format!("    output.opacity = {};\n", opacity));
    fragment_shader.push_str(&format!("    output.ao = {};\n", ao));
    fragment_shader.push_str("    return output;\n");
    fragment_shader.push_str("}\n");

    // Simple vertex shader.
    let vertex_shader = concat!(
        "// Auto-generated vertex shader\n\n",
        "struct VertexInput {\n",
        "    @location(0) position: vec3<f32>,\n",
        "    @location(1) normal: vec3<f32>,\n",
        "    @location(2) uv: vec2<f32>,\n",
        "};\n\n",
        "struct VertexOutput {\n",
        "    @builtin(position) clip_position: vec4<f32>,\n",
        "    @location(0) uv: vec2<f32>,\n",
        "    @location(1) normal: vec3<f32>,\n",
        "    @location(2) world_pos: vec3<f32>,\n",
        "    @location(3) view_dir: vec3<f32>,\n",
        "};\n\n",
        "struct Camera {\n",
        "    view_proj: mat4x4<f32>,\n",
        "    position: vec3<f32>,\n",
        "};\n\n",
        "@group(0) @binding(0) var<uniform> camera: Camera;\n\n",
        "@vertex\n",
        "fn vs_main(input: VertexInput) -> VertexOutput {\n",
        "    var output: VertexOutput;\n",
        "    output.clip_position = camera.view_proj * vec4<f32>(input.position, 1.0);\n",
        "    output.uv = input.uv;\n",
        "    output.normal = input.normal;\n",
        "    output.world_pos = input.position;\n",
        "    output.view_dir = normalize(camera.position - input.position);\n",
        "    return output;\n",
        "}\n",
    ).to_string();

    MaterialShaderCode {
        vertex_shader,
        fragment_shader,
        texture_bindings,
        uniform_parameters,
        errors,
        success: true,
    }
}

/// Resolve an input pin to a WGSL variable name or literal.
fn resolve_input(
    node: &GraphNode,
    input_index: usize,
    pin_to_var: &HashMap<PinId, String>,
    graph: &NodeGraph,
    fallback: &str,
) -> String {
    if input_index >= node.inputs.len() {
        return fallback.to_string();
    }

    let pin = &node.inputs[input_index];
    if let Some(conn) = graph.connection_to(pin.id) {
        if let Some(var) = pin_to_var.get(&conn.from_pin) {
            return var.clone();
        }
    }

    // Use default value if available.
    if let Some(ref default) = pin.default_value {
        return value_to_wgsl(default);
    }

    fallback.to_string()
}

/// Convert a Value to a WGSL literal string.
fn value_to_wgsl(value: &Value) -> String {
    match value {
        Value::Float(v) => format!("{:.6}", v),
        Value::Vec2(v) => format!("vec2<f32>({:.6}, {:.6})", v[0], v[1]),
        Value::Vec3(v) => format!("vec3<f32>({:.6}, {:.6}, {:.6})", v[0], v[1], v[2]),
        Value::Vec4(v) | Value::Color(v) => {
            format!("vec4<f32>({:.6}, {:.6}, {:.6}, {:.6})", v[0], v[1], v[2], v[3])
        }
        Value::Int(v) => format!("{}", v),
        Value::Bool(v) => if *v { "true".into() } else { "false".into() },
        _ => "0.0".into(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_material_editor() {
        let editor = MaterialEditor::new("TestMaterial");
        assert_eq!(editor.name, "TestMaterial");
        assert!(editor.output_node.is_some());
        assert!(editor.graph.node_count() >= 1);
        assert!(!editor.node_palette.is_empty());
    }

    #[test]
    fn add_and_remove_nodes() {
        let mut editor = MaterialEditor::new("Test");
        let initial_count = editor.graph.node_count();

        let node_id = editor.add_node("math_add", [100.0, 100.0]);
        assert!(node_id.is_some());
        assert_eq!(editor.graph.node_count(), initial_count + 1);

        let removed = editor.remove_node(node_id.unwrap());
        assert!(removed);
        assert_eq!(editor.graph.node_count(), initial_count);
    }

    #[test]
    fn cannot_remove_output_node() {
        let mut editor = MaterialEditor::new("Test");
        let output_id = editor.output_node.unwrap();
        let removed = editor.remove_node(output_id);
        assert!(!removed);
    }

    #[test]
    fn compile_default_material() {
        let mut editor = MaterialEditor::new("Default");
        let code = editor.compile();
        assert!(code.success);
        assert!(!code.fragment_shader.is_empty());
        assert!(!code.vertex_shader.is_empty());
        assert!(code.fragment_shader.contains("evaluate_material"));
    }

    #[test]
    fn compile_with_texture() {
        let mut editor = MaterialEditor::new("Textured");
        let tex_id = editor.add_node("texture_sample", [200.0, 200.0]).unwrap();

        // Connect texture RGBA output to material albedo input.
        let tex_node = editor.graph.find_node(tex_id).unwrap();
        let tex_out = tex_node.outputs[0].id;
        let output_id = editor.output_node.unwrap();
        let output_node = editor.graph.find_node(output_id).unwrap();
        let albedo_in = output_node.inputs[0].id;

        let _ = editor.connect(tex_id, tex_out, output_id, albedo_in);
        let code = editor.compile();
        assert!(code.success);
        assert!(!code.texture_bindings.is_empty());
        assert!(code.fragment_shader.contains("textureSample"));
    }

    #[test]
    fn palette_filtering() {
        let editor = MaterialEditor::new("Test");
        let all = editor.filtered_palette();
        assert!(!all.is_empty());

        let mut editor = editor;
        editor.palette_search = "multiply".into();
        let filtered = editor.filtered_palette();
        assert!(filtered.len() < all.len());
        assert!(filtered.iter().any(|d| d.type_id == "math_multiply"));
    }

    #[test]
    fn palette_by_category() {
        let editor = MaterialEditor::new("Test");
        let categories = editor.palette_by_category();
        assert!(!categories.is_empty());
        // Should have Math, Texture, etc.
        assert!(categories.iter().any(|(cat, _)| cat == "Math"));
    }

    #[test]
    fn serialize_deserialize() {
        let mut editor = MaterialEditor::new("SerTest");
        editor.add_node("math_add", [100.0, 100.0]);

        let json = editor.to_json().unwrap();
        let restored = MaterialEditor::from_json(&json).unwrap();
        assert_eq!(restored.name, "SerTest");
        assert_eq!(restored.graph.node_count(), editor.graph.node_count());
    }

    #[test]
    fn value_to_wgsl_literals() {
        assert_eq!(value_to_wgsl(&Value::Float(1.0)), "1.000000");
        assert!(value_to_wgsl(&Value::Vec3([1.0, 2.0, 3.0])).contains("vec3<f32>"));
        assert!(value_to_wgsl(&Value::Color([1.0, 0.0, 0.0, 1.0])).contains("vec4<f32>"));
        assert_eq!(value_to_wgsl(&Value::Bool(true)), "true");
    }

    #[test]
    fn preview_update() {
        let mut editor = MaterialEditor::new("Test");
        editor.preview.auto_rotate = true;
        editor.preview.rotation_speed = 90.0;
        editor.update_preview(1.0);
        assert!((editor.preview_rotation - 90.0).abs() < 1e-3);
    }
}
