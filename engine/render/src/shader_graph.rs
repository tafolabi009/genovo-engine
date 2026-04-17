// engine/render/src/shader_graph.rs
//
// Visual shader authoring system for the Genovo engine. Provides a directed
// acyclic graph (DAG) of shader nodes that can be compiled to WGSL source
// code. The data structures defined here power the node-graph UI in the
// editor; the UI itself lives in the editor crate.
//
// Key capabilities:
//   - 40+ built-in node types spanning math, texture sampling, UV
//     manipulation, vertex attributes, camera properties, and constants.
//   - Full type system with implicit conversions (scalar widening).
//   - Topological-sort-based WGSL code generation.
//   - Graph validation: cycle detection, type-mismatch reporting, and
//     unconnected-required-input checks.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Type system
// ---------------------------------------------------------------------------

/// Data types that can flow through shader graph connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Float,
    Vec2,
    Vec3,
    Vec4,
    Mat3,
    Mat4,
    Texture2D,
    TextureCube,
    Sampler,
    Bool,
}

impl DataType {
    /// WGSL type name.
    pub fn wgsl_name(&self) -> &'static str {
        match self {
            Self::Float => "f32",
            Self::Vec2 => "vec2<f32>",
            Self::Vec3 => "vec3<f32>",
            Self::Vec4 => "vec4<f32>",
            Self::Mat3 => "mat3x3<f32>",
            Self::Mat4 => "mat4x4<f32>",
            Self::Texture2D => "texture_2d<f32>",
            Self::TextureCube => "texture_cube<f32>",
            Self::Sampler => "sampler",
            Self::Bool => "bool",
        }
    }

    /// Number of scalar components (0 for non-numeric types).
    pub fn component_count(&self) -> u32 {
        match self {
            Self::Float | Self::Bool => 1,
            Self::Vec2 => 2,
            Self::Vec3 => 3,
            Self::Vec4 => 4,
            Self::Mat3 => 9,
            Self::Mat4 => 16,
            _ => 0,
        }
    }

    /// Whether an implicit conversion from `self` to `target` is valid.
    pub fn can_convert_to(&self, target: &DataType) -> bool {
        if self == target {
            return true;
        }
        // Scalar widening: Float -> Vec2/Vec3/Vec4.
        if *self == DataType::Float {
            return matches!(target, DataType::Vec2 | DataType::Vec3 | DataType::Vec4);
        }
        // Vec3 -> Vec4 (w=1.0 or w=0.0 depending on context).
        if *self == DataType::Vec3 && *target == DataType::Vec4 {
            return true;
        }
        // Vec4 -> Vec3 (truncate).
        if *self == DataType::Vec4 && *target == DataType::Vec3 {
            return true;
        }
        // Vec2 -> Vec3/Vec4 (zero-extend).
        if *self == DataType::Vec2 && matches!(target, DataType::Vec3 | DataType::Vec4) {
            return true;
        }
        false
    }

    /// Generate WGSL code to convert a variable `var_name` of type `self` to
    /// type `target`. Returns `None` if no conversion is needed, or the WGSL
    /// expression string.
    pub fn conversion_expr(&self, var_name: &str, target: &DataType) -> Option<String> {
        if self == target {
            return None;
        }
        match (self, target) {
            (DataType::Float, DataType::Vec2) => {
                Some(format!("vec2<f32>({0}, {0})", var_name))
            }
            (DataType::Float, DataType::Vec3) => {
                Some(format!("vec3<f32>({0}, {0}, {0})", var_name))
            }
            (DataType::Float, DataType::Vec4) => {
                Some(format!("vec4<f32>({0}, {0}, {0}, {0})", var_name))
            }
            (DataType::Vec2, DataType::Vec3) => {
                Some(format!("vec3<f32>({}.xy, 0.0)", var_name))
            }
            (DataType::Vec2, DataType::Vec4) => {
                Some(format!("vec4<f32>({}.xy, 0.0, 1.0)", var_name))
            }
            (DataType::Vec3, DataType::Vec4) => {
                Some(format!("vec4<f32>({}.xyz, 1.0)", var_name))
            }
            (DataType::Vec4, DataType::Vec3) => {
                Some(format!("{}.xyz", var_name))
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Port
// ---------------------------------------------------------------------------

/// Unique identifier for a node within a shader graph.
pub type NodeId = u64;

/// Identifies a specific input or output port on a node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PortId {
    /// The node this port belongs to.
    pub node: NodeId,
    /// Port name.
    pub name: String,
}

impl PortId {
    pub fn new(node: NodeId, name: impl Into<String>) -> Self {
        Self {
            node,
            name: name.into(),
        }
    }
}

/// Definition of a port (input or output).
#[derive(Debug, Clone)]
pub struct PortDef {
    pub name: String,
    pub data_type: DataType,
    /// Whether this input is required (only meaningful for inputs).
    pub required: bool,
    /// Default value expression (WGSL literal) when unconnected.
    pub default_value: Option<String>,
}

impl PortDef {
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            required: false,
            default_value: None,
        }
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default_value = Some(default.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

/// A directed edge from an output port to an input port.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Connection {
    pub from_node: NodeId,
    pub from_port: String,
    pub to_node: NodeId,
    pub to_port: String,
}

// ---------------------------------------------------------------------------
// NodeKind -- built-in node types
// ---------------------------------------------------------------------------

/// Enumeration of all built-in shader node types.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    // -- Math ----------------------------------------------------------------
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Sqrt,
    Abs,
    Negate,
    Clamp,
    Saturate,
    Lerp,
    Step,
    Smoothstep,
    Min,
    Max,
    Dot,
    Cross,
    Normalize,
    Length,
    Floor,
    Ceil,
    Fract,
    Sin,
    Cos,

    // -- Texture -------------------------------------------------------------
    SampleTexture2D,
    SampleCubemap,
    TextureSize,
    TexelFetch,

    // -- UV ------------------------------------------------------------------
    UV0,
    UV1,
    Tiling,
    Offset,
    Rotate,
    Parallax,
    Triplanar,

    // -- Vertex --------------------------------------------------------------
    Position,
    Normal,
    Tangent,
    Bitangent,
    VertexColor,

    // -- Camera --------------------------------------------------------------
    ViewDirection,
    ViewPosition,
    ScreenPosition,
    Depth,

    // -- Constants -----------------------------------------------------------
    ConstFloat(f32),
    ConstVec2([f32; 2]),
    ConstVec3([f32; 3]),
    ConstVec4([f32; 4]),
    ConstColor([f32; 4]),
    Time,
    SinTime,

    // -- Output --------------------------------------------------------------
    PBROutput,

    // -- Custom / expression -------------------------------------------------
    CustomExpression(String),
}

impl NodeKind {
    /// Return the input port definitions for this node kind.
    pub fn input_ports(&self) -> Vec<PortDef> {
        match self {
            // Binary math ops.
            Self::Add | Self::Subtract | Self::Multiply | Self::Divide | Self::Min | Self::Max => vec![
                PortDef::new("a", DataType::Float).required().with_default("0.0"),
                PortDef::new("b", DataType::Float).required().with_default("0.0"),
            ],
            Self::Power => vec![
                PortDef::new("base", DataType::Float).required().with_default("1.0"),
                PortDef::new("exponent", DataType::Float).required().with_default("1.0"),
            ],
            Self::Sqrt | Self::Abs | Self::Negate | Self::Saturate
            | Self::Normalize | Self::Length | Self::Floor | Self::Ceil
            | Self::Fract | Self::Sin | Self::Cos => vec![
                PortDef::new("value", DataType::Float).required().with_default("0.0"),
            ],
            Self::Clamp => vec![
                PortDef::new("value", DataType::Float).required().with_default("0.0"),
                PortDef::new("min", DataType::Float).with_default("0.0"),
                PortDef::new("max", DataType::Float).with_default("1.0"),
            ],
            Self::Lerp => vec![
                PortDef::new("a", DataType::Float).required().with_default("0.0"),
                PortDef::new("b", DataType::Float).required().with_default("1.0"),
                PortDef::new("t", DataType::Float).required().with_default("0.5"),
            ],
            Self::Step => vec![
                PortDef::new("edge", DataType::Float).with_default("0.5"),
                PortDef::new("x", DataType::Float).required().with_default("0.0"),
            ],
            Self::Smoothstep => vec![
                PortDef::new("edge0", DataType::Float).with_default("0.0"),
                PortDef::new("edge1", DataType::Float).with_default("1.0"),
                PortDef::new("x", DataType::Float).required().with_default("0.5"),
            ],
            Self::Dot => vec![
                PortDef::new("a", DataType::Vec3).required().with_default("vec3<f32>(0.0)"),
                PortDef::new("b", DataType::Vec3).required().with_default("vec3<f32>(0.0)"),
            ],
            Self::Cross => vec![
                PortDef::new("a", DataType::Vec3).required().with_default("vec3<f32>(1.0, 0.0, 0.0)"),
                PortDef::new("b", DataType::Vec3).required().with_default("vec3<f32>(0.0, 1.0, 0.0)"),
            ],
            // Texture sampling.
            Self::SampleTexture2D => vec![
                PortDef::new("uv", DataType::Vec2).required().with_default("in.uv"),
            ],
            Self::SampleCubemap => vec![
                PortDef::new("direction", DataType::Vec3).required().with_default("in.normal"),
            ],
            Self::TextureSize => vec![],
            Self::TexelFetch => vec![
                PortDef::new("coord", DataType::Vec2).required().with_default("vec2<f32>(0.0)"),
                PortDef::new("lod", DataType::Float).with_default("0.0"),
            ],
            // UV nodes.
            Self::UV0 | Self::UV1 => vec![],
            Self::Tiling => vec![
                PortDef::new("uv", DataType::Vec2).required().with_default("in.uv"),
                PortDef::new("scale", DataType::Vec2).with_default("vec2<f32>(1.0, 1.0)"),
            ],
            Self::Offset => vec![
                PortDef::new("uv", DataType::Vec2).required().with_default("in.uv"),
                PortDef::new("offset", DataType::Vec2).with_default("vec2<f32>(0.0, 0.0)"),
            ],
            Self::Rotate => vec![
                PortDef::new("uv", DataType::Vec2).required().with_default("in.uv"),
                PortDef::new("angle", DataType::Float).with_default("0.0"),
                PortDef::new("center", DataType::Vec2).with_default("vec2<f32>(0.5, 0.5)"),
            ],
            Self::Parallax => vec![
                PortDef::new("uv", DataType::Vec2).required().with_default("in.uv"),
                PortDef::new("height", DataType::Float).required().with_default("0.0"),
                PortDef::new("scale", DataType::Float).with_default("0.05"),
                PortDef::new("view_dir", DataType::Vec3).with_default("normalize(camera.position - in.world_position)"),
            ],
            Self::Triplanar => vec![
                PortDef::new("position", DataType::Vec3).required().with_default("in.world_position"),
                PortDef::new("normal", DataType::Vec3).required().with_default("in.world_normal"),
                PortDef::new("sharpness", DataType::Float).with_default("1.0"),
            ],
            // Vertex.
            Self::Position | Self::Normal | Self::Tangent | Self::Bitangent | Self::VertexColor => vec![],
            // Camera.
            Self::ViewDirection | Self::ViewPosition | Self::ScreenPosition | Self::Depth => vec![],
            // Constants.
            Self::ConstFloat(_) | Self::ConstVec2(_) | Self::ConstVec3(_)
            | Self::ConstVec4(_) | Self::ConstColor(_) | Self::Time | Self::SinTime => vec![],
            // PBR output.
            Self::PBROutput => vec![
                PortDef::new("albedo", DataType::Vec3).required().with_default("vec3<f32>(1.0)"),
                PortDef::new("metallic", DataType::Float).with_default("0.0"),
                PortDef::new("roughness", DataType::Float).with_default("0.5"),
                PortDef::new("normal", DataType::Vec3).with_default("in.world_normal"),
                PortDef::new("emissive", DataType::Vec3).with_default("vec3<f32>(0.0)"),
                PortDef::new("ao", DataType::Float).with_default("1.0"),
                PortDef::new("alpha", DataType::Float).with_default("1.0"),
            ],
            Self::CustomExpression(_) => vec![
                PortDef::new("input0", DataType::Float).with_default("0.0"),
                PortDef::new("input1", DataType::Float).with_default("0.0"),
            ],
        }
    }

    /// Return the output port definitions for this node kind.
    pub fn output_ports(&self) -> Vec<PortDef> {
        match self {
            Self::Add | Self::Subtract | Self::Multiply | Self::Divide
            | Self::Min | Self::Max | Self::Power | Self::Clamp | Self::Lerp
            | Self::Step | Self::Smoothstep => vec![
                PortDef::new("result", DataType::Float),
            ],
            Self::Sqrt | Self::Abs | Self::Negate | Self::Saturate
            | Self::Floor | Self::Ceil | Self::Fract | Self::Sin | Self::Cos => vec![
                PortDef::new("result", DataType::Float),
            ],
            Self::Dot | Self::Length => vec![
                PortDef::new("result", DataType::Float),
            ],
            Self::Cross | Self::Normalize => vec![
                PortDef::new("result", DataType::Vec3),
            ],
            Self::SampleTexture2D => vec![
                PortDef::new("color", DataType::Vec4),
                PortDef::new("r", DataType::Float),
                PortDef::new("g", DataType::Float),
                PortDef::new("b", DataType::Float),
                PortDef::new("a", DataType::Float),
            ],
            Self::SampleCubemap => vec![
                PortDef::new("color", DataType::Vec4),
            ],
            Self::TextureSize => vec![
                PortDef::new("size", DataType::Vec2),
            ],
            Self::TexelFetch => vec![
                PortDef::new("color", DataType::Vec4),
            ],
            Self::UV0 | Self::UV1 => vec![
                PortDef::new("uv", DataType::Vec2),
            ],
            Self::Tiling | Self::Offset | Self::Rotate => vec![
                PortDef::new("uv", DataType::Vec2),
            ],
            Self::Parallax => vec![
                PortDef::new("uv", DataType::Vec2),
            ],
            Self::Triplanar => vec![
                PortDef::new("color", DataType::Vec4),
            ],
            Self::Position => vec![
                PortDef::new("position", DataType::Vec3),
            ],
            Self::Normal => vec![
                PortDef::new("normal", DataType::Vec3),
            ],
            Self::Tangent => vec![
                PortDef::new("tangent", DataType::Vec3),
            ],
            Self::Bitangent => vec![
                PortDef::new("bitangent", DataType::Vec3),
            ],
            Self::VertexColor => vec![
                PortDef::new("color", DataType::Vec4),
            ],
            Self::ViewDirection => vec![
                PortDef::new("direction", DataType::Vec3),
            ],
            Self::ViewPosition => vec![
                PortDef::new("position", DataType::Vec3),
            ],
            Self::ScreenPosition => vec![
                PortDef::new("position", DataType::Vec4),
            ],
            Self::Depth => vec![
                PortDef::new("depth", DataType::Float),
            ],
            Self::ConstFloat(_) => vec![
                PortDef::new("value", DataType::Float),
            ],
            Self::ConstVec2(_) => vec![
                PortDef::new("value", DataType::Vec2),
            ],
            Self::ConstVec3(_) => vec![
                PortDef::new("value", DataType::Vec3),
            ],
            Self::ConstVec4(_) | Self::ConstColor(_) => vec![
                PortDef::new("value", DataType::Vec4),
            ],
            Self::Time => vec![
                PortDef::new("time", DataType::Float),
            ],
            Self::SinTime => vec![
                PortDef::new("sin_time", DataType::Float),
            ],
            Self::PBROutput => vec![], // Output node has no outputs.
            Self::CustomExpression(_) => vec![
                PortDef::new("result", DataType::Float),
            ],
        }
    }

    /// Generate WGSL code for this node. `input_vars` maps input port names to
    /// the WGSL variable name that provides the value. `output_prefix` is used
    /// as a prefix for output variable declarations.
    pub fn generate_code(
        &self,
        input_vars: &HashMap<String, String>,
        output_prefix: &str,
    ) -> String {
        let get_input = |name: &str| -> String {
            input_vars
                .get(name)
                .cloned()
                .or_else(|| {
                    self.input_ports()
                        .iter()
                        .find(|p| p.name == name)
                        .and_then(|p| p.default_value.clone())
                })
                .unwrap_or_else(|| "0.0".to_string())
        };

        match self {
            Self::Add => {
                let a = get_input("a");
                let b = get_input("b");
                format!("    let {}_result = {} + {};\n", output_prefix, a, b)
            }
            Self::Subtract => {
                let a = get_input("a");
                let b = get_input("b");
                format!("    let {}_result = {} - {};\n", output_prefix, a, b)
            }
            Self::Multiply => {
                let a = get_input("a");
                let b = get_input("b");
                format!("    let {}_result = {} * {};\n", output_prefix, a, b)
            }
            Self::Divide => {
                let a = get_input("a");
                let b = get_input("b");
                format!("    let {}_result = {} / max({}, 0.00001);\n", output_prefix, a, b)
            }
            Self::Power => {
                let base = get_input("base");
                let exp = get_input("exponent");
                format!("    let {}_result = pow({}, {});\n", output_prefix, base, exp)
            }
            Self::Sqrt => {
                let v = get_input("value");
                format!("    let {}_result = sqrt(max({}, 0.0));\n", output_prefix, v)
            }
            Self::Abs => {
                let v = get_input("value");
                format!("    let {}_result = abs({});\n", output_prefix, v)
            }
            Self::Negate => {
                let v = get_input("value");
                format!("    let {}_result = -{};\n", output_prefix, v)
            }
            Self::Clamp => {
                let v = get_input("value");
                let lo = get_input("min");
                let hi = get_input("max");
                format!("    let {}_result = clamp({}, {}, {});\n", output_prefix, v, lo, hi)
            }
            Self::Saturate => {
                let v = get_input("value");
                format!("    let {}_result = saturate({});\n", output_prefix, v)
            }
            Self::Lerp => {
                let a = get_input("a");
                let b = get_input("b");
                let t = get_input("t");
                format!("    let {}_result = mix({}, {}, {});\n", output_prefix, a, b, t)
            }
            Self::Step => {
                let edge = get_input("edge");
                let x = get_input("x");
                format!("    let {}_result = step({}, {});\n", output_prefix, edge, x)
            }
            Self::Smoothstep => {
                let e0 = get_input("edge0");
                let e1 = get_input("edge1");
                let x = get_input("x");
                format!("    let {}_result = smoothstep({}, {}, {});\n", output_prefix, e0, e1, x)
            }
            Self::Min => {
                let a = get_input("a");
                let b = get_input("b");
                format!("    let {}_result = min({}, {});\n", output_prefix, a, b)
            }
            Self::Max => {
                let a = get_input("a");
                let b = get_input("b");
                format!("    let {}_result = max({}, {});\n", output_prefix, a, b)
            }
            Self::Dot => {
                let a = get_input("a");
                let b = get_input("b");
                format!("    let {}_result = dot({}, {});\n", output_prefix, a, b)
            }
            Self::Cross => {
                let a = get_input("a");
                let b = get_input("b");
                format!("    let {}_result = cross({}, {});\n", output_prefix, a, b)
            }
            Self::Normalize => {
                let v = get_input("value");
                format!("    let {}_result = normalize({});\n", output_prefix, v)
            }
            Self::Length => {
                let v = get_input("value");
                format!("    let {}_result = length({});\n", output_prefix, v)
            }
            Self::Floor => {
                let v = get_input("value");
                format!("    let {}_result = floor({});\n", output_prefix, v)
            }
            Self::Ceil => {
                let v = get_input("value");
                format!("    let {}_result = ceil({});\n", output_prefix, v)
            }
            Self::Fract => {
                let v = get_input("value");
                format!("    let {}_result = fract({});\n", output_prefix, v)
            }
            Self::Sin => {
                let v = get_input("value");
                format!("    let {}_result = sin({});\n", output_prefix, v)
            }
            Self::Cos => {
                let v = get_input("value");
                format!("    let {}_result = cos({});\n", output_prefix, v)
            }
            Self::SampleTexture2D => {
                let uv = get_input("uv");
                format!(
                    "    let {p}_color = textureSample(t_{p}, s_{p}, {uv});\n\
                         let {p}_r = {p}_color.r;\n\
                         let {p}_g = {p}_color.g;\n\
                         let {p}_b = {p}_color.b;\n\
                         let {p}_a = {p}_color.a;\n",
                    p = output_prefix,
                    uv = uv,
                )
            }
            Self::SampleCubemap => {
                let dir = get_input("direction");
                format!(
                    "    let {p}_color = textureSample(t_{p}, s_{p}, {dir});\n",
                    p = output_prefix,
                    dir = dir,
                )
            }
            Self::TextureSize => {
                format!(
                    "    let {p}_size = vec2<f32>(textureDimensions(t_{p}));\n",
                    p = output_prefix,
                )
            }
            Self::TexelFetch => {
                let coord = get_input("coord");
                let lod = get_input("lod");
                format!(
                    "    let {p}_color = textureLoad(t_{p}, vec2<i32>({coord}), i32({lod}));\n",
                    p = output_prefix,
                    coord = coord,
                    lod = lod,
                )
            }
            Self::UV0 => format!("    let {}_uv = in.uv0;\n", output_prefix),
            Self::UV1 => format!("    let {}_uv = in.uv1;\n", output_prefix),
            Self::Tiling => {
                let uv = get_input("uv");
                let scale = get_input("scale");
                format!("    let {}_uv = {} * {};\n", output_prefix, uv, scale)
            }
            Self::Offset => {
                let uv = get_input("uv");
                let off = get_input("offset");
                format!("    let {}_uv = {} + {};\n", output_prefix, uv, off)
            }
            Self::Rotate => {
                let uv = get_input("uv");
                let angle = get_input("angle");
                let center = get_input("center");
                format!(
                    "    let {p}_cos = cos({angle});\n\
                         let {p}_sin = sin({angle});\n\
                         let {p}_centered = {uv} - {center};\n\
                         let {p}_uv = vec2<f32>(\n\
                             {p}_centered.x * {p}_cos - {p}_centered.y * {p}_sin,\n\
                             {p}_centered.x * {p}_sin + {p}_centered.y * {p}_cos\n\
                         ) + {center};\n",
                    p = output_prefix,
                    uv = uv,
                    angle = angle,
                    center = center,
                )
            }
            Self::Parallax => {
                let uv = get_input("uv");
                let height = get_input("height");
                let scale = get_input("scale");
                let view_dir = get_input("view_dir");
                format!(
                    "    let {p}_offset = {view_dir}.xy / max({view_dir}.z, 0.001) * ({height} * {scale});\n\
                         let {p}_uv = {uv} - {p}_offset;\n",
                    p = output_prefix,
                    uv = uv,
                    height = height,
                    scale = scale,
                    view_dir = view_dir,
                )
            }
            Self::Triplanar => {
                let pos = get_input("position");
                let nor = get_input("normal");
                let sharp = get_input("sharpness");
                format!(
                    "    let {p}_blend = pow(abs({nor}), vec3<f32>({sharp}));\n\
                         let {p}_blend_norm = {p}_blend / ({p}_blend.x + {p}_blend.y + {p}_blend.z);\n\
                         let {p}_xy = textureSample(t_{p}, s_{p}, {pos}.xy) * {p}_blend_norm.z;\n\
                         let {p}_xz = textureSample(t_{p}, s_{p}, {pos}.xz) * {p}_blend_norm.y;\n\
                         let {p}_yz = textureSample(t_{p}, s_{p}, {pos}.yz) * {p}_blend_norm.x;\n\
                         let {p}_color = {p}_xy + {p}_xz + {p}_yz;\n",
                    p = output_prefix,
                    pos = pos,
                    nor = nor,
                    sharp = sharp,
                )
            }
            Self::Position => format!("    let {}_position = in.world_position;\n", output_prefix),
            Self::Normal => format!("    let {}_normal = in.world_normal;\n", output_prefix),
            Self::Tangent => format!("    let {}_tangent = in.world_tangent.xyz;\n", output_prefix),
            Self::Bitangent => {
                format!(
                    "    let {p}_bitangent = cross(in.world_normal, in.world_tangent.xyz) * in.world_tangent.w;\n",
                    p = output_prefix,
                )
            }
            Self::VertexColor => format!("    let {}_color = in.vertex_color;\n", output_prefix),
            Self::ViewDirection => {
                format!(
                    "    let {p}_direction = normalize(camera.position.xyz - in.world_position);\n",
                    p = output_prefix,
                )
            }
            Self::ViewPosition => {
                format!("    let {p}_position = camera.position.xyz;\n", p = output_prefix)
            }
            Self::ScreenPosition => {
                format!("    let {p}_position = in.clip_position;\n", p = output_prefix)
            }
            Self::Depth => {
                format!("    let {p}_depth = in.clip_position.z / in.clip_position.w;\n", p = output_prefix)
            }
            Self::ConstFloat(v) => {
                format!("    let {}_value = {:?};\n", output_prefix, v)
            }
            Self::ConstVec2(v) => {
                format!("    let {}_value = vec2<f32>({}, {});\n", output_prefix, v[0], v[1])
            }
            Self::ConstVec3(v) => {
                format!("    let {}_value = vec3<f32>({}, {}, {});\n", output_prefix, v[0], v[1], v[2])
            }
            Self::ConstVec4(v) | Self::ConstColor(v) => {
                format!("    let {}_value = vec4<f32>({}, {}, {}, {});\n", output_prefix, v[0], v[1], v[2], v[3])
            }
            Self::Time => format!("    let {}_time = globals.time;\n", output_prefix),
            Self::SinTime => format!("    let {}_sin_time = sin(globals.time);\n", output_prefix),
            Self::PBROutput => {
                let albedo = get_input("albedo");
                let metallic = get_input("metallic");
                let roughness = get_input("roughness");
                let normal = get_input("normal");
                let emissive = get_input("emissive");
                let ao = get_input("ao");
                let alpha = get_input("alpha");
                format!(
                    "    // PBR Output\n\
                         let pbr_albedo = {albedo};\n\
                         let pbr_metallic = {metallic};\n\
                         let pbr_roughness = {roughness};\n\
                         let pbr_normal = {normal};\n\
                         let pbr_emissive = {emissive};\n\
                         let pbr_ao = {ao};\n\
                         let pbr_alpha = {alpha};\n",
                    albedo = albedo,
                    metallic = metallic,
                    roughness = roughness,
                    normal = normal,
                    emissive = emissive,
                    ao = ao,
                    alpha = alpha,
                )
            }
            Self::CustomExpression(expr) => {
                let input0 = get_input("input0");
                let input1 = get_input("input1");
                let resolved = expr
                    .replace("$input0", &input0)
                    .replace("$input1", &input1);
                format!("    let {}_result = {};\n", output_prefix, resolved)
            }
        }
    }

    /// Whether this node requires texture bindings in the shader.
    pub fn needs_texture_binding(&self) -> bool {
        matches!(
            self,
            Self::SampleTexture2D
                | Self::SampleCubemap
                | Self::TextureSize
                | Self::TexelFetch
                | Self::Triplanar
        )
    }
}

// ---------------------------------------------------------------------------
// ShaderNode
// ---------------------------------------------------------------------------

/// A node in the shader graph.
#[derive(Debug, Clone)]
pub struct ShaderNode {
    /// Unique identifier within the graph.
    pub id: NodeId,
    /// Node kind (determines behaviour and ports).
    pub kind: NodeKind,
    /// Human-readable label (for the editor UI).
    pub label: String,
    /// Position in the editor canvas (x, y).
    pub position: [f32; 2],
    /// Whether this node is collapsed in the UI.
    pub collapsed: bool,
    /// Optional comment / documentation.
    pub comment: String,
    /// Custom texture name for texture-sampling nodes.
    pub texture_name: Option<String>,
}

impl ShaderNode {
    /// Create a new node with the given kind.
    pub fn new(id: NodeId, kind: NodeKind) -> Self {
        let label = format!("{:?}", kind);
        Self {
            id,
            kind,
            label,
            position: [0.0, 0.0],
            collapsed: false,
            comment: String::new(),
            texture_name: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.position = [x, y];
        self
    }

    pub fn with_texture_name(mut self, name: impl Into<String>) -> Self {
        self.texture_name = Some(name.into());
        self
    }

    pub fn input_ports(&self) -> Vec<PortDef> {
        self.kind.input_ports()
    }

    pub fn output_ports(&self) -> Vec<PortDef> {
        self.kind.output_ports()
    }

    /// Generate the WGSL variable prefix for this node.
    pub fn var_prefix(&self) -> String {
        format!("n{}", self.id)
    }
}

// ---------------------------------------------------------------------------
// GraphError
// ---------------------------------------------------------------------------

/// Errors detected during graph validation or compilation.
#[derive(Debug, Clone)]
pub enum GraphError {
    /// The graph contains a cycle involving the listed nodes.
    Cycle(Vec<NodeId>),
    /// A required input port is not connected and has no default value.
    UnconnectedRequiredInput {
        node: NodeId,
        port: String,
    },
    /// Two connected ports have incompatible types.
    TypeMismatch {
        from_node: NodeId,
        from_port: String,
        from_type: DataType,
        to_node: NodeId,
        to_port: String,
        to_type: DataType,
    },
    /// No output node found in the graph.
    NoOutputNode,
    /// Multiple output nodes found.
    MultipleOutputNodes(Vec<NodeId>),
    /// A node references a port that does not exist.
    InvalidPort {
        node: NodeId,
        port: String,
    },
    /// A node referenced by a connection does not exist.
    InvalidNode(NodeId),
    /// A connection forms a self-loop.
    SelfLoop(NodeId),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cycle(nodes) => write!(f, "Graph cycle involving nodes: {:?}", nodes),
            Self::UnconnectedRequiredInput { node, port } => {
                write!(f, "Required input '{}' on node {} is unconnected", port, node)
            }
            Self::TypeMismatch { from_node, from_port, from_type, to_node, to_port, to_type } => {
                write!(
                    f,
                    "Type mismatch: {}:{} ({:?}) -> {}:{} ({:?})",
                    from_node, from_port, from_type, to_node, to_port, to_type
                )
            }
            Self::NoOutputNode => write!(f, "No PBR output node in graph"),
            Self::MultipleOutputNodes(nodes) => {
                write!(f, "Multiple output nodes: {:?}", nodes)
            }
            Self::InvalidPort { node, port } => {
                write!(f, "Invalid port '{}' on node {}", port, node)
            }
            Self::InvalidNode(id) => write!(f, "Invalid node: {}", id),
            Self::SelfLoop(id) => write!(f, "Self-loop on node {}", id),
        }
    }
}

// ---------------------------------------------------------------------------
// ShaderGraph
// ---------------------------------------------------------------------------

/// A directed acyclic graph of shader nodes.
pub struct ShaderGraph {
    /// All nodes in the graph.
    pub nodes: HashMap<NodeId, ShaderNode>,
    /// Directed edges (connections between ports).
    pub connections: Vec<Connection>,
    /// Next node id to assign.
    next_id: NodeId,
    /// Graph name / label.
    pub name: String,
    /// Description / documentation.
    pub description: String,
}

impl ShaderGraph {
    /// Create a new empty shader graph.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            nodes: HashMap::new(),
            connections: Vec::new(),
            next_id: 1,
            name: name.into(),
            description: String::new(),
        }
    }

    /// Add a node to the graph, returning its assigned id.
    pub fn add_node(&mut self, kind: NodeKind) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let node = ShaderNode::new(id, kind);
        self.nodes.insert(id, node);
        id
    }

    /// Add a node with a custom label.
    pub fn add_node_labeled(&mut self, kind: NodeKind, label: impl Into<String>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let node = ShaderNode::new(id, kind).with_label(label);
        self.nodes.insert(id, node);
        id
    }

    /// Remove a node and all its connections.
    pub fn remove_node(&mut self, id: NodeId) {
        self.nodes.remove(&id);
        self.connections
            .retain(|c| c.from_node != id && c.to_node != id);
    }

    /// Connect an output port to an input port.
    pub fn connect(
        &mut self,
        from_node: NodeId,
        from_port: impl Into<String>,
        to_node: NodeId,
        to_port: impl Into<String>,
    ) {
        let to_port_str = to_port.into();
        // Remove any existing connection to this input port.
        self.connections
            .retain(|c| !(c.to_node == to_node && c.to_port == to_port_str));

        self.connections.push(Connection {
            from_node,
            from_port: from_port.into(),
            to_node,
            to_port: to_port_str,
        });
    }

    /// Disconnect a specific connection.
    pub fn disconnect(&mut self, from_node: NodeId, from_port: &str, to_node: NodeId, to_port: &str) {
        self.connections.retain(|c| {
            !(c.from_node == from_node
                && c.from_port == from_port
                && c.to_node == to_node
                && c.to_port == to_port)
        });
    }

    /// Get all connections feeding into a node's input ports.
    pub fn inputs_for(&self, node_id: NodeId) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|c| c.to_node == node_id)
            .collect()
    }

    /// Get all connections from a node's output ports.
    pub fn outputs_from(&self, node_id: NodeId) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|c| c.from_node == node_id)
            .collect()
    }

    /// Get the node providing a given input port's value.
    pub fn input_source(&self, node_id: NodeId, port_name: &str) -> Option<&Connection> {
        self.connections
            .iter()
            .find(|c| c.to_node == node_id && c.to_port == port_name)
    }

    /// Find the output node (PBROutput).
    pub fn find_output_node(&self) -> Option<NodeId> {
        self.nodes
            .iter()
            .find(|(_, n)| matches!(n.kind, NodeKind::PBROutput))
            .map(|(&id, _)| id)
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

// ---------------------------------------------------------------------------
// Graph validation
// ---------------------------------------------------------------------------

/// Validate a shader graph, checking for:
/// - Cycles
/// - Type mismatches
/// - Unconnected required inputs
/// - Missing output node
/// - Self-loops
/// - Invalid node / port references
pub fn validate_graph(graph: &ShaderGraph) -> Vec<GraphError> {
    let mut errors = Vec::new();

    // Check for output node.
    let output_nodes: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter(|(_, n)| matches!(n.kind, NodeKind::PBROutput))
        .map(|(&id, _)| id)
        .collect();

    if output_nodes.is_empty() {
        errors.push(GraphError::NoOutputNode);
    } else if output_nodes.len() > 1 {
        errors.push(GraphError::MultipleOutputNodes(output_nodes));
    }

    // Check connections for validity.
    for conn in &graph.connections {
        // Self-loops.
        if conn.from_node == conn.to_node {
            errors.push(GraphError::SelfLoop(conn.from_node));
            continue;
        }

        // Invalid node references.
        let from_node = match graph.nodes.get(&conn.from_node) {
            Some(n) => n,
            None => {
                errors.push(GraphError::InvalidNode(conn.from_node));
                continue;
            }
        };
        let to_node = match graph.nodes.get(&conn.to_node) {
            Some(n) => n,
            None => {
                errors.push(GraphError::InvalidNode(conn.to_node));
                continue;
            }
        };

        // Invalid port references.
        let from_port = from_node
            .output_ports()
            .iter()
            .find(|p| p.name == conn.from_port)
            .cloned();
        let to_port = to_node
            .input_ports()
            .iter()
            .find(|p| p.name == conn.to_port)
            .cloned();

        if from_port.is_none() {
            errors.push(GraphError::InvalidPort {
                node: conn.from_node,
                port: conn.from_port.clone(),
            });
            continue;
        }
        if to_port.is_none() {
            errors.push(GraphError::InvalidPort {
                node: conn.to_node,
                port: conn.to_port.clone(),
            });
            continue;
        }

        // Type mismatch.
        let from_type = from_port.as_ref().unwrap().data_type;
        let to_type = to_port.as_ref().unwrap().data_type;
        if !from_type.can_convert_to(&to_type) {
            errors.push(GraphError::TypeMismatch {
                from_node: conn.from_node,
                from_port: conn.from_port.clone(),
                from_type,
                to_node: conn.to_node,
                to_port: conn.to_port.clone(),
                to_type,
            });
        }
    }

    // Check unconnected required inputs.
    for (&node_id, node) in &graph.nodes {
        for port in node.input_ports() {
            if port.required && port.default_value.is_none() {
                let is_connected = graph
                    .connections
                    .iter()
                    .any(|c| c.to_node == node_id && c.to_port == port.name);
                if !is_connected {
                    errors.push(GraphError::UnconnectedRequiredInput {
                        node: node_id,
                        port: port.name.clone(),
                    });
                }
            }
        }
    }

    // Check for cycles using DFS.
    if let Some(cycle) = detect_cycle(graph) {
        errors.push(GraphError::Cycle(cycle));
    }

    errors
}

/// Detect a cycle in the graph using DFS. Returns the nodes involved in the
/// cycle, or `None` if the graph is acyclic.
fn detect_cycle(graph: &ShaderGraph) -> Option<Vec<NodeId>> {
    let mut visited = HashMap::new(); // 0 = unvisited, 1 = in-progress, 2 = done
    for &id in graph.nodes.keys() {
        visited.insert(id, 0u8);
    }

    let mut path = Vec::new();

    for &start in graph.nodes.keys() {
        if visited[&start] == 0 {
            if dfs_cycle(graph, start, &mut visited, &mut path) {
                return Some(path);
            }
        }
    }

    None
}

fn dfs_cycle(
    graph: &ShaderGraph,
    node: NodeId,
    visited: &mut HashMap<NodeId, u8>,
    path: &mut Vec<NodeId>,
) -> bool {
    visited.insert(node, 1);
    path.push(node);

    // Find all successors (nodes this node feeds into).
    for conn in &graph.connections {
        if conn.from_node == node {
            let next = conn.to_node;
            match visited.get(&next) {
                Some(1) => {
                    // Found a cycle -- trim path to just the cycle.
                    if let Some(pos) = path.iter().position(|&n| n == next) {
                        *path = path[pos..].to_vec();
                    }
                    return true;
                }
                Some(0) => {
                    if dfs_cycle(graph, next, visited, path) {
                        return true;
                    }
                }
                _ => {} // Already fully processed.
            }
        }
    }

    path.pop();
    visited.insert(node, 2);
    false
}

// ---------------------------------------------------------------------------
// Topological sort
// ---------------------------------------------------------------------------

/// Topologically sort the graph nodes. Returns the node ids in evaluation
/// order (dependencies before dependents).
fn topological_sort(graph: &ShaderGraph) -> Result<Vec<NodeId>, Vec<GraphError>> {
    let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
    for &id in graph.nodes.keys() {
        in_degree.insert(id, 0);
    }

    // Edges go from `from_node` to `to_node`.
    let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for conn in &graph.connections {
        adjacency
            .entry(conn.from_node)
            .or_default()
            .push(conn.to_node);
        *in_degree.entry(conn.to_node).or_insert(0) += 1;
    }

    let mut queue: Vec<NodeId> = in_degree
        .iter()
        .filter(|(_, deg)| **deg == 0)
        .map(|(&id, _)| id)
        .collect();
    queue.sort(); // Deterministic order.

    let mut sorted = Vec::with_capacity(graph.nodes.len());

    while let Some(node) = queue.pop() {
        sorted.push(node);
        if let Some(neighbors) = adjacency.get(&node) {
            for &next in neighbors {
                let deg = in_degree.get_mut(&next).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(next);
                }
            }
        }
    }

    if sorted.len() != graph.nodes.len() {
        let cycle_nodes: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg > 0)
            .map(|(&id, _)| id)
            .collect();
        return Err(vec![GraphError::Cycle(cycle_nodes)]);
    }

    Ok(sorted)
}

// ---------------------------------------------------------------------------
// WGSL code generation
// ---------------------------------------------------------------------------

/// Compile a shader graph to WGSL source code.
///
/// The generated code includes:
/// - Struct definitions for vertex input/output.
/// - Uniform buffer bindings for camera and globals.
/// - Texture/sampler bindings for all texture-sampling nodes.
/// - A vertex shader that passes through standard attributes.
/// - A fragment shader with all node computations in topological order.
/// - PBR output variables ready for a lighting pass.
pub fn compile_to_wgsl(graph: &ShaderGraph) -> Result<String, Vec<GraphError>> {
    // Validate first.
    let errors = validate_graph(graph);
    // Filter out non-fatal errors (allow "required with default" through).
    let fatal: Vec<GraphError> = errors
        .into_iter()
        .filter(|e| {
            !matches!(
                e,
                GraphError::UnconnectedRequiredInput { .. }
            )
        })
        .collect();
    if !fatal.is_empty() {
        return Err(fatal);
    }

    let sorted = topological_sort(graph)?;

    let output_node_id = graph
        .find_output_node()
        .ok_or_else(|| vec![GraphError::NoOutputNode])?;

    // Collect texture nodes for binding declarations.
    let texture_nodes: Vec<&ShaderNode> = sorted
        .iter()
        .filter_map(|id| graph.nodes.get(id))
        .filter(|n| n.kind.needs_texture_binding())
        .collect();

    let mut wgsl = String::with_capacity(8192);

    // ----- Header -----
    wgsl.push_str("// Auto-generated by Genovo Shader Graph\n");
    wgsl.push_str(&format!("// Graph: {}\n\n", graph.name));

    // ----- Struct definitions -----
    wgsl.push_str(
        "struct VertexInput {\n\
             @location(0) position: vec3<f32>,\n\
             @location(1) normal: vec3<f32>,\n\
             @location(2) tangent: vec4<f32>,\n\
             @location(3) uv0: vec2<f32>,\n\
             @location(4) uv1: vec2<f32>,\n\
             @location(5) vertex_color: vec4<f32>,\n\
         };\n\n",
    );

    wgsl.push_str(
        "struct VertexOutput {\n\
             @builtin(position) clip_position: vec4<f32>,\n\
             @location(0) world_position: vec3<f32>,\n\
             @location(1) world_normal: vec3<f32>,\n\
             @location(2) world_tangent: vec4<f32>,\n\
             @location(3) uv0: vec2<f32>,\n\
             @location(4) uv1: vec2<f32>,\n\
             @location(5) vertex_color: vec4<f32>,\n\
         };\n\n",
    );

    // ----- Uniform bindings -----
    wgsl.push_str(
        "struct CameraUniform {\n\
             view: mat4x4<f32>,\n\
             projection: mat4x4<f32>,\n\
             view_projection: mat4x4<f32>,\n\
             inverse_view: mat4x4<f32>,\n\
             inverse_projection: mat4x4<f32>,\n\
             position: vec4<f32>,\n\
             screen_size: vec2<f32>,\n\
             near_far: vec2<f32>,\n\
         };\n\n",
    );

    wgsl.push_str(
        "struct Globals {\n\
             time: f32,\n\
             delta_time: f32,\n\
             frame_count: u32,\n\
             _padding: u32,\n\
         };\n\n",
    );

    wgsl.push_str(
        "struct ModelUniform {\n\
             model: mat4x4<f32>,\n\
             normal_matrix: mat4x4<f32>,\n\
         };\n\n",
    );

    // PBR output struct for the fragment shader return.
    wgsl.push_str(
        "struct FragmentOutput {\n\
             @location(0) albedo: vec4<f32>,\n\
             @location(1) normal: vec4<f32>,\n\
             @location(2) metallic_roughness_ao: vec4<f32>,\n\
             @location(3) emissive: vec4<f32>,\n\
         };\n\n",
    );

    // ----- Bind groups -----
    wgsl.push_str("@group(0) @binding(0) var<uniform> camera: CameraUniform;\n");
    wgsl.push_str("@group(0) @binding(1) var<uniform> globals: Globals;\n");
    wgsl.push_str("@group(0) @binding(2) var<uniform> model: ModelUniform;\n\n");

    // Texture bindings for each texture-sampling node.
    let mut binding_index = 0u32;
    for node in &texture_nodes {
        let prefix = node.var_prefix();
        let tex_name = node
            .texture_name
            .as_deref()
            .unwrap_or(&prefix);
        wgsl.push_str(&format!(
            "@group(1) @binding({}) var t_{}: texture_2d<f32>;\n",
            binding_index, tex_name,
        ));
        binding_index += 1;
        wgsl.push_str(&format!(
            "@group(1) @binding({}) var s_{}: sampler;\n",
            binding_index, tex_name,
        ));
        binding_index += 1;
    }
    if !texture_nodes.is_empty() {
        wgsl.push('\n');
    }

    // ----- Vertex shader -----
    wgsl.push_str(
        "@vertex\n\
         fn vs_main(in: VertexInput) -> VertexOutput {\n\
             var out: VertexOutput;\n\
             let world_pos = model.model * vec4<f32>(in.position, 1.0);\n\
             out.clip_position = camera.view_projection * world_pos;\n\
             out.world_position = world_pos.xyz;\n\
             out.world_normal = normalize((model.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);\n\
             out.world_tangent = vec4<f32>(\n\
                 normalize((model.normal_matrix * vec4<f32>(in.tangent.xyz, 0.0)).xyz),\n\
                 in.tangent.w\n\
             );\n\
             out.uv0 = in.uv0;\n\
             out.uv1 = in.uv1;\n\
             out.vertex_color = in.vertex_color;\n\
             return out;\n\
         }\n\n",
    );

    // ----- Fragment shader -----
    wgsl.push_str("@fragment\nfn fs_main(in: VertexOutput) -> FragmentOutput {\n");

    // Generate code for each node in topological order.
    for &node_id in &sorted {
        let node = match graph.nodes.get(&node_id) {
            Some(n) => n,
            None => continue,
        };

        let prefix = node.var_prefix();

        // Build input variable map.
        let mut input_vars: HashMap<String, String> = HashMap::new();
        for input_port in node.input_ports() {
            if let Some(conn) = graph.input_source(node_id, &input_port.name) {
                let source_node = match graph.nodes.get(&conn.from_node) {
                    Some(n) => n,
                    None => continue,
                };
                let source_prefix = source_node.var_prefix();
                let var_name = format!("{}_{}", source_prefix, conn.from_port);

                // Check if type conversion is needed.
                let source_port_type = source_node
                    .output_ports()
                    .iter()
                    .find(|p| p.name == conn.from_port)
                    .map(|p| p.data_type);
                let target_port_type = Some(input_port.data_type);

                if let (Some(src_type), Some(dst_type)) = (source_port_type, target_port_type) {
                    if let Some(conv) = src_type.conversion_expr(&var_name, &dst_type) {
                        input_vars.insert(input_port.name.clone(), conv);
                    } else {
                        input_vars.insert(input_port.name.clone(), var_name);
                    }
                } else {
                    input_vars.insert(input_port.name.clone(), var_name);
                }
            }
            // If not connected, the default value from `generate_code` will be used.
        }

        // For texture nodes, override the prefix in the generated code to use
        // the texture name if available.
        let effective_prefix = if node.kind.needs_texture_binding() {
            node.texture_name.clone().unwrap_or_else(|| prefix.clone())
        } else {
            prefix
        };

        let code = node.kind.generate_code(&input_vars, &effective_prefix);
        wgsl.push_str(&format!("    // Node {}: {}\n", node_id, node.label));
        wgsl.push_str(&code);
        wgsl.push('\n');
    }

    // ----- Fragment output -----
    wgsl.push_str(
        "    var out: FragmentOutput;\n\
             out.albedo = vec4<f32>(pbr_albedo, pbr_alpha);\n\
             out.normal = vec4<f32>(pbr_normal * 0.5 + 0.5, 1.0);\n\
             out.metallic_roughness_ao = vec4<f32>(pbr_metallic, pbr_roughness, pbr_ao, 1.0);\n\
             out.emissive = vec4<f32>(pbr_emissive, 1.0);\n\
             return out;\n\
         }\n",
    );

    Ok(wgsl)
}

// ---------------------------------------------------------------------------
// Helper: build a minimal PBR graph
// ---------------------------------------------------------------------------

/// Build a simple PBR shader graph with constant albedo, metallic, roughness.
/// Useful as a starting template in the editor.
pub fn build_default_pbr_graph() -> ShaderGraph {
    let mut graph = ShaderGraph::new("Default PBR");

    let albedo = graph.add_node_labeled(NodeKind::ConstVec3([1.0, 1.0, 1.0]), "Albedo Color");
    let metallic = graph.add_node_labeled(NodeKind::ConstFloat(0.0), "Metallic");
    let roughness = graph.add_node_labeled(NodeKind::ConstFloat(0.5), "Roughness");
    let output = graph.add_node_labeled(NodeKind::PBROutput, "PBR Output");

    graph.connect(albedo, "value", output, "albedo");
    graph.connect(metallic, "value", output, "metallic");
    graph.connect(roughness, "value", output, "roughness");

    graph
}

/// Build a textured PBR graph that samples an albedo texture.
pub fn build_textured_pbr_graph() -> ShaderGraph {
    let mut graph = ShaderGraph::new("Textured PBR");

    let uv = graph.add_node_labeled(NodeKind::UV0, "UV Coords");
    let tex = graph.add_node_labeled(NodeKind::SampleTexture2D, "Albedo Texture");
    if let Some(node) = graph.nodes.get_mut(&tex) {
        node.texture_name = Some("albedo".to_string());
    }
    let metallic = graph.add_node_labeled(NodeKind::ConstFloat(0.0), "Metallic");
    let roughness = graph.add_node_labeled(NodeKind::ConstFloat(0.5), "Roughness");
    let output = graph.add_node_labeled(NodeKind::PBROutput, "PBR Output");

    graph.connect(uv, "uv", tex, "uv");
    graph.connect(tex, "color", output, "albedo");
    graph.connect(metallic, "value", output, "metallic");
    graph.connect(roughness, "value", output, "roughness");

    graph
}

/// Build a graph that demonstrates math nodes (lerp between two colors).
pub fn build_lerp_demo_graph() -> ShaderGraph {
    let mut graph = ShaderGraph::new("Lerp Demo");

    let color_a = graph.add_node_labeled(NodeKind::ConstVec3([1.0, 0.0, 0.0]), "Red");
    let color_b = graph.add_node_labeled(NodeKind::ConstVec3([0.0, 0.0, 1.0]), "Blue");
    let uv = graph.add_node_labeled(NodeKind::UV0, "UV");

    // Use the U coordinate as the lerp factor.
    let split_x = graph.add_node_labeled(
        NodeKind::CustomExpression("$input0.x".to_string()),
        "Split X",
    );

    let lerp = graph.add_node_labeled(NodeKind::Lerp, "Color Lerp");
    let output = graph.add_node_labeled(NodeKind::PBROutput, "Output");

    graph.connect(uv, "uv", split_x, "input0");
    graph.connect(color_a, "value", lerp, "a");
    graph.connect(color_b, "value", lerp, "b");
    graph.connect(split_x, "result", lerp, "t");
    graph.connect(lerp, "result", output, "albedo");

    graph
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_conversion_scalar_to_vec() {
        assert!(DataType::Float.can_convert_to(&DataType::Vec3));
        assert!(DataType::Float.can_convert_to(&DataType::Vec4));
        assert!(!DataType::Vec4.can_convert_to(&DataType::Float));
        assert!(DataType::Vec3.can_convert_to(&DataType::Vec4));
        assert!(DataType::Vec4.can_convert_to(&DataType::Vec3));
    }

    #[test]
    fn conversion_expressions() {
        let expr = DataType::Float.conversion_expr("x", &DataType::Vec3);
        assert_eq!(expr, Some("vec3<f32>(x, x, x)".to_string()));

        let expr = DataType::Vec3.conversion_expr("v", &DataType::Vec4);
        assert_eq!(expr, Some("vec4<f32>(v.xyz, 1.0)".to_string()));

        let expr = DataType::Vec4.conversion_expr("v", &DataType::Vec3);
        assert_eq!(expr, Some("v.xyz".to_string()));

        assert!(DataType::Float.conversion_expr("x", &DataType::Float).is_none());
    }

    #[test]
    fn build_and_validate_default_graph() {
        let graph = build_default_pbr_graph();
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.connection_count(), 3);

        let errors = validate_graph(&graph);
        // Should have no fatal errors. May have "unconnected required input"
        // for PBR inputs with defaults.
        for e in &errors {
            match e {
                GraphError::Cycle(_) | GraphError::TypeMismatch { .. } | GraphError::NoOutputNode => {
                    panic!("Unexpected error: {:?}", e);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn compile_default_graph_to_wgsl() {
        let graph = build_default_pbr_graph();
        let result = compile_to_wgsl(&graph);
        assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

        let wgsl = result.unwrap();
        assert!(wgsl.contains("fn vs_main"));
        assert!(wgsl.contains("fn fs_main"));
        assert!(wgsl.contains("pbr_albedo"));
        assert!(wgsl.contains("pbr_metallic"));
        assert!(wgsl.contains("pbr_roughness"));
        assert!(wgsl.contains("FragmentOutput"));
    }

    #[test]
    fn compile_textured_graph_has_texture_bindings() {
        let graph = build_textured_pbr_graph();
        let result = compile_to_wgsl(&graph);
        assert!(result.is_ok());

        let wgsl = result.unwrap();
        assert!(wgsl.contains("t_albedo"));
        assert!(wgsl.contains("s_albedo"));
        assert!(wgsl.contains("textureSample"));
    }

    #[test]
    fn detect_cycle_in_graph() {
        let mut graph = ShaderGraph::new("Cycle Test");
        let a = graph.add_node(NodeKind::Add);
        let b = graph.add_node(NodeKind::Add);
        let c = graph.add_node(NodeKind::PBROutput);

        graph.connect(a, "result", b, "a");
        graph.connect(b, "result", a, "a"); // Cycle!
        graph.connect(b, "result", c, "albedo");

        let errors = validate_graph(&graph);
        let has_cycle = errors.iter().any(|e| matches!(e, GraphError::Cycle(_)));
        assert!(has_cycle, "Expected cycle error, got: {:?}", errors);
    }

    #[test]
    fn type_mismatch_detection() {
        let mut graph = ShaderGraph::new("Type Mismatch");
        let mat = graph.add_node(NodeKind::ConstFloat(1.0));
        let output = graph.add_node(NodeKind::PBROutput);

        // Connect a Float to a Vec3 input -- this should be OK (widening).
        graph.connect(mat, "value", output, "albedo");

        let errors = validate_graph(&graph);
        let type_errors: Vec<_> = errors
            .iter()
            .filter(|e| matches!(e, GraphError::TypeMismatch { .. }))
            .collect();
        // Float -> Vec3 is allowed via implicit conversion, so no type error.
        assert!(type_errors.is_empty());
    }

    #[test]
    fn no_output_node_error() {
        let mut graph = ShaderGraph::new("No Output");
        graph.add_node(NodeKind::ConstFloat(1.0));

        let errors = validate_graph(&graph);
        let has_no_output = errors.iter().any(|e| matches!(e, GraphError::NoOutputNode));
        assert!(has_no_output);
    }

    #[test]
    fn self_loop_detection() {
        let mut graph = ShaderGraph::new("Self Loop");
        let a = graph.add_node(NodeKind::Add);
        let output = graph.add_node(NodeKind::PBROutput);

        graph.connect(a, "result", a, "a"); // Self-loop.
        graph.connect(a, "result", output, "albedo");

        let errors = validate_graph(&graph);
        let has_self_loop = errors.iter().any(|e| matches!(e, GraphError::SelfLoop(_)));
        assert!(has_self_loop);
    }

    #[test]
    fn node_add_and_remove() {
        let mut graph = ShaderGraph::new("Test");
        let a = graph.add_node(NodeKind::ConstFloat(1.0));
        let b = graph.add_node(NodeKind::Add);
        graph.connect(a, "value", b, "a");

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.connection_count(), 1);

        graph.remove_node(a);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.connection_count(), 0);
    }

    #[test]
    fn math_node_code_generation() {
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), "x".to_string());
        inputs.insert("b".to_string(), "y".to_string());

        let code = NodeKind::Add.generate_code(&inputs, "test");
        assert!(code.contains("let test_result = x + y;"));

        let code = NodeKind::Multiply.generate_code(&inputs, "test");
        assert!(code.contains("let test_result = x * y;"));

        let code = NodeKind::Divide.generate_code(&inputs, "test");
        assert!(code.contains("let test_result = x / max(y, 0.00001);"));
    }

    #[test]
    fn lerp_demo_graph_compiles() {
        let graph = build_lerp_demo_graph();
        let result = compile_to_wgsl(&graph);
        assert!(result.is_ok(), "Failed: {:?}", result.err());
        let wgsl = result.unwrap();
        assert!(wgsl.contains("mix("));
    }

    #[test]
    fn data_type_wgsl_names() {
        assert_eq!(DataType::Float.wgsl_name(), "f32");
        assert_eq!(DataType::Vec3.wgsl_name(), "vec3<f32>");
        assert_eq!(DataType::Mat4.wgsl_name(), "mat4x4<f32>");
        assert_eq!(DataType::Texture2D.wgsl_name(), "texture_2d<f32>");
    }

    #[test]
    fn node_kind_port_counts() {
        assert_eq!(NodeKind::Add.input_ports().len(), 2);
        assert_eq!(NodeKind::Add.output_ports().len(), 1);
        assert_eq!(NodeKind::PBROutput.input_ports().len(), 7);
        assert_eq!(NodeKind::PBROutput.output_ports().len(), 0);
        assert_eq!(NodeKind::SampleTexture2D.output_ports().len(), 5);
        assert_eq!(NodeKind::Lerp.input_ports().len(), 3);
    }
}
