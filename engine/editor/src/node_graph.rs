// =============================================================================
// Genovo Engine - Visual Node Graph Editor
// =============================================================================
//
// A fully-featured node graph system used by the material editor and visual
// scripting systems. Supports arbitrary node types, typed pin connections,
// topological evaluation, cycle detection, and JSON serialization.

use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// IDs
// ---------------------------------------------------------------------------

/// Unique identifier for a node within a graph.
pub type NodeId = Uuid;

/// Unique identifier for a pin (input or output) on a node.
pub type PinId = Uuid;

// ---------------------------------------------------------------------------
// Pin Data Types
// ---------------------------------------------------------------------------

/// The data type carried by a pin connection. Used for compatibility checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PinDataType {
    /// Single floating-point scalar.
    Float,
    /// 2-component vector.
    Vec2,
    /// 3-component vector.
    Vec3,
    /// 4-component vector / color.
    Vec4,
    /// RGBA color (semantically distinct from Vec4 for UI purposes).
    Color,
    /// Boolean value.
    Bool,
    /// Integer value.
    Int,
    /// String / identifier.
    String,
    /// 2D texture sampler.
    Texture2D,
    /// A material output aggregate.
    Material,
    /// Execution flow pin (visual scripting).
    Flow,
    /// Accepts any type (auto-converts where possible).
    Any,
}

impl PinDataType {
    /// Returns `true` if `source` can connect to a pin of type `target`.
    pub fn is_compatible(source: PinDataType, target: PinDataType) -> bool {
        if source == target {
            return true;
        }
        if target == PinDataType::Any || source == PinDataType::Any {
            return true;
        }
        // Allow float -> vec promotions.
        if source == PinDataType::Float {
            return matches!(
                target,
                PinDataType::Vec2 | PinDataType::Vec3 | PinDataType::Vec4 | PinDataType::Color
            );
        }
        // Color and Vec4 are interchangeable.
        if (source == PinDataType::Color && target == PinDataType::Vec4)
            || (source == PinDataType::Vec4 && target == PinDataType::Color)
        {
            return true;
        }
        // Int -> Float promotion.
        if source == PinDataType::Int && target == PinDataType::Float {
            return true;
        }
        false
    }

    /// A display-friendly name for the type.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Float => "Float",
            Self::Vec2 => "Vec2",
            Self::Vec3 => "Vec3",
            Self::Vec4 => "Vec4",
            Self::Color => "Color",
            Self::Bool => "Bool",
            Self::Int => "Int",
            Self::String => "String",
            Self::Texture2D => "Texture2D",
            Self::Material => "Material",
            Self::Flow => "Flow",
            Self::Any => "Any",
        }
    }

    /// UI color for rendering this pin type (RGBA).
    pub fn color(&self) -> [f32; 4] {
        match self {
            Self::Float => [0.55, 0.85, 0.55, 1.0],
            Self::Vec2 => [0.55, 0.75, 0.95, 1.0],
            Self::Vec3 => [0.65, 0.55, 0.95, 1.0],
            Self::Vec4 => [0.85, 0.55, 0.85, 1.0],
            Self::Color => [0.95, 0.85, 0.25, 1.0],
            Self::Bool => [0.85, 0.35, 0.35, 1.0],
            Self::Int => [0.35, 0.85, 0.85, 1.0],
            Self::String => [0.85, 0.55, 0.55, 1.0],
            Self::Texture2D => [0.85, 0.65, 0.25, 1.0],
            Self::Material => [0.95, 0.95, 0.95, 1.0],
            Self::Flow => [1.0, 1.0, 1.0, 1.0],
            Self::Any => [0.7, 0.7, 0.7, 1.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Node Pin
// ---------------------------------------------------------------------------

/// A single input or output slot on a graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePin {
    /// Unique pin ID.
    pub id: PinId,
    /// Display name (e.g., "Albedo", "Value", "Out").
    pub name: String,
    /// The data type this pin produces or accepts.
    pub data_type: PinDataType,
    /// For input pins, the ID of the pin connected to this input (if any).
    pub connected_to: Option<PinId>,
    /// Default value used when the pin is unconnected.
    pub default_value: Option<Value>,
}

impl NodePin {
    /// Create a new pin with the given name and data type.
    pub fn new(name: impl Into<String>, data_type: PinDataType) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            data_type,
            connected_to: None,
            default_value: None,
        }
    }

    /// Builder: set a default value for unconnected inputs.
    pub fn with_default(mut self, value: Value) -> Self {
        self.default_value = Some(value);
        self
    }
}

// ---------------------------------------------------------------------------
// Node Connection
// ---------------------------------------------------------------------------

/// A directed edge in the node graph, from an output pin to an input pin.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeConnection {
    /// Source node.
    pub from_node: NodeId,
    /// Output pin on the source node.
    pub from_pin: PinId,
    /// Destination node.
    pub to_node: NodeId,
    /// Input pin on the destination node.
    pub to_pin: PinId,
}

// ---------------------------------------------------------------------------
// Value (runtime evaluation result)
// ---------------------------------------------------------------------------

/// An evaluated value flowing through the graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Value {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
    Bool(bool),
    Int(i32),
    String(String),
    /// Texture reference (UUID of the texture asset).
    Texture(Option<Uuid>),
    /// No value / void.
    None,
}

impl Value {
    /// Attempt to convert this value to a float.
    pub fn as_float(&self) -> f32 {
        match self {
            Value::Float(v) => *v,
            Value::Int(v) => *v as f32,
            Value::Bool(v) => if *v { 1.0 } else { 0.0 },
            _ => 0.0,
        }
    }

    /// Attempt to convert this value to a Vec3.
    pub fn as_vec3(&self) -> [f32; 3] {
        match self {
            Value::Vec3(v) => *v,
            Value::Vec4(v) | Value::Color(v) => [v[0], v[1], v[2]],
            Value::Vec2(v) => [v[0], v[1], 0.0],
            Value::Float(v) => [*v, *v, *v],
            Value::Int(v) => { let f = *v as f32; [f, f, f] }
            _ => [0.0, 0.0, 0.0],
        }
    }

    /// Attempt to convert this value to a Vec4.
    pub fn as_vec4(&self) -> [f32; 4] {
        match self {
            Value::Vec4(v) | Value::Color(v) => *v,
            Value::Vec3(v) => [v[0], v[1], v[2], 1.0],
            Value::Vec2(v) => [v[0], v[1], 0.0, 1.0],
            Value::Float(v) => [*v, *v, *v, 1.0],
            _ => [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Attempt to interpret as a bool.
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(v) => *v,
            Value::Float(v) => *v != 0.0,
            Value::Int(v) => *v != 0,
            _ => false,
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::None
    }
}

// ---------------------------------------------------------------------------
// Node Type Registry
// ---------------------------------------------------------------------------

/// Describes a built-in node type available in the graph editor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTypeDescriptor {
    /// Unique type identifier (e.g., "math_add", "texture_sample").
    pub type_id: String,
    /// Display name shown in the node palette.
    pub display_name: String,
    /// Category for grouping in the creation menu.
    pub category: String,
    /// Description / tooltip.
    pub description: String,
    /// Default input pins for this node type.
    pub default_inputs: Vec<(String, PinDataType, Option<Value>)>,
    /// Default output pins for this node type.
    pub default_outputs: Vec<(String, PinDataType)>,
}

/// The built-in node library containing 30+ standard node types.
pub fn built_in_node_library() -> Vec<NodeTypeDescriptor> {
    vec![
        // --- Material nodes ---
        NodeTypeDescriptor {
            type_id: "material_output".into(),
            display_name: "Material Output".into(),
            category: "Material".into(),
            description: "Final material output node".into(),
            default_inputs: vec![
                ("Albedo".into(), PinDataType::Color, Some(Value::Color([0.8, 0.8, 0.8, 1.0]))),
                ("Metallic".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("Roughness".into(), PinDataType::Float, Some(Value::Float(0.5))),
                ("Normal".into(), PinDataType::Vec3, Some(Value::Vec3([0.0, 0.0, 1.0]))),
                ("Emissive".into(), PinDataType::Color, Some(Value::Color([0.0, 0.0, 0.0, 1.0]))),
                ("Opacity".into(), PinDataType::Float, Some(Value::Float(1.0))),
                ("AO".into(), PinDataType::Float, Some(Value::Float(1.0))),
            ],
            default_outputs: vec![],
        },
        NodeTypeDescriptor {
            type_id: "texture_sample".into(),
            display_name: "Texture Sample".into(),
            category: "Texture".into(),
            description: "Sample a 2D texture at UV coordinates".into(),
            default_inputs: vec![
                ("Texture".into(), PinDataType::Texture2D, None),
                ("UV".into(), PinDataType::Vec2, Some(Value::Vec2([0.0, 0.0]))),
            ],
            default_outputs: vec![
                ("RGBA".into(), PinDataType::Color),
                ("R".into(), PinDataType::Float),
                ("G".into(), PinDataType::Float),
                ("B".into(), PinDataType::Float),
                ("A".into(), PinDataType::Float),
            ],
        },
        NodeTypeDescriptor {
            type_id: "texture_param".into(),
            display_name: "Texture Parameter".into(),
            category: "Texture".into(),
            description: "A texture parameter exposed to material instances".into(),
            default_inputs: vec![],
            default_outputs: vec![("Texture".into(), PinDataType::Texture2D)],
        },
        NodeTypeDescriptor {
            type_id: "color_constant".into(),
            display_name: "Color".into(),
            category: "Constant".into(),
            description: "A constant color value".into(),
            default_inputs: vec![],
            default_outputs: vec![("Color".into(), PinDataType::Color)],
        },
        NodeTypeDescriptor {
            type_id: "float_constant".into(),
            display_name: "Float".into(),
            category: "Constant".into(),
            description: "A constant float value".into(),
            default_inputs: vec![],
            default_outputs: vec![("Value".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "vec2_constant".into(),
            display_name: "Vector2".into(),
            category: "Constant".into(),
            description: "A constant 2D vector".into(),
            default_inputs: vec![],
            default_outputs: vec![("Value".into(), PinDataType::Vec2)],
        },
        NodeTypeDescriptor {
            type_id: "vec3_constant".into(),
            display_name: "Vector3".into(),
            category: "Constant".into(),
            description: "A constant 3D vector".into(),
            default_inputs: vec![],
            default_outputs: vec![("Value".into(), PinDataType::Vec3)],
        },
        // --- Math nodes ---
        NodeTypeDescriptor {
            type_id: "math_add".into(),
            display_name: "Add".into(),
            category: "Math".into(),
            description: "Add two values".into(),
            default_inputs: vec![
                ("A".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("B".into(), PinDataType::Float, Some(Value::Float(0.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_subtract".into(),
            display_name: "Subtract".into(),
            category: "Math".into(),
            description: "Subtract B from A".into(),
            default_inputs: vec![
                ("A".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("B".into(), PinDataType::Float, Some(Value::Float(0.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_multiply".into(),
            display_name: "Multiply".into(),
            category: "Math".into(),
            description: "Multiply two values".into(),
            default_inputs: vec![
                ("A".into(), PinDataType::Float, Some(Value::Float(1.0))),
                ("B".into(), PinDataType::Float, Some(Value::Float(1.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_divide".into(),
            display_name: "Divide".into(),
            category: "Math".into(),
            description: "Divide A by B".into(),
            default_inputs: vec![
                ("A".into(), PinDataType::Float, Some(Value::Float(1.0))),
                ("B".into(), PinDataType::Float, Some(Value::Float(1.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_lerp".into(),
            display_name: "Lerp".into(),
            category: "Math".into(),
            description: "Linear interpolation between A and B by T".into(),
            default_inputs: vec![
                ("A".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("B".into(), PinDataType::Float, Some(Value::Float(1.0))),
                ("T".into(), PinDataType::Float, Some(Value::Float(0.5))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_clamp".into(),
            display_name: "Clamp".into(),
            category: "Math".into(),
            description: "Clamp a value between min and max".into(),
            default_inputs: vec![
                ("Value".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("Min".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("Max".into(), PinDataType::Float, Some(Value::Float(1.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_saturate".into(),
            display_name: "Saturate".into(),
            category: "Math".into(),
            description: "Clamp value to [0, 1]".into(),
            default_inputs: vec![
                ("Value".into(), PinDataType::Float, Some(Value::Float(0.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_power".into(),
            display_name: "Power".into(),
            category: "Math".into(),
            description: "Raise base to the exponent".into(),
            default_inputs: vec![
                ("Base".into(), PinDataType::Float, Some(Value::Float(2.0))),
                ("Exponent".into(), PinDataType::Float, Some(Value::Float(2.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_abs".into(),
            display_name: "Abs".into(),
            category: "Math".into(),
            description: "Absolute value".into(),
            default_inputs: vec![
                ("Value".into(), PinDataType::Float, Some(Value::Float(0.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_one_minus".into(),
            display_name: "One Minus".into(),
            category: "Math".into(),
            description: "1.0 - Value".into(),
            default_inputs: vec![
                ("Value".into(), PinDataType::Float, Some(Value::Float(0.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_sin".into(),
            display_name: "Sin".into(),
            category: "Math".into(),
            description: "Sine of the input (radians)".into(),
            default_inputs: vec![
                ("Value".into(), PinDataType::Float, Some(Value::Float(0.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_cos".into(),
            display_name: "Cos".into(),
            category: "Math".into(),
            description: "Cosine of the input (radians)".into(),
            default_inputs: vec![
                ("Value".into(), PinDataType::Float, Some(Value::Float(0.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_floor".into(),
            display_name: "Floor".into(),
            category: "Math".into(),
            description: "Floor (round down)".into(),
            default_inputs: vec![
                ("Value".into(), PinDataType::Float, Some(Value::Float(0.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_fract".into(),
            display_name: "Fract".into(),
            category: "Math".into(),
            description: "Fractional part".into(),
            default_inputs: vec![
                ("Value".into(), PinDataType::Float, Some(Value::Float(0.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "math_smoothstep".into(),
            display_name: "Smoothstep".into(),
            category: "Math".into(),
            description: "Hermite interpolation between edge0 and edge1".into(),
            default_inputs: vec![
                ("Edge0".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("Edge1".into(), PinDataType::Float, Some(Value::Float(1.0))),
                ("X".into(), PinDataType::Float, Some(Value::Float(0.5))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        // --- Vector nodes ---
        NodeTypeDescriptor {
            type_id: "vec_combine".into(),
            display_name: "Combine".into(),
            category: "Vector".into(),
            description: "Combine components into a vector".into(),
            default_inputs: vec![
                ("R/X".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("G/Y".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("B/Z".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("A/W".into(), PinDataType::Float, Some(Value::Float(1.0))),
            ],
            default_outputs: vec![
                ("Vec4".into(), PinDataType::Vec4),
                ("Vec3".into(), PinDataType::Vec3),
                ("Vec2".into(), PinDataType::Vec2),
            ],
        },
        NodeTypeDescriptor {
            type_id: "vec_split".into(),
            display_name: "Split".into(),
            category: "Vector".into(),
            description: "Split a vector into components".into(),
            default_inputs: vec![
                ("Vector".into(), PinDataType::Vec4, Some(Value::Vec4([0.0, 0.0, 0.0, 1.0]))),
            ],
            default_outputs: vec![
                ("R/X".into(), PinDataType::Float),
                ("G/Y".into(), PinDataType::Float),
                ("B/Z".into(), PinDataType::Float),
                ("A/W".into(), PinDataType::Float),
            ],
        },
        NodeTypeDescriptor {
            type_id: "vec_normalize".into(),
            display_name: "Normalize".into(),
            category: "Vector".into(),
            description: "Normalize a vector to unit length".into(),
            default_inputs: vec![
                ("Vector".into(), PinDataType::Vec3, Some(Value::Vec3([0.0, 1.0, 0.0]))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Vec3)],
        },
        NodeTypeDescriptor {
            type_id: "vec_dot".into(),
            display_name: "Dot Product".into(),
            category: "Vector".into(),
            description: "Dot product of two vectors".into(),
            default_inputs: vec![
                ("A".into(), PinDataType::Vec3, Some(Value::Vec3([1.0, 0.0, 0.0]))),
                ("B".into(), PinDataType::Vec3, Some(Value::Vec3([0.0, 1.0, 0.0]))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "vec_cross".into(),
            display_name: "Cross Product".into(),
            category: "Vector".into(),
            description: "Cross product of two 3D vectors".into(),
            default_inputs: vec![
                ("A".into(), PinDataType::Vec3, Some(Value::Vec3([1.0, 0.0, 0.0]))),
                ("B".into(), PinDataType::Vec3, Some(Value::Vec3([0.0, 1.0, 0.0]))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Vec3)],
        },
        // --- Color nodes ---
        NodeTypeDescriptor {
            type_id: "color_hsv_adjust".into(),
            display_name: "HSV Adjust".into(),
            category: "Color".into(),
            description: "Adjust hue, saturation, and value of a color".into(),
            default_inputs: vec![
                ("Color".into(), PinDataType::Color, Some(Value::Color([1.0, 1.0, 1.0, 1.0]))),
                ("Hue".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("Saturation".into(), PinDataType::Float, Some(Value::Float(1.0))),
                ("Value".into(), PinDataType::Float, Some(Value::Float(1.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Color)],
        },
        NodeTypeDescriptor {
            type_id: "color_gradient".into(),
            display_name: "Gradient".into(),
            category: "Color".into(),
            description: "Sample a color gradient at position T".into(),
            default_inputs: vec![
                ("T".into(), PinDataType::Float, Some(Value::Float(0.0))),
            ],
            default_outputs: vec![("Color".into(), PinDataType::Color)],
        },
        // --- UV nodes ---
        NodeTypeDescriptor {
            type_id: "uv_coords".into(),
            display_name: "UV Coordinates".into(),
            category: "UV".into(),
            description: "Mesh UV coordinates".into(),
            default_inputs: vec![],
            default_outputs: vec![("UV".into(), PinDataType::Vec2)],
        },
        NodeTypeDescriptor {
            type_id: "uv_scale_offset".into(),
            display_name: "UV Scale/Offset".into(),
            category: "UV".into(),
            description: "Scale and offset UV coordinates".into(),
            default_inputs: vec![
                ("UV".into(), PinDataType::Vec2, Some(Value::Vec2([0.0, 0.0]))),
                ("Scale".into(), PinDataType::Vec2, Some(Value::Vec2([1.0, 1.0]))),
                ("Offset".into(), PinDataType::Vec2, Some(Value::Vec2([0.0, 0.0]))),
            ],
            default_outputs: vec![("UV".into(), PinDataType::Vec2)],
        },
        NodeTypeDescriptor {
            type_id: "uv_rotate".into(),
            display_name: "UV Rotate".into(),
            category: "UV".into(),
            description: "Rotate UV coordinates around a center point".into(),
            default_inputs: vec![
                ("UV".into(), PinDataType::Vec2, Some(Value::Vec2([0.0, 0.0]))),
                ("Angle".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("Center".into(), PinDataType::Vec2, Some(Value::Vec2([0.5, 0.5]))),
            ],
            default_outputs: vec![("UV".into(), PinDataType::Vec2)],
        },
        NodeTypeDescriptor {
            type_id: "uv_parallax".into(),
            display_name: "Parallax Offset".into(),
            category: "UV".into(),
            description: "Parallax occlusion mapping UV offset".into(),
            default_inputs: vec![
                ("UV".into(), PinDataType::Vec2, Some(Value::Vec2([0.0, 0.0]))),
                ("Height".into(), PinDataType::Float, Some(Value::Float(0.0))),
                ("Scale".into(), PinDataType::Float, Some(Value::Float(0.04))),
            ],
            default_outputs: vec![("UV".into(), PinDataType::Vec2)],
        },
        // --- Shading nodes ---
        NodeTypeDescriptor {
            type_id: "fresnel".into(),
            display_name: "Fresnel".into(),
            category: "Shading".into(),
            description: "Fresnel effect based on view angle".into(),
            default_inputs: vec![
                ("Normal".into(), PinDataType::Vec3, Some(Value::Vec3([0.0, 0.0, 1.0]))),
                ("Power".into(), PinDataType::Float, Some(Value::Float(5.0))),
            ],
            default_outputs: vec![("Result".into(), PinDataType::Float)],
        },
        NodeTypeDescriptor {
            type_id: "view_direction".into(),
            display_name: "View Direction".into(),
            category: "Shading".into(),
            description: "Camera view direction in world space".into(),
            default_inputs: vec![],
            default_outputs: vec![("Direction".into(), PinDataType::Vec3)],
        },
        NodeTypeDescriptor {
            type_id: "world_position".into(),
            display_name: "World Position".into(),
            category: "Shading".into(),
            description: "Fragment world-space position".into(),
            default_inputs: vec![],
            default_outputs: vec![("Position".into(), PinDataType::Vec3)],
        },
        NodeTypeDescriptor {
            type_id: "world_normal".into(),
            display_name: "World Normal".into(),
            category: "Shading".into(),
            description: "Fragment world-space normal".into(),
            default_inputs: vec![],
            default_outputs: vec![("Normal".into(), PinDataType::Vec3)],
        },
        // --- Visual Scripting nodes ---
        NodeTypeDescriptor {
            type_id: "vs_branch".into(),
            display_name: "Branch".into(),
            category: "Flow Control".into(),
            description: "Conditional branch (if/else)".into(),
            default_inputs: vec![
                ("Exec".into(), PinDataType::Flow, None),
                ("Condition".into(), PinDataType::Bool, Some(Value::Bool(true))),
            ],
            default_outputs: vec![
                ("True".into(), PinDataType::Flow),
                ("False".into(), PinDataType::Flow),
            ],
        },
        NodeTypeDescriptor {
            type_id: "vs_for_loop".into(),
            display_name: "For Loop".into(),
            category: "Flow Control".into(),
            description: "Loop from start index to end index".into(),
            default_inputs: vec![
                ("Exec".into(), PinDataType::Flow, None),
                ("Start".into(), PinDataType::Int, Some(Value::Int(0))),
                ("End".into(), PinDataType::Int, Some(Value::Int(10))),
            ],
            default_outputs: vec![
                ("Body".into(), PinDataType::Flow),
                ("Index".into(), PinDataType::Int),
                ("Completed".into(), PinDataType::Flow),
            ],
        },
        NodeTypeDescriptor {
            type_id: "vs_variable_get".into(),
            display_name: "Get Variable".into(),
            category: "Variables".into(),
            description: "Read a named variable".into(),
            default_inputs: vec![],
            default_outputs: vec![("Value".into(), PinDataType::Any)],
        },
        NodeTypeDescriptor {
            type_id: "vs_variable_set".into(),
            display_name: "Set Variable".into(),
            category: "Variables".into(),
            description: "Write a named variable".into(),
            default_inputs: vec![
                ("Exec".into(), PinDataType::Flow, None),
                ("Value".into(), PinDataType::Any, None),
            ],
            default_outputs: vec![
                ("Exec".into(), PinDataType::Flow),
            ],
        },
        NodeTypeDescriptor {
            type_id: "vs_function_call".into(),
            display_name: "Function Call".into(),
            category: "Functions".into(),
            description: "Call a named function".into(),
            default_inputs: vec![
                ("Exec".into(), PinDataType::Flow, None),
            ],
            default_outputs: vec![
                ("Exec".into(), PinDataType::Flow),
                ("Return".into(), PinDataType::Any),
            ],
        },
        NodeTypeDescriptor {
            type_id: "vs_event_begin_play".into(),
            display_name: "Event: Begin Play".into(),
            category: "Events".into(),
            description: "Fires when gameplay begins".into(),
            default_inputs: vec![],
            default_outputs: vec![
                ("Exec".into(), PinDataType::Flow),
            ],
        },
        NodeTypeDescriptor {
            type_id: "vs_event_tick".into(),
            display_name: "Event: Tick".into(),
            category: "Events".into(),
            description: "Fires every frame".into(),
            default_inputs: vec![],
            default_outputs: vec![
                ("Exec".into(), PinDataType::Flow),
                ("Delta Time".into(), PinDataType::Float),
            ],
        },
        NodeTypeDescriptor {
            type_id: "vs_print".into(),
            display_name: "Print".into(),
            category: "Debug".into(),
            description: "Print a value to the log".into(),
            default_inputs: vec![
                ("Exec".into(), PinDataType::Flow, None),
                ("Message".into(), PinDataType::String, Some(Value::String("Hello".into()))),
            ],
            default_outputs: vec![
                ("Exec".into(), PinDataType::Flow),
            ],
        },
    ]
}

// ---------------------------------------------------------------------------
// Graph Node
// ---------------------------------------------------------------------------

/// A single node in the visual graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier.
    pub id: NodeId,
    /// Display title shown in the node header.
    pub title: String,
    /// Node type identifier (references the node library).
    pub type_id: String,
    /// Position in graph-space coordinates for rendering.
    pub position: [f32; 2],
    /// Input pins.
    pub inputs: Vec<NodePin>,
    /// Output pins.
    pub outputs: Vec<NodePin>,
    /// User-editable parameters stored as key-value pairs.
    pub parameters: HashMap<String, Value>,
    /// Whether this node is collapsed in the editor.
    pub collapsed: bool,
    /// Optional comment text attached to the node.
    pub comment: Option<String>,
}

impl GraphNode {
    /// Create a new node from a type descriptor.
    pub fn from_descriptor(desc: &NodeTypeDescriptor, position: [f32; 2]) -> Self {
        let inputs = desc
            .default_inputs
            .iter()
            .map(|(name, dt, default)| {
                let mut pin = NodePin::new(name.clone(), *dt);
                if let Some(val) = default {
                    pin.default_value = Some(val.clone());
                }
                pin
            })
            .collect();
        let outputs = desc
            .default_outputs
            .iter()
            .map(|(name, dt)| NodePin::new(name.clone(), *dt))
            .collect();
        Self {
            id: Uuid::new_v4(),
            title: desc.display_name.clone(),
            type_id: desc.type_id.clone(),
            position,
            inputs,
            outputs,
            parameters: HashMap::new(),
            collapsed: false,
            comment: None,
        }
    }

    /// Find an input pin by its ID.
    pub fn find_input(&self, pin_id: PinId) -> Option<&NodePin> {
        self.inputs.iter().find(|p| p.id == pin_id)
    }

    /// Find an output pin by its ID.
    pub fn find_output(&self, pin_id: PinId) -> Option<&NodePin> {
        self.outputs.iter().find(|p| p.id == pin_id)
    }

    /// Find an input pin by name.
    pub fn find_input_by_name(&self, name: &str) -> Option<&NodePin> {
        self.inputs.iter().find(|p| p.name == name)
    }

    /// Find an output pin by name.
    pub fn find_output_by_name(&self, name: &str) -> Option<&NodePin> {
        self.outputs.iter().find(|p| p.name == name)
    }

    /// Find a mutable input pin by ID.
    pub fn find_input_mut(&mut self, pin_id: PinId) -> Option<&mut NodePin> {
        self.inputs.iter_mut().find(|p| p.id == pin_id)
    }
}

// ---------------------------------------------------------------------------
// Node Group (visual grouping)
// ---------------------------------------------------------------------------

/// A visual grouping box in the graph editor for organizational purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGroup {
    /// Unique group identifier.
    pub id: Uuid,
    /// Group title.
    pub title: String,
    /// Color for the group background (RGBA).
    pub color: [f32; 4],
    /// Top-left position.
    pub position: [f32; 2],
    /// Size of the group box.
    pub size: [f32; 2],
    /// Node IDs contained in this group (for auto-sizing).
    pub contained_nodes: Vec<NodeId>,
}

impl NodeGroup {
    /// Create a new empty group.
    pub fn new(title: impl Into<String>, position: [f32; 2]) -> Self {
        Self {
            id: Uuid::new_v4(),
            title: title.into(),
            color: [0.3, 0.3, 0.5, 0.3],
            position,
            size: [300.0, 200.0],
            contained_nodes: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Node Graph
// ---------------------------------------------------------------------------

/// The core node graph data structure holding nodes, connections, and groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGraph {
    /// All nodes in the graph.
    pub nodes: Vec<GraphNode>,
    /// All connections between pins.
    pub connections: Vec<NodeConnection>,
    /// Visual grouping boxes.
    pub groups: Vec<NodeGroup>,
    /// Name of this graph.
    pub name: String,
    /// View pan offset (for editor camera).
    pub view_offset: [f32; 2],
    /// View zoom level.
    pub view_zoom: f32,
}

impl NodeGraph {
    /// Create a new empty graph.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            groups: Vec::new(),
            name: name.into(),
            view_offset: [0.0, 0.0],
            view_zoom: 1.0,
        }
    }

    // --- Node management ---

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: GraphNode) -> NodeId {
        let id = node.id;
        self.nodes.push(node);
        id
    }

    /// Remove a node and all its connections.
    pub fn remove_node(&mut self, node_id: NodeId) {
        self.connections.retain(|c| c.from_node != node_id && c.to_node != node_id);
        self.nodes.retain(|n| n.id != node_id);
        for group in &mut self.groups {
            group.contained_nodes.retain(|id| *id != node_id);
        }
    }

    /// Find a node by its ID.
    pub fn find_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == node_id)
    }

    /// Find a mutable node by its ID.
    pub fn find_node_mut(&mut self, node_id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.iter_mut().find(|n| n.id == node_id)
    }

    /// Find which node owns a given pin (input or output).
    pub fn find_node_for_pin(&self, pin_id: PinId) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| {
            n.inputs.iter().any(|p| p.id == pin_id)
                || n.outputs.iter().any(|p| p.id == pin_id)
        })
    }

    // --- Connection management ---

    /// Attempt to add a connection. Returns an error if the connection is invalid.
    pub fn connect(
        &mut self,
        from_node: NodeId,
        from_pin: PinId,
        to_node: NodeId,
        to_pin: PinId,
    ) -> Result<(), GraphError> {
        // Prevent self-connections.
        if from_node == to_node {
            return Err(GraphError::SelfConnection);
        }

        // Validate that both nodes exist.
        let source_node = self
            .find_node(from_node)
            .ok_or(GraphError::NodeNotFound(from_node))?;
        let target_node = self
            .find_node(to_node)
            .ok_or(GraphError::NodeNotFound(to_node))?;

        // Validate pin existence and direction.
        let source_pin = source_node
            .find_output(from_pin)
            .ok_or(GraphError::PinNotFound(from_pin))?;
        let target_pin = target_node
            .find_input(to_pin)
            .ok_or(GraphError::PinNotFound(to_pin))?;

        // Type compatibility check.
        if !PinDataType::is_compatible(source_pin.data_type, target_pin.data_type) {
            return Err(GraphError::TypeMismatch {
                src: source_pin.data_type,
                dst: target_pin.data_type,
            });
        }

        // Check if adding this connection would create a cycle.
        let proposed = NodeConnection {
            from_node,
            from_pin,
            to_node,
            to_pin,
        };
        if self.would_create_cycle(&proposed) {
            return Err(GraphError::CycleDetected);
        }

        // Remove any existing connection to the target input pin (inputs are single-connect).
        self.connections.retain(|c| c.to_pin != to_pin);

        // Update the input pin's connected_to field.
        if let Some(node) = self.find_node_mut(to_node) {
            if let Some(pin) = node.find_input_mut(to_pin) {
                pin.connected_to = Some(from_pin);
            }
        }

        self.connections.push(proposed);
        Ok(())
    }

    /// Disconnect a specific connection.
    pub fn disconnect(&mut self, from_pin: PinId, to_pin: PinId) {
        self.connections
            .retain(|c| !(c.from_pin == from_pin && c.to_pin == to_pin));
        // Clear the connected_to on the target pin.
        for node in &mut self.nodes {
            for pin in &mut node.inputs {
                if pin.id == to_pin && pin.connected_to == Some(from_pin) {
                    pin.connected_to = None;
                }
            }
        }
    }

    /// Disconnect all connections to/from a specific pin.
    pub fn disconnect_pin(&mut self, pin_id: PinId) {
        // Collect affected target pins before mutating.
        let affected_targets: Vec<PinId> = self
            .connections
            .iter()
            .filter(|c| c.from_pin == pin_id)
            .map(|c| c.to_pin)
            .collect();

        self.connections
            .retain(|c| c.from_pin != pin_id && c.to_pin != pin_id);

        // Clear connected_to on affected inputs.
        for node in &mut self.nodes {
            for pin in &mut node.inputs {
                if pin.id == pin_id || affected_targets.contains(&pin.id) {
                    pin.connected_to = None;
                }
            }
        }
    }

    /// Get all connections from a specific output pin.
    pub fn connections_from(&self, pin_id: PinId) -> Vec<&NodeConnection> {
        self.connections.iter().filter(|c| c.from_pin == pin_id).collect()
    }

    /// Get the connection to a specific input pin (if any).
    pub fn connection_to(&self, pin_id: PinId) -> Option<&NodeConnection> {
        self.connections.iter().find(|c| c.to_pin == pin_id)
    }

    // --- Cycle detection ---

    /// Returns true if adding the proposed connection would create a cycle.
    fn would_create_cycle(&self, proposed: &NodeConnection) -> bool {
        // DFS from the proposed target's outputs, checking if we can reach the source.
        let mut visited = HashSet::new();
        let mut stack = vec![proposed.to_node];

        // Also treat the proposed connection's from_node as reachable from to_node.
        while let Some(current) = stack.pop() {
            if current == proposed.from_node {
                return true;
            }
            if !visited.insert(current) {
                continue;
            }
            // Find all nodes that this node connects to (downstream).
            for conn in &self.connections {
                if conn.from_node == current {
                    stack.push(conn.to_node);
                }
            }
        }
        false
    }

    // --- Topological sort and evaluation ---

    /// Perform a topological sort of all nodes.
    /// Returns nodes in evaluation order (sources first, sinks last).
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, GraphError> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for node in &self.nodes {
            in_degree.entry(node.id).or_insert(0);
            adjacency.entry(node.id).or_insert_with(Vec::new);
        }

        for conn in &self.connections {
            *in_degree.entry(conn.to_node).or_insert(0) += 1;
            adjacency
                .entry(conn.from_node)
                .or_insert_with(Vec::new)
                .push(conn.to_node);
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut sorted = Vec::with_capacity(self.nodes.len());

        while let Some(node_id) = queue.pop_front() {
            sorted.push(node_id);
            if let Some(neighbors) = adjacency.get(&node_id) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(&neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if sorted.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected);
        }

        Ok(sorted)
    }

    /// Evaluate the graph and return the computed value for every output pin.
    /// Uses topological sort to process nodes in dependency order.
    pub fn evaluate(&self) -> Result<HashMap<PinId, Value>, GraphError> {
        let order = self.topological_sort()?;
        let mut pin_values: HashMap<PinId, Value> = HashMap::new();

        for node_id in &order {
            let node = self
                .find_node(*node_id)
                .ok_or(GraphError::NodeNotFound(*node_id))?;

            // Gather input values.
            let mut input_values: Vec<(&str, Value)> = Vec::new();
            for input_pin in &node.inputs {
                let value = if let Some(conn) = self.connection_to(input_pin.id) {
                    pin_values
                        .get(&conn.from_pin)
                        .cloned()
                        .unwrap_or(Value::None)
                } else {
                    input_pin.default_value.clone().unwrap_or(Value::None)
                };
                input_values.push((&input_pin.name, value));
            }

            // Evaluate based on node type.
            let outputs = evaluate_node(&node.type_id, &input_values, &node.parameters);

            // Store output values.
            for (i, output_pin) in node.outputs.iter().enumerate() {
                if i < outputs.len() {
                    pin_values.insert(output_pin.id, outputs[i].clone());
                }
            }
        }

        Ok(pin_values)
    }

    // --- Serialization ---

    /// Serialize the graph to a JSON string.
    pub fn to_json(&self) -> Result<String, GraphError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| GraphError::SerializationError(e.to_string()))
    }

    /// Deserialize a graph from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, GraphError> {
        serde_json::from_str(json)
            .map_err(|e| GraphError::SerializationError(e.to_string()))
    }

    // --- Utility ---

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Clear all nodes, connections, and groups.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.connections.clear();
        self.groups.clear();
    }

    /// Get the IDs of nodes that have no outgoing connections (leaf/output nodes).
    pub fn output_nodes(&self) -> Vec<NodeId> {
        let sources: HashSet<NodeId> = self.connections.iter().map(|c| c.from_node).collect();
        self.nodes
            .iter()
            .filter(|n| !sources.contains(&n.id))
            .map(|n| n.id)
            .collect()
    }

    /// Duplicate a selection of nodes, creating new copies with fresh IDs.
    pub fn duplicate_nodes(&mut self, node_ids: &[NodeId]) -> Vec<NodeId> {
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut pin_map: HashMap<PinId, PinId> = HashMap::new();
        let mut new_ids = Vec::new();

        // Clone nodes with new IDs.
        let clones: Vec<GraphNode> = node_ids
            .iter()
            .filter_map(|id| self.find_node(*id).cloned())
            .map(|mut node| {
                let old_id = node.id;
                node.id = Uuid::new_v4();
                node.position[0] += 50.0;
                node.position[1] += 50.0;
                id_map.insert(old_id, node.id);

                for pin in &mut node.inputs {
                    let old_pin = pin.id;
                    pin.id = Uuid::new_v4();
                    pin.connected_to = None;
                    pin_map.insert(old_pin, pin.id);
                }
                for pin in &mut node.outputs {
                    let old_pin = pin.id;
                    pin.id = Uuid::new_v4();
                    pin_map.insert(old_pin, pin.id);
                }

                new_ids.push(node.id);
                node
            })
            .collect();

        for clone in clones {
            self.nodes.push(clone);
        }

        // Duplicate internal connections (between duplicated nodes).
        let internal_conns: Vec<NodeConnection> = self
            .connections
            .iter()
            .filter(|c| id_map.contains_key(&c.from_node) && id_map.contains_key(&c.to_node))
            .filter_map(|c| {
                let new_from_node = *id_map.get(&c.from_node)?;
                let new_from_pin = *pin_map.get(&c.from_pin)?;
                let new_to_node = *id_map.get(&c.to_node)?;
                let new_to_pin = *pin_map.get(&c.to_pin)?;
                Some(NodeConnection {
                    from_node: new_from_node,
                    from_pin: new_from_pin,
                    to_node: new_to_node,
                    to_pin: new_to_pin,
                })
            })
            .collect();

        for conn in internal_conns {
            // Update the input pin's connected_to field.
            if let Some(node) = self.find_node_mut(conn.to_node) {
                if let Some(pin) = node.find_input_mut(conn.to_pin) {
                    pin.connected_to = Some(conn.from_pin);
                }
            }
            self.connections.push(conn);
        }

        new_ids
    }
}

// ---------------------------------------------------------------------------
// Node Evaluation
// ---------------------------------------------------------------------------

/// Evaluate a single node given its type and input values.
/// Returns a vector of output values, one per output pin.
fn evaluate_node(
    type_id: &str,
    inputs: &[(&str, Value)],
    _parameters: &HashMap<String, Value>,
) -> Vec<Value> {
    let get_input = |name: &str| -> Value {
        inputs
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v.clone())
            .unwrap_or(Value::None)
    };

    match type_id {
        "math_add" => {
            let a = get_input("A").as_float();
            let b = get_input("B").as_float();
            vec![Value::Float(a + b)]
        }
        "math_subtract" => {
            let a = get_input("A").as_float();
            let b = get_input("B").as_float();
            vec![Value::Float(a - b)]
        }
        "math_multiply" => {
            let a = get_input("A").as_float();
            let b = get_input("B").as_float();
            vec![Value::Float(a * b)]
        }
        "math_divide" => {
            let a = get_input("A").as_float();
            let b = get_input("B").as_float();
            let result = if b.abs() < 1e-10 { 0.0 } else { a / b };
            vec![Value::Float(result)]
        }
        "math_lerp" => {
            let a = get_input("A").as_float();
            let b = get_input("B").as_float();
            let t = get_input("T").as_float();
            vec![Value::Float(a + (b - a) * t)]
        }
        "math_clamp" => {
            let v = get_input("Value").as_float();
            let min = get_input("Min").as_float();
            let max = get_input("Max").as_float();
            vec![Value::Float(v.clamp(min, max))]
        }
        "math_saturate" => {
            let v = get_input("Value").as_float();
            vec![Value::Float(v.clamp(0.0, 1.0))]
        }
        "math_power" => {
            let base = get_input("Base").as_float();
            let exp = get_input("Exponent").as_float();
            vec![Value::Float(base.powf(exp))]
        }
        "math_abs" => {
            let v = get_input("Value").as_float();
            vec![Value::Float(v.abs())]
        }
        "math_one_minus" => {
            let v = get_input("Value").as_float();
            vec![Value::Float(1.0 - v)]
        }
        "math_sin" => {
            let v = get_input("Value").as_float();
            vec![Value::Float(v.sin())]
        }
        "math_cos" => {
            let v = get_input("Value").as_float();
            vec![Value::Float(v.cos())]
        }
        "math_floor" => {
            let v = get_input("Value").as_float();
            vec![Value::Float(v.floor())]
        }
        "math_fract" => {
            let v = get_input("Value").as_float();
            vec![Value::Float(v.fract())]
        }
        "math_smoothstep" => {
            let e0 = get_input("Edge0").as_float();
            let e1 = get_input("Edge1").as_float();
            let x = get_input("X").as_float();
            let t = ((x - e0) / (e1 - e0).max(1e-8)).clamp(0.0, 1.0);
            vec![Value::Float(t * t * (3.0 - 2.0 * t))]
        }
        "vec_combine" => {
            let x = get_input("R/X").as_float();
            let y = get_input("G/Y").as_float();
            let z = get_input("B/Z").as_float();
            let w = get_input("A/W").as_float();
            vec![
                Value::Vec4([x, y, z, w]),
                Value::Vec3([x, y, z]),
                Value::Vec2([x, y]),
            ]
        }
        "vec_split" => {
            let v = get_input("Vector").as_vec4();
            vec![
                Value::Float(v[0]),
                Value::Float(v[1]),
                Value::Float(v[2]),
                Value::Float(v[3]),
            ]
        }
        "vec_normalize" => {
            let v = get_input("Vector").as_vec3();
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let result = if len < 1e-8 {
                [0.0, 0.0, 0.0]
            } else {
                [v[0] / len, v[1] / len, v[2] / len]
            };
            vec![Value::Vec3(result)]
        }
        "vec_dot" => {
            let a = get_input("A").as_vec3();
            let b = get_input("B").as_vec3();
            vec![Value::Float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])]
        }
        "vec_cross" => {
            let a = get_input("A").as_vec3();
            let b = get_input("B").as_vec3();
            vec![Value::Vec3([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])]
        }
        "fresnel" => {
            let power = get_input("Power").as_float();
            // Approximation: without actual view vector, return parameterized Fresnel.
            let base = 0.04_f32;
            vec![Value::Float(base + (1.0 - base) * (1.0_f32).powf(power))]
        }
        "float_constant" | "vec2_constant" | "vec3_constant" | "color_constant" => {
            // Constants pass through from parameters/defaults.
            if let Some((_, v)) = inputs.first() {
                vec![v.clone()]
            } else {
                vec![Value::Float(0.0)]
            }
        }
        "texture_sample" => {
            // Placeholder: return white for unresolved textures.
            vec![
                Value::Color([1.0, 1.0, 1.0, 1.0]),
                Value::Float(1.0),
                Value::Float(1.0),
                Value::Float(1.0),
                Value::Float(1.0),
            ]
        }
        "uv_coords" => {
            vec![Value::Vec2([0.0, 0.0])]
        }
        "uv_scale_offset" => {
            let uv = get_input("UV");
            let uv = match &uv {
                Value::Vec2(v) => *v,
                _ => [0.0, 0.0],
            };
            let scale = match get_input("Scale") {
                Value::Vec2(v) => v,
                _ => [1.0, 1.0],
            };
            let offset = match get_input("Offset") {
                Value::Vec2(v) => v,
                _ => [0.0, 0.0],
            };
            vec![Value::Vec2([
                uv[0] * scale[0] + offset[0],
                uv[1] * scale[1] + offset[1],
            ])]
        }
        _ => {
            // Unknown node type: return None for all outputs.
            vec![Value::None]
        }
    }
}

// ---------------------------------------------------------------------------
// Graph Error
// ---------------------------------------------------------------------------

/// Errors that can occur during graph operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(NodeId),
    #[error("Pin not found: {0}")]
    PinNotFound(PinId),
    #[error("Cannot connect node to itself")]
    SelfConnection,
    #[error("Type mismatch: {src:?} cannot connect to {dst:?}")]
    TypeMismatch {
        src: PinDataType,
        dst: PinDataType,
    },
    #[error("Connection would create a cycle")]
    CycleDetected,
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> NodeGraph {
        let library = built_in_node_library();
        let mut graph = NodeGraph::new("Test Graph");

        let add_desc = library.iter().find(|d| d.type_id == "math_add").unwrap();
        let mul_desc = library.iter().find(|d| d.type_id == "math_multiply").unwrap();
        let const_desc = library.iter().find(|d| d.type_id == "float_constant").unwrap();

        let const1 = GraphNode::from_descriptor(const_desc, [0.0, 0.0]);
        let const2 = GraphNode::from_descriptor(const_desc, [0.0, 100.0]);
        let add_node = GraphNode::from_descriptor(add_desc, [200.0, 50.0]);
        let mul_node = GraphNode::from_descriptor(mul_desc, [400.0, 50.0]);

        graph.add_node(const1);
        graph.add_node(const2);
        graph.add_node(add_node);
        graph.add_node(mul_node);

        graph
    }

    #[test]
    fn create_graph() {
        let graph = NodeGraph::new("Material");
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.connection_count(), 0);
    }

    #[test]
    fn add_and_remove_nodes() {
        let mut graph = create_test_graph();
        assert_eq!(graph.node_count(), 4);

        let id = graph.nodes[0].id;
        graph.remove_node(id);
        assert_eq!(graph.node_count(), 3);
        assert!(graph.find_node(id).is_none());
    }

    #[test]
    fn connect_compatible_pins() {
        let mut graph = create_test_graph();
        let const_id = graph.nodes[0].id;
        let const_out = graph.nodes[0].outputs[0].id;
        let add_id = graph.nodes[2].id;
        let add_in_a = graph.nodes[2].inputs[0].id;

        let result = graph.connect(const_id, const_out, add_id, add_in_a);
        assert!(result.is_ok());
        assert_eq!(graph.connection_count(), 1);
    }

    #[test]
    fn reject_self_connection() {
        let mut graph = create_test_graph();
        let id = graph.nodes[2].id;
        let out = graph.nodes[2].outputs[0].id;
        let inp = graph.nodes[2].inputs[0].id;

        let result = graph.connect(id, out, id, inp);
        assert!(matches!(result, Err(GraphError::SelfConnection)));
    }

    #[test]
    fn reject_incompatible_types() {
        let library = built_in_node_library();
        let mut graph = NodeGraph::new("Test");

        let tex_desc = library.iter().find(|d| d.type_id == "texture_sample").unwrap();
        let add_desc = library.iter().find(|d| d.type_id == "math_add").unwrap();

        let tex_node = GraphNode::from_descriptor(tex_desc, [0.0, 0.0]);
        let add_node = GraphNode::from_descriptor(add_desc, [200.0, 0.0]);

        let tex_id = tex_node.id;
        let add_id = add_node.id;

        graph.add_node(tex_node);
        graph.add_node(add_node);

        // Try connecting Texture2D input to Float input.
        let tex_input = graph.find_node(tex_id).unwrap().inputs[0].id; // Texture2D input
        let add_output = graph.find_node(add_id).unwrap().outputs[0].id; // Float output

        // Float -> Texture2D should fail.
        let result = graph.connect(add_id, add_output, tex_id, tex_input);
        assert!(matches!(result, Err(GraphError::TypeMismatch { .. })));
    }

    #[test]
    fn detect_cycle() {
        let library = built_in_node_library();
        let mut graph = NodeGraph::new("Test");

        let add_desc = library.iter().find(|d| d.type_id == "math_add").unwrap();

        let node_a = GraphNode::from_descriptor(add_desc, [0.0, 0.0]);
        let node_b = GraphNode::from_descriptor(add_desc, [200.0, 0.0]);

        let a_id = node_a.id;
        let a_out = node_a.outputs[0].id;
        let a_in = node_a.inputs[0].id;
        let b_id = node_b.id;
        let b_out = node_b.outputs[0].id;
        let b_in = node_b.inputs[0].id;

        graph.add_node(node_a);
        graph.add_node(node_b);

        // A -> B (ok)
        assert!(graph.connect(a_id, a_out, b_id, b_in).is_ok());

        // B -> A would create a cycle.
        let result = graph.connect(b_id, b_out, a_id, a_in);
        assert!(matches!(result, Err(GraphError::CycleDetected)));
    }

    #[test]
    fn topological_sort_valid() {
        let mut graph = create_test_graph();
        let const1_id = graph.nodes[0].id;
        let const1_out = graph.nodes[0].outputs[0].id;
        let const2_id = graph.nodes[1].id;
        let const2_out = graph.nodes[1].outputs[0].id;
        let add_id = graph.nodes[2].id;
        let add_in_a = graph.nodes[2].inputs[0].id;
        let add_in_b = graph.nodes[2].inputs[1].id;

        graph.connect(const1_id, const1_out, add_id, add_in_a).unwrap();
        graph.connect(const2_id, const2_out, add_id, add_in_b).unwrap();

        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted.len(), 4);

        // The add node must come after both constants.
        let const1_pos = sorted.iter().position(|&id| id == const1_id).unwrap();
        let const2_pos = sorted.iter().position(|&id| id == const2_id).unwrap();
        let add_pos = sorted.iter().position(|&id| id == add_id).unwrap();

        assert!(const1_pos < add_pos);
        assert!(const2_pos < add_pos);
    }

    #[test]
    fn evaluate_math_chain() {
        let library = built_in_node_library();
        let mut graph = NodeGraph::new("Eval Test");

        let add_desc = library.iter().find(|d| d.type_id == "math_add").unwrap();
        let mul_desc = library.iter().find(|d| d.type_id == "math_multiply").unwrap();

        let mut add_node = GraphNode::from_descriptor(add_desc, [0.0, 0.0]);
        add_node.inputs[0].default_value = Some(Value::Float(3.0));
        add_node.inputs[1].default_value = Some(Value::Float(4.0));
        let add_out = add_node.outputs[0].id;
        let add_id = add_node.id;

        let mut mul_node = GraphNode::from_descriptor(mul_desc, [200.0, 0.0]);
        mul_node.inputs[1].default_value = Some(Value::Float(2.0));
        let mul_in_a = mul_node.inputs[0].id;
        let mul_out = mul_node.outputs[0].id;
        let mul_id = mul_node.id;

        graph.add_node(add_node);
        graph.add_node(mul_node);

        // Connect add output to multiply input A.
        graph.connect(add_id, add_out, mul_id, mul_in_a).unwrap();

        let values = graph.evaluate().unwrap();

        // add: 3 + 4 = 7, multiply: 7 * 2 = 14
        let add_result = values.get(&add_out).unwrap().as_float();
        assert!((add_result - 7.0).abs() < 1e-5);

        let mul_result = values.get(&mul_out).unwrap().as_float();
        assert!((mul_result - 14.0).abs() < 1e-5);
    }

    #[test]
    fn evaluate_lerp() {
        let library = built_in_node_library();
        let mut graph = NodeGraph::new("Lerp Test");

        let lerp_desc = library.iter().find(|d| d.type_id == "math_lerp").unwrap();
        let mut lerp = GraphNode::from_descriptor(lerp_desc, [0.0, 0.0]);
        lerp.inputs[0].default_value = Some(Value::Float(0.0));
        lerp.inputs[1].default_value = Some(Value::Float(10.0));
        lerp.inputs[2].default_value = Some(Value::Float(0.5));
        let out_id = lerp.outputs[0].id;
        graph.add_node(lerp);

        let values = graph.evaluate().unwrap();
        let result = values.get(&out_id).unwrap().as_float();
        assert!((result - 5.0).abs() < 1e-5);
    }

    #[test]
    fn evaluate_vec_operations() {
        let library = built_in_node_library();
        let mut graph = NodeGraph::new("Vec Test");

        let dot_desc = library.iter().find(|d| d.type_id == "vec_dot").unwrap();
        let mut dot_node = GraphNode::from_descriptor(dot_desc, [0.0, 0.0]);
        dot_node.inputs[0].default_value = Some(Value::Vec3([1.0, 0.0, 0.0]));
        dot_node.inputs[1].default_value = Some(Value::Vec3([0.0, 1.0, 0.0]));
        let out_id = dot_node.outputs[0].id;
        graph.add_node(dot_node);

        let values = graph.evaluate().unwrap();
        let result = values.get(&out_id).unwrap().as_float();
        assert!((result - 0.0).abs() < 1e-5); // Perpendicular vectors -> dot = 0
    }

    #[test]
    fn serialize_deserialize_graph() {
        let graph = create_test_graph();
        let json = graph.to_json().unwrap();
        let restored = NodeGraph::from_json(&json).unwrap();
        assert_eq!(restored.node_count(), graph.node_count());
        assert_eq!(restored.name, "Test Graph");
    }

    #[test]
    fn pin_type_compatibility() {
        assert!(PinDataType::is_compatible(PinDataType::Float, PinDataType::Float));
        assert!(PinDataType::is_compatible(PinDataType::Float, PinDataType::Vec3));
        assert!(PinDataType::is_compatible(PinDataType::Color, PinDataType::Vec4));
        assert!(PinDataType::is_compatible(PinDataType::Any, PinDataType::Float));
        assert!(PinDataType::is_compatible(PinDataType::Int, PinDataType::Float));
        assert!(!PinDataType::is_compatible(PinDataType::Texture2D, PinDataType::Float));
        assert!(!PinDataType::is_compatible(PinDataType::Bool, PinDataType::Vec3));
    }

    #[test]
    fn built_in_library_has_enough_nodes() {
        let lib = built_in_node_library();
        assert!(lib.len() >= 30, "Library has {} nodes, expected >= 30", lib.len());
    }

    #[test]
    fn disconnect_pin() {
        let mut graph = create_test_graph();
        let c_id = graph.nodes[0].id;
        let c_out = graph.nodes[0].outputs[0].id;
        let a_id = graph.nodes[2].id;
        let a_in = graph.nodes[2].inputs[0].id;

        graph.connect(c_id, c_out, a_id, a_in).unwrap();
        assert_eq!(graph.connection_count(), 1);

        graph.disconnect_pin(c_out);
        assert_eq!(graph.connection_count(), 0);
    }

    #[test]
    fn duplicate_nodes() {
        let mut graph = create_test_graph();
        let original_ids: Vec<NodeId> = graph.nodes.iter().map(|n| n.id).collect();

        let new_ids = graph.duplicate_nodes(&original_ids[..2]);
        assert_eq!(new_ids.len(), 2);
        assert_eq!(graph.node_count(), 6);

        // Verify new nodes have different IDs.
        for new_id in &new_ids {
            assert!(!original_ids.contains(new_id));
        }
    }

    #[test]
    fn node_group_creation() {
        let group = NodeGroup::new("Shading", [100.0, 100.0]);
        assert_eq!(group.title, "Shading");
        assert!(group.contained_nodes.is_empty());
    }

    #[test]
    fn value_conversions() {
        let f = Value::Float(3.0);
        assert!((f.as_float() - 3.0).abs() < 1e-5);
        assert_eq!(f.as_vec3(), [3.0, 3.0, 3.0]);

        let v3 = Value::Vec3([1.0, 2.0, 3.0]);
        assert_eq!(v3.as_vec4(), [1.0, 2.0, 3.0, 1.0]);

        let b = Value::Bool(true);
        assert!(b.as_bool());
        assert!((b.as_float() - 1.0).abs() < 1e-5);
    }
}
