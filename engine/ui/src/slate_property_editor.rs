//! Slate Property Editor (Auto-Generated Details Panel)
//!
//! Provides a comprehensive auto-generated property editor system for the
//! Genovo engine editor, similar to Unreal Engine's Details panel. Given a
//! reflected type, the system automatically generates the appropriate edit UI
//! for each property: text inputs for strings, sliders for ranged numbers,
//! checkboxes for booleans, color pickers for colors, dropdowns for enums, etc.
//!
//! # Architecture
//!
//! ```text
//!  PropertyEditor ──> PropertyRow ──> PropertyWidget
//!       │                  │                │
//!  PropertyCategory   PropertyFilter    SpecificWidgets:
//!       │                  │            - BoolEditor
//!  PropertyBinding    SearchBox         - StringEditor
//!       │                               - FloatEditor
//!  UndoRedoStack                        - IntEditor
//!                                       - Vec3Editor
//!                                       - ColorEditor
//!                                       - EnumEditor
//!                                       - AssetRefEditor
//!                                       - ArrayEditor
//!                                       - StructEditor
//! ```
//!
//! # Reflection Integration
//!
//! The property editor works with any type that implements `Reflected`, which
//! provides runtime type information including field names, types, ranges, and
//! display metadata.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec2;

use crate::core::UIId;
use crate::render_commands::Color;

// ---------------------------------------------------------------------------
// PropertyId
// ---------------------------------------------------------------------------

/// Unique identifier for a property in the editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PropId(pub u64);

static NEXT_PROP_ID: AtomicU64 = AtomicU64::new(1);

fn next_prop_id() -> PropId {
    PropId(NEXT_PROP_ID.fetch_add(1, Ordering::Relaxed))
}

impl fmt::Display for PropId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Prop({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// PropertyType
// ---------------------------------------------------------------------------

/// The data type of a property, used to select the appropriate editor widget.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyType {
    /// Boolean (checkbox).
    Bool,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit unsigned integer.
    UInt32,
    /// 64-bit unsigned integer.
    UInt64,
    /// 32-bit floating point.
    Float32,
    /// 64-bit floating point.
    Float64,
    /// String.
    String,
    /// 2D vector (two floats).
    Vec2,
    /// 3D vector (three floats).
    Vec3,
    /// 4D vector (four floats).
    Vec4,
    /// Quaternion rotation.
    Quaternion,
    /// Color (RGBA).
    Color,
    /// Enumeration with named variants.
    Enum(Vec<String>),
    /// Flags / bit field with named bits.
    Flags(Vec<String>),
    /// Fixed-size array of a single type.
    Array(Box<PropertyType>),
    /// Nested struct (contains sub-properties).
    Struct(String),
    /// Asset reference (path + type).
    AssetRef(String),
    /// Angle in degrees.
    Angle,
    /// Transform (position + rotation + scale).
    Transform,
    /// A raw byte buffer (displayed as hex).
    Bytes,
    /// Custom type with a type name.
    Custom(String),
}

impl PropertyType {
    /// Returns true if this type is numeric (int or float).
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::Int32
                | Self::Int64
                | Self::UInt32
                | Self::UInt64
                | Self::Float32
                | Self::Float64
                | Self::Angle
        )
    }

    /// Returns true if this type is a vector type.
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vec2 | Self::Vec3 | Self::Vec4 | Self::Quaternion)
    }

    /// Returns true if this type can be edited inline (single row).
    pub fn is_inline(&self) -> bool {
        matches!(
            self,
            Self::Bool
                | Self::Int32
                | Self::Int64
                | Self::UInt32
                | Self::UInt64
                | Self::Float32
                | Self::Float64
                | Self::String
                | Self::Angle
                | Self::Enum(_)
        )
    }

    /// Returns the default display name for this type.
    pub fn display_name(&self) -> &str {
        match self {
            Self::Bool => "Bool",
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::UInt32 => "UInt32",
            Self::UInt64 => "UInt64",
            Self::Float32 => "Float",
            Self::Float64 => "Double",
            Self::String => "String",
            Self::Vec2 => "Vector2",
            Self::Vec3 => "Vector3",
            Self::Vec4 => "Vector4",
            Self::Quaternion => "Quaternion",
            Self::Color => "Color",
            Self::Enum(_) => "Enum",
            Self::Flags(_) => "Flags",
            Self::Array(_) => "Array",
            Self::Struct(name) => name,
            Self::AssetRef(kind) => kind,
            Self::Angle => "Angle",
            Self::Transform => "Transform",
            Self::Bytes => "Bytes",
            Self::Custom(name) => name,
        }
    }
}

impl fmt::Display for PropertyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ---------------------------------------------------------------------------
// PropertyValue
// ---------------------------------------------------------------------------

/// A runtime value for a property, used for reading and writing.
#[derive(Debug, Clone)]
pub enum PropertyValue {
    Bool(bool),
    Int32(i32),
    Int64(i64),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    String(String),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
    Enum(usize, String),
    Flags(u64),
    Array(Vec<PropertyValue>),
    Struct(HashMap<String, PropertyValue>),
    AssetRef(String),
    Angle(f32),
    Bytes(Vec<u8>),
    None,
}

impl PropertyValue {
    /// Returns a human-readable display string for the value.
    pub fn display_string(&self) -> String {
        match self {
            Self::Bool(v) => v.to_string(),
            Self::Int32(v) => v.to_string(),
            Self::Int64(v) => v.to_string(),
            Self::UInt32(v) => v.to_string(),
            Self::UInt64(v) => v.to_string(),
            Self::Float32(v) => format!("{:.3}", v),
            Self::Float64(v) => format!("{:.6}", v),
            Self::String(v) => v.clone(),
            Self::Vec2(v) => format!("({:.2}, {:.2})", v[0], v[1]),
            Self::Vec3(v) => format!("({:.2}, {:.2}, {:.2})", v[0], v[1], v[2]),
            Self::Vec4(v) => format!("({:.2}, {:.2}, {:.2}, {:.2})", v[0], v[1], v[2], v[3]),
            Self::Color(v) => format!(
                "#{:02X}{:02X}{:02X}{:02X}",
                (v[0] * 255.0) as u8,
                (v[1] * 255.0) as u8,
                (v[2] * 255.0) as u8,
                (v[3] * 255.0) as u8,
            ),
            Self::Enum(_, name) => name.clone(),
            Self::Flags(v) => format!("0x{:X}", v),
            Self::Array(v) => format!("[{} elements]", v.len()),
            Self::Struct(_) => "[struct]".to_string(),
            Self::AssetRef(v) => v.clone(),
            Self::Angle(v) => format!("{:.1} deg", v),
            Self::Bytes(v) => format!("[{} bytes]", v.len()),
            Self::None => "None".to_string(),
        }
    }

    /// Returns true if this value equals another (for change detection).
    pub fn equals(&self, other: &PropertyValue) -> bool {
        match (self, other) {
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int32(a), Self::Int32(b)) => a == b,
            (Self::Int64(a), Self::Int64(b)) => a == b,
            (Self::UInt32(a), Self::UInt32(b)) => a == b,
            (Self::UInt64(a), Self::UInt64(b)) => a == b,
            (Self::Float32(a), Self::Float32(b)) => (a - b).abs() < 1e-6,
            (Self::Float64(a), Self::Float64(b)) => (a - b).abs() < 1e-10,
            (Self::String(a), Self::String(b)) => a == b,
            (Self::Vec2(a), Self::Vec2(b)) => a == b,
            (Self::Vec3(a), Self::Vec3(b)) => a == b,
            (Self::Vec4(a), Self::Vec4(b)) => a == b,
            (Self::Color(a), Self::Color(b)) => a == b,
            (Self::Enum(a, _), Self::Enum(b, _)) => a == b,
            (Self::Flags(a), Self::Flags(b)) => a == b,
            (Self::AssetRef(a), Self::AssetRef(b)) => a == b,
            (Self::Angle(a), Self::Angle(b)) => (a - b).abs() < 0.01,
            _ => false,
        }
    }

    /// Converts to a float value if possible.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Int32(v) => Some(*v as f32),
            Self::Int64(v) => Some(*v as f32),
            Self::UInt32(v) => Some(*v as f32),
            Self::UInt64(v) => Some(*v as f32),
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            Self::Angle(v) => Some(*v),
            _ => None,
        }
    }

    /// Converts to a boolean value if possible.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::Int32(v) => Some(*v != 0),
            Self::UInt32(v) => Some(*v != 0),
            _ => None,
        }
    }

    /// Converts to a string if possible.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            Self::AssetRef(s) => Some(s),
            _ => None,
        }
    }
}

impl Default for PropertyValue {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// PropertyMetadata
// ---------------------------------------------------------------------------

/// Metadata about a property that controls how it is displayed and edited.
#[derive(Debug, Clone)]
pub struct PropertyMetadata {
    /// The property's display name.
    pub display_name: String,
    /// The property's internal name (field name).
    pub field_name: String,
    /// Optional tooltip/description.
    pub tooltip: Option<String>,
    /// Category for grouping.
    pub category: String,
    /// Whether this is an "advanced" property (hidden by default).
    pub advanced: bool,
    /// Whether this property is read-only.
    pub read_only: bool,
    /// Whether this property is hidden.
    pub hidden: bool,
    /// Numeric range: minimum value.
    pub range_min: Option<f64>,
    /// Numeric range: maximum value.
    pub range_max: Option<f64>,
    /// Numeric step size for drag/slider.
    pub step: Option<f64>,
    /// Number of decimal places to display for floats.
    pub decimal_places: u32,
    /// Whether to use a slider instead of a drag value.
    pub use_slider: bool,
    /// Whether to clamp the value to the range.
    pub clamp_to_range: bool,
    /// Units label (e.g., "cm", "deg", "kg").
    pub units: Option<String>,
    /// Property order within its category (lower = first).
    pub sort_order: i32,
    /// Whether to show inline edit (vs popup editor).
    pub inline_edit: bool,
    /// Custom editor widget name (overrides auto-detection).
    pub custom_editor: Option<String>,
    /// Whether this property can be animated.
    pub animatable: bool,
    /// Whether changing this property is undoable.
    pub undoable: bool,
    /// Whether multiple objects can be edited at once for this property.
    pub multi_edit: bool,
    /// Display condition: only show when another property has a specific value.
    pub show_condition: Option<ShowCondition>,
    /// Whether the value differs from the default.
    pub differs_from_default: bool,
    /// The default value for reset-to-default.
    pub default_value: Option<PropertyValue>,
}

/// A condition for showing/hiding a property based on another property's value.
#[derive(Debug, Clone)]
pub struct ShowCondition {
    /// The name of the property to check.
    pub property_name: String,
    /// The expected value for the condition to be true.
    pub expected_value: PropertyValue,
    /// Whether to invert the condition.
    pub inverted: bool,
}

impl ShowCondition {
    /// Creates a new show condition.
    pub fn new(property_name: &str, expected_value: PropertyValue) -> Self {
        Self {
            property_name: property_name.to_string(),
            expected_value,
            inverted: false,
        }
    }

    /// Evaluates the condition against a property value.
    pub fn evaluate(&self, value: &PropertyValue) -> bool {
        let result = value.equals(&self.expected_value);
        if self.inverted {
            !result
        } else {
            result
        }
    }
}

impl PropertyMetadata {
    /// Creates new metadata with the given names.
    pub fn new(field_name: &str, display_name: &str) -> Self {
        Self {
            display_name: display_name.to_string(),
            field_name: field_name.to_string(),
            tooltip: None,
            category: "General".to_string(),
            advanced: false,
            read_only: false,
            hidden: false,
            range_min: None,
            range_max: None,
            step: None,
            decimal_places: 3,
            use_slider: false,
            clamp_to_range: false,
            units: None,
            sort_order: 0,
            inline_edit: true,
            custom_editor: None,
            animatable: false,
            undoable: true,
            multi_edit: true,
            show_condition: None,
            differs_from_default: false,
            default_value: None,
        }
    }

    /// Sets the category.
    pub fn with_category(mut self, category: &str) -> Self {
        self.category = category.to_string();
        self
    }

    /// Sets the range.
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.range_min = Some(min);
        self.range_max = Some(max);
        self.clamp_to_range = true;
        self
    }

    /// Sets the step size.
    pub fn with_step(mut self, step: f64) -> Self {
        self.step = Some(step);
        self
    }

    /// Enables slider mode.
    pub fn with_slider(mut self) -> Self {
        self.use_slider = true;
        self
    }

    /// Sets units.
    pub fn with_units(mut self, units: &str) -> Self {
        self.units = Some(units.to_string());
        self
    }

    /// Sets the tooltip.
    pub fn with_tooltip(mut self, tooltip: &str) -> Self {
        self.tooltip = Some(tooltip.to_string());
        self
    }

    /// Marks as read-only.
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Marks as advanced.
    pub fn advanced(mut self) -> Self {
        self.advanced = true;
        self
    }

    /// Sets the sort order.
    pub fn with_order(mut self, order: i32) -> Self {
        self.sort_order = order;
        self
    }

    /// Sets the default value.
    pub fn with_default(mut self, value: PropertyValue) -> Self {
        self.default_value = Some(value);
        self
    }

    /// Returns the effective step size (or a reasonable default).
    pub fn effective_step(&self) -> f64 {
        self.step.unwrap_or(0.01)
    }

    /// Clamps a float value to the configured range.
    pub fn clamp_value(&self, value: f64) -> f64 {
        if !self.clamp_to_range {
            return value;
        }
        let mut v = value;
        if let Some(min) = self.range_min {
            v = v.max(min);
        }
        if let Some(max) = self.range_max {
            v = v.min(max);
        }
        v
    }
}

// ---------------------------------------------------------------------------
// PropertyRow
// ---------------------------------------------------------------------------

/// A single row in the property editor: label on the left, widget on the right.
///
/// The label takes up ~40% of the width and the edit widget takes up ~60%.
/// A reset-to-default button appears on hover when the value differs from
/// the default.
#[derive(Debug, Clone)]
pub struct PropertyRow {
    /// Unique identifier for this row.
    pub id: PropId,
    /// The property metadata.
    pub metadata: PropertyMetadata,
    /// The property type.
    pub property_type: PropertyType,
    /// Current value.
    pub value: PropertyValue,
    /// Previous value (for change detection).
    pub previous_value: PropertyValue,
    /// Whether the row is expanded (for struct/array types).
    pub expanded: bool,
    /// Whether the row is visible (not hidden by filter or condition).
    pub visible: bool,
    /// Whether the row is selected.
    pub selected: bool,
    /// Whether the row is hovered.
    pub hovered: bool,
    /// Whether the value has changed since last commit.
    pub dirty: bool,
    /// Whether the row is being actively edited.
    pub editing: bool,
    /// Depth in the property tree (0 = top-level, 1 = nested, etc.).
    pub depth: u32,
    /// Y position of this row (computed by layout).
    pub y_position: f32,
    /// Height of this row.
    pub height: f32,
    /// Label width fraction (0.0 to 1.0).
    pub label_width_fraction: f32,
    /// Child property rows (for structs/arrays).
    pub children: Vec<PropId>,
    /// Parent property row ID.
    pub parent: Option<PropId>,
    /// Index in array (if this is an array element).
    pub array_index: Option<usize>,
    /// Whether the reset-to-default button is visible (on hover).
    pub show_reset: bool,
    /// Animation: expand/collapse progress (0 to 1).
    pub expand_animation: f32,
    /// The text currently being edited (for text input fields).
    pub edit_text: String,
    /// Whether the edit text field has focus.
    pub text_focused: bool,
    /// Cursor position in the edit text.
    pub text_cursor: usize,
    /// Whether this row represents multiple different values (multi-edit).
    pub mixed_value: bool,
    /// Object indices that this row represents (for multi-edit).
    pub object_indices: Vec<usize>,
}

impl PropertyRow {
    /// Creates a new property row.
    pub fn new(
        metadata: PropertyMetadata,
        property_type: PropertyType,
        value: PropertyValue,
    ) -> Self {
        let edit_text = value.display_string();
        Self {
            id: next_prop_id(),
            metadata,
            property_type,
            previous_value: value.clone(),
            value,
            expanded: false,
            visible: true,
            selected: false,
            hovered: false,
            dirty: false,
            editing: false,
            depth: 0,
            y_position: 0.0,
            height: 24.0,
            label_width_fraction: 0.4,
            children: Vec::new(),
            parent: None,
            array_index: None,
            show_reset: false,
            expand_animation: 0.0,
            edit_text,
            text_focused: false,
            text_cursor: 0,
            mixed_value: false,
            object_indices: vec![0],
        }
    }

    /// Returns the display label for this row.
    pub fn label(&self) -> &str {
        if let Some(idx) = self.array_index {
            // Handled separately for array elements.
            &self.metadata.display_name
        } else {
            &self.metadata.display_name
        }
    }

    /// Returns whether this row differs from the default value.
    pub fn differs_from_default(&self) -> bool {
        if let Some(ref default) = self.metadata.default_value {
            !self.value.equals(default)
        } else {
            false
        }
    }

    /// Resets the value to the default.
    pub fn reset_to_default(&mut self) {
        if let Some(ref default) = self.metadata.default_value {
            self.value = default.clone();
            self.dirty = true;
            self.edit_text = self.value.display_string();
        }
    }

    /// Sets the value and marks as dirty if changed.
    pub fn set_value(&mut self, new_value: PropertyValue) {
        if !self.value.equals(&new_value) {
            self.previous_value = self.value.clone();
            self.value = new_value;
            self.dirty = true;
            self.edit_text = self.value.display_string();
        }
    }

    /// Commits the current edit (saves the value).
    pub fn commit(&mut self) {
        self.dirty = false;
        self.editing = false;
    }

    /// Cancels the current edit (reverts to previous value).
    pub fn cancel_edit(&mut self) {
        self.value = self.previous_value.clone();
        self.dirty = false;
        self.editing = false;
        self.edit_text = self.value.display_string();
    }

    /// Begins editing this row.
    pub fn begin_edit(&mut self) {
        self.editing = true;
        self.edit_text = self.value.display_string();
        self.text_cursor = self.edit_text.len();
    }

    /// Applies the edit text to the value.
    pub fn apply_edit_text(&mut self) -> bool {
        match &self.property_type {
            PropertyType::Float32 => {
                if let Ok(v) = self.edit_text.parse::<f32>() {
                    let clamped = self.metadata.clamp_value(v as f64) as f32;
                    self.set_value(PropertyValue::Float32(clamped));
                    return true;
                }
            }
            PropertyType::Float64 => {
                if let Ok(v) = self.edit_text.parse::<f64>() {
                    let clamped = self.metadata.clamp_value(v);
                    self.set_value(PropertyValue::Float64(clamped));
                    return true;
                }
            }
            PropertyType::Int32 => {
                if let Ok(v) = self.edit_text.parse::<i32>() {
                    let clamped = self.metadata.clamp_value(v as f64) as i32;
                    self.set_value(PropertyValue::Int32(clamped));
                    return true;
                }
            }
            PropertyType::Int64 => {
                if let Ok(v) = self.edit_text.parse::<i64>() {
                    let clamped = self.metadata.clamp_value(v as f64) as i64;
                    self.set_value(PropertyValue::Int64(clamped));
                    return true;
                }
            }
            PropertyType::UInt32 => {
                if let Ok(v) = self.edit_text.parse::<u32>() {
                    let clamped = self.metadata.clamp_value(v as f64) as u32;
                    self.set_value(PropertyValue::UInt32(clamped));
                    return true;
                }
            }
            PropertyType::UInt64 => {
                if let Ok(v) = self.edit_text.parse::<u64>() {
                    let clamped = self.metadata.clamp_value(v as f64) as u64;
                    self.set_value(PropertyValue::UInt64(clamped));
                    return true;
                }
            }
            PropertyType::String => {
                self.set_value(PropertyValue::String(self.edit_text.clone()));
                return true;
            }
            PropertyType::Angle => {
                if let Ok(v) = self.edit_text.parse::<f32>() {
                    let clamped = self.metadata.clamp_value(v as f64) as f32;
                    self.set_value(PropertyValue::Angle(clamped));
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    /// Updates the expand/collapse animation.
    pub fn update_animation(&mut self, dt: f32) {
        let target = if self.expanded { 1.0 } else { 0.0 };
        let speed = 8.0;
        self.expand_animation += (target - self.expand_animation) * (dt * speed).min(1.0);
    }

    /// Returns true if this row has children.
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Returns true if this row should be rendered (visible and parent expanded).
    pub fn should_render(&self) -> bool {
        self.visible && !self.metadata.hidden
    }

    /// Returns the indentation width for this row.
    pub fn indent_width(&self) -> f32 {
        self.depth as f32 * 16.0
    }
}

// ---------------------------------------------------------------------------
// PropertyCategory
// ---------------------------------------------------------------------------

/// A collapsible section grouping related properties.
#[derive(Debug, Clone)]
pub struct PropertyCategory {
    /// Category name.
    pub name: String,
    /// Whether the category is expanded.
    pub expanded: bool,
    /// Property IDs in this category, in sort order.
    pub properties: Vec<PropId>,
    /// Whether this category has any visible properties.
    pub has_visible_properties: bool,
    /// Sort order for the category itself.
    pub sort_order: i32,
    /// Animation: expand/collapse progress.
    pub expand_animation: f32,
    /// Background color.
    pub background_color: Color,
    /// Header text color.
    pub header_color: Color,
    /// Whether the header is hovered.
    pub hovered: bool,
}

impl PropertyCategory {
    /// Creates a new category.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            expanded: true,
            properties: Vec::new(),
            has_visible_properties: true,
            sort_order: 0,
            expand_animation: 1.0,
            background_color: Color::new(0.12, 0.12, 0.15, 1.0),
            header_color: Color::new(0.75, 0.75, 0.8, 1.0),
            hovered: false,
        }
    }

    /// Toggles the expanded state.
    pub fn toggle(&mut self) {
        self.expanded = !self.expanded;
    }

    /// Updates the expand animation.
    pub fn update_animation(&mut self, dt: f32) {
        let target = if self.expanded { 1.0 } else { 0.0 };
        self.expand_animation += (target - self.expand_animation) * (dt * 8.0).min(1.0);
    }
}

// ---------------------------------------------------------------------------
// UndoEntry
// ---------------------------------------------------------------------------

/// A single entry in the undo/redo stack.
#[derive(Debug, Clone)]
pub struct UndoEntry {
    /// Description of the change.
    pub description: String,
    /// Property ID that was changed.
    pub property_id: PropId,
    /// Property field name.
    pub field_name: String,
    /// Old value (before the change).
    pub old_value: PropertyValue,
    /// New value (after the change).
    pub new_value: PropertyValue,
    /// Object indices affected.
    pub object_indices: Vec<usize>,
    /// Frame number when this change was made.
    pub frame: u64,
    /// Timestamp (for grouping rapid changes).
    pub timestamp: u64,
    /// Group ID (for grouping related changes into one undo step).
    pub group_id: u64,
}

impl UndoEntry {
    /// Creates a new undo entry.
    pub fn new(
        property_id: PropId,
        field_name: &str,
        old_value: PropertyValue,
        new_value: PropertyValue,
    ) -> Self {
        Self {
            description: format!("Change {}", field_name),
            property_id,
            field_name: field_name.to_string(),
            old_value,
            new_value,
            object_indices: vec![0],
            frame: 0,
            timestamp: 0,
            group_id: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// UndoRedoStack
// ---------------------------------------------------------------------------

/// Undo/redo stack for property changes.
///
/// Supports grouping rapid changes into a single undo step (e.g., while
/// dragging a slider, intermediate values are grouped). Undo restores the
/// old value; redo restores the new value.
#[derive(Debug, Clone)]
pub struct UndoRedoStack {
    /// Undo stack (newest last).
    pub undo_stack: Vec<UndoEntry>,
    /// Redo stack (newest last).
    pub redo_stack: Vec<UndoEntry>,
    /// Maximum undo depth.
    pub max_depth: usize,
    /// Current group ID for grouping rapid changes.
    pub current_group: u64,
    /// Next group ID.
    pub next_group: u64,
    /// Timestamp of the last push (for auto-grouping).
    pub last_push_timestamp: u64,
    /// Time window for auto-grouping (in milliseconds).
    pub group_window_ms: u64,
    /// Total undos performed.
    pub total_undos: u64,
    /// Total redos performed.
    pub total_redos: u64,
}

impl UndoRedoStack {
    /// Creates a new undo/redo stack.
    pub fn new() -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_depth: 100,
            current_group: 0,
            next_group: 1,
            last_push_timestamp: 0,
            group_window_ms: 500,
            total_undos: 0,
            total_redos: 0,
        }
    }

    /// Pushes a new change onto the undo stack.
    pub fn push(&mut self, mut entry: UndoEntry, timestamp: u64) {
        // Auto-group rapid changes to the same property.
        if timestamp - self.last_push_timestamp < self.group_window_ms
            && !self.undo_stack.is_empty()
        {
            let last = self.undo_stack.last().unwrap();
            if last.property_id == entry.property_id {
                entry.group_id = last.group_id;
                // Merge: keep the old_value from the first entry in the group.
                entry.old_value = last.old_value.clone();
                self.undo_stack.pop();
            } else {
                self.current_group = self.next_group;
                self.next_group += 1;
                entry.group_id = self.current_group;
            }
        } else {
            self.current_group = self.next_group;
            self.next_group += 1;
            entry.group_id = self.current_group;
        }

        entry.timestamp = timestamp;
        self.undo_stack.push(entry);
        self.redo_stack.clear();
        self.last_push_timestamp = timestamp;

        // Enforce max depth.
        if self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    /// Undoes the last change. Returns the entry to apply.
    pub fn undo(&mut self) -> Option<UndoEntry> {
        if let Some(entry) = self.undo_stack.pop() {
            self.redo_stack.push(entry.clone());
            self.total_undos += 1;
            Some(entry)
        } else {
            None
        }
    }

    /// Redoes the last undone change. Returns the entry to apply.
    pub fn redo(&mut self) -> Option<UndoEntry> {
        if let Some(entry) = self.redo_stack.pop() {
            self.undo_stack.push(entry.clone());
            self.total_redos += 1;
            Some(entry)
        } else {
            None
        }
    }

    /// Returns true if undo is possible.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Returns true if redo is possible.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Returns the description of the next undo action.
    pub fn undo_description(&self) -> Option<&str> {
        self.undo_stack.last().map(|e| e.description.as_str())
    }

    /// Returns the description of the next redo action.
    pub fn redo_description(&self) -> Option<&str> {
        self.redo_stack.last().map(|e| e.description.as_str())
    }

    /// Clears all undo/redo history.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Returns the total number of entries in the undo stack.
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// Returns the total number of entries in the redo stack.
    pub fn redo_depth(&self) -> usize {
        self.redo_stack.len()
    }

    /// Begins a new undo group (forces next push to start a new group).
    pub fn begin_group(&mut self) {
        self.current_group = self.next_group;
        self.next_group += 1;
    }
}

impl Default for UndoRedoStack {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PropertyFilter
// ---------------------------------------------------------------------------

/// Filters visible properties based on search text, category, and advanced flag.
#[derive(Debug, Clone)]
pub struct PropertyFilter {
    /// Current search text.
    pub search_text: String,
    /// Whether to show advanced properties.
    pub show_advanced: bool,
    /// Categories to show (empty = all).
    pub visible_categories: HashSet<String>,
    /// Whether filtering is active.
    pub active: bool,
    /// Number of properties matching the filter.
    pub match_count: u32,
    /// Total number of properties.
    pub total_count: u32,
    /// Whether to search in tooltips too.
    pub search_tooltips: bool,
    /// Whether the search is case-sensitive.
    pub case_sensitive: bool,
}

impl PropertyFilter {
    /// Creates a new property filter.
    pub fn new() -> Self {
        Self {
            search_text: String::new(),
            show_advanced: false,
            visible_categories: HashSet::new(),
            active: false,
            match_count: 0,
            total_count: 0,
            search_tooltips: true,
            case_sensitive: false,
        }
    }

    /// Sets the search text.
    pub fn set_search(&mut self, text: &str) {
        self.search_text = text.to_string();
        self.active = !text.is_empty() || !self.show_advanced;
    }

    /// Returns true if a property matches the current filter.
    pub fn matches(&self, row: &PropertyRow) -> bool {
        // Hide advanced properties unless toggled.
        if row.metadata.advanced && !self.show_advanced {
            return false;
        }

        // Hide if category is filtered out.
        if !self.visible_categories.is_empty()
            && !self.visible_categories.contains(&row.metadata.category)
        {
            return false;
        }

        // Search text filter.
        if !self.search_text.is_empty() {
            let search = if self.case_sensitive {
                self.search_text.clone()
            } else {
                self.search_text.to_lowercase()
            };

            let name = if self.case_sensitive {
                row.metadata.display_name.clone()
            } else {
                row.metadata.display_name.to_lowercase()
            };

            let field = if self.case_sensitive {
                row.metadata.field_name.clone()
            } else {
                row.metadata.field_name.to_lowercase()
            };

            let matches_name = name.contains(&search) || field.contains(&search);
            let matches_tooltip = self.search_tooltips
                && row.metadata.tooltip.as_ref().map_or(false, |t| {
                    if self.case_sensitive {
                        t.contains(&search)
                    } else {
                        t.to_lowercase().contains(&search)
                    }
                });

            return matches_name || matches_tooltip;
        }

        true
    }

    /// Toggles visibility of a category.
    pub fn toggle_category(&mut self, category: &str) {
        if self.visible_categories.contains(category) {
            self.visible_categories.remove(category);
        } else {
            self.visible_categories.insert(category.to_string());
        }
    }

    /// Clears the filter.
    pub fn clear(&mut self) {
        self.search_text.clear();
        self.visible_categories.clear();
        self.active = false;
    }
}

impl Default for PropertyFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PropertyChange
// ---------------------------------------------------------------------------

/// Represents a change made in the property editor.
#[derive(Debug, Clone)]
pub struct PropertyChange {
    /// Which property changed.
    pub property_id: PropId,
    /// The field name.
    pub field_name: String,
    /// The old value.
    pub old_value: PropertyValue,
    /// The new value.
    pub new_value: PropertyValue,
    /// Which objects were affected (indices).
    pub object_indices: Vec<usize>,
    /// Whether this change should be added to undo history.
    pub undoable: bool,
}

// ---------------------------------------------------------------------------
// MultiObjectState
// ---------------------------------------------------------------------------

/// Tracks multi-object editing state.
///
/// When multiple entities/objects are selected, the property editor shows
/// their common properties. If all selected objects have the same value for
/// a property, that value is shown normally. If they differ, the field shows
/// a "mixed values" indicator.
#[derive(Debug, Clone)]
pub struct MultiObjectState {
    /// Number of objects being edited.
    pub object_count: usize,
    /// Values for each object, keyed by field name.
    pub object_values: HashMap<String, Vec<PropertyValue>>,
    /// Whether each property has mixed values across objects.
    pub mixed_properties: HashSet<String>,
    /// The common type name (if all objects are the same type).
    pub type_name: Option<String>,
    /// Object display names.
    pub object_names: Vec<String>,
}

impl MultiObjectState {
    /// Creates a new multi-object state.
    pub fn new() -> Self {
        Self {
            object_count: 0,
            object_values: HashMap::new(),
            mixed_properties: HashSet::new(),
            type_name: None,
            object_names: Vec::new(),
        }
    }

    /// Adds an object's values.
    pub fn add_object(&mut self, name: &str, values: HashMap<String, PropertyValue>) {
        self.object_count += 1;
        self.object_names.push(name.to_string());

        for (field, value) in values {
            let entry = self
                .object_values
                .entry(field.clone())
                .or_insert_with(Vec::new);
            entry.push(value);
        }
    }

    /// Computes which properties have mixed values.
    pub fn compute_mixed(&mut self) {
        self.mixed_properties.clear();
        for (field, values) in &self.object_values {
            if values.len() > 1 {
                let first = &values[0];
                if values.iter().any(|v| !v.equals(first)) {
                    self.mixed_properties.insert(field.clone());
                }
            }
        }
    }

    /// Returns the common value for a property, or None if mixed.
    pub fn common_value(&self, field: &str) -> Option<&PropertyValue> {
        if self.mixed_properties.contains(field) {
            None
        } else {
            self.object_values
                .get(field)
                .and_then(|v| v.first())
        }
    }

    /// Returns true if a property has mixed values.
    pub fn is_mixed(&self, field: &str) -> bool {
        self.mixed_properties.contains(field)
    }

    /// Returns true if multi-object editing is active.
    pub fn is_active(&self) -> bool {
        self.object_count > 1
    }

    /// Clears all multi-object state.
    pub fn clear(&mut self) {
        self.object_count = 0;
        self.object_values.clear();
        self.mixed_properties.clear();
        self.object_names.clear();
    }
}

impl Default for MultiObjectState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ColorPickerState
// ---------------------------------------------------------------------------

/// State for the inline color picker popup.
#[derive(Debug, Clone)]
pub struct ColorPickerState {
    /// Whether the picker popup is open.
    pub open: bool,
    /// Current color value (RGBA).
    pub color: [f32; 4],
    /// Previous color (for cancel).
    pub original_color: [f32; 4],
    /// HSV representation.
    pub hue: f32,
    pub saturation: f32,
    pub value_brightness: f32,
    /// Alpha value.
    pub alpha: f32,
    /// Which property this picker is editing.
    pub property_id: Option<PropId>,
    /// Popup position.
    pub position: Vec2,
    /// Popup size.
    pub size: Vec2,
    /// Whether the SV quad is being dragged.
    pub dragging_sv: bool,
    /// Whether the hue strip is being dragged.
    pub dragging_hue: bool,
    /// Whether the alpha strip is being dragged.
    pub dragging_alpha: bool,
    /// Hex color input text.
    pub hex_text: String,
    /// Whether to show the alpha slider.
    pub show_alpha: bool,
    /// Saved color presets.
    pub presets: Vec<[f32; 4]>,
}

impl ColorPickerState {
    /// Creates a new color picker.
    pub fn new() -> Self {
        Self {
            open: false,
            color: [1.0, 1.0, 1.0, 1.0],
            original_color: [1.0, 1.0, 1.0, 1.0],
            hue: 0.0,
            saturation: 0.0,
            value_brightness: 1.0,
            alpha: 1.0,
            property_id: None,
            position: Vec2::ZERO,
            size: Vec2::new(250.0, 300.0),
            dragging_sv: false,
            dragging_hue: false,
            dragging_alpha: false,
            hex_text: String::new(),
            show_alpha: true,
            presets: vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.5, 0.5, 0.5, 1.0],
            ],
        }
    }

    /// Opens the picker for a property.
    pub fn open_for(&mut self, property_id: PropId, color: [f32; 4], position: Vec2) {
        self.open = true;
        self.color = color;
        self.original_color = color;
        self.property_id = Some(property_id);
        self.position = position;
        self.rgb_to_hsv();
        self.update_hex();
    }

    /// Closes the picker.
    pub fn close(&mut self) {
        self.open = false;
        self.property_id = None;
        self.dragging_sv = false;
        self.dragging_hue = false;
        self.dragging_alpha = false;
    }

    /// Cancels the picker (reverts to original color).
    pub fn cancel(&mut self) {
        self.color = self.original_color;
        self.close();
    }

    /// Converts current HSV to RGB.
    pub fn hsv_to_rgb(&mut self) {
        let h = self.hue;
        let s = self.saturation;
        let v = self.value_brightness;
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };
        self.color = [r + m, g + m, b + m, self.alpha];
    }

    /// Converts current RGB to HSV.
    pub fn rgb_to_hsv(&mut self) {
        let r = self.color[0];
        let g = self.color[1];
        let b = self.color[2];
        self.alpha = self.color[3];

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        self.value_brightness = max;
        self.saturation = if max > 0.0 { delta / max } else { 0.0 };

        if delta < 0.0001 {
            self.hue = 0.0;
        } else if (max - r).abs() < 0.0001 {
            self.hue = 60.0 * (((g - b) / delta) % 6.0);
        } else if (max - g).abs() < 0.0001 {
            self.hue = 60.0 * ((b - r) / delta + 2.0);
        } else {
            self.hue = 60.0 * ((r - g) / delta + 4.0);
        }
        if self.hue < 0.0 {
            self.hue += 360.0;
        }
    }

    /// Updates the hex text from the current color.
    pub fn update_hex(&mut self) {
        let r = (self.color[0] * 255.0) as u8;
        let g = (self.color[1] * 255.0) as u8;
        let b = (self.color[2] * 255.0) as u8;
        self.hex_text = format!("{:02X}{:02X}{:02X}", r, g, b);
    }

    /// Parses the hex text and updates the color.
    pub fn apply_hex(&mut self) -> bool {
        let hex = self.hex_text.trim_start_matches('#');
        if hex.len() == 6 {
            if let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&hex[0..2], 16),
                u8::from_str_radix(&hex[2..4], 16),
                u8::from_str_radix(&hex[4..6], 16),
            ) {
                self.color[0] = r as f32 / 255.0;
                self.color[1] = g as f32 / 255.0;
                self.color[2] = b as f32 / 255.0;
                self.rgb_to_hsv();
                return true;
            }
        }
        false
    }

    /// Saves the current color as a preset.
    pub fn save_preset(&mut self) {
        if self.presets.len() >= 32 {
            self.presets.remove(0);
        }
        self.presets.push(self.color);
    }

    /// Applies a preset color.
    pub fn apply_preset(&mut self, index: usize) {
        if index < self.presets.len() {
            self.color = self.presets[index];
            self.rgb_to_hsv();
            self.update_hex();
        }
    }
}

impl Default for ColorPickerState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ArrayEditorState
// ---------------------------------------------------------------------------

/// State for an array property editor.
#[derive(Debug, Clone)]
pub struct ArrayEditorState {
    /// Property ID of the array.
    pub property_id: PropId,
    /// Current element count.
    pub element_count: usize,
    /// Element type.
    pub element_type: PropertyType,
    /// Whether the array is expanded.
    pub expanded: bool,
    /// Whether drag-reorder is in progress.
    pub reordering: bool,
    /// Index being dragged.
    pub drag_index: Option<usize>,
    /// Drop target index.
    pub drop_index: Option<usize>,
    /// Maximum elements allowed (0 = unlimited).
    pub max_elements: usize,
    /// Minimum elements required.
    pub min_elements: usize,
    /// Whether adding is allowed.
    pub can_add: bool,
    /// Whether removing is allowed.
    pub can_remove: bool,
    /// Whether reordering is allowed.
    pub can_reorder: bool,
}

impl ArrayEditorState {
    /// Creates a new array editor state.
    pub fn new(property_id: PropId, element_type: PropertyType, count: usize) -> Self {
        Self {
            property_id,
            element_count: count,
            element_type,
            expanded: true,
            reordering: false,
            drag_index: None,
            drop_index: None,
            max_elements: 0,
            min_elements: 0,
            can_add: true,
            can_remove: true,
            can_reorder: true,
        }
    }

    /// Returns true if an element can be added.
    pub fn can_add_element(&self) -> bool {
        self.can_add && (self.max_elements == 0 || self.element_count < self.max_elements)
    }

    /// Returns true if an element can be removed.
    pub fn can_remove_element(&self) -> bool {
        self.can_remove && self.element_count > self.min_elements
    }

    /// Begins a reorder drag.
    pub fn begin_reorder(&mut self, index: usize) {
        if self.can_reorder {
            self.reordering = true;
            self.drag_index = Some(index);
        }
    }

    /// Updates the drop target during reorder.
    pub fn update_drop_target(&mut self, index: usize) {
        if self.reordering {
            self.drop_index = Some(index);
        }
    }

    /// Finishes a reorder drag, returning the (from, to) indices if valid.
    pub fn finish_reorder(&mut self) -> Option<(usize, usize)> {
        let result = if let (Some(from), Some(to)) = (self.drag_index, self.drop_index) {
            if from != to {
                Some((from, to))
            } else {
                None
            }
        } else {
            None
        };
        self.reordering = false;
        self.drag_index = None;
        self.drop_index = None;
        result
    }

    /// Cancels a reorder drag.
    pub fn cancel_reorder(&mut self) {
        self.reordering = false;
        self.drag_index = None;
        self.drop_index = None;
    }
}

// ---------------------------------------------------------------------------
// AssetRefEditorState
// ---------------------------------------------------------------------------

/// State for an asset reference editor.
#[derive(Debug, Clone)]
pub struct AssetRefEditorState {
    /// Property ID.
    pub property_id: PropId,
    /// Current asset path.
    pub asset_path: String,
    /// Asset type filter (e.g., "Texture", "Mesh", "Material").
    pub asset_type: String,
    /// Whether the browse dialog is open.
    pub browse_open: bool,
    /// Whether this is a valid asset reference.
    pub valid: bool,
    /// Thumbnail texture ID (if available).
    pub thumbnail: Option<u64>,
    /// Whether the drag-drop target is active.
    pub drag_target_active: bool,
    /// Whether a drag-drop is hovering over this field.
    pub drag_hovering: bool,
    /// Display name of the referenced asset.
    pub display_name: String,
}

impl AssetRefEditorState {
    /// Creates a new asset reference editor state.
    pub fn new(property_id: PropId, asset_type: &str) -> Self {
        Self {
            property_id,
            asset_path: String::new(),
            asset_type: asset_type.to_string(),
            browse_open: false,
            valid: false,
            thumbnail: None,
            drag_target_active: true,
            drag_hovering: false,
            display_name: "None".to_string(),
        }
    }

    /// Sets the asset path.
    pub fn set_path(&mut self, path: &str) {
        self.asset_path = path.to_string();
        self.valid = !path.is_empty();
        self.display_name = if path.is_empty() {
            "None".to_string()
        } else {
            // Extract filename from path.
            path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
        };
    }

    /// Clears the asset reference.
    pub fn clear(&mut self) {
        self.asset_path.clear();
        self.valid = false;
        self.display_name = "None".to_string();
        self.thumbnail = None;
    }

    /// Opens the browse dialog.
    pub fn browse(&mut self) {
        self.browse_open = true;
    }
}

// ---------------------------------------------------------------------------
// PropertyEditorConfig
// ---------------------------------------------------------------------------

/// Configuration for the property editor appearance and behavior.
#[derive(Debug, Clone)]
pub struct PropertyEditorConfig {
    /// Label width fraction (0.0 to 1.0).
    pub label_fraction: f32,
    /// Row height in pixels.
    pub row_height: f32,
    /// Category header height.
    pub category_height: f32,
    /// Indent width per depth level.
    pub indent_width: f32,
    /// Background color for even rows.
    pub row_bg_even: Color,
    /// Background color for odd rows.
    pub row_bg_odd: Color,
    /// Background color for hovered rows.
    pub row_bg_hover: Color,
    /// Background color for selected rows.
    pub row_bg_selected: Color,
    /// Label text color.
    pub label_color: Color,
    /// Dim label text color.
    pub label_dim_color: Color,
    /// Value text color.
    pub value_color: Color,
    /// Mixed-value indicator color.
    pub mixed_color: Color,
    /// Category header background color.
    pub category_bg: Color,
    /// Category header text color.
    pub category_text_color: Color,
    /// Reset-to-default button color.
    pub reset_button_color: Color,
    /// Font size for labels.
    pub font_size: f32,
    /// Font size for values.
    pub value_font_size: f32,
    /// Whether to show type indicators.
    pub show_type_indicator: bool,
    /// Whether to show tooltips on hover.
    pub show_tooltips: bool,
    /// Whether to show the search bar.
    pub show_search: bool,
    /// Whether to show the "show advanced" toggle.
    pub show_advanced_toggle: bool,
    /// Color swatch size.
    pub color_swatch_size: f32,
    /// Vec3 axis colors (R, G, B).
    pub axis_colors: [Color; 3],
}

impl PropertyEditorConfig {
    /// Creates a new config with dark theme defaults.
    pub fn dark_theme() -> Self {
        Self {
            label_fraction: 0.4,
            row_height: 24.0,
            category_height: 28.0,
            indent_width: 16.0,
            row_bg_even: Color::new(0.14, 0.14, 0.17, 1.0),
            row_bg_odd: Color::new(0.15, 0.15, 0.18, 1.0),
            row_bg_hover: Color::new(0.18, 0.18, 0.22, 1.0),
            row_bg_selected: Color::new(0.2, 0.25, 0.35, 1.0),
            label_color: Color::new(0.75, 0.75, 0.8, 1.0),
            label_dim_color: Color::new(0.5, 0.5, 0.55, 1.0),
            value_color: Color::new(0.9, 0.9, 0.9, 1.0),
            mixed_color: Color::new(0.6, 0.6, 0.3, 0.7),
            category_bg: Color::new(0.12, 0.12, 0.15, 1.0),
            category_text_color: Color::new(0.8, 0.8, 0.85, 1.0),
            reset_button_color: Color::new(0.4, 0.6, 1.0, 0.6),
            font_size: 12.0,
            value_font_size: 12.0,
            show_type_indicator: false,
            show_tooltips: true,
            show_search: true,
            show_advanced_toggle: true,
            color_swatch_size: 16.0,
            axis_colors: [
                Color::new(0.9, 0.2, 0.2, 1.0),
                Color::new(0.2, 0.9, 0.2, 1.0),
                Color::new(0.3, 0.4, 0.9, 1.0),
            ],
        }
    }
}

impl Default for PropertyEditorConfig {
    fn default() -> Self {
        Self::dark_theme()
    }
}

// ---------------------------------------------------------------------------
// PropertyEditor
// ---------------------------------------------------------------------------

/// Auto-generated property editor panel.
///
/// Given reflected type information, automatically generates the appropriate
/// edit UI for each property. Supports nested structs, arrays, enums, colors,
/// vectors, asset references, and all primitive types.
///
/// # Features
///
/// - **Auto-generation**: Given a list of `PropertyRow` entries, lays out
///   label + widget pairs with proper alignment.
/// - **Categories**: Properties are grouped by category with collapsible headers.
/// - **Filtering**: Search box filters visible properties by name.
/// - **Undo/Redo**: Every change creates an undo entry with grouping.
/// - **Multi-object editing**: Shows common properties across multiple objects.
/// - **Reset to default**: Per-property reset button appears on hover.
/// - **Color picker**: Inline swatch with popup HSV picker.
/// - **Array editing**: Add, remove, reorder elements.
///
/// # Usage
///
/// ```ignore
/// let mut editor = PropertyEditor::new();
///
/// // Add properties from reflected type.
/// editor.add_property(
///     PropertyMetadata::new("health", "Health")
///         .with_range(0.0, 100.0)
///         .with_slider(),
///     PropertyType::Float32,
///     PropertyValue::Float32(75.0),
/// );
///
/// editor.add_property(
///     PropertyMetadata::new("name", "Name"),
///     PropertyType::String,
///     PropertyValue::String("Player".to_string()),
/// );
///
/// // Each frame:
/// editor.update(dt);
/// for change in editor.drain_changes() {
///     // Apply changes to the source object.
/// }
/// ```
#[derive(Debug, Clone)]
pub struct PropertyEditor {
    /// All property rows, keyed by PropId.
    pub rows: HashMap<u64, PropertyRow>,
    /// Property rows in display order.
    pub display_order: Vec<PropId>,
    /// Categories.
    pub categories: HashMap<String, PropertyCategory>,
    /// Category display order.
    pub category_order: Vec<String>,
    /// Property filter.
    pub filter: PropertyFilter,
    /// Undo/redo stack.
    pub undo_redo: UndoRedoStack,
    /// Configuration.
    pub config: PropertyEditorConfig,
    /// Multi-object editing state.
    pub multi_object: MultiObjectState,
    /// Color picker state.
    pub color_picker: ColorPickerState,
    /// Array editor states, keyed by property ID.
    pub array_editors: HashMap<u64, ArrayEditorState>,
    /// Asset reference editor states.
    pub asset_editors: HashMap<u64, AssetRefEditorState>,
    /// Pending property changes (consumed by the application).
    pub pending_changes: Vec<PropertyChange>,
    /// Total width of the editor panel.
    pub width: f32,
    /// Total computed height of all visible rows.
    pub total_height: f32,
    /// Scroll offset.
    pub scroll_offset: f32,
    /// Viewport height.
    pub viewport_height: f32,
    /// Whether the editor is enabled.
    pub enabled: bool,
    /// Current frame number.
    pub current_frame: u64,
    /// Currently hovered row.
    pub hovered_row: Option<PropId>,
    /// Currently selected row.
    pub selected_row: Option<PropId>,
    /// Title of the editor panel.
    pub title: String,
    /// Type name of the object being edited.
    pub type_name: String,
    /// Icon for the type being edited.
    pub type_icon: Option<u64>,
    /// Whether the editor has any content.
    pub has_content: bool,
    /// Total property count.
    pub property_count: u32,
    /// Visible property count.
    pub visible_property_count: u32,
    /// Timestamp counter for undo grouping.
    pub timestamp: u64,
}

impl PropertyEditor {
    /// Creates a new property editor.
    pub fn new() -> Self {
        Self {
            rows: HashMap::new(),
            display_order: Vec::new(),
            categories: HashMap::new(),
            category_order: Vec::new(),
            filter: PropertyFilter::new(),
            undo_redo: UndoRedoStack::new(),
            config: PropertyEditorConfig::dark_theme(),
            multi_object: MultiObjectState::new(),
            color_picker: ColorPickerState::new(),
            array_editors: HashMap::new(),
            asset_editors: HashMap::new(),
            pending_changes: Vec::new(),
            width: 350.0,
            total_height: 0.0,
            scroll_offset: 0.0,
            viewport_height: 600.0,
            enabled: true,
            current_frame: 0,
            hovered_row: None,
            selected_row: None,
            title: "Properties".to_string(),
            type_name: String::new(),
            type_icon: None,
            has_content: false,
            property_count: 0,
            visible_property_count: 0,
            timestamp: 0,
        }
    }

    /// Clears all properties (for when the selection changes).
    pub fn clear(&mut self) {
        self.rows.clear();
        self.display_order.clear();
        self.categories.clear();
        self.category_order.clear();
        self.array_editors.clear();
        self.asset_editors.clear();
        self.pending_changes.clear();
        self.hovered_row = None;
        self.selected_row = None;
        self.has_content = false;
        self.property_count = 0;
        self.visible_property_count = 0;
        self.color_picker.close();
        self.multi_object.clear();
    }

    /// Adds a property to the editor.
    pub fn add_property(
        &mut self,
        metadata: PropertyMetadata,
        property_type: PropertyType,
        value: PropertyValue,
    ) -> PropId {
        let category_name = metadata.category.clone();
        let row = PropertyRow::new(metadata, property_type.clone(), value);
        let id = row.id;

        // Ensure category exists.
        if !self.categories.contains_key(&category_name) {
            self.categories
                .insert(category_name.clone(), PropertyCategory::new(&category_name));
            self.category_order.push(category_name.clone());
        }
        self.categories
            .get_mut(&category_name)
            .unwrap()
            .properties
            .push(id);

        // Create specialized editor state if needed.
        match &property_type {
            PropertyType::Array(element_type) => {
                let count = match &row.value {
                    PropertyValue::Array(arr) => arr.len(),
                    _ => 0,
                };
                self.array_editors.insert(
                    id.0,
                    ArrayEditorState::new(id, (**element_type).clone(), count),
                );
            }
            PropertyType::AssetRef(asset_type) => {
                let mut state = AssetRefEditorState::new(id, asset_type);
                if let PropertyValue::AssetRef(path) = &row.value {
                    state.set_path(path);
                }
                self.asset_editors.insert(id.0, state);
            }
            _ => {}
        }

        self.rows.insert(id.0, row);
        self.display_order.push(id);
        self.property_count += 1;
        self.has_content = true;

        id
    }

    /// Adds a nested struct property with sub-properties.
    pub fn add_struct_property(
        &mut self,
        metadata: PropertyMetadata,
        struct_name: &str,
    ) -> PropId {
        let row = PropertyRow::new(
            metadata,
            PropertyType::Struct(struct_name.to_string()),
            PropertyValue::Struct(HashMap::new()),
        );
        let id = row.id;

        let category_name = row.metadata.category.clone();
        if !self.categories.contains_key(&category_name) {
            self.categories
                .insert(category_name.clone(), PropertyCategory::new(&category_name));
            self.category_order.push(category_name.clone());
        }
        self.categories
            .get_mut(&category_name)
            .unwrap()
            .properties
            .push(id);

        self.rows.insert(id.0, row);
        self.display_order.push(id);
        self.property_count += 1;
        self.has_content = true;

        id
    }

    /// Adds a child property to a parent (struct or array).
    pub fn add_child_property(
        &mut self,
        parent_id: PropId,
        metadata: PropertyMetadata,
        property_type: PropertyType,
        value: PropertyValue,
    ) -> PropId {
        let mut row = PropertyRow::new(metadata, property_type, value);
        row.parent = Some(parent_id);
        if let Some(parent) = self.rows.get(&parent_id.0) {
            row.depth = parent.depth + 1;
        }
        let id = row.id;

        self.rows.insert(id.0, row);
        if let Some(parent) = self.rows.get_mut(&parent_id.0) {
            parent.children.push(id);
        }
        self.display_order.push(id);
        self.property_count += 1;

        id
    }

    /// Sets the value of a property.
    pub fn set_value(&mut self, prop_id: PropId, value: PropertyValue) {
        if let Some(row) = self.rows.get_mut(&prop_id.0) {
            let old_value = row.value.clone();
            row.set_value(value.clone());

            if row.dirty && row.metadata.undoable {
                let entry = UndoEntry::new(
                    prop_id,
                    &row.metadata.field_name,
                    old_value.clone(),
                    value.clone(),
                );
                self.undo_redo.push(entry, self.timestamp);
            }

            if row.dirty {
                self.pending_changes.push(PropertyChange {
                    property_id: prop_id,
                    field_name: row.metadata.field_name.clone(),
                    old_value,
                    new_value: value,
                    object_indices: row.object_indices.clone(),
                    undoable: row.metadata.undoable,
                });
            }
        }
    }

    /// Updates the editor for one frame.
    pub fn update(&mut self, dt: f32) {
        self.current_frame += 1;
        self.timestamp += 1;

        // Update visibility based on filter.
        self.visible_property_count = 0;
        let filter = self.filter.clone();

        // Pre-compute condition results to avoid borrow conflicts.
        let condition_results: HashMap<String, bool> = {
            let rows = &self.rows;
            rows.values()
                .filter_map(|row| {
                    row.metadata.show_condition.as_ref().map(|condition| {
                        let condition_met = rows
                            .values()
                            .find(|r| r.metadata.field_name == condition.property_name)
                            .map(|r| condition.evaluate(&r.value))
                            .unwrap_or(true);
                        (row.metadata.field_name.clone(), condition_met)
                    })
                })
                .collect()
        };

        for row in self.rows.values_mut() {
            let was_visible = row.visible;
            row.visible = filter.matches(row);

            // Check show conditions.
            if row.metadata.show_condition.is_some() {
                let condition_met = condition_results
                    .get(&row.metadata.field_name)
                    .copied()
                    .unwrap_or(true);
                row.visible = row.visible && condition_met;
            }

            if row.visible {
                self.visible_property_count += 1;
            }

            row.update_animation(dt);
            row.show_reset = row.hovered && row.differs_from_default();
        }

        // Update category animations.
        for category in self.categories.values_mut() {
            category.update_animation(dt);
        }

        // Compute layout.
        self.compute_layout();

        // Update filter stats.
        self.filter.match_count = self.visible_property_count;
        self.filter.total_count = self.property_count;
    }

    /// Computes the Y positions of all visible rows.
    fn compute_layout(&mut self) {
        let mut y = 0.0;
        let row_height = self.config.row_height;
        let category_height = self.config.category_height;

        // Sort categories.
        let category_order = self.category_order.clone();
        for cat_name in &category_order {
            if let Some(category) = self.categories.get(cat_name) {
                // Category header.
                y += category_height;

                if category.expanded {
                    // Sort properties within category.
                    let mut props = category.properties.clone();
                    props.sort_by(|a, b| {
                        let a_order = self
                            .rows
                            .get(&a.0)
                            .map(|r| r.metadata.sort_order)
                            .unwrap_or(0);
                        let b_order = self
                            .rows
                            .get(&b.0)
                            .map(|r| r.metadata.sort_order)
                            .unwrap_or(0);
                        a_order.cmp(&b_order)
                    });

                    for prop_id in &props {
                        if let Some(row) = self.rows.get_mut(&prop_id.0) {
                            if row.should_render() {
                                row.y_position = y;
                                y += row_height;

                                // Add children if expanded.
                                if row.expanded && row.has_children() {
                                    for child_id in row.children.clone() {
                                        if let Some(child) = self.rows.get_mut(&child_id.0) {
                                            if child.should_render() {
                                                child.y_position = y;
                                                y += row_height;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        self.total_height = y;
    }

    /// Handles a click at the given position.
    pub fn on_click(&mut self, local_pos: Vec2) {
        let y = local_pos.y + self.scroll_offset;

        // Check category headers.
        let mut toggle_cat = None;
        let category_order = self.category_order.clone();
        let mut cat_y = 0.0;
        for cat_name in &category_order {
            if y >= cat_y && y < cat_y + self.config.category_height {
                toggle_cat = Some(cat_name.clone());
                break;
            }
            cat_y += self.config.category_height;
            if let Some(category) = self.categories.get(cat_name) {
                if category.expanded {
                    for prop_id in &category.properties {
                        if let Some(row) = self.rows.get(&prop_id.0) {
                            if row.should_render() {
                                cat_y += self.config.row_height;
                                if row.expanded {
                                    cat_y += row.children.len() as f32 * self.config.row_height;
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(cat_name) = toggle_cat {
            if let Some(category) = self.categories.get_mut(&cat_name) {
                category.toggle();
            }
            return;
        }

        // Check property rows.
        let mut deferred_bool_toggle: Option<(PropId, bool)> = None;
        for row in self.rows.values_mut() {
            if row.should_render()
                && y >= row.y_position
                && y < row.y_position + self.config.row_height
            {
                // Check if the click is on the label side (expand/collapse).
                let label_width = self.width * self.config.label_fraction;
                if local_pos.x < label_width && row.has_children() {
                    row.expanded = !row.expanded;
                } else if local_pos.x < label_width {
                    self.selected_row = Some(row.id);
                } else {
                    // Click on the value side: begin editing.
                    self.selected_row = Some(row.id);

                    // Handle specific widget types.
                    match &row.property_type {
                        PropertyType::Bool => {
                            if let PropertyValue::Bool(v) = row.value {
                                deferred_bool_toggle = Some((row.id, !v));
                            }
                        }
                        PropertyType::Color => {
                            if let PropertyValue::Color(c) = row.value {
                                let pos = Vec2::new(
                                    local_pos.x,
                                    row.y_position - self.scroll_offset,
                                );
                                self.color_picker.open_for(row.id, c, pos);
                            }
                        }
                        _ => {
                            if row.property_type.is_inline() && !row.metadata.read_only {
                                row.begin_edit();
                            }
                        }
                    }
                }
                break;
            }
        }
        if let Some((id, val)) = deferred_bool_toggle {
            self.set_value(id, PropertyValue::Bool(val));
        }
    }

    /// Handles hover at the given position.
    pub fn on_hover(&mut self, local_pos: Vec2) {
        let y = local_pos.y + self.scroll_offset;
        self.hovered_row = None;

        for row in self.rows.values_mut() {
            row.hovered = false;
            if row.should_render()
                && y >= row.y_position
                && y < row.y_position + self.config.row_height
            {
                row.hovered = true;
                self.hovered_row = Some(row.id);
            }
        }

        // Update category hover state.
        for category in self.categories.values_mut() {
            category.hovered = false;
        }
    }

    /// Handles scroll input.
    pub fn on_scroll(&mut self, delta_y: f32) {
        self.scroll_offset = (self.scroll_offset - delta_y * 30.0)
            .clamp(0.0, (self.total_height - self.viewport_height).max(0.0));
    }

    /// Performs undo.
    pub fn undo(&mut self) {
        if let Some(entry) = self.undo_redo.undo() {
            if let Some(row) = self.rows.get_mut(&entry.property_id.0) {
                row.value = entry.old_value.clone();
                row.edit_text = row.value.display_string();
                self.pending_changes.push(PropertyChange {
                    property_id: entry.property_id,
                    field_name: entry.field_name.clone(),
                    old_value: entry.new_value,
                    new_value: entry.old_value,
                    object_indices: entry.object_indices,
                    undoable: false,
                });
            }
        }
    }

    /// Performs redo.
    pub fn redo(&mut self) {
        if let Some(entry) = self.undo_redo.redo() {
            if let Some(row) = self.rows.get_mut(&entry.property_id.0) {
                row.value = entry.new_value.clone();
                row.edit_text = row.value.display_string();
                self.pending_changes.push(PropertyChange {
                    property_id: entry.property_id,
                    field_name: entry.field_name.clone(),
                    old_value: entry.old_value,
                    new_value: entry.new_value,
                    object_indices: entry.object_indices,
                    undoable: false,
                });
            }
        }
    }

    /// Drains and returns all pending changes since the last drain.
    pub fn drain_changes(&mut self) -> Vec<PropertyChange> {
        std::mem::take(&mut self.pending_changes)
    }

    /// Returns a list of render rows for the UI.
    pub fn visible_rows(&self) -> Vec<PropertyRowRenderData> {
        let mut render_rows = Vec::new();
        let vis_top = self.scroll_offset;
        let vis_bottom = vis_top + self.viewport_height;

        let category_order = &self.category_order;
        for cat_name in category_order {
            if let Some(category) = self.categories.get(cat_name) {
                // Add category header.
                // ... (actual rendering would be done by the UI framework)

                if category.expanded {
                    for prop_id in &category.properties {
                        if let Some(row) = self.rows.get(&prop_id.0) {
                            if row.should_render()
                                && row.y_position + self.config.row_height > vis_top
                                && row.y_position < vis_bottom
                            {
                                render_rows.push(PropertyRowRenderData {
                                    id: row.id,
                                    label: row.label().to_string(),
                                    value_text: row.value.display_string(),
                                    property_type: row.property_type.clone(),
                                    y_position: row.y_position - self.scroll_offset,
                                    height: self.config.row_height,
                                    depth: row.depth,
                                    expanded: row.expanded,
                                    has_children: row.has_children(),
                                    hovered: row.hovered,
                                    selected: self.selected_row == Some(row.id),
                                    editing: row.editing,
                                    differs_from_default: row.differs_from_default(),
                                    mixed_value: row.mixed_value,
                                    read_only: row.metadata.read_only,
                                    tooltip: row.metadata.tooltip.clone(),
                                    units: row.metadata.units.clone(),
                                });
                            }

                            // Children.
                            if row.expanded {
                                for child_id in &row.children {
                                    if let Some(child) = self.rows.get(&child_id.0) {
                                        if child.should_render()
                                            && child.y_position + self.config.row_height > vis_top
                                            && child.y_position < vis_bottom
                                        {
                                            render_rows.push(PropertyRowRenderData {
                                                id: child.id,
                                                label: child.label().to_string(),
                                                value_text: child.value.display_string(),
                                                property_type: child.property_type.clone(),
                                                y_position: child.y_position - self.scroll_offset,
                                                height: self.config.row_height,
                                                depth: child.depth,
                                                expanded: child.expanded,
                                                has_children: child.has_children(),
                                                hovered: child.hovered,
                                                selected: self.selected_row == Some(child.id),
                                                editing: child.editing,
                                                differs_from_default: child.differs_from_default(),
                                                mixed_value: child.mixed_value,
                                                read_only: child.metadata.read_only,
                                                tooltip: child.metadata.tooltip.clone(),
                                                units: child.metadata.units.clone(),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        render_rows
    }

    /// Returns the visible category headers for rendering.
    pub fn visible_categories(&self) -> Vec<CategoryRenderData> {
        self.categories
            .values()
            .filter(|c| c.has_visible_properties)
            .map(|c| CategoryRenderData {
                name: c.name.clone(),
                expanded: c.expanded,
                hovered: c.hovered,
                property_count: c.properties.len() as u32,
            })
            .collect()
    }

    /// Sets the editor to edit a new type.
    pub fn set_type(&mut self, type_name: &str) {
        self.type_name = type_name.to_string();
        self.title = format!("{} Properties", type_name);
    }

    /// Returns a summary string.
    pub fn summary(&self) -> String {
        format!(
            "{}: {} properties ({} visible) | {} categories | Undo: {}",
            self.type_name,
            self.property_count,
            self.visible_property_count,
            self.categories.len(),
            self.undo_redo.undo_depth(),
        )
    }
}

impl Default for PropertyEditor {
    fn default() -> Self {
        Self::new()
    }
}

/// Render data for a single property row.
#[derive(Debug, Clone)]
pub struct PropertyRowRenderData {
    pub id: PropId,
    pub label: String,
    pub value_text: String,
    pub property_type: PropertyType,
    pub y_position: f32,
    pub height: f32,
    pub depth: u32,
    pub expanded: bool,
    pub has_children: bool,
    pub hovered: bool,
    pub selected: bool,
    pub editing: bool,
    pub differs_from_default: bool,
    pub mixed_value: bool,
    pub read_only: bool,
    pub tooltip: Option<String>,
    pub units: Option<String>,
}

/// Render data for a category header.
#[derive(Debug, Clone)]
pub struct CategoryRenderData {
    pub name: String,
    pub expanded: bool,
    pub hovered: bool,
    pub property_count: u32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_value_display() {
        assert_eq!(PropertyValue::Bool(true).display_string(), "true");
        assert_eq!(PropertyValue::Int32(42).display_string(), "42");
        assert_eq!(
            PropertyValue::Float32(3.14).display_string(),
            format!("{:.3}", 3.14)
        );
        assert_eq!(
            PropertyValue::String("hello".to_string()).display_string(),
            "hello"
        );
    }

    #[test]
    fn test_property_value_equality() {
        assert!(PropertyValue::Int32(42).equals(&PropertyValue::Int32(42)));
        assert!(!PropertyValue::Int32(42).equals(&PropertyValue::Int32(43)));
        assert!(PropertyValue::Float32(1.0).equals(&PropertyValue::Float32(1.0)));
    }

    #[test]
    fn test_property_metadata_range() {
        let meta = PropertyMetadata::new("speed", "Speed").with_range(0.0, 100.0);
        assert_eq!(meta.clamp_value(50.0), 50.0);
        assert_eq!(meta.clamp_value(-10.0), 0.0);
        assert_eq!(meta.clamp_value(150.0), 100.0);
    }

    #[test]
    fn test_property_row_reset() {
        let mut row = PropertyRow::new(
            PropertyMetadata::new("test", "Test").with_default(PropertyValue::Float32(0.0)),
            PropertyType::Float32,
            PropertyValue::Float32(42.0),
        );
        assert!(row.differs_from_default());
        row.reset_to_default();
        assert!(!row.differs_from_default());
    }

    #[test]
    fn test_undo_redo_stack() {
        let mut stack = UndoRedoStack::new();
        let id = next_prop_id();
        stack.push(
            UndoEntry::new(
                id,
                "test",
                PropertyValue::Float32(0.0),
                PropertyValue::Float32(1.0),
            ),
            0,
        );
        assert!(stack.can_undo());
        assert!(!stack.can_redo());

        let entry = stack.undo().unwrap();
        assert_eq!(entry.field_name, "test");
        assert!(stack.can_redo());
        assert!(!stack.can_undo());

        let entry = stack.redo().unwrap();
        assert_eq!(entry.field_name, "test");
    }

    #[test]
    fn test_undo_grouping() {
        let mut stack = UndoRedoStack::new();
        let id = next_prop_id();
        // Rapid changes to the same property should group.
        stack.push(
            UndoEntry::new(
                id,
                "val",
                PropertyValue::Float32(0.0),
                PropertyValue::Float32(1.0),
            ),
            100,
        );
        stack.push(
            UndoEntry::new(
                id,
                "val",
                PropertyValue::Float32(1.0),
                PropertyValue::Float32(2.0),
            ),
            200, // within 500ms window
        );
        // Should be merged: undo should go back to 0.0.
        assert_eq!(stack.undo_depth(), 1);
        let entry = stack.undo().unwrap();
        if let PropertyValue::Float32(v) = entry.old_value {
            assert!((v - 0.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_property_filter() {
        let filter = PropertyFilter::new();
        let row = PropertyRow::new(
            PropertyMetadata::new("speed", "Movement Speed"),
            PropertyType::Float32,
            PropertyValue::Float32(10.0),
        );
        assert!(filter.matches(&row));

        let mut filter = PropertyFilter::new();
        filter.set_search("move");
        assert!(filter.matches(&row));

        filter.set_search("xyz");
        assert!(!filter.matches(&row));
    }

    #[test]
    fn test_property_editor_add_and_update() {
        let mut editor = PropertyEditor::new();
        editor.set_type("TestComponent");

        let id = editor.add_property(
            PropertyMetadata::new("health", "Health")
                .with_range(0.0, 100.0)
                .with_category("Stats"),
            PropertyType::Float32,
            PropertyValue::Float32(75.0),
        );

        editor.add_property(
            PropertyMetadata::new("name", "Name").with_category("General"),
            PropertyType::String,
            PropertyValue::String("Player".to_string()),
        );

        assert_eq!(editor.property_count, 2);
        assert_eq!(editor.categories.len(), 2);

        editor.update(0.016);
        assert_eq!(editor.visible_property_count, 2);
    }

    #[test]
    fn test_color_picker_hsv_roundtrip() {
        let mut picker = ColorPickerState::new();
        picker.color = [0.8, 0.2, 0.4, 1.0];
        let original = picker.color;
        picker.rgb_to_hsv();
        picker.hsv_to_rgb();
        assert!((picker.color[0] - original[0]).abs() < 0.01);
        assert!((picker.color[1] - original[1]).abs() < 0.01);
        assert!((picker.color[2] - original[2]).abs() < 0.01);
    }

    #[test]
    fn test_array_editor_add_remove() {
        let id = next_prop_id();
        let mut state = ArrayEditorState::new(id, PropertyType::Float32, 3);
        assert!(state.can_add_element());
        assert!(state.can_remove_element());

        state.min_elements = 3;
        assert!(!state.can_remove_element());
    }

    #[test]
    fn test_asset_ref_editor() {
        let id = next_prop_id();
        let mut state = AssetRefEditorState::new(id, "Texture");
        state.set_path("textures/brick.png");
        assert!(state.valid);
        assert_eq!(state.display_name, "brick.png");

        state.clear();
        assert!(!state.valid);
        assert_eq!(state.display_name, "None");
    }

    #[test]
    fn test_multi_object_state() {
        let mut multi = MultiObjectState::new();
        let mut values1 = HashMap::new();
        values1.insert("speed".to_string(), PropertyValue::Float32(10.0));
        values1.insert("name".to_string(), PropertyValue::String("A".to_string()));

        let mut values2 = HashMap::new();
        values2.insert("speed".to_string(), PropertyValue::Float32(10.0));
        values2.insert("name".to_string(), PropertyValue::String("B".to_string()));

        multi.add_object("Obj1", values1);
        multi.add_object("Obj2", values2);
        multi.compute_mixed();

        assert!(!multi.is_mixed("speed"));
        assert!(multi.is_mixed("name"));
    }

    #[test]
    fn test_show_condition() {
        let cond = ShowCondition::new("mode", PropertyValue::Int32(1));
        assert!(cond.evaluate(&PropertyValue::Int32(1)));
        assert!(!cond.evaluate(&PropertyValue::Int32(2)));
    }

    #[test]
    fn test_property_category() {
        let mut cat = PropertyCategory::new("Transform");
        assert!(cat.expanded);
        cat.toggle();
        assert!(!cat.expanded);
        cat.toggle();
        assert!(cat.expanded);
    }
}
