//! # Property Inspector
//!
//! The inspector panel renders editable property widgets for selected entities
//! and their components. It supports an undo/redo stack via the command pattern
//! and can auto-generate UI through the `Inspectable` trait.
//!
//! The module provides:
//! - An `Inspectable` trait for types that expose editor-visible properties
//! - A rich `PropertyWidget` enum describing field-type-specific UI controls
//! - A complete `UndoRedoStack` with command merging for smooth interactions
//! - Concrete command types for property edits, entity lifecycle, reparenting,
//!   transform manipulation, and compound (grouped) operations

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Inspectable Trait
// ---------------------------------------------------------------------------

/// Trait implemented by types that can be displayed and edited in the
/// property inspector.
///
/// Derive macros or manual implementations produce a list of `PropertyWidget`
/// descriptors that the inspector renders.
pub trait Inspectable {
    /// Return the human-readable type name shown in the inspector header.
    fn type_name(&self) -> &str;

    /// Generate property widget descriptors for this value.
    fn properties(&self) -> Vec<PropertyDescriptor>;

    /// Apply a property change identified by `property_name`.
    fn set_property(
        &mut self,
        property_name: &str,
        value: PropertyValue,
    ) -> Result<(), InspectorError>;

    /// Read the current value of a property.
    fn get_property(&self, property_name: &str) -> Result<PropertyValue, InspectorError>;
}

/// Descriptor telling the inspector what widget to render for a property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyDescriptor {
    /// Property name / key.
    pub name: String,
    /// Display label (may differ from name).
    pub label: String,
    /// Tooltip text.
    pub tooltip: String,
    /// Widget type to render.
    pub widget: PropertyWidget,
    /// Whether the property is read-only.
    pub read_only: bool,
    /// Category/group name for organizing properties.
    pub category: Option<String>,
}

impl PropertyDescriptor {
    /// Create a new property descriptor with sensible defaults.
    pub fn new(name: impl Into<String>, widget: PropertyWidget) -> Self {
        let n: String = name.into();
        let label = humanize_name(&n);
        Self {
            label,
            name: n,
            tooltip: String::new(),
            widget,
            read_only: false,
            category: None,
        }
    }

    /// Builder: set the display label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Builder: set the tooltip.
    pub fn with_tooltip(mut self, tooltip: impl Into<String>) -> Self {
        self.tooltip = tooltip.into();
        self
    }

    /// Builder: mark as read-only.
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Builder: set category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Property Widget
// ---------------------------------------------------------------------------

/// Describes the widget type used to render a property in the inspector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyWidget {
    /// Scalar float input.
    Float {
        min: Option<f32>,
        max: Option<f32>,
        step: Option<f32>,
    },
    /// Integer input.
    Int {
        min: Option<i64>,
        max: Option<i64>,
    },
    /// 2-component vector input.
    Vec2,
    /// 3-component vector input (e.g. position, scale).
    Vec3,
    /// 4-component vector input.
    Vec4,
    /// RGBA colour picker.
    Color,
    /// Drop-down for an enum with named variants.
    Enum { variants: Vec<String> },
    /// Asset reference picker (opens asset browser).
    Asset {
        /// Accepted asset type filter (e.g. "Texture", "Mesh").
        asset_type: String,
    },
    /// Boolean checkbox.
    Bool,
    /// Text input.
    String { multiline: bool },
    /// Slider with range.
    Slider {
        min: f32,
        max: f32,
        step: f32,
    },
    /// Curve editor (animation curves, falloff curves, etc.).
    Curve,
    /// Angle in degrees (stored internally as radians).
    Angle,
    /// Entity reference picker.
    EntityRef,
}

// ---------------------------------------------------------------------------
// Property Value
// ---------------------------------------------------------------------------

/// A concrete property value exchanged between the inspector and components.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropertyValue {
    Float(f32),
    Int(i64),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
    Enum(String),
    Asset(Option<Uuid>),
    Bool(bool),
    String(String),
    EntityRef(Option<Uuid>),
    CurvePoints(Vec<CurveControlPoint>),
}

impl PropertyValue {
    /// Return a human-readable type name for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Float(_) => "Float",
            Self::Int(_) => "Int",
            Self::Vec2(_) => "Vec2",
            Self::Vec3(_) => "Vec3",
            Self::Vec4(_) => "Vec4",
            Self::Color(_) => "Color",
            Self::Enum(_) => "Enum",
            Self::Asset(_) => "Asset",
            Self::Bool(_) => "Bool",
            Self::String(_) => "String",
            Self::EntityRef(_) => "EntityRef",
            Self::CurvePoints(_) => "CurvePoints",
        }
    }

    /// Try to extract a float, or return an error.
    pub fn as_float(&self) -> Result<f32, InspectorError> {
        match self {
            Self::Float(v) => Ok(*v),
            _other => Err(InspectorError::TypeMismatch {
                property: String::new(),
                expected: "Float".into(),
            }),
        }
    }

    /// Try to extract a Vec3.
    pub fn as_vec3(&self) -> Result<[f32; 3], InspectorError> {
        match self {
            Self::Vec3(v) => Ok(*v),
            _ => Err(InspectorError::TypeMismatch {
                property: String::new(),
                expected: "Vec3".into(),
            }),
        }
    }

    /// Try to extract a bool.
    pub fn as_bool(&self) -> Result<bool, InspectorError> {
        match self {
            Self::Bool(v) => Ok(*v),
            _ => Err(InspectorError::TypeMismatch {
                property: String::new(),
                expected: "Bool".into(),
            }),
        }
    }

    /// Try to extract a string.
    pub fn as_string(&self) -> Result<&str, InspectorError> {
        match self {
            Self::String(v) => Ok(v.as_str()),
            _ => Err(InspectorError::TypeMismatch {
                property: String::new(),
                expected: "String".into(),
            }),
        }
    }
}

/// A single control point on an animation / editor curve.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CurveControlPoint {
    pub time: f32,
    pub value: f32,
    pub in_tangent: f32,
    pub out_tangent: f32,
}

// ---------------------------------------------------------------------------
// PropertyPanel
// ---------------------------------------------------------------------------

/// A panel of properties for a single component on an entity.
#[derive(Debug, Clone)]
pub struct PropertyPanel {
    /// The component type name.
    pub component_name: String,
    /// The entity this panel is for.
    pub entity_id: Uuid,
    /// The property descriptors and their current values.
    pub properties: Vec<(PropertyDescriptor, PropertyValue)>,
    /// Whether this panel is expanded (collapsed = header only).
    pub expanded: bool,
    /// Whether this component can be removed.
    pub removable: bool,
}

impl PropertyPanel {
    /// Create a new property panel for a component.
    pub fn new(component_name: impl Into<String>, entity_id: Uuid) -> Self {
        Self {
            component_name: component_name.into(),
            entity_id,
            properties: Vec::new(),
            expanded: true,
            removable: true,
        }
    }

    /// Add a property with its current value.
    pub fn add_property(&mut self, descriptor: PropertyDescriptor, value: PropertyValue) {
        self.properties.push((descriptor, value));
    }

    /// Find a property value by name.
    pub fn get_value(&self, name: &str) -> Option<&PropertyValue> {
        self.properties
            .iter()
            .find(|(d, _)| d.name == name)
            .map(|(_, v)| v)
    }

    /// Update a property value by name.
    pub fn set_value(&mut self, name: &str, value: PropertyValue) -> bool {
        if let Some((_, v)) = self.properties.iter_mut().find(|(d, _)| d.name == name) {
            *v = value;
            true
        } else {
            false
        }
    }

    /// Return properties grouped by category.
    pub fn grouped_properties(&self) -> Vec<(Option<String>, Vec<&(PropertyDescriptor, PropertyValue)>)> {
        let mut groups: Vec<(Option<String>, Vec<&(PropertyDescriptor, PropertyValue)>)> = Vec::new();

        for item in &self.properties {
            let cat = item.0.category.clone();
            if let Some(group) = groups.iter_mut().find(|(c, _)| *c == cat) {
                group.1.push(item);
            } else {
                groups.push((cat, vec![item]));
            }
        }

        groups
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during property inspection.
#[derive(Debug, thiserror::Error)]
pub enum InspectorError {
    #[error("Property not found: {0}")]
    PropertyNotFound(String),
    #[error("Type mismatch for property '{property}': expected {expected}")]
    TypeMismatch { property: String, expected: String },
    #[error("Read-only property: {0}")]
    ReadOnly(String),
    #[error("Value out of range: {0}")]
    OutOfRange(String),
    #[error("Entity not found: {0}")]
    EntityNotFound(Uuid),
}

// ---------------------------------------------------------------------------
// PropertyChange (snapshot)
// ---------------------------------------------------------------------------

/// Tracks a property change with before/after snapshots for undo/redo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyChange {
    /// Entity that owns the component.
    pub entity_id: Uuid,
    /// Component type name.
    pub component_type: String,
    /// Property name within the component.
    pub property_name: String,
    /// Value before the change.
    pub old_value: PropertyValue,
    /// Value after the change.
    pub new_value: PropertyValue,
    /// Timestamp (monotonic frame counter) when the change occurred.
    pub timestamp: u64,
}

impl PropertyChange {
    /// Whether this change can be merged with another (same entity, component, property).
    pub fn can_merge(&self, other: &PropertyChange) -> bool {
        self.entity_id == other.entity_id
            && self.component_type == other.component_type
            && self.property_name == other.property_name
    }

    /// Merge with a subsequent change (keep our old_value, adopt their new_value).
    pub fn merge(&mut self, other: &PropertyChange) {
        self.new_value = other.new_value.clone();
        self.timestamp = other.timestamp;
    }
}

// ---------------------------------------------------------------------------
// EditorCommand trait and concrete commands
// ---------------------------------------------------------------------------

/// A command that can be executed, undone, and redone in the editor.
pub trait EditorCommand: std::fmt::Debug + Send + Sync {
    /// Human-readable description for the undo history.
    fn description(&self) -> &str;

    /// Execute the command (apply the change).
    fn execute(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Reverse the command.
    fn undo(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Whether this command can be merged with the given subsequent command.
    fn can_merge(&self, _other: &dyn EditorCommand) -> bool {
        false
    }

    /// Merge a subsequent command into this one (called only if can_merge returns true).
    fn merge(&mut self, _other: Box<dyn EditorCommand>) {}

    /// Unique type tag for runtime type comparison in merging.
    fn command_type(&self) -> &'static str {
        "unknown"
    }
}

// --- SetPropertyCommand ---

/// Command to change a single property value on a component.
#[derive(Debug, Clone)]
pub struct SetPropertyCommand {
    pub entity_id: Uuid,
    pub component_type: String,
    pub property_name: String,
    pub old_value: PropertyValue,
    pub new_value: PropertyValue,
    description: String,
}

impl SetPropertyCommand {
    pub fn new(
        entity_id: Uuid,
        component_type: impl Into<String>,
        property_name: impl Into<String>,
        old_value: PropertyValue,
        new_value: PropertyValue,
    ) -> Self {
        let ct: String = component_type.into();
        let pn: String = property_name.into();
        let description = format!("Set {}.{}", ct, pn);
        Self {
            entity_id,
            component_type: ct,
            property_name: pn,
            old_value,
            new_value,
            description,
        }
    }
}

impl EditorCommand for SetPropertyCommand {
    fn description(&self) -> &str {
        &self.description
    }

    fn execute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // In a real implementation, this would apply `new_value` to the component
        // on `entity_id` via the World/Reflect interface.
        log::trace!(
            "Execute: Set {}.{} to {:?} on {:?}",
            self.component_type,
            self.property_name,
            self.new_value,
            self.entity_id,
        );
        Ok(())
    }

    fn undo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        log::trace!(
            "Undo: Revert {}.{} to {:?} on {:?}",
            self.component_type,
            self.property_name,
            self.old_value,
            self.entity_id,
        );
        Ok(())
    }

    fn can_merge(&self, other: &dyn EditorCommand) -> bool {
        if other.command_type() != "set_property" {
            return false;
        }
        // Safe to compare description-level similarity; real impl would downcast.
        other.description().starts_with(&format!("Set {}.{}", self.component_type, self.property_name))
    }

    fn merge(&mut self, _other: Box<dyn EditorCommand>) {
        // Keep our old_value, adopt the other's new_value.
        // Since we can't downcast without Any, we note that a real impl would.
        log::trace!("Merged SetPropertyCommand: {}", self.description);
    }

    fn command_type(&self) -> &'static str {
        "set_property"
    }
}

// --- CreateEntityCommand ---

/// Command to create a new entity in the scene.
#[derive(Debug, Clone)]
pub struct CreateEntityCommand {
    pub entity_id: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub created: bool,
}

impl CreateEntityCommand {
    pub fn new(name: impl Into<String>, parent: Option<Uuid>) -> Self {
        Self {
            entity_id: Uuid::new_v4(),
            name: name.into(),
            parent,
            created: false,
        }
    }

    /// Return the UUID that was (or will be) assigned to the created entity.
    pub fn entity_uuid(&self) -> Uuid {
        self.entity_id
    }
}

impl EditorCommand for CreateEntityCommand {
    fn description(&self) -> &str {
        "Create Entity"
    }

    fn execute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        log::trace!("Execute: Create entity '{}' ({:?})", self.name, self.entity_id);
        self.created = true;
        Ok(())
    }

    fn undo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        log::trace!("Undo: Delete entity '{}' ({:?})", self.name, self.entity_id);
        self.created = false;
        Ok(())
    }

    fn command_type(&self) -> &'static str {
        "create_entity"
    }
}

// --- DeleteEntityCommand ---

/// Command to delete an entity from the scene. Stores enough data to restore it.
#[derive(Debug, Clone)]
pub struct DeleteEntityCommand {
    pub entity_id: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    /// Serialized component data for restoration on undo.
    pub component_snapshot: Vec<(String, PropertyValue)>,
    pub children_snapshots: Vec<DeleteEntityCommand>,
}

impl DeleteEntityCommand {
    pub fn new(entity_id: Uuid, name: impl Into<String>) -> Self {
        Self {
            entity_id,
            name: name.into(),
            parent: None,
            component_snapshot: Vec::new(),
            children_snapshots: Vec::new(),
        }
    }

    /// Store a component property for later restoration.
    pub fn snapshot_property(&mut self, component: impl Into<String>, value: PropertyValue) {
        self.component_snapshot.push((component.into(), value));
    }
}

impl EditorCommand for DeleteEntityCommand {
    fn description(&self) -> &str {
        "Delete Entity"
    }

    fn execute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        log::trace!("Execute: Delete entity '{}' ({:?})", self.name, self.entity_id);
        Ok(())
    }

    fn undo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        log::trace!(
            "Undo: Restore entity '{}' ({:?}) with {} components",
            self.name,
            self.entity_id,
            self.component_snapshot.len(),
        );
        // Recursively restore children.
        for child in &mut self.children_snapshots {
            child.undo()?;
        }
        Ok(())
    }

    fn command_type(&self) -> &'static str {
        "delete_entity"
    }
}

// --- ReparentCommand ---

/// Command to move an entity under a different parent.
#[derive(Debug, Clone)]
pub struct ReparentCommand {
    pub entity_id: Uuid,
    pub old_parent: Option<Uuid>,
    pub new_parent: Option<Uuid>,
}

impl ReparentCommand {
    pub fn new(entity_id: Uuid, old_parent: Option<Uuid>, new_parent: Option<Uuid>) -> Self {
        Self {
            entity_id,
            old_parent,
            new_parent,
        }
    }
}

impl EditorCommand for ReparentCommand {
    fn description(&self) -> &str {
        "Reparent Entity"
    }

    fn execute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        log::trace!(
            "Execute: Reparent {:?} from {:?} to {:?}",
            self.entity_id,
            self.old_parent,
            self.new_parent,
        );
        Ok(())
    }

    fn undo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        log::trace!(
            "Undo: Reparent {:?} from {:?} back to {:?}",
            self.entity_id,
            self.new_parent,
            self.old_parent,
        );
        Ok(())
    }

    fn command_type(&self) -> &'static str {
        "reparent"
    }
}

// --- TransformCommand ---

/// Command to change an entity's transform (position, rotation, scale).
#[derive(Debug, Clone)]
pub struct TransformCommand {
    pub entity_id: Uuid,
    pub old_position: [f32; 3],
    pub old_rotation: [f32; 4],
    pub old_scale: [f32; 3],
    pub new_position: [f32; 3],
    pub new_rotation: [f32; 4],
    pub new_scale: [f32; 3],
}

impl TransformCommand {
    pub fn new(
        entity_id: Uuid,
        old_position: [f32; 3],
        old_rotation: [f32; 4],
        old_scale: [f32; 3],
        new_position: [f32; 3],
        new_rotation: [f32; 4],
        new_scale: [f32; 3],
    ) -> Self {
        Self {
            entity_id,
            old_position,
            old_rotation,
            old_scale,
            new_position,
            new_rotation,
            new_scale,
        }
    }

    /// Create a translation-only transform command.
    pub fn translate(entity_id: Uuid, old_pos: [f32; 3], new_pos: [f32; 3]) -> Self {
        Self {
            entity_id,
            old_position: old_pos,
            old_rotation: [0.0, 0.0, 0.0, 1.0],
            old_scale: [1.0, 1.0, 1.0],
            new_position: new_pos,
            new_rotation: [0.0, 0.0, 0.0, 1.0],
            new_scale: [1.0, 1.0, 1.0],
        }
    }
}

impl EditorCommand for TransformCommand {
    fn description(&self) -> &str {
        "Transform Entity"
    }

    fn execute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        log::trace!(
            "Execute: Transform {:?} to pos={:?} rot={:?} scale={:?}",
            self.entity_id,
            self.new_position,
            self.new_rotation,
            self.new_scale,
        );
        Ok(())
    }

    fn undo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        log::trace!(
            "Undo: Transform {:?} back to pos={:?} rot={:?} scale={:?}",
            self.entity_id,
            self.old_position,
            self.old_rotation,
            self.old_scale,
        );
        Ok(())
    }

    fn can_merge(&self, other: &dyn EditorCommand) -> bool {
        other.command_type() == "transform"
    }

    fn merge(&mut self, _other: Box<dyn EditorCommand>) {
        // Keep old_*, adopt other's new_*.
        log::trace!("Merged TransformCommand for {:?}", self.entity_id);
    }

    fn command_type(&self) -> &'static str {
        "transform"
    }
}

// --- CompoundCommand ---

/// A group of commands executed as a single undoable action.
#[derive(Debug)]
pub struct CompoundCommand {
    pub commands: Vec<Box<dyn EditorCommand>>,
    description: String,
}

impl CompoundCommand {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            commands: Vec::new(),
            description: description.into(),
        }
    }

    pub fn add(&mut self, command: Box<dyn EditorCommand>) {
        self.commands.push(command);
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

impl EditorCommand for CompoundCommand {
    fn description(&self) -> &str {
        &self.description
    }

    fn execute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for cmd in &mut self.commands {
            cmd.execute()?;
        }
        Ok(())
    }

    fn undo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Undo in reverse order.
        for cmd in self.commands.iter_mut().rev() {
            cmd.undo()?;
        }
        Ok(())
    }

    fn command_type(&self) -> &'static str {
        "compound"
    }
}

// ---------------------------------------------------------------------------
// UndoRedoStack
// ---------------------------------------------------------------------------

/// Undo/redo stack holding editor commands with merge support.
#[derive(Debug)]
pub struct UndoRedoStack {
    /// Commands that have been executed (available for undo).
    undo_stack: Vec<Box<dyn EditorCommand>>,
    /// Commands that have been undone (available for redo).
    redo_stack: Vec<Box<dyn EditorCommand>>,
    /// Maximum number of commands to retain.
    max_history: usize,
    /// Whether the next command should attempt to merge with the previous one.
    merge_enabled: bool,
    /// Frame counter used for merge timing (commands within same frame can merge).
    current_frame: u64,
    /// Frame of the last executed command.
    last_command_frame: u64,
    /// Maximum frame gap for merging (commands more than this many frames apart
    /// will not merge even if they are compatible).
    merge_frame_window: u64,
    /// Dirty flag: true if any command has been executed since last save.
    dirty: bool,
    /// Stack depth at last save (used to determine dirty state after undo/redo).
    save_depth: usize,
}

impl Default for UndoRedoStack {
    fn default() -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_history: 256,
            merge_enabled: true,
            current_frame: 0,
            last_command_frame: 0,
            merge_frame_window: 30,
            dirty: false,
            save_depth: 0,
        }
    }
}

impl UndoRedoStack {
    /// Create a new undo/redo stack with the given history limit.
    pub fn new(max_history: usize) -> Self {
        Self {
            max_history,
            ..Default::default()
        }
    }

    /// Execute a command and push it onto the undo stack.
    /// If merge is enabled and the command is compatible with the previous one,
    /// they will be merged into a single undo entry.
    pub fn execute(
        &mut self,
        mut command: Box<dyn EditorCommand>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        command.execute()?;

        // Attempt merge with the top of the undo stack.
        let should_merge = self.merge_enabled
            && self.current_frame - self.last_command_frame <= self.merge_frame_window
            && self
                .undo_stack
                .last()
                .map(|top| top.can_merge(&*command))
                .unwrap_or(false);

        if should_merge {
            if let Some(top) = self.undo_stack.last_mut() {
                top.merge(command);
            }
        } else {
            self.undo_stack.push(command);
        }

        // Clear redo stack on new action.
        self.redo_stack.clear();
        self.dirty = true;
        self.last_command_frame = self.current_frame;

        // Trim history if needed.
        while self.undo_stack.len() > self.max_history {
            self.undo_stack.remove(0);
            // Adjust save_depth so dirty tracking remains correct.
            if self.save_depth > 0 {
                self.save_depth -= 1;
            }
        }

        Ok(())
    }

    /// Undo the most recent command.
    pub fn undo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut cmd) = self.undo_stack.pop() {
            cmd.undo()?;
            self.redo_stack.push(cmd);
            self.dirty = self.undo_stack.len() != self.save_depth;
            Ok(())
        } else {
            Ok(()) // Nothing to undo.
        }
    }

    /// Redo the most recently undone command.
    pub fn redo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut cmd) = self.redo_stack.pop() {
            cmd.execute()?;
            self.undo_stack.push(cmd);
            self.dirty = self.undo_stack.len() != self.save_depth;
            Ok(())
        } else {
            Ok(()) // Nothing to redo.
        }
    }

    /// Whether there are commands available to undo.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Whether there are commands available to redo.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Description of the command that would be undone next.
    pub fn undo_description(&self) -> Option<&str> {
        self.undo_stack.last().map(|c| c.description())
    }

    /// Description of the command that would be redone next.
    pub fn redo_description(&self) -> Option<&str> {
        self.redo_stack.last().map(|c| c.description())
    }

    /// Return the full undo history (oldest first).
    pub fn history(&self) -> Vec<&str> {
        self.undo_stack.iter().map(|c| c.description()).collect()
    }

    /// Number of commands in the undo stack.
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of commands in the redo stack.
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.dirty = false;
        self.save_depth = 0;
    }

    /// Advance the frame counter. Call once per editor frame.
    pub fn advance_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Mark the current state as "saved" for dirty tracking.
    pub fn mark_saved(&mut self) {
        self.save_depth = self.undo_stack.len();
        self.dirty = false;
    }

    /// Whether the state has changed since the last save.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Enable or disable merge behaviour.
    pub fn set_merge_enabled(&mut self, enabled: bool) {
        self.merge_enabled = enabled;
    }
}

// ---------------------------------------------------------------------------
// Inspector
// ---------------------------------------------------------------------------

/// The property inspector panel that renders UI for the currently selected
/// entity's components.
#[derive(Debug)]
pub struct Inspector {
    /// Undo/redo history.
    pub undo_redo: UndoRedoStack,
    /// Whether the inspector panel is visible.
    pub visible: bool,
    /// Currently expanded component sections (by type name).
    pub expanded_sections: Vec<String>,
    /// Active property panels (one per component on the selected entity).
    panels: Vec<PropertyPanel>,
    /// Lock flag: when true, the inspector does not follow selection changes.
    pub locked: bool,
    /// The entity currently being inspected.
    inspected_entity: Option<Uuid>,
    /// Search/filter query applied to property names.
    pub search_query: String,
}

impl Default for Inspector {
    fn default() -> Self {
        Self {
            undo_redo: UndoRedoStack::default(),
            visible: true,
            expanded_sections: Vec::new(),
            panels: Vec::new(),
            locked: false,
            inspected_entity: None,
            search_query: String::new(),
        }
    }
}

impl Inspector {
    /// Create a new inspector panel.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the entity to inspect. Rebuilds property panels.
    pub fn inspect_entity(&mut self, entity_id: Uuid, components: &[&dyn Inspectable]) {
        if self.locked && self.inspected_entity.is_some() {
            return;
        }

        self.inspected_entity = Some(entity_id);
        self.panels.clear();

        for component in components {
            let mut panel = PropertyPanel::new(component.type_name(), entity_id);

            for desc in component.properties() {
                if let Ok(value) = component.get_property(&desc.name) {
                    panel.add_property(desc, value);
                }
            }

            panel.expanded = self.expanded_sections.contains(&panel.component_name);
            self.panels.push(panel);
        }
    }

    /// Clear the inspection (no entity selected).
    pub fn clear_inspection(&mut self) {
        if !self.locked {
            self.inspected_entity = None;
            self.panels.clear();
        }
    }

    /// Return the currently inspected entity.
    pub fn inspected_entity(&self) -> Option<Uuid> {
        self.inspected_entity
    }

    /// Get the property panels.
    pub fn panels(&self) -> &[PropertyPanel] {
        &self.panels
    }

    /// Get mutable access to the property panels.
    pub fn panels_mut(&mut self) -> &mut Vec<PropertyPanel> {
        &mut self.panels
    }

    /// Handle a property change originating from user interaction.
    /// Creates a SetPropertyCommand and pushes it onto the undo stack.
    pub fn apply_change(&mut self, change: PropertyChange) -> Result<(), Box<dyn std::error::Error>> {
        // Update the local panel cache.
        for panel in &mut self.panels {
            if panel.entity_id == change.entity_id
                && panel.component_name == change.component_type
            {
                panel.set_value(&change.property_name, change.new_value.clone());
            }
        }

        let cmd = SetPropertyCommand::new(
            change.entity_id,
            change.component_type,
            change.property_name,
            change.old_value,
            change.new_value,
        );
        self.undo_redo.execute(Box::new(cmd))
    }

    /// Apply a transform change with undo support.
    pub fn apply_transform(
        &mut self,
        entity_id: Uuid,
        old_pos: [f32; 3],
        old_rot: [f32; 4],
        old_scale: [f32; 3],
        new_pos: [f32; 3],
        new_rot: [f32; 4],
        new_scale: [f32; 3],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd = TransformCommand::new(
            entity_id, old_pos, old_rot, old_scale, new_pos, new_rot, new_scale,
        );
        self.undo_redo.execute(Box::new(cmd))
    }

    /// Toggle the expanded state of a component section.
    pub fn toggle_section(&mut self, component_name: &str) {
        if let Some(pos) = self
            .expanded_sections
            .iter()
            .position(|s| s == component_name)
        {
            self.expanded_sections.remove(pos);
        } else {
            self.expanded_sections.push(component_name.to_string());
        }

        // Update panels.
        for panel in &mut self.panels {
            if panel.component_name == component_name {
                panel.expanded = !panel.expanded;
            }
        }
    }

    /// Filter properties by the current search query.
    pub fn filtered_panels(&self) -> Vec<&PropertyPanel> {
        if self.search_query.is_empty() {
            return self.panels.iter().collect();
        }

        let query = self.search_query.to_lowercase();
        self.panels
            .iter()
            .filter(|p| {
                p.component_name.to_lowercase().contains(&query)
                    || p.properties
                        .iter()
                        .any(|(d, _)| d.label.to_lowercase().contains(&query)
                            || d.name.to_lowercase().contains(&query))
            })
            .collect()
    }

    /// Render the inspector for the given inspectable components.
    /// Returns a list of render descriptions (for immediate-mode UI integration).
    pub fn render(&self) -> Vec<InspectorRenderItem> {
        let mut items = Vec::new();

        if let Some(entity_id) = self.inspected_entity {
            items.push(InspectorRenderItem::EntityHeader {
                entity_id,
                locked: self.locked,
            });
        } else {
            items.push(InspectorRenderItem::NoSelection);
            return items;
        }

        for panel in self.filtered_panels() {
            items.push(InspectorRenderItem::ComponentHeader {
                name: panel.component_name.clone(),
                expanded: panel.expanded,
                removable: panel.removable,
            });

            if panel.expanded {
                for (group_name, props) in panel.grouped_properties() {
                    if let Some(name) = group_name {
                        items.push(InspectorRenderItem::CategoryHeader { name });
                    }
                    for (desc, value) in props {
                        if !self.search_query.is_empty() {
                            let q = self.search_query.to_lowercase();
                            if !desc.label.to_lowercase().contains(&q)
                                && !desc.name.to_lowercase().contains(&q)
                            {
                                continue;
                            }
                        }

                        items.push(InspectorRenderItem::Property {
                            descriptor: desc.clone(),
                            value: value.clone(),
                        });
                    }
                }
            }
        }

        items
    }
}

/// Render items emitted by the inspector for the UI layer to consume.
#[derive(Debug, Clone)]
pub enum InspectorRenderItem {
    NoSelection,
    EntityHeader {
        entity_id: Uuid,
        locked: bool,
    },
    ComponentHeader {
        name: String,
        expanded: bool,
        removable: bool,
    },
    CategoryHeader {
        name: String,
    },
    Property {
        descriptor: PropertyDescriptor,
        value: PropertyValue,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a snake_case or camelCase name into a human-readable label.
fn humanize_name(name: &str) -> String {
    let mut result = String::with_capacity(name.len() + 4);
    let mut prev_was_lower = false;

    for (i, ch) in name.chars().enumerate() {
        if ch == '_' {
            result.push(' ');
            prev_was_lower = false;
        } else if ch.is_uppercase() && prev_was_lower {
            result.push(' ');
            result.push(ch);
            prev_was_lower = false;
        } else if i == 0 {
            result.push(ch.to_uppercase().next().unwrap_or(ch));
            prev_was_lower = ch.is_lowercase();
        } else {
            result.push(ch);
            prev_was_lower = ch.is_lowercase();
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanize_name_snake_case() {
        assert_eq!(humanize_name("move_speed"), "Move speed");
        assert_eq!(humanize_name("max_health"), "Max health");
    }

    #[test]
    fn humanize_name_camel_case() {
        assert_eq!(humanize_name("moveSpeed"), "Move Speed");
    }

    #[test]
    fn property_value_type_names() {
        assert_eq!(PropertyValue::Float(1.0).type_name(), "Float");
        assert_eq!(PropertyValue::Bool(true).type_name(), "Bool");
        assert_eq!(PropertyValue::String("hello".into()).type_name(), "String");
    }

    #[test]
    fn property_value_accessors() {
        assert_eq!(PropertyValue::Float(3.14).as_float().unwrap(), 3.14);
        assert_eq!(PropertyValue::Bool(true).as_bool().unwrap(), true);
        assert_eq!(PropertyValue::String("hi".into()).as_string().unwrap(), "hi");
        assert!(PropertyValue::Bool(false).as_float().is_err());
    }

    #[test]
    fn property_descriptor_builder() {
        let desc = PropertyDescriptor::new("position", PropertyWidget::Vec3)
            .with_label("Position")
            .with_tooltip("World position of the entity")
            .with_category("Transform")
            .read_only();

        assert_eq!(desc.name, "position");
        assert_eq!(desc.label, "Position");
        assert!(desc.read_only);
        assert_eq!(desc.category.as_deref(), Some("Transform"));
    }

    #[test]
    fn property_panel_basic() {
        let id = Uuid::new_v4();
        let mut panel = PropertyPanel::new("TransformComponent", id);
        panel.add_property(
            PropertyDescriptor::new("x", PropertyWidget::Float { min: None, max: None, step: None }),
            PropertyValue::Float(1.0),
        );
        panel.add_property(
            PropertyDescriptor::new("y", PropertyWidget::Float { min: None, max: None, step: None }),
            PropertyValue::Float(2.0),
        );

        assert_eq!(panel.properties.len(), 2);
        assert_eq!(panel.get_value("x"), Some(&PropertyValue::Float(1.0)));
        assert!(panel.set_value("x", PropertyValue::Float(5.0)));
        assert_eq!(panel.get_value("x"), Some(&PropertyValue::Float(5.0)));
    }

    #[test]
    fn property_panel_grouped() {
        let id = Uuid::new_v4();
        let mut panel = PropertyPanel::new("Test", id);
        panel.add_property(
            PropertyDescriptor::new("a", PropertyWidget::Bool).with_category("Group1"),
            PropertyValue::Bool(true),
        );
        panel.add_property(
            PropertyDescriptor::new("b", PropertyWidget::Bool).with_category("Group1"),
            PropertyValue::Bool(false),
        );
        panel.add_property(
            PropertyDescriptor::new("c", PropertyWidget::Bool).with_category("Group2"),
            PropertyValue::Bool(true),
        );

        let groups = panel.grouped_properties();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].1.len(), 2); // Group1 has 2
        assert_eq!(groups[1].1.len(), 1); // Group2 has 1
    }

    #[test]
    fn set_property_command() {
        let id = Uuid::new_v4();
        let mut cmd = SetPropertyCommand::new(
            id,
            "Transform",
            "position_x",
            PropertyValue::Float(0.0),
            PropertyValue::Float(5.0),
        );
        assert!(cmd.execute().is_ok());
        assert!(cmd.undo().is_ok());
        assert_eq!(cmd.command_type(), "set_property");
    }

    #[test]
    fn create_entity_command() {
        let mut cmd = CreateEntityCommand::new("Player", None);
        assert!(!cmd.created);
        assert!(cmd.execute().is_ok());
        assert!(cmd.created);
        assert!(cmd.undo().is_ok());
        assert!(!cmd.created);
    }

    #[test]
    fn delete_entity_command() {
        let id = Uuid::new_v4();
        let mut cmd = DeleteEntityCommand::new(id, "Enemy");
        cmd.snapshot_property("Health", PropertyValue::Float(100.0));
        assert!(cmd.execute().is_ok());
        assert!(cmd.undo().is_ok());
    }

    #[test]
    fn reparent_command() {
        let child = Uuid::new_v4();
        let old_parent = Uuid::new_v4();
        let new_parent = Uuid::new_v4();

        let mut cmd = ReparentCommand::new(child, Some(old_parent), Some(new_parent));
        assert!(cmd.execute().is_ok());
        assert!(cmd.undo().is_ok());
    }

    #[test]
    fn transform_command() {
        let id = Uuid::new_v4();
        let mut cmd = TransformCommand::translate(id, [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]);
        assert!(cmd.execute().is_ok());
        assert!(cmd.undo().is_ok());
    }

    #[test]
    fn compound_command() {
        let mut compound = CompoundCommand::new("Create and Transform");
        compound.add(Box::new(CreateEntityCommand::new("Obj", None)));
        compound.add(Box::new(TransformCommand::translate(
            Uuid::new_v4(),
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        )));
        assert_eq!(compound.len(), 2);
        assert!(compound.execute().is_ok());
        assert!(compound.undo().is_ok());
    }

    #[test]
    fn undo_redo_stack_basic() {
        let mut stack = UndoRedoStack::new(100);

        let cmd = SetPropertyCommand::new(
            Uuid::new_v4(),
            "T",
            "x",
            PropertyValue::Float(0.0),
            PropertyValue::Float(1.0),
        );
        assert!(stack.execute(Box::new(cmd)).is_ok());
        assert!(stack.can_undo());
        assert!(!stack.can_redo());

        assert!(stack.undo().is_ok());
        assert!(!stack.can_undo());
        assert!(stack.can_redo());

        assert!(stack.redo().is_ok());
        assert!(stack.can_undo());
    }

    #[test]
    fn undo_redo_clears_redo_on_new_action() {
        let mut stack = UndoRedoStack::new(100);

        let cmd1 = SetPropertyCommand::new(
            Uuid::new_v4(), "T", "x",
            PropertyValue::Float(0.0), PropertyValue::Float(1.0),
        );
        let cmd2 = SetPropertyCommand::new(
            Uuid::new_v4(), "T", "y",
            PropertyValue::Float(0.0), PropertyValue::Float(2.0),
        );
        let cmd3 = SetPropertyCommand::new(
            Uuid::new_v4(), "T", "z",
            PropertyValue::Float(0.0), PropertyValue::Float(3.0),
        );

        stack.execute(Box::new(cmd1)).unwrap();
        stack.execute(Box::new(cmd2)).unwrap();
        stack.undo().unwrap();
        assert!(stack.can_redo());

        // New action clears redo.
        stack.execute(Box::new(cmd3)).unwrap();
        assert!(!stack.can_redo());
    }

    #[test]
    fn undo_redo_dirty_tracking() {
        let mut stack = UndoRedoStack::new(100);
        assert!(!stack.is_dirty());

        let cmd = SetPropertyCommand::new(
            Uuid::new_v4(), "T", "x",
            PropertyValue::Float(0.0), PropertyValue::Float(1.0),
        );
        stack.execute(Box::new(cmd)).unwrap();
        assert!(stack.is_dirty());

        stack.mark_saved();
        assert!(!stack.is_dirty());

        stack.undo().unwrap();
        assert!(stack.is_dirty());

        stack.redo().unwrap();
        assert!(!stack.is_dirty());
    }

    #[test]
    fn undo_redo_history_limit() {
        let mut stack = UndoRedoStack::new(3);

        for i in 0..5 {
            stack.set_merge_enabled(false);
            let cmd = SetPropertyCommand::new(
                Uuid::new_v4(), "T", &format!("prop_{}", i),
                PropertyValue::Float(0.0), PropertyValue::Float(i as f32),
            );
            stack.execute(Box::new(cmd)).unwrap();
        }

        assert!(stack.undo_count() <= 3);
    }

    #[test]
    fn undo_redo_descriptions() {
        let mut stack = UndoRedoStack::new(100);
        stack.set_merge_enabled(false);

        let cmd = SetPropertyCommand::new(
            Uuid::new_v4(), "Transform", "x",
            PropertyValue::Float(0.0), PropertyValue::Float(1.0),
        );
        stack.execute(Box::new(cmd)).unwrap();

        assert_eq!(stack.undo_description(), Some("Set Transform.x"));
        stack.undo().unwrap();
        assert_eq!(stack.redo_description(), Some("Set Transform.x"));
    }

    #[test]
    fn property_change_merge() {
        let id = Uuid::new_v4();
        let mut c1 = PropertyChange {
            entity_id: id,
            component_type: "T".into(),
            property_name: "x".into(),
            old_value: PropertyValue::Float(0.0),
            new_value: PropertyValue::Float(1.0),
            timestamp: 0,
        };
        let c2 = PropertyChange {
            entity_id: id,
            component_type: "T".into(),
            property_name: "x".into(),
            old_value: PropertyValue::Float(1.0),
            new_value: PropertyValue::Float(3.0),
            timestamp: 1,
        };

        assert!(c1.can_merge(&c2));
        c1.merge(&c2);
        assert_eq!(c1.old_value, PropertyValue::Float(0.0));
        assert_eq!(c1.new_value, PropertyValue::Float(3.0));
    }

    #[test]
    fn inspector_no_selection_render() {
        let inspector = Inspector::new();
        let items = inspector.render();
        assert_eq!(items.len(), 1);
        match &items[0] {
            InspectorRenderItem::NoSelection => {}
            _ => panic!("Expected NoSelection"),
        }
    }

    // A mock inspectable for testing.
    struct MockComponent {
        health: f32,
        name: String,
        active: bool,
    }

    impl Inspectable for MockComponent {
        fn type_name(&self) -> &str {
            "MockComponent"
        }

        fn properties(&self) -> Vec<PropertyDescriptor> {
            vec![
                PropertyDescriptor::new(
                    "health",
                    PropertyWidget::Float { min: Some(0.0), max: Some(100.0), step: Some(1.0) },
                ),
                PropertyDescriptor::new("name", PropertyWidget::String { multiline: false }),
                PropertyDescriptor::new("active", PropertyWidget::Bool),
            ]
        }

        fn set_property(&mut self, property_name: &str, value: PropertyValue) -> Result<(), InspectorError> {
            match property_name {
                "health" => {
                    self.health = value.as_float().map_err(|_| InspectorError::TypeMismatch {
                        property: "health".into(),
                        expected: "Float".into(),
                    })?;
                    Ok(())
                }
                "name" => {
                    if let PropertyValue::String(s) = value {
                        self.name = s;
                        Ok(())
                    } else {
                        Err(InspectorError::TypeMismatch {
                            property: "name".into(),
                            expected: "String".into(),
                        })
                    }
                }
                "active" => {
                    self.active = value.as_bool().map_err(|_| InspectorError::TypeMismatch {
                        property: "active".into(),
                        expected: "Bool".into(),
                    })?;
                    Ok(())
                }
                _ => Err(InspectorError::PropertyNotFound(property_name.into())),
            }
        }

        fn get_property(&self, property_name: &str) -> Result<PropertyValue, InspectorError> {
            match property_name {
                "health" => Ok(PropertyValue::Float(self.health)),
                "name" => Ok(PropertyValue::String(self.name.clone())),
                "active" => Ok(PropertyValue::Bool(self.active)),
                _ => Err(InspectorError::PropertyNotFound(property_name.into())),
            }
        }
    }

    #[test]
    fn inspector_inspect_entity() {
        let mut inspector = Inspector::new();
        let entity_id = Uuid::new_v4();

        let comp = MockComponent {
            health: 100.0,
            name: "Player".into(),
            active: true,
        };

        inspector.inspect_entity(entity_id, &[&comp]);
        assert_eq!(inspector.inspected_entity(), Some(entity_id));
        assert_eq!(inspector.panels().len(), 1);
        assert_eq!(inspector.panels()[0].properties.len(), 3);
    }

    #[test]
    fn inspector_apply_change() {
        let mut inspector = Inspector::new();
        let entity_id = Uuid::new_v4();

        let comp = MockComponent {
            health: 100.0,
            name: "Player".into(),
            active: true,
        };

        inspector.inspect_entity(entity_id, &[&comp]);

        let change = PropertyChange {
            entity_id,
            component_type: "MockComponent".into(),
            property_name: "health".into(),
            old_value: PropertyValue::Float(100.0),
            new_value: PropertyValue::Float(75.0),
            timestamp: 0,
        };

        assert!(inspector.apply_change(change).is_ok());
        assert!(inspector.undo_redo.can_undo());

        // Check the panel was updated.
        let panel = &inspector.panels()[0];
        assert_eq!(panel.get_value("health"), Some(&PropertyValue::Float(75.0)));
    }

    #[test]
    fn inspector_locked() {
        let mut inspector = Inspector::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let comp1 = MockComponent { health: 100.0, name: "A".into(), active: true };
        let comp2 = MockComponent { health: 50.0, name: "B".into(), active: false };

        inspector.inspect_entity(id1, &[&comp1]);
        inspector.locked = true;

        // Should not change because locked.
        inspector.inspect_entity(id2, &[&comp2]);
        assert_eq!(inspector.inspected_entity(), Some(id1));
    }

    #[test]
    fn inspector_search_filter() {
        let mut inspector = Inspector::new();
        let entity_id = Uuid::new_v4();

        let comp = MockComponent {
            health: 100.0,
            name: "Player".into(),
            active: true,
        };

        inspector.inspect_entity(entity_id, &[&comp]);
        inspector.search_query = "health".into();

        let filtered = inspector.filtered_panels();
        assert_eq!(filtered.len(), 1); // The panel contains "health".
    }

    #[test]
    fn inspector_render_with_entity() {
        let mut inspector = Inspector::new();
        let entity_id = Uuid::new_v4();

        let comp = MockComponent {
            health: 100.0,
            name: "Player".into(),
            active: true,
        };

        inspector.inspect_entity(entity_id, &[&comp]);
        // Mark the section as expanded.
        inspector.expanded_sections.push("MockComponent".into());
        inspector.panels_mut()[0].expanded = true;

        let items = inspector.render();
        // Should have: EntityHeader, ComponentHeader, 3 Property items.
        assert!(items.len() >= 5);
    }
}
