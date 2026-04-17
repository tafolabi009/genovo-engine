//! Component serialization system: serialize/deserialize components by name,
//! component factories, component cloning, and component diffing.
//!
//! This module provides the bridge between the scene format (which works with
//! named component types and serialized data blobs) and the ECS (which works
//! with typed Rust structs). Key features:
//!
//! - **Name-based serialization** — serialize any registered component to/from
//!   a human-readable or binary format using its registered type name.
//! - **Component factories** — create default instances of components by name,
//!   used by the editor for "Add Component" workflows.
//! - **Component cloning** — deep-clone component data using the registered
//!   clone function, even without knowing the concrete type at compile time.
//! - **Component diffing** — compare two sets of component data to produce
//!   a minimal delta, used for undo/redo and networking.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Serialized component representation
// ---------------------------------------------------------------------------

/// A value that can be serialized/deserialized for component fields.
#[derive(Debug, Clone, PartialEq)]
pub enum SerializedFieldValue {
    /// Boolean value.
    Bool(bool),
    /// 32-bit integer.
    Int(i32),
    /// 64-bit integer.
    Long(i64),
    /// 32-bit float.
    Float(f32),
    /// 64-bit float.
    Double(f64),
    /// String value.
    String(String),
    /// Array of values.
    Array(Vec<SerializedFieldValue>),
    /// Map of named values (for nested structs).
    Map(HashMap<String, SerializedFieldValue>),
    /// Raw bytes (for opaque binary data).
    Bytes(Vec<u8>),
    /// Null / missing value.
    Null,
}

impl SerializedFieldValue {
    /// Try to extract as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as i32.
    pub fn as_int(&self) -> Option<i32> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as f32.
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Double(v) => Some(*v as f32),
            Self::Int(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to extract as f64.
    pub fn as_double(&self) -> Option<f64> {
        match self {
            Self::Double(v) => Some(*v),
            Self::Float(v) => Some(*v as f64),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to extract as string.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    /// Try to extract as array.
    pub fn as_array(&self) -> Option<&[SerializedFieldValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Try to extract as map.
    pub fn as_map(&self) -> Option<&HashMap<String, SerializedFieldValue>> {
        match self {
            Self::Map(v) => Some(v),
            _ => None,
        }
    }

    /// Check if this is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Get a human-readable type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Long(_) => "long",
            Self::Float(_) => "float",
            Self::Double(_) => "double",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::Map(_) => "map",
            Self::Bytes(_) => "bytes",
            Self::Null => "null",
        }
    }
}

impl fmt::Display for SerializedFieldValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{}", v),
            Self::Int(v) => write!(f, "{}", v),
            Self::Long(v) => write!(f, "{}L", v),
            Self::Float(v) => write!(f, "{:.4}", v),
            Self::Double(v) => write!(f, "{:.6}", v),
            Self::String(v) => write!(f, "\"{}\"", v),
            Self::Array(v) => write!(f, "[{} elements]", v.len()),
            Self::Map(v) => write!(f, "{{{} fields}}", v.len()),
            Self::Bytes(v) => write!(f, "<{} bytes>", v.len()),
            Self::Null => write!(f, "null"),
        }
    }
}

// ---------------------------------------------------------------------------
// Serialized component
// ---------------------------------------------------------------------------

/// A fully serialized component: type name + field data.
#[derive(Debug, Clone)]
pub struct SerializedComponentData {
    /// The registered type name of this component.
    pub type_name: String,
    /// Serialized field values.
    pub fields: HashMap<String, SerializedFieldValue>,
    /// Optional metadata (editor hints, etc.).
    pub metadata: HashMap<String, String>,
}

impl SerializedComponentData {
    /// Create a new serialized component.
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            fields: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set a field value.
    pub fn set_field(
        &mut self,
        name: impl Into<String>,
        value: SerializedFieldValue,
    ) -> &mut Self {
        self.fields.insert(name.into(), value);
        self
    }

    /// Get a field value.
    pub fn get_field(&self, name: &str) -> Option<&SerializedFieldValue> {
        self.fields.get(name)
    }

    /// Check if a field exists.
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Get the number of fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Set metadata.
    pub fn set_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> &mut Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Component serializer trait
// ---------------------------------------------------------------------------

/// Trait for types that can serialize/deserialize themselves.
pub trait ComponentSerializable: Send + Sync + 'static {
    /// Serialize this component to a field map.
    fn serialize(&self) -> HashMap<String, SerializedFieldValue>;

    /// Deserialize from a field map, returning a new instance.
    fn deserialize(fields: &HashMap<String, SerializedFieldValue>) -> Option<Self>
    where
        Self: Sized;

    /// Get the type name used for serialization.
    fn type_name() -> &'static str
    where
        Self: Sized;
}

// ---------------------------------------------------------------------------
// Component registration
// ---------------------------------------------------------------------------

/// Type-erased component operations for the serialization system.
pub struct ComponentOps {
    /// Component type name.
    pub type_name: String,
    /// Rust TypeId.
    pub rust_type_id: TypeId,
    /// Serialize function: takes Any reference, returns fields.
    pub serialize_fn:
        Box<dyn Fn(&dyn Any) -> HashMap<String, SerializedFieldValue> + Send + Sync>,
    /// Deserialize function: takes fields, returns boxed Any.
    pub deserialize_fn: Box<
        dyn Fn(&HashMap<String, SerializedFieldValue>) -> Option<Box<dyn Any + Send + Sync>>
            + Send
            + Sync,
    >,
    /// Factory function: creates a default instance.
    pub factory_fn: Option<Box<dyn Fn() -> Box<dyn Any + Send + Sync> + Send + Sync>>,
    /// Clone function: clones a boxed Any.
    pub clone_fn:
        Option<Box<dyn Fn(&dyn Any) -> Option<Box<dyn Any + Send + Sync>> + Send + Sync>>,
    /// Category for editor grouping.
    pub category: String,
    /// Documentation string.
    pub doc: String,
    /// Whether this component is visible in the editor.
    pub editor_visible: bool,
    /// Field metadata (name -> metadata).
    pub field_meta: HashMap<String, FieldMetadata>,
}

impl fmt::Debug for ComponentOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentOps")
            .field("type_name", &self.type_name)
            .field("category", &self.category)
            .field("has_factory", &self.factory_fn.is_some())
            .field("has_clone", &self.clone_fn.is_some())
            .field("editor_visible", &self.editor_visible)
            .finish()
    }
}

/// Metadata about a component field (for editor display).
#[derive(Debug, Clone)]
pub struct FieldMetadata {
    /// Field name.
    pub name: String,
    /// Display name (human-readable).
    pub display_name: String,
    /// Field type description.
    pub field_type: String,
    /// Optional tooltip/doc string.
    pub doc: String,
    /// Whether the field is read-only in the editor.
    pub read_only: bool,
    /// Optional minimum value (for numeric fields).
    pub min: Option<f64>,
    /// Optional maximum value.
    pub max: Option<f64>,
    /// Optional step size.
    pub step: Option<f64>,
    /// Optional default value.
    pub default_value: Option<SerializedFieldValue>,
    /// Whether to show a slider instead of a text field.
    pub use_slider: bool,
    /// Category within the component (for grouping fields).
    pub category: Option<String>,
}

impl FieldMetadata {
    /// Create basic metadata.
    pub fn new(
        name: impl Into<String>,
        field_type: impl Into<String>,
    ) -> Self {
        let name = name.into();
        Self {
            display_name: name.clone(),
            name,
            field_type: field_type.into(),
            doc: String::new(),
            read_only: false,
            min: None,
            max: None,
            step: None,
            default_value: None,
            use_slider: false,
            category: None,
        }
    }

    /// Set display name.
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = name.into();
        self
    }

    /// Set range.
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    /// Set step size.
    pub fn with_step(mut self, step: f64) -> Self {
        self.step = Some(step);
        self
    }

    /// Mark as read-only.
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Use slider widget.
    pub fn use_slider(mut self) -> Self {
        self.use_slider = true;
        self
    }
}

// ---------------------------------------------------------------------------
// Component serialization registry
// ---------------------------------------------------------------------------

/// Central registry for component serialization and factory operations.
pub struct ComponentSerializationRegistry {
    /// Registered component operations by type name.
    ops_by_name: HashMap<String, ComponentOps>,
    /// Lookup from Rust TypeId to type name.
    type_id_to_name: HashMap<TypeId, String>,
    /// Registration order.
    registration_order: Vec<String>,
}

impl ComponentSerializationRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            ops_by_name: HashMap::new(),
            type_id_to_name: HashMap::new(),
            registration_order: Vec::new(),
        }
    }

    /// Register component operations.
    pub fn register(&mut self, ops: ComponentOps) {
        let type_name = ops.type_name.clone();
        self.type_id_to_name
            .insert(ops.rust_type_id, type_name.clone());
        self.registration_order.push(type_name.clone());
        self.ops_by_name.insert(type_name, ops);
    }

    /// Register a component type that implements ComponentSerializable.
    pub fn register_serializable<T>(&mut self)
    where
        T: ComponentSerializable + Default + Clone + 'static,
    {
        let type_name = T::type_name().to_string();
        let ops = ComponentOps {
            type_name: type_name.clone(),
            rust_type_id: TypeId::of::<T>(),
            serialize_fn: Box::new(|any| {
                if let Some(comp) = any.downcast_ref::<T>() {
                    comp.serialize()
                } else {
                    HashMap::new()
                }
            }),
            deserialize_fn: Box::new(|fields| {
                T::deserialize(fields).map(|v| Box::new(v) as Box<dyn Any + Send + Sync>)
            }),
            factory_fn: Some(Box::new(|| {
                Box::new(T::default()) as Box<dyn Any + Send + Sync>
            })),
            clone_fn: Some(Box::new(|any| {
                any.downcast_ref::<T>()
                    .map(|v| Box::new(v.clone()) as Box<dyn Any + Send + Sync>)
            })),
            category: "General".to_string(),
            doc: String::new(),
            editor_visible: true,
            field_meta: HashMap::new(),
        };

        self.register(ops);
    }

    /// Serialize a component by type name.
    pub fn serialize(
        &self,
        type_name: &str,
        component: &dyn Any,
    ) -> Option<SerializedComponentData> {
        let ops = self.ops_by_name.get(type_name)?;
        let fields = (ops.serialize_fn)(component);
        let mut data = SerializedComponentData::new(type_name);
        data.fields = fields;
        Some(data)
    }

    /// Serialize a component by Rust type.
    pub fn serialize_typed<T: 'static>(
        &self,
        component: &T,
    ) -> Option<SerializedComponentData> {
        let type_name = self.type_id_to_name.get(&TypeId::of::<T>())?;
        let ops = self.ops_by_name.get(type_name)?;
        let fields = (ops.serialize_fn)(component as &dyn Any);
        let mut data = SerializedComponentData::new(type_name.clone());
        data.fields = fields;
        Some(data)
    }

    /// Deserialize a component from serialized data.
    pub fn deserialize(
        &self,
        data: &SerializedComponentData,
    ) -> Option<Box<dyn Any + Send + Sync>> {
        let ops = self.ops_by_name.get(&data.type_name)?;
        (ops.deserialize_fn)(&data.fields)
    }

    /// Create a default instance of a component by type name.
    pub fn create_default(
        &self,
        type_name: &str,
    ) -> Option<Box<dyn Any + Send + Sync>> {
        let ops = self.ops_by_name.get(type_name)?;
        let factory = ops.factory_fn.as_ref()?;
        Some(factory())
    }

    /// Clone a component by type name.
    pub fn clone_component(
        &self,
        type_name: &str,
        component: &dyn Any,
    ) -> Option<Box<dyn Any + Send + Sync>> {
        let ops = self.ops_by_name.get(type_name)?;
        let clone_fn = ops.clone_fn.as_ref()?;
        clone_fn(component)
    }

    /// Get the type name for a Rust type.
    pub fn type_name_of<T: 'static>(&self) -> Option<&str> {
        self.type_id_to_name
            .get(&TypeId::of::<T>())
            .map(|s| s.as_str())
    }

    /// Check if a type name is registered.
    pub fn is_registered(&self, type_name: &str) -> bool {
        self.ops_by_name.contains_key(type_name)
    }

    /// Get all registered type names.
    pub fn type_names(&self) -> &[String] {
        &self.registration_order
    }

    /// Get operations for a type name.
    pub fn get_ops(&self, type_name: &str) -> Option<&ComponentOps> {
        self.ops_by_name.get(type_name)
    }

    /// Get all type names in a specific category.
    pub fn types_in_category(&self, category: &str) -> Vec<&str> {
        self.ops_by_name
            .values()
            .filter(|ops| ops.category == category)
            .map(|ops| ops.type_name.as_str())
            .collect()
    }

    /// Get all categories.
    pub fn categories(&self) -> Vec<&str> {
        let mut cats: Vec<&str> = self
            .ops_by_name
            .values()
            .map(|ops| ops.category.as_str())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }

    /// Get all editor-visible type names.
    pub fn editor_visible_types(&self) -> Vec<&str> {
        self.ops_by_name
            .values()
            .filter(|ops| ops.editor_visible)
            .map(|ops| ops.type_name.as_str())
            .collect()
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.ops_by_name.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.ops_by_name.is_empty()
    }
}

impl Default for ComponentSerializationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Component diff
// ---------------------------------------------------------------------------

/// A diff between two sets of serialized component data.
#[derive(Debug, Clone)]
pub struct ComponentDiff {
    /// Components that were added.
    pub added: Vec<SerializedComponentData>,
    /// Components that were removed (by type name).
    pub removed: Vec<String>,
    /// Components that were modified.
    pub modified: Vec<ComponentFieldDiff>,
}

impl ComponentDiff {
    /// Create an empty diff.
    pub fn empty() -> Self {
        Self {
            added: Vec::new(),
            removed: Vec::new(),
            modified: Vec::new(),
        }
    }

    /// Check if the diff is empty.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }

    /// Total number of changes.
    pub fn change_count(&self) -> usize {
        self.added.len() + self.removed.len() + self.modified.len()
    }

    /// Compute the diff between two sets of component data.
    pub fn compute(
        old: &[SerializedComponentData],
        new: &[SerializedComponentData],
    ) -> Self {
        let old_map: HashMap<&str, &SerializedComponentData> =
            old.iter().map(|c| (c.type_name.as_str(), c)).collect();
        let new_map: HashMap<&str, &SerializedComponentData> =
            new.iter().map(|c| (c.type_name.as_str(), c)).collect();

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();

        // Find added and modified.
        for (name, new_comp) in &new_map {
            if let Some(old_comp) = old_map.get(name) {
                // Check for field changes.
                let field_diffs =
                    Self::diff_fields(&old_comp.fields, &new_comp.fields);
                if !field_diffs.is_empty() {
                    modified.push(ComponentFieldDiff {
                        type_name: name.to_string(),
                        changed_fields: field_diffs,
                    });
                }
            } else {
                added.push((*new_comp).clone());
            }
        }

        // Find removed.
        for name in old_map.keys() {
            if !new_map.contains_key(name) {
                removed.push(name.to_string());
            }
        }

        Self {
            added,
            removed,
            modified,
        }
    }

    /// Diff two field maps.
    fn diff_fields(
        old: &HashMap<String, SerializedFieldValue>,
        new: &HashMap<String, SerializedFieldValue>,
    ) -> Vec<FieldChange> {
        let mut changes = Vec::new();

        for (name, new_value) in new {
            if let Some(old_value) = old.get(name) {
                if old_value != new_value {
                    changes.push(FieldChange {
                        field_name: name.clone(),
                        old_value: Some(old_value.clone()),
                        new_value: Some(new_value.clone()),
                        kind: FieldChangeKind::Modified,
                    });
                }
            } else {
                changes.push(FieldChange {
                    field_name: name.clone(),
                    old_value: None,
                    new_value: Some(new_value.clone()),
                    kind: FieldChangeKind::Added,
                });
            }
        }

        for name in old.keys() {
            if !new.contains_key(name) {
                changes.push(FieldChange {
                    field_name: name.clone(),
                    old_value: old.get(name).cloned(),
                    new_value: None,
                    kind: FieldChangeKind::Removed,
                });
            }
        }

        changes
    }
}

/// Field-level diff for a single component.
#[derive(Debug, Clone)]
pub struct ComponentFieldDiff {
    /// Component type name.
    pub type_name: String,
    /// Changed fields.
    pub changed_fields: Vec<FieldChange>,
}

/// A change to a single field.
#[derive(Debug, Clone)]
pub struct FieldChange {
    /// Field name.
    pub field_name: String,
    /// Old value (None if added).
    pub old_value: Option<SerializedFieldValue>,
    /// New value (None if removed).
    pub new_value: Option<SerializedFieldValue>,
    /// Kind of change.
    pub kind: FieldChangeKind,
}

/// Kind of field change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldChangeKind {
    Added,
    Removed,
    Modified,
}

impl fmt::Display for FieldChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            FieldChangeKind::Added => {
                write!(f, "+ {}: {}", self.field_name, self.new_value.as_ref().unwrap())
            }
            FieldChangeKind::Removed => {
                write!(f, "- {}: {}", self.field_name, self.old_value.as_ref().unwrap())
            }
            FieldChangeKind::Modified => {
                write!(
                    f,
                    "~ {}: {} -> {}",
                    self.field_name,
                    self.old_value.as_ref().unwrap(),
                    self.new_value.as_ref().unwrap()
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialized_field_types() {
        let v = SerializedFieldValue::Float(3.14);
        assert_eq!(v.as_float(), Some(3.14));
        assert_eq!(v.type_name(), "float");

        let s = SerializedFieldValue::String("hello".to_string());
        assert_eq!(s.as_string(), Some("hello"));
    }

    #[test]
    fn component_diff_basic() {
        let old = vec![
            {
                let mut c = SerializedComponentData::new("Health");
                c.set_field("current", SerializedFieldValue::Float(100.0));
                c.set_field("max", SerializedFieldValue::Float(100.0));
                c
            },
        ];

        let new = vec![
            {
                let mut c = SerializedComponentData::new("Health");
                c.set_field("current", SerializedFieldValue::Float(50.0));
                c.set_field("max", SerializedFieldValue::Float(100.0));
                c
            },
            SerializedComponentData::new("Shield"),
        ];

        let diff = ComponentDiff::compute(&old, &new);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.modified.len(), 1);
        assert_eq!(diff.removed.len(), 0);
    }

    #[test]
    fn field_metadata() {
        let meta = FieldMetadata::new("health", "f32")
            .with_range(0.0, 100.0)
            .with_step(1.0)
            .use_slider();

        assert_eq!(meta.min, Some(0.0));
        assert_eq!(meta.max, Some(100.0));
        assert!(meta.use_slider);
    }

    #[test]
    fn serialized_component_data() {
        let mut comp = SerializedComponentData::new("Transform");
        comp.set_field("x", SerializedFieldValue::Float(1.0));
        comp.set_field("y", SerializedFieldValue::Float(2.0));
        comp.set_field("z", SerializedFieldValue::Float(3.0));

        assert_eq!(comp.field_count(), 3);
        assert!(comp.has_field("x"));
        assert_eq!(comp.get_field("x").unwrap().as_float(), Some(1.0));
    }
}
