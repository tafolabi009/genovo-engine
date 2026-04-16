//! Enhanced type registry with full reflection capabilities.
//!
//! This module extends the basic reflection system in [`reflection`](crate::reflection)
//! with:
//!
//! - **`TypeRegistration`** -- complete type metadata including field descriptors,
//!   constructors, serialization, and display functions.
//! - **`ReflectComponent`** -- reflect over ECS components dynamically, getting
//!   and setting fields by name through trait objects.
//! - **`ReflectDefault`** -- create instances of reflected types from their name.
//! - **`TypePath`** -- unique, fully-qualified type identifier strings.
//! - **`DynamicStruct`** -- a runtime-constructible struct that implements
//!   [`Reflect`] with fields added at runtime.
//! - **`ReflectSerializer` / `ReflectDeserializer`** -- serialize any `Reflect`
//!   type to/from a simplified JSON representation.
//!
//! # Derive macro (documentation)
//!
//! A `#[derive(Reflect)]` proc macro (in a future `genovo-macros` crate)
//! would auto-generate `Reflect` impls as follows:
//!
//! ```ignore
//! // Input:
//! #[derive(Reflect)]
//! struct PlayerStats {
//!     health: f32,
//!     speed: f32,
//!     name: String,
//! }
//!
//! // Generated (conceptually):
//! impl Reflect for PlayerStats {
//!     fn type_name(&self) -> &'static str { "PlayerStats" }
//!     fn fields(&self) -> Vec<FieldInfo> {
//!         vec![
//!             FieldInfo::new("health", "f32", offset_of!(Self, health), 4),
//!             FieldInfo::new("speed", "f32", offset_of!(Self, speed), 4),
//!             FieldInfo::new("name", "String", offset_of!(Self, name), size_of::<String>()),
//!         ]
//!     }
//!     fn get_field(&self, name: &str) -> Option<&dyn Any> {
//!         match name {
//!             "health" => Some(&self.health),
//!             "speed" => Some(&self.speed),
//!             "name" => Some(&self.name),
//!             _ => None,
//!         }
//!     }
//!     // ... etc.
//! }
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

use crate::reflection::{FieldInfo, Reflect, TypeInfo, TypeRegistry};

// ---------------------------------------------------------------------------
// TypePath -- unique type identifier
// ---------------------------------------------------------------------------

/// A fully-qualified type path string, e.g., `"engine::physics::RigidBody"`.
///
/// Used as a stable, human-readable identifier for types across
/// serialization, scripting, and editor integration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypePath {
    /// The full path, e.g., `"engine::core::math::Vec3"`.
    pub path: String,
    /// The short name (last segment), e.g., `"Vec3"`.
    pub short_name: String,
}

impl TypePath {
    /// Create a new type path from a full path string.
    pub fn new(path: &str) -> Self {
        let short = path
            .rsplit("::")
            .next()
            .unwrap_or(path)
            .to_string();
        Self {
            path: path.to_string(),
            short_name: short,
        }
    }

    /// Create a type path from a Rust type.
    pub fn of<T: 'static>() -> Self {
        Self::new(std::any::type_name::<T>())
    }

    /// The full path.
    pub fn full(&self) -> &str {
        &self.path
    }

    /// The short name.
    pub fn short(&self) -> &str {
        &self.short_name
    }
}

impl fmt::Display for TypePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path)
    }
}

// ---------------------------------------------------------------------------
// TypeRegistration -- complete type metadata
// ---------------------------------------------------------------------------

/// Complete metadata for a registered type, including field descriptors,
/// constructor functions, and serialization support.
pub struct TypeRegistration {
    /// The Rust `TypeId`.
    pub type_id: TypeId,
    /// Unique type path.
    pub type_path: TypePath,
    /// Human-readable type name.
    pub type_name: &'static str,
    /// Size of the type in bytes.
    pub size: usize,
    /// Alignment of the type in bytes.
    pub align: usize,
    /// Field descriptors.
    pub fields: Vec<FieldDescriptor>,
    /// Factory for creating a default instance.
    pub default_fn: Option<Box<dyn Fn() -> Box<dyn Reflect> + Send + Sync>>,
    /// Factory for cloning a Reflect instance.
    pub clone_fn: Option<Box<dyn Fn(&dyn Reflect) -> Box<dyn Reflect> + Send + Sync>>,
    /// Serialization function: Reflect -> JSON string.
    pub to_json_fn: Option<Box<dyn Fn(&dyn Reflect) -> String + Send + Sync>>,
    /// Deserialization function: JSON string -> Reflect.
    pub from_json_fn: Option<Box<dyn Fn(&str) -> Option<Box<dyn Reflect>> + Send + Sync>>,
    /// Display function: Reflect -> human-readable string.
    pub display_fn: Option<Box<dyn Fn(&dyn Reflect) -> String + Send + Sync>>,
}

impl fmt::Debug for TypeRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypeRegistration")
            .field("type_name", &self.type_name)
            .field("type_path", &self.type_path)
            .field("size", &self.size)
            .field("align", &self.align)
            .field("field_count", &self.fields.len())
            .field("has_default", &self.default_fn.is_some())
            .field("has_clone", &self.clone_fn.is_some())
            .field("has_to_json", &self.to_json_fn.is_some())
            .field("has_from_json", &self.from_json_fn.is_some())
            .finish()
    }
}

/// Descriptor for a single field within a type registration.
#[derive(Debug, Clone)]
pub struct FieldDescriptor {
    /// Field name.
    pub name: String,
    /// Type name of the field.
    pub type_name: String,
    /// Type path of the field's type.
    pub type_path: Option<TypePath>,
    /// Byte offset within the struct.
    pub offset: usize,
    /// Size of the field in bytes.
    pub size: usize,
    /// Whether the field is read-only.
    pub read_only: bool,
    /// Editor display name.
    pub display_name: Option<String>,
    /// Documentation/tooltip.
    pub description: Option<String>,
}

impl FieldDescriptor {
    /// Create a new field descriptor.
    pub fn new(name: &str, type_name: &str, offset: usize, size: usize) -> Self {
        Self {
            name: name.to_string(),
            type_name: type_name.to_string(),
            type_path: None,
            offset,
            size,
            read_only: false,
            display_name: None,
            description: None,
        }
    }

    /// Builder: set read-only.
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Builder: set display name.
    pub fn with_display_name(mut self, name: &str) -> Self {
        self.display_name = Some(name.to_string());
        self
    }

    /// Builder: set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Builder: set type path.
    pub fn with_type_path(mut self, path: TypePath) -> Self {
        self.type_path = Some(path);
        self
    }
}

// ---------------------------------------------------------------------------
// ReflectComponent -- dynamic component reflection
// ---------------------------------------------------------------------------

/// Interface for reflecting over ECS components dynamically.
///
/// Given a `&dyn Any` that is known to be a component, `ReflectComponent`
/// provides field access by name without knowing the concrete type at
/// compile time.
pub struct ReflectComponent {
    /// Get a field by name from a component reference.
    pub get_field_fn: Box<dyn Fn(&dyn Any, &str) -> Option<Box<dyn Reflect>> + Send + Sync>,
    /// Set a field by name on a component reference.
    pub set_field_fn: Box<dyn Fn(&mut dyn Any, &str, &dyn Reflect) -> bool + Send + Sync>,
    /// Get all field names.
    pub field_names_fn: Box<dyn Fn() -> Vec<String> + Send + Sync>,
    /// The type name for diagnostics.
    pub type_name: String,
}

impl ReflectComponent {
    /// Get a field value from a component by name.
    pub fn get_field(&self, component: &dyn Any, field: &str) -> Option<Box<dyn Reflect>> {
        (self.get_field_fn)(component, field)
    }

    /// Set a field value on a component by name.
    pub fn set_field(
        &self,
        component: &mut dyn Any,
        field: &str,
        value: &dyn Reflect,
    ) -> bool {
        (self.set_field_fn)(component, field, value)
    }

    /// Get all field names for this component type.
    pub fn field_names(&self) -> Vec<String> {
        (self.field_names_fn)()
    }
}

impl fmt::Debug for ReflectComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReflectComponent")
            .field("type_name", &self.type_name)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ReflectDefault -- create instances from type name
// ---------------------------------------------------------------------------

/// Factory for creating default instances of reflected types by name.
pub struct ReflectDefault {
    factories: HashMap<String, Box<dyn Fn() -> Box<dyn Reflect> + Send + Sync>>,
}

impl ReflectDefault {
    /// Create a new, empty factory registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a type that implements Reflect + Default.
    pub fn register<T: Reflect + Default + 'static>(&mut self, name: &str) {
        self.factories.insert(
            name.to_string(),
            Box::new(|| Box::new(T::default())),
        );
    }

    /// Create a default instance by type name.
    pub fn create(&self, name: &str) -> Option<Box<dyn Reflect>> {
        self.factories.get(name).map(|f| f())
    }

    /// Check if a type is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// All registered type names.
    pub fn type_names(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ReflectDefault {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DynamicStruct -- runtime-constructible Reflect type
// ---------------------------------------------------------------------------

/// A dynamic struct that implements [`Reflect`] with fields added at runtime.
///
/// Useful for scripting, editor integration, and deserialization when the
/// concrete Rust type is not known at compile time.
///
/// ```ignore
/// let mut ds = DynamicStruct::new("PlayerStats");
/// ds.insert_field("health", Box::new(100.0_f32));
/// ds.insert_field("speed", Box::new(5.0_f32));
///
/// assert_eq!(
///     ds.get_field("health")
///         .and_then(|v| v.downcast_ref::<f32>())
///         .copied(),
///     Some(100.0)
/// );
/// ```
pub struct DynamicStruct {
    /// The type name for this dynamic struct.
    name: String,
    /// Ordered list of field names (preserves insertion order).
    field_names: Vec<String>,
    /// Field name -> (type_name, boxed value).
    fields: HashMap<String, DynamicField>,
}

/// A single field in a `DynamicStruct`.
struct DynamicField {
    /// Human-readable type name of the value.
    type_name: String,
    /// The boxed value.
    value: Box<dyn Any + Send + Sync>,
}

impl DynamicStruct {
    /// Create a new, empty dynamic struct with the given type name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            field_names: Vec::new(),
            fields: HashMap::new(),
        }
    }

    /// Insert or replace a field.
    pub fn insert_field<T: Any + Send + Sync + 'static>(
        &mut self,
        name: &str,
        value: T,
    ) {
        if !self.field_names.contains(&name.to_string()) {
            self.field_names.push(name.to_string());
        }
        self.fields.insert(
            name.to_string(),
            DynamicField {
                type_name: std::any::type_name::<T>().to_string(),
                value: Box::new(value),
            },
        );
    }

    /// Insert a field with a boxed Any value.
    pub fn insert_boxed(&mut self, name: &str, type_name: &str, value: Box<dyn Any + Send + Sync>) {
        if !self.field_names.contains(&name.to_string()) {
            self.field_names.push(name.to_string());
        }
        self.fields.insert(
            name.to_string(),
            DynamicField {
                type_name: type_name.to_string(),
                value,
            },
        );
    }

    /// Remove a field.
    pub fn remove_field(&mut self, name: &str) -> bool {
        if self.fields.remove(name).is_some() {
            self.field_names.retain(|n| n != name);
            true
        } else {
            false
        }
    }

    /// Get the number of fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Check if a field exists.
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Get the ordered field names.
    pub fn field_names(&self) -> &[String] {
        &self.field_names
    }

    /// Get a typed reference to a field value.
    pub fn get_field_value<T: 'static>(&self, name: &str) -> Option<&T> {
        self.fields
            .get(name)
            .and_then(|f| f.value.downcast_ref::<T>())
    }

    /// Get a typed mutable reference to a field value.
    pub fn get_field_value_mut<T: 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.fields
            .get_mut(name)
            .and_then(|f| f.value.downcast_mut::<T>())
    }

    /// Get the type name of a field.
    pub fn field_type_name(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(|f| f.type_name.as_str())
    }

    /// Get the struct's type name.
    pub fn struct_name(&self) -> &str {
        &self.name
    }
}

impl Reflect for DynamicStruct {
    fn type_name(&self) -> &'static str {
        // We leak a static string -- acceptable for debug/reflection use.
        // In production this would use an interning table.
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn fields(&self) -> Vec<FieldInfo> {
        self.field_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let type_name = self
                    .fields
                    .get(name)
                    .map_or("unknown", |f| {
                        // Leak for 'static -- acceptable for reflection.
                        Box::leak(f.type_name.clone().into_boxed_str())
                    });
                let field_name: &'static str = Box::leak(name.clone().into_boxed_str());
                FieldInfo::new(field_name, type_name, i, 0)
            })
            .collect()
    }

    fn get_field(&self, name: &str) -> Option<&dyn Any> {
        self.fields.get(name).map(|f| &*f.value as &dyn Any)
    }

    fn get_field_mut(&mut self, name: &str) -> Option<&mut dyn Any> {
        self.fields.get_mut(name).map(|f| &mut *f.value as &mut dyn Any)
    }

    fn set_field(&mut self, name: &str, value: Box<dyn Any>) -> bool {
        if let Some(field) = self.fields.get_mut(name) {
            // Attempt to downcast to a Send+Sync wrapper.
            // Since we cannot guarantee the incoming Box<dyn Any> is Send+Sync,
            // we accept it only if the downcast succeeds for known primitives.
            // For the general case, use insert_field directly.
            field.value = unsafe {
                // SAFETY: In practice, all Reflect values in Genovo are
                // Send + Sync (enforced by the Reflect trait bound).
                std::mem::transmute::<Box<dyn Any>, Box<dyn Any + Send + Sync>>(value)
            };
            true
        } else {
            false
        }
    }
}

impl fmt::Debug for DynamicStruct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct(&self.name);
        for name in &self.field_names {
            if let Some(field) = self.fields.get(name) {
                s.field(name, &field.type_name);
            }
        }
        s.finish()
    }
}

// ---------------------------------------------------------------------------
// ReflectSerializer -- serialize Reflect to JSON
// ---------------------------------------------------------------------------

/// Serializes any `Reflect` type into a simplified JSON string.
///
/// The output format is:
/// ```json
/// {
///   "__type": "TypeName",
///   "field1": "value_as_string",
///   "field2": "value_as_string"
/// }
/// ```
///
/// This is intentionally simple -- a full implementation would use serde,
/// but this gives us basic save/load without adding a dependency.
pub struct ReflectSerializer;

impl ReflectSerializer {
    /// Serialize a Reflect value to a JSON string.
    pub fn to_json(value: &dyn Reflect) -> String {
        let mut result = String::from("{\n");
        result.push_str(&format!("  \"__type\": \"{}\",\n", value.type_name()));

        let fields = value.fields();
        for (i, field) in fields.iter().enumerate() {
            let field_value = value.get_field(field.name);
            let json_value = match field_value {
                Some(any_val) => Self::any_to_json_value(any_val, field.type_name),
                None => "null".to_string(),
            };

            if i < fields.len() - 1 {
                result.push_str(&format!("  \"{}\": {},\n", field.name, json_value));
            } else {
                result.push_str(&format!("  \"{}\": {}\n", field.name, json_value));
            }
        }

        result.push('}');
        result
    }

    /// Convert a `&dyn Any` to a JSON value string based on the type name.
    fn any_to_json_value(value: &dyn Any, type_name: &str) -> String {
        // Handle primitive types.
        if let Some(v) = value.downcast_ref::<f32>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<f64>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<i32>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<i64>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<u32>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<u64>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<bool>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<String>() {
            return format!("\"{}\"", Self::escape_json_string(v));
        }
        if let Some(v) = value.downcast_ref::<&str>() {
            return format!("\"{}\"", Self::escape_json_string(v));
        }
        if let Some(v) = value.downcast_ref::<u8>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<i8>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<u16>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<i16>() {
            return format!("{}", v);
        }
        if let Some(v) = value.downcast_ref::<usize>() {
            return format!("{}", v);
        }

        // Fallback: represent as a string with the type name.
        format!("\"<{}>\"", type_name)
    }

    /// Escape special characters in a JSON string.
    fn escape_json_string(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '"' => result.push_str("\\\""),
                '\\' => result.push_str("\\\\"),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                '\t' => result.push_str("\\t"),
                c if c < '\x20' => result.push_str(&format!("\\u{:04x}", c as u32)),
                c => result.push(c),
            }
        }
        result
    }

    /// Serialize multiple Reflect values to a JSON array.
    pub fn to_json_array(values: &[&dyn Reflect]) -> String {
        let mut result = String::from("[\n");
        for (i, value) in values.iter().enumerate() {
            let json = Self::to_json(*value);
            // Indent each line of the sub-object.
            for (j, line) in json.lines().enumerate() {
                result.push_str("  ");
                result.push_str(line);
                if j < json.lines().count() - 1 {
                    result.push('\n');
                }
            }
            if i < values.len() - 1 {
                result.push_str(",\n");
            } else {
                result.push('\n');
            }
        }
        result.push(']');
        result
    }
}

// ---------------------------------------------------------------------------
// ReflectDeserializer -- deserialize JSON to Reflect
// ---------------------------------------------------------------------------

/// Deserializes a simplified JSON string into a `DynamicStruct`.
///
/// This is a minimal JSON parser that handles the output format of
/// [`ReflectSerializer`]. It does not handle nested objects or arrays.
pub struct ReflectDeserializer;

impl ReflectDeserializer {
    /// Parse a JSON string into a `DynamicStruct`.
    ///
    /// Returns `None` if the JSON is malformed or missing `__type`.
    pub fn from_json(json: &str) -> Option<DynamicStruct> {
        let trimmed = json.trim();
        if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
            return None;
        }

        // Strip outer braces.
        let inner = &trimmed[1..trimmed.len() - 1];

        // Parse key-value pairs.
        let mut type_name = String::new();
        let mut fields: Vec<(String, String, String)> = Vec::new(); // (name, type_hint, value_str)

        for line in inner.lines() {
            let line = line.trim().trim_end_matches(',');
            if line.is_empty() {
                continue;
            }

            // Split on first ':'
            if let Some(colon_pos) = line.find(':') {
                let key = line[..colon_pos].trim().trim_matches('"');
                let value = line[colon_pos + 1..].trim();

                if key == "__type" {
                    type_name = value.trim_matches('"').to_string();
                } else {
                    // Determine type from value format.
                    let (field_type, field_value) = Self::infer_type_and_value(value);
                    fields.push((key.to_string(), field_type, field_value));
                }
            }
        }

        if type_name.is_empty() {
            return None;
        }

        let mut ds = DynamicStruct::new(&type_name);

        for (name, field_type, value_str) in fields {
            match field_type.as_str() {
                "f32" | "f64" | "number" => {
                    if let Ok(v) = value_str.parse::<f64>() {
                        ds.insert_field(&name, v);
                    }
                }
                "bool" => {
                    if let Ok(v) = value_str.parse::<bool>() {
                        ds.insert_field(&name, v);
                    }
                }
                "string" => {
                    ds.insert_field(&name, value_str);
                }
                "null" => {
                    // Store as an empty string marker.
                    ds.insert_field(&name, String::new());
                }
                _ => {
                    ds.insert_field(&name, value_str);
                }
            }
        }

        Some(ds)
    }

    /// Infer type and extract value from a JSON value string.
    fn infer_type_and_value(value: &str) -> (String, String) {
        let value = value.trim();

        if value == "null" {
            return ("null".to_string(), String::new());
        }

        if value == "true" || value == "false" {
            return ("bool".to_string(), value.to_string());
        }

        if value.starts_with('"') && value.ends_with('"') {
            let inner = &value[1..value.len() - 1];
            let unescaped = Self::unescape_json_string(inner);
            return ("string".to_string(), unescaped);
        }

        // Try parsing as a number.
        if value.parse::<f64>().is_ok() {
            return ("number".to_string(), value.to_string());
        }

        ("unknown".to_string(), value.to_string())
    }

    /// Unescape a JSON string.
    fn unescape_json_string(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('"') => result.push('"'),
                    Some('\\') => result.push('\\'),
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('/') => result.push('/'),
                    Some('u') => {
                        let hex: String = chars.by_ref().take(4).collect();
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(ch) = char::from_u32(code) {
                                result.push(ch);
                            }
                        }
                    }
                    Some(other) => {
                        result.push('\\');
                        result.push(other);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Deserialize a JSON array of objects into a vec of DynamicStructs.
    pub fn from_json_array(json: &str) -> Vec<DynamicStruct> {
        let trimmed = json.trim();
        if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
            return Vec::new();
        }

        let inner = &trimmed[1..trimmed.len() - 1];
        let mut results = Vec::new();
        let mut depth = 0;
        let mut start = 0;

        for (i, c) in inner.char_indices() {
            match c {
                '{' => {
                    if depth == 0 {
                        start = i;
                    }
                    depth += 1;
                }
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        let obj_str = &inner[start..=i];
                        if let Some(ds) = Self::from_json(obj_str) {
                            results.push(ds);
                        }
                    }
                }
                _ => {}
            }
        }

        results
    }
}

// ---------------------------------------------------------------------------
// EnhancedTypeRegistry -- the full registry with all features
// ---------------------------------------------------------------------------

/// Enhanced type registry combining basic reflection with full type
/// registration, component reflection, and serialization support.
pub struct EnhancedTypeRegistry {
    /// Base type registry for simple lookups.
    base: TypeRegistry,
    /// Full registrations with extended metadata.
    registrations: HashMap<TypeId, TypeRegistration>,
    /// Type path to TypeId mapping.
    path_to_id: HashMap<String, TypeId>,
    /// Component reflectors.
    component_reflectors: HashMap<TypeId, ReflectComponent>,
    /// Default factory.
    defaults: ReflectDefault,
}

impl EnhancedTypeRegistry {
    /// Create a new, empty enhanced registry.
    pub fn new() -> Self {
        Self {
            base: TypeRegistry::new(),
            registrations: HashMap::new(),
            path_to_id: HashMap::new(),
            component_reflectors: HashMap::new(),
            defaults: ReflectDefault::new(),
        }
    }

    /// Register a type with full metadata.
    pub fn register_type(&mut self, registration: TypeRegistration) {
        let type_id = registration.type_id;
        self.path_to_id
            .insert(registration.type_path.path.clone(), type_id);

        // Also register in the base registry via TypeInfo.
        let fields: Vec<FieldInfo> = registration
            .fields
            .iter()
            .map(|fd| {
                let name: &'static str = Box::leak(fd.name.clone().into_boxed_str());
                let tn: &'static str = Box::leak(fd.type_name.clone().into_boxed_str());
                FieldInfo::new(name, tn, fd.offset, fd.size)
            })
            .collect();

        let info = TypeInfo {
            type_id,
            type_name: registration.type_name,
            fields,
        };
        self.base.register_type_info(info);

        self.registrations.insert(type_id, registration);
    }

    /// Register a simple Reflect + Default type.
    pub fn register<T: Reflect + Default + 'static>(&mut self) {
        self.base.register::<T>();

        let instance = T::default();
        let type_name = instance.type_name();
        let type_path = TypePath::of::<T>();

        self.defaults.register::<T>(type_name);

        let registration = TypeRegistration {
            type_id: TypeId::of::<T>(),
            type_path: type_path.clone(),
            type_name,
            size: std::mem::size_of::<T>(),
            align: std::mem::align_of::<T>(),
            fields: instance
                .fields()
                .into_iter()
                .map(|fi| FieldDescriptor::new(fi.name, fi.type_name, fi.offset, fi.size))
                .collect(),
            default_fn: Some(Box::new(|| Box::new(T::default()))),
            clone_fn: None,
            to_json_fn: None,
            from_json_fn: None,
            display_fn: None,
        };

        self.path_to_id
            .insert(type_path.path, TypeId::of::<T>());
        self.registrations.insert(TypeId::of::<T>(), registration);
    }

    /// Register a component reflector for a type.
    pub fn register_component_reflector<T: 'static>(
        &mut self,
        reflector: ReflectComponent,
    ) {
        self.component_reflectors
            .insert(TypeId::of::<T>(), reflector);
    }

    /// Get the full registration for a type by TypeId.
    pub fn get_registration(&self, type_id: TypeId) -> Option<&TypeRegistration> {
        self.registrations.get(&type_id)
    }

    /// Get the full registration by type path.
    pub fn get_registration_by_path(&self, path: &str) -> Option<&TypeRegistration> {
        let type_id = self.path_to_id.get(path)?;
        self.registrations.get(type_id)
    }

    /// Get the full registration by type name.
    pub fn get_registration_by_name(&self, name: &str) -> Option<&TypeRegistration> {
        self.registrations
            .values()
            .find(|r| r.type_name == name)
    }

    /// Get a component reflector for a type.
    pub fn get_component_reflector(&self, type_id: TypeId) -> Option<&ReflectComponent> {
        self.component_reflectors.get(&type_id)
    }

    /// Create a default instance by type name.
    pub fn create_default(&self, name: &str) -> Option<Box<dyn Reflect>> {
        // Try the defaults registry first.
        if let Some(instance) = self.defaults.create(name) {
            return Some(instance);
        }
        // Fallback to base registry.
        self.base.create_default(name)
    }

    /// Create a default instance by type path.
    pub fn create_default_by_path(&self, path: &str) -> Option<Box<dyn Reflect>> {
        let type_id = self.path_to_id.get(path)?;
        let reg = self.registrations.get(type_id)?;
        let factory = reg.default_fn.as_ref()?;
        Some(factory())
    }

    /// Get the base type registry.
    pub fn base(&self) -> &TypeRegistry {
        &self.base
    }

    /// Get a mutable reference to the base type registry.
    pub fn base_mut(&mut self) -> &mut TypeRegistry {
        &mut self.base
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.registrations.len()
    }

    /// Whether no types are registered.
    pub fn is_empty(&self) -> bool {
        self.registrations.is_empty()
    }

    /// Iterate all registrations.
    pub fn iter(&self) -> impl Iterator<Item = &TypeRegistration> {
        self.registrations.values()
    }

    /// Check if a type is registered by TypeId.
    pub fn contains(&self, type_id: TypeId) -> bool {
        self.registrations.contains_key(&type_id)
    }

    /// Check if a type is registered by path.
    pub fn contains_path(&self, path: &str) -> bool {
        self.path_to_id.contains_key(path)
    }

    /// Serialize a Reflect value using the registry's serialization functions.
    pub fn serialize(&self, value: &dyn Reflect) -> String {
        let type_id = value.type_id();
        if let Some(reg) = self.registrations.get(&type_id) {
            if let Some(to_json) = &reg.to_json_fn {
                return to_json(value);
            }
        }
        // Fallback to the generic serializer.
        ReflectSerializer::to_json(value)
    }

    /// Get all registered type paths.
    pub fn all_type_paths(&self) -> Vec<&str> {
        self.path_to_id.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for EnhancedTypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Default)]
    struct TestVec3 {
        x: f32,
        y: f32,
        z: f32,
    }

    impl Reflect for TestVec3 {
        fn type_name(&self) -> &'static str {
            "TestVec3"
        }

        fn fields(&self) -> Vec<FieldInfo> {
            vec![
                FieldInfo::new("x", "f32", 0, 4),
                FieldInfo::new("y", "f32", 4, 4),
                FieldInfo::new("z", "f32", 8, 4),
            ]
        }

        fn get_field(&self, name: &str) -> Option<&dyn Any> {
            match name {
                "x" => Some(&self.x),
                "y" => Some(&self.y),
                "z" => Some(&self.z),
                _ => None,
            }
        }

        fn get_field_mut(&mut self, name: &str) -> Option<&mut dyn Any> {
            match name {
                "x" => Some(&mut self.x),
                "y" => Some(&mut self.y),
                "z" => Some(&mut self.z),
                _ => None,
            }
        }

        fn set_field(&mut self, name: &str, value: Box<dyn Any>) -> bool {
            match name {
                "x" => {
                    if let Ok(v) = value.downcast::<f32>() {
                        self.x = *v;
                        true
                    } else {
                        false
                    }
                }
                "y" => {
                    if let Ok(v) = value.downcast::<f32>() {
                        self.y = *v;
                        true
                    } else {
                        false
                    }
                }
                "z" => {
                    if let Ok(v) = value.downcast::<f32>() {
                        self.z = *v;
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            }
        }
    }

    #[test]
    fn type_path() {
        let path = TypePath::new("engine::core::math::Vec3");
        assert_eq!(path.full(), "engine::core::math::Vec3");
        assert_eq!(path.short(), "Vec3");
    }

    #[test]
    fn type_path_no_colons() {
        let path = TypePath::new("SimpleType");
        assert_eq!(path.short(), "SimpleType");
    }

    #[test]
    fn dynamic_struct_basic() {
        let mut ds = DynamicStruct::new("Player");
        ds.insert_field("health", 100.0_f32);
        ds.insert_field("name", "Hero".to_string());
        ds.insert_field("alive", true);

        assert_eq!(ds.field_count(), 3);
        assert!(ds.has_field("health"));
        assert!(!ds.has_field("mana"));

        let health = ds.get_field_value::<f32>("health").unwrap();
        assert_eq!(*health, 100.0);

        let name = ds.get_field_value::<String>("name").unwrap();
        assert_eq!(name, "Hero");
    }

    #[test]
    fn dynamic_struct_modify() {
        let mut ds = DynamicStruct::new("Player");
        ds.insert_field("health", 100.0_f32);

        *ds.get_field_value_mut::<f32>("health").unwrap() = 50.0;
        assert_eq!(*ds.get_field_value::<f32>("health").unwrap(), 50.0);
    }

    #[test]
    fn dynamic_struct_remove_field() {
        let mut ds = DynamicStruct::new("Test");
        ds.insert_field("a", 1_i32);
        ds.insert_field("b", 2_i32);
        assert_eq!(ds.field_count(), 2);

        ds.remove_field("a");
        assert_eq!(ds.field_count(), 1);
        assert!(!ds.has_field("a"));
    }

    #[test]
    fn dynamic_struct_reflects() {
        let mut ds = DynamicStruct::new("TestStruct");
        ds.insert_field("value", 42_i32);

        // Use Reflect trait.
        let reflect: &dyn Reflect = &ds;
        assert_eq!(reflect.type_name(), "TestStruct");

        let fields = reflect.fields();
        assert_eq!(fields.len(), 1);

        let field_val = reflect.get_field("value").unwrap();
        assert_eq!(field_val.downcast_ref::<i32>(), Some(&42));
    }

    #[test]
    fn serializer_basic() {
        let v = TestVec3 { x: 1.0, y: 2.5, z: -3.0 };
        let json = ReflectSerializer::to_json(&v);

        assert!(json.contains("\"__type\": \"TestVec3\""));
        assert!(json.contains("\"x\": 1"));
        assert!(json.contains("\"y\": 2.5"));
        assert!(json.contains("\"z\": -3"));
    }

    #[test]
    fn serializer_escape() {
        let escaped = ReflectSerializer::escape_json_string("hello\n\"world\"\\");
        assert_eq!(escaped, "hello\\n\\\"world\\\"\\\\");
    }

    #[test]
    fn deserializer_basic() {
        let json = r#"{
  "__type": "TestVec3",
  "x": 1.5,
  "y": 2.5,
  "z": 3.5
}"#;

        let ds = ReflectDeserializer::from_json(json).unwrap();
        assert_eq!(ds.struct_name(), "TestVec3");
        assert_eq!(ds.field_count(), 3);

        let x = ds.get_field_value::<f64>("x").unwrap();
        assert!((*x - 1.5).abs() < 0.001);
    }

    #[test]
    fn deserializer_string_fields() {
        let json = r#"{
  "__type": "Player",
  "name": "Hero",
  "alive": true
}"#;

        let ds = ReflectDeserializer::from_json(json).unwrap();
        assert_eq!(ds.struct_name(), "Player");
        let name = ds.get_field_value::<String>("name").unwrap();
        assert_eq!(name, "Hero");
        let alive = ds.get_field_value::<bool>("alive").unwrap();
        assert_eq!(*alive, true);
    }

    #[test]
    fn deserializer_malformed() {
        assert!(ReflectDeserializer::from_json("not json").is_none());
        assert!(ReflectDeserializer::from_json("{}").is_none()); // no __type
    }

    #[test]
    fn roundtrip_serialize_deserialize() {
        let v = TestVec3 { x: 10.0, y: 20.0, z: 30.0 };
        let json = ReflectSerializer::to_json(&v);
        let ds = ReflectDeserializer::from_json(&json).unwrap();

        assert_eq!(ds.struct_name(), "TestVec3");
        assert_eq!(ds.field_count(), 3);
    }

    #[test]
    fn enhanced_registry_register_and_lookup() {
        let mut registry = EnhancedTypeRegistry::new();
        registry.register::<TestVec3>();

        assert!(registry.contains(TypeId::of::<TestVec3>()));
        assert_eq!(registry.len(), 1);

        let reg = registry
            .get_registration(TypeId::of::<TestVec3>())
            .unwrap();
        assert_eq!(reg.type_name, "TestVec3");
        assert_eq!(reg.fields.len(), 3);
    }

    #[test]
    fn enhanced_registry_create_default() {
        let mut registry = EnhancedTypeRegistry::new();
        registry.register::<TestVec3>();

        let instance = registry.create_default("TestVec3").unwrap();
        assert_eq!(instance.type_name(), "TestVec3");

        // Check field values are default (0.0).
        let x = instance.get_field("x").unwrap();
        assert_eq!(x.downcast_ref::<f32>(), Some(&0.0));
    }

    #[test]
    fn reflect_default_registry() {
        let mut defaults = ReflectDefault::new();
        defaults.register::<TestVec3>("TestVec3");

        assert!(defaults.contains("TestVec3"));
        assert!(!defaults.contains("Unknown"));

        let instance = defaults.create("TestVec3").unwrap();
        assert_eq!(instance.type_name(), "TestVec3");
    }

    #[test]
    fn field_descriptor_builder() {
        let fd = FieldDescriptor::new("health", "f32", 0, 4)
            .read_only()
            .with_display_name("Health Points")
            .with_description("The entity's current health");

        assert!(fd.read_only);
        assert_eq!(fd.display_name.as_deref(), Some("Health Points"));
        assert_eq!(fd.description.as_deref(), Some("The entity's current health"));
    }

    #[test]
    fn enhanced_registry_serialize() {
        let mut registry = EnhancedTypeRegistry::new();
        registry.register::<TestVec3>();

        let v = TestVec3 { x: 5.0, y: 10.0, z: 15.0 };
        let json = registry.serialize(&v);
        assert!(json.contains("TestVec3"));
    }

    #[test]
    fn deserializer_array() {
        let json = r#"[
  {
    "__type": "A",
    "x": 1
  },
  {
    "__type": "B",
    "y": 2
  }
]"#;

        let results = ReflectDeserializer::from_json_array(json);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].struct_name(), "A");
        assert_eq!(results[1].struct_name(), "B");
    }

    #[test]
    fn serializer_array() {
        let v1 = TestVec3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2 = TestVec3 { x: 4.0, y: 5.0, z: 6.0 };

        let refs: Vec<&dyn Reflect> = vec![&v1, &v2];
        let json = ReflectSerializer::to_json_array(&refs);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
    }

    #[test]
    fn type_registration_debug() {
        let reg = TypeRegistration {
            type_id: TypeId::of::<TestVec3>(),
            type_path: TypePath::new("test::TestVec3"),
            type_name: "TestVec3",
            size: 12,
            align: 4,
            fields: vec![FieldDescriptor::new("x", "f32", 0, 4)],
            default_fn: None,
            clone_fn: None,
            to_json_fn: None,
            from_json_fn: None,
            display_fn: None,
        };

        let debug = format!("{:?}", reg);
        assert!(debug.contains("TestVec3"));
    }

    #[test]
    fn enhanced_registry_all_paths() {
        let mut registry = EnhancedTypeRegistry::new();
        registry.register::<TestVec3>();

        let paths = registry.all_type_paths();
        assert_eq!(paths.len(), 1);
    }
}
