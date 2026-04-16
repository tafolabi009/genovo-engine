//! Runtime type information and reflection.
//!
//! This module provides a lightweight reflection system that lets the engine
//! inspect and manipulate types at runtime. Primary use cases:
//!
//! - **Editor integration**: property grids, undo/redo, drag-and-drop
//! - **Serialization**: automatic (de)serialization of components and resources
//! - **Scripting**: exposing Rust types to Lua / WASM scripts
//!
//! # Derive macro
//!
//! A `#[derive(Reflect)]` proc macro is planned for the `genovo-macros` crate,
//! which will auto-generate the [`Reflect`] impl for any struct with public
//! fields.
//!
//! ```ignore
//! // Future usage (once genovo-macros is implemented):
//! #[derive(Reflect)]
//! struct PlayerStats {
//!     health: f32,
//!     speed: f32,
//! }
//! ```
//!
//! # Derive macro (planned)
//!
//! A `#[derive(Reflect)]` proc macro in the `genovo-macros` crate will
//! auto-generate the [`Reflect`] impl for structs with public fields,
//! eliminating boilerplate. Until then, implement the trait manually.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Reflect trait
// ---------------------------------------------------------------------------

/// Trait for types that expose their structure at runtime.
///
/// Implementors provide a type name, a list of fields, and accessors for
/// reading/writing individual fields by name. This is the foundation for
/// generic property editing, serialization, and scripting bridges.
pub trait Reflect: Any + Send + Sync {
    /// Returns the human-readable type name (e.g., `"Transform"`).
    fn type_name(&self) -> &'static str;

    /// Returns metadata about all reflected fields.
    fn fields(&self) -> Vec<FieldInfo>;

    /// Returns a shared reference to the field with the given name, or
    /// `None` if no such field exists (or the types do not match).
    fn get_field(&self, name: &str) -> Option<&dyn Any>;

    /// Returns an exclusive reference to the field with the given name, or
    /// `None` if no such field exists.
    fn get_field_mut(&mut self, name: &str) -> Option<&mut dyn Any>;

    /// Sets the field identified by `name` to `value`.
    ///
    /// Returns `true` on success, `false` if the field does not exist or the
    /// value type does not match the field type.
    fn set_field(&mut self, name: &str, value: Box<dyn Any>) -> bool;

    /// Returns the [`TypeInfo`] descriptor for this type.
    fn type_info(&self) -> TypeInfo {
        TypeInfo {
            type_id: self.type_id(),
            type_name: self.type_name(),
            fields: self.fields(),
        }
    }
}

// ---------------------------------------------------------------------------
// TypeInfo / FieldInfo
// ---------------------------------------------------------------------------

/// Static metadata for a reflected type.
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// The Rust [`TypeId`] of the reflected type.
    pub type_id: TypeId,
    /// Human-readable name.
    pub type_name: &'static str,
    /// Descriptions of all reflected fields.
    pub fields: Vec<FieldInfo>,
}

/// Metadata for a single reflected field.
#[derive(Debug, Clone)]
pub struct FieldInfo {
    /// Field name as it appears in Rust source.
    pub name: &'static str,
    /// Human-readable type name of the field.
    pub type_name: &'static str,
    /// Byte offset of the field from the start of the struct (for unsafe
    /// pointer access).
    pub offset: usize,
    /// Size of the field in bytes.
    pub size: usize,
    /// Whether the field is read-only (e.g., computed properties).
    pub read_only: bool,
    /// Optional display name for editor UIs.
    pub display_name: Option<&'static str>,
    /// Optional tooltip / documentation string.
    pub description: Option<&'static str>,
}

impl FieldInfo {
    /// Creates a new writable field descriptor.
    pub fn new(name: &'static str, type_name: &'static str, offset: usize, size: usize) -> Self {
        Self {
            name,
            type_name,
            offset,
            size,
            read_only: false,
            display_name: None,
            description: None,
        }
    }

    /// Builder method: marks the field as read-only.
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Builder method: sets an editor display name.
    pub fn with_display_name(mut self, display_name: &'static str) -> Self {
        self.display_name = Some(display_name);
        self
    }

    /// Builder method: sets a documentation/tooltip string.
    pub fn with_description(mut self, description: &'static str) -> Self {
        self.description = Some(description);
        self
    }
}

// ---------------------------------------------------------------------------
// TypeRegistry
// ---------------------------------------------------------------------------

/// Central registry mapping [`TypeId`]s to their [`TypeInfo`] descriptors.
///
/// Types must be registered before they can be queried at runtime. Registration
/// typically happens at application startup.
pub struct TypeRegistry {
    /// Map from Rust type id to type metadata.
    types: HashMap<TypeId, RegisteredType>,
    /// Map from type name to type id for name-based lookups.
    name_to_id: HashMap<&'static str, TypeId>,
}

/// Internal storage for a registered type.
struct RegisteredType {
    /// Static metadata.
    info: TypeInfo,
    /// Factory function that creates a default instance, if available.
    default_factory: Option<Box<dyn Fn() -> Box<dyn Reflect> + Send + Sync>>,
}

impl fmt::Debug for RegisteredType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredType")
            .field("info", &self.info)
            .field("has_default_factory", &self.default_factory.is_some())
            .finish()
    }
}

impl TypeRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            name_to_id: HashMap::new(),
        }
    }

    /// Registers a type `T` that implements [`Reflect`] and [`Default`].
    ///
    /// The default instance is used to extract [`TypeInfo`] and to provide
    /// a factory for creating new instances from scripts or editor actions.
    pub fn register<T: Reflect + Default + 'static>(&mut self) {
        let instance = T::default();
        let info = instance.type_info();
        let type_name = info.type_name;
        let type_id = TypeId::of::<T>();

        self.types.insert(
            type_id,
            RegisteredType {
                info,
                default_factory: Some(Box::new(|| Box::new(T::default()))),
            },
        );
        self.name_to_id.insert(type_name, type_id);
    }

    /// Registers a type with pre-built [`TypeInfo`] and no default factory.
    ///
    /// Use this for types that do not implement `Default`.
    pub fn register_type_info(&mut self, info: TypeInfo) {
        let type_id = info.type_id;
        let type_name = info.type_name;
        self.types.insert(
            type_id,
            RegisteredType {
                info,
                default_factory: None,
            },
        );
        self.name_to_id.insert(type_name, type_id);
    }

    /// Looks up type info by [`TypeId`].
    pub fn get_type_info(&self, type_id: TypeId) -> Option<&TypeInfo> {
        self.types.get(&type_id).map(|r| &r.info)
    }

    /// Looks up type info by name.
    pub fn get_type_info_by_name(&self, name: &str) -> Option<&TypeInfo> {
        let type_id = self.name_to_id.get(name)?;
        self.get_type_info(*type_id)
    }

    /// Creates a new default instance of the type identified by `name`, if
    /// a factory is registered.
    pub fn create_default(&self, name: &str) -> Option<Box<dyn Reflect>> {
        let type_id = self.name_to_id.get(name)?;
        let registered = self.types.get(type_id)?;
        let factory = registered.default_factory.as_ref()?;
        Some(factory())
    }

    /// Returns `true` if a type with the given id is registered.
    pub fn contains(&self, type_id: TypeId) -> bool {
        self.types.contains_key(&type_id)
    }

    /// Returns an iterator over all registered type infos.
    pub fn iter(&self) -> impl Iterator<Item = &TypeInfo> {
        self.types.values().map(|r| &r.info)
    }

    /// Returns the number of registered types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Returns `true` if no types are registered.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
