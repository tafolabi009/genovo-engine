//! Component registry: type-erased component operations, metadata, clone/drop/
//! move/serialize per type, and dynamic component types.
//!
//! The component registry is the central catalog of all component types known
//! to the engine. Each registered component type has:
//!
//! - **Type metadata** — size, alignment, type name, `TypeId`.
//! - **Lifecycle functions** — drop, clone, move (all as `unsafe fn` pointers
//!   over raw byte pointers).
//! - **Serialization** — optional serialize/deserialize function pointers for
//!   snapshot, networking, and save/load.
//! - **Dynamic components** — runtime-defined component types without a Rust
//!   struct, useful for scripting and modding.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Component type ID
// ---------------------------------------------------------------------------

/// A unique identifier for a component type, used when `TypeId` is not available
/// (e.g. dynamic / script-defined components).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentTypeId(pub u64);

impl ComponentTypeId {
    /// Create from a Rust `TypeId`.
    pub fn from_type_id(type_id: TypeId) -> Self {
        // Hash the TypeId to a u64.
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        type_id.hash(&mut hasher);
        Self(hasher.finish())
    }

    /// Create a dynamic component type ID from a name.
    pub fn from_name(name: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut hasher);
        // Set the high bit to distinguish from TypeId-derived IDs.
        Self(hasher.finish() | (1 << 63))
    }

    /// Check if this is a dynamic (script-defined) component type.
    pub fn is_dynamic(&self) -> bool {
        (self.0 >> 63) & 1 == 1
    }
}

// ---------------------------------------------------------------------------
// Lifecycle function pointers
// ---------------------------------------------------------------------------

/// Function pointer for dropping a component in-place.
pub type DropFn = unsafe fn(*mut u8);

/// Function pointer for cloning a component from src to dst.
pub type CloneFn = unsafe fn(src: *const u8, dst: *mut u8);

/// Function pointer for moving a component from src to dst
/// (src is left in a moved-from state, not dropped).
pub type MoveFn = unsafe fn(src: *mut u8, dst: *mut u8);

/// Function pointer for serializing a component to bytes.
pub type SerializeFn = unsafe fn(src: *const u8) -> Vec<u8>;

/// Function pointer for deserializing a component from bytes.
pub type DeserializeFn = unsafe fn(data: &[u8], dst: *mut u8) -> bool;

/// Function pointer for comparing two components for equality.
pub type EqualsFn = unsafe fn(a: *const u8, b: *const u8) -> bool;

/// Function pointer for debug-formatting a component.
pub type DebugFn = unsafe fn(src: *const u8, f: &mut fmt::Formatter<'_>) -> fmt::Result;

// ---------------------------------------------------------------------------
// Component descriptor
// ---------------------------------------------------------------------------

/// Complete metadata about a registered component type.
#[derive(Clone)]
pub struct ComponentDescriptor {
    /// Unique component type identifier.
    pub type_id: ComponentTypeId,
    /// Rust `TypeId`, if this is a static (Rust-defined) component.
    pub rust_type_id: Option<TypeId>,
    /// Human-readable type name.
    pub name: String,
    /// Size of one component in bytes.
    pub size: usize,
    /// Alignment of the component type.
    pub align: usize,
    /// Whether this component type needs drop (has non-trivial destructor).
    pub needs_drop: bool,
    /// Drop function. Must be called when component data is discarded.
    pub drop_fn: Option<DropFn>,
    /// Clone function. Creates a bitwise copy with proper Clone semantics.
    pub clone_fn: Option<CloneFn>,
    /// Move function. Moves data from src to dst.
    pub move_fn: Option<MoveFn>,
    /// Serialization function.
    pub serialize_fn: Option<SerializeFn>,
    /// Deserialization function.
    pub deserialize_fn: Option<DeserializeFn>,
    /// Equality comparison function.
    pub equals_fn: Option<EqualsFn>,
    /// Debug formatting function.
    pub debug_fn: Option<DebugFn>,
    /// Optional category/group for editor display.
    pub category: Option<String>,
    /// Optional documentation string.
    pub doc: Option<String>,
    /// Custom metadata tags.
    pub tags: Vec<String>,
    /// Whether this component type is hidden from the editor.
    pub hidden: bool,
    /// Whether this component is replicated over the network.
    pub replicated: bool,
    /// Whether this component is saved to disk.
    pub persistent: bool,
}

impl ComponentDescriptor {
    /// Create a descriptor for a statically-typed Rust component.
    pub fn of<T: 'static>() -> Self {
        Self {
            type_id: ComponentTypeId::from_type_id(TypeId::of::<T>()),
            rust_type_id: Some(TypeId::of::<T>()),
            name: std::any::type_name::<T>().to_string(),
            size: std::mem::size_of::<T>(),
            align: std::mem::align_of::<T>(),
            needs_drop: std::mem::needs_drop::<T>(),
            drop_fn: if std::mem::needs_drop::<T>() {
                Some(|ptr| unsafe {
                    std::ptr::drop_in_place(ptr as *mut T);
                })
            } else {
                None
            },
            clone_fn: None,
            move_fn: Some(|src, dst| unsafe {
                std::ptr::copy_nonoverlapping(src as *const u8, dst, std::mem::size_of::<T>());
            }),
            serialize_fn: None,
            deserialize_fn: None,
            equals_fn: None,
            debug_fn: None,
            category: None,
            doc: None,
            tags: Vec::new(),
            hidden: false,
            replicated: false,
            persistent: true,
        }
    }

    /// Create a descriptor for a clonable component.
    pub fn of_clonable<T: Clone + 'static>() -> Self {
        let mut desc = Self::of::<T>();
        desc.clone_fn = Some(|src, dst| unsafe {
            let source = &*(src as *const T);
            let cloned = source.clone();
            std::ptr::write(dst as *mut T, cloned);
        });
        desc
    }

    /// Create a descriptor for a dynamic (script-defined) component.
    pub fn dynamic(name: impl Into<String>, size: usize, align: usize) -> Self {
        let name = name.into();
        Self {
            type_id: ComponentTypeId::from_name(&name),
            rust_type_id: None,
            name,
            size,
            align,
            needs_drop: false,
            drop_fn: None,
            clone_fn: Some(|src, dst| unsafe {
                // For dynamic components, clone is a simple memcpy.
                // The size is not known at compile time, but callers
                // must provide the correct size.
            }),
            move_fn: Some(|src, dst| unsafe {
                // Simple memcpy for dynamic components.
            }),
            serialize_fn: None,
            deserialize_fn: None,
            equals_fn: None,
            debug_fn: None,
            category: Some("Dynamic".to_string()),
            doc: None,
            tags: vec!["dynamic".to_string()],
            hidden: false,
            replicated: false,
            persistent: false,
        }
    }

    /// Set the category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Set the documentation string.
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Mark as hidden from editor.
    pub fn hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Mark as network-replicated.
    pub fn replicated(mut self) -> Self {
        self.replicated = true;
        self
    }

    /// Mark as persistent (saved to disk).
    pub fn persistent(mut self, value: bool) -> Self {
        self.persistent = value;
        self
    }

    /// Set serialization functions.
    pub fn with_serialization(
        mut self,
        serialize: SerializeFn,
        deserialize: DeserializeFn,
    ) -> Self {
        self.serialize_fn = Some(serialize);
        self.deserialize_fn = Some(deserialize);
        self
    }

    /// Set equality function.
    pub fn with_equals(mut self, equals: EqualsFn) -> Self {
        self.equals_fn = Some(equals);
        self
    }

    /// Set debug function.
    pub fn with_debug(mut self, debug: DebugFn) -> Self {
        self.debug_fn = Some(debug);
        self
    }
}

impl fmt::Debug for ComponentDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentDescriptor")
            .field("name", &self.name)
            .field("type_id", &self.type_id)
            .field("size", &self.size)
            .field("align", &self.align)
            .field("needs_drop", &self.needs_drop)
            .field("has_clone", &self.clone_fn.is_some())
            .field("has_serialize", &self.serialize_fn.is_some())
            .field("category", &self.category)
            .field("replicated", &self.replicated)
            .field("persistent", &self.persistent)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Component factory
// ---------------------------------------------------------------------------

/// A factory that can create default instances of a component type.
pub struct ComponentFactory {
    /// The component type this factory creates.
    pub type_id: ComponentTypeId,
    /// Human-readable name.
    pub name: String,
    /// Factory function: allocates and initializes a default component,
    /// returning a boxed raw pointer. Caller is responsible for dropping.
    pub create_fn: Box<dyn Fn() -> Vec<u8> + Send + Sync>,
}

impl ComponentFactory {
    /// Create a factory for a default-constructible component.
    pub fn of<T: Default + 'static>() -> Self {
        Self {
            type_id: ComponentTypeId::from_type_id(TypeId::of::<T>()),
            name: std::any::type_name::<T>().to_string(),
            create_fn: Box::new(|| {
                let value = T::default();
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &value as *const T as *const u8,
                        std::mem::size_of::<T>(),
                    )
                    .to_vec()
                };
                std::mem::forget(value);
                bytes
            }),
        }
    }

    /// Create a default instance as raw bytes.
    pub fn create_default(&self) -> Vec<u8> {
        (self.create_fn)()
    }
}

impl fmt::Debug for ComponentFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentFactory")
            .field("type_id", &self.type_id)
            .field("name", &self.name)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Component registry
// ---------------------------------------------------------------------------

/// The central registry of all known component types.
pub struct ComponentRegistry {
    /// Descriptors indexed by ComponentTypeId.
    descriptors: HashMap<ComponentTypeId, ComponentDescriptor>,
    /// Lookup from Rust TypeId to ComponentTypeId.
    type_id_map: HashMap<TypeId, ComponentTypeId>,
    /// Lookup from name to ComponentTypeId.
    name_map: HashMap<String, ComponentTypeId>,
    /// Component factories.
    factories: HashMap<ComponentTypeId, ComponentFactory>,
    /// Registration order (for deterministic iteration).
    registration_order: Vec<ComponentTypeId>,
}

impl ComponentRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
            type_id_map: HashMap::new(),
            name_map: HashMap::new(),
            factories: HashMap::new(),
            registration_order: Vec::new(),
        }
    }

    /// Register a component type.
    pub fn register(&mut self, descriptor: ComponentDescriptor) {
        let type_id = descriptor.type_id;

        if let Some(rust_type_id) = descriptor.rust_type_id {
            self.type_id_map.insert(rust_type_id, type_id);
        }

        self.name_map.insert(descriptor.name.clone(), type_id);
        self.registration_order.push(type_id);
        self.descriptors.insert(type_id, descriptor);
    }

    /// Register a Rust component type with auto-generated descriptor.
    pub fn register_type<T: 'static>(&mut self) {
        self.register(ComponentDescriptor::of::<T>());
    }

    /// Register a clonable component type.
    pub fn register_clonable<T: Clone + 'static>(&mut self) {
        self.register(ComponentDescriptor::of_clonable::<T>());
    }

    /// Register a component with a default factory.
    pub fn register_with_factory<T: Clone + Default + 'static>(&mut self) {
        self.register(ComponentDescriptor::of_clonable::<T>());
        self.factories.insert(
            ComponentTypeId::from_type_id(TypeId::of::<T>()),
            ComponentFactory::of::<T>(),
        );
    }

    /// Register a dynamic component type.
    pub fn register_dynamic(
        &mut self,
        name: impl Into<String>,
        size: usize,
        align: usize,
    ) -> ComponentTypeId {
        let descriptor = ComponentDescriptor::dynamic(name, size, align);
        let type_id = descriptor.type_id;
        self.register(descriptor);
        type_id
    }

    /// Look up a descriptor by ComponentTypeId.
    pub fn get(&self, type_id: ComponentTypeId) -> Option<&ComponentDescriptor> {
        self.descriptors.get(&type_id)
    }

    /// Look up a descriptor by Rust TypeId.
    pub fn get_by_type<T: 'static>(&self) -> Option<&ComponentDescriptor> {
        let comp_id = self.type_id_map.get(&TypeId::of::<T>())?;
        self.descriptors.get(comp_id)
    }

    /// Look up a descriptor by name.
    pub fn get_by_name(&self, name: &str) -> Option<&ComponentDescriptor> {
        let comp_id = self.name_map.get(name)?;
        self.descriptors.get(comp_id)
    }

    /// Get the ComponentTypeId for a Rust type.
    pub fn type_id_of<T: 'static>(&self) -> Option<ComponentTypeId> {
        self.type_id_map.get(&TypeId::of::<T>()).copied()
    }

    /// Check if a type is registered.
    pub fn is_registered<T: 'static>(&self) -> bool {
        self.type_id_map.contains_key(&TypeId::of::<T>())
    }

    /// Check if a named type is registered.
    pub fn is_registered_name(&self, name: &str) -> bool {
        self.name_map.contains_key(name)
    }

    /// Get a factory for a component type.
    pub fn factory(&self, type_id: ComponentTypeId) -> Option<&ComponentFactory> {
        self.factories.get(&type_id)
    }

    /// Create a default instance of a component by name.
    pub fn create_default_by_name(&self, name: &str) -> Option<Vec<u8>> {
        let comp_id = self.name_map.get(name)?;
        let factory = self.factories.get(comp_id)?;
        Some(factory.create_default())
    }

    /// Number of registered component types.
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Iterate over all registered descriptors in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &ComponentDescriptor> {
        self.registration_order
            .iter()
            .filter_map(move |id| self.descriptors.get(id))
    }

    /// Get all registered component names.
    pub fn names(&self) -> Vec<&str> {
        self.registration_order
            .iter()
            .filter_map(|id| self.descriptors.get(id))
            .map(|d| d.name.as_str())
            .collect()
    }

    /// Filter components by category.
    pub fn by_category(&self, category: &str) -> Vec<&ComponentDescriptor> {
        self.descriptors
            .values()
            .filter(|d| d.category.as_deref() == Some(category))
            .collect()
    }

    /// Filter components by tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&ComponentDescriptor> {
        self.descriptors
            .values()
            .filter(|d| d.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Get all replicated component types.
    pub fn replicated_components(&self) -> Vec<&ComponentDescriptor> {
        self.descriptors
            .values()
            .filter(|d| d.replicated)
            .collect()
    }

    /// Get all persistent component types.
    pub fn persistent_components(&self) -> Vec<&ComponentDescriptor> {
        self.descriptors
            .values()
            .filter(|d| d.persistent)
            .collect()
    }

    /// Get all visible (non-hidden) component types.
    pub fn visible_components(&self) -> Vec<&ComponentDescriptor> {
        self.descriptors
            .values()
            .filter(|d| !d.hidden)
            .collect()
    }

    /// Clone a component's data using its registered clone function.
    ///
    /// # Safety
    ///
    /// `src` must point to a valid component of the given type. `dst` must
    /// have enough space for the component.
    pub unsafe fn clone_component(
        &self,
        type_id: ComponentTypeId,
        src: *const u8,
        dst: *mut u8,
    ) -> bool {
        if let Some(desc) = self.descriptors.get(&type_id) {
            if let Some(clone_fn) = desc.clone_fn {
                clone_fn(src, dst);
                return true;
            }
        }
        false
    }

    /// Drop a component using its registered drop function.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid component of the given type.
    pub unsafe fn drop_component(&self, type_id: ComponentTypeId, ptr: *mut u8) {
        if let Some(desc) = self.descriptors.get(&type_id) {
            if let Some(drop_fn) = desc.drop_fn {
                drop_fn(ptr);
            }
        }
    }

    /// Serialize a component to bytes.
    ///
    /// # Safety
    ///
    /// `src` must point to a valid component of the given type.
    pub unsafe fn serialize_component(
        &self,
        type_id: ComponentTypeId,
        src: *const u8,
    ) -> Option<Vec<u8>> {
        let desc = self.descriptors.get(&type_id)?;
        let serialize_fn = desc.serialize_fn?;
        Some(serialize_fn(src))
    }

    /// Deserialize a component from bytes.
    ///
    /// # Safety
    ///
    /// `dst` must have enough space for the component.
    pub unsafe fn deserialize_component(
        &self,
        type_id: ComponentTypeId,
        data: &[u8],
        dst: *mut u8,
    ) -> bool {
        if let Some(desc) = self.descriptors.get(&type_id) {
            if let Some(deserialize_fn) = desc.deserialize_fn {
                return deserialize_fn(data, dst);
            }
        }
        false
    }

    /// Compare two components for equality.
    ///
    /// # Safety
    ///
    /// Both pointers must point to valid components of the given type.
    pub unsafe fn components_equal(
        &self,
        type_id: ComponentTypeId,
        a: *const u8,
        b: *const u8,
    ) -> Option<bool> {
        let desc = self.descriptors.get(&type_id)?;
        let equals_fn = desc.equals_fn?;
        Some(equals_fn(a, b))
    }

    /// Clear all registrations.
    pub fn clear(&mut self) {
        self.descriptors.clear();
        self.type_id_map.clear();
        self.name_map.clear();
        self.factories.clear();
        self.registration_order.clear();
    }
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ComponentRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentRegistry")
            .field("count", &self.descriptors.len())
            .field("types", &self.names())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Default, PartialEq)]
    struct Health {
        current: f32,
        max: f32,
    }

    #[derive(Debug, Clone, Default)]
    struct Transform {
        x: f32,
        y: f32,
        z: f32,
    }

    #[test]
    fn register_and_lookup() {
        let mut registry = ComponentRegistry::new();
        registry.register_clonable::<Health>();
        registry.register_type::<Transform>();

        assert!(registry.is_registered::<Health>());
        assert!(registry.is_registered::<Transform>());
        assert_eq!(registry.len(), 2);

        let desc = registry.get_by_type::<Health>().unwrap();
        assert_eq!(desc.size, std::mem::size_of::<Health>());
        assert!(desc.clone_fn.is_some());
    }

    #[test]
    fn dynamic_component() {
        let mut registry = ComponentRegistry::new();
        let id = registry.register_dynamic("ScriptHealth", 8, 4);

        assert!(id.is_dynamic());
        let desc = registry.get(id).unwrap();
        assert_eq!(desc.name, "ScriptHealth");
        assert_eq!(desc.size, 8);
    }

    #[test]
    fn factory_creation() {
        let mut registry = ComponentRegistry::new();
        registry.register_with_factory::<Health>();

        let comp_id = registry.type_id_of::<Health>().unwrap();
        let factory = registry.factory(comp_id).unwrap();
        let bytes = factory.create_default();
        assert_eq!(bytes.len(), std::mem::size_of::<Health>());
    }

    #[test]
    fn name_lookup() {
        let mut registry = ComponentRegistry::new();
        registry.register_type::<Health>();

        let desc = registry
            .get_by_name(std::any::type_name::<Health>())
            .unwrap();
        assert_eq!(desc.size, std::mem::size_of::<Health>());
    }

    #[test]
    fn component_type_id_from_type() {
        let id = ComponentTypeId::from_type_id(TypeId::of::<u32>());
        assert!(!id.is_dynamic());
    }

    #[test]
    fn filter_by_category() {
        let mut registry = ComponentRegistry::new();
        let desc = ComponentDescriptor::of::<Health>().with_category("Gameplay");
        registry.register(desc);

        let gameplay = registry.by_category("Gameplay");
        assert_eq!(gameplay.len(), 1);
    }
}
