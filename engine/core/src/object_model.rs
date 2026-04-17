//! Engine object model for the Genovo engine.
//!
//! Provides a base object hierarchy with unique IDs, reference counting, weak
//! references, type hierarchy, a property system (named properties with
//! get/set/notify), an object factory, serialization, and cloning.
//!
//! This module serves as the foundation for editor-visible entities, allowing
//! runtime reflection, property editing, and save/load without coupling to the
//! ECS world directly.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

// ---------------------------------------------------------------------------
// ObjectId
// ---------------------------------------------------------------------------

/// Globally unique object identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectId(u64);

impl ObjectId {
    /// The null/invalid object ID.
    pub const NULL: ObjectId = ObjectId(0);

    /// Returns the raw numeric value.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Returns `true` if this is the null ID.
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Obj({})", self.0)
    }
}

static NEXT_OBJECT_ID: AtomicU64 = AtomicU64::new(1);

fn alloc_object_id() -> ObjectId {
    ObjectId(NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// PropertyValue
// ---------------------------------------------------------------------------

/// A dynamically-typed property value.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
    ObjectRef(ObjectId),
    Array(Vec<PropertyValue>),
    Map(Vec<(String, PropertyValue)>),
    Bytes(Vec<u8>),
    None,
}

impl PropertyValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PropertyValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            PropertyValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            PropertyValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            PropertyValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_vec3(&self) -> Option<[f32; 3]> {
        match self {
            PropertyValue::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_vec4(&self) -> Option<[f32; 4]> {
        match self {
            PropertyValue::Vec4(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_color(&self) -> Option<[f32; 4]> {
        match self {
            PropertyValue::Color(c) => Some(*c),
            _ => None,
        }
    }

    pub fn as_object_ref(&self) -> Option<ObjectId> {
        match self {
            PropertyValue::ObjectRef(id) => Some(*id),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            PropertyValue::Bool(_) => "bool",
            PropertyValue::Int(_) => "int",
            PropertyValue::Float(_) => "float",
            PropertyValue::String(_) => "string",
            PropertyValue::Vec2(_) => "vec2",
            PropertyValue::Vec3(_) => "vec3",
            PropertyValue::Vec4(_) => "vec4",
            PropertyValue::Color(_) => "color",
            PropertyValue::ObjectRef(_) => "object_ref",
            PropertyValue::Array(_) => "array",
            PropertyValue::Map(_) => "map",
            PropertyValue::Bytes(_) => "bytes",
            PropertyValue::None => "none",
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, PropertyValue::None)
    }
}

impl fmt::Display for PropertyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyValue::Bool(b) => write!(f, "{}", b),
            PropertyValue::Int(i) => write!(f, "{}", i),
            PropertyValue::Float(v) => write!(f, "{:.4}", v),
            PropertyValue::String(s) => write!(f, "\"{}\"", s),
            PropertyValue::Vec2(v) => write!(f, "({}, {})", v[0], v[1]),
            PropertyValue::Vec3(v) => write!(f, "({}, {}, {})", v[0], v[1], v[2]),
            PropertyValue::Vec4(v) => {
                write!(f, "({}, {}, {}, {})", v[0], v[1], v[2], v[3])
            }
            PropertyValue::Color(c) => {
                write!(f, "rgba({}, {}, {}, {})", c[0], c[1], c[2], c[3])
            }
            PropertyValue::ObjectRef(id) => write!(f, "ref({})", id),
            PropertyValue::Array(a) => write!(f, "[{} items]", a.len()),
            PropertyValue::Map(m) => write!(f, "{{{} entries}}", m.len()),
            PropertyValue::Bytes(b) => write!(f, "<{} bytes>", b.len()),
            PropertyValue::None => write!(f, "none"),
        }
    }
}

// ---------------------------------------------------------------------------
// PropertyDescriptor
// ---------------------------------------------------------------------------

/// Metadata describing a named property.
#[derive(Debug, Clone)]
pub struct PropertyDescriptor {
    /// Property name.
    pub name: String,
    /// Type name (for display/validation).
    pub type_name: String,
    /// Human-readable description.
    pub description: String,
    /// Category for grouping in UI.
    pub category: String,
    /// Whether the property is read-only.
    pub read_only: bool,
    /// Whether changes to this property should be propagated.
    pub notify: bool,
    /// Default value.
    pub default: PropertyValue,
    /// Minimum value (for numeric types).
    pub min: Option<f64>,
    /// Maximum value (for numeric types).
    pub max: Option<f64>,
    /// Step size for UI sliders.
    pub step: Option<f64>,
}

impl PropertyDescriptor {
    pub fn new(name: &str, type_name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_name: type_name.to_string(),
            description: String::new(),
            category: "General".to_string(),
            read_only: false,
            notify: true,
            default: PropertyValue::None,
            min: None,
            max: None,
            step: None,
        }
    }

    pub fn description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn category(mut self, cat: &str) -> Self {
        self.category = cat.to_string();
        self
    }

    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    pub fn no_notify(mut self) -> Self {
        self.notify = false;
        self
    }

    pub fn default_value(mut self, val: PropertyValue) -> Self {
        self.default = val;
        self
    }

    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    pub fn step(mut self, step: f64) -> Self {
        self.step = Some(step);
        self
    }
}

// ---------------------------------------------------------------------------
// PropertyChangeEvent
// ---------------------------------------------------------------------------

/// Event emitted when a property value changes.
#[derive(Debug, Clone)]
pub struct PropertyChangeEvent {
    /// ID of the object whose property changed.
    pub object_id: ObjectId,
    /// Name of the property.
    pub property_name: String,
    /// Old value.
    pub old_value: PropertyValue,
    /// New value.
    pub new_value: PropertyValue,
}

// ---------------------------------------------------------------------------
// TypeInfo
// ---------------------------------------------------------------------------

/// Runtime type information for engine objects.
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// Name of this type.
    pub name: String,
    /// Rust TypeId (if available).
    pub type_id: Option<TypeId>,
    /// Parent type name (for inheritance).
    pub parent: Option<String>,
    /// Property descriptors.
    pub properties: Vec<PropertyDescriptor>,
    /// Whether instances can be created via the factory.
    pub instantiable: bool,
    /// Category for the editor.
    pub editor_category: String,
    /// Icon name for the editor.
    pub editor_icon: String,
}

impl TypeInfo {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_id: None,
            parent: None,
            properties: Vec::new(),
            instantiable: true,
            editor_category: "General".to_string(),
            editor_icon: "object".to_string(),
        }
    }

    pub fn with_parent(mut self, parent: &str) -> Self {
        self.parent = Some(parent.to_string());
        self
    }

    pub fn with_property(mut self, prop: PropertyDescriptor) -> Self {
        self.properties.push(prop);
        self
    }

    pub fn with_rust_type<T: 'static>(mut self) -> Self {
        self.type_id = Some(TypeId::of::<T>());
        self
    }

    pub fn not_instantiable(mut self) -> Self {
        self.instantiable = false;
        self
    }

    pub fn editor_category(mut self, cat: &str) -> Self {
        self.editor_category = cat.to_string();
        self
    }

    pub fn editor_icon(mut self, icon: &str) -> Self {
        self.editor_icon = icon.to_string();
        self
    }

    /// Check if this type inherits from `ancestor` (walks the chain).
    pub fn inherits_from(&self, ancestor: &str, registry: &TypeRegistry) -> bool {
        let mut current = self.parent.as_deref();
        while let Some(parent_name) = current {
            if parent_name == ancestor {
                return true;
            }
            current = registry
                .get_type(parent_name)
                .and_then(|t| t.parent.as_deref());
        }
        false
    }

    /// Get all properties including inherited ones.
    pub fn all_properties(&self, registry: &TypeRegistry) -> Vec<PropertyDescriptor> {
        let mut props = Vec::new();

        // Collect inherited properties first.
        if let Some(ref parent_name) = self.parent {
            if let Some(parent_type) = registry.get_type(parent_name) {
                props.extend(parent_type.all_properties(registry));
            }
        }

        // Then own properties (may override inherited ones).
        for prop in &self.properties {
            // Remove any inherited property with the same name.
            props.retain(|p: &PropertyDescriptor| p.name != prop.name);
            props.push(prop.clone());
        }

        props
    }
}

// ---------------------------------------------------------------------------
// TypeRegistry
// ---------------------------------------------------------------------------

/// Registry of all known engine types with their metadata.
pub struct TypeRegistry {
    types: HashMap<String, TypeInfo>,
    type_id_map: HashMap<TypeId, String>,
}

impl TypeRegistry {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            type_id_map: HashMap::new(),
        }
    }

    /// Register a type.
    pub fn register(&mut self, info: TypeInfo) {
        let name = info.name.clone();
        if let Some(tid) = info.type_id {
            self.type_id_map.insert(tid, name.clone());
        }
        self.types.insert(name, info);
    }

    /// Get type info by name.
    pub fn get_type(&self, name: &str) -> Option<&TypeInfo> {
        self.types.get(name)
    }

    /// Get type info by Rust TypeId.
    pub fn get_by_type_id(&self, type_id: TypeId) -> Option<&TypeInfo> {
        self.type_id_map
            .get(&type_id)
            .and_then(|name| self.types.get(name))
    }

    /// Get all registered type names.
    pub fn type_names(&self) -> Vec<&str> {
        self.types.keys().map(|s| s.as_str()).collect()
    }

    /// Get all types that inherit from the given parent.
    pub fn subtypes_of(&self, parent: &str) -> Vec<&str> {
        self.types
            .iter()
            .filter(|(_, info)| info.inherits_from(parent, self))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get all instantiable types.
    pub fn instantiable_types(&self) -> Vec<&str> {
        self.types
            .iter()
            .filter(|(_, info)| info.instantiable)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Number of registered types.
    pub fn type_count(&self) -> usize {
        self.types.len()
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TypeRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypeRegistry")
            .field("types", &self.types.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// EngineObject
// ---------------------------------------------------------------------------

/// Base engine object with unique ID, properties, and type info.
pub struct EngineObject {
    /// Unique object ID.
    id: ObjectId,
    /// Type name.
    type_name: String,
    /// Display name.
    name: String,
    /// Property values.
    properties: HashMap<String, PropertyValue>,
    /// Parent object (weak reference to avoid cycles).
    parent: Option<ObjectId>,
    /// Child objects.
    children: Vec<ObjectId>,
    /// Whether this object is enabled.
    enabled: bool,
    /// Tags for categorization.
    tags: Vec<String>,
    /// Arbitrary metadata.
    metadata: HashMap<String, String>,
    /// Change listeners.
    change_listeners: Vec<Box<dyn Fn(&PropertyChangeEvent) + Send + Sync>>,
}

impl EngineObject {
    /// Create a new engine object.
    pub fn new(type_name: &str, name: &str) -> Self {
        Self {
            id: alloc_object_id(),
            type_name: type_name.to_string(),
            name: name.to_string(),
            properties: HashMap::new(),
            parent: None,
            children: Vec::new(),
            enabled: true,
            tags: Vec::new(),
            metadata: HashMap::new(),
            change_listeners: Vec::new(),
        }
    }

    /// Create with a specific ID (for deserialization).
    pub fn with_id(id: ObjectId, type_name: &str, name: &str) -> Self {
        Self {
            id,
            type_name: type_name.to_string(),
            name: name.to_string(),
            properties: HashMap::new(),
            parent: None,
            children: Vec::new(),
            enabled: true,
            tags: Vec::new(),
            metadata: HashMap::new(),
            change_listeners: Vec::new(),
        }
    }

    // -- Identification --

    /// Returns the unique object ID.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// Returns the type name.
    pub fn type_name(&self) -> &str {
        &self.type_name
    }

    /// Returns the display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the display name.
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    // -- Properties --

    /// Get a property value by name.
    pub fn get_property(&self, name: &str) -> Option<&PropertyValue> {
        self.properties.get(name)
    }

    /// Set a property value, notifying listeners.
    pub fn set_property(&mut self, name: &str, value: PropertyValue) {
        let old = self
            .properties
            .get(name)
            .cloned()
            .unwrap_or(PropertyValue::None);

        if old != value {
            let event = PropertyChangeEvent {
                object_id: self.id,
                property_name: name.to_string(),
                old_value: old,
                new_value: value.clone(),
            };

            self.properties.insert(name.to_string(), value);

            for listener in &self.change_listeners {
                listener(&event);
            }
        }
    }

    /// Set a property without notifying listeners.
    pub fn set_property_silent(&mut self, name: &str, value: PropertyValue) {
        self.properties.insert(name.to_string(), value);
    }

    /// Remove a property.
    pub fn remove_property(&mut self, name: &str) -> Option<PropertyValue> {
        self.properties.remove(name)
    }

    /// Get all property names.
    pub fn property_names(&self) -> Vec<&str> {
        self.properties.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the number of properties.
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Add a property change listener.
    pub fn on_property_changed<F>(&mut self, listener: F)
    where
        F: Fn(&PropertyChangeEvent) + Send + Sync + 'static,
    {
        self.change_listeners.push(Box::new(listener));
    }

    // -- Hierarchy --

    /// Get the parent object ID.
    pub fn parent(&self) -> Option<ObjectId> {
        self.parent
    }

    /// Set the parent object ID.
    pub fn set_parent(&mut self, parent: Option<ObjectId>) {
        self.parent = parent;
    }

    /// Get child object IDs.
    pub fn children(&self) -> &[ObjectId] {
        &self.children
    }

    /// Add a child.
    pub fn add_child(&mut self, child: ObjectId) {
        if !self.children.contains(&child) {
            self.children.push(child);
        }
    }

    /// Remove a child.
    pub fn remove_child(&mut self, child: ObjectId) -> bool {
        let before = self.children.len();
        self.children.retain(|c| *c != child);
        self.children.len() < before
    }

    /// Returns the number of children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    // -- State --

    /// Whether this object is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable this object.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    // -- Tags --

    /// Add a tag.
    pub fn add_tag(&mut self, tag: &str) {
        if !self.tags.iter().any(|t| t == tag) {
            self.tags.push(tag.to_string());
        }
    }

    /// Remove a tag.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        let before = self.tags.len();
        self.tags.retain(|t| t != tag);
        self.tags.len() < before
    }

    /// Check if this object has a tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Get all tags.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    // -- Metadata --

    /// Set a metadata value.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    // -- Serialization --

    /// Serialize this object to an ObjectData structure.
    pub fn serialize(&self) -> ObjectData {
        ObjectData {
            id: self.id,
            type_name: self.type_name.clone(),
            name: self.name.clone(),
            properties: self.properties.clone(),
            parent: self.parent,
            children: self.children.clone(),
            enabled: self.enabled,
            tags: self.tags.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Restore from serialized data.
    pub fn deserialize(data: &ObjectData) -> Self {
        Self {
            id: data.id,
            type_name: data.type_name.clone(),
            name: data.name.clone(),
            properties: data.properties.clone(),
            parent: data.parent,
            children: data.children.clone(),
            enabled: data.enabled,
            tags: data.tags.clone(),
            metadata: data.metadata.clone(),
            change_listeners: Vec::new(),
        }
    }

    /// Deep clone this object (assigns a new ID).
    pub fn deep_clone(&self) -> Self {
        let mut cloned = Self {
            id: alloc_object_id(),
            type_name: self.type_name.clone(),
            name: format!("{} (Clone)", self.name),
            properties: self.properties.clone(),
            parent: None,
            children: Vec::new(),
            enabled: self.enabled,
            tags: self.tags.clone(),
            metadata: self.metadata.clone(),
            change_listeners: Vec::new(),
        };
        cloned
    }
}

impl fmt::Debug for EngineObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EngineObject")
            .field("id", &self.id)
            .field("type", &self.type_name)
            .field("name", &self.name)
            .field("enabled", &self.enabled)
            .field("properties", &self.properties.len())
            .field("children", &self.children.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ObjectData (serialization format)
// ---------------------------------------------------------------------------

/// Serialized representation of an engine object.
#[derive(Debug, Clone)]
pub struct ObjectData {
    pub id: ObjectId,
    pub type_name: String,
    pub name: String,
    pub properties: HashMap<String, PropertyValue>,
    pub parent: Option<ObjectId>,
    pub children: Vec<ObjectId>,
    pub enabled: bool,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// ObjectRef (reference-counted handle)
// ---------------------------------------------------------------------------

/// A strong reference to an engine object.
#[derive(Clone)]
pub struct ObjectRef {
    inner: Arc<Mutex<EngineObject>>,
}

impl ObjectRef {
    /// Create a new object ref.
    pub fn new(object: EngineObject) -> Self {
        Self {
            inner: Arc::new(Mutex::new(object)),
        }
    }

    /// Get the object ID.
    pub fn id(&self) -> ObjectId {
        self.inner.lock().unwrap().id()
    }

    /// Create a weak reference.
    pub fn downgrade(&self) -> WeakObjectRef {
        WeakObjectRef {
            inner: Arc::downgrade(&self.inner),
        }
    }

    /// Access the object immutably.
    pub fn with<R, F: FnOnce(&EngineObject) -> R>(&self, f: F) -> R {
        let obj = self.inner.lock().unwrap();
        f(&obj)
    }

    /// Access the object mutably.
    pub fn with_mut<R, F: FnOnce(&mut EngineObject) -> R>(&self, f: F) -> R {
        let mut obj = self.inner.lock().unwrap();
        f(&mut obj)
    }

    /// Get a property value.
    pub fn get_property(&self, name: &str) -> Option<PropertyValue> {
        self.inner.lock().unwrap().get_property(name).cloned()
    }

    /// Set a property value.
    pub fn set_property(&self, name: &str, value: PropertyValue) {
        self.inner.lock().unwrap().set_property(name, value);
    }

    /// Returns the strong reference count.
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Returns the weak reference count.
    pub fn weak_count(&self) -> usize {
        Arc::weak_count(&self.inner)
    }
}

impl fmt::Debug for ObjectRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let id = self.id();
        f.debug_struct("ObjectRef")
            .field("id", &id)
            .field("strong_count", &Arc::strong_count(&self.inner))
            .finish()
    }
}

/// A weak reference to an engine object.
#[derive(Clone)]
pub struct WeakObjectRef {
    inner: Weak<Mutex<EngineObject>>,
}

impl WeakObjectRef {
    /// Try to upgrade to a strong reference.
    pub fn upgrade(&self) -> Option<ObjectRef> {
        self.inner.upgrade().map(|inner| ObjectRef { inner })
    }

    /// Check if the referenced object is still alive.
    pub fn is_alive(&self) -> bool {
        self.inner.strong_count() > 0
    }
}

impl fmt::Debug for WeakObjectRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeakObjectRef")
            .field("alive", &self.is_alive())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ObjectFactory
// ---------------------------------------------------------------------------

/// Factory for creating engine objects by type name.
pub struct ObjectFactory {
    constructors: HashMap<String, Box<dyn Fn(&str) -> EngineObject + Send + Sync>>,
}

impl ObjectFactory {
    /// Create a new, empty factory.
    pub fn new() -> Self {
        Self {
            constructors: HashMap::new(),
        }
    }

    /// Register a constructor for a type name.
    pub fn register<F>(&mut self, type_name: &str, constructor: F)
    where
        F: Fn(&str) -> EngineObject + Send + Sync + 'static,
    {
        self.constructors
            .insert(type_name.to_string(), Box::new(constructor));
    }

    /// Create an object by type name.
    pub fn create(&self, type_name: &str, name: &str) -> Option<EngineObject> {
        self.constructors.get(type_name).map(|ctor| ctor(name))
    }

    /// Create an ObjectRef by type name.
    pub fn create_ref(&self, type_name: &str, name: &str) -> Option<ObjectRef> {
        self.create(type_name, name).map(ObjectRef::new)
    }

    /// Get all registered type names.
    pub fn registered_types(&self) -> Vec<&str> {
        self.constructors.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a type is registered.
    pub fn has_type(&self, type_name: &str) -> bool {
        self.constructors.contains_key(type_name)
    }
}

impl Default for ObjectFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ObjectFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectFactory")
            .field("types", &self.constructors.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ObjectStore
// ---------------------------------------------------------------------------

/// Central store for managing all engine objects.
pub struct ObjectStore {
    objects: HashMap<ObjectId, ObjectRef>,
    name_index: HashMap<String, Vec<ObjectId>>,
    type_index: HashMap<String, Vec<ObjectId>>,
    tag_index: HashMap<String, Vec<ObjectId>>,
}

impl ObjectStore {
    /// Create a new, empty store.
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            name_index: HashMap::new(),
            type_index: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Add an object to the store.
    pub fn add(&mut self, object: EngineObject) -> ObjectRef {
        let id = object.id();
        let name = object.name().to_string();
        let type_name = object.type_name().to_string();
        let tags: Vec<String> = object.tags().to_vec();

        let obj_ref = ObjectRef::new(object);
        self.objects.insert(id, obj_ref.clone());

        self.name_index
            .entry(name)
            .or_default()
            .push(id);
        self.type_index
            .entry(type_name)
            .or_default()
            .push(id);
        for tag in tags {
            self.tag_index.entry(tag).or_default().push(id);
        }

        obj_ref
    }

    /// Get an object by ID.
    pub fn get(&self, id: ObjectId) -> Option<&ObjectRef> {
        self.objects.get(&id)
    }

    /// Remove an object by ID.
    pub fn remove(&mut self, id: ObjectId) -> Option<ObjectRef> {
        if let Some(obj_ref) = self.objects.remove(&id) {
            // Clean up indices.
            let name = obj_ref.with(|o| o.name().to_string());
            let type_name = obj_ref.with(|o| o.type_name().to_string());

            if let Some(ids) = self.name_index.get_mut(&name) {
                ids.retain(|oid| *oid != id);
            }
            if let Some(ids) = self.type_index.get_mut(&type_name) {
                ids.retain(|oid| *oid != id);
            }

            Some(obj_ref)
        } else {
            None
        }
    }

    /// Find objects by name.
    pub fn find_by_name(&self, name: &str) -> Vec<ObjectRef> {
        self.name_index
            .get(name)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.objects.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find objects by type.
    pub fn find_by_type(&self, type_name: &str) -> Vec<ObjectRef> {
        self.type_index
            .get(type_name)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.objects.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find objects by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<ObjectRef> {
        self.tag_index
            .get(tag)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.objects.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns all object IDs.
    pub fn all_ids(&self) -> Vec<ObjectId> {
        self.objects.keys().copied().collect()
    }

    /// Returns the total number of objects.
    pub fn count(&self) -> usize {
        self.objects.len()
    }

    /// Clear all objects.
    pub fn clear(&mut self) {
        self.objects.clear();
        self.name_index.clear();
        self.type_index.clear();
        self.tag_index.clear();
    }

    /// Serialize all objects.
    pub fn serialize_all(&self) -> Vec<ObjectData> {
        self.objects
            .values()
            .map(|obj_ref| obj_ref.with(|o| o.serialize()))
            .collect()
    }

    /// Deserialize and add objects.
    pub fn deserialize_all(&mut self, data: &[ObjectData]) {
        for obj_data in data {
            let obj = EngineObject::deserialize(obj_data);
            self.add(obj);
        }
    }
}

impl Default for ObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectStore")
            .field("objects", &self.objects.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_creation() {
        let obj = EngineObject::new("Node", "Player");
        assert!(!obj.id().is_null());
        assert_eq!(obj.type_name(), "Node");
        assert_eq!(obj.name(), "Player");
        assert!(obj.is_enabled());
    }

    #[test]
    fn test_properties() {
        let mut obj = EngineObject::new("Node", "Test");
        obj.set_property("health", PropertyValue::Int(100));
        obj.set_property("speed", PropertyValue::Float(5.5));
        obj.set_property("name", PropertyValue::String("Hero".to_string()));

        assert_eq!(obj.get_property("health").unwrap().as_int(), Some(100));
        assert_eq!(obj.get_property("speed").unwrap().as_float(), Some(5.5));
        assert_eq!(obj.get_property("name").unwrap().as_str(), Some("Hero"));
        assert_eq!(obj.property_count(), 3);
    }

    #[test]
    fn test_hierarchy() {
        let parent = EngineObject::new("Node", "Parent");
        let child = EngineObject::new("Node", "Child");
        let parent_id = parent.id();
        let child_id = child.id();

        let mut store = ObjectStore::new();
        store.add(parent);
        store.add(child);

        store.get(&parent_id).unwrap().with_mut(|p| {
            p.add_child(child_id);
        });
        store.get(&child_id).unwrap().with_mut(|c| {
            c.set_parent(Some(parent_id));
        });

        let children = store
            .get(&parent_id)
            .unwrap()
            .with(|p| p.children().to_vec());
        assert_eq!(children, vec![child_id]);
    }

    #[test]
    fn test_object_ref() {
        let obj = EngineObject::new("Node", "Shared");
        let obj_ref = ObjectRef::new(obj);
        let weak = obj_ref.downgrade();

        assert!(weak.is_alive());
        assert_eq!(obj_ref.strong_count(), 1);

        let clone = obj_ref.clone();
        assert_eq!(obj_ref.strong_count(), 2);

        drop(clone);
        assert_eq!(obj_ref.strong_count(), 1);

        drop(obj_ref);
        assert!(!weak.is_alive());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_factory() {
        let mut factory = ObjectFactory::new();
        factory.register("Camera", |name| {
            let mut obj = EngineObject::new("Camera", name);
            obj.set_property("fov", PropertyValue::Float(60.0));
            obj.set_property("near", PropertyValue::Float(0.1));
            obj.set_property("far", PropertyValue::Float(1000.0));
            obj
        });

        let cam = factory.create("Camera", "MainCamera").unwrap();
        assert_eq!(cam.type_name(), "Camera");
        assert_eq!(
            cam.get_property("fov").unwrap().as_float(),
            Some(60.0)
        );
    }

    #[test]
    fn test_store_queries() {
        let mut store = ObjectStore::new();

        let mut obj1 = EngineObject::new("Light", "Sun");
        obj1.add_tag("outdoor");
        store.add(obj1);

        let mut obj2 = EngineObject::new("Light", "Lamp");
        obj2.add_tag("indoor");
        store.add(obj2);

        let obj3 = EngineObject::new("Camera", "MainCam");
        store.add(obj3);

        assert_eq!(store.find_by_type("Light").len(), 2);
        assert_eq!(store.find_by_type("Camera").len(), 1);
        assert_eq!(store.find_by_tag("outdoor").len(), 1);
    }

    #[test]
    fn test_serialization() {
        let mut obj = EngineObject::new("Node", "Serializable");
        obj.set_property("value", PropertyValue::Int(42));
        obj.add_tag("important");

        let data = obj.serialize();
        let restored = EngineObject::deserialize(&data);

        assert_eq!(restored.id(), obj.id());
        assert_eq!(restored.name(), "Serializable");
        assert_eq!(
            restored.get_property("value").unwrap().as_int(),
            Some(42)
        );
        assert!(restored.has_tag("important"));
    }

    #[test]
    fn test_deep_clone() {
        let mut obj = EngineObject::new("Node", "Original");
        obj.set_property("x", PropertyValue::Float(1.0));

        let cloned = obj.deep_clone();
        assert_ne!(cloned.id(), obj.id()); // Different ID.
        assert_eq!(cloned.get_property("x").unwrap().as_float(), Some(1.0));
    }

    #[test]
    fn test_type_registry() {
        let mut registry = TypeRegistry::new();
        registry.register(
            TypeInfo::new("Object")
                .with_property(PropertyDescriptor::new("name", "string")),
        );
        registry.register(
            TypeInfo::new("Node")
                .with_parent("Object")
                .with_property(PropertyDescriptor::new("position", "vec3")),
        );

        let node_info = registry.get_type("Node").unwrap();
        assert!(node_info.inherits_from("Object", &registry));

        let all_props = node_info.all_properties(&registry);
        assert_eq!(all_props.len(), 2); // name + position
    }
}
