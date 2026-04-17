//! Complete scene save/load system for the Genovo engine.
//!
//! This module provides serialization and deserialization of entire world
//! states, including entities, components, parent-child hierarchies, and
//! resources. It uses a simplified JSON format compatible with the
//! [`ReflectSerializer`](genovo_core::type_registry::ReflectSerializer).
//!
//! # Features
//!
//! - **Full scene save**: serialize all entities and their components.
//! - **Full scene load**: reconstruct a world from serialized data.
//! - **Entity hierarchy**: preserve parent-child relationships.
//! - **Resource serialization**: save and restore world resources.
//! - **SceneDiff**: compute differences between two scene states for
//!   undo/redo, network sync, and collaborative editing.
//!
//! # JSON Format
//!
//! ```json
//! {
//!   "version": 1,
//!   "entities": [
//!     {
//!       "id": 0,
//!       "generation": 0,
//!       "parent": null,
//!       "children": [1, 2],
//!       "components": [
//!         {
//!           "__type": "Position",
//!           "x": 1.0,
//!           "y": 2.0
//!         }
//!       ]
//!     }
//!   ],
//!   "resources": [
//!     {
//!       "__type": "TimeOfDay",
//!       "hours": 14.5
//!     }
//!   ]
//! }
//! ```

use std::collections::HashMap;
use std::fmt;

// We reference types from the ECS and core crates conceptually.
// In a real build these would be proper crate dependencies.
// Here we define self-contained types to keep the module compilable.

// ---------------------------------------------------------------------------
// SerializedField -- a single field value in serialized form
// ---------------------------------------------------------------------------

/// A serialized field value -- either a primitive, string, or nested object.
#[derive(Debug, Clone, PartialEq)]
pub enum SerializedValue {
    /// Null / missing value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Integer value (stored as i64).
    Integer(i64),
    /// Floating-point value.
    Float(f64),
    /// String value.
    String(String),
    /// A nested object with named fields.
    Object(HashMap<String, SerializedValue>),
    /// An array of values.
    Array(Vec<SerializedValue>),
}

impl SerializedValue {
    /// Get as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as i64.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get as string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    /// Get as object.
    pub fn as_object(&self) -> Option<&HashMap<String, SerializedValue>> {
        match self {
            Self::Object(v) => Some(v),
            _ => None,
        }
    }

    /// Get as array.
    pub fn as_array(&self) -> Option<&[SerializedValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Check if null.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Convert to a JSON string.
    pub fn to_json(&self) -> String {
        self.to_json_indent(0)
    }

    /// Convert to a JSON string with indentation.
    fn to_json_indent(&self, indent: usize) -> String {
        let pad = " ".repeat(indent);
        let inner_pad = " ".repeat(indent + 2);

        match self {
            Self::Null => "null".to_string(),
            Self::Bool(v) => format!("{}", v),
            Self::Integer(v) => format!("{}", v),
            Self::Float(v) => {
                if v.fract() == 0.0 && v.abs() < 1e15 {
                    format!("{:.1}", v)
                } else {
                    format!("{}", v)
                }
            }
            Self::String(v) => format!("\"{}\"", escape_json(v)),
            Self::Object(fields) => {
                if fields.is_empty() {
                    return "{}".to_string();
                }
                let mut result = "{\n".to_string();
                let mut entries: Vec<_> = fields.iter().collect();
                // Sort keys for deterministic output.
                entries.sort_by_key(|(k, _)| {
                    // Put __type first.
                    if *k == "__type" { "".to_string() } else { k.to_string() }
                });
                for (i, (key, value)) in entries.iter().enumerate() {
                    let val_json = value.to_json_indent(indent + 2);
                    if i < entries.len() - 1 {
                        result.push_str(&format!("{}\"{}\": {},\n", inner_pad, key, val_json));
                    } else {
                        result.push_str(&format!("{}\"{}\": {}\n", inner_pad, key, val_json));
                    }
                }
                result.push_str(&format!("{}}}", pad));
                result
            }
            Self::Array(items) => {
                if items.is_empty() {
                    return "[]".to_string();
                }
                let mut result = "[\n".to_string();
                for (i, item) in items.iter().enumerate() {
                    let item_json = item.to_json_indent(indent + 2);
                    if i < items.len() - 1 {
                        result.push_str(&format!("{}{},\n", inner_pad, item_json));
                    } else {
                        result.push_str(&format!("{}{}\n", inner_pad, item_json));
                    }
                }
                result.push_str(&format!("{}]", pad));
                result
            }
        }
    }
}

/// Escape special characters for JSON string output.
fn escape_json(s: &str) -> String {
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

// ---------------------------------------------------------------------------
// SerializedComponent -- a single component in serialized form
// ---------------------------------------------------------------------------

/// A serialized component -- its type name and field values.
#[derive(Debug, Clone, PartialEq)]
pub struct SerializedComponent {
    /// The component type name.
    pub type_name: String,
    /// Field name -> value map.
    pub fields: HashMap<String, SerializedValue>,
}

impl SerializedComponent {
    /// Create a new serialized component.
    pub fn new(type_name: &str) -> Self {
        Self {
            type_name: type_name.to_string(),
            fields: HashMap::new(),
        }
    }

    /// Set a field value.
    pub fn set_field(&mut self, name: &str, value: SerializedValue) {
        self.fields.insert(name.to_string(), value);
    }

    /// Get a field value.
    pub fn get_field(&self, name: &str) -> Option<&SerializedValue> {
        self.fields.get(name)
    }

    /// Convert to a `SerializedValue` (Object with __type).
    pub fn to_value(&self) -> SerializedValue {
        let mut obj = HashMap::new();
        obj.insert(
            "__type".to_string(),
            SerializedValue::String(self.type_name.clone()),
        );
        for (k, v) in &self.fields {
            obj.insert(k.clone(), v.clone());
        }
        SerializedValue::Object(obj)
    }

    /// Parse from a `SerializedValue`.
    pub fn from_value(value: &SerializedValue) -> Option<Self> {
        let obj = value.as_object()?;
        let type_name = obj.get("__type")?.as_str()?.to_string();

        let mut fields = HashMap::new();
        for (k, v) in obj {
            if k != "__type" {
                fields.insert(k.clone(), v.clone());
            }
        }

        Some(Self { type_name, fields })
    }
}

// ---------------------------------------------------------------------------
// SerializedEntity -- a single entity in serialized form
// ---------------------------------------------------------------------------

/// A serialized entity with its id, hierarchy, and components.
#[derive(Debug, Clone)]
pub struct SerializedEntity {
    /// Entity slot id.
    pub id: u32,
    /// Entity generation.
    pub generation: u32,
    /// Parent entity id, if any.
    pub parent: Option<u32>,
    /// Child entity ids.
    pub children: Vec<u32>,
    /// Serialized components.
    pub components: Vec<SerializedComponent>,
    /// Optional entity name / label.
    pub name: Option<String>,
    /// Whether this entity is enabled/active.
    pub enabled: bool,
}

impl SerializedEntity {
    /// Create a new serialized entity with no components.
    pub fn new(id: u32, generation: u32) -> Self {
        Self {
            id,
            generation,
            parent: None,
            children: Vec::new(),
            components: Vec::new(),
            name: None,
            enabled: true,
        }
    }

    /// Add a component.
    pub fn add_component(&mut self, component: SerializedComponent) {
        self.components.push(component);
    }

    /// Get a component by type name.
    pub fn get_component(&self, type_name: &str) -> Option<&SerializedComponent> {
        self.components.iter().find(|c| c.type_name == type_name)
    }

    /// Check if the entity has a component of the given type.
    pub fn has_component(&self, type_name: &str) -> bool {
        self.components.iter().any(|c| c.type_name == type_name)
    }

    /// Convert to a `SerializedValue`.
    pub fn to_value(&self) -> SerializedValue {
        let mut obj = HashMap::new();
        obj.insert("id".to_string(), SerializedValue::Integer(self.id as i64));
        obj.insert(
            "generation".to_string(),
            SerializedValue::Integer(self.generation as i64),
        );

        match self.parent {
            Some(pid) => {
                obj.insert("parent".to_string(), SerializedValue::Integer(pid as i64));
            }
            None => {
                obj.insert("parent".to_string(), SerializedValue::Null);
            }
        }

        let children_vals: Vec<SerializedValue> = self
            .children
            .iter()
            .map(|&c| SerializedValue::Integer(c as i64))
            .collect();
        obj.insert("children".to_string(), SerializedValue::Array(children_vals));

        let comp_vals: Vec<SerializedValue> = self
            .components
            .iter()
            .map(|c| c.to_value())
            .collect();
        obj.insert("components".to_string(), SerializedValue::Array(comp_vals));

        if let Some(ref name) = self.name {
            obj.insert("name".to_string(), SerializedValue::String(name.clone()));
        }

        obj.insert("enabled".to_string(), SerializedValue::Bool(self.enabled));

        SerializedValue::Object(obj)
    }

    /// Parse from a `SerializedValue`.
    pub fn from_value(value: &SerializedValue) -> Option<Self> {
        let obj = value.as_object()?;

        let id = obj.get("id")?.as_integer()? as u32;
        let generation = obj.get("generation")?.as_integer()? as u32;

        let parent = obj
            .get("parent")
            .and_then(|v| v.as_integer())
            .map(|v| v as u32);

        let children = obj
            .get("children")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_integer().map(|i| i as u32))
                    .collect()
            })
            .unwrap_or_default();

        let components = obj
            .get("components")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| SerializedComponent::from_value(v))
                    .collect()
            })
            .unwrap_or_default();

        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let enabled = obj
            .get("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Some(Self {
            id,
            generation,
            parent,
            children,
            components,
            name,
            enabled,
        })
    }
}

// ---------------------------------------------------------------------------
// SerializedScene -- a complete scene
// ---------------------------------------------------------------------------

/// A complete serialized scene containing entities, hierarchy, and resources.
#[derive(Debug, Clone)]
pub struct SerializedScene {
    /// Format version.
    pub version: u32,
    /// All serialized entities.
    pub entities: Vec<SerializedEntity>,
    /// Serialized resources.
    pub resources: Vec<SerializedComponent>,
    /// Optional scene metadata.
    pub metadata: HashMap<String, String>,
}

impl SerializedScene {
    /// Create a new, empty scene.
    pub fn new() -> Self {
        Self {
            version: 1,
            entities: Vec::new(),
            resources: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add an entity to the scene.
    pub fn add_entity(&mut self, entity: SerializedEntity) {
        self.entities.push(entity);
    }

    /// Add a resource to the scene.
    pub fn add_resource(&mut self, resource: SerializedComponent) {
        self.resources.push(resource);
    }

    /// Set metadata.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get metadata.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Find an entity by id.
    pub fn find_entity(&self, id: u32) -> Option<&SerializedEntity> {
        self.entities.iter().find(|e| e.id == id)
    }

    /// Find a resource by type name.
    pub fn find_resource(&self, type_name: &str) -> Option<&SerializedComponent> {
        self.resources.iter().find(|r| r.type_name == type_name)
    }

    /// Number of entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Number of resources.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Convert the entire scene to JSON.
    pub fn to_json(&self) -> String {
        let value = self.to_value();
        value.to_json()
    }

    /// Convert to a `SerializedValue`.
    pub fn to_value(&self) -> SerializedValue {
        let mut obj = HashMap::new();
        obj.insert(
            "version".to_string(),
            SerializedValue::Integer(self.version as i64),
        );

        let entity_vals: Vec<SerializedValue> = self
            .entities
            .iter()
            .map(|e| e.to_value())
            .collect();
        obj.insert("entities".to_string(), SerializedValue::Array(entity_vals));

        let resource_vals: Vec<SerializedValue> = self
            .resources
            .iter()
            .map(|r| r.to_value())
            .collect();
        obj.insert("resources".to_string(), SerializedValue::Array(resource_vals));

        if !self.metadata.is_empty() {
            let mut meta_obj = HashMap::new();
            for (k, v) in &self.metadata {
                meta_obj.insert(k.clone(), SerializedValue::String(v.clone()));
            }
            obj.insert("metadata".to_string(), SerializedValue::Object(meta_obj));
        }

        SerializedValue::Object(obj)
    }

    /// Parse from JSON string.
    pub fn from_json(json: &str) -> Option<Self> {
        let value = parse_json(json)?;
        Self::from_value(&value)
    }

    /// Parse from a `SerializedValue`.
    pub fn from_value(value: &SerializedValue) -> Option<Self> {
        let obj = value.as_object()?;

        let version = obj
            .get("version")
            .and_then(|v| v.as_integer())
            .unwrap_or(1) as u32;

        let entities = obj
            .get("entities")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| SerializedEntity::from_value(v))
                    .collect()
            })
            .unwrap_or_default();

        let resources = obj
            .get("resources")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| SerializedComponent::from_value(v))
                    .collect()
            })
            .unwrap_or_default();

        let metadata = obj
            .get("metadata")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        Some(Self {
            version,
            entities,
            resources,
            metadata,
        })
    }
}

impl Default for SerializedScene {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SceneSerializer -- facade for save/load operations
// ---------------------------------------------------------------------------

/// Facade for scene serialization / deserialization operations.
///
/// In a full engine integration, this would take a `&World` and serialize
/// all registered components via the reflection system. Here we provide the
/// data-structure layer that the reflection system plugs into.
pub struct SceneSerializer;

impl SceneSerializer {
    /// Serialize a scene to a JSON string.
    pub fn save(scene: &SerializedScene) -> String {
        scene.to_json()
    }

    /// Deserialize a scene from a JSON string.
    pub fn load(json: &str) -> Option<SerializedScene> {
        SerializedScene::from_json(json)
    }

    /// Save a scene to a file (writes to a string -- filesystem I/O would
    /// be added in the platform layer).
    pub fn save_to_string(scene: &SerializedScene) -> String {
        scene.to_json()
    }

    /// Validate a serialized scene for consistency.
    pub fn validate(scene: &SerializedScene) -> Vec<String> {
        let mut errors = Vec::new();

        // Check for duplicate entity ids.
        let mut seen_ids = std::collections::HashSet::new();
        for entity in &scene.entities {
            if !seen_ids.insert(entity.id) {
                errors.push(format!("Duplicate entity id: {}", entity.id));
            }
        }

        // Check parent references.
        for entity in &scene.entities {
            if let Some(parent_id) = entity.parent {
                if !seen_ids.contains(&parent_id) {
                    errors.push(format!(
                        "Entity {} references non-existent parent {}",
                        entity.id, parent_id
                    ));
                }
            }
        }

        // Check children references.
        for entity in &scene.entities {
            for &child_id in &entity.children {
                if !seen_ids.contains(&child_id) {
                    errors.push(format!(
                        "Entity {} references non-existent child {}",
                        entity.id, child_id
                    ));
                }
            }
        }

        // Check for components with no type name.
        for entity in &scene.entities {
            for comp in &entity.components {
                if comp.type_name.is_empty() {
                    errors.push(format!(
                        "Entity {} has a component with no type name",
                        entity.id
                    ));
                }
            }
        }

        errors
    }
}

// ---------------------------------------------------------------------------
// SceneDiff -- compute differences between scene states
// ---------------------------------------------------------------------------

/// The kind of change in a scene diff.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffChange {
    /// An entity was added.
    EntityAdded {
        entity_id: u32,
    },
    /// An entity was removed.
    EntityRemoved {
        entity_id: u32,
    },
    /// A component was added to an entity.
    ComponentAdded {
        entity_id: u32,
        component: SerializedComponent,
    },
    /// A component was removed from an entity.
    ComponentRemoved {
        entity_id: u32,
        type_name: String,
    },
    /// A component field value changed.
    FieldChanged {
        entity_id: u32,
        type_name: String,
        field_name: String,
        old_value: SerializedValue,
        new_value: SerializedValue,
    },
    /// An entity's parent changed.
    ParentChanged {
        entity_id: u32,
        old_parent: Option<u32>,
        new_parent: Option<u32>,
    },
    /// A resource was added.
    ResourceAdded {
        resource: SerializedComponent,
    },
    /// A resource was removed.
    ResourceRemoved {
        type_name: String,
    },
    /// A resource field changed.
    ResourceFieldChanged {
        type_name: String,
        field_name: String,
        old_value: SerializedValue,
        new_value: SerializedValue,
    },
}

/// The result of diffing two scene states.
///
/// Contains all changes needed to transform the "before" scene into the
/// "after" scene. Useful for:
///
/// - **Undo/redo**: apply the diff forward or reverse.
/// - **Network sync**: send only changes, not the full scene.
/// - **Collaborative editing**: merge diffs from multiple editors.
#[derive(Debug, Clone)]
pub struct SceneDiff {
    /// All changes between the two scenes.
    pub changes: Vec<DiffChange>,
}

impl SceneDiff {
    /// Compute the difference between two scenes.
    pub fn compute(before: &SerializedScene, after: &SerializedScene) -> Self {
        let mut changes = Vec::new();

        // Index entities by id.
        let before_map: HashMap<u32, &SerializedEntity> = before
            .entities
            .iter()
            .map(|e| (e.id, e))
            .collect();
        let after_map: HashMap<u32, &SerializedEntity> = after
            .entities
            .iter()
            .map(|e| (e.id, e))
            .collect();

        // Find added and modified entities.
        for (&id, &after_entity) in &after_map {
            match before_map.get(&id) {
                None => {
                    // Entity was added.
                    changes.push(DiffChange::EntityAdded { entity_id: id });
                }
                Some(&before_entity) => {
                    // Entity exists in both -- check for modifications.
                    Self::diff_entity(before_entity, after_entity, &mut changes);
                }
            }
        }

        // Find removed entities.
        for &id in before_map.keys() {
            if !after_map.contains_key(&id) {
                changes.push(DiffChange::EntityRemoved { entity_id: id });
            }
        }

        // Diff resources.
        let before_resources: HashMap<&str, &SerializedComponent> = before
            .resources
            .iter()
            .map(|r| (r.type_name.as_str(), r))
            .collect();
        let after_resources: HashMap<&str, &SerializedComponent> = after
            .resources
            .iter()
            .map(|r| (r.type_name.as_str(), r))
            .collect();

        for (&type_name, &after_res) in &after_resources {
            match before_resources.get(type_name) {
                None => {
                    changes.push(DiffChange::ResourceAdded {
                        resource: after_res.clone(),
                    });
                }
                Some(&before_res) => {
                    // Diff fields.
                    Self::diff_component_fields(
                        before_res,
                        after_res,
                        0, // entity_id unused for resources
                        true,
                        &mut changes,
                    );
                }
            }
        }

        for &type_name in before_resources.keys() {
            if !after_resources.contains_key(type_name) {
                changes.push(DiffChange::ResourceRemoved {
                    type_name: type_name.to_string(),
                });
            }
        }

        Self { changes }
    }

    /// Diff two entities.
    fn diff_entity(
        before: &SerializedEntity,
        after: &SerializedEntity,
        changes: &mut Vec<DiffChange>,
    ) {
        let id = before.id;

        // Check parent change.
        if before.parent != after.parent {
            changes.push(DiffChange::ParentChanged {
                entity_id: id,
                old_parent: before.parent,
                new_parent: after.parent,
            });
        }

        // Check component additions/removals/modifications.
        let before_comps: HashMap<&str, &SerializedComponent> = before
            .components
            .iter()
            .map(|c| (c.type_name.as_str(), c))
            .collect();
        let after_comps: HashMap<&str, &SerializedComponent> = after
            .components
            .iter()
            .map(|c| (c.type_name.as_str(), c))
            .collect();

        for (&type_name, &after_comp) in &after_comps {
            match before_comps.get(type_name) {
                None => {
                    changes.push(DiffChange::ComponentAdded {
                        entity_id: id,
                        component: after_comp.clone(),
                    });
                }
                Some(&before_comp) => {
                    Self::diff_component_fields(before_comp, after_comp, id, false, changes);
                }
            }
        }

        for &type_name in before_comps.keys() {
            if !after_comps.contains_key(type_name) {
                changes.push(DiffChange::ComponentRemoved {
                    entity_id: id,
                    type_name: type_name.to_string(),
                });
            }
        }
    }

    /// Diff the fields of two components.
    fn diff_component_fields(
        before: &SerializedComponent,
        after: &SerializedComponent,
        entity_id: u32,
        is_resource: bool,
        changes: &mut Vec<DiffChange>,
    ) {
        // Check for changed or added fields.
        for (field_name, after_value) in &after.fields {
            let changed = match before.fields.get(field_name) {
                None => true,
                Some(before_value) => before_value != after_value,
            };

            if changed {
                let old_value = before
                    .fields
                    .get(field_name)
                    .cloned()
                    .unwrap_or(SerializedValue::Null);

                if is_resource {
                    changes.push(DiffChange::ResourceFieldChanged {
                        type_name: before.type_name.clone(),
                        field_name: field_name.clone(),
                        old_value,
                        new_value: after_value.clone(),
                    });
                } else {
                    changes.push(DiffChange::FieldChanged {
                        entity_id,
                        type_name: before.type_name.clone(),
                        field_name: field_name.clone(),
                        old_value,
                        new_value: after_value.clone(),
                    });
                }
            }
        }
    }

    /// Apply this diff to a scene (forward direction).
    pub fn apply_forward(&self, scene: &mut SerializedScene) {
        for change in &self.changes {
            match change {
                DiffChange::EntityAdded { entity_id } => {
                    if scene.find_entity(*entity_id).is_none() {
                        scene.add_entity(SerializedEntity::new(*entity_id, 0));
                    }
                }
                DiffChange::EntityRemoved { entity_id } => {
                    scene.entities.retain(|e| e.id != *entity_id);
                }
                DiffChange::ComponentAdded {
                    entity_id,
                    component,
                } => {
                    if let Some(entity) = scene
                        .entities
                        .iter_mut()
                        .find(|e| e.id == *entity_id)
                    {
                        entity.add_component(component.clone());
                    }
                }
                DiffChange::ComponentRemoved {
                    entity_id,
                    type_name,
                } => {
                    if let Some(entity) = scene
                        .entities
                        .iter_mut()
                        .find(|e| e.id == *entity_id)
                    {
                        entity.components.retain(|c| c.type_name != *type_name);
                    }
                }
                DiffChange::FieldChanged {
                    entity_id,
                    type_name,
                    field_name,
                    new_value,
                    ..
                } => {
                    if let Some(entity) = scene
                        .entities
                        .iter_mut()
                        .find(|e| e.id == *entity_id)
                    {
                        if let Some(comp) = entity
                            .components
                            .iter_mut()
                            .find(|c| c.type_name == *type_name)
                        {
                            comp.set_field(field_name, new_value.clone());
                        }
                    }
                }
                DiffChange::ParentChanged {
                    entity_id,
                    new_parent,
                    ..
                } => {
                    if let Some(entity) = scene
                        .entities
                        .iter_mut()
                        .find(|e| e.id == *entity_id)
                    {
                        entity.parent = *new_parent;
                    }
                }
                DiffChange::ResourceAdded { resource } => {
                    scene.add_resource(resource.clone());
                }
                DiffChange::ResourceRemoved { type_name } => {
                    scene.resources.retain(|r| r.type_name != *type_name);
                }
                DiffChange::ResourceFieldChanged {
                    type_name,
                    field_name,
                    new_value,
                    ..
                } => {
                    if let Some(res) = scene
                        .resources
                        .iter_mut()
                        .find(|r| r.type_name == *type_name)
                    {
                        res.set_field(field_name, new_value.clone());
                    }
                }
            }
        }
    }

    /// Apply this diff in reverse (undo).
    pub fn apply_reverse(&self, scene: &mut SerializedScene) {
        // Process changes in reverse order.
        for change in self.changes.iter().rev() {
            match change {
                DiffChange::EntityAdded { entity_id } => {
                    // Reverse of add = remove.
                    scene.entities.retain(|e| e.id != *entity_id);
                }
                DiffChange::EntityRemoved { entity_id } => {
                    // Reverse of remove = add (with no components -- the
                    // component changes will restore them).
                    if scene.find_entity(*entity_id).is_none() {
                        scene.add_entity(SerializedEntity::new(*entity_id, 0));
                    }
                }
                DiffChange::ComponentAdded {
                    entity_id,
                    component,
                } => {
                    // Reverse of add = remove.
                    if let Some(entity) = scene
                        .entities
                        .iter_mut()
                        .find(|e| e.id == *entity_id)
                    {
                        entity
                            .components
                            .retain(|c| c.type_name != component.type_name);
                    }
                }
                DiffChange::ComponentRemoved {
                    entity_id,
                    type_name,
                } => {
                    // Reverse of remove = add (empty component).
                    if let Some(entity) = scene
                        .entities
                        .iter_mut()
                        .find(|e| e.id == *entity_id)
                    {
                        entity.add_component(SerializedComponent::new(type_name));
                    }
                }
                DiffChange::FieldChanged {
                    entity_id,
                    type_name,
                    field_name,
                    old_value,
                    ..
                } => {
                    if let Some(entity) = scene
                        .entities
                        .iter_mut()
                        .find(|e| e.id == *entity_id)
                    {
                        if let Some(comp) = entity
                            .components
                            .iter_mut()
                            .find(|c| c.type_name == *type_name)
                        {
                            comp.set_field(field_name, old_value.clone());
                        }
                    }
                }
                DiffChange::ParentChanged {
                    entity_id,
                    old_parent,
                    ..
                } => {
                    if let Some(entity) = scene
                        .entities
                        .iter_mut()
                        .find(|e| e.id == *entity_id)
                    {
                        entity.parent = *old_parent;
                    }
                }
                DiffChange::ResourceAdded { resource } => {
                    scene
                        .resources
                        .retain(|r| r.type_name != resource.type_name);
                }
                DiffChange::ResourceRemoved { type_name } => {
                    scene.add_resource(SerializedComponent::new(type_name));
                }
                DiffChange::ResourceFieldChanged {
                    type_name,
                    field_name,
                    old_value,
                    ..
                } => {
                    if let Some(res) = scene
                        .resources
                        .iter_mut()
                        .find(|r| r.type_name == *type_name)
                    {
                        res.set_field(field_name, old_value.clone());
                    }
                }
            }
        }
    }

    /// Number of changes.
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Whether there are no changes.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Filter changes to only include those affecting a specific entity.
    pub fn changes_for_entity(&self, entity_id: u32) -> Vec<&DiffChange> {
        self.changes
            .iter()
            .filter(|c| match c {
                DiffChange::EntityAdded { entity_id: id } => *id == entity_id,
                DiffChange::EntityRemoved { entity_id: id } => *id == entity_id,
                DiffChange::ComponentAdded { entity_id: id, .. } => *id == entity_id,
                DiffChange::ComponentRemoved { entity_id: id, .. } => *id == entity_id,
                DiffChange::FieldChanged { entity_id: id, .. } => *id == entity_id,
                DiffChange::ParentChanged { entity_id: id, .. } => *id == entity_id,
                _ => false,
            })
            .collect()
    }

    /// Check if a specific entity was affected.
    pub fn entity_affected(&self, entity_id: u32) -> bool {
        !self.changes_for_entity(entity_id).is_empty()
    }
}

// ---------------------------------------------------------------------------
// Minimal JSON parser (for self-contained scene loading)
// ---------------------------------------------------------------------------

/// Parse a simplified JSON string into a `SerializedValue`.
///
/// This is a minimal recursive descent parser that handles the JSON subset
/// used by the scene serializer. It handles objects, arrays, strings,
/// numbers, booleans, and null.
pub fn parse_json(input: &str) -> Option<SerializedValue> {
    let trimmed = input.trim();
    let (value, _) = parse_value(trimmed)?;
    Some(value)
}

/// Parse a JSON value starting at the given position.
fn parse_value(input: &str) -> Option<(SerializedValue, &str)> {
    let input = input.trim_start();

    if input.is_empty() {
        return None;
    }

    match input.as_bytes()[0] {
        b'{' => parse_object(input),
        b'[' => parse_array(input),
        b'"' => parse_string_value(input),
        b't' | b'f' => parse_bool(input),
        b'n' => parse_null(input),
        b'-' | b'0'..=b'9' => parse_number(input),
        _ => None,
    }
}

/// Parse a JSON object.
fn parse_object(input: &str) -> Option<(SerializedValue, &str)> {
    let input = input.trim_start();
    if !input.starts_with('{') {
        return None;
    }
    let mut rest = input[1..].trim_start();
    let mut fields = HashMap::new();

    if rest.starts_with('}') {
        return Some((SerializedValue::Object(fields), &rest[1..]));
    }

    loop {
        // Parse key.
        let (key_val, r) = parse_string_value(rest)?;
        let key = match key_val {
            SerializedValue::String(s) => s,
            _ => return None,
        };
        rest = r.trim_start();

        // Expect colon.
        if !rest.starts_with(':') {
            return None;
        }
        rest = rest[1..].trim_start();

        // Parse value.
        let (value, r) = parse_value(rest)?;
        rest = r.trim_start();

        fields.insert(key, value);

        // Check for comma or closing brace.
        if rest.starts_with(',') {
            rest = rest[1..].trim_start();
        } else if rest.starts_with('}') {
            rest = &rest[1..];
            break;
        } else {
            return None;
        }
    }

    Some((SerializedValue::Object(fields), rest))
}

/// Parse a JSON array.
fn parse_array(input: &str) -> Option<(SerializedValue, &str)> {
    let input = input.trim_start();
    if !input.starts_with('[') {
        return None;
    }
    let mut rest = input[1..].trim_start();
    let mut items = Vec::new();

    if rest.starts_with(']') {
        return Some((SerializedValue::Array(items), &rest[1..]));
    }

    loop {
        let (value, r) = parse_value(rest)?;
        rest = r.trim_start();
        items.push(value);

        if rest.starts_with(',') {
            rest = rest[1..].trim_start();
        } else if rest.starts_with(']') {
            rest = &rest[1..];
            break;
        } else {
            return None;
        }
    }

    Some((SerializedValue::Array(items), rest))
}

/// Parse a JSON string.
fn parse_string_value(input: &str) -> Option<(SerializedValue, &str)> {
    let input = input.trim_start();
    if !input.starts_with('"') {
        return None;
    }

    let mut result = String::new();
    let mut chars = input[1..].char_indices();
    let mut end_pos = 0;

    loop {
        match chars.next() {
            None => return None, // unterminated string
            Some((pos, '"')) => {
                end_pos = pos + 2; // +1 for opening quote, +1 for closing
                break;
            }
            Some((_, '\\')) => {
                match chars.next() {
                    Some((_, '"')) => result.push('"'),
                    Some((_, '\\')) => result.push('\\'),
                    Some((_, '/')) => result.push('/'),
                    Some((_, 'n')) => result.push('\n'),
                    Some((_, 'r')) => result.push('\r'),
                    Some((_, 't')) => result.push('\t'),
                    Some((_, 'u')) => {
                        let hex: String = (0..4)
                            .filter_map(|_| chars.next().map(|(_, c)| c))
                            .collect();
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(ch) = char::from_u32(code) {
                                result.push(ch);
                            }
                        }
                    }
                    _ => return None,
                }
            }
            Some((_, c)) => result.push(c),
        }
    }

    Some((SerializedValue::String(result), &input[end_pos..]))
}

/// Parse a JSON boolean.
fn parse_bool(input: &str) -> Option<(SerializedValue, &str)> {
    if input.starts_with("true") {
        Some((SerializedValue::Bool(true), &input[4..]))
    } else if input.starts_with("false") {
        Some((SerializedValue::Bool(false), &input[5..]))
    } else {
        None
    }
}

/// Parse a JSON null.
fn parse_null(input: &str) -> Option<(SerializedValue, &str)> {
    if input.starts_with("null") {
        Some((SerializedValue::Null, &input[4..]))
    } else {
        None
    }
}

/// Parse a JSON number.
fn parse_number(input: &str) -> Option<(SerializedValue, &str)> {
    let end = input
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E')
        .unwrap_or(input.len());

    let num_str = &input[..end];

    // Try integer first, then float.
    if let Ok(i) = num_str.parse::<i64>() {
        if !num_str.contains('.') && !num_str.contains('e') && !num_str.contains('E') {
            return Some((SerializedValue::Integer(i), &input[end..]));
        }
    }

    if let Ok(f) = num_str.parse::<f64>() {
        Some((SerializedValue::Float(f), &input[end..]))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialized_value_primitives() {
        assert!(SerializedValue::Null.is_null());
        assert_eq!(SerializedValue::Bool(true).as_bool(), Some(true));
        assert_eq!(SerializedValue::Integer(42).as_integer(), Some(42));
        assert_eq!(SerializedValue::Float(3.14).as_float(), Some(3.14));
        assert_eq!(
            SerializedValue::String("hello".into()).as_str(),
            Some("hello")
        );
    }

    #[test]
    fn serialized_value_integer_as_float() {
        assert_eq!(SerializedValue::Integer(10).as_float(), Some(10.0));
    }

    #[test]
    fn serialized_component_basic() {
        let mut comp = SerializedComponent::new("Position");
        comp.set_field("x", SerializedValue::Float(1.0));
        comp.set_field("y", SerializedValue::Float(2.0));

        assert_eq!(comp.type_name, "Position");
        assert_eq!(comp.get_field("x").unwrap().as_float(), Some(1.0));
    }

    #[test]
    fn serialized_component_to_from_value() {
        let mut comp = SerializedComponent::new("Velocity");
        comp.set_field("dx", SerializedValue::Float(3.0));
        comp.set_field("dy", SerializedValue::Float(4.0));

        let value = comp.to_value();
        let restored = SerializedComponent::from_value(&value).unwrap();
        assert_eq!(restored.type_name, "Velocity");
        assert_eq!(restored.get_field("dx").unwrap().as_float(), Some(3.0));
    }

    #[test]
    fn serialized_entity_basic() {
        let mut entity = SerializedEntity::new(0, 0);
        let mut pos = SerializedComponent::new("Position");
        pos.set_field("x", SerializedValue::Float(10.0));
        entity.add_component(pos);
        entity.parent = Some(5);
        entity.children = vec![1, 2];
        entity.name = Some("player".to_string());

        assert!(entity.has_component("Position"));
        assert!(!entity.has_component("Velocity"));
    }

    #[test]
    fn serialized_entity_roundtrip() {
        let mut entity = SerializedEntity::new(7, 3);
        entity.parent = Some(1);
        entity.children = vec![8, 9];
        entity.name = Some("enemy".to_string());
        entity.enabled = false;

        let mut hp = SerializedComponent::new("Health");
        hp.set_field("value", SerializedValue::Float(50.0));
        entity.add_component(hp);

        let value = entity.to_value();
        let restored = SerializedEntity::from_value(&value).unwrap();

        assert_eq!(restored.id, 7);
        assert_eq!(restored.generation, 3);
        assert_eq!(restored.parent, Some(1));
        assert_eq!(restored.children, vec![8, 9]);
        assert_eq!(restored.name.as_deref(), Some("enemy"));
        assert_eq!(restored.enabled, false);
        assert_eq!(restored.components.len(), 1);
    }

    #[test]
    fn serialized_scene_basic() {
        let mut scene = SerializedScene::new();
        scene.set_metadata("author", "test");

        let mut e0 = SerializedEntity::new(0, 0);
        let mut pos = SerializedComponent::new("Position");
        pos.set_field("x", SerializedValue::Float(1.0));
        pos.set_field("y", SerializedValue::Float(2.0));
        e0.add_component(pos);
        scene.add_entity(e0);

        let mut time = SerializedComponent::new("TimeOfDay");
        time.set_field("hours", SerializedValue::Float(14.5));
        scene.add_resource(time);

        assert_eq!(scene.entity_count(), 1);
        assert_eq!(scene.resource_count(), 1);
        assert_eq!(scene.get_metadata("author"), Some("test"));
    }

    #[test]
    fn scene_json_roundtrip() {
        let mut scene = SerializedScene::new();

        let mut e0 = SerializedEntity::new(0, 0);
        e0.children = vec![1];
        let mut pos = SerializedComponent::new("Position");
        pos.set_field("x", SerializedValue::Float(5.0));
        pos.set_field("y", SerializedValue::Float(10.0));
        e0.add_component(pos);
        scene.add_entity(e0);

        let mut e1 = SerializedEntity::new(1, 0);
        e1.parent = Some(0);
        let mut hp = SerializedComponent::new("Health");
        hp.set_field("value", SerializedValue::Integer(100));
        e1.add_component(hp);
        scene.add_entity(e1);

        let json = scene.to_json();
        let restored = SerializedScene::from_json(&json).unwrap();

        assert_eq!(restored.version, 1);
        assert_eq!(restored.entity_count(), 2);

        let re0 = restored.find_entity(0).unwrap();
        assert_eq!(re0.children, vec![1]);
        let re1 = restored.find_entity(1).unwrap();
        assert_eq!(re1.parent, Some(0));
    }

    #[test]
    fn scene_serializer_save_load() {
        let mut scene = SerializedScene::new();
        let mut e = SerializedEntity::new(0, 0);
        e.name = Some("test_entity".to_string());
        scene.add_entity(e);

        let json = SceneSerializer::save(&scene);
        let loaded = SceneSerializer::load(&json).unwrap();

        assert_eq!(loaded.entity_count(), 1);
        assert_eq!(loaded.find_entity(0).unwrap().name.as_deref(), Some("test_entity"));
    }

    #[test]
    fn scene_validation() {
        let mut scene = SerializedScene::new();

        // Duplicate entity ids.
        scene.add_entity(SerializedEntity::new(0, 0));
        scene.add_entity(SerializedEntity::new(0, 1));

        let errors = SceneSerializer::validate(&scene);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("Duplicate")));
    }

    #[test]
    fn scene_validation_bad_parent() {
        let mut scene = SerializedScene::new();
        let mut e = SerializedEntity::new(0, 0);
        e.parent = Some(99); // non-existent parent
        scene.add_entity(e);

        let errors = SceneSerializer::validate(&scene);
        assert!(errors.iter().any(|e| e.contains("non-existent parent")));
    }

    #[test]
    fn scene_diff_empty() {
        let a = SerializedScene::new();
        let b = SerializedScene::new();
        let diff = SceneDiff::compute(&a, &b);
        assert!(diff.is_empty());
    }

    #[test]
    fn scene_diff_entity_added() {
        let a = SerializedScene::new();
        let mut b = SerializedScene::new();
        b.add_entity(SerializedEntity::new(0, 0));

        let diff = SceneDiff::compute(&a, &b);
        assert_eq!(diff.len(), 1);
        assert!(matches!(&diff.changes[0], DiffChange::EntityAdded { entity_id: 0 }));
    }

    #[test]
    fn scene_diff_entity_removed() {
        let mut a = SerializedScene::new();
        a.add_entity(SerializedEntity::new(0, 0));
        let b = SerializedScene::new();

        let diff = SceneDiff::compute(&a, &b);
        assert_eq!(diff.len(), 1);
        assert!(matches!(&diff.changes[0], DiffChange::EntityRemoved { entity_id: 0 }));
    }

    #[test]
    fn scene_diff_component_added() {
        let mut a = SerializedScene::new();
        a.add_entity(SerializedEntity::new(0, 0));

        let mut b = SerializedScene::new();
        let mut e = SerializedEntity::new(0, 0);
        let mut pos = SerializedComponent::new("Position");
        pos.set_field("x", SerializedValue::Float(1.0));
        e.add_component(pos);
        b.add_entity(e);

        let diff = SceneDiff::compute(&a, &b);
        assert!(diff
            .changes
            .iter()
            .any(|c| matches!(c, DiffChange::ComponentAdded { .. })));
    }

    #[test]
    fn scene_diff_field_changed() {
        let mut a = SerializedScene::new();
        let mut e_a = SerializedEntity::new(0, 0);
        let mut pos_a = SerializedComponent::new("Position");
        pos_a.set_field("x", SerializedValue::Float(1.0));
        e_a.add_component(pos_a);
        a.add_entity(e_a);

        let mut b = SerializedScene::new();
        let mut e_b = SerializedEntity::new(0, 0);
        let mut pos_b = SerializedComponent::new("Position");
        pos_b.set_field("x", SerializedValue::Float(5.0));
        e_b.add_component(pos_b);
        b.add_entity(e_b);

        let diff = SceneDiff::compute(&a, &b);
        let field_changes: Vec<_> = diff
            .changes
            .iter()
            .filter(|c| matches!(c, DiffChange::FieldChanged { .. }))
            .collect();
        assert_eq!(field_changes.len(), 1);
    }

    #[test]
    fn scene_diff_parent_changed() {
        let mut a = SerializedScene::new();
        let mut e_a = SerializedEntity::new(0, 0);
        e_a.parent = None;
        a.add_entity(e_a);

        let mut b = SerializedScene::new();
        let mut e_b = SerializedEntity::new(0, 0);
        e_b.parent = Some(5);
        b.add_entity(e_b);

        let diff = SceneDiff::compute(&a, &b);
        assert!(diff
            .changes
            .iter()
            .any(|c| matches!(c, DiffChange::ParentChanged { .. })));
    }

    #[test]
    fn scene_diff_apply_forward() {
        let mut scene = SerializedScene::new();
        scene.add_entity(SerializedEntity::new(0, 0));

        let mut after = SerializedScene::new();
        let mut e = SerializedEntity::new(0, 0);
        let mut pos = SerializedComponent::new("Position");
        pos.set_field("x", SerializedValue::Float(10.0));
        e.add_component(pos);
        after.add_entity(e);
        after.add_entity(SerializedEntity::new(1, 0));

        let diff = SceneDiff::compute(&scene, &after);
        diff.apply_forward(&mut scene);

        assert_eq!(scene.entity_count(), 2);
        assert!(scene.find_entity(0).unwrap().has_component("Position"));
    }

    #[test]
    fn scene_diff_apply_reverse() {
        let mut original = SerializedScene::new();
        let mut e = SerializedEntity::new(0, 0);
        let mut pos = SerializedComponent::new("Position");
        pos.set_field("x", SerializedValue::Float(1.0));
        e.add_component(pos);
        original.add_entity(e);

        let mut modified = SerializedScene::new();
        let mut e2 = SerializedEntity::new(0, 0);
        let mut pos2 = SerializedComponent::new("Position");
        pos2.set_field("x", SerializedValue::Float(99.0));
        e2.add_component(pos2);
        modified.add_entity(e2);

        let diff = SceneDiff::compute(&original, &modified);

        // Apply forward first.
        let mut scene = original.clone();
        diff.apply_forward(&mut scene);

        // Then reverse to get back to original.
        diff.apply_reverse(&mut scene);

        let restored_x = scene
            .find_entity(0)
            .unwrap()
            .get_component("Position")
            .unwrap()
            .get_field("x")
            .unwrap()
            .as_float()
            .unwrap();
        assert!((restored_x - 1.0).abs() < 0.001);
    }

    #[test]
    fn scene_diff_resource_changes() {
        let mut a = SerializedScene::new();
        let mut time_a = SerializedComponent::new("TimeOfDay");
        time_a.set_field("hours", SerializedValue::Float(10.0));
        a.add_resource(time_a);

        let mut b = SerializedScene::new();
        let mut time_b = SerializedComponent::new("TimeOfDay");
        time_b.set_field("hours", SerializedValue::Float(14.0));
        b.add_resource(time_b);

        let diff = SceneDiff::compute(&a, &b);
        assert!(diff
            .changes
            .iter()
            .any(|c| matches!(c, DiffChange::ResourceFieldChanged { .. })));
    }

    #[test]
    fn scene_diff_changes_for_entity() {
        let a = SerializedScene::new();
        let mut b = SerializedScene::new();
        b.add_entity(SerializedEntity::new(0, 0));
        b.add_entity(SerializedEntity::new(1, 0));

        let diff = SceneDiff::compute(&a, &b);
        let e0_changes = diff.changes_for_entity(0);
        assert_eq!(e0_changes.len(), 1);
        assert!(diff.entity_affected(0));
        assert!(diff.entity_affected(1));
    }

    #[test]
    fn json_parser_primitives() {
        assert_eq!(parse_json("null"), Some(SerializedValue::Null));
        assert_eq!(parse_json("true"), Some(SerializedValue::Bool(true)));
        assert_eq!(parse_json("false"), Some(SerializedValue::Bool(false)));
        assert_eq!(parse_json("42"), Some(SerializedValue::Integer(42)));
        assert_eq!(parse_json("3.14"), Some(SerializedValue::Float(3.14)));
        assert_eq!(
            parse_json("\"hello\""),
            Some(SerializedValue::String("hello".to_string()))
        );
    }

    #[test]
    fn json_parser_object() {
        let json = r#"{"name": "test", "value": 42}"#;
        let parsed = parse_json(json).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("name").unwrap().as_str(), Some("test"));
        assert_eq!(obj.get("value").unwrap().as_integer(), Some(42));
    }

    #[test]
    fn json_parser_array() {
        let json = "[1, 2, 3]";
        let parsed = parse_json(json).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn json_parser_nested() {
        let json = r#"{"entities": [{"id": 0, "name": "test"}]}"#;
        let parsed = parse_json(json).unwrap();
        let obj = parsed.as_object().unwrap();
        let entities = obj.get("entities").unwrap().as_array().unwrap();
        assert_eq!(entities.len(), 1);
    }

    #[test]
    fn json_parser_escaped_string() {
        let json = r#""hello\nworld""#;
        let parsed = parse_json(json).unwrap();
        assert_eq!(parsed.as_str(), Some("hello\nworld"));
    }

    #[test]
    fn json_parser_negative_number() {
        let json = "-42";
        let parsed = parse_json(json).unwrap();
        assert_eq!(parsed.as_integer(), Some(-42));
    }

    #[test]
    fn json_parser_empty_structures() {
        assert_eq!(
            parse_json("{}"),
            Some(SerializedValue::Object(HashMap::new()))
        );
        assert_eq!(
            parse_json("[]"),
            Some(SerializedValue::Array(Vec::new()))
        );
    }

    #[test]
    fn serialized_value_to_json_roundtrip() {
        let mut obj = HashMap::new();
        obj.insert("x".to_string(), SerializedValue::Float(1.5));
        obj.insert("name".to_string(), SerializedValue::String("test".into()));
        obj.insert("active".to_string(), SerializedValue::Bool(true));
        let value = SerializedValue::Object(obj);

        let json = value.to_json();
        let parsed = parse_json(&json).unwrap();
        let parsed_obj = parsed.as_object().unwrap();

        assert_eq!(parsed_obj.get("x").unwrap().as_float(), Some(1.5));
        assert_eq!(parsed_obj.get("name").unwrap().as_str(), Some("test"));
        assert_eq!(parsed_obj.get("active").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn full_scene_roundtrip() {
        let mut scene = SerializedScene::new();
        scene.set_metadata("engine", "genovo");
        scene.set_metadata("version", "0.1.0");

        // Create a hierarchy: root -> child1, child2
        let mut root = SerializedEntity::new(0, 0);
        root.name = Some("Root".to_string());
        root.children = vec![1, 2];
        let mut transform = SerializedComponent::new("Transform");
        transform.set_field("x", SerializedValue::Float(0.0));
        transform.set_field("y", SerializedValue::Float(0.0));
        transform.set_field("z", SerializedValue::Float(0.0));
        root.add_component(transform);
        scene.add_entity(root);

        let mut child1 = SerializedEntity::new(1, 0);
        child1.parent = Some(0);
        child1.name = Some("Child1".to_string());
        let mut pos1 = SerializedComponent::new("Position");
        pos1.set_field("x", SerializedValue::Float(10.0));
        pos1.set_field("y", SerializedValue::Float(20.0));
        child1.add_component(pos1);
        let mut hp1 = SerializedComponent::new("Health");
        hp1.set_field("value", SerializedValue::Integer(100));
        child1.add_component(hp1);
        scene.add_entity(child1);

        let mut child2 = SerializedEntity::new(2, 0);
        child2.parent = Some(0);
        child2.name = Some("Child2".to_string());
        child2.enabled = false;
        scene.add_entity(child2);

        let mut time = SerializedComponent::new("GameTime");
        time.set_field("elapsed", SerializedValue::Float(123.456));
        time.set_field("paused", SerializedValue::Bool(false));
        scene.add_resource(time);

        // Serialize.
        let json = SceneSerializer::save(&scene);

        // Deserialize.
        let loaded = SceneSerializer::load(&json).unwrap();

        // Verify.
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.entity_count(), 3);
        assert_eq!(loaded.resource_count(), 1);

        let l_root = loaded.find_entity(0).unwrap();
        assert_eq!(l_root.name.as_deref(), Some("Root"));
        assert_eq!(l_root.children.len(), 2);
        assert!(l_root.has_component("Transform"));

        let l_child1 = loaded.find_entity(1).unwrap();
        assert_eq!(l_child1.parent, Some(0));
        assert_eq!(l_child1.components.len(), 2);

        let l_child2 = loaded.find_entity(2).unwrap();
        assert_eq!(l_child2.enabled, false);

        let l_time = loaded.find_resource("GameTime").unwrap();
        assert_eq!(
            l_time.get_field("paused").unwrap().as_bool(),
            Some(false)
        );

        // Validate.
        let errors = SceneSerializer::validate(&loaded);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }
}
