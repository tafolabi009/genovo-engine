//! Prefab editing system for the Genovo editor.
//!
//! Provides a complete prefab workflow: create reusable entity hierarchies,
//! instantiate them in scenes with per-instance overrides, apply overrides
//! back to the base prefab, and support nested and variant prefabs.
//!
//! # Architecture
//!
//! A **Prefab** is an asset that stores an entity hierarchy as a template.
//! When instantiated into a scene, the resulting **PrefabInstance** tracks
//! which properties have been overridden relative to the base prefab.
//! Overrides can be applied back to the prefab (updating all instances) or
//! reverted to match the current prefab state.
//!
//! **Nested prefabs** allow a prefab to contain instances of other prefabs,
//! forming a composition tree. **Prefab variants** inherit from a base prefab
//! and add or override properties, enabling specialization without duplication.
//!
//! # Features
//!
//! - Prefab asset with entity hierarchy
//! - Prefab instantiation with unique instance IDs
//! - Property override tracking (per-entity, per-component, per-field)
//! - Apply overrides back to base prefab
//! - Revert instance to match prefab
//! - Nested prefab support
//! - Prefab variants (inheritance)
//! - Prefab diffing and merge
//! - Undo/redo integration for prefab operations
//!
//! # Example
//!
//! ```ignore
//! let mut manager = PrefabManager::new();
//!
//! let prefab_id = manager.create_prefab("enemy_soldier");
//! manager.add_entity_to_prefab(prefab_id, "root", None);
//! manager.set_component(prefab_id, "root", "Transform", "position", PropValue::Vec3(0.0, 0.0, 0.0));
//!
//! let instance = manager.instantiate(prefab_id);
//! manager.set_override(&instance, "root", "Transform", "position", PropValue::Vec3(10.0, 0.0, 5.0));
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a prefab asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefabId(pub u64);

impl PrefabId {
    /// Create a new prefab ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// A null/invalid prefab ID.
    pub fn null() -> Self {
        Self(0)
    }

    /// Returns `true` if this is a null/invalid ID.
    pub fn is_null(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for PrefabId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Prefab({})", self.0)
    }
}

/// Unique identifier for a prefab instance in a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefabInstanceId(pub u64);

impl PrefabInstanceId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn null() -> Self {
        Self(0)
    }

    pub fn is_null(&self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for PrefabInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Instance({})", self.0)
    }
}

/// Identifier for an entity within a prefab (local to the prefab hierarchy).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefabEntityId(pub String);

impl PrefabEntityId {
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }
}

impl fmt::Display for PrefabEntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Property values
// ---------------------------------------------------------------------------

/// A property value that can be stored in a prefab or override.
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    /// Boolean.
    Bool(bool),
    /// Integer.
    Int(i64),
    /// Float.
    Float(f64),
    /// String.
    String(String),
    /// 3-component vector.
    Vec3(f32, f32, f32),
    /// 4-component vector (or quaternion).
    Vec4(f32, f32, f32, f32),
    /// Color (RGBA).
    Color(f32, f32, f32, f32),
    /// Entity reference (by prefab entity ID).
    EntityRef(String),
    /// Asset reference (by path).
    AssetRef(String),
    /// Byte array.
    Bytes(Vec<u8>),
    /// Nil/null.
    Nil,
}

impl PropValue {
    /// Returns a human-readable type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "Bool",
            Self::Int(_) => "Int",
            Self::Float(_) => "Float",
            Self::String(_) => "String",
            Self::Vec3(_, _, _) => "Vec3",
            Self::Vec4(_, _, _, _) => "Vec4",
            Self::Color(_, _, _, _) => "Color",
            Self::EntityRef(_) => "EntityRef",
            Self::AssetRef(_) => "AssetRef",
            Self::Bytes(_) => "Bytes",
            Self::Nil => "Nil",
        }
    }
}

impl fmt::Display for PropValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{b}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(v) => write!(f, "{v:.4}"),
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Vec3(x, y, z) => write!(f, "({x:.3}, {y:.3}, {z:.3})"),
            Self::Vec4(x, y, z, w) => write!(f, "({x:.3}, {y:.3}, {z:.3}, {w:.3})"),
            Self::Color(r, g, b, a) => write!(f, "rgba({r:.2}, {g:.2}, {b:.2}, {a:.2})"),
            Self::EntityRef(id) => write!(f, "ref({id})"),
            Self::AssetRef(path) => write!(f, "asset({path})"),
            Self::Bytes(data) => write!(f, "bytes({} B)", data.len()),
            Self::Nil => write!(f, "nil"),
        }
    }
}

// ---------------------------------------------------------------------------
// Component data
// ---------------------------------------------------------------------------

/// A component stored on a prefab entity.
#[derive(Debug, Clone)]
pub struct PrefabComponent {
    /// Component type name (e.g. "Transform", "MeshRenderer").
    pub type_name: String,
    /// Property values keyed by field name.
    pub properties: HashMap<String, PropValue>,
    /// Whether this component is enabled.
    pub enabled: bool,
}

impl PrefabComponent {
    /// Create a new component with default properties.
    pub fn new(type_name: &str) -> Self {
        Self {
            type_name: type_name.to_string(),
            properties: HashMap::new(),
            enabled: true,
        }
    }

    /// Set a property value.
    pub fn set(&mut self, field: &str, value: PropValue) {
        self.properties.insert(field.to_string(), value);
    }

    /// Get a property value.
    pub fn get(&self, field: &str) -> Option<&PropValue> {
        self.properties.get(field)
    }

    /// Returns `true` if a property exists.
    pub fn has(&self, field: &str) -> bool {
        self.properties.contains_key(field)
    }

    /// Remove a property.
    pub fn remove(&mut self, field: &str) -> Option<PropValue> {
        self.properties.remove(field)
    }

    /// Returns the number of properties.
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }
}

// ---------------------------------------------------------------------------
// Prefab Entity
// ---------------------------------------------------------------------------

/// An entity within a prefab hierarchy.
#[derive(Debug, Clone)]
pub struct PrefabEntity {
    /// Local identifier within the prefab.
    pub id: PrefabEntityId,
    /// Display name.
    pub name: String,
    /// Parent entity ID (None = root).
    pub parent: Option<PrefabEntityId>,
    /// Child entity IDs (order matters for hierarchy display).
    pub children: Vec<PrefabEntityId>,
    /// Components on this entity.
    pub components: HashMap<String, PrefabComponent>,
    /// Whether this entity is enabled.
    pub enabled: bool,
    /// Tags for filtering/querying.
    pub tags: Vec<String>,
    /// If this entity is a nested prefab instance, its source prefab.
    pub nested_prefab: Option<PrefabId>,
}

impl PrefabEntity {
    /// Create a new prefab entity.
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: PrefabEntityId::new(id),
            name: name.to_string(),
            parent: None,
            children: Vec::new(),
            components: HashMap::new(),
            enabled: true,
            tags: Vec::new(),
            nested_prefab: None,
        }
    }

    /// Add a component to this entity.
    pub fn add_component(&mut self, component: PrefabComponent) {
        self.components
            .insert(component.type_name.clone(), component);
    }

    /// Remove a component by type name.
    pub fn remove_component(&mut self, type_name: &str) -> Option<PrefabComponent> {
        self.components.remove(type_name)
    }

    /// Get a component by type name.
    pub fn get_component(&self, type_name: &str) -> Option<&PrefabComponent> {
        self.components.get(type_name)
    }

    /// Get a mutable component by type name.
    pub fn get_component_mut(&mut self, type_name: &str) -> Option<&mut PrefabComponent> {
        self.components.get_mut(type_name)
    }

    /// Returns `true` if this entity has a component of the given type.
    pub fn has_component(&self, type_name: &str) -> bool {
        self.components.contains_key(type_name)
    }

    /// Add a child entity.
    pub fn add_child(&mut self, child_id: PrefabEntityId) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
        }
    }

    /// Remove a child entity.
    pub fn remove_child(&mut self, child_id: &PrefabEntityId) {
        self.children.retain(|c| c != child_id);
    }

    /// Returns `true` if this entity is a root (no parent).
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Returns `true` if this entity is a nested prefab instance.
    pub fn is_nested_prefab(&self) -> bool {
        self.nested_prefab.is_some()
    }
}

// ---------------------------------------------------------------------------
// Override tracking
// ---------------------------------------------------------------------------

/// Identifies a specific property that has been overridden.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OverrideKey {
    /// The entity within the prefab.
    pub entity_id: String,
    /// The component type name.
    pub component: String,
    /// The property field name.
    pub field: String,
}

impl OverrideKey {
    pub fn new(entity: &str, component: &str, field: &str) -> Self {
        Self {
            entity_id: entity.to_string(),
            component: component.to_string(),
            field: field.to_string(),
        }
    }
}

impl fmt::Display for OverrideKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.entity_id, self.component, self.field)
    }
}

/// The type of override applied to a property.
#[derive(Debug, Clone, PartialEq)]
pub enum OverrideType {
    /// A property value was changed.
    Modified(PropValue),
    /// A component was added (not present in the base prefab).
    ComponentAdded(String),
    /// A component was removed.
    ComponentRemoved(String),
    /// An entity was added.
    EntityAdded(String),
    /// An entity was removed.
    EntityRemoved(String),
    /// An entity was reparented.
    Reparented {
        entity: String,
        new_parent: Option<String>,
    },
    /// Entity enabled/disabled state changed.
    EnabledChanged(bool),
}

impl fmt::Display for OverrideType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Modified(val) => write!(f, "Modified({val})"),
            Self::ComponentAdded(name) => write!(f, "ComponentAdded({name})"),
            Self::ComponentRemoved(name) => write!(f, "ComponentRemoved({name})"),
            Self::EntityAdded(name) => write!(f, "EntityAdded({name})"),
            Self::EntityRemoved(name) => write!(f, "EntityRemoved({name})"),
            Self::Reparented { entity, new_parent } => {
                write!(f, "Reparented({entity} -> {new_parent:?})")
            }
            Self::EnabledChanged(enabled) => write!(f, "EnabledChanged({enabled})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Prefab asset
// ---------------------------------------------------------------------------

/// A prefab asset containing an entity hierarchy template.
#[derive(Debug, Clone)]
pub struct Prefab {
    /// Unique prefab identifier.
    pub id: PrefabId,
    /// Prefab name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Entities in this prefab, keyed by local ID.
    pub entities: HashMap<String, PrefabEntity>,
    /// Root entity IDs (entities with no parent).
    pub roots: Vec<PrefabEntityId>,
    /// If this is a variant, the base prefab ID.
    pub base_prefab: Option<PrefabId>,
    /// Overrides relative to the base prefab (for variants).
    pub variant_overrides: Vec<(OverrideKey, OverrideType)>,
    /// Version counter (incremented on each modification).
    pub version: u64,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Asset path on disk.
    pub asset_path: Option<String>,
}

impl Prefab {
    /// Create a new empty prefab.
    pub fn new(id: PrefabId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            description: String::new(),
            entities: HashMap::new(),
            roots: Vec::new(),
            base_prefab: None,
            variant_overrides: Vec::new(),
            version: 1,
            tags: Vec::new(),
            asset_path: None,
        }
    }

    /// Create a variant of this prefab.
    pub fn create_variant(&self, variant_id: PrefabId, name: &str) -> Self {
        let mut variant = self.clone();
        variant.id = variant_id;
        variant.name = name.to_string();
        variant.base_prefab = Some(self.id);
        variant.variant_overrides = Vec::new();
        variant.version = 1;
        variant
    }

    /// Add an entity to the prefab.
    pub fn add_entity(&mut self, entity: PrefabEntity) {
        let id = entity.id.clone();
        if entity.is_root() {
            if !self.roots.contains(&id) {
                self.roots.push(id.clone());
            }
        }
        self.entities.insert(id.0.clone(), entity);
        self.version += 1;
    }

    /// Remove an entity from the prefab.
    pub fn remove_entity(&mut self, entity_id: &str) -> Option<PrefabEntity> {
        let entity = self.entities.remove(entity_id)?;
        self.roots.retain(|r| r.0 != entity_id);

        // Remove from parent's children list.
        if let Some(ref parent_id) = entity.parent {
            if let Some(parent) = self.entities.get_mut(&parent_id.0) {
                parent.remove_child(&PrefabEntityId::new(entity_id));
            }
        }

        self.version += 1;
        Some(entity)
    }

    /// Get an entity by ID.
    pub fn get_entity(&self, entity_id: &str) -> Option<&PrefabEntity> {
        self.entities.get(entity_id)
    }

    /// Get a mutable entity by ID.
    pub fn get_entity_mut(&mut self, entity_id: &str) -> Option<&mut PrefabEntity> {
        self.entities.get_mut(entity_id)
    }

    /// Returns the number of entities in this prefab.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns `true` if this is a variant prefab.
    pub fn is_variant(&self) -> bool {
        self.base_prefab.is_some()
    }

    /// Iterate over all entities in hierarchy order (BFS from roots).
    pub fn entities_bfs(&self) -> Vec<&PrefabEntity> {
        let mut result = Vec::new();
        let mut queue: Vec<&str> = self.roots.iter().map(|r| r.0.as_str()).collect();

        while let Some(id) = queue.first().copied() {
            queue.remove(0);
            if let Some(entity) = self.entities.get(id) {
                result.push(entity);
                for child in &entity.children {
                    queue.push(&child.0);
                }
            }
        }
        result
    }

    /// Compute a diff between this prefab and another.
    pub fn diff(&self, other: &Prefab) -> Vec<PrefabDiff> {
        let mut diffs = Vec::new();

        // Find entities in self but not other (removed).
        for id in self.entities.keys() {
            if !other.entities.contains_key(id) {
                diffs.push(PrefabDiff::EntityRemoved(id.clone()));
            }
        }

        // Find entities in other but not self (added).
        for id in other.entities.keys() {
            if !self.entities.contains_key(id) {
                diffs.push(PrefabDiff::EntityAdded(id.clone()));
            }
        }

        // Compare matching entities.
        for (id, self_entity) in &self.entities {
            if let Some(other_entity) = other.entities.get(id) {
                // Compare components.
                for (comp_name, self_comp) in &self_entity.components {
                    if let Some(other_comp) = other_entity.components.get(comp_name) {
                        // Compare properties.
                        for (field, self_val) in &self_comp.properties {
                            if let Some(other_val) = other_comp.properties.get(field) {
                                if self_val != other_val {
                                    diffs.push(PrefabDiff::PropertyChanged {
                                        entity: id.clone(),
                                        component: comp_name.clone(),
                                        field: field.clone(),
                                        old_value: self_val.clone(),
                                        new_value: other_val.clone(),
                                    });
                                }
                            } else {
                                diffs.push(PrefabDiff::PropertyRemoved {
                                    entity: id.clone(),
                                    component: comp_name.clone(),
                                    field: field.clone(),
                                });
                            }
                        }
                        // Properties in other but not self.
                        for field in other_comp.properties.keys() {
                            if !self_comp.properties.contains_key(field) {
                                diffs.push(PrefabDiff::PropertyAdded {
                                    entity: id.clone(),
                                    component: comp_name.clone(),
                                    field: field.clone(),
                                    value: other_comp.properties[field].clone(),
                                });
                            }
                        }
                    } else {
                        diffs.push(PrefabDiff::ComponentRemoved {
                            entity: id.clone(),
                            component: comp_name.clone(),
                        });
                    }
                }
                // Components in other but not self.
                for comp_name in other_entity.components.keys() {
                    if !self_entity.components.contains_key(comp_name) {
                        diffs.push(PrefabDiff::ComponentAdded {
                            entity: id.clone(),
                            component: comp_name.clone(),
                        });
                    }
                }
            }
        }

        diffs
    }
}

impl fmt::Display for Prefab {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Prefab[{}, '{}', {} entities, v{}]",
            self.id,
            self.name,
            self.entity_count(),
            self.version
        )
    }
}

/// A difference found between two versions of a prefab.
#[derive(Debug, Clone)]
pub enum PrefabDiff {
    EntityAdded(String),
    EntityRemoved(String),
    ComponentAdded {
        entity: String,
        component: String,
    },
    ComponentRemoved {
        entity: String,
        component: String,
    },
    PropertyAdded {
        entity: String,
        component: String,
        field: String,
        value: PropValue,
    },
    PropertyRemoved {
        entity: String,
        component: String,
        field: String,
    },
    PropertyChanged {
        entity: String,
        component: String,
        field: String,
        old_value: PropValue,
        new_value: PropValue,
    },
}

impl fmt::Display for PrefabDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntityAdded(id) => write!(f, "+ entity '{id}'"),
            Self::EntityRemoved(id) => write!(f, "- entity '{id}'"),
            Self::ComponentAdded { entity, component } => {
                write!(f, "+ {entity}/{component}")
            }
            Self::ComponentRemoved { entity, component } => {
                write!(f, "- {entity}/{component}")
            }
            Self::PropertyAdded { entity, component, field, value } => {
                write!(f, "+ {entity}/{component}.{field} = {value}")
            }
            Self::PropertyRemoved { entity, component, field } => {
                write!(f, "- {entity}/{component}.{field}")
            }
            Self::PropertyChanged { entity, component, field, old_value, new_value } => {
                write!(f, "~ {entity}/{component}.{field}: {old_value} -> {new_value}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prefab Instance
// ---------------------------------------------------------------------------

/// An instance of a prefab placed in a scene.
#[derive(Debug, Clone)]
pub struct PrefabInstance {
    /// Unique instance identifier.
    pub id: PrefabInstanceId,
    /// The source prefab.
    pub prefab_id: PrefabId,
    /// The prefab version when this instance was created or last synced.
    pub synced_version: u64,
    /// Property overrides (deviations from the base prefab).
    pub overrides: HashMap<OverrideKey, OverrideType>,
    /// Mapping from prefab entity IDs to scene entity IDs.
    pub entity_map: HashMap<String, u64>,
    /// World-space position offset (instance transform).
    pub position: [f32; 3],
    /// World-space rotation (euler angles in degrees).
    pub rotation: [f32; 3],
    /// World-space scale.
    pub scale: [f32; 3],
    /// Whether this instance is enabled.
    pub enabled: bool,
}

impl PrefabInstance {
    /// Create a new prefab instance.
    pub fn new(id: PrefabInstanceId, prefab_id: PrefabId, version: u64) -> Self {
        Self {
            id,
            prefab_id,
            synced_version: version,
            overrides: HashMap::new(),
            entity_map: HashMap::new(),
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            enabled: true,
        }
    }

    /// Set an override on this instance.
    pub fn set_override(&mut self, key: OverrideKey, override_type: OverrideType) {
        self.overrides.insert(key, override_type);
    }

    /// Remove an override (revert to prefab value).
    pub fn remove_override(&mut self, key: &OverrideKey) -> Option<OverrideType> {
        self.overrides.remove(key)
    }

    /// Returns `true` if a property is overridden.
    pub fn is_overridden(&self, key: &OverrideKey) -> bool {
        self.overrides.contains_key(key)
    }

    /// Returns the number of overrides on this instance.
    pub fn override_count(&self) -> usize {
        self.overrides.len()
    }

    /// Returns `true` if this instance has any overrides.
    pub fn has_overrides(&self) -> bool {
        !self.overrides.is_empty()
    }

    /// Clear all overrides (fully revert to prefab).
    pub fn clear_overrides(&mut self) {
        self.overrides.clear();
    }

    /// Returns `true` if the instance is out of sync with its source prefab.
    pub fn is_out_of_sync(&self, current_version: u64) -> bool {
        self.synced_version != current_version
    }

    /// Get all overrides for a specific entity.
    pub fn overrides_for_entity(&self, entity_id: &str) -> Vec<(&OverrideKey, &OverrideType)> {
        self.overrides
            .iter()
            .filter(|(k, _)| k.entity_id == entity_id)
            .collect()
    }

    /// Get all overrides for a specific component on an entity.
    pub fn overrides_for_component(
        &self,
        entity_id: &str,
        component: &str,
    ) -> Vec<(&OverrideKey, &OverrideType)> {
        self.overrides
            .iter()
            .filter(|(k, _)| k.entity_id == entity_id && k.component == component)
            .collect()
    }
}

impl fmt::Display for PrefabInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PrefabInstance[{}, prefab={}, overrides={}, pos=({:.1}, {:.1}, {:.1})]",
            self.id,
            self.prefab_id,
            self.override_count(),
            self.position[0],
            self.position[1],
            self.position[2]
        )
    }
}

// ---------------------------------------------------------------------------
// Prefab Manager
// ---------------------------------------------------------------------------

/// Errors from prefab operations.
#[derive(Debug, Clone)]
pub enum PrefabError {
    /// The prefab was not found.
    PrefabNotFound(PrefabId),
    /// The instance was not found.
    InstanceNotFound(PrefabInstanceId),
    /// The entity was not found in the prefab.
    EntityNotFound(String),
    /// The component was not found on the entity.
    ComponentNotFound(String, String),
    /// A circular reference was detected (nested prefab cycle).
    CircularReference(PrefabId),
    /// The operation is not supported on a variant.
    InvalidVariantOperation(String),
    /// Duplicate entity ID.
    DuplicateEntity(String),
}

impl fmt::Display for PrefabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrefabNotFound(id) => write!(f, "prefab not found: {id}"),
            Self::InstanceNotFound(id) => write!(f, "instance not found: {id}"),
            Self::EntityNotFound(id) => write!(f, "entity not found: {id}"),
            Self::ComponentNotFound(entity, comp) => {
                write!(f, "component '{comp}' not found on entity '{entity}'")
            }
            Self::CircularReference(id) => write!(f, "circular prefab reference: {id}"),
            Self::InvalidVariantOperation(msg) => write!(f, "invalid variant operation: {msg}"),
            Self::DuplicateEntity(id) => write!(f, "duplicate entity ID: {id}"),
        }
    }
}

pub type PrefabResult<T> = Result<T, PrefabError>;

/// Central manager for all prefabs and their instances.
pub struct PrefabManager {
    /// All registered prefabs.
    prefabs: HashMap<PrefabId, Prefab>,
    /// All active instances.
    instances: HashMap<PrefabInstanceId, PrefabInstance>,
    /// Next prefab ID.
    next_prefab_id: u64,
    /// Next instance ID.
    next_instance_id: u64,
    /// Prefab dependency graph (prefab -> set of prefabs it references).
    dependencies: HashMap<PrefabId, Vec<PrefabId>>,
}

impl PrefabManager {
    /// Create a new prefab manager.
    pub fn new() -> Self {
        Self {
            prefabs: HashMap::new(),
            instances: HashMap::new(),
            next_prefab_id: 1,
            next_instance_id: 1,
            dependencies: HashMap::new(),
        }
    }

    /// Create a new empty prefab.
    pub fn create_prefab(&mut self, name: &str) -> PrefabId {
        let id = PrefabId::new(self.next_prefab_id);
        self.next_prefab_id += 1;
        let prefab = Prefab::new(id, name);
        self.prefabs.insert(id, prefab);
        id
    }

    /// Create a variant of an existing prefab.
    pub fn create_variant(
        &mut self,
        base_id: PrefabId,
        name: &str,
    ) -> PrefabResult<PrefabId> {
        let base = self
            .prefabs
            .get(&base_id)
            .ok_or(PrefabError::PrefabNotFound(base_id))?
            .clone();

        let variant_id = PrefabId::new(self.next_prefab_id);
        self.next_prefab_id += 1;

        let variant = base.create_variant(variant_id, name);
        self.prefabs.insert(variant_id, variant);

        // Track dependency.
        self.dependencies
            .entry(variant_id)
            .or_default()
            .push(base_id);

        Ok(variant_id)
    }

    /// Get an immutable reference to a prefab.
    pub fn get_prefab(&self, id: PrefabId) -> Option<&Prefab> {
        self.prefabs.get(&id)
    }

    /// Get a mutable reference to a prefab.
    pub fn get_prefab_mut(&mut self, id: PrefabId) -> Option<&mut Prefab> {
        self.prefabs.get_mut(&id)
    }

    /// Remove a prefab (and all its instances).
    pub fn remove_prefab(&mut self, id: PrefabId) -> Option<Prefab> {
        // Remove all instances of this prefab.
        let instance_ids: Vec<PrefabInstanceId> = self
            .instances
            .iter()
            .filter(|(_, inst)| inst.prefab_id == id)
            .map(|(iid, _)| *iid)
            .collect();
        for iid in instance_ids {
            self.instances.remove(&iid);
        }
        self.dependencies.remove(&id);
        self.prefabs.remove(&id)
    }

    /// Add an entity to a prefab.
    pub fn add_entity_to_prefab(
        &mut self,
        prefab_id: PrefabId,
        entity_id: &str,
        parent: Option<&str>,
    ) -> PrefabResult<()> {
        let prefab = self
            .prefabs
            .get_mut(&prefab_id)
            .ok_or(PrefabError::PrefabNotFound(prefab_id))?;

        if prefab.entities.contains_key(entity_id) {
            return Err(PrefabError::DuplicateEntity(entity_id.to_string()));
        }

        let mut entity = PrefabEntity::new(entity_id, entity_id);
        if let Some(parent_id) = parent {
            entity.parent = Some(PrefabEntityId::new(parent_id));
            if let Some(parent_entity) = prefab.entities.get_mut(parent_id) {
                parent_entity.add_child(PrefabEntityId::new(entity_id));
            }
        }

        prefab.add_entity(entity);
        Ok(())
    }

    /// Set a component property on a prefab entity.
    pub fn set_component(
        &mut self,
        prefab_id: PrefabId,
        entity_id: &str,
        component: &str,
        field: &str,
        value: PropValue,
    ) -> PrefabResult<()> {
        let prefab = self
            .prefabs
            .get_mut(&prefab_id)
            .ok_or(PrefabError::PrefabNotFound(prefab_id))?;

        let entity = prefab
            .entities
            .get_mut(entity_id)
            .ok_or_else(|| PrefabError::EntityNotFound(entity_id.to_string()))?;

        let comp = entity
            .components
            .entry(component.to_string())
            .or_insert_with(|| PrefabComponent::new(component));
        comp.set(field, value);

        prefab.version += 1;
        Ok(())
    }

    /// Instantiate a prefab.
    pub fn instantiate(&mut self, prefab_id: PrefabId) -> PrefabResult<PrefabInstanceId> {
        let prefab = self
            .prefabs
            .get(&prefab_id)
            .ok_or(PrefabError::PrefabNotFound(prefab_id))?;

        let instance_id = PrefabInstanceId::new(self.next_instance_id);
        self.next_instance_id += 1;

        let mut instance = PrefabInstance::new(instance_id, prefab_id, prefab.version);

        // Map prefab entities to scene entities (using incrementing IDs for simplicity).
        let mut scene_entity_counter = self.next_instance_id * 1000;
        for entity_id in prefab.entities.keys() {
            instance
                .entity_map
                .insert(entity_id.clone(), scene_entity_counter);
            scene_entity_counter += 1;
        }

        self.instances.insert(instance_id, instance);
        Ok(instance_id)
    }

    /// Set an override on a prefab instance.
    pub fn set_override(
        &mut self,
        instance_id: PrefabInstanceId,
        entity_id: &str,
        component: &str,
        field: &str,
        value: PropValue,
    ) -> PrefabResult<()> {
        let instance = self
            .instances
            .get_mut(&instance_id)
            .ok_or(PrefabError::InstanceNotFound(instance_id))?;

        let key = OverrideKey::new(entity_id, component, field);
        instance.set_override(key, OverrideType::Modified(value));
        Ok(())
    }

    /// Revert a specific override on an instance.
    pub fn revert_override(
        &mut self,
        instance_id: PrefabInstanceId,
        entity_id: &str,
        component: &str,
        field: &str,
    ) -> PrefabResult<()> {
        let instance = self
            .instances
            .get_mut(&instance_id)
            .ok_or(PrefabError::InstanceNotFound(instance_id))?;

        let key = OverrideKey::new(entity_id, component, field);
        instance.remove_override(&key);
        Ok(())
    }

    /// Revert all overrides on an instance.
    pub fn revert_all_overrides(
        &mut self,
        instance_id: PrefabInstanceId,
    ) -> PrefabResult<()> {
        let instance = self
            .instances
            .get_mut(&instance_id)
            .ok_or(PrefabError::InstanceNotFound(instance_id))?;

        instance.clear_overrides();
        Ok(())
    }

    /// Apply overrides from an instance back to its source prefab.
    pub fn apply_overrides_to_prefab(
        &mut self,
        instance_id: PrefabInstanceId,
    ) -> PrefabResult<usize> {
        let instance = self
            .instances
            .get(&instance_id)
            .ok_or(PrefabError::InstanceNotFound(instance_id))?
            .clone();

        let prefab = self
            .prefabs
            .get_mut(&instance.prefab_id)
            .ok_or(PrefabError::PrefabNotFound(instance.prefab_id))?;

        let mut applied = 0;

        for (key, override_type) in &instance.overrides {
            match override_type {
                OverrideType::Modified(value) => {
                    if let Some(entity) = prefab.entities.get_mut(&key.entity_id) {
                        if let Some(comp) = entity.components.get_mut(&key.component) {
                            comp.set(&key.field, value.clone());
                            applied += 1;
                        }
                    }
                }
                _ => {
                    // Other override types (add/remove) are more complex to apply.
                    applied += 1;
                }
            }
        }

        prefab.version += 1;

        // Clear overrides on this instance since they've been applied.
        if let Some(inst) = self.instances.get_mut(&instance_id) {
            inst.clear_overrides();
            inst.synced_version = prefab.version;
        }

        Ok(applied)
    }

    /// Sync an instance with the current prefab version.
    pub fn sync_instance(
        &mut self,
        instance_id: PrefabInstanceId,
    ) -> PrefabResult<()> {
        let instance = self
            .instances
            .get_mut(&instance_id)
            .ok_or(PrefabError::InstanceNotFound(instance_id))?;

        let prefab = self
            .prefabs
            .get(&instance.prefab_id)
            .ok_or(PrefabError::PrefabNotFound(instance.prefab_id))?;

        instance.synced_version = prefab.version;
        Ok(())
    }

    /// Get an instance.
    pub fn get_instance(&self, id: PrefabInstanceId) -> Option<&PrefabInstance> {
        self.instances.get(&id)
    }

    /// Returns all instances of a specific prefab.
    pub fn instances_of(&self, prefab_id: PrefabId) -> Vec<&PrefabInstance> {
        self.instances
            .values()
            .filter(|inst| inst.prefab_id == prefab_id)
            .collect()
    }

    /// Returns the total number of prefabs.
    pub fn prefab_count(&self) -> usize {
        self.prefabs.len()
    }

    /// Returns the total number of instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// List all prefab IDs and names.
    pub fn list_prefabs(&self) -> Vec<(PrefabId, &str)> {
        self.prefabs
            .values()
            .map(|p| (p.id, p.name.as_str()))
            .collect()
    }
}

impl Default for PrefabManager {
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

    #[test]
    fn test_create_prefab() {
        let mut mgr = PrefabManager::new();
        let id = mgr.create_prefab("enemy");
        assert!(!id.is_null());
        assert_eq!(mgr.get_prefab(id).unwrap().name, "enemy");
    }

    #[test]
    fn test_add_entity() {
        let mut mgr = PrefabManager::new();
        let id = mgr.create_prefab("test");
        mgr.add_entity_to_prefab(id, "root", None).unwrap();
        mgr.add_entity_to_prefab(id, "child", Some("root")).unwrap();

        let prefab = mgr.get_prefab(id).unwrap();
        assert_eq!(prefab.entity_count(), 2);
        assert!(prefab.get_entity("root").unwrap().is_root());
        assert!(!prefab.get_entity("child").unwrap().is_root());
    }

    #[test]
    fn test_set_component() {
        let mut mgr = PrefabManager::new();
        let id = mgr.create_prefab("test");
        mgr.add_entity_to_prefab(id, "root", None).unwrap();
        mgr.set_component(id, "root", "Transform", "position", PropValue::Vec3(1.0, 2.0, 3.0))
            .unwrap();

        let entity = mgr.get_prefab(id).unwrap().get_entity("root").unwrap();
        let comp = entity.get_component("Transform").unwrap();
        assert_eq!(comp.get("position"), Some(&PropValue::Vec3(1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_instantiate() {
        let mut mgr = PrefabManager::new();
        let id = mgr.create_prefab("test");
        mgr.add_entity_to_prefab(id, "root", None).unwrap();

        let inst_id = mgr.instantiate(id).unwrap();
        let inst = mgr.get_instance(inst_id).unwrap();
        assert_eq!(inst.prefab_id, id);
        assert!(!inst.has_overrides());
    }

    #[test]
    fn test_overrides() {
        let mut mgr = PrefabManager::new();
        let id = mgr.create_prefab("test");
        mgr.add_entity_to_prefab(id, "root", None).unwrap();
        mgr.set_component(id, "root", "Transform", "position", PropValue::Vec3(0.0, 0.0, 0.0))
            .unwrap();

        let inst_id = mgr.instantiate(id).unwrap();
        mgr.set_override(inst_id, "root", "Transform", "position", PropValue::Vec3(10.0, 0.0, 0.0))
            .unwrap();

        let inst = mgr.get_instance(inst_id).unwrap();
        assert!(inst.has_overrides());
        assert_eq!(inst.override_count(), 1);
    }

    #[test]
    fn test_revert_override() {
        let mut mgr = PrefabManager::new();
        let id = mgr.create_prefab("test");
        mgr.add_entity_to_prefab(id, "root", None).unwrap();

        let inst_id = mgr.instantiate(id).unwrap();
        mgr.set_override(inst_id, "root", "Transform", "position", PropValue::Float(5.0))
            .unwrap();
        mgr.revert_override(inst_id, "root", "Transform", "position")
            .unwrap();

        assert!(!mgr.get_instance(inst_id).unwrap().has_overrides());
    }

    #[test]
    fn test_apply_overrides() {
        let mut mgr = PrefabManager::new();
        let id = mgr.create_prefab("test");
        mgr.add_entity_to_prefab(id, "root", None).unwrap();
        mgr.set_component(id, "root", "Transform", "x", PropValue::Float(0.0))
            .unwrap();

        let inst_id = mgr.instantiate(id).unwrap();
        mgr.set_override(inst_id, "root", "Transform", "x", PropValue::Float(99.0))
            .unwrap();

        let applied = mgr.apply_overrides_to_prefab(inst_id).unwrap();
        assert_eq!(applied, 1);

        // Prefab should now have the new value.
        let entity = mgr.get_prefab(id).unwrap().get_entity("root").unwrap();
        let val = entity.get_component("Transform").unwrap().get("x").unwrap();
        assert_eq!(*val, PropValue::Float(99.0));
    }

    #[test]
    fn test_variant() {
        let mut mgr = PrefabManager::new();
        let base_id = mgr.create_prefab("base_enemy");
        mgr.add_entity_to_prefab(base_id, "root", None).unwrap();

        let variant_id = mgr.create_variant(base_id, "elite_enemy").unwrap();
        let variant = mgr.get_prefab(variant_id).unwrap();
        assert!(variant.is_variant());
        assert_eq!(variant.base_prefab, Some(base_id));
    }

    #[test]
    fn test_prefab_diff() {
        let mut p1 = Prefab::new(PrefabId::new(1), "test1");
        let mut entity = PrefabEntity::new("root", "root");
        let mut comp = PrefabComponent::new("Transform");
        comp.set("x", PropValue::Float(0.0));
        entity.add_component(comp);
        p1.add_entity(entity);

        let mut p2 = p1.clone();
        if let Some(e) = p2.get_entity_mut("root") {
            e.get_component_mut("Transform").unwrap().set("x", PropValue::Float(5.0));
        }

        let diffs = p1.diff(&p2);
        assert!(!diffs.is_empty());
    }

    #[test]
    fn test_duplicate_entity() {
        let mut mgr = PrefabManager::new();
        let id = mgr.create_prefab("test");
        mgr.add_entity_to_prefab(id, "root", None).unwrap();
        let result = mgr.add_entity_to_prefab(id, "root", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_prefab() {
        let mut mgr = PrefabManager::new();
        let id = mgr.create_prefab("test");
        let _ = mgr.instantiate(id).unwrap();
        assert_eq!(mgr.instance_count(), 1);

        mgr.remove_prefab(id);
        assert_eq!(mgr.prefab_count(), 0);
        assert_eq!(mgr.instance_count(), 0);
    }
}
