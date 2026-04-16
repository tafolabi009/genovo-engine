//! Scene file format.
//!
//! Defines a serialisable scene description that stores entity hierarchies,
//! component data, and references to external assets (meshes, materials,
//! textures, etc.).  Scene files are serialised as JSON and loaded via the
//! standard asset loader infrastructure.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::loader::{AssetError, AssetLoader};

// =========================================================================
// Scene data types
// =========================================================================

/// A complete scene description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneAsset {
    /// Version number of the scene format.
    #[serde(default = "default_version")]
    pub version: u32,

    /// Human-readable scene name.
    pub name: String,

    /// All entities in the scene, keyed by UUID.
    pub entities: Vec<SceneEntity>,

    /// Scene-level metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Scene-level settings (e.g. ambient colour, fog, skybox).
    #[serde(default)]
    pub settings: SceneSettings,
}

/// An entity within the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEntity {
    /// Unique identifier for this entity.
    pub id: Uuid,

    /// Optional human-readable name.
    #[serde(default)]
    pub name: Option<String>,

    /// UUID of the parent entity (`None` for root entities).
    #[serde(default)]
    pub parent: Option<Uuid>,

    /// Whether this entity is enabled at load time.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Transform component.
    #[serde(default)]
    pub transform: TransformData,

    /// Components attached to this entity.
    #[serde(default)]
    pub components: Vec<ComponentData>,

    /// Tags for grouping and querying.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Transform component data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformData {
    /// Position in world space (or parent-local space).
    #[serde(default)]
    pub position: [f32; 3],

    /// Rotation as a quaternion [x, y, z, w].
    #[serde(default = "default_identity_quat")]
    pub rotation: [f32; 4],

    /// Scale.
    #[serde(default = "default_one_vec3")]
    pub scale: [f32; 3],
}

impl Default for TransformData {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// A dynamically-typed component attached to an entity.
///
/// Components are identified by a type string and carry their data as
/// a JSON value map.  This allows scene files to store arbitrary
/// component types without requiring compile-time knowledge of every
/// component struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentData {
    /// Component type identifier (e.g. `"MeshRenderer"`, `"PointLight"`,
    /// `"RigidBody"`).
    #[serde(rename = "type")]
    pub type_name: String,

    /// Component data as key-value pairs.
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

/// Scene-wide settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSettings {
    /// Ambient light colour (RGB, 0..1).
    #[serde(default = "default_ambient")]
    pub ambient_color: [f32; 3],

    /// Ambient light intensity.
    #[serde(default = "default_ambient_intensity")]
    pub ambient_intensity: f32,

    /// Skybox texture path.
    #[serde(default)]
    pub skybox: Option<String>,

    /// Whether fog is enabled.
    #[serde(default)]
    pub fog_enabled: bool,

    /// Fog colour (RGB).
    #[serde(default = "default_fog_color")]
    pub fog_color: [f32; 3],

    /// Fog start distance.
    #[serde(default = "default_fog_start")]
    pub fog_start: f32,

    /// Fog end distance.
    #[serde(default = "default_fog_end")]
    pub fog_end: f32,

    /// Gravity vector.
    #[serde(default = "default_gravity")]
    pub gravity: [f32; 3],
}

impl Default for SceneSettings {
    fn default() -> Self {
        Self {
            ambient_color: [0.1, 0.1, 0.15],
            ambient_intensity: 1.0,
            skybox: None,
            fog_enabled: false,
            fog_color: [0.5, 0.5, 0.5],
            fog_start: 50.0,
            fog_end: 200.0,
            gravity: [0.0, -9.81, 0.0],
        }
    }
}

// =========================================================================
// Default value helpers
// =========================================================================

fn default_version() -> u32 {
    1
}
fn default_true() -> bool {
    true
}
fn default_identity_quat() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}
fn default_one_vec3() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_ambient() -> [f32; 3] {
    [0.1, 0.1, 0.15]
}
fn default_ambient_intensity() -> f32 {
    1.0
}
fn default_fog_color() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}
fn default_fog_start() -> f32 {
    50.0
}
fn default_fog_end() -> f32 {
    200.0
}
fn default_gravity() -> [f32; 3] {
    [0.0, -9.81, 0.0]
}

// =========================================================================
// SceneAsset implementation
// =========================================================================

impl SceneAsset {
    /// Creates a new empty scene with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            version: 1,
            name: name.into(),
            entities: Vec::new(),
            metadata: HashMap::new(),
            settings: SceneSettings::default(),
        }
    }

    /// Adds an entity to the scene and returns its UUID.
    pub fn add_entity(&mut self, entity: SceneEntity) -> Uuid {
        let id = entity.id;
        self.entities.push(entity);
        id
    }

    /// Removes an entity by UUID.  Returns the removed entity, or `None`.
    pub fn remove_entity(&mut self, id: &Uuid) -> Option<SceneEntity> {
        let idx = self.entities.iter().position(|e| &e.id == id)?;
        Some(self.entities.remove(idx))
    }

    /// Finds an entity by UUID.
    pub fn find_entity(&self, id: &Uuid) -> Option<&SceneEntity> {
        self.entities.iter().find(|e| &e.id == id)
    }

    /// Finds an entity by name (returns the first match).
    pub fn find_entity_by_name(&self, name: &str) -> Option<&SceneEntity> {
        self.entities
            .iter()
            .find(|e| e.name.as_deref() == Some(name))
    }

    /// Returns all root entities (those with no parent).
    pub fn root_entities(&self) -> Vec<&SceneEntity> {
        self.entities
            .iter()
            .filter(|e| e.parent.is_none())
            .collect()
    }

    /// Returns all children of the given parent entity.
    pub fn children_of(&self, parent_id: &Uuid) -> Vec<&SceneEntity> {
        self.entities
            .iter()
            .filter(|e| e.parent.as_ref() == Some(parent_id))
            .collect()
    }

    /// Returns entities that have a specific tag.
    pub fn entities_with_tag(&self, tag: &str) -> Vec<&SceneEntity> {
        self.entities
            .iter()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Returns entities that have a specific component type.
    pub fn entities_with_component(&self, type_name: &str) -> Vec<&SceneEntity> {
        self.entities
            .iter()
            .filter(|e| e.components.iter().any(|c| c.type_name == type_name))
            .collect()
    }

    /// Collects all external asset paths referenced by entities' components.
    pub fn referenced_assets(&self) -> Vec<String> {
        let mut paths = Vec::new();
        for entity in &self.entities {
            for comp in &entity.components {
                // Look for common asset reference keys
                for key in &["mesh", "material", "texture", "audio", "script", "prefab", "asset"] {
                    if let Some(serde_json::Value::String(path)) = comp.properties.get(*key) {
                        if !paths.contains(path) {
                            paths.push(path.clone());
                        }
                    }
                }
            }
        }
        if let Some(ref skybox) = self.settings.skybox {
            if !paths.contains(skybox) {
                paths.push(skybox.clone());
            }
        }
        paths
    }

    /// Serialise the scene to a JSON string.
    pub fn to_json(&self) -> Result<String, AssetError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| AssetError::Other(format!("Scene serialisation failed: {e}")))
    }

    /// Deserialise a scene from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, AssetError> {
        serde_json::from_str(json)
            .map_err(|e| AssetError::Parse(format!("Scene parse error: {e}")))
    }

    /// Total number of entities in the scene.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

// =========================================================================
// SceneEntity builder
// =========================================================================

impl SceneEntity {
    /// Creates a new entity with a fresh UUID and default transform.
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: None,
            parent: None,
            enabled: true,
            transform: TransformData::default(),
            components: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Creates a new named entity.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: Some(name.into()),
            parent: None,
            enabled: true,
            transform: TransformData::default(),
            components: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Builder: set position.
    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.transform.position = [x, y, z];
        self
    }

    /// Builder: set rotation (quaternion).
    pub fn with_rotation(mut self, x: f32, y: f32, z: f32, w: f32) -> Self {
        self.transform.rotation = [x, y, z, w];
        self
    }

    /// Builder: set scale.
    pub fn with_scale(mut self, x: f32, y: f32, z: f32) -> Self {
        self.transform.scale = [x, y, z];
        self
    }

    /// Builder: set parent.
    pub fn with_parent(mut self, parent_id: Uuid) -> Self {
        self.parent = Some(parent_id);
        self
    }

    /// Adds a component to this entity.
    pub fn add_component(&mut self, component: ComponentData) {
        self.components.push(component);
    }

    /// Finds a component by type name.
    pub fn get_component(&self, type_name: &str) -> Option<&ComponentData> {
        self.components.iter().find(|c| c.type_name == type_name)
    }

    /// Removes a component by type name.  Returns `true` if removed.
    pub fn remove_component(&mut self, type_name: &str) -> bool {
        if let Some(idx) = self.components.iter().position(|c| c.type_name == type_name) {
            self.components.remove(idx);
            true
        } else {
            false
        }
    }

    /// Adds a tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Returns `true` if the entity has the given tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

impl Default for SceneEntity {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// ComponentData builder
// =========================================================================

impl ComponentData {
    /// Creates a new component with the given type name.
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            properties: HashMap::new(),
        }
    }

    /// Sets a string property.
    pub fn set_string(&mut self, key: &str, value: &str) {
        self.properties
            .insert(key.to_owned(), serde_json::Value::String(value.to_owned()));
    }

    /// Sets a float property.
    pub fn set_float(&mut self, key: &str, value: f32) {
        self.properties.insert(
            key.to_owned(),
            serde_json::json!(value),
        );
    }

    /// Sets a boolean property.
    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.properties
            .insert(key.to_owned(), serde_json::Value::Bool(value));
    }

    /// Sets an array-of-floats property.
    pub fn set_vec3(&mut self, key: &str, value: [f32; 3]) {
        self.properties.insert(
            key.to_owned(),
            serde_json::json!(value),
        );
    }

    /// Gets a string property.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.properties.get(key).and_then(|v| v.as_str())
    }

    /// Gets a float property.
    pub fn get_float(&self, key: &str) -> Option<f32> {
        self.properties
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
    }

    /// Gets a bool property.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.properties.get(key).and_then(|v| v.as_bool())
    }
}

// =========================================================================
// SceneLoader
// =========================================================================

/// Loads scene files (`.scene`).
pub struct SceneLoader;

impl AssetLoader for SceneLoader {
    type Asset = SceneAsset;

    fn extensions(&self) -> &[&str] {
        &["scene"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<SceneAsset, AssetError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| AssetError::Parse(format!("Scene file not UTF-8: {e}")))?;
        SceneAsset::from_json(text)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_new() {
        let scene = SceneAsset::new("test_scene");
        assert_eq!(scene.name, "test_scene");
        assert_eq!(scene.version, 1);
        assert!(scene.entities.is_empty());
        assert_eq!(scene.entity_count(), 0);
    }

    #[test]
    fn test_scene_add_remove_entity() {
        let mut scene = SceneAsset::new("test");
        let entity = SceneEntity::named("player");
        let id = entity.id;
        scene.add_entity(entity);

        assert_eq!(scene.entity_count(), 1);
        assert!(scene.find_entity(&id).is_some());

        let removed = scene.remove_entity(&id);
        assert!(removed.is_some());
        assert_eq!(scene.entity_count(), 0);
    }

    #[test]
    fn test_scene_find_by_name() {
        let mut scene = SceneAsset::new("test");
        scene.add_entity(SceneEntity::named("camera"));
        scene.add_entity(SceneEntity::named("light"));

        assert!(scene.find_entity_by_name("camera").is_some());
        assert!(scene.find_entity_by_name("light").is_some());
        assert!(scene.find_entity_by_name("missing").is_none());
    }

    #[test]
    fn test_scene_hierarchy() {
        let mut scene = SceneAsset::new("test");
        let parent = SceneEntity::named("parent");
        let parent_id = parent.id;
        scene.add_entity(parent);

        let child = SceneEntity::named("child").with_parent(parent_id);
        scene.add_entity(child);

        let roots = scene.root_entities();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].name.as_deref(), Some("parent"));

        let children = scene.children_of(&parent_id);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].name.as_deref(), Some("child"));
    }

    #[test]
    fn test_scene_tags() {
        let mut scene = SceneAsset::new("test");
        let mut entity = SceneEntity::named("enemy");
        entity.add_tag("hostile");
        entity.add_tag("npc");
        scene.add_entity(entity);

        let hostile = scene.entities_with_tag("hostile");
        assert_eq!(hostile.len(), 1);

        let friendly = scene.entities_with_tag("friendly");
        assert!(friendly.is_empty());
    }

    #[test]
    fn test_scene_components() {
        let mut scene = SceneAsset::new("test");
        let mut entity = SceneEntity::named("obj");

        let mut mesh_comp = ComponentData::new("MeshRenderer");
        mesh_comp.set_string("mesh", "meshes/cube.obj");
        mesh_comp.set_string("material", "materials/stone.mat");
        entity.add_component(mesh_comp);

        let mut light_comp = ComponentData::new("PointLight");
        light_comp.set_float("intensity", 2.5);
        light_comp.set_vec3("color", [1.0, 0.9, 0.7]);
        light_comp.set_bool("cast_shadows", true);
        entity.add_component(light_comp);

        scene.add_entity(entity);

        let with_mesh = scene.entities_with_component("MeshRenderer");
        assert_eq!(with_mesh.len(), 1);

        let with_light = scene.entities_with_component("PointLight");
        assert_eq!(with_light.len(), 1);

        let comp = with_light[0].get_component("PointLight").unwrap();
        assert_eq!(comp.get_float("intensity"), Some(2.5));
        assert_eq!(comp.get_bool("cast_shadows"), Some(true));
    }

    #[test]
    fn test_scene_referenced_assets() {
        let mut scene = SceneAsset::new("test");
        scene.settings.skybox = Some("textures/sky.hdr".to_owned());

        let mut entity = SceneEntity::named("obj");
        let mut comp = ComponentData::new("MeshRenderer");
        comp.set_string("mesh", "meshes/hero.glb");
        comp.set_string("material", "materials/hero.mat");
        entity.add_component(comp);
        scene.add_entity(entity);

        let assets = scene.referenced_assets();
        assert!(assets.contains(&"meshes/hero.glb".to_owned()));
        assert!(assets.contains(&"materials/hero.mat".to_owned()));
        assert!(assets.contains(&"textures/sky.hdr".to_owned()));
    }

    #[test]
    fn test_scene_json_round_trip() {
        let mut scene = SceneAsset::new("my_level");
        scene.settings.fog_enabled = true;
        scene.settings.fog_color = [0.8, 0.8, 0.9];
        scene
            .metadata
            .insert("author".to_owned(), serde_json::json!("Genovo"));

        let root = SceneEntity::named("world_root")
            .with_position(0.0, 0.0, 0.0)
            .with_scale(1.0, 1.0, 1.0);
        let root_id = root.id;
        scene.add_entity(root);

        let child = SceneEntity::named("player")
            .with_parent(root_id)
            .with_position(10.0, 0.0, 5.0);
        scene.add_entity(child);

        let json = scene.to_json().unwrap();
        let restored = SceneAsset::from_json(&json).unwrap();

        assert_eq!(restored.name, "my_level");
        assert_eq!(restored.entity_count(), 2);
        assert!(restored.settings.fog_enabled);
        assert!((restored.settings.fog_color[0] - 0.8).abs() < 0.001);

        let player = restored.find_entity_by_name("player").unwrap();
        assert!((player.transform.position[0] - 10.0).abs() < 0.001);
        assert_eq!(player.parent, Some(root_id));
    }

    #[test]
    fn test_scene_loader_extensions() {
        let loader = SceneLoader;
        assert_eq!(loader.extensions(), &["scene"]);
    }

    #[test]
    fn test_scene_loader_load() {
        let json = r#"{"name":"test","entities":[]}"#;
        let loader = SceneLoader;
        let scene = loader.load(Path::new("test.scene"), json.as_bytes()).unwrap();
        assert_eq!(scene.name, "test");
        assert!(scene.entities.is_empty());
    }

    #[test]
    fn test_scene_loader_invalid() {
        let loader = SceneLoader;
        assert!(loader.load(Path::new("bad.scene"), b"{{not json").is_err());
    }

    #[test]
    fn test_entity_builder() {
        let e = SceneEntity::named("test")
            .with_position(1.0, 2.0, 3.0)
            .with_rotation(0.0, 0.707, 0.0, 0.707)
            .with_scale(2.0, 2.0, 2.0);

        assert_eq!(e.name.as_deref(), Some("test"));
        assert_eq!(e.transform.position, [1.0, 2.0, 3.0]);
        assert!((e.transform.rotation[1] - 0.707).abs() < 0.001);
        assert_eq!(e.transform.scale, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_entity_component_operations() {
        let mut e = SceneEntity::named("test");
        e.add_component(ComponentData::new("A"));
        e.add_component(ComponentData::new("B"));

        assert!(e.get_component("A").is_some());
        assert!(e.get_component("C").is_none());

        assert!(e.remove_component("A"));
        assert!(e.get_component("A").is_none());
        assert!(!e.remove_component("A")); // already removed
    }

    #[test]
    fn test_entity_tags() {
        let mut e = SceneEntity::named("test");
        e.add_tag("enemy");
        e.add_tag("enemy"); // duplicate ignored
        assert!(e.has_tag("enemy"));
        assert!(!e.has_tag("friend"));
        assert_eq!(e.tags.len(), 1);
    }

    #[test]
    fn test_transform_default() {
        let t = TransformData::default();
        assert_eq!(t.position, [0.0, 0.0, 0.0]);
        assert_eq!(t.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(t.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_scene_settings_default() {
        let s = SceneSettings::default();
        assert!(!s.fog_enabled);
        assert!(s.skybox.is_none());
        assert!((s.gravity[1] - (-9.81)).abs() < 0.01);
    }

    #[test]
    fn test_component_data_accessors() {
        let mut c = ComponentData::new("Test");
        c.set_string("name", "hello");
        c.set_float("speed", 3.14);
        c.set_bool("active", false);
        c.set_vec3("direction", [0.0, 1.0, 0.0]);

        assert_eq!(c.get_string("name"), Some("hello"));
        assert!((c.get_float("speed").unwrap() - 3.14).abs() < 0.01);
        assert_eq!(c.get_bool("active"), Some(false));
        assert_eq!(c.get_string("missing"), None);
    }
}
