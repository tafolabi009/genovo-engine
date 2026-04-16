//! # Project Management
//!
//! Handles project-level configuration, serialization, and deserialization of
//! Genovo game projects. A project bundles engine settings, scene references,
//! input mappings, build configuration, and open-scene tracking.
//!
//! Projects are serialized in JSON format (RON planned for future).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Project
// ---------------------------------------------------------------------------

/// Represents a Genovo game project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    /// Unique project identifier.
    pub id: Uuid,
    /// Human-readable project name.
    pub name: String,
    /// Root directory of the project on disk.
    pub root_path: PathBuf,
    /// Project-wide settings.
    pub settings: ProjectSettings,
    /// Path to the default/startup scene.
    pub startup_scene: Option<PathBuf>,
    /// All scene files in the project (relative paths).
    pub scenes: Vec<PathBuf>,
    /// Currently open scenes.
    #[serde(skip)]
    pub open_scenes: Vec<SceneFile>,
    /// Project file format version (for migration).
    pub version: u32,
    /// Custom metadata (user-defined key-value pairs).
    pub metadata: HashMap<String, String>,
}

/// Current version of the project format.
const CURRENT_PROJECT_VERSION: u32 = 2;

impl Project {
    /// Create a new project at the given root path.
    pub fn new(name: String, root_path: PathBuf) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            root_path,
            settings: ProjectSettings::default(),
            startup_scene: None,
            scenes: Vec::new(),
            open_scenes: Vec::new(),
            version: CURRENT_PROJECT_VERSION,
            metadata: HashMap::new(),
        }
    }

    /// Load a project from a `.genovo` project file.
    pub fn load(path: &Path) -> Result<Self, ProjectError> {
        let content =
            std::fs::read_to_string(path).map_err(ProjectError::Io)?;

        let mut project: Project =
            serde_json::from_str(&content).map_err(|e| ProjectError::Serialization(e.to_string()))?;

        // Version check and migration.
        if project.version > CURRENT_PROJECT_VERSION {
            return Err(ProjectError::VersionMismatch {
                found: project.version,
                supported: CURRENT_PROJECT_VERSION,
            });
        }

        if project.version < CURRENT_PROJECT_VERSION {
            project = Self::migrate(project)?;
        }

        // If root_path was relative, resolve it against the project file location.
        if project.root_path.is_relative() {
            if let Some(parent) = path.parent() {
                project.root_path = parent.join(&project.root_path);
            }
        }

        log::info!("Loaded project '{}' from {:?}", project.name, path);
        Ok(project)
    }

    /// Migrate a project from an older format version to the current one.
    fn migrate(mut project: Project) -> Result<Project, ProjectError> {
        // Version 1 -> 2: add metadata field (serde default handles it).
        if project.version < 2 {
            project.version = 2;
            log::info!(
                "Migrated project '{}' from v1 to v{}",
                project.name,
                CURRENT_PROJECT_VERSION,
            );
        }

        Ok(project)
    }

    /// Save the project to its default project file (`<name>.genovo`).
    pub fn save(&self) -> Result<PathBuf, ProjectError> {
        let path = self.project_file_path();
        self.save_to(&path)?;
        Ok(path)
    }

    /// Save the project to a specific path.
    pub fn save_to(&self, path: &Path) -> Result<(), ProjectError> {
        // Ensure directory exists.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ProjectError::Io)?;
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| ProjectError::Serialization(e.to_string()))?;

        std::fs::write(path, json).map_err(ProjectError::Io)?;
        log::info!("Saved project '{}' to {:?}", self.name, path);
        Ok(())
    }

    /// Return the default project file path.
    pub fn project_file_path(&self) -> PathBuf {
        self.root_path.join(format!("{}.genovo", self.name))
    }

    /// Return the assets directory for this project.
    pub fn assets_dir(&self) -> PathBuf {
        self.root_path.join("assets")
    }

    /// Return the build output directory.
    pub fn build_dir(&self) -> PathBuf {
        self.root_path.join("build")
    }

    /// Return the cache directory.
    pub fn cache_dir(&self) -> PathBuf {
        self.root_path.join(".cache")
    }

    /// Add a scene file to the project.
    pub fn add_scene(&mut self, scene_path: PathBuf) {
        if !self.scenes.contains(&scene_path) {
            self.scenes.push(scene_path);
        }
    }

    /// Remove a scene file from the project.
    pub fn remove_scene(&mut self, scene_path: &Path) {
        self.scenes.retain(|p| p != scene_path);
        if self.startup_scene.as_deref() == Some(scene_path) {
            self.startup_scene = None;
        }
    }

    /// Set the startup scene.
    pub fn set_startup_scene(&mut self, scene_path: PathBuf) {
        self.add_scene(scene_path.clone());
        self.startup_scene = Some(scene_path);
    }

    /// Open a scene for editing. Returns the index into `open_scenes`.
    pub fn open_scene(&mut self, path: PathBuf) -> Result<usize, ProjectError> {
        // Check if already open (compare by the relative path we store).
        if let Some(idx) = self.open_scenes.iter().position(|s| s.path == path) {
            return Ok(idx);
        }

        let abs_path = if path.is_relative() {
            self.root_path.join(&path)
        } else {
            path.clone()
        };

        let mut scene = if abs_path.exists() {
            SceneFile::load(&abs_path)?
        } else {
            SceneFile::new(path.clone())
        };

        // Ensure the scene stores the relative path so duplicate detection works.
        scene.path = path.clone();

        self.open_scenes.push(scene);
        self.add_scene(path);
        Ok(self.open_scenes.len() - 1)
    }

    /// Close a scene by index.
    pub fn close_scene(&mut self, index: usize) -> Option<SceneFile> {
        if index < self.open_scenes.len() {
            Some(self.open_scenes.remove(index))
        } else {
            None
        }
    }

    /// Save all open scenes.
    pub fn save_all_scenes(&mut self) -> Result<(), ProjectError> {
        for scene in &mut self.open_scenes {
            let abs_path = self.root_path.join(&scene.path);
            scene.save_to(&abs_path)?;
        }
        Ok(())
    }

    /// Ensure the basic project directory structure exists.
    pub fn ensure_directories(&self) -> Result<(), ProjectError> {
        let dirs = [
            self.assets_dir(),
            self.build_dir(),
            self.cache_dir(),
            self.root_path.join("scenes"),
        ];
        for dir in &dirs {
            std::fs::create_dir_all(dir).map_err(ProjectError::Io)?;
        }
        Ok(())
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// ProjectSettings
// ---------------------------------------------------------------------------

/// Top-level project settings governing engine behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSettings {
    /// Render settings.
    pub render: RenderSettings,
    /// Physics settings.
    pub physics: PhysicsSettings,
    /// Audio settings.
    pub audio: AudioSettings,
    /// Input mapping configuration.
    pub input: InputSettings,
    /// Target platforms for builds.
    pub target_platforms: Vec<TargetPlatform>,
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            render: RenderSettings::default(),
            physics: PhysicsSettings::default(),
            audio: AudioSettings::default(),
            input: InputSettings::default(),
            target_platforms: vec![TargetPlatform::Windows],
        }
    }
}

impl ProjectSettings {
    /// Validate settings and return a list of warnings.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.render.window_width == 0 || self.render.window_height == 0 {
            warnings.push("Window dimensions cannot be zero".into());
        }

        if self.physics.fixed_timestep <= 0.0 {
            warnings.push("Physics timestep must be positive".into());
        }

        if self.physics.max_substeps == 0 {
            warnings.push("Max physics substeps must be at least 1".into());
        }

        if self.audio.master_volume < 0.0 || self.audio.master_volume > 1.0 {
            warnings.push("Master volume should be between 0.0 and 1.0".into());
        }

        if self.target_platforms.is_empty() {
            warnings.push("No target platforms configured".into());
        }

        warnings
    }

    /// Reset all settings to defaults.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// Render Settings
// ---------------------------------------------------------------------------

/// Render configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderSettings {
    /// Default window width.
    pub window_width: u32,
    /// Default window height.
    pub window_height: u32,
    /// Enable VSync.
    pub vsync: bool,
    /// Target frames per second (0 = uncapped).
    pub target_fps: u32,
    /// HDR rendering enabled.
    pub hdr: bool,
    /// Shadow quality level.
    pub shadow_quality: QualityLevel,
    /// Anti-aliasing mode.
    pub anti_aliasing: AntiAliasingMode,
    /// Ambient occlusion.
    pub ambient_occlusion: bool,
    /// Screen-space reflections.
    pub ssr: bool,
    /// Bloom effect.
    pub bloom: bool,
    /// Bloom intensity (0.0 to 1.0).
    pub bloom_intensity: f32,
    /// Gamma correction value.
    pub gamma: f32,
    /// Render scale (1.0 = native resolution).
    pub render_scale: f32,
    /// Maximum shadow cascade count.
    pub shadow_cascade_count: u32,
    /// Shadow map resolution.
    pub shadow_map_resolution: u32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            window_width: 1920,
            window_height: 1080,
            vsync: true,
            target_fps: 0,
            hdr: true,
            shadow_quality: QualityLevel::High,
            anti_aliasing: AntiAliasingMode::Taa,
            ambient_occlusion: true,
            ssr: false,
            bloom: true,
            bloom_intensity: 0.5,
            gamma: 2.2,
            render_scale: 1.0,
            shadow_cascade_count: 4,
            shadow_map_resolution: 2048,
        }
    }
}

impl RenderSettings {
    /// Get the effective resolution (after render scale).
    pub fn effective_resolution(&self) -> (u32, u32) {
        (
            (self.window_width as f32 * self.render_scale) as u32,
            (self.window_height as f32 * self.render_scale) as u32,
        )
    }

    /// Get the aspect ratio.
    pub fn aspect_ratio(&self) -> f32 {
        self.window_width as f32 / self.window_height.max(1) as f32
    }
}

// ---------------------------------------------------------------------------
// Physics Settings
// ---------------------------------------------------------------------------

/// Physics simulation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsSettings {
    /// Gravity vector.
    pub gravity: [f32; 3],
    /// Fixed timestep for physics simulation (seconds).
    pub fixed_timestep: f32,
    /// Maximum number of physics sub-steps per frame.
    pub max_substeps: u32,
    /// Physics backend to use.
    pub backend: PhysicsBackendChoice,
    /// Velocity solver iterations.
    pub solver_iterations: u32,
    /// Position solver iterations.
    pub position_iterations: u32,
    /// Default friction coefficient.
    pub default_friction: f32,
    /// Default restitution (bounciness).
    pub default_restitution: f32,
    /// Whether to enable continuous collision detection.
    pub continuous_collision: bool,
    /// Sleep threshold (linear velocity below which bodies are put to sleep).
    pub sleep_threshold: f32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.81, 0.0],
            fixed_timestep: 1.0 / 60.0,
            max_substeps: 4,
            backend: PhysicsBackendChoice::Builtin,
            solver_iterations: 8,
            position_iterations: 3,
            default_friction: 0.5,
            default_restitution: 0.3,
            continuous_collision: true,
            sleep_threshold: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Audio Settings
// ---------------------------------------------------------------------------

/// Audio configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSettings {
    /// Master volume (0.0 to 1.0).
    pub master_volume: f32,
    /// Music volume multiplier.
    pub music_volume: f32,
    /// Sound effects volume multiplier.
    pub sfx_volume: f32,
    /// Voice volume multiplier.
    pub voice_volume: f32,
    /// Maximum simultaneous audio sources.
    pub max_channels: u32,
    /// Audio sample rate.
    pub sample_rate: u32,
    /// Spatial audio doppler factor.
    pub doppler_factor: f32,
    /// Speed of sound for spatial audio (meters/second).
    pub speed_of_sound: f32,
    /// Distance model for spatial audio attenuation.
    pub distance_model: DistanceModel,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            master_volume: 1.0,
            music_volume: 0.8,
            sfx_volume: 1.0,
            voice_volume: 1.0,
            max_channels: 64,
            sample_rate: 44100,
            doppler_factor: 1.0,
            speed_of_sound: 343.0,
            distance_model: DistanceModel::InverseDistance,
        }
    }
}

/// Spatial audio distance attenuation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceModel {
    /// No attenuation.
    None,
    /// 1 / distance falloff.
    InverseDistance,
    /// Linear falloff between min/max distance.
    Linear,
    /// Exponential falloff.
    Exponential,
}

// ---------------------------------------------------------------------------
// Input Settings
// ---------------------------------------------------------------------------

/// Input mapping configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSettings {
    /// Named input actions with their key/button bindings.
    pub action_mappings: Vec<ActionMapping>,
    /// Named input axes with their key/button bindings.
    pub axis_mappings: Vec<AxisMapping>,
    /// Mouse sensitivity multiplier.
    pub mouse_sensitivity: f32,
    /// Gamepad dead zone.
    pub gamepad_deadzone: f32,
    /// Whether to invert Y axis for mouse look.
    pub invert_y: bool,
}

impl Default for InputSettings {
    fn default() -> Self {
        Self {
            action_mappings: vec![
                ActionMapping {
                    name: "Jump".into(),
                    bindings: vec!["Space".into(), "GamepadA".into()],
                },
                ActionMapping {
                    name: "Fire".into(),
                    bindings: vec!["MouseLeft".into(), "GamepadRightTrigger".into()],
                },
                ActionMapping {
                    name: "Interact".into(),
                    bindings: vec!["E".into(), "GamepadX".into()],
                },
            ],
            axis_mappings: vec![
                AxisMapping {
                    name: "MoveForward".into(),
                    bindings: vec![
                        AxisBinding { input: "W".into(), scale: 1.0 },
                        AxisBinding { input: "S".into(), scale: -1.0 },
                        AxisBinding { input: "GamepadLeftStickY".into(), scale: 1.0 },
                    ],
                },
                AxisMapping {
                    name: "MoveRight".into(),
                    bindings: vec![
                        AxisBinding { input: "D".into(), scale: 1.0 },
                        AxisBinding { input: "A".into(), scale: -1.0 },
                        AxisBinding { input: "GamepadLeftStickX".into(), scale: 1.0 },
                    ],
                },
                AxisMapping {
                    name: "LookUp".into(),
                    bindings: vec![
                        AxisBinding { input: "MouseY".into(), scale: -1.0 },
                        AxisBinding { input: "GamepadRightStickY".into(), scale: 1.0 },
                    ],
                },
                AxisMapping {
                    name: "LookRight".into(),
                    bindings: vec![
                        AxisBinding { input: "MouseX".into(), scale: 1.0 },
                        AxisBinding { input: "GamepadRightStickX".into(), scale: 1.0 },
                    ],
                },
            ],
            mouse_sensitivity: 1.0,
            gamepad_deadzone: 0.15,
            invert_y: false,
        }
    }
}

impl InputSettings {
    /// Find an action mapping by name.
    pub fn find_action(&self, name: &str) -> Option<&ActionMapping> {
        self.action_mappings.iter().find(|a| a.name == name)
    }

    /// Find an axis mapping by name.
    pub fn find_axis(&self, name: &str) -> Option<&AxisMapping> {
        self.axis_mappings.iter().find(|a| a.name == name)
    }

    /// Add or update an action mapping.
    pub fn set_action(&mut self, name: impl Into<String>, bindings: Vec<String>) {
        let n: String = name.into();
        if let Some(action) = self.action_mappings.iter_mut().find(|a| a.name == n) {
            action.bindings = bindings;
        } else {
            self.action_mappings.push(ActionMapping {
                name: n,
                bindings,
            });
        }
    }

    /// Add or update an axis mapping.
    pub fn set_axis(&mut self, name: impl Into<String>, bindings: Vec<AxisBinding>) {
        let n: String = name.into();
        if let Some(axis) = self.axis_mappings.iter_mut().find(|a| a.name == n) {
            axis.bindings = bindings;
        } else {
            self.axis_mappings.push(AxisMapping {
                name: n,
                bindings,
            });
        }
    }

    /// Remove an action mapping by name.
    pub fn remove_action(&mut self, name: &str) {
        self.action_mappings.retain(|a| a.name != name);
    }

    /// Remove an axis mapping by name.
    pub fn remove_axis(&mut self, name: &str) {
        self.axis_mappings.retain(|a| a.name != name);
    }
}

/// A named action (digital input) mapped to one or more keys/buttons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionMapping {
    pub name: String,
    pub bindings: Vec<String>,
}

/// A named axis (analog input) mapped to keys/axes with scale factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisMapping {
    pub name: String,
    pub bindings: Vec<AxisBinding>,
}

/// A single axis binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisBinding {
    pub input: String,
    pub scale: f32,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Quality levels used across various settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityLevel {
    Low,
    Medium,
    High,
    Ultra,
}

impl QualityLevel {
    /// Return all quality levels.
    pub fn all() -> &'static [QualityLevel] {
        &[Self::Low, Self::Medium, Self::High, Self::Ultra]
    }

    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Ultra => "Ultra",
        }
    }
}

/// Anti-aliasing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AntiAliasingMode {
    None,
    Fxaa,
    Smaa,
    Taa,
    Msaa4x,
    Msaa8x,
}

impl AntiAliasingMode {
    /// Return all AA modes.
    pub fn all() -> &'static [AntiAliasingMode] {
        &[Self::None, Self::Fxaa, Self::Smaa, Self::Taa, Self::Msaa4x, Self::Msaa8x]
    }

    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Fxaa => "FXAA",
            Self::Smaa => "SMAA",
            Self::Taa => "TAA",
            Self::Msaa4x => "MSAA 4x",
            Self::Msaa8x => "MSAA 8x",
        }
    }
}

/// Supported target platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetPlatform {
    Windows,
    MacOS,
    Linux,
    Ios,
    Android,
    Xbox,
    PlayStation,
    NintendoSwitch,
    Web,
}

impl TargetPlatform {
    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Windows => "Windows",
            Self::MacOS => "macOS",
            Self::Linux => "Linux",
            Self::Ios => "iOS",
            Self::Android => "Android",
            Self::Xbox => "Xbox",
            Self::PlayStation => "PlayStation",
            Self::NintendoSwitch => "Nintendo Switch",
            Self::Web => "Web",
        }
    }

    /// All platforms.
    pub fn all() -> &'static [TargetPlatform] {
        &[
            Self::Windows, Self::MacOS, Self::Linux, Self::Ios, Self::Android,
            Self::Xbox, Self::PlayStation, Self::NintendoSwitch, Self::Web,
        ]
    }
}

/// Physics backend choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsBackendChoice {
    /// Built-in Rust physics engine.
    Builtin,
    /// NVIDIA PhysX via FFI.
    PhysX,
}

// ---------------------------------------------------------------------------
// SceneFile
// ---------------------------------------------------------------------------

/// Represents a scene being edited, with its serialized entity data
/// and dirty-tracking.
#[derive(Debug, Clone)]
pub struct SceneFile {
    /// Path relative to the project root.
    pub path: PathBuf,
    /// Display name (usually derived from filename).
    pub name: String,
    /// Serialized entity data (JSON string).
    pub data: String,
    /// Whether the scene has unsaved changes.
    pub dirty: bool,
    /// List of entity snapshots for serialization.
    pub entities: Vec<SerializedEntity>,
}

/// A serialized representation of a single entity in a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedEntity {
    /// Entity UUID.
    pub id: Uuid,
    /// Entity name.
    pub name: String,
    /// Parent entity UUID (None for root entities).
    pub parent: Option<Uuid>,
    /// Serialized component data (component_type -> JSON value).
    pub components: HashMap<String, serde_json::Value>,
    /// Whether this entity is active.
    pub active: bool,
}

impl SerializedEntity {
    /// Create a new serialized entity.
    pub fn new(id: Uuid, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            parent: None,
            components: HashMap::new(),
            active: true,
        }
    }

    /// Add a component to the serialized entity.
    pub fn add_component(&mut self, type_name: impl Into<String>, data: serde_json::Value) {
        self.components.insert(type_name.into(), data);
    }

    /// Get a component's data by type name.
    pub fn get_component(&self, type_name: &str) -> Option<&serde_json::Value> {
        self.components.get(type_name)
    }
}

impl SceneFile {
    /// Create a new, empty scene file.
    pub fn new(path: PathBuf) -> Self {
        let name = path
            .file_stem()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Untitled".to_string());
        Self {
            path,
            name,
            data: String::new(),
            dirty: false,
            entities: Vec::new(),
        }
    }

    /// Load a scene from disk.
    pub fn load(path: &Path) -> Result<Self, ProjectError> {
        let content =
            std::fs::read_to_string(path).map_err(ProjectError::Io)?;

        let entities: Vec<SerializedEntity> = serde_json::from_str(&content)
            .map_err(|e| ProjectError::Serialization(e.to_string()))?;

        let name = path
            .file_stem()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Untitled".to_string());

        let rel_path = path.to_path_buf();

        Ok(Self {
            path: rel_path,
            name,
            data: content,
            dirty: false,
            entities,
        })
    }

    /// Save the scene to its default path.
    pub fn save_to(&mut self, abs_path: &Path) -> Result<(), ProjectError> {
        if let Some(parent) = abs_path.parent() {
            std::fs::create_dir_all(parent).map_err(ProjectError::Io)?;
        }

        let json = serde_json::to_string_pretty(&self.entities)
            .map_err(|e| ProjectError::Serialization(e.to_string()))?;

        std::fs::write(abs_path, &json).map_err(ProjectError::Io)?;
        self.data = json;
        self.dirty = false;

        log::info!("Saved scene '{}' to {:?}", self.name, abs_path);
        Ok(())
    }

    /// Serialize the current scene state to a JSON string.
    pub fn serialize(&self) -> Result<String, ProjectError> {
        serde_json::to_string_pretty(&self.entities)
            .map_err(|e| ProjectError::Serialization(e.to_string()))
    }

    /// Deserialize entities from a JSON string into this scene.
    pub fn deserialize(&mut self, data: &str) -> Result<(), ProjectError> {
        let entities: Vec<SerializedEntity> = serde_json::from_str(data)
            .map_err(|e| ProjectError::Serialization(e.to_string()))?;
        self.entities = entities;
        self.data = data.to_string();
        Ok(())
    }

    /// Mark the scene as having unsaved changes.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Add an entity to the scene.
    pub fn add_entity(&mut self, entity: SerializedEntity) {
        self.entities.push(entity);
        self.dirty = true;
    }

    /// Remove an entity from the scene by ID.
    pub fn remove_entity(&mut self, id: Uuid) {
        self.entities.retain(|e| e.id != id);
        // Also remove entities whose parent was the removed entity.
        self.entities.retain(|e| e.parent != Some(id));
        self.dirty = true;
    }

    /// Find an entity by ID.
    pub fn find_entity(&self, id: Uuid) -> Option<&SerializedEntity> {
        self.entities.iter().find(|e| e.id == id)
    }

    /// Find a mutable entity by ID.
    pub fn find_entity_mut(&mut self, id: Uuid) -> Option<&mut SerializedEntity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    /// Get the root entities (those with no parent).
    pub fn root_entities(&self) -> Vec<&SerializedEntity> {
        self.entities.iter().filter(|e| e.parent.is_none()).collect()
    }

    /// Get the children of an entity.
    pub fn children_of(&self, id: Uuid) -> Vec<&SerializedEntity> {
        self.entities.iter().filter(|e| e.parent == Some(id)).collect()
    }

    /// Total entity count.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

// ---------------------------------------------------------------------------
// Scene Serialization (standalone functions)
// ---------------------------------------------------------------------------

/// Serialize a list of entities to a JSON string.
pub fn serialize_scene_to_json(entities: &[SerializedEntity]) -> Result<String, ProjectError> {
    serde_json::to_string_pretty(entities)
        .map_err(|e| ProjectError::Serialization(e.to_string()))
}

/// Deserialize entities from a JSON string.
pub fn deserialize_scene_from_json(json: &str) -> Result<Vec<SerializedEntity>, ProjectError> {
    serde_json::from_str(json).map_err(|e| ProjectError::Serialization(e.to_string()))
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors related to project management.
#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Invalid project format: {0}")]
    InvalidFormat(String),
    #[error("Scene not found: {0}")]
    SceneNotFound(String),
    #[error("Version mismatch: project v{found}, engine supports up to v{supported}")]
    VersionMismatch { found: u32, supported: u32 },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("genovo_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn project_new() {
        let project = Project::new("TestGame".into(), PathBuf::from("/games/test"));
        assert_eq!(project.name, "TestGame");
        assert_eq!(project.version, CURRENT_PROJECT_VERSION);
        assert!(project.scenes.is_empty());
        assert!(project.startup_scene.is_none());
    }

    #[test]
    fn project_add_remove_scene() {
        let mut project = Project::new("Test".into(), PathBuf::from("/games/test"));
        project.add_scene(PathBuf::from("scenes/level1.scene"));
        project.add_scene(PathBuf::from("scenes/level2.scene"));
        assert_eq!(project.scenes.len(), 2);

        // Duplicate add should not increase count.
        project.add_scene(PathBuf::from("scenes/level1.scene"));
        assert_eq!(project.scenes.len(), 2);

        project.remove_scene(Path::new("scenes/level1.scene"));
        assert_eq!(project.scenes.len(), 1);
    }

    #[test]
    fn project_set_startup_scene() {
        let mut project = Project::new("Test".into(), PathBuf::from("/games/test"));
        project.set_startup_scene(PathBuf::from("scenes/main.scene"));
        assert_eq!(
            project.startup_scene.as_deref(),
            Some(Path::new("scenes/main.scene")),
        );
        assert!(project.scenes.contains(&PathBuf::from("scenes/main.scene")));
    }

    #[test]
    fn project_save_and_load() {
        let dir = temp_dir();
        let project = Project::new("SaveTest".into(), dir.clone());

        let save_path = project.save().unwrap();
        assert!(save_path.exists());

        let loaded = Project::load(&save_path).unwrap();
        assert_eq!(loaded.name, "SaveTest");
        assert_eq!(loaded.id, project.id);

        cleanup(&dir);
    }

    #[test]
    fn project_metadata() {
        let mut project = Project::new("Meta".into(), PathBuf::from("/test"));
        project.set_metadata("author", "Jane Doe");
        project.set_metadata("version", "1.0.0");
        assert_eq!(project.get_metadata("author"), Some("Jane Doe"));
        assert_eq!(project.get_metadata("version"), Some("1.0.0"));
        assert_eq!(project.get_metadata("missing"), None);
    }

    #[test]
    fn project_directories() {
        let project = Project::new("Test".into(), PathBuf::from("/games/test"));
        assert_eq!(project.assets_dir(), PathBuf::from("/games/test/assets"));
        assert_eq!(project.build_dir(), PathBuf::from("/games/test/build"));
        assert_eq!(project.cache_dir(), PathBuf::from("/games/test/.cache"));
    }

    #[test]
    fn project_ensure_directories() {
        let dir = temp_dir();
        let project = Project::new("DirTest".into(), dir.clone());
        project.ensure_directories().unwrap();
        assert!(project.assets_dir().exists());
        assert!(project.build_dir().exists());
        assert!(project.cache_dir().exists());
        cleanup(&dir);
    }

    #[test]
    fn project_settings_validate() {
        let settings = ProjectSettings::default();
        let warnings = settings.validate();
        assert!(warnings.is_empty());

        let mut bad_settings = ProjectSettings::default();
        bad_settings.render.window_width = 0;
        bad_settings.physics.fixed_timestep = 0.0;
        bad_settings.target_platforms.clear();
        let warnings = bad_settings.validate();
        assert!(warnings.len() >= 3);
    }

    #[test]
    fn render_settings_effective_resolution() {
        let mut rs = RenderSettings::default();
        rs.window_width = 1920;
        rs.window_height = 1080;
        rs.render_scale = 0.5;
        let (w, h) = rs.effective_resolution();
        assert_eq!(w, 960);
        assert_eq!(h, 540);
    }

    #[test]
    fn render_settings_aspect_ratio() {
        let rs = RenderSettings::default();
        let ar = rs.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn input_settings_find_action() {
        let settings = InputSettings::default();
        let jump = settings.find_action("Jump");
        assert!(jump.is_some());
        assert!(jump.unwrap().bindings.contains(&"Space".to_string()));
        assert!(settings.find_action("Nonexistent").is_none());
    }

    #[test]
    fn input_settings_set_action() {
        let mut settings = InputSettings::default();
        settings.set_action("Dash", vec!["LeftShift".into()]);
        assert!(settings.find_action("Dash").is_some());

        // Update existing.
        settings.set_action("Dash", vec!["LeftCtrl".into()]);
        let dash = settings.find_action("Dash").unwrap();
        assert_eq!(dash.bindings, vec!["LeftCtrl"]);
    }

    #[test]
    fn input_settings_remove_action() {
        let mut settings = InputSettings::default();
        settings.remove_action("Jump");
        assert!(settings.find_action("Jump").is_none());
    }

    #[test]
    fn scene_file_new() {
        let scene = SceneFile::new(PathBuf::from("scenes/test.scene"));
        assert_eq!(scene.name, "test");
        assert!(!scene.dirty);
        assert!(scene.entities.is_empty());
    }

    #[test]
    fn scene_file_add_remove_entity() {
        let mut scene = SceneFile::new(PathBuf::from("test.scene"));
        let id = Uuid::new_v4();
        scene.add_entity(SerializedEntity::new(id, "Player"));
        assert_eq!(scene.entity_count(), 1);
        assert!(scene.dirty);

        scene.remove_entity(id);
        assert_eq!(scene.entity_count(), 0);
    }

    #[test]
    fn scene_file_find_entity() {
        let mut scene = SceneFile::new(PathBuf::from("test.scene"));
        let id = Uuid::new_v4();
        scene.add_entity(SerializedEntity::new(id, "Entity1"));

        assert!(scene.find_entity(id).is_some());
        assert_eq!(scene.find_entity(id).unwrap().name, "Entity1");
        assert!(scene.find_entity(Uuid::new_v4()).is_none());
    }

    #[test]
    fn scene_file_hierarchy() {
        let mut scene = SceneFile::new(PathBuf::from("test.scene"));
        let parent_id = Uuid::new_v4();
        let child_id = Uuid::new_v4();

        scene.add_entity(SerializedEntity::new(parent_id, "Parent"));
        let mut child = SerializedEntity::new(child_id, "Child");
        child.parent = Some(parent_id);
        scene.add_entity(child);

        let roots = scene.root_entities();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, parent_id);

        let children = scene.children_of(parent_id);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, child_id);
    }

    #[test]
    fn scene_file_serialize_deserialize() {
        let mut scene = SceneFile::new(PathBuf::from("test.scene"));
        let id = Uuid::new_v4();
        let mut entity = SerializedEntity::new(id, "TestEntity");
        entity.add_component(
            "Transform",
            serde_json::json!({ "position": [1.0, 2.0, 3.0] }),
        );
        scene.add_entity(entity);

        let json = scene.serialize().unwrap();
        assert!(json.contains("TestEntity"));

        let mut loaded = SceneFile::new(PathBuf::from("loaded.scene"));
        loaded.deserialize(&json).unwrap();
        assert_eq!(loaded.entity_count(), 1);
        assert_eq!(loaded.entities[0].name, "TestEntity");

        let comp = loaded.entities[0].get_component("Transform").unwrap();
        assert_eq!(comp["position"][0], 1.0);
    }

    #[test]
    fn scene_file_save_and_load() {
        let dir = temp_dir();
        let scene_path = dir.join("test.scene");

        let mut scene = SceneFile::new(PathBuf::from("test.scene"));
        scene.add_entity(SerializedEntity::new(Uuid::new_v4(), "Player"));
        scene.save_to(&scene_path).unwrap();
        assert!(!scene.dirty);

        let loaded = SceneFile::load(&scene_path).unwrap();
        assert_eq!(loaded.entity_count(), 1);
        assert_eq!(loaded.entities[0].name, "Player");

        cleanup(&dir);
    }

    #[test]
    fn serialized_entity_components() {
        let mut entity = SerializedEntity::new(Uuid::new_v4(), "Test");
        entity.add_component("Health", serde_json::json!({ "value": 100 }));
        entity.add_component("Position", serde_json::json!({ "x": 1.0, "y": 2.0 }));

        assert!(entity.get_component("Health").is_some());
        assert_eq!(entity.get_component("Health").unwrap()["value"], 100);
        assert!(entity.get_component("Missing").is_none());
    }

    #[test]
    fn serialize_deserialize_scene_functions() {
        let entities = vec![
            SerializedEntity::new(Uuid::new_v4(), "A"),
            SerializedEntity::new(Uuid::new_v4(), "B"),
        ];

        let json = serialize_scene_to_json(&entities).unwrap();
        let loaded = deserialize_scene_from_json(&json).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "A");
        assert_eq!(loaded[1].name, "B");
    }

    #[test]
    fn quality_level_all() {
        let all = QualityLevel::all();
        assert_eq!(all.len(), 4);
        assert_eq!(QualityLevel::High.display_name(), "High");
    }

    #[test]
    fn anti_aliasing_all() {
        let all = AntiAliasingMode::all();
        assert_eq!(all.len(), 6);
        assert_eq!(AntiAliasingMode::Taa.display_name(), "TAA");
    }

    #[test]
    fn target_platform_all() {
        let all = TargetPlatform::all();
        assert_eq!(all.len(), 9);
        assert_eq!(TargetPlatform::Windows.display_name(), "Windows");
    }

    #[test]
    fn project_open_and_close_scene() {
        let dir = temp_dir();
        let mut project = Project::new("Test".into(), dir.clone());
        project.ensure_directories().unwrap();

        // Create a scene file.
        let scene_path = dir.join("scenes").join("level.scene");
        let mut scene = SceneFile::new(PathBuf::from("scenes/level.scene"));
        scene.add_entity(SerializedEntity::new(Uuid::new_v4(), "Entity1"));
        scene.save_to(&scene_path).unwrap();

        let idx = project.open_scene(PathBuf::from("scenes/level.scene")).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(project.open_scenes.len(), 1);

        // Opening again returns same index.
        let idx2 = project.open_scene(PathBuf::from("scenes/level.scene")).unwrap();
        assert_eq!(idx2, 0);

        let closed = project.close_scene(0);
        assert!(closed.is_some());
        assert!(project.open_scenes.is_empty());

        cleanup(&dir);
    }

    #[test]
    fn physics_settings_defaults() {
        let ps = PhysicsSettings::default();
        assert!((ps.gravity[1] - (-9.81)).abs() < 0.01);
        assert_eq!(ps.solver_iterations, 8);
        assert!(ps.continuous_collision);
    }

    #[test]
    fn audio_settings_defaults() {
        let audio = AudioSettings::default();
        assert_eq!(audio.master_volume, 1.0);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.distance_model, DistanceModel::InverseDistance);
    }

    #[test]
    fn project_version_mismatch() {
        let dir = temp_dir();
        let mut project = Project::new("FutureProject".into(), dir.clone());
        project.version = 999;
        let path = project.save().unwrap();

        let result = Project::load(&path);
        assert!(result.is_err());
        match result.unwrap_err() {
            ProjectError::VersionMismatch { found, supported } => {
                assert_eq!(found, 999);
                assert_eq!(supported, CURRENT_PROJECT_VERSION);
            }
            _ => panic!("Expected VersionMismatch error"),
        }

        cleanup(&dir);
    }

    #[test]
    fn settings_reset() {
        let mut settings = ProjectSettings::default();
        settings.render.window_width = 800;
        settings.physics.gravity = [0.0, -20.0, 0.0];
        settings.reset();
        assert_eq!(settings.render.window_width, 1920);
        assert_eq!(settings.physics.gravity[1], -9.81);
    }

    #[test]
    fn remove_startup_scene() {
        let mut project = Project::new("Test".into(), PathBuf::from("/test"));
        project.set_startup_scene(PathBuf::from("main.scene"));
        project.remove_scene(Path::new("main.scene"));
        assert!(project.startup_scene.is_none());
    }
}
