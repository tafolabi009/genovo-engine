//! Level editing tools for the Genovo editor.
//!
//! Provides multi-level management, level streaming configuration, bounds
//! visualization, spawn point placement, trigger zone editing, and gameplay
//! layer management.
//!
//! # Features
//!
//! - Multi-level management (create, delete, load, unload)
//! - Level streaming configuration (distance-based, trigger-based)
//! - Level bounds visualization (AABB, oriented boxes)
//! - Spawn point placement and configuration
//! - Trigger zone editing (box, sphere, cylinder triggers)
//! - Gameplay layer management (filtering, visibility, locking)
//! - Level linking (connections between levels for streaming)
//! - Level metadata and settings
//!
//! # Example
//!
//! ```ignore
//! let mut editor = LevelEditor::new();
//! let level_id = editor.create_level("overworld", LevelSettings::default());
//!
//! editor.add_spawn_point(level_id, SpawnPoint::new("player_start", [0.0, 0.0, 0.0]));
//! editor.add_trigger_zone(level_id, TriggerZone::box_trigger("door_trigger", [-1.0, 0.0, -1.0], [1.0, 3.0, 1.0]));
//! editor.set_streaming_config(level_id, StreamingConfig::distance_based(100.0));
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LevelId(pub u64);

impl LevelId {
    pub fn new(id: u64) -> Self { Self(id) }
    pub fn is_null(&self) -> bool { self.0 == 0 }
}

impl fmt::Display for LevelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Level({})", self.0)
    }
}

/// Unique identifier for a spawn point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpawnPointId(pub u64);

impl fmt::Display for SpawnPointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Spawn({})", self.0)
    }
}

/// Unique identifier for a trigger zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriggerZoneId(pub u64);

impl fmt::Display for TriggerZoneId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trigger({})", self.0)
    }
}

/// Unique identifier for a gameplay layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerId(pub u64);

impl fmt::Display for LayerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layer({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Level bounds
// ---------------------------------------------------------------------------

/// An axis-aligned bounding box defining level bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelBounds {
    /// Minimum corner.
    pub min: [f32; 3],
    /// Maximum corner.
    pub max: [f32; 3],
}

impl LevelBounds {
    /// Create bounds from min and max corners.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Create bounds centered at origin with half-extents.
    pub fn from_center_size(center: [f32; 3], size: [f32; 3]) -> Self {
        Self {
            min: [
                center[0] - size[0] * 0.5,
                center[1] - size[1] * 0.5,
                center[2] - size[2] * 0.5,
            ],
            max: [
                center[0] + size[0] * 0.5,
                center[1] + size[1] * 0.5,
                center[2] + size[2] * 0.5,
            ],
        }
    }

    /// Returns the center of the bounds.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Returns the size (extents) of the bounds.
    pub fn size(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Returns the volume of the bounds.
    pub fn volume(&self) -> f32 {
        let s = self.size();
        s[0] * s[1] * s[2]
    }

    /// Returns `true` if a point is inside the bounds.
    pub fn contains_point(&self, point: [f32; 3]) -> bool {
        point[0] >= self.min[0]
            && point[0] <= self.max[0]
            && point[1] >= self.min[1]
            && point[1] <= self.max[1]
            && point[2] >= self.min[2]
            && point[2] <= self.max[2]
    }

    /// Returns `true` if this bounds intersects another.
    pub fn intersects(&self, other: &LevelBounds) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    /// Expand the bounds to include a point.
    pub fn expand_to_include(&mut self, point: [f32; 3]) {
        for i in 0..3 {
            if point[i] < self.min[i] { self.min[i] = point[i]; }
            if point[i] > self.max[i] { self.max[i] = point[i]; }
        }
    }

    /// Returns the 8 corners of the bounding box.
    pub fn corners(&self) -> [[f32; 3]; 8] {
        [
            [self.min[0], self.min[1], self.min[2]],
            [self.max[0], self.min[1], self.min[2]],
            [self.min[0], self.max[1], self.min[2]],
            [self.max[0], self.max[1], self.min[2]],
            [self.min[0], self.min[1], self.max[2]],
            [self.max[0], self.min[1], self.max[2]],
            [self.min[0], self.max[1], self.max[2]],
            [self.max[0], self.max[1], self.max[2]],
        ]
    }

    /// Returns the 12 edges of the bounding box as pairs of corner indices.
    pub fn edge_indices() -> [(usize, usize); 12] {
        [
            (0, 1), (2, 3), (4, 5), (6, 7), // X edges
            (0, 2), (1, 3), (4, 6), (5, 7), // Y edges
            (0, 4), (1, 5), (2, 6), (3, 7), // Z edges
        ]
    }
}

impl Default for LevelBounds {
    fn default() -> Self {
        Self::from_center_size([0.0, 0.0, 0.0], [100.0, 50.0, 100.0])
    }
}

impl fmt::Display for LevelBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.size();
        write!(f, "Bounds[{:.0}x{:.0}x{:.0}]", s[0], s[1], s[2])
    }
}

// ---------------------------------------------------------------------------
// Level streaming
// ---------------------------------------------------------------------------

/// How a level is streamed in/out.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingMode {
    /// Always loaded.
    AlwaysLoaded,
    /// Loaded when the player is within a distance.
    DistanceBased { load_distance: f32, unload_distance: f32 },
    /// Loaded when a specific trigger is entered.
    TriggerBased { trigger_zone_id: TriggerZoneId },
    /// Manually controlled by gameplay code.
    Manual,
    /// Blueprint/script controlled.
    Scripted { script_name: String },
}

impl Default for StreamingMode {
    fn default() -> Self {
        Self::AlwaysLoaded
    }
}

impl fmt::Display for StreamingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlwaysLoaded => write!(f, "AlwaysLoaded"),
            Self::DistanceBased { load_distance, .. } => write!(f, "Distance({load_distance:.0}m)"),
            Self::TriggerBased { trigger_zone_id } => write!(f, "Trigger({trigger_zone_id})"),
            Self::Manual => write!(f, "Manual"),
            Self::Scripted { script_name } => write!(f, "Script({script_name})"),
        }
    }
}

/// Configuration for level streaming.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// The streaming mode.
    pub mode: StreamingMode,
    /// Priority (higher = loaded first).
    pub priority: i32,
    /// Minimum time (seconds) a level stays loaded after unload triggers.
    pub min_load_time: f32,
    /// Whether to show a loading screen when streaming.
    pub show_loading_screen: bool,
    /// Levels that must be loaded before this one.
    pub prerequisites: Vec<LevelId>,
    /// The streaming reference point (where distance is measured from).
    pub reference_point: [f32; 3],
}

impl StreamingConfig {
    /// Create a distance-based streaming config.
    pub fn distance_based(distance: f32) -> Self {
        Self {
            mode: StreamingMode::DistanceBased {
                load_distance: distance,
                unload_distance: distance * 1.5,
            },
            priority: 0,
            min_load_time: 5.0,
            show_loading_screen: false,
            prerequisites: Vec::new(),
            reference_point: [0.0, 0.0, 0.0],
        }
    }

    /// Create an always-loaded config.
    pub fn always_loaded() -> Self {
        Self {
            mode: StreamingMode::AlwaysLoaded,
            priority: 100,
            min_load_time: 0.0,
            show_loading_screen: false,
            prerequisites: Vec::new(),
            reference_point: [0.0, 0.0, 0.0],
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self::always_loaded()
    }
}

// ---------------------------------------------------------------------------
// Spawn points
// ---------------------------------------------------------------------------

/// Type of spawn point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpawnType {
    /// Player spawn point.
    Player,
    /// Enemy spawn point.
    Enemy,
    /// NPC spawn point.
    Npc,
    /// Item/pickup spawn point.
    Item,
    /// Vehicle spawn point.
    Vehicle,
    /// Custom spawn type.
    Custom,
}

impl Default for SpawnType {
    fn default() -> Self {
        Self::Player
    }
}

impl fmt::Display for SpawnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Player => write!(f, "Player"),
            Self::Enemy => write!(f, "Enemy"),
            Self::Npc => write!(f, "NPC"),
            Self::Item => write!(f, "Item"),
            Self::Vehicle => write!(f, "Vehicle"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// A spawn point in a level.
#[derive(Debug, Clone)]
pub struct SpawnPoint {
    /// Unique identifier.
    pub id: SpawnPointId,
    /// Display name.
    pub name: String,
    /// World position.
    pub position: [f32; 3],
    /// Rotation (euler angles in degrees).
    pub rotation: [f32; 3],
    /// Spawn type.
    pub spawn_type: SpawnType,
    /// Whether this spawn point is active.
    pub enabled: bool,
    /// Maximum number of entities that can spawn here.
    pub max_count: u32,
    /// Respawn delay in seconds.
    pub respawn_delay: f32,
    /// Spawn radius (random offset from position).
    pub spawn_radius: f32,
    /// Tags for filtering.
    pub tags: Vec<String>,
    /// Custom properties.
    pub properties: HashMap<String, String>,
    /// Team/faction identifier.
    pub team: Option<String>,
    /// Gizmo display color.
    pub color: [f32; 4],
}

impl SpawnPoint {
    /// Create a new spawn point.
    pub fn new(name: &str, position: [f32; 3]) -> Self {
        Self {
            id: SpawnPointId(0), // Assigned by the editor.
            name: name.to_string(),
            position,
            rotation: [0.0, 0.0, 0.0],
            spawn_type: SpawnType::Player,
            enabled: true,
            max_count: 1,
            respawn_delay: 0.0,
            spawn_radius: 0.0,
            tags: Vec::new(),
            properties: HashMap::new(),
            team: None,
            color: [0.0, 1.0, 0.0, 1.0],
        }
    }

    /// Create an enemy spawn point.
    pub fn enemy(name: &str, position: [f32; 3], max_count: u32) -> Self {
        let mut sp = Self::new(name, position);
        sp.spawn_type = SpawnType::Enemy;
        sp.max_count = max_count;
        sp.color = [1.0, 0.0, 0.0, 1.0];
        sp
    }
}

impl fmt::Display for SpawnPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Spawn[{}, '{}', ({:.1}, {:.1}, {:.1})]",
            self.spawn_type, self.name, self.position[0], self.position[1], self.position[2]
        )
    }
}

// ---------------------------------------------------------------------------
// Trigger zones
// ---------------------------------------------------------------------------

/// The shape of a trigger zone.
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerShape {
    /// Axis-aligned box.
    Box { min: [f32; 3], max: [f32; 3] },
    /// Sphere.
    Sphere { center: [f32; 3], radius: f32 },
    /// Cylinder (vertical axis).
    Cylinder { center: [f32; 3], radius: f32, height: f32 },
    /// Oriented box (with rotation).
    OrientedBox { center: [f32; 3], half_extents: [f32; 3], rotation: [f32; 3] },
}

impl TriggerShape {
    /// Returns `true` if a point is inside this shape.
    pub fn contains_point(&self, point: [f32; 3]) -> bool {
        match self {
            Self::Box { min, max } => {
                point[0] >= min[0] && point[0] <= max[0]
                    && point[1] >= min[1] && point[1] <= max[1]
                    && point[2] >= min[2] && point[2] <= max[2]
            }
            Self::Sphere { center, radius } => {
                let dx = point[0] - center[0];
                let dy = point[1] - center[1];
                let dz = point[2] - center[2];
                (dx * dx + dy * dy + dz * dz) <= radius * radius
            }
            Self::Cylinder { center, radius, height } => {
                let dx = point[0] - center[0];
                let dz = point[2] - center[2];
                let in_radius = (dx * dx + dz * dz) <= radius * radius;
                let in_height = point[1] >= center[1] && point[1] <= center[1] + height;
                in_radius && in_height
            }
            Self::OrientedBox { center, half_extents, .. } => {
                // Simplified (ignoring rotation).
                let dx = (point[0] - center[0]).abs();
                let dy = (point[1] - center[1]).abs();
                let dz = (point[2] - center[2]).abs();
                dx <= half_extents[0] && dy <= half_extents[1] && dz <= half_extents[2]
            }
        }
    }

    /// Returns the center of the shape.
    pub fn center(&self) -> [f32; 3] {
        match self {
            Self::Box { min, max } => [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ],
            Self::Sphere { center, .. }
            | Self::Cylinder { center, .. }
            | Self::OrientedBox { center, .. } => *center,
        }
    }
}

/// What happens when something enters/exits a trigger.
#[derive(Debug, Clone)]
pub enum TriggerAction {
    /// Load a level.
    LoadLevel(LevelId),
    /// Unload a level.
    UnloadLevel(LevelId),
    /// Send an event by name.
    SendEvent(String),
    /// Execute a script function.
    RunScript(String),
    /// Play a sound.
    PlaySound(String),
    /// Toggle an entity.
    ToggleEntity(String),
    /// Teleport the player.
    Teleport([f32; 3]),
    /// Custom action with parameters.
    Custom { action_type: String, params: HashMap<String, String> },
}

impl fmt::Display for TriggerAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadLevel(id) => write!(f, "LoadLevel({id})"),
            Self::UnloadLevel(id) => write!(f, "UnloadLevel({id})"),
            Self::SendEvent(name) => write!(f, "Event({name})"),
            Self::RunScript(name) => write!(f, "Script({name})"),
            Self::PlaySound(name) => write!(f, "Sound({name})"),
            Self::ToggleEntity(name) => write!(f, "Toggle({name})"),
            Self::Teleport(pos) => write!(f, "Teleport({:.1},{:.1},{:.1})", pos[0], pos[1], pos[2]),
            Self::Custom { action_type, .. } => write!(f, "Custom({action_type})"),
        }
    }
}

/// A trigger zone in a level.
#[derive(Debug, Clone)]
pub struct TriggerZone {
    /// Unique identifier.
    pub id: TriggerZoneId,
    /// Display name.
    pub name: String,
    /// The trigger shape.
    pub shape: TriggerShape,
    /// Actions to perform on enter.
    pub on_enter: Vec<TriggerAction>,
    /// Actions to perform on exit.
    pub on_exit: Vec<TriggerAction>,
    /// Actions to perform while inside (each frame).
    pub on_stay: Vec<TriggerAction>,
    /// Whether this trigger is active.
    pub enabled: bool,
    /// Whether this trigger can fire only once.
    pub one_shot: bool,
    /// Delay before the trigger fires (seconds).
    pub delay: f32,
    /// Tags that filter which entities can activate this trigger.
    pub filter_tags: Vec<String>,
    /// Display color for the editor.
    pub color: [f32; 4],
    /// Whether to show the trigger in the editor.
    pub visible: bool,
}

impl TriggerZone {
    /// Create a box trigger.
    pub fn box_trigger(name: &str, min: [f32; 3], max: [f32; 3]) -> Self {
        Self {
            id: TriggerZoneId(0),
            name: name.to_string(),
            shape: TriggerShape::Box { min, max },
            on_enter: Vec::new(),
            on_exit: Vec::new(),
            on_stay: Vec::new(),
            enabled: true,
            one_shot: false,
            delay: 0.0,
            filter_tags: Vec::new(),
            color: [0.3, 0.3, 1.0, 0.3],
            visible: true,
        }
    }

    /// Create a sphere trigger.
    pub fn sphere_trigger(name: &str, center: [f32; 3], radius: f32) -> Self {
        Self {
            id: TriggerZoneId(0),
            name: name.to_string(),
            shape: TriggerShape::Sphere { center, radius },
            on_enter: Vec::new(),
            on_exit: Vec::new(),
            on_stay: Vec::new(),
            enabled: true,
            one_shot: false,
            delay: 0.0,
            filter_tags: Vec::new(),
            color: [0.3, 1.0, 0.3, 0.3],
            visible: true,
        }
    }

    /// Add an on-enter action.
    pub fn with_on_enter(mut self, action: TriggerAction) -> Self {
        self.on_enter.push(action);
        self
    }

    /// Add an on-exit action.
    pub fn with_on_exit(mut self, action: TriggerAction) -> Self {
        self.on_exit.push(action);
        self
    }

    /// Test if a point is inside the trigger.
    pub fn test_point(&self, point: [f32; 3]) -> bool {
        self.enabled && self.shape.contains_point(point)
    }
}

impl fmt::Display for TriggerZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trigger['{}', enter={}, exit={}]", self.name, self.on_enter.len(), self.on_exit.len())
    }
}

// ---------------------------------------------------------------------------
// Gameplay Layers
// ---------------------------------------------------------------------------

/// A gameplay layer for organizing entities in a level.
#[derive(Debug, Clone)]
pub struct GameplayLayer {
    /// Unique identifier.
    pub id: LayerId,
    /// Layer name.
    pub name: String,
    /// Whether entities on this layer are visible in the editor.
    pub visible: bool,
    /// Whether entities on this layer can be selected/edited.
    pub locked: bool,
    /// Display color for the layer.
    pub color: [f32; 4],
    /// Sort order (lower = drawn first).
    pub sort_order: i32,
    /// Description of what this layer contains.
    pub description: String,
    /// Whether this layer is active at runtime.
    pub runtime_active: bool,
    /// Entity IDs on this layer.
    pub entities: Vec<u64>,
}

impl GameplayLayer {
    /// Create a new gameplay layer.
    pub fn new(id: LayerId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            visible: true,
            locked: false,
            color: [1.0, 1.0, 1.0, 1.0],
            sort_order: 0,
            description: String::new(),
            runtime_active: true,
            entities: Vec::new(),
        }
    }

    /// Add an entity to this layer.
    pub fn add_entity(&mut self, entity_id: u64) {
        if !self.entities.contains(&entity_id) {
            self.entities.push(entity_id);
        }
    }

    /// Remove an entity from this layer.
    pub fn remove_entity(&mut self, entity_id: u64) {
        self.entities.retain(|&e| e != entity_id);
    }

    /// Returns the number of entities on this layer.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

impl fmt::Display for GameplayLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Layer['{}', {} entities, vis={}, lock={}]",
            self.name,
            self.entity_count(),
            self.visible,
            self.locked
        )
    }
}

// ---------------------------------------------------------------------------
// Level
// ---------------------------------------------------------------------------

/// Level load state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LevelState {
    /// Not loaded.
    Unloaded,
    /// Currently loading.
    Loading,
    /// Fully loaded and active.
    Loaded,
    /// Currently unloading.
    Unloading,
    /// Failed to load.
    Error,
}

impl fmt::Display for LevelState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unloaded => write!(f, "Unloaded"),
            Self::Loading => write!(f, "Loading"),
            Self::Loaded => write!(f, "Loaded"),
            Self::Unloading => write!(f, "Unloading"),
            Self::Error => write!(f, "Error"),
        }
    }
}

/// Settings for a level.
#[derive(Debug, Clone)]
pub struct LevelSettings {
    /// Whether this is the persistent/main level.
    pub is_persistent: bool,
    /// Ambient light color.
    pub ambient_color: [f32; 3],
    /// Ambient light intensity.
    pub ambient_intensity: f32,
    /// Fog color.
    pub fog_color: [f32; 3],
    /// Fog density.
    pub fog_density: f32,
    /// Gravity override (None = use global).
    pub gravity: Option<[f32; 3]>,
    /// Navigation mesh baking settings path.
    pub navmesh_settings: Option<String>,
}

impl Default for LevelSettings {
    fn default() -> Self {
        Self {
            is_persistent: false,
            ambient_color: [0.2, 0.2, 0.3],
            ambient_intensity: 0.5,
            fog_color: [0.7, 0.8, 0.9],
            fog_density: 0.0,
            gravity: None,
            navmesh_settings: None,
        }
    }
}

/// A level in the editor.
#[derive(Debug, Clone)]
pub struct Level {
    /// Unique identifier.
    pub id: LevelId,
    /// Level name.
    pub name: String,
    /// Level bounds.
    pub bounds: LevelBounds,
    /// Current state.
    pub state: LevelState,
    /// Streaming configuration.
    pub streaming: StreamingConfig,
    /// Level settings.
    pub settings: LevelSettings,
    /// Spawn points.
    pub spawn_points: Vec<SpawnPoint>,
    /// Trigger zones.
    pub trigger_zones: Vec<TriggerZone>,
    /// Gameplay layers.
    pub layers: Vec<GameplayLayer>,
    /// Connections to other levels.
    pub connections: Vec<LevelConnection>,
    /// Description.
    pub description: String,
    /// Tags.
    pub tags: Vec<String>,
    /// Whether this level is selected in the editor.
    pub selected: bool,
    /// Whether this level is visible in the editor.
    pub visible: bool,
}

impl Level {
    /// Create a new level.
    pub fn new(id: LevelId, name: &str, settings: LevelSettings) -> Self {
        Self {
            id,
            name: name.to_string(),
            bounds: LevelBounds::default(),
            state: LevelState::Unloaded,
            streaming: StreamingConfig::default(),
            settings,
            spawn_points: Vec::new(),
            trigger_zones: Vec::new(),
            layers: Vec::new(),
            connections: Vec::new(),
            description: String::new(),
            tags: Vec::new(),
            selected: false,
            visible: true,
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Level[{}, '{}', {}, spawns={}, triggers={}, layers={}]",
            self.id,
            self.name,
            self.state,
            self.spawn_points.len(),
            self.trigger_zones.len(),
            self.layers.len()
        )
    }
}

/// A connection between two levels (for streaming transitions).
#[derive(Debug, Clone)]
pub struct LevelConnection {
    /// The target level.
    pub target_level: LevelId,
    /// Connection point in this level.
    pub source_point: [f32; 3],
    /// Connection point in the target level.
    pub target_point: [f32; 3],
    /// Whether this connection is bidirectional.
    pub bidirectional: bool,
    /// Display name.
    pub name: String,
}

// ---------------------------------------------------------------------------
// Level Editor
// ---------------------------------------------------------------------------

/// The main level editor.
pub struct LevelEditor {
    /// All levels.
    levels: HashMap<LevelId, Level>,
    /// The currently active/focused level.
    active_level: Option<LevelId>,
    /// Next level ID.
    next_level_id: u64,
    /// Next spawn point ID.
    next_spawn_id: u64,
    /// Next trigger zone ID.
    next_trigger_id: u64,
    /// Next layer ID.
    next_layer_id: u64,
}

impl LevelEditor {
    /// Create a new level editor.
    pub fn new() -> Self {
        Self {
            levels: HashMap::new(),
            active_level: None,
            next_level_id: 1,
            next_spawn_id: 1,
            next_trigger_id: 1,
            next_layer_id: 1,
        }
    }

    /// Create a new level.
    pub fn create_level(&mut self, name: &str, settings: LevelSettings) -> LevelId {
        let id = LevelId::new(self.next_level_id);
        self.next_level_id += 1;
        let level = Level::new(id, name, settings);
        self.levels.insert(id, level);
        if self.active_level.is_none() {
            self.active_level = Some(id);
        }
        id
    }

    /// Remove a level.
    pub fn remove_level(&mut self, id: LevelId) -> Option<Level> {
        if self.active_level == Some(id) {
            self.active_level = None;
        }
        self.levels.remove(&id)
    }

    /// Get a level.
    pub fn get_level(&self, id: LevelId) -> Option<&Level> {
        self.levels.get(&id)
    }

    /// Get a mutable level.
    pub fn get_level_mut(&mut self, id: LevelId) -> Option<&mut Level> {
        self.levels.get_mut(&id)
    }

    /// Set the active level.
    pub fn set_active_level(&mut self, id: LevelId) {
        if self.levels.contains_key(&id) {
            self.active_level = Some(id);
        }
    }

    /// Get the active level ID.
    pub fn active_level(&self) -> Option<LevelId> {
        self.active_level
    }

    /// Add a spawn point to a level.
    pub fn add_spawn_point(&mut self, level_id: LevelId, mut spawn: SpawnPoint) -> Option<SpawnPointId> {
        let sp_id = SpawnPointId(self.next_spawn_id);
        self.next_spawn_id += 1;
        spawn.id = sp_id;
        let level = self.levels.get_mut(&level_id)?;
        level.spawn_points.push(spawn);
        Some(sp_id)
    }

    /// Remove a spawn point.
    pub fn remove_spawn_point(&mut self, level_id: LevelId, spawn_id: SpawnPointId) -> Option<SpawnPoint> {
        let level = self.levels.get_mut(&level_id)?;
        let pos = level.spawn_points.iter().position(|s| s.id == spawn_id)?;
        Some(level.spawn_points.remove(pos))
    }

    /// Add a trigger zone to a level.
    pub fn add_trigger_zone(&mut self, level_id: LevelId, mut trigger: TriggerZone) -> Option<TriggerZoneId> {
        let tz_id = TriggerZoneId(self.next_trigger_id);
        self.next_trigger_id += 1;
        trigger.id = tz_id;
        let level = self.levels.get_mut(&level_id)?;
        level.trigger_zones.push(trigger);
        Some(tz_id)
    }

    /// Remove a trigger zone.
    pub fn remove_trigger_zone(&mut self, level_id: LevelId, trigger_id: TriggerZoneId) -> Option<TriggerZone> {
        let level = self.levels.get_mut(&level_id)?;
        let pos = level.trigger_zones.iter().position(|t| t.id == trigger_id)?;
        Some(level.trigger_zones.remove(pos))
    }

    /// Add a gameplay layer.
    pub fn add_layer(&mut self, level_id: LevelId, name: &str) -> Option<LayerId> {
        let layer_id = LayerId(self.next_layer_id);
        self.next_layer_id += 1;
        let layer = GameplayLayer::new(layer_id, name);
        let level = self.levels.get_mut(&level_id)?;
        level.layers.push(layer);
        Some(layer_id)
    }

    /// Set streaming config for a level.
    pub fn set_streaming_config(&mut self, level_id: LevelId, config: StreamingConfig) {
        if let Some(level) = self.levels.get_mut(&level_id) {
            level.streaming = config;
        }
    }

    /// Add a connection between two levels.
    pub fn add_connection(
        &mut self,
        source: LevelId,
        target: LevelId,
        source_point: [f32; 3],
        target_point: [f32; 3],
    ) {
        if let Some(level) = self.levels.get_mut(&source) {
            level.connections.push(LevelConnection {
                target_level: target,
                source_point,
                target_point,
                bidirectional: true,
                name: format!("{} -> {}", source, target),
            });
        }
    }

    /// List all levels.
    pub fn list_levels(&self) -> Vec<(LevelId, &str)> {
        self.levels.values().map(|l| (l.id, l.name.as_str())).collect()
    }

    /// Returns the total number of levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

impl Default for LevelEditor {
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
    fn test_create_level() {
        let mut editor = LevelEditor::new();
        let id = editor.create_level("overworld", LevelSettings::default());
        assert_eq!(editor.level_count(), 1);
        assert_eq!(editor.get_level(id).unwrap().name, "overworld");
    }

    #[test]
    fn test_spawn_points() {
        let mut editor = LevelEditor::new();
        let level_id = editor.create_level("test", LevelSettings::default());

        let sp_id = editor.add_spawn_point(level_id, SpawnPoint::new("start", [0.0, 0.0, 0.0])).unwrap();
        let level = editor.get_level(level_id).unwrap();
        assert_eq!(level.spawn_points.len(), 1);

        editor.remove_spawn_point(level_id, sp_id).unwrap();
        assert_eq!(editor.get_level(level_id).unwrap().spawn_points.len(), 0);
    }

    #[test]
    fn test_trigger_zones() {
        let mut editor = LevelEditor::new();
        let level_id = editor.create_level("test", LevelSettings::default());

        let trigger = TriggerZone::box_trigger("door", [-1.0, 0.0, -1.0], [1.0, 3.0, 1.0])
            .with_on_enter(TriggerAction::SendEvent("open_door".into()));

        let tz_id = editor.add_trigger_zone(level_id, trigger).unwrap();
        let level = editor.get_level(level_id).unwrap();
        assert_eq!(level.trigger_zones.len(), 1);
        assert_eq!(level.trigger_zones[0].on_enter.len(), 1);

        editor.remove_trigger_zone(level_id, tz_id).unwrap();
    }

    #[test]
    fn test_level_bounds() {
        let bounds = LevelBounds::from_center_size([0.0, 0.0, 0.0], [100.0, 50.0, 100.0]);
        assert!(bounds.contains_point([10.0, 5.0, 10.0]));
        assert!(!bounds.contains_point([60.0, 0.0, 0.0]));
        assert_eq!(bounds.center(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_trigger_shape_contains() {
        let sphere = TriggerShape::Sphere { center: [0.0, 0.0, 0.0], radius: 5.0 };
        assert!(sphere.contains_point([3.0, 0.0, 0.0]));
        assert!(!sphere.contains_point([6.0, 0.0, 0.0]));

        let cylinder = TriggerShape::Cylinder { center: [0.0, 0.0, 0.0], radius: 3.0, height: 5.0 };
        assert!(cylinder.contains_point([1.0, 2.0, 1.0]));
        assert!(!cylinder.contains_point([1.0, 6.0, 1.0]));
    }

    #[test]
    fn test_gameplay_layers() {
        let mut editor = LevelEditor::new();
        let level_id = editor.create_level("test", LevelSettings::default());
        let layer_id = editor.add_layer(level_id, "enemies").unwrap();

        let level = editor.get_level_mut(level_id).unwrap();
        let layer = level.layers.iter_mut().find(|l| l.id == layer_id).unwrap();
        layer.add_entity(100);
        layer.add_entity(101);
        assert_eq!(layer.entity_count(), 2);

        layer.remove_entity(100);
        assert_eq!(layer.entity_count(), 1);
    }

    #[test]
    fn test_streaming_config() {
        let mut editor = LevelEditor::new();
        let level_id = editor.create_level("zone_a", LevelSettings::default());
        editor.set_streaming_config(level_id, StreamingConfig::distance_based(150.0));

        let level = editor.get_level(level_id).unwrap();
        match &level.streaming.mode {
            StreamingMode::DistanceBased { load_distance, .. } => {
                assert_eq!(*load_distance, 150.0);
            }
            _ => panic!("expected distance-based streaming"),
        }
    }

    #[test]
    fn test_level_connections() {
        let mut editor = LevelEditor::new();
        let a = editor.create_level("zone_a", LevelSettings::default());
        let b = editor.create_level("zone_b", LevelSettings::default());
        editor.add_connection(a, b, [50.0, 0.0, 0.0], [-50.0, 0.0, 0.0]);

        let level = editor.get_level(a).unwrap();
        assert_eq!(level.connections.len(), 1);
        assert_eq!(level.connections[0].target_level, b);
    }

    #[test]
    fn test_bounds_intersect() {
        let a = LevelBounds::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let b = LevelBounds::new([5.0, 5.0, 5.0], [15.0, 15.0, 15.0]);
        let c = LevelBounds::new([20.0, 20.0, 20.0], [30.0, 30.0, 30.0]);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }
}
