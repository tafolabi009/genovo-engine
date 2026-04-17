//! Multi-scene management for the Genovo engine.
//!
//! This module provides a complete scene management subsystem that supports:
//!
//! - **Scene loading and unloading** — load scenes by handle with async support.
//! - **Additive scene loading** — layer multiple scenes simultaneously.
//! - **Scene transitions** — fade, dissolve, wipe, and crossfade effects.
//! - **Scene stacking** — push/pop scene stack for menus, pause screens, etc.
//! - **DontDestroyOnLoad** — persistent entities that survive scene transitions.
//! - **Scene dependencies** — declare and resolve inter-scene dependencies.
//! - **Async loading with progress** — non-blocking loads with progress callbacks.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Scene handle and identity
// ---------------------------------------------------------------------------

/// Unique identifier for a scene asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneId(u64);

impl SceneId {
    /// Creates a new unique scene id.
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// Creates a scene id from a raw value (for deserialization).
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the raw numeric value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for SceneId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SceneId({})", self.0)
    }
}

/// A handle to a loaded or loading scene instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneHandle(u64);

impl SceneHandle {
    /// Creates a new unique scene handle.
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// Creates a handle from a raw value.
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the raw numeric value.
    pub fn raw(&self) -> u64 {
        self.0
    }

    /// A sentinel handle representing "no scene".
    pub const INVALID: Self = Self(0);

    /// Whether this handle is valid (non-zero).
    pub fn is_valid(&self) -> bool {
        self.0 != 0
    }
}

impl fmt::Display for SceneHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SceneHandle({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Entity handle (simplified for self-contained module)
// ---------------------------------------------------------------------------

/// Lightweight entity reference used within the scene manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

impl EntityId {
    /// Creates a new entity id.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw numeric value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Scene state
// ---------------------------------------------------------------------------

/// The lifecycle state of a scene instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneState {
    /// The scene is queued for loading but hasn't started yet.
    Pending,
    /// The scene is currently being loaded asynchronously.
    Loading,
    /// All data has been loaded; waiting for activation.
    Loaded,
    /// The scene is fully active and ticking.
    Active,
    /// The scene is being unloaded.
    Unloading,
    /// The scene has been fully unloaded.
    Unloaded,
    /// The scene failed to load.
    Failed,
}

impl SceneState {
    /// Whether the scene is in a state where it receives updates.
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Whether the scene is in a loading state.
    pub fn is_loading(&self) -> bool {
        matches!(self, Self::Pending | Self::Loading)
    }

    /// Whether the scene has completed loading successfully.
    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::Loaded | Self::Active)
    }

    /// Whether the scene is being removed.
    pub fn is_unloading(&self) -> bool {
        matches!(self, Self::Unloading | Self::Unloaded)
    }
}

impl fmt::Display for SceneState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Loading => write!(f, "Loading"),
            Self::Loaded => write!(f, "Loaded"),
            Self::Active => write!(f, "Active"),
            Self::Unloading => write!(f, "Unloading"),
            Self::Unloaded => write!(f, "Unloaded"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Scene load mode
// ---------------------------------------------------------------------------

/// How a scene should be loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadSceneMode {
    /// Replace the current scene entirely (default behaviour). All non-
    /// persistent entities in the currently active scene are destroyed.
    Single,
    /// Load the scene additively on top of whatever is already loaded.
    /// No existing entities are destroyed.
    Additive,
}

impl Default for LoadSceneMode {
    fn default() -> Self {
        Self::Single
    }
}

// ---------------------------------------------------------------------------
// Scene transition effects
// ---------------------------------------------------------------------------

/// The visual effect used when transitioning between scenes.
#[derive(Debug, Clone)]
pub enum TransitionEffect {
    /// Instant cut -- no visual effect.
    None,
    /// Fade to a solid colour, load, then fade back in.
    Fade {
        /// The colour to fade to / from (RGBA, 0.0..1.0).
        color: [f32; 4],
        /// Duration of the fade-out phase in seconds.
        fade_out_duration: f32,
        /// Duration of the fade-in phase in seconds.
        fade_in_duration: f32,
    },
    /// Crossfade (dissolve) between the old and new scenes.
    Dissolve {
        /// Duration of the dissolve in seconds.
        duration: f32,
    },
    /// Directional wipe from one scene to the next.
    Wipe {
        /// Direction of the wipe: 0 = left-to-right, 1 = right-to-left,
        /// 2 = top-to-bottom, 3 = bottom-to-top.
        direction: u8,
        /// Duration in seconds.
        duration: f32,
    },
    /// A custom transition driven by a shader or callback.
    Custom {
        /// Identifier of the custom transition effect.
        name: String,
        /// Duration in seconds.
        duration: f32,
        /// Arbitrary key-value parameters passed to the effect.
        params: HashMap<String, f32>,
    },
}

impl TransitionEffect {
    /// Creates a black-fade transition with the given durations.
    pub fn fade_black(out_sec: f32, in_sec: f32) -> Self {
        Self::Fade {
            color: [0.0, 0.0, 0.0, 1.0],
            fade_out_duration: out_sec,
            fade_in_duration: in_sec,
        }
    }

    /// Creates a white-fade transition.
    pub fn fade_white(out_sec: f32, in_sec: f32) -> Self {
        Self::Fade {
            color: [1.0, 1.0, 1.0, 1.0],
            fade_out_duration: out_sec,
            fade_in_duration: in_sec,
        }
    }

    /// Creates a dissolve transition.
    pub fn dissolve(duration: f32) -> Self {
        Self::Dissolve { duration }
    }

    /// Creates a left-to-right wipe transition.
    pub fn wipe_left_to_right(duration: f32) -> Self {
        Self::Wipe {
            direction: 0,
            duration,
        }
    }

    /// Creates a right-to-left wipe transition.
    pub fn wipe_right_to_left(duration: f32) -> Self {
        Self::Wipe {
            direction: 1,
            duration,
        }
    }

    /// Creates a top-to-bottom wipe transition.
    pub fn wipe_top_to_bottom(duration: f32) -> Self {
        Self::Wipe {
            direction: 2,
            duration,
        }
    }

    /// Creates a bottom-to-top wipe transition.
    pub fn wipe_bottom_to_top(duration: f32) -> Self {
        Self::Wipe {
            direction: 3,
            duration,
        }
    }

    /// Returns the total duration of the transition in seconds.
    pub fn total_duration(&self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Fade {
                fade_out_duration,
                fade_in_duration,
                ..
            } => fade_out_duration + fade_in_duration,
            Self::Dissolve { duration } => *duration,
            Self::Wipe { duration, .. } => *duration,
            Self::Custom { duration, .. } => *duration,
        }
    }
}

impl Default for TransitionEffect {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// Transition state machine
// ---------------------------------------------------------------------------

/// Current phase of an in-progress transition.
#[derive(Debug, Clone)]
pub enum TransitionPhase {
    /// No transition is active.
    Idle,
    /// Fading / wiping out the old scene.
    FadingOut {
        /// Elapsed time in the fade-out phase.
        elapsed: f32,
        /// Total duration of the fade-out.
        duration: f32,
    },
    /// Loading the new scene (screen may be fully covered).
    LoadingNewScene,
    /// Fading / wiping in the new scene.
    FadingIn {
        /// Elapsed time in the fade-in phase.
        elapsed: f32,
        /// Total duration of the fade-in.
        duration: f32,
    },
    /// Dissolve between old and new scenes.
    Dissolving {
        /// Current blend factor 0.0 (old) .. 1.0 (new).
        blend: f32,
        /// Total duration.
        duration: f32,
        /// Elapsed time.
        elapsed: f32,
    },
    /// Wipe between old and new scenes.
    Wiping {
        /// Wipe progress 0.0 .. 1.0.
        progress: f32,
        /// Wipe direction.
        direction: u8,
        /// Total duration.
        duration: f32,
        /// Elapsed time.
        elapsed: f32,
    },
}

impl TransitionPhase {
    /// Returns the visual blend/progress factor (0.0 .. 1.0).
    pub fn progress(&self) -> f32 {
        match self {
            Self::Idle => 1.0,
            Self::FadingOut { elapsed, duration } => {
                if *duration <= 0.0 {
                    1.0
                } else {
                    (elapsed / duration).clamp(0.0, 1.0)
                }
            }
            Self::LoadingNewScene => 1.0,
            Self::FadingIn { elapsed, duration } => {
                if *duration <= 0.0 {
                    1.0
                } else {
                    (elapsed / duration).clamp(0.0, 1.0)
                }
            }
            Self::Dissolving { blend, .. } => *blend,
            Self::Wiping { progress, .. } => *progress,
        }
    }

    /// Whether a transition is currently active.
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Idle)
    }
}

// ---------------------------------------------------------------------------
// Scene dependency
// ---------------------------------------------------------------------------

/// A dependency declaration from one scene to another.
#[derive(Debug, Clone)]
pub struct SceneDependency {
    /// The scene that depends on another.
    pub dependent: SceneId,
    /// The scene being depended upon.
    pub dependency: SceneId,
    /// Whether the dependency is hard (must be loaded) or soft (optional).
    pub required: bool,
    /// Human-readable reason for the dependency.
    pub reason: String,
}

impl SceneDependency {
    /// Creates a hard dependency.
    pub fn required(dependent: SceneId, dependency: SceneId, reason: &str) -> Self {
        Self {
            dependent,
            dependency,
            required: true,
            reason: reason.to_string(),
        }
    }

    /// Creates a soft/optional dependency.
    pub fn optional(dependent: SceneId, dependency: SceneId, reason: &str) -> Self {
        Self {
            dependent,
            dependency,
            required: false,
            reason: reason.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Scene definition (asset data)
// ---------------------------------------------------------------------------

/// Static definition of a scene asset -- the blueprint from which scene
/// instances are created.
#[derive(Debug, Clone)]
pub struct SceneDefinition {
    /// The unique asset id.
    pub id: SceneId,
    /// Human-readable name.
    pub name: String,
    /// Asset path / URI used by the loader.
    pub path: String,
    /// Dependencies on other scenes.
    pub dependencies: Vec<SceneDependency>,
    /// Tags for filtering and categorisation.
    pub tags: HashSet<String>,
    /// Estimated memory footprint in bytes (used for budgeting).
    pub estimated_memory: u64,
    /// Whether this scene should be loaded in the background at low priority.
    pub background_loadable: bool,
}

impl SceneDefinition {
    /// Creates a new scene definition.
    pub fn new(name: &str, path: &str) -> Self {
        Self {
            id: SceneId::new(),
            name: name.to_string(),
            path: path.to_string(),
            dependencies: Vec::new(),
            tags: HashSet::new(),
            estimated_memory: 0,
            background_loadable: true,
        }
    }

    /// Adds a hard dependency.
    pub fn add_dependency(&mut self, dep_id: SceneId, reason: &str) {
        self.dependencies
            .push(SceneDependency::required(self.id, dep_id, reason));
    }

    /// Adds an optional dependency.
    pub fn add_optional_dependency(&mut self, dep_id: SceneId, reason: &str) {
        self.dependencies
            .push(SceneDependency::optional(self.id, dep_id, reason));
    }

    /// Adds a tag.
    pub fn add_tag(&mut self, tag: &str) {
        self.tags.insert(tag.to_string());
    }

    /// Checks whether the definition has a given tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }

    /// Returns all required dependency scene ids.
    pub fn required_dependencies(&self) -> Vec<SceneId> {
        self.dependencies
            .iter()
            .filter(|d| d.required)
            .map(|d| d.dependency)
            .collect()
    }

    /// Returns all dependency scene ids (required + optional).
    pub fn all_dependencies(&self) -> Vec<SceneId> {
        self.dependencies.iter().map(|d| d.dependency).collect()
    }
}

// ---------------------------------------------------------------------------
// Scene instance
// ---------------------------------------------------------------------------

/// A live instance of a loaded (or loading) scene.
#[derive(Debug)]
pub struct SceneInstance {
    /// Unique handle for this instance.
    pub handle: SceneHandle,
    /// The scene definition this instance was created from.
    pub definition_id: SceneId,
    /// Current lifecycle state.
    pub state: SceneState,
    /// How the scene was loaded.
    pub load_mode: LoadSceneMode,
    /// Entity ids that belong to this scene instance.
    pub entities: Vec<EntityId>,
    /// Current load progress (0.0 .. 1.0).
    pub load_progress: f32,
    /// Error message if the scene failed to load.
    pub error: Option<String>,
    /// Whether this scene instance is visible.
    pub visible: bool,
    /// Whether this scene receives updates.
    pub update_enabled: bool,
    /// Scene-local data / metadata.
    pub metadata: HashMap<String, String>,
    /// Priority for update ordering (lower = earlier).
    pub priority: i32,
}

impl SceneInstance {
    /// Creates a new pending scene instance.
    pub fn new(definition_id: SceneId, load_mode: LoadSceneMode) -> Self {
        Self {
            handle: SceneHandle::new(),
            definition_id,
            state: SceneState::Pending,
            load_mode,
            entities: Vec::new(),
            load_progress: 0.0,
            error: None,
            visible: true,
            update_enabled: true,
            metadata: HashMap::new(),
            priority: 0,
        }
    }

    /// Adds an entity to this scene instance.
    pub fn add_entity(&mut self, entity: EntityId) {
        if !self.entities.contains(&entity) {
            self.entities.push(entity);
        }
    }

    /// Removes an entity from this scene instance.
    pub fn remove_entity(&mut self, entity: EntityId) {
        self.entities.retain(|e| *e != entity);
    }

    /// Whether the scene owns a given entity.
    pub fn owns_entity(&self, entity: EntityId) -> bool {
        self.entities.contains(&entity)
    }

    /// Number of entities in this scene.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Set scene metadata.
    pub fn set_meta(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get scene metadata.
    pub fn get_meta(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// Async load request
// ---------------------------------------------------------------------------

/// A queued scene load request.
#[derive(Debug, Clone)]
pub struct SceneLoadRequest {
    /// The scene to load.
    pub scene_id: SceneId,
    /// How to load it.
    pub mode: LoadSceneMode,
    /// Transition effect to use.
    pub transition: TransitionEffect,
    /// Priority (lower = load sooner).
    pub priority: i32,
    /// Whether to activate the scene immediately after loading.
    pub activate_on_load: bool,
}

impl SceneLoadRequest {
    /// Creates a simple single-scene load request.
    pub fn single(scene_id: SceneId) -> Self {
        Self {
            scene_id,
            mode: LoadSceneMode::Single,
            transition: TransitionEffect::None,
            priority: 0,
            activate_on_load: true,
        }
    }

    /// Creates an additive load request.
    pub fn additive(scene_id: SceneId) -> Self {
        Self {
            scene_id,
            mode: LoadSceneMode::Additive,
            transition: TransitionEffect::None,
            priority: 0,
            activate_on_load: true,
        }
    }

    /// Builder: sets the transition effect.
    pub fn with_transition(mut self, transition: TransitionEffect) -> Self {
        self.transition = transition;
        self
    }

    /// Builder: sets the priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Builder: sets whether to activate on load.
    pub fn with_activate_on_load(mut self, activate: bool) -> Self {
        self.activate_on_load = activate;
        self
    }
}

// ---------------------------------------------------------------------------
// Scene load progress
// ---------------------------------------------------------------------------

/// Progress information for an active scene load operation.
#[derive(Debug, Clone)]
pub struct LoadProgress {
    /// The scene being loaded.
    pub scene_id: SceneId,
    /// The handle assigned to the loading instance.
    pub handle: SceneHandle,
    /// Overall progress (0.0 .. 1.0).
    pub progress: f32,
    /// Number of assets loaded so far.
    pub assets_loaded: u32,
    /// Total number of assets to load.
    pub assets_total: u32,
    /// Current phase description.
    pub phase: String,
    /// Whether the load has completed.
    pub complete: bool,
    /// Whether the load failed.
    pub failed: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

impl LoadProgress {
    /// Creates initial progress for a new load.
    pub fn new(scene_id: SceneId, handle: SceneHandle) -> Self {
        Self {
            scene_id,
            handle,
            progress: 0.0,
            assets_loaded: 0,
            assets_total: 0,
            phase: "Initialising".to_string(),
            complete: false,
            failed: false,
            error: None,
        }
    }

    /// Updates the progress.
    pub fn update(&mut self, loaded: u32, total: u32, phase: &str) {
        self.assets_loaded = loaded;
        self.assets_total = total;
        self.phase = phase.to_string();
        if total > 0 {
            self.progress = loaded as f32 / total as f32;
        }
    }

    /// Marks the load as complete.
    pub fn mark_complete(&mut self) {
        self.progress = 1.0;
        self.complete = true;
        self.phase = "Complete".to_string();
    }

    /// Marks the load as failed.
    pub fn mark_failed(&mut self, error: &str) {
        self.failed = true;
        self.error = Some(error.to_string());
        self.phase = "Failed".to_string();
    }
}

// ---------------------------------------------------------------------------
// Scene stack entry
// ---------------------------------------------------------------------------

/// An entry in the scene stack (for push/pop navigation).
#[derive(Debug, Clone)]
pub struct SceneStackEntry {
    /// The handle of the stacked scene.
    pub handle: SceneHandle,
    /// The scene definition id.
    pub scene_id: SceneId,
    /// Whether the stacked scene should remain visible beneath the top.
    pub visible_underneath: bool,
    /// Whether the stacked scene should continue to receive updates.
    pub update_underneath: bool,
    /// Transition effect used when this entry was pushed.
    pub push_transition: TransitionEffect,
}

// ---------------------------------------------------------------------------
// Scene event
// ---------------------------------------------------------------------------

/// Events emitted by the scene manager.
#[derive(Debug, Clone)]
pub enum SceneEvent {
    /// A scene has started loading.
    LoadStarted {
        handle: SceneHandle,
        scene_id: SceneId,
    },
    /// A scene's load progress has been updated.
    LoadProgress {
        handle: SceneHandle,
        progress: f32,
    },
    /// A scene has finished loading.
    LoadCompleted {
        handle: SceneHandle,
        scene_id: SceneId,
    },
    /// A scene failed to load.
    LoadFailed {
        handle: SceneHandle,
        scene_id: SceneId,
        error: String,
    },
    /// A scene has been activated.
    Activated {
        handle: SceneHandle,
        scene_id: SceneId,
    },
    /// A scene has been deactivated.
    Deactivated {
        handle: SceneHandle,
        scene_id: SceneId,
    },
    /// A scene has been unloaded.
    Unloaded {
        handle: SceneHandle,
        scene_id: SceneId,
    },
    /// A transition has started.
    TransitionStarted {
        from: Option<SceneHandle>,
        to: SceneHandle,
    },
    /// A transition has completed.
    TransitionCompleted {
        from: Option<SceneHandle>,
        to: SceneHandle,
    },
    /// A scene was pushed onto the stack.
    ScenePushed {
        handle: SceneHandle,
    },
    /// A scene was popped from the stack.
    ScenePopped {
        handle: SceneHandle,
    },
}

// ---------------------------------------------------------------------------
// SceneManager
// ---------------------------------------------------------------------------

/// The central scene management system.
///
/// Manages the lifecycle of all scene instances, handles transitions,
/// scene stacking (push/pop for menus/overlays), DontDestroyOnLoad
/// entities, and dependency resolution.
pub struct SceneManager {
    /// All registered scene definitions (asset metadata).
    definitions: HashMap<SceneId, SceneDefinition>,

    /// All live scene instances keyed by handle.
    instances: HashMap<SceneHandle, SceneInstance>,

    /// The currently active "main" scene handle.
    active_scene: Option<SceneHandle>,

    /// Additive scenes that are currently active alongside the main scene.
    additive_scenes: Vec<SceneHandle>,

    /// The scene stack for push/pop navigation.
    scene_stack: Vec<SceneStackEntry>,

    /// Entities marked as DontDestroyOnLoad.
    persistent_entities: HashSet<EntityId>,

    /// Pending load requests (priority queue, processed each frame).
    load_queue: VecDeque<SceneLoadRequest>,

    /// Active load progress trackers.
    active_loads: HashMap<SceneHandle, LoadProgress>,

    /// Current transition state.
    transition_phase: TransitionPhase,

    /// The transition effect configuration for the current transition.
    current_transition: TransitionEffect,

    /// Handle of the scene being transitioned *to*.
    transition_target: Option<SceneHandle>,

    /// Handle of the scene being transitioned *from*.
    transition_source: Option<SceneHandle>,

    /// Event queue (drained each frame by the caller).
    events: VecDeque<SceneEvent>,

    /// Memory budget for loaded scenes (0 = unlimited).
    memory_budget: u64,

    /// Current estimated memory usage.
    memory_used: u64,

    /// Whether to allow multiple active scenes.
    allow_additive: bool,

    /// Maximum number of concurrent async loads.
    max_concurrent_loads: usize,

    /// Scene load history for analytics / debugging.
    load_history: Vec<LoadHistoryEntry>,

    /// Maximum history entries to keep.
    max_history: usize,
}

/// A record in the load history log.
#[derive(Debug, Clone)]
pub struct LoadHistoryEntry {
    /// Scene id.
    pub scene_id: SceneId,
    /// Scene name.
    pub name: String,
    /// Load mode.
    pub mode: LoadSceneMode,
    /// Timestamp (frame number or monotonic counter).
    pub timestamp: u64,
    /// Whether the load succeeded.
    pub success: bool,
    /// Load duration in seconds (0 if not measured).
    pub duration_secs: f32,
}

impl SceneManager {
    /// Creates a new, empty scene manager.
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            instances: HashMap::new(),
            active_scene: None,
            additive_scenes: Vec::new(),
            scene_stack: Vec::new(),
            persistent_entities: HashSet::new(),
            load_queue: VecDeque::new(),
            active_loads: HashMap::new(),
            transition_phase: TransitionPhase::Idle,
            current_transition: TransitionEffect::None,
            transition_target: None,
            transition_source: None,
            events: VecDeque::new(),
            memory_budget: 0,
            memory_used: 0,
            allow_additive: true,
            max_concurrent_loads: 2,
            load_history: Vec::new(),
            max_history: 100,
        }
    }

    // -- Definition management ------------------------------------------------

    /// Registers a scene definition.
    pub fn register_scene(&mut self, definition: SceneDefinition) {
        self.definitions.insert(definition.id, definition);
    }

    /// Unregisters a scene definition. Active instances are unaffected.
    pub fn unregister_scene(&mut self, id: SceneId) {
        self.definitions.remove(&id);
    }

    /// Returns a scene definition by id.
    pub fn get_definition(&self, id: SceneId) -> Option<&SceneDefinition> {
        self.definitions.get(&id)
    }

    /// Returns all registered scene definitions.
    pub fn definitions(&self) -> impl Iterator<Item = &SceneDefinition> {
        self.definitions.values()
    }

    /// Finds scene definitions by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<&SceneDefinition> {
        self.definitions
            .values()
            .filter(|d| d.has_tag(tag))
            .collect()
    }

    /// Finds a scene definition by name.
    pub fn find_by_name(&self, name: &str) -> Option<&SceneDefinition> {
        self.definitions.values().find(|d| d.name == name)
    }

    /// Returns the number of registered definitions.
    pub fn definition_count(&self) -> usize {
        self.definitions.len()
    }

    // -- Load / unload --------------------------------------------------------

    /// Queues a scene for loading.
    ///
    /// Returns the handle that will be assigned to the loaded instance.
    pub fn load_scene(&mut self, request: SceneLoadRequest) -> SceneHandle {
        let scene_id = request.scene_id;
        let mode = request.mode;

        // Create the instance immediately so we can return a handle.
        let mut instance = SceneInstance::new(scene_id, mode);
        let handle = instance.handle;
        instance.state = SceneState::Pending;
        self.instances.insert(handle, instance);

        // Create progress tracker.
        let progress = LoadProgress::new(scene_id, handle);
        self.active_loads.insert(handle, progress);

        // Enqueue the request.
        self.load_queue.push_back(request);

        // Emit event.
        self.events.push_back(SceneEvent::LoadStarted {
            handle,
            scene_id,
        });

        handle
    }

    /// Loads a scene synchronously (blocking).
    ///
    /// This is a convenience wrapper that creates a request, processes
    /// the load immediately, and returns the handle.
    pub fn load_scene_sync(
        &mut self,
        scene_id: SceneId,
        mode: LoadSceneMode,
    ) -> Result<SceneHandle, String> {
        // Resolve dependencies first.
        let missing = self.resolve_dependencies(scene_id);
        if !missing.is_empty() {
            return Err(format!(
                "Missing required dependencies: {:?}",
                missing
            ));
        }

        let mut instance = SceneInstance::new(scene_id, mode);
        let handle = instance.handle;

        // Simulate synchronous load.
        instance.state = SceneState::Loading;
        instance.load_progress = 0.5;

        // "Load" completes instantly.
        instance.state = SceneState::Loaded;
        instance.load_progress = 1.0;

        // Handle scene replacement for Single mode.
        if mode == LoadSceneMode::Single {
            self.unload_non_persistent();
        }

        // Activate.
        instance.state = SceneState::Active;
        self.instances.insert(handle, instance);

        // Update active scene tracking.
        match mode {
            LoadSceneMode::Single => {
                self.active_scene = Some(handle);
                self.additive_scenes.clear();
            }
            LoadSceneMode::Additive => {
                self.additive_scenes.push(handle);
            }
        }

        self.events.push_back(SceneEvent::LoadCompleted {
            handle,
            scene_id,
        });
        self.events.push_back(SceneEvent::Activated {
            handle,
            scene_id,
        });

        // Record history.
        if let Some(def) = self.definitions.get(&scene_id) {
            self.record_history(scene_id, &def.name.clone(), mode, true, 0.0);
        }

        Ok(handle)
    }

    /// Unloads a scene instance by handle.
    pub fn unload_scene(&mut self, handle: SceneHandle) {
        if let Some(instance) = self.instances.get_mut(&handle) {
            let scene_id = instance.definition_id;

            // Move persistent entities out before unloading.
            instance
                .entities
                .retain(|e| !self.persistent_entities.contains(e));

            instance.state = SceneState::Unloading;

            self.events.push_back(SceneEvent::Deactivated {
                handle,
                scene_id,
            });

            instance.state = SceneState::Unloaded;
            instance.entities.clear();

            self.events.push_back(SceneEvent::Unloaded {
                handle,
                scene_id,
            });

            // Update estimated memory.
            if let Some(def) = self.definitions.get(&scene_id) {
                self.memory_used = self.memory_used.saturating_sub(def.estimated_memory);
            }
        }

        // Remove from tracking.
        self.instances.remove(&handle);
        self.active_loads.remove(&handle);

        if self.active_scene == Some(handle) {
            self.active_scene = None;
        }
        self.additive_scenes.retain(|h| *h != handle);
    }

    /// Unloads all scenes except persistent entities.
    pub fn unload_all(&mut self) {
        let handles: Vec<SceneHandle> = self.instances.keys().copied().collect();
        for handle in handles {
            self.unload_scene(handle);
        }
    }

    /// Unloads all non-persistent entities across all active scenes.
    fn unload_non_persistent(&mut self) {
        for instance in self.instances.values_mut() {
            instance
                .entities
                .retain(|e| self.persistent_entities.contains(e));
        }
    }

    // -- Active scene queries -------------------------------------------------

    /// Returns the handle of the currently active main scene.
    pub fn active_scene(&self) -> Option<SceneHandle> {
        self.active_scene
    }

    /// Returns the active main scene instance.
    pub fn active_scene_instance(&self) -> Option<&SceneInstance> {
        self.active_scene
            .and_then(|h| self.instances.get(&h))
    }

    /// Returns the active main scene instance (mutable).
    pub fn active_scene_instance_mut(&mut self) -> Option<&mut SceneInstance> {
        self.active_scene
            .and_then(move |h| self.instances.get_mut(&h))
    }

    /// Returns all additive scene handles.
    pub fn additive_scenes(&self) -> &[SceneHandle] {
        &self.additive_scenes
    }

    /// Returns all currently active scene handles (main + additive).
    pub fn all_active_handles(&self) -> Vec<SceneHandle> {
        let mut handles = Vec::new();
        if let Some(h) = self.active_scene {
            handles.push(h);
        }
        handles.extend_from_slice(&self.additive_scenes);
        handles
    }

    /// Returns a scene instance by handle.
    pub fn get_instance(&self, handle: SceneHandle) -> Option<&SceneInstance> {
        self.instances.get(&handle)
    }

    /// Returns a scene instance by handle (mutable).
    pub fn get_instance_mut(&mut self, handle: SceneHandle) -> Option<&mut SceneInstance> {
        self.instances.get_mut(&handle)
    }

    /// Returns the total number of live scene instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Returns all scene instance handles.
    pub fn all_instance_handles(&self) -> Vec<SceneHandle> {
        self.instances.keys().copied().collect()
    }

    // -- DontDestroyOnLoad ----------------------------------------------------

    /// Marks an entity as persistent (DontDestroyOnLoad).
    pub fn mark_persistent(&mut self, entity: EntityId) {
        self.persistent_entities.insert(entity);
    }

    /// Removes the persistent flag from an entity.
    pub fn unmark_persistent(&mut self, entity: EntityId) {
        self.persistent_entities.remove(&entity);
    }

    /// Returns whether an entity is marked as persistent.
    pub fn is_persistent(&self, entity: EntityId) -> bool {
        self.persistent_entities.contains(&entity)
    }

    /// Returns all persistent entity ids.
    pub fn persistent_entities(&self) -> &HashSet<EntityId> {
        &self.persistent_entities
    }

    /// Returns the number of persistent entities.
    pub fn persistent_entity_count(&self) -> usize {
        self.persistent_entities.len()
    }

    // -- Scene stacking -------------------------------------------------------

    /// Pushes a scene onto the stack and loads it.
    ///
    /// The previously active scene is paused and remains in memory.
    pub fn push_scene(
        &mut self,
        scene_id: SceneId,
        transition: TransitionEffect,
        visible_underneath: bool,
        update_underneath: bool,
    ) -> SceneHandle {
        // Save current active scene to the stack.
        if let Some(active_handle) = self.active_scene {
            let entry = SceneStackEntry {
                handle: active_handle,
                scene_id: self
                    .instances
                    .get(&active_handle)
                    .map(|i| i.definition_id)
                    .unwrap_or(SceneId::from_raw(0)),
                visible_underneath,
                update_underneath,
                push_transition: transition.clone(),
            };
            self.scene_stack.push(entry);

            // Optionally hide/pause the current scene.
            if let Some(instance) = self.instances.get_mut(&active_handle) {
                instance.visible = visible_underneath;
                instance.update_enabled = update_underneath;
            }
        }

        // Load the new scene.
        let handle = self
            .load_scene_sync(scene_id, LoadSceneMode::Single)
            .unwrap_or_else(|_| SceneHandle::INVALID);

        self.events.push_back(SceneEvent::ScenePushed { handle });

        handle
    }

    /// Pops the top scene from the stack and restores the previous scene.
    pub fn pop_scene(&mut self) -> Option<SceneHandle> {
        let entry = self.scene_stack.pop()?;

        // Unload the current active scene.
        if let Some(active) = self.active_scene {
            self.events.push_back(SceneEvent::ScenePopped {
                handle: active,
            });
            self.unload_scene(active);
        }

        // Restore the stacked scene.
        self.active_scene = Some(entry.handle);
        if let Some(instance) = self.instances.get_mut(&entry.handle) {
            instance.visible = true;
            instance.update_enabled = true;
            instance.state = SceneState::Active;
        }

        Some(entry.handle)
    }

    /// Returns the depth of the scene stack.
    pub fn stack_depth(&self) -> usize {
        self.scene_stack.len()
    }

    /// Returns whether there are scenes on the stack.
    pub fn has_stacked_scenes(&self) -> bool {
        !self.scene_stack.is_empty()
    }

    /// Peeks at the top of the scene stack without popping.
    pub fn peek_stack(&self) -> Option<&SceneStackEntry> {
        self.scene_stack.last()
    }

    // -- Transitions ----------------------------------------------------------

    /// Starts a transition to a new scene.
    pub fn start_transition(
        &mut self,
        target_scene_id: SceneId,
        mode: LoadSceneMode,
        effect: TransitionEffect,
    ) -> SceneHandle {
        let target_instance = SceneInstance::new(target_scene_id, mode);
        let target_handle = target_instance.handle;
        self.instances.insert(target_handle, target_instance);

        self.current_transition = effect.clone();
        self.transition_source = self.active_scene;
        self.transition_target = Some(target_handle);

        // Begin the appropriate phase.
        match &effect {
            TransitionEffect::None => {
                // Instant transition.
                self.complete_transition();
            }
            TransitionEffect::Fade {
                fade_out_duration, ..
            } => {
                self.transition_phase = TransitionPhase::FadingOut {
                    elapsed: 0.0,
                    duration: *fade_out_duration,
                };
            }
            TransitionEffect::Dissolve { duration } => {
                self.transition_phase = TransitionPhase::Dissolving {
                    blend: 0.0,
                    duration: *duration,
                    elapsed: 0.0,
                };
            }
            TransitionEffect::Wipe {
                direction,
                duration,
            } => {
                self.transition_phase = TransitionPhase::Wiping {
                    progress: 0.0,
                    direction: *direction,
                    duration: *duration,
                    elapsed: 0.0,
                };
            }
            TransitionEffect::Custom { duration, .. } => {
                self.transition_phase = TransitionPhase::FadingOut {
                    elapsed: 0.0,
                    duration: *duration * 0.5,
                };
            }
        }

        self.events.push_back(SceneEvent::TransitionStarted {
            from: self.transition_source,
            to: target_handle,
        });

        target_handle
    }

    /// Updates the transition state machine. Call once per frame with delta time.
    pub fn update_transition(&mut self, dt: f32) {
        match &mut self.transition_phase {
            TransitionPhase::Idle => {}
            TransitionPhase::FadingOut { elapsed, duration } => {
                *elapsed += dt;
                if *elapsed >= *duration {
                    // Start loading the new scene.
                    self.transition_phase = TransitionPhase::LoadingNewScene;
                    // In a real implementation, we would kick off the async
                    // load here. For now, go straight to fading in.
                    if let TransitionEffect::Fade {
                        fade_in_duration, ..
                    } = &self.current_transition
                    {
                        self.transition_phase = TransitionPhase::FadingIn {
                            elapsed: 0.0,
                            duration: *fade_in_duration,
                        };
                    } else {
                        self.complete_transition();
                    }
                }
            }
            TransitionPhase::LoadingNewScene => {
                // Check if the target scene has finished loading.
                if let Some(target) = self.transition_target {
                    if let Some(instance) = self.instances.get(&target) {
                        if instance.state.is_loaded() {
                            if let TransitionEffect::Fade {
                                fade_in_duration, ..
                            } = &self.current_transition
                            {
                                self.transition_phase = TransitionPhase::FadingIn {
                                    elapsed: 0.0,
                                    duration: *fade_in_duration,
                                };
                            } else {
                                self.complete_transition();
                            }
                        }
                    }
                }
            }
            TransitionPhase::FadingIn { elapsed, duration } => {
                *elapsed += dt;
                if *elapsed >= *duration {
                    self.complete_transition();
                }
            }
            TransitionPhase::Dissolving {
                blend,
                duration,
                elapsed,
            } => {
                *elapsed += dt;
                *blend = (*elapsed / *duration).clamp(0.0, 1.0);
                if *elapsed >= *duration {
                    self.complete_transition();
                }
            }
            TransitionPhase::Wiping {
                progress,
                duration,
                elapsed,
                ..
            } => {
                *elapsed += dt;
                *progress = (*elapsed / *duration).clamp(0.0, 1.0);
                if *elapsed >= *duration {
                    self.complete_transition();
                }
            }
        }
    }

    /// Completes the current transition, activating the target scene.
    fn complete_transition(&mut self) {
        if let Some(target) = self.transition_target {
            // Deactivate the source scene.
            if let Some(source) = self.transition_source {
                if let Some(instance) = self.instances.get_mut(&source) {
                    instance.state = SceneState::Loaded;
                    instance.visible = false;
                }
            }

            // Activate the target scene.
            if let Some(instance) = self.instances.get_mut(&target) {
                instance.state = SceneState::Active;
                instance.visible = true;
            }

            self.active_scene = Some(target);

            self.events.push_back(SceneEvent::TransitionCompleted {
                from: self.transition_source,
                to: target,
            });
        }

        self.transition_phase = TransitionPhase::Idle;
        self.transition_source = None;
        self.transition_target = None;
        self.current_transition = TransitionEffect::None;
    }

    /// Returns the current transition phase.
    pub fn transition_phase(&self) -> &TransitionPhase {
        &self.transition_phase
    }

    /// Whether a transition is currently in progress.
    pub fn is_transitioning(&self) -> bool {
        self.transition_phase.is_active()
    }

    // -- Dependency resolution ------------------------------------------------

    /// Resolves dependencies for a scene, returning any that are missing.
    pub fn resolve_dependencies(&self, scene_id: SceneId) -> Vec<SceneId> {
        let mut missing = Vec::new();

        if let Some(def) = self.definitions.get(&scene_id) {
            for dep in &def.dependencies {
                if dep.required && !self.is_scene_loaded(dep.dependency) {
                    missing.push(dep.dependency);
                }
            }
        }

        missing
    }

    /// Resolves dependencies recursively (transitive closure).
    pub fn resolve_dependencies_recursive(&self, scene_id: SceneId) -> Vec<SceneId> {
        let mut all_deps = Vec::new();
        let mut visited = HashSet::new();
        self.collect_dependencies(scene_id, &mut all_deps, &mut visited);
        all_deps
    }

    /// Helper: collects transitive dependencies via DFS.
    fn collect_dependencies(
        &self,
        scene_id: SceneId,
        deps: &mut Vec<SceneId>,
        visited: &mut HashSet<SceneId>,
    ) {
        if !visited.insert(scene_id) {
            return;
        }

        if let Some(def) = self.definitions.get(&scene_id) {
            for dep in &def.dependencies {
                if dep.required {
                    deps.push(dep.dependency);
                    self.collect_dependencies(dep.dependency, deps, visited);
                }
            }
        }
    }

    /// Returns whether a scene (by definition id) is currently loaded.
    pub fn is_scene_loaded(&self, scene_id: SceneId) -> bool {
        self.instances
            .values()
            .any(|i| i.definition_id == scene_id && i.state.is_loaded())
    }

    /// Checks for cyclic dependencies. Returns true if a cycle is detected.
    pub fn has_dependency_cycle(&self, scene_id: SceneId) -> bool {
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        self.detect_cycle(scene_id, &mut visited, &mut stack)
    }

    /// DFS cycle detection.
    fn detect_cycle(
        &self,
        scene_id: SceneId,
        visited: &mut HashSet<SceneId>,
        stack: &mut HashSet<SceneId>,
    ) -> bool {
        if stack.contains(&scene_id) {
            return true; // Cycle!
        }
        if visited.contains(&scene_id) {
            return false;
        }

        visited.insert(scene_id);
        stack.insert(scene_id);

        if let Some(def) = self.definitions.get(&scene_id) {
            for dep in &def.dependencies {
                if self.detect_cycle(dep.dependency, visited, stack) {
                    return true;
                }
            }
        }

        stack.remove(&scene_id);
        false
    }

    // -- Event polling --------------------------------------------------------

    /// Drains the event queue and returns all pending events.
    pub fn drain_events(&mut self) -> Vec<SceneEvent> {
        self.events.drain(..).collect()
    }

    /// Returns the number of pending events.
    pub fn pending_event_count(&self) -> usize {
        self.events.len()
    }

    // -- Progress tracking ----------------------------------------------------

    /// Returns the load progress for a given handle.
    pub fn load_progress(&self, handle: SceneHandle) -> Option<&LoadProgress> {
        self.active_loads.get(&handle)
    }

    /// Simulates a progress update (used by the async loader in a real build).
    pub fn update_load_progress(
        &mut self,
        handle: SceneHandle,
        loaded: u32,
        total: u32,
        phase: &str,
    ) {
        if let Some(progress) = self.active_loads.get_mut(&handle) {
            progress.update(loaded, total, phase);
            let p = progress.progress;
            self.events.push_back(SceneEvent::LoadProgress {
                handle,
                progress: p,
            });
        }

        if let Some(instance) = self.instances.get_mut(&handle) {
            if total > 0 {
                instance.load_progress = loaded as f32 / total as f32;
            }
        }
    }

    /// Completes a load operation (called when async loading finishes).
    pub fn complete_load(&mut self, handle: SceneHandle) {
        if let Some(progress) = self.active_loads.get_mut(&handle) {
            progress.mark_complete();
        }

        if let Some(instance) = self.instances.get_mut(&handle) {
            instance.state = SceneState::Loaded;
            instance.load_progress = 1.0;

            let scene_id = instance.definition_id;
            self.events.push_back(SceneEvent::LoadCompleted {
                handle,
                scene_id,
            });
        }
    }

    /// Fails a load operation.
    pub fn fail_load(&mut self, handle: SceneHandle, error: &str) {
        if let Some(progress) = self.active_loads.get_mut(&handle) {
            progress.mark_failed(error);
        }

        if let Some(instance) = self.instances.get_mut(&handle) {
            instance.state = SceneState::Failed;
            instance.error = Some(error.to_string());

            let scene_id = instance.definition_id;
            self.events.push_back(SceneEvent::LoadFailed {
                handle,
                scene_id,
                error: error.to_string(),
            });
        }
    }

    // -- Memory budgeting -----------------------------------------------------

    /// Sets the memory budget for loaded scenes.
    pub fn set_memory_budget(&mut self, budget: u64) {
        self.memory_budget = budget;
    }

    /// Returns the current memory budget.
    pub fn memory_budget(&self) -> u64 {
        self.memory_budget
    }

    /// Returns the estimated memory currently used by loaded scenes.
    pub fn memory_used(&self) -> u64 {
        self.memory_used
    }

    /// Returns the estimated memory remaining before the budget is exceeded.
    pub fn memory_remaining(&self) -> u64 {
        if self.memory_budget == 0 {
            return u64::MAX;
        }
        self.memory_budget.saturating_sub(self.memory_used)
    }

    /// Whether loading another scene would exceed the memory budget.
    pub fn would_exceed_budget(&self, scene_id: SceneId) -> bool {
        if self.memory_budget == 0 {
            return false;
        }
        if let Some(def) = self.definitions.get(&scene_id) {
            self.memory_used + def.estimated_memory > self.memory_budget
        } else {
            false
        }
    }

    // -- Configuration --------------------------------------------------------

    /// Sets the maximum number of concurrent async loads.
    pub fn set_max_concurrent_loads(&mut self, max: usize) {
        self.max_concurrent_loads = max.max(1);
    }

    /// Enables or disables additive scene loading.
    pub fn set_allow_additive(&mut self, allow: bool) {
        self.allow_additive = allow;
    }

    // -- History / analytics --------------------------------------------------

    fn record_history(
        &mut self,
        scene_id: SceneId,
        name: &str,
        mode: LoadSceneMode,
        success: bool,
        duration: f32,
    ) {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        self.load_history.push(LoadHistoryEntry {
            scene_id,
            name: name.to_string(),
            mode,
            timestamp: COUNTER.fetch_add(1, Ordering::Relaxed),
            success,
            duration_secs: duration,
        });
        if self.load_history.len() > self.max_history {
            self.load_history.remove(0);
        }
    }

    /// Returns the load history.
    pub fn load_history(&self) -> &[LoadHistoryEntry] {
        &self.load_history
    }

    /// Clears the load history.
    pub fn clear_history(&mut self) {
        self.load_history.clear();
    }

    // -- Per-frame update -----------------------------------------------------

    /// Main per-frame update. Call once per frame with the frame delta time.
    ///
    /// Processes queued loads, updates transitions, and drains completed loads.
    pub fn update(&mut self, dt: f32) {
        // Process queued loads up to concurrency limit.
        let current_loading = self
            .instances
            .values()
            .filter(|i| i.state == SceneState::Loading)
            .count();

        let available_slots = self.max_concurrent_loads.saturating_sub(current_loading);
        for _ in 0..available_slots {
            if let Some(request) = self.load_queue.pop_front() {
                // Find the instance we already created for this request.
                let handle = self
                    .instances
                    .iter()
                    .find(|(_, inst)| {
                        inst.definition_id == request.scene_id
                            && inst.state == SceneState::Pending
                    })
                    .map(|(h, _)| *h);

                if let Some(handle) = handle {
                    if let Some(instance) = self.instances.get_mut(&handle) {
                        instance.state = SceneState::Loading;
                    }
                }
            }
        }

        // Update transition.
        self.update_transition(dt);
    }
}

impl Default for SceneManager {
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

    fn make_definition(name: &str) -> SceneDefinition {
        SceneDefinition::new(name, &format!("scenes/{}.scene", name))
    }

    #[test]
    fn scene_id_uniqueness() {
        let a = SceneId::new();
        let b = SceneId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn scene_handle_validity() {
        let h = SceneHandle::new();
        assert!(h.is_valid());
        assert!(!SceneHandle::INVALID.is_valid());
    }

    #[test]
    fn register_and_find_definition() {
        let mut mgr = SceneManager::new();
        let mut def = make_definition("level_01");
        def.add_tag("gameplay");
        mgr.register_scene(def.clone());

        assert!(mgr.get_definition(def.id).is_some());
        assert!(mgr.find_by_name("level_01").is_some());
        assert_eq!(mgr.find_by_tag("gameplay").len(), 1);
        assert_eq!(mgr.definition_count(), 1);
    }

    #[test]
    fn load_scene_sync_single() {
        let mut mgr = SceneManager::new();
        let def = make_definition("main_menu");
        let id = def.id;
        mgr.register_scene(def);

        let handle = mgr.load_scene_sync(id, LoadSceneMode::Single).unwrap();
        assert!(handle.is_valid());
        assert_eq!(mgr.active_scene(), Some(handle));

        let instance = mgr.get_instance(handle).unwrap();
        assert_eq!(instance.state, SceneState::Active);
    }

    #[test]
    fn load_scene_additive() {
        let mut mgr = SceneManager::new();
        let def1 = make_definition("world");
        let def2 = make_definition("ui_overlay");
        let id1 = def1.id;
        let id2 = def2.id;
        mgr.register_scene(def1);
        mgr.register_scene(def2);

        let h1 = mgr.load_scene_sync(id1, LoadSceneMode::Single).unwrap();
        let h2 = mgr.load_scene_sync(id2, LoadSceneMode::Additive).unwrap();

        assert_eq!(mgr.active_scene(), Some(h1));
        assert_eq!(mgr.additive_scenes().len(), 1);
        assert_eq!(mgr.all_active_handles().len(), 2);
        assert!(mgr.all_active_handles().contains(&h2));
    }

    #[test]
    fn unload_scene() {
        let mut mgr = SceneManager::new();
        let def = make_definition("level");
        let id = def.id;
        mgr.register_scene(def);

        let handle = mgr.load_scene_sync(id, LoadSceneMode::Single).unwrap();
        mgr.unload_scene(handle);

        assert!(mgr.active_scene().is_none());
        assert!(mgr.get_instance(handle).is_none());
    }

    #[test]
    fn persistent_entities() {
        let mut mgr = SceneManager::new();
        let entity = EntityId::new(42);

        mgr.mark_persistent(entity);
        assert!(mgr.is_persistent(entity));
        assert_eq!(mgr.persistent_entity_count(), 1);

        mgr.unmark_persistent(entity);
        assert!(!mgr.is_persistent(entity));
    }

    #[test]
    fn scene_stack_push_pop() {
        let mut mgr = SceneManager::new();
        let def1 = make_definition("gameplay");
        let def2 = make_definition("pause_menu");
        let id1 = def1.id;
        let id2 = def2.id;
        mgr.register_scene(def1);
        mgr.register_scene(def2);

        // Load initial scene.
        let _h1 = mgr.load_scene_sync(id1, LoadSceneMode::Single).unwrap();

        // Push pause menu.
        let h2 = mgr.push_scene(id2, TransitionEffect::None, false, false);
        assert!(h2.is_valid());
        assert_eq!(mgr.stack_depth(), 1);
        assert!(mgr.has_stacked_scenes());

        // Pop back to gameplay.
        let restored = mgr.pop_scene();
        assert!(restored.is_some());
        assert_eq!(mgr.stack_depth(), 0);
    }

    #[test]
    fn transition_fade() {
        let mut mgr = SceneManager::new();
        let def1 = make_definition("scene_a");
        let def2 = make_definition("scene_b");
        let id1 = def1.id;
        let id2 = def2.id;
        mgr.register_scene(def1);
        mgr.register_scene(def2);

        mgr.load_scene_sync(id1, LoadSceneMode::Single).unwrap();

        let _target = mgr.start_transition(
            id2,
            LoadSceneMode::Single,
            TransitionEffect::fade_black(0.5, 0.5),
        );
        assert!(mgr.is_transitioning());

        // Simulate time passing.
        mgr.update_transition(0.6); // Finish fade-out, enter fade-in.
        mgr.update_transition(0.6); // Finish fade-in.

        assert!(!mgr.is_transitioning());
    }

    #[test]
    fn transition_dissolve() {
        let mut mgr = SceneManager::new();
        let def = make_definition("scene_dissolve");
        let id = def.id;
        mgr.register_scene(def);

        let _target = mgr.start_transition(
            id,
            LoadSceneMode::Single,
            TransitionEffect::dissolve(1.0),
        );
        assert!(mgr.is_transitioning());

        mgr.update_transition(0.5);
        assert!(mgr.is_transitioning());

        mgr.update_transition(0.6);
        assert!(!mgr.is_transitioning());
    }

    #[test]
    fn transition_wipe() {
        let mut mgr = SceneManager::new();
        let def = make_definition("scene_wipe");
        let id = def.id;
        mgr.register_scene(def);

        let _target = mgr.start_transition(
            id,
            LoadSceneMode::Single,
            TransitionEffect::wipe_left_to_right(0.8),
        );

        mgr.update_transition(0.9);
        assert!(!mgr.is_transitioning());
    }

    #[test]
    fn dependency_resolution() {
        let mut mgr = SceneManager::new();
        let mut def_a = make_definition("base");
        let mut def_b = make_definition("level");
        let id_a = def_a.id;
        let id_b = def_b.id;

        def_b.add_dependency(id_a, "Needs base assets");

        mgr.register_scene(def_a);
        mgr.register_scene(def_b);

        // Base not loaded -- should report missing dependency.
        let missing = mgr.resolve_dependencies(id_b);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0], id_a);

        // Load base.
        mgr.load_scene_sync(id_a, LoadSceneMode::Single).unwrap();

        // Now no missing dependencies.
        let missing = mgr.resolve_dependencies(id_b);
        assert!(missing.is_empty());
    }

    #[test]
    fn dependency_cycle_detection() {
        let mut mgr = SceneManager::new();
        let mut def_a = make_definition("a");
        let mut def_b = make_definition("b");
        let id_a = def_a.id;
        let id_b = def_b.id;

        def_a.add_dependency(id_b, "a needs b");
        def_b.add_dependency(id_a, "b needs a");

        mgr.register_scene(def_a);
        mgr.register_scene(def_b);

        assert!(mgr.has_dependency_cycle(id_a));
        assert!(mgr.has_dependency_cycle(id_b));
    }

    #[test]
    fn memory_budget() {
        let mut mgr = SceneManager::new();
        mgr.set_memory_budget(1024 * 1024); // 1 MB

        let mut def = make_definition("heavy");
        def.estimated_memory = 2 * 1024 * 1024; // 2 MB
        let id = def.id;
        mgr.register_scene(def);

        assert!(mgr.would_exceed_budget(id));
    }

    #[test]
    fn async_load_queue() {
        let mut mgr = SceneManager::new();
        let def = make_definition("async_scene");
        let id = def.id;
        mgr.register_scene(def);

        let request = SceneLoadRequest::single(id);
        let handle = mgr.load_scene(request);

        assert!(handle.is_valid());
        assert!(mgr.load_progress(handle).is_some());
    }

    #[test]
    fn load_progress_tracking() {
        let mut mgr = SceneManager::new();
        let def = make_definition("progress_scene");
        let id = def.id;
        mgr.register_scene(def);

        let request = SceneLoadRequest::single(id);
        let handle = mgr.load_scene(request);

        mgr.update_load_progress(handle, 5, 10, "Loading meshes");
        let progress = mgr.load_progress(handle).unwrap();
        assert!((progress.progress - 0.5).abs() < 0.01);
        assert_eq!(progress.phase, "Loading meshes");

        mgr.complete_load(handle);
        let progress = mgr.load_progress(handle).unwrap();
        assert!(progress.complete);
    }

    #[test]
    fn load_failure() {
        let mut mgr = SceneManager::new();
        let def = make_definition("broken_scene");
        let id = def.id;
        mgr.register_scene(def);

        let request = SceneLoadRequest::single(id);
        let handle = mgr.load_scene(request);

        mgr.fail_load(handle, "File not found");

        let instance = mgr.get_instance(handle).unwrap();
        assert_eq!(instance.state, SceneState::Failed);

        let progress = mgr.load_progress(handle).unwrap();
        assert!(progress.failed);
    }

    #[test]
    fn events_emitted() {
        let mut mgr = SceneManager::new();
        let def = make_definition("event_test");
        let id = def.id;
        mgr.register_scene(def);

        mgr.load_scene_sync(id, LoadSceneMode::Single).unwrap();

        let events = mgr.drain_events();
        assert!(events.len() >= 2); // LoadCompleted + Activated at minimum
    }

    #[test]
    fn unload_all() {
        let mut mgr = SceneManager::new();
        let def1 = make_definition("s1");
        let def2 = make_definition("s2");
        let id1 = def1.id;
        let id2 = def2.id;
        mgr.register_scene(def1);
        mgr.register_scene(def2);

        mgr.load_scene_sync(id1, LoadSceneMode::Single).unwrap();
        mgr.load_scene_sync(id2, LoadSceneMode::Additive).unwrap();

        mgr.unload_all();
        assert_eq!(mgr.instance_count(), 0);
        assert!(mgr.active_scene().is_none());
    }

    #[test]
    fn scene_state_checks() {
        assert!(SceneState::Active.is_active());
        assert!(!SceneState::Loading.is_active());
        assert!(SceneState::Loading.is_loading());
        assert!(SceneState::Pending.is_loading());
        assert!(SceneState::Loaded.is_loaded());
        assert!(SceneState::Unloading.is_unloading());
    }

    #[test]
    fn transition_effect_duration() {
        let fade = TransitionEffect::fade_black(0.5, 0.3);
        assert!((fade.total_duration() - 0.8).abs() < 0.001);

        let dissolve = TransitionEffect::dissolve(1.0);
        assert!((dissolve.total_duration() - 1.0).abs() < 0.001);

        let wipe = TransitionEffect::wipe_left_to_right(0.6);
        assert!((wipe.total_duration() - 0.6).abs() < 0.001);

        let none = TransitionEffect::None;
        assert_eq!(none.total_duration(), 0.0);
    }

    #[test]
    fn scene_definition_builder() {
        let mut def = SceneDefinition::new("test", "scenes/test.scene");
        def.add_tag("gameplay");
        def.add_tag("outdoor");
        def.estimated_memory = 512 * 1024;

        assert!(def.has_tag("gameplay"));
        assert!(def.has_tag("outdoor"));
        assert!(!def.has_tag("indoor"));
    }

    #[test]
    fn load_request_builder() {
        let id = SceneId::new();
        let req = SceneLoadRequest::single(id)
            .with_transition(TransitionEffect::fade_black(0.3, 0.3))
            .with_priority(5)
            .with_activate_on_load(false);

        assert_eq!(req.scene_id, id);
        assert_eq!(req.priority, 5);
        assert!(!req.activate_on_load);
    }

    #[test]
    fn scene_instance_entity_management() {
        let id = SceneId::new();
        let mut instance = SceneInstance::new(id, LoadSceneMode::Single);

        instance.add_entity(EntityId::new(1));
        instance.add_entity(EntityId::new(2));
        instance.add_entity(EntityId::new(1)); // Duplicate, should not add.

        assert_eq!(instance.entity_count(), 2);
        assert!(instance.owns_entity(EntityId::new(1)));
        assert!(!instance.owns_entity(EntityId::new(3)));

        instance.remove_entity(EntityId::new(1));
        assert_eq!(instance.entity_count(), 1);
    }

    #[test]
    fn load_history() {
        let mut mgr = SceneManager::new();
        let def = make_definition("history_test");
        let id = def.id;
        mgr.register_scene(def);

        mgr.load_scene_sync(id, LoadSceneMode::Single).unwrap();

        assert!(!mgr.load_history().is_empty());
        let entry = &mgr.load_history()[0];
        assert!(entry.success);

        mgr.clear_history();
        assert!(mgr.load_history().is_empty());
    }

    #[test]
    fn per_frame_update() {
        let mut mgr = SceneManager::new();
        let def = make_definition("update_test");
        let id = def.id;
        mgr.register_scene(def);

        let request = SceneLoadRequest::single(id);
        let _handle = mgr.load_scene(request);

        // Should process queued load.
        mgr.update(0.016);

        // The instance should now be in Loading state.
        // (The instance was created in Pending state.)
    }

    #[test]
    fn scene_metadata() {
        let id = SceneId::new();
        let mut instance = SceneInstance::new(id, LoadSceneMode::Single);

        instance.set_meta("author", "engine_test");
        assert_eq!(instance.get_meta("author"), Some("engine_test"));
        assert_eq!(instance.get_meta("missing"), None);
    }

    #[test]
    fn recursive_dependencies() {
        let mut mgr = SceneManager::new();
        let mut def_a = make_definition("base");
        let mut def_b = make_definition("mid");
        let mut def_c = make_definition("top");
        let id_a = def_a.id;
        let id_b = def_b.id;
        let id_c = def_c.id;

        def_b.add_dependency(id_a, "mid needs base");
        def_c.add_dependency(id_b, "top needs mid");

        mgr.register_scene(def_a);
        mgr.register_scene(def_b);
        mgr.register_scene(def_c);

        let deps = mgr.resolve_dependencies_recursive(id_c);
        assert!(deps.contains(&id_b));
        assert!(deps.contains(&id_a));
    }
}
