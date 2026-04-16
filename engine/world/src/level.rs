//! Level Management
//!
//! Provides level loading, unloading, seamless transitions, and sub-level
//! management. A [`Level`] is a named collection of world cells with
//! associated metadata (cell size, streaming configuration, spawn points).
//!
//! # Concepts
//!
//! - **Level**: A self-contained map/environment with its own cell data
//!   directory, streaming settings, and spawn points.
//! - **Sub-level**: An additional level loaded additively into the world.
//!   Sub-levels share the same coordinate space and are layered on top of
//!   the base level.
//! - **Level travel**: Seamless or loading-screen transition between levels.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use glam::Vec3;

use crate::partition::{CellCoord, WorldLayer, WorldPartition};
use crate::streaming::{StreamingManager, StreamRequestPriority};

// ---------------------------------------------------------------------------
// LevelSettings
// ---------------------------------------------------------------------------

/// Configuration for a level's world partition and streaming behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelSettings {
    /// Size of each cell in world units.
    pub cell_size: f32,
    /// Base load radius for streaming.
    pub load_radius: f32,
    /// Base unload radius (must be >= load_radius).
    pub unload_radius: f32,
    /// Maximum concurrent cell loads.
    pub max_concurrent_loads: usize,
    /// Maximum concurrent cell unloads.
    pub max_concurrent_unloads: usize,
    /// Memory budget for loaded cells in bytes.
    pub memory_budget: usize,
    /// Streaming layers with per-layer distances.
    pub layers: Vec<WorldLayer>,
    /// Gravity vector for this level.
    pub gravity: [f32; 3],
    /// Ambient light color [r, g, b].
    pub ambient_color: [f32; 3],
    /// Ambient light intensity.
    pub ambient_intensity: f32,
    /// Fog start distance.
    pub fog_start: f32,
    /// Fog end distance.
    pub fog_end: f32,
    /// Fog color [r, g, b].
    pub fog_color: [f32; 3],
}

impl Default for LevelSettings {
    fn default() -> Self {
        Self {
            cell_size: 256.0,
            load_radius: 1024.0,
            unload_radius: 1280.0,
            max_concurrent_loads: 4,
            max_concurrent_unloads: 2,
            memory_budget: 512 * 1024 * 1024,
            layers: vec![
                WorldLayer::new("terrain", 2000.0, 0),
                WorldLayer::new("props", 1000.0, 1),
                WorldLayer::new("foliage", 500.0, 2),
                WorldLayer::new("audio", 300.0, 3),
                WorldLayer::new("ai", 800.0, 4),
            ],
            gravity: [0.0, -9.81, 0.0],
            ambient_color: [0.2, 0.25, 0.3],
            ambient_intensity: 0.5,
            fog_start: 500.0,
            fog_end: 2000.0,
            fog_color: [0.6, 0.65, 0.7],
        }
    }
}

// ---------------------------------------------------------------------------
// SpawnPoint
// ---------------------------------------------------------------------------

/// A named spawn point within a level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnPoint {
    /// Name of this spawn point.
    pub name: String,
    /// World-space position.
    pub position: Vec3,
    /// Orientation as euler angles [yaw, pitch, roll] in degrees.
    pub rotation: [f32; 3],
    /// Whether this is the default spawn point.
    pub is_default: bool,
    /// Optional tags for filtering (e.g., "team_a", "respawn").
    pub tags: Vec<String>,
}

impl SpawnPoint {
    /// Create a spawn point at a position.
    pub fn new(name: impl Into<String>, position: Vec3) -> Self {
        Self {
            name: name.into(),
            position,
            rotation: [0.0, 0.0, 0.0],
            is_default: false,
            tags: Vec::new(),
        }
    }

    /// Mark this as the default spawn point.
    pub fn as_default(mut self) -> Self {
        self.is_default = true;
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

// ---------------------------------------------------------------------------
// LevelState
// ---------------------------------------------------------------------------

/// Current state of a level in the level manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelState {
    /// Level metadata has not been loaded.
    Unloaded,
    /// Level metadata is being loaded.
    Loading,
    /// Level is loaded and active.
    Active,
    /// Level is being unloaded.
    Unloading,
    /// Level transition is in progress.
    Transitioning,
}

// ---------------------------------------------------------------------------
// Level
// ---------------------------------------------------------------------------

/// Metadata for a game level / map.
///
/// A level describes a set of world cells stored on disk, along with the
/// streaming settings and spawn points needed to play in that environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level {
    /// Unique name for this level.
    pub name: String,
    /// Display name shown in UI.
    pub display_name: String,
    /// Path to the directory containing cell data files.
    pub cell_data_path: PathBuf,
    /// Path to the level metadata file.
    pub metadata_path: PathBuf,
    /// Streaming and environment settings.
    pub settings: LevelSettings,
    /// Named spawn points.
    pub spawn_points: Vec<SpawnPoint>,
    /// Bounds of the level in world coordinates (min corner).
    pub bounds_min: Vec3,
    /// Bounds of the level in world coordinates (max corner).
    pub bounds_max: Vec3,
    /// Custom key-value properties.
    pub properties: HashMap<String, String>,
    /// Version of the level data format.
    pub version: u32,
}

impl Level {
    /// Create a new level with default settings.
    pub fn new(name: impl Into<String>, cell_data_path: impl Into<PathBuf>) -> Self {
        let name = name.into();
        let cell_data_path = cell_data_path.into();
        let metadata_path = cell_data_path.join("level.json");

        Self {
            display_name: name.clone(),
            name,
            cell_data_path,
            metadata_path,
            settings: LevelSettings::default(),
            spawn_points: Vec::new(),
            bounds_min: Vec3::splat(-10000.0),
            bounds_max: Vec3::splat(10000.0),
            properties: HashMap::new(),
            version: 1,
        }
    }

    /// Create a level with custom settings.
    pub fn with_settings(
        name: impl Into<String>,
        cell_data_path: impl Into<PathBuf>,
        settings: LevelSettings,
    ) -> Self {
        let mut level = Self::new(name, cell_data_path);
        level.settings = settings;
        level
    }

    /// Add a spawn point to this level.
    pub fn add_spawn_point(&mut self, spawn: SpawnPoint) {
        self.spawn_points.push(spawn);
    }

    /// Find the default spawn point, or the first one, or origin.
    pub fn default_spawn(&self) -> Vec3 {
        self.spawn_points
            .iter()
            .find(|s| s.is_default)
            .or_else(|| self.spawn_points.first())
            .map(|s| s.position)
            .unwrap_or(Vec3::ZERO)
    }

    /// Find a spawn point by name.
    pub fn spawn_point(&self, name: &str) -> Option<&SpawnPoint> {
        self.spawn_points.iter().find(|s| s.name == name)
    }

    /// Check whether a world position is within the level bounds.
    pub fn contains_position(&self, pos: Vec3) -> bool {
        pos.x >= self.bounds_min.x
            && pos.x <= self.bounds_max.x
            && pos.y >= self.bounds_min.y
            && pos.y <= self.bounds_max.y
            && pos.z >= self.bounds_min.z
            && pos.z <= self.bounds_max.z
    }

    /// Calculate the number of cells this level spans.
    pub fn cell_count_estimate(&self) -> usize {
        let cs = self.settings.cell_size;
        let x_cells = ((self.bounds_max.x - self.bounds_min.x) / cs).ceil() as usize;
        let z_cells = ((self.bounds_max.z - self.bounds_min.z) / cs).ceil() as usize;
        x_cells * z_cells
    }

    /// Load level metadata from a JSON file.
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read level file {}: {}", path.display(), e))?;

        serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse level file {}: {}", path.display(), e))
    }

    /// Save level metadata to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize level: {}", e))?;

        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write level file {}: {}", path.display(), e))
    }
}

// ---------------------------------------------------------------------------
// SubLevel
// ---------------------------------------------------------------------------

/// A sub-level that can be loaded additively into the world alongside the
/// base level. Sub-levels share the coordinate space with the base level
/// and add their own cells/entities on top.
#[derive(Debug, Clone)]
pub struct SubLevel {
    /// The level data.
    pub level: Level,
    /// Current state.
    pub state: SubLevelState,
    /// Offset applied to all sub-level positions (for repositioning).
    pub offset: Vec3,
    /// Whether this sub-level should stream independently.
    pub independent_streaming: bool,
    /// Visibility flag.
    pub visible: bool,
}

/// State of a sub-level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubLevelState {
    /// Not loaded.
    Unloaded,
    /// Currently loading.
    Loading,
    /// Loaded and active.
    Active,
    /// Being unloaded.
    Unloading,
}

impl SubLevel {
    /// Create a new sub-level wrapper.
    pub fn new(level: Level) -> Self {
        Self {
            level,
            state: SubLevelState::Unloaded,
            offset: Vec3::ZERO,
            independent_streaming: false,
            visible: true,
        }
    }

    /// Create with an offset.
    pub fn with_offset(mut self, offset: Vec3) -> Self {
        self.offset = offset;
        self
    }

    /// Set independent streaming mode.
    pub fn with_independent_streaming(mut self, independent: bool) -> Self {
        self.independent_streaming = independent;
        self
    }
}

// ---------------------------------------------------------------------------
// LevelTransition
// ---------------------------------------------------------------------------

/// Describes a transition between two levels.
#[derive(Debug, Clone)]
pub struct LevelTransition {
    /// Level being left.
    pub from_level: String,
    /// Level being entered.
    pub to_level: String,
    /// Spawn point in the destination level.
    pub spawn_point: Option<String>,
    /// Whether to use a loading screen.
    pub use_loading_screen: bool,
    /// Transition progress (0.0 = start, 1.0 = complete).
    pub progress: f32,
    /// Current phase of the transition.
    pub phase: TransitionPhase,
    /// Custom data to pass between levels.
    pub user_data: HashMap<String, String>,
}

/// Phase of a level transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionPhase {
    /// Fade out / save state of the current level.
    UnloadingCurrent,
    /// Loading the new level metadata and initial cells.
    LoadingDestination,
    /// Fade in / activate the new level.
    ActivatingDestination,
    /// Transition complete.
    Complete,
}

impl LevelTransition {
    /// Create a new transition.
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from_level: from.into(),
            to_level: to.into(),
            spawn_point: None,
            use_loading_screen: true,
            progress: 0.0,
            phase: TransitionPhase::UnloadingCurrent,
            user_data: HashMap::new(),
        }
    }

    /// Set the target spawn point.
    pub fn with_spawn_point(mut self, name: impl Into<String>) -> Self {
        self.spawn_point = Some(name.into());
        self
    }

    /// Set whether to show a loading screen.
    pub fn with_loading_screen(mut self, show: bool) -> Self {
        self.use_loading_screen = show;
        self
    }

    /// Check if the transition is complete.
    pub fn is_complete(&self) -> bool {
        self.phase == TransitionPhase::Complete
    }

    /// Advance the transition to the next phase.
    pub fn advance_phase(&mut self) {
        self.phase = match self.phase {
            TransitionPhase::UnloadingCurrent => TransitionPhase::LoadingDestination,
            TransitionPhase::LoadingDestination => TransitionPhase::ActivatingDestination,
            TransitionPhase::ActivatingDestination => TransitionPhase::Complete,
            TransitionPhase::Complete => TransitionPhase::Complete,
        };
    }
}

// ---------------------------------------------------------------------------
// LevelManager
// ---------------------------------------------------------------------------

/// Manages level loading, unloading, and transitions.
///
/// The level manager tracks the currently active level, any loaded sub-levels,
/// and any in-progress transitions. It coordinates with the
/// [`WorldPartition`] and [`StreamingManager`] to load the appropriate cells
/// when entering a level.
pub struct LevelManager {
    /// Currently active base level.
    current_level: Option<Level>,
    /// State of the current level.
    current_state: LevelState,
    /// Loaded sub-levels.
    sub_levels: Vec<SubLevel>,
    /// Known levels (preloaded metadata).
    known_levels: HashMap<String, Level>,
    /// Active transition, if any.
    active_transition: Option<LevelTransition>,
    /// History of loaded levels (for back-navigation).
    level_history: Vec<String>,
    /// Maximum history entries.
    max_history: usize,
    /// Callback invoked when a level finishes loading.
    on_level_loaded: Option<Box<dyn Fn(&Level) + Send + Sync>>,
    /// Callback invoked when a level starts unloading.
    on_level_unloading: Option<Box<dyn Fn(&Level) + Send + Sync>>,
}

impl LevelManager {
    /// Create a new level manager.
    pub fn new() -> Self {
        Self {
            current_level: None,
            current_state: LevelState::Unloaded,
            sub_levels: Vec::new(),
            known_levels: HashMap::new(),
            active_transition: None,
            level_history: Vec::new(),
            max_history: 10,
            on_level_loaded: None,
            on_level_unloading: None,
        }
    }

    /// Register a level so it can be loaded by name.
    pub fn register_level(&mut self, level: Level) {
        self.known_levels.insert(level.name.clone(), level);
    }

    /// Unregister a level by name.
    pub fn unregister_level(&mut self, name: &str) {
        self.known_levels.remove(name);
    }

    /// Get a registered level by name.
    pub fn get_level(&self, name: &str) -> Option<&Level> {
        self.known_levels.get(name)
    }

    /// List all registered level names.
    pub fn registered_levels(&self) -> Vec<&str> {
        self.known_levels.keys().map(|s| s.as_str()).collect()
    }

    /// Get the currently active level.
    pub fn current_level(&self) -> Option<&Level> {
        self.current_level.as_ref()
    }

    /// Get the current level state.
    pub fn current_state(&self) -> LevelState {
        self.current_state
    }

    /// Check if a transition is in progress.
    pub fn is_transitioning(&self) -> bool {
        self.active_transition.is_some()
    }

    /// Get the active transition.
    pub fn active_transition(&self) -> Option<&LevelTransition> {
        self.active_transition.as_ref()
    }

    /// Set a callback for level-loaded events.
    pub fn on_level_loaded<F: Fn(&Level) + Send + Sync + 'static>(&mut self, f: F) {
        self.on_level_loaded = Some(Box::new(f));
    }

    /// Set a callback for level-unloading events.
    pub fn on_level_unloading<F: Fn(&Level) + Send + Sync + 'static>(&mut self, f: F) {
        self.on_level_unloading = Some(Box::new(f));
    }

    // -- Level loading -----------------------------------------------------

    /// Load a level by name. The level must have been registered first.
    ///
    /// This sets up the world partition with the level's settings and begins
    /// streaming the initial cells around the default spawn point.
    pub fn load_level(
        &mut self,
        name: &str,
        partition: &mut WorldPartition,
        streaming: &mut StreamingManager,
    ) -> Result<Vec3, String> {
        let level = self
            .known_levels
            .get(name)
            .ok_or_else(|| format!("Level '{}' not registered", name))?
            .clone();

        self.load_level_internal(level, None, partition, streaming)
    }

    /// Load a level from a file path.
    pub fn load_level_from_file(
        &mut self,
        path: &Path,
        partition: &mut WorldPartition,
        streaming: &mut StreamingManager,
    ) -> Result<Vec3, String> {
        let level = Level::load_from_file(path)?;
        self.load_level_internal(level, None, partition, streaming)
    }

    /// Internal level loading logic.
    fn load_level_internal(
        &mut self,
        level: Level,
        spawn_point: Option<&str>,
        partition: &mut WorldPartition,
        streaming: &mut StreamingManager,
    ) -> Result<Vec3, String> {
        // Unload current level if any.
        if self.current_level.is_some() {
            self.unload_current(partition, streaming);
        }

        log::info!("Loading level '{}'", level.name);
        self.current_state = LevelState::Loading;

        // Configure partition with level settings.
        let settings = &level.settings;
        let new_partition = WorldPartition::new(
            settings.cell_size,
            settings.load_radius,
            settings.unload_radius,
        )
        .with_max_concurrent_loads(settings.max_concurrent_loads)
        .with_memory_budget(settings.memory_budget);
        *partition = new_partition;

        // Add layers.
        for layer in &settings.layers {
            partition.add_layer(layer.clone());
        }

        // Configure streaming manager.
        streaming.set_max_concurrent_loads(settings.max_concurrent_loads);
        streaming.set_max_concurrent_unloads(settings.max_concurrent_unloads);
        streaming.set_memory_budget(settings.memory_budget);

        // Determine spawn position.
        let spawn_pos = match spawn_point {
            Some(name) => level
                .spawn_point(name)
                .map(|s| s.position)
                .unwrap_or_else(|| level.default_spawn()),
            None => level.default_spawn(),
        };

        // Request initial cell loads around spawn point.
        let spawn_cell = partition.world_to_cell(spawn_pos);
        streaming.submit_load(spawn_cell, StreamRequestPriority::Critical, 0.0);

        // Also load adjacent cells.
        for neighbor in spawn_cell.neighbors(1) {
            if neighbor != spawn_cell {
                let dist = spawn_cell.distance(&neighbor, settings.cell_size);
                streaming.submit_load(neighbor, StreamRequestPriority::High, dist);
            }
        }

        // Update history.
        self.level_history.push(level.name.clone());
        if self.level_history.len() > self.max_history {
            self.level_history.remove(0);
        }

        // Invoke callback.
        if let Some(ref callback) = self.on_level_loaded {
            callback(&level);
        }

        self.current_level = Some(level);
        self.current_state = LevelState::Active;

        log::info!("Level active, spawn at {:?}", spawn_pos);
        Ok(spawn_pos)
    }

    /// Unload the current level.
    pub fn unload_current(
        &mut self,
        partition: &mut WorldPartition,
        streaming: &mut StreamingManager,
    ) {
        if let Some(ref level) = self.current_level {
            log::info!("Unloading level '{}'", level.name);

            if let Some(ref callback) = self.on_level_unloading {
                callback(level);
            }
        }

        self.current_state = LevelState::Unloading;

        // Unload all sub-levels.
        for sub in &mut self.sub_levels {
            sub.state = SubLevelState::Unloading;
        }
        self.sub_levels.clear();

        // Cancel all streaming and unload all cells.
        streaming.cancel_all();
        partition.unload_all();
        partition.reset();
        streaming.reset();

        self.current_level = None;
        self.current_state = LevelState::Unloaded;
    }

    // -- Level travel ------------------------------------------------------

    /// Begin a seamless transition to another level.
    ///
    /// If `use_loading_screen` is false, the engine will attempt to stream
    /// the new level while the old one is still visible. This requires both
    /// levels to fit in the memory budget simultaneously.
    pub fn travel_to(
        &mut self,
        destination: &str,
        spawn_point: Option<&str>,
        use_loading_screen: bool,
    ) -> Result<(), String> {
        if self.active_transition.is_some() {
            return Err("A transition is already in progress".to_owned());
        }

        if !self.known_levels.contains_key(destination) {
            return Err(format!("Destination level '{}' not registered", destination));
        }

        let from = self
            .current_level
            .as_ref()
            .map(|l| l.name.clone())
            .unwrap_or_else(|| "<none>".to_owned());

        let mut transition = LevelTransition::new(from, destination)
            .with_loading_screen(use_loading_screen);

        if let Some(sp) = spawn_point {
            transition = transition.with_spawn_point(sp);
        }

        log::info!(
            "Starting level travel: {} -> {} (loading_screen={})",
            transition.from_level,
            transition.to_level,
            use_loading_screen,
        );

        self.active_transition = Some(transition);
        self.current_state = LevelState::Transitioning;

        Ok(())
    }

    /// Update the active transition. Call each frame during a transition.
    ///
    /// Returns the spawn position when the transition completes.
    pub fn update_transition(
        &mut self,
        partition: &mut WorldPartition,
        streaming: &mut StreamingManager,
    ) -> Option<Vec3> {
        // Extract the current phase to avoid borrow conflicts.
        let phase = match self.active_transition.as_ref() {
            Some(t) => t.phase,
            None => return None,
        };

        match phase {
            TransitionPhase::UnloadingCurrent => {
                // Unload the current level.
                self.unload_current_internal(partition, streaming);
                if let Some(ref mut t) = self.active_transition {
                    t.progress = 0.33;
                    t.advance_phase();
                }
                None
            }
            TransitionPhase::LoadingDestination => {
                // Extract destination info before calling load_level_internal.
                let (dest_name, spawn_name) = match self.active_transition.as_ref() {
                    Some(t) => (t.to_level.clone(), t.spawn_point.clone()),
                    None => return None,
                };

                let level_opt = self.known_levels.get(&dest_name).cloned();
                match level_opt {
                    Some(level) => {
                        match self.load_level_internal(
                            level,
                            spawn_name.as_deref(),
                            partition,
                            streaming,
                        ) {
                            Ok(_pos) => {
                                if let Some(ref mut t) = self.active_transition {
                                    t.progress = 0.66;
                                    t.advance_phase();
                                }
                            }
                            Err(e) => {
                                log::error!("Failed to load destination level: {}", e);
                                self.active_transition = None;
                                self.current_state = LevelState::Unloaded;
                            }
                        }
                    }
                    None => {
                        log::error!("Destination level '{}' not found", dest_name);
                        self.active_transition = None;
                        self.current_state = LevelState::Unloaded;
                    }
                }
                None
            }
            TransitionPhase::ActivatingDestination => {
                if let Some(ref mut t) = self.active_transition {
                    t.progress = 1.0;
                    t.advance_phase();
                }

                let spawn = self
                    .current_level
                    .as_ref()
                    .map(|l| l.default_spawn())
                    .unwrap_or(Vec3::ZERO);

                Some(spawn)
            }
            TransitionPhase::Complete => {
                self.active_transition = None;
                self.current_state = LevelState::Active;
                None
            }
        }
    }

    /// Internal unload without clearing the level reference (used during transitions).
    fn unload_current_internal(
        &mut self,
        partition: &mut WorldPartition,
        streaming: &mut StreamingManager,
    ) {
        if let Some(ref level) = self.current_level {
            if let Some(ref callback) = self.on_level_unloading {
                callback(level);
            }
        }

        streaming.cancel_all();
        partition.unload_all();
        partition.reset();
        streaming.reset();

        self.sub_levels.clear();
    }

    // -- Sub-levels --------------------------------------------------------

    /// Load a sub-level additively.
    pub fn load_sub_level(
        &mut self,
        name: &str,
        streaming: &mut StreamingManager,
        partition: &WorldPartition,
    ) -> Result<(), String> {
        let level = self
            .known_levels
            .get(name)
            .ok_or_else(|| format!("Sub-level '{}' not registered", name))?
            .clone();

        log::info!("Loading sub-level '{}'", name);

        let mut sub = SubLevel::new(level);
        sub.state = SubLevelState::Loading;

        // Request initial cells of the sub-level.
        let spawn = sub.level.default_spawn() + sub.offset;
        let spawn_cell = partition.world_to_cell(spawn);
        streaming.submit_load(spawn_cell, StreamRequestPriority::Normal, 0.0);

        sub.state = SubLevelState::Active;
        self.sub_levels.push(sub);

        Ok(())
    }

    /// Unload a sub-level by name.
    pub fn unload_sub_level(&mut self, name: &str) -> bool {
        if let Some(idx) = self.sub_levels.iter().position(|s| s.level.name == name) {
            log::info!("Unloading sub-level '{}'", name);
            self.sub_levels.remove(idx);
            true
        } else {
            false
        }
    }

    /// Get a reference to a loaded sub-level.
    pub fn sub_level(&self, name: &str) -> Option<&SubLevel> {
        self.sub_levels.iter().find(|s| s.level.name == name)
    }

    /// Get a mutable reference to a loaded sub-level.
    pub fn sub_level_mut(&mut self, name: &str) -> Option<&mut SubLevel> {
        self.sub_levels.iter_mut().find(|s| s.level.name == name)
    }

    /// List all loaded sub-levels.
    pub fn sub_levels(&self) -> &[SubLevel] {
        &self.sub_levels
    }

    /// Set visibility of a sub-level.
    pub fn set_sub_level_visible(&mut self, name: &str, visible: bool) -> bool {
        if let Some(sub) = self.sub_level_mut(name) {
            sub.visible = visible;
            true
        } else {
            false
        }
    }

    // -- History -----------------------------------------------------------

    /// Get the level history (most recent last).
    pub fn level_history(&self) -> &[String] {
        &self.level_history
    }

    /// Get the previously loaded level name.
    pub fn previous_level(&self) -> Option<&str> {
        if self.level_history.len() >= 2 {
            Some(&self.level_history[self.level_history.len() - 2])
        } else {
            None
        }
    }

    /// Travel back to the previous level.
    pub fn travel_back(&mut self) -> Result<(), String> {
        let prev = self
            .previous_level()
            .map(|s| s.to_owned())
            .ok_or_else(|| "No previous level in history".to_owned())?;

        self.travel_to(&prev, None, true)
    }
}

impl Default for LevelManager {
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

    fn test_level(name: &str) -> Level {
        let mut level = Level::new(name, format!("/tmp/levels/{}", name));
        level.add_spawn_point(
            SpawnPoint::new("default", Vec3::new(100.0, 0.0, 100.0)).as_default(),
        );
        level.add_spawn_point(SpawnPoint::new("spawn_b", Vec3::new(500.0, 0.0, 200.0)));
        level
    }

    #[test]
    fn level_default_spawn() {
        let level = test_level("test_map");
        assert_eq!(level.default_spawn(), Vec3::new(100.0, 0.0, 100.0));
    }

    #[test]
    fn level_named_spawn() {
        let level = test_level("test_map");
        let sp = level.spawn_point("spawn_b").unwrap();
        assert_eq!(sp.position, Vec3::new(500.0, 0.0, 200.0));
    }

    #[test]
    fn level_manager_registration() {
        let mut mgr = LevelManager::new();
        mgr.register_level(test_level("map_a"));
        mgr.register_level(test_level("map_b"));

        assert!(mgr.get_level("map_a").is_some());
        assert!(mgr.get_level("map_b").is_some());
        assert!(mgr.get_level("map_c").is_none());
    }

    #[test]
    fn level_contains_position() {
        let mut level = test_level("bounded");
        level.bounds_min = Vec3::new(-100.0, -50.0, -100.0);
        level.bounds_max = Vec3::new(100.0, 50.0, 100.0);

        assert!(level.contains_position(Vec3::ZERO));
        assert!(!level.contains_position(Vec3::new(200.0, 0.0, 0.0)));
    }

    #[test]
    fn level_settings_default() {
        let settings = LevelSettings::default();
        assert_eq!(settings.cell_size, 256.0);
        assert_eq!(settings.layers.len(), 5);
        assert!(settings.unload_radius > settings.load_radius);
    }

    #[test]
    fn transition_phases() {
        let mut transition = LevelTransition::new("map_a", "map_b");
        assert_eq!(transition.phase, TransitionPhase::UnloadingCurrent);

        transition.advance_phase();
        assert_eq!(transition.phase, TransitionPhase::LoadingDestination);

        transition.advance_phase();
        assert_eq!(transition.phase, TransitionPhase::ActivatingDestination);

        transition.advance_phase();
        assert!(transition.is_complete());
    }
}
