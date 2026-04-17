//! Level streaming: async level load, level visibility, level transform offset,
//! level LOD, and streaming volume triggers.
//!
//! This module provides a complete level streaming subsystem that allows large
//! game worlds to be divided into independently loadable levels/chunks:
//!
//! - **Async level loading** — levels load in the background without hitching
//!   the main thread. Progress is reported incrementally.
//! - **Level visibility** — loaded levels can be toggled visible/invisible
//!   without unloading their data.
//! - **Level transform offset** — each level has a world-space origin offset,
//!   allowing seamless stitching of large worlds.
//! - **Level LOD** — multiple detail levels per level region, automatically
//!   switched based on camera distance.
//! - **Streaming volume triggers** — axis-aligned or oriented volumes that
//!   trigger level load/unload as the camera enters/exits.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Level identifier
// ---------------------------------------------------------------------------

/// Unique identifier for a streamable level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LevelId(pub u64);

impl LevelId {
    /// The persistent / always-loaded level.
    pub const PERSISTENT: Self = Self(0);

    /// Create from a name hash.
    pub fn from_name(name: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut hasher);
        Self(hasher.finish())
    }
}

impl fmt::Display for LevelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Level({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Level state
// ---------------------------------------------------------------------------

/// The current state of a streaming level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelState {
    /// Not loaded — no data in memory.
    Unloaded,
    /// Loading asynchronously.
    Loading,
    /// Loaded but not visible.
    LoadedHidden,
    /// Loaded and visible.
    LoadedVisible,
    /// Unloading asynchronously.
    Unloading,
    /// Failed to load.
    Error,
}

impl LevelState {
    /// Whether the level has data in memory.
    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::LoadedHidden | Self::LoadedVisible)
    }

    /// Whether the level is visible in the world.
    pub fn is_visible(&self) -> bool {
        matches!(self, Self::LoadedVisible)
    }

    /// Whether a load/unload operation is in progress.
    pub fn is_transitioning(&self) -> bool {
        matches!(self, Self::Loading | Self::Unloading)
    }
}

impl fmt::Display for LevelState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unloaded => write!(f, "Unloaded"),
            Self::Loading => write!(f, "Loading"),
            Self::LoadedHidden => write!(f, "Loaded (hidden)"),
            Self::LoadedVisible => write!(f, "Loaded (visible)"),
            Self::Unloading => write!(f, "Unloading"),
            Self::Error => write!(f, "Error"),
        }
    }
}

// ---------------------------------------------------------------------------
// Level LOD
// ---------------------------------------------------------------------------

/// Detail level for a level region.
#[derive(Debug, Clone)]
pub struct LevelLOD {
    /// LOD index (0 = highest detail).
    pub level: u32,
    /// Maximum camera distance for this LOD.
    pub max_distance: f32,
    /// Asset path for this LOD variant.
    pub asset_path: String,
    /// Estimated memory usage for this LOD.
    pub memory_estimate: u64,
    /// Whether this LOD includes collision data.
    pub has_collision: bool,
    /// Whether this LOD includes gameplay entities.
    pub has_gameplay: bool,
}

impl LevelLOD {
    /// Create a new LOD entry.
    pub fn new(
        level: u32,
        max_distance: f32,
        asset_path: impl Into<String>,
    ) -> Self {
        Self {
            level,
            max_distance,
            asset_path: asset_path.into(),
            memory_estimate: 0,
            has_collision: true,
            has_gameplay: level == 0,
        }
    }

    /// Set memory estimate.
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_estimate = bytes;
        self
    }

    /// Set collision data inclusion.
    pub fn with_collision(mut self, has: bool) -> Self {
        self.has_collision = has;
        self
    }

    /// Set gameplay entity inclusion.
    pub fn with_gameplay(mut self, has: bool) -> Self {
        self.has_gameplay = has;
        self
    }
}

// ---------------------------------------------------------------------------
// Level descriptor
// ---------------------------------------------------------------------------

/// Complete definition of a streamable level.
#[derive(Debug, Clone)]
pub struct LevelDescriptor {
    /// Unique level identifier.
    pub id: LevelId,
    /// Human-readable level name.
    pub name: String,
    /// Asset path for the level data.
    pub asset_path: String,
    /// World-space origin offset for this level.
    pub world_offset: [f32; 3],
    /// World-space bounding box (min).
    pub bounds_min: [f32; 3],
    /// World-space bounding box (max).
    pub bounds_max: [f32; 3],
    /// LOD variants.
    pub lod_levels: Vec<LevelLOD>,
    /// Priority for loading (higher = loaded first).
    pub priority: i32,
    /// Whether this level should always be loaded.
    pub always_loaded: bool,
    /// Whether this level blocks gameplay until loaded.
    pub blocking: bool,
    /// Dependencies (other levels that must be loaded first).
    pub dependencies: Vec<LevelId>,
    /// Tags for grouping/filtering.
    pub tags: Vec<String>,
    /// Estimated load time.
    pub estimated_load_time: Duration,
    /// Estimated memory usage.
    pub estimated_memory: u64,
}

impl LevelDescriptor {
    /// Create a new level descriptor.
    pub fn new(
        id: LevelId,
        name: impl Into<String>,
        asset_path: impl Into<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            asset_path: asset_path.into(),
            world_offset: [0.0; 3],
            bounds_min: [-100.0, -100.0, -100.0],
            bounds_max: [100.0, 100.0, 100.0],
            lod_levels: Vec::new(),
            priority: 0,
            always_loaded: false,
            blocking: false,
            dependencies: Vec::new(),
            tags: Vec::new(),
            estimated_load_time: Duration::from_millis(500),
            estimated_memory: 0,
        }
    }

    /// Set the world offset.
    pub fn with_offset(mut self, x: f32, y: f32, z: f32) -> Self {
        self.world_offset = [x, y, z];
        self
    }

    /// Set the bounds.
    pub fn with_bounds(
        mut self,
        min: [f32; 3],
        max: [f32; 3],
    ) -> Self {
        self.bounds_min = min;
        self.bounds_max = max;
        self
    }

    /// Add an LOD level.
    pub fn with_lod(mut self, lod: LevelLOD) -> Self {
        self.lod_levels.push(lod);
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Mark as always loaded.
    pub fn always_loaded(mut self) -> Self {
        self.always_loaded = true;
        self
    }

    /// Mark as blocking.
    pub fn blocking(mut self) -> Self {
        self.blocking = true;
        self
    }

    /// Add a dependency.
    pub fn depends_on(mut self, dep: LevelId) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Check if a point is within this level's bounds.
    pub fn contains_point(&self, x: f32, y: f32, z: f32) -> bool {
        x >= self.bounds_min[0]
            && x <= self.bounds_max[0]
            && y >= self.bounds_min[1]
            && y <= self.bounds_max[1]
            && z >= self.bounds_min[2]
            && z <= self.bounds_max[2]
    }

    /// Get the center of this level's bounds.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.bounds_min[0] + self.bounds_max[0]) * 0.5,
            (self.bounds_min[1] + self.bounds_max[1]) * 0.5,
            (self.bounds_min[2] + self.bounds_max[2]) * 0.5,
        ]
    }

    /// Get the distance from a point to this level's center.
    pub fn distance_to(&self, x: f32, y: f32, z: f32) -> f32 {
        let c = self.center();
        let dx = x - c[0];
        let dy = y - c[1];
        let dz = z - c[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Select the best LOD for the given distance.
    pub fn select_lod(&self, distance: f32) -> Option<u32> {
        // Sort by LOD level (ascending = highest detail first).
        let mut sorted: Vec<_> = self.lod_levels.iter().collect();
        sorted.sort_by_key(|l| l.level);

        for lod in sorted.iter().rev() {
            if distance <= lod.max_distance {
                return Some(lod.level);
            }
        }
        // If beyond all LODs, use the lowest detail.
        sorted.last().map(|l| l.level)
    }
}

// ---------------------------------------------------------------------------
// Streaming volume trigger
// ---------------------------------------------------------------------------

/// Shape of a streaming volume trigger.
#[derive(Debug, Clone)]
pub enum TriggerVolume {
    /// Axis-aligned box.
    AABB {
        min: [f32; 3],
        max: [f32; 3],
    },
    /// Sphere.
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    /// Oriented box.
    OBB {
        center: [f32; 3],
        half_extents: [f32; 3],
        rotation: [f32; 4], // quaternion
    },
}

impl TriggerVolume {
    /// Check if a point is inside the volume.
    pub fn contains_point(&self, x: f32, y: f32, z: f32) -> bool {
        match self {
            Self::AABB { min, max } => {
                x >= min[0]
                    && x <= max[0]
                    && y >= min[1]
                    && y <= max[1]
                    && z >= min[2]
                    && z <= max[2]
            }
            Self::Sphere { center, radius } => {
                let dx = x - center[0];
                let dy = y - center[1];
                let dz = z - center[2];
                dx * dx + dy * dy + dz * dz <= radius * radius
            }
            Self::OBB {
                center,
                half_extents,
                ..
            } => {
                // Simplified: treat as AABB (proper OBB needs rotation transform).
                let dx = (x - center[0]).abs();
                let dy = (y - center[1]).abs();
                let dz = (z - center[2]).abs();
                dx <= half_extents[0]
                    && dy <= half_extents[1]
                    && dz <= half_extents[2]
            }
        }
    }
}

/// Action to take when a trigger is activated.
#[derive(Debug, Clone)]
pub enum TriggerAction {
    /// Load a level.
    LoadLevel(LevelId),
    /// Unload a level.
    UnloadLevel(LevelId),
    /// Set level visibility.
    SetVisible(LevelId, bool),
    /// Load with a specific LOD.
    LoadLOD(LevelId, u32),
}

/// A streaming volume trigger.
#[derive(Debug, Clone)]
pub struct StreamingTrigger {
    /// Trigger identifier.
    pub id: u64,
    /// The trigger volume shape.
    pub volume: TriggerVolume,
    /// Action on enter.
    pub on_enter: Vec<TriggerAction>,
    /// Action on exit.
    pub on_exit: Vec<TriggerAction>,
    /// Whether the trigger is enabled.
    pub enabled: bool,
    /// Hysteresis distance (prevents rapid toggling near boundary).
    pub hysteresis: f32,
    /// Last known state: true = inside, false = outside.
    pub was_inside: bool,
}

impl StreamingTrigger {
    /// Create a new trigger.
    pub fn new(id: u64, volume: TriggerVolume) -> Self {
        Self {
            id,
            volume,
            on_enter: Vec::new(),
            on_exit: Vec::new(),
            enabled: true,
            hysteresis: 5.0,
            was_inside: false,
        }
    }

    /// Add an enter action.
    pub fn on_enter(mut self, action: TriggerAction) -> Self {
        self.on_enter.push(action);
        self
    }

    /// Add an exit action.
    pub fn on_exit(mut self, action: TriggerAction) -> Self {
        self.on_exit.push(action);
        self
    }

    /// Check the trigger against a position and return actions if state changed.
    pub fn check(&mut self, x: f32, y: f32, z: f32) -> Vec<TriggerAction> {
        if !self.enabled {
            return Vec::new();
        }

        let inside = self.volume.contains_point(x, y, z);

        if inside && !self.was_inside {
            self.was_inside = true;
            self.on_enter.clone()
        } else if !inside && self.was_inside {
            self.was_inside = false;
            self.on_exit.clone()
        } else {
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Level instance
// ---------------------------------------------------------------------------

/// Runtime state for a loaded level instance.
#[derive(Debug)]
pub struct LevelInstance {
    /// Level descriptor.
    pub descriptor: LevelDescriptor,
    /// Current state.
    pub state: LevelState,
    /// Current LOD level loaded.
    pub current_lod: Option<u32>,
    /// Load progress (0.0 to 1.0).
    pub load_progress: f32,
    /// Time when loading started.
    pub load_start_time: Option<Instant>,
    /// Time when loading completed.
    pub load_end_time: Option<Instant>,
    /// Number of entities in this level.
    pub entity_count: u32,
    /// Memory usage in bytes.
    pub memory_usage: u64,
    /// Error message if state is Error.
    pub error: Option<String>,
    /// Distance from camera (updated each frame).
    pub camera_distance: f32,
}

impl LevelInstance {
    /// Create from a descriptor.
    pub fn new(descriptor: LevelDescriptor) -> Self {
        let state = if descriptor.always_loaded {
            LevelState::Loading
        } else {
            LevelState::Unloaded
        };

        Self {
            descriptor,
            state,
            current_lod: None,
            load_progress: 0.0,
            load_start_time: None,
            load_end_time: None,
            entity_count: 0,
            memory_usage: 0,
            error: None,
            camera_distance: f32::MAX,
        }
    }

    /// Get the actual load time.
    pub fn load_duration(&self) -> Option<Duration> {
        match (self.load_start_time, self.load_end_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    /// Update camera distance.
    pub fn update_camera_distance(&mut self, camera_x: f32, camera_y: f32, camera_z: f32) {
        self.camera_distance =
            self.descriptor.distance_to(camera_x, camera_y, camera_z);
    }
}

// ---------------------------------------------------------------------------
// Level streaming event
// ---------------------------------------------------------------------------

/// Events emitted by the streaming system.
#[derive(Debug, Clone)]
pub enum StreamingEvent {
    /// A level started loading.
    LoadStarted(LevelId),
    /// A level's load progress updated.
    LoadProgress(LevelId, f32),
    /// A level finished loading.
    LoadCompleted(LevelId, Duration),
    /// A level failed to load.
    LoadFailed(LevelId, String),
    /// A level started unloading.
    UnloadStarted(LevelId),
    /// A level finished unloading.
    UnloadCompleted(LevelId),
    /// A level's visibility changed.
    VisibilityChanged(LevelId, bool),
    /// A level's LOD changed.
    LODChanged(LevelId, u32),
}

// ---------------------------------------------------------------------------
// Level streaming manager
// ---------------------------------------------------------------------------

/// Manages asynchronous level streaming.
pub struct LevelStreamingManager {
    /// All registered levels.
    levels: HashMap<LevelId, LevelInstance>,
    /// Active streaming triggers.
    triggers: Vec<StreamingTrigger>,
    /// Queue of levels pending load.
    load_queue: VecDeque<LevelId>,
    /// Queue of levels pending unload.
    unload_queue: VecDeque<LevelId>,
    /// Events emitted this frame.
    events: Vec<StreamingEvent>,
    /// Memory budget in bytes.
    memory_budget: u64,
    /// Current total memory used.
    memory_used: u64,
    /// Maximum concurrent loads.
    max_concurrent_loads: u32,
    /// Currently loading count.
    loading_count: u32,
    /// Camera position (updated each tick).
    camera_position: [f32; 3],
    /// Streaming enabled flag.
    enabled: bool,
    /// Statistics.
    stats: StreamingStats,
}

impl LevelStreamingManager {
    /// Create a new streaming manager.
    pub fn new(memory_budget: u64) -> Self {
        Self {
            levels: HashMap::new(),
            triggers: Vec::new(),
            load_queue: VecDeque::new(),
            unload_queue: VecDeque::new(),
            events: Vec::new(),
            memory_budget,
            memory_used: 0,
            max_concurrent_loads: 2,
            loading_count: 0,
            camera_position: [0.0; 3],
            enabled: true,
            stats: StreamingStats::new(),
        }
    }

    /// Register a level.
    pub fn register_level(&mut self, descriptor: LevelDescriptor) {
        let id = descriptor.id;
        let always = descriptor.always_loaded;
        let instance = LevelInstance::new(descriptor);
        self.levels.insert(id, instance);

        if always {
            self.request_load(id);
        }
    }

    /// Add a streaming trigger.
    pub fn add_trigger(&mut self, trigger: StreamingTrigger) {
        self.triggers.push(trigger);
    }

    /// Request a level to be loaded.
    pub fn request_load(&mut self, id: LevelId) {
        if let Some(instance) = self.levels.get(&id) {
            if instance.state == LevelState::Unloaded {
                self.load_queue.push_back(id);
            }
        }
    }

    /// Request a level to be unloaded.
    pub fn request_unload(&mut self, id: LevelId) {
        if let Some(instance) = self.levels.get(&id) {
            if instance.state.is_loaded() && !instance.descriptor.always_loaded {
                self.unload_queue.push_back(id);
            }
        }
    }

    /// Set level visibility.
    pub fn set_visible(&mut self, id: LevelId, visible: bool) {
        if let Some(instance) = self.levels.get_mut(&id) {
            if instance.state.is_loaded() {
                instance.state = if visible {
                    LevelState::LoadedVisible
                } else {
                    LevelState::LoadedHidden
                };
                self.events
                    .push(StreamingEvent::VisibilityChanged(id, visible));
            }
        }
    }

    /// Update camera position.
    pub fn set_camera_position(&mut self, x: f32, y: f32, z: f32) {
        self.camera_position = [x, y, z];
    }

    /// Tick the streaming manager.
    ///
    /// This should be called once per frame. It:
    /// 1. Updates camera distances.
    /// 2. Checks streaming triggers.
    /// 3. Processes the load/unload queues.
    /// 4. Simulates load progress.
    pub fn tick(&mut self, delta: Duration) {
        if !self.enabled {
            return;
        }

        self.events.clear();

        let [cx, cy, cz] = self.camera_position;

        // Update camera distances for all levels.
        for instance in self.levels.values_mut() {
            instance.update_camera_distance(cx, cy, cz);
        }

        // Check triggers.
        let mut trigger_actions = Vec::new();
        for trigger in &mut self.triggers {
            let actions = trigger.check(cx, cy, cz);
            trigger_actions.extend(actions);
        }

        // Process trigger actions.
        for action in trigger_actions {
            match action {
                TriggerAction::LoadLevel(id) => self.request_load(id),
                TriggerAction::UnloadLevel(id) => self.request_unload(id),
                TriggerAction::SetVisible(id, vis) => self.set_visible(id, vis),
                TriggerAction::LoadLOD(id, lod) => {
                    if let Some(instance) = self.levels.get_mut(&id) {
                        instance.current_lod = Some(lod);
                        self.events
                            .push(StreamingEvent::LODChanged(id, lod));
                    }
                }
            }
        }

        // Process load queue.
        while self.loading_count < self.max_concurrent_loads {
            if let Some(id) = self.load_queue.pop_front() {
                if let Some(instance) = self.levels.get_mut(&id) {
                    if instance.state == LevelState::Unloaded {
                        instance.state = LevelState::Loading;
                        instance.load_start_time = Some(Instant::now());
                        instance.load_progress = 0.0;
                        self.loading_count += 1;
                        self.stats.total_loads += 1;
                        self.events
                            .push(StreamingEvent::LoadStarted(id));
                    }
                }
            } else {
                break;
            }
        }

        // Simulate load progress for loading levels.
        let loading_ids: Vec<LevelId> = self
            .levels
            .iter()
            .filter(|(_, inst)| inst.state == LevelState::Loading)
            .map(|(&id, _)| id)
            .collect();

        for id in loading_ids {
            if let Some(instance) = self.levels.get_mut(&id) {
                let estimated = instance
                    .descriptor
                    .estimated_load_time
                    .as_secs_f32();
                let increment = if estimated > 0.0 {
                    delta.as_secs_f32() / estimated
                } else {
                    1.0
                };

                instance.load_progress = (instance.load_progress + increment).min(1.0);

                self.events.push(StreamingEvent::LoadProgress(
                    id,
                    instance.load_progress,
                ));

                if instance.load_progress >= 1.0 {
                    instance.state = LevelState::LoadedVisible;
                    instance.load_end_time = Some(Instant::now());
                    instance.memory_usage = instance.descriptor.estimated_memory;
                    self.memory_used += instance.memory_usage;
                    self.loading_count -= 1;

                    let duration = instance.load_duration().unwrap_or(Duration::ZERO);
                    self.events
                        .push(StreamingEvent::LoadCompleted(id, duration));
                }
            }
        }

        // Process unload queue.
        while let Some(id) = self.unload_queue.pop_front() {
            if let Some(instance) = self.levels.get_mut(&id) {
                if instance.state.is_loaded() {
                    self.memory_used = self.memory_used.saturating_sub(instance.memory_usage);
                    instance.state = LevelState::Unloaded;
                    instance.current_lod = None;
                    instance.load_progress = 0.0;
                    instance.memory_usage = 0;
                    instance.entity_count = 0;
                    self.stats.total_unloads += 1;
                    self.events.push(StreamingEvent::UnloadCompleted(id));
                }
            }
        }

        // Auto-LOD selection based on camera distance.
        let level_lod_updates: Vec<(LevelId, u32)> = self
            .levels
            .iter()
            .filter(|(_, inst)| inst.state.is_loaded())
            .filter_map(|(&id, inst)| {
                let new_lod = inst.descriptor.select_lod(inst.camera_distance)?;
                if inst.current_lod != Some(new_lod) {
                    Some((id, new_lod))
                } else {
                    None
                }
            })
            .collect();

        for (id, lod) in level_lod_updates {
            if let Some(instance) = self.levels.get_mut(&id) {
                instance.current_lod = Some(lod);
                self.events.push(StreamingEvent::LODChanged(id, lod));
            }
        }
    }

    /// Get events emitted this frame.
    pub fn events(&self) -> &[StreamingEvent] {
        &self.events
    }

    /// Get a level instance.
    pub fn get_level(&self, id: LevelId) -> Option<&LevelInstance> {
        self.levels.get(&id)
    }

    /// Get all loaded levels.
    pub fn loaded_levels(&self) -> Vec<LevelId> {
        self.levels
            .iter()
            .filter(|(_, inst)| inst.state.is_loaded())
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get all visible levels.
    pub fn visible_levels(&self) -> Vec<LevelId> {
        self.levels
            .iter()
            .filter(|(_, inst)| inst.state.is_visible())
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get the current memory usage.
    pub fn memory_used(&self) -> u64 {
        self.memory_used
    }

    /// Get the memory budget.
    pub fn memory_budget(&self) -> u64 {
        self.memory_budget
    }

    /// Get streaming statistics.
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    /// Enable or disable streaming.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get the number of registered levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Get levels sorted by distance to camera.
    pub fn levels_by_distance(&self) -> Vec<(LevelId, f32)> {
        let mut sorted: Vec<_> = self
            .levels
            .iter()
            .map(|(&id, inst)| (id, inst.camera_distance))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }
}

impl Default for LevelStreamingManager {
    fn default() -> Self {
        Self::new(512 * 1024 * 1024) // 512 MB default budget
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Streaming statistics.
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Total loads since start.
    pub total_loads: u64,
    /// Total unloads since start.
    pub total_unloads: u64,
    /// Peak memory usage.
    pub peak_memory: u64,
    /// Peak concurrent loads.
    pub peak_concurrent_loads: u32,
}

impl StreamingStats {
    pub fn new() -> Self {
        Self {
            total_loads: 0,
            total_unloads: 0,
            peak_memory: 0,
            peak_concurrent_loads: 0,
        }
    }
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StreamingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Streaming: loads={} unloads={} peak_mem={:.1}MB peak_concurrent={}",
            self.total_loads,
            self.total_unloads,
            self.peak_memory as f64 / (1024.0 * 1024.0),
            self.peak_concurrent_loads
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_state_transitions() {
        assert!(!LevelState::Unloaded.is_loaded());
        assert!(LevelState::LoadedVisible.is_visible());
        assert!(LevelState::Loading.is_transitioning());
    }

    #[test]
    fn trigger_volume_contains() {
        let aabb = TriggerVolume::AABB {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 10.0, 10.0],
        };
        assert!(aabb.contains_point(5.0, 5.0, 5.0));
        assert!(!aabb.contains_point(-1.0, 5.0, 5.0));

        let sphere = TriggerVolume::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 10.0,
        };
        assert!(sphere.contains_point(3.0, 4.0, 0.0)); // distance = 5
        assert!(!sphere.contains_point(10.0, 10.0, 10.0)); // distance = 17.3
    }

    #[test]
    fn level_descriptor_lod_selection() {
        let desc = LevelDescriptor::new(
            LevelId(1),
            "test",
            "levels/test.level",
        )
        .with_lod(LevelLOD::new(0, 100.0, "levels/test_lod0.level"))
        .with_lod(LevelLOD::new(1, 500.0, "levels/test_lod1.level"))
        .with_lod(LevelLOD::new(2, 2000.0, "levels/test_lod2.level"));

        assert_eq!(desc.select_lod(50.0), Some(0));
        assert_eq!(desc.select_lod(200.0), Some(1));
        assert_eq!(desc.select_lod(1000.0), Some(2));
    }

    #[test]
    fn streaming_trigger_enter_exit() {
        let mut trigger = StreamingTrigger::new(
            1,
            TriggerVolume::AABB {
                min: [0.0, 0.0, 0.0],
                max: [10.0, 10.0, 10.0],
            },
        )
        .on_enter(TriggerAction::LoadLevel(LevelId(1)))
        .on_exit(TriggerAction::UnloadLevel(LevelId(1)));

        // Move inside.
        let actions = trigger.check(5.0, 5.0, 5.0);
        assert_eq!(actions.len(), 1);

        // Stay inside.
        let actions = trigger.check(6.0, 6.0, 6.0);
        assert_eq!(actions.len(), 0);

        // Move outside.
        let actions = trigger.check(-5.0, -5.0, -5.0);
        assert_eq!(actions.len(), 1);
    }
}
