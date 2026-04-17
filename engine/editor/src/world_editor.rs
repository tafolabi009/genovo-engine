//! World/level editor: multi-level editing, level streaming setup, world bounds
//! visualization, and level connection editor.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldLevelId(pub u64);

impl WorldLevelId {
    pub const PERSISTENT: Self = Self(0);
    pub fn from_name(name: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut h);
        Self(h.finish())
    }
}

impl fmt::Display for WorldLevelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Level({})", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelEditState { Unloaded, Loaded, Active, Modified, Saving }

#[derive(Debug, Clone)]
pub struct LevelBoundsVis {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub color: [f32; 4],
    pub visible: bool,
    pub wireframe: bool,
}

impl LevelBoundsVis {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max, color: [0.2, 0.8, 0.2, 0.3], visible: true, wireframe: true }
    }
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
    pub fn size(&self) -> [f32; 3] {
        [self.max[0] - self.min[0], self.max[1] - self.min[1], self.max[2] - self.min[2]]
    }
    pub fn contains(&self, p: [f32; 3]) -> bool {
        p[0] >= self.min[0] && p[0] <= self.max[0]
            && p[1] >= self.min[1] && p[1] <= self.max[1]
            && p[2] >= self.min[2] && p[2] <= self.max[2]
    }
    pub fn volume(&self) -> f32 {
        let s = self.size();
        s[0] * s[1] * s[2]
    }
}

#[derive(Debug, Clone)]
pub struct EditorLevel {
    pub id: WorldLevelId,
    pub name: String,
    pub state: LevelEditState,
    pub bounds: LevelBoundsVis,
    pub world_offset: [f32; 3],
    pub asset_path: String,
    pub entity_count: u32,
    pub modified: bool,
    pub locked: bool,
    pub visible: bool,
    pub streaming_distance: f32,
    pub always_loaded: bool,
    pub color: [f32; 4],
    pub tags: Vec<String>,
    pub layer_index: u32,
    pub description: String,
}

impl EditorLevel {
    pub fn new(id: WorldLevelId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            state: LevelEditState::Unloaded,
            bounds: LevelBoundsVis::new([-100.0; 3], [100.0; 3]),
            world_offset: [0.0; 3],
            asset_path: String::new(),
            entity_count: 0,
            modified: false,
            locked: false,
            visible: true,
            streaming_distance: 500.0,
            always_loaded: false,
            color: [0.3, 0.6, 0.9, 1.0],
            tags: Vec::new(),
            layer_index: 0,
            description: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionType { Portal, Teleport, Seamless, LoadTrigger }

#[derive(Debug, Clone)]
pub struct LevelConnection {
    pub id: u64,
    pub source_level: WorldLevelId,
    pub target_level: WorldLevelId,
    pub connection_type: ConnectionType,
    pub source_position: [f32; 3],
    pub target_position: [f32; 3],
    pub bidirectional: bool,
    pub auto_load: bool,
    pub label: String,
    pub color: [f32; 4],
    pub trigger_radius: f32,
}

impl LevelConnection {
    pub fn new(id: u64, source: WorldLevelId, target: WorldLevelId, conn_type: ConnectionType) -> Self {
        Self {
            id, source_level: source, target_level: target, connection_type: conn_type,
            source_position: [0.0; 3], target_position: [0.0; 3],
            bidirectional: true, auto_load: true, label: String::new(),
            color: [1.0, 0.8, 0.2, 1.0], trigger_radius: 50.0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum WorldEditorEvent {
    LevelCreated(WorldLevelId),
    LevelRemoved(WorldLevelId),
    LevelLoaded(WorldLevelId),
    LevelUnloaded(WorldLevelId),
    LevelActivated(WorldLevelId),
    LevelModified(WorldLevelId),
    LevelSaved(WorldLevelId),
    LevelBoundsChanged(WorldLevelId),
    ConnectionCreated(u64),
    ConnectionRemoved(u64),
    WorldBoundsUpdated,
    SelectionChanged(Vec<WorldLevelId>),
}

pub struct WorldEditorState {
    pub levels: HashMap<WorldLevelId, EditorLevel>,
    pub connections: Vec<LevelConnection>,
    pub active_level: Option<WorldLevelId>,
    pub world_bounds: LevelBoundsVis,
    pub selected_levels: Vec<WorldLevelId>,
    pub events: Vec<WorldEditorEvent>,
    pub next_connection_id: u64,
    pub show_bounds: bool,
    pub show_connections: bool,
    pub show_grid: bool,
    pub show_streaming_radius: bool,
    pub grid_size: f32,
    pub snap_to_grid: bool,
    pub auto_save: bool,
    pub auto_save_interval_secs: f32,
    pub last_auto_save: f32,
}

impl WorldEditorState {
    pub fn new() -> Self {
        Self {
            levels: HashMap::new(),
            connections: Vec::new(),
            active_level: None,
            world_bounds: LevelBoundsVis::new([-1000.0; 3], [1000.0; 3]),
            selected_levels: Vec::new(),
            events: Vec::new(),
            next_connection_id: 1,
            show_bounds: true,
            show_connections: true,
            show_grid: true,
            show_streaming_radius: false,
            grid_size: 100.0,
            snap_to_grid: true,
            auto_save: false,
            auto_save_interval_secs: 300.0,
            last_auto_save: 0.0,
        }
    }

    pub fn create_level(&mut self, name: impl Into<String>) -> WorldLevelId {
        let name = name.into();
        let id = WorldLevelId::from_name(&name);
        let level = EditorLevel::new(id, name);
        self.levels.insert(id, level);
        self.events.push(WorldEditorEvent::LevelCreated(id));
        self.update_world_bounds();
        id
    }

    pub fn remove_level(&mut self, id: WorldLevelId) -> bool {
        if self.levels.remove(&id).is_some() {
            self.connections.retain(|c| c.source_level != id && c.target_level != id);
            if self.active_level == Some(id) { self.active_level = None; }
            self.selected_levels.retain(|&l| l != id);
            self.events.push(WorldEditorEvent::LevelRemoved(id));
            self.update_world_bounds();
            true
        } else {
            false
        }
    }

    pub fn set_active(&mut self, id: WorldLevelId) {
        if self.levels.contains_key(&id) {
            self.active_level = Some(id);
            self.events.push(WorldEditorEvent::LevelActivated(id));
        }
    }

    pub fn load_level(&mut self, id: WorldLevelId) {
        if let Some(level) = self.levels.get_mut(&id) {
            level.state = LevelEditState::Loaded;
            self.events.push(WorldEditorEvent::LevelLoaded(id));
        }
    }

    pub fn unload_level(&mut self, id: WorldLevelId) {
        if let Some(level) = self.levels.get_mut(&id) {
            level.state = LevelEditState::Unloaded;
            self.events.push(WorldEditorEvent::LevelUnloaded(id));
        }
    }

    pub fn save_level(&mut self, id: WorldLevelId) {
        if let Some(level) = self.levels.get_mut(&id) {
            level.state = LevelEditState::Loaded;
            level.modified = false;
            self.events.push(WorldEditorEvent::LevelSaved(id));
        }
    }

    pub fn add_connection(
        &mut self,
        source: WorldLevelId,
        target: WorldLevelId,
        conn_type: ConnectionType,
    ) -> u64 {
        let id = self.next_connection_id;
        self.next_connection_id += 1;
        self.connections.push(LevelConnection::new(id, source, target, conn_type));
        self.events.push(WorldEditorEvent::ConnectionCreated(id));
        id
    }

    pub fn remove_connection(&mut self, id: u64) -> bool {
        let len = self.connections.len();
        self.connections.retain(|c| c.id != id);
        if self.connections.len() < len {
            self.events.push(WorldEditorEvent::ConnectionRemoved(id));
            true
        } else {
            false
        }
    }

    pub fn set_level_bounds(&mut self, id: WorldLevelId, min: [f32; 3], max: [f32; 3]) {
        if let Some(level) = self.levels.get_mut(&id) {
            level.bounds = LevelBoundsVis::new(min, max);
            level.modified = true;
            self.events.push(WorldEditorEvent::LevelBoundsChanged(id));
            self.update_world_bounds();
        }
    }

    pub fn set_level_offset(&mut self, id: WorldLevelId, offset: [f32; 3]) {
        if let Some(level) = self.levels.get_mut(&id) {
            level.world_offset = offset;
            level.modified = true;
            self.update_world_bounds();
        }
    }

    fn update_world_bounds(&mut self) {
        if self.levels.is_empty() { return; }
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for level in self.levels.values() {
            for i in 0..3 {
                min[i] = min[i].min(level.bounds.min[i] + level.world_offset[i]);
                max[i] = max[i].max(level.bounds.max[i] + level.world_offset[i]);
            }
        }
        self.world_bounds = LevelBoundsVis::new(min, max);
        self.events.push(WorldEditorEvent::WorldBoundsUpdated);
    }

    pub fn loaded_levels(&self) -> Vec<WorldLevelId> {
        self.levels.iter()
            .filter(|(_, l)| matches!(l.state, LevelEditState::Loaded | LevelEditState::Active | LevelEditState::Modified))
            .map(|(&id, _)| id)
            .collect()
    }

    pub fn modified_levels(&self) -> Vec<WorldLevelId> {
        self.levels.iter().filter(|(_, l)| l.modified).map(|(&id, _)| id).collect()
    }

    pub fn save_all_modified(&mut self) {
        let modified: Vec<WorldLevelId> = self.modified_levels();
        for id in modified { self.save_level(id); }
    }

    pub fn select_level(&mut self, id: WorldLevelId) {
        if !self.selected_levels.contains(&id) {
            self.selected_levels.push(id);
            self.events.push(WorldEditorEvent::SelectionChanged(self.selected_levels.clone()));
        }
    }

    pub fn deselect_all_levels(&mut self) {
        self.selected_levels.clear();
        self.events.push(WorldEditorEvent::SelectionChanged(Vec::new()));
    }

    pub fn connections_for_level(&self, id: WorldLevelId) -> Vec<&LevelConnection> {
        self.connections.iter()
            .filter(|c| c.source_level == id || c.target_level == id)
            .collect()
    }

    pub fn level_count(&self) -> usize { self.levels.len() }
    pub fn connection_count(&self) -> usize { self.connections.len() }

    pub fn drain_events(&mut self) -> Vec<WorldEditorEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn tick(&mut self, delta: f32) {
        if self.auto_save {
            self.last_auto_save += delta;
            if self.last_auto_save >= self.auto_save_interval_secs {
                self.last_auto_save = 0.0;
                self.save_all_modified();
            }
        }
    }
}

impl Default for WorldEditorState {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_remove_level() {
        let mut editor = WorldEditorState::new();
        let id = editor.create_level("TestLevel");
        assert_eq!(editor.level_count(), 1);
        editor.remove_level(id);
        assert_eq!(editor.level_count(), 0);
    }

    #[test]
    fn connections() {
        let mut editor = WorldEditorState::new();
        let a = editor.create_level("A");
        let b = editor.create_level("B");
        let conn = editor.add_connection(a, b, ConnectionType::Seamless);
        assert_eq!(editor.connection_count(), 1);
        editor.remove_connection(conn);
        assert_eq!(editor.connection_count(), 0);
    }

    #[test]
    fn world_bounds_update() {
        let mut editor = WorldEditorState::new();
        let id = editor.create_level("Test");
        editor.set_level_bounds(id, [0.0, 0.0, 0.0], [100.0, 100.0, 100.0]);
        let center = editor.world_bounds.center();
        assert!((center[0] - 50.0).abs() < 1.0);
    }
}
