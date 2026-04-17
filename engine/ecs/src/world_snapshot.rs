// engine/ecs/src/world_snapshot.rs
//
// ECS world snapshot system for the Genovo ECS.
//
// Provides serialize entire world state, diff two snapshots, restore from
// snapshot, and support for undo/redo and networking.

use std::any::TypeId;
use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;

pub type Entity = u64;
pub type SnapshotId = u64;
pub type ComponentTypeId = u64;

pub const MAX_SNAPSHOT_HISTORY: usize = 100;
pub const MAX_UNDO_STEPS: usize = 50;

#[derive(Debug, Clone)]
pub struct ComponentData {
    pub type_id: ComponentTypeId,
    pub type_name: String,
    pub data: Vec<u8>,
    pub version: u32,
}

impl ComponentData {
    pub fn new(type_id: ComponentTypeId, type_name: &str, data: Vec<u8>) -> Self {
        Self { type_id, type_name: type_name.to_string(), data, version: 1 }
    }
    pub fn size(&self) -> usize { self.data.len() }
}

#[derive(Debug, Clone)]
pub struct EntitySnapshot {
    pub entity: Entity,
    pub components: Vec<ComponentData>,
    pub alive: bool,
}

impl EntitySnapshot {
    pub fn new(entity: Entity) -> Self { Self { entity, components: Vec::new(), alive: true } }
    pub fn add_component(&mut self, comp: ComponentData) { self.components.push(comp); }
    pub fn has_component(&self, type_id: ComponentTypeId) -> bool { self.components.iter().any(|c| c.type_id == type_id) }
    pub fn get_component(&self, type_id: ComponentTypeId) -> Option<&ComponentData> { self.components.iter().find(|c| c.type_id == type_id) }
    pub fn component_count(&self) -> usize { self.components.len() }
    pub fn total_size(&self) -> usize { self.components.iter().map(|c| c.size()).sum() }
}

#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    pub type_id: ComponentTypeId,
    pub type_name: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct WorldSnapshot {
    pub id: SnapshotId,
    pub frame: u64,
    pub timestamp: f64,
    pub entities: HashMap<Entity, EntitySnapshot>,
    pub resources: HashMap<ComponentTypeId, ResourceSnapshot>,
    pub metadata: HashMap<String, String>,
    pub total_entities: u32,
    pub total_components: u32,
}

impl WorldSnapshot {
    pub fn new(id: SnapshotId, frame: u64, timestamp: f64) -> Self {
        Self {
            id, frame, timestamp, entities: HashMap::new(), resources: HashMap::new(),
            metadata: HashMap::new(), total_entities: 0, total_components: 0,
        }
    }

    pub fn add_entity(&mut self, snapshot: EntitySnapshot) {
        self.total_components += snapshot.component_count() as u32;
        self.total_entities += 1;
        self.entities.insert(snapshot.entity, snapshot);
    }

    pub fn add_resource(&mut self, resource: ResourceSnapshot) {
        self.resources.insert(resource.type_id, resource);
    }

    pub fn get_entity(&self, entity: Entity) -> Option<&EntitySnapshot> { self.entities.get(&entity) }

    pub fn total_size(&self) -> usize {
        self.entities.values().map(|e| e.total_size()).sum::<usize>()
            + self.resources.values().map(|r| r.data.len()).sum::<usize>()
    }

    pub fn entity_count(&self) -> usize { self.entities.len() }
    pub fn set_metadata(&mut self, key: &str, value: &str) { self.metadata.insert(key.to_string(), value.to_string()); }
}

// ---------------------------------------------------------------------------
// Snapshot diff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum DiffOperation {
    EntityAdded(Entity),
    EntityRemoved(Entity),
    ComponentAdded { entity: Entity, type_id: ComponentTypeId, type_name: String, data: Vec<u8> },
    ComponentRemoved { entity: Entity, type_id: ComponentTypeId, type_name: String },
    ComponentModified { entity: Entity, type_id: ComponentTypeId, type_name: String, old_data: Vec<u8>, new_data: Vec<u8> },
    ResourceAdded { type_id: ComponentTypeId, data: Vec<u8> },
    ResourceRemoved { type_id: ComponentTypeId },
    ResourceModified { type_id: ComponentTypeId, old_data: Vec<u8>, new_data: Vec<u8> },
}

#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    pub from_id: SnapshotId,
    pub to_id: SnapshotId,
    pub from_frame: u64,
    pub to_frame: u64,
    pub operations: Vec<DiffOperation>,
}

impl SnapshotDiff {
    pub fn compute(from: &WorldSnapshot, to: &WorldSnapshot) -> Self {
        let mut ops = Vec::new();
        // Find added and modified entities.
        for (entity, to_snap) in &to.entities {
            if let Some(from_snap) = from.entities.get(entity) {
                // Entity exists in both. Check components.
                for comp in &to_snap.components {
                    if let Some(from_comp) = from_snap.get_component(comp.type_id) {
                        if from_comp.data != comp.data {
                            ops.push(DiffOperation::ComponentModified {
                                entity: *entity, type_id: comp.type_id, type_name: comp.type_name.clone(),
                                old_data: from_comp.data.clone(), new_data: comp.data.clone(),
                            });
                        }
                    } else {
                        ops.push(DiffOperation::ComponentAdded {
                            entity: *entity, type_id: comp.type_id, type_name: comp.type_name.clone(), data: comp.data.clone(),
                        });
                    }
                }
                for comp in &from_snap.components {
                    if !to_snap.has_component(comp.type_id) {
                        ops.push(DiffOperation::ComponentRemoved { entity: *entity, type_id: comp.type_id, type_name: comp.type_name.clone() });
                    }
                }
            } else {
                ops.push(DiffOperation::EntityAdded(*entity));
                for comp in &to_snap.components {
                    ops.push(DiffOperation::ComponentAdded {
                        entity: *entity, type_id: comp.type_id, type_name: comp.type_name.clone(), data: comp.data.clone(),
                    });
                }
            }
        }
        // Find removed entities.
        for entity in from.entities.keys() {
            if !to.entities.contains_key(entity) { ops.push(DiffOperation::EntityRemoved(*entity)); }
        }
        // Resource diffs.
        for (type_id, to_res) in &to.resources {
            if let Some(from_res) = from.resources.get(type_id) {
                if from_res.data != to_res.data {
                    ops.push(DiffOperation::ResourceModified { type_id: *type_id, old_data: from_res.data.clone(), new_data: to_res.data.clone() });
                }
            } else {
                ops.push(DiffOperation::ResourceAdded { type_id: *type_id, data: to_res.data.clone() });
            }
        }
        for type_id in from.resources.keys() {
            if !to.resources.contains_key(type_id) { ops.push(DiffOperation::ResourceRemoved { type_id: *type_id }); }
        }

        Self { from_id: from.id, to_id: to.id, from_frame: from.frame, to_frame: to.frame, operations: ops }
    }

    pub fn operation_count(&self) -> usize { self.operations.len() }
    pub fn is_empty(&self) -> bool { self.operations.is_empty() }
    pub fn has_entity_changes(&self) -> bool {
        self.operations.iter().any(|op| matches!(op, DiffOperation::EntityAdded(_) | DiffOperation::EntityRemoved(_)))
    }
    pub fn changed_entities(&self) -> HashSet<Entity> {
        let mut entities = HashSet::new();
        for op in &self.operations {
            match op {
                DiffOperation::EntityAdded(e) | DiffOperation::EntityRemoved(e) => { entities.insert(*e); }
                DiffOperation::ComponentAdded { entity, .. } | DiffOperation::ComponentRemoved { entity, .. } | DiffOperation::ComponentModified { entity, .. } => { entities.insert(*entity); }
                _ => {}
            }
        }
        entities
    }
    pub fn total_data_size(&self) -> usize {
        self.operations.iter().map(|op| match op {
            DiffOperation::ComponentAdded { data, .. } => data.len(),
            DiffOperation::ComponentModified { old_data, new_data, .. } => old_data.len() + new_data.len(),
            DiffOperation::ResourceAdded { data, .. } => data.len(),
            DiffOperation::ResourceModified { old_data, new_data, .. } => old_data.len() + new_data.len(),
            _ => 0,
        }).sum()
    }
}

// ---------------------------------------------------------------------------
// Undo/Redo manager
// ---------------------------------------------------------------------------

pub struct UndoRedoManager {
    undo_stack: Vec<WorldSnapshot>,
    redo_stack: Vec<WorldSnapshot>,
    max_undo: usize,
    next_id: SnapshotId,
}

impl UndoRedoManager {
    pub fn new(max_undo: usize) -> Self {
        Self { undo_stack: Vec::new(), redo_stack: Vec::new(), max_undo: max_undo.min(MAX_UNDO_STEPS), next_id: 1 }
    }

    pub fn push_state(&mut self, snapshot: WorldSnapshot) {
        if self.undo_stack.len() >= self.max_undo { self.undo_stack.remove(0); }
        self.undo_stack.push(snapshot);
        self.redo_stack.clear();
    }

    pub fn undo(&mut self) -> Option<&WorldSnapshot> {
        if let Some(state) = self.undo_stack.pop() {
            self.redo_stack.push(state);
            self.undo_stack.last()
        } else { None }
    }

    pub fn redo(&mut self) -> Option<&WorldSnapshot> {
        if let Some(state) = self.redo_stack.pop() {
            self.undo_stack.push(state);
            self.undo_stack.last()
        } else { None }
    }

    pub fn can_undo(&self) -> bool { self.undo_stack.len() > 1 }
    pub fn can_redo(&self) -> bool { !self.redo_stack.is_empty() }
    pub fn undo_count(&self) -> usize { self.undo_stack.len() }
    pub fn redo_count(&self) -> usize { self.redo_stack.len() }
    pub fn clear(&mut self) { self.undo_stack.clear(); self.redo_stack.clear(); }
    pub fn current(&self) -> Option<&WorldSnapshot> { self.undo_stack.last() }
}

// ---------------------------------------------------------------------------
// Snapshot manager
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct SnapshotManagerStats {
    pub snapshots_taken: u64,
    pub total_snapshot_size: usize,
    pub avg_snapshot_size: usize,
    pub diffs_computed: u64,
    pub undo_operations: u64,
    pub redo_operations: u64,
}

pub struct SnapshotManager {
    history: Vec<WorldSnapshot>,
    max_history: usize,
    undo_redo: UndoRedoManager,
    next_id: SnapshotId,
    stats: SnapshotManagerStats,
}

impl SnapshotManager {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(), max_history: max_history.min(MAX_SNAPSHOT_HISTORY),
            undo_redo: UndoRedoManager::new(MAX_UNDO_STEPS),
            next_id: 1, stats: SnapshotManagerStats::default(),
        }
    }

    pub fn take_snapshot(&mut self, frame: u64, timestamp: f64) -> SnapshotId {
        let id = self.next_id; self.next_id += 1;
        let snapshot = WorldSnapshot::new(id, frame, timestamp);
        if self.history.len() >= self.max_history { self.history.remove(0); }
        self.history.push(snapshot);
        self.stats.snapshots_taken += 1;
        id
    }

    pub fn get_current_mut(&mut self) -> Option<&mut WorldSnapshot> { self.history.last_mut() }
    pub fn get_snapshot(&self, id: SnapshotId) -> Option<&WorldSnapshot> { self.history.iter().find(|s| s.id == id) }
    pub fn latest(&self) -> Option<&WorldSnapshot> { self.history.last() }

    pub fn diff(&mut self, from_id: SnapshotId, to_id: SnapshotId) -> Option<SnapshotDiff> {
        let from = self.history.iter().find(|s| s.id == from_id)?.clone();
        let to = self.history.iter().find(|s| s.id == to_id)?;
        self.stats.diffs_computed += 1;
        Some(SnapshotDiff::compute(&from, to))
    }

    pub fn push_undo_state(&mut self, snapshot: WorldSnapshot) { self.undo_redo.push_state(snapshot); }
    pub fn undo(&mut self) -> Option<&WorldSnapshot> { self.stats.undo_operations += 1; self.undo_redo.undo() }
    pub fn redo(&mut self) -> Option<&WorldSnapshot> { self.stats.redo_operations += 1; self.undo_redo.redo() }
    pub fn can_undo(&self) -> bool { self.undo_redo.can_undo() }
    pub fn can_redo(&self) -> bool { self.undo_redo.can_redo() }
    pub fn stats(&self) -> &SnapshotManagerStats { &self.stats }
    pub fn history_count(&self) -> usize { self.history.len() }
    pub fn clear_history(&mut self) { self.history.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_snapshot() {
        let mut es = EntitySnapshot::new(1);
        es.add_component(ComponentData::new(100, "Health", vec![1, 2, 3, 4]));
        assert_eq!(es.component_count(), 1);
        assert!(es.has_component(100));
    }

    #[test]
    fn test_world_snapshot() {
        let mut ws = WorldSnapshot::new(1, 0, 0.0);
        let mut es = EntitySnapshot::new(1);
        es.add_component(ComponentData::new(100, "Health", vec![0; 8]));
        ws.add_entity(es);
        assert_eq!(ws.entity_count(), 1);
    }

    #[test]
    fn test_diff() {
        let mut s1 = WorldSnapshot::new(1, 0, 0.0);
        let mut e1 = EntitySnapshot::new(1);
        e1.add_component(ComponentData::new(100, "Health", vec![10]));
        s1.add_entity(e1);

        let mut s2 = WorldSnapshot::new(2, 1, 0.016);
        let mut e2 = EntitySnapshot::new(1);
        e2.add_component(ComponentData::new(100, "Health", vec![5]));
        s2.add_entity(e2);
        let mut e3 = EntitySnapshot::new(2);
        e3.add_component(ComponentData::new(200, "Position", vec![0; 12]));
        s2.add_entity(e3);

        let diff = SnapshotDiff::compute(&s1, &s2);
        assert!(!diff.is_empty());
        assert!(diff.has_entity_changes());
    }

    #[test]
    fn test_undo_redo() {
        let mut mgr = UndoRedoManager::new(10);
        mgr.push_state(WorldSnapshot::new(1, 0, 0.0));
        mgr.push_state(WorldSnapshot::new(2, 1, 0.016));
        assert!(mgr.can_undo());
        mgr.undo();
        assert!(mgr.can_redo());
        mgr.redo();
    }
}
