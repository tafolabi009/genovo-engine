// engine/ecs/src/world_access.rs
//
// Safe world access patterns for the Genovo ECS.
//
// Provides SystemState for cached parameter access, QueryState lifecycle
// management, and exclusive world access guards.

use std::collections::HashMap;
use std::marker::PhantomData;

pub type ComponentTypeId = u64;
pub type EntityId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessMode { Read, Write, Exclusive }

#[derive(Debug, Clone)]
pub struct ComponentAccess { pub type_id: ComponentTypeId, pub mode: AccessMode, pub type_name: String }

/// SystemState caches parameter access metadata so repeated system runs
/// don't need to re-derive access patterns each tick.
#[derive(Debug)]
pub struct SystemState {
    pub system_name: String,
    pub reads: Vec<ComponentTypeId>,
    pub writes: Vec<ComponentTypeId>,
    pub exclusive: bool,
    pub initialized: bool,
    pub last_tick: u64,
    pub change_tick: u64,
    access_cache: Vec<ComponentAccess>,
}

impl SystemState {
    pub fn new(name: &str) -> Self {
        Self { system_name: name.to_string(), reads: Vec::new(), writes: Vec::new(), exclusive: false,
            initialized: false, last_tick: 0, change_tick: 0, access_cache: Vec::new() }
    }

    pub fn add_read(&mut self, type_id: ComponentTypeId, name: &str) {
        self.reads.push(type_id);
        self.access_cache.push(ComponentAccess { type_id, mode: AccessMode::Read, type_name: name.to_string() });
    }

    pub fn add_write(&mut self, type_id: ComponentTypeId, name: &str) {
        self.writes.push(type_id);
        self.access_cache.push(ComponentAccess { type_id, mode: AccessMode::Write, type_name: name.to_string() });
    }

    pub fn set_exclusive(&mut self) { self.exclusive = true; }

    pub fn conflicts_with(&self, other: &SystemState) -> bool {
        if self.exclusive || other.exclusive { return true; }
        for w in &self.writes { if other.reads.contains(w) || other.writes.contains(w) { return true; } }
        for w in &other.writes { if self.reads.contains(w) { return true; } }
        false
    }

    pub fn validate(&self) -> Result<(), String> {
        // Check no component is both read and written.
        for r in &self.reads {
            if self.writes.contains(r) {
                return Err(format!("Component {} is both read and written in system {}", r, self.system_name));
            }
        }
        Ok(())
    }

    pub fn initialize(&mut self, tick: u64) { self.initialized = true; self.last_tick = tick; self.change_tick = tick; }
    pub fn update_tick(&mut self, tick: u64) { self.last_tick = tick; }
    pub fn accesses(&self) -> &[ComponentAccess] { &self.access_cache }
}

/// QueryState manages the lifecycle of an ECS query, including archetype
/// matching cache and iteration state.
#[derive(Debug)]
pub struct QueryState {
    pub query_id: u64,
    pub component_types: Vec<ComponentTypeId>,
    pub filter_types: Vec<ComponentTypeId>,
    pub exclude_types: Vec<ComponentTypeId>,
    matched_archetypes: Vec<u32>,
    pub is_cached: bool,
    pub last_archetype_gen: u64,
    pub entity_count: usize,
}

static mut NEXT_QUERY_ID: u64 = 1;

impl QueryState {
    pub fn new(components: Vec<ComponentTypeId>) -> Self {
        let id = unsafe { let v = NEXT_QUERY_ID; NEXT_QUERY_ID += 1; v };
        Self { query_id: id, component_types: components, filter_types: Vec::new(), exclude_types: Vec::new(),
            matched_archetypes: Vec::new(), is_cached: false, last_archetype_gen: 0, entity_count: 0 }
    }

    pub fn with_filter(mut self, filter: ComponentTypeId) -> Self { self.filter_types.push(filter); self }
    pub fn without(mut self, exclude: ComponentTypeId) -> Self { self.exclude_types.push(exclude); self }

    pub fn update_cache(&mut self, archetype_gen: u64, archetypes: &[(u32, Vec<ComponentTypeId>)]) {
        if self.is_cached && self.last_archetype_gen == archetype_gen { return; }
        self.matched_archetypes.clear();
        for (id, components) in archetypes {
            let all_present = self.component_types.iter().all(|c| components.contains(c));
            let none_excluded = self.exclude_types.iter().all(|c| !components.contains(c));
            if all_present && none_excluded { self.matched_archetypes.push(*id); }
        }
        self.is_cached = true;
        self.last_archetype_gen = archetype_gen;
    }

    pub fn matched_archetypes(&self) -> &[u32] { &self.matched_archetypes }
    pub fn invalidate(&mut self) { self.is_cached = false; }
}

/// ExclusiveWorldAccess provides a guard that ensures only one system
/// has mutable access to the world at a time.
#[derive(Debug)]
pub struct ExclusiveWorldAccess {
    locked: bool,
    holder: Option<String>,
    queue: Vec<String>,
}

impl ExclusiveWorldAccess {
    pub fn new() -> Self { Self { locked: false, holder: None, queue: Vec::new() } }

    pub fn try_acquire(&mut self, system_name: &str) -> Option<ExclusiveGuard> {
        if self.locked { self.queue.push(system_name.to_string()); None }
        else { self.locked = true; self.holder = Some(system_name.to_string()); Some(ExclusiveGuard { _phantom: PhantomData }) }
    }

    pub fn release(&mut self) { self.locked = false; self.holder = None; }
    pub fn is_locked(&self) -> bool { self.locked }
    pub fn current_holder(&self) -> Option<&str> { self.holder.as_deref() }
    pub fn queue_length(&self) -> usize { self.queue.len() }
}

/// RAII guard for exclusive world access.
#[derive(Debug)]
pub struct ExclusiveGuard { _phantom: PhantomData<*mut ()> }

/// SystemParam trait-like abstraction for cached parameter extraction.
#[derive(Debug)]
pub struct CachedParam<T> { pub value: T, pub type_id: ComponentTypeId, pub is_changed: bool }

/// World access statistics.
#[derive(Debug, Clone, Default)]
pub struct WorldAccessStats {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub exclusive_acquisitions: u64,
    pub conflict_count: u64,
}

impl WorldAccessStats {
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f32 / total as f32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_state() {
        let mut s = SystemState::new("movement");
        s.add_read(1, "Position");
        s.add_write(2, "Velocity");
        assert!(s.validate().is_ok());
        assert_eq!(s.accesses().len(), 2);
    }

    #[test]
    fn test_conflict_detection() {
        let mut a = SystemState::new("a");
        a.add_write(1, "Position");
        let mut b = SystemState::new("b");
        b.add_read(1, "Position");
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn test_no_conflict() {
        let mut a = SystemState::new("a");
        a.add_read(1, "Position");
        let mut b = SystemState::new("b");
        b.add_read(1, "Position");
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn test_query_state() {
        let mut qs = QueryState::new(vec![1, 2]);
        let archetypes = vec![(0, vec![1, 2, 3]), (1, vec![1, 4]), (2, vec![1, 2])];
        qs.update_cache(1, &archetypes);
        assert_eq!(qs.matched_archetypes().len(), 2); // 0 and 2
    }

    #[test]
    fn test_exclusive_access() {
        let mut exc = ExclusiveWorldAccess::new();
        let guard = exc.try_acquire("system_a");
        assert!(guard.is_some());
        assert!(exc.try_acquire("system_b").is_none());
        exc.release();
        assert!(exc.try_acquire("system_b").is_some());
    }
}
