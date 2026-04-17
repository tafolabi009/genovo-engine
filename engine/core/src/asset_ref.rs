// engine/core/src/asset_ref.rs
//
// Typed asset reference system for the Genovo engine.
//
// Provides strongly-typed handles to assets with load-on-demand,
// dependency tracking, weak references, and hot-reload notification.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetId(pub u64);
impl AssetId { pub fn new() -> Self { Self(NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed)) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetState { Unloaded, Loading, Loaded, Failed, Unloading }

/// A strongly-typed handle to an asset. The type parameter `T` identifies
/// the asset type at compile time but is erased at runtime.
#[derive(Debug)]
pub struct AssetRef<T> {
    pub id: AssetId,
    pub path: String,
    _marker: PhantomData<T>,
}

impl<T> AssetRef<T> {
    pub fn new(path: &str) -> Self { Self { id: AssetId::new(), path: path.to_string(), _marker: PhantomData } }
    pub fn from_id(id: AssetId, path: &str) -> Self { Self { id, path: path.to_string(), _marker: PhantomData } }
}

impl<T> Clone for AssetRef<T> {
    fn clone(&self) -> Self { Self { id: self.id, path: self.path.clone(), _marker: PhantomData } }
}

impl<T> PartialEq for AssetRef<T> {
    fn eq(&self, other: &Self) -> bool { self.id == other.id }
}
impl<T> Eq for AssetRef<T> {}

impl<T> std::hash::Hash for AssetRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.id.hash(state); }
}

/// A weak reference that does not keep the asset alive.
#[derive(Debug, Clone)]
pub struct WeakAssetRef<T> { pub id: AssetId, pub path: String, _marker: PhantomData<T> }

impl<T> WeakAssetRef<T> {
    pub fn from_strong(strong: &AssetRef<T>) -> Self { Self { id: strong.id, path: strong.path.clone(), _marker: PhantomData } }
    pub fn upgrade(&self, registry: &AssetRegistry) -> Option<AssetRef<T>> {
        if registry.is_loaded(self.id) { Some(AssetRef::from_id(self.id, &self.path)) } else { None }
    }
}

/// Metadata for a tracked asset.
#[derive(Debug, Clone)]
pub struct AssetEntry {
    pub id: AssetId,
    pub path: String,
    pub type_name: String,
    pub state: AssetState,
    pub ref_count: u32,
    pub size_bytes: u64,
    pub load_time_ms: f32,
    pub dependencies: Vec<AssetId>,
    pub dependents: Vec<AssetId>,
    pub version: u32,
    pub last_modified: u64,
}

impl AssetEntry {
    pub fn new(id: AssetId, path: &str, type_name: &str) -> Self {
        Self { id, path: path.to_string(), type_name: type_name.to_string(), state: AssetState::Unloaded,
            ref_count: 0, size_bytes: 0, load_time_ms: 0.0, dependencies: Vec::new(), dependents: Vec::new(), version: 1, last_modified: 0 }
    }
}

#[derive(Debug, Clone)]
pub enum AssetEvent {
    Loaded(AssetId),
    Unloaded(AssetId),
    Reloaded(AssetId),
    Failed(AssetId, String),
    DependencyChanged(AssetId),
}

/// Central asset registry.
#[derive(Debug)]
pub struct AssetRegistry {
    pub entries: HashMap<AssetId, AssetEntry>,
    pub path_to_id: HashMap<String, AssetId>,
    pub events: Vec<AssetEvent>,
    pub total_memory: u64,
    pub memory_budget: u64,
}

impl AssetRegistry {
    pub fn new(memory_budget: u64) -> Self {
        Self { entries: HashMap::new(), path_to_id: HashMap::new(), events: Vec::new(), total_memory: 0, memory_budget }
    }

    pub fn register<T>(&mut self, asset_ref: &AssetRef<T>, type_name: &str) -> AssetId {
        if let Some(&id) = self.path_to_id.get(&asset_ref.path) { return id; }
        let entry = AssetEntry::new(asset_ref.id, &asset_ref.path, type_name);
        self.path_to_id.insert(asset_ref.path.clone(), asset_ref.id);
        self.entries.insert(asset_ref.id, entry);
        asset_ref.id
    }

    pub fn add_ref(&mut self, id: AssetId) {
        if let Some(e) = self.entries.get_mut(&id) { e.ref_count += 1; }
    }

    pub fn release_ref(&mut self, id: AssetId) {
        if let Some(e) = self.entries.get_mut(&id) {
            e.ref_count = e.ref_count.saturating_sub(1);
            if e.ref_count == 0 { self.request_unload(id); }
        }
    }

    pub fn mark_loaded(&mut self, id: AssetId, size_bytes: u64, load_time_ms: f32) {
        if let Some(e) = self.entries.get_mut(&id) {
            e.state = AssetState::Loaded;
            e.size_bytes = size_bytes;
            e.load_time_ms = load_time_ms;
            self.total_memory += size_bytes;
            self.events.push(AssetEvent::Loaded(id));
        }
    }

    pub fn mark_failed(&mut self, id: AssetId, error: &str) {
        if let Some(e) = self.entries.get_mut(&id) {
            e.state = AssetState::Failed;
            self.events.push(AssetEvent::Failed(id, error.to_string()));
        }
    }

    fn request_unload(&mut self, id: AssetId) {
        if let Some(e) = self.entries.get_mut(&id) {
            if e.state == AssetState::Loaded {
                self.total_memory = self.total_memory.saturating_sub(e.size_bytes);
                e.state = AssetState::Unloaded;
                e.size_bytes = 0;
                self.events.push(AssetEvent::Unloaded(id));
            }
        }
    }

    pub fn add_dependency(&mut self, asset_id: AssetId, dependency_id: AssetId) {
        if let Some(e) = self.entries.get_mut(&asset_id) { if !e.dependencies.contains(&dependency_id) { e.dependencies.push(dependency_id); } }
        if let Some(e) = self.entries.get_mut(&dependency_id) { if !e.dependents.contains(&asset_id) { e.dependents.push(asset_id); } }
    }

    pub fn notify_reload(&mut self, id: AssetId) {
        if let Some(e) = self.entries.get_mut(&id) {
            e.version += 1;
            self.events.push(AssetEvent::Reloaded(id));
            let dependents = e.dependents.clone();
            for dep in dependents { self.events.push(AssetEvent::DependencyChanged(dep)); }
        }
    }

    pub fn is_loaded(&self, id: AssetId) -> bool { self.entries.get(&id).map_or(false, |e| e.state == AssetState::Loaded) }
    pub fn get_state(&self, id: AssetId) -> AssetState { self.entries.get(&id).map_or(AssetState::Unloaded, |e| e.state) }
    pub fn get_by_path(&self, path: &str) -> Option<AssetId> { self.path_to_id.get(path).copied() }
    pub fn loaded_count(&self) -> usize { self.entries.values().filter(|e| e.state == AssetState::Loaded).count() }
    pub fn memory_fraction(&self) -> f32 { self.total_memory as f32 / self.memory_budget.max(1) as f32 }
    pub fn drain_events(&mut self) -> Vec<AssetEvent> { std::mem::take(&mut self.events) }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Texture;
    struct Mesh;

    #[test]
    fn test_asset_ref() {
        let r1: AssetRef<Texture> = AssetRef::new("textures/grass.png");
        let r2: AssetRef<Texture> = AssetRef::new("textures/stone.png");
        assert_ne!(r1.id, r2.id);
    }

    #[test]
    fn test_registry() {
        let mut reg = AssetRegistry::new(1024 * 1024);
        let r: AssetRef<Mesh> = AssetRef::new("meshes/hero.glb");
        reg.register(&r, "Mesh");
        reg.add_ref(r.id);
        reg.mark_loaded(r.id, 5000, 1.5);
        assert!(reg.is_loaded(r.id));
        assert_eq!(reg.total_memory, 5000);
        reg.release_ref(r.id);
        assert!(!reg.is_loaded(r.id));
    }

    #[test]
    fn test_dependencies() {
        let mut reg = AssetRegistry::new(1024 * 1024);
        let mat: AssetRef<Texture> = AssetRef::new("materials/wood.mat");
        let tex: AssetRef<Texture> = AssetRef::new("textures/wood.png");
        reg.register(&mat, "Material");
        reg.register(&tex, "Texture");
        reg.add_dependency(mat.id, tex.id);
        reg.notify_reload(tex.id);
        let events = reg.drain_events();
        assert!(events.iter().any(|e| matches!(e, AssetEvent::DependencyChanged(id) if *id == mat.id)));
    }

    #[test]
    fn test_weak_ref() {
        let mut reg = AssetRegistry::new(1024 * 1024);
        let strong: AssetRef<Texture> = AssetRef::new("tex.png");
        reg.register(&strong, "Texture");
        reg.add_ref(strong.id);
        reg.mark_loaded(strong.id, 100, 0.1);
        let weak = WeakAssetRef::from_strong(&strong);
        assert!(weak.upgrade(&reg).is_some());
        reg.release_ref(strong.id);
        assert!(weak.upgrade(&reg).is_none());
    }
}
