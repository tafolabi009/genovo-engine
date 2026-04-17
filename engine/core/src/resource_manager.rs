// engine/core/src/resource_manager.rs
//
// Resource lifecycle management: reference-counted resources, dependency
// tracking, unload when refcount=0, resource reload, resource events
// (loaded/unloaded/error), resource statistics.
//
// The ResourceManager provides a centralized system for loading, caching,
// and managing the lifecycle of all engine resources (textures, meshes,
// shaders, sounds, etc.). Resources are reference-counted and automatically
// unloaded when no longer referenced.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime};

// ---------------------------------------------------------------------------
// Resource handle
// ---------------------------------------------------------------------------

/// A unique identifier for a resource.
pub type ResourceId = u64;

/// Generate a resource ID from a path.
pub fn resource_id_from_path(path: &str) -> ResourceId {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in path.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// A reference-counted handle to a resource.
#[derive(Debug, Clone)]
pub struct ResourceHandle {
    pub id: ResourceId,
    pub path: String,
    pub resource_type: ResourceType,
    ref_count: Arc<Mutex<u32>>,
}

impl ResourceHandle {
    pub fn new(id: ResourceId, path: &str, resource_type: ResourceType) -> Self {
        Self {
            id,
            path: path.to_string(),
            resource_type,
            ref_count: Arc::new(Mutex::new(1)),
        }
    }

    /// Increment the reference count.
    pub fn add_ref(&self) {
        *self.ref_count.lock().unwrap() += 1;
    }

    /// Decrement the reference count and return the new count.
    pub fn release(&self) -> u32 {
        let mut count = self.ref_count.lock().unwrap();
        *count = count.saturating_sub(1);
        *count
    }

    /// Get the current reference count.
    pub fn ref_count(&self) -> u32 {
        *self.ref_count.lock().unwrap()
    }
}

impl PartialEq for ResourceHandle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ResourceHandle {}

impl std::hash::Hash for ResourceHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// ---------------------------------------------------------------------------
// Resource types
// ---------------------------------------------------------------------------

/// Supported resource types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Texture,
    Mesh,
    Shader,
    Material,
    Sound,
    Animation,
    Font,
    Script,
    Prefab,
    Scene,
    Config,
    Binary,
    Custom(u32),
}

impl ResourceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Texture => "texture",
            Self::Mesh => "mesh",
            Self::Shader => "shader",
            Self::Material => "material",
            Self::Sound => "sound",
            Self::Animation => "animation",
            Self::Font => "font",
            Self::Script => "script",
            Self::Prefab => "prefab",
            Self::Scene => "scene",
            Self::Config => "config",
            Self::Binary => "binary",
            Self::Custom(_) => "custom",
        }
    }

    /// Guess the resource type from a file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "png" | "jpg" | "jpeg" | "tga" | "bmp" | "dds" | "ktx" | "hdr" => Self::Texture,
            "obj" | "fbx" | "gltf" | "glb" | "dae" => Self::Mesh,
            "glsl" | "hlsl" | "wgsl" | "spv" | "frag" | "vert" | "comp" => Self::Shader,
            "mat" | "material" => Self::Material,
            "wav" | "ogg" | "mp3" | "flac" => Self::Sound,
            "anim" | "clip" => Self::Animation,
            "ttf" | "otf" | "fnt" => Self::Font,
            "lua" | "rhai" | "wasm" => Self::Script,
            "prefab" => Self::Prefab,
            "scene" | "level" | "map" => Self::Scene,
            "json" | "toml" | "yaml" | "ini" | "cfg" => Self::Config,
            _ => Self::Binary,
        }
    }
}

// ---------------------------------------------------------------------------
// Resource state
// ---------------------------------------------------------------------------

/// The current state of a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceState {
    /// Not yet loaded.
    Unloaded,
    /// Currently loading (async).
    Loading,
    /// Successfully loaded and ready to use.
    Loaded,
    /// Failed to load.
    Failed,
    /// Unloading (being freed).
    Unloading,
    /// Marked for reload.
    PendingReload,
}

// ---------------------------------------------------------------------------
// Resource entry
// ---------------------------------------------------------------------------

/// Internal entry for a managed resource.
#[derive(Debug, Clone)]
pub struct ResourceEntry {
    pub id: ResourceId,
    pub path: String,
    pub resource_type: ResourceType,
    pub state: ResourceState,
    pub ref_count: u32,
    /// Size in bytes (after loading).
    pub size_bytes: u64,
    /// Time when the resource was loaded.
    pub loaded_at: Option<Instant>,
    /// Time when the resource was last accessed.
    pub last_accessed: Instant,
    /// Number of times this resource has been loaded.
    pub load_count: u32,
    /// Error message if loading failed.
    pub error: Option<String>,
    /// Resource dependencies (other resources this one depends on).
    pub dependencies: Vec<ResourceId>,
    /// Resources that depend on this one.
    pub dependents: Vec<ResourceId>,
    /// Source file modification time (for hot reload).
    pub source_modified: Option<SystemTime>,
    /// Custom metadata.
    pub metadata: HashMap<String, String>,
}

impl ResourceEntry {
    pub fn new(id: ResourceId, path: &str, resource_type: ResourceType) -> Self {
        Self {
            id,
            path: path.to_string(),
            resource_type,
            state: ResourceState::Unloaded,
            ref_count: 0,
            size_bytes: 0,
            loaded_at: None,
            last_accessed: Instant::now(),
            load_count: 0,
            error: None,
            dependencies: Vec::new(),
            dependents: Vec::new(),
            source_modified: None,
            metadata: HashMap::new(),
        }
    }

    pub fn is_loaded(&self) -> bool { self.state == ResourceState::Loaded }
    pub fn is_loading(&self) -> bool { self.state == ResourceState::Loading }
    pub fn has_error(&self) -> bool { self.state == ResourceState::Failed }
}

// ---------------------------------------------------------------------------
// Resource events
// ---------------------------------------------------------------------------

/// Events emitted by the resource manager.
#[derive(Debug, Clone)]
pub enum ResourceEvent {
    /// A resource started loading.
    LoadStarted { id: ResourceId, path: String },
    /// A resource finished loading successfully.
    Loaded { id: ResourceId, path: String, size_bytes: u64 },
    /// A resource failed to load.
    LoadFailed { id: ResourceId, path: String, error: String },
    /// A resource was unloaded.
    Unloaded { id: ResourceId, path: String },
    /// A resource is being reloaded.
    Reloading { id: ResourceId, path: String },
    /// A resource's reference count changed.
    RefCountChanged { id: ResourceId, new_count: u32 },
    /// A dependency was added.
    DependencyAdded { resource: ResourceId, dependency: ResourceId },
}

/// Callback for resource events.
pub type ResourceEventCallback = Box<dyn Fn(&ResourceEvent) + Send + Sync>;

// ---------------------------------------------------------------------------
// Resource loader trait
// ---------------------------------------------------------------------------

/// Trait for loading a specific type of resource.
pub trait ResourceLoader: Send + Sync {
    /// The resource type this loader handles.
    fn resource_type(&self) -> ResourceType;

    /// File extensions this loader supports.
    fn supported_extensions(&self) -> &[&str];

    /// Load a resource from a file path. Returns the loaded data as bytes
    /// and the size in bytes.
    fn load(&self, path: &str) -> Result<(Vec<u8>, u64), String>;

    /// Unload a resource (free any GPU or system resources).
    fn unload(&self, id: ResourceId);

    /// Reload a resource (typically just calls load again).
    fn reload(&self, path: &str) -> Result<(Vec<u8>, u64), String> {
        self.load(path)
    }
}

// ---------------------------------------------------------------------------
// Resource statistics
// ---------------------------------------------------------------------------

/// Statistics about the resource system.
#[derive(Debug, Clone, Default)]
pub struct ResourceStats {
    /// Total number of resources tracked.
    pub total_resources: u32,
    /// Number of loaded resources.
    pub loaded_resources: u32,
    /// Number of resources currently loading.
    pub loading_resources: u32,
    /// Number of failed resources.
    pub failed_resources: u32,
    /// Total memory used by loaded resources (bytes).
    pub total_memory_bytes: u64,
    /// Total memory in megabytes.
    pub total_memory_mb: f64,
    /// Number of resources with refcount > 0.
    pub referenced_resources: u32,
    /// Number of resources with refcount == 0.
    pub unreferenced_resources: u32,
    /// Total load operations performed.
    pub total_loads: u64,
    /// Total unload operations performed.
    pub total_unloads: u64,
    /// Total reload operations performed.
    pub total_reloads: u64,
    /// Per-type counts.
    pub type_counts: HashMap<String, u32>,
    /// Per-type memory.
    pub type_memory: HashMap<String, u64>,
}

// ---------------------------------------------------------------------------
// Resource manager
// ---------------------------------------------------------------------------

/// The main resource manager.
pub struct ResourceManager {
    /// All tracked resources.
    resources: RwLock<HashMap<ResourceId, ResourceEntry>>,
    /// Registered loaders by resource type.
    loaders: RwLock<HashMap<ResourceType, Arc<dyn ResourceLoader>>>,
    /// Extension -> resource type mapping.
    extension_map: RwLock<HashMap<String, ResourceType>>,
    /// Event callbacks.
    event_callbacks: RwLock<Vec<ResourceEventCallback>>,
    /// Event queue for deferred processing.
    event_queue: Mutex<Vec<ResourceEvent>>,
    /// Running statistics.
    stats: RwLock<ResourceStats>,
    /// Root search paths for resources.
    search_paths: RwLock<Vec<PathBuf>>,
    /// Whether to automatically unload resources when refcount reaches 0.
    pub auto_unload: bool,
    /// Grace period before auto-unloading (seconds).
    pub auto_unload_delay: f32,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            resources: RwLock::new(HashMap::new()),
            loaders: RwLock::new(HashMap::new()),
            extension_map: RwLock::new(HashMap::new()),
            event_callbacks: RwLock::new(Vec::new()),
            event_queue: Mutex::new(Vec::new()),
            stats: RwLock::new(ResourceStats::default()),
            search_paths: RwLock::new(Vec::new()),
            auto_unload: false,
            auto_unload_delay: 5.0,
        }
    }

    /// Add a search path for resource files.
    pub fn add_search_path(&self, path: &str) {
        self.search_paths.write().unwrap().push(PathBuf::from(path));
    }

    /// Register a resource loader.
    pub fn register_loader(&self, loader: Arc<dyn ResourceLoader>) {
        let rt = loader.resource_type();
        let extensions: Vec<String> = loader.supported_extensions()
            .iter().map(|s| s.to_string()).collect();

        self.loaders.write().unwrap().insert(rt, loader);
        let mut ext_map = self.extension_map.write().unwrap();
        for ext in extensions {
            ext_map.insert(ext, rt);
        }
    }

    /// Register an event callback.
    pub fn on_event<F>(&self, callback: F)
    where
        F: Fn(&ResourceEvent) + Send + Sync + 'static,
    {
        self.event_callbacks.write().unwrap().push(Box::new(callback));
    }

    /// Load a resource by path. Returns a handle.
    pub fn load(&self, path: &str) -> Result<ResourceHandle, String> {
        let id = resource_id_from_path(path);

        // Check if already loaded.
        {
            let resources = self.resources.read().unwrap();
            if let Some(entry) = resources.get(&id) {
                if entry.is_loaded() {
                    return Ok(ResourceHandle::new(id, path, entry.resource_type));
                }
            }
        }

        // Determine resource type.
        let ext = Path::new(path).extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let resource_type = {
            let ext_map = self.extension_map.read().unwrap();
            ext_map.get(ext).copied().unwrap_or(ResourceType::from_extension(ext))
        };

        // Create entry.
        let mut entry = ResourceEntry::new(id, path, resource_type);
        entry.state = ResourceState::Loading;
        entry.ref_count = 1;
        self.resources.write().unwrap().insert(id, entry);

        self.emit_event(ResourceEvent::LoadStarted { id, path: path.to_string() });

        // Find the full path.
        let full_path = self.resolve_path(path);

        // Load using the appropriate loader.
        let loaders = self.loaders.read().unwrap();
        if let Some(loader) = loaders.get(&resource_type) {
            match loader.load(&full_path) {
                Ok((_, size)) => {
                    let mut resources = self.resources.write().unwrap();
                    if let Some(entry) = resources.get_mut(&id) {
                        entry.state = ResourceState::Loaded;
                        entry.size_bytes = size;
                        entry.loaded_at = Some(Instant::now());
                        entry.load_count += 1;
                    }
                    self.emit_event(ResourceEvent::Loaded { id, path: path.to_string(), size_bytes: size });
                    self.update_stats();
                    Ok(ResourceHandle::new(id, path, resource_type))
                }
                Err(e) => {
                    let mut resources = self.resources.write().unwrap();
                    if let Some(entry) = resources.get_mut(&id) {
                        entry.state = ResourceState::Failed;
                        entry.error = Some(e.clone());
                    }
                    self.emit_event(ResourceEvent::LoadFailed { id, path: path.to_string(), error: e.clone() });
                    Err(e)
                }
            }
        } else {
            // No loader registered; mark as loaded with zero size (binary passthrough).
            let mut resources = self.resources.write().unwrap();
            if let Some(entry) = resources.get_mut(&id) {
                entry.state = ResourceState::Loaded;
                entry.loaded_at = Some(Instant::now());
                entry.load_count += 1;
            }
            self.emit_event(ResourceEvent::Loaded { id, path: path.to_string(), size_bytes: 0 });
            Ok(ResourceHandle::new(id, path, resource_type))
        }
    }

    /// Unload a resource by ID.
    pub fn unload(&self, id: ResourceId) {
        let mut resources = self.resources.write().unwrap();
        if let Some(entry) = resources.get_mut(&id) {
            let path = entry.path.clone();
            let rt = entry.resource_type;

            // Unload via loader.
            let loaders = self.loaders.read().unwrap();
            if let Some(loader) = loaders.get(&rt) {
                loader.unload(id);
            }

            entry.state = ResourceState::Unloaded;
            entry.size_bytes = 0;
            entry.loaded_at = None;

            drop(resources);
            self.emit_event(ResourceEvent::Unloaded { id, path });
            self.update_stats();
        }
    }

    /// Reload a resource.
    pub fn reload(&self, id: ResourceId) -> Result<(), String> {
        let (path, rt) = {
            let resources = self.resources.read().unwrap();
            let entry = resources.get(&id).ok_or("Resource not found")?;
            (entry.path.clone(), entry.resource_type)
        };

        self.emit_event(ResourceEvent::Reloading { id, path: path.clone() });

        let full_path = self.resolve_path(&path);
        let loaders = self.loaders.read().unwrap();
        if let Some(loader) = loaders.get(&rt) {
            match loader.reload(&full_path) {
                Ok((_, size)) => {
                    let mut resources = self.resources.write().unwrap();
                    if let Some(entry) = resources.get_mut(&id) {
                        entry.state = ResourceState::Loaded;
                        entry.size_bytes = size;
                        entry.loaded_at = Some(Instant::now());
                        entry.load_count += 1;
                    }
                    drop(resources);
                    self.emit_event(ResourceEvent::Loaded { id, path, size_bytes: size });
                    self.update_stats();
                    Ok(())
                }
                Err(e) => {
                    let mut resources = self.resources.write().unwrap();
                    if let Some(entry) = resources.get_mut(&id) {
                        entry.state = ResourceState::Failed;
                        entry.error = Some(e.clone());
                    }
                    Err(e)
                }
            }
        } else {
            Ok(())
        }
    }

    /// Add a dependency between two resources.
    pub fn add_dependency(&self, resource: ResourceId, dependency: ResourceId) {
        let mut resources = self.resources.write().unwrap();
        if let Some(entry) = resources.get_mut(&resource) {
            if !entry.dependencies.contains(&dependency) {
                entry.dependencies.push(dependency);
            }
        }
        if let Some(entry) = resources.get_mut(&dependency) {
            if !entry.dependents.contains(&resource) {
                entry.dependents.push(resource);
            }
        }
        drop(resources);
        self.emit_event(ResourceEvent::DependencyAdded { resource, dependency });
    }

    /// Increment the reference count for a resource.
    pub fn add_ref(&self, id: ResourceId) {
        let mut resources = self.resources.write().unwrap();
        if let Some(entry) = resources.get_mut(&id) {
            entry.ref_count += 1;
            entry.last_accessed = Instant::now();
            let count = entry.ref_count;
            drop(resources);
            self.emit_event(ResourceEvent::RefCountChanged { id, new_count: count });
        }
    }

    /// Decrement the reference count. May trigger auto-unload.
    pub fn release(&self, id: ResourceId) {
        let should_unload;
        {
            let mut resources = self.resources.write().unwrap();
            if let Some(entry) = resources.get_mut(&id) {
                entry.ref_count = entry.ref_count.saturating_sub(1);
                let count = entry.ref_count;
                should_unload = self.auto_unload && count == 0;
                drop(resources);
                self.emit_event(ResourceEvent::RefCountChanged { id, new_count: count });
            } else {
                return;
            }
        }

        if should_unload {
            self.unload(id);
        }
    }

    /// Get the state of a resource.
    pub fn get_state(&self, id: ResourceId) -> Option<ResourceState> {
        self.resources.read().unwrap().get(&id).map(|e| e.state)
    }

    /// Get resource entry info.
    pub fn get_info(&self, id: ResourceId) -> Option<ResourceEntry> {
        self.resources.read().unwrap().get(&id).cloned()
    }

    /// Get all loaded resources.
    pub fn loaded_resources(&self) -> Vec<ResourceEntry> {
        self.resources.read().unwrap().values()
            .filter(|e| e.is_loaded())
            .cloned()
            .collect()
    }

    /// Get resource statistics.
    pub fn stats(&self) -> ResourceStats {
        self.stats.read().unwrap().clone()
    }

    /// Resolve a resource path using search paths.
    fn resolve_path(&self, path: &str) -> String {
        let search_paths = self.search_paths.read().unwrap();
        for sp in search_paths.iter() {
            let full = sp.join(path);
            if full.exists() {
                return full.to_string_lossy().to_string();
            }
        }
        path.to_string()
    }

    /// Emit a resource event to all callbacks.
    fn emit_event(&self, event: ResourceEvent) {
        let callbacks = self.event_callbacks.read().unwrap();
        for cb in callbacks.iter() {
            cb(&event);
        }
        self.event_queue.lock().unwrap().push(event);
    }

    /// Drain the event queue.
    pub fn drain_events(&self) -> Vec<ResourceEvent> {
        let mut queue = self.event_queue.lock().unwrap();
        std::mem::take(&mut *queue)
    }

    /// Update statistics.
    fn update_stats(&self) {
        let resources = self.resources.read().unwrap();
        let mut stats = ResourceStats::default();

        for entry in resources.values() {
            stats.total_resources += 1;
            match entry.state {
                ResourceState::Loaded => stats.loaded_resources += 1,
                ResourceState::Loading => stats.loading_resources += 1,
                ResourceState::Failed => stats.failed_resources += 1,
                _ => {}
            }
            stats.total_memory_bytes += entry.size_bytes;
            if entry.ref_count > 0 { stats.referenced_resources += 1; }
            else { stats.unreferenced_resources += 1; }

            let type_name = entry.resource_type.as_str().to_string();
            *stats.type_counts.entry(type_name.clone()).or_insert(0) += 1;
            *stats.type_memory.entry(type_name).or_insert(0) += entry.size_bytes;
        }

        stats.total_memory_mb = stats.total_memory_bytes as f64 / (1024.0 * 1024.0);
        *self.stats.write().unwrap() = stats;
    }

    /// Collect garbage: unload all resources with zero references.
    pub fn collect_garbage(&self) -> u32 {
        let ids: Vec<ResourceId> = {
            let resources = self.resources.read().unwrap();
            resources.values()
                .filter(|e| e.ref_count == 0 && e.is_loaded())
                .map(|e| e.id)
                .collect()
        };
        let count = ids.len() as u32;
        for id in ids {
            self.unload(id);
        }
        count
    }

    /// Get the total memory used by loaded resources.
    pub fn total_memory(&self) -> u64 {
        self.resources.read().unwrap().values()
            .filter(|e| e.is_loaded())
            .map(|e| e.size_bytes)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_id_deterministic() {
        let id1 = resource_id_from_path("textures/brick.png");
        let id2 = resource_id_from_path("textures/brick.png");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_resource_id_unique() {
        let id1 = resource_id_from_path("textures/brick.png");
        let id2 = resource_id_from_path("textures/stone.png");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_resource_type_from_extension() {
        assert_eq!(ResourceType::from_extension("png"), ResourceType::Texture);
        assert_eq!(ResourceType::from_extension("wav"), ResourceType::Sound);
        assert_eq!(ResourceType::from_extension("fbx"), ResourceType::Mesh);
        assert_eq!(ResourceType::from_extension("xyz"), ResourceType::Binary);
    }

    #[test]
    fn test_resource_handle_ref_counting() {
        let handle = ResourceHandle::new(1, "test.png", ResourceType::Texture);
        assert_eq!(handle.ref_count(), 1);
        handle.add_ref();
        assert_eq!(handle.ref_count(), 2);
        handle.release();
        assert_eq!(handle.ref_count(), 1);
        handle.release();
        assert_eq!(handle.ref_count(), 0);
    }

    #[test]
    fn test_resource_manager_basic() {
        let manager = ResourceManager::new();
        // Load without a registered loader -- should still work (binary passthrough).
        let result = manager.load("test_resource.bin");
        assert!(result.is_ok());
        let handle = result.unwrap();
        assert_eq!(manager.get_state(handle.id), Some(ResourceState::Loaded));
    }

    #[test]
    fn test_resource_manager_events() {
        let manager = ResourceManager::new();
        let events_received = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events_received.clone();
        manager.on_event(move |event| {
            events_clone.lock().unwrap().push(format!("{:?}", event));
        });

        manager.load("test.bin").unwrap();
        let events = events_received.lock().unwrap();
        assert!(events.len() >= 2); // LoadStarted + Loaded.
    }

    #[test]
    fn test_resource_manager_unload() {
        let manager = ResourceManager::new();
        let handle = manager.load("test.bin").unwrap();
        manager.unload(handle.id);
        assert_eq!(manager.get_state(handle.id), Some(ResourceState::Unloaded));
    }

    #[test]
    fn test_dependency_tracking() {
        let manager = ResourceManager::new();
        let h1 = manager.load("material.mat").unwrap();
        let h2 = manager.load("texture.png").unwrap();
        manager.add_dependency(h1.id, h2.id);

        let info = manager.get_info(h1.id).unwrap();
        assert!(info.dependencies.contains(&h2.id));
    }
}
