//! Plugin System
//!
//! Provides a trait-based plugin interface for extending the Genovo engine
//! with optional modules. Plugins declare their dependencies, receive
//! lifecycle callbacks, and can be loaded from dynamic libraries at runtime.
//!
//! # Architecture
//!
//! ```text
//!  ┌──────────────────────────────────────────────────┐
//!  │ PluginManager                                     │
//!  │                                                   │
//!  │  ┌────────┐  ┌────────┐  ┌────────┐              │
//!  │  │Plugin  │  │ Plugin │  │ Plugin │  ...          │
//!  │  │ "audio" │  │ "net"  │  │ "user" │              │
//!  │  └────┬───┘  └───┬────┘  └───┬────┘              │
//!  │       │          │           │                    │
//!  │  ┌────▼──────────▼───────────▼────────┐           │
//!  │  │  PluginContext                      │           │
//!  │  │  (scoped access to engine systems) │           │
//!  │  └────────────────────────────────────┘           │
//!  └──────────────────────────────────────────────────┘
//! ```
//!
//! # Dynamic Plugins
//!
//! On supported platforms, plugins can be loaded from shared libraries
//! (`.dll` on Windows, `.so` on Linux, `.dylib` on macOS). The library
//! must export a `create_plugin` function with C linkage:
//!
//! ```ignore
//! #[no_mangle]
//! pub extern "C" fn create_plugin() -> *mut dyn Plugin {
//!     Box::into_raw(Box::new(MyPlugin::new()))
//! }
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// PluginVersion
// ---------------------------------------------------------------------------

/// Semantic version for a plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PluginVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl PluginVersion {
    /// Create a new version.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with `required`.
    ///
    /// Compatible means same major version and >= minor.patch.
    pub fn is_compatible_with(&self, required: &PluginVersion) -> bool {
        self.major == required.major
            && (self.minor > required.minor
                || (self.minor == required.minor && self.patch >= required.patch))
    }
}

impl fmt::Display for PluginVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ---------------------------------------------------------------------------
// PluginState
// ---------------------------------------------------------------------------

/// Lifecycle state of a plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    /// Plugin is registered but not loaded.
    Registered,
    /// Plugin is in the process of loading.
    Loading,
    /// Plugin is fully loaded and active.
    Active,
    /// Plugin is in the process of unloading.
    Unloading,
    /// Plugin has been unloaded.
    Unloaded,
    /// Plugin failed to load.
    Failed,
}

// ---------------------------------------------------------------------------
// Plugin trait
// ---------------------------------------------------------------------------

/// The core plugin trait. All engine extensions implement this trait.
///
/// Plugins receive lifecycle callbacks and can interact with the engine
/// through the [`PluginContext`].
///
/// # Implementation
///
/// ```ignore
/// struct AudioPlugin {
///     volume: f32,
/// }
///
/// impl Plugin for AudioPlugin {
///     fn name(&self) -> &str { "audio" }
///
///     fn version(&self) -> PluginVersion {
///         PluginVersion::new(1, 0, 0)
///     }
///
///     fn dependencies(&self) -> &[&str] { &["core"] }
///
///     fn on_load(&mut self, ctx: &mut PluginContext) {
///         ctx.set_resource("audio_volume", Box::new(self.volume));
///         log::info!("Audio plugin loaded");
///     }
///
///     fn on_update(&mut self, dt: f32, ctx: &mut PluginContext) {
///         // Process audio...
///     }
///
///     fn on_unload(&mut self, ctx: &mut PluginContext) {
///         log::info!("Audio plugin unloaded");
///     }
/// }
/// ```
pub trait Plugin: Send + Sync + Any {
    /// Unique name of this plugin.
    fn name(&self) -> &str;

    /// Semantic version of this plugin.
    fn version(&self) -> PluginVersion;

    /// Names of plugins that must be loaded before this one.
    ///
    /// The plugin manager performs a topological sort on these dependencies
    /// to determine load order.
    fn dependencies(&self) -> &[&str] {
        &[]
    }

    /// Called when the plugin is loaded into the engine.
    ///
    /// Use this to register systems, add resources, set up initial state.
    fn on_load(&mut self, ctx: &mut PluginContext);

    /// Called when the plugin is being unloaded.
    ///
    /// Clean up resources, deregister systems, etc.
    fn on_unload(&mut self, ctx: &mut PluginContext);

    /// Called once per frame while the plugin is active.
    fn on_update(&mut self, dt: f32, ctx: &mut PluginContext) {
        let _ = (dt, ctx);
    }

    /// Called at a fixed timestep (typically for physics/simulation).
    fn on_fixed_update(&mut self, fixed_dt: f32, ctx: &mut PluginContext) {
        let _ = (fixed_dt, ctx);
    }

    /// Called after all plugins have been updated (post-processing).
    fn on_late_update(&mut self, dt: f32, ctx: &mut PluginContext) {
        let _ = (dt, ctx);
    }

    /// Called when the engine is shutting down (before unload).
    fn on_shutdown(&mut self, ctx: &mut PluginContext) {
        let _ = ctx;
    }

    /// Called when the window is resized.
    fn on_resize(&mut self, width: u32, height: u32, ctx: &mut PluginContext) {
        let _ = (width, height, ctx);
    }

    /// Whether this plugin supports hot-reloading.
    fn supports_hot_reload(&self) -> bool {
        false
    }

    /// Save plugin state before hot-reload. Returns serialized state.
    fn save_state(&self) -> Option<Vec<u8>> {
        None
    }

    /// Restore plugin state after hot-reload.
    fn restore_state(&mut self, _state: &[u8]) {}

    /// Get plugin-specific configuration as a generic key-value map.
    fn config(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Apply configuration to the plugin.
    fn apply_config(&mut self, _config: &HashMap<String, String>) {}

    /// Downcast helper: returns self as a `&dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Downcast helper: returns self as a `&mut dyn Any` for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ---------------------------------------------------------------------------
// PluginContext
// ---------------------------------------------------------------------------

/// Scoped access to engine systems, passed to plugins during lifecycle
/// callbacks.
///
/// The plugin context acts as a service locator: plugins can register
/// and retrieve shared resources by type or by string key, post events,
/// and query the engine configuration.
pub struct PluginContext {
    /// Named resources accessible to all plugins.
    resources: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// Typed resources, keyed by TypeId.
    typed_resources: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Pending messages between plugins: (source, target, payload).
    messages: Vec<PluginMessage>,
    /// Engine configuration.
    engine_config: HashMap<String, String>,
    /// Registered service factories.
    services: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// Log of events for debugging.
    event_log: Vec<String>,
    /// Maximum event log size.
    max_event_log: usize,
}

/// A message sent between plugins.
#[derive(Debug, Clone)]
pub struct PluginMessage {
    /// Name of the sending plugin.
    pub source: String,
    /// Name of the target plugin (empty string = broadcast).
    pub target: String,
    /// Message type identifier.
    pub message_type: String,
    /// Message payload as string (JSON recommended).
    pub payload: String,
}

impl PluginContext {
    /// Create a new plugin context.
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            typed_resources: HashMap::new(),
            messages: Vec::new(),
            engine_config: HashMap::new(),
            services: HashMap::new(),
            event_log: Vec::new(),
            max_event_log: 1000,
        }
    }

    // -- Named resources ---------------------------------------------------

    /// Set a named resource accessible to all plugins.
    pub fn set_resource(&mut self, name: impl Into<String>, value: Box<dyn Any + Send + Sync>) {
        self.resources.insert(name.into(), value);
    }

    /// Get a named resource by reference.
    pub fn get_resource<T: 'static>(&self, name: &str) -> Option<&T> {
        self.resources.get(name).and_then(|v| v.downcast_ref::<T>())
    }

    /// Get a named resource by mutable reference.
    pub fn get_resource_mut<T: 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.resources
            .get_mut(name)
            .and_then(|v| v.downcast_mut::<T>())
    }

    /// Remove a named resource, returning it if it existed.
    pub fn remove_resource(&mut self, name: &str) -> Option<Box<dyn Any + Send + Sync>> {
        self.resources.remove(name)
    }

    /// Check if a named resource exists.
    pub fn has_resource(&self, name: &str) -> bool {
        self.resources.contains_key(name)
    }

    // -- Typed resources ---------------------------------------------------

    /// Insert a typed resource.
    pub fn insert_typed<T: Send + Sync + 'static>(&mut self, value: T) {
        self.typed_resources
            .insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Get a typed resource by reference.
    pub fn get_typed<T: 'static>(&self) -> Option<&T> {
        self.typed_resources
            .get(&TypeId::of::<T>())
            .and_then(|v| v.downcast_ref::<T>())
    }

    /// Get a typed resource by mutable reference.
    pub fn get_typed_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.typed_resources
            .get_mut(&TypeId::of::<T>())
            .and_then(|v| v.downcast_mut::<T>())
    }

    /// Remove a typed resource.
    pub fn remove_typed<T: 'static>(&mut self) -> Option<T> {
        self.typed_resources
            .remove(&TypeId::of::<T>())
            .and_then(|v| v.downcast::<T>().ok())
            .map(|b| *b)
    }

    // -- Messaging ---------------------------------------------------------

    /// Send a message to a specific plugin.
    pub fn send_message(
        &mut self,
        source: impl Into<String>,
        target: impl Into<String>,
        message_type: impl Into<String>,
        payload: impl Into<String>,
    ) {
        self.messages.push(PluginMessage {
            source: source.into(),
            target: target.into(),
            message_type: message_type.into(),
            payload: payload.into(),
        });
    }

    /// Broadcast a message to all plugins.
    pub fn broadcast(
        &mut self,
        source: impl Into<String>,
        message_type: impl Into<String>,
        payload: impl Into<String>,
    ) {
        self.messages.push(PluginMessage {
            source: source.into(),
            target: String::new(),
            message_type: message_type.into(),
            payload: payload.into(),
        });
    }

    /// Take all pending messages for a specific plugin.
    pub fn take_messages(&mut self, plugin_name: &str) -> Vec<PluginMessage> {
        let mut for_plugin = Vec::new();
        let mut remaining = Vec::new();

        for msg in std::mem::take(&mut self.messages) {
            if msg.target == plugin_name || msg.target.is_empty() {
                for_plugin.push(msg);
            } else {
                remaining.push(msg);
            }
        }

        self.messages = remaining;
        for_plugin
    }

    /// Drain all pending messages.
    pub fn drain_messages(&mut self) -> Vec<PluginMessage> {
        std::mem::take(&mut self.messages)
    }

    // -- Engine config -----------------------------------------------------

    /// Get an engine configuration value.
    pub fn config(&self, key: &str) -> Option<&str> {
        self.engine_config.get(key).map(|s| s.as_str())
    }

    /// Set an engine configuration value.
    pub fn set_config(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.engine_config.insert(key.into(), value.into());
    }

    // -- Event log ---------------------------------------------------------

    /// Log an event for debugging.
    pub fn log_event(&mut self, event: impl Into<String>) {
        if self.event_log.len() >= self.max_event_log {
            self.event_log.remove(0);
        }
        self.event_log.push(event.into());
    }

    /// Get the event log.
    pub fn event_log(&self) -> &[String] {
        &self.event_log
    }

    /// Clear the event log.
    pub fn clear_event_log(&mut self) {
        self.event_log.clear();
    }
}

impl Default for PluginContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PluginInfo
// ---------------------------------------------------------------------------

/// Metadata about a registered plugin.
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// Plugin name.
    pub name: String,
    /// Plugin version.
    pub version: PluginVersion,
    /// Plugin dependencies.
    pub dependencies: Vec<String>,
    /// Current state.
    pub state: PluginState,
    /// Whether the plugin was loaded from a dynamic library.
    pub is_dynamic: bool,
    /// Path to the dynamic library (if applicable).
    pub library_path: Option<PathBuf>,
    /// Load order index.
    pub load_order: usize,
}

// ---------------------------------------------------------------------------
// PluginManager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of all engine plugins.
///
/// The plugin manager handles registration, dependency resolution,
/// load ordering (topological sort), and lifecycle dispatching.
///
/// # Usage
///
/// ```ignore
/// let mut manager = PluginManager::new();
///
/// // Register plugins.
/// manager.register_plugin(Box::new(AudioPlugin::new()));
/// manager.register_plugin(Box::new(NetworkPlugin::new()));
///
/// // Load all plugins in dependency order.
/// let mut ctx = PluginContext::new();
/// manager.load_all(&mut ctx)?;
///
/// // Each frame:
/// manager.update_all(dt, &mut ctx);
///
/// // Shutdown:
/// manager.unload_all(&mut ctx);
/// ```
pub struct PluginManager {
    /// Registered plugins, keyed by name.
    plugins: Vec<PluginEntry>,
    /// Name-to-index lookup.
    name_to_index: HashMap<String, usize>,
    /// Computed load order (indices into `plugins`).
    load_order: Vec<usize>,
    /// Whether plugins have been loaded.
    loaded: bool,
    /// Dynamic library handles (kept alive while plugins are loaded).
    #[cfg(feature = "dynamic_plugins")]
    _libraries: Vec<libloading::Library>,
}

/// Internal entry for a registered plugin.
struct PluginEntry {
    /// The plugin instance.
    plugin: Box<dyn Plugin>,
    /// Current state.
    state: PluginState,
    /// Whether loaded from a dynamic library.
    is_dynamic: bool,
    /// Library path (for dynamic plugins).
    library_path: Option<PathBuf>,
    /// Load order position.
    load_order: usize,
}

impl PluginManager {
    /// Create a new plugin manager.
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            name_to_index: HashMap::new(),
            load_order: Vec::new(),
            loaded: false,
            #[cfg(feature = "dynamic_plugins")]
            _libraries: Vec::new(),
        }
    }

    /// Register a statically-linked plugin.
    ///
    /// Returns an error if a plugin with the same name is already registered.
    pub fn register_plugin(&mut self, plugin: Box<dyn Plugin>) -> Result<(), PluginError> {
        let name = plugin.name().to_owned();

        if self.name_to_index.contains_key(&name) {
            return Err(PluginError::AlreadyRegistered(name));
        }

        let index = self.plugins.len();
        self.name_to_index.insert(name.clone(), index);
        self.plugins.push(PluginEntry {
            plugin,
            state: PluginState::Registered,
            is_dynamic: false,
            library_path: None,
            load_order: 0,
        });

        log::info!("Plugin '{}' registered", name);
        Ok(())
    }

    /// Load a plugin from a dynamic library.
    ///
    /// The library must export a `create_plugin` function with C linkage.
    /// This is a placeholder on platforms that do not support dynamic loading.
    pub fn load_dynamic_plugin(
        &mut self,
        _path: &Path,
    ) -> Result<(), PluginError> {
        // In a production engine, this would use libloading:
        //
        //   let lib = unsafe { libloading::Library::new(path)? };
        //   let create: Symbol<extern "C" fn() -> *mut dyn Plugin> =
        //       unsafe { lib.get(b"create_plugin")? };
        //   let plugin = unsafe { Box::from_raw(create()) };
        //   self.register_plugin(plugin)?;
        //   self._libraries.push(lib);
        //
        // For now, return an error indicating dynamic loading is not compiled in.
        Err(PluginError::DynamicLoadNotSupported)
    }

    /// Get the number of registered plugins.
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Check if a plugin is registered.
    pub fn has_plugin(&self, name: &str) -> bool {
        self.name_to_index.contains_key(name)
    }

    /// Get information about a plugin.
    pub fn plugin_info(&self, name: &str) -> Option<PluginInfo> {
        let index = *self.name_to_index.get(name)?;
        let entry = &self.plugins[index];
        let plugin = &entry.plugin;

        Some(PluginInfo {
            name: plugin.name().to_owned(),
            version: plugin.version(),
            dependencies: plugin.dependencies().iter().map(|s| s.to_string()).collect(),
            state: entry.state,
            is_dynamic: entry.is_dynamic,
            library_path: entry.library_path.clone(),
            load_order: entry.load_order,
        })
    }

    /// Get a reference to a plugin by name, downcasted to the concrete type.
    pub fn get_plugin<T: Plugin + 'static>(&self, name: &str) -> Option<&T> {
        let index = *self.name_to_index.get(name)?;
        self.plugins[index].plugin.as_any().downcast_ref::<T>()
    }

    /// Get a mutable reference to a plugin by name, downcasted.
    pub fn get_plugin_mut<T: Plugin + 'static>(&mut self, name: &str) -> Option<&mut T> {
        let index = *self.name_to_index.get(name)?;
        self.plugins[index]
            .plugin
            .as_any_mut()
            .downcast_mut::<T>()
    }

    /// Get all plugin infos.
    pub fn all_plugins(&self) -> Vec<PluginInfo> {
        self.plugins
            .iter()
            .map(|entry| {
                let plugin = &entry.plugin;
                PluginInfo {
                    name: plugin.name().to_owned(),
                    version: plugin.version(),
                    dependencies: plugin
                        .dependencies()
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    state: entry.state,
                    is_dynamic: entry.is_dynamic,
                    library_path: entry.library_path.clone(),
                    load_order: entry.load_order,
                }
            })
            .collect()
    }

    // -- Dependency resolution ---------------------------------------------

    /// Compute the load order by topological sort on dependencies.
    ///
    /// Uses Kahn's algorithm for topological sorting.
    fn compute_load_order(&mut self) -> Result<(), PluginError> {
        let n = self.plugins.len();
        let mut in_degree: Vec<usize> = vec![0; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Build adjacency list.
        for (i, entry) in self.plugins.iter().enumerate() {
            for dep_name in entry.plugin.dependencies() {
                let dep_index = self.name_to_index.get(*dep_name).ok_or_else(|| {
                    PluginError::MissingDependency {
                        plugin: entry.plugin.name().to_owned(),
                        dependency: dep_name.to_string(),
                    }
                })?;
                adj[*dep_index].push(i);
                in_degree[i] += 1;
            }
        }

        // Kahn's algorithm.
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(idx) = queue.pop() {
            order.push(idx);
            for &next in &adj[idx] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push(next);
                }
            }
        }

        if order.len() != n {
            // Find a cycle.
            let in_cycle: Vec<String> = (0..n)
                .filter(|&i| in_degree[i] > 0)
                .map(|i| self.plugins[i].plugin.name().to_owned())
                .collect();
            return Err(PluginError::CyclicDependency(in_cycle));
        }

        // Assign load order.
        for (pos, &idx) in order.iter().enumerate() {
            self.plugins[idx].load_order = pos;
        }

        self.load_order = order;
        Ok(())
    }

    // -- Lifecycle ---------------------------------------------------------

    /// Load all registered plugins in dependency order.
    pub fn load_all(&mut self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        if self.loaded {
            return Err(PluginError::AlreadyLoaded);
        }

        self.compute_load_order()?;

        log::info!(
            "Loading {} plugins in order: [{}]",
            self.plugins.len(),
            self.load_order
                .iter()
                .map(|&i| self.plugins[i].plugin.name())
                .collect::<Vec<_>>()
                .join(", "),
        );

        for &idx in &self.load_order.clone() {
            let entry = &mut self.plugins[idx];
            let name = entry.plugin.name().to_owned();
            let version = entry.plugin.version();

            log::info!("Loading plugin '{}' v{}", name, version);
            entry.state = PluginState::Loading;

            entry.plugin.on_load(ctx);
            entry.state = PluginState::Active;

            ctx.log_event(format!("Plugin '{}' loaded", name));
        }

        self.loaded = true;
        Ok(())
    }

    /// Unload all plugins in reverse dependency order.
    pub fn unload_all(&mut self, ctx: &mut PluginContext) {
        if !self.loaded {
            return;
        }

        // Reverse order.
        let order: Vec<usize> = self.load_order.iter().copied().rev().collect();

        for idx in order {
            let entry = &mut self.plugins[idx];
            let name = entry.plugin.name().to_owned();

            log::info!("Unloading plugin '{}'", name);
            entry.state = PluginState::Unloading;

            entry.plugin.on_shutdown(ctx);
            entry.plugin.on_unload(ctx);
            entry.state = PluginState::Unloaded;

            ctx.log_event(format!("Plugin '{}' unloaded", name));
        }

        self.loaded = false;
    }

    /// Update all active plugins.
    pub fn update_all(&mut self, dt: f32, ctx: &mut PluginContext) {
        for &idx in &self.load_order {
            let entry = &mut self.plugins[idx];
            if entry.state == PluginState::Active {
                entry.plugin.on_update(dt, ctx);
            }
        }
    }

    /// Run fixed update on all active plugins.
    pub fn fixed_update_all(&mut self, fixed_dt: f32, ctx: &mut PluginContext) {
        for &idx in &self.load_order {
            let entry = &mut self.plugins[idx];
            if entry.state == PluginState::Active {
                entry.plugin.on_fixed_update(fixed_dt, ctx);
            }
        }
    }

    /// Run late update on all active plugins.
    pub fn late_update_all(&mut self, dt: f32, ctx: &mut PluginContext) {
        for &idx in &self.load_order {
            let entry = &mut self.plugins[idx];
            if entry.state == PluginState::Active {
                entry.plugin.on_late_update(dt, ctx);
            }
        }
    }

    /// Notify all plugins of a window resize.
    pub fn resize_all(&mut self, width: u32, height: u32, ctx: &mut PluginContext) {
        for &idx in &self.load_order {
            let entry = &mut self.plugins[idx];
            if entry.state == PluginState::Active {
                entry.plugin.on_resize(width, height, ctx);
            }
        }
    }

    // -- Hot-reload --------------------------------------------------------

    /// Reload a single plugin by name.
    ///
    /// This saves the plugin's state, unloads it, and reloads it (from
    /// the dynamic library if applicable). State is restored after reload.
    pub fn reload_plugin(
        &mut self,
        name: &str,
        ctx: &mut PluginContext,
    ) -> Result<(), PluginError> {
        let index = *self
            .name_to_index
            .get(name)
            .ok_or_else(|| PluginError::NotFound(name.to_owned()))?;

        let entry = &mut self.plugins[index];

        if !entry.plugin.supports_hot_reload() {
            return Err(PluginError::HotReloadNotSupported(name.to_owned()));
        }

        log::info!("Hot-reloading plugin '{}'", name);

        // Save state.
        let saved_state = entry.plugin.save_state();

        // Unload.
        entry.state = PluginState::Unloading;
        entry.plugin.on_unload(ctx);
        entry.state = PluginState::Unloaded;

        // In a real engine, we would re-compile and re-load the dynamic library
        // here. For now, we simply call on_load again.

        // Reload.
        entry.state = PluginState::Loading;
        entry.plugin.on_load(ctx);

        // Restore state.
        if let Some(state_data) = saved_state {
            entry.plugin.restore_state(&state_data);
        }

        entry.state = PluginState::Active;
        ctx.log_event(format!("Plugin '{}' hot-reloaded", name));

        log::info!("Plugin '{}' hot-reloaded successfully", name);
        Ok(())
    }

    /// Check whether all plugins have been loaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PluginError
// ---------------------------------------------------------------------------

/// Errors related to the plugin system.
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("plugin '{0}' is already registered")]
    AlreadyRegistered(String),

    #[error("plugins are already loaded")]
    AlreadyLoaded,

    #[error("plugin '{0}' not found")]
    NotFound(String),

    #[error("plugin '{plugin}' depends on '{dependency}' which is not registered")]
    MissingDependency {
        plugin: String,
        dependency: String,
    },

    #[error("cyclic dependency detected among plugins: {0:?}")]
    CyclicDependency(Vec<String>),

    #[error("plugin '{0}' does not support hot-reloading")]
    HotReloadNotSupported(String),

    #[error("dynamic plugin loading is not supported (compile with 'dynamic_plugins' feature)")]
    DynamicLoadNotSupported,

    #[error("version mismatch for plugin '{name}': got {got}, required {required}")]
    VersionMismatch {
        name: String,
        got: PluginVersion,
        required: PluginVersion,
    },

    #[error("failed to load dynamic library '{path}': {reason}")]
    LibraryLoadFailed {
        path: PathBuf,
        reason: String,
    },

    #[error("failed to find symbol '{symbol}' in library '{path}'")]
    SymbolNotFound {
        path: PathBuf,
        symbol: String,
    },

    #[error("{0}")]
    Other(String),
}

// ---------------------------------------------------------------------------
// PluginBuilder
// ---------------------------------------------------------------------------

/// Helper for creating simple plugins with closures (for testing / scripting).
pub struct ClosurePlugin {
    name: String,
    version: PluginVersion,
    deps: Vec<&'static str>,
    on_load_fn: Option<Box<dyn FnMut(&mut PluginContext) + Send + Sync>>,
    on_unload_fn: Option<Box<dyn FnMut(&mut PluginContext) + Send + Sync>>,
    on_update_fn: Option<Box<dyn FnMut(f32, &mut PluginContext) + Send + Sync>>,
}

impl ClosurePlugin {
    /// Create a new closure plugin.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: PluginVersion::new(0, 1, 0),
            deps: Vec::new(),
            on_load_fn: None,
            on_unload_fn: None,
            on_update_fn: None,
        }
    }

    /// Set the version.
    pub fn with_version(mut self, major: u32, minor: u32, patch: u32) -> Self {
        self.version = PluginVersion::new(major, minor, patch);
        self
    }

    /// Set the on_load callback.
    pub fn on_load<F: FnMut(&mut PluginContext) + Send + Sync + 'static>(
        mut self,
        f: F,
    ) -> Self {
        self.on_load_fn = Some(Box::new(f));
        self
    }

    /// Set the on_unload callback.
    pub fn on_unload<F: FnMut(&mut PluginContext) + Send + Sync + 'static>(
        mut self,
        f: F,
    ) -> Self {
        self.on_unload_fn = Some(Box::new(f));
        self
    }

    /// Set the on_update callback.
    pub fn on_update<F: FnMut(f32, &mut PluginContext) + Send + Sync + 'static>(
        mut self,
        f: F,
    ) -> Self {
        self.on_update_fn = Some(Box::new(f));
        self
    }
}

impl Plugin for ClosurePlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> PluginVersion {
        self.version
    }

    fn dependencies(&self) -> &[&str] {
        &self.deps
    }

    fn on_load(&mut self, ctx: &mut PluginContext) {
        if let Some(ref mut f) = self.on_load_fn {
            f(ctx);
        }
    }

    fn on_unload(&mut self, ctx: &mut PluginContext) {
        if let Some(ref mut f) = self.on_unload_fn {
            f(ctx);
        }
    }

    fn on_update(&mut self, dt: f32, ctx: &mut PluginContext) {
        if let Some(ref mut f) = self.on_update_fn {
            f(dt, ctx);
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPluginA;
    struct TestPluginB;

    impl Plugin for TestPluginA {
        fn name(&self) -> &str {
            "test_a"
        }
        fn version(&self) -> PluginVersion {
            PluginVersion::new(1, 0, 0)
        }
        fn on_load(&mut self, ctx: &mut PluginContext) {
            ctx.set_resource("a_loaded", Box::new(true));
        }
        fn on_unload(&mut self, ctx: &mut PluginContext) {
            ctx.remove_resource("a_loaded");
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl Plugin for TestPluginB {
        fn name(&self) -> &str {
            "test_b"
        }
        fn version(&self) -> PluginVersion {
            PluginVersion::new(1, 2, 0)
        }
        fn dependencies(&self) -> &[&str] {
            &["test_a"]
        }
        fn on_load(&mut self, ctx: &mut PluginContext) {
            // Should be able to access A's resource.
            let a_loaded = ctx.get_resource::<bool>("a_loaded").copied().unwrap_or(false);
            ctx.set_resource("b_checked_a", Box::new(a_loaded));
        }
        fn on_unload(&mut self, ctx: &mut PluginContext) {
            ctx.remove_resource("b_checked_a");
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn register_and_load() {
        let mut mgr = PluginManager::new();
        let mut ctx = PluginContext::new();

        mgr.register_plugin(Box::new(TestPluginA)).unwrap();
        mgr.register_plugin(Box::new(TestPluginB)).unwrap();

        mgr.load_all(&mut ctx).unwrap();

        assert!(mgr.is_loaded());
        assert_eq!(*ctx.get_resource::<bool>("a_loaded").unwrap(), true);
        assert_eq!(*ctx.get_resource::<bool>("b_checked_a").unwrap(), true);
    }

    #[test]
    fn dependency_order() {
        let mut mgr = PluginManager::new();
        let mut ctx = PluginContext::new();

        // Register B before A to test dependency ordering.
        mgr.register_plugin(Box::new(TestPluginB)).unwrap();
        mgr.register_plugin(Box::new(TestPluginA)).unwrap();

        mgr.load_all(&mut ctx).unwrap();

        // B depends on A, so A should have been loaded first.
        assert_eq!(*ctx.get_resource::<bool>("b_checked_a").unwrap(), true);
    }

    #[test]
    fn missing_dependency() {
        let mut mgr = PluginManager::new();
        let mut ctx = PluginContext::new();

        // Only register B (which depends on A).
        mgr.register_plugin(Box::new(TestPluginB)).unwrap();

        let result = mgr.load_all(&mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn unload_reverse_order() {
        let mut mgr = PluginManager::new();
        let mut ctx = PluginContext::new();

        mgr.register_plugin(Box::new(TestPluginA)).unwrap();
        mgr.register_plugin(Box::new(TestPluginB)).unwrap();
        mgr.load_all(&mut ctx).unwrap();

        mgr.unload_all(&mut ctx);

        assert!(!mgr.is_loaded());
        assert!(!ctx.has_resource("a_loaded"));
    }

    #[test]
    fn duplicate_registration_fails() {
        let mut mgr = PluginManager::new();
        mgr.register_plugin(Box::new(TestPluginA)).unwrap();
        let result = mgr.register_plugin(Box::new(TestPluginA));
        assert!(result.is_err());
    }

    #[test]
    fn closure_plugin() {
        let mut mgr = PluginManager::new();
        let mut ctx = PluginContext::new();

        let plugin = ClosurePlugin::new("closure_test")
            .with_version(1, 0, 0)
            .on_load(|ctx| {
                ctx.set_resource("closure_loaded", Box::new(42u32));
            });

        mgr.register_plugin(Box::new(plugin)).unwrap();
        mgr.load_all(&mut ctx).unwrap();

        assert_eq!(*ctx.get_resource::<u32>("closure_loaded").unwrap(), 42);
    }

    #[test]
    fn plugin_context_typed_resources() {
        let mut ctx = PluginContext::new();
        ctx.insert_typed(42u32);
        ctx.insert_typed("hello".to_owned());

        assert_eq!(*ctx.get_typed::<u32>().unwrap(), 42);
        assert_eq!(ctx.get_typed::<String>().unwrap(), "hello");
    }

    #[test]
    fn plugin_messaging() {
        let mut ctx = PluginContext::new();

        ctx.send_message("audio", "physics", "collision", "{}");
        ctx.broadcast("core", "shutdown", "");

        let physics_msgs = ctx.take_messages("physics");
        // Should get both the targeted message and the broadcast.
        assert_eq!(physics_msgs.len(), 2);
    }

    #[test]
    fn version_compatibility() {
        let v1 = PluginVersion::new(1, 2, 3);
        let v2 = PluginVersion::new(1, 2, 0);
        let v3 = PluginVersion::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v2));
        assert!(!v2.is_compatible_with(&v1)); // 1.2.0 < 1.2.3
        assert!(!v1.is_compatible_with(&v3)); // major mismatch
    }
}
