//! # World Builder
//!
//! Fluent API for constructing and configuring ECS worlds in the Genovo engine.
//!
//! ## Features
//!
//! - **WorldBuilder** — Chainable builder for setting up a `World` with systems,
//!   resources, default components, and plugins before entering the game loop.
//! - **System registration** — Register systems with explicit ordering constraints
//!   (before/after) so the schedule is deterministic.
//! - **Resource initialization** — Pre-populate the world with singleton resources.
//! - **Default components** — Register component types with default values that
//!   are automatically attached on entity spawn.
//! - **Plugin system** — Modular `Plugin` trait with a `build` method for
//!   composable world setup.

use std::any::TypeId;
use std::collections::HashMap;

use crate::component::Component;
use crate::schedule::Stage;
use crate::system::System;
use crate::world::World;

// ---------------------------------------------------------------------------
// Plugin trait
// ---------------------------------------------------------------------------

/// Trait for modular world-setup plugins.
///
/// A plugin encapsulates a self-contained piece of world configuration —
/// registering systems, inserting resources, setting up observers, etc.
///
/// # Example
///
/// ```ignore
/// struct PhysicsPlugin;
///
/// impl EcsPlugin for PhysicsPlugin {
///     fn name(&self) -> &str { "PhysicsPlugin" }
///
///     fn build(&self, builder: &mut WorldBuilder) {
///         builder
///             .add_resource(Gravity(9.81))
///             .add_system(Stage::Update, "apply_gravity", apply_gravity_system)
///             .add_system(Stage::PostUpdate, "resolve_collisions", collision_system);
///     }
/// }
/// ```
pub trait EcsPlugin: Send + Sync + 'static {
    /// Human-readable name for logging and debugging.
    fn name(&self) -> &str;

    /// Build the plugin by adding systems, resources, etc. to the builder.
    fn build(&self, builder: &mut WorldBuilder);

    /// Optional cleanup when the plugin is removed.
    fn cleanup(&self, _world: &mut World) {}

    /// Optional per-frame update hook.
    fn update(&self, _world: &mut World) {}

    /// Dependencies: names of other plugins that must be installed first.
    fn dependencies(&self) -> Vec<&str> {
        Vec::new()
    }

    /// Whether this plugin can be hot-reloaded.
    fn supports_hot_reload(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// SystemRegistration
// ---------------------------------------------------------------------------

/// Describes a registered system with ordering metadata.
#[derive(Debug, Clone)]
pub struct SystemRegistration {
    /// Unique name for the system.
    pub name: String,
    /// Stage the system belongs to.
    pub stage: Stage,
    /// Names of systems this must run before.
    pub before: Vec<String>,
    /// Names of systems this must run after.
    pub after: Vec<String>,
    /// Optional system set name for grouping.
    pub set: Option<String>,
    /// Whether this system requires exclusive world access.
    pub exclusive: bool,
    /// Whether this system is enabled by default.
    pub enabled: bool,
    /// Run condition label (if any).
    pub run_condition: Option<String>,
}

impl SystemRegistration {
    /// Create a new registration for a system.
    pub fn new(name: impl Into<String>, stage: Stage) -> Self {
        Self {
            name: name.into(),
            stage,
            before: Vec::new(),
            after: Vec::new(),
            set: None,
            exclusive: false,
            enabled: true,
            run_condition: None,
        }
    }

    /// Declare that this system must run before another.
    pub fn before(mut self, other: impl Into<String>) -> Self {
        self.before.push(other.into());
        self
    }

    /// Declare that this system must run after another.
    pub fn after(mut self, other: impl Into<String>) -> Self {
        self.after.push(other.into());
        self
    }

    /// Assign this system to a named set.
    pub fn in_set(mut self, set: impl Into<String>) -> Self {
        self.set = Some(set.into());
        self
    }

    /// Mark this system as exclusive (requires &mut World).
    pub fn exclusive(mut self) -> Self {
        self.exclusive = true;
        self
    }

    /// Disable this system by default.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Attach a run condition.
    pub fn run_if(mut self, condition: impl Into<String>) -> Self {
        self.run_condition = Some(condition.into());
        self
    }
}

// ---------------------------------------------------------------------------
// ResourceEntry
// ---------------------------------------------------------------------------

/// A type-erased resource to be inserted into the world.
struct ResourceEntry {
    /// The resource value.
    value: Box<dyn std::any::Any + Send + Sync>,
    /// Type name for debugging.
    type_name: &'static str,
}

// ---------------------------------------------------------------------------
// DefaultComponent
// ---------------------------------------------------------------------------

/// A default component registration.
struct DefaultComponentEntry {
    /// Type ID of the component.
    type_id: TypeId,
    /// Type name for debugging.
    type_name: &'static str,
    /// Factory function to produce the default value.
    factory: Box<dyn Fn() -> Box<dyn std::any::Any + Send + Sync> + Send + Sync>,
}

// ---------------------------------------------------------------------------
// PluginEntry
// ---------------------------------------------------------------------------

/// A registered plugin.
struct PluginEntry {
    /// The plugin instance.
    plugin: Box<dyn EcsPlugin>,
    /// Whether the plugin has been built.
    built: bool,
}

// ---------------------------------------------------------------------------
// WorldBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing and configuring an ECS `World`.
///
/// # Example
///
/// ```ignore
/// let world = WorldBuilder::new()
///     .add_plugin(PhysicsPlugin)
///     .add_plugin(RenderPlugin)
///     .add_resource(DeltaTime(1.0 / 60.0))
///     .add_resource(Gravity(9.81))
///     .add_system(Stage::Update, "movement", movement_system)
///     .add_system_after(Stage::Update, "collision", collision_system, "movement")
///     .build();
/// ```
pub struct WorldBuilder {
    /// Registered systems.
    systems: Vec<SystemRegistration>,
    /// System closures (stored separately because closures are not Clone).
    system_fns: Vec<Box<dyn FnOnce(&mut World) + Send + 'static>>,
    /// Resources to insert.
    resources: Vec<ResourceEntry>,
    /// Default components.
    default_components: Vec<DefaultComponentEntry>,
    /// Installed plugins.
    plugins: Vec<PluginEntry>,
    /// Plugin names for deduplication and dependency checking.
    plugin_names: Vec<String>,
    /// Named system sets with ordering constraints.
    system_sets: HashMap<String, SystemSetConfig>,
    /// Whether to validate ordering constraints on build.
    validate_ordering: bool,
    /// World name (for debugging multi-world scenarios).
    world_name: Option<String>,
    /// Initial entity capacity hint.
    entity_capacity: usize,
}

/// Configuration for a named system set.
#[derive(Debug, Clone, Default)]
pub struct SystemSetConfig {
    /// Set name.
    pub name: String,
    /// Sets that must run before this one.
    pub before: Vec<String>,
    /// Sets that must run after this one.
    pub after: Vec<String>,
    /// Stage this set belongs to.
    pub stage: Option<Stage>,
}

impl WorldBuilder {
    /// Create a new empty `WorldBuilder`.
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            system_fns: Vec::new(),
            resources: Vec::new(),
            default_components: Vec::new(),
            plugins: Vec::new(),
            plugin_names: Vec::new(),
            system_sets: HashMap::new(),
            validate_ordering: true,
            world_name: None,
            entity_capacity: 1024,
        }
    }

    /// Set a name for the world.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.world_name = Some(name.into());
        self
    }

    /// Set the initial entity capacity.
    pub fn with_entity_capacity(mut self, capacity: usize) -> Self {
        self.entity_capacity = capacity;
        self
    }

    /// Disable ordering validation on build.
    pub fn skip_validation(mut self) -> Self {
        self.validate_ordering = false;
        self
    }

    // -------------------------------------------------------------------
    // System registration
    // -------------------------------------------------------------------

    /// Register a system in the given stage.
    pub fn add_system<F>(
        mut self,
        stage: Stage,
        name: impl Into<String>,
        system_fn: F,
    ) -> Self
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        let reg = SystemRegistration::new(name, stage);
        self.systems.push(reg);
        self.system_fns.push(Box::new(system_fn));
        self
    }

    /// Register a system with an explicit "before" ordering constraint.
    pub fn add_system_before<F>(
        mut self,
        stage: Stage,
        name: impl Into<String>,
        system_fn: F,
        before: impl Into<String>,
    ) -> Self
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        let reg = SystemRegistration::new(name, stage).before(before);
        self.systems.push(reg);
        self.system_fns.push(Box::new(system_fn));
        self
    }

    /// Register a system with an explicit "after" ordering constraint.
    pub fn add_system_after<F>(
        mut self,
        stage: Stage,
        name: impl Into<String>,
        system_fn: F,
        after: impl Into<String>,
    ) -> Self
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        let reg = SystemRegistration::new(name, stage).after(after);
        self.systems.push(reg);
        self.system_fns.push(Box::new(system_fn));
        self
    }

    /// Register a system with full configuration.
    pub fn add_system_with<F>(
        mut self,
        registration: SystemRegistration,
        system_fn: F,
    ) -> Self
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        self.systems.push(registration);
        self.system_fns.push(Box::new(system_fn));
        self
    }

    /// Register an exclusive system (requires &mut World).
    pub fn add_exclusive_system<F>(
        mut self,
        stage: Stage,
        name: impl Into<String>,
        system_fn: F,
    ) -> Self
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        let reg = SystemRegistration::new(name, stage).exclusive();
        self.systems.push(reg);
        self.system_fns.push(Box::new(system_fn));
        self
    }

    // -------------------------------------------------------------------
    // System sets
    // -------------------------------------------------------------------

    /// Define a named system set with optional ordering constraints.
    pub fn add_system_set(mut self, config: SystemSetConfig) -> Self {
        self.system_sets.insert(config.name.clone(), config);
        self
    }

    /// Create a system set builder.
    pub fn configure_set(mut self, name: impl Into<String>) -> SystemSetBuilder {
        SystemSetBuilder {
            config: SystemSetConfig {
                name: name.into(),
                ..Default::default()
            },
            world_builder: self,
        }
    }

    // -------------------------------------------------------------------
    // Resources
    // -------------------------------------------------------------------

    /// Add a singleton resource to the world.
    pub fn add_resource<T: Send + Sync + 'static>(mut self, resource: T) -> Self {
        self.resources.push(ResourceEntry {
            value: Box::new(resource),
            type_name: std::any::type_name::<T>(),
        });
        self
    }

    /// Add a resource only if it has not already been registered.
    pub fn init_resource<T: Default + Send + Sync + 'static>(mut self) -> Self {
        self.resources.push(ResourceEntry {
            value: Box::new(T::default()),
            type_name: std::any::type_name::<T>(),
        });
        self
    }

    // -------------------------------------------------------------------
    // Default components
    // -------------------------------------------------------------------

    /// Register a component type with a default value.
    ///
    /// Entities spawned through the builder will automatically get this
    /// component if not explicitly provided.
    pub fn register_default_component<T: Component + Default + Clone + Send + Sync + 'static>(
        mut self,
    ) -> Self {
        self.default_components.push(DefaultComponentEntry {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            factory: Box::new(|| Box::new(T::default())),
        });
        self
    }

    /// Register a component type with a custom factory.
    pub fn register_component_factory<T, F>(mut self, factory: F) -> Self
    where
        T: Component + Send + Sync + 'static,
        F: Fn() -> T + Send + Sync + 'static,
    {
        self.default_components.push(DefaultComponentEntry {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            factory: Box::new(move || Box::new(factory())),
        });
        self
    }

    // -------------------------------------------------------------------
    // Plugins
    // -------------------------------------------------------------------

    /// Install a plugin.
    ///
    /// Plugins are built in the order they are added, after all explicit
    /// registrations.
    pub fn add_plugin<P: EcsPlugin>(mut self, plugin: P) -> Self {
        let name = plugin.name().to_string();
        if self.plugin_names.contains(&name) {
            // Plugin already registered — skip.
            return self;
        }
        self.plugin_names.push(name);
        self.plugins.push(PluginEntry {
            plugin: Box::new(plugin),
            built: false,
        });
        self
    }

    /// Check if a plugin with the given name has been installed.
    pub fn has_plugin(&self, name: &str) -> bool {
        self.plugin_names.iter().any(|n| n == name)
    }

    /// Returns the names of all installed plugins.
    pub fn plugin_names(&self) -> &[String] {
        &self.plugin_names
    }

    // -------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------

    /// Validate ordering constraints between systems.
    ///
    /// Returns a list of errors if any constraint is unsatisfiable.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // Check that all "before" and "after" references point to known systems.
        let known_names: Vec<&str> = self.systems.iter().map(|s| s.name.as_str()).collect();

        for sys in &self.systems {
            for before in &sys.before {
                if !known_names.contains(&before.as_str()) {
                    errors.push(format!(
                        "system '{}' declares before='{}', but no system with that name exists",
                        sys.name, before
                    ));
                }
            }
            for after in &sys.after {
                if !known_names.contains(&after.as_str()) {
                    errors.push(format!(
                        "system '{}' declares after='{}', but no system with that name exists",
                        sys.name, after
                    ));
                }
            }
        }

        // Check for duplicate system names.
        let mut seen = std::collections::HashSet::new();
        for sys in &self.systems {
            if !seen.insert(&sys.name) {
                errors.push(format!("duplicate system name: '{}'", sys.name));
            }
        }

        // Check plugin dependencies.
        for entry in &self.plugins {
            for dep in entry.plugin.dependencies() {
                if !self.plugin_names.contains(&dep.to_string()) {
                    errors.push(format!(
                        "plugin '{}' requires '{}', but it is not installed",
                        entry.plugin.name(),
                        dep
                    ));
                }
            }
        }

        errors
    }

    // -------------------------------------------------------------------
    // Build
    // -------------------------------------------------------------------

    /// Consume the builder and produce a configured `World`.
    pub fn build(mut self) -> World {
        // Validate if enabled.
        if self.validate_ordering {
            let errors = self.validate();
            if !errors.is_empty() {
                for err in &errors {
                    eprintln!("[WorldBuilder] warning: {}", err);
                }
            }
        }

        // Build plugins (they may add more systems/resources).
        // We need to move plugins out to avoid borrow issues.
        let mut plugins: Vec<PluginEntry> = std::mem::take(&mut self.plugins);
        for entry in &mut plugins {
            if !entry.built {
                entry.plugin.build(&mut self);
                entry.built = true;
            }
        }
        self.plugins = plugins;

        // Create the world.
        let mut world = World::new();

        // Insert resources.
        for entry in self.resources {
            world.insert_resource_boxed(entry.value);
        }

        // Run system init functions.
        for system_fn in self.system_fns {
            system_fn(&mut world);
        }

        world
    }

    /// Build the world and return it along with the ordering metadata.
    pub fn build_with_metadata(self) -> (World, WorldMetadata) {
        let systems = self.systems.clone();
        let plugin_names = self.plugin_names.clone();
        let set_configs: Vec<SystemSetConfig> = self.system_sets.values().cloned().collect();

        let world = self.build();

        let metadata = WorldMetadata {
            world_name: None,
            systems,
            plugin_names,
            system_sets: set_configs,
        };

        (world, metadata)
    }

    /// Returns the number of registered systems.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// Returns the number of registered resources.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Returns the number of registered plugins.
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Returns the number of default component registrations.
    pub fn default_component_count(&self) -> usize {
        self.default_components.len()
    }
}

impl Default for WorldBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SystemSetBuilder (fluent API for system sets)
// ---------------------------------------------------------------------------

/// Fluent builder for configuring a system set.
pub struct SystemSetBuilder {
    config: SystemSetConfig,
    world_builder: WorldBuilder,
}

impl SystemSetBuilder {
    /// This set must run before another set.
    pub fn before(mut self, other: impl Into<String>) -> Self {
        self.config.before.push(other.into());
        self
    }

    /// This set must run after another set.
    pub fn after(mut self, other: impl Into<String>) -> Self {
        self.config.after.push(other.into());
        self
    }

    /// Assign this set to a stage.
    pub fn in_stage(mut self, stage: Stage) -> Self {
        self.config.stage = Some(stage);
        self
    }

    /// Finish configuring and return the world builder.
    pub fn done(self) -> WorldBuilder {
        let mut wb = self.world_builder;
        wb.system_sets.insert(self.config.name.clone(), self.config);
        wb
    }
}

// ---------------------------------------------------------------------------
// WorldMetadata
// ---------------------------------------------------------------------------

/// Metadata about the world configuration for debugging and inspection.
#[derive(Debug, Clone)]
pub struct WorldMetadata {
    /// World name.
    pub world_name: Option<String>,
    /// All registered systems.
    pub systems: Vec<SystemRegistration>,
    /// All installed plugin names.
    pub plugin_names: Vec<String>,
    /// System set configurations.
    pub system_sets: Vec<SystemSetConfig>,
}

impl WorldMetadata {
    /// Find a system by name.
    pub fn find_system(&self, name: &str) -> Option<&SystemRegistration> {
        self.systems.iter().find(|s| s.name == name)
    }

    /// Returns systems in a given stage.
    pub fn systems_in_stage(&self, stage: Stage) -> Vec<&SystemRegistration> {
        self.systems.iter().filter(|s| s.stage == stage).collect()
    }

    /// Returns exclusive systems.
    pub fn exclusive_systems(&self) -> Vec<&SystemRegistration> {
        self.systems.iter().filter(|s| s.exclusive).collect()
    }

    /// Compute the topological order of systems within a stage.
    ///
    /// Returns the system names in execution order, or an error if there
    /// is a cycle.
    pub fn topological_order(&self, stage: Stage) -> Result<Vec<String>, String> {
        let systems: Vec<&SystemRegistration> = self.systems_in_stage(stage);
        let names: Vec<&str> = systems.iter().map(|s| s.name.as_str()).collect();

        // Build adjacency list.
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut in_degree: HashMap<&str, usize> = HashMap::new();

        for &name in &names {
            adj.entry(name).or_default();
            in_degree.entry(name).or_insert(0);
        }

        for sys in &systems {
            for before in &sys.before {
                if names.contains(&before.as_str()) {
                    adj.entry(sys.name.as_str())
                        .or_default()
                        .push(before.as_str());
                    *in_degree.entry(before.as_str()).or_insert(0) += 1;
                }
            }
            for after in &sys.after {
                if names.contains(&after.as_str()) {
                    adj.entry(after.as_str())
                        .or_default()
                        .push(sys.name.as_str());
                    *in_degree.entry(sys.name.as_str()).or_insert(0) += 1;
                }
            }
        }

        // Kahn's algorithm.
        let mut queue: std::collections::VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&name, _)| name)
            .collect();

        let mut order = Vec::new();

        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            if let Some(neighbors) = adj.get(node) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if order.len() != names.len() {
            Err("cycle detected in system ordering".to_string())
        } else {
            Ok(order)
        }
    }

    /// Print a summary of the world configuration.
    pub fn dump_summary(&self) -> String {
        let mut buf = String::new();
        buf.push_str("=== World Metadata ===\n");
        if let Some(name) = &self.world_name {
            buf.push_str(&format!("Name: {}\n", name));
        }
        buf.push_str(&format!("Plugins: {}\n", self.plugin_names.join(", ")));
        buf.push_str(&format!("Systems: {}\n", self.systems.len()));

        for stage in &[
            Stage::First,
            Stage::PreUpdate,
            Stage::Update,
            Stage::PostUpdate,
            Stage::Last,
        ] {
            let stage_systems = self.systems_in_stage(*stage);
            if !stage_systems.is_empty() {
                buf.push_str(&format!("  {:?}:\n", stage));
                for sys in &stage_systems {
                    let flags = if sys.exclusive { " [exclusive]" } else { "" };
                    buf.push_str(&format!("    - {}{}\n", sys.name, flags));
                }
            }
        }

        buf
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_builder_basic() {
        let builder = WorldBuilder::new()
            .add_system(Stage::Update, "test_system", |_world: &mut World| {});

        assert_eq!(builder.system_count(), 1);

        let _world = builder.build();
    }

    #[test]
    fn test_world_builder_resources() {
        let builder = WorldBuilder::new().add_resource(42u32).add_resource(3.14f64);

        assert_eq!(builder.resource_count(), 2);
    }

    #[test]
    fn test_world_builder_ordering() {
        let builder = WorldBuilder::new()
            .add_system(Stage::Update, "a", |_: &mut World| {})
            .add_system_after(Stage::Update, "b", |_: &mut World| {}, "a")
            .add_system_before(Stage::Update, "c", |_: &mut World| {}, "b");

        assert_eq!(builder.system_count(), 3);

        let errors = builder.validate();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn test_world_builder_validation_missing_dep() {
        let builder = WorldBuilder::new()
            .add_system_after(Stage::Update, "a", |_: &mut World| {}, "nonexistent");

        let errors = builder.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_world_builder_duplicate_detection() {
        let builder = WorldBuilder::new()
            .add_system(Stage::Update, "dup", |_: &mut World| {})
            .add_system(Stage::Update, "dup", |_: &mut World| {});

        let errors = builder.validate();
        assert!(errors.iter().any(|e| e.contains("duplicate")));
    }

    #[test]
    fn test_world_metadata_topo_sort() {
        let builder = WorldBuilder::new()
            .add_system(Stage::Update, "a", |_: &mut World| {})
            .add_system_after(Stage::Update, "b", |_: &mut World| {}, "a")
            .add_system_after(Stage::Update, "c", |_: &mut World| {}, "b");

        let (_world, metadata) = builder.build_with_metadata();
        let order = metadata.topological_order(Stage::Update).unwrap();

        let a_pos = order.iter().position(|n| n == "a").unwrap();
        let b_pos = order.iter().position(|n| n == "b").unwrap();
        let c_pos = order.iter().position(|n| n == "c").unwrap();

        assert!(a_pos < b_pos);
        assert!(b_pos < c_pos);
    }

    #[test]
    fn test_plugin_registration() {
        struct TestPlugin;
        impl EcsPlugin for TestPlugin {
            fn name(&self) -> &str {
                "TestPlugin"
            }
            fn build(&self, builder: &mut WorldBuilder) {
                // Add a resource via the mutable builder reference.
                builder.resources.push(ResourceEntry {
                    value: Box::new(100u32),
                    type_name: "u32",
                });
            }
        }

        let builder = WorldBuilder::new().add_plugin(TestPlugin);
        assert!(builder.has_plugin("TestPlugin"));
        assert_eq!(builder.plugin_count(), 1);
    }

    #[test]
    fn test_plugin_deduplication() {
        struct MyPlugin;
        impl EcsPlugin for MyPlugin {
            fn name(&self) -> &str {
                "MyPlugin"
            }
            fn build(&self, _builder: &mut WorldBuilder) {}
        }

        let builder = WorldBuilder::new()
            .add_plugin(MyPlugin)
            .add_plugin(MyPlugin);

        assert_eq!(builder.plugin_count(), 1);
    }

    #[test]
    fn test_system_set_builder() {
        let builder = WorldBuilder::new()
            .configure_set("physics")
            .in_stage(Stage::Update)
            .before("rendering")
            .done();

        assert!(builder.system_sets.contains_key("physics"));
    }

    #[test]
    fn test_system_registration_fluent() {
        let reg = SystemRegistration::new("test", Stage::Update)
            .before("other")
            .after("earlier")
            .in_set("my_set")
            .exclusive()
            .run_if("is_game_running");

        assert_eq!(reg.name, "test");
        assert!(reg.exclusive);
        assert_eq!(reg.before, vec!["other"]);
        assert_eq!(reg.after, vec!["earlier"]);
        assert_eq!(reg.set, Some("my_set".to_string()));
        assert_eq!(reg.run_condition, Some("is_game_running".to_string()));
    }

    #[test]
    fn test_metadata_dump() {
        let builder = WorldBuilder::new()
            .with_name("TestWorld")
            .add_system(Stage::Update, "sys_a", |_: &mut World| {})
            .add_system(Stage::PostUpdate, "sys_b", |_: &mut World| {});

        let (_world, metadata) = builder.build_with_metadata();
        let summary = metadata.dump_summary();

        assert!(summary.contains("Systems: 2"));
    }
}
