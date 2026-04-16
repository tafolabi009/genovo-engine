//! Deferred command queue for the Genovo ECS.
//!
//! During system execution it is often impossible (or unsafe) to mutate the
//! world directly — for example, spawning or despawning entities while
//! iterating them. The [`CommandQueue`] collects these mutations as deferred
//! commands that are flushed between system runs.
//!
//! # Usage
//!
//! ```ignore
//! let mut commands = CommandQueue::new();
//!
//! commands.spawn((Position { x: 0.0, y: 0.0 }, Velocity { dx: 1.0, dy: 0.0 }));
//! commands.despawn(stale_entity);
//! commands.add_component(entity, Marker);
//! commands.remove_component::<Health>(entity);
//! commands.insert_resource(DeltaTime(1.0 / 60.0));
//!
//! // Apply all queued commands.
//! commands.flush(&mut world);
//! ```

use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

// ---------------------------------------------------------------------------
// Command trait
// ---------------------------------------------------------------------------

/// A single, type-erased command that can be applied to a [`World`].
pub trait Command: Send + Sync + 'static {
    /// Apply this command to the world. Consumes `self`.
    fn apply(self: Box<Self>, world: &mut World);
}

// ---------------------------------------------------------------------------
// Built-in command types
// ---------------------------------------------------------------------------

/// Command to spawn an entity with a bundle of components.
struct SpawnCommand {
    /// Closures that insert components into the world for the spawned entity.
    inserts: Vec<Box<dyn FnOnce(&mut World, Entity) + Send + Sync>>,
}

impl Command for SpawnCommand {
    fn apply(self: Box<Self>, world: &mut World) {
        let entity = world.spawn_entity().build();
        for insert in self.inserts {
            insert(world, entity);
        }
    }
}

/// Command to despawn an entity.
struct DespawnCommand {
    entity: Entity,
}

impl Command for DespawnCommand {
    fn apply(self: Box<Self>, world: &mut World) {
        world.despawn(self.entity);
    }
}

/// Command to add a component to an entity.
struct AddComponentCommand<T: Component> {
    entity: Entity,
    component: T,
}

impl<T: Component> Command for AddComponentCommand<T> {
    fn apply(self: Box<Self>, world: &mut World) {
        world.add_component(self.entity, self.component);
    }
}

/// Command to remove a component from an entity.
struct RemoveComponentCommand<T: Component> {
    entity: Entity,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Component> Command for RemoveComponentCommand<T> {
    fn apply(self: Box<Self>, world: &mut World) {
        world.remove_component::<T>(self.entity);
    }
}

/// Command to insert a resource.
struct InsertResourceCommand<R: 'static + Send + Sync> {
    resource: R,
}

impl<R: 'static + Send + Sync> Command for InsertResourceCommand<R> {
    fn apply(self: Box<Self>, world: &mut World) {
        world.add_resource(self.resource);
    }
}

/// Command to remove a resource.
struct RemoveResourceCommand<R: 'static + Send + Sync> {
    _marker: std::marker::PhantomData<R>,
}

impl<R: 'static + Send + Sync> Command for RemoveResourceCommand<R> {
    fn apply(self: Box<Self>, world: &mut World) {
        world.remove_resource::<R>();
    }
}

/// Command that runs a custom closure.
struct ClosureCommand {
    f: Box<dyn FnOnce(&mut World) + Send + Sync>,
}

impl Command for ClosureCommand {
    fn apply(self: Box<Self>, world: &mut World) {
        (self.f)(world);
    }
}

// ---------------------------------------------------------------------------
// EntityCommands — builder for a specific entity
// ---------------------------------------------------------------------------

/// A builder that accumulates component additions/removals for a single entity.
/// Created by [`CommandQueue::entity`].
pub struct EntityCommands<'q> {
    entity: Entity,
    queue: &'q mut CommandQueue,
}

impl<'q> EntityCommands<'q> {
    /// Add a component to this entity.
    pub fn add<T: Component>(self, component: T) -> Self {
        self.queue.add_component(self.entity, component);
        self
    }

    /// Remove a component from this entity.
    pub fn remove<T: Component>(self) -> Self {
        self.queue.remove_component::<T>(self.entity);
        self
    }

    /// Despawn this entity.
    pub fn despawn(self) {
        self.queue.despawn(self.entity);
    }

    /// Return the entity handle.
    pub fn id(&self) -> Entity {
        self.entity
    }
}

// ---------------------------------------------------------------------------
// SpawnBuilder — builder for spawning a new entity
// ---------------------------------------------------------------------------

/// Builder for spawning an entity with multiple components via commands.
pub struct SpawnBuilder<'q> {
    queue: &'q mut CommandQueue,
    inserts: Vec<Box<dyn FnOnce(&mut World, Entity) + Send + Sync>>,
}

impl<'q> SpawnBuilder<'q> {
    /// Add a component to the entity being spawned.
    pub fn with<T: Component>(mut self, component: T) -> Self {
        self.inserts.push(Box::new(move |world, entity| {
            world.add_component(entity, component);
        }));
        self
    }

    /// Finalize the spawn command and add it to the queue.
    pub fn build(self) {
        self.queue
            .commands
            .push(Box::new(SpawnCommand { inserts: self.inserts }));
    }
}

// ---------------------------------------------------------------------------
// CommandQueue
// ---------------------------------------------------------------------------

/// A queue of deferred world mutations.
///
/// Commands are pushed during system execution and applied later via
/// [`flush`](CommandQueue::flush). This is the primary mechanism for safe
/// structural changes during iteration.
pub struct CommandQueue {
    commands: Vec<Box<dyn Command>>,
}

impl CommandQueue {
    /// Create a new, empty command queue.
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Number of queued commands.
    #[inline]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether there are no queued commands.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    // -- Entity commands ----------------------------------------------------

    /// Queue a spawn command that returns a `SpawnBuilder`.
    pub fn spawn(&mut self) -> SpawnBuilder<'_> {
        SpawnBuilder {
            queue: self,
            inserts: Vec::new(),
        }
    }

    /// Queue a spawn with a single component.
    pub fn spawn_with<T: Component>(&mut self, component: T) {
        self.commands.push(Box::new(SpawnCommand {
            inserts: vec![Box::new(move |world: &mut World, entity: Entity| {
                world.add_component(entity, component);
            })],
        }));
    }

    /// Queue a despawn command.
    pub fn despawn(&mut self, entity: Entity) {
        self.commands.push(Box::new(DespawnCommand { entity }));
    }

    /// Get an [`EntityCommands`] builder for an existing entity.
    pub fn entity(&mut self, entity: Entity) -> EntityCommands<'_> {
        EntityCommands {
            entity,
            queue: self,
        }
    }

    // -- Component commands -------------------------------------------------

    /// Queue adding a component to an entity.
    pub fn add_component<T: Component>(&mut self, entity: Entity, component: T) {
        self.commands.push(Box::new(AddComponentCommand {
            entity,
            component,
        }));
    }

    /// Queue removing a component from an entity.
    pub fn remove_component<T: Component>(&mut self, entity: Entity) {
        self.commands.push(Box::new(RemoveComponentCommand::<T> {
            entity,
            _marker: std::marker::PhantomData,
        }));
    }

    // -- Resource commands --------------------------------------------------

    /// Queue inserting a resource.
    pub fn insert_resource<R: 'static + Send + Sync>(&mut self, resource: R) {
        self.commands
            .push(Box::new(InsertResourceCommand { resource }));
    }

    /// Queue removing a resource.
    pub fn remove_resource<R: 'static + Send + Sync>(&mut self) {
        self.commands.push(Box::new(RemoveResourceCommand::<R> {
            _marker: std::marker::PhantomData,
        }));
    }

    // -- Custom commands ----------------------------------------------------

    /// Queue a custom closure command.
    pub fn add_command<F>(&mut self, f: F)
    where
        F: FnOnce(&mut World) + Send + Sync + 'static,
    {
        self.commands.push(Box::new(ClosureCommand {
            f: Box::new(f),
        }));
    }

    // -- Flush --------------------------------------------------------------

    /// Apply all queued commands to the world, in order. The queue is cleared.
    pub fn flush(&mut self, world: &mut World) {
        let commands: Vec<Box<dyn Command>> = self.commands.drain(..).collect();
        for cmd in commands {
            cmd.apply(world);
        }
    }

    /// Clear all queued commands without applying them.
    pub fn clear(&mut self) {
        self.commands.clear();
    }

    /// Merge another command queue into this one (appends).
    pub fn append(&mut self, other: &mut CommandQueue) {
        self.commands.append(&mut other.commands);
    }
}

impl Default for CommandQueue {
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

    #[derive(Debug, PartialEq, Clone)]
    struct Pos {
        x: f32,
        y: f32,
    }
    impl Component for Pos {}

    #[derive(Debug, PartialEq, Clone)]
    struct Vel {
        dx: f32,
        dy: f32,
    }
    impl Component for Vel {}

    #[derive(Debug, PartialEq, Clone)]
    struct Health(f32);
    impl Component for Health {}

    #[test]
    fn spawn_via_commands() {
        let mut world = World::new();
        let mut commands = CommandQueue::new();

        commands.spawn_with(Pos { x: 1.0, y: 2.0 });
        commands.spawn_with(Pos { x: 3.0, y: 4.0 });

        assert_eq!(world.entity_count(), 0);
        commands.flush(&mut world);
        assert_eq!(world.entity_count(), 2);
    }

    #[test]
    fn despawn_via_commands() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Pos { x: 1.0, y: 2.0 })
            .build();

        let mut commands = CommandQueue::new();
        commands.despawn(e);
        commands.flush(&mut world);

        assert!(!world.is_alive(e));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn add_component_via_commands() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        let mut commands = CommandQueue::new();
        commands.add_component(e, Health(100.0));
        commands.flush(&mut world);

        assert!(world.has_component::<Health>(e));
        assert_eq!(
            world.get_component::<Health>(e).map(|h| h.0),
            Some(100.0)
        );
    }

    #[test]
    fn remove_component_via_commands() {
        let mut world = World::new();
        let e = world.spawn_entity().with(Health(100.0)).build();

        let mut commands = CommandQueue::new();
        commands.remove_component::<Health>(e);
        commands.flush(&mut world);

        assert!(!world.has_component::<Health>(e));
    }

    #[test]
    fn insert_resource_via_commands() {
        let mut world = World::new();

        let mut commands = CommandQueue::new();
        commands.insert_resource(42.0_f64);
        commands.flush(&mut world);

        assert_eq!(world.get_resource::<f64>(), Some(&42.0));
    }

    #[test]
    fn remove_resource_via_commands() {
        let mut world = World::new();
        world.add_resource(42.0_f64);

        let mut commands = CommandQueue::new();
        commands.remove_resource::<f64>();
        commands.flush(&mut world);

        assert!(!world.has_resource::<f64>());
    }

    #[test]
    fn custom_command() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        let mut commands = CommandQueue::new();
        commands.add_command(move |world: &mut World| {
            world.add_component(e, Pos { x: 99.0, y: 99.0 });
        });
        commands.flush(&mut world);

        assert_eq!(
            world.get_component::<Pos>(e),
            Some(&Pos { x: 99.0, y: 99.0 })
        );
    }

    #[test]
    fn entity_commands_builder() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        let mut commands = CommandQueue::new();
        commands
            .entity(e)
            .add(Pos { x: 1.0, y: 2.0 })
            .add(Vel { dx: 3.0, dy: 4.0 });
        commands.flush(&mut world);

        assert!(world.has_component::<Pos>(e));
        assert!(world.has_component::<Vel>(e));
    }

    #[test]
    fn entity_commands_remove_then_despawn() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Pos { x: 1.0, y: 2.0 })
            .with(Health(100.0))
            .build();

        let mut commands = CommandQueue::new();
        commands.entity(e).remove::<Health>().despawn();
        commands.flush(&mut world);

        assert!(!world.is_alive(e));
    }

    #[test]
    fn spawn_builder() {
        let mut world = World::new();
        let mut commands = CommandQueue::new();

        commands
            .spawn()
            .with(Pos { x: 1.0, y: 2.0 })
            .with(Vel { dx: 3.0, dy: 4.0 })
            .build();

        commands.flush(&mut world);
        assert_eq!(world.entity_count(), 1);

        let count = world.query::<(&Pos, &Vel)>().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn flush_clears_queue() {
        let mut world = World::new();
        let mut commands = CommandQueue::new();

        commands.spawn_with(Pos { x: 1.0, y: 2.0 });
        assert_eq!(commands.len(), 1);

        commands.flush(&mut world);
        assert_eq!(commands.len(), 0);
        assert!(commands.is_empty());
    }

    #[test]
    fn append_merges_queues() {
        let mut world = World::new();

        let mut q1 = CommandQueue::new();
        q1.spawn_with(Pos { x: 1.0, y: 0.0 });

        let mut q2 = CommandQueue::new();
        q2.spawn_with(Pos { x: 2.0, y: 0.0 });

        q1.append(&mut q2);
        assert_eq!(q1.len(), 2);
        assert_eq!(q2.len(), 0);

        q1.flush(&mut world);
        assert_eq!(world.entity_count(), 2);
    }

    #[test]
    fn clear_discards_commands() {
        let mut commands = CommandQueue::new();
        commands.spawn_with(Pos { x: 1.0, y: 2.0 });
        commands.spawn_with(Pos { x: 3.0, y: 4.0 });
        assert_eq!(commands.len(), 2);

        commands.clear();
        assert!(commands.is_empty());
    }

    #[test]
    fn commands_execute_in_order() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        let mut commands = CommandQueue::new();
        // First add Health, then change it via a closure.
        commands.add_component(e, Health(50.0));
        commands.add_command(move |world: &mut World| {
            if let Some(h) = world.get_component_mut::<Health>(e) {
                h.0 = 100.0;
            }
        });

        commands.flush(&mut world);
        assert_eq!(
            world.get_component::<Health>(e).map(|h| h.0),
            Some(100.0)
        );
    }

    #[test]
    fn multiple_spawn_with_components() {
        let mut world = World::new();
        let mut commands = CommandQueue::new();

        for i in 0..10 {
            commands
                .spawn()
                .with(Pos {
                    x: i as f32,
                    y: 0.0,
                })
                .with(Health(i as f32 * 10.0))
                .build();
        }

        commands.flush(&mut world);
        assert_eq!(world.entity_count(), 10);

        let count = world.query::<(&Pos, &Health)>().count();
        assert_eq!(count, 10);
    }
}
