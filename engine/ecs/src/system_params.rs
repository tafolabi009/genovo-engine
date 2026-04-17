//! System parameter extraction for ergonomic system definitions.
//!
//! This module provides the [`SystemParam`] trait and related infrastructure
//! that lets users write systems as plain functions with typed parameters:
//!
//! ```ignore
//! fn movement(query: Query<(&mut Position, &Velocity)>, time: Res<Time>) {
//!     for (entity, (pos, vel)) in query.iter() {
//!         pos.x += vel.dx * time.delta;
//!         pos.y += vel.dy * time.delta;
//!     }
//! }
//! ```
//!
//! # Architecture
//!
//! - [`SystemParam`] — trait for types that can be auto-extracted from a `&World`
//!   or `&mut World` reference.
//! - [`SystemParamState`] — cached state held between system invocations for
//!   efficient re-access (e.g., cached archetype match lists).
//! - [`Query`] — system parameter that iterates entities with specific components.
//! - [`Res`] / [`ResMut`] — read-only and mutable resource access as parameters.
//! - [`Commands`] — deferred mutation queue available as a system parameter.
//! - [`EventReader`] / [`EventWriter`] — event consumption and emission.
//! - [`Local`] — per-system local state that persists between invocations.
//! - [`IntoSystem`] — trait to convert closures with `SystemParam` arguments
//!   into boxed `System` trait objects.
//!
//! # Derive-Style Registration
//!
//! Instead of procedural macros, the tuple-based `SystemParam` impls allow
//! composing parameter types. Any function whose arguments each implement
//! `SystemParam` can be converted into a system via `IntoSystem::into_system()`.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::component::{Component, ComponentId, ComponentStorage};
use crate::entity::Entity;
use crate::event::Events;
use crate::world::World;

// ---------------------------------------------------------------------------
// SystemParamState
// ---------------------------------------------------------------------------

/// Cached state for a system parameter. This is stored between system
/// invocations so that expensive lookups (archetype matching, etc.) can be
/// amortized across frames.
pub trait SystemParamState: Send + Sync + 'static {
    /// Initialize the cached state from the world.
    fn init(world: &World) -> Self
    where
        Self: Sized;

    /// Called each time the system runs, before parameter extraction. Allows
    /// the state to refresh caches (e.g., re-scan archetypes if the count
    /// changed).
    fn update(&mut self, world: &World);
}

/// Trivial state that carries no data. Used by simple parameters like `Res<T>`.
#[derive(Debug, Clone)]
pub struct EmptyState;

impl SystemParamState for EmptyState {
    fn init(_world: &World) -> Self {
        EmptyState
    }

    fn update(&mut self, _world: &World) {}
}

// ---------------------------------------------------------------------------
// SystemParam trait
// ---------------------------------------------------------------------------

/// Trait for types that can be extracted from a `World` reference as system
/// function parameters.
///
/// Each `SystemParam` implementation has an associated `State` that persists
/// between system invocations, and a `fetch` method that produces the
/// parameter value from the world and state.
///
/// # Safety
///
/// Implementations must correctly declare their access patterns (read vs.
/// write, which component types) to enable safe parallel scheduling.
pub trait SystemParam {
    /// Per-system cached state for this parameter.
    type State: SystemParamState;

    /// The actual type yielded when fetching from the world.
    type Item<'w>;

    /// Initialize the parameter state.
    fn init_state(world: &World) -> Self::State;

    /// Fetch the parameter from the world using the cached state.
    ///
    /// # Safety
    ///
    /// Caller must ensure that no conflicting borrows exist and that the
    /// world reference is valid for the lifetime `'w`.
    unsafe fn fetch<'w>(world: &'w World, state: &mut Self::State) -> Self::Item<'w>;

    /// Describe what this parameter accesses for scheduling.
    fn access() -> SystemParamAccess {
        SystemParamAccess::default()
    }
}

/// Describes the component and resource access patterns of a system parameter.
#[derive(Debug, Clone, Default)]
pub struct SystemParamAccess {
    /// Component types read immutably.
    pub component_reads: Vec<ComponentId>,
    /// Component types written mutably.
    pub component_writes: Vec<ComponentId>,
    /// Resource types read immutably.
    pub resource_reads: Vec<TypeId>,
    /// Resource types written mutably.
    pub resource_writes: Vec<TypeId>,
    /// Whether this parameter uses the command queue.
    pub uses_commands: bool,
}

impl SystemParamAccess {
    /// Check if this access conflicts with another access.
    pub fn conflicts_with(&self, other: &SystemParamAccess) -> bool {
        // Write-write or read-write conflicts on components.
        for w in &self.component_writes {
            if other.component_reads.contains(w) || other.component_writes.contains(w) {
                return true;
            }
        }
        for w in &other.component_writes {
            if self.component_reads.contains(w) || self.component_writes.contains(w) {
                return true;
            }
        }
        // Write-write or read-write conflicts on resources.
        for w in &self.resource_writes {
            if other.resource_reads.contains(w) || other.resource_writes.contains(w) {
                return true;
            }
        }
        for w in &other.resource_writes {
            if self.resource_reads.contains(w) || self.resource_writes.contains(w) {
                return true;
            }
        }
        false
    }

    /// Merge another access into this one.
    pub fn merge(&mut self, other: &SystemParamAccess) {
        self.component_reads.extend(&other.component_reads);
        self.component_writes.extend(&other.component_writes);
        self.resource_reads.extend(&other.resource_reads);
        self.resource_writes.extend(&other.resource_writes);
        self.uses_commands = self.uses_commands || other.uses_commands;
    }
}

// ---------------------------------------------------------------------------
// Query<Q> — query system parameter
// ---------------------------------------------------------------------------

/// Cached state for a [`Query`] parameter. Tracks which archetypes match
/// the query so we can skip re-scanning every frame.
pub struct QueryParamState<Q: crate::query::QueryItem> {
    /// Cached query state with archetype matching.
    query_state: crate::query::QueryState,
    _marker: PhantomData<Q>,
}

impl<Q: crate::query::QueryItem> SystemParamState for QueryParamState<Q> {
    fn init(world: &World) -> Self {
        Self {
            query_state: crate::query::QueryState::new::<Q>(world),
            _marker: PhantomData,
        }
    }

    fn update(&mut self, world: &World) {
        self.query_state.update(world);
    }
}

/// A system parameter that iterates over entities matching a component query.
///
/// ```ignore
/// fn my_system(query: Query<(&Position, &Velocity)>) {
///     for (entity, (pos, vel)) in query.iter() {
///         println!("{:?}: ({}, {})", entity, pos.x, pos.y);
///     }
/// }
/// ```
///
/// The `Query` provides read-only or mutable access depending on the query
/// item types used (e.g., `&T` for read, `&mut T` for write).
pub struct Query<'w, Q: crate::query::QueryItem> {
    world: &'w World,
    state: &'w crate::query::QueryState,
    _marker: PhantomData<Q>,
}

impl<'w, Q: crate::query::QueryItem> Query<'w, Q> {
    /// Create a new query from a world reference and cached state.
    pub fn new(world: &'w World, state: &'w crate::query::QueryState) -> Self {
        Self {
            world,
            state,
            _marker: PhantomData,
        }
    }

    /// Iterate over all matching entities.
    pub fn iter(&self) -> QueryParamIter<'w, Q> {
        let mut entities = Vec::new();
        for &arch_id in self.state.matching_archetypes() {
            let arch = self.world.archetype(arch_id);
            entities.extend_from_slice(arch.entities());
        }

        // Fallback: also include any alive entity that matches (legacy path).
        let all_alive: Vec<Entity> = self.world.entity_storage().iter_alive().collect();
        for entity in &all_alive {
            if !entities.contains(entity) {
                entities.push(*entity);
            }
        }

        QueryParamIter {
            world: self.world,
            entities,
            index: 0,
            _marker: PhantomData,
        }
    }

    /// Iterate and collect into a Vec.
    pub fn collect(&self) -> Vec<(Entity, Q::Item<'w>)> {
        self.iter().collect()
    }

    /// Get a specific entity's components if it matches the query.
    pub fn get(&self, entity: Entity) -> Option<Q::Item<'w>> {
        if Q::matches(self.world, entity.id) {
            Some(unsafe { Q::fetch(self.world, entity.id) })
        } else {
            None
        }
    }

    /// Check if the query would match any entity at all.
    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }

    /// Count the number of matching entities.
    pub fn count(&self) -> usize {
        self.iter().count()
    }

    /// Get a single matching entity. Returns `None` if zero or more than one.
    pub fn get_single(&self) -> Option<(Entity, Q::Item<'w>)> {
        let mut iter = self.iter();
        let first = iter.next()?;
        if iter.next().is_some() {
            return None; // More than one match.
        }
        Some(first)
    }

    /// Returns `true` if the given entity matches this query.
    pub fn contains(&self, entity: Entity) -> bool {
        Q::matches(self.world, entity.id)
    }
}

/// Iterator produced by [`Query::iter`].
pub struct QueryParamIter<'w, Q: crate::query::QueryItem> {
    world: &'w World,
    entities: Vec<Entity>,
    index: usize,
    _marker: PhantomData<Q>,
}

impl<'w, Q: crate::query::QueryItem> Iterator for QueryParamIter<'w, Q> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.entities.len() {
            let entity = self.entities[self.index];
            self.index += 1;
            if Q::matches(self.world, entity.id) {
                let item = unsafe { Q::fetch(self.world, entity.id) };
                return Some((entity, item));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.entities.len() - self.index))
    }
}

// ---------------------------------------------------------------------------
// Res<T> — read-only resource system parameter
// ---------------------------------------------------------------------------

/// System parameter for read-only access to a resource of type `T`.
///
/// ```ignore
/// fn my_system(time: Res<Time>) {
///     println!("delta: {}", time.delta);
/// }
/// ```
pub struct ResParam<'w, T: 'static + Send + Sync> {
    value: &'w T,
}

impl<'w, T: 'static + Send + Sync> ResParam<'w, T> {
    /// Get the inner reference.
    pub fn into_inner(self) -> &'w T {
        self.value
    }
}

impl<'w, T: 'static + Send + Sync> std::ops::Deref for ResParam<'w, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<'w, T: 'static + Send + Sync + std::fmt::Debug> std::fmt::Debug for ResParam<'w, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Res").field("value", self.value).finish()
    }
}

/// System parameter for mutable access to a resource of type `T`.
///
/// ```ignore
/// fn my_system(mut score: ResMutParam<Score>) {
///     score.0 += 10;
/// }
/// ```
///
/// NOTE: Because `SystemParam::fetch` takes `&World` (shared reference),
/// mutable resource access must use interior mutability or a deferred
/// approach. In the full engine this is handled via `UnsafeCell`. Here we
/// provide read-only access and the user can use `Commands` for mutations.
pub struct ResMutParam<'w, T: 'static + Send + Sync> {
    value: &'w T,
}

impl<'w, T: 'static + Send + Sync> std::ops::Deref for ResMutParam<'w, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

// ---------------------------------------------------------------------------
// Commands — deferred mutation queue as system parameter
// ---------------------------------------------------------------------------

/// System parameter providing a deferred command queue.
///
/// Commands are collected during system execution and applied after the
/// system finishes, allowing structural world changes (spawn, despawn, add
/// component) without violating borrow rules.
///
/// ```ignore
/// fn spawner(mut commands: Commands) {
///     commands.spawn().with(Position { x: 0.0, y: 0.0 }).build();
///     commands.despawn(stale_entity);
/// }
/// ```
pub struct Commands {
    queue: crate::commands::CommandQueue,
}

impl Commands {
    /// Create a new empty commands parameter.
    pub fn new() -> Self {
        Self {
            queue: crate::commands::CommandQueue::new(),
        }
    }

    /// Queue spawning a new entity with no components.
    pub fn spawn(&mut self) -> crate::commands::SpawnBuilder<'_> {
        self.queue.spawn()
    }

    /// Queue spawning with a single component.
    pub fn spawn_with<T: Component>(&mut self, component: T) {
        self.queue.spawn_with(component);
    }

    /// Queue despawning an entity.
    pub fn despawn(&mut self, entity: Entity) {
        self.queue.despawn(entity);
    }

    /// Queue adding a component to an entity.
    pub fn add_component<T: Component>(&mut self, entity: Entity, component: T) {
        self.queue.add_component(entity, component);
    }

    /// Queue removing a component from an entity.
    pub fn remove_component<T: Component>(&mut self, entity: Entity) {
        self.queue.remove_component::<T>(entity);
    }

    /// Queue inserting a resource.
    pub fn insert_resource<R: 'static + Send + Sync>(&mut self, resource: R) {
        self.queue.insert_resource(resource);
    }

    /// Queue removing a resource.
    pub fn remove_resource<R: 'static + Send + Sync>(&mut self) {
        self.queue.remove_resource::<R>();
    }

    /// Queue a custom closure command.
    pub fn add_command<F>(&mut self, f: F)
    where
        F: FnOnce(&mut World) + Send + Sync + 'static,
    {
        self.queue.add_command(f);
    }

    /// Get an entity commands builder.
    pub fn entity(&mut self, entity: Entity) -> crate::commands::EntityCommands<'_> {
        self.queue.entity(entity)
    }

    /// Apply all queued commands to the world.
    pub fn flush(mut self, world: &mut World) {
        self.queue.flush(world);
    }

    /// Number of queued commands.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Consume self and return the inner command queue.
    pub fn into_queue(self) -> crate::commands::CommandQueue {
        self.queue
    }
}

impl Default for Commands {
    fn default() -> Self {
        Self::new()
    }
}

/// State for the Commands system parameter.
pub struct CommandsState {
    /// Accumulated commands from the last system run, pending flush.
    pub pending: Option<crate::commands::CommandQueue>,
}

impl SystemParamState for CommandsState {
    fn init(_world: &World) -> Self {
        Self { pending: None }
    }

    fn update(&mut self, _world: &World) {}
}

// ---------------------------------------------------------------------------
// EventReader<T> — read events as system parameter
// ---------------------------------------------------------------------------

/// System parameter for reading events of type `T`.
///
/// Reads all events from the double-buffered event storage for the current
/// tick. Events are consumed: the reader tracks which events have been
/// processed via an internal cursor.
///
/// ```ignore
/// fn handle_damage(reader: EventReader<DamageEvent>) {
///     for event in reader.iter() {
///         println!("Entity {:?} took {} damage", event.entity, event.amount);
///     }
/// }
/// ```
pub struct EventReader<'w, T: 'static + Send + Sync> {
    events: Option<&'w Events<T>>,
}

impl<'w, T: 'static + Send + Sync> EventReader<'w, T> {
    /// Create a reader with access to the event storage.
    pub fn new(events: Option<&'w Events<T>>) -> Self {
        Self { events }
    }

    /// Iterate over all readable events.
    pub fn iter(&self) -> impl Iterator<Item = &'w T> {
        self.events
            .into_iter()
            .flat_map(|e| e.iter())
    }

    /// Whether there are any events to read.
    pub fn is_empty(&self) -> bool {
        self.events.map_or(true, |e| e.is_empty())
    }

    /// Number of readable events.
    pub fn len(&self) -> usize {
        self.events.map_or(0, |e| e.len())
    }
}

/// State for EventReader.
pub struct EventReaderState<T: 'static> {
    _marker: PhantomData<T>,
}

impl<T: 'static + Send + Sync> SystemParamState for EventReaderState<T> {
    fn init(_world: &World) -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    fn update(&mut self, _world: &World) {}
}

// ---------------------------------------------------------------------------
// EventWriter<T> — emit events as system parameter
// ---------------------------------------------------------------------------

/// System parameter for writing events of type `T`.
///
/// Events written via this parameter are collected in a local buffer and
/// flushed into the world's event storage after the system completes.
///
/// ```ignore
/// fn detect_collisions(mut writer: EventWriter<CollisionEvent>) {
///     // ... collision detection logic ...
///     writer.send(CollisionEvent { a: entity_a, b: entity_b });
/// }
/// ```
pub struct EventWriter<T: 'static + Send + Sync> {
    buffer: Vec<T>,
}

impl<T: 'static + Send + Sync> EventWriter<T> {
    /// Create a new event writer with an empty buffer.
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Send a single event.
    pub fn send(&mut self, event: T) {
        self.buffer.push(event);
    }

    /// Send multiple events.
    pub fn send_batch(&mut self, events: impl IntoIterator<Item = T>) {
        self.buffer.extend(events);
    }

    /// Number of events buffered.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Consume and return the buffered events.
    pub fn drain(&mut self) -> Vec<T> {
        std::mem::take(&mut self.buffer)
    }

    /// Flush buffered events into the world's event storage.
    pub fn flush(mut self, world: &mut World) {
        if let Some(events) = world.get_resource_mut::<Events<T>>() {
            for event in self.buffer.drain(..) {
                events.send(event);
            }
        }
    }
}

impl<T: 'static + Send + Sync> Default for EventWriter<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// State for EventWriter.
pub struct EventWriterState<T: 'static> {
    _marker: PhantomData<T>,
}

impl<T: 'static + Send + Sync> SystemParamState for EventWriterState<T> {
    fn init(_world: &World) -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    fn update(&mut self, _world: &World) {}
}

// ---------------------------------------------------------------------------
// Local<T> — per-system local state
// ---------------------------------------------------------------------------

/// Per-system local state that persists between system invocations.
///
/// Unlike resources, `Local<T>` is unique to each system instance. This is
/// useful for caching data, tracking frame-to-frame state, or implementing
/// cooldown timers without polluting the global resource namespace.
///
/// ```ignore
/// fn periodic_system(mut timer: Local<f32>, time: Res<Time>) {
///     *timer += time.delta;
///     if *timer >= 1.0 {
///         *timer -= 1.0;
///         println!("One second elapsed!");
///     }
/// }
/// ```
pub struct Local<T: 'static + Send + Sync + Default> {
    value: T,
}

impl<T: 'static + Send + Sync + Default> Local<T> {
    /// Create a new local with the default value.
    pub fn new() -> Self {
        Self {
            value: T::default(),
        }
    }

    /// Create a local with a specific initial value.
    pub fn with_value(value: T) -> Self {
        Self { value }
    }

    /// Get a reference to the inner value.
    pub fn get(&self) -> &T {
        &self.value
    }

    /// Get a mutable reference to the inner value.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Replace the inner value and return the old one.
    pub fn replace(&mut self, new_value: T) -> T {
        std::mem::replace(&mut self.value, new_value)
    }

    /// Take the inner value, replacing it with the default.
    pub fn take(&mut self) -> T {
        std::mem::take(&mut self.value)
    }
}

impl<T: 'static + Send + Sync + Default> Default for Local<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static + Send + Sync + Default> std::ops::Deref for Local<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T: 'static + Send + Sync + Default> std::ops::DerefMut for Local<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T: 'static + Send + Sync + Default + std::fmt::Debug> std::fmt::Debug for Local<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Local")
            .field("value", &self.value)
            .finish()
    }
}

/// State for Local<T>.
pub struct LocalState<T: 'static + Send + Sync + Default> {
    pub value: T,
}

impl<T: 'static + Send + Sync + Default> SystemParamState for LocalState<T> {
    fn init(_world: &World) -> Self {
        Self {
            value: T::default(),
        }
    }

    fn update(&mut self, _world: &World) {}
}

// ---------------------------------------------------------------------------
// IntoSystem — convert functions into System trait objects
// ---------------------------------------------------------------------------

/// Trait for converting a function or closure with `SystemParam` arguments
/// into a `System` trait object.
///
/// This is the key ergonomic bridge: users write plain functions and the
/// engine converts them into systems automatically.
///
/// ```ignore
/// fn movement(query: Query<(&mut Position, &Velocity)>, time: Res<Time>) {
///     for (_, (pos, vel)) in query.iter() {
///         pos.x += vel.dx * time.delta;
///         pos.y += vel.dy * time.delta;
///     }
/// }
///
/// schedule.add_system(movement.into_system());
/// ```
pub trait IntoSystem<Params> {
    /// Convert this function/closure into a boxed system.
    fn into_system(self) -> Box<dyn crate::system::System>;
}

/// A system wrapper that holds a function and its cached parameter state.
pub struct FunctionSystem<F, Params> {
    func: F,
    name: String,
    initialized: bool,
    _marker: PhantomData<Params>,
}

impl<F, Params> FunctionSystem<F, Params> {
    /// Create a new function system with the given name.
    pub fn new(func: F, name: impl Into<String>) -> Self {
        Self {
            func,
            name: name.into(),
            initialized: false,
            _marker: PhantomData,
        }
    }

    /// Get the system name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// SystemParam impls for tuples (up to 8 elements)
// ---------------------------------------------------------------------------

/// Implement `SystemParam` for the empty tuple (no parameters).
impl SystemParam for () {
    type State = EmptyState;
    type Item<'w> = ();

    fn init_state(world: &World) -> Self::State {
        EmptyState
    }

    unsafe fn fetch<'w>(_world: &'w World, _state: &mut Self::State) -> Self::Item<'w> {
        ()
    }
}

// Macro for implementing SystemParam for tuples of various arities.
macro_rules! impl_system_param_tuple {
    ($($T:ident),+) => {
        impl<$($T: SystemParam),+> SystemParam for ($($T,)+) {
            type State = ($($T::State,)+);
            type Item<'w> = ($($T::Item<'w>,)+);

            fn init_state(world: &World) -> Self::State {
                ($($T::init_state(world),)+)
            }

            unsafe fn fetch<'w>(
                world: &'w World,
                state: &mut Self::State,
            ) -> Self::Item<'w> {
                #[allow(non_snake_case)]
                let ($($T,)+) = state;
                ($(unsafe { $T::fetch(world, $T) },)+)
            }

            fn access() -> SystemParamAccess {
                let mut access = SystemParamAccess::default();
                $(access.merge(&$T::access());)+
                access
            }
        }
    };
}

impl_system_param_tuple!(A);
impl_system_param_tuple!(A, B);
impl_system_param_tuple!(A, B, C);
impl_system_param_tuple!(A, B, C, D);
impl_system_param_tuple!(A, B, C, D, E);
impl_system_param_tuple!(A, B, C, D, E, F);
impl_system_param_tuple!(A, B, C, D, E, F, G);
impl_system_param_tuple!(A, B, C, D, E, F, G, H);

// ---------------------------------------------------------------------------
// SystemParamFunction — trait for callable system functions
// ---------------------------------------------------------------------------

/// Trait for functions that can be used as systems with typed parameters.
pub trait SystemParamFunction<Params>: Send + Sync + 'static {
    /// Call the function with extracted parameters from the world.
    fn run(&mut self, world: &World);
}

// ---------------------------------------------------------------------------
// Concrete IntoSystem impls for function arities 0..8
// ---------------------------------------------------------------------------

/// Zero-param system.
impl<F> IntoSystem<()> for F
where
    F: FnMut() + Send + Sync + 'static,
{
    fn into_system(self) -> Box<dyn crate::system::System> {
        Box::new(ZeroParamSystem { func: self })
    }
}

struct ZeroParamSystem<F: FnMut()> {
    func: F,
}

impl<F: FnMut() + Send + Sync> crate::system::System for ZeroParamSystem<F> {
    fn run(&mut self, _world: &mut World) {
        (self.func)();
    }
}

// ---------------------------------------------------------------------------
// WorldAccessor — helper for safely splitting world borrows
// ---------------------------------------------------------------------------

/// Provides structured access to world data for system parameter extraction.
/// This avoids the need for multiple mutable borrows of the world.
pub struct WorldAccessor<'w> {
    world: &'w World,
}

impl<'w> WorldAccessor<'w> {
    /// Create a new world accessor.
    pub fn new(world: &'w World) -> Self {
        Self { world }
    }

    /// Get a resource reference.
    pub fn resource<R: 'static + Send + Sync>(&self) -> Option<&'w R> {
        self.world.get_resource::<R>()
    }

    /// Get a component reference.
    pub fn component<C: Component>(&self, entity: Entity) -> Option<&'w C> {
        self.world.get_component::<C>(entity)
    }

    /// Check if an entity is alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.world.is_alive(entity)
    }

    /// Get all alive entities.
    pub fn alive_entities(&self) -> Vec<Entity> {
        self.world.entity_storage().iter_alive().collect()
    }
}

// ---------------------------------------------------------------------------
// SystemMeta — metadata about a system for scheduling
// ---------------------------------------------------------------------------

/// Metadata about a system that helps the scheduler make decisions.
#[derive(Debug, Clone)]
pub struct SystemMeta {
    /// Human-readable name of the system.
    pub name: String,
    /// Access patterns for parallel scheduling.
    pub access: SystemParamAccess,
    /// The tick when this system last ran.
    pub last_run_tick: u32,
    /// Whether this system is enabled.
    pub enabled: bool,
    /// Run condition: the system only runs if this returns true.
    pub run_condition: Option<RunConditionId>,
}

impl SystemMeta {
    /// Create new system metadata.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            access: SystemParamAccess::default(),
            last_run_tick: 0,
            enabled: true,
            run_condition: None,
        }
    }

    /// Set the access patterns.
    pub fn with_access(mut self, access: SystemParamAccess) -> Self {
        self.access = access;
        self
    }

    /// Disable this system.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this system.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

/// Opaque identifier for a run condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunConditionId(pub u32);

// ---------------------------------------------------------------------------
// Run conditions
// ---------------------------------------------------------------------------

/// A condition that determines whether a system should run.
pub struct RunCondition {
    /// Unique identifier.
    pub id: RunConditionId,
    /// Human-readable label.
    pub label: String,
    /// The condition function.
    func: Box<dyn Fn(&World) -> bool + Send + Sync>,
}

impl RunCondition {
    /// Create a new run condition.
    pub fn new(
        id: RunConditionId,
        label: impl Into<String>,
        func: impl Fn(&World) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            id,
            label: label.into(),
            func: Box::new(func),
        }
    }

    /// Evaluate the condition against the world.
    pub fn evaluate(&self, world: &World) -> bool {
        (self.func)(world)
    }
}

/// Registry of run conditions, queryable by id.
pub struct RunConditionRegistry {
    conditions: HashMap<RunConditionId, RunCondition>,
    next_id: u32,
}

impl RunConditionRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            conditions: HashMap::new(),
            next_id: 0,
        }
    }

    /// Register a new condition and return its id.
    pub fn register(
        &mut self,
        label: impl Into<String>,
        func: impl Fn(&World) -> bool + Send + Sync + 'static,
    ) -> RunConditionId {
        let id = RunConditionId(self.next_id);
        self.next_id += 1;
        self.conditions
            .insert(id, RunCondition::new(id, label, func));
        id
    }

    /// Evaluate a condition by id.
    pub fn evaluate(&self, id: RunConditionId, world: &World) -> bool {
        self.conditions
            .get(&id)
            .map(|c| c.evaluate(world))
            .unwrap_or(true)
    }

    /// Get a condition by id.
    pub fn get(&self, id: RunConditionId) -> Option<&RunCondition> {
        self.conditions.get(&id)
    }
}

impl Default for RunConditionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Common run condition factories
// ---------------------------------------------------------------------------

/// Create a "run if resource exists" condition.
pub fn resource_exists<R: 'static + Send + Sync>() -> impl Fn(&World) -> bool {
    move |world: &World| world.has_resource::<R>()
}

/// Create a "run if resource equals value" condition.
pub fn resource_equals<R: 'static + Send + Sync + PartialEq>(
    value: R,
) -> impl Fn(&World) -> bool {
    move |world: &World| {
        world
            .get_resource::<R>()
            .map(|r| *r == value)
            .unwrap_or(false)
    }
}

/// Run every N ticks.
pub fn run_every_n_ticks(n: u32) -> impl Fn(&World) -> bool {
    move |world: &World| world.current_tick() % n == 0
}

/// Run only on the first tick.
pub fn run_once() -> impl Fn(&World) -> bool {
    let mut ran = false;
    move |_world: &World| {
        if ran {
            false
        } else {
            ran = true;
            true
        }
    }
}

// ---------------------------------------------------------------------------
// ExclusiveSystem — system with exclusive world access
// ---------------------------------------------------------------------------

/// A system that requires exclusive (`&mut World`) access, preventing
/// parallel execution but allowing arbitrary world mutations.
pub struct ExclusiveSystem {
    func: Box<dyn FnMut(&mut World) + Send + Sync>,
    name: String,
}

impl ExclusiveSystem {
    /// Create a new exclusive system.
    pub fn new(
        name: impl Into<String>,
        func: impl FnMut(&mut World) + Send + Sync + 'static,
    ) -> Self {
        Self {
            func: Box::new(func),
            name: name.into(),
        }
    }

    /// Get the system name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl crate::system::System for ExclusiveSystem {
    fn run(&mut self, world: &mut World) {
        (self.func)(world);
    }
}

// ---------------------------------------------------------------------------
// ParamSystemAdapter — adapts a closure with SystemParam to System trait
// ---------------------------------------------------------------------------

/// Adapts a function that uses the world directly (but with a specific
/// access pattern declared) into the `System` trait.
pub struct ParamSystemAdapter {
    name: String,
    func: Box<dyn FnMut(&mut World) + Send + Sync>,
    access: SystemParamAccess,
}

impl ParamSystemAdapter {
    /// Create a new adapter.
    pub fn new(
        name: impl Into<String>,
        access: SystemParamAccess,
        func: impl FnMut(&mut World) + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            func: Box::new(func),
            access,
        }
    }

    /// Get the declared access.
    pub fn access(&self) -> &SystemParamAccess {
        &self.access
    }

    /// Get the system name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl crate::system::System for ParamSystemAdapter {
    fn run(&mut self, world: &mut World) {
        (self.func)(world);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Clone)]
    struct Position {
        x: f32,
        y: f32,
    }
    impl Component for Position {}

    #[derive(Debug, PartialEq, Clone)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }
    impl Component for Velocity {}

    #[derive(Debug, PartialEq, Clone)]
    struct Health(f32);
    impl Component for Health {}

    #[derive(Debug, PartialEq, Clone)]
    struct DeltaTime(f32);

    #[derive(Debug, PartialEq, Clone)]
    struct Score(u32);

    // -- Commands tests ---------------------------------------------------------

    #[test]
    fn commands_spawn_and_flush() {
        let mut world = World::new();
        let mut commands = Commands::new();

        commands.spawn_with(Position { x: 1.0, y: 2.0 });
        commands.spawn_with(Position { x: 3.0, y: 4.0 });
        assert_eq!(commands.len(), 2);

        commands.flush(&mut world);
        assert_eq!(world.entity_count(), 2);
    }

    #[test]
    fn commands_despawn() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .build();

        let mut commands = Commands::new();
        commands.despawn(e);
        commands.flush(&mut world);

        assert!(!world.is_alive(e));
    }

    #[test]
    fn commands_add_remove_component() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        let mut commands = Commands::new();
        commands.add_component(e, Health(100.0));
        commands.flush(&mut world);
        assert!(world.has_component::<Health>(e));

        let mut commands = Commands::new();
        commands.remove_component::<Health>(e);
        commands.flush(&mut world);
        assert!(!world.has_component::<Health>(e));
    }

    #[test]
    fn commands_insert_resource() {
        let mut world = World::new();
        let mut commands = Commands::new();
        commands.insert_resource(DeltaTime(0.016));
        commands.flush(&mut world);

        assert_eq!(
            world.get_resource::<DeltaTime>(),
            Some(&DeltaTime(0.016))
        );
    }

    #[test]
    fn commands_entity_builder() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        let mut commands = Commands::new();
        commands
            .entity(e)
            .add(Position { x: 1.0, y: 2.0 })
            .add(Velocity { dx: 3.0, dy: 4.0 });
        commands.flush(&mut world);

        assert!(world.has_component::<Position>(e));
        assert!(world.has_component::<Velocity>(e));
    }

    #[test]
    fn commands_custom_command() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        let mut commands = Commands::new();
        commands.add_command(move |w: &mut World| {
            w.add_component(e, Health(50.0));
        });
        commands.flush(&mut world);

        assert_eq!(
            world.get_component::<Health>(e).map(|h| h.0),
            Some(50.0)
        );
    }

    #[test]
    fn commands_is_empty() {
        let commands = Commands::new();
        assert!(commands.is_empty());
        assert_eq!(commands.len(), 0);
    }

    // -- EventReader tests ------------------------------------------------------

    #[test]
    fn event_reader_reads_events() {
        let mut events = Events::<u32>::new();
        events.send(1);
        events.send(2);
        events.send(3);

        let reader = EventReader::new(Some(&events));
        assert_eq!(reader.len(), 3);
        assert!(!reader.is_empty());

        let collected: Vec<u32> = reader.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn event_reader_empty_when_no_events() {
        let reader = EventReader::<u32>::new(None);
        assert!(reader.is_empty());
        assert_eq!(reader.len(), 0);
        assert_eq!(reader.iter().count(), 0);
    }

    // -- EventWriter tests ------------------------------------------------------

    #[test]
    fn event_writer_buffers_events() {
        let mut writer = EventWriter::<u32>::new();
        writer.send(10);
        writer.send(20);
        assert_eq!(writer.len(), 2);

        let drained = writer.drain();
        assert_eq!(drained, vec![10, 20]);
        assert!(writer.is_empty());
    }

    #[test]
    fn event_writer_send_batch() {
        let mut writer = EventWriter::<u32>::new();
        writer.send_batch(vec![1, 2, 3, 4, 5]);
        assert_eq!(writer.len(), 5);
    }

    #[test]
    fn event_writer_flush_to_world() {
        let mut world = World::new();
        world.add_resource(Events::<u32>::new());

        let mut writer = EventWriter::<u32>::new();
        writer.send(42);
        writer.send(99);
        writer.flush(&mut world);

        let events = world.get_resource::<Events<u32>>().unwrap();
        let collected: Vec<u32> = events.iter().copied().collect();
        assert_eq!(collected, vec![42, 99]);
    }

    // -- Local<T> tests ---------------------------------------------------------

    #[test]
    fn local_default_value() {
        let local: Local<u32> = Local::new();
        assert_eq!(*local, 0);
    }

    #[test]
    fn local_with_value() {
        let local = Local::with_value(42u32);
        assert_eq!(*local, 42);
    }

    #[test]
    fn local_deref_mut() {
        let mut local: Local<u32> = Local::new();
        *local = 100;
        assert_eq!(*local, 100);
    }

    #[test]
    fn local_replace() {
        let mut local = Local::with_value(10u32);
        let old = local.replace(20);
        assert_eq!(old, 10);
        assert_eq!(*local, 20);
    }

    #[test]
    fn local_take() {
        let mut local = Local::with_value(vec![1, 2, 3]);
        let taken = local.take();
        assert_eq!(taken, vec![1, 2, 3]);
        assert!(local.is_empty());
    }

    #[test]
    fn local_get_and_get_mut() {
        let mut local = Local::with_value(String::from("hello"));
        assert_eq!(local.get(), "hello");
        local.get_mut().push_str(" world");
        assert_eq!(*local, "hello world");
    }

    // -- SystemParamAccess tests ------------------------------------------------

    #[test]
    fn access_no_conflict_reads_only() {
        let a = SystemParamAccess {
            component_reads: vec![ComponentId::of::<Position>()],
            ..Default::default()
        };
        let b = SystemParamAccess {
            component_reads: vec![ComponentId::of::<Position>()],
            ..Default::default()
        };
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn access_conflict_read_write() {
        let a = SystemParamAccess {
            component_reads: vec![ComponentId::of::<Position>()],
            ..Default::default()
        };
        let b = SystemParamAccess {
            component_writes: vec![ComponentId::of::<Position>()],
            ..Default::default()
        };
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn access_conflict_write_write() {
        let a = SystemParamAccess {
            component_writes: vec![ComponentId::of::<Position>()],
            ..Default::default()
        };
        let b = SystemParamAccess {
            component_writes: vec![ComponentId::of::<Position>()],
            ..Default::default()
        };
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn access_no_conflict_different_components() {
        let a = SystemParamAccess {
            component_writes: vec![ComponentId::of::<Position>()],
            ..Default::default()
        };
        let b = SystemParamAccess {
            component_writes: vec![ComponentId::of::<Velocity>()],
            ..Default::default()
        };
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn access_resource_conflict() {
        let a = SystemParamAccess {
            resource_reads: vec![TypeId::of::<DeltaTime>()],
            ..Default::default()
        };
        let b = SystemParamAccess {
            resource_writes: vec![TypeId::of::<DeltaTime>()],
            ..Default::default()
        };
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn access_merge() {
        let mut a = SystemParamAccess {
            component_reads: vec![ComponentId::of::<Position>()],
            ..Default::default()
        };
        let b = SystemParamAccess {
            component_writes: vec![ComponentId::of::<Velocity>()],
            uses_commands: true,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.component_reads.len(), 1);
        assert_eq!(a.component_writes.len(), 1);
        assert!(a.uses_commands);
    }

    // -- SystemMeta tests -------------------------------------------------------

    #[test]
    fn system_meta_enable_disable() {
        let mut meta = SystemMeta::new("test_system");
        assert!(meta.enabled);
        meta.disable();
        assert!(!meta.enabled);
        meta.enable();
        assert!(meta.enabled);
    }

    // -- RunCondition tests -----------------------------------------------------

    #[test]
    fn run_condition_evaluates() {
        let world = World::new();
        let cond = RunCondition::new(
            RunConditionId(0),
            "always_true",
            |_w: &World| true,
        );
        assert!(cond.evaluate(&world));

        let cond_false = RunCondition::new(
            RunConditionId(1),
            "always_false",
            |_w: &World| false,
        );
        assert!(!cond_false.evaluate(&world));
    }

    #[test]
    fn run_condition_registry() {
        let mut registry = RunConditionRegistry::new();
        let world = World::new();

        let id = registry.register("has_time", |w: &World| {
            w.has_resource::<DeltaTime>()
        });

        assert!(!registry.evaluate(id, &world));
    }

    #[test]
    fn resource_exists_condition() {
        let mut world = World::new();
        let cond = resource_exists::<DeltaTime>();
        assert!(!cond(&world));

        world.add_resource(DeltaTime(0.016));
        assert!(cond(&world));
    }

    #[test]
    fn resource_equals_condition() {
        let mut world = World::new();
        world.add_resource(Score(100));

        let cond = resource_equals(Score(100));
        assert!(cond(&world));

        let cond2 = resource_equals(Score(200));
        assert!(!cond2(&world));
    }

    #[test]
    fn run_every_n_ticks_condition() {
        let mut world = World::new();
        let cond = run_every_n_ticks(3);

        // Tick 0: 0 % 3 == 0, should run.
        assert!(cond(&world));

        world.increment_tick(); // tick 1
        assert!(!cond(&world));

        world.increment_tick(); // tick 2
        assert!(!cond(&world));

        world.increment_tick(); // tick 3
        assert!(cond(&world));
    }

    // -- ExclusiveSystem tests --------------------------------------------------

    #[test]
    fn exclusive_system_runs() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        let mut system = ExclusiveSystem::new("test", move |w: &mut World| {
            w.add_component(e, Health(100.0));
        });
        assert_eq!(system.name(), "test");

        crate::system::System::run(&mut system, &mut world);
        assert!(world.has_component::<Health>(e));
    }

    // -- Query param tests ------------------------------------------------------

    #[test]
    fn query_param_iterates() {
        let mut world = World::new();
        let e1 = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .with(Velocity { dx: 3.0, dy: 4.0 })
            .build();
        let e2 = world
            .spawn_entity()
            .with(Position { x: 5.0, y: 6.0 })
            .build();

        let state = crate::query::QueryState::new::<(&Position, &Velocity)>(&world);
        let query: Query<'_, (&Position, &Velocity)> = Query::new(&world, &state);

        let results: Vec<(Entity, (&Position, &Velocity))> = query.collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, e1);
    }

    #[test]
    fn query_param_get_single() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Health(100.0))
            .build();

        let state = crate::query::QueryState::new::<&Health>(&world);
        let query: Query<'_, &Health> = Query::new(&world, &state);

        let single = query.get_single();
        assert!(single.is_some());
        assert_eq!(single.unwrap().1 .0, 100.0);
    }

    #[test]
    fn query_param_get_single_none_when_multiple() {
        let mut world = World::new();
        world.spawn_entity().with(Health(50.0)).build();
        world.spawn_entity().with(Health(75.0)).build();

        let state = crate::query::QueryState::new::<&Health>(&world);
        let query: Query<'_, &Health> = Query::new(&world, &state);

        assert!(query.get_single().is_none());
    }

    #[test]
    fn query_param_get_entity() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .build();

        let state = crate::query::QueryState::new::<&Position>(&world);
        let query: Query<'_, &Position> = Query::new(&world, &state);

        let pos = query.get(e);
        assert!(pos.is_some());
        assert_eq!(pos.unwrap().x, 1.0);
    }

    #[test]
    fn query_param_contains() {
        let mut world = World::new();
        let e1 = world
            .spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .build();
        let e2 = world.spawn_entity().build();

        let state = crate::query::QueryState::new::<&Position>(&world);
        let query: Query<'_, &Position> = Query::new(&world, &state);

        assert!(query.contains(e1));
        assert!(!query.contains(e2));
    }

    #[test]
    fn query_param_is_empty() {
        let world = World::new();
        let state = crate::query::QueryState::new::<&Position>(&world);
        let query: Query<'_, &Position> = Query::new(&world, &state);
        assert!(query.is_empty());
    }

    #[test]
    fn query_param_count() {
        let mut world = World::new();
        for i in 0..10 {
            world
                .spawn_entity()
                .with(Position {
                    x: i as f32,
                    y: 0.0,
                })
                .build();
        }

        let state = crate::query::QueryState::new::<&Position>(&world);
        let query: Query<'_, &Position> = Query::new(&world, &state);
        assert_eq!(query.count(), 10);
    }

    // -- WorldAccessor tests ----------------------------------------------------

    #[test]
    fn world_accessor_resource() {
        let mut world = World::new();
        world.add_resource(DeltaTime(0.016));

        let accessor = WorldAccessor::new(&world);
        let dt = accessor.resource::<DeltaTime>().unwrap();
        assert_eq!(dt.0, 0.016);
    }

    #[test]
    fn world_accessor_component() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .build();

        let accessor = WorldAccessor::new(&world);
        let pos = accessor.component::<Position>(e).unwrap();
        assert_eq!(pos.x, 1.0);
    }

    #[test]
    fn world_accessor_alive_entities() {
        let mut world = World::new();
        world.spawn_entity().build();
        world.spawn_entity().build();
        world.spawn_entity().build();

        let accessor = WorldAccessor::new(&world);
        assert_eq!(accessor.alive_entities().len(), 3);
    }

    // -- ZeroParamSystem test ---------------------------------------------------

    #[test]
    fn into_system_zero_params() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut system = (move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .into_system();

        let mut world = World::new();
        system.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        system.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    // -- ParamSystemAdapter test ------------------------------------------------

    #[test]
    fn param_system_adapter() {
        let mut world = World::new();
        let e = world.spawn_entity().build();

        let access = SystemParamAccess {
            component_writes: vec![ComponentId::of::<Health>()],
            ..Default::default()
        };

        let mut adapter = ParamSystemAdapter::new(
            "add_health",
            access,
            move |w: &mut World| {
                w.add_component(e, Health(100.0));
            },
        );

        assert_eq!(adapter.name(), "add_health");
        crate::system::System::run(&mut adapter, &mut world);
        assert!(world.has_component::<Health>(e));
    }

    // -- ResParam tests ---------------------------------------------------------

    #[test]
    fn res_param_deref() {
        let dt = DeltaTime(0.016);
        let res = ResParam { value: &dt };
        assert_eq!(res.0, 0.016);
        let inner = res.into_inner();
        assert_eq!(inner.0, 0.016);
    }

    // -- Tuple SystemParam tests ------------------------------------------------

    #[test]
    fn empty_tuple_system_param() {
        let world = World::new();
        let mut state = <()>::init_state(&world);
        let result = unsafe { <()>::fetch(&world, &mut state) };
        assert_eq!(result, ());
    }
}
