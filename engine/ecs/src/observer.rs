//! Event observer system for the Genovo ECS.
//!
//! Observers react to component lifecycle events -- when components are added,
//! removed, or changed on entities. Unlike the poll-based [`Changed`] filter,
//! observers receive event callbacks and can respond immediately.
//!
//! # Event types
//!
//! - [`OnAdd<T>`] -- triggered when component `T` is added to an entity.
//! - [`OnRemove<T>`] -- triggered when component `T` is removed from an entity.
//! - [`OnChange<T>`] -- triggered when component `T`'s value changes.
//!
//! # Observer registration
//!
//! ```ignore
//! registry.on_add::<Health>(|entity, world| {
//!     println!("Entity {:?} gained Health!", entity);
//! });
//!
//! registry.on_remove::<Health>(|entity, world| {
//!     println!("Entity {:?} lost Health!", entity);
//! });
//! ```
//!
//! # Batch processing
//!
//! Events are collected during world mutations and dispatched in bulk at the
//! end of the frame (or when explicitly flushed). This avoids re-entrant
//! mutation problems and gives the observer system a consistent world view.

use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::world::World;

// ---------------------------------------------------------------------------
// Lifecycle event types
// ---------------------------------------------------------------------------

/// Marker for "component T was added" events.
pub struct OnAdd<T: Component>(PhantomData<T>);

/// Marker for "component T was removed" events.
pub struct OnRemove<T: Component>(PhantomData<T>);

/// Marker for "component T value changed" events.
pub struct OnChange<T: Component>(PhantomData<T>);

/// The kind of lifecycle event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifecycleEventKind {
    /// A component was added to an entity.
    Add,
    /// A component was removed from an entity.
    Remove,
    /// A component value was changed.
    Change,
}

/// A concrete lifecycle event instance ready for dispatch.
#[derive(Debug, Clone)]
pub struct LifecycleEvent {
    /// The entity affected.
    pub entity: Entity,
    /// Which component type.
    pub component_id: ComponentId,
    /// What happened.
    pub kind: LifecycleEventKind,
}

impl LifecycleEvent {
    /// Create a new Add event.
    pub fn add(entity: Entity, component_id: ComponentId) -> Self {
        Self {
            entity,
            component_id,
            kind: LifecycleEventKind::Add,
        }
    }

    /// Create a new Remove event.
    pub fn remove(entity: Entity, component_id: ComponentId) -> Self {
        Self {
            entity,
            component_id,
            kind: LifecycleEventKind::Remove,
        }
    }

    /// Create a new Change event.
    pub fn change(entity: Entity, component_id: ComponentId) -> Self {
        Self {
            entity,
            component_id,
            kind: LifecycleEventKind::Change,
        }
    }
}

// ---------------------------------------------------------------------------
// Observer trait
// ---------------------------------------------------------------------------

/// Trait for observer callbacks that react to component lifecycle events.
///
/// Observers receive the entity handle and can inspect (but should not
/// structurally modify) the world during dispatch. Structural changes
/// should be deferred via a command queue.
pub trait Observer: Send + Sync {
    /// Called when the observed event occurs.
    ///
    /// The `world` reference is immutable to prevent re-entrant mutations
    /// during dispatch. Use a [`CommandQueue`](crate::CommandQueue) for
    /// deferred changes.
    fn on_event(&self, event: &LifecycleEvent, world: &World);
}

/// A closure-based observer for convenience.
struct ClosureObserver<F>
where
    F: Fn(&LifecycleEvent, &World) + Send + Sync,
{
    callback: F,
}

impl<F> Observer for ClosureObserver<F>
where
    F: Fn(&LifecycleEvent, &World) + Send + Sync,
{
    fn on_event(&self, event: &LifecycleEvent, world: &World) {
        (self.callback)(event, world);
    }
}

/// A simpler closure observer that only receives the entity.
struct SimpleObserver<F>
where
    F: Fn(Entity, &World) + Send + Sync,
{
    callback: F,
}

impl<F> Observer for SimpleObserver<F>
where
    F: Fn(Entity, &World) + Send + Sync,
{
    fn on_event(&self, event: &LifecycleEvent, world: &World) {
        (self.callback)(event.entity, world);
    }
}

// ---------------------------------------------------------------------------
// ObserverKey -- identifies a registered observer
// ---------------------------------------------------------------------------

/// The key used to match observers to events: (component_type, event_kind).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ObserverKey {
    component_id: ComponentId,
    kind: LifecycleEventKind,
}

/// A unique handle returned when an observer is registered, used to
/// unregister it later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObserverId(u64);

// ---------------------------------------------------------------------------
// ObserverRegistry
// ---------------------------------------------------------------------------

/// Central registry for lifecycle observers.
///
/// The registry collects events during world mutations and dispatches them
/// to registered observers. Events are buffered and flushed explicitly,
/// ensuring observers see a consistent world state.
///
/// # Thread safety
///
/// The registry itself is `Send + Sync`. Observer callbacks must also be
/// `Send + Sync`.
pub struct ObserverRegistry {
    /// Map from observer key to list of (ObserverId, Observer).
    observers: HashMap<ObserverKey, Vec<(ObserverId, Box<dyn Observer>)>>,
    /// Buffered events waiting to be dispatched.
    pending_events: Vec<LifecycleEvent>,
    /// Next observer id to assign.
    next_id: u64,
    /// Whether event buffering is enabled. When false, events are dispatched
    /// immediately.
    buffered: bool,
    /// Maximum number of events to buffer before auto-flushing.
    max_buffer_size: usize,
    /// Total number of events dispatched (for diagnostics).
    total_dispatched: u64,
}

impl ObserverRegistry {
    /// Create a new, empty observer registry with buffered dispatch.
    pub fn new() -> Self {
        Self {
            observers: HashMap::new(),
            pending_events: Vec::new(),
            next_id: 0,
            buffered: true,
            max_buffer_size: 10_000,
            total_dispatched: 0,
        }
    }

    /// Create a registry that dispatches events immediately (no buffering).
    pub fn immediate() -> Self {
        Self {
            buffered: false,
            ..Self::new()
        }
    }

    /// Set the maximum buffer size before auto-flush.
    pub fn set_max_buffer_size(&mut self, size: usize) {
        self.max_buffer_size = size;
    }

    // -- Registration ---------------------------------------------------------

    /// Register an observer for `OnAdd<T>` events.
    pub fn on_add<T: Component, F>(&mut self, callback: F) -> ObserverId
    where
        F: Fn(Entity, &World) + Send + Sync + 'static,
    {
        self.register_observer(
            ComponentId::of::<T>(),
            LifecycleEventKind::Add,
            Box::new(SimpleObserver { callback }),
        )
    }

    /// Register an observer for `OnRemove<T>` events.
    pub fn on_remove<T: Component, F>(&mut self, callback: F) -> ObserverId
    where
        F: Fn(Entity, &World) + Send + Sync + 'static,
    {
        self.register_observer(
            ComponentId::of::<T>(),
            LifecycleEventKind::Remove,
            Box::new(SimpleObserver { callback }),
        )
    }

    /// Register an observer for `OnChange<T>` events.
    pub fn on_change<T: Component, F>(&mut self, callback: F) -> ObserverId
    where
        F: Fn(Entity, &World) + Send + Sync + 'static,
    {
        self.register_observer(
            ComponentId::of::<T>(),
            LifecycleEventKind::Change,
            Box::new(SimpleObserver { callback }),
        )
    }

    /// Register a full-event observer.
    pub fn on_event<T: Component, F>(
        &mut self,
        kind: LifecycleEventKind,
        callback: F,
    ) -> ObserverId
    where
        F: Fn(&LifecycleEvent, &World) + Send + Sync + 'static,
    {
        self.register_observer(
            ComponentId::of::<T>(),
            kind,
            Box::new(ClosureObserver { callback }),
        )
    }

    /// Register a raw observer for any component and event kind.
    pub fn register_observer(
        &mut self,
        component_id: ComponentId,
        kind: LifecycleEventKind,
        observer: Box<dyn Observer>,
    ) -> ObserverId {
        let id = ObserverId(self.next_id);
        self.next_id += 1;

        let key = ObserverKey { component_id, kind };
        self.observers
            .entry(key)
            .or_insert_with(Vec::new)
            .push((id, observer));

        id
    }

    /// Unregister an observer by its id. Returns `true` if found and removed.
    pub fn unregister(&mut self, observer_id: ObserverId) -> bool {
        for observers in self.observers.values_mut() {
            if let Some(pos) = observers.iter().position(|(id, _)| *id == observer_id) {
                observers.remove(pos);
                return true;
            }
        }
        false
    }

    /// Remove all observers for a specific component and event kind.
    pub fn clear_observers(&mut self, component_id: ComponentId, kind: LifecycleEventKind) {
        let key = ObserverKey { component_id, kind };
        self.observers.remove(&key);
    }

    /// Remove all registered observers.
    pub fn clear_all(&mut self) {
        self.observers.clear();
    }

    // -- Event emission -------------------------------------------------------

    /// Emit a lifecycle event. If buffered, the event is queued for later
    /// dispatch. If immediate, it is dispatched right away.
    pub fn emit(&mut self, event: LifecycleEvent, world: &World) {
        if self.buffered {
            self.pending_events.push(event);

            // Auto-flush if buffer is full.
            if self.pending_events.len() >= self.max_buffer_size {
                self.flush(world);
            }
        } else {
            self.dispatch_event(&event, world);
        }
    }

    /// Emit an add event for component `T` on the given entity.
    pub fn emit_add<T: Component>(&mut self, entity: Entity, world: &World) {
        self.emit(LifecycleEvent::add(entity, ComponentId::of::<T>()), world);
    }

    /// Emit a remove event for component `T` on the given entity.
    pub fn emit_remove<T: Component>(&mut self, entity: Entity, world: &World) {
        self.emit(
            LifecycleEvent::remove(entity, ComponentId::of::<T>()),
            world,
        );
    }

    /// Emit a change event for component `T` on the given entity.
    pub fn emit_change<T: Component>(&mut self, entity: Entity, world: &World) {
        self.emit(
            LifecycleEvent::change(entity, ComponentId::of::<T>()),
            world,
        );
    }

    /// Emit a batch of events at once.
    pub fn emit_batch(&mut self, events: Vec<LifecycleEvent>, world: &World) {
        if self.buffered {
            self.pending_events.extend(events);
            if self.pending_events.len() >= self.max_buffer_size {
                self.flush(world);
            }
        } else {
            for event in &events {
                self.dispatch_event(event, world);
            }
        }
    }

    // -- Dispatch -------------------------------------------------------------

    /// Flush all pending events, dispatching them to registered observers.
    pub fn flush(&mut self, world: &World) {
        // Take ownership of the pending events to avoid borrow issues.
        let events: Vec<LifecycleEvent> = self.pending_events.drain(..).collect();
        for event in &events {
            self.dispatch_event(event, world);
        }
    }

    /// Dispatch a single event to all matching observers.
    fn dispatch_event(&mut self, event: &LifecycleEvent, world: &World) {
        let key = ObserverKey {
            component_id: event.component_id,
            kind: event.kind,
        };
        if let Some(observers) = self.observers.get(&key) {
            for (_, observer) in observers {
                observer.on_event(event, world);
            }
        }
        self.total_dispatched += 1;
    }

    // -- Diagnostics ----------------------------------------------------------

    /// Number of pending (unbuffered) events.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending_events.len()
    }

    /// Whether there are any pending events.
    #[inline]
    pub fn has_pending(&self) -> bool {
        !self.pending_events.is_empty()
    }

    /// Total number of observers registered.
    pub fn observer_count(&self) -> usize {
        self.observers.values().map(|v| v.len()).sum()
    }

    /// Total number of events dispatched since creation.
    pub fn total_dispatched(&self) -> u64 {
        self.total_dispatched
    }

    /// Whether the registry is in buffered mode.
    pub fn is_buffered(&self) -> bool {
        self.buffered
    }

    /// Set buffered mode.
    pub fn set_buffered(&mut self, buffered: bool) {
        self.buffered = buffered;
    }

    /// Clear all pending events without dispatching them.
    pub fn discard_pending(&mut self) {
        self.pending_events.clear();
    }

    /// Get a list of component/kind pairs that have observers registered.
    pub fn observed_events(&self) -> Vec<(ComponentId, LifecycleEventKind)> {
        self.observers
            .keys()
            .map(|k| (k.component_id, k.kind))
            .collect()
    }
}

impl Default for ObserverRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ObserverSystem -- system that flushes observer events each frame
// ---------------------------------------------------------------------------

/// A system that flushes pending observer events at the end of each frame.
///
/// Add this to your schedule to ensure observers are dispatched:
///
/// ```ignore
/// schedule.add_system_to_stage(Stage::Last, ObserverFlushSystem);
/// ```
pub struct ObserverFlushSystem;

impl crate::system::System for ObserverFlushSystem {
    fn run(&mut self, world: &mut World) {
        // The ObserverRegistry is stored as a world resource.
        // We need to temporarily remove it to avoid borrow conflicts.
        if let Some(mut registry) = world.remove_resource::<ObserverRegistry>() {
            registry.flush(world);
            world.add_resource(registry);
        }
    }
}

// ---------------------------------------------------------------------------
// EventCollector -- helper for batch event collection during mutations
// ---------------------------------------------------------------------------

/// Collects lifecycle events during a batch of world mutations, then
/// dispatches them all at once.
///
/// ```ignore
/// let mut collector = EventCollector::new();
///
/// // During mutations:
/// collector.record_add::<Position>(entity);
/// collector.record_change::<Velocity>(entity);
///
/// // After mutations:
/// collector.dispatch(&mut registry, &world);
/// ```
pub struct EventCollector {
    events: Vec<LifecycleEvent>,
}

impl EventCollector {
    /// Create a new, empty collector.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Create a collector with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
        }
    }

    /// Record a component addition event.
    pub fn record_add<T: Component>(&mut self, entity: Entity) {
        self.events.push(LifecycleEvent::add(
            entity,
            ComponentId::of::<T>(),
        ));
    }

    /// Record a component removal event.
    pub fn record_remove<T: Component>(&mut self, entity: Entity) {
        self.events.push(LifecycleEvent::remove(
            entity,
            ComponentId::of::<T>(),
        ));
    }

    /// Record a component change event.
    pub fn record_change<T: Component>(&mut self, entity: Entity) {
        self.events.push(LifecycleEvent::change(
            entity,
            ComponentId::of::<T>(),
        ));
    }

    /// Record a raw event.
    pub fn record(&mut self, event: LifecycleEvent) {
        self.events.push(event);
    }

    /// Dispatch all collected events to the registry.
    pub fn dispatch(self, registry: &mut ObserverRegistry, world: &World) {
        registry.emit_batch(self.events, world);
    }

    /// Number of collected events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether no events have been collected.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Clear all collected events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Drain collected events into a vec.
    pub fn drain(&mut self) -> Vec<LifecycleEvent> {
        self.events.drain(..).collect()
    }
}

impl Default for EventCollector {
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
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[derive(Debug, PartialEq, Clone)]
    struct Position { x: f32, y: f32 }
    impl Component for Position {}

    #[derive(Debug, PartialEq, Clone)]
    struct Velocity { dx: f32, dy: f32 }
    impl Component for Velocity {}

    #[derive(Debug, PartialEq, Clone)]
    struct Health(f32);
    impl Component for Health {}

    #[test]
    fn observer_on_add_buffered() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut registry = ObserverRegistry::new();
        registry.on_add::<Health, _>(move |entity, _world| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let e = world.spawn_entity().build();

        // Emit an add event.
        registry.emit_add::<Health>(e, &world);
        // Not yet dispatched.
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // Flush.
        registry.flush(&world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn observer_on_remove_immediate() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut registry = ObserverRegistry::immediate();
        registry.on_remove::<Health, _>(move |_entity, _world| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let e = world.spawn_entity().with(Health(100.0)).build();

        // Immediate dispatch.
        registry.emit_remove::<Health>(e, &world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn observer_on_change() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut registry = ObserverRegistry::new();
        registry.on_change::<Position, _>(move |_entity, _world| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let e = world.spawn_entity()
            .with(Position { x: 0.0, y: 0.0 })
            .build();

        registry.emit_change::<Position>(e, &world);
        registry.flush(&world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unregister_observer() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut registry = ObserverRegistry::new();
        let id = registry.on_add::<Health, _>(move |_entity, _world| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let e = world.spawn_entity().build();

        registry.emit_add::<Health>(e, &world);
        registry.flush(&world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Unregister.
        assert!(registry.unregister(id));

        registry.emit_add::<Health>(e, &world);
        registry.flush(&world);
        // Counter should not increase.
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn multiple_observers_same_event() {
        let mut world = World::new();
        let counter_a = Arc::new(AtomicU32::new(0));
        let counter_b = Arc::new(AtomicU32::new(0));
        let ca = counter_a.clone();
        let cb = counter_b.clone();

        let mut registry = ObserverRegistry::new();
        registry.on_add::<Health, _>(move |_, _| {
            ca.fetch_add(1, Ordering::SeqCst);
        });
        registry.on_add::<Health, _>(move |_, _| {
            cb.fetch_add(10, Ordering::SeqCst);
        });

        let e = world.spawn_entity().build();
        registry.emit_add::<Health>(e, &world);
        registry.flush(&world);

        assert_eq!(counter_a.load(Ordering::SeqCst), 1);
        assert_eq!(counter_b.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn event_collector() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let cc = counter.clone();

        let mut registry = ObserverRegistry::new();
        registry.on_add::<Position, _>(move |_, _| {
            cc.fetch_add(1, Ordering::SeqCst);
        });

        let e1 = world.spawn_entity().build();
        let e2 = world.spawn_entity().build();
        let e3 = world.spawn_entity().build();

        let mut collector = EventCollector::new();
        collector.record_add::<Position>(e1);
        collector.record_add::<Position>(e2);
        collector.record_add::<Position>(e3);

        assert_eq!(collector.len(), 3);
        collector.dispatch(&mut registry, &world);
        registry.flush(&world);

        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn clear_observers() {
        let mut registry = ObserverRegistry::new();
        registry.on_add::<Health, _>(|_, _| {});
        registry.on_add::<Health, _>(|_, _| {});
        assert_eq!(registry.observer_count(), 2);

        registry.clear_observers(ComponentId::of::<Health>(), LifecycleEventKind::Add);
        assert_eq!(registry.observer_count(), 0);
    }

    #[test]
    fn clear_all_observers() {
        let mut registry = ObserverRegistry::new();
        registry.on_add::<Health, _>(|_, _| {});
        registry.on_remove::<Position, _>(|_, _| {});
        registry.on_change::<Velocity, _>(|_, _| {});
        assert_eq!(registry.observer_count(), 3);

        registry.clear_all();
        assert_eq!(registry.observer_count(), 0);
    }

    #[test]
    fn discard_pending() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let cc = counter.clone();

        let mut registry = ObserverRegistry::new();
        registry.on_add::<Health, _>(move |_, _| {
            cc.fetch_add(1, Ordering::SeqCst);
        });

        let e = world.spawn_entity().build();
        registry.emit_add::<Health>(e, &world);
        assert!(registry.has_pending());

        registry.discard_pending();
        assert!(!registry.has_pending());

        registry.flush(&world);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn diagnostics() {
        let mut world = World::new();
        let mut registry = ObserverRegistry::immediate();
        registry.on_add::<Health, _>(|_, _| {});

        let e = world.spawn_entity().build();
        registry.emit_add::<Health>(e, &world);
        registry.emit_add::<Health>(e, &world);

        assert_eq!(registry.total_dispatched(), 2);
        assert!(registry.observer_count() > 0);
    }

    #[test]
    fn observer_flush_system() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let cc = counter.clone();

        let mut registry = ObserverRegistry::new();
        registry.on_add::<Health, _>(move |_, _| {
            cc.fetch_add(1, Ordering::SeqCst);
        });

        let e = world.spawn_entity().build();
        registry.emit_add::<Health>(e, &world);
        world.add_resource(registry);

        // Run the flush system.
        let mut system = ObserverFlushSystem;
        crate::system::System::run(&mut system, &mut world);

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn buffered_mode_toggle() {
        let mut registry = ObserverRegistry::new();
        assert!(registry.is_buffered());

        registry.set_buffered(false);
        assert!(!registry.is_buffered());
    }

    #[test]
    fn observed_events_list() {
        let mut registry = ObserverRegistry::new();
        registry.on_add::<Health, _>(|_, _| {});
        registry.on_remove::<Position, _>(|_, _| {});

        let observed = registry.observed_events();
        assert_eq!(observed.len(), 2);
    }

    #[test]
    fn on_event_full() {
        let mut world = World::new();
        let last_entity = Arc::new(std::sync::Mutex::new(Entity::PLACEHOLDER));
        let le = last_entity.clone();

        let mut registry = ObserverRegistry::new();
        registry.on_event::<Health, _>(LifecycleEventKind::Add, move |event, _world| {
            *le.lock().unwrap() = event.entity;
        });

        let e = world.spawn_entity().build();
        registry.emit_add::<Health>(e, &world);
        registry.flush(&world);

        assert_eq!(*last_entity.lock().unwrap(), e);
    }
}
