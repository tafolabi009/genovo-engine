//! Global event bus for decoupled communication.
//!
//! Provides a publish/subscribe pattern that allows engine subsystems and
//! gameplay code to communicate without direct dependencies. Events are
//! routed by type using `TypeId` for type-safe dispatch.
//!
//! # Features
//!
//! - **Type-safe routing**: events are dispatched based on their concrete
//!   Rust type via `TypeId`.
//! - **Immediate and deferred dispatch**: publish events for immediate
//!   processing or queue them for batch processing later.
//! - **Subscriber management**: register/unregister handlers with stable IDs.
//! - **Common engine events**: `EntitySpawned`, `EntityDespawned`,
//!   `SceneLoaded`, `WindowResized`, `KeyPressed`, `PhysicsStep`.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Event trait
// ---------------------------------------------------------------------------

/// Marker trait for types that can be published on the event bus.
///
/// Any `'static + Send + Sync` type can be an event. Implement this trait
/// to make intent explicit.
pub trait Event: Any + Send + Sync + fmt::Debug {
    /// Returns a human-readable name for this event type (for logging).
    fn event_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

// ---------------------------------------------------------------------------
// EventId
// ---------------------------------------------------------------------------

/// Unique identifier for an event type, derived from `TypeId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(TypeId);

impl EventId {
    /// Creates an `EventId` for the given event type.
    pub fn of<E: Event>() -> Self {
        Self(TypeId::of::<E>())
    }
}

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EventId({:?})", self.0)
    }
}

// ---------------------------------------------------------------------------
// SubscriberId
// ---------------------------------------------------------------------------

/// Unique identifier for a subscription, used to unsubscribe later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriberId(u64);

impl SubscriberId {
    /// Generates a new unique subscriber ID.
    fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for SubscriberId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sub({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Type-erased handler
// ---------------------------------------------------------------------------

/// A type-erased event handler that wraps a concrete `Fn(&E)`.
struct HandlerEntry {
    /// The subscriber ID.
    id: SubscriberId,
    /// Human-readable label (for debugging).
    label: String,
    /// The actual handler function, type-erased.
    handler: Box<dyn Fn(&dyn Any) + Send + Sync>,
    /// Whether this handler is enabled.
    enabled: bool,
    /// Priority: higher priority handlers are called first.
    priority: i32,
}

impl fmt::Debug for HandlerEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandlerEntry")
            .field("id", &self.id)
            .field("label", &self.label)
            .field("enabled", &self.enabled)
            .field("priority", &self.priority)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// QueuedEvent
// ---------------------------------------------------------------------------

/// A type-erased event waiting in the deferred queue.
struct QueuedEvent {
    /// The event type ID for routing.
    type_id: EventId,
    /// The boxed event value.
    event: Box<dyn Any + Send + Sync>,
}

impl fmt::Debug for QueuedEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueuedEvent")
            .field("type_id", &self.type_id)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// EventBus
// ---------------------------------------------------------------------------

/// A global event bus for decoupled publish/subscribe communication.
///
/// # Usage
///
/// ```rust,ignore
/// let mut bus = EventBus::new();
///
/// // Subscribe to an event type
/// let sub_id = bus.subscribe::<WindowResized>(|event| {
///     println!("Window resized to {}x{}", event.width, event.height);
/// });
///
/// // Publish an event (immediate dispatch)
/// bus.publish(WindowResized { width: 1920, height: 1080 });
///
/// // Queue an event for deferred processing
/// bus.queue(EntitySpawned { entity_id: 42 });
/// bus.process_queued();
///
/// // Unsubscribe
/// bus.unsubscribe(sub_id);
/// ```
pub struct EventBus {
    /// Handlers keyed by event type ID.
    handlers: HashMap<EventId, Vec<HandlerEntry>>,
    /// Deferred event queue.
    queue: Vec<QueuedEvent>,
    /// Whether the bus is enabled (disabled = all publishes are no-ops).
    enabled: bool,
    /// Statistics: total events published.
    stats_published: u64,
    /// Statistics: total events queued.
    stats_queued: u64,
    /// Statistics: total events dispatched (handlers called).
    stats_dispatched: u64,
}

impl fmt::Debug for EventBus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventBus")
            .field("handler_types", &self.handlers.len())
            .field("queue_size", &self.queue.len())
            .field("enabled", &self.enabled)
            .field("stats_published", &self.stats_published)
            .field("stats_queued", &self.stats_queued)
            .field("stats_dispatched", &self.stats_dispatched)
            .finish()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    /// Creates a new empty event bus.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            queue: Vec::new(),
            enabled: true,
            stats_published: 0,
            stats_queued: 0,
            stats_dispatched: 0,
        }
    }

    /// Sets whether the event bus is enabled.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether the event bus is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Subscribes a handler for events of type `E`.
    ///
    /// Returns a [`SubscriberId`] that can be used to unsubscribe later.
    pub fn subscribe<E: Event>(
        &mut self,
        handler: impl Fn(&E) + Send + Sync + 'static,
    ) -> SubscriberId {
        self.subscribe_with_options::<E>(handler, "", 0)
    }

    /// Subscribes a handler with a label and priority.
    ///
    /// Higher priority handlers are called before lower priority ones.
    pub fn subscribe_with_options<E: Event>(
        &mut self,
        handler: impl Fn(&E) + Send + Sync + 'static,
        label: impl Into<String>,
        priority: i32,
    ) -> SubscriberId {
        let id = SubscriberId::next();
        let event_id = EventId::of::<E>();

        let erased_handler = Box::new(move |any: &dyn Any| {
            if let Some(event) = any.downcast_ref::<E>() {
                handler(event);
            }
        });

        let entry = HandlerEntry {
            id,
            label: label.into(),
            handler: erased_handler,
            enabled: true,
            priority,
        };

        let handlers = self.handlers.entry(event_id).or_default();
        handlers.push(entry);

        // Sort by priority (descending) so highest priority is first
        handlers.sort_by(|a, b| b.priority.cmp(&a.priority));

        id
    }

    /// Unsubscribes a handler by its subscriber ID.
    ///
    /// Returns `true` if the handler was found and removed.
    pub fn unsubscribe(&mut self, id: SubscriberId) -> bool {
        for handlers in self.handlers.values_mut() {
            if let Some(pos) = handlers.iter().position(|h| h.id == id) {
                handlers.remove(pos);
                return true;
            }
        }
        false
    }

    /// Enables or disables a specific subscriber.
    pub fn set_subscriber_enabled(&mut self, id: SubscriberId, enabled: bool) -> bool {
        for handlers in self.handlers.values_mut() {
            if let Some(entry) = handlers.iter_mut().find(|h| h.id == id) {
                entry.enabled = enabled;
                return true;
            }
        }
        false
    }

    /// Publishes an event immediately, dispatching to all matching handlers.
    pub fn publish<E: Event>(&mut self, event: E) {
        if !self.enabled {
            return;
        }

        self.stats_published += 1;
        let event_id = EventId::of::<E>();

        if let Some(handlers) = self.handlers.get(&event_id) {
            for entry in handlers {
                if entry.enabled {
                    (entry.handler)(&event);
                    self.stats_dispatched += 1;
                }
            }
        }
    }

    /// Queues an event for deferred processing.
    ///
    /// The event will be dispatched when [`process_queued`] is called.
    pub fn queue<E: Event>(&mut self, event: E) {
        if !self.enabled {
            return;
        }

        self.stats_queued += 1;
        self.queue.push(QueuedEvent {
            type_id: EventId::of::<E>(),
            event: Box::new(event),
        });
    }

    /// Processes all queued events, dispatching each to its handlers.
    ///
    /// Events are processed in FIFO order. The queue is drained completely.
    /// Returns the number of events processed.
    pub fn process_queued(&mut self) -> usize {
        if !self.enabled {
            self.queue.clear();
            return 0;
        }

        let queued: Vec<QueuedEvent> = self.queue.drain(..).collect();
        let count = queued.len();

        for queued_event in &queued {
            if let Some(handlers) = self.handlers.get(&queued_event.type_id) {
                for entry in handlers {
                    if entry.enabled {
                        (entry.handler)(queued_event.event.as_ref());
                        self.stats_dispatched += 1;
                    }
                }
            }
        }

        count
    }

    /// Returns the number of queued events.
    pub fn queued_count(&self) -> usize {
        self.queue.len()
    }

    /// Clears the event queue without processing.
    pub fn clear_queue(&mut self) {
        self.queue.clear();
    }

    /// Returns the number of registered handler types.
    pub fn handler_type_count(&self) -> usize {
        self.handlers.len()
    }

    /// Returns the total number of registered handlers across all types.
    pub fn total_handler_count(&self) -> usize {
        self.handlers.values().map(|v| v.len()).sum()
    }

    /// Returns the number of handlers for a specific event type.
    pub fn handler_count_for<E: Event>(&self) -> usize {
        let event_id = EventId::of::<E>();
        self.handlers.get(&event_id).map(|v| v.len()).unwrap_or(0)
    }

    /// Removes all handlers for a specific event type.
    pub fn clear_handlers_for<E: Event>(&mut self) {
        let event_id = EventId::of::<E>();
        self.handlers.remove(&event_id);
    }

    /// Removes all handlers and clears the queue.
    pub fn clear_all(&mut self) {
        self.handlers.clear();
        self.queue.clear();
    }

    /// Returns statistics about the event bus.
    pub fn stats(&self) -> EventBusStats {
        EventBusStats {
            total_published: self.stats_published,
            total_queued: self.stats_queued,
            total_dispatched: self.stats_dispatched,
            handler_types: self.handlers.len(),
            total_handlers: self.total_handler_count(),
            pending_queue: self.queue.len(),
        }
    }

    /// Resets statistics counters.
    pub fn reset_stats(&mut self) {
        self.stats_published = 0;
        self.stats_queued = 0;
        self.stats_dispatched = 0;
    }
}

// ---------------------------------------------------------------------------
// EventBusStats
// ---------------------------------------------------------------------------

/// Statistics about the event bus.
#[derive(Debug, Clone, Copy)]
pub struct EventBusStats {
    /// Total number of events published (immediate).
    pub total_published: u64,
    /// Total number of events queued (deferred).
    pub total_queued: u64,
    /// Total number of handler invocations.
    pub total_dispatched: u64,
    /// Number of distinct event types with handlers.
    pub handler_types: usize,
    /// Total number of handler registrations.
    pub total_handlers: usize,
    /// Number of events currently in the queue.
    pub pending_queue: usize,
}

// ---------------------------------------------------------------------------
// Common engine events
// ---------------------------------------------------------------------------

/// An entity was spawned into the world.
#[derive(Debug, Clone)]
pub struct EntitySpawned {
    /// Unique entity ID.
    pub entity_id: u64,
    /// Optional name of the entity.
    pub name: Option<String>,
    /// Archetype or prefab name that was instantiated.
    pub archetype: Option<String>,
}

impl Event for EntitySpawned {}

/// An entity was despawned from the world.
#[derive(Debug, Clone)]
pub struct EntityDespawned {
    /// Unique entity ID that was removed.
    pub entity_id: u64,
    /// Reason for despawning.
    pub reason: DespawnReason,
}

impl Event for EntityDespawned {}

/// Reason an entity was despawned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DespawnReason {
    /// Explicitly destroyed by game code.
    Destroyed,
    /// Removed due to scene unload.
    SceneUnload,
    /// Pooled back for reuse.
    Pooled,
    /// Entity lifetime expired.
    LifetimeExpired,
}

/// A scene was loaded and is ready.
#[derive(Debug, Clone)]
pub struct SceneLoaded {
    /// Path or identifier of the loaded scene.
    pub scene_path: String,
    /// Number of entities in the scene.
    pub entity_count: usize,
    /// Load time in seconds.
    pub load_time_secs: f32,
}

impl Event for SceneLoaded {}

/// A scene is about to be unloaded.
#[derive(Debug, Clone)]
pub struct SceneUnloading {
    /// Path or identifier of the scene being unloaded.
    pub scene_path: String,
}

impl Event for SceneUnloading {}

/// The window was resized.
#[derive(Debug, Clone, Copy)]
pub struct WindowResized {
    /// New width in pixels.
    pub width: u32,
    /// New height in pixels.
    pub height: u32,
    /// The scale factor (DPI).
    pub scale_factor: f32,
}

impl Event for WindowResized {}

/// A keyboard key was pressed.
#[derive(Debug, Clone, Copy)]
pub struct KeyPressed {
    /// Key code.
    pub key_code: u32,
    /// Whether this is a repeat event (key held down).
    pub is_repeat: bool,
    /// Modifier keys held during the press.
    pub modifiers: KeyModifiers,
}

impl Event for KeyPressed {}

/// A keyboard key was released.
#[derive(Debug, Clone, Copy)]
pub struct KeyReleased {
    /// Key code.
    pub key_code: u32,
    /// Modifier keys held during the release.
    pub modifiers: KeyModifiers,
}

impl Event for KeyReleased {}

/// Modifier key state.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyModifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub super_key: bool,
}

/// A physics simulation step has completed.
#[derive(Debug, Clone, Copy)]
pub struct PhysicsStep {
    /// Fixed timestep duration in seconds.
    pub delta_time: f32,
    /// Total simulation time in seconds.
    pub total_time: f64,
    /// Step number since simulation start.
    pub step_number: u64,
}

impl Event for PhysicsStep {}

/// A collision occurred between two entities.
#[derive(Debug, Clone)]
pub struct CollisionEvent {
    /// First entity in the collision.
    pub entity_a: u64,
    /// Second entity in the collision.
    pub entity_b: u64,
    /// Contact point in world space.
    pub contact_point: [f32; 3],
    /// Contact normal (from A to B).
    pub contact_normal: [f32; 3],
    /// Penetration depth.
    pub penetration: f32,
}

impl Event for CollisionEvent {}

/// Application lifecycle event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppLifecycleEvent {
    /// Application is about to pause (e.g. lost focus, mobile background).
    Suspending,
    /// Application is resuming from pause.
    Resuming,
    /// Application is about to quit.
    Quitting,
}

impl Event for AppLifecycleEvent {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_subscribe_and_publish() {
        let mut bus = EventBus::new();
        let received = Arc::new(Mutex::new(Vec::new()));
        let r = received.clone();

        bus.subscribe::<WindowResized>(move |event| {
            r.lock().unwrap().push((event.width, event.height));
        });

        bus.publish(WindowResized {
            width: 1920,
            height: 1080,
            scale_factor: 1.0,
        });

        let events = received.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], (1920, 1080));
    }

    #[test]
    fn test_multiple_subscribers() {
        let mut bus = EventBus::new();
        let count = Arc::new(Mutex::new(0u32));

        let c1 = count.clone();
        bus.subscribe::<PhysicsStep>(move |_| {
            *c1.lock().unwrap() += 1;
        });

        let c2 = count.clone();
        bus.subscribe::<PhysicsStep>(move |_| {
            *c2.lock().unwrap() += 10;
        });

        bus.publish(PhysicsStep {
            delta_time: 1.0 / 60.0,
            total_time: 0.0,
            step_number: 0,
        });

        assert_eq!(*count.lock().unwrap(), 11);
    }

    #[test]
    fn test_unsubscribe() {
        let mut bus = EventBus::new();
        let count = Arc::new(Mutex::new(0u32));
        let c = count.clone();

        let sub_id = bus.subscribe::<PhysicsStep>(move |_| {
            *c.lock().unwrap() += 1;
        });

        bus.publish(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.0,
            step_number: 0,
        });
        assert_eq!(*count.lock().unwrap(), 1);

        assert!(bus.unsubscribe(sub_id));
        bus.publish(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.016,
            step_number: 1,
        });
        assert_eq!(*count.lock().unwrap(), 1); // No increment
    }

    #[test]
    fn test_queued_events() {
        let mut bus = EventBus::new();
        let received = Arc::new(Mutex::new(Vec::new()));
        let r = received.clone();

        bus.subscribe::<EntitySpawned>(move |event| {
            r.lock().unwrap().push(event.entity_id);
        });

        bus.queue(EntitySpawned {
            entity_id: 1,
            name: None,
            archetype: None,
        });
        bus.queue(EntitySpawned {
            entity_id: 2,
            name: Some("player".to_string()),
            archetype: None,
        });

        // Not yet dispatched
        assert!(received.lock().unwrap().is_empty());
        assert_eq!(bus.queued_count(), 2);

        let processed = bus.process_queued();
        assert_eq!(processed, 2);
        assert_eq!(bus.queued_count(), 0);

        let ids = received.lock().unwrap();
        assert_eq!(*ids, vec![1, 2]);
    }

    #[test]
    fn test_disabled_bus() {
        let mut bus = EventBus::new();
        let count = Arc::new(Mutex::new(0u32));
        let c = count.clone();

        bus.subscribe::<PhysicsStep>(move |_| {
            *c.lock().unwrap() += 1;
        });

        bus.set_enabled(false);
        bus.publish(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.0,
            step_number: 0,
        });
        assert_eq!(*count.lock().unwrap(), 0);

        bus.set_enabled(true);
        bus.publish(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.016,
            step_number: 1,
        });
        assert_eq!(*count.lock().unwrap(), 1);
    }

    #[test]
    fn test_type_isolation() {
        let mut bus = EventBus::new();
        let count = Arc::new(Mutex::new(0u32));
        let c = count.clone();

        bus.subscribe::<WindowResized>(move |_| {
            *c.lock().unwrap() += 1;
        });

        // Publishing a different event type should not trigger the handler
        bus.publish(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.0,
            step_number: 0,
        });
        assert_eq!(*count.lock().unwrap(), 0);

        bus.publish(WindowResized {
            width: 800,
            height: 600,
            scale_factor: 1.0,
        });
        assert_eq!(*count.lock().unwrap(), 1);
    }

    #[test]
    fn test_event_id() {
        let id1 = EventId::of::<WindowResized>();
        let id2 = EventId::of::<WindowResized>();
        let id3 = EventId::of::<PhysicsStep>();

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_handler_count() {
        let mut bus = EventBus::new();
        assert_eq!(bus.total_handler_count(), 0);

        bus.subscribe::<WindowResized>(|_| {});
        bus.subscribe::<WindowResized>(|_| {});
        bus.subscribe::<PhysicsStep>(|_| {});

        assert_eq!(bus.handler_count_for::<WindowResized>(), 2);
        assert_eq!(bus.handler_count_for::<PhysicsStep>(), 1);
        assert_eq!(bus.total_handler_count(), 3);
        assert_eq!(bus.handler_type_count(), 2);
    }

    #[test]
    fn test_clear_handlers_for_type() {
        let mut bus = EventBus::new();
        bus.subscribe::<WindowResized>(|_| {});
        bus.subscribe::<WindowResized>(|_| {});
        bus.subscribe::<PhysicsStep>(|_| {});

        bus.clear_handlers_for::<WindowResized>();
        assert_eq!(bus.handler_count_for::<WindowResized>(), 0);
        assert_eq!(bus.handler_count_for::<PhysicsStep>(), 1);
    }

    #[test]
    fn test_stats() {
        let mut bus = EventBus::new();
        bus.subscribe::<PhysicsStep>(|_| {});
        bus.subscribe::<PhysicsStep>(|_| {});

        bus.publish(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.0,
            step_number: 0,
        });
        bus.queue(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.016,
            step_number: 1,
        });
        bus.process_queued();

        let stats = bus.stats();
        assert_eq!(stats.total_published, 1);
        assert_eq!(stats.total_queued, 1);
        assert_eq!(stats.total_dispatched, 4); // 2 handlers * 2 events
        assert_eq!(stats.total_handlers, 2);
    }

    #[test]
    fn test_subscriber_enable_disable() {
        let mut bus = EventBus::new();
        let count = Arc::new(Mutex::new(0u32));
        let c = count.clone();

        let sub_id = bus.subscribe::<PhysicsStep>(move |_| {
            *c.lock().unwrap() += 1;
        });

        bus.set_subscriber_enabled(sub_id, false);
        bus.publish(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.0,
            step_number: 0,
        });
        assert_eq!(*count.lock().unwrap(), 0);

        bus.set_subscriber_enabled(sub_id, true);
        bus.publish(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.016,
            step_number: 1,
        });
        assert_eq!(*count.lock().unwrap(), 1);
    }

    #[test]
    fn test_priority_ordering() {
        let mut bus = EventBus::new();
        let order = Arc::new(Mutex::new(Vec::new()));

        let o1 = order.clone();
        bus.subscribe_with_options::<PhysicsStep>(
            move |_| o1.lock().unwrap().push(1),
            "low",
            0,
        );

        let o2 = order.clone();
        bus.subscribe_with_options::<PhysicsStep>(
            move |_| o2.lock().unwrap().push(2),
            "high",
            10,
        );

        let o3 = order.clone();
        bus.subscribe_with_options::<PhysicsStep>(
            move |_| o3.lock().unwrap().push(3),
            "medium",
            5,
        );

        bus.publish(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.0,
            step_number: 0,
        });

        let result = order.lock().unwrap();
        assert_eq!(*result, vec![2, 3, 1]); // high, medium, low
    }

    #[test]
    fn test_clear_all() {
        let mut bus = EventBus::new();
        bus.subscribe::<PhysicsStep>(|_| {});
        bus.queue(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.0,
            step_number: 0,
        });

        bus.clear_all();
        assert_eq!(bus.total_handler_count(), 0);
        assert_eq!(bus.queued_count(), 0);
    }

    #[test]
    fn test_common_events_debug() {
        // Verify all common events implement Debug and Event
        let e1 = EntitySpawned {
            entity_id: 1,
            name: Some("test".into()),
            archetype: None,
        };
        let _ = format!("{:?}", e1);
        assert!(e1.event_name().contains("EntitySpawned"));

        let e2 = EntityDespawned {
            entity_id: 1,
            reason: DespawnReason::Destroyed,
        };
        let _ = format!("{:?}", e2);

        let e3 = SceneLoaded {
            scene_path: "test.scene".into(),
            entity_count: 100,
            load_time_secs: 0.5,
        };
        let _ = format!("{:?}", e3);
    }

    #[test]
    fn test_clear_queue() {
        let mut bus = EventBus::new();
        bus.queue(PhysicsStep {
            delta_time: 0.016,
            total_time: 0.0,
            step_number: 0,
        });
        assert_eq!(bus.queued_count(), 1);
        bus.clear_queue();
        assert_eq!(bus.queued_count(), 0);
    }
}
