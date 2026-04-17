//! Signal-slot pattern implementation for the Genovo engine.
//!
//! Provides a type-safe signal-slot mechanism inspired by Qt's signal-slot
//! system, enabling decoupled communication between engine subsystems.
//!
//! # Features
//!
//! - `Signal<T>` that can connect to multiple receivers
//! - Disconnect by connection ID
//! - Emit to all connected slots
//! - One-shot connections (auto-disconnect after first emit)
//! - Weak connections (auto-disconnect when receiver drops)
//! - Signal-to-signal forwarding
//! - Connection counting and statistics
//!
//! # Example
//!
//! ```ignore
//! let mut signal = Signal::new();
//! let conn = signal.connect(|value: &i32| {
//!     println!("Received: {}", value);
//! });
//! signal.emit(&42); // Prints "Received: 42"
//! signal.disconnect(conn);
//! signal.emit(&100); // No output
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

// ---------------------------------------------------------------------------
// ConnectionId
// ---------------------------------------------------------------------------

/// Unique identifier for a signal-slot connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId(u64);

impl ConnectionId {
    /// Returns the raw numeric value.
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Connection({})", self.0)
    }
}

static NEXT_CONNECTION_ID: AtomicU64 = AtomicU64::new(1);

fn alloc_connection_id() -> ConnectionId {
    ConnectionId(NEXT_CONNECTION_ID.fetch_add(1, Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// ConnectionType
// ---------------------------------------------------------------------------

/// The type of a connection, controlling its lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionType {
    /// A persistent connection that stays active until explicitly disconnected.
    Persistent,
    /// A one-shot connection that auto-disconnects after the first emission.
    OneShot,
    /// A weak connection that auto-disconnects when the associated guard drops.
    Weak,
}

impl fmt::Display for ConnectionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConnectionType::Persistent => write!(f, "Persistent"),
            ConnectionType::OneShot => write!(f, "OneShot"),
            ConnectionType::Weak => write!(f, "Weak"),
        }
    }
}

// ---------------------------------------------------------------------------
// Connection Guard (for weak connections)
// ---------------------------------------------------------------------------

/// A guard object that keeps a weak connection alive. When all clones of
/// this guard are dropped, the associated weak connection is automatically
/// disconnected.
#[derive(Debug, Clone)]
pub struct ConnectionGuard {
    alive: Arc<AtomicBool>,
    connection_id: ConnectionId,
}

impl ConnectionGuard {
    fn new(connection_id: ConnectionId) -> Self {
        Self {
            alive: Arc::new(AtomicBool::new(true)),
            connection_id,
        }
    }

    /// Returns the connection ID this guard protects.
    pub fn connection_id(&self) -> ConnectionId {
        self.connection_id
    }

    /// Returns `true` if the connection is still alive.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Manually disconnect (same as dropping the guard).
    pub fn disconnect(&self) {
        self.alive.store(false, Ordering::SeqCst);
    }

    fn weak_ref(&self) -> Weak<AtomicBool> {
        Arc::downgrade(&self.alive)
    }
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        // If this is the last strong reference, mark the connection as dead.
        if Arc::strong_count(&self.alive) <= 1 {
            self.alive.store(false, Ordering::SeqCst);
        }
    }
}

// ---------------------------------------------------------------------------
// Slot
// ---------------------------------------------------------------------------

/// Internal representation of a connected slot.
struct SlotEntry<T: 'static> {
    id: ConnectionId,
    callback: Box<dyn Fn(&T) + Send + Sync>,
    connection_type: ConnectionType,
    /// For weak connections: a weak reference to the guard's alive flag.
    weak_alive: Option<Weak<AtomicBool>>,
    /// Whether this slot is currently enabled.
    enabled: bool,
    /// Number of times this slot has been invoked.
    invoke_count: u64,
    /// Priority (higher = called first).
    priority: i32,
}

impl<T: 'static> SlotEntry<T> {
    /// Returns `true` if this connection is still valid.
    fn is_alive(&self) -> bool {
        if !self.enabled {
            return false;
        }
        match self.connection_type {
            ConnectionType::Persistent => true,
            ConnectionType::OneShot => self.invoke_count == 0,
            ConnectionType::Weak => {
                if let Some(ref weak) = self.weak_alive {
                    weak.upgrade().map_or(false, |flag| flag.load(Ordering::SeqCst))
                } else {
                    true
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Signal
// ---------------------------------------------------------------------------

/// A signal that can emit values to connected slots.
///
/// Thread-safe: connections and emissions can happen from any thread.
pub struct Signal<T: 'static> {
    slots: Vec<SlotEntry<T>>,
    /// Whether emissions are currently blocked.
    blocked: bool,
    /// Total number of emissions.
    emit_count: u64,
    /// Name for debugging.
    name: Option<String>,
}

impl<T: 'static> Signal<T> {
    /// Create a new, empty signal.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            blocked: false,
            emit_count: 0,
            name: None,
        }
    }

    /// Create a new signal with a debug name.
    pub fn named(name: &str) -> Self {
        Self {
            slots: Vec::new(),
            blocked: false,
            emit_count: 0,
            name: Some(name.to_string()),
        }
    }

    /// Connect a slot function. Returns the connection ID.
    pub fn connect<F>(&mut self, callback: F) -> ConnectionId
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        let id = alloc_connection_id();
        self.slots.push(SlotEntry {
            id,
            callback: Box::new(callback),
            connection_type: ConnectionType::Persistent,
            weak_alive: None,
            enabled: true,
            invoke_count: 0,
            priority: 0,
        });
        id
    }

    /// Connect a slot with a specific priority (higher = called first).
    pub fn connect_with_priority<F>(&mut self, priority: i32, callback: F) -> ConnectionId
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        let id = alloc_connection_id();
        self.slots.push(SlotEntry {
            id,
            callback: Box::new(callback),
            connection_type: ConnectionType::Persistent,
            weak_alive: None,
            enabled: true,
            invoke_count: 0,
            priority,
        });
        // Sort by priority (descending).
        self.slots.sort_by(|a, b| b.priority.cmp(&a.priority));
        id
    }

    /// Connect a one-shot slot that auto-disconnects after the first emission.
    pub fn connect_once<F>(&mut self, callback: F) -> ConnectionId
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        let id = alloc_connection_id();
        self.slots.push(SlotEntry {
            id,
            callback: Box::new(callback),
            connection_type: ConnectionType::OneShot,
            weak_alive: None,
            enabled: true,
            invoke_count: 0,
            priority: 0,
        });
        id
    }

    /// Connect a weak slot. Returns `(ConnectionId, ConnectionGuard)`.
    /// The slot auto-disconnects when the guard is dropped.
    pub fn connect_weak<F>(&mut self, callback: F) -> (ConnectionId, ConnectionGuard)
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        let id = alloc_connection_id();
        let guard = ConnectionGuard::new(id);
        let weak = guard.weak_ref();
        self.slots.push(SlotEntry {
            id,
            callback: Box::new(callback),
            connection_type: ConnectionType::Weak,
            weak_alive: Some(weak),
            enabled: true,
            invoke_count: 0,
            priority: 0,
        });
        (id, guard)
    }

    /// Disconnect a slot by its connection ID.
    pub fn disconnect(&mut self, id: ConnectionId) -> bool {
        let before = self.slots.len();
        self.slots.retain(|s| s.id != id);
        self.slots.len() < before
    }

    /// Disconnect all slots.
    pub fn disconnect_all(&mut self) {
        self.slots.clear();
    }

    /// Enable or disable a specific connection.
    pub fn set_enabled(&mut self, id: ConnectionId, enabled: bool) -> bool {
        for slot in &mut self.slots {
            if slot.id == id {
                slot.enabled = enabled;
                return true;
            }
        }
        false
    }

    /// Emit a value to all connected slots.
    ///
    /// One-shot connections are removed after invocation.
    /// Dead weak connections are cleaned up.
    pub fn emit(&mut self, value: &T) {
        if self.blocked {
            return;
        }

        self.emit_count += 1;

        // Invoke all alive slots.
        for slot in &mut self.slots {
            if slot.is_alive() {
                (slot.callback)(value);
                slot.invoke_count += 1;
            }
        }

        // Garbage-collect dead connections.
        self.slots.retain(|s| s.is_alive());
    }

    /// Emit a value and collect results from slots that return values.
    /// This variant uses a mutable accumulator closure.
    pub fn emit_with_accumulator<A, F>(&mut self, value: &T, initial: A, mut accumulator: F) -> A
    where
        F: FnMut(A, &T) -> A,
    {
        if self.blocked {
            return initial;
        }

        self.emit_count += 1;
        let mut acc = initial;

        for slot in &mut self.slots {
            if slot.is_alive() {
                (slot.callback)(value);
                slot.invoke_count += 1;
                acc = accumulator(acc, value);
            }
        }

        self.slots.retain(|s| s.is_alive());
        acc
    }

    /// Block all emissions.
    pub fn block(&mut self) {
        self.blocked = true;
    }

    /// Unblock emissions.
    pub fn unblock(&mut self) {
        self.blocked = false;
    }

    /// Returns `true` if emissions are blocked.
    pub fn is_blocked(&self) -> bool {
        self.blocked
    }

    /// Returns the number of currently connected slots.
    pub fn connection_count(&self) -> usize {
        self.slots.len()
    }

    /// Returns the number of *alive* (non-dead) connections.
    pub fn alive_connection_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_alive()).count()
    }

    /// Returns the total number of emissions.
    pub fn emit_count(&self) -> u64 {
        self.emit_count
    }

    /// Returns `true` if no slots are connected.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Returns the signal's debug name, if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the invoke count for a specific connection.
    pub fn invoke_count(&self, id: ConnectionId) -> Option<u64> {
        self.slots.iter().find(|s| s.id == id).map(|s| s.invoke_count)
    }

    /// Get all connection IDs.
    pub fn connection_ids(&self) -> Vec<ConnectionId> {
        self.slots.iter().map(|s| s.id).collect()
    }

    /// Clean up dead weak connections without emitting.
    pub fn gc(&mut self) {
        self.slots.retain(|s| s.is_alive());
    }
}

impl<T: 'static> Default for Signal<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> fmt::Debug for Signal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Signal")
            .field("name", &self.name)
            .field("connections", &self.slots.len())
            .field("blocked", &self.blocked)
            .field("emit_count", &self.emit_count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Signal Forwarding
// ---------------------------------------------------------------------------

/// Forward emissions from one signal to another.
///
/// Returns a `ConnectionId` that can be used to disconnect the forwarding.
pub fn forward_signal<T: Clone + 'static>(
    source: &mut Signal<T>,
    target: Arc<Mutex<Signal<T>>>,
) -> ConnectionId {
    source.connect(move |value: &T| {
        let cloned = value.clone();
        if let Ok(mut sig) = target.lock() {
            sig.emit(&cloned);
        }
    })
}

// ---------------------------------------------------------------------------
// SignalMap (multiple named signals)
// ---------------------------------------------------------------------------

/// A map of named signals, useful for dynamic event systems.
pub struct SignalMap<T: 'static> {
    signals: HashMap<String, Signal<T>>,
}

impl<T: 'static> SignalMap<T> {
    /// Create an empty signal map.
    pub fn new() -> Self {
        Self {
            signals: HashMap::new(),
        }
    }

    /// Get or create a signal by name.
    pub fn signal(&mut self, name: &str) -> &mut Signal<T> {
        self.signals
            .entry(name.to_string())
            .or_insert_with(|| Signal::named(name))
    }

    /// Emit a value on a named signal (no-op if the signal doesn't exist).
    pub fn emit(&mut self, name: &str, value: &T) {
        if let Some(signal) = self.signals.get_mut(name) {
            signal.emit(value);
        }
    }

    /// Connect a slot to a named signal.
    pub fn connect<F>(&mut self, name: &str, callback: F) -> ConnectionId
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        self.signal(name).connect(callback)
    }

    /// Disconnect a slot from a named signal.
    pub fn disconnect(&mut self, name: &str, id: ConnectionId) -> bool {
        if let Some(signal) = self.signals.get_mut(name) {
            signal.disconnect(id)
        } else {
            false
        }
    }

    /// Remove a named signal entirely.
    pub fn remove_signal(&mut self, name: &str) -> bool {
        self.signals.remove(name).is_some()
    }

    /// Returns the names of all signals.
    pub fn signal_names(&self) -> Vec<&str> {
        self.signals.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the total number of connections across all signals.
    pub fn total_connections(&self) -> usize {
        self.signals.values().map(|s| s.connection_count()).sum()
    }
}

impl<T: 'static> Default for SignalMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> fmt::Debug for SignalMap<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SignalMap")
            .field("signal_count", &self.signals.len())
            .field("total_connections", &self.total_connections())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Threadsafe Signal wrapper
// ---------------------------------------------------------------------------

/// A thread-safe wrapper around `Signal<T>`, using a `Mutex` internally.
pub struct SharedSignal<T: 'static> {
    inner: Arc<Mutex<Signal<T>>>,
}

impl<T: 'static> SharedSignal<T> {
    /// Create a new shared signal.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Signal::new())),
        }
    }

    /// Create a named shared signal.
    pub fn named(name: &str) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Signal::named(name))),
        }
    }

    /// Connect a slot.
    pub fn connect<F>(&self, callback: F) -> ConnectionId
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        self.inner.lock().unwrap().connect(callback)
    }

    /// Connect a one-shot slot.
    pub fn connect_once<F>(&self, callback: F) -> ConnectionId
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        self.inner.lock().unwrap().connect_once(callback)
    }

    /// Disconnect a slot.
    pub fn disconnect(&self, id: ConnectionId) -> bool {
        self.inner.lock().unwrap().disconnect(id)
    }

    /// Emit a value.
    pub fn emit(&self, value: &T) {
        self.inner.lock().unwrap().emit(value);
    }

    /// Block emissions.
    pub fn block(&self) {
        self.inner.lock().unwrap().block();
    }

    /// Unblock emissions.
    pub fn unblock(&self) {
        self.inner.lock().unwrap().unblock();
    }

    /// Connection count.
    pub fn connection_count(&self) -> usize {
        self.inner.lock().unwrap().connection_count()
    }

    /// Get a clone of the inner Arc<Mutex<Signal<T>>> for forwarding.
    pub fn inner_arc(&self) -> Arc<Mutex<Signal<T>>> {
        self.inner.clone()
    }
}

impl<T: 'static> Clone for SharedSignal<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: 'static> Default for SharedSignal<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> fmt::Debug for SharedSignal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedSignal")
            .field("connections", &self.connection_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Connection Handle (RAII disconnect)
// ---------------------------------------------------------------------------

/// An RAII handle that automatically disconnects a signal connection when
/// dropped. Useful for ensuring cleanup.
pub struct ScopedConnection<T: 'static> {
    signal: Arc<Mutex<Signal<T>>>,
    id: ConnectionId,
}

impl<T: 'static> ScopedConnection<T> {
    /// Create a new scoped connection.
    pub fn new(signal: Arc<Mutex<Signal<T>>>, id: ConnectionId) -> Self {
        Self { signal, id }
    }

    /// Returns the connection ID.
    pub fn id(&self) -> ConnectionId {
        self.id
    }

    /// Manually disconnect (same as drop).
    pub fn disconnect(self) {
        // Drop will handle it.
    }
}

impl<T: 'static> Drop for ScopedConnection<T> {
    fn drop(&mut self) {
        if let Ok(mut signal) = self.signal.lock() {
            signal.disconnect(self.id);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI32;

    #[test]
    fn test_basic_connect_emit() {
        let mut signal = Signal::<i32>::new();
        let counter = Arc::new(AtomicI32::new(0));
        let counter2 = counter.clone();

        signal.connect(move |value: &i32| {
            counter2.fetch_add(*value, Ordering::SeqCst);
        });

        signal.emit(&10);
        signal.emit(&20);
        assert_eq!(counter.load(Ordering::SeqCst), 30);
    }

    #[test]
    fn test_disconnect() {
        let mut signal = Signal::<i32>::new();
        let counter = Arc::new(AtomicI32::new(0));
        let counter2 = counter.clone();

        let id = signal.connect(move |_: &i32| {
            counter2.fetch_add(1, Ordering::SeqCst);
        });

        signal.emit(&0);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        signal.disconnect(id);
        signal.emit(&0);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_one_shot() {
        let mut signal = Signal::<()>::new();
        let counter = Arc::new(AtomicI32::new(0));
        let counter2 = counter.clone();

        signal.connect_once(move |_: &()| {
            counter2.fetch_add(1, Ordering::SeqCst);
        });

        signal.emit(&());
        signal.emit(&());
        signal.emit(&());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_weak_connection() {
        let mut signal = Signal::<i32>::new();
        let counter = Arc::new(AtomicI32::new(0));
        let counter2 = counter.clone();

        let (id, guard) = signal.connect_weak(move |_: &i32| {
            counter2.fetch_add(1, Ordering::SeqCst);
        });

        signal.emit(&0);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        drop(guard);
        signal.emit(&0);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_blocked_signal() {
        let mut signal = Signal::<i32>::new();
        let counter = Arc::new(AtomicI32::new(0));
        let counter2 = counter.clone();

        signal.connect(move |_: &i32| {
            counter2.fetch_add(1, Ordering::SeqCst);
        });

        signal.block();
        signal.emit(&0);
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        signal.unblock();
        signal.emit(&0);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_priority_ordering() {
        let mut signal = Signal::<()>::new();
        let order = Arc::new(Mutex::new(Vec::new()));

        let order1 = order.clone();
        signal.connect_with_priority(10, move |_| {
            order1.lock().unwrap().push("high");
        });

        let order2 = order.clone();
        signal.connect_with_priority(1, move |_| {
            order2.lock().unwrap().push("low");
        });

        let order3 = order.clone();
        signal.connect_with_priority(5, move |_| {
            order3.lock().unwrap().push("mid");
        });

        signal.emit(&());
        let result = order.lock().unwrap().clone();
        assert_eq!(result, vec!["high", "mid", "low"]);
    }

    #[test]
    fn test_signal_map() {
        let mut map = SignalMap::<String>::new();
        let received = Arc::new(Mutex::new(Vec::new()));
        let received2 = received.clone();

        map.connect("player_died", move |msg: &String| {
            received2.lock().unwrap().push(msg.clone());
        });

        map.emit("player_died", &"Player 1".to_string());
        map.emit("player_died", &"Player 2".to_string());
        map.emit("unknown_signal", &"ignored".to_string());

        let result = received.lock().unwrap().clone();
        assert_eq!(result, vec!["Player 1", "Player 2"]);
    }

    #[test]
    fn test_shared_signal() {
        let signal = SharedSignal::<i32>::new();
        let counter = Arc::new(AtomicI32::new(0));
        let counter2 = counter.clone();

        signal.connect(move |val: &i32| {
            counter2.fetch_add(*val, Ordering::SeqCst);
        });

        signal.emit(&5);
        signal.emit(&3);
        assert_eq!(counter.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn test_connection_count() {
        let mut signal = Signal::<()>::new();
        assert_eq!(signal.connection_count(), 0);

        let id1 = signal.connect(|_| {});
        let id2 = signal.connect(|_| {});
        assert_eq!(signal.connection_count(), 2);

        signal.disconnect(id1);
        assert_eq!(signal.connection_count(), 1);

        signal.disconnect_all();
        assert_eq!(signal.connection_count(), 0);
    }
}
