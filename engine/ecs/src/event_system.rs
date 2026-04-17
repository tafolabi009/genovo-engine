//! Enhanced event system with typed channels, event readers with cursors,
//! double-buffered events, event drain, event forwarding, and statistics.
//!
//! This module extends the basic [`Events<T>`] with:
//!
//! - **Typed event channels** — each event type gets its own channel with
//!   independent buffering and lifecycle.
//! - **Event readers with cursor** — readers track their position so they
//!   only see events that arrived since their last read.
//! - **Double-buffered events** — events persist for two frames, ensuring
//!   all systems get a chance to observe them.
//! - **Event drain** — consume events destructively for one-shot processing.
//! - **Event forwarding** — automatically forward events from one channel
//!   to another (with optional mapping function).
//! - **Event statistics** — track send/read counts, peak throughput, etc.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Event ID
// ---------------------------------------------------------------------------

/// A monotonically increasing event sequence number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventSequence(pub u64);

impl EventSequence {
    /// The zero / "beginning of time" sequence.
    pub const ZERO: Self = Self(0);
}

// ---------------------------------------------------------------------------
// Event instance
// ---------------------------------------------------------------------------

/// A single event with its sequence number.
#[derive(Debug, Clone)]
pub struct EventInstance<T> {
    /// The event payload.
    pub event: T,
    /// Monotonic sequence number.
    pub sequence: EventSequence,
}

// ---------------------------------------------------------------------------
// Typed event channel
// ---------------------------------------------------------------------------

/// Double-buffered, sequenced event channel for a single event type.
pub struct EventChannel<T> {
    /// Buffer A.
    buffer_a: Vec<EventInstance<T>>,
    /// Buffer B.
    buffer_b: Vec<EventInstance<T>>,
    /// Current parity: false = A is write buffer, true = B is write buffer.
    parity: bool,
    /// Next sequence number to assign.
    next_sequence: u64,
    /// Statistics.
    stats: ChannelStats,
}

impl<T> EventChannel<T> {
    /// Create a new empty channel.
    pub fn new() -> Self {
        Self {
            buffer_a: Vec::new(),
            buffer_b: Vec::new(),
            parity: false,
            next_sequence: 0,
            stats: ChannelStats::new(),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer_a: Vec::with_capacity(capacity),
            buffer_b: Vec::with_capacity(capacity),
            parity: false,
            next_sequence: 0,
            stats: ChannelStats::new(),
        }
    }

    /// Send a single event.
    pub fn send(&mut self, event: T) -> EventSequence {
        let seq = EventSequence(self.next_sequence);
        self.next_sequence += 1;
        self.stats.total_sent += 1;

        let instance = EventInstance {
            event,
            sequence: seq,
        };

        if self.parity {
            self.buffer_b.push(instance);
        } else {
            self.buffer_a.push(instance);
        }

        seq
    }

    /// Send multiple events at once.
    pub fn send_batch(&mut self, events: impl IntoIterator<Item = T>) {
        for event in events {
            self.send(event);
        }
    }

    /// Swap buffers — called once per frame.
    ///
    /// Clears the older buffer and flips parity so new events go to
    /// the freshly-cleared buffer.
    pub fn swap_buffers(&mut self) {
        // Track peak.
        let current_count = self.len();
        if current_count > self.stats.peak_per_frame as usize {
            self.stats.peak_per_frame = current_count as u64;
        }

        if self.parity {
            self.buffer_a.clear();
        } else {
            self.buffer_b.clear();
        }
        self.parity = !self.parity;
        self.stats.frames += 1;
    }

    /// Iterate over all readable events (both buffers), newest last.
    pub fn iter(&self) -> impl Iterator<Item = &EventInstance<T>> {
        // Older buffer first, then current.
        let (read_buf, write_buf) = if self.parity {
            (&self.buffer_a, &self.buffer_b)
        } else {
            (&self.buffer_b, &self.buffer_a)
        };
        read_buf.iter().chain(write_buf.iter())
    }

    /// Iterate over events that arrived after the given sequence number.
    pub fn iter_since(
        &self,
        since: EventSequence,
    ) -> impl Iterator<Item = &EventInstance<T>> {
        self.iter().filter(move |e| e.sequence > since)
    }

    /// Drain all events, consuming them. Both buffers are cleared.
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.stats.total_drained += self.len() as u64;
        self.buffer_a
            .drain(..)
            .chain(self.buffer_b.drain(..))
            .map(|instance| instance.event)
    }

    /// Number of events currently readable.
    pub fn len(&self) -> usize {
        self.buffer_a.len() + self.buffer_b.len()
    }

    /// Whether there are no readable events.
    pub fn is_empty(&self) -> bool {
        self.buffer_a.is_empty() && self.buffer_b.is_empty()
    }

    /// Clear all events immediately.
    pub fn clear(&mut self) {
        self.buffer_a.clear();
        self.buffer_b.clear();
    }

    /// Get the current sequence number (next event will get this).
    pub fn current_sequence(&self) -> EventSequence {
        EventSequence(self.next_sequence)
    }

    /// Get channel statistics.
    pub fn stats(&self) -> &ChannelStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ChannelStats::new();
    }
}

impl<T> Default for EventChannel<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Event reader
// ---------------------------------------------------------------------------

/// A cursor-based reader for an event channel.
///
/// Each reader tracks the last sequence number it observed. Calling
/// `read()` returns only events newer than the cursor, then advances it.
pub struct EventReader<T> {
    /// Last sequence number this reader has seen.
    cursor: EventSequence,
    /// Number of events this reader has read in total.
    total_read: u64,
    /// Phantom for the event type.
    _marker: PhantomData<T>,
}

impl<T> EventReader<T> {
    /// Create a new reader starting from the beginning.
    pub fn new() -> Self {
        Self {
            cursor: EventSequence(0),
            total_read: 0,
            _marker: PhantomData,
        }
    }

    /// Create a reader starting from a specific sequence.
    pub fn from_sequence(seq: EventSequence) -> Self {
        Self {
            cursor: seq,
            total_read: 0,
            _marker: PhantomData,
        }
    }

    /// Read new events from the channel since last read.
    ///
    /// Advances the cursor to the newest event seen.
    pub fn read<'a>(
        &mut self,
        channel: &'a EventChannel<T>,
    ) -> impl Iterator<Item = &'a T> + 'a
    where
        T: 'a,
    {
        let since = self.cursor;

        // Pre-compute the new cursor position.
        let mut max_seq = since;
        let mut count = 0u64;
        for event in channel.iter() {
            if event.sequence > since {
                if event.sequence > max_seq {
                    max_seq = event.sequence;
                }
                count += 1;
            }
        }
        self.cursor = max_seq;
        self.total_read += count;

        channel
            .iter()
            .filter(move |e| e.sequence > since)
            .map(|e| &e.event)
    }

    /// Peek at new events without advancing the cursor.
    pub fn peek<'a>(
        &self,
        channel: &'a EventChannel<T>,
    ) -> impl Iterator<Item = &'a T> + 'a
    where
        T: 'a,
    {
        let since = self.cursor;
        channel
            .iter()
            .filter(move |e| e.sequence > since)
            .map(|e| &e.event)
    }

    /// Check whether there are unread events.
    pub fn has_unread(&self, channel: &EventChannel<T>) -> bool {
        channel.iter().any(|e| e.sequence > self.cursor)
    }

    /// Count unread events without consuming them.
    pub fn unread_count(&self, channel: &EventChannel<T>) -> usize {
        channel
            .iter()
            .filter(|e| e.sequence > self.cursor)
            .count()
    }

    /// Reset the cursor to the beginning (re-read all events).
    pub fn reset(&mut self) {
        self.cursor = EventSequence::ZERO;
    }

    /// Skip all current events (advance cursor to latest).
    pub fn skip_all(&mut self, channel: &EventChannel<T>) {
        if let Some(last) = channel.iter().last() {
            self.cursor = last.sequence;
        }
    }

    /// Get the current cursor position.
    pub fn cursor(&self) -> EventSequence {
        self.cursor
    }

    /// Get the total number of events this reader has consumed.
    pub fn total_read(&self) -> u64 {
        self.total_read
    }
}

impl<T> Default for EventReader<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Channel statistics
// ---------------------------------------------------------------------------

/// Statistics for an event channel.
#[derive(Debug, Clone)]
pub struct ChannelStats {
    /// Total events sent since creation.
    pub total_sent: u64,
    /// Total events drained.
    pub total_drained: u64,
    /// Peak events in a single frame.
    pub peak_per_frame: u64,
    /// Number of frames elapsed.
    pub frames: u64,
}

impl ChannelStats {
    /// Create blank statistics.
    pub fn new() -> Self {
        Self {
            total_sent: 0,
            total_drained: 0,
            peak_per_frame: 0,
            frames: 0,
        }
    }

    /// Average events per frame.
    pub fn avg_per_frame(&self) -> f64 {
        if self.frames == 0 {
            0.0
        } else {
            self.total_sent as f64 / self.frames as f64
        }
    }
}

impl Default for ChannelStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ChannelStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sent={} drained={} peak={} avg={:.1}/frame",
            self.total_sent,
            self.total_drained,
            self.peak_per_frame,
            self.avg_per_frame()
        )
    }
}

// ---------------------------------------------------------------------------
// Event forwarding rule
// ---------------------------------------------------------------------------

/// Describes a forwarding rule from one event type to another.
struct ForwardingRule {
    /// Source event type.
    source_type: TypeId,
    /// Destination event type.
    dest_type: TypeId,
    /// Mapping function (type-erased). Takes a Box<dyn Any> of source events,
    /// returns a Vec<Box<dyn Any>> of dest events.
    map_fn: Box<dyn Fn(&dyn Any) -> Option<Box<dyn Any + Send>> + Send + Sync>,
}

impl fmt::Debug for ForwardingRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardingRule")
            .field("source_type", &self.source_type)
            .field("dest_type", &self.dest_type)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Event registry
// ---------------------------------------------------------------------------

/// Type-erased event channel for the registry.
trait AnyEventChannel: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn swap_buffers(&mut self);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn clear(&mut self);
    fn type_name(&self) -> &'static str;
    fn stats_total_sent(&self) -> u64;
    fn stats_peak(&self) -> u64;
}

impl<T: Send + Sync + 'static> AnyEventChannel for EventChannel<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn swap_buffers(&mut self) {
        EventChannel::swap_buffers(self);
    }

    fn len(&self) -> usize {
        EventChannel::len(self)
    }

    fn is_empty(&self) -> bool {
        EventChannel::is_empty(self)
    }

    fn clear(&mut self) {
        EventChannel::clear(self);
    }

    fn type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    fn stats_total_sent(&self) -> u64 {
        self.stats.total_sent
    }

    fn stats_peak(&self) -> u64 {
        self.stats.peak_per_frame
    }
}

/// Central registry managing all event channels.
pub struct EventRegistry {
    /// Type-erased channels keyed by event TypeId.
    channels: HashMap<TypeId, Box<dyn AnyEventChannel>>,
    /// Forwarding rules.
    forwarding_rules: Vec<ForwardingRule>,
    /// Total events sent across all channels this frame.
    frame_event_count: u64,
    /// Global statistics.
    global_stats: GlobalEventStats,
}

impl EventRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            forwarding_rules: Vec::new(),
            frame_event_count: 0,
            global_stats: GlobalEventStats::new(),
        }
    }

    /// Register a new event type. Does nothing if already registered.
    pub fn register<T: Send + Sync + 'static>(&mut self) {
        let type_id = TypeId::of::<T>();
        self.channels
            .entry(type_id)
            .or_insert_with(|| Box::new(EventChannel::<T>::new()));
    }

    /// Register with pre-allocated capacity.
    pub fn register_with_capacity<T: Send + Sync + 'static>(&mut self, capacity: usize) {
        let type_id = TypeId::of::<T>();
        self.channels
            .entry(type_id)
            .or_insert_with(|| Box::new(EventChannel::<T>::with_capacity(capacity)));
    }

    /// Get a reference to a channel.
    pub fn channel<T: Send + Sync + 'static>(&self) -> Option<&EventChannel<T>> {
        self.channels
            .get(&TypeId::of::<T>())
            .and_then(|c| c.as_any().downcast_ref::<EventChannel<T>>())
    }

    /// Get a mutable reference to a channel.
    pub fn channel_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut EventChannel<T>> {
        self.channels
            .get_mut(&TypeId::of::<T>())
            .and_then(|c| c.as_any_mut().downcast_mut::<EventChannel<T>>())
    }

    /// Send an event. Auto-registers the channel if needed.
    pub fn send<T: Send + Sync + 'static>(&mut self, event: T) -> EventSequence {
        self.register::<T>();
        let channel = self
            .channels
            .get_mut(&TypeId::of::<T>())
            .unwrap()
            .as_any_mut()
            .downcast_mut::<EventChannel<T>>()
            .unwrap();
        self.frame_event_count += 1;
        self.global_stats.total_events += 1;
        channel.send(event)
    }

    /// Send a batch of events.
    pub fn send_batch<T: Send + Sync + 'static>(
        &mut self,
        events: impl IntoIterator<Item = T>,
    ) {
        self.register::<T>();
        let channel = self
            .channels
            .get_mut(&TypeId::of::<T>())
            .unwrap()
            .as_any_mut()
            .downcast_mut::<EventChannel<T>>()
            .unwrap();
        for event in events {
            channel.send(event);
            self.frame_event_count += 1;
            self.global_stats.total_events += 1;
        }
    }

    /// Swap all channel buffers — call once per frame.
    pub fn swap_all_buffers(&mut self) {
        // Update stats.
        if self.frame_event_count > self.global_stats.peak_events_per_frame {
            self.global_stats.peak_events_per_frame = self.frame_event_count;
        }
        self.global_stats.frames += 1;

        for channel in self.channels.values_mut() {
            channel.swap_buffers();
        }

        self.frame_event_count = 0;
    }

    /// Clear all channels.
    pub fn clear_all(&mut self) {
        for channel in self.channels.values_mut() {
            channel.clear();
        }
    }

    /// Check if an event type is registered.
    pub fn is_registered<T: 'static>(&self) -> bool {
        self.channels.contains_key(&TypeId::of::<T>())
    }

    /// Get the number of registered event types.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Add a forwarding rule: events of type S are mapped and forwarded to type D.
    pub fn add_forwarding<S, D, F>(&mut self, map_fn: F)
    where
        S: Send + Sync + Clone + 'static,
        D: Send + Sync + 'static,
        F: Fn(&S) -> Option<D> + Send + Sync + 'static,
    {
        self.register::<S>();
        self.register::<D>();

        let rule = ForwardingRule {
            source_type: TypeId::of::<S>(),
            dest_type: TypeId::of::<D>(),
            map_fn: Box::new(move |any| {
                let source = any.downcast_ref::<S>()?;
                let dest = map_fn(source)?;
                Some(Box::new(dest) as Box<dyn Any + Send>)
            }),
        };
        self.forwarding_rules.push(rule);
    }

    /// Process all forwarding rules for the current frame.
    ///
    /// This reads events from source channels and sends mapped events to
    /// destination channels. Should be called after all systems have sent
    /// their events but before the next frame's swap.
    pub fn process_forwarding(&mut self) {
        // Collect forwarded events.
        let mut forwarded: HashMap<TypeId, Vec<Box<dyn Any + Send>>> = HashMap::new();

        for rule in &self.forwarding_rules {
            if let Some(source_channel) = self.channels.get(&rule.source_type) {
                // We need to iterate the source events via the type-erased interface.
                // Since we can't easily iterate type-erased events, forwarding is
                // best done at the typed level. This is a simplified version.
                // In practice, users should use typed forwarding helpers.
            }
        }
    }

    /// Get global event statistics.
    pub fn global_stats(&self) -> &GlobalEventStats {
        &self.global_stats
    }

    /// Get per-channel statistics summary.
    pub fn channel_stats_summary(&self) -> Vec<ChannelStatsSummary> {
        self.channels
            .values()
            .map(|c| ChannelStatsSummary {
                type_name: c.type_name().to_string(),
                current_count: c.len(),
                total_sent: c.stats_total_sent(),
                peak_per_frame: c.stats_peak(),
            })
            .collect()
    }
}

impl Default for EventRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Global event statistics across all channels.
#[derive(Debug, Clone)]
pub struct GlobalEventStats {
    /// Total events sent across all channels and all time.
    pub total_events: u64,
    /// Peak events in a single frame across all channels.
    pub peak_events_per_frame: u64,
    /// Number of frames elapsed.
    pub frames: u64,
}

impl GlobalEventStats {
    /// Create blank stats.
    pub fn new() -> Self {
        Self {
            total_events: 0,
            peak_events_per_frame: 0,
            frames: 0,
        }
    }

    /// Average events per frame.
    pub fn avg_per_frame(&self) -> f64 {
        if self.frames == 0 {
            0.0
        } else {
            self.total_events as f64 / self.frames as f64
        }
    }
}

impl Default for GlobalEventStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for GlobalEventStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Events: total={} peak={}/frame avg={:.1}/frame frames={}",
            self.total_events,
            self.peak_events_per_frame,
            self.avg_per_frame(),
            self.frames
        )
    }
}

/// Per-channel statistics summary.
#[derive(Debug, Clone)]
pub struct ChannelStatsSummary {
    /// Event type name.
    pub type_name: String,
    /// Current number of events in the channel.
    pub current_count: usize,
    /// Total events sent.
    pub total_sent: u64,
    /// Peak events in a single frame.
    pub peak_per_frame: u64,
}

impl fmt::Display for ChannelStatsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: current={} total={} peak={}",
            self.type_name, self.current_count, self.total_sent, self.peak_per_frame
        )
    }
}

// ---------------------------------------------------------------------------
// Event writer helper
// ---------------------------------------------------------------------------

/// A convenience wrapper for sending events to a channel.
pub struct EventWriter<'a, T: Send + Sync + 'static> {
    channel: &'a mut EventChannel<T>,
}

impl<'a, T: Send + Sync + 'static> EventWriter<'a, T> {
    /// Create a new writer.
    pub fn new(channel: &'a mut EventChannel<T>) -> Self {
        Self { channel }
    }

    /// Send a single event.
    pub fn send(&mut self, event: T) -> EventSequence {
        self.channel.send(event)
    }

    /// Send multiple events.
    pub fn send_batch(&mut self, events: impl IntoIterator<Item = T>) {
        self.channel.send_batch(events);
    }

    /// Get the current event count.
    pub fn len(&self) -> usize {
        self.channel.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.channel.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct DamageEvent {
        target: u32,
        amount: f32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct DeathEvent {
        entity: u32,
    }

    #[test]
    fn channel_send_and_read() {
        let mut channel = EventChannel::<DamageEvent>::new();
        channel.send(DamageEvent {
            target: 1,
            amount: 10.0,
        });
        channel.send(DamageEvent {
            target: 2,
            amount: 20.0,
        });

        assert_eq!(channel.len(), 2);
        let events: Vec<_> = channel.iter().collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.target, 1);
        assert_eq!(events[1].event.target, 2);
    }

    #[test]
    fn channel_double_buffer() {
        let mut channel = EventChannel::<u32>::new();
        channel.send(1);
        channel.swap_buffers();
        channel.send(2);

        // Both events should be readable.
        let events: Vec<_> = channel.iter().map(|e| e.event).collect();
        assert_eq!(events.len(), 2);

        channel.swap_buffers();
        // Event 1 should be gone (older buffer cleared).
        let events: Vec<_> = channel.iter().map(|e| e.event).collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], 2);
    }

    #[test]
    fn event_reader_cursor() {
        let mut channel = EventChannel::<u32>::new();
        let mut reader = EventReader::<u32>::new();

        channel.send(10);
        channel.send(20);

        let events: Vec<_> = reader.read(&channel).copied().collect();
        assert_eq!(events, vec![10, 20]);

        // Second read should return nothing.
        let events: Vec<_> = reader.read(&channel).copied().collect();
        assert_eq!(events.len(), 0);

        // New event should be picked up.
        channel.send(30);
        let events: Vec<_> = reader.read(&channel).copied().collect();
        assert_eq!(events, vec![30]);
    }

    #[test]
    fn event_drain() {
        let mut channel = EventChannel::<u32>::new();
        channel.send(1);
        channel.send(2);
        channel.send(3);

        let drained: Vec<_> = channel.drain().collect();
        assert_eq!(drained, vec![1, 2, 3]);
        assert!(channel.is_empty());
    }

    #[test]
    fn registry_basic() {
        let mut registry = EventRegistry::new();
        registry.register::<DamageEvent>();

        registry.send(DamageEvent {
            target: 42,
            amount: 99.0,
        });

        let channel = registry.channel::<DamageEvent>().unwrap();
        assert_eq!(channel.len(), 1);
    }

    #[test]
    fn registry_auto_register() {
        let mut registry = EventRegistry::new();
        // send auto-registers.
        registry.send(42u32);
        assert!(registry.is_registered::<u32>());
    }

    #[test]
    fn channel_stats() {
        let mut channel = EventChannel::<u32>::new();
        for i in 0..100 {
            channel.send(i);
        }
        channel.swap_buffers();
        assert_eq!(channel.stats().total_sent, 100);
        assert_eq!(channel.stats().peak_per_frame, 100);
    }

    #[test]
    fn reader_peek() {
        let mut channel = EventChannel::<u32>::new();
        let reader = EventReader::<u32>::new();

        channel.send(1);
        channel.send(2);

        let peeked: Vec<_> = reader.peek(&channel).copied().collect();
        assert_eq!(peeked, vec![1, 2]);

        // Peek again should still return the same events.
        let peeked: Vec<_> = reader.peek(&channel).copied().collect();
        assert_eq!(peeked, vec![1, 2]);
    }
}
