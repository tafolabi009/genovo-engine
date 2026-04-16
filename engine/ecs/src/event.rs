//! Event system for the Genovo ECS.
//!
//! Events are a decoupled communication mechanism between systems. Events are
//! stored in a double-buffered layout: events written during tick N are
//! readable during ticks N and N+1. After two `swap_buffers()` calls they are
//! discarded.

// ---------------------------------------------------------------------------
// Events<T>
// ---------------------------------------------------------------------------

/// Double-buffered event storage for events of type `T`.
///
/// Events written during the current tick go into the "write" buffer. On
/// `swap_buffers()`, the read buffer is cleared and the buffers swap roles.
/// This ensures every consumer has at least one full tick to process events.
pub struct Events<T> {
    /// Buffer A (current write target on even ticks).
    buffer_a: Vec<T>,
    /// Buffer B (current write target on odd ticks).
    buffer_b: Vec<T>,
    /// Current parity: `false` = A is write buffer, `true` = B is write buffer.
    parity: bool,
}

impl<T> Events<T> {
    /// Create a new, empty event storage.
    pub fn new() -> Self {
        Self {
            buffer_a: Vec::new(),
            buffer_b: Vec::new(),
            parity: false,
        }
    }

    /// Send an event. It becomes readable on the current and next tick.
    pub fn send(&mut self, event: T) {
        if self.parity {
            self.buffer_b.push(event);
        } else {
            self.buffer_a.push(event);
        }
    }

    /// Send multiple events at once.
    pub fn send_batch(&mut self, events: impl IntoIterator<Item = T>) {
        for event in events {
            self.send(event);
        }
    }

    /// Swap buffers: clear the older buffer and flip parity.
    ///
    /// Call this once per frame, typically at the beginning of a tick.
    pub fn swap_buffers(&mut self) {
        if self.parity {
            self.buffer_a.clear();
        } else {
            self.buffer_b.clear();
        }
        self.parity = !self.parity;
    }

    /// Iterate over all events readable this tick (both buffers).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer_a.iter().chain(self.buffer_b.iter())
    }

    /// Drain all readable events, leaving both buffers empty.
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.buffer_a.drain(..).chain(self.buffer_b.drain(..))
    }

    /// Number of events currently readable (across both buffers).
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer_a.len() + self.buffer_b.len()
    }

    /// Whether there are no readable events.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer_a.is_empty() && self.buffer_b.is_empty()
    }

    /// Remove all events without advancing the tick.
    pub fn clear(&mut self) {
        self.buffer_a.clear();
        self.buffer_b.clear();
    }
}

impl<T> Default for Events<T> {
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

    #[test]
    fn send_and_iterate() {
        let mut events = Events::<u32>::new();
        events.send(1);
        events.send(2);
        events.send(3);

        let collected: Vec<u32> = events.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn double_buffer_lifecycle() {
        let mut events = Events::<u32>::new();

        // Tick 0: send 1 and 2.
        events.send(1);
        events.send(2);
        assert_eq!(events.len(), 2);

        // Tick 1: swap. Old events still readable. Send 3.
        events.swap_buffers();
        events.send(3);
        assert_eq!(events.iter().count(), 3); // 1, 2 (old) + 3 (new)

        // Tick 2: swap again. Events from tick 0 are cleared.
        events.swap_buffers();
        assert_eq!(events.iter().count(), 1); // only 3 remains
        assert_eq!(events.iter().next(), Some(&3));
    }

    #[test]
    fn drain_empties_both_buffers() {
        let mut events = Events::<&str>::new();
        events.send("hello");
        events.swap_buffers();
        events.send("world");

        let drained: Vec<&str> = events.drain().collect();
        assert_eq!(drained, vec!["hello", "world"]);
        assert!(events.is_empty());
    }

    #[test]
    fn send_batch() {
        let mut events = Events::<i32>::new();
        events.send_batch(vec![10, 20, 30]);
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn clear_removes_all() {
        let mut events = Events::<i32>::new();
        events.send(1);
        events.swap_buffers();
        events.send(2);

        events.clear();
        assert!(events.is_empty());
    }

    #[test]
    fn empty_events() {
        let events = Events::<f64>::new();
        assert!(events.is_empty());
        assert_eq!(events.len(), 0);
        assert_eq!(events.iter().count(), 0);
    }

    #[test]
    fn multiple_swap_cycles() {
        let mut events = Events::<u32>::new();

        // Tick 0
        events.send(1);
        events.swap_buffers();

        // Tick 1
        events.send(2);
        events.swap_buffers();

        // Tick 2: 1 is gone, 2 remains.
        events.send(3);
        let vals: Vec<u32> = events.iter().copied().collect();
        assert!(vals.contains(&2));
        assert!(vals.contains(&3));
        assert!(!vals.contains(&1));

        events.swap_buffers();

        // Tick 3: 2 is gone, 3 remains.
        let vals: Vec<u32> = events.iter().copied().collect();
        assert!(vals.contains(&3));
        assert!(!vals.contains(&2));
    }
}
