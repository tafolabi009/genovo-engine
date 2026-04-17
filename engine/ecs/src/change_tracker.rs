// engine/ecs/src/change_tracker.rs
//
// Change tracking for ECS components: per-component tick tracking,
// "changed this frame" queries, "added this frame", "removed this frame",
// global change tick, and efficient bitfield-based tracking.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A monotonically increasing tick counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChangeTick(pub u64);

impl ChangeTick {
    pub const ZERO: Self = Self(0);

    #[inline]
    pub fn new(tick: u64) -> Self { Self(tick) }

    #[inline]
    pub fn is_newer_than(&self, other: ChangeTick) -> bool {
        self.0 > other.0
    }

    #[inline]
    pub fn age(&self, current: ChangeTick) -> u64 {
        current.0.saturating_sub(self.0)
    }
}

/// Entity ID (simplified for this module).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

/// Component type ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentTypeId(pub u64);

impl ComponentTypeId {
    pub fn of<T: 'static>() -> Self {
        // Use type name hash as a stable ID.
        let name = std::any::type_name::<T>();
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in name.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Self(hash)
    }
}

// ---------------------------------------------------------------------------
// Per-component change record
// ---------------------------------------------------------------------------

/// Tracks when a specific component on a specific entity was last changed.
#[derive(Debug, Clone)]
struct ComponentChangeRecord {
    /// When this component was added.
    added_tick: ChangeTick,
    /// When this component was last modified.
    changed_tick: ChangeTick,
    /// Whether the component was removed (and which tick).
    removed_tick: Option<ChangeTick>,
}

// ---------------------------------------------------------------------------
// ChangeTracker
// ---------------------------------------------------------------------------

/// Tracks changes to all components in the ECS.
pub struct ChangeTracker {
    current_tick: ChangeTick,
    last_check_tick: ChangeTick,

    /// Per-entity, per-component change records.
    records: HashMap<(EntityId, ComponentTypeId), ComponentChangeRecord>,

    /// Added this frame.
    added_this_frame: Vec<(EntityId, ComponentTypeId)>,

    /// Changed this frame.
    changed_this_frame: Vec<(EntityId, ComponentTypeId)>,

    /// Removed this frame.
    removed_this_frame: Vec<(EntityId, ComponentTypeId)>,

    /// Entities that were spawned this frame.
    spawned_this_frame: HashSet<EntityId>,

    /// Entities that were despawned this frame.
    despawned_this_frame: HashSet<EntityId>,

    /// Statistics.
    stats: ChangeTrackerStats,
}

/// Statistics for the change tracker.
#[derive(Debug, Clone, Default)]
pub struct ChangeTrackerStats {
    pub current_tick: u64,
    pub total_records: usize,
    pub added_count: usize,
    pub changed_count: usize,
    pub removed_count: usize,
    pub spawned_count: usize,
    pub despawned_count: usize,
    pub total_notifications: u64,
}

impl ChangeTracker {
    pub fn new() -> Self {
        Self {
            current_tick: ChangeTick(1),
            last_check_tick: ChangeTick(0),
            records: HashMap::new(),
            added_this_frame: Vec::new(),
            changed_this_frame: Vec::new(),
            removed_this_frame: Vec::new(),
            spawned_this_frame: HashSet::new(),
            despawned_this_frame: HashSet::new(),
            stats: ChangeTrackerStats::default(),
        }
    }

    /// Get the current tick.
    pub fn current_tick(&self) -> ChangeTick {
        self.current_tick
    }

    /// Get the last check tick.
    pub fn last_check_tick(&self) -> ChangeTick {
        self.last_check_tick
    }

    /// Advance to the next tick (call at the start of each frame).
    pub fn advance_tick(&mut self) {
        self.last_check_tick = self.current_tick;
        self.current_tick = ChangeTick(self.current_tick.0 + 1);

        // Clear per-frame lists.
        self.added_this_frame.clear();
        self.changed_this_frame.clear();
        self.removed_this_frame.clear();
        self.spawned_this_frame.clear();
        self.despawned_this_frame.clear();

        self.update_stats();
    }

    // -----------------------------------------------------------------------
    // Notifications
    // -----------------------------------------------------------------------

    /// Notify that a component was added to an entity.
    pub fn notify_added(&mut self, entity: EntityId, component: ComponentTypeId) {
        let tick = self.current_tick;
        self.records.insert(
            (entity, component),
            ComponentChangeRecord {
                added_tick: tick,
                changed_tick: tick,
                removed_tick: None,
            },
        );
        self.added_this_frame.push((entity, component));
        self.stats.total_notifications += 1;
    }

    /// Notify that a component was changed on an entity.
    pub fn notify_changed(&mut self, entity: EntityId, component: ComponentTypeId) {
        let tick = self.current_tick;
        if let Some(record) = self.records.get_mut(&(entity, component)) {
            record.changed_tick = tick;
        } else {
            // Auto-add if not tracked.
            self.records.insert(
                (entity, component),
                ComponentChangeRecord {
                    added_tick: tick,
                    changed_tick: tick,
                    removed_tick: None,
                },
            );
        }
        self.changed_this_frame.push((entity, component));
        self.stats.total_notifications += 1;
    }

    /// Notify that a component was removed from an entity.
    pub fn notify_removed(&mut self, entity: EntityId, component: ComponentTypeId) {
        let tick = self.current_tick;
        if let Some(record) = self.records.get_mut(&(entity, component)) {
            record.removed_tick = Some(tick);
        }
        self.removed_this_frame.push((entity, component));
        self.stats.total_notifications += 1;
    }

    /// Notify that an entity was spawned.
    pub fn notify_spawned(&mut self, entity: EntityId) {
        self.spawned_this_frame.insert(entity);
    }

    /// Notify that an entity was despawned.
    pub fn notify_despawned(&mut self, entity: EntityId) {
        self.despawned_this_frame.insert(entity);
        // Remove all records for this entity.
        self.records.retain(|(e, _), _| *e != entity);
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Was this component added this frame?
    pub fn is_added(&self, entity: EntityId, component: ComponentTypeId) -> bool {
        self.added_this_frame.contains(&(entity, component))
    }

    /// Was this component changed this frame?
    pub fn is_changed(&self, entity: EntityId, component: ComponentTypeId) -> bool {
        self.changed_this_frame.contains(&(entity, component))
    }

    /// Was this component removed this frame?
    pub fn is_removed(&self, entity: EntityId, component: ComponentTypeId) -> bool {
        self.removed_this_frame.contains(&(entity, component))
    }

    /// Was this component added or changed this frame?
    pub fn is_added_or_changed(&self, entity: EntityId, component: ComponentTypeId) -> bool {
        self.is_added(entity, component) || self.is_changed(entity, component)
    }

    /// Was this entity spawned this frame?
    pub fn is_spawned(&self, entity: EntityId) -> bool {
        self.spawned_this_frame.contains(&entity)
    }

    /// Was this entity despawned this frame?
    pub fn is_despawned(&self, entity: EntityId) -> bool {
        self.despawned_this_frame.contains(&entity)
    }

    /// Get the tick when a component was added.
    pub fn added_tick(&self, entity: EntityId, component: ComponentTypeId) -> Option<ChangeTick> {
        self.records.get(&(entity, component)).map(|r| r.added_tick)
    }

    /// Get the tick when a component was last changed.
    pub fn changed_tick(&self, entity: EntityId, component: ComponentTypeId) -> Option<ChangeTick> {
        self.records.get(&(entity, component)).map(|r| r.changed_tick)
    }

    /// Was this component changed since a given tick?
    pub fn changed_since(&self, entity: EntityId, component: ComponentTypeId, since: ChangeTick) -> bool {
        self.records.get(&(entity, component))
            .map(|r| r.changed_tick.is_newer_than(since))
            .unwrap_or(false)
    }

    /// Was this component added since a given tick?
    pub fn added_since(&self, entity: EntityId, component: ComponentTypeId, since: ChangeTick) -> bool {
        self.records.get(&(entity, component))
            .map(|r| r.added_tick.is_newer_than(since))
            .unwrap_or(false)
    }

    /// Get all entities with a specific component that were added this frame.
    pub fn added_entities(&self, component: ComponentTypeId) -> Vec<EntityId> {
        self.added_this_frame.iter()
            .filter(|(_, c)| *c == component)
            .map(|(e, _)| *e)
            .collect()
    }

    /// Get all entities with a specific component that were changed this frame.
    pub fn changed_entities(&self, component: ComponentTypeId) -> Vec<EntityId> {
        self.changed_this_frame.iter()
            .filter(|(_, c)| *c == component)
            .map(|(e, _)| *e)
            .collect()
    }

    /// Get all entities with a specific component that were removed this frame.
    pub fn removed_entities(&self, component: ComponentTypeId) -> Vec<EntityId> {
        self.removed_this_frame.iter()
            .filter(|(_, c)| *c == component)
            .map(|(e, _)| *e)
            .collect()
    }

    /// Get all entities that were added this frame (for any component).
    pub fn all_added_this_frame(&self) -> &[(EntityId, ComponentTypeId)] {
        &self.added_this_frame
    }

    /// Get all entities that were changed this frame.
    pub fn all_changed_this_frame(&self) -> &[(EntityId, ComponentTypeId)] {
        &self.changed_this_frame
    }

    /// Get all entities that were removed this frame.
    pub fn all_removed_this_frame(&self) -> &[(EntityId, ComponentTypeId)] {
        &self.removed_this_frame
    }

    /// Get all spawned entities this frame.
    pub fn all_spawned_this_frame(&self) -> &HashSet<EntityId> {
        &self.spawned_this_frame
    }

    /// Get all despawned entities this frame.
    pub fn all_despawned_this_frame(&self) -> &HashSet<EntityId> {
        &self.despawned_this_frame
    }

    /// How many frames ago was a component last changed?
    pub fn change_age(&self, entity: EntityId, component: ComponentTypeId) -> Option<u64> {
        self.records.get(&(entity, component))
            .map(|r| r.changed_tick.age(self.current_tick))
    }

    /// Clean up old records (removed components older than `max_age` ticks).
    pub fn cleanup(&mut self, max_age: u64) {
        let threshold = ChangeTick(self.current_tick.0.saturating_sub(max_age));
        self.records.retain(|_, record| {
            if let Some(removed_tick) = record.removed_tick {
                removed_tick.is_newer_than(threshold)
            } else {
                true
            }
        });
    }

    pub fn stats(&self) -> &ChangeTrackerStats {
        &self.stats
    }

    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    fn update_stats(&mut self) {
        self.stats.current_tick = self.current_tick.0;
        self.stats.total_records = self.records.len();
        self.stats.added_count = self.added_this_frame.len();
        self.stats.changed_count = self.changed_this_frame.len();
        self.stats.removed_count = self.removed_this_frame.len();
        self.stats.spawned_count = self.spawned_this_frame.len();
        self.stats.despawned_count = self.despawned_this_frame.len();
    }
}

impl Default for ChangeTracker {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Bitfield-based change detection for dense component arrays
// ---------------------------------------------------------------------------

/// A compact bitfield for tracking which entries in a dense array have changed.
pub struct ChangeBitfield {
    bits: Vec<u64>,
    len: usize,
    tick: ChangeTick,
}

impl ChangeBitfield {
    /// Create a bitfield for `count` entries.
    pub fn new(count: usize) -> Self {
        let words = (count + 63) / 64;
        Self {
            bits: vec![0u64; words],
            len: count,
            tick: ChangeTick::ZERO,
        }
    }

    /// Mark entry `index` as changed.
    #[inline]
    pub fn mark(&mut self, index: usize) {
        if index < self.len {
            let word = index / 64;
            let bit = index % 64;
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Check if entry `index` is marked as changed.
    #[inline]
    pub fn is_marked(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        let word = index / 64;
        let bit = index % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    /// Clear all marks.
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
    }

    /// Count the number of marked entries.
    pub fn count_marked(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate over all marked indices.
    pub fn iter_marked(&self) -> MarkedIter<'_> {
        MarkedIter {
            bits: &self.bits,
            current_word: 0,
            current_bits: if self.bits.is_empty() { 0 } else { self.bits[0] },
            base_index: 0,
            max_index: self.len,
        }
    }

    /// Set the tick this bitfield was last reset.
    pub fn set_tick(&mut self, tick: ChangeTick) {
        self.tick = tick;
    }

    pub fn tick(&self) -> ChangeTick {
        self.tick
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Resize the bitfield.
    pub fn resize(&mut self, new_len: usize) {
        let new_words = (new_len + 63) / 64;
        self.bits.resize(new_words, 0);
        self.len = new_len;
    }

    /// Bitwise OR with another bitfield.
    pub fn merge(&mut self, other: &ChangeBitfield) {
        let words = self.bits.len().min(other.bits.len());
        for i in 0..words {
            self.bits[i] |= other.bits[i];
        }
    }

    /// Bitwise AND with another bitfield (intersection).
    pub fn intersect(&mut self, other: &ChangeBitfield) {
        let words = self.bits.len().min(other.bits.len());
        for i in 0..words {
            self.bits[i] &= other.bits[i];
        }
        // Clear words beyond other's length.
        for i in words..self.bits.len() {
            self.bits[i] = 0;
        }
    }
}

/// Iterator over marked indices in a change bitfield.
pub struct MarkedIter<'a> {
    bits: &'a [u64],
    current_word: usize,
    current_bits: u64,
    base_index: usize,
    max_index: usize,
}

impl<'a> Iterator for MarkedIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current_bits != 0 {
                let bit = self.current_bits.trailing_zeros() as usize;
                self.current_bits &= self.current_bits - 1; // Clear lowest bit.
                let index = self.base_index + bit;
                if index < self.max_index {
                    return Some(index);
                }
                return None;
            }

            self.current_word += 1;
            if self.current_word >= self.bits.len() {
                return None;
            }

            self.base_index = self.current_word * 64;
            self.current_bits = self.bits[self.current_word];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tracking() {
        let mut tracker = ChangeTracker::new();
        let entity = EntityId(1);
        let comp = ComponentTypeId(100);

        tracker.notify_added(entity, comp);
        assert!(tracker.is_added(entity, comp));
        assert!(!tracker.is_changed(entity, comp));

        tracker.advance_tick();
        assert!(!tracker.is_added(entity, comp));

        tracker.notify_changed(entity, comp);
        assert!(tracker.is_changed(entity, comp));
    }

    #[test]
    fn test_changed_since() {
        let mut tracker = ChangeTracker::new();
        let entity = EntityId(1);
        let comp = ComponentTypeId(100);

        tracker.notify_added(entity, comp);
        let added_tick = tracker.current_tick();

        tracker.advance_tick();
        tracker.advance_tick();

        assert!(tracker.changed_since(entity, comp, ChangeTick::ZERO));
        assert!(!tracker.changed_since(entity, comp, added_tick));

        tracker.notify_changed(entity, comp);
        assert!(tracker.changed_since(entity, comp, added_tick));
    }

    #[test]
    fn test_removed() {
        let mut tracker = ChangeTracker::new();
        let entity = EntityId(1);
        let comp = ComponentTypeId(100);

        tracker.notify_added(entity, comp);
        tracker.advance_tick();
        tracker.notify_removed(entity, comp);
        assert!(tracker.is_removed(entity, comp));
    }

    #[test]
    fn test_spawned_despawned() {
        let mut tracker = ChangeTracker::new();
        let entity = EntityId(1);

        tracker.notify_spawned(entity);
        assert!(tracker.is_spawned(entity));

        tracker.advance_tick();
        assert!(!tracker.is_spawned(entity));

        tracker.notify_despawned(entity);
        assert!(tracker.is_despawned(entity));
    }

    #[test]
    fn test_bitfield() {
        let mut bf = ChangeBitfield::new(128);
        bf.mark(0);
        bf.mark(63);
        bf.mark(64);
        bf.mark(127);

        assert!(bf.is_marked(0));
        assert!(bf.is_marked(63));
        assert!(bf.is_marked(64));
        assert!(bf.is_marked(127));
        assert!(!bf.is_marked(1));
        assert_eq!(bf.count_marked(), 4);

        let marked: Vec<usize> = bf.iter_marked().collect();
        assert_eq!(marked, vec![0, 63, 64, 127]);

        bf.clear();
        assert_eq!(bf.count_marked(), 0);
    }

    #[test]
    fn test_bitfield_merge() {
        let mut a = ChangeBitfield::new(64);
        let mut b = ChangeBitfield::new(64);
        a.mark(0);
        b.mark(1);
        a.merge(&b);
        assert!(a.is_marked(0));
        assert!(a.is_marked(1));
    }

    #[test]
    fn test_change_age() {
        let mut tracker = ChangeTracker::new();
        let entity = EntityId(1);
        let comp = ComponentTypeId(100);

        tracker.notify_added(entity, comp);
        assert_eq!(tracker.change_age(entity, comp), Some(0));

        tracker.advance_tick();
        tracker.advance_tick();
        assert_eq!(tracker.change_age(entity, comp), Some(2));
    }

    #[test]
    fn test_component_type_id() {
        let id1 = ComponentTypeId::of::<u32>();
        let id2 = ComponentTypeId::of::<u32>();
        let id3 = ComponentTypeId::of::<f64>();
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }
}
