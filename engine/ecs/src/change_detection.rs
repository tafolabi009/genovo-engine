//! Change detection for the Genovo ECS.
//!
//! Tracks which components have been modified or added during a frame, enabling
//! efficient reactive systems that only process entities whose data actually
//! changed.
//!
//! # Concepts
//!
//! - **Tick** — a monotonically increasing counter, bumped once per frame.
//! - **Changed** — a component was mutated since the system's last run.
//! - **Added** — a component was newly inserted since the system's last run.
//!
//! # Usage
//!
//! ```ignore
//! // Track changes per-component using the ChangeTracker.
//! let mut tracker = ChangeTracker::new();
//!
//! // Mark a component as changed this tick.
//! tracker.mark_changed(entity, ComponentId::of::<Position>(), current_tick);
//!
//! // Check if a component changed since a given tick.
//! if tracker.is_changed(entity, ComponentId::of::<Position>(), since_tick) {
//!     // Process the change.
//! }
//! ```

use std::collections::HashMap;

use crate::component::{Component, ComponentId};
use crate::entity::Entity;

// ---------------------------------------------------------------------------
// Ticks — per-component change tick
// ---------------------------------------------------------------------------

/// Per-component-instance tick tracking.
///
/// Stores the tick when a component was last written to and the tick when it
/// was first added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentTicks {
    /// The tick when this component was first added to the entity.
    pub added: u32,
    /// The tick when this component was last mutably accessed.
    pub changed: u32,
}

impl ComponentTicks {
    /// Create ticks for a newly-added component.
    pub fn new(current_tick: u32) -> Self {
        Self {
            added: current_tick,
            changed: current_tick,
        }
    }

    /// Returns `true` if the component was added since `last_run_tick`.
    pub fn is_added(&self, last_run_tick: u32, current_tick: u32) -> bool {
        // Handle wrapping by checking the range.
        tick_in_range(self.added, last_run_tick, current_tick)
    }

    /// Returns `true` if the component was changed since `last_run_tick`.
    pub fn is_changed(&self, last_run_tick: u32, current_tick: u32) -> bool {
        tick_in_range(self.changed, last_run_tick, current_tick)
    }

    /// Mark as changed at the given tick.
    #[inline]
    pub fn set_changed(&mut self, tick: u32) {
        self.changed = tick;
    }
}

/// Check if a tick `t` falls within the range `(after, up_to]`, handling
/// wrapping.
fn tick_in_range(t: u32, after: u32, up_to: u32) -> bool {
    // Calculate the number of ticks elapsed with wrapping.
    let elapsed_t = up_to.wrapping_sub(t);
    let elapsed_after = up_to.wrapping_sub(after);
    // t is in range if it is more recent than `after` (i.e., elapsed less).
    elapsed_t < elapsed_after
}

// ---------------------------------------------------------------------------
// ChangeTracker — central change tracking
// ---------------------------------------------------------------------------

/// Central change tracker that records per-(entity, component) ticks.
pub struct ChangeTracker {
    /// (Entity, ComponentId) → ComponentTicks.
    ticks: HashMap<(Entity, ComponentId), ComponentTicks>,
}

impl ChangeTracker {
    /// Create a new, empty change tracker.
    pub fn new() -> Self {
        Self {
            ticks: HashMap::new(),
        }
    }

    /// Record that a component was added to an entity.
    pub fn mark_added(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        tick: u32,
    ) {
        self.ticks
            .insert((entity, component_id), ComponentTicks::new(tick));
    }

    /// Record that a component was changed (mutably accessed).
    pub fn mark_changed(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        tick: u32,
    ) {
        self.ticks
            .entry((entity, component_id))
            .and_modify(|t| t.set_changed(tick))
            .or_insert_with(|| ComponentTicks::new(tick));
    }

    /// Remove tracking for an entity's component (e.g., on component removal).
    pub fn remove(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
    ) {
        self.ticks.remove(&(entity, component_id));
    }

    /// Remove all tracking for an entity (e.g., on despawn).
    pub fn remove_entity(&mut self, entity: Entity) {
        self.ticks.retain(|&(e, _), _| e != entity);
    }

    /// Check if a component was added since `last_run_tick`.
    pub fn is_added(
        &self,
        entity: Entity,
        component_id: ComponentId,
        last_run_tick: u32,
        current_tick: u32,
    ) -> bool {
        self.ticks
            .get(&(entity, component_id))
            .map_or(false, |t| t.is_added(last_run_tick, current_tick))
    }

    /// Check if a component was changed since `last_run_tick`.
    pub fn is_changed(
        &self,
        entity: Entity,
        component_id: ComponentId,
        last_run_tick: u32,
        current_tick: u32,
    ) -> bool {
        self.ticks
            .get(&(entity, component_id))
            .map_or(false, |t| t.is_changed(last_run_tick, current_tick))
    }

    /// Get the ticks for a specific (entity, component) pair.
    pub fn get_ticks(
        &self,
        entity: Entity,
        component_id: ComponentId,
    ) -> Option<&ComponentTicks> {
        self.ticks.get(&(entity, component_id))
    }

    /// Clear all tracked ticks.
    pub fn clear(&mut self) {
        self.ticks.clear();
    }

    /// Number of tracked (entity, component) pairs.
    pub fn len(&self) -> usize {
        self.ticks.len()
    }

    /// Whether the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.ticks.is_empty()
    }

    /// Iterate all entities that have a changed component since `last_run_tick`.
    pub fn changed_entities(
        &self,
        component_id: ComponentId,
        last_run_tick: u32,
        current_tick: u32,
    ) -> Vec<Entity> {
        self.ticks
            .iter()
            .filter_map(|(&(entity, cid), ticks)| {
                if cid == component_id
                    && ticks.is_changed(last_run_tick, current_tick)
                {
                    Some(entity)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Iterate all entities that have an added component since `last_run_tick`.
    pub fn added_entities(
        &self,
        component_id: ComponentId,
        last_run_tick: u32,
        current_tick: u32,
    ) -> Vec<Entity> {
        self.ticks
            .iter()
            .filter_map(|(&(entity, cid), ticks)| {
                if cid == component_id
                    && ticks.is_added(last_run_tick, current_tick)
                {
                    Some(entity)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for ChangeTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Ref<T> — read reference with change tracking info
// ---------------------------------------------------------------------------

/// Immutable component reference that carries change detection metadata.
///
/// ```ignore
/// let pos: Ref<Position> = ...; // obtained from a change-aware query
/// if pos.is_changed() {
///     // process the change
/// }
/// let value: &Position = &*pos;
/// ```
pub struct Ref<'a, T> {
    /// Reference to the component data.
    value: &'a T,
    /// The tick when this component was last changed.
    ticks: ComponentTicks,
    /// The tick of the last system run.
    last_run_tick: u32,
    /// The current tick.
    current_tick: u32,
}

impl<'a, T> Ref<'a, T> {
    /// Create a new `Ref`.
    pub fn new(
        value: &'a T,
        ticks: ComponentTicks,
        last_run_tick: u32,
        current_tick: u32,
    ) -> Self {
        Self {
            value,
            ticks,
            last_run_tick,
            current_tick,
        }
    }

    /// Returns `true` if the component was added since the last system run.
    pub fn is_added(&self) -> bool {
        self.ticks.is_added(self.last_run_tick, self.current_tick)
    }

    /// Returns `true` if the component was changed since the last system run.
    pub fn is_changed(&self) -> bool {
        self.ticks
            .is_changed(self.last_run_tick, self.current_tick)
    }

    /// Get the underlying ticks.
    pub fn ticks(&self) -> ComponentTicks {
        self.ticks
    }
}

impl<'a, T> std::ops::Deref for Ref<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, T: std::fmt::Debug> std::fmt::Debug for Ref<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ref")
            .field("value", &self.value)
            .field("changed", &self.is_changed())
            .field("added", &self.is_added())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Mut<T> — mutable reference that auto-marks changed
// ---------------------------------------------------------------------------

/// Mutable component reference that automatically marks the component as
/// changed when accessed.
///
/// ```ignore
/// let mut vel: Mut<Velocity> = ...; // obtained from a change-aware query
/// vel.dx = 10.0; // automatically marks as changed
/// ```
pub struct Mut<'a, T> {
    /// Mutable reference to the component data.
    value: &'a mut T,
    /// The tick when this component was last changed (updated on deref_mut).
    ticks: &'a mut ComponentTicks,
    /// The current tick — written into `ticks.changed` on mutation.
    current_tick: u32,
    /// The tick of the last system run.
    last_run_tick: u32,
}

impl<'a, T> Mut<'a, T> {
    /// Create a new `Mut`.
    pub fn new(
        value: &'a mut T,
        ticks: &'a mut ComponentTicks,
        last_run_tick: u32,
        current_tick: u32,
    ) -> Self {
        Self {
            value,
            ticks,
            current_tick,
            last_run_tick,
        }
    }

    /// Returns `true` if the component was added since the last system run.
    pub fn is_added(&self) -> bool {
        self.ticks.is_added(self.last_run_tick, self.current_tick)
    }

    /// Returns `true` if the component was changed since the last system run.
    pub fn is_changed(&self) -> bool {
        self.ticks
            .is_changed(self.last_run_tick, self.current_tick)
    }

    /// Manually set the component as changed at the given tick.
    pub fn set_changed(&mut self) {
        self.ticks.set_changed(self.current_tick);
    }

    /// Get the underlying ticks.
    pub fn ticks(&self) -> ComponentTicks {
        *self.ticks
    }
}

impl<'a, T> std::ops::Deref for Mut<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, T> std::ops::DerefMut for Mut<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.ticks.set_changed(self.current_tick);
        self.value
    }
}

impl<'a, T: std::fmt::Debug> std::fmt::Debug for Mut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mut")
            .field("value", &self.value)
            .field("changed", &self.is_changed())
            .field("added", &self.is_added())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Changed<T> and Added<T> query filters
// ---------------------------------------------------------------------------

/// Query filter that only yields entities where component `T` was modified
/// since the system's last run.
///
/// ```ignore
/// for (entity, pos) in world.query_filtered::<&Position, Changed<Position>>() {
///     // Only entities whose Position changed
/// }
/// ```
pub struct Changed<T: Component> {
    _marker: std::marker::PhantomData<T>,
}

/// Query filter that only yields entities where component `T` was added
/// since the system's last run.
///
/// ```ignore
/// for (entity, pos) in world.query_filtered::<&Position, Added<Position>>() {
///     // Only entities that just got a Position
/// }
/// ```
pub struct Added<T: Component> {
    _marker: std::marker::PhantomData<T>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }
    impl crate::Component for Position {}

    #[derive(Debug, Clone, PartialEq)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }
    impl crate::Component for Velocity {}

    #[test]
    fn component_ticks_new() {
        let ticks = ComponentTicks::new(5);
        assert_eq!(ticks.added, 5);
        assert_eq!(ticks.changed, 5);
    }

    #[test]
    fn component_ticks_is_added() {
        let ticks = ComponentTicks::new(5);
        // Added at tick 5, last_run at tick 3, current at tick 6 → added.
        assert!(ticks.is_added(3, 6));
        // Added at tick 5, last_run at tick 5, current at tick 6 → not added.
        assert!(!ticks.is_added(5, 6));
        // Added at tick 5, last_run at tick 6, current at tick 7 → not added.
        assert!(!ticks.is_added(6, 7));
    }

    #[test]
    fn component_ticks_is_changed() {
        let mut ticks = ComponentTicks::new(5);
        ticks.set_changed(8);
        // Changed at tick 8, last_run at tick 7, current at tick 9 → changed.
        assert!(ticks.is_changed(7, 9));
        // Changed at tick 8, last_run at tick 8, current at tick 9 → not changed.
        assert!(!ticks.is_changed(8, 9));
    }

    #[test]
    fn change_tracker_mark_and_check() {
        let mut tracker = ChangeTracker::new();
        let entity = Entity::new(0, 0);
        let comp_id = ComponentId::of::<Position>();

        tracker.mark_added(entity, comp_id, 1);
        assert!(tracker.is_added(entity, comp_id, 0, 2));
        assert!(tracker.is_changed(entity, comp_id, 0, 2));

        // Mark changed at a later tick.
        tracker.mark_changed(entity, comp_id, 5);
        assert!(tracker.is_changed(entity, comp_id, 4, 6));
        assert!(!tracker.is_changed(entity, comp_id, 5, 6));
    }

    #[test]
    fn change_tracker_remove() {
        let mut tracker = ChangeTracker::new();
        let entity = Entity::new(0, 0);
        let comp_id = ComponentId::of::<Position>();

        tracker.mark_added(entity, comp_id, 1);
        assert_eq!(tracker.len(), 1);

        tracker.remove(entity, comp_id);
        assert_eq!(tracker.len(), 0);
        assert!(!tracker.is_changed(entity, comp_id, 0, 2));
    }

    #[test]
    fn change_tracker_remove_entity() {
        let mut tracker = ChangeTracker::new();
        let entity = Entity::new(0, 0);
        let pos_id = ComponentId::of::<Position>();
        let vel_id = ComponentId::of::<Velocity>();

        tracker.mark_added(entity, pos_id, 1);
        tracker.mark_added(entity, vel_id, 1);
        assert_eq!(tracker.len(), 2);

        tracker.remove_entity(entity);
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn change_tracker_changed_entities() {
        let mut tracker = ChangeTracker::new();
        let e0 = Entity::new(0, 0);
        let e1 = Entity::new(1, 0);
        let e2 = Entity::new(2, 0);
        let comp_id = ComponentId::of::<Position>();

        tracker.mark_added(e0, comp_id, 1);
        tracker.mark_changed(e0, comp_id, 5);
        tracker.mark_added(e1, comp_id, 2);
        tracker.mark_added(e2, comp_id, 3);
        tracker.mark_changed(e2, comp_id, 5);

        let changed = tracker.changed_entities(comp_id, 4, 6);
        assert_eq!(changed.len(), 2);
        assert!(changed.contains(&e0));
        assert!(changed.contains(&e2));
    }

    #[test]
    fn change_tracker_added_entities() {
        let mut tracker = ChangeTracker::new();
        let e0 = Entity::new(0, 0);
        let e1 = Entity::new(1, 0);
        let comp_id = ComponentId::of::<Position>();

        tracker.mark_added(e0, comp_id, 3);
        tracker.mark_added(e1, comp_id, 5);

        let added = tracker.added_entities(comp_id, 4, 6);
        assert_eq!(added.len(), 1);
        assert!(added.contains(&e1));
    }

    #[test]
    fn ref_deref() {
        let pos = Position { x: 1.0, y: 2.0 };
        let ticks = ComponentTicks::new(1);
        let r = Ref::new(&pos, ticks, 0, 2);
        assert_eq!(r.x, 1.0);
        assert_eq!(r.y, 2.0);
        assert!(r.is_added());
        assert!(r.is_changed());
    }

    #[test]
    fn ref_not_changed() {
        let pos = Position { x: 1.0, y: 2.0 };
        let ticks = ComponentTicks {
            added: 1,
            changed: 1,
        };
        let r = Ref::new(&pos, ticks, 5, 6);
        assert!(!r.is_added());
        assert!(!r.is_changed());
    }

    #[test]
    fn mut_auto_marks_changed() {
        let mut pos = Position { x: 1.0, y: 2.0 };
        let mut ticks = ComponentTicks::new(1);
        let current_tick = 10;
        {
            let mut m = Mut::new(&mut pos, &mut ticks, 0, current_tick);
            m.x = 99.0; // triggers DerefMut → set_changed
        }
        assert_eq!(ticks.changed, current_tick);
        assert_eq!(pos.x, 99.0);
    }

    #[test]
    fn mut_is_added_and_changed() {
        let mut pos = Position { x: 1.0, y: 2.0 };
        let mut ticks = ComponentTicks::new(5);
        let m = Mut::new(&mut pos, &mut ticks, 3, 6);
        assert!(m.is_added());
        assert!(m.is_changed());
    }

    #[test]
    fn mut_deref_read_does_not_mark_changed() {
        let mut pos = Position { x: 1.0, y: 2.0 };
        let mut ticks = ComponentTicks::new(1);
        {
            let m = Mut::new(&mut pos, &mut ticks, 0, 10);
            // Read-only access through Deref.
            let _x = m.x;
        }
        // Changed tick should still be 1, not 10.
        assert_eq!(ticks.changed, 1);
    }

    #[test]
    fn tick_wrapping() {
        // Test near u32::MAX wrapping.
        let ticks = ComponentTicks {
            added: u32::MAX - 1,
            changed: u32::MAX,
        };
        // last_run = MAX-2, current = 1 (wrapped).
        assert!(ticks.is_added(u32::MAX - 2, 1));
        assert!(ticks.is_changed(u32::MAX - 1, 1));
    }

    #[test]
    fn change_tracker_clear() {
        let mut tracker = ChangeTracker::new();
        tracker.mark_added(Entity::new(0, 0), ComponentId::of::<Position>(), 1);
        tracker.mark_added(Entity::new(1, 0), ComponentId::of::<Position>(), 1);
        assert_eq!(tracker.len(), 2);

        tracker.clear();
        assert!(tracker.is_empty());
    }

    #[test]
    fn change_tracker_get_ticks() {
        let mut tracker = ChangeTracker::new();
        let entity = Entity::new(0, 0);
        let comp_id = ComponentId::of::<Position>();

        assert!(tracker.get_ticks(entity, comp_id).is_none());

        tracker.mark_added(entity, comp_id, 5);
        let ticks = tracker.get_ticks(entity, comp_id).unwrap();
        assert_eq!(ticks.added, 5);
        assert_eq!(ticks.changed, 5);
    }
}
