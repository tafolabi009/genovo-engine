//! Resource system for the Genovo ECS.
//!
//! Resources are singleton values stored in the [`World`](crate::World). Unlike
//! components, a resource type has at most one instance globally rather than
//! per-entity. Common uses include time, input state, and rendering context.
//!
//! # Typed wrappers
//!
//! - [`Res<T>`] — shared (immutable) resource reference.
//! - [`ResMut<T>`] — mutable resource reference with automatic change detection.
//!
//! These wrappers are the primary way systems interact with resources. They
//! carry change-detection metadata so the scheduler can track which resources
//! are read vs. written, enabling safe parallelism.

use std::ops::{Deref, DerefMut};

use crate::change_detection::ComponentTicks;

// ---------------------------------------------------------------------------
// Resource trait
// ---------------------------------------------------------------------------

/// Marker trait for ECS resources.
///
/// Any `'static + Send + Sync` type can be a resource. The trait exists to
/// provide a clear opt-in point and future derive-macro support.
///
/// ```ignore
/// struct DeltaTime(f32);
/// impl Resource for DeltaTime {}
///
/// world.add_resource(DeltaTime(1.0 / 60.0));
/// ```
pub trait Resource: 'static + Send + Sync {}

// Blanket impl so any compatible type works without manual impl.
// (Users can still `impl Resource for T {}` explicitly for documentation.)
impl<T: 'static + Send + Sync> Resource for T {}

// ---------------------------------------------------------------------------
// ResourceTicks — per-resource change tracking
// ---------------------------------------------------------------------------

/// Change-detection ticks for a resource.
#[derive(Debug, Clone)]
pub struct ResourceTicks {
    /// When the resource was inserted.
    pub added: u32,
    /// When the resource was last mutably accessed.
    pub changed: u32,
}

impl ResourceTicks {
    /// Create ticks for a newly-inserted resource.
    pub fn new(tick: u32) -> Self {
        Self {
            added: tick,
            changed: tick,
        }
    }

    /// Mark the resource as changed.
    pub fn set_changed(&mut self, tick: u32) {
        self.changed = tick;
    }

    /// Convert to [`ComponentTicks`] for reuse with change detection utilities.
    pub fn as_component_ticks(&self) -> ComponentTicks {
        ComponentTicks {
            added: self.added,
            changed: self.changed,
        }
    }
}

// ---------------------------------------------------------------------------
// Res<T> — shared resource reference
// ---------------------------------------------------------------------------

/// Shared (immutable) reference to a resource of type `T`.
///
/// Carries change-detection metadata so systems can check whether the resource
/// was recently added or modified.
///
/// ```ignore
/// fn my_system(time: Res<Time>) {
///     let dt = time.delta;
///     if time.is_changed() {
///         println!("time was modified");
///     }
/// }
/// ```
pub struct Res<'a, T: 'static> {
    value: &'a T,
    ticks: ResourceTicks,
    last_run_tick: u32,
    current_tick: u32,
}

impl<'a, T: 'static> Res<'a, T> {
    /// Create a new `Res`.
    pub fn new(
        value: &'a T,
        ticks: ResourceTicks,
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

    /// Returns `true` if the resource was inserted since the last system run.
    pub fn is_added(&self) -> bool {
        self.ticks
            .as_component_ticks()
            .is_added(self.last_run_tick, self.current_tick)
    }

    /// Returns `true` if the resource was changed since the last system run.
    pub fn is_changed(&self) -> bool {
        self.ticks
            .as_component_ticks()
            .is_changed(self.last_run_tick, self.current_tick)
    }

    /// Get the underlying ticks.
    pub fn ticks(&self) -> &ResourceTicks {
        &self.ticks
    }

    /// Get the inner reference.
    pub fn into_inner(self) -> &'a T {
        self.value
    }
}

impl<'a, T: 'static> Deref for Res<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, T: std::fmt::Debug + 'static> std::fmt::Debug for Res<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Res")
            .field("value", &self.value)
            .field("is_changed", &self.is_changed())
            .field("is_added", &self.is_added())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ResMut<T> — mutable resource reference
// ---------------------------------------------------------------------------

/// Mutable reference to a resource of type `T` with automatic change detection.
///
/// When the inner value is accessed mutably (via `DerefMut`), the resource is
/// automatically marked as changed at the current tick.
///
/// ```ignore
/// fn update_time(mut time: ResMut<Time>) {
///     time.elapsed += time.delta; // auto-marks as changed
/// }
/// ```
pub struct ResMut<'a, T: 'static> {
    value: &'a mut T,
    ticks: &'a mut ResourceTicks,
    last_run_tick: u32,
    current_tick: u32,
}

impl<'a, T: 'static> ResMut<'a, T> {
    /// Create a new `ResMut`.
    pub fn new(
        value: &'a mut T,
        ticks: &'a mut ResourceTicks,
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

    /// Returns `true` if the resource was inserted since the last system run.
    pub fn is_added(&self) -> bool {
        self.ticks
            .as_component_ticks()
            .is_added(self.last_run_tick, self.current_tick)
    }

    /// Returns `true` if the resource was changed since the last system run.
    pub fn is_changed(&self) -> bool {
        self.ticks
            .as_component_ticks()
            .is_changed(self.last_run_tick, self.current_tick)
    }

    /// Manually mark the resource as changed.
    pub fn set_changed(&mut self) {
        self.ticks.set_changed(self.current_tick);
    }

    /// Get the underlying ticks.
    pub fn ticks(&self) -> &ResourceTicks {
        self.ticks
    }

    /// Bypass change detection and get a mutable reference without marking as
    /// changed.
    pub fn bypass_change_detection(&mut self) -> &mut T {
        self.value
    }

    /// Get the inner mutable reference.
    pub fn into_inner(self) -> &'a mut T {
        self.value
    }
}

impl<'a, T: 'static> Deref for ResMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, T: 'static> DerefMut for ResMut<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.ticks.set_changed(self.current_tick);
        self.value
    }
}

impl<'a, T: std::fmt::Debug + 'static> std::fmt::Debug for ResMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResMut")
            .field("value", &self.value)
            .field("is_changed", &self.is_changed())
            .field("is_added", &self.is_added())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Clone)]
    struct Time {
        delta: f32,
        elapsed: f32,
    }

    #[derive(Debug, PartialEq, Clone)]
    struct Score(u32);

    #[test]
    fn resource_ticks_new() {
        let ticks = ResourceTicks::new(5);
        assert_eq!(ticks.added, 5);
        assert_eq!(ticks.changed, 5);
    }

    #[test]
    fn resource_ticks_set_changed() {
        let mut ticks = ResourceTicks::new(1);
        ticks.set_changed(5);
        assert_eq!(ticks.changed, 5);
        assert_eq!(ticks.added, 1);
    }

    #[test]
    fn res_deref() {
        let time = Time {
            delta: 0.016,
            elapsed: 1.0,
        };
        let ticks = ResourceTicks::new(1);
        let res = Res::new(&time, ticks, 0, 2);
        assert_eq!(res.delta, 0.016);
        assert_eq!(res.elapsed, 1.0);
    }

    #[test]
    fn res_is_added_and_changed() {
        let time = Time {
            delta: 0.016,
            elapsed: 1.0,
        };
        let ticks = ResourceTicks::new(5);
        let res = Res::new(&time, ticks, 3, 6);
        assert!(res.is_added());
        assert!(res.is_changed());
    }

    #[test]
    fn res_not_changed() {
        let time = Time {
            delta: 0.016,
            elapsed: 1.0,
        };
        let ticks = ResourceTicks::new(1);
        let res = Res::new(&time, ticks, 5, 6);
        assert!(!res.is_added());
        assert!(!res.is_changed());
    }

    #[test]
    fn res_into_inner() {
        let time = Time {
            delta: 0.016,
            elapsed: 1.0,
        };
        let ticks = ResourceTicks::new(1);
        let res = Res::new(&time, ticks, 0, 2);
        let inner: &Time = res.into_inner();
        assert_eq!(inner.delta, 0.016);
    }

    #[test]
    fn res_mut_deref_marks_changed() {
        let mut time = Time {
            delta: 0.016,
            elapsed: 1.0,
        };
        let mut ticks = ResourceTicks::new(1);
        {
            let mut res = ResMut::new(&mut time, &mut ticks, 0, 10);
            res.elapsed = 2.0; // triggers DerefMut → set_changed
        }
        assert_eq!(ticks.changed, 10);
        assert_eq!(time.elapsed, 2.0);
    }

    #[test]
    fn res_mut_read_does_not_mark_changed() {
        let mut time = Time {
            delta: 0.016,
            elapsed: 1.0,
        };
        let mut ticks = ResourceTicks::new(1);
        {
            let res = ResMut::new(&mut time, &mut ticks, 0, 10);
            let _d = res.delta; // Deref only, not DerefMut
        }
        assert_eq!(ticks.changed, 1);
    }

    #[test]
    fn res_mut_bypass_change_detection() {
        let mut time = Time {
            delta: 0.016,
            elapsed: 1.0,
        };
        let mut ticks = ResourceTicks::new(1);
        {
            let mut res = ResMut::new(&mut time, &mut ticks, 0, 10);
            let inner = res.bypass_change_detection();
            inner.elapsed = 5.0;
        }
        // bypass should not mark changed.
        assert_eq!(ticks.changed, 1);
        assert_eq!(time.elapsed, 5.0);
    }

    #[test]
    fn res_mut_is_added_and_changed() {
        let mut time = Time {
            delta: 0.016,
            elapsed: 1.0,
        };
        let mut ticks = ResourceTicks::new(5);
        let res = ResMut::new(&mut time, &mut ticks, 3, 6);
        assert!(res.is_added());
        assert!(res.is_changed());
    }

    #[test]
    fn res_mut_set_changed_manually() {
        let mut score = Score(0);
        let mut ticks = ResourceTicks::new(1);
        {
            let mut res = ResMut::new(&mut score, &mut ticks, 0, 10);
            res.set_changed();
        }
        assert_eq!(ticks.changed, 10);
    }

    #[test]
    fn res_mut_into_inner() {
        let mut score = Score(0);
        let mut ticks = ResourceTicks::new(1);
        let res = ResMut::new(&mut score, &mut ticks, 0, 2);
        let inner: &mut Score = res.into_inner();
        inner.0 = 42;
        assert_eq!(score.0, 42);
    }

    #[test]
    fn resource_with_world() {
        let mut world = crate::World::new();
        world.add_resource(Time {
            delta: 0.016,
            elapsed: 0.0,
        });

        {
            let time = world.get_resource::<Time>().unwrap();
            assert_eq!(time.delta, 0.016);
        }

        {
            let time = world.get_resource_mut::<Time>().unwrap();
            assert_eq!(time.delta, 0.016);
        }
    }

    #[test]
    fn resource_ticks_as_component_ticks() {
        let ticks = ResourceTicks::new(5);
        let ct = ticks.as_component_ticks();
        assert_eq!(ct.added, 5);
        assert_eq!(ct.changed, 5);
    }

    #[test]
    fn res_debug_format() {
        let score = Score(42);
        let ticks = ResourceTicks::new(1);
        let res = Res::new(&score, ticks, 0, 2);
        let debug = format!("{:?}", res);
        assert!(debug.contains("42"));
    }

    #[test]
    fn res_mut_debug_format() {
        let mut score = Score(99);
        let mut ticks = ResourceTicks::new(1);
        let res = ResMut::new(&mut score, &mut ticks, 0, 2);
        let debug = format!("{:?}", res);
        assert!(debug.contains("99"));
    }
}
