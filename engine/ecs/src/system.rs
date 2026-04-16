//! System scheduling for the Genovo ECS.
//!
//! Systems are units of game logic that operate on entity data via the World.
//! The scheduler arranges systems into an ordered list and runs them
//! sequentially.

// ---------------------------------------------------------------------------
// System trait
// ---------------------------------------------------------------------------

/// Trait that all systems must implement.
///
/// A system reads/writes component data through the [`World`](crate::World)
/// reference. The `run` method takes `&mut World` so systems can modify
/// entities and components.
pub trait System: Send + Sync {
    /// Execute this system for one tick.
    fn run(&mut self, world: &mut crate::World);
}

/// Blanket implementation that lets `FnMut(&mut World)` closures be used as
/// systems for prototyping.
impl<F> System for F
where
    F: FnMut(&mut crate::World) + Send + Sync,
{
    fn run(&mut self, world: &mut crate::World) {
        (self)(world);
    }
}

// ---------------------------------------------------------------------------
// SystemSchedule
// ---------------------------------------------------------------------------

/// A simple ordered list of systems that are run sequentially.
///
/// Systems are executed in the order they were added. For parallel or
/// stage-based execution, a more advanced scheduler can be built on top.
pub struct SystemSchedule {
    systems: Vec<Box<dyn System>>,
}

impl SystemSchedule {
    /// Create an empty schedule.
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
        }
    }

    /// Add a system to the end of the schedule.
    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.systems.push(Box::new(system));
    }

    /// Run all systems in order, passing `&mut World` to each.
    pub fn run_all(&mut self, world: &mut crate::World) {
        for system in &mut self.systems {
            system.run(world);
        }
    }

    /// Returns the number of registered systems.
    #[inline]
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Whether there are no registered systems.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// Remove all systems from the schedule.
    pub fn clear(&mut self) {
        self.systems.clear();
    }
}

impl Default for SystemSchedule {
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
    struct Counter(u32);
    impl crate::Component for Counter {}

    #[test]
    fn run_closure_system() {
        let mut world = crate::World::new();
        let mut schedule = SystemSchedule::new();

        let ran = Arc::new(AtomicU32::new(0));
        let ran_clone = ran.clone();

        schedule.add_system(move |_w: &mut crate::World| {
            ran_clone.fetch_add(1, Ordering::SeqCst);
        });

        schedule.run_all(&mut world);
        assert_eq!(ran.load(Ordering::SeqCst), 1);

        schedule.run_all(&mut world);
        assert_eq!(ran.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn system_modifies_world() {
        let mut world = crate::World::new();
        let e = world.spawn_entity().with(Counter(0)).build();

        let mut schedule = SystemSchedule::new();
        schedule.add_system(move |world: &mut crate::World| {
            if let Some(counter) = world.get_component_mut::<Counter>(e) {
                counter.0 += 1;
            }
        });

        schedule.run_all(&mut world);
        assert_eq!(world.get_component::<Counter>(e).map(|c| c.0), Some(1));

        schedule.run_all(&mut world);
        assert_eq!(world.get_component::<Counter>(e).map(|c| c.0), Some(2));
    }

    #[test]
    fn systems_run_in_order() {
        let mut world = crate::World::new();
        let order = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));

        let mut schedule = SystemSchedule::new();

        let o1 = order.clone();
        schedule.add_system(move |_: &mut crate::World| {
            o1.lock().unwrap().push(1);
        });

        let o2 = order.clone();
        schedule.add_system(move |_: &mut crate::World| {
            o2.lock().unwrap().push(2);
        });

        let o3 = order.clone();
        schedule.add_system(move |_: &mut crate::World| {
            o3.lock().unwrap().push(3);
        });

        schedule.run_all(&mut world);
        assert_eq!(*order.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn len_and_clear() {
        let mut schedule = SystemSchedule::new();
        assert!(schedule.is_empty());
        assert_eq!(schedule.len(), 0);

        schedule.add_system(|_: &mut crate::World| {});
        schedule.add_system(|_: &mut crate::World| {});
        assert_eq!(schedule.len(), 2);

        schedule.clear();
        assert!(schedule.is_empty());
    }

    /// Test a struct-based system.
    #[test]
    fn struct_system() {
        struct GravitySystem {
            gravity: f32,
        }

        #[derive(Debug, PartialEq, Clone)]
        struct YPos(f32);
        impl crate::Component for YPos {}

        impl System for GravitySystem {
            fn run(&mut self, world: &mut crate::World) {
                // Collect entity ids first to avoid borrow issues.
                let entities: Vec<crate::Entity> = world
                    .query::<&YPos>()
                    .map(|(e, _)| e)
                    .collect();

                for entity in entities {
                    if let Some(y) = world.get_component_mut::<YPos>(entity) {
                        y.0 += self.gravity;
                    }
                }
            }
        }

        let mut world = crate::World::new();
        let e = world.spawn_entity().with(YPos(100.0)).build();

        let mut schedule = SystemSchedule::new();
        schedule.add_system(GravitySystem { gravity: -9.8 });

        schedule.run_all(&mut world);
        let y = world.get_component::<YPos>(e).unwrap().0;
        assert!((y - 90.2).abs() < 0.001);
    }
}
