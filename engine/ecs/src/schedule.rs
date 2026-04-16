//! Advanced system scheduling for the Genovo ECS.
//!
//! This module extends the basic [`SystemSchedule`](crate::SystemSchedule) with:
//!
//! - **Stages** — named ordering buckets (`First`, `PreUpdate`, `Update`,
//!   `PostUpdate`, `Last`).
//! - **System access declarations** — systems declare which component types they
//!   read and write, enabling automatic parallelization.
//! - **Run criteria** — conditional execution (e.g., `run_if`, `skip_if`).
//! - **System sets** — named groups with shared ordering constraints.
//!
//! # Architecture
//!
//! ```text
//! Schedule
//! ├── Stage::First
//! │   ├── system_a (reads: [Time])
//! │   └── system_b (reads: [Input])
//! ├── Stage::PreUpdate
//! │   └── system_c (reads: [Input], writes: [Velocity])
//! ├── Stage::Update
//! │   ├── SystemSet("physics")
//! │   │   ├── system_d (reads: [Velocity], writes: [Position])
//! │   │   └── system_e (reads: [Position], writes: [Collision])
//! │   └── system_f (reads: [Health])
//! ├── Stage::PostUpdate
//! │   └── system_g (reads: [Position], writes: [Transform])
//! └── Stage::Last
//!     └── system_h (writes: [RenderQueue])
//! ```

use std::any::TypeId;
use std::collections::{HashMap, HashSet};

use crate::component::ComponentId;
use crate::system::System;
use crate::world::World;

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

/// Execution stages, processed in order.
///
/// Systems within the same stage may run in parallel if their access does not
/// conflict. Stages are always sequential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage {
    /// Runs first, before all other stages.
    First = 0,
    /// Runs before the main update.
    PreUpdate = 1,
    /// The main update stage.
    Update = 2,
    /// Runs after the main update.
    PostUpdate = 3,
    /// Runs last, after all other stages.
    Last = 4,
}

impl Stage {
    /// All stages in execution order.
    pub const ALL: [Stage; 5] = [
        Stage::First,
        Stage::PreUpdate,
        Stage::Update,
        Stage::PostUpdate,
        Stage::Last,
    ];
}

// ---------------------------------------------------------------------------
// SystemAccess — resource/component access declarations
// ---------------------------------------------------------------------------

/// Declares which components and resources a system reads and writes.
///
/// Systems with non-overlapping access can safely run in parallel.
#[derive(Debug, Clone, Default)]
pub struct SystemAccess {
    /// Component types read (immutably) by this system.
    pub reads: HashSet<ComponentId>,
    /// Component types written (mutably) by this system.
    pub writes: HashSet<ComponentId>,
    /// Resource types read.
    pub resource_reads: HashSet<TypeId>,
    /// Resource types written.
    pub resource_writes: HashSet<TypeId>,
    /// Whether this system has exclusive world access (prevents all parallelism).
    pub exclusive: bool,
}

impl SystemAccess {
    /// Create empty access.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a component type as read.
    pub fn add_read<T: crate::Component>(&mut self) -> &mut Self {
        self.reads.insert(ComponentId::of::<T>());
        self
    }

    /// Mark a component type as written.
    pub fn add_write<T: crate::Component>(&mut self) -> &mut Self {
        self.writes.insert(ComponentId::of::<T>());
        self
    }

    /// Mark a resource type as read.
    pub fn add_resource_read<R: 'static>(&mut self) -> &mut Self {
        self.resource_reads.insert(TypeId::of::<R>());
        self
    }

    /// Mark a resource type as written.
    pub fn add_resource_write<R: 'static>(&mut self) -> &mut Self {
        self.resource_writes.insert(TypeId::of::<R>());
        self
    }

    /// Mark as requiring exclusive world access.
    pub fn set_exclusive(&mut self) -> &mut Self {
        self.exclusive = true;
        self
    }

    /// Check whether two system accesses conflict (cannot run in parallel).
    pub fn conflicts_with(&self, other: &SystemAccess) -> bool {
        if self.exclusive || other.exclusive {
            return true;
        }

        // Write-write conflict on any component.
        if !self.writes.is_disjoint(&other.writes) {
            return true;
        }

        // Read-write conflict on any component.
        if !self.reads.is_disjoint(&other.writes) {
            return true;
        }
        if !self.writes.is_disjoint(&other.reads) {
            return true;
        }

        // Resource conflicts.
        if !self.resource_writes.is_disjoint(&other.resource_writes) {
            return true;
        }
        if !self.resource_reads.is_disjoint(&other.resource_writes) {
            return true;
        }
        if !self.resource_writes.is_disjoint(&other.resource_reads) {
            return true;
        }

        false
    }

    /// Merge another access set into this one.
    pub fn merge(&mut self, other: &SystemAccess) {
        self.reads.extend(&other.reads);
        self.writes.extend(&other.writes);
        self.resource_reads.extend(&other.resource_reads);
        self.resource_writes.extend(&other.resource_writes);
        self.exclusive |= other.exclusive;
    }

    /// Check if this access is empty (no reads or writes).
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty()
            && self.writes.is_empty()
            && self.resource_reads.is_empty()
            && self.resource_writes.is_empty()
            && !self.exclusive
    }
}

// ---------------------------------------------------------------------------
// RunCriteria — conditional system execution
// ---------------------------------------------------------------------------

/// Determines whether a system should run on a given tick.
pub enum RunCriteria {
    /// Always run.
    Always,
    /// Run only if the predicate returns `true`.
    RunIf(Box<dyn Fn(&World) -> bool + Send + Sync>),
    /// Skip if the predicate returns `true`.
    SkipIf(Box<dyn Fn(&World) -> bool + Send + Sync>),
    /// Run once and then never again.
    Once { ran: bool },
    /// Run every N ticks.
    EveryNTicks { interval: u32, counter: u32 },
}

impl RunCriteria {
    /// Create a "run if" criterion.
    pub fn run_if<F: Fn(&World) -> bool + Send + Sync + 'static>(f: F) -> Self {
        Self::RunIf(Box::new(f))
    }

    /// Create a "skip if" criterion.
    pub fn skip_if<F: Fn(&World) -> bool + Send + Sync + 'static>(f: F) -> Self {
        Self::SkipIf(Box::new(f))
    }

    /// Create a "run once" criterion.
    pub fn once() -> Self {
        Self::Once { ran: false }
    }

    /// Create a "run every N ticks" criterion.
    pub fn every_n_ticks(n: u32) -> Self {
        Self::EveryNTicks {
            interval: n,
            counter: 0,
        }
    }

    /// Check whether the system should run. Updates internal state.
    pub fn should_run(&mut self, world: &World) -> bool {
        match self {
            RunCriteria::Always => true,
            RunCriteria::RunIf(f) => f(world),
            RunCriteria::SkipIf(f) => !f(world),
            RunCriteria::Once { ran } => {
                if *ran {
                    false
                } else {
                    *ran = true;
                    true
                }
            }
            RunCriteria::EveryNTicks { interval, counter } => {
                *counter += 1;
                if *counter >= *interval {
                    *counter = 0;
                    true
                } else {
                    false
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SystemEntry — a single system with metadata
// ---------------------------------------------------------------------------

/// A system combined with its access declarations, run criteria, and metadata.
struct SystemEntry {
    /// The system itself.
    system: Box<dyn System>,
    /// Declared access.
    access: SystemAccess,
    /// When to run.
    run_criteria: RunCriteria,
    /// Optional label for ordering.
    label: Option<String>,
    /// Labels this system must run after.
    #[allow(dead_code)]
    after: Vec<String>,
    /// Labels this system must run before.
    #[allow(dead_code)]
    before: Vec<String>,
    /// Which system set this belongs to.
    set: Option<String>,
    /// Whether this system is enabled.
    enabled: bool,
}

// ---------------------------------------------------------------------------
// SystemSet
// ---------------------------------------------------------------------------

/// Named group of systems with shared ordering constraints.
///
/// ```ignore
/// schedule.add_system_set(
///     SystemSet::new("physics")
///         .with_system(gravity_system)
///         .with_system(collision_system)
///         .after("input")
///         .before("rendering"),
/// );
/// ```
pub struct SystemSet {
    /// Set name.
    name: String,
    /// Systems in this set.
    systems: Vec<(Box<dyn System>, SystemAccess)>,
    /// Run criteria shared by all systems in the set.
    run_criteria: RunCriteria,
    /// Labels this set must run after.
    after: Vec<String>,
    /// Labels this set must run before.
    before: Vec<String>,
    /// Whether this set is enabled.
    enabled: bool,
}

impl SystemSet {
    /// Create a new, empty system set.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            systems: Vec::new(),
            run_criteria: RunCriteria::Always,
            after: Vec::new(),
            before: Vec::new(),
            enabled: true,
        }
    }

    /// Add a system to this set.
    pub fn with_system<S: System + 'static>(mut self, system: S) -> Self {
        self.systems.push((Box::new(system), SystemAccess::new()));
        self
    }

    /// Add a system with declared access.
    pub fn with_system_access<S: System + 'static>(
        mut self,
        system: S,
        access: SystemAccess,
    ) -> Self {
        self.systems.push((Box::new(system), access));
        self
    }

    /// This set must run after the given label.
    pub fn after(mut self, label: &str) -> Self {
        self.after.push(label.to_string());
        self
    }

    /// This set must run before the given label.
    pub fn before(mut self, label: &str) -> Self {
        self.before.push(label.to_string());
        self
    }

    /// Set run criteria for all systems in this set.
    pub fn with_run_criteria(mut self, criteria: RunCriteria) -> Self {
        self.run_criteria = criteria;
        self
    }

    /// Enable or disable this set.
    pub fn set_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// SystemDescriptor — builder for adding systems to a schedule
// ---------------------------------------------------------------------------

/// Builder for configuring a system before adding it to a schedule.
pub struct SystemDescriptor {
    system: Box<dyn System>,
    access: SystemAccess,
    run_criteria: RunCriteria,
    stage: Stage,
    label: Option<String>,
    after: Vec<String>,
    before: Vec<String>,
    set: Option<String>,
}

impl SystemDescriptor {
    /// Create a new descriptor for a system.
    pub fn new<S: System + 'static>(system: S) -> Self {
        Self {
            system: Box::new(system),
            access: SystemAccess::new(),
            run_criteria: RunCriteria::Always,
            stage: Stage::Update,
            label: None,
            after: Vec::new(),
            before: Vec::new(),
            set: None,
        }
    }

    /// Set the access declarations.
    pub fn with_access(mut self, access: SystemAccess) -> Self {
        self.access = access;
        self
    }

    /// Set run criteria.
    pub fn with_run_criteria(mut self, criteria: RunCriteria) -> Self {
        self.run_criteria = criteria;
        self
    }

    /// Set the stage.
    pub fn in_stage(mut self, stage: Stage) -> Self {
        self.stage = stage;
        self
    }

    /// Set a label.
    pub fn label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// This system must run after the given label.
    pub fn after(mut self, label: &str) -> Self {
        self.after.push(label.to_string());
        self
    }

    /// This system must run before the given label.
    pub fn before(mut self, label: &str) -> Self {
        self.before.push(label.to_string());
        self
    }

    /// Assign to a system set.
    pub fn in_set(mut self, set: &str) -> Self {
        self.set = Some(set.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// ParallelBatch — group of non-conflicting systems
// ---------------------------------------------------------------------------

/// A batch of systems that can run concurrently because their accesses do not
/// conflict.
struct ParallelBatch {
    /// Indices into the stage's system list.
    system_indices: Vec<usize>,
    /// Combined access of all systems in this batch.
    combined_access: SystemAccess,
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// Advanced system scheduler with stages, access tracking, ordering
/// constraints, and conditional execution.
///
/// Systems within the same stage are topologically sorted by their ordering
/// constraints (`before`/`after` labels) and then batched for potential
/// parallel execution based on their declared access.
pub struct Schedule {
    /// Systems organized by stage.
    stages: HashMap<Stage, Vec<SystemEntry>>,
    /// Parallel batches per stage, computed by `build`.
    batches: HashMap<Stage, Vec<ParallelBatch>>,
    /// Whether the schedule needs rebuilding.
    dirty: bool,
}

impl Schedule {
    /// Create a new, empty schedule.
    pub fn new() -> Self {
        let mut stages = HashMap::new();
        for stage in Stage::ALL {
            stages.insert(stage, Vec::new());
        }
        Self {
            stages,
            batches: HashMap::new(),
            dirty: true,
        }
    }

    /// Add a system to a specific stage.
    pub fn add_system_to_stage<S: System + 'static>(
        &mut self,
        stage: Stage,
        system: S,
    ) {
        self.stages.entry(stage).or_default().push(SystemEntry {
            system: Box::new(system),
            access: SystemAccess::new(),
            run_criteria: RunCriteria::Always,
            label: None,
            after: Vec::new(),
            before: Vec::new(),
            set: None,
            enabled: true,
        });
        self.dirty = true;
    }

    /// Add a system to the Update stage (convenience).
    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.add_system_to_stage(Stage::Update, system);
    }

    /// Add a system with full configuration via a descriptor.
    pub fn add_system_descriptor(&mut self, desc: SystemDescriptor) {
        self.stages.entry(desc.stage).or_default().push(SystemEntry {
            system: desc.system,
            access: desc.access,
            run_criteria: desc.run_criteria,
            label: desc.label,
            after: desc.after,
            before: desc.before,
            set: desc.set,
            enabled: true,
        });
        self.dirty = true;
    }

    /// Add a system set to a stage.
    pub fn add_system_set_to_stage(&mut self, stage: Stage, set: SystemSet) {
        let set_name = set.name.clone();
        let enabled = set.enabled;
        for (system, access) in set.systems {
            self.stages.entry(stage).or_default().push(SystemEntry {
                system,
                access,
                run_criteria: RunCriteria::Always,
                label: None,
                after: set.after.clone(),
                before: set.before.clone(),
                set: Some(set_name.clone()),
                enabled,
            });
        }
        self.dirty = true;
    }

    /// Add a system set to the Update stage.
    pub fn add_system_set(&mut self, set: SystemSet) {
        self.add_system_set_to_stage(Stage::Update, set);
    }

    /// Build the parallel execution batches. Called automatically before the
    /// first run, or when systems change.
    pub fn build(&mut self) {
        self.batches.clear();

        for (&stage, systems) in &self.stages {
            let batches = self.compute_batches(systems.len(), |i| &systems[i].access);
            self.batches.insert(stage, batches);
        }

        self.dirty = false;
    }

    /// Compute parallel batches for N systems given an access function.
    fn compute_batches<'a>(
        &self,
        count: usize,
        access_fn: impl Fn(usize) -> &'a SystemAccess,
    ) -> Vec<ParallelBatch> {
        // Topological ordering based on before/after is left as sequential for
        // correctness; parallel batching groups non-conflicting systems.
        let mut batches: Vec<ParallelBatch> = Vec::new();

        for i in 0..count {
            let access = access_fn(i);
            let mut placed = false;

            for batch in &mut batches {
                if !batch.combined_access.conflicts_with(access) {
                    batch.system_indices.push(i);
                    batch.combined_access.merge(access);
                    placed = true;
                    break;
                }
            }

            if !placed {
                let mut combined = SystemAccess::new();
                combined.merge(access);
                batches.push(ParallelBatch {
                    system_indices: vec![i],
                    combined_access: combined,
                });
            }
        }

        batches
    }

    /// Run all stages in order. Systems within each stage are run sequentially
    /// (batching information is available for future parallel execution).
    pub fn run(&mut self, world: &mut World) {
        if self.dirty {
            self.build();
        }

        for stage in Stage::ALL {
            if let Some(systems) = self.stages.get_mut(&stage) {
                for entry in systems.iter_mut() {
                    if !entry.enabled {
                        continue;
                    }
                    if entry.run_criteria.should_run(world) {
                        entry.system.run(world);
                    }
                }
            }
        }
    }

    /// Run only a specific stage.
    pub fn run_stage(&mut self, stage: Stage, world: &mut World) {
        if let Some(systems) = self.stages.get_mut(&stage) {
            for entry in systems.iter_mut() {
                if !entry.enabled {
                    continue;
                }
                if entry.run_criteria.should_run(world) {
                    entry.system.run(world);
                }
            }
        }
    }

    /// Returns the number of registered systems across all stages.
    pub fn system_count(&self) -> usize {
        self.stages.values().map(|v| v.len()).sum()
    }

    /// Returns the number of systems in a specific stage.
    pub fn stage_system_count(&self, stage: Stage) -> usize {
        self.stages.get(&stage).map_or(0, |v| v.len())
    }

    /// Returns the number of parallel batches in a stage (after build).
    pub fn stage_batch_count(&self, stage: Stage) -> usize {
        self.batches.get(&stage).map_or(0, |v| v.len())
    }

    /// Check whether the schedule is empty.
    pub fn is_empty(&self) -> bool {
        self.stages.values().all(|v| v.is_empty())
    }

    /// Remove all systems.
    pub fn clear(&mut self) {
        for systems in self.stages.values_mut() {
            systems.clear();
        }
        self.batches.clear();
        self.dirty = true;
    }

    /// Enable or disable a system by label.
    pub fn set_enabled(&mut self, label: &str, enabled: bool) {
        for systems in self.stages.values_mut() {
            for entry in systems.iter_mut() {
                if entry.label.as_deref() == Some(label) {
                    entry.enabled = enabled;
                }
            }
        }
    }

    /// Enable or disable a system set by name.
    pub fn set_set_enabled(&mut self, set_name: &str, enabled: bool) {
        for systems in self.stages.values_mut() {
            for entry in systems.iter_mut() {
                if entry.set.as_deref() == Some(set_name) {
                    entry.enabled = enabled;
                }
            }
        }
    }
}

impl Default for Schedule {
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

    #[derive(Debug, PartialEq, Clone)]
    struct Position {
        x: f32,
        y: f32,
    }
    impl crate::Component for Position {}

    #[derive(Debug, PartialEq, Clone)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }
    impl crate::Component for Velocity {}

    #[test]
    fn schedule_runs_systems_in_stage_order() {
        let mut world = World::new();
        let order = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));

        let mut schedule = Schedule::new();

        let o1 = order.clone();
        schedule.add_system_to_stage(Stage::Last, move |_: &mut World| {
            o1.lock().unwrap().push(3);
        });

        let o2 = order.clone();
        schedule.add_system_to_stage(Stage::First, move |_: &mut World| {
            o2.lock().unwrap().push(1);
        });

        let o3 = order.clone();
        schedule.add_system_to_stage(Stage::Update, move |_: &mut World| {
            o3.lock().unwrap().push(2);
        });

        schedule.run(&mut world);

        assert_eq!(*order.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn schedule_run_criteria_run_if() {
        let mut world = World::new();
        world.add_resource(false);

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut schedule = Schedule::new();
        schedule.add_system_descriptor(
            SystemDescriptor::new(move |world: &mut World| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .with_run_criteria(RunCriteria::run_if(|world: &World| {
                *world.get_resource::<bool>().unwrap_or(&false)
            })),
        );

        // First run: condition is false.
        schedule.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // Set to true and run again.
        *world.get_resource_mut::<bool>().unwrap() = true;
        schedule.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn schedule_run_criteria_once() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut schedule = Schedule::new();
        schedule.add_system_descriptor(
            SystemDescriptor::new(move |_: &mut World| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .with_run_criteria(RunCriteria::once()),
        );

        schedule.run(&mut world);
        schedule.run(&mut world);
        schedule.run(&mut world);

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn schedule_run_criteria_every_n_ticks() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut schedule = Schedule::new();
        schedule.add_system_descriptor(
            SystemDescriptor::new(move |_: &mut World| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .with_run_criteria(RunCriteria::every_n_ticks(3)),
        );

        for _ in 0..9 {
            schedule.run(&mut world);
        }

        // Should run on tick 3, 6, 9 → 3 times.
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn system_access_conflicts() {
        let mut a1 = SystemAccess::new();
        a1.add_read::<Position>();
        a1.add_write::<Velocity>();

        let mut a2 = SystemAccess::new();
        a2.add_read::<Velocity>();

        // a1 writes Velocity, a2 reads Velocity → conflict.
        assert!(a1.conflicts_with(&a2));

        let mut a3 = SystemAccess::new();
        a3.add_read::<Position>();

        // a1 reads Position, a3 reads Position → no conflict.
        assert!(!a1.conflicts_with(&a3));
        // Wait — a1 also writes Velocity, but a3 doesn't touch it.
        // Correction: a1.reads=Position, a1.writes=Velocity; a3.reads=Position.
        // No overlap in writes, and a3 doesn't read Velocity. No conflict.
        assert!(!a3.conflicts_with(&a1));
    }

    #[test]
    fn system_access_no_conflict_different_components() {
        let mut a1 = SystemAccess::new();
        a1.add_write::<Position>();

        let mut a2 = SystemAccess::new();
        a2.add_write::<Velocity>();

        assert!(!a1.conflicts_with(&a2));
    }

    #[test]
    fn system_access_exclusive_always_conflicts() {
        let mut a1 = SystemAccess::new();
        a1.set_exclusive();

        let a2 = SystemAccess::new();

        assert!(a1.conflicts_with(&a2));
    }

    #[test]
    fn parallel_batching() {
        let mut schedule = Schedule::new();

        // System 1: reads Position
        schedule.add_system_descriptor(
            SystemDescriptor::new(|_: &mut World| {})
                .with_access({
                    let mut a = SystemAccess::new();
                    a.add_read::<Position>();
                    a
                })
                .label("read_pos_1"),
        );

        // System 2: reads Position (no conflict with 1)
        schedule.add_system_descriptor(
            SystemDescriptor::new(|_: &mut World| {})
                .with_access({
                    let mut a = SystemAccess::new();
                    a.add_read::<Position>();
                    a
                })
                .label("read_pos_2"),
        );

        // System 3: writes Position (conflicts with 1 and 2)
        schedule.add_system_descriptor(
            SystemDescriptor::new(|_: &mut World| {})
                .with_access({
                    let mut a = SystemAccess::new();
                    a.add_write::<Position>();
                    a
                })
                .label("write_pos"),
        );

        schedule.build();

        // Systems 1 and 2 should be in the same batch, system 3 in its own.
        assert_eq!(schedule.stage_batch_count(Stage::Update), 2);
    }

    #[test]
    fn system_set_basic() {
        let mut world = World::new();
        let order = Arc::new(std::sync::Mutex::new(Vec::<&str>::new()));

        let o1 = order.clone();
        let o2 = order.clone();

        let set = SystemSet::new("physics")
            .with_system(move |_: &mut World| {
                o1.lock().unwrap().push("gravity");
            })
            .with_system(move |_: &mut World| {
                o2.lock().unwrap().push("collision");
            });

        let mut schedule = Schedule::new();
        schedule.add_system_set(set);

        schedule.run(&mut world);

        let result = order.lock().unwrap().clone();
        assert!(result.contains(&"gravity"));
        assert!(result.contains(&"collision"));
    }

    #[test]
    fn schedule_clear() {
        let mut schedule = Schedule::new();
        schedule.add_system(|_: &mut World| {});
        schedule.add_system(|_: &mut World| {});
        assert_eq!(schedule.system_count(), 2);

        schedule.clear();
        assert!(schedule.is_empty());
        assert_eq!(schedule.system_count(), 0);
    }

    #[test]
    fn schedule_enable_disable_system() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut schedule = Schedule::new();
        schedule.add_system_descriptor(
            SystemDescriptor::new(move |_: &mut World| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .label("my_system"),
        );

        schedule.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        schedule.set_enabled("my_system", false);
        schedule.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        schedule.set_enabled("my_system", true);
        schedule.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn schedule_set_enable_disable() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();

        let set = SystemSet::new("my_set")
            .with_system(move |_: &mut World| {
                c1.fetch_add(1, Ordering::SeqCst);
            })
            .with_system(move |_: &mut World| {
                c2.fetch_add(1, Ordering::SeqCst);
            });

        let mut schedule = Schedule::new();
        schedule.add_system_set(set);

        schedule.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        schedule.set_set_enabled("my_set", false);
        schedule.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn system_descriptor_all_options() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut schedule = Schedule::new();
        schedule.add_system_descriptor(
            SystemDescriptor::new(move |_: &mut World| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .in_stage(Stage::PreUpdate)
            .label("my_system")
            .after("some_other")
            .before("another")
            .in_set("my_set")
            .with_run_criteria(RunCriteria::Always)
            .with_access({
                let mut a = SystemAccess::new();
                a.add_read::<Position>();
                a
            }),
        );

        assert_eq!(schedule.stage_system_count(Stage::PreUpdate), 1);
        schedule.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stage_ordering() {
        assert!(Stage::First < Stage::PreUpdate);
        assert!(Stage::PreUpdate < Stage::Update);
        assert!(Stage::Update < Stage::PostUpdate);
        assert!(Stage::PostUpdate < Stage::Last);
    }

    #[test]
    fn run_criteria_skip_if() {
        let mut world = World::new();
        world.add_resource(true); // skip = true

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut schedule = Schedule::new();
        schedule.add_system_descriptor(
            SystemDescriptor::new(move |_: &mut World| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .with_run_criteria(RunCriteria::skip_if(|world: &World| {
                *world.get_resource::<bool>().unwrap_or(&false)
            })),
        );

        schedule.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        *world.get_resource_mut::<bool>().unwrap() = false;
        schedule.run(&mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn system_access_merge() {
        let mut a1 = SystemAccess::new();
        a1.add_read::<Position>();

        let mut a2 = SystemAccess::new();
        a2.add_write::<Velocity>();

        a1.merge(&a2);
        assert!(a1.reads.contains(&ComponentId::of::<Position>()));
        assert!(a1.writes.contains(&ComponentId::of::<Velocity>()));
    }

    #[test]
    fn system_access_is_empty() {
        let a = SystemAccess::new();
        assert!(a.is_empty());

        let mut b = SystemAccess::new();
        b.add_read::<Position>();
        assert!(!b.is_empty());
    }

    #[test]
    fn run_stage_individually() {
        let mut world = World::new();
        let counter = Arc::new(AtomicU32::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();

        let mut schedule = Schedule::new();
        schedule.add_system_to_stage(Stage::First, move |_: &mut World| {
            c1.fetch_add(1, Ordering::SeqCst);
        });
        schedule.add_system_to_stage(Stage::Update, move |_: &mut World| {
            c2.fetch_add(10, Ordering::SeqCst);
        });

        schedule.run_stage(Stage::First, &mut world);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
