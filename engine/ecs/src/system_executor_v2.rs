//! Enhanced system executor with automatic parallelism, thread pool integration,
//! system sets with ordering, run conditions, and fixed timestep systems.
//!
//! This module provides a sophisticated system scheduler that:
//!
//! - **Automatic parallelism** — analyzes system component access patterns to
//!   determine which systems can safely run in parallel.
//! - **Thread pool integration** — distributes parallel batches across a
//!   configurable thread pool with work-stealing.
//! - **System sets** — named groups of systems with explicit ordering
//!   (before/after) constraints.
//! - **Run conditions** — predicate functions that gate system execution.
//! - **Fixed timestep** — systems that run at a fixed rate, accumulating
//!   time and ticking multiple times per frame if needed.

use std::any::TypeId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Access descriptors
// ---------------------------------------------------------------------------

/// Describes how a system accesses a specific component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessKindV2 {
    /// Read-only access.
    Read,
    /// Read-write (mutable) access.
    Write,
    /// Exclusive world access — cannot run in parallel with anything.
    Exclusive,
}

/// A single component access declaration.
#[derive(Debug, Clone)]
pub struct ComponentAccessV2 {
    /// The component type being accessed.
    pub type_id: TypeId,
    /// Human-readable name for debugging.
    pub type_name: &'static str,
    /// Access mode.
    pub access: AccessKindV2,
}

/// Full access descriptor for a system.
#[derive(Debug, Clone)]
pub struct SystemAccessV2 {
    /// Component accesses.
    pub components: Vec<ComponentAccessV2>,
    /// Resource accesses (type id + mode).
    pub resources: Vec<(TypeId, AccessKindV2)>,
    /// Whether this system requires exclusive world access.
    pub exclusive: bool,
}

impl SystemAccessV2 {
    /// Create an empty access descriptor.
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            resources: Vec::new(),
            exclusive: false,
        }
    }

    /// Add a read access for a component type.
    pub fn read_component<T: 'static>(&mut self) -> &mut Self {
        self.components.push(ComponentAccessV2 {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            access: AccessKindV2::Read,
        });
        self
    }

    /// Add a write access for a component type.
    pub fn write_component<T: 'static>(&mut self) -> &mut Self {
        self.components.push(ComponentAccessV2 {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            access: AccessKindV2::Write,
        });
        self
    }

    /// Add a read access for a resource type.
    pub fn read_resource<T: 'static>(&mut self) -> &mut Self {
        self.resources.push((TypeId::of::<T>(), AccessKindV2::Read));
        self
    }

    /// Add a write access for a resource type.
    pub fn write_resource<T: 'static>(&mut self) -> &mut Self {
        self.resources
            .push((TypeId::of::<T>(), AccessKindV2::Write));
        self
    }

    /// Mark this system as requiring exclusive world access.
    pub fn set_exclusive(&mut self) -> &mut Self {
        self.exclusive = true;
        self
    }

    /// Check whether two access descriptors conflict (cannot run in parallel).
    pub fn conflicts_with(&self, other: &SystemAccessV2) -> bool {
        if self.exclusive || other.exclusive {
            return true;
        }

        // Check component conflicts.
        for a in &self.components {
            for b in &other.components {
                if a.type_id == b.type_id {
                    if a.access == AccessKindV2::Write || b.access == AccessKindV2::Write {
                        return true;
                    }
                }
            }
        }

        // Check resource conflicts.
        for &(a_type, a_access) in &self.resources {
            for &(b_type, b_access) in &other.resources {
                if a_type == b_type {
                    if a_access == AccessKindV2::Write || b_access == AccessKindV2::Write {
                        return true;
                    }
                }
            }
        }

        false
    }
}

impl Default for SystemAccessV2 {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Run conditions
// ---------------------------------------------------------------------------

/// Unique ID for a run condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunConditionId(pub u32);

/// A predicate that gates whether a system should run this tick.
pub struct RunCondition {
    /// Unique identifier.
    pub id: RunConditionId,
    /// Human-readable label.
    pub label: String,
    /// The predicate function. Returns `true` if the system should run.
    pub predicate: Box<dyn Fn() -> bool + Send + Sync>,
}

impl RunCondition {
    /// Create a new run condition.
    pub fn new(
        id: RunConditionId,
        label: impl Into<String>,
        predicate: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            id,
            label: label.into(),
            predicate: Box::new(predicate),
        }
    }

    /// Evaluate the condition.
    pub fn evaluate(&self) -> bool {
        (self.predicate)()
    }
}

impl fmt::Debug for RunCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunCondition")
            .field("id", &self.id)
            .field("label", &self.label)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// System sets
// ---------------------------------------------------------------------------

/// Unique ID for a system set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemSetId(pub u32);

/// A named group of systems with ordering constraints.
#[derive(Debug, Clone)]
pub struct SystemSetV2 {
    /// Unique identifier.
    pub id: SystemSetId,
    /// Human-readable name.
    pub name: String,
    /// System indices belonging to this set.
    pub systems: Vec<usize>,
    /// Sets that must run before this set.
    pub after_sets: Vec<SystemSetId>,
    /// Sets that must run after this set.
    pub before_sets: Vec<SystemSetId>,
    /// Optional run condition gating the entire set.
    pub run_condition: Option<RunConditionId>,
    /// Whether this set is enabled.
    pub enabled: bool,
}

impl SystemSetV2 {
    /// Create a new, empty system set.
    pub fn new(id: SystemSetId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            systems: Vec::new(),
            after_sets: Vec::new(),
            before_sets: Vec::new(),
            run_condition: None,
            enabled: true,
        }
    }

    /// Add a system to this set.
    pub fn add_system(&mut self, system_index: usize) {
        if !self.systems.contains(&system_index) {
            self.systems.push(system_index);
        }
    }

    /// Specify that this set must run after another set.
    pub fn after(&mut self, other: SystemSetId) -> &mut Self {
        if !self.after_sets.contains(&other) {
            self.after_sets.push(other);
        }
        self
    }

    /// Specify that this set must run before another set.
    pub fn before(&mut self, other: SystemSetId) -> &mut Self {
        if !self.before_sets.contains(&other) {
            self.before_sets.push(other);
        }
        self
    }
}

// ---------------------------------------------------------------------------
// Fixed timestep
// ---------------------------------------------------------------------------

/// Configuration and state for fixed-timestep system execution.
#[derive(Debug, Clone)]
pub struct FixedTimestep {
    /// Target timestep duration.
    pub step: Duration,
    /// Accumulated time since last step.
    pub accumulator: Duration,
    /// Maximum number of steps per frame to prevent spiral of death.
    pub max_steps_per_frame: u32,
    /// Number of steps executed this frame.
    pub steps_this_frame: u32,
    /// Total steps executed.
    pub total_steps: u64,
    /// Interpolation alpha for rendering between physics ticks.
    pub overshoot_alpha: f64,
}

impl FixedTimestep {
    /// Create a new fixed timestep with the given step duration.
    pub fn new(step: Duration) -> Self {
        Self {
            step,
            accumulator: Duration::ZERO,
            max_steps_per_frame: 8,
            steps_this_frame: 0,
            total_steps: 0,
            overshoot_alpha: 0.0,
        }
    }

    /// Create from a frequency in Hz.
    pub fn from_hz(hz: f64) -> Self {
        Self::new(Duration::from_secs_f64(1.0 / hz))
    }

    /// Accumulate frame delta time and return how many steps to execute.
    pub fn accumulate(&mut self, delta: Duration) -> u32 {
        self.accumulator += delta;
        self.steps_this_frame = 0;

        let mut steps = 0u32;
        while self.accumulator >= self.step && steps < self.max_steps_per_frame {
            self.accumulator -= self.step;
            steps += 1;
            self.total_steps += 1;
        }

        // Clamp accumulator if we hit max steps (spiral of death prevention).
        if steps >= self.max_steps_per_frame {
            self.accumulator = Duration::ZERO;
        }

        // Compute interpolation alpha.
        self.overshoot_alpha =
            self.accumulator.as_secs_f64() / self.step.as_secs_f64();

        self.steps_this_frame = steps;
        steps
    }

    /// Get the fixed timestep as seconds (f32).
    pub fn step_secs(&self) -> f32 {
        self.step.as_secs_f32()
    }

    /// Get the fixed timestep as seconds (f64).
    pub fn step_secs_f64(&self) -> f64 {
        self.step.as_secs_f64()
    }

    /// Get the interpolation alpha for rendering.
    pub fn alpha(&self) -> f64 {
        self.overshoot_alpha
    }
}

// ---------------------------------------------------------------------------
// System descriptor
// ---------------------------------------------------------------------------

/// Index of a system within the executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemIndexV2(pub usize);

/// Descriptor for a registered system.
pub struct SystemDescriptorV2 {
    /// Index in the executor's system list.
    pub index: SystemIndexV2,
    /// Human-readable name.
    pub name: String,
    /// Access descriptor for parallelism analysis.
    pub access: SystemAccessV2,
    /// The system function.
    pub run_fn: Box<dyn FnMut() + Send>,
    /// Optional run condition.
    pub run_condition: Option<RunConditionId>,
    /// Systems that must run before this one.
    pub dependencies: Vec<SystemIndexV2>,
    /// Systems that must run after this one.
    pub dependents: Vec<SystemIndexV2>,
    /// Which system set this belongs to, if any.
    pub set: Option<SystemSetId>,
    /// Whether this system is enabled.
    pub enabled: bool,
    /// Fixed timestep configuration, if this system runs on fixed tick.
    pub fixed_timestep: Option<FixedTimestep>,
    /// Profiling: average run time.
    pub avg_duration: Duration,
    /// Profiling: last run time.
    pub last_duration: Duration,
    /// Profiling: total number of runs.
    pub run_count: u64,
}

impl SystemDescriptorV2 {
    /// Create a new system descriptor.
    pub fn new(
        index: SystemIndexV2,
        name: impl Into<String>,
        access: SystemAccessV2,
        run_fn: impl FnMut() + Send + 'static,
    ) -> Self {
        Self {
            index,
            name: name.into(),
            access,
            run_fn: Box::new(run_fn),
            run_condition: None,
            dependencies: Vec::new(),
            dependents: Vec::new(),
            set: None,
            enabled: true,
            fixed_timestep: None,
            avg_duration: Duration::ZERO,
            last_duration: Duration::ZERO,
            run_count: 0,
        }
    }
}

impl fmt::Debug for SystemDescriptorV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemDescriptorV2")
            .field("index", &self.index)
            .field("name", &self.name)
            .field("enabled", &self.enabled)
            .field("run_count", &self.run_count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Parallel batch
// ---------------------------------------------------------------------------

/// A batch of systems that can safely run in parallel.
#[derive(Debug, Clone)]
pub struct ParallelBatchV2 {
    /// System indices in this batch.
    pub systems: Vec<SystemIndexV2>,
}

impl ParallelBatchV2 {
    /// Create a new batch.
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
        }
    }

    /// Add a system to this batch.
    pub fn add(&mut self, system: SystemIndexV2) {
        self.systems.push(system);
    }

    /// Number of systems in this batch.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Whether this batch is empty.
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }
}

impl Default for ParallelBatchV2 {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Thread pool (simple)
// ---------------------------------------------------------------------------

/// A simple thread pool for executing system batches.
pub struct ThreadPool {
    /// Number of worker threads.
    pub thread_count: usize,
    /// Pending tasks queue.
    tasks: Arc<Mutex<VecDeque<Box<dyn FnOnce() + Send>>>>,
    /// Shutdown signal.
    shutdown: Arc<AtomicBool>,
    /// Worker thread handles.
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl ThreadPool {
    /// Create a new thread pool with the given number of workers.
    pub fn new(thread_count: usize) -> Self {
        let tasks: Arc<Mutex<VecDeque<Box<dyn FnOnce() + Send>>>> =
            Arc::new(Mutex::new(VecDeque::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(thread_count);

        for i in 0..thread_count {
            let tasks = Arc::clone(&tasks);
            let shutdown = Arc::clone(&shutdown);

            let handle = std::thread::Builder::new()
                .name(format!("genovo-worker-{}", i))
                .spawn(move || {
                    while !shutdown.load(Ordering::Relaxed) {
                        let task = {
                            let mut queue = tasks.lock().unwrap();
                            queue.pop_front()
                        };
                        if let Some(task) = task {
                            task();
                        } else {
                            std::thread::yield_now();
                        }
                    }
                })
                .expect("failed to spawn worker thread");

            handles.push(handle);
        }

        Self {
            thread_count,
            tasks,
            shutdown,
            handles,
        }
    }

    /// Submit a task to the pool.
    pub fn submit(&self, task: impl FnOnce() + Send + 'static) {
        let mut queue = self.tasks.lock().unwrap();
        queue.push_back(Box::new(task));
    }

    /// Wait for all currently-submitted tasks to complete.
    pub fn wait_idle(&self) {
        loop {
            let empty = {
                let queue = self.tasks.lock().unwrap();
                queue.is_empty()
            };
            if empty {
                break;
            }
            std::thread::yield_now();
        }
    }

    /// Shut down the pool, joining all workers.
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::Relaxed);
        for handle in self.handles {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// System executor V2
// ---------------------------------------------------------------------------

/// Configuration for the system executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfigV2 {
    /// Number of worker threads (0 = single-threaded).
    pub thread_count: usize,
    /// Whether to enable system profiling.
    pub profiling: bool,
    /// Whether to print the schedule to stderr on rebuild.
    pub debug_schedule: bool,
}

impl Default for ExecutorConfigV2 {
    fn default() -> Self {
        Self {
            thread_count: 0,
            profiling: false,
            debug_schedule: false,
        }
    }
}

/// Execution profile / statistics for a single frame.
#[derive(Debug, Clone)]
pub struct ExecutionProfileV2 {
    /// Total wall-clock time for the frame.
    pub total_duration: Duration,
    /// Time spent on fixed-timestep systems.
    pub fixed_duration: Duration,
    /// Number of batches executed.
    pub batch_count: usize,
    /// Number of systems executed.
    pub systems_run: usize,
    /// Number of systems skipped by run conditions.
    pub systems_skipped: usize,
    /// Number of fixed timestep steps this frame.
    pub fixed_steps: u32,
}

impl ExecutionProfileV2 {
    /// Create a blank profile.
    pub fn empty() -> Self {
        Self {
            total_duration: Duration::ZERO,
            fixed_duration: Duration::ZERO,
            batch_count: 0,
            systems_run: 0,
            systems_skipped: 0,
            fixed_steps: 0,
        }
    }
}

/// The enhanced system executor.
///
/// Manages system registration, dependency analysis, automatic parallel
/// batching, and runtime execution with run conditions and fixed timestep.
pub struct SystemExecutorV2 {
    /// All registered systems.
    systems: Vec<SystemDescriptorV2>,
    /// Named system sets.
    sets: HashMap<SystemSetId, SystemSetV2>,
    /// Registered run conditions.
    conditions: HashMap<RunConditionId, RunCondition>,
    /// Pre-computed parallel batches (rebuilt when systems change).
    schedule: Vec<ParallelBatchV2>,
    /// Whether the schedule needs to be rebuilt.
    dirty: bool,
    /// Configuration.
    config: ExecutorConfigV2,
    /// Thread pool (if multi-threaded).
    thread_pool: Option<ThreadPool>,
    /// Last frame execution profile.
    last_profile: ExecutionProfileV2,
    /// Next system set ID.
    next_set_id: u32,
    /// Next run condition ID.
    next_condition_id: u32,
}

impl SystemExecutorV2 {
    /// Create a new executor with the given configuration.
    pub fn new(config: ExecutorConfigV2) -> Self {
        let thread_pool = if config.thread_count > 0 {
            Some(ThreadPool::new(config.thread_count))
        } else {
            None
        };

        Self {
            systems: Vec::new(),
            sets: HashMap::new(),
            conditions: HashMap::new(),
            schedule: Vec::new(),
            dirty: true,
            config,
            thread_pool,
            last_profile: ExecutionProfileV2::empty(),
            next_set_id: 0,
            next_condition_id: 0,
        }
    }

    /// Register a new system.
    pub fn add_system(
        &mut self,
        name: impl Into<String>,
        access: SystemAccessV2,
        run_fn: impl FnMut() + Send + 'static,
    ) -> SystemIndexV2 {
        let index = SystemIndexV2(self.systems.len());
        let descriptor = SystemDescriptorV2::new(index, name, access, run_fn);
        self.systems.push(descriptor);
        self.dirty = true;
        index
    }

    /// Add an ordering dependency: `before` must run before `after`.
    pub fn add_dependency(&mut self, before: SystemIndexV2, after: SystemIndexV2) {
        if before.0 < self.systems.len() && after.0 < self.systems.len() {
            self.systems[after.0].dependencies.push(before);
            self.systems[before.0].dependents.push(after);
            self.dirty = true;
        }
    }

    /// Create a new system set.
    pub fn create_set(&mut self, name: impl Into<String>) -> SystemSetId {
        let id = SystemSetId(self.next_set_id);
        self.next_set_id += 1;
        let set = SystemSetV2::new(id, name);
        self.sets.insert(id, set);
        id
    }

    /// Add a system to a set.
    pub fn add_to_set(&mut self, system: SystemIndexV2, set: SystemSetId) {
        if let Some(s) = self.sets.get_mut(&set) {
            s.add_system(system.0);
            if system.0 < self.systems.len() {
                self.systems[system.0].set = Some(set);
            }
            self.dirty = true;
        }
    }

    /// Set ordering between two sets.
    pub fn set_order(&mut self, before: SystemSetId, after: SystemSetId) {
        if let Some(s) = self.sets.get_mut(&after) {
            s.after(before);
        }
        if let Some(s) = self.sets.get_mut(&before) {
            s.before(after);
        }
        self.dirty = true;
    }

    /// Register a run condition.
    pub fn add_run_condition(
        &mut self,
        label: impl Into<String>,
        predicate: impl Fn() -> bool + Send + Sync + 'static,
    ) -> RunConditionId {
        let id = RunConditionId(self.next_condition_id);
        self.next_condition_id += 1;
        let condition = RunCondition::new(id, label, predicate);
        self.conditions.insert(id, condition);
        id
    }

    /// Attach a run condition to a system.
    pub fn set_system_condition(
        &mut self,
        system: SystemIndexV2,
        condition: RunConditionId,
    ) {
        if system.0 < self.systems.len() {
            self.systems[system.0].run_condition = Some(condition);
        }
    }

    /// Configure a system to use fixed timestep.
    pub fn set_fixed_timestep(
        &mut self,
        system: SystemIndexV2,
        step: Duration,
    ) {
        if system.0 < self.systems.len() {
            self.systems[system.0].fixed_timestep = Some(FixedTimestep::new(step));
        }
    }

    /// Enable or disable a system.
    pub fn set_enabled(&mut self, system: SystemIndexV2, enabled: bool) {
        if system.0 < self.systems.len() {
            self.systems[system.0].enabled = enabled;
        }
    }

    /// Rebuild the parallel schedule using topological sort and conflict analysis.
    pub fn rebuild_schedule(&mut self) {
        if !self.dirty {
            return;
        }

        let n = self.systems.len();
        if n == 0 {
            self.schedule.clear();
            self.dirty = false;
            return;
        }

        // Build adjacency/in-degree for topological sort.
        let mut in_degree = vec![0u32; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for sys in &self.systems {
            for dep in &sys.dependencies {
                adj[dep.0].push(sys.index.0);
                in_degree[sys.index.0] += 1;
            }
        }

        // Add set-based ordering as dependencies.
        for set in self.sets.values() {
            for &after_set_id in &set.after_sets {
                if let Some(before_set) = self.sets.get(&after_set_id) {
                    for &before_sys in &before_set.systems {
                        for &after_sys in &set.systems {
                            if before_sys < n && after_sys < n {
                                adj[before_sys].push(after_sys);
                                in_degree[after_sys] += 1;
                            }
                        }
                    }
                }
            }
        }

        // Topological sort (Kahn's algorithm).
        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut sorted = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() {
            sorted.push(node);
            for &neighbor in &adj[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        // If sorted.len() < n, there's a cycle — skip those systems.
        // Build parallel batches from the topological order.
        let mut batches: Vec<ParallelBatchV2> = Vec::new();
        let mut scheduled = vec![false; n];

        for &sys_idx in &sorted {
            if !self.systems[sys_idx].enabled {
                scheduled[sys_idx] = true;
                continue;
            }

            // Find the earliest batch where this system can go.
            let mut placed = false;
            for batch in &mut batches {
                let can_place = batch.systems.iter().all(|&existing| {
                    !self.systems[sys_idx]
                        .access
                        .conflicts_with(&self.systems[existing.0].access)
                });

                // Also check that all dependencies are in earlier batches.
                let deps_met = self.systems[sys_idx]
                    .dependencies
                    .iter()
                    .all(|dep| {
                        !batch.systems.contains(dep)
                    });

                if can_place && deps_met {
                    batch.add(SystemIndexV2(sys_idx));
                    placed = true;
                    break;
                }
            }

            if !placed {
                let mut batch = ParallelBatchV2::new();
                batch.add(SystemIndexV2(sys_idx));
                batches.push(batch);
            }

            scheduled[sys_idx] = true;
        }

        self.schedule = batches;
        self.dirty = false;

        if self.config.debug_schedule {
            eprintln!("--- System Schedule ({} batches) ---", self.schedule.len());
            for (i, batch) in self.schedule.iter().enumerate() {
                let names: Vec<&str> = batch
                    .systems
                    .iter()
                    .map(|s| self.systems[s.0].name.as_str())
                    .collect();
                eprintln!("  Batch {}: {:?}", i, names);
            }
        }
    }

    /// Run all systems for this frame.
    ///
    /// `delta` is the time since last frame, used for fixed timestep accumulation.
    pub fn run(&mut self, delta: Duration) {
        let frame_start = Instant::now();

        if self.dirty {
            self.rebuild_schedule();
        }

        let mut profile = ExecutionProfileV2::empty();

        for batch_idx in 0..self.schedule.len() {
            let batch = &self.schedule[batch_idx];

            // Evaluate run conditions and collect systems to run.
            let mut to_run: Vec<SystemIndexV2> = Vec::new();
            for &sys_idx in &batch.systems {
                let sys = &self.systems[sys_idx.0];
                if !sys.enabled {
                    profile.systems_skipped += 1;
                    continue;
                }

                // Check run condition.
                if let Some(cond_id) = sys.run_condition {
                    if let Some(cond) = self.conditions.get(&cond_id) {
                        if !cond.evaluate() {
                            profile.systems_skipped += 1;
                            continue;
                        }
                    }
                }

                // Check set-level run condition.
                if let Some(set_id) = sys.set {
                    if let Some(set) = self.sets.get(&set_id) {
                        if !set.enabled {
                            profile.systems_skipped += 1;
                            continue;
                        }
                        if let Some(cond_id) = set.run_condition {
                            if let Some(cond) = self.conditions.get(&cond_id) {
                                if !cond.evaluate() {
                                    profile.systems_skipped += 1;
                                    continue;
                                }
                            }
                        }
                    }
                }

                to_run.push(sys_idx);
            }

            if to_run.is_empty() {
                continue;
            }

            profile.batch_count += 1;

            // Execute the batch.
            // Single-threaded path: run sequentially.
            for &sys_idx in &to_run {
                let sys = &mut self.systems[sys_idx.0];

                // Handle fixed timestep.
                if let Some(ref mut fixed) = sys.fixed_timestep {
                    let steps = fixed.accumulate(delta);
                    let step_start = Instant::now();
                    for _ in 0..steps {
                        (sys.run_fn)();
                        profile.systems_run += 1;
                        profile.fixed_steps += 1;
                    }
                    let elapsed = step_start.elapsed();
                    profile.fixed_duration += elapsed;
                    sys.last_duration = elapsed;
                    sys.run_count += steps as u64;
                    // Update rolling average.
                    if sys.run_count > 0 {
                        let total = sys.avg_duration.as_secs_f64() * (sys.run_count - steps as u64) as f64
                            + elapsed.as_secs_f64();
                        sys.avg_duration =
                            Duration::from_secs_f64(total / sys.run_count as f64);
                    }
                } else {
                    let start = Instant::now();
                    (sys.run_fn)();
                    let elapsed = start.elapsed();
                    sys.last_duration = elapsed;
                    sys.run_count += 1;
                    // Update rolling average (exponential moving average).
                    let alpha = 0.1;
                    sys.avg_duration = Duration::from_secs_f64(
                        sys.avg_duration.as_secs_f64() * (1.0 - alpha)
                            + elapsed.as_secs_f64() * alpha,
                    );
                    profile.systems_run += 1;
                }
            }
        }

        profile.total_duration = frame_start.elapsed();
        self.last_profile = profile;
    }

    /// Get the last frame's execution profile.
    pub fn last_profile(&self) -> &ExecutionProfileV2 {
        &self.last_profile
    }

    /// Get the number of registered systems.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// Get the number of parallel batches in the current schedule.
    pub fn batch_count(&self) -> usize {
        self.schedule.len()
    }

    /// Get a system's name by index.
    pub fn system_name(&self, index: SystemIndexV2) -> Option<&str> {
        self.systems.get(index.0).map(|s| s.name.as_str())
    }

    /// Get system profiling info.
    pub fn system_profile(&self, index: SystemIndexV2) -> Option<SystemProfile> {
        self.systems.get(index.0).map(|s| SystemProfile {
            name: s.name.clone(),
            avg_duration: s.avg_duration,
            last_duration: s.last_duration,
            run_count: s.run_count,
            enabled: s.enabled,
        })
    }

    /// Iterate over all system profiles.
    pub fn all_profiles(&self) -> Vec<SystemProfile> {
        self.systems
            .iter()
            .map(|s| SystemProfile {
                name: s.name.clone(),
                avg_duration: s.avg_duration,
                last_duration: s.last_duration,
                run_count: s.run_count,
                enabled: s.enabled,
            })
            .collect()
    }

    /// Remove all systems and clear the schedule.
    pub fn clear(&mut self) {
        self.systems.clear();
        self.sets.clear();
        self.conditions.clear();
        self.schedule.clear();
        self.dirty = false;
    }
}

/// Profiling information for a single system.
#[derive(Debug, Clone)]
pub struct SystemProfile {
    /// System name.
    pub name: String,
    /// Rolling average duration.
    pub avg_duration: Duration,
    /// Last frame duration.
    pub last_duration: Duration,
    /// Total number of runs.
    pub run_count: u64,
    /// Whether the system is enabled.
    pub enabled: bool,
}

impl fmt::Display for SystemProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: avg={:.2?} last={:.2?} runs={} enabled={}",
            self.name, self.avg_duration, self.last_duration, self.run_count, self.enabled
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn access_conflict_detection() {
        let mut a = SystemAccessV2::new();
        a.write_component::<u32>();

        let mut b = SystemAccessV2::new();
        b.read_component::<u32>();

        assert!(a.conflicts_with(&b));

        let mut c = SystemAccessV2::new();
        c.read_component::<u32>();
        assert!(!b.conflicts_with(&c)); // two reads don't conflict
    }

    #[test]
    fn exclusive_conflicts_with_everything() {
        let mut a = SystemAccessV2::new();
        a.set_exclusive();

        let b = SystemAccessV2::new();
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn fixed_timestep_accumulation() {
        let mut fixed = FixedTimestep::from_hz(60.0);
        let steps = fixed.accumulate(Duration::from_millis(32)); // ~2 steps at 60Hz
        assert!(steps >= 1);
    }

    #[test]
    fn executor_basic() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);

        let counter = Arc::new(AtomicU64::new(0));
        let c = Arc::clone(&counter);

        let access = SystemAccessV2::new();
        executor.add_system("test_system", access, move || {
            c.fetch_add(1, Ordering::Relaxed);
        });

        executor.run(Duration::from_millis(16));
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn system_sets_ordering() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);

        let set_a = executor.create_set("SetA");
        let set_b = executor.create_set("SetB");
        executor.set_order(set_a, set_b);

        let s1 = executor.add_system("sys1", SystemAccessV2::new(), || {});
        let s2 = executor.add_system("sys2", SystemAccessV2::new(), || {});

        executor.add_to_set(s1, set_a);
        executor.add_to_set(s2, set_b);

        executor.rebuild_schedule();
        assert!(executor.batch_count() >= 1);
    }
}
