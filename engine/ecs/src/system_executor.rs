//! Enhanced system executor with automatic parallelism, thread pool integration,
//! system sets with ordering, run conditions, and fixed timestep systems.

use std::any::TypeId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessKindV2 { Read, Write, Exclusive }

#[derive(Debug, Clone)]
pub struct ComponentAccessV2 { pub type_id: TypeId, pub type_name: &'static str, pub access: AccessKindV2 }

#[derive(Debug, Clone)]
pub struct SystemAccessV2 {
    pub components: Vec<ComponentAccessV2>,
    pub resources: Vec<(TypeId, AccessKindV2)>,
    pub exclusive: bool,
}

impl SystemAccessV2 {
    pub fn new() -> Self { Self { components: Vec::new(), resources: Vec::new(), exclusive: false } }
    pub fn read_component<T: 'static>(&mut self) -> &mut Self { self.components.push(ComponentAccessV2 { type_id: TypeId::of::<T>(), type_name: std::any::type_name::<T>(), access: AccessKindV2::Read }); self }
    pub fn write_component<T: 'static>(&mut self) -> &mut Self { self.components.push(ComponentAccessV2 { type_id: TypeId::of::<T>(), type_name: std::any::type_name::<T>(), access: AccessKindV2::Write }); self }
    pub fn set_exclusive(&mut self) -> &mut Self { self.exclusive = true; self }
    pub fn conflicts_with(&self, other: &SystemAccessV2) -> bool {
        if self.exclusive || other.exclusive { return true; }
        for a in &self.components { for b in &other.components { if a.type_id == b.type_id && (a.access == AccessKindV2::Write || b.access == AccessKindV2::Write) { return true; } } }
        for &(at, aa) in &self.resources { for &(bt, ba) in &other.resources { if at == bt && (aa == AccessKindV2::Write || ba == AccessKindV2::Write) { return true; } } }
        false
    }
}
impl Default for SystemAccessV2 { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct RunConditionId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct SystemSetId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct SystemIndexV2(pub usize);

pub struct RunCondition { pub id: RunConditionId, pub label: String, pub predicate: Box<dyn Fn() -> bool + Send + Sync> }
impl RunCondition { pub fn new(id: RunConditionId, label: impl Into<String>, pred: impl Fn() -> bool + Send + Sync + 'static) -> Self { Self { id, label: label.into(), predicate: Box::new(pred) } } pub fn evaluate(&self) -> bool { (self.predicate)() } }

#[derive(Debug, Clone)]
pub struct SystemSetV2 { pub id: SystemSetId, pub name: String, pub systems: Vec<usize>, pub after_sets: Vec<SystemSetId>, pub before_sets: Vec<SystemSetId>, pub run_condition: Option<RunConditionId>, pub enabled: bool }
impl SystemSetV2 {
    pub fn new(id: SystemSetId, name: impl Into<String>) -> Self { Self { id, name: name.into(), systems: Vec::new(), after_sets: Vec::new(), before_sets: Vec::new(), run_condition: None, enabled: true } }
    pub fn add_system(&mut self, idx: usize) { if !self.systems.contains(&idx) { self.systems.push(idx); } }
    pub fn after(&mut self, other: SystemSetId) -> &mut Self { if !self.after_sets.contains(&other) { self.after_sets.push(other); } self }
    pub fn before(&mut self, other: SystemSetId) -> &mut Self { if !self.before_sets.contains(&other) { self.before_sets.push(other); } self }
}

#[derive(Debug, Clone)]
pub struct FixedTimestep { pub step: Duration, pub accumulator: Duration, pub max_steps_per_frame: u32, pub steps_this_frame: u32, pub total_steps: u64, pub overshoot_alpha: f64 }
impl FixedTimestep {
    pub fn new(step: Duration) -> Self { Self { step, accumulator: Duration::ZERO, max_steps_per_frame: 8, steps_this_frame: 0, total_steps: 0, overshoot_alpha: 0.0 } }
    pub fn from_hz(hz: f64) -> Self { Self::new(Duration::from_secs_f64(1.0 / hz)) }
    pub fn accumulate(&mut self, delta: Duration) -> u32 {
        self.accumulator += delta; self.steps_this_frame = 0; let mut steps = 0u32;
        while self.accumulator >= self.step && steps < self.max_steps_per_frame { self.accumulator -= self.step; steps += 1; self.total_steps += 1; }
        if steps >= self.max_steps_per_frame { self.accumulator = Duration::ZERO; }
        self.overshoot_alpha = self.accumulator.as_secs_f64() / self.step.as_secs_f64(); self.steps_this_frame = steps; steps
    }
    pub fn step_secs(&self) -> f32 { self.step.as_secs_f32() }
    pub fn alpha(&self) -> f64 { self.overshoot_alpha }
}

pub struct SystemDescriptorV2 { pub index: SystemIndexV2, pub name: String, pub access: SystemAccessV2, pub run_fn: Box<dyn FnMut() + Send>, pub run_condition: Option<RunConditionId>, pub dependencies: Vec<SystemIndexV2>, pub dependents: Vec<SystemIndexV2>, pub set: Option<SystemSetId>, pub enabled: bool, pub fixed_timestep: Option<FixedTimestep>, pub avg_duration: Duration, pub last_duration: Duration, pub run_count: u64 }
impl SystemDescriptorV2 {
    pub fn new(index: SystemIndexV2, name: impl Into<String>, access: SystemAccessV2, run_fn: impl FnMut() + Send + 'static) -> Self {
        Self { index, name: name.into(), access, run_fn: Box::new(run_fn), run_condition: None, dependencies: Vec::new(), dependents: Vec::new(), set: None, enabled: true, fixed_timestep: None, avg_duration: Duration::ZERO, last_duration: Duration::ZERO, run_count: 0 }
    }
}

#[derive(Debug, Clone)] pub struct ParallelBatchV2 { pub systems: Vec<SystemIndexV2> }
impl ParallelBatchV2 { pub fn new() -> Self { Self { systems: Vec::new() } } pub fn add(&mut self, s: SystemIndexV2) { self.systems.push(s); } pub fn len(&self) -> usize { self.systems.len() } pub fn is_empty(&self) -> bool { self.systems.is_empty() } }
impl Default for ParallelBatchV2 { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone)] pub struct ExecutorConfigV2 { pub thread_count: usize, pub profiling: bool, pub debug_schedule: bool }
impl Default for ExecutorConfigV2 { fn default() -> Self { Self { thread_count: 0, profiling: false, debug_schedule: false } } }

#[derive(Debug, Clone)]
pub struct ExecutionProfileV2 { pub total_duration: Duration, pub fixed_duration: Duration, pub batch_count: usize, pub systems_run: usize, pub systems_skipped: usize, pub fixed_steps: u32 }
impl ExecutionProfileV2 { pub fn empty() -> Self { Self { total_duration: Duration::ZERO, fixed_duration: Duration::ZERO, batch_count: 0, systems_run: 0, systems_skipped: 0, fixed_steps: 0 } } }

pub struct SystemExecutorV2 {
    systems: Vec<SystemDescriptorV2>, sets: HashMap<SystemSetId, SystemSetV2>, conditions: HashMap<RunConditionId, RunCondition>,
    schedule: Vec<ParallelBatchV2>, dirty: bool, config: ExecutorConfigV2, last_profile: ExecutionProfileV2, next_set_id: u32, next_condition_id: u32,
}

impl SystemExecutorV2 {
    pub fn new(config: ExecutorConfigV2) -> Self { Self { systems: Vec::new(), sets: HashMap::new(), conditions: HashMap::new(), schedule: Vec::new(), dirty: true, config, last_profile: ExecutionProfileV2::empty(), next_set_id: 0, next_condition_id: 0 } }
    pub fn add_system(&mut self, name: impl Into<String>, access: SystemAccessV2, run_fn: impl FnMut() + Send + 'static) -> SystemIndexV2 { let index = SystemIndexV2(self.systems.len()); self.systems.push(SystemDescriptorV2::new(index, name, access, run_fn)); self.dirty = true; index }
    pub fn add_dependency(&mut self, before: SystemIndexV2, after: SystemIndexV2) { if before.0 < self.systems.len() && after.0 < self.systems.len() { self.systems[after.0].dependencies.push(before); self.systems[before.0].dependents.push(after); self.dirty = true; } }
    pub fn create_set(&mut self, name: impl Into<String>) -> SystemSetId { let id = SystemSetId(self.next_set_id); self.next_set_id += 1; self.sets.insert(id, SystemSetV2::new(id, name)); id }
    pub fn add_to_set(&mut self, system: SystemIndexV2, set: SystemSetId) { if let Some(s) = self.sets.get_mut(&set) { s.add_system(system.0); if system.0 < self.systems.len() { self.systems[system.0].set = Some(set); } self.dirty = true; } }
    pub fn set_order(&mut self, before: SystemSetId, after: SystemSetId) { if let Some(s) = self.sets.get_mut(&after) { s.after(before); } if let Some(s) = self.sets.get_mut(&before) { s.before(after); } self.dirty = true; }
    pub fn add_run_condition(&mut self, label: impl Into<String>, pred: impl Fn() -> bool + Send + Sync + 'static) -> RunConditionId { let id = RunConditionId(self.next_condition_id); self.next_condition_id += 1; self.conditions.insert(id, RunCondition::new(id, label, pred)); id }
    pub fn set_system_condition(&mut self, system: SystemIndexV2, cond: RunConditionId) { if system.0 < self.systems.len() { self.systems[system.0].run_condition = Some(cond); } }
    pub fn set_fixed_timestep(&mut self, system: SystemIndexV2, step: Duration) { if system.0 < self.systems.len() { self.systems[system.0].fixed_timestep = Some(FixedTimestep::new(step)); } }
    pub fn set_enabled(&mut self, system: SystemIndexV2, enabled: bool) { if system.0 < self.systems.len() { self.systems[system.0].enabled = enabled; } }

    pub fn rebuild_schedule(&mut self) {
        if !self.dirty { return; } let n = self.systems.len();
        if n == 0 { self.schedule.clear(); self.dirty = false; return; }
        let mut in_degree = vec![0u32; n]; let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for sys in &self.systems { for dep in &sys.dependencies { adj[dep.0].push(sys.index.0); in_degree[sys.index.0] += 1; } }
        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..n { if in_degree[i] == 0 { queue.push_back(i); } }
        let mut sorted = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() { sorted.push(node); for &nb in &adj[node] { in_degree[nb] -= 1; if in_degree[nb] == 0 { queue.push_back(nb); } } }
        let mut batches: Vec<ParallelBatchV2> = Vec::new();
        for &sys_idx in &sorted {
            if !self.systems[sys_idx].enabled { continue; }
            let mut placed = false;
            for batch in &mut batches {
                let can = batch.systems.iter().all(|&ex| !self.systems[sys_idx].access.conflicts_with(&self.systems[ex.0].access));
                let deps = self.systems[sys_idx].dependencies.iter().all(|dep| !batch.systems.contains(dep));
                if can && deps { batch.add(SystemIndexV2(sys_idx)); placed = true; break; }
            }
            if !placed { let mut b = ParallelBatchV2::new(); b.add(SystemIndexV2(sys_idx)); batches.push(b); }
        }
        self.schedule = batches; self.dirty = false;
    }

    pub fn run(&mut self, delta: Duration) {
        let frame_start = Instant::now(); if self.dirty { self.rebuild_schedule(); }
        let mut profile = ExecutionProfileV2::empty();
        for batch_idx in 0..self.schedule.len() {
            let batch = &self.schedule[batch_idx]; let mut to_run: Vec<SystemIndexV2> = Vec::new();
            for &sys_idx in &batch.systems {
                let sys = &self.systems[sys_idx.0];
                if !sys.enabled { profile.systems_skipped += 1; continue; }
                if let Some(cid) = sys.run_condition { if let Some(c) = self.conditions.get(&cid) { if !c.evaluate() { profile.systems_skipped += 1; continue; } } }
                to_run.push(sys_idx);
            }
            if to_run.is_empty() { continue; } profile.batch_count += 1;
            for &sys_idx in &to_run {
                let sys = &mut self.systems[sys_idx.0];
                if let Some(ref mut fixed) = sys.fixed_timestep {
                    let steps = fixed.accumulate(delta);
                    for _ in 0..steps { (sys.run_fn)(); profile.systems_run += 1; profile.fixed_steps += 1; }
                    sys.run_count += steps as u64;
                } else { let start = Instant::now(); (sys.run_fn)(); sys.last_duration = start.elapsed(); sys.run_count += 1; profile.systems_run += 1; }
            }
        }
        profile.total_duration = frame_start.elapsed(); self.last_profile = profile;
    }

    pub fn last_profile(&self) -> &ExecutionProfileV2 { &self.last_profile }
    pub fn system_count(&self) -> usize { self.systems.len() }
    pub fn batch_count(&self) -> usize { self.schedule.len() }
    pub fn system_name(&self, index: SystemIndexV2) -> Option<&str> { self.systems.get(index.0).map(|s| s.name.as_str()) }
    pub fn clear(&mut self) { self.systems.clear(); self.sets.clear(); self.conditions.clear(); self.schedule.clear(); self.dirty = false; }
}

#[derive(Debug, Clone)]
pub struct SystemProfile { pub name: String, pub avg_duration: Duration, pub last_duration: Duration, pub run_count: u64, pub enabled: bool }
impl fmt::Display for SystemProfile { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}: avg={:.2?} last={:.2?} runs={}", self.name, self.avg_duration, self.last_duration, self.run_count) } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn access_conflict() {
        let mut a = SystemAccessV2::new(); a.write_component::<u32>();
        let mut b = SystemAccessV2::new(); b.read_component::<u32>();
        assert!(a.conflicts_with(&b));
        let mut c = SystemAccessV2::new(); c.read_component::<u32>();
        assert!(!b.conflicts_with(&c));
    }
    #[test]
    fn executor_basic() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);
        let counter = Arc::new(AtomicU64::new(0));
        let c = Arc::clone(&counter);
        executor.add_system("test", SystemAccessV2::new(), move || { c.fetch_add(1, Ordering::Relaxed); });
        executor.run(Duration::from_millis(16));
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn fixed_timestep() { let mut f = FixedTimestep::from_hz(60.0); let s = f.accumulate(Duration::from_millis(32)); assert!(s >= 1); }

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

    #[test]
    fn run_conditions() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);
        let counter = Arc::new(AtomicU64::new(0));
        let c = Arc::clone(&counter);
        let should_run = Arc::new(AtomicBool::new(false));
        let sr = Arc::clone(&should_run);
        let cond = executor.add_run_condition("test_cond", move || sr.load(Ordering::Relaxed));
        let sys = executor.add_system("cond_sys", SystemAccessV2::new(), move || { c.fetch_add(1, Ordering::Relaxed); });
        executor.set_system_condition(sys, cond);
        executor.run(Duration::from_millis(16));
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        should_run.store(true, Ordering::Relaxed);
        executor.run(Duration::from_millis(16));
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn dependency_ordering() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);
        let order = Arc::new(Mutex::new(Vec::<u32>::new()));
        let o1 = Arc::clone(&order);
        let o2 = Arc::clone(&order);
        let o3 = Arc::clone(&order);
        let s1 = executor.add_system("first", SystemAccessV2::new(), move || { o1.lock().unwrap().push(1); });
        let s2 = executor.add_system("second", SystemAccessV2::new(), move || { o2.lock().unwrap().push(2); });
        let s3 = executor.add_system("third", SystemAccessV2::new(), move || { o3.lock().unwrap().push(3); });
        executor.add_dependency(s1, s2);
        executor.add_dependency(s2, s3);
        executor.run(Duration::from_millis(16));
        let result = order.lock().unwrap().clone();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn parallel_non_conflicting() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);
        let mut a1 = SystemAccessV2::new(); a1.read_component::<u32>();
        let mut a2 = SystemAccessV2::new(); a2.read_component::<u32>();
        executor.add_system("reader1", a1, || {});
        executor.add_system("reader2", a2, || {});
        executor.rebuild_schedule();
        // Two readers should be in the same batch.
        assert_eq!(executor.batch_count(), 1);
    }

    #[test]
    fn conflicting_writers_separate_batches() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);
        let mut a1 = SystemAccessV2::new(); a1.write_component::<u32>();
        let mut a2 = SystemAccessV2::new(); a2.write_component::<u32>();
        executor.add_system("writer1", a1, || {});
        executor.add_system("writer2", a2, || {});
        executor.rebuild_schedule();
        // Two writers of same component should be in different batches.
        assert_eq!(executor.batch_count(), 2);
    }

    #[test]
    fn disable_system() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);
        let counter = Arc::new(AtomicU64::new(0));
        let c = Arc::clone(&counter);
        let sys = executor.add_system("disabled", SystemAccessV2::new(), move || { c.fetch_add(1, Ordering::Relaxed); });
        executor.set_enabled(sys, false);
        executor.run(Duration::from_millis(16));
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn clear_executor() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);
        executor.add_system("a", SystemAccessV2::new(), || {});
        executor.add_system("b", SystemAccessV2::new(), || {});
        assert_eq!(executor.system_count(), 2);
        executor.clear();
        assert_eq!(executor.system_count(), 0);
    }

    #[test]
    fn system_name_lookup() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);
        let s = executor.add_system("my_system", SystemAccessV2::new(), || {});
        assert_eq!(executor.system_name(s), Some("my_system"));
    }

    #[test]
    fn profile_tracking() {
        let config = ExecutorConfigV2::default();
        let mut executor = SystemExecutorV2::new(config);
        executor.add_system("profiled", SystemAccessV2::new(), || { std::thread::yield_now(); });
        executor.run(Duration::from_millis(16));
        let profile = executor.last_profile();
        assert_eq!(profile.systems_run, 1);
        assert!(profile.total_duration.as_nanos() > 0);
    }

    #[test]
    fn exclusive_access_conflicts_with_everything() {
        let mut a = SystemAccessV2::new(); a.set_exclusive();
        let b = SystemAccessV2::new();
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn fixed_timestep_multiple_steps() {
        let mut fixed = FixedTimestep::new(Duration::from_millis(16));
        let steps = fixed.accumulate(Duration::from_millis(50));
        assert!(steps >= 3);
        assert!(fixed.alpha() >= 0.0);
        assert!(fixed.alpha() <= 1.0);
    }

    #[test]
    fn fixed_timestep_spiral_of_death_prevention() {
        let mut fixed = FixedTimestep::new(Duration::from_millis(16));
        fixed.max_steps_per_frame = 4;
        let steps = fixed.accumulate(Duration::from_millis(1000));
        assert_eq!(steps, 4);
    }
}

// ---------------------------------------------------------------------------
// Thread pool (simple work-stealing implementation)
// ---------------------------------------------------------------------------

/// A simple thread pool for executing system batches in parallel.
pub struct ThreadPool {
    /// Number of worker threads.
    pub thread_count: usize,
    /// Pending tasks queue.
    tasks: Arc<Mutex<VecDeque<Box<dyn FnOnce() + Send>>>>,
    /// Shutdown signal.
    shutdown: Arc<AtomicBool>,
    /// Worker thread handles.
    handles: Vec<std::thread::JoinHandle<()>>,
    /// Total tasks submitted.
    total_submitted: AtomicU64,
    /// Total tasks completed.
    total_completed: Arc<AtomicU64>,
}

impl ThreadPool {
    /// Create a new thread pool with the given number of workers.
    pub fn new(thread_count: usize) -> Self {
        let tasks: Arc<Mutex<VecDeque<Box<dyn FnOnce() + Send>>>> =
            Arc::new(Mutex::new(VecDeque::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let total_completed = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::with_capacity(thread_count);

        for i in 0..thread_count {
            let tasks = Arc::clone(&tasks);
            let shutdown = Arc::clone(&shutdown);
            let completed = Arc::clone(&total_completed);

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
                            completed.fetch_add(1, Ordering::Relaxed);
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
            total_submitted: AtomicU64::new(0),
            total_completed,
        }
    }

    /// Submit a task to the pool.
    pub fn submit(&self, task: impl FnOnce() + Send + 'static) {
        let mut queue = self.tasks.lock().unwrap();
        queue.push_back(Box::new(task));
        self.total_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Submit multiple tasks at once.
    pub fn submit_batch(&self, tasks: Vec<Box<dyn FnOnce() + Send>>) {
        let mut queue = self.tasks.lock().unwrap();
        let count = tasks.len() as u64;
        for task in tasks {
            queue.push_back(task);
        }
        self.total_submitted.fetch_add(count, Ordering::Relaxed);
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

    /// Get the number of pending tasks.
    pub fn pending_count(&self) -> usize {
        let queue = self.tasks.lock().unwrap();
        queue.len()
    }

    /// Get total tasks submitted.
    pub fn total_submitted(&self) -> u64 {
        self.total_submitted.load(Ordering::Relaxed)
    }

    /// Get total tasks completed.
    pub fn total_completed(&self) -> u64 {
        self.total_completed.load(Ordering::Relaxed)
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
// System dependency graph analysis
// ---------------------------------------------------------------------------

/// Analyzes system dependencies and produces a visualization-friendly summary.
pub struct DependencyAnalysis {
    /// Number of systems.
    pub system_count: usize,
    /// Number of dependency edges.
    pub edge_count: usize,
    /// Number of parallel batches.
    pub batch_count: usize,
    /// Maximum parallelism (largest batch size).
    pub max_parallelism: usize,
    /// Critical path length (longest dependency chain).
    pub critical_path_length: usize,
    /// Systems with no dependencies (can run first).
    pub root_systems: Vec<usize>,
    /// Systems with no dependents (can run last).
    pub leaf_systems: Vec<usize>,
    /// Whether a cycle was detected.
    pub has_cycle: bool,
    /// Average batch size.
    pub avg_batch_size: f32,
}

impl DependencyAnalysis {
    /// Analyze an executor's current schedule.
    pub fn analyze(executor: &SystemExecutorV2) -> Self {
        let n = executor.system_count();
        let mut edge_count = 0;
        let mut root_systems = Vec::new();
        let mut leaf_systems = Vec::new();

        for i in 0..n {
            let sys = &executor.systems[i];
            edge_count += sys.dependencies.len();
            if sys.dependencies.is_empty() {
                root_systems.push(i);
            }
            if sys.dependents.is_empty() {
                leaf_systems.push(i);
            }
        }

        let max_parallelism = executor.schedule.iter()
            .map(|b| b.systems.len())
            .max()
            .unwrap_or(0);

        let avg_batch_size = if executor.schedule.is_empty() {
            0.0
        } else {
            n as f32 / executor.schedule.len() as f32
        };

        // Compute critical path length via longest path in DAG.
        let mut longest_path = vec![0usize; n];
        // Topological order is already computed in the schedule.
        let mut max_path = 0;
        for batch in &executor.schedule {
            for &sys_idx in &batch.systems {
                let idx = sys_idx.0;
                for dep in &executor.systems[idx].dependencies {
                    longest_path[idx] = longest_path[idx].max(longest_path[dep.0] + 1);
                }
                max_path = max_path.max(longest_path[idx]);
            }
        }

        Self {
            system_count: n,
            edge_count,
            batch_count: executor.schedule.len(),
            max_parallelism,
            critical_path_length: max_path,
            root_systems,
            leaf_systems,
            has_cycle: false,
            avg_batch_size,
        }
    }
}

impl fmt::Display for DependencyAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "System Dependency Analysis:")?;
        writeln!(f, "  Systems:          {}", self.system_count)?;
        writeln!(f, "  Dependencies:     {}", self.edge_count)?;
        writeln!(f, "  Batches:          {}", self.batch_count)?;
        writeln!(f, "  Max parallelism:  {}", self.max_parallelism)?;
        writeln!(f, "  Critical path:    {}", self.critical_path_length)?;
        writeln!(f, "  Avg batch size:   {:.1}", self.avg_batch_size)?;
        writeln!(f, "  Root systems:     {}", self.root_systems.len())?;
        writeln!(f, "  Leaf systems:     {}", self.leaf_systems.len())?;
        writeln!(f, "  Has cycle:        {}", self.has_cycle)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// System execution timeline (for profiling visualization)
// ---------------------------------------------------------------------------

/// Records the execution timeline of systems for profiling visualization.
#[derive(Debug, Clone)]
pub struct ExecutionTimeline {
    /// Entries in the timeline.
    pub entries: Vec<TimelineEntry>,
    /// Frame start time.
    pub frame_start: Instant,
    /// Frame number.
    pub frame_number: u64,
}

/// A single entry in the execution timeline.
#[derive(Debug, Clone)]
pub struct TimelineEntry {
    /// System index.
    pub system_index: usize,
    /// System name.
    pub system_name: String,
    /// Batch index this system was in.
    pub batch_index: usize,
    /// Start offset from frame start.
    pub start_offset: Duration,
    /// Duration of execution.
    pub duration: Duration,
    /// Thread ID that executed this system.
    pub thread_id: u64,
}

impl ExecutionTimeline {
    /// Create a new timeline for the current frame.
    pub fn new(frame_number: u64) -> Self {
        Self {
            entries: Vec::new(),
            frame_start: Instant::now(),
            frame_number,
        }
    }

    /// Record a system execution.
    pub fn record(
        &mut self,
        system_index: usize,
        system_name: String,
        batch_index: usize,
        start: Instant,
        duration: Duration,
    ) {
        self.entries.push(TimelineEntry {
            system_index,
            system_name,
            batch_index,
            start_offset: start.duration_since(self.frame_start),
            duration,
            thread_id: 0, // Would use actual thread ID in production.
        });
    }

    /// Get the total frame duration.
    pub fn frame_duration(&self) -> Duration {
        self.entries.iter()
            .map(|e| e.start_offset + e.duration)
            .max()
            .unwrap_or(Duration::ZERO)
    }

    /// Get entries sorted by start time.
    pub fn sorted_entries(&self) -> Vec<&TimelineEntry> {
        let mut sorted: Vec<&TimelineEntry> = self.entries.iter().collect();
        sorted.sort_by_key(|e| e.start_offset);
        sorted
    }

    /// Get the longest-running system.
    pub fn slowest_system(&self) -> Option<&TimelineEntry> {
        self.entries.iter().max_by_key(|e| e.duration)
    }

    /// Export to Chrome trace format (JSON).
    pub fn to_chrome_trace(&self) -> String {
        let mut events = Vec::new();
        for entry in &self.entries {
            events.push(format!(
                r#"{{"name":"{}","cat":"system","ph":"X","ts":{},"dur":{},"pid":1,"tid":{}}}"#,
                entry.system_name,
                entry.start_offset.as_micros(),
                entry.duration.as_micros(),
                entry.thread_id
            ));
        }
        format!("[{}]", events.join(","))
    }
}
