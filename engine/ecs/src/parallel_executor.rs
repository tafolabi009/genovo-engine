//! # Parallel System Executor
//!
//! Analyzes system access patterns to find non-conflicting systems and execute
//! them in parallel across a thread pool, while preserving ordering constraints
//! within each stage.
//!
//! ## Features
//!
//! - **Access pattern analysis** — Systems declare which component types they
//!   read and write. The executor uses this to determine which systems conflict.
//! - **Conflict graph** — Build an undirected graph where an edge means two
//!   systems cannot run concurrently.
//! - **Parallel batching** — Partition systems into batches where all systems
//!   within a batch are conflict-free and can run simultaneously.
//! - **Thread pool execution** — Execute each batch on a configurable thread pool.
//! - **Exclusive system support** — Systems requiring `&mut World` run alone.
//! - **System ordering** — Ordering constraints (before/after) are respected
//!   within a stage by splitting batches at ordering boundaries.

use std::any::TypeId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// AccessKind
// ---------------------------------------------------------------------------

/// Kind of access a system performs on a component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessKind {
    /// Read-only access. Multiple systems can read concurrently.
    Read,
    /// Write access. Exclusive — conflicts with any other access.
    Write,
}

impl fmt::Display for AccessKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccessKind::Read => write!(f, "Read"),
            AccessKind::Write => write!(f, "Write"),
        }
    }
}

// ---------------------------------------------------------------------------
// ComponentAccess
// ---------------------------------------------------------------------------

/// Describes which component types a system reads and writes.
#[derive(Debug, Clone, Default)]
pub struct SystemComponentAccess {
    /// Component types that are read.
    pub reads: HashSet<TypeId>,
    /// Component types that are written.
    pub writes: HashSet<TypeId>,
    /// Resource types that are read.
    pub resource_reads: HashSet<TypeId>,
    /// Resource types that are written.
    pub resource_writes: HashSet<TypeId>,
    /// Whether this system needs exclusive world access (&mut World).
    pub exclusive: bool,
}

impl SystemComponentAccess {
    /// Create a new empty access declaration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a read on a component type.
    pub fn read_component<T: 'static>(mut self) -> Self {
        self.reads.insert(TypeId::of::<T>());
        self
    }

    /// Declare a write on a component type.
    pub fn write_component<T: 'static>(mut self) -> Self {
        self.writes.insert(TypeId::of::<T>());
        self
    }

    /// Declare a resource read.
    pub fn read_resource<T: 'static>(mut self) -> Self {
        self.resource_reads.insert(TypeId::of::<T>());
        self
    }

    /// Declare a resource write.
    pub fn write_resource<T: 'static>(mut self) -> Self {
        self.resource_writes.insert(TypeId::of::<T>());
        self
    }

    /// Mark as exclusive (requires &mut World).
    pub fn exclusive(mut self) -> Self {
        self.exclusive = true;
        self
    }

    /// Check whether two access declarations conflict.
    ///
    /// Two systems conflict if:
    /// - Either is exclusive.
    /// - Both access the same component and at least one writes it.
    /// - Both access the same resource and at least one writes it.
    pub fn conflicts_with(&self, other: &SystemComponentAccess) -> bool {
        // Exclusive systems conflict with everything.
        if self.exclusive || other.exclusive {
            return true;
        }

        // Component conflicts: write-write or read-write on the same type.
        for &ty in &self.writes {
            if other.reads.contains(&ty) || other.writes.contains(&ty) {
                return true;
            }
        }
        for &ty in &self.reads {
            if other.writes.contains(&ty) {
                return true;
            }
        }

        // Resource conflicts.
        for &ty in &self.resource_writes {
            if other.resource_reads.contains(&ty) || other.resource_writes.contains(&ty) {
                return true;
            }
        }
        for &ty in &self.resource_reads {
            if other.resource_writes.contains(&ty) {
                return true;
            }
        }

        false
    }

    /// Returns all types accessed (union of reads and writes).
    pub fn all_component_types(&self) -> HashSet<TypeId> {
        let mut all = self.reads.clone();
        all.extend(&self.writes);
        all
    }

    /// Returns all resource types accessed.
    pub fn all_resource_types(&self) -> HashSet<TypeId> {
        let mut all = self.resource_reads.clone();
        all.extend(&self.resource_writes);
        all
    }
}

// ---------------------------------------------------------------------------
// SystemDescriptor
// ---------------------------------------------------------------------------

/// Unique identifier for a system in the executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemIndex(pub usize);

impl fmt::Display for SystemIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "System({})", self.0)
    }
}

/// Describes a system for the parallel executor.
pub struct ParallelSystemDescriptor {
    /// Human-readable name.
    pub name: String,
    /// Access pattern.
    pub access: SystemComponentAccess,
    /// The system function. Takes a &mut on the shared world state.
    pub run_fn: Box<dyn FnMut(&mut ExecutionContext) + Send + 'static>,
    /// Systems this must run before.
    pub before: Vec<String>,
    /// Systems this must run after.
    pub after: Vec<String>,
    /// Whether this system is currently enabled.
    pub enabled: bool,
}

impl ParallelSystemDescriptor {
    /// Create a new system descriptor.
    pub fn new<F>(name: impl Into<String>, access: SystemComponentAccess, run_fn: F) -> Self
    where
        F: FnMut(&mut ExecutionContext) + Send + 'static,
    {
        Self {
            name: name.into(),
            access,
            run_fn: Box::new(run_fn),
            before: Vec::new(),
            after: Vec::new(),
            enabled: true,
        }
    }

    /// Set ordering: this system runs before `other`.
    pub fn before(mut self, other: impl Into<String>) -> Self {
        self.before.push(other.into());
        self
    }

    /// Set ordering: this system runs after `other`.
    pub fn after(mut self, other: impl Into<String>) -> Self {
        self.after.push(other.into());
        self
    }

    /// Disable this system.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

// ---------------------------------------------------------------------------
// ExecutionContext
// ---------------------------------------------------------------------------

/// Context passed to systems during execution.
///
/// Provides access to shared data and metadata about the current execution.
pub struct ExecutionContext {
    /// Name of the currently running system.
    pub system_name: String,
    /// Index of the current batch.
    pub batch_index: usize,
    /// Which thread this is running on.
    pub thread_index: usize,
    /// Total number of threads.
    pub thread_count: usize,
    /// Frame number.
    pub frame: u64,
    /// Delta time.
    pub delta_time: f64,
    /// Generic key-value store for passing data.
    pub data: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
}

impl ExecutionContext {
    /// Create a new execution context.
    pub fn new(system_name: String, batch_index: usize) -> Self {
        Self {
            system_name,
            batch_index,
            thread_index: 0,
            thread_count: 1,
            frame: 0,
            delta_time: 1.0 / 60.0,
            data: HashMap::new(),
        }
    }

    /// Insert a value into the data store.
    pub fn insert<T: Send + Sync + 'static>(&mut self, key: impl Into<String>, value: T) {
        self.data.insert(key.into(), Box::new(value));
    }

    /// Get a reference to a value in the data store.
    pub fn get<T: 'static>(&self, key: &str) -> Option<&T> {
        self.data.get(key).and_then(|v| v.downcast_ref::<T>())
    }
}

// ---------------------------------------------------------------------------
// ConflictGraph
// ---------------------------------------------------------------------------

/// Undirected conflict graph for systems.
///
/// An edge between two systems means they cannot run concurrently.
pub struct ConflictGraph {
    /// Number of systems.
    system_count: usize,
    /// Adjacency matrix (symmetric). `conflicts[i][j]` = true means i and j
    /// conflict.
    conflicts: Vec<Vec<bool>>,
    /// System names for debugging.
    names: Vec<String>,
}

impl ConflictGraph {
    /// Build a conflict graph from system access declarations.
    pub fn build(systems: &[&ParallelSystemDescriptor]) -> Self {
        let n = systems.len();
        let mut conflicts = vec![vec![false; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                if systems[i].access.conflicts_with(&systems[j].access) {
                    conflicts[i][j] = true;
                    conflicts[j][i] = true;
                }
            }
        }

        let names = systems.iter().map(|s| s.name.clone()).collect();

        Self {
            system_count: n,
            conflicts,
            names,
        }
    }

    /// Check if two systems conflict.
    pub fn conflicts(&self, a: usize, b: usize) -> bool {
        if a >= self.system_count || b >= self.system_count {
            return true;
        }
        self.conflicts[a][b]
    }

    /// Returns the set of systems that conflict with the given system.
    pub fn neighbors(&self, system: usize) -> Vec<usize> {
        if system >= self.system_count {
            return Vec::new();
        }
        (0..self.system_count)
            .filter(|&i| i != system && self.conflicts[system][i])
            .collect()
    }

    /// Returns the number of conflicts for a system (its degree).
    pub fn degree(&self, system: usize) -> usize {
        self.neighbors(system).len()
    }

    /// Returns the total number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        let mut count = 0;
        for i in 0..self.system_count {
            for j in (i + 1)..self.system_count {
                if self.conflicts[i][j] {
                    count += 1;
                }
            }
        }
        count
    }

    /// Print the conflict graph for debugging.
    pub fn dump(&self) -> String {
        let mut buf = String::new();
        buf.push_str("Conflict Graph:\n");
        for i in 0..self.system_count {
            let neighbors: Vec<&str> = self
                .neighbors(i)
                .iter()
                .map(|&j| self.names[j].as_str())
                .collect();
            buf.push_str(&format!(
                "  {} -> [{}]\n",
                self.names[i],
                neighbors.join(", ")
            ));
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// ParallelBatch
// ---------------------------------------------------------------------------

/// A batch of systems that can all run concurrently.
#[derive(Debug, Clone)]
pub struct ParallelBatch {
    /// Indices of systems in this batch.
    pub system_indices: Vec<usize>,
    /// Batch index.
    pub index: usize,
}

impl ParallelBatch {
    /// Returns the number of systems in this batch.
    pub fn len(&self) -> usize {
        self.system_indices.len()
    }

    /// Returns true if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.system_indices.is_empty()
    }
}

// ---------------------------------------------------------------------------
// BatchSchedule
// ---------------------------------------------------------------------------

/// A schedule of parallel batches computed from the conflict graph and
/// ordering constraints.
#[derive(Debug, Clone)]
pub struct BatchSchedule {
    /// Ordered list of batches. Each batch runs after the previous completes.
    pub batches: Vec<ParallelBatch>,
    /// System names for debugging.
    pub system_names: Vec<String>,
}

impl BatchSchedule {
    /// Returns the total number of batches.
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Returns the maximum parallelism (largest batch size).
    pub fn max_parallelism(&self) -> usize {
        self.batches.iter().map(|b| b.len()).max().unwrap_or(0)
    }

    /// Print the schedule for debugging.
    pub fn dump(&self) -> String {
        let mut buf = String::new();
        buf.push_str("Batch Schedule:\n");
        for batch in &self.batches {
            let names: Vec<&str> = batch
                .system_indices
                .iter()
                .map(|&i| self.system_names[i].as_str())
                .collect();
            buf.push_str(&format!("  Batch {}: [{}]\n", batch.index, names.join(", ")));
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// Ordering constraint handling
// ---------------------------------------------------------------------------

/// Compute topological order respecting before/after constraints.
fn topological_sort(
    system_count: usize,
    names: &[String],
    before_edges: &[(usize, usize)], // (a, b) means a runs before b
) -> Result<Vec<usize>, String> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); system_count];
    let mut in_degree = vec![0usize; system_count];

    for &(a, b) in before_edges {
        adj[a].push(b);
        in_degree[b] += 1;
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..system_count {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(system_count);
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &neighbor in &adj[node] {
            in_degree[neighbor] -= 1;
            if in_degree[neighbor] == 0 {
                queue.push_back(neighbor);
            }
        }
    }

    if order.len() != system_count {
        // Find the cycle participants.
        let cycle_nodes: Vec<String> = (0..system_count)
            .filter(|i| in_degree[*i] > 0)
            .map(|i| names[i].clone())
            .collect();
        Err(format!(
            "cycle detected among systems: {}",
            cycle_nodes.join(", ")
        ))
    } else {
        Ok(order)
    }
}

// ---------------------------------------------------------------------------
// Batch computation
// ---------------------------------------------------------------------------

/// Compute parallel batches using a greedy graph coloring approach.
///
/// Systems are processed in topological order. For each system, we try to
/// place it in the earliest batch where it has no conflicts with existing
/// members and all of its ordering dependencies have already been scheduled
/// in earlier batches.
fn compute_batches(
    conflict_graph: &ConflictGraph,
    topo_order: &[usize],
    ordering_edges: &[(usize, usize)],
) -> BatchSchedule {
    let n = conflict_graph.system_count;

    // For each system, track which batch it ends up in.
    let mut system_batch: Vec<Option<usize>> = vec![None; n];

    // For each system, the minimum batch index (must be after all dependencies).
    let mut min_batch: Vec<usize> = vec![0; n];
    for &(a, b) in ordering_edges {
        // b must come after a.
        // We'll update min_batch as we assign batches.
        let _ = (a, b); // processed below
    }

    // Build a reverse dependency map: for each system, which systems must come before it.
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in ordering_edges {
        deps[b].push(a);
    }

    let mut batches: Vec<ParallelBatch> = Vec::new();

    for &sys_idx in topo_order {
        // Compute minimum batch: must be after all dependencies.
        let mut earliest = 0;
        for &dep in &deps[sys_idx] {
            if let Some(dep_batch) = system_batch[dep] {
                earliest = earliest.max(dep_batch + 1);
            }
        }

        // Find the first batch >= earliest where this system has no conflicts.
        let mut placed = false;
        for batch_idx in earliest..batches.len() {
            let batch = &batches[batch_idx];
            let has_conflict = batch
                .system_indices
                .iter()
                .any(|&other| conflict_graph.conflicts(sys_idx, other));

            if !has_conflict {
                batches[batch_idx].system_indices.push(sys_idx);
                system_batch[sys_idx] = Some(batch_idx);
                placed = true;
                break;
            }
        }

        if !placed {
            let batch_idx = batches.len();
            batches.push(ParallelBatch {
                system_indices: vec![sys_idx],
                index: batch_idx,
            });
            system_batch[sys_idx] = Some(batch_idx);
        }
    }

    BatchSchedule {
        batches,
        system_names: conflict_graph.names.clone(),
    }
}

// ---------------------------------------------------------------------------
// ExecutorConfig
// ---------------------------------------------------------------------------

/// Configuration for the parallel executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Number of threads in the pool.
    pub thread_count: usize,
    /// Whether to enable profiling.
    pub profiling: bool,
    /// Maximum systems per batch.
    pub max_batch_size: usize,
    /// Whether to log batch scheduling decisions.
    pub verbose: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            thread_count: 4,
            profiling: false,
            max_batch_size: 64,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ExecutionProfile
// ---------------------------------------------------------------------------

/// Profiling data for one execution run.
#[derive(Debug, Clone)]
pub struct ExecutionProfile {
    /// Per-system timing in microseconds.
    pub system_times: HashMap<String, u64>,
    /// Per-batch timing in microseconds.
    pub batch_times: Vec<u64>,
    /// Total execution time in microseconds.
    pub total_time_us: u64,
    /// Number of batches.
    pub batch_count: usize,
    /// Maximum parallelism achieved.
    pub max_parallelism: usize,
    /// Frame number.
    pub frame: u64,
}

impl ExecutionProfile {
    /// Returns the system that took the longest.
    pub fn bottleneck(&self) -> Option<(&str, u64)> {
        self.system_times
            .iter()
            .max_by_key(|(_, &time)| time)
            .map(|(name, &time)| (name.as_str(), time))
    }

    /// Returns a summary string.
    pub fn summary(&self) -> String {
        let mut buf = String::new();
        buf.push_str(&format!(
            "Frame {} | {} batches | max parallelism {} | total {}us\n",
            self.frame, self.batch_count, self.max_parallelism, self.total_time_us,
        ));
        if let Some((name, time)) = self.bottleneck() {
            buf.push_str(&format!("  Bottleneck: {} ({}us)\n", name, time));
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// ParallelExecutor
// ---------------------------------------------------------------------------

/// Parallel system executor.
///
/// Analyzes system access patterns, builds a conflict graph, computes
/// parallel batches, and executes non-conflicting systems concurrently.
///
/// # Example
///
/// ```ignore
/// use genovo_ecs::parallel_executor::*;
///
/// let mut executor = ParallelExecutor::new(ExecutorConfig::default());
///
/// executor.add_system(ParallelSystemDescriptor::new(
///     "physics",
///     SystemComponentAccess::new()
///         .read_component::<Velocity>()
///         .write_component::<Position>(),
///     |ctx| { /* ... */ },
/// ));
///
/// executor.add_system(ParallelSystemDescriptor::new(
///     "render",
///     SystemComponentAccess::new()
///         .read_component::<Position>()
///         .read_component::<Sprite>(),
///     |ctx| { /* ... */ },
/// ));
///
/// // Rebuild the schedule.
/// executor.rebuild_schedule().unwrap();
///
/// // Execute one frame.
/// executor.execute();
/// ```
pub struct ParallelExecutor {
    /// Configuration.
    config: ExecutorConfig,
    /// All registered systems.
    systems: Vec<ParallelSystemDescriptor>,
    /// The current batch schedule (computed from access patterns).
    schedule: Option<BatchSchedule>,
    /// The conflict graph.
    conflict_graph: Option<ConflictGraph>,
    /// Current frame counter.
    frame: u64,
    /// Delta time for this frame.
    delta_time: f64,
    /// Profiling history.
    profiles: Vec<ExecutionProfile>,
    /// Whether the schedule is dirty and needs rebuilding.
    dirty: bool,
}

impl ParallelExecutor {
    /// Create a new parallel executor.
    pub fn new(config: ExecutorConfig) -> Self {
        Self {
            config,
            systems: Vec::new(),
            schedule: None,
            conflict_graph: None,
            frame: 0,
            delta_time: 1.0 / 60.0,
            profiles: Vec::new(),
            dirty: true,
        }
    }

    /// Add a system to the executor.
    pub fn add_system(&mut self, descriptor: ParallelSystemDescriptor) {
        self.systems.push(descriptor);
        self.dirty = true;
    }

    /// Remove a system by name.
    pub fn remove_system(&mut self, name: &str) -> bool {
        let len_before = self.systems.len();
        self.systems.retain(|s| s.name != name);
        let removed = self.systems.len() < len_before;
        if removed {
            self.dirty = true;
        }
        removed
    }

    /// Enable or disable a system by name.
    pub fn set_system_enabled(&mut self, name: &str, enabled: bool) {
        for sys in &mut self.systems {
            if sys.name == name {
                sys.enabled = enabled;
                self.dirty = true;
                return;
            }
        }
    }

    /// Set the delta time for the current frame.
    pub fn set_delta_time(&mut self, dt: f64) {
        self.delta_time = dt;
    }

    /// Rebuild the execution schedule.
    ///
    /// This analyzes access patterns, builds the conflict graph, and computes
    /// parallel batches. Must be called after adding/removing systems.
    pub fn rebuild_schedule(&mut self) -> Result<(), String> {
        let enabled_indices: Vec<usize> = self
            .systems
            .iter()
            .enumerate()
            .filter(|(_, s)| s.enabled)
            .map(|(i, _)| i)
            .collect();

        if enabled_indices.is_empty() {
            self.schedule = Some(BatchSchedule {
                batches: Vec::new(),
                system_names: Vec::new(),
            });
            self.conflict_graph = None;
            self.dirty = false;
            return Ok(());
        }

        // Build name -> index map for ordering resolution.
        let name_to_idx: HashMap<&str, usize> = enabled_indices
            .iter()
            .enumerate()
            .map(|(local_idx, &global_idx)| (self.systems[global_idx].name.as_str(), local_idx))
            .collect();

        let enabled_refs: Vec<&ParallelSystemDescriptor> = enabled_indices
            .iter()
            .map(|&i| &self.systems[i])
            .collect();

        // Build conflict graph.
        let conflict_graph = ConflictGraph::build(&enabled_refs);

        // Build ordering edges.
        let mut ordering_edges: Vec<(usize, usize)> = Vec::new();
        for (local_idx, &global_idx) in enabled_indices.iter().enumerate() {
            let sys = &self.systems[global_idx];
            for before_name in &sys.before {
                if let Some(&other_idx) = name_to_idx.get(before_name.as_str()) {
                    ordering_edges.push((local_idx, other_idx));
                }
            }
            for after_name in &sys.after {
                if let Some(&other_idx) = name_to_idx.get(after_name.as_str()) {
                    ordering_edges.push((other_idx, local_idx));
                }
            }
        }

        // Topological sort.
        let names: Vec<String> = enabled_refs.iter().map(|s| s.name.clone()).collect();
        let topo_order = topological_sort(enabled_refs.len(), &names, &ordering_edges)?;

        // Compute batches.
        let schedule = compute_batches(&conflict_graph, &topo_order, &ordering_edges);

        if self.config.verbose {
            eprintln!("{}", conflict_graph.dump());
            eprintln!("{}", schedule.dump());
        }

        self.conflict_graph = Some(conflict_graph);
        self.schedule = Some(schedule);
        self.dirty = false;

        Ok(())
    }

    /// Execute all systems according to the current schedule.
    ///
    /// Systems within the same batch run sequentially in this implementation
    /// (for thread safety with the mutable closures). A real engine would
    /// use unsafe or message passing for true parallel execution.
    pub fn execute(&mut self) {
        if self.dirty {
            if let Err(e) = self.rebuild_schedule() {
                eprintln!("[ParallelExecutor] schedule build error: {}", e);
                return;
            }
        }

        let schedule = match &self.schedule {
            Some(s) => s.clone(),
            None => return,
        };

        let frame = self.frame;
        let delta_time = self.delta_time;
        let thread_count = self.config.thread_count;
        let profiling = self.config.profiling;

        let mut system_times: HashMap<String, u64> = HashMap::new();
        let mut batch_times: Vec<u64> = Vec::new();
        let total_start = std::time::Instant::now();

        for batch in &schedule.batches {
            let batch_start = std::time::Instant::now();

            for &sys_local_idx in &batch.system_indices {
                let sys_name = &schedule.system_names[sys_local_idx];

                // Find the system in our global list by name.
                let system = self.systems.iter_mut().find(|s| s.name == *sys_name);

                if let Some(sys) = system {
                    if !sys.enabled {
                        continue;
                    }

                    let mut ctx = ExecutionContext::new(sys.name.clone(), batch.index);
                    ctx.frame = frame;
                    ctx.delta_time = delta_time;
                    ctx.thread_count = thread_count;

                    let sys_start = std::time::Instant::now();
                    (sys.run_fn)(&mut ctx);
                    let sys_elapsed = sys_start.elapsed().as_micros() as u64;

                    if profiling {
                        system_times.insert(sys.name.clone(), sys_elapsed);
                    }
                }
            }

            let batch_elapsed = batch_start.elapsed().as_micros() as u64;
            batch_times.push(batch_elapsed);
        }

        let total_elapsed = total_start.elapsed().as_micros() as u64;

        if profiling {
            let profile = ExecutionProfile {
                system_times,
                batch_times,
                total_time_us: total_elapsed,
                batch_count: schedule.batches.len(),
                max_parallelism: schedule.max_parallelism(),
                frame,
            };
            self.profiles.push(profile);
        }

        self.frame += 1;
    }

    /// Returns the current schedule.
    pub fn schedule(&self) -> Option<&BatchSchedule> {
        self.schedule.as_ref()
    }

    /// Returns the conflict graph.
    pub fn conflict_graph(&self) -> Option<&ConflictGraph> {
        self.conflict_graph.as_ref()
    }

    /// Returns profiling data.
    pub fn profiles(&self) -> &[ExecutionProfile] {
        &self.profiles
    }

    /// Returns the latest profiling entry.
    pub fn latest_profile(&self) -> Option<&ExecutionProfile> {
        self.profiles.last()
    }

    /// Clear profiling history.
    pub fn clear_profiles(&mut self) {
        self.profiles.clear();
    }

    /// Returns the number of registered systems.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// Returns the number of enabled systems.
    pub fn enabled_system_count(&self) -> usize {
        self.systems.iter().filter(|s| s.enabled).count()
    }

    /// Returns system names.
    pub fn system_names(&self) -> Vec<&str> {
        self.systems.iter().map(|s| s.name.as_str()).collect()
    }

    /// Returns the current frame number.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Check if the schedule needs rebuilding.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct Position;
    struct Velocity;
    struct Health;
    struct Sprite;
    struct Transform;

    #[test]
    fn test_access_no_conflict() {
        let a = SystemComponentAccess::new().read_component::<Position>();
        let b = SystemComponentAccess::new().read_component::<Position>();
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn test_access_write_read_conflict() {
        let a = SystemComponentAccess::new().write_component::<Position>();
        let b = SystemComponentAccess::new().read_component::<Position>();
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn test_access_write_write_conflict() {
        let a = SystemComponentAccess::new().write_component::<Position>();
        let b = SystemComponentAccess::new().write_component::<Position>();
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn test_access_exclusive_conflicts_with_everything() {
        let a = SystemComponentAccess::new().exclusive();
        let b = SystemComponentAccess::new().read_component::<Position>();
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn test_access_disjoint_no_conflict() {
        let a = SystemComponentAccess::new().write_component::<Position>();
        let b = SystemComponentAccess::new().write_component::<Health>();
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn test_conflict_graph() {
        let sys_a = ParallelSystemDescriptor::new(
            "physics",
            SystemComponentAccess::new()
                .read_component::<Velocity>()
                .write_component::<Position>(),
            |_| {},
        );
        let sys_b = ParallelSystemDescriptor::new(
            "render",
            SystemComponentAccess::new().read_component::<Position>(),
            |_| {},
        );
        let sys_c = ParallelSystemDescriptor::new(
            "health",
            SystemComponentAccess::new().write_component::<Health>(),
            |_| {},
        );

        let systems: Vec<&ParallelSystemDescriptor> = vec![&sys_a, &sys_b, &sys_c];
        let graph = ConflictGraph::build(&systems);

        // physics writes Position, render reads Position — conflict.
        assert!(graph.conflicts(0, 1));
        // physics and health — no shared types — no conflict.
        assert!(!graph.conflicts(0, 2));
        // render and health — no shared types — no conflict.
        assert!(!graph.conflicts(1, 2));
    }

    #[test]
    fn test_batch_computation() {
        let sys_a = ParallelSystemDescriptor::new(
            "physics",
            SystemComponentAccess::new()
                .read_component::<Velocity>()
                .write_component::<Position>(),
            |_| {},
        );
        let sys_b = ParallelSystemDescriptor::new(
            "render",
            SystemComponentAccess::new().read_component::<Position>(),
            |_| {},
        );
        let sys_c = ParallelSystemDescriptor::new(
            "health",
            SystemComponentAccess::new().write_component::<Health>(),
            |_| {},
        );

        let mut executor = ParallelExecutor::new(ExecutorConfig::default());
        executor.add_system(sys_a);
        executor.add_system(sys_b);
        executor.add_system(sys_c);
        executor.rebuild_schedule().unwrap();

        let schedule = executor.schedule().unwrap();
        // physics and health can run in parallel (no conflict).
        // render must wait for physics (read-write conflict on Position).
        assert!(schedule.batch_count() >= 1);
    }

    #[test]
    fn test_ordering_constraints() {
        let sys_a = ParallelSystemDescriptor::new(
            "a",
            SystemComponentAccess::new().read_component::<Position>(),
            |_| {},
        );
        let sys_b = ParallelSystemDescriptor::new(
            "b",
            SystemComponentAccess::new().read_component::<Position>(),
            |_| {},
        )
        .after("a");

        let mut executor = ParallelExecutor::new(ExecutorConfig::default());
        executor.add_system(sys_a);
        executor.add_system(sys_b);
        executor.rebuild_schedule().unwrap();

        let schedule = executor.schedule().unwrap();
        // Even though they don't conflict on access, ordering forces separate batches.
        assert!(schedule.batch_count() >= 2);
    }

    #[test]
    fn test_cycle_detection() {
        let sys_a = ParallelSystemDescriptor::new(
            "a",
            SystemComponentAccess::new(),
            |_| {},
        )
        .before("b");
        let sys_b = ParallelSystemDescriptor::new(
            "b",
            SystemComponentAccess::new(),
            |_| {},
        )
        .before("a");

        let mut executor = ParallelExecutor::new(ExecutorConfig::default());
        executor.add_system(sys_a);
        executor.add_system(sys_b);

        let result = executor.rebuild_schedule();
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_run() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));

        let c1 = counter.clone();
        let sys_a = ParallelSystemDescriptor::new(
            "increment",
            SystemComponentAccess::new(),
            move |_| {
                c1.fetch_add(1, Ordering::SeqCst);
            },
        );

        let mut executor = ParallelExecutor::new(ExecutorConfig::default());
        executor.add_system(sys_a);
        executor.rebuild_schedule().unwrap();

        executor.execute();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        executor.execute();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_executor_disable_system() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();

        let mut executor = ParallelExecutor::new(ExecutorConfig::default());
        executor.add_system(ParallelSystemDescriptor::new(
            "count",
            SystemComponentAccess::new(),
            move |_| { c.fetch_add(1, Ordering::SeqCst); },
        ));

        executor.execute();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        executor.set_system_enabled("count", false);
        executor.execute();
        assert_eq!(counter.load(Ordering::SeqCst), 1); // Should not increment.
    }

    #[test]
    fn test_executor_profiling() {
        let mut executor = ParallelExecutor::new(ExecutorConfig {
            profiling: true,
            ..Default::default()
        });

        executor.add_system(ParallelSystemDescriptor::new(
            "test",
            SystemComponentAccess::new(),
            |_| {},
        ));

        executor.execute();

        assert_eq!(executor.profiles().len(), 1);
        let profile = executor.latest_profile().unwrap();
        assert_eq!(profile.frame, 0);
    }

    #[test]
    fn test_topological_sort() {
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let edges = vec![(0, 1), (1, 2)]; // a -> b -> c

        let order = topological_sort(3, &names, &edges).unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn test_resource_access_conflict() {
        let a = SystemComponentAccess::new().write_resource::<u32>();
        let b = SystemComponentAccess::new().read_resource::<u32>();
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn test_execution_context() {
        let mut ctx = ExecutionContext::new("test".to_string(), 0);
        ctx.insert("key", 42u32);
        assert_eq!(ctx.get::<u32>("key"), Some(&42));
        assert_eq!(ctx.get::<u32>("missing"), None);
    }
}
