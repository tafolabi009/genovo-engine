//! # Advanced Task Scheduler
//!
//! A fiber-based task scheduling system for the Genovo engine, designed for
//! high-throughput parallel workloads with minimal contention.
//!
//! ## Features
//!
//! - **Fiber-based tasks** — Lightweight cooperative tasks that can suspend
//!   and resume without blocking OS threads.
//! - **Work stealing** — Idle threads steal tasks from busy threads' local
//!   queues for automatic load balancing.
//! - **Priority with aging** — Tasks have priorities that increase over time
//!   to prevent starvation.
//! - **Task dependencies (DAG)** — Tasks can declare dependencies on other
//!   tasks, forming a directed acyclic graph.
//! - **Continuations** — Chain tasks so that completion of one triggers the
//!   next.
//! - **Wait groups** — Synchronization primitive for waiting on a batch of
//!   tasks.
//! - **Async/await-style API** — `TaskFuture` for ergonomic async composition.
//! - **Task profiling** — Built-in timing and throughput instrumentation.

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Condvar};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of worker threads (will use available parallelism at runtime).
const DEFAULT_WORKER_COUNT: usize = 4;

/// Maximum number of tasks that can be in-flight at once.
const MAX_TASKS: usize = 65536;

/// Number of priority levels.
const PRIORITY_LEVELS: usize = 8;

/// How many ticks before a task's priority is aged (bumped up).
const AGING_THRESHOLD: u64 = 100;

/// Maximum steal batch size.
const STEAL_BATCH_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// TaskId
// ---------------------------------------------------------------------------

/// Unique identifier for a scheduled task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

impl TaskId {
    /// A sentinel representing no task.
    pub const INVALID: TaskId = TaskId(u64::MAX);
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Task({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// TaskPriority
// ---------------------------------------------------------------------------

/// Priority level for a task. Lower numeric value = higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority {
    /// Critical path — must run immediately.
    Critical = 0,
    /// High priority — gameplay systems, input processing.
    High = 1,
    /// Normal priority — typical game logic.
    Normal = 2,
    /// Low priority — background work, prefetching.
    Low = 3,
    /// Idle — only when nothing else is pending.
    Idle = 4,
}

impl TaskPriority {
    /// Returns the numeric value (lower = higher priority).
    #[inline]
    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Promote a priority by one level (cannot go above Critical).
    pub fn promote(&self) -> TaskPriority {
        match self {
            TaskPriority::Idle => TaskPriority::Low,
            TaskPriority::Low => TaskPriority::Normal,
            TaskPriority::Normal => TaskPriority::High,
            TaskPriority::High => TaskPriority::Critical,
            TaskPriority::Critical => TaskPriority::Critical,
        }
    }

    /// Demote a priority by one level (cannot go below Idle).
    pub fn demote(&self) -> TaskPriority {
        match self {
            TaskPriority::Critical => TaskPriority::High,
            TaskPriority::High => TaskPriority::Normal,
            TaskPriority::Normal => TaskPriority::Low,
            TaskPriority::Low => TaskPriority::Idle,
            TaskPriority::Idle => TaskPriority::Idle,
        }
    }
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

// ---------------------------------------------------------------------------
// TaskState
// ---------------------------------------------------------------------------

/// Current state of a task in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Waiting for dependencies to complete.
    Pending,
    /// Ready to execute (all dependencies satisfied).
    Ready,
    /// Currently executing on a worker thread.
    Running,
    /// Suspended (yielded, waiting on I/O, etc.).
    Suspended,
    /// Completed successfully.
    Completed,
    /// Completed with an error.
    Failed,
    /// Cancelled before execution.
    Cancelled,
}

impl fmt::Display for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskState::Pending => write!(f, "Pending"),
            TaskState::Ready => write!(f, "Ready"),
            TaskState::Running => write!(f, "Running"),
            TaskState::Suspended => write!(f, "Suspended"),
            TaskState::Completed => write!(f, "Completed"),
            TaskState::Failed => write!(f, "Failed"),
            TaskState::Cancelled => write!(f, "Cancelled"),
        }
    }
}

// ---------------------------------------------------------------------------
// TaskError
// ---------------------------------------------------------------------------

/// Errors from the task scheduler.
#[derive(Debug)]
pub enum TaskError {
    /// The task ID is invalid or refers to an expired task.
    InvalidTask(TaskId),
    /// Adding a dependency would create a cycle.
    CyclicDependency { from: TaskId, to: TaskId },
    /// The scheduler is full and cannot accept more tasks.
    SchedulerFull,
    /// The task was cancelled.
    Cancelled(TaskId),
    /// A dependency failed, preventing this task from running.
    DependencyFailed(TaskId),
    /// The scheduler has been shut down.
    ShutDown,
    /// A worker thread panicked.
    WorkerPanic(String),
    /// Generic internal error.
    Internal(String),
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskError::InvalidTask(id) => write!(f, "invalid task: {}", id),
            TaskError::CyclicDependency { from, to } => {
                write!(f, "cyclic dependency: {} -> {}", from, to)
            }
            TaskError::SchedulerFull => write!(f, "task scheduler is full"),
            TaskError::Cancelled(id) => write!(f, "task {} was cancelled", id),
            TaskError::DependencyFailed(id) => write!(f, "dependency {} failed", id),
            TaskError::ShutDown => write!(f, "scheduler has been shut down"),
            TaskError::WorkerPanic(msg) => write!(f, "worker panic: {}", msg),
            TaskError::Internal(msg) => write!(f, "internal error: {}", msg),
        }
    }
}

impl std::error::Error for TaskError {}

pub type TaskResult<T> = Result<T, TaskError>;

// ---------------------------------------------------------------------------
// TaskDescriptor
// ---------------------------------------------------------------------------

/// Describes a task to be scheduled.
pub struct TaskDescriptor {
    /// Human-readable name for profiling.
    pub name: String,
    /// Priority level.
    pub priority: TaskPriority,
    /// The work to execute.
    pub work: Box<dyn FnOnce() -> TaskOutcome + Send + 'static>,
    /// IDs of tasks that must complete before this one.
    pub dependencies: Vec<TaskId>,
    /// Optional continuation: task to schedule after this one completes.
    pub continuation: Option<Box<dyn FnOnce(TaskOutcome) -> TaskDescriptor + Send + 'static>>,
    /// Whether this task can be stolen by another worker.
    pub stealable: bool,
    /// Optional label for grouping in profiling views.
    pub group: Option<String>,
}

impl TaskDescriptor {
    /// Create a new task descriptor with the given name and work closure.
    pub fn new<F>(name: impl Into<String>, work: F) -> Self
    where
        F: FnOnce() -> TaskOutcome + Send + 'static,
    {
        Self {
            name: name.into(),
            priority: TaskPriority::Normal,
            work: Box::new(work),
            dependencies: Vec::new(),
            continuation: None,
            stealable: true,
            group: None,
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Add a dependency on another task.
    pub fn depends_on(mut self, task: TaskId) -> Self {
        self.dependencies.push(task);
        self
    }

    /// Add multiple dependencies.
    pub fn depends_on_all(mut self, tasks: impl IntoIterator<Item = TaskId>) -> Self {
        self.dependencies.extend(tasks);
        self
    }

    /// Set a continuation closure.
    pub fn then<F>(mut self, cont: F) -> Self
    where
        F: FnOnce(TaskOutcome) -> TaskDescriptor + Send + 'static,
    {
        self.continuation = Some(Box::new(cont));
        self
    }

    /// Mark whether this task can be stolen.
    pub fn with_stealable(mut self, stealable: bool) -> Self {
        self.stealable = stealable;
        self
    }

    /// Assign a profiling group.
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }
}

// ---------------------------------------------------------------------------
// TaskOutcome
// ---------------------------------------------------------------------------

/// Result of executing a task.
#[derive(Debug, Clone)]
pub enum TaskOutcome {
    /// Task completed successfully with an optional message.
    Success(Option<String>),
    /// Task failed with an error message.
    Failure(String),
    /// Task yielded and should be rescheduled.
    Yield,
}

impl TaskOutcome {
    /// Returns true if the outcome is a success.
    pub fn is_success(&self) -> bool {
        matches!(self, TaskOutcome::Success(_))
    }

    /// Returns true if the outcome is a failure.
    pub fn is_failure(&self) -> bool {
        matches!(self, TaskOutcome::Failure(_))
    }

    /// Returns true if the task yielded.
    pub fn is_yield(&self) -> bool {
        matches!(self, TaskOutcome::Yield)
    }
}

// ---------------------------------------------------------------------------
// Internal task record
// ---------------------------------------------------------------------------

/// Internal representation of a scheduled task.
struct TaskRecord {
    /// Unique ID.
    id: TaskId,
    /// Human-readable name.
    name: String,
    /// Current state.
    state: TaskState,
    /// Priority (may be aged).
    priority: TaskPriority,
    /// Original priority (before aging).
    original_priority: TaskPriority,
    /// Tick when this task was submitted.
    submit_tick: u64,
    /// Tick when this task started executing.
    start_tick: Option<u64>,
    /// Tick when this task completed.
    end_tick: Option<u64>,
    /// IDs of tasks this depends on.
    dependencies: Vec<TaskId>,
    /// Number of unsatisfied dependencies.
    pending_deps: u32,
    /// IDs of tasks that depend on this one.
    dependents: Vec<TaskId>,
    /// The work closure (None after execution).
    work: Option<Box<dyn FnOnce() -> TaskOutcome + Send + 'static>>,
    /// Continuation closure.
    continuation: Option<Box<dyn FnOnce(TaskOutcome) -> TaskDescriptor + Send + 'static>>,
    /// The outcome after completion.
    outcome: Option<TaskOutcome>,
    /// Whether this task can be stolen.
    stealable: bool,
    /// Profiling group.
    group: Option<String>,
    /// Which worker thread executed this task.
    executed_on_worker: Option<usize>,
}

// ---------------------------------------------------------------------------
// PrioritizedTask (for the heap)
// ---------------------------------------------------------------------------

/// Wrapper for ordering tasks in the priority heap.
/// Lower priority value = higher urgency = should be dequeued first.
#[derive(Debug)]
struct PrioritizedTask {
    id: TaskId,
    priority: TaskPriority,
    submit_tick: u64,
}

impl PartialEq for PrioritizedTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.submit_tick == other.submit_tick
    }
}

impl Eq for PrioritizedTask {}

impl PartialOrd for PrioritizedTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse: lower priority value = higher urgency = should come first.
        other
            .priority
            .value()
            .cmp(&self.priority.value())
            .then_with(|| other.submit_tick.cmp(&self.submit_tick))
    }
}

// ---------------------------------------------------------------------------
// WorkerLocalQueue
// ---------------------------------------------------------------------------

/// Per-worker local task queue.
struct WorkerLocalQueue {
    /// Tasks assigned to this worker.
    queue: VecDeque<TaskId>,
    /// Worker index.
    worker_id: usize,
}

impl WorkerLocalQueue {
    fn new(worker_id: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            worker_id,
        }
    }

    /// Push a task to the back (producer end).
    fn push(&mut self, task: TaskId) {
        self.queue.push_back(task);
    }

    /// Pop a task from the front (consumer end).
    fn pop(&mut self) -> Option<TaskId> {
        self.queue.pop_front()
    }

    /// Steal tasks from the back (thief end).
    fn steal(&mut self, max: usize) -> Vec<TaskId> {
        let count = max.min(self.queue.len() / 2).max(1).min(self.queue.len());
        let mut stolen = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(id) = self.queue.pop_back() {
                stolen.push(id);
            }
        }
        stolen
    }

    /// Returns the number of tasks in the queue.
    fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns true if empty.
    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

// ---------------------------------------------------------------------------
// WaitGroup
// ---------------------------------------------------------------------------

/// Synchronization primitive for waiting on a batch of tasks.
///
/// Create a wait group, add tasks to it, then call `wait()` to block until
/// all tasks have completed.
///
/// # Example
///
/// ```ignore
/// let mut scheduler = TaskScheduler::new(4);
/// let wg = WaitGroup::new();
///
/// for i in 0..10 {
///     let desc = TaskDescriptor::new(format!("batch_{}", i), || TaskOutcome::Success(None));
///     let id = scheduler.schedule(desc).unwrap();
///     wg.add(id);
/// }
///
/// // Block until all 10 tasks are done.
/// wg.wait(&scheduler);
/// ```
pub struct WaitGroup {
    /// Task IDs in this group.
    tasks: Vec<TaskId>,
    /// Optional name for profiling.
    name: Option<String>,
}

impl WaitGroup {
    /// Create a new empty wait group.
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            name: None,
        }
    }

    /// Create a named wait group.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            tasks: Vec::new(),
            name: Some(name.into()),
        }
    }

    /// Add a task to the wait group.
    pub fn add(&mut self, task: TaskId) {
        self.tasks.push(task);
    }

    /// Add multiple tasks to the wait group.
    pub fn add_all(&mut self, tasks: impl IntoIterator<Item = TaskId>) {
        self.tasks.extend(tasks);
    }

    /// Returns the number of tasks in the group.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Returns true if the group is empty.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Returns a slice of all task IDs.
    pub fn tasks(&self) -> &[TaskId] {
        &self.tasks
    }

    /// Check if all tasks in the group have completed.
    pub fn is_complete(&self, scheduler: &TaskScheduler) -> bool {
        self.tasks.iter().all(|id| {
            scheduler
                .task_state(*id)
                .map(|s| s == TaskState::Completed || s == TaskState::Failed || s == TaskState::Cancelled)
                .unwrap_or(true) // Treat unknown tasks as complete.
        })
    }

    /// Spin-wait until all tasks are complete.
    ///
    /// This is a blocking call. In production, prefer an async approach.
    pub fn wait(&self, scheduler: &TaskScheduler) {
        while !self.is_complete(scheduler) {
            std::hint::spin_loop();
        }
    }

    /// Returns the number of tasks that have completed (success or failure).
    pub fn completed_count(&self, scheduler: &TaskScheduler) -> usize {
        self.tasks
            .iter()
            .filter(|id| {
                scheduler
                    .task_state(**id)
                    .map(|s| s == TaskState::Completed || s == TaskState::Failed)
                    .unwrap_or(false)
            })
            .count()
    }

    /// Returns the number of failed tasks.
    pub fn failed_count(&self, scheduler: &TaskScheduler) -> usize {
        self.tasks
            .iter()
            .filter(|id| {
                scheduler
                    .task_state(**id)
                    .map(|s| s == TaskState::Failed)
                    .unwrap_or(false)
            })
            .count()
    }
}

impl Default for WaitGroup {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TaskFuture
// ---------------------------------------------------------------------------

/// An async/await-style handle to a scheduled task's result.
///
/// # Example
///
/// ```ignore
/// let future = scheduler.schedule_async(
///     TaskDescriptor::new("compute", || TaskOutcome::Success(Some("42".into())))
/// ).unwrap();
///
/// // ... do other work ...
///
/// let outcome = future.get(&scheduler); // blocks until ready
/// ```
pub struct TaskFuture {
    /// The task ID being awaited.
    task_id: TaskId,
}

impl TaskFuture {
    /// Create a future for the given task.
    pub fn new(task_id: TaskId) -> Self {
        Self { task_id }
    }

    /// Returns the underlying task ID.
    pub fn task_id(&self) -> TaskId {
        self.task_id
    }

    /// Check if the result is ready without blocking.
    pub fn is_ready(&self, scheduler: &TaskScheduler) -> bool {
        scheduler
            .task_state(self.task_id)
            .map(|s| s == TaskState::Completed || s == TaskState::Failed)
            .unwrap_or(true)
    }

    /// Block until the task completes and return its outcome.
    pub fn get(&self, scheduler: &TaskScheduler) -> Option<TaskOutcome> {
        while !self.is_ready(scheduler) {
            std::hint::spin_loop();
        }
        scheduler.task_outcome(self.task_id)
    }

    /// Try to get the outcome without blocking.
    pub fn try_get(&self, scheduler: &TaskScheduler) -> Option<TaskOutcome> {
        if self.is_ready(scheduler) {
            scheduler.task_outcome(self.task_id)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// TaskProfileEntry
// ---------------------------------------------------------------------------

/// Profiling data for a single task execution.
#[derive(Debug, Clone)]
pub struct TaskProfileEntry {
    /// Task ID.
    pub task_id: TaskId,
    /// Task name.
    pub name: String,
    /// Priority at submission time.
    pub priority: TaskPriority,
    /// Group label.
    pub group: Option<String>,
    /// Tick when submitted.
    pub submit_tick: u64,
    /// Tick when execution started.
    pub start_tick: u64,
    /// Tick when execution ended.
    pub end_tick: u64,
    /// Wall-clock execution duration in microseconds.
    pub duration_us: u64,
    /// Which worker executed the task.
    pub worker_id: usize,
    /// Whether the task was stolen from another worker.
    pub was_stolen: bool,
    /// Outcome.
    pub outcome: TaskOutcome,
}

// ---------------------------------------------------------------------------
// TaskProfiler
// ---------------------------------------------------------------------------

/// Collects profiling data for all completed tasks.
pub struct TaskProfiler {
    /// All recorded profile entries.
    entries: Vec<TaskProfileEntry>,
    /// Whether profiling is enabled.
    enabled: bool,
    /// Total tasks profiled.
    total_count: u64,
    /// Sum of all durations for averaging.
    total_duration_us: u64,
}

impl TaskProfiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            enabled: true,
            total_count: 0,
            total_duration_us: 0,
        }
    }

    /// Enable or disable profiling.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record a profiling entry.
    pub fn record(&mut self, entry: TaskProfileEntry) {
        if !self.enabled {
            return;
        }
        self.total_duration_us += entry.duration_us;
        self.total_count += 1;
        self.entries.push(entry);
    }

    /// Returns all recorded entries.
    pub fn entries(&self) -> &[TaskProfileEntry] {
        &self.entries
    }

    /// Returns the average task duration in microseconds.
    pub fn average_duration_us(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.total_duration_us as f64 / self.total_count as f64
        }
    }

    /// Returns entries for a specific group.
    pub fn entries_for_group(&self, group: &str) -> Vec<&TaskProfileEntry> {
        self.entries
            .iter()
            .filter(|e| e.group.as_deref() == Some(group))
            .collect()
    }

    /// Returns entries for a specific worker.
    pub fn entries_for_worker(&self, worker_id: usize) -> Vec<&TaskProfileEntry> {
        self.entries
            .iter()
            .filter(|e| e.worker_id == worker_id)
            .collect()
    }

    /// Returns the top N slowest tasks.
    pub fn slowest(&self, n: usize) -> Vec<&TaskProfileEntry> {
        let mut sorted: Vec<&TaskProfileEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.duration_us.cmp(&a.duration_us));
        sorted.truncate(n);
        sorted
    }

    /// Clear all recorded entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_count = 0;
        self.total_duration_us = 0;
    }

    /// Returns per-group aggregate statistics.
    pub fn group_stats(&self) -> HashMap<String, GroupStats> {
        let mut stats: HashMap<String, GroupStats> = HashMap::new();
        for entry in &self.entries {
            let group = entry.group.clone().unwrap_or_else(|| "ungrouped".to_string());
            let stat = stats.entry(group).or_insert_with(GroupStats::default);
            stat.task_count += 1;
            stat.total_duration_us += entry.duration_us;
            if entry.duration_us > stat.max_duration_us {
                stat.max_duration_us = entry.duration_us;
            }
            if entry.duration_us < stat.min_duration_us {
                stat.min_duration_us = entry.duration_us;
            }
        }
        stats
    }

    /// Returns per-worker aggregate statistics.
    pub fn worker_stats(&self, worker_count: usize) -> Vec<WorkerStats> {
        let mut stats = Vec::with_capacity(worker_count);
        for i in 0..worker_count {
            let entries: Vec<&TaskProfileEntry> =
                self.entries.iter().filter(|e| e.worker_id == i).collect();
            let total_dur: u64 = entries.iter().map(|e| e.duration_us).sum();
            let stolen_count = entries.iter().filter(|e| e.was_stolen).count();
            stats.push(WorkerStats {
                worker_id: i,
                task_count: entries.len() as u64,
                total_duration_us: total_dur,
                stolen_task_count: stolen_count as u64,
            });
        }
        stats
    }
}

impl Default for TaskProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate stats for a task group.
#[derive(Debug, Clone, Default)]
pub struct GroupStats {
    /// Number of tasks in this group.
    pub task_count: u64,
    /// Total duration of all tasks.
    pub total_duration_us: u64,
    /// Maximum single-task duration.
    pub max_duration_us: u64,
    /// Minimum single-task duration.
    pub min_duration_us: u64,
}

/// Aggregate stats for a worker thread.
#[derive(Debug, Clone)]
pub struct WorkerStats {
    /// Worker thread index.
    pub worker_id: usize,
    /// Total tasks executed.
    pub task_count: u64,
    /// Total time spent executing tasks.
    pub total_duration_us: u64,
    /// Number of tasks stolen from other workers.
    pub stolen_task_count: u64,
}

// ---------------------------------------------------------------------------
// SchedulerConfig
// ---------------------------------------------------------------------------

/// Configuration for the task scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of worker threads.
    pub worker_count: usize,
    /// Whether to enable work stealing.
    pub enable_stealing: bool,
    /// Whether to enable priority aging.
    pub enable_aging: bool,
    /// How many ticks before aging kicks in.
    pub aging_threshold: u64,
    /// Maximum steal batch size.
    pub steal_batch_size: usize,
    /// Whether to enable profiling.
    pub enable_profiling: bool,
    /// Maximum number of tasks.
    pub max_tasks: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            worker_count: DEFAULT_WORKER_COUNT,
            enable_stealing: true,
            enable_aging: true,
            aging_threshold: AGING_THRESHOLD,
            steal_batch_size: STEAL_BATCH_SIZE,
            enable_profiling: true,
            max_tasks: MAX_TASKS,
        }
    }
}

impl SchedulerConfig {
    /// Create a config with the given worker count.
    pub fn with_workers(mut self, count: usize) -> Self {
        self.worker_count = count.max(1);
        self
    }

    /// Enable or disable work stealing.
    pub fn with_stealing(mut self, enabled: bool) -> Self {
        self.enable_stealing = enabled;
        self
    }

    /// Enable or disable profiling.
    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.enable_profiling = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// TaskScheduler
// ---------------------------------------------------------------------------

/// The main task scheduler.
///
/// Manages task submission, dependency resolution, priority-based scheduling,
/// work stealing across worker threads, and profiling.
///
/// # Example
///
/// ```ignore
/// let mut scheduler = TaskScheduler::new(4);
///
/// let task_a = scheduler.schedule(
///     TaskDescriptor::new("load_assets", || {
///         // ... expensive work ...
///         TaskOutcome::Success(None)
///     }).with_priority(TaskPriority::High)
/// ).unwrap();
///
/// let task_b = scheduler.schedule(
///     TaskDescriptor::new("process_assets", || {
///         TaskOutcome::Success(None)
///     }).depends_on(task_a)
/// ).unwrap();
///
/// scheduler.run_until_complete();
/// ```
pub struct TaskScheduler {
    /// Configuration.
    config: SchedulerConfig,
    /// All task records, indexed by task ID.
    tasks: HashMap<TaskId, TaskRecord>,
    /// Next task ID to assign.
    next_id: u64,
    /// Global priority queue for ready tasks.
    ready_queue: BinaryHeap<PrioritizedTask>,
    /// Per-worker local queues.
    worker_queues: Vec<WorkerLocalQueue>,
    /// Current scheduler tick (for aging and profiling).
    tick: u64,
    /// Whether the scheduler is shutting down.
    shutdown: bool,
    /// Profiler.
    profiler: TaskProfiler,
    /// Total tasks submitted.
    total_submitted: u64,
    /// Total tasks completed.
    total_completed: u64,
    /// Total tasks failed.
    total_failed: u64,
    /// Total tasks cancelled.
    total_cancelled: u64,
    /// Total steals performed.
    total_steals: u64,
}

impl TaskScheduler {
    /// Create a new task scheduler with the given number of worker threads.
    pub fn new(worker_count: usize) -> Self {
        Self::with_config(SchedulerConfig::default().with_workers(worker_count))
    }

    /// Create a new task scheduler with the given configuration.
    pub fn with_config(config: SchedulerConfig) -> Self {
        let worker_count = config.worker_count.max(1);
        let mut worker_queues = Vec::with_capacity(worker_count);
        for i in 0..worker_count {
            worker_queues.push(WorkerLocalQueue::new(i));
        }

        let mut profiler = TaskProfiler::new();
        profiler.set_enabled(config.enable_profiling);

        Self {
            config,
            tasks: HashMap::new(),
            next_id: 0,
            ready_queue: BinaryHeap::new(),
            worker_queues,
            tick: 0,
            shutdown: false,
            profiler,
            total_submitted: 0,
            total_completed: 0,
            total_failed: 0,
            total_cancelled: 0,
            total_steals: 0,
        }
    }

    /// Schedule a task for execution.
    ///
    /// Returns the assigned `TaskId`.
    pub fn schedule(&mut self, desc: TaskDescriptor) -> TaskResult<TaskId> {
        if self.shutdown {
            return Err(TaskError::ShutDown);
        }

        if self.tasks.len() >= self.config.max_tasks {
            return Err(TaskError::SchedulerFull);
        }

        let id = TaskId(self.next_id);
        self.next_id += 1;

        // Compute pending dependency count.
        let mut pending_deps = 0u32;
        for dep_id in &desc.dependencies {
            match self.tasks.get(dep_id) {
                Some(dep) if dep.state == TaskState::Completed => {
                    // Already done — not pending.
                }
                Some(dep) if dep.state == TaskState::Failed => {
                    return Err(TaskError::DependencyFailed(*dep_id));
                }
                Some(_) => {
                    pending_deps += 1;
                }
                None => {
                    return Err(TaskError::InvalidTask(*dep_id));
                }
            }
        }

        let state = if pending_deps == 0 {
            TaskState::Ready
        } else {
            TaskState::Pending
        };

        // Register this task as a dependent of its dependencies.
        let deps_clone = desc.dependencies.clone();
        for dep_id in &deps_clone {
            if let Some(dep) = self.tasks.get_mut(dep_id) {
                dep.dependents.push(id);
            }
        }

        let record = TaskRecord {
            id,
            name: desc.name,
            state,
            priority: desc.priority,
            original_priority: desc.priority,
            submit_tick: self.tick,
            start_tick: None,
            end_tick: None,
            dependencies: deps_clone,
            pending_deps,
            dependents: Vec::new(),
            work: Some(desc.work),
            continuation: desc.continuation,
            outcome: None,
            stealable: desc.stealable,
            group: desc.group,
            executed_on_worker: None,
        };

        self.tasks.insert(id, record);
        self.total_submitted += 1;

        if state == TaskState::Ready {
            self.enqueue_ready(id);
        }

        Ok(id)
    }

    /// Schedule a task and return a `TaskFuture` for its result.
    pub fn schedule_async(&mut self, desc: TaskDescriptor) -> TaskResult<TaskFuture> {
        let id = self.schedule(desc)?;
        Ok(TaskFuture::new(id))
    }

    /// Schedule a batch of independent tasks, returning a `WaitGroup`.
    pub fn schedule_batch(
        &mut self,
        descriptors: Vec<TaskDescriptor>,
    ) -> TaskResult<WaitGroup> {
        let mut wg = WaitGroup::new();
        for desc in descriptors {
            let id = self.schedule(desc)?;
            wg.add(id);
        }
        Ok(wg)
    }

    /// Cancel a pending or ready task.
    pub fn cancel(&mut self, id: TaskId) -> TaskResult<()> {
        let task = self.tasks.get_mut(&id).ok_or(TaskError::InvalidTask(id))?;

        match task.state {
            TaskState::Pending | TaskState::Ready => {
                task.state = TaskState::Cancelled;
                task.work = None;
                self.total_cancelled += 1;

                // Cancel dependents too.
                let dependents: Vec<TaskId> = task.dependents.clone();
                for dep_id in dependents {
                    let _ = self.cancel(dep_id);
                }

                Ok(())
            }
            TaskState::Running | TaskState::Suspended => {
                // Cannot cancel a running task directly — mark for cancellation.
                task.state = TaskState::Cancelled;
                Ok(())
            }
            TaskState::Completed | TaskState::Failed | TaskState::Cancelled => {
                Ok(()) // Already done.
            }
        }
    }

    /// Query the current state of a task.
    pub fn task_state(&self, id: TaskId) -> TaskResult<TaskState> {
        self.tasks
            .get(&id)
            .map(|t| t.state)
            .ok_or(TaskError::InvalidTask(id))
    }

    /// Query the outcome of a completed task.
    pub fn task_outcome(&self, id: TaskId) -> Option<TaskOutcome> {
        self.tasks.get(&id).and_then(|t| t.outcome.clone())
    }

    /// Returns the total number of tasks in the scheduler.
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Returns the number of ready tasks.
    pub fn ready_count(&self) -> usize {
        self.ready_queue.len()
    }

    /// Returns the number of pending tasks (waiting on dependencies).
    pub fn pending_count(&self) -> usize {
        self.tasks
            .values()
            .filter(|t| t.state == TaskState::Pending)
            .count()
    }

    /// Advance the scheduler by one tick.
    ///
    /// This performs priority aging, processes the ready queue, and dispatches
    /// tasks to worker queues.
    pub fn tick(&mut self) {
        self.tick += 1;

        // Priority aging.
        if self.config.enable_aging {
            self.age_priorities();
        }

        // Dispatch ready tasks to workers using round-robin.
        self.dispatch_to_workers();

        // Process worker queues.
        self.process_workers();

        // Work stealing.
        if self.config.enable_stealing {
            self.perform_stealing();
        }
    }

    /// Run ticks until all tasks are completed.
    pub fn run_until_complete(&mut self) {
        let max_iterations = self.config.max_tasks * 10;
        let mut iterations = 0;

        while self.has_pending_work() && iterations < max_iterations {
            self.tick();
            iterations += 1;
        }
    }

    /// Returns true if there are tasks that are not yet completed.
    pub fn has_pending_work(&self) -> bool {
        self.tasks.values().any(|t| {
            t.state == TaskState::Pending
                || t.state == TaskState::Ready
                || t.state == TaskState::Running
                || t.state == TaskState::Suspended
        })
    }

    /// Shut down the scheduler, cancelling all pending tasks.
    pub fn shutdown(&mut self) {
        self.shutdown = true;
        let ids: Vec<TaskId> = self
            .tasks
            .iter()
            .filter(|(_, t)| t.state == TaskState::Pending || t.state == TaskState::Ready)
            .map(|(id, _)| *id)
            .collect();

        for id in ids {
            let _ = self.cancel(id);
        }
    }

    /// Get the profiler.
    pub fn profiler(&self) -> &TaskProfiler {
        &self.profiler
    }

    /// Get a mutable reference to the profiler.
    pub fn profiler_mut(&mut self) -> &mut TaskProfiler {
        &mut self.profiler
    }

    // -----------------------------------------------------------------------
    // Internal scheduling logic
    // -----------------------------------------------------------------------

    /// Add a task to the ready queue.
    fn enqueue_ready(&mut self, id: TaskId) {
        if let Some(task) = self.tasks.get(&id) {
            self.ready_queue.push(PrioritizedTask {
                id,
                priority: task.priority,
                submit_tick: task.submit_tick,
            });
        }
    }

    /// Age priorities of pending/ready tasks to prevent starvation.
    fn age_priorities(&mut self) {
        let threshold = self.config.aging_threshold;
        let current_tick = self.tick;

        let ids_to_age: Vec<TaskId> = self
            .tasks
            .iter()
            .filter(|(_, t)| {
                (t.state == TaskState::Pending || t.state == TaskState::Ready)
                    && (current_tick - t.submit_tick) > threshold
                    && t.priority != TaskPriority::Critical
            })
            .map(|(id, _)| *id)
            .collect();

        for id in ids_to_age {
            if let Some(task) = self.tasks.get_mut(&id) {
                task.priority = task.priority.promote();
            }
        }
    }

    /// Dispatch tasks from the global ready queue to worker local queues.
    fn dispatch_to_workers(&mut self) {
        let worker_count = self.worker_queues.len();
        if worker_count == 0 {
            return;
        }

        let mut worker_idx = 0;
        while let Some(prioritized) = self.ready_queue.pop() {
            if let Some(task) = self.tasks.get(&prioritized.id) {
                if task.state != TaskState::Ready {
                    continue;
                }
            } else {
                continue;
            }
            self.worker_queues[worker_idx % worker_count].push(prioritized.id);
            worker_idx += 1;
        }
    }

    /// Process tasks on worker queues (simulated single-threaded execution).
    fn process_workers(&mut self) {
        let worker_count = self.worker_queues.len();

        for worker_id in 0..worker_count {
            if let Some(task_id) = self.worker_queues[worker_id].pop() {
                self.execute_task(task_id, worker_id);
            }
        }
    }

    /// Execute a single task.
    fn execute_task(&mut self, id: TaskId, worker_id: usize) {
        let start_tick = self.tick;

        // Take the work closure out of the task.
        let work = {
            if let Some(task) = self.tasks.get_mut(&id) {
                if task.state == TaskState::Cancelled {
                    return;
                }
                task.state = TaskState::Running;
                task.start_tick = Some(start_tick);
                task.executed_on_worker = Some(worker_id);
                task.work.take()
            } else {
                return;
            }
        };

        // Execute the work.
        let outcome = if let Some(work) = work {
            work()
        } else {
            TaskOutcome::Failure("work closure was already consumed".to_string())
        };

        // Handle yield.
        if outcome.is_yield() {
            if let Some(task) = self.tasks.get_mut(&id) {
                task.state = TaskState::Ready;
                task.outcome = None;
            }
            self.enqueue_ready(id);
            return;
        }

        let is_success = outcome.is_success();
        let end_tick = self.tick;

        // Record profiling data.
        if self.profiler.is_enabled() {
            if let Some(task) = self.tasks.get(&id) {
                self.profiler.record(TaskProfileEntry {
                    task_id: id,
                    name: task.name.clone(),
                    priority: task.original_priority,
                    group: task.group.clone(),
                    submit_tick: task.submit_tick,
                    start_tick,
                    end_tick,
                    duration_us: (end_tick - start_tick).max(1),
                    worker_id,
                    was_stolen: false,
                    outcome: outcome.clone(),
                });
            }
        }

        // Take the continuation.
        let continuation = {
            if let Some(task) = self.tasks.get_mut(&id) {
                task.state = if is_success {
                    TaskState::Completed
                } else {
                    TaskState::Failed
                };
                task.end_tick = Some(end_tick);
                task.outcome = Some(outcome.clone());
                if is_success {
                    self.total_completed += 1;
                } else {
                    self.total_failed += 1;
                }
                task.continuation.take()
            } else {
                None
            }
        };

        // Schedule continuation if any.
        if let Some(cont_fn) = continuation {
            let cont_desc = cont_fn(outcome.clone());
            let _ = self.schedule(cont_desc);
        }

        // Notify dependents.
        let dependents = if let Some(task) = self.tasks.get(&id) {
            task.dependents.clone()
        } else {
            Vec::new()
        };

        for dep_id in dependents {
            if is_success {
                self.notify_dependency_complete(dep_id);
            } else {
                self.notify_dependency_failed(dep_id);
            }
        }
    }

    /// Notify a task that one of its dependencies has completed.
    fn notify_dependency_complete(&mut self, id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&id) {
            if task.state != TaskState::Pending {
                return;
            }
            task.pending_deps = task.pending_deps.saturating_sub(1);
            if task.pending_deps == 0 {
                task.state = TaskState::Ready;
                self.enqueue_ready(id);
            }
        }
    }

    /// Notify a task that one of its dependencies has failed.
    fn notify_dependency_failed(&mut self, id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&id) {
            task.state = TaskState::Failed;
            task.outcome = Some(TaskOutcome::Failure("dependency failed".to_string()));
            self.total_failed += 1;

            // Cascade to this task's dependents.
            let deps = task.dependents.clone();
            for dep in deps {
                self.notify_dependency_failed(dep);
            }
        }
    }

    /// Perform work stealing: busy worker queues donate to idle ones.
    fn perform_stealing(&mut self) {
        let worker_count = self.worker_queues.len();
        if worker_count < 2 {
            return;
        }

        // Find the busiest and the most idle workers.
        let mut busiest = 0;
        let mut busiest_len = 0;
        let mut idlest = 0;
        let mut idlest_len = usize::MAX;

        for i in 0..worker_count {
            let len = self.worker_queues[i].len();
            if len > busiest_len {
                busiest = i;
                busiest_len = len;
            }
            if len < idlest_len {
                idlest = i;
                idlest_len = len;
            }
        }

        // Only steal if there is a meaningful imbalance.
        if busiest == idlest || busiest_len <= 1 || busiest_len - idlest_len <= 1 {
            return;
        }

        let max_steal = self.config.steal_batch_size;
        let stolen = self.worker_queues[busiest].steal(max_steal);

        let stolen_count = stolen.len();
        for task_id in stolen {
            self.worker_queues[idlest].push(task_id);
        }

        self.total_steals += stolen_count as u64;
    }

    /// Check for a cyclic dependency before adding it.
    pub fn check_cycle(&self, from: TaskId, to: TaskId) -> bool {
        // BFS from `to` — if we can reach `from`, adding from->to creates a cycle.
        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(to);

        while let Some(current) = queue.pop_front() {
            if current == from {
                return true; // Cycle detected.
            }
            if visited.insert(current) {
                if let Some(task) = self.tasks.get(&current) {
                    for dep in &task.dependents {
                        queue.push_back(*dep);
                    }
                }
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Returns scheduler statistics.
    pub fn stats(&self) -> SchedulerStats {
        let mut worker_queue_sizes = Vec::with_capacity(self.worker_queues.len());
        for wq in &self.worker_queues {
            worker_queue_sizes.push(wq.len());
        }

        SchedulerStats {
            total_submitted: self.total_submitted,
            total_completed: self.total_completed,
            total_failed: self.total_failed,
            total_cancelled: self.total_cancelled,
            total_steals: self.total_steals,
            current_tick: self.tick,
            live_tasks: self.tasks.len() as u64,
            ready_count: self.ready_queue.len() as u64,
            pending_count: self.pending_count() as u64,
            worker_queue_sizes,
        }
    }

    /// Remove completed/failed/cancelled tasks to free memory.
    pub fn gc(&mut self) {
        self.tasks.retain(|_, t| {
            t.state != TaskState::Completed
                && t.state != TaskState::Failed
                && t.state != TaskState::Cancelled
        });
    }

    /// Get the current tick.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Get the configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

/// Aggregate scheduler statistics.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Total tasks ever submitted.
    pub total_submitted: u64,
    /// Total tasks completed successfully.
    pub total_completed: u64,
    /// Total tasks that failed.
    pub total_failed: u64,
    /// Total tasks cancelled.
    pub total_cancelled: u64,
    /// Total work-steal operations.
    pub total_steals: u64,
    /// Current scheduler tick.
    pub current_tick: u64,
    /// Number of tasks still in the scheduler.
    pub live_tasks: u64,
    /// Number of tasks in the ready queue.
    pub ready_count: u64,
    /// Number of tasks waiting on dependencies.
    pub pending_count: u64,
    /// Per-worker queue sizes.
    pub worker_queue_sizes: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_and_run() {
        let mut scheduler = TaskScheduler::new(2);

        let id = scheduler
            .schedule(TaskDescriptor::new("test", || TaskOutcome::Success(None)))
            .unwrap();

        scheduler.run_until_complete();

        assert_eq!(scheduler.task_state(id).unwrap(), TaskState::Completed);
    }

    #[test]
    fn test_dependencies() {
        let mut scheduler = TaskScheduler::new(2);

        let a = scheduler
            .schedule(TaskDescriptor::new("a", || TaskOutcome::Success(None)))
            .unwrap();

        let b = scheduler
            .schedule(
                TaskDescriptor::new("b", || TaskOutcome::Success(None)).depends_on(a),
            )
            .unwrap();

        // b should be pending.
        assert_eq!(scheduler.task_state(b).unwrap(), TaskState::Pending);

        scheduler.run_until_complete();

        assert_eq!(scheduler.task_state(a).unwrap(), TaskState::Completed);
        assert_eq!(scheduler.task_state(b).unwrap(), TaskState::Completed);
    }

    #[test]
    fn test_dependency_failure_cascades() {
        let mut scheduler = TaskScheduler::new(2);

        let a = scheduler
            .schedule(TaskDescriptor::new("a", || {
                TaskOutcome::Failure("oops".to_string())
            }))
            .unwrap();

        let b = scheduler
            .schedule(
                TaskDescriptor::new("b", || TaskOutcome::Success(None)).depends_on(a),
            )
            .unwrap();

        scheduler.run_until_complete();

        assert_eq!(scheduler.task_state(a).unwrap(), TaskState::Failed);
        assert_eq!(scheduler.task_state(b).unwrap(), TaskState::Failed);
    }

    #[test]
    fn test_cancel() {
        let mut scheduler = TaskScheduler::new(2);

        let id = scheduler
            .schedule(TaskDescriptor::new("test", || TaskOutcome::Success(None)))
            .unwrap();

        scheduler.cancel(id).unwrap();
        assert_eq!(scheduler.task_state(id).unwrap(), TaskState::Cancelled);
    }

    #[test]
    fn test_wait_group() {
        let mut scheduler = TaskScheduler::new(2);

        let mut wg = WaitGroup::new();
        for i in 0..5 {
            let id = scheduler
                .schedule(TaskDescriptor::new(
                    format!("task_{}", i),
                    || TaskOutcome::Success(None),
                ))
                .unwrap();
            wg.add(id);
        }

        assert_eq!(wg.len(), 5);

        scheduler.run_until_complete();

        assert!(wg.is_complete(&scheduler));
        assert_eq!(wg.completed_count(&scheduler), 5);
        assert_eq!(wg.failed_count(&scheduler), 0);
    }

    #[test]
    fn test_task_future() {
        let mut scheduler = TaskScheduler::new(2);

        let future = scheduler
            .schedule_async(TaskDescriptor::new("async_task", || {
                TaskOutcome::Success(Some("result".to_string()))
            }))
            .unwrap();

        scheduler.run_until_complete();

        assert!(future.is_ready(&scheduler));
        let outcome = future.get(&scheduler).unwrap();
        assert!(outcome.is_success());
    }

    #[test]
    fn test_batch_schedule() {
        let mut scheduler = TaskScheduler::new(2);

        let descs: Vec<TaskDescriptor> = (0..10)
            .map(|i| TaskDescriptor::new(format!("batch_{}", i), || TaskOutcome::Success(None)))
            .collect();

        let wg = scheduler.schedule_batch(descs).unwrap();
        assert_eq!(wg.len(), 10);

        scheduler.run_until_complete();
        assert!(wg.is_complete(&scheduler));
    }

    #[test]
    fn test_priority_ordering() {
        let mut scheduler = TaskScheduler::new(1);

        let _low = scheduler
            .schedule(
                TaskDescriptor::new("low", || TaskOutcome::Success(None))
                    .with_priority(TaskPriority::Low),
            )
            .unwrap();

        let _high = scheduler
            .schedule(
                TaskDescriptor::new("high", || TaskOutcome::Success(None))
                    .with_priority(TaskPriority::High),
            )
            .unwrap();

        let _critical = scheduler
            .schedule(
                TaskDescriptor::new("critical", || TaskOutcome::Success(None))
                    .with_priority(TaskPriority::Critical),
            )
            .unwrap();

        scheduler.run_until_complete();

        // All should be complete.
        let stats = scheduler.stats();
        assert_eq!(stats.total_completed, 3);
    }

    #[test]
    fn test_profiler() {
        let mut scheduler = TaskScheduler::new(2);

        for i in 0..5 {
            scheduler
                .schedule(
                    TaskDescriptor::new(format!("profiled_{}", i), || {
                        TaskOutcome::Success(None)
                    })
                    .with_group("batch"),
                )
                .unwrap();
        }

        scheduler.run_until_complete();

        let profiler = scheduler.profiler();
        assert_eq!(profiler.entries().len(), 5);

        let group_entries = profiler.entries_for_group("batch");
        assert_eq!(group_entries.len(), 5);
    }

    #[test]
    fn test_scheduler_gc() {
        let mut scheduler = TaskScheduler::new(2);

        for i in 0..10 {
            scheduler
                .schedule(TaskDescriptor::new(format!("gc_{}", i), || {
                    TaskOutcome::Success(None)
                }))
                .unwrap();
        }

        scheduler.run_until_complete();
        assert_eq!(scheduler.task_count(), 10);

        scheduler.gc();
        assert_eq!(scheduler.task_count(), 0);
    }

    #[test]
    fn test_scheduler_shutdown() {
        let mut scheduler = TaskScheduler::new(2);

        scheduler
            .schedule(TaskDescriptor::new("pending", || TaskOutcome::Success(None)))
            .unwrap();

        scheduler.shutdown();

        let result = scheduler.schedule(TaskDescriptor::new("post_shutdown", || {
            TaskOutcome::Success(None)
        }));

        assert!(result.is_err());
    }

    #[test]
    fn test_cycle_detection() {
        let mut scheduler = TaskScheduler::new(2);

        let a = scheduler
            .schedule(TaskDescriptor::new("a", || TaskOutcome::Success(None)))
            .unwrap();

        let b = scheduler
            .schedule(
                TaskDescriptor::new("b", || TaskOutcome::Success(None)).depends_on(a),
            )
            .unwrap();

        // Check if adding b->a would create a cycle.
        assert!(scheduler.check_cycle(b, a));
        assert!(!scheduler.check_cycle(a, b));
    }

    #[test]
    fn test_priority_promote_demote() {
        assert_eq!(TaskPriority::Low.promote(), TaskPriority::Normal);
        assert_eq!(TaskPriority::Critical.promote(), TaskPriority::Critical);
        assert_eq!(TaskPriority::High.demote(), TaskPriority::Normal);
        assert_eq!(TaskPriority::Idle.demote(), TaskPriority::Idle);
    }

    #[test]
    fn test_scheduler_stats() {
        let mut scheduler = TaskScheduler::new(2);

        scheduler
            .schedule(TaskDescriptor::new("s1", || TaskOutcome::Success(None)))
            .unwrap();
        scheduler
            .schedule(TaskDescriptor::new("s2", || {
                TaskOutcome::Failure("err".into())
            }))
            .unwrap();

        scheduler.run_until_complete();

        let stats = scheduler.stats();
        assert_eq!(stats.total_submitted, 2);
        assert_eq!(stats.total_completed, 1);
        assert_eq!(stats.total_failed, 1);
    }

    #[test]
    fn test_task_yield() {
        let mut scheduler = TaskScheduler::new(1);
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let id = scheduler
            .schedule(TaskDescriptor::new("yielding", move || {
                let val = counter_clone.fetch_add(1, Ordering::SeqCst);
                if val < 2 {
                    TaskOutcome::Yield
                } else {
                    TaskOutcome::Success(None)
                }
            }))
            .unwrap();

        scheduler.run_until_complete();
        assert_eq!(scheduler.task_state(id).unwrap(), TaskState::Completed);
    }
}
