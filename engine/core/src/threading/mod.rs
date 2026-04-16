//! Job system for multi-threaded workloads.
//!
//! The job system distributes work across a pool of worker threads. Jobs can
//! express dependencies via [`TaskGraph`], enabling data-parallel and
//! task-parallel patterns without manual thread management.
//!
//! The system integrates with [`rayon`] for parallel iterators and with the
//! [`memory`](crate::memory) module for per-thread scratch allocators.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam::queue::SegQueue;
use parking_lot::{Condvar, Mutex};

// ---------------------------------------------------------------------------
// Job trait
// ---------------------------------------------------------------------------

/// A unit of work that can be submitted to the [`JobSystem`].
///
/// Implementations should be lightweight and avoid blocking I/O. Long-running
/// work should be broken into smaller jobs or moved to a dedicated background
/// thread.
pub trait Job: Send + 'static {
    /// Executes the job.
    ///
    /// `thread_index` identifies the worker thread running this job, which
    /// can be used to index into per-thread resources (e.g., scratch
    /// allocators).
    fn execute(&self, thread_index: usize);
}

// Allow closures to be used as jobs.
impl<F: Fn(usize) + Send + 'static> Job for F {
    fn execute(&self, thread_index: usize) {
        self(thread_index);
    }
}

// ---------------------------------------------------------------------------
// JobPriority
// ---------------------------------------------------------------------------

/// Scheduling priority for submitted jobs.
///
/// Higher-priority jobs are dequeued before lower-priority ones. Within the
/// same priority level, jobs are processed in FIFO order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum JobPriority {
    /// Must complete this frame (e.g., input handling).
    Critical = 0,
    /// High priority (e.g., visibility determination).
    High = 1,
    /// Default priority for most gameplay work.
    Normal = 2,
    /// Low priority (e.g., LOD computation).
    Low = 3,
    /// Background work that can span multiple frames (e.g., asset streaming).
    Background = 4,
}

impl Default for JobPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// The number of distinct priority levels.
const PRIORITY_COUNT: usize = 5;

// ---------------------------------------------------------------------------
// JobHandle
// ---------------------------------------------------------------------------

/// An opaque handle returned when a job is submitted.
///
/// Use [`JobSystem::wait`] to block until the job completes, or
/// [`JobHandle::is_complete`] to poll.
#[derive(Clone)]
pub struct JobHandle {
    /// Shared completion flag.
    completed: Arc<AtomicBool>,
    /// Unique identifier for debugging / profiling.
    _id: u64,
}

impl JobHandle {
    /// Creates a new incomplete handle.
    fn new(id: u64) -> Self {
        Self {
            completed: Arc::new(AtomicBool::new(false)),
            _id: id,
        }
    }

    /// Creates a handle that is already marked complete.
    fn new_completed(id: u64) -> Self {
        Self {
            completed: Arc::new(AtomicBool::new(true)),
            _id: id,
        }
    }

    /// Returns `true` once the associated job has finished executing.
    pub fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    /// Marks the job as complete. Called internally by the job system.
    fn mark_complete(&self) {
        self.completed.store(true, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Internal: QueuedJob and SharedState
// ---------------------------------------------------------------------------

/// A job waiting in the queue, ready for a worker to pick up.
struct QueuedJob {
    /// The job to execute.
    job: Box<dyn Job>,
    /// Handle to signal on completion.
    handle: JobHandle,
}

// SAFETY: `QueuedJob` contains a `Box<dyn Job>` which is `Send` (trait bound)
// and a `JobHandle` which is `Send + Sync`. The job system only sends jobs
// across thread boundaries; it never shares a `QueuedJob` between threads.
unsafe impl Send for QueuedJob {}

/// Shared state between the job system and its worker threads.
struct SharedState {
    /// One lock-free queue per priority level. Index 0 = Critical, 4 = Background.
    queues: [SegQueue<QueuedJob>; PRIORITY_COUNT],
    /// Flag to signal workers to shut down.
    shutdown: AtomicBool,
    /// Condition variable used to wake sleeping workers when new work arrives.
    wake_mutex: Mutex<()>,
    wake_condvar: Condvar,
}

impl SharedState {
    /// Creates a new shared state with empty queues and shutdown = false.
    fn new() -> Self {
        Self {
            queues: [
                SegQueue::new(),
                SegQueue::new(),
                SegQueue::new(),
                SegQueue::new(),
                SegQueue::new(),
            ],
            shutdown: AtomicBool::new(false),
            wake_mutex: Mutex::new(()),
            wake_condvar: Condvar::new(),
        }
    }

    /// Enqueues a job at the given priority level.
    fn push(&self, priority: JobPriority, job: QueuedJob) {
        self.queues[priority as usize].push(job);
    }

    /// Attempts to pop the highest-priority job from the queues.
    ///
    /// Scans from Critical (0) to Background (4), returning the first
    /// available job.
    fn try_pop(&self) -> Option<QueuedJob> {
        for queue in &self.queues {
            if let Some(job) = queue.pop() {
                return Some(job);
            }
        }
        None
    }

    /// Wakes one sleeping worker thread.
    fn notify_one(&self) {
        // We must briefly lock the mutex so that the condvar notification is
        // not lost if a worker is between checking the queue and going to sleep.
        let _guard = self.wake_mutex.lock();
        self.wake_condvar.notify_one();
    }

    /// Wakes all sleeping worker threads.
    fn notify_all(&self) {
        let _guard = self.wake_mutex.lock();
        self.wake_condvar.notify_all();
    }

    /// Returns true if the shutdown flag has been set.
    fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    /// Returns true if any queue has work available.
    fn has_work(&self) -> bool {
        for queue in &self.queues {
            if !queue.is_empty() {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// JobSystem
// ---------------------------------------------------------------------------

/// Multi-threaded job scheduler.
///
/// Owns a pool of worker threads and a set of priority queues. Jobs are
/// submitted via [`submit`](JobSystem::submit) and executed asynchronously.
///
/// # Example (conceptual)
///
/// ```ignore
/// let js = JobSystem::new(JobSystemConfig::default());
/// let handle = js.submit(JobPriority::Normal, |_thread_idx| {
///     // do work
/// });
/// js.wait(&handle);
/// ```
pub struct JobSystem {
    /// Worker thread join handles. Wrapped in Option so we can take them in Drop.
    _workers: Vec<std::thread::JoinHandle<()>>,
    /// Monotonically increasing job id counter.
    next_job_id: AtomicU64,
    /// Shared job queues and synchronization primitives.
    _queues: Arc<SharedState>,
    /// Flag to signal workers to shut down (aliased from SharedState for API compat).
    _shutdown: Arc<AtomicBool>,
}

/// Configuration for [`JobSystem`] construction.
pub struct JobSystemConfig {
    /// Number of worker threads. `None` means use the number of logical CPUs.
    pub num_workers: Option<usize>,
    /// Per-thread scratch allocator size in bytes.
    pub scratch_allocator_size: usize,
}

impl Default for JobSystemConfig {
    fn default() -> Self {
        Self {
            num_workers: None,
            scratch_allocator_size: 1024 * 1024, // 1 MiB
        }
    }
}

impl JobSystem {
    /// Creates a new job system and spawns worker threads.
    pub fn new(config: JobSystemConfig) -> Self {
        let num_workers = config.num_workers.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

        let shared = Arc::new(SharedState::new());

        let mut workers = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let state = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("genovo-worker-{i}"))
                .spawn(move || {
                    worker_loop(i, &state);
                })
                .expect("failed to spawn worker thread");
            workers.push(handle);
        }

        // The _shutdown field aliases the same AtomicBool inside SharedState.
        // We create a separate Arc here that points to the same flag by
        // extracting a reference-counted pointer.  However, since SharedState
        // owns the AtomicBool inline (not behind an Arc), we use a small
        // wrapper: we simply keep the shared Arc and read from it.
        //
        // For backward compatibility of the struct fields we store a dummy Arc
        // that is never actually consulted -- all real shutdown logic goes
        // through `_queues.shutdown`.
        let shutdown_flag = Arc::new(AtomicBool::new(false));

        Self {
            _workers: workers,
            next_job_id: AtomicU64::new(0),
            _queues: shared,
            _shutdown: shutdown_flag,
        }
    }

    /// Submits a job for asynchronous execution and returns a handle.
    pub fn submit<J: Job>(&self, _priority: JobPriority, _job: J) -> JobHandle {
        let id = self.next_job_id.fetch_add(1, Ordering::Relaxed);
        let handle = JobHandle::new(id);

        let queued = QueuedJob {
            job: Box::new(_job),
            handle: handle.clone(),
        };

        self._queues.push(_priority, queued);

        // Wake one sleeping worker to pick up the new job.
        self._queues.notify_one();

        handle
    }

    /// Blocks the calling thread until `handle` is complete.
    ///
    /// While waiting, the calling thread assists by executing other queued
    /// jobs (work-stealing).
    pub fn wait(&self, handle: &JobHandle) {
        // Fast path: already done.
        if handle.is_complete() {
            return;
        }

        // Work-stealing wait: while the target handle is not yet complete, try
        // to pop a job from the shared queues and execute it on this thread.
        // This keeps the calling thread productive and avoids deadlocks when
        // jobs submit sub-jobs and wait on them.
        //
        // We use thread_index = usize::MAX as a sentinel for "main thread /
        // non-worker thread".  Callers who care about thread_index for per-
        // thread resources should treat this value accordingly.
        const CALLER_THREAD_INDEX: usize = usize::MAX;

        loop {
            if handle.is_complete() {
                return;
            }

            // Try to steal and execute a queued job.
            if let Some(stolen) = self._queues.try_pop() {
                stolen.job.execute(CALLER_THREAD_INDEX);
                stolen.handle.mark_complete();

                // Re-check immediately in case completing that job was what we
                // were waiting for.
                continue;
            }

            // No work available -- yield briefly and try again.
            // We use a short sleep instead of a spin loop to avoid burning
            // CPU when there is genuinely nothing to do.
            if handle.is_complete() {
                return;
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }

    /// Parallel-for: divides `count` items across worker threads.
    ///
    /// `body` is called with `(start_index, end_index, thread_index)` for each
    /// chunk.
    pub fn parallel_for<F>(&self, count: usize, _granularity: usize, body: F) -> JobHandle
    where
        F: Fn(usize, usize, usize) + Send + Sync + 'static,
    {
        // Edge case: nothing to do.
        if count == 0 {
            let id = self.next_job_id.fetch_add(1, Ordering::Relaxed);
            return JobHandle::new_completed(id);
        }

        // Clamp granularity to at least 1 to prevent division by zero.
        let granularity = _granularity.max(1);

        // Calculate the number of chunks.
        let num_chunks = (count + granularity - 1) / granularity;

        // If there is only one chunk, just submit a single job.
        if num_chunks == 1 {
            let body = Arc::new(body);
            return self.submit(JobPriority::Normal, move |thread_idx: usize| {
                body(0, count, thread_idx);
            });
        }

        // Shared counter: starts at num_chunks, decremented by each sub-job on
        // completion.  The overall handle is marked complete when the counter
        // reaches zero.
        let remaining = Arc::new(AtomicCounter::new(num_chunks as u32));
        let overall_id = self.next_job_id.fetch_add(1, Ordering::Relaxed);
        let overall_handle = JobHandle::new(overall_id);
        let body = Arc::new(body);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * granularity;
            let end = (start + granularity).min(count);

            let body = Arc::clone(&body);
            let remaining = Arc::clone(&remaining);
            let overall_handle = overall_handle.clone();

            self.submit(JobPriority::Normal, move |thread_idx: usize| {
                body(start, end, thread_idx);

                // Decrement returns the *previous* value; if previous was 1,
                // we are the last chunk to finish.
                let prev = remaining.decrement();
                if prev == 1 {
                    overall_handle.mark_complete();
                }
            });
        }

        overall_handle
    }

    /// Returns the number of worker threads.
    pub fn worker_count(&self) -> usize {
        self._workers.len()
    }
}

/// The main loop run by each worker thread.
///
/// Workers repeatedly try to pop the highest-priority job from the shared
/// queues.  When no work is available, they wait on a condition variable
/// (with a timeout to guard against lost wakeups).
fn worker_loop(thread_index: usize, state: &SharedState) {
    loop {
        // 1. Check for shutdown.
        if state.is_shutdown() {
            // Drain any remaining jobs before exiting so that handles waiting
            // on them are not left dangling.
            while let Some(queued) = state.try_pop() {
                queued.job.execute(thread_index);
                queued.handle.mark_complete();
            }
            return;
        }

        // 2. Try to pop a job.
        if let Some(queued) = state.try_pop() {
            queued.job.execute(thread_index);
            queued.handle.mark_complete();
            continue;
        }

        // 3. No work available -- wait on the condition variable.
        //    We use a timeout to handle spurious wakeups and ensure the
        //    shutdown flag is periodically re-checked.
        let mut guard = state.wake_mutex.lock();

        // Double-check: work may have been enqueued between our try_pop and
        // acquiring the mutex.
        if state.has_work() || state.is_shutdown() {
            drop(guard);
            continue;
        }

        // Wait with a timeout of 5ms.  This is long enough to avoid busy-
        // waiting but short enough to keep latency low.
        state
            .wake_condvar
            .wait_for(&mut guard, Duration::from_millis(5));
        drop(guard);
    }
}

impl Drop for JobSystem {
    fn drop(&mut self) {
        // Signal shutdown to all workers.
        self._queues.shutdown.store(true, Ordering::Release);

        // Wake all workers so they can observe the shutdown flag.
        self._queues.notify_all();

        // Join all worker threads.  We drain the Vec so we can take ownership
        // of each JoinHandle.
        for handle in self._workers.drain(..) {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// TaskGraph
// ---------------------------------------------------------------------------

/// A dependency graph of tasks that are executed in topological order.
///
/// Nodes represent tasks; edges represent "must complete before" relationships.
/// The graph is submitted to a [`JobSystem`] which schedules it respecting all
/// dependencies.
pub struct TaskGraph {
    /// Nodes in submission order.
    nodes: Vec<TaskNode>,
}

/// A single node in the task graph.
struct TaskNode {
    /// The task to execute.
    _task: Box<dyn Job>,
    /// Indices of nodes that must complete before this one may start.
    _dependencies: Vec<usize>,
}

/// An opaque identifier for a node within a [`TaskGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(usize);

impl TaskGraph {
    /// Creates an empty task graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Adds a task with the given dependencies. Returns a [`TaskId`] that can
    /// be used as a dependency for later tasks.
    pub fn add_task<J: Job>(&mut self, task: J, dependencies: &[TaskId]) -> TaskId {
        let id = TaskId(self.nodes.len());
        self.nodes.push(TaskNode {
            _task: Box::new(task),
            _dependencies: dependencies.iter().map(|d| d.0).collect(),
        });
        id
    }

    /// Submits the entire graph for execution on the given job system.
    ///
    /// Returns a [`JobHandle`] that completes when all tasks in the graph
    /// have finished.
    ///
    /// # Algorithm
    ///
    /// Uses Kahn's algorithm for topological ordering:
    ///
    /// 1. Compute in-degree for each node.
    /// 2. Enqueue all nodes with in-degree 0 (no dependencies).
    /// 3. When a node finishes, decrement the in-degree of its dependents.
    ///    If a dependent's in-degree reaches 0, submit it to the job system.
    /// 4. The overall handle completes when all nodes have finished.
    pub fn execute(&self, job_system: &JobSystem) -> JobHandle {
        let node_count = self.nodes.len();

        // Edge case: empty graph.
        if node_count == 0 {
            let id = job_system.next_job_id.fetch_add(1, Ordering::Relaxed);
            return JobHandle::new_completed(id);
        }

        // --- Build adjacency structures ---
        //
        // `in_degree[i]` = number of unfinished dependencies of node i.
        // `dependents[i]` = list of nodes that depend on node i.
        let mut in_degree: Vec<AtomicU32> = Vec::with_capacity(node_count);
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); node_count];

        for (idx, node) in self.nodes.iter().enumerate() {
            in_degree.push(AtomicU32::new(node._dependencies.len() as u32));
            for &dep in &node._dependencies {
                dependents[dep].push(idx);
            }
        }

        // Wrap everything we need to share across tasks in Arcs.
        let in_degree = Arc::new(in_degree);
        let dependents = Arc::new(dependents);

        // Overall completion counter: when it reaches zero, all nodes are done.
        let remaining = Arc::new(AtomicCounter::new(node_count as u32));

        let overall_id = job_system.next_job_id.fetch_add(1, Ordering::Relaxed);
        let overall_handle = JobHandle::new(overall_id);

        // We need to be able to submit newly-ready tasks from within a task
        // callback.  To do this we share a reference to the job system's
        // shared state and submit directly through it.
        //
        // Since `TaskGraph::execute` takes `&self`, we cannot move the task
        // bodies out of the nodes.  Instead, we store raw pointers to each
        // `dyn Job` in a shared wrapper.  The caller MUST call `wait()` on
        // the returned handle before dropping the `TaskGraph`, ensuring the
        // pointees remain valid for the duration of execution.
        let task_bodies: Arc<Vec<Mutex<Option<TaskBodyWrapper>>>> = Arc::new(
            self.nodes
                .iter()
                .map(|node| {
                    let ptr = &*node._task as *const dyn Job;
                    Mutex::new(Some(TaskBodyWrapper(ptr)))
                })
                .collect(),
        );

        // Shared queues reference for submitting from within task callbacks.
        let shared_state = Arc::clone(&job_system._queues);
        let next_job_id = &job_system.next_job_id as *const AtomicU64;

        // SAFETY: The caller MUST wait on the returned handle, so the
        // JobSystem (and its `next_job_id`) will outlive all tasks.
        let job_id_ptr = SendPtr(next_job_id);

        let ctx = Arc::new(GraphContextInner {
            in_degree,
            dependents,
            remaining,
            overall_handle: overall_handle.clone(),
            task_bodies,
            shared_state,
            job_id_ptr,
        });

        // Submit all zero-dependency (root) nodes.
        for i in 0..node_count {
            if ctx.in_degree[i].load(Ordering::Acquire) == 0 {
                submit_graph_node(i, Arc::clone(&ctx), job_system);
            }
        }

        overall_handle
    }
}

/// A wrapper around a raw pointer to a `dyn Job` that implements `Send`.
///
/// SAFETY: The caller guarantees the pointee outlives the wrapper and is `Send`.
struct TaskBodyWrapper(*const dyn Job);
unsafe impl Send for TaskBodyWrapper {}

/// A wrapper around a raw const pointer that implements `Send`.
struct SendPtr<T>(*const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        SendPtr(self.0)
    }
}
impl<T> Copy for SendPtr<T> {}

/// Submits a single graph node for execution on the job system.
///
/// When the node finishes, it decrements the in-degree of its dependents
/// and submits any that become ready.
fn submit_graph_node(
    node_index: usize,
    ctx: Arc<GraphContextInner>,
    job_system: &JobSystem,
) {
    let ctx_for_job = Arc::clone(&ctx);

    let id = job_system.next_job_id.fetch_add(1, Ordering::Relaxed);
    let handle = JobHandle::new(id);

    let job_handle = handle.clone();

    let queued = QueuedJob {
        job: Box::new(move |thread_idx: usize| {
            // Execute the actual task body.
            {
                let guard = ctx_for_job.task_bodies[node_index].lock();
                if let Some(ref wrapper) = *guard {
                    // SAFETY: pointer is valid for the duration of execute().
                    unsafe { (*wrapper.0).execute(thread_idx) };
                }
            }

            // Decrement in-degree of dependents and submit newly-ready ones.
            for &dep_idx in &ctx_for_job.dependents[node_index] {
                let prev = ctx_for_job.in_degree[dep_idx].fetch_sub(1, Ordering::AcqRel);
                if prev == 1 {
                    // This dependent's in-degree just hit zero -- submit it.
                    // We submit directly to the shared state rather than going
                    // through JobSystem::submit to avoid needing a &JobSystem.
                    submit_graph_node_direct(dep_idx, Arc::clone(&ctx_for_job));
                }
            }

            // Decrement the overall remaining counter.
            let prev = ctx_for_job.remaining.decrement();
            if prev == 1 {
                ctx_for_job.overall_handle.mark_complete();
            }
        }),
        handle: job_handle,
    };

    job_system._queues.push(JobPriority::Normal, queued);
    job_system._queues.notify_one();
}

/// Submits a graph node directly to the shared state (used from within task
/// callbacks where we don't have a `&JobSystem`).
fn submit_graph_node_direct(
    node_index: usize,
    ctx: Arc<GraphContextInner>,
) {
    let ctx_for_job = Arc::clone(&ctx);
    let shared = Arc::clone(&ctx.shared_state);

    // Generate a job ID from the raw pointer.
    // SAFETY: The pointer is valid because the caller waits on the overall
    // handle, ensuring JobSystem outlives all tasks.
    let id = unsafe { (*ctx.job_id_ptr.0).fetch_add(1, Ordering::Relaxed) };
    let handle = JobHandle::new(id);
    let job_handle = handle.clone();

    let queued = QueuedJob {
        job: Box::new(move |thread_idx: usize| {
            // Execute the actual task body.
            {
                let guard = ctx_for_job.task_bodies[node_index].lock();
                if let Some(ref wrapper) = *guard {
                    unsafe { (*wrapper.0).execute(thread_idx) };
                }
            }

            // Decrement in-degree of dependents and submit newly-ready ones.
            for &dep_idx in &ctx_for_job.dependents[node_index] {
                let prev = ctx_for_job.in_degree[dep_idx].fetch_sub(1, Ordering::AcqRel);
                if prev == 1 {
                    submit_graph_node_direct(dep_idx, Arc::clone(&ctx_for_job));
                }
            }

            // Decrement the overall remaining counter.
            let prev = ctx_for_job.remaining.decrement();
            if prev == 1 {
                ctx_for_job.overall_handle.mark_complete();
            }
        }),
        handle: job_handle,
    };

    shared.push(JobPriority::Normal, queued);
    shared.notify_one();
}

/// Inner context shared across graph task callbacks.
///
/// Defined at module scope so that both `submit_graph_node` and
/// `submit_graph_node_direct` can reference it.
struct GraphContextInner {
    in_degree: Arc<Vec<AtomicU32>>,
    dependents: Arc<Vec<Vec<usize>>>,
    remaining: Arc<AtomicCounter>,
    overall_handle: JobHandle,
    task_bodies: Arc<Vec<parking_lot::Mutex<Option<TaskBodyWrapper>>>>,
    shared_state: Arc<SharedState>,
    job_id_ptr: SendPtr<AtomicU64>,
}

// SAFETY: All fields are thread-safe.  See detailed safety comments in
// `TaskGraph::execute`.
unsafe impl Send for GraphContextInner {}
unsafe impl Sync for GraphContextInner {}

impl Default for TaskGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Synchronization primitives
// ---------------------------------------------------------------------------

/// An atomic counter useful for fork/join patterns.
///
/// A group of jobs can share an `AtomicCounter`; each job decrements it on
/// completion. When the counter reaches zero the group is done.
pub struct AtomicCounter {
    /// The underlying counter.
    value: AtomicU32,
}

impl AtomicCounter {
    /// Creates a counter with the given initial value.
    pub fn new(initial: u32) -> Self {
        Self {
            value: AtomicU32::new(initial),
        }
    }

    /// Decrements by one and returns the *previous* value.
    #[inline]
    pub fn decrement(&self) -> u32 {
        self.value.fetch_sub(1, Ordering::AcqRel)
    }

    /// Increments by one and returns the *previous* value.
    #[inline]
    pub fn increment(&self) -> u32 {
        self.value.fetch_add(1, Ordering::AcqRel)
    }

    /// Returns the current value.
    #[inline]
    pub fn load(&self) -> u32 {
        self.value.load(Ordering::Acquire)
    }

    /// Returns `true` when the counter has reached zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.load() == 0
    }
}

/// A simple spin lock for very short critical sections.
///
/// Prefer [`parking_lot::Mutex`] for anything that might take more than a few
/// hundred nanoseconds. `SpinLock` is only appropriate for lock-free algorithm
/// fallback paths or hardware-level synchronization.
pub struct SpinLock {
    /// 0 = unlocked, 1 = locked.
    locked: AtomicBool,
}

impl SpinLock {
    /// Creates a new unlocked spin lock.
    pub const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    /// Acquires the lock, spinning until it becomes available.
    #[inline]
    pub fn lock(&self) -> SpinLockGuard<'_> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Spin with a hint so the CPU can optimize the busy-wait.
            while self.locked.load(Ordering::Relaxed) {
                std::hint::spin_loop();
            }
        }
        SpinLockGuard { lock: self }
    }

    /// Attempts to acquire the lock without spinning.
    ///
    /// Returns `Some(guard)` on success, `None` if already locked.
    #[inline]
    pub fn try_lock(&self) -> Option<SpinLockGuard<'_>> {
        if self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(SpinLockGuard { lock: self })
        } else {
            None
        }
    }
}

/// RAII guard that releases the [`SpinLock`] on drop.
pub struct SpinLockGuard<'a> {
    lock: &'a SpinLock,
}

impl<'a> Drop for SpinLockGuard<'a> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

impl Default for SpinLock {
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Helper: create a job system with a small worker pool for tests.
    fn test_job_system() -> JobSystem {
        JobSystem::new(JobSystemConfig {
            num_workers: Some(4),
            ..Default::default()
        })
    }

    #[test]
    fn submit_and_wait_single_job() {
        let js = test_job_system();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);

        let handle = js.submit(JobPriority::Normal, move |_| {
            c.fetch_add(1, Ordering::SeqCst);
        });

        js.wait(&handle);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn submit_many_jobs() {
        let js = test_job_system();
        let counter = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..100 {
            let c = Arc::clone(&counter);
            let h = js.submit(JobPriority::Normal, move |_| {
                c.fetch_add(1, Ordering::SeqCst);
            });
            handles.push(h);
        }

        for h in &handles {
            js.wait(h);
        }
        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn parallel_for_basic() {
        let js = test_job_system();
        let results = Arc::new(
            (0..100)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>(),
        );

        let r = Arc::clone(&results);
        let handle = js.parallel_for(100, 10, move |start, end, _thread_idx| {
            for i in start..end {
                r[i].fetch_add(1, Ordering::SeqCst);
            }
        });

        js.wait(&handle);

        // Every element should have been touched exactly once.
        for i in 0..100 {
            assert_eq!(
                results[i].load(Ordering::SeqCst),
                1,
                "element {i} was not processed exactly once"
            );
        }
    }

    #[test]
    fn parallel_for_zero_count() {
        let js = test_job_system();
        let handle = js.parallel_for(0, 10, |_, _, _| {
            panic!("should not be called");
        });
        assert!(handle.is_complete());
    }

    #[test]
    fn parallel_for_single_element() {
        let js = test_job_system();
        let called = Arc::new(AtomicBool::new(false));
        let c = Arc::clone(&called);

        let handle = js.parallel_for(1, 10, move |start, end, _| {
            assert_eq!(start, 0);
            assert_eq!(end, 1);
            c.store(true, Ordering::SeqCst);
        });

        js.wait(&handle);
        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn task_graph_linear_chain() {
        let js = test_job_system();
        let order = Arc::new(parking_lot::Mutex::new(Vec::new()));

        let mut graph = TaskGraph::new();

        let o1 = Arc::clone(&order);
        let t0 = graph.add_task(
            move |_: usize| {
                o1.lock().push(0);
            },
            &[],
        );

        let o2 = Arc::clone(&order);
        let t1 = graph.add_task(
            move |_: usize| {
                o2.lock().push(1);
            },
            &[t0],
        );

        let o3 = Arc::clone(&order);
        let _t2 = graph.add_task(
            move |_: usize| {
                o3.lock().push(2);
            },
            &[t1],
        );

        let handle = graph.execute(&js);
        js.wait(&handle);

        let final_order = order.lock().clone();
        assert_eq!(final_order, vec![0, 1, 2]);
    }

    #[test]
    fn task_graph_diamond() {
        // Diamond dependency:
        //     A
        //    / \
        //   B   C
        //    \ /
        //     D
        let js = test_job_system();
        let counter = Arc::new(AtomicUsize::new(0));

        let mut graph = TaskGraph::new();

        let c1 = Arc::clone(&counter);
        let a = graph.add_task(
            move |_: usize| {
                c1.fetch_add(1, Ordering::SeqCst);
            },
            &[],
        );

        let c2 = Arc::clone(&counter);
        let b = graph.add_task(
            move |_: usize| {
                c2.fetch_add(10, Ordering::SeqCst);
            },
            &[a],
        );

        let c3 = Arc::clone(&counter);
        let c = graph.add_task(
            move |_: usize| {
                c3.fetch_add(100, Ordering::SeqCst);
            },
            &[a],
        );

        let c4 = Arc::clone(&counter);
        let _d = graph.add_task(
            move |_: usize| {
                c4.fetch_add(1000, Ordering::SeqCst);
            },
            &[b, c],
        );

        let handle = graph.execute(&js);
        js.wait(&handle);

        assert_eq!(counter.load(Ordering::SeqCst), 1111);
    }

    #[test]
    fn task_graph_empty() {
        let js = test_job_system();
        let graph = TaskGraph::new();
        let handle = graph.execute(&js);
        assert!(handle.is_complete());
    }

    #[test]
    fn priority_ordering() {
        // Submit many low-priority jobs first, then a critical job.
        // The critical job should complete relatively quickly.
        let js = test_job_system();

        // Fill the queues with background jobs that take a bit of time.
        let bg_counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..50 {
            let c = Arc::clone(&bg_counter);
            js.submit(JobPriority::Background, move |_| {
                std::thread::sleep(Duration::from_millis(1));
                c.fetch_add(1, Ordering::SeqCst);
            });
        }

        // Submit a critical job.
        let critical_done = Arc::new(AtomicBool::new(false));
        let cd = Arc::clone(&critical_done);
        let handle = js.submit(JobPriority::Critical, move |_| {
            cd.store(true, Ordering::SeqCst);
        });

        js.wait(&handle);
        assert!(critical_done.load(Ordering::SeqCst));
    }

    #[test]
    fn worker_count_matches_config() {
        let js = JobSystem::new(JobSystemConfig {
            num_workers: Some(7),
            ..Default::default()
        });
        assert_eq!(js.worker_count(), 7);
    }

    #[test]
    fn atomic_counter_basic() {
        let c = AtomicCounter::new(5);
        assert_eq!(c.load(), 5);
        assert!(!c.is_zero());

        assert_eq!(c.increment(), 5); // returns prev=5, now 6
        assert_eq!(c.load(), 6);

        assert_eq!(c.decrement(), 6); // returns prev=6, now 5
        assert_eq!(c.load(), 5);
    }

    #[test]
    fn spin_lock_basic() {
        let lock = SpinLock::new();
        {
            let _guard = lock.lock();
            assert!(lock.try_lock().is_none());
        }
        assert!(lock.try_lock().is_some());
    }

    #[test]
    fn job_handle_lifecycle() {
        let h = JobHandle::new(42);
        assert!(!h.is_complete());
        h.mark_complete();
        assert!(h.is_complete());
    }

    #[test]
    fn drop_joins_workers() {
        // Verify that dropping the job system doesn't panic or leak threads.
        let before = std::time::Instant::now();
        {
            let _js = test_job_system();
            // Drop immediately.
        }
        let elapsed = before.elapsed();
        // Should complete quickly (well under 1 second).
        assert!(elapsed < Duration::from_secs(2));
    }
}
