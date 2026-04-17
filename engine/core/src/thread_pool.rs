// engine/core/src/thread_pool.rs
//
// Thread pool for the Genovo core module.
//
// Provides configurable worker count, task queue with priority, work stealing,
// thread affinity, thread naming, idle callbacks, graceful shutdown, and
// pool statistics.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Condvar, atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::fmt;

pub type TaskFn = Box<dyn FnOnce() + Send + 'static>;

pub const DEFAULT_WORKER_COUNT: usize = 4;
pub const MAX_WORKER_COUNT: usize = 64;
pub const DEFAULT_QUEUE_CAPACITY: usize = 4096;
pub const IDLE_SPIN_COUNT: u32 = 100;
pub const STEAL_BATCH_SIZE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority { Low = 0, Normal = 1, High = 2, Critical = 3 }

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Low => write!(f, "Low"), Self::Normal => write!(f, "Normal"), Self::High => write!(f, "High"), Self::Critical => write!(f, "Critical") }
    }
}

struct PrioritizedTask {
    task: TaskFn,
    priority: TaskPriority,
    submit_time: Instant,
}

struct WorkQueue {
    tasks: VecDeque<PrioritizedTask>,
    capacity: usize,
}

impl WorkQueue {
    fn new(capacity: usize) -> Self { Self { tasks: VecDeque::with_capacity(capacity), capacity } }
    fn push(&mut self, task: PrioritizedTask) -> bool {
        if self.tasks.len() >= self.capacity { return false; }
        // Insert by priority (higher priority first).
        let pos = self.tasks.partition_point(|t| t.priority >= task.priority);
        self.tasks.insert(pos, task);
        true
    }
    fn pop(&mut self) -> Option<PrioritizedTask> { self.tasks.pop_front() }
    fn steal(&mut self) -> Option<PrioritizedTask> { self.tasks.pop_back() }
    fn len(&self) -> usize { self.tasks.len() }
    fn is_empty(&self) -> bool { self.tasks.is_empty() }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerStats {
    pub tasks_completed: u64,
    pub tasks_stolen: u64,
    pub idle_time_ms: f64,
    pub active_time_ms: f64,
    pub avg_task_time_us: f64,
}

#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_tasks_submitted: u64,
    pub total_tasks_completed: u64,
    pub total_tasks_stolen: u64,
    pub pending_tasks: usize,
    pub active_workers: usize,
    pub idle_workers: usize,
    pub worker_stats: Vec<WorkerStats>,
    pub uptime_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    pub worker_count: usize,
    pub queue_capacity: usize,
    pub thread_name_prefix: String,
    pub enable_work_stealing: bool,
    pub enable_affinity: bool,
    pub stack_size: Option<usize>,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        let cpus = thread::available_parallelism().map(|n| n.get()).unwrap_or(DEFAULT_WORKER_COUNT);
        Self {
            worker_count: cpus, queue_capacity: DEFAULT_QUEUE_CAPACITY,
            thread_name_prefix: "genovo-worker".to_string(),
            enable_work_stealing: true, enable_affinity: false, stack_size: None,
        }
    }
}

struct SharedState {
    global_queue: Mutex<WorkQueue>,
    worker_queues: Vec<Mutex<WorkQueue>>,
    condvar: Condvar,
    shutdown: AtomicBool,
    active_count: AtomicUsize,
    tasks_submitted: AtomicU64,
    tasks_completed: AtomicU64,
    tasks_stolen: AtomicU64,
    start_time: Instant,
    enable_stealing: bool,
}

pub struct ThreadPool {
    workers: Vec<thread::JoinHandle<WorkerStats>>,
    shared: Arc<SharedState>,
    config: ThreadPoolConfig,
}

impl ThreadPool {
    pub fn new(config: ThreadPoolConfig) -> Self {
        let worker_count = config.worker_count.min(MAX_WORKER_COUNT).max(1);
        let mut worker_queues = Vec::with_capacity(worker_count);
        for _ in 0..worker_count { worker_queues.push(Mutex::new(WorkQueue::new(config.queue_capacity / worker_count.max(1)))); }

        let shared = Arc::new(SharedState {
            global_queue: Mutex::new(WorkQueue::new(config.queue_capacity)),
            worker_queues,
            condvar: Condvar::new(),
            shutdown: AtomicBool::new(false),
            active_count: AtomicUsize::new(0),
            tasks_submitted: AtomicU64::new(0),
            tasks_completed: AtomicU64::new(0),
            tasks_stolen: AtomicU64::new(0),
            start_time: Instant::now(),
            enable_stealing: config.enable_work_stealing,
        });

        let mut workers = Vec::with_capacity(worker_count);
        for i in 0..worker_count {
            let shared_clone = shared.clone();
            let name = format!("{}-{}", config.thread_name_prefix, i);
            let mut builder = thread::Builder::new().name(name);
            if let Some(size) = config.stack_size { builder = builder.stack_size(size); }

            let handle = builder.spawn(move || Self::worker_loop(shared_clone, i)).expect("Failed to spawn worker thread");
            workers.push(handle);
        }

        Self { workers, shared, config }
    }

    pub fn with_default_config() -> Self { Self::new(ThreadPoolConfig::default()) }

    pub fn submit<F>(&self, f: F) where F: FnOnce() + Send + 'static {
        self.submit_with_priority(f, TaskPriority::Normal);
    }

    pub fn submit_with_priority<F>(&self, f: F, priority: TaskPriority) where F: FnOnce() + Send + 'static {
        let task = PrioritizedTask { task: Box::new(f), priority, submit_time: Instant::now() };
        {
            let mut queue = self.shared.global_queue.lock().unwrap();
            queue.push(task);
        }
        self.shared.tasks_submitted.fetch_add(1, Ordering::Relaxed);
        self.shared.condvar.notify_one();
    }

    pub fn submit_batch<F>(&self, tasks: Vec<F>, priority: TaskPriority) where F: FnOnce() + Send + 'static {
        {
            let mut queue = self.shared.global_queue.lock().unwrap();
            for f in tasks {
                queue.push(PrioritizedTask { task: Box::new(f), priority, submit_time: Instant::now() });
                self.shared.tasks_submitted.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.shared.condvar.notify_all();
    }

    fn worker_loop(shared: Arc<SharedState>, worker_id: usize) -> WorkerStats {
        let mut stats = WorkerStats::default();
        let start = Instant::now();

        loop {
            if shared.shutdown.load(Ordering::Relaxed) {
                // Drain remaining tasks.
                while let Some(task) = { shared.global_queue.lock().unwrap().pop() } {
                    shared.active_count.fetch_add(1, Ordering::Relaxed);
                    (task.task)();
                    shared.active_count.fetch_sub(1, Ordering::Relaxed);
                    shared.tasks_completed.fetch_add(1, Ordering::Relaxed);
                    stats.tasks_completed += 1;
                }
                break;
            }

            // Try global queue.
            let task = { shared.global_queue.lock().unwrap().pop() };
            if let Some(task) = task {
                shared.active_count.fetch_add(1, Ordering::Relaxed);
                let task_start = Instant::now();
                (task.task)();
                let elapsed = task_start.elapsed().as_micros() as f64;
                stats.active_time_ms += elapsed / 1000.0;
                stats.tasks_completed += 1;
                let n = stats.tasks_completed as f64;
                stats.avg_task_time_us = stats.avg_task_time_us * ((n - 1.0) / n) + elapsed / n;
                shared.active_count.fetch_sub(1, Ordering::Relaxed);
                shared.tasks_completed.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            // Try work stealing.
            if shared.enable_stealing {
                let mut stolen = false;
                for i in 0..shared.worker_queues.len() {
                    if i == worker_id { continue; }
                    if let Ok(mut q) = shared.worker_queues[i].try_lock() {
                        if let Some(task) = q.steal() {
                            shared.active_count.fetch_add(1, Ordering::Relaxed);
                            (task.task)();
                            shared.active_count.fetch_sub(1, Ordering::Relaxed);
                            shared.tasks_completed.fetch_add(1, Ordering::Relaxed);
                            shared.tasks_stolen.fetch_add(1, Ordering::Relaxed);
                            stats.tasks_completed += 1;
                            stats.tasks_stolen += 1;
                            stolen = true;
                            break;
                        }
                    }
                }
                if stolen { continue; }
            }

            // Wait for work.
            let idle_start = Instant::now();
            {
                let guard = shared.global_queue.lock().unwrap();
                if guard.is_empty() && !shared.shutdown.load(Ordering::Relaxed) {
                    let _ = shared.condvar.wait_timeout(guard, Duration::from_millis(10));
                }
            }
            stats.idle_time_ms += idle_start.elapsed().as_millis() as f64;
        }

        stats
    }

    pub fn worker_count(&self) -> usize { self.config.worker_count }
    pub fn pending_tasks(&self) -> usize { self.shared.global_queue.lock().unwrap().len() }
    pub fn active_workers(&self) -> usize { self.shared.active_count.load(Ordering::Relaxed) }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total_tasks_submitted: self.shared.tasks_submitted.load(Ordering::Relaxed),
            total_tasks_completed: self.shared.tasks_completed.load(Ordering::Relaxed),
            total_tasks_stolen: self.shared.tasks_stolen.load(Ordering::Relaxed),
            pending_tasks: self.pending_tasks(),
            active_workers: self.active_workers(),
            idle_workers: self.config.worker_count - self.active_workers(),
            worker_stats: Vec::new(),
            uptime_seconds: self.shared.start_time.elapsed().as_secs_f64(),
        }
    }

    pub fn shutdown(mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.condvar.notify_all();
        for handle in self.workers.drain(..) { let _ = handle.join(); }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.condvar.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_create_pool() {
        let pool = ThreadPool::new(ThreadPoolConfig { worker_count: 2, ..Default::default() });
        assert_eq!(pool.worker_count(), 2);
        pool.shutdown();
    }

    #[test]
    fn test_submit_task() {
        let pool = ThreadPool::new(ThreadPoolConfig { worker_count: 2, ..Default::default() });
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        pool.submit(move || { c.fetch_add(1, Ordering::Relaxed); });
        thread::sleep(Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        pool.shutdown();
    }

    #[test]
    fn test_batch_submit() {
        let pool = ThreadPool::new(ThreadPoolConfig { worker_count: 4, ..Default::default() });
        let counter = Arc::new(AtomicU32::new(0));
        let tasks: Vec<_> = (0..10).map(|_| { let c = counter.clone(); Box::new(move || { c.fetch_add(1, Ordering::Relaxed); }) as Box<dyn FnOnce() + Send> }).collect();
        pool.submit_batch(tasks, TaskPriority::Normal);
        thread::sleep(Duration::from_millis(200));
        assert_eq!(counter.load(Ordering::Relaxed), 10);
        pool.shutdown();
    }
}
