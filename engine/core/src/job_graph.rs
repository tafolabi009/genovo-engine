// engine/core/src/job_graph.rs
//
// Job graph system for the Genovo engine.
//
// Provides a directed acyclic graph (DAG) of jobs with typed inputs/outputs
// and automatic dependency resolution from data flow:
//
// - Jobs with typed inputs and outputs.
// - Automatic dependency inference from data flow connections.
// - Fan-out (one output feeds multiple jobs) and fan-in (multiple inputs).
// - Job stealing for load balancing across worker threads.
// - Job priorities for controlling execution order among independent jobs.
// - Job graph visualization data for debugging.
// - Topological sort for execution ordering.
// - Cycle detection during graph construction.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum jobs in a single graph.
const MAX_JOBS: usize = 4096;

/// Maximum connections per job.
const MAX_CONNECTIONS_PER_JOB: usize = 16;

/// Maximum worker threads for job execution.
const MAX_WORKERS: usize = 32;

/// Default job priority.
const DEFAULT_PRIORITY: i32 = 0;

// ---------------------------------------------------------------------------
// Job Handle
// ---------------------------------------------------------------------------

/// Unique identifier for a job in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobHandle(pub u32);

impl JobHandle {
    /// Invalid handle.
    pub const INVALID: Self = Self(u32::MAX);

    /// Check validity.
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

/// Unique identifier for a data port (input or output).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortId {
    /// Job this port belongs to.
    pub job: JobHandle,
    /// Port index.
    pub index: u32,
    /// Whether this is an input port.
    pub is_input: bool,
}

// ---------------------------------------------------------------------------
// Data Type
// ---------------------------------------------------------------------------

/// Type descriptor for job data.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataType {
    /// No data (void/unit).
    Void,
    /// Boolean.
    Bool,
    /// 32-bit integer.
    Int32,
    /// 64-bit integer.
    Int64,
    /// 32-bit float.
    Float32,
    /// 64-bit float.
    Float64,
    /// Byte buffer.
    Buffer,
    /// 3D vector.
    Vec3,
    /// 4x4 matrix.
    Mat4,
    /// Texture handle.
    Texture,
    /// Mesh handle.
    Mesh,
    /// Custom named type.
    Custom(String),
}

impl DataType {
    /// Byte size of this data type (0 for variable-size types).
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Void => 0,
            Self::Bool => 1,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Buffer => 0,
            Self::Vec3 => 12,
            Self::Mat4 => 64,
            Self::Texture => 8,
            Self::Mesh => 8,
            Self::Custom(_) => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Port Definition
// ---------------------------------------------------------------------------

/// Definition of an input or output port on a job.
#[derive(Debug, Clone)]
pub struct PortDef {
    /// Port name.
    pub name: String,
    /// Data type.
    pub data_type: DataType,
    /// Whether this port is required (for inputs).
    pub required: bool,
    /// Default value (for optional inputs).
    pub default_value: Option<Vec<u8>>,
}

impl PortDef {
    /// Create a required input port.
    pub fn required(name: &str, data_type: DataType) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            required: true,
            default_value: None,
        }
    }

    /// Create an optional input port with a default.
    pub fn optional(name: &str, data_type: DataType) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            required: false,
            default_value: None,
        }
    }

    /// Create an output port.
    pub fn output(name: &str, data_type: DataType) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            required: false,
            default_value: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Job Definition
// ---------------------------------------------------------------------------

/// Status of a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    /// Not yet ready to execute (waiting for dependencies).
    Pending,
    /// All dependencies satisfied, ready to execute.
    Ready,
    /// Currently being executed by a worker.
    Running,
    /// Execution completed successfully.
    Complete,
    /// Execution failed.
    Failed,
    /// Cancelled.
    Cancelled,
}

/// A job in the job graph.
#[derive(Debug, Clone)]
pub struct Job {
    /// Job handle.
    pub handle: JobHandle,
    /// Job name (for debugging).
    pub name: String,
    /// Input port definitions.
    pub inputs: Vec<PortDef>,
    /// Output port definitions.
    pub outputs: Vec<PortDef>,
    /// Execution priority (higher = executed first among ready jobs).
    pub priority: i32,
    /// Current status.
    pub status: JobStatus,
    /// Estimated cost (for load balancing).
    pub estimated_cost: f32,
    /// Worker affinity (-1 = any worker).
    pub worker_affinity: i32,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Execution time in microseconds (after completion).
    pub execution_time_us: f64,
    /// Which worker executed this job.
    pub executed_by_worker: i32,
}

impl Job {
    /// Create a new job.
    pub fn new(handle: JobHandle, name: &str) -> Self {
        Self {
            handle,
            name: name.to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            priority: DEFAULT_PRIORITY,
            status: JobStatus::Pending,
            estimated_cost: 1.0,
            worker_affinity: -1,
            tags: Vec::new(),
            execution_time_us: 0.0,
            executed_by_worker: -1,
        }
    }

    /// Add an input port.
    pub fn add_input(&mut self, port: PortDef) -> u32 {
        let idx = self.inputs.len() as u32;
        self.inputs.push(port);
        idx
    }

    /// Add an output port.
    pub fn add_output(&mut self, port: PortDef) -> u32 {
        let idx = self.outputs.len() as u32;
        self.outputs.push(port);
        idx
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Whether all dependencies are satisfied.
    pub fn is_ready(&self) -> bool {
        self.status == JobStatus::Ready
    }

    /// Whether the job is finished.
    pub fn is_finished(&self) -> bool {
        matches!(self.status, JobStatus::Complete | JobStatus::Failed | JobStatus::Cancelled)
    }
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

/// A data flow connection between an output port and an input port.
#[derive(Debug, Clone)]
pub struct Connection {
    /// Source job.
    pub from_job: JobHandle,
    /// Source output port index.
    pub from_port: u32,
    /// Destination job.
    pub to_job: JobHandle,
    /// Destination input port index.
    pub to_port: u32,
}

// ---------------------------------------------------------------------------
// Job Graph
// ---------------------------------------------------------------------------

/// A directed acyclic graph of jobs with data flow connections.
#[derive(Debug)]
pub struct JobGraph {
    /// All jobs in the graph.
    pub jobs: HashMap<JobHandle, Job>,
    /// Data flow connections.
    pub connections: Vec<Connection>,
    /// Topologically sorted execution order.
    pub execution_order: Vec<JobHandle>,
    /// Next job handle ID.
    next_job_id: u32,
    /// Whether the execution order is stale.
    pub order_dirty: bool,
    /// Graph name.
    pub name: String,
}

impl JobGraph {
    /// Create a new empty job graph.
    pub fn new(name: &str) -> Self {
        Self {
            jobs: HashMap::new(),
            connections: Vec::new(),
            execution_order: Vec::new(),
            next_job_id: 0,
            order_dirty: true,
            name: name.to_string(),
        }
    }

    /// Add a job to the graph. Returns the job handle.
    pub fn add_job(&mut self, name: &str) -> JobHandle {
        let handle = JobHandle(self.next_job_id);
        self.next_job_id += 1;

        let job = Job::new(handle, name);
        self.jobs.insert(handle, job);
        self.order_dirty = true;
        handle
    }

    /// Add a job with ports.
    pub fn add_job_with_ports(
        &mut self,
        name: &str,
        inputs: Vec<PortDef>,
        outputs: Vec<PortDef>,
    ) -> JobHandle {
        let handle = JobHandle(self.next_job_id);
        self.next_job_id += 1;

        let mut job = Job::new(handle, name);
        job.inputs = inputs;
        job.outputs = outputs;
        self.jobs.insert(handle, job);
        self.order_dirty = true;
        handle
    }

    /// Connect an output port to an input port.
    pub fn connect(
        &mut self,
        from_job: JobHandle,
        from_port: u32,
        to_job: JobHandle,
        to_port: u32,
    ) -> Result<(), JobGraphError> {
        // Validate jobs exist.
        if !self.jobs.contains_key(&from_job) {
            return Err(JobGraphError::JobNotFound(from_job));
        }
        if !self.jobs.contains_key(&to_job) {
            return Err(JobGraphError::JobNotFound(to_job));
        }

        // Validate ports.
        if let Some(job) = self.jobs.get(&from_job) {
            if from_port as usize >= job.outputs.len() {
                return Err(JobGraphError::PortNotFound(from_job, from_port));
            }
        }
        if let Some(job) = self.jobs.get(&to_job) {
            if to_port as usize >= job.inputs.len() {
                return Err(JobGraphError::PortNotFound(to_job, to_port));
            }
        }

        // Type check.
        let from_type = self.jobs[&from_job].outputs[from_port as usize].data_type.clone();
        let to_type = self.jobs[&to_job].inputs[to_port as usize].data_type.clone();
        if from_type != to_type {
            return Err(JobGraphError::TypeMismatch {
                from: from_type,
                to: to_type,
            });
        }

        self.connections.push(Connection {
            from_job,
            from_port,
            to_job,
            to_port,
        });
        self.order_dirty = true;

        Ok(())
    }

    /// Remove a job and all its connections.
    pub fn remove_job(&mut self, handle: JobHandle) {
        self.jobs.remove(&handle);
        self.connections.retain(|c| c.from_job != handle && c.to_job != handle);
        self.order_dirty = true;
    }

    /// Compute topological sort of jobs.
    pub fn topological_sort(&mut self) -> Result<(), JobGraphError> {
        let job_handles: Vec<JobHandle> = self.jobs.keys().copied().collect();

        // Build adjacency list and in-degree count.
        let mut in_degree: HashMap<JobHandle, u32> = HashMap::new();
        let mut adj: HashMap<JobHandle, Vec<JobHandle>> = HashMap::new();

        for &handle in &job_handles {
            in_degree.insert(handle, 0);
            adj.insert(handle, Vec::new());
        }

        for conn in &self.connections {
            *in_degree.entry(conn.to_job).or_insert(0) += 1;
            adj.entry(conn.from_job).or_insert_with(Vec::new).push(conn.to_job);
        }

        // Kahn's algorithm.
        let mut queue: Vec<JobHandle> = in_degree.iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&handle, _)| handle)
            .collect();

        // Sort by priority (higher priority first).
        queue.sort_by(|a, b| {
            let pa = self.jobs.get(a).map(|j| j.priority).unwrap_or(0);
            let pb = self.jobs.get(b).map(|j| j.priority).unwrap_or(0);
            pb.cmp(&pa)
        });

        let mut sorted = Vec::with_capacity(job_handles.len());

        while let Some(handle) = queue.pop() {
            sorted.push(handle);

            if let Some(neighbors) = adj.get(&handle) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(&neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(neighbor);
                        }
                    }
                }
            }

            // Re-sort by priority.
            queue.sort_by(|a, b| {
                let pa = self.jobs.get(a).map(|j| j.priority).unwrap_or(0);
                let pb = self.jobs.get(b).map(|j| j.priority).unwrap_or(0);
                pb.cmp(&pa)
            });
        }

        if sorted.len() != job_handles.len() {
            return Err(JobGraphError::CycleDetected);
        }

        self.execution_order = sorted;
        self.order_dirty = false;
        Ok(())
    }

    /// Get the dependencies of a job (jobs that must complete before it).
    pub fn dependencies(&self, handle: JobHandle) -> Vec<JobHandle> {
        self.connections.iter()
            .filter(|c| c.to_job == handle)
            .map(|c| c.from_job)
            .collect()
    }

    /// Get the dependents of a job (jobs waiting on this one).
    pub fn dependents(&self, handle: JobHandle) -> Vec<JobHandle> {
        self.connections.iter()
            .filter(|c| c.from_job == handle)
            .map(|c| c.to_job)
            .collect()
    }

    /// Mark a job as ready if all its dependencies are complete.
    pub fn update_readiness(&mut self) {
        let handles: Vec<JobHandle> = self.jobs.keys().copied().collect();
        for handle in handles {
            let deps = self.dependencies(handle);
            let all_complete = deps.iter().all(|d| {
                self.jobs.get(d).map(|j| j.status == JobStatus::Complete).unwrap_or(true)
            });

            if let Some(job) = self.jobs.get_mut(&handle) {
                if job.status == JobStatus::Pending && all_complete {
                    job.status = JobStatus::Ready;
                }
            }
        }
    }

    /// Get all jobs that are ready to execute.
    pub fn ready_jobs(&self) -> Vec<JobHandle> {
        let mut ready: Vec<JobHandle> = self.jobs.iter()
            .filter(|(_, j)| j.status == JobStatus::Ready)
            .map(|(&h, _)| h)
            .collect();
        ready.sort_by(|a, b| {
            let pa = self.jobs.get(a).map(|j| j.priority).unwrap_or(0);
            let pb = self.jobs.get(b).map(|j| j.priority).unwrap_or(0);
            pb.cmp(&pa)
        });
        ready
    }

    /// Mark a job as complete.
    pub fn complete_job(&mut self, handle: JobHandle) {
        if let Some(job) = self.jobs.get_mut(&handle) {
            job.status = JobStatus::Complete;
        }
        self.update_readiness();
    }

    /// Mark a job as failed.
    pub fn fail_job(&mut self, handle: JobHandle) {
        if let Some(job) = self.jobs.get_mut(&handle) {
            job.status = JobStatus::Failed;
        }
    }

    /// Check if all jobs are complete.
    pub fn is_complete(&self) -> bool {
        self.jobs.values().all(|j| j.is_finished())
    }

    /// Reset all jobs to pending.
    pub fn reset(&mut self) {
        for job in self.jobs.values_mut() {
            job.status = JobStatus::Pending;
            job.execution_time_us = 0.0;
            job.executed_by_worker = -1;
        }
    }

    /// Job count.
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Connection count.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Get a job reference.
    pub fn get_job(&self, handle: JobHandle) -> Option<&Job> {
        self.jobs.get(&handle)
    }

    /// Get a mutable job reference.
    pub fn get_job_mut(&mut self, handle: JobHandle) -> Option<&mut Job> {
        self.jobs.get_mut(&handle)
    }
}

// ---------------------------------------------------------------------------
// Job Graph Errors
// ---------------------------------------------------------------------------

/// Errors from job graph operations.
#[derive(Debug, Clone)]
pub enum JobGraphError {
    /// A referenced job was not found.
    JobNotFound(JobHandle),
    /// A referenced port was not found.
    PortNotFound(JobHandle, u32),
    /// Type mismatch between connected ports.
    TypeMismatch { from: DataType, to: DataType },
    /// The graph contains a cycle.
    CycleDetected,
    /// Maximum job count exceeded.
    TooManyJobs,
}

// ---------------------------------------------------------------------------
// Job Stealing
// ---------------------------------------------------------------------------

/// Work-stealing queue for distributing jobs across workers.
#[derive(Debug)]
pub struct JobStealingQueue {
    /// Per-worker job queues.
    pub queues: Vec<Vec<JobHandle>>,
    /// Number of workers.
    pub worker_count: usize,
    /// Total jobs submitted.
    pub total_submitted: u64,
    /// Total jobs stolen.
    pub total_stolen: u64,
}

impl JobStealingQueue {
    /// Create a new work-stealing queue.
    pub fn new(worker_count: usize) -> Self {
        let count = worker_count.min(MAX_WORKERS);
        Self {
            queues: (0..count).map(|_| Vec::new()).collect(),
            worker_count: count,
            total_submitted: 0,
            total_stolen: 0,
        }
    }

    /// Push a job to a specific worker's queue.
    pub fn push(&mut self, worker: usize, job: JobHandle) {
        if worker < self.worker_count {
            self.queues[worker].push(job);
            self.total_submitted += 1;
        }
    }

    /// Push a job to the least-loaded worker.
    pub fn push_balanced(&mut self, job: JobHandle) {
        let min_worker = (0..self.worker_count)
            .min_by_key(|&w| self.queues[w].len())
            .unwrap_or(0);
        self.push(min_worker, job);
    }

    /// Pop a job from a worker's queue.
    pub fn pop(&mut self, worker: usize) -> Option<JobHandle> {
        if worker < self.worker_count {
            self.queues[worker].pop()
        } else {
            None
        }
    }

    /// Steal a job from another worker's queue.
    pub fn steal(&mut self, thief: usize) -> Option<JobHandle> {
        let max_worker = (0..self.worker_count)
            .filter(|&w| w != thief)
            .max_by_key(|&w| self.queues[w].len());

        if let Some(victim) = max_worker {
            if !self.queues[victim].is_empty() {
                self.total_stolen += 1;
                return Some(self.queues[victim].remove(0));
            }
        }
        None
    }

    /// Total pending jobs.
    pub fn total_pending(&self) -> usize {
        self.queues.iter().map(|q| q.len()).sum()
    }

    /// Check if all queues are empty.
    pub fn is_empty(&self) -> bool {
        self.queues.iter().all(|q| q.is_empty())
    }
}

// ---------------------------------------------------------------------------
// Graph Visualization
// ---------------------------------------------------------------------------

/// Visualization data for a job graph.
#[derive(Debug, Clone)]
pub struct GraphVisualization {
    /// Nodes (jobs).
    pub nodes: Vec<VizNode>,
    /// Edges (connections).
    pub edges: Vec<VizEdge>,
}

/// A node in the visualization.
#[derive(Debug, Clone)]
pub struct VizNode {
    /// Job handle.
    pub handle: JobHandle,
    /// Display name.
    pub name: String,
    /// Status.
    pub status: JobStatus,
    /// Priority.
    pub priority: i32,
    /// Execution time (if complete).
    pub execution_time_us: f64,
    /// Position in visualization layout.
    pub position: [f32; 2],
    /// Color based on status.
    pub color: [f32; 4],
}

/// An edge in the visualization.
#[derive(Debug, Clone)]
pub struct VizEdge {
    /// Source node.
    pub from: JobHandle,
    /// Destination node.
    pub to: JobHandle,
    /// Data type label.
    pub label: String,
}

impl JobGraph {
    /// Generate visualization data.
    pub fn visualize(&self) -> GraphVisualization {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for (i, (&handle, job)) in self.jobs.iter().enumerate() {
            let color = match job.status {
                JobStatus::Pending => [0.5, 0.5, 0.5, 1.0],
                JobStatus::Ready => [1.0, 1.0, 0.0, 1.0],
                JobStatus::Running => [0.0, 0.5, 1.0, 1.0],
                JobStatus::Complete => [0.0, 1.0, 0.0, 1.0],
                JobStatus::Failed => [1.0, 0.0, 0.0, 1.0],
                JobStatus::Cancelled => [0.3, 0.3, 0.3, 1.0],
            };

            nodes.push(VizNode {
                handle,
                name: job.name.clone(),
                status: job.status,
                priority: job.priority,
                execution_time_us: job.execution_time_us,
                position: [i as f32 * 150.0, 0.0],
                color,
            });
        }

        for conn in &self.connections {
            let label = self.jobs.get(&conn.from_job)
                .and_then(|j| j.outputs.get(conn.from_port as usize))
                .map(|p| format!("{}: {:?}", p.name, p.data_type))
                .unwrap_or_default();

            edges.push(VizEdge {
                from: conn.from_job,
                to: conn.to_job,
                label,
            });
        }

        GraphVisualization { nodes, edges }
    }
}
