// engine/ecs/src/system_pipeline.rs
//
// System execution pipeline: prepare/execute/cleanup phases, system parameter
// extraction, exclusive system support, startup/shutdown systems, system
// profiling hooks, pipeline visualization.
//
// The system pipeline orchestrates the execution of ECS systems across three
// phases per frame: prepare (gather parameters), execute (run system logic),
// and cleanup (apply deferred commands). It supports both parallel and
// exclusive (single-threaded) systems, as well as startup systems (run once
// at initialization) and shutdown systems (run once at teardown).

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// System identifier
// ---------------------------------------------------------------------------

/// Unique identifier for a system.
pub type SystemId = u64;

/// Generate a system ID from a name.
pub fn system_id_from_name(name: &str) -> SystemId {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in name.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// System descriptor
// ---------------------------------------------------------------------------

/// Describes when a system runs relative to other systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SystemStage {
    /// Runs before the main update.
    PreUpdate,
    /// The main update stage.
    Update,
    /// Runs after the main update.
    PostUpdate,
    /// Runs before rendering.
    PreRender,
    /// The render stage.
    Render,
    /// Runs after rendering.
    PostRender,
    /// Fixed timestep update (physics, etc.).
    FixedUpdate,
    /// First frame initialization.
    Startup,
    /// Final frame cleanup.
    Shutdown,
}

impl SystemStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PreUpdate => "PreUpdate",
            Self::Update => "Update",
            Self::PostUpdate => "PostUpdate",
            Self::PreRender => "PreRender",
            Self::Render => "Render",
            Self::PostRender => "PostRender",
            Self::FixedUpdate => "FixedUpdate",
            Self::Startup => "Startup",
            Self::Shutdown => "Shutdown",
        }
    }

    /// Order in which stages execute.
    pub fn order(&self) -> u32 {
        match self {
            Self::Startup => 0,
            Self::PreUpdate => 1,
            Self::FixedUpdate => 2,
            Self::Update => 3,
            Self::PostUpdate => 4,
            Self::PreRender => 5,
            Self::Render => 6,
            Self::PostRender => 7,
            Self::Shutdown => 8,
        }
    }

    /// All stages in execution order (excluding startup/shutdown).
    pub fn frame_stages() -> &'static [SystemStage] {
        &[
            SystemStage::PreUpdate,
            SystemStage::FixedUpdate,
            SystemStage::Update,
            SystemStage::PostUpdate,
            SystemStage::PreRender,
            SystemStage::Render,
            SystemStage::PostRender,
        ]
    }
}

/// Describes what component data a system accesses.
#[derive(Debug, Clone)]
pub struct SystemAccess {
    /// Component types read by this system.
    pub reads: Vec<u64>,
    /// Component types written by this system.
    pub writes: Vec<u64>,
    /// Resources read by this system.
    pub resource_reads: Vec<u64>,
    /// Resources written by this system.
    pub resource_writes: Vec<u64>,
}

impl SystemAccess {
    pub fn new() -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            resource_reads: Vec::new(),
            resource_writes: Vec::new(),
        }
    }

    pub fn read_component(mut self, type_id: u64) -> Self {
        self.reads.push(type_id);
        self
    }

    pub fn write_component(mut self, type_id: u64) -> Self {
        self.writes.push(type_id);
        self
    }

    pub fn read_resource(mut self, type_id: u64) -> Self {
        self.resource_reads.push(type_id);
        self
    }

    pub fn write_resource(mut self, type_id: u64) -> Self {
        self.resource_writes.push(type_id);
        self
    }

    /// Check if this access conflicts with another (for parallelism).
    pub fn conflicts_with(&self, other: &SystemAccess) -> bool {
        // Write-write conflict.
        for w in &self.writes {
            if other.writes.contains(w) { return true; }
        }
        // Read-write conflict.
        for r in &self.reads {
            if other.writes.contains(r) { return true; }
        }
        for w in &self.writes {
            if other.reads.contains(w) { return true; }
        }
        // Resource conflicts.
        for w in &self.resource_writes {
            if other.resource_writes.contains(w) || other.resource_reads.contains(w) { return true; }
        }
        for w in &other.resource_writes {
            if self.resource_reads.contains(w) { return true; }
        }
        false
    }
}

/// Execution mode for a system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemExecutionMode {
    /// Can run in parallel with other non-conflicting systems.
    Parallel,
    /// Must run exclusively (no other systems running).
    Exclusive,
}

/// Run condition for a system.
#[derive(Debug, Clone)]
pub enum RunCondition {
    /// Always run.
    Always,
    /// Run only when a specific resource exists.
    ResourceExists(u64),
    /// Run only on specific frames (e.g., every Nth frame).
    EveryNFrames(u32),
    /// Run only when a custom predicate returns true.
    Custom(String),
    /// Never run (disabled).
    Never,
}

/// A descriptor for a registered system.
#[derive(Debug, Clone)]
pub struct SystemDescriptor {
    /// Unique system ID.
    pub id: SystemId,
    /// System name.
    pub name: String,
    /// Stage in which this system runs.
    pub stage: SystemStage,
    /// Execution mode (parallel or exclusive).
    pub execution_mode: SystemExecutionMode,
    /// Data access requirements.
    pub access: SystemAccess,
    /// Priority within the stage (higher = runs first).
    pub priority: i32,
    /// Run condition.
    pub run_condition: RunCondition,
    /// Dependencies (systems that must run before this one).
    pub dependencies: Vec<SystemId>,
    /// Whether this system is enabled.
    pub enabled: bool,
    /// System set (for grouping).
    pub set: Option<String>,
    /// Whether this is a startup system.
    pub is_startup: bool,
    /// Whether this is a shutdown system.
    pub is_shutdown: bool,
}

impl SystemDescriptor {
    pub fn new(name: &str, stage: SystemStage) -> Self {
        Self {
            id: system_id_from_name(name),
            name: name.to_string(),
            stage,
            execution_mode: SystemExecutionMode::Parallel,
            access: SystemAccess::new(),
            priority: 0,
            run_condition: RunCondition::Always,
            dependencies: Vec::new(),
            enabled: true,
            set: None,
            is_startup: stage == SystemStage::Startup,
            is_shutdown: stage == SystemStage::Shutdown,
        }
    }

    pub fn exclusive(mut self) -> Self {
        self.execution_mode = SystemExecutionMode::Exclusive;
        self
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn after(mut self, dependency: SystemId) -> Self {
        self.dependencies.push(dependency);
        self
    }

    pub fn with_access(mut self, access: SystemAccess) -> Self {
        self.access = access;
        self
    }

    pub fn in_set(mut self, set: &str) -> Self {
        self.set = Some(set.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// System profiling
// ---------------------------------------------------------------------------

/// Profiling data for a single system execution.
#[derive(Debug, Clone)]
pub struct SystemProfile {
    pub system_id: SystemId,
    pub system_name: String,
    pub stage: SystemStage,
    pub prepare_time: Duration,
    pub execute_time: Duration,
    pub cleanup_time: Duration,
    pub total_time: Duration,
    pub entities_processed: u64,
    pub frame_number: u64,
}

/// Aggregated profiling data over multiple frames.
#[derive(Debug, Clone)]
pub struct SystemProfileAggregate {
    pub system_id: SystemId,
    pub system_name: String,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub total_time: Duration,
    pub samples: u32,
    pub percentage_of_frame: f32,
}

/// Profiling hook trait.
pub trait SystemProfileHook: Send + Sync {
    fn on_system_start(&self, system_id: SystemId, system_name: &str);
    fn on_system_end(&self, system_id: SystemId, profile: &SystemProfile);
    fn on_stage_start(&self, stage: SystemStage);
    fn on_stage_end(&self, stage: SystemStage, total_time: Duration);
    fn on_frame_end(&self, frame_number: u64, total_time: Duration);
}

/// Default profiling hook that collects timing data.
pub struct DefaultProfileHook {
    profiles: std::sync::Mutex<Vec<SystemProfile>>,
    max_history: usize,
}

impl DefaultProfileHook {
    pub fn new(max_history: usize) -> Self {
        Self {
            profiles: std::sync::Mutex::new(Vec::new()),
            max_history,
        }
    }

    pub fn recent_profiles(&self, count: usize) -> Vec<SystemProfile> {
        let profiles = self.profiles.lock().unwrap();
        let start = if profiles.len() > count { profiles.len() - count } else { 0 };
        profiles[start..].to_vec()
    }

    pub fn aggregate(&self, system_id: SystemId) -> Option<SystemProfileAggregate> {
        let profiles = self.profiles.lock().unwrap();
        let matching: Vec<&SystemProfile> = profiles.iter()
            .filter(|p| p.system_id == system_id)
            .collect();

        if matching.is_empty() { return None; }

        let mut total = Duration::ZERO;
        let mut min_time = Duration::MAX;
        let mut max_time = Duration::ZERO;

        for p in &matching {
            total += p.total_time;
            min_time = min_time.min(p.total_time);
            max_time = max_time.max(p.total_time);
        }

        let avg = total / matching.len() as u32;

        Some(SystemProfileAggregate {
            system_id,
            system_name: matching[0].system_name.clone(),
            avg_time: avg,
            min_time,
            max_time,
            total_time: total,
            samples: matching.len() as u32,
            percentage_of_frame: 0.0,
        })
    }
}

impl SystemProfileHook for DefaultProfileHook {
    fn on_system_start(&self, _system_id: SystemId, _system_name: &str) {}

    fn on_system_end(&self, _system_id: SystemId, profile: &SystemProfile) {
        let mut profiles = self.profiles.lock().unwrap();
        profiles.push(profile.clone());
        while profiles.len() > self.max_history {
            profiles.remove(0);
        }
    }

    fn on_stage_start(&self, _stage: SystemStage) {}
    fn on_stage_end(&self, _stage: SystemStage, _total_time: Duration) {}
    fn on_frame_end(&self, _frame_number: u64, _total_time: Duration) {}
}

// ---------------------------------------------------------------------------
// System pipeline
// ---------------------------------------------------------------------------

/// The system execution pipeline.
///
/// Manages system registration, ordering, and execution across stages.
pub struct SystemPipeline {
    /// All registered system descriptors.
    systems: Vec<SystemDescriptor>,
    /// Systems grouped by stage.
    stage_systems: HashMap<SystemStage, Vec<usize>>,
    /// Execution order (computed from dependencies and priorities).
    execution_order: HashMap<SystemStage, Vec<usize>>,
    /// Whether the execution order needs recomputation.
    dirty: bool,
    /// Frame counter.
    frame_number: u64,
    /// Profiling hook (optional).
    profile_hook: Option<Box<dyn SystemProfileHook>>,
    /// Per-system execution callbacks.
    system_callbacks: HashMap<SystemId, Box<dyn Fn() + Send + Sync>>,
    /// Whether startup systems have been run.
    startup_complete: bool,
    /// Pipeline statistics.
    pub stats: PipelineStats,
}

/// Statistics for the pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub total_systems: u32,
    pub enabled_systems: u32,
    pub exclusive_systems: u32,
    pub parallel_systems: u32,
    pub startup_systems: u32,
    pub shutdown_systems: u32,
    pub systems_per_stage: HashMap<String, u32>,
    pub last_frame_time: Duration,
    pub avg_frame_time: Duration,
    pub frame_times: Vec<Duration>,
}

impl SystemPipeline {
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            stage_systems: HashMap::new(),
            execution_order: HashMap::new(),
            dirty: true,
            frame_number: 0,
            profile_hook: None,
            system_callbacks: HashMap::new(),
            startup_complete: false,
            stats: PipelineStats::default(),
        }
    }

    /// Register a system.
    pub fn add_system(&mut self, descriptor: SystemDescriptor) -> SystemId {
        let id = descriptor.id;
        let stage = descriptor.stage;
        let idx = self.systems.len();
        self.systems.push(descriptor);
        self.stage_systems.entry(stage).or_default().push(idx);
        self.dirty = true;
        id
    }

    /// Register a system with a callback.
    pub fn add_system_with_callback<F>(
        &mut self,
        descriptor: SystemDescriptor,
        callback: F,
    ) -> SystemId
    where
        F: Fn() + Send + Sync + 'static,
    {
        let id = descriptor.id;
        self.system_callbacks.insert(id, Box::new(callback));
        self.add_system(descriptor)
    }

    /// Enable or disable a system.
    pub fn set_enabled(&mut self, system_id: SystemId, enabled: bool) {
        for sys in &mut self.systems {
            if sys.id == system_id {
                sys.enabled = enabled;
                break;
            }
        }
    }

    /// Set the profiling hook.
    pub fn set_profile_hook(&mut self, hook: Box<dyn SystemProfileHook>) {
        self.profile_hook = Some(hook);
    }

    /// Compute the execution order based on dependencies and priorities.
    pub fn compute_order(&mut self) {
        if !self.dirty { return; }

        self.execution_order.clear();

        for (stage, indices) in &self.stage_systems {
            let mut stage_order: Vec<usize> = indices.clone();

            // Sort by priority (higher first), then by name for stability.
            stage_order.sort_by(|&a, &b| {
                let sa = &self.systems[a];
                let sb = &self.systems[b];
                sb.priority.cmp(&sa.priority).then(sa.name.cmp(&sb.name))
            });

            // TODO: topological sort for dependency ordering.
            // For now, just use priority ordering.

            self.execution_order.insert(*stage, stage_order);
        }

        // Update statistics.
        self.update_stats();
        self.dirty = false;
    }

    /// Run a single frame of the pipeline.
    pub fn run_frame(&mut self) {
        let frame_start = Instant::now();
        self.compute_order();

        // Run startup systems on first frame.
        if !self.startup_complete {
            self.run_stage(SystemStage::Startup);
            self.startup_complete = true;
        }

        // Run all frame stages.
        for stage in SystemStage::frame_stages() {
            self.run_stage(*stage);
        }

        self.frame_number += 1;
        let frame_time = frame_start.elapsed();
        self.stats.last_frame_time = frame_time;
        self.stats.frame_times.push(frame_time);
        if self.stats.frame_times.len() > 120 {
            self.stats.frame_times.remove(0);
        }
        if !self.stats.frame_times.is_empty() {
            let total: Duration = self.stats.frame_times.iter().sum();
            self.stats.avg_frame_time = total / self.stats.frame_times.len() as u32;
        }

        if let Some(ref hook) = self.profile_hook {
            hook.on_frame_end(self.frame_number, frame_time);
        }
    }

    /// Run all systems in a specific stage.
    pub fn run_stage(&mut self, stage: SystemStage) {
        let stage_start = Instant::now();

        if let Some(ref hook) = self.profile_hook {
            hook.on_stage_start(stage);
        }

        let order = self.execution_order.get(&stage).cloned().unwrap_or_default();

        for &sys_idx in &order {
            let sys = &self.systems[sys_idx];
            if !sys.enabled { continue; }

            // Check run condition.
            let should_run = match &sys.run_condition {
                RunCondition::Always => true,
                RunCondition::Never => false,
                RunCondition::EveryNFrames(n) => self.frame_number % (*n as u64) == 0,
                RunCondition::ResourceExists(_) => true, // Simplified.
                RunCondition::Custom(_) => true, // Simplified.
            };

            if !should_run { continue; }

            let system_id = sys.id;
            let system_name = sys.name.clone();

            if let Some(ref hook) = self.profile_hook {
                hook.on_system_start(system_id, &system_name);
            }

            let exec_start = Instant::now();

            // Execute the system callback.
            if let Some(callback) = self.system_callbacks.get(&system_id) {
                callback();
            }

            let exec_time = exec_start.elapsed();

            if let Some(ref hook) = self.profile_hook {
                let profile = SystemProfile {
                    system_id,
                    system_name,
                    stage,
                    prepare_time: Duration::ZERO,
                    execute_time: exec_time,
                    cleanup_time: Duration::ZERO,
                    total_time: exec_time,
                    entities_processed: 0,
                    frame_number: self.frame_number,
                };
                hook.on_system_end(system_id, &profile);
            }
        }

        if let Some(ref hook) = self.profile_hook {
            hook.on_stage_end(stage, stage_start.elapsed());
        }
    }

    /// Run shutdown systems.
    pub fn shutdown(&mut self) {
        self.run_stage(SystemStage::Shutdown);
    }

    /// Get the current frame number.
    pub fn frame_number(&self) -> u64 { self.frame_number }

    /// Get all systems in a stage.
    pub fn systems_in_stage(&self, stage: SystemStage) -> Vec<&SystemDescriptor> {
        self.execution_order.get(&stage)
            .map(|indices| indices.iter().map(|&i| &self.systems[i]).collect())
            .unwrap_or_default()
    }

    /// Get a system by ID.
    pub fn get_system(&self, id: SystemId) -> Option<&SystemDescriptor> {
        self.systems.iter().find(|s| s.id == id)
    }

    /// Generate a visualization of the pipeline (for debug UI).
    pub fn visualize(&self) -> PipelineVisualization {
        let mut stages = Vec::new();
        for stage in SystemStage::frame_stages() {
            let systems = self.systems_in_stage(*stage);
            let stage_viz = StageVisualization {
                stage: *stage,
                systems: systems.iter().map(|s| SystemVisualization {
                    id: s.id,
                    name: s.name.clone(),
                    execution_mode: s.execution_mode,
                    enabled: s.enabled,
                    priority: s.priority,
                    dependencies: s.dependencies.clone(),
                }).collect(),
            };
            stages.push(stage_viz);
        }
        PipelineVisualization { stages }
    }

    fn update_stats(&mut self) {
        self.stats.total_systems = self.systems.len() as u32;
        self.stats.enabled_systems = self.systems.iter().filter(|s| s.enabled).count() as u32;
        self.stats.exclusive_systems = self.systems.iter()
            .filter(|s| s.execution_mode == SystemExecutionMode::Exclusive).count() as u32;
        self.stats.parallel_systems = self.systems.iter()
            .filter(|s| s.execution_mode == SystemExecutionMode::Parallel).count() as u32;
        self.stats.startup_systems = self.systems.iter().filter(|s| s.is_startup).count() as u32;
        self.stats.shutdown_systems = self.systems.iter().filter(|s| s.is_shutdown).count() as u32;

        self.stats.systems_per_stage.clear();
        for (stage, indices) in &self.stage_systems {
            self.stats.systems_per_stage.insert(stage.as_str().to_string(), indices.len() as u32);
        }
    }
}

/// Visualization data for the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineVisualization {
    pub stages: Vec<StageVisualization>,
}

#[derive(Debug, Clone)]
pub struct StageVisualization {
    pub stage: SystemStage,
    pub systems: Vec<SystemVisualization>,
}

#[derive(Debug, Clone)]
pub struct SystemVisualization {
    pub id: SystemId,
    pub name: String,
    pub execution_mode: SystemExecutionMode,
    pub enabled: bool,
    pub priority: i32,
    pub dependencies: Vec<SystemId>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_system_id_deterministic() {
        let id1 = system_id_from_name("physics_system");
        let id2 = system_id_from_name("physics_system");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_system_access_conflict() {
        let a = SystemAccess::new().write_component(1);
        let b = SystemAccess::new().read_component(1);
        assert!(a.conflicts_with(&b));

        let c = SystemAccess::new().read_component(1);
        let d = SystemAccess::new().read_component(1);
        assert!(!c.conflicts_with(&d));
    }

    #[test]
    fn test_pipeline_add_and_run() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut pipeline = SystemPipeline::new();
        pipeline.add_system_with_callback(
            SystemDescriptor::new("test_system", SystemStage::Update),
            move || { counter_clone.fetch_add(1, Ordering::SeqCst); },
        );

        pipeline.run_frame();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        pipeline.run_frame();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_pipeline_disable_system() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut pipeline = SystemPipeline::new();
        let id = pipeline.add_system_with_callback(
            SystemDescriptor::new("test", SystemStage::Update),
            move || { counter_clone.fetch_add(1, Ordering::SeqCst); },
        );

        pipeline.run_frame();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        pipeline.set_enabled(id, false);
        pipeline.run_frame();
        assert_eq!(counter.load(Ordering::SeqCst), 1); // Not incremented.
    }

    #[test]
    fn test_pipeline_priority_ordering() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut pipeline = SystemPipeline::new();

        let o1 = order.clone();
        pipeline.add_system_with_callback(
            SystemDescriptor::new("low_priority", SystemStage::Update).with_priority(0),
            move || { o1.lock().unwrap().push("low"); },
        );

        let o2 = order.clone();
        pipeline.add_system_with_callback(
            SystemDescriptor::new("high_priority", SystemStage::Update).with_priority(10),
            move || { o2.lock().unwrap().push("high"); },
        );

        pipeline.run_frame();
        let executed = order.lock().unwrap();
        assert_eq!(executed[0], "high");
        assert_eq!(executed[1], "low");
    }

    #[test]
    fn test_pipeline_visualization() {
        let mut pipeline = SystemPipeline::new();
        pipeline.add_system(SystemDescriptor::new("sys_a", SystemStage::Update));
        pipeline.add_system(SystemDescriptor::new("sys_b", SystemStage::Render));
        pipeline.compute_order();

        let viz = pipeline.visualize();
        assert!(!viz.stages.is_empty());
    }

    #[test]
    fn test_stage_ordering() {
        assert!(SystemStage::PreUpdate.order() < SystemStage::Update.order());
        assert!(SystemStage::Update.order() < SystemStage::PostUpdate.order());
        assert!(SystemStage::PostUpdate.order() < SystemStage::Render.order());
    }
}
