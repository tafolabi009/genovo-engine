//! Physics performance profiling and diagnostics.
//!
//! Provides:
//! - Per-phase timing: broadphase, narrowphase, solver, integration
//! - Pair counts (broad/narrow phase)
//! - Island counts and sizes
//! - Sleeping body counts
//! - Constraint iteration counts and convergence metrics
//! - Memory usage estimation
//! - Worst-frame tracking and history
//! - Frame-over-frame comparison
//! - Exportable profiling data for external analysis

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default history buffer size (number of frames to keep).
const DEFAULT_HISTORY_SIZE: usize = 300;
/// Number of "worst frame" records to maintain.
const MAX_WORST_FRAMES: usize = 10;
/// Small epsilon for timing comparisons.
const EPSILON: f64 = 1e-9;
/// Bytes per kilobyte.
const KB: usize = 1024;
/// Bytes per megabyte.
const MB: usize = 1024 * 1024;

// ---------------------------------------------------------------------------
// Phase timing
// ---------------------------------------------------------------------------

/// Individual phase timing within a physics step.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimings {
    /// Broadphase collision detection time in seconds.
    pub broadphase: f64,
    /// Narrowphase collision detection time in seconds.
    pub narrowphase: f64,
    /// Constraint solver time in seconds.
    pub solver: f64,
    /// Position/velocity integration time in seconds.
    pub integration: f64,
    /// Island building time in seconds.
    pub island_building: f64,
    /// Continuous collision detection time in seconds.
    pub ccd: f64,
    /// Spatial query update time in seconds.
    pub spatial_update: f64,
    /// Trigger volume detection time in seconds.
    pub trigger_detection: f64,
    /// Callback/event dispatch time in seconds.
    pub event_dispatch: f64,
    /// Total physics step time in seconds.
    pub total: f64,
}

impl PhaseTimings {
    /// Compute the total from individual phases.
    pub fn compute_total(&mut self) {
        self.total = self.broadphase
            + self.narrowphase
            + self.solver
            + self.integration
            + self.island_building
            + self.ccd
            + self.spatial_update
            + self.trigger_detection
            + self.event_dispatch;
    }

    /// Get the percentage of total time spent in each phase.
    pub fn percentages(&self) -> PhasePercentages {
        let total = self.total.max(EPSILON);
        PhasePercentages {
            broadphase: (self.broadphase / total * 100.0) as f32,
            narrowphase: (self.narrowphase / total * 100.0) as f32,
            solver: (self.solver / total * 100.0) as f32,
            integration: (self.integration / total * 100.0) as f32,
            island_building: (self.island_building / total * 100.0) as f32,
            ccd: (self.ccd / total * 100.0) as f32,
            spatial_update: (self.spatial_update / total * 100.0) as f32,
            trigger_detection: (self.trigger_detection / total * 100.0) as f32,
            event_dispatch: (self.event_dispatch / total * 100.0) as f32,
        }
    }

    /// Get the duration of the most expensive phase.
    pub fn bottleneck_phase(&self) -> (&'static str, f64) {
        let phases = [
            ("broadphase", self.broadphase),
            ("narrowphase", self.narrowphase),
            ("solver", self.solver),
            ("integration", self.integration),
            ("island_building", self.island_building),
            ("ccd", self.ccd),
            ("spatial_update", self.spatial_update),
            ("trigger_detection", self.trigger_detection),
            ("event_dispatch", self.event_dispatch),
        ];
        phases
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(("unknown", 0.0))
    }

    /// Convert all timings to milliseconds.
    pub fn to_milliseconds(&self) -> PhaseTimingsMs {
        PhaseTimingsMs {
            broadphase: (self.broadphase * 1000.0) as f32,
            narrowphase: (self.narrowphase * 1000.0) as f32,
            solver: (self.solver * 1000.0) as f32,
            integration: (self.integration * 1000.0) as f32,
            island_building: (self.island_building * 1000.0) as f32,
            ccd: (self.ccd * 1000.0) as f32,
            spatial_update: (self.spatial_update * 1000.0) as f32,
            trigger_detection: (self.trigger_detection * 1000.0) as f32,
            event_dispatch: (self.event_dispatch * 1000.0) as f32,
            total: (self.total * 1000.0) as f32,
        }
    }
}

/// Phase timing percentages.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhasePercentages {
    pub broadphase: f32,
    pub narrowphase: f32,
    pub solver: f32,
    pub integration: f32,
    pub island_building: f32,
    pub ccd: f32,
    pub spatial_update: f32,
    pub trigger_detection: f32,
    pub event_dispatch: f32,
}

/// Phase timings in milliseconds for display.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimingsMs {
    pub broadphase: f32,
    pub narrowphase: f32,
    pub solver: f32,
    pub integration: f32,
    pub island_building: f32,
    pub ccd: f32,
    pub spatial_update: f32,
    pub trigger_detection: f32,
    pub event_dispatch: f32,
    pub total: f32,
}

// ---------------------------------------------------------------------------
// Object counts
// ---------------------------------------------------------------------------

/// Counts of physics objects in the simulation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhysicsObjectCounts {
    /// Total rigid bodies.
    pub total_bodies: usize,
    /// Active (non-sleeping) rigid bodies.
    pub active_bodies: usize,
    /// Sleeping rigid bodies.
    pub sleeping_bodies: usize,
    /// Static bodies.
    pub static_bodies: usize,
    /// Kinematic bodies.
    pub kinematic_bodies: usize,
    /// Total colliders.
    pub total_colliders: usize,
    /// Total joints/constraints.
    pub total_joints: usize,
    /// Active joints.
    pub active_joints: usize,
    /// Trigger volumes.
    pub trigger_volumes: usize,
}

impl PhysicsObjectCounts {
    /// Get the dynamic body count.
    pub fn dynamic_bodies(&self) -> usize {
        self.total_bodies
            .saturating_sub(self.static_bodies)
            .saturating_sub(self.kinematic_bodies)
    }

    /// Get the percentage of bodies that are sleeping.
    pub fn sleep_percentage(&self) -> f32 {
        if self.total_bodies == 0 {
            return 0.0;
        }
        (self.sleeping_bodies as f32 / self.total_bodies as f32) * 100.0
    }
}

// ---------------------------------------------------------------------------
// Collision pair counts
// ---------------------------------------------------------------------------

/// Collision pair counts from broad and narrow phases.
#[derive(Debug, Clone, Copy, Default)]
pub struct CollisionPairCounts {
    /// Pairs found by the broadphase (AABB overlaps).
    pub broadphase_pairs: usize,
    /// Pairs confirmed by narrowphase (actual contacts).
    pub narrowphase_pairs: usize,
    /// Total contact points generated.
    pub contact_points: usize,
    /// Pairs filtered by collision layers.
    pub filtered_pairs: usize,
    /// CCD sweep tests performed.
    pub ccd_sweeps: usize,
    /// CCD time-of-impact hits.
    pub ccd_hits: usize,
}

impl CollisionPairCounts {
    /// Get the narrowphase-to-broadphase ratio (how many broadphase pairs are real contacts).
    pub fn narrowphase_ratio(&self) -> f32 {
        if self.broadphase_pairs == 0 {
            return 0.0;
        }
        self.narrowphase_pairs as f32 / self.broadphase_pairs as f32
    }

    /// Get the average number of contact points per narrowphase pair.
    pub fn avg_contacts_per_pair(&self) -> f32 {
        if self.narrowphase_pairs == 0 {
            return 0.0;
        }
        self.contact_points as f32 / self.narrowphase_pairs as f32
    }
}

// ---------------------------------------------------------------------------
// Island information
// ---------------------------------------------------------------------------

/// Information about simulation islands.
#[derive(Debug, Clone, Default)]
pub struct IslandInfo {
    /// Number of islands.
    pub island_count: usize,
    /// Size of the largest island (body count).
    pub largest_island: usize,
    /// Size of the smallest island.
    pub smallest_island: usize,
    /// Average island size.
    pub average_island_size: f32,
    /// Sizes of all islands.
    pub island_sizes: Vec<usize>,
}

impl IslandInfo {
    /// Update from a list of island sizes.
    pub fn from_sizes(sizes: &[usize]) -> Self {
        if sizes.is_empty() {
            return Self::default();
        }
        let island_count = sizes.len();
        let largest = sizes.iter().copied().max().unwrap_or(0);
        let smallest = sizes.iter().copied().min().unwrap_or(0);
        let total: usize = sizes.iter().sum();
        let average = total as f32 / island_count as f32;

        Self {
            island_count,
            largest_island: largest,
            smallest_island: smallest,
            average_island_size: average,
            island_sizes: sizes.to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// Solver statistics
// ---------------------------------------------------------------------------

/// Constraint solver performance metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct SolverStats {
    /// Number of solver iterations performed.
    pub iterations: usize,
    /// Maximum allowed iterations.
    pub max_iterations: usize,
    /// Whether the solver converged within the iteration limit.
    pub converged: bool,
    /// Final residual error.
    pub residual_error: f32,
    /// Target residual error for convergence.
    pub target_error: f32,
    /// Number of velocity constraints solved.
    pub velocity_constraints: usize,
    /// Number of position constraints solved.
    pub position_constraints: usize,
    /// Warm starting effectiveness (ratio of warm-started vs cold-started).
    pub warm_start_ratio: f32,
}

impl SolverStats {
    /// Get the iteration utilization (iterations used / max iterations).
    pub fn iteration_utilization(&self) -> f32 {
        if self.max_iterations == 0 {
            return 0.0;
        }
        self.iterations as f32 / self.max_iterations as f32
    }

    /// Whether the solver is hitting the iteration limit (potential issue).
    pub fn is_iteration_limited(&self) -> bool {
        !self.converged && self.iterations >= self.max_iterations
    }
}

// ---------------------------------------------------------------------------
// Memory usage
// ---------------------------------------------------------------------------

/// Estimated memory usage of the physics system.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryUsage {
    /// Rigid body storage in bytes.
    pub rigid_bodies: usize,
    /// Collider storage in bytes.
    pub colliders: usize,
    /// Broadphase data structure in bytes.
    pub broadphase: usize,
    /// Contact manifold storage in bytes.
    pub contacts: usize,
    /// Joint/constraint storage in bytes.
    pub joints: usize,
    /// Island data in bytes.
    pub islands: usize,
    /// Spatial query structures in bytes.
    pub spatial_queries: usize,
    /// Miscellaneous/overhead in bytes.
    pub misc: usize,
    /// Total estimated memory in bytes.
    pub total: usize,
}

impl MemoryUsage {
    /// Compute the total from individual components.
    pub fn compute_total(&mut self) {
        self.total = self.rigid_bodies
            + self.colliders
            + self.broadphase
            + self.contacts
            + self.joints
            + self.islands
            + self.spatial_queries
            + self.misc;
    }

    /// Get the total in kilobytes.
    pub fn total_kb(&self) -> f32 {
        self.total as f32 / KB as f32
    }

    /// Get the total in megabytes.
    pub fn total_mb(&self) -> f32 {
        self.total as f32 / MB as f32
    }

    /// Format as a human-readable string.
    pub fn format_summary(&self) -> String {
        if self.total < KB {
            format!("{} B", self.total)
        } else if self.total < MB {
            format!("{:.1} KB", self.total_kb())
        } else {
            format!("{:.2} MB", self.total_mb())
        }
    }

    /// Get the largest memory consumer.
    pub fn largest_consumer(&self) -> (&'static str, usize) {
        let categories = [
            ("rigid_bodies", self.rigid_bodies),
            ("colliders", self.colliders),
            ("broadphase", self.broadphase),
            ("contacts", self.contacts),
            ("joints", self.joints),
            ("islands", self.islands),
            ("spatial_queries", self.spatial_queries),
            ("misc", self.misc),
        ];
        categories
            .iter()
            .max_by_key(|c| c.1)
            .copied()
            .unwrap_or(("unknown", 0))
    }
}

// ---------------------------------------------------------------------------
// Frame record
// ---------------------------------------------------------------------------

/// Complete profiling data for a single physics frame.
#[derive(Debug, Clone)]
pub struct FrameRecord {
    /// Frame number.
    pub frame: u64,
    /// Delta time of this frame.
    pub dt: f32,
    /// Phase timings.
    pub timings: PhaseTimings,
    /// Object counts.
    pub object_counts: PhysicsObjectCounts,
    /// Collision pair counts.
    pub pair_counts: CollisionPairCounts,
    /// Island information.
    pub island_info: IslandInfo,
    /// Solver statistics.
    pub solver_stats: SolverStats,
    /// Memory usage.
    pub memory_usage: MemoryUsage,
}

impl FrameRecord {
    /// Create a new empty frame record.
    pub fn new(frame: u64, dt: f32) -> Self {
        Self {
            frame,
            dt,
            timings: PhaseTimings::default(),
            object_counts: PhysicsObjectCounts::default(),
            pair_counts: CollisionPairCounts::default(),
            island_info: IslandInfo::default(),
            solver_stats: SolverStats::default(),
            memory_usage: MemoryUsage::default(),
        }
    }

    /// Get the physics step time as a fraction of the frame budget (at target FPS).
    pub fn budget_utilization(&self, target_fps: f32) -> f32 {
        let budget = 1.0 / target_fps as f64;
        (self.timings.total / budget) as f32
    }

    /// Check if this frame exceeded the physics time budget.
    pub fn exceeded_budget(&self, target_fps: f32) -> bool {
        self.budget_utilization(target_fps) > 1.0
    }
}

// ---------------------------------------------------------------------------
// Worst frame record
// ---------------------------------------------------------------------------

/// A record of a particularly bad frame for post-mortem analysis.
#[derive(Debug, Clone)]
pub struct WorstFrameRecord {
    /// The frame data.
    pub record: FrameRecord,
    /// Why this frame was flagged (e.g., "total time exceeded budget").
    pub reason: String,
    /// Wall-clock timestamp (if available).
    pub timestamp: f64,
}

// ---------------------------------------------------------------------------
// Running statistics (exponential moving average)
// ---------------------------------------------------------------------------

/// Running statistics with exponential moving average.
#[derive(Debug, Clone)]
pub struct RunningStats {
    /// Current average.
    pub average: f64,
    /// Current maximum (all-time).
    pub maximum: f64,
    /// Current minimum (all-time).
    pub minimum: f64,
    /// Exponential moving average smoothing factor (0-1, higher = more responsive).
    pub alpha: f64,
    /// Number of samples.
    pub sample_count: u64,
    /// Variance estimate (EMA).
    pub variance: f64,
}

impl RunningStats {
    /// Create new running statistics with the given smoothing factor.
    pub fn new(alpha: f64) -> Self {
        Self {
            average: 0.0,
            maximum: 0.0,
            minimum: f64::INFINITY,
            alpha,
            sample_count: 0,
            variance: 0.0,
        }
    }

    /// Add a new sample.
    pub fn push(&mut self, value: f64) {
        self.sample_count += 1;

        if self.sample_count == 1 {
            self.average = value;
            self.maximum = value;
            self.minimum = value;
            self.variance = 0.0;
        } else {
            let delta = value - self.average;
            self.average = self.average * (1.0 - self.alpha) + value * self.alpha;
            let delta2 = value - self.average;
            self.variance = self.variance * (1.0 - self.alpha) + (delta * delta2) * self.alpha;

            self.maximum = self.maximum.max(value);
            self.minimum = self.minimum.min(value);
        }
    }

    /// Get the standard deviation estimate.
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.average = 0.0;
        self.maximum = 0.0;
        self.minimum = f64::INFINITY;
        self.sample_count = 0;
        self.variance = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Physics profiler
// ---------------------------------------------------------------------------

/// The main physics profiler that collects and analyzes performance data.
#[derive(Debug)]
pub struct PhysicsProfiler {
    /// Whether profiling is enabled.
    pub enabled: bool,
    /// History of frame records.
    history: VecDeque<FrameRecord>,
    /// Maximum history size.
    pub history_size: usize,
    /// Worst frame records.
    worst_frames: Vec<WorstFrameRecord>,
    /// Current frame number.
    frame_counter: u64,
    /// Running statistics for total step time.
    pub total_time_stats: RunningStats,
    /// Running statistics for broadphase time.
    pub broadphase_stats: RunningStats,
    /// Running statistics for narrowphase time.
    pub narrowphase_stats: RunningStats,
    /// Running statistics for solver time.
    pub solver_stats: RunningStats,
    /// Running statistics for integration time.
    pub integration_stats: RunningStats,
    /// Running statistics for body count.
    pub body_count_stats: RunningStats,
    /// Running statistics for pair count.
    pub pair_count_stats: RunningStats,
    /// Target FPS for budget calculations.
    pub target_fps: f32,
    /// Physics time budget in seconds (derived from target FPS and budget fraction).
    pub time_budget: f64,
    /// Fraction of frame time allocated to physics (default 0.3 = 30%).
    pub budget_fraction: f64,
    /// Total accumulated physics time across all frames.
    pub total_accumulated_time: f64,
    /// Number of frames that exceeded the time budget.
    pub budget_exceeded_count: u64,
}

impl PhysicsProfiler {
    /// Create a new physics profiler.
    pub fn new() -> Self {
        Self {
            enabled: false,
            history: VecDeque::with_capacity(DEFAULT_HISTORY_SIZE),
            history_size: DEFAULT_HISTORY_SIZE,
            worst_frames: Vec::new(),
            frame_counter: 0,
            total_time_stats: RunningStats::new(0.1),
            broadphase_stats: RunningStats::new(0.1),
            narrowphase_stats: RunningStats::new(0.1),
            solver_stats: RunningStats::new(0.1),
            integration_stats: RunningStats::new(0.1),
            body_count_stats: RunningStats::new(0.1),
            pair_count_stats: RunningStats::new(0.1),
            target_fps: 60.0,
            time_budget: 1.0 / 60.0 * 0.3,
            budget_fraction: 0.3,
            total_accumulated_time: 0.0,
            budget_exceeded_count: 0,
        }
    }

    /// Create a profiler with custom target FPS and budget fraction.
    pub fn with_budget(target_fps: f32, budget_fraction: f64) -> Self {
        let mut profiler = Self::new();
        profiler.target_fps = target_fps;
        profiler.budget_fraction = budget_fraction;
        profiler.time_budget = (1.0 / target_fps as f64) * budget_fraction;
        profiler
    }

    /// Record a new frame's profiling data.
    pub fn record_frame(&mut self, record: FrameRecord) {
        if !self.enabled {
            self.frame_counter += 1;
            return;
        }

        // Update running statistics
        self.total_time_stats.push(record.timings.total);
        self.broadphase_stats.push(record.timings.broadphase);
        self.narrowphase_stats.push(record.timings.narrowphase);
        self.solver_stats.push(record.timings.solver);
        self.integration_stats.push(record.timings.integration);
        self.body_count_stats.push(record.object_counts.total_bodies as f64);
        self.pair_count_stats.push(record.pair_counts.broadphase_pairs as f64);

        self.total_accumulated_time += record.timings.total;

        // Check budget
        if record.timings.total > self.time_budget {
            self.budget_exceeded_count += 1;

            // Record as worst frame if applicable
            self.maybe_record_worst_frame(&record, "exceeded time budget");
        }

        // Check for other anomalies
        if record.solver_stats.is_iteration_limited() {
            self.maybe_record_worst_frame(&record, "solver iteration limited");
        }

        // Add to history
        if self.history.len() >= self.history_size {
            self.history.pop_front();
        }
        self.history.push_back(record);

        self.frame_counter += 1;
    }

    /// Maybe record a frame as a worst frame.
    fn maybe_record_worst_frame(&mut self, record: &FrameRecord, reason: &str) {
        let worst = WorstFrameRecord {
            record: record.clone(),
            reason: reason.to_string(),
            timestamp: self.total_accumulated_time,
        };

        if self.worst_frames.len() < MAX_WORST_FRAMES {
            self.worst_frames.push(worst);
        } else {
            // Replace the least bad worst frame
            if let Some(min_idx) = self
                .worst_frames
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.record
                        .timings
                        .total
                        .partial_cmp(&b.1.record.timings.total)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                if record.timings.total > self.worst_frames[min_idx].record.timings.total {
                    self.worst_frames[min_idx] = worst;
                }
            }
        }
    }

    /// Get the current frame number.
    pub fn frame_count(&self) -> u64 {
        self.frame_counter
    }

    /// Get the last N frame records.
    pub fn last_n_frames(&self, n: usize) -> Vec<&FrameRecord> {
        let start = self.history.len().saturating_sub(n);
        self.history.range(start..).collect()
    }

    /// Get the most recent frame record.
    pub fn last_frame(&self) -> Option<&FrameRecord> {
        self.history.back()
    }

    /// Get the worst frame records.
    pub fn worst_frames(&self) -> &[WorstFrameRecord] {
        &self.worst_frames
    }

    /// Get the average timings over the history window.
    pub fn average_timings(&self) -> PhaseTimings {
        if self.history.is_empty() {
            return PhaseTimings::default();
        }

        let n = self.history.len() as f64;
        let mut avg = PhaseTimings::default();

        for record in &self.history {
            avg.broadphase += record.timings.broadphase;
            avg.narrowphase += record.timings.narrowphase;
            avg.solver += record.timings.solver;
            avg.integration += record.timings.integration;
            avg.island_building += record.timings.island_building;
            avg.ccd += record.timings.ccd;
            avg.spatial_update += record.timings.spatial_update;
            avg.trigger_detection += record.timings.trigger_detection;
            avg.event_dispatch += record.timings.event_dispatch;
            avg.total += record.timings.total;
        }

        avg.broadphase /= n;
        avg.narrowphase /= n;
        avg.solver /= n;
        avg.integration /= n;
        avg.island_building /= n;
        avg.ccd /= n;
        avg.spatial_update /= n;
        avg.trigger_detection /= n;
        avg.event_dispatch /= n;
        avg.total /= n;

        avg
    }

    /// Get the peak (maximum) timings over the history window.
    pub fn peak_timings(&self) -> PhaseTimings {
        let mut peak = PhaseTimings::default();

        for record in &self.history {
            peak.broadphase = peak.broadphase.max(record.timings.broadphase);
            peak.narrowphase = peak.narrowphase.max(record.timings.narrowphase);
            peak.solver = peak.solver.max(record.timings.solver);
            peak.integration = peak.integration.max(record.timings.integration);
            peak.island_building = peak.island_building.max(record.timings.island_building);
            peak.ccd = peak.ccd.max(record.timings.ccd);
            peak.spatial_update = peak.spatial_update.max(record.timings.spatial_update);
            peak.trigger_detection = peak.trigger_detection.max(record.timings.trigger_detection);
            peak.event_dispatch = peak.event_dispatch.max(record.timings.event_dispatch);
            peak.total = peak.total.max(record.timings.total);
        }

        peak
    }

    /// Get the average object counts over the history window.
    pub fn average_counts(&self) -> PhysicsObjectCounts {
        if self.history.is_empty() {
            return PhysicsObjectCounts::default();
        }

        let n = self.history.len();
        let mut sum = PhysicsObjectCounts::default();

        for record in &self.history {
            sum.total_bodies += record.object_counts.total_bodies;
            sum.active_bodies += record.object_counts.active_bodies;
            sum.sleeping_bodies += record.object_counts.sleeping_bodies;
            sum.static_bodies += record.object_counts.static_bodies;
            sum.total_colliders += record.object_counts.total_colliders;
            sum.total_joints += record.object_counts.total_joints;
        }

        PhysicsObjectCounts {
            total_bodies: sum.total_bodies / n,
            active_bodies: sum.active_bodies / n,
            sleeping_bodies: sum.sleeping_bodies / n,
            static_bodies: sum.static_bodies / n,
            kinematic_bodies: sum.kinematic_bodies / n,
            total_colliders: sum.total_colliders / n,
            total_joints: sum.total_joints / n,
            active_joints: sum.active_joints / n,
            trigger_volumes: sum.trigger_volumes / n,
        }
    }

    /// Get the budget exceeded percentage.
    pub fn budget_exceeded_percentage(&self) -> f32 {
        if self.frame_counter == 0 {
            return 0.0;
        }
        (self.budget_exceeded_count as f32 / self.frame_counter as f32) * 100.0
    }

    /// Generate a summary report as a formatted string.
    pub fn summary_report(&self) -> String {
        let avg = self.average_timings().to_milliseconds();
        let peak = self.peak_timings().to_milliseconds();
        let counts = self.average_counts();

        let mut report = String::new();
        report.push_str("=== Physics Profiler Summary ===\n");
        report.push_str(&format!("Frames profiled: {}\n", self.frame_counter));
        report.push_str(&format!(
            "Budget exceeded: {} ({:.1}%)\n",
            self.budget_exceeded_count,
            self.budget_exceeded_percentage()
        ));
        report.push_str("\n--- Average Timings (ms) ---\n");
        report.push_str(&format!("  Total:        {:.3}\n", avg.total));
        report.push_str(&format!("  Broadphase:   {:.3}\n", avg.broadphase));
        report.push_str(&format!("  Narrowphase:  {:.3}\n", avg.narrowphase));
        report.push_str(&format!("  Solver:       {:.3}\n", avg.solver));
        report.push_str(&format!("  Integration:  {:.3}\n", avg.integration));
        report.push_str(&format!("  CCD:          {:.3}\n", avg.ccd));
        report.push_str("\n--- Peak Timings (ms) ---\n");
        report.push_str(&format!("  Total:        {:.3}\n", peak.total));
        report.push_str(&format!("  Broadphase:   {:.3}\n", peak.broadphase));
        report.push_str(&format!("  Narrowphase:  {:.3}\n", peak.narrowphase));
        report.push_str(&format!("  Solver:       {:.3}\n", peak.solver));
        report.push_str("\n--- Average Object Counts ---\n");
        report.push_str(&format!("  Bodies:    {}\n", counts.total_bodies));
        report.push_str(&format!("  Active:    {}\n", counts.active_bodies));
        report.push_str(&format!("  Sleeping:  {}\n", counts.sleeping_bodies));
        report.push_str(&format!("  Colliders: {}\n", counts.total_colliders));
        report.push_str(&format!("  Joints:    {}\n", counts.total_joints));

        if let Some(last) = self.last_frame() {
            report.push_str(&format!(
                "\n--- Memory: {} ---\n",
                last.memory_usage.format_summary()
            ));
        }

        report
    }

    /// Reset all profiling data.
    pub fn reset(&mut self) {
        self.history.clear();
        self.worst_frames.clear();
        self.frame_counter = 0;
        self.total_time_stats.reset();
        self.broadphase_stats.reset();
        self.narrowphase_stats.reset();
        self.solver_stats.reset();
        self.integration_stats.reset();
        self.body_count_stats.reset();
        self.pair_count_stats.reset();
        self.total_accumulated_time = 0.0;
        self.budget_exceeded_count = 0;
    }

    /// Enable profiling.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Get the history as a slice.
    pub fn history(&self) -> &VecDeque<FrameRecord> {
        &self.history
    }

    /// Export timing data as CSV-compatible rows.
    /// Returns (header, rows).
    pub fn export_csv(&self) -> (String, Vec<String>) {
        let header = "frame,dt,total_ms,broadphase_ms,narrowphase_ms,solver_ms,integration_ms,bodies,active,sleeping,bp_pairs,np_pairs,contacts".to_string();
        let rows: Vec<String> = self
            .history
            .iter()
            .map(|r| {
                let ms = r.timings.to_milliseconds();
                format!(
                    "{},{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{}",
                    r.frame,
                    r.dt,
                    ms.total,
                    ms.broadphase,
                    ms.narrowphase,
                    ms.solver,
                    ms.integration,
                    r.object_counts.total_bodies,
                    r.object_counts.active_bodies,
                    r.object_counts.sleeping_bodies,
                    r.pair_counts.broadphase_pairs,
                    r.pair_counts.narrowphase_pairs,
                    r.pair_counts.contact_points,
                )
            })
            .collect();
        (header, rows)
    }

    /// Get timing data as arrays suitable for graphing.
    pub fn timing_arrays(&self) -> TimingArrays {
        let len = self.history.len();
        let mut arrays = TimingArrays {
            frames: Vec::with_capacity(len),
            total: Vec::with_capacity(len),
            broadphase: Vec::with_capacity(len),
            narrowphase: Vec::with_capacity(len),
            solver: Vec::with_capacity(len),
            integration: Vec::with_capacity(len),
        };

        for record in &self.history {
            let ms = record.timings.to_milliseconds();
            arrays.frames.push(record.frame);
            arrays.total.push(ms.total);
            arrays.broadphase.push(ms.broadphase);
            arrays.narrowphase.push(ms.narrowphase);
            arrays.solver.push(ms.solver);
            arrays.integration.push(ms.integration);
        }

        arrays
    }
}

/// Timing data organized as arrays for graphing.
#[derive(Debug, Clone, Default)]
pub struct TimingArrays {
    /// Frame numbers.
    pub frames: Vec<u64>,
    /// Total step time in ms.
    pub total: Vec<f32>,
    /// Broadphase time in ms.
    pub broadphase: Vec<f32>,
    /// Narrowphase time in ms.
    pub narrowphase: Vec<f32>,
    /// Solver time in ms.
    pub solver: Vec<f32>,
    /// Integration time in ms.
    pub integration: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Scoped timer helper
// ---------------------------------------------------------------------------

/// A scoped timer that records elapsed time when dropped.
/// Uses `std::time::Instant` for high-resolution timing.
#[derive(Debug)]
pub struct ScopedTimer {
    /// Start time.
    start: std::time::Instant,
    /// Where to store the elapsed time.
    target: *mut f64,
}

impl ScopedTimer {
    /// Start a new scoped timer that will write the elapsed time to `target`.
    ///
    /// # Safety
    /// The caller must ensure `target` outlives this timer.
    pub unsafe fn new(target: &mut f64) -> Self {
        Self {
            start: std::time::Instant::now(),
            target: target as *mut f64,
        }
    }

    /// Get the elapsed time so far without stopping.
    pub fn elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        // SAFETY: The caller of `new` guarantees the target pointer is valid.
        unsafe {
            *self.target = elapsed;
        }
    }
}

/// Manually measure a closure's execution time.
pub fn measure_time<F, R>(f: F) -> (R, f64)
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let elapsed = start.elapsed().as_secs_f64();
    (result, elapsed)
}

/// Manually measure a closure's execution time in milliseconds.
pub fn measure_time_ms<F, R>(f: F) -> (R, f32)
where
    F: FnOnce() -> R,
{
    let (result, secs) = measure_time(f);
    (result, (secs * 1000.0) as f32)
}
