//! Hierarchical CPU frame profiler for the Genovo engine.
//!
//! The profiler collects per-frame timing data organized into a hierarchical
//! tree of scopes. Scopes can be opened manually via [`Profiler::begin_scope`]
//! / [`Profiler::end_scope`], or automatically with the RAII [`ScopeGuard`]
//! helper (and the [`profile_scope!`] macro).
//!
//! # Architecture
//!
//! Each thread maintains its own scope stack via `thread_local!`. At the end of
//! every frame, the per-thread trees are merged into a single
//! [`ProfileFrame`] that is pushed into a rolling history ring buffer (default
//! 120 frames). Statistics queries (`get_average`, `get_max`) operate over
//! this history.
//!
//! # Chrome Tracing
//!
//! The profiler can export data in the Chrome Trace Event format (JSON),
//! compatible with `chrome://tracing` or [Perfetto](https://ui.perfetto.dev).

pub mod gpu_profiler;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Maximum number of frames retained in the rolling history.
const DEFAULT_HISTORY_SIZE: usize = 120;

/// Threshold (in microseconds) below which a scope is omitted from reports.
const REPORT_MIN_DURATION_US: u64 = 10;

// ---------------------------------------------------------------------------
// ProfileNode — a single scope measurement
// ---------------------------------------------------------------------------

/// A single node in the hierarchical profile tree.
///
/// Each node records the name of the scope, its start time, measured duration,
/// how many times it was invoked in the frame, and its child scopes.
#[derive(Debug, Clone)]
pub struct ProfileNode {
    /// Human-readable scope name (e.g. `"PhysicsStep"`).
    pub name: String,
    /// Wall-clock time when the scope was entered.
    pub start_time: Instant,
    /// Measured wall-clock duration of the scope.
    pub duration: Duration,
    /// Number of times this scope was entered in the current frame.
    pub call_count: u32,
    /// Child scopes nested inside this one.
    pub children: Vec<ProfileNode>,
    /// Depth in the tree (0 = root).
    pub depth: u32,
}

impl ProfileNode {
    /// Create a new profile node with the given name and start time.
    fn new(name: &str, start_time: Instant, depth: u32) -> Self {
        Self {
            name: name.to_string(),
            start_time,
            duration: Duration::ZERO,
            call_count: 1,
            children: Vec::new(),
            depth,
        }
    }

    /// Recursively find a child node by name.
    pub fn find(&self, name: &str) -> Option<&ProfileNode> {
        if self.name == name {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find(name) {
                return Some(found);
            }
        }
        None
    }

    /// Recursively find a mutable child node by name.
    pub fn find_mut(&mut self, name: &str) -> Option<&mut ProfileNode> {
        if self.name == name {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.find_mut(name) {
                return Some(found);
            }
        }
        None
    }

    /// Collect all nodes into a flat list for easy iteration.
    pub fn flatten(&self) -> Vec<&ProfileNode> {
        let mut result = vec![self];
        for child in &self.children {
            result.extend(child.flatten());
        }
        result
    }

    /// Compute self-time: this node's duration minus children's durations.
    pub fn self_time(&self) -> Duration {
        let child_total: Duration = self.children.iter().map(|c| c.duration).sum();
        self.duration.saturating_sub(child_total)
    }

    /// Format this node (and children) as indented text.
    fn format_tree(&self, out: &mut String, indent: usize) {
        let prefix = "  ".repeat(indent);
        let ms = self.duration.as_secs_f64() * 1000.0;
        let self_ms = self.self_time().as_secs_f64() * 1000.0;
        out.push_str(&format!(
            "{}{} — {:.3}ms (self: {:.3}ms, calls: {})\n",
            prefix, self.name, ms, self_ms, self.call_count
        ));
        for child in &self.children {
            child.format_tree(out, indent + 1);
        }
    }
}

impl fmt::Display for ProfileNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = String::new();
        self.format_tree(&mut buf, 0);
        write!(f, "{}", buf)
    }
}

// ---------------------------------------------------------------------------
// ProfileFrame — one complete frame's profile data
// ---------------------------------------------------------------------------

/// A complete frame of profiling data, containing the root nodes of the scope
/// hierarchy and the overall frame duration.
#[derive(Debug, Clone)]
pub struct ProfileFrame {
    /// Frame index (monotonically increasing).
    pub frame_index: u64,
    /// Root-level scopes for this frame.
    pub roots: Vec<ProfileNode>,
    /// Total wall-clock frame time.
    pub frame_duration: Duration,
    /// Timestamp when the frame started.
    pub frame_start: Instant,
}

impl ProfileFrame {
    /// Create an empty profile frame.
    fn new(frame_index: u64, frame_start: Instant) -> Self {
        Self {
            frame_index,
            roots: Vec::new(),
            frame_duration: Duration::ZERO,
            frame_start,
        }
    }

    /// Find a node by name anywhere in this frame's tree.
    pub fn find_node(&self, name: &str) -> Option<&ProfileNode> {
        for root in &self.roots {
            if let Some(node) = root.find(name) {
                return Some(node);
            }
        }
        None
    }

    /// Collect all nodes (flattened) from this frame.
    pub fn all_nodes(&self) -> Vec<&ProfileNode> {
        let mut result = Vec::new();
        for root in &self.roots {
            result.extend(root.flatten());
        }
        result
    }
}

impl fmt::Display for ProfileFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Frame #{} — {:.3}ms",
            self.frame_index,
            self.frame_duration.as_secs_f64() * 1000.0,
        )?;
        for root in &self.roots {
            write!(f, "{}", root)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Thread-local scope stack
// ---------------------------------------------------------------------------

/// Per-thread profiling state: a stack of open scopes and completed root nodes.
struct ThreadProfileState {
    /// Stack of nodes currently being timed (the last one is the innermost
    /// open scope).
    scope_stack: Vec<ProfileNode>,
    /// Root-level completed scopes for the current frame.
    completed_roots: Vec<ProfileNode>,
}

impl ThreadProfileState {
    fn new() -> Self {
        Self {
            scope_stack: Vec::new(),
            completed_roots: Vec::new(),
        }
    }

    fn begin_scope(&mut self, name: &str) {
        let depth = self.scope_stack.len() as u32;
        let node = ProfileNode::new(name, Instant::now(), depth);
        self.scope_stack.push(node);
    }

    fn end_scope(&mut self) {
        if let Some(mut node) = self.scope_stack.pop() {
            node.duration = node.start_time.elapsed();
            if let Some(parent) = self.scope_stack.last_mut() {
                // Merge into existing child of the same name if present.
                if let Some(existing) = parent.children.iter_mut().find(|c| c.name == node.name) {
                    existing.duration += node.duration;
                    existing.call_count += 1;
                    existing.children.extend(node.children);
                } else {
                    parent.children.push(node);
                }
            } else {
                // No parent — this is a root-level scope.
                self.completed_roots.push(node);
            }
        }
    }

    fn take_roots(&mut self) -> Vec<ProfileNode> {
        // Close any still-open scopes (should not happen in well-formed code).
        while !self.scope_stack.is_empty() {
            self.end_scope();
        }
        std::mem::take(&mut self.completed_roots)
    }
}

thread_local! {
    static THREAD_PROFILE: RefCell<ThreadProfileState> = RefCell::new(ThreadProfileState::new());
}

// ---------------------------------------------------------------------------
// ProfileEvent — timeline data for export
// ---------------------------------------------------------------------------

/// A single event for timeline / trace visualization.
#[derive(Debug, Clone, Serialize)]
pub struct ProfileEvent {
    /// Event name.
    pub name: String,
    /// Category string.
    pub cat: String,
    /// Phase: "B" (begin), "E" (end), or "X" (complete).
    pub ph: String,
    /// Timestamp in microseconds from an arbitrary epoch.
    pub ts: u64,
    /// Duration in microseconds (only valid for phase "X").
    pub dur: u64,
    /// Process id.
    pub pid: u32,
    /// Thread id.
    pub tid: u32,
}

// ---------------------------------------------------------------------------
// ProfileReport
// ---------------------------------------------------------------------------

/// A formatted text report of profiling hotspots.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Report header.
    pub title: String,
    /// Lines of the report.
    pub lines: Vec<ProfileReportLine>,
    /// Number of frames averaged over.
    pub frame_count: usize,
    /// Average frame time.
    pub avg_frame_time: Duration,
    /// Worst frame time.
    pub max_frame_time: Duration,
}

/// A single line in a profile report.
#[derive(Debug, Clone)]
pub struct ProfileReportLine {
    /// Scope name.
    pub name: String,
    /// Average time over the sampled frames.
    pub avg_duration: Duration,
    /// Maximum time over the sampled frames.
    pub max_duration: Duration,
    /// Average call count per frame.
    pub avg_calls: f64,
    /// Percentage of frame time.
    pub pct_of_frame: f64,
    /// Indentation depth.
    pub depth: u32,
}

impl fmt::Display for ProfileReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== {} ===", self.title)?;
        writeln!(
            f,
            "Frames: {}  |  Avg frame: {:.3}ms  |  Worst frame: {:.3}ms",
            self.frame_count,
            self.avg_frame_time.as_secs_f64() * 1000.0,
            self.max_frame_time.as_secs_f64() * 1000.0,
        )?;
        writeln!(
            f,
            "{:<40} {:>10} {:>10} {:>8} {:>8}",
            "Scope", "Avg (ms)", "Max (ms)", "Calls", "% Frame"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;
        for line in &self.lines {
            let indent = "  ".repeat(line.depth as usize);
            writeln!(
                f,
                "{}{:<width$} {:>10.3} {:>10.3} {:>8.1} {:>7.1}%",
                indent,
                line.name,
                line.avg_duration.as_secs_f64() * 1000.0,
                line.max_duration.as_secs_f64() * 1000.0,
                line.avg_calls,
                line.pct_of_frame,
                width = 40 - (line.depth as usize * 2),
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Profiler — the main interface
// ---------------------------------------------------------------------------

/// Hierarchical frame profiler.
///
/// # Usage
///
/// ```ignore
/// static PROFILER: Lazy<Profiler> = Lazy::new(Profiler::new);
///
/// // Start of frame
/// PROFILER.begin_frame();
///
/// {
///     let _guard = PROFILER.scope("Physics");
///     // ... physics work ...
///     {
///         let _guard = PROFILER.scope("Collision");
///         // ... collision detection ...
///     }
/// }
///
/// // End of frame
/// PROFILER.end_frame();
///
/// // Query statistics
/// let avg = PROFILER.get_average("Physics");
/// ```
pub struct Profiler {
    /// Rolling frame history (ring buffer).
    history: Mutex<Vec<ProfileFrame>>,
    /// Maximum number of frames to retain.
    history_size: usize,
    /// Monotonically increasing frame counter.
    frame_counter: AtomicU64,
    /// Whether profiling is currently enabled.
    enabled: AtomicBool,
    /// The instant when the current frame started.
    frame_start: Mutex<Option<Instant>>,
    /// Global epoch for computing absolute timestamps.
    epoch: Instant,
    /// Collected thread roots waiting to be merged (per-thread results are
    /// pushed here at frame end).
    pending_roots: Mutex<Vec<Vec<ProfileNode>>>,
}

impl Profiler {
    /// Create a new profiler with default history size (120 frames).
    pub fn new() -> Self {
        Self::with_history_size(DEFAULT_HISTORY_SIZE)
    }

    /// Create a new profiler with a custom history size.
    pub fn with_history_size(size: usize) -> Self {
        Self {
            history: Mutex::new(Vec::with_capacity(size)),
            history_size: size,
            frame_counter: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
            frame_start: Mutex::new(None),
            epoch: Instant::now(),
            pending_roots: Mutex::new(Vec::new()),
        }
    }

    /// Enable or disable the profiler at runtime.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Returns `true` if the profiler is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Signal the start of a new frame. Must be called once per frame before
    /// any scopes are opened.
    pub fn begin_frame(&self) {
        if !self.is_enabled() {
            return;
        }
        *self.frame_start.lock() = Some(Instant::now());
    }

    /// Signal the end of the current frame. Collects per-thread data, merges
    /// it into a [`ProfileFrame`], and pushes the result into history.
    pub fn end_frame(&self) {
        if !self.is_enabled() {
            return;
        }

        let frame_start = match self.frame_start.lock().take() {
            Some(t) => t,
            None => return,
        };

        let frame_index = self.frame_counter.fetch_add(1, Ordering::Relaxed);
        let frame_duration = frame_start.elapsed();

        // Collect the current thread's roots.
        let local_roots = THREAD_PROFILE.with(|tp| tp.borrow_mut().take_roots());

        // Merge with roots from other threads (if any were submitted).
        let mut all_roots = {
            let mut pending = self.pending_roots.lock();
            let mut merged: Vec<ProfileNode> = Vec::new();
            for thread_roots in pending.drain(..) {
                merged.extend(thread_roots);
            }
            merged
        };
        all_roots.extend(local_roots);

        let frame = ProfileFrame {
            frame_index,
            roots: all_roots,
            frame_duration,
            frame_start,
        };

        let mut history = self.history.lock();
        if history.len() >= self.history_size {
            history.remove(0);
        }
        history.push(frame);
    }

    /// Submit the current thread's profiling data for merging at frame end.
    ///
    /// Call this from worker threads before `end_frame` is called on the main
    /// thread.
    pub fn submit_thread_data(&self) {
        let roots = THREAD_PROFILE.with(|tp| tp.borrow_mut().take_roots());
        if !roots.is_empty() {
            self.pending_roots.lock().push(roots);
        }
    }

    /// Begin a named scope on the current thread.
    pub fn begin_scope(&self, name: &str) {
        if !self.is_enabled() {
            return;
        }
        THREAD_PROFILE.with(|tp| tp.borrow_mut().begin_scope(name));
    }

    /// End the innermost open scope on the current thread.
    pub fn end_scope(&self) {
        if !self.is_enabled() {
            return;
        }
        THREAD_PROFILE.with(|tp| tp.borrow_mut().end_scope());
    }

    /// Create an RAII scope guard. The scope starts immediately and ends when
    /// the guard is dropped.
    pub fn scope(&self, name: &str) -> ScopeGuard<'_> {
        self.begin_scope(name);
        ScopeGuard { profiler: self }
    }

    // -- History queries ---------------------------------------------------

    /// Get the last N frames of profile data (most recent last).
    pub fn get_history(&self, count: usize) -> Vec<ProfileFrame> {
        let history = self.history.lock();
        let start = history.len().saturating_sub(count);
        history[start..].to_vec()
    }

    /// Get the most recent completed frame.
    pub fn get_last_frame(&self) -> Option<ProfileFrame> {
        self.history.lock().last().cloned()
    }

    /// Get average duration for a named scope over the last `n` frames.
    /// If `n` is 0, averages over all available history.
    pub fn get_average(&self, name: &str) -> Duration {
        self.get_average_over(name, 0)
    }

    /// Get average duration for a named scope over the last `n` frames.
    pub fn get_average_over(&self, name: &str, n: usize) -> Duration {
        let history = self.history.lock();
        let frames = if n == 0 || n > history.len() {
            &history[..]
        } else {
            &history[history.len() - n..]
        };

        if frames.is_empty() {
            return Duration::ZERO;
        }

        let mut total = Duration::ZERO;
        let mut count = 0u64;
        for frame in frames {
            if let Some(node) = frame.find_node(name) {
                total += node.duration;
                count += 1;
            }
        }

        if count == 0 {
            Duration::ZERO
        } else {
            total / count as u32
        }
    }

    /// Get the maximum duration for a named scope over all history.
    pub fn get_max(&self, name: &str) -> Duration {
        self.get_max_over(name, 0)
    }

    /// Get the maximum duration for a named scope over the last `n` frames.
    pub fn get_max_over(&self, name: &str, n: usize) -> Duration {
        let history = self.history.lock();
        let frames = if n == 0 || n > history.len() {
            &history[..]
        } else {
            &history[history.len() - n..]
        };

        let mut max = Duration::ZERO;
        for frame in frames {
            if let Some(node) = frame.find_node(name) {
                if node.duration > max {
                    max = node.duration;
                }
            }
        }
        max
    }

    /// Get the minimum duration for a named scope over all history.
    pub fn get_min(&self, name: &str) -> Duration {
        let history = self.history.lock();
        let mut min = Duration::MAX;
        let mut found = false;
        for frame in history.iter() {
            if let Some(node) = frame.find_node(name) {
                if node.duration < min {
                    min = node.duration;
                    found = true;
                }
            }
        }
        if found {
            min
        } else {
            Duration::ZERO
        }
    }

    /// Get the total number of frames recorded.
    pub fn frame_count(&self) -> u64 {
        self.frame_counter.load(Ordering::Relaxed)
    }

    /// Get the number of frames currently in history.
    pub fn history_len(&self) -> usize {
        self.history.lock().len()
    }

    /// Clear all profiling history.
    pub fn clear_history(&self) {
        self.history.lock().clear();
    }

    // -- Reporting ---------------------------------------------------------

    /// Generate a [`ProfileReport`] summarizing the last `n` frames.
    /// If `n` is 0, uses all available history.
    pub fn generate_report(&self, n: usize) -> ProfileReport {
        let history = self.history.lock();
        let frames = if n == 0 || n > history.len() {
            &history[..]
        } else {
            &history[history.len() - n..]
        };

        let frame_count = frames.len();
        if frame_count == 0 {
            return ProfileReport {
                title: "Profiler Report (no data)".into(),
                lines: Vec::new(),
                frame_count: 0,
                avg_frame_time: Duration::ZERO,
                max_frame_time: Duration::ZERO,
            };
        }

        // Frame time statistics.
        let total_frame_time: Duration = frames.iter().map(|f| f.frame_duration).sum();
        let avg_frame_time = total_frame_time / frame_count as u32;
        let max_frame_time = frames
            .iter()
            .map(|f| f.frame_duration)
            .max()
            .unwrap_or(Duration::ZERO);

        // Aggregate per-scope stats across frames.
        let mut scope_stats: HashMap<String, ScopeAggregate> = HashMap::new();
        for frame in frames {
            for node in frame.all_nodes() {
                let entry = scope_stats
                    .entry(node.name.clone())
                    .or_insert_with(|| ScopeAggregate {
                        total_duration: Duration::ZERO,
                        max_duration: Duration::ZERO,
                        total_calls: 0,
                        frame_appearances: 0,
                        depth: node.depth,
                    });
                entry.total_duration += node.duration;
                if node.duration > entry.max_duration {
                    entry.max_duration = node.duration;
                }
                entry.total_calls += node.call_count as u64;
                entry.frame_appearances += 1;
                // Keep the shallowest depth.
                if node.depth < entry.depth {
                    entry.depth = node.depth;
                }
            }
        }

        // Build report lines sorted by average duration (descending).
        let mut lines: Vec<ProfileReportLine> = scope_stats
            .iter()
            .filter(|(_, agg)| {
                agg.total_duration.as_micros() as u64 / agg.frame_appearances.max(1)
                    >= REPORT_MIN_DURATION_US
            })
            .map(|(name, agg)| {
                let avg_dur = agg.total_duration / agg.frame_appearances.max(1) as u32;
                let avg_calls = agg.total_calls as f64 / frame_count as f64;
                let pct =
                    avg_dur.as_secs_f64() / avg_frame_time.as_secs_f64().max(1e-9) * 100.0;
                ProfileReportLine {
                    name: name.clone(),
                    avg_duration: avg_dur,
                    max_duration: agg.max_duration,
                    avg_calls,
                    pct_of_frame: pct,
                    depth: agg.depth,
                }
            })
            .collect();

        lines.sort_by(|a, b| b.avg_duration.cmp(&a.avg_duration));

        ProfileReport {
            title: format!("Profiler Report ({} frames)", frame_count),
            lines,
            frame_count,
            avg_frame_time,
            max_frame_time,
        }
    }

    // -- Chrome Trace Export ------------------------------------------------

    /// Export the last `n` frames as a Chrome Trace Event JSON string.
    ///
    /// The output is compatible with `chrome://tracing` and Perfetto. If `n`
    /// is 0, all history is exported.
    pub fn export_chrome_trace(&self, n: usize) -> String {
        let history = self.history.lock();
        let frames = if n == 0 || n > history.len() {
            &history[..]
        } else {
            &history[history.len() - n..]
        };

        let mut events: Vec<ProfileEvent> = Vec::new();
        let pid = 1u32;
        let tid = 1u32;

        for frame in frames {
            Self::collect_trace_events(&frame.roots, &self.epoch, pid, tid, &mut events);
        }

        // Serialize manually for maximum control over the output.
        let mut out = String::from("{\"traceEvents\":[\n");
        for (i, event) in events.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            out.push_str(&format!(
                "  {{\"name\":\"{}\",\"cat\":\"{}\",\"ph\":\"{}\",\"ts\":{},\"dur\":{},\"pid\":{},\"tid\":{}}}",
                event.name, event.cat, event.ph, event.ts, event.dur, event.pid, event.tid
            ));
        }
        out.push_str("\n]}\n");
        out
    }

    /// Write the Chrome trace JSON to a file.
    pub fn save_chrome_trace(&self, path: &std::path::Path, n: usize) -> std::io::Result<()> {
        let json = self.export_chrome_trace(n);
        let mut file = std::fs::File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Recursively collect trace events from a slice of profile nodes.
    fn collect_trace_events(
        nodes: &[ProfileNode],
        epoch: &Instant,
        pid: u32,
        tid: u32,
        events: &mut Vec<ProfileEvent>,
    ) {
        for node in nodes {
            let ts = node
                .start_time
                .checked_duration_since(*epoch)
                .unwrap_or(Duration::ZERO)
                .as_micros() as u64;
            let dur = node.duration.as_micros() as u64;

            events.push(ProfileEvent {
                name: node.name.clone(),
                cat: "profile".into(),
                ph: "X".into(),
                ts,
                dur,
                pid,
                tid,
            });

            Self::collect_trace_events(&node.children, epoch, pid, tid, events);
        }
    }

    /// Export a compact text summary of the last frame to a string.
    pub fn last_frame_summary(&self) -> String {
        match self.get_last_frame() {
            Some(frame) => format!("{}", frame),
            None => "No profile data available.".into(),
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

// We implement Send + Sync manually because all mutable state is behind
// Mutex or atomics, and the thread_local data is inherently per-thread.
unsafe impl Send for Profiler {}
unsafe impl Sync for Profiler {}

/// Helper struct for aggregating scope stats across frames.
struct ScopeAggregate {
    total_duration: Duration,
    max_duration: Duration,
    total_calls: u64,
    frame_appearances: u64,
    depth: u32,
}

// ---------------------------------------------------------------------------
// ScopeGuard — RAII scope helper
// ---------------------------------------------------------------------------

/// RAII guard that calls [`Profiler::end_scope`] when dropped.
///
/// Created by [`Profiler::scope`].
pub struct ScopeGuard<'a> {
    profiler: &'a Profiler,
}

impl<'a> Drop for ScopeGuard<'a> {
    fn drop(&mut self) {
        self.profiler.end_scope();
    }
}

// ---------------------------------------------------------------------------
// ProfileScope — standalone RAII guard (no profiler reference)
// ---------------------------------------------------------------------------

/// Standalone RAII profile scope that uses thread-local state directly.
///
/// This is lighter weight than [`ScopeGuard`] because it does not require a
/// reference to a [`Profiler`] instance.
pub struct ProfileScope {
    _private: (),
}

impl ProfileScope {
    /// Create a new profile scope with the given name. The timer starts
    /// immediately.
    pub fn new(name: &str) -> Self {
        THREAD_PROFILE.with(|tp| tp.borrow_mut().begin_scope(name));
        Self { _private: () }
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        THREAD_PROFILE.with(|tp| tp.borrow_mut().end_scope());
    }
}

// ---------------------------------------------------------------------------
// profile_scope! macro
// ---------------------------------------------------------------------------

/// Convenience macro that creates a [`ProfileScope`] with the given name.
///
/// The scope lasts until the end of the enclosing block.
///
/// ```ignore
/// fn physics_step() {
///     profile_scope!("PhysicsStep");
///     // ... work ...
///     {
///         profile_scope!("BroadPhase");
///         // ... broad-phase collision ...
///     }
/// }
/// ```
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _profile_guard = $crate::profiler::ProfileScope::new($name);
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_profiling_workflow() {
        let profiler = Profiler::new();

        profiler.begin_frame();
        {
            let _g = profiler.scope("Frame");
            {
                let _g2 = profiler.scope("Physics");
                std::thread::sleep(Duration::from_micros(100));
            }
            {
                let _g2 = profiler.scope("Render");
                std::thread::sleep(Duration::from_micros(50));
            }
        }
        profiler.end_frame();

        assert_eq!(profiler.history_len(), 1);
        let frame = profiler.get_last_frame().unwrap();
        assert_eq!(frame.roots.len(), 1);
        assert_eq!(frame.roots[0].name, "Frame");
        assert_eq!(frame.roots[0].children.len(), 2);
    }

    #[test]
    fn get_average_and_max() {
        let profiler = Profiler::new();

        for _ in 0..5 {
            profiler.begin_frame();
            profiler.begin_scope("Test");
            std::thread::sleep(Duration::from_micros(10));
            profiler.end_scope();
            profiler.end_frame();
        }

        let avg = profiler.get_average("Test");
        assert!(avg > Duration::ZERO);

        let max = profiler.get_max("Test");
        assert!(max >= avg);
    }

    #[test]
    fn disabled_profiler_skips_work() {
        let profiler = Profiler::new();
        profiler.set_enabled(false);

        profiler.begin_frame();
        profiler.begin_scope("Skipped");
        profiler.end_scope();
        profiler.end_frame();

        assert_eq!(profiler.history_len(), 0);
    }

    #[test]
    fn chrome_trace_export() {
        let profiler = Profiler::new();
        profiler.begin_frame();
        profiler.begin_scope("Root");
        profiler.begin_scope("Child");
        profiler.end_scope();
        profiler.end_scope();
        profiler.end_frame();

        let json = profiler.export_chrome_trace(0);
        assert!(json.contains("traceEvents"));
        assert!(json.contains("Root"));
        assert!(json.contains("Child"));
    }

    #[test]
    fn generate_report_works() {
        let profiler = Profiler::new();
        for _ in 0..3 {
            profiler.begin_frame();
            profiler.begin_scope("Expensive");
            std::thread::sleep(Duration::from_micros(50));
            profiler.end_scope();
            profiler.end_frame();
        }

        let report = profiler.generate_report(0);
        assert_eq!(report.frame_count, 3);
        let text = format!("{}", report);
        assert!(text.contains("Expensive"));
    }

    #[test]
    fn profile_node_self_time() {
        let now = Instant::now();
        let child = ProfileNode {
            name: "Child".into(),
            start_time: now,
            duration: Duration::from_millis(3),
            call_count: 1,
            children: Vec::new(),
            depth: 1,
        };
        let parent = ProfileNode {
            name: "Parent".into(),
            start_time: now,
            duration: Duration::from_millis(10),
            call_count: 1,
            children: vec![child],
            depth: 0,
        };
        assert_eq!(parent.self_time(), Duration::from_millis(7));
    }

    #[test]
    fn rolling_history_eviction() {
        let profiler = Profiler::with_history_size(3);
        for _ in 0..5 {
            profiler.begin_frame();
            profiler.end_frame();
        }
        assert_eq!(profiler.history_len(), 3);
    }

    #[test]
    fn clear_history() {
        let profiler = Profiler::new();
        profiler.begin_frame();
        profiler.end_frame();
        assert_eq!(profiler.history_len(), 1);
        profiler.clear_history();
        assert_eq!(profiler.history_len(), 0);
    }

    #[test]
    fn profile_scope_macro() {
        let profiler = Profiler::new();
        profiler.begin_frame();
        {
            profile_scope!("MacroTest");
        }
        profiler.end_frame();

        let frame = profiler.get_last_frame().unwrap();
        assert!(frame.find_node("MacroTest").is_some());
    }
}
