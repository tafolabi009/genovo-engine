//! Script profiler for the Genovo engine scripting runtime.
//!
//! Tracks per-function call counts and timing, builds call graphs, detects
//! hotspots, monitors memory allocation, counts opcode executions, generates
//! profile reports, and estimates profiling overhead.
//!
//! # Usage
//!
//! ```ignore
//! let mut profiler = ScriptProfiler::new();
//! profiler.start();
//!
//! // ... execute scripts ...
//! profiler.enter_function("update", "game.gs", 10);
//! // ... function body ...
//! profiler.exit_function("update");
//!
//! profiler.stop();
//! let report = profiler.generate_report();
//! println!("{}", report);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// FunctionProfile
// ---------------------------------------------------------------------------

/// Profiling data for a single script function.
#[derive(Debug, Clone)]
pub struct FunctionProfile {
    /// Fully-qualified function name.
    pub name: String,
    /// Source file path.
    pub source_file: String,
    /// Line number in source.
    pub source_line: u32,
    /// Number of times this function was called.
    pub call_count: u64,
    /// Total time spent in this function (exclusive of callees).
    pub self_time: Duration,
    /// Total time including callees (inclusive time).
    pub total_time: Duration,
    /// Maximum single-call duration.
    pub max_call_time: Duration,
    /// Minimum single-call duration.
    pub min_call_time: Duration,
    /// Memory allocated during this function's execution (bytes).
    pub memory_allocated: u64,
    /// Memory freed during this function's execution (bytes).
    pub memory_freed: u64,
    /// Number of opcodes executed within this function.
    pub opcodes_executed: u64,
    /// Callers of this function (name -> call count).
    pub callers: HashMap<String, u64>,
    /// Callees of this function (name -> call count).
    pub callees: HashMap<String, u64>,
    /// Recursive call depth.
    pub max_recursion_depth: u32,
    /// Current recursion depth (transient).
    current_depth: u32,
    /// Start time of the current invocation.
    current_start: Option<Instant>,
    /// Accumulated self-time for the current invocation.
    current_self_start: Option<Instant>,
}

impl FunctionProfile {
    /// Create a new profile entry.
    pub fn new(name: &str, source_file: &str, source_line: u32) -> Self {
        Self {
            name: name.to_string(),
            source_file: source_file.to_string(),
            source_line,
            call_count: 0,
            self_time: Duration::ZERO,
            total_time: Duration::ZERO,
            max_call_time: Duration::ZERO,
            min_call_time: Duration::from_secs(u64::MAX),
            memory_allocated: 0,
            memory_freed: 0,
            opcodes_executed: 0,
            callers: HashMap::new(),
            callees: HashMap::new(),
            max_recursion_depth: 0,
            current_depth: 0,
            current_start: None,
            current_self_start: None,
        }
    }

    /// Average call duration (total time / call count).
    pub fn avg_total_time(&self) -> Duration {
        if self.call_count == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.call_count as u32
        }
    }

    /// Average self time per call.
    pub fn avg_self_time(&self) -> Duration {
        if self.call_count == 0 {
            Duration::ZERO
        } else {
            self.self_time / self.call_count as u32
        }
    }

    /// Net memory change (allocated - freed).
    pub fn net_memory(&self) -> i64 {
        self.memory_allocated as i64 - self.memory_freed as i64
    }

    /// Self-time percentage of total program time.
    pub fn self_time_percent(&self, total_program_time: Duration) -> f64 {
        if total_program_time.is_zero() {
            0.0
        } else {
            self.self_time.as_secs_f64() / total_program_time.as_secs_f64() * 100.0
        }
    }
}

impl fmt::Display for FunctionProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} calls, self={:.3}ms, total={:.3}ms, alloc={} bytes",
            self.name,
            self.call_count,
            self.self_time.as_secs_f64() * 1000.0,
            self.total_time.as_secs_f64() * 1000.0,
            self.memory_allocated,
        )
    }
}

// ---------------------------------------------------------------------------
// CallGraphEdge
// ---------------------------------------------------------------------------

/// An edge in the call graph.
#[derive(Debug, Clone)]
pub struct CallGraphEdge {
    /// Caller function name.
    pub caller: String,
    /// Callee function name.
    pub callee: String,
    /// Number of calls along this edge.
    pub call_count: u64,
    /// Total time spent in callee from this caller.
    pub total_time: Duration,
}

impl fmt::Display for CallGraphEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} -> {} ({} calls, {:.3}ms)",
            self.caller,
            self.callee,
            self.call_count,
            self.total_time.as_secs_f64() * 1000.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Hotspot
// ---------------------------------------------------------------------------

/// A detected hotspot in the script code.
#[derive(Debug, Clone)]
pub struct Hotspot {
    /// Function name.
    pub function: String,
    /// Source location.
    pub source: String,
    /// Self-time percentage of total.
    pub self_percent: f64,
    /// Total-time percentage of total.
    pub total_percent: f64,
    /// Call count.
    pub call_count: u64,
    /// Severity: how impactful this hotspot is.
    pub severity: HotspotSeverity,
    /// Suggested optimization.
    pub suggestion: String,
}

/// Severity of a hotspot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HotspotSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for HotspotSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HotspotSeverity::Low => write!(f, "LOW"),
            HotspotSeverity::Medium => write!(f, "MEDIUM"),
            HotspotSeverity::High => write!(f, "HIGH"),
            HotspotSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ---------------------------------------------------------------------------
// OpcodeProfile
// ---------------------------------------------------------------------------

/// Execution counts per opcode.
#[derive(Debug, Clone, Default)]
pub struct OpcodeProfile {
    /// Opcode name -> execution count.
    pub counts: HashMap<String, u64>,
    /// Total opcodes executed.
    pub total: u64,
}

impl OpcodeProfile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an opcode execution.
    pub fn record(&mut self, opcode: &str) {
        *self.counts.entry(opcode.to_string()).or_insert(0) += 1;
        self.total += 1;
    }

    /// Get the count for a specific opcode.
    pub fn count(&self, opcode: &str) -> u64 {
        self.counts.get(opcode).copied().unwrap_or(0)
    }

    /// Get the top N most executed opcodes.
    pub fn top_opcodes(&self, n: usize) -> Vec<(&str, u64)> {
        let mut entries: Vec<_> = self.counts.iter().map(|(k, &v)| (k.as_str(), v)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
        entries
    }

    /// Percentage of total for a given opcode.
    pub fn percentage(&self, opcode: &str) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count(opcode) as f64 / self.total as f64 * 100.0
    }
}

impl fmt::Display for OpcodeProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Opcode Execution Profile ({} total):", self.total)?;
        for (opcode, count) in self.top_opcodes(20) {
            writeln!(
                f,
                "  {:20} {:>10} ({:.1}%)",
                opcode,
                count,
                self.percentage(opcode)
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MemoryTracker
// ---------------------------------------------------------------------------

/// Tracks memory allocations during script execution.
#[derive(Debug, Clone, Default)]
pub struct MemoryTracker {
    /// Total bytes allocated.
    pub total_allocated: u64,
    /// Total bytes freed.
    pub total_freed: u64,
    /// Peak memory usage.
    pub peak_usage: u64,
    /// Current memory usage.
    pub current_usage: u64,
    /// Number of allocation operations.
    pub alloc_count: u64,
    /// Number of free operations.
    pub free_count: u64,
    /// Allocation sizes histogram (bucket -> count).
    pub size_histogram: HashMap<String, u64>,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an allocation.
    pub fn record_alloc(&mut self, bytes: u64) {
        self.total_allocated += bytes;
        self.current_usage += bytes;
        self.alloc_count += 1;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }

        // Histogram bucket.
        let bucket = if bytes <= 16 {
            "0-16"
        } else if bytes <= 64 {
            "17-64"
        } else if bytes <= 256 {
            "65-256"
        } else if bytes <= 1024 {
            "257-1024"
        } else if bytes <= 4096 {
            "1025-4096"
        } else {
            "4097+"
        };
        *self.size_histogram.entry(bucket.to_string()).or_insert(0) += 1;
    }

    /// Record a deallocation.
    pub fn record_free(&mut self, bytes: u64) {
        self.total_freed += bytes;
        self.current_usage = self.current_usage.saturating_sub(bytes);
        self.free_count += 1;
    }

    /// Net memory change.
    pub fn net_usage(&self) -> i64 {
        self.total_allocated as i64 - self.total_freed as i64
    }

    /// Average allocation size.
    pub fn avg_alloc_size(&self) -> f64 {
        if self.alloc_count == 0 {
            0.0
        } else {
            self.total_allocated as f64 / self.alloc_count as f64
        }
    }
}

impl fmt::Display for MemoryTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Memory Profile:")?;
        writeln!(f, "  allocated: {} bytes ({} ops)", self.total_allocated, self.alloc_count)?;
        writeln!(f, "  freed:     {} bytes ({} ops)", self.total_freed, self.free_count)?;
        writeln!(f, "  peak:      {} bytes", self.peak_usage)?;
        writeln!(f, "  current:   {} bytes", self.current_usage)?;
        writeln!(f, "  avg alloc: {:.0} bytes", self.avg_alloc_size())?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ProfilerState
// ---------------------------------------------------------------------------

/// State of the profiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilerState {
    /// Profiler is idle (not collecting data).
    Idle,
    /// Profiler is actively collecting data.
    Running,
    /// Profiler is paused (data preserved but not collecting).
    Paused,
}

// ---------------------------------------------------------------------------
// ScriptProfiler
// ---------------------------------------------------------------------------

/// The main script profiler that tracks function execution, memory, opcodes,
/// and generates reports.
pub struct ScriptProfiler {
    /// Per-function profiles.
    functions: HashMap<String, FunctionProfile>,
    /// Call stack (function names).
    call_stack: Vec<String>,
    /// Opcode execution profile.
    opcodes: OpcodeProfile,
    /// Memory tracker.
    memory: MemoryTracker,
    /// Profiler state.
    state: ProfilerState,
    /// When profiling started.
    start_time: Option<Instant>,
    /// Total profiling duration.
    total_duration: Duration,
    /// Overhead: time spent in profiler itself.
    overhead: Duration,
    /// Number of enter/exit calls.
    total_enter_exit: u64,
}

impl ScriptProfiler {
    /// Create a new, idle profiler.
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            call_stack: Vec::new(),
            opcodes: OpcodeProfile::new(),
            memory: MemoryTracker::new(),
            state: ProfilerState::Idle,
            start_time: None,
            total_duration: Duration::ZERO,
            overhead: Duration::ZERO,
            total_enter_exit: 0,
        }
    }

    /// Start profiling.
    pub fn start(&mut self) {
        self.state = ProfilerState::Running;
        self.start_time = Some(Instant::now());
    }

    /// Stop profiling.
    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.total_duration += start.elapsed();
        }
        self.state = ProfilerState::Idle;
    }

    /// Pause profiling (preserves data).
    pub fn pause(&mut self) {
        if self.state == ProfilerState::Running {
            if let Some(start) = self.start_time.take() {
                self.total_duration += start.elapsed();
            }
            self.state = ProfilerState::Paused;
        }
    }

    /// Resume profiling.
    pub fn resume(&mut self) {
        if self.state == ProfilerState::Paused {
            self.start_time = Some(Instant::now());
            self.state = ProfilerState::Running;
        }
    }

    /// Reset all profiling data.
    pub fn reset(&mut self) {
        self.functions.clear();
        self.call_stack.clear();
        self.opcodes = OpcodeProfile::new();
        self.memory = MemoryTracker::new();
        self.state = ProfilerState::Idle;
        self.start_time = None;
        self.total_duration = Duration::ZERO;
        self.overhead = Duration::ZERO;
        self.total_enter_exit = 0;
    }

    /// Returns the profiler state.
    pub fn state(&self) -> ProfilerState {
        self.state
    }

    /// Returns `true` if the profiler is actively collecting data.
    pub fn is_running(&self) -> bool {
        self.state == ProfilerState::Running
    }

    // -- Function tracking --

    /// Record entering a function.
    pub fn enter_function(&mut self, name: &str, source_file: &str, source_line: u32) {
        if self.state != ProfilerState::Running {
            return;
        }

        let overhead_start = Instant::now();
        self.total_enter_exit += 1;

        // Pause self-time timer for the current top-of-stack function.
        if let Some(caller_name) = self.call_stack.last() {
            if let Some(caller) = self.functions.get_mut(caller_name) {
                if let Some(self_start) = caller.current_self_start.take() {
                    caller.self_time += self_start.elapsed();
                }
            }
        }

        // Record caller relationship.
        let caller_name = self.call_stack.last().cloned();

        let profile = self
            .functions
            .entry(name.to_string())
            .or_insert_with(|| FunctionProfile::new(name, source_file, source_line));

        profile.call_count += 1;
        profile.current_depth += 1;
        if profile.current_depth > profile.max_recursion_depth {
            profile.max_recursion_depth = profile.current_depth;
        }

        let now = Instant::now();
        profile.current_start = Some(now);
        profile.current_self_start = Some(now);

        if let Some(ref caller) = caller_name {
            *profile.callers.entry(caller.clone()).or_insert(0) += 1;
            // Also record callee in the caller's profile.
            if let Some(caller_profile) = self.functions.get_mut(caller) {
                *caller_profile.callees.entry(name.to_string()).or_insert(0) += 1;
            }
        }

        self.call_stack.push(name.to_string());

        self.overhead += overhead_start.elapsed();
    }

    /// Record exiting a function.
    pub fn exit_function(&mut self, name: &str) {
        if self.state != ProfilerState::Running {
            return;
        }

        let overhead_start = Instant::now();
        self.total_enter_exit += 1;

        // Pop from call stack.
        if let Some(top) = self.call_stack.last() {
            if top == name {
                self.call_stack.pop();
            }
        }

        if let Some(profile) = self.functions.get_mut(name) {
            // Record self-time.
            if let Some(self_start) = profile.current_self_start.take() {
                profile.self_time += self_start.elapsed();
            }

            // Record total time.
            if let Some(start) = profile.current_start.take() {
                let call_duration = start.elapsed();
                profile.total_time += call_duration;

                // Update min/max.
                if call_duration > profile.max_call_time {
                    profile.max_call_time = call_duration;
                }
                if call_duration < profile.min_call_time {
                    profile.min_call_time = call_duration;
                }
            }

            profile.current_depth = profile.current_depth.saturating_sub(1);
        }

        // Resume self-time timer for the new top-of-stack.
        if let Some(caller_name) = self.call_stack.last() {
            if let Some(caller) = self.functions.get_mut(caller_name) {
                caller.current_self_start = Some(Instant::now());
            }
        }

        self.overhead += overhead_start.elapsed();
    }

    /// Record an opcode execution.
    pub fn record_opcode(&mut self, opcode: &str) {
        if self.state != ProfilerState::Running {
            return;
        }
        self.opcodes.record(opcode);

        // Credit opcodes to current function.
        if let Some(func_name) = self.call_stack.last() {
            if let Some(profile) = self.functions.get_mut(func_name) {
                profile.opcodes_executed += 1;
            }
        }
    }

    /// Record a memory allocation.
    pub fn record_alloc(&mut self, bytes: u64) {
        if self.state != ProfilerState::Running {
            return;
        }
        self.memory.record_alloc(bytes);

        if let Some(func_name) = self.call_stack.last() {
            if let Some(profile) = self.functions.get_mut(func_name) {
                profile.memory_allocated += bytes;
            }
        }
    }

    /// Record a memory deallocation.
    pub fn record_free(&mut self, bytes: u64) {
        if self.state != ProfilerState::Running {
            return;
        }
        self.memory.record_free(bytes);

        if let Some(func_name) = self.call_stack.last() {
            if let Some(profile) = self.functions.get_mut(func_name) {
                profile.memory_freed += bytes;
            }
        }
    }

    // -- Analysis --

    /// Detect hotspot functions.
    pub fn detect_hotspots(&self) -> Vec<Hotspot> {
        let total_time = self.total_duration;
        let mut hotspots: Vec<Hotspot> = self
            .functions
            .values()
            .filter(|p| p.call_count > 0)
            .map(|p| {
                let self_pct = p.self_time_percent(total_time);
                let total_pct = if total_time.is_zero() {
                    0.0
                } else {
                    p.total_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
                };

                let severity = if self_pct > 30.0 {
                    HotspotSeverity::Critical
                } else if self_pct > 15.0 {
                    HotspotSeverity::High
                } else if self_pct > 5.0 {
                    HotspotSeverity::Medium
                } else {
                    HotspotSeverity::Low
                };

                let suggestion = if p.call_count > 10000 {
                    "Consider caching results or reducing call frequency".to_string()
                } else if p.memory_allocated > 1024 * 1024 {
                    "High memory allocation; consider object pooling".to_string()
                } else if p.max_recursion_depth > 10 {
                    "Deep recursion; consider iterative approach".to_string()
                } else {
                    "Review algorithm efficiency".to_string()
                };

                Hotspot {
                    function: p.name.clone(),
                    source: format!("{}:{}", p.source_file, p.source_line),
                    self_percent: self_pct,
                    total_percent: total_pct,
                    call_count: p.call_count,
                    severity,
                    suggestion,
                }
            })
            .collect();

        // Sort by self-time percentage (descending).
        hotspots.sort_by(|a, b| b.self_percent.partial_cmp(&a.self_percent).unwrap());
        hotspots
    }

    /// Build the call graph edges.
    pub fn call_graph(&self) -> Vec<CallGraphEdge> {
        let mut edges = Vec::new();
        for profile in self.functions.values() {
            for (callee, &count) in &profile.callees {
                let callee_time = self
                    .functions
                    .get(callee)
                    .map(|p| p.total_time)
                    .unwrap_or(Duration::ZERO);
                edges.push(CallGraphEdge {
                    caller: profile.name.clone(),
                    callee: callee.clone(),
                    call_count: count,
                    total_time: callee_time,
                });
            }
        }
        edges
    }

    /// Estimated profiling overhead as a percentage of total time.
    pub fn overhead_percent(&self) -> f64 {
        if self.total_duration.is_zero() {
            0.0
        } else {
            self.overhead.as_secs_f64() / self.total_duration.as_secs_f64() * 100.0
        }
    }

    /// Generate a full-text profile report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Script Profile Report ===\n\n");

        // Summary.
        report.push_str(&format!(
            "Total duration: {:.3}ms\n",
            self.total_duration.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!("Functions profiled: {}\n", self.functions.len()));
        report.push_str(&format!("Total opcodes: {}\n", self.opcodes.total));
        report.push_str(&format!(
            "Profiler overhead: {:.3}ms ({:.1}%)\n\n",
            self.overhead.as_secs_f64() * 1000.0,
            self.overhead_percent()
        ));

        // Top functions by self-time.
        report.push_str("--- Top Functions by Self-Time ---\n");
        let mut profiles: Vec<&FunctionProfile> = self.functions.values().collect();
        profiles.sort_by(|a, b| b.self_time.cmp(&a.self_time));
        for (i, p) in profiles.iter().take(20).enumerate() {
            report.push_str(&format!(
                "  {:2}. {:30} {:>8} calls  self={:>8.3}ms  total={:>8.3}ms  ({:.1}%)\n",
                i + 1,
                p.name,
                p.call_count,
                p.self_time.as_secs_f64() * 1000.0,
                p.total_time.as_secs_f64() * 1000.0,
                p.self_time_percent(self.total_duration),
            ));
        }

        // Hotspots.
        let hotspots = self.detect_hotspots();
        let critical: Vec<_> = hotspots
            .iter()
            .filter(|h| h.severity >= HotspotSeverity::Medium)
            .collect();
        if !critical.is_empty() {
            report.push_str("\n--- Hotspots ---\n");
            for h in critical {
                report.push_str(&format!(
                    "  [{}] {} ({}) - {:.1}% self-time - {}\n",
                    h.severity, h.function, h.source, h.self_percent, h.suggestion,
                ));
            }
        }

        // Memory.
        report.push_str(&format!("\n{}", self.memory));

        // Opcodes.
        report.push_str(&format!("\n{}", self.opcodes));

        report
    }

    // -- Accessors --

    /// Get profile data for a specific function.
    pub fn get_function(&self, name: &str) -> Option<&FunctionProfile> {
        self.functions.get(name)
    }

    /// Get all function profiles.
    pub fn all_functions(&self) -> &HashMap<String, FunctionProfile> {
        &self.functions
    }

    /// Get the opcode profile.
    pub fn opcodes(&self) -> &OpcodeProfile {
        &self.opcodes
    }

    /// Get the memory tracker.
    pub fn memory(&self) -> &MemoryTracker {
        &self.memory
    }

    /// Get total profiling duration.
    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    /// Get the current call stack.
    pub fn call_stack(&self) -> &[String] {
        &self.call_stack
    }

    /// Number of profiled functions.
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }
}

impl Default for ScriptProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ScriptProfiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScriptProfiler")
            .field("state", &self.state)
            .field("functions", &self.functions.len())
            .field("opcodes", &self.opcodes.total)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_profiling() {
        let mut profiler = ScriptProfiler::new();
        profiler.start();

        profiler.enter_function("main", "game.gs", 1);
        thread::sleep(Duration::from_millis(10));
        profiler.exit_function("main");

        profiler.stop();

        let main_profile = profiler.get_function("main").unwrap();
        assert_eq!(main_profile.call_count, 1);
        assert!(main_profile.total_time >= Duration::from_millis(5));
    }

    #[test]
    fn test_nested_calls() {
        let mut profiler = ScriptProfiler::new();
        profiler.start();

        profiler.enter_function("outer", "test.gs", 1);
        profiler.enter_function("inner", "test.gs", 10);
        profiler.exit_function("inner");
        profiler.exit_function("outer");

        profiler.stop();

        let outer = profiler.get_function("outer").unwrap();
        let inner = profiler.get_function("inner").unwrap();
        assert_eq!(outer.callees.get("inner"), Some(&1));
        assert_eq!(inner.callers.get("outer"), Some(&1));
    }

    #[test]
    fn test_opcode_tracking() {
        let mut profiler = ScriptProfiler::new();
        profiler.start();

        profiler.record_opcode("LOAD");
        profiler.record_opcode("ADD");
        profiler.record_opcode("LOAD");
        profiler.record_opcode("STORE");

        profiler.stop();

        assert_eq!(profiler.opcodes().total, 4);
        assert_eq!(profiler.opcodes().count("LOAD"), 2);
        assert_eq!(profiler.opcodes().count("ADD"), 1);
    }

    #[test]
    fn test_memory_tracking() {
        let mut profiler = ScriptProfiler::new();
        profiler.start();

        profiler.record_alloc(1024);
        profiler.record_alloc(512);
        profiler.record_free(256);

        profiler.stop();

        assert_eq!(profiler.memory().total_allocated, 1536);
        assert_eq!(profiler.memory().total_freed, 256);
        assert_eq!(profiler.memory().current_usage, 1280);
    }

    #[test]
    fn test_hotspot_detection() {
        let mut profiler = ScriptProfiler::new();
        profiler.start();

        for _ in 0..1000 {
            profiler.enter_function("hot_func", "game.gs", 5);
            profiler.exit_function("hot_func");
        }
        profiler.enter_function("cold_func", "game.gs", 20);
        profiler.exit_function("cold_func");

        profiler.stop();

        let hotspots = profiler.detect_hotspots();
        assert!(!hotspots.is_empty());
        assert_eq!(hotspots[0].function, "hot_func");
    }

    #[test]
    fn test_call_graph() {
        let mut profiler = ScriptProfiler::new();
        profiler.start();

        profiler.enter_function("A", "test.gs", 1);
        profiler.enter_function("B", "test.gs", 10);
        profiler.enter_function("C", "test.gs", 20);
        profiler.exit_function("C");
        profiler.exit_function("B");
        profiler.exit_function("A");

        profiler.stop();

        let edges = profiler.call_graph();
        assert!(edges.len() >= 2);
    }

    #[test]
    fn test_report_generation() {
        let mut profiler = ScriptProfiler::new();
        profiler.start();

        profiler.enter_function("update", "game.gs", 1);
        profiler.record_opcode("LOAD");
        profiler.record_alloc(100);
        profiler.exit_function("update");

        profiler.stop();

        let report = profiler.generate_report();
        assert!(report.contains("Script Profile Report"));
        assert!(report.contains("update"));
    }

    #[test]
    fn test_pause_resume() {
        let mut profiler = ScriptProfiler::new();
        profiler.start();
        assert_eq!(profiler.state(), ProfilerState::Running);

        profiler.pause();
        assert_eq!(profiler.state(), ProfilerState::Paused);

        profiler.resume();
        assert_eq!(profiler.state(), ProfilerState::Running);

        profiler.stop();
        assert_eq!(profiler.state(), ProfilerState::Idle);
    }
}
