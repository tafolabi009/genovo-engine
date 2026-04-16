//! Memory profiler for tracking allocations, per-system memory budgets, and
//! detecting leaks in the Genovo engine.
//!
//! The [`MemoryProfiler`] tracks allocations by category (render, physics,
//! audio, ECS, etc.) and provides per-frame snapshots, peak tracking, and
//! leak detection.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// MemoryCategory
// ---------------------------------------------------------------------------

/// Categories for tagging memory allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryCategory {
    /// General / uncategorized allocations.
    General,
    /// Rendering subsystem (textures, buffers, shaders).
    Render,
    /// Physics simulation (rigid bodies, collision shapes).
    Physics,
    /// Audio subsystem (samples, buffers, streams).
    Audio,
    /// Entity-Component-System storage.
    Ecs,
    /// Scene graph and transforms.
    Scene,
    /// Animation data (skeletons, clips, blend trees).
    Animation,
    /// Asset loading and caching.
    Assets,
    /// Scripting runtime.
    Scripting,
    /// Networking buffers.
    Networking,
    /// AI subsystem (navmesh, behavior trees).
    Ai,
    /// UI elements and layout.
    Ui,
    /// Debug / profiling overhead.
    Debug,
    /// Temporary / scratch allocations.
    Temporary,
    /// String storage.
    Strings,
}

impl MemoryCategory {
    /// All category variants for iteration.
    pub const ALL: &'static [MemoryCategory] = &[
        MemoryCategory::General,
        MemoryCategory::Render,
        MemoryCategory::Physics,
        MemoryCategory::Audio,
        MemoryCategory::Ecs,
        MemoryCategory::Scene,
        MemoryCategory::Animation,
        MemoryCategory::Assets,
        MemoryCategory::Scripting,
        MemoryCategory::Networking,
        MemoryCategory::Ai,
        MemoryCategory::Ui,
        MemoryCategory::Debug,
        MemoryCategory::Temporary,
        MemoryCategory::Strings,
    ];

    /// Human-readable label for this category.
    pub fn label(&self) -> &'static str {
        match self {
            MemoryCategory::General => "General",
            MemoryCategory::Render => "Render",
            MemoryCategory::Physics => "Physics",
            MemoryCategory::Audio => "Audio",
            MemoryCategory::Ecs => "ECS",
            MemoryCategory::Scene => "Scene",
            MemoryCategory::Animation => "Animation",
            MemoryCategory::Assets => "Assets",
            MemoryCategory::Scripting => "Scripting",
            MemoryCategory::Networking => "Networking",
            MemoryCategory::Ai => "AI",
            MemoryCategory::Ui => "UI",
            MemoryCategory::Debug => "Debug",
            MemoryCategory::Temporary => "Temporary",
            MemoryCategory::Strings => "Strings",
        }
    }
}

impl fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// AllocatorStats
// ---------------------------------------------------------------------------

/// Statistics for a single allocator / memory category.
#[derive(Debug, Clone)]
pub struct AllocatorStats {
    /// Current allocated bytes.
    pub allocated: u64,
    /// Peak allocated bytes (high-water mark).
    pub peak: u64,
    /// Total number of active allocations.
    pub allocation_count: u64,
    /// Total number of allocations ever made.
    pub total_allocations: u64,
    /// Total number of deallocations ever made.
    pub total_deallocations: u64,
    /// Total bytes allocated over the lifetime.
    pub total_bytes_allocated: u64,
    /// Total bytes freed over the lifetime.
    pub total_bytes_freed: u64,
}

impl AllocatorStats {
    /// Create zero-initialized stats.
    pub fn new() -> Self {
        Self {
            allocated: 0,
            peak: 0,
            allocation_count: 0,
            total_allocations: 0,
            total_deallocations: 0,
            total_bytes_allocated: 0,
            total_bytes_freed: 0,
        }
    }

    /// Record an allocation of `size` bytes.
    pub fn record_alloc(&mut self, size: u64) {
        self.allocated += size;
        self.allocation_count += 1;
        self.total_allocations += 1;
        self.total_bytes_allocated += size;
        if self.allocated > self.peak {
            self.peak = self.allocated;
        }
    }

    /// Record a deallocation of `size` bytes.
    pub fn record_dealloc(&mut self, size: u64) {
        self.allocated = self.allocated.saturating_sub(size);
        self.allocation_count = self.allocation_count.saturating_sub(1);
        self.total_deallocations += 1;
        self.total_bytes_freed += size;
    }

    /// Reset all stats to zero.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Check if there are potential leaks (allocations > deallocations).
    pub fn potential_leaks(&self) -> u64 {
        self.total_allocations.saturating_sub(self.total_deallocations)
    }
}

impl Default for AllocatorStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AllocatorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (peak: {}, allocs: {}, active: {})",
            format_bytes(self.allocated),
            format_bytes(self.peak),
            self.total_allocations,
            self.allocation_count,
        )
    }
}

// ---------------------------------------------------------------------------
// MemorySnapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of memory usage.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Frame index when this snapshot was taken.
    pub frame_index: u64,
    /// Timestamp.
    pub timestamp: Instant,
    /// Total allocated memory.
    pub total_allocated: u64,
    /// Per-category breakdown.
    pub categories: HashMap<MemoryCategory, u64>,
}

// ---------------------------------------------------------------------------
// LargeAllocation
// ---------------------------------------------------------------------------

/// Record of a single large allocation (above threshold).
#[derive(Debug, Clone)]
pub struct LargeAllocation {
    /// Size in bytes.
    pub size: u64,
    /// Category.
    pub category: MemoryCategory,
    /// Frame index.
    pub frame_index: u64,
    /// Optional label.
    pub label: String,
    /// Timestamp.
    pub timestamp: Instant,
}

// ---------------------------------------------------------------------------
// LeakCheckResult
// ---------------------------------------------------------------------------

/// Result of a leak detection check.
#[derive(Debug, Clone)]
pub struct LeakCheckResult {
    /// Whether any leaks were detected.
    pub has_leaks: bool,
    /// Per-category leak info.
    pub category_leaks: Vec<(MemoryCategory, u64)>,
    /// Total leaked allocation count.
    pub total_leaked_count: u64,
    /// Total leaked bytes (approximate).
    pub total_leaked_bytes: u64,
}

impl fmt::Display for LeakCheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.has_leaks {
            return write!(f, "No memory leaks detected.");
        }
        writeln!(
            f,
            "MEMORY LEAKS DETECTED: {} allocations, ~{}",
            self.total_leaked_count,
            format_bytes(self.total_leaked_bytes),
        )?;
        for (cat, count) in &self.category_leaks {
            if *count > 0 {
                writeln!(f, "  {}: {} leaked allocations", cat, count)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MemoryReport
// ---------------------------------------------------------------------------

/// Formatted text report of memory usage.
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Report title.
    pub title: String,
    /// Total allocated memory.
    pub total_allocated: u64,
    /// Total peak memory.
    pub total_peak: u64,
    /// Per-category stats.
    pub categories: Vec<(MemoryCategory, AllocatorStats)>,
    /// Large allocations (if any).
    pub large_allocations: Vec<LargeAllocation>,
    /// Frame index.
    pub frame_index: u64,
}

impl fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== {} ===", self.title)?;
        writeln!(f, "Frame: {}", self.frame_index)?;
        writeln!(
            f,
            "Total Allocated: {} (Peak: {})",
            format_bytes(self.total_allocated),
            format_bytes(self.total_peak),
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "{:<15} {:>12} {:>12} {:>10} {:>10}",
            "Category", "Current", "Peak", "Allocs", "Active"
        )?;
        writeln!(f, "{}", "-".repeat(62))?;
        for (cat, stats) in &self.categories {
            if stats.total_allocations > 0 {
                writeln!(
                    f,
                    "{:<15} {:>12} {:>12} {:>10} {:>10}",
                    cat.label(),
                    format_bytes(stats.allocated),
                    format_bytes(stats.peak),
                    stats.total_allocations,
                    stats.allocation_count,
                )?;
            }
        }

        if !self.large_allocations.is_empty() {
            writeln!(f)?;
            writeln!(f, "Large Allocations ({}):", self.large_allocations.len())?;
            for alloc in &self.large_allocations {
                writeln!(
                    f,
                    "  {} — {} [{}] (frame {})",
                    alloc.label,
                    format_bytes(alloc.size),
                    alloc.category,
                    alloc.frame_index,
                )?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MemoryProfiler
// ---------------------------------------------------------------------------

/// Memory profiler that tracks per-category allocations, peak usage,
/// allocation timelines, and large allocation logging.
pub struct MemoryProfiler {
    /// Per-category stats.
    stats: Mutex<HashMap<MemoryCategory, AllocatorStats>>,
    /// Timeline of memory snapshots.
    timeline: Mutex<Vec<MemorySnapshot>>,
    /// Large allocation log.
    large_allocations: Mutex<Vec<LargeAllocation>>,
    /// Threshold for logging large allocations (bytes).
    large_alloc_threshold: AtomicU64,
    /// How many frames between automatic snapshots.
    snapshot_interval: u64,
    /// Current frame index.
    frame_counter: AtomicU64,
    /// Maximum number of snapshots to retain.
    max_snapshots: usize,
    /// Maximum number of large allocations to retain.
    max_large_allocations: usize,
    /// Whether the profiler is enabled.
    enabled: bool,
    /// Saved baseline for leak detection.
    baseline: Mutex<Option<HashMap<MemoryCategory, AllocatorStats>>>,
}

impl MemoryProfiler {
    /// Create a new memory profiler with default settings.
    pub fn new() -> Self {
        let mut stats = HashMap::new();
        for cat in MemoryCategory::ALL {
            stats.insert(*cat, AllocatorStats::new());
        }
        Self {
            stats: Mutex::new(stats),
            timeline: Mutex::new(Vec::new()),
            large_allocations: Mutex::new(Vec::new()),
            large_alloc_threshold: AtomicU64::new(1024 * 1024), // 1 MB default
            snapshot_interval: 60,
            frame_counter: AtomicU64::new(0),
            max_snapshots: 600,
            max_large_allocations: 200,
            enabled: true,
            baseline: Mutex::new(None),
        }
    }

    /// Enable or disable the memory profiler.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the profiler is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set the threshold (in bytes) for logging large allocations.
    pub fn set_large_alloc_threshold(&self, bytes: u64) {
        self.large_alloc_threshold.store(bytes, Ordering::Relaxed);
    }

    /// Get the large allocation threshold.
    pub fn large_alloc_threshold(&self) -> u64 {
        self.large_alloc_threshold.load(Ordering::Relaxed)
    }

    /// Set the snapshot interval (in frames).
    pub fn set_snapshot_interval(&mut self, frames: u64) {
        self.snapshot_interval = frames;
    }

    // -- Allocation tracking ------------------------------------------------

    /// Record an allocation.
    pub fn record_alloc(&self, category: MemoryCategory, size: u64) {
        self.record_alloc_labeled(category, size, "");
    }

    /// Record an allocation with a label.
    pub fn record_alloc_labeled(&self, category: MemoryCategory, size: u64, label: &str) {
        if !self.enabled {
            return;
        }

        {
            let mut stats = self.stats.lock();
            let entry = stats
                .entry(category)
                .or_insert_with(AllocatorStats::new);
            entry.record_alloc(size);
        }

        // Log large allocations.
        let threshold = self.large_alloc_threshold.load(Ordering::Relaxed);
        if size >= threshold {
            let frame = self.frame_counter.load(Ordering::Relaxed);
            let alloc = LargeAllocation {
                size,
                category,
                frame_index: frame,
                label: if label.is_empty() {
                    format!("{}:{}", category, size)
                } else {
                    label.to_string()
                },
                timestamp: Instant::now(),
            };
            log::warn!(
                "Large allocation: {} ({}) [{}]",
                format_bytes(size),
                category,
                alloc.label,
            );
            let mut large = self.large_allocations.lock();
            if large.len() >= self.max_large_allocations {
                large.remove(0);
            }
            large.push(alloc);
        }
    }

    /// Record a deallocation.
    pub fn record_dealloc(&self, category: MemoryCategory, size: u64) {
        if !self.enabled {
            return;
        }
        let mut stats = self.stats.lock();
        let entry = stats
            .entry(category)
            .or_insert_with(AllocatorStats::new);
        entry.record_dealloc(size);
    }

    // -- Frame management ---------------------------------------------------

    /// Call once per frame. Takes periodic snapshots.
    pub fn end_frame(&self) {
        if !self.enabled {
            return;
        }
        let frame = self.frame_counter.fetch_add(1, Ordering::Relaxed);

        if self.snapshot_interval > 0 && frame % self.snapshot_interval == 0 {
            self.take_snapshot(frame);
        }
    }

    /// Take a memory snapshot at the current moment.
    pub fn take_snapshot(&self, frame_index: u64) {
        let stats = self.stats.lock();
        let mut total = 0u64;
        let mut categories = HashMap::new();
        for (cat, s) in stats.iter() {
            categories.insert(*cat, s.allocated);
            total += s.allocated;
        }

        let snapshot = MemorySnapshot {
            frame_index,
            timestamp: Instant::now(),
            total_allocated: total,
            categories,
        };

        let mut timeline = self.timeline.lock();
        if timeline.len() >= self.max_snapshots {
            timeline.remove(0);
        }
        timeline.push(snapshot);
    }

    // -- Queries ------------------------------------------------------------

    /// Get the current stats for a category.
    pub fn get_stats(&self, category: MemoryCategory) -> AllocatorStats {
        self.stats
            .lock()
            .get(&category)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all category stats.
    pub fn get_all_stats(&self) -> HashMap<MemoryCategory, AllocatorStats> {
        self.stats.lock().clone()
    }

    /// Get total allocated memory across all categories.
    pub fn total_allocated(&self) -> u64 {
        self.stats.lock().values().map(|s| s.allocated).sum()
    }

    /// Get total peak memory across all categories.
    pub fn total_peak(&self) -> u64 {
        self.stats.lock().values().map(|s| s.peak).sum()
    }

    /// Get the timeline of memory snapshots.
    pub fn get_timeline(&self) -> Vec<MemorySnapshot> {
        self.timeline.lock().clone()
    }

    /// Get the list of logged large allocations.
    pub fn get_large_allocations(&self) -> Vec<LargeAllocation> {
        self.large_allocations.lock().clone()
    }

    /// Get the current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_counter.load(Ordering::Relaxed)
    }

    // -- Leak detection -----------------------------------------------------

    /// Save the current state as a baseline for leak detection.
    pub fn save_baseline(&self) {
        let stats = self.stats.lock();
        let baseline: HashMap<MemoryCategory, AllocatorStats> = stats.clone();
        *self.baseline.lock() = Some(baseline);
    }

    /// Compare current state against the saved baseline to detect leaks.
    pub fn check_leaks(&self) -> LeakCheckResult {
        let current = self.stats.lock();
        let baseline = self.baseline.lock();

        let baseline = match baseline.as_ref() {
            Some(b) => b,
            None => {
                return LeakCheckResult {
                    has_leaks: false,
                    category_leaks: Vec::new(),
                    total_leaked_count: 0,
                    total_leaked_bytes: 0,
                };
            }
        };

        let mut category_leaks = Vec::new();
        let mut total_count = 0u64;
        let mut total_bytes = 0u64;

        for cat in MemoryCategory::ALL {
            let current_stats = current.get(cat).cloned().unwrap_or_default();
            let baseline_stats = baseline.get(cat).cloned().unwrap_or_default();

            let leaked_count = current_stats
                .allocation_count
                .saturating_sub(baseline_stats.allocation_count);
            let leaked_bytes = current_stats
                .allocated
                .saturating_sub(baseline_stats.allocated);

            if leaked_count > 0 {
                category_leaks.push((*cat, leaked_count));
                total_count += leaked_count;
                total_bytes += leaked_bytes;
            }
        }

        LeakCheckResult {
            has_leaks: total_count > 0,
            category_leaks,
            total_leaked_count: total_count,
            total_leaked_bytes: total_bytes,
        }
    }

    // -- Reporting ----------------------------------------------------------

    /// Generate a memory report.
    pub fn generate_report(&self) -> MemoryReport {
        let stats = self.stats.lock();
        let mut categories: Vec<(MemoryCategory, AllocatorStats)> = stats
            .iter()
            .map(|(cat, s)| (*cat, s.clone()))
            .collect();
        categories.sort_by(|a, b| b.1.allocated.cmp(&a.1.allocated));

        let total_allocated: u64 = stats.values().map(|s| s.allocated).sum();
        let total_peak: u64 = stats.values().map(|s| s.peak).sum();
        let frame = self.frame_counter.load(Ordering::Relaxed);
        let large = self.large_allocations.lock().clone();

        MemoryReport {
            title: "Memory Profile Report".into(),
            total_allocated,
            total_peak,
            categories,
            large_allocations: large,
            frame_index: frame,
        }
    }

    /// Reset all stats.
    pub fn reset(&self) {
        let mut stats = self.stats.lock();
        for s in stats.values_mut() {
            s.reset();
        }
        self.timeline.lock().clear();
        self.large_allocations.lock().clear();
        *self.baseline.lock() = None;
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Format a byte count as a human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_stats_basic() {
        let mut stats = AllocatorStats::new();
        stats.record_alloc(1024);
        assert_eq!(stats.allocated, 1024);
        assert_eq!(stats.peak, 1024);
        assert_eq!(stats.allocation_count, 1);

        stats.record_alloc(2048);
        assert_eq!(stats.allocated, 3072);
        assert_eq!(stats.peak, 3072);

        stats.record_dealloc(1024);
        assert_eq!(stats.allocated, 2048);
        assert_eq!(stats.peak, 3072); // peak unchanged
        assert_eq!(stats.allocation_count, 1);
    }

    #[test]
    fn memory_profiler_categories() {
        let profiler = MemoryProfiler::new();
        profiler.record_alloc(MemoryCategory::Render, 1024 * 1024);
        profiler.record_alloc(MemoryCategory::Physics, 512 * 1024);

        let render_stats = profiler.get_stats(MemoryCategory::Render);
        assert_eq!(render_stats.allocated, 1024 * 1024);

        let physics_stats = profiler.get_stats(MemoryCategory::Physics);
        assert_eq!(physics_stats.allocated, 512 * 1024);

        let total = profiler.total_allocated();
        assert_eq!(total, 1024 * 1024 + 512 * 1024);
    }

    #[test]
    fn large_allocation_logging() {
        let profiler = MemoryProfiler::new();
        profiler.set_large_alloc_threshold(1024);
        profiler.record_alloc(MemoryCategory::Render, 2048);

        let large = profiler.get_large_allocations();
        assert_eq!(large.len(), 1);
        assert_eq!(large[0].size, 2048);
    }

    #[test]
    fn leak_detection() {
        let profiler = MemoryProfiler::new();
        profiler.record_alloc(MemoryCategory::Render, 1024);
        profiler.save_baseline();

        // Allocate more without freeing.
        profiler.record_alloc(MemoryCategory::Render, 2048);

        let result = profiler.check_leaks();
        assert!(result.has_leaks);
        assert_eq!(result.total_leaked_count, 1);
    }

    #[test]
    fn no_leaks_when_balanced() {
        let profiler = MemoryProfiler::new();
        profiler.record_alloc(MemoryCategory::Render, 1024);
        profiler.save_baseline();

        profiler.record_alloc(MemoryCategory::Render, 2048);
        profiler.record_dealloc(MemoryCategory::Render, 2048);

        let result = profiler.check_leaks();
        assert!(!result.has_leaks);
    }

    #[test]
    fn snapshot_timeline() {
        let profiler = MemoryProfiler::new();
        profiler.record_alloc(MemoryCategory::Render, 1024);
        profiler.take_snapshot(0);
        profiler.record_alloc(MemoryCategory::Render, 2048);
        profiler.take_snapshot(1);

        let timeline = profiler.get_timeline();
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline[0].total_allocated, 1024);
        assert_eq!(timeline[1].total_allocated, 3072);
    }

    #[test]
    fn report_generation() {
        let profiler = MemoryProfiler::new();
        profiler.record_alloc(MemoryCategory::Render, 4 * 1024 * 1024);
        profiler.record_alloc(MemoryCategory::Physics, 2 * 1024 * 1024);

        let report = profiler.generate_report();
        let text = format!("{}", report);
        assert!(text.contains("Render"));
        assert!(text.contains("Physics"));
        assert!(text.contains("MB"));
    }

    #[test]
    fn format_bytes_display() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn category_labels() {
        assert_eq!(MemoryCategory::Render.label(), "Render");
        assert_eq!(MemoryCategory::Ecs.label(), "ECS");
        assert_eq!(MemoryCategory::Ai.label(), "AI");
    }

    #[test]
    fn reset_clears_all() {
        let profiler = MemoryProfiler::new();
        profiler.record_alloc(MemoryCategory::Render, 1024);
        profiler.take_snapshot(0);
        profiler.save_baseline();

        profiler.reset();

        assert_eq!(profiler.total_allocated(), 0);
        assert!(profiler.get_timeline().is_empty());
        assert!(profiler.get_large_allocations().is_empty());
    }

    #[test]
    fn all_categories_covers_all() {
        assert!(MemoryCategory::ALL.len() >= 15);
    }
}
