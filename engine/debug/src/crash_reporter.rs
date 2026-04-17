//! Crash handling: panic hook, stack trace capture, minidump generation stub,
//! crash log with system info, and auto-save on crash.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static CRASH_HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub struct SystemInfo { pub os: String, pub os_version: String, pub cpu: String, pub cpu_cores: u32, pub ram_mb: u64, pub gpu: String, pub gpu_driver: String, pub display_resolution: (u32, u32), pub app_version: String, pub build_config: String }
impl SystemInfo {
    pub fn gather() -> Self {
        Self { os: std::env::consts::OS.to_string(), os_version: String::new(), cpu: String::new(), cpu_cores: 1, ram_mb: 0, gpu: String::new(), gpu_driver: String::new(), display_resolution: (1920, 1080), app_version: env!("CARGO_PKG_VERSION").to_string(), build_config: if cfg!(debug_assertions) { "Debug" } else { "Release" }.to_string() }
    }
}
impl fmt::Display for SystemInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OS: {} {}", self.os, self.os_version)?;
        writeln!(f, "CPU: {} ({} cores)", self.cpu, self.cpu_cores)?;
        writeln!(f, "RAM: {} MB", self.ram_mb)?;
        writeln!(f, "GPU: {} ({})", self.gpu, self.gpu_driver)?;
        writeln!(f, "Resolution: {}x{}", self.display_resolution.0, self.display_resolution.1)?;
        writeln!(f, "App: {} ({})", self.app_version, self.build_config)
    }
}

#[derive(Debug, Clone)]
pub struct StackFrame { pub function: String, pub file: Option<String>, pub line: Option<u32>, pub address: u64 }
impl fmt::Display for StackFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "  {:#018x} {}", self.address, self.function)?;
        if let Some(ref file) = self.file { write!(f, " at {}", file)?; if let Some(line) = self.line { write!(f, ":{}", line)?; } }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CrashReport {
    pub timestamp: u64, pub crash_message: String, pub stack_frames: Vec<StackFrame>,
    pub system_info: SystemInfo, pub thread_name: Option<String>,
    pub log_tail: Vec<String>, pub active_scene: Option<String>,
    pub entity_count: u32, pub frame_number: u64, pub uptime_seconds: f64,
    pub memory_used_mb: u64, pub custom_data: Vec<(String, String)>,
}

impl CrashReport {
    pub fn new(message: impl Into<String>) -> Self {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        Self {
            timestamp: ts, crash_message: message.into(), stack_frames: Vec::new(),
            system_info: SystemInfo::gather(), thread_name: std::thread::current().name().map(|s| s.to_string()),
            log_tail: Vec::new(), active_scene: None, entity_count: 0, frame_number: 0,
            uptime_seconds: 0.0, memory_used_mb: 0, custom_data: Vec::new(),
        }
    }

    pub fn add_stack_frame(&mut self, frame: StackFrame) { self.stack_frames.push(frame); }
    pub fn add_log_line(&mut self, line: impl Into<String>) { self.log_tail.push(line.into()); if self.log_tail.len() > 100 { self.log_tail.remove(0); } }
    pub fn add_custom_data(&mut self, key: impl Into<String>, value: impl Into<String>) { self.custom_data.push((key.into(), value.into())); }

    pub fn format_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== GENOVO ENGINE CRASH REPORT ===

");
        report.push_str(&format!("Timestamp: {}
", self.timestamp));
        report.push_str(&format!("Thread: {}
", self.thread_name.as_deref().unwrap_or("unknown")));
        report.push_str(&format!("Crash: {}

", self.crash_message));
        report.push_str("--- Stack Trace ---
");
        for frame in &self.stack_frames { report.push_str(&format!("{}
", frame)); }
        report.push_str("
--- System Info ---
");
        report.push_str(&format!("{}", self.system_info));
        report.push_str(&format!("
Scene: {}
", self.active_scene.as_deref().unwrap_or("none")));
        report.push_str(&format!("Entities: {}
Frame: {}
Uptime: {:.1}s
Memory: {} MB
", self.entity_count, self.frame_number, self.uptime_seconds, self.memory_used_mb));
        if !self.custom_data.is_empty() {
            report.push_str("
--- Custom Data ---
");
            for (k, v) in &self.custom_data { report.push_str(&format!("  {}: {}
", k, v)); }
        }
        if !self.log_tail.is_empty() {
            report.push_str("
--- Recent Log ---
");
            for line in &self.log_tail { report.push_str(&format!("  {}
", line)); }
        }
        report
    }

    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.format_report())
    }
}

pub struct CrashReporter {
    pub auto_save_on_crash: bool,
    pub crash_log_dir: String,
    pub max_crash_logs: u32,
    pub report_url: Option<String>,
    pub custom_data_providers: Vec<Box<dyn Fn() -> Vec<(String, String)> + Send + Sync>>,
    pub log_buffer: Vec<String>,
    pub log_buffer_size: usize,
    pub installed: bool,
}

impl CrashReporter {
    pub fn new() -> Self {
        Self { auto_save_on_crash: true, crash_log_dir: "crash_logs".to_string(), max_crash_logs: 10, report_url: None, custom_data_providers: Vec::new(), log_buffer: Vec::new(), log_buffer_size: 200, installed: false }
    }

    pub fn install(&mut self) {
        if CRASH_HANDLER_INSTALLED.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
            let auto_save = self.auto_save_on_crash;
            let log_dir = self.crash_log_dir.clone();
            std::panic::set_hook(Box::new(move |info| {
                let message = if let Some(s) = info.payload().downcast_ref::<&str>() { s.to_string() }
                    else if let Some(s) = info.payload().downcast_ref::<String>() { s.clone() }
                    else { "Unknown panic".to_string() };
                let mut report = CrashReport::new(&message);
                if let Some(loc) = info.location() {
                    report.add_stack_frame(StackFrame { function: "panic".to_string(), file: Some(loc.file().to_string()), line: Some(loc.line()), address: 0 });
                }
                let _ = std::fs::create_dir_all(&log_dir);
                let path = format!("{}/crash_{}.log", log_dir, report.timestamp);
                let _ = report.write_to_file(&path);
                eprintln!("CRASH: {} (report saved to {})", message, path);
            }));
            self.installed = true;
        }
    }

    pub fn log(&mut self, message: impl Into<String>) {
        let msg = message.into();
        self.log_buffer.push(msg);
        if self.log_buffer.len() > self.log_buffer_size { self.log_buffer.remove(0); }
    }

    pub fn generate_report(&self, message: impl Into<String>) -> CrashReport {
        let mut report = CrashReport::new(message);
        for line in &self.log_buffer { report.add_log_line(line.clone()); }
        for provider in &self.custom_data_providers {
            for (k, v) in provider() { report.add_custom_data(k, v); }
        }
        report
    }

    pub fn generate_minidump_stub(&self) -> Vec<u8> {
        let header = b"MDMP"; // Minidump magic
        let mut data = header.to_vec();
        data.extend_from_slice(&[0u8; 28]); // Stub header
        data
    }
}
impl Default for CrashReporter { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crash_report_format() {
        let report = CrashReport::new("test crash");
        let text = report.format_report();
        assert!(text.contains("test crash"));
        assert!(text.contains("CRASH REPORT"));
    }
    #[test]
    fn system_info() {
        let info = SystemInfo::gather();
        assert!(!info.os.is_empty());
    }

    #[test]
    fn stack_frame_display() {
        let frame = StackFrame {
            function: "my_function".to_string(),
            file: Some("src/main.rs".to_string()),
            line: Some(42),
            address: 0xDEADBEEF,
        };
        let s = format!("{}", frame);
        assert!(s.contains("my_function"));
        assert!(s.contains("main.rs"));
    }

    #[test]
    fn crash_report_custom_data() {
        let mut report = CrashReport::new("test");
        report.add_custom_data("player_id", "12345");
        report.add_custom_data("level", "dungeon_03");
        let text = report.format_report();
        assert!(text.contains("player_id"));
        assert!(text.contains("dungeon_03"));
    }

    #[test]
    fn crash_report_log_tail() {
        let mut report = CrashReport::new("test");
        for i in 0..150 {
            report.add_log_line(format!("Log line {}", i));
        }
        assert_eq!(report.log_tail.len(), 100); // capped at 100
    }

    #[test]
    fn crash_reporter_log_buffer() {
        let mut reporter = CrashReporter::new();
        for i in 0..300 {
            reporter.log(format!("Line {}", i));
        }
        assert_eq!(reporter.log_buffer.len(), 200); // capped at buffer_size
    }

    #[test]
    fn minidump_stub() {
        let reporter = CrashReporter::new();
        let dump = reporter.generate_minidump_stub();
        assert!(dump.len() >= 4);
        assert_eq!(&dump[0..4], b"MDMP");
    }

    #[test]
    fn generate_report_with_context() {
        let mut reporter = CrashReporter::new();
        reporter.log("Starting game");
        reporter.log("Loading level");
        reporter.log("Spawning player");
        let report = reporter.generate_report("Out of memory");
        assert!(report.crash_message.contains("Out of memory"));
        assert_eq!(report.log_tail.len(), 3);
    }
}

// ---------------------------------------------------------------------------
// Crash analytics aggregation
// ---------------------------------------------------------------------------

/// Aggregated crash statistics from multiple crash reports.
#[derive(Debug, Clone)]
pub struct CrashAnalytics {
    /// Total number of crashes recorded.
    pub total_crashes: u64,
    /// Crashes per unique error message.
    pub crashes_by_message: HashMap<String, u64>,
    /// Crashes per function name.
    pub crashes_by_function: HashMap<String, u64>,
    /// Crashes per file.
    pub crashes_by_file: HashMap<String, u64>,
    /// Average uptime at crash (seconds).
    pub avg_uptime: f64,
    /// Minimum uptime at crash.
    pub min_uptime: f64,
    /// Maximum uptime at crash.
    pub max_uptime: f64,
    /// Crashes per OS.
    pub crashes_by_os: HashMap<String, u64>,
    /// Crashes per GPU.
    pub crashes_by_gpu: HashMap<String, u64>,
    /// Crashes per build version.
    pub crashes_by_version: HashMap<String, u64>,
    /// Most recent crash timestamp.
    pub last_crash_time: u64,
    /// Crash rate (crashes per hour of gameplay).
    pub crash_rate: f64,
}

impl CrashAnalytics {
    /// Create empty analytics.
    pub fn new() -> Self {
        Self {
            total_crashes: 0,
            crashes_by_message: HashMap::new(),
            crashes_by_function: HashMap::new(),
            crashes_by_file: HashMap::new(),
            avg_uptime: 0.0,
            min_uptime: f64::MAX,
            max_uptime: 0.0,
            crashes_by_os: HashMap::new(),
            crashes_by_gpu: HashMap::new(),
            crashes_by_version: HashMap::new(),
            last_crash_time: 0,
            crash_rate: 0.0,
        }
    }

    /// Incorporate a crash report into the analytics.
    pub fn add_report(&mut self, report: &CrashReport) {
        self.total_crashes += 1;

        *self.crashes_by_message
            .entry(report.crash_message.clone())
            .or_insert(0) += 1;

        if let Some(frame) = report.stack_frames.first() {
            *self.crashes_by_function
                .entry(frame.function.clone())
                .or_insert(0) += 1;

            if let Some(ref file) = frame.file {
                *self.crashes_by_file
                    .entry(file.clone())
                    .or_insert(0) += 1;
            }
        }

        let uptime = report.uptime_seconds;
        self.min_uptime = self.min_uptime.min(uptime);
        self.max_uptime = self.max_uptime.max(uptime);
        self.avg_uptime = (self.avg_uptime * (self.total_crashes - 1) as f64 + uptime)
            / self.total_crashes as f64;

        *self.crashes_by_os
            .entry(report.system_info.os.clone())
            .or_insert(0) += 1;

        *self.crashes_by_gpu
            .entry(report.system_info.gpu.clone())
            .or_insert(0) += 1;

        *self.crashes_by_version
            .entry(report.system_info.app_version.clone())
            .or_insert(0) += 1;

        if report.timestamp > self.last_crash_time {
            self.last_crash_time = report.timestamp;
        }
    }

    /// Get the top N most common crash messages.
    pub fn top_crashes(&self, n: usize) -> Vec<(&str, u64)> {
        let mut sorted: Vec<_> = self.crashes_by_message
            .iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(n);
        sorted
    }

    /// Get the top N most common crash locations (functions).
    pub fn top_functions(&self, n: usize) -> Vec<(&str, u64)> {
        let mut sorted: Vec<_> = self.crashes_by_function
            .iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(n);
        sorted
    }

    /// Generate a summary report string.
    pub fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("Crash Analytics Summary\n"));
        report.push_str(&format!("  Total crashes: {}\n", self.total_crashes));
        report.push_str(&format!("  Avg uptime: {:.1}s\n", self.avg_uptime));
        report.push_str(&format!("  Crash rate: {:.4}/hr\n", self.crash_rate));
        report.push_str(&format!("\n  Top 5 crashes:\n"));
        for (msg, count) in self.top_crashes(5) {
            report.push_str(&format!("    [{:4}x] {}\n", count, msg));
        }
        report.push_str(&format!("\n  Top 5 functions:\n"));
        for (func, count) in self.top_functions(5) {
            report.push_str(&format!("    [{:4}x] {}\n", count, func));
        }
        report
    }
}

impl Default for CrashAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Breadcrumb trail for crash context
// ---------------------------------------------------------------------------

/// A breadcrumb is a timestamped note left by the application to help
/// diagnose what was happening when a crash occurred.
#[derive(Debug, Clone)]
pub struct Breadcrumb {
    /// Timestamp (relative to app start, in seconds).
    pub time: f64,
    /// Category (e.g., "navigation", "network", "ui").
    pub category: String,
    /// Message.
    pub message: String,
    /// Severity level.
    pub level: BreadcrumbLevel,
    /// Optional data.
    pub data: HashMap<String, String>,
}

/// Severity level for breadcrumbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreadcrumbLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

impl Breadcrumb {
    /// Create a new breadcrumb.
    pub fn new(
        time: f64,
        category: impl Into<String>,
        message: impl Into<String>,
        level: BreadcrumbLevel,
    ) -> Self {
        Self {
            time,
            category: category.into(),
            message: message.into(),
            level,
            data: HashMap::new(),
        }
    }

    /// Add data to the breadcrumb.
    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }
}

impl fmt::Display for Breadcrumb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.2}s] [{:?}] {}: {}",
            self.time, self.level, self.category, self.message
        )?;
        if !self.data.is_empty() {
            write!(f, " {{")?;
            for (i, (k, v)) in self.data.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}: {}", k, v)?;
            }
            write!(f, "}}")?;
        }
        Ok(())
    }
}

/// Trail of breadcrumbs for crash context.
pub struct BreadcrumbTrail {
    /// The breadcrumbs in chronological order.
    pub crumbs: Vec<Breadcrumb>,
    /// Maximum number of breadcrumbs to keep.
    pub max_crumbs: usize,
    /// Current application time.
    pub current_time: f64,
}

impl BreadcrumbTrail {
    /// Create a new trail.
    pub fn new(max_crumbs: usize) -> Self {
        Self {
            crumbs: Vec::with_capacity(max_crumbs),
            max_crumbs,
            current_time: 0.0,
        }
    }

    /// Drop a breadcrumb.
    pub fn drop_crumb(
        &mut self,
        category: impl Into<String>,
        message: impl Into<String>,
        level: BreadcrumbLevel,
    ) {
        if self.crumbs.len() >= self.max_crumbs {
            self.crumbs.remove(0);
        }
        self.crumbs.push(Breadcrumb::new(
            self.current_time,
            category,
            message,
            level,
        ));
    }

    /// Drop an info breadcrumb.
    pub fn info(&mut self, category: impl Into<String>, message: impl Into<String>) {
        self.drop_crumb(category, message, BreadcrumbLevel::Info);
    }

    /// Drop a warning breadcrumb.
    pub fn warn(&mut self, category: impl Into<String>, message: impl Into<String>) {
        self.drop_crumb(category, message, BreadcrumbLevel::Warning);
    }

    /// Drop an error breadcrumb.
    pub fn error(&mut self, category: impl Into<String>, message: impl Into<String>) {
        self.drop_crumb(category, message, BreadcrumbLevel::Error);
    }

    /// Update the current time.
    pub fn update_time(&mut self, time: f64) {
        self.current_time = time;
    }

    /// Get all breadcrumbs.
    pub fn all(&self) -> &[Breadcrumb] {
        &self.crumbs
    }

    /// Get the last N breadcrumbs.
    pub fn last_n(&self, n: usize) -> &[Breadcrumb] {
        let start = self.crumbs.len().saturating_sub(n);
        &self.crumbs[start..]
    }

    /// Get breadcrumbs of a specific category.
    pub fn by_category(&self, category: &str) -> Vec<&Breadcrumb> {
        self.crumbs.iter().filter(|c| c.category == category).collect()
    }

    /// Format all breadcrumbs as a string.
    pub fn format(&self) -> String {
        let mut s = String::new();
        for crumb in &self.crumbs {
            s.push_str(&format!("{}\n", crumb));
        }
        s
    }

    /// Clear all breadcrumbs.
    pub fn clear(&mut self) {
        self.crumbs.clear();
    }

    /// Get the count.
    pub fn len(&self) -> usize {
        self.crumbs.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.crumbs.is_empty()
    }
}
