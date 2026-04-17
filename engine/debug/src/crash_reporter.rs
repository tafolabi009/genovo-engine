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
}
