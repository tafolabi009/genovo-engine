// engine/core/src/logging.rs
//
// Structured logging: log levels (Trace/Debug/Info/Warn/Error), log targets
// (console/file/callback), log formatting with timestamp/module/level, log
// filtering per module, ring buffer for recent logs, log file rotation,
// colored console output.
//
// This module provides a self-contained structured logging system that does not
// depend on external logging crates. It supports multiple output targets,
// per-module filtering, and thread-safe operation.

use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Log level
// ---------------------------------------------------------------------------

/// Log severity levels, ordered from most verbose to most critical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Fatal = 5,
    Off   = 6,
}

impl LogLevel {
    /// Return the short display string for this level.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info  => "INFO",
            Self::Warn  => "WARN",
            Self::Error => "ERROR",
            Self::Fatal => "FATAL",
            Self::Off   => "OFF",
        }
    }

    /// ANSI color code for this level (for colored console output).
    pub fn ansi_color(&self) -> &'static str {
        match self {
            Self::Trace => "\x1b[90m",   // Dark gray
            Self::Debug => "\x1b[36m",   // Cyan
            Self::Info  => "\x1b[32m",   // Green
            Self::Warn  => "\x1b[33m",   // Yellow
            Self::Error => "\x1b[31m",   // Red
            Self::Fatal => "\x1b[1;31m", // Bold red
            Self::Off   => "\x1b[0m",    // Reset
        }
    }

    /// Parse a log level from a string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(Self::Trace),
            "DEBUG" => Some(Self::Debug),
            "INFO"  => Some(Self::Info),
            "WARN" | "WARNING" => Some(Self::Warn),
            "ERROR" => Some(Self::Error),
            "FATAL" => Some(Self::Fatal),
            "OFF"   => Some(Self::Off),
            _ => None,
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Log record
// ---------------------------------------------------------------------------

/// A single log entry.
#[derive(Debug, Clone)]
pub struct LogRecord {
    /// Timestamp (milliseconds since UNIX epoch).
    pub timestamp_ms: u64,
    /// Log level.
    pub level: LogLevel,
    /// Module or category that produced this log.
    pub module: String,
    /// The log message.
    pub message: String,
    /// Thread ID that produced this log.
    pub thread_id: u64,
    /// Thread name (if available).
    pub thread_name: Option<String>,
    /// Source file (if available).
    pub file: Option<String>,
    /// Source line (if available).
    pub line: Option<u32>,
    /// Structured key-value fields.
    pub fields: Vec<(String, String)>,
}

impl LogRecord {
    /// Create a new log record with the current timestamp.
    pub fn new(level: LogLevel, module: &str, message: String) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            timestamp_ms,
            level,
            module: module.to_string(),
            message,
            thread_id: 0,
            thread_name: None,
            file: None,
            line: None,
            fields: Vec::new(),
        }
    }

    /// Add a structured field.
    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.push((key.to_string(), value.to_string()));
        self
    }

    /// Add source location.
    pub fn with_source(mut self, file: &str, line: u32) -> Self {
        self.file = Some(file.to_string());
        self.line = Some(line);
        self
    }

    /// Format the timestamp as ISO 8601 (simplified).
    pub fn format_timestamp(&self) -> String {
        let secs = self.timestamp_ms / 1000;
        let millis = self.timestamp_ms % 1000;
        // Simplified: just seconds.millis
        let total_secs = secs % 86400;
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let s = total_secs % 60;
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, s, millis)
    }
}

// ---------------------------------------------------------------------------
// Log formatter
// ---------------------------------------------------------------------------

/// Controls how log records are formatted to text.
#[derive(Debug, Clone)]
pub struct LogFormatter {
    /// Whether to include timestamps.
    pub show_timestamp: bool,
    /// Whether to include the module name.
    pub show_module: bool,
    /// Whether to include the log level.
    pub show_level: bool,
    /// Whether to include thread info.
    pub show_thread: bool,
    /// Whether to include source location.
    pub show_source: bool,
    /// Whether to include structured fields.
    pub show_fields: bool,
    /// Whether to use ANSI color codes.
    pub colored: bool,
    /// Custom format pattern (if empty, uses default).
    pub pattern: String,
}

impl Default for LogFormatter {
    fn default() -> Self {
        Self {
            show_timestamp: true,
            show_module: true,
            show_level: true,
            show_thread: false,
            show_source: false,
            show_fields: true,
            colored: false,
            pattern: String::new(),
        }
    }
}

impl LogFormatter {
    /// Format a log record to a string.
    pub fn format(&self, record: &LogRecord) -> String {
        let mut parts = Vec::new();

        if self.show_timestamp {
            parts.push(record.format_timestamp());
        }

        if self.show_level {
            if self.colored {
                parts.push(format!("{}[{}]\x1b[0m", record.level.ansi_color(), record.level.as_str()));
            } else {
                parts.push(format!("[{}]", record.level.as_str()));
            }
        }

        if self.show_module && !record.module.is_empty() {
            parts.push(format!("[{}]", record.module));
        }

        if self.show_thread {
            if let Some(ref name) = record.thread_name {
                parts.push(format!("[{}]", name));
            } else {
                parts.push(format!("[thread-{}]", record.thread_id));
            }
        }

        parts.push(record.message.clone());

        if self.show_source {
            if let (Some(ref file), Some(line)) = (&record.file, record.line) {
                parts.push(format!("({}:{})", file, line));
            }
        }

        let mut result = parts.join(" ");

        if self.show_fields && !record.fields.is_empty() {
            let fields_str: Vec<String> = record.fields.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            result.push_str(&format!(" {{{}}}", fields_str.join(", ")));
        }

        result
    }

    /// Create a formatter for console output (with colors).
    pub fn console() -> Self {
        Self {
            colored: true,
            show_thread: false,
            ..Default::default()
        }
    }

    /// Create a formatter for file output (no colors).
    pub fn file() -> Self {
        Self {
            colored: false,
            show_thread: true,
            show_source: true,
            ..Default::default()
        }
    }

    /// Create a compact formatter (minimal output).
    pub fn compact() -> Self {
        Self {
            show_timestamp: false,
            show_module: false,
            show_level: true,
            show_thread: false,
            show_source: false,
            show_fields: false,
            colored: false,
            pattern: String::new(),
        }
    }

    /// Create a JSON formatter.
    pub fn json() -> Self {
        Self {
            show_timestamp: true,
            show_module: true,
            show_level: true,
            show_thread: true,
            show_source: true,
            show_fields: true,
            colored: false,
            pattern: "json".to_string(),
        }
    }

    /// Format as JSON (used when pattern == "json").
    pub fn format_json(&self, record: &LogRecord) -> String {
        let mut json = format!(
            "{{\"timestamp\":{},\"level\":\"{}\",\"module\":\"{}\",\"message\":\"{}\"",
            record.timestamp_ms,
            record.level.as_str(),
            escape_json(&record.module),
            escape_json(&record.message),
        );

        if let Some(ref thread_name) = record.thread_name {
            json.push_str(&format!(",\"thread\":\"{}\"", escape_json(thread_name)));
        }

        if let (Some(ref file), Some(line)) = (&record.file, record.line) {
            json.push_str(&format!(",\"file\":\"{}\",\"line\":{}", escape_json(file), line));
        }

        if !record.fields.is_empty() {
            json.push_str(",\"fields\":{");
            let fields: Vec<String> = record.fields.iter()
                .map(|(k, v)| format!("\"{}\":\"{}\"", escape_json(k), escape_json(v)))
                .collect();
            json.push_str(&fields.join(","));
            json.push('}');
        }

        json.push('}');
        json
    }
}

/// Escape a string for JSON output.
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ---------------------------------------------------------------------------
// Log targets
// ---------------------------------------------------------------------------

/// A target that receives formatted log output.
pub trait LogTarget: Send + Sync {
    /// Write a formatted log line.
    fn write(&self, record: &LogRecord, formatted: &str);
    /// Flush any buffered output.
    fn flush(&self);
    /// Name of this target (for diagnostics).
    fn name(&self) -> &str;
}

/// Console (stdout/stderr) log target.
pub struct ConsoleTarget {
    formatter: LogFormatter,
    /// Whether to use stderr for Warn/Error/Fatal.
    use_stderr_for_errors: bool,
}

impl ConsoleTarget {
    pub fn new(colored: bool) -> Self {
        let mut formatter = LogFormatter::console();
        formatter.colored = colored;
        Self {
            formatter,
            use_stderr_for_errors: true,
        }
    }

    pub fn with_formatter(formatter: LogFormatter) -> Self {
        Self {
            formatter,
            use_stderr_for_errors: true,
        }
    }
}

impl LogTarget for ConsoleTarget {
    fn write(&self, record: &LogRecord, _formatted: &str) {
        let output = if self.formatter.pattern == "json" {
            self.formatter.format_json(record)
        } else {
            self.formatter.format(record)
        };

        if self.use_stderr_for_errors && record.level >= LogLevel::Warn {
            eprintln!("{}", output);
        } else {
            println!("{}", output);
        }
    }

    fn flush(&self) {
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
    }

    fn name(&self) -> &str { "console" }
}

/// File log target with rotation support.
pub struct FileTarget {
    formatter: LogFormatter,
    file_path: PathBuf,
    writer: Mutex<Option<std::fs::File>>,
    max_file_size: u64,
    max_rotated_files: u32,
    current_size: Mutex<u64>,
}

impl FileTarget {
    pub fn new(path: &str, max_file_size: u64, max_rotated_files: u32) -> Self {
        let file_path = PathBuf::from(path);
        let writer = Mutex::new(None);
        Self {
            formatter: LogFormatter::file(),
            file_path,
            writer,
            max_file_size,
            max_rotated_files,
            current_size: Mutex::new(0),
        }
    }

    pub fn with_formatter(mut self, formatter: LogFormatter) -> Self {
        self.formatter = formatter;
        self
    }

    fn ensure_open(&self) {
        let mut writer = self.writer.lock().unwrap();
        if writer.is_none() {
            if let Some(parent) = self.file_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.file_path)
            {
                let size = file.metadata().map(|m| m.len()).unwrap_or(0);
                *self.current_size.lock().unwrap() = size;
                *writer = Some(file);
            }
        }
    }

    fn rotate(&self) {
        let mut writer = self.writer.lock().unwrap();
        *writer = None; // Close current file.

        // Rotate existing files.
        for i in (1..self.max_rotated_files).rev() {
            let from = format!("{}.{}", self.file_path.display(), i);
            let to = format!("{}.{}", self.file_path.display(), i + 1);
            let _ = std::fs::rename(&from, &to);
        }
        // Rename current file to .1
        let rotated = format!("{}.1", self.file_path.display());
        let _ = std::fs::rename(&self.file_path, &rotated);

        // Delete oldest if over limit.
        let oldest = format!("{}.{}", self.file_path.display(), self.max_rotated_files);
        let _ = std::fs::remove_file(&oldest);

        // Open new file.
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&self.file_path)
        {
            *self.current_size.lock().unwrap() = 0;
            *writer = Some(file);
        }
    }
}

impl LogTarget for FileTarget {
    fn write(&self, record: &LogRecord, _formatted: &str) {
        self.ensure_open();

        let output = if self.formatter.pattern == "json" {
            self.formatter.format_json(record)
        } else {
            self.formatter.format(record)
        };
        let line = format!("{}\n", output);
        let line_len = line.len() as u64;

        let mut writer = self.writer.lock().unwrap();
        if let Some(ref mut file) = *writer {
            let _ = file.write_all(line.as_bytes());
            drop(writer);

            let mut size = self.current_size.lock().unwrap();
            *size += line_len;

            if *size >= self.max_file_size {
                drop(size);
                self.rotate();
            }
        }
    }

    fn flush(&self) {
        let mut writer = self.writer.lock().unwrap();
        if let Some(ref mut file) = *writer {
            let _ = file.flush();
        }
    }

    fn name(&self) -> &str { "file" }
}

/// Callback log target.
pub struct CallbackTarget {
    name: String,
    callback: Box<dyn Fn(&LogRecord) + Send + Sync>,
}

impl CallbackTarget {
    pub fn new<F>(name: &str, callback: F) -> Self
    where
        F: Fn(&LogRecord) + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            callback: Box::new(callback),
        }
    }
}

impl LogTarget for CallbackTarget {
    fn write(&self, record: &LogRecord, _formatted: &str) {
        (self.callback)(record);
    }
    fn flush(&self) {}
    fn name(&self) -> &str { &self.name }
}

// ---------------------------------------------------------------------------
// Ring buffer for recent logs
// ---------------------------------------------------------------------------

/// A ring buffer that stores the N most recent log records.
#[derive(Debug)]
pub struct LogRingBuffer {
    records: Vec<LogRecord>,
    capacity: usize,
    write_pos: usize,
    count: usize,
}

impl LogRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            records: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, record: LogRecord) {
        if self.records.len() < self.capacity {
            self.records.push(record);
        } else {
            self.records[self.write_pos] = record;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.count += 1;
    }

    /// Get all records in chronological order.
    pub fn records(&self) -> Vec<&LogRecord> {
        let len = self.records.len();
        if len < self.capacity {
            self.records.iter().collect()
        } else {
            let mut result = Vec::with_capacity(len);
            for i in 0..len {
                let idx = (self.write_pos + i) % len;
                result.push(&self.records[idx]);
            }
            result
        }
    }

    /// Get records filtered by level.
    pub fn records_by_level(&self, min_level: LogLevel) -> Vec<&LogRecord> {
        self.records().into_iter().filter(|r| r.level >= min_level).collect()
    }

    /// Get records filtered by module.
    pub fn records_by_module(&self, module: &str) -> Vec<&LogRecord> {
        self.records().into_iter().filter(|r| r.module == module).collect()
    }

    /// Total number of records ever pushed.
    pub fn total_count(&self) -> usize { self.count }

    /// Number of records currently in the buffer.
    pub fn current_count(&self) -> usize { self.records.len() }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.records.clear();
        self.write_pos = 0;
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

/// Configuration for the logger.
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Global minimum log level.
    pub global_level: LogLevel,
    /// Per-module log level overrides.
    pub module_levels: HashMap<String, LogLevel>,
    /// Ring buffer capacity for recent logs.
    pub ring_buffer_capacity: usize,
    /// Whether to enable the ring buffer.
    pub enable_ring_buffer: bool,
    /// Whether colored console output is enabled.
    pub colored_console: bool,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            global_level: LogLevel::Info,
            module_levels: HashMap::new(),
            ring_buffer_capacity: 1000,
            enable_ring_buffer: true,
            colored_console: true,
        }
    }
}

/// Per-level counters.
#[derive(Debug, Clone, Default)]
pub struct LogStats {
    pub trace_count: u64,
    pub debug_count: u64,
    pub info_count: u64,
    pub warn_count: u64,
    pub error_count: u64,
    pub fatal_count: u64,
    pub total_count: u64,
    pub filtered_count: u64,
}

impl LogStats {
    pub fn increment(&mut self, level: LogLevel) {
        match level {
            LogLevel::Trace => self.trace_count += 1,
            LogLevel::Debug => self.debug_count += 1,
            LogLevel::Info  => self.info_count += 1,
            LogLevel::Warn  => self.warn_count += 1,
            LogLevel::Error => self.error_count += 1,
            LogLevel::Fatal => self.fatal_count += 1,
            LogLevel::Off   => {}
        }
        self.total_count += 1;
    }
}

/// The main logger.
pub struct Logger {
    config: RwLock<LoggerConfig>,
    targets: RwLock<Vec<Arc<dyn LogTarget>>>,
    ring_buffer: Mutex<LogRingBuffer>,
    stats: Mutex<LogStats>,
}

impl Logger {
    /// Create a new logger with the given configuration.
    pub fn new(config: LoggerConfig) -> Self {
        let ring_cap = config.ring_buffer_capacity;
        Self {
            config: RwLock::new(config),
            targets: RwLock::new(Vec::new()),
            ring_buffer: Mutex::new(LogRingBuffer::new(ring_cap)),
            stats: Mutex::new(LogStats::default()),
        }
    }

    /// Create a logger with default configuration and console target.
    pub fn default_with_console() -> Self {
        let config = LoggerConfig::default();
        let logger = Self::new(config);
        logger.add_target(Arc::new(ConsoleTarget::new(true)));
        logger
    }

    /// Add a log target.
    pub fn add_target(&self, target: Arc<dyn LogTarget>) {
        self.targets.write().unwrap().push(target);
    }

    /// Remove all targets with a given name.
    pub fn remove_target(&self, name: &str) {
        self.targets.write().unwrap().retain(|t| t.name() != name);
    }

    /// Set the global log level.
    pub fn set_global_level(&self, level: LogLevel) {
        self.config.write().unwrap().global_level = level;
    }

    /// Set a per-module log level override.
    pub fn set_module_level(&self, module: &str, level: LogLevel) {
        self.config.write().unwrap().module_levels.insert(module.to_string(), level);
    }

    /// Check whether a log record at the given level and module would be logged.
    pub fn is_enabled(&self, level: LogLevel, module: &str) -> bool {
        let config = self.config.read().unwrap();
        let effective_level = config.module_levels.get(module).copied()
            .unwrap_or(config.global_level);
        level >= effective_level
    }

    /// Log a record.
    pub fn log(&self, record: LogRecord) {
        if !self.is_enabled(record.level, &record.module) {
            self.stats.lock().unwrap().filtered_count += 1;
            return;
        }

        // Update stats.
        self.stats.lock().unwrap().increment(record.level);

        // Write to ring buffer.
        {
            let config = self.config.read().unwrap();
            if config.enable_ring_buffer {
                self.ring_buffer.lock().unwrap().push(record.clone());
            }
        }

        // Write to all targets.
        let targets = self.targets.read().unwrap();
        for target in targets.iter() {
            target.write(&record, "");
        }
    }

    /// Convenience methods for logging at specific levels.
    pub fn trace(&self, module: &str, message: impl Into<String>) {
        self.log(LogRecord::new(LogLevel::Trace, module, message.into()));
    }

    pub fn debug(&self, module: &str, message: impl Into<String>) {
        self.log(LogRecord::new(LogLevel::Debug, module, message.into()));
    }

    pub fn info(&self, module: &str, message: impl Into<String>) {
        self.log(LogRecord::new(LogLevel::Info, module, message.into()));
    }

    pub fn warn(&self, module: &str, message: impl Into<String>) {
        self.log(LogRecord::new(LogLevel::Warn, module, message.into()));
    }

    pub fn error(&self, module: &str, message: impl Into<String>) {
        self.log(LogRecord::new(LogLevel::Error, module, message.into()));
    }

    pub fn fatal(&self, module: &str, message: impl Into<String>) {
        self.log(LogRecord::new(LogLevel::Fatal, module, message.into()));
    }

    /// Flush all targets.
    pub fn flush(&self) {
        let targets = self.targets.read().unwrap();
        for target in targets.iter() {
            target.flush();
        }
    }

    /// Get a snapshot of the log statistics.
    pub fn stats(&self) -> LogStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get recent log records from the ring buffer.
    pub fn recent_logs(&self, count: usize) -> Vec<LogRecord> {
        let ring = self.ring_buffer.lock().unwrap();
        let records = ring.records();
        let start = if records.len() > count { records.len() - count } else { 0 };
        records[start..].iter().map(|r| (*r).clone()).collect()
    }

    /// Get recent log records filtered by level.
    pub fn recent_logs_by_level(&self, min_level: LogLevel, count: usize) -> Vec<LogRecord> {
        let ring = self.ring_buffer.lock().unwrap();
        let records = ring.records_by_level(min_level);
        let start = if records.len() > count { records.len() - count } else { 0 };
        records[start..].iter().map(|r| (*r).clone()).collect()
    }

    /// Clear the ring buffer.
    pub fn clear_ring_buffer(&self) {
        self.ring_buffer.lock().unwrap().clear();
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        *self.stats.lock().unwrap() = LogStats::default();
    }
}

// ---------------------------------------------------------------------------
// Scoped logger (for passing module context)
// ---------------------------------------------------------------------------

/// A scoped logger that automatically adds a module name to all log calls.
pub struct ScopedLogger {
    logger: Arc<Logger>,
    module: String,
}

impl ScopedLogger {
    pub fn new(logger: Arc<Logger>, module: &str) -> Self {
        Self {
            logger,
            module: module.to_string(),
        }
    }

    pub fn trace(&self, message: impl Into<String>) {
        self.logger.trace(&self.module, message);
    }

    pub fn debug(&self, message: impl Into<String>) {
        self.logger.debug(&self.module, message);
    }

    pub fn info(&self, message: impl Into<String>) {
        self.logger.info(&self.module, message);
    }

    pub fn warn(&self, message: impl Into<String>) {
        self.logger.warn(&self.module, message);
    }

    pub fn error(&self, message: impl Into<String>) {
        self.logger.error(&self.module, message);
    }

    pub fn fatal(&self, message: impl Into<String>) {
        self.logger.fatal(&self.module, message);
    }

    pub fn is_enabled(&self, level: LogLevel) -> bool {
        self.logger.is_enabled(level, &self.module)
    }
}

// ---------------------------------------------------------------------------
// Macros (as functions since we can't export macros from a module)
// ---------------------------------------------------------------------------

/// Create a log record with source location.
pub fn make_record(level: LogLevel, module: &str, message: String, file: &str, line: u32) -> LogRecord {
    LogRecord::new(level, module, message).with_source(file, line)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn test_log_level_parse() {
        assert_eq!(LogLevel::from_str("info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str("WARNING"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("unknown"), None);
    }

    #[test]
    fn test_formatter_default() {
        let fmt = LogFormatter::default();
        let record = LogRecord::new(LogLevel::Info, "test", "Hello world".to_string());
        let output = fmt.format(&record);
        assert!(output.contains("[INFO]"));
        assert!(output.contains("[test]"));
        assert!(output.contains("Hello world"));
    }

    #[test]
    fn test_formatter_json() {
        let fmt = LogFormatter::json();
        let record = LogRecord::new(LogLevel::Error, "render", "GPU lost".to_string())
            .with_field("gpu", "AMD");
        let output = fmt.format_json(&record);
        assert!(output.contains("\"level\":\"ERROR\""));
        assert!(output.contains("\"module\":\"render\""));
        assert!(output.contains("\"gpu\":\"AMD\""));
    }

    #[test]
    fn test_ring_buffer() {
        let mut ring = LogRingBuffer::new(3);
        for i in 0..5 {
            ring.push(LogRecord::new(LogLevel::Info, "test", format!("msg {}", i)));
        }
        assert_eq!(ring.current_count(), 3);
        assert_eq!(ring.total_count(), 5);
        let records = ring.records();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].message, "msg 2");
        assert_eq!(records[1].message, "msg 3");
        assert_eq!(records[2].message, "msg 4");
    }

    #[test]
    fn test_logger_filtering() {
        let config = LoggerConfig {
            global_level: LogLevel::Warn,
            ..Default::default()
        };
        let logger = Logger::new(config);
        assert!(!logger.is_enabled(LogLevel::Debug, "any"));
        assert!(logger.is_enabled(LogLevel::Warn, "any"));
        assert!(logger.is_enabled(LogLevel::Error, "any"));
    }

    #[test]
    fn test_logger_module_level_override() {
        let mut config = LoggerConfig::default();
        config.global_level = LogLevel::Warn;
        config.module_levels.insert("render".into(), LogLevel::Trace);
        let logger = Logger::new(config);

        assert!(!logger.is_enabled(LogLevel::Debug, "physics"));
        assert!(logger.is_enabled(LogLevel::Trace, "render"));
    }

    #[test]
    fn test_logger_stats() {
        let logger = Logger::new(LoggerConfig {
            global_level: LogLevel::Trace,
            ..Default::default()
        });
        logger.log(LogRecord::new(LogLevel::Info, "test", "msg".into()));
        logger.log(LogRecord::new(LogLevel::Error, "test", "err".into()));

        let stats = logger.stats();
        assert_eq!(stats.info_count, 1);
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.total_count, 2);
    }

    #[test]
    fn test_callback_target() {
        let received = Arc::new(Mutex::new(Vec::new()));
        let received_clone = received.clone();
        let target = CallbackTarget::new("test", move |record| {
            received_clone.lock().unwrap().push(record.message.clone());
        });

        let logger = Logger::new(LoggerConfig {
            global_level: LogLevel::Trace,
            ..Default::default()
        });
        logger.add_target(Arc::new(target));
        logger.info("mod", "hello");

        let msgs = received.lock().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0], "hello");
    }
}
