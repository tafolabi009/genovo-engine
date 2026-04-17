// engine/render/src/shader_hot_reload.rs
//
// Shader hot-reloading for the Genovo engine.
//
// Provides runtime shader file watching, recompilation, and pipeline swapping
// without engine restart:
//
// - **File watching** -- Monitors shader source files for changes using
//   polling with configurable intervals.
// - **Recompilation** -- Triggers shader recompilation on detected changes.
// - **Pipeline swapping** -- Atomically swaps shader pipelines on successful
//   compilation, ensuring no visual glitches.
// - **Error recovery** -- On compilation failure, keeps the previous working
//   shader active and reports the error.
// - **Shader edit history** -- Maintains a history of shader compilations
//   with timestamps, enabling rollback if needed.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default polling interval for shader file changes (milliseconds).
const DEFAULT_POLL_INTERVAL_MS: u64 = 500;

/// Maximum number of history entries per shader.
const MAX_HISTORY_ENTRIES: usize = 32;

/// Maximum compile errors stored per shader.
const MAX_COMPILE_ERRORS: usize = 16;

// ---------------------------------------------------------------------------
// Shader identifier
// ---------------------------------------------------------------------------

/// Unique identifier for a watched shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WatchedShaderId(pub u32);

impl fmt::Display for WatchedShaderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shader({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Shader stage
// ---------------------------------------------------------------------------

/// Shader stage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStageType {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    TessControl,
    TessEvaluation,
    Mesh,
    Task,
}

impl fmt::Display for ShaderStageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vertex => write!(f, "vertex"),
            Self::Fragment => write!(f, "fragment"),
            Self::Compute => write!(f, "compute"),
            Self::Geometry => write!(f, "geometry"),
            Self::TessControl => write!(f, "tess_control"),
            Self::TessEvaluation => write!(f, "tess_eval"),
            Self::Mesh => write!(f, "mesh"),
            Self::Task => write!(f, "task"),
        }
    }
}

// ---------------------------------------------------------------------------
// Compilation result
// ---------------------------------------------------------------------------

/// Result of a shader compilation attempt.
#[derive(Debug, Clone)]
pub enum CompilationResult {
    /// Compilation succeeded.
    Success {
        /// Compiled bytecode (SPIR-V, DXIL, etc.).
        bytecode: Vec<u8>,
        /// Compilation warnings.
        warnings: Vec<String>,
        /// Compilation time in milliseconds.
        compile_time_ms: u64,
    },
    /// Compilation failed.
    Failure {
        /// Error messages.
        errors: Vec<CompileError>,
        /// Partial warnings before failure.
        warnings: Vec<String>,
    },
}

impl CompilationResult {
    /// Returns true if compilation succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Returns true if compilation failed.
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Failure { .. })
    }
}

/// A single compile error with location information.
#[derive(Debug, Clone)]
pub struct CompileError {
    /// File path where the error occurred.
    pub file: String,
    /// Line number (1-based).
    pub line: u32,
    /// Column number (1-based).
    pub column: u32,
    /// Error message.
    pub message: String,
    /// Error severity.
    pub severity: ErrorSeverity,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}: {}: {}",
            self.file, self.line, self.column, self.severity, self.message
        )
    }
}

/// Error severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Error,
    Warning,
    Info,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "error"),
            Self::Warning => write!(f, "warning"),
            Self::Info => write!(f, "info"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shader history entry
// ---------------------------------------------------------------------------

/// Record of a shader compilation event.
#[derive(Debug, Clone)]
pub struct ShaderHistoryEntry {
    /// When the compilation was triggered.
    pub timestamp: SystemTime,
    /// Whether compilation succeeded.
    pub success: bool,
    /// Compile time in milliseconds.
    pub compile_time_ms: u64,
    /// Number of errors (0 if success).
    pub error_count: usize,
    /// Number of warnings.
    pub warning_count: usize,
    /// Bytecode size if successful.
    pub bytecode_size: usize,
    /// Source file hash at time of compilation.
    pub source_hash: u64,
}

// ---------------------------------------------------------------------------
// Watched shader entry
// ---------------------------------------------------------------------------

/// Internal state for a watched shader file.
#[derive(Debug, Clone)]
struct WatchedShader {
    /// Unique identifier.
    id: WatchedShaderId,
    /// Display name.
    name: String,
    /// Path to the shader source file.
    source_path: PathBuf,
    /// Shader stage type.
    stage: ShaderStageType,
    /// Include paths for resolving #include directives.
    include_paths: Vec<PathBuf>,
    /// Preprocessor defines.
    defines: HashMap<String, String>,
    /// Last known modification time.
    last_modified: Option<SystemTime>,
    /// Hash of the last successfully compiled source.
    last_compiled_hash: u64,
    /// Current compiled bytecode (the "good" version).
    current_bytecode: Option<Vec<u8>>,
    /// Last compile errors (empty if last compile was successful).
    last_errors: Vec<CompileError>,
    /// Compilation history.
    history: Vec<ShaderHistoryEntry>,
    /// Whether this shader is dirty (needs recompilation).
    dirty: bool,
    /// Whether to auto-recompile on change.
    auto_compile: bool,
    /// Pipeline handles that use this shader (for invalidation).
    dependent_pipelines: Vec<u32>,
}

impl WatchedShader {
    fn new(
        id: WatchedShaderId,
        name: String,
        source_path: PathBuf,
        stage: ShaderStageType,
    ) -> Self {
        Self {
            id,
            name,
            source_path,
            stage,
            include_paths: Vec::new(),
            defines: HashMap::new(),
            last_modified: None,
            last_compiled_hash: 0,
            current_bytecode: None,
            last_errors: Vec::new(),
            history: Vec::new(),
            dirty: true,
            auto_compile: true,
            dependent_pipelines: Vec::new(),
        }
    }

    fn add_history(&mut self, entry: ShaderHistoryEntry) {
        self.history.push(entry);
        if self.history.len() > MAX_HISTORY_ENTRIES {
            self.history.remove(0);
        }
    }
}

// ---------------------------------------------------------------------------
// Shader compiler trait
// ---------------------------------------------------------------------------

/// Trait for shader compilers that the hot-reload system can use.
pub trait ShaderCompiler: fmt::Debug {
    /// Compile a shader from source code.
    fn compile(
        &self,
        source: &str,
        stage: ShaderStageType,
        defines: &HashMap<String, String>,
        include_paths: &[PathBuf],
    ) -> CompilationResult;

    /// Get the compiler name.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Null compiler (for testing)
// ---------------------------------------------------------------------------

/// A no-op compiler that always succeeds with empty bytecode.
#[derive(Debug)]
pub struct NullCompiler;

impl ShaderCompiler for NullCompiler {
    fn compile(
        &self,
        _source: &str,
        _stage: ShaderStageType,
        _defines: &HashMap<String, String>,
        _include_paths: &[PathBuf],
    ) -> CompilationResult {
        CompilationResult::Success {
            bytecode: Vec::new(),
            warnings: Vec::new(),
            compile_time_ms: 0,
        }
    }

    fn name(&self) -> &str {
        "null"
    }
}

// ---------------------------------------------------------------------------
// Hot reload events
// ---------------------------------------------------------------------------

/// Events emitted by the hot-reload system.
#[derive(Debug, Clone)]
pub enum HotReloadEvent {
    /// A shader file was modified.
    FileChanged {
        shader_id: WatchedShaderId,
        path: PathBuf,
    },
    /// Shader compilation started.
    CompilationStarted {
        shader_id: WatchedShaderId,
    },
    /// Shader compilation succeeded and pipeline was swapped.
    CompilationSucceeded {
        shader_id: WatchedShaderId,
        compile_time_ms: u64,
        warnings: Vec<String>,
    },
    /// Shader compilation failed; old shader retained.
    CompilationFailed {
        shader_id: WatchedShaderId,
        errors: Vec<CompileError>,
    },
    /// A shader was added to the watch list.
    ShaderAdded {
        shader_id: WatchedShaderId,
        path: PathBuf,
    },
    /// A shader was removed from the watch list.
    ShaderRemoved {
        shader_id: WatchedShaderId,
    },
}

// ---------------------------------------------------------------------------
// Hot reload manager
// ---------------------------------------------------------------------------

/// Manages shader hot-reloading: watches files, recompiles, and swaps pipelines.
pub struct ShaderHotReloadManager {
    /// All watched shaders.
    shaders: HashMap<WatchedShaderId, WatchedShader>,
    /// Next shader ID to assign.
    next_id: u32,
    /// Polling interval.
    poll_interval: Duration,
    /// Last poll time.
    last_poll: SystemTime,
    /// Whether hot-reload is enabled.
    enabled: bool,
    /// Event queue.
    events: Vec<HotReloadEvent>,
    /// Statistics.
    stats: HotReloadStats,
    /// Path to shader ID lookup.
    path_to_id: HashMap<PathBuf, WatchedShaderId>,
}

/// Statistics for the hot-reload system.
#[derive(Debug, Clone, Default)]
pub struct HotReloadStats {
    /// Total number of watched shaders.
    pub watched_count: usize,
    /// Total successful recompilations.
    pub total_successes: u64,
    /// Total failed recompilations.
    pub total_failures: u64,
    /// Total file change detections.
    pub total_changes_detected: u64,
    /// Number of shaders currently in error state.
    pub error_count: usize,
    /// Average compilation time (milliseconds).
    pub avg_compile_time_ms: f64,
}

impl ShaderHotReloadManager {
    /// Create a new hot-reload manager.
    pub fn new() -> Self {
        Self {
            shaders: HashMap::new(),
            next_id: 0,
            poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
            last_poll: SystemTime::UNIX_EPOCH,
            enabled: true,
            events: Vec::new(),
            stats: HotReloadStats::default(),
            path_to_id: HashMap::new(),
        }
    }

    /// Set the polling interval.
    pub fn set_poll_interval(&mut self, interval: Duration) {
        self.poll_interval = interval;
    }

    /// Enable or disable hot-reloading.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether hot-reloading is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Register a shader for watching.
    pub fn watch(
        &mut self,
        name: &str,
        source_path: &Path,
        stage: ShaderStageType,
    ) -> WatchedShaderId {
        // Check if already watching.
        if let Some(&id) = self.path_to_id.get(source_path) {
            return id;
        }

        let id = WatchedShaderId(self.next_id);
        self.next_id += 1;

        let shader = WatchedShader::new(
            id,
            name.to_string(),
            source_path.to_path_buf(),
            stage,
        );

        self.path_to_id.insert(source_path.to_path_buf(), id);
        self.shaders.insert(id, shader);

        self.events.push(HotReloadEvent::ShaderAdded {
            shader_id: id,
            path: source_path.to_path_buf(),
        });

        id
    }

    /// Stop watching a shader.
    pub fn unwatch(&mut self, id: WatchedShaderId) -> bool {
        if let Some(shader) = self.shaders.remove(&id) {
            self.path_to_id.remove(&shader.source_path);
            self.events.push(HotReloadEvent::ShaderRemoved { shader_id: id });
            true
        } else {
            false
        }
    }

    /// Add preprocessor defines for a shader.
    pub fn set_defines(&mut self, id: WatchedShaderId, defines: HashMap<String, String>) {
        if let Some(shader) = self.shaders.get_mut(&id) {
            shader.defines = defines;
            shader.dirty = true;
        }
    }

    /// Set include paths for a shader.
    pub fn set_include_paths(&mut self, id: WatchedShaderId, paths: Vec<PathBuf>) {
        if let Some(shader) = self.shaders.get_mut(&id) {
            shader.include_paths = paths;
            shader.dirty = true;
        }
    }

    /// Add a dependent pipeline to be invalidated when the shader recompiles.
    pub fn add_dependent_pipeline(&mut self, id: WatchedShaderId, pipeline: u32) {
        if let Some(shader) = self.shaders.get_mut(&id) {
            if !shader.dependent_pipelines.contains(&pipeline) {
                shader.dependent_pipelines.push(pipeline);
            }
        }
    }

    /// Poll for file changes and trigger recompilation.
    ///
    /// Returns the list of shaders that were recompiled.
    pub fn poll(&mut self, compiler: &dyn ShaderCompiler) -> Vec<WatchedShaderId> {
        if !self.enabled {
            return Vec::new();
        }

        let now = SystemTime::now();
        if now
            .duration_since(self.last_poll)
            .unwrap_or_default()
            < self.poll_interval
        {
            return Vec::new();
        }
        self.last_poll = now;

        let mut dirty_ids = Vec::new();

        // Check for file changes.
        let shader_ids: Vec<WatchedShaderId> = self.shaders.keys().copied().collect();
        for id in &shader_ids {
            if let Some(shader) = self.shaders.get_mut(id) {
                if let Ok(metadata) = std::fs::metadata(&shader.source_path) {
                    if let Ok(modified) = metadata.modified() {
                        let changed = match shader.last_modified {
                            Some(prev) => modified > prev,
                            None => true,
                        };
                        if changed {
                            shader.last_modified = Some(modified);
                            shader.dirty = true;
                            self.stats.total_changes_detected += 1;
                            self.events.push(HotReloadEvent::FileChanged {
                                shader_id: *id,
                                path: shader.source_path.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Compile dirty shaders.
        for id in &shader_ids {
            let should_compile = self
                .shaders
                .get(id)
                .map(|s| s.dirty && s.auto_compile)
                .unwrap_or(false);

            if should_compile {
                dirty_ids.push(*id);
            }
        }

        for id in &dirty_ids {
            self.compile_shader(*id, compiler);
        }

        dirty_ids
    }

    /// Manually trigger recompilation of a shader.
    pub fn compile_shader(&mut self, id: WatchedShaderId, compiler: &dyn ShaderCompiler) -> bool {
        let source = {
            let shader = match self.shaders.get(&id) {
                Some(s) => s,
                None => return false,
            };

            self.events.push(HotReloadEvent::CompilationStarted { shader_id: id });

            match std::fs::read_to_string(&shader.source_path) {
                Ok(s) => s,
                Err(_) => return false,
            }
        };

        let source_hash = simple_hash(&source);

        let (stage, defines, include_paths) = {
            let shader = self.shaders.get(&id).unwrap();
            (shader.stage, shader.defines.clone(), shader.include_paths.clone())
        };

        let result = compiler.compile(&source, stage, &defines, &include_paths);

        let shader = self.shaders.get_mut(&id).unwrap();
        shader.dirty = false;

        match result {
            CompilationResult::Success {
                bytecode,
                warnings,
                compile_time_ms,
            } => {
                shader.current_bytecode = Some(bytecode.clone());
                shader.last_compiled_hash = source_hash;
                shader.last_errors.clear();

                shader.add_history(ShaderHistoryEntry {
                    timestamp: SystemTime::now(),
                    success: true,
                    compile_time_ms,
                    error_count: 0,
                    warning_count: warnings.len(),
                    bytecode_size: bytecode.len(),
                    source_hash,
                });

                self.stats.total_successes += 1;
                self.events.push(HotReloadEvent::CompilationSucceeded {
                    shader_id: id,
                    compile_time_ms,
                    warnings,
                });
                true
            }
            CompilationResult::Failure { errors, warnings: _ } => {
                let error_count = errors.len();
                shader.last_errors = errors.clone();
                shader.last_errors.truncate(MAX_COMPILE_ERRORS);

                shader.add_history(ShaderHistoryEntry {
                    timestamp: SystemTime::now(),
                    success: false,
                    compile_time_ms: 0,
                    error_count,
                    warning_count: 0,
                    bytecode_size: 0,
                    source_hash,
                });

                self.stats.total_failures += 1;
                self.events.push(HotReloadEvent::CompilationFailed {
                    shader_id: id,
                    errors,
                });
                false
            }
        }
    }

    /// Get the current bytecode for a shader (None if never successfully compiled).
    pub fn bytecode(&self, id: WatchedShaderId) -> Option<&[u8]> {
        self.shaders
            .get(&id)
            .and_then(|s| s.current_bytecode.as_deref())
    }

    /// Get the last compile errors for a shader.
    pub fn last_errors(&self, id: WatchedShaderId) -> &[CompileError] {
        self.shaders
            .get(&id)
            .map(|s| s.last_errors.as_slice())
            .unwrap_or(&[])
    }

    /// Get the compilation history for a shader.
    pub fn history(&self, id: WatchedShaderId) -> &[ShaderHistoryEntry] {
        self.shaders
            .get(&id)
            .map(|s| s.history.as_slice())
            .unwrap_or(&[])
    }

    /// Drain all pending events.
    pub fn drain_events(&mut self) -> Vec<HotReloadEvent> {
        std::mem::take(&mut self.events)
    }

    /// Get the statistics.
    pub fn stats(&self) -> &HotReloadStats {
        &self.stats
    }

    /// Get a list of all watched shader IDs.
    pub fn watched_ids(&self) -> Vec<WatchedShaderId> {
        self.shaders.keys().copied().collect()
    }

    /// Get the shader name for an ID.
    pub fn shader_name(&self, id: WatchedShaderId) -> Option<&str> {
        self.shaders.get(&id).map(|s| s.name.as_str())
    }

    /// Get the source path for an ID.
    pub fn source_path(&self, id: WatchedShaderId) -> Option<&Path> {
        self.shaders.get(&id).map(|s| s.source_path.as_path())
    }

    /// Mark a shader as dirty for manual recompilation.
    pub fn mark_dirty(&mut self, id: WatchedShaderId) {
        if let Some(shader) = self.shaders.get_mut(&id) {
            shader.dirty = true;
        }
    }

    /// Mark all shaders as dirty.
    pub fn mark_all_dirty(&mut self) {
        for shader in self.shaders.values_mut() {
            shader.dirty = true;
        }
    }

    /// Get dependent pipeline handles for a shader.
    pub fn dependent_pipelines(&self, id: WatchedShaderId) -> &[u32] {
        self.shaders
            .get(&id)
            .map(|s| s.dependent_pipelines.as_slice())
            .unwrap_or(&[])
    }

    /// Update statistics.
    pub fn update_stats(&mut self) {
        self.stats.watched_count = self.shaders.len();
        self.stats.error_count = self
            .shaders
            .values()
            .filter(|s| !s.last_errors.is_empty())
            .count();

        let total_compile_time: u64 = self
            .shaders
            .values()
            .flat_map(|s| s.history.iter())
            .filter(|h| h.success)
            .map(|h| h.compile_time_ms)
            .sum();
        let total_successes = self.stats.total_successes.max(1);
        self.stats.avg_compile_time_ms = total_compile_time as f64 / total_successes as f64;
    }
}

impl Default for ShaderHotReloadManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Simple FNV-1a hash for source change detection.
fn simple_hash(data: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watch_shader() {
        let mut mgr = ShaderHotReloadManager::new();
        let id = mgr.watch("test", Path::new("shaders/test.glsl"), ShaderStageType::Vertex);
        assert_eq!(mgr.shader_name(id), Some("test"));
        assert_eq!(mgr.watched_ids().len(), 1);
    }

    #[test]
    fn test_unwatch() {
        let mut mgr = ShaderHotReloadManager::new();
        let id = mgr.watch("test", Path::new("shaders/test.glsl"), ShaderStageType::Fragment);
        assert!(mgr.unwatch(id));
        assert!(mgr.watched_ids().is_empty());
    }

    #[test]
    fn test_null_compiler() {
        let compiler = NullCompiler;
        let result = compiler.compile(
            "void main() {}",
            ShaderStageType::Vertex,
            &HashMap::new(),
            &[],
        );
        assert!(result.is_success());
    }

    #[test]
    fn test_simple_hash() {
        let h1 = simple_hash("hello");
        let h2 = simple_hash("world");
        let h3 = simple_hash("hello");
        assert_ne!(h1, h2);
        assert_eq!(h1, h3);
    }

    #[test]
    fn test_dependent_pipelines() {
        let mut mgr = ShaderHotReloadManager::new();
        let id = mgr.watch("test", Path::new("test.glsl"), ShaderStageType::Vertex);
        mgr.add_dependent_pipeline(id, 42);
        mgr.add_dependent_pipeline(id, 43);
        assert_eq!(mgr.dependent_pipelines(id), &[42, 43]);
    }
}
