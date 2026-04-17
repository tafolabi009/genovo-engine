//! Asset hot-reloading for live development.
//!
//! Provides filesystem watching and asset reload coordination without
//! external dependencies. Uses polling-based file watching with modification
//! timestamp comparison and content hashing for reliable change detection.
//!
//! # Features
//!
//! - **FileWatcher**: polls directories for file changes using `std::fs`
//!   metadata timestamps.
//! - **Debouncing**: ignores rapid successive changes, only firing a reload
//!   after a configurable quiet period (default 200ms).
//! - **HotReloadManager**: coordinates file watching, asset reloading, and
//!   dependent notification.
//! - **AssetReloadEvent**: emitted when an asset file changes on disk.

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// AssetType
// ---------------------------------------------------------------------------

/// Broad categories of asset types for reload handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetType {
    /// WGSL, GLSL, HLSL, or SPIR-V shader source.
    Shader,
    /// Texture image (PNG, TGA, DDS, HDR, etc.).
    Texture,
    /// 3D mesh / model (glTF, OBJ, FBX, etc.).
    Mesh,
    /// Material definition.
    Material,
    /// Audio clip (WAV, OGG, etc.).
    Audio,
    /// Scene file.
    Scene,
    /// Font file (TTF, OTF).
    Font,
    /// Script file (Lua, Rhai, etc.).
    Script,
    /// Configuration or data file (JSON, TOML, RON, etc.).
    Config,
    /// Unknown / other.
    Other,
}

impl AssetType {
    /// Infers the asset type from a file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "wgsl" | "glsl" | "vert" | "frag" | "comp" | "hlsl" | "spv" => Self::Shader,
            "png" | "jpg" | "jpeg" | "tga" | "dds" | "hdr" | "bmp" | "webp" => Self::Texture,
            "gltf" | "glb" | "obj" | "fbx" => Self::Mesh,
            "mat" | "material" => Self::Material,
            "wav" | "ogg" | "mp3" | "flac" => Self::Audio,
            "scene" | "scn" => Self::Scene,
            "ttf" | "otf" => Self::Font,
            "lua" | "rhai" | "wasm" => Self::Script,
            "json" | "toml" | "ron" | "yaml" | "yml" | "xml" | "csv" => Self::Config,
            _ => Self::Other,
        }
    }
}

// ---------------------------------------------------------------------------
// WatchedFile
// ---------------------------------------------------------------------------

/// Tracked state of a single file being watched.
#[derive(Debug, Clone)]
pub struct WatchedFile {
    /// Canonical path to the file.
    pub path: PathBuf,
    /// Last known modification timestamp.
    pub last_modified: SystemTime,
    /// Hash of the file contents (for detecting saves that don't change content).
    pub content_hash: u64,
    /// File size in bytes at last check.
    pub file_size: u64,
    /// Inferred asset type.
    pub asset_type: AssetType,
}

impl WatchedFile {
    /// Creates a new watched file entry from the current filesystem state.
    ///
    /// Returns `None` if the file cannot be read or metadata is unavailable.
    pub fn from_path(path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref().to_path_buf();
        let metadata = std::fs::metadata(&path).ok()?;
        let last_modified = metadata.modified().ok()?;
        let file_size = metadata.len();

        let content_hash = Self::compute_hash(&path).unwrap_or(0);

        let asset_type = path
            .extension()
            .and_then(|e| e.to_str())
            .map(AssetType::from_extension)
            .unwrap_or(AssetType::Other);

        Some(Self {
            path,
            last_modified,
            content_hash,
            file_size,
            asset_type,
        })
    }

    /// Computes a fast non-cryptographic hash of the file contents.
    ///
    /// Uses FNV-1a hashing for speed.
    fn compute_hash(path: &Path) -> Option<u64> {
        let data = std::fs::read(path).ok()?;
        let mut hasher = FnvHasher::new();
        hasher.write(&data);
        Some(hasher.finish())
    }

    /// Checks if the file has been modified since last check.
    ///
    /// Compares modification timestamp first (fast), then content hash
    /// if the timestamp changed (to catch false positives from editors
    /// that write the same content).
    pub fn check_modified(&self) -> Option<FileChangeInfo> {
        let metadata = std::fs::metadata(&self.path).ok()?;
        let current_modified = metadata.modified().ok()?;
        let current_size = metadata.len();

        // Quick check: has the timestamp changed?
        if current_modified == self.last_modified && current_size == self.file_size {
            return None;
        }

        // Timestamp changed — verify content actually changed
        let current_hash = Self::compute_hash(&self.path).unwrap_or(0);
        if current_hash == self.content_hash {
            // Same content, just a metadata touch
            return None;
        }

        Some(FileChangeInfo {
            path: self.path.clone(),
            previous_modified: self.last_modified,
            current_modified,
            previous_hash: self.content_hash,
            current_hash,
            previous_size: self.file_size,
            current_size,
            asset_type: self.asset_type,
        })
    }

    /// Updates the cached state to the current filesystem state.
    pub fn refresh(&mut self) -> bool {
        if let Some(updated) = Self::from_path(&self.path) {
            self.last_modified = updated.last_modified;
            self.content_hash = updated.content_hash;
            self.file_size = updated.file_size;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// FnvHasher
// ---------------------------------------------------------------------------

/// Simple FNV-1a hasher for fast content hashing.
struct FnvHasher {
    state: u64,
}

impl FnvHasher {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self {
            state: Self::OFFSET_BASIS,
        }
    }
}

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }
}

// ---------------------------------------------------------------------------
// FileChangeInfo
// ---------------------------------------------------------------------------

/// Information about a detected file change.
#[derive(Debug, Clone)]
pub struct FileChangeInfo {
    /// Path of the changed file.
    pub path: PathBuf,
    /// Previous modification timestamp.
    pub previous_modified: SystemTime,
    /// Current modification timestamp.
    pub current_modified: SystemTime,
    /// Previous content hash.
    pub previous_hash: u64,
    /// Current content hash.
    pub current_hash: u64,
    /// Previous file size.
    pub previous_size: u64,
    /// Current file size.
    pub current_size: u64,
    /// Asset type of the changed file.
    pub asset_type: AssetType,
}

// ---------------------------------------------------------------------------
// FileWatcher
// ---------------------------------------------------------------------------

/// Polls the filesystem for changes by comparing modification timestamps.
///
/// Unlike inotify/FSEvents-based watchers, this works on all platforms
/// without external dependencies. The trade-off is a small poll delay.
#[derive(Debug)]
pub struct FileWatcher {
    /// Watched files keyed by their canonical path.
    files: HashMap<PathBuf, WatchedFile>,
    /// Root directories being watched.
    watch_roots: Vec<PathBuf>,
    /// File extensions to include (empty = all files).
    include_extensions: Vec<String>,
    /// Directories to exclude from scanning.
    exclude_dirs: Vec<String>,
    /// Poll interval.
    poll_interval: Duration,
    /// Last time a poll was performed.
    last_poll: Instant,
    /// Whether recursive directory scanning is enabled.
    recursive: bool,
    /// Maximum file size to hash (larger files use timestamp-only detection).
    max_hash_size: u64,
}

impl FileWatcher {
    /// Creates a new file watcher with default settings.
    ///
    /// Default poll interval is 1 second.
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            watch_roots: Vec::new(),
            include_extensions: Vec::new(),
            exclude_dirs: vec![
                ".git".to_string(),
                "target".to_string(),
                "node_modules".to_string(),
                ".cache".to_string(),
            ],
            poll_interval: Duration::from_secs(1),
            last_poll: Instant::now(),
            recursive: true,
            max_hash_size: 50 * 1024 * 1024, // 50 MB
        }
    }

    /// Sets the poll interval.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Sets the file extensions to watch (e.g., `["wgsl", "png", "gltf"]`).
    /// An empty list watches all files.
    pub fn with_extensions(mut self, extensions: &[&str]) -> Self {
        self.include_extensions = extensions.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Adds directory names to exclude from scanning.
    pub fn with_exclude_dirs(mut self, dirs: &[&str]) -> Self {
        self.exclude_dirs
            .extend(dirs.iter().map(|s| s.to_string()));
        self
    }

    /// Sets whether to scan directories recursively.
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    /// Adds a root directory to watch.
    pub fn watch_directory(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref().to_path_buf();
        if !self.watch_roots.contains(&path) {
            self.watch_roots.push(path.clone());
        }
        self.scan_directory(&path);
    }

    /// Adds a single file to watch.
    pub fn watch_file(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref().to_path_buf();
        if !self.files.contains_key(&path) {
            if let Some(watched) = WatchedFile::from_path(&path) {
                self.files.insert(path, watched);
            }
        }
    }

    /// Removes a file from the watch list.
    pub fn unwatch_file(&mut self, path: impl AsRef<Path>) {
        self.files.remove(path.as_ref());
    }

    /// Scans a directory and adds all matching files to the watch list.
    fn scan_directory(&mut self, dir: &Path) {
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                // Check exclusion list
                if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                    if self.exclude_dirs.contains(&dir_name.to_string()) {
                        continue;
                    }
                }
                if self.recursive {
                    self.scan_directory(&path);
                }
            } else if path.is_file() {
                // Check extension filter
                if !self.include_extensions.is_empty() {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    if !self.include_extensions.contains(&ext) {
                        continue;
                    }
                }

                // Skip files too large to hash
                if let Ok(metadata) = std::fs::metadata(&path) {
                    if metadata.len() > self.max_hash_size {
                        continue;
                    }
                }

                if !self.files.contains_key(&path) {
                    if let Some(watched) = WatchedFile::from_path(&path) {
                        self.files.insert(path, watched);
                    }
                }
            }
        }
    }

    /// Checks all watched files for changes.
    ///
    /// Returns a list of changed files. Also updates the internal state
    /// so the same change is not reported twice.
    pub fn check_changes(&mut self) -> Vec<FileChangeInfo> {
        let mut changes = Vec::new();

        // Collect changed paths first (to avoid borrowing issues)
        let changed_paths: Vec<PathBuf> = self
            .files
            .iter()
            .filter_map(|(path, watched)| {
                watched.check_modified().map(|info| {
                    changes.push(info);
                    path.clone()
                })
            })
            .collect();

        // Refresh the state of changed files
        for path in changed_paths {
            if let Some(watched) = self.files.get_mut(&path) {
                watched.refresh();
            }
        }

        // Also check for new files in watched directories
        for root in self.watch_roots.clone() {
            self.scan_directory(&root);
        }

        // Check for deleted files
        let deleted: Vec<PathBuf> = self
            .files
            .keys()
            .filter(|path| !path.exists())
            .cloned()
            .collect();
        for path in deleted {
            self.files.remove(&path);
        }

        changes
    }

    /// Polls for changes if enough time has elapsed since the last poll.
    ///
    /// Returns changes if the poll interval has been reached, otherwise
    /// returns an empty vec.
    pub fn poll(&mut self) -> Vec<FileChangeInfo> {
        let now = Instant::now();
        if now.duration_since(self.last_poll) < self.poll_interval {
            return Vec::new();
        }
        self.last_poll = now;
        self.check_changes()
    }

    /// Returns the number of watched files.
    pub fn watched_file_count(&self) -> usize {
        self.files.len()
    }

    /// Returns the number of watched root directories.
    pub fn watched_root_count(&self) -> usize {
        self.watch_roots.len()
    }

    /// Returns an iterator over all watched files.
    pub fn watched_files(&self) -> impl Iterator<Item = &WatchedFile> {
        self.files.values()
    }

    /// Returns the poll interval.
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    /// Sets the poll interval.
    pub fn set_poll_interval(&mut self, interval: Duration) {
        self.poll_interval = interval;
    }

    /// Forces an immediate full rescan of all watch roots.
    pub fn rescan(&mut self) {
        self.files.clear();
        for root in self.watch_roots.clone() {
            self.scan_directory(&root);
        }
        self.last_poll = Instant::now();
    }
}

impl Default for FileWatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DebouncedChange
// ---------------------------------------------------------------------------

/// A pending file change waiting for debounce timeout.
#[derive(Debug, Clone)]
struct DebouncedChange {
    /// The change info.
    info: FileChangeInfo,
    /// When the change was first detected.
    first_seen: Instant,
    /// When the most recent change for this path was detected.
    last_seen: Instant,
}

// ---------------------------------------------------------------------------
// AssetReloadEvent
// ---------------------------------------------------------------------------

/// Event emitted when an asset file has changed and should be reloaded.
#[derive(Debug, Clone)]
pub struct AssetReloadEvent {
    /// Path of the changed asset file.
    pub path: PathBuf,
    /// Asset type of the changed file.
    pub asset_type: AssetType,
    /// Previous content hash.
    pub previous_hash: u64,
    /// New content hash.
    pub new_hash: u64,
}

impl fmt::Display for AssetReloadEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AssetReload({:?}: {})",
            self.asset_type,
            self.path.display()
        )
    }
}

// ---------------------------------------------------------------------------
// ReloadCallback
// ---------------------------------------------------------------------------

/// A callback invoked when a specific asset type is reloaded.
type ReloadCallback = Box<dyn Fn(&AssetReloadEvent) + Send + Sync>;

// ---------------------------------------------------------------------------
// HotReloadManager
// ---------------------------------------------------------------------------

/// Coordinates file watching, debouncing, and asset reload dispatch.
///
/// # Usage
///
/// ```rust,ignore
/// let mut manager = HotReloadManager::new();
/// manager.watch_directory("assets/shaders", &["wgsl", "glsl"]);
/// manager.watch_directory("assets/textures", &["png", "tga"]);
///
/// manager.on_reload(AssetType::Shader, |event| {
///     println!("Shader changed: {}", event.path.display());
///     // Re-compile shader pipeline...
/// });
///
/// // In game loop:
/// let events = manager.update();
/// for event in &events {
///     println!("Reloaded: {}", event);
/// }
/// ```
pub struct HotReloadManager {
    /// The file watcher.
    watcher: FileWatcher,
    /// Debounce timeout: changes within this window are merged.
    debounce_timeout: Duration,
    /// Pending debounced changes keyed by path.
    pending: HashMap<PathBuf, DebouncedChange>,
    /// Callbacks keyed by asset type.
    callbacks: HashMap<AssetType, Vec<ReloadCallback>>,
    /// Global callbacks invoked for all asset types.
    global_callbacks: Vec<ReloadCallback>,
    /// Whether hot reloading is enabled.
    enabled: bool,
    /// Total number of reloads performed.
    total_reloads: u64,
    /// Reloads per asset type.
    reloads_by_type: HashMap<AssetType, u64>,
    /// Paths to ignore (e.g. temporary editor files).
    ignore_patterns: Vec<String>,
}

impl fmt::Debug for HotReloadManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HotReloadManager")
            .field("enabled", &self.enabled)
            .field("watched_files", &self.watcher.watched_file_count())
            .field("pending_changes", &self.pending.len())
            .field("total_reloads", &self.total_reloads)
            .field("debounce_ms", &self.debounce_timeout.as_millis())
            .finish()
    }
}

impl HotReloadManager {
    /// Creates a new hot reload manager with default settings.
    pub fn new() -> Self {
        Self {
            watcher: FileWatcher::new(),
            debounce_timeout: Duration::from_millis(200),
            pending: HashMap::new(),
            callbacks: HashMap::new(),
            global_callbacks: Vec::new(),
            enabled: true,
            total_reloads: 0,
            reloads_by_type: HashMap::new(),
            ignore_patterns: vec![
                "~".to_string(),
                ".tmp".to_string(),
                ".swp".to_string(),
                ".bak".to_string(),
            ],
        }
    }

    /// Sets the debounce timeout.
    pub fn with_debounce(mut self, timeout: Duration) -> Self {
        self.debounce_timeout = timeout;
        self
    }

    /// Sets the poll interval.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.watcher.set_poll_interval(interval);
        self
    }

    /// Adds file extension patterns to ignore.
    pub fn with_ignore_patterns(mut self, patterns: &[&str]) -> Self {
        self.ignore_patterns
            .extend(patterns.iter().map(|s| s.to_string()));
        self
    }

    /// Starts watching an asset directory for changes.
    ///
    /// `extensions`: file extensions to watch (empty = all files).
    pub fn watch_directory(
        &mut self,
        path: impl AsRef<Path>,
        extensions: &[&str],
    ) {
        let mut watcher = FileWatcher::new()
            .with_poll_interval(self.watcher.poll_interval())
            .with_extensions(extensions);
        watcher.watch_directory(path);

        // Merge watched files into the main watcher
        for (file_path, watched) in watcher.files.drain() {
            self.watcher.files.insert(file_path, watched);
        }
        for root in watcher.watch_roots {
            if !self.watcher.watch_roots.contains(&root) {
                self.watcher.watch_roots.push(root);
            }
        }
    }

    /// Watches a single file.
    pub fn watch_file(&mut self, path: impl AsRef<Path>) {
        self.watcher.watch_file(path);
    }

    /// Registers a callback for reloads of a specific asset type.
    pub fn on_reload(
        &mut self,
        asset_type: AssetType,
        callback: impl Fn(&AssetReloadEvent) + Send + Sync + 'static,
    ) {
        self.callbacks
            .entry(asset_type)
            .or_default()
            .push(Box::new(callback));
    }

    /// Registers a callback invoked for all asset type reloads.
    pub fn on_any_reload(
        &mut self,
        callback: impl Fn(&AssetReloadEvent) + Send + Sync + 'static,
    ) {
        self.global_callbacks.push(Box::new(callback));
    }

    /// Sets whether hot reloading is enabled.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether hot reloading is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Checks if a path should be ignored (temporary/editor files).
    fn should_ignore(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.ignore_patterns
            .iter()
            .any(|pattern| path_str.ends_with(pattern))
    }

    /// Updates the manager: polls for file changes, applies debouncing,
    /// and dispatches reload events.
    ///
    /// Call this once per frame (or on a timer). Returns the list of
    /// reload events that fired this update.
    pub fn update(&mut self) -> Vec<AssetReloadEvent> {
        if !self.enabled {
            return Vec::new();
        }

        // Poll for raw changes
        let raw_changes = self.watcher.poll();

        // Add new changes to the debounce buffer
        let now = Instant::now();
        for change in raw_changes {
            if self.should_ignore(&change.path) {
                continue;
            }

            self.pending
                .entry(change.path.clone())
                .and_modify(|entry| {
                    entry.info = change.clone();
                    entry.last_seen = now;
                })
                .or_insert(DebouncedChange {
                    info: change,
                    first_seen: now,
                    last_seen: now,
                });
        }

        // Check which pending changes have passed the debounce timeout
        let mut ready_events = Vec::new();
        let mut completed_paths = Vec::new();

        for (path, debounced) in &self.pending {
            let since_last = now.duration_since(debounced.last_seen);
            if since_last >= self.debounce_timeout {
                let event = AssetReloadEvent {
                    path: debounced.info.path.clone(),
                    asset_type: debounced.info.asset_type,
                    previous_hash: debounced.info.previous_hash,
                    new_hash: debounced.info.current_hash,
                };
                ready_events.push(event);
                completed_paths.push(path.clone());
            }
        }

        // Remove completed debounced entries
        for path in &completed_paths {
            self.pending.remove(path);
        }

        // Dispatch callbacks for ready events
        for event in &ready_events {
            self.total_reloads += 1;
            *self.reloads_by_type.entry(event.asset_type).or_insert(0) += 1;

            // Type-specific callbacks
            if let Some(callbacks) = self.callbacks.get(&event.asset_type) {
                for callback in callbacks {
                    callback(event);
                }
            }

            // Global callbacks
            for callback in &self.global_callbacks {
                callback(event);
            }
        }

        ready_events
    }

    /// Returns the total number of reloads performed.
    pub fn total_reloads(&self) -> u64 {
        self.total_reloads
    }

    /// Returns the number of reloads for a specific asset type.
    pub fn reloads_for_type(&self, asset_type: AssetType) -> u64 {
        self.reloads_by_type
            .get(&asset_type)
            .copied()
            .unwrap_or(0)
    }

    /// Returns the number of currently watched files.
    pub fn watched_file_count(&self) -> usize {
        self.watcher.watched_file_count()
    }

    /// Returns the number of pending debounced changes.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Returns a reference to the underlying file watcher.
    pub fn watcher(&self) -> &FileWatcher {
        &self.watcher
    }

    /// Returns a mutable reference to the underlying file watcher.
    pub fn watcher_mut(&mut self) -> &mut FileWatcher {
        &mut self.watcher
    }

    /// Forces an immediate rescan of all watched directories.
    pub fn rescan(&mut self) {
        self.watcher.rescan();
    }

    /// Returns reload statistics.
    pub fn stats(&self) -> HotReloadStats {
        HotReloadStats {
            total_reloads: self.total_reloads,
            watched_files: self.watcher.watched_file_count(),
            watched_roots: self.watcher.watched_root_count(),
            pending_debounced: self.pending.len(),
            reloads_by_type: self.reloads_by_type.clone(),
        }
    }

    /// Resets all statistics.
    pub fn reset_stats(&mut self) {
        self.total_reloads = 0;
        self.reloads_by_type.clear();
    }
}

impl Default for HotReloadManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HotReloadStats
// ---------------------------------------------------------------------------

/// Statistics about the hot reload system.
#[derive(Debug, Clone)]
pub struct HotReloadStats {
    /// Total number of reloads dispatched.
    pub total_reloads: u64,
    /// Number of files being watched.
    pub watched_files: usize,
    /// Number of root directories being watched.
    pub watched_roots: usize,
    /// Number of changes waiting for debounce timeout.
    pub pending_debounced: usize,
    /// Reload counts per asset type.
    pub reloads_by_type: HashMap<AssetType, u64>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "genovo_hot_reload_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup_temp_dir(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_asset_type_from_extension() {
        assert_eq!(AssetType::from_extension("wgsl"), AssetType::Shader);
        assert_eq!(AssetType::from_extension("GLSL"), AssetType::Shader);
        assert_eq!(AssetType::from_extension("png"), AssetType::Texture);
        assert_eq!(AssetType::from_extension("gltf"), AssetType::Mesh);
        assert_eq!(AssetType::from_extension("wav"), AssetType::Audio);
        assert_eq!(AssetType::from_extension("json"), AssetType::Config);
        assert_eq!(AssetType::from_extension("ttf"), AssetType::Font);
        assert_eq!(AssetType::from_extension("lua"), AssetType::Script);
        assert_eq!(AssetType::from_extension("xyz"), AssetType::Other);
    }

    #[test]
    fn test_watched_file_creation() {
        let dir = create_temp_dir();
        let file_path = dir.join("test.wgsl");
        std::fs::write(&file_path, "// test shader").unwrap();

        let watched = WatchedFile::from_path(&file_path).unwrap();
        assert_eq!(watched.path, file_path);
        assert_eq!(watched.asset_type, AssetType::Shader);
        assert!(watched.content_hash != 0);
        assert!(watched.file_size > 0);

        cleanup_temp_dir(&dir);
    }

    #[test]
    fn test_watched_file_no_change() {
        let dir = create_temp_dir();
        let file_path = dir.join("test.txt");
        std::fs::write(&file_path, "hello").unwrap();

        let watched = WatchedFile::from_path(&file_path).unwrap();
        assert!(watched.check_modified().is_none());

        cleanup_temp_dir(&dir);
    }

    #[test]
    fn test_watched_file_detect_change() {
        let dir = create_temp_dir();
        let file_path = dir.join("test.txt");
        std::fs::write(&file_path, "hello").unwrap();

        let watched = WatchedFile::from_path(&file_path).unwrap();

        // Modify the file
        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(&file_path, "world").unwrap();

        let change = watched.check_modified();
        assert!(change.is_some());
        let info = change.unwrap();
        assert_eq!(info.path, file_path);
        assert_ne!(info.previous_hash, info.current_hash);

        cleanup_temp_dir(&dir);
    }

    #[test]
    fn test_file_watcher_scan() {
        let dir = create_temp_dir();
        std::fs::write(dir.join("a.wgsl"), "shader a").unwrap();
        std::fs::write(dir.join("b.wgsl"), "shader b").unwrap();
        std::fs::write(dir.join("c.txt"), "not a shader").unwrap();

        let mut watcher = FileWatcher::new().with_extensions(&["wgsl"]);
        watcher.watch_directory(&dir);

        assert_eq!(watcher.watched_file_count(), 2);

        cleanup_temp_dir(&dir);
    }

    #[test]
    fn test_file_watcher_detect_changes() {
        let dir = create_temp_dir();
        let file_path = dir.join("test.wgsl");
        std::fs::write(&file_path, "version 1").unwrap();

        let mut watcher = FileWatcher::new();
        watcher.watch_file(&file_path);
        assert_eq!(watcher.watched_file_count(), 1);

        // No changes yet
        let changes = watcher.check_changes();
        assert!(changes.is_empty());

        // Modify the file
        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(&file_path, "version 2").unwrap();

        let changes = watcher.check_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, file_path);

        // After refresh, no more changes
        let changes = watcher.check_changes();
        assert!(changes.is_empty());

        cleanup_temp_dir(&dir);
    }

    #[test]
    fn test_fnv_hasher() {
        let mut h1 = FnvHasher::new();
        h1.write(b"hello");
        let hash1 = h1.finish();

        let mut h2 = FnvHasher::new();
        h2.write(b"hello");
        let hash2 = h2.finish();

        let mut h3 = FnvHasher::new();
        h3.write(b"world");
        let hash3 = h3.finish();

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_hot_reload_manager_creation() {
        let manager = HotReloadManager::new();
        assert!(manager.is_enabled());
        assert_eq!(manager.total_reloads(), 0);
        assert_eq!(manager.watched_file_count(), 0);
    }

    #[test]
    fn test_ignore_patterns() {
        let manager = HotReloadManager::new();
        assert!(manager.should_ignore(Path::new("file.tmp")));
        assert!(manager.should_ignore(Path::new("file~")));
        assert!(manager.should_ignore(Path::new("file.swp")));
        assert!(!manager.should_ignore(Path::new("file.wgsl")));
    }

    #[test]
    fn test_stats() {
        let manager = HotReloadManager::new();
        let stats = manager.stats();
        assert_eq!(stats.total_reloads, 0);
        assert_eq!(stats.watched_files, 0);
        assert!(stats.reloads_by_type.is_empty());
    }

    #[test]
    fn test_watcher_exclude_dirs() {
        let dir = create_temp_dir();
        let git_dir = dir.join(".git");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::write(git_dir.join("config"), "git config").unwrap();
        std::fs::write(dir.join("main.rs"), "fn main() {}").unwrap();

        let mut watcher = FileWatcher::new();
        watcher.watch_directory(&dir);

        // .git directory should be excluded
        let has_git_file = watcher
            .watched_files()
            .any(|f| f.path.to_string_lossy().contains(".git"));
        assert!(!has_git_file);

        cleanup_temp_dir(&dir);
    }

    #[test]
    fn test_unwatch_file() {
        let dir = create_temp_dir();
        let file_path = dir.join("test.txt");
        std::fs::write(&file_path, "content").unwrap();

        let mut watcher = FileWatcher::new();
        watcher.watch_file(&file_path);
        assert_eq!(watcher.watched_file_count(), 1);

        watcher.unwatch_file(&file_path);
        assert_eq!(watcher.watched_file_count(), 0);

        cleanup_temp_dir(&dir);
    }

    #[test]
    fn test_watched_file_refresh() {
        let dir = create_temp_dir();
        let file_path = dir.join("test.txt");
        std::fs::write(&file_path, "v1").unwrap();

        let mut watched = WatchedFile::from_path(&file_path).unwrap();
        let old_hash = watched.content_hash;

        std::thread::sleep(Duration::from_millis(50));
        std::fs::write(&file_path, "v2 with more content").unwrap();

        assert!(watched.refresh());
        assert_ne!(watched.content_hash, old_hash);

        cleanup_temp_dir(&dir);
    }

    #[test]
    fn test_asset_reload_event_display() {
        let event = AssetReloadEvent {
            path: PathBuf::from("assets/shaders/test.wgsl"),
            asset_type: AssetType::Shader,
            previous_hash: 123,
            new_hash: 456,
        };
        let display = format!("{}", event);
        assert!(display.contains("Shader"));
        assert!(display.contains("test.wgsl"));
    }

    #[test]
    fn test_disabled_manager() {
        let mut manager = HotReloadManager::new();
        manager.set_enabled(false);
        let events = manager.update();
        assert!(events.is_empty());
    }

    #[test]
    fn test_poll_interval_throttle() {
        let dir = create_temp_dir();
        let file_path = dir.join("test.txt");
        std::fs::write(&file_path, "content").unwrap();

        let mut watcher = FileWatcher::new()
            .with_poll_interval(Duration::from_secs(10)); // very long interval
        watcher.watch_file(&file_path);

        // First poll should work (enough time has passed since creation)
        let _ = watcher.poll();

        // Immediate second poll should return empty (interval not elapsed)
        let changes = watcher.poll();
        assert!(changes.is_empty());

        cleanup_temp_dir(&dir);
    }
}
