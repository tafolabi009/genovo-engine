//! # Native File Dialogs
//!
//! Provides cross-platform native file dialog support: open file, save file,
//! select folder, file type filters, multi-select, default path, dialog title,
//! recent files tracking, and last-used directory memory.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from file dialog operations.
#[derive(Debug, Clone)]
pub enum FileDialogError {
    /// The user cancelled the dialog.
    Cancelled,
    /// The selected path does not exist.
    PathNotFound(PathBuf),
    /// A platform error occurred.
    PlatformError(String),
    /// The dialog type is not supported on this platform.
    NotSupported(String),
    /// An invalid filter specification was provided.
    InvalidFilter(String),
}

impl fmt::Display for FileDialogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => write!(f, "dialog cancelled"),
            Self::PathNotFound(p) => write!(f, "path not found: {}", p.display()),
            Self::PlatformError(msg) => write!(f, "platform error: {msg}"),
            Self::NotSupported(msg) => write!(f, "not supported: {msg}"),
            Self::InvalidFilter(msg) => write!(f, "invalid filter: {msg}"),
        }
    }
}

impl std::error::Error for FileDialogError {}

/// Result type for file dialog operations.
pub type FileDialogResult<T> = Result<T, FileDialogError>;

// ---------------------------------------------------------------------------
// File type filter
// ---------------------------------------------------------------------------

/// A file type filter for file dialogs (e.g. "Image files (*.png, *.jpg)").
#[derive(Debug, Clone)]
pub struct FileFilter {
    /// Display name (e.g. "Image Files").
    pub name: String,
    /// File extensions without dots (e.g. ["png", "jpg", "jpeg"]).
    pub extensions: Vec<String>,
}

impl FileFilter {
    /// Create a new file filter.
    pub fn new(name: impl Into<String>, extensions: &[&str]) -> Self {
        Self {
            name: name.into(),
            extensions: extensions.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Check if a path matches this filter.
    pub fn matches(&self, path: &Path) -> bool {
        if self.extensions.is_empty() {
            return true; // "All files"
        }
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            self.extensions
                .iter()
                .any(|e| e.to_lowercase() == ext_lower)
        } else {
            false
        }
    }

    /// Format extensions for display (e.g. "*.png;*.jpg").
    pub fn pattern_string(&self) -> String {
        if self.extensions.is_empty() {
            return "*.*".to_string();
        }
        self.extensions
            .iter()
            .map(|e| format!("*.{e}"))
            .collect::<Vec<_>>()
            .join(";")
    }

    /// Full display string (e.g. "Image Files (*.png;*.jpg)").
    pub fn display_string(&self) -> String {
        format!("{} ({})", self.name, self.pattern_string())
    }
}

/// Common file filters.
impl FileFilter {
    /// All files.
    pub fn all_files() -> Self {
        Self::new("All Files", &[])
    }

    /// Image files.
    pub fn images() -> Self {
        Self::new("Image Files", &["png", "jpg", "jpeg", "bmp", "tga", "hdr", "dds"])
    }

    /// 3D model files.
    pub fn models() -> Self {
        Self::new("3D Models", &["gltf", "glb", "obj", "fbx"])
    }

    /// Audio files.
    pub fn audio() -> Self {
        Self::new("Audio Files", &["wav", "ogg", "mp3", "flac"])
    }

    /// Scene files.
    pub fn scenes() -> Self {
        Self::new("Scene Files", &["scene", "json", "yaml"])
    }

    /// Shader files.
    pub fn shaders() -> Self {
        Self::new("Shader Files", &["wgsl", "glsl", "hlsl", "spv"])
    }

    /// Font files.
    pub fn fonts() -> Self {
        Self::new("Font Files", &["ttf", "otf", "woff", "woff2"])
    }

    /// Script files.
    pub fn scripts() -> Self {
        Self::new("Script Files", &["lua", "py", "js"])
    }

    /// Text files.
    pub fn text() -> Self {
        Self::new("Text Files", &["txt", "md", "csv", "log"])
    }
}

// ---------------------------------------------------------------------------
// Dialog builder
// ---------------------------------------------------------------------------

/// The type of file dialog to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogType {
    /// Open a single file.
    OpenFile,
    /// Open multiple files.
    OpenMultiple,
    /// Save a file.
    SaveFile,
    /// Select a folder.
    SelectFolder,
}

/// Builder for configuring a file dialog before showing it.
#[derive(Debug, Clone)]
pub struct FileDialogBuilder {
    /// Dialog type.
    pub dialog_type: DialogType,
    /// Window title.
    pub title: String,
    /// Initial directory to open in.
    pub default_path: Option<PathBuf>,
    /// Default file name (for save dialogs).
    pub default_name: Option<String>,
    /// File type filters.
    pub filters: Vec<FileFilter>,
    /// Index of the initially selected filter (0-based).
    pub selected_filter: usize,
    /// Whether to show hidden files.
    pub show_hidden: bool,
    /// Whether to allow creating new directories (save dialog).
    pub can_create_dir: bool,
    /// Custom OK button label.
    pub ok_label: Option<String>,
    /// Custom Cancel button label.
    pub cancel_label: Option<String>,
    /// Owner window handle (platform-specific).
    pub owner_handle: Option<u64>,
}

impl FileDialogBuilder {
    /// Create a new open-file dialog builder.
    pub fn open_file() -> Self {
        Self {
            dialog_type: DialogType::OpenFile,
            title: "Open File".into(),
            default_path: None,
            default_name: None,
            filters: Vec::new(),
            selected_filter: 0,
            show_hidden: false,
            can_create_dir: false,
            ok_label: None,
            cancel_label: None,
            owner_handle: None,
        }
    }

    /// Create a new open-multiple dialog builder.
    pub fn open_multiple() -> Self {
        Self {
            dialog_type: DialogType::OpenMultiple,
            title: "Open Files".into(),
            ..Self::open_file()
        }
    }

    /// Create a new save-file dialog builder.
    pub fn save_file() -> Self {
        Self {
            dialog_type: DialogType::SaveFile,
            title: "Save File".into(),
            can_create_dir: true,
            ..Self::open_file()
        }
    }

    /// Create a new select-folder dialog builder.
    pub fn select_folder() -> Self {
        Self {
            dialog_type: DialogType::SelectFolder,
            title: "Select Folder".into(),
            ..Self::open_file()
        }
    }

    /// Set the dialog title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the initial directory.
    pub fn default_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.default_path = Some(path.into());
        self
    }

    /// Set the default file name (save dialog).
    pub fn default_name(mut self, name: impl Into<String>) -> Self {
        self.default_name = Some(name.into());
        self
    }

    /// Add a file filter.
    pub fn filter(mut self, filter: FileFilter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Add multiple filters at once.
    pub fn filters(mut self, filters: Vec<FileFilter>) -> Self {
        self.filters.extend(filters);
        self
    }

    /// Set the initially selected filter index.
    pub fn selected_filter_index(mut self, index: usize) -> Self {
        self.selected_filter = index;
        self
    }

    /// Show hidden files.
    pub fn show_hidden(mut self, show: bool) -> Self {
        self.show_hidden = show;
        self
    }

    /// Allow creating new directories.
    pub fn can_create_dir(mut self, allow: bool) -> Self {
        self.can_create_dir = allow;
        self
    }

    /// Set a custom OK button label.
    pub fn ok_label(mut self, label: impl Into<String>) -> Self {
        self.ok_label = Some(label.into());
        self
    }

    /// Set a custom Cancel button label.
    pub fn cancel_label(mut self, label: impl Into<String>) -> Self {
        self.cancel_label = Some(label.into());
        self
    }

    /// Set the owner window handle.
    pub fn owner(mut self, handle: u64) -> Self {
        self.owner_handle = Some(handle);
        self
    }
}

// ---------------------------------------------------------------------------
// Dialog result
// ---------------------------------------------------------------------------

/// The result of showing a file dialog.
#[derive(Debug, Clone)]
pub struct FileDialogResponse {
    /// Selected file paths (empty if cancelled).
    pub paths: Vec<PathBuf>,
    /// Index of the filter that was selected by the user.
    pub selected_filter: usize,
    /// Whether the dialog was confirmed (not cancelled).
    pub confirmed: bool,
}

impl FileDialogResponse {
    /// Create a cancelled response.
    pub fn cancelled() -> Self {
        Self {
            paths: Vec::new(),
            selected_filter: 0,
            confirmed: false,
        }
    }

    /// Create a confirmed response with a single path.
    pub fn single(path: PathBuf) -> Self {
        Self {
            paths: vec![path],
            selected_filter: 0,
            confirmed: true,
        }
    }

    /// Create a confirmed response with multiple paths.
    pub fn multiple(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            selected_filter: 0,
            confirmed: true,
        }
    }

    /// Get the first selected path (convenience for single-select dialogs).
    pub fn path(&self) -> Option<&Path> {
        self.paths.first().map(|p| p.as_path())
    }

    /// Number of selected paths.
    pub fn count(&self) -> usize {
        self.paths.len()
    }
}

// ---------------------------------------------------------------------------
// File dialog trait
// ---------------------------------------------------------------------------

/// Platform-agnostic file dialog interface.
pub trait FileDialog: Send + Sync {
    /// Show the dialog and block until the user makes a selection or cancels.
    fn show(&self, builder: &FileDialogBuilder) -> FileDialogResult<FileDialogResponse>;
}

// ---------------------------------------------------------------------------
// Software (stub) implementation
// ---------------------------------------------------------------------------

/// A software file dialog that returns preconfigured paths for testing.
pub struct SoftwareFileDialog {
    /// Preconfigured responses for testing.
    responses: Vec<FileDialogResponse>,
    /// Index into responses.
    call_count: std::sync::atomic::AtomicUsize,
}

impl SoftwareFileDialog {
    /// Create with a list of preconfigured responses.
    pub fn new(responses: Vec<FileDialogResponse>) -> Self {
        Self {
            responses,
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Create one that always cancels.
    pub fn always_cancel() -> Self {
        Self::new(vec![FileDialogResponse::cancelled()])
    }

    /// Create one that always returns a specific path.
    pub fn always_return(path: PathBuf) -> Self {
        Self::new(vec![FileDialogResponse::single(path)])
    }
}

impl FileDialog for SoftwareFileDialog {
    fn show(&self, _builder: &FileDialogBuilder) -> FileDialogResult<FileDialogResponse> {
        let idx = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if idx < self.responses.len() {
            Ok(self.responses[idx].clone())
        } else if !self.responses.is_empty() {
            Ok(self.responses.last().unwrap().clone())
        } else {
            Err(FileDialogError::Cancelled)
        }
    }
}

// ---------------------------------------------------------------------------
// Recent files tracker
// ---------------------------------------------------------------------------

/// Tracks recently opened/saved files for quick access.
pub struct RecentFiles {
    /// List of recent file paths (most recent first).
    files: VecDeque<RecentFileEntry>,
    /// Maximum number of entries.
    max_entries: usize,
}

/// An entry in the recent files list.
#[derive(Debug, Clone)]
pub struct RecentFileEntry {
    /// Full file path.
    pub path: PathBuf,
    /// When the file was last accessed.
    pub last_accessed: SystemTime,
    /// Number of times the file has been opened.
    pub access_count: u32,
    /// Whether the file is pinned (always visible).
    pub pinned: bool,
    /// Optional display name override.
    pub display_name: Option<String>,
}

impl RecentFileEntry {
    /// Create a new entry.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            last_accessed: SystemTime::now(),
            access_count: 1,
            pinned: false,
            display_name: None,
        }
    }

    /// Get the display name (file name by default).
    pub fn display(&self) -> &str {
        if let Some(ref name) = self.display_name {
            return name;
        }
        self.path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
    }

    /// Get the parent directory.
    pub fn directory(&self) -> Option<&Path> {
        self.path.parent()
    }

    /// Get the file extension.
    pub fn extension(&self) -> Option<&str> {
        self.path.extension().and_then(|e| e.to_str())
    }
}

impl RecentFiles {
    /// Create a new recent files tracker.
    pub fn new(max_entries: usize) -> Self {
        Self {
            files: VecDeque::new(),
            max_entries,
        }
    }

    /// Record a file as recently accessed.
    pub fn record(&mut self, path: PathBuf) {
        // If already present, update it and move to front
        if let Some(pos) = self.files.iter().position(|e| e.path == path) {
            let mut entry = self.files.remove(pos).unwrap();
            entry.last_accessed = SystemTime::now();
            entry.access_count += 1;
            self.files.push_front(entry);
        } else {
            // Add new entry
            if self.files.len() >= self.max_entries {
                // Remove oldest non-pinned entry
                if let Some(pos) = self.files.iter().rposition(|e| !e.pinned) {
                    self.files.remove(pos);
                }
            }
            self.files.push_front(RecentFileEntry::new(path));
        }
    }

    /// Get all recent file entries.
    pub fn entries(&self) -> &VecDeque<RecentFileEntry> {
        &self.files
    }

    /// Get the most recent entry.
    pub fn most_recent(&self) -> Option<&RecentFileEntry> {
        self.files.front()
    }

    /// Get entries filtered by extension.
    pub fn filter_by_extension(&self, ext: &str) -> Vec<&RecentFileEntry> {
        let ext_lower = ext.to_lowercase();
        self.files
            .iter()
            .filter(|e| {
                e.extension()
                    .map(|x| x.to_lowercase() == ext_lower)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.files.clear();
    }

    /// Remove entries where the file no longer exists on disk.
    pub fn prune_missing(&mut self) {
        self.files.retain(|e| e.path.exists() || e.pinned);
    }

    /// Pin an entry.
    pub fn pin(&mut self, path: &Path) {
        if let Some(entry) = self.files.iter_mut().find(|e| e.path == path) {
            entry.pinned = true;
        }
    }

    /// Unpin an entry.
    pub fn unpin(&mut self, path: &Path) {
        if let Some(entry) = self.files.iter_mut().find(|e| e.path == path) {
            entry.pinned = false;
        }
    }
}

impl Default for RecentFiles {
    fn default() -> Self {
        Self::new(20)
    }
}

// ---------------------------------------------------------------------------
// Last-used directory memory
// ---------------------------------------------------------------------------

/// Remembers the last directory used for each dialog context (e.g. "open_texture",
/// "save_scene"), so the next dialog opens in the same location.
pub struct DirectoryMemory {
    /// Context name -> last used directory.
    directories: HashMap<String, PathBuf>,
}

impl DirectoryMemory {
    /// Create a new directory memory.
    pub fn new() -> Self {
        Self {
            directories: HashMap::new(),
        }
    }

    /// Remember the directory for a context.
    pub fn remember(&mut self, context: impl Into<String>, dir: PathBuf) {
        self.directories.insert(context.into(), dir);
    }

    /// Remember the directory from a file path (uses the parent directory).
    pub fn remember_from_file(&mut self, context: impl Into<String>, file_path: &Path) {
        if let Some(parent) = file_path.parent() {
            self.directories
                .insert(context.into(), parent.to_path_buf());
        }
    }

    /// Recall the last directory for a context.
    pub fn recall(&self, context: &str) -> Option<&Path> {
        self.directories.get(context).map(|p| p.as_path())
    }

    /// Clear all remembered directories.
    pub fn clear(&mut self) {
        self.directories.clear();
    }

    /// Number of remembered contexts.
    pub fn context_count(&self) -> usize {
        self.directories.len()
    }
}

impl Default for DirectoryMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Show a simple open-file dialog with common filters.
pub fn open_file_dialog(
    dialog: &dyn FileDialog,
    title: &str,
    filters: Vec<FileFilter>,
    default_path: Option<PathBuf>,
) -> FileDialogResult<Option<PathBuf>> {
    let mut builder = FileDialogBuilder::open_file().title(title).filters(filters);
    if let Some(path) = default_path {
        builder = builder.default_path(path);
    }
    let response = dialog.show(&builder)?;
    Ok(response.path().map(|p| p.to_path_buf()))
}

/// Show a simple save-file dialog.
pub fn save_file_dialog(
    dialog: &dyn FileDialog,
    title: &str,
    default_name: Option<&str>,
    filters: Vec<FileFilter>,
    default_path: Option<PathBuf>,
) -> FileDialogResult<Option<PathBuf>> {
    let mut builder = FileDialogBuilder::save_file().title(title).filters(filters);
    if let Some(name) = default_name {
        builder = builder.default_name(name);
    }
    if let Some(path) = default_path {
        builder = builder.default_path(path);
    }
    let response = dialog.show(&builder)?;
    Ok(response.path().map(|p| p.to_path_buf()))
}

/// Show a folder selection dialog.
pub fn select_folder_dialog(
    dialog: &dyn FileDialog,
    title: &str,
    default_path: Option<PathBuf>,
) -> FileDialogResult<Option<PathBuf>> {
    let mut builder = FileDialogBuilder::select_folder().title(title);
    if let Some(path) = default_path {
        builder = builder.default_path(path);
    }
    let response = dialog.show(&builder)?;
    Ok(response.path().map(|p| p.to_path_buf()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_filter_matches() {
        let filter = FileFilter::images();
        assert!(filter.matches(Path::new("texture.png")));
        assert!(filter.matches(Path::new("photo.JPG")));
        assert!(!filter.matches(Path::new("script.lua")));
    }

    #[test]
    fn test_file_filter_all_files() {
        let filter = FileFilter::all_files();
        assert!(filter.matches(Path::new("anything.xyz")));
    }

    #[test]
    fn test_file_filter_pattern_string() {
        let filter = FileFilter::new("Test", &["png", "jpg"]);
        assert_eq!(filter.pattern_string(), "*.png;*.jpg");
    }

    #[test]
    fn test_file_filter_display_string() {
        let filter = FileFilter::new("Images", &["png"]);
        assert_eq!(filter.display_string(), "Images (*.png)");
    }

    #[test]
    fn test_dialog_builder_open() {
        let builder = FileDialogBuilder::open_file()
            .title("Open Texture")
            .filter(FileFilter::images())
            .default_path("/home/user");
        assert_eq!(builder.dialog_type, DialogType::OpenFile);
        assert_eq!(builder.title, "Open Texture");
        assert_eq!(builder.filters.len(), 1);
    }

    #[test]
    fn test_dialog_builder_save() {
        let builder = FileDialogBuilder::save_file()
            .title("Save Scene")
            .default_name("untitled.scene");
        assert_eq!(builder.dialog_type, DialogType::SaveFile);
        assert!(builder.can_create_dir);
        assert_eq!(builder.default_name.unwrap(), "untitled.scene");
    }

    #[test]
    fn test_dialog_response_single() {
        let response = FileDialogResponse::single(PathBuf::from("/tmp/test.png"));
        assert!(response.confirmed);
        assert_eq!(response.count(), 1);
        assert_eq!(response.path().unwrap().to_str(), Some("/tmp/test.png"));
    }

    #[test]
    fn test_dialog_response_cancelled() {
        let response = FileDialogResponse::cancelled();
        assert!(!response.confirmed);
        assert_eq!(response.count(), 0);
        assert!(response.path().is_none());
    }

    #[test]
    fn test_software_dialog_returns() {
        let dialog = SoftwareFileDialog::always_return(PathBuf::from("/test/file.txt"));
        let builder = FileDialogBuilder::open_file();
        let response = dialog.show(&builder).unwrap();
        assert!(response.confirmed);
        assert_eq!(response.path().unwrap().to_str(), Some("/test/file.txt"));
    }

    #[test]
    fn test_software_dialog_cancels() {
        let dialog = SoftwareFileDialog::always_cancel();
        let builder = FileDialogBuilder::open_file();
        let response = dialog.show(&builder).unwrap();
        assert!(!response.confirmed);
    }

    #[test]
    fn test_recent_files_basic() {
        let mut recent = RecentFiles::new(5);
        recent.record(PathBuf::from("/a.txt"));
        recent.record(PathBuf::from("/b.txt"));
        recent.record(PathBuf::from("/c.txt"));
        assert_eq!(recent.len(), 3);
        assert_eq!(
            recent.most_recent().unwrap().path,
            PathBuf::from("/c.txt")
        );
    }

    #[test]
    fn test_recent_files_dedup() {
        let mut recent = RecentFiles::new(5);
        recent.record(PathBuf::from("/a.txt"));
        recent.record(PathBuf::from("/b.txt"));
        recent.record(PathBuf::from("/a.txt")); // should move to front
        assert_eq!(recent.len(), 2);
        assert_eq!(
            recent.most_recent().unwrap().path,
            PathBuf::from("/a.txt")
        );
        assert_eq!(recent.most_recent().unwrap().access_count, 2);
    }

    #[test]
    fn test_recent_files_eviction() {
        let mut recent = RecentFiles::new(3);
        recent.record(PathBuf::from("/1.txt"));
        recent.record(PathBuf::from("/2.txt"));
        recent.record(PathBuf::from("/3.txt"));
        recent.record(PathBuf::from("/4.txt"));
        assert_eq!(recent.len(), 3);
        // Oldest (/1.txt) should be evicted
        let paths: Vec<&Path> = recent.entries().iter().map(|e| e.path.as_path()).collect();
        assert!(!paths.contains(&Path::new("/1.txt")));
    }

    #[test]
    fn test_recent_files_filter_extension() {
        let mut recent = RecentFiles::new(10);
        recent.record(PathBuf::from("/a.png"));
        recent.record(PathBuf::from("/b.txt"));
        recent.record(PathBuf::from("/c.png"));
        let pngs = recent.filter_by_extension("png");
        assert_eq!(pngs.len(), 2);
    }

    #[test]
    fn test_directory_memory() {
        let mut mem = DirectoryMemory::new();
        mem.remember("open_texture", PathBuf::from("/assets/textures"));
        assert_eq!(
            mem.recall("open_texture").unwrap(),
            Path::new("/assets/textures")
        );
        assert!(mem.recall("open_model").is_none());
    }

    #[test]
    fn test_directory_memory_from_file() {
        let mut mem = DirectoryMemory::new();
        mem.remember_from_file("save_scene", Path::new("/project/scenes/level1.scene"));
        assert_eq!(
            mem.recall("save_scene").unwrap(),
            Path::new("/project/scenes")
        );
    }

    #[test]
    fn test_recent_file_entry_display() {
        let entry = RecentFileEntry::new(PathBuf::from("/data/hero_texture.png"));
        assert_eq!(entry.display(), "hero_texture.png");
        assert_eq!(entry.extension(), Some("png"));
    }

    #[test]
    fn test_recent_file_entry_custom_name() {
        let mut entry = RecentFileEntry::new(PathBuf::from("/data/file.bin"));
        entry.display_name = Some("My Custom Name".into());
        assert_eq!(entry.display(), "My Custom Name");
    }

    #[test]
    fn test_open_file_dialog_helper() {
        let dialog = SoftwareFileDialog::always_return(PathBuf::from("/selected.png"));
        let result = open_file_dialog(
            &dialog,
            "Open",
            vec![FileFilter::images()],
            None,
        )
        .unwrap();
        assert_eq!(result, Some(PathBuf::from("/selected.png")));
    }

    #[test]
    fn test_dialog_type_values() {
        assert_ne!(DialogType::OpenFile, DialogType::SaveFile);
        assert_ne!(DialogType::OpenFile, DialogType::SelectFolder);
    }

    #[test]
    fn test_file_dialog_error_display() {
        assert_eq!(
            FileDialogError::Cancelled.to_string(),
            "dialog cancelled"
        );
    }

    #[test]
    fn test_common_filters() {
        assert!(!FileFilter::images().extensions.is_empty());
        assert!(!FileFilter::models().extensions.is_empty());
        assert!(!FileFilter::audio().extensions.is_empty());
        assert!(!FileFilter::shaders().extensions.is_empty());
    }
}
