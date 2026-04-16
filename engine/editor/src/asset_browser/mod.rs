//! # Asset Browser
//!
//! Provides a file-system backed asset browser panel with directory tree
//! navigation, thumbnail previews, asset importing, drag-and-drop support,
//! and fuzzy search.
//!
//! The browser displays project assets organized in a directory tree with
//! extension-based type detection and an LRU thumbnail cache for preview
//! images.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Asset Kind (type detection)
// ---------------------------------------------------------------------------

/// Broad classification of an asset based on its file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetKind {
    Folder,
    Mesh,
    Texture,
    Audio,
    Script,
    Material,
    Prefab,
    Scene,
    Shader,
    Font,
    Animation,
    Unknown,
}

impl AssetKind {
    /// Detect asset kind from a file extension (case-insensitive).
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            // Meshes
            "obj" | "fbx" | "gltf" | "glb" | "dae" | "blend" | "stl" | "ply" => Self::Mesh,
            // Textures
            "png" | "jpg" | "jpeg" | "bmp" | "tga" | "dds" | "ktx" | "ktx2" | "hdr" | "exr"
            | "webp" | "svg" | "tiff" | "tif" => Self::Texture,
            // Audio
            "wav" | "ogg" | "mp3" | "flac" | "aac" | "opus" => Self::Audio,
            // Scripts
            "lua" | "wasm" | "rs" | "py" | "js" | "ts" => Self::Script,
            // Materials
            "mat" | "material" => Self::Material,
            // Prefabs
            "prefab" => Self::Prefab,
            // Scenes
            "scene" | "ron" | "json" => Self::Scene,
            // Shaders
            "wgsl" | "glsl" | "hlsl" | "vert" | "frag" | "comp" | "spv" => Self::Shader,
            // Fonts
            "ttf" | "otf" | "woff" | "woff2" => Self::Font,
            // Animations
            "anim" | "animation" => Self::Animation,
            _ => Self::Unknown,
        }
    }

    /// Return a display-friendly name for this asset kind.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Folder => "Folder",
            Self::Mesh => "Mesh",
            Self::Texture => "Texture",
            Self::Audio => "Audio",
            Self::Script => "Script",
            Self::Material => "Material",
            Self::Prefab => "Prefab",
            Self::Scene => "Scene",
            Self::Shader => "Shader",
            Self::Font => "Font",
            Self::Animation => "Animation",
            Self::Unknown => "Unknown",
        }
    }

    /// Return a representative icon character for this asset kind.
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Folder => "[D]",
            Self::Mesh => "[M]",
            Self::Texture => "[T]",
            Self::Audio => "[A]",
            Self::Script => "[S]",
            Self::Material => "[Mt]",
            Self::Prefab => "[P]",
            Self::Scene => "[Sc]",
            Self::Shader => "[Sh]",
            Self::Font => "[F]",
            Self::Animation => "[An]",
            Self::Unknown => "[?]",
        }
    }
}

// ---------------------------------------------------------------------------
// Asset Entry
// ---------------------------------------------------------------------------

/// A single file or folder entry displayed in the asset browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetEntry {
    /// Full path relative to the asset root.
    pub path: PathBuf,
    /// File/folder name (last component of path).
    pub name: String,
    /// Kind of asset (determined by extension).
    pub kind: AssetKind,
    /// File size in bytes (0 for folders).
    pub size_bytes: u64,
    /// Asset UUID (assigned by the asset database, if any).
    pub asset_id: Option<Uuid>,
    /// File extension (empty string for folders).
    pub extension: String,
    /// Whether this entry is a directory.
    pub is_directory: bool,
    /// Last modified timestamp (seconds since epoch).
    pub last_modified: u64,
}

impl AssetEntry {
    /// Create a new file entry.
    pub fn new_file(path: PathBuf, size_bytes: u64, last_modified: u64) -> Self {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_string())
            .unwrap_or_default();
        let kind = AssetKind::from_extension(&extension);
        Self {
            path,
            name,
            kind,
            size_bytes,
            asset_id: None,
            extension,
            is_directory: false,
            last_modified,
        }
    }

    /// Create a new folder entry.
    pub fn new_folder(path: PathBuf) -> Self {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        Self {
            path,
            name,
            kind: AssetKind::Folder,
            size_bytes: 0,
            asset_id: None,
            extension: String::new(),
            is_directory: true,
            last_modified: 0,
        }
    }

    /// Format the file size for display.
    pub fn display_size(&self) -> String {
        if self.is_directory {
            return String::new();
        }
        if self.size_bytes < 1024 {
            format!("{} B", self.size_bytes)
        } else if self.size_bytes < 1024 * 1024 {
            format!("{:.1} KB", self.size_bytes as f64 / 1024.0)
        } else if self.size_bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", self.size_bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!(
                "{:.2} GB",
                self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Directory Entry (tree structure for the directory panel)
// ---------------------------------------------------------------------------

/// An entry in the directory tree (left panel of the asset browser).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryEntry {
    /// Entry name (file or directory name).
    pub name: String,
    /// Full path relative to asset root.
    pub path: PathBuf,
    /// Whether this is a directory.
    pub is_directory: bool,
    /// Children (populated for directories).
    pub children: Vec<DirectoryEntry>,
    /// Asset UUID if this is a registered asset.
    pub asset_id: Option<Uuid>,
    /// File extension.
    pub extension: Option<String>,
    /// Whether this directory is expanded in the tree view.
    pub expanded: bool,
}

impl DirectoryEntry {
    /// Create a new directory entry for the tree panel.
    pub fn new_directory(name: impl Into<String>, path: PathBuf) -> Self {
        Self {
            name: name.into(),
            path,
            is_directory: true,
            children: Vec::new(),
            asset_id: None,
            extension: None,
            expanded: false,
        }
    }

    /// Recursively count all entries.
    pub fn count(&self) -> usize {
        1 + self.children.iter().map(|c| c.count()).sum::<usize>()
    }

    /// Find a directory entry by path.
    pub fn find(&self, target: &Path) -> Option<&DirectoryEntry> {
        if self.path == target {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find(target) {
                return Some(found);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Thumbnail Cache
// ---------------------------------------------------------------------------

/// State of a thumbnail in the cache.
#[derive(Debug, Clone)]
pub enum ThumbnailState {
    /// Thumbnail is being generated.
    Loading,
    /// Thumbnail is ready (contains raw RGBA pixel data and dimensions).
    Ready { width: u32, height: u32, data: Vec<u8> },
    /// Thumbnail generation failed.
    Failed(String),
}

/// LRU-evicting cache for generated asset thumbnails.
#[derive(Debug)]
pub struct AssetThumbnailCache {
    /// Cached thumbnails ordered by access time (most recent at back).
    entries: VecDeque<ThumbnailCacheEntry>,
    /// Maximum number of thumbnails to retain.
    max_entries: usize,
}

#[derive(Debug)]
struct ThumbnailCacheEntry {
    path: PathBuf,
    state: ThumbnailState,
    /// Monotonic counter updated on each access for LRU ordering.
    last_access: u64,
}

impl AssetThumbnailCache {
    /// Create a new cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
        }
    }

    /// Look up a thumbnail by path. Returns `None` if not cached.
    pub fn get(&mut self, path: &Path, counter: u64) -> Option<&ThumbnailState> {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.path == path) {
            entry.last_access = counter;
            Some(&entry.state)
        } else {
            None
        }
    }

    /// Insert or update a thumbnail in the cache.
    pub fn insert(&mut self, path: PathBuf, state: ThumbnailState, counter: u64) {
        // Update existing entry.
        if let Some(entry) = self.entries.iter_mut().find(|e| e.path == path) {
            entry.state = state;
            entry.last_access = counter;
            return;
        }

        // Evict LRU if at capacity.
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        self.entries.push_back(ThumbnailCacheEntry {
            path,
            state,
            last_access: counter,
        });
    }

    /// Evict the least-recently-used entry.
    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        let mut min_access = u64::MAX;
        let mut min_idx = 0;

        for (i, entry) in self.entries.iter().enumerate() {
            if entry.last_access < min_access {
                min_access = entry.last_access;
                min_idx = i;
            }
        }

        self.entries.remove(min_idx);
    }

    /// Remove a specific path from the cache.
    pub fn invalidate(&mut self, path: &Path) {
        self.entries.retain(|e| e.path != path);
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Thumbnail Generator
// ---------------------------------------------------------------------------

/// Generates preview thumbnails for assets displayed in the browser.
///
/// Textures get downscaled previews, meshes get rendered previews,
/// audio files get waveform visualizations, etc.
#[derive(Debug)]
pub struct AssetThumbnailGenerator {
    /// Thumbnail size in pixels (square).
    pub thumbnail_size: u32,
    /// Maximum number of thumbnails to generate per frame.
    pub batch_size: usize,
    /// Queue of pending generation requests.
    pending: VecDeque<ThumbnailRequest>,
}

#[derive(Debug, Clone)]
struct ThumbnailRequest {
    path: PathBuf,
    kind: AssetKind,
    priority: u32,
}

impl Default for AssetThumbnailGenerator {
    fn default() -> Self {
        Self {
            thumbnail_size: 128,
            batch_size: 4,
            pending: VecDeque::new(),
        }
    }
}

impl AssetThumbnailGenerator {
    /// Create a new thumbnail generator with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request thumbnail generation for an asset.
    pub fn request_thumbnail(&mut self, path: PathBuf, kind: AssetKind) {
        // Avoid duplicate requests.
        if self.pending.iter().any(|r| r.path == path) {
            return;
        }

        let priority = match kind {
            AssetKind::Texture => 0, // Highest priority (fast to generate).
            AssetKind::Material => 1,
            AssetKind::Mesh => 2,
            AssetKind::Prefab => 3,
            _ => 10,
        };

        self.pending.push_back(ThumbnailRequest {
            path,
            kind,
            priority,
        });
    }

    /// Process pending thumbnail generation requests (call each frame).
    /// Returns paths of completed thumbnails and their state.
    pub fn process_pending(&mut self, cache: &mut AssetThumbnailCache, counter: u64) -> Vec<PathBuf> {
        // Sort by priority (stable sort to preserve insertion order for equal priority).
        let mut pending: Vec<_> = self.pending.drain(..).collect();
        pending.sort_by_key(|r| r.priority);
        let to_process: Vec<_> = pending.drain(..pending.len().min(self.batch_size)).collect();
        // Put the rest back.
        for remaining in pending {
            self.pending.push_back(remaining);
        }

        let mut completed = Vec::new();

        for request in to_process {
            let state = self.generate_thumbnail(&request);
            cache.insert(request.path.clone(), state, counter);
            completed.push(request.path);
        }

        completed
    }

    /// Generate a placeholder thumbnail for an asset. In a real engine this
    /// would render the mesh, downscale the texture, etc.
    fn generate_thumbnail(&self, request: &ThumbnailRequest) -> ThumbnailState {
        let size = self.thumbnail_size;
        let pixel_count = (size * size * 4) as usize;
        let mut data = vec![0u8; pixel_count];

        // Generate a simple color-coded placeholder based on asset kind.
        let color: [u8; 4] = match request.kind {
            AssetKind::Texture => [100, 150, 200, 255],
            AssetKind::Mesh => [200, 150, 100, 255],
            AssetKind::Audio => [100, 200, 100, 255],
            AssetKind::Material => [200, 100, 200, 255],
            AssetKind::Shader => [200, 200, 100, 255],
            AssetKind::Script => [150, 150, 150, 255],
            AssetKind::Prefab => [100, 200, 200, 255],
            AssetKind::Scene => [200, 200, 200, 255],
            _ => [128, 128, 128, 255],
        };

        for pixel in data.chunks_exact_mut(4) {
            pixel[0] = color[0];
            pixel[1] = color[1];
            pixel[2] = color[2];
            pixel[3] = color[3];
        }

        ThumbnailState::Ready {
            width: size,
            height: size,
            data,
        }
    }

    /// Number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Whether there are no pending requests.
    pub fn is_idle(&self) -> bool {
        self.pending.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Asset Browser
// ---------------------------------------------------------------------------

/// Sort mode for asset entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetSortMode {
    NameAsc,
    NameDesc,
    TypeAsc,
    TypeDesc,
    SizeAsc,
    SizeDesc,
    DateAsc,
    DateDesc,
}

/// The asset browser panel showing project assets organized in a directory tree
/// with thumbnail previews.
#[derive(Debug)]
pub struct AssetBrowser {
    /// Root directory of the project's assets.
    pub root_path: PathBuf,
    /// Currently navigated directory path (relative to root).
    pub current_path: PathBuf,
    /// Cached directory tree structure (left panel).
    pub directory_tree: Vec<DirectoryEntry>,
    /// Flat list of entries in the current directory (right panel).
    pub entries: Vec<AssetEntry>,
    /// Thumbnail cache.
    pub thumbnail_cache: AssetThumbnailCache,
    /// Current search/filter query.
    pub search_query: String,
    /// Asset type filter (empty = show all).
    pub type_filter: Vec<AssetKind>,
    /// Whether the browser panel is visible.
    pub visible: bool,
    /// Thumbnail generator.
    pub thumbnail_generator: AssetThumbnailGenerator,
    /// Navigation history for back/forward.
    history: Vec<PathBuf>,
    /// Current position in the history stack.
    history_index: usize,
    /// Sort mode.
    pub sort_mode: AssetSortMode,
    /// Access counter for LRU cache.
    access_counter: u64,
    /// Selected entries (by path).
    pub selected_entries: Vec<PathBuf>,
}

impl AssetBrowser {
    /// Create a new asset browser rooted at the given directory.
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            current_path: PathBuf::new(),
            directory_tree: Vec::new(),
            entries: Vec::new(),
            thumbnail_cache: AssetThumbnailCache::new(512),
            search_query: String::new(),
            type_filter: Vec::new(),
            visible: true,
            thumbnail_generator: AssetThumbnailGenerator::new(),
            history: vec![PathBuf::new()],
            history_index: 0,
            sort_mode: AssetSortMode::NameAsc,
            access_counter: 0,
            selected_entries: Vec::new(),
            root_path,
        }
    }

    /// Refresh the directory tree and current directory entries from disk.
    pub fn refresh(&mut self) {
        let abs_root = self.root_path.clone();

        // Build directory tree.
        self.directory_tree.clear();
        if abs_root.exists() && abs_root.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&abs_root) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let name = path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();
                        // Skip hidden directories.
                        if name.starts_with('.') {
                            continue;
                        }
                        let rel_path = path
                            .strip_prefix(&abs_root)
                            .unwrap_or(&path)
                            .to_path_buf();
                        let mut dir_entry = DirectoryEntry::new_directory(name, rel_path);
                        self.scan_directory_tree(&path, &abs_root, &mut dir_entry, 0, 5);
                        self.directory_tree.push(dir_entry);
                    }
                }
            }
        }

        self.directory_tree.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

        // Refresh current directory entries.
        self.refresh_current_directory();
    }

    /// Recursively scan subdirectories for the tree view, up to max_depth.
    fn scan_directory_tree(
        &self,
        abs_path: &Path,
        root: &Path,
        parent: &mut DirectoryEntry,
        depth: usize,
        max_depth: usize,
    ) {
        if depth >= max_depth {
            return;
        }

        if let Ok(entries) = std::fs::read_dir(abs_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if name.starts_with('.') {
                        continue;
                    }
                    let rel_path = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
                    let mut dir_entry = DirectoryEntry::new_directory(name, rel_path);
                    self.scan_directory_tree(&path, root, &mut dir_entry, depth + 1, max_depth);
                    parent.children.push(dir_entry);
                }
            }
            parent
                .children
                .sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        }
    }

    /// Refresh the flat entry list for the current directory.
    fn refresh_current_directory(&mut self) {
        self.entries.clear();
        let abs_dir = self.root_path.join(&self.current_path);

        if !abs_dir.exists() || !abs_dir.is_dir() {
            return;
        }

        if let Ok(read_dir) = std::fs::read_dir(&abs_dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();

                // Skip hidden files/dirs.
                if name.starts_with('.') {
                    continue;
                }

                let rel_path = path
                    .strip_prefix(&self.root_path)
                    .unwrap_or(&path)
                    .to_path_buf();

                if path.is_dir() {
                    self.entries.push(AssetEntry::new_folder(rel_path));
                } else {
                    let metadata = std::fs::metadata(&path);
                    let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
                    let modified = metadata
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    self.entries.push(AssetEntry::new_file(rel_path, size, modified));
                }
            }
        }

        self.sort_entries();
    }

    /// Sort the current entry list according to the sort mode.
    fn sort_entries(&mut self) {
        // Always show folders first.
        self.entries.sort_by(|a, b| {
            match (a.is_directory, b.is_directory) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => match self.sort_mode {
                    AssetSortMode::NameAsc => {
                        a.name.to_lowercase().cmp(&b.name.to_lowercase())
                    }
                    AssetSortMode::NameDesc => {
                        b.name.to_lowercase().cmp(&a.name.to_lowercase())
                    }
                    AssetSortMode::TypeAsc => a.extension.cmp(&b.extension),
                    AssetSortMode::TypeDesc => b.extension.cmp(&a.extension),
                    AssetSortMode::SizeAsc => a.size_bytes.cmp(&b.size_bytes),
                    AssetSortMode::SizeDesc => b.size_bytes.cmp(&a.size_bytes),
                    AssetSortMode::DateAsc => a.last_modified.cmp(&b.last_modified),
                    AssetSortMode::DateDesc => b.last_modified.cmp(&a.last_modified),
                },
            }
        });
    }

    /// Navigate to a subdirectory.
    pub fn navigate_to(&mut self, path: &Path) {
        self.current_path = path.to_path_buf();
        self.selected_entries.clear();

        // Trim forward history.
        self.history.truncate(self.history_index + 1);
        self.history.push(self.current_path.clone());
        self.history_index = self.history.len() - 1;

        self.refresh_current_directory();
    }

    /// Navigate up one directory level.
    pub fn navigate_up(&mut self) {
        if let Some(parent) = self.current_path.parent() {
            let parent = parent.to_path_buf();
            self.navigate_to(&parent.clone());
        }
    }

    /// Navigate back in history.
    pub fn navigate_back(&mut self) {
        if self.history_index > 0 {
            self.history_index -= 1;
            self.current_path = self.history[self.history_index].clone();
            self.selected_entries.clear();
            self.refresh_current_directory();
        }
    }

    /// Navigate forward in history.
    pub fn navigate_forward(&mut self) {
        if self.history_index + 1 < self.history.len() {
            self.history_index += 1;
            self.current_path = self.history[self.history_index].clone();
            self.selected_entries.clear();
            self.refresh_current_directory();
        }
    }

    /// Whether back navigation is possible.
    pub fn can_go_back(&self) -> bool {
        self.history_index > 0
    }

    /// Whether forward navigation is possible.
    pub fn can_go_forward(&self) -> bool {
        self.history_index + 1 < self.history.len()
    }

    /// Return the current breadcrumb path segments.
    pub fn breadcrumbs(&self) -> Vec<(String, PathBuf)> {
        let mut crumbs = vec![("Assets".to_string(), PathBuf::new())];
        let mut accumulated = PathBuf::new();
        for component in self.current_path.components() {
            accumulated.push(component);
            let name = component.as_os_str().to_string_lossy().to_string();
            crumbs.push((name, accumulated.clone()));
        }
        crumbs
    }

    /// Fuzzy search for assets matching the query across all directories.
    pub fn search(&self, query: &str) -> Vec<AssetEntry> {
        if query.is_empty() {
            return Vec::new();
        }

        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        self.search_recursive(&self.root_path, &query_lower, &mut results, 0, 10);

        // Sort by match quality (name starts with query first, then contains).
        results.sort_by(|a, b| {
            let a_starts = a.name.to_lowercase().starts_with(&query_lower);
            let b_starts = b.name.to_lowercase().starts_with(&query_lower);
            match (a_starts, b_starts) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
            }
        });

        results
    }

    /// Recursively search directories for matching file names.
    fn search_recursive(
        &self,
        dir: &Path,
        query: &str,
        results: &mut Vec<AssetEntry>,
        depth: usize,
        max_depth: usize,
    ) {
        if depth >= max_depth || results.len() >= 200 {
            return;
        }

        let read_dir = match std::fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(_) => return,
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            if name.starts_with('.') {
                continue;
            }

            if path.is_dir() {
                self.search_recursive(&path, query, results, depth + 1, max_depth);
            } else if fuzzy_match(&name.to_lowercase(), query) {
                let rel_path = path
                    .strip_prefix(&self.root_path)
                    .unwrap_or(&path)
                    .to_path_buf();
                let metadata = std::fs::metadata(&path);
                let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
                let modified = metadata
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                results.push(AssetEntry::new_file(rel_path, size, modified));
            }
        }
    }

    /// Return the filtered entries (applying type filter and search query).
    pub fn filtered_entries(&self) -> Vec<&AssetEntry> {
        let query_lower = self.search_query.to_lowercase();

        self.entries
            .iter()
            .filter(|e| {
                // Type filter.
                if !self.type_filter.is_empty() && !e.is_directory {
                    if !self.type_filter.contains(&e.kind) {
                        return false;
                    }
                }

                // Search filter.
                if !query_lower.is_empty() {
                    return fuzzy_match(&e.name.to_lowercase(), &query_lower);
                }

                true
            })
            .collect()
    }

    /// Select an entry by path.
    pub fn select_entry(&mut self, path: &Path, additive: bool) {
        if !additive {
            self.selected_entries.clear();
        }
        let p = path.to_path_buf();
        if !self.selected_entries.contains(&p) {
            self.selected_entries.push(p);
        }
    }

    /// Handle an asset being double-clicked (navigate into folder or open).
    pub fn on_double_click(&mut self, entry: &AssetEntry) -> AssetBrowserAction {
        if entry.is_directory {
            self.navigate_to(&entry.path.clone());
            AssetBrowserAction::NavigatedToDirectory
        } else {
            AssetBrowserAction::OpenAsset {
                path: entry.path.clone(),
                kind: entry.kind,
            }
        }
    }

    /// Create a drag-drop payload for the given entry.
    pub fn create_drag_payload(&self, entry: &AssetEntry) -> Option<DragDropPayload> {
        if entry.is_directory {
            return None;
        }
        Some(DragDropPayload::new(
            entry.asset_id.unwrap_or_else(Uuid::new_v4),
            entry.name.clone(),
            entry.kind.display_name().to_string(),
            entry.path.clone(),
        ))
    }

    /// Per-frame update (processes thumbnail generation).
    pub fn update(&mut self) {
        self.access_counter += 1;
        self.thumbnail_generator
            .process_pending(&mut self.thumbnail_cache, self.access_counter);
    }

    /// Request thumbnails for all currently visible entries.
    pub fn request_visible_thumbnails(&mut self) {
        for entry in &self.entries {
            if !entry.is_directory && entry.kind != AssetKind::Unknown {
                if self.thumbnail_cache.get(&entry.path, self.access_counter).is_none() {
                    self.thumbnail_generator
                        .request_thumbnail(entry.path.clone(), entry.kind);
                }
            }
        }
    }
}

/// Action returned from asset browser interactions.
#[derive(Debug, Clone)]
pub enum AssetBrowserAction {
    /// Navigated into a directory.
    NavigatedToDirectory,
    /// Request to open an asset in its editor.
    OpenAsset { path: PathBuf, kind: AssetKind },
}

// ---------------------------------------------------------------------------
// Asset Import Dialog
// ---------------------------------------------------------------------------

/// Configuration dialog presented when importing new assets into the project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetImportDialog {
    /// Source file path on disk.
    pub source_path: PathBuf,
    /// Target directory within the asset root.
    pub target_directory: PathBuf,
    /// Whether to import with default settings.
    pub use_defaults: bool,
    /// Import-specific settings (format-dependent).
    pub settings: ImportSettings,
}

/// Format-specific import settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportSettings {
    /// Settings for importing textures.
    Texture {
        generate_mipmaps: bool,
        srgb: bool,
        max_resolution: Option<u32>,
        compression: TextureCompression,
    },
    /// Settings for importing 3D models.
    Model {
        import_animations: bool,
        import_materials: bool,
        scale_factor: f32,
        recalculate_normals: bool,
    },
    /// Settings for importing audio files.
    Audio {
        sample_rate: u32,
        channels: u8,
        streaming: bool,
    },
    /// Generic / pass-through import.
    Raw,
}

/// Texture compression modes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TextureCompression {
    None,
    Bc1,
    Bc3,
    Bc5,
    Bc7,
    Astc4x4,
    Astc6x6,
    Astc8x8,
}

impl AssetImportDialog {
    /// Create a new import dialog for the given source file.
    pub fn new(source_path: PathBuf, target_directory: PathBuf) -> Self {
        let settings = Self::detect_import_settings(&source_path);
        Self {
            source_path,
            target_directory,
            use_defaults: true,
            settings,
        }
    }

    /// Auto-detect import settings from the source file extension.
    fn detect_import_settings(path: &Path) -> ImportSettings {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        match AssetKind::from_extension(&ext) {
            AssetKind::Texture => ImportSettings::Texture {
                generate_mipmaps: true,
                srgb: true,
                max_resolution: None,
                compression: TextureCompression::Bc7,
            },
            AssetKind::Mesh => ImportSettings::Model {
                import_animations: true,
                import_materials: true,
                scale_factor: 1.0,
                recalculate_normals: false,
            },
            AssetKind::Audio => ImportSettings::Audio {
                sample_rate: 44100,
                channels: 2,
                streaming: false,
            },
            _ => ImportSettings::Raw,
        }
    }

    /// Execute the import with current settings.
    pub fn execute_import(&self) -> Result<Uuid, AssetImportError> {
        // Validate source file exists.
        if !self.source_path.exists() {
            return Err(AssetImportError::FileNotFound(self.source_path.clone()));
        }

        // Validate target directory exists.
        if !self.target_directory.exists() {
            std::fs::create_dir_all(&self.target_directory)
                .map_err(AssetImportError::Io)?;
        }

        // Determine destination filename.
        let filename = self
            .source_path
            .file_name()
            .ok_or_else(|| AssetImportError::ImportFailed("No filename".into()))?;
        let dest = self.target_directory.join(filename);

        // Copy the file.
        std::fs::copy(&self.source_path, &dest).map_err(AssetImportError::Io)?;

        // Generate an asset UUID.
        let asset_id = Uuid::new_v4();
        log::info!(
            "Imported {:?} -> {:?} as {:?}",
            self.source_path,
            dest,
            asset_id,
        );

        Ok(asset_id)
    }
}

/// Errors during asset import.
#[derive(Debug, thiserror::Error)]
pub enum AssetImportError {
    #[error("Source file not found: {0}")]
    FileNotFound(PathBuf),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Import failed: {0}")]
    ImportFailed(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Drag & Drop
// ---------------------------------------------------------------------------

/// Payload carried during a drag-and-drop operation from the asset browser
/// to the viewport or hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragDropPayload {
    /// The asset UUID being dragged.
    pub asset_id: Uuid,
    /// Human-readable asset name.
    pub asset_name: String,
    /// Asset type identifier (e.g. "Mesh", "Texture", "Material").
    pub asset_type: String,
    /// File path relative to asset root.
    pub asset_path: PathBuf,
}

impl DragDropPayload {
    /// Create a new drag-drop payload.
    pub fn new(
        asset_id: Uuid,
        asset_name: String,
        asset_type: String,
        asset_path: PathBuf,
    ) -> Self {
        Self {
            asset_id,
            asset_name,
            asset_type,
            asset_path,
        }
    }
}

// ---------------------------------------------------------------------------
// Fuzzy match helper
// ---------------------------------------------------------------------------

/// Simple fuzzy match: checks if all characters of `query` appear in `text`
/// in order (case-insensitive comparison assumed by caller).
fn fuzzy_match(text: &str, query: &str) -> bool {
    if query.is_empty() {
        return true;
    }

    // First try substring match (higher quality).
    if text.contains(query) {
        return true;
    }

    // Fall back to character-order match.
    let mut query_chars = query.chars();
    let mut current = query_chars.next();

    for ch in text.chars() {
        if let Some(q) = current {
            if ch == q {
                current = query_chars.next();
            }
        } else {
            break;
        }
    }

    current.is_none()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_kind_from_extension() {
        assert_eq!(AssetKind::from_extension("png"), AssetKind::Texture);
        assert_eq!(AssetKind::from_extension("PNG"), AssetKind::Texture);
        assert_eq!(AssetKind::from_extension("fbx"), AssetKind::Mesh);
        assert_eq!(AssetKind::from_extension("wav"), AssetKind::Audio);
        assert_eq!(AssetKind::from_extension("lua"), AssetKind::Script);
        assert_eq!(AssetKind::from_extension("mat"), AssetKind::Material);
        assert_eq!(AssetKind::from_extension("prefab"), AssetKind::Prefab);
        assert_eq!(AssetKind::from_extension("scene"), AssetKind::Scene);
        assert_eq!(AssetKind::from_extension("wgsl"), AssetKind::Shader);
        assert_eq!(AssetKind::from_extension("ttf"), AssetKind::Font);
        assert_eq!(AssetKind::from_extension("xyz"), AssetKind::Unknown);
    }

    #[test]
    fn asset_entry_file() {
        let entry = AssetEntry::new_file(
            PathBuf::from("textures/brick.png"),
            1024 * 500,
            1700000000,
        );
        assert_eq!(entry.name, "brick.png");
        assert_eq!(entry.extension, "png");
        assert_eq!(entry.kind, AssetKind::Texture);
        assert!(!entry.is_directory);
        assert_eq!(entry.display_size(), "500.0 KB");
    }

    #[test]
    fn asset_entry_folder() {
        let entry = AssetEntry::new_folder(PathBuf::from("textures"));
        assert_eq!(entry.name, "textures");
        assert_eq!(entry.kind, AssetKind::Folder);
        assert!(entry.is_directory);
        assert!(entry.display_size().is_empty());
    }

    #[test]
    fn asset_entry_display_sizes() {
        let small = AssetEntry::new_file(PathBuf::from("a.txt"), 500, 0);
        assert_eq!(small.display_size(), "500 B");

        let medium = AssetEntry::new_file(PathBuf::from("b.txt"), 1024 * 1024 + 500000, 0);
        assert!(medium.display_size().contains("MB"));

        let large = AssetEntry::new_file(
            PathBuf::from("c.txt"),
            2 * 1024 * 1024 * 1024,
            0,
        );
        assert!(large.display_size().contains("GB"));
    }

    #[test]
    fn directory_entry_tree() {
        let mut root = DirectoryEntry::new_directory("assets", PathBuf::from("assets"));
        let child = DirectoryEntry::new_directory("textures", PathBuf::from("assets/textures"));
        root.children.push(child);
        assert_eq!(root.count(), 2);
        assert!(root.find(Path::new("assets/textures")).is_some());
        assert!(root.find(Path::new("assets/models")).is_none());
    }

    #[test]
    fn thumbnail_cache_lru() {
        let mut cache = AssetThumbnailCache::new(3);

        cache.insert(PathBuf::from("a"), ThumbnailState::Loading, 1);
        cache.insert(PathBuf::from("b"), ThumbnailState::Loading, 2);
        cache.insert(PathBuf::from("c"), ThumbnailState::Loading, 3);
        assert_eq!(cache.len(), 3);

        // Access "a" to make it more recent.
        cache.get(Path::new("a"), 4);

        // Insert "d" should evict "b" (least recently used).
        cache.insert(PathBuf::from("d"), ThumbnailState::Loading, 5);
        assert_eq!(cache.len(), 3);
        assert!(cache.get(Path::new("b"), 6).is_none());
        assert!(cache.get(Path::new("a"), 6).is_some());
    }

    #[test]
    fn thumbnail_cache_invalidate() {
        let mut cache = AssetThumbnailCache::new(10);
        cache.insert(PathBuf::from("x"), ThumbnailState::Loading, 1);
        cache.invalidate(Path::new("x"));
        assert!(cache.is_empty());
    }

    #[test]
    fn thumbnail_generator_deduplicates() {
        let mut thumb_gen = AssetThumbnailGenerator::new();
        thumb_gen.request_thumbnail(PathBuf::from("a.png"), AssetKind::Texture);
        thumb_gen.request_thumbnail(PathBuf::from("a.png"), AssetKind::Texture);
        assert_eq!(thumb_gen.pending_count(), 1);
    }

    #[test]
    fn thumbnail_generator_processes_batch() {
        let mut thumb_gen = AssetThumbnailGenerator::new();
        thumb_gen.batch_size = 2;
        thumb_gen.request_thumbnail(PathBuf::from("a.png"), AssetKind::Texture);
        thumb_gen.request_thumbnail(PathBuf::from("b.fbx"), AssetKind::Mesh);
        thumb_gen.request_thumbnail(PathBuf::from("c.wav"), AssetKind::Audio);

        let mut cache = AssetThumbnailCache::new(10);
        let completed = thumb_gen.process_pending(&mut cache, 1);
        assert_eq!(completed.len(), 2);
        assert_eq!(thumb_gen.pending_count(), 1);

        // Second batch completes the rest.
        let completed2 = thumb_gen.process_pending(&mut cache, 2);
        assert_eq!(completed2.len(), 1);
        assert!(thumb_gen.is_idle());
    }

    #[test]
    fn fuzzy_match_basic() {
        assert!(fuzzy_match("player_sprite.png", "play"));
        assert!(fuzzy_match("player_sprite.png", "sprite"));
        assert!(fuzzy_match("player_sprite.png", "plr")); // character-order match
        assert!(!fuzzy_match("player_sprite.png", "xyz"));
    }

    #[test]
    fn asset_browser_creation() {
        let browser = AssetBrowser::new(PathBuf::from("/project/assets"));
        assert!(browser.visible);
        assert!(browser.entries.is_empty());
        assert_eq!(browser.current_path, PathBuf::new());
    }

    #[test]
    fn asset_browser_breadcrumbs() {
        let mut browser = AssetBrowser::new(PathBuf::from("/project/assets"));
        browser.current_path = PathBuf::from("textures/environment");
        let crumbs = browser.breadcrumbs();
        assert_eq!(crumbs.len(), 3);
        assert_eq!(crumbs[0].0, "Assets");
        assert_eq!(crumbs[1].0, "textures");
        assert_eq!(crumbs[2].0, "environment");
    }

    #[test]
    fn asset_browser_navigation_history() {
        let mut browser = AssetBrowser::new(PathBuf::from("/nonexistent"));
        assert!(!browser.can_go_back());
        assert!(!browser.can_go_forward());

        browser.navigate_to(Path::new("a"));
        browser.navigate_to(Path::new("b"));
        assert!(browser.can_go_back());
        assert!(!browser.can_go_forward());

        browser.navigate_back();
        assert_eq!(browser.current_path, PathBuf::from("a"));
        assert!(browser.can_go_forward());

        browser.navigate_forward();
        assert_eq!(browser.current_path, PathBuf::from("b"));
    }

    #[test]
    fn asset_browser_type_filter() {
        let mut browser = AssetBrowser::new(PathBuf::from("/nonexistent"));
        // Manually add entries for testing.
        browser.entries.push(AssetEntry::new_file(PathBuf::from("a.png"), 100, 0));
        browser.entries.push(AssetEntry::new_file(PathBuf::from("b.fbx"), 200, 0));
        browser.entries.push(AssetEntry::new_file(PathBuf::from("c.wav"), 300, 0));

        browser.type_filter = vec![AssetKind::Texture];
        let filtered = browser.filtered_entries();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].kind, AssetKind::Texture);
    }

    #[test]
    fn asset_browser_search_filter() {
        let mut browser = AssetBrowser::new(PathBuf::from("/nonexistent"));
        browser.entries.push(AssetEntry::new_file(PathBuf::from("player.png"), 100, 0));
        browser.entries.push(AssetEntry::new_file(PathBuf::from("enemy.png"), 200, 0));
        browser.entries.push(AssetEntry::new_file(PathBuf::from("background.png"), 300, 0));

        browser.search_query = "play".to_string();
        let filtered = browser.filtered_entries();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "player.png");
    }

    #[test]
    fn asset_browser_select_entry() {
        let mut browser = AssetBrowser::new(PathBuf::from("/nonexistent"));
        browser.select_entry(Path::new("a.png"), false);
        assert_eq!(browser.selected_entries.len(), 1);

        browser.select_entry(Path::new("b.png"), true);
        assert_eq!(browser.selected_entries.len(), 2);

        browser.select_entry(Path::new("c.png"), false);
        assert_eq!(browser.selected_entries.len(), 1);
    }

    #[test]
    fn drag_drop_payload() {
        let payload = DragDropPayload::new(
            Uuid::new_v4(),
            "brick.png".into(),
            "Texture".into(),
            PathBuf::from("textures/brick.png"),
        );
        assert_eq!(payload.asset_name, "brick.png");
        assert_eq!(payload.asset_type, "Texture");
    }

    #[test]
    fn import_dialog_auto_detect() {
        let dialog = AssetImportDialog::new(
            PathBuf::from("source/hero.png"),
            PathBuf::from("/project/assets/textures"),
        );
        match dialog.settings {
            ImportSettings::Texture { generate_mipmaps, srgb, .. } => {
                assert!(generate_mipmaps);
                assert!(srgb);
            }
            _ => panic!("Expected Texture import settings"),
        }
    }

    #[test]
    fn import_dialog_mesh_detect() {
        let dialog = AssetImportDialog::new(
            PathBuf::from("source/model.fbx"),
            PathBuf::from("/project/assets/models"),
        );
        match dialog.settings {
            ImportSettings::Model { import_animations, scale_factor, .. } => {
                assert!(import_animations);
                assert!((scale_factor - 1.0).abs() < 1e-5);
            }
            _ => panic!("Expected Model import settings"),
        }
    }

    #[test]
    fn asset_kind_display_name() {
        assert_eq!(AssetKind::Texture.display_name(), "Texture");
        assert_eq!(AssetKind::Mesh.display_name(), "Mesh");
        assert_eq!(AssetKind::Folder.display_name(), "Folder");
    }
}
