//! # Clipboard Access
//!
//! Provides cross-platform clipboard operations for the Genovo engine,
//! including text and image copy/paste, clipboard change notifications,
//! clipboard history, and platform-specific implementation stubs for
//! Win32, macOS (NSPasteboard), and X11 (selection).

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during clipboard operations.
#[derive(Debug, Clone)]
pub enum ClipboardError {
    /// The clipboard could not be opened (locked by another process).
    Unavailable,
    /// The requested data format is not on the clipboard.
    FormatNotAvailable(String),
    /// Failed to set the clipboard data.
    SetFailed(String),
    /// Failed to read clipboard data.
    ReadFailed(String),
    /// The platform does not support this operation.
    NotSupported(String),
    /// An internal platform error occurred.
    PlatformError(String),
    /// The clipboard data is too large.
    DataTooLarge { size: usize, max: usize },
}

impl fmt::Display for ClipboardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => write!(f, "clipboard unavailable"),
            Self::FormatNotAvailable(fmt_name) => write!(f, "format not available: {fmt_name}"),
            Self::SetFailed(msg) => write!(f, "set failed: {msg}"),
            Self::ReadFailed(msg) => write!(f, "read failed: {msg}"),
            Self::NotSupported(msg) => write!(f, "not supported: {msg}"),
            Self::PlatformError(msg) => write!(f, "platform error: {msg}"),
            Self::DataTooLarge { size, max } => {
                write!(f, "data too large: {size} bytes (max: {max})")
            }
        }
    }
}

impl std::error::Error for ClipboardError {}

/// Result type for clipboard operations.
pub type ClipboardResult<T> = Result<T, ClipboardError>;

// ---------------------------------------------------------------------------
// Data formats
// ---------------------------------------------------------------------------

/// Data formats that can be stored on the clipboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClipboardFormat {
    /// Plain UTF-8 text.
    Text,
    /// Rich text (HTML or RTF).
    RichText,
    /// Image data (RGBA pixels).
    Image,
    /// File paths (list of files).
    FilePaths,
    /// Raw binary data.
    RawData,
}

impl fmt::Display for ClipboardFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "text/plain"),
            Self::RichText => write!(f, "text/html"),
            Self::Image => write!(f, "image/rgba"),
            Self::FilePaths => write!(f, "text/uri-list"),
            Self::RawData => write!(f, "application/octet-stream"),
        }
    }
}

// ---------------------------------------------------------------------------
// Clipboard image
// ---------------------------------------------------------------------------

/// An image on the clipboard.
#[derive(Debug, Clone)]
pub struct ClipboardImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA pixel data (4 bytes per pixel, row-major).
    pub data: Vec<u8>,
}

impl ClipboardImage {
    /// Create a new clipboard image.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self { width, height, data }
    }

    /// Create a solid-colour image (for testing).
    pub fn solid(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Self {
        let pixel_count = (width * height) as usize;
        let mut data = Vec::with_capacity(pixel_count * 4);
        for _ in 0..pixel_count {
            data.push(r);
            data.push(g);
            data.push(b);
            data.push(a);
        }
        Self { width, height, data }
    }

    /// Total size in bytes.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Expected size based on dimensions.
    pub fn expected_size(&self) -> usize {
        (self.width * self.height * 4) as usize
    }

    /// Validate that the data length matches the dimensions.
    pub fn is_valid(&self) -> bool {
        self.data.len() == self.expected_size()
    }
}

// ---------------------------------------------------------------------------
// Clipboard content
// ---------------------------------------------------------------------------

/// The contents of the clipboard at a point in time.
#[derive(Debug, Clone)]
pub enum ClipboardContent {
    /// Text content.
    Text(String),
    /// Rich text (HTML).
    RichText(String),
    /// Image data.
    Image(ClipboardImage),
    /// List of file paths.
    FilePaths(Vec<String>),
    /// Raw binary data with a MIME type.
    RawData { mime: String, data: Vec<u8> },
    /// The clipboard is empty.
    Empty,
}

impl ClipboardContent {
    /// Returns the format of this content.
    pub fn format(&self) -> Option<ClipboardFormat> {
        match self {
            Self::Text(_) => Some(ClipboardFormat::Text),
            Self::RichText(_) => Some(ClipboardFormat::RichText),
            Self::Image(_) => Some(ClipboardFormat::Image),
            Self::FilePaths(_) => Some(ClipboardFormat::FilePaths),
            Self::RawData { .. } => Some(ClipboardFormat::RawData),
            Self::Empty => None,
        }
    }

    /// Get the text content, if any.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Get the image content, if any.
    pub fn as_image(&self) -> Option<&ClipboardImage> {
        match self {
            Self::Image(img) => Some(img),
            _ => None,
        }
    }

    /// Approximate size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Text(s) => s.len(),
            Self::RichText(s) => s.len(),
            Self::Image(img) => img.byte_size(),
            Self::FilePaths(paths) => paths.iter().map(|p| p.len()).sum(),
            Self::RawData { data, .. } => data.len(),
            Self::Empty => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Clipboard change notification
// ---------------------------------------------------------------------------

/// A clipboard change event.
#[derive(Debug, Clone)]
pub struct ClipboardChangeEvent {
    /// When the change was detected.
    pub timestamp: Instant,
    /// The format that was placed on the clipboard.
    pub format: Option<ClipboardFormat>,
    /// Size of the new clipboard content.
    pub content_size: usize,
    /// Which application placed the data (if detectable, else empty).
    pub source_app: String,
    /// Sequence number (monotonically increasing).
    pub sequence: u64,
}

/// Callback type for clipboard change notifications.
pub type ClipboardChangeCallback = Box<dyn Fn(&ClipboardChangeEvent) + Send + Sync>;

// ---------------------------------------------------------------------------
// Clipboard history
// ---------------------------------------------------------------------------

/// Tracks a history of clipboard entries.
pub struct ClipboardHistory {
    /// Historical entries (newest first).
    entries: VecDeque<ClipboardHistoryEntry>,
    /// Maximum number of entries to retain.
    max_entries: usize,
    /// Maximum total memory budget in bytes.
    max_bytes: usize,
    /// Current total size in bytes.
    current_bytes: usize,
}

/// A single entry in the clipboard history.
#[derive(Debug, Clone)]
pub struct ClipboardHistoryEntry {
    /// The clipboard content.
    pub content: ClipboardContent,
    /// When this entry was recorded.
    pub timestamp: SystemTime,
    /// Size in bytes.
    pub size: usize,
    /// Whether this entry has been pinned (immune to eviction).
    pub pinned: bool,
}

impl ClipboardHistory {
    /// Create a new clipboard history with the given capacity.
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            max_bytes,
            current_bytes: 0,
        }
    }

    /// Add a new entry to the history.
    pub fn push(&mut self, content: ClipboardContent) {
        let size = content.size_bytes();
        // Evict old entries if necessary
        while self.entries.len() >= self.max_entries
            || (self.current_bytes + size > self.max_bytes && !self.entries.is_empty())
        {
            // Remove the oldest non-pinned entry
            if let Some(pos) = self.entries.iter().rposition(|e| !e.pinned) {
                let removed = self.entries.remove(pos).unwrap();
                self.current_bytes = self.current_bytes.saturating_sub(removed.size);
            } else {
                break; // All entries are pinned
            }
        }
        let entry = ClipboardHistoryEntry {
            content,
            timestamp: SystemTime::now(),
            size,
            pinned: false,
        };
        self.current_bytes += size;
        self.entries.push_front(entry);
    }

    /// Get an entry by index (0 = most recent).
    pub fn get(&self, index: usize) -> Option<&ClipboardHistoryEntry> {
        self.entries.get(index)
    }

    /// Number of entries in the history.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_bytes = 0;
    }

    /// Pin an entry so it is not evicted.
    pub fn pin(&mut self, index: usize) {
        if let Some(entry) = self.entries.get_mut(index) {
            entry.pinned = true;
        }
    }

    /// Unpin an entry.
    pub fn unpin(&mut self, index: usize) {
        if let Some(entry) = self.entries.get_mut(index) {
            entry.pinned = false;
        }
    }

    /// Total memory used by history entries.
    pub fn total_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Search history entries containing the given text.
    pub fn search_text(&self, query: &str) -> Vec<usize> {
        let query_lower = query.to_lowercase();
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if let ClipboardContent::Text(ref s) = e.content {
                    if s.to_lowercase().contains(&query_lower) {
                        return Some(i);
                    }
                }
                None
            })
            .collect()
    }

    /// Iterator over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &ClipboardHistoryEntry> {
        self.entries.iter()
    }
}

impl Default for ClipboardHistory {
    fn default() -> Self {
        Self::new(100, 50 * 1024 * 1024) // 100 entries, 50 MB max
    }
}

// ---------------------------------------------------------------------------
// Clipboard trait
// ---------------------------------------------------------------------------

/// Platform-agnostic clipboard interface.
pub trait Clipboard: Send + Sync {
    /// Get the current text from the clipboard.
    fn get_text(&self) -> ClipboardResult<String>;

    /// Set the clipboard text.
    fn set_text(&self, text: &str) -> ClipboardResult<()>;

    /// Get image data from the clipboard.
    fn get_image(&self) -> ClipboardResult<ClipboardImage>;

    /// Set image data on the clipboard.
    fn set_image(&self, image: &ClipboardImage) -> ClipboardResult<()>;

    /// Check if a specific format is available.
    fn has_format(&self, format: ClipboardFormat) -> bool;

    /// Get all available formats.
    fn available_formats(&self) -> Vec<ClipboardFormat>;

    /// Clear the clipboard.
    fn clear(&self) -> ClipboardResult<()>;
}

// ---------------------------------------------------------------------------
// Platform stubs
// ---------------------------------------------------------------------------

/// Win32 clipboard implementation stub.
///
/// In production, this would call `OpenClipboard`, `GetClipboardData`,
/// `SetClipboardData`, and `CloseClipboard` via the Win32 API.
pub struct Win32Clipboard {
    /// In-memory store for testing.
    store: Arc<Mutex<ClipboardStore>>,
    /// Change sequence counter.
    sequence: Arc<Mutex<u64>>,
}

/// Internal store for the software clipboard implementation.
#[derive(Debug, Clone)]
struct ClipboardStore {
    text: Option<String>,
    image: Option<ClipboardImage>,
    formats: Vec<ClipboardFormat>,
}

impl Default for ClipboardStore {
    fn default() -> Self {
        Self {
            text: None,
            image: None,
            formats: Vec::new(),
        }
    }
}

impl Win32Clipboard {
    /// Create a new Win32 clipboard instance.
    ///
    /// In production, this would call `OpenClipboard(NULL)`.
    pub fn new() -> Self {
        Self {
            store: Arc::new(Mutex::new(ClipboardStore::default())),
            sequence: Arc::new(Mutex::new(0)),
        }
    }

    fn bump_sequence(&self) {
        if let Ok(mut seq) = self.sequence.lock() {
            *seq += 1;
        }
    }
}

impl Default for Win32Clipboard {
    fn default() -> Self {
        Self::new()
    }
}

impl Clipboard for Win32Clipboard {
    fn get_text(&self) -> ClipboardResult<String> {
        let store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store
            .text
            .clone()
            .ok_or(ClipboardError::FormatNotAvailable("text".into()))
    }

    fn set_text(&self, text: &str) -> ClipboardResult<()> {
        let mut store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store.text = Some(text.to_string());
        if !store.formats.contains(&ClipboardFormat::Text) {
            store.formats.push(ClipboardFormat::Text);
        }
        self.bump_sequence();
        Ok(())
    }

    fn get_image(&self) -> ClipboardResult<ClipboardImage> {
        let store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store
            .image
            .clone()
            .ok_or(ClipboardError::FormatNotAvailable("image".into()))
    }

    fn set_image(&self, image: &ClipboardImage) -> ClipboardResult<()> {
        let mut store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store.image = Some(image.clone());
        if !store.formats.contains(&ClipboardFormat::Image) {
            store.formats.push(ClipboardFormat::Image);
        }
        self.bump_sequence();
        Ok(())
    }

    fn has_format(&self, format: ClipboardFormat) -> bool {
        let store = self.store.lock().unwrap_or_else(|e| e.into_inner());
        store.formats.contains(&format)
    }

    fn available_formats(&self) -> Vec<ClipboardFormat> {
        let store = self.store.lock().unwrap_or_else(|e| e.into_inner());
        store.formats.clone()
    }

    fn clear(&self) -> ClipboardResult<()> {
        let mut store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store.text = None;
        store.image = None;
        store.formats.clear();
        self.bump_sequence();
        Ok(())
    }
}

/// macOS NSPasteboard clipboard implementation stub.
pub struct MacOsClipboard {
    store: Arc<Mutex<ClipboardStore>>,
}

impl MacOsClipboard {
    pub fn new() -> Self {
        Self {
            store: Arc::new(Mutex::new(ClipboardStore::default())),
        }
    }
}

impl Default for MacOsClipboard {
    fn default() -> Self {
        Self::new()
    }
}

impl Clipboard for MacOsClipboard {
    fn get_text(&self) -> ClipboardResult<String> {
        let store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store
            .text
            .clone()
            .ok_or(ClipboardError::FormatNotAvailable("text".into()))
    }

    fn set_text(&self, text: &str) -> ClipboardResult<()> {
        let mut store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store.text = Some(text.to_string());
        if !store.formats.contains(&ClipboardFormat::Text) {
            store.formats.push(ClipboardFormat::Text);
        }
        Ok(())
    }

    fn get_image(&self) -> ClipboardResult<ClipboardImage> {
        let store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store
            .image
            .clone()
            .ok_or(ClipboardError::FormatNotAvailable("image".into()))
    }

    fn set_image(&self, image: &ClipboardImage) -> ClipboardResult<()> {
        let mut store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store.image = Some(image.clone());
        if !store.formats.contains(&ClipboardFormat::Image) {
            store.formats.push(ClipboardFormat::Image);
        }
        Ok(())
    }

    fn has_format(&self, format: ClipboardFormat) -> bool {
        let store = self.store.lock().unwrap_or_else(|e| e.into_inner());
        store.formats.contains(&format)
    }

    fn available_formats(&self) -> Vec<ClipboardFormat> {
        let store = self.store.lock().unwrap_or_else(|e| e.into_inner());
        store.formats.clone()
    }

    fn clear(&self) -> ClipboardResult<()> {
        let mut store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store.text = None;
        store.image = None;
        store.formats.clear();
        Ok(())
    }
}

/// X11 selection clipboard implementation stub.
pub struct X11Clipboard {
    store: Arc<Mutex<ClipboardStore>>,
}

impl X11Clipboard {
    pub fn new() -> Self {
        Self {
            store: Arc::new(Mutex::new(ClipboardStore::default())),
        }
    }
}

impl Default for X11Clipboard {
    fn default() -> Self {
        Self::new()
    }
}

impl Clipboard for X11Clipboard {
    fn get_text(&self) -> ClipboardResult<String> {
        let store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store
            .text
            .clone()
            .ok_or(ClipboardError::FormatNotAvailable("text".into()))
    }

    fn set_text(&self, text: &str) -> ClipboardResult<()> {
        let mut store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store.text = Some(text.to_string());
        if !store.formats.contains(&ClipboardFormat::Text) {
            store.formats.push(ClipboardFormat::Text);
        }
        Ok(())
    }

    fn get_image(&self) -> ClipboardResult<ClipboardImage> {
        let store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store
            .image
            .clone()
            .ok_or(ClipboardError::FormatNotAvailable("image".into()))
    }

    fn set_image(&self, image: &ClipboardImage) -> ClipboardResult<()> {
        let mut store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store.image = Some(image.clone());
        if !store.formats.contains(&ClipboardFormat::Image) {
            store.formats.push(ClipboardFormat::Image);
        }
        Ok(())
    }

    fn has_format(&self, format: ClipboardFormat) -> bool {
        let store = self.store.lock().unwrap_or_else(|e| e.into_inner());
        store.formats.contains(&format)
    }

    fn available_formats(&self) -> Vec<ClipboardFormat> {
        let store = self.store.lock().unwrap_or_else(|e| e.into_inner());
        store.formats.clone()
    }

    fn clear(&self) -> ClipboardResult<()> {
        let mut store = self.store.lock().map_err(|_| ClipboardError::Unavailable)?;
        store.text = None;
        store.image = None;
        store.formats.clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create the appropriate clipboard implementation for the current platform.
pub fn create_clipboard() -> Box<dyn Clipboard> {
    #[cfg(target_os = "windows")]
    {
        Box::new(Win32Clipboard::new())
    }
    #[cfg(target_os = "macos")]
    {
        Box::new(MacOsClipboard::new())
    }
    #[cfg(target_os = "linux")]
    {
        Box::new(X11Clipboard::new())
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        Box::new(Win32Clipboard::new()) // fallback
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clipboard() -> Box<dyn Clipboard> {
        Box::new(Win32Clipboard::new())
    }

    #[test]
    fn test_set_and_get_text() {
        let cb = make_clipboard();
        cb.set_text("hello world").unwrap();
        assert_eq!(cb.get_text().unwrap(), "hello world");
    }

    #[test]
    fn test_empty_clipboard() {
        let cb = make_clipboard();
        let result = cb.get_text();
        assert!(result.is_err());
    }

    #[test]
    fn test_clear() {
        let cb = make_clipboard();
        cb.set_text("data").unwrap();
        cb.clear().unwrap();
        assert!(!cb.has_format(ClipboardFormat::Text));
    }

    #[test]
    fn test_image_clipboard() {
        let cb = make_clipboard();
        let img = ClipboardImage::solid(2, 2, 255, 0, 0, 255);
        cb.set_image(&img).unwrap();
        let retrieved = cb.get_image().unwrap();
        assert_eq!(retrieved.width, 2);
        assert_eq!(retrieved.height, 2);
        assert_eq!(retrieved.data.len(), 16);
    }

    #[test]
    fn test_available_formats() {
        let cb = make_clipboard();
        assert!(cb.available_formats().is_empty());
        cb.set_text("test").unwrap();
        assert!(cb.has_format(ClipboardFormat::Text));
        assert!(!cb.has_format(ClipboardFormat::Image));
    }

    #[test]
    fn test_clipboard_history() {
        let mut history = ClipboardHistory::new(5, 1024 * 1024);
        history.push(ClipboardContent::Text("first".into()));
        history.push(ClipboardContent::Text("second".into()));
        history.push(ClipboardContent::Text("third".into()));
        assert_eq!(history.len(), 3);
        assert_eq!(history.get(0).unwrap().content.as_text(), Some("third"));
        assert_eq!(history.get(2).unwrap().content.as_text(), Some("first"));
    }

    #[test]
    fn test_clipboard_history_eviction() {
        let mut history = ClipboardHistory::new(3, 1024 * 1024);
        history.push(ClipboardContent::Text("a".into()));
        history.push(ClipboardContent::Text("b".into()));
        history.push(ClipboardContent::Text("c".into()));
        history.push(ClipboardContent::Text("d".into()));
        assert_eq!(history.len(), 3);
        // Oldest ("a") should have been evicted
        let texts: Vec<&str> = history
            .iter()
            .filter_map(|e| e.content.as_text())
            .collect();
        assert!(!texts.contains(&"a"));
        assert!(texts.contains(&"d"));
    }

    #[test]
    fn test_clipboard_history_pin() {
        let mut history = ClipboardHistory::new(2, 1024 * 1024);
        history.push(ClipboardContent::Text("pinned".into()));
        history.pin(0);
        history.push(ClipboardContent::Text("new1".into()));
        history.push(ClipboardContent::Text("new2".into()));
        // Pinned entry should survive
        let has_pinned = history
            .iter()
            .any(|e| e.content.as_text() == Some("pinned"));
        assert!(has_pinned);
    }

    #[test]
    fn test_clipboard_history_search() {
        let mut history = ClipboardHistory::new(10, 1024 * 1024);
        history.push(ClipboardContent::Text("Hello World".into()));
        history.push(ClipboardContent::Text("foo bar".into()));
        history.push(ClipboardContent::Text("hello again".into()));
        let results = history.search_text("hello");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_clipboard_history_clear() {
        let mut history = ClipboardHistory::new(10, 1024 * 1024);
        history.push(ClipboardContent::Text("data".into()));
        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.total_bytes(), 0);
    }

    #[test]
    fn test_clipboard_image_validation() {
        let valid = ClipboardImage::solid(4, 4, 0, 0, 0, 255);
        assert!(valid.is_valid());
        let invalid = ClipboardImage::new(4, 4, vec![0; 10]); // wrong size
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_clipboard_content_format() {
        let text = ClipboardContent::Text("hi".into());
        assert_eq!(text.format(), Some(ClipboardFormat::Text));
        let empty = ClipboardContent::Empty;
        assert_eq!(empty.format(), None);
    }

    #[test]
    fn test_clipboard_content_size() {
        let text = ClipboardContent::Text("hello".into());
        assert_eq!(text.size_bytes(), 5);
        let img = ClipboardContent::Image(ClipboardImage::solid(2, 2, 0, 0, 0, 0));
        assert_eq!(img.size_bytes(), 16);
    }

    #[test]
    fn test_macos_clipboard() {
        let cb = MacOsClipboard::new();
        cb.set_text("mac text").unwrap();
        assert_eq!(cb.get_text().unwrap(), "mac text");
    }

    #[test]
    fn test_x11_clipboard() {
        let cb = X11Clipboard::new();
        cb.set_text("x11 text").unwrap();
        assert_eq!(cb.get_text().unwrap(), "x11 text");
    }

    #[test]
    fn test_clipboard_format_display() {
        assert_eq!(ClipboardFormat::Text.to_string(), "text/plain");
        assert_eq!(ClipboardFormat::Image.to_string(), "image/rgba");
    }

    #[test]
    fn test_clipboard_error_display() {
        let err = ClipboardError::Unavailable;
        assert_eq!(err.to_string(), "clipboard unavailable");
    }
}
