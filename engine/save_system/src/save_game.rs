//! Save game data structures and the save manager.
//!
//! This module provides the core save/load infrastructure:
//!
//! - [`SaveMetadata`] — timestamps, playtime, screenshot path, etc.
//! - [`SaveGame`] — complete save file: metadata + serialized world state.
//! - [`SaveManager`] — manages save slots, auto-save, quick-save/quick-load.
//!
//! # Save File Format
//!
//! Save files are JSON-based by default, with optional compression. Each save
//! file includes:
//!
//! 1. A header with magic bytes and version info
//! 2. Metadata (name, timestamp, playtime, screenshot path)
//! 3. Serialized world state (all entities and their components)
//! 4. An integrity checksum
//!
//! Cloud save compatibility is ensured by including platform-agnostic metadata.

use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize as SerdeSerialize};
use thiserror::Error;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during save/load operations.
#[derive(Debug, Error)]
pub enum SaveError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Save file is corrupt or has been tampered with.
    #[error("Integrity check failed: {0}")]
    IntegrityError(String),

    /// Save file version is not supported.
    #[error("Unsupported save version: {0}")]
    UnsupportedVersion(u32),

    /// Save slot is empty.
    #[error("Save slot {0} is empty")]
    EmptySlot(usize),

    /// Save slot does not exist.
    #[error("Save slot {0} does not exist")]
    InvalidSlot(usize),

    /// Save operation failed.
    #[error("Save failed: {0}")]
    SaveFailed(String),

    /// Load operation failed.
    #[error("Load failed: {0}")]
    LoadFailed(String),

    /// JSON parse error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type for save operations.
pub type SaveResult<T> = Result<T, SaveError>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes for save file identification.
const SAVE_MAGIC: &[u8; 4] = b"GNSV";

/// Current save format version.
const SAVE_VERSION: u32 = 1;

/// Maximum number of save slots.
const MAX_SAVE_SLOTS: usize = 100;

/// Default auto-save interval (in seconds).
const DEFAULT_AUTO_SAVE_INTERVAL: f64 = 300.0; // 5 minutes

// ---------------------------------------------------------------------------
// SaveMetadata
// ---------------------------------------------------------------------------

/// Metadata stored alongside the save game data.
#[derive(Debug, Clone, SerdeSerialize, Deserialize)]
pub struct SaveMetadata {
    /// Unique identifier for this save.
    pub id: Uuid,
    /// User-visible name for the save (e.g., "Autosave - Level 3").
    pub name: String,
    /// Optional description.
    pub description: String,
    /// Unix timestamp when the save was created.
    pub timestamp: u64,
    /// Total playtime in seconds at the time of saving.
    pub playtime_seconds: f64,
    /// Path to a screenshot taken at save time (if any).
    pub screenshot_path: Option<String>,
    /// Name of the current level/scene.
    pub level_name: String,
    /// Custom key-value metadata for game-specific use.
    pub custom_data: HashMap<String, String>,
    /// Save format version.
    pub format_version: u32,
    /// Platform identifier (for cloud save compatibility).
    pub platform: String,
    /// Engine version that created this save.
    pub engine_version: String,
    /// Whether this save was created by auto-save.
    pub is_auto_save: bool,
    /// Checksum of the world state data.
    pub checksum: u32,
}

impl SaveMetadata {
    /// Create new metadata with sensible defaults.
    pub fn new(name: impl Into<String>, level_name: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: String::new(),
            timestamp: now,
            playtime_seconds: 0.0,
            screenshot_path: None,
            level_name: level_name.into(),
            custom_data: HashMap::new(),
            format_version: SAVE_VERSION,
            platform: Self::detect_platform(),
            engine_version: "0.1.0".to_string(),
            is_auto_save: false,
            checksum: 0,
        }
    }

    /// Set the playtime.
    pub fn with_playtime(mut self, seconds: f64) -> Self {
        self.playtime_seconds = seconds;
        self
    }

    /// Set the screenshot path.
    pub fn with_screenshot(mut self, path: impl Into<String>) -> Self {
        self.screenshot_path = Some(path.into());
        self
    }

    /// Set a custom key-value pair.
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_data.insert(key.into(), value.into());
        self
    }

    /// Mark as auto-save.
    pub fn as_auto_save(mut self) -> Self {
        self.is_auto_save = true;
        self
    }

    /// Format the timestamp as a human-readable string.
    pub fn formatted_timestamp(&self) -> String {
        // Simple formatting: seconds since epoch as a readable number.
        let duration = Duration::from_secs(self.timestamp);
        let total_days = duration.as_secs() / 86400;
        let time_of_day = duration.as_secs() % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;
        format!(
            "Day {} {:02}:{:02}:{:02}",
            total_days, hours, minutes, seconds
        )
    }

    /// Format playtime as HH:MM:SS.
    pub fn formatted_playtime(&self) -> String {
        let total = self.playtime_seconds as u64;
        let hours = total / 3600;
        let minutes = (total % 3600) / 60;
        let seconds = total % 60;
        format!("{}:{:02}:{:02}", hours, minutes, seconds)
    }

    /// Detect the current platform.
    fn detect_platform() -> String {
        #[cfg(target_os = "windows")]
        {
            "windows".to_string()
        }
        #[cfg(target_os = "linux")]
        {
            "linux".to_string()
        }
        #[cfg(target_os = "macos")]
        {
            "macos".to_string()
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        {
            "unknown".to_string()
        }
    }
}

impl fmt::Display for SaveMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}) — Playtime: {} — Level: {}",
            self.name,
            self.formatted_timestamp(),
            self.formatted_playtime(),
            self.level_name,
        )
    }
}

// ---------------------------------------------------------------------------
// ComponentData / EntityData — serialized world state
// ---------------------------------------------------------------------------

/// Serialized component data (type-erased).
#[derive(Debug, Clone, SerdeSerialize, Deserialize)]
pub struct ComponentData {
    /// Component type name (used for reconstruction).
    pub type_name: String,
    /// Serialized component data (JSON value).
    pub data: serde_json::Value,
}

/// Serialized entity data.
#[derive(Debug, Clone, SerdeSerialize, Deserialize)]
pub struct EntityData {
    /// Entity id.
    pub id: u32,
    /// Entity generation.
    pub generation: u32,
    /// List of serialized components.
    pub components: Vec<ComponentData>,
}

/// Serialized world state.
#[derive(Debug, Clone, SerdeSerialize, Deserialize)]
pub struct WorldState {
    /// All entities and their components.
    pub entities: Vec<EntityData>,
    /// Serialized singleton resources.
    pub resources: HashMap<String, serde_json::Value>,
}

impl WorldState {
    /// Create an empty world state.
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            resources: HashMap::new(),
        }
    }

    /// Get the total number of entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get the total number of components across all entities.
    pub fn component_count(&self) -> usize {
        self.entities.iter().map(|e| e.components.len()).sum()
    }

    /// Get the total number of resources.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }
}

impl Default for WorldState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SaveGame
// ---------------------------------------------------------------------------

/// A complete save game file.
///
/// Contains metadata and the serialized world state.
#[derive(Debug, Clone, SerdeSerialize, Deserialize)]
pub struct SaveGame {
    /// Save metadata.
    pub metadata: SaveMetadata,
    /// Serialized world state.
    pub world_state: WorldState,
}

impl SaveGame {
    /// Create a new save game.
    pub fn new(metadata: SaveMetadata, world_state: WorldState) -> Self {
        Self {
            metadata,
            world_state,
        }
    }

    /// Serialize the save game to a JSON byte vector.
    pub fn to_bytes(&self) -> SaveResult<Vec<u8>> {
        let json = serde_json::to_string_pretty(self)?;
        let json_bytes = json.as_bytes();

        // Build the save file: magic + version + checksum + json data.
        let checksum = Self::compute_checksum(json_bytes);
        let mut buffer = Vec::with_capacity(12 + json_bytes.len());
        buffer.extend_from_slice(SAVE_MAGIC);
        buffer.extend_from_slice(&SAVE_VERSION.to_le_bytes());
        buffer.extend_from_slice(&checksum.to_le_bytes());
        buffer.extend_from_slice(json_bytes);

        Ok(buffer)
    }

    /// Deserialize a save game from bytes.
    pub fn from_bytes(bytes: &[u8]) -> SaveResult<Self> {
        if bytes.len() < 12 {
            return Err(SaveError::IntegrityError(
                "File too small to be a valid save".into(),
            ));
        }

        // Validate magic.
        if &bytes[0..4] != SAVE_MAGIC {
            return Err(SaveError::IntegrityError(
                "Invalid save file magic bytes".into(),
            ));
        }

        // Read version.
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version > SAVE_VERSION {
            return Err(SaveError::UnsupportedVersion(version));
        }

        // Read and verify checksum.
        let stored_checksum =
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let json_data = &bytes[12..];
        let computed_checksum = Self::compute_checksum(json_data);

        if stored_checksum != computed_checksum {
            return Err(SaveError::IntegrityError(format!(
                "Checksum mismatch: stored 0x{:08X}, computed 0x{:08X}",
                stored_checksum, computed_checksum,
            )));
        }

        // Parse JSON.
        let save: SaveGame = serde_json::from_slice(json_data)?;
        Ok(save)
    }

    /// Save to a file.
    pub fn save_to_file(&self, path: &Path) -> SaveResult<()> {
        let bytes = self.to_bytes()?;

        // Ensure parent directory exists.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        file.flush()?;

        log::info!(
            "Saved game to {} ({} bytes)",
            path.display(),
            bytes.len()
        );
        Ok(())
    }

    /// Load from a file.
    pub fn load_from_file(path: &Path) -> SaveResult<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let save = Self::from_bytes(&bytes)?;
        log::info!(
            "Loaded save from {} ({} entities, {} components)",
            path.display(),
            save.world_state.entity_count(),
            save.world_state.component_count(),
        );
        Ok(save)
    }

    /// Compute a simple CRC-like checksum over the data.
    fn compute_checksum(data: &[u8]) -> u32 {
        // Simple Adler-32-like checksum for integrity verification.
        let mut a: u32 = 1;
        let mut b: u32 = 0;
        for &byte in data {
            a = (a + byte as u32) % 65521;
            b = (b + a) % 65521;
        }
        (b << 16) | a
    }

    /// Get the save file size estimate (without actually serializing).
    pub fn estimated_size(&self) -> usize {
        // Rough estimate based on entity/component counts.
        let base = 512; // metadata overhead
        let per_entity = 128;
        let per_component = 256;
        base + self.world_state.entity_count() * per_entity
            + self.world_state.component_count() * per_component
    }
}

impl fmt::Display for SaveGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SaveGame({}, entities: {}, components: {})",
            self.metadata,
            self.world_state.entity_count(),
            self.world_state.component_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// SaveSlot
// ---------------------------------------------------------------------------

/// A save slot that may or may not contain a save game.
#[derive(Debug, Clone)]
pub struct SaveSlot {
    /// Slot index.
    pub index: usize,
    /// Optional metadata (present if a save exists in this slot).
    pub metadata: Option<SaveMetadata>,
    /// File path for this slot.
    pub file_path: PathBuf,
}

impl SaveSlot {
    /// Create a new empty save slot.
    pub fn new(index: usize, base_dir: &Path) -> Self {
        Self {
            index,
            metadata: None,
            file_path: base_dir.join(format!("save_{:03}.sav", index)),
        }
    }

    /// Check if this slot has a save.
    pub fn is_occupied(&self) -> bool {
        self.metadata.is_some()
    }

    /// Check if the save file exists on disk.
    pub fn file_exists(&self) -> bool {
        self.file_path.exists()
    }

    /// Get the file size (or 0 if the file doesn't exist).
    pub fn file_size(&self) -> u64 {
        std::fs::metadata(&self.file_path)
            .map(|m| m.len())
            .unwrap_or(0)
    }
}

impl fmt::Display for SaveSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.metadata {
            Some(meta) => write!(f, "Slot {}: {}", self.index, meta),
            None => write!(f, "Slot {}: <empty>", self.index),
        }
    }
}

// ---------------------------------------------------------------------------
// SaveManager
// ---------------------------------------------------------------------------

/// Manages save slots, auto-save, and quick-save/quick-load.
///
/// # Example
///
/// ```ignore
/// let mut manager = SaveManager::new(Path::new("saves"));
///
/// // Save to a slot.
/// let world_state = WorldState::new();
/// let metadata = SaveMetadata::new("My Save", "Level 1");
/// manager.save_to_slot(0, metadata, world_state)?;
///
/// // Load from a slot.
/// let save = manager.load_from_slot(0)?;
///
/// // Quick-save.
/// manager.quick_save(metadata, world_state)?;
///
/// // Quick-load.
/// let save = manager.quick_load()?;
///
/// // Auto-save (called every frame, only saves at interval).
/// manager.auto_save_tick(0.016, || {
///     (SaveMetadata::new("Autosave", "Level 1"), WorldState::new())
/// });
/// ```
pub struct SaveManager {
    /// Base directory for save files.
    save_dir: PathBuf,
    /// Maximum number of save slots.
    max_slots: usize,
    /// Cached slot metadata.
    slots: Vec<SaveSlot>,
    /// Quick-save file path.
    quick_save_path: PathBuf,
    /// Auto-save file path.
    auto_save_path: PathBuf,
    /// Auto-save interval in seconds.
    auto_save_interval: f64,
    /// Time since last auto-save (seconds).
    auto_save_timer: f64,
    /// Whether auto-save is enabled.
    auto_save_enabled: bool,
    /// Maximum number of auto-save rotations.
    auto_save_rotations: usize,
    /// Current auto-save rotation index.
    auto_save_index: usize,
    /// Cloud save metadata path.
    cloud_metadata_path: PathBuf,
}

/// Cloud sync metadata for tracking which saves need uploading.
#[derive(Debug, SerdeSerialize, Deserialize)]
pub struct CloudSyncMetadata {
    /// Map of save slot index -> (save_id, last_sync_timestamp).
    pub synced_saves: HashMap<String, u64>,
    /// Whether there are pending uploads.
    pub pending_upload: bool,
    /// Last sync timestamp.
    pub last_sync: u64,
}

impl SaveManager {
    /// Create a new save manager with the given base directory.
    pub fn new(save_dir: &Path) -> Self {
        let mut manager = Self {
            save_dir: save_dir.to_path_buf(),
            max_slots: MAX_SAVE_SLOTS,
            slots: Vec::new(),
            quick_save_path: save_dir.join("quicksave.sav"),
            auto_save_path: save_dir.join("autosave_000.sav"),
            auto_save_interval: DEFAULT_AUTO_SAVE_INTERVAL,
            auto_save_timer: 0.0,
            auto_save_enabled: true,
            auto_save_rotations: 3,
            auto_save_index: 0,
            cloud_metadata_path: save_dir.join("cloud_sync.json"),
        };
        manager.init_slots();
        manager
    }

    /// Initialize save slots.
    fn init_slots(&mut self) {
        self.slots.clear();
        for i in 0..self.max_slots {
            let mut slot = SaveSlot::new(i, &self.save_dir);
            // Try to read metadata from existing save files.
            if slot.file_exists() {
                if let Ok(save) = SaveGame::load_from_file(&slot.file_path) {
                    slot.metadata = Some(save.metadata);
                }
            }
            self.slots.push(slot);
        }
    }

    /// Get the save directory.
    pub fn save_dir(&self) -> &Path {
        &self.save_dir
    }

    /// Set the maximum number of save slots.
    pub fn set_max_slots(&mut self, max: usize) {
        self.max_slots = max;
        self.init_slots();
    }

    // -- Slot management ---------------------------------------------------

    /// Get all save slots.
    pub fn slots(&self) -> &[SaveSlot] {
        &self.slots
    }

    /// Get a specific slot.
    pub fn get_slot(&self, index: usize) -> SaveResult<&SaveSlot> {
        self.slots
            .get(index)
            .ok_or(SaveError::InvalidSlot(index))
    }

    /// Get the number of occupied slots.
    pub fn occupied_slot_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_occupied()).count()
    }

    /// Get the first empty slot, if any.
    pub fn first_empty_slot(&self) -> Option<usize> {
        self.slots.iter().position(|s| !s.is_occupied())
    }

    // -- Save/Load ---------------------------------------------------------

    /// Save to a specific slot.
    pub fn save_to_slot(
        &mut self,
        slot_index: usize,
        metadata: SaveMetadata,
        world_state: WorldState,
    ) -> SaveResult<()> {
        if slot_index >= self.max_slots {
            return Err(SaveError::InvalidSlot(slot_index));
        }

        let save = SaveGame::new(metadata.clone(), world_state);
        let path = self.save_dir.join(format!("save_{:03}.sav", slot_index));
        save.save_to_file(&path)?;

        // Update cached slot metadata.
        if slot_index < self.slots.len() {
            self.slots[slot_index].metadata = Some(metadata);
            self.slots[slot_index].file_path = path;
        }

        Ok(())
    }

    /// Load from a specific slot.
    pub fn load_from_slot(&self, slot_index: usize) -> SaveResult<SaveGame> {
        let slot = self.get_slot(slot_index)?;
        if !slot.is_occupied() {
            return Err(SaveError::EmptySlot(slot_index));
        }
        SaveGame::load_from_file(&slot.file_path)
    }

    /// Delete a save in a specific slot.
    pub fn delete_slot(&mut self, slot_index: usize) -> SaveResult<()> {
        if slot_index >= self.slots.len() {
            return Err(SaveError::InvalidSlot(slot_index));
        }

        let slot = &self.slots[slot_index];
        if slot.file_exists() {
            std::fs::remove_file(&slot.file_path)?;
        }
        self.slots[slot_index].metadata = None;

        log::info!("Deleted save slot {}", slot_index);
        Ok(())
    }

    // -- Quick-save/load ---------------------------------------------------

    /// Quick-save (overwrites the quick-save slot).
    pub fn quick_save(
        &mut self,
        metadata: SaveMetadata,
        world_state: WorldState,
    ) -> SaveResult<()> {
        let save = SaveGame::new(metadata, world_state);
        save.save_to_file(&self.quick_save_path)?;
        log::info!("Quick-saved to {}", self.quick_save_path.display());
        Ok(())
    }

    /// Quick-load (loads from the quick-save slot).
    pub fn quick_load(&self) -> SaveResult<SaveGame> {
        if !self.quick_save_path.exists() {
            return Err(SaveError::LoadFailed(
                "No quick-save exists".into(),
            ));
        }
        SaveGame::load_from_file(&self.quick_save_path)
    }

    /// Check if a quick-save exists.
    pub fn has_quick_save(&self) -> bool {
        self.quick_save_path.exists()
    }

    // -- Auto-save ---------------------------------------------------------

    /// Enable or disable auto-save.
    pub fn set_auto_save_enabled(&mut self, enabled: bool) {
        self.auto_save_enabled = enabled;
    }

    /// Check if auto-save is enabled.
    pub fn is_auto_save_enabled(&self) -> bool {
        self.auto_save_enabled
    }

    /// Set the auto-save interval (in seconds).
    pub fn set_auto_save_interval(&mut self, seconds: f64) {
        self.auto_save_interval = seconds;
    }

    /// Get the auto-save interval.
    pub fn auto_save_interval(&self) -> f64 {
        self.auto_save_interval
    }

    /// Set the number of auto-save rotations.
    pub fn set_auto_save_rotations(&mut self, count: usize) {
        self.auto_save_rotations = count.max(1);
    }

    /// Called every frame. Performs auto-save when the interval has elapsed.
    ///
    /// The `create_save_fn` closure is only called when an auto-save is
    /// actually triggered.
    pub fn auto_save_tick<F>(&mut self, delta_time: f64, create_save_fn: F)
    where
        F: FnOnce() -> (SaveMetadata, WorldState),
    {
        if !self.auto_save_enabled {
            return;
        }

        self.auto_save_timer += delta_time;
        if self.auto_save_timer < self.auto_save_interval {
            return;
        }

        self.auto_save_timer = 0.0;

        let (mut metadata, world_state) = create_save_fn();
        metadata.is_auto_save = true;

        let path = self
            .save_dir
            .join(format!("autosave_{:03}.sav", self.auto_save_index));
        self.auto_save_index = (self.auto_save_index + 1) % self.auto_save_rotations;

        let save = SaveGame::new(metadata, world_state);
        match save.save_to_file(&path) {
            Ok(()) => log::info!("Auto-saved to {}", path.display()),
            Err(e) => log::error!("Auto-save failed: {}", e),
        }
    }

    /// Reset the auto-save timer.
    pub fn reset_auto_save_timer(&mut self) {
        self.auto_save_timer = 0.0;
    }

    // -- Cloud sync metadata ------------------------------------------------

    /// Write cloud sync metadata.
    pub fn write_cloud_metadata(&self, synced_saves: &HashMap<String, u64>) -> SaveResult<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let metadata = CloudSyncMetadata {
            synced_saves: synced_saves.clone(),
            pending_upload: false,
            last_sync: now,
        };

        let json = serde_json::to_string_pretty(&metadata)?;
        if let Some(parent) = self.cloud_metadata_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.cloud_metadata_path, json)?;
        Ok(())
    }

    /// Read cloud sync metadata.
    pub fn read_cloud_metadata(&self) -> SaveResult<Option<CloudSyncMetadata>> {
        if !self.cloud_metadata_path.exists() {
            return Ok(None);
        }
        let json = std::fs::read_to_string(&self.cloud_metadata_path)?;
        let metadata: CloudSyncMetadata = serde_json::from_str(&json)?;
        Ok(Some(metadata))
    }

    /// Scan the save directory and refresh slot metadata.
    pub fn refresh_slots(&mut self) {
        self.init_slots();
    }

    /// Get the total disk space used by all saves.
    pub fn total_save_size(&self) -> u64 {
        self.slots.iter().map(|s| s.file_size()).sum()
    }

    /// Get a summary of all saves for display.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Save Directory: {}\n",
            self.save_dir.display()
        ));
        out.push_str(&format!(
            "Occupied Slots: {} / {}\n",
            self.occupied_slot_count(),
            self.max_slots,
        ));
        out.push_str(&format!(
            "Total Size: {} bytes\n",
            self.total_save_size(),
        ));
        out.push_str(&format!(
            "Quick-save: {}\n",
            if self.has_quick_save() { "yes" } else { "no" },
        ));
        out.push_str(&format!(
            "Auto-save: {} (interval: {:.0}s)\n",
            if self.auto_save_enabled {
                "enabled"
            } else {
                "disabled"
            },
            self.auto_save_interval,
        ));
        out
    }
}

impl fmt::Debug for SaveManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SaveManager")
            .field("save_dir", &self.save_dir)
            .field("max_slots", &self.max_slots)
            .field("occupied_slots", &self.occupied_slot_count())
            .field("auto_save_enabled", &self.auto_save_enabled)
            .field("auto_save_interval", &self.auto_save_interval)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_metadata_creation() {
        let meta = SaveMetadata::new("Test Save", "Level 1");
        assert_eq!(meta.name, "Test Save");
        assert_eq!(meta.level_name, "Level 1");
        assert!(!meta.is_auto_save);
        assert_eq!(meta.format_version, SAVE_VERSION);
    }

    #[test]
    fn save_metadata_builder_pattern() {
        let meta = SaveMetadata::new("Test", "Level")
            .with_playtime(3600.0)
            .with_screenshot("/tmp/screenshot.png")
            .with_custom("difficulty", "hard")
            .as_auto_save();

        assert_eq!(meta.playtime_seconds, 3600.0);
        assert_eq!(meta.screenshot_path, Some("/tmp/screenshot.png".into()));
        assert_eq!(meta.custom_data.get("difficulty").unwrap(), "hard");
        assert!(meta.is_auto_save);
    }

    #[test]
    fn formatted_playtime() {
        let meta = SaveMetadata::new("Test", "Level").with_playtime(3661.0);
        assert_eq!(meta.formatted_playtime(), "1:01:01");
    }

    #[test]
    fn world_state_counts() {
        let mut state = WorldState::new();
        state.entities.push(EntityData {
            id: 0,
            generation: 0,
            components: vec![
                ComponentData {
                    type_name: "Position".into(),
                    data: serde_json::json!({"x": 1.0, "y": 2.0}),
                },
                ComponentData {
                    type_name: "Health".into(),
                    data: serde_json::json!(100),
                },
            ],
        });
        assert_eq!(state.entity_count(), 1);
        assert_eq!(state.component_count(), 2);
    }

    #[test]
    fn save_game_round_trip_bytes() {
        let meta = SaveMetadata::new("Test Save", "Level 1");
        let state = WorldState::new();
        let save = SaveGame::new(meta, state);

        let bytes = save.to_bytes().unwrap();
        let restored = SaveGame::from_bytes(&bytes).unwrap();

        assert_eq!(restored.metadata.name, "Test Save");
        assert_eq!(restored.metadata.level_name, "Level 1");
    }

    #[test]
    fn save_game_checksum_verification() {
        let meta = SaveMetadata::new("Test", "L1");
        let state = WorldState::new();
        let save = SaveGame::new(meta, state);
        let mut bytes = save.to_bytes().unwrap();

        // Corrupt one byte of the JSON data.
        if bytes.len() > 15 {
            bytes[15] ^= 0xFF;
        }

        assert!(SaveGame::from_bytes(&bytes).is_err());
    }

    #[test]
    fn save_game_invalid_magic() {
        let bytes = vec![0u8; 20];
        let result = SaveGame::from_bytes(&bytes);
        assert!(result.is_err());
        match result.unwrap_err() {
            SaveError::IntegrityError(_) => {} // expected
            other => panic!("Expected IntegrityError, got {:?}", other),
        }
    }

    #[test]
    fn save_game_too_small() {
        let bytes = vec![0u8; 4];
        assert!(SaveGame::from_bytes(&bytes).is_err());
    }

    #[test]
    fn save_slot_creation() {
        let slot = SaveSlot::new(0, Path::new("/tmp/saves"));
        assert_eq!(slot.index, 0);
        assert!(!slot.is_occupied());
        assert!(slot.file_path.to_str().unwrap().contains("save_000.sav"));
    }

    #[test]
    fn save_manager_creation() {
        let manager = SaveManager::new(Path::new("/tmp/test_saves"));
        assert_eq!(manager.max_slots, MAX_SAVE_SLOTS);
        assert!(manager.is_auto_save_enabled());
        assert_eq!(manager.occupied_slot_count(), 0);
    }

    #[test]
    fn save_manager_first_empty_slot() {
        let manager = SaveManager::new(Path::new("/tmp/test_saves"));
        assert_eq!(manager.first_empty_slot(), Some(0));
    }

    #[test]
    fn save_game_file_round_trip() {
        let temp_dir = std::env::temp_dir().join("genovo_save_test");
        let _ = std::fs::create_dir_all(&temp_dir);
        let file_path = temp_dir.join("test_save.sav");

        let meta = SaveMetadata::new("File Test", "Level 2");
        let mut state = WorldState::new();
        state.entities.push(EntityData {
            id: 1,
            generation: 0,
            components: vec![ComponentData {
                type_name: "Position".into(),
                data: serde_json::json!({"x": 10.0, "y": 20.0, "z": 30.0}),
            }],
        });
        state
            .resources
            .insert("TimeScale".into(), serde_json::json!(1.0));

        let save = SaveGame::new(meta, state);
        save.save_to_file(&file_path).unwrap();

        let loaded = SaveGame::load_from_file(&file_path).unwrap();
        assert_eq!(loaded.metadata.name, "File Test");
        assert_eq!(loaded.world_state.entity_count(), 1);
        assert_eq!(loaded.world_state.resource_count(), 1);

        // Cleanup.
        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&temp_dir);
    }

    #[test]
    fn auto_save_timer() {
        let mut manager = SaveManager::new(Path::new("/tmp/test_saves"));
        manager.set_auto_save_interval(1.0);

        // Should not trigger (not enough time).
        let mut triggered = false;
        manager.auto_save_tick(0.5, || {
            triggered = true;
            (
                SaveMetadata::new("Auto", "L1"),
                WorldState::new(),
            )
        });
        assert!(!triggered);
    }

    #[test]
    fn checksum_deterministic() {
        let data = b"Hello, World!";
        let c1 = SaveGame::compute_checksum(data);
        let c2 = SaveGame::compute_checksum(data);
        assert_eq!(c1, c2);

        let different = b"Different data";
        let c3 = SaveGame::compute_checksum(different);
        assert_ne!(c1, c3);
    }

    #[test]
    fn estimated_size() {
        let state = WorldState::new();
        let save = SaveGame::new(
            SaveMetadata::new("Test", "L1"),
            state,
        );
        assert!(save.estimated_size() > 0);
    }
}
