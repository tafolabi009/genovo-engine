// engine/gameplay/src/save_game_v2.rs
//
// Advanced save game system for the Genovo gameplay framework.
//
// Provides screenshot thumbnails, playtime tracking, difficulty level,
// save versioning, cloud save metadata, save slot management (10 slots
// + autosave), and save/load events.

use std::collections::HashMap;
use std::fmt;

pub type SaveSlotId = u32;

pub const MAX_SAVE_SLOTS: usize = 10;
pub const AUTOSAVE_SLOT_ID: SaveSlotId = 0;
pub const QUICKSAVE_SLOT_ID: SaveSlotId = 11;
pub const SAVE_FORMAT_VERSION: u32 = 3;
pub const MAX_THUMBNAIL_WIDTH: u32 = 320;
pub const MAX_THUMBNAIL_HEIGHT: u32 = 180;
pub const MAX_SAVE_DATA_SIZE: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DifficultyLevel { Easy, Normal, Hard, VeryHard, Custom }

impl fmt::Display for DifficultyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Easy => write!(f, "Easy"), Self::Normal => write!(f, "Normal"), Self::Hard => write!(f, "Hard"), Self::VeryHard => write!(f, "Very Hard"), Self::Custom => write!(f, "Custom") }
    }
}

#[derive(Debug, Clone)]
pub struct Thumbnail {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub format: ThumbnailFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailFormat { Rgba8, Rgb8, Jpeg, Png }

impl Thumbnail {
    pub fn new(width: u32, height: u32, format: ThumbnailFormat) -> Self {
        let size = match format { ThumbnailFormat::Rgba8 => (width * height * 4) as usize, ThumbnailFormat::Rgb8 => (width * height * 3) as usize, _ => 0 };
        Self { width, height, pixels: vec![0u8; size], format }
    }
    pub fn is_empty(&self) -> bool { self.pixels.is_empty() }
    pub fn byte_size(&self) -> usize { self.pixels.len() }
}

#[derive(Debug, Clone)]
pub struct PlaytimeTracker {
    pub total_seconds: f64,
    pub session_seconds: f64,
    pub session_start_time: f64,
    pub total_sessions: u32,
    pub longest_session_seconds: f64,
}

impl Default for PlaytimeTracker {
    fn default() -> Self {
        Self { total_seconds: 0.0, session_seconds: 0.0, session_start_time: 0.0, total_sessions: 0, longest_session_seconds: 0.0 }
    }
}

impl PlaytimeTracker {
    pub fn new() -> Self { Self::default() }
    pub fn start_session(&mut self, time: f64) { self.session_start_time = time; self.session_seconds = 0.0; self.total_sessions += 1; }
    pub fn update(&mut self, dt: f64) { self.session_seconds += dt; self.total_seconds += dt; if self.session_seconds > self.longest_session_seconds { self.longest_session_seconds = self.session_seconds; } }
    pub fn format_playtime(&self) -> String {
        let hours = (self.total_seconds / 3600.0) as u32;
        let minutes = ((self.total_seconds % 3600.0) / 60.0) as u32;
        let seconds = (self.total_seconds % 60.0) as u32;
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    }
}

#[derive(Debug, Clone)]
pub struct SaveVersionInfo {
    pub format_version: u32,
    pub game_version: String,
    pub engine_version: String,
    pub platform: String,
    pub compatible_versions: Vec<u32>,
}

impl Default for SaveVersionInfo {
    fn default() -> Self {
        Self { format_version: SAVE_FORMAT_VERSION, game_version: "1.0.0".to_string(), engine_version: "0.1.0".to_string(), platform: "unknown".to_string(), compatible_versions: vec![1, 2, 3] }
    }
}

impl SaveVersionInfo {
    pub fn is_compatible(&self, version: u32) -> bool { self.compatible_versions.contains(&version) || version == self.format_version }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudSyncStatus { NotSynced, Syncing, Synced, Conflict, Error }

#[derive(Debug, Clone)]
pub struct CloudSaveMetadata {
    pub cloud_id: Option<String>,
    pub sync_status: CloudSyncStatus,
    pub last_sync_timestamp: u64,
    pub cloud_timestamp: u64,
    pub local_timestamp: u64,
    pub checksum: u64,
    pub cloud_provider: String,
}

impl Default for CloudSaveMetadata {
    fn default() -> Self {
        Self { cloud_id: None, sync_status: CloudSyncStatus::NotSynced, last_sync_timestamp: 0, cloud_timestamp: 0, local_timestamp: 0, checksum: 0, cloud_provider: String::new() }
    }
}

#[derive(Debug, Clone)]
pub struct SaveSlotHeader {
    pub slot_id: SaveSlotId,
    pub name: String,
    pub description: String,
    pub timestamp: u64,
    pub playtime: PlaytimeTracker,
    pub difficulty: DifficultyLevel,
    pub version: SaveVersionInfo,
    pub cloud: CloudSaveMetadata,
    pub thumbnail: Option<Thumbnail>,
    pub character_name: String,
    pub character_level: u32,
    pub location_name: String,
    pub completion_percentage: f32,
    pub is_autosave: bool,
    pub is_quicksave: bool,
    pub occupied: bool,
    pub data_size: usize,
    pub custom_metadata: HashMap<String, String>,
}

impl SaveSlotHeader {
    pub fn new(slot_id: SaveSlotId) -> Self {
        Self {
            slot_id, name: format!("Save {}", slot_id), description: String::new(),
            timestamp: 0, playtime: PlaytimeTracker::new(), difficulty: DifficultyLevel::Normal,
            version: SaveVersionInfo::default(), cloud: CloudSaveMetadata::default(),
            thumbnail: None, character_name: String::new(), character_level: 1,
            location_name: String::new(), completion_percentage: 0.0,
            is_autosave: slot_id == AUTOSAVE_SLOT_ID, is_quicksave: slot_id == QUICKSAVE_SLOT_ID,
            occupied: false, data_size: 0, custom_metadata: HashMap::new(),
        }
    }
    pub fn is_compatible(&self) -> bool { self.version.is_compatible(SAVE_FORMAT_VERSION) }
}

#[derive(Debug, Clone)]
pub struct SaveData {
    pub header: SaveSlotHeader,
    pub world_state: Vec<u8>,
    pub player_state: Vec<u8>,
    pub quest_state: Vec<u8>,
    pub inventory_state: Vec<u8>,
    pub custom_data: HashMap<String, Vec<u8>>,
}

impl SaveData {
    pub fn new(header: SaveSlotHeader) -> Self {
        Self { header, world_state: Vec::new(), player_state: Vec::new(), quest_state: Vec::new(), inventory_state: Vec::new(), custom_data: HashMap::new() }
    }
    pub fn total_size(&self) -> usize { self.world_state.len() + self.player_state.len() + self.quest_state.len() + self.inventory_state.len() + self.custom_data.values().map(|v| v.len()).sum::<usize>() }
    pub fn set_custom(&mut self, key: &str, data: Vec<u8>) { self.custom_data.insert(key.to_string(), data); }
    pub fn get_custom(&self, key: &str) -> Option<&[u8]> { self.custom_data.get(key).map(|v| v.as_slice()) }
}

#[derive(Debug, Clone)]
pub enum SaveEvent {
    SaveStarted(SaveSlotId),
    SaveCompleted(SaveSlotId),
    SaveFailed { slot: SaveSlotId, error: String },
    LoadStarted(SaveSlotId),
    LoadCompleted(SaveSlotId),
    LoadFailed { slot: SaveSlotId, error: String },
    AutosaveTriggered,
    SlotDeleted(SaveSlotId),
    CloudSyncStarted(SaveSlotId),
    CloudSyncCompleted(SaveSlotId),
    CloudConflict(SaveSlotId),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SaveSystemStats {
    pub total_saves: u32,
    pub total_loads: u32,
    pub autosaves: u32,
    pub quicksaves: u32,
    pub last_save_size: usize,
    pub last_save_time_ms: f32,
    pub last_load_time_ms: f32,
    pub occupied_slots: u32,
}

pub struct SaveGameSystem {
    slots: Vec<SaveSlotHeader>,
    events: Vec<SaveEvent>,
    stats: SaveSystemStats,
    autosave_interval: f32,
    autosave_timer: f32,
    autosave_enabled: bool,
    max_autosaves: u32,
    playtime: PlaytimeTracker,
    current_difficulty: DifficultyLevel,
    save_in_progress: bool,
    load_in_progress: bool,
}

impl SaveGameSystem {
    pub fn new() -> Self {
        let mut slots = Vec::with_capacity(MAX_SAVE_SLOTS + 2);
        slots.push(SaveSlotHeader::new(AUTOSAVE_SLOT_ID));
        for i in 1..=MAX_SAVE_SLOTS as u32 { slots.push(SaveSlotHeader::new(i)); }
        slots.push(SaveSlotHeader::new(QUICKSAVE_SLOT_ID));
        Self {
            slots, events: Vec::new(), stats: SaveSystemStats::default(),
            autosave_interval: 300.0, autosave_timer: 0.0, autosave_enabled: true,
            max_autosaves: 3, playtime: PlaytimeTracker::new(),
            current_difficulty: DifficultyLevel::Normal,
            save_in_progress: false, load_in_progress: false,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.playtime.update(dt as f64);
        if self.autosave_enabled {
            self.autosave_timer += dt;
            if self.autosave_timer >= self.autosave_interval {
                self.autosave_timer = 0.0;
                self.events.push(SaveEvent::AutosaveTriggered);
                self.stats.autosaves += 1;
            }
        }
    }

    pub fn save_to_slot(&mut self, slot_id: SaveSlotId, data: SaveData) -> Result<(), String> {
        if self.save_in_progress { return Err("Save already in progress".to_string()); }
        if data.total_size() > MAX_SAVE_DATA_SIZE { return Err("Save data too large".to_string()); }
        let slot = self.slots.iter_mut().find(|s| s.slot_id == slot_id);
        if let Some(slot) = slot {
            self.save_in_progress = true;
            self.events.push(SaveEvent::SaveStarted(slot_id));
            slot.occupied = true;
            slot.name = data.header.name.clone();
            slot.description = data.header.description.clone();
            slot.playtime = self.playtime.clone();
            slot.difficulty = self.current_difficulty;
            slot.character_name = data.header.character_name.clone();
            slot.character_level = data.header.character_level;
            slot.location_name = data.header.location_name.clone();
            slot.completion_percentage = data.header.completion_percentage;
            slot.data_size = data.total_size();
            slot.thumbnail = data.header.thumbnail.clone();
            self.stats.total_saves += 1;
            self.stats.last_save_size = data.total_size();
            self.save_in_progress = false;
            self.events.push(SaveEvent::SaveCompleted(slot_id));
            Ok(())
        } else {
            Err(format!("Invalid slot: {}", slot_id))
        }
    }

    pub fn delete_slot(&mut self, slot_id: SaveSlotId) -> bool {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.slot_id == slot_id) {
            *slot = SaveSlotHeader::new(slot_id);
            self.events.push(SaveEvent::SlotDeleted(slot_id));
            true
        } else { false }
    }

    pub fn get_slot(&self, slot_id: SaveSlotId) -> Option<&SaveSlotHeader> { self.slots.iter().find(|s| s.slot_id == slot_id) }
    pub fn occupied_slots(&self) -> Vec<&SaveSlotHeader> { self.slots.iter().filter(|s| s.occupied).collect() }
    pub fn empty_slots(&self) -> Vec<&SaveSlotHeader> { self.slots.iter().filter(|s| !s.occupied && !s.is_autosave).collect() }
    pub fn first_empty_slot(&self) -> Option<SaveSlotId> { self.empty_slots().first().map(|s| s.slot_id) }
    pub fn drain_events(&mut self) -> Vec<SaveEvent> { std::mem::take(&mut self.events) }
    pub fn stats(&self) -> &SaveSystemStats { &self.stats }
    pub fn playtime(&self) -> &PlaytimeTracker { &self.playtime }
    pub fn set_difficulty(&mut self, d: DifficultyLevel) { self.current_difficulty = d; }
    pub fn set_autosave_enabled(&mut self, e: bool) { self.autosave_enabled = e; }
    pub fn set_autosave_interval(&mut self, s: f32) { self.autosave_interval = s.max(30.0); }
    pub fn start_session(&mut self, time: f64) { self.playtime.start_session(time); }
    pub fn all_slots(&self) -> &[SaveSlotHeader] { &self.slots }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_system_creation() {
        let system = SaveGameSystem::new();
        assert_eq!(system.all_slots().len(), MAX_SAVE_SLOTS + 2);
    }

    #[test]
    fn test_save_and_load() {
        let mut system = SaveGameSystem::new();
        let mut header = SaveSlotHeader::new(1);
        header.character_name = "Hero".to_string();
        let data = SaveData::new(header);
        assert!(system.save_to_slot(1, data).is_ok());
        assert!(system.get_slot(1).unwrap().occupied);
    }

    #[test]
    fn test_delete_slot() {
        let mut system = SaveGameSystem::new();
        let header = SaveSlotHeader::new(1);
        let data = SaveData::new(header);
        system.save_to_slot(1, data).unwrap();
        system.delete_slot(1);
        assert!(!system.get_slot(1).unwrap().occupied);
    }

    #[test]
    fn test_playtime_format() {
        let mut tracker = PlaytimeTracker::new();
        tracker.total_seconds = 3661.0;
        assert_eq!(tracker.format_playtime(), "01:01:01");
    }
}
