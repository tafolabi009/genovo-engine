//! Enhanced replay: bookmark system, highlight detection, replay export,
//! replay sharing, replay analysis (stats), and cinematic replay mode.

use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReplayId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BookmarkId(pub u64);

#[derive(Debug, Clone)]
pub struct ReplayBookmark { pub id: BookmarkId, pub time: f32, pub name: String, pub description: String, pub color: [f32; 4], pub camera_position: Option<[f32; 3]>, pub camera_target: Option<[f32; 3]> }
impl ReplayBookmark { pub fn new(id: BookmarkId, time: f32, name: impl Into<String>) -> Self { Self { id, time, name: name.into(), description: String::new(), color: [1.0, 0.8, 0.2, 1.0], camera_position: None, camera_target: None } } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighlightType { Kill, Death, Objective, MultiKill, Streak, Custom }
impl HighlightType { pub fn score(&self) -> f32 { match self { Self::Kill => 1.0, Self::Death => 0.3, Self::Objective => 2.0, Self::MultiKill => 3.0, Self::Streak => 2.5, Self::Custom => 1.0 } } }

#[derive(Debug, Clone)]
pub struct ReplayHighlight { pub time: f32, pub duration: f32, pub highlight_type: HighlightType, pub score: f32, pub description: String, pub entities: Vec<u64>, pub auto_detected: bool }
impl ReplayHighlight { pub fn new(time: f32, ht: HighlightType) -> Self { Self { time, duration: 3.0, highlight_type: ht, score: ht.score(), description: String::new(), entities: Vec::new(), auto_detected: true } } }

#[derive(Debug, Clone)]
pub struct ReplayStats { pub total_duration: f32, pub total_kills: u32, pub total_deaths: u32, pub total_damage: f64, pub total_healing: f64, pub distance_traveled: f64, pub shots_fired: u32, pub shots_hit: u32, pub accuracy: f32, pub highest_streak: u32, pub objectives_completed: u32, pub custom_stats: HashMap<String, f64> }
impl ReplayStats {
    pub fn new() -> Self { Self { total_duration: 0.0, total_kills: 0, total_deaths: 0, total_damage: 0.0, total_healing: 0.0, distance_traveled: 0.0, shots_fired: 0, shots_hit: 0, accuracy: 0.0, highest_streak: 0, objectives_completed: 0, custom_stats: HashMap::new() } }
    pub fn kd_ratio(&self) -> f32 { if self.total_deaths == 0 { self.total_kills as f32 } else { self.total_kills as f32 / self.total_deaths as f32 } }
    pub fn compute_accuracy(&mut self) { self.accuracy = if self.shots_fired == 0 { 0.0 } else { self.shots_hit as f32 / self.shots_fired as f32 * 100.0 }; }
}
impl Default for ReplayStats { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CinematicMode { FreeCam, FollowEntity, PathCamera, OrbitalCamera, TopDown }

#[derive(Debug, Clone)]
pub struct CinematicSettings { pub mode: CinematicMode, pub follow_entity: Option<u64>, pub camera_speed: f32, pub smooth_factor: f32, pub dof_enabled: bool, pub dof_focus_distance: f32, pub motion_blur: bool, pub letterbox: bool, pub letterbox_aspect: f32, pub time_scale: f32, pub hide_hud: bool, pub hide_ui_entities: bool }
impl Default for CinematicSettings { fn default() -> Self { Self { mode: CinematicMode::FreeCam, follow_entity: None, camera_speed: 5.0, smooth_factor: 5.0, dof_enabled: false, dof_focus_distance: 10.0, motion_blur: false, letterbox: true, letterbox_aspect: 2.35, time_scale: 1.0, hide_hud: true, hide_ui_entities: true } } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat { Video, GIF, FrameSequence, ReplayFile }

#[derive(Debug, Clone)]
pub struct ExportConfig { pub format: ExportFormat, pub resolution: (u32, u32), pub framerate: u32, pub quality: f32, pub start_time: f32, pub end_time: f32, pub include_audio: bool, pub include_hud: bool, pub output_path: String }
impl Default for ExportConfig { fn default() -> Self { Self { format: ExportFormat::Video, resolution: (1920, 1080), framerate: 60, quality: 0.9, start_time: 0.0, end_time: 0.0, include_audio: true, include_hud: false, output_path: String::new() } } }

#[derive(Debug, Clone)]
pub struct ReplayFrame { pub time: f32, pub entity_positions: HashMap<u64, [f32; 3]>, pub entity_rotations: HashMap<u64, [f32; 4]>, pub events: Vec<String>, pub input_state: Vec<u8> }

#[derive(Debug, Clone)]
pub struct ReplayDataV2 { pub id: ReplayId, pub name: String, pub frames: Vec<ReplayFrame>, pub bookmarks: Vec<ReplayBookmark>, pub highlights: Vec<ReplayHighlight>, pub stats: ReplayStats, pub duration: f32, pub player_name: String, pub map_name: String, pub game_mode: String, pub timestamp: u64, pub version: u32 }
impl ReplayDataV2 {
    pub fn new(id: ReplayId, name: impl Into<String>) -> Self {
        Self { id, name: name.into(), frames: Vec::new(), bookmarks: Vec::new(), highlights: Vec::new(), stats: ReplayStats::new(), duration: 0.0, player_name: String::new(), map_name: String::new(), game_mode: String::new(), timestamp: 0, version: 1 }
    }
    pub fn frame_count(&self) -> usize { self.frames.len() }
    pub fn bookmark_count(&self) -> usize { self.bookmarks.len() }
    pub fn highlight_count(&self) -> usize { self.highlights.len() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackStateV2 { Stopped, Playing, Paused, Recording, Exporting }

#[derive(Debug, Clone)]
pub enum ReplayEvent { PlaybackStarted(ReplayId), PlaybackStopped, PlaybackPaused, PlaybackResumed, BookmarkAdded(BookmarkId), BookmarkRemoved(BookmarkId), HighlightDetected(usize), SeekTo(f32), SpeedChanged(f32), ExportStarted, ExportCompleted(String), ExportFailed(String), RecordingStarted, RecordingStopped }

pub struct ReplaySystemV2 {
    pub replays: HashMap<ReplayId, ReplayDataV2>,
    pub active_replay: Option<ReplayId>,
    pub playback_state: PlaybackStateV2,
    pub playback_time: f32,
    pub playback_speed: f32,
    pub cinematic: CinematicSettings,
    pub export_config: ExportConfig,
    pub events: Vec<ReplayEvent>,
    pub next_replay_id: u64,
    pub next_bookmark_id: u64,
    pub recording_data: Option<ReplayDataV2>,
    pub recording_interval: f32,
    pub recording_timer: f32,
    pub max_replay_duration: f32,
    pub auto_detect_highlights: bool,
}

impl ReplaySystemV2 {
    pub fn new() -> Self {
        Self { replays: HashMap::new(), active_replay: None, playback_state: PlaybackStateV2::Stopped, playback_time: 0.0, playback_speed: 1.0, cinematic: CinematicSettings::default(), export_config: ExportConfig::default(), events: Vec::new(), next_replay_id: 1, next_bookmark_id: 1, recording_data: None, recording_interval: 1.0 / 30.0, recording_timer: 0.0, max_replay_duration: 3600.0, auto_detect_highlights: true }
    }

    pub fn start_recording(&mut self, name: impl Into<String>) -> ReplayId {
        let id = ReplayId(self.next_replay_id); self.next_replay_id += 1;
        let data = ReplayDataV2::new(id, name);
        self.recording_data = Some(data);
        self.playback_state = PlaybackStateV2::Recording;
        self.events.push(ReplayEvent::RecordingStarted);
        id
    }

    pub fn stop_recording(&mut self) -> Option<ReplayId> {
        if let Some(data) = self.recording_data.take() {
            let id = data.id;
            self.replays.insert(id, data);
            self.playback_state = PlaybackStateV2::Stopped;
            self.events.push(ReplayEvent::RecordingStopped);
            Some(id)
        } else { None }
    }

    pub fn record_frame(&mut self, frame: ReplayFrame) {
        if let Some(ref mut data) = self.recording_data { data.duration = frame.time; data.frames.push(frame); }
    }

    pub fn play(&mut self, id: ReplayId) -> bool {
        if self.replays.contains_key(&id) { self.active_replay = Some(id); self.playback_state = PlaybackStateV2::Playing; self.playback_time = 0.0; self.events.push(ReplayEvent::PlaybackStarted(id)); true } else { false }
    }

    pub fn pause(&mut self) { if self.playback_state == PlaybackStateV2::Playing { self.playback_state = PlaybackStateV2::Paused; self.events.push(ReplayEvent::PlaybackPaused); } }
    pub fn resume(&mut self) { if self.playback_state == PlaybackStateV2::Paused { self.playback_state = PlaybackStateV2::Playing; self.events.push(ReplayEvent::PlaybackResumed); } }
    pub fn stop(&mut self) { self.playback_state = PlaybackStateV2::Stopped; self.active_replay = None; self.events.push(ReplayEvent::PlaybackStopped); }
    pub fn seek(&mut self, time: f32) { self.playback_time = time.max(0.0); self.events.push(ReplayEvent::SeekTo(time)); }
    pub fn set_speed(&mut self, speed: f32) { self.playback_speed = speed.clamp(0.1, 16.0); self.events.push(ReplayEvent::SpeedChanged(speed)); }

    pub fn add_bookmark(&mut self, replay_id: ReplayId, time: f32, name: impl Into<String>) -> Option<BookmarkId> {
        let id = BookmarkId(self.next_bookmark_id); self.next_bookmark_id += 1;
        if let Some(replay) = self.replays.get_mut(&replay_id) { replay.bookmarks.push(ReplayBookmark::new(id, time, name)); self.events.push(ReplayEvent::BookmarkAdded(id)); Some(id) } else { None }
    }

    pub fn detect_highlights(&mut self, replay_id: ReplayId) {
        if !self.auto_detect_highlights { return; }
        if let Some(replay) = self.replays.get_mut(&replay_id) {
            let mut highlights = Vec::new();
            for frame in &replay.frames {
                for event in &frame.events {
                    if event.contains("kill") { highlights.push(ReplayHighlight::new(frame.time, HighlightType::Kill)); }
                    if event.contains("objective") { highlights.push(ReplayHighlight::new(frame.time, HighlightType::Objective)); }
                }
            }
            for (i, h) in highlights.iter().enumerate() { self.events.push(ReplayEvent::HighlightDetected(i)); }
            replay.highlights.extend(highlights);
        }
    }

    pub fn update(&mut self, dt: f32) {
        if self.playback_state == PlaybackStateV2::Playing {
            self.playback_time += dt * self.playback_speed;
            if let Some(id) = self.active_replay {
                if let Some(replay) = self.replays.get(&id) {
                    if self.playback_time >= replay.duration { self.stop(); }
                }
            }
        }
    }

    pub fn current_frame(&self) -> Option<&ReplayFrame> {
        let id = self.active_replay?;
        let replay = self.replays.get(&id)?;
        replay.frames.iter().min_by(|a, b| {
            let da = (a.time - self.playback_time).abs();
            let db = (b.time - self.playback_time).abs();
            da.partial_cmp(&db).unwrap()
        })
    }

    pub fn replay_count(&self) -> usize { self.replays.len() }
    pub fn is_recording(&self) -> bool { self.playback_state == PlaybackStateV2::Recording }
    pub fn is_playing(&self) -> bool { self.playback_state == PlaybackStateV2::Playing }
    pub fn drain_events(&mut self) -> Vec<ReplayEvent> { std::mem::take(&mut self.events) }
}
impl Default for ReplaySystemV2 { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn record_and_playback() {
        let mut sys = ReplaySystemV2::new();
        let id = sys.start_recording("Test");
        sys.record_frame(ReplayFrame { time: 0.0, entity_positions: HashMap::new(), entity_rotations: HashMap::new(), events: Vec::new(), input_state: Vec::new() });
        sys.record_frame(ReplayFrame { time: 1.0, entity_positions: HashMap::new(), entity_rotations: HashMap::new(), events: Vec::new(), input_state: Vec::new() });
        let id = sys.stop_recording().unwrap();
        assert!(sys.play(id));
        assert!(sys.is_playing());
    }
    #[test]
    fn bookmark_add() {
        let mut sys = ReplaySystemV2::new();
        let id = sys.start_recording("BM Test");
        let id = sys.stop_recording().unwrap();
        let bm = sys.add_bookmark(id, 0.5, "Important");
        assert!(bm.is_some());
    }
    #[test]
    fn stats_kd() {
        let mut stats = ReplayStats::new();
        stats.total_kills = 10; stats.total_deaths = 5;
        assert!((stats.kd_ratio() - 2.0).abs() < 0.01);
    }
}
