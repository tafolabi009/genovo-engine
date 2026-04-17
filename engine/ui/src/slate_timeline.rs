//! Timeline widget: track-based timeline, keyframe display, playhead scrubbing,
//! zoom/scroll, track mute/solo, and multi-track selection.
//!
//! Provides a professional-grade timeline UI component for animation editing,
//! cinematic sequencing, and any time-based data editing.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Time and range types
// ---------------------------------------------------------------------------

/// Time position in the timeline (in seconds).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TimePosition(pub f64);

impl TimePosition {
    pub const ZERO: Self = Self(0.0);

    pub fn from_seconds(s: f64) -> Self {
        Self(s)
    }

    pub fn from_frames(frame: u32, fps: f64) -> Self {
        Self(frame as f64 / fps)
    }

    pub fn to_frames(&self, fps: f64) -> u32 {
        (self.0 * fps).round() as u32
    }

    pub fn format_mmss(&self) -> String {
        let total_secs = self.0.max(0.0) as u64;
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        let frac = ((self.0 - total_secs as f64) * 100.0) as u32;
        format!("{:02}:{:02}.{:02}", mins, secs, frac)
    }

    pub fn format_hmmss(&self) -> String {
        let total_secs = self.0.max(0.0) as u64;
        let hrs = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        format!("{:01}:{:02}:{:02}", hrs, mins, secs)
    }
}

impl std::ops::Add for TimePosition {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
}

impl std::ops::Sub for TimePosition {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
}

/// Visible time range in the timeline.
#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    pub start: TimePosition,
    pub end: TimePosition,
}

impl TimeRange {
    pub fn new(start: f64, end: f64) -> Self {
        Self {
            start: TimePosition(start),
            end: TimePosition(end),
        }
    }

    pub fn duration(&self) -> f64 {
        self.end.0 - self.start.0
    }

    pub fn contains(&self, time: TimePosition) -> bool {
        time.0 >= self.start.0 && time.0 <= self.end.0
    }

    pub fn clamp(&self, time: TimePosition) -> TimePosition {
        TimePosition(time.0.clamp(self.start.0, self.end.0))
    }

    pub fn time_to_normalized(&self, time: TimePosition) -> f64 {
        if self.duration() <= 0.0 { return 0.0; }
        (time.0 - self.start.0) / self.duration()
    }

    pub fn normalized_to_time(&self, t: f64) -> TimePosition {
        TimePosition(self.start.0 + t * self.duration())
    }

    pub fn zoom(&self, factor: f64, pivot: TimePosition) -> Self {
        let start_offset = (self.start.0 - pivot.0) / factor;
        let end_offset = (self.end.0 - pivot.0) / factor;
        Self {
            start: TimePosition(pivot.0 + start_offset),
            end: TimePosition(pivot.0 + end_offset),
        }
    }

    pub fn pan(&self, offset: f64) -> Self {
        Self {
            start: TimePosition(self.start.0 + offset),
            end: TimePosition(self.end.0 + offset),
        }
    }
}

// ---------------------------------------------------------------------------
// Keyframe
// ---------------------------------------------------------------------------

/// Unique keyframe identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyframeId(pub u64);

/// Interpolation mode between keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    Constant,
    Linear,
    Cubic,
    Bezier,
}

/// A keyframe on a track.
#[derive(Debug, Clone)]
pub struct TimelineKeyframe {
    pub id: KeyframeId,
    pub time: TimePosition,
    pub value: f64,
    pub interpolation: InterpolationMode,
    pub selected: bool,
    pub tangent_in: f64,
    pub tangent_out: f64,
    pub color: Option<[f32; 4]>,
    pub label: Option<String>,
}

impl TimelineKeyframe {
    pub fn new(id: KeyframeId, time: TimePosition, value: f64) -> Self {
        Self {
            id,
            time,
            value,
            interpolation: InterpolationMode::Linear,
            selected: false,
            tangent_in: 0.0,
            tangent_out: 0.0,
            color: None,
            label: None,
        }
    }

    pub fn with_interpolation(mut self, mode: InterpolationMode) -> Self {
        self.interpolation = mode;
        self
    }
}

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------

/// Unique track identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackId(pub u64);

/// Type of content on a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    Float,
    Vec3,
    Color,
    Bool,
    Event,
    Audio,
    Camera,
    Custom,
}

/// Track state flags.
#[derive(Debug, Clone, Copy)]
pub struct TrackFlags {
    pub muted: bool,
    pub soloed: bool,
    pub locked: bool,
    pub visible: bool,
    pub expanded: bool,
    pub record_enabled: bool,
}

impl Default for TrackFlags {
    fn default() -> Self {
        Self {
            muted: false,
            soloed: false,
            locked: false,
            visible: true,
            expanded: true,
            record_enabled: false,
        }
    }
}

/// A track in the timeline.
#[derive(Debug, Clone)]
pub struct TimelineTrack {
    pub id: TrackId,
    pub name: String,
    pub track_type: TrackType,
    pub flags: TrackFlags,
    pub keyframes: Vec<TimelineKeyframe>,
    pub color: [f32; 4],
    pub height: f32,
    pub selected: bool,
    pub parent: Option<TrackId>,
    pub children: Vec<TrackId>,
    pub depth: u32,
    pub target_path: String,
    pub value_range: (f64, f64),
}

impl TimelineTrack {
    pub fn new(id: TrackId, name: impl Into<String>, track_type: TrackType) -> Self {
        Self {
            id,
            name: name.into(),
            track_type,
            flags: TrackFlags::default(),
            keyframes: Vec::new(),
            color: [0.4, 0.6, 0.8, 1.0],
            height: 24.0,
            selected: false,
            parent: None,
            children: Vec::new(),
            depth: 0,
            target_path: String::new(),
            value_range: (0.0, 1.0),
        }
    }

    pub fn add_keyframe(&mut self, keyframe: TimelineKeyframe) {
        // Insert sorted by time.
        let pos = self.keyframes.iter().position(|k| k.time.0 > keyframe.time.0);
        match pos {
            Some(idx) => self.keyframes.insert(idx, keyframe),
            None => self.keyframes.push(keyframe),
        }
    }

    pub fn remove_keyframe(&mut self, id: KeyframeId) -> bool {
        let len = self.keyframes.len();
        self.keyframes.retain(|k| k.id != id);
        self.keyframes.len() < len
    }

    pub fn keyframe_at(&self, time: TimePosition, tolerance: f64) -> Option<&TimelineKeyframe> {
        self.keyframes.iter().find(|k| (k.time.0 - time.0).abs() < tolerance)
    }

    pub fn keyframe_at_mut(&mut self, time: TimePosition, tolerance: f64) -> Option<&mut TimelineKeyframe> {
        self.keyframes.iter_mut().find(|k| (k.time.0 - time.0).abs() < tolerance)
    }

    pub fn evaluate(&self, time: TimePosition) -> f64 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].value;
        }

        // Find surrounding keyframes.
        if time.0 <= self.keyframes[0].time.0 {
            return self.keyframes[0].value;
        }
        let last = self.keyframes.last().unwrap();
        if time.0 >= last.time.0 {
            return last.value;
        }

        for i in 0..self.keyframes.len() - 1 {
            let a = &self.keyframes[i];
            let b = &self.keyframes[i + 1];
            if time.0 >= a.time.0 && time.0 <= b.time.0 {
                let t = (time.0 - a.time.0) / (b.time.0 - a.time.0);
                return match a.interpolation {
                    InterpolationMode::Constant => a.value,
                    InterpolationMode::Linear => a.value + (b.value - a.value) * t,
                    InterpolationMode::Cubic => {
                        let t2 = t * t;
                        let t3 = t2 * t;
                        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                        let h10 = t3 - 2.0 * t2 + t;
                        let h01 = -2.0 * t3 + 3.0 * t2;
                        let h11 = t3 - t2;
                        h00 * a.value + h10 * a.tangent_out + h01 * b.value + h11 * b.tangent_in
                    }
                    InterpolationMode::Bezier => {
                        // Simplified cubic bezier.
                        let u = 1.0 - t;
                        u * u * u * a.value + 3.0 * u * u * t * (a.value + a.tangent_out)
                            + 3.0 * u * t * t * (b.value + b.tangent_in) + t * t * t * b.value
                    }
                };
            }
        }

        0.0
    }

    pub fn duration(&self) -> f64 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        self.keyframes.last().unwrap().time.0 - self.keyframes[0].time.0
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    pub fn selected_keyframes(&self) -> Vec<KeyframeId> {
        self.keyframes.iter().filter(|k| k.selected).map(|k| k.id).collect()
    }

    pub fn select_keyframes_in_range(&mut self, range: TimeRange) {
        for k in &mut self.keyframes {
            k.selected = range.contains(k.time);
        }
    }

    pub fn deselect_all_keyframes(&mut self) {
        for k in &mut self.keyframes {
            k.selected = false;
        }
    }

    pub fn is_effectively_muted(&self) -> bool {
        self.flags.muted
    }
}

// ---------------------------------------------------------------------------
// Marker
// ---------------------------------------------------------------------------

/// A named marker on the timeline.
#[derive(Debug, Clone)]
pub struct TimelineMarker {
    pub id: u64,
    pub time: TimePosition,
    pub name: String,
    pub color: [f32; 4],
    pub is_range: bool,
    pub range_end: Option<TimePosition>,
}

impl TimelineMarker {
    pub fn new(id: u64, time: TimePosition, name: impl Into<String>) -> Self {
        Self {
            id,
            time,
            name: name.into(),
            color: [1.0, 0.8, 0.2, 1.0],
            is_range: false,
            range_end: None,
        }
    }

    pub fn range(id: u64, start: TimePosition, end: TimePosition, name: impl Into<String>) -> Self {
        Self {
            id,
            time: start,
            name: name.into(),
            color: [0.2, 0.8, 0.6, 0.5],
            is_range: true,
            range_end: Some(end),
        }
    }
}

// ---------------------------------------------------------------------------
// Playhead / transport
// ---------------------------------------------------------------------------

/// Transport state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportState {
    Stopped,
    Playing,
    Paused,
    Recording,
}

/// Playhead and transport controls.
#[derive(Debug, Clone)]
pub struct TimelineTransport {
    pub state: TransportState,
    pub playhead: TimePosition,
    pub play_range: Option<TimeRange>,
    pub loop_enabled: bool,
    pub speed: f64,
    pub fps: f64,
    pub snap_to_frames: bool,
}

impl TimelineTransport {
    pub fn new(fps: f64) -> Self {
        Self {
            state: TransportState::Stopped,
            playhead: TimePosition::ZERO,
            play_range: None,
            loop_enabled: false,
            speed: 1.0,
            fps,
            snap_to_frames: true,
        }
    }

    pub fn play(&mut self) {
        self.state = TransportState::Playing;
    }

    pub fn pause(&mut self) {
        self.state = TransportState::Paused;
    }

    pub fn stop(&mut self) {
        self.state = TransportState::Stopped;
        self.playhead = TimePosition::ZERO;
    }

    pub fn toggle_play(&mut self) {
        match self.state {
            TransportState::Playing => self.pause(),
            _ => self.play(),
        }
    }

    pub fn seek(&mut self, time: TimePosition) {
        self.playhead = time;
        if self.snap_to_frames {
            let frame = self.playhead.to_frames(self.fps);
            self.playhead = TimePosition::from_frames(frame, self.fps);
        }
    }

    pub fn advance(&mut self, delta_seconds: f64) {
        if self.state != TransportState::Playing {
            return;
        }

        self.playhead.0 += delta_seconds * self.speed;

        if let Some(range) = &self.play_range {
            if self.playhead.0 > range.end.0 {
                if self.loop_enabled {
                    self.playhead.0 = range.start.0 + (self.playhead.0 - range.end.0);
                } else {
                    self.playhead = range.end;
                    self.state = TransportState::Stopped;
                }
            }
        }
    }

    pub fn current_frame(&self) -> u32 {
        self.playhead.to_frames(self.fps)
    }

    pub fn is_playing(&self) -> bool {
        self.state == TransportState::Playing
    }
}

// ---------------------------------------------------------------------------
// Timeline widget state
// ---------------------------------------------------------------------------

/// Selection mode for the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimelineSelectionMode {
    Single,
    Multi,
    Range,
}

/// Events emitted by the timeline widget.
#[derive(Debug, Clone)]
pub enum TimelineEvent {
    PlayheadMoved(TimePosition),
    KeyframeAdded(TrackId, KeyframeId),
    KeyframeRemoved(TrackId, KeyframeId),
    KeyframeMoved(TrackId, KeyframeId, TimePosition),
    KeyframeSelected(TrackId, KeyframeId),
    TrackAdded(TrackId),
    TrackRemoved(TrackId),
    TrackMuted(TrackId, bool),
    TrackSoloed(TrackId, bool),
    TrackSelected(TrackId),
    ZoomChanged(f64),
    RangeChanged(TimeRange),
    MarkerAdded(u64),
    TransportStateChanged(TransportState),
}

/// Main timeline widget state.
pub struct TimelineState {
    pub tracks: Vec<TimelineTrack>,
    pub markers: Vec<TimelineMarker>,
    pub transport: TimelineTransport,
    pub visible_range: TimeRange,
    pub total_duration: f64,
    pub zoom: f64,
    pub scroll_x: f64,
    pub scroll_y: f64,
    pub selection_mode: TimelineSelectionMode,
    pub selected_tracks: Vec<TrackId>,
    pub events: Vec<TimelineEvent>,
    pub next_track_id: u64,
    pub next_keyframe_id: u64,
    pub next_marker_id: u64,
    pub show_curves: bool,
    pub show_values: bool,
    pub header_width: f32,
    pub track_default_height: f32,
    pub snap_enabled: bool,
    pub snap_interval: f64,
    pub auto_scroll: bool,
}

impl TimelineState {
    pub fn new(fps: f64, duration: f64) -> Self {
        Self {
            tracks: Vec::new(),
            markers: Vec::new(),
            transport: TimelineTransport::new(fps),
            visible_range: TimeRange::new(0.0, duration.min(10.0)),
            total_duration: duration,
            zoom: 1.0,
            scroll_x: 0.0,
            scroll_y: 0.0,
            selection_mode: TimelineSelectionMode::Multi,
            selected_tracks: Vec::new(),
            events: Vec::new(),
            next_track_id: 1,
            next_keyframe_id: 1,
            next_marker_id: 1,
            show_curves: false,
            show_values: true,
            header_width: 200.0,
            track_default_height: 24.0,
            snap_enabled: true,
            snap_interval: 0.0,
            auto_scroll: true,
        }
    }

    pub fn add_track(&mut self, name: impl Into<String>, track_type: TrackType) -> TrackId {
        let id = TrackId(self.next_track_id);
        self.next_track_id += 1;
        let track = TimelineTrack::new(id, name, track_type);
        self.tracks.push(track);
        self.events.push(TimelineEvent::TrackAdded(id));
        id
    }

    pub fn remove_track(&mut self, id: TrackId) -> bool {
        let len = self.tracks.len();
        self.tracks.retain(|t| t.id != id);
        if self.tracks.len() < len {
            self.events.push(TimelineEvent::TrackRemoved(id));
            true
        } else {
            false
        }
    }

    pub fn get_track(&self, id: TrackId) -> Option<&TimelineTrack> {
        self.tracks.iter().find(|t| t.id == id)
    }

    pub fn get_track_mut(&mut self, id: TrackId) -> Option<&mut TimelineTrack> {
        self.tracks.iter_mut().find(|t| t.id == id)
    }

    pub fn add_keyframe(&mut self, track_id: TrackId, time: TimePosition, value: f64) -> Option<KeyframeId> {
        let kf_id = KeyframeId(self.next_keyframe_id);
        self.next_keyframe_id += 1;
        let keyframe = TimelineKeyframe::new(kf_id, time, value);
        if let Some(track) = self.tracks.iter_mut().find(|t| t.id == track_id) {
            track.add_keyframe(keyframe);
            self.events.push(TimelineEvent::KeyframeAdded(track_id, kf_id));
            Some(kf_id)
        } else {
            None
        }
    }

    pub fn remove_keyframe(&mut self, track_id: TrackId, kf_id: KeyframeId) -> bool {
        if let Some(track) = self.tracks.iter_mut().find(|t| t.id == track_id) {
            if track.remove_keyframe(kf_id) {
                self.events.push(TimelineEvent::KeyframeRemoved(track_id, kf_id));
                return true;
            }
        }
        false
    }

    pub fn add_marker(&mut self, time: TimePosition, name: impl Into<String>) -> u64 {
        let id = self.next_marker_id;
        self.next_marker_id += 1;
        self.markers.push(TimelineMarker::new(id, time, name));
        self.events.push(TimelineEvent::MarkerAdded(id));
        id
    }

    pub fn toggle_mute(&mut self, track_id: TrackId) {
        if let Some(track) = self.tracks.iter_mut().find(|t| t.id == track_id) {
            track.flags.muted = !track.flags.muted;
            self.events.push(TimelineEvent::TrackMuted(track_id, track.flags.muted));
        }
    }

    pub fn toggle_solo(&mut self, track_id: TrackId) {
        if let Some(track) = self.tracks.iter_mut().find(|t| t.id == track_id) {
            track.flags.soloed = !track.flags.soloed;
            self.events.push(TimelineEvent::TrackSoloed(track_id, track.flags.soloed));
        }
    }

    pub fn set_zoom(&mut self, zoom: f64) {
        self.zoom = zoom.clamp(0.01, 100.0);
        self.events.push(TimelineEvent::ZoomChanged(self.zoom));
    }

    pub fn zoom_in(&mut self) { self.set_zoom(self.zoom * 1.5); }
    pub fn zoom_out(&mut self) { self.set_zoom(self.zoom / 1.5); }

    pub fn zoom_to_fit(&mut self) {
        self.visible_range = TimeRange::new(0.0, self.total_duration);
    }

    pub fn time_to_pixel(&self, time: TimePosition, widget_width: f64) -> f64 {
        self.visible_range.time_to_normalized(time) * widget_width
    }

    pub fn pixel_to_time(&self, pixel: f64, widget_width: f64) -> TimePosition {
        self.visible_range.normalized_to_time(pixel / widget_width)
    }

    pub fn snap_time(&self, time: TimePosition) -> TimePosition {
        if !self.snap_enabled || self.snap_interval <= 0.0 {
            if self.transport.snap_to_frames {
                let frame = time.to_frames(self.transport.fps);
                return TimePosition::from_frames(frame, self.transport.fps);
            }
            return time;
        }
        let snapped = (time.0 / self.snap_interval).round() * self.snap_interval;
        TimePosition(snapped)
    }

    pub fn update(&mut self, delta_seconds: f64) {
        self.transport.advance(delta_seconds);

        // Auto-scroll to keep playhead visible.
        if self.auto_scroll && self.transport.is_playing() {
            if !self.visible_range.contains(self.transport.playhead) {
                let duration = self.visible_range.duration();
                self.visible_range = TimeRange::new(
                    self.transport.playhead.0,
                    self.transport.playhead.0 + duration,
                );
            }
        }
    }

    pub fn select_all_keyframes_in_range(&mut self, range: TimeRange) {
        for track in &mut self.tracks {
            track.select_keyframes_in_range(range);
        }
    }

    pub fn deselect_all(&mut self) {
        for track in &mut self.tracks {
            track.selected = false;
            track.deselect_all_keyframes();
        }
        self.selected_tracks.clear();
    }

    pub fn delete_selected_keyframes(&mut self) {
        for track in &mut self.tracks {
            let selected: Vec<KeyframeId> = track.selected_keyframes();
            for kf_id in selected {
                track.remove_keyframe(kf_id);
            }
        }
    }

    pub fn evaluate_all(&self, time: TimePosition) -> HashMap<TrackId, f64> {
        let mut values = HashMap::new();
        for track in &self.tracks {
            if !track.is_effectively_muted() {
                values.insert(track.id, track.evaluate(time));
            }
        }
        values
    }

    pub fn track_count(&self) -> usize { self.tracks.len() }
    pub fn marker_count(&self) -> usize { self.markers.len() }

    pub fn drain_events(&mut self) -> Vec<TimelineEvent> {
        std::mem::take(&mut self.events)
    }
}

impl Default for TimelineState {
    fn default() -> Self {
        Self::new(30.0, 10.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_position_format() {
        let t = TimePosition::from_seconds(125.5);
        assert_eq!(t.format_mmss(), "02:05.50");
    }

    #[test]
    fn time_range_operations() {
        let range = TimeRange::new(1.0, 5.0);
        assert_eq!(range.duration(), 4.0);
        assert!(range.contains(TimePosition(3.0)));
        assert!(!range.contains(TimePosition(6.0)));
    }

    #[test]
    fn track_keyframe_evaluation() {
        let mut track = TimelineTrack::new(TrackId(1), "test", TrackType::Float);
        track.add_keyframe(TimelineKeyframe::new(KeyframeId(1), TimePosition(0.0), 0.0));
        track.add_keyframe(TimelineKeyframe::new(KeyframeId(2), TimePosition(1.0), 10.0));

        let v = track.evaluate(TimePosition(0.5));
        assert!((v - 5.0).abs() < 0.01);
    }

    #[test]
    fn transport_advance() {
        let mut transport = TimelineTransport::new(30.0);
        transport.play();
        transport.advance(1.0 / 30.0);
        assert!(transport.playhead.0 > 0.0);
    }

    #[test]
    fn timeline_state_basic() {
        let mut state = TimelineState::new(30.0, 10.0);
        let track = state.add_track("Position.X", TrackType::Float);
        state.add_keyframe(track, TimePosition(0.0), 0.0);
        state.add_keyframe(track, TimePosition(1.0), 100.0);

        assert_eq!(state.track_count(), 1);

        let values = state.evaluate_all(TimePosition(0.5));
        assert!((values[&track] - 50.0).abs() < 0.01);
    }
}
