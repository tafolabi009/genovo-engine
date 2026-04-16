//! # Timeline Editor Data Model
//!
//! Provides the editor-facing data model for multi-track sequence editing.
//! The [`Timeline`] is the top-level container that groups tracks, manages
//! selection state, supports copy/paste operations, and provides
//! serialization to/from JSON.
//!
//! ## Key concepts
//!
//! - **TimelineGroup** -- a named group of tracks (e.g., "Camera", "NPC_01").
//! - **TimelineTrackMeta** -- per-track metadata (visibility, lock, mute, solo).
//! - **Marker** -- a named time marker with a color for marking important moments.
//! - **TimelineSelection** -- tracks the currently selected keyframes, tracks,
//!   and time ranges.
//! - **CurveEditor** -- per-keyframe tangent handles for curve editing.
//! - **CutsceneAsset** -- a complete serializable cutscene definition.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::camera::CameraPath;
use crate::sequencer::{AudioCue, Sequence, TrackKind};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the timeline subsystem.
#[derive(Debug, thiserror::Error)]
pub enum TimelineError {
    #[error("Group not found: {0}")]
    GroupNotFound(String),
    #[error("Track not found: {0}")]
    TrackNotFound(Uuid),
    #[error("Marker not found: {0}")]
    MarkerNotFound(String),
    #[error("Nothing selected")]
    EmptySelection,
    #[error("Clipboard is empty")]
    EmptyClipboard,
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Invalid time range: {0}..{1}")]
    InvalidTimeRange(f32, f32),
}

// ---------------------------------------------------------------------------
// TimelineTrackMeta
// ---------------------------------------------------------------------------

/// Per-track metadata for the timeline editor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineTrackMeta {
    /// The track this metadata belongs to.
    pub track_id: Uuid,
    /// Display name override (if different from the track's own name).
    pub display_name: Option<String>,
    /// Whether this track is visible in the editor.
    pub visible: bool,
    /// Whether this track is locked (prevents editing).
    pub locked: bool,
    /// Whether this track is muted (skipped during playback).
    pub muted: bool,
    /// Whether this track is in solo mode (only solo tracks play).
    pub solo: bool,
    /// Display color for the track in the editor.
    pub color: TrackColor,
    /// Vertical height in the editor (in pixels, for UI).
    pub height: f32,
    /// Whether the track is expanded in the editor (shows keyframe details).
    pub expanded: bool,
}

impl TimelineTrackMeta {
    /// Create default metadata for a track.
    pub fn new(track_id: Uuid) -> Self {
        Self {
            track_id,
            display_name: None,
            visible: true,
            locked: false,
            muted: false,
            solo: false,
            color: TrackColor::default(),
            height: 24.0,
            expanded: false,
        }
    }
}

/// Display color for a track in the editor.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrackColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl TrackColor {
    pub const RED: Self = Self { r: 220, g: 60, b: 60 };
    pub const GREEN: Self = Self { r: 60, g: 180, b: 60 };
    pub const BLUE: Self = Self { r: 60, g: 100, b: 220 };
    pub const YELLOW: Self = Self { r: 220, g: 200, b: 60 };
    pub const CYAN: Self = Self { r: 60, g: 200, b: 200 };
    pub const MAGENTA: Self = Self { r: 200, g: 60, b: 200 };
    pub const ORANGE: Self = Self { r: 230, g: 140, b: 40 };
    pub const WHITE: Self = Self { r: 200, g: 200, b: 200 };

    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Get a color from a palette by index (cycles through 8 colors).
    pub fn from_palette(index: usize) -> Self {
        const PALETTE: [TrackColor; 8] = [
            TrackColor::BLUE,
            TrackColor::RED,
            TrackColor::GREEN,
            TrackColor::YELLOW,
            TrackColor::CYAN,
            TrackColor::MAGENTA,
            TrackColor::ORANGE,
            TrackColor::WHITE,
        ];
        PALETTE[index % PALETTE.len()]
    }
}

impl Default for TrackColor {
    fn default() -> Self {
        Self::BLUE
    }
}

// ---------------------------------------------------------------------------
// TimelineGroup
// ---------------------------------------------------------------------------

/// A named group of tracks in the timeline editor.
///
/// Groups organize related tracks (e.g., all tracks for a character, or all
/// camera tracks) and can be collapsed, locked, or muted as a unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineGroup {
    /// Unique identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Track IDs belonging to this group.
    pub track_ids: Vec<Uuid>,
    /// Track metadata keyed by track ID.
    pub track_meta: HashMap<Uuid, TimelineTrackMeta>,
    /// Whether the group is expanded (shows its tracks).
    pub expanded: bool,
    /// Whether the group is locked (all tracks locked).
    pub locked: bool,
    /// Whether the group is muted (all tracks muted).
    pub muted: bool,
    /// Display color for the group header.
    pub color: TrackColor,
    /// Sort order in the timeline.
    pub order: i32,
}

impl TimelineGroup {
    /// Create a new empty group.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            track_ids: Vec::new(),
            track_meta: HashMap::new(),
            expanded: true,
            locked: false,
            muted: false,
            color: TrackColor::default(),
            order: 0,
        }
    }

    /// Add a track to this group with default metadata.
    pub fn add_track(&mut self, track_id: Uuid) {
        if !self.track_ids.contains(&track_id) {
            let mut meta = TimelineTrackMeta::new(track_id);
            meta.color = TrackColor::from_palette(self.track_ids.len());
            self.track_meta.insert(track_id, meta);
            self.track_ids.push(track_id);
        }
    }

    /// Remove a track from this group.
    pub fn remove_track(&mut self, track_id: Uuid) -> bool {
        if let Some(pos) = self.track_ids.iter().position(|&id| id == track_id) {
            self.track_ids.remove(pos);
            self.track_meta.remove(&track_id);
            true
        } else {
            false
        }
    }

    /// Get metadata for a track.
    pub fn get_meta(&self, track_id: Uuid) -> Option<&TimelineTrackMeta> {
        self.track_meta.get(&track_id)
    }

    /// Get mutable metadata for a track.
    pub fn get_meta_mut(&mut self, track_id: Uuid) -> Option<&mut TimelineTrackMeta> {
        self.track_meta.get_mut(&track_id)
    }

    /// Lock or unlock all tracks in the group.
    pub fn set_all_locked(&mut self, locked: bool) {
        self.locked = locked;
        for meta in self.track_meta.values_mut() {
            meta.locked = locked;
        }
    }

    /// Mute or unmute all tracks in the group.
    pub fn set_all_muted(&mut self, muted: bool) {
        self.muted = muted;
        for meta in self.track_meta.values_mut() {
            meta.muted = muted;
        }
    }
}

// ---------------------------------------------------------------------------
// Marker
// ---------------------------------------------------------------------------

/// A named marker on the timeline.
///
/// Markers identify important moments in the sequence (e.g., "Impact",
/// "Cut to close-up", "Music hit") and can be used for snapping and
/// navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    /// Unique identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Time in seconds.
    pub time: f32,
    /// Display color.
    pub color: TrackColor,
    /// Optional description / notes.
    pub description: String,
    /// Whether this is a navigation bookmark (vs. a simple label).
    pub bookmark: bool,
}

impl Marker {
    /// Create a new marker.
    pub fn new(name: impl Into<String>, time: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            time,
            color: TrackColor::YELLOW,
            description: String::new(),
            bookmark: false,
        }
    }

    /// Create a bookmark marker.
    pub fn bookmark(name: impl Into<String>, time: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            time,
            color: TrackColor::CYAN,
            description: String::new(),
            bookmark: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Snap settings
// ---------------------------------------------------------------------------

/// Snap mode for the timeline editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapMode {
    /// No snapping.
    None,
    /// Snap to frame boundaries.
    Frame,
    /// Snap to beat boundaries (for music sync).
    Beat,
    /// Snap to markers.
    Marker,
    /// Snap to other keyframes.
    Keyframe,
    /// Snap to all of the above.
    All,
}

impl Default for SnapMode {
    fn default() -> Self {
        Self::None
    }
}

/// Snap settings for the timeline editor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapSettings {
    /// Active snap mode.
    pub mode: SnapMode,
    /// Frame rate for frame snapping (e.g., 24, 30, 60).
    pub frame_rate: f32,
    /// BPM for beat snapping.
    pub bpm: f32,
    /// Subdivisions per beat (e.g., 4 for sixteenth notes).
    pub subdivisions: u32,
    /// Snap threshold in seconds (how close to snap).
    pub threshold: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            mode: SnapMode::None,
            frame_rate: 30.0,
            bpm: 120.0,
            subdivisions: 4,
            threshold: 0.05,
        }
    }
}

impl SnapSettings {
    /// Snap a time value according to the current settings.
    pub fn snap_time(&self, time: f32, markers: &[Marker]) -> f32 {
        match self.mode {
            SnapMode::None => time,
            SnapMode::Frame => self.snap_to_frame(time),
            SnapMode::Beat => self.snap_to_beat(time),
            SnapMode::Marker => self.snap_to_marker(time, markers),
            SnapMode::Keyframe => time, // Requires keyframe data from the caller.
            SnapMode::All => {
                let frame_snapped = self.snap_to_frame(time);
                let beat_snapped = self.snap_to_beat(time);
                let marker_snapped = self.snap_to_marker(time, markers);

                // Return the closest snap point.
                let mut best = time;
                let mut best_dist = f32::MAX;
                for candidate in [frame_snapped, beat_snapped, marker_snapped] {
                    let dist = (candidate - time).abs();
                    if dist < best_dist && dist < self.threshold {
                        best = candidate;
                        best_dist = dist;
                    }
                }
                best
            }
        }
    }

    /// Snap to the nearest frame boundary.
    fn snap_to_frame(&self, time: f32) -> f32 {
        if self.frame_rate <= 0.0 {
            return time;
        }
        let frame = (time * self.frame_rate).round();
        frame / self.frame_rate
    }

    /// Snap to the nearest beat boundary.
    fn snap_to_beat(&self, time: f32) -> f32 {
        if self.bpm <= 0.0 {
            return time;
        }
        let beat_duration = 60.0 / self.bpm;
        let sub_duration = beat_duration / self.subdivisions.max(1) as f32;
        let sub = (time / sub_duration).round();
        sub * sub_duration
    }

    /// Snap to the nearest marker.
    fn snap_to_marker(&self, time: f32, markers: &[Marker]) -> f32 {
        let mut best = time;
        let mut best_dist = self.threshold;
        for marker in markers {
            let dist = (marker.time - time).abs();
            if dist < best_dist {
                best = marker.time;
                best_dist = dist;
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// TangentMode & CurveEditor
// ---------------------------------------------------------------------------

/// Tangent mode for a keyframe in the curve editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TangentMode {
    /// Automatically computed tangent (Catmull-Rom from neighbors).
    Auto,
    /// Linear tangent (straight line to neighbor).
    Linear,
    /// Step tangent (flat, no interpolation).
    Step,
    /// Free Bezier handles (user-positioned).
    Bezier,
    /// Weighted Bezier handles (tangent length affects curve shape).
    Weighted,
    /// Flat tangent (zero slope).
    Flat,
}

impl Default for TangentMode {
    fn default() -> Self {
        Self::Auto
    }
}

/// Per-keyframe curve editor data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveKeyframeData {
    /// Index of the keyframe in its track.
    pub keyframe_index: usize,
    /// Track ID.
    pub track_id: Uuid,
    /// Incoming tangent mode.
    pub in_tangent_mode: TangentMode,
    /// Outgoing tangent mode.
    pub out_tangent_mode: TangentMode,
    /// Incoming tangent handle position (relative to keyframe).
    pub in_handle: (f32, f32),
    /// Outgoing tangent handle position (relative to keyframe).
    pub out_handle: (f32, f32),
    /// Whether the tangent is broken (in and out can differ).
    pub broken: bool,
    /// Tangent weight for weighted mode.
    pub in_weight: f32,
    /// Tangent weight for weighted mode.
    pub out_weight: f32,
}

impl CurveKeyframeData {
    /// Create default curve data for a keyframe.
    pub fn new(track_id: Uuid, keyframe_index: usize) -> Self {
        Self {
            keyframe_index,
            track_id,
            in_tangent_mode: TangentMode::Auto,
            out_tangent_mode: TangentMode::Auto,
            in_handle: (-0.1, 0.0),
            out_handle: (0.1, 0.0),
            broken: false,
            in_weight: 1.0,
            out_weight: 1.0,
        }
    }

    /// Compute auto tangent from neighboring keyframe values.
    ///
    /// Uses the Catmull-Rom formula: tangent = (next_value - prev_value) / 2.
    pub fn compute_auto_tangent(
        prev: Option<(f32, f32)>,
        current: (f32, f32),
        next: Option<(f32, f32)>,
    ) -> (f32, f32) {
        match (prev, next) {
            (Some((pt, pv)), Some((nt, nv))) => {
                let dt = nt - pt;
                let dv = nv - pv;
                if dt.abs() < 1e-8 {
                    (0.0, 0.0)
                } else {
                    let slope = dv / dt;
                    // Scale handle length to ~1/3 of the interval
                    let handle_len = (nt - current.0).min(current.0 - pt) / 3.0;
                    (handle_len, slope * handle_len)
                }
            }
            (Some((pt, pv)), None) => {
                let dt = current.0 - pt;
                let dv = current.1 - pv;
                if dt.abs() < 1e-8 {
                    (0.0, 0.0)
                } else {
                    let slope = dv / dt;
                    let handle_len = dt / 3.0;
                    (handle_len, slope * handle_len)
                }
            }
            (None, Some((nt, nv))) => {
                let dt = nt - current.0;
                let dv = nv - current.1;
                if dt.abs() < 1e-8 {
                    (0.0, 0.0)
                } else {
                    let slope = dv / dt;
                    let handle_len = dt / 3.0;
                    (handle_len, slope * handle_len)
                }
            }
            (None, None) => (0.0, 0.0),
        }
    }
}

/// Curve editor state for a timeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CurveEditor {
    /// Per-keyframe curve data, keyed by `(track_id, keyframe_index)`.
    pub keyframe_data: HashMap<(Uuid, usize), CurveKeyframeData>,
    /// Visible value range (for zoom/scroll in the curve editor).
    pub visible_range_min: f32,
    /// Visible value range max.
    pub visible_range_max: f32,
    /// Whether the curve editor is open.
    pub open: bool,
    /// IDs of tracks currently shown in the curve editor.
    pub visible_tracks: Vec<Uuid>,
}

impl CurveEditor {
    /// Create a new curve editor.
    pub fn new() -> Self {
        Self {
            keyframe_data: HashMap::new(),
            visible_range_min: -1.0,
            visible_range_max: 1.0,
            open: false,
            visible_tracks: Vec::new(),
        }
    }

    /// Get or create curve data for a keyframe.
    pub fn get_or_create(&mut self, track_id: Uuid, index: usize) -> &mut CurveKeyframeData {
        self.keyframe_data
            .entry((track_id, index))
            .or_insert_with(|| CurveKeyframeData::new(track_id, index))
    }

    /// Remove curve data for a keyframe (e.g., when the keyframe is deleted).
    pub fn remove(&mut self, track_id: Uuid, index: usize) {
        self.keyframe_data.remove(&(track_id, index));
    }

    /// Recompute auto tangents for all keyframes in a float track.
    pub fn recompute_auto_tangents(&mut self, track_id: Uuid, times_values: &[(f32, f32)]) {
        for i in 0..times_values.len() {
            let key = (track_id, i);
            let data = self
                .keyframe_data
                .entry(key)
                .or_insert_with(|| CurveKeyframeData::new(track_id, i));

            let prev = if i > 0 { Some(times_values[i - 1]) } else { None };
            let next = if i + 1 < times_values.len() {
                Some(times_values[i + 1])
            } else {
                None
            };
            let current = times_values[i];

            if data.in_tangent_mode == TangentMode::Auto {
                let (dx, dy) = CurveKeyframeData::compute_auto_tangent(prev, current, next);
                data.in_handle = (-dx, -dy);
            }
            if data.out_tangent_mode == TangentMode::Auto {
                let (dx, dy) = CurveKeyframeData::compute_auto_tangent(prev, current, next);
                data.out_handle = (dx, dy);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TimelineSelection
// ---------------------------------------------------------------------------

/// A selected keyframe reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyframeRef {
    /// Track containing the keyframe.
    pub track_id: Uuid,
    /// Index of the keyframe within the track.
    pub index: usize,
}

/// Selection state for the timeline editor.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimelineSelection {
    /// Selected keyframes.
    pub keyframes: Vec<KeyframeRef>,
    /// Selected track IDs.
    pub tracks: Vec<Uuid>,
    /// Selected time range (start, end).
    pub time_range: Option<(f32, f32)>,
    /// Whether the selection is active.
    pub active: bool,
}

impl TimelineSelection {
    /// Create an empty selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select a single keyframe (replaces existing selection).
    pub fn select_keyframe(&mut self, track_id: Uuid, index: usize) {
        self.keyframes.clear();
        self.keyframes.push(KeyframeRef { track_id, index });
        self.active = true;
    }

    /// Add a keyframe to the selection (for multi-select).
    pub fn add_keyframe(&mut self, track_id: Uuid, index: usize) {
        let kf = KeyframeRef { track_id, index };
        if !self.keyframes.contains(&kf) {
            self.keyframes.push(kf);
        }
        self.active = true;
    }

    /// Remove a keyframe from the selection.
    pub fn deselect_keyframe(&mut self, track_id: Uuid, index: usize) {
        let kf = KeyframeRef { track_id, index };
        self.keyframes.retain(|k| k != &kf);
        if self.keyframes.is_empty() && self.tracks.is_empty() {
            self.active = false;
        }
    }

    /// Select a track.
    pub fn select_track(&mut self, track_id: Uuid) {
        self.tracks.clear();
        self.tracks.push(track_id);
        self.active = true;
    }

    /// Add a track to the selection.
    pub fn add_track(&mut self, track_id: Uuid) {
        if !self.tracks.contains(&track_id) {
            self.tracks.push(track_id);
        }
        self.active = true;
    }

    /// Set the selected time range.
    pub fn select_range(&mut self, start: f32, end: f32) {
        self.time_range = Some((start.min(end), start.max(end)));
        self.active = true;
    }

    /// Clear all selections.
    pub fn clear(&mut self) {
        self.keyframes.clear();
        self.tracks.clear();
        self.time_range = None;
        self.active = false;
    }

    /// Whether any keyframes are selected.
    pub fn has_keyframes(&self) -> bool {
        !self.keyframes.is_empty()
    }

    /// Whether any tracks are selected.
    pub fn has_tracks(&self) -> bool {
        !self.tracks.is_empty()
    }

    /// Number of selected keyframes.
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }
}

// ---------------------------------------------------------------------------
// Clipboard
// ---------------------------------------------------------------------------

/// Clipboard data for copy/paste operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineClipboard {
    /// Copied float keyframes: (relative_time, value).
    pub float_keyframes: Vec<(f32, f32)>,
    /// Copied tracks (full clones).
    pub tracks: Vec<TrackKind>,
    /// Time offset of the copied data (time of the first copied keyframe).
    pub base_time: f32,
}

impl TimelineClipboard {
    /// Create an empty clipboard.
    pub fn new() -> Self {
        Self {
            float_keyframes: Vec::new(),
            tracks: Vec::new(),
            base_time: 0.0,
        }
    }

    /// Whether the clipboard has content.
    pub fn has_content(&self) -> bool {
        !self.float_keyframes.is_empty() || !self.tracks.is_empty()
    }
}

impl Default for TimelineClipboard {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

/// The top-level timeline editor data model.
///
/// A timeline wraps a [`Sequence`] and layers on editor-specific state:
/// groups, markers, selection, snap settings, and a clipboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    /// The underlying sequence.
    pub sequence: Sequence,
    /// Named groups of tracks.
    pub groups: Vec<TimelineGroup>,
    /// Time markers.
    pub markers: Vec<Marker>,
    /// Snap settings.
    pub snap: SnapSettings,
    /// Current selection state.
    pub selection: TimelineSelection,
    /// Curve editor state.
    pub curve_editor: CurveEditor,
    /// Clipboard for copy/paste.
    #[serde(skip)]
    pub clipboard: TimelineClipboard,
    /// Current playhead position (seconds).
    pub playhead: f32,
    /// Visible time range in the editor (start, end).
    pub visible_range: (f32, f32),
    /// Zoom level (pixels per second).
    pub zoom: f32,
    /// Scroll offset in pixels.
    pub scroll_x: f32,
    /// Vertical scroll offset.
    pub scroll_y: f32,
    /// Whether the timeline is playing (preview mode).
    pub previewing: bool,
}

impl Timeline {
    /// Create a new timeline wrapping a sequence.
    pub fn new(sequence: Sequence) -> Self {
        let duration = sequence.duration;
        Self {
            sequence,
            groups: Vec::new(),
            markers: Vec::new(),
            snap: SnapSettings::default(),
            selection: TimelineSelection::new(),
            curve_editor: CurveEditor::new(),
            clipboard: TimelineClipboard::new(),
            playhead: 0.0,
            visible_range: (0.0, duration),
            zoom: 100.0,
            scroll_x: 0.0,
            scroll_y: 0.0,
            previewing: false,
        }
    }

    /// Add a group.
    pub fn add_group(&mut self, group: TimelineGroup) {
        self.groups.push(group);
    }

    /// Remove a group by name.
    pub fn remove_group(&mut self, name: &str) -> Option<TimelineGroup> {
        if let Some(pos) = self.groups.iter().position(|g| g.name == name) {
            Some(self.groups.remove(pos))
        } else {
            None
        }
    }

    /// Find a group by name.
    pub fn find_group(&self, name: &str) -> Option<&TimelineGroup> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// Find a group by name (mutable).
    pub fn find_group_mut(&mut self, name: &str) -> Option<&mut TimelineGroup> {
        self.groups.iter_mut().find(|g| g.name == name)
    }

    /// Add a marker.
    pub fn add_marker(&mut self, marker: Marker) {
        let pos = self
            .markers
            .binary_search_by(|m| m.time.partial_cmp(&marker.time).unwrap())
            .unwrap_or_else(|e| e);
        self.markers.insert(pos, marker);
    }

    /// Remove a marker by name.
    pub fn remove_marker(&mut self, name: &str) -> Option<Marker> {
        if let Some(pos) = self.markers.iter().position(|m| m.name == name) {
            Some(self.markers.remove(pos))
        } else {
            None
        }
    }

    /// Navigate to the next marker after the current playhead.
    pub fn next_marker(&self) -> Option<&Marker> {
        self.markers.iter().find(|m| m.time > self.playhead)
    }

    /// Navigate to the previous marker before the current playhead.
    pub fn prev_marker(&self) -> Option<&Marker> {
        self.markers.iter().rev().find(|m| m.time < self.playhead)
    }

    /// Snap a time value using the current snap settings.
    pub fn snap_time(&self, time: f32) -> f32 {
        self.snap.snap_time(time, &self.markers)
    }

    /// Set the playhead position (snapped).
    pub fn set_playhead(&mut self, time: f32) {
        self.playhead = self.snap_time(time).clamp(0.0, self.sequence.duration);
    }

    // -- Multi-track operations ---------------------------------------------

    /// Move all selected keyframes by a time offset.
    pub fn move_selected_keyframes(&mut self, time_offset: f32) {
        // Collect the track IDs and indices we need to modify.
        let refs: Vec<KeyframeRef> = self.selection.keyframes.clone();
        for kf_ref in &refs {
            if let Some(track) = self.sequence.find_track_mut(kf_ref.track_id) {
                match track {
                    TrackKind::Float(t) => {
                        if kf_ref.index < t.keyframes.len() {
                            t.keyframes[kf_ref.index].time += time_offset;
                        }
                    }
                    TrackKind::Transform(t) => {
                        if kf_ref.index < t.keyframes.len() {
                            t.keyframes[kf_ref.index].time += time_offset;
                        }
                    }
                    TrackKind::Color(t) => {
                        if kf_ref.index < t.keyframes.len() {
                            t.keyframes[kf_ref.index].time += time_offset;
                        }
                    }
                    TrackKind::Bool(t) => {
                        if kf_ref.index < t.keyframes.len() {
                            t.keyframes[kf_ref.index].time += time_offset;
                        }
                    }
                    TrackKind::Event(t) => {
                        if kf_ref.index < t.keyframes.len() {
                            t.keyframes[kf_ref.index].time += time_offset;
                        }
                    }
                    TrackKind::Camera(t) => {
                        if kf_ref.index < t.keyframes.len() {
                            t.keyframes[kf_ref.index].time += time_offset;
                        }
                    }
                    TrackKind::Audio(t) => {
                        if kf_ref.index < t.keyframes.len() {
                            t.keyframes[kf_ref.index].time += time_offset;
                        }
                    }
                    TrackKind::SubSequence(t) => {
                        if kf_ref.index < t.keyframes.len() {
                            t.keyframes[kf_ref.index].time += time_offset;
                        }
                    }
                }
            }
        }
    }

    /// Scale selected keyframes around a pivot time.
    pub fn scale_selected_keyframes(&mut self, factor: f32, pivot: f32) {
        let refs: Vec<KeyframeRef> = self.selection.keyframes.clone();
        for kf_ref in &refs {
            if let Some(track) = self.sequence.find_track_mut(kf_ref.track_id) {
                // Helper macro to avoid repeating the same pattern for every variant.
                macro_rules! scale_kf {
                    ($t:expr) => {
                        if kf_ref.index < $t.keyframes.len() {
                            let kf = &mut $t.keyframes[kf_ref.index];
                            kf.time = pivot + (kf.time - pivot) * factor;
                        }
                    };
                }
                match track {
                    TrackKind::Float(t) => scale_kf!(t),
                    TrackKind::Transform(t) => scale_kf!(t),
                    TrackKind::Color(t) => scale_kf!(t),
                    TrackKind::Bool(t) => scale_kf!(t),
                    TrackKind::Event(t) => scale_kf!(t),
                    TrackKind::Camera(t) => scale_kf!(t),
                    TrackKind::Audio(t) => scale_kf!(t),
                    TrackKind::SubSequence(t) => scale_kf!(t),
                }
            }
        }
    }

    /// Delete all selected keyframes.
    pub fn delete_selected_keyframes(&mut self) {
        // Sort by index descending so removals don't invalidate subsequent indices.
        let mut refs: Vec<KeyframeRef> = self.selection.keyframes.clone();
        refs.sort_by(|a, b| b.index.cmp(&a.index));

        for kf_ref in &refs {
            if let Some(track) = self.sequence.find_track_mut(kf_ref.track_id) {
                macro_rules! remove_kf {
                    ($t:expr) => {
                        if kf_ref.index < $t.keyframes.len() {
                            $t.keyframes.remove(kf_ref.index);
                        }
                    };
                }
                match track {
                    TrackKind::Float(t) => remove_kf!(t),
                    TrackKind::Transform(t) => remove_kf!(t),
                    TrackKind::Color(t) => remove_kf!(t),
                    TrackKind::Bool(t) => remove_kf!(t),
                    TrackKind::Event(t) => remove_kf!(t),
                    TrackKind::Camera(t) => remove_kf!(t),
                    TrackKind::Audio(t) => remove_kf!(t),
                    TrackKind::SubSequence(t) => remove_kf!(t),
                }
            }
        }
        self.selection.keyframes.clear();
    }

    /// Duplicate selected tracks and add them to the sequence.
    pub fn duplicate_selected_tracks(&mut self) {
        let track_ids: Vec<Uuid> = self.selection.tracks.clone();
        for tid in &track_ids {
            if let Some(track) = self.sequence.find_track(*tid) {
                let mut cloned = track.clone();
                // Assign a new UUID to the cloned track.
                match &mut cloned {
                    TrackKind::Float(t) => t.track_id = Uuid::new_v4(),
                    TrackKind::Transform(t) => t.track_id = Uuid::new_v4(),
                    TrackKind::Color(t) => t.track_id = Uuid::new_v4(),
                    TrackKind::Bool(t) => t.track_id = Uuid::new_v4(),
                    TrackKind::Event(t) => t.track_id = Uuid::new_v4(),
                    TrackKind::Camera(t) => t.track_id = Uuid::new_v4(),
                    TrackKind::Audio(t) => t.track_id = Uuid::new_v4(),
                    TrackKind::SubSequence(t) => t.track_id = Uuid::new_v4(),
                }
                self.sequence.tracks.push(cloned);
            }
        }
    }

    // -- Serialization ------------------------------------------------------

    /// Serialize the timeline to JSON.
    pub fn to_json(&self) -> Result<String, TimelineError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| TimelineError::SerializationError(e.to_string()))
    }

    /// Deserialize a timeline from JSON.
    pub fn from_json(json: &str) -> Result<Self, TimelineError> {
        serde_json::from_str(json).map_err(|e| TimelineError::SerializationError(e.to_string()))
    }
}

impl fmt::Display for Timeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Timeline(\"{}\", {:.2}s, {} groups, {} markers, {} tracks)",
            self.sequence.name,
            self.sequence.duration,
            self.groups.len(),
            self.markers.len(),
            self.sequence.tracks.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// CutsceneAsset
// ---------------------------------------------------------------------------

/// An aspect ratio for cutscene rendering.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AspectRatio {
    pub width: f32,
    pub height: f32,
}

impl AspectRatio {
    pub const RATIO_16_9: Self = Self {
        width: 16.0,
        height: 9.0,
    };
    pub const RATIO_21_9: Self = Self {
        width: 21.0,
        height: 9.0,
    };
    pub const RATIO_4_3: Self = Self {
        width: 4.0,
        height: 3.0,
    };
    pub const RATIO_1_1: Self = Self {
        width: 1.0,
        height: 1.0,
    };

    pub fn value(&self) -> f32 {
        self.width / self.height
    }
}

impl Default for AspectRatio {
    fn default() -> Self {
        Self::RATIO_16_9
    }
}

/// A subtitle trigger within a cutscene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleTrigger {
    /// Time in seconds when the subtitle appears.
    pub start_time: f32,
    /// Duration the subtitle is visible.
    pub duration: f32,
    /// Localization key for the subtitle text.
    pub text_key: String,
    /// Speaker name (localization key).
    pub speaker_key: String,
    /// Position on screen (0=bottom, 1=top).
    pub vertical_position: f32,
}

impl SubtitleTrigger {
    pub fn new(
        start_time: f32,
        duration: f32,
        text_key: impl Into<String>,
        speaker_key: impl Into<String>,
    ) -> Self {
        Self {
            start_time,
            duration,
            text_key: text_key.into(),
            speaker_key: speaker_key.into(),
            vertical_position: 0.1,
        }
    }

    /// End time of this subtitle.
    pub fn end_time(&self) -> f32 {
        self.start_time + self.duration
    }
}

/// A complete cutscene definition, serializable as a game asset.
///
/// Combines sequences, camera paths, audio cues, and subtitle triggers
/// into a single self-contained unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutsceneAsset {
    /// Unique identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// The main sequence timeline.
    pub timeline: Timeline,
    /// Camera paths used in this cutscene.
    pub camera_paths: Vec<CameraPath>,
    /// Audio cue overrides (supplementary to track audio).
    pub audio_cues: Vec<(f32, AudioCue)>,
    /// Subtitle triggers.
    pub subtitles: Vec<SubtitleTrigger>,
    /// Target aspect ratio.
    pub aspect_ratio: AspectRatio,
    /// Letterbox color (for aspect ratio adaptation).
    pub letterbox_color: (u8, u8, u8),
    /// Whether player input is blocked during this cutscene.
    pub block_input: bool,
    /// Whether the cutscene can be skipped.
    pub skippable: bool,
    /// Fade-in duration at the start (seconds).
    pub fade_in: f32,
    /// Fade-out duration at the end (seconds).
    pub fade_out: f32,
    /// Tags for categorization and filtering.
    pub tags: Vec<String>,
    /// Version number for asset migration.
    pub version: u32,
}

impl CutsceneAsset {
    /// Create a new cutscene asset.
    pub fn new(name: impl Into<String>, timeline: Timeline) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            timeline,
            camera_paths: Vec::new(),
            audio_cues: Vec::new(),
            subtitles: Vec::new(),
            aspect_ratio: AspectRatio::default(),
            letterbox_color: (0, 0, 0),
            block_input: true,
            skippable: true,
            fade_in: 0.5,
            fade_out: 0.5,
            tags: Vec::new(),
            version: 1,
        }
    }

    /// Total duration of the cutscene.
    pub fn duration(&self) -> f32 {
        self.timeline.sequence.duration
    }

    /// Add a camera path.
    pub fn add_camera_path(&mut self, path: CameraPath) {
        self.camera_paths.push(path);
    }

    /// Add a subtitle trigger.
    pub fn add_subtitle(&mut self, subtitle: SubtitleTrigger) {
        let pos = self
            .subtitles
            .binary_search_by(|s| {
                s.start_time
                    .partial_cmp(&subtitle.start_time)
                    .unwrap()
            })
            .unwrap_or_else(|e| e);
        self.subtitles.insert(pos, subtitle);
    }

    /// Collect subtitles active at a given time.
    pub fn active_subtitles(&self, time: f32) -> Vec<&SubtitleTrigger> {
        self.subtitles
            .iter()
            .filter(|s| time >= s.start_time && time <= s.end_time())
            .collect()
    }

    /// Adapt the cutscene to a different aspect ratio by computing
    /// letterbox dimensions.
    pub fn compute_letterbox(&self, screen_width: f32, screen_height: f32) -> (f32, f32) {
        let target = self.aspect_ratio.value();
        let screen = screen_width / screen_height;
        if (screen - target).abs() < 0.01 {
            return (0.0, 0.0);
        }
        if screen > target {
            // Screen is wider: pillarbox (vertical bars on sides).
            let content_width = screen_height * target;
            let bar = (screen_width - content_width) / 2.0;
            (bar, 0.0)
        } else {
            // Screen is taller: letterbox (horizontal bars top/bottom).
            let content_height = screen_width / target;
            let bar = (screen_height - content_height) / 2.0;
            (0.0, bar)
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, TimelineError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| TimelineError::SerializationError(e.to_string()))
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, TimelineError> {
        serde_json::from_str(json).map_err(|e| TimelineError::SerializationError(e.to_string()))
    }
}

impl fmt::Display for CutsceneAsset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CutsceneAsset(\"{}\" {:.2}s, {} paths, {} subtitles, v{})",
            self.name,
            self.duration(),
            self.camera_paths.len(),
            self.subtitles.len(),
            self.version,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequencer::{FloatTrack, Keyframe, Sequence};
    use genovo_ecs::Entity;

    #[test]
    fn timeline_groups() {
        let seq = Sequence::new("test", 10.0);
        let mut timeline = Timeline::new(seq);

        let mut group = TimelineGroup::new("Camera");
        let track_id = Uuid::new_v4();
        group.add_track(track_id);
        timeline.add_group(group);

        assert_eq!(timeline.groups.len(), 1);
        assert_eq!(timeline.groups[0].track_ids.len(), 1);
    }

    #[test]
    fn markers() {
        let seq = Sequence::new("test", 10.0);
        let mut timeline = Timeline::new(seq);

        timeline.add_marker(Marker::new("Impact", 3.0));
        timeline.add_marker(Marker::new("Cut", 5.0));
        timeline.add_marker(Marker::new("Fade", 8.0));

        assert_eq!(timeline.markers.len(), 3);
        assert_eq!(timeline.markers[0].name, "Impact");
        assert_eq!(timeline.markers[1].name, "Cut");
    }

    #[test]
    fn snap_to_frame() {
        let snap = SnapSettings {
            mode: SnapMode::Frame,
            frame_rate: 24.0,
            ..Default::default()
        };

        let snapped = snap.snap_time(0.123, &[]);
        let expected = (0.123 * 24.0_f32).round() / 24.0;
        assert!((snapped - expected).abs() < 1e-6);
    }

    #[test]
    fn snap_to_beat() {
        let snap = SnapSettings {
            mode: SnapMode::Beat,
            bpm: 120.0,
            subdivisions: 4,
            ..Default::default()
        };

        // At 120 BPM, one beat = 0.5s, one subdivision = 0.125s
        let snapped = snap.snap_time(0.13, &[]);
        assert!((snapped - 0.125).abs() < 1e-4);
    }

    #[test]
    fn snap_to_marker() {
        let markers = vec![
            Marker::new("A", 1.0),
            Marker::new("B", 3.0),
        ];
        let snap = SnapSettings {
            mode: SnapMode::Marker,
            threshold: 0.1,
            ..Default::default()
        };

        let snapped = snap.snap_time(1.05, &markers);
        assert!((snapped - 1.0).abs() < 1e-6);
    }

    #[test]
    fn selection_operations() {
        let mut sel = TimelineSelection::new();
        let track_id = Uuid::new_v4();

        sel.select_keyframe(track_id, 0);
        assert_eq!(sel.keyframe_count(), 1);

        sel.add_keyframe(track_id, 1);
        assert_eq!(sel.keyframe_count(), 2);

        sel.deselect_keyframe(track_id, 0);
        assert_eq!(sel.keyframe_count(), 1);

        sel.clear();
        assert!(!sel.active);
    }

    #[test]
    fn auto_tangent_computation() {
        let tangent = CurveKeyframeData::compute_auto_tangent(
            Some((0.0, 0.0)),
            (1.0, 1.0),
            Some((2.0, 2.0)),
        );
        // Slope is 1.0, handle length should be ~0.333
        assert!(tangent.0 > 0.0);
        assert!((tangent.1 / tangent.0 - 1.0).abs() < 1e-4, "slope should be ~1.0");
    }

    #[test]
    fn cutscene_letterbox_16_9() {
        let seq = Sequence::new("test", 5.0);
        let timeline = Timeline::new(seq);
        let cutscene = CutsceneAsset::new("intro", timeline);

        // 16:9 screen should have no letterbox.
        let (h, v) = cutscene.compute_letterbox(1920.0, 1080.0);
        assert!(h.abs() < 1.0 && v.abs() < 1.0);
    }

    #[test]
    fn cutscene_letterbox_ultrawide() {
        let seq = Sequence::new("test", 5.0);
        let timeline = Timeline::new(seq);
        let mut cutscene = CutsceneAsset::new("intro", timeline);
        cutscene.aspect_ratio = AspectRatio::RATIO_16_9;

        // 21:9 screen should produce pillarbox (horizontal bars).
        let (h, _v) = cutscene.compute_letterbox(2560.0, 1080.0);
        assert!(h > 0.0, "Expected pillarbox bars on ultrawide");
    }

    #[test]
    fn subtitles_active() {
        let seq = Sequence::new("test", 10.0);
        let timeline = Timeline::new(seq);
        let mut cutscene = CutsceneAsset::new("intro", timeline);

        cutscene.add_subtitle(SubtitleTrigger::new(1.0, 2.0, "sub_1", "speaker_1"));
        cutscene.add_subtitle(SubtitleTrigger::new(4.0, 3.0, "sub_2", "speaker_2"));

        assert_eq!(cutscene.active_subtitles(1.5).len(), 1);
        assert_eq!(cutscene.active_subtitles(5.0).len(), 1);
        assert_eq!(cutscene.active_subtitles(3.5).len(), 0);
    }
}
