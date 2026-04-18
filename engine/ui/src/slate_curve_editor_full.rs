//! Full curve editor for the Genovo Slate UI, inspired by Unreal Engine.
//!
//! Provides:
//! - Canvas with zoom/pan (scroll + middle mouse)
//! - Grid lines with value labels
//! - Keyframes as draggable points
//! - Bezier tangent handles (in/out)
//! - Tangent modes: Auto, Linear, Step, Free, Weighted
//! - Auto-tangent computation (Catmull-Rom)
//! - Multi-select keyframes (box select, Ctrl+click)
//! - Delete/copy/paste keyframes
//! - Snap to grid
//! - Multiple curves overlay (different colours)
//! - Value/time readout at cursor
//! - Right-click context menu integration

use std::collections::HashMap;

use glam::Vec2;
use genovo_core::Rect;

use crate::core::{KeyCode, KeyModifiers, MouseButton, Padding, UIEvent, UIId};
use crate::render_commands::{
    Border as BorderSpec, Color, CornerRadii, DrawCommand, DrawList, TextAlign,
    TextVerticalAlign, TextureId,
};
use crate::slate_widgets::EventReply;

// =========================================================================
// Constants
// =========================================================================

const KEY_RADIUS: f32 = 5.0;
const KEY_HIT_RADIUS: f32 = 8.0;
const TANGENT_HANDLE_RADIUS: f32 = 4.0;
const TANGENT_LINE_LENGTH: f32 = 50.0;
const TANGENT_HIT_RADIUS: f32 = 7.0;
const GRID_MIN_SPACING: f32 = 40.0;
const LABEL_MARGIN: f32 = 4.0;
const MARGIN_LEFT: f32 = 50.0;
const MARGIN_BOTTOM: f32 = 24.0;
const MARGIN_TOP: f32 = 8.0;
const MARGIN_RIGHT: f32 = 8.0;
const READOUT_HEIGHT: f32 = 18.0;
const BOX_SELECT_MIN: f32 = 4.0;
const ZOOM_SPEED: f32 = 0.1;
const PAN_SPEED: f32 = 1.0;
const SNAP_THRESHOLD: f32 = 8.0;
const CURVE_SEGMENTS: usize = 64;
const MAX_UNDO_DEPTH: usize = 64;

// =========================================================================
// CurveKeyTangentMode
// =========================================================================

/// Tangent mode for a keyframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CurveKeyTangentMode {
    /// Catmull-Rom style auto-tangent.
    Auto,
    /// Flat tangent (zero slope).
    Flat,
    /// Linear interpolation.
    Linear,
    /// Step/constant (hold value until next key).
    Step,
    /// User-defined tangent angles.
    Free,
    /// User-defined tangent angles + weights.
    Weighted,
}

impl Default for CurveKeyTangentMode {
    fn default() -> Self {
        CurveKeyTangentMode::Auto
    }
}

// =========================================================================
// CurveKeyframe
// =========================================================================

/// A single keyframe on a curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullCurveKeyframe {
    /// Time position.
    pub time: f32,
    /// Value at this keyframe.
    pub value: f32,
    /// Tangent mode.
    pub tangent_mode: CurveKeyTangentMode,
    /// Incoming tangent angle (radians, for Free/Weighted modes).
    pub tangent_in: f32,
    /// Outgoing tangent angle (radians).
    pub tangent_out: f32,
    /// Incoming tangent weight (for Weighted mode).
    pub weight_in: f32,
    /// Outgoing tangent weight.
    pub weight_out: f32,
    /// Whether in/out tangents are linked (broken = independent).
    pub tangent_broken: bool,
}

impl FullCurveKeyframe {
    pub fn new(time: f32, value: f32) -> Self {
        Self {
            time,
            value,
            tangent_mode: CurveKeyTangentMode::Auto,
            tangent_in: 0.0,
            tangent_out: 0.0,
            weight_in: 1.0 / 3.0,
            weight_out: 1.0 / 3.0,
            tangent_broken: false,
        }
    }

    pub fn with_tangent_mode(mut self, mode: CurveKeyTangentMode) -> Self {
        self.tangent_mode = mode;
        self
    }

    pub fn with_tangents(mut self, tan_in: f32, tan_out: f32) -> Self {
        self.tangent_in = tan_in;
        self.tangent_out = tan_out;
        self
    }

    pub fn with_weights(mut self, w_in: f32, w_out: f32) -> Self {
        self.weight_in = w_in;
        self.weight_out = w_out;
        self
    }

    /// Set both tangents to the same value (linked).
    pub fn set_tangent(&mut self, tangent: f32) {
        self.tangent_in = tangent;
        self.tangent_out = tangent;
    }
}

impl Default for FullCurveKeyframe {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

// =========================================================================
// CurveData
// =========================================================================

/// A single curve (sequence of keyframes).
#[derive(Debug, Clone)]
pub struct CurveData {
    /// Unique identifier for this curve.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Curve colour.
    pub color: Color,
    /// Keyframes (sorted by time).
    pub keyframes: Vec<FullCurveKeyframe>,
    /// Whether this curve is visible.
    pub visible: bool,
    /// Whether this curve is locked (not editable).
    pub locked: bool,
    /// Line thickness.
    pub thickness: f32,
}

impl CurveData {
    pub fn new(id: u32, name: &str, color: Color) -> Self {
        Self {
            id,
            name: name.to_string(),
            color,
            keyframes: Vec::new(),
            visible: true,
            locked: false,
            thickness: 2.0,
        }
    }

    /// Add a keyframe, keeping sorted by time.
    pub fn add_key(&mut self, key: FullCurveKeyframe) {
        let pos = self.keyframes.partition_point(|k| k.time < key.time);
        self.keyframes.insert(pos, key);
        self.compute_auto_tangents();
    }

    /// Remove a keyframe by index.
    pub fn remove_key(&mut self, index: usize) -> Option<FullCurveKeyframe> {
        if index < self.keyframes.len() {
            let key = self.keyframes.remove(index);
            self.compute_auto_tangents();
            Some(key)
        } else {
            None
        }
    }

    /// Move a keyframe to a new time/value.
    pub fn move_key(&mut self, index: usize, time: f32, value: f32) {
        if index < self.keyframes.len() {
            self.keyframes[index].time = time;
            self.keyframes[index].value = value;
            // Re-sort.
            self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
            self.compute_auto_tangents();
        }
    }

    /// Evaluate the curve at a given time.
    pub fn evaluate(&self, time: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].value;
        }

        // Before first key.
        if time <= self.keyframes[0].time {
            return self.keyframes[0].value;
        }
        // After last key.
        let last = self.keyframes.len() - 1;
        if time >= self.keyframes[last].time {
            return self.keyframes[last].value;
        }

        // Find the two surrounding keyframes.
        let idx = self.keyframes.partition_point(|k| k.time <= time);
        let k0 = &self.keyframes[idx - 1];
        let k1 = &self.keyframes[idx];

        match k0.tangent_mode {
            CurveKeyTangentMode::Step => k0.value,
            CurveKeyTangentMode::Linear => {
                let t = (time - k0.time) / (k1.time - k0.time);
                k0.value + t * (k1.value - k0.value)
            }
            _ => {
                // Hermite interpolation.
                let dt = k1.time - k0.time;
                if dt < 1e-6 {
                    return k0.value;
                }
                let t = (time - k0.time) / dt;
                let t2 = t * t;
                let t3 = t2 * t;

                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + t;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;

                let m0 = k0.tangent_out * dt;
                let m1 = k1.tangent_in * dt;

                h00 * k0.value + h10 * m0 + h01 * k1.value + h11 * m1
            }
        }
    }

    /// Compute auto-tangents (Catmull-Rom) for all keys with Auto mode.
    pub fn compute_auto_tangents(&mut self) {
        let count = self.keyframes.len();
        if count < 2 {
            return;
        }

        // Collect the tangent values we need to set.
        let mut tangents: Vec<(usize, f32)> = Vec::new();

        for i in 0..count {
            if self.keyframes[i].tangent_mode != CurveKeyTangentMode::Auto
                && self.keyframes[i].tangent_mode != CurveKeyTangentMode::Flat
            {
                continue;
            }

            if self.keyframes[i].tangent_mode == CurveKeyTangentMode::Flat {
                tangents.push((i, 0.0));
                continue;
            }

            // Catmull-Rom tangent.
            let tangent = if i == 0 {
                let k0 = &self.keyframes[0];
                let k1 = &self.keyframes[1];
                let dt = k1.time - k0.time;
                if dt.abs() > 1e-6 {
                    (k1.value - k0.value) / dt
                } else {
                    0.0
                }
            } else if i == count - 1 {
                let k0 = &self.keyframes[count - 2];
                let k1 = &self.keyframes[count - 1];
                let dt = k1.time - k0.time;
                if dt.abs() > 1e-6 {
                    (k1.value - k0.value) / dt
                } else {
                    0.0
                }
            } else {
                let k_prev = &self.keyframes[i - 1];
                let k_next = &self.keyframes[i + 1];
                let dt = k_next.time - k_prev.time;
                if dt.abs() > 1e-6 {
                    (k_next.value - k_prev.value) / dt
                } else {
                    0.0
                }
            };

            tangents.push((i, tangent));
        }

        // Apply tangents.
        for (i, tangent) in tangents {
            self.keyframes[i].tangent_in = tangent;
            self.keyframes[i].tangent_out = tangent;
        }
    }

    /// Time range of the curve.
    pub fn time_range(&self) -> (f32, f32) {
        if self.keyframes.is_empty() {
            return (0.0, 1.0);
        }
        (
            self.keyframes.first().unwrap().time,
            self.keyframes.last().unwrap().time,
        )
    }

    /// Value range of the curve.
    pub fn value_range(&self) -> (f32, f32) {
        if self.keyframes.is_empty() {
            return (0.0, 1.0);
        }
        let min = self.keyframes.iter().map(|k| k.value).fold(f32::MAX, f32::min);
        let max = self.keyframes.iter().map(|k| k.value).fold(f32::MIN, f32::max);
        if (max - min).abs() < 1e-6 {
            (min - 0.5, max + 0.5)
        } else {
            (min, max)
        }
    }
}

// =========================================================================
// KeySelection
// =========================================================================

/// Identifies a selected keyframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyRef {
    pub curve_id: u32,
    pub key_index: usize,
}

/// Selection state for the curve editor.
#[derive(Debug, Clone)]
pub struct CurveSelection {
    pub selected: Vec<KeyRef>,
    pub clipboard: Vec<(u32, FullCurveKeyframe)>,
}

impl CurveSelection {
    pub fn new() -> Self {
        Self {
            selected: Vec::new(),
            clipboard: Vec::new(),
        }
    }

    pub fn is_selected(&self, curve_id: u32, key_index: usize) -> bool {
        self.selected.iter().any(|kr| kr.curve_id == curve_id && kr.key_index == key_index)
    }

    pub fn select(&mut self, curve_id: u32, key_index: usize, ctrl: bool) {
        let kr = KeyRef { curve_id, key_index };
        if ctrl {
            if let Some(pos) = self.selected.iter().position(|k| *k == kr) {
                self.selected.remove(pos);
            } else {
                self.selected.push(kr);
            }
        } else {
            self.selected.clear();
            self.selected.push(kr);
        }
    }

    pub fn clear(&mut self) {
        self.selected.clear();
    }

    pub fn has_selection(&self) -> bool {
        !self.selected.is_empty()
    }

    pub fn select_in_rect(&mut self, keys: &[(u32, usize, Vec2)], rect: Rect, ctrl: bool) {
        if !ctrl {
            self.selected.clear();
        }
        for &(curve_id, key_index, pos) in keys {
            if rect.contains(pos) {
                let kr = KeyRef { curve_id, key_index };
                if !self.selected.contains(&kr) {
                    self.selected.push(kr);
                }
            }
        }
    }
}

impl Default for CurveSelection {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// CurveEditorStyle
// =========================================================================

/// Visual style for the curve editor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveEditorStyle {
    /// Background colour.
    pub background: Color,
    /// Grid line colour (major).
    pub grid_major: Color,
    /// Grid line colour (minor).
    pub grid_minor: Color,
    /// Grid label colour.
    pub label_color: Color,
    /// Axis colour.
    pub axis_color: Color,
    /// Selection box colour.
    pub selection_box: Color,
    /// Selection box border.
    pub selection_border: Color,
    /// Key normal colour.
    pub key_normal: Color,
    /// Key selected colour.
    pub key_selected: Color,
    /// Key hovered colour.
    pub key_hovered: Color,
    /// Tangent handle colour.
    pub tangent_color: Color,
    /// Tangent line colour.
    pub tangent_line: Color,
    /// Readout background.
    pub readout_bg: Color,
    /// Readout text.
    pub readout_text: Color,
    /// Border colour.
    pub border_color: Color,
    /// Font size.
    pub font_size: f32,
    /// Font ID.
    pub font_id: u32,
}

impl Default for CurveEditorStyle {
    fn default() -> Self {
        Self {
            background: Color::from_hex("#1A1A1A"),
            grid_major: Color::new(0.3, 0.3, 0.3, 0.5),
            grid_minor: Color::new(0.2, 0.2, 0.2, 0.3),
            label_color: Color::from_hex("#888888"),
            axis_color: Color::new(0.5, 0.5, 0.5, 0.8),
            selection_box: Color::new(0.2, 0.4, 0.8, 0.2),
            selection_border: Color::new(0.3, 0.5, 0.9, 0.8),
            key_normal: Color::from_hex("#CCCCCC"),
            key_selected: Color::from_hex("#FF8800"),
            key_hovered: Color::from_hex("#FFFFFF"),
            tangent_color: Color::from_hex("#FFCC00"),
            tangent_line: Color::new(0.8, 0.6, 0.0, 0.6),
            readout_bg: Color::new(0.0, 0.0, 0.0, 0.7),
            readout_text: Color::from_hex("#CCCCCC"),
            border_color: Color::from_hex("#3F3F46"),
            font_size: 11.0,
            font_id: 0,
        }
    }
}

// =========================================================================
// ViewTransform
// =========================================================================

/// Handles the mapping between graph-space (time/value) and screen-space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewTransform {
    /// Visible time range (min, max).
    pub time_min: f32,
    pub time_max: f32,
    /// Visible value range (min, max).
    pub value_min: f32,
    pub value_max: f32,
}

impl ViewTransform {
    pub fn new(time_min: f32, time_max: f32, value_min: f32, value_max: f32) -> Self {
        Self {
            time_min,
            time_max,
            value_min,
            value_max,
        }
    }

    /// Map graph-space (time, value) to screen-space pixel position.
    pub fn to_screen(&self, time: f32, value: f32, canvas: Rect) -> Vec2 {
        let x = canvas.min.x
            + (time - self.time_min) / (self.time_max - self.time_min) * canvas.width();
        let y = canvas.max.y
            - (value - self.value_min) / (self.value_max - self.value_min) * canvas.height();
        Vec2::new(x, y)
    }

    /// Map screen-space pixel position to graph-space (time, value).
    pub fn to_graph(&self, screen: Vec2, canvas: Rect) -> (f32, f32) {
        let t = self.time_min
            + (screen.x - canvas.min.x) / canvas.width() * (self.time_max - self.time_min);
        let v = self.value_min
            + (canvas.max.y - screen.y) / canvas.height() * (self.value_max - self.value_min);
        (t, v)
    }

    /// Zoom centered on a screen position.
    pub fn zoom(&mut self, center: Vec2, factor: f32, canvas: Rect) {
        let (ct, cv) = self.to_graph(center, canvas);
        let time_range = self.time_max - self.time_min;
        let value_range = self.value_max - self.value_min;

        let new_time_range = time_range * factor;
        let new_value_range = value_range * factor;

        let t_ratio = (ct - self.time_min) / time_range;
        let v_ratio = (cv - self.value_min) / value_range;

        self.time_min = ct - t_ratio * new_time_range;
        self.time_max = ct + (1.0 - t_ratio) * new_time_range;
        self.value_min = cv - v_ratio * new_value_range;
        self.value_max = cv + (1.0 - v_ratio) * new_value_range;
    }

    /// Pan by a screen-space delta.
    pub fn pan(&mut self, delta: Vec2, canvas: Rect) {
        let time_range = self.time_max - self.time_min;
        let value_range = self.value_max - self.value_min;

        let dt = -delta.x / canvas.width() * time_range;
        let dv = delta.y / canvas.height() * value_range;

        self.time_min += dt;
        self.time_max += dt;
        self.value_min += dv;
        self.value_max += dv;
    }

    /// Fit the view to show all curves.
    pub fn fit_to_curves(&mut self, curves: &[CurveData], margin: f32) {
        if curves.is_empty() {
            *self = ViewTransform::default();
            return;
        }

        let mut t_min = f32::MAX;
        let mut t_max = f32::MIN;
        let mut v_min = f32::MAX;
        let mut v_max = f32::MIN;

        for curve in curves {
            if !curve.visible || curve.keyframes.is_empty() {
                continue;
            }
            for k in &curve.keyframes {
                t_min = t_min.min(k.time);
                t_max = t_max.max(k.time);
                v_min = v_min.min(k.value);
                v_max = v_max.max(k.value);
            }
        }

        if t_min >= t_max {
            t_min -= 0.5;
            t_max += 0.5;
        }
        if v_min >= v_max {
            v_min -= 0.5;
            v_max += 0.5;
        }

        let t_range = t_max - t_min;
        let v_range = v_max - v_min;
        self.time_min = t_min - t_range * margin;
        self.time_max = t_max + t_range * margin;
        self.value_min = v_min - v_range * margin;
        self.value_max = v_max + v_range * margin;
    }

    /// Pixels per time unit.
    pub fn time_scale(&self, canvas: Rect) -> f32 {
        canvas.width() / (self.time_max - self.time_min).max(1e-6)
    }

    /// Pixels per value unit.
    pub fn value_scale(&self, canvas: Rect) -> f32 {
        canvas.height() / (self.value_max - self.value_min).max(1e-6)
    }
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self {
            time_min: -0.5,
            time_max: 5.0,
            value_min: -0.5,
            value_max: 1.5,
        }
    }
}

// =========================================================================
// InteractionMode
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum InteractionMode {
    None,
    Pan { start: Vec2, start_view: ViewTransform },
    DragKey { curve_id: u32, key_index: usize, start_time: f32, start_value: f32 },
    DragTangentIn { curve_id: u32, key_index: usize },
    DragTangentOut { curve_id: u32, key_index: usize },
    BoxSelect { start: Vec2 },
}

// =========================================================================
// FullCurveEditor
// =========================================================================

/// A full-featured curve editor widget.
#[derive(Debug, Clone)]
pub struct FullCurveEditor {
    /// Widget ID.
    pub id: UIId,
    /// Curves being edited.
    pub curves: Vec<CurveData>,
    /// View transform.
    pub view: ViewTransform,
    /// Selection state.
    pub selection: CurveSelection,
    /// Style.
    pub style: CurveEditorStyle,

    // --- Interaction ---
    interaction: InteractionMode,
    /// Currently hovered key.
    hovered_key: Option<KeyRef>,
    /// Whether a tangent handle is hovered (true=in, false=out).
    hovered_tangent: Option<(KeyRef, bool)>,
    /// Current mouse position in screen space.
    mouse_pos: Vec2,
    /// Cursor position in graph space.
    pub cursor_graph: (f32, f32),
    /// Whether to show the cursor readout.
    pub show_readout: bool,

    // --- Options ---
    /// Whether snap to grid is enabled.
    pub snap_enabled: bool,
    /// Grid snap value for time.
    pub snap_time: f32,
    /// Grid snap value for values.
    pub snap_value: f32,
    /// Whether to show tangent handles.
    pub show_tangents: bool,
    /// Whether the editor is enabled.
    pub enabled: bool,
    /// Whether the editor is visible.
    pub visible: bool,
    /// Whether a modification occurred this frame.
    pub modified: bool,

    // --- Undo ---
    undo_stack: Vec<Vec<CurveData>>,
    redo_stack: Vec<Vec<CurveData>>,
}

impl FullCurveEditor {
    /// Create a new curve editor.
    pub fn new() -> Self {
        Self {
            id: UIId::INVALID,
            curves: Vec::new(),
            view: ViewTransform::default(),
            selection: CurveSelection::new(),
            style: CurveEditorStyle::default(),
            interaction: InteractionMode::None,
            hovered_key: None,
            hovered_tangent: None,
            mouse_pos: Vec2::ZERO,
            cursor_graph: (0.0, 0.0),
            show_readout: true,
            snap_enabled: false,
            snap_time: 1.0 / 30.0,
            snap_value: 0.1,
            show_tangents: true,
            enabled: true,
            visible: true,
            modified: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    /// Add a curve.
    pub fn add_curve(&mut self, curve: CurveData) {
        self.curves.push(curve);
    }

    /// Remove a curve by ID.
    pub fn remove_curve(&mut self, id: u32) {
        self.curves.retain(|c| c.id != id);
        self.selection.selected.retain(|kr| kr.curve_id != id);
    }

    /// Get a curve by ID.
    pub fn get_curve(&self, id: u32) -> Option<&CurveData> {
        self.curves.iter().find(|c| c.id == id)
    }

    /// Get a mutable curve by ID.
    pub fn get_curve_mut(&mut self, id: u32) -> Option<&mut CurveData> {
        self.curves.iter_mut().find(|c| c.id == id)
    }

    /// Fit the view to all curves.
    pub fn fit_to_content(&mut self) {
        self.view.fit_to_curves(&self.curves, 0.1);
    }

    /// Canvas rect (the actual drawing area excluding margins).
    fn canvas_rect(&self, rect: Rect) -> Rect {
        Rect::new(
            Vec2::new(rect.min.x + MARGIN_LEFT, rect.min.y + MARGIN_TOP),
            Vec2::new(rect.max.x - MARGIN_RIGHT, rect.max.y - MARGIN_BOTTOM),
        )
    }

    /// Snap a value to the grid.
    fn snap_to_grid(&self, time: f32, value: f32) -> (f32, f32) {
        if !self.snap_enabled {
            return (time, value);
        }
        let t = (time / self.snap_time).round() * self.snap_time;
        let v = (value / self.snap_value).round() * self.snap_value;
        (t, v)
    }

    /// Push current state to undo stack.
    fn push_undo(&mut self) {
        self.undo_stack.push(self.curves.clone());
        if self.undo_stack.len() > MAX_UNDO_DEPTH {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    /// Undo.
    pub fn undo(&mut self) {
        if let Some(state) = self.undo_stack.pop() {
            self.redo_stack.push(self.curves.clone());
            self.curves = state;
        }
    }

    /// Redo.
    pub fn redo(&mut self) {
        if let Some(state) = self.redo_stack.pop() {
            self.undo_stack.push(self.curves.clone());
            self.curves = state;
        }
    }

    /// Delete selected keyframes.
    pub fn delete_selected(&mut self) {
        if self.selection.selected.is_empty() {
            return;
        }
        self.push_undo();

        // Sort descending by key_index to avoid index shifting issues.
        let mut to_delete = self.selection.selected.clone();
        to_delete.sort_by(|a, b| {
            if a.curve_id == b.curve_id {
                b.key_index.cmp(&a.key_index)
            } else {
                a.curve_id.cmp(&b.curve_id)
            }
        });

        for kr in &to_delete {
            if let Some(curve) = self.curves.iter_mut().find(|c| c.id == kr.curve_id) {
                if kr.key_index < curve.keyframes.len() {
                    curve.keyframes.remove(kr.key_index);
                    curve.compute_auto_tangents();
                }
            }
        }

        self.selection.clear();
        self.modified = true;
    }

    /// Copy selected keyframes.
    pub fn copy_selected(&mut self) {
        self.selection.clipboard.clear();
        for kr in &self.selection.selected {
            if let Some(curve) = self.curves.iter().find(|c| c.id == kr.curve_id) {
                if kr.key_index < curve.keyframes.len() {
                    self.selection
                        .clipboard
                        .push((kr.curve_id, curve.keyframes[kr.key_index]));
                }
            }
        }
    }

    /// Paste keyframes at the current cursor position.
    pub fn paste(&mut self) {
        if self.selection.clipboard.is_empty() {
            return;
        }
        self.push_undo();

        let (cursor_time, _) = self.cursor_graph;
        let min_time = self
            .selection
            .clipboard
            .iter()
            .map(|(_, k)| k.time)
            .fold(f32::MAX, f32::min);
        let offset = cursor_time - min_time;

        self.selection.selected.clear();

        for (curve_id, key) in self.selection.clipboard.clone() {
            if let Some(curve) = self.curves.iter_mut().find(|c| c.id == curve_id) {
                let mut new_key = key;
                new_key.time += offset;
                let idx = curve.keyframes.partition_point(|k| k.time < new_key.time);
                curve.keyframes.insert(idx, new_key);
                curve.compute_auto_tangents();
                self.selection.selected.push(KeyRef {
                    curve_id,
                    key_index: idx,
                });
            }
        }

        self.modified = true;
    }

    /// Add a key to a curve at the given graph-space position.
    pub fn add_key_at(&mut self, curve_id: u32, time: f32, value: f32) {
        self.push_undo();
        if let Some(curve) = self.curves.iter_mut().find(|c| c.id == curve_id) {
            let key = FullCurveKeyframe::new(time, value);
            curve.add_key(key);
            self.modified = true;
        }
    }

    /// Set tangent mode for all selected keys.
    pub fn set_tangent_mode(&mut self, mode: CurveKeyTangentMode) {
        self.push_undo();
        for kr in &self.selection.selected {
            if let Some(curve) = self.curves.iter_mut().find(|c| c.id == kr.curve_id) {
                if kr.key_index < curve.keyframes.len() {
                    curve.keyframes[kr.key_index].tangent_mode = mode;
                }
            }
        }
        for curve in &mut self.curves {
            curve.compute_auto_tangents();
        }
        self.modified = true;
    }

    // =====================================================================
    // All key screen positions (for box selection).
    // =====================================================================

    fn all_key_positions(&self, canvas: Rect) -> Vec<(u32, usize, Vec2)> {
        let mut positions = Vec::new();
        for curve in &self.curves {
            if !curve.visible {
                continue;
            }
            for (ki, key) in curve.keyframes.iter().enumerate() {
                let pos = self.view.to_screen(key.time, key.value, canvas);
                positions.push((curve.id, ki, pos));
            }
        }
        positions
    }

    // =====================================================================
    // Compute grid spacing.
    // =====================================================================

    fn grid_spacing(range: f32, canvas_size: f32) -> f32 {
        let ideal = range * GRID_MIN_SPACING / canvas_size.max(1.0);
        let pow = 10.0f32.powf(ideal.log10().floor());
        let norm = ideal / pow;
        let nice = if norm <= 1.0 {
            1.0
        } else if norm <= 2.0 {
            2.0
        } else if norm <= 5.0 {
            5.0
        } else {
            10.0
        };
        nice * pow
    }

    // =====================================================================
    // Paint
    // =====================================================================

    /// Paint the curve editor.
    pub fn paint(&self, rect: Rect, draw: &mut DrawList) {
        if !self.visible {
            return;
        }

        let canvas = self.canvas_rect(rect);

        // Background.
        draw.commands.push(DrawCommand::Rect {
            rect,
            color: self.style.background,
            corner_radii: CornerRadii::all(2.0),
            border: BorderSpec::new(self.style.border_color, 1.0),
            shadow: None,
        });

        draw.commands.push(DrawCommand::PushClip { rect });

        self.paint_grid(canvas, draw);
        self.paint_axes(canvas, draw);
        self.paint_curves(canvas, draw);
        self.paint_keys(canvas, draw);
        self.paint_box_select(draw);
        self.paint_readout(rect, canvas, draw);

        draw.commands.push(DrawCommand::PopClip);
    }

    fn paint_grid(&self, canvas: Rect, draw: &mut DrawList) {
        let time_range = self.view.time_max - self.view.time_min;
        let value_range = self.view.value_max - self.view.value_min;

        let time_spacing = Self::grid_spacing(time_range, canvas.width());
        let value_spacing = Self::grid_spacing(value_range, canvas.height());

        // Vertical grid lines (time).
        let t_start = (self.view.time_min / time_spacing).floor() * time_spacing;
        let mut t = t_start;
        while t <= self.view.time_max {
            let x = canvas.min.x
                + (t - self.view.time_min) / time_range * canvas.width();
            if x >= canvas.min.x && x <= canvas.max.x {
                let is_major = (t / (time_spacing * 5.0)).fract().abs() < 0.01;
                draw.commands.push(DrawCommand::Line {
                    start: Vec2::new(x, canvas.min.y),
                    end: Vec2::new(x, canvas.max.y),
                    color: if is_major {
                        self.style.grid_major
                    } else {
                        self.style.grid_minor
                    },
                    thickness: if is_major { 1.0 } else { 0.5 },
                });
                // Label.
                draw.commands.push(DrawCommand::Text {
                    text: format!("{:.1}", t),
                    position: Vec2::new(x + 2.0, canvas.max.y + 4.0),
                    font_size: self.style.font_size,
                    color: self.style.label_color,
                    font_id: self.style.font_id,
                    max_width: None,
                    align: TextAlign::Left,
                    vertical_align: TextVerticalAlign::Top,
                });
            }
            t += time_spacing;
        }

        // Horizontal grid lines (value).
        let v_start = (self.view.value_min / value_spacing).floor() * value_spacing;
        let mut v = v_start;
        while v <= self.view.value_max {
            let y = canvas.max.y
                - (v - self.view.value_min) / value_range * canvas.height();
            if y >= canvas.min.y && y <= canvas.max.y {
                let is_major = (v / (value_spacing * 5.0)).fract().abs() < 0.01;
                draw.commands.push(DrawCommand::Line {
                    start: Vec2::new(canvas.min.x, y),
                    end: Vec2::new(canvas.max.x, y),
                    color: if is_major {
                        self.style.grid_major
                    } else {
                        self.style.grid_minor
                    },
                    thickness: if is_major { 1.0 } else { 0.5 },
                });
                // Label.
                draw.commands.push(DrawCommand::Text {
                    text: format!("{:.2}", v),
                    position: Vec2::new(canvas.min.x - MARGIN_LEFT + LABEL_MARGIN, y - 6.0),
                    font_size: self.style.font_size,
                    color: self.style.label_color,
                    font_id: self.style.font_id,
                    max_width: Some(MARGIN_LEFT - LABEL_MARGIN * 2.0),
                    align: TextAlign::Right,
                    vertical_align: TextVerticalAlign::Top,
                });
            }
            v += value_spacing;
        }
    }

    fn paint_axes(&self, canvas: Rect, draw: &mut DrawList) {
        // Zero lines if visible.
        let zero_screen = self.view.to_screen(0.0, 0.0, canvas);

        if zero_screen.x >= canvas.min.x && zero_screen.x <= canvas.max.x {
            draw.commands.push(DrawCommand::Line {
                start: Vec2::new(zero_screen.x, canvas.min.y),
                end: Vec2::new(zero_screen.x, canvas.max.y),
                color: self.style.axis_color,
                thickness: 1.0,
            });
        }

        if zero_screen.y >= canvas.min.y && zero_screen.y <= canvas.max.y {
            draw.commands.push(DrawCommand::Line {
                start: Vec2::new(canvas.min.x, zero_screen.y),
                end: Vec2::new(canvas.max.x, zero_screen.y),
                color: self.style.axis_color,
                thickness: 1.0,
            });
        }
    }

    fn paint_curves(&self, canvas: Rect, draw: &mut DrawList) {
        for curve in &self.curves {
            if !curve.visible || curve.keyframes.len() < 2 {
                continue;
            }

            let time_range = self.view.time_max - self.view.time_min;
            let step = time_range / CURVE_SEGMENTS as f32;
            let mut points = Vec::with_capacity(CURVE_SEGMENTS + 1);

            for i in 0..=CURVE_SEGMENTS {
                let t = self.view.time_min + i as f32 * step;
                let v = curve.evaluate(t);
                let screen = self.view.to_screen(t, v, canvas);
                points.push(screen);
            }

            draw.commands.push(DrawCommand::Polyline {
                points,
                color: curve.color,
                thickness: curve.thickness,
                closed: false,
            });
        }
    }

    fn paint_keys(&self, canvas: Rect, draw: &mut DrawList) {
        for curve in &self.curves {
            if !curve.visible {
                continue;
            }

            for (ki, key) in curve.keyframes.iter().enumerate() {
                let pos = self.view.to_screen(key.time, key.value, canvas);
                let is_selected = self.selection.is_selected(curve.id, ki);
                let is_hovered = self.hovered_key == Some(KeyRef {
                    curve_id: curve.id,
                    key_index: ki,
                });

                // Tangent handles (only for selected keys).
                if self.show_tangents && is_selected {
                    let time_scale = self.view.time_scale(canvas);
                    let value_scale = self.view.value_scale(canvas);

                    // Out tangent.
                    let out_dx = TANGENT_LINE_LENGTH;
                    let out_dy = -key.tangent_out * out_dx * value_scale / time_scale;
                    let out_pos = Vec2::new(pos.x + out_dx, pos.y + out_dy);

                    draw.commands.push(DrawCommand::Line {
                        start: pos,
                        end: out_pos,
                        color: self.style.tangent_line,
                        thickness: 1.0,
                    });
                    draw.commands.push(DrawCommand::Circle {
                        center: out_pos,
                        radius: TANGENT_HANDLE_RADIUS,
                        color: self.style.tangent_color,
                        border: BorderSpec::new(Color::BLACK, 1.0),
                    });

                    // In tangent.
                    let in_dx = -TANGENT_LINE_LENGTH;
                    let in_dy = key.tangent_in * TANGENT_LINE_LENGTH * value_scale / time_scale;
                    let in_pos = Vec2::new(pos.x + in_dx, pos.y + in_dy);

                    draw.commands.push(DrawCommand::Line {
                        start: pos,
                        end: in_pos,
                        color: self.style.tangent_line,
                        thickness: 1.0,
                    });
                    draw.commands.push(DrawCommand::Circle {
                        center: in_pos,
                        radius: TANGENT_HANDLE_RADIUS,
                        color: self.style.tangent_color,
                        border: BorderSpec::new(Color::BLACK, 1.0),
                    });
                }

                // Key diamond shape (using a rotated rect approximation).
                let key_color = if is_selected {
                    self.style.key_selected
                } else if is_hovered {
                    self.style.key_hovered
                } else {
                    curve.color.lighten(0.2)
                };

                let kr = KEY_RADIUS;
                draw.commands.push(DrawCommand::Triangle {
                    p0: Vec2::new(pos.x, pos.y - kr),
                    p1: Vec2::new(pos.x + kr, pos.y),
                    p2: Vec2::new(pos.x, pos.y + kr),
                    color: key_color,
                });
                draw.commands.push(DrawCommand::Triangle {
                    p0: Vec2::new(pos.x, pos.y - kr),
                    p1: Vec2::new(pos.x - kr, pos.y),
                    p2: Vec2::new(pos.x, pos.y + kr),
                    color: key_color,
                });

                // Border.
                draw.commands.push(DrawCommand::Circle {
                    center: pos,
                    radius: kr,
                    color: Color::TRANSPARENT,
                    border: BorderSpec::new(
                        if is_selected { Color::WHITE } else { Color::BLACK },
                        1.0,
                    ),
                });
            }
        }
    }

    fn paint_box_select(&self, draw: &mut DrawList) {
        if let InteractionMode::BoxSelect { start } = self.interaction {
            let r = Rect::new(
                Vec2::new(start.x.min(self.mouse_pos.x), start.y.min(self.mouse_pos.y)),
                Vec2::new(start.x.max(self.mouse_pos.x), start.y.max(self.mouse_pos.y)),
            );
            draw.commands.push(DrawCommand::Rect {
                rect: r,
                color: self.style.selection_box,
                corner_radii: CornerRadii::ZERO,
                border: BorderSpec::new(self.style.selection_border, 1.0),
                shadow: None,
            });
        }
    }

    fn paint_readout(&self, rect: Rect, canvas: Rect, draw: &mut DrawList) {
        if !self.show_readout {
            return;
        }

        let readout_text = format!("T: {:.3}  V: {:.3}", self.cursor_graph.0, self.cursor_graph.1);
        let readout_w = readout_text.len() as f32 * 7.0 + 16.0;
        let readout_rect = Rect::new(
            Vec2::new(rect.max.x - readout_w - 4.0, rect.min.y + 4.0),
            Vec2::new(rect.max.x - 4.0, rect.min.y + 4.0 + READOUT_HEIGHT),
        );

        draw.commands.push(DrawCommand::Rect {
            rect: readout_rect,
            color: self.style.readout_bg,
            corner_radii: CornerRadii::all(3.0),
            border: BorderSpec::default(),
            shadow: None,
        });
        draw.commands.push(DrawCommand::Text {
            text: readout_text,
            position: Vec2::new(readout_rect.min.x + 8.0, readout_rect.min.y + 3.0),
            font_size: self.style.font_size,
            color: self.style.readout_text,
            font_id: self.style.font_id,
            max_width: None,
            align: TextAlign::Left,
            vertical_align: TextVerticalAlign::Top,
        });
    }

    // =====================================================================
    // Event handling
    // =====================================================================

    /// Handle events.
    pub fn handle_event(&mut self, event: &UIEvent, rect: Rect) -> EventReply {
        if !self.enabled || !self.visible {
            return EventReply::Unhandled;
        }

        self.modified = false;
        let canvas = self.canvas_rect(rect);

        match event {
            UIEvent::Hover { position } | UIEvent::DragMove { position, .. } => {
                let pos = *position;
                self.mouse_pos = pos;
                self.cursor_graph = self.view.to_graph(pos, canvas);
                self.handle_mouse_move(pos, canvas)
            }

            UIEvent::Click { position, button, modifiers } => {
                let pos = *position;
                if !rect.contains(pos) {
                    return EventReply::Unhandled;
                }
                self.handle_mouse_down(pos, *button, *modifiers, canvas)
            }

            UIEvent::MouseUp { position, button } => {
                let mods = KeyModifiers::default();
                self.handle_mouse_up(*position, *button, mods, canvas)
            }

            UIEvent::Scroll { delta, .. } => {
                let factor = if delta.y > 0.0 {
                    1.0 - ZOOM_SPEED
                } else {
                    1.0 + ZOOM_SPEED
                };
                self.view.zoom(self.mouse_pos, factor, canvas);
                EventReply::Handled
            }

            UIEvent::KeyInput { key, pressed, modifiers } => {
                if *pressed {
                    self.handle_key(*key, *modifiers, canvas)
                } else {
                    EventReply::Unhandled
                }
            }

            _ => EventReply::Unhandled,
        }
    }

    fn handle_mouse_move(&mut self, pos: Vec2, canvas: Rect) -> EventReply {
        match self.interaction {
            InteractionMode::Pan { start, start_view } => {
                let delta = pos - start;
                self.view = start_view;
                self.view.pan(-delta, canvas);
                EventReply::Handled
            }

            InteractionMode::DragKey {
                curve_id,
                key_index,
                start_time,
                start_value,
            } => {
                let (new_t, new_v) = self.view.to_graph(pos, canvas);
                let (snapped_t, snapped_v) = self.snap_to_grid(new_t, new_v);
                if let Some(curve) = self.curves.iter_mut().find(|c| c.id == curve_id) {
                    if key_index < curve.keyframes.len() {
                        curve.keyframes[key_index].time = snapped_t;
                        curve.keyframes[key_index].value = snapped_v;
                    }
                }
                EventReply::Handled
            }

            InteractionMode::DragTangentOut { curve_id, key_index } => {
                if let Some(curve) = self.curves.iter_mut().find(|c| c.id == curve_id) {
                    if key_index < curve.keyframes.len() {
                        let key_pos = self.view.to_screen(
                            curve.keyframes[key_index].time,
                            curve.keyframes[key_index].value,
                            canvas,
                        );
                        let dx = pos.x - key_pos.x;
                        let dy = -(pos.y - key_pos.y);
                        let time_scale = self.view.time_scale(canvas);
                        let value_scale = self.view.value_scale(canvas);
                        if dx.abs() > 1.0 {
                            let tangent = (dy / value_scale) / (dx / time_scale);
                            curve.keyframes[key_index].tangent_out = tangent;
                            if !curve.keyframes[key_index].tangent_broken {
                                curve.keyframes[key_index].tangent_in = tangent;
                            }
                            curve.keyframes[key_index].tangent_mode = CurveKeyTangentMode::Free;
                        }
                    }
                }
                EventReply::Handled
            }

            InteractionMode::DragTangentIn { curve_id, key_index } => {
                if let Some(curve) = self.curves.iter_mut().find(|c| c.id == curve_id) {
                    if key_index < curve.keyframes.len() {
                        let key_pos = self.view.to_screen(
                            curve.keyframes[key_index].time,
                            curve.keyframes[key_index].value,
                            canvas,
                        );
                        let dx = key_pos.x - pos.x;
                        let dy = -(pos.y - key_pos.y);
                        let time_scale = self.view.time_scale(canvas);
                        let value_scale = self.view.value_scale(canvas);
                        if dx.abs() > 1.0 {
                            let tangent = (dy / value_scale) / (dx / time_scale);
                            curve.keyframes[key_index].tangent_in = tangent;
                            if !curve.keyframes[key_index].tangent_broken {
                                curve.keyframes[key_index].tangent_out = tangent;
                            }
                            curve.keyframes[key_index].tangent_mode = CurveKeyTangentMode::Free;
                        }
                    }
                }
                EventReply::Handled
            }

            InteractionMode::BoxSelect { .. } => EventReply::Handled,

            InteractionMode::None => {
                // Update hover state.
                self.hovered_key = None;
                self.hovered_tangent = None;

                for curve in &self.curves {
                    if !curve.visible {
                        continue;
                    }
                    for (ki, key) in curve.keyframes.iter().enumerate() {
                        let kp = self.view.to_screen(key.time, key.value, canvas);
                        if (pos - kp).length() <= KEY_HIT_RADIUS {
                            self.hovered_key = Some(KeyRef {
                                curve_id: curve.id,
                                key_index: ki,
                            });
                            return EventReply::Handled;
                        }

                        // Tangent handles.
                        if self.show_tangents && self.selection.is_selected(curve.id, ki) {
                            let ts = self.view.time_scale(canvas);
                            let vs = self.view.value_scale(canvas);

                            let out_dx = TANGENT_LINE_LENGTH;
                            let out_dy = -key.tangent_out * out_dx * vs / ts;
                            let out_pos = Vec2::new(kp.x + out_dx, kp.y + out_dy);
                            if (pos - out_pos).length() <= TANGENT_HIT_RADIUS {
                                self.hovered_tangent = Some((
                                    KeyRef {
                                        curve_id: curve.id,
                                        key_index: ki,
                                    },
                                    false,
                                ));
                                return EventReply::Handled;
                            }

                            let in_dx = -TANGENT_LINE_LENGTH;
                            let in_dy = key.tangent_in * TANGENT_LINE_LENGTH * vs / ts;
                            let in_pos = Vec2::new(kp.x + in_dx, kp.y + in_dy);
                            if (pos - in_pos).length() <= TANGENT_HIT_RADIUS {
                                self.hovered_tangent = Some((
                                    KeyRef {
                                        curve_id: curve.id,
                                        key_index: ki,
                                    },
                                    true,
                                ));
                                return EventReply::Handled;
                            }
                        }
                    }
                }
                EventReply::Unhandled
            }
        }
    }

    fn handle_mouse_down(
        &mut self,
        pos: Vec2,
        button: MouseButton,
        modifiers: KeyModifiers,
        canvas: Rect,
    ) -> EventReply {
        match button {
            MouseButton::Middle => {
                self.interaction = InteractionMode::Pan {
                    start: pos,
                    start_view: self.view,
                };
                EventReply::CaptureMouse
            }

            MouseButton::Left => {
                // Tangent handle?
                if let Some((kr, is_in)) = self.hovered_tangent {
                    self.push_undo();
                    self.interaction = if is_in {
                        InteractionMode::DragTangentIn {
                            curve_id: kr.curve_id,
                            key_index: kr.key_index,
                        }
                    } else {
                        InteractionMode::DragTangentOut {
                            curve_id: kr.curve_id,
                            key_index: kr.key_index,
                        }
                    };
                    return EventReply::CaptureMouse;
                }

                // Key?
                if let Some(kr) = self.hovered_key {
                    self.selection.select(kr.curve_id, kr.key_index, modifiers.ctrl);
                    let key_data = self.curves.iter()
                        .find(|c| c.id == kr.curve_id)
                        .and_then(|curve| {
                            curve.keyframes.get(kr.key_index).map(|k| (k.time, k.value))
                        });
                    if let Some((start_time, start_value)) = key_data {
                        self.push_undo();
                        self.interaction = InteractionMode::DragKey {
                            curve_id: kr.curve_id,
                            key_index: kr.key_index,
                            start_time,
                            start_value,
                        };
                    }
                    return EventReply::CaptureMouse;
                }

                // Empty space -> start box select.
                if canvas.contains(pos) {
                    if !modifiers.ctrl {
                        self.selection.clear();
                    }
                    self.interaction = InteractionMode::BoxSelect { start: pos };
                    return EventReply::CaptureMouse;
                }

                EventReply::Unhandled
            }

            MouseButton::Right => {
                // Provide context for a right-click menu (handled externally).
                EventReply::Unhandled
            }

            _ => EventReply::Unhandled,
        }
    }

    fn handle_mouse_up(
        &mut self,
        pos: Vec2,
        button: MouseButton,
        modifiers: KeyModifiers,
        canvas: Rect,
    ) -> EventReply {
        match self.interaction {
            InteractionMode::Pan { .. } => {
                if button == MouseButton::Middle {
                    self.interaction = InteractionMode::None;
                    return EventReply::ReleaseMouse;
                }
            }

            InteractionMode::DragKey { curve_id, .. } => {
                if button == MouseButton::Left {
                    // Re-sort keyframes.
                    if let Some(curve) = self.curves.iter_mut().find(|c| c.id == curve_id) {
                        curve.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
                        curve.compute_auto_tangents();
                    }
                    self.interaction = InteractionMode::None;
                    self.modified = true;
                    return EventReply::ReleaseMouse;
                }
            }

            InteractionMode::DragTangentIn { .. } | InteractionMode::DragTangentOut { .. } => {
                if button == MouseButton::Left {
                    self.interaction = InteractionMode::None;
                    self.modified = true;
                    return EventReply::ReleaseMouse;
                }
            }

            InteractionMode::BoxSelect { start } => {
                if button == MouseButton::Left {
                    let select_rect = Rect::new(
                        Vec2::new(start.x.min(pos.x), start.y.min(pos.y)),
                        Vec2::new(start.x.max(pos.x), start.y.max(pos.y)),
                    );
                    if select_rect.width() > BOX_SELECT_MIN
                        || select_rect.height() > BOX_SELECT_MIN
                    {
                        let positions = self.all_key_positions(canvas);
                        self.selection
                            .select_in_rect(&positions, select_rect, modifiers.ctrl);
                    }
                    self.interaction = InteractionMode::None;
                    return EventReply::ReleaseMouse;
                }
            }

            InteractionMode::None => {}
        }

        EventReply::Unhandled
    }

    fn handle_key(&mut self, key: KeyCode, modifiers: KeyModifiers, canvas: Rect) -> EventReply {
        match key {
            KeyCode::Delete | KeyCode::Backspace => {
                self.delete_selected();
                EventReply::Handled
            }

            KeyCode::A if modifiers.ctrl => {
                // Select all.
                self.selection.selected.clear();
                for curve in &self.curves {
                    if !curve.visible {
                        continue;
                    }
                    for ki in 0..curve.keyframes.len() {
                        self.selection.selected.push(KeyRef {
                            curve_id: curve.id,
                            key_index: ki,
                        });
                    }
                }
                EventReply::Handled
            }

            KeyCode::C if modifiers.ctrl => {
                self.copy_selected();
                EventReply::Handled
            }

            KeyCode::V if modifiers.ctrl => {
                self.paste();
                EventReply::Handled
            }

            KeyCode::Z if modifiers.ctrl => {
                if modifiers.shift {
                    self.redo();
                } else {
                    self.undo();
                }
                EventReply::Handled
            }

            KeyCode::F => {
                self.fit_to_content();
                EventReply::Handled
            }

            KeyCode::G => {
                self.snap_enabled = !self.snap_enabled;
                EventReply::Handled
            }

            KeyCode::T => {
                self.show_tangents = !self.show_tangents;
                EventReply::Handled
            }

            _ => EventReply::Unhandled,
        }
    }

    /// Take the modified flag.
    pub fn take_modified(&mut self) -> bool {
        let m = self.modified;
        self.modified = false;
        m
    }
}

impl Default for FullCurveEditor {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curve_evaluate_linear() {
        let mut curve = CurveData::new(0, "test", Color::RED);
        curve.add_key(FullCurveKeyframe::new(0.0, 0.0).with_tangent_mode(CurveKeyTangentMode::Linear));
        curve.add_key(FullCurveKeyframe::new(1.0, 1.0).with_tangent_mode(CurveKeyTangentMode::Linear));
        assert!((curve.evaluate(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_curve_evaluate_step() {
        let mut curve = CurveData::new(0, "test", Color::RED);
        curve.add_key(FullCurveKeyframe::new(0.0, 0.0).with_tangent_mode(CurveKeyTangentMode::Step));
        curve.add_key(FullCurveKeyframe::new(1.0, 1.0));
        assert!((curve.evaluate(0.5) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_curve_add_key_sorted() {
        let mut curve = CurveData::new(0, "test", Color::RED);
        curve.add_key(FullCurveKeyframe::new(2.0, 0.0));
        curve.add_key(FullCurveKeyframe::new(0.0, 0.0));
        curve.add_key(FullCurveKeyframe::new(1.0, 0.0));
        assert_eq!(curve.keyframes[0].time, 0.0);
        assert_eq!(curve.keyframes[1].time, 1.0);
        assert_eq!(curve.keyframes[2].time, 2.0);
    }

    #[test]
    fn test_view_transform_roundtrip() {
        let view = ViewTransform::default();
        let canvas = Rect::new(Vec2::ZERO, Vec2::new(800.0, 600.0));
        let screen = view.to_screen(1.0, 0.5, canvas);
        let (t, v) = view.to_graph(screen, canvas);
        assert!((t - 1.0).abs() < 0.01);
        assert!((v - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_selection() {
        let mut sel = CurveSelection::new();
        sel.select(0, 1, false);
        assert!(sel.is_selected(0, 1));
        assert!(!sel.is_selected(0, 2));
    }

    #[test]
    fn test_auto_tangents() {
        let mut curve = CurveData::new(0, "test", Color::RED);
        curve.add_key(FullCurveKeyframe::new(0.0, 0.0));
        curve.add_key(FullCurveKeyframe::new(1.0, 1.0));
        curve.add_key(FullCurveKeyframe::new(2.0, 0.0));
        curve.compute_auto_tangents();
        // Middle key should have auto-tangent = 0.0 (symmetric around max).
        assert!((curve.keyframes[1].tangent_out - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_undo_redo() {
        let mut editor = FullCurveEditor::new();
        let mut curve = CurveData::new(0, "test", Color::RED);
        curve.add_key(FullCurveKeyframe::new(0.0, 0.0));
        editor.add_curve(curve);

        editor.add_key_at(0, 1.0, 1.0);
        assert_eq!(editor.curves[0].keyframes.len(), 2);

        editor.undo();
        assert_eq!(editor.curves[0].keyframes.len(), 1);

        editor.redo();
        assert_eq!(editor.curves[0].keyframes.len(), 2);
    }

    #[test]
    fn test_grid_spacing() {
        let spacing = FullCurveEditor::grid_spacing(10.0, 800.0);
        assert!(spacing > 0.0);
    }
}
