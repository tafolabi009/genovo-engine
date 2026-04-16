// =============================================================================
// Genovo Engine - Animation Curve Editor
// =============================================================================
//
// Visual editor for animation curves. Provides keyframe manipulation,
// tangent handle editing, multi-curve overlay, zoom/pan, box selection,
// copy/paste, and snap-to-grid functionality.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Tangent Mode
// ---------------------------------------------------------------------------

/// How tangent handles behave at a keyframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TangentMode {
    /// Tangents are automatically computed for smooth interpolation (Catmull-Rom style).
    Auto,
    /// Linear interpolation between keyframes (tangent slopes match segment slope).
    Linear,
    /// Both tangent handles move independently.
    Free,
    /// Both handles are co-linear but may have different lengths (weighted).
    Weighted,
    /// Step function: holds the value until the next keyframe.
    Constant,
}

impl Default for TangentMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl TangentMode {
    /// Display-friendly name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Linear => "Linear",
            Self::Free => "Free",
            Self::Weighted => "Weighted",
            Self::Constant => "Constant",
        }
    }

    /// All tangent modes.
    pub fn all() -> &'static [TangentMode] {
        &[Self::Auto, Self::Linear, Self::Free, Self::Weighted, Self::Constant]
    }
}

// ---------------------------------------------------------------------------
// Keyframe
// ---------------------------------------------------------------------------

/// A single keyframe on an animation curve.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Keyframe {
    /// Unique identifier for this keyframe.
    pub id: Uuid,
    /// Time position (x-axis).
    pub time: f32,
    /// Value at this time (y-axis).
    pub value: f32,
    /// Incoming tangent slope (left handle).
    pub in_tangent: f32,
    /// Outgoing tangent slope (right handle).
    pub out_tangent: f32,
    /// Incoming tangent weight (for weighted mode).
    pub in_weight: f32,
    /// Outgoing tangent weight (for weighted mode).
    pub out_weight: f32,
    /// Tangent mode for this keyframe.
    pub tangent_mode: TangentMode,
}

impl Keyframe {
    /// Create a new keyframe at the given time and value.
    pub fn new(time: f32, value: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            time,
            value,
            in_tangent: 0.0,
            out_tangent: 0.0,
            in_weight: 1.0 / 3.0,
            out_weight: 1.0 / 3.0,
            tangent_mode: TangentMode::Auto,
        }
    }

    /// Create a keyframe with a specific tangent mode.
    pub fn with_tangent_mode(mut self, mode: TangentMode) -> Self {
        self.tangent_mode = mode;
        self
    }

    /// Create a keyframe with explicit tangent slopes.
    pub fn with_tangents(mut self, in_tangent: f32, out_tangent: f32) -> Self {
        self.in_tangent = in_tangent;
        self.out_tangent = out_tangent;
        self
    }
}

// ---------------------------------------------------------------------------
// Animation Curve
// ---------------------------------------------------------------------------

/// A single animation curve consisting of ordered keyframes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationCurve {
    /// Unique identifier for this curve.
    pub id: Uuid,
    /// Display name (e.g., "Position.X", "Opacity").
    pub name: String,
    /// Display color for this curve (RGBA).
    pub color: [f32; 4],
    /// Sorted list of keyframes.
    pub keyframes: Vec<Keyframe>,
    /// Whether this curve is visible in the editor.
    pub visible: bool,
    /// Whether this curve is locked (read-only in the editor).
    pub locked: bool,
    /// Pre-infinity behavior.
    pub pre_infinity: WrapMode,
    /// Post-infinity behavior.
    pub post_infinity: WrapMode,
}

/// How the curve behaves outside its keyframe range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WrapMode {
    /// Clamp to the first/last keyframe value.
    Clamp,
    /// Repeat the curve (loop).
    Loop,
    /// Ping-pong (mirror) the curve.
    PingPong,
    /// Continue with the tangent slope of the boundary keyframe.
    Linear,
}

impl Default for WrapMode {
    fn default() -> Self {
        Self::Clamp
    }
}

impl AnimationCurve {
    /// Create a new empty curve.
    pub fn new(name: impl Into<String>, color: [f32; 4]) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            color,
            keyframes: Vec::new(),
            visible: true,
            locked: false,
            pre_infinity: WrapMode::Clamp,
            post_infinity: WrapMode::Clamp,
        }
    }

    /// Add a keyframe, maintaining sorted order by time.
    pub fn add_keyframe(&mut self, key: Keyframe) -> Uuid {
        let id = key.id;
        let insert_pos = self
            .keyframes
            .binary_search_by(|k| k.time.partial_cmp(&key.time).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|pos| pos);
        self.keyframes.insert(insert_pos, key);
        self.auto_compute_tangents();
        id
    }

    /// Remove a keyframe by ID.
    pub fn remove_keyframe(&mut self, id: Uuid) -> bool {
        let len_before = self.keyframes.len();
        self.keyframes.retain(|k| k.id != id);
        let removed = self.keyframes.len() < len_before;
        if removed {
            self.auto_compute_tangents();
        }
        removed
    }

    /// Move a keyframe to a new time and value.
    pub fn move_keyframe(&mut self, id: Uuid, new_time: f32, new_value: f32) {
        if let Some(key) = self.keyframes.iter_mut().find(|k| k.id == id) {
            key.time = new_time;
            key.value = new_value;
        }
        // Re-sort.
        self.keyframes
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        self.auto_compute_tangents();
    }

    /// Set the tangent mode for a specific keyframe.
    pub fn set_tangent_mode(&mut self, id: Uuid, mode: TangentMode) {
        if let Some(key) = self.keyframes.iter_mut().find(|k| k.id == id) {
            key.tangent_mode = mode;
        }
        self.auto_compute_tangents();
    }

    /// Set tangent slopes manually for a keyframe.
    pub fn set_tangents(&mut self, id: Uuid, in_tangent: f32, out_tangent: f32) {
        if let Some(key) = self.keyframes.iter_mut().find(|k| k.id == id) {
            key.in_tangent = in_tangent;
            key.out_tangent = out_tangent;
            // Manual tangent editing implies Free mode.
            if key.tangent_mode == TangentMode::Auto || key.tangent_mode == TangentMode::Linear {
                key.tangent_mode = TangentMode::Free;
            }
        }
    }

    /// Auto-compute tangents for keyframes in Auto or Linear mode.
    pub fn auto_compute_tangents(&mut self) {
        let count = self.keyframes.len();
        if count == 0 {
            return;
        }

        // Collect times and values for immutable access.
        let times: Vec<f32> = self.keyframes.iter().map(|k| k.time).collect();
        let values: Vec<f32> = self.keyframes.iter().map(|k| k.value).collect();
        let modes: Vec<TangentMode> = self.keyframes.iter().map(|k| k.tangent_mode).collect();

        for i in 0..count {
            match modes[i] {
                TangentMode::Auto => {
                    let slope = if count == 1 {
                        0.0
                    } else if i == 0 {
                        // Forward difference.
                        let dt = times[1] - times[0];
                        if dt.abs() < 1e-8 { 0.0 } else { (values[1] - values[0]) / dt }
                    } else if i == count - 1 {
                        // Backward difference.
                        let dt = times[i] - times[i - 1];
                        if dt.abs() < 1e-8 { 0.0 } else { (values[i] - values[i - 1]) / dt }
                    } else {
                        // Catmull-Rom: average of forward and backward slopes.
                        let dt_prev = times[i] - times[i - 1];
                        let dt_next = times[i + 1] - times[i];
                        let slope_prev = if dt_prev.abs() < 1e-8 {
                            0.0
                        } else {
                            (values[i] - values[i - 1]) / dt_prev
                        };
                        let slope_next = if dt_next.abs() < 1e-8 {
                            0.0
                        } else {
                            (values[i + 1] - values[i]) / dt_next
                        };
                        (slope_prev + slope_next) * 0.5
                    };
                    self.keyframes[i].in_tangent = slope;
                    self.keyframes[i].out_tangent = slope;
                }
                TangentMode::Linear => {
                    // In tangent: slope from previous key.
                    if i > 0 {
                        let dt = times[i] - times[i - 1];
                        self.keyframes[i].in_tangent = if dt.abs() < 1e-8 {
                            0.0
                        } else {
                            (values[i] - values[i - 1]) / dt
                        };
                    }
                    // Out tangent: slope to next key.
                    if i + 1 < count {
                        let dt = times[i + 1] - times[i];
                        self.keyframes[i].out_tangent = if dt.abs() < 1e-8 {
                            0.0
                        } else {
                            (values[i + 1] - values[i]) / dt
                        };
                    }
                }
                TangentMode::Free | TangentMode::Weighted | TangentMode::Constant => {
                    // Leave tangents as-is.
                }
            }
        }
    }

    /// Evaluate the curve at a given time using Hermite interpolation.
    pub fn evaluate(&self, time: f32) -> f32 {
        let count = self.keyframes.len();
        if count == 0 {
            return 0.0;
        }
        if count == 1 {
            return self.keyframes[0].value;
        }

        let first = &self.keyframes[0];
        let last = &self.keyframes[count - 1];

        // Handle extrapolation.
        if time <= first.time {
            return match self.pre_infinity {
                WrapMode::Clamp => first.value,
                WrapMode::Linear => {
                    first.value + first.in_tangent * (time - first.time)
                }
                WrapMode::Loop => {
                    let range = last.time - first.time;
                    if range.abs() < 1e-8 {
                        first.value
                    } else {
                        let wrapped = last.time - ((first.time - time) % range);
                        self.evaluate_inner(wrapped)
                    }
                }
                WrapMode::PingPong => {
                    let range = last.time - first.time;
                    if range.abs() < 1e-8 {
                        first.value
                    } else {
                        let offset = (first.time - time) % (range * 2.0);
                        if offset <= range {
                            self.evaluate_inner(first.time + offset)
                        } else {
                            self.evaluate_inner(last.time - (offset - range))
                        }
                    }
                }
            };
        }

        if time >= last.time {
            return match self.post_infinity {
                WrapMode::Clamp => last.value,
                WrapMode::Linear => {
                    last.value + last.out_tangent * (time - last.time)
                }
                WrapMode::Loop => {
                    let range = last.time - first.time;
                    if range.abs() < 1e-8 {
                        last.value
                    } else {
                        let wrapped = first.time + ((time - first.time) % range);
                        self.evaluate_inner(wrapped)
                    }
                }
                WrapMode::PingPong => {
                    let range = last.time - first.time;
                    if range.abs() < 1e-8 {
                        last.value
                    } else {
                        let offset = (time - first.time) % (range * 2.0);
                        if offset <= range {
                            self.evaluate_inner(first.time + offset)
                        } else {
                            self.evaluate_inner(last.time - (offset - range))
                        }
                    }
                }
            };
        }

        self.evaluate_inner(time)
    }

    /// Evaluate the curve within its valid range.
    fn evaluate_inner(&self, time: f32) -> f32 {
        // Find the segment.
        let count = self.keyframes.len();
        for i in 0..count - 1 {
            let k0 = &self.keyframes[i];
            let k1 = &self.keyframes[i + 1];

            if time >= k0.time && time <= k1.time {
                // Constant mode: step function.
                if k0.tangent_mode == TangentMode::Constant {
                    return k0.value;
                }

                let dt = k1.time - k0.time;
                if dt.abs() < 1e-8 {
                    return k0.value;
                }

                let t = (time - k0.time) / dt;

                // Hermite interpolation.
                let t2 = t * t;
                let t3 = t2 * t;

                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + t;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;

                let m0 = k0.out_tangent * dt;
                let m1 = k1.in_tangent * dt;

                return h00 * k0.value + h10 * m0 + h01 * k1.value + h11 * m1;
            }
        }

        self.keyframes.last().map(|k| k.value).unwrap_or(0.0)
    }

    /// Time range of the curve.
    pub fn time_range(&self) -> Option<(f32, f32)> {
        if self.keyframes.is_empty() {
            return None;
        }
        Some((
            self.keyframes.first().unwrap().time,
            self.keyframes.last().unwrap().time,
        ))
    }

    /// Value range of the curve (min, max).
    pub fn value_range(&self) -> Option<(f32, f32)> {
        if self.keyframes.is_empty() {
            return None;
        }
        let min = self
            .keyframes
            .iter()
            .map(|k| k.value)
            .fold(f32::INFINITY, f32::min);
        let max = self
            .keyframes
            .iter()
            .map(|k| k.value)
            .fold(f32::NEG_INFINITY, f32::max);
        Some((min, max))
    }

    /// Number of keyframes.
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Find a keyframe by ID.
    pub fn find_keyframe(&self, id: Uuid) -> Option<&Keyframe> {
        self.keyframes.iter().find(|k| k.id == id)
    }

    /// Find a mutable keyframe by ID.
    pub fn find_keyframe_mut(&mut self, id: Uuid) -> Option<&mut Keyframe> {
        self.keyframes.iter_mut().find(|k| k.id == id)
    }

    /// Sample the curve at regular intervals for display.
    pub fn sample(&self, start_time: f32, end_time: f32, num_samples: usize) -> Vec<[f32; 2]> {
        if num_samples < 2 {
            return Vec::new();
        }
        let step = (end_time - start_time) / (num_samples - 1) as f32;
        (0..num_samples)
            .map(|i| {
                let t = start_time + step * i as f32;
                [t, self.evaluate(t)]
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Curve Editor
// ---------------------------------------------------------------------------

/// Visual editor state for manipulating animation curves.
#[derive(Debug, Clone)]
pub struct CurveEditor {
    /// The curves being edited.
    pub curves: Vec<AnimationCurve>,
    /// Currently selected keyframe IDs.
    pub selected_keyframes: Vec<Uuid>,
    /// The curve index + keyframe being hovered.
    pub hovered_keyframe: Option<(usize, Uuid)>,
    /// Whether we are in tangent-handle editing mode.
    pub editing_tangent: Option<TangentEditState>,
    /// View state.
    pub view: CurveEditorView,
    /// Grid snapping.
    pub snap_to_grid: bool,
    /// Frame snapping (snap time to integer frame boundaries).
    pub snap_to_frame: bool,
    /// Frames per second (for frame snapping).
    pub fps: f32,
    /// Box selection state.
    box_select: Option<BoxSelectState>,
    /// Clipboard for copy/paste.
    clipboard: Vec<Keyframe>,
    /// Drag state for moving keyframes.
    drag_state: Option<KeyframeDragState>,
    /// Whether the editor is active/visible.
    pub active: bool,
}

/// View parameters for the curve editor canvas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveEditorView {
    /// Visible time range (start, end).
    pub time_range: (f32, f32),
    /// Visible value range (min, max).
    pub value_range: (f32, f32),
    /// Canvas width in pixels.
    pub canvas_width: f32,
    /// Canvas height in pixels.
    pub canvas_height: f32,
    /// Grid line spacing for time axis.
    pub time_grid_spacing: f32,
    /// Grid line spacing for value axis.
    pub value_grid_spacing: f32,
}

impl Default for CurveEditorView {
    fn default() -> Self {
        Self {
            time_range: (0.0, 5.0),
            value_range: (-1.0, 1.0),
            canvas_width: 800.0,
            canvas_height: 400.0,
            time_grid_spacing: 1.0,
            value_grid_spacing: 0.25,
        }
    }
}

impl CurveEditorView {
    /// Convert a time+value pair to pixel coordinates.
    pub fn to_screen(&self, time: f32, value: f32) -> [f32; 2] {
        let t_range = self.time_range.1 - self.time_range.0;
        let v_range = self.value_range.1 - self.value_range.0;
        let x = if t_range.abs() > 1e-8 {
            (time - self.time_range.0) / t_range * self.canvas_width
        } else {
            self.canvas_width * 0.5
        };
        let y = if v_range.abs() > 1e-8 {
            (1.0 - (value - self.value_range.0) / v_range) * self.canvas_height
        } else {
            self.canvas_height * 0.5
        };
        [x, y]
    }

    /// Convert pixel coordinates to time+value.
    pub fn from_screen(&self, x: f32, y: f32) -> (f32, f32) {
        let t_range = self.time_range.1 - self.time_range.0;
        let v_range = self.value_range.1 - self.value_range.0;
        let time = self.time_range.0 + (x / self.canvas_width.max(1.0)) * t_range;
        let value = self.value_range.0 + (1.0 - y / self.canvas_height.max(1.0)) * v_range;
        (time, value)
    }

    /// Pan the view by a pixel delta.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let t_range = self.time_range.1 - self.time_range.0;
        let v_range = self.value_range.1 - self.value_range.0;
        let dt = -dx / self.canvas_width.max(1.0) * t_range;
        let dv = dy / self.canvas_height.max(1.0) * v_range;
        self.time_range.0 += dt;
        self.time_range.1 += dt;
        self.value_range.0 += dv;
        self.value_range.1 += dv;
    }

    /// Zoom the view around a pivot point (in pixels).
    pub fn zoom(&mut self, factor: f32, pivot_x: f32, pivot_y: f32) {
        let (pivot_time, pivot_value) = self.from_screen(pivot_x, pivot_y);
        let scale = 1.0 - factor * 0.1;
        let scale = scale.clamp(0.5, 2.0);

        self.time_range.0 = pivot_time + (self.time_range.0 - pivot_time) * scale;
        self.time_range.1 = pivot_time + (self.time_range.1 - pivot_time) * scale;
        self.value_range.0 = pivot_value + (self.value_range.0 - pivot_value) * scale;
        self.value_range.1 = pivot_value + (self.value_range.1 - pivot_value) * scale;

        self.update_grid_spacing();
    }

    /// Zoom to fit all keyframes in the view with some padding.
    pub fn zoom_to_fit(&mut self, curves: &[AnimationCurve]) {
        let mut t_min = f32::INFINITY;
        let mut t_max = f32::NEG_INFINITY;
        let mut v_min = f32::INFINITY;
        let mut v_max = f32::NEG_INFINITY;

        for curve in curves {
            for key in &curve.keyframes {
                t_min = t_min.min(key.time);
                t_max = t_max.max(key.time);
                v_min = v_min.min(key.value);
                v_max = v_max.max(key.value);
            }
        }

        if t_min > t_max {
            // No keyframes.
            self.time_range = (0.0, 5.0);
            self.value_range = (-1.0, 1.0);
            return;
        }

        let t_padding = ((t_max - t_min) * 0.1).max(0.5);
        let v_padding = ((v_max - v_min) * 0.1).max(0.1);

        self.time_range = (t_min - t_padding, t_max + t_padding);
        self.value_range = (v_min - v_padding, v_max + v_padding);
        self.update_grid_spacing();
    }

    /// Update grid spacing based on current zoom level.
    fn update_grid_spacing(&mut self) {
        let t_range = self.time_range.1 - self.time_range.0;
        let v_range = self.value_range.1 - self.value_range.0;

        // Aim for roughly 8-12 grid lines per axis.
        self.time_grid_spacing = nice_grid_spacing(t_range, 10.0);
        self.value_grid_spacing = nice_grid_spacing(v_range, 8.0);
    }

    /// Get the time and value at the cursor position.
    pub fn cursor_readout(&self, x: f32, y: f32) -> (f32, f32) {
        self.from_screen(x, y)
    }
}

/// State tracking tangent handle editing.
#[derive(Debug, Clone)]
pub struct TangentEditState {
    /// Curve index being edited.
    pub curve_index: usize,
    /// Keyframe whose tangent is being edited.
    pub keyframe_id: Uuid,
    /// Whether editing the in-tangent (true) or out-tangent (false).
    pub is_in_tangent: bool,
}

/// State for box selection.
#[derive(Debug, Clone)]
struct BoxSelectState {
    start: [f32; 2],
    current: [f32; 2],
}

/// State for dragging selected keyframes.
#[derive(Debug, Clone)]
struct KeyframeDragState {
    /// Starting mouse position in screen coords.
    start_mouse: [f32; 2],
    /// Snapshot of selected keyframes' original positions.
    original_positions: Vec<(Uuid, f32, f32)>,
}

impl Default for CurveEditor {
    fn default() -> Self {
        Self {
            curves: Vec::new(),
            selected_keyframes: Vec::new(),
            hovered_keyframe: None,
            editing_tangent: None,
            view: CurveEditorView::default(),
            snap_to_grid: false,
            snap_to_frame: false,
            fps: 30.0,
            box_select: None,
            clipboard: Vec::new(),
            drag_state: None,
            active: true,
        }
    }
}

impl CurveEditor {
    /// Create a new curve editor.
    pub fn new() -> Self {
        Self::default()
    }

    // --- Curve management ---

    /// Add a curve to the editor.
    pub fn add_curve(&mut self, curve: AnimationCurve) -> usize {
        self.curves.push(curve);
        self.curves.len() - 1
    }

    /// Remove a curve by index.
    pub fn remove_curve(&mut self, index: usize) -> Option<AnimationCurve> {
        if index < self.curves.len() {
            // Remove selected keyframes that belong to this curve.
            let curve = &self.curves[index];
            let curve_key_ids: Vec<Uuid> = curve.keyframes.iter().map(|k| k.id).collect();
            self.selected_keyframes.retain(|id| !curve_key_ids.contains(id));
            Some(self.curves.remove(index))
        } else {
            None
        }
    }

    // --- Keyframe selection ---

    /// Select a single keyframe.
    pub fn select_keyframe(&mut self, keyframe_id: Uuid) {
        self.selected_keyframes.clear();
        self.selected_keyframes.push(keyframe_id);
    }

    /// Add a keyframe to the selection.
    pub fn add_to_selection(&mut self, keyframe_id: Uuid) {
        if !self.selected_keyframes.contains(&keyframe_id) {
            self.selected_keyframes.push(keyframe_id);
        }
    }

    /// Toggle a keyframe's selection state.
    pub fn toggle_selection(&mut self, keyframe_id: Uuid) {
        if let Some(pos) = self.selected_keyframes.iter().position(|&id| id == keyframe_id) {
            self.selected_keyframes.remove(pos);
        } else {
            self.selected_keyframes.push(keyframe_id);
        }
    }

    /// Clear the selection.
    pub fn clear_selection(&mut self) {
        self.selected_keyframes.clear();
    }

    /// Whether a keyframe is selected.
    pub fn is_selected(&self, keyframe_id: Uuid) -> bool {
        self.selected_keyframes.contains(&keyframe_id)
    }

    // --- Box selection ---

    /// Begin a box selection drag.
    pub fn begin_box_select(&mut self, start: [f32; 2]) {
        self.box_select = Some(BoxSelectState {
            start,
            current: start,
        });
    }

    /// Update the box selection rectangle.
    pub fn update_box_select(&mut self, current: [f32; 2]) {
        if let Some(ref mut bs) = self.box_select {
            bs.current = current;
        }
    }

    /// End box selection, selecting all keyframes within the rectangle.
    pub fn end_box_select(&mut self, additive: bool) {
        let bs = match self.box_select.take() {
            Some(b) => b,
            None => return,
        };

        if !additive {
            self.selected_keyframes.clear();
        }

        let min_x = bs.start[0].min(bs.current[0]);
        let max_x = bs.start[0].max(bs.current[0]);
        let min_y = bs.start[1].min(bs.current[1]);
        let max_y = bs.start[1].max(bs.current[1]);

        for curve in &self.curves {
            if !curve.visible {
                continue;
            }
            for key in &curve.keyframes {
                let screen = self.view.to_screen(key.time, key.value);
                if screen[0] >= min_x
                    && screen[0] <= max_x
                    && screen[1] >= min_y
                    && screen[1] <= max_y
                {
                    if !self.selected_keyframes.contains(&key.id) {
                        self.selected_keyframes.push(key.id);
                    }
                }
            }
        }
    }

    /// Whether a box selection is in progress.
    pub fn is_box_selecting(&self) -> bool {
        self.box_select.is_some()
    }

    // --- Keyframe manipulation ---

    /// Add a keyframe at a screen position.
    pub fn add_keyframe_at_screen(
        &mut self,
        curve_index: usize,
        screen_x: f32,
        screen_y: f32,
    ) -> Option<Uuid> {
        let (time, value) = self.view.from_screen(screen_x, screen_y);
        let (time, value) = self.apply_snap(time, value);

        if curve_index < self.curves.len() {
            let key = Keyframe::new(time, value);
            let id = self.curves[curve_index].add_keyframe(key);
            Some(id)
        } else {
            None
        }
    }

    /// Delete all selected keyframes.
    pub fn delete_selected(&mut self) {
        let selected = self.selected_keyframes.clone();
        for curve in &mut self.curves {
            for id in &selected {
                curve.remove_keyframe(*id);
            }
        }
        self.selected_keyframes.clear();
    }

    /// Begin dragging the selected keyframes.
    pub fn begin_drag(&mut self, mouse_pos: [f32; 2]) {
        let mut original_positions = Vec::new();
        for curve in &self.curves {
            for key in &curve.keyframes {
                if self.selected_keyframes.contains(&key.id) {
                    original_positions.push((key.id, key.time, key.value));
                }
            }
        }
        self.drag_state = Some(KeyframeDragState {
            start_mouse: mouse_pos,
            original_positions,
        });
    }

    /// Update keyframe positions during drag.
    pub fn update_drag(&mut self, mouse_pos: [f32; 2]) {
        let drag = match &self.drag_state {
            Some(d) => d.clone(),
            None => return,
        };

        let (start_time, start_value) = self.view.from_screen(drag.start_mouse[0], drag.start_mouse[1]);
        let (current_time, current_value) = self.view.from_screen(mouse_pos[0], mouse_pos[1]);
        let dt = current_time - start_time;
        let dv = current_value - start_value;

        for (id, orig_time, orig_value) in &drag.original_positions {
            let new_time = orig_time + dt;
            let new_value = orig_value + dv;
            let (snapped_time, snapped_value) = self.apply_snap(new_time, new_value);
            for curve in &mut self.curves {
                if let Some(key) = curve.find_keyframe_mut(*id) {
                    key.time = snapped_time;
                    key.value = snapped_value;
                }
            }
        }

        // Re-sort keyframes.
        for curve in &mut self.curves {
            curve.keyframes.sort_by(|a, b| {
                a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal)
            });
            curve.auto_compute_tangents();
        }
    }

    /// End the drag operation.
    pub fn end_drag(&mut self) {
        self.drag_state = None;
    }

    /// Whether a drag is in progress.
    pub fn is_dragging(&self) -> bool {
        self.drag_state.is_some()
    }

    // --- Copy/paste ---

    /// Copy selected keyframes to clipboard.
    pub fn copy_selected(&mut self) {
        self.clipboard.clear();
        for curve in &self.curves {
            for key in &curve.keyframes {
                if self.selected_keyframes.contains(&key.id) {
                    self.clipboard.push(key.clone());
                }
            }
        }
    }

    /// Paste keyframes from clipboard into a curve, optionally at a time offset.
    pub fn paste(&mut self, curve_index: usize, time_offset: f32) {
        if curve_index >= self.curves.len() || self.clipboard.is_empty() {
            return;
        }

        self.selected_keyframes.clear();

        // Find the earliest time in the clipboard.
        let min_time = self
            .clipboard
            .iter()
            .map(|k| k.time)
            .fold(f32::INFINITY, f32::min);

        for orig in &self.clipboard {
            let mut key = orig.clone();
            key.id = Uuid::new_v4();
            key.time = key.time - min_time + time_offset;
            let id = self.curves[curve_index].add_keyframe(key);
            self.selected_keyframes.push(id);
        }
    }

    // --- Snapping ---

    /// Apply snap settings to a time/value pair.
    fn apply_snap(&self, time: f32, value: f32) -> (f32, f32) {
        let snapped_time = if self.snap_to_frame {
            (time * self.fps).round() / self.fps
        } else if self.snap_to_grid {
            snap_to_increment(time, self.view.time_grid_spacing)
        } else {
            time
        };

        let snapped_value = if self.snap_to_grid {
            snap_to_increment(value, self.view.value_grid_spacing)
        } else {
            value
        };

        (snapped_time, snapped_value)
    }

    // --- View operations ---

    /// Zoom to fit all curves.
    pub fn zoom_to_fit(&mut self) {
        self.view.zoom_to_fit(&self.curves);
    }

    /// Frame the selected keyframes (zoom to fit selection).
    pub fn frame_selected(&mut self) {
        if self.selected_keyframes.is_empty() {
            return;
        }

        let mut t_min = f32::INFINITY;
        let mut t_max = f32::NEG_INFINITY;
        let mut v_min = f32::INFINITY;
        let mut v_max = f32::NEG_INFINITY;

        for curve in &self.curves {
            for key in &curve.keyframes {
                if self.selected_keyframes.contains(&key.id) {
                    t_min = t_min.min(key.time);
                    t_max = t_max.max(key.time);
                    v_min = v_min.min(key.value);
                    v_max = v_max.max(key.value);
                }
            }
        }

        if t_min > t_max {
            return;
        }

        let padding_t = ((t_max - t_min) * 0.2).max(0.5);
        let padding_v = ((v_max - v_min) * 0.2).max(0.1);
        self.view.time_range = (t_min - padding_t, t_max + padding_t);
        self.view.value_range = (v_min - padding_v, v_max + padding_v);
    }

    // --- Hit testing ---

    /// Find the keyframe nearest to a screen position, within a pixel threshold.
    pub fn hit_test_keyframe(
        &self,
        screen_x: f32,
        screen_y: f32,
        threshold: f32,
    ) -> Option<(usize, Uuid)> {
        let mut best: Option<(usize, Uuid, f32)> = None;

        for (ci, curve) in self.curves.iter().enumerate() {
            if !curve.visible {
                continue;
            }
            for key in &curve.keyframes {
                let screen = self.view.to_screen(key.time, key.value);
                let dx = screen[0] - screen_x;
                let dy = screen[1] - screen_y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < threshold {
                    if best.is_none() || dist < best.unwrap().2 {
                        best = Some((ci, key.id, dist));
                    }
                }
            }
        }

        best.map(|(ci, id, _)| (ci, id))
    }

    /// Collect render data for the curve editor (for the UI layer to display).
    pub fn render_data(&self) -> CurveEditorRenderData {
        let mut curve_samples = Vec::new();
        let mut keyframe_points = Vec::new();

        for (ci, curve) in self.curves.iter().enumerate() {
            if !curve.visible {
                continue;
            }

            // Sample the curve.
            let samples = curve.sample(
                self.view.time_range.0,
                self.view.time_range.1,
                200,
            );
            let screen_samples: Vec<[f32; 2]> = samples
                .iter()
                .map(|s| self.view.to_screen(s[0], s[1]))
                .collect();

            curve_samples.push(CurveSampleData {
                curve_index: ci,
                color: curve.color,
                points: screen_samples,
            });

            // Keyframe points.
            for key in &curve.keyframes {
                let screen = self.view.to_screen(key.time, key.value);
                keyframe_points.push(KeyframeRenderData {
                    curve_index: ci,
                    keyframe_id: key.id,
                    screen_pos: screen,
                    selected: self.selected_keyframes.contains(&key.id),
                    tangent_mode: key.tangent_mode,
                    color: curve.color,
                });
            }
        }

        CurveEditorRenderData {
            curve_samples,
            keyframe_points,
            box_select: self.box_select.as_ref().map(|bs| {
                let min_x = bs.start[0].min(bs.current[0]);
                let min_y = bs.start[1].min(bs.current[1]);
                let max_x = bs.start[0].max(bs.current[0]);
                let max_y = bs.start[1].max(bs.current[1]);
                ([min_x, min_y], [max_x, max_y])
            }),
        }
    }
}

/// Render data produced by the curve editor.
#[derive(Debug, Clone)]
pub struct CurveEditorRenderData {
    pub curve_samples: Vec<CurveSampleData>,
    pub keyframe_points: Vec<KeyframeRenderData>,
    pub box_select: Option<([f32; 2], [f32; 2])>,
}

/// Sampled curve data for rendering.
#[derive(Debug, Clone)]
pub struct CurveSampleData {
    pub curve_index: usize,
    pub color: [f32; 4],
    pub points: Vec<[f32; 2]>,
}

/// Keyframe data for rendering.
#[derive(Debug, Clone)]
pub struct KeyframeRenderData {
    pub curve_index: usize,
    pub keyframe_id: Uuid,
    pub screen_pos: [f32; 2],
    pub selected: bool,
    pub tangent_mode: TangentMode,
    pub color: [f32; 4],
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Snap a value to the nearest multiple of `increment`.
fn snap_to_increment(value: f32, increment: f32) -> f32 {
    if increment <= 0.0 {
        return value;
    }
    (value / increment).round() * increment
}

/// Compute a "nice" grid spacing for a given range and target number of divisions.
fn nice_grid_spacing(range: f32, target_divisions: f32) -> f32 {
    let raw = range / target_divisions.max(1.0);
    let magnitude = 10.0_f32.powf(raw.abs().log10().floor());
    let normalized = raw / magnitude;

    let nice = if normalized <= 1.5 {
        1.0
    } else if normalized <= 3.5 {
        2.0
    } else if normalized <= 7.5 {
        5.0
    } else {
        10.0
    };

    nice * magnitude
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_curve() -> AnimationCurve {
        let mut curve = AnimationCurve::new("Test", [1.0, 0.0, 0.0, 1.0]);
        curve.add_keyframe(Keyframe::new(0.0, 0.0));
        curve.add_keyframe(Keyframe::new(1.0, 1.0));
        curve.add_keyframe(Keyframe::new(2.0, 0.0));
        curve
    }

    #[test]
    fn create_curve() {
        let curve = AnimationCurve::new("Position.X", [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(curve.keyframe_count(), 0);
        assert!(curve.time_range().is_none());
    }

    #[test]
    fn add_keyframes_sorted() {
        let mut curve = AnimationCurve::new("Test", [1.0, 0.0, 0.0, 1.0]);
        curve.add_keyframe(Keyframe::new(2.0, 1.0));
        curve.add_keyframe(Keyframe::new(0.0, 0.0));
        curve.add_keyframe(Keyframe::new(1.0, 0.5));

        assert_eq!(curve.keyframe_count(), 3);
        assert!((curve.keyframes[0].time - 0.0).abs() < 1e-5);
        assert!((curve.keyframes[1].time - 1.0).abs() < 1e-5);
        assert!((curve.keyframes[2].time - 2.0).abs() < 1e-5);
    }

    #[test]
    fn remove_keyframe() {
        let mut curve = build_test_curve();
        let id = curve.keyframes[1].id;
        assert!(curve.remove_keyframe(id));
        assert_eq!(curve.keyframe_count(), 2);
        assert!(curve.find_keyframe(id).is_none());
    }

    #[test]
    fn evaluate_linear_curve() {
        let mut curve = AnimationCurve::new("Test", [1.0, 0.0, 0.0, 1.0]);
        curve.add_keyframe(Keyframe::new(0.0, 0.0).with_tangent_mode(TangentMode::Linear));
        curve.add_keyframe(Keyframe::new(1.0, 1.0).with_tangent_mode(TangentMode::Linear));

        let v0 = curve.evaluate(0.0);
        assert!((v0 - 0.0).abs() < 1e-4);

        let v_mid = curve.evaluate(0.5);
        assert!((v_mid - 0.5).abs() < 1e-4);

        let v1 = curve.evaluate(1.0);
        assert!((v1 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn evaluate_clamped_extrapolation() {
        let curve = build_test_curve();

        let before = curve.evaluate(-1.0);
        assert!((before - 0.0).abs() < 1e-4);

        let after = curve.evaluate(5.0);
        assert!((after - 0.0).abs() < 1e-4);
    }

    #[test]
    fn evaluate_constant_mode() {
        let mut curve = AnimationCurve::new("Test", [1.0, 0.0, 0.0, 1.0]);
        curve.add_keyframe(Keyframe::new(0.0, 0.0).with_tangent_mode(TangentMode::Constant));
        curve.add_keyframe(Keyframe::new(1.0, 1.0).with_tangent_mode(TangentMode::Constant));

        // Between 0 and 1, should hold at 0 (step function).
        let v = curve.evaluate(0.5);
        assert!((v - 0.0).abs() < 1e-4);
    }

    #[test]
    fn time_and_value_range() {
        let curve = build_test_curve();
        let (t_min, t_max) = curve.time_range().unwrap();
        assert!((t_min - 0.0).abs() < 1e-5);
        assert!((t_max - 2.0).abs() < 1e-5);

        let (v_min, v_max) = curve.value_range().unwrap();
        assert!((v_min - 0.0).abs() < 1e-5);
        assert!((v_max - 1.0).abs() < 1e-5);
    }

    #[test]
    fn move_keyframe() {
        let mut curve = build_test_curve();
        let id = curve.keyframes[1].id;
        curve.move_keyframe(id, 1.5, 2.0);

        let key = curve.find_keyframe(id).unwrap();
        assert!((key.time - 1.5).abs() < 1e-5);
        assert!((key.value - 2.0).abs() < 1e-5);
    }

    #[test]
    fn sample_curve() {
        let curve = build_test_curve();
        let samples = curve.sample(0.0, 2.0, 21);
        assert_eq!(samples.len(), 21);
        assert!((samples[0][0] - 0.0).abs() < 1e-5);
        assert!((samples[20][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn view_coordinate_conversion() {
        let view = CurveEditorView {
            time_range: (0.0, 10.0),
            value_range: (0.0, 1.0),
            canvas_width: 1000.0,
            canvas_height: 200.0,
            time_grid_spacing: 1.0,
            value_grid_spacing: 0.1,
        };

        let screen = view.to_screen(5.0, 0.5);
        assert!((screen[0] - 500.0).abs() < 1.0);
        assert!((screen[1] - 100.0).abs() < 1.0);

        let (time, value) = view.from_screen(500.0, 100.0);
        assert!((time - 5.0).abs() < 0.1);
        assert!((value - 0.5).abs() < 0.1);
    }

    #[test]
    fn editor_selection() {
        let mut editor = CurveEditor::new();
        editor.add_curve(build_test_curve());

        let id0 = editor.curves[0].keyframes[0].id;
        let id1 = editor.curves[0].keyframes[1].id;

        editor.select_keyframe(id0);
        assert!(editor.is_selected(id0));
        assert!(!editor.is_selected(id1));

        editor.add_to_selection(id1);
        assert!(editor.is_selected(id1));

        editor.toggle_selection(id0);
        assert!(!editor.is_selected(id0));

        editor.clear_selection();
        assert!(editor.selected_keyframes.is_empty());
    }

    #[test]
    fn editor_copy_paste() {
        let mut editor = CurveEditor::new();
        editor.add_curve(build_test_curve());

        let id0 = editor.curves[0].keyframes[0].id;
        let id1 = editor.curves[0].keyframes[1].id;

        editor.select_keyframe(id0);
        editor.add_to_selection(id1);
        editor.copy_selected();

        assert_eq!(editor.clipboard.len(), 2);

        editor.paste(0, 5.0);
        assert_eq!(editor.curves[0].keyframe_count(), 5);
    }

    #[test]
    fn editor_delete_selected() {
        let mut editor = CurveEditor::new();
        editor.add_curve(build_test_curve());

        let id = editor.curves[0].keyframes[1].id;
        editor.select_keyframe(id);
        editor.delete_selected();

        assert_eq!(editor.curves[0].keyframe_count(), 2);
        assert!(editor.selected_keyframes.is_empty());
    }

    #[test]
    fn editor_hit_test() {
        let mut editor = CurveEditor::new();
        editor.add_curve(build_test_curve());

        let key = &editor.curves[0].keyframes[0];
        let screen = editor.view.to_screen(key.time, key.value);

        let hit = editor.hit_test_keyframe(screen[0], screen[1], 10.0);
        assert!(hit.is_some());
    }

    #[test]
    fn nice_grid_spacing_values() {
        assert!((nice_grid_spacing(10.0, 10.0) - 1.0).abs() < 1e-5);
        assert!((nice_grid_spacing(100.0, 10.0) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn snap_to_increment_values() {
        assert!((snap_to_increment(0.3, 0.25) - 0.25).abs() < 1e-5);
        assert!((snap_to_increment(0.6, 0.25) - 0.5).abs() < 1e-5);
        assert!((snap_to_increment(0.9, 0.25) - 1.0).abs() < 1e-5);
    }
}
