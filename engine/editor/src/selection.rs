//! Selection system for the editor.
//!
//! Provides a rich selection model for the editor viewport, including:
//!
//! - **Single and multi-select** — click, Ctrl+click, Shift+click.
//! - **Box (rectangle) selection** — drag to select all entities within a
//!   screen-space rectangle.
//! - **Selection modes** — Object, Face, Edge, Vertex for mesh editing.
//! - **Selection transform** — compute the center, average rotation, and
//!   bounding box of the current selection.
//! - **Outliner** — highlight selected entities in the viewport.
//! - **Selection history** — remember recent selections for quick re-select.
//!
//! # Box Selection Math
//!
//! Box selection works by projecting entity positions from world space to
//! screen (NDC) space using the camera's view-projection matrix, then
//! testing whether the projected point falls within the screen-space
//! rectangle drawn by the user.
//!
//! For a more accurate approach, the system can also test entity bounding
//! boxes against the selection frustum (the 3D volume defined by the box
//! selection rectangle and the camera).

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// EntityHandle — lightweight entity reference for selection
// ---------------------------------------------------------------------------

/// Lightweight entity handle for the selection system. Decoupled from the
/// ECS entity type to keep this module self-contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityHandle {
    pub id: u64,
}

impl EntityHandle {
    pub fn new(id: u64) -> Self {
        Self { id }
    }
}

impl std::fmt::Display for EntityHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entity({})", self.id)
    }
}

// ---------------------------------------------------------------------------
// SelectionMode
// ---------------------------------------------------------------------------

/// Selection mode determines what granularity of objects can be selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectionMode {
    /// Select whole entities/objects.
    Object,
    /// Select individual mesh faces (for mesh editing).
    Face,
    /// Select mesh edges.
    Edge,
    /// Select mesh vertices.
    Vertex,
}

impl Default for SelectionMode {
    fn default() -> Self {
        Self::Object
    }
}

impl SelectionMode {
    /// All available modes.
    pub fn all() -> &'static [SelectionMode] {
        &[
            SelectionMode::Object,
            SelectionMode::Face,
            SelectionMode::Edge,
            SelectionMode::Vertex,
        ]
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            SelectionMode::Object => "Object",
            SelectionMode::Face => "Face",
            SelectionMode::Edge => "Edge",
            SelectionMode::Vertex => "Vertex",
        }
    }
}

// ---------------------------------------------------------------------------
// ScreenRect — 2D screen-space rectangle for box selection
// ---------------------------------------------------------------------------

/// A 2D rectangle in screen coordinates (pixels or NDC).
#[derive(Debug, Clone, Copy)]
pub struct ScreenRect {
    /// Minimum x (left edge).
    pub min_x: f32,
    /// Minimum y (top edge).
    pub min_y: f32,
    /// Maximum x (right edge).
    pub max_x: f32,
    /// Maximum y (bottom edge).
    pub max_y: f32,
}

impl ScreenRect {
    /// Create a rectangle from two corners (automatically normalized so
    /// min <= max).
    pub fn from_corners(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            min_x: x1.min(x2),
            min_y: y1.min(y2),
            max_x: x1.max(x2),
            max_y: y1.max(y2),
        }
    }

    /// Create from top-left corner and size.
    pub fn from_pos_size(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            min_x: x,
            min_y: y,
            max_x: x + width,
            max_y: y + height,
        }
    }

    /// Width of the rectangle.
    pub fn width(&self) -> f32 {
        self.max_x - self.min_x
    }

    /// Height of the rectangle.
    pub fn height(&self) -> f32 {
        self.max_y - self.min_y
    }

    /// Area of the rectangle.
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Center point.
    pub fn center(&self) -> (f32, f32) {
        (
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        )
    }

    /// Test if a point is inside the rectangle.
    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    /// Test if another rectangle overlaps this one.
    pub fn overlaps(&self, other: &ScreenRect) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    /// Compute the intersection of two rectangles. Returns None if they
    /// do not overlap.
    pub fn intersection(&self, other: &ScreenRect) -> Option<ScreenRect> {
        if !self.overlaps(other) {
            return None;
        }
        Some(ScreenRect {
            min_x: self.min_x.max(other.min_x),
            min_y: self.min_y.max(other.min_y),
            max_x: self.max_x.min(other.max_x),
            max_y: self.max_y.min(other.max_y),
        })
    }

    /// Test if the rectangle is degenerate (zero or negative area).
    pub fn is_degenerate(&self) -> bool {
        self.width() <= 0.0 || self.height() <= 0.0
    }

    /// Expand the rectangle to include a point.
    pub fn expand_to_include(&mut self, x: f32, y: f32) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }

    /// Grow the rectangle by a margin on all sides.
    pub fn inflate(&self, margin: f32) -> ScreenRect {
        ScreenRect {
            min_x: self.min_x - margin,
            min_y: self.min_y - margin,
            max_x: self.max_x + margin,
            max_y: self.max_y + margin,
        }
    }
}

// ---------------------------------------------------------------------------
// CameraProjection — minimal camera data for box selection
// ---------------------------------------------------------------------------

/// Minimal camera data needed for projecting world positions to screen space.
#[derive(Debug, Clone)]
pub struct CameraProjection {
    /// Combined view-projection matrix (4x4, column-major).
    pub view_projection: [f32; 16],
    /// Viewport width in pixels.
    pub viewport_width: f32,
    /// Viewport height in pixels.
    pub viewport_height: f32,
}

impl CameraProjection {
    /// Create a new camera projection.
    pub fn new(view_projection: [f32; 16], viewport_width: f32, viewport_height: f32) -> Self {
        Self {
            view_projection,
            viewport_width,
            viewport_height,
        }
    }

    /// Project a world-space point to screen-space (pixel coordinates).
    ///
    /// Returns `None` if the point is behind the camera (w <= 0).
    ///
    /// The math:
    /// 1. Multiply world position by view-projection matrix to get clip space.
    /// 2. Perform perspective divide to get NDC (-1 to 1).
    /// 3. Map NDC to pixel coordinates.
    pub fn world_to_screen(&self, world_x: f32, world_y: f32, world_z: f32) -> Option<(f32, f32)> {
        let m = &self.view_projection;

        // Multiply by 4x4 matrix (column-major).
        let clip_x = m[0] * world_x + m[4] * world_y + m[8] * world_z + m[12];
        let clip_y = m[1] * world_x + m[5] * world_y + m[9] * world_z + m[13];
        let clip_z = m[2] * world_x + m[6] * world_y + m[10] * world_z + m[14];
        let clip_w = m[3] * world_x + m[7] * world_y + m[11] * world_z + m[15];

        // Point is behind the camera.
        if clip_w <= 0.0001 {
            return None;
        }

        // Perspective divide: clip -> NDC.
        let ndc_x = clip_x / clip_w;
        let ndc_y = clip_y / clip_w;

        // NDC to screen coordinates.
        // NDC x: -1 (left) to +1 (right) -> pixel: 0 to viewport_width
        // NDC y: -1 (bottom) to +1 (top) -> pixel: viewport_height to 0
        //   (screen y is typically top-down)
        let screen_x = (ndc_x + 1.0) * 0.5 * self.viewport_width;
        let screen_y = (1.0 - ndc_y) * 0.5 * self.viewport_height;

        Some((screen_x, screen_y))
    }

    /// Test if a world-space point falls within a screen-space rectangle.
    pub fn point_in_rect(
        &self,
        world_x: f32,
        world_y: f32,
        world_z: f32,
        rect: &ScreenRect,
    ) -> bool {
        if let Some((sx, sy)) = self.world_to_screen(world_x, world_y, world_z) {
            rect.contains_point(sx, sy)
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// SelectionTransform — computed transform of the selection group
// ---------------------------------------------------------------------------

/// Computed transform properties of the current selection.
///
/// This is recomputed whenever the selection changes and is used by the
/// gizmo system to position and orient the manipulation handles.
#[derive(Debug, Clone)]
pub struct SelectionTransform {
    /// Center position (average of all selected entity positions).
    pub center: [f32; 3],
    /// Average rotation (quaternion). For mixed rotations this is an
    /// approximation.
    pub average_rotation: [f32; 4],
    /// Axis-aligned bounding box of all selected entities.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    /// Number of entities in the selection.
    pub count: usize,
}

impl SelectionTransform {
    /// Compute the selection transform from entity positions and rotations.
    ///
    /// `entities` is a list of `(position, rotation_quat)` tuples.
    pub fn compute(entities: &[([f32; 3], [f32; 4])]) -> Self {
        if entities.is_empty() {
            return Self {
                center: [0.0; 3],
                average_rotation: [0.0, 0.0, 0.0, 1.0],
                bounds_min: [0.0; 3],
                bounds_max: [0.0; 3],
                count: 0,
            };
        }

        let count = entities.len();

        // Compute center (average position).
        let mut sum = [0.0f32; 3];
        for (pos, _) in entities {
            sum[0] += pos[0];
            sum[1] += pos[1];
            sum[2] += pos[2];
        }
        let center = [
            sum[0] / count as f32,
            sum[1] / count as f32,
            sum[2] / count as f32,
        ];

        // Compute bounds.
        let mut bounds_min = entities[0].0;
        let mut bounds_max = entities[0].0;
        for (pos, _) in entities {
            bounds_min[0] = bounds_min[0].min(pos[0]);
            bounds_min[1] = bounds_min[1].min(pos[1]);
            bounds_min[2] = bounds_min[2].min(pos[2]);
            bounds_max[0] = bounds_max[0].max(pos[0]);
            bounds_max[1] = bounds_max[1].max(pos[1]);
            bounds_max[2] = bounds_max[2].max(pos[2]);
        }

        // Compute average rotation.
        // For a proper average of quaternions, we accumulate and normalize.
        // This is an approximation; true quaternion averaging requires
        // eigenvalue decomposition.
        let mut quat_sum = [0.0f32; 4];
        let first_quat = entities[0].1;

        for (_, quat) in entities {
            // Ensure quaternions are in the same hemisphere (dot product > 0).
            let dot = quat[0] * first_quat[0]
                + quat[1] * first_quat[1]
                + quat[2] * first_quat[2]
                + quat[3] * first_quat[3];

            let sign = if dot < 0.0 { -1.0 } else { 1.0 };
            quat_sum[0] += quat[0] * sign;
            quat_sum[1] += quat[1] * sign;
            quat_sum[2] += quat[2] * sign;
            quat_sum[3] += quat[3] * sign;
        }

        // Normalize the averaged quaternion.
        let len = (quat_sum[0] * quat_sum[0]
            + quat_sum[1] * quat_sum[1]
            + quat_sum[2] * quat_sum[2]
            + quat_sum[3] * quat_sum[3])
            .sqrt();

        let average_rotation = if len > 1e-6 {
            [
                quat_sum[0] / len,
                quat_sum[1] / len,
                quat_sum[2] / len,
                quat_sum[3] / len,
            ]
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };

        Self {
            center,
            average_rotation,
            bounds_min,
            bounds_max,
            count,
        }
    }

    /// Size of the bounding box.
    pub fn bounds_size(&self) -> [f32; 3] {
        [
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2],
        ]
    }

    /// Diagonal length of the bounding box.
    pub fn bounds_diagonal(&self) -> f32 {
        let s = self.bounds_size();
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Outliner — highlight style for selected entities
// ---------------------------------------------------------------------------

/// Visual highlighting configuration for selected entities.
#[derive(Debug, Clone)]
pub struct Outliner {
    /// Outline color (RGBA).
    pub color: [f32; 4],
    /// Outline width in pixels.
    pub width: f32,
    /// Whether to render a filled overlay.
    pub fill_overlay: bool,
    /// Fill overlay color (RGBA, typically low alpha).
    pub fill_color: [f32; 4],
    /// Whether to pulse the outline (animated selection indicator).
    pub pulse: bool,
    /// Pulse speed (cycles per second).
    pub pulse_speed: f32,
    /// Whether to show outlines through occluders (x-ray mode).
    pub xray: bool,
}

impl Default for Outliner {
    fn default() -> Self {
        Self {
            color: [1.0, 0.6, 0.0, 1.0], // Orange.
            width: 2.0,
            fill_overlay: true,
            fill_color: [1.0, 0.6, 0.0, 0.1],
            pulse: false,
            pulse_speed: 1.0,
            xray: false,
        }
    }
}

impl Outliner {
    /// Create a new outliner with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the outline color.
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Set the outline width.
    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Compute the current outline alpha considering pulse animation.
    pub fn current_alpha(&self, time: f32) -> f32 {
        if self.pulse {
            let t = (time * self.pulse_speed * std::f32::consts::TAU).sin();
            let alpha_min = 0.5;
            let alpha_max = 1.0;
            alpha_min + (alpha_max - alpha_min) * (t * 0.5 + 0.5)
        } else {
            self.color[3]
        }
    }
}

// ---------------------------------------------------------------------------
// SelectionHistory — remember recent selections for quick re-select
// ---------------------------------------------------------------------------

/// Remembers recent selections so the user can quickly cycle back to
/// previous selections.
#[derive(Debug, Clone)]
pub struct SelectionHistory {
    /// History ring buffer.
    entries: VecDeque<Vec<EntityHandle>>,
    /// Maximum number of history entries.
    max_entries: usize,
    /// Current position in the history.
    cursor: usize,
}

impl SelectionHistory {
    /// Create a new history with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            cursor: 0,
        }
    }

    /// Record a selection. Entries after the cursor are discarded.
    pub fn push(&mut self, selection: Vec<EntityHandle>) {
        // Don't record if identical to the current entry.
        if let Some(current) = self.entries.back() {
            if *current == selection {
                return;
            }
        }

        // Trim entries after cursor (forward history discarded on new selection).
        while self.entries.len() > self.cursor + 1 && !self.entries.is_empty() {
            self.entries.pop_back();
        }

        self.entries.push_back(selection);
        self.cursor = self.entries.len().saturating_sub(1);

        // Enforce max size.
        while self.entries.len() > self.max_entries {
            self.entries.pop_front();
            self.cursor = self.cursor.saturating_sub(1);
        }
    }

    /// Go back to the previous selection.
    pub fn go_back(&mut self) -> Option<&[EntityHandle]> {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.entries.get(self.cursor).map(|v| v.as_slice())
        } else {
            None
        }
    }

    /// Go forward to the next selection.
    pub fn go_forward(&mut self) -> Option<&[EntityHandle]> {
        if self.cursor + 1 < self.entries.len() {
            self.cursor += 1;
            self.entries.get(self.cursor).map(|v| v.as_slice())
        } else {
            None
        }
    }

    /// Whether we can go back.
    pub fn can_go_back(&self) -> bool {
        self.cursor > 0
    }

    /// Whether we can go forward.
    pub fn can_go_forward(&self) -> bool {
        self.cursor + 1 < self.entries.len()
    }

    /// Current entry.
    pub fn current(&self) -> Option<&[EntityHandle]> {
        self.entries.get(self.cursor).map(|v| v.as_slice())
    }

    /// Number of history entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the history.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.cursor = 0;
    }
}

impl Default for SelectionHistory {
    fn default() -> Self {
        Self::new(32)
    }
}

// ---------------------------------------------------------------------------
// Selection — the main selection state
// ---------------------------------------------------------------------------

/// The main selection state for the editor.
///
/// Manages which entities are selected, the selection mode, visual
/// highlighting, and history.
pub struct Selection {
    /// Currently selected entities.
    selected: HashSet<EntityHandle>,
    /// Ordered list preserving selection order (for Shift+click range select).
    ordered: Vec<EntityHandle>,
    /// Current selection mode.
    pub mode: SelectionMode,
    /// Visual outliner settings.
    pub outliner: Outliner,
    /// Selection history for back/forward navigation.
    pub history: SelectionHistory,
    /// Cached selection transform.
    cached_transform: Option<SelectionTransform>,
    /// Whether the cached transform needs recomputation.
    transform_dirty: bool,
    /// The "anchor" entity for shift-click range selection.
    anchor: Option<EntityHandle>,
}

impl Selection {
    /// Create a new empty selection.
    pub fn new() -> Self {
        Self {
            selected: HashSet::new(),
            ordered: Vec::new(),
            mode: SelectionMode::Object,
            outliner: Outliner::default(),
            history: SelectionHistory::default(),
            cached_transform: None,
            transform_dirty: true,
            anchor: None,
        }
    }

    // -- Single / multi-select --------------------------------------------------

    /// Select a single entity, deselecting all others.
    pub fn select(&mut self, entity: EntityHandle) {
        self.selected.clear();
        self.ordered.clear();
        self.selected.insert(entity);
        self.ordered.push(entity);
        self.anchor = Some(entity);
        self.transform_dirty = true;
        self.record_history();
    }

    /// Add an entity to the selection (Ctrl+click).
    pub fn add(&mut self, entity: EntityHandle) {
        if self.selected.insert(entity) {
            self.ordered.push(entity);
        }
        self.anchor = Some(entity);
        self.transform_dirty = true;
        self.record_history();
    }

    /// Toggle an entity in/out of the selection (Ctrl+click).
    pub fn toggle(&mut self, entity: EntityHandle) {
        if self.selected.contains(&entity) {
            self.selected.remove(&entity);
            self.ordered.retain(|e| *e != entity);
        } else {
            self.selected.insert(entity);
            self.ordered.push(entity);
            self.anchor = Some(entity);
        }
        self.transform_dirty = true;
        self.record_history();
    }

    /// Deselect a specific entity.
    pub fn deselect(&mut self, entity: EntityHandle) {
        self.selected.remove(&entity);
        self.ordered.retain(|e| *e != entity);
        self.transform_dirty = true;
        self.record_history();
    }

    /// Clear the selection entirely.
    pub fn clear(&mut self) {
        if self.selected.is_empty() {
            return;
        }
        self.selected.clear();
        self.ordered.clear();
        self.anchor = None;
        self.cached_transform = None;
        self.transform_dirty = true;
        self.record_history();
    }

    /// Select multiple entities at once, replacing the current selection.
    pub fn select_all(&mut self, entities: &[EntityHandle]) {
        self.selected.clear();
        self.ordered.clear();
        for &e in entities {
            if self.selected.insert(e) {
                self.ordered.push(e);
            }
        }
        self.anchor = entities.last().copied();
        self.transform_dirty = true;
        self.record_history();
    }

    // -- Box selection ----------------------------------------------------------

    /// Perform a box (rectangle) selection.
    ///
    /// Takes a screen-space rectangle and a camera projection, then tests
    /// each candidate entity's world position against the rectangle.
    ///
    /// `entity_positions` is a list of `(EntityHandle, world_x, world_y, world_z)`.
    ///
    /// Returns the list of entities that fall within the rectangle.
    pub fn select_rect(
        &mut self,
        rect: &ScreenRect,
        camera: &CameraProjection,
        entity_positions: &[(EntityHandle, f32, f32, f32)],
    ) -> Vec<EntityHandle> {
        let mut hits = Vec::new();

        for &(entity, wx, wy, wz) in entity_positions {
            if camera.point_in_rect(wx, wy, wz, rect) {
                hits.push(entity);
            }
        }

        // Replace current selection with box-selected entities.
        self.select_all(&hits);

        hits
    }

    /// Perform an additive box selection (Shift+drag): add entities in the
    /// rectangle to the existing selection.
    pub fn select_rect_additive(
        &mut self,
        rect: &ScreenRect,
        camera: &CameraProjection,
        entity_positions: &[(EntityHandle, f32, f32, f32)],
    ) -> Vec<EntityHandle> {
        let mut hits = Vec::new();

        for &(entity, wx, wy, wz) in entity_positions {
            if camera.point_in_rect(wx, wy, wz, rect) {
                hits.push(entity);
                self.add(entity);
            }
        }

        hits
    }

    // -- Query ------------------------------------------------------------------

    /// Whether the given entity is selected.
    pub fn is_selected(&self, entity: EntityHandle) -> bool {
        self.selected.contains(&entity)
    }

    /// Get all selected entities as a slice (preserving selection order).
    pub fn entities(&self) -> &[EntityHandle] {
        &self.ordered
    }

    /// Get all selected entities as a set.
    pub fn entity_set(&self) -> &HashSet<EntityHandle> {
        &self.selected
    }

    /// Number of selected entities.
    pub fn count(&self) -> usize {
        self.selected.len()
    }

    /// Whether the selection is empty.
    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    /// Get the first selected entity (useful for single-selection contexts).
    pub fn primary(&self) -> Option<EntityHandle> {
        self.ordered.first().copied()
    }

    /// Get the last selected entity (anchor for Shift+click).
    pub fn anchor(&self) -> Option<EntityHandle> {
        self.anchor
    }

    // -- Selection transform ----------------------------------------------------

    /// Compute the selection transform from entity world positions.
    ///
    /// Call this to update the cached transform when entity positions change.
    pub fn compute_transform(&mut self, entity_data: &[(EntityHandle, [f32; 3], [f32; 4])]) {
        let relevant: Vec<([f32; 3], [f32; 4])> = entity_data
            .iter()
            .filter(|(e, _, _)| self.selected.contains(e))
            .map(|(_, pos, rot)| (*pos, *rot))
            .collect();

        self.cached_transform = Some(SelectionTransform::compute(&relevant));
        self.transform_dirty = false;
    }

    /// Get the cached selection transform, if computed.
    pub fn transform(&self) -> Option<&SelectionTransform> {
        self.cached_transform.as_ref()
    }

    /// Whether the transform needs recomputation.
    pub fn is_transform_dirty(&self) -> bool {
        self.transform_dirty
    }

    /// Mark the transform as needing recomputation.
    pub fn invalidate_transform(&mut self) {
        self.transform_dirty = true;
    }

    // -- History ----------------------------------------------------------------

    fn record_history(&mut self) {
        self.history.push(self.ordered.clone());
    }

    /// Restore the previous selection from history.
    pub fn history_back(&mut self) -> bool {
        if let Some(prev) = self.history.go_back() {
            let prev = prev.to_vec();
            self.selected.clear();
            self.ordered.clear();
            for e in &prev {
                self.selected.insert(*e);
                self.ordered.push(*e);
            }
            self.transform_dirty = true;
            true
        } else {
            false
        }
    }

    /// Restore the next selection from history.
    pub fn history_forward(&mut self) -> bool {
        if let Some(next) = self.history.go_forward() {
            let next = next.to_vec();
            self.selected.clear();
            self.ordered.clear();
            for e in &next {
                self.selected.insert(*e);
                self.ordered.push(*e);
            }
            self.transform_dirty = true;
            true
        } else {
            false
        }
    }
}

impl Default for Selection {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Selection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Selection")
            .field("count", &self.selected.len())
            .field("mode", &self.mode)
            .field("transform_dirty", &self.transform_dirty)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn e(id: u64) -> EntityHandle {
        EntityHandle::new(id)
    }

    // -- ScreenRect tests -------------------------------------------------------

    #[test]
    fn screen_rect_from_corners() {
        let r = ScreenRect::from_corners(10.0, 20.0, 5.0, 15.0);
        assert_eq!(r.min_x, 5.0);
        assert_eq!(r.min_y, 15.0);
        assert_eq!(r.max_x, 10.0);
        assert_eq!(r.max_y, 20.0);
    }

    #[test]
    fn screen_rect_dimensions() {
        let r = ScreenRect::from_corners(0.0, 0.0, 10.0, 20.0);
        assert_eq!(r.width(), 10.0);
        assert_eq!(r.height(), 20.0);
        assert_eq!(r.area(), 200.0);
    }

    #[test]
    fn screen_rect_center() {
        let r = ScreenRect::from_corners(0.0, 0.0, 10.0, 20.0);
        let (cx, cy) = r.center();
        assert!((cx - 5.0).abs() < 1e-5);
        assert!((cy - 10.0).abs() < 1e-5);
    }

    #[test]
    fn screen_rect_contains_point() {
        let r = ScreenRect::from_corners(10.0, 10.0, 50.0, 50.0);
        assert!(r.contains_point(25.0, 25.0));
        assert!(r.contains_point(10.0, 10.0)); // Edge.
        assert!(r.contains_point(50.0, 50.0)); // Edge.
        assert!(!r.contains_point(5.0, 25.0));
        assert!(!r.contains_point(55.0, 25.0));
    }

    #[test]
    fn screen_rect_overlaps() {
        let a = ScreenRect::from_corners(0.0, 0.0, 10.0, 10.0);
        let b = ScreenRect::from_corners(5.0, 5.0, 15.0, 15.0);
        let c = ScreenRect::from_corners(20.0, 20.0, 30.0, 30.0);

        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn screen_rect_intersection() {
        let a = ScreenRect::from_corners(0.0, 0.0, 10.0, 10.0);
        let b = ScreenRect::from_corners(5.0, 5.0, 15.0, 15.0);

        let inter = a.intersection(&b).unwrap();
        assert!((inter.min_x - 5.0).abs() < 1e-5);
        assert!((inter.min_y - 5.0).abs() < 1e-5);
        assert!((inter.max_x - 10.0).abs() < 1e-5);
        assert!((inter.max_y - 10.0).abs() < 1e-5);
    }

    #[test]
    fn screen_rect_no_intersection() {
        let a = ScreenRect::from_corners(0.0, 0.0, 5.0, 5.0);
        let b = ScreenRect::from_corners(10.0, 10.0, 15.0, 15.0);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn screen_rect_inflate() {
        let r = ScreenRect::from_corners(10.0, 10.0, 20.0, 20.0);
        let inflated = r.inflate(5.0);
        assert!((inflated.min_x - 5.0).abs() < 1e-5);
        assert!((inflated.max_x - 25.0).abs() < 1e-5);
    }

    #[test]
    fn screen_rect_expand_to_include() {
        let mut r = ScreenRect::from_corners(5.0, 5.0, 10.0, 10.0);
        r.expand_to_include(0.0, 15.0);
        assert!((r.min_x - 0.0).abs() < 1e-5);
        assert!((r.max_y - 15.0).abs() < 1e-5);
    }

    // -- CameraProjection tests -------------------------------------------------

    #[test]
    fn camera_projection_identity() {
        // Using an identity matrix as the view-projection.
        // Points at (0,0,0) should project to center of screen.
        #[rustfmt::skip]
        let vp = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let cam = CameraProjection::new(vp, 800.0, 600.0);
        let result = cam.world_to_screen(0.0, 0.0, 0.0);
        assert!(result.is_some());
        let (sx, sy) = result.unwrap();
        assert!((sx - 400.0).abs() < 1e-3);
        assert!((sy - 300.0).abs() < 1e-3);
    }

    #[test]
    fn camera_projection_offcenter() {
        #[rustfmt::skip]
        let vp = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let cam = CameraProjection::new(vp, 800.0, 600.0);

        // Point at (1, 1, 0) with identity VP should map to NDC (1, 1).
        // Screen: x = (1+1)/2 * 800 = 800, y = (1-1)/2 * 600 = 0
        let result = cam.world_to_screen(1.0, 1.0, 0.0);
        assert!(result.is_some());
        let (sx, sy) = result.unwrap();
        assert!((sx - 800.0).abs() < 1e-3);
        assert!((sy - 0.0).abs() < 1e-3);
    }

    #[test]
    fn camera_point_in_rect() {
        #[rustfmt::skip]
        let vp = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let cam = CameraProjection::new(vp, 800.0, 600.0);
        let rect = ScreenRect::from_corners(350.0, 250.0, 450.0, 350.0);

        // Origin should project to (400, 300) which is inside the rect.
        assert!(cam.point_in_rect(0.0, 0.0, 0.0, &rect));

        // Point at (1, 1, 0) projects to (800, 0) which is outside.
        assert!(!cam.point_in_rect(1.0, 1.0, 0.0, &rect));
    }

    // -- SelectionTransform tests -----------------------------------------------

    #[test]
    fn selection_transform_empty() {
        let st = SelectionTransform::compute(&[]);
        assert_eq!(st.count, 0);
        assert_eq!(st.center, [0.0; 3]);
    }

    #[test]
    fn selection_transform_single() {
        let entities = vec![
            ([5.0, 10.0, 15.0], [0.0, 0.0, 0.0, 1.0]),
        ];
        let st = SelectionTransform::compute(&entities);
        assert_eq!(st.count, 1);
        assert!((st.center[0] - 5.0).abs() < 1e-5);
        assert!((st.center[1] - 10.0).abs() < 1e-5);
        assert!((st.center[2] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn selection_transform_center() {
        let entities = vec![
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            ([10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            ([0.0, 10.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ];
        let st = SelectionTransform::compute(&entities);
        assert_eq!(st.count, 3);

        // Center should be average: (10/3, 10/3, 0)
        assert!((st.center[0] - 10.0 / 3.0).abs() < 1e-4);
        assert!((st.center[1] - 10.0 / 3.0).abs() < 1e-4);
        assert!((st.center[2] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn selection_transform_bounds() {
        let entities = vec![
            ([-5.0, 0.0, -3.0], [0.0, 0.0, 0.0, 1.0]),
            ([10.0, 8.0, 7.0], [0.0, 0.0, 0.0, 1.0]),
        ];
        let st = SelectionTransform::compute(&entities);

        assert!((st.bounds_min[0] - (-5.0)).abs() < 1e-5);
        assert!((st.bounds_min[1] - 0.0).abs() < 1e-5);
        assert!((st.bounds_min[2] - (-3.0)).abs() < 1e-5);

        assert!((st.bounds_max[0] - 10.0).abs() < 1e-5);
        assert!((st.bounds_max[1] - 8.0).abs() < 1e-5);
        assert!((st.bounds_max[2] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn selection_transform_bounds_size() {
        let entities = vec![
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            ([10.0, 20.0, 30.0], [0.0, 0.0, 0.0, 1.0]),
        ];
        let st = SelectionTransform::compute(&entities);
        let size = st.bounds_size();
        assert!((size[0] - 10.0).abs() < 1e-5);
        assert!((size[1] - 20.0).abs() < 1e-5);
        assert!((size[2] - 30.0).abs() < 1e-5);
    }

    // -- Selection tests --------------------------------------------------------

    #[test]
    fn select_single() {
        let mut sel = Selection::new();
        sel.select(e(1));
        assert!(sel.is_selected(e(1)));
        assert!(!sel.is_selected(e(2)));
        assert_eq!(sel.count(), 1);
    }

    #[test]
    fn select_replaces() {
        let mut sel = Selection::new();
        sel.select(e(1));
        sel.select(e(2));
        assert!(!sel.is_selected(e(1)));
        assert!(sel.is_selected(e(2)));
        assert_eq!(sel.count(), 1);
    }

    #[test]
    fn add_to_selection() {
        let mut sel = Selection::new();
        sel.select(e(1));
        sel.add(e(2));
        sel.add(e(3));
        assert_eq!(sel.count(), 3);
        assert!(sel.is_selected(e(1)));
        assert!(sel.is_selected(e(2)));
        assert!(sel.is_selected(e(3)));
    }

    #[test]
    fn add_duplicate_noop() {
        let mut sel = Selection::new();
        sel.select(e(1));
        sel.add(e(1));
        assert_eq!(sel.count(), 1);
    }

    #[test]
    fn toggle_selection() {
        let mut sel = Selection::new();
        sel.toggle(e(1));
        assert!(sel.is_selected(e(1)));

        sel.toggle(e(1));
        assert!(!sel.is_selected(e(1)));
        assert!(sel.is_empty());
    }

    #[test]
    fn deselect() {
        let mut sel = Selection::new();
        sel.select(e(1));
        sel.add(e(2));
        sel.deselect(e(1));
        assert!(!sel.is_selected(e(1)));
        assert!(sel.is_selected(e(2)));
        assert_eq!(sel.count(), 1);
    }

    #[test]
    fn clear_selection() {
        let mut sel = Selection::new();
        sel.select(e(1));
        sel.add(e(2));
        sel.clear();
        assert!(sel.is_empty());
        assert_eq!(sel.count(), 0);
    }

    #[test]
    fn select_all() {
        let mut sel = Selection::new();
        sel.select_all(&[e(1), e(2), e(3)]);
        assert_eq!(sel.count(), 3);
        assert_eq!(sel.entities(), &[e(1), e(2), e(3)]);
    }

    #[test]
    fn primary_entity() {
        let mut sel = Selection::new();
        assert!(sel.primary().is_none());

        sel.select(e(5));
        assert_eq!(sel.primary(), Some(e(5)));

        sel.add(e(10));
        assert_eq!(sel.primary(), Some(e(5)));
    }

    #[test]
    fn selection_preserves_order() {
        let mut sel = Selection::new();
        sel.select(e(3));
        sel.add(e(1));
        sel.add(e(2));
        assert_eq!(sel.entities(), &[e(3), e(1), e(2)]);
    }

    // -- Box selection tests ----------------------------------------------------

    #[test]
    fn box_selection() {
        let mut sel = Selection::new();

        #[rustfmt::skip]
        let vp = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let cam = CameraProjection::new(vp, 800.0, 600.0);

        // Entity at (0,0,0) projects to (400, 300) — center of screen.
        // Entity at (0.5, 0.5, 0) projects to (600, 150).
        let entities = vec![
            (e(1), 0.0f32, 0.0f32, 0.0f32),
            (e(2), 0.5, 0.5, 0.0),
            (e(3), -1.0, -1.0, 0.0),
        ];

        // Selection rect around center.
        let rect = ScreenRect::from_corners(350.0, 250.0, 450.0, 350.0);
        let hits = sel.select_rect(&rect, &cam, &entities);

        // Only entity at origin should be selected.
        assert_eq!(hits.len(), 1);
        assert!(sel.is_selected(e(1)));
        assert!(!sel.is_selected(e(2)));
    }

    #[test]
    fn box_selection_additive() {
        let mut sel = Selection::new();
        sel.select(e(10)); // Pre-existing selection.

        #[rustfmt::skip]
        let vp = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let cam = CameraProjection::new(vp, 800.0, 600.0);

        let entities = vec![
            (e(1), 0.0f32, 0.0f32, 0.0f32),
        ];

        let rect = ScreenRect::from_corners(350.0, 250.0, 450.0, 350.0);
        sel.select_rect_additive(&rect, &cam, &entities);

        // Both pre-existing and new should be selected.
        assert!(sel.is_selected(e(10)));
        assert!(sel.is_selected(e(1)));
    }

    // -- Selection mode tests ---------------------------------------------------

    #[test]
    fn selection_modes() {
        assert_eq!(SelectionMode::default(), SelectionMode::Object);
        assert_eq!(SelectionMode::all().len(), 4);
        assert_eq!(SelectionMode::Object.label(), "Object");
        assert_eq!(SelectionMode::Face.label(), "Face");
        assert_eq!(SelectionMode::Edge.label(), "Edge");
        assert_eq!(SelectionMode::Vertex.label(), "Vertex");
    }

    // -- Outliner tests ---------------------------------------------------------

    #[test]
    fn outliner_defaults() {
        let o = Outliner::new();
        assert_eq!(o.width, 2.0);
        assert!(!o.pulse);
    }

    #[test]
    fn outliner_pulse_alpha() {
        let o = Outliner {
            pulse: true,
            pulse_speed: 1.0,
            ..Default::default()
        };

        let a1 = o.current_alpha(0.0);
        let a2 = o.current_alpha(0.25);
        // At t=0, sin(0) = 0, alpha = 0.5 + 0.5 * 0.5 = 0.75
        assert!((a1 - 0.75).abs() < 1e-3);
        // They should differ because of the pulse.
        assert!((a1 - a2).abs() > 0.01);
    }

    #[test]
    fn outliner_no_pulse_constant() {
        let o = Outliner::default();
        assert_eq!(o.current_alpha(0.0), o.color[3]);
        assert_eq!(o.current_alpha(1.0), o.color[3]);
    }

    // -- SelectionHistory tests -------------------------------------------------

    #[test]
    fn history_push_and_navigate() {
        let mut hist = SelectionHistory::new(10);

        hist.push(vec![e(1)]);
        hist.push(vec![e(1), e(2)]);
        hist.push(vec![e(3)]);

        assert_eq!(hist.len(), 3);
        assert_eq!(hist.current().unwrap(), &[e(3)]);

        let prev = hist.go_back().unwrap();
        assert_eq!(prev, &[e(1), e(2)]);

        let next = hist.go_forward().unwrap();
        assert_eq!(next, &[e(3)]);
    }

    #[test]
    fn history_max_size() {
        let mut hist = SelectionHistory::new(3);

        for i in 0..5 {
            hist.push(vec![e(i)]);
        }

        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn history_duplicate_ignored() {
        let mut hist = SelectionHistory::new(10);
        hist.push(vec![e(1)]);
        hist.push(vec![e(1)]); // Same as current, should be ignored.
        assert_eq!(hist.len(), 1);
    }

    #[test]
    fn history_clear() {
        let mut hist = SelectionHistory::new(10);
        hist.push(vec![e(1)]);
        hist.push(vec![e(2)]);
        hist.clear();
        assert!(hist.is_empty());
        assert!(!hist.can_go_back());
        assert!(!hist.can_go_forward());
    }

    #[test]
    fn history_new_push_discards_forward() {
        let mut hist = SelectionHistory::new(10);
        hist.push(vec![e(1)]);
        hist.push(vec![e(2)]);
        hist.push(vec![e(3)]);

        hist.go_back(); // cursor at [e(2)]
        hist.go_back(); // cursor at [e(1)]

        // New push should discard entries after cursor.
        hist.push(vec![e(4)]);
        assert!(!hist.can_go_forward());
        assert_eq!(hist.current().unwrap(), &[e(4)]);
    }

    // -- SelectionTransform quaternion averaging --------------------------------

    #[test]
    fn selection_transform_quat_average_same() {
        let quat = [0.0, 0.0, 0.0, 1.0]; // Identity.
        let entities = vec![
            ([0.0, 0.0, 0.0], quat),
            ([1.0, 0.0, 0.0], quat),
        ];
        let st = SelectionTransform::compute(&entities);
        // Average of identical quats should be the same quat.
        assert!((st.average_rotation[3] - 1.0).abs() < 1e-4);
    }

    // -- Integration test: compute_transform ------------------------------------

    #[test]
    fn selection_compute_transform() {
        let mut sel = Selection::new();
        sel.select_all(&[e(1), e(2)]);

        let entity_data = vec![
            (e(1), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            (e(2), [10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            (e(3), [100.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), // Not selected.
        ];

        sel.compute_transform(&entity_data);
        let transform = sel.transform().unwrap();

        assert_eq!(transform.count, 2);
        assert!((transform.center[0] - 5.0).abs() < 1e-5);
        assert!(!sel.is_transform_dirty());
    }

    // -- EntityHandle display ---------------------------------------------------

    #[test]
    fn entity_handle_display() {
        assert_eq!(format!("{}", e(42)), "Entity(42)");
    }
}
