// =============================================================================
// Genovo Engine - Scene View Features
// =============================================================================
//
// Extended scene view functionality: camera bookmarks, snap tools,
// alignment helpers, measurement, multi-viewport layouts, and statistics
// overlays.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Camera Bookmark
// ---------------------------------------------------------------------------

/// A saved camera position/orientation that can be restored quickly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraBookmark {
    /// Unique identifier.
    pub id: Uuid,
    /// Display name.
    pub name: String,
    /// Camera position.
    pub position: [f32; 3],
    /// Camera look-at target.
    pub target: [f32; 3],
    /// Yaw angle.
    pub yaw: f32,
    /// Pitch angle.
    pub pitch: f32,
    /// Orbit distance (or fly-mode distance from target).
    pub distance: f32,
    /// Orthographic size (if applicable).
    pub ortho_size: f32,
    /// Whether this was a perspective or ortho camera.
    pub is_perspective: bool,
    /// Keyboard shortcut (F1-F8 slots).
    pub slot: Option<u8>,
}

impl CameraBookmark {
    /// Create a new bookmark from current camera state.
    pub fn new(
        name: impl Into<String>,
        position: [f32; 3],
        target: [f32; 3],
        yaw: f32,
        pitch: f32,
        distance: f32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            position,
            target,
            yaw,
            pitch,
            distance,
            ortho_size: 10.0,
            is_perspective: true,
            slot: None,
        }
    }
}

/// Manager for camera bookmarks.
#[derive(Debug, Clone, Default)]
pub struct BookmarkManager {
    /// Saved bookmarks.
    pub bookmarks: Vec<CameraBookmark>,
    /// Maximum number of bookmarks.
    pub max_bookmarks: usize,
}

impl BookmarkManager {
    /// Create a new bookmark manager.
    pub fn new() -> Self {
        Self {
            bookmarks: Vec::new(),
            max_bookmarks: 32,
        }
    }

    /// Save a bookmark.
    pub fn save_bookmark(&mut self, bookmark: CameraBookmark) -> Uuid {
        let id = bookmark.id;
        if self.bookmarks.len() >= self.max_bookmarks {
            self.bookmarks.remove(0);
        }
        self.bookmarks.push(bookmark);
        id
    }

    /// Get a bookmark by ID.
    pub fn get_bookmark(&self, id: Uuid) -> Option<&CameraBookmark> {
        self.bookmarks.iter().find(|b| b.id == id)
    }

    /// Get a bookmark by slot number.
    pub fn get_by_slot(&self, slot: u8) -> Option<&CameraBookmark> {
        self.bookmarks.iter().find(|b| b.slot == Some(slot))
    }

    /// Assign a slot to a bookmark.
    pub fn assign_slot(&mut self, id: Uuid, slot: u8) {
        // Remove existing assignment for this slot.
        for b in &mut self.bookmarks {
            if b.slot == Some(slot) {
                b.slot = None;
            }
        }
        if let Some(b) = self.bookmarks.iter_mut().find(|b| b.id == id) {
            b.slot = Some(slot);
        }
    }

    /// Remove a bookmark.
    pub fn remove_bookmark(&mut self, id: Uuid) {
        self.bookmarks.retain(|b| b.id != id);
    }

    /// Rename a bookmark.
    pub fn rename_bookmark(&mut self, id: Uuid, new_name: &str) {
        if let Some(b) = self.bookmarks.iter_mut().find(|b| b.id == id) {
            b.name = new_name.to_string();
        }
    }
}

// ---------------------------------------------------------------------------
// Snap Settings
// ---------------------------------------------------------------------------

/// Global snap settings for the scene view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapSettings {
    /// Snap to grid enabled.
    pub grid_snap: bool,
    /// Grid snap increment.
    pub grid_size: f32,
    /// Snap to surface (raycast to nearest surface below).
    pub surface_snap: bool,
    /// Snap to vertices of nearby meshes.
    pub vertex_snap: bool,
    /// Vertex snap radius in screen pixels.
    pub vertex_snap_radius: f32,
    /// Rotation snap enabled.
    pub rotation_snap: bool,
    /// Rotation snap increment in degrees.
    pub rotation_increment: f32,
    /// Scale snap enabled.
    pub scale_snap: bool,
    /// Scale snap increment.
    pub scale_increment: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            grid_snap: false,
            grid_size: 1.0,
            surface_snap: false,
            vertex_snap: false,
            vertex_snap_radius: 20.0,
            rotation_snap: false,
            rotation_increment: 15.0,
            scale_snap: false,
            scale_increment: 0.1,
        }
    }
}

impl SnapSettings {
    /// Snap a position to the grid.
    pub fn snap_position(&self, position: [f32; 3]) -> [f32; 3] {
        if !self.grid_snap || self.grid_size <= 0.0 {
            return position;
        }
        [
            (position[0] / self.grid_size).round() * self.grid_size,
            (position[1] / self.grid_size).round() * self.grid_size,
            (position[2] / self.grid_size).round() * self.grid_size,
        ]
    }

    /// Snap a rotation angle to the rotation increment.
    pub fn snap_rotation(&self, angle_degrees: f32) -> f32 {
        if !self.rotation_snap || self.rotation_increment <= 0.0 {
            return angle_degrees;
        }
        (angle_degrees / self.rotation_increment).round() * self.rotation_increment
    }

    /// Snap a scale value to the scale increment.
    pub fn snap_scale(&self, scale: f32) -> f32 {
        if !self.scale_snap || self.scale_increment <= 0.0 {
            return scale;
        }
        (scale / self.scale_increment).round() * self.scale_increment
    }
}

// ---------------------------------------------------------------------------
// Alignment Tools
// ---------------------------------------------------------------------------

/// Alignment operation to apply to selected entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentOperation {
    /// Align selected entities to the ground (raycast down).
    AlignToGround,
    /// Center the view on the selected entities.
    CenterOnSelection,
    /// Align selected entities' X positions to match the first selected.
    AlignX,
    /// Align selected entities' Y positions to match the first selected.
    AlignY,
    /// Align selected entities' Z positions to match the first selected.
    AlignZ,
    /// Distribute selected entities evenly along the X axis.
    DistributeX,
    /// Distribute selected entities evenly along the Y axis.
    DistributeY,
    /// Distribute selected entities evenly along the Z axis.
    DistributeZ,
    /// Reset rotation of selected entities to identity.
    ResetRotation,
    /// Reset scale of selected entities to 1,1,1.
    ResetScale,
}

/// Align entities to the ground by raycasting downward.
/// Returns a list of (entity_id, old_y, new_y) for undo support.
pub fn align_to_ground(
    entity_positions: &[(Uuid, [f32; 3])],
    ground_height_fn: impl Fn(f32, f32) -> f32,
) -> Vec<(Uuid, f32, f32)> {
    let mut changes = Vec::new();
    for (id, pos) in entity_positions {
        let ground_y = ground_height_fn(pos[0], pos[2]);
        if (pos[1] - ground_y).abs() > 1e-4 {
            changes.push((*id, pos[1], ground_y));
        }
    }
    changes
}

/// Distribute entities evenly along an axis.
/// Returns new positions for each entity.
pub fn distribute_evenly(
    positions: &[(Uuid, [f32; 3])],
    axis: usize,
) -> Vec<(Uuid, [f32; 3])> {
    if positions.len() < 3 {
        return positions.to_vec();
    }

    let mut sorted: Vec<(Uuid, [f32; 3])> = positions.to_vec();
    sorted.sort_by(|a, b| a.1[axis].partial_cmp(&b.1[axis]).unwrap_or(std::cmp::Ordering::Equal));

    let first = sorted.first().unwrap().1[axis];
    let last = sorted.last().unwrap().1[axis];
    let step = (last - first) / (sorted.len() - 1) as f32;

    let mut result = Vec::new();
    for (i, (id, mut pos)) in sorted.into_iter().enumerate() {
        pos[axis] = first + step * i as f32;
        result.push((id, pos));
    }
    result
}

// ---------------------------------------------------------------------------
// Measurement Tool
// ---------------------------------------------------------------------------

/// A distance measurement between two points.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Unique identifier.
    pub id: Uuid,
    /// Start point in world space.
    pub start: [f32; 3],
    /// End point in world space.
    pub end: [f32; 3],
    /// Measured distance.
    pub distance: f32,
    /// Whether the measurement is pinned (persists until removed).
    pub pinned: bool,
    /// Display color.
    pub color: [f32; 4],
}

impl Measurement {
    /// Create a new measurement.
    pub fn new(start: [f32; 3], end: [f32; 3]) -> Self {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dz = end[2] - start[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        Self {
            id: Uuid::new_v4(),
            start,
            end,
            distance,
            pinned: false,
            color: [1.0, 1.0, 0.0, 1.0],
        }
    }

    /// Update the end point and recalculate distance.
    pub fn update_end(&mut self, end: [f32; 3]) {
        self.end = end;
        let dx = end[0] - self.start[0];
        let dy = end[1] - self.start[1];
        let dz = end[2] - self.start[2];
        self.distance = (dx * dx + dy * dy + dz * dz).sqrt();
    }

    /// Component-wise distances.
    pub fn component_distances(&self) -> [f32; 3] {
        [
            (self.end[0] - self.start[0]).abs(),
            (self.end[1] - self.start[1]).abs(),
            (self.end[2] - self.start[2]).abs(),
        ]
    }

    /// Midpoint of the measurement.
    pub fn midpoint(&self) -> [f32; 3] {
        [
            (self.start[0] + self.end[0]) * 0.5,
            (self.start[1] + self.end[1]) * 0.5,
            (self.start[2] + self.end[2]) * 0.5,
        ]
    }

    /// Format the distance as a string with units.
    pub fn format_distance(&self) -> String {
        if self.distance < 1.0 {
            format!("{:.1} cm", self.distance * 100.0)
        } else if self.distance < 1000.0 {
            format!("{:.2} m", self.distance)
        } else {
            format!("{:.2} km", self.distance / 1000.0)
        }
    }
}

/// Manager for measurement tools.
#[derive(Debug, Clone, Default)]
pub struct MeasurementTool {
    /// Active (in-progress) measurement.
    pub active_measurement: Option<Measurement>,
    /// Completed/pinned measurements.
    pub measurements: Vec<Measurement>,
    /// Whether the tool is active.
    pub enabled: bool,
}

impl MeasurementTool {
    /// Create a new measurement tool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a new measurement at a world position.
    pub fn begin_measurement(&mut self, start: [f32; 3]) {
        self.active_measurement = Some(Measurement::new(start, start));
    }

    /// Update the active measurement's end point.
    pub fn update_measurement(&mut self, end: [f32; 3]) {
        if let Some(ref mut m) = self.active_measurement {
            m.update_end(end);
        }
    }

    /// Complete the active measurement. Pin it if `pin` is true.
    pub fn complete_measurement(&mut self, pin: bool) -> Option<Uuid> {
        if let Some(mut m) = self.active_measurement.take() {
            if m.distance < 0.001 {
                return None;
            }
            m.pinned = pin;
            let id = m.id;
            if pin {
                self.measurements.push(m);
            }
            Some(id)
        } else {
            None
        }
    }

    /// Remove a pinned measurement.
    pub fn remove_measurement(&mut self, id: Uuid) {
        self.measurements.retain(|m| m.id != id);
    }

    /// Clear all measurements.
    pub fn clear_all(&mut self) {
        self.measurements.clear();
        self.active_measurement = None;
    }
}

// ---------------------------------------------------------------------------
// Multi-Viewport Layout
// ---------------------------------------------------------------------------

/// Layout modes for multi-viewport display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViewportLayout {
    /// Single full-screen viewport.
    Single,
    /// Two viewports side by side.
    SplitHorizontal,
    /// Two viewports stacked.
    SplitVertical,
    /// Four viewports in a 2x2 grid.
    Quad,
    /// One large viewport + two small stacked on the right.
    TwoPlusOne,
}

impl Default for ViewportLayout {
    fn default() -> Self {
        Self::Single
    }
}

impl ViewportLayout {
    /// All layouts.
    pub fn all() -> &'static [ViewportLayout] {
        &[
            Self::Single,
            Self::SplitHorizontal,
            Self::SplitVertical,
            Self::Quad,
            Self::TwoPlusOne,
        ]
    }

    /// Number of viewports in this layout.
    pub fn viewport_count(&self) -> usize {
        match self {
            Self::Single => 1,
            Self::SplitHorizontal | Self::SplitVertical => 2,
            Self::Quad => 4,
            Self::TwoPlusOne => 3,
        }
    }

    /// Compute viewport rectangles (x, y, width, height) normalized to [0, 1].
    pub fn compute_rects(&self) -> Vec<ViewportRect> {
        match self {
            Self::Single => vec![ViewportRect::new(0.0, 0.0, 1.0, 1.0)],
            Self::SplitHorizontal => vec![
                ViewportRect::new(0.0, 0.0, 0.5, 1.0),
                ViewportRect::new(0.5, 0.0, 0.5, 1.0),
            ],
            Self::SplitVertical => vec![
                ViewportRect::new(0.0, 0.0, 1.0, 0.5),
                ViewportRect::new(0.0, 0.5, 1.0, 0.5),
            ],
            Self::Quad => vec![
                ViewportRect::new(0.0, 0.0, 0.5, 0.5),
                ViewportRect::new(0.5, 0.0, 0.5, 0.5),
                ViewportRect::new(0.0, 0.5, 0.5, 0.5),
                ViewportRect::new(0.5, 0.5, 0.5, 0.5),
            ],
            Self::TwoPlusOne => vec![
                ViewportRect::new(0.0, 0.0, 0.7, 1.0),
                ViewportRect::new(0.7, 0.0, 0.3, 0.5),
                ViewportRect::new(0.7, 0.5, 0.3, 0.5),
            ],
        }
    }
}

/// A normalized viewport rectangle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ViewportRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ViewportRect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Convert to pixel coordinates given a total window size.
    pub fn to_pixels(&self, window_width: u32, window_height: u32) -> (u32, u32, u32, u32) {
        (
            (self.x * window_width as f32) as u32,
            (self.y * window_height as f32) as u32,
            (self.width * window_width as f32) as u32,
            (self.height * window_height as f32) as u32,
        )
    }

    /// Test if a point (in normalized coords) is inside this rect.
    pub fn contains(&self, nx: f32, ny: f32) -> bool {
        nx >= self.x && nx < self.x + self.width && ny >= self.y && ny < self.y + self.height
    }
}

/// State for a multi-viewport layout.
#[derive(Debug, Clone)]
pub struct MultiViewportState {
    /// Current layout.
    pub layout: ViewportLayout,
    /// Which viewport is focused (receives input).
    pub focused_viewport: usize,
    /// Whether to maximize the focused viewport.
    pub maximized: bool,
    /// Per-viewport camera presets: 0=perspective, 1=top, 2=front, 3=right.
    pub viewport_presets: Vec<u8>,
}

impl Default for MultiViewportState {
    fn default() -> Self {
        Self {
            layout: ViewportLayout::Single,
            focused_viewport: 0,
            maximized: false,
            viewport_presets: vec![0, 1, 2, 3],
        }
    }
}

impl MultiViewportState {
    /// Create a new multi-viewport state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the layout and reset focus.
    pub fn set_layout(&mut self, layout: ViewportLayout) {
        self.layout = layout;
        self.focused_viewport = 0;
        self.maximized = false;
        // Ensure presets array is large enough.
        while self.viewport_presets.len() < layout.viewport_count() {
            self.viewport_presets.push(self.viewport_presets.len() as u8);
        }
    }

    /// Focus a viewport by index.
    pub fn focus_viewport(&mut self, index: usize) {
        if index < self.layout.viewport_count() {
            self.focused_viewport = index;
        }
    }

    /// Toggle maximization of the focused viewport.
    pub fn toggle_maximize(&mut self) {
        self.maximized = !self.maximized;
    }

    /// Find which viewport a normalized screen position belongs to.
    pub fn viewport_at_position(&self, nx: f32, ny: f32) -> Option<usize> {
        if self.maximized {
            return Some(self.focused_viewport);
        }
        let rects = self.layout.compute_rects();
        rects.iter().position(|r| r.contains(nx, ny))
    }
}

// ---------------------------------------------------------------------------
// Statistics Overlay
// ---------------------------------------------------------------------------

/// Real-time statistics displayed in the scene view.
#[derive(Debug, Clone, Default)]
pub struct SceneStatistics {
    /// Frames per second.
    pub fps: f32,
    /// Frame time in milliseconds.
    pub frame_time_ms: f32,
    /// Number of draw calls this frame.
    pub draw_calls: u32,
    /// Total triangles rendered.
    pub triangles: u64,
    /// Total vertices rendered.
    pub vertices: u64,
    /// Number of entities in the scene.
    pub entity_count: u32,
    /// Number of visible entities (after culling).
    pub visible_entities: u32,
    /// GPU memory used in MB.
    pub gpu_memory_mb: f32,
    /// CPU memory used in MB.
    pub cpu_memory_mb: f32,
    /// Number of active lights.
    pub light_count: u32,
    /// Number of active particle systems.
    pub particle_systems: u32,
    /// Total active particles.
    pub active_particles: u32,
    /// Number of shadow-casting lights.
    pub shadow_casters: u32,
}

impl SceneStatistics {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update FPS from delta time.
    pub fn update_fps(&mut self, dt: f32) {
        self.frame_time_ms = dt * 1000.0;
        self.fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
    }

    /// Format statistics as a multi-line string.
    pub fn format(&self) -> String {
        format!(
            "FPS: {:.1} ({:.1}ms)\n\
             Draw Calls: {}\n\
             Triangles: {}\n\
             Vertices: {}\n\
             Entities: {} ({} visible)\n\
             Lights: {} ({} shadow)\n\
             Particles: {} ({} systems)\n\
             GPU: {:.1} MB | CPU: {:.1} MB",
            self.fps,
            self.frame_time_ms,
            self.draw_calls,
            self.triangles,
            self.vertices,
            self.entity_count,
            self.visible_entities,
            self.light_count,
            self.shadow_casters,
            self.active_particles,
            self.particle_systems,
            self.gpu_memory_mb,
            self.cpu_memory_mb,
        )
    }
}

/// Overlay visibility settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlaySettings {
    /// Show statistics overlay.
    pub show_statistics: bool,
    /// Show grid.
    pub show_grid: bool,
    /// Show wireframe overlay.
    pub show_wireframe: bool,
    /// Show bounding boxes.
    pub show_bounds: bool,
    /// Show light icons and ranges.
    pub show_lights: bool,
    /// Show camera icons.
    pub show_cameras: bool,
    /// Show collider shapes.
    pub show_colliders: bool,
    /// Show navigation mesh.
    pub show_navmesh: bool,
    /// Show audio source ranges.
    pub show_audio_sources: bool,
    /// Statistics position on screen.
    pub stats_position: OverlayPosition,
}

impl Default for OverlaySettings {
    fn default() -> Self {
        Self {
            show_statistics: true,
            show_grid: true,
            show_wireframe: false,
            show_bounds: false,
            show_lights: true,
            show_cameras: true,
            show_colliders: false,
            show_navmesh: false,
            show_audio_sources: false,
            stats_position: OverlayPosition::TopRight,
        }
    }
}

/// Position of the overlay on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OverlayPosition {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

// ---------------------------------------------------------------------------
// Scene View State
// ---------------------------------------------------------------------------

/// Composite state for all scene view features.
#[derive(Debug, Clone)]
pub struct SceneViewState {
    /// Camera bookmarks.
    pub bookmarks: BookmarkManager,
    /// Snap settings.
    pub snap: SnapSettings,
    /// Multi-viewport layout.
    pub viewports: MultiViewportState,
    /// Measurement tool.
    pub measurement: MeasurementTool,
    /// Statistics.
    pub statistics: SceneStatistics,
    /// Overlay settings.
    pub overlays: OverlaySettings,
}

impl Default for SceneViewState {
    fn default() -> Self {
        Self {
            bookmarks: BookmarkManager::new(),
            snap: SnapSettings::default(),
            viewports: MultiViewportState::new(),
            measurement: MeasurementTool::new(),
            statistics: SceneStatistics::new(),
            overlays: OverlaySettings::default(),
        }
    }
}

impl SceneViewState {
    /// Create a new scene view state.
    pub fn new() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bookmark_save_and_restore() {
        let mut mgr = BookmarkManager::new();
        let bm = CameraBookmark::new("Front", [0.0, 0.0, 10.0], [0.0, 0.0, 0.0], 0.0, 0.0, 10.0);
        let id = mgr.save_bookmark(bm);
        assert_eq!(mgr.bookmarks.len(), 1);

        let restored = mgr.get_bookmark(id).unwrap();
        assert_eq!(restored.name, "Front");
        assert!((restored.position[2] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn bookmark_slots() {
        let mut mgr = BookmarkManager::new();
        let bm = CameraBookmark::new("Slot1", [0.0; 3], [0.0; 3], 0.0, 0.0, 5.0);
        let id = mgr.save_bookmark(bm);
        mgr.assign_slot(id, 1);
        assert!(mgr.get_by_slot(1).is_some());
        assert_eq!(mgr.get_by_slot(1).unwrap().id, id);
    }

    #[test]
    fn snap_position() {
        let snap = SnapSettings {
            grid_snap: true,
            grid_size: 0.5,
            ..Default::default()
        };
        let snapped = snap.snap_position([1.3, 2.7, 0.1]);
        assert!((snapped[0] - 1.5).abs() < 1e-5);
        assert!((snapped[1] - 2.5).abs() < 1e-5);
        assert!((snapped[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn snap_rotation() {
        let snap = SnapSettings {
            rotation_snap: true,
            rotation_increment: 15.0,
            ..Default::default()
        };
        assert!((snap.snap_rotation(37.0) - 30.0).abs() < 1e-3);
        assert!((snap.snap_rotation(42.0) - 45.0).abs() < 1e-3);
    }

    #[test]
    fn snap_disabled() {
        let snap = SnapSettings::default(); // All disabled.
        let pos = [1.3, 2.7, 0.1];
        let snapped = snap.snap_position(pos);
        assert_eq!(snapped, pos);
    }

    #[test]
    fn measurement_distance() {
        let m = Measurement::new([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        assert!((m.distance - 5.0).abs() < 1e-5);
        assert_eq!(m.midpoint(), [1.5, 2.0, 0.0]);
    }

    #[test]
    fn measurement_update() {
        let mut m = Measurement::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!((m.distance - 1.0).abs() < 1e-5);
        m.update_end([5.0, 0.0, 0.0]);
        assert!((m.distance - 5.0).abs() < 1e-5);
    }

    #[test]
    fn measurement_format() {
        let m1 = Measurement::new([0.0; 3], [0.005, 0.0, 0.0]);
        assert!(m1.format_distance().contains("cm"));

        let m2 = Measurement::new([0.0; 3], [5.0, 0.0, 0.0]);
        assert!(m2.format_distance().contains("m"));

        let m3 = Measurement::new([0.0; 3], [2000.0, 0.0, 0.0]);
        assert!(m3.format_distance().contains("km"));
    }

    #[test]
    fn measurement_tool_workflow() {
        let mut tool = MeasurementTool::new();
        tool.begin_measurement([0.0, 0.0, 0.0]);
        tool.update_measurement([10.0, 0.0, 0.0]);
        let id = tool.complete_measurement(true);
        assert!(id.is_some());
        assert_eq!(tool.measurements.len(), 1);
    }

    #[test]
    fn viewport_layout_rects() {
        let rects = ViewportLayout::Single.compute_rects();
        assert_eq!(rects.len(), 1);
        assert!((rects[0].width - 1.0).abs() < 1e-5);

        let quad = ViewportLayout::Quad.compute_rects();
        assert_eq!(quad.len(), 4);
        for r in &quad {
            assert!((r.width - 0.5).abs() < 1e-5);
            assert!((r.height - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn viewport_rect_contains() {
        let r = ViewportRect::new(0.0, 0.0, 0.5, 0.5);
        assert!(r.contains(0.25, 0.25));
        assert!(!r.contains(0.75, 0.75));
    }

    #[test]
    fn viewport_rect_pixels() {
        let r = ViewportRect::new(0.5, 0.0, 0.5, 1.0);
        let (x, y, w, h) = r.to_pixels(1920, 1080);
        assert_eq!(x, 960);
        assert_eq!(y, 0);
        assert_eq!(w, 960);
        assert_eq!(h, 1080);
    }

    #[test]
    fn multi_viewport_focus() {
        let mut state = MultiViewportState::new();
        state.set_layout(ViewportLayout::Quad);
        assert_eq!(state.focused_viewport, 0);

        state.focus_viewport(2);
        assert_eq!(state.focused_viewport, 2);

        state.toggle_maximize();
        assert!(state.maximized);
        assert_eq!(state.viewport_at_position(0.1, 0.1), Some(2));
    }

    #[test]
    fn statistics_format() {
        let mut stats = SceneStatistics::new();
        stats.update_fps(1.0 / 60.0);
        stats.draw_calls = 150;
        stats.triangles = 1_000_000;
        stats.entity_count = 500;
        stats.visible_entities = 300;

        let text = stats.format();
        assert!(text.contains("FPS"));
        assert!(text.contains("Draw Calls"));
        assert!(text.contains("1000000"));
    }

    #[test]
    fn distribute_evenly_x() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        let positions = vec![
            (id1, [0.0, 0.0, 0.0]),
            (id2, [5.0, 0.0, 0.0]),
            (id3, [10.0, 0.0, 0.0]),
        ];

        let result = distribute_evenly(&positions, 0);
        assert_eq!(result.len(), 3);
        // Should be evenly spaced: 0, 5, 10.
        assert!((result[0].1[0] - 0.0).abs() < 1e-5);
        assert!((result[1].1[0] - 5.0).abs() < 1e-5);
        assert!((result[2].1[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn align_to_ground_basic() {
        let id = Uuid::new_v4();
        let positions = vec![(id, [5.0, 10.0, 5.0])];

        let changes = align_to_ground(&positions, |_, _| 0.0);
        assert_eq!(changes.len(), 1);
        assert!((changes[0].1 - 10.0).abs() < 1e-5);
        assert!((changes[0].2 - 0.0).abs() < 1e-5);
    }

    #[test]
    fn scene_view_state_creation() {
        let state = SceneViewState::new();
        assert!(state.bookmarks.bookmarks.is_empty());
        assert!(!state.snap.grid_snap);
        assert_eq!(state.viewports.layout, ViewportLayout::Single);
    }
}
