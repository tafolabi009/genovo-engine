//! Viewport widget: 3D scene rendering in UI panel, camera controls
//! (orbit/fly/pan), gizmo overlay, selection highlighting, and grid rendering.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Camera modes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportCameraMode { Orbit, Fly, Pan, FirstPerson, Locked }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportProjection { Perspective, Orthographic }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportOrientation { Free, Top, Bottom, Front, Back, Left, Right }

// ---------------------------------------------------------------------------
// Camera state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ViewportCamera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
    pub near_clip: f32,
    pub far_clip: f32,
    pub ortho_size: f32,
    pub projection: ViewportProjection,
    pub orientation: ViewportOrientation,
    pub mode: ViewportCameraMode,
    pub orbit_distance: f32,
    pub orbit_yaw: f32,
    pub orbit_pitch: f32,
    pub move_speed: f32,
    pub rotate_speed: f32,
    pub zoom_speed: f32,
    pub smooth_movement: bool,
    pub smooth_factor: f32,
    pub velocity: [f32; 3],
}

impl ViewportCamera {
    pub fn new() -> Self {
        Self {
            position: [5.0, 5.0, 5.0], target: [0.0, 0.0, 0.0], up: [0.0, 1.0, 0.0],
            fov: 60.0, near_clip: 0.01, far_clip: 10000.0, ortho_size: 10.0,
            projection: ViewportProjection::Perspective, orientation: ViewportOrientation::Free,
            mode: ViewportCameraMode::Orbit, orbit_distance: 8.66, orbit_yaw: 45.0,
            orbit_pitch: 35.0, move_speed: 5.0, rotate_speed: 0.3, zoom_speed: 1.0,
            smooth_movement: true, smooth_factor: 0.15, velocity: [0.0; 3],
        }
    }

    pub fn forward(&self) -> [f32; 3] {
        let dx = self.target[0] - self.position[0];
        let dy = self.target[1] - self.position[1];
        let dz = self.target[2] - self.position[2];
        let len = (dx*dx + dy*dy + dz*dz).sqrt();
        if len < 1e-10 { return [0.0, 0.0, -1.0]; }
        [dx/len, dy/len, dz/len]
    }

    pub fn right(&self) -> [f32; 3] {
        let f = self.forward();
        let ux = self.up[1]*f[2] - self.up[2]*f[1];
        let uy = self.up[2]*f[0] - self.up[0]*f[2];
        let uz = self.up[0]*f[1] - self.up[1]*f[0];
        let len = (ux*ux + uy*uy + uz*uz).sqrt();
        if len < 1e-10 { return [1.0, 0.0, 0.0]; }
        [ux/len, uy/len, uz/len]
    }

    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.orbit_yaw += delta_yaw * self.rotate_speed;
        self.orbit_pitch = (self.orbit_pitch + delta_pitch * self.rotate_speed).clamp(-89.0, 89.0);
        self.update_orbit_position();
    }

    pub fn zoom(&mut self, delta: f32) {
        match self.projection {
            ViewportProjection::Perspective => {
                self.orbit_distance = (self.orbit_distance - delta * self.zoom_speed).max(0.1);
                self.update_orbit_position();
            }
            ViewportProjection::Orthographic => {
                self.ortho_size = (self.ortho_size - delta * self.zoom_speed).max(0.1);
            }
        }
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let r = self.right();
        let factor = self.orbit_distance * 0.002;
        self.target[0] += (-r[0] * dx + self.up[0] * dy) * factor;
        self.target[1] += (-r[1] * dx + self.up[1] * dy) * factor;
        self.target[2] += (-r[2] * dx + self.up[2] * dy) * factor;
        self.update_orbit_position();
    }

    pub fn fly_move(&mut self, forward: f32, right: f32, up: f32, dt: f32) {
        let f = self.forward();
        let r = self.right();
        let speed = self.move_speed * dt;
        self.position[0] += (f[0]*forward + r[0]*right + self.up[0]*up) * speed;
        self.position[1] += (f[1]*forward + r[1]*right + self.up[1]*up) * speed;
        self.position[2] += (f[2]*forward + r[2]*right + self.up[2]*up) * speed;
        self.target[0] = self.position[0] + f[0];
        self.target[1] = self.position[1] + f[1];
        self.target[2] = self.position[2] + f[2];
    }

    pub fn focus_on(&mut self, center: [f32; 3], radius: f32) {
        self.target = center;
        self.orbit_distance = radius * 2.5;
        self.update_orbit_position();
    }

    pub fn set_orientation(&mut self, orientation: ViewportOrientation) {
        self.orientation = orientation;
        match orientation {
            ViewportOrientation::Top => { self.orbit_yaw = 0.0; self.orbit_pitch = -89.0; }
            ViewportOrientation::Bottom => { self.orbit_yaw = 0.0; self.orbit_pitch = 89.0; }
            ViewportOrientation::Front => { self.orbit_yaw = 0.0; self.orbit_pitch = 0.0; }
            ViewportOrientation::Back => { self.orbit_yaw = 180.0; self.orbit_pitch = 0.0; }
            ViewportOrientation::Left => { self.orbit_yaw = 90.0; self.orbit_pitch = 0.0; }
            ViewportOrientation::Right => { self.orbit_yaw = -90.0; self.orbit_pitch = 0.0; }
            ViewportOrientation::Free => {}
        }
        self.update_orbit_position();
    }

    fn update_orbit_position(&mut self) {
        let yaw_rad = self.orbit_yaw.to_radians();
        let pitch_rad = self.orbit_pitch.to_radians();
        let cos_pitch = pitch_rad.cos();
        self.position[0] = self.target[0] + self.orbit_distance * cos_pitch * yaw_rad.sin();
        self.position[1] = self.target[1] + self.orbit_distance * pitch_rad.sin();
        self.position[2] = self.target[2] + self.orbit_distance * cos_pitch * yaw_rad.cos();
    }

    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let f = self.forward();
        let r = self.right();
        let u = [
            r[1]*f[2] - r[2]*f[1],
            r[2]*f[0] - r[0]*f[2],
            r[0]*f[1] - r[1]*f[0],
        ];

        [
            [r[0], u[0], -f[0], 0.0],
            [r[1], u[1], -f[1], 0.0],
            [r[2], u[2], -f[2], 0.0],
            [
                -(r[0]*self.position[0] + r[1]*self.position[1] + r[2]*self.position[2]),
                -(u[0]*self.position[0] + u[1]*self.position[1] + u[2]*self.position[2]),
                f[0]*self.position[0] + f[1]*self.position[1] + f[2]*self.position[2],
                1.0,
            ],
        ]
    }
}

impl Default for ViewportCamera {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Gizmo
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoType { Translate, Rotate, Scale, Universal }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoSpace { Local, World }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoAxis { None, X, Y, Z, XY, XZ, YZ, XYZ, View }

#[derive(Debug, Clone)]
pub struct GizmoState {
    pub gizmo_type: GizmoType,
    pub space: GizmoSpace,
    pub active_axis: GizmoAxis,
    pub visible: bool,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: f32,
    pub dragging: bool,
    pub drag_start: [f32; 3],
    pub drag_current: [f32; 3],
    pub snap_translate: f32,
    pub snap_rotate: f32,
    pub snap_scale: f32,
    pub snap_enabled: bool,
    pub axis_colors: [[f32; 4]; 3],
    pub highlight_color: [f32; 4],
}

impl GizmoState {
    pub fn new() -> Self {
        Self {
            gizmo_type: GizmoType::Translate, space: GizmoSpace::World,
            active_axis: GizmoAxis::None, visible: true, position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0], scale: 1.0, dragging: false,
            drag_start: [0.0; 3], drag_current: [0.0; 3],
            snap_translate: 1.0, snap_rotate: 15.0, snap_scale: 0.1,
            snap_enabled: false,
            axis_colors: [
                [1.0, 0.2, 0.2, 1.0],
                [0.2, 1.0, 0.2, 1.0],
                [0.2, 0.2, 1.0, 1.0],
            ],
            highlight_color: [1.0, 1.0, 0.0, 1.0],
        }
    }

    pub fn cycle_type(&mut self) {
        self.gizmo_type = match self.gizmo_type {
            GizmoType::Translate => GizmoType::Rotate,
            GizmoType::Rotate => GizmoType::Scale,
            GizmoType::Scale => GizmoType::Universal,
            GizmoType::Universal => GizmoType::Translate,
        };
    }

    pub fn toggle_space(&mut self) {
        self.space = match self.space {
            GizmoSpace::Local => GizmoSpace::World,
            GizmoSpace::World => GizmoSpace::Local,
        };
    }
}

impl Default for GizmoState {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GridSettings {
    pub visible: bool,
    pub size: f32,
    pub divisions: u32,
    pub major_color: [f32; 4],
    pub minor_color: [f32; 4],
    pub axis_color_x: [f32; 4],
    pub axis_color_z: [f32; 4],
    pub fade_distance: f32,
    pub snap_to_grid: bool,
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            visible: true, size: 100.0, divisions: 10,
            major_color: [0.3, 0.3, 0.3, 0.5],
            minor_color: [0.2, 0.2, 0.2, 0.3],
            axis_color_x: [1.0, 0.2, 0.2, 0.5],
            axis_color_z: [0.2, 0.2, 1.0, 0.5],
            fade_distance: 200.0, snap_to_grid: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Selection highlighting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionHighlightMode { Outline, Fill, Both, None }

#[derive(Debug, Clone)]
pub struct SelectionHighlight {
    pub mode: SelectionHighlightMode,
    pub color: [f32; 4],
    pub hover_color: [f32; 4],
    pub outline_width: f32,
    pub fill_opacity: f32,
    pub pulse_speed: f32,
    pub show_bounding_box: bool,
}

impl Default for SelectionHighlight {
    fn default() -> Self {
        Self {
            mode: SelectionHighlightMode::Outline,
            color: [1.0, 0.6, 0.0, 1.0],
            hover_color: [0.5, 0.8, 1.0, 0.5],
            outline_width: 2.0, fill_opacity: 0.15,
            pulse_speed: 2.0, show_bounding_box: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Viewport overlay info
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ViewportOverlay {
    pub show_fps: bool,
    pub show_stats: bool,
    pub show_camera_info: bool,
    pub show_selection_info: bool,
    pub show_compass: bool,
    pub show_safe_area: bool,
    pub fps: f32,
    pub draw_calls: u32,
    pub triangles: u32,
    pub vertices: u32,
}

impl Default for ViewportOverlay {
    fn default() -> Self {
        Self {
            show_fps: true, show_stats: false, show_camera_info: false,
            show_selection_info: true, show_compass: true, show_safe_area: false,
            fps: 0.0, draw_calls: 0, triangles: 0, vertices: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Render mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Lit, Unlit, Wireframe, WireframeOnLit, Normals, UVs,
    Overdraw, MotionVectors, Depth, AmbientOcclusion, BaseColor,
    Metallic, Roughness, LightComplexity,
}

impl RenderMode {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Lit => "Lit", Self::Unlit => "Unlit", Self::Wireframe => "Wireframe",
            Self::WireframeOnLit => "Wireframe on Lit", Self::Normals => "Normals",
            Self::UVs => "UVs", Self::Overdraw => "Overdraw",
            Self::MotionVectors => "Motion Vectors", Self::Depth => "Depth",
            Self::AmbientOcclusion => "AO", Self::BaseColor => "Base Color",
            Self::Metallic => "Metallic", Self::Roughness => "Roughness",
            Self::LightComplexity => "Light Complexity",
        }
    }
}

// ---------------------------------------------------------------------------
// Viewport state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ViewportEvent {
    CameraMoved, CameraZoomed, SelectionChanged(Vec<u64>),
    GizmoStarted(GizmoType), GizmoDragged(GizmoAxis, [f32; 3]),
    GizmoFinished, RenderModeChanged(RenderMode), ViewportResized(f32, f32),
}

pub struct ViewportState {
    pub camera: ViewportCamera,
    pub gizmo: GizmoState,
    pub grid: GridSettings,
    pub selection: SelectionHighlight,
    pub overlay: ViewportOverlay,
    pub render_mode: RenderMode,
    pub width: f32,
    pub height: f32,
    pub dpi_scale: f32,
    pub selected_entities: Vec<u64>,
    pub hovered_entity: Option<u64>,
    pub events: Vec<ViewportEvent>,
    pub show_debug_shapes: bool,
    pub show_lights: bool,
    pub show_cameras: bool,
    pub show_colliders: bool,
    pub background_color: [f32; 4],
    pub anti_aliasing: bool,
    pub enable_shadows: bool,
    pub enable_post_processing: bool,
    pub realtime: bool,
}

impl ViewportState {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            camera: ViewportCamera::new(), gizmo: GizmoState::new(),
            grid: GridSettings::default(), selection: SelectionHighlight::default(),
            overlay: ViewportOverlay::default(), render_mode: RenderMode::Lit,
            width, height, dpi_scale: 1.0, selected_entities: Vec::new(),
            hovered_entity: None, events: Vec::new(),
            show_debug_shapes: false, show_lights: true, show_cameras: true,
            show_colliders: false, background_color: [0.1, 0.1, 0.1, 1.0],
            anti_aliasing: true, enable_shadows: true,
            enable_post_processing: true, realtime: true,
        }
    }

    pub fn select_entity(&mut self, entity_id: u64) {
        if !self.selected_entities.contains(&entity_id) {
            self.selected_entities.push(entity_id);
        }
        self.events.push(ViewportEvent::SelectionChanged(self.selected_entities.clone()));
    }

    pub fn deselect_all(&mut self) {
        self.selected_entities.clear();
        self.events.push(ViewportEvent::SelectionChanged(Vec::new()));
    }

    pub fn set_render_mode(&mut self, mode: RenderMode) {
        self.render_mode = mode;
        self.events.push(ViewportEvent::RenderModeChanged(mode));
    }

    pub fn resize(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
        self.events.push(ViewportEvent::ViewportResized(width, height));
    }

    pub fn aspect_ratio(&self) -> f32 { self.width / self.height.max(1.0) }

    pub fn screen_to_ray(&self, screen_x: f32, screen_y: f32) -> ([f32; 3], [f32; 3]) {
        let nx = (2.0 * screen_x / self.width - 1.0);
        let ny = (1.0 - 2.0 * screen_y / self.height);
        let fov_rad = self.camera.fov.to_radians();
        let aspect = self.aspect_ratio();
        let dir_x = nx * aspect * (fov_rad * 0.5).tan();
        let dir_y = ny * (fov_rad * 0.5).tan();
        let dir = [dir_x, dir_y, -1.0];
        let len = (dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]).sqrt();
        (self.camera.position, [dir[0]/len, dir[1]/len, dir[2]/len])
    }

    pub fn drain_events(&mut self) -> Vec<ViewportEvent> { std::mem::take(&mut self.events) }
}

impl Default for ViewportState {
    fn default() -> Self { Self::new(800.0, 600.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_orbit() {
        let mut cam = ViewportCamera::new();
        cam.orbit(10.0, 5.0);
        assert_ne!(cam.orbit_yaw, 45.0);
    }

    #[test]
    fn viewport_select() {
        let mut vp = ViewportState::new(800.0, 600.0);
        vp.select_entity(42);
        assert!(vp.selected_entities.contains(&42));
    }

    #[test]
    fn gizmo_cycle() {
        let mut g = GizmoState::new();
        assert_eq!(g.gizmo_type, GizmoType::Translate);
        g.cycle_type();
        assert_eq!(g.gizmo_type, GizmoType::Rotate);
    }
}
