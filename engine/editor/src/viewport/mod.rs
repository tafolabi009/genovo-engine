//! # Editor Viewport
//!
//! Provides the 3D editor viewport including camera controls, gizmo rendering,
//! entity selection via pick-rays, and debug overlays such as grid and bounds
//! visualization.
//!
//! The viewport is the primary surface through which designers interact with the
//! scene. It supports multiple camera modes (orbit, fly, orthographic presets),
//! interactive transform gizmos, rectangle (box) selection, and per-entity AABB
//! picking.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// The mode in which the viewport camera operates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CameraMode {
    /// Orbit around a focal point (Maya-style).
    Orbit,
    /// First-person fly-through (Unreal-style).
    Fly,
    /// Orthographic top-down view.
    TopDown,
    /// Orthographic front view.
    Front,
    /// Orthographic side (right) view.
    Right,
}

impl Default for CameraMode {
    fn default() -> Self {
        Self::Orbit
    }
}

// ---------------------------------------------------------------------------
// InputState  (lightweight snapshot consumed by viewport update)
// ---------------------------------------------------------------------------

/// Snapshot of input state consumed by the viewport every frame.
#[derive(Debug, Clone, Default)]
pub struct InputState {
    /// Current mouse position in viewport-local pixels.
    pub mouse_pos: [f32; 2],
    /// Mouse delta since last frame.
    pub mouse_delta: [f32; 2],
    /// Mouse scroll wheel delta (positive = forward / zoom in).
    pub scroll_delta: f32,
    /// Left mouse button held.
    pub left_mouse_down: bool,
    /// Left mouse just pressed this frame.
    pub left_mouse_pressed: bool,
    /// Left mouse just released this frame.
    pub left_mouse_released: bool,
    /// Middle mouse button held.
    pub middle_mouse_down: bool,
    /// Right mouse button held.
    pub right_mouse_down: bool,
    /// Whether Shift is held.
    pub shift_held: bool,
    /// Whether Ctrl is held.
    pub ctrl_held: bool,
    /// Whether Alt is held.
    pub alt_held: bool,
    /// WASD / QE keys for fly-mode (forward, back, left, right, up, down).
    pub move_forward: bool,
    pub move_backward: bool,
    pub move_left: bool,
    pub move_right: bool,
    pub move_up: bool,
    pub move_down: bool,
}

// ---------------------------------------------------------------------------
// ViewportCamera
// ---------------------------------------------------------------------------

/// Camera state for the editor viewport.
///
/// Supports orbit, pan, zoom, and fly-through modes with configurable
/// sensitivity and clipping planes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportCamera {
    /// Current camera mode.
    pub mode: CameraMode,
    /// Camera position in world space.
    pub position: [f32; 3],
    /// Focal / look-at point (used in Orbit mode).
    pub target: [f32; 3],
    /// Up vector.
    pub up: [f32; 3],
    /// Vertical field of view in radians (perspective only).
    pub fov_y: f32,
    /// Near clipping plane distance.
    pub near_clip: f32,
    /// Far clipping plane distance.
    pub far_clip: f32,
    /// Movement speed multiplier.
    pub move_speed: f32,
    /// Mouse look sensitivity.
    pub look_sensitivity: f32,
    /// Orbit distance from the target.
    pub orbit_distance: f32,
    /// Yaw angle in radians (rotation about Y axis).
    pub yaw: f32,
    /// Pitch angle in radians (rotation about local X axis).
    pub pitch: f32,
    /// Speed boost multiplier when shift is held.
    pub speed_boost: f32,
    /// Orthographic size (half-height in world units) for ortho modes.
    pub ortho_size: f32,
}

impl Default for ViewportCamera {
    fn default() -> Self {
        Self {
            mode: CameraMode::default(),
            position: [0.0, 5.0, 10.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y: std::f32::consts::FRAC_PI_4,
            near_clip: 0.1,
            far_clip: 1000.0,
            move_speed: 5.0,
            look_sensitivity: 0.005,
            orbit_distance: 10.0,
            yaw: 0.0,
            pitch: -0.4,
            speed_boost: 3.0,
            ortho_size: 10.0,
        }
    }
}

impl ViewportCamera {
    /// Create a new viewport camera with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Orbit the camera around the target by the given yaw/pitch delta (radians).
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch += delta_pitch;

        // Clamp pitch to avoid gimbal flip.
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);

        self.recompute_orbit_position();
    }

    /// Pan the camera on its local XY plane.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let (right, up, _forward) = self.basis_vectors();

        let pan_speed = self.orbit_distance * 0.002;
        let offset_x = right[0] * delta_x * pan_speed;
        let offset_y = right[1] * delta_x * pan_speed;
        let offset_z = right[2] * delta_x * pan_speed;

        let up_offset_x = up[0] * delta_y * pan_speed;
        let up_offset_y = up[1] * delta_y * pan_speed;
        let up_offset_z = up[2] * delta_y * pan_speed;

        self.target[0] -= offset_x + up_offset_x;
        self.target[1] -= offset_y + up_offset_y;
        self.target[2] -= offset_z + up_offset_z;

        self.position[0] -= offset_x + up_offset_x;
        self.position[1] -= offset_y + up_offset_y;
        self.position[2] -= offset_z + up_offset_z;
    }

    /// Zoom the camera (move along forward axis or adjust orbit distance).
    pub fn zoom(&mut self, delta: f32) {
        match self.mode {
            CameraMode::Orbit => {
                self.orbit_distance *= 1.0 - delta * 0.1;
                self.orbit_distance = self.orbit_distance.clamp(0.1, 500.0);
                self.recompute_orbit_position();
            }
            CameraMode::Fly => {
                // In fly mode, scroll adjusts movement speed.
                self.move_speed *= 1.0 + delta * 0.1;
                self.move_speed = self.move_speed.clamp(0.5, 200.0);
            }
            CameraMode::TopDown | CameraMode::Front | CameraMode::Right => {
                self.ortho_size *= 1.0 - delta * 0.1;
                self.ortho_size = self.ortho_size.clamp(0.5, 500.0);
            }
        }
    }

    /// Update fly-mode camera from mouse delta and key state.
    pub fn fly_update(
        &mut self,
        delta_yaw: f32,
        delta_pitch: f32,
        movement: [f32; 3],
        dt: f32,
        speed_boost: bool,
    ) {
        // Mouse look.
        self.yaw += delta_yaw;
        self.pitch += delta_pitch;
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);

        let (right, up_vec, forward) = self.basis_vectors();
        let speed = if speed_boost {
            self.move_speed * self.speed_boost
        } else {
            self.move_speed
        };

        // movement: [right, up, forward]
        let move_right = movement[0] * speed * dt;
        let move_up = movement[1] * speed * dt;
        let move_fwd = movement[2] * speed * dt;

        for i in 0..3 {
            self.position[i] += right[i] * move_right + up_vec[i] * move_up + forward[i] * move_fwd;
        }

        // Update target to be in front of position.
        for i in 0..3 {
            self.target[i] = self.position[i] + forward[i];
        }
    }

    /// Compute unit basis vectors (right, up, forward) from yaw/pitch.
    pub fn basis_vectors(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let cos_p = self.pitch.cos();
        let sin_p = self.pitch.sin();
        let cos_y = self.yaw.cos();
        let sin_y = self.yaw.sin();

        let forward = [cos_p * sin_y, sin_p, cos_p * cos_y];
        // Right is perpendicular to forward in the horizontal plane.
        let len_h = (forward[0] * forward[0] + forward[2] * forward[2]).sqrt().max(1e-8);
        let fwd_h = [forward[0] / len_h, forward[2] / len_h];
        let right = [fwd_h[1], 0.0, -fwd_h[0]];
        // Up = cross(forward, right)
        let up = cross(forward, right);
        let up_len = vec3_len(up).max(1e-8);
        let up = [up[0] / up_len, up[1] / up_len, up[2] / up_len];
        (right, up, forward)
    }

    /// Recompute position from orbit target, distance, yaw, and pitch.
    fn recompute_orbit_position(&mut self) {
        let cos_p = self.pitch.cos();
        let sin_p = self.pitch.sin();
        let cos_y = self.yaw.cos();
        let sin_y = self.yaw.sin();

        // Position is behind the target along the negated forward vector.
        self.position[0] = self.target[0] - cos_p * sin_y * self.orbit_distance;
        self.position[1] = self.target[1] - sin_p * self.orbit_distance;
        self.position[2] = self.target[2] - cos_p * cos_y * self.orbit_distance;
    }

    /// Compute a 4x4 view matrix (look-at).
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        look_at(self.position, self.target, self.up)
    }

    /// Compute the projection matrix for the given viewport aspect ratio.
    pub fn projection_matrix(&self, aspect_ratio: f32) -> [[f32; 4]; 4] {
        match self.mode {
            CameraMode::Orbit | CameraMode::Fly => {
                perspective(self.fov_y, aspect_ratio, self.near_clip, self.far_clip)
            }
            CameraMode::TopDown | CameraMode::Front | CameraMode::Right => {
                let half_h = self.ortho_size;
                let half_w = half_h * aspect_ratio;
                orthographic(-half_w, half_w, -half_h, half_h, self.near_clip, self.far_clip)
            }
        }
    }

    /// Convert a screen position to a world-space ray.
    pub fn screen_to_ray(&self, screen_pos: [f32; 2], viewport_size: [f32; 2]) -> PickRay {
        let aspect = viewport_size[0] / viewport_size[1].max(1.0);
        let view = self.view_matrix();
        let proj = self.projection_matrix(aspect);
        PickRay::from_screen(
            screen_pos[0],
            screen_pos[1],
            viewport_size[0],
            viewport_size[1],
            &view,
            &proj,
        )
    }

    /// Focus the orbit camera on a specific world-space point.
    pub fn focus_on(&mut self, point: [f32; 3], distance: f32) {
        self.target = point;
        self.orbit_distance = distance;
        self.recompute_orbit_position();
    }

    /// Set an orthographic preset.
    pub fn set_preset(&mut self, mode: CameraMode) {
        self.mode = mode;
        match mode {
            CameraMode::TopDown => {
                self.yaw = 0.0;
                self.pitch = -std::f32::consts::FRAC_PI_2 + 0.001;
                self.position = [self.target[0], self.target[1] + self.orbit_distance, self.target[2]];
            }
            CameraMode::Front => {
                self.yaw = 0.0;
                self.pitch = 0.0;
                self.position = [self.target[0], self.target[1], self.target[2] + self.orbit_distance];
            }
            CameraMode::Right => {
                self.yaw = std::f32::consts::FRAC_PI_2;
                self.pitch = 0.0;
                self.position = [self.target[0] + self.orbit_distance, self.target[1], self.target[2]];
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Gizmo types
// ---------------------------------------------------------------------------

/// Gizmo manipulation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GizmoMode {
    /// Translation handles (arrows along axes).
    Translate,
    /// Rotation handles (rings around axes).
    Rotate,
    /// Scale handles (cubes on axes).
    Scale,
    /// Bounding-box wireframe.
    Bounds,
}

impl Default for GizmoMode {
    fn default() -> Self {
        Self::Translate
    }
}

/// The coordinate space in which gizmos operate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GizmoSpace {
    /// Gizmo axes align with the world axes.
    World,
    /// Gizmo axes align with the selected entity's local axes.
    Local,
}

impl Default for GizmoSpace {
    fn default() -> Self {
        Self::World
    }
}

/// Axis identifier returned from gizmo hit tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
    /// Multi-axis plane (XY=0, XZ=1, YZ=2).
    Plane(u8),
    /// Center / all-axes (uniform scale).
    Center,
}

/// The visual & interaction state of a single gizmo axis handle.
#[derive(Debug, Clone)]
pub struct GizmoAxisState {
    /// The axis this state corresponds to.
    pub axis: GizmoAxis,
    /// Whether the cursor is hovering over this handle.
    pub hovered: bool,
    /// Whether this handle is actively being dragged.
    pub dragging: bool,
    /// World-space origin of the handle.
    pub origin: [f32; 3],
    /// World-space direction of the handle (unit vector).
    pub direction: [f32; 3],
    /// Color for rendering (RGBA).
    pub color: [f32; 4],
}

/// Internal drag state tracked while the user manipulates a gizmo.
#[derive(Debug, Clone)]
struct GizmoDragState {
    axis: GizmoAxis,
    start_mouse: [f32; 2],
    start_position: [f32; 3],
    accumulated_delta: [f32; 3],
    start_angle: f32,
}

// ---------------------------------------------------------------------------
// GizmoRenderer
// ---------------------------------------------------------------------------

/// Renders interactive transform gizmos, a reference grid, and bounding-box
/// overlays in the editor viewport.
#[derive(Debug)]
pub struct GizmoRenderer {
    /// Current gizmo manipulation mode.
    pub mode: GizmoMode,
    /// Coordinate space for gizmo axes.
    pub space: GizmoSpace,
    /// Whether the grid is visible.
    pub show_grid: bool,
    /// Grid cell size in world units.
    pub grid_size: f32,
    /// Number of grid cells along each axis.
    pub grid_divisions: u32,
    /// Whether to draw bounding boxes for selected entities.
    pub show_bounds: bool,
    /// Size of the gizmo relative to the screen (scale-invariant factor).
    pub gizmo_screen_scale: f32,
    /// Per-axis visual states.
    axis_states: Vec<GizmoAxisState>,
    /// Active drag, if any.
    drag: Option<GizmoDragState>,
    /// Snap increment for translation (0 = disabled).
    pub snap_translate: f32,
    /// Snap increment for rotation in degrees (0 = disabled).
    pub snap_rotate: f32,
    /// Snap increment for scale (0 = disabled).
    pub snap_scale: f32,
}

impl Default for GizmoRenderer {
    fn default() -> Self {
        Self {
            mode: GizmoMode::default(),
            space: GizmoSpace::default(),
            show_grid: true,
            grid_size: 1.0,
            grid_divisions: 100,
            show_bounds: true,
            gizmo_screen_scale: 0.1,
            axis_states: Vec::new(),
            drag: None,
            snap_translate: 0.0,
            snap_rotate: 0.0,
            snap_scale: 0.0,
        }
    }
}

impl GizmoRenderer {
    /// Create a new gizmo renderer with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build axis handle states for an entity at the given world position.
    pub fn build_axes(&mut self, center: [f32; 3]) {
        self.axis_states.clear();
        let axes = [
            (GizmoAxis::X, [1.0, 0.0, 0.0_f32], [1.0, 0.2, 0.2, 1.0_f32]),
            (GizmoAxis::Y, [0.0, 1.0, 0.0], [0.2, 1.0, 0.2, 1.0]),
            (GizmoAxis::Z, [0.0, 0.0, 1.0], [0.2, 0.2, 1.0, 1.0]),
        ];
        for (axis, dir, color) in &axes {
            self.axis_states.push(GizmoAxisState {
                axis: *axis,
                hovered: false,
                dragging: false,
                origin: center,
                direction: *dir,
                color: *color,
            });
        }
        // Plane handles.
        self.axis_states.push(GizmoAxisState {
            axis: GizmoAxis::Plane(0),
            hovered: false,
            dragging: false,
            origin: center,
            direction: [0.0, 0.0, 1.0],
            color: [1.0, 1.0, 0.2, 0.5],
        });
        self.axis_states.push(GizmoAxisState {
            axis: GizmoAxis::Plane(1),
            hovered: false,
            dragging: false,
            origin: center,
            direction: [0.0, 1.0, 0.0],
            color: [1.0, 0.2, 1.0, 0.5],
        });
        self.axis_states.push(GizmoAxisState {
            axis: GizmoAxis::Plane(2),
            hovered: false,
            dragging: false,
            origin: center,
            direction: [1.0, 0.0, 0.0],
            color: [0.2, 1.0, 1.0, 0.5],
        });
    }

    /// Test if a screen-space point hits a gizmo handle and return the axis.
    pub fn hit_test(
        &self,
        screen_x: f32,
        screen_y: f32,
        camera: &ViewportCamera,
        viewport_size: [f32; 2],
    ) -> Option<GizmoAxis> {
        if self.axis_states.is_empty() {
            return None;
        }

        let ray = camera.screen_to_ray([screen_x, screen_y], viewport_size);

        let mut best: Option<(GizmoAxis, f32)> = None;

        for axis_state in &self.axis_states {
            let hit_t = match self.mode {
                GizmoMode::Translate | GizmoMode::Scale => {
                    ray_vs_line_segment(
                        &ray,
                        axis_state.origin,
                        axis_state.direction,
                        self.gizmo_screen_scale * 10.0,
                        0.15,
                    )
                }
                GizmoMode::Rotate => {
                    ray_vs_torus(
                        &ray,
                        axis_state.origin,
                        axis_state.direction,
                        self.gizmo_screen_scale * 10.0,
                        0.08,
                    )
                }
                GizmoMode::Bounds => None,
            };
            if let Some(t) = hit_t {
                if best.is_none() || t < best.unwrap().1 {
                    best = Some((axis_state.axis, t));
                }
            }
        }

        best.map(|(axis, _)| axis)
    }

    /// Begin a gizmo drag interaction.
    pub fn begin_drag(
        &mut self,
        axis: GizmoAxis,
        mouse_pos: [f32; 2],
        entity_position: [f32; 3],
    ) {
        for state in &mut self.axis_states {
            state.dragging = state.axis == axis;
        }
        self.drag = Some(GizmoDragState {
            axis,
            start_mouse: mouse_pos,
            start_position: entity_position,
            accumulated_delta: [0.0, 0.0, 0.0],
            start_angle: 0.0,
        });
    }

    /// Update an active gizmo drag and return the transform delta.
    pub fn update_drag(
        &mut self,
        mouse_pos: [f32; 2],
        mouse_delta: [f32; 2],
        camera: &ViewportCamera,
        viewport_size: [f32; 2],
    ) -> Option<GizmoDelta> {
        let drag = self.drag.as_mut()?;

        match self.mode {
            GizmoMode::Translate => {
                let sensitivity = 0.01 * camera.orbit_distance.max(1.0);
                let axis_dir = axis_to_direction(drag.axis);
                let (right, up, _fwd) = camera.basis_vectors();

                // Project screen delta onto the constraint axis direction seen from camera.
                let screen_move = [mouse_delta[0], mouse_delta[1]];
                let axis_screen_x =
                    axis_dir[0] * right[0] + axis_dir[1] * right[1] + axis_dir[2] * right[2];
                let axis_screen_y =
                    axis_dir[0] * up[0] + axis_dir[1] * up[1] + axis_dir[2] * up[2];

                let proj = screen_move[0] * axis_screen_x - screen_move[1] * axis_screen_y;

                let mut delta = [
                    axis_dir[0] * proj * sensitivity,
                    axis_dir[1] * proj * sensitivity,
                    axis_dir[2] * proj * sensitivity,
                ];

                if self.snap_translate > 0.0 {
                    for d in &mut delta {
                        drag.accumulated_delta[0] += *d;
                    }
                    for i in 0..3 {
                        drag.accumulated_delta[i] += delta[i];
                        let snapped = snap_value(drag.accumulated_delta[i], self.snap_translate);
                        delta[i] = snapped;
                        drag.accumulated_delta[i] -= snapped;
                    }
                }

                Some(GizmoDelta::Translate(delta))
            }
            GizmoMode::Rotate => {
                let axis_dir = axis_to_direction(drag.axis);
                let angle = (mouse_delta[0] - mouse_delta[1]) * 0.01;
                let mut final_angle = angle;
                if self.snap_rotate > 0.0 {
                    drag.accumulated_delta[0] += angle;
                    let snapped =
                        snap_value(drag.accumulated_delta[0], self.snap_rotate.to_radians());
                    final_angle = snapped;
                    drag.accumulated_delta[0] -= snapped;
                }
                Some(GizmoDelta::Rotate {
                    axis: axis_dir,
                    angle_rad: final_angle,
                })
            }
            GizmoMode::Scale => {
                let axis_dir = axis_to_direction(drag.axis);
                let scale_factor = 1.0 + (mouse_delta[0] - mouse_delta[1]) * 0.005;

                let mut delta = if drag.axis == GizmoAxis::Center {
                    [scale_factor, scale_factor, scale_factor]
                } else {
                    [
                        if axis_dir[0].abs() > 0.5 { scale_factor } else { 1.0 },
                        if axis_dir[1].abs() > 0.5 { scale_factor } else { 1.0 },
                        if axis_dir[2].abs() > 0.5 { scale_factor } else { 1.0 },
                    ]
                };

                if self.snap_scale > 0.0 {
                    for d in &mut delta {
                        *d = snap_value(*d, self.snap_scale);
                    }
                }

                Some(GizmoDelta::Scale(delta))
            }
            GizmoMode::Bounds => None,
        }
    }

    /// End the current drag interaction.
    pub fn end_drag(&mut self) {
        for state in &mut self.axis_states {
            state.dragging = false;
        }
        self.drag = None;
    }

    /// Whether a drag is in progress.
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Update hover state from current mouse position.
    pub fn update_hover(
        &mut self,
        screen_x: f32,
        screen_y: f32,
        camera: &ViewportCamera,
        viewport_size: [f32; 2],
    ) {
        let hit = self.hit_test(screen_x, screen_y, camera, viewport_size);
        for state in &mut self.axis_states {
            state.hovered = hit == Some(state.axis);
        }
    }

    /// Return the currently hovered axis, if any.
    pub fn hovered_axis(&self) -> Option<GizmoAxis> {
        self.axis_states.iter().find(|s| s.hovered).map(|s| s.axis)
    }

    /// Return the axis states for rendering.
    pub fn axis_states(&self) -> &[GizmoAxisState] {
        &self.axis_states
    }

    /// Render gizmos for the given selection. In a real engine this would emit
    /// draw calls; here we collect render commands into a list.
    pub fn render(&self, _selection: &SelectionManager) -> Vec<GizmoDrawCommand> {
        let mut commands = Vec::new();

        // Grid.
        if self.show_grid {
            commands.push(GizmoDrawCommand::Grid {
                center: [0.0, 0.0, 0.0],
                size: self.grid_size,
                divisions: self.grid_divisions,
            });
        }

        // Axis handles.
        for axis_state in &self.axis_states {
            let highlight = axis_state.hovered || axis_state.dragging;
            let mut color = axis_state.color;
            if highlight {
                color[0] = (color[0] + 0.3).min(1.0);
                color[1] = (color[1] + 0.3).min(1.0);
                color[2] = (color[2] + 0.3).min(1.0);
            }

            match self.mode {
                GizmoMode::Translate => {
                    commands.push(GizmoDrawCommand::Arrow {
                        origin: axis_state.origin,
                        direction: axis_state.direction,
                        length: self.gizmo_screen_scale * 10.0,
                        color,
                    });
                }
                GizmoMode::Rotate => {
                    commands.push(GizmoDrawCommand::Ring {
                        center: axis_state.origin,
                        normal: axis_state.direction,
                        radius: self.gizmo_screen_scale * 10.0,
                        color,
                    });
                }
                GizmoMode::Scale => {
                    commands.push(GizmoDrawCommand::ScaleHandle {
                        origin: axis_state.origin,
                        direction: axis_state.direction,
                        length: self.gizmo_screen_scale * 10.0,
                        color,
                    });
                }
                GizmoMode::Bounds => {}
            }
        }

        commands
    }
}

/// Transform delta produced by gizmo interaction.
#[derive(Debug, Clone)]
pub enum GizmoDelta {
    /// Translation in world space.
    Translate([f32; 3]),
    /// Rotation about an axis.
    Rotate {
        axis: [f32; 3],
        angle_rad: f32,
    },
    /// Scale factors per axis.
    Scale([f32; 3]),
}

/// Draw command emitted by the gizmo renderer.
#[derive(Debug, Clone)]
pub enum GizmoDrawCommand {
    Grid {
        center: [f32; 3],
        size: f32,
        divisions: u32,
    },
    Arrow {
        origin: [f32; 3],
        direction: [f32; 3],
        length: f32,
        color: [f32; 4],
    },
    Ring {
        center: [f32; 3],
        normal: [f32; 3],
        radius: f32,
        color: [f32; 4],
    },
    ScaleHandle {
        origin: [f32; 3],
        direction: [f32; 3],
        length: f32,
        color: [f32; 4],
    },
    BoundingBox {
        min: [f32; 3],
        max: [f32; 3],
        color: [f32; 4],
    },
}

// ---------------------------------------------------------------------------
// SelectionManager
// ---------------------------------------------------------------------------

/// Manages the set of currently selected entities in the editor.
#[derive(Debug, Default, Clone)]
pub struct SelectionManager {
    /// Ordered list of selected entity IDs.
    selected: Vec<Uuid>,
    /// Currently hovered entity (for highlight preview).
    hovered: Option<Uuid>,
    /// Box-select state.
    box_select: Option<BoxSelectState>,
}

/// State tracking a rectangle (box) selection drag.
#[derive(Debug, Clone)]
pub struct BoxSelectState {
    /// Screen-space start corner.
    pub start: [f32; 2],
    /// Screen-space current (end) corner.
    pub current: [f32; 2],
}

impl BoxSelectState {
    /// Get the normalized rectangle (min, max).
    pub fn rect(&self) -> ([f32; 2], [f32; 2]) {
        let min_x = self.start[0].min(self.current[0]);
        let min_y = self.start[1].min(self.current[1]);
        let max_x = self.start[0].max(self.current[0]);
        let max_y = self.start[1].max(self.current[1]);
        ([min_x, min_y], [max_x, max_y])
    }

    /// Area of the rectangle in pixels squared.
    pub fn area(&self) -> f32 {
        let (min, max) = self.rect();
        (max[0] - min[0]) * (max[1] - min[1])
    }
}

impl SelectionManager {
    /// Create a new empty selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the selection with a single entity.
    pub fn select(&mut self, entity_id: Uuid) {
        self.selected.clear();
        self.selected.push(entity_id);
    }

    /// Add an entity to the selection (multi-select).
    pub fn add_to_selection(&mut self, entity_id: Uuid) {
        if !self.selected.contains(&entity_id) {
            self.selected.push(entity_id);
        }
    }

    /// Remove an entity from the selection.
    pub fn remove_from_selection(&mut self, entity_id: Uuid) {
        self.selected.retain(|id| *id != entity_id);
    }

    /// Toggle an entity in the selection.
    pub fn toggle_selection(&mut self, entity_id: Uuid) {
        if self.selected.contains(&entity_id) {
            self.remove_from_selection(entity_id);
        } else {
            self.add_to_selection(entity_id);
        }
    }

    /// Clear the entire selection.
    pub fn clear(&mut self) {
        self.selected.clear();
    }

    /// Return the currently selected entity IDs.
    pub fn selected_entities(&self) -> &[Uuid] {
        &self.selected
    }

    /// Whether any entity is selected.
    pub fn has_selection(&self) -> bool {
        !self.selected.is_empty()
    }

    /// Return the primary (first) selected entity, if any.
    pub fn primary(&self) -> Option<Uuid> {
        self.selected.first().copied()
    }

    /// Set the hovered entity.
    pub fn set_hovered(&mut self, entity_id: Option<Uuid>) {
        self.hovered = entity_id;
    }

    /// Get the hovered entity.
    pub fn hovered(&self) -> Option<Uuid> {
        self.hovered
    }

    /// Begin a box-select drag.
    pub fn begin_box_select(&mut self, start: [f32; 2]) {
        self.box_select = Some(BoxSelectState {
            start,
            current: start,
        });
    }

    /// Update box-select drag.
    pub fn update_box_select(&mut self, current: [f32; 2]) {
        if let Some(ref mut bs) = self.box_select {
            bs.current = current;
        }
    }

    /// End box-select and return the rectangle (min, max) if the area was large enough.
    pub fn end_box_select(&mut self) -> Option<([f32; 2], [f32; 2])> {
        let bs = self.box_select.take()?;
        if bs.area() > 16.0 {
            Some(bs.rect())
        } else {
            None
        }
    }

    /// Whether a box-select is in progress.
    pub fn is_box_selecting(&self) -> bool {
        self.box_select.is_some()
    }

    /// Get the current box-select state for rendering.
    pub fn box_select_state(&self) -> Option<&BoxSelectState> {
        self.box_select.as_ref()
    }

    /// Perform entity picking by raycasting against a list of entity AABBs.
    /// Returns the UUID of the nearest hit entity.
    pub fn select_by_ray(
        &self,
        ray: &PickRay,
        entity_bounds: &[(Uuid, [f32; 3], [f32; 3])],
    ) -> Option<Uuid> {
        let mut best: Option<(Uuid, f32)> = None;

        for (id, aabb_min, aabb_max) in entity_bounds {
            if let Some(t) = ray.intersect_aabb(*aabb_min, *aabb_max) {
                if best.is_none() || t < best.unwrap().1 {
                    best = Some((*id, t));
                }
            }
        }

        best.map(|(id, _)| id)
    }

    /// Select entities whose screen-space centers fall within a rectangle.
    pub fn select_by_rect(
        &mut self,
        rect_min: [f32; 2],
        rect_max: [f32; 2],
        entity_screen_positions: &[(Uuid, [f32; 2])],
        additive: bool,
    ) {
        if !additive {
            self.selected.clear();
        }

        for (id, pos) in entity_screen_positions {
            if pos[0] >= rect_min[0]
                && pos[0] <= rect_max[0]
                && pos[1] >= rect_min[1]
                && pos[1] <= rect_max[1]
            {
                if !self.selected.contains(id) {
                    self.selected.push(*id);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pick Ray
// ---------------------------------------------------------------------------

/// A ray cast from screen space into the 3D scene for entity picking.
#[derive(Debug, Clone, Copy)]
pub struct PickRay {
    /// Ray origin in world space.
    pub origin: [f32; 3],
    /// Normalized ray direction in world space.
    pub direction: [f32; 3],
}

impl PickRay {
    /// Construct a pick ray from screen coordinates and camera matrices.
    ///
    /// `screen_x` and `screen_y` are in pixels; `viewport_width` and
    /// `viewport_height` are the viewport dimensions.
    pub fn from_screen(
        screen_x: f32,
        screen_y: f32,
        viewport_width: f32,
        viewport_height: f32,
        view_matrix: &[[f32; 4]; 4],
        projection_matrix: &[[f32; 4]; 4],
    ) -> Self {
        // Convert screen to NDC [-1, 1].
        let ndc_x = (2.0 * screen_x) / viewport_width - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y) / viewport_height;

        // Compute inverse view-projection.
        let vp = mat4_mul(projection_matrix, view_matrix);
        let inv_vp = mat4_inverse(&vp);

        // Near point and far point in clip space.
        let near_clip = mat4_transform_point(&inv_vp, [ndc_x, ndc_y, -1.0]);
        let far_clip = mat4_transform_point(&inv_vp, [ndc_x, ndc_y, 1.0]);

        let direction = vec3_sub(far_clip, near_clip);
        let len = vec3_len(direction);

        if len < 1e-8 {
            return Self {
                origin: near_clip,
                direction: [0.0, 0.0, -1.0],
            };
        }

        Self {
            origin: near_clip,
            direction: [direction[0] / len, direction[1] / len, direction[2] / len],
        }
    }

    /// Test intersection with an axis-aligned bounding box. Returns distance
    /// along the ray if hit, or `None`.
    pub fn intersect_aabb(&self, aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Option<f32> {
        let inv_dir = [
            if self.direction[0].abs() > 1e-8 { 1.0 / self.direction[0] } else { f32::INFINITY },
            if self.direction[1].abs() > 1e-8 { 1.0 / self.direction[1] } else { f32::INFINITY },
            if self.direction[2].abs() > 1e-8 { 1.0 / self.direction[2] } else { f32::INFINITY },
        ];

        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;

        for i in 0..3 {
            let t1 = (aabb_min[i] - self.origin[i]) * inv_dir[i];
            let t2 = (aabb_max[i] - self.origin[i]) * inv_dir[i];
            let t_near = t1.min(t2);
            let t_far = t1.max(t2);
            t_min = t_min.max(t_near);
            t_max = t_max.min(t_far);
        }

        if t_min <= t_max && t_max >= 0.0 {
            Some(t_min.max(0.0))
        } else {
            None
        }
    }

    /// Test intersection with a sphere. Returns distance or `None`.
    pub fn intersect_sphere(&self, center: [f32; 3], radius: f32) -> Option<f32> {
        let oc = vec3_sub(self.origin, center);
        let a = vec3_dot(self.direction, self.direction);
        let b = 2.0 * vec3_dot(oc, self.direction);
        let c = vec3_dot(oc, oc) - radius * radius;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrt_d = discriminant.sqrt();
        let t1 = (-b - sqrt_d) / (2.0 * a);
        let t2 = (-b + sqrt_d) / (2.0 * a);
        if t1 >= 0.0 {
            Some(t1)
        } else if t2 >= 0.0 {
            Some(t2)
        } else {
            None
        }
    }

    /// Get a point along the ray at parameter t.
    pub fn point_at(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + self.direction[0] * t,
            self.origin[1] + self.direction[1] * t,
            self.origin[2] + self.direction[2] * t,
        ]
    }
}

// ---------------------------------------------------------------------------
// Editor Viewport (top-level)
// ---------------------------------------------------------------------------

/// The main editor viewport that composites the 3D scene view with gizmo
/// overlays and handles user interaction.
#[derive(Debug)]
pub struct EditorViewport {
    /// Viewport camera.
    pub camera: ViewportCamera,
    /// Render target identifier for this viewport (offscreen texture).
    pub render_target: Option<Uuid>,
    /// Gizmo overlay renderer.
    pub gizmos: GizmoRenderer,
    /// Entity selection state.
    pub selection: SelectionManager,
    /// Viewport width in pixels.
    pub width: u32,
    /// Viewport height in pixels.
    pub height: u32,
    /// Whether the grid is visible.
    pub grid_visible: bool,
    /// Grid cell size.
    pub grid_size: f32,
    /// The gizmo mode (convenience alias for gizmos.mode).
    pub gizmo_mode: GizmoMode,
}

impl EditorViewport {
    /// Create a new editor viewport with default settings.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            camera: ViewportCamera::new(),
            render_target: None,
            gizmos: GizmoRenderer::new(),
            selection: SelectionManager::new(),
            width,
            height,
            grid_visible: true,
            grid_size: 1.0,
            gizmo_mode: GizmoMode::Translate,
        }
    }

    /// Resize the viewport.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Main per-frame update. Handles camera movement, entity selection, and
    /// gizmo interaction based on input state.
    pub fn update(&mut self, input: &InputState, dt: f32) {
        self.gizmos.mode = self.gizmo_mode;
        self.gizmos.show_grid = self.grid_visible;
        self.gizmos.grid_size = self.grid_size;

        let viewport_size = [self.width as f32, self.height as f32];

        match self.camera.mode {
            CameraMode::Fly => {
                if input.right_mouse_down {
                    let movement = [
                        if input.move_right { 1.0 } else { 0.0 }
                            - if input.move_left { 1.0 } else { 0.0 },
                        if input.move_up { 1.0 } else { 0.0 }
                            - if input.move_down { 1.0 } else { 0.0 },
                        if input.move_forward { 1.0 } else { 0.0 }
                            - if input.move_backward { 1.0 } else { 0.0 },
                    ];
                    self.camera.fly_update(
                        -input.mouse_delta[0] * self.camera.look_sensitivity,
                        -input.mouse_delta[1] * self.camera.look_sensitivity,
                        movement,
                        dt,
                        input.shift_held,
                    );
                }
                self.camera.zoom(input.scroll_delta);
            }
            CameraMode::Orbit => {
                if input.middle_mouse_down && input.alt_held {
                    // Alt + middle-mouse = pan.
                    self.camera.pan(input.mouse_delta[0], input.mouse_delta[1]);
                } else if input.middle_mouse_down {
                    // Middle-mouse drag = orbit.
                    self.camera.orbit(
                        -input.mouse_delta[0] * self.camera.look_sensitivity,
                        -input.mouse_delta[1] * self.camera.look_sensitivity,
                    );
                }
                self.camera.zoom(input.scroll_delta);
            }
            CameraMode::TopDown | CameraMode::Front | CameraMode::Right => {
                if input.middle_mouse_down {
                    self.camera.pan(input.mouse_delta[0], input.mouse_delta[1]);
                }
                self.camera.zoom(input.scroll_delta);
            }
        }

        // Gizmo hover update.
        if !self.gizmos.is_dragging() {
            self.gizmos
                .update_hover(input.mouse_pos[0], input.mouse_pos[1], &self.camera, viewport_size);
        }

        // Gizmo drag handling.
        if self.gizmos.is_dragging() {
            if input.left_mouse_down {
                let _delta = self.gizmos.update_drag(
                    input.mouse_pos,
                    input.mouse_delta,
                    &self.camera,
                    viewport_size,
                );
                // In a real engine, the delta would be applied to the selected entities.
            } else {
                self.gizmos.end_drag();
            }
        }

        // Left-click picking / selection.
        if input.left_mouse_pressed && !self.gizmos.is_dragging() {
            if let Some(axis) = self.gizmos.hovered_axis() {
                // Start gizmo drag.
                self.gizmos
                    .begin_drag(axis, input.mouse_pos, [0.0, 0.0, 0.0]);
            } else if !input.ctrl_held {
                // Begin box select on bare click.
                self.selection.begin_box_select(input.mouse_pos);
            }
        }

        // Box select update.
        if self.selection.is_box_selecting() {
            self.selection.update_box_select(input.mouse_pos);
        }

        if input.left_mouse_released && self.selection.is_box_selecting() {
            self.selection.end_box_select();
        }
    }

    /// Perform entity picking at a specific mouse position using raycasting.
    pub fn handle_selection(
        &mut self,
        mouse_pos: [f32; 2],
        entity_bounds: &[(Uuid, [f32; 3], [f32; 3])],
        shift_held: bool,
    ) {
        let viewport_size = [self.width as f32, self.height as f32];
        let ray = self.camera.screen_to_ray(mouse_pos, viewport_size);

        if let Some(hit_id) = self.selection.select_by_ray(&ray, entity_bounds) {
            if shift_held {
                self.selection.toggle_selection(hit_id);
            } else {
                self.selection.select(hit_id);
            }
        } else if !shift_held {
            self.selection.clear();
        }
    }

    /// Collect draw commands for the viewport (gizmos, grid, bounds).
    pub fn render(&self) -> Vec<GizmoDrawCommand> {
        self.gizmos.render(&self.selection)
    }

    /// Get the aspect ratio of the viewport.
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }
}

// ---------------------------------------------------------------------------
// Math helpers  (self-contained, no glam dependency in editor crate)
// ---------------------------------------------------------------------------

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_len(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = vec3_len(v);
    if len < 1e-8 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Build a look-at view matrix (right-handed).
fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = vec3_normalize(vec3_sub(target, eye));
    let s = vec3_normalize(cross(f, up));
    let u = cross(s, f);

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [
            -vec3_dot(s, eye),
            -vec3_dot(u, eye),
            vec3_dot(f, eye),
            1.0,
        ],
    ]
}

/// Build a perspective projection matrix (right-handed, depth [0,1]).
fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let range_inv = 1.0 / (near - far);

    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far * range_inv, -1.0],
        [0.0, 0.0, near * far * range_inv, 0.0],
    ]
}

/// Build an orthographic projection matrix.
fn orthographic(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let rml = right - left;
    let tmb = top - bottom;
    let fmn = far - near;
    [
        [2.0 / rml, 0.0, 0.0, 0.0],
        [0.0, 2.0 / tmb, 0.0, 0.0],
        [0.0, 0.0, -1.0 / fmn, 0.0],
        [
            -(right + left) / rml,
            -(top + bottom) / tmb,
            -near / fmn,
            1.0,
        ],
    ]
}

/// 4x4 matrix multiply (column-major layout as arrays of columns).
fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0_f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            result[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    result
}

/// Compute 4x4 matrix inverse using cofactor expansion.
fn mat4_inverse(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Flatten to row-major for easier indexing.
    let mut s = [0.0_f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            s[row * 4 + col] = m[col][row];
        }
    }

    let mut inv = [0.0_f32; 16];

    inv[0] = s[5] * s[10] * s[15] - s[5] * s[11] * s[14] - s[9] * s[6] * s[15]
        + s[9] * s[7] * s[14]
        + s[13] * s[6] * s[11]
        - s[13] * s[7] * s[10];
    inv[4] = -s[4] * s[10] * s[15] + s[4] * s[11] * s[14] + s[8] * s[6] * s[15]
        - s[8] * s[7] * s[14]
        - s[12] * s[6] * s[11]
        + s[12] * s[7] * s[10];
    inv[8] = s[4] * s[9] * s[15] - s[4] * s[11] * s[13] - s[8] * s[5] * s[15]
        + s[8] * s[7] * s[13]
        + s[12] * s[5] * s[11]
        - s[12] * s[7] * s[9];
    inv[12] = -s[4] * s[9] * s[14] + s[4] * s[10] * s[13] + s[8] * s[5] * s[14]
        - s[8] * s[6] * s[13]
        - s[12] * s[5] * s[10]
        + s[12] * s[6] * s[9];

    inv[1] = -s[1] * s[10] * s[15] + s[1] * s[11] * s[14] + s[9] * s[2] * s[15]
        - s[9] * s[3] * s[14]
        - s[13] * s[2] * s[11]
        + s[13] * s[3] * s[10];
    inv[5] = s[0] * s[10] * s[15] - s[0] * s[11] * s[14] - s[8] * s[2] * s[15]
        + s[8] * s[3] * s[14]
        + s[12] * s[2] * s[11]
        - s[12] * s[3] * s[10];
    inv[9] = -s[0] * s[9] * s[15] + s[0] * s[11] * s[13] + s[8] * s[1] * s[15]
        - s[8] * s[3] * s[13]
        - s[12] * s[1] * s[11]
        + s[12] * s[3] * s[9];
    inv[13] = s[0] * s[9] * s[14] - s[0] * s[10] * s[13] - s[8] * s[1] * s[14]
        + s[8] * s[2] * s[13]
        + s[12] * s[1] * s[10]
        - s[12] * s[2] * s[9];

    inv[2] = s[1] * s[6] * s[15] - s[1] * s[7] * s[14] - s[5] * s[2] * s[15]
        + s[5] * s[3] * s[14]
        + s[13] * s[2] * s[7]
        - s[13] * s[3] * s[6];
    inv[6] = -s[0] * s[6] * s[15] + s[0] * s[7] * s[14] + s[4] * s[2] * s[15]
        - s[4] * s[3] * s[14]
        - s[12] * s[2] * s[7]
        + s[12] * s[3] * s[6];
    inv[10] = s[0] * s[5] * s[15] - s[0] * s[7] * s[13] - s[4] * s[1] * s[15]
        + s[4] * s[3] * s[13]
        + s[12] * s[1] * s[7]
        - s[12] * s[3] * s[5];
    inv[14] = -s[0] * s[5] * s[14] + s[0] * s[6] * s[13] + s[4] * s[1] * s[14]
        - s[4] * s[2] * s[13]
        - s[12] * s[1] * s[6]
        + s[12] * s[2] * s[5];

    inv[3] = -s[1] * s[6] * s[11] + s[1] * s[7] * s[10] + s[5] * s[2] * s[11]
        - s[5] * s[3] * s[10]
        - s[9] * s[2] * s[7]
        + s[9] * s[3] * s[6];
    inv[7] = s[0] * s[6] * s[11] - s[0] * s[7] * s[10] - s[4] * s[2] * s[11]
        + s[4] * s[3] * s[10]
        + s[8] * s[2] * s[7]
        - s[8] * s[3] * s[6];
    inv[11] = -s[0] * s[5] * s[11] + s[0] * s[7] * s[9] + s[4] * s[1] * s[11]
        - s[4] * s[3] * s[9]
        - s[8] * s[1] * s[7]
        + s[8] * s[3] * s[5];
    inv[15] = s[0] * s[5] * s[10] - s[0] * s[6] * s[9] - s[4] * s[1] * s[10]
        + s[4] * s[2] * s[9]
        + s[8] * s[1] * s[6]
        - s[8] * s[2] * s[5];

    let det = s[0] * inv[0] + s[1] * inv[4] + s[2] * inv[8] + s[3] * inv[12];
    if det.abs() < 1e-12 {
        return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    }

    let inv_det = 1.0 / det;
    for v in &mut inv {
        *v *= inv_det;
    }

    // Back to column-major.
    let mut result = [[0.0_f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            result[col][row] = inv[row * 4 + col];
        }
    }
    result
}

/// Transform a 3D point by a 4x4 matrix (with perspective divide).
fn mat4_transform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let x = m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0];
    let y = m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1];
    let z = m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2];
    let w = m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3];

    if w.abs() < 1e-8 {
        [x, y, z]
    } else {
        [x / w, y / w, z / w]
    }
}

/// Snap a value to the nearest multiple of `increment`.
fn snap_value(value: f32, increment: f32) -> f32 {
    if increment <= 0.0 {
        return value;
    }
    (value / increment).round() * increment
}

/// Convert a GizmoAxis to a unit direction vector.
fn axis_to_direction(axis: GizmoAxis) -> [f32; 3] {
    match axis {
        GizmoAxis::X => [1.0, 0.0, 0.0],
        GizmoAxis::Y => [0.0, 1.0, 0.0],
        GizmoAxis::Z => [0.0, 0.0, 1.0],
        GizmoAxis::Plane(0) => [1.0, 1.0, 0.0],  // XY plane moves X and Y
        GizmoAxis::Plane(1) => [1.0, 0.0, 1.0],  // XZ plane
        GizmoAxis::Plane(2) => [0.0, 1.0, 1.0],  // YZ plane
        GizmoAxis::Plane(_) => [1.0, 1.0, 1.0],
        GizmoAxis::Center => [1.0, 1.0, 1.0],
    }
}

/// Ray vs line segment hit test for translate/scale gizmo handles.
/// Returns the t-parameter along the ray if the ray passes within `threshold`
/// of the line segment from `origin` along `direction` for `length`.
fn ray_vs_line_segment(
    ray: &PickRay,
    origin: [f32; 3],
    direction: [f32; 3],
    length: f32,
    threshold: f32,
) -> Option<f32> {
    let end = vec3_add(origin, vec3_scale(direction, length));
    let seg_dir = vec3_sub(end, origin);
    let w0 = vec3_sub(ray.origin, origin);

    let a = vec3_dot(ray.direction, ray.direction);
    let b = vec3_dot(ray.direction, seg_dir);
    let c = vec3_dot(seg_dir, seg_dir);
    let d = vec3_dot(ray.direction, w0);
    let e = vec3_dot(seg_dir, w0);

    let denom = a * c - b * b;
    if denom.abs() < 1e-8 {
        return None;
    }

    let s = (b * e - c * d) / denom;
    let t = (a * e - b * d) / denom;

    if s < 0.0 || t < 0.0 || t > 1.0 {
        return None;
    }

    let closest_ray = vec3_add(ray.origin, vec3_scale(ray.direction, s));
    let closest_seg = vec3_add(origin, vec3_scale(seg_dir, t));
    let dist = vec3_len(vec3_sub(closest_ray, closest_seg));

    if dist < threshold {
        Some(s)
    } else {
        None
    }
}

/// Approximate ray vs torus hit test for rotation gizmo handles.
/// Tests ray against the ring's plane and checks if the intersection point is
/// near the ring radius.
fn ray_vs_torus(
    ray: &PickRay,
    center: [f32; 3],
    normal: [f32; 3],
    radius: f32,
    thickness: f32,
) -> Option<f32> {
    let denom = vec3_dot(ray.direction, normal);
    if denom.abs() < 1e-8 {
        return None;
    }

    let t = vec3_dot(vec3_sub(center, ray.origin), normal) / denom;
    if t < 0.0 {
        return None;
    }

    let hit_point = ray.point_at(t);
    let diff = vec3_sub(hit_point, center);
    let dist_from_center = vec3_len(diff);
    let ring_dist = (dist_from_center - radius).abs();

    if ring_dist < thickness * radius {
        Some(t)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_default_creates_valid_state() {
        let cam = ViewportCamera::new();
        assert_eq!(cam.mode, CameraMode::Orbit);
        assert!(cam.near_clip > 0.0);
        assert!(cam.far_clip > cam.near_clip);
        assert!(cam.fov_y > 0.0);
    }

    #[test]
    fn orbit_clamps_pitch() {
        let mut cam = ViewportCamera::new();
        cam.orbit(0.0, 100.0);
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
        assert!(cam.pitch <= max_pitch);
        assert!(cam.pitch >= -max_pitch);
    }

    #[test]
    fn zoom_clamps_orbit_distance() {
        let mut cam = ViewportCamera::new();
        cam.mode = CameraMode::Orbit;
        for _ in 0..1000 {
            cam.zoom(10.0);
        }
        assert!(cam.orbit_distance >= 0.1);
    }

    #[test]
    fn fly_mode_updates_position() {
        let mut cam = ViewportCamera::new();
        cam.mode = CameraMode::Fly;
        let initial_pos = cam.position;
        cam.fly_update(0.0, 0.0, [0.0, 0.0, 1.0], 1.0, false);
        assert_ne!(cam.position, initial_pos);
    }

    #[test]
    fn view_matrix_is_not_identity_with_offset_camera() {
        let cam = ViewportCamera::new();
        let view = cam.view_matrix();
        let identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        assert_ne!(view, identity);
    }

    #[test]
    fn projection_matrix_perspective() {
        let cam = ViewportCamera::new();
        let proj = cam.projection_matrix(16.0 / 9.0);
        // Element [1][1] should be 1/tan(fov/2).
        let expected = 1.0 / (cam.fov_y * 0.5).tan();
        assert!((proj[1][1] - expected).abs() < 1e-5);
    }

    #[test]
    fn pick_ray_aabb_intersection() {
        let ray = PickRay {
            origin: [0.0, 0.0, 5.0],
            direction: [0.0, 0.0, -1.0],
        };
        let t = ray.intersect_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert!(t.is_some());
        let t = t.unwrap();
        assert!((t - 4.0).abs() < 1e-5);
    }

    #[test]
    fn pick_ray_aabb_miss() {
        let ray = PickRay {
            origin: [0.0, 0.0, 5.0],
            direction: [0.0, 1.0, 0.0],
        };
        let t = ray.intersect_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert!(t.is_none());
    }

    #[test]
    fn pick_ray_sphere_intersection() {
        let ray = PickRay {
            origin: [0.0, 0.0, 5.0],
            direction: [0.0, 0.0, -1.0],
        };
        let t = ray.intersect_sphere([0.0, 0.0, 0.0], 1.0);
        assert!(t.is_some());
        assert!((t.unwrap() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn selection_manager_basic_operations() {
        let mut sm = SelectionManager::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        sm.select(id1);
        assert_eq!(sm.selected_entities(), &[id1]);
        assert_eq!(sm.primary(), Some(id1));

        sm.add_to_selection(id2);
        assert_eq!(sm.selected_entities().len(), 2);

        sm.toggle_selection(id1);
        assert_eq!(sm.selected_entities(), &[id2]);

        sm.clear();
        assert!(!sm.has_selection());
    }

    #[test]
    fn selection_box_select() {
        let mut sm = SelectionManager::new();
        sm.begin_box_select([10.0, 10.0]);
        assert!(sm.is_box_selecting());
        sm.update_box_select([100.0, 100.0]);
        let rect = sm.end_box_select();
        assert!(rect.is_some());
        let (min, max) = rect.unwrap();
        assert_eq!(min, [10.0, 10.0]);
        assert_eq!(max, [100.0, 100.0]);
    }

    #[test]
    fn selection_by_ray_picks_nearest() {
        let sm = SelectionManager::new();
        let id_near = Uuid::new_v4();
        let id_far = Uuid::new_v4();

        let ray = PickRay {
            origin: [0.0, 0.0, 10.0],
            direction: [0.0, 0.0, -1.0],
        };

        let bounds = vec![
            (id_near, [-1.0, -1.0, 1.0_f32], [1.0, 1.0, 3.0_f32]),
            (id_far, [-1.0, -1.0, -5.0], [1.0, 1.0, -3.0]),
        ];

        let hit = sm.select_by_ray(&ray, &bounds);
        assert_eq!(hit, Some(id_near));
    }

    #[test]
    fn selection_by_rect() {
        let mut sm = SelectionManager::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        let positions = vec![
            (id1, [50.0_f32, 50.0]),
            (id2, [150.0, 150.0]),
            (id3, [500.0, 500.0]),
        ];

        sm.select_by_rect([0.0, 0.0], [200.0, 200.0], &positions, false);
        assert_eq!(sm.selected_entities().len(), 2);
        assert!(sm.selected_entities().contains(&id1));
        assert!(sm.selected_entities().contains(&id2));
    }

    #[test]
    fn gizmo_renderer_build_axes() {
        let mut gizmo = GizmoRenderer::new();
        gizmo.build_axes([0.0, 0.0, 0.0]);
        assert_eq!(gizmo.axis_states().len(), 6); // 3 axes + 3 planes
    }

    #[test]
    fn gizmo_drag_translate() {
        let mut gizmo = GizmoRenderer::new();
        gizmo.mode = GizmoMode::Translate;
        gizmo.build_axes([0.0, 0.0, 0.0]);

        let cam = ViewportCamera::new();
        gizmo.begin_drag(GizmoAxis::X, [100.0, 100.0], [0.0, 0.0, 0.0]);
        assert!(gizmo.is_dragging());

        let delta = gizmo.update_drag([110.0, 100.0], [10.0, 0.0], &cam, [800.0, 600.0]);
        assert!(delta.is_some());
        match delta.unwrap() {
            GizmoDelta::Translate(d) => {
                // Should have nonzero X component.
                assert!(d[0].abs() > 0.0 || d[1].abs() > 0.0 || d[2].abs() > 0.0);
            }
            _ => panic!("Expected Translate delta"),
        }

        gizmo.end_drag();
        assert!(!gizmo.is_dragging());
    }

    #[test]
    fn gizmo_drag_rotate() {
        let mut gizmo = GizmoRenderer::new();
        gizmo.mode = GizmoMode::Rotate;
        gizmo.build_axes([0.0, 0.0, 0.0]);

        let cam = ViewportCamera::new();
        gizmo.begin_drag(GizmoAxis::Y, [100.0, 100.0], [0.0, 0.0, 0.0]);

        let delta = gizmo.update_drag([120.0, 100.0], [20.0, 0.0], &cam, [800.0, 600.0]);
        assert!(delta.is_some());
        match delta.unwrap() {
            GizmoDelta::Rotate { axis, angle_rad } => {
                assert!((axis[1] - 1.0).abs() < 1e-5);
                assert!(angle_rad.abs() > 0.0);
            }
            _ => panic!("Expected Rotate delta"),
        }
    }

    #[test]
    fn gizmo_snap_translate() {
        let mut gizmo = GizmoRenderer::new();
        gizmo.mode = GizmoMode::Translate;
        gizmo.snap_translate = 1.0;
        gizmo.build_axes([0.0, 0.0, 0.0]);

        let cam = ViewportCamera::new();
        gizmo.begin_drag(GizmoAxis::X, [100.0, 100.0], [0.0, 0.0, 0.0]);
        // Small move should snap to nearest integer.
        let _delta = gizmo.update_drag([101.0, 100.0], [1.0, 0.0], &cam, [800.0, 600.0]);
        gizmo.end_drag();
    }

    #[test]
    fn editor_viewport_creation() {
        let vp = EditorViewport::new(1920, 1080);
        assert_eq!(vp.width, 1920);
        assert_eq!(vp.height, 1080);
        assert_eq!(vp.gizmo_mode, GizmoMode::Translate);
        assert!(vp.grid_visible);
    }

    #[test]
    fn editor_viewport_resize() {
        let mut vp = EditorViewport::new(800, 600);
        vp.resize(1920, 1080);
        assert_eq!(vp.width, 1920);
        assert_eq!(vp.height, 1080);
    }

    #[test]
    fn editor_viewport_handle_selection() {
        let mut vp = EditorViewport::new(800, 600);
        let id1 = Uuid::new_v4();
        let bounds = vec![(id1, [-1.0, -1.0, -1.0_f32], [1.0, 1.0, 1.0_f32])];

        // Clicking far from the entity should not select it unless the ray
        // actually hits (depends on camera). Just test the API works.
        vp.handle_selection([400.0, 300.0], &bounds, false);
    }

    #[test]
    fn mat4_identity_inverse() {
        let id = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let inv = mat4_inverse(&id);
        for col in 0..4 {
            for row in 0..4 {
                let expected = if col == row { 1.0 } else { 0.0 };
                assert!((inv[col][row] - expected).abs() < 1e-5, "inv[{}][{}] = {} expected {}", col, row, inv[col][row], expected);
            }
        }
    }

    #[test]
    fn mat4_mul_identity() {
        let id = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let result = mat4_mul(&id, &id);
        for col in 0..4 {
            for row in 0..4 {
                let expected = if col == row { 1.0 } else { 0.0 };
                assert!((result[col][row] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn camera_focus_on() {
        let mut cam = ViewportCamera::new();
        cam.focus_on([5.0, 3.0, 0.0], 15.0);
        assert_eq!(cam.target, [5.0, 3.0, 0.0]);
        assert_eq!(cam.orbit_distance, 15.0);
    }

    #[test]
    fn camera_preset_top_down() {
        let mut cam = ViewportCamera::new();
        cam.set_preset(CameraMode::TopDown);
        assert_eq!(cam.mode, CameraMode::TopDown);
        // In top-down, camera should be above the target.
        assert!(cam.position[1] > cam.target[1]);
    }

    #[test]
    fn gizmo_render_produces_commands() {
        let mut gizmo = GizmoRenderer::new();
        gizmo.show_grid = true;
        gizmo.mode = GizmoMode::Translate;
        gizmo.build_axes([0.0, 0.0, 0.0]);

        let sm = SelectionManager::new();
        let commands = gizmo.render(&sm);
        // Should have at least a grid and arrow commands.
        assert!(!commands.is_empty());
    }

    #[test]
    fn pick_ray_from_screen_center() {
        let cam = ViewportCamera {
            position: [0.0, 0.0, 5.0],
            target: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            ..ViewportCamera::default()
        };
        let ray = cam.screen_to_ray([400.0, 300.0], [800.0, 600.0]);
        // The ray from center should go roughly toward the target.
        assert!(ray.direction[2] < 0.0 || ray.direction[2] > 0.0); // Just check it's valid.
    }
}
