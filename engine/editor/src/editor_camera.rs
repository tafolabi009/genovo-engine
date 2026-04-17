// engine/editor/src/editor_camera.rs
//
// Editor camera controller for the Genovo editor.
// Orbit/fly/focus modes, smooth transitions between modes,
// frame selected, camera bookmarks, multiple viewports
// (perspective/top/front/right), camera speed adjustment.

use std::collections::HashMap;

pub const DEFAULT_ORBIT_DISTANCE: f32 = 10.0;
pub const DEFAULT_FLY_SPEED: f32 = 5.0;
pub const DEFAULT_ORBIT_SPEED: f32 = 0.005;
pub const DEFAULT_ZOOM_SPEED: f32 = 1.0;
pub const DEFAULT_PAN_SPEED: f32 = 0.01;
pub const MIN_ORBIT_DISTANCE: f32 = 0.1;
pub const MAX_ORBIT_DISTANCE: f32 = 10000.0;
pub const SMOOTH_FACTOR: f32 = 8.0;
pub const MAX_BOOKMARKS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraControlMode { Orbit, Fly, Focus, Pan, Zoom, FreeLook }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportType { Perspective, Top, Front, Right, Bottom, Back, Left }

impl ViewportType {
    pub fn is_orthographic(&self) -> bool { !matches!(self, Self::Perspective) }
    pub fn view_direction(&self) -> [f32; 3] {
        match self {
            Self::Perspective => [0.0, 0.0, -1.0], Self::Top => [0.0, -1.0, 0.0],
            Self::Front => [0.0, 0.0, -1.0], Self::Right => [-1.0, 0.0, 0.0],
            Self::Bottom => [0.0, 1.0, 0.0], Self::Back => [0.0, 0.0, 1.0],
            Self::Left => [1.0, 0.0, 0.0],
        }
    }
    pub fn up_direction(&self) -> [f32; 3] {
        match self {
            Self::Top | Self::Bottom => [0.0, 0.0, -1.0],
            _ => [0.0, 1.0, 0.0],
        }
    }
}

#[derive(Debug, Clone)]
pub struct EditorCameraState {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub orbit_distance: f32,
    pub fov_degrees: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub ortho_size: f32,
}

impl Default for EditorCameraState {
    fn default() -> Self {
        Self {
            position: [0.0, 5.0, 10.0], target: [0.0, 0.0, 0.0], up: [0.0, 1.0, 0.0],
            yaw: 0.0, pitch: -0.3, orbit_distance: DEFAULT_ORBIT_DISTANCE,
            fov_degrees: 60.0, near_plane: 0.1, far_plane: 10000.0, ortho_size: 10.0,
        }
    }
}

impl EditorCameraState {
    pub fn forward(&self) -> [f32; 3] {
        let dx = self.target[0] - self.position[0];
        let dy = self.target[1] - self.position[1];
        let dz = self.target[2] - self.position[2];
        let len = (dx*dx + dy*dy + dz*dz).sqrt().max(1e-6);
        [dx/len, dy/len, dz/len]
    }
    pub fn right(&self) -> [f32; 3] {
        let fwd = self.forward();
        let rx = self.up[1] * fwd[2] - self.up[2] * fwd[1];
        let ry = self.up[2] * fwd[0] - self.up[0] * fwd[2];
        let rz = self.up[0] * fwd[1] - self.up[1] * fwd[0];
        let len = (rx*rx + ry*ry + rz*rz).sqrt().max(1e-6);
        [rx/len, ry/len, rz/len]
    }
}

#[derive(Debug, Clone)]
pub struct CameraBookmark {
    pub name: String,
    pub state: EditorCameraState,
    pub viewport_type: ViewportType,
    pub timestamp: f64,
}

impl CameraBookmark {
    pub fn new(name: &str, state: EditorCameraState, viewport: ViewportType) -> Self {
        Self { name: name.to_string(), state, viewport_type: viewport, timestamp: 0.0 }
    }
}

#[derive(Debug, Clone)]
pub struct CameraSpeedSettings {
    pub base_speed: f32,
    pub speed_multiplier: f32,
    pub slow_multiplier: f32,
    pub fast_multiplier: f32,
    pub scroll_speed: f32,
    pub pan_speed: f32,
    pub orbit_speed: f32,
    pub smoothing: f32,
    pub inertia: f32,
    pub inertia_decay: f32,
}

impl Default for CameraSpeedSettings {
    fn default() -> Self {
        Self {
            base_speed: DEFAULT_FLY_SPEED, speed_multiplier: 1.0,
            slow_multiplier: 0.25, fast_multiplier: 3.0,
            scroll_speed: DEFAULT_ZOOM_SPEED, pan_speed: DEFAULT_PAN_SPEED,
            orbit_speed: DEFAULT_ORBIT_SPEED, smoothing: SMOOTH_FACTOR,
            inertia: 0.0, inertia_decay: 5.0,
        }
    }
}

impl CameraSpeedSettings {
    pub fn effective_speed(&self, shift: bool, ctrl: bool) -> f32 {
        let mut speed = self.base_speed * self.speed_multiplier;
        if shift { speed *= self.fast_multiplier; }
        if ctrl { speed *= self.slow_multiplier; }
        speed
    }
    pub fn increase_speed(&mut self) { self.speed_multiplier = (self.speed_multiplier * 1.5).min(100.0); }
    pub fn decrease_speed(&mut self) { self.speed_multiplier = (self.speed_multiplier / 1.5).max(0.01); }
    pub fn reset_speed(&mut self) { self.speed_multiplier = 1.0; }
}

#[derive(Debug)]
pub struct EditorCameraController {
    pub state: EditorCameraState,
    pub target_state: EditorCameraState,
    pub mode: CameraControlMode,
    pub viewport_type: ViewportType,
    pub speed: CameraSpeedSettings,
    pub bookmarks: Vec<CameraBookmark>,
    pub transitioning: bool,
    pub transition_progress: f32,
    pub transition_speed: f32,
    pub transition_start: EditorCameraState,
    pub transition_end: EditorCameraState,
    pub velocity: [f32; 3],
    pub lock_pitch: bool,
    pub lock_yaw: bool,
    pub grid_visible: bool,
    pub active: bool,
}

impl EditorCameraController {
    pub fn new() -> Self {
        let state = EditorCameraState::default();
        Self {
            target_state: state.clone(), state,
            mode: CameraControlMode::Orbit, viewport_type: ViewportType::Perspective,
            speed: CameraSpeedSettings::default(),
            bookmarks: Vec::new(), transitioning: false, transition_progress: 0.0,
            transition_speed: 2.0, transition_start: EditorCameraState::default(),
            transition_end: EditorCameraState::default(),
            velocity: [0.0; 3], lock_pitch: false, lock_yaw: false,
            grid_visible: true, active: true,
        }
    }

    pub fn set_mode(&mut self, mode: CameraControlMode) { self.mode = mode; }

    pub fn set_viewport_type(&mut self, vt: ViewportType) {
        self.viewport_type = vt;
        if vt.is_orthographic() { self.mode = CameraControlMode::Orbit; }
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        if self.lock_yaw && self.lock_pitch { return; }
        if !self.lock_yaw { self.target_state.yaw += dx * self.speed.orbit_speed; }
        if !self.lock_pitch { self.target_state.pitch = (self.target_state.pitch + dy * self.speed.orbit_speed).clamp(-1.5, 1.5); }
        self.update_orbit_position();
    }

    pub fn zoom(&mut self, delta: f32) {
        if self.viewport_type.is_orthographic() {
            self.target_state.ortho_size = (self.target_state.ortho_size - delta * self.speed.scroll_speed).clamp(0.1, 1000.0);
        } else {
            self.target_state.orbit_distance = (self.target_state.orbit_distance - delta * self.speed.scroll_speed).clamp(MIN_ORBIT_DISTANCE, MAX_ORBIT_DISTANCE);
            self.update_orbit_position();
        }
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let right = self.state.right();
        let up = self.state.up;
        let scale = self.speed.pan_speed * self.state.orbit_distance;
        for i in 0..3 {
            self.target_state.target[i] += -right[i] * dx * scale + up[i] * dy * scale;
        }
        self.update_orbit_position();
    }

    pub fn fly_move(&mut self, forward: f32, right: f32, up: f32, dt: f32, shift: bool, ctrl: bool) {
        let speed = self.speed.effective_speed(shift, ctrl);
        let fwd = self.state.forward();
        let rt = self.state.right();
        for i in 0..3 {
            self.target_state.position[i] += (fwd[i] * forward + rt[i] * right + self.state.up[i] * up) * speed * dt;
        }
        self.target_state.target = [
            self.target_state.position[0] + fwd[0],
            self.target_state.position[1] + fwd[1],
            self.target_state.position[2] + fwd[2],
        ];
    }

    pub fn focus_on(&mut self, target: [f32; 3], radius: f32) {
        self.transition_start = self.state.clone();
        self.transition_end = self.state.clone();
        self.transition_end.target = target;
        self.transition_end.orbit_distance = radius * 2.5;
        self.transitioning = true;
        self.transition_progress = 0.0;
    }

    pub fn frame_selection(&mut self, center: [f32; 3], radius: f32) {
        self.focus_on(center, radius.max(1.0));
    }

    fn update_orbit_position(&mut self) {
        let dist = self.target_state.orbit_distance;
        let yaw = self.target_state.yaw;
        let pitch = self.target_state.pitch;
        self.target_state.position[0] = self.target_state.target[0] + dist * pitch.cos() * yaw.sin();
        self.target_state.position[1] = self.target_state.target[1] + dist * pitch.sin();
        self.target_state.position[2] = self.target_state.target[2] + dist * pitch.cos() * yaw.cos();
    }

    pub fn update(&mut self, dt: f32) {
        if !self.active { return; }
        if self.transitioning {
            self.transition_progress += dt * self.transition_speed;
            if self.transition_progress >= 1.0 {
                self.transition_progress = 1.0;
                self.transitioning = false;
            }
            let t = smooth_step(self.transition_progress);
            lerp_camera_state(&self.transition_start, &self.transition_end, t, &mut self.state);
            self.target_state = self.state.clone();
        } else {
            let factor = (self.speed.smoothing * dt).min(1.0);
            let current = self.state.clone();
            lerp_camera_state(&current, &self.target_state, factor, &mut self.state);
        }
    }

    pub fn add_bookmark(&mut self, name: &str) {
        if self.bookmarks.len() < MAX_BOOKMARKS {
            self.bookmarks.push(CameraBookmark::new(name, self.state.clone(), self.viewport_type));
        }
    }

    pub fn goto_bookmark(&mut self, index: usize) {
        if let Some(bookmark) = self.bookmarks.get(index) {
            self.transition_start = self.state.clone();
            self.transition_end = bookmark.state.clone();
            self.transitioning = true;
            self.transition_progress = 0.0;
            self.viewport_type = bookmark.viewport_type;
        }
    }

    pub fn remove_bookmark(&mut self, index: usize) {
        if index < self.bookmarks.len() { self.bookmarks.remove(index); }
    }

    pub fn bookmark_count(&self) -> usize { self.bookmarks.len() }
}

impl Default for EditorCameraController { fn default() -> Self { Self::new() } }

fn smooth_step(t: f32) -> f32 { let t = t.clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) }

fn lerp_camera_state(a: &EditorCameraState, b: &EditorCameraState, t: f32, out: &mut EditorCameraState) {
    for i in 0..3 {
        out.position[i] = a.position[i] + (b.position[i] - a.position[i]) * t;
        out.target[i] = a.target[i] + (b.target[i] - a.target[i]) * t;
    }
    out.yaw = a.yaw + (b.yaw - a.yaw) * t;
    out.pitch = a.pitch + (b.pitch - a.pitch) * t;
    out.orbit_distance = a.orbit_distance + (b.orbit_distance - a.orbit_distance) * t;
    out.fov_degrees = a.fov_degrees + (b.fov_degrees - a.fov_degrees) * t;
    out.ortho_size = a.ortho_size + (b.ortho_size - a.ortho_size) * t;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_camera_modes() {
        let mut cam = EditorCameraController::new();
        cam.set_mode(CameraControlMode::Fly);
        assert_eq!(cam.mode, CameraControlMode::Fly);
        cam.set_viewport_type(ViewportType::Top);
        assert!(cam.viewport_type.is_orthographic());
    }
    #[test]
    fn test_camera_orbit() {
        let mut cam = EditorCameraController::new();
        cam.orbit(0.1, 0.1);
        cam.update(0.016);
    }
    #[test]
    fn test_bookmarks() {
        let mut cam = EditorCameraController::new();
        cam.add_bookmark("Position A");
        cam.add_bookmark("Position B");
        assert_eq!(cam.bookmark_count(), 2);
        cam.goto_bookmark(0);
        assert!(cam.transitioning);
    }
    #[test]
    fn test_speed_settings() {
        let mut speed = CameraSpeedSettings::default();
        let base = speed.effective_speed(false, false);
        speed.increase_speed();
        assert!(speed.effective_speed(false, false) > base);
        speed.reset_speed();
        assert!((speed.effective_speed(false, false) - base).abs() < 0.01);
    }
}
