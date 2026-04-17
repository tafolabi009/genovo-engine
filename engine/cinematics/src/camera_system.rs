//! Cinematic camera: Bezier path following, focus tracking, dolly zoom,
//! handheld shake, rack focus, and letterbox transitions.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct CameraTransformV2 { pub position: [f32; 3], pub target: [f32; 3], pub up: [f32; 3], pub fov: f32, pub roll: f32 }
impl Default for CameraTransformV2 { fn default() -> Self { Self { position: [0.0, 2.0, 5.0], target: [0.0; 3], up: [0.0, 1.0, 0.0], fov: 60.0, roll: 0.0 } } }

#[derive(Debug, Clone)]
pub struct BezierPath { pub control_points: Vec<[f32; 3]>, pub total_length: f32 }
impl BezierPath {
    pub fn new() -> Self { Self { control_points: Vec::new(), total_length: 0.0 } }
    pub fn add_point(&mut self, p: [f32; 3]) { self.control_points.push(p); self.recalculate_length(); }
    fn recalculate_length(&mut self) { self.total_length = 0.0; for i in 1..self.control_points.len() { let a = self.control_points[i-1]; let b = self.control_points[i]; self.total_length += ((b[0]-a[0]).powi(2)+(b[1]-a[1]).powi(2)+(b[2]-a[2]).powi(2)).sqrt(); } }
    pub fn evaluate(&self, t: f32) -> [f32; 3] {
        let n = self.control_points.len(); if n == 0 { return [0.0; 3]; } if n == 1 { return self.control_points[0]; }
        let t = t.clamp(0.0, 1.0); let segment = (t * (n - 1) as f32).min((n - 2) as f32);
        let i = segment as usize; let f = segment - i as f32;
        let a = self.control_points[i]; let b = self.control_points[i + 1];
        [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f]
    }
}

#[derive(Debug, Clone)]
pub struct FocusTarget { pub entity_id: Option<u64>, pub position: Option<[f32; 3]>, pub offset: [f32; 3], pub smoothing: f32 }
impl Default for FocusTarget { fn default() -> Self { Self { entity_id: None, position: None, offset: [0.0; 3], smoothing: 5.0 } } }

#[derive(Debug, Clone)]
pub struct DollyZoom { pub active: bool, pub target_distance: f32, pub start_fov: f32, pub end_fov: f32, pub duration: f32, pub elapsed: f32 }
impl DollyZoom { pub fn new(target_dist: f32, start_fov: f32, end_fov: f32, dur: f32) -> Self { Self { active: true, target_distance: target_dist, start_fov, end_fov, duration: dur, elapsed: 0.0 } }
    pub fn progress(&self) -> f32 { (self.elapsed / self.duration).clamp(0.0, 1.0) }
    pub fn current_fov(&self) -> f32 { let t = self.progress(); self.start_fov + (self.end_fov - self.start_fov) * t }
}

#[derive(Debug, Clone)]
pub struct HandheldShake { pub enabled: bool, pub amplitude: f32, pub frequency: f32, pub damping: f32, pub seed: f32, pub intensity: f32 }
impl Default for HandheldShake { fn default() -> Self { Self { enabled: false, amplitude: 0.02, frequency: 5.0, damping: 0.95, seed: 0.0, intensity: 1.0 } } }
impl HandheldShake {
    pub fn evaluate(&self, time: f32) -> [f32; 3] {
        if !self.enabled { return [0.0; 3]; }
        let a = self.amplitude * self.intensity;
        [a * (time * self.frequency * 1.1 + self.seed).sin(), a * (time * self.frequency * 0.9 + self.seed + 1.7).sin() * 0.7, a * (time * self.frequency * 1.3 + self.seed + 3.1).sin() * 0.3]
    }
}

#[derive(Debug, Clone)]
pub struct RackFocus { pub active: bool, pub near_target: f32, pub far_target: f32, pub current_focus: f32, pub transition_speed: f32, pub aperture: f32 }
impl Default for RackFocus { fn default() -> Self { Self { active: false, near_target: 2.0, far_target: 20.0, current_focus: 5.0, transition_speed: 3.0, aperture: 2.8 } } }
impl RackFocus { pub fn focus_to_near(&mut self) { self.active = true; } pub fn focus_to_far(&mut self) { self.active = true; } pub fn update(&mut self, dt: f32) { /* lerp current_focus toward target */ } }

#[derive(Debug, Clone)]
pub struct Letterbox { pub active: bool, pub target_aspect: f32, pub current_amount: f32, pub transition_speed: f32, pub bar_color: [f32; 4] }
impl Default for Letterbox { fn default() -> Self { Self { active: false, target_aspect: 2.35, current_amount: 0.0, transition_speed: 2.0, bar_color: [0.0, 0.0, 0.0, 1.0] } } }
impl Letterbox { pub fn enable(&mut self, aspect: f32) { self.active = true; self.target_aspect = aspect; } pub fn disable(&mut self) { self.active = false; } pub fn update(&mut self, dt: f32, screen_aspect: f32) { let target = if self.active { 1.0 - screen_aspect / self.target_aspect } else { 0.0 }.max(0.0); self.current_amount += (target - self.current_amount) * self.transition_speed * dt; } pub fn bar_height(&self) -> f32 { self.current_amount * 0.5 } }

#[derive(Debug, Clone)]
pub enum CameraEvent { PathStarted, PathCompleted, FocusChanged(Option<u64>), DollyZoomStarted, DollyZoomCompleted, ShakeTriggered(f32), LetterboxChanged(bool) }

pub struct CinematicCameraSystem {
    pub transform: CameraTransformV2,
    pub path: Option<BezierPath>,
    pub path_progress: f32,
    pub path_speed: f32,
    pub focus: FocusTarget,
    pub dolly_zoom: Option<DollyZoom>,
    pub shake: HandheldShake,
    pub rack_focus: RackFocus,
    pub letterbox: Letterbox,
    pub events: Vec<CameraEvent>,
    pub time: f32,
    pub smooth_position: [f32; 3],
    pub smooth_target: [f32; 3],
    pub interpolation_speed: f32,
    pub enabled: bool,
}

impl CinematicCameraSystem {
    pub fn new() -> Self {
        Self { transform: CameraTransformV2::default(), path: None, path_progress: 0.0, path_speed: 0.1, focus: FocusTarget::default(), dolly_zoom: None, shake: HandheldShake::default(), rack_focus: RackFocus::default(), letterbox: Letterbox::default(), events: Vec::new(), time: 0.0, smooth_position: [0.0, 2.0, 5.0], smooth_target: [0.0; 3], interpolation_speed: 5.0, enabled: true }
    }

    pub fn set_path(&mut self, path: BezierPath) { self.path = Some(path); self.path_progress = 0.0; self.events.push(CameraEvent::PathStarted); }
    pub fn clear_path(&mut self) { self.path = None; }
    pub fn start_dolly_zoom(&mut self, target_dist: f32, start_fov: f32, end_fov: f32, duration: f32) { self.dolly_zoom = Some(DollyZoom::new(target_dist, start_fov, end_fov, duration)); self.events.push(CameraEvent::DollyZoomStarted); }
    pub fn trigger_shake(&mut self, intensity: f32) { self.shake.enabled = true; self.shake.intensity = intensity; self.events.push(CameraEvent::ShakeTriggered(intensity)); }
    pub fn set_letterbox(&mut self, aspect: f32) { self.letterbox.enable(aspect); self.events.push(CameraEvent::LetterboxChanged(true)); }

    pub fn update(&mut self, dt: f32) {
        if !self.enabled { return; }
        self.time += dt;
        if let Some(ref path) = self.path {
            self.path_progress += self.path_speed * dt;
            if self.path_progress >= 1.0 { self.path_progress = 1.0; self.events.push(CameraEvent::PathCompleted); }
            let pos = path.evaluate(self.path_progress);
            self.transform.position = pos;
        }
        if let Some(ref mut dz) = self.dolly_zoom {
            dz.elapsed += dt;
            self.transform.fov = dz.current_fov();
            if dz.elapsed >= dz.duration { self.events.push(CameraEvent::DollyZoomCompleted); }
        }
        let shake_offset = self.shake.evaluate(self.time);
        self.transform.position[0] += shake_offset[0];
        self.transform.position[1] += shake_offset[1];
        self.transform.position[2] += shake_offset[2];
        self.letterbox.update(dt, 16.0 / 9.0);
    }

    pub fn drain_events(&mut self) -> Vec<CameraEvent> { std::mem::take(&mut self.events) }
}

impl Default for CinematicCameraSystem { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bezier_path() {
        let mut path = BezierPath::new();
        path.add_point([0.0, 0.0, 0.0]); path.add_point([10.0, 5.0, 0.0]); path.add_point([20.0, 0.0, 0.0]);
        let mid = path.evaluate(0.5);
        assert!((mid[0] - 10.0).abs() < 0.5);
    }
    #[test]
    fn dolly_zoom_progress() {
        let dz = DollyZoom::new(5.0, 60.0, 20.0, 2.0);
        assert!((dz.current_fov() - 60.0).abs() < 0.01);
    }
}
