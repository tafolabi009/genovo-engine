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

    #[test]
    fn handheld_shake_disabled() {
        let shake = HandheldShake::default();
        let offset = shake.evaluate(1.0);
        assert_eq!(offset, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn handheld_shake_enabled() {
        let mut shake = HandheldShake::default();
        shake.enabled = true;
        let offset = shake.evaluate(1.0);
        // Should produce non-zero offsets.
        assert!(offset[0].abs() > 0.0 || offset[1].abs() > 0.0);
    }

    #[test]
    fn letterbox_update() {
        let mut lb = Letterbox::default();
        lb.enable(2.35);
        lb.update(0.5, 16.0 / 9.0);
        assert!(lb.current_amount > 0.0);
    }

    #[test]
    fn cinematic_camera_path_update() {
        let mut cam = CinematicCameraSystem::new();
        let mut path = BezierPath::new();
        path.add_point([0.0, 0.0, 0.0]);
        path.add_point([10.0, 0.0, 0.0]);
        path.add_point([20.0, 0.0, 0.0]);
        cam.path_speed = 1.0;
        cam.set_path(path);
        cam.update(0.5);
        assert!(cam.path_progress > 0.0);
    }

    #[test]
    fn cinematic_camera_dolly_zoom() {
        let mut cam = CinematicCameraSystem::new();
        cam.start_dolly_zoom(5.0, 60.0, 20.0, 1.0);
        cam.update(0.5);
        let fov = cam.transform.fov;
        assert!(fov < 60.0 && fov > 20.0);
    }
}

// ---------------------------------------------------------------------------
// Camera rail system
// ---------------------------------------------------------------------------

/// A rail for camera movement along a predefined path.
#[derive(Debug, Clone)]
pub struct CameraRail {
    /// Rail name for identification.
    pub name: String,
    /// Control points defining the rail path.
    pub points: Vec<RailPoint>,
    /// Whether the rail loops back to the start.
    pub looping: bool,
    /// Speed curve along the rail (maps rail parameter to speed multiplier).
    pub speed_curve: Vec<(f32, f32)>,
    /// Total rail length in world units.
    pub total_length: f32,
}

/// A point on a camera rail.
#[derive(Debug, Clone, Copy)]
pub struct RailPoint {
    /// World position of this rail point.
    pub position: [f32; 3],
    /// Look-at target from this point.
    pub look_at: [f32; 3],
    /// Field of view at this point.
    pub fov: f32,
    /// Roll angle at this point.
    pub roll: f32,
    /// Parameter along the rail (0..1).
    pub parameter: f32,
}

impl RailPoint {
    /// Create a new rail point.
    pub fn new(position: [f32; 3], look_at: [f32; 3]) -> Self {
        Self {
            position,
            look_at,
            fov: 60.0,
            roll: 0.0,
            parameter: 0.0,
        }
    }

    /// Interpolate between two rail points.
    pub fn lerp(&self, other: &RailPoint, t: f32) -> RailPoint {
        RailPoint {
            position: [
                self.position[0] + (other.position[0] - self.position[0]) * t,
                self.position[1] + (other.position[1] - self.position[1]) * t,
                self.position[2] + (other.position[2] - self.position[2]) * t,
            ],
            look_at: [
                self.look_at[0] + (other.look_at[0] - self.look_at[0]) * t,
                self.look_at[1] + (other.look_at[1] - self.look_at[1]) * t,
                self.look_at[2] + (other.look_at[2] - self.look_at[2]) * t,
            ],
            fov: self.fov + (other.fov - self.fov) * t,
            roll: self.roll + (other.roll - self.roll) * t,
            parameter: self.parameter + (other.parameter - self.parameter) * t,
        }
    }
}

impl CameraRail {
    /// Create a new empty rail.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            points: Vec::new(),
            looping: false,
            speed_curve: vec![(0.0, 1.0), (1.0, 1.0)],
            total_length: 0.0,
        }
    }

    /// Add a point to the rail.
    pub fn add_point(&mut self, point: RailPoint) {
        if let Some(last) = self.points.last() {
            let dx = point.position[0] - last.position[0];
            let dy = point.position[1] - last.position[1];
            let dz = point.position[2] - last.position[2];
            self.total_length += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        self.points.push(point);
        self.recalculate_parameters();
    }

    /// Recalculate the parameter values along the rail.
    fn recalculate_parameters(&mut self) {
        if self.points.len() < 2 || self.total_length <= 0.0 {
            if let Some(p) = self.points.first_mut() {
                p.parameter = 0.0;
            }
            return;
        }

        self.points[0].parameter = 0.0;
        let mut accumulated = 0.0f32;

        for i in 1..self.points.len() {
            let dx = self.points[i].position[0] - self.points[i - 1].position[0];
            let dy = self.points[i].position[1] - self.points[i - 1].position[1];
            let dz = self.points[i].position[2] - self.points[i - 1].position[2];
            accumulated += (dx * dx + dy * dy + dz * dz).sqrt();
            self.points[i].parameter = accumulated / self.total_length;
        }
    }

    /// Evaluate the rail at parameter t (0..1).
    pub fn evaluate(&self, t: f32) -> Option<RailPoint> {
        if self.points.is_empty() {
            return None;
        }
        if self.points.len() == 1 {
            return Some(self.points[0]);
        }

        let t = if self.looping {
            t.rem_euclid(1.0)
        } else {
            t.clamp(0.0, 1.0)
        };

        // Find the two surrounding points.
        for i in 0..self.points.len() - 1 {
            let a = &self.points[i];
            let b = &self.points[i + 1];
            if t >= a.parameter && t <= b.parameter {
                let segment_t = if (b.parameter - a.parameter).abs() < 1e-6 {
                    0.0
                } else {
                    (t - a.parameter) / (b.parameter - a.parameter)
                };
                return Some(a.lerp(b, segment_t));
            }
        }

        Some(*self.points.last().unwrap())
    }

    /// Get the speed multiplier at parameter t.
    pub fn speed_at(&self, t: f32) -> f32 {
        if self.speed_curve.is_empty() {
            return 1.0;
        }
        if self.speed_curve.len() == 1 {
            return self.speed_curve[0].1;
        }

        let t = t.clamp(0.0, 1.0);
        for i in 0..self.speed_curve.len() - 1 {
            let (ta, sa) = self.speed_curve[i];
            let (tb, sb) = self.speed_curve[i + 1];
            if t >= ta && t <= tb {
                let f = (t - ta) / (tb - ta).max(0.001);
                return sa + (sb - sa) * f;
            }
        }

        self.speed_curve.last().unwrap().1
    }

    /// Get the number of points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }
}

// ---------------------------------------------------------------------------
// Camera blend (for transitioning between cameras)
// ---------------------------------------------------------------------------

/// Blend state for transitioning between two camera transforms.
#[derive(Debug, Clone)]
pub struct CameraBlend {
    /// Source camera transform.
    pub source: CameraTransformV2,
    /// Target camera transform.
    pub target: CameraTransformV2,
    /// Current blend factor (0 = source, 1 = target).
    pub blend_factor: f32,
    /// Blend duration in seconds.
    pub duration: f32,
    /// Elapsed time.
    pub elapsed: f32,
    /// Blend curve type.
    pub curve: BlendCurve,
    /// Whether the blend is active.
    pub active: bool,
}

/// Blend curve shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendCurve {
    /// Linear interpolation.
    Linear,
    /// Smooth start and end (ease in/out).
    EaseInOut,
    /// Quick start, slow end.
    EaseOut,
    /// Slow start, quick end.
    EaseIn,
    /// Instant snap (no interpolation).
    Snap,
}

impl CameraBlend {
    /// Create a new blend.
    pub fn new(
        source: CameraTransformV2,
        target: CameraTransformV2,
        duration: f32,
        curve: BlendCurve,
    ) -> Self {
        Self {
            source,
            target,
            blend_factor: 0.0,
            duration,
            elapsed: 0.0,
            curve,
            active: true,
        }
    }

    /// Update the blend, advancing time.
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }
        self.elapsed += dt;
        let raw_t = (self.elapsed / self.duration.max(0.001)).clamp(0.0, 1.0);

        self.blend_factor = match self.curve {
            BlendCurve::Linear => raw_t,
            BlendCurve::EaseInOut => {
                let t = raw_t;
                t * t * (3.0 - 2.0 * t)
            }
            BlendCurve::EaseOut => {
                1.0 - (1.0 - raw_t) * (1.0 - raw_t)
            }
            BlendCurve::EaseIn => {
                raw_t * raw_t
            }
            BlendCurve::Snap => {
                if raw_t >= 1.0 { 1.0 } else { 0.0 }
            }
        };

        if raw_t >= 1.0 {
            self.active = false;
        }
    }

    /// Get the blended camera transform.
    pub fn current_transform(&self) -> CameraTransformV2 {
        let t = self.blend_factor;
        CameraTransformV2 {
            position: [
                self.source.position[0] + (self.target.position[0] - self.source.position[0]) * t,
                self.source.position[1] + (self.target.position[1] - self.source.position[1]) * t,
                self.source.position[2] + (self.target.position[2] - self.source.position[2]) * t,
            ],
            target: [
                self.source.target[0] + (self.target.target[0] - self.source.target[0]) * t,
                self.source.target[1] + (self.target.target[1] - self.source.target[1]) * t,
                self.source.target[2] + (self.target.target[2] - self.source.target[2]) * t,
            ],
            up: self.source.up, // Keep up vector stable during blend.
            fov: self.source.fov + (self.target.fov - self.source.fov) * t,
            roll: self.source.roll + (self.target.roll - self.source.roll) * t,
        }
    }

    /// Check if the blend is complete.
    pub fn is_complete(&self) -> bool {
        !self.active
    }

    /// Get progress (0..1).
    pub fn progress(&self) -> f32 {
        (self.elapsed / self.duration.max(0.001)).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Camera shake presets
// ---------------------------------------------------------------------------

/// Pre-built shake presets for common effects.
pub struct ShakePresets;

impl ShakePresets {
    /// Explosion shake - strong, short, high frequency.
    pub fn explosion() -> HandheldShake {
        HandheldShake {
            enabled: true,
            amplitude: 0.15,
            frequency: 15.0,
            damping: 0.9,
            seed: 0.0,
            intensity: 1.0,
        }
    }

    /// Footstep shake - subtle, rhythmic.
    pub fn footstep() -> HandheldShake {
        HandheldShake {
            enabled: true,
            amplitude: 0.005,
            frequency: 2.0,
            damping: 0.98,
            seed: 0.0,
            intensity: 1.0,
        }
    }

    /// Earthquake shake - low frequency, large amplitude.
    pub fn earthquake() -> HandheldShake {
        HandheldShake {
            enabled: true,
            amplitude: 0.1,
            frequency: 3.0,
            damping: 0.99,
            seed: 0.0,
            intensity: 1.0,
        }
    }

    /// Vehicle rumble - continuous, moderate.
    pub fn vehicle_rumble() -> HandheldShake {
        HandheldShake {
            enabled: true,
            amplitude: 0.02,
            frequency: 8.0,
            damping: 1.0,
            seed: 0.0,
            intensity: 1.0,
        }
    }

    /// Gunfire recoil - sharp impulse.
    pub fn gunfire() -> HandheldShake {
        HandheldShake {
            enabled: true,
            amplitude: 0.04,
            frequency: 20.0,
            damping: 0.85,
            seed: 0.0,
            intensity: 1.0,
        }
    }
}
