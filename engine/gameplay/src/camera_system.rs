// engine/gameplay/src/camera_system_v2.rs
//
// Enhanced camera system for the Genovo engine.
//
// Provides a flexible camera framework with multiple modes and effects:
//
// - Multiple camera modes: orbit, follow, first-person, top-down, free, rail.
// - Smooth transitions between camera modes.
// - Camera effects stack: shake, zoom pulse, focus blur, flash.
// - Cinematic camera with keyframed paths.
// - Split-screen support with configurable layouts.
// - Camera collision avoidance (push camera forward when blocked).
// - Input-driven camera control with deadzone and sensitivity.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default field of view in degrees.
const DEFAULT_FOV: f32 = 60.0;

/// Minimum field of view.
const MIN_FOV: f32 = 10.0;

/// Maximum field of view.
const MAX_FOV: f32 = 120.0;

/// Default near clip plane.
const DEFAULT_NEAR: f32 = 0.1;

/// Default far clip plane.
const DEFAULT_FAR: f32 = 1000.0;

/// Default camera transition duration.
const DEFAULT_TRANSITION_DURATION: f32 = 0.5;

/// Maximum camera effects on the stack.
const MAX_EFFECTS: usize = 16;

/// Smoothing factor for camera movement.
const DEFAULT_SMOOTHING: f32 = 8.0;

// ---------------------------------------------------------------------------
// Camera Transform
// ---------------------------------------------------------------------------

/// Camera transform in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct CameraTransformV2 {
    /// World position.
    pub position: [f32; 3],
    /// Rotation as Euler angles (pitch, yaw, roll) in radians.
    pub rotation: [f32; 3],
    /// Forward direction (computed from rotation).
    pub forward: [f32; 3],
    /// Right direction (computed from rotation).
    pub right: [f32; 3],
    /// Up direction (computed from rotation).
    pub up: [f32; 3],
}

impl Default for CameraTransformV2 {
    fn default() -> Self {
        Self {
            position: [0.0, 5.0, -10.0],
            rotation: [0.0, 0.0, 0.0],
            forward: [0.0, 0.0, 1.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
        }
    }
}

impl CameraTransformV2 {
    /// Create a camera transform from position and look-at target.
    pub fn look_at(position: [f32; 3], target: [f32; 3], world_up: [f32; 3]) -> Self {
        let forward = [
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
        ];
        let len = (forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]).sqrt();
        let forward = if len > 1e-6 {
            [forward[0] / len, forward[1] / len, forward[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        };

        let right = [
            world_up[1] * forward[2] - world_up[2] * forward[1],
            world_up[2] * forward[0] - world_up[0] * forward[2],
            world_up[0] * forward[1] - world_up[1] * forward[0],
        ];
        let r_len = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
        let right = if r_len > 1e-6 {
            [right[0] / r_len, right[1] / r_len, right[2] / r_len]
        } else {
            [1.0, 0.0, 0.0]
        };

        let up = [
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        ];

        let pitch = (-forward[1]).asin();
        let yaw = forward[0].atan2(forward[2]);

        Self {
            position,
            rotation: [pitch, yaw, 0.0],
            forward,
            right,
            up,
        }
    }

    /// Update directions from rotation.
    pub fn update_directions(&mut self) {
        let (sp, cp) = self.rotation[0].sin_cos();
        let (sy, cy) = self.rotation[1].sin_cos();
        let (sr, cr) = self.rotation[2].sin_cos();

        self.forward = [cp * sy, -sp, cp * cy];
        self.right = [cr * cy + sr * sp * sy, sr * cp, -cr * sy + sr * sp * cy];
        self.up = [
            -sr * cy + cr * sp * sy,
            cr * cp,
            sr * sy + cr * sp * cy,
        ];
    }

    /// Linearly interpolate between two camera transforms.
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let inv_t = 1.0 - t;

        let mut result = Self {
            position: [
                a.position[0] * inv_t + b.position[0] * t,
                a.position[1] * inv_t + b.position[1] * t,
                a.position[2] * inv_t + b.position[2] * t,
            ],
            rotation: [
                a.rotation[0] * inv_t + b.rotation[0] * t,
                a.rotation[1] * inv_t + b.rotation[1] * t,
                a.rotation[2] * inv_t + b.rotation[2] * t,
            ],
            forward: [0.0; 3],
            right: [0.0; 3],
            up: [0.0; 3],
        };
        result.update_directions();
        result
    }
}

// ---------------------------------------------------------------------------
// Camera Mode
// ---------------------------------------------------------------------------

/// Available camera modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CameraMode {
    /// Orbit around a target point.
    Orbit,
    /// Follow behind a target.
    Follow,
    /// First-person view.
    FirstPerson,
    /// Top-down view.
    TopDown,
    /// Free-fly camera (editor-style).
    Free,
    /// Rail camera (follows a spline path).
    Rail,
    /// Cinematic camera (keyframed animation).
    Cinematic,
    /// Fixed camera (static position).
    Fixed,
}

// ---------------------------------------------------------------------------
// Camera Mode Configs
// ---------------------------------------------------------------------------

/// Configuration for orbit camera mode.
#[derive(Debug, Clone)]
pub struct OrbitConfig {
    /// Distance from target.
    pub distance: f32,
    /// Minimum distance.
    pub min_distance: f32,
    /// Maximum distance.
    pub max_distance: f32,
    /// Minimum pitch (looking up limit).
    pub min_pitch: f32,
    /// Maximum pitch (looking down limit).
    pub max_pitch: f32,
    /// Rotation speed (degrees per pixel of mouse movement).
    pub rotation_speed: f32,
    /// Zoom speed.
    pub zoom_speed: f32,
    /// Smoothing factor.
    pub smoothing: f32,
    /// Target offset (relative to target position).
    pub target_offset: [f32; 3],
    /// Auto-rotate speed (degrees per second, 0 = disabled).
    pub auto_rotate_speed: f32,
}

impl Default for OrbitConfig {
    fn default() -> Self {
        Self {
            distance: 10.0,
            min_distance: 2.0,
            max_distance: 50.0,
            min_pitch: -80.0f32.to_radians(),
            max_pitch: 80.0f32.to_radians(),
            rotation_speed: 0.3,
            zoom_speed: 2.0,
            smoothing: DEFAULT_SMOOTHING,
            target_offset: [0.0, 1.5, 0.0],
            auto_rotate_speed: 0.0,
        }
    }
}

/// Configuration for follow camera mode.
#[derive(Debug, Clone)]
pub struct FollowConfig {
    /// Ideal offset behind the target (local space).
    pub offset: [f32; 3],
    /// Look-at offset on the target.
    pub look_at_offset: [f32; 3],
    /// Position smoothing.
    pub position_smoothing: f32,
    /// Rotation smoothing.
    pub rotation_smoothing: f32,
    /// Whether to match the target's Y rotation.
    pub follow_rotation: bool,
    /// Collision avoidance radius.
    pub collision_radius: f32,
    /// Minimum distance from the target.
    pub min_distance: f32,
}

impl Default for FollowConfig {
    fn default() -> Self {
        Self {
            offset: [0.0, 3.0, -6.0],
            look_at_offset: [0.0, 1.5, 0.0],
            position_smoothing: 5.0,
            rotation_smoothing: 8.0,
            follow_rotation: true,
            collision_radius: 0.3,
            min_distance: 1.0,
        }
    }
}

/// Configuration for first-person camera.
#[derive(Debug, Clone)]
pub struct FirstPersonConfigV2 {
    /// Eye offset from character position.
    pub eye_offset: [f32; 3],
    /// Mouse sensitivity.
    pub sensitivity: f32,
    /// Pitch limits.
    pub min_pitch: f32,
    /// Maximum pitch.
    pub max_pitch: f32,
    /// Head bobbing amplitude.
    pub bob_amplitude: f32,
    /// Head bobbing frequency.
    pub bob_frequency: f32,
    /// Whether to enable head bobbing.
    pub enable_bob: bool,
}

impl Default for FirstPersonConfigV2 {
    fn default() -> Self {
        Self {
            eye_offset: [0.0, 1.7, 0.0],
            sensitivity: 0.2,
            min_pitch: -85.0f32.to_radians(),
            max_pitch: 85.0f32.to_radians(),
            bob_amplitude: 0.02,
            bob_frequency: 8.0,
            enable_bob: true,
        }
    }
}

/// Configuration for top-down camera.
#[derive(Debug, Clone)]
pub struct TopDownConfigV2 {
    /// Height above the target.
    pub height: f32,
    /// Minimum height.
    pub min_height: f32,
    /// Maximum height.
    pub max_height: f32,
    /// Camera angle from vertical (0 = straight down).
    pub angle: f32,
    /// Zoom speed.
    pub zoom_speed: f32,
    /// Pan speed.
    pub pan_speed: f32,
    /// Whether to follow the player.
    pub follow_player: bool,
    /// Edge scrolling margin (pixels).
    pub edge_scroll_margin: f32,
    /// Edge scrolling speed.
    pub edge_scroll_speed: f32,
}

impl Default for TopDownConfigV2 {
    fn default() -> Self {
        Self {
            height: 20.0,
            min_height: 5.0,
            max_height: 50.0,
            angle: 60.0f32.to_radians(),
            zoom_speed: 5.0,
            pan_speed: 20.0,
            follow_player: true,
            edge_scroll_margin: 20.0,
            edge_scroll_speed: 15.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera Effects
// ---------------------------------------------------------------------------

/// A camera effect that modifies the camera output.
#[derive(Debug, Clone)]
pub struct CameraEffect {
    /// Effect type.
    pub effect_type: CameraEffectType,
    /// Elapsed time.
    pub elapsed: f32,
    /// Total duration (0 = infinite).
    pub duration: f32,
    /// Effect intensity (0.0 to 1.0).
    pub intensity: f32,
    /// Whether the effect is active.
    pub active: bool,
    /// Priority (higher = applied later, overrides lower).
    pub priority: i32,
}

/// Types of camera effects.
#[derive(Debug, Clone)]
pub enum CameraEffectType {
    /// Camera shake with configurable amplitude and frequency.
    Shake {
        amplitude: f32,
        frequency: f32,
        decay: f32,
    },
    /// Zoom pulse (sudden FOV change and return).
    ZoomPulse {
        fov_delta: f32,
        speed: f32,
    },
    /// Screen flash (white/red/etc.).
    Flash {
        color: [f32; 4],
        fade_speed: f32,
    },
    /// Focus pull (depth of field change).
    FocusPull {
        target_focus_distance: f32,
        aperture: f32,
    },
    /// Slow motion effect on camera movement.
    SlowMotion {
        time_scale: f32,
    },
    /// Chromatic aberration.
    ChromaticAberration {
        strength: f32,
    },
    /// Vignette.
    Vignette {
        inner_radius: f32,
        outer_radius: f32,
        color: [f32; 4],
    },
}

impl CameraEffect {
    /// Create a shake effect.
    pub fn shake(amplitude: f32, frequency: f32, duration: f32) -> Self {
        Self {
            effect_type: CameraEffectType::Shake {
                amplitude,
                frequency,
                decay: 2.0,
            },
            elapsed: 0.0,
            duration,
            intensity: 1.0,
            active: true,
            priority: 0,
        }
    }

    /// Create a zoom pulse effect.
    pub fn zoom_pulse(fov_delta: f32, duration: f32) -> Self {
        Self {
            effect_type: CameraEffectType::ZoomPulse {
                fov_delta,
                speed: PI / duration,
            },
            elapsed: 0.0,
            duration,
            intensity: 1.0,
            active: true,
            priority: 0,
        }
    }

    /// Create a flash effect.
    pub fn flash(color: [f32; 4], duration: f32) -> Self {
        Self {
            effect_type: CameraEffectType::Flash {
                color,
                fade_speed: 1.0 / duration,
            },
            elapsed: 0.0,
            duration,
            intensity: 1.0,
            active: true,
            priority: 0,
        }
    }

    /// Update the effect.
    pub fn update(&mut self, dt: f32) {
        self.elapsed += dt;
        if self.duration > 0.0 && self.elapsed >= self.duration {
            self.active = false;
        }

        // Decay intensity.
        match &self.effect_type {
            CameraEffectType::Shake { decay, .. } => {
                if self.duration > 0.0 {
                    let t = self.elapsed / self.duration;
                    self.intensity = (1.0 - t).max(0.0).powf(*decay);
                }
            }
            CameraEffectType::Flash { fade_speed, .. } => {
                self.intensity = (1.0 - self.elapsed * fade_speed).max(0.0);
            }
            _ => {}
        }
    }

    /// Compute position offset from this effect.
    pub fn position_offset(&self) -> [f32; 3] {
        match &self.effect_type {
            CameraEffectType::Shake { amplitude, frequency, .. } => {
                let t = self.elapsed * frequency * 2.0 * PI;
                let x = (t * 1.0).sin() * amplitude * self.intensity;
                let y = (t * 1.3).sin() * amplitude * self.intensity * 0.7;
                let z = (t * 0.7).sin() * amplitude * self.intensity * 0.3;
                [x, y, z]
            }
            _ => [0.0; 3],
        }
    }

    /// Compute FOV offset from this effect.
    pub fn fov_offset(&self) -> f32 {
        match &self.effect_type {
            CameraEffectType::ZoomPulse { fov_delta, speed } => {
                let t = self.elapsed * speed;
                fov_delta * t.sin() * self.intensity
            }
            _ => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Cinematic Path
// ---------------------------------------------------------------------------

/// A keyframe on a cinematic camera path.
#[derive(Debug, Clone)]
pub struct CinematicKeyframe {
    /// Time of this keyframe (in seconds from path start).
    pub time: f32,
    /// Camera position.
    pub position: [f32; 3],
    /// Look-at target.
    pub look_at: [f32; 3],
    /// Field of view at this keyframe.
    pub fov: f32,
    /// Roll angle.
    pub roll: f32,
    /// Interpolation mode.
    pub interpolation: InterpolationMode,
}

/// Interpolation mode between keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Linear interpolation.
    Linear,
    /// Smooth (cubic) interpolation.
    Smooth,
    /// Ease-in.
    EaseIn,
    /// Ease-out.
    EaseOut,
    /// Ease-in-out.
    EaseInOut,
}

/// A cinematic camera path.
#[derive(Debug, Clone)]
pub struct CinematicPath {
    /// Path name.
    pub name: String,
    /// Keyframes sorted by time.
    pub keyframes: Vec<CinematicKeyframe>,
    /// Whether to loop the path.
    pub looping: bool,
    /// Total path duration.
    pub duration: f32,
}

impl CinematicPath {
    /// Create a new cinematic path.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            keyframes: Vec::new(),
            looping: false,
            duration: 0.0,
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, keyframe: CinematicKeyframe) {
        self.duration = self.duration.max(keyframe.time);
        self.keyframes.push(keyframe);
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Evaluate the path at a given time.
    pub fn evaluate(&self, time: f32) -> CameraTransformV2 {
        if self.keyframes.is_empty() {
            return CameraTransformV2::default();
        }

        let t = if self.looping && self.duration > 0.0 {
            time % self.duration
        } else {
            time.min(self.duration)
        };

        // Find surrounding keyframes.
        let mut kf_a = &self.keyframes[0];
        let mut kf_b = &self.keyframes[self.keyframes.len() - 1];

        for i in 0..self.keyframes.len() - 1 {
            if t >= self.keyframes[i].time && t <= self.keyframes[i + 1].time {
                kf_a = &self.keyframes[i];
                kf_b = &self.keyframes[i + 1];
                break;
            }
        }

        let segment_duration = kf_b.time - kf_a.time;
        let local_t = if segment_duration > 1e-6 {
            ((t - kf_a.time) / segment_duration).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let smooth_t = apply_interpolation(local_t, kf_a.interpolation);

        let position = [
            kf_a.position[0] + (kf_b.position[0] - kf_a.position[0]) * smooth_t,
            kf_a.position[1] + (kf_b.position[1] - kf_a.position[1]) * smooth_t,
            kf_a.position[2] + (kf_b.position[2] - kf_a.position[2]) * smooth_t,
        ];
        let look_at = [
            kf_a.look_at[0] + (kf_b.look_at[0] - kf_a.look_at[0]) * smooth_t,
            kf_a.look_at[1] + (kf_b.look_at[1] - kf_a.look_at[1]) * smooth_t,
            kf_a.look_at[2] + (kf_b.look_at[2] - kf_a.look_at[2]) * smooth_t,
        ];

        CameraTransformV2::look_at(position, look_at, [0.0, 1.0, 0.0])
    }
}

/// Apply interpolation easing.
fn apply_interpolation(t: f32, mode: InterpolationMode) -> f32 {
    match mode {
        InterpolationMode::Linear => t,
        InterpolationMode::Smooth => t * t * (3.0 - 2.0 * t),
        InterpolationMode::EaseIn => t * t,
        InterpolationMode::EaseOut => t * (2.0 - t),
        InterpolationMode::EaseInOut => {
            if t < 0.5 { 2.0 * t * t } else { -1.0 + (4.0 - 2.0 * t) * t }
        }
    }
}

// ---------------------------------------------------------------------------
// Split Screen
// ---------------------------------------------------------------------------

/// Split screen layout configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitScreenLayout {
    /// Full screen (no split).
    FullScreen,
    /// Horizontal split (top/bottom).
    HorizontalSplit,
    /// Vertical split (left/right).
    VerticalSplit,
    /// Four-way split.
    QuadSplit,
    /// Custom layout.
    Custom,
}

/// A viewport for split-screen rendering.
#[derive(Debug, Clone)]
pub struct SplitScreenViewport {
    /// Normalized viewport rectangle (x, y, width, height) in [0, 1].
    pub rect: [f32; 4],
    /// Player index.
    pub player_index: u32,
    /// Camera index within the camera system.
    pub camera_index: u32,
    /// Whether this viewport is active.
    pub active: bool,
}

/// Split screen manager.
#[derive(Debug, Clone)]
pub struct SplitScreenManager {
    /// Layout mode.
    pub layout: SplitScreenLayout,
    /// Viewports.
    pub viewports: Vec<SplitScreenViewport>,
    /// Screen resolution.
    pub screen_width: u32,
    /// Screen height.
    pub screen_height: u32,
}

impl SplitScreenManager {
    /// Create a full-screen single-player layout.
    pub fn single_player() -> Self {
        Self {
            layout: SplitScreenLayout::FullScreen,
            viewports: vec![SplitScreenViewport {
                rect: [0.0, 0.0, 1.0, 1.0],
                player_index: 0,
                camera_index: 0,
                active: true,
            }],
            screen_width: 1920,
            screen_height: 1080,
        }
    }

    /// Create a two-player vertical split.
    pub fn two_player_vertical() -> Self {
        Self {
            layout: SplitScreenLayout::VerticalSplit,
            viewports: vec![
                SplitScreenViewport { rect: [0.0, 0.0, 0.5, 1.0], player_index: 0, camera_index: 0, active: true },
                SplitScreenViewport { rect: [0.5, 0.0, 0.5, 1.0], player_index: 1, camera_index: 1, active: true },
            ],
            screen_width: 1920,
            screen_height: 1080,
        }
    }

    /// Create a two-player horizontal split.
    pub fn two_player_horizontal() -> Self {
        Self {
            layout: SplitScreenLayout::HorizontalSplit,
            viewports: vec![
                SplitScreenViewport { rect: [0.0, 0.0, 1.0, 0.5], player_index: 0, camera_index: 0, active: true },
                SplitScreenViewport { rect: [0.0, 0.5, 1.0, 0.5], player_index: 1, camera_index: 1, active: true },
            ],
            screen_width: 1920,
            screen_height: 1080,
        }
    }

    /// Create a four-player quad split.
    pub fn four_player() -> Self {
        Self {
            layout: SplitScreenLayout::QuadSplit,
            viewports: vec![
                SplitScreenViewport { rect: [0.0, 0.0, 0.5, 0.5], player_index: 0, camera_index: 0, active: true },
                SplitScreenViewport { rect: [0.5, 0.0, 0.5, 0.5], player_index: 1, camera_index: 1, active: true },
                SplitScreenViewport { rect: [0.0, 0.5, 0.5, 0.5], player_index: 2, camera_index: 2, active: true },
                SplitScreenViewport { rect: [0.5, 0.5, 0.5, 0.5], player_index: 3, camera_index: 3, active: true },
            ],
            screen_width: 1920,
            screen_height: 1080,
        }
    }

    /// Get pixel-space viewport rect for a given viewport.
    pub fn pixel_rect(&self, viewport_index: usize) -> (u32, u32, u32, u32) {
        if let Some(vp) = self.viewports.get(viewport_index) {
            let x = (vp.rect[0] * self.screen_width as f32) as u32;
            let y = (vp.rect[1] * self.screen_height as f32) as u32;
            let w = (vp.rect[2] * self.screen_width as f32) as u32;
            let h = (vp.rect[3] * self.screen_height as f32) as u32;
            (x, y, w, h)
        } else {
            (0, 0, self.screen_width, self.screen_height)
        }
    }

    /// Get the aspect ratio for a viewport.
    pub fn aspect_ratio(&self, viewport_index: usize) -> f32 {
        let (_, _, w, h) = self.pixel_rect(viewport_index);
        if h == 0 { return 1.0; }
        w as f32 / h as f32
    }
}

// ---------------------------------------------------------------------------
// Camera System V2
// ---------------------------------------------------------------------------

/// The main enhanced camera system.
#[derive(Debug)]
pub struct CameraSystemV2 {
    /// Current camera transform.
    pub transform: CameraTransformV2,
    /// Current camera mode.
    pub mode: CameraMode,
    /// Field of view in degrees.
    pub fov: f32,
    /// Near clip plane.
    pub near: f32,
    /// Far clip plane.
    pub far: f32,
    /// Target position (for orbit/follow modes).
    pub target_position: [f32; 3],
    /// Target rotation (for follow mode).
    pub target_rotation: f32,
    /// Orbit configuration.
    pub orbit_config: OrbitConfig,
    /// Follow configuration.
    pub follow_config: FollowConfig,
    /// First-person configuration.
    pub fp_config: FirstPersonConfigV2,
    /// Top-down configuration.
    pub top_down_config: TopDownConfigV2,
    /// Active camera effects.
    pub effects: Vec<CameraEffect>,
    /// Cinematic paths.
    pub cinematic_paths: Vec<CinematicPath>,
    /// Current cinematic path index (-1 = none).
    pub active_cinematic: i32,
    /// Cinematic playback time.
    pub cinematic_time: f32,
    /// Split screen manager.
    pub split_screen: SplitScreenManager,
    /// Transition state.
    pub transition: Option<CameraTransition>,
    /// Orbit yaw angle (radians).
    pub orbit_yaw: f32,
    /// Orbit pitch angle (radians).
    pub orbit_pitch: f32,
    /// Orbit distance.
    pub orbit_distance: f32,
}

/// Camera transition state.
#[derive(Debug, Clone)]
pub struct CameraTransition {
    /// Source transform.
    pub from: CameraTransformV2,
    /// Target transform.
    pub to: CameraTransformV2,
    /// Source FOV.
    pub from_fov: f32,
    /// Target FOV.
    pub to_fov: f32,
    /// Duration.
    pub duration: f32,
    /// Elapsed time.
    pub elapsed: f32,
    /// Source mode.
    pub from_mode: CameraMode,
    /// Target mode.
    pub to_mode: CameraMode,
}

impl CameraSystemV2 {
    /// Create a new camera system.
    pub fn new() -> Self {
        Self {
            transform: CameraTransformV2::default(),
            mode: CameraMode::Follow,
            fov: DEFAULT_FOV,
            near: DEFAULT_NEAR,
            far: DEFAULT_FAR,
            target_position: [0.0; 3],
            target_rotation: 0.0,
            orbit_config: OrbitConfig::default(),
            follow_config: FollowConfig::default(),
            fp_config: FirstPersonConfigV2::default(),
            top_down_config: TopDownConfigV2::default(),
            effects: Vec::new(),
            cinematic_paths: Vec::new(),
            active_cinematic: -1,
            cinematic_time: 0.0,
            split_screen: SplitScreenManager::single_player(),
            transition: None,
            orbit_yaw: 0.0,
            orbit_pitch: 0.3,
            orbit_distance: 10.0,
        }
    }

    /// Switch to a new camera mode with an optional transition.
    pub fn switch_mode(&mut self, mode: CameraMode, transition_duration: f32) {
        if transition_duration > 0.0 {
            self.transition = Some(CameraTransition {
                from: self.transform,
                to: self.transform,
                from_fov: self.fov,
                to_fov: self.fov,
                duration: transition_duration,
                elapsed: 0.0,
                from_mode: self.mode,
                to_mode: mode,
            });
        }
        self.mode = mode;
    }

    /// Add a camera effect.
    pub fn add_effect(&mut self, effect: CameraEffect) {
        if self.effects.len() < MAX_EFFECTS {
            self.effects.push(effect);
        }
    }

    /// Add a cinematic path.
    pub fn add_cinematic_path(&mut self, path: CinematicPath) -> usize {
        let idx = self.cinematic_paths.len();
        self.cinematic_paths.push(path);
        idx
    }

    /// Start playing a cinematic path.
    pub fn play_cinematic(&mut self, path_index: usize) {
        if path_index < self.cinematic_paths.len() {
            self.active_cinematic = path_index as i32;
            self.cinematic_time = 0.0;
            self.mode = CameraMode::Cinematic;
        }
    }

    /// Stop the cinematic.
    pub fn stop_cinematic(&mut self) {
        self.active_cinematic = -1;
    }

    /// Update the camera system.
    pub fn update(&mut self, dt: f32) {
        // Update effects.
        for effect in &mut self.effects {
            effect.update(dt);
        }
        self.effects.retain(|e| e.active);

        // Update transition.
        if let Some(transition) = &mut self.transition {
            transition.elapsed += dt;
            if transition.elapsed >= transition.duration {
                self.transition = None;
            }
        }

        // Update cinematic.
        if self.active_cinematic >= 0 {
            self.cinematic_time += dt;
            let idx = self.active_cinematic as usize;
            if let Some(path) = self.cinematic_paths.get(idx) {
                if self.cinematic_time >= path.duration && !path.looping {
                    self.stop_cinematic();
                } else {
                    self.transform = path.evaluate(self.cinematic_time);
                }
            }
        }

        // Apply effects to transform.
        let mut position_offset = [0.0f32; 3];
        let mut fov_offset = 0.0f32;

        for effect in &self.effects {
            let off = effect.position_offset();
            position_offset[0] += off[0];
            position_offset[1] += off[1];
            position_offset[2] += off[2];
            fov_offset += effect.fov_offset();
        }

        self.transform.position[0] += position_offset[0];
        self.transform.position[1] += position_offset[1];
        self.transform.position[2] += position_offset[2];
        self.fov = (self.fov + fov_offset).clamp(MIN_FOV, MAX_FOV);
    }

    /// Get the current effective FOV.
    pub fn effective_fov(&self) -> f32 {
        self.fov.clamp(MIN_FOV, MAX_FOV)
    }

    /// Whether the camera is in a cinematic.
    pub fn is_cinematic_playing(&self) -> bool {
        self.active_cinematic >= 0
    }

    /// Whether a transition is in progress.
    pub fn is_transitioning(&self) -> bool {
        self.transition.is_some()
    }
}
