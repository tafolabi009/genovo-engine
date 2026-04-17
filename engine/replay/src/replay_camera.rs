//! # Replay Camera Modes
//!
//! Provides camera systems for replay playback: free camera, follow player,
//! orbit player, directed camera (cinematic angles), picture-in-picture,
//! split screen replay, slow motion with smooth speed ramping, and time
//! reversal.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Math types (minimal, self-contained)
// ---------------------------------------------------------------------------

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const FORWARD: Self = Self { x: 0.0, y: 0.0, z: -1.0 };
    pub const RIGHT: Self = Self { x: 1.0, y: 0.0, z: 0.0 };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-8 { return Self::ZERO; }
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    pub fn distance(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

/// Quaternion rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    pub fn from_euler(yaw: f32, pitch: f32, roll: f32) -> Self {
        let (sy, cy) = (yaw * 0.5).sin_cos();
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sr, cr) = (roll * 0.5).sin_cos();
        Self {
            x: cr * sp * cy + sr * cp * sy,
            y: cr * cp * sy - sr * sp * cy,
            z: sr * cp * cy - cr * sp * sy,
            w: cr * cp * cy + sr * sp * sy,
        }
    }

    pub fn slerp(self, other: Self, t: f32) -> Self {
        let mut dot = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w;
        let mut b = other;
        if dot < 0.0 {
            dot = -dot;
            b = Self { x: -b.x, y: -b.y, z: -b.z, w: -b.w };
        }
        if dot > 0.9995 {
            let len = ((self.x + (b.x - self.x) * t).powi(2) + (self.y + (b.y - self.y) * t).powi(2) + (self.z + (b.z - self.z) * t).powi(2) + (self.w + (b.w - self.w) * t).powi(2)).sqrt();
            return Self {
                x: (self.x + (b.x - self.x) * t) / len,
                y: (self.y + (b.y - self.y) * t) / len,
                z: (self.z + (b.z - self.z) * t) / len,
                w: (self.w + (b.w - self.w) * t) / len,
            };
        }
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;
        Self {
            x: self.x * wa + b.x * wb,
            y: self.y * wa + b.y * wb,
            z: self.z * wa + b.z * wb,
            w: self.w * wa + b.w * wb,
        }
    }

    pub fn rotate_vector(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        let dot_uv = u.dot(v);
        let dot_uu = u.dot(u);
        let cross_uv = u.cross(v);
        Vec3 {
            x: 2.0 * dot_uv * u.x + (s * s - dot_uu) * v.x + 2.0 * s * cross_uv.x,
            y: 2.0 * dot_uv * u.y + (s * s - dot_uu) * v.y + 2.0 * s * cross_uv.y,
            z: 2.0 * dot_uv * u.z + (s * s - dot_uu) * v.z + 2.0 * s * cross_uv.z,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera transform
// ---------------------------------------------------------------------------

/// A camera's spatial configuration.
#[derive(Debug, Clone, Copy)]
pub struct CameraTransform {
    /// World-space position.
    pub position: Vec3,
    /// World-space rotation.
    pub rotation: Quat,
    /// Field of view in degrees (vertical).
    pub fov: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance.
    pub far: f32,
}

impl Default for CameraTransform {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 5.0),
            rotation: Quat::IDENTITY,
            fov: 60.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl CameraTransform {
    /// Compute the forward direction.
    pub fn forward(&self) -> Vec3 {
        self.rotation.rotate_vector(Vec3::FORWARD)
    }

    /// Compute the right direction.
    pub fn right(&self) -> Vec3 {
        self.rotation.rotate_vector(Vec3::RIGHT)
    }

    /// Compute the up direction.
    pub fn up(&self) -> Vec3 {
        self.rotation.rotate_vector(Vec3::UP)
    }

    /// Interpolate between two camera transforms.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(other.position, t),
            rotation: self.rotation.slerp(other.rotation, t),
            fov: self.fov + (other.fov - self.fov) * t,
            near: self.near + (other.near - self.near) * t,
            far: self.far + (other.far - self.far) * t,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera mode enum
// ---------------------------------------------------------------------------

/// Available replay camera modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CameraMode {
    /// Free-flying camera controlled by user input.
    Free,
    /// Follows a target entity from behind.
    Follow,
    /// Orbits around a target entity.
    Orbit,
    /// Pre-scripted cinematic camera angles.
    Directed,
    /// Fixed position looking at the target.
    Fixed,
    /// First-person view from the target entity.
    FirstPerson,
}

impl fmt::Display for CameraMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Free => write!(f, "Free"),
            Self::Follow => write!(f, "Follow"),
            Self::Orbit => write!(f, "Orbit"),
            Self::Directed => write!(f, "Directed"),
            Self::Fixed => write!(f, "Fixed"),
            Self::FirstPerson => write!(f, "First Person"),
        }
    }
}

// ---------------------------------------------------------------------------
// Free camera
// ---------------------------------------------------------------------------

/// A free-flying camera for replay viewing.
#[derive(Debug, Clone)]
pub struct FreeCamera {
    /// Current transform.
    pub transform: CameraTransform,
    /// Movement speed (units per second).
    pub move_speed: f32,
    /// Fast movement speed (shift held).
    pub fast_speed: f32,
    /// Rotation sensitivity (degrees per pixel).
    pub sensitivity: f32,
    /// Current yaw angle (radians).
    pub yaw: f32,
    /// Current pitch angle (radians).
    pub pitch: f32,
    /// Smoothing factor for movement (0 = no smoothing, 1 = max smoothing).
    pub smoothing: f32,
    /// Target position (for smoothing).
    target_position: Vec3,
}

impl FreeCamera {
    /// Create a new free camera.
    pub fn new() -> Self {
        Self {
            transform: CameraTransform::default(),
            move_speed: 5.0,
            fast_speed: 15.0,
            sensitivity: 0.2,
            yaw: 0.0,
            pitch: 0.0,
            smoothing: 0.1,
            target_position: Vec3::new(0.0, 2.0, 5.0),
        }
    }

    /// Update the free camera.
    pub fn update(&mut self, dt: f32, input: &CameraInput) {
        // Rotation
        self.yaw += input.mouse_dx * self.sensitivity * dt;
        self.pitch += input.mouse_dy * self.sensitivity * dt;
        self.pitch = self.pitch.clamp(-89.0f32.to_radians(), 89.0f32.to_radians());

        self.transform.rotation = Quat::from_euler(self.yaw, self.pitch, 0.0);

        // Movement
        let speed = if input.fast { self.fast_speed } else { self.move_speed };
        let forward = self.transform.forward();
        let right = self.transform.right();
        let up = Vec3::UP;

        let mut velocity = Vec3::ZERO;
        if input.forward { velocity = velocity + forward; }
        if input.backward { velocity = velocity - forward; }
        if input.right { velocity = velocity + right; }
        if input.left { velocity = velocity - right; }
        if input.up { velocity = velocity + up; }
        if input.down { velocity = velocity - up; }

        let vel_len = velocity.length();
        if vel_len > 0.01 {
            velocity = velocity.normalized().scale(speed);
        }

        self.target_position = self.target_position + velocity.scale(dt);

        // Smooth interpolation
        let smooth_factor = (1.0 - self.smoothing).clamp(0.01, 1.0);
        let lerp_t = (smooth_factor * dt * 60.0).min(1.0);
        self.transform.position = self.transform.position.lerp(self.target_position, lerp_t);
    }

    /// Set position directly.
    pub fn set_position(&mut self, pos: Vec3) {
        self.transform.position = pos;
        self.target_position = pos;
    }
}

impl Default for FreeCamera {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Camera input
// ---------------------------------------------------------------------------

/// Input state for camera control.
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraInput {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub fast: bool,
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub scroll: f32,
}

// ---------------------------------------------------------------------------
// Follow camera
// ---------------------------------------------------------------------------

/// A camera that follows a target from behind.
#[derive(Debug, Clone)]
pub struct FollowCamera {
    /// Current transform.
    pub transform: CameraTransform,
    /// Offset behind and above the target.
    pub follow_offset: Vec3,
    /// Look-at offset (above the target's feet).
    pub look_at_offset: Vec3,
    /// Smoothing speed for position.
    pub position_smooth: f32,
    /// Smoothing speed for rotation.
    pub rotation_smooth: f32,
    /// Current smoothed position.
    smoothed_position: Vec3,
}

impl FollowCamera {
    /// Create a new follow camera.
    pub fn new() -> Self {
        Self {
            transform: CameraTransform::default(),
            follow_offset: Vec3::new(0.0, 3.0, 8.0),
            look_at_offset: Vec3::new(0.0, 1.5, 0.0),
            position_smooth: 5.0,
            rotation_smooth: 8.0,
            smoothed_position: Vec3::ZERO,
        }
    }

    /// Update the follow camera.
    pub fn update(&mut self, dt: f32, target_position: Vec3, target_forward: Vec3) {
        // Desired position: behind and above the target
        let behind = -target_forward.normalized();
        let desired_pos = target_position
            + behind.scale(self.follow_offset.z)
            + Vec3::UP.scale(self.follow_offset.y)
            + target_forward.cross(Vec3::UP).normalized().scale(self.follow_offset.x);

        // Smooth position
        let pos_t = (self.position_smooth * dt).min(1.0);
        self.smoothed_position = self.smoothed_position.lerp(desired_pos, pos_t);
        self.transform.position = self.smoothed_position;

        // Look at target + offset
        let look_target = target_position + self.look_at_offset;
        let look_dir = (look_target - self.transform.position).normalized();

        // Compute rotation from look direction
        let pitch = (-look_dir.y).asin();
        let yaw = look_dir.x.atan2(-look_dir.z);
        let desired_rotation = Quat::from_euler(yaw, pitch, 0.0);

        let rot_t = (self.rotation_smooth * dt).min(1.0);
        self.transform.rotation = self.transform.rotation.slerp(desired_rotation, rot_t);
    }

    /// Set to a specific target immediately (no smoothing).
    pub fn snap_to(&mut self, target_position: Vec3, target_forward: Vec3) {
        let behind = -target_forward.normalized();
        let pos = target_position
            + behind.scale(self.follow_offset.z)
            + Vec3::UP.scale(self.follow_offset.y);
        self.smoothed_position = pos;
        self.transform.position = pos;
    }
}

impl Default for FollowCamera {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Orbit camera
// ---------------------------------------------------------------------------

/// A camera that orbits around a target.
#[derive(Debug, Clone)]
pub struct OrbitCamera {
    /// Current transform.
    pub transform: CameraTransform,
    /// Distance from the target.
    pub distance: f32,
    /// Minimum distance.
    pub min_distance: f32,
    /// Maximum distance.
    pub max_distance: f32,
    /// Current azimuth angle (radians, around Y axis).
    pub azimuth: f32,
    /// Current elevation angle (radians, above horizon).
    pub elevation: f32,
    /// Minimum elevation.
    pub min_elevation: f32,
    /// Maximum elevation.
    pub max_elevation: f32,
    /// Orbit rotation speed.
    pub rotation_speed: f32,
    /// Zoom speed.
    pub zoom_speed: f32,
    /// Target position smoothing.
    pub smooth_speed: f32,
    /// Smoothed target position.
    smoothed_target: Vec3,
    /// Auto-rotate speed (radians/sec, 0 = disabled).
    pub auto_rotate_speed: f32,
}

impl OrbitCamera {
    /// Create a new orbit camera.
    pub fn new() -> Self {
        Self {
            transform: CameraTransform::default(),
            distance: 10.0,
            min_distance: 2.0,
            max_distance: 50.0,
            azimuth: 0.0,
            elevation: 0.3,
            min_elevation: -1.0,
            max_elevation: 1.4,
            rotation_speed: 2.0,
            zoom_speed: 5.0,
            smooth_speed: 8.0,
            smoothed_target: Vec3::ZERO,
            auto_rotate_speed: 0.0,
        }
    }

    /// Update the orbit camera.
    pub fn update(&mut self, dt: f32, target_position: Vec3, input: &CameraInput) {
        // Auto-rotate
        self.azimuth += self.auto_rotate_speed * dt;

        // Input-driven rotation
        self.azimuth += input.mouse_dx * self.rotation_speed * dt;
        self.elevation += input.mouse_dy * self.rotation_speed * dt;
        self.elevation = self.elevation.clamp(self.min_elevation, self.max_elevation);

        // Zoom
        self.distance -= input.scroll * self.zoom_speed * dt;
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);

        // Smooth target
        let target_t = (self.smooth_speed * dt).min(1.0);
        self.smoothed_target = self.smoothed_target.lerp(target_position, target_t);

        // Compute camera position on the orbit sphere
        let cos_elev = self.elevation.cos();
        let sin_elev = self.elevation.sin();
        let cos_azim = self.azimuth.cos();
        let sin_azim = self.azimuth.sin();

        let offset = Vec3::new(
            cos_elev * sin_azim * self.distance,
            sin_elev * self.distance,
            cos_elev * cos_azim * self.distance,
        );

        self.transform.position = self.smoothed_target + offset;

        // Look at target
        let look_dir = (self.smoothed_target - self.transform.position).normalized();
        let pitch = (-look_dir.y).asin();
        let yaw = look_dir.x.atan2(-look_dir.z);
        self.transform.rotation = Quat::from_euler(yaw, pitch, 0.0);
    }

    /// Set the orbit distance.
    pub fn set_distance(&mut self, distance: f32) {
        self.distance = distance.clamp(self.min_distance, self.max_distance);
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Directed camera (cinematic)
// ---------------------------------------------------------------------------

/// A single shot in a directed camera sequence.
#[derive(Debug, Clone)]
pub struct CameraShot {
    /// Name of this shot.
    pub name: String,
    /// Start time in the replay (seconds).
    pub start_time: f32,
    /// End time in the replay (seconds).
    pub end_time: f32,
    /// Starting camera transform.
    pub start_transform: CameraTransform,
    /// Ending camera transform.
    pub end_transform: CameraTransform,
    /// Easing function for the interpolation.
    pub easing: EasingFunction,
    /// Target entity to focus on (None = use transform's forward).
    pub focus_entity: Option<u32>,
}

/// Easing functions for camera interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    SmootherStep,
}

impl EasingFunction {
    /// Evaluate the easing function at `t` in [0, 1].
    pub fn evaluate(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            Self::SmootherStep => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
        }
    }
}

/// A sequence of directed camera shots.
#[derive(Debug, Clone)]
pub struct DirectedCameraSequence {
    /// Name of this sequence.
    pub name: String,
    /// Ordered list of camera shots.
    pub shots: Vec<CameraShot>,
    /// Total duration of the sequence.
    pub duration: f32,
}

impl DirectedCameraSequence {
    /// Create a new sequence.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            shots: Vec::new(),
            duration: 0.0,
        }
    }

    /// Add a shot to the sequence.
    pub fn add_shot(&mut self, shot: CameraShot) {
        if shot.end_time > self.duration {
            self.duration = shot.end_time;
        }
        self.shots.push(shot);
    }

    /// Evaluate the sequence at a given replay time.
    pub fn evaluate(&self, time: f32) -> Option<CameraTransform> {
        // Find the active shot
        for shot in &self.shots {
            if time >= shot.start_time && time <= shot.end_time {
                let shot_duration = shot.end_time - shot.start_time;
                let local_t = if shot_duration > 0.0 {
                    (time - shot.start_time) / shot_duration
                } else {
                    1.0
                };
                let eased_t = shot.easing.evaluate(local_t);
                return Some(shot.start_transform.lerp(shot.end_transform, eased_t));
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Playback speed controller
// ---------------------------------------------------------------------------

/// Controls replay playback speed with smooth ramping.
#[derive(Debug, Clone)]
pub struct PlaybackSpeedController {
    /// Current playback speed (1.0 = normal).
    pub current_speed: f32,
    /// Target speed we're ramping toward.
    pub target_speed: f32,
    /// Speed of the ramp (units per second).
    pub ramp_speed: f32,
    /// Minimum allowed speed.
    pub min_speed: f32,
    /// Maximum allowed speed.
    pub max_speed: f32,
    /// Whether playback is paused.
    pub paused: bool,
    /// Whether playing in reverse.
    pub reversed: bool,
    /// Preset speeds for quick switching.
    pub presets: Vec<f32>,
    /// Current preset index.
    pub preset_index: usize,
}

impl PlaybackSpeedController {
    /// Create with default settings.
    pub fn new() -> Self {
        Self {
            current_speed: 1.0,
            target_speed: 1.0,
            ramp_speed: 3.0,
            min_speed: 0.01,
            max_speed: 16.0,
            paused: false,
            reversed: false,
            presets: vec![0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            preset_index: 3, // 1.0x
        }
    }

    /// Update the speed controller.
    pub fn update(&mut self, dt: f32) {
        if (self.current_speed - self.target_speed).abs() > 0.001 {
            let diff = self.target_speed - self.current_speed;
            let step = self.ramp_speed * dt;
            if diff.abs() <= step {
                self.current_speed = self.target_speed;
            } else {
                self.current_speed += diff.signum() * step;
            }
        }
    }

    /// Get the effective speed for advancing replay time.
    pub fn effective_speed(&self) -> f32 {
        if self.paused {
            return 0.0;
        }
        let speed = self.current_speed;
        if self.reversed { -speed } else { speed }
    }

    /// Set the target speed.
    pub fn set_speed(&mut self, speed: f32) {
        self.target_speed = speed.clamp(self.min_speed, self.max_speed);
    }

    /// Set speed immediately (no ramping).
    pub fn set_speed_immediate(&mut self, speed: f32) {
        let speed = speed.clamp(self.min_speed, self.max_speed);
        self.current_speed = speed;
        self.target_speed = speed;
    }

    /// Toggle pause.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Toggle reverse.
    pub fn toggle_reverse(&mut self) {
        self.reversed = !self.reversed;
    }

    /// Switch to the next preset speed.
    pub fn next_preset(&mut self) {
        if self.preset_index + 1 < self.presets.len() {
            self.preset_index += 1;
            self.target_speed = self.presets[self.preset_index];
        }
    }

    /// Switch to the previous preset speed.
    pub fn prev_preset(&mut self) {
        if self.preset_index > 0 {
            self.preset_index -= 1;
            self.target_speed = self.presets[self.preset_index];
        }
    }

    /// Enter slow motion (ramp to a slow speed).
    pub fn enter_slow_motion(&mut self, slow_speed: f32) {
        self.target_speed = slow_speed.clamp(self.min_speed, 1.0);
    }

    /// Exit slow motion (ramp back to 1.0).
    pub fn exit_slow_motion(&mut self) {
        self.target_speed = 1.0;
    }
}

impl Default for PlaybackSpeedController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Picture-in-picture
// ---------------------------------------------------------------------------

/// Configuration for a picture-in-picture viewport.
#[derive(Debug, Clone)]
pub struct PipViewport {
    /// Normalised screen rectangle [x, y, width, height] in [0, 1].
    pub rect: [f32; 4],
    /// Camera mode for this viewport.
    pub mode: CameraMode,
    /// Target entity (for follow/orbit modes).
    pub target_entity: Option<u32>,
    /// Border colour (RGBA).
    pub border_color: [f32; 4],
    /// Border width in pixels.
    pub border_width: f32,
    /// Whether this PiP is visible.
    pub visible: bool,
    /// Opacity (0 = transparent, 1 = opaque).
    pub opacity: f32,
}

impl PipViewport {
    /// Create a default PiP viewport in the bottom-right corner.
    pub fn bottom_right() -> Self {
        Self {
            rect: [0.7, 0.7, 0.28, 0.28],
            mode: CameraMode::Follow,
            target_entity: None,
            border_color: [1.0, 1.0, 1.0, 1.0],
            border_width: 2.0,
            visible: true,
            opacity: 1.0,
        }
    }

    /// Create a PiP viewport in the top-right corner.
    pub fn top_right() -> Self {
        Self {
            rect: [0.7, 0.02, 0.28, 0.28],
            ..Self::bottom_right()
        }
    }
}

// ---------------------------------------------------------------------------
// Split screen
// ---------------------------------------------------------------------------

/// Split-screen layout for replay viewing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitLayout {
    /// No split (single view).
    Single,
    /// Two views side by side.
    Horizontal,
    /// Two views stacked vertically.
    Vertical,
    /// Four views in a 2x2 grid.
    Quad,
}

/// A single split-screen viewport.
#[derive(Debug, Clone)]
pub struct SplitViewport {
    /// Normalised screen rectangle.
    pub rect: [f32; 4],
    /// Camera mode.
    pub mode: CameraMode,
    /// Target entity.
    pub target_entity: Option<u32>,
    /// Whether this is the active (user-controlled) viewport.
    pub active: bool,
}

/// Generate viewport rectangles for a split layout.
pub fn generate_split_viewports(layout: SplitLayout) -> Vec<SplitViewport> {
    match layout {
        SplitLayout::Single => vec![SplitViewport {
            rect: [0.0, 0.0, 1.0, 1.0],
            mode: CameraMode::Free,
            target_entity: None,
            active: true,
        }],
        SplitLayout::Horizontal => vec![
            SplitViewport {
                rect: [0.0, 0.0, 0.5, 1.0],
                mode: CameraMode::Free,
                target_entity: None,
                active: true,
            },
            SplitViewport {
                rect: [0.5, 0.0, 0.5, 1.0],
                mode: CameraMode::Follow,
                target_entity: None,
                active: false,
            },
        ],
        SplitLayout::Vertical => vec![
            SplitViewport {
                rect: [0.0, 0.0, 1.0, 0.5],
                mode: CameraMode::Free,
                target_entity: None,
                active: true,
            },
            SplitViewport {
                rect: [0.0, 0.5, 1.0, 0.5],
                mode: CameraMode::Follow,
                target_entity: None,
                active: false,
            },
        ],
        SplitLayout::Quad => vec![
            SplitViewport {
                rect: [0.0, 0.0, 0.5, 0.5],
                mode: CameraMode::Free,
                target_entity: None,
                active: true,
            },
            SplitViewport {
                rect: [0.5, 0.0, 0.5, 0.5],
                mode: CameraMode::Follow,
                target_entity: None,
                active: false,
            },
            SplitViewport {
                rect: [0.0, 0.5, 0.5, 0.5],
                mode: CameraMode::Orbit,
                target_entity: None,
                active: false,
            },
            SplitViewport {
                rect: [0.5, 0.5, 0.5, 0.5],
                mode: CameraMode::Directed,
                target_entity: None,
                active: false,
            },
        ],
    }
}

// ---------------------------------------------------------------------------
// Replay camera controller (master)
// ---------------------------------------------------------------------------

/// Master replay camera controller that manages all camera modes.
pub struct ReplayCameraController {
    /// Current active mode.
    pub mode: CameraMode,
    /// Free camera.
    pub free_camera: FreeCamera,
    /// Follow camera.
    pub follow_camera: FollowCamera,
    /// Orbit camera.
    pub orbit_camera: OrbitCamera,
    /// Directed camera sequence.
    pub directed_sequence: Option<DirectedCameraSequence>,
    /// Playback speed controller.
    pub speed: PlaybackSpeedController,
    /// PiP viewports.
    pub pip_viewports: Vec<PipViewport>,
    /// Split-screen layout.
    pub split_layout: SplitLayout,
    /// Split viewports.
    pub split_viewports: Vec<SplitViewport>,
    /// Current replay time (seconds).
    pub replay_time: f32,
    /// Total replay duration (seconds).
    pub total_duration: f32,
    /// Target entity position (updated each frame from replay data).
    pub target_position: Vec3,
    /// Target entity forward direction.
    pub target_forward: Vec3,
}

impl ReplayCameraController {
    /// Create a new controller.
    pub fn new(total_duration: f32) -> Self {
        Self {
            mode: CameraMode::Free,
            free_camera: FreeCamera::new(),
            follow_camera: FollowCamera::new(),
            orbit_camera: OrbitCamera::new(),
            directed_sequence: None,
            speed: PlaybackSpeedController::new(),
            pip_viewports: Vec::new(),
            split_layout: SplitLayout::Single,
            split_viewports: generate_split_viewports(SplitLayout::Single),
            replay_time: 0.0,
            total_duration,
            target_position: Vec3::ZERO,
            target_forward: Vec3::FORWARD,
        }
    }

    /// Update the camera controller.
    pub fn update(&mut self, dt: f32, input: &CameraInput) {
        // Update speed
        self.speed.update(dt);

        // Advance replay time
        let effective_dt = dt * self.speed.effective_speed();
        self.replay_time += effective_dt;
        self.replay_time = self.replay_time.clamp(0.0, self.total_duration);

        // Update active camera
        match self.mode {
            CameraMode::Free => {
                self.free_camera.update(dt, input);
            }
            CameraMode::Follow => {
                self.follow_camera
                    .update(dt, self.target_position, self.target_forward);
            }
            CameraMode::Orbit => {
                self.orbit_camera
                    .update(dt, self.target_position, input);
            }
            CameraMode::Directed => {
                // Directed camera is evaluated from the sequence
            }
            CameraMode::Fixed | CameraMode::FirstPerson => {
                // These modes are simpler and just use the target data
            }
        }
    }

    /// Get the current camera transform.
    pub fn current_transform(&self) -> CameraTransform {
        match self.mode {
            CameraMode::Free => self.free_camera.transform,
            CameraMode::Follow => self.follow_camera.transform,
            CameraMode::Orbit => self.orbit_camera.transform,
            CameraMode::Directed => {
                if let Some(ref seq) = self.directed_sequence {
                    seq.evaluate(self.replay_time)
                        .unwrap_or_else(CameraTransform::default)
                } else {
                    CameraTransform::default()
                }
            }
            CameraMode::Fixed => self.free_camera.transform,
            CameraMode::FirstPerson => {
                CameraTransform {
                    position: self.target_position + Vec3::UP.scale(1.7),
                    rotation: Quat::from_euler(
                        self.target_forward.x.atan2(-self.target_forward.z),
                        0.0,
                        0.0,
                    ),
                    ..CameraTransform::default()
                }
            }
        }
    }

    /// Switch camera mode.
    pub fn set_mode(&mut self, mode: CameraMode) {
        self.mode = mode;
    }

    /// Set the target entity data.
    pub fn set_target(&mut self, position: Vec3, forward: Vec3) {
        self.target_position = position;
        self.target_forward = forward;
    }

    /// Set the split-screen layout.
    pub fn set_split_layout(&mut self, layout: SplitLayout) {
        self.split_layout = layout;
        self.split_viewports = generate_split_viewports(layout);
    }

    /// Add a PiP viewport.
    pub fn add_pip(&mut self, pip: PipViewport) {
        self.pip_viewports.push(pip);
    }

    /// Remove all PiP viewports.
    pub fn clear_pips(&mut self) {
        self.pip_viewports.clear();
    }

    /// Seek to a specific time.
    pub fn seek(&mut self, time: f32) {
        self.replay_time = time.clamp(0.0, self.total_duration);
    }

    /// Get playback progress [0, 1].
    pub fn progress(&self) -> f32 {
        if self.total_duration > 0.0 {
            self.replay_time / self.total_duration
        } else {
            0.0
        }
    }
}

impl Default for ReplayCameraController {
    fn default() -> Self {
        Self::new(60.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_camera_movement() {
        let mut cam = FreeCamera::new();
        cam.set_position(Vec3::ZERO);
        let input = CameraInput {
            forward: true,
            ..Default::default()
        };
        cam.update(1.0, &input);
        // Should have moved forward
        assert!(cam.target_position.z < 0.0 || cam.target_position.z != 0.0);
    }

    #[test]
    fn test_orbit_camera_zoom() {
        let mut cam = OrbitCamera::new();
        cam.set_distance(5.0);
        assert!((cam.distance - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_orbit_camera_clamp() {
        let mut cam = OrbitCamera::new();
        cam.set_distance(0.5); // Below min
        assert!((cam.distance - cam.min_distance).abs() < 0.01);
    }

    #[test]
    fn test_playback_speed_ramping() {
        let mut speed = PlaybackSpeedController::new();
        speed.set_speed(0.25);
        // Ramp over several frames
        for _ in 0..100 {
            speed.update(0.016);
        }
        assert!((speed.current_speed - 0.25).abs() < 0.05);
    }

    #[test]
    fn test_playback_pause() {
        let mut speed = PlaybackSpeedController::new();
        speed.toggle_pause();
        assert_eq!(speed.effective_speed(), 0.0);
        speed.toggle_pause();
        assert!(speed.effective_speed() > 0.0);
    }

    #[test]
    fn test_playback_reverse() {
        let mut speed = PlaybackSpeedController::new();
        speed.set_speed_immediate(1.0);
        speed.toggle_reverse();
        assert!(speed.effective_speed() < 0.0);
    }

    #[test]
    fn test_playback_presets() {
        let mut speed = PlaybackSpeedController::new();
        speed.next_preset();
        assert!(speed.target_speed > 1.0);
        speed.prev_preset();
        assert!((speed.target_speed - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_easing_functions() {
        assert!((EasingFunction::Linear.evaluate(0.5) - 0.5).abs() < 0.01);
        assert!(EasingFunction::EaseIn.evaluate(0.5) < 0.5);
        assert!(EasingFunction::EaseOut.evaluate(0.5) > 0.5);
        assert!((EasingFunction::EaseInOut.evaluate(0.0)).abs() < 0.01);
        assert!((EasingFunction::EaseInOut.evaluate(1.0) - 1.0).abs() < 0.01);
        assert!((EasingFunction::SmootherStep.evaluate(0.0)).abs() < 0.01);
        assert!((EasingFunction::SmootherStep.evaluate(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_directed_camera_sequence() {
        let mut seq = DirectedCameraSequence::new("intro");
        seq.add_shot(CameraShot {
            name: "wide".into(),
            start_time: 0.0,
            end_time: 5.0,
            start_transform: CameraTransform {
                position: Vec3::new(0.0, 10.0, 20.0),
                ..Default::default()
            },
            end_transform: CameraTransform {
                position: Vec3::new(0.0, 5.0, 10.0),
                ..Default::default()
            },
            easing: EasingFunction::EaseInOut,
            focus_entity: None,
        });
        let result = seq.evaluate(2.5).unwrap();
        // Should be halfway between the transforms
        assert!(result.position.y < 10.0 && result.position.y > 5.0);
    }

    #[test]
    fn test_pip_viewport() {
        let pip = PipViewport::bottom_right();
        assert!(pip.visible);
        assert!((pip.rect[0] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_split_layout_single() {
        let vps = generate_split_viewports(SplitLayout::Single);
        assert_eq!(vps.len(), 1);
        assert!((vps[0].rect[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_split_layout_quad() {
        let vps = generate_split_viewports(SplitLayout::Quad);
        assert_eq!(vps.len(), 4);
    }

    #[test]
    fn test_camera_controller_seek() {
        let mut ctrl = ReplayCameraController::new(120.0);
        ctrl.seek(60.0);
        assert!((ctrl.replay_time - 60.0).abs() < 0.01);
        assert!((ctrl.progress() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_camera_controller_mode_switch() {
        let mut ctrl = ReplayCameraController::new(60.0);
        ctrl.set_mode(CameraMode::Orbit);
        assert_eq!(ctrl.mode, CameraMode::Orbit);
    }

    #[test]
    fn test_camera_transform_lerp() {
        let a = CameraTransform {
            position: Vec3::ZERO,
            fov: 60.0,
            ..Default::default()
        };
        let b = CameraTransform {
            position: Vec3::new(10.0, 0.0, 0.0),
            fov: 90.0,
            ..Default::default()
        };
        let mid = a.lerp(b, 0.5);
        assert!((mid.position.x - 5.0).abs() < 0.01);
        assert!((mid.fov - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_camera_mode_display() {
        assert_eq!(CameraMode::Free.to_string(), "Free");
        assert_eq!(CameraMode::Follow.to_string(), "Follow");
        assert_eq!(CameraMode::Orbit.to_string(), "Orbit");
    }

    #[test]
    fn test_slow_motion() {
        let mut speed = PlaybackSpeedController::new();
        speed.enter_slow_motion(0.1);
        for _ in 0..200 {
            speed.update(0.016);
        }
        assert!((speed.current_speed - 0.1).abs() < 0.05);
        speed.exit_slow_motion();
        for _ in 0..200 {
            speed.update(0.016);
        }
        assert!((speed.current_speed - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_follow_camera_snap() {
        let mut cam = FollowCamera::new();
        cam.snap_to(Vec3::new(10.0, 0.0, 10.0), Vec3::FORWARD);
        assert!(cam.transform.position.distance(Vec3::new(10.0, 0.0, 10.0)) < 20.0);
    }
}
