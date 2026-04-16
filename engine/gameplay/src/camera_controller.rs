//! Camera controller systems for common game camera modes.
//!
//! Provides ready-to-use camera controllers for third-person, first-person,
//! and top-down perspectives, plus a composable [`CameraRig`] that can blend
//! and stack multiple behaviors.
//!
//! # Overview
//!
//! | Controller            | Use case                                    |
//! |-----------------------|---------------------------------------------|
//! | [`ThirdPersonCamera`] | Over-the-shoulder or orbiting camera        |
//! | [`FirstPersonCamera`] | FPS-style mouse look with head bob          |
//! | [`TopDownCamera`]     | RTS/ARPG fixed-angle overhead view          |
//! | [`CameraRig`]         | Composable stack of camera behaviors        |

use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Smooth-damp implementation (critically damped spring).
///
/// Returns the new value and updated velocity. Works like Unity's
/// `Mathf.SmoothDamp`.
fn smooth_damp(
    current: f32,
    target: f32,
    velocity: &mut f32,
    smooth_time: f32,
    dt: f32,
) -> f32 {
    let smooth = smooth_time.max(0.0001);
    let omega = 2.0 / smooth;
    let x = omega * dt;
    let exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x);
    let delta = current - target;
    let temp = (*velocity + omega * delta) * dt;
    *velocity = (*velocity - omega * temp) * exp;
    target + (delta + temp) * exp
}

/// Smooth-damp for Vec3.
fn smooth_damp_vec3(
    current: Vec3,
    target: Vec3,
    velocity: &mut Vec3,
    smooth_time: f32,
    dt: f32,
) -> Vec3 {
    Vec3::new(
        smooth_damp(current.x, target.x, &mut velocity.x, smooth_time, dt),
        smooth_damp(current.y, target.y, &mut velocity.y, smooth_time, dt),
        smooth_damp(current.z, target.z, &mut velocity.z, smooth_time, dt),
    )
}

/// Clamp an angle to the range [-pi, pi].
fn normalize_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

// ---------------------------------------------------------------------------
// Camera output
// ---------------------------------------------------------------------------

/// The computed camera transform, output by all camera controllers.
#[derive(Debug, Clone, Copy)]
pub struct CameraTransform {
    /// World-space position of the camera.
    pub position: Vec3,
    /// Forward direction (unit vector).
    pub forward: Vec3,
    /// Up direction (unit vector).
    pub up: Vec3,
    /// Field of view in radians (vertical).
    pub fov: f32,
}

impl CameraTransform {
    /// Compute the view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward, self.up)
    }

    /// The right direction.
    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up).normalize()
    }
}

impl Default for CameraTransform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov: std::f32::consts::FRAC_PI_4 * 1.5, // ~67.5 degrees
        }
    }
}

// ---------------------------------------------------------------------------
// Raycast helper (for collision avoidance)
// ---------------------------------------------------------------------------

/// Minimal raycast result for camera collision avoidance.
#[derive(Debug, Clone, Copy)]
pub struct CameraRayHit {
    /// Distance from the ray origin to the hit.
    pub distance: f32,
    /// Hit point in world space.
    pub point: Vec3,
    /// Surface normal at the hit.
    pub normal: Vec3,
}

/// Trait for camera collision queries. Implement this to let cameras avoid
/// clipping through geometry.
pub trait CameraCollisionProvider {
    /// Cast a ray and return the nearest hit, if any.
    fn raycast(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<CameraRayHit>;

    /// Cast a sphere (or thick ray) for softer collision avoidance.
    fn sphere_cast(
        &self,
        origin: Vec3,
        direction: Vec3,
        radius: f32,
        max_distance: f32,
    ) -> Option<CameraRayHit>;
}

// ---------------------------------------------------------------------------
// Third-person camera
// ---------------------------------------------------------------------------

/// Configuration for the third-person orbit camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThirdPersonConfig {
    /// Default distance from the target.
    pub default_distance: f32,
    /// Minimum orbit distance (closest zoom).
    pub min_distance: f32,
    /// Maximum orbit distance (farthest zoom).
    pub max_distance: f32,
    /// Minimum pitch angle in radians (looking up limit).
    pub min_pitch: f32,
    /// Maximum pitch angle in radians (looking down limit).
    pub max_pitch: f32,
    /// Horizontal rotation sensitivity (radians per pixel).
    pub yaw_sensitivity: f32,
    /// Vertical rotation sensitivity (radians per pixel).
    pub pitch_sensitivity: f32,
    /// Zoom speed (units per scroll step).
    pub zoom_speed: f32,
    /// Smooth follow time (seconds). Lower = snappier.
    pub follow_smooth_time: f32,
    /// Camera rotation smooth time.
    pub rotation_smooth_time: f32,
    /// Offset from the target's origin (e.g., shoulder offset).
    pub target_offset: Vec3,
    /// Whether to auto-rotate behind the character when moving.
    pub auto_rotate: bool,
    /// Speed of auto-rotation (radians/sec).
    pub auto_rotate_speed: f32,
    /// Delay before auto-rotation starts (seconds of no input).
    pub auto_rotate_delay: f32,
    /// Camera collision sphere radius for avoidance.
    pub collision_radius: f32,
    /// Whether collision avoidance is enabled.
    pub collision_avoidance: bool,
    /// Field of view in radians.
    pub fov: f32,
}

impl Default for ThirdPersonConfig {
    fn default() -> Self {
        Self {
            default_distance: 5.0,
            min_distance: 1.5,
            max_distance: 15.0,
            min_pitch: -1.2,           // ~ -70 degrees
            max_pitch: 1.2,            // ~ +70 degrees
            yaw_sensitivity: 0.003,
            pitch_sensitivity: 0.003,
            zoom_speed: 1.0,
            follow_smooth_time: 0.1,
            rotation_smooth_time: 0.05,
            target_offset: Vec3::new(0.0, 1.5, 0.0),
            auto_rotate: false,
            auto_rotate_speed: 2.0,
            auto_rotate_delay: 2.0,
            collision_radius: 0.2,
            collision_avoidance: true,
            fov: std::f32::consts::FRAC_PI_4 * 1.5,
        }
    }
}

/// Input for the third-person camera.
#[derive(Debug, Clone, Default)]
pub struct ThirdPersonInput {
    /// Mouse delta X (pixels).
    pub mouse_delta_x: f32,
    /// Mouse delta Y (pixels).
    pub mouse_delta_y: f32,
    /// Scroll wheel delta (positive = zoom in).
    pub scroll_delta: f32,
    /// Target position to follow (character world position).
    pub target_position: Vec3,
    /// Target's forward direction for auto-rotation.
    pub target_forward: Vec3,
    /// Whether the player is providing mouse input (disables auto-rotate timer).
    pub has_mouse_input: bool,
}

/// Third-person orbit camera.
///
/// Orbits around a target point (usually the player character) with configurable
/// distance, pitch clamping, zoom, collision avoidance, and auto-rotation.
pub struct ThirdPersonCamera {
    /// Configuration.
    pub config: ThirdPersonConfig,
    /// Current yaw (horizontal angle in radians).
    yaw: f32,
    /// Current pitch (vertical angle in radians).
    pitch: f32,
    /// Current orbit distance.
    distance: f32,
    /// Target distance (for smooth zoom).
    target_distance: f32,
    /// Smoothed follow position.
    smoothed_target: Vec3,
    /// Velocity for smooth-damp position.
    follow_velocity: Vec3,
    /// Time since last mouse input (for auto-rotate delay).
    time_since_input: f32,
    /// Actual camera distance after collision avoidance.
    effective_distance: f32,
}

impl ThirdPersonCamera {
    /// Create a new third-person camera with the given config.
    pub fn new(config: ThirdPersonConfig) -> Self {
        let distance = config.default_distance;
        Self {
            config,
            yaw: 0.0,
            pitch: 0.2, // Slight downward angle by default
            distance,
            target_distance: distance,
            smoothed_target: Vec3::ZERO,
            follow_velocity: Vec3::ZERO,
            time_since_input: 0.0,
            effective_distance: distance,
        }
    }

    /// Update the camera and return the computed transform.
    pub fn update(
        &mut self,
        input: &ThirdPersonInput,
        dt: f32,
        collision: Option<&dyn CameraCollisionProvider>,
    ) -> CameraTransform {
        // Update rotation from mouse input.
        if input.has_mouse_input || input.mouse_delta_x.abs() > 0.1 || input.mouse_delta_y.abs() > 0.1 {
            self.yaw -= input.mouse_delta_x * self.config.yaw_sensitivity;
            self.pitch += input.mouse_delta_y * self.config.pitch_sensitivity;
            self.yaw = normalize_angle(self.yaw);
            self.pitch = self.pitch.clamp(self.config.min_pitch, self.config.max_pitch);
            self.time_since_input = 0.0;
        } else {
            self.time_since_input += dt;
        }

        // Zoom.
        if input.scroll_delta.abs() > 0.01 {
            self.target_distance -= input.scroll_delta * self.config.zoom_speed;
            self.target_distance = self.target_distance.clamp(
                self.config.min_distance,
                self.config.max_distance,
            );
        }

        // Smooth zoom.
        self.distance += (self.target_distance - self.distance) * (1.0 - (-10.0 * dt).exp());

        // Auto-rotate behind character.
        if self.config.auto_rotate
            && self.time_since_input > self.config.auto_rotate_delay
            && input.target_forward.length_squared() > 0.01
        {
            let target_yaw = input.target_forward.z.atan2(input.target_forward.x);
            let diff = normalize_angle(target_yaw - self.yaw);
            self.yaw += diff * self.config.auto_rotate_speed * dt;
            self.yaw = normalize_angle(self.yaw);
        }

        // Smooth follow the target position.
        let target_with_offset = input.target_position + self.config.target_offset;
        self.smoothed_target = smooth_damp_vec3(
            self.smoothed_target,
            target_with_offset,
            &mut self.follow_velocity,
            self.config.follow_smooth_time,
            dt,
        );

        // Compute camera position on the orbit sphere.
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let offset = Vec3::new(
            cos_pitch * sin_yaw,
            sin_pitch,
            cos_pitch * cos_yaw,
        ) * self.distance;

        let ideal_position = self.smoothed_target + offset;

        // Collision avoidance: pull camera forward if geometry is between
        // target and camera.
        self.effective_distance = self.distance;
        let final_position = if self.config.collision_avoidance {
            self.resolve_collision(
                self.smoothed_target,
                ideal_position,
                collision,
            )
        } else {
            ideal_position
        };

        // Compute forward direction (from camera to target).
        let forward = (self.smoothed_target - final_position).normalize_or_zero();

        CameraTransform {
            position: final_position,
            forward,
            up: Vec3::Y,
            fov: self.config.fov,
        }
    }

    /// Collision avoidance: cast from target to ideal camera position and
    /// pull the camera forward if geometry blocks the view.
    fn resolve_collision(
        &mut self,
        target: Vec3,
        ideal_position: Vec3,
        collision: Option<&dyn CameraCollisionProvider>,
    ) -> Vec3 {
        let Some(collision) = collision else {
            return ideal_position;
        };

        let to_camera = ideal_position - target;
        let distance = to_camera.length();
        if distance < 0.01 {
            return ideal_position;
        }

        let direction = to_camera / distance;

        // Use sphere cast for smoother avoidance.
        if let Some(hit) = collision.sphere_cast(
            target,
            direction,
            self.config.collision_radius,
            distance,
        ) {
            let safe_distance = (hit.distance - self.config.collision_radius * 2.0)
                .max(self.config.min_distance * 0.5);
            self.effective_distance = safe_distance;
            target + direction * safe_distance
        } else {
            self.effective_distance = distance;
            ideal_position
        }
    }

    /// Current yaw angle.
    #[inline]
    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    /// Current pitch angle.
    #[inline]
    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Effective distance after collision avoidance.
    #[inline]
    pub fn effective_distance(&self) -> f32 {
        self.effective_distance
    }

    /// Set yaw directly (e.g., to match character facing after a cutscene).
    pub fn set_yaw(&mut self, yaw: f32) {
        self.yaw = normalize_angle(yaw);
    }

    /// Set pitch directly.
    pub fn set_pitch(&mut self, pitch: f32) {
        self.pitch = pitch.clamp(self.config.min_pitch, self.config.max_pitch);
    }

    /// Reset the camera to look at the target from the given yaw.
    pub fn reset(&mut self, target_position: Vec3, yaw: f32) {
        self.yaw = normalize_angle(yaw);
        self.pitch = 0.2;
        self.distance = self.config.default_distance;
        self.target_distance = self.distance;
        self.smoothed_target = target_position + self.config.target_offset;
        self.follow_velocity = Vec3::ZERO;
        self.time_since_input = 0.0;
    }
}

// ---------------------------------------------------------------------------
// First-person camera
// ---------------------------------------------------------------------------

/// Configuration for the first-person camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstPersonConfig {
    /// Mouse sensitivity (radians per pixel).
    pub sensitivity: f32,
    /// Minimum pitch (looking down limit) in radians.
    pub min_pitch: f32,
    /// Maximum pitch (looking up limit) in radians.
    pub max_pitch: f32,
    /// Base field of view in radians.
    pub base_fov: f32,
    /// FOV when sprinting (wider for speed feel).
    pub sprint_fov: f32,
    /// FOV interpolation speed.
    pub fov_lerp_speed: f32,
    /// Whether head bob is enabled.
    pub head_bob_enabled: bool,
    /// Head bob frequency (cycles per second at normal walk speed).
    pub head_bob_frequency: f32,
    /// Head bob vertical amplitude.
    pub head_bob_amplitude_y: f32,
    /// Head bob horizontal amplitude.
    pub head_bob_amplitude_x: f32,
    /// Eye height offset from character position.
    pub eye_height: f32,
    /// Smooth rotation (slight lag for cinematic feel). 0 = instant.
    pub rotation_smooth_time: f32,
}

impl Default for FirstPersonConfig {
    fn default() -> Self {
        Self {
            sensitivity: 0.002,
            min_pitch: -1.4,           // ~ -80 degrees
            max_pitch: 1.4,            // ~ +80 degrees
            base_fov: std::f32::consts::FRAC_PI_4 * 1.5,
            sprint_fov: std::f32::consts::FRAC_PI_4 * 1.8,
            fov_lerp_speed: 6.0,
            head_bob_enabled: true,
            head_bob_frequency: 1.8,
            head_bob_amplitude_y: 0.04,
            head_bob_amplitude_x: 0.02,
            eye_height: 1.6,
            rotation_smooth_time: 0.0,
        }
    }
}

/// Input for the first-person camera.
#[derive(Debug, Clone, Default)]
pub struct FirstPersonInput {
    /// Mouse delta X (pixels).
    pub mouse_delta_x: f32,
    /// Mouse delta Y (pixels).
    pub mouse_delta_y: f32,
    /// Character position (feet).
    pub character_position: Vec3,
    /// Whether the character is sprinting (for FOV effect).
    pub is_sprinting: bool,
    /// Whether the character is grounded (for head bob).
    pub is_grounded: bool,
    /// Character horizontal speed (for head bob intensity).
    pub horizontal_speed: f32,
}

/// First-person camera with mouse look, pitch clamping, head bob, and FOV effects.
pub struct FirstPersonCamera {
    /// Configuration.
    pub config: FirstPersonConfig,
    /// Current yaw.
    yaw: f32,
    /// Current pitch.
    pitch: f32,
    /// Smoothed yaw (if rotation smoothing is enabled).
    smoothed_yaw: f32,
    /// Smoothed pitch.
    smoothed_pitch: f32,
    /// Current FOV (interpolated).
    current_fov: f32,
    /// Head bob phase accumulator.
    bob_phase: f32,
    /// Yaw smooth velocity.
    yaw_velocity: f32,
    /// Pitch smooth velocity.
    pitch_velocity: f32,
}

impl FirstPersonCamera {
    /// Create a new first-person camera.
    pub fn new(config: FirstPersonConfig) -> Self {
        let fov = config.base_fov;
        Self {
            config,
            yaw: 0.0,
            pitch: 0.0,
            smoothed_yaw: 0.0,
            smoothed_pitch: 0.0,
            current_fov: fov,
            bob_phase: 0.0,
            yaw_velocity: 0.0,
            pitch_velocity: 0.0,
        }
    }

    /// Update the camera and return the computed transform.
    pub fn update(&mut self, input: &FirstPersonInput, dt: f32) -> CameraTransform {
        // Apply mouse look.
        self.yaw -= input.mouse_delta_x * self.config.sensitivity;
        self.pitch -= input.mouse_delta_y * self.config.sensitivity;
        self.yaw = normalize_angle(self.yaw);
        self.pitch = self.pitch.clamp(self.config.min_pitch, self.config.max_pitch);

        // Smooth rotation.
        if self.config.rotation_smooth_time > 0.001 {
            self.smoothed_yaw = smooth_damp(
                self.smoothed_yaw,
                self.yaw,
                &mut self.yaw_velocity,
                self.config.rotation_smooth_time,
                dt,
            );
            self.smoothed_pitch = smooth_damp(
                self.smoothed_pitch,
                self.pitch,
                &mut self.pitch_velocity,
                self.config.rotation_smooth_time,
                dt,
            );
        } else {
            self.smoothed_yaw = self.yaw;
            self.smoothed_pitch = self.pitch;
        }

        // Compute forward direction from yaw and pitch.
        let (sin_yaw, cos_yaw) = self.smoothed_yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.smoothed_pitch.sin_cos();
        let forward = Vec3::new(
            cos_pitch * sin_yaw,
            sin_pitch,
            cos_pitch * cos_yaw,
        )
        .normalize();

        // Compute right vector (for head bob).
        let right = forward.cross(Vec3::Y).normalize_or_zero();

        // Eye position.
        let mut eye_pos = input.character_position + Vec3::new(0.0, self.config.eye_height, 0.0);

        // Head bob.
        if self.config.head_bob_enabled && input.is_grounded && input.horizontal_speed > 0.5 {
            let speed_factor = (input.horizontal_speed / 6.0).min(1.5);
            self.bob_phase += dt * self.config.head_bob_frequency * std::f32::consts::TAU * speed_factor;

            let bob_y = self.bob_phase.sin() * self.config.head_bob_amplitude_y * speed_factor;
            let bob_x =
                (self.bob_phase * 0.5).sin() * self.config.head_bob_amplitude_x * speed_factor;

            eye_pos.y += bob_y;
            eye_pos += right * bob_x;
        } else {
            // Gradually reset bob phase to avoid jarring snap.
            self.bob_phase *= (1.0 - 5.0 * dt).max(0.0);
        }

        // FOV interpolation for sprint.
        let target_fov = if input.is_sprinting {
            self.config.sprint_fov
        } else {
            self.config.base_fov
        };
        let fov_t = 1.0 - (-self.config.fov_lerp_speed * dt).exp();
        self.current_fov += (target_fov - self.current_fov) * fov_t;

        CameraTransform {
            position: eye_pos,
            forward,
            up: Vec3::Y,
            fov: self.current_fov,
        }
    }

    /// Current yaw angle.
    #[inline]
    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    /// Current pitch angle.
    #[inline]
    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Set yaw directly.
    pub fn set_yaw(&mut self, yaw: f32) {
        self.yaw = normalize_angle(yaw);
        self.smoothed_yaw = self.yaw;
    }

    /// Set pitch directly.
    pub fn set_pitch(&mut self, pitch: f32) {
        self.pitch = pitch.clamp(self.config.min_pitch, self.config.max_pitch);
        self.smoothed_pitch = self.pitch;
    }

    /// Reset the camera orientation.
    pub fn reset(&mut self, yaw: f32, pitch: f32) {
        self.yaw = normalize_angle(yaw);
        self.pitch = pitch.clamp(self.config.min_pitch, self.config.max_pitch);
        self.smoothed_yaw = self.yaw;
        self.smoothed_pitch = self.pitch;
        self.current_fov = self.config.base_fov;
        self.bob_phase = 0.0;
        self.yaw_velocity = 0.0;
        self.pitch_velocity = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Top-down camera
// ---------------------------------------------------------------------------

/// Configuration for the top-down camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopDownConfig {
    /// Camera height above the target plane.
    pub height: f32,
    /// Fixed viewing angle from vertical in radians (0 = straight down).
    pub angle: f32,
    /// Minimum zoom height.
    pub min_height: f32,
    /// Maximum zoom height.
    pub max_height: f32,
    /// Zoom speed (units per scroll step).
    pub zoom_speed: f32,
    /// Edge scroll speed (units/sec). 0 to disable.
    pub edge_scroll_speed: f32,
    /// Edge scroll margin as fraction of screen width (e.g., 0.02 = 2%).
    pub edge_scroll_margin: f32,
    /// Keyboard pan speed (units/sec).
    pub pan_speed: f32,
    /// Smooth follow time.
    pub follow_smooth_time: f32,
    /// Whether to follow a target or free-pan.
    pub follow_target: bool,
    /// Fixed rotation around Y axis (radians).
    pub rotation: f32,
    /// Field of view.
    pub fov: f32,
    /// Whether to use orthographic projection.
    pub orthographic: bool,
    /// Orthographic size (half-height of the visible area).
    pub ortho_size: f32,
}

impl Default for TopDownConfig {
    fn default() -> Self {
        Self {
            height: 15.0,
            angle: 1.0, // ~57 degrees from vertical
            min_height: 5.0,
            max_height: 30.0,
            zoom_speed: 2.0,
            edge_scroll_speed: 15.0,
            edge_scroll_margin: 0.02,
            pan_speed: 10.0,
            follow_smooth_time: 0.15,
            follow_target: true,
            rotation: 0.0,
            fov: std::f32::consts::FRAC_PI_4,
            orthographic: false,
            ortho_size: 10.0,
        }
    }
}

/// Input for the top-down camera.
#[derive(Debug, Clone, Default)]
pub struct TopDownInput {
    /// Target position to follow (if follow is enabled).
    pub target_position: Vec3,
    /// Keyboard pan direction (normalized).
    pub pan_direction: Vec2,
    /// Scroll wheel delta (positive = zoom in).
    pub scroll_delta: f32,
    /// Mouse position in normalized screen coordinates (0..1).
    pub mouse_screen_position: Vec2,
    /// Screen width in pixels (for edge scroll calculation).
    pub screen_width: f32,
    /// Screen height in pixels.
    pub screen_height: f32,
}

/// Top-down camera with fixed angle, scroll zoom, and edge-scroll panning.
pub struct TopDownCamera {
    /// Configuration.
    pub config: TopDownConfig,
    /// Current look-at position on the ground.
    look_at: Vec3,
    /// Target look-at (for smoothing).
    target_look_at: Vec3,
    /// Follow velocity for smooth-damp.
    follow_velocity: Vec3,
    /// Current height.
    current_height: f32,
    /// Target height (for smooth zoom).
    target_height: f32,
}

impl TopDownCamera {
    /// Create a new top-down camera.
    pub fn new(config: TopDownConfig) -> Self {
        let height = config.height;
        Self {
            config,
            look_at: Vec3::ZERO,
            target_look_at: Vec3::ZERO,
            follow_velocity: Vec3::ZERO,
            current_height: height,
            target_height: height,
        }
    }

    /// Update the camera and return the computed transform.
    pub fn update(&mut self, input: &TopDownInput, dt: f32) -> CameraTransform {
        // Update target look-at.
        if self.config.follow_target {
            self.target_look_at = input.target_position;
        }

        // Keyboard panning.
        if input.pan_direction.length_squared() > 0.001 {
            let (sin_rot, cos_rot) = self.config.rotation.sin_cos();
            let pan_world = Vec3::new(
                input.pan_direction.x * cos_rot - input.pan_direction.y * sin_rot,
                0.0,
                input.pan_direction.x * sin_rot + input.pan_direction.y * cos_rot,
            );
            self.target_look_at += pan_world * self.config.pan_speed * dt;
        }

        // Edge scrolling.
        if self.config.edge_scroll_speed > 0.0
            && input.screen_width > 0.0
            && input.screen_height > 0.0
        {
            let margin = self.config.edge_scroll_margin;
            let mut edge_pan = Vec2::ZERO;

            if input.mouse_screen_position.x < margin {
                edge_pan.x -= 1.0;
            } else if input.mouse_screen_position.x > 1.0 - margin {
                edge_pan.x += 1.0;
            }
            if input.mouse_screen_position.y < margin {
                edge_pan.y += 1.0; // Up on screen = forward in world
            } else if input.mouse_screen_position.y > 1.0 - margin {
                edge_pan.y -= 1.0;
            }

            if edge_pan.length_squared() > 0.0 {
                let (sin_rot, cos_rot) = self.config.rotation.sin_cos();
                let pan_world = Vec3::new(
                    edge_pan.x * cos_rot - edge_pan.y * sin_rot,
                    0.0,
                    edge_pan.x * sin_rot + edge_pan.y * cos_rot,
                );
                self.target_look_at +=
                    pan_world.normalize_or_zero() * self.config.edge_scroll_speed * dt;
            }
        }

        // Zoom.
        if input.scroll_delta.abs() > 0.01 {
            self.target_height -= input.scroll_delta * self.config.zoom_speed;
            self.target_height = self.target_height.clamp(
                self.config.min_height,
                self.config.max_height,
            );
        }

        // Smooth zoom.
        self.current_height +=
            (self.target_height - self.current_height) * (1.0 - (-8.0 * dt).exp());

        // Smooth follow.
        self.look_at = smooth_damp_vec3(
            self.look_at,
            self.target_look_at,
            &mut self.follow_velocity,
            self.config.follow_smooth_time,
            dt,
        );

        // Compute camera position from angle and height.
        let (sin_angle, cos_angle) = self.config.angle.sin_cos();
        let (sin_rot, cos_rot) = self.config.rotation.sin_cos();

        let offset_forward = -sin_angle * self.current_height;
        let offset = Vec3::new(
            offset_forward * sin_rot,
            cos_angle * self.current_height,
            offset_forward * cos_rot,
        );

        let cam_pos = self.look_at + offset;
        let forward = (self.look_at - cam_pos).normalize_or_zero();

        CameraTransform {
            position: cam_pos,
            forward,
            up: Vec3::Y,
            fov: self.config.fov,
        }
    }

    /// Current look-at position.
    #[inline]
    pub fn look_at(&self) -> Vec3 {
        self.look_at
    }

    /// Set the look-at position immediately (no smoothing).
    pub fn set_look_at(&mut self, position: Vec3) {
        self.look_at = position;
        self.target_look_at = position;
        self.follow_velocity = Vec3::ZERO;
    }

    /// Set zoom height.
    pub fn set_height(&mut self, height: f32) {
        let h = height.clamp(self.config.min_height, self.config.max_height);
        self.current_height = h;
        self.target_height = h;
    }
}

// ---------------------------------------------------------------------------
// Camera rig (composable behavior stack)
// ---------------------------------------------------------------------------

/// A named camera behavior that can be stacked in a [`CameraRig`].
pub trait CameraBehavior: Send + Sync {
    /// Name of this behavior (for debugging).
    fn name(&self) -> &str;

    /// Process the camera transform. Each behavior in the stack receives
    /// the output of the previous behavior and can modify it.
    fn process(&mut self, transform: CameraTransform, dt: f32) -> CameraTransform;

    /// Weight of this behavior (0..1). Used for blending.
    fn weight(&self) -> f32 {
        1.0
    }
}

/// Camera shake behavior -- adds procedural shake (trauma-based).
pub struct CameraShake {
    /// Current trauma value (0..1). Decays over time.
    pub trauma: f32,
    /// Trauma decay rate (per second).
    pub decay_rate: f32,
    /// Maximum translational offset.
    pub max_offset: f32,
    /// Maximum rotational offset (radians).
    pub max_rotation: f32,
    /// Shake frequency.
    pub frequency: f32,
    /// Internal phase accumulator.
    phase: f32,
}

impl CameraShake {
    /// Create a new camera shake behavior.
    pub fn new() -> Self {
        Self {
            trauma: 0.0,
            decay_rate: 1.5,
            max_offset: 0.3,
            max_rotation: 0.05,
            frequency: 15.0,
            phase: 0.0,
        }
    }

    /// Add trauma (clamped to 0..1). Use ~0.3 for small hits, ~0.7 for explosions.
    pub fn add_trauma(&mut self, amount: f32) {
        self.trauma = (self.trauma + amount).min(1.0);
    }
}

impl Default for CameraShake {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraBehavior for CameraShake {
    fn name(&self) -> &str {
        "CameraShake"
    }

    fn process(&mut self, mut transform: CameraTransform, dt: f32) -> CameraTransform {
        if self.trauma <= 0.001 {
            return transform;
        }

        self.phase += dt * self.frequency;

        // Shake intensity is trauma squared for a nice feel curve.
        let shake = self.trauma * self.trauma;

        // Use simple sine waves at different frequencies for pseudo-random shake.
        let offset_x = (self.phase * 1.0).sin() * self.max_offset * shake;
        let offset_y = (self.phase * 1.3 + 0.7).sin() * self.max_offset * shake;
        let offset_z = (self.phase * 0.9 + 1.3).sin() * self.max_offset * shake * 0.5;

        let right = transform.forward.cross(transform.up).normalize_or_zero();
        transform.position += right * offset_x + transform.up * offset_y + transform.forward * offset_z;

        // Decay trauma.
        self.trauma = (self.trauma - self.decay_rate * dt).max(0.0);

        transform
    }

    fn weight(&self) -> f32 {
        1.0
    }
}

/// Smooth follow behavior -- adds smooth following with configurable lag.
pub struct SmoothFollow {
    /// Smooth time.
    pub smooth_time: f32,
    /// Target position to follow.
    pub target: Vec3,
    /// Internal velocity.
    velocity: Vec3,
}

impl SmoothFollow {
    /// Create a new smooth follow behavior.
    pub fn new(smooth_time: f32) -> Self {
        Self {
            smooth_time,
            target: Vec3::ZERO,
            velocity: Vec3::ZERO,
        }
    }
}

impl CameraBehavior for SmoothFollow {
    fn name(&self) -> &str {
        "SmoothFollow"
    }

    fn process(&mut self, mut transform: CameraTransform, dt: f32) -> CameraTransform {
        transform.position = smooth_damp_vec3(
            transform.position,
            self.target,
            &mut self.velocity,
            self.smooth_time,
            dt,
        );
        transform
    }
}

/// Composable camera rig that chains multiple behaviors.
///
/// Each behavior processes the camera transform in order, allowing
/// effects to be layered (e.g., follow -> shake -> smooth).
pub struct CameraRig {
    /// Stack of behaviors applied in order.
    behaviors: Vec<Box<dyn CameraBehavior>>,
    /// Base transform before behaviors are applied.
    base_transform: CameraTransform,
}

impl CameraRig {
    /// Create a new empty camera rig.
    pub fn new() -> Self {
        Self {
            behaviors: Vec::new(),
            base_transform: CameraTransform::default(),
        }
    }

    /// Add a behavior to the stack.
    pub fn add_behavior(&mut self, behavior: Box<dyn CameraBehavior>) {
        self.behaviors.push(behavior);
    }

    /// Remove a behavior by name.
    pub fn remove_behavior(&mut self, name: &str) -> bool {
        if let Some(idx) = self.behaviors.iter().position(|b| b.name() == name) {
            self.behaviors.remove(idx);
            true
        } else {
            false
        }
    }

    /// Set the base transform (input to the behavior chain).
    pub fn set_base_transform(&mut self, transform: CameraTransform) {
        self.base_transform = transform;
    }

    /// Process all behaviors and return the final transform.
    pub fn update(&mut self, dt: f32) -> CameraTransform {
        let mut transform = self.base_transform;
        for behavior in &mut self.behaviors {
            transform = behavior.process(transform, dt);
        }
        transform
    }

    /// Get a reference to a behavior by name.
    pub fn get_behavior(&self, name: &str) -> Option<&dyn CameraBehavior> {
        self.behaviors
            .iter()
            .find(|b| b.name() == name)
            .map(|b| b.as_ref())
    }

    /// Get a mutable reference to a behavior by name.
    pub fn get_behavior_mut(&mut self, name: &str) -> Option<&mut (dyn CameraBehavior + 'static)> {
        for b in &mut self.behaviors {
            if b.name() == name {
                return Some(&mut **b);
            }
        }
        None
    }

    /// Number of behaviors in the stack.
    pub fn behavior_count(&self) -> usize {
        self.behaviors.len()
    }
}

impl Default for CameraRig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_angle_wraps() {
        assert!((normalize_angle(std::f32::consts::TAU + 0.1) - 0.1).abs() < 0.01);
        assert!((normalize_angle(-std::f32::consts::TAU - 0.1) + 0.1).abs() < 0.01);
    }

    #[test]
    fn camera_transform_view_matrix() {
        let ct = CameraTransform::default();
        let mat = ct.view_matrix();
        // The view matrix should be valid (non-zero determinant).
        assert!(mat.determinant().abs() > 0.001);
    }

    #[test]
    fn third_person_camera_basic() {
        let config = ThirdPersonConfig::default();
        let mut cam = ThirdPersonCamera::new(config);
        let input = ThirdPersonInput {
            target_position: Vec3::ZERO,
            target_forward: Vec3::Z,
            ..Default::default()
        };

        let transform = cam.update(&input, 1.0 / 60.0, None);
        // Camera should be behind and above the target.
        assert!(transform.position.y > 0.0, "Camera should be above target");
    }

    #[test]
    fn third_person_zoom() {
        let config = ThirdPersonConfig::default();
        let mut cam = ThirdPersonCamera::new(config);
        let mut input = ThirdPersonInput::default();

        // Zoom in.
        input.scroll_delta = 5.0;
        let _t1 = cam.update(&input, 1.0 / 60.0, None);

        input.scroll_delta = 0.0;
        // Run several frames to let zoom settle.
        for _ in 0..60 {
            cam.update(&input, 1.0 / 60.0, None);
        }

        assert!(
            cam.effective_distance() < cam.config.default_distance,
            "Should have zoomed in"
        );
    }

    #[test]
    fn first_person_pitch_clamp() {
        let config = FirstPersonConfig::default();
        let mut cam = FirstPersonCamera::new(config);

        // Push pitch beyond limits.
        let input = FirstPersonInput {
            mouse_delta_y: -10000.0,
            ..Default::default()
        };
        cam.update(&input, 1.0 / 60.0);

        assert!(
            cam.pitch() >= cam.config.min_pitch - 0.01,
            "Pitch {} should be >= min {}",
            cam.pitch(),
            cam.config.min_pitch
        );
    }

    #[test]
    fn first_person_fov_sprint() {
        let config = FirstPersonConfig::default();
        let mut cam = FirstPersonCamera::new(config);
        let input = FirstPersonInput {
            is_sprinting: true,
            ..Default::default()
        };

        // Run several frames.
        for _ in 0..120 {
            cam.update(&input, 1.0 / 60.0);
        }

        assert!(
            cam.current_fov > cam.config.base_fov,
            "FOV should increase during sprint"
        );
    }

    #[test]
    fn top_down_zoom() {
        let config = TopDownConfig::default();
        let mut cam = TopDownCamera::new(config);
        let initial_height = cam.current_height;

        let input = TopDownInput {
            scroll_delta: 3.0,
            ..Default::default()
        };
        for _ in 0..60 {
            cam.update(&input, 1.0 / 60.0);
        }

        assert!(
            cam.current_height < initial_height,
            "Should have zoomed in (lower height)"
        );
    }

    #[test]
    fn top_down_pan() {
        let config = TopDownConfig {
            follow_target: false,
            ..Default::default()
        };
        let mut cam = TopDownCamera::new(config);
        cam.set_look_at(Vec3::ZERO);

        let input = TopDownInput {
            pan_direction: Vec2::new(1.0, 0.0),
            ..Default::default()
        };
        for _ in 0..60 {
            cam.update(&input, 1.0 / 60.0);
        }

        assert!(
            cam.look_at().x > 0.0,
            "Look-at should have panned right: {}",
            cam.look_at().x
        );
    }

    #[test]
    fn camera_shake_decays() {
        let mut shake = CameraShake::new();
        shake.add_trauma(1.0);
        assert!((shake.trauma - 1.0).abs() < 0.01);

        let base = CameraTransform::default();
        for _ in 0..120 {
            shake.process(base, 1.0 / 60.0);
        }

        assert!(shake.trauma < 0.1, "Trauma should have decayed: {}", shake.trauma);
    }

    #[test]
    fn camera_rig_chains_behaviors() {
        let mut rig = CameraRig::new();
        let mut shake = CameraShake::new();
        shake.add_trauma(1.0);
        rig.add_behavior(Box::new(shake));

        rig.set_base_transform(CameraTransform::default());
        let result = rig.update(1.0 / 60.0);

        // The position should be offset from the base due to shake.
        let base = CameraTransform::default();
        let diff = (result.position - base.position).length();
        assert!(diff > 0.0, "Shake should offset the camera");
    }

    #[test]
    fn camera_rig_remove_behavior() {
        let mut rig = CameraRig::new();
        rig.add_behavior(Box::new(CameraShake::new()));
        assert_eq!(rig.behavior_count(), 1);
        assert!(rig.remove_behavior("CameraShake"));
        assert_eq!(rig.behavior_count(), 0);
        assert!(!rig.remove_behavior("NonExistent"));
    }

    #[test]
    fn smooth_damp_converges() {
        let mut vel = 0.0f32;
        let mut val = 0.0f32;
        for _ in 0..300 {
            val = smooth_damp(val, 10.0, &mut vel, 0.1, 1.0 / 60.0);
        }
        assert!(
            (val - 10.0).abs() < 0.01,
            "Should converge to 10: {}",
            val
        );
    }
}
