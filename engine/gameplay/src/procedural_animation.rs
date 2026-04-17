//! Procedural animation system for dynamic motion synthesis.
//!
//! Provides:
//! - **Look-at constraint**: head tracking toward a target
//! - **Breathing animation**: chest oscillation at configurable rate
//! - **Idle fidgets**: random small motions for natural idle poses
//! - **Wind-affected hints**: procedural hair/cloth motion from wind data
//! - **Procedural walk cycle**: IK-driven foot placement from velocity
//! - **Lean into turns**: body tilt based on angular velocity
//! - **Recoil animation**: weapon recoil spring with recovery
//! - **Eye blinking**: periodic blink animation
//! - **Tail physics**: simple chain simulation for tails/appendages
//! - **ECS integration**: `ProceduralAnimComponent`, `ProceduralAnimSystem`
//!
//! # Design
//!
//! Each procedural effect is a [`ProceduralLayer`] that modifies bone transforms.
//! Layers are composited on top of the base animation using additive blending.
//! The [`ProceduralAnimController`] manages all layers for one character.

use glam::{Vec3, Quat, Mat4};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default breathing rate (cycles per second).
pub const DEFAULT_BREATHING_RATE: f32 = 0.25;
/// Default breathing amplitude (meters of chest expansion).
pub const DEFAULT_BREATHING_AMPLITUDE: f32 = 0.015;
/// Default head tracking speed (degrees per second).
pub const DEFAULT_HEAD_TRACK_SPEED: f32 = 120.0;
/// Maximum head yaw angle (degrees).
pub const MAX_HEAD_YAW: f32 = 80.0;
/// Maximum head pitch angle (degrees).
pub const MAX_HEAD_PITCH: f32 = 45.0;
/// Default fidget interval (seconds between fidgets).
pub const DEFAULT_FIDGET_INTERVAL: f32 = 5.0;
/// Default fidget amplitude (degrees of rotation).
pub const DEFAULT_FIDGET_AMPLITUDE: f32 = 3.0;
/// Default lean angle per unit angular velocity.
pub const DEFAULT_LEAN_FACTOR: f32 = 5.0;
/// Maximum lean angle (degrees).
pub const MAX_LEAN_ANGLE: f32 = 15.0;
/// Default recoil recovery speed.
pub const DEFAULT_RECOIL_RECOVERY: f32 = 8.0;
/// Default blink interval (seconds).
pub const DEFAULT_BLINK_INTERVAL: f32 = 4.0;
/// Default blink duration (seconds).
pub const DEFAULT_BLINK_DURATION: f32 = 0.15;
/// Maximum number of procedural layers.
pub const MAX_LAYERS: usize = 16;
/// Pi constant.
const PI: f32 = std::f32::consts::PI;
/// Two PI.
const TWO_PI: f32 = PI * 2.0;
/// Small epsilon.
const EPSILON: f32 = 1e-6;
/// Degrees to radians conversion.
const DEG_TO_RAD: f32 = PI / 180.0;
/// Radians to degrees.
const RAD_TO_DEG: f32 = 180.0 / PI;
/// Default walk cycle frequency (steps per second at unit speed).
pub const DEFAULT_STEP_FREQUENCY: f32 = 2.0;
/// Default step height (meters).
pub const DEFAULT_STEP_HEIGHT: f32 = 0.1;
/// Default stride length (meters at walk speed).
pub const DEFAULT_STRIDE_LENGTH: f32 = 0.8;
/// Maximum number of tail segments.
pub const MAX_TAIL_SEGMENTS: usize = 12;
/// Tail damping.
pub const TAIL_DAMPING: f32 = 5.0;
/// Tail stiffness.
pub const TAIL_STIFFNESS: f32 = 20.0;

// ---------------------------------------------------------------------------
// BoneId
// ---------------------------------------------------------------------------

/// Identifier for a bone in the skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoneId(pub u32);

impl BoneId {
    pub fn new(id: u32) -> Self { Self(id) }
}

// ---------------------------------------------------------------------------
// BoneTransform
// ---------------------------------------------------------------------------

/// A transform to apply to a bone (additive or override).
#[derive(Debug, Clone, Copy)]
pub struct BoneTransform {
    /// Translation offset.
    pub translation: Vec3,
    /// Rotation offset (as quaternion).
    pub rotation: Quat,
    /// Scale multiplier.
    pub scale: Vec3,
    /// Blend weight (0..1).
    pub weight: f32,
}

impl Default for BoneTransform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            weight: 1.0,
        }
    }
}

impl BoneTransform {
    /// Create from translation only.
    pub fn from_translation(t: Vec3) -> Self {
        Self { translation: t, ..Default::default() }
    }

    /// Create from rotation only.
    pub fn from_rotation(r: Quat) -> Self {
        Self { rotation: r, ..Default::default() }
    }

    /// Blend this transform with an identity transform.
    pub fn blend(&self, weight: f32) -> BoneTransform {
        BoneTransform {
            translation: self.translation * weight,
            rotation: Quat::IDENTITY.slerp(self.rotation, weight),
            scale: Vec3::ONE.lerp(self.scale, weight),
            weight,
        }
    }

    /// Combine two additive transforms.
    pub fn add(&self, other: &BoneTransform) -> BoneTransform {
        BoneTransform {
            translation: self.translation + other.translation,
            rotation: self.rotation * other.rotation,
            scale: self.scale * other.scale,
            weight: (self.weight + other.weight).min(1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// LookAtConstraint
// ---------------------------------------------------------------------------

/// Head look-at constraint for tracking a target.
#[derive(Debug, Clone)]
pub struct LookAtConstraint {
    /// Target position to look at.
    pub target: Vec3,
    /// Whether the constraint is active.
    pub active: bool,
    /// Tracking speed (degrees/second).
    pub speed: f32,
    /// Maximum yaw angle.
    pub max_yaw: f32,
    /// Maximum pitch angle.
    pub max_pitch: f32,
    /// Head bone ID.
    pub head_bone: BoneId,
    /// Neck bone ID (optional, for distributing rotation).
    pub neck_bone: Option<BoneId>,
    /// Fraction of rotation applied to neck (rest goes to head).
    pub neck_fraction: f32,
    /// Current yaw (internal).
    current_yaw: f32,
    /// Current pitch (internal).
    current_pitch: f32,
    /// Blend weight.
    pub weight: f32,
}

impl LookAtConstraint {
    /// Create a new look-at constraint.
    pub fn new(head_bone: BoneId) -> Self {
        Self {
            target: Vec3::ZERO,
            active: false,
            speed: DEFAULT_HEAD_TRACK_SPEED,
            max_yaw: MAX_HEAD_YAW,
            max_pitch: MAX_HEAD_PITCH,
            head_bone,
            neck_bone: None,
            neck_fraction: 0.3,
            current_yaw: 0.0,
            current_pitch: 0.0,
            weight: 1.0,
        }
    }

    /// Set the look-at target.
    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
        self.active = true;
    }

    /// Clear the look-at target.
    pub fn clear_target(&mut self) {
        self.active = false;
    }

    /// Update the constraint and return bone transforms.
    pub fn update(
        &mut self,
        dt: f32,
        character_pos: Vec3,
        character_forward: Vec3,
    ) -> Vec<(BoneId, BoneTransform)> {
        let mut results = Vec::new();

        if !self.active {
            // Smoothly return to forward
            self.current_yaw = self.current_yaw * (1.0 - dt * 3.0);
            self.current_pitch = self.current_pitch * (1.0 - dt * 3.0);
        } else {
            // Calculate target yaw/pitch
            let to_target = self.target - character_pos;
            let to_target_flat = Vec3::new(to_target.x, 0.0, to_target.z);
            let forward_flat = Vec3::new(character_forward.x, 0.0, character_forward.z);

            let target_yaw = if to_target_flat.length() > EPSILON && forward_flat.length() > EPSILON {
                let cross = forward_flat.cross(to_target_flat).y;
                let dot = forward_flat.dot(to_target_flat);
                cross.atan2(dot) * RAD_TO_DEG
            } else {
                0.0
            };

            let target_pitch = if to_target.length() > EPSILON {
                let horizontal_dist = to_target_flat.length();
                (to_target.y / horizontal_dist.max(EPSILON)).atan() * RAD_TO_DEG
            } else {
                0.0
            };

            // Clamp to limits
            let clamped_yaw = target_yaw.clamp(-self.max_yaw, self.max_yaw);
            let clamped_pitch = target_pitch.clamp(-self.max_pitch, self.max_pitch);

            // Smooth interpolation
            let max_delta = self.speed * dt;
            let yaw_diff = clamped_yaw - self.current_yaw;
            let pitch_diff = clamped_pitch - self.current_pitch;

            self.current_yaw += yaw_diff.clamp(-max_delta, max_delta);
            self.current_pitch += pitch_diff.clamp(-max_delta, max_delta);
        }

        // Generate bone transforms
        let head_yaw = self.current_yaw * (1.0 - self.neck_fraction);
        let head_pitch = self.current_pitch * (1.0 - self.neck_fraction);
        let head_rot = Quat::from_euler(
            glam::EulerRot::YXZ,
            head_yaw * DEG_TO_RAD,
            head_pitch * DEG_TO_RAD,
            0.0,
        );
        results.push((self.head_bone, BoneTransform::from_rotation(head_rot).blend(self.weight)));

        if let Some(neck_bone) = self.neck_bone {
            let neck_yaw = self.current_yaw * self.neck_fraction;
            let neck_pitch = self.current_pitch * self.neck_fraction;
            let neck_rot = Quat::from_euler(
                glam::EulerRot::YXZ,
                neck_yaw * DEG_TO_RAD,
                neck_pitch * DEG_TO_RAD,
                0.0,
            );
            results.push((neck_bone, BoneTransform::from_rotation(neck_rot).blend(self.weight)));
        }

        results
    }
}

// ---------------------------------------------------------------------------
// BreathingLayer
// ---------------------------------------------------------------------------

/// Procedural breathing animation.
#[derive(Debug, Clone)]
pub struct BreathingLayer {
    /// Breathing rate (cycles per second).
    pub rate: f32,
    /// Breathing amplitude.
    pub amplitude: f32,
    /// Spine/chest bone ID.
    pub chest_bone: BoneId,
    /// Optional shoulder bones.
    pub shoulder_bones: Vec<BoneId>,
    /// Internal phase.
    phase: f32,
    /// Blend weight.
    pub weight: f32,
    /// Whether breathing is active.
    pub active: bool,
    /// Exertion level (0..1, increases breathing rate and amplitude).
    pub exertion: f32,
}

impl BreathingLayer {
    /// Create a new breathing layer.
    pub fn new(chest_bone: BoneId) -> Self {
        Self {
            rate: DEFAULT_BREATHING_RATE,
            amplitude: DEFAULT_BREATHING_AMPLITUDE,
            chest_bone,
            shoulder_bones: Vec::new(),
            phase: 0.0,
            weight: 1.0,
            active: true,
            exertion: 0.0,
        }
    }

    /// Update and return bone transforms.
    pub fn update(&mut self, dt: f32) -> Vec<(BoneId, BoneTransform)> {
        if !self.active {
            return Vec::new();
        }

        let effective_rate = self.rate * (1.0 + self.exertion * 2.0);
        let effective_amp = self.amplitude * (1.0 + self.exertion);

        self.phase += effective_rate * dt * TWO_PI;
        if self.phase > TWO_PI {
            self.phase -= TWO_PI;
        }

        let breath = self.phase.sin();
        let chest_expand = breath * effective_amp;

        let mut results = Vec::new();

        // Chest expands upward and forward
        let chest_transform = BoneTransform {
            translation: Vec3::new(0.0, chest_expand, chest_expand * 0.3),
            rotation: Quat::from_rotation_x(-chest_expand * 0.5),
            scale: Vec3::ONE,
            weight: self.weight,
        };
        results.push((self.chest_bone, chest_transform));

        // Shoulders rise slightly
        let shoulder_offset = breath * effective_amp * 0.3;
        for &bone in &self.shoulder_bones {
            results.push((bone, BoneTransform {
                translation: Vec3::new(0.0, shoulder_offset, 0.0),
                ..BoneTransform::default()
            }.blend(self.weight)));
        }

        results
    }
}

// ---------------------------------------------------------------------------
// IdleFidgetLayer
// ---------------------------------------------------------------------------

/// Procedural idle fidgets (small random motions).
#[derive(Debug, Clone)]
pub struct IdleFidgetLayer {
    /// Interval between fidgets (seconds).
    pub interval: f32,
    /// Maximum rotation amplitude (degrees).
    pub amplitude: f32,
    /// Bones to apply fidgets to.
    pub bones: Vec<BoneId>,
    /// Time since last fidget.
    timer: f32,
    /// Whether currently in a fidget.
    in_fidget: bool,
    /// Fidget progress (0..1).
    fidget_progress: f32,
    /// Fidget duration.
    pub fidget_duration: f32,
    /// Current fidget target rotation per bone.
    fidget_targets: HashMap<BoneId, Quat>,
    /// Blend weight.
    pub weight: f32,
    /// Whether active.
    pub active: bool,
    /// RNG seed state (simple LCG).
    rng_state: u32,
}

impl IdleFidgetLayer {
    /// Create a new fidget layer.
    pub fn new(bones: Vec<BoneId>) -> Self {
        Self {
            interval: DEFAULT_FIDGET_INTERVAL,
            amplitude: DEFAULT_FIDGET_AMPLITUDE,
            bones,
            timer: 0.0,
            in_fidget: false,
            fidget_progress: 0.0,
            fidget_duration: 0.5,
            fidget_targets: HashMap::new(),
            weight: 0.5,
            active: true,
            rng_state: 12345,
        }
    }

    /// Simple pseudo-random number generator (0..1).
    fn random(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_state >> 16) & 0x7FFF) as f32 / 32767.0
    }

    /// Update and return bone transforms.
    pub fn update(&mut self, dt: f32) -> Vec<(BoneId, BoneTransform)> {
        if !self.active {
            return Vec::new();
        }

        self.timer += dt;

        if !self.in_fidget {
            if self.timer >= self.interval {
                // Start a fidget
                self.in_fidget = true;
                self.fidget_progress = 0.0;
                self.timer = 0.0;

                // Generate random targets
                self.fidget_targets.clear();
                for &bone in &self.bones {
                    let yaw = (self.random() - 0.5) * self.amplitude * 2.0 * DEG_TO_RAD;
                    let pitch = (self.random() - 0.5) * self.amplitude * 2.0 * DEG_TO_RAD;
                    let roll = (self.random() - 0.5) * self.amplitude * DEG_TO_RAD;
                    self.fidget_targets.insert(bone, Quat::from_euler(
                        glam::EulerRot::YXZ, yaw, pitch, roll,
                    ));
                }
            }
            return Vec::new();
        }

        // Progress the fidget
        self.fidget_progress += dt / self.fidget_duration;
        if self.fidget_progress >= 1.0 {
            self.in_fidget = false;
            self.timer = 0.0;
            // Reset interval with some randomness
            self.interval = DEFAULT_FIDGET_INTERVAL * (0.5 + self.random());
            return Vec::new();
        }

        // Ease in-out
        let t = self.fidget_progress;
        let ease = if t < 0.5 {
            2.0 * t * t
        } else {
            1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
        };
        // Mirror: go to target then come back
        let blend = if t < 0.5 { ease * 2.0 } else { (1.0 - ease) * 2.0 };

        let mut results = Vec::new();
        for (&bone, &target_rot) in &self.fidget_targets {
            let rot = Quat::IDENTITY.slerp(target_rot, blend.min(1.0));
            results.push((bone, BoneTransform::from_rotation(rot).blend(self.weight)));
        }

        results
    }
}

// ---------------------------------------------------------------------------
// LeanLayer
// ---------------------------------------------------------------------------

/// Procedural lean into turns.
#[derive(Debug, Clone)]
pub struct LeanLayer {
    /// Lean factor (degrees per unit angular velocity).
    pub lean_factor: f32,
    /// Maximum lean angle (degrees).
    pub max_lean: f32,
    /// Spine bone to apply lean to.
    pub spine_bone: BoneId,
    /// Current lean angle.
    current_lean: f32,
    /// Smoothing speed.
    pub smooth_speed: f32,
    /// Blend weight.
    pub weight: f32,
    /// Whether active.
    pub active: bool,
}

impl LeanLayer {
    /// Create a new lean layer.
    pub fn new(spine_bone: BoneId) -> Self {
        Self {
            lean_factor: DEFAULT_LEAN_FACTOR,
            max_lean: MAX_LEAN_ANGLE,
            spine_bone,
            current_lean: 0.0,
            smooth_speed: 8.0,
            weight: 1.0,
            active: true,
        }
    }

    /// Update with angular velocity (degrees/second around Y axis).
    pub fn update(&mut self, dt: f32, angular_velocity_y: f32) -> Vec<(BoneId, BoneTransform)> {
        if !self.active {
            return Vec::new();
        }

        let target_lean = (-angular_velocity_y * self.lean_factor).clamp(-self.max_lean, self.max_lean);
        self.current_lean += (target_lean - self.current_lean) * (self.smooth_speed * dt).min(1.0);

        let rotation = Quat::from_rotation_z(self.current_lean * DEG_TO_RAD);
        vec![(self.spine_bone, BoneTransform::from_rotation(rotation).blend(self.weight))]
    }
}

// ---------------------------------------------------------------------------
// RecoilLayer
// ---------------------------------------------------------------------------

/// Procedural weapon recoil animation.
#[derive(Debug, Clone)]
pub struct RecoilLayer {
    /// Recovery speed.
    pub recovery_speed: f32,
    /// Current recoil displacement.
    current_recoil: Vec3,
    /// Current recoil rotation.
    current_rotation: Vec3,
    /// Weapon/hand bone.
    pub weapon_bone: BoneId,
    /// Optional shoulder bone for distributed recoil.
    pub shoulder_bone: Option<BoneId>,
    /// Shoulder fraction of recoil.
    pub shoulder_fraction: f32,
    /// Blend weight.
    pub weight: f32,
    /// Whether active.
    pub active: bool,
    /// Spring damping.
    pub damping: f32,
    /// Spring stiffness.
    pub stiffness: f32,
    /// Recoil velocity.
    velocity: Vec3,
    /// Rotational velocity.
    rot_velocity: Vec3,
}

impl RecoilLayer {
    /// Create a new recoil layer.
    pub fn new(weapon_bone: BoneId) -> Self {
        Self {
            recovery_speed: DEFAULT_RECOIL_RECOVERY,
            current_recoil: Vec3::ZERO,
            current_rotation: Vec3::ZERO,
            weapon_bone,
            shoulder_bone: None,
            shoulder_fraction: 0.2,
            weight: 1.0,
            active: true,
            damping: 10.0,
            stiffness: 50.0,
            velocity: Vec3::ZERO,
            rot_velocity: Vec3::ZERO,
        }
    }

    /// Apply a recoil impulse.
    pub fn apply_recoil(&mut self, displacement: Vec3, rotation: Vec3) {
        self.velocity += displacement;
        self.rot_velocity += rotation;
    }

    /// Update and return bone transforms.
    pub fn update(&mut self, dt: f32) -> Vec<(BoneId, BoneTransform)> {
        if !self.active {
            return Vec::new();
        }

        // Spring-damper system for translation
        let spring_force = -self.current_recoil * self.stiffness;
        let damping_force = -self.velocity * self.damping;
        self.velocity += (spring_force + damping_force) * dt;
        self.current_recoil += self.velocity * dt;

        // Spring-damper for rotation
        let rot_spring = -self.current_rotation * self.stiffness;
        let rot_damping = -self.rot_velocity * self.damping;
        self.rot_velocity += (rot_spring + rot_damping) * dt;
        self.current_rotation += self.rot_velocity * dt;

        let mut results = Vec::new();

        let weapon_rot = Quat::from_euler(
            glam::EulerRot::YXZ,
            self.current_rotation.y * (1.0 - self.shoulder_fraction),
            self.current_rotation.x * (1.0 - self.shoulder_fraction),
            self.current_rotation.z * (1.0 - self.shoulder_fraction),
        );
        let weapon_trans = self.current_recoil * (1.0 - self.shoulder_fraction);

        results.push((self.weapon_bone, BoneTransform {
            translation: weapon_trans,
            rotation: weapon_rot,
            scale: Vec3::ONE,
            weight: self.weight,
        }));

        if let Some(shoulder) = self.shoulder_bone {
            let s_rot = Quat::from_euler(
                glam::EulerRot::YXZ,
                self.current_rotation.y * self.shoulder_fraction,
                self.current_rotation.x * self.shoulder_fraction,
                self.current_rotation.z * self.shoulder_fraction,
            );
            results.push((shoulder, BoneTransform::from_rotation(s_rot).blend(self.weight)));
        }

        results
    }
}

// ---------------------------------------------------------------------------
// BlinkLayer
// ---------------------------------------------------------------------------

/// Procedural eye blinking.
#[derive(Debug, Clone)]
pub struct BlinkLayer {
    /// Blink interval (seconds).
    pub interval: f32,
    /// Blink duration (seconds).
    pub duration: f32,
    /// Eyelid bone(s).
    pub eyelid_bones: Vec<BoneId>,
    /// Timer until next blink.
    timer: f32,
    /// Current blink progress (0 = open, 1 = closed).
    blink_progress: f32,
    /// Whether currently blinking.
    blinking: bool,
    /// Blend weight.
    pub weight: f32,
    /// RNG state.
    rng_state: u32,
    /// Closed rotation for eyelids.
    pub closed_rotation: Quat,
}

impl BlinkLayer {
    /// Create a new blink layer.
    pub fn new(eyelid_bones: Vec<BoneId>) -> Self {
        Self {
            interval: DEFAULT_BLINK_INTERVAL,
            duration: DEFAULT_BLINK_DURATION,
            eyelid_bones,
            timer: DEFAULT_BLINK_INTERVAL,
            blink_progress: 0.0,
            blinking: false,
            weight: 1.0,
            rng_state: 54321,
            closed_rotation: Quat::from_rotation_x(-30.0 * DEG_TO_RAD),
        }
    }

    fn random(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_state >> 16) & 0x7FFF) as f32 / 32767.0
    }

    /// Update and return bone transforms.
    pub fn update(&mut self, dt: f32) -> Vec<(BoneId, BoneTransform)> {
        if self.eyelid_bones.is_empty() {
            return Vec::new();
        }

        if !self.blinking {
            self.timer -= dt;
            if self.timer <= 0.0 {
                self.blinking = true;
                self.blink_progress = 0.0;
            }
            return Vec::new();
        }

        self.blink_progress += dt / self.duration;
        if self.blink_progress >= 1.0 {
            self.blinking = false;
            self.timer = self.interval * (0.5 + self.random());
            return Vec::new();
        }

        // Triangle wave: 0->1->0
        let close = if self.blink_progress < 0.5 {
            self.blink_progress * 2.0
        } else {
            (1.0 - self.blink_progress) * 2.0
        };

        let rot = Quat::IDENTITY.slerp(self.closed_rotation, close);
        self.eyelid_bones.iter()
            .map(|&bone| (bone, BoneTransform::from_rotation(rot).blend(self.weight)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ProceduralWalkCycle
// ---------------------------------------------------------------------------

/// Procedural walk cycle driven by character velocity.
#[derive(Debug, Clone)]
pub struct ProceduralWalkCycle {
    /// Step frequency at unit speed (Hz).
    pub step_frequency: f32,
    /// Step height.
    pub step_height: f32,
    /// Stride length.
    pub stride_length: f32,
    /// Left foot bone.
    pub left_foot: BoneId,
    /// Right foot bone.
    pub right_foot: BoneId,
    /// Hip bone (for vertical bob).
    pub hip_bone: BoneId,
    /// Phase (0..TWO_PI).
    phase: f32,
    /// Blend weight.
    pub weight: f32,
    /// Whether active.
    pub active: bool,
    /// Hip bob amplitude.
    pub hip_bob: f32,
    /// Hip sway amplitude.
    pub hip_sway: f32,
}

impl ProceduralWalkCycle {
    /// Create a new procedural walk cycle.
    pub fn new(left_foot: BoneId, right_foot: BoneId, hip_bone: BoneId) -> Self {
        Self {
            step_frequency: DEFAULT_STEP_FREQUENCY,
            step_height: DEFAULT_STEP_HEIGHT,
            stride_length: DEFAULT_STRIDE_LENGTH,
            left_foot,
            right_foot,
            hip_bone,
            phase: 0.0,
            weight: 1.0,
            active: true,
            hip_bob: 0.02,
            hip_sway: 0.01,
        }
    }

    /// Update with character speed (m/s).
    pub fn update(&mut self, dt: f32, speed: f32) -> Vec<(BoneId, BoneTransform)> {
        if !self.active || speed < 0.1 {
            return Vec::new();
        }

        let frequency = self.step_frequency * speed;
        self.phase += frequency * dt * TWO_PI;
        if self.phase > TWO_PI {
            self.phase -= TWO_PI;
        }

        let speed_factor = speed.min(5.0) / 5.0;

        // Left foot: sin wave
        let left_y = (self.phase.sin().max(0.0)) * self.step_height * speed_factor;
        let left_z = self.phase.cos() * self.stride_length * 0.5 * speed_factor;

        // Right foot: opposite phase
        let right_phase = self.phase + PI;
        let right_y = (right_phase.sin().max(0.0)) * self.step_height * speed_factor;
        let right_z = right_phase.cos() * self.stride_length * 0.5 * speed_factor;

        // Hip bob and sway
        let hip_y = -(self.phase * 2.0).sin().abs() * self.hip_bob * speed_factor;
        let hip_x = (self.phase).sin() * self.hip_sway * speed_factor;

        let mut results = Vec::new();
        results.push((self.left_foot, BoneTransform::from_translation(
            Vec3::new(0.0, left_y, left_z),
        ).blend(self.weight)));
        results.push((self.right_foot, BoneTransform::from_translation(
            Vec3::new(0.0, right_y, right_z),
        ).blend(self.weight)));
        results.push((self.hip_bone, BoneTransform::from_translation(
            Vec3::new(hip_x, hip_y, 0.0),
        ).blend(self.weight)));

        results
    }
}

// ---------------------------------------------------------------------------
// TailPhysics
// ---------------------------------------------------------------------------

/// Simple chain physics for tails/appendages.
#[derive(Debug, Clone)]
pub struct TailPhysics {
    /// Bone IDs for each segment (root to tip).
    pub bones: Vec<BoneId>,
    /// Segment length.
    pub segment_length: f32,
    /// Stiffness (spring constant).
    pub stiffness: f32,
    /// Damping.
    pub damping: f32,
    /// Gravity influence.
    pub gravity: f32,
    /// Positions of each segment.
    positions: Vec<Vec3>,
    /// Velocities of each segment.
    velocities: Vec<Vec3>,
    /// Blend weight.
    pub weight: f32,
    /// Whether active.
    pub active: bool,
}

impl TailPhysics {
    /// Create a new tail physics system.
    pub fn new(bones: Vec<BoneId>, segment_length: f32) -> Self {
        let count = bones.len();
        Self {
            bones,
            segment_length,
            stiffness: TAIL_STIFFNESS,
            damping: TAIL_DAMPING,
            gravity: 5.0,
            positions: vec![Vec3::ZERO; count],
            velocities: vec![Vec3::ZERO; count],
            weight: 1.0,
            active: true,
        }
    }

    /// Initialize positions from a root position and direction.
    pub fn initialize(&mut self, root: Vec3, direction: Vec3) {
        let dir = direction.normalize_or_zero();
        for (i, pos) in self.positions.iter_mut().enumerate() {
            *pos = root + dir * (i as f32 * self.segment_length);
        }
    }

    /// Update with root bone position.
    pub fn update(&mut self, dt: f32, root_pos: Vec3) -> Vec<(BoneId, BoneTransform)> {
        if !self.active || self.bones.is_empty() {
            return Vec::new();
        }

        // Pin root
        self.positions[0] = root_pos;
        self.velocities[0] = Vec3::ZERO;

        // Simulate chain
        for i in 1..self.positions.len() {
            let parent_pos = self.positions[i - 1];
            let current_pos = self.positions[i];

            // Spring force toward parent at rest length
            let delta = current_pos - parent_pos;
            let dist = delta.length();
            let rest = self.segment_length;

            let spring_dir = if dist > EPSILON { delta / dist } else { Vec3::new(0.0, -1.0, 0.0) };
            let spring_force = -spring_dir * (dist - rest) * self.stiffness;

            // Gravity
            let gravity_force = Vec3::new(0.0, -self.gravity, 0.0);

            // Damping
            let damping_force = -self.velocities[i] * self.damping;

            let total_force = spring_force + gravity_force + damping_force;
            self.velocities[i] += total_force * dt;
            self.positions[i] += self.velocities[i] * dt;

            // Constrain distance from parent
            let new_delta = self.positions[i] - parent_pos;
            let new_dist = new_delta.length();
            if new_dist > rest * 1.5 {
                let correction = new_delta.normalize_or_zero() * rest * 1.5;
                self.positions[i] = parent_pos + correction;
            }
        }

        // Generate bone transforms as rotations from parent to child
        let mut results = Vec::new();
        for i in 0..self.bones.len().saturating_sub(1) {
            let dir = (self.positions[i + 1] - self.positions[i]).normalize_or_zero();
            let default_dir = Vec3::new(0.0, -1.0, 0.0);

            if dir.length() > EPSILON {
                let rotation = Quat::from_rotation_arc(default_dir, dir);
                results.push((self.bones[i], BoneTransform::from_rotation(rotation).blend(self.weight)));
            }
        }

        results
    }
}

// ---------------------------------------------------------------------------
// ProceduralAnimController
// ---------------------------------------------------------------------------

/// Controller managing all procedural animation layers for one character.
pub struct ProceduralAnimController {
    /// Look-at constraint.
    pub look_at: Option<LookAtConstraint>,
    /// Breathing layer.
    pub breathing: Option<BreathingLayer>,
    /// Fidget layer.
    pub fidgets: Option<IdleFidgetLayer>,
    /// Lean layer.
    pub lean: Option<LeanLayer>,
    /// Recoil layer.
    pub recoil: Option<RecoilLayer>,
    /// Blink layer.
    pub blink: Option<BlinkLayer>,
    /// Walk cycle.
    pub walk_cycle: Option<ProceduralWalkCycle>,
    /// Tail physics.
    pub tail: Option<TailPhysics>,
    /// Accumulated bone transforms from all layers.
    bone_transforms: HashMap<BoneId, BoneTransform>,
    /// Whether the controller is enabled.
    pub enabled: bool,
}

impl ProceduralAnimController {
    /// Create a new empty controller.
    pub fn new() -> Self {
        Self {
            look_at: None,
            breathing: None,
            fidgets: None,
            lean: None,
            recoil: None,
            blink: None,
            walk_cycle: None,
            tail: None,
            bone_transforms: HashMap::new(),
            enabled: true,
        }
    }

    /// Update all layers and accumulate bone transforms.
    pub fn update(
        &mut self,
        dt: f32,
        character_pos: Vec3,
        character_forward: Vec3,
        speed: f32,
        angular_velocity_y: f32,
    ) {
        if !self.enabled {
            self.bone_transforms.clear();
            return;
        }

        self.bone_transforms.clear();

        // Collect transforms from each layer
        let mut all_transforms: Vec<(BoneId, BoneTransform)> = Vec::new();

        if let Some(look_at) = &mut self.look_at {
            all_transforms.extend(look_at.update(dt, character_pos, character_forward));
        }
        if let Some(breathing) = &mut self.breathing {
            all_transforms.extend(breathing.update(dt));
        }
        if let Some(fidgets) = &mut self.fidgets {
            all_transforms.extend(fidgets.update(dt));
        }
        if let Some(lean) = &mut self.lean {
            all_transforms.extend(lean.update(dt, angular_velocity_y));
        }
        if let Some(recoil) = &mut self.recoil {
            all_transforms.extend(recoil.update(dt));
        }
        if let Some(blink) = &mut self.blink {
            all_transforms.extend(blink.update(dt));
        }
        if let Some(walk) = &mut self.walk_cycle {
            all_transforms.extend(walk.update(dt, speed));
        }
        if let Some(tail) = &mut self.tail {
            all_transforms.extend(tail.update(dt, character_pos));
        }

        // Accumulate transforms per bone
        for (bone, transform) in all_transforms {
            let entry = self.bone_transforms.entry(bone).or_insert(BoneTransform::default());
            *entry = entry.add(&transform);
        }
    }

    /// Get the final bone transform for a specific bone.
    pub fn get_bone_transform(&self, bone: BoneId) -> Option<&BoneTransform> {
        self.bone_transforms.get(&bone)
    }

    /// Get all bone transforms.
    pub fn all_bone_transforms(&self) -> &HashMap<BoneId, BoneTransform> {
        &self.bone_transforms
    }

    /// Get bone IDs that have transforms this frame.
    pub fn affected_bones(&self) -> Vec<BoneId> {
        self.bone_transforms.keys().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// ProceduralAnimComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component for procedural animation.
#[derive(Debug)]
pub struct ProceduralAnimComponent {
    /// Whether procedural animation is enabled for this entity.
    pub enabled: bool,
    /// Controller reference (stored externally due to size).
    pub controller_id: u64,
}

impl ProceduralAnimComponent {
    /// Create a new component.
    pub fn new(controller_id: u64) -> Self {
        Self {
            enabled: true,
            controller_id,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bone_transform_blend() {
        let t = BoneTransform {
            translation: Vec3::new(1.0, 0.0, 0.0),
            rotation: Quat::from_rotation_y(0.5),
            scale: Vec3::ONE,
            weight: 1.0,
        };
        let blended = t.blend(0.5);
        assert!((blended.translation.x - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_breathing_layer() {
        let mut breathing = BreathingLayer::new(BoneId(0));
        let transforms = breathing.update(0.016);
        assert!(!transforms.is_empty());
    }

    #[test]
    fn test_look_at_constraint() {
        let mut look_at = LookAtConstraint::new(BoneId(10));
        look_at.set_target(Vec3::new(0.0, 1.6, 5.0));

        let transforms = look_at.update(
            0.016,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!(!transforms.is_empty());
    }

    #[test]
    fn test_recoil_layer() {
        let mut recoil = RecoilLayer::new(BoneId(20));
        recoil.apply_recoil(Vec3::new(0.0, 0.1, -0.2), Vec3::new(-5.0, 0.0, 0.0));
        let transforms = recoil.update(0.016);
        assert!(!transforms.is_empty());
    }

    #[test]
    fn test_procedural_walk_cycle() {
        let mut walk = ProceduralWalkCycle::new(BoneId(30), BoneId(31), BoneId(0));
        let transforms = walk.update(0.016, 3.0);
        assert!(!transforms.is_empty());
    }

    #[test]
    fn test_controller() {
        let mut controller = ProceduralAnimController::new();
        controller.breathing = Some(BreathingLayer::new(BoneId(5)));
        controller.update(0.016, Vec3::ZERO, Vec3::Z, 0.0, 0.0);
        assert!(!controller.all_bone_transforms().is_empty());
    }
}
