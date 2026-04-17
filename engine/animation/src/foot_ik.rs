//! # Foot Placement IK
//!
//! Provides ground-aware foot placement using inverse kinematics. Raycasts
//! determine the ground height under each foot, two-bone IK plants the foot
//! on the surface, the pelvis height is adjusted to prevent over-extension,
//! and the toe is aligned to the surface normal for slopes.
//!
//! The system operates as a post-process on the final animation pose,
//! ensuring feet make solid contact with uneven terrain.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Math helpers (minimal, engine-independent)
// ---------------------------------------------------------------------------

/// A 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const DOWN: Self = Self { x: 0.0, y: -1.0, z: 0.0 };
    pub const FORWARD: Self = Self { x: 0.0, y: 0.0, z: -1.0 };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-8 {
            return Self::ZERO;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
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
        (self - other).length()
    }

    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// A quaternion rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half = angle * 0.5;
        let s = half.sin();
        let c = half.cos();
        let a = axis.normalized();
        Self {
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
            w: c,
        }
    }

    pub fn from_to_rotation(from: Vec3, to: Vec3) -> Self {
        let from = from.normalized();
        let to = to.normalized();
        let d = from.dot(to);
        if d > 0.9999 {
            return Self::IDENTITY;
        }
        if d < -0.9999 {
            // 180 degree rotation around an arbitrary perpendicular axis
            let perp = if from.x.abs() < 0.9 {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            let axis = from.cross(perp).normalized();
            return Self::from_axis_angle(axis, std::f32::consts::PI);
        }
        let axis = from.cross(to);
        let w = (1.0 + d).sqrt() * 0.5;  // actually this is the simpler formula
        let s = 0.5 / w;
        Self {
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
            w,
        }
    }

    pub fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if len < 1e-8 {
            return Self::IDENTITY;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
            w: self.w / len,
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
            // Linear interpolation for very close quaternions
            return Self {
                x: self.x + (b.x - self.x) * t,
                y: self.y + (b.y - self.y) * t,
                z: self.z + (b.z - self.z) * t,
                w: self.w + (b.w - self.w) * t,
            }
            .normalize();
        }
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        if sin_theta.abs() < 1e-8 {
            return self;
        }
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;
        Self {
            x: self.x * wa + b.x * wb,
            y: self.y * wa + b.y * wb,
            z: self.z * wa + b.z * wb,
            w: self.w * wa + b.w * wb,
        }
    }

    /// Rotate a vector by this quaternion.
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

    /// Multiply two quaternions.
    pub fn mul(self, other: Self) -> Self {
        Self {
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        }
    }

    /// Inverse (conjugate for unit quaternions).
    pub fn inverse(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }
}

// ---------------------------------------------------------------------------
// Raycast abstraction
// ---------------------------------------------------------------------------

/// Result of a ground raycast.
#[derive(Debug, Clone, Copy)]
pub struct GroundHit {
    /// World-space hit position.
    pub position: Vec3,
    /// Surface normal at the hit point.
    pub normal: Vec3,
    /// Distance from the ray origin.
    pub distance: f32,
    /// Whether a hit was detected.
    pub hit: bool,
}

impl Default for GroundHit {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: Vec3::UP,
            distance: 0.0,
            hit: false,
        }
    }
}

/// Trait for providing ground raycasting to the foot IK system.
pub trait GroundRaycaster {
    /// Cast a ray downward from `origin` with length `max_distance`.
    fn raycast_down(&self, origin: Vec3, max_distance: f32) -> GroundHit;
}

/// A simple flat-ground raycaster for testing.
pub struct FlatGroundRaycaster {
    /// Y-coordinate of the ground plane.
    pub ground_y: f32,
}

impl FlatGroundRaycaster {
    pub fn new(ground_y: f32) -> Self {
        Self { ground_y }
    }
}

impl GroundRaycaster for FlatGroundRaycaster {
    fn raycast_down(&self, origin: Vec3, max_distance: f32) -> GroundHit {
        let distance = origin.y - self.ground_y;
        if distance >= 0.0 && distance <= max_distance {
            GroundHit {
                position: Vec3::new(origin.x, self.ground_y, origin.z),
                normal: Vec3::UP,
                distance,
                hit: true,
            }
        } else {
            GroundHit::default()
        }
    }
}

/// A height-map based raycaster for testing slopes.
pub struct HeightMapRaycaster {
    /// Width and depth of the heightmap.
    pub width: usize,
    pub depth: usize,
    /// Scale factors (world units per cell).
    pub scale_x: f32,
    pub scale_z: f32,
    /// Height values.
    pub heights: Vec<f32>,
}

impl HeightMapRaycaster {
    /// Create a new flat heightmap.
    pub fn new_flat(width: usize, depth: usize, height: f32) -> Self {
        Self {
            width,
            depth,
            scale_x: 1.0,
            scale_z: 1.0,
            heights: vec![height; width * depth],
        }
    }

    /// Sample the height at a world-space position (bilinear interpolation).
    pub fn sample_height(&self, x: f32, z: f32) -> f32 {
        let gx = x / self.scale_x;
        let gz = z / self.scale_z;
        let ix = (gx.floor() as usize).min(self.width.saturating_sub(2));
        let iz = (gz.floor() as usize).min(self.depth.saturating_sub(2));
        let fx = gx - ix as f32;
        let fz = gz - iz as f32;

        let h00 = self.heights[iz * self.width + ix];
        let h10 = self.heights[iz * self.width + (ix + 1).min(self.width - 1)];
        let h01 = self.heights[(iz + 1).min(self.depth - 1) * self.width + ix];
        let h11 = self.heights[(iz + 1).min(self.depth - 1) * self.width + (ix + 1).min(self.width - 1)];

        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;
        h0 + (h1 - h0) * fz
    }

    /// Compute the surface normal at a world position via finite differences.
    pub fn sample_normal(&self, x: f32, z: f32) -> Vec3 {
        let eps = 0.1;
        let h_left = self.sample_height(x - eps, z);
        let h_right = self.sample_height(x + eps, z);
        let h_back = self.sample_height(x, z - eps);
        let h_front = self.sample_height(x, z + eps);
        Vec3::new(h_left - h_right, 2.0 * eps, h_back - h_front).normalized()
    }
}

impl GroundRaycaster for HeightMapRaycaster {
    fn raycast_down(&self, origin: Vec3, max_distance: f32) -> GroundHit {
        let ground_y = self.sample_height(origin.x, origin.z);
        let distance = origin.y - ground_y;
        if distance >= 0.0 && distance <= max_distance {
            GroundHit {
                position: Vec3::new(origin.x, ground_y, origin.z),
                normal: self.sample_normal(origin.x, origin.z),
                distance,
                hit: true,
            }
        } else {
            GroundHit::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Bone transform
// ---------------------------------------------------------------------------

/// A bone transform in the skeleton pose.
#[derive(Debug, Clone, Copy)]
pub struct BoneTransform {
    /// World-space position of the bone.
    pub position: Vec3,
    /// World-space rotation of the bone.
    pub rotation: Quat,
}

impl Default for BoneTransform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

// ---------------------------------------------------------------------------
// Two-bone IK solver
// ---------------------------------------------------------------------------

/// Solves two-bone IK (e.g. thigh -> shin -> foot).
///
/// Given three joint positions (root, mid, end) and a target position,
/// computes the required rotations for root and mid joints to place
/// the end effector at the target.
pub struct TwoBoneIkSolver;

impl TwoBoneIkSolver {
    /// Solve two-bone IK.
    ///
    /// # Parameters
    /// - `root_pos`: Position of the root joint (e.g. hip/thigh).
    /// - `mid_pos`: Position of the mid joint (e.g. knee).
    /// - `end_pos`: Position of the end effector (e.g. ankle/foot).
    /// - `target`: Desired position of the end effector.
    /// - `pole_target`: Hint for the knee/elbow direction.
    /// - `weight`: IK weight [0, 1].
    ///
    /// # Returns
    /// Updated (mid_pos, end_pos) after IK solve.
    pub fn solve(
        root_pos: Vec3,
        mid_pos: Vec3,
        end_pos: Vec3,
        target: Vec3,
        pole_target: Option<Vec3>,
        weight: f32,
    ) -> TwoBoneIkResult {
        let weight = weight.clamp(0.0, 1.0);
        if weight < 1e-6 {
            return TwoBoneIkResult {
                mid_position: mid_pos,
                end_position: end_pos,
                root_rotation: Quat::IDENTITY,
                mid_rotation: Quat::IDENTITY,
            };
        }

        let upper_len = root_pos.distance(mid_pos);
        let lower_len = mid_pos.distance(end_pos);
        let chain_len = upper_len + lower_len;

        // Weighted target
        let target = end_pos.lerp(target, weight);
        let target_dist = root_pos.distance(target).min(chain_len - 0.001);
        let target_dist = target_dist.max(0.001);

        // Direction from root to target
        let dir_to_target = (target - root_pos).normalized();

        // Use law of cosines to find the angle at the mid joint
        let a = upper_len;
        let b = lower_len;
        let c = target_dist;

        // Angle at root
        let cos_root = ((a * a + c * c - b * b) / (2.0 * a * c)).clamp(-1.0, 1.0);
        let root_angle = cos_root.acos();

        // Angle at mid
        let cos_mid = ((a * a + b * b - c * c) / (2.0 * a * b)).clamp(-1.0, 1.0);
        let mid_angle = cos_mid.acos();

        // Compute the plane normal (for the IK plane)
        let chain_dir = (end_pos - root_pos).normalized();
        let pole = pole_target.unwrap_or(mid_pos + Vec3::FORWARD);
        let pole_dir = (pole - root_pos).normalized();
        let plane_normal = chain_dir.cross(pole_dir).normalized();

        // Compute new mid position
        let up = plane_normal.cross(dir_to_target).normalized();
        let new_mid = root_pos
            + dir_to_target.scale(a * cos_root)
            + up.scale(a * root_angle.sin());

        // Compute new end position
        let mid_to_end_dir = (target - new_mid).normalized();
        let new_end = new_mid + mid_to_end_dir.scale(b);

        // Compute rotation corrections
        let orig_upper_dir = (mid_pos - root_pos).normalized();
        let new_upper_dir = (new_mid - root_pos).normalized();
        let root_rotation = Quat::from_to_rotation(orig_upper_dir, new_upper_dir);

        let orig_lower_dir = (end_pos - mid_pos).normalized();
        let new_lower_dir = (new_end - new_mid).normalized();
        let mid_rotation = Quat::from_to_rotation(orig_lower_dir, new_lower_dir);

        TwoBoneIkResult {
            mid_position: new_mid,
            end_position: new_end,
            root_rotation,
            mid_rotation,
        }
    }
}

/// Result of a two-bone IK solve.
#[derive(Debug, Clone)]
pub struct TwoBoneIkResult {
    /// New mid-joint position.
    pub mid_position: Vec3,
    /// New end-effector position.
    pub end_position: Vec3,
    /// Rotation correction for the root joint.
    pub root_rotation: Quat,
    /// Rotation correction for the mid joint.
    pub mid_rotation: Quat,
}

// ---------------------------------------------------------------------------
// Foot definition
// ---------------------------------------------------------------------------

/// Which foot (for multi-legged characters, extend with more variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FootSide {
    Left,
    Right,
}

impl fmt::Display for FootSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => write!(f, "left"),
            Self::Right => write!(f, "right"),
        }
    }
}

/// Configuration for a single foot in the IK system.
#[derive(Debug, Clone)]
pub struct FootDefinition {
    /// Which foot this is.
    pub side: FootSide,
    /// Bone name for the hip/thigh joint (IK root).
    pub hip_bone: String,
    /// Bone name for the knee joint (IK mid).
    pub knee_bone: String,
    /// Bone name for the ankle/foot joint (IK end effector).
    pub ankle_bone: String,
    /// Bone name for the toe joint (optional, for toe alignment).
    pub toe_bone: Option<String>,
    /// Vertical offset from the ankle bone to the bottom of the foot.
    pub foot_offset: f32,
    /// Raycast origin offset (height above the foot bone).
    pub raycast_height: f32,
    /// Maximum raycast distance downward.
    pub raycast_distance: f32,
    /// Pole target direction (world-space hint for knee direction).
    pub pole_direction: Vec3,
}

impl FootDefinition {
    /// Create a default left foot definition with standard bone names.
    pub fn left_default() -> Self {
        Self {
            side: FootSide::Left,
            hip_bone: "left_hip".into(),
            knee_bone: "left_knee".into(),
            ankle_bone: "left_ankle".into(),
            toe_bone: Some("left_toe".into()),
            foot_offset: 0.05,
            raycast_height: 0.5,
            raycast_distance: 1.5,
            pole_direction: Vec3::FORWARD,
        }
    }

    /// Create a default right foot definition.
    pub fn right_default() -> Self {
        Self {
            side: FootSide::Right,
            hip_bone: "right_hip".into(),
            knee_bone: "right_knee".into(),
            ankle_bone: "right_ankle".into(),
            toe_bone: Some("right_toe".into()),
            foot_offset: 0.05,
            raycast_height: 0.5,
            raycast_distance: 1.5,
            pole_direction: Vec3::FORWARD,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-foot runtime state
// ---------------------------------------------------------------------------

/// Runtime state tracked per foot for smooth IK transitions.
#[derive(Debug, Clone)]
pub struct FootIkState {
    /// Current IK target position (smoothed).
    pub target_position: Vec3,
    /// Current ground normal (smoothed).
    pub ground_normal: Vec3,
    /// Whether the foot is currently grounded.
    pub grounded: bool,
    /// Pelvis offset contribution from this foot.
    pub pelvis_offset: f32,
    /// IK weight (smoothed).
    pub ik_weight: f32,
    /// Previous frame's target (for smoothing).
    pub prev_target: Vec3,
    /// Previous ground normal.
    pub prev_normal: Vec3,
    /// Raycast result from the last frame.
    pub last_hit: GroundHit,
}

impl Default for FootIkState {
    fn default() -> Self {
        Self {
            target_position: Vec3::ZERO,
            ground_normal: Vec3::UP,
            grounded: false,
            pelvis_offset: 0.0,
            ik_weight: 0.0,
            prev_target: Vec3::ZERO,
            prev_normal: Vec3::UP,
            last_hit: GroundHit::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Foot IK settings
// ---------------------------------------------------------------------------

/// Configuration for the overall foot IK system.
#[derive(Debug, Clone)]
pub struct FootIkSettings {
    /// Whether foot IK is enabled.
    pub enabled: bool,
    /// Global IK weight (0 = no IK, 1 = full IK).
    pub weight: f32,
    /// Maximum pelvis adjustment (world units).
    pub max_pelvis_offset: f32,
    /// Smoothing speed for target position (higher = faster).
    pub position_smooth_speed: f32,
    /// Smoothing speed for ground normal alignment.
    pub normal_smooth_speed: f32,
    /// Smoothing speed for IK weight changes.
    pub weight_smooth_speed: f32,
    /// Whether to enable toe alignment to surface normal.
    pub align_toes: bool,
    /// Whether to adjust pelvis height.
    pub adjust_pelvis: bool,
    /// Pelvis smoothing speed.
    pub pelvis_smooth_speed: f32,
    /// Minimum distance from foot to ground to activate IK (hysteresis).
    pub activation_threshold: f32,
    /// Maximum slope angle (degrees) for foot alignment.
    pub max_slope_angle: f32,
}

impl Default for FootIkSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            weight: 1.0,
            max_pelvis_offset: 0.3,
            position_smooth_speed: 15.0,
            normal_smooth_speed: 10.0,
            weight_smooth_speed: 8.0,
            align_toes: true,
            adjust_pelvis: true,
            pelvis_smooth_speed: 10.0,
            activation_threshold: 0.02,
            max_slope_angle: 60.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Foot IK system
// ---------------------------------------------------------------------------

/// The main foot IK system that processes all feet.
pub struct FootIkSystem {
    /// Per-foot configuration.
    pub feet: Vec<FootDefinition>,
    /// Per-foot runtime state.
    pub foot_states: Vec<FootIkState>,
    /// System settings.
    pub settings: FootIkSettings,
    /// Name of the pelvis bone.
    pub pelvis_bone: String,
    /// Current smoothed pelvis offset.
    pelvis_offset: f32,
    /// Previous pelvis offset (for smoothing).
    prev_pelvis_offset: f32,
}

impl FootIkSystem {
    /// Create a new foot IK system with default humanoid settings.
    pub fn new_humanoid() -> Self {
        let feet = vec![
            FootDefinition::left_default(),
            FootDefinition::right_default(),
        ];
        let foot_states = vec![FootIkState::default(); feet.len()];
        Self {
            feet,
            foot_states,
            settings: FootIkSettings::default(),
            pelvis_bone: "pelvis".into(),
            pelvis_offset: 0.0,
            prev_pelvis_offset: 0.0,
        }
    }

    /// Create with custom foot definitions.
    pub fn new(feet: Vec<FootDefinition>, pelvis_bone: impl Into<String>) -> Self {
        let foot_states = vec![FootIkState::default(); feet.len()];
        Self {
            feet,
            foot_states,
            settings: FootIkSettings::default(),
            pelvis_bone: pelvis_bone.into(),
            pelvis_offset: 0.0,
            prev_pelvis_offset: 0.0,
        }
    }

    /// Update the foot IK system for one frame.
    ///
    /// Takes the current pose bone positions and modifies them in-place.
    pub fn update(
        &mut self,
        dt: f32,
        raycaster: &dyn GroundRaycaster,
        bone_positions: &mut HashMap<String, BoneTransform>,
    ) {
        if !self.settings.enabled || self.settings.weight < 1e-6 {
            return;
        }

        let global_weight = self.settings.weight;
        let mut min_pelvis_offset = 0.0f32;

        for i in 0..self.feet.len() {
            let foot_def = &self.feet[i];

            // Get current bone positions
            let ankle_pos = bone_positions
                .get(&foot_def.ankle_bone)
                .map(|b| b.position)
                .unwrap_or(Vec3::ZERO);

            // Raycast from above the foot downward
            let ray_origin = Vec3::new(
                ankle_pos.x,
                ankle_pos.y + foot_def.raycast_height,
                ankle_pos.z,
            );
            let hit = raycaster.raycast_down(ray_origin, foot_def.raycast_distance);

            let state = &mut self.foot_states[i];
            state.last_hit = hit;

            if hit.hit {
                // Compute target position (hit point + foot offset along normal)
                let target = hit.position + hit.normal.scale(foot_def.foot_offset);

                // Check slope angle
                let slope_cos = hit.normal.dot(Vec3::UP);
                let slope_angle = slope_cos.clamp(-1.0, 1.0).acos().to_degrees();
                let slope_valid = slope_angle <= self.settings.max_slope_angle;

                if slope_valid {
                    // Smooth target position
                    let smooth_rate = (self.settings.position_smooth_speed * dt).min(1.0);
                    state.target_position = state.prev_target.lerp(target, smooth_rate);
                    state.prev_target = state.target_position;

                    // Smooth ground normal
                    let normal_rate = (self.settings.normal_smooth_speed * dt).min(1.0);
                    state.ground_normal = state.prev_normal.lerp(hit.normal, normal_rate);
                    state.prev_normal = state.ground_normal;

                    // Smooth IK weight
                    let weight_target = global_weight;
                    let weight_rate = (self.settings.weight_smooth_speed * dt).min(1.0);
                    state.ik_weight += (weight_target - state.ik_weight) * weight_rate;

                    state.grounded = true;

                    // Pelvis offset: how much the foot needs to move down
                    let foot_delta = ankle_pos.y - target.y;
                    state.pelvis_offset = foot_delta.max(0.0);
                    min_pelvis_offset = min_pelvis_offset.max(state.pelvis_offset);
                } else {
                    // Too steep, fade out IK
                    let weight_rate = (self.settings.weight_smooth_speed * dt).min(1.0);
                    state.ik_weight += (0.0 - state.ik_weight) * weight_rate;
                    state.grounded = false;
                }
            } else {
                // No ground hit, fade out IK
                let weight_rate = (self.settings.weight_smooth_speed * dt).min(1.0);
                state.ik_weight += (0.0 - state.ik_weight) * weight_rate;
                state.grounded = false;
                state.pelvis_offset = 0.0;
            }
        }

        // Update pelvis offset
        if self.settings.adjust_pelvis {
            let target_offset = min_pelvis_offset.min(self.settings.max_pelvis_offset);
            let smooth_rate = (self.settings.pelvis_smooth_speed * dt).min(1.0);
            self.pelvis_offset += (target_offset - self.pelvis_offset) * smooth_rate;

            // Apply pelvis offset
            if let Some(pelvis) = bone_positions.get_mut(&self.pelvis_bone) {
                pelvis.position.y -= self.pelvis_offset;
            }
        }

        // Apply two-bone IK for each foot
        for i in 0..self.feet.len() {
            let foot_def = &self.feet[i];
            let state = &self.foot_states[i];

            if state.ik_weight < 1e-4 {
                continue;
            }

            let hip_pos = bone_positions
                .get(&foot_def.hip_bone)
                .map(|b| b.position)
                .unwrap_or(Vec3::ZERO);
            let knee_pos = bone_positions
                .get(&foot_def.knee_bone)
                .map(|b| b.position)
                .unwrap_or(Vec3::ZERO);
            let ankle_pos = bone_positions
                .get(&foot_def.ankle_bone)
                .map(|b| b.position)
                .unwrap_or(Vec3::ZERO);

            // Adjust target for pelvis offset
            let mut target = state.target_position;
            if self.settings.adjust_pelvis {
                target.y += self.pelvis_offset * (1.0 - state.ik_weight);
            }

            // Pole target (knee direction hint)
            let pole = knee_pos + foot_def.pole_direction;

            let ik_result = TwoBoneIkSolver::solve(
                hip_pos,
                knee_pos,
                ankle_pos,
                target,
                Some(pole),
                state.ik_weight,
            );

            // Update bone positions
            if let Some(knee) = bone_positions.get_mut(&foot_def.knee_bone) {
                knee.position = ik_result.mid_position;
                knee.rotation = ik_result.mid_rotation.mul(knee.rotation);
            }
            if let Some(ankle) = bone_positions.get_mut(&foot_def.ankle_bone) {
                ankle.position = ik_result.end_position;
            }

            // Align toe to surface normal
            if self.settings.align_toes {
                if let Some(ref toe_name) = foot_def.toe_bone {
                    if let Some(toe) = bone_positions.get_mut(toe_name) {
                        let foot_rotation = Quat::from_to_rotation(Vec3::UP, state.ground_normal);
                        let blended = Quat::IDENTITY.slerp(foot_rotation, state.ik_weight);
                        toe.rotation = blended.mul(toe.rotation);
                    }
                }
            }
        }
    }

    /// Get the current pelvis offset.
    pub fn pelvis_offset(&self) -> f32 {
        self.pelvis_offset
    }

    /// Get the state for a specific foot.
    pub fn foot_state(&self, side: FootSide) -> Option<&FootIkState> {
        self.feet
            .iter()
            .position(|f| f.side == side)
            .and_then(|idx| self.foot_states.get(idx))
    }

    /// Reset all foot states.
    pub fn reset(&mut self) {
        for state in &mut self.foot_states {
            *state = FootIkState::default();
        }
        self.pelvis_offset = 0.0;
        self.prev_pelvis_offset = 0.0;
    }

    /// Enable or disable foot IK.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.settings.enabled = enabled;
    }

    /// Set the global IK weight.
    pub fn set_weight(&mut self, weight: f32) {
        self.settings.weight = weight.clamp(0.0, 1.0);
    }
}

impl Default for FootIkSystem {
    fn default() -> Self {
        Self::new_humanoid()
    }
}

// ---------------------------------------------------------------------------
// Debug visualisation data
// ---------------------------------------------------------------------------

/// Data for debug-drawing the foot IK state.
#[derive(Debug, Clone)]
pub struct FootIkDebugInfo {
    /// Per-foot debug data.
    pub feet: Vec<FootDebugData>,
    /// Current pelvis offset.
    pub pelvis_offset: f32,
}

/// Debug data for a single foot.
#[derive(Debug, Clone)]
pub struct FootDebugData {
    /// Foot side.
    pub side: FootSide,
    /// Raycast origin.
    pub ray_origin: Vec3,
    /// Raycast hit point (if any).
    pub hit_point: Option<Vec3>,
    /// Hit normal.
    pub hit_normal: Vec3,
    /// IK target position.
    pub target: Vec3,
    /// Current IK weight.
    pub weight: f32,
    /// Whether grounded.
    pub grounded: bool,
}

impl FootIkSystem {
    /// Generate debug visualisation data.
    pub fn debug_info(&self) -> FootIkDebugInfo {
        let mut feet = Vec::new();
        for (i, def) in self.feet.iter().enumerate() {
            let state = &self.foot_states[i];
            feet.push(FootDebugData {
                side: def.side,
                ray_origin: Vec3::ZERO, // Would be the actual ray origin
                hit_point: if state.last_hit.hit {
                    Some(state.last_hit.position)
                } else {
                    None
                },
                hit_normal: state.ground_normal,
                target: state.target_position,
                weight: state.ik_weight,
                grounded: state.grounded,
            });
        }
        FootIkDebugInfo {
            feet,
            pelvis_offset: self.pelvis_offset,
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
    fn test_vec3_basic() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a + b;
        assert_eq!(c.x, 5.0);
        assert_eq!(c.y, 7.0);
        assert_eq!(c.z, 9.0);
    }

    #[test]
    fn test_vec3_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(0.0, 5.0, 0.0);
        let n = v.normalized();
        assert!((n.length() - 1.0).abs() < 1e-5);
        assert!((n.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!((a.dot(b)).abs() < 1e-5);
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!((z.x).abs() < 1e-5);
        assert!((z.y).abs() < 1e-5);
        assert!((z.z - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_quat_identity() {
        let q = Quat::IDENTITY;
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = q.rotate_vector(v);
        assert!((rotated.x - v.x).abs() < 1e-5);
        assert!((rotated.y - v.y).abs() < 1e-5);
        assert!((rotated.z - v.z).abs() < 1e-5);
    }

    #[test]
    fn test_quat_from_to_rotation() {
        let from = Vec3::UP;
        let to = Vec3::new(1.0, 0.0, 0.0).normalized();
        let q = Quat::from_to_rotation(from, to);
        let result = q.rotate_vector(from).normalized();
        assert!((result.x - to.x).abs() < 0.1);
    }

    #[test]
    fn test_flat_ground_raycaster() {
        let raycaster = FlatGroundRaycaster::new(0.0);
        let hit = raycaster.raycast_down(Vec3::new(0.0, 1.0, 0.0), 2.0);
        assert!(hit.hit);
        assert!((hit.position.y).abs() < 1e-5);
        assert!((hit.distance - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_flat_ground_raycaster_miss() {
        let raycaster = FlatGroundRaycaster::new(0.0);
        let hit = raycaster.raycast_down(Vec3::new(0.0, 5.0, 0.0), 2.0);
        assert!(!hit.hit);
    }

    #[test]
    fn test_heightmap_raycaster() {
        let raycaster = HeightMapRaycaster::new_flat(10, 10, 0.5);
        let hit = raycaster.raycast_down(Vec3::new(5.0, 2.0, 5.0), 5.0);
        assert!(hit.hit);
        assert!((hit.position.y - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_two_bone_ik_straight() {
        let root = Vec3::new(0.0, 2.0, 0.0);
        let mid = Vec3::new(0.0, 1.0, 0.0);
        let end = Vec3::new(0.0, 0.0, 0.0);
        let target = Vec3::new(0.0, -0.1, 0.0);
        let result = TwoBoneIkSolver::solve(root, mid, end, target, None, 1.0);
        // End should be close to target (or as close as chain length allows)
        assert!(result.end_position.distance(target) < 0.5);
    }

    #[test]
    fn test_two_bone_ik_zero_weight() {
        let root = Vec3::new(0.0, 2.0, 0.0);
        let mid = Vec3::new(0.0, 1.0, 0.0);
        let end = Vec3::new(0.0, 0.0, 0.0);
        let target = Vec3::new(1.0, 0.0, 0.0);
        let result = TwoBoneIkSolver::solve(root, mid, end, target, None, 0.0);
        // With zero weight, positions should be unchanged
        assert!((result.mid_position.x - mid.x).abs() < 1e-5);
        assert!((result.end_position.x - end.x).abs() < 1e-5);
    }

    #[test]
    fn test_foot_ik_system_creation() {
        let system = FootIkSystem::new_humanoid();
        assert_eq!(system.feet.len(), 2);
        assert_eq!(system.foot_states.len(), 2);
        assert!(system.settings.enabled);
    }

    #[test]
    fn test_foot_ik_system_update_flat_ground() {
        let mut system = FootIkSystem::new_humanoid();
        let raycaster = FlatGroundRaycaster::new(0.0);
        let mut bones = HashMap::new();
        bones.insert("pelvis".into(), BoneTransform {
            position: Vec3::new(0.0, 1.0, 0.0),
            rotation: Quat::IDENTITY,
        });
        bones.insert("left_hip".into(), BoneTransform {
            position: Vec3::new(-0.1, 0.9, 0.0),
            rotation: Quat::IDENTITY,
        });
        bones.insert("left_knee".into(), BoneTransform {
            position: Vec3::new(-0.1, 0.5, 0.0),
            rotation: Quat::IDENTITY,
        });
        bones.insert("left_ankle".into(), BoneTransform {
            position: Vec3::new(-0.1, 0.1, 0.0),
            rotation: Quat::IDENTITY,
        });
        bones.insert("right_hip".into(), BoneTransform {
            position: Vec3::new(0.1, 0.9, 0.0),
            rotation: Quat::IDENTITY,
        });
        bones.insert("right_knee".into(), BoneTransform {
            position: Vec3::new(0.1, 0.5, 0.0),
            rotation: Quat::IDENTITY,
        });
        bones.insert("right_ankle".into(), BoneTransform {
            position: Vec3::new(0.1, 0.1, 0.0),
            rotation: Quat::IDENTITY,
        });

        // Run several frames to let smoothing converge
        for _ in 0..60 {
            system.update(1.0 / 60.0, &raycaster, &mut bones);
        }

        // After IK, feet should be near the ground
        let left_state = system.foot_state(FootSide::Left).unwrap();
        assert!(left_state.grounded);
    }

    #[test]
    fn test_foot_ik_disabled() {
        let mut system = FootIkSystem::new_humanoid();
        system.set_enabled(false);
        let raycaster = FlatGroundRaycaster::new(0.0);
        let mut bones = HashMap::new();
        bones.insert("pelvis".into(), BoneTransform {
            position: Vec3::new(0.0, 1.0, 0.0),
            rotation: Quat::IDENTITY,
        });
        let original_y = bones["pelvis"].position.y;
        system.update(0.016, &raycaster, &mut bones);
        // Pelvis should be unchanged when disabled
        assert!((bones["pelvis"].position.y - original_y).abs() < 1e-5);
    }

    #[test]
    fn test_foot_ik_reset() {
        let mut system = FootIkSystem::new_humanoid();
        system.pelvis_offset = 0.5;
        system.foot_states[0].ik_weight = 0.8;
        system.reset();
        assert!((system.pelvis_offset).abs() < 1e-5);
        assert!((system.foot_states[0].ik_weight).abs() < 1e-5);
    }

    #[test]
    fn test_foot_ik_debug_info() {
        let system = FootIkSystem::new_humanoid();
        let debug = system.debug_info();
        assert_eq!(debug.feet.len(), 2);
        assert_eq!(debug.feet[0].side, FootSide::Left);
        assert_eq!(debug.feet[1].side, FootSide::Right);
    }

    #[test]
    fn test_quat_slerp() {
        let a = Quat::IDENTITY;
        let b = Quat::from_axis_angle(Vec3::UP, std::f32::consts::FRAC_PI_2);
        let mid = a.slerp(b, 0.5);
        // At t=0.5, the angle should be half of 90 degrees
        let result = mid.rotate_vector(Vec3::new(1.0, 0.0, 0.0));
        // Should be roughly 45 degrees rotated
        assert!(result.x > 0.5);
        assert!(result.z.abs() > 0.3);
    }

    #[test]
    fn test_vec3_lerp() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 10.0, 10.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-5);
    }
}
