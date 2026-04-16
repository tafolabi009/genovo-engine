//! Extended root motion system for the Genovo animation engine.
//!
//! Root motion allows animation-driven character movement by extracting
//! the root bone's translation and rotation from animation clips and
//! applying them to the entity's world transform or physics body.
//!
//! This module extends the basic `RootMotion` struct in `skeleton.rs` with:
//!
//! - Multiple extraction modes (animation-only, physics-only, hybrid)
//! - Root motion accumulation across frames
//! - Ground alignment (project motion onto terrain normal)
//! - Velocity computation for physics integration
//! - Root motion filtering (separate horizontal/vertical components)

use genovo_core::Transform;
use glam::{Quat, Vec3};

use crate::skeleton::{AnimationClip, RootMotion};

// ===========================================================================
// RootMotionMode
// ===========================================================================

/// How root motion is applied to the entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootMotionMode {
    /// Root motion is extracted from the animation and applied directly
    /// to the entity's transform. The root bone is zeroed out in the
    /// animation pose so the character doesn't move twice.
    Animation,

    /// Root motion is extracted but applied to the physics body instead
    /// of the transform. Useful for physics-driven characters that need
    /// animation-accurate velocity.
    Physics,

    /// Root motion drives both the transform and provides velocity hints
    /// to the physics system. The physics system can override or blend
    /// with the animation motion.
    Both,

    /// Root motion is not extracted. The root bone moves naturally as
    /// part of the animation (in-place animation style).
    None,
}

impl Default for RootMotionMode {
    fn default() -> Self {
        Self::Animation
    }
}

// ===========================================================================
// RootMotionExtractor
// ===========================================================================

/// Configurable root motion extractor.
///
/// Provides fine-grained control over which components of root motion
/// are extracted and how they are applied.
#[derive(Debug, Clone)]
pub struct RootMotionExtractor {
    /// The extraction mode.
    pub mode: RootMotionMode,

    /// Index of the root bone in the skeleton.
    pub root_bone_index: usize,

    /// Whether to extract horizontal (XZ plane) translation.
    pub extract_horizontal: bool,

    /// Whether to extract vertical (Y axis) translation.
    pub extract_vertical: bool,

    /// Whether to extract rotation.
    pub extract_rotation: bool,

    /// Whether to zero out the extracted components from the animation
    /// pose (preventing double movement).
    pub zero_extracted: bool,

    /// Accumulated root motion (for multi-frame accumulation).
    accumulated_position: Vec3,

    /// Accumulated root rotation.
    accumulated_rotation: Quat,

    /// Previous frame's root position for delta computation.
    prev_root_position: Vec3,

    /// Previous frame's root rotation.
    prev_root_rotation: Quat,

    /// Whether we have a valid previous frame.
    has_previous: bool,

    /// Ground normal for terrain alignment (default: Vec3::Y).
    ground_normal: Vec3,

    /// Whether ground alignment is enabled.
    pub ground_alignment: bool,

    /// Computed velocity from root motion (units per second).
    pub velocity: Vec3,

    /// Computed angular velocity from root motion (radians per second).
    pub angular_velocity: f32,
}

impl RootMotionExtractor {
    /// Create a new root motion extractor.
    pub fn new(root_bone_index: usize) -> Self {
        Self {
            mode: RootMotionMode::Animation,
            root_bone_index,
            extract_horizontal: true,
            extract_vertical: false,
            extract_rotation: true,
            zero_extracted: true,
            accumulated_position: Vec3::ZERO,
            accumulated_rotation: Quat::IDENTITY,
            prev_root_position: Vec3::ZERO,
            prev_root_rotation: Quat::IDENTITY,
            has_previous: false,
            ground_normal: Vec3::Y,
            ground_alignment: false,
            velocity: Vec3::ZERO,
            angular_velocity: 0.0,
        }
    }

    /// Set the extraction mode.
    pub fn with_mode(mut self, mode: RootMotionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable or disable vertical extraction.
    pub fn with_vertical(mut self, extract: bool) -> Self {
        self.extract_vertical = extract;
        self
    }

    /// Enable or disable horizontal extraction.
    pub fn with_horizontal(mut self, extract: bool) -> Self {
        self.extract_horizontal = extract;
        self
    }

    /// Enable or disable rotation extraction.
    pub fn with_rotation(mut self, extract: bool) -> Self {
        self.extract_rotation = extract;
        self
    }

    /// Enable ground alignment.
    pub fn with_ground_alignment(mut self, enabled: bool) -> Self {
        self.ground_alignment = enabled;
        self
    }

    /// Set the ground normal for terrain alignment.
    pub fn set_ground_normal(&mut self, normal: Vec3) {
        self.ground_normal = normal.normalize_or_zero();
        if self.ground_normal.length_squared() < f32::EPSILON {
            self.ground_normal = Vec3::Y;
        }
    }

    /// Extract root motion from a clip between two time points.
    ///
    /// Returns the filtered `RootMotion` delta and updates internal state
    /// for velocity computation and accumulation.
    pub fn extract(
        &mut self,
        clip: &AnimationClip,
        prev_time: f32,
        curr_time: f32,
        dt: f32,
    ) -> RootMotion {
        if self.mode == RootMotionMode::None {
            return RootMotion::default();
        }

        // Use the basic extraction from the skeleton module.
        let raw = RootMotion::extract(clip, self.root_bone_index, prev_time, curr_time);

        // Filter the motion components.
        let mut delta_pos = raw.delta_position;
        let mut delta_rot = raw.delta_rotation;

        // Apply component filtering.
        if !self.extract_horizontal {
            delta_pos.x = 0.0;
            delta_pos.z = 0.0;
        }
        if !self.extract_vertical {
            delta_pos.y = 0.0;
        }
        if !self.extract_rotation {
            delta_rot = Quat::IDENTITY;
        }

        // Ground alignment: project horizontal motion onto the terrain plane.
        if self.ground_alignment && self.ground_normal != Vec3::Y {
            delta_pos = self.project_onto_ground(delta_pos);
        }

        // Compute velocity.
        if dt > f32::EPSILON {
            self.velocity = delta_pos / dt;
            // Angular velocity: extract the angle from the delta rotation.
            let (axis, angle) = delta_rot.to_axis_angle();
            self.angular_velocity = if axis.dot(Vec3::Y) > 0.0 {
                angle / dt
            } else {
                -angle / dt
            };
        }

        // Accumulate.
        self.accumulated_position += delta_pos;
        self.accumulated_rotation = self.accumulated_rotation * delta_rot;

        // Update previous state.
        if let Some(track) = clip.track_for_bone(self.root_bone_index) {
            self.prev_root_position = track.sample_position(curr_time);
            self.prev_root_rotation = track.sample_rotation(curr_time);
        }
        self.has_previous = true;

        RootMotion {
            delta_position: delta_pos,
            delta_rotation: delta_rot,
        }
    }

    /// Project a motion vector onto the ground plane defined by `ground_normal`.
    fn project_onto_ground(&self, motion: Vec3) -> Vec3 {
        // Decompose into horizontal and vertical components relative to ground.
        let vertical_component = self.ground_normal * motion.dot(self.ground_normal);
        let horizontal_component = motion - vertical_component;

        if self.extract_vertical {
            horizontal_component + vertical_component
        } else {
            horizontal_component
        }
    }

    /// Apply the extracted root motion to an entity transform.
    ///
    /// This is the most common way to consume root motion. It adds the
    /// delta position (rotated by the entity's current orientation) and
    /// multiplies the delta rotation.
    pub fn apply_to_transform(&self, motion: &RootMotion, transform: &mut Transform) {
        match self.mode {
            RootMotionMode::Animation | RootMotionMode::Both => {
                // Rotate the delta position by the entity's current orientation
                // so the motion is in world space.
                let world_delta = transform.rotation * motion.delta_position;
                transform.position += world_delta;
                transform.rotation = transform.rotation * motion.delta_rotation;
            }
            RootMotionMode::Physics | RootMotionMode::None => {
                // Physics mode: don't modify the transform directly.
                // The caller should read `self.velocity` and apply it to the
                // physics body instead.
            }
        }
    }

    /// Zero out the extracted root motion components from a pose.
    ///
    /// Call this after extraction to prevent the character from moving
    /// twice (once from root motion, once from the animation).
    pub fn zero_root_in_pose(&self, pose: &mut [Transform]) {
        if !self.zero_extracted {
            return;
        }

        if self.root_bone_index >= pose.len() {
            return;
        }

        let root = &mut pose[self.root_bone_index];

        if self.extract_horizontal {
            root.position.x = 0.0;
            root.position.z = 0.0;
        }
        if self.extract_vertical {
            root.position.y = 0.0;
        }
        if self.extract_rotation {
            root.rotation = Quat::IDENTITY;
        }
    }

    /// Get the accumulated position since the last reset.
    pub fn accumulated_position(&self) -> Vec3 {
        self.accumulated_position
    }

    /// Get the accumulated rotation since the last reset.
    pub fn accumulated_rotation(&self) -> Quat {
        self.accumulated_rotation
    }

    /// Reset the accumulated motion.
    pub fn reset_accumulation(&mut self) {
        self.accumulated_position = Vec3::ZERO;
        self.accumulated_rotation = Quat::IDENTITY;
    }

    /// Reset all state (for animation transitions or teleports).
    pub fn reset(&mut self) {
        self.accumulated_position = Vec3::ZERO;
        self.accumulated_rotation = Quat::IDENTITY;
        self.prev_root_position = Vec3::ZERO;
        self.prev_root_rotation = Quat::IDENTITY;
        self.has_previous = false;
        self.velocity = Vec3::ZERO;
        self.angular_velocity = 0.0;
    }

    /// Compute the root motion velocity in the entity's local space.
    pub fn local_velocity(&self) -> Vec3 {
        self.velocity
    }

    /// Compute the root motion velocity in world space given the entity rotation.
    pub fn world_velocity(&self, entity_rotation: Quat) -> Vec3 {
        entity_rotation * self.velocity
    }

    /// Get the current speed (magnitude of velocity).
    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }

    /// Get the horizontal speed (XZ plane only).
    pub fn horizontal_speed(&self) -> f32 {
        Vec3::new(self.velocity.x, 0.0, self.velocity.z).length()
    }
}

impl Default for RootMotionExtractor {
    fn default() -> Self {
        Self::new(0)
    }
}

// ===========================================================================
// RootMotionBlender
// ===========================================================================

/// Blends root motion from multiple sources.
///
/// Used when crossfading between animations or blending multiple animation
/// layers, each of which may produce root motion.
#[derive(Debug, Clone)]
pub struct RootMotionBlender {
    /// Accumulated blended delta position.
    pub blended_position: Vec3,
    /// Accumulated blended delta rotation.
    pub blended_rotation: Quat,
    /// Total weight accumulated (for normalization).
    total_weight: f32,
}

impl RootMotionBlender {
    /// Create a new blender.
    pub fn new() -> Self {
        Self {
            blended_position: Vec3::ZERO,
            blended_rotation: Quat::IDENTITY,
            total_weight: 0.0,
        }
    }

    /// Add a root motion contribution with the given weight.
    pub fn add(&mut self, motion: &RootMotion, weight: f32) {
        self.blended_position += motion.delta_position * weight;
        self.blended_rotation = self.blended_rotation.slerp(
            self.blended_rotation * motion.delta_rotation,
            weight / (self.total_weight + weight).max(f32::EPSILON),
        );
        self.total_weight += weight;
    }

    /// Get the final blended root motion.
    pub fn result(&self) -> RootMotion {
        RootMotion {
            delta_position: self.blended_position,
            delta_rotation: self.blended_rotation,
        }
    }

    /// Reset the blender for a new frame.
    pub fn reset(&mut self) {
        self.blended_position = Vec3::ZERO;
        self.blended_rotation = Quat::IDENTITY;
        self.total_weight = 0.0;
    }
}

impl Default for RootMotionBlender {
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
    use crate::skeleton::{AnimationClip, BoneTrack, Keyframe};

    fn make_walk_clip() -> AnimationClip {
        let mut clip = AnimationClip::new("Walk", 1.0);
        clip.looping = true;

        let mut root_track = BoneTrack::new(0);
        root_track.position_keys = vec![
            Keyframe::new(0.0, Vec3::ZERO),
            Keyframe::new(1.0, Vec3::new(0.0, 0.0, 2.0)),
        ];
        root_track.rotation_keys = vec![
            Keyframe::new(0.0, Quat::IDENTITY),
            Keyframe::new(1.0, Quat::IDENTITY),
        ];
        root_track.scale_keys = vec![
            Keyframe::new(0.0, Vec3::ONE),
            Keyframe::new(1.0, Vec3::ONE),
        ];
        clip.add_track(root_track);

        clip
    }

    #[test]
    fn extractor_basic() {
        let mut extractor = RootMotionExtractor::new(0);
        let clip = make_walk_clip();

        let motion = extractor.extract(&clip, 0.0, 0.5, 0.5);
        assert!(motion.delta_position.z > 0.0, "Should have forward motion");
    }

    #[test]
    fn extractor_no_vertical() {
        let mut extractor = RootMotionExtractor::new(0).with_vertical(false);
        let mut clip = AnimationClip::new("Jump", 1.0);
        let mut track = BoneTrack::new(0);
        track.position_keys = vec![
            Keyframe::new(0.0, Vec3::ZERO),
            Keyframe::new(1.0, Vec3::new(0.0, 5.0, 2.0)),
        ];
        track.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
        track.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
        clip.add_track(track);

        let motion = extractor.extract(&clip, 0.0, 0.5, 0.5);
        assert!((motion.delta_position.y).abs() < f32::EPSILON, "Vertical should be filtered out");
        assert!(motion.delta_position.z > 0.0, "Horizontal should remain");
    }

    #[test]
    fn extractor_no_horizontal() {
        let mut extractor = RootMotionExtractor::new(0).with_horizontal(false);
        let clip = make_walk_clip();

        let motion = extractor.extract(&clip, 0.0, 0.5, 0.5);
        assert!((motion.delta_position.x).abs() < f32::EPSILON);
        assert!((motion.delta_position.z).abs() < f32::EPSILON);
    }

    #[test]
    fn extractor_none_mode() {
        let mut extractor = RootMotionExtractor::new(0).with_mode(RootMotionMode::None);
        let clip = make_walk_clip();

        let motion = extractor.extract(&clip, 0.0, 0.5, 0.5);
        assert!((motion.delta_position).length() < f32::EPSILON);
    }

    #[test]
    fn extractor_velocity() {
        let mut extractor = RootMotionExtractor::new(0);
        let clip = make_walk_clip();

        extractor.extract(&clip, 0.0, 0.5, 0.5);
        assert!(extractor.speed() > 0.0);
        assert!(extractor.horizontal_speed() > 0.0);
    }

    #[test]
    fn extractor_accumulation() {
        let mut extractor = RootMotionExtractor::new(0);
        let clip = make_walk_clip();

        extractor.extract(&clip, 0.0, 0.5, 0.5);
        extractor.extract(&clip, 0.5, 1.0, 0.5);

        let accum = extractor.accumulated_position();
        // Should have accumulated ~2.0 units of Z movement.
        assert!((accum.z - 2.0).abs() < 0.1, "Accumulated Z should be ~2.0, got {}", accum.z);
    }

    #[test]
    fn extractor_reset() {
        let mut extractor = RootMotionExtractor::new(0);
        let clip = make_walk_clip();

        extractor.extract(&clip, 0.0, 0.5, 0.5);
        assert!(extractor.accumulated_position().length() > 0.0);

        extractor.reset();
        assert!((extractor.accumulated_position().length()).abs() < f32::EPSILON);
        assert!((extractor.speed()).abs() < f32::EPSILON);
    }

    #[test]
    fn extractor_apply_to_transform() {
        let mut extractor = RootMotionExtractor::new(0);
        let clip = make_walk_clip();
        let motion = extractor.extract(&clip, 0.0, 0.5, 0.5);

        let mut transform = Transform::IDENTITY;
        extractor.apply_to_transform(&motion, &mut transform);

        assert!(transform.position.z > 0.0, "Transform should move forward");
    }

    #[test]
    fn extractor_zero_root_in_pose() {
        let extractor = RootMotionExtractor::new(0);
        let mut pose = vec![
            Transform::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_y(0.5), Vec3::ONE),
            Transform::IDENTITY,
        ];

        extractor.zero_root_in_pose(&mut pose);
        assert!((pose[0].position.x).abs() < f32::EPSILON, "X should be zeroed");
        assert!((pose[0].position.z).abs() < f32::EPSILON, "Z should be zeroed");
        // Y should NOT be zeroed (extract_vertical defaults to false).
        assert!((pose[0].position.y - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn extractor_world_velocity() {
        let mut extractor = RootMotionExtractor::new(0);
        let clip = make_walk_clip();
        extractor.extract(&clip, 0.0, 0.5, 0.5);

        // Rotate entity 90 degrees around Y.
        let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let world_vel = extractor.world_velocity(rotation);
        // Forward (Z) motion should become sideways (X) in world space.
        assert!(world_vel.x.abs() > 0.1 || world_vel.z.abs() > 0.1);
    }

    #[test]
    fn blender_single_source() {
        let mut blender = RootMotionBlender::new();
        let motion = RootMotion {
            delta_position: Vec3::new(0.0, 0.0, 1.0),
            delta_rotation: Quat::IDENTITY,
        };
        blender.add(&motion, 1.0);
        let result = blender.result();
        assert!((result.delta_position.z - 1.0).abs() < 0.01);
    }

    #[test]
    fn blender_two_sources() {
        let mut blender = RootMotionBlender::new();
        let motion1 = RootMotion {
            delta_position: Vec3::new(0.0, 0.0, 2.0),
            delta_rotation: Quat::IDENTITY,
        };
        let motion2 = RootMotion {
            delta_position: Vec3::new(0.0, 0.0, 4.0),
            delta_rotation: Quat::IDENTITY,
        };
        blender.add(&motion1, 0.5);
        blender.add(&motion2, 0.5);
        let result = blender.result();
        // 0.5 * 2.0 + 0.5 * 4.0 = 3.0
        assert!((result.delta_position.z - 3.0).abs() < 0.01, "Blended Z should be 3.0, got {}", result.delta_position.z);
    }

    #[test]
    fn blender_reset() {
        let mut blender = RootMotionBlender::new();
        blender.add(
            &RootMotion {
                delta_position: Vec3::ONE,
                delta_rotation: Quat::IDENTITY,
            },
            1.0,
        );
        blender.reset();
        let result = blender.result();
        assert!((result.delta_position.length()).abs() < f32::EPSILON);
    }

    #[test]
    fn ground_alignment() {
        let mut extractor = RootMotionExtractor::new(0).with_ground_alignment(true);
        // Slope: 45 degrees forward.
        extractor.set_ground_normal(Vec3::new(0.0, 1.0, 1.0).normalize());

        let mut clip = AnimationClip::new("Walk", 1.0);
        let mut track = BoneTrack::new(0);
        track.position_keys = vec![
            Keyframe::new(0.0, Vec3::ZERO),
            Keyframe::new(1.0, Vec3::new(0.0, 0.0, 2.0)),
        ];
        track.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
        track.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
        clip.add_track(track);

        let motion = extractor.extract(&clip, 0.0, 1.0, 1.0);
        // The motion should be projected onto the slope.
        assert!(motion.delta_position.length() > 0.0);
    }
}
