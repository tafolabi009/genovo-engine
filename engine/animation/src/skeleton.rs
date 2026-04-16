//! Skeletal animation data structures and runtime player.
//!
//! Defines bones, skeletons, animation clips with keyframed tracks, and the
//! animation player that evaluates clips and produces final bone transforms.
//!
//! # Architecture
//!
//! The animation pipeline flows through these stages:
//!
//! 1. **Skeleton** -- static hierarchy of bones with bind poses.
//! 2. **AnimationClip** -- keyframed tracks per bone (position, rotation, scale).
//! 3. **AnimationPlayer** -- manages playback state, crossfading, and sampling.
//! 4. **Pose evaluation** -- samples clips at the current time to produce
//!    per-bone `Transform` values (local space).
//! 5. **Hierarchy walk** -- multiplies local transforms up the hierarchy to
//!    produce world-space bone matrices.
//! 6. **Skinning** -- multiplies world matrices by inverse-bind matrices to
//!    produce the final GPU-ready skinning matrices.

use std::collections::HashMap;

use genovo_core::Transform;
use glam::{Mat4, Quat, Vec3};
// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------

/// A skeleton definition consisting of a hierarchy of bones.
///
/// Skeletons are typically loaded from asset files (glTF, FBX) and shared
/// across all entities using the same mesh. Bones are stored in a flat array
/// with parent indices, and parent bones always appear before their children
/// (topological order).
#[derive(Debug, Clone)]
pub struct Skeleton {
    /// All bones in the skeleton, stored in a flat array.
    /// Parent bones always appear before their children.
    pub bones: Vec<Bone>,

    /// Fast name-to-index lookup map, built alongside the bone array.
    pub bone_names: HashMap<String, usize>,

    /// Index of the root bone in the `bones` array.
    pub root_bone_index: usize,

    /// Human-readable name (e.g. "Humanoid", "Spider").
    pub name: String,
}

impl Skeleton {
    /// Create a new skeleton from a list of bones.
    ///
    /// Automatically builds the `bone_names` lookup table. The bones must be
    /// in topological order (parents before children).
    pub fn new(name: impl Into<String>, bones: Vec<Bone>) -> Self {
        let bone_names: HashMap<String, usize> = bones
            .iter()
            .enumerate()
            .map(|(i, b)| (b.name.clone(), i))
            .collect();

        let root_bone_index = bones
            .iter()
            .position(|b| b.parent_index.is_none())
            .unwrap_or(0);

        Self {
            bones,
            bone_names,
            root_bone_index,
            name: name.into(),
        }
    }

    /// Number of bones in the skeleton.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Find a bone by name. Returns `None` if not found.
    pub fn find_bone(&self, name: &str) -> Option<usize> {
        self.bone_names.get(name).copied()
    }

    /// Get a reference to a bone by index.
    #[inline]
    pub fn bone(&self, index: usize) -> Option<&Bone> {
        self.bones.get(index)
    }

    /// Rebuild the `bone_names` lookup map from the current bones array.
    /// Call this after deserialization or manual modification of the bones.
    pub fn rebuild_name_map(&mut self) {
        self.bone_names.clear();
        for (i, bone) in self.bones.iter().enumerate() {
            self.bone_names.insert(bone.name.clone(), i);
        }
    }

    /// Returns the list of children for a given bone index.
    pub fn children_of(&self, bone_index: usize) -> Vec<usize> {
        self.bones
            .iter()
            .enumerate()
            .filter(|(_, b)| b.parent_index == Some(bone_index))
            .map(|(i, _)| i)
            .collect()
    }

    /// Computes a depth-first traversal order of all bone indices starting
    /// from the root. Useful for hierarchical processing.
    pub fn depth_first_order(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.bones.len());
        let mut stack = vec![self.root_bone_index];
        while let Some(idx) = stack.pop() {
            order.push(idx);
            // Push children in reverse so they come out in forward order.
            let children = self.children_of(idx);
            for &child in children.iter().rev() {
                stack.push(child);
            }
        }
        order
    }

    /// Compute the world-space bind pose transforms for all bones.
    ///
    /// This walks the hierarchy from root to leaves, multiplying each bone's
    /// local bind pose by its parent's world transform.
    pub fn compute_world_bind_poses(&self) -> Vec<Mat4> {
        let mut world_poses = vec![Mat4::IDENTITY; self.bones.len()];
        for (i, bone) in self.bones.iter().enumerate() {
            let local = bone.local_bind_pose.to_matrix();
            world_poses[i] = match bone.parent_index {
                Some(parent) => world_poses[parent] * local,
                None => local,
            };
        }
        world_poses
    }

    /// Compute world-space transforms from an array of local-space poses.
    ///
    /// `local_poses` must have the same length as `self.bones`. Each entry
    /// is the local-space transform for the bone at that index. The function
    /// walks the hierarchy and multiplies parent * child to produce world-space
    /// matrices.
    ///
    /// # Panics
    ///
    /// Panics if `local_poses.len() != self.bone_count()`.
    pub fn compute_world_transforms(&self, local_poses: &[Transform]) -> Vec<Mat4> {
        assert_eq!(
            local_poses.len(),
            self.bones.len(),
            "local_poses length ({}) must match bone count ({})",
            local_poses.len(),
            self.bones.len()
        );

        let mut world_transforms = vec![Mat4::IDENTITY; self.bones.len()];

        for (i, bone) in self.bones.iter().enumerate() {
            let local_mat = local_poses[i].to_matrix();
            world_transforms[i] = match bone.parent_index {
                Some(parent) => world_transforms[parent] * local_mat,
                None => local_mat,
            };
        }

        world_transforms
    }

    /// Compute GPU-ready skinning matrices from world-space transforms.
    ///
    /// Each skinning matrix is `world_transform * inverse_bind_matrix`, which
    /// transforms vertices from their bind-pose model space into the current
    /// animated world space for the GPU vertex shader.
    ///
    /// # Panics
    ///
    /// Panics if `world_transforms.len() != self.bone_count()`.
    pub fn compute_skin_matrices(&self, world_transforms: &[Mat4]) -> Vec<Mat4> {
        assert_eq!(
            world_transforms.len(),
            self.bones.len(),
            "world_transforms length ({}) must match bone count ({})",
            world_transforms.len(),
            self.bones.len()
        );

        world_transforms
            .iter()
            .zip(self.bones.iter())
            .map(|(world, bone)| *world * bone.inverse_bind_matrix)
            .collect()
    }

    /// Full pipeline: from local poses to skinning matrices.
    ///
    /// Equivalent to calling `compute_world_transforms` followed by
    /// `compute_skin_matrices`, but avoids a separate allocation.
    pub fn compute_skinning_matrices(&self, local_poses: &[Transform]) -> Vec<Mat4> {
        let world = self.compute_world_transforms(local_poses);
        self.compute_skin_matrices(&world)
    }

    /// Extract the bind pose as an array of `Transform` values (one per bone).
    pub fn bind_pose(&self) -> Vec<Transform> {
        self.bones.iter().map(|b| b.local_bind_pose).collect()
    }

    /// Validate the skeleton integrity.
    ///
    /// Returns a list of problems found (empty = valid). Checks:
    /// - Parent indices point to valid, earlier bones
    /// - Exactly one root bone
    /// - No duplicate names
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();

        let mut root_count = 0;
        let mut seen_names = HashMap::new();

        for (i, bone) in self.bones.iter().enumerate() {
            // Check parent index validity
            if let Some(parent) = bone.parent_index {
                if parent >= i {
                    problems.push(format!(
                        "Bone {} ('{}') has parent index {} which is not before it in the array",
                        i, bone.name, parent
                    ));
                }
                if parent >= self.bones.len() {
                    problems.push(format!(
                        "Bone {} ('{}') has out-of-bounds parent index {}",
                        i, bone.name, parent
                    ));
                }
            } else {
                root_count += 1;
            }

            // Check for duplicate names
            if let Some(prev) = seen_names.insert(bone.name.clone(), i) {
                problems.push(format!(
                    "Duplicate bone name '{}' at indices {} and {}",
                    bone.name, prev, i
                ));
            }
        }

        if root_count == 0 {
            problems.push("No root bone found (no bone with parent_index == None)".to_string());
        } else if root_count > 1 {
            problems.push(format!(
                "Multiple root bones found ({} bones with parent_index == None)",
                root_count
            ));
        }

        problems
    }

    /// Builds a chain of bone indices from `start_bone` up to `end_bone`
    /// by walking from `end_bone` toward the root until `start_bone` is found.
    ///
    /// Returns `None` if `start_bone` is not an ancestor of `end_bone`.
    pub fn build_chain(&self, start_bone: usize, end_bone: usize) -> Option<Vec<usize>> {
        let mut chain = Vec::new();
        let mut current = end_bone;
        loop {
            chain.push(current);
            if current == start_bone {
                chain.reverse();
                return Some(chain);
            }
            match self.bones[current].parent_index {
                Some(parent) => current = parent,
                None => return None, // reached root without finding start
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bone
// ---------------------------------------------------------------------------

/// A single bone in a skeleton hierarchy.
///
/// Bones are identified by their index in the skeleton's flat array. Each bone
/// stores its own name, parent reference, local bind pose, and precomputed
/// inverse-bind matrix for GPU skinning.
#[derive(Debug, Clone)]
pub struct Bone {
    /// Human-readable name (e.g. "LeftUpperArm", "Spine1").
    pub name: String,

    /// Index of the parent bone, or `None` for the root.
    pub parent_index: Option<usize>,

    /// Local-space bind pose transform (relative to parent).
    pub local_bind_pose: Transform,

    /// Inverse of the world-space bind pose transform. Used to transform
    /// vertices from model space to bone space for skinning.
    pub inverse_bind_matrix: Mat4,
}

impl Bone {
    /// Create a new bone with an explicit inverse-bind matrix.
    pub fn new(
        name: impl Into<String>,
        parent_index: Option<usize>,
        local_bind_pose: Transform,
        inverse_bind_matrix: Mat4,
    ) -> Self {
        Self {
            name: name.into(),
            parent_index,
            local_bind_pose,
            inverse_bind_matrix,
        }
    }

    /// Create a root bone (no parent) at identity.
    pub fn root(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parent_index: None,
            local_bind_pose: Transform::IDENTITY,
            inverse_bind_matrix: Mat4::IDENTITY,
        }
    }

    /// Create a child bone with a given parent and local offset.
    pub fn child(
        name: impl Into<String>,
        parent_index: usize,
        local_bind_pose: Transform,
    ) -> Self {
        Self {
            name: name.into(),
            parent_index: Some(parent_index),
            local_bind_pose,
            inverse_bind_matrix: Mat4::IDENTITY, // must be computed later
        }
    }
}

// ---------------------------------------------------------------------------
// Keyframe
// ---------------------------------------------------------------------------

/// A generic keyframe with a time stamp and value.
///
/// Keyframes should be sorted by time within a track. The interpolation
/// behavior depends on the `InterpolationMode` of the containing track.
#[derive(Debug, Clone)]
pub struct Keyframe<T> {
    /// Time in seconds from the start of the clip.
    pub time: f32,
    /// Value at this keyframe.
    pub value: T,
}

impl<T> Keyframe<T> {
    /// Create a new keyframe.
    pub fn new(time: f32, value: T) -> Self {
        Self { time, value }
    }
}

/// Interpolation method between keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// No interpolation; hold previous value until next keyframe.
    Step,
    /// Linear interpolation (lerp for Vec3, slerp for Quat).
    Linear,
    /// Cubic spline interpolation (smooth curves).
    CubicSpline,
}

impl Default for InterpolationMode {
    fn default() -> Self {
        Self::Linear
    }
}

// ---------------------------------------------------------------------------
// Bone track
// ---------------------------------------------------------------------------

/// Animation data for a single bone within a clip.
///
/// Each track stores separate keyframe arrays for position, rotation, and
/// scale, allowing independent sample rates per channel. Keyframes within
/// each array must be sorted by time.
#[derive(Debug, Clone)]
pub struct BoneTrack {
    /// Index of the bone this track animates.
    pub bone_index: usize,

    /// Position keyframes (translation relative to parent).
    pub position_keys: Vec<Keyframe<Vec3>>,

    /// Rotation keyframes (orientation relative to parent).
    pub rotation_keys: Vec<Keyframe<Quat>>,

    /// Scale keyframes.
    pub scale_keys: Vec<Keyframe<Vec3>>,

    /// Interpolation mode for position keys.
    pub position_interpolation: InterpolationMode,

    /// Interpolation mode for rotation keys.
    pub rotation_interpolation: InterpolationMode,

    /// Interpolation mode for scale keys.
    pub scale_interpolation: InterpolationMode,
}

impl BoneTrack {
    /// Create a new empty track for the given bone.
    pub fn new(bone_index: usize) -> Self {
        Self {
            bone_index,
            position_keys: Vec::new(),
            rotation_keys: Vec::new(),
            scale_keys: Vec::new(),
            position_interpolation: InterpolationMode::Linear,
            rotation_interpolation: InterpolationMode::Linear,
            scale_interpolation: InterpolationMode::Linear,
        }
    }

    /// Sample the position at the given time by interpolating keyframes.
    ///
    /// Uses binary search to find the surrounding keyframes and then
    /// interpolates according to the configured interpolation mode.
    pub fn sample_position(&self, time: f32) -> Vec3 {
        Self::sample_vec3(&self.position_keys, time, self.position_interpolation)
    }

    /// Sample the rotation at the given time by interpolating keyframes.
    ///
    /// Uses binary search and slerp for smooth rotational interpolation.
    pub fn sample_rotation(&self, time: f32) -> Quat {
        Self::sample_quat(&self.rotation_keys, time, self.rotation_interpolation)
    }

    /// Sample the scale at the given time by interpolating keyframes.
    pub fn sample_scale(&self, time: f32) -> Vec3 {
        Self::sample_vec3(&self.scale_keys, time, self.scale_interpolation)
    }

    /// Sample the full local-space transform at the given time.
    pub fn sample_transform(&self, time: f32) -> Transform {
        Transform::new(
            self.sample_position(time),
            self.sample_rotation(time),
            self.sample_scale(time),
        )
    }

    /// Binary search for the keyframe index whose time is <= `time`.
    /// Returns the index of the earlier keyframe in the pair that
    /// brackets `time`. If time is before the first keyframe, returns 0.
    /// If time is after the last, returns `keys.len() - 1`.
    fn binary_search_keyframe_index<T>(keys: &[Keyframe<T>], time: f32) -> usize {
        if keys.len() <= 1 {
            return 0;
        }

        // Binary search: find the rightmost keyframe with time <= `time`.
        let mut lo = 0usize;
        let mut hi = keys.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if keys[mid].time <= time {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // `lo` is now the index of the first keyframe with time > `time`.
        // We want the one before it.
        if lo == 0 {
            0
        } else {
            lo - 1
        }
    }

    /// Compute the interpolation factor `t` between two keyframes.
    fn compute_t(k0_time: f32, k1_time: f32, time: f32) -> f32 {
        let duration = k1_time - k0_time;
        if duration.abs() < f32::EPSILON {
            return 0.0;
        }
        ((time - k0_time) / duration).clamp(0.0, 1.0)
    }

    /// Hermite cubic spline interpolation for Vec3.
    fn cubic_hermite_vec3(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, t: f32) -> Vec3 {
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
    }

    fn sample_vec3(
        keys: &[Keyframe<Vec3>],
        time: f32,
        mode: InterpolationMode,
    ) -> Vec3 {
        if keys.is_empty() {
            return Vec3::ZERO;
        }
        if keys.len() == 1 || time <= keys[0].time {
            return keys[0].value;
        }
        let last_idx = keys.len() - 1;
        if time >= keys[last_idx].time {
            return keys[last_idx].value;
        }

        let i = Self::binary_search_keyframe_index(keys, time);
        let next = (i + 1).min(last_idx);

        if i == next {
            return keys[i].value;
        }

        let t = Self::compute_t(keys[i].time, keys[next].time, time);

        match mode {
            InterpolationMode::Step => keys[i].value,
            InterpolationMode::Linear => keys[i].value.lerp(keys[next].value, t),
            InterpolationMode::CubicSpline => {
                // Catmull-Rom style cubic using neighbors as tangent guides.
                let prev = if i > 0 { i - 1 } else { i };
                let next_next = if next + 1 < keys.len() { next + 1 } else { next };

                let dt = keys[next].time - keys[i].time;
                // Catmull-Rom tangents
                let m0 = (keys[next].value - keys[prev].value) * 0.5 * dt
                    / ((keys[next].time - keys[prev].time).max(f32::EPSILON));
                let m1 = (keys[next_next].value - keys[i].value) * 0.5 * dt
                    / ((keys[next_next].time - keys[i].time).max(f32::EPSILON));

                Self::cubic_hermite_vec3(keys[i].value, m0, keys[next].value, m1, t)
            }
        }
    }

    fn sample_quat(
        keys: &[Keyframe<Quat>],
        time: f32,
        mode: InterpolationMode,
    ) -> Quat {
        if keys.is_empty() {
            return Quat::IDENTITY;
        }
        if keys.len() == 1 || time <= keys[0].time {
            return keys[0].value;
        }
        let last_idx = keys.len() - 1;
        if time >= keys[last_idx].time {
            return keys[last_idx].value;
        }

        let i = Self::binary_search_keyframe_index(keys, time);
        let next = (i + 1).min(last_idx);

        if i == next {
            return keys[i].value;
        }

        let t = Self::compute_t(keys[i].time, keys[next].time, time);

        match mode {
            InterpolationMode::Step => keys[i].value,
            InterpolationMode::Linear => keys[i].value.slerp(keys[next].value, t),
            InterpolationMode::CubicSpline => {
                // Squad interpolation: use slerp as a reasonable approximation
                // for cubic quaternion interpolation. For a proper squad we need
                // intermediate control quaternions from neighbors.
                let prev = if i > 0 { i - 1 } else { i };
                let next_next = if next + 1 < keys.len() { next + 1 } else { next };

                // Compute inner control quaternions (squad tangents)
                let s_i = Self::squad_intermediate(
                    keys[prev].value,
                    keys[i].value,
                    keys[next].value,
                );
                let s_next = Self::squad_intermediate(
                    keys[i].value,
                    keys[next].value,
                    keys[next_next].value,
                );

                // Squad interpolation
                let slerp1 = keys[i].value.slerp(keys[next].value, t);
                let slerp2 = s_i.slerp(s_next, t);
                slerp1.slerp(slerp2, 2.0 * t * (1.0 - t))
            }
        }
    }

    /// Compute the squad intermediate quaternion for cubic quaternion interpolation.
    fn squad_intermediate(q_prev: Quat, q_curr: Quat, q_next: Quat) -> Quat {
        let inv = q_curr.conjugate();
        let log_prev = Self::quat_log(inv * q_prev);
        let log_next = Self::quat_log(inv * q_next);
        let sum = Vec3::new(
            log_prev.x + log_next.x,
            log_prev.y + log_next.y,
            log_prev.z + log_next.z,
        );
        q_curr * Self::quat_exp(sum * -0.25)
    }

    /// Approximate quaternion logarithm (returns a Vec3).
    fn quat_log(q: Quat) -> Vec3 {
        let v = Vec3::new(q.x, q.y, q.z);
        let s = v.length();
        if s < f32::EPSILON {
            return Vec3::ZERO;
        }
        let angle = s.atan2(q.w);
        v * (angle / s)
    }

    /// Approximate quaternion exponential (from a Vec3).
    fn quat_exp(v: Vec3) -> Quat {
        let angle = v.length();
        if angle < f32::EPSILON {
            return Quat::IDENTITY;
        }
        let axis = v / angle;
        let (s, c) = angle.sin_cos();
        Quat::from_xyzw(axis.x * s, axis.y * s, axis.z * s, c)
    }

    /// Sort all keyframe arrays by time. Call after building a track manually.
    pub fn sort_keyframes(&mut self) {
        self.position_keys
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        self.rotation_keys
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        self.scale_keys
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Returns the time range of this track (min, max) across all channels.
    pub fn time_range(&self) -> (f32, f32) {
        let mut min_time = f32::MAX;
        let mut max_time = f32::MIN;

        for k in &self.position_keys {
            min_time = min_time.min(k.time);
            max_time = max_time.max(k.time);
        }
        for k in &self.rotation_keys {
            min_time = min_time.min(k.time);
            max_time = max_time.max(k.time);
        }
        for k in &self.scale_keys {
            min_time = min_time.min(k.time);
            max_time = max_time.max(k.time);
        }

        if min_time > max_time {
            (0.0, 0.0)
        } else {
            (min_time, max_time)
        }
    }

    /// Returns the total number of keyframes across all channels.
    pub fn keyframe_count(&self) -> usize {
        self.position_keys.len() + self.rotation_keys.len() + self.scale_keys.len()
    }

    /// Returns true if this track has no keyframe data.
    pub fn is_empty(&self) -> bool {
        self.position_keys.is_empty()
            && self.rotation_keys.is_empty()
            && self.scale_keys.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Animation clip
// ---------------------------------------------------------------------------

/// A single animation clip (e.g. "Walk", "Run", "Attack").
///
/// Contains timed keyframe data for one or more bones in a skeleton.
/// Clips can be sampled at any time to produce local-space bone transforms.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    /// Human-readable name.
    pub name: String,

    /// Total duration of the clip in seconds.
    pub duration: f32,

    /// Sample rate the clip was authored at (used for compression reference).
    pub sample_rate: f32,

    /// Per-bone animation tracks.
    pub bone_tracks: Vec<BoneTrack>,

    /// Whether this clip is additive (applied on top of a base pose).
    pub is_additive: bool,

    /// Whether this clip should loop by default.
    pub looping: bool,
}

impl AnimationClip {
    /// Create a new animation clip.
    pub fn new(name: impl Into<String>, duration: f32) -> Self {
        Self {
            name: name.into(),
            duration,
            sample_rate: 30.0,
            bone_tracks: Vec::new(),
            is_additive: false,
            looping: false,
        }
    }

    /// Add a bone track to this clip.
    pub fn add_track(&mut self, track: BoneTrack) {
        self.bone_tracks.push(track);
    }

    /// Find the track for a given bone index.
    pub fn track_for_bone(&self, bone_index: usize) -> Option<&BoneTrack> {
        self.bone_tracks.iter().find(|t| t.bone_index == bone_index)
    }

    /// Find the track for a given bone index (mutable).
    pub fn track_for_bone_mut(&mut self, bone_index: usize) -> Option<&mut BoneTrack> {
        self.bone_tracks
            .iter_mut()
            .find(|t| t.bone_index == bone_index)
    }

    /// Evaluate all bone tracks at the given time.
    ///
    /// Returns a vector of `(bone_index, position, rotation, scale)` tuples,
    /// one per animated bone. Time is wrapped for looping clips or clamped
    /// for non-looping clips.
    pub fn sample(
        &self,
        time: f32,
    ) -> Vec<(usize, Vec3, Quat, Vec3)> {
        let clamped_time = if self.looping && self.duration > 0.0 {
            time.rem_euclid(self.duration)
        } else {
            time.clamp(0.0, self.duration)
        };

        self.bone_tracks
            .iter()
            .map(|track| {
                (
                    track.bone_index,
                    track.sample_position(clamped_time),
                    track.sample_rotation(clamped_time),
                    track.sample_scale(clamped_time),
                )
            })
            .collect()
    }

    /// Sample all tracks and write into a pre-allocated Transform array.
    ///
    /// `output` must be at least as large as the number of bones in the
    /// skeleton. Bones without tracks are left unchanged (typically set to
    /// the bind pose beforehand).
    pub fn sample_into(&self, time: f32, output: &mut [Transform]) {
        let clamped_time = if self.looping && self.duration > 0.0 {
            time.rem_euclid(self.duration)
        } else {
            time.clamp(0.0, self.duration)
        };

        for track in &self.bone_tracks {
            if track.bone_index < output.len() {
                output[track.bone_index] = track.sample_transform(clamped_time);
            }
        }
    }

    /// Sample and produce a full pose as a `Vec<Transform>` for a skeleton
    /// with `bone_count` bones. Bones without tracks get the identity pose.
    pub fn sample_pose(&self, time: f32, bone_count: usize) -> Vec<Transform> {
        let mut pose = vec![Transform::IDENTITY; bone_count];
        self.sample_into(time, &mut pose);
        pose
    }

    /// Compute the total number of keyframes in all tracks.
    pub fn total_keyframes(&self) -> usize {
        self.bone_tracks.iter().map(|t| t.keyframe_count()).sum()
    }

    /// Get the normalized playback position [0, 1] for a given time.
    pub fn normalized_time(&self, time: f32) -> f32 {
        if self.duration <= 0.0 {
            return 0.0;
        }
        (time / self.duration).clamp(0.0, 1.0)
    }

    /// Create a reversed copy of this clip.
    pub fn reversed(&self) -> Self {
        let mut clip = self.clone();
        for track in &mut clip.bone_tracks {
            for k in &mut track.position_keys {
                k.time = self.duration - k.time;
            }
            for k in &mut track.rotation_keys {
                k.time = self.duration - k.time;
            }
            for k in &mut track.scale_keys {
                k.time = self.duration - k.time;
            }
            track.sort_keyframes();
        }
        clip.name = format!("{}_reversed", self.name);
        clip
    }

    /// Create an additive clip by computing the difference from a reference pose.
    ///
    /// For each track, the additive keyframes store the difference from the
    /// first keyframe (reference frame), so when applied on top of any base
    /// pose, it adds the motion delta.
    pub fn make_additive(&self) -> Self {
        let mut additive = self.clone();
        additive.is_additive = true;
        additive.name = format!("{}_additive", self.name);

        for track in &mut additive.bone_tracks {
            // Reference = first keyframe values
            let ref_pos = track
                .position_keys
                .first()
                .map(|k| k.value)
                .unwrap_or(Vec3::ZERO);
            let ref_rot = track
                .rotation_keys
                .first()
                .map(|k| k.value)
                .unwrap_or(Quat::IDENTITY);
            let ref_scale = track
                .scale_keys
                .first()
                .map(|k| k.value)
                .unwrap_or(Vec3::ONE);

            for k in &mut track.position_keys {
                k.value -= ref_pos;
            }
            for k in &mut track.rotation_keys {
                k.value = ref_rot.conjugate() * k.value;
            }
            for k in &mut track.scale_keys {
                // Multiplicative delta for scale
                k.value = Vec3::new(
                    if ref_scale.x.abs() > f32::EPSILON {
                        k.value.x / ref_scale.x
                    } else {
                        1.0
                    },
                    if ref_scale.y.abs() > f32::EPSILON {
                        k.value.y / ref_scale.y
                    } else {
                        1.0
                    },
                    if ref_scale.z.abs() > f32::EPSILON {
                        k.value.z / ref_scale.z
                    } else {
                        1.0
                    },
                );
            }
        }

        additive
    }
}

// ---------------------------------------------------------------------------
// Animation player
// ---------------------------------------------------------------------------

/// Current state of an animation player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerState {
    /// No clip is playing; output is the last sampled pose or bind pose.
    Stopped,
    /// Actively advancing playback each frame.
    Playing,
    /// Playback is paused at the current time; can be resumed.
    Paused,
}

/// Runtime animation playback controller.
///
/// Manages the current clip, playback time, speed, crossfading, and
/// produces the final sampled pose each frame. The player does not own
/// clip data -- it references clips by index into an external collection.
///
/// # Crossfading
///
/// When `crossfade()` is called, the player smoothly blends from the
/// old clip to the new one over the specified duration. During the fade,
/// both clips advance in parallel and their sampled poses are blended.
#[derive(Debug)]
pub struct AnimationPlayer {
    /// Currently playing clip (index into an external clip collection).
    pub current_clip: Option<usize>,

    /// Playback time within the current clip, in seconds.
    pub playback_time: f32,

    /// Playback speed multiplier (1.0 = normal, 0.5 = half, -1.0 = reverse).
    pub speed: f32,

    /// Current playback state.
    pub state: PlayerState,

    /// Whether to loop the current clip.
    pub looping: bool,

    // -- Crossfade state --
    /// Clip being faded out during a crossfade.
    pub crossfade_from_clip: Option<usize>,

    /// Playback time in the fading-out clip.
    pub crossfade_from_time: f32,

    /// Total crossfade duration in seconds.
    pub crossfade_duration: f32,

    /// Elapsed time within the crossfade.
    pub crossfade_elapsed: f32,

    /// Whether a crossfade is currently in progress.
    pub crossfading: bool,

    /// Weight of the current clip during crossfade [0.0, 1.0].
    pub crossfade_weight: f32,

    /// Queue of clips to play after the current one finishes (non-looping).
    pub queue: Vec<usize>,

    /// Whether playback has reached the end of a non-looping clip.
    pub finished: bool,
}

impl Default for AnimationPlayer {
    fn default() -> Self {
        Self {
            current_clip: None,
            playback_time: 0.0,
            speed: 1.0,
            state: PlayerState::Stopped,
            looping: true,
            crossfade_from_clip: None,
            crossfade_from_time: 0.0,
            crossfade_duration: 0.2,
            crossfade_elapsed: 0.0,
            crossfading: false,
            crossfade_weight: 1.0,
            queue: Vec::new(),
            finished: false,
        }
    }
}

impl AnimationPlayer {
    /// Create a new animation player.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start playing the given clip index from the beginning.
    pub fn play(&mut self, clip_index: usize) {
        self.current_clip = Some(clip_index);
        self.playback_time = 0.0;
        self.state = PlayerState::Playing;
        self.crossfading = false;
        self.finished = false;
    }

    /// Start playing the clip from a specific time.
    pub fn play_from(&mut self, clip_index: usize, start_time: f32) {
        self.current_clip = Some(clip_index);
        self.playback_time = start_time;
        self.state = PlayerState::Playing;
        self.crossfading = false;
        self.finished = false;
    }

    /// Queue a clip to play after the current clip finishes.
    pub fn queue_clip(&mut self, clip_index: usize) {
        self.queue.push(clip_index);
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        if self.state == PlayerState::Playing {
            self.state = PlayerState::Paused;
        }
    }

    /// Resume playback.
    pub fn resume(&mut self) {
        if self.state == PlayerState::Paused {
            self.state = PlayerState::Playing;
        }
    }

    /// Stop playback and reset time.
    pub fn stop(&mut self) {
        self.state = PlayerState::Stopped;
        self.playback_time = 0.0;
        self.crossfading = false;
        self.finished = false;
        self.queue.clear();
    }

    /// Begin crossfading from the current clip to a new clip.
    ///
    /// The old clip continues to advance during the fade while the new
    /// clip starts from time 0. The blend weight transitions from 0 (old)
    /// to 1 (new) over `duration` seconds.
    pub fn crossfade(&mut self, new_clip_index: usize, duration: f32) {
        if self.current_clip == Some(new_clip_index) {
            return;
        }
        self.crossfade_from_clip = self.current_clip;
        self.crossfade_from_time = self.playback_time;
        self.current_clip = Some(new_clip_index);
        self.playback_time = 0.0;
        self.crossfade_duration = duration.max(0.001);
        self.crossfade_elapsed = 0.0;
        self.crossfading = true;
        self.crossfade_weight = 0.0;
        self.state = PlayerState::Playing;
        self.finished = false;
    }

    /// Set the playback speed (1.0 = normal, 0.5 = half speed, -1.0 = reverse).
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Set whether the current clip should loop.
    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Whether the player is currently in a crossfade.
    pub fn is_crossfading(&self) -> bool {
        self.crossfading
    }

    /// Whether the player has finished a non-looping clip.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Whether the player is currently playing (not stopped or paused).
    pub fn is_playing(&self) -> bool {
        self.state == PlayerState::Playing
    }

    /// Advance playback by `dt` seconds.
    ///
    /// Updates the playback time, handles looping/clamping, advances
    /// crossfade state, and processes the playback queue.
    ///
    /// Returns the current crossfade weight (1.0 = fully new clip, 0.0 = fully old clip).
    pub fn advance(&mut self, dt: f32, clip_duration: f32) -> f32 {
        if self.state != PlayerState::Playing {
            return self.crossfade_weight;
        }

        let effective_dt = dt * self.speed;
        self.playback_time += effective_dt;

        // Handle looping and end-of-clip.
        if self.looping {
            if clip_duration > 0.0 {
                self.playback_time = self.playback_time.rem_euclid(clip_duration);
            }
        } else if self.playback_time >= clip_duration {
            self.playback_time = clip_duration;
            self.finished = true;

            // Check the queue for the next clip.
            if let Some(next_clip) = self.queue.first().copied() {
                self.queue.remove(0);
                self.play(next_clip);
            } else {
                self.state = PlayerState::Stopped;
            }
        } else if self.playback_time < 0.0 {
            self.playback_time = 0.0;
            self.finished = true;
            self.state = PlayerState::Stopped;
        }

        // Advance crossfade.
        if self.crossfading {
            self.crossfade_elapsed += dt;
            self.crossfade_weight =
                (self.crossfade_elapsed / self.crossfade_duration).clamp(0.0, 1.0);

            self.crossfade_from_time += effective_dt;

            if self.crossfade_elapsed >= self.crossfade_duration {
                self.crossfading = false;
                self.crossfade_from_clip = None;
                self.crossfade_weight = 1.0;
            }
        }

        self.crossfade_weight
    }

    /// Sample the current clip(s) and produce a blended pose.
    ///
    /// This is the main sampling entry point. It handles crossfading by
    /// blending between the old and new clip poses.
    pub fn sample(&self, clips: &[AnimationClip], bone_count: usize) -> Vec<Transform> {
        let mut pose = vec![Transform::IDENTITY; bone_count];

        if let Some(clip_idx) = self.current_clip {
            if clip_idx < clips.len() {
                clips[clip_idx].sample_into(self.playback_time, &mut pose);
            }
        }

        if self.crossfading {
            if let Some(from_idx) = self.crossfade_from_clip {
                if from_idx < clips.len() {
                    let mut from_pose = vec![Transform::IDENTITY; bone_count];
                    clips[from_idx].sample_into(self.crossfade_from_time, &mut from_pose);

                    // Blend: from_pose * (1 - weight) + current_pose * weight
                    let w = self.crossfade_weight;
                    for i in 0..bone_count {
                        pose[i] = from_pose[i].lerp(&pose[i], w);
                    }
                }
            }
        }

        pose
    }
}

// ---------------------------------------------------------------------------
// Pose utility functions
// ---------------------------------------------------------------------------

/// Blend two poses together.
///
/// For each bone, position and scale are lerped, rotation is slerped.
/// `t = 0.0` returns `a`, `t = 1.0` returns `b`.
pub fn blend_poses(a: &[Transform], b: &[Transform], t: f32) -> Vec<Transform> {
    assert_eq!(a.len(), b.len(), "Pose lengths must match");
    let t = t.clamp(0.0, 1.0);
    a.iter()
        .zip(b.iter())
        .map(|(ta, tb)| ta.lerp(tb, t))
        .collect()
}

/// Apply an additive pose on top of a base pose.
///
/// For each bone:
/// - Position: `base.position + additive.position * weight`
/// - Rotation: `base.rotation * slerp(IDENTITY, additive.rotation, weight)`
/// - Scale: `base.scale * lerp(ONE, additive.scale, weight)` (multiplicative)
pub fn additive_blend(
    base: &[Transform],
    additive: &[Transform],
    weight: f32,
) -> Vec<Transform> {
    assert_eq!(base.len(), additive.len(), "Pose lengths must match");
    let w = weight.clamp(0.0, 1.0);
    base.iter()
        .zip(additive.iter())
        .map(|(b, a)| {
            Transform::new(
                b.position + a.position * w,
                b.rotation * Quat::IDENTITY.slerp(a.rotation, w),
                b.scale * Vec3::ONE.lerp(a.scale, w),
            )
        })
        .collect()
}

/// Apply a masked override: replace specific bones from the override pose,
/// leaving other bones from the base pose.
pub fn masked_blend(
    base: &[Transform],
    override_pose: &[Transform],
    mask: &[usize],
    weight: f32,
) -> Vec<Transform> {
    assert_eq!(base.len(), override_pose.len(), "Pose lengths must match");
    let w = weight.clamp(0.0, 1.0);
    let mut result = base.to_vec();
    for &bone_idx in mask {
        if bone_idx < result.len() {
            result[bone_idx] = base[bone_idx].lerp(&override_pose[bone_idx], w);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Skinned mesh renderer
// ---------------------------------------------------------------------------

/// Component data for rendering a skinned (skeletal) mesh.
///
/// Holds the final bone matrices that are uploaded to the GPU each frame
/// for vertex skinning in the shader.
#[derive(Debug, Clone)]
pub struct SkinnedMeshRenderer {
    /// Reference to the skeleton asset.
    pub skeleton_name: String,

    /// Final bone matrices (model-space * inverse-bind-matrix) ready for the
    /// GPU skinning shader. Length equals the skeleton's bone count.
    pub bone_matrices: Vec<Mat4>,

    /// Maximum number of bones influencing each vertex (typically 4).
    pub max_bones_per_vertex: u8,

    /// Whether to update bone matrices this frame.
    pub enabled: bool,
}

impl Default for SkinnedMeshRenderer {
    fn default() -> Self {
        Self {
            skeleton_name: String::new(),
            bone_matrices: Vec::new(),
            max_bones_per_vertex: 4,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Animation event
// ---------------------------------------------------------------------------

/// An event that fires at a specific time during an animation clip.
///
/// Used for gameplay synchronization (footstep sounds, weapon hit frames,
/// particle spawning, etc.).
#[derive(Debug, Clone)]
pub struct AnimationEvent {
    /// Time in seconds from the start of the clip.
    pub time: f32,
    /// Event name (e.g. "foot_plant_left", "attack_hit").
    pub name: String,
    /// Optional integer parameter.
    pub int_param: i32,
    /// Optional float parameter.
    pub float_param: f32,
    /// Optional string parameter.
    pub string_param: String,
}

impl AnimationEvent {
    /// Create a new animation event.
    pub fn new(time: f32, name: impl Into<String>) -> Self {
        Self {
            time,
            name: name.into(),
            int_param: 0,
            float_param: 0.0,
            string_param: String::new(),
        }
    }
}

/// Collects events that fired during a frame.
pub fn collect_events(
    events: &[AnimationEvent],
    prev_time: f32,
    curr_time: f32,
    looping: bool,
    duration: f32,
) -> Vec<&AnimationEvent> {
    if events.is_empty() {
        return Vec::new();
    }

    let mut fired = Vec::new();

    if looping && curr_time < prev_time {
        // Wrapped around: collect events from prev_time to end, then 0 to curr_time.
        for event in events {
            if event.time >= prev_time && event.time <= duration {
                fired.push(event);
            }
            if event.time >= 0.0 && event.time <= curr_time {
                fired.push(event);
            }
        }
    } else {
        let (start, end) = if prev_time <= curr_time {
            (prev_time, curr_time)
        } else {
            (curr_time, prev_time)
        };
        for event in events {
            if event.time > start && event.time <= end {
                fired.push(event);
            }
        }
    }

    fired
}

// ---------------------------------------------------------------------------
// Root motion
// ---------------------------------------------------------------------------

/// Extracted root motion from an animation clip.
///
/// Root motion removes the root bone's translation/rotation from the
/// animation and provides it separately so the game can apply it to the
/// entity's world transform (for physics-accurate movement).
#[derive(Debug, Clone, Copy)]
pub struct RootMotion {
    /// Delta translation this frame.
    pub delta_position: Vec3,
    /// Delta rotation this frame.
    pub delta_rotation: Quat,
}

impl Default for RootMotion {
    fn default() -> Self {
        Self {
            delta_position: Vec3::ZERO,
            delta_rotation: Quat::IDENTITY,
        }
    }
}

impl RootMotion {
    /// Extract root motion from a clip between two time points.
    ///
    /// Samples the root bone track at both times and returns the delta.
    pub fn extract(clip: &AnimationClip, root_bone_index: usize, prev_time: f32, curr_time: f32) -> Self {
        let track = match clip.track_for_bone(root_bone_index) {
            Some(t) => t,
            None => return Self::default(),
        };

        let prev_pos = track.sample_position(prev_time);
        let curr_pos = track.sample_position(curr_time);
        let prev_rot = track.sample_rotation(prev_time);
        let curr_rot = track.sample_rotation(curr_time);

        Self {
            delta_position: curr_pos - prev_pos,
            delta_rotation: prev_rot.conjugate() * curr_rot,
        }
    }
}

// ---------------------------------------------------------------------------
// Bone mask
// ---------------------------------------------------------------------------

/// A mask that selects which bones are affected by a blending operation.
///
/// Stores per-bone weights allowing partial blending (e.g., upper body
/// override at 100% while lower body stays at 0%).
#[derive(Debug, Clone)]
pub struct BoneMask {
    /// Per-bone weight [0.0, 1.0]. Length must match skeleton bone count.
    pub weights: Vec<f32>,
    /// Human-readable name for the mask (e.g. "UpperBody", "LeftArm").
    pub name: String,
}

impl BoneMask {
    /// Create a mask with all bones at zero weight.
    pub fn empty(bone_count: usize) -> Self {
        Self {
            weights: vec![0.0; bone_count],
            name: String::new(),
        }
    }

    /// Create a mask with all bones at full weight.
    pub fn full(bone_count: usize) -> Self {
        Self {
            weights: vec![1.0; bone_count],
            name: String::new(),
        }
    }

    /// Create a mask from a set of bone indices (included bones get weight 1.0).
    pub fn from_indices(bone_count: usize, indices: &[usize]) -> Self {
        let mut weights = vec![0.0; bone_count];
        for &idx in indices {
            if idx < bone_count {
                weights[idx] = 1.0;
            }
        }
        Self {
            weights,
            name: String::new(),
        }
    }

    /// Create a mask that includes a bone and all its descendants.
    pub fn from_bone_and_descendants(skeleton: &Skeleton, root_bone: usize) -> Self {
        let mut weights = vec![0.0; skeleton.bone_count()];
        let mut stack = vec![root_bone];
        while let Some(idx) = stack.pop() {
            if idx < weights.len() {
                weights[idx] = 1.0;
                for child in skeleton.children_of(idx) {
                    stack.push(child);
                }
            }
        }
        Self {
            weights,
            name: String::new(),
        }
    }

    /// Invert the mask: 0 becomes 1, 1 becomes 0.
    pub fn inverted(&self) -> Self {
        Self {
            weights: self.weights.iter().map(|w| 1.0 - w).collect(),
            name: format!("{}_inverted", self.name),
        }
    }

    /// Get the weight for a specific bone.
    pub fn weight(&self, bone_index: usize) -> f32 {
        self.weights.get(bone_index).copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a simple 3-bone skeleton (root -> spine -> head).
    fn make_test_skeleton() -> Skeleton {
        let bones = vec![
            Bone::new(
                "Root",
                None,
                Transform::IDENTITY,
                Mat4::IDENTITY,
            ),
            Bone::new(
                "Spine",
                Some(0),
                Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
                Mat4::IDENTITY,
            ),
            Bone::new(
                "Head",
                Some(1),
                Transform::from_position(Vec3::new(0.0, 0.5, 0.0)),
                Mat4::IDENTITY,
            ),
        ];
        Skeleton::new("TestSkeleton", bones)
    }

    /// Helper: build a 5-bone humanoid-like skeleton.
    fn make_humanoid_skeleton() -> Skeleton {
        let bones = vec![
            Bone::new("Hips", None, Transform::IDENTITY, Mat4::IDENTITY),
            Bone::new(
                "Spine",
                Some(0),
                Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
                Mat4::IDENTITY,
            ),
            Bone::new(
                "LeftArm",
                Some(1),
                Transform::from_position(Vec3::new(-0.5, 0.5, 0.0)),
                Mat4::IDENTITY,
            ),
            Bone::new(
                "RightArm",
                Some(1),
                Transform::from_position(Vec3::new(0.5, 0.5, 0.0)),
                Mat4::IDENTITY,
            ),
            Bone::new(
                "Head",
                Some(1),
                Transform::from_position(Vec3::new(0.0, 0.5, 0.0)),
                Mat4::IDENTITY,
            ),
        ];
        Skeleton::new("Humanoid", bones)
    }

    /// Helper: build a simple walk animation clip.
    fn make_walk_clip(bone_count: usize) -> AnimationClip {
        let mut clip = AnimationClip::new("Walk", 1.0);
        clip.looping = true;

        for bone_idx in 0..bone_count {
            let mut track = BoneTrack::new(bone_idx);
            track.position_keys = vec![
                Keyframe::new(0.0, Vec3::ZERO),
                Keyframe::new(0.5, Vec3::new(0.0, 0.1, 0.0)),
                Keyframe::new(1.0, Vec3::ZERO),
            ];
            track.rotation_keys = vec![
                Keyframe::new(0.0, Quat::IDENTITY),
                Keyframe::new(1.0, Quat::IDENTITY),
            ];
            track.scale_keys = vec![
                Keyframe::new(0.0, Vec3::ONE),
                Keyframe::new(1.0, Vec3::ONE),
            ];
            clip.add_track(track);
        }
        clip
    }

    // -- Skeleton tests --

    #[test]
    fn test_skeleton_creation() {
        let skel = make_test_skeleton();
        assert_eq!(skel.bone_count(), 3);
        assert_eq!(skel.root_bone_index, 0);
        assert_eq!(skel.name, "TestSkeleton");
    }

    #[test]
    fn test_find_bone() {
        let skel = make_test_skeleton();
        assert_eq!(skel.find_bone("Root"), Some(0));
        assert_eq!(skel.find_bone("Spine"), Some(1));
        assert_eq!(skel.find_bone("Head"), Some(2));
        assert_eq!(skel.find_bone("NonExistent"), None);
    }

    #[test]
    fn test_children_of() {
        let skel = make_humanoid_skeleton();
        let root_children = skel.children_of(0);
        assert_eq!(root_children, vec![1]); // Spine
        let spine_children = skel.children_of(1);
        assert_eq!(spine_children, vec![2, 3, 4]); // LeftArm, RightArm, Head
    }

    #[test]
    fn test_depth_first_order() {
        let skel = make_humanoid_skeleton();
        let order = skel.depth_first_order();
        assert_eq!(order.len(), 5);
        assert_eq!(order[0], 0); // Hips (root)
        assert_eq!(order[1], 1); // Spine
    }

    #[test]
    fn test_compute_world_transforms() {
        let skel = make_test_skeleton();
        let local_poses = vec![
            Transform::IDENTITY,
            Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
            Transform::from_position(Vec3::new(0.0, 0.5, 0.0)),
        ];
        let world = skel.compute_world_transforms(&local_poses);
        assert_eq!(world.len(), 3);

        // Root is identity
        let root_pos = world[0].col(3);
        assert!((root_pos.x).abs() < f32::EPSILON);

        // Spine should be at Y=1
        let spine_pos = world[1].col(3);
        assert!((spine_pos.y - 1.0).abs() < f32::EPSILON);

        // Head should be at Y=1.5 (1.0 + 0.5)
        let head_pos = world[2].col(3);
        assert!((head_pos.y - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_skin_matrices() {
        let skel = make_test_skeleton();
        let world = vec![Mat4::IDENTITY; 3];
        let skin = skel.compute_skin_matrices(&world);
        assert_eq!(skin.len(), 3);
        // With identity inverse-bind and identity world, skin = identity
        for m in &skin {
            assert!((m.col(3).x).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_skeleton_validate() {
        let skel = make_test_skeleton();
        let problems = skel.validate();
        assert!(problems.is_empty(), "Valid skeleton should have no problems");
    }

    #[test]
    fn test_skeleton_validate_duplicate_names() {
        let bones = vec![
            Bone::new("Root", None, Transform::IDENTITY, Mat4::IDENTITY),
            Bone::new("Root", Some(0), Transform::IDENTITY, Mat4::IDENTITY),
        ];
        let skel = Skeleton::new("Bad", bones);
        let problems = skel.validate();
        assert!(!problems.is_empty());
    }

    #[test]
    fn test_build_chain() {
        let skel = make_test_skeleton();
        let chain = skel.build_chain(0, 2).unwrap();
        assert_eq!(chain, vec![0, 1, 2]);

        // Cannot build chain if start is not ancestor of end
        let bad_chain = skel.build_chain(2, 0);
        assert!(bad_chain.is_none());
    }

    // -- Keyframe / BoneTrack tests --

    #[test]
    fn test_sample_position_empty() {
        let track = BoneTrack::new(0);
        let pos = track.sample_position(0.5);
        assert_eq!(pos, Vec3::ZERO);
    }

    #[test]
    fn test_sample_position_single_key() {
        let mut track = BoneTrack::new(0);
        track.position_keys = vec![Keyframe::new(0.0, Vec3::new(1.0, 2.0, 3.0))];
        let pos = track.sample_position(0.5);
        assert_eq!(pos, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_sample_position_linear() {
        let mut track = BoneTrack::new(0);
        track.position_keys = vec![
            Keyframe::new(0.0, Vec3::ZERO),
            Keyframe::new(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ];
        let pos = track.sample_position(0.5);
        assert!((pos.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_sample_position_before_first() {
        let mut track = BoneTrack::new(0);
        track.position_keys = vec![
            Keyframe::new(0.5, Vec3::new(1.0, 0.0, 0.0)),
            Keyframe::new(1.0, Vec3::new(2.0, 0.0, 0.0)),
        ];
        let pos = track.sample_position(0.0);
        assert_eq!(pos, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_sample_position_after_last() {
        let mut track = BoneTrack::new(0);
        track.position_keys = vec![
            Keyframe::new(0.0, Vec3::new(1.0, 0.0, 0.0)),
            Keyframe::new(0.5, Vec3::new(2.0, 0.0, 0.0)),
        ];
        let pos = track.sample_position(1.0);
        assert_eq!(pos, Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn test_sample_position_step_mode() {
        let mut track = BoneTrack::new(0);
        track.position_interpolation = InterpolationMode::Step;
        track.position_keys = vec![
            Keyframe::new(0.0, Vec3::ZERO),
            Keyframe::new(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ];
        let pos = track.sample_position(0.5);
        assert_eq!(pos, Vec3::ZERO); // Step holds previous value
    }

    #[test]
    fn test_sample_rotation_slerp() {
        let mut track = BoneTrack::new(0);
        track.rotation_keys = vec![
            Keyframe::new(0.0, Quat::IDENTITY),
            Keyframe::new(1.0, Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
        ];
        let rot = track.sample_rotation(0.5);
        let expected = Quat::IDENTITY
            .slerp(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2), 0.5);
        let diff = rot.dot(expected).abs();
        assert!(diff > 0.999, "Rotation slerp should be close");
    }

    #[test]
    fn test_sample_position_multiple_keyframes() {
        let mut track = BoneTrack::new(0);
        track.position_keys = vec![
            Keyframe::new(0.0, Vec3::ZERO),
            Keyframe::new(0.25, Vec3::new(5.0, 0.0, 0.0)),
            Keyframe::new(0.5, Vec3::new(10.0, 0.0, 0.0)),
            Keyframe::new(0.75, Vec3::new(5.0, 0.0, 0.0)),
            Keyframe::new(1.0, Vec3::ZERO),
        ];

        // At 0.125, should be between 0 and 5 -> 2.5
        let pos = track.sample_position(0.125);
        assert!((pos.x - 2.5).abs() < 0.01);

        // At 0.5, should be exactly 10
        let pos = track.sample_position(0.5);
        assert!((pos.x - 10.0).abs() < 0.01);

        // At 0.875, should be between 5 and 0 -> 2.5
        let pos = track.sample_position(0.875);
        assert!((pos.x - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_bone_track_sort_keyframes() {
        let mut track = BoneTrack::new(0);
        track.position_keys = vec![
            Keyframe::new(1.0, Vec3::X),
            Keyframe::new(0.0, Vec3::ZERO),
            Keyframe::new(0.5, Vec3::Y),
        ];
        track.sort_keyframes();
        assert!((track.position_keys[0].time).abs() < f32::EPSILON);
        assert!((track.position_keys[1].time - 0.5).abs() < f32::EPSILON);
        assert!((track.position_keys[2].time - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bone_track_time_range() {
        let mut track = BoneTrack::new(0);
        track.position_keys = vec![
            Keyframe::new(0.1, Vec3::ZERO),
            Keyframe::new(0.9, Vec3::X),
        ];
        track.rotation_keys = vec![
            Keyframe::new(0.0, Quat::IDENTITY),
            Keyframe::new(1.0, Quat::IDENTITY),
        ];
        let (min, max) = track.time_range();
        assert!((min - 0.0).abs() < f32::EPSILON);
        assert!((max - 1.0).abs() < f32::EPSILON);
    }

    // -- AnimationClip tests --

    #[test]
    fn test_clip_sample() {
        let clip = make_walk_clip(3);
        let samples = clip.sample(0.5);
        assert_eq!(samples.len(), 3);
        for (_, pos, _, _) in &samples {
            assert!((pos.y - 0.1).abs() < 0.01, "At t=0.5, position should peak");
        }
    }

    #[test]
    fn test_clip_sample_looping() {
        let clip = make_walk_clip(3);
        // At t=1.5 (looping with duration 1.0), should be same as t=0.5
        let samples = clip.sample(1.5);
        assert_eq!(samples.len(), 3);
        for (_, pos, _, _) in &samples {
            assert!((pos.y - 0.1).abs() < 0.01);
        }
    }

    #[test]
    fn test_clip_sample_pose() {
        let clip = make_walk_clip(3);
        let pose = clip.sample_pose(0.0, 3);
        assert_eq!(pose.len(), 3);
        for t in &pose {
            assert!((t.position - Vec3::ZERO).length() < 0.01);
        }
    }

    #[test]
    fn test_clip_reversed() {
        let clip = make_walk_clip(1);
        let reversed = clip.reversed();
        assert_eq!(reversed.name, "Walk_reversed");
        // The peak position should now be at t=0.5 from the other direction
        let sample_orig = clip.sample(0.25);
        let sample_rev = reversed.sample(0.75);
        assert!((sample_orig[0].1 - sample_rev[0].1).length() < 0.01);
    }

    #[test]
    fn test_clip_normalized_time() {
        let clip = AnimationClip::new("Test", 2.0);
        assert!((clip.normalized_time(1.0) - 0.5).abs() < f32::EPSILON);
        assert!((clip.normalized_time(0.0) - 0.0).abs() < f32::EPSILON);
        assert!((clip.normalized_time(2.0) - 1.0).abs() < f32::EPSILON);
    }

    // -- AnimationPlayer tests --

    #[test]
    fn test_player_play() {
        let mut player = AnimationPlayer::new();
        player.play(0);
        assert_eq!(player.current_clip, Some(0));
        assert_eq!(player.state, PlayerState::Playing);
        assert!((player.playback_time).abs() < f32::EPSILON);
    }

    #[test]
    fn test_player_pause_resume() {
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.pause();
        assert_eq!(player.state, PlayerState::Paused);
        player.resume();
        assert_eq!(player.state, PlayerState::Playing);
    }

    #[test]
    fn test_player_stop() {
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.advance(0.5, 1.0);
        player.stop();
        assert_eq!(player.state, PlayerState::Stopped);
        assert!((player.playback_time).abs() < f32::EPSILON);
    }

    #[test]
    fn test_player_advance() {
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.set_looping(true);
        player.advance(0.5, 1.0);
        assert!((player.playback_time - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_player_advance_looping() {
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.set_looping(true);
        player.advance(1.5, 1.0);
        assert!((player.playback_time - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_player_advance_non_looping() {
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.set_looping(false);
        player.advance(1.5, 1.0);
        assert!((player.playback_time - 1.0).abs() < 0.01);
        assert!(player.is_finished());
    }

    #[test]
    fn test_player_crossfade() {
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.advance(0.5, 1.0);
        player.crossfade(1, 0.5);
        assert_eq!(player.current_clip, Some(1));
        assert_eq!(player.crossfade_from_clip, Some(0));
        assert!(player.crossfading);

        // Advance partway through the fade
        player.advance(0.25, 1.0);
        assert!(player.crossfading);
        assert!(player.crossfade_weight > 0.0 && player.crossfade_weight < 1.0);

        // Advance past the fade
        player.advance(0.3, 1.0);
        assert!(!player.crossfading);
        assert!((player.crossfade_weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_player_crossfade_same_clip() {
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.crossfade(0, 0.5);
        // Should not start a crossfade to the same clip
        assert!(!player.crossfading);
    }

    #[test]
    fn test_player_speed() {
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.set_speed(2.0);
        player.set_looping(false);
        player.advance(0.25, 1.0);
        // At 2x speed, 0.25s of real time = 0.5s of playback
        assert!((player.playback_time - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_player_sample() {
        let clips = vec![make_walk_clip(3)];
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.advance(0.5, 1.0);
        let pose = player.sample(&clips, 3);
        assert_eq!(pose.len(), 3);
    }

    #[test]
    fn test_player_sample_crossfade() {
        let clips = vec![
            make_walk_clip(3),
            {
                let mut c = AnimationClip::new("Idle", 1.0);
                c.looping = true;
                for i in 0..3 {
                    let mut t = BoneTrack::new(i);
                    t.position_keys = vec![
                        Keyframe::new(0.0, Vec3::ZERO),
                        Keyframe::new(1.0, Vec3::ZERO),
                    ];
                    t.rotation_keys = vec![
                        Keyframe::new(0.0, Quat::IDENTITY),
                        Keyframe::new(1.0, Quat::IDENTITY),
                    ];
                    t.scale_keys = vec![
                        Keyframe::new(0.0, Vec3::ONE),
                        Keyframe::new(1.0, Vec3::ONE),
                    ];
                    c.add_track(t);
                }
                c
            },
        ];
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.advance(0.5, 1.0);
        player.crossfade(1, 0.5);
        player.advance(0.25, 1.0);
        let pose = player.sample(&clips, 3);
        assert_eq!(pose.len(), 3);
        // Should be blended -- not fully walk, not fully idle
    }

    // -- Pose blending tests --

    #[test]
    fn test_blend_poses() {
        let a = vec![Transform::IDENTITY; 3];
        let b = vec![
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0));
            3
        ];
        let blended = blend_poses(&a, &b, 0.5);
        for t in &blended {
            assert!((t.position.x - 5.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_blend_poses_at_zero() {
        let a = vec![Transform::from_position(Vec3::X); 2];
        let b = vec![Transform::from_position(Vec3::Y); 2];
        let blended = blend_poses(&a, &b, 0.0);
        for t in &blended {
            assert!((t.position - Vec3::X).length() < 0.01);
        }
    }

    #[test]
    fn test_blend_poses_at_one() {
        let a = vec![Transform::from_position(Vec3::X); 2];
        let b = vec![Transform::from_position(Vec3::Y); 2];
        let blended = blend_poses(&a, &b, 1.0);
        for t in &blended {
            assert!((t.position - Vec3::Y).length() < 0.01);
        }
    }

    #[test]
    fn test_additive_blend() {
        let base = vec![Transform::from_position(Vec3::new(1.0, 0.0, 0.0)); 2];
        let additive = vec![Transform::from_position(Vec3::new(0.0, 1.0, 0.0)); 2];
        let result = additive_blend(&base, &additive, 1.0);
        for t in &result {
            assert!((t.position.x - 1.0).abs() < 0.01);
            assert!((t.position.y - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_additive_blend_half_weight() {
        let base = vec![Transform::from_position(Vec3::ZERO); 2];
        let additive = vec![Transform::from_position(Vec3::new(2.0, 0.0, 0.0)); 2];
        let result = additive_blend(&base, &additive, 0.5);
        for t in &result {
            assert!((t.position.x - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_masked_blend() {
        let base = vec![
            Transform::from_position(Vec3::ZERO),
            Transform::from_position(Vec3::ZERO),
            Transform::from_position(Vec3::ZERO),
        ];
        let override_pose = vec![
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0)),
        ];
        // Only override bone 1
        let result = masked_blend(&base, &override_pose, &[1], 1.0);
        assert!((result[0].position.x).abs() < 0.01); // untouched
        assert!((result[1].position.x - 10.0).abs() < 0.01); // overridden
        assert!((result[2].position.x).abs() < 0.01); // untouched
    }

    // -- BoneMask tests --

    #[test]
    fn test_bone_mask_from_indices() {
        let mask = BoneMask::from_indices(5, &[1, 3]);
        assert!((mask.weight(0)).abs() < f32::EPSILON);
        assert!((mask.weight(1) - 1.0).abs() < f32::EPSILON);
        assert!((mask.weight(2)).abs() < f32::EPSILON);
        assert!((mask.weight(3) - 1.0).abs() < f32::EPSILON);
        assert!((mask.weight(4)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bone_mask_inverted() {
        let mask = BoneMask::from_indices(3, &[1]);
        let inv = mask.inverted();
        assert!((inv.weight(0) - 1.0).abs() < f32::EPSILON);
        assert!((inv.weight(1)).abs() < f32::EPSILON);
        assert!((inv.weight(2) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bone_mask_from_descendants() {
        let skel = make_humanoid_skeleton();
        // Spine (1) and its children: LeftArm (2), RightArm (3), Head (4)
        let mask = BoneMask::from_bone_and_descendants(&skel, 1);
        assert!((mask.weight(0)).abs() < f32::EPSILON); // Hips excluded
        assert!((mask.weight(1) - 1.0).abs() < f32::EPSILON); // Spine
        assert!((mask.weight(2) - 1.0).abs() < f32::EPSILON); // LeftArm
        assert!((mask.weight(3) - 1.0).abs() < f32::EPSILON); // RightArm
        assert!((mask.weight(4) - 1.0).abs() < f32::EPSILON); // Head
    }

    // -- Animation event tests --

    #[test]
    fn test_collect_events() {
        let events = vec![
            AnimationEvent::new(0.25, "step_left"),
            AnimationEvent::new(0.75, "step_right"),
        ];
        let fired = collect_events(&events, 0.0, 0.5, false, 1.0);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].name, "step_left");
    }

    #[test]
    fn test_collect_events_looping_wrap() {
        let events = vec![
            AnimationEvent::new(0.1, "start"),
            AnimationEvent::new(0.9, "end"),
        ];
        // Looping wrap: prev=0.8, curr=0.2 (wrapped around)
        let fired = collect_events(&events, 0.8, 0.2, true, 1.0);
        assert_eq!(fired.len(), 2); // both events fire during wrap
    }

    // -- Root motion tests --

    #[test]
    fn test_root_motion_extract() {
        let mut clip = AnimationClip::new("Walk", 1.0);
        let mut track = BoneTrack::new(0);
        track.position_keys = vec![
            Keyframe::new(0.0, Vec3::ZERO),
            Keyframe::new(1.0, Vec3::new(0.0, 0.0, 2.0)),
        ];
        track.rotation_keys = vec![
            Keyframe::new(0.0, Quat::IDENTITY),
            Keyframe::new(1.0, Quat::IDENTITY),
        ];
        clip.add_track(track);

        let motion = RootMotion::extract(&clip, 0, 0.0, 0.5);
        assert!((motion.delta_position.z - 1.0).abs() < 0.01);
    }

    // -- Full pipeline integration test --

    #[test]
    fn test_full_animation_pipeline() {
        let skel = make_test_skeleton();
        let clips = vec![make_walk_clip(3)];
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.set_looping(true);

        // Advance several frames
        for _ in 0..10 {
            player.advance(1.0 / 60.0, clips[0].duration);
        }

        // Sample the current pose
        let local_pose = player.sample(&clips, skel.bone_count());
        assert_eq!(local_pose.len(), 3);

        // Compute world transforms
        let world = skel.compute_world_transforms(&local_pose);
        assert_eq!(world.len(), 3);

        // Compute skin matrices
        let skin = skel.compute_skin_matrices(&world);
        assert_eq!(skin.len(), 3);
    }

    #[test]
    fn test_cubic_spline_interpolation() {
        let mut track = BoneTrack::new(0);
        track.position_interpolation = InterpolationMode::CubicSpline;
        track.position_keys = vec![
            Keyframe::new(0.0, Vec3::ZERO),
            Keyframe::new(0.33, Vec3::new(1.0, 0.0, 0.0)),
            Keyframe::new(0.66, Vec3::new(2.0, 1.0, 0.0)),
            Keyframe::new(1.0, Vec3::new(3.0, 0.0, 0.0)),
        ];
        // Cubic should produce a smooth curve; just verify it doesn't crash
        // and values are in a reasonable range.
        for i in 0..100 {
            let t = i as f32 / 99.0;
            let pos = track.sample_position(t);
            assert!(pos.x >= -0.5 && pos.x <= 4.0, "Cubic position out of range at t={}: {:?}", t, pos);
        }
    }

    #[test]
    fn test_player_queue() {
        let mut player = AnimationPlayer::new();
        player.play(0);
        player.set_looping(false);
        player.queue_clip(1);

        // Advance past the end of clip 0
        player.advance(1.5, 1.0);

        // Should now be playing clip 1
        assert_eq!(player.current_clip, Some(1));
    }
}
