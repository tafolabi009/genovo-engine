//! Additive animation layers and multi-layer animation stacking.
//!
//! Additive animations store the *difference* between a reference pose and a
//! target pose. They can be layered on top of any base animation to add
//! secondary motion such as leaning, breathing, aiming offsets, or hit
//! reactions without replacing the underlying locomotion.
//!
//! # Architecture
//!
//! - [`AdditiveClip`]: stores per-bone delta transforms (position offset +
//!   delta rotation) computed from a reference/target clip pair.
//! - [`AnimationLayer`]: a single layer in the stack with weight, mask, and
//!   blend mode (override or additive).
//! - [`LayerStack`]: evaluates an ordered stack of layers to produce the
//!   final composite pose.
//! - [`AimOffset`]: a 2D grid of additive clips indexed by pitch/yaw for
//!   weapon aiming.

use genovo_core::Transform;
use glam::{Quat, Vec2, Vec3};

use crate::skeleton::{AnimationClip, BoneMask, BoneTrack, Keyframe, Skeleton};

// ---------------------------------------------------------------------------
// DeltaTransform
// ---------------------------------------------------------------------------

/// A per-bone delta transform representing the difference between two poses.
///
/// When applied additively to a base pose:
/// - Position: `base.position + delta_position * weight`
/// - Rotation: `base.rotation * slerp(IDENTITY, delta_rotation, weight)`
/// - Scale:    `base.scale * lerp(ONE, delta_scale, weight)` (multiplicative)
#[derive(Debug, Clone, Copy)]
pub struct DeltaTransform {
    /// Positional offset.
    pub delta_position: Vec3,
    /// Rotational offset as a quaternion.
    pub delta_rotation: Quat,
    /// Scale offset (multiplicative; `ONE` means no change).
    pub delta_scale: Vec3,
}

impl Default for DeltaTransform {
    fn default() -> Self {
        Self {
            delta_position: Vec3::ZERO,
            delta_rotation: Quat::IDENTITY,
            delta_scale: Vec3::ONE,
        }
    }
}

impl DeltaTransform {
    /// Create a new delta transform.
    pub fn new(delta_position: Vec3, delta_rotation: Quat, delta_scale: Vec3) -> Self {
        Self {
            delta_position,
            delta_rotation,
            delta_scale,
        }
    }

    /// Compute the delta between a reference and a target transform.
    pub fn from_difference(reference: &Transform, target: &Transform) -> Self {
        let dp = target.position - reference.position;
        let dr = reference.rotation.conjugate() * target.rotation;
        let ds = Vec3::new(
            if reference.scale.x.abs() > f32::EPSILON {
                target.scale.x / reference.scale.x
            } else {
                1.0
            },
            if reference.scale.y.abs() > f32::EPSILON {
                target.scale.y / reference.scale.y
            } else {
                1.0
            },
            if reference.scale.z.abs() > f32::EPSILON {
                target.scale.z / reference.scale.z
            } else {
                1.0
            },
        );
        Self {
            delta_position: dp,
            delta_rotation: dr,
            delta_scale: ds,
        }
    }

    /// Apply this delta to a base transform with the given weight.
    pub fn apply(&self, base: &Transform, weight: f32) -> Transform {
        let w = weight.clamp(0.0, 1.0);
        Transform::new(
            base.position + self.delta_position * w,
            base.rotation * Quat::IDENTITY.slerp(self.delta_rotation, w),
            base.scale * Vec3::ONE.lerp(self.delta_scale, w),
        )
    }

    /// Blend two deltas together.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            delta_position: self.delta_position.lerp(other.delta_position, t),
            delta_rotation: self.delta_rotation.slerp(other.delta_rotation, t),
            delta_scale: self.delta_scale.lerp(other.delta_scale, t),
        }
    }

    /// Check if this delta is approximately identity (no effect).
    pub fn is_identity(&self, epsilon: f32) -> bool {
        self.delta_position.length() < epsilon
            && (self.delta_rotation.dot(Quat::IDENTITY).abs() - 1.0).abs() < epsilon
            && (self.delta_scale - Vec3::ONE).length() < epsilon
    }
}

// ---------------------------------------------------------------------------
// AdditiveClip
// ---------------------------------------------------------------------------

/// An additive animation clip: per-bone delta transforms over time.
///
/// Created by computing the difference between a *reference* clip (typically
/// the first frame of an idle) and a *target* clip (the full animation). The
/// resulting deltas can be layered on top of any base pose.
#[derive(Debug, Clone)]
pub struct AdditiveClip {
    /// Human-readable name.
    pub name: String,

    /// Duration in seconds.
    pub duration: f32,

    /// Per-bone delta tracks. Each track stores keyframed delta transforms.
    pub tracks: Vec<AdditiveBoneTrack>,

    /// Whether to loop.
    pub looping: bool,

    /// Sample rate used during creation.
    pub sample_rate: f32,
}

/// A single bone's additive delta track.
#[derive(Debug, Clone)]
pub struct AdditiveBoneTrack {
    /// Bone index.
    pub bone_index: usize,

    /// Keyframed delta positions.
    pub position_keys: Vec<Keyframe<Vec3>>,

    /// Keyframed delta rotations.
    pub rotation_keys: Vec<Keyframe<Quat>>,

    /// Keyframed delta scales.
    pub scale_keys: Vec<Keyframe<Vec3>>,
}

impl AdditiveBoneTrack {
    /// Create a new empty additive bone track.
    pub fn new(bone_index: usize) -> Self {
        Self {
            bone_index,
            position_keys: Vec::new(),
            rotation_keys: Vec::new(),
            scale_keys: Vec::new(),
        }
    }

    /// Sample the delta transform at the given time.
    pub fn sample(&self, time: f32) -> DeltaTransform {
        let pos = sample_vec3_keys(&self.position_keys, time, Vec3::ZERO);
        let rot = sample_quat_keys(&self.rotation_keys, time, Quat::IDENTITY);
        let scale = sample_vec3_keys(&self.scale_keys, time, Vec3::ONE);
        DeltaTransform::new(pos, rot, scale)
    }

    /// Whether this track has any keyframe data.
    pub fn is_empty(&self) -> bool {
        self.position_keys.is_empty()
            && self.rotation_keys.is_empty()
            && self.scale_keys.is_empty()
    }
}

impl AdditiveClip {
    /// Create an additive clip by computing the difference between a reference
    /// clip and a target clip.
    ///
    /// Both clips must be authored for the same skeleton. The reference is
    /// typically a single-frame idle/T-pose and the target is the full
    /// animation.
    pub fn create_additive(
        reference_clip: &AnimationClip,
        target_clip: &AnimationClip,
        bone_count: usize,
        sample_rate: f32,
    ) -> Self {
        let duration = target_clip.duration;
        let sample_count = (duration * sample_rate).ceil() as usize + 1;

        let mut tracks: Vec<AdditiveBoneTrack> = (0..bone_count)
            .map(AdditiveBoneTrack::new)
            .collect();

        for s in 0..sample_count {
            let t = if sample_count > 1 {
                (s as f32 / (sample_count - 1) as f32) * duration
            } else {
                0.0
            };

            let ref_pose = reference_clip.sample_pose(t.min(reference_clip.duration), bone_count);
            let tgt_pose = target_clip.sample_pose(t, bone_count);

            for bone_idx in 0..bone_count {
                let delta = DeltaTransform::from_difference(&ref_pose[bone_idx], &tgt_pose[bone_idx]);
                tracks[bone_idx].position_keys.push(Keyframe::new(t, delta.delta_position));
                tracks[bone_idx].rotation_keys.push(Keyframe::new(t, delta.delta_rotation));
                tracks[bone_idx].scale_keys.push(Keyframe::new(t, delta.delta_scale));
            }
        }

        Self {
            name: format!("{}_additive", target_clip.name),
            duration,
            tracks,
            looping: target_clip.looping,
            sample_rate,
        }
    }

    /// Create an additive clip from a single reference pose (e.g., bind pose)
    /// and a target clip.
    pub fn from_reference_pose(
        reference_pose: &[Transform],
        target_clip: &AnimationClip,
        sample_rate: f32,
    ) -> Self {
        let bone_count = reference_pose.len();
        let duration = target_clip.duration;
        let sample_count = (duration * sample_rate).ceil() as usize + 1;

        let mut tracks: Vec<AdditiveBoneTrack> = (0..bone_count)
            .map(AdditiveBoneTrack::new)
            .collect();

        for s in 0..sample_count {
            let t = if sample_count > 1 {
                (s as f32 / (sample_count - 1) as f32) * duration
            } else {
                0.0
            };

            let tgt_pose = target_clip.sample_pose(t, bone_count);

            for bone_idx in 0..bone_count {
                let delta = DeltaTransform::from_difference(
                    &reference_pose[bone_idx],
                    &tgt_pose[bone_idx],
                );
                tracks[bone_idx].position_keys.push(Keyframe::new(t, delta.delta_position));
                tracks[bone_idx].rotation_keys.push(Keyframe::new(t, delta.delta_rotation));
                tracks[bone_idx].scale_keys.push(Keyframe::new(t, delta.delta_scale));
            }
        }

        Self {
            name: format!("{}_additive", target_clip.name),
            duration,
            tracks,
            looping: target_clip.looping,
            sample_rate,
        }
    }

    /// Sample the additive clip at the given time, producing per-bone deltas.
    pub fn sample(&self, time: f32, bone_count: usize) -> Vec<DeltaTransform> {
        let clamped = if self.looping && self.duration > 0.0 {
            time.rem_euclid(self.duration)
        } else {
            time.clamp(0.0, self.duration)
        };

        let mut deltas = vec![DeltaTransform::default(); bone_count];
        for track in &self.tracks {
            if track.bone_index < bone_count {
                deltas[track.bone_index] = track.sample(clamped);
            }
        }
        deltas
    }

    /// Get the number of bone tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
}

/// Apply an additive pose (array of deltas) on top of a base pose.
pub fn apply_additive(
    base_pose: &[Transform],
    additive: &[DeltaTransform],
    weight: f32,
) -> Vec<Transform> {
    let w = weight.clamp(0.0, 1.0);
    base_pose
        .iter()
        .zip(additive.iter())
        .map(|(base, delta)| delta.apply(base, w))
        .collect()
}

/// Apply an additive pose with a per-bone mask.
pub fn apply_additive_masked(
    base_pose: &[Transform],
    additive: &[DeltaTransform],
    weight: f32,
    mask: &BoneMask,
) -> Vec<Transform> {
    let w = weight.clamp(0.0, 1.0);
    base_pose
        .iter()
        .zip(additive.iter())
        .enumerate()
        .map(|(i, (base, delta))| {
            let bone_weight = mask.weight(i) * w;
            if bone_weight > f32::EPSILON {
                delta.apply(base, bone_weight)
            } else {
                *base
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// AnimationLayer
// ---------------------------------------------------------------------------

/// Blend mode for an animation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerBlendMode {
    /// Override: replaces the base pose (with optional mask for partial body).
    Override,
    /// Additive: adds delta transforms on top of the base pose.
    Additive,
}

/// A single layer in the animation layer stack.
///
/// Each layer can play its own clip (or use an additive clip) and is combined
/// with the layers below it according to its blend mode, weight, and mask.
#[derive(Debug, Clone)]
pub struct AnimationLayer {
    /// Human-readable name (e.g., "Base", "UpperBody", "Breathing").
    pub name: String,

    /// Layer weight in [0, 1].
    pub weight: f32,

    /// Blend mode.
    pub blend_mode: LayerBlendMode,

    /// Optional bone mask (None = affect all bones).
    pub mask: Option<BoneMask>,

    /// For Override mode: clip index to play.
    pub clip_index: Option<usize>,

    /// For Additive mode: the additive clip.
    pub additive_clip: Option<AdditiveClip>,

    /// Current playback time.
    pub playback_time: f32,

    /// Playback speed multiplier.
    pub speed: f32,

    /// Whether this layer is active.
    pub enabled: bool,

    /// Whether to loop the clip.
    pub looping: bool,
}

impl AnimationLayer {
    /// Create an override layer.
    pub fn new_override(name: impl Into<String>, clip_index: usize) -> Self {
        Self {
            name: name.into(),
            weight: 1.0,
            blend_mode: LayerBlendMode::Override,
            mask: None,
            clip_index: Some(clip_index),
            additive_clip: None,
            playback_time: 0.0,
            speed: 1.0,
            enabled: true,
            looping: true,
        }
    }

    /// Create an additive layer.
    pub fn new_additive(name: impl Into<String>, additive_clip: AdditiveClip) -> Self {
        let looping = additive_clip.looping;
        Self {
            name: name.into(),
            weight: 1.0,
            blend_mode: LayerBlendMode::Additive,
            mask: None,
            clip_index: None,
            additive_clip: Some(additive_clip),
            playback_time: 0.0,
            speed: 1.0,
            enabled: true,
            looping,
        }
    }

    /// Set a bone mask for this layer.
    pub fn with_mask(mut self, mask: BoneMask) -> Self {
        self.mask = Some(mask);
        self
    }

    /// Set the layer weight.
    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight.clamp(0.0, 1.0);
    }

    /// Advance the layer's playback time.
    pub fn advance(&mut self, dt: f32, clip_duration: f32) {
        if !self.enabled {
            return;
        }
        self.playback_time += dt * self.speed;
        if self.looping && clip_duration > 0.0 {
            self.playback_time = self.playback_time.rem_euclid(clip_duration);
        } else {
            self.playback_time = self.playback_time.clamp(0.0, clip_duration);
        }
    }

    /// Reset playback to the beginning.
    pub fn reset(&mut self) {
        self.playback_time = 0.0;
    }
}

// ---------------------------------------------------------------------------
// LayerStack
// ---------------------------------------------------------------------------

/// A stack of animation layers evaluated bottom-to-top.
///
/// The first layer (index 0) is the base layer and should typically be an
/// Override layer with full body. Subsequent layers modify the result using
/// their blend mode, weight, and mask.
#[derive(Debug, Clone)]
pub struct LayerStack {
    /// Ordered list of layers (evaluated from index 0 upward).
    pub layers: Vec<AnimationLayer>,

    /// Number of bones in the skeleton.
    bone_count: usize,
}

impl LayerStack {
    /// Create a new empty layer stack.
    pub fn new(bone_count: usize) -> Self {
        Self {
            layers: Vec::new(),
            bone_count,
        }
    }

    /// Add a layer to the top of the stack.
    pub fn push_layer(&mut self, layer: AnimationLayer) {
        self.layers.push(layer);
    }

    /// Insert a layer at a specific position.
    pub fn insert_layer(&mut self, index: usize, layer: AnimationLayer) {
        self.layers.insert(index.min(self.layers.len()), layer);
    }

    /// Remove a layer by index.
    pub fn remove_layer(&mut self, index: usize) -> Option<AnimationLayer> {
        if index < self.layers.len() {
            Some(self.layers.remove(index))
        } else {
            None
        }
    }

    /// Find a layer by name.
    pub fn find_layer(&self, name: &str) -> Option<usize> {
        self.layers.iter().position(|l| l.name == name)
    }

    /// Get a mutable reference to a layer by name.
    pub fn layer_mut(&mut self, name: &str) -> Option<&mut AnimationLayer> {
        self.layers.iter_mut().find(|l| l.name == name)
    }

    /// Number of layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Evaluate the entire layer stack and produce the final blended pose.
    ///
    /// `clips` is the external clip collection used by Override layers.
    /// `dt` is the frame delta time for advancing playback.
    pub fn evaluate(&mut self, clips: &[AnimationClip], dt: f32) -> Vec<Transform> {
        let bone_count = self.bone_count;
        let mut result = vec![Transform::IDENTITY; bone_count];
        let mut base_set = false;

        for layer in &mut self.layers {
            if !layer.enabled || layer.weight < f32::EPSILON {
                continue;
            }

            match layer.blend_mode {
                LayerBlendMode::Override => {
                    if let Some(clip_idx) = layer.clip_index {
                        if clip_idx < clips.len() {
                            let clip = &clips[clip_idx];
                            layer.advance(dt, clip.duration);
                            let pose = clip.sample_pose(layer.playback_time, bone_count);

                            if !base_set {
                                result = pose;
                                base_set = true;
                            } else {
                                // Apply with weight and mask.
                                let w = layer.weight;
                                if let Some(ref mask) = layer.mask {
                                    for i in 0..bone_count {
                                        let bone_w = mask.weight(i) * w;
                                        if bone_w > f32::EPSILON {
                                            result[i] = result[i].lerp(&pose[i], bone_w);
                                        }
                                    }
                                } else {
                                    for i in 0..bone_count {
                                        result[i] = result[i].lerp(&pose[i], w);
                                    }
                                }
                            }
                        }
                    }
                }

                LayerBlendMode::Additive => {
                    let duration = layer.additive_clip.as_ref().map(|a| a.duration);
                    if let Some(dur) = duration {
                        layer.advance(dt, dur);
                        if let Some(ref additive) = layer.additive_clip {
                            let deltas = additive.sample(layer.playback_time, bone_count);
                            if let Some(ref mask) = layer.mask {
                                result = apply_additive_masked(&result, &deltas, layer.weight, mask);
                            } else {
                                result = apply_additive(&result, &deltas, layer.weight);
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// AimOffset
// ---------------------------------------------------------------------------

/// A 2D grid of additive clips for aiming offset, indexed by (pitch, yaw).
///
/// The grid stores pre-computed additive clips at discrete pitch/yaw
/// increments. At runtime, the current aim angles are used to bilinearly
/// interpolate between the four surrounding grid cells.
#[derive(Debug, Clone)]
pub struct AimOffset {
    /// Human-readable name.
    pub name: String,

    /// Grid of additive clips: `grid[row][col]` where row = pitch, col = yaw.
    grid: Vec<Vec<Option<AdditiveClip>>>,

    /// Number of pitch divisions (rows).
    pitch_divisions: usize,

    /// Number of yaw divisions (columns).
    yaw_divisions: usize,

    /// Pitch range in radians (min, max). Typically (-PI/4, PI/4) for 45 deg up/down.
    pub pitch_range: (f32, f32),

    /// Yaw range in radians (min, max). Typically (-PI/2, PI/2) for 90 deg left/right.
    pub yaw_range: (f32, f32),

    /// Number of bones.
    bone_count: usize,
}

impl AimOffset {
    /// Create a new aim offset grid.
    ///
    /// `pitch_divisions` and `yaw_divisions` define the grid resolution. A
    /// 3x3 grid is common (up-left, up, up-right, left, center, right,
    /// down-left, down, down-right).
    pub fn new(
        name: impl Into<String>,
        pitch_divisions: usize,
        yaw_divisions: usize,
        pitch_range: (f32, f32),
        yaw_range: (f32, f32),
        bone_count: usize,
    ) -> Self {
        let grid = vec![vec![None; yaw_divisions]; pitch_divisions];
        Self {
            name: name.into(),
            grid,
            pitch_divisions,
            yaw_divisions,
            pitch_range,
            yaw_range,
            bone_count,
        }
    }

    /// Set the additive clip at a specific grid cell.
    pub fn set_clip(&mut self, pitch_idx: usize, yaw_idx: usize, clip: AdditiveClip) {
        if pitch_idx < self.pitch_divisions && yaw_idx < self.yaw_divisions {
            self.grid[pitch_idx][yaw_idx] = Some(clip);
        }
    }

    /// Get the grid dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.pitch_divisions, self.yaw_divisions)
    }

    /// Sample the aim offset at the given pitch and yaw angles (in radians).
    ///
    /// Bilinearly interpolates between the four surrounding grid cells.
    pub fn sample(&self, pitch: f32, yaw: f32, time: f32) -> Vec<DeltaTransform> {
        let bone_count = self.bone_count;
        let mut result = vec![DeltaTransform::default(); bone_count];

        // Normalize pitch and yaw to [0, 1] within their ranges.
        let pitch_range = self.pitch_range.1 - self.pitch_range.0;
        let yaw_range = self.yaw_range.1 - self.yaw_range.0;

        if pitch_range.abs() < f32::EPSILON || yaw_range.abs() < f32::EPSILON {
            return result;
        }

        let pitch_norm = ((pitch - self.pitch_range.0) / pitch_range).clamp(0.0, 1.0);
        let yaw_norm = ((yaw - self.yaw_range.0) / yaw_range).clamp(0.0, 1.0);

        // Map to grid coordinates.
        let row_f = pitch_norm * (self.pitch_divisions - 1) as f32;
        let col_f = yaw_norm * (self.yaw_divisions - 1) as f32;

        let row0 = (row_f.floor() as usize).min(self.pitch_divisions - 1);
        let col0 = (col_f.floor() as usize).min(self.yaw_divisions - 1);
        let row1 = (row0 + 1).min(self.pitch_divisions - 1);
        let col1 = (col0 + 1).min(self.yaw_divisions - 1);

        let row_t = row_f - row0 as f32;
        let col_t = col_f - col0 as f32;

        // Sample the four corners.
        let d00 = self.sample_cell(row0, col0, time, bone_count);
        let d10 = self.sample_cell(row1, col0, time, bone_count);
        let d01 = self.sample_cell(row0, col1, time, bone_count);
        let d11 = self.sample_cell(row1, col1, time, bone_count);

        // Bilinear interpolation.
        for i in 0..bone_count {
            let top = d00[i].lerp(&d01[i], col_t);
            let bottom = d10[i].lerp(&d11[i], col_t);
            result[i] = top.lerp(&bottom, row_t);
        }

        result
    }

    /// Apply the aim offset to a base pose.
    pub fn apply(
        &self,
        base_pose: &[Transform],
        pitch: f32,
        yaw: f32,
        weight: f32,
        time: f32,
    ) -> Vec<Transform> {
        let deltas = self.sample(pitch, yaw, time);
        apply_additive(base_pose, &deltas, weight)
    }

    /// Apply the aim offset with a bone mask.
    pub fn apply_masked(
        &self,
        base_pose: &[Transform],
        pitch: f32,
        yaw: f32,
        weight: f32,
        time: f32,
        mask: &BoneMask,
    ) -> Vec<Transform> {
        let deltas = self.sample(pitch, yaw, time);
        apply_additive_masked(base_pose, &deltas, weight, mask)
    }

    // ----- helpers -----

    fn sample_cell(
        &self,
        row: usize,
        col: usize,
        time: f32,
        bone_count: usize,
    ) -> Vec<DeltaTransform> {
        if let Some(ref clip) = self.grid[row][col] {
            clip.sample(time, bone_count)
        } else {
            vec![DeltaTransform::default(); bone_count]
        }
    }
}

// ---------------------------------------------------------------------------
// Keyframe sampling helpers
// ---------------------------------------------------------------------------

fn sample_vec3_keys(keys: &[Keyframe<Vec3>], time: f32, default: Vec3) -> Vec3 {
    if keys.is_empty() {
        return default;
    }
    if keys.len() == 1 || time <= keys[0].time {
        return keys[0].value;
    }
    let last = keys.len() - 1;
    if time >= keys[last].time {
        return keys[last].value;
    }
    let idx = binary_search_keys(keys, time);
    let next = (idx + 1).min(last);
    if idx == next {
        return keys[idx].value;
    }
    let t = compute_t(keys[idx].time, keys[next].time, time);
    keys[idx].value.lerp(keys[next].value, t)
}

fn sample_quat_keys(keys: &[Keyframe<Quat>], time: f32, default: Quat) -> Quat {
    if keys.is_empty() {
        return default;
    }
    if keys.len() == 1 || time <= keys[0].time {
        return keys[0].value;
    }
    let last = keys.len() - 1;
    if time >= keys[last].time {
        return keys[last].value;
    }
    let idx = binary_search_keys(keys, time);
    let next = (idx + 1).min(last);
    if idx == next {
        return keys[idx].value;
    }
    let t = compute_t(keys[idx].time, keys[next].time, time);
    keys[idx].value.slerp(keys[next].value, t)
}

fn binary_search_keys<T>(keys: &[Keyframe<T>], time: f32) -> usize {
    if keys.len() <= 1 {
        return 0;
    }
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
    if lo == 0 { 0 } else { lo - 1 }
}

fn compute_t(t0: f32, t1: f32, time: f32) -> f32 {
    let d = t1 - t0;
    if d.abs() < f32::EPSILON {
        return 0.0;
    }
    ((time - t0) / d).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{AnimationClip, Bone, BoneMask, BoneTrack, Keyframe, Skeleton};
    use glam::{Mat4, Quat, Vec3};

    fn make_skeleton() -> Skeleton {
        Skeleton::new(
            "Test",
            vec![
                Bone::new("Root", None, Transform::IDENTITY, Mat4::IDENTITY),
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
            ],
        )
    }

    fn make_clip(name: &str, bone_count: usize) -> AnimationClip {
        let mut clip = AnimationClip::new(name, 1.0);
        clip.looping = true;
        for i in 0..bone_count {
            let mut track = BoneTrack::new(i);
            track.position_keys = vec![
                Keyframe::new(0.0, Vec3::new(0.0, i as f32, 0.0)),
                Keyframe::new(1.0, Vec3::new(0.0, i as f32, 0.0)),
            ];
            track.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
            track.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
            clip.add_track(track);
        }
        clip
    }

    fn make_motion_clip(name: &str, bone_count: usize) -> AnimationClip {
        let mut clip = AnimationClip::new(name, 1.0);
        clip.looping = true;
        for i in 0..bone_count {
            let mut track = BoneTrack::new(i);
            track.position_keys = vec![
                Keyframe::new(0.0, Vec3::new(0.0, i as f32, 0.0)),
                Keyframe::new(0.5, Vec3::new(0.0, i as f32 + 0.2, 0.0)),
                Keyframe::new(1.0, Vec3::new(0.0, i as f32, 0.0)),
            ];
            track.rotation_keys = vec![
                Keyframe::new(0.0, Quat::IDENTITY),
                Keyframe::new(0.5, Quat::from_rotation_z(0.1)),
                Keyframe::new(1.0, Quat::IDENTITY),
            ];
            track.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
            clip.add_track(track);
        }
        clip
    }

    // -- DeltaTransform tests --

    #[test]
    fn test_delta_identity() {
        let dt = DeltaTransform::default();
        assert!(dt.is_identity(0.001));
    }

    #[test]
    fn test_delta_from_difference() {
        let a = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));
        let b = Transform::from_position(Vec3::new(3.0, 0.0, 0.0));
        let d = DeltaTransform::from_difference(&a, &b);
        assert!((d.delta_position.x - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_delta_apply() {
        let base = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));
        let delta = DeltaTransform::new(
            Vec3::new(2.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        );
        let result = delta.apply(&base, 1.0);
        assert!((result.position.x - 3.0).abs() < 0.01);

        let half = delta.apply(&base, 0.5);
        assert!((half.position.x - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_delta_apply_rotation() {
        let base = Transform::IDENTITY;
        let angle = std::f32::consts::FRAC_PI_4;
        let delta = DeltaTransform::new(
            Vec3::ZERO,
            Quat::from_rotation_z(angle),
            Vec3::ONE,
        );
        let result = delta.apply(&base, 1.0);
        let expected = Quat::from_rotation_z(angle);
        assert!(result.rotation.dot(expected).abs() > 0.99);
    }

    #[test]
    fn test_delta_lerp() {
        let a = DeltaTransform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        let b = DeltaTransform::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.delta_position.x - 1.0).abs() < 0.01);
    }

    // -- AdditiveClip tests --

    #[test]
    fn test_create_additive_from_clips() {
        let ref_clip = make_clip("ref", 3);
        let tgt_clip = make_motion_clip("motion", 3);
        let additive = AdditiveClip::create_additive(&ref_clip, &tgt_clip, 3, 10.0);
        assert_eq!(additive.track_count(), 3);
        assert!((additive.duration - 1.0).abs() < f32::EPSILON);

        // At t=0.5, the motion clip has extra Y displacement.
        let deltas = additive.sample(0.5, 3);
        // Bone 0: ref pos = (0,0,0), target pos at t=0.5 = (0, 0.2, 0)
        assert!(
            deltas[0].delta_position.y.abs() > 0.01,
            "Delta should have Y offset at t=0.5"
        );
    }

    #[test]
    fn test_create_additive_from_reference_pose() {
        let ref_pose = vec![Transform::IDENTITY; 3];
        let tgt_clip = make_motion_clip("motion", 3);
        let additive = AdditiveClip::from_reference_pose(&ref_pose, &tgt_clip, 10.0);
        assert_eq!(additive.track_count(), 3);
    }

    #[test]
    fn test_additive_looping() {
        let ref_clip = make_clip("ref", 2);
        let tgt_clip = make_motion_clip("motion", 2);
        let additive = AdditiveClip::create_additive(&ref_clip, &tgt_clip, 2, 10.0);

        // Sample past duration (should wrap).
        let d1 = additive.sample(0.5, 2);
        let d2 = additive.sample(1.5, 2); // Should wrap to 0.5.
        assert!(
            (d1[0].delta_position - d2[0].delta_position).length() < 0.1,
            "Looping additive should wrap"
        );
    }

    // -- apply_additive tests --

    #[test]
    fn test_apply_additive_zero_weight() {
        let base = vec![Transform::from_position(Vec3::X); 2];
        let deltas = vec![
            DeltaTransform::new(Vec3::Y, Quat::IDENTITY, Vec3::ONE);
            2
        ];
        let result = apply_additive(&base, &deltas, 0.0);
        for t in &result {
            assert!((t.position - Vec3::X).length() < 0.01);
        }
    }

    #[test]
    fn test_apply_additive_full_weight() {
        let base = vec![Transform::from_position(Vec3::X); 2];
        let deltas = vec![
            DeltaTransform::new(Vec3::Y, Quat::IDENTITY, Vec3::ONE);
            2
        ];
        let result = apply_additive(&base, &deltas, 1.0);
        for t in &result {
            assert!((t.position - Vec3::new(1.0, 1.0, 0.0)).length() < 0.01);
        }
    }

    #[test]
    fn test_apply_additive_masked() {
        let base = vec![
            Transform::from_position(Vec3::ZERO),
            Transform::from_position(Vec3::ZERO),
            Transform::from_position(Vec3::ZERO),
        ];
        let deltas = vec![
            DeltaTransform::new(Vec3::X, Quat::IDENTITY, Vec3::ONE),
            DeltaTransform::new(Vec3::X, Quat::IDENTITY, Vec3::ONE),
            DeltaTransform::new(Vec3::X, Quat::IDENTITY, Vec3::ONE),
        ];
        // Only affect bone 1.
        let mask = BoneMask::from_indices(3, &[1]);
        let result = apply_additive_masked(&base, &deltas, 1.0, &mask);
        assert!((result[0].position.x).abs() < 0.01); // Unmasked.
        assert!((result[1].position.x - 1.0).abs() < 0.01); // Masked.
        assert!((result[2].position.x).abs() < 0.01); // Unmasked.
    }

    // -- AnimationLayer tests --

    #[test]
    fn test_layer_creation() {
        let layer = AnimationLayer::new_override("Base", 0);
        assert_eq!(layer.name, "Base");
        assert_eq!(layer.blend_mode, LayerBlendMode::Override);
        assert!(layer.enabled);
        assert!((layer.weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_layer_advance() {
        let mut layer = AnimationLayer::new_override("Base", 0);
        layer.advance(0.5, 1.0);
        assert!((layer.playback_time - 0.5).abs() < 0.01);

        layer.advance(0.7, 1.0); // Should wrap around for looping.
        assert!(layer.playback_time < 1.0);
    }

    #[test]
    fn test_layer_disabled_no_advance() {
        let mut layer = AnimationLayer::new_override("Base", 0);
        layer.enabled = false;
        layer.advance(0.5, 1.0);
        assert!((layer.playback_time).abs() < f32::EPSILON);
    }

    // -- LayerStack tests --

    #[test]
    fn test_layer_stack_single_override() {
        let clips = vec![make_clip("idle", 3)];
        let mut stack = LayerStack::new(3);
        stack.push_layer(AnimationLayer::new_override("Base", 0));

        let pose = stack.evaluate(&clips, 0.016);
        assert_eq!(pose.len(), 3);
    }

    #[test]
    fn test_layer_stack_override_and_additive() {
        let ref_clip = make_clip("ref", 3);
        let motion_clip = make_motion_clip("motion", 3);
        let additive = AdditiveClip::create_additive(&ref_clip, &motion_clip, 3, 10.0);

        let clips = vec![make_clip("idle", 3)];
        let mut stack = LayerStack::new(3);
        stack.push_layer(AnimationLayer::new_override("Base", 0));
        stack.push_layer(AnimationLayer::new_additive("Breathing", additive));

        let pose = stack.evaluate(&clips, 0.016);
        assert_eq!(pose.len(), 3);
    }

    #[test]
    fn test_layer_stack_masked_override() {
        let clips = vec![make_clip("idle", 3), make_motion_clip("attack", 3)];
        let mut stack = LayerStack::new(3);
        stack.push_layer(AnimationLayer::new_override("Base", 0));

        let upper_body_mask = BoneMask::from_indices(3, &[1, 2]); // Spine + Head.
        stack.push_layer(
            AnimationLayer::new_override("UpperBody", 1).with_mask(upper_body_mask),
        );

        let pose = stack.evaluate(&clips, 0.016);
        assert_eq!(pose.len(), 3);
    }

    #[test]
    fn test_layer_stack_find_layer() {
        let mut stack = LayerStack::new(3);
        stack.push_layer(AnimationLayer::new_override("Base", 0));
        stack.push_layer(AnimationLayer::new_override("Upper", 1));

        assert_eq!(stack.find_layer("Base"), Some(0));
        assert_eq!(stack.find_layer("Upper"), Some(1));
        assert_eq!(stack.find_layer("Missing"), None);
    }

    #[test]
    fn test_layer_stack_modify_weight() {
        let clips = vec![make_clip("idle", 3), make_motion_clip("run", 3)];
        let mut stack = LayerStack::new(3);
        stack.push_layer(AnimationLayer::new_override("Base", 0));
        stack.push_layer(AnimationLayer::new_override("Run", 1));

        // Set run layer to half weight.
        if let Some(layer) = stack.layer_mut("Run") {
            layer.set_weight(0.5);
        }

        let pose = stack.evaluate(&clips, 0.016);
        assert_eq!(pose.len(), 3);
    }

    #[test]
    fn test_layer_stack_empty() {
        let clips: Vec<AnimationClip> = vec![];
        let mut stack = LayerStack::new(3);
        let pose = stack.evaluate(&clips, 0.016);
        assert_eq!(pose.len(), 3);
        // Should be identity poses.
        for t in &pose {
            assert!((t.position - Vec3::ZERO).length() < 0.01);
        }
    }

    #[test]
    fn test_layer_stack_remove() {
        let mut stack = LayerStack::new(3);
        stack.push_layer(AnimationLayer::new_override("Base", 0));
        stack.push_layer(AnimationLayer::new_override("Upper", 1));
        assert_eq!(stack.layer_count(), 2);

        let removed = stack.remove_layer(1);
        assert!(removed.is_some());
        assert_eq!(stack.layer_count(), 1);
    }

    // -- AimOffset tests --

    #[test]
    fn test_aim_offset_creation() {
        let ao = AimOffset::new(
            "Aim",
            3,
            3,
            (-0.8, 0.8),
            (-1.0, 1.0),
            3,
        );
        assert_eq!(ao.dimensions(), (3, 3));
    }

    #[test]
    fn test_aim_offset_sample_center() {
        let mut ao = AimOffset::new(
            "Aim",
            3,
            3,
            (-0.8, 0.8),
            (-1.0, 1.0),
            2,
        );

        // Set center cell (1, 1) with a simple additive clip.
        let ref_clip = make_clip("ref", 2);
        let motion = make_motion_clip("aim_center", 2);
        let center_additive = AdditiveClip::create_additive(&ref_clip, &motion, 2, 10.0);
        ao.set_clip(1, 1, center_additive);

        // Sample at center (pitch=0, yaw=0).
        let deltas = ao.sample(0.0, 0.0, 0.5);
        assert_eq!(deltas.len(), 2);
    }

    #[test]
    fn test_aim_offset_apply() {
        let mut ao = AimOffset::new(
            "Aim",
            3,
            3,
            (-0.8, 0.8),
            (-1.0, 1.0),
            2,
        );

        let ref_clip = make_clip("ref", 2);
        let motion = make_motion_clip("aim_center", 2);
        let additive = AdditiveClip::create_additive(&ref_clip, &motion, 2, 10.0);
        ao.set_clip(1, 1, additive);

        let base_pose = vec![Transform::IDENTITY; 2];
        let result = ao.apply(&base_pose, 0.0, 0.0, 1.0, 0.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_aim_offset_bilinear_interpolation() {
        let mut ao = AimOffset::new(
            "Aim",
            2,
            2,
            (-1.0, 1.0),
            (-1.0, 1.0),
            1,
        );

        // Create 4 different additive clips for the 4 corners.
        let ref_pose = vec![Transform::IDENTITY];
        for row in 0..2 {
            for col in 0..2 {
                let offset = Vec3::new(row as f32, col as f32, 0.0);
                let mut clip = AnimationClip::new(format!("aim_{}_{}", row, col), 1.0);
                clip.looping = true;
                let mut track = BoneTrack::new(0);
                track.position_keys = vec![Keyframe::new(0.0, offset)];
                track.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
                track.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
                clip.add_track(track);

                let additive = AdditiveClip::from_reference_pose(&ref_pose, &clip, 10.0);
                ao.set_clip(row, col, additive);
            }
        }

        // At the center, should be a blend of all 4 corners.
        let deltas = ao.sample(0.0, 0.0, 0.0);
        assert_eq!(deltas.len(), 1);
        // The blended position should be approximately (0.5, 0.5, 0).
        assert!(
            (deltas[0].delta_position.x - 0.5).abs() < 0.15,
            "Expected ~0.5, got {}",
            deltas[0].delta_position.x
        );
    }
}
