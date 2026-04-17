//! GPU bridge: connects the animation system's bone transforms to the
//! GPU skinning pipeline.
//!
//! This module provides the glue between `engine/animation` (which produces
//! local-space bone poses) and `engine/render/gpu_skeletal` (which uploads
//! bone matrices to the GPU for vertex skinning).
//!
//! # Pipeline
//!
//! 1. **AnimationPlayer** samples clips to produce local-space `Transform` per bone.
//! 2. **Skeleton::compute_world_transforms** converts local poses to world-space `Mat4`.
//! 3. **Skeleton::compute_skin_matrices** multiplies by inverse-bind matrices.
//! 4. **This module** converts the resulting `Mat4` array to `[f32; 16]` column-major
//!    arrays suitable for GPU upload, and writes them to a `wgpu::Buffer`.
//!
//! # Usage
//!
//! ```ignore
//! use genovo_animation::gpu_bridge::*;
//!
//! // Each frame:
//! let local_pose = player.sample(&clips, skeleton.bone_count());
//! let gpu_matrices = compute_skinning_matrices(&skeleton, &local_pose);
//! upload_bone_palette(&queue, &buffer, &gpu_matrices);
//! ```

use glam::{Mat4, Quat, Vec3, Vec4};

use genovo_core::Transform;
use crate::skeleton::{AnimationClip, AnimationPlayer, Bone, Skeleton, SkinnedMeshRenderer};

// ---------------------------------------------------------------------------
// Core conversion functions
// ---------------------------------------------------------------------------

/// Convert animation bone transforms to a GPU-ready array of column-major
/// `[f32; 16]` matrices.
///
/// This is the main entry point for the animation-to-GPU bridge. It performs
/// the full pipeline:
///
/// 1. Converts local-space poses to world-space via the skeleton hierarchy.
/// 2. Multiplies each world-space matrix by the bone's inverse-bind matrix
///    to produce the final skinning matrix.
/// 3. Converts each `Mat4` to a flat `[f32; 16]` for GPU upload.
///
/// # Arguments
///
/// * `skeleton` - The skeleton defining the bone hierarchy and inverse-bind poses.
/// * `local_poses` - Per-bone local-space transforms, typically from `AnimationPlayer::sample`.
///
/// # Returns
///
/// A `Vec` of column-major 4x4 matrices, one per bone, ready for GPU upload.
///
/// # Panics
///
/// Panics if `local_poses.len() != skeleton.bone_count()`.
pub fn compute_skinning_matrices(
    skeleton: &Skeleton,
    local_poses: &[Transform],
) -> Vec<[f32; 16]> {
    let world_transforms = skeleton.compute_world_transforms(local_poses);
    skeleton
        .compute_skin_matrices(&world_transforms)
        .iter()
        .map(|m| m.to_cols_array())
        .collect()
}

/// Upload bone matrices to a GPU storage/uniform buffer.
///
/// Writes the raw matrix data to the beginning of the buffer using
/// `queue.write_buffer`. The buffer must be large enough to hold all
/// matrices (each matrix is 64 bytes = 16 x f32).
///
/// # Arguments
///
/// * `queue` - The wgpu command queue for buffer writes.
/// * `buffer` - The GPU buffer to write into (must have `COPY_DST` usage).
/// * `matrices` - The skinning matrices to upload.
pub fn upload_bone_palette(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    matrices: &[[f32; 16]],
) {
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(matrices));
}

// ---------------------------------------------------------------------------
// Dual quaternion conversion
// ---------------------------------------------------------------------------

/// A dual quaternion representation suitable for GPU upload.
///
/// DQS (Dual Quaternion Skinning) avoids the "candy wrapper" artifact
/// of Linear Blend Skinning at twisting joints.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuDualQuat {
    /// Real part (rotation quaternion) — [x, y, z, w].
    pub real: [f32; 4],
    /// Dual part (encodes translation) — [x, y, z, w].
    pub dual: [f32; 4],
}

impl Default for GpuDualQuat {
    fn default() -> Self {
        Self {
            real: [0.0, 0.0, 0.0, 1.0],
            dual: [0.0; 4],
        }
    }
}

/// Convert a rigid transform (rotation + translation, no scale) to a dual quaternion.
///
/// The real part is simply the rotation quaternion. The dual part encodes
/// the translation as `0.5 * t_quat * rotation`.
pub fn transform_to_dual_quat(rotation: Quat, translation: Vec3) -> GpuDualQuat {
    let r = rotation;
    let t_quat = Quat::from_xyzw(translation.x, translation.y, translation.z, 0.0);
    let d = t_quat * r * 0.5;
    GpuDualQuat {
        real: [r.x, r.y, r.z, r.w],
        dual: [d.x, d.y, d.z, d.w],
    }
}

/// Convert skinning matrices to dual quaternion representation for DQS.
///
/// Each input `Mat4` is decomposed into a rotation quaternion and translation
/// vector. Scale is discarded (DQS assumes rigid transforms).
///
/// # Arguments
///
/// * `matrices` - Skinning matrices (world * inverse-bind) from the animation pipeline.
///
/// # Returns
///
/// A `Vec<GpuDualQuat>` ready for GPU upload.
pub fn matrices_to_dual_quaternions(matrices: &[Mat4]) -> Vec<GpuDualQuat> {
    matrices
        .iter()
        .map(|m| {
            let (_, rotation, translation) = m.to_scale_rotation_translation();
            transform_to_dual_quat(rotation, translation)
        })
        .collect()
}

/// Upload dual quaternion bone data to a GPU buffer.
///
/// Each dual quaternion occupies 32 bytes (8 x f32).
pub fn upload_dual_quaternion_palette(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    dqs: &[GpuDualQuat],
) {
    // Safety: GpuDualQuat is repr(C) with only f32 fields, so it can be
    // safely reinterpreted as bytes.
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            dqs.as_ptr() as *const u8,
            dqs.len() * std::mem::size_of::<GpuDualQuat>(),
        )
    };
    queue.write_buffer(buffer, 0, byte_slice);
}

// ---------------------------------------------------------------------------
// Full-frame animation update
// ---------------------------------------------------------------------------

/// Per-entity skinning state used by the animation update loop.
pub struct SkinnedEntityState {
    /// Index into the animation clip collection.
    pub clip_index: usize,
    /// Current playback time in seconds.
    pub playback_time: f32,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Whether the animation loops.
    pub looping: bool,
    /// Whether this entity is visible and should be updated.
    pub active: bool,
}

/// Result of updating a skinned entity's animation for one frame.
pub struct SkinningResult {
    /// The GPU-ready skinning matrices (column-major [f32; 16]).
    pub matrices: Vec<[f32; 16]>,
    /// Number of bones.
    pub bone_count: usize,
    /// Whether the animation has finished (non-looping clip reached end).
    pub finished: bool,
}

/// Update animation and produce GPU-ready skinning matrices for a single entity.
///
/// This function encapsulates the full per-frame pipeline:
///
/// 1. Advances the animation player by `dt`.
/// 2. Samples the current clip(s) to produce a local-space pose.
/// 3. Converts to world-space and applies inverse-bind matrices.
/// 4. Returns the GPU-ready matrix array.
///
/// # Arguments
///
/// * `player` - The entity's animation player (mutated to advance time).
/// * `skeleton` - The skeleton asset.
/// * `clips` - All available animation clips.
/// * `dt` - Frame delta time in seconds.
///
/// # Returns
///
/// A `SkinningResult` containing the matrices and metadata.
pub fn update_skinned_entity(
    player: &mut AnimationPlayer,
    skeleton: &Skeleton,
    clips: &[AnimationClip],
    dt: f32,
) -> SkinningResult {
    // Advance playback
    let clip_duration = player
        .current_clip
        .and_then(|idx| clips.get(idx))
        .map(|c| c.duration)
        .unwrap_or(0.0);
    player.advance(dt, clip_duration);

    // Sample the pose (handles crossfading internally)
    let local_pose = player.sample(clips, skeleton.bone_count());

    // Compute skinning matrices
    let matrices = compute_skinning_matrices(skeleton, &local_pose);
    let bone_count = matrices.len();
    let finished = player.is_finished();

    SkinningResult {
        matrices,
        bone_count,
        finished,
    }
}

/// Batch-update all skinned entities and upload their bone palettes.
///
/// Iterates over a collection of (player, skeleton, clips, buffer) tuples,
/// updates each entity's animation, and uploads the resulting matrices.
///
/// This is the highest-level convenience function for integrating animation
/// with rendering in the main loop.
pub fn batch_update_and_upload(
    queue: &wgpu::Queue,
    entities: &mut [(
        &mut AnimationPlayer,
        &Skeleton,
        &[AnimationClip],
        &wgpu::Buffer,
    )],
    dt: f32,
) {
    for (player, skeleton, clips, buffer) in entities.iter_mut() {
        let result = update_skinned_entity(*player, *skeleton, *clips, dt);
        upload_bone_palette(queue, *buffer, &result.matrices);
    }
}

// ---------------------------------------------------------------------------
// SkinnedMeshRenderer integration
// ---------------------------------------------------------------------------

/// Update a `SkinnedMeshRenderer` component with fresh bone matrices from
/// the animation system. This is the bridge between the animation ECS
/// component and the render ECS component.
///
/// After calling this, the renderer's `bone_matrices` field contains the
/// current frame's skinning matrices, ready to be uploaded to the GPU.
pub fn update_skinned_renderer(
    renderer: &mut SkinnedMeshRenderer,
    skeleton: &Skeleton,
    local_pose: &[Transform],
) {
    if !renderer.enabled {
        return;
    }
    let world_transforms = skeleton.compute_world_transforms(local_pose);
    renderer.bone_matrices = skeleton.compute_skin_matrices(&world_transforms);
}

/// Convert a `SkinnedMeshRenderer`'s bone matrices to GPU-ready format.
pub fn renderer_to_gpu_matrices(renderer: &SkinnedMeshRenderer) -> Vec<[f32; 16]> {
    renderer
        .bone_matrices
        .iter()
        .map(|m| m.to_cols_array())
        .collect()
}

/// Full pipeline: update renderer from animation and produce GPU matrices.
pub fn update_renderer_and_get_gpu_data(
    renderer: &mut SkinnedMeshRenderer,
    skeleton: &Skeleton,
    local_pose: &[Transform],
) -> Vec<[f32; 16]> {
    update_skinned_renderer(renderer, skeleton, local_pose);
    renderer_to_gpu_matrices(renderer)
}

// ---------------------------------------------------------------------------
// Buffer size utilities
// ---------------------------------------------------------------------------

/// Maximum number of bones supported by the GPU pipeline.
pub const MAX_GPU_BONES: usize = 256;

/// Calculate the GPU buffer size needed for a bone palette.
///
/// Each bone requires one 4x4 matrix = 64 bytes.
pub const fn bone_palette_buffer_size(bone_count: usize) -> usize {
    bone_count * 64 // 4x4 * 4 bytes per float
}

/// Calculate the GPU buffer size for a dual quaternion palette.
///
/// Each bone requires 8 floats = 32 bytes.
pub const fn dq_palette_buffer_size(bone_count: usize) -> usize {
    bone_count * 32 // 2 quaternions * 4 floats * 4 bytes
}

/// Validate that a bone count is within GPU limits.
pub fn validate_bone_count(bone_count: usize) -> Result<(), String> {
    if bone_count > MAX_GPU_BONES {
        Err(format!(
            "Skeleton has {} bones, exceeding GPU limit of {}",
            bone_count, MAX_GPU_BONES
        ))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Interpolation helpers for smooth transitions
// ---------------------------------------------------------------------------

/// Linearly interpolate two sets of GPU matrices.
///
/// Useful for blending between animation states at the matrix level
/// (though blending at the pose level via `blend_poses` is preferred).
pub fn lerp_gpu_matrices(
    a: &[[f32; 16]],
    b: &[[f32; 16]],
    t: f32,
) -> Vec<[f32; 16]> {
    assert_eq!(a.len(), b.len(), "Matrix arrays must have same length");
    let t = t.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;
    a.iter()
        .zip(b.iter())
        .map(|(ma, mb)| {
            let mut result = [0.0f32; 16];
            for i in 0..16 {
                result[i] = ma[i] * one_minus_t + mb[i] * t;
            }
            result
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Bone;

    fn make_test_skeleton() -> Skeleton {
        let bones = vec![
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
        ];
        Skeleton::new("TestSkeleton", bones)
    }

    #[test]
    fn test_compute_skinning_matrices() {
        let skel = make_test_skeleton();
        let poses = vec![
            Transform::IDENTITY,
            Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
            Transform::from_position(Vec3::new(0.0, 0.5, 0.0)),
        ];
        let matrices = compute_skinning_matrices(&skel, &poses);
        assert_eq!(matrices.len(), 3);
        // Each matrix should be 16 floats
        for m in &matrices {
            assert_eq!(m.len(), 16);
        }
    }

    #[test]
    fn test_transform_to_dual_quat() {
        let dq = transform_to_dual_quat(Quat::IDENTITY, Vec3::ZERO);
        assert!((dq.real[3] - 1.0).abs() < 1e-5, "Identity rotation w should be 1");
        assert!((dq.dual[0]).abs() < 1e-5, "Zero translation dual x should be 0");
    }

    #[test]
    fn test_matrices_to_dual_quaternions() {
        let matrices = vec![Mat4::IDENTITY; 3];
        let dqs = matrices_to_dual_quaternions(&matrices);
        assert_eq!(dqs.len(), 3);
        for dq in &dqs {
            assert!((dq.real[3] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_lerp_gpu_matrices() {
        let a = vec![[0.0f32; 16]; 2];
        let b = vec![[1.0f32; 16]; 2];
        let result = lerp_gpu_matrices(&a, &b, 0.5);
        assert_eq!(result.len(), 2);
        for m in &result {
            for &v in m {
                assert!((v - 0.5).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_lerp_gpu_matrices_at_zero() {
        let a = vec![[1.0f32; 16]];
        let b = vec![[2.0f32; 16]];
        let result = lerp_gpu_matrices(&a, &b, 0.0);
        for &v in &result[0] {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_lerp_gpu_matrices_at_one() {
        let a = vec![[1.0f32; 16]];
        let b = vec![[2.0f32; 16]];
        let result = lerp_gpu_matrices(&a, &b, 1.0);
        for &v in &result[0] {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_validate_bone_count() {
        assert!(validate_bone_count(64).is_ok());
        assert!(validate_bone_count(256).is_ok());
        assert!(validate_bone_count(257).is_err());
    }

    #[test]
    fn test_bone_palette_buffer_size() {
        assert_eq!(bone_palette_buffer_size(1), 64);
        assert_eq!(bone_palette_buffer_size(256), 16384);
    }

    #[test]
    fn test_dq_palette_buffer_size() {
        assert_eq!(dq_palette_buffer_size(1), 32);
        assert_eq!(dq_palette_buffer_size(256), 8192);
    }

    #[test]
    fn test_renderer_to_gpu_matrices() {
        let renderer = SkinnedMeshRenderer {
            skeleton_name: "test".to_string(),
            bone_matrices: vec![Mat4::IDENTITY; 3],
            max_bones_per_vertex: 4,
            enabled: true,
        };
        let gpu = renderer_to_gpu_matrices(&renderer);
        assert_eq!(gpu.len(), 3);
        // Identity matrix check: column 0 = [1,0,0,0], etc.
        assert!((gpu[0][0] - 1.0).abs() < 1e-5);
        assert!((gpu[0][5] - 1.0).abs() < 1e-5);
        assert!((gpu[0][10] - 1.0).abs() < 1e-5);
        assert!((gpu[0][15] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_update_skinned_renderer() {
        let skel = make_test_skeleton();
        let mut renderer = SkinnedMeshRenderer {
            skeleton_name: "test".to_string(),
            bone_matrices: Vec::new(),
            max_bones_per_vertex: 4,
            enabled: true,
        };
        let pose = skel.bind_pose();
        update_skinned_renderer(&mut renderer, &skel, &pose);
        assert_eq!(renderer.bone_matrices.len(), 3);
    }

    #[test]
    fn test_update_skinned_renderer_disabled() {
        let skel = make_test_skeleton();
        let mut renderer = SkinnedMeshRenderer {
            skeleton_name: "test".to_string(),
            bone_matrices: Vec::new(),
            max_bones_per_vertex: 4,
            enabled: false,
        };
        let pose = skel.bind_pose();
        update_skinned_renderer(&mut renderer, &skel, &pose);
        // Should not update when disabled
        assert!(renderer.bone_matrices.is_empty());
    }

    #[test]
    fn test_gpu_dual_quat_default() {
        let dq = GpuDualQuat::default();
        assert_eq!(dq.real, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(dq.dual, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dual_quat_with_translation() {
        let dq = transform_to_dual_quat(Quat::IDENTITY, Vec3::new(2.0, 0.0, 0.0));
        // For identity rotation, dual part encodes half the translation
        // dual = 0.5 * [tx, ty, tz, 0] * [0, 0, 0, 1] = 0.5 * [tx, ty, tz, 0]
        // But quaternion multiplication is more involved. Let's just check it's non-zero.
        let dual_magnitude = (dq.dual[0] * dq.dual[0] + dq.dual[1] * dq.dual[1]
            + dq.dual[2] * dq.dual[2] + dq.dual[3] * dq.dual[3])
            .sqrt();
        assert!(dual_magnitude > 0.0, "Dual part should be non-zero for non-zero translation");
    }
}
