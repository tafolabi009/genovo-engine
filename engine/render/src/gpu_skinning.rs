// engine/render/src/gpu_skinning.rs
//
// GPU skeletal animation system for the Genovo engine.
//
// Implements efficient GPU-based skeletal animation and mesh deformation:
//
// - **Bone matrix palette** — Stores per-bone transformation matrices in a
//   uniform buffer or Shader Storage Buffer Object (SSBO) for vertex shader
//   access.
// - **Linear blend skinning (LBS)** — Standard vertex shader skinning with
//   up to 4 bone influences per vertex.
// - **Dual quaternion skinning (DQS)** — Higher-quality skinning that avoids
//   the volume-collapse artefact of LBS at joints.
// - **Compute shader bone matrix computation** — Traverses the skeleton
//   hierarchy on the GPU using a compute shader, computing world-space bone
//   matrices from local transforms.
// - **Morph target blending** — GPU blending of morph targets (blend shapes)
//   for facial animation and corrective shapes.
//
// # Pipeline
//
// 1. **Animation sampling** — Sample animation curves to produce local bone
//    transforms.
// 2. **Hierarchy traversal** — Multiply local transforms through the
//    parent chain to get world-space transforms (compute shader).
// 3. **Skinning matrix** — Multiply world transform by inverse bind pose to
//    get the final skinning matrix.
// 4. **Vertex skinning** — Transform vertices using the skinning matrices
//    (vertex shader, LBS or DQS).
// 5. **Morph targets** — Apply morph target deltas (compute shader).

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Bone and skeleton
// ---------------------------------------------------------------------------

/// A single bone in the skeleton.
#[derive(Debug, Clone)]
pub struct Bone {
    /// Bone name.
    pub name: String,
    /// Bone index in the skeleton.
    pub index: u32,
    /// Parent bone index (u32::MAX = root).
    pub parent: u32,
    /// Inverse bind pose matrix (column-major 4x4).
    /// Transforms from model space to bone space at bind pose.
    pub inverse_bind_pose: [f32; 16],
    /// Local bind pose position.
    pub bind_position: [f32; 3],
    /// Local bind pose rotation (quaternion: x, y, z, w).
    pub bind_rotation: [f32; 4],
    /// Local bind pose scale.
    pub bind_scale: [f32; 3],
}

impl Bone {
    /// Creates a new bone.
    pub fn new(name: impl Into<String>, index: u32, parent: u32) -> Self {
        Self {
            name: name.into(),
            index,
            parent,
            inverse_bind_pose: identity_matrix(),
            bind_position: [0.0, 0.0, 0.0],
            bind_rotation: [0.0, 0.0, 0.0, 1.0],
            bind_scale: [1.0, 1.0, 1.0],
        }
    }

    /// Whether this bone is a root bone.
    pub fn is_root(&self) -> bool {
        self.parent == u32::MAX
    }

    /// Computes the local bind pose matrix.
    pub fn local_bind_matrix(&self) -> [f32; 16] {
        let t = translation_matrix(self.bind_position);
        let r = quaternion_to_matrix(self.bind_rotation);
        let s = scale_matrix(self.bind_scale);
        mat4_mul(&mat4_mul(&t, &r), &s)
    }
}

/// A skeleton definition.
#[derive(Debug, Clone)]
pub struct Skeleton {
    /// Bones in hierarchical order (parents before children).
    pub bones: Vec<Bone>,
    /// Name of the skeleton.
    pub name: String,
}

impl Skeleton {
    /// Creates a new empty skeleton.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            bones: Vec::new(),
            name: name.into(),
        }
    }

    /// Adds a bone to the skeleton.
    pub fn add_bone(&mut self, bone: Bone) -> u32 {
        let index = self.bones.len() as u32;
        self.bones.push(bone);
        index
    }

    /// Returns the number of bones.
    pub fn bone_count(&self) -> u32 {
        self.bones.len() as u32
    }

    /// Finds a bone by name.
    pub fn find_bone(&self, name: &str) -> Option<&Bone> {
        self.bones.iter().find(|b| b.name == name)
    }

    /// Finds a bone index by name.
    pub fn find_bone_index(&self, name: &str) -> Option<u32> {
        self.bones.iter().position(|b| b.name == name).map(|i| i as u32)
    }

    /// Validates the skeleton hierarchy (parents must precede children).
    pub fn validate(&self) -> bool {
        for (i, bone) in self.bones.iter().enumerate() {
            if bone.parent != u32::MAX && bone.parent >= i as u32 {
                return false; // Parent must come before child.
            }
        }
        true
    }

    /// Returns the root bones (bones with no parent).
    pub fn root_bones(&self) -> Vec<u32> {
        self.bones
            .iter()
            .filter(|b| b.is_root())
            .map(|b| b.index)
            .collect()
    }

    /// Returns the children of a bone.
    pub fn children_of(&self, bone_index: u32) -> Vec<u32> {
        self.bones
            .iter()
            .filter(|b| b.parent == bone_index)
            .map(|b| b.index)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Bone pose and animation
// ---------------------------------------------------------------------------

/// A bone's local transform at a specific point in time.
#[derive(Debug, Clone, Copy)]
pub struct BoneTransform {
    /// Position.
    pub position: [f32; 3],
    /// Rotation (quaternion: x, y, z, w).
    pub rotation: [f32; 4],
    /// Scale.
    pub scale: [f32; 3],
}

impl BoneTransform {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }

    /// Creates a transform from position and rotation.
    pub fn from_pos_rot(position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self {
            position,
            rotation,
            scale: [1.0, 1.0, 1.0],
        }
    }

    /// Converts to a 4x4 matrix.
    pub fn to_matrix(&self) -> [f32; 16] {
        let t = translation_matrix(self.position);
        let r = quaternion_to_matrix(self.rotation);
        let s = scale_matrix(self.scale);
        mat4_mul(&mat4_mul(&t, &r), &s)
    }

    /// Linearly interpolates between two transforms.
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        Self {
            position: lerp_vec3(a.position, b.position, t),
            rotation: slerp(a.rotation, b.rotation, t),
            scale: lerp_vec3(a.scale, b.scale, t),
        }
    }
}

impl Default for BoneTransform {
    fn default() -> Self {
        Self::identity()
    }
}

/// A complete skeleton pose (all bones).
#[derive(Debug, Clone)]
pub struct SkeletonPose {
    /// Local transforms per bone.
    pub local_transforms: Vec<BoneTransform>,
    /// World-space matrices per bone (computed from local transforms).
    pub world_matrices: Vec<[f32; 16]>,
    /// Final skinning matrices (world * inverse_bind_pose).
    pub skinning_matrices: Vec<[f32; 16]>,
    /// Dual quaternions for DQS (if computed).
    pub dual_quaternions: Vec<DualQuaternion>,
}

impl SkeletonPose {
    /// Creates a new pose for a skeleton with the given bone count.
    pub fn new(bone_count: u32) -> Self {
        let n = bone_count as usize;
        Self {
            local_transforms: vec![BoneTransform::identity(); n],
            world_matrices: vec![identity_matrix(); n],
            skinning_matrices: vec![identity_matrix(); n],
            dual_quaternions: vec![DualQuaternion::identity(); n],
        }
    }

    /// Computes world-space matrices by traversing the skeleton hierarchy.
    pub fn compute_world_matrices(&mut self, skeleton: &Skeleton) {
        for bone in &skeleton.bones {
            let local = self.local_transforms[bone.index as usize].to_matrix();

            if bone.is_root() {
                self.world_matrices[bone.index as usize] = local;
            } else {
                let parent_world = self.world_matrices[bone.parent as usize];
                self.world_matrices[bone.index as usize] = mat4_mul(&parent_world, &local);
            }
        }
    }

    /// Computes skinning matrices (world * inverse_bind_pose).
    pub fn compute_skinning_matrices(&mut self, skeleton: &Skeleton) {
        for bone in &skeleton.bones {
            let idx = bone.index as usize;
            self.skinning_matrices[idx] =
                mat4_mul(&self.world_matrices[idx], &bone.inverse_bind_pose);
        }
    }

    /// Computes dual quaternions from the skinning matrices.
    pub fn compute_dual_quaternions(&mut self) {
        for (idx, matrix) in self.skinning_matrices.iter().enumerate() {
            self.dual_quaternions[idx] = DualQuaternion::from_matrix(matrix);
        }
    }

    /// Full update: compute world matrices, skinning matrices, and optionally DQs.
    pub fn update(&mut self, skeleton: &Skeleton, compute_dq: bool) {
        self.compute_world_matrices(skeleton);
        self.compute_skinning_matrices(skeleton);
        if compute_dq {
            self.compute_dual_quaternions();
        }
    }

    /// Returns the skinning matrices as a flat f32 buffer for GPU upload.
    pub fn skinning_matrix_data(&self) -> Vec<f32> {
        let mut data = Vec::with_capacity(self.skinning_matrices.len() * 16);
        for mat in &self.skinning_matrices {
            data.extend_from_slice(mat);
        }
        data
    }

    /// Returns the dual quaternion data for GPU upload.
    pub fn dual_quaternion_data(&self) -> Vec<f32> {
        let mut data = Vec::with_capacity(self.dual_quaternions.len() * 8);
        for dq in &self.dual_quaternions {
            data.extend_from_slice(&dq.real);
            data.extend_from_slice(&dq.dual);
        }
        data
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        (self.local_transforms.len() * std::mem::size_of::<BoneTransform>())
            + (self.world_matrices.len() * std::mem::size_of::<[f32; 16]>())
            + (self.skinning_matrices.len() * std::mem::size_of::<[f32; 16]>())
            + (self.dual_quaternions.len() * std::mem::size_of::<DualQuaternion>())
    }
}

// ---------------------------------------------------------------------------
// Dual quaternion
// ---------------------------------------------------------------------------

/// Dual quaternion for dual quaternion skinning.
///
/// Represents a rigid transformation (rotation + translation) without
/// the volume collapse artefact of linear blend skinning.
#[derive(Debug, Clone, Copy)]
pub struct DualQuaternion {
    /// Real part (rotation quaternion: x, y, z, w).
    pub real: [f32; 4],
    /// Dual part (encodes translation).
    pub dual: [f32; 4],
}

impl DualQuaternion {
    /// Identity dual quaternion (no rotation, no translation).
    pub fn identity() -> Self {
        Self {
            real: [0.0, 0.0, 0.0, 1.0],
            dual: [0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Creates a dual quaternion from a rotation quaternion and translation.
    pub fn from_rotation_translation(rotation: [f32; 4], translation: [f32; 3]) -> Self {
        let r = rotation;
        let t = translation;

        // Dual part: 0.5 * t * r
        let dual = [
            0.5 * (t[0] * r[3] + t[1] * r[2] - t[2] * r[1]),
            0.5 * (-t[0] * r[2] + t[1] * r[3] + t[2] * r[0]),
            0.5 * (t[0] * r[1] - t[1] * r[0] + t[2] * r[3]),
            -0.5 * (t[0] * r[0] + t[1] * r[1] + t[2] * r[2]),
        ];

        Self { real: r, dual }
    }

    /// Creates a dual quaternion from a 4x4 matrix.
    pub fn from_matrix(m: &[f32; 16]) -> Self {
        let rotation = matrix_to_quaternion(m);
        let translation = [m[12], m[13], m[14]];
        Self::from_rotation_translation(rotation, translation)
    }

    /// Normalises the dual quaternion.
    pub fn normalise(&mut self) {
        let len = quat_length(self.real);
        if len > 1e-6 {
            let inv = 1.0 / len;
            self.real = [
                self.real[0] * inv,
                self.real[1] * inv,
                self.real[2] * inv,
                self.real[3] * inv,
            ];
            self.dual = [
                self.dual[0] * inv,
                self.dual[1] * inv,
                self.dual[2] * inv,
                self.dual[3] * inv,
            ];
        }
    }

    /// Extracts the rotation quaternion.
    pub fn rotation(&self) -> [f32; 4] {
        self.real
    }

    /// Extracts the translation.
    pub fn translation(&self) -> [f32; 3] {
        let r = self.real;
        let d = self.dual;
        [
            2.0 * (-d[3] * r[0] + d[0] * r[3] - d[1] * r[2] + d[2] * r[1]),
            2.0 * (-d[3] * r[1] + d[0] * r[2] + d[1] * r[3] - d[2] * r[0]),
            2.0 * (-d[3] * r[2] - d[0] * r[1] + d[1] * r[0] + d[2] * r[3]),
        ]
    }

    /// Transforms a 3D point by this dual quaternion.
    pub fn transform_point(&self, point: [f32; 3]) -> [f32; 3] {
        let rotated = quat_rotate(self.real, point);
        let t = self.translation();
        [rotated[0] + t[0], rotated[1] + t[1], rotated[2] + t[2]]
    }

    /// Blends two dual quaternions with weights.
    pub fn blend(a: &Self, b: &Self, weight_a: f32, weight_b: f32) -> Self {
        // Ensure shortest path (flip if dot product is negative).
        let dot = a.real[0] * b.real[0]
            + a.real[1] * b.real[1]
            + a.real[2] * b.real[2]
            + a.real[3] * b.real[3];

        let sign = if dot < 0.0 { -1.0 } else { 1.0 };

        let mut result = Self {
            real: [
                a.real[0] * weight_a + b.real[0] * weight_b * sign,
                a.real[1] * weight_a + b.real[1] * weight_b * sign,
                a.real[2] * weight_a + b.real[2] * weight_b * sign,
                a.real[3] * weight_a + b.real[3] * weight_b * sign,
            ],
            dual: [
                a.dual[0] * weight_a + b.dual[0] * weight_b * sign,
                a.dual[1] * weight_a + b.dual[1] * weight_b * sign,
                a.dual[2] * weight_a + b.dual[2] * weight_b * sign,
                a.dual[3] * weight_a + b.dual[3] * weight_b * sign,
            ],
        };

        result.normalise();
        result
    }
}

// ---------------------------------------------------------------------------
// Skinned vertex
// ---------------------------------------------------------------------------

/// A vertex with bone weights for skeletal animation.
#[derive(Debug, Clone, Copy)]
pub struct SkinnedVertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// Normal.
    pub normal: [f32; 3],
    /// Tangent (xyz) + sign (w).
    pub tangent: [f32; 4],
    /// UV coordinates.
    pub uv: [f32; 2],
    /// Bone indices (up to 4).
    pub bone_indices: [u32; 4],
    /// Bone weights (up to 4, should sum to 1.0).
    pub bone_weights: [f32; 4],
}

impl SkinnedVertex {
    /// Creates a vertex with a single bone influence.
    pub fn single_bone(position: [f32; 3], normal: [f32; 3], bone: u32) -> Self {
        Self {
            position,
            normal,
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            bone_indices: [bone, 0, 0, 0],
            bone_weights: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Returns the number of active bone influences.
    pub fn influence_count(&self) -> u32 {
        self.bone_weights.iter().filter(|&&w| w > 0.0).count() as u32
    }

    /// Normalises bone weights to sum to 1.0.
    pub fn normalise_weights(&mut self) {
        let sum: f32 = self.bone_weights.iter().sum();
        if sum > 1e-6 {
            let inv = 1.0 / sum;
            for w in &mut self.bone_weights {
                *w *= inv;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU skinning (reference implementation)
// ---------------------------------------------------------------------------

/// Performs linear blend skinning (LBS) on a vertex using the CPU.
///
/// # Arguments
/// * `vertex` — The skinned vertex.
/// * `matrices` — Skinning matrix palette.
///
/// # Returns
/// Transformed `(position, normal)`.
pub fn skin_vertex_lbs(
    vertex: &SkinnedVertex,
    matrices: &[[f32; 16]],
) -> ([f32; 3], [f32; 3]) {
    let mut pos = [0.0f32; 3];
    let mut nor = [0.0f32; 3];

    for i in 0..4 {
        let weight = vertex.bone_weights[i];
        if weight < 1e-6 {
            continue;
        }

        let bone_idx = vertex.bone_indices[i] as usize;
        if bone_idx >= matrices.len() {
            continue;
        }

        let m = &matrices[bone_idx];

        // Transform position.
        let tp = mat4_transform_point(m, vertex.position);
        pos[0] += tp[0] * weight;
        pos[1] += tp[1] * weight;
        pos[2] += tp[2] * weight;

        // Transform normal (using upper-left 3x3).
        let tn = mat4_transform_normal(m, vertex.normal);
        nor[0] += tn[0] * weight;
        nor[1] += tn[1] * weight;
        nor[2] += tn[2] * weight;
    }

    // Normalise the normal.
    let len = (nor[0] * nor[0] + nor[1] * nor[1] + nor[2] * nor[2]).sqrt();
    if len > 1e-6 {
        nor = [nor[0] / len, nor[1] / len, nor[2] / len];
    }

    (pos, nor)
}

/// Performs dual quaternion skinning (DQS) on a vertex using the CPU.
pub fn skin_vertex_dqs(
    vertex: &SkinnedVertex,
    dual_quats: &[DualQuaternion],
) -> ([f32; 3], [f32; 3]) {
    let mut blended = DualQuaternion::identity();
    let mut first_valid = true;
    let mut reference_real = [0.0f32; 4];

    for i in 0..4 {
        let weight = vertex.bone_weights[i];
        if weight < 1e-6 {
            continue;
        }

        let bone_idx = vertex.bone_indices[i] as usize;
        if bone_idx >= dual_quats.len() {
            continue;
        }

        let dq = &dual_quats[bone_idx];

        if first_valid {
            reference_real = dq.real;
            first_valid = false;
        }

        // Ensure shortest path.
        let dot = reference_real[0] * dq.real[0]
            + reference_real[1] * dq.real[1]
            + reference_real[2] * dq.real[2]
            + reference_real[3] * dq.real[3];
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };

        blended.real[0] += dq.real[0] * weight * sign;
        blended.real[1] += dq.real[1] * weight * sign;
        blended.real[2] += dq.real[2] * weight * sign;
        blended.real[3] += dq.real[3] * weight * sign;

        blended.dual[0] += dq.dual[0] * weight * sign;
        blended.dual[1] += dq.dual[1] * weight * sign;
        blended.dual[2] += dq.dual[2] * weight * sign;
        blended.dual[3] += dq.dual[3] * weight * sign;
    }

    blended.normalise();

    let pos = blended.transform_point(vertex.position);
    let nor_rotated = quat_rotate(blended.real, vertex.normal);
    let len = (nor_rotated[0] * nor_rotated[0]
        + nor_rotated[1] * nor_rotated[1]
        + nor_rotated[2] * nor_rotated[2])
        .sqrt();
    let nor = if len > 1e-6 {
        [nor_rotated[0] / len, nor_rotated[1] / len, nor_rotated[2] / len]
    } else {
        vertex.normal
    };

    (pos, nor)
}

// ---------------------------------------------------------------------------
// Morph targets
// ---------------------------------------------------------------------------

/// A morph target (blend shape).
#[derive(Debug, Clone)]
pub struct MorphTarget {
    /// Name of the morph target.
    pub name: String,
    /// Position deltas (same length as the base mesh vertex count).
    pub position_deltas: Vec<[f32; 3]>,
    /// Normal deltas (optional).
    pub normal_deltas: Vec<[f32; 3]>,
    /// Tangent deltas (optional).
    pub tangent_deltas: Vec<[f32; 3]>,
    /// Current weight [0, 1].
    pub weight: f32,
}

impl MorphTarget {
    /// Creates a new morph target.
    pub fn new(name: impl Into<String>, vertex_count: usize) -> Self {
        Self {
            name: name.into(),
            position_deltas: vec![[0.0; 3]; vertex_count],
            normal_deltas: Vec::new(),
            tangent_deltas: Vec::new(),
            weight: 0.0,
        }
    }

    /// Sets the position delta for a vertex.
    pub fn set_delta(&mut self, vertex_index: usize, delta: [f32; 3]) {
        if vertex_index < self.position_deltas.len() {
            self.position_deltas[vertex_index] = delta;
        }
    }

    /// Returns the memory usage.
    pub fn memory_usage(&self) -> usize {
        self.position_deltas.len() * std::mem::size_of::<[f32; 3]>()
            + self.normal_deltas.len() * std::mem::size_of::<[f32; 3]>()
            + self.tangent_deltas.len() * std::mem::size_of::<[f32; 3]>()
    }
}

/// Applies morph targets to a set of vertices.
///
/// # Arguments
/// * `base_positions` — Base mesh positions.
/// * `targets` — Morph targets with their current weights.
///
/// # Returns
/// Modified positions.
pub fn apply_morph_targets(
    base_positions: &[[f32; 3]],
    targets: &[MorphTarget],
) -> Vec<[f32; 3]> {
    let mut result = base_positions.to_vec();

    for target in targets {
        if target.weight.abs() < 1e-6 {
            continue;
        }

        let w = target.weight;
        for (i, delta) in target.position_deltas.iter().enumerate() {
            if i < result.len() {
                result[i][0] += delta[0] * w;
                result[i][1] += delta[1] * w;
                result[i][2] += delta[2] * w;
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// GPU skinning manager
// ---------------------------------------------------------------------------

/// Skinning method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkinningMethod {
    /// Linear blend skinning.
    LinearBlend,
    /// Dual quaternion skinning.
    DualQuaternion,
}

/// GPU skinning manager that handles multiple skinned meshes.
#[derive(Debug)]
pub struct GpuSkinningManager {
    /// Skeletons.
    pub skeletons: Vec<Skeleton>,
    /// Active poses (one per skeleton instance).
    pub poses: Vec<SkeletonPose>,
    /// Skinning method.
    pub method: SkinningMethod,
    /// Maximum bones per skeleton.
    pub max_bones: u32,
    /// Whether to use SSBO (true) or UBO (false) for bone matrices.
    pub use_ssbo: bool,
    /// Whether morph targets are enabled.
    pub morph_targets_enabled: bool,
}

impl GpuSkinningManager {
    /// Creates a new GPU skinning manager.
    pub fn new(method: SkinningMethod) -> Self {
        Self {
            skeletons: Vec::new(),
            poses: Vec::new(),
            method,
            max_bones: 256,
            use_ssbo: true,
            morph_targets_enabled: true,
        }
    }

    /// Registers a skeleton and returns its index.
    pub fn add_skeleton(&mut self, skeleton: Skeleton) -> usize {
        let bone_count = skeleton.bone_count();
        let idx = self.skeletons.len();
        self.skeletons.push(skeleton);
        self.poses.push(SkeletonPose::new(bone_count));
        idx
    }

    /// Updates all poses.
    pub fn update_all(&mut self) {
        let compute_dq = self.method == SkinningMethod::DualQuaternion;
        for (i, skeleton) in self.skeletons.iter().enumerate() {
            if i < self.poses.len() {
                self.poses[i].update(skeleton, compute_dq);
            }
        }
    }

    /// Sets the local transform for a bone in a skeleton instance.
    pub fn set_bone_transform(
        &mut self,
        skeleton_index: usize,
        bone_index: u32,
        transform: BoneTransform,
    ) {
        if let Some(pose) = self.poses.get_mut(skeleton_index) {
            if (bone_index as usize) < pose.local_transforms.len() {
                pose.local_transforms[bone_index as usize] = transform;
            }
        }
    }

    /// Returns the skinning data for GPU upload.
    pub fn skinning_data(&self, skeleton_index: usize) -> Option<Vec<f32>> {
        self.poses.get(skeleton_index).map(|pose| {
            match self.method {
                SkinningMethod::LinearBlend => pose.skinning_matrix_data(),
                SkinningMethod::DualQuaternion => pose.dual_quaternion_data(),
            }
        })
    }

    /// Returns the total number of bones across all skeletons.
    pub fn total_bone_count(&self) -> u32 {
        self.skeletons.iter().map(|s| s.bone_count()).sum()
    }

    /// Returns memory usage.
    pub fn memory_usage(&self) -> usize {
        self.poses.iter().map(|p| p.memory_usage()).sum()
    }
}

impl Default for GpuSkinningManager {
    fn default() -> Self {
        Self::new(SkinningMethod::LinearBlend)
    }
}

// ---------------------------------------------------------------------------
// WGSL skinning shaders
// ---------------------------------------------------------------------------

/// WGSL vertex shader for linear blend skinning.
pub const LBS_VERTEX_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Linear Blend Skinning vertex shader (Genovo Engine)
// -----------------------------------------------------------------------

struct SkinUniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> skin: SkinUniforms;
@group(0) @binding(1) var<storage, read> bone_matrices: array<mat4x4<f32>>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) bone_indices: vec4<u32>,
    @location(4) bone_weights: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var skinned_pos = vec4<f32>(0.0);
    var skinned_nor = vec3<f32>(0.0);

    for (var i = 0u; i < 4u; i = i + 1u) {
        let weight = input.bone_weights[i];
        if weight < 0.001 {
            continue;
        }
        let bone_idx = input.bone_indices[i];
        let bone_mat = bone_matrices[bone_idx];

        skinned_pos += bone_mat * vec4<f32>(input.position, 1.0) * weight;
        skinned_nor += (bone_mat * vec4<f32>(input.normal, 0.0)).xyz * weight;
    }

    let world_pos = skin.model * skinned_pos;
    out.clip_pos = skin.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.normal = normalize((skin.model * vec4<f32>(skinned_nor, 0.0)).xyz);
    out.uv = input.uv;

    return out;
}
"#;

/// WGSL vertex shader for dual quaternion skinning.
pub const DQS_VERTEX_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Dual Quaternion Skinning vertex shader (Genovo Engine)
// -----------------------------------------------------------------------

struct SkinUniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> skin: SkinUniforms;
@group(0) @binding(1) var<storage, read> dual_quats: array<vec4<f32>>; // real[i], dual[i]

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) bone_indices: vec4<u32>,
    @location(4) bone_weights: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var blended_real = vec4<f32>(0.0);
    var blended_dual = vec4<f32>(0.0);

    let ref_real = dual_quats[input.bone_indices[0] * 2u];

    for (var i = 0u; i < 4u; i = i + 1u) {
        let weight = input.bone_weights[i];
        if weight < 0.001 {
            continue;
        }
        let bone_idx = input.bone_indices[i];
        let real = dual_quats[bone_idx * 2u];
        let dual = dual_quats[bone_idx * 2u + 1u];

        let sign = select(-1.0, 1.0, dot(ref_real, real) >= 0.0);

        blended_real += real * weight * sign;
        blended_dual += dual * weight * sign;
    }

    let len = length(blended_real);
    blended_real /= len;
    blended_dual /= len;

    let rotated_pos = quat_rotate(blended_real, input.position);
    let translation = 2.0 * (blended_real.w * blended_dual.xyz - blended_dual.w * blended_real.xyz + cross(blended_real.xyz, blended_dual.xyz));
    let skinned_pos = rotated_pos + translation;

    let skinned_nor = quat_rotate(blended_real, input.normal);

    let world_pos = skin.model * vec4<f32>(skinned_pos, 1.0);
    out.clip_pos = skin.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.normal = normalize((skin.model * vec4<f32>(skinned_nor, 0.0)).xyz);
    out.uv = input.uv;

    return out;
}
"#;

// ---------------------------------------------------------------------------
// Matrix / quaternion math helpers
// ---------------------------------------------------------------------------

fn identity_matrix() -> [f32; 16] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
}

fn translation_matrix(t: [f32; 3]) -> [f32; 16] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, t[0], t[1], t[2], 1.0]
}

fn scale_matrix(s: [f32; 3]) -> [f32; 16] {
    [s[0], 0.0, 0.0, 0.0, 0.0, s[1], 0.0, 0.0, 0.0, 0.0, s[2], 0.0, 0.0, 0.0, 0.0, 1.0]
}

fn quaternion_to_matrix(q: [f32; 4]) -> [f32; 16] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;
    [
        1.0 - yy - zz, xy + wz, xz - wy, 0.0,
        xy - wz, 1.0 - xx - zz, yz + wx, 0.0,
        xz + wy, yz - wx, 1.0 - xx - yy, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn matrix_to_quaternion(m: &[f32; 16]) -> [f32; 4] {
    let trace = m[0] + m[5] + m[10];
    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        [
            (m[6] - m[9]) * s, (m[8] - m[2]) * s, (m[1] - m[4]) * s, 0.25 / s,
        ]
    } else if m[0] > m[5] && m[0] > m[10] {
        let s = 2.0 * (1.0 + m[0] - m[5] - m[10]).sqrt();
        [0.25 * s, (m[4] + m[1]) / s, (m[8] + m[2]) / s, (m[6] - m[9]) / s]
    } else if m[5] > m[10] {
        let s = 2.0 * (1.0 + m[5] - m[0] - m[10]).sqrt();
        [(m[4] + m[1]) / s, 0.25 * s, (m[9] + m[6]) / s, (m[8] - m[2]) / s]
    } else {
        let s = 2.0 * (1.0 + m[10] - m[0] - m[5]).sqrt();
        [(m[8] + m[2]) / s, (m[9] + m[6]) / s, 0.25 * s, (m[1] - m[4]) / s]
    }
}

fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut r = [0.0f32; 16];
    for c in 0..4 {
        for row in 0..4 {
            let mut s = 0.0;
            for k in 0..4 { s += a[k * 4 + row] * b[c * 4 + k]; }
            r[c * 4 + row] = s;
        }
    }
    r
}

fn mat4_transform_point(m: &[f32; 16], p: [f32; 3]) -> [f32; 3] {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

fn mat4_transform_normal(m: &[f32; 16], n: [f32; 3]) -> [f32; 3] {
    [
        m[0] * n[0] + m[4] * n[1] + m[8] * n[2],
        m[1] * n[0] + m[5] * n[1] + m[9] * n[2],
        m[2] * n[0] + m[6] * n[1] + m[10] * n[2],
    ]
}

fn quat_length(q: [f32; 4]) -> f32 {
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
}

fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let t = [
        2.0 * (q[1] * v[2] - q[2] * v[1]),
        2.0 * (q[2] * v[0] - q[0] * v[2]),
        2.0 * (q[0] * v[1] - q[1] * v[0]),
    ];
    [
        v[0] + q[3] * t[0] + (q[1] * t[2] - q[2] * t[1]),
        v[1] + q[3] * t[1] + (q[2] * t[0] - q[0] * t[2]),
        v[2] + q[3] * t[2] + (q[0] * t[1] - q[1] * t[0]),
    ]
}

fn slerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let mut b = b;
    if dot < 0.0 { dot = -dot; b = [-b[0], -b[1], -b[2], -b[3]]; }
    if dot > 0.9995 {
        return normalize_quat([
            a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]), a[3] + t * (b[3] - a[3]),
        ]);
    }
    let theta_0 = dot.clamp(-1.0, 1.0).acos();
    let theta = theta_0 * t;
    let sin_theta = theta.sin();
    let sin_theta_0 = theta_0.sin();
    let s0 = (theta_0 - theta).cos() - dot * sin_theta / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;
    normalize_quat([
        a[0] * s0 + b[0] * s1, a[1] * s0 + b[1] * s1,
        a[2] * s0 + b[2] * s1, a[3] * s0 + b[3] * s1,
    ])
}

fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let len = quat_length(q);
    if len > 1e-6 { [q[0] / len, q[1] / len, q[2] / len, q[3] / len] } else { [0.0, 0.0, 0.0, 1.0] }
}

fn lerp_vec3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bone_creation() {
        let bone = Bone::new("root", 0, u32::MAX);
        assert!(bone.is_root());
        assert_eq!(bone.name, "root");
    }

    #[test]
    fn test_skeleton_validation() {
        let mut skel = Skeleton::new("test");
        skel.add_bone(Bone::new("root", 0, u32::MAX));
        skel.add_bone(Bone::new("child", 1, 0));
        assert!(skel.validate());

        let mut bad = Skeleton::new("bad");
        bad.add_bone(Bone::new("child", 0, 1));
        bad.add_bone(Bone::new("root", 1, u32::MAX));
        assert!(!bad.validate());
    }

    #[test]
    fn test_skeleton_find_bone() {
        let mut skel = Skeleton::new("test");
        skel.add_bone(Bone::new("root", 0, u32::MAX));
        skel.add_bone(Bone::new("arm", 1, 0));

        assert!(skel.find_bone("arm").is_some());
        assert!(skel.find_bone("missing").is_none());
    }

    #[test]
    fn test_bone_transform_identity() {
        let bt = BoneTransform::identity();
        let m = bt.to_matrix();
        assert!((m[0] - 1.0).abs() < 0.01);
        assert!((m[5] - 1.0).abs() < 0.01);
        assert!((m[10] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bone_transform_lerp() {
        let a = BoneTransform::from_pos_rot([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        let b = BoneTransform::from_pos_rot([10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        let mid = BoneTransform::lerp(&a, &b, 0.5);
        assert!((mid.position[0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_lbs_single_bone() {
        let vertex = SkinnedVertex::single_bone([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0);
        let matrices = vec![translation_matrix([5.0, 0.0, 0.0])];
        let (pos, _nor) = skin_vertex_lbs(&vertex, &matrices);
        assert!((pos[0] - 6.0).abs() < 0.01, "Position should be translated: {:?}", pos);
    }

    #[test]
    fn test_dqs_single_bone() {
        let vertex = SkinnedVertex::single_bone([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0);
        let dq = DualQuaternion::from_rotation_translation(
            [0.0, 0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
        );
        let (pos, _nor) = skin_vertex_dqs(&vertex, &[dq]);
        assert!((pos[0] - 6.0).abs() < 0.01, "DQS position should be translated: {:?}", pos);
    }

    #[test]
    fn test_dual_quaternion_roundtrip() {
        let rotation = [0.0, 0.0, 0.0, 1.0]; // Identity rotation.
        let translation = [3.0, 4.0, 5.0];
        let dq = DualQuaternion::from_rotation_translation(rotation, translation);
        let t = dq.translation();
        assert!((t[0] - 3.0).abs() < 0.01);
        assert!((t[1] - 4.0).abs() < 0.01);
        assert!((t[2] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_morph_target_application() {
        let base = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut target = MorphTarget::new("smile", 2);
        target.set_delta(0, [0.0, 0.1, 0.0]);
        target.set_delta(1, [0.0, 0.2, 0.0]);
        target.weight = 1.0;

        let result = apply_morph_targets(&base, &[target]);
        assert!((result[0][1] - 0.1).abs() < 0.01);
        assert!((result[1][1] - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_morph_target_half_weight() {
        let base = vec![[0.0, 0.0, 0.0]];
        let mut target = MorphTarget::new("test", 1);
        target.set_delta(0, [0.0, 1.0, 0.0]);
        target.weight = 0.5;

        let result = apply_morph_targets(&base, &[target]);
        assert!((result[0][1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_skeleton_pose_update() {
        let mut skel = Skeleton::new("test");
        skel.add_bone(Bone::new("root", 0, u32::MAX));
        skel.add_bone(Bone::new("child", 1, 0));

        let mut pose = SkeletonPose::new(2);
        pose.local_transforms[0] = BoneTransform::from_pos_rot([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        pose.local_transforms[1] = BoneTransform::from_pos_rot([0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0]);

        pose.update(&skel, false);

        // Child should be at (0, 5, 0) in world space.
        let child_pos = [
            pose.world_matrices[1][12],
            pose.world_matrices[1][13],
            pose.world_matrices[1][14],
        ];
        assert!((child_pos[1] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_gpu_skinning_manager() {
        let mut mgr = GpuSkinningManager::new(SkinningMethod::LinearBlend);
        let mut skel = Skeleton::new("test");
        skel.add_bone(Bone::new("root", 0, u32::MAX));
        let idx = mgr.add_skeleton(skel);

        mgr.set_bone_transform(idx, 0, BoneTransform::from_pos_rot([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]));
        mgr.update_all();

        let data = mgr.skinning_data(idx);
        assert!(data.is_some());
        assert_eq!(data.unwrap().len(), 16);
    }

    #[test]
    fn test_skinned_vertex_weight_normalise() {
        let mut v = SkinnedVertex::single_bone([0.0; 3], [0.0, 1.0, 0.0], 0);
        v.bone_weights = [0.5, 0.3, 0.1, 0.05];
        v.normalise_weights();
        let sum: f32 = v.bone_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
