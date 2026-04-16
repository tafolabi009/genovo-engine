//! Animation retargeting -- transferring animations between skeletons with
//! different proportions and bone structures.
//!
//! # Overview
//!
//! Retargeting allows an animation authored for one skeleton (the *source*) to
//! be played on a different skeleton (the *target*). This is essential for
//! sharing motion-capture data or a shared animation library across characters
//! with different body proportions, bone counts, or naming conventions.
//!
//! # Pipeline
//!
//! 1. Build a [`RetargetMap`] that maps source bone indices to target bone
//!    indices using name matching, hierarchy matching, or manual pairs.
//! 2. Optionally detect a [`SkeletonProfile`] for both skeletons so that
//!    standardised bone names can be used for matching.
//! 3. Call [`retarget_pose`] per frame or [`retarget_clip`] for offline
//!    conversion.

use std::collections::HashMap;

use genovo_core::Transform;
use glam::{Quat, Vec3};

use crate::skeleton::{AnimationClip, BoneTrack, Keyframe, Skeleton};

// ---------------------------------------------------------------------------
// Retarget map
// ---------------------------------------------------------------------------

/// Describes how source-skeleton bone indices map to target-skeleton bone
/// indices, together with per-bone scale adjustments derived from the ratio
/// of bone lengths between the two skeletons.
#[derive(Debug, Clone)]
pub struct RetargetMap {
    /// Pairs of `(source_bone_index, target_bone_index)`.
    pub bone_pairs: Vec<BoneMapping>,

    /// Quick lookup: target bone index -> source bone index.
    pub target_to_source: HashMap<usize, usize>,

    /// Quick lookup: source bone index -> target bone index.
    pub source_to_target: HashMap<usize, usize>,

    /// Per-mapping position scale factor derived from bone-length ratios.
    /// Indexed by the position in `bone_pairs`.
    pub position_scales: Vec<f32>,

    /// Name of the source skeleton (informational).
    pub source_name: String,

    /// Name of the target skeleton (informational).
    pub target_name: String,
}

/// A single bone-to-bone mapping entry.
#[derive(Debug, Clone)]
pub struct BoneMapping {
    /// Index in the source skeleton.
    pub source_index: usize,
    /// Index in the target skeleton.
    pub target_index: usize,
    /// How this mapping was established.
    pub method: MappingMethod,
    /// Confidence in [0, 1] for heuristic matches.
    pub confidence: f32,
}

/// How a bone mapping was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingMethod {
    /// Bones share the exact same name.
    ExactName,
    /// Bones share a similar name (fuzzy / normalized match).
    SimilarName,
    /// Bones occupy the same position in the hierarchy tree.
    HierarchyMatch,
    /// Mapping was specified explicitly by the user.
    Manual,
    /// Mapping was derived from a skeleton profile.
    Profile,
}

impl RetargetMap {
    // ----- constructors -----

    /// Create an empty retarget map between two named skeletons.
    pub fn new(source_name: impl Into<String>, target_name: impl Into<String>) -> Self {
        Self {
            bone_pairs: Vec::new(),
            target_to_source: HashMap::new(),
            source_to_target: HashMap::new(),
            position_scales: Vec::new(),
            source_name: source_name.into(),
            target_name: target_name.into(),
        }
    }

    /// Build a retarget map by matching bones with the **exact same name**.
    pub fn by_name_matching(source: &Skeleton, target: &Skeleton) -> Self {
        let mut map = Self::new(&source.name, &target.name);
        for (src_idx, src_bone) in source.bones.iter().enumerate() {
            if let Some(&tgt_idx) = target.bone_names.get(&src_bone.name) {
                map.add_pair(src_idx, tgt_idx, MappingMethod::ExactName, 1.0);
            }
        }
        map.compute_position_scales(source, target);
        map
    }

    /// Build a retarget map using **fuzzy / normalized name matching**.
    ///
    /// Bone names are lowercased, common prefixes/suffixes like `"mixamorig:"`,
    /// `"Bip01_"` etc. are stripped, and the cleaned names are compared.
    pub fn by_similar_name_matching(source: &Skeleton, target: &Skeleton) -> Self {
        let mut map = Self::new(&source.name, &target.name);

        let normalize = |name: &str| -> String {
            let mut s = name.to_lowercase();
            // Strip common prefixes from popular DCC tools and mo-cap formats.
            for prefix in &[
                "mixamorig:", "bip01_", "bip01 ", "bip_", "bone_", "def_",
                "jnt_", "sk_", "rig_",
            ] {
                if let Some(rest) = s.strip_prefix(prefix) {
                    s = rest.to_string();
                }
            }
            // Remove underscores and spaces for lenient comparison.
            s.retain(|c| c != '_' && c != ' ' && c != '-');
            s
        };

        // Build a lookup from normalized target name to index.
        let target_lookup: HashMap<String, usize> = target
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (normalize(&b.name), i))
            .collect();

        for (src_idx, src_bone) in source.bones.iter().enumerate() {
            let src_norm = normalize(&src_bone.name);
            // Exact name match after normalization.
            if let Some(&tgt_idx) = target_lookup.get(&src_norm) {
                if !map.target_to_source.contains_key(&tgt_idx) {
                    let method = if src_bone.name == target.bones[tgt_idx].name {
                        MappingMethod::ExactName
                    } else {
                        MappingMethod::SimilarName
                    };
                    map.add_pair(src_idx, tgt_idx, method, 0.9);
                }
            }
        }

        // Second pass: substring matching for remaining unmapped bones.
        for (src_idx, src_bone) in source.bones.iter().enumerate() {
            if map.source_to_target.contains_key(&src_idx) {
                continue;
            }
            let src_norm = normalize(&src_bone.name);
            if src_norm.len() < 3 {
                continue; // Too short for reliable substring matching.
            }
            let mut best: Option<(usize, f32)> = None;
            for (tgt_idx, tgt_bone) in target.bones.iter().enumerate() {
                if map.target_to_source.contains_key(&tgt_idx) {
                    continue;
                }
                let tgt_norm = normalize(&tgt_bone.name);
                let score = Self::name_similarity(&src_norm, &tgt_norm);
                if score > 0.6 {
                    if best.map_or(true, |(_, bs)| score > bs) {
                        best = Some((tgt_idx, score));
                    }
                }
            }
            if let Some((tgt_idx, score)) = best {
                map.add_pair(src_idx, tgt_idx, MappingMethod::SimilarName, score);
            }
        }

        map.compute_position_scales(source, target);
        map
    }

    /// Build a retarget map by **hierarchy structure matching**.
    ///
    /// Walks both skeleton trees depth-first and matches bones that occupy the
    /// same position in the hierarchy regardless of name.
    pub fn by_hierarchy_matching(source: &Skeleton, target: &Skeleton) -> Self {
        let mut map = Self::new(&source.name, &target.name);

        // Walk both trees in lock-step depth-first order.
        Self::match_subtrees(source, target, source.root_bone_index, target.root_bone_index, &mut map);

        map.compute_position_scales(source, target);
        map
    }

    /// Build a retarget map from **explicit manual pairs**.
    ///
    /// Each tuple is `(source_bone_index, target_bone_index)`.
    pub fn from_manual_pairs(
        source: &Skeleton,
        target: &Skeleton,
        pairs: &[(usize, usize)],
    ) -> Self {
        let mut map = Self::new(&source.name, &target.name);
        for &(src, tgt) in pairs {
            if src < source.bone_count() && tgt < target.bone_count() {
                map.add_pair(src, tgt, MappingMethod::Manual, 1.0);
            }
        }
        map.compute_position_scales(source, target);
        map
    }

    /// Build a retarget map by matching bones using **name pairs**.
    ///
    /// Each tuple is `(source_bone_name, target_bone_name)`.
    pub fn from_name_pairs(
        source: &Skeleton,
        target: &Skeleton,
        pairs: &[(&str, &str)],
    ) -> Self {
        let mut map = Self::new(&source.name, &target.name);
        for &(src_name, tgt_name) in pairs {
            if let (Some(src_idx), Some(tgt_idx)) =
                (source.find_bone(src_name), target.find_bone(tgt_name))
            {
                map.add_pair(src_idx, tgt_idx, MappingMethod::Manual, 1.0);
            }
        }
        map.compute_position_scales(source, target);
        map
    }

    // ----- helpers -----

    /// Add a mapping pair and update the lookup tables.
    pub fn add_pair(
        &mut self,
        source_index: usize,
        target_index: usize,
        method: MappingMethod,
        confidence: f32,
    ) {
        self.bone_pairs.push(BoneMapping {
            source_index,
            target_index,
            method,
            confidence,
        });
        self.source_to_target.insert(source_index, target_index);
        self.target_to_source.insert(target_index, source_index);
    }

    /// Number of mapped bone pairs.
    pub fn pair_count(&self) -> usize {
        self.bone_pairs.len()
    }

    /// Look up the target bone for a given source bone index.
    pub fn target_for_source(&self, source_index: usize) -> Option<usize> {
        self.source_to_target.get(&source_index).copied()
    }

    /// Look up the source bone for a given target bone index.
    pub fn source_for_target(&self, target_index: usize) -> Option<usize> {
        self.target_to_source.get(&target_index).copied()
    }

    /// Get the position scale factor for a mapping pair by pair index.
    pub fn position_scale(&self, pair_index: usize) -> f32 {
        self.position_scales
            .get(pair_index)
            .copied()
            .unwrap_or(1.0)
    }

    /// Compute position scale factors based on bone-length ratios between the
    /// two skeletons. This allows translation values to be scaled so that a
    /// "step" on a tall skeleton produces a proportionally correct step on a
    /// short skeleton.
    pub fn compute_position_scales(
        &mut self,
        source: &Skeleton,
        target: &Skeleton,
    ) {
        let source_lengths = compute_bone_lengths(source);
        let target_lengths = compute_bone_lengths(target);

        self.position_scales.clear();
        for pair in &self.bone_pairs {
            let src_len = source_lengths
                .get(pair.source_index)
                .copied()
                .unwrap_or(1.0);
            let tgt_len = target_lengths
                .get(pair.target_index)
                .copied()
                .unwrap_or(1.0);

            let scale = if src_len.abs() > f32::EPSILON {
                tgt_len / src_len
            } else {
                1.0
            };
            self.position_scales.push(scale);
        }
    }

    /// Return a summary of the mapping as a list of `(source_name, target_name)`.
    pub fn summary(&self, source: &Skeleton, target: &Skeleton) -> Vec<(String, String)> {
        self.bone_pairs
            .iter()
            .map(|p| {
                let sn = source
                    .bones
                    .get(p.source_index)
                    .map(|b| b.name.clone())
                    .unwrap_or_else(|| format!("#{}", p.source_index));
                let tn = target
                    .bones
                    .get(p.target_index)
                    .map(|b| b.name.clone())
                    .unwrap_or_else(|| format!("#{}", p.target_index));
                (sn, tn)
            })
            .collect()
    }

    // ----- private helpers -----

    /// Simple Levenshtein-distance based similarity in [0, 1].
    fn name_similarity(a: &str, b: &str) -> f32 {
        if a == b {
            return 1.0;
        }
        let max_len = a.len().max(b.len());
        if max_len == 0 {
            return 1.0;
        }
        let dist = Self::levenshtein(a, b);
        1.0 - (dist as f32 / max_len as f32)
    }

    /// Levenshtein edit distance.
    fn levenshtein(a: &str, b: &str) -> usize {
        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();
        let m = a_bytes.len();
        let n = b_bytes.len();

        let mut prev = (0..=n).collect::<Vec<_>>();
        let mut curr = vec![0usize; n + 1];

        for i in 1..=m {
            curr[0] = i;
            for j in 1..=n {
                let cost = if a_bytes[i - 1] == b_bytes[j - 1] {
                    0
                } else {
                    1
                };
                curr[j] = (prev[j] + 1)
                    .min(curr[j - 1] + 1)
                    .min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        prev[n]
    }

    /// Recursively match subtrees depth-first.
    fn match_subtrees(
        source: &Skeleton,
        target: &Skeleton,
        src_root: usize,
        tgt_root: usize,
        map: &mut RetargetMap,
    ) {
        // Map the current roots.
        if !map.source_to_target.contains_key(&src_root)
            && !map.target_to_source.contains_key(&tgt_root)
        {
            map.add_pair(src_root, tgt_root, MappingMethod::HierarchyMatch, 0.7);
        }

        let src_children = source.children_of(src_root);
        let tgt_children = target.children_of(tgt_root);

        // Match children pairwise by order.
        let count = src_children.len().min(tgt_children.len());
        for i in 0..count {
            Self::match_subtrees(source, target, src_children[i], tgt_children[i], map);
        }
    }
}

// ---------------------------------------------------------------------------
// Bone-length computation
// ---------------------------------------------------------------------------

/// Compute the length (distance from parent) for each bone in a skeleton.
/// Root bones get a length of 0.
fn compute_bone_lengths(skeleton: &Skeleton) -> Vec<f32> {
    let bind_poses = skeleton.bind_pose();
    let world = skeleton.compute_world_transforms(&bind_poses);

    let mut lengths = vec![0.0f32; skeleton.bone_count()];
    for (i, bone) in skeleton.bones.iter().enumerate() {
        if let Some(parent) = bone.parent_index {
            let parent_pos = Vec3::new(
                world[parent].col(3).x,
                world[parent].col(3).y,
                world[parent].col(3).z,
            );
            let bone_pos = Vec3::new(
                world[i].col(3).x,
                world[i].col(3).y,
                world[i].col(3).z,
            );
            lengths[i] = (bone_pos - parent_pos).length();
        }
    }
    lengths
}

/// Compute the total chain length from bone `start` to bone `end` (walking
/// from end toward start via parent links).
fn chain_length(skeleton: &Skeleton, start: usize, end: usize) -> f32 {
    let lengths = compute_bone_lengths(skeleton);
    let mut total = 0.0;
    let mut current = end;
    while current != start {
        total += lengths[current];
        match skeleton.bones[current].parent_index {
            Some(p) => current = p,
            None => break,
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Retargeting functions
// ---------------------------------------------------------------------------

/// Retarget a single pose from the source skeleton to the target skeleton.
///
/// # Algorithm
///
/// For each mapped bone pair:
/// - **Rotation**: Copied directly because rotations are stored relative to
///   the parent bone and thus remain valid across different proportions.
/// - **Position (translation)**: Scaled by the bone-length ratio between
///   source and target so that movements look proportionally correct.
/// - **Scale**: Copied directly.
///
/// Unmapped target bones receive the target skeleton's bind pose. Bones
/// that are missing in the source but whose parent *is* mapped will have
/// their rotation interpolated from the parent mapping.
pub fn retarget_pose(
    source_pose: &[Transform],
    source_skeleton: &Skeleton,
    target_skeleton: &Skeleton,
    map: &RetargetMap,
) -> Vec<Transform> {
    let target_bind = target_skeleton.bind_pose();
    let source_bind = source_skeleton.bind_pose();
    let mut target_pose = target_bind.clone();

    // Apply mapped bones.
    for (pair_idx, pair) in map.bone_pairs.iter().enumerate() {
        let src_idx = pair.source_index;
        let tgt_idx = pair.target_index;

        if src_idx >= source_pose.len() || tgt_idx >= target_pose.len() {
            continue;
        }

        let src_transform = &source_pose[src_idx];
        let src_bind = &source_bind[src_idx];
        let tgt_bind = &target_bind[tgt_idx];

        // -- Rotation: copy the delta rotation from source --
        // delta = src_bind.rotation.inverse() * src_transform.rotation
        // result = tgt_bind.rotation * delta
        let delta_rotation = src_bind.rotation.conjugate() * src_transform.rotation;
        target_pose[tgt_idx].rotation = tgt_bind.rotation * delta_rotation;

        // -- Position: apply scaled delta --
        let scale = map.position_scales.get(pair_idx).copied().unwrap_or(1.0);
        let delta_position = src_transform.position - src_bind.position;
        target_pose[tgt_idx].position = tgt_bind.position + delta_position * scale;

        // -- Scale: copy directly --
        target_pose[tgt_idx].scale = src_transform.scale;
    }

    // Fill in unmapped target bones by interpolating from their mapped parent.
    fill_unmapped_bones(target_skeleton, map, &target_bind, &mut target_pose);

    target_pose
}

/// Retarget an entire animation clip from one skeleton to another.
///
/// Produces a new `AnimationClip` with bone tracks remapped according to the
/// [`RetargetMap`]. The clip duration and looping settings are preserved.
pub fn retarget_clip(
    source_clip: &AnimationClip,
    source_skeleton: &Skeleton,
    target_skeleton: &Skeleton,
    map: &RetargetMap,
) -> AnimationClip {
    let mut target_clip = AnimationClip::new(
        format!("{}_retargeted", source_clip.name),
        source_clip.duration,
    );
    target_clip.looping = source_clip.looping;
    target_clip.sample_rate = source_clip.sample_rate;

    let source_bind = source_skeleton.bind_pose();
    let target_bind = target_skeleton.bind_pose();

    for (pair_idx, pair) in map.bone_pairs.iter().enumerate() {
        let src_track = match source_clip.track_for_bone(pair.source_index) {
            Some(t) => t,
            None => continue,
        };

        let scale = map.position_scales.get(pair_idx).copied().unwrap_or(1.0);

        let src_bind_pos = source_bind
            .get(pair.source_index)
            .map(|t| t.position)
            .unwrap_or(Vec3::ZERO);
        let src_bind_rot = source_bind
            .get(pair.source_index)
            .map(|t| t.rotation)
            .unwrap_or(Quat::IDENTITY);
        let tgt_bind_pos = target_bind
            .get(pair.target_index)
            .map(|t| t.position)
            .unwrap_or(Vec3::ZERO);
        let tgt_bind_rot = target_bind
            .get(pair.target_index)
            .map(|t| t.rotation)
            .unwrap_or(Quat::IDENTITY);

        let mut tgt_track = BoneTrack::new(pair.target_index);
        tgt_track.position_interpolation = src_track.position_interpolation;
        tgt_track.rotation_interpolation = src_track.rotation_interpolation;
        tgt_track.scale_interpolation = src_track.scale_interpolation;

        // Retarget position keyframes.
        for key in &src_track.position_keys {
            let delta = key.value - src_bind_pos;
            let retargeted = tgt_bind_pos + delta * scale;
            tgt_track.position_keys.push(Keyframe::new(key.time, retargeted));
        }

        // Retarget rotation keyframes.
        for key in &src_track.rotation_keys {
            let delta = src_bind_rot.conjugate() * key.value;
            let retargeted = tgt_bind_rot * delta;
            tgt_track.rotation_keys.push(Keyframe::new(key.time, retargeted));
        }

        // Scale keyframes: copy directly.
        for key in &src_track.scale_keys {
            tgt_track.scale_keys.push(Keyframe::new(key.time, key.value));
        }

        target_clip.add_track(tgt_track);
    }

    target_clip
}

/// Retarget a clip by sampling at a fixed rate and then producing a new clip
/// with uniform keyframes. This is useful when the source clip has complex
/// interpolation or when bone mappings require per-frame fixup.
pub fn retarget_clip_sampled(
    source_clip: &AnimationClip,
    source_skeleton: &Skeleton,
    target_skeleton: &Skeleton,
    map: &RetargetMap,
    sample_rate: f32,
) -> AnimationClip {
    let duration = source_clip.duration;
    let sample_count = (duration * sample_rate).ceil() as usize + 1;
    let bone_count_src = source_skeleton.bone_count();
    let bone_count_tgt = target_skeleton.bone_count();

    let mut target_clip = AnimationClip::new(
        format!("{}_retargeted", source_clip.name),
        duration,
    );
    target_clip.looping = source_clip.looping;
    target_clip.sample_rate = sample_rate;

    // Pre-allocate tracks for all target bones.
    let mut tgt_tracks: Vec<BoneTrack> = (0..bone_count_tgt)
        .map(BoneTrack::new)
        .collect();

    // Sample each frame.
    for s in 0..sample_count {
        let t = if sample_count > 1 {
            (s as f32 / (sample_count - 1) as f32) * duration
        } else {
            0.0
        };

        let src_pose = source_clip.sample_pose(t, bone_count_src);
        let tgt_pose = retarget_pose(&src_pose, source_skeleton, target_skeleton, map);

        for (bone_idx, transform) in tgt_pose.iter().enumerate() {
            if bone_idx < tgt_tracks.len() {
                tgt_tracks[bone_idx].position_keys.push(Keyframe::new(t, transform.position));
                tgt_tracks[bone_idx].rotation_keys.push(Keyframe::new(t, transform.rotation));
                tgt_tracks[bone_idx].scale_keys.push(Keyframe::new(t, transform.scale));
            }
        }
    }

    for track in tgt_tracks {
        if !track.is_empty() {
            target_clip.add_track(track);
        }
    }

    target_clip
}

/// For unmapped target bones, interpolate their transform from the nearest
/// mapped ancestor in the target hierarchy.
fn fill_unmapped_bones(
    target_skeleton: &Skeleton,
    map: &RetargetMap,
    bind_pose: &[Transform],
    pose: &mut [Transform],
) {
    for tgt_idx in 0..target_skeleton.bone_count() {
        if map.target_to_source.contains_key(&tgt_idx) {
            continue; // Already mapped.
        }

        // Walk up the hierarchy looking for a mapped ancestor.
        let mut parent_opt = target_skeleton.bones[tgt_idx].parent_index;
        let mut mapped_ancestor: Option<usize> = None;
        while let Some(parent) = parent_opt {
            if map.target_to_source.contains_key(&parent) {
                mapped_ancestor = Some(parent);
                break;
            }
            parent_opt = target_skeleton.bones[parent].parent_index;
        }

        if let Some(ancestor_idx) = mapped_ancestor {
            // Inherit the rotation delta from the mapped ancestor so that
            // unmapped child bones follow their parent's retargeted motion.
            let ancestor_bind_rot = bind_pose[ancestor_idx].rotation;
            let ancestor_curr_rot = pose[ancestor_idx].rotation;
            let ancestor_delta = ancestor_bind_rot.conjugate() * ancestor_curr_rot;

            let bone_bind_rot = bind_pose[tgt_idx].rotation;
            pose[tgt_idx].rotation = bone_bind_rot * ancestor_delta;
        }
        // If no mapped ancestor, the bone keeps its bind pose (already set).
    }
}

// ---------------------------------------------------------------------------
// Skeleton profile
// ---------------------------------------------------------------------------

/// Standard bone-naming convention for automatic retarget-map generation.
///
/// A profile defines a set of canonical bone names (e.g. the *Humanoid*
/// profile has "Hips", "Spine", "Head", etc.) and associates each canonical
/// name with the actual bone name/index in a specific skeleton.
#[derive(Debug, Clone)]
pub struct SkeletonProfile {
    /// Profile identifier (e.g. "Humanoid").
    pub name: String,

    /// Canonical bone name to actual bone index in a skeleton.
    pub bone_map: HashMap<String, usize>,

    /// The skeleton this profile was built for.
    pub skeleton_name: String,
}

/// All canonical bone slots in a humanoid profile.
pub const HUMANOID_BONE_NAMES: &[&str] = &[
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Chest",
    "UpperChest",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftLowerArm",
    "LeftHand",
    "RightShoulder",
    "RightUpperArm",
    "RightLowerArm",
    "RightHand",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
    "LeftToes",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
    "RightToes",
    "LeftThumbProximal",
    "LeftThumbIntermediate",
    "LeftThumbDistal",
    "LeftIndexProximal",
    "LeftIndexIntermediate",
    "LeftIndexDistal",
    "LeftMiddleProximal",
    "LeftMiddleIntermediate",
    "LeftMiddleDistal",
    "LeftRingProximal",
    "LeftRingIntermediate",
    "LeftRingDistal",
    "LeftLittleProximal",
    "LeftLittleIntermediate",
    "LeftLittleDistal",
    "RightThumbProximal",
    "RightThumbIntermediate",
    "RightThumbDistal",
    "RightIndexProximal",
    "RightIndexIntermediate",
    "RightIndexDistal",
    "RightMiddleProximal",
    "RightMiddleIntermediate",
    "RightMiddleDistal",
    "RightRingProximal",
    "RightRingIntermediate",
    "RightRingDistal",
    "RightLittleProximal",
    "RightLittleIntermediate",
    "RightLittleDistal",
];

/// Common name aliases that map to canonical humanoid bone names.
/// Each entry is `(alias_pattern, canonical_name)`.
const HUMANOID_ALIASES: &[(&str, &str)] = &[
    ("pelvis", "Hips"),
    ("hip", "Hips"),
    ("root", "Hips"),
    ("spine", "Spine"),
    ("spine1", "Spine1"),
    ("spine2", "Spine2"),
    ("chest", "Chest"),
    ("upperchest", "UpperChest"),
    ("neck", "Neck"),
    ("head", "Head"),
    ("leftshoulder", "LeftShoulder"),
    ("leftupperarm", "LeftUpperArm"),
    ("leftarm", "LeftUpperArm"),
    ("leftforearm", "LeftLowerArm"),
    ("leftlowerarm", "LeftLowerArm"),
    ("lefthand", "LeftHand"),
    ("rightshoulder", "RightShoulder"),
    ("rightupperarm", "RightUpperArm"),
    ("rightarm", "RightUpperArm"),
    ("rightforearm", "RightLowerArm"),
    ("rightlowerarm", "RightLowerArm"),
    ("righthand", "RightHand"),
    ("leftupleg", "LeftUpperLeg"),
    ("leftupperleg", "LeftUpperLeg"),
    ("leftthigh", "LeftUpperLeg"),
    ("leftleg", "LeftLowerLeg"),
    ("leftlowerleg", "LeftLowerLeg"),
    ("leftshin", "LeftLowerLeg"),
    ("leftfoot", "LeftFoot"),
    ("lefttoe", "LeftToes"),
    ("lefttoes", "LeftToes"),
    ("lefttoebase", "LeftToes"),
    ("rightupleg", "RightUpperLeg"),
    ("rightupperleg", "RightUpperLeg"),
    ("rightthigh", "RightUpperLeg"),
    ("rightleg", "RightLowerLeg"),
    ("rightlowerleg", "RightLowerLeg"),
    ("rightshin", "RightLowerLeg"),
    ("rightfoot", "RightFoot"),
    ("righttoe", "RightToes"),
    ("righttoes", "RightToes"),
    ("righttoebase", "RightToes"),
];

impl SkeletonProfile {
    /// Create an empty profile.
    pub fn new(name: impl Into<String>, skeleton_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bone_map: HashMap::new(),
            skeleton_name: skeleton_name.into(),
        }
    }

    /// Create a humanoid profile by auto-detecting bone names in the skeleton.
    ///
    /// Uses normalized name matching against the canonical humanoid bone names
    /// and common aliases (Mixamo, UE, etc.).
    pub fn detect_humanoid(skeleton: &Skeleton) -> Self {
        let mut profile = Self::new("Humanoid", &skeleton.name);

        let normalize = |name: &str| -> String {
            let mut s = name.to_lowercase();
            for prefix in &[
                "mixamorig:", "bip01_", "bip01 ", "bip_", "bone_", "def_",
                "jnt_", "sk_", "rig_",
            ] {
                if let Some(rest) = s.strip_prefix(prefix) {
                    s = rest.to_string();
                }
            }
            s.retain(|c| c != '_' && c != ' ' && c != '-');
            s
        };

        // Build a lookup from normalized bone name to bone index.
        let bone_lookup: Vec<(String, usize)> = skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (normalize(&b.name), i))
            .collect();

        // First pass: exact canonical name match.
        for &canonical in HUMANOID_BONE_NAMES {
            let canonical_norm = normalize(canonical);
            for &(ref norm, idx) in &bone_lookup {
                if *norm == canonical_norm {
                    profile.bone_map.insert(canonical.to_string(), idx);
                    break;
                }
            }
        }

        // Second pass: alias matching for unmapped slots.
        for &(alias, canonical) in HUMANOID_ALIASES {
            if profile.bone_map.contains_key(canonical) {
                continue;
            }
            for &(ref norm, idx) in &bone_lookup {
                if *norm == alias {
                    profile.bone_map.insert(canonical.to_string(), idx);
                    break;
                }
            }
        }

        profile
    }

    /// Create a retarget map from two skeleton profiles by matching canonical
    /// bone names.
    pub fn create_retarget_map(
        source_profile: &SkeletonProfile,
        target_profile: &SkeletonProfile,
        source_skeleton: &Skeleton,
        target_skeleton: &Skeleton,
    ) -> RetargetMap {
        let mut map = RetargetMap::new(&source_skeleton.name, &target_skeleton.name);

        for (canonical, &src_idx) in &source_profile.bone_map {
            if let Some(&tgt_idx) = target_profile.bone_map.get(canonical) {
                map.add_pair(src_idx, tgt_idx, MappingMethod::Profile, 1.0);
            }
        }

        map.compute_position_scales(source_skeleton, target_skeleton);
        map
    }

    /// Check if a canonical bone slot is mapped.
    pub fn has_bone(&self, canonical_name: &str) -> bool {
        self.bone_map.contains_key(canonical_name)
    }

    /// Get the skeleton bone index for a canonical name.
    pub fn bone_index(&self, canonical_name: &str) -> Option<usize> {
        self.bone_map.get(canonical_name).copied()
    }

    /// Number of mapped canonical bones.
    pub fn mapped_count(&self) -> usize {
        self.bone_map.len()
    }

    /// Return a list of canonical names that are NOT yet mapped.
    pub fn unmapped_slots(&self) -> Vec<&'static str> {
        HUMANOID_BONE_NAMES
            .iter()
            .filter(|&&name| !self.bone_map.contains_key(name))
            .copied()
            .collect()
    }

    /// Validate that required humanoid bones are present.
    ///
    /// Returns a list of missing required bone names. For a minimal humanoid,
    /// we require: Hips, Spine, Head, and at least the upper arm + leg bones.
    pub fn validate_humanoid(&self) -> Vec<&'static str> {
        let required = &[
            "Hips", "Spine", "Head",
            "LeftUpperArm", "RightUpperArm",
            "LeftUpperLeg", "RightUpperLeg",
        ];
        required
            .iter()
            .filter(|&&name| !self.bone_map.contains_key(name))
            .copied()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Pose difference utilities
// ---------------------------------------------------------------------------

/// Compute the per-bone difference between two poses of the same skeleton.
///
/// For each bone: `delta.position = b.position - a.position`,
/// `delta.rotation = a.rotation.inverse() * b.rotation`,
/// `delta.scale = b.scale / a.scale`.
pub fn pose_difference(a: &[Transform], b: &[Transform]) -> Vec<Transform> {
    assert_eq!(a.len(), b.len(), "Pose lengths must match");
    a.iter()
        .zip(b.iter())
        .map(|(ta, tb)| {
            let dp = tb.position - ta.position;
            let dr = ta.rotation.conjugate() * tb.rotation;
            let ds = Vec3::new(
                if ta.scale.x.abs() > f32::EPSILON {
                    tb.scale.x / ta.scale.x
                } else {
                    1.0
                },
                if ta.scale.y.abs() > f32::EPSILON {
                    tb.scale.y / ta.scale.y
                } else {
                    1.0
                },
                if ta.scale.z.abs() > f32::EPSILON {
                    tb.scale.z / ta.scale.z
                } else {
                    1.0
                },
            );
            Transform::new(dp, dr, ds)
        })
        .collect()
}

/// Apply a pose difference (delta) to a base pose.
///
/// For each bone: `result.position = base.position + delta.position`,
/// `result.rotation = base.rotation * delta.rotation`,
/// `result.scale = base.scale * delta.scale`.
pub fn apply_pose_difference(
    base: &[Transform],
    delta: &[Transform],
    weight: f32,
) -> Vec<Transform> {
    assert_eq!(base.len(), delta.len(), "Pose lengths must match");
    let w = weight.clamp(0.0, 1.0);
    base.iter()
        .zip(delta.iter())
        .map(|(b, d)| {
            Transform::new(
                b.position + d.position * w,
                b.rotation * Quat::IDENTITY.slerp(d.rotation, w),
                b.scale * Vec3::ONE.lerp(d.scale, w),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{AnimationClip, Bone, BoneTrack, Keyframe, Skeleton};
    use glam::{Mat4, Quat, Vec3};

    fn make_skeleton_a() -> Skeleton {
        Skeleton::new(
            "SkeletonA",
            vec![
                Bone::new("Hips", None, Transform::IDENTITY, Mat4::IDENTITY),
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
                Bone::new(
                    "LeftUpperArm",
                    Some(1),
                    Transform::from_position(Vec3::new(-0.3, 0.4, 0.0)),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "RightUpperArm",
                    Some(1),
                    Transform::from_position(Vec3::new(0.3, 0.4, 0.0)),
                    Mat4::IDENTITY,
                ),
            ],
        )
    }

    fn make_skeleton_b() -> Skeleton {
        // Same bone names but different proportions (taller character).
        Skeleton::new(
            "SkeletonB",
            vec![
                Bone::new("Hips", None, Transform::IDENTITY, Mat4::IDENTITY),
                Bone::new(
                    "Spine",
                    Some(0),
                    Transform::from_position(Vec3::new(0.0, 1.5, 0.0)),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "Head",
                    Some(1),
                    Transform::from_position(Vec3::new(0.0, 0.75, 0.0)),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "LeftUpperArm",
                    Some(1),
                    Transform::from_position(Vec3::new(-0.45, 0.6, 0.0)),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "RightUpperArm",
                    Some(1),
                    Transform::from_position(Vec3::new(0.45, 0.6, 0.0)),
                    Mat4::IDENTITY,
                ),
            ],
        )
    }

    fn make_skeleton_different_names() -> Skeleton {
        Skeleton::new(
            "MixamoSkeleton",
            vec![
                Bone::new("mixamorig:Hips", None, Transform::IDENTITY, Mat4::IDENTITY),
                Bone::new(
                    "mixamorig:Spine",
                    Some(0),
                    Transform::from_position(Vec3::new(0.0, 1.2, 0.0)),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "mixamorig:Head",
                    Some(1),
                    Transform::from_position(Vec3::new(0.0, 0.6, 0.0)),
                    Mat4::IDENTITY,
                ),
            ],
        )
    }

    #[test]
    fn test_exact_name_matching() {
        let a = make_skeleton_a();
        let b = make_skeleton_b();
        let map = RetargetMap::by_name_matching(&a, &b);
        assert_eq!(map.pair_count(), 5);
        assert_eq!(map.target_for_source(0), Some(0));
        assert_eq!(map.target_for_source(1), Some(1));
        assert_eq!(map.target_for_source(2), Some(2));
    }

    #[test]
    fn test_similar_name_matching() {
        let a = make_skeleton_a();
        let b = make_skeleton_different_names();
        let map = RetargetMap::by_similar_name_matching(&a, &b);
        // Should match Hips->mixamorig:Hips, Spine->mixamorig:Spine, Head->mixamorig:Head.
        assert!(map.pair_count() >= 3);
    }

    #[test]
    fn test_hierarchy_matching() {
        let a = make_skeleton_a();
        let b = make_skeleton_b();
        let map = RetargetMap::by_hierarchy_matching(&a, &b);
        assert!(map.pair_count() >= 3);
        // Root should map to root.
        assert_eq!(map.target_for_source(a.root_bone_index), Some(b.root_bone_index));
    }

    #[test]
    fn test_manual_pairs() {
        let a = make_skeleton_a();
        let b = make_skeleton_b();
        let map = RetargetMap::from_manual_pairs(&a, &b, &[(0, 0), (1, 1), (2, 2)]);
        assert_eq!(map.pair_count(), 3);
    }

    #[test]
    fn test_name_pairs() {
        let a = make_skeleton_a();
        let b = make_skeleton_b();
        let map = RetargetMap::from_name_pairs(&a, &b, &[("Hips", "Hips"), ("Spine", "Spine")]);
        assert_eq!(map.pair_count(), 2);
    }

    #[test]
    fn test_position_scales() {
        let a = make_skeleton_a();
        let b = make_skeleton_b();
        let map = RetargetMap::by_name_matching(&a, &b);
        // Skeleton B is 1.5x taller, so scale for Spine should be ~1.5.
        let spine_pair = map
            .bone_pairs
            .iter()
            .position(|p| p.source_index == 1 && p.target_index == 1)
            .unwrap();
        let scale = map.position_scale(spine_pair);
        assert!(
            (scale - 1.5).abs() < 0.01,
            "Expected ~1.5 scale for spine, got {}",
            scale
        );
    }

    #[test]
    fn test_retarget_pose_identity() {
        let a = make_skeleton_a();
        let b = make_skeleton_b();
        let map = RetargetMap::by_name_matching(&a, &b);

        // Source pose = source bind pose -> target should get target bind pose.
        let source_pose = a.bind_pose();
        let result = retarget_pose(&source_pose, &a, &b, &map);
        let target_bind = b.bind_pose();
        assert_eq!(result.len(), target_bind.len());
        for (i, (r, t)) in result.iter().zip(target_bind.iter()).enumerate() {
            assert!(
                (r.position - t.position).length() < 0.01,
                "Bone {} position mismatch: {:?} vs {:?}",
                i, r.position, t.position
            );
        }
    }

    #[test]
    fn test_retarget_pose_with_motion() {
        let a = make_skeleton_a();
        let b = make_skeleton_b();
        let map = RetargetMap::by_name_matching(&a, &b);

        let mut source_pose = a.bind_pose();
        // Add some motion to the spine.
        source_pose[1].position += Vec3::new(0.0, 0.1, 0.0);
        source_pose[1].rotation = Quat::from_rotation_z(0.1);

        let result = retarget_pose(&source_pose, &a, &b, &map);
        let target_bind = b.bind_pose();

        // The target spine should have moved, but scaled proportionally.
        assert!(
            (result[1].position.y - target_bind[1].position.y).abs() > 0.01,
            "Spine should have moved"
        );
        // Rotation should have been transferred.
        let dot = result[1].rotation.dot(target_bind[1].rotation).abs();
        assert!(dot < 0.999, "Spine rotation should differ from bind pose");
    }

    #[test]
    fn test_retarget_clip() {
        let a = make_skeleton_a();
        let b = make_skeleton_b();
        let map = RetargetMap::by_name_matching(&a, &b);

        let mut clip = AnimationClip::new("Walk", 1.0);
        clip.looping = true;
        for i in 0..a.bone_count() {
            let mut track = BoneTrack::new(i);
            track.position_keys = vec![
                Keyframe::new(0.0, Vec3::new(0.0, i as f32 * 0.5, 0.0)),
                Keyframe::new(0.5, Vec3::new(0.0, i as f32 * 0.5 + 0.1, 0.0)),
                Keyframe::new(1.0, Vec3::new(0.0, i as f32 * 0.5, 0.0)),
            ];
            track.rotation_keys = vec![
                Keyframe::new(0.0, Quat::IDENTITY),
                Keyframe::new(1.0, Quat::IDENTITY),
            ];
            track.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
            clip.add_track(track);
        }

        let retargeted = retarget_clip(&clip, &a, &b, &map);
        assert_eq!(retargeted.bone_tracks.len(), a.bone_count());
        assert!(retargeted.looping);
        assert!((retargeted.duration - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_retarget_clip_sampled() {
        let a = make_skeleton_a();
        let b = make_skeleton_b();
        let map = RetargetMap::by_name_matching(&a, &b);

        let mut clip = AnimationClip::new("Idle", 1.0);
        clip.looping = true;
        for i in 0..a.bone_count() {
            let mut track = BoneTrack::new(i);
            track.position_keys = vec![Keyframe::new(0.0, Vec3::ZERO)];
            track.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
            track.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
            clip.add_track(track);
        }

        let retargeted = retarget_clip_sampled(&clip, &a, &b, &map, 30.0);
        assert!(!retargeted.bone_tracks.is_empty());
        // At 30 fps for 1 second, we should have ~31 keyframes per track.
        let first_track = &retargeted.bone_tracks[0];
        assert!(first_track.position_keys.len() >= 30);
    }

    #[test]
    fn test_skeleton_profile_detect() {
        let skel = make_skeleton_a();
        let profile = SkeletonProfile::detect_humanoid(&skel);
        assert!(profile.has_bone("Hips"));
        assert!(profile.has_bone("Spine"));
        assert!(profile.has_bone("Head"));
        assert!(profile.has_bone("LeftUpperArm"));
        assert!(profile.has_bone("RightUpperArm"));
    }

    #[test]
    fn test_skeleton_profile_detect_mixamo() {
        let skel = make_skeleton_different_names();
        let profile = SkeletonProfile::detect_humanoid(&skel);
        assert!(
            profile.has_bone("Hips"),
            "Should detect Hips from mixamorig:Hips"
        );
        assert!(
            profile.has_bone("Spine"),
            "Should detect Spine from mixamorig:Spine"
        );
        assert!(
            profile.has_bone("Head"),
            "Should detect Head from mixamorig:Head"
        );
    }

    #[test]
    fn test_profile_based_retarget_map() {
        let a = make_skeleton_a();
        let b = make_skeleton_different_names();
        let profile_a = SkeletonProfile::detect_humanoid(&a);
        let profile_b = SkeletonProfile::detect_humanoid(&b);
        let map = SkeletonProfile::create_retarget_map(&profile_a, &profile_b, &a, &b);
        // At least Hips, Spine, Head should be mapped.
        assert!(map.pair_count() >= 3);
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(RetargetMap::levenshtein("kitten", "sitting"), 3);
        assert_eq!(RetargetMap::levenshtein("hello", "hello"), 0);
        assert_eq!(RetargetMap::levenshtein("", "abc"), 3);
    }

    #[test]
    fn test_name_similarity() {
        assert!((RetargetMap::name_similarity("spine", "spine") - 1.0).abs() < f32::EPSILON);
        assert!(RetargetMap::name_similarity("spine", "spine1") > 0.7);
        assert!(RetargetMap::name_similarity("head", "leftfoot") < 0.5);
    }

    #[test]
    fn test_pose_difference_and_apply() {
        let a = vec![
            Transform::from_position(Vec3::new(0.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
        ];
        let b = vec![
            Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(0.0, 2.0, 0.0)),
        ];
        let delta = pose_difference(&a, &b);
        assert!((delta[0].position.x - 1.0).abs() < f32::EPSILON);
        assert!((delta[1].position.y - 1.0).abs() < f32::EPSILON);

        let applied = apply_pose_difference(&a, &delta, 1.0);
        assert!((applied[0].position.x - 1.0).abs() < 0.01);
        assert!((applied[1].position.y - 2.0).abs() < 0.01);

        let half = apply_pose_difference(&a, &delta, 0.5);
        assert!((half[0].position.x - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fill_unmapped_bones() {
        // 4-bone skeleton: Root -> Spine -> Head -> Accessory
        let skel = Skeleton::new(
            "Test",
            vec![
                Bone::new("Root", None, Transform::IDENTITY, Mat4::IDENTITY),
                Bone::new(
                    "Spine",
                    Some(0),
                    Transform::from_position(Vec3::Y),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "Head",
                    Some(1),
                    Transform::from_position(Vec3::new(0.0, 0.5, 0.0)),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "Accessory",
                    Some(2),
                    Transform::from_position(Vec3::new(0.0, 0.1, 0.0)),
                    Mat4::IDENTITY,
                ),
            ],
        );

        // Map only Root, Spine, Head -- leave Accessory unmapped.
        let mut map = RetargetMap::new("src", "tgt");
        map.add_pair(0, 0, MappingMethod::Manual, 1.0);
        map.add_pair(1, 1, MappingMethod::Manual, 1.0);
        map.add_pair(2, 2, MappingMethod::Manual, 1.0);

        let bind = skel.bind_pose();
        let mut pose = bind.clone();
        // Rotate the Head.
        pose[2].rotation = Quat::from_rotation_z(0.5);

        fill_unmapped_bones(&skel, &map, &bind, &mut pose);

        // Accessory (index 3) should inherit Head's rotation delta.
        let acc_rot = pose[3].rotation;
        let head_delta = bind[2].rotation.conjugate() * pose[2].rotation;
        let expected = bind[3].rotation * head_delta;
        assert!(
            acc_rot.dot(expected).abs() > 0.99,
            "Unmapped bone should inherit parent delta"
        );
    }

    #[test]
    fn test_bone_lengths() {
        let skel = make_skeleton_a();
        let lengths = compute_bone_lengths(&skel);
        assert!((lengths[0]).abs() < f32::EPSILON); // Root has length 0.
        assert!((lengths[1] - 1.0).abs() < 0.01); // Spine is 1 unit from root.
        assert!((lengths[2] - 0.5).abs() < 0.01); // Head is 0.5 from spine.
    }

    #[test]
    fn test_chain_length_fn() {
        let skel = make_skeleton_a();
        let len = chain_length(&skel, 0, 2);
        assert!(
            (len - 1.5).abs() < 0.01,
            "Chain from Root to Head should be ~1.5, got {}",
            len
        );
    }

    #[test]
    fn test_profile_validate_humanoid() {
        let skel = make_skeleton_a();
        let profile = SkeletonProfile::detect_humanoid(&skel);
        let missing = profile.validate_humanoid();
        // We have Hips, Spine, Head, LeftUpperArm, RightUpperArm but lack legs.
        assert!(
            missing.contains(&"LeftUpperLeg"),
            "Should report missing LeftUpperLeg"
        );
    }

    #[test]
    fn test_retarget_map_summary() {
        let a = make_skeleton_a();
        let b = make_skeleton_b();
        let map = RetargetMap::by_name_matching(&a, &b);
        let summary = map.summary(&a, &b);
        assert_eq!(summary.len(), map.pair_count());
        // First pair should be ("Hips", "Hips").
        assert_eq!(summary[0].0, "Hips");
        assert_eq!(summary[0].1, "Hips");
    }
}
