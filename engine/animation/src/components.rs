//! ECS components and systems for animation integration.
//!
//! Bridges the animation runtime (skeleton evaluation, IK, blend trees,
//! state machines) with the entity-component-system. Provides the
//! [`AnimatorComponent`] for per-entity animation state and the
//! [`AnimationSystem`] that drives evaluation each frame.
//!
//! # Processing Pipeline
//!
//! Each frame, the [`AnimationSystem`] processes every entity with an
//! `AnimatorComponent` through these stages:
//!
//! 1. Update state machine (if present) to determine active state/blend tree.
//! 2. Evaluate blend tree to get the blended clip pose.
//! 3. Advance the animation player (or use state machine output directly).
//! 4. Apply IK solvers to modify the pose.
//! 5. Walk the bone hierarchy to compute world-space transforms.
//! 6. Multiply by inverse bind matrices to produce skinning matrices.
//! 7. Store results in `SkinnedMeshRenderer::bone_matrices` for GPU upload.

use genovo_core::Transform;
use glam::{Mat4, Quat, Vec3};

use crate::blend_tree::{AnimationStateMachine, BlendParams, BlendTree};
use crate::ik::{IKChain, IKSolver};
use crate::skeleton::{AnimationClip, AnimationPlayer, Skeleton, SkinnedMeshRenderer};

// ---------------------------------------------------------------------------
// BoneTransform
// ---------------------------------------------------------------------------

/// A local-space transform for a single bone.
///
/// This is the output of animation sampling before hierarchy multiplication.
#[derive(Debug, Clone, Copy)]
pub struct BoneTransform {
    /// Position relative to parent bone.
    pub position: Vec3,
    /// Rotation relative to parent bone.
    pub rotation: Quat,
    /// Scale relative to parent bone.
    pub scale: Vec3,
}

impl Default for BoneTransform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl BoneTransform {
    /// Convert to a `Transform`.
    pub fn to_transform(&self) -> Transform {
        Transform::new(self.position, self.rotation, self.scale)
    }

    /// Create from a `Transform`.
    pub fn from_transform(t: &Transform) -> Self {
        Self {
            position: t.position,
            rotation: t.rotation,
            scale: t.scale,
        }
    }
}

// ---------------------------------------------------------------------------
// IKChainBinding
// ---------------------------------------------------------------------------

/// Binds an IK chain to a specific solver for processing by the animation system.
#[derive(Debug)]
pub struct IKChainBinding {
    /// The IK chain definition.
    pub chain: IKChain,
    /// Name of the solver to use (matched against registered solvers).
    pub solver_name: String,
    /// Whether this IK chain is active.
    pub enabled: bool,
    /// Evaluation priority (lower values evaluated first).
    pub priority: u32,
}

impl IKChainBinding {
    /// Create a new IK chain binding.
    pub fn new(chain: IKChain, solver_name: impl Into<String>) -> Self {
        Self {
            chain,
            solver_name: solver_name.into(),
            enabled: true,
            priority: 0,
        }
    }

    /// Set the evaluation priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Enable or disable this binding.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

// ---------------------------------------------------------------------------
// AnimatorComponent
// ---------------------------------------------------------------------------

/// Component that drives animation on an entity.
///
/// Combines a skeleton reference, animation player, blend tree or state machine,
/// IK chains, and the skinned mesh renderer into a single component that the
/// [`AnimationSystem`] processes each frame.
///
/// # Usage Modes
///
/// The animator supports three usage modes:
///
/// 1. **Direct playback**: Use the `player` to play clips directly.
/// 2. **Blend tree**: Set `blend_tree` for procedural blending driven by
///    parameters (e.g., speed, direction).
/// 3. **State machine**: Set `state_machine` for state-driven animation
///    with automatic transitions.
///
/// When a state machine is present, it takes precedence over the player
/// and blend tree.
#[derive(Debug)]
pub struct AnimatorComponent {
    /// Reference to the skeleton asset (name or handle).
    pub skeleton_name: String,

    /// Loaded skeleton data (set by AnimationSystem on initialization).
    pub skeleton: Option<Skeleton>,

    /// All animation clips available to this animator.
    pub clips: Vec<AnimationClip>,

    /// The animation player managing playback state.
    pub player: AnimationPlayer,

    /// Optional blend tree for procedural blending.
    pub blend_tree: Option<BlendTree>,

    /// Optional state machine for state-driven animation.
    pub state_machine: Option<AnimationStateMachine>,

    /// IK chains to solve after animation evaluation.
    pub ik_chains: Vec<IKChainBinding>,

    /// Skinned mesh renderer data (final bone matrices for GPU upload).
    pub skinned_mesh: SkinnedMeshRenderer,

    /// Whether the animator is enabled.
    pub enabled: bool,

    /// Current local-space bone transforms from the most recent evaluation.
    pub local_bone_transforms: Vec<BoneTransform>,

    /// Current world-space bone transforms (computed from hierarchy traversal).
    pub world_bone_transforms: Vec<Mat4>,

    /// Current pose as Transform array (for blending operations).
    pub current_pose: Vec<Transform>,

    /// Shared blend parameters (accessible from gameplay code).
    pub blend_params: BlendParams,

    /// Root motion extracted from the current frame.
    pub root_motion: Option<crate::skeleton::RootMotion>,

    /// Whether to extract root motion from animations.
    pub root_motion_enabled: bool,
}

impl Default for AnimatorComponent {
    fn default() -> Self {
        Self {
            skeleton_name: String::new(),
            skeleton: None,
            clips: Vec::new(),
            player: AnimationPlayer::default(),
            blend_tree: None,
            state_machine: None,
            ik_chains: Vec::new(),
            skinned_mesh: SkinnedMeshRenderer::default(),
            enabled: true,
            local_bone_transforms: Vec::new(),
            world_bone_transforms: Vec::new(),
            current_pose: Vec::new(),
            blend_params: BlendParams::new(),
            root_motion: None,
            root_motion_enabled: false,
        }
    }
}

impl AnimatorComponent {
    /// Create a new animator component with a skeleton and clips.
    pub fn new(skeleton: Skeleton, clips: Vec<AnimationClip>) -> Self {
        let bone_count = skeleton.bone_count();
        Self {
            skeleton_name: skeleton.name.clone(),
            local_bone_transforms: vec![BoneTransform::default(); bone_count],
            world_bone_transforms: vec![Mat4::IDENTITY; bone_count],
            current_pose: vec![Transform::IDENTITY; bone_count],
            skinned_mesh: SkinnedMeshRenderer {
                skeleton_name: skeleton.name.clone(),
                bone_matrices: vec![Mat4::IDENTITY; bone_count],
                max_bones_per_vertex: 4,
                enabled: true,
            },
            skeleton: Some(skeleton),
            clips,
            ..Default::default()
        }
    }

    /// Set up a blend tree for this animator.
    pub fn set_blend_tree(&mut self, tree: BlendTree) {
        self.blend_tree = Some(tree);
    }

    /// Set up a state machine for this animator.
    pub fn set_state_machine(&mut self, sm: AnimationStateMachine) {
        self.state_machine = Some(sm);
    }

    /// Add an IK chain binding.
    pub fn add_ik_chain(&mut self, binding: IKChainBinding) {
        self.ik_chains.push(binding);
    }

    /// Play an animation clip by name.
    pub fn play(&mut self, clip_name: &str) {
        if let Some(index) = self.clips.iter().position(|c| c.name == clip_name) {
            self.player.play(index);
        } else {
            log::warn!(
                "Animation clip '{}' not found on animator '{}'",
                clip_name,
                self.skeleton_name
            );
        }
    }

    /// Play an animation clip by index.
    pub fn play_index(&mut self, clip_index: usize) {
        self.player.play(clip_index);
    }

    /// Crossfade to an animation clip by name.
    pub fn crossfade(&mut self, clip_name: &str, duration: f32) {
        if let Some(index) = self.clips.iter().position(|c| c.name == clip_name) {
            self.player.crossfade(index, duration);
        } else {
            log::warn!(
                "Animation clip '{}' not found on animator '{}'",
                clip_name,
                self.skeleton_name
            );
        }
    }

    /// Set a float parameter on both the state machine and blend tree.
    pub fn set_float(&mut self, name: &str, value: f32) {
        self.blend_params.set(name, value);
        if let Some(sm) = self.state_machine.as_mut() {
            sm.set_float(name, value);
        }
        if let Some(bt) = self.blend_tree.as_mut() {
            bt.set_parameter(name, value);
        }
    }

    /// Set a bool parameter on the state machine.
    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.blend_params.set_bool(name, value);
        if let Some(sm) = self.state_machine.as_mut() {
            sm.set_bool(name, value);
        }
    }

    /// Fire a trigger on the state machine.
    pub fn set_trigger(&mut self, name: &str) {
        if let Some(sm) = self.state_machine.as_mut() {
            sm.set_trigger(name);
        }
    }

    /// Get the current animation state name (from state machine).
    pub fn current_state(&self) -> Option<&str> {
        self.state_machine
            .as_ref()
            .map(|sm| sm.current_state_name())
    }

    /// Get the number of bones in the skeleton.
    pub fn bone_count(&self) -> usize {
        self.skeleton
            .as_ref()
            .map(|s| s.bone_count())
            .unwrap_or(0)
    }

    /// Get the world-space position of a bone by index.
    pub fn bone_world_position(&self, bone_index: usize) -> Option<Vec3> {
        self.world_bone_transforms.get(bone_index).map(|m| {
            let col = m.col(3);
            Vec3::new(col.x, col.y, col.z)
        })
    }

    /// Get the world-space position of a bone by name.
    pub fn bone_world_position_by_name(&self, name: &str) -> Option<Vec3> {
        let skeleton = self.skeleton.as_ref()?;
        let idx = skeleton.find_bone(name)?;
        self.bone_world_position(idx)
    }

    /// Enable or disable root motion extraction.
    pub fn set_root_motion(&mut self, enabled: bool) {
        self.root_motion_enabled = enabled;
    }

    /// Get the root motion delta for the current frame.
    pub fn get_root_motion(&self) -> Option<&crate::skeleton::RootMotion> {
        self.root_motion.as_ref()
    }

    /// Initialize IK chain joint positions from the current world transforms.
    pub fn sync_ik_chains(&mut self) {
        let world_transforms = &self.world_bone_transforms;
        for binding in &mut self.ik_chains {
            for (i, &bone_idx) in binding.chain.joints.iter().enumerate() {
                if bone_idx < world_transforms.len() {
                    let col = world_transforms[bone_idx].col(3);
                    binding.chain.joint_positions[i] = Vec3::new(col.x, col.y, col.z);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Animation system
// ---------------------------------------------------------------------------

/// System that evaluates animations and produces final bone matrices each frame.
///
/// The `AnimationSystem` is the central coordinator for all animation
/// processing. It maintains a registry of IK solvers and processes each
/// entity's `AnimatorComponent` through the full evaluation pipeline.
///
/// # Processing Order
///
/// For each entity with an `AnimatorComponent`:
///
/// 1. Update state machine (if present)
/// 2. Evaluate blend tree or sample clips
/// 3. Apply IK solvers
/// 4. Compute world-space bone transforms
/// 5. Compute skinning matrices
pub struct AnimationSystem {
    /// Registered IK solvers available for use by animator components.
    ik_solvers: Vec<(String, Box<dyn IKSolver>)>,

    /// Whether the animation system is paused.
    pub paused: bool,

    /// Global speed multiplier for all animations.
    pub global_speed: f32,
}

impl std::fmt::Debug for AnimationSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationSystem")
            .field("paused", &self.paused)
            .field("global_speed", &self.global_speed)
            .field("solver_count", &self.ik_solvers.len())
            .finish()
    }
}

impl Default for AnimationSystem {
    fn default() -> Self {
        Self {
            ik_solvers: Vec::new(),
            paused: false,
            global_speed: 1.0,
        }
    }
}

impl AnimationSystem {
    /// Create a new animation system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an IK solver with a given name.
    pub fn register_ik_solver(&mut self, name: impl Into<String>, solver: Box<dyn IKSolver>) {
        self.ik_solvers.push((name.into(), solver));
    }

    /// Find a registered IK solver by name.
    pub fn find_solver(&self, name: &str) -> Option<&dyn IKSolver> {
        self.ik_solvers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, s)| s.as_ref())
    }

    /// Get the number of registered IK solvers.
    pub fn solver_count(&self) -> usize {
        self.ik_solvers.len()
    }

    /// Per-frame update: evaluate all animator components.
    ///
    /// In a real ECS integration, this would iterate over all entities with
    /// `AnimatorComponent`. Here, we provide a method that processes a
    /// slice of animators.
    pub fn update(&self, dt: f32, animators: &mut [AnimatorComponent]) {
        if self.paused {
            return;
        }

        let effective_dt = dt * self.global_speed;

        profiling::scope!("AnimationSystem::update");

        for animator in animators.iter_mut() {
            self.evaluate_animator(animator, effective_dt);
        }
    }

    /// Process a single animator component through the full pipeline.
    ///
    /// This is the core animation evaluation function. It handles all
    /// three usage modes (direct playback, blend tree, state machine)
    /// and produces final skinning matrices.
    pub fn evaluate_animator(
        &self,
        animator: &mut AnimatorComponent,
        dt: f32,
    ) {
        if !animator.enabled {
            return;
        }

        let skeleton = match animator.skeleton.as_ref() {
            Some(s) => s,
            None => return,
        };

        let bone_count = skeleton.bone_count();

        // Ensure arrays are correctly sized.
        Self::ensure_buffer_sizes(animator, bone_count);

        // Stage 1: Determine the current pose.
        let pose = self.compute_pose(animator, dt, bone_count);

        // Store the current pose.
        animator.current_pose = pose.clone();

        // Update local bone transforms from the pose.
        for (i, transform) in pose.iter().enumerate() {
            if i < animator.local_bone_transforms.len() {
                animator.local_bone_transforms[i] = BoneTransform::from_transform(transform);
            }
        }

        // Stage 2: Compute world-space bone transforms.
        let skeleton = animator.skeleton.as_ref().unwrap();
        let world_transforms = skeleton.compute_world_transforms(&pose);
        animator.world_bone_transforms = world_transforms;

        // Stage 3: Apply IK solvers.
        self.apply_ik_solvers(animator);

        // Stage 4: Compute skinning matrices.
        let skeleton = animator.skeleton.as_ref().unwrap();
        if animator.skinned_mesh.enabled {
            let skin_matrices = skeleton.compute_skin_matrices(&animator.world_bone_transforms);
            animator.skinned_mesh.bone_matrices = skin_matrices;
        }
    }

    /// Ensure all buffer arrays in the animator are the correct size.
    fn ensure_buffer_sizes(animator: &mut AnimatorComponent, bone_count: usize) {
        if animator.local_bone_transforms.len() != bone_count {
            animator
                .local_bone_transforms
                .resize(bone_count, BoneTransform::default());
        }
        if animator.world_bone_transforms.len() != bone_count {
            animator
                .world_bone_transforms
                .resize(bone_count, Mat4::IDENTITY);
        }
        if animator.skinned_mesh.bone_matrices.len() != bone_count {
            animator
                .skinned_mesh
                .bone_matrices
                .resize(bone_count, Mat4::IDENTITY);
        }
        if animator.current_pose.len() != bone_count {
            animator
                .current_pose
                .resize(bone_count, Transform::IDENTITY);
        }
    }

    /// Compute the current animation pose based on the animator's mode.
    ///
    /// Priority: state machine > blend tree > direct player.
    fn compute_pose(
        &self,
        animator: &mut AnimatorComponent,
        dt: f32,
        bone_count: usize,
    ) -> Vec<Transform> {
        // Start with the bind pose as a fallback.
        let bind_pose = animator
            .skeleton
            .as_ref()
            .map(|s| s.bind_pose())
            .unwrap_or_else(|| vec![Transform::IDENTITY; bone_count]);

        // Mode 1: State machine (highest priority).
        if let Some(ref mut sm) = animator.state_machine {
            sm.update(dt);

            // Extract root motion before the pose changes.
            let prev_time = sm.state_time - dt;
            let curr_time = sm.state_time;
            if animator.root_motion_enabled {
                if let Some(skeleton) = animator.skeleton.as_ref() {
                    // Try to extract root motion from the current state's first clip.
                    let state = &sm.states[sm.current_state];
                    let clip_indices = state.blend_tree.clip_indices();
                    if let Some(&clip_idx) = clip_indices.first() {
                        if clip_idx < animator.clips.len() {
                            animator.root_motion = Some(crate::skeleton::RootMotion::extract(
                                &animator.clips[clip_idx],
                                skeleton.root_bone_index,
                                prev_time,
                                curr_time,
                            ));
                        }
                    }
                }
            }

            return sm.evaluate(&animator.clips, bone_count);
        }

        // Mode 2: Blend tree.
        if let Some(ref blend_tree) = animator.blend_tree {
            let clip_duration = animator
                .clips
                .first()
                .map(|c| c.duration)
                .unwrap_or(1.0);

            let prev_time = animator.player.playback_time;
            animator.player.advance(dt, clip_duration);

            // Extract root motion.
            if animator.root_motion_enabled {
                if let Some(skeleton) = animator.skeleton.as_ref() {
                    if let Some(&clip_idx) = blend_tree.clip_indices().first() {
                        if clip_idx < animator.clips.len() {
                            animator.root_motion = Some(crate::skeleton::RootMotion::extract(
                                &animator.clips[clip_idx],
                                skeleton.root_bone_index,
                                prev_time,
                                animator.player.playback_time,
                            ));
                        }
                    }
                }
            }

            return blend_tree.evaluate_with_params(
                &animator.clips,
                &animator.blend_params,
                animator.player.playback_time,
                bone_count,
            );
        }

        // Mode 3: Direct player.
        if let Some(clip_idx) = animator.player.current_clip {
            let clip_duration = if clip_idx < animator.clips.len() {
                animator.clips[clip_idx].duration
            } else {
                1.0
            };

            let prev_time = animator.player.playback_time;
            animator.player.advance(dt, clip_duration);

            // Extract root motion.
            if animator.root_motion_enabled {
                if let Some(skeleton) = animator.skeleton.as_ref() {
                    if clip_idx < animator.clips.len() {
                        animator.root_motion = Some(crate::skeleton::RootMotion::extract(
                            &animator.clips[clip_idx],
                            skeleton.root_bone_index,
                            prev_time,
                            animator.player.playback_time,
                        ));
                    }
                }
            }

            let pose = animator.player.sample(&animator.clips, bone_count);
            return pose;
        }

        // No animation active: return bind pose.
        bind_pose
    }

    /// Apply IK solvers to the animator's world-space bone transforms.
    fn apply_ik_solvers(&self, animator: &mut AnimatorComponent) {
        if animator.ik_chains.is_empty() {
            return;
        }

        // Sort IK chains by priority.
        animator
            .ik_chains
            .sort_by_key(|binding| binding.priority);

        // Sync IK chain joint positions from current world transforms.
        for binding in &mut animator.ik_chains {
            if !binding.enabled {
                continue;
            }

            // Read world positions into the chain.
            for (i, &bone_idx) in binding.chain.joints.iter().enumerate() {
                if bone_idx < animator.world_bone_transforms.len() {
                    let col = animator.world_bone_transforms[bone_idx].col(3);
                    binding.chain.joint_positions[i] = Vec3::new(col.x, col.y, col.z);
                }
            }

            // Find and apply the solver.
            if let Some(solver) = self.find_solver(&binding.solver_name) {
                solver.solve(&mut binding.chain);

                // Write solved positions back into world transforms.
                // This is a simplified approach -- a production engine would
                // recompute the full hierarchy from the modified joints.
                for (i, &bone_idx) in binding.chain.joints.iter().enumerate() {
                    if bone_idx < animator.world_bone_transforms.len() {
                        let pos = binding.chain.joint_positions[i];
                        let rot = binding.chain.joint_rotations[i];

                        // Preserve the existing scale from the world transform.
                        let existing = animator.world_bone_transforms[bone_idx];
                        let scale_x = existing.col(0).truncate().length();
                        let scale_y = existing.col(1).truncate().length();
                        let scale_z = existing.col(2).truncate().length();

                        let rot_mat = Mat4::from_quat(rot);
                        let scale_mat =
                            Mat4::from_scale(Vec3::new(scale_x, scale_y, scale_z));
                        let translation = Mat4::from_translation(pos);

                        animator.world_bone_transforms[bone_idx] =
                            translation * rot_mat * scale_mat;
                    }
                }
            }
        }

        // Recompute skinning matrices after IK modification.
        if let Some(skeleton) = animator.skeleton.as_ref() {
            if animator.skinned_mesh.enabled {
                let skin = skeleton.compute_skin_matrices(&animator.world_bone_transforms);
                animator.skinned_mesh.bone_matrices = skin;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blend_tree::{
        AnimationState, AnimationStateMachine, BlendNode, BlendParameter, BlendTree,
        StateTransition, TransitionCondition,
    };
    use crate::ik::{CCDSolver, FABRIKSolver, TwoBoneIK};
    use crate::skeleton::{AnimationClip, Bone, BoneTrack, Keyframe, Skeleton};

    /// Helper: build a simple skeleton.
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

    /// Helper: build a simple animation clip.
    fn make_clip(name: &str, bone_count: usize) -> AnimationClip {
        let mut clip = AnimationClip::new(name, 1.0);
        clip.looping = true;
        for i in 0..bone_count {
            let mut track = BoneTrack::new(i);
            track.position_keys = vec![
                Keyframe::new(0.0, Vec3::new(0.0, i as f32, 0.0)),
                Keyframe::new(0.5, Vec3::new(0.0, i as f32 + 0.1, 0.0)),
                Keyframe::new(1.0, Vec3::new(0.0, i as f32, 0.0)),
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

    // -- BoneTransform tests --

    #[test]
    fn test_bone_transform_default() {
        let bt = BoneTransform::default();
        assert_eq!(bt.position, Vec3::ZERO);
        assert_eq!(bt.rotation, Quat::IDENTITY);
        assert_eq!(bt.scale, Vec3::ONE);
    }

    #[test]
    fn test_bone_transform_conversion() {
        let t = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(0.5),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let bt = BoneTransform::from_transform(&t);
        let back = bt.to_transform();
        assert!((back.position - t.position).length() < f32::EPSILON);
        assert!(back.rotation.dot(t.rotation).abs() > 0.999);
        assert!((back.scale - t.scale).length() < f32::EPSILON);
    }

    // -- IKChainBinding tests --

    #[test]
    fn test_ik_chain_binding() {
        let chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.0, 0.0, 0.0));
        let binding = IKChainBinding::new(chain, "TwoBone").with_priority(10);
        assert_eq!(binding.solver_name, "TwoBone");
        assert_eq!(binding.priority, 10);
        assert!(binding.enabled);
    }

    // -- AnimatorComponent tests --

    #[test]
    fn test_animator_creation() {
        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];
        let animator = AnimatorComponent::new(skel, clips);
        assert_eq!(animator.bone_count(), 3);
        assert_eq!(animator.local_bone_transforms.len(), 3);
        assert_eq!(animator.world_bone_transforms.len(), 3);
        assert_eq!(animator.skinned_mesh.bone_matrices.len(), 3);
    }

    #[test]
    fn test_animator_play() {
        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3), make_clip("Run", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);

        animator.play("Walk");
        assert_eq!(animator.player.current_clip, Some(0));

        animator.play("Run");
        assert_eq!(animator.player.current_clip, Some(1));
    }

    #[test]
    fn test_animator_play_nonexistent() {
        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);
        animator.play("Nonexistent"); // Should warn but not crash.
        assert_eq!(animator.player.current_clip, None);
    }

    #[test]
    fn test_animator_crossfade() {
        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3), make_clip("Run", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);

        animator.play("Walk");
        animator.crossfade("Run", 0.3);
        assert!(animator.player.crossfading);
        assert_eq!(animator.player.current_clip, Some(1));
        assert_eq!(animator.player.crossfade_from_clip, Some(0));
    }

    #[test]
    fn test_animator_set_parameters() {
        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);

        // Set up a state machine with parameters.
        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_float_parameter("Speed", 0.0, 10.0, 0.0);
        sm.add_bool_parameter("IsGrounded", true);
        animator.set_state_machine(sm);

        animator.set_float("Speed", 5.0);
        animator.set_bool("IsGrounded", false);

        let sm = animator.state_machine.as_ref().unwrap();
        let speed = sm.parameters.iter().find(|p| p.name == "Speed").unwrap();
        assert!((speed.value - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_animator_bone_position() {
        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);

        // Set up some world transforms.
        animator.world_bone_transforms[0] = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let pos = animator.bone_world_position(0).unwrap();
        assert!((pos - Vec3::new(1.0, 2.0, 3.0)).length() < f32::EPSILON);
    }

    #[test]
    fn test_animator_bone_position_by_name() {
        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);

        animator.world_bone_transforms[1] = Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0));
        let pos = animator.bone_world_position_by_name("Spine").unwrap();
        assert!((pos.y - 5.0).abs() < f32::EPSILON);
    }

    // -- AnimationSystem tests --

    #[test]
    fn test_system_creation() {
        let system = AnimationSystem::new();
        assert!(!system.paused);
        assert!((system.global_speed - 1.0).abs() < f32::EPSILON);
        assert_eq!(system.solver_count(), 0);
    }

    #[test]
    fn test_system_register_solver() {
        let mut system = AnimationSystem::new();
        system.register_ik_solver("TwoBone", Box::new(TwoBoneIK));
        system.register_ik_solver("CCD", Box::new(CCDSolver::default()));
        system.register_ik_solver("FABRIK", Box::new(FABRIKSolver::default()));
        assert_eq!(system.solver_count(), 3);

        assert!(system.find_solver("TwoBone").is_some());
        assert!(system.find_solver("CCD").is_some());
        assert!(system.find_solver("FABRIK").is_some());
        assert!(system.find_solver("Unknown").is_none());
    }

    #[test]
    fn test_system_evaluate_direct_playback() {
        let system = AnimationSystem::new();
        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);
        animator.play("Walk");

        // Evaluate for a few frames.
        for _ in 0..10 {
            system.evaluate_animator(&mut animator, 1.0 / 60.0);
        }

        // Check that world transforms were computed.
        assert_eq!(animator.world_bone_transforms.len(), 3);
        // Check that skinning matrices were computed.
        assert_eq!(animator.skinned_mesh.bone_matrices.len(), 3);

        // The root bone should have a valid world transform.
        let root_pos = animator.world_bone_transforms[0].col(3);
        // Root bone position should come from the animation.
        assert!(root_pos.w > 0.0, "Root transform should be valid");
    }

    #[test]
    fn test_system_evaluate_blend_tree() {
        let system = AnimationSystem::new();
        let skel = make_skeleton();

        // Create two clips with different positions.
        let mut clip_a = AnimationClip::new("Idle", 1.0);
        clip_a.looping = true;
        for i in 0..3 {
            let mut t = BoneTrack::new(i);
            t.position_keys = vec![
                Keyframe::new(0.0, Vec3::ZERO),
                Keyframe::new(1.0, Vec3::ZERO),
            ];
            t.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
            t.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
            clip_a.add_track(t);
        }

        let mut clip_b = AnimationClip::new("Walk", 1.0);
        clip_b.looping = true;
        for i in 0..3 {
            let mut t = BoneTrack::new(i);
            t.position_keys = vec![
                Keyframe::new(0.0, Vec3::new(10.0, 0.0, 0.0)),
                Keyframe::new(1.0, Vec3::new(10.0, 0.0, 0.0)),
            ];
            t.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
            t.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
            clip_b.add_track(t);
        }

        let clips = vec![clip_a, clip_b];
        let mut animator = AnimatorComponent::new(skel, clips);

        // Set up a 1D blend tree.
        let tree = BlendTree::new(
            BlendNode::blend_1d(
                "Speed",
                vec![
                    (0.0, BlendNode::clip(0)),
                    (1.0, BlendNode::clip(1)),
                ],
            ),
            vec![BlendParameter::new("Speed", 0.0, 1.0, 0.5)],
        );
        animator.set_blend_tree(tree);
        animator.blend_params.add_float("Speed", 0.0, 1.0, 0.5);
        animator.player.play(0);

        system.evaluate_animator(&mut animator, 0.016);

        // Pose should be blended between Idle and Walk.
        assert_eq!(animator.current_pose.len(), 3);
    }

    #[test]
    fn test_system_evaluate_state_machine() {
        let system = AnimationSystem::new();
        let skel = make_skeleton();
        let clips = vec![make_clip("Idle", 3), make_clip("Walk", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);

        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_state(AnimationState::from_clip("Walk", 1));
        sm.add_float_parameter("Speed", 0.0, 10.0, 0.0);
        sm.add_transition(StateTransition::new(
            0,
            1,
            0.2,
            vec![TransitionCondition::float_gt("Speed", 0.5)],
        ));
        animator.set_state_machine(sm);

        // Evaluate in Idle state.
        system.evaluate_animator(&mut animator, 0.016);
        assert_eq!(
            animator.state_machine.as_ref().unwrap().current_state_name(),
            "Idle"
        );

        // Trigger transition to Walk.
        animator.set_float("Speed", 1.0);
        for _ in 0..30 {
            system.evaluate_animator(&mut animator, 0.016);
        }
        assert_eq!(
            animator.state_machine.as_ref().unwrap().current_state_name(),
            "Walk"
        );
    }

    #[test]
    fn test_system_evaluate_disabled() {
        let system = AnimationSystem::new();
        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);
        animator.enabled = false;
        animator.play("Walk");

        system.evaluate_animator(&mut animator, 0.016);
        // Should not modify transforms when disabled.
        // Player should not advance.
        assert!((animator.player.playback_time).abs() < f32::EPSILON);
    }

    #[test]
    fn test_system_evaluate_no_skeleton() {
        let system = AnimationSystem::new();
        let mut animator = AnimatorComponent::default();
        animator.play_index(0);

        // Should not crash with no skeleton.
        system.evaluate_animator(&mut animator, 0.016);
    }

    #[test]
    fn test_system_paused() {
        let system = AnimationSystem {
            ik_solvers: Vec::new(),
            paused: true,
            global_speed: 1.0,
        };

        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);
        animator.play("Walk");

        system.update(0.016, std::slice::from_mut(&mut animator));
        // When paused, player should not advance.
        assert!((animator.player.playback_time).abs() < f32::EPSILON);
    }

    #[test]
    fn test_system_global_speed() {
        let system = AnimationSystem {
            ik_solvers: Vec::new(),
            paused: false,
            global_speed: 2.0,
        };

        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);
        animator.play("Walk");
        animator.player.set_looping(false);

        system.update(0.5, std::slice::from_mut(&mut animator));
        // At 2x global speed, 0.5s real time = 1.0s playback.
        assert!(
            (animator.player.playback_time - 1.0).abs() < 0.01,
            "Global speed should multiply: {}",
            animator.player.playback_time
        );
    }

    #[test]
    fn test_system_with_ik() {
        let mut system = AnimationSystem::new();
        system.register_ik_solver("TwoBone", Box::new(TwoBoneIK));

        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];
        let mut animator = AnimatorComponent::new(skel, clips);
        animator.play("Walk");

        // Add an IK chain for the 3-bone chain.
        let chain = IKChain::new(vec![0, 1, 2], Vec3::new(0.5, 0.5, 0.0));
        animator.add_ik_chain(IKChainBinding::new(chain, "TwoBone"));

        // Evaluate.
        system.evaluate_animator(&mut animator, 0.016);

        // Should have valid transforms after IK.
        assert_eq!(animator.world_bone_transforms.len(), 3);
        assert_eq!(animator.skinned_mesh.bone_matrices.len(), 3);
    }

    #[test]
    fn test_system_multiple_animators() {
        let system = AnimationSystem::new();
        let skel = make_skeleton();
        let clips = vec![make_clip("Walk", 3)];

        let mut animator1 = AnimatorComponent::new(skel.clone(), clips.clone());
        animator1.play("Walk");

        let mut animator2 = AnimatorComponent::new(skel, clips);
        animator2.play("Walk");
        animator2.player.set_speed(2.0);

        let mut animators = vec![animator1, animator2];
        system.update(0.5, &mut animators);

        // Both should have been evaluated.
        assert!(!animators[0].current_pose.is_empty());
        assert!(!animators[1].current_pose.is_empty());
    }

    #[test]
    fn test_system_root_motion() {
        let system = AnimationSystem::new();

        let mut skel = make_skeleton();
        skel.root_bone_index = 0;

        let mut clip = AnimationClip::new("Walk", 1.0);
        clip.looping = true;
        let mut root_track = BoneTrack::new(0);
        root_track.position_keys = vec![
            Keyframe::new(0.0, Vec3::ZERO),
            Keyframe::new(1.0, Vec3::new(0.0, 0.0, 2.0)),
        ];
        root_track.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
        root_track.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
        clip.add_track(root_track);

        for i in 1..3 {
            let mut t = BoneTrack::new(i);
            t.position_keys = vec![Keyframe::new(0.0, Vec3::new(0.0, i as f32, 0.0))];
            t.rotation_keys = vec![Keyframe::new(0.0, Quat::IDENTITY)];
            t.scale_keys = vec![Keyframe::new(0.0, Vec3::ONE)];
            clip.add_track(t);
        }

        let mut animator = AnimatorComponent::new(skel, vec![clip]);
        animator.set_root_motion(true);
        animator.play("Walk");

        system.evaluate_animator(&mut animator, 0.5);

        // Root motion should have been extracted.
        assert!(animator.root_motion.is_some());
        let rm = animator.root_motion.unwrap();
        assert!(rm.delta_position.z > 0.0, "Root motion should have Z displacement");
    }

    // -- Full integration test --

    #[test]
    fn test_full_pipeline_integration() {
        let mut system = AnimationSystem::new();
        system.register_ik_solver("TwoBone", Box::new(TwoBoneIK));
        system.register_ik_solver("CCD", Box::new(CCDSolver::default()));
        system.register_ik_solver("FABRIK", Box::new(FABRIKSolver::default()));

        let skel = Skeleton::new(
            "Humanoid",
            vec![
                Bone::new("Hips", None, Transform::IDENTITY, Mat4::IDENTITY),
                Bone::new(
                    "Spine",
                    Some(0),
                    Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "LeftUpperArm",
                    Some(1),
                    Transform::from_position(Vec3::new(-0.3, 0.5, 0.0)),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "LeftForeArm",
                    Some(2),
                    Transform::from_position(Vec3::new(0.0, -0.3, 0.0)),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "LeftHand",
                    Some(3),
                    Transform::from_position(Vec3::new(0.0, -0.3, 0.0)),
                    Mat4::IDENTITY,
                ),
                Bone::new(
                    "Head",
                    Some(1),
                    Transform::from_position(Vec3::new(0.0, 0.3, 0.0)),
                    Mat4::IDENTITY,
                ),
            ],
        );

        let bone_count = skel.bone_count();

        // Create clips.
        let make_full_clip = |name: &str| -> AnimationClip {
            let mut clip = AnimationClip::new(name, 1.0);
            clip.looping = true;
            for i in 0..bone_count {
                let mut t = BoneTrack::new(i);
                t.position_keys = vec![
                    Keyframe::new(0.0, Vec3::ZERO),
                    Keyframe::new(0.5, Vec3::new(0.0, 0.05, 0.0)),
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
                clip.add_track(t);
            }
            clip
        };

        let clips = vec![
            make_full_clip("Idle"),
            make_full_clip("Walk"),
            make_full_clip("Run"),
        ];

        let mut animator = AnimatorComponent::new(skel, clips);

        // Set up state machine.
        let locomotion_tree = BlendTree::new(
            BlendNode::blend_1d(
                "Speed",
                vec![
                    (0.0, BlendNode::clip(1)),  // Walk
                    (1.0, BlendNode::clip(2)),  // Run
                ],
            ),
            vec![BlendParameter::new("Speed", 0.0, 1.0, 0.0)],
        );

        let mut sm = AnimationStateMachine::new(AnimationState::from_clip("Idle", 0));
        sm.add_state(AnimationState::new("Locomotion", locomotion_tree));
        sm.add_float_parameter("Speed", 0.0, 10.0, 0.0);

        sm.add_transition(StateTransition::new(
            0,
            1,
            0.3,
            vec![TransitionCondition::float_gt("Speed", 0.1)],
        ));
        sm.add_transition(StateTransition::new(
            1,
            0,
            0.3,
            vec![TransitionCondition::float_lt("Speed", 0.1)],
        ));

        animator.set_state_machine(sm);

        // Add IK for left arm.
        let arm_chain = IKChain::new(vec![2, 3, 4], Vec3::new(-0.5, 1.0, 0.3));
        animator.add_ik_chain(IKChainBinding::new(arm_chain, "TwoBone"));

        // Simulate several frames in Idle.
        for _ in 0..10 {
            system.evaluate_animator(&mut animator, 1.0 / 60.0);
        }
        assert_eq!(
            animator.state_machine.as_ref().unwrap().current_state_name(),
            "Idle"
        );
        assert!(!animator.current_pose.is_empty());
        assert!(!animator.skinned_mesh.bone_matrices.is_empty());

        // Transition to Locomotion.
        animator.set_float("Speed", 5.0);
        for _ in 0..30 {
            system.evaluate_animator(&mut animator, 1.0 / 60.0);
        }
        assert_eq!(
            animator.state_machine.as_ref().unwrap().current_state_name(),
            "Locomotion"
        );

        // Verify all outputs are valid.
        assert_eq!(animator.current_pose.len(), bone_count);
        assert_eq!(animator.world_bone_transforms.len(), bone_count);
        assert_eq!(animator.skinned_mesh.bone_matrices.len(), bone_count);

        // Transition back to Idle.
        animator.set_float("Speed", 0.0);
        for _ in 0..30 {
            system.evaluate_animator(&mut animator, 1.0 / 60.0);
        }
        assert_eq!(
            animator.state_machine.as_ref().unwrap().current_state_name(),
            "Idle"
        );
    }
}
