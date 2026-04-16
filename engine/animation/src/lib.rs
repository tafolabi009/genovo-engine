//! # Genovo Animation
//!
//! Skeletal animation, inverse kinematics, and blend tree module for the
//! Genovo game engine. Provides runtime animation evaluation, IK solvers,
//! state machines, and ECS-integrated components.

pub mod blend_tree;
pub mod components;
pub mod ik;
pub mod morph_targets;
pub mod root_motion;
pub mod skeleton;

// Re-exports for ergonomic top-level access.
pub use blend_tree::{
    AnimationStateMachine, BlendNode, BlendParameter, BlendParams, BlendTree, BlendTreeOutput,
    StateTransition, TransitionCondition,
};
pub use components::{AnimationSystem, AnimatorComponent, BoneTransform, IKChainBinding};
pub use ik::{CCDSolver, FABRIKSolver, IKChain, IKSolver, LookAtIK, TwoBoneIK};
pub use morph_targets::{
    MorphTarget, MorphTargetAnimation, MorphTargetSet, MorphTargetWeights, MorphVertex,
    VertexDelta, apply_morph_targets,
};
pub use root_motion::{RootMotionBlender, RootMotionExtractor, RootMotionMode};
pub use skeleton::{
    AnimationClip, AnimationEvent, AnimationPlayer, Bone, BoneMask, BoneTrack, Keyframe,
    RootMotion, Skeleton, SkinnedMeshRenderer,
};
