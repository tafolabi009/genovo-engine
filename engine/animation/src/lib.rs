//! # Genovo Animation
//!
//! Skeletal animation, inverse kinematics, and blend tree module for the
//! Genovo game engine. Provides runtime animation evaluation, IK solvers,
//! state machines, blend spaces, animation retargeting, additive animation
//! layers, spring/jiggle bone physics, and ECS-integrated components.

pub mod additive_animations;
pub mod animation_graph;
pub mod blend_spaces;
pub mod blend_tree;
pub mod components;
pub mod foot_ik;
pub mod ik;
pub mod morph_targets;
pub mod retargeting;
pub mod root_motion;
pub mod skeleton;
pub mod spring_bone;

// Re-exports for ergonomic top-level access.
pub use additive_animations::{
    AdditiveClip, AdditiveBoneTrack, AimOffset, AnimationLayer, DeltaTransform, LayerBlendMode,
    LayerStack, apply_additive, apply_additive_masked,
};
pub use blend_spaces::{BlendSpace1D, BlendSpace2D, DirectionalBlendSpace};
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
pub use retargeting::{
    BoneMapping, MappingMethod, RetargetMap, SkeletonProfile, retarget_clip, retarget_clip_sampled,
    retarget_pose,
};
pub use root_motion::{RootMotionBlender, RootMotionExtractor, RootMotionMode};
pub use skeleton::{
    AnimationClip, AnimationEvent, AnimationPlayer, Bone, BoneMask, BoneTrack, Keyframe,
    RootMotion, Skeleton, SkinnedMeshRenderer,
};
pub use spring_bone::{
    JiggleBone, SpringBone, SpringBoneChain, SpringBoneSystem, SpringCollider, WindSource,
};
pub use animation_graph::{
    AnimGraphEvent, AnimationGraph, BlendMode, BlendNode, CameraMode as AnimCameraMode,
    EdgeId, GraphEdge, GraphNodeData, GraphOutput, IkNode, IkTarget, LayerNode, NodeId,
    NodeState, NodeType, ParamDescriptor, ParamId, ParamValue, SelectNode, StateNode,
    SubGraphNode, TransitionCondition, TransitionState,
};
pub use foot_ik::{
    FootDefinition, FootIkDebugInfo, FootIkSettings, FootIkState, FootIkSystem, FootSide,
    GroundHit, GroundRaycaster, TwoBoneIkResult, TwoBoneIkSolver,
};
