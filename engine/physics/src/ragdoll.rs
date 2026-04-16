//! Ragdoll physics system for skeleton-to-physics mapping.
//!
//! Provides:
//! - `RagdollDefinition`: describes how a skeleton maps to physics bodies and joints
//! - `RagdollInstance`: active ragdoll in the physics world
//! - Humanoid presets with proper joint limits
//! - Partial ragdoll: blend between animation and physics per bone
//! - ECS integration via `RagdollComponent` and `RagdollSystem`

use glam::{Quat, Vec3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default blend factor for animation-to-physics transition.
const DEFAULT_BLEND_FACTOR: f32 = 1.0;
/// Maximum number of bones supported.
const MAX_BONES: usize = 128;

// ---------------------------------------------------------------------------
// Body shape for ragdoll bones
// ---------------------------------------------------------------------------

/// Shape of a ragdoll body part.
#[derive(Debug, Clone)]
pub enum RagdollShape {
    /// Capsule oriented along the bone axis.
    Capsule {
        /// Radius of the capsule.
        radius: f32,
        /// Half-height of the cylindrical portion.
        half_height: f32,
    },
    /// Box shape.
    Box {
        /// Half-extents of the box.
        half_extents: Vec3,
    },
    /// Sphere shape.
    Sphere {
        /// Radius.
        radius: f32,
    },
}

impl RagdollShape {
    /// Estimate the volume of this shape.
    pub fn volume(&self) -> f32 {
        match self {
            RagdollShape::Capsule { radius, half_height } => {
                let sphere = (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius;
                let cylinder = std::f32::consts::PI * radius * radius * 2.0 * half_height;
                sphere + cylinder
            }
            RagdollShape::Box { half_extents } => {
                8.0 * half_extents.x * half_extents.y * half_extents.z
            }
            RagdollShape::Sphere { radius } => {
                (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Joint type and limits
// ---------------------------------------------------------------------------

/// Type of joint connecting a ragdoll body to its parent.
#[derive(Debug, Clone)]
pub enum RagdollJointType {
    /// Ball-and-socket joint (3 rotational DOF) with cone limit.
    BallSocket {
        /// Maximum cone angle in radians.
        cone_angle: f32,
        /// Twist limit around the bone axis (min, max) in radians.
        twist_limit: Option<(f32, f32)>,
    },
    /// Hinge joint (1 rotational DOF).
    Hinge {
        /// Hinge axis in parent's local space.
        axis: Vec3,
        /// Angular limits (min, max) in radians.
        limits: (f32, f32),
    },
    /// Fixed joint (0 DOF).
    Fixed,
}

/// Description of how a joint connects to the parent body.
#[derive(Debug, Clone)]
pub struct RagdollJointDesc {
    /// Type of joint.
    pub joint_type: RagdollJointType,
    /// Anchor point in the parent body's local space.
    pub parent_anchor: Vec3,
    /// Anchor point in this body's local space.
    pub child_anchor: Vec3,
}

// ---------------------------------------------------------------------------
// Bone body description
// ---------------------------------------------------------------------------

/// Description of a single body in the ragdoll, corresponding to one bone.
#[derive(Debug, Clone)]
pub struct BoneBody {
    /// Index of the bone in the skeleton.
    pub bone_index: usize,
    /// Name of the bone (for debugging).
    pub bone_name: String,
    /// The collision shape for this body.
    pub shape: RagdollShape,
    /// Mass of this body in kg.
    pub mass: f32,
    /// Local offset from the bone origin.
    pub local_offset: Vec3,
    /// Local rotation offset from the bone.
    pub local_rotation: Quat,
    /// Joint connecting to the parent bone (None for root).
    pub joint_to_parent: Option<RagdollJointDesc>,
    /// Index of the parent bone body in the ragdoll definition (-1 for root).
    pub parent_index: i32,
    /// Linear damping for this body.
    pub linear_damping: f32,
    /// Angular damping for this body.
    pub angular_damping: f32,
}

impl BoneBody {
    /// Create a new bone body with default damping.
    pub fn new(bone_index: usize, bone_name: &str, shape: RagdollShape, mass: f32) -> Self {
        Self {
            bone_index,
            bone_name: bone_name.to_string(),
            shape,
            mass,
            local_offset: Vec3::ZERO,
            local_rotation: Quat::IDENTITY,
            joint_to_parent: None,
            parent_index: -1,
            linear_damping: 0.05,
            angular_damping: 0.2,
        }
    }

    /// Set the joint connecting this body to its parent.
    pub fn with_joint(mut self, joint: RagdollJointDesc, parent_index: i32) -> Self {
        self.joint_to_parent = Some(joint);
        self.parent_index = parent_index;
        self
    }

    /// Set local offset and rotation.
    pub fn with_offset(mut self, offset: Vec3, rotation: Quat) -> Self {
        self.local_offset = offset;
        self.local_rotation = rotation;
        self
    }
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// A rigid transform (position + rotation) for bone readback.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    /// World-space position.
    pub position: Vec3,
    /// World-space rotation.
    pub rotation: Quat,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

impl Transform {
    /// Create a new transform.
    pub fn new(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }

    /// Linearly interpolate between two transforms.
    pub fn lerp(&self, other: &Transform, t: f32) -> Transform {
        Transform {
            position: self.position + (other.position - self.position) * t,
            rotation: self.rotation.slerp(other.rotation, t),
        }
    }
}

// ---------------------------------------------------------------------------
// Skeleton info (simplified representation)
// ---------------------------------------------------------------------------

/// Simplified skeleton information for ragdoll creation.
#[derive(Debug, Clone)]
pub struct SkeletonInfo {
    /// Bone transforms in bind pose (world space).
    pub bone_transforms: Vec<Transform>,
    /// Parent indices (-1 for root).
    pub parent_indices: Vec<i32>,
    /// Bone names.
    pub bone_names: Vec<String>,
    /// Bone lengths (distance to child, estimated).
    pub bone_lengths: Vec<f32>,
}

impl SkeletonInfo {
    /// Create a skeleton with the given number of bones.
    pub fn new(num_bones: usize) -> Self {
        Self {
            bone_transforms: vec![Transform::default(); num_bones],
            parent_indices: vec![-1; num_bones],
            bone_names: (0..num_bones).map(|i| format!("bone_{}", i)).collect(),
            bone_lengths: vec![0.2; num_bones],
        }
    }

    /// Get the number of bones.
    pub fn bone_count(&self) -> usize {
        self.bone_transforms.len()
    }
}

// ---------------------------------------------------------------------------
// RagdollDefinition
// ---------------------------------------------------------------------------

/// Complete description of a ragdoll: body shapes, masses, and joint configurations.
///
/// This is a blueprint that can be instantiated multiple times.
#[derive(Debug, Clone)]
pub struct RagdollDefinition {
    /// All body parts in the ragdoll.
    pub bodies: Vec<BoneBody>,
    /// Name of this ragdoll definition.
    pub name: String,
}

impl RagdollDefinition {
    /// Create an empty ragdoll definition.
    pub fn new(name: &str) -> Self {
        Self {
            bodies: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Add a body to the ragdoll definition.
    pub fn add_body(&mut self, body: BoneBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    /// Get the total number of bodies.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Get the total mass of the ragdoll.
    pub fn total_mass(&self) -> f32 {
        self.bodies.iter().map(|b| b.mass).sum()
    }
}

// ---------------------------------------------------------------------------
// Humanoid ragdoll preset
// ---------------------------------------------------------------------------

/// Standard humanoid bone indices for reference.
#[derive(Debug, Clone, Copy)]
pub struct HumanoidBoneIndices {
    pub pelvis: usize,
    pub spine: usize,
    pub chest: usize,
    pub head: usize,
    pub upper_arm_l: usize,
    pub lower_arm_l: usize,
    pub hand_l: usize,
    pub upper_arm_r: usize,
    pub lower_arm_r: usize,
    pub hand_r: usize,
    pub upper_leg_l: usize,
    pub lower_leg_l: usize,
    pub foot_l: usize,
    pub upper_leg_r: usize,
    pub lower_leg_r: usize,
    pub foot_r: usize,
}

impl Default for HumanoidBoneIndices {
    fn default() -> Self {
        Self {
            pelvis: 0,
            spine: 1,
            chest: 2,
            head: 3,
            upper_arm_l: 4,
            lower_arm_l: 5,
            hand_l: 6,
            upper_arm_r: 7,
            lower_arm_r: 8,
            hand_r: 9,
            upper_leg_l: 10,
            lower_leg_l: 11,
            foot_l: 12,
            upper_leg_r: 13,
            lower_leg_r: 14,
            foot_r: 15,
        }
    }
}

/// Create a humanoid ragdoll definition with proper joint limits.
///
/// The ragdoll has 16 body parts:
/// - Pelvis (root)
/// - Spine
/// - Chest
/// - Head
/// - Upper arms (L/R)
/// - Lower arms (L/R)
/// - Hands (L/R)
/// - Upper legs (L/R)
/// - Lower legs (L/R)
/// - Feet (L/R)
pub fn create_humanoid_ragdoll(skeleton: &SkeletonInfo) -> RagdollDefinition {
    let _bones = HumanoidBoneIndices::default();
    let mut def = RagdollDefinition::new("humanoid");

    // Helper closure for ball socket joints
    let ball_joint = |cone: f32, parent_anchor: Vec3, child_anchor: Vec3| -> RagdollJointDesc {
        RagdollJointDesc {
            joint_type: RagdollJointType::BallSocket {
                cone_angle: cone,
                twist_limit: Some((-0.5, 0.5)),
            },
            parent_anchor,
            child_anchor,
        }
    };

    // Helper closure for hinge joints
    let hinge_joint =
        |axis: Vec3, limits: (f32, f32), parent_anchor: Vec3, child_anchor: Vec3| -> RagdollJointDesc {
            RagdollJointDesc {
                joint_type: RagdollJointType::Hinge {
                    axis,
                    limits,
                },
                parent_anchor,
                child_anchor,
            }
        };

    // 0: Pelvis (root)
    let pelvis = BoneBody::new(
        0,
        "pelvis",
        RagdollShape::Box {
            half_extents: Vec3::new(0.15, 0.08, 0.1),
        },
        8.0,
    );
    let pelvis_idx = def.add_body(pelvis) as i32;

    // 1: Spine
    let spine = BoneBody::new(
        1,
        "spine",
        RagdollShape::Box {
            half_extents: Vec3::new(0.14, 0.1, 0.08),
        },
        6.0,
    )
    .with_joint(
        ball_joint(0.3, Vec3::new(0.0, 0.08, 0.0), Vec3::new(0.0, -0.1, 0.0)),
        pelvis_idx,
    );
    let spine_idx = def.add_body(spine) as i32;

    // 2: Chest
    let chest = BoneBody::new(
        2,
        "chest",
        RagdollShape::Box {
            half_extents: Vec3::new(0.16, 0.12, 0.1),
        },
        12.0,
    )
    .with_joint(
        ball_joint(0.25, Vec3::new(0.0, 0.1, 0.0), Vec3::new(0.0, -0.12, 0.0)),
        spine_idx,
    );
    let chest_idx = def.add_body(chest) as i32;

    // 3: Head
    let head = BoneBody::new(
        3,
        "head",
        RagdollShape::Sphere { radius: 0.1 },
        5.0,
    )
    .with_joint(
        ball_joint(0.5, Vec3::new(0.0, 0.12, 0.0), Vec3::new(0.0, -0.1, 0.0)),
        chest_idx,
    );
    let _head_idx = def.add_body(head) as i32;

    // 4: Upper arm left
    let upper_arm_l = BoneBody::new(
        4,
        "upper_arm_l",
        RagdollShape::Capsule {
            radius: 0.04,
            half_height: 0.12,
        },
        3.0,
    )
    .with_joint(
        ball_joint(1.2, Vec3::new(-0.16, 0.1, 0.0), Vec3::new(0.0, 0.12, 0.0)),
        chest_idx,
    );
    let upper_arm_l_idx = def.add_body(upper_arm_l) as i32;

    // 5: Lower arm left (elbow - hinge joint)
    let lower_arm_l = BoneBody::new(
        5,
        "lower_arm_l",
        RagdollShape::Capsule {
            radius: 0.035,
            half_height: 0.11,
        },
        2.0,
    )
    .with_joint(
        hinge_joint(
            Vec3::Z,
            (0.0, 2.5), // Elbow only bends one way
            Vec3::new(0.0, -0.12, 0.0),
            Vec3::new(0.0, 0.11, 0.0),
        ),
        upper_arm_l_idx,
    );
    let lower_arm_l_idx = def.add_body(lower_arm_l) as i32;

    // 6: Hand left
    let hand_l = BoneBody::new(
        6,
        "hand_l",
        RagdollShape::Box {
            half_extents: Vec3::new(0.04, 0.01, 0.06),
        },
        0.5,
    )
    .with_joint(
        ball_joint(0.5, Vec3::new(0.0, -0.11, 0.0), Vec3::new(0.0, 0.01, 0.0)),
        lower_arm_l_idx,
    );
    let _hand_l_idx = def.add_body(hand_l) as i32;

    // 7: Upper arm right
    let upper_arm_r = BoneBody::new(
        7,
        "upper_arm_r",
        RagdollShape::Capsule {
            radius: 0.04,
            half_height: 0.12,
        },
        3.0,
    )
    .with_joint(
        ball_joint(1.2, Vec3::new(0.16, 0.1, 0.0), Vec3::new(0.0, 0.12, 0.0)),
        chest_idx,
    );
    let upper_arm_r_idx = def.add_body(upper_arm_r) as i32;

    // 8: Lower arm right
    let lower_arm_r = BoneBody::new(
        8,
        "lower_arm_r",
        RagdollShape::Capsule {
            radius: 0.035,
            half_height: 0.11,
        },
        2.0,
    )
    .with_joint(
        hinge_joint(
            Vec3::Z,
            (0.0, 2.5),
            Vec3::new(0.0, -0.12, 0.0),
            Vec3::new(0.0, 0.11, 0.0),
        ),
        upper_arm_r_idx,
    );
    let lower_arm_r_idx = def.add_body(lower_arm_r) as i32;

    // 9: Hand right
    let hand_r = BoneBody::new(
        9,
        "hand_r",
        RagdollShape::Box {
            half_extents: Vec3::new(0.04, 0.01, 0.06),
        },
        0.5,
    )
    .with_joint(
        ball_joint(0.5, Vec3::new(0.0, -0.11, 0.0), Vec3::new(0.0, 0.01, 0.0)),
        lower_arm_r_idx,
    );
    let _hand_r_idx = def.add_body(hand_r) as i32;

    // 10: Upper leg left
    let upper_leg_l = BoneBody::new(
        10,
        "upper_leg_l",
        RagdollShape::Capsule {
            radius: 0.055,
            half_height: 0.18,
        },
        8.0,
    )
    .with_joint(
        ball_joint(0.8, Vec3::new(-0.1, -0.08, 0.0), Vec3::new(0.0, 0.18, 0.0)),
        pelvis_idx,
    );
    let upper_leg_l_idx = def.add_body(upper_leg_l) as i32;

    // 11: Lower leg left (knee - hinge joint)
    let lower_leg_l = BoneBody::new(
        11,
        "lower_leg_l",
        RagdollShape::Capsule {
            radius: 0.045,
            half_height: 0.18,
        },
        5.0,
    )
    .with_joint(
        hinge_joint(
            Vec3::X,
            (-2.5, 0.0), // Knee only bends backward
            Vec3::new(0.0, -0.18, 0.0),
            Vec3::new(0.0, 0.18, 0.0),
        ),
        upper_leg_l_idx,
    );
    let lower_leg_l_idx = def.add_body(lower_leg_l) as i32;

    // 12: Foot left
    let foot_l = BoneBody::new(
        12,
        "foot_l",
        RagdollShape::Box {
            half_extents: Vec3::new(0.04, 0.03, 0.1),
        },
        1.0,
    )
    .with_joint(
        hinge_joint(
            Vec3::X,
            (-0.5, 0.7), // Ankle range
            Vec3::new(0.0, -0.18, 0.0),
            Vec3::new(0.0, 0.03, -0.04),
        ),
        lower_leg_l_idx,
    );
    let _foot_l_idx = def.add_body(foot_l) as i32;

    // 13: Upper leg right
    let upper_leg_r = BoneBody::new(
        13,
        "upper_leg_r",
        RagdollShape::Capsule {
            radius: 0.055,
            half_height: 0.18,
        },
        8.0,
    )
    .with_joint(
        ball_joint(0.8, Vec3::new(0.1, -0.08, 0.0), Vec3::new(0.0, 0.18, 0.0)),
        pelvis_idx,
    );
    let upper_leg_r_idx = def.add_body(upper_leg_r) as i32;

    // 14: Lower leg right
    let lower_leg_r = BoneBody::new(
        14,
        "lower_leg_r",
        RagdollShape::Capsule {
            radius: 0.045,
            half_height: 0.18,
        },
        5.0,
    )
    .with_joint(
        hinge_joint(
            Vec3::X,
            (-2.5, 0.0),
            Vec3::new(0.0, -0.18, 0.0),
            Vec3::new(0.0, 0.18, 0.0),
        ),
        upper_leg_r_idx,
    );
    let lower_leg_r_idx = def.add_body(lower_leg_r) as i32;

    // 15: Foot right
    let foot_r = BoneBody::new(
        15,
        "foot_r",
        RagdollShape::Box {
            half_extents: Vec3::new(0.04, 0.03, 0.1),
        },
        1.0,
    )
    .with_joint(
        hinge_joint(
            Vec3::X,
            (-0.5, 0.7),
            Vec3::new(0.0, -0.18, 0.0),
            Vec3::new(0.0, 0.03, -0.04),
        ),
        lower_leg_r_idx,
    );
    let _foot_r_idx = def.add_body(foot_r) as i32;

    let _ = skeleton; // Used for bone_transforms in a full implementation
    def
}

// ---------------------------------------------------------------------------
// RagdollInstance — active ragdoll
// ---------------------------------------------------------------------------

/// Blend mode for a single bone in a partial ragdoll.
#[derive(Debug, Clone, Copy)]
pub enum BoneBlendMode {
    /// Fully animation-driven.
    Animation,
    /// Fully physics-driven.
    Physics,
    /// Blended between animation and physics.
    Blend(f32),
}

impl Default for BoneBlendMode {
    fn default() -> Self {
        Self::Physics
    }
}

/// Per-body runtime state in the ragdoll instance.
#[derive(Debug, Clone)]
pub struct RagdollBodyState {
    /// Current world-space position.
    pub position: Vec3,
    /// Current world-space rotation.
    pub rotation: Quat,
    /// Current velocity.
    pub velocity: Vec3,
    /// Current angular velocity.
    pub angular_velocity: Vec3,
    /// Mass of this body.
    pub mass: f32,
    /// Inverse mass.
    pub inv_mass: f32,
    /// Blend mode for this bone.
    pub blend_mode: BoneBlendMode,
    /// The body index in the definition.
    pub def_index: usize,
    /// Whether this body is active in the physics world.
    pub active: bool,
}

/// An active ragdoll instance in the physics world.
///
/// Contains runtime state for all bodies and provides methods for
/// activating/deactivating the ragdoll and reading back poses.
pub struct RagdollInstance {
    /// The ragdoll definition this instance was created from.
    pub definition: RagdollDefinition,
    /// Per-body runtime state.
    pub body_states: Vec<RagdollBodyState>,
    /// Whether the ragdoll is currently active.
    pub active: bool,
    /// Global blend factor [0, 1]. 0 = all animation, 1 = all physics.
    pub global_blend_factor: f32,
    /// Gravity vector.
    pub gravity: Vec3,
    /// Accumulated simulation time.
    sim_time: f32,
}

impl RagdollInstance {
    /// Create a new ragdoll instance from a definition.
    pub fn new(definition: RagdollDefinition) -> Self {
        let body_states: Vec<RagdollBodyState> = definition
            .bodies
            .iter()
            .enumerate()
            .map(|(i, body)| {
                let inv_mass = if body.mass > 1e-8 { 1.0 / body.mass } else { 0.0 };
                RagdollBodyState {
                    position: Vec3::ZERO,
                    rotation: Quat::IDENTITY,
                    velocity: Vec3::ZERO,
                    angular_velocity: Vec3::ZERO,
                    mass: body.mass,
                    inv_mass,
                    blend_mode: BoneBlendMode::Physics,
                    def_index: i,
                    active: false,
                }
            })
            .collect();

        Self {
            definition,
            body_states,
            active: false,
            global_blend_factor: DEFAULT_BLEND_FACTOR,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            sim_time: 0.0,
        }
    }

    /// Activate the ragdoll from the current animation pose.
    ///
    /// Sets all body positions/rotations from the given skeleton transforms
    /// and marks the ragdoll as active.
    pub fn activate(&mut self, bone_transforms: &[Transform]) {
        self.active = true;

        for (i, body_def) in self.definition.bodies.iter().enumerate() {
            if i >= self.body_states.len() {
                break;
            }
            let bone_idx = body_def.bone_index;
            if bone_idx < bone_transforms.len() {
                let bt = &bone_transforms[bone_idx];
                self.body_states[i].position = bt.position + bt.rotation * body_def.local_offset;
                self.body_states[i].rotation = bt.rotation * body_def.local_rotation;
                self.body_states[i].velocity = Vec3::ZERO;
                self.body_states[i].angular_velocity = Vec3::ZERO;
                self.body_states[i].active = true;
            }
        }
    }

    /// Activate with an initial velocity (e.g., from an impact).
    pub fn activate_with_impulse(
        &mut self,
        bone_transforms: &[Transform],
        impulse: Vec3,
        hit_point: Vec3,
    ) {
        self.activate(bone_transforms);

        // Apply impulse to all bodies, with falloff based on distance from hit point
        for state in &mut self.body_states {
            if !state.active {
                continue;
            }
            let diff = state.position - hit_point;
            let dist = diff.length().max(0.1);
            let falloff = 1.0 / (1.0 + dist * dist);
            state.velocity += impulse * falloff * state.inv_mass;

            // Add angular velocity from the impulse
            let torque_arm = diff.normalize_or_zero();
            state.angular_velocity += torque_arm.cross(impulse) * falloff * 0.5;
        }
    }

    /// Deactivate the ragdoll.
    pub fn deactivate(&mut self) {
        self.active = false;
        for state in &mut self.body_states {
            state.active = false;
            state.velocity = Vec3::ZERO;
            state.angular_velocity = Vec3::ZERO;
        }
    }

    /// Read the current physics pose as a list of bone transforms.
    ///
    /// The returned transforms are indexed by bone index from the definition.
    pub fn read_pose(&self) -> Vec<Transform> {
        self.body_states
            .iter()
            .map(|state| Transform {
                position: state.position,
                rotation: state.rotation,
            })
            .collect()
    }

    /// Read the blended pose, interpolating between animation and physics.
    ///
    /// `animation_pose` is the current animation transforms for all bones.
    pub fn read_blended_pose(&self, animation_pose: &[Transform]) -> Vec<Transform> {
        let mut result = Vec::with_capacity(self.body_states.len());

        for (i, state) in self.body_states.iter().enumerate() {
            let physics_transform = Transform {
                position: state.position,
                rotation: state.rotation,
            };

            let bone_idx = self.definition.bodies[i].bone_index;
            let anim_transform = if bone_idx < animation_pose.len() {
                animation_pose[bone_idx]
            } else {
                physics_transform
            };

            let blend = match state.blend_mode {
                BoneBlendMode::Animation => 0.0,
                BoneBlendMode::Physics => 1.0,
                BoneBlendMode::Blend(f) => f,
            };

            let effective_blend = blend * self.global_blend_factor;
            result.push(anim_transform.lerp(&physics_transform, effective_blend));
        }

        result
    }

    /// Set the blend mode for a specific bone body.
    pub fn set_bone_blend(&mut self, body_index: usize, mode: BoneBlendMode) {
        if body_index < self.body_states.len() {
            self.body_states[body_index].blend_mode = mode;
        }
    }

    /// Set all bones above the given body index to animation-driven.
    /// Useful for partial ragdoll (e.g., upper body physics, lower body animation).
    pub fn set_partial_ragdoll(&mut self, physics_from_index: usize) {
        for (i, state) in self.body_states.iter_mut().enumerate() {
            if i < physics_from_index {
                state.blend_mode = BoneBlendMode::Animation;
            } else {
                state.blend_mode = BoneBlendMode::Physics;
            }
        }
    }

    /// Simple physics step for the ragdoll (without full physics world integration).
    ///
    /// This provides basic gravity, damping, and ground collision.
    /// For full physics, bodies should be registered with the PhysicsWorld instead.
    pub fn step_simple(&mut self, dt: f32, ground_y: f32) {
        if !self.active || dt <= 0.0 {
            return;
        }
        self.sim_time += dt;

        for (i, state) in self.body_states.iter_mut().enumerate() {
            if !state.active {
                continue;
            }
            if matches!(state.blend_mode, BoneBlendMode::Animation) {
                continue;
            }

            // Apply gravity
            state.velocity += self.gravity * dt;

            // Damping
            let body_def = &self.definition.bodies[i];
            state.velocity *= 1.0 - body_def.linear_damping;
            state.angular_velocity *= 1.0 - body_def.angular_damping;

            // Integrate position
            state.position += state.velocity * dt;

            // Integrate rotation
            let omega = state.angular_velocity;
            if omega.length_squared() > 1e-12 {
                let omega_quat = Quat::from_xyzw(omega.x, omega.y, omega.z, 0.0);
                let dq = omega_quat * state.rotation * 0.5;
                state.rotation = Quat::from_xyzw(
                    state.rotation.x + dq.x * dt,
                    state.rotation.y + dq.y * dt,
                    state.rotation.z + dq.z * dt,
                    state.rotation.w + dq.w * dt,
                )
                .normalize();
            }

            // Ground collision
            let body_radius = match &body_def.shape {
                RagdollShape::Capsule { radius, .. } => *radius,
                RagdollShape::Box { half_extents } => half_extents.y,
                RagdollShape::Sphere { radius } => *radius,
            };

            if state.position.y - body_radius < ground_y {
                state.position.y = ground_y + body_radius;
                if state.velocity.y < 0.0 {
                    state.velocity.y *= -0.3; // Bounce
                    // Friction
                    state.velocity.x *= 0.8;
                    state.velocity.z *= 0.8;
                }
            }
        }

        // Simple distance constraint between connected bodies
        self.solve_distance_constraints(dt);
    }

    /// Simple distance constraints to keep bodies connected.
    fn solve_distance_constraints(&mut self, _dt: f32) {
        for _ in 0..4 {
            for i in 0..self.definition.bodies.len() {
                let parent_idx = self.definition.bodies[i].parent_index;
                if parent_idx < 0 {
                    continue;
                }
                let pi = parent_idx as usize;
                if pi >= self.body_states.len() {
                    continue;
                }

                if let Some(joint) = &self.definition.bodies[i].joint_to_parent {
                    let parent_anchor =
                        self.body_states[pi].position + self.body_states[pi].rotation * joint.parent_anchor;
                    let child_anchor =
                        self.body_states[i].position + self.body_states[i].rotation * joint.child_anchor;

                    let diff = parent_anchor - child_anchor;
                    let dist = diff.length();
                    if dist > 0.01 {
                        let correction = diff * 0.5;
                        let inv_a = self.body_states[i].inv_mass;
                        let inv_b = self.body_states[pi].inv_mass;
                        let w_sum = inv_a + inv_b;
                        if w_sum > 1e-8 {
                            self.body_states[i].position += correction * (inv_a / w_sum);
                            self.body_states[pi].position -= correction * (inv_b / w_sum);
                        }
                    }
                }
            }
        }
    }

    /// Get the center of mass of the ragdoll.
    pub fn center_of_mass(&self) -> Vec3 {
        let mut total_mass = 0.0f32;
        let mut weighted_pos = Vec3::ZERO;
        for state in &self.body_states {
            if state.active {
                weighted_pos += state.position * state.mass;
                total_mass += state.mass;
            }
        }
        if total_mass > 1e-8 {
            weighted_pos / total_mass
        } else {
            Vec3::ZERO
        }
    }

    /// Check if the ragdoll has settled (all velocities below threshold).
    pub fn is_settled(&self, velocity_threshold: f32) -> bool {
        self.body_states.iter().all(|s| {
            !s.active
                || (s.velocity.length() < velocity_threshold
                    && s.angular_velocity.length() < velocity_threshold)
        })
    }
}

// ---------------------------------------------------------------------------
// ECS integration
// ---------------------------------------------------------------------------

/// ECS component for attaching a ragdoll to an entity.
pub struct RagdollComponent {
    /// The ragdoll instance.
    pub instance: RagdollInstance,
    /// Whether to use the simple built-in physics or defer to PhysicsWorld.
    pub use_simple_physics: bool,
    /// Ground height for simple physics.
    pub ground_y: f32,
}

impl RagdollComponent {
    /// Create a new ragdoll component from a definition.
    pub fn new(definition: RagdollDefinition) -> Self {
        Self {
            instance: RagdollInstance::new(definition),
            use_simple_physics: true,
            ground_y: 0.0,
        }
    }

    /// Create a humanoid ragdoll component.
    pub fn humanoid() -> Self {
        let skeleton = SkeletonInfo::new(16);
        let def = create_humanoid_ragdoll(&skeleton);
        Self::new(def)
    }
}

/// System that steps all ragdoll simulations.
pub struct RagdollSystem {
    /// Fixed time step.
    pub fixed_timestep: f32,
    /// Accumulated time.
    time_accumulator: f32,
}

impl Default for RagdollSystem {
    fn default() -> Self {
        Self {
            fixed_timestep: 1.0 / 60.0,
            time_accumulator: 0.0,
        }
    }
}

impl RagdollSystem {
    /// Create a new ragdoll system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update all ragdolls.
    pub fn update(&mut self, dt: f32, ragdolls: &mut [RagdollComponent]) {
        self.time_accumulator += dt;
        let mut steps = 0u32;

        while self.time_accumulator >= self.fixed_timestep && steps < 4 {
            for ragdoll in ragdolls.iter_mut() {
                if ragdoll.instance.active && ragdoll.use_simple_physics {
                    ragdoll
                        .instance
                        .step_simple(self.fixed_timestep, ragdoll.ground_y);
                }
            }
            self.time_accumulator -= self.fixed_timestep;
            steps += 1;
        }

        if self.time_accumulator > self.fixed_timestep {
            self.time_accumulator = 0.0;
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_humanoid_ragdoll_creation() {
        let skeleton = SkeletonInfo::new(16);
        let def = create_humanoid_ragdoll(&skeleton);

        assert_eq!(def.body_count(), 16);
        assert!(def.total_mass() > 50.0); // Should be around 70kg
    }

    #[test]
    fn test_ragdoll_instance_creation() {
        let skeleton = SkeletonInfo::new(16);
        let def = create_humanoid_ragdoll(&skeleton);
        let instance = RagdollInstance::new(def);

        assert!(!instance.active);
        assert_eq!(instance.body_states.len(), 16);
    }

    #[test]
    fn test_ragdoll_activation() {
        let skeleton = SkeletonInfo::new(16);
        let def = create_humanoid_ragdoll(&skeleton);
        let mut instance = RagdollInstance::new(def);

        let pose: Vec<Transform> = (0..16)
            .map(|i| Transform::new(Vec3::new(0.0, 1.0 + i as f32 * 0.1, 0.0), Quat::IDENTITY))
            .collect();

        instance.activate(&pose);
        assert!(instance.active);

        // Body states should have positions from the pose
        for state in &instance.body_states {
            assert!(state.active);
            assert!(state.position.y > 0.0);
        }
    }

    #[test]
    fn test_ragdoll_deactivation() {
        let skeleton = SkeletonInfo::new(16);
        let def = create_humanoid_ragdoll(&skeleton);
        let mut instance = RagdollInstance::new(def);

        let pose: Vec<Transform> = vec![Transform::default(); 16];
        instance.activate(&pose);
        instance.deactivate();

        assert!(!instance.active);
    }

    #[test]
    fn test_ragdoll_falls_under_gravity() {
        let skeleton = SkeletonInfo::new(16);
        let def = create_humanoid_ragdoll(&skeleton);
        let mut instance = RagdollInstance::new(def);

        let pose: Vec<Transform> = (0..16)
            .map(|_| Transform::new(Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY))
            .collect();

        instance.activate(&pose);

        let initial_com = instance.center_of_mass();

        for _ in 0..60 {
            instance.step_simple(1.0 / 60.0, 0.0);
        }

        let final_com = instance.center_of_mass();
        assert!(
            final_com.y < initial_com.y,
            "Ragdoll should fall: {} -> {}",
            initial_com.y,
            final_com.y
        );
    }

    #[test]
    fn test_ragdoll_ground_collision() {
        let skeleton = SkeletonInfo::new(16);
        let def = create_humanoid_ragdoll(&skeleton);
        let mut instance = RagdollInstance::new(def);

        let pose: Vec<Transform> = (0..16)
            .map(|_| Transform::new(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY))
            .collect();

        instance.activate(&pose);

        // Simulate long enough for the ragdoll to settle on the ground
        for _ in 0..600 {
            instance.step_simple(1.0 / 60.0, 0.0);
        }

        // All bodies should be above ground (with generous tolerance for bouncing)
        for (i, state) in instance.body_states.iter().enumerate() {
            assert!(
                state.position.y >= -0.5,
                "Body {} below ground: y = {}",
                i,
                state.position.y
            );
        }
    }

    #[test]
    fn test_ragdoll_blending() {
        let skeleton = SkeletonInfo::new(16);
        let def = create_humanoid_ragdoll(&skeleton);
        let mut instance = RagdollInstance::new(def);

        let pose: Vec<Transform> = (0..16)
            .map(|_| Transform::new(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY))
            .collect();
        instance.activate(&pose);

        // Physics pose (moved)
        instance.body_states[0].position = Vec3::new(1.0, 1.0, 0.0);

        // Animation pose (original)
        let anim_pose: Vec<Transform> = (0..16)
            .map(|_| Transform::new(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY))
            .collect();

        // Full physics blend
        instance.global_blend_factor = 1.0;
        let blended = instance.read_blended_pose(&anim_pose);
        assert!((blended[0].position.x - 1.0).abs() < 0.01);

        // Half blend
        instance.global_blend_factor = 0.5;
        let blended = instance.read_blended_pose(&anim_pose);
        assert!((blended[0].position.x - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_partial_ragdoll() {
        let skeleton = SkeletonInfo::new(16);
        let def = create_humanoid_ragdoll(&skeleton);
        let mut instance = RagdollInstance::new(def);

        // Upper body physics (from index 4), lower body animation
        instance.set_partial_ragdoll(4);

        for i in 0..4 {
            assert!(matches!(
                instance.body_states[i].blend_mode,
                BoneBlendMode::Animation
            ));
        }
        for i in 4..16 {
            assert!(matches!(
                instance.body_states[i].blend_mode,
                BoneBlendMode::Physics
            ));
        }
    }

    #[test]
    fn test_ragdoll_is_settled() {
        let skeleton = SkeletonInfo::new(16);
        let mut def = create_humanoid_ragdoll(&skeleton);

        // Increase damping for faster settling in test
        for body in &mut def.bodies {
            body.linear_damping = 0.15;
            body.angular_damping = 0.3;
        }

        let mut instance = RagdollInstance::new(def);

        // Start above ground, let it fall and settle
        let pose: Vec<Transform> = (0..16)
            .map(|_| Transform::new(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY))
            .collect();
        instance.activate(&pose);

        // Initially not settled (due to gravity velocity)
        instance.step_simple(1.0 / 60.0, 0.0);
        assert!(!instance.is_settled(0.01));

        // After settling on ground with damping
        for _ in 0..600 {
            instance.step_simple(1.0 / 60.0, 0.0);
        }
        // With ground collision and high damping, velocities should be low
        assert!(instance.is_settled(2.0));
    }

    #[test]
    fn test_ragdoll_impulse() {
        let skeleton = SkeletonInfo::new(16);
        let def = create_humanoid_ragdoll(&skeleton);
        let mut instance = RagdollInstance::new(def);

        let pose: Vec<Transform> = (0..16)
            .map(|_| Transform::new(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY))
            .collect();

        let impulse = Vec3::new(10.0, 5.0, 0.0);
        instance.activate_with_impulse(&pose, impulse, Vec3::new(0.0, 1.0, 0.0));

        // Bodies should have velocity from the impulse
        let has_velocity = instance
            .body_states
            .iter()
            .any(|s| s.velocity.length() > 0.1);
        assert!(has_velocity, "Impulse should impart velocity");
    }

    #[test]
    fn test_ragdoll_component() {
        let component = RagdollComponent::humanoid();
        assert_eq!(component.instance.body_states.len(), 16);
    }

    #[test]
    fn test_ragdoll_system() {
        let mut system = RagdollSystem::new();
        let mut ragdolls = vec![RagdollComponent::humanoid()];

        let pose: Vec<Transform> = (0..16)
            .map(|_| Transform::new(Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY))
            .collect();
        ragdolls[0].instance.activate(&pose);

        system.update(1.0 / 60.0, &mut ragdolls);
        // Should not panic
    }

    #[test]
    fn test_transform_lerp() {
        let a = Transform::new(Vec3::ZERO, Quat::IDENTITY);
        let b = Transform::new(Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY);

        let mid = a.lerp(&b, 0.5);
        assert!((mid.position.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_ragdoll_shape_volume() {
        let capsule = RagdollShape::Capsule {
            radius: 0.05,
            half_height: 0.15,
        };
        assert!(capsule.volume() > 0.0);

        let sphere = RagdollShape::Sphere { radius: 0.1 };
        assert!((sphere.volume() - (4.0 / 3.0) * std::f32::consts::PI * 0.001).abs() < 1e-4);
    }
}
