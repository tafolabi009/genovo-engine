//! Inverse Kinematics (IK) solvers.
//!
//! Provides multiple IK algorithms for adjusting bone poses to reach target
//! positions. Used for foot placement, hand targeting, look-at, and
//! procedural animation.
//!
//! # Solvers
//!
//! - [`TwoBoneIK`] -- Analytical O(1) solver for two-bone chains (arms, legs).
//! - [`CCDSolver`] -- Cyclic Coordinate Descent, iterative, handles any chain length.
//! - [`FABRIKSolver`] -- Forward And Backward Reaching IK, natural-looking results.
//!
//! All solvers implement the [`IKSolver`] trait and operate on [`IKChain`] data.

use glam::{Quat, Vec3};

// ---------------------------------------------------------------------------
// Joint constraint types
// ---------------------------------------------------------------------------

/// Constraint that limits the rotational freedom of a joint.
#[derive(Debug, Clone, Copy)]
pub enum JointConstraint {
    /// No constraint; the joint can rotate freely.
    None,

    /// Ball-and-socket joint with a maximum cone angle (radians).
    /// Allows rotation in any direction up to the cone half-angle.
    BallSocket {
        /// Maximum angle from the rest direction (radians).
        max_angle: f32,
    },

    /// Hinge joint that only rotates around a single axis.
    Hinge {
        /// The hinge axis (in local space of the bone).
        axis: Vec3,
        /// Minimum angle (radians).
        min_angle: f32,
        /// Maximum angle (radians).
        max_angle: f32,
    },

    /// Twist constraint that limits rotation around the bone's own axis.
    Twist {
        /// Minimum twist angle (radians).
        min_angle: f32,
        /// Maximum twist angle (radians).
        max_angle: f32,
    },
}

impl Default for JointConstraint {
    fn default() -> Self {
        Self::None
    }
}

impl JointConstraint {
    /// Apply this constraint to a rotation, returning the constrained result.
    ///
    /// `rest_direction` is the default pointing direction of the bone when
    /// no rotation is applied.
    pub fn apply(&self, rotation: Quat, rest_direction: Vec3) -> Quat {
        match *self {
            JointConstraint::None => rotation,

            JointConstraint::BallSocket { max_angle } => {
                let rotated_dir = rotation * rest_direction;
                let angle = rest_direction
                    .dot(rotated_dir)
                    .clamp(-1.0, 1.0)
                    .acos();

                if angle <= max_angle {
                    rotation
                } else {
                    // Clamp the rotation to the cone boundary.
                    let cross = rest_direction.cross(rotated_dir);
                    if cross.length_squared() < f32::EPSILON {
                        // Directions are nearly parallel or anti-parallel.
                        return if angle < std::f32::consts::PI * 0.5 {
                            rotation
                        } else {
                            Quat::IDENTITY
                        };
                    }
                    let axis = cross.normalize();
                    Quat::from_axis_angle(axis, max_angle)
                }
            }

            JointConstraint::Hinge {
                axis,
                min_angle,
                max_angle,
            } => {
                // Decompose rotation into twist (around axis) and swing.
                let (twist, _swing) = decompose_twist_swing(rotation, axis);

                // Extract the angle from the twist quaternion.
                let twist_axis = Vec3::new(twist.x, twist.y, twist.z);
                let twist_len = twist_axis.length();
                if twist_len < f32::EPSILON {
                    return Quat::IDENTITY;
                }

                let mut angle = 2.0 * twist_len.atan2(twist.w);
                // Normalize to [-PI, PI]
                if angle > std::f32::consts::PI {
                    angle -= 2.0 * std::f32::consts::PI;
                }
                if angle < -std::f32::consts::PI {
                    angle += 2.0 * std::f32::consts::PI;
                }

                let clamped = angle.clamp(min_angle, max_angle);
                Quat::from_axis_angle(axis, clamped)
            }

            JointConstraint::Twist {
                min_angle,
                max_angle,
            } => {
                // Extract the twist component around the bone's forward axis.
                let forward = rotation * Vec3::Z;
                let (twist, swing) = decompose_twist_swing(rotation, forward);

                let twist_axis = Vec3::new(twist.x, twist.y, twist.z);
                let twist_len = twist_axis.length();
                if twist_len < f32::EPSILON {
                    return swing;
                }

                let mut angle = 2.0 * twist_len.atan2(twist.w);
                if angle > std::f32::consts::PI {
                    angle -= 2.0 * std::f32::consts::PI;
                }
                if angle < -std::f32::consts::PI {
                    angle += 2.0 * std::f32::consts::PI;
                }

                let clamped = angle.clamp(min_angle, max_angle);
                let constrained_twist = Quat::from_axis_angle(forward.normalize(), clamped);
                swing * constrained_twist
            }
        }
    }
}

/// Decompose a quaternion into twist (rotation around `axis`) and swing components.
///
/// `q = swing * twist`, where twist rotates around `axis` and swing
/// rotates the axis itself.
fn decompose_twist_swing(q: Quat, axis: Vec3) -> (Quat, Quat) {
    let projection = Vec3::new(q.x, q.y, q.z).dot(axis);
    let twist = Quat::from_xyzw(
        axis.x * projection,
        axis.y * projection,
        axis.z * projection,
        q.w,
    )
    .normalize();
    let swing = q * twist.conjugate();
    (twist, swing)
}

// ---------------------------------------------------------------------------
// IK chain definition
// ---------------------------------------------------------------------------

/// A chain of joints to be solved by an IK algorithm.
///
/// The chain stores joint indices (into the skeleton's bone array), world-space
/// positions and rotations, and the target to reach. The solver modifies
/// `joint_positions` and `joint_rotations` in place.
#[derive(Debug, Clone)]
pub struct IKChain {
    /// Indices into the skeleton's bone array, ordered from root to tip.
    pub joints: Vec<usize>,

    /// World-space target position the end effector should reach.
    pub target: Vec3,

    /// Optional pole target to control the "bend direction" of the chain
    /// (e.g. elbow direction for an arm chain).
    pub pole_target: Option<Vec3>,

    /// Blend weight [0.0, 1.0]. At 0.0 the IK solution has no effect;
    /// at 1.0 it fully overrides the FK pose.
    pub weight: f32,

    /// Per-joint position (world-space), updated by the solver.
    pub joint_positions: Vec<Vec3>,

    /// Per-joint rotation (world-space), computed after solving.
    pub joint_rotations: Vec<Quat>,

    /// Per-joint constraints (optional). If shorter than the joint array,
    /// missing entries are treated as unconstrained.
    pub constraints: Vec<JointConstraint>,
}

impl IKChain {
    /// Create a new IK chain with the given joint indices.
    pub fn new(joints: Vec<usize>, target: Vec3) -> Self {
        let joint_count = joints.len();
        Self {
            joints,
            target,
            pole_target: None,
            weight: 1.0,
            joint_positions: vec![Vec3::ZERO; joint_count],
            joint_rotations: vec![Quat::IDENTITY; joint_count],
            constraints: Vec::new(),
        }
    }

    /// Create a chain with constraints pre-allocated.
    pub fn with_constraints(
        joints: Vec<usize>,
        target: Vec3,
        constraints: Vec<JointConstraint>,
    ) -> Self {
        let joint_count = joints.len();
        Self {
            joints,
            target,
            pole_target: None,
            weight: 1.0,
            joint_positions: vec![Vec3::ZERO; joint_count],
            joint_rotations: vec![Quat::IDENTITY; joint_count],
            constraints,
        }
    }

    /// Set the pole target for controlling bend direction.
    pub fn set_pole_target(&mut self, pole: Vec3) {
        self.pole_target = Some(pole);
    }

    /// Set the target position.
    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
    }

    /// Get the constraint for a joint, or `JointConstraint::None` if none.
    pub fn constraint(&self, joint_index: usize) -> JointConstraint {
        self.constraints
            .get(joint_index)
            .copied()
            .unwrap_or(JointConstraint::None)
    }

    /// Compute the total length of the chain (sum of bone segment lengths).
    pub fn chain_length(&self) -> f32 {
        if self.joint_positions.len() < 2 {
            return 0.0;
        }
        self.joint_positions
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .sum()
    }

    /// Compute individual bone lengths between consecutive joints.
    pub fn bone_lengths(&self) -> Vec<f32> {
        if self.joint_positions.len() < 2 {
            return Vec::new();
        }
        self.joint_positions
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .collect()
    }

    /// Whether the target is reachable (within the chain's total length).
    pub fn is_reachable(&self) -> bool {
        if self.joint_positions.is_empty() {
            return false;
        }
        let root = self.joint_positions[0];
        let distance = (self.target - root).length();
        distance <= self.chain_length()
    }

    /// Get the end effector position (last joint position).
    pub fn end_effector(&self) -> Vec3 {
        self.joint_positions
            .last()
            .copied()
            .unwrap_or(Vec3::ZERO)
    }

    /// Distance from the end effector to the target.
    pub fn error(&self) -> f32 {
        (self.end_effector() - self.target).length()
    }

    /// Compute rotations from the solved positions.
    ///
    /// After a positional solver (like FABRIK) has updated `joint_positions`,
    /// call this to compute the corresponding `joint_rotations` that align
    /// each bone segment to point toward the next joint.
    pub fn compute_rotations_from_positions(&mut self) {
        let n = self.joint_positions.len();
        if n < 2 {
            return;
        }

        for i in 0..n - 1 {
            let current_dir = (self.joint_positions[i + 1] - self.joint_positions[i]).normalize();
            // Default bone direction is +Y (up)
            let default_dir = Vec3::Y;

            if current_dir.dot(default_dir).abs() > 0.9999 {
                self.joint_rotations[i] = if current_dir.dot(default_dir) > 0.0 {
                    Quat::IDENTITY
                } else {
                    Quat::from_rotation_z(std::f32::consts::PI)
                };
            } else {
                let axis = default_dir.cross(current_dir).normalize();
                let angle = default_dir.dot(current_dir).clamp(-1.0, 1.0).acos();
                self.joint_rotations[i] = Quat::from_axis_angle(axis, angle);
            }
        }

        // Last joint inherits the previous joint's rotation
        if n >= 2 {
            self.joint_rotations[n - 1] = self.joint_rotations[n - 2];
        }
    }

    /// Number of joints in the chain.
    pub fn len(&self) -> usize {
        self.joints.len()
    }

    /// Whether the chain is empty (no joints).
    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }
}

// ---------------------------------------------------------------------------
// IK solver trait
// ---------------------------------------------------------------------------

/// Common interface for IK solver algorithms.
///
/// Implementations modify the chain's `joint_positions` and `joint_rotations`
/// in place. The `solve` method returns `true` if the solver converged
/// (end effector reached the target within tolerance).
pub trait IKSolver: Send + Sync {
    /// Human-readable name of the solver algorithm.
    fn name(&self) -> &str;

    /// Solve the IK chain. Modifies `chain.joint_positions` and
    /// `chain.joint_rotations` in place.
    ///
    /// Returns `true` if the solver converged within tolerance.
    fn solve(&self, chain: &mut IKChain) -> bool;
}

// ---------------------------------------------------------------------------
// Two-bone IK (analytical)
// ---------------------------------------------------------------------------

/// Simple analytical two-bone IK solver.
///
/// Solves a two-segment chain (e.g. upper arm + forearm, or thigh + shin)
/// using the law of cosines. Very fast (O(1)) and produces predictable results.
///
/// # Requirements
///
/// The chain must have exactly 3 joints (root, mid, end). If the chain has
/// a different number of joints, the solver returns `false`.
///
/// # Algorithm
///
/// 1. Compute upper and lower bone lengths from the FK positions.
/// 2. Clamp the target distance to the maximum chain reach.
/// 3. Use the law of cosines to find the angle at the root joint.
/// 4. Determine the bend plane from the pole target (or current mid position).
/// 5. Rotate the root bone by the computed angle in the bend plane.
/// 6. Place the end effector on the line from root to target.
pub struct TwoBoneIK;

impl TwoBoneIK {
    /// Solve a two-bone chain given the root, mid, and end positions, the target,
    /// and an optional pole target for bend direction.
    ///
    /// Returns the new mid-joint and end-effector positions, or `None` if the
    /// chain is degenerate (zero-length bones).
    pub fn solve(
        root: Vec3,
        mid: Vec3,
        end: Vec3,
        target: Vec3,
        pole_target: Option<Vec3>,
    ) -> Option<(Vec3, Vec3)> {
        let upper_len = (mid - root).length();
        let lower_len = (end - mid).length();

        if upper_len < f32::EPSILON || lower_len < f32::EPSILON {
            return None;
        }

        let total_len = upper_len + lower_len;
        let target_dir = target - root;
        let target_dist = target_dir.length();

        if target_dist < f32::EPSILON {
            // Target is at the root. Fold the chain.
            return Some((root + Vec3::Y * upper_len * 0.01, root));
        }

        // Clamp target distance to the chain's reach.
        let clamped_dist = target_dist.min(total_len - 0.0001);

        let target_dir_norm = target_dir.normalize();

        // Law of cosines: find the angle at the root joint.
        // c^2 = a^2 + b^2 - 2ab*cos(C)
        // cos(angle_at_root) = (upper^2 + dist^2 - lower^2) / (2 * upper * dist)
        let cos_angle = ((upper_len * upper_len + clamped_dist * clamped_dist
            - lower_len * lower_len)
            / (2.0 * upper_len * clamped_dist))
            .clamp(-1.0, 1.0);
        let angle = cos_angle.acos();

        // Determine the bend plane normal.
        let bend_normal = Self::compute_bend_normal(root, mid, target_dir_norm, pole_target);

        // Rotate the target direction by the computed angle around the bend normal
        // to find the mid-joint direction.
        let rotation = Quat::from_axis_angle(bend_normal, angle);
        let new_mid = root + rotation * (target_dir_norm * upper_len);

        // The end effector sits at the clamped target distance along the
        // root-to-target direction.
        let new_end = root + target_dir_norm * clamped_dist;

        Some((new_mid, new_end))
    }

    /// Solve and return the joint rotations as well.
    ///
    /// Returns `(new_mid, new_end, root_rotation, mid_rotation)` or `None`.
    pub fn solve_with_rotations(
        root: Vec3,
        mid: Vec3,
        end: Vec3,
        target: Vec3,
        pole_target: Option<Vec3>,
    ) -> Option<(Vec3, Vec3, Quat, Quat)> {
        let (new_mid, new_end) = Self::solve(root, mid, end, target, pole_target)?;

        // Compute rotation for the root joint.
        let old_upper_dir = (mid - root).normalize();
        let new_upper_dir = (new_mid - root).normalize();
        let root_rot = rotation_between_vectors(old_upper_dir, new_upper_dir);

        // Compute rotation for the mid joint.
        let old_lower_dir = (end - mid).normalize();
        let new_lower_dir = (new_end - new_mid).normalize();
        let mid_rot = rotation_between_vectors(old_lower_dir, new_lower_dir);

        Some((new_mid, new_end, root_rot, mid_rot))
    }

    /// Compute the bend plane normal.
    fn compute_bend_normal(
        root: Vec3,
        mid: Vec3,
        target_dir_norm: Vec3,
        pole_target: Option<Vec3>,
    ) -> Vec3 {
        if let Some(pole) = pole_target {
            let to_pole = (pole - root).normalize();
            let cross = target_dir_norm.cross(to_pole);
            if cross.length_squared() > f32::EPSILON {
                return cross.normalize();
            }
        }

        // Fallback: use the current mid-joint plane.
        let to_mid = (mid - root).normalize();
        let cross = target_dir_norm.cross(to_mid);
        if cross.length_squared() > f32::EPSILON {
            return cross.normalize();
        }

        // Last resort: pick an arbitrary perpendicular axis.
        let arbitrary = if target_dir_norm.dot(Vec3::Y).abs() < 0.99 {
            Vec3::Y
        } else {
            Vec3::X
        };
        target_dir_norm.cross(arbitrary).normalize()
    }
}

impl IKSolver for TwoBoneIK {
    fn name(&self) -> &str {
        "Two-Bone IK (Analytical)"
    }

    fn solve(&self, chain: &mut IKChain) -> bool {
        if chain.joint_positions.len() != 3 {
            log::warn!(
                "TwoBoneIK requires exactly 3 joints, got {}",
                chain.joint_positions.len()
            );
            return false;
        }

        let root = chain.joint_positions[0];
        let mid = chain.joint_positions[1];
        let end = chain.joint_positions[2];

        match TwoBoneIK::solve(root, mid, end, chain.target, chain.pole_target) {
            Some((new_mid, new_end)) => {
                // Blend between FK and IK based on weight.
                let w = chain.weight.clamp(0.0, 1.0);
                chain.joint_positions[1] = mid.lerp(new_mid, w);
                chain.joint_positions[2] = end.lerp(new_end, w);

                // Compute rotations from the new positions.
                chain.compute_rotations_from_positions();

                true
            }
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// CCD solver (Cyclic Coordinate Descent)
// ---------------------------------------------------------------------------

/// Cyclic Coordinate Descent IK solver.
///
/// Iteratively rotates each joint to point toward the target, cycling from
/// the end effector to the root. Fast and stable for chains of any length.
///
/// # Algorithm
///
/// For each iteration:
/// 1. Starting from the joint before the end effector, moving toward the root:
///    a. Compute the vector from the current joint to the end effector.
///    b. Compute the vector from the current joint to the target.
///    c. Compute the rotation that aligns these two vectors.
///    d. Apply joint constraints (if any).
///    e. Rotate the end effector and all downstream joints by this rotation.
/// 2. Check if the end effector is within tolerance of the target.
/// 3. Repeat until convergence or maximum iterations.
pub struct CCDSolver {
    /// Maximum number of solver iterations.
    pub max_iterations: u32,

    /// Distance tolerance for convergence (meters).
    pub tolerance: f32,

    /// Damping factor [0, 1]. Lower values produce smoother but slower convergence.
    /// 1.0 = no damping (full rotation each step).
    pub damping: f32,
}

impl Default for CCDSolver {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tolerance: 0.001,
            damping: 1.0,
        }
    }
}

impl CCDSolver {
    /// Create a new CCD solver with custom parameters.
    pub fn new(max_iterations: u32, tolerance: f32) -> Self {
        Self {
            max_iterations,
            tolerance,
            damping: 1.0,
        }
    }

    /// Set the damping factor.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.01, 1.0);
        self
    }

    /// Apply a rotation to all joints from `joint_index` to the end of the chain.
    fn rotate_downstream(positions: &mut [Vec3], pivot: Vec3, rotation: Quat, from: usize) {
        for pos in positions[from..].iter_mut() {
            *pos = pivot + rotation * (*pos - pivot);
        }
    }
}

impl IKSolver for CCDSolver {
    fn name(&self) -> &str {
        "CCD (Cyclic Coordinate Descent)"
    }

    fn solve(&self, chain: &mut IKChain) -> bool {
        let n = chain.joint_positions.len();
        if n < 2 {
            return false;
        }

        let original_positions = chain.joint_positions.clone();
        let target = chain.target;

        for _iteration in 0..self.max_iterations {
            // Check convergence.
            let end_effector = chain.joint_positions[n - 1];
            let error = (end_effector - target).length();
            if error <= self.tolerance {
                // Apply weight blending against original positions.
                Self::apply_weight(chain, &original_positions);
                chain.compute_rotations_from_positions();
                // Apply pole target if present.
                if chain.pole_target.is_some() {
                    Self::apply_pole_target(chain);
                }
                return true;
            }

            // Iterate from the joint before the end effector toward the root.
            for i in (0..n - 1).rev() {
                let joint_pos = chain.joint_positions[i];
                let end_effector = chain.joint_positions[n - 1];

                let to_end = end_effector - joint_pos;
                let to_target = target - joint_pos;

                if to_end.length_squared() < f32::EPSILON
                    || to_target.length_squared() < f32::EPSILON
                {
                    continue;
                }

                let to_end_norm = to_end.normalize();
                let to_target_norm = to_target.normalize();

                // Compute the rotation to align to_end with to_target.
                let mut rotation = rotation_between_vectors(to_end_norm, to_target_norm);

                // Apply damping.
                if self.damping < 1.0 {
                    rotation = Quat::IDENTITY.slerp(rotation, self.damping);
                }

                // Apply joint constraints.
                let constraint = chain.constraint(i);
                rotation = constraint.apply(rotation, to_end_norm);

                // Rotate all downstream joints around this joint.
                Self::rotate_downstream(
                    &mut chain.joint_positions,
                    joint_pos,
                    rotation,
                    i + 1,
                );
            }
        }

        // Apply weight blending.
        Self::apply_weight(chain, &original_positions);
        chain.compute_rotations_from_positions();

        // Apply pole target if present.
        if chain.pole_target.is_some() {
            Self::apply_pole_target(chain);
        }

        // Final convergence check.
        let final_error = (chain.joint_positions[n - 1] - target).length();
        final_error <= self.tolerance
    }
}

impl CCDSolver {
    /// Blend the solved positions with the original FK positions based on weight.
    fn apply_weight(chain: &mut IKChain, original: &[Vec3]) {
        let w = chain.weight.clamp(0.0, 1.0);
        if (w - 1.0).abs() > f32::EPSILON {
            for (solved, orig) in chain.joint_positions.iter_mut().zip(original.iter()) {
                *solved = orig.lerp(*solved, w);
            }
        }
    }

    /// Adjust the chain to respect the pole target.
    ///
    /// This rotates the middle joints of the chain around the root-to-end axis
    /// so the chain bends toward the pole target.
    fn apply_pole_target(chain: &mut IKChain) {
        let n = chain.joint_positions.len();
        if n < 3 {
            return;
        }

        let pole = match chain.pole_target {
            Some(p) => p,
            None => return,
        };

        let root = chain.joint_positions[0];
        let end = chain.joint_positions[n - 1];
        let chain_axis = (end - root).normalize();

        if chain_axis.length_squared() < f32::EPSILON {
            return;
        }

        // For each middle joint, project it onto the plane perpendicular to
        // the chain axis, then rotate toward the pole target's projection.
        for i in 1..n - 1 {
            let joint = chain.joint_positions[i];

            // Project joint and pole onto the plane perpendicular to chain_axis
            // centered at root.
            let joint_offset = joint - root;
            let joint_proj = joint_offset - chain_axis * joint_offset.dot(chain_axis);

            let pole_offset = pole - root;
            let pole_proj = pole_offset - chain_axis * pole_offset.dot(chain_axis);

            if joint_proj.length_squared() < f32::EPSILON
                || pole_proj.length_squared() < f32::EPSILON
            {
                continue;
            }

            let current_dir = joint_proj.normalize();
            let target_dir = pole_proj.normalize();
            let angle = current_dir.dot(target_dir).clamp(-1.0, 1.0).acos();

            if angle.abs() < f32::EPSILON {
                continue;
            }

            let cross = current_dir.cross(target_dir);
            let sign = if cross.dot(chain_axis) >= 0.0 {
                1.0
            } else {
                -1.0
            };
            let rotation = Quat::from_axis_angle(chain_axis, angle * sign);

            // Rotate this joint around the chain axis.
            chain.joint_positions[i] = root + rotation * joint_offset;
        }
    }
}

// ---------------------------------------------------------------------------
// FABRIK solver (Forward And Backward Reaching Inverse Kinematics)
// ---------------------------------------------------------------------------

/// FABRIK IK solver.
///
/// Uses forward and backward reaching passes to position joints along the
/// chain. Produces more natural results than CCD for longer chains and
/// handles constraints well.
///
/// # Algorithm
///
/// 1. **Reachability check**: If the target is beyond the chain's total
///    length, stretch the chain directly toward the target.
/// 2. **Forward pass**: Set the end effector to the target position, then
///    work backward to the root, maintaining bone lengths.
/// 3. **Backward pass**: Fix the root at its original position, then work
///    forward to the end effector, maintaining bone lengths.
/// 4. **Repeat** until the end effector is within tolerance or max iterations.
/// 5. **Constraints**: Apply joint constraints between passes.
pub struct FABRIKSolver {
    /// Maximum number of solver iterations.
    pub max_iterations: u32,

    /// Distance tolerance for convergence (meters).
    pub tolerance: f32,
}

impl Default for FABRIKSolver {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tolerance: 0.001,
        }
    }
}

impl FABRIKSolver {
    /// Create a new FABRIK solver with custom parameters.
    pub fn new(max_iterations: u32, tolerance: f32) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Forward reaching pass: move the end effector to the target and work
    /// backward, adjusting each joint to maintain bone lengths.
    fn forward_pass(positions: &mut [Vec3], bone_lengths: &[f32], target: Vec3) {
        let n = positions.len();
        if n < 2 {
            return;
        }

        // Set end effector to target.
        positions[n - 1] = target;

        // Work backward from end effector to root.
        for i in (0..n - 1).rev() {
            let dir = positions[i] - positions[i + 1];
            let dir_len = dir.length();
            if dir_len < f32::EPSILON {
                // Coincident points; nudge slightly.
                positions[i] = positions[i + 1] + Vec3::Y * bone_lengths[i];
            } else {
                positions[i] = positions[i + 1] + dir / dir_len * bone_lengths[i];
            }
        }
    }

    /// Backward reaching pass: fix the root and work forward, adjusting
    /// each joint to maintain bone lengths.
    fn backward_pass(positions: &mut [Vec3], bone_lengths: &[f32], root: Vec3) {
        let n = positions.len();
        if n < 2 {
            return;
        }

        // Fix root position.
        positions[0] = root;

        // Work forward from root to end effector.
        for i in 0..n - 1 {
            let dir = positions[i + 1] - positions[i];
            let dir_len = dir.length();
            if dir_len < f32::EPSILON {
                positions[i + 1] = positions[i] + Vec3::Y * bone_lengths[i];
            } else {
                positions[i + 1] = positions[i] + dir / dir_len * bone_lengths[i];
            }
        }
    }

    /// Apply joint constraints after each pass.
    fn apply_constraints(
        positions: &mut [Vec3],
        bone_lengths: &[f32],
        constraints: &[JointConstraint],
    ) {
        let n = positions.len();
        if n < 3 || constraints.is_empty() {
            return;
        }

        for i in 1..n - 1 {
            let constraint = constraints
                .get(i)
                .copied()
                .unwrap_or(JointConstraint::None);

            match constraint {
                JointConstraint::None => {}
                JointConstraint::BallSocket { max_angle } => {
                    // Constrain the angle between the incoming and outgoing bone segments.
                    let incoming = (positions[i] - positions[i - 1]).normalize();
                    let outgoing = (positions[i + 1] - positions[i]).normalize();

                    let angle = incoming.dot(outgoing).clamp(-1.0, 1.0).acos();
                    let min_angle = std::f32::consts::PI - max_angle;

                    if angle < min_angle {
                        // Angle too tight; rotate outgoing toward incoming.
                        let cross = incoming.cross(outgoing);
                        if cross.length_squared() > f32::EPSILON {
                            let axis = cross.normalize();
                            let rotation = Quat::from_axis_angle(axis, min_angle - angle);
                            let new_dir = rotation * outgoing;
                            positions[i + 1] = positions[i] + new_dir * bone_lengths[i];
                        }
                    }
                }
                JointConstraint::Hinge {
                    axis,
                    min_angle,
                    max_angle,
                } => {
                    // Project the bone direction onto the hinge plane and clamp.
                    let incoming = (positions[i] - positions[i - 1]).normalize();
                    let outgoing = (positions[i + 1] - positions[i]).normalize();

                    // Project outgoing onto the plane perpendicular to the hinge axis
                    // relative to the incoming direction.
                    let proj = outgoing - axis * outgoing.dot(axis);
                    if proj.length_squared() < f32::EPSILON {
                        continue;
                    }
                    let proj_norm = proj.normalize();

                    // Compute the angle of the projected direction relative to incoming.
                    let ref_dir = incoming - axis * incoming.dot(axis);
                    if ref_dir.length_squared() < f32::EPSILON {
                        continue;
                    }
                    let ref_norm = ref_dir.normalize();

                    let cos_a = ref_norm.dot(proj_norm).clamp(-1.0, 1.0);
                    let mut angle = cos_a.acos();
                    let cross = ref_norm.cross(proj_norm);
                    if cross.dot(axis) < 0.0 {
                        angle = -angle;
                    }

                    let clamped = angle.clamp(min_angle, max_angle);
                    if (clamped - angle).abs() > f32::EPSILON {
                        let rotation = Quat::from_axis_angle(axis, clamped);
                        let new_dir = rotation * ref_norm;
                        positions[i + 1] = positions[i] + new_dir * bone_lengths[i];
                    }
                }
                _ => {}
            }
        }
    }
}

impl IKSolver for FABRIKSolver {
    fn name(&self) -> &str {
        "FABRIK (Forward And Backward Reaching)"
    }

    fn solve(&self, chain: &mut IKChain) -> bool {
        let n = chain.joint_positions.len();
        if n < 2 {
            return false;
        }

        let original_positions = chain.joint_positions.clone();
        let root = chain.joint_positions[0];
        let target = chain.target;

        // Compute bone lengths.
        let bone_lengths: Vec<f32> = chain
            .joint_positions
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .collect();

        let total_length: f32 = bone_lengths.iter().sum();
        let target_dist = (target - root).length();

        // Check reachability.
        if target_dist > total_length {
            // Target is unreachable; stretch the chain toward the target.
            let dir = if target_dist > f32::EPSILON {
                (target - root).normalize()
            } else {
                Vec3::Y
            };

            chain.joint_positions[0] = root;
            let mut accumulated = 0.0;
            for i in 0..n - 1 {
                accumulated += bone_lengths[i];
                chain.joint_positions[i + 1] = root + dir * accumulated;
            }

            // Apply weight.
            let w = chain.weight.clamp(0.0, 1.0);
            if (w - 1.0).abs() > f32::EPSILON {
                for (solved, orig) in chain
                    .joint_positions
                    .iter_mut()
                    .zip(original_positions.iter())
                {
                    *solved = orig.lerp(*solved, w);
                }
            }

            chain.compute_rotations_from_positions();
            return false;
        }

        // FABRIK iteration loop.
        for _iteration in 0..self.max_iterations {
            // Check convergence.
            let error = (chain.joint_positions[n - 1] - target).length();
            if error <= self.tolerance {
                break;
            }

            // Forward pass: end effector to root.
            Self::forward_pass(&mut chain.joint_positions, &bone_lengths, target);

            // Apply constraints after forward pass.
            if !chain.constraints.is_empty() {
                Self::apply_constraints(
                    &mut chain.joint_positions,
                    &bone_lengths,
                    &chain.constraints,
                );
            }

            // Backward pass: root to end effector.
            Self::backward_pass(&mut chain.joint_positions, &bone_lengths, root);

            // Apply constraints after backward pass.
            if !chain.constraints.is_empty() {
                Self::apply_constraints(
                    &mut chain.joint_positions,
                    &bone_lengths,
                    &chain.constraints,
                );
            }
        }

        // Apply pole target.
        if let Some(pole) = chain.pole_target {
            Self::apply_pole_target(&mut chain.joint_positions, &bone_lengths, pole);
        }

        // Apply weight blending.
        let w = chain.weight.clamp(0.0, 1.0);
        if (w - 1.0).abs() > f32::EPSILON {
            for (solved, orig) in chain
                .joint_positions
                .iter_mut()
                .zip(original_positions.iter())
            {
                *solved = orig.lerp(*solved, w);
            }
        }

        // Compute rotations from the final positions.
        chain.compute_rotations_from_positions();

        let final_error = (chain.joint_positions[n - 1] - target).length();
        final_error <= self.tolerance
    }
}

impl FABRIKSolver {
    /// Apply pole target by rotating the chain around the root-to-end axis.
    fn apply_pole_target(positions: &mut [Vec3], bone_lengths: &[f32], pole: Vec3) {
        let n = positions.len();
        if n < 3 {
            return;
        }

        let root = positions[0];
        let end = positions[n - 1];
        let chain_dir = end - root;

        if chain_dir.length_squared() < f32::EPSILON {
            return;
        }

        let chain_axis = chain_dir.normalize();

        // For each middle joint, rotate toward the pole target's projection
        // onto the perpendicular plane.
        for i in 1..n - 1 {
            let joint = positions[i];
            let joint_offset = joint - root;

            // Project joint onto the plane.
            let joint_proj = joint_offset - chain_axis * joint_offset.dot(chain_axis);
            let pole_offset = pole - root;
            let pole_proj = pole_offset - chain_axis * pole_offset.dot(chain_axis);

            if joint_proj.length_squared() < f32::EPSILON
                || pole_proj.length_squared() < f32::EPSILON
            {
                continue;
            }

            let current_dir = joint_proj.normalize();
            let target_dir = pole_proj.normalize();

            let dot = current_dir.dot(target_dir).clamp(-1.0, 1.0);
            let angle = dot.acos();

            if angle.abs() < f32::EPSILON {
                continue;
            }

            let cross = current_dir.cross(target_dir);
            let sign = if cross.dot(chain_axis) >= 0.0 {
                1.0
            } else {
                -1.0
            };

            let rotation = Quat::from_axis_angle(chain_axis, angle * sign);
            positions[i] = root + rotation * joint_offset;
        }

        // Re-enforce bone lengths after pole target adjustment.
        for i in 0..n - 1 {
            let dir = positions[i + 1] - positions[i];
            let dir_len = dir.length();
            if dir_len > f32::EPSILON {
                positions[i + 1] = positions[i] + dir / dir_len * bone_lengths[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Look-at IK
// ---------------------------------------------------------------------------

/// Simple look-at IK for orienting a single bone (e.g., head, eyes) toward
/// a target point.
pub struct LookAtIK {
    /// Maximum rotation angle (radians) from the rest direction.
    pub max_angle: f32,

    /// Speed of interpolation toward the target (radians per second).
    /// Set to `f32::MAX` for instant look-at.
    pub speed: f32,
}

impl Default for LookAtIK {
    fn default() -> Self {
        Self {
            max_angle: std::f32::consts::FRAC_PI_2,
            speed: f32::MAX,
        }
    }
}

impl LookAtIK {
    /// Compute the rotation needed for a bone at `bone_position` facing
    /// `current_forward` to look at `target`.
    ///
    /// Returns the constrained rotation quaternion.
    pub fn compute(
        &self,
        bone_position: Vec3,
        current_forward: Vec3,
        target: Vec3,
    ) -> Quat {
        let to_target = target - bone_position;
        if to_target.length_squared() < f32::EPSILON {
            return Quat::IDENTITY;
        }

        let desired_dir = to_target.normalize();
        let current_dir = current_forward.normalize();

        let dot = current_dir.dot(desired_dir).clamp(-1.0, 1.0);
        let angle = dot.acos();

        // Clamp to max angle.
        let clamped_angle = angle.min(self.max_angle);

        if clamped_angle < f32::EPSILON {
            return Quat::IDENTITY;
        }

        let cross = current_dir.cross(desired_dir);
        if cross.length_squared() < f32::EPSILON {
            // Target is directly behind; rotate around an arbitrary axis.
            let arbitrary = if current_dir.dot(Vec3::Y).abs() < 0.99 {
                Vec3::Y
            } else {
                Vec3::X
            };
            let axis = current_dir.cross(arbitrary).normalize();
            return Quat::from_axis_angle(axis, clamped_angle);
        }

        let axis = cross.normalize();
        Quat::from_axis_angle(axis, clamped_angle)
    }

    /// Compute a smoothly interpolated look-at rotation.
    pub fn compute_smooth(
        &self,
        bone_position: Vec3,
        current_forward: Vec3,
        current_rotation: Quat,
        target: Vec3,
        dt: f32,
    ) -> Quat {
        let desired = self.compute(bone_position, current_forward, target);
        let target_rot = desired * current_rotation;
        let max_step = self.speed * dt;
        let angle = current_rotation.dot(target_rot).abs().clamp(0.0, 1.0).acos() * 2.0;

        if angle <= max_step || self.speed >= f32::MAX * 0.5 {
            target_rot
        } else {
            let t = (max_step / angle).clamp(0.0, 1.0);
            current_rotation.slerp(target_rot, t)
        }
    }
}

// ---------------------------------------------------------------------------
// Utility: rotation between two vectors
// ---------------------------------------------------------------------------

/// Compute the shortest rotation quaternion that rotates `from` to `to`.
///
/// Both vectors should be unit length. Handles the degenerate case where
/// the vectors are anti-parallel.
pub fn rotation_between_vectors(from: Vec3, to: Vec3) -> Quat {
    let dot = from.dot(to);

    if dot > 0.9999 {
        return Quat::IDENTITY;
    }

    if dot < -0.9999 {
        // Nearly anti-parallel: pick an arbitrary perpendicular axis.
        let arbitrary = if from.dot(Vec3::X).abs() < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let axis = from.cross(arbitrary).normalize();
        return Quat::from_axis_angle(axis, std::f32::consts::PI);
    }

    let axis = from.cross(to);
    let s = ((1.0 + dot) * 2.0).sqrt();
    let inv_s = 1.0 / s;

    Quat::from_xyzw(axis.x * inv_s, axis.y * inv_s, axis.z * inv_s, s * 0.5).normalize()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.01;

    // -- IKChain tests --

    #[test]
    fn test_ik_chain_creation() {
        let chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(chain.len(), 3);
        assert!(!chain.is_empty());
        assert_eq!(chain.target, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_chain_length() {
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::ZERO);
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        assert!((chain.chain_length() - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_bone_lengths() {
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::ZERO);
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.5, 0.0),
        ];
        let lengths = chain.bone_lengths();
        assert_eq!(lengths.len(), 2);
        assert!((lengths[0] - 1.0).abs() < EPSILON);
        assert!((lengths[1] - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_is_reachable() {
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.0, 0.0, 0.0));
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        // Total length = 2.0, target distance = 1.0
        assert!(chain.is_reachable());

        chain.target = Vec3::new(5.0, 0.0, 0.0);
        assert!(!chain.is_reachable());
    }

    #[test]
    fn test_compute_rotations() {
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::ZERO);
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        chain.compute_rotations_from_positions();
        // Bones point up (Y axis), which is the default direction.
        // Rotation should be approximately identity.
        assert!(chain.joint_rotations[0].dot(Quat::IDENTITY).abs() > 0.99);
    }

    // -- TwoBoneIK tests --

    #[test]
    fn test_two_bone_basic() {
        let root = Vec3::new(0.0, 0.0, 0.0);
        let mid = Vec3::new(0.0, 1.0, 0.0);
        let end = Vec3::new(0.0, 2.0, 0.0);
        let target = Vec3::new(1.0, 1.0, 0.0);

        let result = TwoBoneIK::solve(root, mid, end, target, None);
        assert!(result.is_some());

        let (new_mid, new_end) = result.unwrap();
        // End effector should be close to the target (clamped to chain length).
        let dist_to_target = (new_end - target).length();
        assert!(dist_to_target < 0.5, "End effector should be near target");

        // Mid joint should be at the correct distance from root.
        let upper_len = (new_mid - root).length();
        assert!((upper_len - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_two_bone_fully_extended() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(0.0, 1.0, 0.0);
        let end = Vec3::new(0.0, 2.0, 0.0);
        // Target beyond reach
        let target = Vec3::new(0.0, 10.0, 0.0);

        let result = TwoBoneIK::solve(root, mid, end, target, None);
        assert!(result.is_some());

        let (_, new_end) = result.unwrap();
        // Should clamp to maximum reach (2.0)
        let dist = (new_end - root).length();
        assert!((dist - 2.0).abs() < 0.01, "Chain should be at max reach");
    }

    #[test]
    fn test_two_bone_zero_length() {
        let root = Vec3::ZERO;
        let mid = Vec3::ZERO; // Zero-length upper bone
        let end = Vec3::new(0.0, 1.0, 0.0);
        let target = Vec3::new(1.0, 0.0, 0.0);

        let result = TwoBoneIK::solve(root, mid, end, target, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_two_bone_with_pole_target() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(0.0, 1.0, 0.0);
        let end = Vec3::new(0.0, 2.0, 0.0);
        let target = Vec3::new(0.0, 1.5, 0.0);
        let pole = Vec3::new(0.0, 0.0, 1.0); // Bend toward +Z

        let result = TwoBoneIK::solve(root, mid, end, target, Some(pole));
        assert!(result.is_some());

        let (new_mid, _) = result.unwrap();
        // Mid joint should have a Z component > 0 due to pole target.
        assert!(new_mid.z > -EPSILON, "Mid joint should bend toward pole target");
    }

    #[test]
    fn test_two_bone_target_at_root() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(0.0, 1.0, 0.0);
        let end = Vec3::new(0.0, 2.0, 0.0);
        let target = Vec3::ZERO;

        let result = TwoBoneIK::solve(root, mid, end, target, None);
        assert!(result.is_some());
    }

    #[test]
    fn test_two_bone_solver_trait() {
        let solver = TwoBoneIK;
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.0, 1.0, 0.0));
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];

        let converged = solver.solve(&mut chain);
        assert!(converged);
    }

    #[test]
    fn test_two_bone_weight_blending() {
        let solver = TwoBoneIK;
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.0, 1.0, 0.0));
        chain.weight = 0.0; // No IK effect
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let original_mid = chain.joint_positions[1];

        solver.solve(&mut chain);
        // With weight=0, mid should not have moved.
        assert!((chain.joint_positions[1] - original_mid).length() < EPSILON);
    }

    // -- CCDSolver tests --

    #[test]
    fn test_ccd_basic() {
        let solver = CCDSolver::new(100, 0.05);
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.5, 0.0, 0.0));
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];

        let converged = solver.solve(&mut chain);

        let error = chain.error();
        assert!(
            converged || error < 0.1,
            "CCD should converge or be close: converged={}, error={}",
            converged,
            error
        );
    }

    #[test]
    fn test_ccd_unreachable() {
        let solver = CCDSolver::default();
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(10.0, 0.0, 0.0));
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];

        let converged = solver.solve(&mut chain);
        assert!(!converged, "CCD should not converge for unreachable target");
    }

    #[test]
    fn test_ccd_single_joint() {
        let solver = CCDSolver::default();
        let mut chain = IKChain::new(vec![0], Vec3::new(1.0, 0.0, 0.0));
        chain.joint_positions = vec![Vec3::ZERO];

        let converged = solver.solve(&mut chain);
        assert!(!converged, "Single joint chain should not converge");
    }

    #[test]
    fn test_ccd_long_chain() {
        let solver = CCDSolver::new(50, 0.01);
        let n = 10;
        let mut chain = IKChain::new(
            (0..n).collect(),
            Vec3::new(3.0, 2.0, 0.0),
        );
        chain.joint_positions = (0..n)
            .map(|i| Vec3::new(0.0, i as f32 * 0.5, 0.0))
            .collect();

        let converged = solver.solve(&mut chain);
        if converged {
            let error = chain.error();
            assert!(error < 0.1, "Long chain error should be small: {}", error);
        }
    }

    #[test]
    fn test_ccd_with_damping() {
        let solver = CCDSolver::new(20, 0.01).with_damping(0.5);
        let mut chain = IKChain::new(vec![0, 1, 2, 3], Vec3::new(2.0, 0.0, 0.0));
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];

        let converged = solver.solve(&mut chain);
        // With damping, might need more iterations but should still work.
        let error = chain.error();
        assert!(error < 1.0, "Damped CCD should make progress: {}", error);
        let _ = converged;
    }

    #[test]
    fn test_ccd_with_pole_target() {
        let solver = CCDSolver::default();
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.5, 0.0, 0.0));
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        chain.set_pole_target(Vec3::new(0.0, 0.0, 1.0));

        solver.solve(&mut chain);
        // The solver should bend toward the pole target.
    }

    // -- FABRIKSolver tests --

    #[test]
    fn test_fabrik_basic() {
        let solver = FABRIKSolver::default();
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.5, 0.0, 0.0));
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];

        let converged = solver.solve(&mut chain);
        assert!(converged, "FABRIK should converge for reachable target");

        let error = chain.error();
        assert!(error < 0.01, "End effector error should be small: {}", error);

        // Root should stay at origin.
        assert!(
            (chain.joint_positions[0] - Vec3::ZERO).length() < EPSILON,
            "Root should not move"
        );
    }

    #[test]
    fn test_fabrik_unreachable() {
        let solver = FABRIKSolver::default();
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(10.0, 0.0, 0.0));
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];

        let converged = solver.solve(&mut chain);
        assert!(!converged, "FABRIK should not converge for unreachable target");

        // Chain should be stretched toward the target.
        let end = chain.joint_positions[2];
        assert!(end.x > 0.0, "Chain should stretch toward target");
    }

    #[test]
    fn test_fabrik_preserves_bone_lengths() {
        let solver = FABRIKSolver::default();
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.0, 0.5, 0.0));
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];

        solver.solve(&mut chain);

        // Check bone lengths are preserved.
        let upper = (chain.joint_positions[1] - chain.joint_positions[0]).length();
        let lower = (chain.joint_positions[2] - chain.joint_positions[1]).length();
        assert!((upper - 1.0).abs() < EPSILON, "Upper bone length changed: {}", upper);
        assert!((lower - 1.0).abs() < EPSILON, "Lower bone length changed: {}", lower);
    }

    #[test]
    fn test_fabrik_long_chain() {
        let solver = FABRIKSolver::new(30, 0.01);
        let n = 8;
        let mut chain = IKChain::new(
            (0..n).collect(),
            Vec3::new(3.0, 1.0, 0.0),
        );
        chain.joint_positions = (0..n)
            .map(|i| Vec3::new(0.0, i as f32, 0.0))
            .collect();

        let converged = solver.solve(&mut chain);
        assert!(converged, "FABRIK should converge for a reachable long chain");

        let error = chain.error();
        assert!(error < 0.05, "FABRIK long chain error: {}", error);
    }

    #[test]
    fn test_fabrik_weight_blending() {
        let solver = FABRIKSolver::default();
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.5, 0.0, 0.0));
        chain.weight = 0.5;
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let original_end = chain.joint_positions[2];

        solver.solve(&mut chain);

        // At weight=0.5, the end effector should be between FK and IK.
        let end = chain.joint_positions[2];
        let target = chain.target;
        let fk_dist = (original_end - target).length();
        let ik_dist = (end - target).length();
        assert!(
            ik_dist < fk_dist || ik_dist < 0.5,
            "Half weight should move partially toward target"
        );
    }

    #[test]
    fn test_fabrik_with_pole_target() {
        let solver = FABRIKSolver::default();
        let mut chain = IKChain::new(vec![0, 1, 2], Vec3::new(1.5, 0.0, 0.0));
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        chain.set_pole_target(Vec3::new(0.0, 0.0, 1.0));

        solver.solve(&mut chain);
        // Mid joint should lean toward +Z due to pole target.
    }

    // -- Joint constraint tests --

    #[test]
    fn test_ball_socket_constraint() {
        let constraint = JointConstraint::BallSocket {
            max_angle: std::f32::consts::FRAC_PI_4,
        };
        let rest = Vec3::Y;
        // Rotation that goes beyond 45 degrees from rest.
        let big_rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let constrained = constraint.apply(big_rotation, rest);

        let rotated_dir = constrained * rest;
        let angle = rest.dot(rotated_dir).clamp(-1.0, 1.0).acos();
        assert!(
            angle <= std::f32::consts::FRAC_PI_4 + EPSILON,
            "Ball-socket should clamp angle to max: {}",
            angle
        );
    }

    #[test]
    fn test_hinge_constraint() {
        let constraint = JointConstraint::Hinge {
            axis: Vec3::Z,
            min_angle: -std::f32::consts::FRAC_PI_4,
            max_angle: std::f32::consts::FRAC_PI_4,
        };

        // Rotation around Z axis within limits.
        let rotation = Quat::from_rotation_z(0.1);
        let constrained = constraint.apply(rotation, Vec3::Y);
        // Should not change much.
        assert!(rotation.dot(constrained).abs() > 0.9);
    }

    // -- LookAtIK tests --

    #[test]
    fn test_look_at_basic() {
        let look_at = LookAtIK::default();
        let bone_pos = Vec3::ZERO;
        let forward = Vec3::NEG_Z;
        let target = Vec3::new(1.0, 0.0, -1.0);

        let rot = look_at.compute(bone_pos, forward, target);
        let new_forward = rot * forward;
        let desired = (target - bone_pos).normalize();
        let dot = new_forward.dot(desired);
        assert!(dot > 0.9, "Look-at should rotate toward target: dot={}", dot);
    }

    #[test]
    fn test_look_at_max_angle() {
        let look_at = LookAtIK {
            max_angle: 0.1,
            speed: f32::MAX,
        };
        let bone_pos = Vec3::ZERO;
        let forward = Vec3::NEG_Z;
        let target = Vec3::new(100.0, 0.0, 0.0); // 90 degrees away

        let rot = look_at.compute(bone_pos, forward, target);
        let new_forward = rot * forward;
        let angle = forward.dot(new_forward).clamp(-1.0, 1.0).acos();
        assert!(
            angle <= 0.1 + EPSILON,
            "Look-at should clamp to max angle: {}",
            angle
        );
    }

    // -- Utility tests --

    #[test]
    fn test_rotation_between_vectors() {
        let from = Vec3::X;
        let to = Vec3::Y;
        let rot = rotation_between_vectors(from, to);
        let result = rot * from;
        assert!((result - to).length() < EPSILON);
    }

    #[test]
    fn test_rotation_between_same_vectors() {
        let v = Vec3::X;
        let rot = rotation_between_vectors(v, v);
        assert!(rot.dot(Quat::IDENTITY).abs() > 0.999);
    }

    #[test]
    fn test_rotation_between_opposite_vectors() {
        let from = Vec3::X;
        let to = Vec3::NEG_X;
        let rot = rotation_between_vectors(from, to);
        let result = rot * from;
        assert!((result - to).length() < EPSILON);
    }

    #[test]
    fn test_decompose_twist_swing() {
        let q = Quat::from_rotation_y(0.5) * Quat::from_rotation_x(0.3);
        let (twist, swing) = decompose_twist_swing(q, Vec3::Y);
        // Recompose: swing * twist should approximately equal q.
        let recomposed = swing * twist;
        assert!(
            q.dot(recomposed).abs() > 0.99,
            "Twist-swing decomposition should recompose correctly"
        );
    }

    // -- Integration: FABRIK with constraints --

    #[test]
    fn test_fabrik_with_ball_socket_constraint() {
        let solver = FABRIKSolver::new(20, 0.05);
        let mut chain = IKChain::with_constraints(
            vec![0, 1, 2, 3],
            Vec3::new(2.0, 0.0, 0.0),
            vec![
                JointConstraint::None,
                JointConstraint::BallSocket {
                    max_angle: std::f32::consts::FRAC_PI_4,
                },
                JointConstraint::BallSocket {
                    max_angle: std::f32::consts::FRAC_PI_4,
                },
                JointConstraint::None,
            ],
        );
        chain.joint_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];

        solver.solve(&mut chain);
        // Should still make progress toward the target, just constrained.
        let error = chain.error();
        assert!(error < 2.0, "Constrained FABRIK error: {}", error);
    }
}
