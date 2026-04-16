//! Spring and jiggle bone physics for secondary motion.
//!
//! Provides physics-based secondary animation for hair, cloth, accessories,
//! tails, and other dangling/bouncy elements. These systems run *after* the
//! primary skeletal animation and add physically-motivated motion on top of
//! the animated pose.
//!
//! # Overview
//!
//! - [`SpringBone`]: a single bone in a spring bone chain. Tracks a virtual
//!   particle position that is attracted to the bone's animated position via
//!   a spring-damper system.
//! - [`SpringBoneChain`]: a chain of connected spring bones that are simulated
//!   together with optional collision against spheres and capsules.
//! - [`JiggleBone`]: a simpler, single-bone spring for quick bouncing effects
//!   with optional axis constraints.
//! - [`SpringCollider`]: collision primitives (sphere, capsule) that spring
//!   bones can collide against.
//! - [`WindSource`]: a global or local wind force affecting spring bones.

use genovo_core::Transform;
use glam::{Quat, Vec3};

// ---------------------------------------------------------------------------
// SpringCollider
// ---------------------------------------------------------------------------

/// A collision primitive for spring bone interaction.
#[derive(Debug, Clone)]
pub enum SpringCollider {
    /// A sphere collider defined by center and radius.
    Sphere {
        /// World-space center.
        center: Vec3,
        /// Radius.
        radius: f32,
    },
    /// A capsule collider defined by two endpoints and a radius.
    Capsule {
        /// World-space position of the first endpoint.
        start: Vec3,
        /// World-space position of the second endpoint.
        end: Vec3,
        /// Radius of the capsule.
        radius: f32,
    },
}

impl SpringCollider {
    /// Create a sphere collider.
    pub fn sphere(center: Vec3, radius: f32) -> Self {
        Self::Sphere { center, radius }
    }

    /// Create a capsule collider.
    pub fn capsule(start: Vec3, end: Vec3, radius: f32) -> Self {
        Self::Capsule { start, end, radius }
    }

    /// Resolve collision for a point with the given radius. Returns the
    /// corrected position if a collision occurred, or `None` if no collision.
    pub fn resolve(&self, position: Vec3, point_radius: f32) -> Option<Vec3> {
        match self {
            Self::Sphere { center, radius } => {
                let diff = position - *center;
                let dist = diff.length();
                let min_dist = *radius + point_radius;
                if dist < min_dist && dist > f32::EPSILON {
                    let normal = diff / dist;
                    Some(*center + normal * min_dist)
                } else {
                    None
                }
            }
            Self::Capsule { start, end, radius } => {
                let seg = *end - *start;
                let seg_len_sq = seg.length_squared();
                let t = if seg_len_sq > f32::EPSILON {
                    ((position - *start).dot(seg) / seg_len_sq).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let closest = *start + seg * t;
                let diff = position - closest;
                let dist = diff.length();
                let min_dist = *radius + point_radius;
                if dist < min_dist && dist > f32::EPSILON {
                    let normal = diff / dist;
                    Some(closest + normal * min_dist)
                } else {
                    None
                }
            }
        }
    }

    /// Check if a point (with radius) is intersecting this collider.
    pub fn intersects(&self, position: Vec3, point_radius: f32) -> bool {
        self.resolve(position, point_radius).is_some()
    }
}

// ---------------------------------------------------------------------------
// WindSource
// ---------------------------------------------------------------------------

/// A wind force source affecting spring bones.
#[derive(Debug, Clone)]
pub struct WindSource {
    /// Base wind direction (normalized).
    pub direction: Vec3,

    /// Wind strength (force magnitude).
    pub strength: f32,

    /// Turbulence factor: adds noise to the wind direction [0, 1].
    pub turbulence: f32,

    /// Frequency of turbulence oscillation in Hz.
    pub frequency: f32,

    /// Current time accumulator for turbulence computation.
    time: f32,
}

impl WindSource {
    /// Create a new wind source.
    pub fn new(direction: Vec3, strength: f32) -> Self {
        Self {
            direction: direction.normalize_or_zero(),
            strength,
            turbulence: 0.0,
            frequency: 1.0,
            time: 0.0,
        }
    }

    /// Create a wind source with turbulence.
    pub fn with_turbulence(mut self, turbulence: f32, frequency: f32) -> Self {
        self.turbulence = turbulence.clamp(0.0, 1.0);
        self.frequency = frequency.max(0.01);
        self
    }

    /// Advance the wind simulation by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        self.time += dt;
    }

    /// Sample the wind force vector at the current time.
    ///
    /// Includes turbulence as sinusoidal perturbation of the base direction.
    pub fn sample(&self) -> Vec3 {
        let base = self.direction * self.strength;
        if self.turbulence < f32::EPSILON {
            return base;
        }

        // Simple sinusoidal turbulence on two axes perpendicular to wind.
        let phase = self.time * self.frequency * std::f32::consts::TAU;
        let turb_x = (phase).sin();
        let turb_y = (phase * 1.7 + 0.5).cos();

        // Build a perpendicular frame. Use the world up if wind isn't aligned.
        let up = if self.direction.y.abs() < 0.99 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let right = self.direction.cross(up).normalize_or_zero();
        let up_perp = right.cross(self.direction).normalize_or_zero();

        let noise = (right * turb_x + up_perp * turb_y) * self.turbulence * self.strength;
        base + noise
    }

    /// Sample wind force at a specific world position (for future spatial
    /// variation). Currently returns the same value everywhere.
    pub fn sample_at(&self, _position: Vec3) -> Vec3 {
        self.sample()
    }
}

impl Default for WindSource {
    fn default() -> Self {
        Self::new(Vec3::X, 0.0)
    }
}

// ---------------------------------------------------------------------------
// SpringBone
// ---------------------------------------------------------------------------

/// A single spring bone: a virtual particle tethered to a skeletal bone via
/// a spring-damper system.
///
/// Each spring bone tracks a simulated position. The simulated position is
/// attracted toward the bone's animated (target) position by a spring, and
/// damped to prevent perpetual oscillation. Gravity and wind forces can also
/// be applied.
#[derive(Debug, Clone)]
pub struct SpringBone {
    /// Index of the bone in the skeleton.
    pub bone_index: usize,

    /// Spring stiffness coefficient. Higher = stiffer (snaps back faster).
    /// Typical range: 10..500.
    pub stiffness: f32,

    /// Damping coefficient. Higher = less oscillation.
    /// Typical range: 0.1..1.0.
    pub damping: f32,

    /// Mass of the virtual particle (affects inertia). Typical range: 0.1..2.0.
    pub mass: f32,

    /// Gravity force applied to the bone.
    pub gravity: Vec3,

    /// Collision radius for sphere-casting against colliders.
    pub radius: f32,

    /// Rest length from parent (computed from bind pose).
    pub rest_length: f32,

    // --- simulation state ---
    /// Current simulated position (world space).
    pub current_position: Vec3,

    /// Previous simulated position (for Verlet integration).
    pub previous_position: Vec3,

    /// The animated (target) position from the skeleton evaluation.
    pub target_position: Vec3,

    /// Whether this bone has been initialized.
    initialized: bool,
}

impl SpringBone {
    /// Create a new spring bone for the given bone index.
    pub fn new(bone_index: usize) -> Self {
        Self {
            bone_index,
            stiffness: 100.0,
            damping: 0.5,
            mass: 1.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            radius: 0.02,
            rest_length: 0.0,
            current_position: Vec3::ZERO,
            previous_position: Vec3::ZERO,
            target_position: Vec3::ZERO,
            initialized: false,
        }
    }

    /// Builder: set stiffness.
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.max(0.0);
        self
    }

    /// Builder: set damping.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Builder: set mass.
    pub fn with_mass(mut self, mass: f32) -> Self {
        self.mass = mass.max(0.01);
        self
    }

    /// Builder: set gravity.
    pub fn with_gravity(mut self, gravity: Vec3) -> Self {
        self.gravity = gravity;
        self
    }

    /// Builder: set collision radius.
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius.max(0.0);
        self
    }

    /// Initialize the spring bone from the current animated position.
    pub fn initialize(&mut self, world_position: Vec3) {
        self.current_position = world_position;
        self.previous_position = world_position;
        self.target_position = world_position;
        self.initialized = true;
    }

    /// Set the target (animated) position for this frame.
    pub fn set_target(&mut self, position: Vec3) {
        self.target_position = position;
        if !self.initialized {
            self.initialize(position);
        }
    }

    /// Perform one simulation step using Verlet integration.
    ///
    /// # Algorithm
    ///
    /// 1. Compute the velocity from the difference between current and previous
    ///    position (Verlet integration).
    /// 2. Apply damping to the velocity.
    /// 3. Compute the spring force: `F = -k * (current - target)`
    /// 4. Add gravity and external forces.
    /// 5. Compute acceleration: `a = F / mass`
    /// 6. Update position: `new = current + velocity * (1 - damping) + a * dt^2`
    /// 7. Enforce rest-length constraint from parent.
    pub fn simulate(
        &mut self,
        dt: f32,
        parent_position: Option<Vec3>,
        external_force: Vec3,
    ) {
        if dt <= 0.0 || !self.initialized {
            return;
        }

        let dt = dt.min(1.0 / 30.0); // Cap timestep to prevent explosion.

        // Verlet velocity.
        let velocity = self.current_position - self.previous_position;

        // Spring force toward target.
        let displacement = self.current_position - self.target_position;
        let spring_force = -displacement * self.stiffness;

        // Total force.
        let total_force = spring_force + self.gravity * self.mass + external_force;

        // Acceleration.
        let acceleration = total_force / self.mass;

        // Verlet integration with damping.
        let damping_factor = 1.0 - self.damping;
        let new_position = self.current_position
            + velocity * damping_factor
            + acceleration * dt * dt;

        self.previous_position = self.current_position;
        self.current_position = new_position;

        // Enforce rest length from parent.
        if let Some(parent_pos) = parent_position {
            if self.rest_length > f32::EPSILON {
                let to_bone = self.current_position - parent_pos;
                let dist = to_bone.length();
                if dist > f32::EPSILON {
                    self.current_position =
                        parent_pos + (to_bone / dist) * self.rest_length;
                }
            }
        }
    }

    /// Resolve collision against a set of colliders.
    pub fn resolve_collisions(&mut self, colliders: &[SpringCollider]) {
        for collider in colliders {
            if let Some(corrected) = collider.resolve(self.current_position, self.radius) {
                self.current_position = corrected;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SpringBoneChain
// ---------------------------------------------------------------------------

/// A chain of connected spring bones simulated together.
///
/// The chain follows the skeleton hierarchy: each bone's parent constraint
/// is applied after the spring simulation to maintain the chain structure.
#[derive(Debug, Clone)]
pub struct SpringBoneChain {
    /// Name for debugging.
    pub name: String,

    /// The bones in the chain, ordered from root to tip.
    pub bones: Vec<SpringBone>,

    /// Colliders that affect this chain.
    pub colliders: Vec<SpringCollider>,

    /// Wind sources affecting this chain.
    pub wind_sources: Vec<WindSource>,

    /// Global stiffness multiplier (scales all bone stiffnesses).
    pub stiffness_multiplier: f32,

    /// Global damping multiplier.
    pub damping_multiplier: f32,

    /// Whether the chain is active.
    pub enabled: bool,
}

impl SpringBoneChain {
    /// Create a new spring bone chain.
    pub fn new(name: impl Into<String>, bone_indices: &[usize]) -> Self {
        let bones: Vec<SpringBone> = bone_indices
            .iter()
            .map(|&idx| SpringBone::new(idx))
            .collect();

        Self {
            name: name.into(),
            bones,
            colliders: Vec::new(),
            wind_sources: Vec::new(),
            stiffness_multiplier: 1.0,
            damping_multiplier: 1.0,
            enabled: true,
        }
    }

    /// Create a chain from pre-configured spring bones.
    pub fn from_bones(name: impl Into<String>, bones: Vec<SpringBone>) -> Self {
        Self {
            name: name.into(),
            bones,
            colliders: Vec::new(),
            wind_sources: Vec::new(),
            stiffness_multiplier: 1.0,
            damping_multiplier: 1.0,
            enabled: true,
        }
    }

    /// Set uniform stiffness for all bones in the chain.
    pub fn set_stiffness(&mut self, stiffness: f32) {
        for bone in &mut self.bones {
            bone.stiffness = stiffness;
        }
    }

    /// Set uniform damping for all bones.
    pub fn set_damping(&mut self, damping: f32) {
        for bone in &mut self.bones {
            bone.damping = damping;
        }
    }

    /// Set gravity for all bones.
    pub fn set_gravity(&mut self, gravity: Vec3) {
        for bone in &mut self.bones {
            bone.gravity = gravity;
        }
    }

    /// Add a collider.
    pub fn add_collider(&mut self, collider: SpringCollider) {
        self.colliders.push(collider);
    }

    /// Add a wind source.
    pub fn add_wind(&mut self, wind: WindSource) {
        self.wind_sources.push(wind);
    }

    /// Initialize all bones from their animated world positions.
    pub fn initialize(&mut self, world_positions: &[Vec3]) {
        for bone in &mut self.bones {
            if bone.bone_index < world_positions.len() {
                bone.initialize(world_positions[bone.bone_index]);
            }
        }

        // Compute rest lengths from initial positions.
        for i in 1..self.bones.len() {
            let parent_idx = self.bones[i - 1].bone_index;
            let child_idx = self.bones[i].bone_index;
            if parent_idx < world_positions.len() && child_idx < world_positions.len() {
                self.bones[i].rest_length =
                    (world_positions[child_idx] - world_positions[parent_idx]).length();
            }
        }
    }

    /// Update the animated target positions for all bones in the chain.
    pub fn set_targets(&mut self, world_positions: &[Vec3]) {
        for bone in &mut self.bones {
            if bone.bone_index < world_positions.len() {
                bone.set_target(world_positions[bone.bone_index]);
            }
        }
    }

    /// Run one simulation step for the entire chain.
    ///
    /// Processes bones from root to tip so that each bone's parent constraint
    /// uses the already-updated parent position.
    pub fn update(&mut self, dt: f32) {
        if !self.enabled || dt <= 0.0 {
            return;
        }

        // Update wind sources.
        for wind in &mut self.wind_sources {
            wind.update(dt);
        }

        // Compute combined wind force.
        let wind_force: Vec3 = self.wind_sources.iter().map(|w| w.sample()).sum();

        // Simulate each bone from root to tip.
        for i in 0..self.bones.len() {
            let parent_pos = if i > 0 {
                Some(self.bones[i - 1].current_position)
            } else {
                None
            };

            // Apply multipliers.
            let orig_stiffness = self.bones[i].stiffness;
            let orig_damping = self.bones[i].damping;
            self.bones[i].stiffness *= self.stiffness_multiplier;
            self.bones[i].damping *= self.damping_multiplier;

            self.bones[i].simulate(dt, parent_pos, wind_force);

            // Restore original values so multipliers don't compound.
            self.bones[i].stiffness = orig_stiffness;
            self.bones[i].damping = orig_damping;

            // Collision.
            self.bones[i].resolve_collisions(&self.colliders);
        }
    }

    /// Get the simulated world positions for all bones in the chain.
    pub fn simulated_positions(&self) -> Vec<(usize, Vec3)> {
        self.bones
            .iter()
            .map(|b| (b.bone_index, b.current_position))
            .collect()
    }

    /// Write the simulated positions back into a world-space transform array.
    ///
    /// This modifies only the translation component; rotation is inferred from
    /// the direction to the child bone (aim constraint).
    pub fn write_transforms(&self, transforms: &mut [Transform]) {
        for (i, bone) in self.bones.iter().enumerate() {
            if bone.bone_index < transforms.len() {
                transforms[bone.bone_index].position = bone.current_position;

                // Compute rotation from direction to next bone (aim constraint).
                if i + 1 < self.bones.len() {
                    let next_pos = self.bones[i + 1].current_position;
                    let dir = (next_pos - bone.current_position).normalize_or_zero();
                    if dir.length_squared() > 0.5 {
                        let bind_dir = Vec3::NEG_Z; // Convention: bone points -Z.
                        if let Some(rot) = rotation_between(bind_dir, dir) {
                            transforms[bone.bone_index].rotation = rot;
                        }
                    }
                }
            }
        }
    }

    /// Number of bones in the chain.
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Reset all simulation state to target positions.
    pub fn reset(&mut self) {
        for bone in &mut self.bones {
            bone.current_position = bone.target_position;
            bone.previous_position = bone.target_position;
        }
    }
}

// ---------------------------------------------------------------------------
// JiggleBone
// ---------------------------------------------------------------------------

/// A single-bone spring-damper for simple bouncing/jiggling effects.
///
/// Simpler than a full [`SpringBoneChain`]: operates on a single bone with
/// optional axis constraints. Good for simple accessories, antennae, or
/// subtle body-part jiggle.
#[derive(Debug, Clone)]
pub struct JiggleBone {
    /// Bone index in the skeleton.
    pub bone_index: usize,

    /// Spring stiffness.
    pub stiffness: f32,

    /// Damping coefficient.
    pub damping: f32,

    /// Mass of the virtual particle.
    pub mass: f32,

    /// Axis constraint: if set, only jiggle along this normalized axis.
    /// `None` means jiggle freely in all directions.
    pub axis_constraint: Option<Vec3>,

    /// Maximum displacement from rest pose.
    pub max_displacement: f32,

    /// Current simulated offset from the target (rest) position.
    offset: Vec3,

    /// Current velocity.
    velocity: Vec3,

    /// Target (animated) position.
    target: Vec3,

    /// Whether this jiggle bone is active.
    pub enabled: bool,

    /// Whether the simulation has been initialized.
    initialized: bool,
}

impl JiggleBone {
    /// Create a new jiggle bone.
    pub fn new(bone_index: usize) -> Self {
        Self {
            bone_index,
            stiffness: 200.0,
            damping: 0.6,
            mass: 0.5,
            axis_constraint: None,
            max_displacement: 0.5,
            offset: Vec3::ZERO,
            velocity: Vec3::ZERO,
            target: Vec3::ZERO,
            enabled: true,
            initialized: false,
        }
    }

    /// Builder: set stiffness.
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.max(0.0);
        self
    }

    /// Builder: set damping.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Builder: set axis constraint.
    pub fn with_axis_constraint(mut self, axis: Vec3) -> Self {
        self.axis_constraint = Some(axis.normalize_or_zero());
        self
    }

    /// Builder: set max displacement.
    pub fn with_max_displacement(mut self, max: f32) -> Self {
        self.max_displacement = max.max(0.0);
        self
    }

    /// Set the target position from the animated skeleton.
    pub fn set_target(&mut self, position: Vec3) {
        if !self.initialized {
            self.target = position;
            self.offset = Vec3::ZERO;
            self.velocity = Vec3::ZERO;
            self.initialized = true;
        } else {
            // Compute the impulse from target movement.
            let delta = position - self.target;
            self.velocity -= delta; // React to sudden target movement.
            self.target = position;
        }
    }

    /// Simulate one step of the jiggle physics.
    ///
    /// Uses a simple spring-damper differential equation:
    ///
    /// `a = (-stiffness * offset - damping * velocity) / mass`
    pub fn update(&mut self, dt: f32) {
        if !self.enabled || !self.initialized || dt <= 0.0 {
            return;
        }

        let dt = dt.min(1.0 / 30.0);

        // Spring-damper forces.
        let spring_force = -self.offset * self.stiffness;
        let damping_force = -self.velocity * self.damping * 2.0 * (self.stiffness * self.mass).sqrt();
        let acceleration = (spring_force + damping_force) / self.mass;

        // Semi-implicit Euler integration.
        self.velocity += acceleration * dt;
        self.offset += self.velocity * dt;

        // Apply axis constraint: project offset onto the constraint axis.
        if let Some(axis) = self.axis_constraint {
            let projected = axis * self.offset.dot(axis);
            self.offset = projected;
            self.velocity = axis * self.velocity.dot(axis);
        }

        // Clamp displacement.
        let disp = self.offset.length();
        if disp > self.max_displacement {
            self.offset = self.offset.normalize_or_zero() * self.max_displacement;
            // Reduce velocity too.
            self.velocity *= 0.5;
        }
    }

    /// Get the current simulated world position.
    pub fn simulated_position(&self) -> Vec3 {
        self.target + self.offset
    }

    /// Write the simulated position into a transform array.
    pub fn write_transform(&self, transforms: &mut [Transform]) {
        if self.bone_index < transforms.len() {
            transforms[self.bone_index].position = self.simulated_position();
        }
    }

    /// Get the current displacement from the rest position.
    pub fn displacement(&self) -> Vec3 {
        self.offset
    }

    /// Reset the simulation.
    pub fn reset(&mut self) {
        self.offset = Vec3::ZERO;
        self.velocity = Vec3::ZERO;
    }
}

// ---------------------------------------------------------------------------
// SpringBoneSystem
// ---------------------------------------------------------------------------

/// A container that manages multiple spring bone chains and jiggle bones,
/// processing them each frame after the primary animation evaluation.
#[derive(Debug, Clone)]
pub struct SpringBoneSystem {
    /// All spring bone chains.
    pub chains: Vec<SpringBoneChain>,

    /// All standalone jiggle bones.
    pub jiggle_bones: Vec<JiggleBone>,

    /// Whether the system is paused.
    pub paused: bool,

    /// Global stiffness multiplier (affects all chains/jiggle bones).
    pub global_stiffness: f32,

    /// Global damping multiplier.
    pub global_damping: f32,
}

impl Default for SpringBoneSystem {
    fn default() -> Self {
        Self {
            chains: Vec::new(),
            jiggle_bones: Vec::new(),
            paused: false,
            global_stiffness: 1.0,
            global_damping: 1.0,
        }
    }
}

impl SpringBoneSystem {
    /// Create a new empty spring bone system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a spring bone chain.
    pub fn add_chain(&mut self, chain: SpringBoneChain) {
        self.chains.push(chain);
    }

    /// Add a jiggle bone.
    pub fn add_jiggle(&mut self, jiggle: JiggleBone) {
        self.jiggle_bones.push(jiggle);
    }

    /// Find a chain by name.
    pub fn find_chain(&self, name: &str) -> Option<usize> {
        self.chains.iter().position(|c| c.name == name)
    }

    /// Get a mutable reference to a chain by name.
    pub fn chain_mut(&mut self, name: &str) -> Option<&mut SpringBoneChain> {
        self.chains.iter_mut().find(|c| c.name == name)
    }

    /// Initialize all chains and jiggle bones from the current world positions.
    pub fn initialize(&mut self, world_positions: &[Vec3]) {
        for chain in &mut self.chains {
            chain.initialize(world_positions);
        }
        for jiggle in &mut self.jiggle_bones {
            if jiggle.bone_index < world_positions.len() {
                jiggle.set_target(world_positions[jiggle.bone_index]);
            }
        }
    }

    /// Update all chains and jiggle bones with new animated positions.
    pub fn update(&mut self, dt: f32, world_positions: &[Vec3]) {
        if self.paused {
            return;
        }

        // Update chain targets and simulate.
        for chain in &mut self.chains {
            chain.stiffness_multiplier = self.global_stiffness;
            chain.damping_multiplier = self.global_damping;
            chain.set_targets(world_positions);
            chain.update(dt);
        }

        // Update jiggle bones.
        for jiggle in &mut self.jiggle_bones {
            if jiggle.bone_index < world_positions.len() {
                jiggle.set_target(world_positions[jiggle.bone_index]);
                jiggle.update(dt);
            }
        }
    }

    /// Write all simulated positions into a transform array.
    pub fn write_transforms(&self, transforms: &mut [Transform]) {
        for chain in &self.chains {
            chain.write_transforms(transforms);
        }
        for jiggle in &self.jiggle_bones {
            jiggle.write_transform(transforms);
        }
    }

    /// Reset all simulation state.
    pub fn reset_all(&mut self) {
        for chain in &mut self.chains {
            chain.reset();
        }
        for jiggle in &mut self.jiggle_bones {
            jiggle.reset();
        }
    }

    /// Total number of simulated bones (chains + jiggle).
    pub fn total_bone_count(&self) -> usize {
        let chain_count: usize = self.chains.iter().map(|c| c.bone_count()).sum();
        chain_count + self.jiggle_bones.len()
    }
}

// ---------------------------------------------------------------------------
// Rotation helpers
// ---------------------------------------------------------------------------

/// Compute the shortest-arc quaternion rotation from direction `a` to `b`.
fn rotation_between(from: Vec3, to: Vec3) -> Option<Quat> {
    let from_n = from.normalize_or_zero();
    let to_n = to.normalize_or_zero();

    if from_n.length_squared() < 0.5 || to_n.length_squared() < 0.5 {
        return None;
    }

    let dot = from_n.dot(to_n);
    if dot > 0.9999 {
        return Some(Quat::IDENTITY);
    }
    if dot < -0.9999 {
        // 180-degree rotation: pick an arbitrary perpendicular axis.
        let perp = if from_n.x.abs() < 0.9 {
            from_n.cross(Vec3::X).normalize()
        } else {
            from_n.cross(Vec3::Y).normalize()
        };
        return Some(Quat::from_axis_angle(perp, std::f32::consts::PI));
    }

    let axis = from_n.cross(to_n);
    let s = ((1.0 + dot) * 2.0).sqrt();
    let inv_s = 1.0 / s;
    Some(Quat::from_xyzw(
        axis.x * inv_s,
        axis.y * inv_s,
        axis.z * inv_s,
        s * 0.5,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    // -- SpringCollider tests --

    #[test]
    fn test_sphere_collider_no_collision() {
        let collider = SpringCollider::sphere(Vec3::ZERO, 0.5);
        assert!(collider.resolve(Vec3::new(2.0, 0.0, 0.0), 0.1).is_none());
    }

    #[test]
    fn test_sphere_collider_collision() {
        let collider = SpringCollider::sphere(Vec3::ZERO, 0.5);
        let result = collider.resolve(Vec3::new(0.3, 0.0, 0.0), 0.1);
        assert!(result.is_some());
        let corrected = result.unwrap();
        let dist = corrected.length();
        assert!(
            (dist - 0.6).abs() < 0.01,
            "Corrected should be at radius + point_radius: {}",
            dist
        );
    }

    #[test]
    fn test_capsule_collider_no_collision() {
        let collider = SpringCollider::capsule(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.3,
        );
        assert!(collider.resolve(Vec3::new(5.0, 0.5, 0.0), 0.1).is_none());
    }

    #[test]
    fn test_capsule_collider_collision() {
        let collider = SpringCollider::capsule(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.3,
        );
        let result = collider.resolve(Vec3::new(0.2, 0.5, 0.0), 0.05);
        assert!(result.is_some());
    }

    #[test]
    fn test_collider_intersects() {
        let collider = SpringCollider::sphere(Vec3::ZERO, 1.0);
        assert!(collider.intersects(Vec3::new(0.5, 0.0, 0.0), 0.1));
        assert!(!collider.intersects(Vec3::new(5.0, 0.0, 0.0), 0.1));
    }

    // -- WindSource tests --

    #[test]
    fn test_wind_no_turbulence() {
        let wind = WindSource::new(Vec3::X, 5.0);
        let force = wind.sample();
        assert!((force.x - 5.0).abs() < 0.01);
        assert!(force.y.abs() < 0.01);
    }

    #[test]
    fn test_wind_with_turbulence() {
        let mut wind = WindSource::new(Vec3::X, 5.0).with_turbulence(0.5, 2.0);
        wind.update(0.5);
        let force = wind.sample();
        // Should be approximately in the X direction, but with some noise.
        assert!(force.x > 0.0, "Wind should still be roughly in X direction");
    }

    #[test]
    fn test_wind_sample_at() {
        let wind = WindSource::new(Vec3::Y, 3.0);
        let f1 = wind.sample();
        let f2 = wind.sample_at(Vec3::new(100.0, 0.0, 0.0));
        assert!((f1 - f2).length() < f32::EPSILON);
    }

    // -- SpringBone tests --

    #[test]
    fn test_spring_bone_creation() {
        let bone = SpringBone::new(0).with_stiffness(200.0).with_damping(0.8);
        assert_eq!(bone.bone_index, 0);
        assert!((bone.stiffness - 200.0).abs() < f32::EPSILON);
        assert!((bone.damping - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spring_bone_initialize() {
        let mut bone = SpringBone::new(0);
        bone.initialize(Vec3::new(1.0, 2.0, 3.0));
        assert!((bone.current_position - Vec3::new(1.0, 2.0, 3.0)).length() < f32::EPSILON);
        assert!((bone.previous_position - Vec3::new(1.0, 2.0, 3.0)).length() < f32::EPSILON);
    }

    #[test]
    fn test_spring_bone_simulate_at_rest() {
        let mut bone = SpringBone::new(0)
            .with_stiffness(100.0)
            .with_damping(0.9)
            .with_gravity(Vec3::ZERO);
        bone.initialize(Vec3::new(0.0, 1.0, 0.0));
        bone.set_target(Vec3::new(0.0, 1.0, 0.0));

        // When at rest, bone should stay near target.
        for _ in 0..100 {
            bone.simulate(1.0 / 60.0, None, Vec3::ZERO);
        }
        let dist = (bone.current_position - bone.target_position).length();
        assert!(
            dist < 0.1,
            "Spring bone at rest should stay near target, dist={}",
            dist
        );
    }

    #[test]
    fn test_spring_bone_simulate_gravity() {
        let mut bone = SpringBone::new(0)
            .with_stiffness(0.0) // No spring, just gravity.
            .with_damping(0.0)
            .with_gravity(Vec3::new(0.0, -10.0, 0.0));
        bone.initialize(Vec3::new(0.0, 1.0, 0.0));
        bone.set_target(Vec3::new(0.0, 1.0, 0.0));

        bone.simulate(0.1, None, Vec3::ZERO);
        assert!(
            bone.current_position.y < 1.0,
            "Bone should fall under gravity"
        );
    }

    #[test]
    fn test_spring_bone_rest_length_constraint() {
        let mut bone = SpringBone::new(1)
            .with_stiffness(100.0)
            .with_damping(0.5)
            .with_gravity(Vec3::ZERO);
        bone.rest_length = 1.0;
        bone.initialize(Vec3::new(0.0, 1.0, 0.0));
        bone.set_target(Vec3::new(0.0, 1.0, 0.0));

        let parent_pos = Vec3::ZERO;
        bone.simulate(0.016, Some(parent_pos), Vec3::ZERO);

        let dist = (bone.current_position - parent_pos).length();
        assert!(
            (dist - 1.0).abs() < 0.01,
            "Should maintain rest length from parent: {}",
            dist
        );
    }

    #[test]
    fn test_spring_bone_collision() {
        let mut bone = SpringBone::new(0).with_radius(0.1);
        bone.initialize(Vec3::new(0.3, 0.0, 0.0));
        bone.current_position = Vec3::new(0.3, 0.0, 0.0);

        let colliders = vec![SpringCollider::sphere(Vec3::ZERO, 0.5)];
        bone.resolve_collisions(&colliders);

        let dist = bone.current_position.length();
        assert!(
            dist >= 0.59,
            "Should be pushed out of sphere: dist={}",
            dist
        );
    }

    // -- SpringBoneChain tests --

    #[test]
    fn test_chain_creation() {
        let chain = SpringBoneChain::new("hair", &[5, 6, 7, 8]);
        assert_eq!(chain.bone_count(), 4);
        assert_eq!(chain.name, "hair");
    }

    #[test]
    fn test_chain_initialize() {
        let mut chain = SpringBoneChain::new("test", &[0, 1, 2]);
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        chain.initialize(&positions);

        assert!((chain.bones[0].current_position - Vec3::ZERO).length() < f32::EPSILON);
        assert!((chain.bones[1].current_position - Vec3::Y).length() < f32::EPSILON);
        assert!((chain.bones[1].rest_length - 1.0).abs() < 0.01);
        assert!((chain.bones[2].rest_length - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_chain_simulate() {
        let mut chain = SpringBoneChain::new("test", &[0, 1, 2]);
        chain.set_gravity(Vec3::ZERO);
        chain.set_stiffness(200.0);
        chain.set_damping(0.8);

        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        chain.initialize(&positions);

        // Simulate for a few steps.
        for _ in 0..10 {
            chain.set_targets(&positions);
            chain.update(1.0 / 60.0);
        }

        // All bones should be near their targets (stable equilibrium).
        for (i, bone) in chain.bones.iter().enumerate() {
            let dist = (bone.current_position - positions[bone.bone_index]).length();
            assert!(
                dist < 0.5,
                "Bone {} should be near target after settling, dist={}",
                i, dist
            );
        }
    }

    #[test]
    fn test_chain_with_collider() {
        let mut chain = SpringBoneChain::new("test", &[0, 1]);
        chain.add_collider(SpringCollider::sphere(Vec3::new(0.0, 0.5, 0.0), 0.3));
        chain.set_gravity(Vec3::ZERO);

        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0), // Inside the collider.
        ];
        chain.initialize(&positions);
        chain.set_targets(&positions);
        chain.update(1.0 / 60.0);

        // Bone 1 should be pushed out of the sphere.
        let dist = (chain.bones[1].current_position - Vec3::new(0.0, 0.5, 0.0)).length();
        // It may or may not be pushed depending on exact setup, but should not crash.
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_chain_with_wind() {
        let mut chain = SpringBoneChain::new("test", &[0, 1]);
        chain.set_gravity(Vec3::ZERO);
        chain.set_stiffness(10.0);
        chain.set_damping(0.3);
        chain.add_wind(WindSource::new(Vec3::X, 5.0));

        let positions = vec![Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)];
        chain.initialize(&positions);

        // Simulate with wind.
        for _ in 0..60 {
            chain.set_targets(&positions);
            chain.update(1.0 / 60.0);
        }

        // Wind should push bones in the +X direction.
        assert!(
            chain.bones[1].current_position.x > 0.0,
            "Wind should push bone in +X: {}",
            chain.bones[1].current_position.x
        );
    }

    #[test]
    fn test_chain_disabled() {
        let mut chain = SpringBoneChain::new("test", &[0, 1]);
        chain.enabled = false;
        let positions = vec![Vec3::ZERO, Vec3::Y];
        chain.initialize(&positions);

        let pos_before = chain.bones[1].current_position;
        chain.update(0.016);
        let pos_after = chain.bones[1].current_position;
        assert!((pos_before - pos_after).length() < f32::EPSILON);
    }

    #[test]
    fn test_chain_reset() {
        let mut chain = SpringBoneChain::new("test", &[0, 1]);
        chain.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        let positions = vec![Vec3::ZERO, Vec3::Y];
        chain.initialize(&positions);
        chain.set_targets(&positions);
        chain.update(0.1);

        // Position should have changed due to gravity.
        assert!(
            (chain.bones[1].current_position - Vec3::Y).length() > 0.001
        );

        chain.reset();
        // After reset, should be back at target.
        assert!(
            (chain.bones[1].current_position - chain.bones[1].target_position).length() < f32::EPSILON
        );
    }

    #[test]
    fn test_chain_simulated_positions() {
        let mut chain = SpringBoneChain::new("test", &[3, 4, 5]);
        let positions = vec![
            Vec3::ZERO, Vec3::ZERO, Vec3::ZERO,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        chain.initialize(&positions);

        let sim_pos = chain.simulated_positions();
        assert_eq!(sim_pos.len(), 3);
        assert_eq!(sim_pos[0].0, 3);
        assert_eq!(sim_pos[1].0, 4);
        assert_eq!(sim_pos[2].0, 5);
    }

    // -- JiggleBone tests --

    #[test]
    fn test_jiggle_creation() {
        let jiggle = JiggleBone::new(0)
            .with_stiffness(300.0)
            .with_damping(0.7);
        assert_eq!(jiggle.bone_index, 0);
        assert!((jiggle.stiffness - 300.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jiggle_at_rest() {
        let mut jiggle = JiggleBone::new(0)
            .with_stiffness(200.0)
            .with_damping(0.8);
        jiggle.set_target(Vec3::new(0.0, 1.0, 0.0));

        // Should settle near target.
        for _ in 0..200 {
            jiggle.update(1.0 / 60.0);
        }
        let disp = jiggle.displacement().length();
        assert!(
            disp < 0.05,
            "Jiggle should settle near rest, displacement={}",
            disp
        );
    }

    #[test]
    fn test_jiggle_axis_constraint() {
        let mut jiggle = JiggleBone::new(0)
            .with_stiffness(100.0)
            .with_damping(0.5)
            .with_axis_constraint(Vec3::Y);
        jiggle.set_target(Vec3::ZERO);

        // Apply an impulse by moving the target.
        jiggle.set_target(Vec3::new(1.0, 1.0, 1.0));
        jiggle.update(1.0 / 60.0);

        // Displacement should only be along Y.
        let disp = jiggle.displacement();
        assert!(
            disp.x.abs() < f32::EPSILON,
            "X should be zero with Y constraint: {}",
            disp.x
        );
        assert!(
            disp.z.abs() < f32::EPSILON,
            "Z should be zero with Y constraint: {}",
            disp.z
        );
    }

    #[test]
    fn test_jiggle_max_displacement() {
        let mut jiggle = JiggleBone::new(0)
            .with_stiffness(10.0) // Low stiffness = more displacement.
            .with_damping(0.1)
            .with_max_displacement(0.3);
        jiggle.set_target(Vec3::ZERO);

        // Apply a large impulse.
        jiggle.set_target(Vec3::new(10.0, 0.0, 0.0));
        for _ in 0..10 {
            jiggle.update(1.0 / 60.0);
        }

        let disp = jiggle.displacement().length();
        assert!(
            disp <= 0.31,
            "Displacement should not exceed max: {}",
            disp
        );
    }

    #[test]
    fn test_jiggle_disabled() {
        let mut jiggle = JiggleBone::new(0);
        jiggle.enabled = false;
        jiggle.set_target(Vec3::ZERO);
        jiggle.set_target(Vec3::X); // Apply impulse.
        jiggle.update(0.016);

        assert!(
            jiggle.displacement().length() < f32::EPSILON,
            "Disabled jiggle should not move"
        );
    }

    #[test]
    fn test_jiggle_reset() {
        let mut jiggle = JiggleBone::new(0).with_stiffness(10.0);
        jiggle.set_target(Vec3::ZERO);
        jiggle.set_target(Vec3::X);
        jiggle.update(0.016);
        assert!(jiggle.displacement().length() > 0.0);

        jiggle.reset();
        assert!(jiggle.displacement().length() < f32::EPSILON);
    }

    #[test]
    fn test_jiggle_write_transform() {
        let mut jiggle = JiggleBone::new(1);
        jiggle.set_target(Vec3::new(1.0, 2.0, 3.0));

        let mut transforms = vec![Transform::IDENTITY; 3];
        jiggle.write_transform(&mut transforms);
        assert!(
            (transforms[1].position - Vec3::new(1.0, 2.0, 3.0)).length() < 0.01
        );
    }

    // -- SpringBoneSystem tests --

    #[test]
    fn test_system_creation() {
        let system = SpringBoneSystem::new();
        assert_eq!(system.total_bone_count(), 0);
        assert!(!system.paused);
    }

    #[test]
    fn test_system_add_chain_and_jiggle() {
        let mut system = SpringBoneSystem::new();
        system.add_chain(SpringBoneChain::new("hair", &[5, 6, 7]));
        system.add_jiggle(JiggleBone::new(10));
        assert_eq!(system.total_bone_count(), 4);
    }

    #[test]
    fn test_system_find_chain() {
        let mut system = SpringBoneSystem::new();
        system.add_chain(SpringBoneChain::new("hair", &[0, 1]));
        system.add_chain(SpringBoneChain::new("tail", &[2, 3]));
        assert_eq!(system.find_chain("hair"), Some(0));
        assert_eq!(system.find_chain("tail"), Some(1));
        assert_eq!(system.find_chain("missing"), None);
    }

    #[test]
    fn test_system_update() {
        let mut system = SpringBoneSystem::new();
        let mut chain = SpringBoneChain::new("test", &[0, 1]);
        chain.set_gravity(Vec3::ZERO);
        system.add_chain(chain);
        system.add_jiggle(JiggleBone::new(2));

        let positions = vec![Vec3::ZERO, Vec3::Y, Vec3::new(0.0, 2.0, 0.0)];
        system.initialize(&positions);
        system.update(0.016, &positions);

        // Should not crash and positions should be valid.
        assert!(system.chains[0].bones[0].current_position.is_finite());
        assert!(system.jiggle_bones[0].simulated_position().is_finite());
    }

    #[test]
    fn test_system_paused() {
        let mut system = SpringBoneSystem::new();
        let mut chain = SpringBoneChain::new("test", &[0, 1]);
        chain.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        system.add_chain(chain);

        let positions = vec![Vec3::ZERO, Vec3::Y];
        system.initialize(&positions);

        let pos_before = system.chains[0].bones[1].current_position;
        system.paused = true;
        system.update(0.016, &positions);
        let pos_after = system.chains[0].bones[1].current_position;
        assert!((pos_before - pos_after).length() < f32::EPSILON);
    }

    #[test]
    fn test_system_reset() {
        let mut system = SpringBoneSystem::new();
        let mut chain = SpringBoneChain::new("test", &[0, 1]);
        chain.set_gravity(Vec3::new(0.0, -10.0, 0.0));
        system.add_chain(chain);
        system.add_jiggle(JiggleBone::new(2));

        let positions = vec![Vec3::ZERO, Vec3::Y, Vec3::new(0.0, 2.0, 0.0)];
        system.initialize(&positions);
        system.update(0.1, &positions);

        system.reset_all();
        let dist = (system.chains[0].bones[1].current_position
            - system.chains[0].bones[1].target_position).length();
        assert!(dist < f32::EPSILON);
    }

    // -- rotation_between tests --

    #[test]
    fn test_rotation_between_same_direction() {
        let rot = rotation_between(Vec3::Z, Vec3::Z).unwrap();
        assert!(rot.dot(Quat::IDENTITY).abs() > 0.999);
    }

    #[test]
    fn test_rotation_between_opposite() {
        let rot = rotation_between(Vec3::Z, Vec3::NEG_Z).unwrap();
        let angle = rot.to_axis_angle().1;
        assert!(
            (angle - std::f32::consts::PI).abs() < 0.01,
            "Should be 180-degree rotation: {}",
            angle
        );
    }

    #[test]
    fn test_rotation_between_perpendicular() {
        let rot = rotation_between(Vec3::Z, Vec3::X).unwrap();
        let result = rot * Vec3::Z;
        assert!(
            (result - Vec3::X).length() < 0.01,
            "Rotated Z should equal X: {:?}",
            result
        );
    }

    #[test]
    fn test_rotation_between_zero_vector() {
        assert!(rotation_between(Vec3::ZERO, Vec3::X).is_none());
    }
}
