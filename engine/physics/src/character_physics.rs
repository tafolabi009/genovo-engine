//! Physics-based character controller.
//!
//! Unlike the gameplay-level character controller in the `gameplay` crate, this
//! module implements the low-level physics interactions: capsule sweeps, slope
//! handling, step detection, ground queries, and moving-platform tracking.
//!
//! The main entry point is [`move_character`], which takes a desired velocity
//! and resolves it against the physics world geometry using iterative slide-and-
//! clip projection (up to [`MAX_CLIP_ITERATIONS`] iterations).
//!
//! ## Capsule representation
//!
//! Characters are modeled as a [`CharacterCapsule`] — two hemispheres joined by
//! a cylinder — which provides the smoothest collision response and avoids edge
//! catching. The capsule is always oriented along the world Y axis.
//!
//! ## Ground detection
//!
//! [`GroundQuery`] uses a combination of downward raycasts and sphere-casts to
//! robustly determine whether the character is grounded, what the surface normal
//! is, and which surface material is underfoot. This information feeds slope
//! handling (slope angle > `max_slope` → character slides) and step detection.

use glam::Vec3;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of slide-clip iterations per move.
pub const MAX_CLIP_ITERATIONS: usize = 4;

/// Small epsilon for floating-point comparisons in sweep tests.
const EPSILON: f32 = 1e-6;

/// Minimum distance to maintain from geometry (contact offset / skin width).
const DEFAULT_SKIN_WIDTH: f32 = 0.01;

/// Default maximum slope angle in radians (~46 degrees).
const DEFAULT_MAX_SLOPE: f32 = 0.8028; // ~46 degrees

/// Default step height in meters.
const DEFAULT_STEP_HEIGHT: f32 = 0.35;

/// Gravity magnitude used for slope sliding when no external gravity is given.
const DEFAULT_GRAVITY: f32 = 9.81;

/// Downward cast distance for ground probing.
const GROUND_PROBE_DISTANCE: f32 = 0.15;

/// Threshold dot product between ground normal and up vector to be "on ground".
const GROUND_DOT_THRESHOLD: f32 = 0.45;

/// Maximum ground-snap distance (prevents flying off small bumps).
const MAX_SNAP_DISTANCE: f32 = 0.3;

/// Speed of light for clamping absurd velocities.
const MAX_VELOCITY: f32 = 200.0;

// ---------------------------------------------------------------------------
// Surface material identifier
// ---------------------------------------------------------------------------

/// Identifies the physical surface material at a contact point.
///
/// The physics world tags colliders with a material index; the gameplay layer
/// maps this to footstep sounds, friction overrides, and VFX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SurfaceMaterial(pub u32);

impl SurfaceMaterial {
    /// Sentinel value meaning "no material / air".
    pub const NONE: Self = Self(u32::MAX);

    /// Whether this is a valid (non-sentinel) material.
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

// ---------------------------------------------------------------------------
// CharacterCapsule
// ---------------------------------------------------------------------------

/// An upright capsule shape used for character collisions.
///
/// The capsule is always aligned along the world Y axis:
///
/// ```text
///      ___
///     /   \        ← top hemisphere  (center at base_pos + height - radius)
///    |     |
///    |     |       ← cylinder
///    |     |
///     \___/        ← bottom hemisphere (center at base_pos + radius)
/// ```
///
/// `total_height = 2 * half_height + 2 * radius`
#[derive(Debug, Clone, Copy)]
pub struct CharacterCapsule {
    /// Radius of the hemispheres and cylinder.
    pub radius: f32,
    /// Half the height of the cylindrical section (excludes hemispheres).
    pub half_height: f32,
}

impl CharacterCapsule {
    /// Creates a capsule from radius and total height.
    ///
    /// The cylindrical section height is `total_height - 2*radius`. If the
    /// total height is less than `2*radius`, the capsule degenerates into a
    /// sphere.
    pub fn new(radius: f32, total_height: f32) -> Self {
        let half_height = ((total_height - 2.0 * radius) * 0.5).max(0.0);
        Self {
            radius,
            half_height,
        }
    }

    /// Total height of the capsule (including hemispheres).
    #[inline]
    pub fn total_height(&self) -> f32 {
        2.0 * self.half_height + 2.0 * self.radius
    }

    /// World-space center of the top hemisphere given the capsule's base position.
    #[inline]
    pub fn top_center(&self, base: Vec3) -> Vec3 {
        Vec3::new(base.x, base.y + self.radius + 2.0 * self.half_height, base.z)
    }

    /// World-space center of the bottom hemisphere given the capsule's base position.
    #[inline]
    pub fn bottom_center(&self, base: Vec3) -> Vec3 {
        Vec3::new(base.x, base.y + self.radius, base.z)
    }

    /// World-space center of the capsule (midpoint of the segment).
    #[inline]
    pub fn center(&self, base: Vec3) -> Vec3 {
        Vec3::new(
            base.x,
            base.y + self.radius + self.half_height,
            base.z,
        )
    }

    /// Performs a capsule sweep test against a triangle in the physics world.
    ///
    /// The sweep moves the capsule from `from` to `to` (both are *base*
    /// positions). Returns the first hit, if any.
    ///
    /// The implementation reduces the capsule-vs-triangle sweep to a
    /// sphere-vs-swept-prism problem: we shrink the triangle by the capsule
    /// radius along its normal, then sweep the capsule's inner segment (a
    /// point or line) against the inflated geometry.
    pub fn sweep_test_triangle(
        &self,
        from: Vec3,
        to: Vec3,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
    ) -> Option<SweepHit> {
        let direction = to - from;
        let move_len = direction.length();
        if move_len < EPSILON {
            return None;
        }
        let dir = direction / move_len;

        // Capsule segment endpoints at the start position.
        let seg_a = self.bottom_center(from);
        let seg_b = self.top_center(from);

        // Triangle normal.
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let tri_normal = edge1.cross(edge2);
        let tri_normal_len = tri_normal.length();
        if tri_normal_len < EPSILON {
            return None;
        }
        let tri_normal = tri_normal / tri_normal_len;

        // --- Test each sphere on the capsule segment against the triangle ---
        // We test the bottom and top hemispheres independently and take the
        // earliest hit.
        let mut best: Option<SweepHit> = None;

        for &sphere_center in &[seg_a, seg_b] {
            if let Some(hit) = sweep_sphere_triangle(
                sphere_center,
                dir,
                self.radius,
                v0,
                v1,
                v2,
                tri_normal,
                move_len,
            ) {
                if best.is_none() || hit.t < best.as_ref().unwrap().t {
                    best = Some(hit);
                }
            }
        }

        // --- Test cylinder section (edge sweeps) ---
        // Sweep the capsule's inner segment against each triangle edge,
        // treated as a capsule-vs-capsule (edge is a zero-radius capsule).
        let tri_edges = [
            (v0, v1),
            (v1, v2),
            (v2, v0),
        ];
        for &(ea, eb) in &tri_edges {
            if let Some(hit) = sweep_capsule_edge(
                seg_a, seg_b, self.radius, dir, ea, eb, move_len,
            ) {
                if best.is_none() || hit.t < best.as_ref().unwrap().t {
                    best = Some(hit);
                }
            }
        }

        // --- Test capsule segment against triangle vertices ---
        for &vert in &[v0, v1, v2] {
            if let Some(hit) = sweep_capsule_point(
                seg_a, seg_b, self.radius, dir, vert, move_len,
            ) {
                if best.is_none() || hit.t < best.as_ref().unwrap().t {
                    best = Some(hit);
                }
            }
        }

        best
    }

    /// Performs a capsule sweep against an axis-aligned bounding box.
    ///
    /// Reduces to a Minkowski-expanded AABB (rounded box) ray test.
    pub fn sweep_test_aabb(
        &self,
        from: Vec3,
        to: Vec3,
        aabb_min: Vec3,
        aabb_max: Vec3,
    ) -> Option<SweepHit> {
        let direction = to - from;
        let move_len = direction.length();
        if move_len < EPSILON {
            return None;
        }
        let dir = direction / move_len;

        // Minkowski sum: expand the AABB by the capsule radius.
        let expanded_min = aabb_min - Vec3::splat(self.radius);
        let expanded_max = aabb_max + Vec3::splat(self.radius);

        // Test both hemisphere centers.
        let seg_a = self.bottom_center(from);
        let seg_b = self.top_center(from);

        let mut best: Option<SweepHit> = None;
        for &center in &[seg_a, seg_b] {
            if let Some(t) = ray_vs_aabb(center, dir, expanded_min, expanded_max, move_len) {
                let point = center + dir * t;
                // Compute normal from the closest face.
                let normal = aabb_contact_normal(point, aabb_min, aabb_max);
                let hit = SweepHit {
                    t,
                    point,
                    normal,
                    surface_material: SurfaceMaterial::NONE,
                };
                if best.is_none() || t < best.as_ref().unwrap().t {
                    best = Some(hit);
                }
            }
        }

        best
    }

    /// Performs a capsule sweep against a sphere collider.
    pub fn sweep_test_sphere(
        &self,
        from: Vec3,
        to: Vec3,
        sphere_center: Vec3,
        sphere_radius: f32,
    ) -> Option<SweepHit> {
        let direction = to - from;
        let move_len = direction.length();
        if move_len < EPSILON {
            return None;
        }
        let dir = direction / move_len;
        let combined_radius = self.radius + sphere_radius;

        // Test both hemisphere centers.
        let seg_a = self.bottom_center(from);
        let seg_b = self.top_center(from);

        let mut best: Option<SweepHit> = None;
        for &center in &[seg_a, seg_b] {
            if let Some(t) = sweep_sphere_vs_point(center, dir, combined_radius, sphere_center, move_len) {
                let swept_pos = center + dir * t;
                let diff = swept_pos - sphere_center;
                let dist = diff.length();
                let normal = if dist > EPSILON { diff / dist } else { Vec3::Y };
                let point = sphere_center + normal * sphere_radius;
                let hit = SweepHit {
                    t,
                    point,
                    normal,
                    surface_material: SurfaceMaterial::NONE,
                };
                if best.is_none() || t < best.as_ref().unwrap().t {
                    best = Some(hit);
                }
            }
        }

        // Also test the cylinder's inner segment vs the sphere as a capsule-sphere test.
        if let Some(hit) = sweep_capsule_point(seg_a, seg_b, combined_radius, dir, sphere_center, move_len) {
            if best.is_none() || hit.t < best.as_ref().unwrap().t {
                best = Some(hit);
            }
        }

        best
    }

    /// Performs a capsule sweep against another capsule.
    pub fn sweep_test_capsule(
        &self,
        from: Vec3,
        to: Vec3,
        other_base: Vec3,
        other: &CharacterCapsule,
    ) -> Option<SweepHit> {
        let direction = to - from;
        let move_len = direction.length();
        if move_len < EPSILON {
            return None;
        }
        let dir = direction / move_len;
        let combined_radius = self.radius + other.radius;

        let my_a = self.bottom_center(from);
        let my_b = self.top_center(from);
        let oth_a = other.bottom_center(other_base);
        let oth_b = other.top_center(other_base);

        let mut best: Option<SweepHit> = None;

        // Sphere-sphere tests for all endpoint pairs.
        for &my_pt in &[my_a, my_b] {
            for &oth_pt in &[oth_a, oth_b] {
                if let Some(t) = sweep_sphere_vs_point(my_pt, dir, combined_radius, oth_pt, move_len) {
                    let swept = my_pt + dir * t;
                    let diff = swept - oth_pt;
                    let dist = diff.length();
                    let normal = if dist > EPSILON { diff / dist } else { Vec3::Y };
                    let point = oth_pt + normal * other.radius;
                    let hit = SweepHit { t, point, normal, surface_material: SurfaceMaterial::NONE };
                    if best.is_none() || t < best.as_ref().unwrap().t {
                        best = Some(hit);
                    }
                }
            }
        }

        // Edge-edge test: my capsule segment vs other capsule segment.
        if let Some(hit) = sweep_capsule_edge(my_a, my_b, combined_radius, dir, oth_a, oth_b, move_len) {
            if best.is_none() || hit.t < best.as_ref().unwrap().t {
                best = Some(hit);
            }
        }

        best
    }
}

impl Default for CharacterCapsule {
    fn default() -> Self {
        Self::new(0.3, 1.8) // 0.3m radius, 1.8m tall
    }
}

// ---------------------------------------------------------------------------
// SweepHit
// ---------------------------------------------------------------------------

/// Result of a capsule sweep test against geometry.
#[derive(Debug, Clone, Copy)]
pub struct SweepHit {
    /// Parametric hit distance in \[0, move_length\].
    pub t: f32,
    /// World-space contact point.
    pub point: Vec3,
    /// Surface normal at the contact point (pointing away from the surface).
    pub normal: Vec3,
    /// Surface material of the hit geometry.
    pub surface_material: SurfaceMaterial,
}

// ---------------------------------------------------------------------------
// PhysicsCharacter
// ---------------------------------------------------------------------------

/// Physics-level character controller configuration.
///
/// This is the physical representation of a character — capsule shape, mass,
/// movement constraints. It does **not** store gameplay state (health, inventory,
/// etc.); that belongs in the `gameplay` crate.
#[derive(Debug, Clone)]
pub struct PhysicsCharacter {
    /// Capsule shape for collisions.
    pub capsule: CharacterCapsule,
    /// Character mass in kg (affects push-back against dynamic bodies).
    pub mass: f32,
    /// Maximum climbable step height (meters).
    pub step_height: f32,
    /// Maximum walkable slope angle (radians from vertical).
    pub max_slope: f32,
    /// Contact offset / skin width — minimum separation distance maintained
    /// from geometry to prevent numerical tunneling.
    pub skin_width: f32,
    /// Current world-space position (base of capsule — feet).
    pub position: Vec3,
    /// Current velocity (updated by `move_character`).
    pub velocity: Vec3,
    /// Unique handle for this character (for external lookup).
    pub handle: CharacterHandle,
    /// Whether the character is currently on the ground.
    pub grounded: bool,
    /// Normal of the ground surface (valid only when `grounded` is true).
    pub ground_normal: Vec3,
    /// Surface material of the ground (valid only when `grounded` is true).
    pub ground_material: SurfaceMaterial,
    /// Whether the character is standing on a moving platform.
    pub on_moving_platform: bool,
    /// Handle of the platform body, if on one.
    pub platform_handle: Option<PlatformHandle>,
    /// Velocity of the platform the character is standing on.
    pub platform_velocity: Vec3,
    /// Angular velocity of the platform.
    pub platform_angular_velocity: Vec3,
    /// Point on the platform where the character is standing (local to platform).
    pub platform_contact_local: Vec3,
    /// Gravity scale (1.0 = normal, 0.0 = no gravity, 2.0 = double).
    pub gravity_scale: f32,
    /// Whether the character can push dynamic bodies.
    pub push_bodies: bool,
    /// Push force multiplier.
    pub push_force: f32,
}

/// Opaque handle to a character controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CharacterHandle(pub u64);

/// Opaque handle to a platform body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlatformHandle(pub u64);

impl Default for PhysicsCharacter {
    fn default() -> Self {
        Self {
            capsule: CharacterCapsule::default(),
            mass: 80.0,
            step_height: DEFAULT_STEP_HEIGHT,
            max_slope: DEFAULT_MAX_SLOPE,
            skin_width: DEFAULT_SKIN_WIDTH,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            handle: CharacterHandle(0),
            grounded: false,
            ground_normal: Vec3::Y,
            ground_material: SurfaceMaterial::NONE,
            on_moving_platform: false,
            platform_handle: None,
            platform_velocity: Vec3::ZERO,
            platform_angular_velocity: Vec3::ZERO,
            platform_contact_local: Vec3::ZERO,
            gravity_scale: 1.0,
            push_bodies: true,
            push_force: 50.0,
        }
    }
}

impl PhysicsCharacter {
    /// Creates a new character with the given capsule dimensions.
    pub fn new(radius: f32, height: f32) -> Self {
        Self {
            capsule: CharacterCapsule::new(radius, height),
            ..Default::default()
        }
    }

    /// Creates a new character with a specific handle.
    pub fn with_handle(mut self, handle: CharacterHandle) -> Self {
        self.handle = handle;
        self
    }

    /// Creates a new character at a specific position.
    pub fn with_position(mut self, position: Vec3) -> Self {
        self.position = position;
        self
    }

    /// Sets the step height.
    pub fn with_step_height(mut self, step_height: f32) -> Self {
        self.step_height = step_height;
        self
    }

    /// Sets the max slope angle (radians).
    pub fn with_max_slope(mut self, max_slope: f32) -> Self {
        self.max_slope = max_slope;
        self
    }
}

// ---------------------------------------------------------------------------
// CharacterMoveResult
// ---------------------------------------------------------------------------

/// Result of a [`move_character`] call.
#[derive(Debug, Clone)]
pub struct CharacterMoveResult {
    /// New position after the move.
    pub position: Vec3,
    /// New velocity after the move (may differ from input due to collisions).
    pub velocity: Vec3,
    /// Whether the character is on the ground after the move.
    pub grounded: bool,
    /// Ground normal (valid only if `grounded`).
    pub ground_normal: Vec3,
    /// Ground surface material (valid only if `grounded`).
    pub ground_material: SurfaceMaterial,
    /// Collisions that occurred during the move.
    pub collisions: Vec<CharacterCollision>,
    /// Whether a step-up occurred during this move.
    pub stepped_up: bool,
    /// Whether the character slid on a steep slope.
    pub sliding_on_slope: bool,
    /// Platform velocity applied (zero if not on a platform).
    pub platform_velocity: Vec3,
}

/// A collision that occurred during character movement.
#[derive(Debug, Clone, Copy)]
pub struct CharacterCollision {
    /// World-space contact point.
    pub point: Vec3,
    /// Contact normal (pointing away from the obstacle).
    pub normal: Vec3,
    /// Penetration depth (positive means overlapping).
    pub depth: f32,
    /// Surface material of the collided geometry.
    pub surface_material: SurfaceMaterial,
    /// Whether this collision was with the ground plane.
    pub is_ground: bool,
    /// Whether this collision was with a wall.
    pub is_wall: bool,
    /// Whether this collision was with a ceiling.
    pub is_ceiling: bool,
}

impl CharacterCollision {
    /// Classifies the collision based on the surface normal.
    pub fn classify(normal: Vec3, max_slope: f32) -> (bool, bool, bool) {
        let up_dot = normal.dot(Vec3::Y);
        let slope_cos = max_slope.cos();
        let is_ground = up_dot >= slope_cos;
        let is_ceiling = up_dot <= -slope_cos;
        let is_wall = !is_ground && !is_ceiling;
        (is_ground, is_wall, is_ceiling)
    }
}

// ---------------------------------------------------------------------------
// GroundInfo / GroundQuery
// ---------------------------------------------------------------------------

/// Information about the ground beneath the character.
#[derive(Debug, Clone, Copy)]
pub struct GroundInfo {
    /// Whether a ground surface was detected.
    pub hit: bool,
    /// Ground surface normal (valid only if `hit`).
    pub normal: Vec3,
    /// Distance from the character's feet to the ground surface.
    pub distance: f32,
    /// Surface material of the ground.
    pub surface_material: SurfaceMaterial,
    /// Slope angle in radians (0 = flat, PI/2 = vertical wall).
    pub slope_angle: f32,
    /// Whether the slope is walkable (angle <= max_slope).
    pub walkable: bool,
}

impl Default for GroundInfo {
    fn default() -> Self {
        Self {
            hit: false,
            normal: Vec3::Y,
            distance: f32::INFINITY,
            surface_material: SurfaceMaterial::NONE,
            slope_angle: 0.0,
            walkable: false,
        }
    }
}

/// Provides robust ground detection using combined raycast and sphere-cast.
///
/// A single downward raycast can miss edges and sharp geometry. By also
/// performing a sphere-cast (sweeping a small sphere downward), we catch
/// geometry that the ray would slip between.
pub struct GroundQuery;

impl GroundQuery {
    /// Probes for ground beneath the character.
    ///
    /// Casts a ray and a sphere downward from the character's feet position
    /// and returns combined ground information.
    ///
    /// # Arguments
    /// * `colliders` — the static/dynamic geometry to test against
    /// * `position` — character's base (feet) position
    /// * `capsule_radius` — radius of the character capsule
    /// * `skin_width` — contact offset
    /// * `max_slope` — maximum walkable slope in radians
    /// * `probe_distance` — how far down to look
    pub fn is_grounded(
        colliders: &[StaticCollider],
        position: Vec3,
        capsule_radius: f32,
        skin_width: f32,
        max_slope: f32,
        probe_distance: f32,
    ) -> GroundInfo {
        let ray_origin = position + Vec3::new(0.0, skin_width + 0.01, 0.0);
        let ray_dir = -Vec3::Y;
        let max_dist = probe_distance + skin_width + 0.01;

        // --- Raycast ---
        let ray_hit = raycast_colliders(colliders, ray_origin, ray_dir, max_dist);

        // --- Sphere cast (small sphere swept downward) ---
        let sphere_radius = capsule_radius * 0.5;
        let sphere_origin = position + Vec3::new(0.0, sphere_radius + skin_width + 0.01, 0.0);
        let sphere_hit = spherecast_colliders(
            colliders,
            sphere_origin,
            ray_dir,
            sphere_radius,
            max_dist,
        );

        // Combine: prefer the closest hit from either method.
        let (best_normal, best_dist, best_material, found) =
            match (ray_hit, sphere_hit) {
                (Some(rh), Some(sh)) => {
                    if rh.distance <= sh.distance {
                        (rh.normal, rh.distance, rh.surface_material, true)
                    } else {
                        (sh.normal, sh.distance, sh.surface_material, true)
                    }
                }
                (Some(rh), None) => (rh.normal, rh.distance, rh.surface_material, true),
                (None, Some(sh)) => (sh.normal, sh.distance, sh.surface_material, true),
                (None, None) => (Vec3::Y, f32::INFINITY, SurfaceMaterial::NONE, false),
            };

        if !found {
            return GroundInfo::default();
        }

        let slope_angle = best_normal.dot(Vec3::Y).clamp(-1.0, 1.0).acos();
        let walkable = slope_angle <= max_slope;

        GroundInfo {
            hit: true,
            normal: best_normal,
            distance: best_dist,
            surface_material: best_material,
            slope_angle,
            walkable,
        }
    }

    /// Snap the character to the ground if they are within `snap_distance`.
    ///
    /// Useful at the end of a move to keep the character firmly on the ground
    /// when walking over small bumps or descending slopes.
    pub fn snap_to_ground(
        colliders: &[StaticCollider],
        position: Vec3,
        capsule_radius: f32,
        skin_width: f32,
        max_slope: f32,
        snap_distance: f32,
    ) -> Option<Vec3> {
        let info = Self::is_grounded(
            colliders,
            position,
            capsule_radius,
            skin_width,
            max_slope,
            snap_distance,
        );
        if info.hit && info.walkable && info.distance <= snap_distance {
            Some(Vec3::new(position.x, position.y - info.distance, position.z))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Static collider (simplified world geometry for character queries)
// ---------------------------------------------------------------------------

/// Simplified collider representation used by the character controller for
/// sweep tests.
///
/// In a full engine, this would be an interface into `PhysicsWorld`; here we
/// provide a self-contained representation so the character controller can
/// be tested independently.
#[derive(Debug, Clone)]
pub enum StaticCollider {
    /// Triangle soup (mesh collider).
    TriMesh {
        vertices: Vec<Vec3>,
        indices: Vec<[u32; 3]>,
        material: SurfaceMaterial,
    },
    /// Axis-aligned box.
    AABB {
        min: Vec3,
        max: Vec3,
        material: SurfaceMaterial,
    },
    /// Sphere collider.
    Sphere {
        center: Vec3,
        radius: f32,
        material: SurfaceMaterial,
    },
    /// Infinite plane (normal + distance from origin).
    Plane {
        normal: Vec3,
        distance: f32,
        material: SurfaceMaterial,
    },
    /// Another capsule (e.g., another character or a pillar).
    Capsule {
        base: Vec3,
        capsule: CharacterCapsule,
        material: SurfaceMaterial,
    },
}

impl StaticCollider {
    /// Returns the surface material of this collider.
    pub fn material(&self) -> SurfaceMaterial {
        match self {
            StaticCollider::TriMesh { material, .. }
            | StaticCollider::AABB { material, .. }
            | StaticCollider::Sphere { material, .. }
            | StaticCollider::Plane { material, .. }
            | StaticCollider::Capsule { material, .. } => *material,
        }
    }
}

// ---------------------------------------------------------------------------
// Moving platform tracking
// ---------------------------------------------------------------------------

/// Tracks moving platforms so that characters ride them smoothly.
///
/// Each tick, the character controller checks whether its ground collider is
/// a platform. If so, the platform's velocity is applied to the character
/// before the desired movement, and the contact point is updated.
#[derive(Debug, Clone)]
pub struct MovingPlatformTracker {
    /// Currently tracked platforms: platform handle → state.
    platforms: HashMap<PlatformHandle, PlatformState>,
}

/// Per-platform tracking state.
#[derive(Debug, Clone, Copy)]
pub struct PlatformState {
    /// Platform position last frame.
    pub prev_position: Vec3,
    /// Platform position this frame.
    pub curr_position: Vec3,
    /// Computed velocity (curr - prev) / dt.
    pub velocity: Vec3,
    /// Platform rotation last frame (as Euler Y angle for simplicity).
    pub prev_yaw: f32,
    /// Platform rotation this frame.
    pub curr_yaw: f32,
    /// Angular velocity (radians/sec around Y).
    pub angular_velocity_y: f32,
}

impl Default for PlatformState {
    fn default() -> Self {
        Self {
            prev_position: Vec3::ZERO,
            curr_position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            prev_yaw: 0.0,
            curr_yaw: 0.0,
            angular_velocity_y: 0.0,
        }
    }
}

impl MovingPlatformTracker {
    /// Creates a new platform tracker.
    pub fn new() -> Self {
        Self {
            platforms: HashMap::new(),
        }
    }

    /// Update a platform's position this frame.
    pub fn update_platform(
        &mut self,
        handle: PlatformHandle,
        position: Vec3,
        yaw: f32,
        dt: f32,
    ) {
        let entry = self.platforms.entry(handle).or_insert(PlatformState {
            prev_position: position,
            curr_position: position,
            velocity: Vec3::ZERO,
            prev_yaw: yaw,
            curr_yaw: yaw,
            angular_velocity_y: 0.0,
        });

        entry.prev_position = entry.curr_position;
        entry.curr_position = position;
        entry.prev_yaw = entry.curr_yaw;
        entry.curr_yaw = yaw;

        if dt > EPSILON {
            entry.velocity = (entry.curr_position - entry.prev_position) / dt;
            let mut dyaw = entry.curr_yaw - entry.prev_yaw;
            // Wrap to [-PI, PI].
            while dyaw > std::f32::consts::PI {
                dyaw -= std::f32::consts::TAU;
            }
            while dyaw < -std::f32::consts::PI {
                dyaw += std::f32::consts::TAU;
            }
            entry.angular_velocity_y = dyaw / dt;
        }
    }

    /// Get the velocity of a platform.
    pub fn platform_velocity(&self, handle: PlatformHandle) -> Vec3 {
        self.platforms
            .get(&handle)
            .map(|s| s.velocity)
            .unwrap_or(Vec3::ZERO)
    }

    /// Get the angular velocity (Y-axis) of a platform.
    pub fn platform_angular_velocity(&self, handle: PlatformHandle) -> f32 {
        self.platforms
            .get(&handle)
            .map(|s| s.angular_velocity_y)
            .unwrap_or(0.0)
    }

    /// Compute the effective velocity of a point on a rotating platform.
    ///
    /// For a character standing at `contact_world`, the effective velocity is
    /// `platform_linear_vel + angular_vel x (contact_world - platform_center)`.
    pub fn effective_velocity_at(
        &self,
        handle: PlatformHandle,
        contact_world: Vec3,
    ) -> Vec3 {
        let state = match self.platforms.get(&handle) {
            Some(s) => s,
            None => return Vec3::ZERO,
        };

        let r = contact_world - state.curr_position;
        // Angular velocity is around Y axis.
        let angular = Vec3::new(0.0, state.angular_velocity_y, 0.0);
        let tangential = angular.cross(r);

        state.velocity + tangential
    }

    /// Removes stale platforms that haven't been updated.
    pub fn remove_platform(&mut self, handle: PlatformHandle) {
        self.platforms.remove(&handle);
    }

    /// Clears all tracked platforms.
    pub fn clear(&mut self) {
        self.platforms.clear();
    }
}

impl Default for MovingPlatformTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Character controller manager
// ---------------------------------------------------------------------------

/// Manages all physics character controllers in the world.
///
/// Provides the high-level API for creating characters and moving them each
/// tick. The geometry is provided via a slice of [`StaticCollider`]s; in
/// production this would be backed by the `PhysicsWorld` broadphase.
#[derive(Debug)]
pub struct CharacterControllerManager {
    /// All characters, indexed by handle.
    characters: HashMap<CharacterHandle, PhysicsCharacter>,
    /// Next handle to allocate.
    next_handle: u64,
    /// Platform tracker.
    pub platform_tracker: MovingPlatformTracker,
    /// Global gravity (applied to non-grounded characters).
    pub gravity: Vec3,
}

impl CharacterControllerManager {
    /// Create a new manager with the given gravity.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            characters: HashMap::new(),
            next_handle: 1,
            platform_tracker: MovingPlatformTracker::new(),
            gravity,
        }
    }

    /// Add a character controller and return its handle.
    pub fn add_character(&mut self, mut character: PhysicsCharacter) -> CharacterHandle {
        let handle = CharacterHandle(self.next_handle);
        self.next_handle += 1;
        character.handle = handle;
        self.characters.insert(handle, character);
        handle
    }

    /// Remove a character controller.
    pub fn remove_character(&mut self, handle: CharacterHandle) -> Option<PhysicsCharacter> {
        self.characters.remove(&handle)
    }

    /// Get a reference to a character.
    pub fn get_character(&self, handle: CharacterHandle) -> Option<&PhysicsCharacter> {
        self.characters.get(&handle)
    }

    /// Get a mutable reference to a character.
    pub fn get_character_mut(&mut self, handle: CharacterHandle) -> Option<&mut PhysicsCharacter> {
        self.characters.get_mut(&handle)
    }

    /// Returns all character handles.
    pub fn all_handles(&self) -> Vec<CharacterHandle> {
        self.characters.keys().copied().collect()
    }

    /// Move a character with the given desired velocity for a time step.
    ///
    /// This is the main API: call it once per physics tick for each character.
    pub fn move_character(
        &mut self,
        handle: CharacterHandle,
        desired_velocity: Vec3,
        dt: f32,
        colliders: &[StaticCollider],
    ) -> Option<CharacterMoveResult> {
        let gravity = self.gravity;
        let character = self.characters.get_mut(&handle)?;

        // Clamp the input velocity.
        let clamped_vel = clamp_velocity(desired_velocity, MAX_VELOCITY);

        let result = move_character_impl(character, clamped_vel, dt, gravity, colliders, &self.platform_tracker);

        // Update character state from the result.
        character.position = result.position;
        character.velocity = result.velocity;
        character.grounded = result.grounded;
        character.ground_normal = result.ground_normal;
        character.ground_material = result.ground_material;

        Some(result)
    }
}

impl Default for CharacterControllerManager {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, -DEFAULT_GRAVITY, 0.0))
    }
}

// ===========================================================================
// Core movement algorithm
// ===========================================================================

/// The main character movement routine.
///
/// Implements the "collide and slide" algorithm:
///
/// 1. Combine desired velocity with gravity (if airborne) and platform velocity.
/// 2. Cast the capsule along the velocity direction.
/// 3. If hit: advance to the hit point, project remaining velocity onto the
///    tangent plane of the surface, and repeat (up to `MAX_CLIP_ITERATIONS`).
/// 4. After movement, perform step-up detection if a low obstacle was hit.
/// 5. Perform ground detection and apply ground snapping.
/// 6. If on a steep slope, apply a slide-off force.
fn move_character_impl(
    character: &mut PhysicsCharacter,
    desired_velocity: Vec3,
    dt: f32,
    gravity: Vec3,
    colliders: &[StaticCollider],
    platforms: &MovingPlatformTracker,
) -> CharacterMoveResult {
    let mut result = CharacterMoveResult {
        position: character.position,
        velocity: desired_velocity,
        grounded: false,
        ground_normal: Vec3::Y,
        ground_material: SurfaceMaterial::NONE,
        collisions: Vec::new(),
        stepped_up: false,
        sliding_on_slope: false,
        platform_velocity: Vec3::ZERO,
    };

    // --- 1. Compute effective velocity ---
    let mut effective_vel = desired_velocity;

    // Apply platform velocity if on a moving platform.
    if character.on_moving_platform {
        if let Some(ph) = character.platform_handle {
            let pv = platforms.effective_velocity_at(ph, character.position);
            effective_vel += pv;
            result.platform_velocity = pv;
        }
    }

    // Apply gravity if not grounded.
    if !character.grounded {
        effective_vel += gravity * character.gravity_scale * dt;
    }

    // Total displacement this frame.
    let mut displacement = effective_vel * dt;

    // --- 2-3. Iterative collide-and-slide ---
    let mut position = character.position;
    let mut remaining_vel = displacement;
    let mut hit_normals: Vec<Vec3> = Vec::new();

    for _iteration in 0..MAX_CLIP_ITERATIONS {
        let move_len = remaining_vel.length();
        if move_len < EPSILON {
            break;
        }
        let move_dir = remaining_vel / move_len;
        let target = position + remaining_vel;

        // Sweep the capsule.
        let hit = sweep_character_capsule(
            &character.capsule,
            position,
            target,
            colliders,
            character.skin_width,
        );

        match hit {
            Some(sweep_hit) => {
                // Advance to the hit point, pulled back by skin width.
                let safe_t = (sweep_hit.t - character.skin_width).max(0.0);
                let advance = move_dir * safe_t;
                position += advance;

                // Record the collision.
                let (is_ground, is_wall, is_ceiling) =
                    CharacterCollision::classify(sweep_hit.normal, character.max_slope);
                result.collisions.push(CharacterCollision {
                    point: sweep_hit.point,
                    normal: sweep_hit.normal,
                    depth: character.skin_width,
                    surface_material: sweep_hit.surface_material,
                    is_ground,
                    is_wall,
                    is_ceiling,
                });

                // --- Step detection ---
                if is_wall && sweep_hit.point.y - position.y < character.step_height {
                    if let Some(stepped_pos) = try_step_up(
                        &character.capsule,
                        position,
                        remaining_vel,
                        character.step_height,
                        character.skin_width,
                        colliders,
                    ) {
                        position = stepped_pos;
                        result.stepped_up = true;
                        // After stepping up, we continue in the original direction
                        // but with vertical component removed.
                        remaining_vel.y = 0.0;
                        let leftover_len = move_len - safe_t;
                        remaining_vel = move_dir * leftover_len;
                        remaining_vel.y = 0.0;
                        continue;
                    }
                }

                // --- Slide along surface ---
                let leftover = remaining_vel - advance;
                remaining_vel = slide_velocity(leftover, sweep_hit.normal, &hit_normals);
                hit_normals.push(sweep_hit.normal);
            }
            None => {
                // No hit — free to move the full distance.
                position += remaining_vel;
                remaining_vel = Vec3::ZERO;
                break;
            }
        }
    }

    // --- 4. Ground detection ---
    let ground_info = GroundQuery::is_grounded(
        colliders,
        position,
        character.capsule.radius,
        character.skin_width,
        character.max_slope,
        GROUND_PROBE_DISTANCE,
    );

    result.grounded = ground_info.hit && ground_info.walkable;
    result.ground_normal = ground_info.normal;
    result.ground_material = ground_info.surface_material;

    // --- 5. Ground snapping ---
    if result.grounded && desired_velocity.y <= 0.0 {
        if let Some(snapped) = GroundQuery::snap_to_ground(
            colliders,
            position,
            character.capsule.radius,
            character.skin_width,
            character.max_slope,
            MAX_SNAP_DISTANCE,
        ) {
            position = snapped;
        }
    }

    // --- 6. Slope sliding ---
    if ground_info.hit && !ground_info.walkable {
        result.sliding_on_slope = true;
        // Project gravity onto the slope tangent plane.
        let gravity_proj = gravity * character.gravity_scale;
        let slide_force = gravity_proj - ground_info.normal * gravity_proj.dot(ground_info.normal);
        let slide_vel = slide_force * dt;
        position += slide_vel * dt;
        result.velocity = slide_vel;
    }

    // --- Finalize ---
    if result.grounded {
        // Zero out downward velocity when on ground.
        result.velocity.y = result.velocity.y.max(0.0);
    } else {
        result.velocity = effective_vel + gravity * character.gravity_scale * dt;
    }

    result.position = position;
    result
}

// ===========================================================================
// Slide velocity projection
// ===========================================================================

/// Projects the remaining velocity onto the tangent plane of the hit surface.
///
/// If two surfaces have been hit in this iteration, we project onto the
/// crease line (intersection of the two planes) to prevent the character
/// from oscillating between surfaces.
fn slide_velocity(velocity: Vec3, normal: Vec3, previous_normals: &[Vec3]) -> Vec3 {
    // Single-plane clip.
    let mut clipped = velocity - normal * velocity.dot(normal);

    // If we already hit another surface, clip against the crease.
    if let Some(&prev_normal) = previous_normals.last() {
        // Crease direction is the cross product of the two normals.
        let crease = prev_normal.cross(normal);
        let crease_len = crease.length();
        if crease_len > EPSILON {
            let crease_dir = crease / crease_len;
            clipped = crease_dir * clipped.dot(crease_dir);
        } else {
            // Normals are parallel — push back.
            clipped = Vec3::ZERO;
        }
    }

    // If the clipped velocity points back into a surface, zero it.
    if clipped.dot(normal) < -EPSILON {
        return Vec3::ZERO;
    }

    clipped
}

// ===========================================================================
// Step-up detection
// ===========================================================================

/// Attempt to step up over a low obstacle.
///
/// The algorithm:
/// 1. Cast upward from the current position by `step_height` to check for ceiling.
/// 2. Cast forward from the raised position by the remaining move distance.
/// 3. Cast downward to find the step surface.
/// 4. If the step surface is walkable and the height delta is within `step_height`,
///    return the new position on the step.
fn try_step_up(
    capsule: &CharacterCapsule,
    position: Vec3,
    velocity: Vec3,
    step_height: f32,
    skin_width: f32,
    colliders: &[StaticCollider],
) -> Option<Vec3> {
    // 1. Check headroom: cast upward.
    let up_target = position + Vec3::new(0.0, step_height, 0.0);
    let up_hit = sweep_character_capsule(capsule, position, up_target, colliders, skin_width);

    let raised_pos = match up_hit {
        Some(hit) => {
            // Can only go up partially.
            let safe_t = (hit.t - skin_width).max(0.0);
            position + Vec3::new(0.0, safe_t, 0.0)
        }
        None => up_target,
    };

    // 2. Cast forward from the raised position.
    let horizontal = Vec3::new(velocity.x, 0.0, velocity.z);
    let horizontal_len = horizontal.length();
    if horizontal_len < EPSILON {
        return None;
    }
    let horizontal_dir = horizontal / horizontal_len;
    let forward_target = raised_pos + horizontal_dir * (horizontal_len.min(capsule.radius * 2.0));
    let forward_hit = sweep_character_capsule(capsule, raised_pos, forward_target, colliders, skin_width);

    let forward_pos = match forward_hit {
        Some(hit) => {
            // Blocked — can't step up.
            if hit.t < skin_width * 2.0 {
                return None;
            }
            let safe_t = (hit.t - skin_width).max(0.0);
            raised_pos + horizontal_dir * safe_t
        }
        None => forward_target,
    };

    // 3. Cast downward to find the step surface.
    let down_target = Vec3::new(forward_pos.x, position.y, forward_pos.z);
    let down_hit = sweep_character_capsule(capsule, forward_pos, down_target, colliders, skin_width);

    match down_hit {
        Some(hit) => {
            let step_pos = forward_pos + Vec3::new(0.0, -(hit.t - skin_width).max(0.0), 0.0);
            let height_delta = step_pos.y - position.y;
            if height_delta > 0.0 && height_delta <= step_height {
                // Verify the step surface is walkable.
                let up_dot = hit.normal.dot(Vec3::Y);
                if up_dot > 0.7 {
                    return Some(step_pos);
                }
            }
            None
        }
        None => {
            // No ground found after stepping — would be floating.
            None
        }
    }
}

// ===========================================================================
// Capsule sweep against world geometry
// ===========================================================================

/// Sweep the character capsule through the world geometry and return the
/// closest hit.
fn sweep_character_capsule(
    capsule: &CharacterCapsule,
    from: Vec3,
    to: Vec3,
    colliders: &[StaticCollider],
    skin_width: f32,
) -> Option<SweepHit> {
    let direction = to - from;
    let move_len = direction.length();
    if move_len < EPSILON {
        return None;
    }

    let mut closest: Option<SweepHit> = None;

    for collider in colliders {
        let hit = match collider {
            StaticCollider::TriMesh {
                vertices,
                indices,
                material,
            } => {
                let mut best_tri: Option<SweepHit> = None;
                for tri_idx in indices {
                    let v0 = vertices[tri_idx[0] as usize];
                    let v1 = vertices[tri_idx[1] as usize];
                    let v2 = vertices[tri_idx[2] as usize];
                    if let Some(mut h) = capsule.sweep_test_triangle(from, to, v0, v1, v2) {
                        h.surface_material = *material;
                        if best_tri.is_none() || h.t < best_tri.as_ref().unwrap().t {
                            best_tri = Some(h);
                        }
                    }
                }
                best_tri
            }
            StaticCollider::AABB { min, max, material } => {
                capsule.sweep_test_aabb(from, to, *min, *max).map(|mut h| {
                    h.surface_material = *material;
                    h
                })
            }
            StaticCollider::Sphere {
                center,
                radius,
                material,
            } => capsule
                .sweep_test_sphere(from, to, *center, *radius)
                .map(|mut h| {
                    h.surface_material = *material;
                    h
                }),
            StaticCollider::Plane {
                normal,
                distance,
                material,
            } => {
                sweep_capsule_plane(capsule, from, to, *normal, *distance).map(|mut h| {
                    h.surface_material = *material;
                    h
                })
            }
            StaticCollider::Capsule {
                base,
                capsule: other,
                material,
            } => capsule
                .sweep_test_capsule(from, to, *base, other)
                .map(|mut h| {
                    h.surface_material = *material;
                    h
                }),
        };

        if let Some(h) = hit {
            if h.t >= 0.0 {
                if closest.is_none() || h.t < closest.as_ref().unwrap().t {
                    closest = Some(h);
                }
            }
        }
    }

    closest
}

// ===========================================================================
// Primitive sweep helpers
// ===========================================================================

/// Sweep a sphere along a direction against a triangle.
///
/// Uses the Moller-Trumbore ray-triangle intersection on the triangle offset
/// by the sphere radius along its normal, plus edge and vertex tests.
fn sweep_sphere_triangle(
    center: Vec3,
    dir: Vec3,
    radius: f32,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    tri_normal: Vec3,
    max_dist: f32,
) -> Option<SweepHit> {
    // 1. Test sphere vs the triangle plane (offset by radius along normal).
    let d = tri_normal.dot(v0);
    let dist_to_plane = tri_normal.dot(center) - d;
    let denom = tri_normal.dot(dir);

    if denom.abs() < EPSILON {
        // Ray is parallel to the plane.
        if dist_to_plane.abs() > radius {
            return None;
        }
        // Ray is within the slab — test edges/vertices.
        return sweep_sphere_triangle_edges(center, dir, radius, v0, v1, v2, max_dist);
    }

    // Time to hit the plane offset by radius.
    let t_plane = (radius - dist_to_plane) / denom;
    if t_plane < -EPSILON || t_plane > max_dist {
        // Check the back side of the plane too.
        let t_plane_back = (-radius - dist_to_plane) / denom;
        if t_plane_back < -EPSILON || t_plane_back > max_dist {
            return sweep_sphere_triangle_edges(center, dir, radius, v0, v1, v2, max_dist);
        }
    }

    // Check if the plane-hit point is inside the triangle.
    let hit_point = center + dir * t_plane.max(0.0) - tri_normal * radius;
    if point_in_triangle(hit_point, v0, v1, v2) {
        if t_plane >= -EPSILON && t_plane <= max_dist {
            return Some(SweepHit {
                t: t_plane.max(0.0),
                point: hit_point,
                normal: if dist_to_plane >= 0.0 { tri_normal } else { -tri_normal },
                surface_material: SurfaceMaterial::NONE,
            });
        }
    }

    // 2. Test edges and vertices.
    sweep_sphere_triangle_edges(center, dir, radius, v0, v1, v2, max_dist)
}

/// Test sphere sweep against triangle edges and vertices.
fn sweep_sphere_triangle_edges(
    center: Vec3,
    dir: Vec3,
    radius: f32,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    max_dist: f32,
) -> Option<SweepHit> {
    let mut best: Option<SweepHit> = None;

    // Test vertices.
    for &vert in &[v0, v1, v2] {
        if let Some(t) = sweep_sphere_vs_point(center, dir, radius, vert, max_dist) {
            let pos = center + dir * t;
            let diff = pos - vert;
            let dist = diff.length();
            let normal = if dist > EPSILON { diff / dist } else { Vec3::Y };
            let hit = SweepHit {
                t,
                point: vert,
                normal,
                surface_material: SurfaceMaterial::NONE,
            };
            if best.is_none() || t < best.as_ref().unwrap().t {
                best = Some(hit);
            }
        }
    }

    // Test edges.
    let edges = [(v0, v1), (v1, v2), (v2, v0)];
    for &(ea, eb) in &edges {
        if let Some(hit) = sweep_sphere_vs_edge(center, dir, radius, ea, eb, max_dist) {
            if best.is_none() || hit.t < best.as_ref().unwrap().t {
                best = Some(hit);
            }
        }
    }

    best
}

/// Sweep a sphere against a point (sphere vs static point).
///
/// Solves: |center + dir * t - point| = radius
/// which gives a quadratic in t.
fn sweep_sphere_vs_point(
    center: Vec3,
    dir: Vec3,
    radius: f32,
    point: Vec3,
    max_dist: f32,
) -> Option<f32> {
    let m = center - point;
    let b = m.dot(dir);
    let c = m.dot(m) - radius * radius;

    // If c > 0 (center outside sphere) and b > 0 (moving away), no hit.
    if c > 0.0 && b > 0.0 {
        return None;
    }

    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }

    let t = -b - discriminant.sqrt();
    let t = if t < 0.0 { 0.0 } else { t };

    if t <= max_dist {
        Some(t)
    } else {
        None
    }
}

/// Sweep a sphere against an edge segment.
fn sweep_sphere_vs_edge(
    center: Vec3,
    dir: Vec3,
    radius: f32,
    edge_a: Vec3,
    edge_b: Vec3,
    max_dist: f32,
) -> Option<SweepHit> {
    let edge = edge_b - edge_a;
    let edge_len_sq = edge.dot(edge);
    if edge_len_sq < EPSILON {
        return None;
    }

    // Parameterize: closest point on the infinite line to the ray is found by
    // solving a 2D quadratic (projecting out the edge direction).
    let m = center - edge_a;
    let md = m.dot(dir);
    let me = m.dot(edge);
    let de = dir.dot(edge);
    let dd = dir.dot(dir);
    let mm = m.dot(m);

    let a_coeff = dd - de * de / edge_len_sq;
    let b_coeff = md - me * de / edge_len_sq;
    let c_coeff = mm - me * me / edge_len_sq - radius * radius;

    if a_coeff.abs() < EPSILON {
        return None;
    }

    let discriminant = b_coeff * b_coeff - a_coeff * c_coeff;
    if discriminant < 0.0 {
        return None;
    }

    let t = (-b_coeff - discriminant.sqrt()) / a_coeff;
    let t = if t < 0.0 { 0.0 } else { t };
    if t > max_dist {
        return None;
    }

    // Check that the closest point on the edge is within the segment.
    let s = (me + t * de) / edge_len_sq;
    if s < 0.0 || s > 1.0 {
        return None;
    }

    let edge_point = edge_a + edge * s;
    let sphere_at_t = center + dir * t;
    let diff = sphere_at_t - edge_point;
    let dist = diff.length();
    let normal = if dist > EPSILON { diff / dist } else { Vec3::Y };

    Some(SweepHit {
        t,
        point: edge_point,
        normal,
        surface_material: SurfaceMaterial::NONE,
    })
}

/// Sweep a capsule (defined by its segment endpoints + radius) against
/// a single edge segment (zero-radius capsule).
fn sweep_capsule_edge(
    cap_a: Vec3,
    cap_b: Vec3,
    radius: f32,
    dir: Vec3,
    edge_a: Vec3,
    edge_b: Vec3,
    max_dist: f32,
) -> Option<SweepHit> {
    // Find closest approach between the two line segments and check if
    // within radius. This is a simplified version using the midpoint
    // of the capsule segment.
    let cap_mid = (cap_a + cap_b) * 0.5;
    let cap_dir = cap_b - cap_a;
    let cap_half_len = cap_dir.length() * 0.5;

    if cap_half_len < EPSILON {
        // Degenerate to sphere.
        return sweep_sphere_vs_edge(cap_mid, dir, radius, edge_a, edge_b, max_dist);
    }

    // Test several sample points along the capsule segment.
    let steps = 4;
    let mut best: Option<SweepHit> = None;
    for i in 0..=steps {
        let frac = i as f32 / steps as f32;
        let sample = cap_a + (cap_b - cap_a) * frac;
        if let Some(hit) = sweep_sphere_vs_edge(sample, dir, radius, edge_a, edge_b, max_dist) {
            if best.is_none() || hit.t < best.as_ref().unwrap().t {
                best = Some(hit);
            }
        }
    }

    best
}

/// Sweep a capsule segment against a point.
fn sweep_capsule_point(
    cap_a: Vec3,
    cap_b: Vec3,
    radius: f32,
    dir: Vec3,
    point: Vec3,
    max_dist: f32,
) -> Option<SweepHit> {
    // Test closest point on the capsule segment to the point.
    let ab = cap_b - cap_a;
    let len_sq = ab.dot(ab);

    if len_sq < EPSILON {
        // Degenerate to sphere.
        let t = sweep_sphere_vs_point(cap_a, dir, radius, point, max_dist)?;
        let pos = cap_a + dir * t;
        let diff = pos - point;
        let dist = diff.length();
        let normal = if dist > EPSILON { diff / dist } else { Vec3::Y };
        return Some(SweepHit {
            t,
            point,
            normal,
            surface_material: SurfaceMaterial::NONE,
        });
    }

    // Sample several points along the segment.
    let steps = 4;
    let mut best: Option<SweepHit> = None;
    for i in 0..=steps {
        let frac = i as f32 / steps as f32;
        let sample = cap_a + ab * frac;
        if let Some(t) = sweep_sphere_vs_point(sample, dir, radius, point, max_dist) {
            let pos = sample + dir * t;
            let diff = pos - point;
            let dist = diff.length();
            let normal = if dist > EPSILON { diff / dist } else { Vec3::Y };
            let hit = SweepHit {
                t,
                point,
                normal,
                surface_material: SurfaceMaterial::NONE,
            };
            if best.is_none() || t < best.as_ref().unwrap().t {
                best = Some(hit);
            }
        }
    }

    best
}

/// Sweep a capsule against an infinite plane.
fn sweep_capsule_plane(
    capsule: &CharacterCapsule,
    from: Vec3,
    to: Vec3,
    plane_normal: Vec3,
    plane_distance: f32,
) -> Option<SweepHit> {
    let direction = to - from;
    let move_len = direction.length();
    if move_len < EPSILON {
        return None;
    }
    let dir = direction / move_len;

    let seg_a = capsule.bottom_center(from);
    let seg_b = capsule.top_center(from);

    let mut best: Option<SweepHit> = None;
    for &center in &[seg_a, seg_b] {
        let dist_to_plane = plane_normal.dot(center) - plane_distance;
        let denom = plane_normal.dot(dir);

        if denom.abs() < EPSILON {
            continue;
        }

        // Time to reach the plane, offset by radius.
        let t = (capsule.radius - dist_to_plane) / denom;
        if t >= -EPSILON && t <= move_len {
            let t_clamped = t.max(0.0);
            let point = center + dir * t_clamped - plane_normal * capsule.radius;
            let hit = SweepHit {
                t: t_clamped,
                point,
                normal: plane_normal,
                surface_material: SurfaceMaterial::NONE,
            };
            if best.is_none() || t_clamped < best.as_ref().unwrap().t {
                best = Some(hit);
            }
        }
    }

    best
}

// ===========================================================================
// Ray / sphere cast helpers for ground detection
// ===========================================================================

/// Raycast hit for ground detection.
#[derive(Debug, Clone, Copy)]
struct GroundRayHit {
    distance: f32,
    normal: Vec3,
    surface_material: SurfaceMaterial,
}

/// Cast a ray against the collider set.
fn raycast_colliders(
    colliders: &[StaticCollider],
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
) -> Option<GroundRayHit> {
    let mut closest: Option<GroundRayHit> = None;

    for collider in colliders {
        let hit = match collider {
            StaticCollider::Plane {
                normal,
                distance,
                material,
            } => ray_vs_plane(origin, dir, *normal, *distance, max_dist).map(|t| GroundRayHit {
                distance: t,
                normal: *normal,
                surface_material: *material,
            }),
            StaticCollider::AABB { min, max, material } => {
                ray_vs_aabb(origin, dir, *min, *max, max_dist).map(|t| {
                    let point = origin + dir * t;
                    GroundRayHit {
                        distance: t,
                        normal: aabb_contact_normal(point, *min, *max),
                        surface_material: *material,
                    }
                })
            }
            StaticCollider::Sphere {
                center,
                radius,
                material,
            } => {
                let m = origin - *center;
                let b = m.dot(dir);
                let c = m.dot(m) - radius * radius;
                let disc = b * b - c;
                if disc < 0.0 {
                    None
                } else {
                    let t = -b - disc.sqrt();
                    if t >= 0.0 && t <= max_dist {
                        let point = origin + dir * t;
                        let normal = (point - *center).normalize();
                        Some(GroundRayHit {
                            distance: t,
                            normal,
                            surface_material: *material,
                        })
                    } else {
                        None
                    }
                }
            }
            StaticCollider::TriMesh {
                vertices,
                indices,
                material,
            } => {
                let mut best_tri: Option<GroundRayHit> = None;
                for tri_idx in indices {
                    let v0 = vertices[tri_idx[0] as usize];
                    let v1 = vertices[tri_idx[1] as usize];
                    let v2 = vertices[tri_idx[2] as usize];
                    if let Some((t, normal)) = ray_vs_triangle(origin, dir, v0, v1, v2, max_dist) {
                        let hit = GroundRayHit {
                            distance: t,
                            normal,
                            surface_material: *material,
                        };
                        if best_tri.is_none() || t < best_tri.as_ref().unwrap().distance {
                            best_tri = Some(hit);
                        }
                    }
                }
                best_tri
            }
            StaticCollider::Capsule { .. } => None, // Skip capsule colliders for ground ray
        };

        if let Some(h) = hit {
            if closest.is_none() || h.distance < closest.as_ref().unwrap().distance {
                closest = Some(h);
            }
        }
    }

    closest
}

/// Sphere-cast against the collider set (for more robust ground detection).
fn spherecast_colliders(
    colliders: &[StaticCollider],
    origin: Vec3,
    dir: Vec3,
    radius: f32,
    max_dist: f32,
) -> Option<GroundRayHit> {
    let mut closest: Option<GroundRayHit> = None;

    for collider in colliders {
        let hit = match collider {
            StaticCollider::Plane {
                normal,
                distance,
                material,
            } => {
                let dist_to_plane = normal.dot(origin) - *distance;
                let denom = normal.dot(dir);
                if denom.abs() < EPSILON {
                    None
                } else {
                    let t = (radius - dist_to_plane) / denom;
                    if t >= 0.0 && t <= max_dist {
                        Some(GroundRayHit {
                            distance: t,
                            normal: *normal,
                            surface_material: *material,
                        })
                    } else {
                        None
                    }
                }
            }
            StaticCollider::AABB { min, max, material } => {
                let expanded_min = *min - Vec3::splat(radius);
                let expanded_max = *max + Vec3::splat(radius);
                ray_vs_aabb(origin, dir, expanded_min, expanded_max, max_dist).map(|t| {
                    let point = origin + dir * t;
                    GroundRayHit {
                        distance: t,
                        normal: aabb_contact_normal(point, *min, *max),
                        surface_material: *material,
                    }
                })
            }
            _ => None,
        };

        if let Some(h) = hit {
            if closest.is_none() || h.distance < closest.as_ref().unwrap().distance {
                closest = Some(h);
            }
        }
    }

    closest
}

// ===========================================================================
// Geometric primitives
// ===========================================================================

/// Ray vs AABB intersection (slab method). Returns parametric t of entry.
fn ray_vs_aabb(origin: Vec3, dir: Vec3, aabb_min: Vec3, aabb_max: Vec3, max_dist: f32) -> Option<f32> {
    let inv_dir = Vec3::new(
        if dir.x.abs() > EPSILON { 1.0 / dir.x } else { f32::INFINITY * dir.x.signum() },
        if dir.y.abs() > EPSILON { 1.0 / dir.y } else { f32::INFINITY * dir.y.signum() },
        if dir.z.abs() > EPSILON { 1.0 / dir.z } else { f32::INFINITY * dir.z.signum() },
    );

    let t1 = (aabb_min.x - origin.x) * inv_dir.x;
    let t2 = (aabb_max.x - origin.x) * inv_dir.x;
    let t3 = (aabb_min.y - origin.y) * inv_dir.y;
    let t4 = (aabb_max.y - origin.y) * inv_dir.y;
    let t5 = (aabb_min.z - origin.z) * inv_dir.z;
    let t6 = (aabb_max.z - origin.z) * inv_dir.z;

    let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
    let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

    if tmax < 0.0 || tmin > tmax || tmin > max_dist {
        return None;
    }

    let t = if tmin >= 0.0 { tmin } else { 0.0 };
    Some(t)
}

/// Compute the contact normal for a point near an AABB.
fn aabb_contact_normal(point: Vec3, aabb_min: Vec3, aabb_max: Vec3) -> Vec3 {
    let center = (aabb_min + aabb_max) * 0.5;
    let half = (aabb_max - aabb_min) * 0.5;
    let local = point - center;

    // Find which face is closest.
    let dx = (local.x.abs() - half.x).abs();
    let dy = (local.y.abs() - half.y).abs();
    let dz = (local.z.abs() - half.z).abs();

    if dx <= dy && dx <= dz {
        Vec3::new(local.x.signum(), 0.0, 0.0)
    } else if dy <= dz {
        Vec3::new(0.0, local.y.signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, local.z.signum())
    }
}

/// Ray vs infinite plane. Returns parametric t of intersection.
fn ray_vs_plane(origin: Vec3, dir: Vec3, normal: Vec3, distance: f32, max_dist: f32) -> Option<f32> {
    let denom = normal.dot(dir);
    if denom.abs() < EPSILON {
        return None;
    }
    let t = (distance - normal.dot(origin)) / denom;
    if t >= 0.0 && t <= max_dist {
        Some(t)
    } else {
        None
    }
}

/// Moller-Trumbore ray-triangle intersection.
fn ray_vs_triangle(
    origin: Vec3,
    dir: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = dir.cross(edge2);
    let a = edge1.dot(h);

    if a.abs() < EPSILON {
        return None;
    }

    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(edge1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * edge2.dot(q);
    if t >= 0.0 && t <= max_dist {
        let normal = edge1.cross(edge2).normalize();
        Some((t, normal))
    } else {
        None
    }
}

/// Check if a point lies inside a triangle (using barycentric coordinates).
fn point_in_triangle(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> bool {
    let v0v1 = v1 - v0;
    let v0v2 = v2 - v0;
    let v0p = p - v0;

    let dot00 = v0v2.dot(v0v2);
    let dot01 = v0v2.dot(v0v1);
    let dot02 = v0v2.dot(v0p);
    let dot11 = v0v1.dot(v0v1);
    let dot12 = v0v1.dot(v0p);

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    u >= -EPSILON && v >= -EPSILON && (u + v) <= 1.0 + EPSILON
}

/// Clamp a velocity vector to a maximum magnitude.
fn clamp_velocity(vel: Vec3, max_speed: f32) -> Vec3 {
    let speed = vel.length();
    if speed > max_speed {
        vel * (max_speed / speed)
    } else {
        vel
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const UP: Vec3 = Vec3::Y;

    fn ground_plane() -> StaticCollider {
        StaticCollider::Plane {
            normal: Vec3::Y,
            distance: 0.0,
            material: SurfaceMaterial(0),
        }
    }

    fn box_collider(min: Vec3, max: Vec3) -> StaticCollider {
        StaticCollider::AABB {
            min,
            max,
            material: SurfaceMaterial(1),
        }
    }

    #[test]
    fn capsule_creation() {
        let cap = CharacterCapsule::new(0.3, 1.8);
        assert!((cap.total_height() - 1.8).abs() < EPSILON);
        assert!((cap.radius - 0.3).abs() < EPSILON);
        assert!((cap.half_height - 0.6).abs() < EPSILON);
    }

    #[test]
    fn capsule_degenerate_to_sphere() {
        let cap = CharacterCapsule::new(0.5, 0.5);
        assert!((cap.half_height - 0.0).abs() < EPSILON);
        assert!((cap.total_height() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn capsule_centers() {
        let cap = CharacterCapsule::new(0.3, 1.8);
        let base = Vec3::new(0.0, 0.0, 0.0);
        let bottom = cap.bottom_center(base);
        let top = cap.top_center(base);
        let center = cap.center(base);

        assert!((bottom.y - 0.3).abs() < EPSILON);
        assert!((top.y - 1.5).abs() < EPSILON);
        assert!((center.y - 0.9).abs() < EPSILON);
    }

    #[test]
    fn ground_detection_flat() {
        let colliders = vec![ground_plane()];
        let position = Vec3::new(0.0, 0.01, 0.0);
        let info = GroundQuery::is_grounded(&colliders, position, 0.3, 0.01, DEFAULT_MAX_SLOPE, 0.15);
        assert!(info.hit);
        assert!(info.walkable);
        assert!((info.normal - Vec3::Y).length() < EPSILON);
    }

    #[test]
    fn ground_detection_airborne() {
        let colliders = vec![ground_plane()];
        let position = Vec3::new(0.0, 5.0, 0.0);
        let info = GroundQuery::is_grounded(&colliders, position, 0.3, 0.01, DEFAULT_MAX_SLOPE, 0.15);
        assert!(!info.hit);
    }

    #[test]
    fn character_falls_to_ground() {
        let colliders = vec![ground_plane()];
        let mut manager = CharacterControllerManager::default();
        let handle = manager.add_character(
            PhysicsCharacter::new(0.3, 1.8).with_position(Vec3::new(0.0, 2.0, 0.0)),
        );

        // Simulate several ticks — the character should move downward.
        let mut last_y = 2.0;
        for _ in 0..10 {
            let result = manager
                .move_character(handle, Vec3::ZERO, 1.0 / 60.0, &colliders)
                .unwrap();
            assert!(result.position.y <= last_y + EPSILON);
            last_y = result.position.y;
        }
    }

    #[test]
    fn character_walks_on_ground() {
        let colliders = vec![ground_plane()];
        let mut manager = CharacterControllerManager::default();
        let mut char = PhysicsCharacter::new(0.3, 1.8);
        char.position = Vec3::new(0.0, 0.0, 0.0);
        char.grounded = true;
        let handle = manager.add_character(char);

        let result = manager
            .move_character(handle, Vec3::new(5.0, 0.0, 0.0), 1.0 / 60.0, &colliders)
            .unwrap();
        // Should have moved in X.
        assert!(result.position.x > 0.0);
    }

    #[test]
    fn character_slides_on_wall() {
        let colliders = vec![
            ground_plane(),
            box_collider(Vec3::new(1.0, 0.0, -5.0), Vec3::new(1.5, 3.0, 5.0)),
        ];
        let mut manager = CharacterControllerManager::default();
        let mut char = PhysicsCharacter::new(0.3, 1.8);
        char.position = Vec3::new(0.0, 0.0, 0.0);
        char.grounded = true;
        let handle = manager.add_character(char);

        // Move diagonally toward the wall.
        let result = manager
            .move_character(handle, Vec3::new(5.0, 0.0, 5.0), 1.0 / 60.0, &colliders)
            .unwrap();
        // Should have slid along the wall in Z.
        assert!(result.position.z > 0.0);
    }

    #[test]
    fn ray_vs_aabb_basic() {
        let origin = Vec3::new(0.0, 5.0, 0.0);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let aabb_min = Vec3::new(-1.0, 0.0, -1.0);
        let aabb_max = Vec3::new(1.0, 1.0, 1.0);

        let t = ray_vs_aabb(origin, dir, aabb_min, aabb_max, 100.0);
        assert!(t.is_some());
        assert!((t.unwrap() - 4.0).abs() < EPSILON);
    }

    #[test]
    fn ray_vs_aabb_miss() {
        let origin = Vec3::new(5.0, 5.0, 0.0);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let aabb_min = Vec3::new(-1.0, 0.0, -1.0);
        let aabb_max = Vec3::new(1.0, 1.0, 1.0);

        let t = ray_vs_aabb(origin, dir, aabb_min, aabb_max, 100.0);
        assert!(t.is_none());
    }

    #[test]
    fn ray_vs_triangle_basic() {
        let origin = Vec3::new(0.0, 1.0, 0.0);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let v0 = Vec3::new(-1.0, 0.0, -1.0);
        let v1 = Vec3::new(1.0, 0.0, -1.0);
        let v2 = Vec3::new(0.0, 0.0, 1.0);

        let result = ray_vs_triangle(origin, dir, v0, v1, v2, 100.0);
        assert!(result.is_some());
        let (t, normal) = result.unwrap();
        assert!((t - 1.0).abs() < 0.01);
        assert!(normal.dot(Vec3::Y).abs() > 0.9);
    }

    #[test]
    fn point_in_triangle_inside() {
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(2.0, 0.0, 0.0);
        let v2 = Vec3::new(1.0, 0.0, 2.0);

        assert!(point_in_triangle(Vec3::new(1.0, 0.0, 0.5), v0, v1, v2));
    }

    #[test]
    fn point_in_triangle_outside() {
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(2.0, 0.0, 0.0);
        let v2 = Vec3::new(1.0, 0.0, 2.0);

        assert!(!point_in_triangle(Vec3::new(-1.0, 0.0, -1.0), v0, v1, v2));
    }

    #[test]
    fn sweep_sphere_vs_point_basic() {
        let center = Vec3::new(0.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let radius = 0.5;
        let point = Vec3::new(3.0, 0.0, 0.0);

        let t = sweep_sphere_vs_point(center, dir, radius, point, 10.0);
        assert!(t.is_some());
        assert!((t.unwrap() - 2.5).abs() < 0.01);
    }

    #[test]
    fn surface_material_default() {
        assert_eq!(SurfaceMaterial::default(), SurfaceMaterial(0));
        assert!(!SurfaceMaterial::NONE.is_valid());
        assert!(SurfaceMaterial(0).is_valid());
    }

    #[test]
    fn platform_tracker_velocity() {
        let mut tracker = MovingPlatformTracker::new();
        let handle = PlatformHandle(1);
        let dt = 1.0 / 60.0;

        tracker.update_platform(handle, Vec3::new(0.0, 0.0, 0.0), 0.0, dt);
        tracker.update_platform(handle, Vec3::new(1.0, 0.0, 0.0), 0.0, dt);

        let vel = tracker.platform_velocity(handle);
        assert!((vel.x - 60.0).abs() < 0.1);
    }

    #[test]
    fn slide_velocity_single_plane() {
        let vel = Vec3::new(1.0, 0.0, 1.0);
        let normal = Vec3::new(1.0, 0.0, 0.0);
        let result = slide_velocity(vel, normal, &[]);
        // Should have no X component.
        assert!(result.x.abs() < EPSILON);
        assert!((result.z - 1.0).abs() < EPSILON);
    }

    #[test]
    fn collision_classify_ground() {
        let (is_ground, is_wall, is_ceiling) =
            CharacterCollision::classify(Vec3::Y, DEFAULT_MAX_SLOPE);
        assert!(is_ground);
        assert!(!is_wall);
        assert!(!is_ceiling);
    }

    #[test]
    fn collision_classify_wall() {
        let (is_ground, is_wall, is_ceiling) =
            CharacterCollision::classify(Vec3::X, DEFAULT_MAX_SLOPE);
        assert!(!is_ground);
        assert!(is_wall);
        assert!(!is_ceiling);
    }

    #[test]
    fn collision_classify_ceiling() {
        let (is_ground, is_wall, is_ceiling) =
            CharacterCollision::classify(-Vec3::Y, DEFAULT_MAX_SLOPE);
        assert!(!is_ground);
        assert!(!is_wall);
        assert!(is_ceiling);
    }

    #[test]
    fn clamp_velocity_below_max() {
        let vel = Vec3::new(1.0, 2.0, 3.0);
        let clamped = clamp_velocity(vel, 100.0);
        assert!((clamped - vel).length() < EPSILON);
    }

    #[test]
    fn clamp_velocity_above_max() {
        let vel = Vec3::new(100.0, 100.0, 100.0);
        let clamped = clamp_velocity(vel, 10.0);
        assert!((clamped.length() - 10.0).abs() < 0.01);
    }

    #[test]
    fn capsule_sweep_vs_sphere() {
        let cap = CharacterCapsule::new(0.3, 1.8);
        let from = Vec3::new(0.0, 0.0, 0.0);
        let to = Vec3::new(10.0, 0.0, 0.0);
        let sphere_center = Vec3::new(5.0, 0.9, 0.0);
        let sphere_radius = 0.5;

        let hit = cap.sweep_test_sphere(from, to, sphere_center, sphere_radius);
        assert!(hit.is_some());
        assert!(hit.unwrap().t < 10.0);
    }

    #[test]
    fn manager_add_remove() {
        let mut manager = CharacterControllerManager::default();
        let handle = manager.add_character(PhysicsCharacter::default());
        assert!(manager.get_character(handle).is_some());
        manager.remove_character(handle);
        assert!(manager.get_character(handle).is_none());
    }
}
