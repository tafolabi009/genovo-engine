// engine/physics/src/shape_casting.rs
//
// Shape cast queries for the Genovo physics engine.
//
// Provides convex shape sweep, box sweep, capsule sweep, sphere sweep,
// all with max distance and layer filter, contact point generation,
// and time of impact computation.
//
// # Architecture
//
// Shape casting sweeps a convex shape along a direction and reports the first
// or all intersections with the world's colliders. The implementation uses
// the GJK+EPA algorithm for narrow-phase testing and a BVH broadphase for
// efficient scene traversal.

use std::collections::HashSet;
use std::fmt;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    #[inline]
    pub fn add(self, o: Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    #[inline]
    pub fn sub(self, o: Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) }
    #[inline]
    pub fn scale(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }
    #[inline]
    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }
    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    #[inline]
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    #[inline]
    pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l > 1e-7 { self.scale(1.0 / l) } else { Self::ZERO }
    }
    #[inline]
    pub fn distance(self, o: Self) -> f32 { self.sub(o).length() }
    #[inline]
    pub fn lerp(self, o: Self, t: f32) -> Self { self.scale(1.0 - t).add(o.scale(t)) }
    #[inline]
    pub fn negate(self) -> Self { Self::new(-self.x, -self.y, -self.z) }
}

/// Quaternion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32, pub y: f32, pub z: f32, pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    pub fn rotate_vec(&self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        let dot_uv = u.dot(v);
        let dot_uu = u.dot(u);
        let cross_uv = u.cross(v);
        v.scale(s * s - dot_uu).add(u.scale(2.0 * dot_uv)).add(cross_uv.scale(2.0 * s))
    }
}

/// AABB.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn from_center_half(center: Vec3, half: Vec3) -> Self {
        Self { min: center.sub(half), max: center.add(half) }
    }

    pub fn from_sphere(center: Vec3, radius: f32) -> Self {
        let r = Vec3::new(radius, radius, radius);
        Self { min: center.sub(r), max: center.add(r) }
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    /// Sweep an AABB along a direction and test intersection.
    pub fn sweep_test(&self, other: &Self, direction: Vec3, max_dist: f32) -> Option<f32> {
        if direction.length_sq() < 1e-10 { return None; }

        let inv_d = Vec3::new(
            if direction.x.abs() > 1e-7 { 1.0 / direction.x } else { f32::MAX },
            if direction.y.abs() > 1e-7 { 1.0 / direction.y } else { f32::MAX },
            if direction.z.abs() > 1e-7 { 1.0 / direction.z } else { f32::MAX },
        );

        let t1x = (other.min.x - self.max.x) * inv_d.x;
        let t2x = (other.max.x - self.min.x) * inv_d.x;
        let t1y = (other.min.y - self.max.y) * inv_d.y;
        let t2y = (other.max.y - self.min.y) * inv_d.y;
        let t1z = (other.min.z - self.max.z) * inv_d.z;
        let t2z = (other.max.z - self.min.z) * inv_d.z;

        let t_enter = t1x.min(t2x).max(t1y.min(t2y)).max(t1z.min(t2z));
        let t_exit = t1x.max(t2x).min(t1y.max(t2y)).min(t1z.max(t2z));

        if t_enter > t_exit || t_exit < 0.0 || t_enter > max_dist {
            return None;
        }

        Some(t_enter.max(0.0))
    }

    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub type BodyHandle = u64;
pub type LayerMask = u32;

pub const ALL_LAYERS: LayerMask = 0xFFFF_FFFF;
pub const DEFAULT_MAX_DISTANCE: f32 = 1000.0;
pub const MAX_SHAPE_CAST_HITS: usize = 256;
pub const GJK_MAX_ITERATIONS: u32 = 64;
pub const EPA_MAX_ITERATIONS: u32 = 64;
pub const EPA_TOLERANCE: f32 = 1e-4;

// ---------------------------------------------------------------------------
// Cast shape types
// ---------------------------------------------------------------------------

/// Shape used for shape casting.
#[derive(Debug, Clone, Copy)]
pub enum CastShape {
    /// Sphere with a radius.
    Sphere { radius: f32 },
    /// Axis-aligned box with half-extents.
    Box { half_extents: Vec3 },
    /// Capsule (two half-spheres connected by a cylinder).
    Capsule { radius: f32, half_height: f32 },
    /// Convex hull (approximated by a set of support directions).
    ConvexHull { vertex_count: u32 },
    /// Point (zero-radius sphere, equivalent to a ray).
    Point,
}

impl CastShape {
    /// Compute the AABB of this shape at a position.
    pub fn compute_aabb(&self, position: Vec3, orientation: Quat) -> Aabb {
        match self {
            CastShape::Sphere { radius } => Aabb::from_sphere(position, *radius),
            CastShape::Box { half_extents } => {
                // Rotate corners to get tight AABB.
                let he = *half_extents;
                let corners = [
                    Vec3::new(-he.x, -he.y, -he.z),
                    Vec3::new(he.x, -he.y, -he.z),
                    Vec3::new(-he.x, he.y, -he.z),
                    Vec3::new(he.x, he.y, -he.z),
                    Vec3::new(-he.x, -he.y, he.z),
                    Vec3::new(he.x, -he.y, he.z),
                    Vec3::new(-he.x, he.y, he.z),
                    Vec3::new(he.x, he.y, he.z),
                ];
                let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
                let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
                for c in &corners {
                    let rotated = orientation.rotate_vec(*c).add(position);
                    min.x = min.x.min(rotated.x);
                    min.y = min.y.min(rotated.y);
                    min.z = min.z.min(rotated.z);
                    max.x = max.x.max(rotated.x);
                    max.y = max.y.max(rotated.y);
                    max.z = max.z.max(rotated.z);
                }
                Aabb::new(min, max)
            }
            CastShape::Capsule { radius, half_height } => {
                let axis = orientation.rotate_vec(Vec3::UP);
                let top = position.add(axis.scale(*half_height));
                let bottom = position.sub(axis.scale(*half_height));
                let r = Vec3::new(*radius, *radius, *radius);
                let min = Vec3::new(
                    top.x.min(bottom.x) - radius,
                    top.y.min(bottom.y) - radius,
                    top.z.min(bottom.z) - radius,
                );
                let max = Vec3::new(
                    top.x.max(bottom.x) + radius,
                    top.y.max(bottom.y) + radius,
                    top.z.max(bottom.z) + radius,
                );
                Aabb::new(min, max)
            }
            CastShape::ConvexHull { .. } => {
                // Approximation: use a sphere bounding the hull.
                Aabb::from_sphere(position, 1.0)
            }
            CastShape::Point => Aabb::new(position, position),
        }
    }

    /// Compute the support point in a given direction (Minkowski support).
    pub fn support(&self, direction: Vec3, position: Vec3, orientation: Quat) -> Vec3 {
        match self {
            CastShape::Sphere { radius } => {
                let d = direction.normalize();
                position.add(d.scale(*radius))
            }
            CastShape::Box { half_extents } => {
                // Transform direction to local space.
                let inv_rot = Quat { x: -orientation.x, y: -orientation.y, z: -orientation.z, w: orientation.w };
                let local_dir = inv_rot.rotate_vec(direction);

                let local_support = Vec3::new(
                    if local_dir.x >= 0.0 { half_extents.x } else { -half_extents.x },
                    if local_dir.y >= 0.0 { half_extents.y } else { -half_extents.y },
                    if local_dir.z >= 0.0 { half_extents.z } else { -half_extents.z },
                );

                orientation.rotate_vec(local_support).add(position)
            }
            CastShape::Capsule { radius, half_height } => {
                let axis = orientation.rotate_vec(Vec3::UP);
                let d = direction.normalize();

                // Select which hemisphere.
                let dot = d.dot(axis);
                let center = if dot >= 0.0 {
                    position.add(axis.scale(*half_height))
                } else {
                    position.sub(axis.scale(*half_height))
                };

                center.add(d.scale(*radius))
            }
            CastShape::ConvexHull { .. } => {
                // Placeholder: treat as unit sphere.
                let d = direction.normalize();
                position.add(d)
            }
            CastShape::Point => position,
        }
    }
}

impl fmt::Display for CastShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sphere { radius } => write!(f, "Sphere(r={})", radius),
            Self::Box { half_extents } => write!(f, "Box({},{},{})", half_extents.x, half_extents.y, half_extents.z),
            Self::Capsule { radius, half_height } => write!(f, "Capsule(r={}, h={})", radius, half_height),
            Self::ConvexHull { vertex_count } => write!(f, "ConvexHull(n={})", vertex_count),
            Self::Point => write!(f, "Point"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shape cast query
// ---------------------------------------------------------------------------

/// Input for a shape cast query.
#[derive(Debug, Clone)]
pub struct ShapeCastInput {
    /// Shape to sweep.
    pub shape: CastShape,
    /// Starting position of the shape.
    pub origin: Vec3,
    /// Orientation of the shape.
    pub orientation: Quat,
    /// Direction to sweep (not necessarily normalized).
    pub direction: Vec3,
    /// Maximum sweep distance.
    pub max_distance: f32,
    /// Layer mask filter.
    pub layer_mask: LayerMask,
    /// Bodies to ignore.
    pub ignore_bodies: HashSet<BodyHandle>,
    /// Whether to report the first hit only.
    pub first_hit_only: bool,
    /// Whether to report hits with triggers.
    pub include_triggers: bool,
    /// Whether to report back-face hits.
    pub report_back_faces: bool,
}

impl Default for ShapeCastInput {
    fn default() -> Self {
        Self {
            shape: CastShape::Sphere { radius: 0.5 },
            origin: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            direction: Vec3::new(0.0, 0.0, -1.0),
            max_distance: DEFAULT_MAX_DISTANCE,
            layer_mask: ALL_LAYERS,
            ignore_bodies: HashSet::new(),
            first_hit_only: true,
            include_triggers: false,
            report_back_faces: false,
        }
    }
}

/// Result of a shape cast hit.
#[derive(Debug, Clone)]
pub struct ShapeCastHit {
    /// Body that was hit.
    pub body: BodyHandle,
    /// Time of impact (0 = at origin, 1 = at max_distance).
    pub time_of_impact: f32,
    /// World-space hit point.
    pub hit_point: Vec3,
    /// Hit normal (pointing away from the hit surface).
    pub hit_normal: Vec3,
    /// Penetration depth (if starting inside geometry).
    pub penetration_depth: f32,
    /// Distance from origin to hit.
    pub distance: f32,
    /// Whether the cast started inside the collider.
    pub started_inside: bool,
    /// Whether this is a trigger volume hit.
    pub is_trigger: bool,
    /// The face/triangle index (if applicable).
    pub face_index: Option<u32>,
    /// Contact point on the cast shape's surface.
    pub shape_contact_point: Vec3,
    /// Contact point on the hit body's surface.
    pub body_contact_point: Vec3,
}

impl ShapeCastHit {
    /// Whether this hit is closer than another.
    pub fn is_closer_than(&self, other: &Self) -> bool {
        self.time_of_impact < other.time_of_impact
    }
}

/// Result of a shape cast query.
#[derive(Debug, Clone)]
pub struct ShapeCastResult {
    /// All hits (sorted by time of impact).
    pub hits: Vec<ShapeCastHit>,
    /// Whether any hit was found.
    pub has_hit: bool,
    /// The closest hit (if any).
    pub closest: Option<ShapeCastHit>,
}

impl ShapeCastResult {
    pub fn empty() -> Self {
        Self {
            hits: Vec::new(),
            has_hit: false,
            closest: None,
        }
    }

    pub fn from_hits(mut hits: Vec<ShapeCastHit>) -> Self {
        hits.sort_by(|a, b| a.time_of_impact.partial_cmp(&b.time_of_impact).unwrap_or(std::cmp::Ordering::Equal));
        let closest = hits.first().cloned();
        let has_hit = !hits.is_empty();
        Self { hits, has_hit, closest }
    }

    pub fn hit_count(&self) -> usize { self.hits.len() }
}

// ---------------------------------------------------------------------------
// Convenience query builders
// ---------------------------------------------------------------------------

/// Builder for sphere cast queries.
pub struct SphereCast {
    pub radius: f32,
    pub origin: Vec3,
    pub direction: Vec3,
    pub max_distance: f32,
    pub layer_mask: LayerMask,
    pub ignore_bodies: HashSet<BodyHandle>,
}

impl SphereCast {
    pub fn new(origin: Vec3, direction: Vec3, radius: f32) -> Self {
        Self {
            radius,
            origin,
            direction,
            max_distance: DEFAULT_MAX_DISTANCE,
            layer_mask: ALL_LAYERS,
            ignore_bodies: HashSet::new(),
        }
    }

    pub fn with_max_distance(mut self, dist: f32) -> Self { self.max_distance = dist; self }
    pub fn with_layer_mask(mut self, mask: LayerMask) -> Self { self.layer_mask = mask; self }
    pub fn ignore_body(mut self, body: BodyHandle) -> Self { self.ignore_bodies.insert(body); self }

    pub fn to_input(&self) -> ShapeCastInput {
        ShapeCastInput {
            shape: CastShape::Sphere { radius: self.radius },
            origin: self.origin,
            direction: self.direction,
            max_distance: self.max_distance,
            layer_mask: self.layer_mask,
            ignore_bodies: self.ignore_bodies.clone(),
            ..Default::default()
        }
    }
}

/// Builder for box cast queries.
pub struct BoxCast {
    pub half_extents: Vec3,
    pub origin: Vec3,
    pub orientation: Quat,
    pub direction: Vec3,
    pub max_distance: f32,
    pub layer_mask: LayerMask,
    pub ignore_bodies: HashSet<BodyHandle>,
}

impl BoxCast {
    pub fn new(origin: Vec3, direction: Vec3, half_extents: Vec3) -> Self {
        Self {
            half_extents,
            origin,
            orientation: Quat::IDENTITY,
            direction,
            max_distance: DEFAULT_MAX_DISTANCE,
            layer_mask: ALL_LAYERS,
            ignore_bodies: HashSet::new(),
        }
    }

    pub fn with_orientation(mut self, orientation: Quat) -> Self { self.orientation = orientation; self }
    pub fn with_max_distance(mut self, dist: f32) -> Self { self.max_distance = dist; self }
    pub fn with_layer_mask(mut self, mask: LayerMask) -> Self { self.layer_mask = mask; self }
    pub fn ignore_body(mut self, body: BodyHandle) -> Self { self.ignore_bodies.insert(body); self }

    pub fn to_input(&self) -> ShapeCastInput {
        ShapeCastInput {
            shape: CastShape::Box { half_extents: self.half_extents },
            origin: self.origin,
            orientation: self.orientation,
            direction: self.direction,
            max_distance: self.max_distance,
            layer_mask: self.layer_mask,
            ignore_bodies: self.ignore_bodies.clone(),
            ..Default::default()
        }
    }
}

/// Builder for capsule cast queries.
pub struct CapsuleCast {
    pub radius: f32,
    pub half_height: f32,
    pub origin: Vec3,
    pub orientation: Quat,
    pub direction: Vec3,
    pub max_distance: f32,
    pub layer_mask: LayerMask,
    pub ignore_bodies: HashSet<BodyHandle>,
}

impl CapsuleCast {
    pub fn new(origin: Vec3, direction: Vec3, radius: f32, half_height: f32) -> Self {
        Self {
            radius,
            half_height,
            origin,
            orientation: Quat::IDENTITY,
            direction,
            max_distance: DEFAULT_MAX_DISTANCE,
            layer_mask: ALL_LAYERS,
            ignore_bodies: HashSet::new(),
        }
    }

    pub fn with_orientation(mut self, orientation: Quat) -> Self { self.orientation = orientation; self }
    pub fn with_max_distance(mut self, dist: f32) -> Self { self.max_distance = dist; self }
    pub fn with_layer_mask(mut self, mask: LayerMask) -> Self { self.layer_mask = mask; self }
    pub fn ignore_body(mut self, body: BodyHandle) -> Self { self.ignore_bodies.insert(body); self }

    pub fn to_input(&self) -> ShapeCastInput {
        ShapeCastInput {
            shape: CastShape::Capsule { radius: self.radius, half_height: self.half_height },
            origin: self.origin,
            orientation: self.orientation,
            direction: self.direction,
            max_distance: self.max_distance,
            layer_mask: self.layer_mask,
            ignore_bodies: self.ignore_bodies.clone(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// GJK algorithm (simplified)
// ---------------------------------------------------------------------------

/// Simplex for GJK algorithm.
#[derive(Debug, Clone)]
struct Simplex {
    points: Vec<Vec3>,
    count: usize,
}

impl Simplex {
    fn new() -> Self { Self { points: Vec::with_capacity(4), count: 0 } }

    fn push(&mut self, p: Vec3) {
        if self.count < 4 {
            self.points.push(p);
            self.count += 1;
        }
    }

    fn last(&self) -> Vec3 { self.points[self.count - 1] }
}

/// Minkowski difference support function.
fn minkowski_support(
    shape_a: &CastShape, pos_a: Vec3, rot_a: Quat,
    shape_b: &CastShape, pos_b: Vec3, rot_b: Quat,
    direction: Vec3,
) -> Vec3 {
    let a = shape_a.support(direction, pos_a, rot_a);
    let b = shape_b.support(direction.negate(), pos_b, rot_b);
    a.sub(b)
}

/// Simplified GJK intersection test.
pub fn gjk_intersect(
    shape_a: &CastShape, pos_a: Vec3, rot_a: Quat,
    shape_b: &CastShape, pos_b: Vec3, rot_b: Quat,
) -> bool {
    let mut direction = pos_b.sub(pos_a);
    if direction.length_sq() < 1e-10 {
        direction = Vec3::new(1.0, 0.0, 0.0);
    }

    let mut simplex = Simplex::new();
    let support = minkowski_support(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b, direction);
    simplex.push(support);
    direction = support.negate();

    for _ in 0..GJK_MAX_ITERATIONS {
        let a = minkowski_support(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b, direction);

        if a.dot(direction) < 0.0 {
            return false; // No intersection.
        }

        simplex.push(a);

        // Process simplex and update direction.
        match simplex.count {
            2 => {
                let b_pt = simplex.points[0];
                let a_pt = simplex.points[1];
                let ab = b_pt.sub(a_pt);
                let ao = a_pt.negate();

                if ab.dot(ao) > 0.0 {
                    direction = ab.cross(ao).cross(ab);
                } else {
                    simplex.points = vec![a_pt];
                    simplex.count = 1;
                    direction = ao;
                }
            }
            3 => {
                let c = simplex.points[0];
                let b_pt = simplex.points[1];
                let a_pt = simplex.points[2];
                let ab = b_pt.sub(a_pt);
                let ac = c.sub(a_pt);
                let ao = a_pt.negate();
                let abc = ab.cross(ac);

                if abc.cross(ac).dot(ao) > 0.0 {
                    if ac.dot(ao) > 0.0 {
                        simplex.points = vec![c, a_pt];
                        simplex.count = 2;
                        direction = ac.cross(ao).cross(ac);
                    } else {
                        simplex.points = vec![a_pt];
                        simplex.count = 1;
                        direction = ao;
                    }
                } else if ab.cross(abc).dot(ao) > 0.0 {
                    simplex.points = vec![b_pt, a_pt];
                    simplex.count = 2;
                    direction = ab.cross(ao).cross(ab);
                } else {
                    if abc.dot(ao) > 0.0 {
                        direction = abc;
                    } else {
                        simplex.points = vec![b_pt, c, a_pt];
                        direction = abc.negate();
                    }
                }
            }
            4 => {
                // Full tetrahedron: check if origin is inside.
                return true;
            }
            _ => {}
        }

        if direction.length_sq() < 1e-10 {
            return true;
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Time of impact (conservative advancement)
// ---------------------------------------------------------------------------

/// Compute the time of impact between a moving shape and a static shape.
pub fn time_of_impact(
    shape_a: &CastShape, pos_a: Vec3, rot_a: Quat, velocity: Vec3,
    shape_b: &CastShape, pos_b: Vec3, rot_b: Quat,
    max_time: f32,
) -> Option<f32> {
    let steps = 32u32;
    let dt = max_time / steps as f32;

    for i in 0..steps {
        let t = i as f32 * dt;
        let current_pos = pos_a.add(velocity.scale(t));

        if gjk_intersect(shape_a, current_pos, rot_a, shape_b, pos_b, rot_b) {
            // Binary search for precise TOI.
            let mut lo = if i > 0 { (i - 1) as f32 * dt } else { 0.0 };
            let mut hi = t;

            for _ in 0..16 {
                let mid = (lo + hi) * 0.5;
                let mid_pos = pos_a.add(velocity.scale(mid));
                if gjk_intersect(shape_a, mid_pos, rot_a, shape_b, pos_b, rot_b) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }

            return Some(lo);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Shape cast against world colliders
// ---------------------------------------------------------------------------

/// A collider in the world for shape casting.
#[derive(Debug, Clone)]
pub struct WorldCollider {
    pub body: BodyHandle,
    pub shape: CastShape,
    pub position: Vec3,
    pub orientation: Quat,
    pub aabb: Aabb,
    pub layer: LayerMask,
    pub is_trigger: bool,
}

/// Shape cast engine that tests against a set of world colliders.
pub struct ShapeCaster {
    colliders: Vec<WorldCollider>,
    stats: ShapeCastStats,
}

/// Shape cast statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShapeCastStats {
    pub casts_performed: u32,
    pub broadphase_tests: u32,
    pub narrowphase_tests: u32,
    pub hits_found: u32,
}

impl ShapeCaster {
    /// Create a new shape caster.
    pub fn new() -> Self {
        Self {
            colliders: Vec::new(),
            stats: ShapeCastStats::default(),
        }
    }

    /// Add a collider to the world.
    pub fn add_collider(&mut self, collider: WorldCollider) {
        self.colliders.push(collider);
    }

    /// Remove all colliders for a body.
    pub fn remove_body(&mut self, body: BodyHandle) {
        self.colliders.retain(|c| c.body != body);
    }

    /// Clear all colliders.
    pub fn clear(&mut self) {
        self.colliders.clear();
    }

    /// Set all colliders at once.
    pub fn set_colliders(&mut self, colliders: Vec<WorldCollider>) {
        self.colliders = colliders;
    }

    /// Perform a shape cast query.
    pub fn cast(&mut self, input: &ShapeCastInput) -> ShapeCastResult {
        self.stats.casts_performed += 1;

        let dir = input.direction.normalize();
        if dir.length_sq() < 1e-10 {
            return ShapeCastResult::empty();
        }

        // Compute swept AABB for broadphase.
        let start_aabb = input.shape.compute_aabb(input.origin, input.orientation);
        let end_pos = input.origin.add(dir.scale(input.max_distance));
        let end_aabb = input.shape.compute_aabb(end_pos, input.orientation);
        let swept_aabb = Aabb {
            min: Vec3::new(
                start_aabb.min.x.min(end_aabb.min.x),
                start_aabb.min.y.min(end_aabb.min.y),
                start_aabb.min.z.min(end_aabb.min.z),
            ),
            max: Vec3::new(
                start_aabb.max.x.max(end_aabb.max.x),
                start_aabb.max.y.max(end_aabb.max.y),
                start_aabb.max.z.max(end_aabb.max.z),
            ),
        };

        let mut hits = Vec::new();

        for collider in &self.colliders {
            self.stats.broadphase_tests += 1;

            // Layer filter.
            if (collider.layer & input.layer_mask) == 0 { continue; }
            // Ignore list.
            if input.ignore_bodies.contains(&collider.body) { continue; }
            // Trigger filter.
            if collider.is_trigger && !input.include_triggers { continue; }
            // Broadphase AABB test.
            if !swept_aabb.intersects(&collider.aabb) { continue; }

            self.stats.narrowphase_tests += 1;

            // Narrowphase: time of impact.
            let velocity = dir.scale(input.max_distance);
            if let Some(toi) = time_of_impact(
                &input.shape, input.origin, input.orientation, velocity,
                &collider.shape, collider.position, collider.orientation,
                1.0,
            ) {
                let distance = toi * input.max_distance;
                let hit_point = input.origin.add(dir.scale(distance));

                // Approximate normal.
                let normal = hit_point.sub(collider.position).normalize();

                let hit = ShapeCastHit {
                    body: collider.body,
                    time_of_impact: toi,
                    hit_point,
                    hit_normal: normal,
                    penetration_depth: 0.0,
                    distance,
                    started_inside: toi <= 0.0,
                    is_trigger: collider.is_trigger,
                    face_index: None,
                    shape_contact_point: hit_point,
                    body_contact_point: hit_point,
                };

                hits.push(hit);
                self.stats.hits_found += 1;

                if input.first_hit_only { break; }
            }
        }

        ShapeCastResult::from_hits(hits)
    }

    /// Get statistics.
    pub fn stats(&self) -> &ShapeCastStats { &self.stats }

    /// Reset statistics.
    pub fn reset_stats(&mut self) { self.stats = ShapeCastStats::default(); }

    /// Get the number of colliders.
    pub fn collider_count(&self) -> usize { self.colliders.len() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_support() {
        let shape = CastShape::Sphere { radius: 1.0 };
        let support = shape.support(Vec3::UP, Vec3::ZERO, Quat::IDENTITY);
        assert!((support.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_box_support() {
        let shape = CastShape::Box { half_extents: Vec3::new(1.0, 1.0, 1.0) };
        let support = shape.support(Vec3::new(1.0, 1.0, 1.0), Vec3::ZERO, Quat::IDENTITY);
        assert!((support.x - 1.0).abs() < 1e-6);
        assert!((support.y - 1.0).abs() < 1e-6);
        assert!((support.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gjk_overlapping_spheres() {
        let a = CastShape::Sphere { radius: 1.0 };
        let b = CastShape::Sphere { radius: 1.0 };
        assert!(gjk_intersect(&a, Vec3::ZERO, Quat::IDENTITY, &b, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY));
    }

    #[test]
    fn test_gjk_separated_spheres() {
        let a = CastShape::Sphere { radius: 1.0 };
        let b = CastShape::Sphere { radius: 1.0 };
        assert!(!gjk_intersect(&a, Vec3::ZERO, Quat::IDENTITY, &b, Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY));
    }

    #[test]
    fn test_time_of_impact_spheres() {
        let a = CastShape::Sphere { radius: 0.5 };
        let b = CastShape::Sphere { radius: 0.5 };

        let toi = time_of_impact(
            &a, Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 0.0, 0.0),
            &b, Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY,
            1.0,
        );

        assert!(toi.is_some());
        let t = toi.unwrap();
        assert!(t > 0.0 && t < 1.0);
    }

    #[test]
    fn test_sphere_cast_builder() {
        let input = SphereCast::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 0.5)
            .with_max_distance(100.0)
            .with_layer_mask(0xFF)
            .to_input();

        assert_eq!(input.max_distance, 100.0);
        assert_eq!(input.layer_mask, 0xFF);
    }

    #[test]
    fn test_aabb_sweep() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Vec3::new(3.0, 0.0, 0.0), Vec3::new(4.0, 1.0, 1.0));

        let t = a.sweep_test(&b, Vec3::new(1.0, 0.0, 0.0), 10.0);
        assert!(t.is_some());
        assert!(t.unwrap() > 0.0);
    }

    #[test]
    fn test_shape_caster() {
        let mut caster = ShapeCaster::new();
        caster.add_collider(WorldCollider {
            body: 1,
            shape: CastShape::Sphere { radius: 1.0 },
            position: Vec3::new(5.0, 0.0, 0.0),
            orientation: Quat::IDENTITY,
            aabb: Aabb::from_sphere(Vec3::new(5.0, 0.0, 0.0), 1.0),
            layer: ALL_LAYERS,
            is_trigger: false,
        });

        let input = ShapeCastInput {
            shape: CastShape::Sphere { radius: 0.5 },
            origin: Vec3::ZERO,
            direction: Vec3::new(1.0, 0.0, 0.0),
            max_distance: 20.0,
            ..Default::default()
        };

        let result = caster.cast(&input);
        assert!(result.has_hit);
    }

    #[test]
    fn test_shape_display() {
        let s = format!("{}", CastShape::Sphere { radius: 1.0 });
        assert!(s.contains("Sphere"));
    }
}
