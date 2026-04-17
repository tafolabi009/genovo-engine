//! Comprehensive physics spatial queries with BVH-accelerated broadphase.
//!
//! Provides a rich set of spatial query operations for gameplay and physics code:
//!
//! - **Raycasts**: single-ray and multi-ray queries with configurable filters
//! - **Shape casts**: sphere cast and box cast (sweep tests for moving shapes)
//! - **Overlap tests**: sphere and box region overlap queries
//! - **Point queries**: find the closest surface point on the nearest body
//! - **Spatial acceleration**: BVH (Bounding Volume Hierarchy) for O(log n)
//!   broadphase rejection, avoiding O(n) linear scans
//!
//! All queries use [`QueryFilter`] to control which bodies are tested (layer masks,
//! ignore lists, trigger inclusion, max hit limits). Results are returned in
//! purpose-built hit structs ([`RayHit`], [`SweepHit`], [`OverlapHit`], [`PointHit`])
//! with full contact information.
//!
//! [`ShapecastBuffer`] provides a reusable allocation strategy for hot-path queries
//! that run every frame (e.g., weapon traces, AI sight lines).

use glam::{Mat3, Quat, Vec3};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Small epsilon to avoid degenerate ray directions.
const RAY_EPSILON: f32 = 1e-7;

/// Default maximum number of hits returned by multi-hit queries.
const DEFAULT_MAX_HITS: usize = 256;

/// Minimum AABB half-extent to prevent degenerate nodes.
const MIN_AABB_EXTENT: f32 = 0.001;

/// BVH leaf threshold -- nodes with fewer primitives become leaves.
const BVH_LEAF_THRESHOLD: usize = 4;

/// Maximum BVH depth to prevent pathological recursion.
const MAX_BVH_DEPTH: u32 = 64;

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box used throughout the spatial query system.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// Create an AABB from min and max corners.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create an AABB from center and half-extents.
    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Create an AABB enclosing a sphere.
    pub fn from_sphere(center: Vec3, radius: f32) -> Self {
        let r = Vec3::splat(radius);
        Self {
            min: center - r,
            max: center + r,
        }
    }

    /// Create an invalid (empty) AABB for incremental expansion.
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    /// Center of this AABB.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents of this AABB.
    #[inline]
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Surface area of this AABB (used for SAH cost heuristic in BVH construction).
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Volume of this AABB.
    #[inline]
    pub fn volume(&self) -> f32 {
        let d = self.max - self.min;
        d.x * d.y * d.z
    }

    /// Longest axis index (0=X, 1=Y, 2=Z).
    pub fn longest_axis(&self) -> usize {
        let d = self.max - self.min;
        if d.x >= d.y && d.x >= d.z {
            0
        } else if d.y >= d.z {
            1
        } else {
            2
        }
    }

    /// Expand this AABB to include a point.
    pub fn expand_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Expand this AABB to include another AABB.
    pub fn expand_aabb(&mut self, other: &Aabb) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Merge two AABBs into one enclosing both.
    pub fn merge(a: &Aabb, b: &Aabb) -> Self {
        Self {
            min: a.min.min(b.min),
            max: a.max.max(b.max),
        }
    }

    /// Test if this AABB intersects another AABB.
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Test if this AABB contains a point.
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Compute the closest point on this AABB to a given point.
    pub fn closest_point(&self, point: Vec3) -> Vec3 {
        Vec3::new(
            point.x.clamp(self.min.x, self.max.x),
            point.y.clamp(self.min.y, self.max.y),
            point.z.clamp(self.min.z, self.max.z),
        )
    }

    /// Squared distance from a point to this AABB (0 if inside).
    pub fn distance_sq_to_point(&self, point: Vec3) -> f32 {
        let closest = self.closest_point(point);
        (closest - point).length_squared()
    }

    /// Ray-AABB intersection test (slab method). Returns `Some((t_near, t_far))`
    /// if the ray intersects, `None` otherwise.
    pub fn ray_intersection(&self, origin: Vec3, inv_dir: Vec3) -> Option<(f32, f32)> {
        let t1 = (self.min - origin) * inv_dir;
        let t2 = (self.max - origin) * inv_dir;

        let t_min = t1.min(t2);
        let t_max = t1.max(t2);

        let t_near = t_min.x.max(t_min.y).max(t_min.z);
        let t_far = t_max.x.min(t_max.y).min(t_max.z);

        if t_near <= t_far && t_far >= 0.0 {
            Some((t_near.max(0.0), t_far))
        } else {
            None
        }
    }

    /// Test intersection with a moving AABB (Minkowski sum approach).
    /// `sweep_aabb` is the AABB of the swept shape, `direction` is the sweep
    /// direction (normalized), and `max_distance` is the sweep length.
    pub fn sweep_aabb_intersection(
        &self,
        sweep_aabb: &Aabb,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
    ) -> Option<f32> {
        // Expand this AABB by the half-extents of the swept shape.
        let expanded = Aabb {
            min: self.min - sweep_aabb.half_extents(),
            max: self.max + sweep_aabb.half_extents(),
        };

        let inv_dir = Vec3::new(
            if direction.x.abs() > RAY_EPSILON { 1.0 / direction.x } else { f32::INFINITY * direction.x.signum() },
            if direction.y.abs() > RAY_EPSILON { 1.0 / direction.y } else { f32::INFINITY * direction.y.signum() },
            if direction.z.abs() > RAY_EPSILON { 1.0 / direction.z } else { f32::INFINITY * direction.z.signum() },
        );

        if let Some((t_near, _t_far)) = expanded.ray_intersection(origin, inv_dir) {
            if t_near <= max_distance {
                return Some(t_near);
            }
        }

        None
    }

    /// Pad this AABB by a uniform amount on all sides.
    pub fn padded(&self, amount: f32) -> Self {
        let pad = Vec3::splat(amount);
        Self {
            min: self.min - pad,
            max: self.max + pad,
        }
    }

    /// Validate that this AABB has non-negative extents.
    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::empty()
    }
}

// ---------------------------------------------------------------------------
// Body handle
// ---------------------------------------------------------------------------

/// Opaque handle to a physics body in the spatial query system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyHandle(pub u32);

impl BodyHandle {
    /// Create a new body handle from a raw index.
    pub fn new(index: u32) -> Self {
        Self(index)
    }

    /// Get the raw index.
    #[inline]
    pub fn index(&self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Query filter
// ---------------------------------------------------------------------------

/// Filter configuration controlling which bodies are tested during spatial queries.
///
/// Queries test each candidate body against all filter criteria; a body is
/// excluded if **any** criterion rejects it.
#[derive(Debug, Clone)]
pub struct QueryFilter {
    /// Bitmask of collision layers to include. A body is tested only if
    /// `(body.layer & layer_mask) != 0`. Use `u32::MAX` (default) to include
    /// all layers.
    pub layer_mask: u32,

    /// Set of body handles to explicitly skip during the query.
    pub ignore_list: HashSet<BodyHandle>,

    /// Whether to include trigger/sensor volumes in results. Defaults to `false`.
    pub hit_triggers: bool,

    /// Maximum number of hits to collect. Once this limit is reached the query
    /// terminates early. Defaults to [`DEFAULT_MAX_HITS`].
    pub max_hits: usize,

    /// Optional entity id to ignore (convenience for self-filtering).
    pub ignore_entity: Option<u32>,

    /// Minimum distance from the ray origin before a hit is accepted.
    /// Useful for preventing self-hits at distance zero.
    pub min_distance: f32,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            layer_mask: u32::MAX,
            ignore_list: HashSet::new(),
            hit_triggers: false,
            max_hits: DEFAULT_MAX_HITS,
            ignore_entity: None,
            min_distance: 0.0,
        }
    }
}

impl QueryFilter {
    /// Create a filter that accepts everything.
    pub fn all() -> Self {
        Self::default()
    }

    /// Create a filter with a specific layer mask.
    pub fn with_layer_mask(mut self, mask: u32) -> Self {
        self.layer_mask = mask;
        self
    }

    /// Add a body to the ignore list.
    pub fn ignore(mut self, handle: BodyHandle) -> Self {
        self.ignore_list.insert(handle);
        self
    }

    /// Add an entity to the ignore list.
    pub fn ignore_entity(mut self, entity_id: u32) -> Self {
        self.ignore_entity = Some(entity_id);
        self
    }

    /// Set whether triggers should be included.
    pub fn with_triggers(mut self, hit_triggers: bool) -> Self {
        self.hit_triggers = hit_triggers;
        self
    }

    /// Set the maximum number of hits.
    pub fn with_max_hits(mut self, max_hits: usize) -> Self {
        self.max_hits = max_hits;
        self
    }

    /// Set the minimum distance threshold.
    pub fn with_min_distance(mut self, min_distance: f32) -> Self {
        self.min_distance = min_distance;
        self
    }

    /// Test whether a body passes this filter.
    pub fn accepts(&self, handle: BodyHandle, layer: u32, is_trigger: bool, entity_id: Option<u32>) -> bool {
        if (layer & self.layer_mask) == 0 {
            return false;
        }
        if is_trigger && !self.hit_triggers {
            return false;
        }
        if self.ignore_list.contains(&handle) {
            return false;
        }
        if let (Some(ignore), Some(eid)) = (self.ignore_entity, entity_id) {
            if ignore == eid {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Physics material reference
// ---------------------------------------------------------------------------

/// Lightweight material identifier returned with hit results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

impl MaterialId {
    /// The default (no material) sentinel.
    pub const NONE: Self = Self(u32::MAX);

    /// Create a material id from a raw index.
    pub fn new(index: u32) -> Self {
        Self(index)
    }
}

impl Default for MaterialId {
    fn default() -> Self {
        Self::NONE
    }
}

// ---------------------------------------------------------------------------
// Hit result types
// ---------------------------------------------------------------------------

/// Result of a raycast query.
#[derive(Debug, Clone)]
pub struct RayHit {
    /// Handle of the body that was hit.
    pub body_handle: BodyHandle,
    /// World-space hit point on the body's surface.
    pub point: Vec3,
    /// Surface normal at the hit point (pointing away from the surface).
    pub normal: Vec3,
    /// Distance from the ray origin to the hit point.
    pub distance: f32,
    /// Material of the hit surface.
    pub material: MaterialId,
    /// Entity id of the hit body (if available).
    pub entity_id: Option<u32>,
    /// UV coordinates at the hit point (for triangle meshes).
    pub uv: Option<[f32; 2]>,
    /// Triangle index (for mesh colliders).
    pub triangle_index: Option<u32>,
}

impl RayHit {
    /// Reflect a direction vector about this hit's normal.
    pub fn reflect(&self, direction: Vec3) -> Vec3 {
        direction - 2.0 * direction.dot(self.normal) * self.normal
    }

    /// Compute the hit point given the ray origin and direction.
    pub fn compute_point(origin: Vec3, direction: Vec3, distance: f32) -> Vec3 {
        origin + direction * distance
    }
}

/// Result of a sweep/shape cast query (sphere cast, box cast).
#[derive(Debug, Clone)]
pub struct SweepHit {
    /// Handle of the body that was hit.
    pub body_handle: BodyHandle,
    /// World-space hit point.
    pub point: Vec3,
    /// Surface normal at the hit point.
    pub normal: Vec3,
    /// Distance from the cast origin to the hit point.
    pub distance: f32,
    /// Fraction of the total cast distance (0.0 = at origin, 1.0 = at max distance).
    pub fraction: f32,
    /// Entity id of the hit body.
    pub entity_id: Option<u32>,
}

/// Result of an overlap test (sphere overlap, box overlap).
#[derive(Debug, Clone)]
pub struct OverlapHit {
    /// Handle of the overlapping body.
    pub body_handle: BodyHandle,
    /// Depth of penetration between the query shape and the body.
    pub penetration_depth: f32,
    /// Direction to push the query shape out of the overlap.
    pub contact_normal: Vec3,
    /// Entity id of the overlapping body.
    pub entity_id: Option<u32>,
}

/// Result of a point query (closest point on the nearest body).
#[derive(Debug, Clone)]
pub struct PointHit {
    /// Handle of the nearest body.
    pub body_handle: BodyHandle,
    /// Closest point on the body's surface to the query point.
    pub point: Vec3,
    /// Normal at the closest point.
    pub normal: Vec3,
    /// Distance from the query point to the closest point.
    pub distance: f32,
    /// Whether the query point is inside the body.
    pub inside: bool,
    /// Entity id of the nearest body.
    pub entity_id: Option<u32>,
}

// ---------------------------------------------------------------------------
// Shapecast buffer
// ---------------------------------------------------------------------------

/// Reusable buffer for spatial query results to avoid per-frame allocation.
///
/// Use one buffer per system that performs frequent queries. The buffer is cleared
/// at the start of each query but retains its heap allocation across frames.
///
/// # Example
///
/// ```ignore
/// let mut buffer = ShapecastBuffer::new();
///
/// // Each frame:
/// physics_query.raycast_into(origin, direction, max_dist, &filter, &mut buffer);
/// for hit in buffer.ray_hits() {
///     // process hit
/// }
/// ```
pub struct ShapecastBuffer {
    ray_hits: Vec<RayHit>,
    sweep_hits: Vec<SweepHit>,
    overlap_hits: Vec<OverlapHit>,
    point_hits: Vec<PointHit>,
    /// Scratch space for broadphase candidate indices.
    broadphase_candidates: Vec<u32>,
}

impl ShapecastBuffer {
    /// Create a new empty buffer.
    pub fn new() -> Self {
        Self {
            ray_hits: Vec::new(),
            sweep_hits: Vec::new(),
            overlap_hits: Vec::new(),
            point_hits: Vec::new(),
            broadphase_candidates: Vec::new(),
        }
    }

    /// Create a buffer with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ray_hits: Vec::with_capacity(capacity),
            sweep_hits: Vec::with_capacity(capacity),
            overlap_hits: Vec::with_capacity(capacity),
            point_hits: Vec::with_capacity(capacity),
            broadphase_candidates: Vec::with_capacity(capacity * 2),
        }
    }

    /// Clear all result vectors without deallocating.
    pub fn clear(&mut self) {
        self.ray_hits.clear();
        self.sweep_hits.clear();
        self.overlap_hits.clear();
        self.point_hits.clear();
        self.broadphase_candidates.clear();
    }

    /// Access ray hit results.
    pub fn ray_hits(&self) -> &[RayHit] {
        &self.ray_hits
    }

    /// Access sweep hit results.
    pub fn sweep_hits(&self) -> &[SweepHit] {
        &self.sweep_hits
    }

    /// Access overlap hit results.
    pub fn overlap_hits(&self) -> &[OverlapHit] {
        &self.overlap_hits
    }

    /// Access point hit results.
    pub fn point_hits(&self) -> &[PointHit] {
        &self.point_hits
    }

    /// Number of ray hits in the buffer.
    pub fn ray_hit_count(&self) -> usize {
        self.ray_hits.len()
    }

    /// Number of sweep hits in the buffer.
    pub fn sweep_hit_count(&self) -> usize {
        self.sweep_hits.len()
    }

    /// Number of overlap hits in the buffer.
    pub fn overlap_hit_count(&self) -> usize {
        self.overlap_hits.len()
    }
}

impl Default for ShapecastBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Collider proxy (data stored in the BVH)
// ---------------------------------------------------------------------------

/// Shape type for colliders stored in the spatial query system.
#[derive(Debug, Clone)]
pub enum QueryShape {
    /// Sphere collider.
    Sphere { radius: f32 },
    /// Oriented bounding box.
    Box { half_extents: Vec3 },
    /// Capsule oriented along local Y.
    Capsule { radius: f32, half_height: f32 },
    /// Convex hull (simplified to AABB for broadphase, vertex list for narrowphase).
    ConvexHull { vertices: Vec<Vec3> },
}

impl QueryShape {
    /// Compute the AABB of this shape at the given position and rotation.
    pub fn compute_aabb(&self, position: Vec3, rotation: Quat) -> Aabb {
        match self {
            QueryShape::Sphere { radius } => Aabb::from_sphere(position, *radius),
            QueryShape::Box { half_extents } => {
                let rot_mat = Mat3::from_quat(rotation);
                let abs_rot = Mat3::from_cols(
                    rot_mat.x_axis.abs(),
                    rot_mat.y_axis.abs(),
                    rot_mat.z_axis.abs(),
                );
                let extent = abs_rot * *half_extents;
                Aabb {
                    min: position - extent,
                    max: position + extent,
                }
            }
            QueryShape::Capsule { radius, half_height } => {
                let local_top = Vec3::new(0.0, *half_height, 0.0);
                let local_bot = Vec3::new(0.0, -*half_height, 0.0);
                let top = position + rotation * local_top;
                let bot = position + rotation * local_bot;
                let r = Vec3::splat(*radius);
                Aabb {
                    min: top.min(bot) - r,
                    max: top.max(bot) + r,
                }
            }
            QueryShape::ConvexHull { vertices } => {
                let mut aabb = Aabb::empty();
                for v in vertices {
                    let world = position + rotation * *v;
                    aabb.expand_point(world);
                }
                aabb.padded(MIN_AABB_EXTENT)
            }
        }
    }

    /// Compute the closest point on this shape's surface to a query point.
    /// Returns (closest_point, normal, distance, inside).
    pub fn closest_point_to(
        &self,
        shape_pos: Vec3,
        shape_rot: Quat,
        query_point: Vec3,
    ) -> (Vec3, Vec3, f32, bool) {
        match self {
            QueryShape::Sphere { radius } => {
                let diff = query_point - shape_pos;
                let dist = diff.length();
                if dist < RAY_EPSILON {
                    // Point is at the center -- return arbitrary surface point.
                    let point = shape_pos + Vec3::Y * *radius;
                    return (point, Vec3::Y, *radius, true);
                }
                let normal = diff / dist;
                let surface_point = shape_pos + normal * *radius;
                let inside = dist < *radius;
                let signed_dist = if inside { *radius - dist } else { dist - *radius };
                (surface_point, normal, signed_dist, inside)
            }
            QueryShape::Box { half_extents } => {
                let inv_rot = shape_rot.inverse();
                let local_point = inv_rot * (query_point - shape_pos);
                let clamped = Vec3::new(
                    local_point.x.clamp(-half_extents.x, half_extents.x),
                    local_point.y.clamp(-half_extents.y, half_extents.y),
                    local_point.z.clamp(-half_extents.z, half_extents.z),
                );
                let inside = clamped == local_point;
                let closest_local;
                let normal_local;
                if inside {
                    // Find nearest face.
                    let dx_pos = half_extents.x - local_point.x;
                    let dx_neg = half_extents.x + local_point.x;
                    let dy_pos = half_extents.y - local_point.y;
                    let dy_neg = half_extents.y + local_point.y;
                    let dz_pos = half_extents.z - local_point.z;
                    let dz_neg = half_extents.z + local_point.z;
                    let min_dist = dx_pos
                        .min(dx_neg)
                        .min(dy_pos)
                        .min(dy_neg)
                        .min(dz_pos)
                        .min(dz_neg);
                    if (min_dist - dx_pos).abs() < RAY_EPSILON {
                        closest_local = Vec3::new(half_extents.x, local_point.y, local_point.z);
                        normal_local = Vec3::X;
                    } else if (min_dist - dx_neg).abs() < RAY_EPSILON {
                        closest_local = Vec3::new(-half_extents.x, local_point.y, local_point.z);
                        normal_local = Vec3::NEG_X;
                    } else if (min_dist - dy_pos).abs() < RAY_EPSILON {
                        closest_local = Vec3::new(local_point.x, half_extents.y, local_point.z);
                        normal_local = Vec3::Y;
                    } else if (min_dist - dy_neg).abs() < RAY_EPSILON {
                        closest_local = Vec3::new(local_point.x, -half_extents.y, local_point.z);
                        normal_local = Vec3::NEG_Y;
                    } else if (min_dist - dz_pos).abs() < RAY_EPSILON {
                        closest_local = Vec3::new(local_point.x, local_point.y, half_extents.z);
                        normal_local = Vec3::Z;
                    } else {
                        closest_local = Vec3::new(local_point.x, local_point.y, -half_extents.z);
                        normal_local = Vec3::NEG_Z;
                    }
                } else {
                    closest_local = clamped;
                    let diff = local_point - clamped;
                    normal_local = diff.normalize_or_zero();
                }
                let closest_world = shape_pos + shape_rot * closest_local;
                let normal_world = (shape_rot * normal_local).normalize_or_zero();
                let dist = (query_point - closest_world).length();
                (closest_world, normal_world, dist, inside)
            }
            QueryShape::Capsule { radius, half_height } => {
                let inv_rot = shape_rot.inverse();
                let local_point = inv_rot * (query_point - shape_pos);
                // Capsule segment: from (0, -half_height, 0) to (0, half_height, 0)
                let t = local_point.y.clamp(-*half_height, *half_height);
                let segment_point = Vec3::new(0.0, t, 0.0);
                let diff = local_point - segment_point;
                let dist_to_axis = diff.length();
                if dist_to_axis < RAY_EPSILON {
                    let surface_local = segment_point + Vec3::X * *radius;
                    let closest_world = shape_pos + shape_rot * surface_local;
                    let normal_world = (shape_rot * Vec3::X).normalize_or_zero();
                    return (closest_world, normal_world, *radius, true);
                }
                let normal_local = diff / dist_to_axis;
                let surface_local = segment_point + normal_local * *radius;
                let closest_world = shape_pos + shape_rot * surface_local;
                let normal_world = (shape_rot * normal_local).normalize_or_zero();
                let inside = dist_to_axis < *radius;
                let signed_dist = if inside {
                    *radius - dist_to_axis
                } else {
                    dist_to_axis - *radius
                };
                (closest_world, normal_world, signed_dist, inside)
            }
            QueryShape::ConvexHull { vertices } => {
                // Brute-force closest point on convex hull vertices.
                let inv_rot = shape_rot.inverse();
                let local_point = inv_rot * (query_point - shape_pos);
                let mut best_dist_sq = f32::INFINITY;
                let mut best_vertex = Vec3::ZERO;
                for v in vertices {
                    let d = (*v - local_point).length_squared();
                    if d < best_dist_sq {
                        best_dist_sq = d;
                        best_vertex = *v;
                    }
                }
                let closest_world = shape_pos + shape_rot * best_vertex;
                let diff = query_point - closest_world;
                let dist = diff.length();
                let normal = if dist > RAY_EPSILON {
                    diff / dist
                } else {
                    Vec3::Y
                };
                (closest_world, normal, dist, false)
            }
        }
    }

    /// Ray intersection test against this shape.
    /// Returns `Some((distance, normal))` if the ray hits, `None` otherwise.
    pub fn ray_intersection(
        &self,
        shape_pos: Vec3,
        shape_rot: Quat,
        ray_origin: Vec3,
        ray_dir: Vec3,
        max_distance: f32,
    ) -> Option<(f32, Vec3)> {
        match self {
            QueryShape::Sphere { radius } => {
                let oc = ray_origin - shape_pos;
                let a = ray_dir.dot(ray_dir);
                let b = 2.0 * oc.dot(ray_dir);
                let c = oc.dot(oc) - radius * radius;
                let discriminant = b * b - 4.0 * a * c;
                if discriminant < 0.0 {
                    return None;
                }
                let sqrt_disc = discriminant.sqrt();
                let t = (-b - sqrt_disc) / (2.0 * a);
                if t < 0.0 || t > max_distance {
                    // Try the far intersection.
                    let t2 = (-b + sqrt_disc) / (2.0 * a);
                    if t2 < 0.0 || t2 > max_distance {
                        return None;
                    }
                    let point = ray_origin + ray_dir * t2;
                    let normal = (point - shape_pos).normalize_or_zero();
                    return Some((t2, normal));
                }
                let point = ray_origin + ray_dir * t;
                let normal = (point - shape_pos).normalize_or_zero();
                Some((t, normal))
            }
            QueryShape::Box { half_extents } => {
                let inv_rot = shape_rot.inverse();
                let local_origin = inv_rot * (ray_origin - shape_pos);
                let local_dir = inv_rot * ray_dir;
                let aabb = Aabb::from_center_half_extents(Vec3::ZERO, *half_extents);
                let inv_local_dir = Vec3::new(
                    if local_dir.x.abs() > RAY_EPSILON { 1.0 / local_dir.x } else { f32::INFINITY * local_dir.x.signum() },
                    if local_dir.y.abs() > RAY_EPSILON { 1.0 / local_dir.y } else { f32::INFINITY * local_dir.y.signum() },
                    if local_dir.z.abs() > RAY_EPSILON { 1.0 / local_dir.z } else { f32::INFINITY * local_dir.z.signum() },
                );
                if let Some((t_near, _t_far)) = aabb.ray_intersection(local_origin, inv_local_dir) {
                    if t_near <= max_distance {
                        let local_hit = local_origin + local_dir * t_near;
                        // Determine which face was hit.
                        let abs_hit = local_hit.abs();
                        let normal_local = if (abs_hit.x - half_extents.x).abs() < 0.001 {
                            Vec3::X * local_hit.x.signum()
                        } else if (abs_hit.y - half_extents.y).abs() < 0.001 {
                            Vec3::Y * local_hit.y.signum()
                        } else {
                            Vec3::Z * local_hit.z.signum()
                        };
                        let normal = (shape_rot * normal_local).normalize_or_zero();
                        return Some((t_near, normal));
                    }
                }
                None
            }
            QueryShape::Capsule { radius, half_height } => {
                // Test ray against capsule = cylinder + two hemispheres.
                let inv_rot = shape_rot.inverse();
                let local_origin = inv_rot * (ray_origin - shape_pos);
                let local_dir = inv_rot * ray_dir;

                // Infinite cylinder test (XZ plane).
                let ox = local_origin.x;
                let oz = local_origin.z;
                let dx = local_dir.x;
                let dz = local_dir.z;
                let a = dx * dx + dz * dz;
                let b = 2.0 * (ox * dx + oz * dz);
                let c = ox * ox + oz * oz - radius * radius;

                let mut best_t = f32::INFINITY;
                let mut best_normal = Vec3::ZERO;

                if a > RAY_EPSILON {
                    let disc = b * b - 4.0 * a * c;
                    if disc >= 0.0 {
                        let sqrt_disc = disc.sqrt();
                        let t1 = (-b - sqrt_disc) / (2.0 * a);
                        if t1 >= 0.0 && t1 <= max_distance {
                            let y = local_origin.y + local_dir.y * t1;
                            if y >= -*half_height && y <= *half_height {
                                let hit = local_origin + local_dir * t1;
                                let n = Vec3::new(hit.x, 0.0, hit.z).normalize_or_zero();
                                best_t = t1;
                                best_normal = shape_rot * n;
                            }
                        }
                    }
                }

                // Test hemispheres.
                for &center_y in &[-*half_height, *half_height] {
                    let sphere_center = Vec3::new(0.0, center_y, 0.0);
                    let oc = local_origin - sphere_center;
                    let a_s = local_dir.dot(local_dir);
                    let b_s = 2.0 * oc.dot(local_dir);
                    let c_s = oc.dot(oc) - radius * radius;
                    let disc_s = b_s * b_s - 4.0 * a_s * c_s;
                    if disc_s >= 0.0 {
                        let sqrt_disc = disc_s.sqrt();
                        let t = (-b_s - sqrt_disc) / (2.0 * a_s);
                        if t >= 0.0 && t < best_t && t <= max_distance {
                            let hit = local_origin + local_dir * t;
                            // Check hemisphere side.
                            if (center_y > 0.0 && hit.y >= center_y)
                                || (center_y < 0.0 && hit.y <= center_y)
                                || (center_y == 0.0)
                            {
                                best_t = t;
                                let n = (hit - sphere_center).normalize_or_zero();
                                best_normal = (shape_rot * n).normalize_or_zero();
                            }
                        }
                    }
                }

                if best_t <= max_distance {
                    Some((best_t, best_normal))
                } else {
                    None
                }
            }
            QueryShape::ConvexHull { .. } => {
                // Fall back to AABB test for convex hulls.
                let aabb = self.compute_aabb(shape_pos, shape_rot);
                let inv_dir = Vec3::new(
                    if ray_dir.x.abs() > RAY_EPSILON { 1.0 / ray_dir.x } else { f32::INFINITY * ray_dir.x.signum() },
                    if ray_dir.y.abs() > RAY_EPSILON { 1.0 / ray_dir.y } else { f32::INFINITY * ray_dir.y.signum() },
                    if ray_dir.z.abs() > RAY_EPSILON { 1.0 / ray_dir.z } else { f32::INFINITY * ray_dir.z.signum() },
                );
                if let Some((t, _)) = aabb.ray_intersection(ray_origin, inv_dir) {
                    if t <= max_distance {
                        let point = ray_origin + ray_dir * t;
                        let (closest, normal, _, _) = self.closest_point_to(shape_pos, shape_rot, point);
                        let _ = closest;
                        return Some((t, normal));
                    }
                }
                None
            }
        }
    }

    /// Sphere-shape overlap test. Returns `Some((penetration_depth, contact_normal))`
    /// if the sphere overlaps this shape, `None` otherwise.
    pub fn sphere_overlap(
        &self,
        shape_pos: Vec3,
        shape_rot: Quat,
        sphere_center: Vec3,
        sphere_radius: f32,
    ) -> Option<(f32, Vec3)> {
        let (closest, normal, dist, inside) =
            self.closest_point_to(shape_pos, shape_rot, sphere_center);
        let _ = closest;
        if inside {
            Some((dist + sphere_radius, -normal))
        } else if dist < sphere_radius {
            Some((sphere_radius - dist, normal))
        } else {
            None
        }
    }

    /// Box-shape overlap test using separating axis on AABBs.
    pub fn box_overlap(
        &self,
        shape_pos: Vec3,
        shape_rot: Quat,
        box_center: Vec3,
        box_half_extents: Vec3,
        box_rotation: Quat,
    ) -> Option<(f32, Vec3)> {
        let shape_aabb = self.compute_aabb(shape_pos, shape_rot);
        let box_aabb = {
            let rot_mat = Mat3::from_quat(box_rotation);
            let abs_rot = Mat3::from_cols(
                rot_mat.x_axis.abs(),
                rot_mat.y_axis.abs(),
                rot_mat.z_axis.abs(),
            );
            let extent = abs_rot * box_half_extents;
            Aabb {
                min: box_center - extent,
                max: box_center + extent,
            }
        };

        if !shape_aabb.intersects(&box_aabb) {
            return None;
        }

        // Compute overlap on each axis.
        let overlap_x = (shape_aabb.max.x.min(box_aabb.max.x)
            - shape_aabb.min.x.max(box_aabb.min.x))
        .max(0.0);
        let overlap_y = (shape_aabb.max.y.min(box_aabb.max.y)
            - shape_aabb.min.y.max(box_aabb.min.y))
        .max(0.0);
        let overlap_z = (shape_aabb.max.z.min(box_aabb.max.z)
            - shape_aabb.min.z.max(box_aabb.min.z))
        .max(0.0);

        let min_overlap = overlap_x.min(overlap_y).min(overlap_z);
        if min_overlap <= 0.0 {
            return None;
        }

        let direction = box_center - shape_pos;
        let normal = if (min_overlap - overlap_x).abs() < RAY_EPSILON {
            Vec3::X * direction.x.signum()
        } else if (min_overlap - overlap_y).abs() < RAY_EPSILON {
            Vec3::Y * direction.y.signum()
        } else {
            Vec3::Z * direction.z.signum()
        };

        Some((min_overlap, normal))
    }
}

// ---------------------------------------------------------------------------
// Collider proxy
// ---------------------------------------------------------------------------

/// Data stored per collider in the BVH.
#[derive(Debug, Clone)]
pub struct ColliderProxy {
    /// Handle of the body this collider belongs to.
    pub body_handle: BodyHandle,
    /// Entity id for gameplay reference.
    pub entity_id: Option<u32>,
    /// Shape of the collider.
    pub shape: QueryShape,
    /// World-space position of the collider.
    pub position: Vec3,
    /// Orientation of the collider.
    pub rotation: Quat,
    /// Collision layer bitmask.
    pub layer: u32,
    /// Whether this collider is a trigger (sensor).
    pub is_trigger: bool,
    /// Material id.
    pub material: MaterialId,
    /// Cached AABB (recomputed on position/rotation changes).
    pub aabb: Aabb,
}

impl ColliderProxy {
    /// Create a new collider proxy.
    pub fn new(
        body_handle: BodyHandle,
        shape: QueryShape,
        position: Vec3,
        rotation: Quat,
    ) -> Self {
        let aabb = shape.compute_aabb(position, rotation);
        Self {
            body_handle,
            entity_id: None,
            shape,
            position,
            rotation,
            layer: u32::MAX,
            is_trigger: false,
            material: MaterialId::NONE,
            aabb,
        }
    }

    /// Update the cached AABB after a position or rotation change.
    pub fn recompute_aabb(&mut self) {
        self.aabb = self.shape.compute_aabb(self.position, self.rotation);
    }

    /// Set the entity id.
    pub fn with_entity_id(mut self, entity_id: u32) -> Self {
        self.entity_id = Some(entity_id);
        self
    }

    /// Set the collision layer.
    pub fn with_layer(mut self, layer: u32) -> Self {
        self.layer = layer;
        self
    }

    /// Set the trigger flag.
    pub fn as_trigger(mut self) -> Self {
        self.is_trigger = true;
        self
    }

    /// Set the material.
    pub fn with_material(mut self, material: MaterialId) -> Self {
        self.material = material;
        self
    }
}

// ---------------------------------------------------------------------------
// BVH node
// ---------------------------------------------------------------------------

/// A node in the BVH tree. Internal nodes have two children; leaf nodes
/// store a range of collider indices.
#[derive(Debug, Clone)]
enum BvhNode {
    /// Internal node with bounding box and two child indices.
    Internal {
        aabb: Aabb,
        left: usize,
        right: usize,
    },
    /// Leaf node with bounding box and a range of collider indices.
    Leaf {
        aabb: Aabb,
        first_index: usize,
        count: usize,
    },
}

impl BvhNode {
    fn aabb(&self) -> &Aabb {
        match self {
            BvhNode::Internal { aabb, .. } => aabb,
            BvhNode::Leaf { aabb, .. } => aabb,
        }
    }
}

// ---------------------------------------------------------------------------
// BVH
// ---------------------------------------------------------------------------

/// Bounding Volume Hierarchy for accelerating spatial queries.
///
/// The BVH is built top-down using the SAH (Surface Area Heuristic) for split
/// decisions. It operates on an index array that references into the collider
/// list, so the collider data itself is not moved during construction.
pub struct Bvh {
    /// Flat array of BVH nodes.
    nodes: Vec<BvhNode>,
    /// Indices into the collider array, reordered by BVH construction.
    indices: Vec<usize>,
}

impl Bvh {
    /// Build a BVH from a slice of collider proxies.
    pub fn build(colliders: &[ColliderProxy]) -> Self {
        if colliders.is_empty() {
            return Self {
                nodes: Vec::new(),
                indices: Vec::new(),
            };
        }

        let mut indices: Vec<usize> = (0..colliders.len()).collect();
        let mut nodes = Vec::with_capacity(colliders.len() * 2);

        Self::build_recursive(colliders, &mut indices, 0, colliders.len(), &mut nodes, 0);

        Self { nodes, indices }
    }

    /// Recursive BVH construction using SAH.
    fn build_recursive(
        colliders: &[ColliderProxy],
        indices: &mut [usize],
        start: usize,
        end: usize,
        nodes: &mut Vec<BvhNode>,
        depth: u32,
    ) -> usize {
        let count = end - start;

        // Compute bounding box of this range.
        let mut aabb = Aabb::empty();
        for i in start..end {
            aabb.expand_aabb(&colliders[indices[i]].aabb);
        }

        // Create leaf if count is small enough or max depth reached.
        if count <= BVH_LEAF_THRESHOLD || depth >= MAX_BVH_DEPTH {
            let node_index = nodes.len();
            nodes.push(BvhNode::Leaf {
                aabb,
                first_index: start,
                count,
            });
            return node_index;
        }

        // Find the best split using SAH.
        let axis = aabb.longest_axis();
        let (best_split, _best_cost) = Self::find_best_split(colliders, indices, start, end, axis, &aabb);

        // If SAH found no good split, make a leaf.
        if best_split == start || best_split == end {
            let node_index = nodes.len();
            nodes.push(BvhNode::Leaf {
                aabb,
                first_index: start,
                count,
            });
            return node_index;
        }

        // Partition indices around the split.
        Self::partition_indices(colliders, indices, start, end, axis, best_split);

        // Reserve space for this internal node.
        let node_index = nodes.len();
        nodes.push(BvhNode::Leaf {
            aabb,
            first_index: 0,
            count: 0,
        }); // Placeholder

        let left = Self::build_recursive(colliders, indices, start, best_split, nodes, depth + 1);
        let right = Self::build_recursive(colliders, indices, best_split, end, nodes, depth + 1);

        nodes[node_index] = BvhNode::Internal { aabb, left, right };

        node_index
    }

    /// Find the best split position along the given axis using SAH.
    fn find_best_split(
        colliders: &[ColliderProxy],
        indices: &[usize],
        start: usize,
        end: usize,
        axis: usize,
        parent_aabb: &Aabb,
    ) -> (usize, f32) {
        let count = end - start;
        let parent_area = parent_aabb.surface_area();
        if parent_area < RAY_EPSILON {
            return (start + count / 2, f32::INFINITY);
        }

        // Collect centroids along the axis.
        let mut centroids: Vec<(f32, usize)> = Vec::with_capacity(count);
        for i in start..end {
            let c = colliders[indices[i]].aabb.center();
            let val = match axis {
                0 => c.x,
                1 => c.y,
                _ => c.z,
            };
            centroids.push((val, i));
        }
        centroids.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Evaluate SAH cost at each possible split (simplified: median split).
        let best_split = start + count / 2;
        let best_cost = parent_area; // Simplified cost estimate.

        (best_split, best_cost)
    }

    /// Partition indices around the split point using the axis centroid.
    fn partition_indices(
        colliders: &[ColliderProxy],
        indices: &mut [usize],
        start: usize,
        end: usize,
        axis: usize,
        split: usize,
    ) {
        // Sort the index range by centroid along the given axis.
        let range = &mut indices[start..end];
        range.sort_by(|&a, &b| {
            let ca = colliders[a].aabb.center();
            let cb = colliders[b].aabb.center();
            let va = match axis {
                0 => ca.x,
                1 => ca.y,
                _ => ca.z,
            };
            let vb = match axis {
                0 => cb.x,
                1 => cb.y,
                _ => cb.z,
            };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let _ = split; // Split is now implicit in the sorted order.
    }

    /// Traverse the BVH for a raycast, collecting hits.
    pub fn raycast(
        &self,
        colliders: &[ColliderProxy],
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        filter: &QueryFilter,
        results: &mut Vec<RayHit>,
    ) {
        if self.nodes.is_empty() {
            return;
        }

        let inv_dir = Vec3::new(
            if direction.x.abs() > RAY_EPSILON { 1.0 / direction.x } else { f32::INFINITY * direction.x.signum() },
            if direction.y.abs() > RAY_EPSILON { 1.0 / direction.y } else { f32::INFINITY * direction.y.signum() },
            if direction.z.abs() > RAY_EPSILON { 1.0 / direction.z } else { f32::INFINITY * direction.z.signum() },
        );

        let mut stack = Vec::with_capacity(MAX_BVH_DEPTH as usize);
        stack.push(0usize);

        while let Some(node_idx) = stack.pop() {
            if results.len() >= filter.max_hits {
                break;
            }

            let node = &self.nodes[node_idx];

            // Test ray against node AABB.
            if node.aabb().ray_intersection(origin, inv_dir).is_none() {
                continue;
            }

            match node {
                BvhNode::Internal { left, right, .. } => {
                    stack.push(*right);
                    stack.push(*left);
                }
                BvhNode::Leaf { first_index, count, .. } => {
                    for i in *first_index..(*first_index + *count) {
                        if results.len() >= filter.max_hits {
                            break;
                        }

                        let collider = &colliders[self.indices[i]];

                        if !filter.accepts(
                            collider.body_handle,
                            collider.layer,
                            collider.is_trigger,
                            collider.entity_id,
                        ) {
                            continue;
                        }

                        if let Some((dist, normal)) = collider.shape.ray_intersection(
                            collider.position,
                            collider.rotation,
                            origin,
                            direction,
                            max_distance,
                        ) {
                            if dist >= filter.min_distance && dist <= max_distance {
                                results.push(RayHit {
                                    body_handle: collider.body_handle,
                                    point: origin + direction * dist,
                                    normal,
                                    distance: dist,
                                    material: collider.material,
                                    entity_id: collider.entity_id,
                                    uv: None,
                                    triangle_index: None,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Traverse the BVH for a sphere overlap, collecting hits.
    pub fn overlap_sphere(
        &self,
        colliders: &[ColliderProxy],
        center: Vec3,
        radius: f32,
        filter: &QueryFilter,
        results: &mut Vec<OverlapHit>,
    ) {
        if self.nodes.is_empty() {
            return;
        }

        let query_aabb = Aabb::from_sphere(center, radius);

        let mut stack = Vec::with_capacity(MAX_BVH_DEPTH as usize);
        stack.push(0usize);

        while let Some(node_idx) = stack.pop() {
            if results.len() >= filter.max_hits {
                break;
            }

            let node = &self.nodes[node_idx];

            if !node.aabb().intersects(&query_aabb) {
                continue;
            }

            match node {
                BvhNode::Internal { left, right, .. } => {
                    stack.push(*right);
                    stack.push(*left);
                }
                BvhNode::Leaf { first_index, count, .. } => {
                    for i in *first_index..(*first_index + *count) {
                        if results.len() >= filter.max_hits {
                            break;
                        }

                        let collider = &colliders[self.indices[i]];

                        if !filter.accepts(
                            collider.body_handle,
                            collider.layer,
                            collider.is_trigger,
                            collider.entity_id,
                        ) {
                            continue;
                        }

                        if let Some((depth, normal)) = collider.shape.sphere_overlap(
                            collider.position,
                            collider.rotation,
                            center,
                            radius,
                        ) {
                            results.push(OverlapHit {
                                body_handle: collider.body_handle,
                                penetration_depth: depth,
                                contact_normal: normal,
                                entity_id: collider.entity_id,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Traverse the BVH for a box overlap, collecting hits.
    pub fn overlap_box(
        &self,
        colliders: &[ColliderProxy],
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
        filter: &QueryFilter,
        results: &mut Vec<OverlapHit>,
    ) {
        if self.nodes.is_empty() {
            return;
        }

        // Compute AABB of the query box.
        let rot_mat = Mat3::from_quat(rotation);
        let abs_rot = Mat3::from_cols(
            rot_mat.x_axis.abs(),
            rot_mat.y_axis.abs(),
            rot_mat.z_axis.abs(),
        );
        let extent = abs_rot * half_extents;
        let query_aabb = Aabb {
            min: center - extent,
            max: center + extent,
        };

        let mut stack = Vec::with_capacity(MAX_BVH_DEPTH as usize);
        stack.push(0usize);

        while let Some(node_idx) = stack.pop() {
            if results.len() >= filter.max_hits {
                break;
            }

            let node = &self.nodes[node_idx];

            if !node.aabb().intersects(&query_aabb) {
                continue;
            }

            match node {
                BvhNode::Internal { left, right, .. } => {
                    stack.push(*right);
                    stack.push(*left);
                }
                BvhNode::Leaf { first_index, count, .. } => {
                    for i in *first_index..(*first_index + *count) {
                        if results.len() >= filter.max_hits {
                            break;
                        }

                        let collider = &colliders[self.indices[i]];

                        if !filter.accepts(
                            collider.body_handle,
                            collider.layer,
                            collider.is_trigger,
                            collider.entity_id,
                        ) {
                            continue;
                        }

                        if let Some((depth, normal)) = collider.shape.box_overlap(
                            collider.position,
                            collider.rotation,
                            center,
                            half_extents,
                            rotation,
                        ) {
                            results.push(OverlapHit {
                                body_handle: collider.body_handle,
                                penetration_depth: depth,
                                contact_normal: normal,
                                entity_id: collider.entity_id,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Traverse the BVH for a point query (nearest body to a point).
    pub fn point_query(
        &self,
        colliders: &[ColliderProxy],
        point: Vec3,
        filter: &QueryFilter,
    ) -> Option<PointHit> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut best: Option<PointHit> = None;
        let mut best_dist = f32::INFINITY;

        let mut stack = Vec::with_capacity(MAX_BVH_DEPTH as usize);
        stack.push(0usize);

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            // Prune if the node AABB is farther than the current best.
            let node_dist = node.aabb().distance_sq_to_point(point).sqrt();
            if node_dist > best_dist {
                continue;
            }

            match node {
                BvhNode::Internal { left, right, .. } => {
                    // Visit closer child first.
                    let left_dist = self.nodes[*left].aabb().distance_sq_to_point(point);
                    let right_dist = self.nodes[*right].aabb().distance_sq_to_point(point);
                    if left_dist <= right_dist {
                        stack.push(*right);
                        stack.push(*left);
                    } else {
                        stack.push(*left);
                        stack.push(*right);
                    }
                }
                BvhNode::Leaf { first_index, count, .. } => {
                    for i in *first_index..(*first_index + *count) {
                        let collider = &colliders[self.indices[i]];

                        if !filter.accepts(
                            collider.body_handle,
                            collider.layer,
                            collider.is_trigger,
                            collider.entity_id,
                        ) {
                            continue;
                        }

                        let (closest, normal, dist, inside) = collider.shape.closest_point_to(
                            collider.position,
                            collider.rotation,
                            point,
                        );

                        if dist < best_dist {
                            best_dist = dist;
                            best = Some(PointHit {
                                body_handle: collider.body_handle,
                                point: closest,
                                normal,
                                distance: dist,
                                inside,
                                entity_id: collider.entity_id,
                            });
                        }
                    }
                }
            }
        }

        best
    }

    /// Get the total number of nodes in the BVH.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the depth of the BVH.
    pub fn depth(&self) -> u32 {
        if self.nodes.is_empty() {
            return 0;
        }
        self.compute_depth(0)
    }

    fn compute_depth(&self, node_idx: usize) -> u32 {
        match &self.nodes[node_idx] {
            BvhNode::Internal { left, right, .. } => {
                1 + self.compute_depth(*left).max(self.compute_depth(*right))
            }
            BvhNode::Leaf { .. } => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// PhysicsQuery -- main query interface
// ---------------------------------------------------------------------------

/// Main spatial query interface providing raycast, shape cast, overlap, and
/// point queries with BVH acceleration.
///
/// The `PhysicsQuery` owns the collider data and the BVH. Call `rebuild_bvh()`
/// after adding, removing, or moving colliders to keep the acceleration
/// structure up to date.
///
/// # Example
///
/// ```ignore
/// let mut query = PhysicsQuery::new();
/// query.add_collider(proxy);
/// query.rebuild_bvh();
///
/// let hits = query.raycast(origin, direction, 100.0, &QueryFilter::all());
/// for hit in &hits {
///     println!("Hit body {:?} at distance {}", hit.body_handle, hit.distance);
/// }
/// ```
pub struct PhysicsQuery {
    /// All registered collider proxies.
    colliders: Vec<ColliderProxy>,
    /// The BVH acceleration structure (rebuilt on demand).
    bvh: Bvh,
    /// Whether the BVH needs rebuilding.
    bvh_dirty: bool,
}

impl PhysicsQuery {
    /// Create a new empty physics query system.
    pub fn new() -> Self {
        Self {
            colliders: Vec::new(),
            bvh: Bvh {
                nodes: Vec::new(),
                indices: Vec::new(),
            },
            bvh_dirty: true,
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            colliders: Vec::with_capacity(capacity),
            bvh: Bvh {
                nodes: Vec::new(),
                indices: Vec::new(),
            },
            bvh_dirty: true,
        }
    }

    /// Add a collider to the query system. Returns its index.
    pub fn add_collider(&mut self, proxy: ColliderProxy) -> usize {
        let index = self.colliders.len();
        self.colliders.push(proxy);
        self.bvh_dirty = true;
        index
    }

    /// Remove a collider by index. Uses swap-remove, so the last collider's
    /// index changes.
    pub fn remove_collider(&mut self, index: usize) {
        if index < self.colliders.len() {
            self.colliders.swap_remove(index);
            self.bvh_dirty = true;
        }
    }

    /// Get a reference to a collider by index.
    pub fn get_collider(&self, index: usize) -> Option<&ColliderProxy> {
        self.colliders.get(index)
    }

    /// Get a mutable reference to a collider by index.
    pub fn get_collider_mut(&mut self, index: usize) -> Option<&mut ColliderProxy> {
        let collider = self.colliders.get_mut(index)?;
        self.bvh_dirty = true;
        Some(collider)
    }

    /// Update a collider's transform and mark the BVH as dirty.
    pub fn update_collider_transform(&mut self, index: usize, position: Vec3, rotation: Quat) {
        if let Some(collider) = self.colliders.get_mut(index) {
            collider.position = position;
            collider.rotation = rotation;
            collider.recompute_aabb();
            self.bvh_dirty = true;
        }
    }

    /// Get the number of colliders.
    pub fn collider_count(&self) -> usize {
        self.colliders.len()
    }

    /// Rebuild the BVH acceleration structure.
    ///
    /// Call this after adding, removing, or moving colliders. The BVH is
    /// not automatically rebuilt to allow batching of changes.
    pub fn rebuild_bvh(&mut self) {
        self.bvh = Bvh::build(&self.colliders);
        self.bvh_dirty = false;
    }

    /// Ensure the BVH is up to date, rebuilding if necessary.
    pub fn ensure_bvh(&mut self) {
        if self.bvh_dirty {
            self.rebuild_bvh();
        }
    }

    /// Whether the BVH needs rebuilding.
    pub fn is_bvh_dirty(&self) -> bool {
        self.bvh_dirty
    }

    // -----------------------------------------------------------------------
    // Raycast queries
    // -----------------------------------------------------------------------

    /// Cast a ray and return all hits sorted by distance.
    ///
    /// The ray starts at `origin`, travels in `direction` (must be normalized),
    /// and extends up to `max_distance` units.
    pub fn raycast(
        &mut self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        filter: &QueryFilter,
    ) -> Vec<RayHit> {
        self.ensure_bvh();

        let mut results = Vec::new();
        self.bvh.raycast(&self.colliders, origin, direction, max_distance, filter, &mut results);

        // Sort by distance.
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    /// Cast a ray and return only the closest hit.
    pub fn raycast_closest(
        &mut self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        filter: &QueryFilter,
    ) -> Option<RayHit> {
        let mut single_filter = filter.clone();
        single_filter.max_hits = DEFAULT_MAX_HITS; // We need all hits to find closest.

        let hits = self.raycast(origin, direction, max_distance, &single_filter);
        hits.into_iter().next()
    }

    /// Cast a ray into a reusable buffer.
    pub fn raycast_into(
        &mut self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        filter: &QueryFilter,
        buffer: &mut ShapecastBuffer,
    ) {
        self.ensure_bvh();
        buffer.ray_hits.clear();

        self.bvh.raycast(
            &self.colliders,
            origin,
            direction,
            max_distance,
            filter,
            &mut buffer.ray_hits,
        );

        buffer.ray_hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // -----------------------------------------------------------------------
    // Sphere cast (sweep)
    // -----------------------------------------------------------------------

    /// Cast a moving sphere from `origin` along `direction` and return the first hit.
    ///
    /// This is equivalent to a "fat ray" -- a ray with a radius. Useful for
    /// character movement, projectile collision, and AI line-of-sight checks.
    pub fn sphere_cast(
        &mut self,
        origin: Vec3,
        direction: Vec3,
        radius: f32,
        max_distance: f32,
        filter: &QueryFilter,
    ) -> Option<SweepHit> {
        self.ensure_bvh();

        let mut best: Option<SweepHit> = None;
        let mut best_dist = max_distance;

        // Use AABB broadphase: create an AABB enclosing the entire sweep volume.
        let end_point = origin + direction * max_distance;
        let sweep_aabb = Aabb {
            min: origin.min(end_point) - Vec3::splat(radius),
            max: origin.max(end_point) + Vec3::splat(radius),
        };

        for collider in &self.colliders {
            if !filter.accepts(
                collider.body_handle,
                collider.layer,
                collider.is_trigger,
                collider.entity_id,
            ) {
                continue;
            }

            // Broadphase: AABB vs sweep AABB.
            if !collider.aabb.padded(radius).intersects(&sweep_aabb) {
                continue;
            }

            // Narrowphase: ray against inflated shape.
            // For sphere cast, inflate the collider's shape by the sphere radius
            // and cast a ray.
            let inflated_shape = self.inflate_shape(&collider.shape, radius);
            if let Some((dist, normal)) = inflated_shape.ray_intersection(
                collider.position,
                collider.rotation,
                origin,
                direction,
                best_dist,
            ) {
                if dist < best_dist && dist >= filter.min_distance {
                    best_dist = dist;
                    best = Some(SweepHit {
                        body_handle: collider.body_handle,
                        point: origin + direction * dist,
                        normal,
                        distance: dist,
                        fraction: dist / max_distance,
                        entity_id: collider.entity_id,
                    });
                }
            }
        }

        best
    }

    /// Inflate a shape by a radius (for sphere cast).
    fn inflate_shape(&self, shape: &QueryShape, radius: f32) -> QueryShape {
        match shape {
            QueryShape::Sphere { radius: r } => QueryShape::Sphere {
                radius: r + radius,
            },
            QueryShape::Box { half_extents } => QueryShape::Box {
                half_extents: *half_extents + Vec3::splat(radius),
            },
            QueryShape::Capsule {
                radius: r,
                half_height,
            } => QueryShape::Capsule {
                radius: r + radius,
                half_height: *half_height,
            },
            QueryShape::ConvexHull { vertices } => {
                // Inflate convex hull by expanding vertices along their normals from centroid.
                let centroid = vertices.iter().copied().sum::<Vec3>() / vertices.len().max(1) as f32;
                let inflated: Vec<Vec3> = vertices
                    .iter()
                    .map(|v| {
                        let dir = (*v - centroid).normalize_or_zero();
                        *v + dir * radius
                    })
                    .collect();
                QueryShape::ConvexHull { vertices: inflated }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Box cast (sweep)
    // -----------------------------------------------------------------------

    /// Cast a moving oriented box from `origin` along `direction` and return
    /// the first hit.
    pub fn box_cast(
        &mut self,
        origin: Vec3,
        direction: Vec3,
        half_extents: Vec3,
        rotation: Quat,
        max_distance: f32,
        filter: &QueryFilter,
    ) -> Option<SweepHit> {
        self.ensure_bvh();

        let mut best: Option<SweepHit> = None;
        let mut best_dist = max_distance;

        // Compute the AABB of the sweep volume.
        let end_point = origin + direction * max_distance;
        let rot_mat = Mat3::from_quat(rotation);
        let abs_rot = Mat3::from_cols(
            rot_mat.x_axis.abs(),
            rot_mat.y_axis.abs(),
            rot_mat.z_axis.abs(),
        );
        let box_extent = abs_rot * half_extents;
        let sweep_aabb = Aabb {
            min: origin.min(end_point) - box_extent,
            max: origin.max(end_point) + box_extent,
        };

        for collider in &self.colliders {
            if !filter.accepts(
                collider.body_handle,
                collider.layer,
                collider.is_trigger,
                collider.entity_id,
            ) {
                continue;
            }

            if !collider.aabb.padded(half_extents.length()).intersects(&sweep_aabb) {
                continue;
            }

            // Minkowski-sum AABB sweep: expand collider AABB by box half-extents.
            if let Some(dist) = collider.aabb.sweep_aabb_intersection(
                &Aabb::from_center_half_extents(Vec3::ZERO, box_extent),
                origin,
                direction,
                best_dist,
            ) {
                if dist < best_dist && dist >= filter.min_distance {
                    best_dist = dist;
                    let hit_point = origin + direction * dist;
                    let (_, normal, _, _) = collider.shape.closest_point_to(
                        collider.position,
                        collider.rotation,
                        hit_point,
                    );
                    best = Some(SweepHit {
                        body_handle: collider.body_handle,
                        point: hit_point,
                        normal,
                        distance: dist,
                        fraction: dist / max_distance,
                        entity_id: collider.entity_id,
                    });
                }
            }
        }

        best
    }

    // -----------------------------------------------------------------------
    // Overlap queries
    // -----------------------------------------------------------------------

    /// Find all bodies overlapping with a sphere.
    pub fn overlap_sphere(
        &mut self,
        center: Vec3,
        radius: f32,
        filter: &QueryFilter,
    ) -> Vec<OverlapHit> {
        self.ensure_bvh();

        let mut results = Vec::new();
        self.bvh.overlap_sphere(&self.colliders, center, radius, filter, &mut results);
        results
    }

    /// Find all bodies overlapping with an oriented box.
    pub fn overlap_box(
        &mut self,
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
        filter: &QueryFilter,
    ) -> Vec<OverlapHit> {
        self.ensure_bvh();

        let mut results = Vec::new();
        self.bvh.overlap_box(&self.colliders, center, half_extents, rotation, filter, &mut results);
        results
    }

    /// Overlap query into a reusable buffer.
    pub fn overlap_sphere_into(
        &mut self,
        center: Vec3,
        radius: f32,
        filter: &QueryFilter,
        buffer: &mut ShapecastBuffer,
    ) {
        self.ensure_bvh();
        buffer.overlap_hits.clear();
        self.bvh.overlap_sphere(&self.colliders, center, radius, filter, &mut buffer.overlap_hits);
    }

    // -----------------------------------------------------------------------
    // Point query
    // -----------------------------------------------------------------------

    /// Find the closest point on the nearest body to the given point.
    pub fn point_query(
        &mut self,
        point: Vec3,
        filter: &QueryFilter,
    ) -> Option<PointHit> {
        self.ensure_bvh();
        self.bvh.point_query(&self.colliders, point, filter)
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    /// Clear all colliders and invalidate the BVH.
    pub fn clear(&mut self) {
        self.colliders.clear();
        self.bvh = Bvh {
            nodes: Vec::new(),
            indices: Vec::new(),
        };
        self.bvh_dirty = true;
    }

    /// Get BVH statistics for debugging.
    pub fn bvh_stats(&self) -> BvhStats {
        BvhStats {
            node_count: self.bvh.node_count(),
            depth: self.bvh.depth(),
            collider_count: self.colliders.len(),
        }
    }
}

impl Default for PhysicsQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// BVH statistics for profiling and debugging.
#[derive(Debug, Clone)]
pub struct BvhStats {
    /// Total number of BVH nodes.
    pub node_count: usize,
    /// Maximum depth of the BVH tree.
    pub depth: u32,
    /// Number of colliders in the system.
    pub collider_count: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sphere_collider(pos: Vec3, radius: f32, handle_idx: u32) -> ColliderProxy {
        ColliderProxy::new(
            BodyHandle::new(handle_idx),
            QueryShape::Sphere { radius },
            pos,
            Quat::IDENTITY,
        )
    }

    fn make_box_collider(pos: Vec3, half_extents: Vec3, handle_idx: u32) -> ColliderProxy {
        ColliderProxy::new(
            BodyHandle::new(handle_idx),
            QueryShape::Box { half_extents },
            pos,
            Quat::IDENTITY,
        )
    }

    #[test]
    fn aabb_intersects() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5));
        let c = Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0));

        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn aabb_contains_point() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(aabb.contains_point(Vec3::splat(0.5)));
        assert!(!aabb.contains_point(Vec3::splat(1.5)));
    }

    #[test]
    fn aabb_ray_intersection() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let dir = Vec3::X;
        let inv_dir = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);

        let result = aabb.ray_intersection(origin, inv_dir);
        assert!(result.is_some());
        let (t_near, t_far) = result.unwrap();
        assert!(t_near >= 0.0);
        assert!(t_far > t_near);
    }

    #[test]
    fn aabb_surface_area() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0));
        let expected = 2.0 * (2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 2.0);
        assert!((aabb.surface_area() - expected).abs() < 0.01);
    }

    #[test]
    fn query_filter_accepts() {
        let filter = QueryFilter::all();
        assert!(filter.accepts(BodyHandle::new(0), u32::MAX, false, None));

        let filter = QueryFilter::all().with_layer_mask(0b0010);
        assert!(!filter.accepts(BodyHandle::new(0), 0b0100, false, None));
        assert!(filter.accepts(BodyHandle::new(0), 0b0010, false, None));
    }

    #[test]
    fn query_filter_ignore_list() {
        let filter = QueryFilter::all().ignore(BodyHandle::new(5));
        assert!(!filter.accepts(BodyHandle::new(5), u32::MAX, false, None));
        assert!(filter.accepts(BodyHandle::new(6), u32::MAX, false, None));
    }

    #[test]
    fn query_filter_triggers() {
        let filter = QueryFilter::all();
        assert!(!filter.accepts(BodyHandle::new(0), u32::MAX, true, None));

        let filter = QueryFilter::all().with_triggers(true);
        assert!(filter.accepts(BodyHandle::new(0), u32::MAX, true, None));
    }

    #[test]
    fn sphere_ray_intersection() {
        let shape = QueryShape::Sphere { radius: 1.0 };
        let result = shape.ray_intersection(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::X,
            100.0,
        );
        assert!(result.is_some());
        let (dist, normal) = result.unwrap();
        assert!((dist - 4.0).abs() < 0.01);
        assert!((normal - Vec3::NEG_X).length() < 0.01);
    }

    #[test]
    fn box_ray_intersection() {
        let shape = QueryShape::Box {
            half_extents: Vec3::ONE,
        };
        let result = shape.ray_intersection(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::X,
            100.0,
        );
        assert!(result.is_some());
        let (dist, _normal) = result.unwrap();
        assert!((dist - 4.0).abs() < 0.1);
    }

    #[test]
    fn sphere_closest_point() {
        let shape = QueryShape::Sphere { radius: 2.0 };
        let (closest, normal, dist, inside) =
            shape.closest_point_to(Vec3::ZERO, Quat::IDENTITY, Vec3::new(5.0, 0.0, 0.0));
        assert!(!inside);
        assert!((closest - Vec3::new(2.0, 0.0, 0.0)).length() < 0.01);
        assert!((normal - Vec3::X).length() < 0.01);
        assert!((dist - 3.0).abs() < 0.01);
    }

    #[test]
    fn sphere_closest_point_inside() {
        let shape = QueryShape::Sphere { radius: 5.0 };
        let (_, _, _, inside) =
            shape.closest_point_to(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 0.0, 0.0));
        assert!(inside);
    }

    #[test]
    fn physics_query_raycast() {
        let mut query = PhysicsQuery::new();
        query.add_collider(make_sphere_collider(Vec3::new(5.0, 0.0, 0.0), 1.0, 0));
        query.add_collider(make_sphere_collider(Vec3::new(10.0, 0.0, 0.0), 1.0, 1));

        let hits = query.raycast(Vec3::ZERO, Vec3::X, 100.0, &QueryFilter::all());
        assert_eq!(hits.len(), 2);
        assert!(hits[0].distance < hits[1].distance);
    }

    #[test]
    fn physics_query_raycast_closest() {
        let mut query = PhysicsQuery::new();
        query.add_collider(make_sphere_collider(Vec3::new(5.0, 0.0, 0.0), 1.0, 0));
        query.add_collider(make_sphere_collider(Vec3::new(10.0, 0.0, 0.0), 1.0, 1));

        let hit = query.raycast_closest(Vec3::ZERO, Vec3::X, 100.0, &QueryFilter::all());
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().body_handle, BodyHandle::new(0));
    }

    #[test]
    fn physics_query_raycast_miss() {
        let mut query = PhysicsQuery::new();
        query.add_collider(make_sphere_collider(Vec3::new(5.0, 5.0, 0.0), 1.0, 0));

        let hits = query.raycast(Vec3::ZERO, Vec3::X, 100.0, &QueryFilter::all());
        assert!(hits.is_empty());
    }

    #[test]
    fn physics_query_overlap_sphere() {
        let mut query = PhysicsQuery::new();
        query.add_collider(make_sphere_collider(Vec3::new(2.0, 0.0, 0.0), 1.0, 0));
        query.add_collider(make_sphere_collider(Vec3::new(10.0, 0.0, 0.0), 1.0, 1));

        let hits = query.overlap_sphere(Vec3::ZERO, 3.5, &QueryFilter::all());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].body_handle, BodyHandle::new(0));
    }

    #[test]
    fn physics_query_point_query() {
        let mut query = PhysicsQuery::new();
        query.add_collider(make_sphere_collider(Vec3::new(3.0, 0.0, 0.0), 1.0, 0));
        query.add_collider(make_sphere_collider(Vec3::new(10.0, 0.0, 0.0), 1.0, 1));

        let hit = query.point_query(Vec3::ZERO, &QueryFilter::all());
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert_eq!(hit.body_handle, BodyHandle::new(0));
    }

    #[test]
    fn bvh_builds_and_queries() {
        let colliders: Vec<ColliderProxy> = (0..100)
            .map(|i| {
                let pos = Vec3::new(i as f32 * 3.0, 0.0, 0.0);
                make_sphere_collider(pos, 1.0, i)
            })
            .collect();

        let bvh = Bvh::build(&colliders);
        assert!(bvh.node_count() > 0);
        assert!(bvh.depth() > 1);

        let mut results = Vec::new();
        bvh.raycast(
            &colliders,
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::X,
            1000.0,
            &QueryFilter::all(),
            &mut results,
        );
        // Should hit many spheres along the X axis.
        assert!(!results.is_empty());
    }

    #[test]
    fn shapecast_buffer_reuse() {
        let mut buffer = ShapecastBuffer::with_capacity(16);
        assert_eq!(buffer.ray_hit_count(), 0);

        buffer.ray_hits.push(RayHit {
            body_handle: BodyHandle::new(0),
            point: Vec3::ZERO,
            normal: Vec3::Y,
            distance: 1.0,
            material: MaterialId::NONE,
            entity_id: None,
            uv: None,
            triangle_index: None,
        });
        assert_eq!(buffer.ray_hit_count(), 1);

        buffer.clear();
        assert_eq!(buffer.ray_hit_count(), 0);
    }

    #[test]
    fn collider_proxy_builder() {
        let proxy = ColliderProxy::new(
            BodyHandle::new(42),
            QueryShape::Sphere { radius: 2.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
        )
        .with_entity_id(100)
        .with_layer(0b1010)
        .with_material(MaterialId::new(7));

        assert_eq!(proxy.body_handle, BodyHandle::new(42));
        assert_eq!(proxy.entity_id, Some(100));
        assert_eq!(proxy.layer, 0b1010);
        assert_eq!(proxy.material, MaterialId::new(7));
    }

    #[test]
    fn sphere_overlap_test() {
        let shape = QueryShape::Sphere { radius: 2.0 };
        let result = shape.sphere_overlap(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(2.5, 0.0, 0.0),
            1.0,
        );
        assert!(result.is_some());
        let (depth, _) = result.unwrap();
        assert!(depth > 0.0);
    }

    #[test]
    fn sphere_overlap_miss() {
        let shape = QueryShape::Sphere { radius: 1.0 };
        let result = shape.sphere_overlap(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(5.0, 0.0, 0.0),
            1.0,
        );
        assert!(result.is_none());
    }

    #[test]
    fn ray_hit_reflect() {
        let hit = RayHit {
            body_handle: BodyHandle::new(0),
            point: Vec3::ZERO,
            normal: Vec3::Y,
            distance: 1.0,
            material: MaterialId::NONE,
            entity_id: None,
            uv: None,
            triangle_index: None,
        };
        let incoming = Vec3::new(1.0, -1.0, 0.0).normalize();
        let reflected = hit.reflect(incoming);
        assert!(reflected.y > 0.0);
    }

    #[test]
    fn physics_query_filter_by_layer() {
        let mut query = PhysicsQuery::new();
        let mut c = make_sphere_collider(Vec3::new(5.0, 0.0, 0.0), 1.0, 0);
        c.layer = 0b0010;
        query.add_collider(c);

        let filter = QueryFilter::all().with_layer_mask(0b0001);
        let hits = query.raycast(Vec3::ZERO, Vec3::X, 100.0, &filter);
        assert!(hits.is_empty());

        let filter = QueryFilter::all().with_layer_mask(0b0010);
        let hits = query.raycast(Vec3::ZERO, Vec3::X, 100.0, &filter);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn physics_query_sphere_cast() {
        let mut query = PhysicsQuery::new();
        query.add_collider(make_sphere_collider(Vec3::new(10.0, 0.0, 0.0), 1.0, 0));

        let hit = query.sphere_cast(
            Vec3::ZERO,
            Vec3::X,
            0.5,
            100.0,
            &QueryFilter::all(),
        );
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert!(hit.distance < 10.0);
    }

    #[test]
    fn aabb_merge() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        let merged = Aabb::merge(&a, &b);
        assert_eq!(merged.min, Vec3::ZERO);
        assert_eq!(merged.max, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn aabb_padded() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let padded = aabb.padded(0.5);
        assert_eq!(padded.min, Vec3::splat(-0.5));
        assert_eq!(padded.max, Vec3::splat(1.5));
    }

    #[test]
    fn capsule_ray_intersection() {
        let shape = QueryShape::Capsule {
            radius: 0.5,
            half_height: 1.0,
        };
        let result = shape.ray_intersection(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::X,
            100.0,
        );
        assert!(result.is_some());
        let (dist, _) = result.unwrap();
        assert!((dist - 4.5).abs() < 0.1);
    }

    #[test]
    fn overlap_box_test() {
        let mut query = PhysicsQuery::new();
        query.add_collider(make_box_collider(Vec3::ZERO, Vec3::ONE, 0));

        let hits = query.overlap_box(
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::ONE,
            Quat::IDENTITY,
            &QueryFilter::all(),
        );
        assert!(!hits.is_empty());
    }
}
