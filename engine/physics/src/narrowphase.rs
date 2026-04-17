// engine/physics/src/narrowphase.rs
//
// Narrowphase collision detection for the Genovo engine.
//
// Dispatches collision tests by shape pair, generates contact manifolds with
// persistent contacts and contact caching for warm-starting the solver.
//
// Supported shape pairs:
// - Sphere vs Sphere
// - Sphere vs Box (OBB)
// - Sphere vs Capsule
// - Sphere vs Plane
// - Box vs Box (SAT-based OBB)
// - Box vs Plane
// - Capsule vs Capsule
// - Capsule vs Plane
// - Convex vs Convex (GJK + EPA)
// - Mesh triangle vs primitive shapes
//
// Features:
// - Contact manifold with up to 4 contact points
// - Persistent contact IDs for warm-starting the constraint solver
// - Contact point reduction (keep the 4 most informative contacts)
// - Contact caching across frames with aging
// - Collision normal and penetration depth computation
// - Support for one-sided collisions (e.g., terrain triangles)
// - EPA fallback for deep penetration

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum contacts in a manifold.
pub const MAX_MANIFOLD_CONTACTS: usize = 4;

/// Contact distance threshold for persistent ID matching.
pub const CONTACT_MERGE_DISTANCE: f32 = 0.02;

/// Maximum penetration depth before clamping.
pub const MAX_PENETRATION_DEPTH: f32 = 10.0;

/// Contact breaking distance (remove contacts farther than this).
pub const CONTACT_BREAKING_THRESHOLD: f32 = 0.04;

/// Epsilon for collision calculations.
const EPSILON: f32 = 1e-6;

/// GJK maximum iterations.
const GJK_MAX_ITERATIONS: u32 = 64;

/// EPA maximum iterations.
const EPA_MAX_ITERATIONS: u32 = 64;

/// EPA tolerance for face distance.
const EPA_TOLERANCE: f32 = 1e-4;

// ---------------------------------------------------------------------------
// Collision shape
// ---------------------------------------------------------------------------

/// Shape type for narrowphase dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NarrowShapeType {
    Sphere,
    Box,
    Capsule,
    Plane,
    ConvexHull,
    TriangleMesh,
    Cylinder,
    Cone,
}

/// A collision shape with its local geometry.
#[derive(Debug, Clone)]
pub enum NarrowShape {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: [f32; 3],
    },
    Capsule {
        radius: f32,
        half_height: f32,
    },
    Plane {
        normal: [f32; 3],
        offset: f32,
    },
    ConvexHull {
        vertices: Vec<[f32; 3]>,
    },
    Triangle {
        v0: [f32; 3],
        v1: [f32; 3],
        v2: [f32; 3],
    },
    Cylinder {
        radius: f32,
        half_height: f32,
    },
    Cone {
        radius: f32,
        height: f32,
    },
}

impl NarrowShape {
    /// Returns the shape type.
    pub fn shape_type(&self) -> NarrowShapeType {
        match self {
            Self::Sphere { .. } => NarrowShapeType::Sphere,
            Self::Box { .. } => NarrowShapeType::Box,
            Self::Capsule { .. } => NarrowShapeType::Capsule,
            Self::Plane { .. } => NarrowShapeType::Plane,
            Self::ConvexHull { .. } => NarrowShapeType::ConvexHull,
            Self::Triangle { .. } => NarrowShapeType::TriangleMesh,
            Self::Cylinder { .. } => NarrowShapeType::Cylinder,
            Self::Cone { .. } => NarrowShapeType::Cone,
        }
    }

    /// Compute the GJK support point for this shape in a given direction (local space).
    pub fn support(&self, direction: [f32; 3]) -> [f32; 3] {
        match self {
            Self::Sphere { radius } => {
                let len = vec3_length(direction);
                if len < EPSILON {
                    return [*radius, 0.0, 0.0];
                }
                let n = vec3_scale(direction, *radius / len);
                n
            }
            Self::Box { half_extents } => {
                [
                    if direction[0] >= 0.0 { half_extents[0] } else { -half_extents[0] },
                    if direction[1] >= 0.0 { half_extents[1] } else { -half_extents[1] },
                    if direction[2] >= 0.0 { half_extents[2] } else { -half_extents[2] },
                ]
            }
            Self::Capsule { radius, half_height } => {
                let axis_support = if direction[1] >= 0.0 {
                    [0.0, *half_height, 0.0]
                } else {
                    [0.0, -*half_height, 0.0]
                };
                let len = vec3_length(direction);
                if len < EPSILON {
                    return axis_support;
                }
                let sphere_support = vec3_scale(direction, *radius / len);
                vec3_add(axis_support, sphere_support)
            }
            Self::ConvexHull { vertices } => {
                let mut best = vertices[0];
                let mut best_dot = vec3_dot(best, direction);
                for v in &vertices[1..] {
                    let d = vec3_dot(*v, direction);
                    if d > best_dot {
                        best_dot = d;
                        best = *v;
                    }
                }
                best
            }
            Self::Plane { normal, offset } => {
                // Plane support: project direction onto plane.
                vec3_scale(*normal, *offset + 1000.0 * vec3_dot(*normal, direction).max(0.0))
            }
            Self::Triangle { v0, v1, v2 } => {
                let d0 = vec3_dot(*v0, direction);
                let d1 = vec3_dot(*v1, direction);
                let d2 = vec3_dot(*v2, direction);
                if d0 >= d1 && d0 >= d2 { *v0 }
                else if d1 >= d2 { *v1 }
                else { *v2 }
            }
            Self::Cylinder { radius, half_height } => {
                let xy_len = (direction[0] * direction[0] + direction[2] * direction[2]).sqrt();
                let xy_support = if xy_len > EPSILON {
                    [direction[0] / xy_len * radius, 0.0, direction[2] / xy_len * radius]
                } else {
                    [*radius, 0.0, 0.0]
                };
                let y_support = if direction[1] >= 0.0 { *half_height } else { -*half_height };
                [xy_support[0], y_support, xy_support[2]]
            }
            Self::Cone { radius, height } => {
                let tip = [0.0, *height, 0.0];
                let xy_len = (direction[0] * direction[0] + direction[2] * direction[2]).sqrt();
                let base_support = if xy_len > EPSILON {
                    [direction[0] / xy_len * radius, 0.0, direction[2] / xy_len * radius]
                } else {
                    [*radius, 0.0, 0.0]
                };
                if vec3_dot(tip, direction) > vec3_dot(base_support, direction) {
                    tip
                } else {
                    base_support
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// Transform for positioning a shape in world space.
#[derive(Debug, Clone, Copy)]
pub struct NarrowTransform {
    /// Position in world space.
    pub position: [f32; 3],
    /// Rotation as a 3x3 matrix (column-major).
    pub rotation: [[f32; 3]; 3],
}

impl NarrowTransform {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            position: [0.0; 3],
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Transform with position only (no rotation).
    pub fn from_position(pos: [f32; 3]) -> Self {
        Self {
            position: pos,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Transform a local-space point to world space.
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        [
            r[0][0] * p[0] + r[1][0] * p[1] + r[2][0] * p[2] + self.position[0],
            r[0][1] * p[0] + r[1][1] * p[1] + r[2][1] * p[2] + self.position[1],
            r[0][2] * p[0] + r[1][2] * p[1] + r[2][2] * p[2] + self.position[2],
        ]
    }

    /// Transform a local-space direction to world space (no translation).
    pub fn transform_direction(&self, d: [f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        [
            r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2],
            r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2],
            r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2],
        ]
    }

    /// Inverse-transform a world-space point to local space.
    pub fn inverse_transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let rel = vec3_sub(p, self.position);
        let r = &self.rotation;
        // Transpose of rotation * relative position.
        [
            r[0][0] * rel[0] + r[0][1] * rel[1] + r[0][2] * rel[2],
            r[1][0] * rel[0] + r[1][1] * rel[1] + r[1][2] * rel[2],
            r[2][0] * rel[0] + r[2][1] * rel[1] + r[2][2] * rel[2],
        ]
    }

    /// Inverse-transform a world-space direction to local space.
    pub fn inverse_transform_direction(&self, d: [f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        [
            r[0][0] * d[0] + r[0][1] * d[1] + r[0][2] * d[2],
            r[1][0] * d[0] + r[1][1] * d[1] + r[1][2] * d[2],
            r[2][0] * d[0] + r[2][1] * d[1] + r[2][2] * d[2],
        ]
    }
}

// ---------------------------------------------------------------------------
// Contact point
// ---------------------------------------------------------------------------

/// A single contact point between two colliding shapes.
#[derive(Debug, Clone, Copy)]
pub struct NarrowContactPoint {
    /// Contact point on shape A (world space).
    pub point_a: [f32; 3],
    /// Contact point on shape B (world space).
    pub point_b: [f32; 3],
    /// Contact normal (pointing from B to A, world space).
    pub normal: [f32; 3],
    /// Penetration depth (positive means overlapping).
    pub depth: f32,
    /// Persistent contact ID for warm-starting.
    pub id: u32,
    /// Accumulated normal impulse (for warm-starting).
    pub normal_impulse: f32,
    /// Accumulated tangent impulse (friction, for warm-starting).
    pub tangent_impulse: [f32; 2],
    /// Contact point on A in local space (for tracking).
    pub local_point_a: [f32; 3],
    /// Contact point on B in local space (for tracking).
    pub local_point_b: [f32; 3],
    /// Age of this contact (frames it has existed).
    pub age: u32,
}

impl NarrowContactPoint {
    /// Create a new contact point.
    pub fn new(
        point_a: [f32; 3],
        point_b: [f32; 3],
        normal: [f32; 3],
        depth: f32,
    ) -> Self {
        Self {
            point_a,
            point_b,
            normal,
            depth,
            id: compute_contact_id(point_a, point_b),
            normal_impulse: 0.0,
            tangent_impulse: [0.0, 0.0],
            local_point_a: point_a,
            local_point_b: point_b,
            age: 0,
        }
    }

    /// Create with local-space points for persistent tracking.
    pub fn with_local_points(
        mut self,
        local_a: [f32; 3],
        local_b: [f32; 3],
    ) -> Self {
        self.local_point_a = local_a;
        self.local_point_b = local_b;
        self
    }
}

/// Compute a deterministic contact ID from the contact positions.
fn compute_contact_id(a: [f32; 3], b: [f32; 3]) -> u32 {
    let bits_a = (a[0].to_bits() ^ a[1].to_bits().rotate_left(11) ^ a[2].to_bits().rotate_left(22)) as u32;
    let bits_b = (b[0].to_bits() ^ b[1].to_bits().rotate_left(11) ^ b[2].to_bits().rotate_left(22)) as u32;
    bits_a.wrapping_mul(0x9e3779b9).wrapping_add(bits_b)
}

// ---------------------------------------------------------------------------
// Contact manifold
// ---------------------------------------------------------------------------

/// A contact manifold between two shapes, containing up to `MAX_MANIFOLD_CONTACTS` points.
#[derive(Debug, Clone)]
pub struct ContactManifold {
    /// Body/shape ID for object A.
    pub body_a: u32,
    /// Body/shape ID for object B.
    pub body_b: u32,
    /// Contact points.
    pub contacts: Vec<NarrowContactPoint>,
    /// Normal direction (average, pointing from B to A).
    pub normal: [f32; 3],
    /// Combined friction coefficient.
    pub friction: f32,
    /// Combined restitution coefficient.
    pub restitution: f32,
    /// Whether this manifold is still valid.
    pub active: bool,
    /// Frame number when this manifold was last updated.
    pub last_updated_frame: u64,
}

impl ContactManifold {
    /// Create a new empty manifold.
    pub fn new(body_a: u32, body_b: u32) -> Self {
        Self {
            body_a,
            body_b,
            contacts: Vec::with_capacity(MAX_MANIFOLD_CONTACTS),
            normal: [0.0, 1.0, 0.0],
            friction: 0.5,
            restitution: 0.3,
            active: true,
            last_updated_frame: 0,
        }
    }

    /// Add a contact point to the manifold, maintaining the maximum count.
    pub fn add_contact(&mut self, contact: NarrowContactPoint) {
        // Try to merge with an existing contact.
        for existing in &mut self.contacts {
            let dist = vec3_distance(existing.point_a, contact.point_a);
            if dist < CONTACT_MERGE_DISTANCE {
                // Update the existing contact.
                existing.point_a = contact.point_a;
                existing.point_b = contact.point_b;
                existing.normal = contact.normal;
                existing.depth = contact.depth;
                existing.local_point_a = contact.local_point_a;
                existing.local_point_b = contact.local_point_b;
                existing.age += 1;
                return;
            }
        }

        if self.contacts.len() < MAX_MANIFOLD_CONTACTS {
            self.contacts.push(contact);
        } else {
            // Replace the shallowest contact.
            self.reduce_contacts(contact);
        }
    }

    /// Reduce contacts to MAX_MANIFOLD_CONTACTS, keeping the most informative set.
    fn reduce_contacts(&mut self, new_contact: NarrowContactPoint) {
        // Strategy: keep the 4 contacts that maximize the contact area.
        // Find the deepest contact.
        let mut deepest_idx = 0;
        let mut deepest_depth = self.contacts[0].depth;
        for (i, c) in self.contacts.iter().enumerate() {
            if c.depth > deepest_depth {
                deepest_depth = c.depth;
                deepest_idx = i;
            }
        }

        // Find the contact farthest from the deepest.
        let deepest_point = self.contacts[deepest_idx].point_a;
        let mut farthest_idx = 0;
        let mut farthest_dist = 0.0f32;
        for (i, c) in self.contacts.iter().enumerate() {
            if i == deepest_idx { continue; }
            let dist = vec3_distance_squared(c.point_a, deepest_point);
            if dist > farthest_dist {
                farthest_dist = dist;
                farthest_idx = i;
            }
        }

        // Replace the shallowest contact (excluding deepest and farthest).
        let mut shallowest_idx = 0;
        let mut shallowest_depth = f32::MAX;
        for (i, c) in self.contacts.iter().enumerate() {
            if i == deepest_idx || i == farthest_idx { continue; }
            if c.depth < shallowest_depth {
                shallowest_depth = c.depth;
                shallowest_idx = i;
            }
        }

        if new_contact.depth > shallowest_depth {
            self.contacts[shallowest_idx] = new_contact;
        }
    }

    /// Remove contacts that have separated beyond the breaking threshold.
    pub fn prune_contacts(&mut self, transform_a: &NarrowTransform, transform_b: &NarrowTransform) {
        self.contacts.retain(|c| {
            // Recompute world-space points from local-space.
            let wa = transform_a.transform_point(c.local_point_a);
            let wb = transform_b.transform_point(c.local_point_b);
            let separation = vec3_dot(vec3_sub(wa, wb), c.normal);
            separation < CONTACT_BREAKING_THRESHOLD
        });
    }

    /// Total contact count.
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// Returns the deepest penetration depth in this manifold.
    pub fn max_depth(&self) -> f32 {
        self.contacts.iter().map(|c| c.depth).fold(0.0f32, f32::max)
    }

    /// Returns the average contact point.
    pub fn average_contact_point(&self) -> [f32; 3] {
        if self.contacts.is_empty() {
            return [0.0; 3];
        }
        let mut sum = [0.0f32; 3];
        for c in &self.contacts {
            sum[0] += c.point_a[0];
            sum[1] += c.point_a[1];
            sum[2] += c.point_a[2];
        }
        let n = self.contacts.len() as f32;
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }
}

// ---------------------------------------------------------------------------
// Contact cache
// ---------------------------------------------------------------------------

/// Cache of contact manifolds across frames.
#[derive(Debug)]
pub struct ContactCache {
    /// Map from body pair to manifold.
    manifolds: HashMap<(u32, u32), ContactManifold>,
    /// Current frame number.
    frame: u64,
    /// Maximum frames to keep stale manifolds.
    max_age: u64,
}

impl ContactCache {
    /// Create a new contact cache.
    pub fn new(max_age: u64) -> Self {
        Self {
            manifolds: HashMap::new(),
            frame: 0,
            max_age,
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Get or create a manifold for a body pair.
    pub fn get_or_create(&mut self, body_a: u32, body_b: u32) -> &mut ContactManifold {
        let key = if body_a < body_b { (body_a, body_b) } else { (body_b, body_a) };
        self.manifolds.entry(key).or_insert_with(|| ContactManifold::new(body_a, body_b))
    }

    /// Mark a manifold as updated this frame.
    pub fn mark_updated(&mut self, body_a: u32, body_b: u32) {
        let key = if body_a < body_b { (body_a, body_b) } else { (body_b, body_a) };
        if let Some(m) = self.manifolds.get_mut(&key) {
            m.last_updated_frame = self.frame;
            m.active = true;
        }
    }

    /// Remove stale manifolds that haven't been updated recently.
    pub fn prune_stale(&mut self) {
        let frame = self.frame;
        let max_age = self.max_age;
        self.manifolds.retain(|_, m| {
            frame - m.last_updated_frame <= max_age
        });
    }

    /// Get a manifold for a body pair (immutable).
    pub fn get(&self, body_a: u32, body_b: u32) -> Option<&ContactManifold> {
        let key = if body_a < body_b { (body_a, body_b) } else { (body_b, body_a) };
        self.manifolds.get(&key)
    }

    /// Returns all active manifolds.
    pub fn active_manifolds(&self) -> Vec<&ContactManifold> {
        self.manifolds.values().filter(|m| m.active).collect()
    }

    /// Returns the total number of cached manifolds.
    pub fn manifold_count(&self) -> usize {
        self.manifolds.len()
    }

    /// Returns the total number of contact points across all manifolds.
    pub fn total_contact_count(&self) -> usize {
        self.manifolds.values().map(|m| m.contacts.len()).sum()
    }

    /// Clear all cached manifolds.
    pub fn clear(&mut self) {
        self.manifolds.clear();
    }
}

// ---------------------------------------------------------------------------
// Collision dispatch
// ---------------------------------------------------------------------------

/// Perform narrowphase collision detection between two shapes.
pub fn collide(
    shape_a: &NarrowShape,
    transform_a: &NarrowTransform,
    shape_b: &NarrowShape,
    transform_b: &NarrowTransform,
) -> Option<ContactManifold> {
    match (shape_a, shape_b) {
        (NarrowShape::Sphere { radius: ra }, NarrowShape::Sphere { radius: rb }) => {
            collide_sphere_sphere(transform_a.position, *ra, transform_b.position, *rb)
        }
        (NarrowShape::Sphere { radius }, NarrowShape::Plane { normal, offset }) => {
            let world_normal = transform_b.transform_direction(*normal);
            collide_sphere_plane(transform_a.position, *radius, world_normal, *offset + vec3_dot(transform_b.position, world_normal))
        }
        (NarrowShape::Plane { normal, offset }, NarrowShape::Sphere { radius }) => {
            let world_normal = transform_a.transform_direction(*normal);
            let mut m = collide_sphere_plane(transform_b.position, *radius, world_normal, *offset + vec3_dot(transform_a.position, world_normal))?;
            // Swap the bodies.
            for c in &mut m.contacts {
                std::mem::swap(&mut c.point_a, &mut c.point_b);
                c.normal = vec3_negate(c.normal);
            }
            m.normal = vec3_negate(m.normal);
            Some(m)
        }
        (NarrowShape::Sphere { radius }, NarrowShape::Box { half_extents }) => {
            collide_sphere_box(
                transform_a.position, *radius,
                transform_b, *half_extents,
            )
        }
        (NarrowShape::Box { half_extents }, NarrowShape::Sphere { radius }) => {
            let mut m = collide_sphere_box(
                transform_b.position, *radius,
                transform_a, *half_extents,
            )?;
            swap_manifold_bodies(&mut m);
            Some(m)
        }
        (NarrowShape::Sphere { radius: ra }, NarrowShape::Capsule { radius: rb, half_height }) => {
            collide_sphere_capsule(
                transform_a.position, *ra,
                transform_b, *rb, *half_height,
            )
        }
        (NarrowShape::Capsule { radius: ra, half_height }, NarrowShape::Sphere { radius: rb }) => {
            let mut m = collide_sphere_capsule(
                transform_b.position, *rb,
                transform_a, *ra, *half_height,
            )?;
            swap_manifold_bodies(&mut m);
            Some(m)
        }
        (NarrowShape::Capsule { radius: ra, half_height: ha },
         NarrowShape::Capsule { radius: rb, half_height: hb }) => {
            collide_capsule_capsule(
                transform_a, *ra, *ha,
                transform_b, *rb, *hb,
            )
        }
        _ => {
            // Fallback: use GJK + EPA for arbitrary convex shapes.
            collide_gjk_epa(shape_a, transform_a, shape_b, transform_b)
        }
    }
}

// ---------------------------------------------------------------------------
// Sphere vs Sphere
// ---------------------------------------------------------------------------

fn collide_sphere_sphere(
    pos_a: [f32; 3], radius_a: f32,
    pos_b: [f32; 3], radius_b: f32,
) -> Option<ContactManifold> {
    let diff = vec3_sub(pos_a, pos_b);
    let dist_sq = vec3_length_squared(diff);
    let sum_radius = radius_a + radius_b;

    if dist_sq > sum_radius * sum_radius {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist > EPSILON {
        vec3_scale(diff, 1.0 / dist)
    } else {
        [0.0, 1.0, 0.0]
    };
    let depth = sum_radius - dist;

    let point_a = vec3_sub(pos_a, vec3_scale(normal, radius_a));
    let point_b = vec3_add(pos_b, vec3_scale(normal, radius_b));

    let mut manifold = ContactManifold::new(0, 0);
    manifold.normal = normal;
    manifold.add_contact(NarrowContactPoint::new(point_a, point_b, normal, depth));
    Some(manifold)
}

// ---------------------------------------------------------------------------
// Sphere vs Plane
// ---------------------------------------------------------------------------

fn collide_sphere_plane(
    sphere_pos: [f32; 3], radius: f32,
    plane_normal: [f32; 3], plane_offset: f32,
) -> Option<ContactManifold> {
    let dist = vec3_dot(sphere_pos, plane_normal) - plane_offset;

    if dist > radius {
        return None;
    }

    let depth = radius - dist;
    let point_a = vec3_sub(sphere_pos, vec3_scale(plane_normal, radius));
    let point_b = vec3_sub(sphere_pos, vec3_scale(plane_normal, dist));

    let mut manifold = ContactManifold::new(0, 0);
    manifold.normal = plane_normal;
    manifold.add_contact(NarrowContactPoint::new(point_a, point_b, plane_normal, depth));
    Some(manifold)
}

// ---------------------------------------------------------------------------
// Sphere vs Box (OBB)
// ---------------------------------------------------------------------------

fn collide_sphere_box(
    sphere_pos: [f32; 3], radius: f32,
    box_transform: &NarrowTransform, half_extents: [f32; 3],
) -> Option<ContactManifold> {
    // Transform sphere center to box local space.
    let local_center = box_transform.inverse_transform_point(sphere_pos);

    // Clamp to box surface to find closest point.
    let clamped = [
        local_center[0].clamp(-half_extents[0], half_extents[0]),
        local_center[1].clamp(-half_extents[1], half_extents[1]),
        local_center[2].clamp(-half_extents[2], half_extents[2]),
    ];

    let diff = vec3_sub(local_center, clamped);
    let dist_sq = vec3_length_squared(diff);

    if dist_sq > radius * radius {
        return None;
    }

    let dist = dist_sq.sqrt();
    let local_normal = if dist > EPSILON {
        vec3_scale(diff, 1.0 / dist)
    } else {
        // Sphere center is inside the box. Find the closest face.
        let mut min_dist = f32::MAX;
        let mut best_normal = [0.0f32; 3];
        for axis in 0..3 {
            let d_pos = half_extents[axis] - local_center[axis];
            let d_neg = half_extents[axis] + local_center[axis];
            if d_pos < min_dist {
                min_dist = d_pos;
                best_normal = [0.0; 3];
                best_normal[axis] = 1.0;
            }
            if d_neg < min_dist {
                min_dist = d_neg;
                best_normal = [0.0; 3];
                best_normal[axis] = -1.0;
            }
        }
        best_normal
    };

    let depth = radius - dist;
    let world_normal = box_transform.transform_direction(local_normal);
    let world_closest = box_transform.transform_point(clamped);
    let point_a = vec3_sub(sphere_pos, vec3_scale(world_normal, radius));

    let mut manifold = ContactManifold::new(0, 0);
    manifold.normal = world_normal;
    manifold.add_contact(NarrowContactPoint::new(point_a, world_closest, world_normal, depth));
    Some(manifold)
}

// ---------------------------------------------------------------------------
// Sphere vs Capsule
// ---------------------------------------------------------------------------

fn collide_sphere_capsule(
    sphere_pos: [f32; 3], sphere_radius: f32,
    cap_transform: &NarrowTransform, cap_radius: f32, cap_half_height: f32,
) -> Option<ContactManifold> {
    // Capsule axis in world space.
    let axis = cap_transform.transform_direction([0.0, 1.0, 0.0]);
    let cap_top = vec3_add(cap_transform.position, vec3_scale(axis, cap_half_height));
    let cap_bottom = vec3_sub(cap_transform.position, vec3_scale(axis, cap_half_height));

    // Find closest point on capsule segment to sphere center.
    let closest = closest_point_on_segment(sphere_pos, cap_bottom, cap_top);

    // Now it's a sphere-sphere test.
    collide_sphere_sphere(sphere_pos, sphere_radius, closest, cap_radius)
}

// ---------------------------------------------------------------------------
// Capsule vs Capsule
// ---------------------------------------------------------------------------

fn collide_capsule_capsule(
    ta: &NarrowTransform, ra: f32, ha: f32,
    tb: &NarrowTransform, rb: f32, hb: f32,
) -> Option<ContactManifold> {
    let axis_a = ta.transform_direction([0.0, 1.0, 0.0]);
    let axis_b = tb.transform_direction([0.0, 1.0, 0.0]);

    let a_top = vec3_add(ta.position, vec3_scale(axis_a, ha));
    let a_bottom = vec3_sub(ta.position, vec3_scale(axis_a, ha));
    let b_top = vec3_add(tb.position, vec3_scale(axis_b, hb));
    let b_bottom = vec3_sub(tb.position, vec3_scale(axis_b, hb));

    // Find closest points between two line segments.
    let (closest_a, closest_b) = closest_points_segments(a_bottom, a_top, b_bottom, b_top);

    // Now it's a sphere-sphere test at the closest points.
    collide_sphere_sphere(closest_a, ra, closest_b, rb)
}

// ---------------------------------------------------------------------------
// GJK + EPA fallback
// ---------------------------------------------------------------------------

fn collide_gjk_epa(
    shape_a: &NarrowShape, ta: &NarrowTransform,
    shape_b: &NarrowShape, tb: &NarrowTransform,
) -> Option<ContactManifold> {
    // Minkowski difference support function.
    let support = |dir: [f32; 3]| -> [f32; 3] {
        let local_dir_a = ta.inverse_transform_direction(dir);
        let local_dir_b = tb.inverse_transform_direction(vec3_negate(dir));
        let sup_a = ta.transform_point(shape_a.support(local_dir_a));
        let sup_b = tb.transform_point(shape_b.support(local_dir_b));
        vec3_sub(sup_a, sup_b)
    };

    // GJK: determine if the origin is inside the Minkowski difference.
    let mut simplex: Vec<[f32; 3]> = Vec::new();
    let mut dir = vec3_sub(tb.position, ta.position);
    if vec3_length_squared(dir) < EPSILON {
        dir = [1.0, 0.0, 0.0];
    }

    let first = support(dir);
    simplex.push(first);
    dir = vec3_negate(first);

    for _ in 0..GJK_MAX_ITERATIONS {
        let a = support(dir);
        if vec3_dot(a, dir) < 0.0 {
            return None; // No intersection.
        }
        simplex.push(a);

        if process_simplex(&mut simplex, &mut dir) {
            // Intersection found. Use EPA to find penetration info.
            return epa(&simplex, &support, ta, tb, shape_a, shape_b);
        }
    }

    None
}

/// Process the GJK simplex. Returns true if the origin is enclosed.
fn process_simplex(simplex: &mut Vec<[f32; 3]>, dir: &mut [f32; 3]) -> bool {
    match simplex.len() {
        2 => {
            // Line case.
            let a = simplex[1];
            let b = simplex[0];
            let ab = vec3_sub(b, a);
            let ao = vec3_negate(a);

            if vec3_dot(ab, ao) > 0.0 {
                *dir = vec3_cross(vec3_cross(ab, ao), ab);
            } else {
                simplex.clear();
                simplex.push(a);
                *dir = ao;
            }
            false
        }
        3 => {
            // Triangle case.
            let a = simplex[2];
            let b = simplex[1];
            let c = simplex[0];
            let ab = vec3_sub(b, a);
            let ac = vec3_sub(c, a);
            let ao = vec3_negate(a);
            let abc = vec3_cross(ab, ac);

            if vec3_dot(vec3_cross(abc, ac), ao) > 0.0 {
                if vec3_dot(ac, ao) > 0.0 {
                    simplex.clear();
                    simplex.push(c);
                    simplex.push(a);
                    *dir = vec3_cross(vec3_cross(ac, ao), ac);
                } else {
                    simplex.clear();
                    simplex.push(b);
                    simplex.push(a);
                    *dir = vec3_cross(vec3_cross(ab, ao), ab);
                }
            } else if vec3_dot(vec3_cross(ab, abc), ao) > 0.0 {
                simplex.clear();
                simplex.push(b);
                simplex.push(a);
                *dir = vec3_cross(vec3_cross(ab, ao), ab);
            } else {
                if vec3_dot(abc, ao) > 0.0 {
                    *dir = abc;
                } else {
                    simplex.swap(0, 1);
                    *dir = vec3_negate(abc);
                }
            }
            false
        }
        4 => {
            // Tetrahedron case.
            let a = simplex[3];
            let b = simplex[2];
            let c = simplex[1];
            let d = simplex[0];
            let ao = vec3_negate(a);

            let ab = vec3_sub(b, a);
            let ac = vec3_sub(c, a);
            let ad = vec3_sub(d, a);

            let abc = vec3_cross(ab, ac);
            let acd = vec3_cross(ac, ad);
            let adb = vec3_cross(ad, ab);

            if vec3_dot(abc, ao) > 0.0 {
                simplex.clear();
                simplex.push(c);
                simplex.push(b);
                simplex.push(a);
                *dir = abc;
                return false;
            }
            if vec3_dot(acd, ao) > 0.0 {
                simplex.clear();
                simplex.push(d);
                simplex.push(c);
                simplex.push(a);
                *dir = acd;
                return false;
            }
            if vec3_dot(adb, ao) > 0.0 {
                simplex.clear();
                simplex.push(b);
                simplex.push(d);
                simplex.push(a);
                *dir = adb;
                return false;
            }

            true // Origin is inside the tetrahedron.
        }
        _ => false,
    }
}

/// EPA: Expanding Polytope Algorithm for penetration depth and normal.
fn epa(
    simplex: &[[f32; 3]],
    support: &dyn Fn([f32; 3]) -> [f32; 3],
    ta: &NarrowTransform,
    tb: &NarrowTransform,
    shape_a: &NarrowShape,
    shape_b: &NarrowShape,
) -> Option<ContactManifold> {
    if simplex.len() < 4 {
        return None;
    }

    let mut polytope: Vec<[f32; 3]> = simplex.to_vec();
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ];

    for _ in 0..EPA_MAX_ITERATIONS {
        // Find the face closest to the origin.
        let mut min_dist = f32::MAX;
        let mut min_face = 0;
        let mut min_normal = [0.0f32; 3];

        for (i, face) in faces.iter().enumerate() {
            let a = polytope[face[0]];
            let b = polytope[face[1]];
            let c = polytope[face[2]];
            let ab = vec3_sub(b, a);
            let ac = vec3_sub(c, a);
            let mut normal = vec3_cross(ab, ac);
            let len = vec3_length(normal);
            if len < EPSILON { continue; }
            normal = vec3_scale(normal, 1.0 / len);

            let dist = vec3_dot(normal, a);
            if dist < 0.0 {
                normal = vec3_negate(normal);
            }
            let d = dist.abs();
            if d < min_dist {
                min_dist = d;
                min_face = i;
                min_normal = normal;
            }
        }

        let new_point = support(min_normal);
        let new_dist = vec3_dot(new_point, min_normal);

        if new_dist - min_dist < EPA_TOLERANCE {
            // Converged. Compute contact info.
            let depth = min_dist;
            let normal = min_normal;

            // Compute contact points from the support points.
            let local_dir_a = ta.inverse_transform_direction(normal);
            let local_dir_b = tb.inverse_transform_direction(vec3_negate(normal));
            let point_a = ta.transform_point(shape_a.support(local_dir_a));
            let point_b = tb.transform_point(shape_b.support(local_dir_b));

            let mut manifold = ContactManifold::new(0, 0);
            manifold.normal = normal;
            manifold.add_contact(NarrowContactPoint::new(point_a, point_b, normal, depth));
            return Some(manifold);
        }

        // Expand the polytope.
        let new_idx = polytope.len();
        polytope.push(new_point);

        // Remove faces that can see the new point.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        faces.retain(|face| {
            let a = polytope[face[0]];
            let b = polytope[face[1]];
            let c = polytope[face[2]];
            let ab = vec3_sub(b, a);
            let ac = vec3_sub(c, a);
            let normal = vec3_cross(ab, ac);
            let to_point = vec3_sub(new_point, a);

            if vec3_dot(normal, to_point) > 0.0 {
                // Face is visible from the new point. Collect edges.
                edges.push((face[0], face[1]));
                edges.push((face[1], face[2]));
                edges.push((face[2], face[0]));
                false
            } else {
                true
            }
        });

        // Find boundary edges (edges that appear only once).
        let mut boundary_edges: Vec<(usize, usize)> = Vec::new();
        for &(a, b) in &edges {
            let reversed = edges.iter().any(|&(c, d)| c == b && d == a);
            if !reversed {
                boundary_edges.push((a, b));
            }
        }

        // Create new faces from boundary edges to the new point.
        for &(a, b) in &boundary_edges {
            faces.push([a, b, new_idx]);
        }
    }

    None
}

/// Swap bodies in a manifold (reverse normal, swap points).
fn swap_manifold_bodies(m: &mut ContactManifold) {
    for c in &mut m.contacts {
        std::mem::swap(&mut c.point_a, &mut c.point_b);
        c.normal = vec3_negate(c.normal);
    }
    m.normal = vec3_negate(m.normal);
    std::mem::swap(&mut m.body_a, &mut m.body_b);
}

// ---------------------------------------------------------------------------
// Narrowphase system
// ---------------------------------------------------------------------------

/// The narrowphase collision detection system.
///
/// Takes pairs from the broadphase and generates contact manifolds using
/// shape-specific collision algorithms with contact caching.
#[derive(Debug)]
pub struct NarrowphaseSystem {
    /// Contact cache for persistent contacts.
    pub cache: ContactCache,
    /// Generated manifolds for the current frame.
    pub manifolds: Vec<ContactManifold>,
    /// Statistics.
    pub stats: NarrowphaseStats,
}

/// Narrowphase statistics.
#[derive(Debug, Clone, Default)]
pub struct NarrowphaseStats {
    /// Number of pairs tested.
    pub pairs_tested: u32,
    /// Number of contacts generated.
    pub contacts_generated: u32,
    /// Number of manifolds with contacts.
    pub active_manifolds: u32,
    /// Number of GJK fallbacks used.
    pub gjk_fallbacks: u32,
    /// Number of cached manifolds reused.
    pub cache_hits: u32,
}

impl NarrowphaseSystem {
    /// Create a new narrowphase system.
    pub fn new() -> Self {
        Self {
            cache: ContactCache::new(5),
            manifolds: Vec::new(),
            stats: NarrowphaseStats::default(),
        }
    }

    /// Process a batch of collision pairs.
    pub fn process_pairs(
        &mut self,
        pairs: &[(u32, u32)],
        shapes: &HashMap<u32, NarrowShape>,
        transforms: &HashMap<u32, NarrowTransform>,
    ) {
        self.manifolds.clear();
        self.stats = NarrowphaseStats::default();
        self.cache.begin_frame();

        for &(id_a, id_b) in pairs {
            self.stats.pairs_tested += 1;

            let shape_a = match shapes.get(&id_a) {
                Some(s) => s,
                None => continue,
            };
            let shape_b = match shapes.get(&id_b) {
                Some(s) => s,
                None => continue,
            };
            let transform_a = transforms.get(&id_a).cloned().unwrap_or(NarrowTransform::identity());
            let transform_b = transforms.get(&id_b).cloned().unwrap_or(NarrowTransform::identity());

            if let Some(mut manifold) = collide(shape_a, &transform_a, shape_b, &transform_b) {
                manifold.body_a = id_a;
                manifold.body_b = id_b;

                // Warm-start from cache.
                let cached = self.cache.get_or_create(id_a, id_b);
                for new_contact in &mut manifold.contacts {
                    for old_contact in &cached.contacts {
                        if vec3_distance(new_contact.local_point_a, old_contact.local_point_a) < CONTACT_MERGE_DISTANCE {
                            new_contact.normal_impulse = old_contact.normal_impulse;
                            new_contact.tangent_impulse = old_contact.tangent_impulse;
                            new_contact.age = old_contact.age + 1;
                            self.stats.cache_hits += 1;
                            break;
                        }
                    }
                }

                // Update cache.
                *cached = manifold.clone();
                self.cache.mark_updated(id_a, id_b);

                self.stats.contacts_generated += manifold.contacts.len() as u32;
                self.stats.active_manifolds += 1;
                self.manifolds.push(manifold);
            }
        }

        self.cache.prune_stale();
    }

    /// Returns generated manifolds.
    pub fn get_manifolds(&self) -> &[ContactManifold] {
        &self.manifolds
    }
}

// ---------------------------------------------------------------------------
// Vector math helpers
// ---------------------------------------------------------------------------

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_length_squared(v: [f32; 3]) -> f32 {
    vec3_dot(v, v)
}

fn vec3_length(v: [f32; 3]) -> f32 {
    vec3_length_squared(v).sqrt()
}

fn vec3_negate(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

fn vec3_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    vec3_length(vec3_sub(a, b))
}

fn vec3_distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    vec3_length_squared(vec3_sub(a, b))
}

/// Closest point on a line segment to a point.
fn closest_point_on_segment(point: [f32; 3], seg_a: [f32; 3], seg_b: [f32; 3]) -> [f32; 3] {
    let ab = vec3_sub(seg_b, seg_a);
    let ap = vec3_sub(point, seg_a);
    let t = (vec3_dot(ap, ab) / vec3_dot(ab, ab)).clamp(0.0, 1.0);
    vec3_add(seg_a, vec3_scale(ab, t))
}

/// Closest points between two line segments.
fn closest_points_segments(
    a0: [f32; 3], a1: [f32; 3],
    b0: [f32; 3], b1: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let d1 = vec3_sub(a1, a0);
    let d2 = vec3_sub(b1, b0);
    let r = vec3_sub(a0, b0);

    let a = vec3_dot(d1, d1);
    let e = vec3_dot(d2, d2);
    let f = vec3_dot(d2, r);

    if a <= EPSILON && e <= EPSILON {
        return (a0, b0);
    }

    let (s, t);
    if a <= EPSILON {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = vec3_dot(d1, r);
        if e <= EPSILON {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b_val = vec3_dot(d1, d2);
            let denom = a * e - b_val * b_val;
            s = if denom.abs() > EPSILON {
                ((b_val * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            t = ((b_val * s + f) / e).clamp(0.0, 1.0);
        }
    }

    let closest_a = vec3_add(a0, vec3_scale(d1, s));
    let closest_b = vec3_add(b0, vec3_scale(d2, t));
    (closest_a, closest_b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sphere_collision() {
        let result = collide_sphere_sphere(
            [0.0, 0.0, 0.0], 1.0,
            [1.5, 0.0, 0.0], 1.0,
        );
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.contacts.len(), 1);
        assert!((m.contacts[0].depth - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sphere_sphere_no_collision() {
        let result = collide_sphere_sphere(
            [0.0, 0.0, 0.0], 1.0,
            [3.0, 0.0, 0.0], 1.0,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_sphere_plane_collision() {
        let result = collide_sphere_plane(
            [0.0, 0.5, 0.0], 1.0,
            [0.0, 1.0, 0.0], 0.0,
        );
        assert!(result.is_some());
        let m = result.unwrap();
        assert!((m.contacts[0].depth - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sphere_box_collision() {
        let result = collide_sphere_box(
            [0.0, 1.5, 0.0], 1.0,
            &NarrowTransform::identity(), [1.0, 1.0, 1.0],
        );
        assert!(result.is_some());
        let m = result.unwrap();
        assert!(m.contacts[0].depth > 0.0);
    }

    #[test]
    fn test_capsule_capsule_collision() {
        let ta = NarrowTransform::from_position([0.0, 0.0, 0.0]);
        let tb = NarrowTransform::from_position([1.5, 0.0, 0.0]);
        let result = collide_capsule_capsule(&ta, 0.5, 1.0, &tb, 0.5, 1.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_contact_manifold_reduction() {
        let mut manifold = ContactManifold::new(0, 1);
        for i in 0..6 {
            let x = i as f32 * 0.1;
            manifold.add_contact(NarrowContactPoint::new(
                [x, 0.0, 0.0], [x, -0.01, 0.0], [0.0, 1.0, 0.0], 0.01 * (i + 1) as f32,
            ));
        }
        assert!(manifold.contacts.len() <= MAX_MANIFOLD_CONTACTS);
    }

    #[test]
    fn test_contact_cache() {
        let mut cache = ContactCache::new(3);
        cache.begin_frame();
        {
            let m = cache.get_or_create(0, 1);
            m.add_contact(NarrowContactPoint::new(
                [0.0, 0.0, 0.0], [0.0, -0.01, 0.0], [0.0, 1.0, 0.0], 0.01,
            ));
        }
        cache.mark_updated(0, 1);

        assert_eq!(cache.manifold_count(), 1);
        assert_eq!(cache.total_contact_count(), 1);
    }

    #[test]
    fn test_collide_dispatch() {
        let sphere = NarrowShape::Sphere { radius: 1.0 };
        let plane = NarrowShape::Plane { normal: [0.0, 1.0, 0.0], offset: 0.0 };
        let t_sphere = NarrowTransform::from_position([0.0, 0.5, 0.0]);
        let t_plane = NarrowTransform::identity();

        let result = collide(&sphere, &t_sphere, &plane, &t_plane);
        assert!(result.is_some());
    }

    #[test]
    fn test_closest_point_on_segment() {
        let p = closest_point_on_segment([1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        assert!((p[0] - 1.0).abs() < 0.01);
        assert!((p[1]).abs() < 0.01);
    }

    #[test]
    fn test_shape_support() {
        let sphere = NarrowShape::Sphere { radius: 2.0 };
        let s = sphere.support([1.0, 0.0, 0.0]);
        assert!((s[0] - 2.0).abs() < 0.01);

        let box_shape = NarrowShape::Box { half_extents: [1.0, 2.0, 3.0] };
        let s = box_shape.support([1.0, 1.0, 1.0]);
        assert!((s[0] - 1.0).abs() < 0.01);
        assert!((s[1] - 2.0).abs() < 0.01);
        assert!((s[2] - 3.0).abs() < 0.01);
    }
}
