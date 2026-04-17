// engine/physics/src/collision_detection_v2.rs
//
// Enhanced collision detection: persistent contact manifold with contact
// reduction, speculative contacts for CCD, one-shot manifold generation,
// warm starting cache, and contact point aging.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self { x: self.y*r.z - self.z*r.y, y: self.z*r.x - self.x*r.z, z: self.x*r.y - self.y*r.x } }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f32 { self.dot(self) }
    pub fn normalize(self) -> Self { let l = self.length(); if l < 1e-12 { Self::ZERO } else { Self { x:self.x/l, y:self.y/l, z:self.z/l } } }
    pub fn scale(self, s: f32) -> Self { Self { x:self.x*s, y:self.y*s, z:self.z*s } }
    pub fn add(self, r: Self) -> Self { Self { x:self.x+r.x, y:self.y+r.y, z:self.z+r.z } }
    pub fn sub(self, r: Self) -> Self { Self { x:self.x-r.x, y:self.y-r.y, z:self.z-r.z } }
    pub fn neg(self) -> Self { Self { x:-self.x, y:-self.y, z:-self.z } }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn distance(self, r: Self) -> f32 { self.sub(r).length() }
    pub fn abs(self) -> Self { Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() } }
    pub fn min_component(self) -> f32 { self.x.min(self.y).min(self.z) }
    pub fn max_component(self) -> f32 { self.x.max(self.y).max(self.z) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    pub fn rotate_vec(&self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        let dot_uv = u.dot(v);
        let dot_uu = u.dot(u);
        let cross_uv = u.cross(v);
        u.scale(2.0 * dot_uv)
            .add(v.scale(s * s - dot_uu))
            .add(cross_uv.scale(2.0 * s))
    }
}

// ---------------------------------------------------------------------------
// Body / Collider IDs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColliderId(pub u32);

/// A pair of colliders that might be in contact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColliderPair {
    pub a: ColliderId,
    pub b: ColliderId,
}

impl ColliderPair {
    pub fn new(a: ColliderId, b: ColliderId) -> Self {
        if a.0 <= b.0 { Self { a, b } } else { Self { a: b, b: a } }
    }
}

// ---------------------------------------------------------------------------
// Collision shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum CollisionShape {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { radius: f32, half_height: f32 },
    Cylinder { radius: f32, half_height: f32 },
    ConvexHull { vertices: Vec<Vec3> },
    TriangleMesh { vertices: Vec<Vec3>, indices: Vec<u32> },
}

impl CollisionShape {
    /// Support function for GJK/EPA: find the point furthest in a given direction.
    pub fn support(&self, direction: Vec3, position: Vec3, rotation: Quat) -> Vec3 {
        let local_dir = rotation_inverse_rotate(&rotation, direction);
        let local_support = match self {
            CollisionShape::Sphere { radius } => {
                local_dir.normalize().scale(*radius)
            }
            CollisionShape::Box { half_extents } => {
                Vec3::new(
                    if local_dir.x >= 0.0 { half_extents.x } else { -half_extents.x },
                    if local_dir.y >= 0.0 { half_extents.y } else { -half_extents.y },
                    if local_dir.z >= 0.0 { half_extents.z } else { -half_extents.z },
                )
            }
            CollisionShape::Capsule { radius, half_height } => {
                let mut support = local_dir.normalize().scale(*radius);
                support.y += if local_dir.y >= 0.0 { *half_height } else { -*half_height };
                support
            }
            CollisionShape::Cylinder { radius, half_height } => {
                let horizontal = Vec3::new(local_dir.x, 0.0, local_dir.z);
                let h_len = horizontal.length();
                let mut support = if h_len > 1e-6 {
                    horizontal.scale(*radius / h_len)
                } else {
                    Vec3::ZERO
                };
                support.y = if local_dir.y >= 0.0 { *half_height } else { -*half_height };
                support
            }
            CollisionShape::ConvexHull { vertices } => {
                let mut best = Vec3::ZERO;
                let mut best_dot = f32::NEG_INFINITY;
                for &v in vertices {
                    let d = v.dot(local_dir);
                    if d > best_dot {
                        best_dot = d;
                        best = v;
                    }
                }
                best
            }
            CollisionShape::TriangleMesh { vertices, .. } => {
                // For GJK, treat mesh as convex hull of all vertices
                let mut best = Vec3::ZERO;
                let mut best_dot = f32::NEG_INFINITY;
                for &v in vertices {
                    let d = v.dot(local_dir);
                    if d > best_dot {
                        best_dot = d;
                        best = v;
                    }
                }
                best
            }
        };

        position.add(rotation.rotate_vec(local_support))
    }

    /// Compute AABB for this shape at the given transform.
    pub fn compute_aabb(&self, position: Vec3, _rotation: Quat) -> AABB {
        let extent = match self {
            CollisionShape::Sphere { radius } => Vec3::new(*radius, *radius, *radius),
            CollisionShape::Box { half_extents } => *half_extents,
            CollisionShape::Capsule { radius, half_height } => Vec3::new(*radius, half_height + radius, *radius),
            CollisionShape::Cylinder { radius, half_height } => Vec3::new(*radius, *half_height, *radius),
            CollisionShape::ConvexHull { vertices } => {
                let mut max_ext = Vec3::ZERO;
                for v in vertices { max_ext = Vec3::new(max_ext.x.max(v.x.abs()), max_ext.y.max(v.y.abs()), max_ext.z.max(v.z.abs())); }
                max_ext
            }
            CollisionShape::TriangleMesh { vertices, .. } => {
                let mut max_ext = Vec3::ZERO;
                for v in vertices { max_ext = Vec3::new(max_ext.x.max(v.x.abs()), max_ext.y.max(v.y.abs()), max_ext.z.max(v.z.abs())); }
                max_ext
            }
        };
        AABB {
            min: position.sub(extent),
            max: position.add(extent),
        }
    }
}

fn rotation_inverse_rotate(q: &Quat, v: Vec3) -> Vec3 {
    let conj = Quat { x: -q.x, y: -q.y, z: -q.z, w: q.w };
    conj.rotate_vec(v)
}

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
        && self.min.y <= other.max.y && self.max.y >= other.min.y
        && self.min.z <= other.max.z && self.max.z >= other.min.z
    }
}

// ---------------------------------------------------------------------------
// Contact point
// ---------------------------------------------------------------------------

/// A single contact point between two colliders.
#[derive(Debug, Clone, Copy)]
pub struct ContactPoint {
    /// Contact position on body A in world space.
    pub position_on_a: Vec3,
    /// Contact position on body B in world space.
    pub position_on_b: Vec3,
    /// Contact normal (from B to A).
    pub normal: Vec3,
    /// Penetration depth (positive = overlapping).
    pub depth: f32,
    /// Contact position on A in local space (for warm starting).
    pub local_a: Vec3,
    /// Contact position on B in local space.
    pub local_b: Vec3,
    /// Accumulated normal impulse (for warm starting).
    pub normal_impulse: f32,
    /// Accumulated tangent impulses (for warm starting).
    pub tangent_impulse: [f32; 2],
    /// Tangent directions.
    pub tangent1: Vec3,
    pub tangent2: Vec3,
    /// Age of this contact in frames.
    pub age: u32,
    /// Feature ID for matching contacts across frames.
    pub feature_id: u64,
}

impl ContactPoint {
    pub fn new(pos_a: Vec3, pos_b: Vec3, normal: Vec3, depth: f32) -> Self {
        // Compute tangent basis
        let (t1, t2) = compute_tangent_basis(normal);
        Self {
            position_on_a: pos_a,
            position_on_b: pos_b,
            normal,
            depth,
            local_a: Vec3::ZERO,
            local_b: Vec3::ZERO,
            normal_impulse: 0.0,
            tangent_impulse: [0.0, 0.0],
            tangent1: t1,
            tangent2: t2,
            age: 0,
            feature_id: 0,
        }
    }
}

fn compute_tangent_basis(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.y.abs() < 0.99 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t1 = normal.cross(up).normalize();
    let t2 = normal.cross(t1).normalize();
    (t1, t2)
}

// ---------------------------------------------------------------------------
// Contact manifold
// ---------------------------------------------------------------------------

/// Maximum contacts per manifold.
pub const MAX_MANIFOLD_CONTACTS: usize = 4;

/// A persistent contact manifold between two colliders.
#[derive(Debug, Clone)]
pub struct ContactManifold {
    pub pair: ColliderPair,
    pub body_a: BodyId,
    pub body_b: BodyId,
    pub contacts: Vec<ContactPoint>,
    pub normal: Vec3,
    pub is_active: bool,
    pub age: u32,
    pub max_depth: f32,
    pub friction: f32,
    pub restitution: f32,
}

impl ContactManifold {
    pub fn new(pair: ColliderPair, body_a: BodyId, body_b: BodyId) -> Self {
        Self {
            pair,
            body_a,
            body_b,
            contacts: Vec::with_capacity(MAX_MANIFOLD_CONTACTS),
            normal: Vec3::ZERO,
            is_active: true,
            age: 0,
            max_depth: 0.0,
            friction: 0.5,
            restitution: 0.3,
        }
    }

    /// Add a new contact point, performing contact reduction if full.
    pub fn add_contact(&mut self, new_contact: ContactPoint) {
        // Try to merge with existing contact (within distance threshold)
        let merge_threshold_sq = 0.01 * 0.01; // 1 cm
        for existing in &mut self.contacts {
            let dist_sq = existing.position_on_a.distance(new_contact.position_on_a);
            if dist_sq * dist_sq < merge_threshold_sq {
                // Update existing contact
                existing.position_on_a = new_contact.position_on_a;
                existing.position_on_b = new_contact.position_on_b;
                existing.normal = new_contact.normal;
                existing.depth = new_contact.depth;
                existing.age = 0;
                return;
            }
        }

        if self.contacts.len() < MAX_MANIFOLD_CONTACTS {
            self.contacts.push(new_contact);
        } else {
            // Contact reduction: replace the contact that contributes least to the manifold area
            let replace_idx = self.find_least_important_contact(&new_contact);
            self.contacts[replace_idx] = new_contact;
        }

        // Update manifold normal
        self.update_normal();
        self.max_depth = self.contacts.iter().map(|c| c.depth).fold(0.0_f32, f32::max);
    }

    /// Find the contact that contributes least to manifold area.
    fn find_least_important_contact(&self, new_contact: &ContactPoint) -> usize {
        let mut min_area = f32::MAX;
        let mut min_idx = 0;

        for skip_idx in 0..self.contacts.len() {
            // Compute area of manifold if we replace this contact with the new one
            let mut points = Vec::new();
            for (i, c) in self.contacts.iter().enumerate() {
                if i != skip_idx {
                    points.push(c.position_on_a);
                }
            }
            points.push(new_contact.position_on_a);

            let area = compute_manifold_area(&points);
            if area < min_area {
                min_area = area;
                min_idx = skip_idx;
            }
        }

        min_idx
    }

    fn update_normal(&mut self) {
        if self.contacts.is_empty() {
            return;
        }
        let mut avg_normal = Vec3::ZERO;
        for c in &self.contacts {
            avg_normal = avg_normal.add(c.normal);
        }
        let len = avg_normal.length();
        if len > 1e-6 {
            self.normal = avg_normal.scale(1.0 / len);
        }
    }

    /// Refresh contacts: remove stale ones, update positions.
    pub fn refresh(
        &mut self,
        pos_a: Vec3,
        rot_a: Quat,
        pos_b: Vec3,
        rot_b: Quat,
        break_distance: f32,
    ) {
        self.contacts.retain_mut(|contact| {
            // Recompute world positions from local positions
            contact.position_on_a = pos_a.add(rot_a.rotate_vec(contact.local_a));
            contact.position_on_b = pos_b.add(rot_b.rotate_vec(contact.local_b));

            // Compute new depth and check validity
            let diff = contact.position_on_a.sub(contact.position_on_b);
            let projected_depth = diff.dot(contact.normal);
            contact.depth = projected_depth;

            // Remove contact if bodies have separated too much
            let lateral = diff.sub(contact.normal.scale(projected_depth));
            let lateral_dist = lateral.length();

            contact.age += 1;

            projected_depth > -break_distance && lateral_dist < break_distance
        });

        self.is_active = !self.contacts.is_empty();
        if self.is_active {
            self.max_depth = self.contacts.iter().map(|c| c.depth).fold(0.0_f32, f32::max);
        }
        self.age += 1;
    }

    /// Warm start: restore impulses from a previous manifold.
    pub fn warm_start_from(&mut self, previous: &ContactManifold) {
        for contact in &mut self.contacts {
            // Find matching contact in previous manifold by feature ID or proximity
            let best_match = previous.contacts.iter()
                .filter(|prev| {
                    let dist = contact.local_a.distance(prev.local_a);
                    dist < 0.05 // 5cm matching threshold
                })
                .min_by(|a, b| {
                    let da = contact.local_a.distance(a.local_a);
                    let db = contact.local_a.distance(b.local_a);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some(prev) = best_match {
                contact.normal_impulse = prev.normal_impulse;
                contact.tangent_impulse = prev.tangent_impulse;
            }
        }
    }
}

fn compute_manifold_area(points: &[Vec3]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    // Cross product of diagonals gives approximate area
    let d1 = points[2].sub(points[0]);
    let d2 = if points.len() > 3 {
        points[3].sub(points[1])
    } else {
        points[1].sub(points[0])
    };
    d1.cross(d2).length() * 0.5
}

// ---------------------------------------------------------------------------
// Speculative contacts
// ---------------------------------------------------------------------------

/// Speculative (predictive) contact for continuous collision detection.
#[derive(Debug, Clone)]
pub struct SpeculativeContact {
    pub point: ContactPoint,
    /// Time of impact (0..1 within the timestep).
    pub toi: f32,
    /// Whether this is a speculative contact (not yet touching).
    pub is_speculative: bool,
    /// Allowed penetration velocity (closing speed threshold).
    pub slop: f32,
}

impl SpeculativeContact {
    pub fn new(point: ContactPoint, toi: f32) -> Self {
        Self {
            point,
            toi,
            is_speculative: toi > 0.0 && point.depth < 0.0,
            slop: 0.01,
        }
    }

    /// Compute the maximum allowed closing velocity for this speculative contact.
    pub fn max_closing_velocity(&self, dt: f32) -> f32 {
        if self.is_speculative {
            // Allow objects to approach but not penetrate
            (-self.point.depth + self.slop) / dt
        } else {
            f32::MAX // regular contact, no limit
        }
    }
}

// ---------------------------------------------------------------------------
// GJK algorithm
// ---------------------------------------------------------------------------

/// GJK simplex point (Minkowski difference point).
#[derive(Debug, Clone, Copy)]
struct SimplexPoint {
    a: Vec3,     // support point on shape A
    b: Vec3,     // support point on shape B
    w: Vec3,     // Minkowski difference a - b
}

/// GJK intersection test result.
#[derive(Debug, Clone)]
pub struct GjkResult {
    pub intersecting: bool,
    pub closest_distance: f32,
    pub closest_on_a: Vec3,
    pub closest_on_b: Vec3,
    pub direction: Vec3,
    pub iterations: u32,
}

/// Run the GJK algorithm to test intersection between two convex shapes.
pub fn gjk_test(
    shape_a: &CollisionShape, pos_a: Vec3, rot_a: Quat,
    shape_b: &CollisionShape, pos_b: Vec3, rot_b: Quat,
) -> GjkResult {
    let mut direction = pos_b.sub(pos_a);
    if direction.length_sq() < 1e-12 {
        direction = Vec3::new(1.0, 0.0, 0.0);
    }

    let mut simplex: Vec<SimplexPoint> = Vec::with_capacity(4);
    let mut iterations = 0u32;
    let max_iterations = 64;

    // Initial support point
    let sup = minkowski_support(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b, direction);
    simplex.push(sup);
    direction = sup.w.neg();

    loop {
        iterations += 1;
        if iterations > max_iterations {
            break;
        }

        let new_point = minkowski_support(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b, direction);

        // Check if new point passed the origin
        if new_point.w.dot(direction) < 0.0 {
            // No intersection
            return GjkResult {
                intersecting: false,
                closest_distance: new_point.w.length(),
                closest_on_a: new_point.a,
                closest_on_b: new_point.b,
                direction,
                iterations,
            };
        }

        simplex.push(new_point);

        // Process simplex
        if process_simplex(&mut simplex, &mut direction) {
            return GjkResult {
                intersecting: true,
                closest_distance: 0.0,
                closest_on_a: Vec3::ZERO,
                closest_on_b: Vec3::ZERO,
                direction,
                iterations,
            };
        }
    }

    GjkResult {
        intersecting: false,
        closest_distance: direction.length(),
        closest_on_a: Vec3::ZERO,
        closest_on_b: Vec3::ZERO,
        direction,
        iterations,
    }
}

fn minkowski_support(
    shape_a: &CollisionShape, pos_a: Vec3, rot_a: Quat,
    shape_b: &CollisionShape, pos_b: Vec3, rot_b: Quat,
    direction: Vec3,
) -> SimplexPoint {
    let a = shape_a.support(direction, pos_a, rot_a);
    let b = shape_b.support(direction.neg(), pos_b, rot_b);
    SimplexPoint { a, b, w: a.sub(b) }
}

fn process_simplex(simplex: &mut Vec<SimplexPoint>, direction: &mut Vec3) -> bool {
    match simplex.len() {
        2 => process_line(simplex, direction),
        3 => process_triangle(simplex, direction),
        4 => process_tetrahedron(simplex, direction),
        _ => false,
    }
}

fn process_line(simplex: &mut Vec<SimplexPoint>, direction: &mut Vec3) -> bool {
    let a = simplex[1].w;
    let b = simplex[0].w;
    let ab = b.sub(a);
    let ao = a.neg();

    if ab.dot(ao) > 0.0 {
        *direction = triple_product(ab, ao, ab);
    } else {
        simplex.clear();
        simplex.push(SimplexPoint { a: simplex[1].a, b: simplex[1].b, w: a });
        *direction = ao;
    }
    false
}

fn process_triangle(simplex: &mut Vec<SimplexPoint>, direction: &mut Vec3) -> bool {
    let a = simplex[2].w;
    let b = simplex[1].w;
    let c = simplex[0].w;
    let ab = b.sub(a);
    let ac = c.sub(a);
    let ao = a.neg();
    let abc = ab.cross(ac);

    if abc.cross(ac).dot(ao) > 0.0 {
        if ac.dot(ao) > 0.0 {
            let sp_a = simplex[2];
            let sp_c = simplex[0];
            simplex.clear();
            simplex.push(sp_c);
            simplex.push(sp_a);
            *direction = triple_product(ac, ao, ac);
        } else {
            let sp_a = simplex[2];
            let sp_b = simplex[1];
            simplex.clear();
            simplex.push(sp_b);
            simplex.push(sp_a);
            return process_line(simplex, direction);
        }
    } else if ab.cross(abc).dot(ao) > 0.0 {
        let sp_a = simplex[2];
        let sp_b = simplex[1];
        simplex.clear();
        simplex.push(sp_b);
        simplex.push(sp_a);
        return process_line(simplex, direction);
    } else {
        if abc.dot(ao) > 0.0 {
            *direction = abc;
        } else {
            let sp_a = simplex[2];
            let sp_b = simplex[1];
            let sp_c = simplex[0];
            simplex.clear();
            simplex.push(sp_b);
            simplex.push(sp_c);
            simplex.push(sp_a);
            *direction = abc.neg();
        }
    }
    false
}

fn process_tetrahedron(simplex: &mut Vec<SimplexPoint>, direction: &mut Vec3) -> bool {
    let a = simplex[3].w;
    let b = simplex[2].w;
    let c = simplex[1].w;
    let d = simplex[0].w;
    let ab = b.sub(a);
    let ac = c.sub(a);
    let ad = d.sub(a);
    let ao = a.neg();

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    if abc.dot(ao) > 0.0 {
        let sp_a = simplex[3];
        let sp_b = simplex[2];
        let sp_c = simplex[1];
        simplex.clear();
        simplex.push(sp_c);
        simplex.push(sp_b);
        simplex.push(sp_a);
        return process_triangle(simplex, direction);
    }

    if acd.dot(ao) > 0.0 {
        let sp_a = simplex[3];
        let sp_c = simplex[1];
        let sp_d = simplex[0];
        simplex.clear();
        simplex.push(sp_d);
        simplex.push(sp_c);
        simplex.push(sp_a);
        return process_triangle(simplex, direction);
    }

    if adb.dot(ao) > 0.0 {
        let sp_a = simplex[3];
        let sp_d = simplex[0];
        let sp_b = simplex[2];
        simplex.clear();
        simplex.push(sp_b);
        simplex.push(sp_d);
        simplex.push(sp_a);
        return process_triangle(simplex, direction);
    }

    // Origin is inside the tetrahedron
    true
}

fn triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    a.cross(b).cross(c)
}

// ---------------------------------------------------------------------------
// EPA algorithm (Expanding Polytope Algorithm)
// ---------------------------------------------------------------------------

/// Result of EPA penetration depth query.
#[derive(Debug, Clone)]
pub struct EpaResult {
    pub normal: Vec3,
    pub depth: f32,
    pub contact_a: Vec3,
    pub contact_b: Vec3,
    pub iterations: u32,
}

/// EPA face on the polytope.
#[derive(Debug, Clone)]
struct EpaFace {
    indices: [usize; 3],
    normal: Vec3,
    distance: f32,
}

/// Run EPA to find penetration depth and contact normal.
pub fn epa_penetration(
    shape_a: &CollisionShape, pos_a: Vec3, rot_a: Quat,
    shape_b: &CollisionShape, pos_b: Vec3, rot_b: Quat,
    initial_simplex: &[Vec3],
) -> Option<EpaResult> {
    if initial_simplex.len() < 4 {
        return None;
    }

    let mut vertices: Vec<SimplexPoint> = initial_simplex.iter().map(|&w| {
        let sup_a = shape_a.support(w, pos_a, rot_a);
        let sup_b = shape_b.support(w.neg(), pos_b, rot_b);
        SimplexPoint { a: sup_a, b: sup_b, w: sup_a.sub(sup_b) }
    }).collect();

    let mut faces: Vec<EpaFace> = vec![
        make_epa_face(&vertices, 0, 1, 2),
        make_epa_face(&vertices, 0, 3, 1),
        make_epa_face(&vertices, 0, 2, 3),
        make_epa_face(&vertices, 1, 3, 2),
    ];

    let max_iterations = 64;
    let tolerance = 1e-4;

    for iter in 0..max_iterations {
        // Find closest face to origin
        let closest_idx = faces.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)?;

        let closest = faces[closest_idx].clone();

        // Get new support point in direction of closest face normal
        let new_point = minkowski_support(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b, closest.normal);
        let new_dist = new_point.w.dot(closest.normal);

        if new_dist - closest.distance < tolerance {
            // Converged
            let bary = barycentric_coords(
                vertices[closest.indices[0]].w,
                vertices[closest.indices[1]].w,
                vertices[closest.indices[2]].w,
                closest.normal.scale(closest.distance),
            );

            let contact_a = vertices[closest.indices[0]].a.scale(bary.x)
                .add(vertices[closest.indices[1]].a.scale(bary.y))
                .add(vertices[closest.indices[2]].a.scale(bary.z));

            let contact_b = vertices[closest.indices[0]].b.scale(bary.x)
                .add(vertices[closest.indices[1]].b.scale(bary.y))
                .add(vertices[closest.indices[2]].b.scale(bary.z));

            return Some(EpaResult {
                normal: closest.normal,
                depth: closest.distance,
                contact_a,
                contact_b,
                iterations: iter as u32,
            });
        }

        // Expand polytope: remove visible faces, add new ones
        let new_idx = vertices.len();
        vertices.push(new_point);

        let mut horizon_edges: Vec<(usize, usize)> = Vec::new();

        // Remove faces visible from new point
        faces.retain(|face| {
            let visible = face.normal.dot(new_point.w.sub(vertices[face.indices[0]].w)) > 0.0;
            if visible {
                // Add edges to horizon
                for e in 0..3 {
                    let e0 = face.indices[e];
                    let e1 = face.indices[(e + 1) % 3];
                    // Check if edge is shared (if so, remove it from horizon)
                    let reversed = horizon_edges.iter().position(|&(a, b)| a == e1 && b == e0);
                    if let Some(idx) = reversed {
                        horizon_edges.remove(idx);
                    } else {
                        horizon_edges.push((e0, e1));
                    }
                }
            }
            !visible
        });

        // Create new faces from horizon edges to new point
        for &(e0, e1) in &horizon_edges {
            let new_face = make_epa_face(&vertices, e0, e1, new_idx);
            faces.push(new_face);
        }

        if faces.is_empty() {
            return None;
        }
    }

    None
}

fn make_epa_face(verts: &[SimplexPoint], a: usize, b: usize, c: usize) -> EpaFace {
    let ab = verts[b].w.sub(verts[a].w);
    let ac = verts[c].w.sub(verts[a].w);
    let normal = ab.cross(ac).normalize();
    let distance = normal.dot(verts[a].w);

    // Ensure normal points away from origin
    if distance < 0.0 {
        EpaFace { indices: [a, c, b], normal: normal.neg(), distance: -distance }
    } else {
        EpaFace { indices: [a, b, c], normal, distance }
    }
}

fn barycentric_coords(a: Vec3, b: Vec3, c: Vec3, p: Vec3) -> Vec3 {
    let v0 = b.sub(a);
    let v1 = c.sub(a);
    let v2 = p.sub(a);

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return Vec3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    Vec3::new(u, v, w)
}

// ---------------------------------------------------------------------------
// Warm starting cache
// ---------------------------------------------------------------------------

/// Cache for warm starting contact manifolds across frames.
pub struct WarmStartCache {
    manifolds: HashMap<ColliderPair, ContactManifold>,
    max_age: u32,
}

impl WarmStartCache {
    pub fn new() -> Self {
        Self {
            manifolds: HashMap::new(),
            max_age: 10,
        }
    }

    /// Store a manifold for warm starting next frame.
    pub fn store(&mut self, manifold: ContactManifold) {
        self.manifolds.insert(manifold.pair, manifold);
    }

    /// Retrieve warm-started impulses for a collider pair.
    pub fn retrieve(&self, pair: &ColliderPair) -> Option<&ContactManifold> {
        self.manifolds.get(pair)
    }

    /// Age out stale manifolds.
    pub fn cleanup(&mut self) {
        let max_age = self.max_age;
        self.manifolds.retain(|_, m| m.age < max_age);
    }

    /// Clear all cached data.
    pub fn clear(&mut self) {
        self.manifolds.clear();
    }

    pub fn len(&self) -> usize {
        self.manifolds.len()
    }
}

// ---------------------------------------------------------------------------
// Sphere-sphere intersection (fast path)
// ---------------------------------------------------------------------------

pub fn intersect_sphere_sphere(
    pos_a: Vec3, radius_a: f32,
    pos_b: Vec3, radius_b: f32,
) -> Option<ContactPoint> {
    let diff = pos_b.sub(pos_a);
    let dist_sq = diff.length_sq();
    let sum_radii = radius_a + radius_b;

    if dist_sq > sum_radii * sum_radii {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 { diff.scale(1.0 / dist) } else { Vec3::new(0.0, 1.0, 0.0) };
    let depth = sum_radii - dist;

    let point_a = pos_a.add(normal.scale(radius_a));
    let point_b = pos_b.sub(normal.scale(radius_b));

    Some(ContactPoint::new(point_a, point_b, normal, depth))
}

// ---------------------------------------------------------------------------
// Sphere-box intersection
// ---------------------------------------------------------------------------

pub fn intersect_sphere_box(
    sphere_pos: Vec3, sphere_radius: f32,
    box_pos: Vec3, box_rot: Quat, box_half: Vec3,
) -> Option<ContactPoint> {
    // Transform sphere center to box local space
    let local_center = rotation_inverse_rotate(&box_rot, sphere_pos.sub(box_pos));

    // Find closest point on box
    let closest = Vec3::new(
        local_center.x.clamp(-box_half.x, box_half.x),
        local_center.y.clamp(-box_half.y, box_half.y),
        local_center.z.clamp(-box_half.z, box_half.z),
    );

    let diff = local_center.sub(closest);
    let dist_sq = diff.length_sq();

    if dist_sq > sphere_radius * sphere_radius {
        return None;
    }

    let dist = dist_sq.sqrt();
    let local_normal = if dist > 1e-6 {
        diff.scale(1.0 / dist)
    } else {
        // Sphere center inside box - find nearest face
        let dx = box_half.x - local_center.x.abs();
        let dy = box_half.y - local_center.y.abs();
        let dz = box_half.z - local_center.z.abs();
        if dx < dy && dx < dz {
            Vec3::new(if local_center.x > 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0)
        } else if dy < dz {
            Vec3::new(0.0, if local_center.y > 0.0 { 1.0 } else { -1.0 }, 0.0)
        } else {
            Vec3::new(0.0, 0.0, if local_center.z > 0.0 { 1.0 } else { -1.0 })
        }
    };

    let world_normal = box_rot.rotate_vec(local_normal);
    let world_closest = box_pos.add(box_rot.rotate_vec(closest));
    let point_on_sphere = sphere_pos.sub(world_normal.scale(sphere_radius));

    let depth = sphere_radius - dist;

    Some(ContactPoint::new(point_on_sphere, world_closest, world_normal, depth))
}

// ---------------------------------------------------------------------------
// Collision detection pipeline
// ---------------------------------------------------------------------------

/// Collider data for the collision detection pipeline.
#[derive(Debug, Clone)]
pub struct Collider {
    pub id: ColliderId,
    pub body: BodyId,
    pub shape: CollisionShape,
    pub position: Vec3,
    pub rotation: Quat,
    pub aabb: AABB,
    pub friction: f32,
    pub restitution: f32,
    pub is_trigger: bool,
    pub layer: u32,
    pub mask: u32,
}

/// The collision detection system.
pub struct CollisionDetectionSystem {
    pub colliders: Vec<Collider>,
    pub manifolds: Vec<ContactManifold>,
    pub warm_cache: WarmStartCache,
    pub broad_phase_pairs: Vec<ColliderPair>,
    pub speculative_margin: f32,
    pub contact_break_distance: f32,
    pub stats: CollisionStats,
}

#[derive(Debug, Clone, Default)]
pub struct CollisionStats {
    pub broad_phase_pairs: u32,
    pub narrow_phase_tests: u32,
    pub active_manifolds: u32,
    pub total_contacts: u32,
    pub warm_started: u32,
    pub gjk_calls: u32,
    pub epa_calls: u32,
}

impl CollisionDetectionSystem {
    pub fn new() -> Self {
        Self {
            colliders: Vec::new(),
            manifolds: Vec::new(),
            warm_cache: WarmStartCache::new(),
            broad_phase_pairs: Vec::new(),
            speculative_margin: 0.05,
            contact_break_distance: 0.02,
            stats: CollisionStats::default(),
        }
    }

    /// Run the full collision detection pipeline.
    pub fn detect_collisions(&mut self) {
        self.stats = CollisionStats::default();

        // Update AABBs
        for collider in &mut self.colliders {
            collider.aabb = collider.shape.compute_aabb(collider.position, collider.rotation);
        }

        // Broad phase: brute-force AABB overlap test
        self.broad_phase_pairs.clear();
        let n = self.colliders.len();
        for i in 0..n {
            for j in (i+1)..n {
                if self.colliders[i].layer & self.colliders[j].mask == 0 { continue; }
                if self.colliders[j].layer & self.colliders[i].mask == 0 { continue; }

                if self.colliders[i].aabb.overlaps(&self.colliders[j].aabb) {
                    self.broad_phase_pairs.push(ColliderPair::new(
                        self.colliders[i].id,
                        self.colliders[j].id,
                    ));
                }
            }
        }
        self.stats.broad_phase_pairs = self.broad_phase_pairs.len() as u32;

        // Narrow phase
        let mut new_manifolds = Vec::new();

        for pair in &self.broad_phase_pairs {
            let col_a = match self.colliders.iter().find(|c| c.id == pair.a) {
                Some(c) => c,
                None => continue,
            };
            let col_b = match self.colliders.iter().find(|c| c.id == pair.b) {
                Some(c) => c,
                None => continue,
            };

            self.stats.narrow_phase_tests += 1;

            // Fast path for sphere-sphere
            if let (CollisionShape::Sphere { radius: ra }, CollisionShape::Sphere { radius: rb }) = (&col_a.shape, &col_b.shape) {
                if let Some(contact) = intersect_sphere_sphere(col_a.position, *ra, col_b.position, *rb) {
                    let mut manifold = ContactManifold::new(*pair, col_a.body, col_b.body);
                    manifold.friction = (col_a.friction + col_b.friction) * 0.5;
                    manifold.restitution = col_a.restitution.max(col_b.restitution);
                    manifold.add_contact(contact);
                    new_manifolds.push(manifold);
                }
                continue;
            }

            // GJK for general shapes
            self.stats.gjk_calls += 1;
            let gjk_result = gjk_test(
                &col_a.shape, col_a.position, col_a.rotation,
                &col_b.shape, col_b.position, col_b.rotation,
            );

            if gjk_result.intersecting {
                let mut manifold = ContactManifold::new(*pair, col_a.body, col_b.body);
                manifold.friction = (col_a.friction + col_b.friction) * 0.5;
                manifold.restitution = col_a.restitution.max(col_b.restitution);

                // Use contact from GJK distance or generate one
                let contact = ContactPoint::new(
                    gjk_result.closest_on_a,
                    gjk_result.closest_on_b,
                    gjk_result.direction.normalize(),
                    gjk_result.closest_distance,
                );
                manifold.add_contact(contact);

                // Warm start from cache
                if let Some(prev) = self.warm_cache.retrieve(pair) {
                    manifold.warm_start_from(prev);
                    self.stats.warm_started += 1;
                }

                new_manifolds.push(manifold);
            }
        }

        // Store current manifolds in warm cache for next frame
        for manifold in &new_manifolds {
            self.warm_cache.store(manifold.clone());
        }
        self.warm_cache.cleanup();

        self.manifolds = new_manifolds;
        self.stats.active_manifolds = self.manifolds.len() as u32;
        self.stats.total_contacts = self.manifolds.iter().map(|m| m.contacts.len() as u32).sum();
    }

    /// Add a collider to the system.
    pub fn add_collider(&mut self, collider: Collider) {
        self.colliders.push(collider);
    }

    /// Remove a collider.
    pub fn remove_collider(&mut self, id: ColliderId) {
        self.colliders.retain(|c| c.id != id);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sphere_intersect() {
        let result = intersect_sphere_sphere(
            Vec3::ZERO, 1.0,
            Vec3::new(1.5, 0.0, 0.0), 1.0,
        );
        assert!(result.is_some());
        let contact = result.unwrap();
        assert!(contact.depth > 0.0);
    }

    #[test]
    fn test_sphere_sphere_no_intersect() {
        let result = intersect_sphere_sphere(
            Vec3::ZERO, 1.0,
            Vec3::new(3.0, 0.0, 0.0), 1.0,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_gjk_sphere_sphere() {
        let result = gjk_test(
            &CollisionShape::Sphere { radius: 1.0 }, Vec3::ZERO, Quat::IDENTITY,
            &CollisionShape::Sphere { radius: 1.0 }, Vec3::new(1.5, 0.0, 0.0), Quat::IDENTITY,
        );
        assert!(result.intersecting);
    }

    #[test]
    fn test_gjk_no_intersection() {
        let result = gjk_test(
            &CollisionShape::Sphere { radius: 1.0 }, Vec3::ZERO, Quat::IDENTITY,
            &CollisionShape::Sphere { radius: 1.0 }, Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY,
        );
        assert!(!result.intersecting);
    }

    #[test]
    fn test_contact_manifold() {
        let pair = ColliderPair::new(ColliderId(0), ColliderId(1));
        let mut manifold = ContactManifold::new(pair, BodyId(0), BodyId(1));

        for i in 0..6 {
            let contact = ContactPoint::new(
                Vec3::new(i as f32 * 0.1, 0.0, 0.0),
                Vec3::new(i as f32 * 0.1, -0.01, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                0.01,
            );
            manifold.add_contact(contact);
        }

        assert!(manifold.contacts.len() <= MAX_MANIFOLD_CONTACTS);
    }

    #[test]
    fn test_warm_start_cache() {
        let mut cache = WarmStartCache::new();
        let pair = ColliderPair::new(ColliderId(0), ColliderId(1));
        let manifold = ContactManifold::new(pair, BodyId(0), BodyId(1));
        cache.store(manifold);
        assert!(cache.retrieve(&pair).is_some());
    }

    #[test]
    fn test_support_function() {
        let sphere = CollisionShape::Sphere { radius: 1.0 };
        let support = sphere.support(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, Quat::IDENTITY);
        assert!((support.x - 1.0).abs() < 0.01);
    }
}
