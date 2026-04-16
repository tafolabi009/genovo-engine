//! Collision detection pipeline: broad phase (spatial hash), narrow phase (SAT/analytical),
//! shape definitions, layer filtering, and contact manifold generation.

use std::collections::HashMap;

use bitflags::bitflags;
use glam::{Mat3, Quat, Vec3};

use crate::interface::{ColliderHandle, PhysicsMaterial, RigidBodyHandle};

// ---------------------------------------------------------------------------
// Collision shapes
// ---------------------------------------------------------------------------

/// Geometric primitives used as collider shapes.
#[derive(Debug, Clone)]
pub enum CollisionShape {
    /// Sphere defined by radius.
    Sphere { radius: f32 },
    /// Oriented box defined by half-extents (rotated by the body's orientation).
    Box { half_extents: Vec3 },
    /// Capsule oriented along the local Y axis.
    Capsule { radius: f32, half_height: f32 },
    /// Convex hull built from a set of points.
    ConvexHull { points: Vec<Vec3> },
    /// Triangle mesh for complex static geometry.
    TriMesh {
        vertices: Vec<Vec3>,
        indices: Vec<[u32; 3]>,
    },
    /// Height field for terrain.
    HeightField {
        columns: u32,
        rows: u32,
        heights: Vec<f32>,
        scale: Vec3,
    },
}

impl CollisionShape {
    /// Compute the axis-aligned bounding box for this shape at a given position and rotation.
    pub fn compute_aabb(&self, position: Vec3, rotation: Quat) -> AABB {
        match self {
            CollisionShape::Sphere { radius } => AABB {
                min: position - Vec3::splat(*radius),
                max: position + Vec3::splat(*radius),
            },
            CollisionShape::Box { half_extents } => {
                let rot_mat = Mat3::from_quat(rotation);
                // For an OBB, the AABB is computed from the absolute values of the rotation
                // matrix columns scaled by half_extents.
                let abs_rot = Mat3::from_cols(
                    rot_mat.x_axis.abs(),
                    rot_mat.y_axis.abs(),
                    rot_mat.z_axis.abs(),
                );
                let extent = abs_rot * *half_extents;
                AABB {
                    min: position - extent,
                    max: position + extent,
                }
            }
            CollisionShape::Capsule {
                radius,
                half_height,
            } => {
                // The capsule is along the local Y axis.
                let local_top = Vec3::new(0.0, *half_height, 0.0);
                let local_bot = Vec3::new(0.0, -*half_height, 0.0);
                let top = position + rotation * local_top;
                let bot = position + rotation * local_bot;
                let min = top.min(bot) - Vec3::splat(*radius);
                let max = top.max(bot) + Vec3::splat(*radius);
                AABB { min, max }
            }
            CollisionShape::ConvexHull { points } => {
                let mut aabb = AABB {
                    min: Vec3::splat(f32::INFINITY),
                    max: Vec3::splat(f32::NEG_INFINITY),
                };
                for p in points {
                    let world_p = position + rotation * *p;
                    aabb.min = aabb.min.min(world_p);
                    aabb.max = aabb.max.max(world_p);
                }
                aabb
            }
            CollisionShape::TriMesh { vertices, .. } => {
                let mut aabb = AABB {
                    min: Vec3::splat(f32::INFINITY),
                    max: Vec3::splat(f32::NEG_INFINITY),
                };
                for v in vertices {
                    let world_v = position + rotation * *v;
                    aabb.min = aabb.min.min(world_v);
                    aabb.max = aabb.max.max(world_v);
                }
                aabb
            }
            CollisionShape::HeightField {
                columns,
                rows,
                scale,
                heights,
            } => {
                let mut min_h = f32::INFINITY;
                let mut max_h = f32::NEG_INFINITY;
                for &h in heights {
                    min_h = min_h.min(h);
                    max_h = max_h.max(h);
                }
                let x_extent = (*columns as f32 - 1.0) * scale.x;
                let z_extent = (*rows as f32 - 1.0) * scale.z;
                AABB {
                    min: position + Vec3::new(0.0, min_h * scale.y, 0.0),
                    max: position + Vec3::new(x_extent, max_h * scale.y, z_extent),
                }
            }
        }
    }

    /// Compute the volume of this shape.
    pub fn volume(&self) -> f32 {
        match self {
            CollisionShape::Sphere { radius } => {
                (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius
            }
            CollisionShape::Box { half_extents } => {
                8.0 * half_extents.x * half_extents.y * half_extents.z
            }
            CollisionShape::Capsule {
                radius,
                half_height,
            } => {
                let sphere_vol =
                    (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius;
                let cylinder_vol =
                    std::f32::consts::PI * radius * radius * (2.0 * half_height);
                sphere_vol + cylinder_vol
            }
            _ => 1.0, // Fallback for complex shapes
        }
    }

    /// Compute the local inertia tensor for this shape given its mass.
    pub fn compute_inertia_tensor(&self, mass: f32) -> Mat3 {
        match self {
            CollisionShape::Sphere { radius } => {
                // I = 2/5 * m * r^2
                let i = (2.0 / 5.0) * mass * radius * radius;
                Mat3::from_diagonal(Vec3::splat(i))
            }
            CollisionShape::Box { half_extents } => {
                // I = m/12 * diag(h^2+d^2, w^2+d^2, w^2+h^2) where w,h,d are full extents
                let w = 2.0 * half_extents.x;
                let h = 2.0 * half_extents.y;
                let d = 2.0 * half_extents.z;
                let factor = mass / 12.0;
                Mat3::from_diagonal(Vec3::new(
                    factor * (h * h + d * d),
                    factor * (w * w + d * d),
                    factor * (w * w + h * h),
                ))
            }
            CollisionShape::Capsule {
                radius,
                half_height,
            } => {
                // Approximate as cylinder + two hemispheres
                let r2 = radius * radius;
                let h = 2.0 * half_height;
                let cylinder_mass = mass * (h / (h + (4.0 / 3.0) * radius));
                let hemisphere_mass = (mass - cylinder_mass) * 0.5;

                // Cylinder: Ixx = Izz = m*(3r^2 + h^2)/12, Iyy = m*r^2/2
                let cyl_iy = cylinder_mass * r2 * 0.5;
                let cyl_ixz = cylinder_mass * (3.0 * r2 + h * h) / 12.0;

                // Each hemisphere offset by half_height from center, using parallel axis theorem
                let hem_iy = 2.0 * hemisphere_mass * (2.0 / 5.0) * r2;
                let hem_ixz_local = hemisphere_mass * (2.0 / 5.0) * r2;
                let hem_ixz =
                    2.0 * (hem_ixz_local + hemisphere_mass * half_height * half_height);

                Mat3::from_diagonal(Vec3::new(
                    cyl_ixz + hem_ixz,
                    cyl_iy + hem_iy,
                    cyl_ixz + hem_ixz,
                ))
            }
            _ => {
                // Fallback: use sphere approximation
                let i = (2.0 / 5.0) * mass;
                Mat3::from_diagonal(Vec3::splat(i))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Collider descriptor
// ---------------------------------------------------------------------------

/// Descriptor used to create a collider and attach it to a rigid body.
#[derive(Debug, Clone)]
pub struct ColliderDesc {
    /// The geometric shape of this collider.
    pub shape: CollisionShape,
    /// Local-space offset from the parent rigid body origin.
    pub offset: Vec3,
    /// Material properties (friction, restitution, density).
    pub material: PhysicsMaterial,
    /// Whether this collider is a sensor (trigger) that does not generate contact forces.
    pub is_sensor: bool,
    /// Collision layer this collider belongs to.
    pub collision_layer: CollisionLayer,
    /// Mask defining which layers this collider can collide with.
    pub collision_mask: CollisionMask,
}

impl Default for ColliderDesc {
    fn default() -> Self {
        Self {
            shape: CollisionShape::Sphere { radius: 0.5 },
            offset: Vec3::ZERO,
            material: PhysicsMaterial::default(),
            is_sensor: false,
            collision_layer: CollisionLayer::DEFAULT,
            collision_mask: CollisionMask::ALL,
        }
    }
}

// ---------------------------------------------------------------------------
// Collision layers and masks (bitflags)
// ---------------------------------------------------------------------------

bitflags! {
    /// Collision layer assigned to a collider. A collider belongs to one or more layers.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CollisionLayer: u32 {
        const DEFAULT    = 1 << 0;
        const STATIC     = 1 << 1;
        const DYNAMIC    = 1 << 2;
        const KINEMATIC  = 1 << 3;
        const PLAYER     = 1 << 4;
        const ENEMY      = 1 << 5;
        const PROJECTILE = 1 << 6;
        const TRIGGER    = 1 << 7;
        const TERRAIN    = 1 << 8;
        const DEBRIS     = 1 << 9;
        const UI         = 1 << 10;
    }
}

bitflags! {
    /// Mask specifying which [`CollisionLayer`]s a collider can interact with.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CollisionMask: u32 {
        const NONE = 0;
        const ALL  = u32::MAX;
    }
}

/// Returns true if two colliders can interact based on their layer/mask settings.
#[inline]
pub fn layers_interact(
    layer_a: CollisionLayer,
    mask_a: CollisionMask,
    layer_b: CollisionLayer,
    mask_b: CollisionMask,
) -> bool {
    let a_sees_b = (mask_a.bits() & layer_b.bits()) != 0;
    let b_sees_a = (mask_b.bits() & layer_a.bits()) != 0;
    a_sees_b && b_sees_a
}

// ---------------------------------------------------------------------------
// Contact and collision events
// ---------------------------------------------------------------------------

/// A single contact point generated by the narrow phase.
#[derive(Debug, Clone)]
pub struct ContactPoint {
    /// World-space position of the contact.
    pub position: Vec3,
    /// Contact normal pointing from body A to body B.
    pub normal: Vec3,
    /// Penetration depth (positive means overlapping).
    pub penetration_depth: f32,
    /// Impulse applied to resolve the contact (filled by solver).
    pub impulse: f32,
    /// Tangent impulse (friction), filled by solver.
    pub tangent_impulse: Vec3,
}

impl ContactPoint {
    pub fn new(position: Vec3, normal: Vec3, penetration_depth: f32) -> Self {
        Self {
            position,
            normal,
            penetration_depth,
            impulse: 0.0,
            tangent_impulse: Vec3::ZERO,
        }
    }
}

/// A contact manifold between two colliders.
#[derive(Debug, Clone)]
pub struct ContactManifold {
    pub collider_a: ColliderHandle,
    pub collider_b: ColliderHandle,
    pub body_a: RigidBodyHandle,
    pub body_b: RigidBodyHandle,
    pub contacts: Vec<ContactPoint>,
    /// Combined friction coefficient of the pair.
    pub friction: f32,
    /// Combined restitution coefficient of the pair.
    pub restitution: f32,
}

/// Events generated by the collision pipeline each simulation step.
#[derive(Debug, Clone)]
pub enum CollisionEvent {
    /// Two colliders began overlapping.
    Begin {
        collider_a: ColliderHandle,
        collider_b: ColliderHandle,
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
    },
    /// Two colliders stopped overlapping.
    End {
        collider_a: ColliderHandle,
        collider_b: ColliderHandle,
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
    },
    /// Two colliders remain in contact.
    Stay {
        collider_a: ColliderHandle,
        collider_b: ColliderHandle,
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        contacts: Vec<ContactPoint>,
    },
}

// ---------------------------------------------------------------------------
// AABB (local to collision module)
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box used for broad phase culling.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Expand AABB by a margin on all sides.
    pub fn expanded(&self, margin: f32) -> AABB {
        AABB {
            min: self.min - Vec3::splat(margin),
            max: self.max + Vec3::splat(margin),
        }
    }

    /// Test ray intersection. Returns Some(t) at the nearest hit distance, or None.
    pub fn ray_intersect(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<f32> {
        let inv_dir = Vec3::ONE / dir;
        let t1 = (self.min - origin) * inv_dir;
        let t2 = (self.max - origin) * inv_dir;
        let t_min = t1.min(t2);
        let t_max = t1.max(t2);
        let t_enter = t_min.x.max(t_min.y).max(t_min.z);
        let t_exit = t_max.x.min(t_max.y).min(t_max.z);
        if t_enter <= t_exit && t_exit >= 0.0 && t_enter <= max_dist {
            Some(t_enter.max(0.0))
        } else {
            None
        }
    }
}

/// A candidate collision pair emitted by the broad phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BroadPhasePair {
    pub collider_a: ColliderHandle,
    pub collider_b: ColliderHandle,
}

// ===========================================================================
// Broad Phase -- Spatial Hash Grid
// ===========================================================================

/// Spatial hash grid for O(1) broad-phase pair lookups.
///
/// The world is divided into uniform cells of `cell_size`. Each collider registers
/// in every cell its AABB overlaps. Pairs sharing at least one cell are reported.
pub struct SpatialHashGrid {
    /// The size of each cell. Exposed for debug/diagnostics.
    pub cell_size: f32,
    inv_cell_size: f32,
    /// Maps cell key -> list of collider handles in that cell.
    cells: HashMap<(i32, i32, i32), Vec<ColliderHandle>>,
    /// Per-collider AABB storage for updates.
    aabbs: HashMap<ColliderHandle, AABB>,
}

impl SpatialHashGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
            aabbs: HashMap::new(),
        }
    }

    fn cell_coord(&self, v: f32) -> i32 {
        (v * self.inv_cell_size).floor() as i32
    }

    fn cell_range(&self, aabb: &AABB) -> ((i32, i32, i32), (i32, i32, i32)) {
        let min = (
            self.cell_coord(aabb.min.x),
            self.cell_coord(aabb.min.y),
            self.cell_coord(aabb.min.z),
        );
        let max = (
            self.cell_coord(aabb.max.x),
            self.cell_coord(aabb.max.y),
            self.cell_coord(aabb.max.z),
        );
        (min, max)
    }

    /// Insert or update a collider's bounding volume.
    pub fn update(&mut self, handle: ColliderHandle, aabb: AABB) {
        // Remove old entries if present
        self.remove(handle);
        // Insert into all overlapping cells
        let (min, max) = self.cell_range(&aabb);
        for x in min.0..=max.0 {
            for y in min.1..=max.1 {
                for z in min.2..=max.2 {
                    self.cells.entry((x, y, z)).or_default().push(handle);
                }
            }
        }
        self.aabbs.insert(handle, aabb);
    }

    /// Remove a collider from the broad phase.
    pub fn remove(&mut self, handle: ColliderHandle) {
        if let Some(aabb) = self.aabbs.remove(&handle) {
            let (min, max) = self.cell_range(&aabb);
            for x in min.0..=max.0 {
                for y in min.1..=max.1 {
                    for z in min.2..=max.2 {
                        if let Some(cell) = self.cells.get_mut(&(x, y, z)) {
                            cell.retain(|h| *h != handle);
                            // Don't remove empty cells here for perf; they'll be reused.
                        }
                    }
                }
            }
        }
    }

    /// Query all potentially overlapping pairs. De-duplicated.
    pub fn query_pairs(&self) -> Vec<BroadPhasePair> {
        let mut seen = std::collections::HashSet::new();
        let mut pairs = Vec::new();

        for cell in self.cells.values() {
            let len = cell.len();
            for i in 0..len {
                for j in (i + 1)..len {
                    let a = cell[i];
                    let b = cell[j];
                    // Canonical ordering for dedup
                    let (lo, hi) = if a.0 < b.0 { (a, b) } else { (b, a) };
                    if seen.insert((lo, hi)) {
                        // Fine AABB check before reporting
                        if let (Some(aabb_a), Some(aabb_b)) =
                            (self.aabbs.get(&lo), self.aabbs.get(&hi))
                        {
                            if aabb_a.intersects(aabb_b) {
                                pairs.push(BroadPhasePair {
                                    collider_a: lo,
                                    collider_b: hi,
                                });
                            }
                        }
                    }
                }
            }
        }
        pairs
    }

    /// Query all colliders whose AABB overlaps the given AABB.
    pub fn query_aabb(&self, aabb: &AABB) -> Vec<ColliderHandle> {
        let mut result = std::collections::HashSet::new();
        let (min, max) = self.cell_range(aabb);
        for x in min.0..=max.0 {
            for y in min.1..=max.1 {
                for z in min.2..=max.2 {
                    if let Some(cell) = self.cells.get(&(x, y, z)) {
                        for &handle in cell {
                            if let Some(h_aabb) = self.aabbs.get(&handle) {
                                if h_aabb.intersects(aabb) {
                                    result.insert(handle);
                                }
                            }
                        }
                    }
                }
            }
        }
        result.into_iter().collect()
    }

    /// Clear all cells (called at the start of each broad phase rebuild).
    pub fn clear(&mut self) {
        self.cells.clear();
        self.aabbs.clear();
    }

    /// Get the AABB for a collider handle, if present.
    pub fn get_aabb(&self, handle: ColliderHandle) -> Option<&AABB> {
        self.aabbs.get(&handle)
    }
}

// ===========================================================================
// Narrow Phase -- Intersection Tests
// ===========================================================================

/// The narrow phase performs exact intersection tests and generates contact manifolds.
pub struct NarrowPhase;

impl NarrowPhase {
    /// Test two shapes for intersection and produce contact points.
    /// `pos_a/rot_a` and `pos_b/rot_b` are the world transforms of each shape.
    pub fn test_pair(
        shape_a: &CollisionShape,
        pos_a: Vec3,
        rot_a: Quat,
        shape_b: &CollisionShape,
        pos_b: Vec3,
        rot_b: Quat,
    ) -> Option<Vec<ContactPoint>> {
        match (shape_a, shape_b) {
            (CollisionShape::Sphere { radius: ra }, CollisionShape::Sphere { radius: rb }) => {
                sphere_vs_sphere(pos_a, *ra, pos_b, *rb)
            }
            (
                CollisionShape::Sphere { radius },
                CollisionShape::Box { half_extents },
            ) => sphere_vs_box(pos_a, *radius, pos_b, rot_b, *half_extents),
            (
                CollisionShape::Box { half_extents },
                CollisionShape::Sphere { radius },
            ) => {
                // Flip the result: swap A and B, negate normals
                sphere_vs_box(pos_b, *radius, pos_a, rot_a, *half_extents).map(|contacts| {
                    contacts
                        .into_iter()
                        .map(|mut c| {
                            c.normal = -c.normal;
                            c
                        })
                        .collect()
                })
            }
            (
                CollisionShape::Box {
                    half_extents: he_a,
                },
                CollisionShape::Box {
                    half_extents: he_b,
                },
            ) => box_vs_box(pos_a, rot_a, *he_a, pos_b, rot_b, *he_b),
            (
                CollisionShape::Capsule {
                    radius: ra,
                    half_height: hh_a,
                },
                CollisionShape::Capsule {
                    radius: rb,
                    half_height: hh_b,
                },
            ) => capsule_vs_capsule(pos_a, rot_a, *ra, *hh_a, pos_b, rot_b, *rb, *hh_b),
            (
                CollisionShape::Sphere { radius },
                CollisionShape::Capsule {
                    radius: cap_r,
                    half_height,
                },
            ) => sphere_vs_capsule(pos_a, *radius, pos_b, rot_b, *cap_r, *half_height),
            (
                CollisionShape::Capsule {
                    radius: cap_r,
                    half_height,
                },
                CollisionShape::Sphere { radius },
            ) => {
                sphere_vs_capsule(pos_b, *radius, pos_a, rot_a, *cap_r, *half_height).map(
                    |contacts| {
                        contacts
                            .into_iter()
                            .map(|mut c| {
                                c.normal = -c.normal;
                                c
                            })
                            .collect()
                    },
                )
            }
            _ => None, // Unsupported pair; complex shapes need GJK (future work)
        }
    }
}

// ---------------------------------------------------------------------------
// Sphere vs Sphere
// ---------------------------------------------------------------------------

fn sphere_vs_sphere(
    pos_a: Vec3,
    radius_a: f32,
    pos_b: Vec3,
    radius_b: f32,
) -> Option<Vec<ContactPoint>> {
    let diff = pos_b - pos_a;
    let dist_sq = diff.length_squared();
    let sum_radii = radius_a + radius_b;

    if dist_sq >= sum_radii * sum_radii {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 {
        diff / dist
    } else {
        Vec3::Y // Degenerate: coincident centers, pick arbitrary normal
    };

    let penetration = sum_radii - dist;
    let contact_pos = pos_a + normal * (radius_a - penetration * 0.5);

    Some(vec![ContactPoint::new(contact_pos, normal, penetration)])
}

// ---------------------------------------------------------------------------
// Sphere vs Box (OBB)
// ---------------------------------------------------------------------------

fn sphere_vs_box(
    sphere_pos: Vec3,
    sphere_radius: f32,
    box_pos: Vec3,
    box_rot: Quat,
    box_half_extents: Vec3,
) -> Option<Vec<ContactPoint>> {
    // Transform sphere center into box local space
    let inv_rot = box_rot.inverse();
    let local_center = inv_rot * (sphere_pos - box_pos);

    // Find closest point on the box to the sphere center (clamping)
    let closest_local = Vec3::new(
        local_center.x.clamp(-box_half_extents.x, box_half_extents.x),
        local_center.y.clamp(-box_half_extents.y, box_half_extents.y),
        local_center.z.clamp(-box_half_extents.z, box_half_extents.z),
    );

    let diff = local_center - closest_local;
    let dist_sq = diff.length_squared();

    if dist_sq > sphere_radius * sphere_radius && dist_sq > 1e-12 {
        return None;
    }

    // Sphere center is inside the box or touching
    if dist_sq < 1e-12 {
        // Sphere center is inside the box: push out along the axis of least penetration
        let mut min_pen = f32::INFINITY;
        let mut normal_local = Vec3::X;

        let axes = [Vec3::X, Vec3::Y, Vec3::Z];
        for axis in &axes {
            let pen_pos = box_half_extents.dot(*axis) - local_center.dot(*axis);
            let pen_neg = box_half_extents.dot(*axis) + local_center.dot(*axis);
            if pen_pos < min_pen {
                min_pen = pen_pos;
                normal_local = *axis;
            }
            if pen_neg < min_pen {
                min_pen = pen_neg;
                normal_local = -*axis;
            }
        }

        let penetration = min_pen + sphere_radius;
        let normal_world = box_rot * normal_local;
        let contact_pos = sphere_pos - normal_world * (sphere_radius - penetration * 0.5);

        return Some(vec![ContactPoint::new(contact_pos, -normal_world, penetration)]);
    }

    let dist = dist_sq.sqrt();
    let normal_local = diff / dist;
    let penetration = sphere_radius - dist;

    let normal_world = box_rot * normal_local;
    let closest_world = box_pos + box_rot * closest_local;

    Some(vec![ContactPoint::new(closest_world, -normal_world, penetration)])
}

// ---------------------------------------------------------------------------
// Box vs Box -- Separating Axis Theorem (SAT) on 15 axes
// ---------------------------------------------------------------------------

fn box_vs_box(
    pos_a: Vec3,
    rot_a: Quat,
    he_a: Vec3,
    pos_b: Vec3,
    rot_b: Quat,
    he_b: Vec3,
) -> Option<Vec<ContactPoint>> {
    let rot_mat_a = Mat3::from_quat(rot_a);
    let rot_mat_b = Mat3::from_quat(rot_b);

    let axes_a = [rot_mat_a.x_axis, rot_mat_a.y_axis, rot_mat_a.z_axis];
    let axes_b = [rot_mat_b.x_axis, rot_mat_b.y_axis, rot_mat_b.z_axis];

    let d = pos_b - pos_a;

    // Precompute the rotation matrix expressing B in A's frame and its absolute value
    // c[i][j] = axes_a[i] . axes_b[j]
    let mut c = [[0.0f32; 3]; 3];
    let mut abs_c = [[0.0f32; 3]; 3];
    let epsilon = 1e-6;

    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = axes_a[i].dot(axes_b[j]);
            abs_c[i][j] = c[i][j].abs() + epsilon;
        }
    }

    let he_a_arr = [he_a.x, he_a.y, he_a.z];
    let he_b_arr = [he_b.x, he_b.y, he_b.z];

    let mut min_penetration = f32::INFINITY;
    let mut best_axis = Vec3::ZERO;
    let mut _best_axis_index = 0usize;

    // Helper: project half-extents onto axis and test separation
    macro_rules! test_axis {
        ($axis:expr, $ra:expr, $rb:expr, $idx:expr) => {{
            let axis = $axis;
            let axis_len_sq = axis.length_squared();
            if axis_len_sq > 1e-10 {
                let axis_norm = axis / axis_len_sq.sqrt();
                let separation = d.dot(axis_norm).abs();
                let ra_proj: f32 = $ra;
                let rb_proj: f32 = $rb;
                let pen = ra_proj + rb_proj - separation;
                if pen < 0.0 {
                    return None; // Separating axis found
                }
                if pen < min_penetration {
                    min_penetration = pen;
                    best_axis = if d.dot(axis_norm) < 0.0 {
                        -axis_norm
                    } else {
                        axis_norm
                    };
                    _best_axis_index = $idx;
                }
            }
        }};
    }

    // 3 face axes of A
    for i in 0..3 {
        let ra = he_a_arr[i];
        let rb = he_b_arr[0] * abs_c[i][0]
            + he_b_arr[1] * abs_c[i][1]
            + he_b_arr[2] * abs_c[i][2];
        test_axis!(axes_a[i], ra, rb, i);
    }

    // 3 face axes of B
    for i in 0..3 {
        let ra = he_a_arr[0] * abs_c[0][i]
            + he_a_arr[1] * abs_c[1][i]
            + he_a_arr[2] * abs_c[2][i];
        let rb = he_b_arr[i];
        test_axis!(axes_b[i], ra, rb, 3 + i);
    }

    // 9 edge-edge axes: axes_a[i] x axes_b[j]
    for i in 0..3 {
        for j in 0..3 {
            let axis = axes_a[i].cross(axes_b[j]);
            let i1 = (i + 1) % 3;
            let i2 = (i + 2) % 3;
            let j1 = (j + 1) % 3;
            let j2 = (j + 2) % 3;

            let ra = he_a_arr[i1] * abs_c[i2][j] + he_a_arr[i2] * abs_c[i1][j];
            let rb = he_b_arr[j1] * abs_c[i][j2] + he_b_arr[j2] * abs_c[i][j1];
            test_axis!(axis, ra, rb, 6 + i * 3 + j);
        }
    }

    // All 15 axes tested, no separating axis => collision
    // Generate contact point at the midpoint adjusted by penetration
    let contact_pos = pos_a + d * 0.5 + best_axis * (min_penetration * 0.5);
    let normal = best_axis;

    Some(vec![ContactPoint::new(contact_pos, normal, min_penetration)])
}

// ---------------------------------------------------------------------------
// Capsule vs Capsule
// ---------------------------------------------------------------------------

/// Closest points between two line segments P0P1 and Q0Q1.
/// Returns (s, t) parameters along each segment in [0,1] and the squared distance.
fn closest_points_segments(
    p0: Vec3,
    p1: Vec3,
    q0: Vec3,
    q1: Vec3,
) -> (f32, f32, f32) {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let epsilon = 1e-7;

    let (mut s, t);

    if a <= epsilon && e <= epsilon {
        // Both segments degenerate to points
        s = 0.0;
        t = 0.0;
    } else if a <= epsilon {
        // First segment degenerates
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= epsilon {
            // Second segment degenerates
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // General case
            let b = d1.dot(d2);
            let denom = a * e - b * b;

            if denom.abs() > epsilon {
                s = ((b * f - c * e) / denom).clamp(0.0, 1.0);
            } else {
                s = 0.0;
            }

            let t_nom = b * s + f;
            if t_nom < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0); // Reclamp
            } else if t_nom > e {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0); // Reclamp
            } else {
                t = t_nom / e;
            }
        }
    }

    let closest_p = p0 + d1 * s;
    let closest_q = q0 + d2 * t;
    let dist_sq = (closest_p - closest_q).length_squared();

    (s, t, dist_sq)
}

fn capsule_vs_capsule(
    pos_a: Vec3,
    rot_a: Quat,
    radius_a: f32,
    half_height_a: f32,
    pos_b: Vec3,
    rot_b: Quat,
    radius_b: f32,
    half_height_b: f32,
) -> Option<Vec<ContactPoint>> {
    // Capsule A segment endpoints
    let up_a = rot_a * Vec3::Y;
    let p0 = pos_a - up_a * half_height_a;
    let p1 = pos_a + up_a * half_height_a;

    // Capsule B segment endpoints
    let up_b = rot_b * Vec3::Y;
    let q0 = pos_b - up_b * half_height_b;
    let q1 = pos_b + up_b * half_height_b;

    let (s, t, dist_sq) = closest_points_segments(p0, p1, q0, q1);
    let sum_radii = radius_a + radius_b;

    if dist_sq >= sum_radii * sum_radii {
        return None;
    }

    let closest_p = p0 + (p1 - p0) * s;
    let closest_q = q0 + (q1 - q0) * t;
    let diff = closest_q - closest_p;
    let dist = dist_sq.sqrt();

    let normal = if dist > 1e-6 {
        diff / dist
    } else {
        Vec3::Y
    };

    let penetration = sum_radii - dist;
    let contact_pos = closest_p + normal * (radius_a - penetration * 0.5);

    Some(vec![ContactPoint::new(contact_pos, normal, penetration)])
}

// ---------------------------------------------------------------------------
// Sphere vs Capsule
// ---------------------------------------------------------------------------

fn sphere_vs_capsule(
    sphere_pos: Vec3,
    sphere_radius: f32,
    cap_pos: Vec3,
    cap_rot: Quat,
    cap_radius: f32,
    cap_half_height: f32,
) -> Option<Vec<ContactPoint>> {
    let up = cap_rot * Vec3::Y;
    let seg_start = cap_pos - up * cap_half_height;
    let seg_end = cap_pos + up * cap_half_height;

    // Find closest point on the capsule segment to the sphere center
    let seg = seg_end - seg_start;
    let t = if seg.length_squared() > 1e-12 {
        ((sphere_pos - seg_start).dot(seg) / seg.length_squared()).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let closest = seg_start + seg * t;

    let diff = sphere_pos - closest;
    let dist_sq = diff.length_squared();
    let sum_radii = sphere_radius + cap_radius;

    if dist_sq >= sum_radii * sum_radii {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 {
        diff / dist
    } else {
        Vec3::Y
    };

    let penetration = sum_radii - dist;
    let contact_pos = closest + normal * (cap_radius - penetration * 0.5);

    // Normal points from capsule to sphere (A=sphere, B=capsule conceptually flipped)
    Some(vec![ContactPoint::new(contact_pos, normal, penetration)])
}

// ---------------------------------------------------------------------------
// Ray vs Shape
// ---------------------------------------------------------------------------

/// Cast a ray against a collision shape at a given world transform.
/// Returns (distance, world-space hit point, world-space normal).
pub fn ray_vs_shape(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    shape: &CollisionShape,
    shape_pos: Vec3,
    shape_rot: Quat,
) -> Option<(f32, Vec3, Vec3)> {
    match shape {
        CollisionShape::Sphere { radius } => {
            ray_vs_sphere(origin, dir, max_dist, shape_pos, *radius)
        }
        CollisionShape::Box { half_extents } => {
            ray_vs_obb(origin, dir, max_dist, shape_pos, shape_rot, *half_extents)
        }
        CollisionShape::Capsule {
            radius,
            half_height,
        } => ray_vs_capsule(
            origin,
            dir,
            max_dist,
            shape_pos,
            shape_rot,
            *radius,
            *half_height,
        ),
        _ => None, // Complex shapes not supported for raycast in this implementation
    }
}

fn ray_vs_sphere(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    center: Vec3,
    radius: f32,
) -> Option<(f32, Vec3, Vec3)> {
    let oc = origin - center;
    let a = dir.dot(dir);
    let b = 2.0 * oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None;
    }

    let sqrt_d = discriminant.sqrt();
    let t = (-b - sqrt_d) / (2.0 * a);

    if t < 0.0 || t > max_dist {
        // Try the other root
        let t2 = (-b + sqrt_d) / (2.0 * a);
        if t2 < 0.0 || t2 > max_dist {
            return None;
        }
        let hit = origin + dir * t2;
        let normal = (hit - center).normalize();
        return Some((t2, hit, normal));
    }

    let hit = origin + dir * t;
    let normal = (hit - center).normalize();
    Some((t, hit, normal))
}

fn ray_vs_obb(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    box_pos: Vec3,
    box_rot: Quat,
    half_extents: Vec3,
) -> Option<(f32, Vec3, Vec3)> {
    // Transform ray into box local space
    let inv_rot = box_rot.inverse();
    let local_origin = inv_rot * (origin - box_pos);
    let local_dir = inv_rot * dir;

    // Standard AABB slab test in local space
    let inv_dir = Vec3::ONE / local_dir;
    let t1 = (-half_extents - local_origin) * inv_dir;
    let t2 = (half_extents - local_origin) * inv_dir;
    let t_min = t1.min(t2);
    let t_max = t1.max(t2);

    let t_enter = t_min.x.max(t_min.y).max(t_min.z);
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);

    if t_enter > t_exit || t_exit < 0.0 || t_enter > max_dist {
        return None;
    }

    let t = if t_enter >= 0.0 { t_enter } else { t_exit };
    if t > max_dist {
        return None;
    }

    let local_hit = local_origin + local_dir * t;
    let world_hit = box_pos + box_rot * local_hit;

    // Determine which face was hit by finding which component of local_hit is closest
    // to a face of the box
    let abs_hit = local_hit.abs();
    let ratios = abs_hit / half_extents;
    let local_normal = if ratios.x >= ratios.y && ratios.x >= ratios.z {
        Vec3::new(local_hit.x.signum(), 0.0, 0.0)
    } else if ratios.y >= ratios.x && ratios.y >= ratios.z {
        Vec3::new(0.0, local_hit.y.signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, local_hit.z.signum())
    };

    let world_normal = (box_rot * local_normal).normalize();
    Some((t, world_hit, world_normal))
}

fn ray_vs_capsule(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    cap_pos: Vec3,
    cap_rot: Quat,
    cap_radius: f32,
    cap_half_height: f32,
) -> Option<(f32, Vec3, Vec3)> {
    // Test against both hemispheres and the cylinder
    let up = cap_rot * Vec3::Y;
    let top = cap_pos + up * cap_half_height;
    let bot = cap_pos - up * cap_half_height;

    let mut best: Option<(f32, Vec3, Vec3)> = None;

    // Test hemispheres
    for center in [top, bot] {
        if let Some((t, hit, normal)) = ray_vs_sphere(origin, dir, max_dist, center, cap_radius) {
            if best.is_none() || t < best.as_ref().unwrap().0 {
                best = Some((t, hit, normal));
            }
        }
    }

    // Test cylinder (infinite cylinder around the capsule axis, then clip)
    // Transform to capsule local space
    let inv_rot = cap_rot.inverse();
    let local_origin = inv_rot * (origin - cap_pos);
    let local_dir = inv_rot * dir;

    // In local space, capsule is along Y axis. Cylinder test in XZ plane.
    let ox = local_origin.x;
    let oz = local_origin.z;
    let dx = local_dir.x;
    let dz = local_dir.z;

    let a = dx * dx + dz * dz;
    if a > 1e-10 {
        let b = 2.0 * (ox * dx + oz * dz);
        let c = ox * ox + oz * oz - cap_radius * cap_radius;
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            for sign in [-1.0f32, 1.0f32] {
                let t = (-b + sign * sqrt_disc) / (2.0 * a);
                if t >= 0.0 && t <= max_dist {
                    let local_hit = local_origin + local_dir * t;
                    // Check if within capsule height
                    if local_hit.y >= -cap_half_height && local_hit.y <= cap_half_height {
                        let world_hit = cap_pos + cap_rot * local_hit;
                        let local_normal =
                            Vec3::new(local_hit.x, 0.0, local_hit.z).normalize();
                        let world_normal = (cap_rot * local_normal).normalize();
                        if best.is_none() || t < best.as_ref().unwrap().0 {
                            best = Some((t, world_hit, world_normal));
                        }
                    }
                }
            }
        }
    }

    best
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_vs_sphere_collision() {
        // Two spheres overlapping
        let result = sphere_vs_sphere(Vec3::ZERO, 1.0, Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(result.is_some());
        let contacts = result.unwrap();
        assert_eq!(contacts.len(), 1);
        let c = &contacts[0];
        assert!((c.penetration_depth - 0.5).abs() < 1e-4);
        assert!((c.normal - Vec3::X).length() < 1e-4);
    }

    #[test]
    fn test_sphere_vs_sphere_no_collision() {
        let result = sphere_vs_sphere(Vec3::ZERO, 1.0, Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_sphere_vs_sphere_touching() {
        // Exactly touching (penetration = 0)
        let result = sphere_vs_sphere(Vec3::ZERO, 1.0, Vec3::new(2.0, 0.0, 0.0), 1.0);
        // penetration = 2.0 - 2.0 = 0.0, but dist_sq == sum_radii^2, so >= check means no collision
        assert!(result.is_none());
    }

    #[test]
    fn test_sphere_vs_box_collision() {
        let result = sphere_vs_box(
            Vec3::new(1.4, 0.0, 0.0),
            0.5,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
        );
        assert!(result.is_some());
        let contacts = result.unwrap();
        assert_eq!(contacts.len(), 1);
        assert!(contacts[0].penetration_depth > 0.0);
    }

    #[test]
    fn test_sphere_vs_box_no_collision() {
        let result = sphere_vs_box(
            Vec3::new(3.0, 0.0, 0.0),
            0.5,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_box_vs_box_sat_collision() {
        // Two overlapping boxes
        let result = box_vs_box(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            Vec3::new(1.5, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        );
        assert!(result.is_some());
        let contacts = result.unwrap();
        assert!(contacts[0].penetration_depth > 0.0);
        // Penetration should be 2.0 - 1.5 = 0.5
        assert!((contacts[0].penetration_depth - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_box_vs_box_sat_no_collision() {
        let result = box_vs_box(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            Vec3::new(3.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_box_vs_box_rotated() {
        // A box rotated 45 degrees around Y
        let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let result = box_vs_box(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            Vec3::new(2.0, 0.0, 0.0),
            rot,
            Vec3::ONE,
        );
        // The rotated box extends sqrt(2) along x, so it should overlap
        // Box A extends to x=1.0, rotated box B at x=2.0 extends from x=2-sqrt(2) to x=2+sqrt(2)
        // 2 - sqrt(2) ~ 0.586, so there is overlap with box A ending at 1.0
        assert!(result.is_some());
    }

    #[test]
    fn test_capsule_vs_capsule_collision() {
        let result = capsule_vs_capsule(
            Vec3::ZERO,
            Quat::IDENTITY,
            0.5,
            1.0,
            Vec3::new(0.8, 0.0, 0.0),
            Quat::IDENTITY,
            0.5,
            1.0,
        );
        assert!(result.is_some());
        let contacts = result.unwrap();
        assert!(contacts[0].penetration_depth > 0.0);
    }

    #[test]
    fn test_capsule_vs_capsule_no_collision() {
        let result = capsule_vs_capsule(
            Vec3::ZERO,
            Quat::IDENTITY,
            0.5,
            1.0,
            Vec3::new(3.0, 0.0, 0.0),
            Quat::IDENTITY,
            0.5,
            1.0,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_spatial_hash_grid_basic() {
        let mut grid = SpatialHashGrid::new(2.0);
        let h1 = ColliderHandle(1);
        let h2 = ColliderHandle(2);
        let h3 = ColliderHandle(3);

        grid.update(
            h1,
            AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        );
        grid.update(
            h2,
            AABB::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5)),
        );
        grid.update(
            h3,
            AABB::new(Vec3::new(10.0, 10.0, 10.0), Vec3::new(11.0, 11.0, 11.0)),
        );

        let pairs = grid.query_pairs();
        // h1 and h2 overlap, h3 is far away
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn test_layer_filtering() {
        assert!(layers_interact(
            CollisionLayer::DEFAULT,
            CollisionMask::ALL,
            CollisionLayer::DYNAMIC,
            CollisionMask::ALL,
        ));

        assert!(!layers_interact(
            CollisionLayer::DEFAULT,
            CollisionMask::NONE,
            CollisionLayer::DYNAMIC,
            CollisionMask::ALL,
        ));
    }

    #[test]
    fn test_ray_vs_sphere() {
        let result = ray_vs_sphere(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::X,
            100.0,
            Vec3::ZERO,
            1.0,
        );
        assert!(result.is_some());
        let (t, hit, normal) = result.unwrap();
        assert!((t - 4.0).abs() < 1e-4);
        assert!((hit.x - (-1.0)).abs() < 1e-4);
        assert!((normal - Vec3::NEG_X).length() < 1e-4);
    }

    #[test]
    fn test_ray_vs_sphere_miss() {
        let result = ray_vs_sphere(
            Vec3::new(-5.0, 3.0, 0.0),
            Vec3::X,
            100.0,
            Vec3::ZERO,
            1.0,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_aabb_computation_sphere() {
        let shape = CollisionShape::Sphere { radius: 2.0 };
        let aabb = shape.compute_aabb(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        assert!((aabb.min - Vec3::new(-1.0, 0.0, 1.0)).length() < 1e-4);
        assert!((aabb.max - Vec3::new(3.0, 4.0, 5.0)).length() < 1e-4);
    }

    #[test]
    fn test_aabb_computation_box() {
        let shape = CollisionShape::Box {
            half_extents: Vec3::ONE,
        };
        let aabb = shape.compute_aabb(Vec3::ZERO, Quat::IDENTITY);
        assert!((aabb.min - Vec3::new(-1.0, -1.0, -1.0)).length() < 1e-4);
        assert!((aabb.max - Vec3::new(1.0, 1.0, 1.0)).length() < 1e-4);
    }

    #[test]
    fn test_inertia_tensor_sphere() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let tensor = shape.compute_inertia_tensor(10.0);
        let expected = (2.0 / 5.0) * 10.0 * 1.0;
        assert!((tensor.x_axis.x - expected).abs() < 1e-4);
        assert!((tensor.y_axis.y - expected).abs() < 1e-4);
        assert!((tensor.z_axis.z - expected).abs() < 1e-4);
    }

    #[test]
    fn test_inertia_tensor_box() {
        let shape = CollisionShape::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let mass = 12.0;
        let tensor = shape.compute_inertia_tensor(mass);
        // Full extents: 2, 4, 6
        let w = 2.0;
        let h = 4.0;
        let d = 6.0;
        let factor = mass / 12.0;
        assert!((tensor.x_axis.x - factor * (h * h + d * d)).abs() < 1e-4);
        assert!((tensor.y_axis.y - factor * (w * w + d * d)).abs() < 1e-4);
        assert!((tensor.z_axis.z - factor * (w * w + h * h)).abs() < 1e-4);
    }

    #[test]
    fn test_closest_points_segments_parallel() {
        let (s, t, dist_sq) = closest_points_segments(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );
        assert!((dist_sq - 1.0).abs() < 1e-4);
        assert!(s >= 0.0 && s <= 1.0);
        assert!(t >= 0.0 && t <= 1.0);
    }

    #[test]
    fn test_sphere_volume() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let vol = shape.volume();
        let expected = (4.0 / 3.0) * std::f32::consts::PI;
        assert!((vol - expected).abs() < 1e-4);
    }

    #[test]
    fn test_box_volume() {
        let shape = CollisionShape::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let vol = shape.volume();
        assert!((vol - 48.0).abs() < 1e-4);
    }
}
