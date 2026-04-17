// engine/render/src/raymarching.rs
//
// Signed distance field (SDF) ray-marching system for the Genovo engine.
//
// Implements a full CPU-side SDF ray-marching pipeline:
//
// - **SDF primitives** — Sphere, box, cylinder, torus, cone, plane, capsule,
//   ellipsoid, hexagonal prism, triangular prism.
// - **Boolean operations** — Union, intersection, subtraction, smooth union,
//   smooth intersection, smooth subtraction.
// - **SDF transformations** — Translate, rotate, scale, elongate, round, onion,
//   twist, bend, repetition (infinite and clamped).
// - **Scene representation** — Tree of SDF operations with material IDs.
// - **Ray-marcher** — Sphere tracing with configurable step count and epsilon.
// - **Normal estimation** — Central difference gradient.
// - **Ambient occlusion** — SDF-based horizon-based AO.
// - **Soft shadows** — SDF-based penumbra shadow estimation.
// - **Material blending** — Smooth union carries interpolated material weights.
//
// # GPU pipeline overview
//
// In production, these functions would run in a compute or fragment shader.
// The CPU implementation here serves as a reference and for offline/preview
// rendering (e.g. editor thumbnails).

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Vector helpers (self-contained, no external dependencies)
// ---------------------------------------------------------------------------

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const RIGHT: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
    pub const FORWARD: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            Self::ZERO
        } else {
            self * (1.0 / len)
        }
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    #[inline]
    pub fn max_comp(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    #[inline]
    pub fn min_comp(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    #[inline]
    pub fn max_element(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    #[inline]
    pub fn min_element(self) -> f32 {
        self.x.min(self.y).min(self.z)
    }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }

    #[inline]
    pub fn reflect(self, normal: Self) -> Self {
        self - normal * (2.0 * self.dot(normal))
    }

    /// Component-wise clamp.
    #[inline]
    pub fn clamp(self, min: f32, max: f32) -> Self {
        Self {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
        }
    }

    /// XY swizzle.
    #[inline]
    pub fn xy(self) -> (f32, f32) {
        (self.x, self.y)
    }

    /// XZ swizzle.
    #[inline]
    pub fn xz(self) -> (f32, f32) {
        (self.x, self.z)
    }

    /// Rotate around the Y axis.
    pub fn rotate_y(self, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x: self.x * c + self.z * s,
            y: self.y,
            z: -self.x * s + self.z * c,
        }
    }

    /// Rotate around the X axis.
    pub fn rotate_x(self, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x: self.x,
            y: self.y * c - self.z * s,
            z: self.y * s + self.z * c,
        }
    }

    /// Rotate around the Z axis.
    pub fn rotate_z(self, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x: self.x * c - self.y * s,
            y: self.x * s + self.y * c,
            z: self.z,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs }
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3 { x: self * rhs.x, y: self * rhs.y, z: self * rhs.z }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self { x: self.x * inv, y: self.y * inv, z: self.z * inv }
    }
}

impl std::ops::Mul for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self { x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z }
    }
}

// ---------------------------------------------------------------------------
// SDF result
// ---------------------------------------------------------------------------

/// Result of an SDF evaluation: distance + material ID.
#[derive(Debug, Clone, Copy)]
pub struct SdfResult {
    /// Signed distance to the surface.
    pub distance: f32,
    /// Material identifier for shading.
    pub material_id: u32,
    /// Blending weight (used during smooth boolean operations).
    pub blend_weight: f32,
}

impl SdfResult {
    #[inline]
    pub fn new(distance: f32, material_id: u32) -> Self {
        Self { distance, material_id, blend_weight: 1.0 }
    }

    /// Combine two results via hard union (take the closer surface).
    #[inline]
    pub fn union(self, other: Self) -> Self {
        if self.distance < other.distance { self } else { other }
    }

    /// Combine two results via hard intersection (take the farther surface).
    #[inline]
    pub fn intersection(self, other: Self) -> Self {
        if self.distance > other.distance { self } else { other }
    }

    /// Combine two results via hard subtraction (A minus B).
    #[inline]
    pub fn subtraction(self, other: Self) -> Self {
        let neg = SdfResult::new(-other.distance, other.material_id);
        if self.distance > neg.distance { self } else { neg }
    }
}

// ---------------------------------------------------------------------------
// SDF Primitives
// ---------------------------------------------------------------------------

/// Signed distance to a sphere centred at the origin.
///
/// # Arguments
/// * `p` — Query point.
/// * `radius` — Sphere radius.
#[inline]
pub fn sdf_sphere(p: Vec3, radius: f32) -> f32 {
    p.length() - radius
}

/// Signed distance to an axis-aligned box centred at the origin.
///
/// # Arguments
/// * `p` — Query point.
/// * `half_extents` — Half-size along each axis.
#[inline]
pub fn sdf_box(p: Vec3, half_extents: Vec3) -> f32 {
    let q = p.abs() - half_extents;
    let outside = q.max_comp(Vec3::ZERO).length();
    let inside = q.max_element().min(0.0);
    outside + inside
}

/// Signed distance to a cylinder aligned along the Y axis.
///
/// # Arguments
/// * `p` — Query point.
/// * `radius` — Cylinder radius.
/// * `half_height` — Half-height of the cylinder.
pub fn sdf_cylinder(p: Vec3, radius: f32, half_height: f32) -> f32 {
    let d_xz = (p.x * p.x + p.z * p.z).sqrt() - radius;
    let d_y = p.y.abs() - half_height;
    let outside = Vec3::new(d_xz.max(0.0), d_y.max(0.0), 0.0).length();
    let inside = d_xz.max(d_y).min(0.0);
    outside + inside
}

/// Signed distance to a torus centred at the origin, lying in the XZ plane.
///
/// # Arguments
/// * `p` — Query point.
/// * `major_radius` — Radius of the ring.
/// * `minor_radius` — Radius of the tube.
#[inline]
pub fn sdf_torus(p: Vec3, major_radius: f32, minor_radius: f32) -> f32 {
    let q_x = (p.x * p.x + p.z * p.z).sqrt() - major_radius;
    let q = Vec3::new(q_x, p.y, 0.0);
    q.length() - minor_radius
}

/// Signed distance to a cone with the tip at the origin pointing up.
///
/// # Arguments
/// * `p` — Query point.
/// * `angle` — Half-angle of the cone (radians).
/// * `height` — Cone height.
pub fn sdf_cone(p: Vec3, angle: f32, height: f32) -> f32 {
    let (sin_a, cos_a) = angle.sin_cos();
    let q = Vec3::new((p.x * p.x + p.z * p.z).sqrt(), p.y, 0.0);

    let tip = q.dot(Vec3::new(sin_a, cos_a, 0.0));
    let body = q.dot(Vec3::new(cos_a, -sin_a, 0.0));

    let d1 = tip;
    let d2 = q.y - height;

    let w = d1.max(d2);
    if body < 0.0 && d2 < 0.0 {
        w
    } else {
        let clamped_y = q.y.clamp(0.0, height);
        let r_at_h = clamped_y * (sin_a / cos_a);
        let proj = Vec3::new(r_at_h, clamped_y, 0.0);
        (q - proj).length()
    }
}

/// Signed distance to an infinite plane with normal `n` at distance `d` from origin.
///
/// # Arguments
/// * `p` — Query point.
/// * `n` — Plane normal (must be normalised).
/// * `d` — Plane distance from origin.
#[inline]
pub fn sdf_plane(p: Vec3, n: Vec3, d: f32) -> f32 {
    p.dot(n) - d
}

/// Signed distance to a capsule (line segment with rounded ends).
///
/// # Arguments
/// * `p` — Query point.
/// * `a` — Start of the capsule line.
/// * `b` — End of the capsule line.
/// * `radius` — Capsule radius.
pub fn sdf_capsule(p: Vec3, a: Vec3, b: Vec3, radius: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    (pa - ba * h).length() - radius
}

/// Signed distance to an ellipsoid at the origin.
///
/// # Arguments
/// * `p` — Query point.
/// * `radii` — Radii along each axis.
pub fn sdf_ellipsoid(p: Vec3, radii: Vec3) -> f32 {
    // Approximate: scale space then use sphere SDF.
    let scaled = Vec3::new(p.x / radii.x, p.y / radii.y, p.z / radii.z);
    let k0 = scaled.length();
    let k1 = Vec3::new(
        p.x / (radii.x * radii.x),
        p.y / (radii.y * radii.y),
        p.z / (radii.z * radii.z),
    )
    .length();
    if k1 < 1e-10 {
        return 0.0;
    }
    k0 * (k0 - 1.0) / k1
}

/// Signed distance to a hexagonal prism aligned along the Y axis.
///
/// # Arguments
/// * `p` — Query point.
/// * `h` — Half-height along Y.
/// * `r` — Radius of the circumscribed circle.
pub fn sdf_hex_prism(p: Vec3, h: f32, r: f32) -> f32 {
    let k = Vec3::new(-0.8660254038, 0.5, 0.57735027); // cos(60), sin(60), tan(30)
    let ap = p.abs();

    let dot2 = 2.0 * (k.x * ap.x + k.y * ap.z).min(0.0);
    let px = ap.x - dot2 * k.x;
    let pz = ap.z - dot2 * k.y;

    let d_xz = Vec3::new(
        px - (px).clamp(-k.z * r, k.z * r),
        pz - r,
        0.0,
    )
    .length()
        * if pz - r > 0.0 { 1.0 } else { -1.0 };
    let d_y = ap.y - h;

    d_xz.max(d_y)
}

/// Signed distance to a triangular prism aligned along Z.
///
/// # Arguments
/// * `p` — Query point.
/// * `h` — Half-height along Z.
/// * `r` — Radius of the circumscribed circle in the XY plane.
pub fn sdf_tri_prism(p: Vec3, h: f32, r: f32) -> f32 {
    let q = p.abs();
    let d_z = q.z - h;
    let d_xy = 0.5 * q.x + 0.8660254038 * q.y - r;
    let d_max = d_xy.max(-q.y * 0.8660254038 + q.x * 0.5 - r * 0.5);
    d_max.max(d_z)
}

/// Signed distance to a rounded box (box with rounded edges).
///
/// # Arguments
/// * `p` — Query point.
/// * `half_extents` — Half-size along each axis.
/// * `radius` — Rounding radius.
#[inline]
pub fn sdf_rounded_box(p: Vec3, half_extents: Vec3, radius: f32) -> f32 {
    sdf_box(p, half_extents - Vec3::splat(radius)) - radius
}

// ---------------------------------------------------------------------------
// Boolean operations
// ---------------------------------------------------------------------------

/// Hard union of two distances.
#[inline]
pub fn op_union(d1: f32, d2: f32) -> f32 {
    d1.min(d2)
}

/// Hard intersection of two distances.
#[inline]
pub fn op_intersection(d1: f32, d2: f32) -> f32 {
    d1.max(d2)
}

/// Hard subtraction (d1 minus d2).
#[inline]
pub fn op_subtraction(d1: f32, d2: f32) -> f32 {
    d1.max(-d2)
}

/// Smooth union (smooth minimum) with blending factor `k`.
///
/// Larger `k` → smoother blend.
pub fn op_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return d1.min(d2);
    }
    let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
    lerp(d2, d1, h) - k * h * (1.0 - h)
}

/// Smooth intersection with blending factor `k`.
pub fn op_smooth_intersection(d1: f32, d2: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return d1.max(d2);
    }
    let h = (0.5 - 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
    lerp(d2, d1, h) + k * h * (1.0 - h)
}

/// Smooth subtraction with blending factor `k`.
pub fn op_smooth_subtraction(d1: f32, d2: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return d1.max(-d2);
    }
    let h = (0.5 - 0.5 * (d2 + d1) / k).clamp(0.0, 1.0);
    lerp(d1, -d2, h) + k * h * (1.0 - h)
}

/// Smooth union with material blending: returns (distance, blend_weight for d1).
pub fn op_smooth_union_mat(d1: f32, d2: f32, k: f32) -> (f32, f32) {
    if k <= 0.0 {
        return if d1 < d2 { (d1, 1.0) } else { (d2, 0.0) };
    }
    let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
    let d = lerp(d2, d1, h) - k * h * (1.0 - h);
    (d, h)
}

/// Scalar linear interpolation.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// SDF transformations
// ---------------------------------------------------------------------------

/// Translate the SDF query point.
#[inline]
pub fn op_translate(p: Vec3, offset: Vec3) -> Vec3 {
    p - offset
}

/// Scale the SDF query point (remember to multiply the result by `scale`).
#[inline]
pub fn op_scale(p: Vec3, scale: f32) -> Vec3 {
    p * (1.0 / scale)
}

/// Twist deformation around the Y axis.
///
/// # Arguments
/// * `p` — Query point.
/// * `k` — Twist amount (radians per unit along Y).
pub fn op_twist(p: Vec3, k: f32) -> Vec3 {
    let angle = k * p.y;
    let (s, c) = angle.sin_cos();
    Vec3::new(p.x * c - p.z * s, p.y, p.x * s + p.z * c)
}

/// Bend deformation around the Z axis.
///
/// # Arguments
/// * `p` — Query point.
/// * `k` — Bend amount (radians per unit along X).
pub fn op_bend(p: Vec3, k: f32) -> Vec3 {
    let angle = k * p.x;
    let (s, c) = angle.sin_cos();
    Vec3::new(p.x * c - p.y * s, p.x * s + p.y * c, p.z)
}

/// Infinite repetition: returns the local-space point within the repeated cell.
///
/// # Arguments
/// * `p` — Query point.
/// * `cell_size` — Size of each repetition cell.
pub fn op_repeat_infinite(p: Vec3, cell_size: Vec3) -> Vec3 {
    let modp = |v: f32, s: f32| -> f32 {
        if s <= 0.0 {
            return v;
        }
        v - s * (v / s + 0.5).floor()
    };
    Vec3::new(
        modp(p.x, cell_size.x),
        modp(p.y, cell_size.y),
        modp(p.z, cell_size.z),
    )
}

/// Clamped repetition: repeats `count` times in each direction.
///
/// # Arguments
/// * `p` — Query point.
/// * `cell_size` — Size of each repetition cell.
/// * `count` — Number of repetitions in each direction (total = 2*count+1).
pub fn op_repeat_clamped(p: Vec3, cell_size: Vec3, count: Vec3) -> Vec3 {
    let clamp_rep = |v: f32, s: f32, c: f32| -> f32 {
        if s <= 0.0 {
            return v;
        }
        v - s * (v / s + 0.5).floor().clamp(-c, c)
    };
    Vec3::new(
        clamp_rep(p.x, cell_size.x, count.x),
        clamp_rep(p.y, cell_size.y, count.y),
        clamp_rep(p.z, cell_size.z, count.z),
    )
}

/// Elongation: stretches the SDF along each axis by the given half-extents.
///
/// Returns the transformed point. After evaluating the SDF, add back the
/// clamped offset length.
pub fn op_elongate(p: Vec3, half_ext: Vec3) -> (Vec3, f32) {
    let q = p.abs() - half_ext;
    let clamped = Vec3::new(q.x.min(0.0), q.y.min(0.0), q.z.min(0.0));
    let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
    let correction = clamped.length();
    (outside, correction)
}

/// Rounding: applies a rounding radius to any SDF.
#[inline]
pub fn op_round(d: f32, radius: f32) -> f32 {
    d - radius
}

/// Onion: creates a shell of the given thickness from any SDF.
#[inline]
pub fn op_onion(d: f32, thickness: f32) -> f32 {
    d.abs() - thickness
}

/// Symmetric fold along a plane.
///
/// Reflects the point to the positive side of the plane defined by `normal`.
pub fn op_fold(p: Vec3, normal: Vec3) -> Vec3 {
    let d = p.dot(normal);
    if d < 0.0 {
        p - normal * (2.0 * d)
    } else {
        p
    }
}

// ---------------------------------------------------------------------------
// SDF Scene
// ---------------------------------------------------------------------------

/// Operations that combine child SDF nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SdfOp {
    /// Leaf primitive.
    Primitive,
    /// Union of children.
    Union,
    /// Intersection of children.
    Intersection,
    /// Subtraction: first child minus all others.
    Subtraction,
    /// Smooth union with blending factor.
    SmoothUnion(f32),
    /// Smooth intersection with blending factor.
    SmoothIntersection(f32),
    /// Smooth subtraction with blending factor.
    SmoothSubtraction(f32),
}

/// SDF primitive type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SdfPrimitive {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Cylinder { radius: f32, half_height: f32 },
    Torus { major: f32, minor: f32 },
    Cone { angle: f32, height: f32 },
    Plane { normal: Vec3, dist: f32 },
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    Ellipsoid { radii: Vec3 },
    HexPrism { half_height: f32, radius: f32 },
    TriPrism { half_height: f32, radius: f32 },
    RoundedBox { half_extents: Vec3, radius: f32 },
}

impl SdfPrimitive {
    /// Evaluate the distance from point `p` to this primitive.
    pub fn evaluate(&self, p: Vec3) -> f32 {
        match *self {
            Self::Sphere { radius } => sdf_sphere(p, radius),
            Self::Box { half_extents } => sdf_box(p, half_extents),
            Self::Cylinder { radius, half_height } => sdf_cylinder(p, radius, half_height),
            Self::Torus { major, minor } => sdf_torus(p, major, minor),
            Self::Cone { angle, height } => sdf_cone(p, angle, height),
            Self::Plane { normal, dist } => sdf_plane(p, normal, dist),
            Self::Capsule { a, b, radius } => sdf_capsule(p, a, b, radius),
            Self::Ellipsoid { radii } => sdf_ellipsoid(p, radii),
            Self::HexPrism { half_height, radius } => sdf_hex_prism(p, half_height, radius),
            Self::TriPrism { half_height, radius } => sdf_tri_prism(p, half_height, radius),
            Self::RoundedBox { half_extents, radius } => sdf_rounded_box(p, half_extents, radius),
        }
    }
}

/// A transform applied to an SDF node before evaluation.
#[derive(Debug, Clone, Copy)]
pub struct SdfTransform {
    /// Translation offset.
    pub translation: Vec3,
    /// Rotation around each axis (Euler angles in radians, applied X→Y→Z).
    pub rotation: Vec3,
    /// Uniform scale factor.
    pub scale: f32,
    /// Twist factor (radians per unit along Y).
    pub twist: f32,
    /// Bend factor (radians per unit along X).
    pub bend: f32,
}

impl Default for SdfTransform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: 1.0,
            twist: 0.0,
            bend: 0.0,
        }
    }
}

impl SdfTransform {
    /// Identity transform.
    pub fn identity() -> Self {
        Self::default()
    }

    /// Translation-only transform.
    pub fn translate(offset: Vec3) -> Self {
        Self { translation: offset, ..Self::default() }
    }

    /// Uniform scale transform.
    pub fn uniform_scale(scale: f32) -> Self {
        Self { scale, ..Self::default() }
    }

    /// Apply this transform to a query point (inverse transform).
    pub fn apply(&self, p: Vec3) -> (Vec3, f32) {
        let mut q = p - self.translation;

        // Inverse rotation (apply in reverse order Z→Y→X with negated angles).
        if self.rotation.z != 0.0 {
            q = q.rotate_z(-self.rotation.z);
        }
        if self.rotation.y != 0.0 {
            q = q.rotate_y(-self.rotation.y);
        }
        if self.rotation.x != 0.0 {
            q = q.rotate_x(-self.rotation.x);
        }

        // Inverse scale.
        let scale_factor = self.scale;
        if scale_factor != 1.0 && scale_factor != 0.0 {
            q = q * (1.0 / scale_factor);
        }

        // Twist deformation.
        if self.twist != 0.0 {
            q = op_twist(q, self.twist);
        }

        // Bend deformation.
        if self.bend != 0.0 {
            q = op_bend(q, self.bend);
        }

        (q, scale_factor)
    }
}

/// A node in the SDF scene tree.
#[derive(Debug, Clone)]
pub struct SdfNode {
    /// Combination operation.
    pub op: SdfOp,
    /// Local transform.
    pub transform: SdfTransform,
    /// Primitive data (only used when `op == SdfOp::Primitive`).
    pub primitive: Option<SdfPrimitive>,
    /// Material ID for shading.
    pub material_id: u32,
    /// Child nodes (used for boolean operations).
    pub children: Vec<SdfNode>,
    /// Optional rounding radius.
    pub rounding: f32,
    /// Optional onion (shell) thickness. 0 = disabled.
    pub onion: f32,
    /// Optional repetition cell size. `Vec3::ZERO` = disabled.
    pub repetition: Vec3,
    /// Optional repetition count (per direction). Only used if repetition != ZERO.
    pub repetition_count: Option<Vec3>,
}

impl SdfNode {
    /// Create a leaf node with a primitive.
    pub fn primitive(prim: SdfPrimitive, material_id: u32) -> Self {
        Self {
            op: SdfOp::Primitive,
            transform: SdfTransform::default(),
            primitive: Some(prim),
            material_id,
            children: Vec::new(),
            rounding: 0.0,
            onion: 0.0,
            repetition: Vec3::ZERO,
            repetition_count: None,
        }
    }

    /// Create a boolean combination node.
    pub fn combine(op: SdfOp, children: Vec<SdfNode>) -> Self {
        Self {
            op,
            transform: SdfTransform::default(),
            primitive: None,
            material_id: 0,
            children,
            rounding: 0.0,
            onion: 0.0,
            repetition: Vec3::ZERO,
            repetition_count: None,
        }
    }

    /// Set the node's local transform.
    pub fn with_transform(mut self, transform: SdfTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Set rounding.
    pub fn with_rounding(mut self, radius: f32) -> Self {
        self.rounding = radius;
        self
    }

    /// Set onion (shell) mode.
    pub fn with_onion(mut self, thickness: f32) -> Self {
        self.onion = thickness;
        self
    }

    /// Set infinite repetition.
    pub fn with_repetition(mut self, cell_size: Vec3) -> Self {
        self.repetition = cell_size;
        self.repetition_count = None;
        self
    }

    /// Set clamped repetition.
    pub fn with_clamped_repetition(mut self, cell_size: Vec3, count: Vec3) -> Self {
        self.repetition = cell_size;
        self.repetition_count = Some(count);
        self
    }

    /// Recursively evaluate this SDF node at point `p`.
    pub fn evaluate(&self, p: Vec3) -> SdfResult {
        // Apply the node's local transform.
        let (mut q, scale) = self.transform.apply(p);

        // Apply repetition if set.
        if self.repetition.x > 0.0 || self.repetition.y > 0.0 || self.repetition.z > 0.0 {
            if let Some(count) = self.repetition_count {
                q = op_repeat_clamped(q, self.repetition, count);
            } else {
                q = op_repeat_infinite(q, self.repetition);
            }
        }

        let mut result = match self.op {
            SdfOp::Primitive => {
                if let Some(ref prim) = self.primitive {
                    let d = prim.evaluate(q);
                    SdfResult::new(d, self.material_id)
                } else {
                    SdfResult::new(f32::MAX, 0)
                }
            }
            SdfOp::Union => {
                let mut r = SdfResult::new(f32::MAX, 0);
                for child in &self.children {
                    r = r.union(child.evaluate(q));
                }
                r
            }
            SdfOp::Intersection => {
                let mut r = SdfResult::new(f32::MIN, 0);
                for child in &self.children {
                    r = r.intersection(child.evaluate(q));
                }
                r
            }
            SdfOp::Subtraction => {
                if self.children.is_empty() {
                    SdfResult::new(f32::MAX, 0)
                } else {
                    let mut r = self.children[0].evaluate(q);
                    for child in &self.children[1..] {
                        r = r.subtraction(child.evaluate(q));
                    }
                    r
                }
            }
            SdfOp::SmoothUnion(k) => {
                let mut r = SdfResult::new(f32::MAX, 0);
                for child in &self.children {
                    let child_r = child.evaluate(q);
                    let (d, w) = op_smooth_union_mat(r.distance, child_r.distance, k);
                    r = SdfResult {
                        distance: d,
                        material_id: if w > 0.5 { r.material_id } else { child_r.material_id },
                        blend_weight: w,
                    };
                }
                r
            }
            SdfOp::SmoothIntersection(k) => {
                let mut r = SdfResult::new(f32::MIN, 0);
                for (i, child) in self.children.iter().enumerate() {
                    let child_r = child.evaluate(q);
                    if i == 0 {
                        r = child_r;
                    } else {
                        r.distance = op_smooth_intersection(r.distance, child_r.distance, k);
                    }
                }
                r
            }
            SdfOp::SmoothSubtraction(k) => {
                if self.children.is_empty() {
                    SdfResult::new(f32::MAX, 0)
                } else {
                    let mut r = self.children[0].evaluate(q);
                    for child in &self.children[1..] {
                        let child_r = child.evaluate(q);
                        r.distance = op_smooth_subtraction(r.distance, child_r.distance, k);
                    }
                    r
                }
            }
        };

        // Apply rounding.
        if self.rounding > 0.0 {
            result.distance = op_round(result.distance, self.rounding);
        }

        // Apply onion (shell).
        if self.onion > 0.0 {
            result.distance = op_onion(result.distance, self.onion);
        }

        // Scale the distance back.
        result.distance *= scale;

        result
    }
}

/// A complete SDF scene.
#[derive(Debug, Clone)]
pub struct SdfScene {
    /// Root node of the scene.
    pub root: SdfNode,
    /// Scene bounding sphere radius (for early-out ray clipping).
    pub bounds_radius: f32,
}

impl SdfScene {
    /// Create a new scene with a single root node.
    pub fn new(root: SdfNode, bounds_radius: f32) -> Self {
        Self { root, bounds_radius }
    }

    /// Evaluate the scene SDF at point `p`.
    #[inline]
    pub fn evaluate(&self, p: Vec3) -> SdfResult {
        self.root.evaluate(p)
    }
}

// ---------------------------------------------------------------------------
// Ray and ray-marching
// ---------------------------------------------------------------------------

/// A ray with origin and normalised direction.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Create a new ray.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalize() }
    }

    /// Point along the ray at parameter `t`.
    #[inline]
    pub fn at(self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Result of a ray-march hit.
#[derive(Debug, Clone, Copy)]
pub struct RayMarchHit {
    /// Was a surface hit?
    pub hit: bool,
    /// Distance from origin along the ray.
    pub t: f32,
    /// World-space hit position.
    pub position: Vec3,
    /// Surface normal at the hit point.
    pub normal: Vec3,
    /// Number of marching steps taken.
    pub steps: u32,
    /// Material ID at the hit point.
    pub material_id: u32,
    /// Material blend weight (for smooth booleans).
    pub blend_weight: f32,
}

impl RayMarchHit {
    /// A miss result.
    pub fn miss() -> Self {
        Self {
            hit: false,
            t: f32::MAX,
            position: Vec3::ZERO,
            normal: Vec3::UP,
            steps: 0,
            material_id: 0,
            blend_weight: 1.0,
        }
    }
}

/// Configuration for the ray-marcher.
#[derive(Debug, Clone)]
pub struct RayMarchConfig {
    /// Maximum number of marching steps.
    pub max_steps: u32,
    /// Maximum ray distance.
    pub max_distance: f32,
    /// Surface hit epsilon.
    pub epsilon: f32,
    /// Epsilon for normal estimation (central differences).
    pub normal_epsilon: f32,
    /// Minimum step size (to avoid getting stuck).
    pub min_step: f32,
    /// Step size relaxation factor (omega for sphere tracing).
    /// 1.0 = standard sphere tracing, > 1.0 = over-relaxation.
    pub relaxation: f32,
}

impl Default for RayMarchConfig {
    fn default() -> Self {
        Self {
            max_steps: 256,
            max_distance: 200.0,
            epsilon: 0.0005,
            normal_epsilon: 0.0005,
            min_step: 0.0001,
            relaxation: 1.0,
        }
    }
}

impl RayMarchConfig {
    /// High quality preset.
    pub fn high_quality() -> Self {
        Self {
            max_steps: 512,
            max_distance: 500.0,
            epsilon: 0.0001,
            normal_epsilon: 0.0001,
            min_step: 0.00005,
            relaxation: 1.0,
        }
    }

    /// Low quality / fast preview preset.
    pub fn preview() -> Self {
        Self {
            max_steps: 64,
            max_distance: 100.0,
            epsilon: 0.005,
            normal_epsilon: 0.002,
            min_step: 0.001,
            relaxation: 1.2,
        }
    }
}

/// The SDF ray-marcher.
pub struct RayMarcher {
    pub config: RayMarchConfig,
}

impl RayMarcher {
    /// Create a new ray-marcher with the given configuration.
    pub fn new(config: RayMarchConfig) -> Self {
        Self { config }
    }

    /// March a ray through the SDF scene using sphere tracing.
    ///
    /// # Arguments
    /// * `ray` — The ray to march.
    /// * `scene` — The SDF scene to evaluate.
    pub fn march(&self, ray: &Ray, scene: &SdfScene) -> RayMarchHit {
        let mut t = 0.0_f32;
        let mut prev_d = f32::MAX;

        for step in 0..self.config.max_steps {
            let p = ray.at(t);
            let result = scene.evaluate(p);
            let d = result.distance;

            // Over-relaxation sphere tracing.
            let step_size = if self.config.relaxation > 1.0 {
                // Check if the over-relaxed step missed a surface.
                let candidate = d * self.config.relaxation;
                if candidate + prev_d < prev_d {
                    // Fallback to standard step.
                    d
                } else {
                    candidate
                }
            } else {
                d
            };

            if d < self.config.epsilon {
                let normal = self.estimate_normal(p, scene);
                return RayMarchHit {
                    hit: true,
                    t,
                    position: p,
                    normal,
                    steps: step,
                    material_id: result.material_id,
                    blend_weight: result.blend_weight,
                };
            }

            t += step_size.max(self.config.min_step);
            prev_d = d;

            if t > self.config.max_distance {
                break;
            }
        }

        RayMarchHit {
            hit: false,
            t,
            position: ray.at(t),
            normal: Vec3::UP,
            steps: self.config.max_steps,
            material_id: 0,
            blend_weight: 1.0,
        }
    }

    /// Estimate the surface normal at point `p` using the central difference gradient.
    pub fn estimate_normal(&self, p: Vec3, scene: &SdfScene) -> Vec3 {
        let e = self.config.normal_epsilon;
        // Tetrahedron technique for 4 evaluations instead of 6.
        let k0 = Vec3::new(1.0, -1.0, -1.0);
        let k1 = Vec3::new(-1.0, -1.0, 1.0);
        let k2 = Vec3::new(-1.0, 1.0, -1.0);
        let k3 = Vec3::new(1.0, 1.0, 1.0);

        let d0 = scene.evaluate(p + k0 * e).distance;
        let d1 = scene.evaluate(p + k1 * e).distance;
        let d2 = scene.evaluate(p + k2 * e).distance;
        let d3 = scene.evaluate(p + k3 * e).distance;

        let n = k0 * d0 + k1 * d1 + k2 * d2 + k3 * d3;
        n.normalize()
    }

    /// Estimate the surface normal using 6-tap central differences (more robust).
    pub fn estimate_normal_central(&self, p: Vec3, scene: &SdfScene) -> Vec3 {
        let e = self.config.normal_epsilon;
        let ex = Vec3::new(e, 0.0, 0.0);
        let ey = Vec3::new(0.0, e, 0.0);
        let ez = Vec3::new(0.0, 0.0, e);

        let nx = scene.evaluate(p + ex).distance - scene.evaluate(p - ex).distance;
        let ny = scene.evaluate(p + ey).distance - scene.evaluate(p - ey).distance;
        let nz = scene.evaluate(p + ez).distance - scene.evaluate(p - ez).distance;

        Vec3::new(nx, ny, nz).normalize()
    }

    /// Compute SDF-based ambient occlusion at the given point.
    ///
    /// # Arguments
    /// * `p` — Surface point.
    /// * `n` — Surface normal.
    /// * `scene` — The SDF scene.
    /// * `num_steps` — Number of AO sampling steps (typically 5).
    /// * `step_size` — Distance between each AO sample (typically 0.02-0.1).
    /// * `falloff` — AO attenuation factor per step (typically 0.5-0.85).
    pub fn ambient_occlusion(
        &self,
        p: Vec3,
        n: Vec3,
        scene: &SdfScene,
        num_steps: u32,
        step_size: f32,
        falloff: f32,
    ) -> f32 {
        let mut ao = 0.0_f32;
        let mut weight = 1.0_f32;

        for i in 1..=num_steps {
            let dist = step_size * i as f32;
            let sample_point = p + n * dist;
            let sdf_dist = scene.evaluate(sample_point).distance;

            // How much less distance than expected → occlusion.
            ao += weight * (dist - sdf_dist).max(0.0);
            weight *= falloff;
        }

        (1.0 - ao.min(1.0)).max(0.0)
    }

    /// Compute soft shadows using the SDF.
    ///
    /// # Arguments
    /// * `p` — Surface point (slightly offset along normal).
    /// * `light_dir` — Direction *toward* the light (normalised).
    /// * `scene` — The SDF scene.
    /// * `min_t` — Minimum ray parameter (bias to avoid self-intersection).
    /// * `max_t` — Maximum shadow ray distance.
    /// * `k` — Softness factor (higher = harder shadow). Typical: 8-64.
    pub fn soft_shadow(
        &self,
        p: Vec3,
        light_dir: Vec3,
        scene: &SdfScene,
        min_t: f32,
        max_t: f32,
        k: f32,
    ) -> f32 {
        let mut result = 1.0_f32;
        let mut t = min_t;
        let mut prev_h = f32::MAX;

        for _ in 0..self.config.max_steps {
            if t >= max_t {
                break;
            }

            let sample = p + light_dir * t;
            let h = scene.evaluate(sample).distance;

            if h < self.config.epsilon {
                return 0.0;
            }

            // Improved soft shadow (Quilez 2010).
            let y = h * h / (2.0 * prev_h);
            let d = (h * h - y * y).max(0.0).sqrt();
            result = result.min(k * d / (t - y).max(0.0));

            prev_h = h;
            t += h.max(self.config.min_step);
        }

        result.clamp(0.0, 1.0)
    }

    /// Hard shadow test: returns true if occluded, false if lit.
    pub fn hard_shadow(
        &self,
        p: Vec3,
        light_dir: Vec3,
        scene: &SdfScene,
        min_t: f32,
        max_t: f32,
    ) -> bool {
        let mut t = min_t;

        for _ in 0..self.config.max_steps {
            if t >= max_t {
                return false;
            }

            let sample = p + light_dir * t;
            let h = scene.evaluate(sample).distance;

            if h < self.config.epsilon {
                return true;
            }

            t += h.max(self.config.min_step);
        }

        false
    }
}

// ---------------------------------------------------------------------------
// Camera helpers for generating rays
// ---------------------------------------------------------------------------

/// A simple pinhole camera for generating ray-march rays.
#[derive(Debug, Clone)]
pub struct SdfCamera {
    /// Camera position in world space.
    pub position: Vec3,
    /// Camera look-at target.
    pub target: Vec3,
    /// Up vector.
    pub up: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Image aspect ratio (width / height).
    pub aspect: f32,
}

impl SdfCamera {
    /// Create a new SDF camera.
    pub fn new(position: Vec3, target: Vec3, fov_y_deg: f32, aspect: f32) -> Self {
        Self {
            position,
            target,
            up: Vec3::UP,
            fov_y: fov_y_deg * PI / 180.0,
            aspect,
        }
    }

    /// Generate a ray for the given pixel coordinates (normalised to [-1, 1]).
    pub fn ray(&self, ndc_x: f32, ndc_y: f32) -> Ray {
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(self.up).normalize();
        let cam_up = right.cross(forward).normalize();

        let half_h = (self.fov_y * 0.5).tan();
        let half_w = half_h * self.aspect;

        let direction = (forward + right * (ndc_x * half_w) + cam_up * (ndc_y * half_h)).normalize();

        Ray::new(self.position, direction)
    }
}

// ---------------------------------------------------------------------------
// Simple shading for the ray-marcher
// ---------------------------------------------------------------------------

/// Colour as linear RGB.
#[derive(Debug, Clone, Copy)]
pub struct Color3 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color3 {
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };

    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub fn scale(self, s: f32) -> Self {
        Self { r: self.r * s, g: self.g * s, b: self.b * s }
    }

    pub fn add(self, other: Self) -> Self {
        Self { r: self.r + other.r, g: self.g + other.g, b: self.b + other.b }
    }

    pub fn mul(self, other: Self) -> Self {
        Self { r: self.r * other.r, g: self.g * other.g, b: self.b * other.b }
    }

    pub fn clamp01(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }
}

/// A simple material definition for SDF shading.
#[derive(Debug, Clone)]
pub struct SdfMaterial {
    /// Base albedo colour.
    pub albedo: Color3,
    /// Roughness [0, 1].
    pub roughness: f32,
    /// Metalness [0, 1].
    pub metalness: f32,
    /// Emission colour (linear HDR).
    pub emission: Color3,
    /// Ambient colour contribution.
    pub ambient: Color3,
}

impl Default for SdfMaterial {
    fn default() -> Self {
        Self {
            albedo: Color3::new(0.8, 0.8, 0.8),
            roughness: 0.5,
            metalness: 0.0,
            emission: Color3::BLACK,
            ambient: Color3::new(0.05, 0.05, 0.08),
        }
    }
}

/// A directional light for SDF shading.
#[derive(Debug, Clone)]
pub struct SdfLight {
    /// Direction *toward* the light (normalised).
    pub direction: Vec3,
    /// Light colour and intensity (linear HDR).
    pub color: Color3,
    /// Whether to compute soft shadows for this light.
    pub cast_shadow: bool,
    /// Shadow softness factor (higher = harder).
    pub shadow_hardness: f32,
}

impl SdfLight {
    /// Create a default directional light.
    pub fn directional(direction: Vec3, color: Color3) -> Self {
        Self {
            direction: direction.normalize(),
            color,
            cast_shadow: true,
            shadow_hardness: 16.0,
        }
    }
}

/// Shade a ray-march hit with a simple Blinn-Phong-like model.
pub fn shade_hit(
    hit: &RayMarchHit,
    ray: &Ray,
    scene: &SdfScene,
    marcher: &RayMarcher,
    materials: &[SdfMaterial],
    lights: &[SdfLight],
    background: Color3,
) -> Color3 {
    if !hit.hit {
        return background;
    }

    let mat_idx = (hit.material_id as usize).min(materials.len().saturating_sub(1));
    let mat = if materials.is_empty() {
        &SdfMaterial::default()
    } else {
        &materials[mat_idx]
    };

    let n = hit.normal;
    let v = (ray.origin - hit.position).normalize();
    let p = hit.position + n * (marcher.config.epsilon * 2.0);

    // Ambient occlusion.
    let ao = marcher.ambient_occlusion(p, n, scene, 5, 0.05, 0.75);

    let mut color = mat.ambient.scale(ao);
    color = color.add(mat.emission);

    for light in lights {
        let l = light.direction;
        let h = (l + v).normalize();

        let n_dot_l = n.dot(l).max(0.0);
        let n_dot_h = n.dot(h).max(0.0);

        if n_dot_l <= 0.0 {
            continue;
        }

        // Shadow.
        let shadow = if light.cast_shadow {
            marcher.soft_shadow(p, l, scene, 0.01, 50.0, light.shadow_hardness)
        } else {
            1.0
        };

        // Diffuse.
        let diffuse = mat.albedo.mul(light.color).scale(n_dot_l * shadow);

        // Specular (Blinn-Phong approximation).
        let spec_power = (2.0 / (mat.roughness * mat.roughness + 0.01) - 2.0).max(1.0);
        let spec_intensity = n_dot_h.powf(spec_power) * shadow;
        let spec_color = if mat.metalness > 0.0 {
            mat.albedo.lerp(Color3::WHITE, 1.0 - mat.metalness)
        } else {
            Color3::new(0.04, 0.04, 0.04)
        };
        let specular = spec_color.mul(light.color).scale(spec_intensity);

        color = color.add(diffuse).add(specular);
    }

    color
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

/// Render the scene to a flat pixel buffer (RGBA, u8).
///
/// # Arguments
/// * `width` — Image width in pixels.
/// * `height` — Image height in pixels.
/// * `camera` — The camera.
/// * `scene` — The SDF scene.
/// * `materials` — Material palette.
/// * `lights` — Lights.
/// * `background` — Background colour.
/// * `config` — Ray marcher configuration.
pub fn render_scene(
    width: u32,
    height: u32,
    camera: &SdfCamera,
    scene: &SdfScene,
    materials: &[SdfMaterial],
    lights: &[SdfLight],
    background: Color3,
    config: RayMarchConfig,
) -> Vec<u8> {
    let marcher = RayMarcher::new(config);
    let mut pixels = vec![0u8; (width * height * 4) as usize];

    for y in 0..height {
        for x in 0..width {
            let ndc_x = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
            let ndc_y = 1.0 - (y as f32 + 0.5) / height as f32 * 2.0;

            let ray = camera.ray(ndc_x, ndc_y);
            let hit = marcher.march(&ray, scene);
            let color = shade_hit(&hit, &ray, scene, &marcher, materials, lights, background);

            let idx = ((y * width + x) * 4) as usize;
            pixels[idx] = (color.r.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
            pixels[idx + 1] = (color.g.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
            pixels[idx + 2] = (color.b.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
            pixels[idx + 3] = 255;
        }
    }

    pixels
}

/// Compute the depth map (linear depth values) for a rendered scene.
pub fn render_depth(
    width: u32,
    height: u32,
    camera: &SdfCamera,
    scene: &SdfScene,
    config: RayMarchConfig,
) -> Vec<f32> {
    let marcher = RayMarcher::new(config);
    let mut depth = vec![0.0f32; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let ndc_x = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
            let ndc_y = 1.0 - (y as f32 + 0.5) / height as f32 * 2.0;

            let ray = camera.ray(ndc_x, ndc_y);
            let hit = marcher.march(&ray, scene);
            let idx = (y * width + x) as usize;
            depth[idx] = if hit.hit { hit.t } else { f32::MAX };
        }
    }

    depth
}

/// Compute a normal map for a rendered scene.
pub fn render_normals(
    width: u32,
    height: u32,
    camera: &SdfCamera,
    scene: &SdfScene,
    config: RayMarchConfig,
) -> Vec<u8> {
    let marcher = RayMarcher::new(config);
    let mut pixels = vec![0u8; (width * height * 4) as usize];

    for y in 0..height {
        for x in 0..width {
            let ndc_x = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
            let ndc_y = 1.0 - (y as f32 + 0.5) / height as f32 * 2.0;

            let ray = camera.ray(ndc_x, ndc_y);
            let hit = marcher.march(&ray, scene);

            let idx = ((y * width + x) * 4) as usize;
            if hit.hit {
                pixels[idx] = ((hit.normal.x * 0.5 + 0.5) * 255.0) as u8;
                pixels[idx + 1] = ((hit.normal.y * 0.5 + 0.5) * 255.0) as u8;
                pixels[idx + 2] = ((hit.normal.z * 0.5 + 0.5) * 255.0) as u8;
            }
            pixels[idx + 3] = 255;
        }
    }

    pixels
}

/// Compute the AO buffer for a rendered scene.
pub fn render_ao(
    width: u32,
    height: u32,
    camera: &SdfCamera,
    scene: &SdfScene,
    config: RayMarchConfig,
    ao_steps: u32,
    ao_step_size: f32,
) -> Vec<f32> {
    let marcher = RayMarcher::new(config);
    let mut ao_buf = vec![1.0f32; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let ndc_x = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
            let ndc_y = 1.0 - (y as f32 + 0.5) / height as f32 * 2.0;

            let ray = camera.ray(ndc_x, ndc_y);
            let hit = marcher.march(&ray, scene);
            let idx = (y * width + x) as usize;
            if hit.hit {
                let p = hit.position + hit.normal * (marcher.config.epsilon * 2.0);
                ao_buf[idx] = marcher.ambient_occlusion(p, hit.normal, scene, ao_steps, ao_step_size, 0.75);
            }
        }
    }

    ao_buf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sdf() {
        let d = sdf_sphere(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!((d - 2.0).abs() < 1e-6);

        let d = sdf_sphere(Vec3::new(0.5, 0.0, 0.0), 1.0);
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_box_sdf() {
        let d = sdf_box(Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert!((d - 1.0).abs() < 1e-6);

        let d = sdf_box(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        assert!(d < 0.0);
    }

    #[test]
    fn test_smooth_union() {
        let d1 = 1.0_f32;
        let d2 = 2.0_f32;
        let su = op_smooth_union(d1, d2, 0.5);
        assert!(su <= d1);
        assert!(su < d2);
    }

    #[test]
    fn test_ray_march_sphere() {
        let scene = SdfScene::new(
            SdfNode::primitive(SdfPrimitive::Sphere { radius: 1.0 }, 0),
            10.0,
        );
        let marcher = RayMarcher::new(RayMarchConfig::default());
        let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = marcher.march(&ray, &scene);

        assert!(hit.hit);
        assert!((hit.t - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_normal_estimation() {
        let scene = SdfScene::new(
            SdfNode::primitive(SdfPrimitive::Sphere { radius: 1.0 }, 0),
            10.0,
        );
        let marcher = RayMarcher::new(RayMarchConfig::default());
        let p = Vec3::new(1.0, 0.0, 0.0);
        let n = marcher.estimate_normal(p, &scene);

        assert!((n.x - 1.0).abs() < 0.01);
        assert!(n.y.abs() < 0.01);
        assert!(n.z.abs() < 0.01);
    }

    #[test]
    fn test_ambient_occlusion() {
        let scene = SdfScene::new(
            SdfNode::primitive(SdfPrimitive::Sphere { radius: 1.0 }, 0),
            10.0,
        );
        let marcher = RayMarcher::new(RayMarchConfig::default());
        let p = Vec3::new(1.001, 0.0, 0.0);
        let n = Vec3::new(1.0, 0.0, 0.0);
        let ao = marcher.ambient_occlusion(p, n, &scene, 5, 0.05, 0.75);

        // AO on the outside of a sphere should be close to 1.0.
        assert!(ao > 0.8);
    }

    #[test]
    fn test_soft_shadow() {
        let scene = SdfScene::new(
            SdfNode::primitive(SdfPrimitive::Sphere { radius: 1.0 }, 0),
            10.0,
        );
        let marcher = RayMarcher::new(RayMarchConfig::default());

        // Point behind the sphere, light in front.
        let p = Vec3::new(0.0, 0.0, -3.0);
        let light_dir = Vec3::new(0.0, 0.0, 1.0);
        let shadow = marcher.soft_shadow(p, light_dir, &scene, 0.01, 50.0, 16.0);
        assert!(shadow < 0.1); // Should be heavily shadowed.
    }
}
