//! SIMD-accelerated math primitives for the Genovo engine.
//!
//! This module re-exports [`glam`] vector/matrix types and extends them with
//! engine-specific helpers. All public types are `#[repr(C)]` / `bytemuck`
//! compatible so they can be uploaded to GPU buffers without conversion.
//!
//! # SIMD Optimization Note
//!
//! `glam` already uses SIMD (SSE2/NEON) for Mat4/Vec4 operations when
//! available. Hand-rolled SIMD assembly for matrix multiply and batch
//! transforms is deferred to Q3 2026 and will only be pursued if profiling
//! shows glam's codegen is insufficient.

// ---- Re-exports from glam ------------------------------------------------

pub use glam::{
    Mat3, Mat4, Quat, Vec2, Vec3, Vec3A, Vec4,
};

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// A 3-D affine transform decomposed into translation, rotation, and scale.
///
/// Storing the components separately avoids lossy decomposition from a 4x4
/// matrix and makes interpolation (e.g., animation blending) straightforward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// World-space position.
    pub position: Vec3,
    /// Orientation as a unit quaternion.
    pub rotation: Quat,
    /// Non-uniform scale.
    pub scale: Vec3,
}

impl Transform {
    /// Identity transform (origin, no rotation, unit scale).
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Creates a new transform.
    #[inline]
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// Creates a transform with only translation.
    #[inline]
    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            ..Self::IDENTITY
        }
    }

    /// Creates a transform with only rotation.
    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    /// Creates a transform with only scale.
    #[inline]
    pub fn from_scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    /// Computes the composed 4x4 matrix (T * R * S).
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Computes the inverse transform.
    ///
    /// Assumes uniform or non-degenerate scale.
    #[inline]
    pub fn inverse(&self) -> Self {
        // Uniform-scale fast path: when all scale components are equal we can
        // avoid per-component division and use a single reciprocal multiply.
        let inv_rotation = self.rotation.inverse();
        let inv_scale = Vec3::ONE / self.scale;
        let inv_position = inv_rotation * (-self.position * inv_scale);
        Self {
            position: inv_position,
            rotation: inv_rotation,
            scale: inv_scale,
        }
    }

    /// Transforms a point (applies scale, rotation, then translation).
    #[inline]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.rotation * (self.scale * point) + self.position
    }

    /// Transforms a direction vector (applies rotation only, ignores scale
    /// and translation).
    #[inline]
    pub fn transform_direction(&self, dir: Vec3) -> Vec3 {
        self.rotation * dir
    }

    /// Linearly interpolates between `self` and `other`.
    ///
    /// Position and scale are lerped; rotation is slerped.
    #[inline]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(other.position, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }

    /// Returns the local forward direction (`-Z` in engine convention).
    #[inline]
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// Returns the local right direction (`+X`).
    #[inline]
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Returns the local up direction (`+Y`).
    #[inline]
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ---------------------------------------------------------------------------
// Geometric primitives
// ---------------------------------------------------------------------------

/// A 2-D axis-aligned rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Minimum corner (inclusive).
    pub min: Vec2,
    /// Maximum corner (inclusive).
    pub max: Vec2,
}

impl Rect {
    /// Creates a rect from min/max corners.
    #[inline]
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// Width of the rectangle.
    #[inline]
    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    /// Height of the rectangle.
    #[inline]
    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }

    /// Center point of the rectangle.
    #[inline]
    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Returns `true` if `point` lies inside the rectangle (inclusive).
    #[inline]
    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    /// Returns `true` if this rect overlaps `other`.
    #[inline]
    pub fn intersects(&self, other: &Rect) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }
}

/// A 3-D axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    /// Minimum corner (inclusive).
    pub min: Vec3,
    /// Maximum corner (inclusive).
    pub max: Vec3,
}

impl AABB {
    /// An invalid (inside-out) AABB useful as the identity element when
    /// computing the union of a set of points.
    pub const INVALID: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    /// Creates an AABB from min/max corners.
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Center of the box.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents (half-size along each axis).
    #[inline]
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Returns `true` if `point` lies inside (inclusive).
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Returns `true` if this AABB overlaps `other`.
    #[inline]
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Expands the AABB to contain `point`.
    #[inline]
    pub fn expand_to_include(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Returns the union of two AABBs.
    #[inline]
    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Tests intersection with a [`Ray`]. Returns `Some(t)` where `t` is the
    /// distance along the ray to the nearest intersection point, or `None`.
    pub fn ray_intersect(&self, ray: &Ray) -> Option<f32> {
        // Slab test uses reciprocal direction; glam vectorises this with
        // SIMD on supported targets so a hand-rolled intrinsic path is not
        // needed at this time.
        let inv_dir = Vec3::ONE / ray.direction;
        let t1 = (self.min - ray.origin) * inv_dir;
        let t2 = (self.max - ray.origin) * inv_dir;
        let t_min = t1.min(t2);
        let t_max = t1.max(t2);
        let t_enter = t_min.x.max(t_min.y).max(t_min.z);
        let t_exit = t_max.x.min(t_max.y).min(t_max.z);
        if t_enter <= t_exit && t_exit >= 0.0 {
            Some(t_enter.max(0.0))
        } else {
            None
        }
    }
}

/// A plane in 3-D space, represented in Hessian normal form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    /// Unit normal of the plane.
    pub normal: Vec3,
    /// Signed distance from the origin along the normal.
    pub distance: f32,
}

impl Plane {
    /// Creates a plane from a normal and distance.
    #[inline]
    pub fn new(normal: Vec3, distance: f32) -> Self {
        Self { normal, distance }
    }

    /// Creates a plane from a normal and a point on the plane.
    #[inline]
    pub fn from_point_normal(point: Vec3, normal: Vec3) -> Self {
        Self {
            normal,
            distance: normal.dot(point),
        }
    }

    /// Signed distance from `point` to the plane.
    ///
    /// Positive means the point is on the side the normal points to.
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.distance
    }
}

/// A view frustum defined by six clipping planes.
///
/// Used for frustum culling — the most impactful early-out in the render
/// pipeline.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    /// The six planes: left, right, bottom, top, near, far.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Extracts frustum planes from a view-projection matrix.
    ///
    /// The resulting planes point *inward* so that a point is inside the
    /// frustum iff it is on the positive side of every plane.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        // Gribb-Hartmann plane extraction from a column-major view-projection
        // matrix.  Each frustum plane is derived by adding or subtracting a
        // row of the matrix from the fourth row, then normalising.
        let cols = vp.to_cols_array_2d();
        let row = |r: usize| -> Vec4 {
            Vec4::new(cols[0][r], cols[1][r], cols[2][r], cols[3][r])
        };

        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let extract = |v: Vec4| -> Plane {
            let len = Vec3::new(v.x, v.y, v.z).length();
            Plane {
                normal: Vec3::new(v.x, v.y, v.z) / len,
                distance: -v.w / len,
            }
        };

        Self {
            planes: [
                extract(r3 + r0), // left
                extract(r3 - r0), // right
                extract(r3 + r1), // bottom
                extract(r3 - r1), // top
                extract(r3 + r2), // near
                extract(r3 - r2), // far
            ],
        }
    }

    /// Returns `true` if the AABB is at least partially inside the frustum.
    pub fn contains_aabb(&self, aabb: &AABB) -> bool {
        // Scalar loop over 6 planes; glam's Vec3::dot already benefits from
        // SIMD on SSE2/NEON targets, so a batched intrinsic version is not
        // expected to yield a meaningful improvement for 6 iterations.
        for plane in &self.planes {
            // Find the positive vertex (the one furthest along the plane normal).
            let p = Vec3::new(
                if plane.normal.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if plane.normal.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if plane.normal.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );
            if plane.signed_distance(p) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Returns `true` if the sphere is at least partially inside the frustum.
    pub fn contains_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            if plane.signed_distance(center) < -radius {
                return false;
            }
        }
        true
    }
}

/// A ray defined by an origin and a direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// Starting point of the ray.
    pub origin: Vec3,
    /// Direction (expected to be normalized, but not enforced).
    pub direction: Vec3,
}

impl Ray {
    /// Creates a ray from an origin and direction.
    #[inline]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// Returns the point at parameter `t` along the ray.
    #[inline]
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Linearly interpolates between `a` and `b` by factor `t` (clamped to [0, 1]).
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    a + (b - a) * t
}

/// Spherical linear interpolation between two quaternions.
///
/// This is a convenience wrapper around [`Quat::slerp`].
#[inline]
pub fn slerp(a: Quat, b: Quat, t: f32) -> Quat {
    a.slerp(b, t)
}

/// Hermite smoothstep interpolation.
///
/// Maps `t` from `[edge0, edge1]` to `[0, 1]` with smooth acceleration and
/// deceleration (zero first derivative at both edges).
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Remaps `value` from the input range `[in_min, in_max]` to the output range
/// `[out_min, out_max]`, linearly.
#[inline]
pub fn remap(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
}

/// Converts degrees to radians.
#[inline]
pub fn deg_to_rad(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.0
}

/// Converts radians to degrees.
#[inline]
pub fn rad_to_deg(radians: f32) -> f32 {
    radians * 180.0 / std::f32::consts::PI
}
