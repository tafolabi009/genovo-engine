// engine/render/src/frustum_culling.rs
//
// Production frustum culling system. Extracts six clipping planes from
// a View-Projection matrix, tests AABBs against those planes, and outputs
// a visible set. Supports batch testing with SIMD-friendly data layout,
// early-out on fully-outside results, and coherency hints.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Vec3 / Vec4 / Mat4 (local lightweight math, no external dependency)
// ---------------------------------------------------------------------------

/// A simple 3-component vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    #[inline]
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 {
            return Self::ZERO;
        }
        Self { x: self.x / l, y: self.y / l, z: self.z / l }
    }

    #[inline]
    pub fn min_components(a: Self, b: Self) -> Self {
        Self { x: a.x.min(b.x), y: a.y.min(b.y), z: a.z.min(b.z) }
    }

    #[inline]
    pub fn max_components(a: Self, b: Self) -> Self {
        Self { x: a.x.max(b.x), y: a.y.max(b.y), z: a.z.max(b.z) }
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
    fn mul(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

/// A simple 4-component vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn xyz(self) -> Vec3 {
        Vec3 { x: self.x, y: self.y, z: self.z }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
}

/// Row-major 4x4 matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub m: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    #[inline]
    pub fn from_rows(r0: [f32; 4], r1: [f32; 4], r2: [f32; 4], r3: [f32; 4]) -> Self {
        Self { m: [r0, r1, r2, r3] }
    }

    /// Return column `c` as Vec4.
    #[inline]
    pub fn col(&self, c: usize) -> Vec4 {
        Vec4::new(self.m[0][c], self.m[1][c], self.m[2][c], self.m[3][c])
    }

    /// Return row `r` as Vec4.
    #[inline]
    pub fn row(&self, r: usize) -> Vec4 {
        Vec4::new(self.m[r][0], self.m[r][1], self.m[r][2], self.m[r][3])
    }

    /// Multiply self * rhs.
    pub fn mul_mat4(&self, rhs: &Mat4) -> Mat4 {
        let mut out = [[0.0f32; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                out[r][c] = self.m[r][0] * rhs.m[0][c]
                    + self.m[r][1] * rhs.m[1][c]
                    + self.m[r][2] * rhs.m[2][c]
                    + self.m[r][3] * rhs.m[3][c];
            }
        }
        Mat4 { m: out }
    }

    /// Build perspective projection (OpenGL clip space, -1..1 Z).
    pub fn perspective(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y_radians * 0.5).tan();
        let nf = 1.0 / (near - far);
        Self::from_rows(
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) * nf, 2.0 * far * near * nf],
            [0.0, 0.0, -1.0, 0.0],
        )
    }

    /// Build a look-at view matrix (right-handed).
    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalized();
        let s = f.cross(up).normalized();
        let u = s.cross(f);
        Self::from_rows(
            [s.x, s.y, s.z, -s.dot(eye)],
            [u.x, u.y, u.z, -u.dot(eye)],
            [-f.x, -f.y, -f.z, f.dot(eye)],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Build an orthographic projection matrix.
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let rl = 1.0 / (right - left);
        let tb = 1.0 / (top - bottom);
        let fn_ = 1.0 / (far - near);
        Self::from_rows(
            [2.0 * rl, 0.0, 0.0, -(right + left) * rl],
            [0.0, 2.0 * tb, 0.0, -(top + bottom) * tb],
            [0.0, 0.0, -2.0 * fn_, -(far + near) * fn_],
            [0.0, 0.0, 0.0, 1.0],
        )
    }

    /// Transform Vec4 by this matrix.
    #[inline]
    pub fn transform_vec4(&self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z + self.m[0][3] * v.w,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z + self.m[1][3] * v.w,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z + self.m[2][3] * v.w,
            self.m[3][0] * v.x + self.m[3][1] * v.y + self.m[3][2] * v.z + self.m[3][3] * v.w,
        )
    }
}

// ---------------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------------

/// A plane in Hessian normal form: `n.dot(p) + d = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrustumPlane {
    pub normal: Vec3,
    pub d: f32,
}

impl FrustumPlane {
    #[inline]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        let len = (a * a + b * b + c * c).sqrt();
        if len < 1e-12 {
            return Self { normal: Vec3::new(0.0, 1.0, 0.0), d: 0.0 };
        }
        let inv = 1.0 / len;
        Self {
            normal: Vec3::new(a * inv, b * inv, c * inv),
            d: d * inv,
        }
    }

    /// Signed distance from point to plane (positive = in front).
    #[inline]
    pub fn distance_to_point(&self, p: Vec3) -> f32 {
        self.normal.dot(p) + self.d
    }

    /// Distance from AABB center to plane along the AABB extents.
    #[inline]
    pub fn distance_to_aabb(&self, aabb: &AABB) -> (f32, f32) {
        let center = aabb.center();
        let extents = aabb.extents();
        let dist = self.normal.dot(center) + self.d;
        let radius = extents.x * self.normal.x.abs()
            + extents.y * self.normal.y.abs()
            + extents.z * self.normal.z.abs();
        (dist - radius, dist + radius)
    }
}

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub const EMPTY: Self = Self {
        min: Vec3 { x: f32::MAX, y: f32::MAX, z: f32::MAX },
        max: Vec3 { x: f32::MIN, y: f32::MIN, z: f32::MIN },
    };

    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn from_center_extents(center: Vec3, extents: Vec3) -> Self {
        Self {
            min: center - extents,
            max: center + extents,
        }
    }

    #[inline]
    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    #[inline]
    pub fn extents(&self) -> Vec3 {
        Vec3::new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        )
    }

    #[inline]
    pub fn size(&self) -> Vec3 {
        Vec3::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }

    #[inline]
    pub fn surface_area(&self) -> f32 {
        let s = self.size();
        2.0 * (s.x * s.y + s.y * s.z + s.z * s.x)
    }

    #[inline]
    pub fn volume(&self) -> f32 {
        let s = self.size();
        s.x * s.y * s.z
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    #[inline]
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    #[inline]
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    #[inline]
    pub fn merge(&self, other: &AABB) -> AABB {
        AABB {
            min: Vec3::min_components(self.min, other.min),
            max: Vec3::max_components(self.max, other.max),
        }
    }

    #[inline]
    pub fn expand_point(&self, p: Vec3) -> AABB {
        AABB {
            min: Vec3::min_components(self.min, p),
            max: Vec3::max_components(self.max, p),
        }
    }

    /// Returns the 8 corner vertices of this AABB.
    pub fn corners(&self) -> [Vec3; 8] {
        [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// Get the positive vertex (farthest along normal direction).
    #[inline]
    pub fn positive_vertex(&self, normal: Vec3) -> Vec3 {
        Vec3::new(
            if normal.x >= 0.0 { self.max.x } else { self.min.x },
            if normal.y >= 0.0 { self.max.y } else { self.min.y },
            if normal.z >= 0.0 { self.max.z } else { self.min.z },
        )
    }

    /// Get the negative vertex (closest along normal direction).
    #[inline]
    pub fn negative_vertex(&self, normal: Vec3) -> Vec3 {
        Vec3::new(
            if normal.x >= 0.0 { self.min.x } else { self.max.x },
            if normal.y >= 0.0 { self.min.y } else { self.max.y },
            if normal.z >= 0.0 { self.min.z } else { self.max.z },
        )
    }
}

// ---------------------------------------------------------------------------
// Bounding Sphere
// ---------------------------------------------------------------------------

/// A bounding sphere defined by center and radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

impl BoundingSphere {
    #[inline]
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Compute a tight bounding sphere for an AABB.
    pub fn from_aabb(aabb: &AABB) -> Self {
        let center = aabb.center();
        let radius = (aabb.max - center).length();
        Self { center, radius }
    }

    /// Merge two bounding spheres.
    pub fn merge(a: &Self, b: &Self) -> Self {
        let diff = b.center - a.center;
        let dist = diff.length();
        if dist + b.radius <= a.radius {
            return *a;
        }
        if dist + a.radius <= b.radius {
            return *b;
        }
        let new_radius = (dist + a.radius + b.radius) * 0.5;
        let t = (new_radius - a.radius) / dist;
        let new_center = a.center + diff * t;
        Self { center: new_center, radius: new_radius }
    }
}

// ---------------------------------------------------------------------------
// Frustum
// ---------------------------------------------------------------------------

/// Which plane of the frustum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrustumPlaneId {
    Left = 0,
    Right = 1,
    Bottom = 2,
    Top = 3,
    Near = 4,
    Far = 5,
}

/// Result of a frustum test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrustumTestResult {
    /// Fully inside the frustum (all corners inside).
    Inside,
    /// Partially inside the frustum (intersects at least one plane).
    Intersect,
    /// Fully outside the frustum.
    Outside,
}

/// A camera frustum represented as six planes extracted from a VP matrix.
#[derive(Debug, Clone)]
pub struct Frustum {
    pub planes: [FrustumPlane; 6],
    /// Cached absolute normal components for fast AABB tests.
    abs_normals: [[f32; 3]; 6],
}

impl Frustum {
    /// Extract frustum planes from a View-Projection matrix (row-major).
    /// Uses the Gribb-Hartmann method.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let row0 = vp.row(0);
        let row1 = vp.row(1);
        let row2 = vp.row(2);
        let row3 = vp.row(3);

        // Left:   row3 + row0
        let left = FrustumPlane::new(
            row3.x + row0.x,
            row3.y + row0.y,
            row3.z + row0.z,
            row3.w + row0.w,
        );
        // Right:  row3 - row0
        let right = FrustumPlane::new(
            row3.x - row0.x,
            row3.y - row0.y,
            row3.z - row0.z,
            row3.w - row0.w,
        );
        // Bottom: row3 + row1
        let bottom = FrustumPlane::new(
            row3.x + row1.x,
            row3.y + row1.y,
            row3.z + row1.z,
            row3.w + row1.w,
        );
        // Top:    row3 - row1
        let top = FrustumPlane::new(
            row3.x - row1.x,
            row3.y - row1.y,
            row3.z - row1.z,
            row3.w - row1.w,
        );
        // Near:   row3 + row2
        let near = FrustumPlane::new(
            row3.x + row2.x,
            row3.y + row2.y,
            row3.z + row2.z,
            row3.w + row2.w,
        );
        // Far:    row3 - row2
        let far = FrustumPlane::new(
            row3.x - row2.x,
            row3.y - row2.y,
            row3.z - row2.z,
            row3.w - row2.w,
        );

        let planes = [left, right, bottom, top, near, far];
        let abs_normals = [
            [left.normal.x.abs(), left.normal.y.abs(), left.normal.z.abs()],
            [right.normal.x.abs(), right.normal.y.abs(), right.normal.z.abs()],
            [bottom.normal.x.abs(), bottom.normal.y.abs(), bottom.normal.z.abs()],
            [top.normal.x.abs(), top.normal.y.abs(), top.normal.z.abs()],
            [near.normal.x.abs(), near.normal.y.abs(), near.normal.z.abs()],
            [far.normal.x.abs(), far.normal.y.abs(), far.normal.z.abs()],
        ];

        Self { planes, abs_normals }
    }

    /// Test an AABB against the frustum.
    /// Returns `Outside`, `Inside`, or `Intersect`.
    pub fn test_aabb(&self, aabb: &AABB) -> FrustumTestResult {
        let center = aabb.center();
        let extents = aabb.extents();
        let mut all_inside = true;

        for i in 0..6 {
            let plane = &self.planes[i];
            let abs_n = &self.abs_normals[i];

            // Signed distance from center to plane.
            let dist = plane.normal.x * center.x
                + plane.normal.y * center.y
                + plane.normal.z * center.z
                + plane.d;

            // Projection interval radius.
            let radius = abs_n[0] * extents.x + abs_n[1] * extents.y + abs_n[2] * extents.z;

            // If the nearest point is outside, the whole AABB is outside.
            if dist < -radius {
                return FrustumTestResult::Outside;
            }

            // If the farthest point is outside, we have an intersection (not fully inside).
            if dist < radius {
                all_inside = false;
            }
        }

        if all_inside {
            FrustumTestResult::Inside
        } else {
            FrustumTestResult::Intersect
        }
    }

    /// Fast boolean test: is the AABB at least partially inside?
    #[inline]
    pub fn is_aabb_visible(&self, aabb: &AABB) -> bool {
        self.test_aabb(aabb) != FrustumTestResult::Outside
    }

    /// Test a bounding sphere against the frustum.
    pub fn test_sphere(&self, sphere: &BoundingSphere) -> FrustumTestResult {
        let mut all_inside = true;

        for plane in &self.planes {
            let dist = plane.normal.dot(sphere.center) + plane.d;
            if dist < -sphere.radius {
                return FrustumTestResult::Outside;
            }
            if dist < sphere.radius {
                all_inside = false;
            }
        }

        if all_inside {
            FrustumTestResult::Inside
        } else {
            FrustumTestResult::Intersect
        }
    }

    /// Test a single point.
    pub fn test_point(&self, point: Vec3) -> bool {
        for plane in &self.planes {
            if plane.distance_to_point(point) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Get one of the six planes by ID.
    #[inline]
    pub fn plane(&self, id: FrustumPlaneId) -> &FrustumPlane {
        &self.planes[id as usize]
    }

    /// Compute the 8 corners of the frustum by intersecting plane triples.
    pub fn corners(&self) -> Option<[Vec3; 8]> {
        // Near plane corners
        let intersect = |p0: &FrustumPlane, p1: &FrustumPlane, p2: &FrustumPlane| -> Option<Vec3> {
            let n0 = p0.normal;
            let n1 = p1.normal;
            let n2 = p2.normal;
            let denom = n0.dot(n1.cross(n2));
            if denom.abs() < 1e-8 {
                return None;
            }
            let p = (n1.cross(n2) * (-p0.d) + n2.cross(n0) * (-p1.d) + n0.cross(n1) * (-p2.d))
                * (1.0 / denom);
            Some(p)
        };

        let l = &self.planes[0];
        let r = &self.planes[1];
        let b = &self.planes[2];
        let t = &self.planes[3];
        let n = &self.planes[4];
        let f = &self.planes[5];

        Some([
            intersect(n, l, b)?,  // near-left-bottom
            intersect(n, r, b)?,  // near-right-bottom
            intersect(n, l, t)?,  // near-left-top
            intersect(n, r, t)?,  // near-right-top
            intersect(f, l, b)?,  // far-left-bottom
            intersect(f, r, b)?,  // far-right-bottom
            intersect(f, l, t)?,  // far-left-top
            intersect(f, r, t)?,  // far-right-top
        ])
    }
}

// ---------------------------------------------------------------------------
// CullObjectId — a strongly-typed handle for objects in the culling system
// ---------------------------------------------------------------------------

/// A unique identifier for a cullable object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CullObjectId(pub u64);

impl CullObjectId {
    #[inline]
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

// ---------------------------------------------------------------------------
// CullObject — per-object data for the culling system
// ---------------------------------------------------------------------------

/// Per-object data for frustum culling.
#[derive(Debug, Clone)]
pub struct CullObject {
    pub id: CullObjectId,
    pub aabb: AABB,
    pub bounding_sphere: BoundingSphere,
    /// Bit flags for custom layers (e.g. shadow caster, main camera, etc.).
    pub layer_mask: u32,
    /// If true, this object is always visible (never culled).
    pub always_visible: bool,
    /// Screen-size threshold below which the object is culled (in pixels^2).
    pub min_screen_size: f32,
    /// Cached result from the last frame for temporal coherence.
    last_visible: bool,
    /// Which plane rejected this object last frame (coherence hint).
    last_rejecting_plane: u8,
}

impl CullObject {
    pub fn new(id: CullObjectId, aabb: AABB) -> Self {
        Self {
            id,
            aabb,
            bounding_sphere: BoundingSphere::from_aabb(&aabb),
            layer_mask: 0xFFFF_FFFF,
            always_visible: false,
            min_screen_size: 0.0,
            last_visible: true,
            last_rejecting_plane: 0,
        }
    }

    pub fn with_layer_mask(mut self, mask: u32) -> Self {
        self.layer_mask = mask;
        self
    }

    pub fn with_always_visible(mut self, always: bool) -> Self {
        self.always_visible = always;
        self
    }

    pub fn with_min_screen_size(mut self, pixels_sq: f32) -> Self {
        self.min_screen_size = pixels_sq;
        self
    }

    /// Update AABB and recompute bounding sphere.
    pub fn update_bounds(&mut self, aabb: AABB) {
        self.aabb = aabb;
        self.bounding_sphere = BoundingSphere::from_aabb(&aabb);
    }
}

// ---------------------------------------------------------------------------
// VisibleSet — the output of a culling pass
// ---------------------------------------------------------------------------

/// The result of a frustum culling pass.
#[derive(Debug, Clone)]
pub struct VisibleSet {
    /// IDs of visible objects, in the order they were found.
    pub visible_ids: Vec<CullObjectId>,
    /// Per-object test result (only for those tested).
    pub results: Vec<(CullObjectId, FrustumTestResult)>,
    /// Total objects tested.
    pub total_tested: u32,
    /// Total objects visible.
    pub total_visible: u32,
    /// Total objects culled.
    pub total_culled: u32,
    /// Time taken for the cull pass in microseconds.
    pub cull_time_us: u64,
}

impl VisibleSet {
    pub fn new() -> Self {
        Self {
            visible_ids: Vec::new(),
            results: Vec::new(),
            total_tested: 0,
            total_visible: 0,
            total_culled: 0,
            cull_time_us: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            visible_ids: Vec::with_capacity(cap),
            results: Vec::with_capacity(cap),
            total_tested: 0,
            total_visible: 0,
            total_culled: 0,
            cull_time_us: 0,
        }
    }

    pub fn clear(&mut self) {
        self.visible_ids.clear();
        self.results.clear();
        self.total_tested = 0;
        self.total_visible = 0;
        self.total_culled = 0;
        self.cull_time_us = 0;
    }

    pub fn is_visible(&self, id: CullObjectId) -> bool {
        self.visible_ids.contains(&id)
    }
}

impl Default for VisibleSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CullCamera — a camera configuration for culling
// ---------------------------------------------------------------------------

/// Camera parameters used for screen-size culling.
#[derive(Debug, Clone, Copy)]
pub struct CullCamera {
    pub position: Vec3,
    pub forward: Vec3,
    /// Half-height of the projection at distance 1 (for perspective).
    pub projection_factor: f32,
    /// Screen height in pixels (used for screen-size culling).
    pub screen_height: f32,
    /// Whether this is an orthographic camera.
    pub is_orthographic: bool,
    /// Orthographic height (world units visible vertically) if ortho.
    pub ortho_height: f32,
}

impl CullCamera {
    pub fn perspective(position: Vec3, forward: Vec3, fov_y: f32, screen_height: f32) -> Self {
        Self {
            position,
            forward: forward.normalized(),
            projection_factor: 1.0 / (fov_y * 0.5).tan(),
            screen_height,
            is_orthographic: false,
            ortho_height: 0.0,
        }
    }

    pub fn orthographic(position: Vec3, forward: Vec3, ortho_height: f32, screen_height: f32) -> Self {
        Self {
            position,
            forward: forward.normalized(),
            projection_factor: 0.0,
            screen_height,
            is_orthographic: true,
            ortho_height,
        }
    }

    /// Estimate the screen-space area (in pixels^2) of a sphere.
    pub fn screen_size_sphere(&self, sphere: &BoundingSphere) -> f32 {
        if self.is_orthographic {
            let ratio = sphere.radius * 2.0 / self.ortho_height;
            let pixel_size = ratio * self.screen_height;
            pixel_size * pixel_size
        } else {
            let dist = (sphere.center - self.position).dot(self.forward);
            if dist <= 0.0 {
                return f32::MAX; // Behind camera, don't cull.
            }
            let projected_radius = sphere.radius * self.projection_factor / dist;
            let pixel_radius = projected_radius * self.screen_height * 0.5;
            std::f32::consts::PI * pixel_radius * pixel_radius
        }
    }
}

// ---------------------------------------------------------------------------
// FrustumCuller — the main culling system
// ---------------------------------------------------------------------------

/// Configuration for the frustum culling system.
#[derive(Debug, Clone)]
pub struct FrustumCullerConfig {
    /// Use temporal coherence (test last rejecting plane first).
    pub use_coherence: bool,
    /// Use bounding sphere pre-test for quick rejection.
    pub use_sphere_pretest: bool,
    /// Enable screen-size culling.
    pub use_screen_size_culling: bool,
    /// Minimum batch size for parallel culling.
    pub parallel_batch_size: usize,
    /// Layer mask to filter objects.
    pub active_layer_mask: u32,
}

impl Default for FrustumCullerConfig {
    fn default() -> Self {
        Self {
            use_coherence: true,
            use_sphere_pretest: true,
            use_screen_size_culling: true,
            parallel_batch_size: 256,
            active_layer_mask: 0xFFFF_FFFF,
        }
    }
}

/// Statistics from the last culling pass.
#[derive(Debug, Clone, Default)]
pub struct CullingStats {
    pub total_objects: u32,
    pub visible_objects: u32,
    pub culled_by_frustum: u32,
    pub culled_by_screen_size: u32,
    pub culled_by_layer: u32,
    pub always_visible: u32,
    pub coherence_hits: u32,
    pub sphere_pretest_culled: u32,
    pub elapsed_microseconds: u64,
}

impl CullingStats {
    pub fn cull_ratio(&self) -> f32 {
        if self.total_objects == 0 {
            return 0.0;
        }
        (self.total_objects - self.visible_objects) as f32 / self.total_objects as f32
    }
}

/// The main frustum culling system. Manages a set of cullable objects and
/// performs efficient frustum tests each frame.
pub struct FrustumCuller {
    objects: Vec<CullObject>,
    id_to_index: HashMap<u64, usize>,
    config: FrustumCullerConfig,
    stats: CullingStats,
    next_id: u64,
    generation: u64,
}

impl FrustumCuller {
    pub fn new() -> Self {
        Self::with_config(FrustumCullerConfig::default())
    }

    pub fn with_config(config: FrustumCullerConfig) -> Self {
        Self {
            objects: Vec::new(),
            id_to_index: HashMap::new(),
            config,
            stats: CullingStats::default(),
            next_id: 1,
            generation: 0,
        }
    }

    pub fn config(&self) -> &FrustumCullerConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut FrustumCullerConfig {
        &mut self.config
    }

    pub fn stats(&self) -> &CullingStats {
        &self.stats
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Allocate a new CullObjectId.
    pub fn allocate_id(&mut self) -> CullObjectId {
        let id = CullObjectId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add an object to the culling system.
    pub fn add_object(&mut self, obj: CullObject) {
        let idx = self.objects.len();
        self.id_to_index.insert(obj.id.0, idx);
        self.objects.push(obj);
    }

    /// Remove an object by ID. Uses swap-remove for O(1).
    pub fn remove_object(&mut self, id: CullObjectId) -> bool {
        if let Some(&idx) = self.id_to_index.get(&id.0) {
            self.id_to_index.remove(&id.0);
            let last = self.objects.len() - 1;
            if idx != last {
                let last_id = self.objects[last].id.0;
                self.objects.swap(idx, last);
                self.id_to_index.insert(last_id, idx);
            }
            self.objects.pop();
            true
        } else {
            false
        }
    }

    /// Update the AABB for an existing object.
    pub fn update_object_bounds(&mut self, id: CullObjectId, aabb: AABB) {
        if let Some(&idx) = self.id_to_index.get(&id.0) {
            self.objects[idx].update_bounds(aabb);
        }
    }

    /// Get a reference to an object by ID.
    pub fn get_object(&self, id: CullObjectId) -> Option<&CullObject> {
        self.id_to_index.get(&id.0).map(|&idx| &self.objects[idx])
    }

    /// Get a mutable reference to an object by ID.
    pub fn get_object_mut(&mut self, id: CullObjectId) -> Option<&mut CullObject> {
        if let Some(&idx) = self.id_to_index.get(&id.0) {
            Some(&mut self.objects[idx])
        } else {
            None
        }
    }

    /// Perform frustum culling and output the visible set.
    pub fn cull(
        &mut self,
        frustum: &Frustum,
        camera: Option<&CullCamera>,
        output: &mut VisibleSet,
    ) {
        let start = std::time::Instant::now();
        output.clear();
        self.generation += 1;

        let mut stats = CullingStats::default();
        stats.total_objects = self.objects.len() as u32;

        let layer_mask = self.config.active_layer_mask;
        let use_coherence = self.config.use_coherence;
        let use_sphere_pretest = self.config.use_sphere_pretest;
        let use_screen_size = self.config.use_screen_size_culling && camera.is_some();

        for obj in self.objects.iter_mut() {
            // Layer filtering
            if obj.layer_mask & layer_mask == 0 {
                stats.culled_by_layer += 1;
                obj.last_visible = false;
                continue;
            }

            // Always visible override
            if obj.always_visible {
                output.visible_ids.push(obj.id);
                output.results.push((obj.id, FrustumTestResult::Inside));
                stats.always_visible += 1;
                stats.visible_objects += 1;
                obj.last_visible = true;
                continue;
            }

            // Screen-size culling
            if use_screen_size {
                if let Some(cam) = camera {
                    if obj.min_screen_size > 0.0 {
                        let screen_area = cam.screen_size_sphere(&obj.bounding_sphere);
                        if screen_area < obj.min_screen_size {
                            stats.culled_by_screen_size += 1;
                            obj.last_visible = false;
                            continue;
                        }
                    }
                }
            }

            // Bounding sphere pre-test (quick rejection)
            if use_sphere_pretest {
                let sphere_result = frustum.test_sphere(&obj.bounding_sphere);
                if sphere_result == FrustumTestResult::Outside {
                    stats.sphere_pretest_culled += 1;
                    stats.culled_by_frustum += 1;
                    obj.last_visible = false;
                    continue;
                }
                // If fully inside by sphere, skip AABB test
                if sphere_result == FrustumTestResult::Inside {
                    output.visible_ids.push(obj.id);
                    output.results.push((obj.id, FrustumTestResult::Inside));
                    stats.visible_objects += 1;
                    obj.last_visible = true;
                    continue;
                }
            }

            // AABB test with coherence hint
            let result = if use_coherence {
                self.test_aabb_coherent(frustum, &obj.aabb, &mut obj.last_rejecting_plane)
            } else {
                frustum.test_aabb(&obj.aabb)
            };

            if use_coherence && result != FrustumTestResult::Outside && !obj.last_visible {
                stats.coherence_hits += 1;
            }

            match result {
                FrustumTestResult::Outside => {
                    stats.culled_by_frustum += 1;
                    obj.last_visible = false;
                }
                other => {
                    output.visible_ids.push(obj.id);
                    output.results.push((obj.id, other));
                    stats.visible_objects += 1;
                    obj.last_visible = true;
                }
            }
        }

        output.total_tested = stats.total_objects;
        output.total_visible = stats.visible_objects;
        output.total_culled = stats.total_objects - stats.visible_objects;

        let elapsed = start.elapsed();
        stats.elapsed_microseconds = elapsed.as_micros() as u64;
        output.cull_time_us = stats.elapsed_microseconds;

        self.stats = stats;
    }

    /// AABB test with temporal coherence: test the last rejecting plane first.
    fn test_aabb_coherent(
        &self,
        frustum: &Frustum,
        aabb: &AABB,
        last_plane: &mut u8,
    ) -> FrustumTestResult {
        let center = aabb.center();
        let extents = aabb.extents();
        let mut all_inside = true;

        // Test the last rejecting plane first for coherence
        let start = *last_plane as usize;
        for offset in 0..6 {
            let i = (start + offset) % 6;
            let plane = &frustum.planes[i];
            let abs_n = &frustum.abs_normals[i];

            let dist = plane.normal.x * center.x
                + plane.normal.y * center.y
                + plane.normal.z * center.z
                + plane.d;

            let radius = abs_n[0] * extents.x + abs_n[1] * extents.y + abs_n[2] * extents.z;

            if dist < -radius {
                *last_plane = i as u8;
                return FrustumTestResult::Outside;
            }
            if dist < radius {
                all_inside = false;
            }
        }

        if all_inside {
            FrustumTestResult::Inside
        } else {
            FrustumTestResult::Intersect
        }
    }

    /// Batch test multiple AABBs against a frustum, writing results into the provided slice.
    /// Returns the number of visible AABBs.
    pub fn batch_test_aabbs(
        frustum: &Frustum,
        aabbs: &[AABB],
        results: &mut [FrustumTestResult],
    ) -> usize {
        assert!(results.len() >= aabbs.len());
        let mut visible_count = 0;
        for (i, aabb) in aabbs.iter().enumerate() {
            let result = frustum.test_aabb(aabb);
            results[i] = result;
            if result != FrustumTestResult::Outside {
                visible_count += 1;
            }
        }
        visible_count
    }

    /// Batch test using SoA (struct-of-arrays) data layout for cache efficiency.
    /// `centers_x/y/z` and `extents_x/y/z` are parallel arrays.
    pub fn batch_test_soa(
        frustum: &Frustum,
        count: usize,
        centers_x: &[f32],
        centers_y: &[f32],
        centers_z: &[f32],
        extents_x: &[f32],
        extents_y: &[f32],
        extents_z: &[f32],
        visible_mask: &mut [bool],
    ) -> usize {
        assert!(centers_x.len() >= count);
        assert!(centers_y.len() >= count);
        assert!(centers_z.len() >= count);
        assert!(extents_x.len() >= count);
        assert!(extents_y.len() >= count);
        assert!(extents_z.len() >= count);
        assert!(visible_mask.len() >= count);

        let mut visible_count = 0;

        for i in 0..count {
            let cx = centers_x[i];
            let cy = centers_y[i];
            let cz = centers_z[i];
            let ex = extents_x[i];
            let ey = extents_y[i];
            let ez = extents_z[i];

            let mut culled = false;
            for p in 0..6 {
                let plane = &frustum.planes[p];
                let abs_n = &frustum.abs_normals[p];

                let dist = plane.normal.x * cx + plane.normal.y * cy + plane.normal.z * cz + plane.d;
                let radius = abs_n[0] * ex + abs_n[1] * ey + abs_n[2] * ez;

                if dist < -radius {
                    culled = true;
                    break;
                }
            }

            visible_mask[i] = !culled;
            if !culled {
                visible_count += 1;
            }
        }

        visible_count
    }

    /// Cull objects against multiple frustums (e.g. shadow cascades).
    /// Returns one VisibleSet per frustum.
    pub fn cull_multi_frustum(
        &mut self,
        frustums: &[Frustum],
        camera: Option<&CullCamera>,
    ) -> Vec<VisibleSet> {
        let mut results = Vec::with_capacity(frustums.len());
        for frustum in frustums {
            let mut vis = VisibleSet::with_capacity(self.objects.len());
            self.cull(frustum, camera, &mut vis);
            results.push(vis);
        }
        results
    }

    /// Clear all objects.
    pub fn clear(&mut self) {
        self.objects.clear();
        self.id_to_index.clear();
    }
}

impl Default for FrustumCuller {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Hierarchical culling with octree acceleration
// ---------------------------------------------------------------------------

/// An octree node for hierarchical frustum culling.
#[derive(Debug)]
struct OctreeNode {
    bounds: AABB,
    children: Option<Box<[OctreeNode; 8]>>,
    object_ids: Vec<CullObjectId>,
    depth: u32,
}

impl OctreeNode {
    fn new(bounds: AABB, depth: u32) -> Self {
        Self {
            bounds,
            children: None,
            object_ids: Vec::new(),
            depth,
        }
    }

    fn child_bounds(&self, index: usize) -> AABB {
        let center = self.bounds.center();
        let min = self.bounds.min;
        let max = self.bounds.max;
        let child_min = Vec3::new(
            if index & 1 == 0 { min.x } else { center.x },
            if index & 2 == 0 { min.y } else { center.y },
            if index & 4 == 0 { min.z } else { center.z },
        );
        let child_max = Vec3::new(
            if index & 1 == 0 { center.x } else { max.x },
            if index & 2 == 0 { center.y } else { max.y },
            if index & 4 == 0 { center.z } else { max.z },
        );
        AABB::new(child_min, child_max)
    }

    fn subdivide(&mut self) {
        if self.children.is_some() {
            return;
        }
        let d = self.depth + 1;
        self.children = Some(Box::new([
            OctreeNode::new(self.child_bounds(0), d),
            OctreeNode::new(self.child_bounds(1), d),
            OctreeNode::new(self.child_bounds(2), d),
            OctreeNode::new(self.child_bounds(3), d),
            OctreeNode::new(self.child_bounds(4), d),
            OctreeNode::new(self.child_bounds(5), d),
            OctreeNode::new(self.child_bounds(6), d),
            OctreeNode::new(self.child_bounds(7), d),
        ]));
    }

    fn insert(&mut self, id: CullObjectId, aabb: &AABB, max_depth: u32, max_objects: usize) {
        if self.depth >= max_depth || self.object_ids.len() < max_objects {
            self.object_ids.push(id);
            return;
        }

        if self.children.is_none() {
            self.subdivide();
        }

        if let Some(ref mut children) = self.children {
            let center = self.bounds.center();
            let aabb_center = aabb.center();
            let idx = ((if aabb_center.x >= center.x { 1 } else { 0 })
                | (if aabb_center.y >= center.y { 2 } else { 0 })
                | (if aabb_center.z >= center.z { 4 } else { 0 })) as usize;

            if children[idx].bounds.intersects(aabb) {
                children[idx].insert(id, aabb, max_depth, max_objects);
            } else {
                self.object_ids.push(id);
            }
        }
    }

    fn query_frustum(&self, frustum: &Frustum, output: &mut Vec<CullObjectId>) {
        let result = frustum.test_aabb(&self.bounds);
        match result {
            FrustumTestResult::Outside => return,
            FrustumTestResult::Inside => {
                self.collect_all(output);
                return;
            }
            FrustumTestResult::Intersect => {}
        }

        output.extend_from_slice(&self.object_ids);

        if let Some(ref children) = self.children {
            for child in children.iter() {
                child.query_frustum(frustum, output);
            }
        }
    }

    fn collect_all(&self, output: &mut Vec<CullObjectId>) {
        output.extend_from_slice(&self.object_ids);
        if let Some(ref children) = self.children {
            for child in children.iter() {
                child.collect_all(output);
            }
        }
    }

    fn total_objects(&self) -> usize {
        let mut count = self.object_ids.len();
        if let Some(ref children) = self.children {
            for child in children.iter() {
                count += child.total_objects();
            }
        }
        count
    }

    fn max_depth_used(&self) -> u32 {
        let mut d = self.depth;
        if let Some(ref children) = self.children {
            for child in children.iter() {
                d = d.max(child.max_depth_used());
            }
        }
        d
    }
}

/// Hierarchical frustum culler using an octree for large scenes.
pub struct HierarchicalCuller {
    root: OctreeNode,
    max_depth: u32,
    max_objects_per_node: usize,
    object_aabbs: HashMap<u64, AABB>,
}

impl HierarchicalCuller {
    pub fn new(world_bounds: AABB, max_depth: u32, max_objects_per_node: usize) -> Self {
        Self {
            root: OctreeNode::new(world_bounds, 0),
            max_depth,
            max_objects_per_node,
            object_aabbs: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: CullObjectId, aabb: AABB) {
        self.object_aabbs.insert(id.0, aabb);
        self.root.insert(id, &aabb, self.max_depth, self.max_objects_per_node);
    }

    /// Rebuild the octree from scratch (call after many insertions/removals).
    pub fn rebuild(&mut self, world_bounds: AABB) {
        self.root = OctreeNode::new(world_bounds, 0);
        let entries: Vec<_> = self.object_aabbs.iter().map(|(&k, &v)| (k, v)).collect();
        for (id, aabb) in entries {
            self.root.insert(
                CullObjectId(id),
                &aabb,
                self.max_depth,
                self.max_objects_per_node,
            );
        }
    }

    /// Query visible objects against a frustum.
    pub fn query(&self, frustum: &Frustum) -> Vec<CullObjectId> {
        let mut output = Vec::with_capacity(self.object_aabbs.len() / 2);
        self.root.query_frustum(frustum, &mut output);
        output
    }

    pub fn total_objects(&self) -> usize {
        self.root.total_objects()
    }

    pub fn max_depth_used(&self) -> u32 {
        self.root.max_depth_used()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vp() -> Mat4 {
        let view = Mat4::look_at(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let proj = Mat4::perspective(
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            100.0,
        );
        proj.mul_mat4(&view)
    }

    #[test]
    fn test_aabb_center_extents() {
        let aabb = AABB::new(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        let c = aabb.center();
        assert!((c.x).abs() < 1e-6);
        assert!((c.y).abs() < 1e-6);
        assert!((c.z).abs() < 1e-6);
        let e = aabb.extents();
        assert!((e.x - 1.0).abs() < 1e-6);
        assert!((e.y - 2.0).abs() < 1e-6);
        assert!((e.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_aabb_at_origin_visible() {
        let vp = make_test_vp();
        let frustum = Frustum::from_view_projection(&vp);
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(frustum.is_aabb_visible(&aabb));
    }

    #[test]
    fn test_aabb_behind_camera_culled() {
        let vp = make_test_vp();
        let frustum = Frustum::from_view_projection(&vp);
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, 10.0), Vec3::new(1.0, 1.0, 12.0));
        assert!(!frustum.is_aabb_visible(&aabb));
    }

    #[test]
    fn test_aabb_far_left_culled() {
        let vp = make_test_vp();
        let frustum = Frustum::from_view_projection(&vp);
        let aabb = AABB::new(Vec3::new(-100.0, -1.0, -1.0), Vec3::new(-90.0, 1.0, 1.0));
        assert!(!frustum.is_aabb_visible(&aabb));
    }

    #[test]
    fn test_sphere_visible() {
        let vp = make_test_vp();
        let frustum = Frustum::from_view_projection(&vp);
        let sphere = BoundingSphere::new(Vec3::ZERO, 1.0);
        assert_ne!(frustum.test_sphere(&sphere), FrustumTestResult::Outside);
    }

    #[test]
    fn test_point_inside() {
        let vp = make_test_vp();
        let frustum = Frustum::from_view_projection(&vp);
        assert!(frustum.test_point(Vec3::ZERO));
    }

    #[test]
    fn test_frustum_culler_basic() {
        let vp = make_test_vp();
        let frustum = Frustum::from_view_projection(&vp);

        let mut culler = FrustumCuller::new();
        let id1 = culler.allocate_id();
        let id2 = culler.allocate_id();
        let id3 = culler.allocate_id();

        culler.add_object(CullObject::new(
            id1,
            AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
        ));
        culler.add_object(CullObject::new(
            id2,
            AABB::new(Vec3::new(-100.0, -1.0, -1.0), Vec3::new(-90.0, 1.0, 1.0)),
        ));
        culler.add_object(CullObject::new(
            id3,
            AABB::new(Vec3::new(0.0, 0.0, -50.0), Vec3::new(1.0, 1.0, -49.0)),
        ));

        let mut visible = VisibleSet::new();
        culler.cull(&frustum, None, &mut visible);

        assert!(visible.is_visible(id1));
        assert!(!visible.is_visible(id2));
        assert!(visible.is_visible(id3));
        assert_eq!(visible.total_visible, 2);
    }

    #[test]
    fn test_batch_soa() {
        let vp = make_test_vp();
        let frustum = Frustum::from_view_projection(&vp);

        let cx = vec![0.0f32, -95.0, 0.5];
        let cy = vec![0.0f32, 0.0, 0.0];
        let cz = vec![0.0f32, 0.0, -20.0];
        let ex = vec![1.0f32, 5.0, 1.0];
        let ey = vec![1.0f32, 1.0, 1.0];
        let ez = vec![1.0f32, 1.0, 1.0];
        let mut mask = vec![false; 3];

        let vis = FrustumCuller::batch_test_soa(
            &frustum, 3, &cx, &cy, &cz, &ex, &ey, &ez, &mut mask,
        );
        assert!(mask[0]);
        assert!(!mask[1]);
        assert!(mask[2]);
        assert_eq!(vis, 2);
    }

    #[test]
    fn test_hierarchical_culler() {
        let vp = make_test_vp();
        let frustum = Frustum::from_view_projection(&vp);

        let world = AABB::new(Vec3::new(-500.0, -500.0, -500.0), Vec3::new(500.0, 500.0, 500.0));
        let mut hc = HierarchicalCuller::new(world, 6, 16);

        let id1 = CullObjectId(1);
        let id2 = CullObjectId(2);

        hc.insert(id1, AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)));
        hc.insert(id2, AABB::new(Vec3::new(-200.0, -1.0, -1.0), Vec3::new(-190.0, 1.0, 1.0)));

        let visible = hc.query(&frustum);
        assert!(visible.contains(&id1));
        assert!(!visible.contains(&id2));
    }

    #[test]
    fn test_remove_object() {
        let mut culler = FrustumCuller::new();
        let id1 = culler.allocate_id();
        let id2 = culler.allocate_id();

        culler.add_object(CullObject::new(
            id1,
            AABB::new(Vec3::ZERO, Vec3::ONE),
        ));
        culler.add_object(CullObject::new(
            id2,
            AABB::new(Vec3::ZERO, Vec3::ONE),
        ));

        assert_eq!(culler.object_count(), 2);
        culler.remove_object(id1);
        assert_eq!(culler.object_count(), 1);
        assert!(culler.get_object(id2).is_some());
    }

    #[test]
    fn test_always_visible() {
        let vp = make_test_vp();
        let frustum = Frustum::from_view_projection(&vp);

        let mut culler = FrustumCuller::new();
        let id = culler.allocate_id();
        culler.add_object(
            CullObject::new(
                id,
                AABB::new(Vec3::new(-200.0, -200.0, -200.0), Vec3::new(-190.0, -190.0, -190.0)),
            )
            .with_always_visible(true),
        );

        let mut vis = VisibleSet::new();
        culler.cull(&frustum, None, &mut vis);
        assert!(vis.is_visible(id));
    }
}
