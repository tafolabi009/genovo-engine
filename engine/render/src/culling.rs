// engine/render/src/culling.rs
//
// Comprehensive visibility and culling system for the Genovo engine.
//
// Provides frustum culling, distance culling, small-object culling,
// layer masking, shadow caster culling, octree-based spatial acceleration,
// a precomputed potentially-visible-set (PVS) system for indoor
// environments, and batched SIMD frustum testing.
//
// # Architecture
//
// The `VisibilitySystem` is the main entry point. It accepts a list of
// `CullObject`s (each with AABB, bounding sphere, layer, LOD ranges, etc.)
// and a `CullCamera` (frustum, position, max render distance, layer mask).
// It produces a `CullResult` per object indicating visibility, shadow
// visibility, and LOD level.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Math types (self-contained, no glam dependency in this module)
// ---------------------------------------------------------------------------

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn distance_sq(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        self.distance_sq(other).sqrt()
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
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
    /// Create from min and max corners.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create from center and half-extents.
    pub fn from_center_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center.sub(half_extents),
            max: center.add(half_extents),
        }
    }

    /// Center of the AABB.
    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    /// Half-extents of the AABB.
    pub fn half_extents(&self) -> Vec3 {
        Vec3::new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        )
    }

    /// Size of the AABB (width, height, depth).
    pub fn size(&self) -> Vec3 {
        self.max.sub(self.min)
    }

    /// Maximum extent (longest axis).
    pub fn max_extent(&self) -> f32 {
        let s = self.size();
        s.x.max(s.y).max(s.z)
    }

    /// Bounding sphere radius.
    pub fn bounding_radius(&self) -> f32 {
        self.half_extents().length()
    }

    /// Whether this AABB contains a point.
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Whether this AABB overlaps another.
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Expand this AABB to include a point.
    pub fn expand_to_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    /// Merge another AABB into this one.
    pub fn merge(&mut self, other: &AABB) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Return the merged AABB of two AABBs.
    pub fn merged(a: &AABB, b: &AABB) -> AABB {
        AABB {
            min: a.min.min(b.min),
            max: a.max.max(b.max),
        }
    }

    /// Volume of the AABB.
    pub fn volume(&self) -> f32 {
        let s = self.size();
        s.x * s.y * s.z
    }

    /// Surface area of the AABB.
    pub fn surface_area(&self) -> f32 {
        let s = self.size();
        2.0 * (s.x * s.y + s.y * s.z + s.z * s.x)
    }
}

impl Default for AABB {
    fn default() -> Self {
        Self {
            min: Vec3::new(f32::MAX, f32::MAX, f32::MAX),
            max: Vec3::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }
}

// ---------------------------------------------------------------------------
// Frustum plane and frustum
// ---------------------------------------------------------------------------

/// A plane in Hessian normal form: normal . x + d = 0.
/// Normal points inward (into the frustum).
#[derive(Debug, Clone, Copy)]
pub struct CullPlane {
    /// Inward normal (x, y, z).
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    /// Signed distance from origin.
    pub d: f32,
}

impl CullPlane {
    /// Create from ABCD coefficients and normalise.
    pub fn from_abcd(a: f32, b: f32, c: f32, d: f32) -> Self {
        let len = (a * a + b * b + c * c).sqrt();
        if len < 1e-7 {
            return Self {
                nx: 0.0,
                ny: 0.0,
                nz: 0.0,
                d: 0.0,
            };
        }
        let inv = 1.0 / len;
        Self {
            nx: a * inv,
            ny: b * inv,
            nz: c * inv,
            d: d * inv,
        }
    }

    /// Signed distance from a point to this plane.
    #[inline]
    pub fn signed_distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.nx * x + self.ny * y + self.nz * z + self.d
    }

    /// Signed distance from an AABB's "positive vertex" (the vertex farthest
    /// along the plane normal). If this is negative, the AABB is entirely
    /// outside the plane.
    #[inline]
    pub fn aabb_positive_dist(&self, aabb: &AABB) -> f32 {
        let px = if self.nx >= 0.0 { aabb.max.x } else { aabb.min.x };
        let py = if self.ny >= 0.0 { aabb.max.y } else { aabb.min.y };
        let pz = if self.nz >= 0.0 { aabb.max.z } else { aabb.min.z };
        self.signed_distance(px, py, pz)
    }

    /// Signed distance from an AABB's "negative vertex" (the vertex closest
    /// along the plane normal). If this is positive, the AABB is entirely
    /// inside the plane.
    #[inline]
    pub fn aabb_negative_dist(&self, aabb: &AABB) -> f32 {
        let nx = if self.nx >= 0.0 { aabb.min.x } else { aabb.max.x };
        let ny = if self.ny >= 0.0 { aabb.min.y } else { aabb.max.y };
        let nz = if self.nz >= 0.0 { aabb.min.z } else { aabb.max.z };
        self.signed_distance(nx, ny, nz)
    }
}

/// A six-plane frustum for culling.
///
/// Plane order: Left, Right, Bottom, Top, Near, Far.
#[derive(Debug, Clone)]
pub struct CullFrustum {
    pub planes: [CullPlane; 6],
}

impl CullFrustum {
    /// Extract frustum planes from a column-major 4x4 view-projection matrix
    /// stored as 16 floats.
    pub fn from_view_projection(m: &[f32; 16]) -> Self {
        // Column-major indexing: m[col*4+row].
        // Row 0: m[0], m[4], m[8],  m[12]
        // Row 1: m[1], m[5], m[9],  m[13]
        // Row 2: m[2], m[6], m[10], m[14]
        // Row 3: m[3], m[7], m[11], m[15]

        let r0 = [m[0], m[4], m[8], m[12]];
        let r1 = [m[1], m[5], m[9], m[13]];
        let r2 = [m[2], m[6], m[10], m[14]];
        let r3 = [m[3], m[7], m[11], m[15]];

        // Left:   row3 + row0
        let left = CullPlane::from_abcd(
            r3[0] + r0[0],
            r3[1] + r0[1],
            r3[2] + r0[2],
            r3[3] + r0[3],
        );
        // Right:  row3 - row0
        let right = CullPlane::from_abcd(
            r3[0] - r0[0],
            r3[1] - r0[1],
            r3[2] - r0[2],
            r3[3] - r0[3],
        );
        // Bottom: row3 + row1
        let bottom = CullPlane::from_abcd(
            r3[0] + r1[0],
            r3[1] + r1[1],
            r3[2] + r1[2],
            r3[3] + r1[3],
        );
        // Top:    row3 - row1
        let top = CullPlane::from_abcd(
            r3[0] - r1[0],
            r3[1] - r1[1],
            r3[2] - r1[2],
            r3[3] - r1[3],
        );
        // Near:   row3 + row2
        let near = CullPlane::from_abcd(
            r3[0] + r2[0],
            r3[1] + r2[1],
            r3[2] + r2[2],
            r3[3] + r2[3],
        );
        // Far:    row3 - row2
        let far = CullPlane::from_abcd(
            r3[0] - r2[0],
            r3[1] - r2[1],
            r3[2] - r2[2],
            r3[3] - r2[3],
        );

        Self {
            planes: [left, right, bottom, top, near, far],
        }
    }

    /// Test if an AABB is inside or intersects the frustum.
    #[inline]
    pub fn test_aabb(&self, aabb: &AABB) -> FrustumTestResult {
        let mut all_inside = true;
        for plane in &self.planes {
            let p_dist = plane.aabb_positive_dist(aabb);
            if p_dist < 0.0 {
                return FrustumTestResult::Outside;
            }
            let n_dist = plane.aabb_negative_dist(aabb);
            if n_dist < 0.0 {
                all_inside = false;
            }
        }
        if all_inside {
            FrustumTestResult::Inside
        } else {
            FrustumTestResult::Intersect
        }
    }

    /// Quick test: is the AABB at least partially inside?
    #[inline]
    pub fn is_visible(&self, aabb: &AABB) -> bool {
        for plane in &self.planes {
            if plane.aabb_positive_dist(aabb) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Test a bounding sphere.
    #[inline]
    pub fn test_sphere(&self, center: Vec3, radius: f32) -> FrustumTestResult {
        let mut all_inside = true;
        for plane in &self.planes {
            let dist = plane.signed_distance(center.x, center.y, center.z);
            if dist < -radius {
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
}

/// Result of a frustum test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrustumTestResult {
    /// Completely outside the frustum.
    Outside,
    /// Partially inside (straddling a plane).
    Intersect,
    /// Completely inside the frustum.
    Inside,
}

// ---------------------------------------------------------------------------
// SIMD batched frustum testing
// ---------------------------------------------------------------------------

/// SOA (Structure of Arrays) layout for 4 AABBs, suitable for SIMD testing.
/// Even on platforms without hardware SIMD intrinsics, this layout enables
/// the compiler to auto-vectorize the inner loops.
#[derive(Debug, Clone)]
pub struct SimdAabbBatch {
    /// Min X for 4 AABBs.
    pub min_x: [f32; 4],
    /// Min Y for 4 AABBs.
    pub min_y: [f32; 4],
    /// Min Z for 4 AABBs.
    pub min_z: [f32; 4],
    /// Max X for 4 AABBs.
    pub max_x: [f32; 4],
    /// Max Y for 4 AABBs.
    pub max_y: [f32; 4],
    /// Max Z for 4 AABBs.
    pub max_z: [f32; 4],
}

impl SimdAabbBatch {
    /// Create a batch from 4 AABBs. If fewer than 4 are provided, the
    /// remaining slots are filled with degenerate AABBs that are always
    /// outside any frustum.
    pub fn from_aabbs(aabbs: &[AABB]) -> Self {
        let degenerate = AABB::new(
            Vec3::new(f32::MAX, f32::MAX, f32::MAX),
            Vec3::new(f32::MIN, f32::MIN, f32::MIN),
        );
        let get = |i: usize| aabbs.get(i).unwrap_or(&degenerate);

        Self {
            min_x: [get(0).min.x, get(1).min.x, get(2).min.x, get(3).min.x],
            min_y: [get(0).min.y, get(1).min.y, get(2).min.y, get(3).min.y],
            min_z: [get(0).min.z, get(1).min.z, get(2).min.z, get(3).min.z],
            max_x: [get(0).max.x, get(1).max.x, get(2).max.x, get(3).max.x],
            max_y: [get(0).max.y, get(1).max.y, get(2).max.y, get(3).max.y],
            max_z: [get(0).max.z, get(1).max.z, get(2).max.z, get(3).max.z],
        }
    }

    /// Create a batch from a slice of AABBs at a given offset.
    pub fn from_slice(aabbs: &[AABB], offset: usize) -> Self {
        Self::from_aabbs(&aabbs[offset..aabbs.len().min(offset + 4)])
    }
}

/// Test 4 AABBs against a frustum in a SIMD-friendly manner.
/// Returns a bitmask: bit i is set if AABB i is at least partially visible.
///
/// This function is written to allow the Rust compiler to auto-vectorize
/// the inner loop using SSE/AVX/NEON depending on the target.
#[inline]
pub fn frustum_test_batch_4(frustum: &CullFrustum, batch: &SimdAabbBatch) -> u8 {
    let mut result_mask: u8 = 0b1111; // assume all visible

    for plane in &frustum.planes {
        // For each AABB, compute the positive vertex distance.
        let mut visible = [true; 4];

        // The positive vertex for each AABB: choose max if normal >= 0, else min.
        // This loop processes 4 AABBs and should auto-vectorize.
        for i in 0..4 {
            let px = if plane.nx >= 0.0 {
                batch.max_x[i]
            } else {
                batch.min_x[i]
            };
            let py = if plane.ny >= 0.0 {
                batch.max_y[i]
            } else {
                batch.min_y[i]
            };
            let pz = if plane.nz >= 0.0 {
                batch.max_z[i]
            } else {
                batch.min_z[i]
            };
            let dist = plane.nx * px + plane.ny * py + plane.nz * pz + plane.d;
            visible[i] = dist >= 0.0;
        }

        // Clear bits for AABBs outside this plane.
        for i in 0..4 {
            if !visible[i] {
                result_mask &= !(1 << i);
            }
        }

        // Early out if nothing visible.
        if result_mask == 0 {
            return 0;
        }
    }

    result_mask
}

/// Test 4 AABBs against a frustum using explicit SIMD-style f32x4 operations.
/// The results are identical to `frustum_test_batch_4` but uses a layout
/// that more explicitly hints vectorization.
#[inline]
pub fn frustum_test_batch_4_explicit(
    frustum: &CullFrustum,
    batch: &SimdAabbBatch,
) -> [bool; 4] {
    let mut visible = [true; 4];

    for plane in &frustum.planes {
        // Compute positive vertex x, y, z for all 4 AABBs.
        let mut px = [0.0f32; 4];
        let mut py = [0.0f32; 4];
        let mut pz = [0.0f32; 4];

        // Select positive vertex components.
        if plane.nx >= 0.0 {
            px = batch.max_x;
        } else {
            px = batch.min_x;
        }
        if plane.ny >= 0.0 {
            py = batch.max_y;
        } else {
            py = batch.min_y;
        }
        if plane.nz >= 0.0 {
            pz = batch.max_z;
        } else {
            pz = batch.min_z;
        }

        // Compute dot product + d for all 4.
        let mut dist = [0.0f32; 4];
        for i in 0..4 {
            dist[i] = plane.nx * px[i] + plane.ny * py[i] + plane.nz * pz[i] + plane.d;
        }

        // Mark outside.
        for i in 0..4 {
            if dist[i] < 0.0 {
                visible[i] = false;
            }
        }
    }

    visible
}

/// Batch test many AABBs against a frustum. Returns a Vec<bool> indicating
/// visibility for each input AABB.
pub fn frustum_test_batch(frustum: &CullFrustum, aabbs: &[AABB]) -> Vec<bool> {
    let n = aabbs.len();
    let mut results = vec![false; n];
    let batches = (n + 3) / 4;

    for batch_idx in 0..batches {
        let offset = batch_idx * 4;
        let batch = SimdAabbBatch::from_slice(aabbs, offset);
        let mask = frustum_test_batch_4(frustum, &batch);

        for i in 0..4 {
            let global_idx = offset + i;
            if global_idx < n {
                results[global_idx] = (mask & (1 << i)) != 0;
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Cull object and camera
// ---------------------------------------------------------------------------

/// Layer mask for selective rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerMask(pub u32);

impl LayerMask {
    pub const ALL: Self = Self(u32::MAX);
    pub const NONE: Self = Self(0);
    pub const DEFAULT: Self = Self(1);

    /// Check if a specific layer bit is set.
    #[inline]
    pub fn contains(self, layer: u32) -> bool {
        (self.0 & (1 << layer)) != 0
    }

    /// Check if this mask overlaps with another.
    #[inline]
    pub fn overlaps(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

/// LOD distance ranges.
#[derive(Debug, Clone)]
pub struct LodRanges {
    /// Distance thresholds for LOD transitions. `lod_distances[i]` is the
    /// maximum distance for LOD level `i`. There are `n` entries for `n+1`
    /// LOD levels (the last LOD extends to infinity / max render distance).
    pub distances: Vec<f32>,
}

impl LodRanges {
    /// Create LOD ranges from a set of distances.
    pub fn new(distances: Vec<f32>) -> Self {
        Self { distances }
    }

    /// Determine the LOD level for a given distance.
    pub fn lod_for_distance(&self, distance: f32) -> u32 {
        for (i, &threshold) in self.distances.iter().enumerate() {
            if distance <= threshold {
                return i as u32;
            }
        }
        self.distances.len() as u32
    }
}

impl Default for LodRanges {
    fn default() -> Self {
        Self {
            distances: vec![50.0, 100.0, 200.0],
        }
    }
}

/// An object to be tested for visibility.
#[derive(Debug, Clone)]
pub struct CullObject {
    /// Unique identifier.
    pub id: u64,
    /// World-space AABB.
    pub aabb: AABB,
    /// Bounding sphere center (world-space).
    pub sphere_center: Vec3,
    /// Bounding sphere radius.
    pub sphere_radius: f32,
    /// Layer this object belongs to.
    pub layer: LayerMask,
    /// LOD distance ranges.
    pub lod_ranges: LodRanges,
    /// Maximum render distance override (0 = use camera default).
    pub max_render_distance: f32,
    /// Minimum screen-space size in pixels (0 = no small-object culling).
    pub min_screen_pixels: f32,
    /// Whether this object casts shadows.
    pub casts_shadow: bool,
    /// Whether this object is always visible (never culled).
    pub always_visible: bool,
}

impl CullObject {
    /// Create a simple cull object with default settings.
    pub fn simple(id: u64, aabb: AABB) -> Self {
        let center = aabb.center();
        let radius = aabb.bounding_radius();
        Self {
            id,
            aabb,
            sphere_center: center,
            sphere_radius: radius,
            layer: LayerMask::DEFAULT,
            lod_ranges: LodRanges::default(),
            max_render_distance: 0.0,
            min_screen_pixels: 0.0,
            casts_shadow: true,
            always_visible: false,
        }
    }
}

/// Camera configuration for culling.
#[derive(Debug, Clone)]
pub struct CullCamera {
    /// View-projection matrix (column-major).
    pub view_projection: [f32; 16],
    /// Camera world position.
    pub position: Vec3,
    /// Maximum render distance.
    pub max_render_distance: f32,
    /// Layer mask: only objects on these layers are visible.
    pub layer_mask: LayerMask,
    /// Screen width in pixels (for small-object culling).
    pub screen_width: f32,
    /// Screen height in pixels.
    pub screen_height: f32,
    /// Vertical field of view in radians (for screen-space size computation).
    pub fov_y: f32,
}

impl CullCamera {
    /// Compute the approximate screen-space size (in pixels) of an object
    /// at the given distance with the given world-space radius.
    pub fn screen_size_pixels(&self, distance: f32, radius: f32) -> f32 {
        if distance <= 0.0 {
            return self.screen_height;
        }
        let projected_size = radius / (distance * (self.fov_y * 0.5).tan());
        projected_size * self.screen_height * 0.5
    }
}

// ---------------------------------------------------------------------------
// Cull result
// ---------------------------------------------------------------------------

/// Culling result for a single object.
#[derive(Debug, Clone, Copy)]
pub struct CullResult {
    /// Whether the object is visible to the camera.
    pub visible: bool,
    /// Whether the object is visible for shadow casting.
    pub shadow_visible: bool,
    /// LOD level (0 = highest detail).
    pub lod_level: u32,
    /// Distance from the camera.
    pub distance: f32,
    /// Reason the object was culled (if not visible).
    pub cull_reason: CullReason,
}

/// Reason an object was culled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullReason {
    /// Not culled (visible).
    Visible,
    /// Outside the view frustum.
    Frustum,
    /// Beyond the maximum render distance.
    Distance,
    /// Too small on screen.
    SmallObject,
    /// Layer mismatch.
    LayerMask,
}

impl fmt::Display for CullReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Visible => write!(f, "visible"),
            Self::Frustum => write!(f, "frustum-culled"),
            Self::Distance => write!(f, "distance-culled"),
            Self::SmallObject => write!(f, "small-object-culled"),
            Self::LayerMask => write!(f, "layer-culled"),
        }
    }
}

// ---------------------------------------------------------------------------
// VisibilitySystem
// ---------------------------------------------------------------------------

/// Statistics from a visibility pass.
#[derive(Debug, Clone, Default)]
pub struct CullStats {
    pub total_objects: u32,
    pub visible_objects: u32,
    pub frustum_culled: u32,
    pub distance_culled: u32,
    pub small_object_culled: u32,
    pub layer_culled: u32,
    pub shadow_visible: u32,
}

impl fmt::Display for CullStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cull Stats: {} total, {} visible", self.total_objects, self.visible_objects)?;
        writeln!(f, "  Frustum: {} culled", self.frustum_culled)?;
        writeln!(f, "  Distance: {} culled", self.distance_culled)?;
        writeln!(f, "  Small object: {} culled", self.small_object_culled)?;
        writeln!(f, "  Layer: {} culled", self.layer_culled)?;
        writeln!(f, "  Shadow visible: {}", self.shadow_visible)?;
        Ok(())
    }
}

/// Configuration for the visibility system.
#[derive(Debug, Clone)]
pub struct CullConfig {
    /// Whether to use batched SIMD frustum testing.
    pub use_simd_batching: bool,
    /// Whether to perform distance culling.
    pub enable_distance_culling: bool,
    /// Whether to perform small-object culling.
    pub enable_small_object_culling: bool,
    /// Whether to perform layer-mask culling.
    pub enable_layer_culling: bool,
    /// Whether to compute shadow visibility.
    pub compute_shadow_visibility: bool,
}

impl Default for CullConfig {
    fn default() -> Self {
        Self {
            use_simd_batching: true,
            enable_distance_culling: true,
            enable_small_object_culling: true,
            enable_layer_culling: true,
            compute_shadow_visibility: true,
        }
    }
}

/// The main visibility system.
pub struct VisibilitySystem {
    config: CullConfig,
}

impl VisibilitySystem {
    /// Create a new visibility system with default config.
    pub fn new() -> Self {
        Self {
            config: CullConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: CullConfig) -> Self {
        Self { config }
    }

    /// Cull a set of objects against a camera.
    pub fn cull(
        &self,
        objects: &[CullObject],
        camera: &CullCamera,
        shadow_frustum: Option<&CullFrustum>,
    ) -> (Vec<CullResult>, CullStats) {
        let frustum = CullFrustum::from_view_projection(&camera.view_projection);
        let mut results = Vec::with_capacity(objects.len());
        let mut stats = CullStats {
            total_objects: objects.len() as u32,
            ..Default::default()
        };

        if self.config.use_simd_batching {
            // Phase 1: batched frustum culling.
            let aabbs: Vec<AABB> = objects.iter().map(|o| o.aabb).collect();
            let frustum_visible = frustum_test_batch(&frustum, &aabbs);

            // Phase 2: per-object filtering.
            for (i, obj) in objects.iter().enumerate() {
                let result = self.cull_single(
                    obj,
                    camera,
                    frustum_visible[i],
                    shadow_frustum,
                );
                self.update_stats(&result, &mut stats);
                results.push(result);
            }
        } else {
            // Scalar path.
            for obj in objects {
                let frustum_vis = frustum.is_visible(&obj.aabb);
                let result = self.cull_single(obj, camera, frustum_vis, shadow_frustum);
                self.update_stats(&result, &mut stats);
                results.push(result);
            }
        }

        (results, stats)
    }

    /// Cull a single object.
    fn cull_single(
        &self,
        obj: &CullObject,
        camera: &CullCamera,
        frustum_visible: bool,
        shadow_frustum: Option<&CullFrustum>,
    ) -> CullResult {
        let distance = obj.sphere_center.distance(camera.position);

        // Always-visible objects bypass all culling.
        if obj.always_visible {
            return CullResult {
                visible: true,
                shadow_visible: obj.casts_shadow,
                lod_level: obj.lod_ranges.lod_for_distance(distance),
                distance,
                cull_reason: CullReason::Visible,
            };
        }

        // Layer mask check.
        if self.config.enable_layer_culling && !camera.layer_mask.overlaps(obj.layer) {
            return CullResult {
                visible: false,
                shadow_visible: false,
                lod_level: 0,
                distance,
                cull_reason: CullReason::LayerMask,
            };
        }

        // Frustum check.
        if !frustum_visible {
            // Check shadow visibility even if main-camera frustum fails.
            let shadow_vis = if self.config.compute_shadow_visibility && obj.casts_shadow {
                shadow_frustum
                    .map(|sf| sf.is_visible(&obj.aabb))
                    .unwrap_or(false)
            } else {
                false
            };
            return CullResult {
                visible: false,
                shadow_visible: shadow_vis,
                lod_level: obj.lod_ranges.lod_for_distance(distance),
                distance,
                cull_reason: CullReason::Frustum,
            };
        }

        // Distance culling.
        if self.config.enable_distance_culling {
            let max_dist = if obj.max_render_distance > 0.0 {
                obj.max_render_distance
            } else {
                camera.max_render_distance
            };
            if distance > max_dist {
                return CullResult {
                    visible: false,
                    shadow_visible: false,
                    lod_level: obj.lod_ranges.lod_for_distance(distance),
                    distance,
                    cull_reason: CullReason::Distance,
                };
            }
        }

        // Small object culling.
        if self.config.enable_small_object_culling && obj.min_screen_pixels > 0.0 {
            let screen_size = camera.screen_size_pixels(distance, obj.sphere_radius);
            if screen_size < obj.min_screen_pixels {
                return CullResult {
                    visible: false,
                    shadow_visible: false,
                    lod_level: obj.lod_ranges.lod_for_distance(distance),
                    distance,
                    cull_reason: CullReason::SmallObject,
                };
            }
        }

        // Object is visible.
        let lod = obj.lod_ranges.lod_for_distance(distance);
        let shadow_vis = obj.casts_shadow;

        CullResult {
            visible: true,
            shadow_visible: shadow_vis,
            lod_level: lod,
            distance,
            cull_reason: CullReason::Visible,
        }
    }

    fn update_stats(&self, result: &CullResult, stats: &mut CullStats) {
        if result.visible {
            stats.visible_objects += 1;
        } else {
            match result.cull_reason {
                CullReason::Frustum => stats.frustum_culled += 1,
                CullReason::Distance => stats.distance_culled += 1,
                CullReason::SmallObject => stats.small_object_culled += 1,
                CullReason::LayerMask => stats.layer_culled += 1,
                CullReason::Visible => {} // shouldn't happen
            }
        }
        if result.shadow_visible {
            stats.shadow_visible += 1;
        }
    }
}

impl Default for VisibilitySystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shadow caster culling
// ---------------------------------------------------------------------------

/// Cull shadow casters for a directional light.
///
/// Given a list of objects and the light's shadow frustum (e.g. the cascade
/// frustum for cascaded shadow maps), returns a bitmask or list of objects
/// that should be rendered into the shadow map.
pub struct ShadowCasterCuller;

impl ShadowCasterCuller {
    /// Cull shadow casters against a light frustum.
    pub fn cull(
        objects: &[CullObject],
        shadow_frustum: &CullFrustum,
        _light_direction: Vec3,
    ) -> Vec<bool> {
        let aabbs: Vec<AABB> = objects
            .iter()
            .map(|o| o.aabb)
            .collect();
        let frustum_results = frustum_test_batch(shadow_frustum, &aabbs);

        objects
            .iter()
            .zip(frustum_results.iter())
            .map(|(obj, &in_frustum)| obj.casts_shadow && in_frustum)
            .collect()
    }

    /// Cull shadow casters for a specific cascade of a directional light.
    pub fn cull_cascade(
        objects: &[CullObject],
        cascade_frustum: &CullFrustum,
        _light_direction: Vec3,
        max_shadow_distance: f32,
        camera_pos: Vec3,
    ) -> Vec<bool> {
        let aabbs: Vec<AABB> = objects.iter().map(|o| o.aabb).collect();
        let frustum_results = frustum_test_batch(cascade_frustum, &aabbs);

        objects
            .iter()
            .zip(frustum_results.iter())
            .map(|(obj, &in_frustum)| {
                if !obj.casts_shadow || !in_frustum {
                    return false;
                }
                let dist = obj.sphere_center.distance(camera_pos);
                dist <= max_shadow_distance
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Octree
// ---------------------------------------------------------------------------

/// Maximum depth of the octree.
const OCTREE_MAX_DEPTH: u32 = 8;
/// Maximum objects per leaf before splitting.
const OCTREE_MAX_OBJECTS_PER_LEAF: usize = 16;

/// An octree node (either leaf or internal).
#[derive(Debug)]
enum OctreeNode {
    Leaf {
        bounds: AABB,
        objects: Vec<u64>, // object IDs
    },
    Internal {
        bounds: AABB,
        children: Box<[Option<OctreeNode>; 8]>,
    },
}

impl OctreeNode {
    fn bounds(&self) -> &AABB {
        match self {
            Self::Leaf { bounds, .. } => bounds,
            Self::Internal { bounds, .. } => bounds,
        }
    }
}

/// Spatial acceleration structure for culling large worlds.
pub struct OctreeCulling {
    root: Option<OctreeNode>,
    /// Map from object ID to AABB for fast updates.
    object_bounds: HashMap<u64, AABB>,
    /// World bounds.
    world_bounds: AABB,
}

impl OctreeCulling {
    /// Create an octree for the given world bounds.
    pub fn new(world_bounds: AABB) -> Self {
        Self {
            root: Some(OctreeNode::Leaf {
                bounds: world_bounds,
                objects: Vec::new(),
            }),
            object_bounds: HashMap::new(),
            world_bounds,
        }
    }

    /// Insert an object into the octree.
    pub fn insert(&mut self, id: u64, aabb: AABB) {
        self.object_bounds.insert(id, aabb);
        if let Some(root) = self.root.take() {
            self.root = Some(Self::insert_into(root, id, &aabb, 0));
        }
    }

    fn insert_into(node: OctreeNode, id: u64, aabb: &AABB, depth: u32) -> OctreeNode {
        match node {
            OctreeNode::Leaf {
                bounds,
                mut objects,
            } => {
                if objects.len() < OCTREE_MAX_OBJECTS_PER_LEAF
                    || depth >= OCTREE_MAX_DEPTH
                {
                    objects.push(id);
                    OctreeNode::Leaf { bounds, objects }
                } else {
                    // Split into internal node.
                    let mut children: [Option<OctreeNode>; 8] = Default::default();
                    let center = bounds.center();

                    // Create child bounds.
                    for ci in 0..8 {
                        let child_bounds = Self::child_bounds(&bounds, &center, ci);
                        children[ci] = Some(OctreeNode::Leaf {
                            bounds: child_bounds,
                            objects: Vec::new(),
                        });
                    }

                    let mut internal = OctreeNode::Internal {
                        bounds,
                        children: Box::new(children),
                    };

                    // Re-insert existing objects.
                    for existing_id in objects {
                        internal = Self::insert_into_internal(internal, existing_id, aabb, depth);
                    }
                    // Insert new object.
                    internal = Self::insert_into_internal(internal, id, aabb, depth);

                    internal
                }
            }
            OctreeNode::Internal { .. } => {
                Self::insert_into_internal(node, id, aabb, depth)
            }
        }
    }

    fn insert_into_internal(
        node: OctreeNode,
        id: u64,
        aabb: &AABB,
        depth: u32,
    ) -> OctreeNode {
        match node {
            OctreeNode::Internal {
                bounds,
                mut children,
            } => {
                let center = bounds.center();
                // Find which child(ren) the AABB overlaps.
                let mut inserted = false;
                for ci in 0..8 {
                    let child_bounds = Self::child_bounds(&bounds, &center, ci);
                    if child_bounds.overlaps(aabb) {
                        if let Some(child) = children[ci].take() {
                            children[ci] =
                                Some(Self::insert_into(child, id, aabb, depth + 1));
                            inserted = true;
                        }
                    }
                }
                if !inserted {
                    // Object doesn't fit any child -- put it in the first
                    // available leaf (shouldn't normally happen with proper bounds).
                    if let Some(child) = children[0].take() {
                        children[0] = Some(Self::insert_into(child, id, aabb, depth + 1));
                    }
                }
                OctreeNode::Internal {
                    bounds,
                    children,
                }
            }
            leaf => leaf,
        }
    }

    fn child_bounds(parent: &AABB, center: &Vec3, index: usize) -> AABB {
        let min_x = if index & 1 == 0 {
            parent.min.x
        } else {
            center.x
        };
        let max_x = if index & 1 == 0 {
            center.x
        } else {
            parent.max.x
        };
        let min_y = if index & 2 == 0 {
            parent.min.y
        } else {
            center.y
        };
        let max_y = if index & 2 == 0 {
            center.y
        } else {
            parent.max.y
        };
        let min_z = if index & 4 == 0 {
            parent.min.z
        } else {
            center.z
        };
        let max_z = if index & 4 == 0 {
            center.z
        } else {
            parent.max.z
        };
        AABB::new(
            Vec3::new(min_x, min_y, min_z),
            Vec3::new(max_x, max_y, max_z),
        )
    }

    /// Query the octree for all objects whose AABBs are inside or intersect
    /// the given frustum.
    pub fn query_frustum(&self, frustum: &CullFrustum) -> Vec<u64> {
        let mut result = Vec::new();
        if let Some(ref root) = self.root {
            self.query_node(root, frustum, &mut result);
        }
        result
    }

    fn query_node(
        &self,
        node: &OctreeNode,
        frustum: &CullFrustum,
        result: &mut Vec<u64>,
    ) {
        let bounds = node.bounds();
        let test = frustum.test_aabb(bounds);

        match test {
            FrustumTestResult::Outside => return,
            FrustumTestResult::Inside => {
                // Everything in this subtree is visible.
                self.collect_all(node, result);
                return;
            }
            FrustumTestResult::Intersect => {
                // Partially visible -- recurse.
            }
        }

        match node {
            OctreeNode::Leaf { objects, .. } => {
                // Test each object individually.
                for &id in objects {
                    if let Some(aabb) = self.object_bounds.get(&id) {
                        if frustum.is_visible(aabb) {
                            result.push(id);
                        }
                    }
                }
            }
            OctreeNode::Internal { children, .. } => {
                for child in children.iter() {
                    if let Some(child_node) = child {
                        self.query_node(child_node, frustum, result);
                    }
                }
            }
        }
    }

    fn collect_all(&self, node: &OctreeNode, result: &mut Vec<u64>) {
        match node {
            OctreeNode::Leaf { objects, .. } => {
                result.extend(objects);
            }
            OctreeNode::Internal { children, .. } => {
                for child in children.iter() {
                    if let Some(child_node) = child {
                        self.collect_all(child_node, result);
                    }
                }
            }
        }
    }

    /// Remove an object from the octree.
    pub fn remove(&mut self, id: u64) {
        self.object_bounds.remove(&id);
        if let Some(root) = self.root.take() {
            self.root = Some(Self::remove_from(root, id));
        }
    }

    fn remove_from(node: OctreeNode, id: u64) -> OctreeNode {
        match node {
            OctreeNode::Leaf {
                bounds,
                mut objects,
            } => {
                objects.retain(|&oid| oid != id);
                OctreeNode::Leaf { bounds, objects }
            }
            OctreeNode::Internal {
                bounds,
                mut children,
            } => {
                for ci in 0..8 {
                    if let Some(child) = children[ci].take() {
                        children[ci] = Some(Self::remove_from(child, id));
                    }
                }
                OctreeNode::Internal { bounds, children }
            }
        }
    }

    /// Number of objects in the octree.
    pub fn len(&self) -> usize {
        self.object_bounds.len()
    }

    /// Whether the octree is empty.
    pub fn is_empty(&self) -> bool {
        self.object_bounds.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Potentially Visible Set (PVS)
// ---------------------------------------------------------------------------

/// A cell identifier in the PVS grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PvsCell(pub u32);

/// Precomputed visibility data for indoor environments.
///
/// The world is divided into a grid of cells. For each cell, we precompute
/// which other cells are visible from any point within it. At runtime, we
/// determine which cell the camera is in and immediately know the set of
/// potentially visible cells.
pub struct PotentiallyVisibleSet {
    /// Cell size in world units.
    pub cell_size: Vec3,
    /// Grid origin (min corner of cell 0,0,0).
    pub origin: Vec3,
    /// Grid dimensions.
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    /// Visibility data: for each cell, the set of visible cell indices.
    visibility: Vec<HashSet<u32>>,
}

impl PotentiallyVisibleSet {
    /// Create a new empty PVS.
    pub fn new(origin: Vec3, cell_size: Vec3, grid_x: u32, grid_y: u32, grid_z: u32) -> Self {
        let total = (grid_x * grid_y * grid_z) as usize;
        Self {
            cell_size,
            origin,
            grid_x,
            grid_y,
            grid_z,
            visibility: vec![HashSet::new(); total],
        }
    }

    /// Total number of cells.
    pub fn cell_count(&self) -> u32 {
        self.grid_x * self.grid_y * self.grid_z
    }

    /// Convert a 3D cell coordinate to a linear index.
    pub fn cell_index(&self, x: u32, y: u32, z: u32) -> Option<u32> {
        if x < self.grid_x && y < self.grid_y && z < self.grid_z {
            Some(x + y * self.grid_x + z * self.grid_x * self.grid_y)
        } else {
            None
        }
    }

    /// Convert a world position to a cell coordinate.
    pub fn world_to_cell(&self, pos: Vec3) -> Option<(u32, u32, u32)> {
        let local = pos.sub(self.origin);
        let cx = (local.x / self.cell_size.x) as i32;
        let cy = (local.y / self.cell_size.y) as i32;
        let cz = (local.z / self.cell_size.z) as i32;
        if cx >= 0
            && (cx as u32) < self.grid_x
            && cy >= 0
            && (cy as u32) < self.grid_y
            && cz >= 0
            && (cz as u32) < self.grid_z
        {
            Some((cx as u32, cy as u32, cz as u32))
        } else {
            None
        }
    }

    /// Mark cell `target` as visible from cell `source`.
    pub fn set_visible(&mut self, source: u32, target: u32) {
        if let Some(vis) = self.visibility.get_mut(source as usize) {
            vis.insert(target);
        }
    }

    /// Mark bidirectional visibility.
    pub fn set_mutually_visible(&mut self, a: u32, b: u32) {
        self.set_visible(a, b);
        self.set_visible(b, a);
    }

    /// Query which cells are visible from the given cell.
    pub fn visible_from(&self, cell: u32) -> Option<&HashSet<u32>> {
        self.visibility.get(cell as usize)
    }

    /// Query which cells are visible from a world position.
    pub fn visible_from_position(&self, pos: Vec3) -> Option<&HashSet<u32>> {
        let (cx, cy, cz) = self.world_to_cell(pos)?;
        let idx = self.cell_index(cx, cy, cz)?;
        self.visible_from(idx)
    }
}

/// Builder for computing PVS data.
pub struct PvsBuilder {
    /// Portals connecting cells.
    portals: Vec<Portal>,
    /// PVS being built.
    pvs: PotentiallyVisibleSet,
}

/// A portal connecting two cells.
#[derive(Debug, Clone)]
pub struct Portal {
    /// First cell.
    pub cell_a: u32,
    /// Second cell.
    pub cell_b: u32,
    /// Portal center position.
    pub center: Vec3,
    /// Portal normal (points from cell_a toward cell_b).
    pub normal: Vec3,
    /// Portal half-extents (width/2, height/2).
    pub half_width: f32,
    pub half_height: f32,
}

impl PvsBuilder {
    /// Create a new PVS builder.
    pub fn new(
        origin: Vec3,
        cell_size: Vec3,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
    ) -> Self {
        Self {
            portals: Vec::new(),
            pvs: PotentiallyVisibleSet::new(origin, cell_size, grid_x, grid_y, grid_z),
        }
    }

    /// Add a portal between two cells.
    pub fn add_portal(&mut self, portal: Portal) {
        self.portals.push(portal);
    }

    /// Compute the PVS using flood-fill through portals.
    ///
    /// This uses a simplified version of the PVS algorithm: from each cell,
    /// we do a BFS through connected portals, marking reachable cells as
    /// visible. A more advanced implementation would use the anti-penumbra
    /// approach to clip visibility through portal sequences.
    pub fn compute(mut self) -> PotentiallyVisibleSet {
        let cell_count = self.pvs.cell_count();

        // Build adjacency from portals.
        let mut adjacency: HashMap<u32, Vec<u32>> = HashMap::new();
        for portal in &self.portals {
            adjacency
                .entry(portal.cell_a)
                .or_default()
                .push(portal.cell_b);
            adjacency
                .entry(portal.cell_b)
                .or_default()
                .push(portal.cell_a);
        }

        // For each cell, flood-fill through portals.
        for source in 0..cell_count {
            let mut visited: HashSet<u32> = HashSet::new();
            let mut queue: VecDeque<u32> = VecDeque::new();

            visited.insert(source);
            queue.push_back(source);
            self.pvs.set_visible(source, source); // cell sees itself

            while let Some(current) = queue.pop_front() {
                if let Some(neighbors) = adjacency.get(&current) {
                    for &neighbor in neighbors {
                        if visited.insert(neighbor) {
                            self.pvs.set_visible(source, neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        self.pvs
    }

    /// Compute PVS with a maximum portal depth (limits how many portals
    /// visibility can chain through).
    pub fn compute_with_depth(mut self, max_depth: u32) -> PotentiallyVisibleSet {
        let cell_count = self.pvs.cell_count();

        let mut adjacency: HashMap<u32, Vec<u32>> = HashMap::new();
        for portal in &self.portals {
            adjacency
                .entry(portal.cell_a)
                .or_default()
                .push(portal.cell_b);
            adjacency
                .entry(portal.cell_b)
                .or_default()
                .push(portal.cell_a);
        }

        for source in 0..cell_count {
            let mut visited: HashSet<u32> = HashSet::new();
            let mut queue: VecDeque<(u32, u32)> = VecDeque::new();

            visited.insert(source);
            queue.push_back((source, 0));
            self.pvs.set_visible(source, source);

            while let Some((current, depth)) = queue.pop_front() {
                if depth >= max_depth {
                    continue;
                }
                if let Some(neighbors) = adjacency.get(&current) {
                    for &neighbor in neighbors {
                        if visited.insert(neighbor) {
                            self.pvs.set_visible(source, neighbor);
                            queue.push_back((neighbor, depth + 1));
                        }
                    }
                }
            }
        }

        self.pvs
    }
}

// ---------------------------------------------------------------------------
// PVS query helper
// ---------------------------------------------------------------------------

/// Runtime PVS query: given a camera position, returns the set of visible
/// cell indices.
pub struct PvsQuery<'a> {
    pvs: &'a PotentiallyVisibleSet,
}

impl<'a> PvsQuery<'a> {
    pub fn new(pvs: &'a PotentiallyVisibleSet) -> Self {
        Self { pvs }
    }

    /// Query visible cells from a world position.
    pub fn query(&self, camera_pos: Vec3) -> Vec<u32> {
        match self.pvs.visible_from_position(camera_pos) {
            Some(cells) => cells.iter().copied().collect(),
            None => {
                // Camera outside the PVS grid -- return all cells.
                (0..self.pvs.cell_count()).collect()
            }
        }
    }

    /// Query visible cells and also expand to neighboring cells for smoother
    /// transitions (useful when camera is near a cell boundary).
    pub fn query_expanded(&self, camera_pos: Vec3) -> Vec<u32> {
        let mut result = HashSet::new();

        // Query the primary cell.
        if let Some(cells) = self.pvs.visible_from_position(camera_pos) {
            result.extend(cells);
        }

        // Also query neighboring cells for smooth transitions.
        let offsets = [
            Vec3::new(self.pvs.cell_size.x, 0.0, 0.0),
            Vec3::new(-self.pvs.cell_size.x, 0.0, 0.0),
            Vec3::new(0.0, self.pvs.cell_size.y, 0.0),
            Vec3::new(0.0, -self.pvs.cell_size.y, 0.0),
            Vec3::new(0.0, 0.0, self.pvs.cell_size.z),
            Vec3::new(0.0, 0.0, -self.pvs.cell_size.z),
        ];

        for offset in &offsets {
            let neighbor_pos = camera_pos.add(*offset);
            if let Some(cells) = self.pvs.visible_from_position(neighbor_pos) {
                result.extend(cells);
            }
        }

        result.into_iter().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_vp() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, // col0
            0.0, 1.0, 0.0, 0.0, // col1
            0.0, 0.0, 1.0, 0.0, // col2
            0.0, 0.0, 0.0, 1.0, // col3
        ]
    }

    fn make_ortho_vp() -> [f32; 16] {
        // Simple orthographic: maps [-10,10]^3 to [-1,1]^3.
        let s = 0.1f32; // 1/10
        [
            s, 0.0, 0.0, 0.0,
            0.0, s, 0.0, 0.0,
            0.0, 0.0, s, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    #[test]
    fn test_aabb_basics() {
        let a = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(a.center().x, 0.0);
        assert_eq!(a.half_extents().x, 1.0);
        assert!(a.contains_point(Vec3::ZERO));
        assert!(!a.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_aabb_overlap() {
        let a = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let b = AABB::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        let c = AABB::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_frustum_aabb_visible() {
        let vp = make_ortho_vp();
        let frustum = CullFrustum::from_view_projection(&vp);

        // AABB at origin should be visible.
        let inside = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(frustum.is_visible(&inside));

        // AABB far away should not be visible.
        let outside = AABB::new(
            Vec3::new(100.0, 100.0, 100.0),
            Vec3::new(101.0, 101.0, 101.0),
        );
        assert!(!frustum.is_visible(&outside));
    }

    #[test]
    fn test_frustum_sphere() {
        let vp = make_ortho_vp();
        let frustum = CullFrustum::from_view_projection(&vp);

        let result = frustum.test_sphere(Vec3::ZERO, 1.0);
        assert_ne!(result, FrustumTestResult::Outside);

        let result2 = frustum.test_sphere(Vec3::new(100.0, 100.0, 100.0), 1.0);
        assert_eq!(result2, FrustumTestResult::Outside);
    }

    #[test]
    fn test_simd_batch_frustum() {
        let vp = make_ortho_vp();
        let frustum = CullFrustum::from_view_projection(&vp);

        let aabbs = [
            // Inside.
            AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            // Outside.
            AABB::new(
                Vec3::new(100.0, 100.0, 100.0),
                Vec3::new(101.0, 101.0, 101.0),
            ),
            // Inside.
            AABB::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0)),
            // Outside.
            AABB::new(
                Vec3::new(-200.0, -200.0, -200.0),
                Vec3::new(-199.0, -199.0, -199.0),
            ),
        ];

        let batch = SimdAabbBatch::from_aabbs(&aabbs);
        let mask = frustum_test_batch_4(&frustum, &batch);

        assert!((mask & 1) != 0, "AABB 0 should be visible");
        assert!((mask & 2) == 0, "AABB 1 should be outside");
        assert!((mask & 4) != 0, "AABB 2 should be visible");
        assert!((mask & 8) == 0, "AABB 3 should be outside");
    }

    #[test]
    fn test_batch_frustum_many() {
        let vp = make_ortho_vp();
        let frustum = CullFrustum::from_view_projection(&vp);

        let mut aabbs = Vec::new();
        for i in 0..17 {
            let x = (i as f32) * 2.0 - 8.0;
            aabbs.push(AABB::new(
                Vec3::new(x, -0.5, -0.5),
                Vec3::new(x + 1.0, 0.5, 0.5),
            ));
        }

        let results = frustum_test_batch(&frustum, &aabbs);
        assert_eq!(results.len(), 17);

        // Verify against scalar.
        for (i, aabb) in aabbs.iter().enumerate() {
            let scalar = frustum.is_visible(aabb);
            assert_eq!(
                results[i], scalar,
                "Batch and scalar disagree for AABB {}",
                i
            );
        }
    }

    #[test]
    fn test_visibility_system() {
        let vp = make_ortho_vp();
        let camera = CullCamera {
            view_projection: vp,
            position: Vec3::ZERO,
            max_render_distance: 50.0,
            layer_mask: LayerMask::ALL,
            screen_width: 1920.0,
            screen_height: 1080.0,
            fov_y: std::f32::consts::PI / 3.0,
        };

        let objects = vec![
            CullObject::simple(
                0,
                AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            ),
            CullObject::simple(
                1,
                AABB::new(
                    Vec3::new(100.0, 100.0, 100.0),
                    Vec3::new(101.0, 101.0, 101.0),
                ),
            ),
        ];

        let system = VisibilitySystem::new();
        let (results, stats) = system.cull(&objects, &camera, None);

        assert!(results[0].visible);
        assert!(!results[1].visible);
        assert_eq!(stats.visible_objects, 1);
    }

    #[test]
    fn test_distance_culling() {
        let vp = make_ortho_vp();
        let camera = CullCamera {
            view_projection: vp,
            position: Vec3::ZERO,
            max_render_distance: 5.0,
            layer_mask: LayerMask::ALL,
            screen_width: 1920.0,
            screen_height: 1080.0,
            fov_y: std::f32::consts::PI / 3.0,
        };

        let objects = vec![CullObject::simple(
            0,
            AABB::new(Vec3::new(3.0, 3.0, 3.0), Vec3::new(4.0, 4.0, 4.0)),
        )];

        let system = VisibilitySystem::new();
        let (results, stats) = system.cull(&objects, &camera, None);

        // The center of this AABB is (3.5, 3.5, 3.5), distance ~6.06 > 5.0
        assert!(!results[0].visible);
        assert_eq!(results[0].cull_reason, CullReason::Distance);
        assert_eq!(stats.distance_culled, 1);
    }

    #[test]
    fn test_layer_culling() {
        let vp = make_ortho_vp();
        let camera = CullCamera {
            view_projection: vp,
            position: Vec3::ZERO,
            max_render_distance: 1000.0,
            layer_mask: LayerMask(0b0001), // only layer 0
            screen_width: 1920.0,
            screen_height: 1080.0,
            fov_y: std::f32::consts::PI / 3.0,
        };

        let mut obj =
            CullObject::simple(0, AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)));
        obj.layer = LayerMask(0b0010); // layer 1

        let system = VisibilitySystem::new();
        let (results, _) = system.cull(&[obj], &camera, None);

        assert!(!results[0].visible);
        assert_eq!(results[0].cull_reason, CullReason::LayerMask);
    }

    #[test]
    fn test_lod_ranges() {
        let lod = LodRanges::new(vec![10.0, 25.0, 50.0]);
        assert_eq!(lod.lod_for_distance(5.0), 0);
        assert_eq!(lod.lod_for_distance(10.0), 0);
        assert_eq!(lod.lod_for_distance(11.0), 1);
        assert_eq!(lod.lod_for_distance(30.0), 2);
        assert_eq!(lod.lod_for_distance(100.0), 3);
    }

    #[test]
    fn test_octree_insert_query() {
        let world = AABB::new(
            Vec3::new(-100.0, -100.0, -100.0),
            Vec3::new(100.0, 100.0, 100.0),
        );
        let mut octree = OctreeCulling::new(world);

        // Insert objects.
        for i in 0..20 {
            let x = (i as f32) * 5.0 - 50.0;
            let aabb = AABB::new(Vec3::new(x, -1.0, -1.0), Vec3::new(x + 2.0, 1.0, 1.0));
            octree.insert(i, aabb);
        }

        assert_eq!(octree.len(), 20);

        // Query with a frustum that sees the center.
        let vp = make_ortho_vp();
        let frustum = CullFrustum::from_view_projection(&vp);
        let visible = octree.query_frustum(&frustum);

        // Some objects near the center should be visible.
        assert!(!visible.is_empty());
    }

    #[test]
    fn test_pvs_basic() {
        let mut pvs = PotentiallyVisibleSet::new(
            Vec3::ZERO,
            Vec3::new(10.0, 10.0, 10.0),
            3,
            3,
            1,
        );

        // Cells: 0..8 in a 3x3x1 grid.
        let center = pvs.cell_index(1, 1, 0).unwrap(); // cell 4
        let left = pvs.cell_index(0, 1, 0).unwrap();   // cell 3
        let right = pvs.cell_index(2, 1, 0).unwrap();   // cell 5

        pvs.set_mutually_visible(center, left);
        pvs.set_mutually_visible(center, right);

        let vis = pvs.visible_from(center).unwrap();
        assert!(vis.contains(&left));
        assert!(vis.contains(&right));
    }

    #[test]
    fn test_pvs_builder() {
        let builder = PvsBuilder::new(
            Vec3::ZERO,
            Vec3::new(10.0, 10.0, 10.0),
            3,
            1,
            1,
        );

        // Cells 0, 1, 2 in a row. Portals: 0-1, 1-2.
        let mut builder = builder;
        builder.add_portal(Portal {
            cell_a: 0,
            cell_b: 1,
            center: Vec3::new(10.0, 5.0, 5.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
            half_width: 5.0,
            half_height: 5.0,
        });
        builder.add_portal(Portal {
            cell_a: 1,
            cell_b: 2,
            center: Vec3::new(20.0, 5.0, 5.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
            half_width: 5.0,
            half_height: 5.0,
        });

        let pvs = builder.compute();

        // Cell 0 can see 0, 1, 2 (through portals).
        let vis0 = pvs.visible_from(0).unwrap();
        assert!(vis0.contains(&0));
        assert!(vis0.contains(&1));
        assert!(vis0.contains(&2));
    }

    #[test]
    fn test_pvs_builder_depth_limited() {
        let mut builder = PvsBuilder::new(
            Vec3::ZERO,
            Vec3::new(10.0, 10.0, 10.0),
            4,
            1,
            1,
        );

        // Linear chain: 0-1-2-3.
        builder.add_portal(Portal {
            cell_a: 0,
            cell_b: 1,
            center: Vec3::ZERO,
            normal: Vec3::new(1.0, 0.0, 0.0),
            half_width: 5.0,
            half_height: 5.0,
        });
        builder.add_portal(Portal {
            cell_a: 1,
            cell_b: 2,
            center: Vec3::ZERO,
            normal: Vec3::new(1.0, 0.0, 0.0),
            half_width: 5.0,
            half_height: 5.0,
        });
        builder.add_portal(Portal {
            cell_a: 2,
            cell_b: 3,
            center: Vec3::ZERO,
            normal: Vec3::new(1.0, 0.0, 0.0),
            half_width: 5.0,
            half_height: 5.0,
        });

        let pvs = builder.compute_with_depth(1);

        // Cell 0 should see 0 and 1 (depth 1) but NOT 2 or 3.
        let vis0 = pvs.visible_from(0).unwrap();
        assert!(vis0.contains(&0));
        assert!(vis0.contains(&1));
        assert!(!vis0.contains(&2));
        assert!(!vis0.contains(&3));
    }

    #[test]
    fn test_cull_plane_distance() {
        let plane = CullPlane::from_abcd(0.0, 1.0, 0.0, 0.0); // y = 0 plane, normal up
        assert!(plane.signed_distance(0.0, 5.0, 0.0) > 0.0);
        assert!(plane.signed_distance(0.0, -5.0, 0.0) < 0.0);
    }

    #[test]
    fn test_layer_mask() {
        let a = LayerMask(0b1010);
        let b = LayerMask(0b0010);
        assert!(a.overlaps(b));
        assert!(a.contains(1));
        assert!(!a.contains(0));
    }
}
