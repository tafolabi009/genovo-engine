// engine/render/src/forward_plus.rs
//
// Forward+ (tiled forward) rendering pipeline.
//
// The Forward+ technique splits the screen into tiles (typically 16x16 pixels),
// performs a depth prepass to establish min/max depth per tile, then assigns
// lights to tiles via a compute pass. The final forward pass reads the per-tile
// light list from an SSBO and only evaluates lights that actually affect each
// tile. This gives the shading quality of forward rendering with the light
// culling efficiency of deferred.
//
// Pipeline stages:
//   1. Depth prepass -- render scene depth-only to populate the depth buffer.
//   2. Light assignment -- compute shader reads depth buffer, builds per-tile
//      min/max depth, tests each light against the tile frustum, and writes
//      per-tile light indices into an SSBO.
//   3. Forward render -- full-shading pass reads per-tile light list and
//      evaluates only relevant lights per fragment.
//   4. Transparent pass -- back-to-front transparent objects with per-tile
//      light list (no depth prepass contribution).
//   5. Debug visualization -- optional overlay showing tile heat map, light
//      counts, and tile boundaries.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default tile size in pixels (width and height).
pub const DEFAULT_TILE_SIZE: u32 = 16;

/// Maximum number of lights that can be assigned to a single tile.
pub const MAX_LIGHTS_PER_TILE: usize = 256;

/// Maximum total lights the system can handle in a single frame.
pub const MAX_TOTAL_LIGHTS: usize = 8192;

/// Depth prepass shader entry point name.
pub const DEPTH_PREPASS_ENTRY: &str = "depth_prepass_main";

/// Light assignment compute shader entry point name.
pub const LIGHT_ASSIGN_ENTRY: &str = "light_assign_main";

/// Forward shading entry point name.
pub const FORWARD_SHADE_ENTRY: &str = "forward_shade_main";

/// Number of threads per workgroup in the light assignment compute shader.
pub const LIGHT_ASSIGN_WORKGROUP_SIZE: u32 = 256;

/// Minimum number of lights before Forward+ provides a benefit over plain forward.
pub const FORWARD_PLUS_LIGHT_THRESHOLD: usize = 16;

// ---------------------------------------------------------------------------
// Vec3 / Vec4 / Mat4 -- lightweight math types used internally
// ---------------------------------------------------------------------------

/// A 3-component vector.
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

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

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
    pub fn length(self) -> f32 { self.dot(self).sqrt() }

    #[inline]
    pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-12 { return Self::ZERO; }
        let inv = 1.0 / len;
        Self { x: self.x * inv, y: self.y * inv, z: self.z * inv }
    }

    #[inline]
    pub fn distance(self, rhs: Self) -> f32 {
        let dx = self.x - rhs.x;
        let dy = self.y - rhs.y;
        let dz = self.z - rhs.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }

    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    #[inline]
    pub fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }

    #[inline]
    pub fn min_components(self, rhs: Self) -> Self {
        Self {
            x: self.x.min(rhs.x),
            y: self.y.min(rhs.y),
            z: self.z.min(rhs.z),
        }
    }

    #[inline]
    pub fn max_components(self, rhs: Self) -> Self {
        Self {
            x: self.x.max(rhs.x),
            y: self.y.max(rhs.y),
            z: self.z.max(rhs.z),
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self::add(self, rhs) }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self::sub(self, rhs) }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self { self.scale(rhs) }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self { Self::neg(self) }
}

/// A 4-component vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }

    #[inline]
    pub fn from_vec3(v: Vec3, w: f32) -> Self { Self { x: v.x, y: v.y, z: v.z, w } }

    #[inline]
    pub fn xyz(self) -> Vec3 { Vec3 { x: self.x, y: self.y, z: self.z } }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
}

/// A 4x4 matrix stored in column-major order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        cols: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Create a perspective projection matrix (Vulkan clip space, depth [0, 1]).
    pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y_rad * 0.5).tan();
        let range_inv = 1.0 / (near - far);
        Self {
            cols: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, far * range_inv, -1.0],
                [0.0, 0.0, near * far * range_inv, 0.0],
            ],
        }
    }

    /// Create a look-at view matrix.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = (target - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self {
            cols: [
                [s.x, u.x, -f.x, 0.0],
                [s.y, u.y, -f.y, 0.0],
                [s.z, u.z, -f.z, 0.0],
                [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
            ],
        }
    }

    /// Multiply this matrix by a Vec4.
    pub fn mul_vec4(&self, v: Vec4) -> Vec4 {
        Vec4 {
            x: self.cols[0][0] * v.x + self.cols[1][0] * v.y + self.cols[2][0] * v.z + self.cols[3][0] * v.w,
            y: self.cols[0][1] * v.x + self.cols[1][1] * v.y + self.cols[2][1] * v.z + self.cols[3][1] * v.w,
            z: self.cols[0][2] * v.x + self.cols[1][2] * v.y + self.cols[2][2] * v.z + self.cols[3][2] * v.w,
            w: self.cols[0][3] * v.x + self.cols[1][3] * v.y + self.cols[2][3] * v.z + self.cols[3][3] * v.w,
        }
    }

    /// Multiply two matrices.
    pub fn mul_mat4(&self, rhs: &Self) -> Self {
        let mut result = Self { cols: [[0.0; 4]; 4] };
        for c in 0..4 {
            for r in 0..4 {
                result.cols[c][r] =
                    self.cols[0][r] * rhs.cols[c][0]
                    + self.cols[1][r] * rhs.cols[c][1]
                    + self.cols[2][r] * rhs.cols[c][2]
                    + self.cols[3][r] * rhs.cols[c][3];
            }
        }
        result
    }

    /// Compute the inverse of this matrix. Returns `None` if singular.
    pub fn inverse(&self) -> Option<Self> {
        let m = &self.cols;
        let mut inv = [[0.0f32; 4]; 4];

        inv[0][0] = m[1][1] * m[2][2] * m[3][3] - m[1][1] * m[2][3] * m[3][2]
            - m[2][1] * m[1][2] * m[3][3] + m[2][1] * m[1][3] * m[3][2]
            + m[3][1] * m[1][2] * m[2][3] - m[3][1] * m[1][3] * m[2][2];

        inv[1][0] = -m[1][0] * m[2][2] * m[3][3] + m[1][0] * m[2][3] * m[3][2]
            + m[2][0] * m[1][2] * m[3][3] - m[2][0] * m[1][3] * m[3][2]
            - m[3][0] * m[1][2] * m[2][3] + m[3][0] * m[1][3] * m[2][2];

        inv[2][0] = m[1][0] * m[2][1] * m[3][3] - m[1][0] * m[2][3] * m[3][1]
            - m[2][0] * m[1][1] * m[3][3] + m[2][0] * m[1][3] * m[3][1]
            + m[3][0] * m[1][1] * m[2][3] - m[3][0] * m[1][3] * m[2][1];

        inv[3][0] = -m[1][0] * m[2][1] * m[3][2] + m[1][0] * m[2][2] * m[3][1]
            + m[2][0] * m[1][1] * m[3][2] - m[2][0] * m[1][2] * m[3][1]
            - m[3][0] * m[1][1] * m[2][2] + m[3][0] * m[1][2] * m[2][1];

        let det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];
        if det.abs() < 1e-12 { return None; }

        inv[0][1] = -m[0][1] * m[2][2] * m[3][3] + m[0][1] * m[2][3] * m[3][2]
            + m[2][1] * m[0][2] * m[3][3] - m[2][1] * m[0][3] * m[3][2]
            - m[3][1] * m[0][2] * m[2][3] + m[3][1] * m[0][3] * m[2][2];

        inv[1][1] = m[0][0] * m[2][2] * m[3][3] - m[0][0] * m[2][3] * m[3][2]
            - m[2][0] * m[0][2] * m[3][3] + m[2][0] * m[0][3] * m[3][2]
            + m[3][0] * m[0][2] * m[2][3] - m[3][0] * m[0][3] * m[2][2];

        inv[2][1] = -m[0][0] * m[2][1] * m[3][3] + m[0][0] * m[2][3] * m[3][1]
            + m[2][0] * m[0][1] * m[3][3] - m[2][0] * m[0][3] * m[3][1]
            - m[3][0] * m[0][1] * m[2][3] + m[3][0] * m[0][3] * m[2][1];

        inv[3][1] = m[0][0] * m[2][1] * m[3][2] - m[0][0] * m[2][2] * m[3][1]
            - m[2][0] * m[0][1] * m[3][2] + m[2][0] * m[0][2] * m[3][1]
            + m[3][0] * m[0][1] * m[2][2] - m[3][0] * m[0][2] * m[2][1];

        inv[0][2] = m[0][1] * m[1][2] * m[3][3] - m[0][1] * m[1][3] * m[3][2]
            - m[1][1] * m[0][2] * m[3][3] + m[1][1] * m[0][3] * m[3][2]
            + m[3][1] * m[0][2] * m[1][3] - m[3][1] * m[0][3] * m[1][2];

        inv[1][2] = -m[0][0] * m[1][2] * m[3][3] + m[0][0] * m[1][3] * m[3][2]
            + m[1][0] * m[0][2] * m[3][3] - m[1][0] * m[0][3] * m[3][2]
            - m[3][0] * m[0][2] * m[1][3] + m[3][0] * m[0][3] * m[1][2];

        inv[2][2] = m[0][0] * m[1][1] * m[3][3] - m[0][0] * m[1][3] * m[3][1]
            - m[1][0] * m[0][1] * m[3][3] + m[1][0] * m[0][3] * m[3][1]
            + m[3][0] * m[0][1] * m[1][3] - m[3][0] * m[0][3] * m[1][1];

        inv[3][2] = -m[0][0] * m[1][1] * m[3][2] + m[0][0] * m[1][2] * m[3][1]
            + m[1][0] * m[0][1] * m[3][2] - m[1][0] * m[0][2] * m[3][1]
            - m[3][0] * m[0][1] * m[1][2] + m[3][0] * m[0][2] * m[1][1];

        inv[0][3] = -m[0][1] * m[1][2] * m[2][3] + m[0][1] * m[1][3] * m[2][2]
            + m[1][1] * m[0][2] * m[2][3] - m[1][1] * m[0][3] * m[2][2]
            - m[2][1] * m[0][2] * m[1][3] + m[2][1] * m[0][3] * m[1][2];

        inv[1][3] = m[0][0] * m[1][2] * m[2][3] - m[0][0] * m[1][3] * m[2][2]
            - m[1][0] * m[0][2] * m[2][3] + m[1][0] * m[0][3] * m[2][2]
            + m[2][0] * m[0][2] * m[1][3] - m[2][0] * m[0][3] * m[1][2];

        inv[2][3] = -m[0][0] * m[1][1] * m[2][3] + m[0][0] * m[1][3] * m[2][1]
            + m[1][0] * m[0][1] * m[2][3] - m[1][0] * m[0][3] * m[2][1]
            - m[2][0] * m[0][1] * m[1][3] + m[2][0] * m[0][3] * m[1][1];

        inv[3][3] = m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1]
            - m[1][0] * m[0][1] * m[2][2] + m[1][0] * m[0][2] * m[2][1]
            + m[2][0] * m[0][1] * m[1][2] - m[2][0] * m[0][2] * m[1][1];

        let inv_det = 1.0 / det;
        for c in 0..4 {
            for r in 0..4 {
                inv[c][r] *= inv_det;
            }
        }
        Some(Self { cols: inv })
    }
}

// ---------------------------------------------------------------------------
// Frustum planes for tile culling
// ---------------------------------------------------------------------------

/// A plane represented by the equation: normal.dot(point) + d = 0.
#[derive(Debug, Clone, Copy)]
pub struct FrustumPlane {
    pub normal: Vec3,
    pub d: f32,
}

impl FrustumPlane {
    pub fn new(normal: Vec3, d: f32) -> Self { Self { normal, d } }

    /// Signed distance from a point to this plane.
    #[inline]
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }

    /// Test whether a sphere (center, radius) is at least partially in front of
    /// the plane.
    #[inline]
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.distance_to_point(center) > -radius
    }
}

/// Six-plane frustum used for light-vs-tile culling.
#[derive(Debug, Clone)]
pub struct TileFrustum {
    pub planes: [FrustumPlane; 6],
}

impl TileFrustum {
    /// Construct a tile frustum from 4 screen-space corner rays and near/far depths.
    pub fn from_tile_corners(
        corners_vs: [Vec3; 4],
        near_depth: f32,
        far_depth: f32,
        view_dir: Vec3,
    ) -> Self {
        // Near plane
        let near_plane = FrustumPlane::new(view_dir.neg(), near_depth);
        // Far plane
        let far_plane = FrustumPlane::new(view_dir, -far_depth);

        // Side planes from consecutive corner pairs
        let make_side = |a: Vec3, b: Vec3| -> FrustumPlane {
            let edge = b - a;
            let n = edge.cross(view_dir).normalize();
            let d = -n.dot(a);
            FrustumPlane::new(n, d)
        };

        let left = make_side(corners_vs[0], corners_vs[1]);
        let top = make_side(corners_vs[1], corners_vs[2]);
        let right = make_side(corners_vs[2], corners_vs[3]);
        let bottom = make_side(corners_vs[3], corners_vs[0]);

        Self {
            planes: [near_plane, far_plane, left, right, top, bottom],
        }
    }

    /// Test a point light (sphere) against this tile frustum.
    pub fn test_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            if !plane.intersects_sphere(center, radius) {
                return false;
            }
        }
        true
    }

    /// Test a spot light cone (approximated as a bounding sphere) against this
    /// tile frustum.
    pub fn test_cone_as_sphere(&self, apex: Vec3, direction: Vec3, range: f32, outer_angle: f32) -> bool {
        // Approximate the spot cone as a bounding sphere centered at the midpoint
        // along the cone axis, with radius covering the base disk.
        let half_range = range * 0.5;
        let center = apex + direction * half_range;
        let base_radius = range * outer_angle.sin();
        let bounding_radius = (half_range * half_range + base_radius * base_radius).sqrt();
        self.test_sphere(center, bounding_radius)
    }
}

// ---------------------------------------------------------------------------
// Light types
// ---------------------------------------------------------------------------

/// Unique identifier for a light in the Forward+ system.
pub type LightId = u32;

/// The type of light source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightKind {
    /// Omnidirectional point light with a finite range.
    Point,
    /// Spot light with inner/outer cone angles.
    Spot,
    /// Directional light (sun/moon) -- affects all tiles.
    Directional,
    /// Area light (rectangular or disk) -- approximated as point for culling.
    AreaRect,
    AreaDisk,
}

/// GPU-friendly light data packed for the light SSBO.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuLightData {
    /// World-space position (xyz) and range (w).
    pub position_range: [f32; 4],
    /// Direction (xyz) for spot/directional lights; w = light kind (as u32 bits).
    pub direction_kind: [f32; 4],
    /// Color (rgb) and intensity (a).
    pub color_intensity: [f32; 4],
    /// Spot inner angle (x), spot outer angle (y), shadow map index (z), flags (w).
    pub spot_params: [f32; 4],
}

impl GpuLightData {
    pub fn zeroed() -> Self {
        Self {
            position_range: [0.0; 4],
            direction_kind: [0.0; 4],
            color_intensity: [0.0; 4],
            spot_params: [0.0; 4],
        }
    }
}

/// High-level light descriptor used by the CPU-side pipeline.
#[derive(Debug, Clone)]
pub struct ForwardPlusLight {
    pub id: LightId,
    pub kind: LightKind,
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_angle: f32,
    pub outer_angle: f32,
    pub casts_shadow: bool,
    pub shadow_map_index: i32,
    pub enabled: bool,
    pub layer_mask: u32,
    /// Area light dimensions (width, height) for rect lights; (radius, 0) for disk.
    pub area_size: [f32; 2],
}

impl ForwardPlusLight {
    pub fn point(id: LightId, position: Vec3, color: Vec3, intensity: f32, range: f32) -> Self {
        Self {
            id,
            kind: LightKind::Point,
            position,
            direction: Vec3::ZERO,
            color,
            intensity,
            range,
            inner_angle: 0.0,
            outer_angle: 0.0,
            casts_shadow: false,
            shadow_map_index: -1,
            enabled: true,
            layer_mask: u32::MAX,
            area_size: [0.0; 2],
        }
    }

    pub fn spot(
        id: LightId,
        position: Vec3,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        Self {
            id,
            kind: LightKind::Spot,
            position,
            direction: direction.normalize(),
            color,
            intensity,
            range,
            inner_angle,
            outer_angle,
            casts_shadow: false,
            shadow_map_index: -1,
            enabled: true,
            layer_mask: u32::MAX,
            area_size: [0.0; 2],
        }
    }

    pub fn directional(id: LightId, direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            id,
            kind: LightKind::Directional,
            position: Vec3::ZERO,
            direction: direction.normalize(),
            color,
            intensity,
            range: f32::MAX,
            inner_angle: 0.0,
            outer_angle: 0.0,
            casts_shadow: true,
            shadow_map_index: -1,
            enabled: true,
            layer_mask: u32::MAX,
            area_size: [0.0; 2],
        }
    }

    /// Convert to GPU-packed format.
    pub fn to_gpu_data(&self) -> GpuLightData {
        let kind_bits = self.kind as u32;
        GpuLightData {
            position_range: [self.position.x, self.position.y, self.position.z, self.range],
            direction_kind: [
                self.direction.x,
                self.direction.y,
                self.direction.z,
                f32::from_bits(kind_bits),
            ],
            color_intensity: [self.color.x, self.color.y, self.color.z, self.intensity],
            spot_params: [
                self.inner_angle,
                self.outer_angle,
                self.shadow_map_index as f32,
                if self.casts_shadow { 1.0 } else { 0.0 },
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Tile data structures
// ---------------------------------------------------------------------------

/// Per-tile data computed during light assignment.
#[derive(Debug, Clone)]
pub struct TileData {
    /// Tile x index in the grid.
    pub tile_x: u32,
    /// Tile y index in the grid.
    pub tile_y: u32,
    /// Minimum depth value in this tile (from the depth prepass).
    pub min_depth: f32,
    /// Maximum depth value in this tile.
    pub max_depth: f32,
    /// Indices into the global light array for lights affecting this tile.
    pub light_indices: Vec<u32>,
    /// Number of opaque lights.
    pub opaque_light_count: u32,
    /// Number of transparent lights (lights that also affect transparent objects).
    pub transparent_light_count: u32,
}

impl TileData {
    pub fn new(tile_x: u32, tile_y: u32) -> Self {
        Self {
            tile_x,
            tile_y,
            min_depth: 1.0,
            max_depth: 0.0,
            light_indices: Vec::new(),
            opaque_light_count: 0,
            transparent_light_count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.min_depth = 1.0;
        self.max_depth = 0.0;
        self.light_indices.clear();
        self.opaque_light_count = 0;
        self.transparent_light_count = 0;
    }

    pub fn total_lights(&self) -> usize {
        self.light_indices.len()
    }
}

/// The per-tile light list SSBO layout.
///
/// The SSBO is laid out as:
/// - Header: [tiles_x, tiles_y, max_lights_per_tile, padding]
/// - Per tile: [light_count, light_index_0, light_index_1, ..., light_index_{max-1}]
///
/// Total size = 4 * sizeof(u32) + tiles_x * tiles_y * (1 + max_lights_per_tile) * sizeof(u32)
#[derive(Debug, Clone)]
pub struct TileLightListBuffer {
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub max_lights_per_tile: u32,
    /// Flattened buffer data ready for GPU upload.
    pub data: Vec<u32>,
}

impl TileLightListBuffer {
    pub fn new(tiles_x: u32, tiles_y: u32, max_lights_per_tile: u32) -> Self {
        let stride = 1 + max_lights_per_tile as usize;
        let tile_count = (tiles_x * tiles_y) as usize;
        let total_size = 4 + tile_count * stride;
        let mut data = vec![0u32; total_size];
        data[0] = tiles_x;
        data[1] = tiles_y;
        data[2] = max_lights_per_tile;
        data[3] = 0; // padding
        Self {
            tiles_x,
            tiles_y,
            max_lights_per_tile,
            data,
        }
    }

    /// Write the light list for a specific tile.
    pub fn write_tile(&mut self, tile_x: u32, tile_y: u32, light_indices: &[u32]) {
        let stride = 1 + self.max_lights_per_tile as usize;
        let tile_index = (tile_y * self.tiles_x + tile_x) as usize;
        let offset = 4 + tile_index * stride;
        let count = light_indices.len().min(self.max_lights_per_tile as usize);
        self.data[offset] = count as u32;
        for i in 0..count {
            self.data[offset + 1 + i] = light_indices[i];
        }
        // Zero out remaining slots to avoid stale data.
        for i in count..(self.max_lights_per_tile as usize) {
            self.data[offset + 1 + i] = 0;
        }
    }

    /// Read the light count for a specific tile.
    pub fn read_tile_count(&self, tile_x: u32, tile_y: u32) -> u32 {
        let stride = 1 + self.max_lights_per_tile as usize;
        let tile_index = (tile_y * self.tiles_x + tile_x) as usize;
        let offset = 4 + tile_index * stride;
        self.data[offset]
    }

    /// Read the light indices for a specific tile.
    pub fn read_tile_lights(&self, tile_x: u32, tile_y: u32) -> &[u32] {
        let stride = 1 + self.max_lights_per_tile as usize;
        let tile_index = (tile_y * self.tiles_x + tile_x) as usize;
        let offset = 4 + tile_index * stride;
        let count = self.data[offset] as usize;
        &self.data[offset + 1..offset + 1 + count]
    }

    /// Total buffer size in bytes for GPU upload.
    pub fn byte_size(&self) -> usize {
        self.data.len() * std::mem::size_of::<u32>()
    }

    /// Clear all tile light lists.
    pub fn clear(&mut self) {
        let stride = 1 + self.max_lights_per_tile as usize;
        let tile_count = (self.tiles_x * self.tiles_y) as usize;
        for t in 0..tile_count {
            let offset = 4 + t * stride;
            self.data[offset] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Depth prepass
// ---------------------------------------------------------------------------

/// Configuration for the depth prepass.
#[derive(Debug, Clone)]
pub struct DepthPrepassConfig {
    /// Whether to enable the depth prepass (can be disabled for small light counts).
    pub enabled: bool,
    /// Depth buffer format (typically D32Float or D24UnormS8Uint).
    pub depth_format: DepthFormat,
    /// Whether to also output a linear depth texture (useful for SSAO, etc.).
    pub output_linear_depth: bool,
    /// Whether to use masked alpha testing in the depth prepass.
    pub alpha_test: bool,
    /// Alpha threshold for alpha-tested objects.
    pub alpha_threshold: f32,
}

impl Default for DepthPrepassConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            depth_format: DepthFormat::D32Float,
            output_linear_depth: true,
            alpha_test: true,
            alpha_threshold: 0.5,
        }
    }
}

/// Supported depth buffer formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthFormat {
    D16Unorm,
    D24UnormS8Uint,
    D32Float,
    D32FloatS8Uint,
}

impl DepthFormat {
    pub fn byte_size(&self) -> u32 {
        match self {
            DepthFormat::D16Unorm => 2,
            DepthFormat::D24UnormS8Uint => 4,
            DepthFormat::D32Float => 4,
            DepthFormat::D32FloatS8Uint => 8,
        }
    }

    pub fn has_stencil(&self) -> bool {
        matches!(self, DepthFormat::D24UnormS8Uint | DepthFormat::D32FloatS8Uint)
    }
}

/// Result of the depth prepass for a single tile.
#[derive(Debug, Clone, Copy)]
pub struct TileDepthRange {
    pub min_depth: f32,
    pub max_depth: f32,
}

impl TileDepthRange {
    pub const EMPTY: Self = Self { min_depth: 1.0, max_depth: 0.0 };

    pub fn is_empty(&self) -> bool { self.min_depth > self.max_depth }

    pub fn update(&mut self, depth: f32) {
        self.min_depth = self.min_depth.min(depth);
        self.max_depth = self.max_depth.max(depth);
    }
}

/// CPU-side depth prepass processor.
///
/// In a real engine this would be a GPU render pass, but here we provide the
/// CPU-side logic for computing per-tile min/max depth from a depth buffer.
pub struct DepthPrepassProcessor {
    pub config: DepthPrepassConfig,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub tile_depths: Vec<TileDepthRange>,
}

impl DepthPrepassProcessor {
    pub fn new(width: u32, height: u32, tile_size: u32, config: DepthPrepassConfig) -> Self {
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;
        let tile_count = (tiles_x * tiles_y) as usize;
        Self {
            config,
            width,
            height,
            tile_size,
            tiles_x,
            tiles_y,
            tile_depths: vec![TileDepthRange::EMPTY; tile_count],
        }
    }

    /// Reset all tile depth ranges for a new frame.
    pub fn reset(&mut self) {
        for td in &mut self.tile_depths {
            *td = TileDepthRange::EMPTY;
        }
    }

    /// Process a depth buffer (row-major, one f32 per pixel) and compute
    /// per-tile min/max depth.
    pub fn process_depth_buffer(&mut self, depth_buffer: &[f32]) {
        assert_eq!(depth_buffer.len(), (self.width * self.height) as usize);
        self.reset();

        for y in 0..self.height {
            let tile_y = y / self.tile_size;
            for x in 0..self.width {
                let tile_x = x / self.tile_size;
                let tile_index = (tile_y * self.tiles_x + tile_x) as usize;
                let pixel_index = (y * self.width + x) as usize;
                let depth = depth_buffer[pixel_index];
                self.tile_depths[tile_index].update(depth);
            }
        }
    }

    /// Process a depth buffer with linearization. Converts from hyperbolic
    /// depth [0,1] to linear view-space depth [near, far].
    pub fn process_depth_buffer_linearized(
        &mut self,
        depth_buffer: &[f32],
        near: f32,
        far: f32,
    ) {
        assert_eq!(depth_buffer.len(), (self.width * self.height) as usize);
        self.reset();

        for y in 0..self.height {
            let tile_y = y / self.tile_size;
            for x in 0..self.width {
                let tile_x = x / self.tile_size;
                let tile_index = (tile_y * self.tiles_x + tile_x) as usize;
                let pixel_index = (y * self.width + x) as usize;
                let z_ndc = depth_buffer[pixel_index];
                // Linearize: reverse-Z to linear depth.
                let linear_depth = if z_ndc > 0.0 {
                    (near * far) / (far - z_ndc * (far - near))
                } else {
                    far
                };
                self.tile_depths[tile_index].update(linear_depth);
            }
        }
    }

    /// Get the depth range for a specific tile.
    pub fn tile_depth(&self, tile_x: u32, tile_y: u32) -> TileDepthRange {
        let idx = (tile_y * self.tiles_x + tile_x) as usize;
        self.tile_depths[idx]
    }

    /// Resize for a new resolution.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.tiles_x = (width + self.tile_size - 1) / self.tile_size;
        self.tiles_y = (height + self.tile_size - 1) / self.tile_size;
        let tile_count = (self.tiles_x * self.tiles_y) as usize;
        self.tile_depths.resize(tile_count, TileDepthRange::EMPTY);
    }
}

// ---------------------------------------------------------------------------
// Light assignment (CPU-side compute simulation)
// ---------------------------------------------------------------------------

/// Light assignment configuration.
#[derive(Debug, Clone)]
pub struct LightAssignConfig {
    /// Maximum lights per tile.
    pub max_lights_per_tile: u32,
    /// Whether to include directional lights in all tiles.
    pub include_directional_in_all_tiles: bool,
    /// Whether to cull lights for transparent objects (uses 0..far instead of
    /// min_depth..max_depth for the tile).
    pub transparent_light_assignment: bool,
    /// Layer mask for the camera (only lights matching this mask are considered).
    pub camera_layer_mask: u32,
}

impl Default for LightAssignConfig {
    fn default() -> Self {
        Self {
            max_lights_per_tile: MAX_LIGHTS_PER_TILE as u32,
            include_directional_in_all_tiles: true,
            transparent_light_assignment: true,
            camera_layer_mask: u32::MAX,
        }
    }
}

/// Per-tile light assignment result.
#[derive(Debug, Clone)]
pub struct TileLightAssignment {
    pub tile_x: u32,
    pub tile_y: u32,
    /// Light indices for opaque geometry (culled by tile depth range).
    pub opaque_lights: Vec<u32>,
    /// Light indices for transparent geometry (uses full tile frustum).
    pub transparent_lights: Vec<u32>,
}

/// The light assignment processor.
///
/// Performs per-tile light culling using tile frustums constructed from the
/// depth prepass and camera projection.
pub struct LightAssignProcessor {
    pub config: LightAssignConfig,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub tile_size: u32,
    pub assignments: Vec<TileLightAssignment>,
    /// Statistics for the last assignment pass.
    pub stats: LightAssignStats,
}

/// Statistics about the light assignment pass.
#[derive(Debug, Clone, Default)]
pub struct LightAssignStats {
    pub total_lights: u32,
    pub total_tiles: u32,
    pub min_lights_per_tile: u32,
    pub max_lights_per_tile: u32,
    pub avg_lights_per_tile: f32,
    pub tiles_at_max: u32,
    pub empty_tiles: u32,
    pub directional_lights: u32,
    pub point_lights: u32,
    pub spot_lights: u32,
    pub time_microseconds: u64,
}

impl LightAssignProcessor {
    pub fn new(tiles_x: u32, tiles_y: u32, tile_size: u32, config: LightAssignConfig) -> Self {
        let tile_count = (tiles_x * tiles_y) as usize;
        let mut assignments = Vec::with_capacity(tile_count);
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                assignments.push(TileLightAssignment {
                    tile_x: tx,
                    tile_y: ty,
                    opaque_lights: Vec::new(),
                    transparent_lights: Vec::new(),
                });
            }
        }
        Self {
            config,
            tiles_x,
            tiles_y,
            tile_size,
            assignments,
            stats: LightAssignStats::default(),
        }
    }

    /// Assign lights to tiles.
    ///
    /// `lights` is the full light list. `tile_depths` are min/max depths per tile
    /// from the depth prepass. `view_matrix` and `proj_matrix` are the camera
    /// matrices used to construct tile frustums.
    pub fn assign(
        &mut self,
        lights: &[ForwardPlusLight],
        tile_depths: &[TileDepthRange],
        view_matrix: &Mat4,
        proj_matrix: &Mat4,
        near: f32,
        far: f32,
    ) {
        let start = std::time::Instant::now();

        // Reset assignments.
        for a in &mut self.assignments {
            a.opaque_lights.clear();
            a.transparent_lights.clear();
        }

        // Classify lights.
        let mut directional_indices = Vec::new();
        let mut point_spot_indices = Vec::new();
        let mut dir_count = 0u32;
        let mut point_count = 0u32;
        let mut spot_count = 0u32;

        for (i, light) in lights.iter().enumerate() {
            if !light.enabled { continue; }
            if light.layer_mask & self.config.camera_layer_mask == 0 { continue; }

            match light.kind {
                LightKind::Directional => {
                    directional_indices.push(i as u32);
                    dir_count += 1;
                }
                LightKind::Point | LightKind::AreaRect | LightKind::AreaDisk => {
                    point_spot_indices.push(i as u32);
                    point_count += 1;
                }
                LightKind::Spot => {
                    point_spot_indices.push(i as u32);
                    spot_count += 1;
                }
            }
        }

        // Transform light positions into view space for culling.
        let mut light_positions_vs: Vec<Vec3> = Vec::with_capacity(lights.len());
        for light in lights {
            let pos_ws = Vec4::from_vec3(light.position, 1.0);
            let pos_vs = view_matrix.mul_vec4(pos_ws);
            light_positions_vs.push(pos_vs.xyz());
        }

        // For each tile, test each non-directional light.
        let inv_proj = proj_matrix.inverse().unwrap_or(Mat4::IDENTITY);

        for ty in 0..self.tiles_y {
            for tx in 0..self.tiles_x {
                let tile_index = (ty * self.tiles_x + tx) as usize;
                let depth_range = &tile_depths[tile_index];
                let assignment = &mut self.assignments[tile_index];

                // Add directional lights to all tiles.
                if self.config.include_directional_in_all_tiles {
                    for &idx in &directional_indices {
                        assignment.opaque_lights.push(idx);
                        assignment.transparent_lights.push(idx);
                    }
                }

                // Compute tile frustum corners in NDC.
                let x0 = (tx * self.tile_size) as f32 / (self.tiles_x * self.tile_size) as f32 * 2.0 - 1.0;
                let x1 = ((tx + 1) * self.tile_size) as f32 / (self.tiles_x * self.tile_size) as f32 * 2.0 - 1.0;
                let y0 = (ty * self.tile_size) as f32 / (self.tiles_y * self.tile_size) as f32 * 2.0 - 1.0;
                let y1 = ((ty + 1) * self.tile_size) as f32 / (self.tiles_y * self.tile_size) as f32 * 2.0 - 1.0;

                // Convert NDC corners to view space.
                let corners_ndc = [
                    Vec4::new(x0, y0, 0.0, 1.0),
                    Vec4::new(x1, y0, 0.0, 1.0),
                    Vec4::new(x1, y1, 0.0, 1.0),
                    Vec4::new(x0, y1, 0.0, 1.0),
                ];
                let corners_vs: [Vec3; 4] = corners_ndc.map(|c| {
                    let v = inv_proj.mul_vec4(c);
                    let w_inv = if v.w.abs() > 1e-12 { 1.0 / v.w } else { 1.0 };
                    Vec3::new(v.x * w_inv, v.y * w_inv, v.z * w_inv).normalize()
                });

                // Tile frustum with depth range.
                let tile_min_depth = if depth_range.is_empty() { near } else { depth_range.min_depth.max(near) };
                let tile_max_depth = if depth_range.is_empty() { far } else { depth_range.max_depth.min(far) };

                let tile_frustum = TileFrustum::from_tile_corners(
                    corners_vs,
                    tile_min_depth,
                    tile_max_depth,
                    Vec3::new(0.0, 0.0, -1.0),
                );

                // Transparent frustum uses near..far.
                let transparent_frustum = if self.config.transparent_light_assignment {
                    Some(TileFrustum::from_tile_corners(
                        corners_vs,
                        near,
                        far,
                        Vec3::new(0.0, 0.0, -1.0),
                    ))
                } else {
                    None
                };

                // Test each point/spot light against the tile frustum.
                for &light_idx in &point_spot_indices {
                    let light = &lights[light_idx as usize];
                    let pos_vs = light_positions_vs[light_idx as usize];
                    let radius = light.range;

                    // Simple sphere-frustum test for opaque.
                    let hits_opaque = match light.kind {
                        LightKind::Spot => {
                            let dir_vs = view_matrix.mul_vec4(Vec4::from_vec3(light.direction, 0.0)).xyz();
                            tile_frustum.test_cone_as_sphere(pos_vs, dir_vs, radius, light.outer_angle)
                        }
                        _ => tile_frustum.test_sphere(pos_vs, radius),
                    };

                    if hits_opaque {
                        if assignment.opaque_lights.len() < self.config.max_lights_per_tile as usize {
                            assignment.opaque_lights.push(light_idx);
                        }
                    }

                    // Test against transparent frustum.
                    if let Some(ref tf) = transparent_frustum {
                        let hits_transparent = match light.kind {
                            LightKind::Spot => {
                                let dir_vs = view_matrix.mul_vec4(Vec4::from_vec3(light.direction, 0.0)).xyz();
                                tf.test_cone_as_sphere(pos_vs, dir_vs, radius, light.outer_angle)
                            }
                            _ => tf.test_sphere(pos_vs, radius),
                        };
                        if hits_transparent {
                            if assignment.transparent_lights.len() < self.config.max_lights_per_tile as usize {
                                assignment.transparent_lights.push(light_idx);
                            }
                        }
                    }
                }
            }
        }

        // Compute statistics.
        let total_tiles = self.tiles_x * self.tiles_y;
        let mut min_l = u32::MAX;
        let mut max_l = 0u32;
        let mut total_l = 0u64;
        let mut at_max = 0u32;
        let mut empty = 0u32;
        for a in &self.assignments {
            let c = a.opaque_lights.len() as u32;
            min_l = min_l.min(c);
            max_l = max_l.max(c);
            total_l += c as u64;
            if c >= self.config.max_lights_per_tile { at_max += 1; }
            if c == 0 { empty += 1; }
        }
        if total_tiles == 0 { min_l = 0; }

        self.stats = LightAssignStats {
            total_lights: lights.len() as u32,
            total_tiles,
            min_lights_per_tile: min_l,
            max_lights_per_tile: max_l,
            avg_lights_per_tile: if total_tiles > 0 { total_l as f32 / total_tiles as f32 } else { 0.0 },
            tiles_at_max: at_max,
            empty_tiles: empty,
            directional_lights: dir_count,
            point_lights: point_count,
            spot_lights: spot_count,
            time_microseconds: start.elapsed().as_micros() as u64,
        };
    }

    /// Build the SSBO buffer from current assignments.
    pub fn build_ssbo(&self) -> TileLightListBuffer {
        let mut buffer = TileLightListBuffer::new(
            self.tiles_x,
            self.tiles_y,
            self.config.max_lights_per_tile,
        );
        for a in &self.assignments {
            buffer.write_tile(a.tile_x, a.tile_y, &a.opaque_lights);
        }
        buffer
    }

    /// Resize for a new resolution.
    pub fn resize(&mut self, tiles_x: u32, tiles_y: u32) {
        self.tiles_x = tiles_x;
        self.tiles_y = tiles_y;
        let tile_count = (tiles_x * tiles_y) as usize;
        self.assignments.clear();
        self.assignments.reserve(tile_count);
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                self.assignments.push(TileLightAssignment {
                    tile_x: tx,
                    tile_y: ty,
                    opaque_lights: Vec::new(),
                    transparent_lights: Vec::new(),
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Forward+ pipeline
// ---------------------------------------------------------------------------

/// Configuration for the entire Forward+ pipeline.
#[derive(Debug, Clone)]
pub struct ForwardPlusConfig {
    /// Tile size in pixels.
    pub tile_size: u32,
    /// Depth prepass configuration.
    pub depth_prepass: DepthPrepassConfig,
    /// Light assignment configuration.
    pub light_assign: LightAssignConfig,
    /// Whether to enable debug visualization.
    pub debug_visualization: bool,
    /// Debug visualization mode.
    pub debug_mode: DebugVisualizationMode,
    /// Whether to use clustered tiling (3D tiles) instead of 2D tiles.
    pub clustered: bool,
    /// Number of depth slices for clustered mode.
    pub depth_slices: u32,
    /// Enable transparent object support.
    pub transparent_support: bool,
    /// Whether to sort transparent objects back-to-front.
    pub sort_transparent: bool,
}

impl Default for ForwardPlusConfig {
    fn default() -> Self {
        Self {
            tile_size: DEFAULT_TILE_SIZE,
            depth_prepass: DepthPrepassConfig::default(),
            light_assign: LightAssignConfig::default(),
            debug_visualization: false,
            debug_mode: DebugVisualizationMode::LightCount,
            clustered: false,
            depth_slices: 24,
            transparent_support: true,
            sort_transparent: true,
        }
    }
}

/// Debug visualization modes for the Forward+ pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugVisualizationMode {
    /// Heat map showing light count per tile (blue = few, red = many).
    LightCount,
    /// Show tile boundaries as a grid overlay.
    TileGrid,
    /// Show per-tile depth range as grayscale.
    DepthRange,
    /// Show which lights affect the tile under the cursor.
    SelectedTileLights,
    /// Show overdraw (number of fragments per pixel).
    Overdraw,
}

/// A renderable object in the Forward+ pipeline.
#[derive(Debug, Clone)]
pub struct ForwardPlusDrawCall {
    pub mesh_id: u64,
    pub material_id: u64,
    pub model_matrix: Mat4,
    pub bounding_sphere_center: Vec3,
    pub bounding_sphere_radius: f32,
    pub is_transparent: bool,
    pub sort_key: u64,
    pub cast_shadow: bool,
    pub layer_mask: u32,
    /// Distance from camera (computed during sorting).
    pub camera_distance: f32,
}

/// The Forward+ rendering pipeline.
///
/// Orchestrates the depth prepass, light assignment, forward render, and
/// transparent passes.
pub struct ForwardPlusPipeline {
    pub config: ForwardPlusConfig,
    pub width: u32,
    pub height: u32,
    pub depth_processor: DepthPrepassProcessor,
    pub light_processor: LightAssignProcessor,
    pub lights: Vec<ForwardPlusLight>,
    pub gpu_light_buffer: Vec<GpuLightData>,
    pub tile_light_buffer: TileLightListBuffer,
    pub opaque_queue: Vec<ForwardPlusDrawCall>,
    pub transparent_queue: Vec<ForwardPlusDrawCall>,
    pub shadow_queue: Vec<ForwardPlusDrawCall>,
    pub frame_stats: ForwardPlusFrameStats,
    next_light_id: LightId,
    light_map: HashMap<LightId, usize>,
}

/// Per-frame statistics for the Forward+ pipeline.
#[derive(Debug, Clone, Default)]
pub struct ForwardPlusFrameStats {
    pub depth_prepass_time_us: u64,
    pub light_assign_time_us: u64,
    pub opaque_render_time_us: u64,
    pub transparent_render_time_us: u64,
    pub total_draw_calls: u32,
    pub opaque_draw_calls: u32,
    pub transparent_draw_calls: u32,
    pub shadow_draw_calls: u32,
    pub total_lights: u32,
    pub active_lights: u32,
    pub light_assign_stats: LightAssignStats,
    pub tile_light_buffer_bytes: usize,
    pub gpu_light_buffer_bytes: usize,
}

impl ForwardPlusPipeline {
    /// Create a new Forward+ pipeline for the given resolution.
    pub fn new(width: u32, height: u32, config: ForwardPlusConfig) -> Self {
        let tile_size = config.tile_size;
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;

        let depth_processor = DepthPrepassProcessor::new(
            width, height, tile_size, config.depth_prepass.clone(),
        );
        let light_processor = LightAssignProcessor::new(
            tiles_x, tiles_y, tile_size, config.light_assign.clone(),
        );
        let tile_light_buffer = TileLightListBuffer::new(
            tiles_x, tiles_y, config.light_assign.max_lights_per_tile,
        );

        Self {
            config,
            width,
            height,
            depth_processor,
            light_processor,
            lights: Vec::new(),
            gpu_light_buffer: Vec::new(),
            tile_light_buffer,
            opaque_queue: Vec::new(),
            transparent_queue: Vec::new(),
            shadow_queue: Vec::new(),
            frame_stats: ForwardPlusFrameStats::default(),
            next_light_id: 1,
            light_map: HashMap::new(),
        }
    }

    /// Add a light to the pipeline and return its ID.
    pub fn add_light(&mut self, mut light: ForwardPlusLight) -> LightId {
        let id = self.next_light_id;
        self.next_light_id += 1;
        light.id = id;
        let index = self.lights.len();
        self.light_map.insert(id, index);
        self.lights.push(light);
        id
    }

    /// Remove a light by ID.
    pub fn remove_light(&mut self, id: LightId) -> bool {
        if let Some(index) = self.light_map.remove(&id) {
            self.lights.swap_remove(index);
            // Update the map for the swapped light.
            if index < self.lights.len() {
                let swapped_id = self.lights[index].id;
                self.light_map.insert(swapped_id, index);
            }
            true
        } else {
            false
        }
    }

    /// Get a mutable reference to a light by ID.
    pub fn get_light_mut(&mut self, id: LightId) -> Option<&mut ForwardPlusLight> {
        self.light_map.get(&id).copied().map(|i| &mut self.lights[i])
    }

    /// Get an immutable reference to a light by ID.
    pub fn get_light(&self, id: LightId) -> Option<&ForwardPlusLight> {
        self.light_map.get(&id).copied().map(|i| &self.lights[i])
    }

    /// Submit a draw call to the appropriate queue.
    pub fn submit_draw_call(&mut self, draw_call: ForwardPlusDrawCall) {
        if draw_call.is_transparent {
            self.transparent_queue.push(draw_call);
        } else {
            self.opaque_queue.push(draw_call);
        }
    }

    /// Submit a shadow caster draw call.
    pub fn submit_shadow_caster(&mut self, draw_call: ForwardPlusDrawCall) {
        self.shadow_queue.push(draw_call);
    }

    /// Clear all queues for a new frame.
    pub fn begin_frame(&mut self) {
        self.opaque_queue.clear();
        self.transparent_queue.clear();
        self.shadow_queue.clear();
        self.frame_stats = ForwardPlusFrameStats::default();
    }

    /// Sort the render queues.
    ///
    /// Opaque objects are sorted front-to-back by distance to minimize overdraw.
    /// Transparent objects are sorted back-to-front for correct blending.
    pub fn sort_queues(&mut self, camera_position: Vec3) {
        // Compute camera distances.
        for dc in &mut self.opaque_queue {
            dc.camera_distance = camera_position.distance(dc.bounding_sphere_center);
        }
        for dc in &mut self.transparent_queue {
            dc.camera_distance = camera_position.distance(dc.bounding_sphere_center);
        }

        // Opaque: front-to-back (ascending distance).
        self.opaque_queue.sort_by(|a, b| {
            a.camera_distance.partial_cmp(&b.camera_distance).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Transparent: back-to-front (descending distance).
        if self.config.sort_transparent {
            self.transparent_queue.sort_by(|a, b| {
                b.camera_distance.partial_cmp(&a.camera_distance).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    /// Execute the full Forward+ pipeline for one frame.
    ///
    /// This performs:
    /// 1. Depth prepass (using the provided depth buffer)
    /// 2. Light assignment
    /// 3. SSBO construction
    /// 4. Queue sorting
    ///
    /// The actual GPU rendering commands would be issued by the caller using
    /// the results of this method.
    pub fn execute_frame(
        &mut self,
        depth_buffer: &[f32],
        view_matrix: &Mat4,
        proj_matrix: &Mat4,
        camera_position: Vec3,
        near: f32,
        far: f32,
    ) -> &ForwardPlusFrameStats {
        // 1. Depth prepass.
        let dp_start = std::time::Instant::now();
        if self.config.depth_prepass.enabled {
            self.depth_processor.process_depth_buffer_linearized(depth_buffer, near, far);
        }
        let dp_time = dp_start.elapsed().as_micros() as u64;

        // 2. Light assignment.
        let la_start = std::time::Instant::now();
        self.light_processor.assign(
            &self.lights,
            &self.depth_processor.tile_depths,
            view_matrix,
            proj_matrix,
            near,
            far,
        );
        let la_time = la_start.elapsed().as_micros() as u64;

        // 3. Build GPU buffers.
        self.gpu_light_buffer.clear();
        for light in &self.lights {
            self.gpu_light_buffer.push(light.to_gpu_data());
        }
        self.tile_light_buffer = self.light_processor.build_ssbo();

        // 4. Sort queues.
        self.sort_queues(camera_position);

        // 5. Update stats.
        self.frame_stats = ForwardPlusFrameStats {
            depth_prepass_time_us: dp_time,
            light_assign_time_us: la_time,
            opaque_render_time_us: 0,
            transparent_render_time_us: 0,
            total_draw_calls: (self.opaque_queue.len() + self.transparent_queue.len() + self.shadow_queue.len()) as u32,
            opaque_draw_calls: self.opaque_queue.len() as u32,
            transparent_draw_calls: self.transparent_queue.len() as u32,
            shadow_draw_calls: self.shadow_queue.len() as u32,
            total_lights: self.lights.len() as u32,
            active_lights: self.lights.iter().filter(|l| l.enabled).count() as u32,
            light_assign_stats: self.light_processor.stats.clone(),
            tile_light_buffer_bytes: self.tile_light_buffer.byte_size(),
            gpu_light_buffer_bytes: self.gpu_light_buffer.len() * std::mem::size_of::<GpuLightData>(),
        };

        &self.frame_stats
    }

    /// Resize the pipeline for a new resolution.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let tiles_x = (width + self.config.tile_size - 1) / self.config.tile_size;
        let tiles_y = (height + self.config.tile_size - 1) / self.config.tile_size;
        self.depth_processor.resize(width, height);
        self.light_processor.resize(tiles_x, tiles_y);
        self.tile_light_buffer = TileLightListBuffer::new(
            tiles_x, tiles_y, self.config.light_assign.max_lights_per_tile,
        );
    }

    /// Get the number of tiles in each dimension.
    pub fn tile_count(&self) -> (u32, u32) {
        (self.depth_processor.tiles_x, self.depth_processor.tiles_y)
    }

    /// Get the total number of tiles.
    pub fn total_tiles(&self) -> u32 {
        self.depth_processor.tiles_x * self.depth_processor.tiles_y
    }
}

// ---------------------------------------------------------------------------
// Debug visualization
// ---------------------------------------------------------------------------

/// A debug pixel for tile visualization.
#[derive(Debug, Clone, Copy)]
pub struct DebugPixel {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl DebugPixel {
    pub const TRANSPARENT: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self { r, g, b, a } }
}

/// Generates debug visualization overlays for the Forward+ pipeline.
pub struct ForwardPlusDebugVis {
    pub mode: DebugVisualizationMode,
    pub overlay_alpha: f32,
    pub max_lights_for_heatmap: u32,
    pub grid_color: [f32; 4],
    pub grid_width: u32,
}

impl Default for ForwardPlusDebugVis {
    fn default() -> Self {
        Self {
            mode: DebugVisualizationMode::LightCount,
            overlay_alpha: 0.5,
            max_lights_for_heatmap: 64,
            grid_color: [1.0, 1.0, 1.0, 0.3],
            grid_width: 1,
        }
    }
}

impl ForwardPlusDebugVis {
    /// Generate a heat-map color for a light count.
    /// 0 lights = blue, mid = green, max = red.
    pub fn heat_map_color(&self, light_count: u32) -> DebugPixel {
        let t = (light_count as f32 / self.max_lights_for_heatmap as f32).clamp(0.0, 1.0);
        let (r, g, b) = if t < 0.5 {
            let s = t * 2.0;
            (0.0, s, 1.0 - s)
        } else {
            let s = (t - 0.5) * 2.0;
            (s, 1.0 - s, 0.0)
        };
        DebugPixel::new(r, g, b, self.overlay_alpha)
    }

    /// Generate a full debug overlay image.
    pub fn generate_overlay(
        &self,
        pipeline: &ForwardPlusPipeline,
        width: u32,
        height: u32,
    ) -> Vec<DebugPixel> {
        let pixel_count = (width * height) as usize;
        let mut pixels = vec![DebugPixel::TRANSPARENT; pixel_count];
        let tiles_x = pipeline.depth_processor.tiles_x;
        let tiles_y = pipeline.depth_processor.tiles_y;
        let tile_size = pipeline.config.tile_size;

        match self.mode {
            DebugVisualizationMode::LightCount => {
                for ty in 0..tiles_y {
                    for tx in 0..tiles_x {
                        let tile_index = (ty * tiles_x + tx) as usize;
                        if tile_index >= pipeline.light_processor.assignments.len() { continue; }
                        let count = pipeline.light_processor.assignments[tile_index].opaque_lights.len() as u32;
                        let color = self.heat_map_color(count);

                        let px_start_x = tx * tile_size;
                        let px_start_y = ty * tile_size;
                        let px_end_x = ((tx + 1) * tile_size).min(width);
                        let px_end_y = ((ty + 1) * tile_size).min(height);

                        for py in px_start_y..px_end_y {
                            for px in px_start_x..px_end_x {
                                let idx = (py * width + px) as usize;
                                if idx < pixel_count {
                                    pixels[idx] = color;
                                }
                            }
                        }
                    }
                }
            }
            DebugVisualizationMode::TileGrid => {
                for y in 0..height {
                    for x in 0..width {
                        let on_grid = (x % tile_size < self.grid_width) || (y % tile_size < self.grid_width);
                        if on_grid {
                            let idx = (y * width + x) as usize;
                            pixels[idx] = DebugPixel::new(
                                self.grid_color[0],
                                self.grid_color[1],
                                self.grid_color[2],
                                self.grid_color[3],
                            );
                        }
                    }
                }
            }
            DebugVisualizationMode::DepthRange => {
                for ty in 0..tiles_y {
                    for tx in 0..tiles_x {
                        let tile_index = (ty * tiles_x + tx) as usize;
                        let depth = pipeline.depth_processor.tile_depths[tile_index];
                        let intensity = if depth.is_empty() { 0.0 } else { depth.max_depth.clamp(0.0, 1.0) };
                        let color = DebugPixel::new(intensity, intensity, intensity, self.overlay_alpha);

                        let px_start_x = tx * tile_size;
                        let px_start_y = ty * tile_size;
                        let px_end_x = ((tx + 1) * tile_size).min(width);
                        let px_end_y = ((ty + 1) * tile_size).min(height);

                        for py in px_start_y..px_end_y {
                            for px in px_start_x..px_end_x {
                                let idx = (py * width + px) as usize;
                                if idx < pixel_count {
                                    pixels[idx] = color;
                                }
                            }
                        }
                    }
                }
            }
            DebugVisualizationMode::SelectedTileLights | DebugVisualizationMode::Overdraw => {
                // These require additional per-pixel data not available here.
                // Fill with a placeholder pattern.
                for y in 0..height {
                    for x in 0..width {
                        let checker = ((x / tile_size) + (y / tile_size)) % 2 == 0;
                        if checker {
                            let idx = (y * width + x) as usize;
                            pixels[idx] = DebugPixel::new(0.1, 0.1, 0.1, 0.2);
                        }
                    }
                }
            }
        }

        pixels
    }

    /// Get text description of a specific tile for debug UI.
    pub fn describe_tile(&self, pipeline: &ForwardPlusPipeline, tile_x: u32, tile_y: u32) -> String {
        let tiles_x = pipeline.depth_processor.tiles_x;
        let tile_index = (tile_y * tiles_x + tile_x) as usize;
        if tile_index >= pipeline.light_processor.assignments.len() {
            return "Invalid tile".to_string();
        }
        let assignment = &pipeline.light_processor.assignments[tile_index];
        let depth = pipeline.depth_processor.tile_depths[tile_index];

        format!(
            "Tile ({}, {}): {} opaque lights, {} transparent lights, depth [{:.4}, {:.4}]",
            tile_x, tile_y,
            assignment.opaque_lights.len(),
            assignment.transparent_lights.len(),
            depth.min_depth, depth.max_depth,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_light_buffer_read_write() {
        let mut buf = TileLightListBuffer::new(4, 4, 8);
        buf.write_tile(1, 2, &[3, 7, 11]);
        assert_eq!(buf.read_tile_count(1, 2), 3);
        assert_eq!(buf.read_tile_lights(1, 2), &[3, 7, 11]);
        assert_eq!(buf.read_tile_count(0, 0), 0);
    }

    #[test]
    fn test_depth_prepass_processor() {
        let config = DepthPrepassConfig::default();
        let mut proc = DepthPrepassProcessor::new(32, 32, 16, config);
        let mut depth = vec![0.5f32; 32 * 32];
        // Make one tile have different depth.
        for y in 0..16 {
            for x in 0..16 {
                depth[y * 32 + x] = 0.1;
            }
        }
        proc.process_depth_buffer(&depth);
        let d00 = proc.tile_depth(0, 0);
        assert!((d00.min_depth - 0.1).abs() < 1e-6);
        assert!((d00.max_depth - 0.1).abs() < 1e-6);
        let d10 = proc.tile_depth(1, 0);
        assert!((d10.min_depth - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_add_remove_light() {
        let mut pipeline = ForwardPlusPipeline::new(320, 240, ForwardPlusConfig::default());
        let id1 = pipeline.add_light(ForwardPlusLight::point(0, Vec3::ZERO, Vec3::ONE, 1.0, 10.0));
        let id2 = pipeline.add_light(ForwardPlusLight::point(0, Vec3::new(5.0, 0.0, 0.0), Vec3::ONE, 2.0, 20.0));
        assert_eq!(pipeline.lights.len(), 2);
        assert!(pipeline.remove_light(id1));
        assert_eq!(pipeline.lights.len(), 1);
        assert!(pipeline.get_light(id2).is_some());
    }

    #[test]
    fn test_heat_map_color() {
        let vis = ForwardPlusDebugVis::default();
        let c0 = vis.heat_map_color(0);
        assert!(c0.b > 0.5); // Low count = blue.
        let c_max = vis.heat_map_color(64);
        assert!(c_max.r > 0.5); // Max count = red.
    }

    #[test]
    fn test_frustum_sphere_intersection() {
        let frustum = TileFrustum::from_tile_corners(
            [
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(1.0, 1.0, -1.0),
                Vec3::new(-1.0, 1.0, -1.0),
            ],
            0.1,
            100.0,
            Vec3::new(0.0, 0.0, -1.0),
        );
        // Sphere at origin with large radius should intersect.
        assert!(frustum.test_sphere(Vec3::new(0.0, 0.0, -5.0), 10.0));
    }

    #[test]
    fn test_directional_light_in_all_tiles() {
        let mut pipeline = ForwardPlusPipeline::new(64, 64, ForwardPlusConfig::default());
        pipeline.add_light(ForwardPlusLight::directional(
            0,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::ONE,
            1.0,
        ));
        let depth = vec![0.5f32; 64 * 64];
        let view = Mat4::IDENTITY;
        let proj = Mat4::perspective(1.0, 1.0, 0.1, 100.0);
        pipeline.execute_frame(&depth, &view, &proj, Vec3::ZERO, 0.1, 100.0);

        // Every tile should have the directional light.
        for a in &pipeline.light_processor.assignments {
            assert!(!a.opaque_lights.is_empty(), "Tile ({}, {}) has no lights", a.tile_x, a.tile_y);
        }
    }

    #[test]
    fn test_mat4_inverse() {
        let m = Mat4::perspective(1.0, 1.33, 0.1, 100.0);
        let inv = m.inverse().expect("Perspective matrix should be invertible");
        let identity = m.mul_mat4(&inv);
        // Check diagonal is approximately 1.
        for i in 0..4 {
            assert!((identity.cols[i][i] - 1.0).abs() < 1e-4, "col[{}][{}] = {}", i, i, identity.cols[i][i]);
        }
    }
}
