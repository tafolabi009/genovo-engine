// engine/world/src/occlusion.rs
//
// Software occlusion culling system for the Genovo engine.
//
// Provides CPU-side occlusion culling for large open worlds using a low-resolution
// software depth buffer, plus an indoor portal-based visibility system for
// architectural environments with rooms connected by doorways/windows.
//
// # Architecture
//
// The occlusion pipeline runs *before* GPU submission each frame:
// 1. Select large, static occluder meshes near the camera.
// 2. Rasterize their simplified bounding boxes into a small (e.g., 256x144)
//    software depth buffer using fixed-point scanline rasterization.
// 3. Test every renderable AABB against the depth buffer; skip draw calls for
//    fully occluded objects.
//
// The portal system works independently for indoor scenes:
// 1. Determine which room the camera is in.
// 2. Recursively traverse portals, narrowing the view frustum at each portal.
// 3. Collect visible rooms; only render objects belonging to visible rooms.

use glam::{Mat4, Vec2, Vec3, Vec4};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default software depth buffer width.
pub const DEFAULT_BUFFER_WIDTH: usize = 256;
/// Default software depth buffer height.
pub const DEFAULT_BUFFER_HEIGHT: usize = 144;

/// Fixed-point fractional bits for sub-pixel precision during rasterization.
const FIXED_POINT_SHIFT: i32 = 4;
/// Fixed-point multiplier: 1 pixel = (1 << FIXED_POINT_SHIFT) sub-pixels.
const FIXED_POINT_SCALE: f32 = (1 << FIXED_POINT_SHIFT) as f32;

/// Maximum number of occluders to rasterize per frame (performance budget).
const MAX_OCCLUDERS_PER_FRAME: usize = 128;

/// Maximum number of renderables that can be tested per frame.
const MAX_TESTABLE_PER_FRAME: usize = 16384;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-7;

// ---------------------------------------------------------------------------
// AABB (local definition matching core::math::AABB)
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box, mirroring `core::math::AABB` for standalone use
/// within this module without requiring a crate dependency at the type level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
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

    /// Surface area of the AABB (for occluder prioritization).
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Volume of the AABB.
    #[inline]
    pub fn volume(&self) -> f32 {
        let d = self.max - self.min;
        d.x * d.y * d.z
    }

    /// Returns the 8 corners of the AABB.
    pub fn corners(&self) -> [Vec3; 8] {
        [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
        ]
    }
}

// ===========================================================================
// OcclusionBuffer -- software-rasterized depth buffer
// ===========================================================================

/// A low-resolution software depth buffer for CPU-side occlusion testing.
///
/// The buffer stores depth values as `f32` in the range [0, 1] where 0 is the
/// near plane and 1 is the far plane. Each cell stores the *closest* depth
/// written by an occluder. When testing a renderable, we check whether all of
/// its screen-space footprint is behind the stored depth values.
///
/// Rasterization uses scanline traversal with fixed-point edge walking for
/// sub-pixel accuracy and consistent results independent of triangle
/// orientation.
#[derive(Debug, Clone)]
pub struct OcclusionBuffer {
    /// Depth values (row-major, top-to-bottom). Initialized to 1.0 (far plane).
    depth: Vec<f32>,
    /// Buffer width in pixels.
    width: usize,
    /// Buffer height in pixels.
    height: usize,
    /// Reciprocal width for NDC-to-pixel conversion.
    inv_width: f32,
    /// Reciprocal height for NDC-to-pixel conversion.
    inv_height: f32,
    /// Number of triangles rasterized this frame (stats).
    triangles_rasterized: u32,
    /// Number of pixels written this frame (stats).
    pixels_written: u64,
}

/// Statistics for one frame of occlusion buffer usage.
#[derive(Debug, Clone, Copy, Default)]
pub struct OcclusionBufferStats {
    pub triangles_rasterized: u32,
    pub pixels_written: u64,
    pub buffer_width: usize,
    pub buffer_height: usize,
}

impl OcclusionBuffer {
    /// Creates a new occlusion buffer of the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0, "OcclusionBuffer dimensions must be > 0");
        Self {
            depth: vec![1.0; width * height],
            width,
            height,
            inv_width: 1.0 / width as f32,
            inv_height: 1.0 / height as f32,
            triangles_rasterized: 0,
            pixels_written: 0,
        }
    }

    /// Creates a buffer with the default dimensions (256x144).
    pub fn default_size() -> Self {
        Self::new(DEFAULT_BUFFER_WIDTH, DEFAULT_BUFFER_HEIGHT)
    }

    /// Returns the buffer width.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the buffer height.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Clears the depth buffer to 1.0 (far plane) and resets statistics.
    pub fn clear(&mut self) {
        for d in self.depth.iter_mut() {
            *d = 1.0;
        }
        self.triangles_rasterized = 0;
        self.pixels_written = 0;
    }

    /// Returns current frame statistics.
    pub fn stats(&self) -> OcclusionBufferStats {
        OcclusionBufferStats {
            triangles_rasterized: self.triangles_rasterized,
            pixels_written: self.pixels_written,
            buffer_width: self.width,
            buffer_height: self.height,
        }
    }

    /// Reads the depth value at a specific pixel. Returns 1.0 if out of bounds.
    #[inline]
    pub fn sample(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.depth[y * self.width + x]
        } else {
            1.0
        }
    }

    /// Writes a depth value at a specific pixel if it is closer than what is
    /// already stored (depth test: less-or-equal).
    #[inline]
    fn write_depth(&mut self, x: usize, y: usize, z: f32) {
        debug_assert!(x < self.width && y < self.height);
        let idx = y * self.width + x;
        if z <= self.depth[idx] {
            self.depth[idx] = z;
            self.pixels_written += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Triangle rasterization (scanline, fixed-point)
    // -----------------------------------------------------------------------

    /// Rasterizes a triangle into the depth buffer.
    ///
    /// The three vertices `v0`, `v1`, `v2` must be in *screen space*:
    /// - x in `[0, width)`
    /// - y in `[0, height)`
    /// - z in `[0, 1]` (depth, 0 = near)
    ///
    /// Uses a top-left fill rule and fixed-point edge walking for consistent
    /// results.
    pub fn rasterize_triangle(&mut self, v0: Vec3, v1: Vec3, v2: Vec3) {
        self.triangles_rasterized += 1;

        // Sort vertices by y (top-to-bottom). We work in fixed-point to avoid
        // sub-pixel seams.
        let mut verts = [v0, v1, v2];
        // Simple insertion sort (3 elements).
        if verts[0].y > verts[1].y {
            verts.swap(0, 1);
        }
        if verts[1].y > verts[2].y {
            verts.swap(1, 2);
        }
        if verts[0].y > verts[1].y {
            verts.swap(0, 1);
        }

        let [top, mid, bot] = verts;

        // Convert to fixed-point sub-pixel coordinates.
        let top_y_fp = (top.y * FIXED_POINT_SCALE) as i32;
        let mid_y_fp = (mid.y * FIXED_POINT_SCALE) as i32;
        let bot_y_fp = (bot.y * FIXED_POINT_SCALE) as i32;

        // Degenerate triangle check.
        if top_y_fp == bot_y_fp {
            return;
        }

        // Scan the upper half: top -> mid.
        if top_y_fp != mid_y_fp {
            self.rasterize_half(top, mid, bot, top_y_fp, mid_y_fp, true);
        }

        // Scan the lower half: mid -> bot.
        if mid_y_fp != bot_y_fp {
            self.rasterize_half(top, mid, bot, mid_y_fp, bot_y_fp, false);
        }
    }

    /// Rasterizes one half of a triangle (upper or lower) using scanline
    /// traversal with linearly interpolated edge positions and depth.
    fn rasterize_half(
        &mut self,
        top: Vec3,
        mid: Vec3,
        bot: Vec3,
        y_start_fp: i32,
        y_end_fp: i32,
        is_upper: bool,
    ) {
        let height = self.height as i32;
        let width = self.width;

        // Compute the starting and ending scanline indices (pixel rows).
        let y_start = (y_start_fp + (1 << FIXED_POINT_SHIFT) - 1) >> FIXED_POINT_SHIFT;
        let y_end = (y_end_fp + (1 << FIXED_POINT_SHIFT) - 1) >> FIXED_POINT_SHIFT;

        if y_start >= height || y_end <= 0 {
            return;
        }

        let y_start = y_start.max(0) as usize;
        let y_end = (y_end.min(height)) as usize;

        // The long edge always goes from top to bot.
        let long_dy = bot.y - top.y;
        if long_dy.abs() < EPSILON {
            return;
        }
        let long_inv_dy = 1.0 / long_dy;

        // The short edge goes from top to mid (upper half) or mid to bot (lower half).
        let (short_start, short_end) = if is_upper {
            (top, mid)
        } else {
            (mid, bot)
        };
        let short_dy = short_end.y - short_start.y;
        if short_dy.abs() < EPSILON {
            return;
        }
        let short_inv_dy = 1.0 / short_dy;

        for y in y_start..y_end {
            let yf = y as f32 + 0.5;

            // Interpolate x and z along the long edge.
            let t_long = (yf - top.y) * long_inv_dy;
            let t_long = t_long.clamp(0.0, 1.0);
            let x_long = top.x + (bot.x - top.x) * t_long;
            let z_long = top.z + (bot.z - top.z) * t_long;

            // Interpolate x and z along the short edge.
            let t_short = (yf - short_start.y) * short_inv_dy;
            let t_short = t_short.clamp(0.0, 1.0);
            let x_short = short_start.x + (short_end.x - short_start.x) * t_short;
            let z_short = short_start.z + (short_end.z - short_start.z) * t_short;

            // Determine left and right endpoints for this scanline.
            let (mut xl, mut zl, mut xr, mut zr) = if x_long < x_short {
                (x_long, z_long, x_short, z_short)
            } else {
                (x_short, z_short, x_long, z_long)
            };

            // Clamp to buffer width.
            let px_left = (xl as i32).max(0) as usize;
            let px_right = ((xr as i32) + 1).min(width as i32) as usize;

            if px_left >= width || px_right == 0 || px_left >= px_right {
                continue;
            }

            // Interpolate depth across the scanline.
            let span = xr - xl;
            if span.abs() < EPSILON {
                // Single-pixel span.
                let z = zl.min(zr).clamp(0.0, 1.0);
                if px_left < width {
                    self.write_depth(px_left, y, z);
                }
                continue;
            }

            let inv_span = 1.0 / span;

            for px in px_left..px_right {
                let t = ((px as f32 + 0.5) - xl) * inv_span;
                let t = t.clamp(0.0, 1.0);
                let z = (zl + (zr - zl) * t).clamp(0.0, 1.0);
                self.write_depth(px, y, z);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Clip-space triangle rasterization (projects from clip space)
    // -----------------------------------------------------------------------

    /// Rasterizes a triangle given in *clip space* (post-MVP).
    ///
    /// Performs perspective divide, viewport transform, back-face culling, and
    /// near-plane clipping before dispatching to the scanline rasterizer.
    pub fn rasterize_triangle_clip(&mut self, c0: Vec4, c1: Vec4, c2: Vec4) {
        // Near-plane clipping: discard vertices behind the near plane.
        // A vertex is behind the near plane if w <= 0.
        let in_front = [c0.w > EPSILON, c1.w > EPSILON, c2.w > EPSILON];
        let count_in = in_front.iter().filter(|&&b| b).count();

        if count_in == 0 {
            // Entire triangle behind near plane.
            return;
        }

        if count_in == 3 {
            // All vertices in front -- rasterize directly.
            let s0 = self.ndc_to_screen(c0);
            let s1 = self.ndc_to_screen(c1);
            let s2 = self.ndc_to_screen(c2);
            // Back-face cull (CW winding in screen space = back-facing).
            if Self::edge_function_2d(s0, s1, s2) <= 0.0 {
                return;
            }
            self.rasterize_triangle(s0, s1, s2);
            return;
        }

        // Partial clipping: 1 or 2 vertices behind near plane.
        // Clip against w = EPSILON plane. This produces 1 or 2 triangles.
        let clips = [c0, c1, c2];
        let mut out_verts: [Vec4; 4] = [Vec4::ZERO; 4];
        let mut out_count = 0usize;

        for i in 0..3 {
            let curr = clips[i];
            let next = clips[(i + 1) % 3];
            let curr_in = in_front[i];
            let next_in = in_front[(i + 1) % 3];

            if curr_in {
                out_verts[out_count] = curr;
                out_count += 1;
            }

            if curr_in != next_in {
                // Edge crosses the near plane -- compute intersection.
                let t = (EPSILON - curr.w) / (next.w - curr.w);
                let clipped = curr + (next - curr) * t;
                out_verts[out_count] = clipped;
                out_count += 1;
            }
        }

        // Fan-triangulate the clipped polygon (3 or 4 vertices).
        if out_count >= 3 {
            let s: Vec<Vec3> = out_verts[..out_count]
                .iter()
                .map(|v| self.ndc_to_screen(*v))
                .collect();

            for i in 1..out_count - 1 {
                if Self::edge_function_2d(s[0], s[i], s[i + 1]) > 0.0 {
                    self.rasterize_triangle(s[0], s[i], s[i + 1]);
                }
            }
        }
    }

    /// Converts a clip-space vertex to screen-space (pixel coordinates + depth).
    #[inline]
    fn ndc_to_screen(&self, clip: Vec4) -> Vec3 {
        let inv_w = 1.0 / clip.w;
        let ndc_x = clip.x * inv_w;
        let ndc_y = clip.y * inv_w;
        let ndc_z = clip.z * inv_w;

        // NDC is in [-1, 1] for x and y. Map to pixel coordinates.
        // Note: y is flipped (NDC +y is up, screen +y is down).
        let sx = (ndc_x * 0.5 + 0.5) * self.width as f32;
        let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * self.height as f32;
        // Depth: map from [-1, 1] to [0, 1].
        let sz = ndc_z * 0.5 + 0.5;

        Vec3::new(sx, sy, sz.clamp(0.0, 1.0))
    }

    /// 2D edge function: positive if p2 is to the left of the edge p0->p1
    /// (counter-clockwise winding). Used for back-face culling.
    #[inline]
    fn edge_function_2d(p0: Vec3, p1: Vec3, p2: Vec3) -> f32 {
        (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
    }

    // -----------------------------------------------------------------------
    // AABB occlusion test
    // -----------------------------------------------------------------------

    /// Tests whether an AABB is visible (not fully occluded) against the depth
    /// buffer.
    ///
    /// The AABB is projected to screen space using the model-view-projection
    /// matrix. If the AABB's screen-space bounding rectangle is entirely behind
    /// the stored depth values, the AABB is considered occluded.
    ///
    /// Returns `true` if the AABB is *visible* (should be rendered).
    pub fn test_aabb(&self, aabb: &AABB, mvp: &Mat4) -> bool {
        // Project all 8 corners of the AABB to clip space.
        let corners = aabb.corners();
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut any_in_front = false;

        for corner in &corners {
            let clip = *mvp * Vec4::new(corner.x, corner.y, corner.z, 1.0);
            if clip.w <= EPSILON {
                // Vertex is behind the near plane -- conservatively mark visible.
                // (If any vertex is behind us, the object may be partially visible.)
                return true;
            }

            let inv_w = 1.0 / clip.w;
            let ndc_x = clip.x * inv_w;
            let ndc_y = clip.y * inv_w;
            let ndc_z = clip.z * inv_w;

            // Map to screen coordinates.
            let sx = (ndc_x * 0.5 + 0.5) * self.width as f32;
            let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * self.height as f32;
            let sz = ndc_z * 0.5 + 0.5;

            min_x = min_x.min(sx);
            max_x = max_x.max(sx);
            min_y = min_y.min(sy);
            max_y = max_y.max(sy);
            min_z = min_z.min(sz);
            any_in_front = true;
        }

        if !any_in_front {
            return true; // Degenerate -- assume visible.
        }

        // Clamp to buffer bounds.
        let px_left = (min_x as i32).max(0) as usize;
        let px_right = ((max_x as i32) + 1).min(self.width as i32) as usize;
        let py_top = (min_y as i32).max(0) as usize;
        let py_bottom = ((max_y as i32) + 1).min(self.height as i32) as usize;

        if px_left >= self.width || py_top >= self.height || px_right == 0 || py_bottom == 0 {
            // Off-screen entirely -- not visible.
            return false;
        }

        // The AABB is occluded if every pixel in its screen-space rect has a
        // stored depth value that is closer (smaller) than the AABB's nearest
        // depth.
        let test_z = min_z.clamp(0.0, 1.0);

        // Use hierarchical test: sample a grid of points in the bounding rect
        // to avoid testing every single pixel for large screen-space AABBs.
        let sample_step_x = ((px_right - px_left) / 8).max(1);
        let sample_step_y = ((py_bottom - py_top) / 8).max(1);

        let mut x = px_left;
        while x < px_right {
            let mut y = py_top;
            while y < py_bottom {
                if self.depth[y * self.width + x] >= test_z {
                    // Found a pixel where the AABB could be visible.
                    return true;
                }
                y += sample_step_y;
            }
            x += sample_step_x;
        }

        // All sampled pixels are closer than the AABB -- occluded.
        false
    }

    /// Tests an AABB for visibility using a conservative screen-space rect test
    /// with no MVP projection. The caller provides the pre-computed screen rect
    /// and nearest depth.
    pub fn test_screen_rect(
        &self,
        left: usize,
        top: usize,
        right: usize,
        bottom: usize,
        nearest_depth: f32,
    ) -> bool {
        let left = left.min(self.width);
        let right = right.min(self.width);
        let top = top.min(self.height);
        let bottom = bottom.min(self.height);

        if left >= right || top >= bottom {
            return false;
        }

        let step_x = ((right - left) / 4).max(1);
        let step_y = ((bottom - top) / 4).max(1);

        let mut x = left;
        while x < right {
            let mut y = top;
            while y < bottom {
                if self.depth[y * self.width + x] >= nearest_depth {
                    return true;
                }
                y += step_y;
            }
            x += step_x;
        }

        false
    }

    // -----------------------------------------------------------------------
    // AABB rasterization (rasterize the 6 faces of a box as occluder)
    // -----------------------------------------------------------------------

    /// Rasterizes an AABB as an occluder by drawing its 6 faces (12 triangles)
    /// into the depth buffer.
    pub fn rasterize_aabb(&mut self, aabb: &AABB, mvp: &Mat4) {
        let corners = aabb.corners();
        // Project all 8 corners to clip space.
        let clip: Vec<Vec4> = corners
            .iter()
            .map(|c| *mvp * Vec4::new(c.x, c.y, c.z, 1.0))
            .collect();

        // The 6 faces of the box as pairs of triangles (indices into corners).
        // Each face is defined by 4 corner indices; we split into 2 triangles.
        const FACES: [[usize; 4]; 6] = [
            [0, 1, 2, 3], // front  (-Z)
            [5, 4, 7, 6], // back   (+Z)
            [4, 0, 3, 7], // left   (-X)
            [1, 5, 6, 2], // right  (+X)
            [3, 2, 6, 7], // top    (+Y)
            [4, 5, 1, 0], // bottom (-Y)
        ];

        for face in &FACES {
            let c0 = clip[face[0]];
            let c1 = clip[face[1]];
            let c2 = clip[face[2]];
            let c3 = clip[face[3]];

            self.rasterize_triangle_clip(c0, c1, c2);
            self.rasterize_triangle_clip(c0, c2, c3);
        }
    }
}

// ===========================================================================
// OccluderComponent -- marks an entity as an occluder
// ===========================================================================

/// Component attached to entities that should act as occluders for the
/// software occlusion culling system.
///
/// Occluders are typically large, static objects like buildings, terrain chunks,
/// or walls. They do not need to be renderable themselves (e.g., occluder-only
/// proxy geometry is valid).
#[derive(Debug, Clone)]
pub struct OccluderComponent {
    /// The axis-aligned bounding box of the occluder in *local* space.
    pub local_aabb: AABB,
    /// Whether this occluder is currently enabled.
    pub enabled: bool,
    /// Priority for occluder selection (higher = selected first when budget is
    /// limited). A value of 0 means automatic priority based on screen-space
    /// size.
    pub priority: u32,
    /// If true, use a simplified box proxy for rasterization instead of the
    /// actual mesh. This is almost always desired for performance.
    pub use_box_proxy: bool,
    /// An optional set of simplified triangles (in local space) to rasterize
    /// instead of the bounding box. Each entry is (v0, v1, v2).
    pub simplified_mesh: Vec<[Vec3; 3]>,
}

impl OccluderComponent {
    /// Creates a new occluder component with default settings.
    pub fn new(local_aabb: AABB) -> Self {
        Self {
            local_aabb,
            enabled: true,
            priority: 0,
            use_box_proxy: true,
            simplified_mesh: Vec::new(),
        }
    }

    /// Sets the priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Provides a simplified mesh for more accurate occlusion.
    pub fn with_simplified_mesh(mut self, triangles: Vec<[Vec3; 3]>) -> Self {
        self.simplified_mesh = triangles;
        self.use_box_proxy = false;
        self
    }
}

// ===========================================================================
// OccluderCandidate -- internal structure for occluder selection
// ===========================================================================

/// An occluder candidate with precomputed scoring metrics.
#[derive(Debug, Clone)]
struct OccluderCandidate {
    /// Index into the occluder array.
    index: usize,
    /// Estimated screen-space coverage (0..1). Larger = better occluder.
    screen_coverage: f32,
    /// Distance from camera (world units).
    distance: f32,
    /// Combined score: higher is better.
    score: f32,
    /// The world-space AABB.
    world_aabb: AABB,
    /// The model matrix (local -> world).
    model_matrix: Mat4,
}

// ===========================================================================
// OcclusionCuller -- the main culling coordinator
// ===========================================================================

/// The main occlusion culling system that coordinates occluder selection,
/// depth buffer rasterization, and renderable testing.
///
/// # Usage
///
/// ```text
/// let mut culler = OcclusionCuller::new(256, 144);
///
/// // Each frame:
/// let visible = culler.cull(&camera, &occluders, &renderables);
/// // `visible` contains indices of renderables that passed the occlusion test.
/// ```
#[derive(Debug)]
pub struct OcclusionCuller {
    /// The software depth buffer.
    buffer: OcclusionBuffer,
    /// Maximum number of occluders to rasterize per frame.
    max_occluders: usize,
    /// Whether occlusion culling is enabled.
    enabled: bool,
    /// Minimum screen-space coverage for an object to be used as an occluder.
    min_occluder_coverage: f32,
    /// Maximum distance at which objects are considered as occluders.
    max_occluder_distance: f32,
    /// Statistics from the last frame.
    last_stats: OcclusionCullStats,
}

/// Result statistics from a single occlusion culling pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct OcclusionCullStats {
    /// Number of occluders considered.
    pub occluders_considered: u32,
    /// Number of occluders rasterized.
    pub occluders_rasterized: u32,
    /// Number of renderables tested.
    pub renderables_tested: u32,
    /// Number of renderables that passed (visible).
    pub renderables_visible: u32,
    /// Number of renderables that were occluded (culled).
    pub renderables_occluded: u32,
    /// Depth buffer statistics.
    pub buffer_stats: OcclusionBufferStats,
}

/// Camera data needed for occlusion culling.
#[derive(Debug, Clone)]
pub struct OcclusionCamera {
    /// Camera world-space position.
    pub position: Vec3,
    /// View matrix (world -> view).
    pub view: Mat4,
    /// Projection matrix (view -> clip).
    pub projection: Mat4,
    /// Combined view-projection matrix.
    pub view_projection: Mat4,
    /// Camera forward direction.
    pub forward: Vec3,
    /// Near plane distance.
    pub near: f32,
    /// Far plane distance.
    pub far: f32,
}

impl OcclusionCamera {
    /// Creates an occlusion camera from view and projection matrices.
    pub fn new(position: Vec3, forward: Vec3, view: Mat4, projection: Mat4, near: f32, far: f32) -> Self {
        Self {
            position,
            view,
            projection,
            view_projection: projection * view,
            forward: forward.normalize(),
            near,
            far,
        }
    }
}

/// An occluder with its world-space transform.
#[derive(Debug, Clone)]
pub struct OccluderEntry {
    /// The occluder component.
    pub component: OccluderComponent,
    /// The model matrix (local -> world).
    pub model_matrix: Mat4,
}

/// A renderable with its world-space AABB.
#[derive(Debug, Clone)]
pub struct RenderableEntry {
    /// The world-space AABB.
    pub world_aabb: AABB,
    /// Whether this renderable should be frustum-culled as well.
    pub frustum_cull: bool,
}

impl OcclusionCuller {
    /// Creates a new occlusion culler with the given depth buffer dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            buffer: OcclusionBuffer::new(width, height),
            max_occluders: MAX_OCCLUDERS_PER_FRAME,
            enabled: true,
            min_occluder_coverage: 0.005, // 0.5% of screen
            max_occluder_distance: 500.0,
            last_stats: OcclusionCullStats::default(),
        }
    }

    /// Creates a culler with default buffer dimensions.
    pub fn default_size() -> Self {
        Self::new(DEFAULT_BUFFER_WIDTH, DEFAULT_BUFFER_HEIGHT)
    }

    /// Enables or disables occlusion culling.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether occlusion culling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Sets the maximum number of occluders to rasterize per frame.
    pub fn set_max_occluders(&mut self, max: usize) {
        self.max_occluders = max;
    }

    /// Sets the minimum screen-space coverage for occluder selection.
    pub fn set_min_occluder_coverage(&mut self, coverage: f32) {
        self.min_occluder_coverage = coverage.clamp(0.0, 1.0);
    }

    /// Sets the maximum occluder distance.
    pub fn set_max_occluder_distance(&mut self, distance: f32) {
        self.max_occluder_distance = distance.max(1.0);
    }

    /// Returns statistics from the last culling pass.
    pub fn last_stats(&self) -> &OcclusionCullStats {
        &self.last_stats
    }

    /// Returns a reference to the internal depth buffer (e.g., for debug
    /// visualization).
    pub fn depth_buffer(&self) -> &OcclusionBuffer {
        &self.buffer
    }

    /// Performs the full occlusion culling pipeline:
    /// 1. Select and score occluders.
    /// 2. Rasterize top-scoring occluders into the depth buffer.
    /// 3. Test all renderables against the depth buffer.
    ///
    /// Returns a `Vec<usize>` containing the indices of visible renderables.
    pub fn cull(
        &mut self,
        camera: &OcclusionCamera,
        occluders: &[OccluderEntry],
        renderables: &[RenderableEntry],
    ) -> Vec<usize> {
        let mut stats = OcclusionCullStats::default();

        if !self.enabled || renderables.is_empty() {
            // If disabled, everything is visible.
            stats.renderables_tested = renderables.len() as u32;
            stats.renderables_visible = renderables.len() as u32;
            self.last_stats = stats;
            return (0..renderables.len()).collect();
        }

        // Step 1: Clear the depth buffer.
        self.buffer.clear();

        // Step 2: Select and score occluders.
        let mut candidates = self.select_occluders(camera, occluders);
        stats.occluders_considered = candidates.len() as u32;

        // Sort by score (descending) and take the top N.
        candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(self.max_occluders);

        // Step 3: Rasterize selected occluders.
        for candidate in &candidates {
            let mvp = camera.view_projection * candidate.model_matrix;
            self.buffer.rasterize_aabb(&candidate.world_aabb, &mvp);
            stats.occluders_rasterized += 1;
        }

        // Step 4: Test renderables against the depth buffer.
        let mut visible = Vec::with_capacity(renderables.len());
        for (i, renderable) in renderables.iter().enumerate() {
            stats.renderables_tested += 1;

            if self.buffer.test_aabb(&renderable.world_aabb, &camera.view_projection) {
                visible.push(i);
                stats.renderables_visible += 1;
            } else {
                stats.renderables_occluded += 1;
            }
        }

        stats.buffer_stats = self.buffer.stats();
        self.last_stats = stats;

        visible
    }

    /// Selects and scores occluder candidates based on distance and estimated
    /// screen-space coverage.
    fn select_occluders(
        &self,
        camera: &OcclusionCamera,
        occluders: &[OccluderEntry],
    ) -> Vec<OccluderCandidate> {
        let mut candidates = Vec::new();
        let screen_area = (self.buffer.width * self.buffer.height) as f32;

        for (i, entry) in occluders.iter().enumerate() {
            if !entry.component.enabled {
                continue;
            }

            // Transform local AABB to world space (conservative: use the 8
            // corners and recompute a world-space AABB).
            let world_aabb = self.transform_aabb(&entry.component.local_aabb, &entry.model_matrix);

            // Distance from camera to AABB center.
            let center = world_aabb.center();
            let to_center = center - camera.position;
            let distance = to_center.length();

            if distance > self.max_occluder_distance {
                continue;
            }

            // Check if the occluder is behind the camera.
            if to_center.dot(camera.forward) < -world_aabb.half_extents().length() {
                continue;
            }

            // Estimate screen-space coverage.
            let screen_coverage = self.estimate_screen_coverage(
                &world_aabb,
                &camera.view_projection,
            );

            if screen_coverage < self.min_occluder_coverage {
                continue;
            }

            // Score: prefer large, close occluders.
            let distance_factor = 1.0 / (1.0 + distance * 0.01);
            let priority_bonus = entry.component.priority as f32 * 0.1;
            let score = screen_coverage * distance_factor + priority_bonus;

            candidates.push(OccluderCandidate {
                index: i,
                screen_coverage,
                distance,
                score,
                world_aabb,
                model_matrix: entry.model_matrix,
            });
        }

        candidates
    }

    /// Transforms a local-space AABB by a matrix, returning a conservative
    /// world-space AABB.
    fn transform_aabb(&self, aabb: &AABB, matrix: &Mat4) -> AABB {
        let corners = aabb.corners();
        let mut result = AABB {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        };

        for corner in &corners {
            let world = matrix.transform_point3(*corner);
            result.min = result.min.min(world);
            result.max = result.max.max(world);
        }

        result
    }

    /// Estimates the screen-space coverage of an AABB as a fraction of the
    /// total screen area.
    fn estimate_screen_coverage(&self, aabb: &AABB, vp: &Mat4) -> f32 {
        let corners = aabb.corners();
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for corner in &corners {
            let clip = *vp * Vec4::new(corner.x, corner.y, corner.z, 1.0);
            if clip.w <= EPSILON {
                // Very close to camera -- assume large coverage.
                return 1.0;
            }
            let inv_w = 1.0 / clip.w;
            let ndc_x = clip.x * inv_w;
            let ndc_y = clip.y * inv_w;

            min_x = min_x.min(ndc_x);
            max_x = max_x.max(ndc_x);
            min_y = min_y.min(ndc_y);
            max_y = max_y.max(ndc_y);
        }

        // NDC is [-1, 1], so full screen = 4.0 area.
        let ndc_area = (max_x - min_x).max(0.0) * (max_y - min_y).max(0.0);
        (ndc_area / 4.0).clamp(0.0, 1.0)
    }
}

// ===========================================================================
// RenderQueueFilter -- integration with the render queue
// ===========================================================================

/// A draw-call filter that removes occluded objects from the render queue.
///
/// This sits between the scene graph / ECS gather phase and GPU command
/// recording. It receives a list of draw call descriptors, runs them through
/// the occlusion culler, and produces a filtered list.
#[derive(Debug)]
pub struct RenderQueueFilter {
    culler: OcclusionCuller,
}

/// A simplified draw-call descriptor for occlusion filtering.
#[derive(Debug, Clone)]
pub struct DrawCallDescriptor {
    /// World-space AABB of the drawable.
    pub world_aabb: AABB,
    /// Unique draw call ID (opaque to the filter; passed through).
    pub draw_call_id: u64,
    /// Whether this draw call is from an occluder (should be drawn regardless).
    pub is_occluder: bool,
}

/// Result of filtering draw calls through occlusion culling.
#[derive(Debug)]
pub struct FilteredDrawCalls {
    /// Draw call IDs that survived the occlusion test.
    pub visible_ids: Vec<u64>,
    /// Statistics from the culling pass.
    pub stats: OcclusionCullStats,
}

impl RenderQueueFilter {
    /// Creates a new render queue filter with the given depth buffer dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            culler: OcclusionCuller::new(width, height),
        }
    }

    /// Creates a filter with default buffer dimensions.
    pub fn default_size() -> Self {
        Self {
            culler: OcclusionCuller::default_size(),
        }
    }

    /// Enables or disables occlusion culling.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.culler.set_enabled(enabled);
    }

    /// Filters a set of draw calls, returning only the visible ones.
    ///
    /// Occluder draw calls are always included in the output. Non-occluder
    /// draw calls are tested against the depth buffer.
    pub fn filter(
        &mut self,
        camera: &OcclusionCamera,
        draw_calls: &[DrawCallDescriptor],
    ) -> FilteredDrawCalls {
        // Separate occluders from non-occluders.
        let mut occluders = Vec::new();
        let mut testables = Vec::new();
        let mut testable_indices = Vec::new();

        for (i, dc) in draw_calls.iter().enumerate() {
            if dc.is_occluder {
                occluders.push(OccluderEntry {
                    component: OccluderComponent::new(dc.world_aabb),
                    model_matrix: Mat4::IDENTITY, // AABB is already in world space.
                });
            } else {
                testables.push(RenderableEntry {
                    world_aabb: dc.world_aabb,
                    frustum_cull: false,
                });
                testable_indices.push(i);
            }
        }

        // Run occlusion culling.
        let visible_testable_indices = self.culler.cull(camera, &occluders, &testables);

        // Build the result: occluders always visible + visible testables.
        let mut visible_ids = Vec::with_capacity(draw_calls.len());
        for dc in draw_calls.iter() {
            if dc.is_occluder {
                visible_ids.push(dc.draw_call_id);
            }
        }
        for &vi in &visible_testable_indices {
            let dc_idx = testable_indices[vi];
            visible_ids.push(draw_calls[dc_idx].draw_call_id);
        }

        FilteredDrawCalls {
            visible_ids,
            stats: *self.culler.last_stats(),
        }
    }
}

// ===========================================================================
// PortalOcclusion -- indoor portal-based visibility
// ===========================================================================

/// Unique identifier for a room in the portal system.
pub type RoomId = u32;

/// A portal connecting two rooms.
///
/// A portal is a convex polygon lying on a plane. It defines a passage (e.g.,
/// a doorway or window) through which one room can see into another.
#[derive(Debug, Clone)]
pub struct Portal {
    /// Unique ID for this portal.
    pub id: u32,
    /// The plane on which the portal polygon lies.
    pub plane: PortalPlane,
    /// Vertices of the portal polygon (convex, coplanar), in world space.
    /// Winding order defines the "front" side.
    pub vertices: Vec<Vec3>,
    /// The room on the front side of the portal.
    pub room_front: RoomId,
    /// The room on the back side of the portal.
    pub room_back: RoomId,
    /// Whether the portal is currently open (can be closed by a door, etc.).
    pub open: bool,
}

/// A plane used by the portal system.
#[derive(Debug, Clone, Copy)]
pub struct PortalPlane {
    /// Outward-facing normal (towards the front room).
    pub normal: Vec3,
    /// Signed distance from the origin.
    pub distance: f32,
}

impl PortalPlane {
    /// Creates a plane from a normal and distance.
    pub fn new(normal: Vec3, distance: f32) -> Self {
        Self { normal, distance }
    }

    /// Creates a plane from three points (CCW winding).
    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let normal = (b - a).cross(c - a).normalize();
        let distance = normal.dot(a);
        Self { normal, distance }
    }

    /// Signed distance from a point to this plane.
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.distance
    }
}

impl Portal {
    /// Creates a new portal.
    pub fn new(
        id: u32,
        vertices: Vec<Vec3>,
        room_front: RoomId,
        room_back: RoomId,
    ) -> Self {
        let plane = if vertices.len() >= 3 {
            PortalPlane::from_points(vertices[0], vertices[1], vertices[2])
        } else {
            PortalPlane::new(Vec3::Y, 0.0)
        };

        Self {
            id,
            plane,
            vertices,
            room_front,
            room_back,
            open: true,
        }
    }

    /// Returns the center of the portal polygon.
    pub fn center(&self) -> Vec3 {
        if self.vertices.is_empty() {
            return Vec3::ZERO;
        }
        let sum: Vec3 = self.vertices.iter().copied().sum();
        sum / self.vertices.len() as f32
    }

    /// Returns the AABB of the portal polygon.
    pub fn aabb(&self) -> AABB {
        let mut aabb = AABB {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        };
        for v in &self.vertices {
            aabb.min = aabb.min.min(*v);
            aabb.max = aabb.max.max(*v);
        }
        aabb
    }

    /// Returns the room on the other side of the portal from the given room.
    pub fn other_room(&self, from_room: RoomId) -> RoomId {
        if from_room == self.room_front {
            self.room_back
        } else {
            self.room_front
        }
    }

    /// Returns the approximate area of the portal polygon.
    pub fn area(&self) -> f32 {
        if self.vertices.len() < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        let v0 = self.vertices[0];
        for i in 1..self.vertices.len() - 1 {
            let v1 = self.vertices[i];
            let v2 = self.vertices[i + 1];
            area += (v1 - v0).cross(v2 - v0).length() * 0.5;
        }
        area
    }
}

/// A room in the portal system.
///
/// Rooms are convex (or approximately convex) volumes that contain geometry.
/// They are connected to other rooms via portals.
#[derive(Debug, Clone)]
pub struct Room {
    /// Unique room ID.
    pub id: RoomId,
    /// Human-readable name.
    pub name: String,
    /// AABB of the room volume.
    pub bounds: AABB,
    /// Indices into the portal array for portals touching this room.
    pub portal_indices: Vec<usize>,
    /// Whether this room is an outdoor area (affects culling heuristics).
    pub is_outdoor: bool,
}

impl Room {
    /// Creates a new room.
    pub fn new(id: RoomId, name: impl Into<String>, bounds: AABB) -> Self {
        Self {
            id,
            name: name.into(),
            bounds,
            portal_indices: Vec::new(),
            is_outdoor: false,
        }
    }

    /// Returns true if the given point is inside the room's bounding volume.
    pub fn contains_point(&self, point: Vec3) -> bool {
        self.bounds.contains_point(point)
    }
}

/// A frustum used during portal traversal, gradually narrowed as we pass
/// through successive portals.
#[derive(Debug, Clone)]
struct PortalFrustum {
    /// Clipping planes of the narrowed frustum. Normals point inward.
    planes: Vec<PortalPlane>,
}

impl PortalFrustum {
    /// Creates the initial frustum from the camera's view-projection matrix.
    fn from_view_projection(vp: &Mat4) -> Self {
        let cols = vp.to_cols_array_2d();
        let row = |r: usize| -> Vec4 {
            Vec4::new(cols[0][r], cols[1][r], cols[2][r], cols[3][r])
        };

        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let extract = |v: Vec4| -> PortalPlane {
            let len = Vec3::new(v.x, v.y, v.z).length();
            if len < EPSILON {
                return PortalPlane::new(Vec3::Y, 0.0);
            }
            PortalPlane {
                normal: Vec3::new(v.x, v.y, v.z) / len,
                distance: -v.w / len,
            }
        };

        let planes = vec![
            extract(r3 + r0), // left
            extract(r3 - r0), // right
            extract(r3 + r1), // bottom
            extract(r3 - r1), // top
            extract(r3 + r2), // near
            extract(r3 - r2), // far
        ];

        Self { planes }
    }

    /// Tests whether an AABB is at least partially inside this frustum.
    fn contains_aabb(&self, aabb: &AABB) -> bool {
        for plane in &self.planes {
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

    /// Narrows this frustum through a portal polygon, as seen from `eye`.
    ///
    /// Creates a new frustum whose planes are formed by the eye position and
    /// each edge of the portal polygon. This is the key operation for
    /// recursive portal visibility.
    fn narrow_through_portal(&self, eye: Vec3, portal: &Portal) -> Option<PortalFrustum> {
        let verts = &portal.vertices;
        if verts.len() < 3 {
            return None;
        }

        // Check that the portal polygon is at least partially inside the
        // current frustum.
        let portal_aabb = portal.aabb();
        if !self.contains_aabb(&portal_aabb) {
            return None;
        }

        // Build frustum planes from the eye through each edge of the portal.
        let n = verts.len();
        let mut planes = Vec::with_capacity(n + 1);

        for i in 0..n {
            let v0 = verts[i];
            let v1 = verts[(i + 1) % n];
            let edge = v1 - v0;
            let to_v0 = v0 - eye;

            // Normal = cross(edge, to_v0), pointing inward.
            let normal = edge.cross(to_v0);
            let len = normal.length();
            if len < EPSILON {
                continue;
            }
            let normal = normal / len;

            // Ensure the normal points inward (toward the portal center).
            let center = portal.center();
            if normal.dot(center - eye) < 0.0 {
                planes.push(PortalPlane {
                    normal: -normal,
                    distance: (-normal).dot(eye),
                });
            } else {
                planes.push(PortalPlane {
                    normal,
                    distance: normal.dot(eye),
                });
            }
        }

        // Add the portal plane itself as a near plane to prevent seeing
        // behind the portal.
        let portal_normal = portal.plane.normal;
        let eye_side = portal.plane.signed_distance(eye);
        if eye_side > 0.0 {
            // Eye is on the front side -- cull things behind the portal plane.
            planes.push(portal.plane);
        } else {
            // Eye is on the back side -- flip the portal plane.
            planes.push(PortalPlane {
                normal: -portal_normal,
                distance: -portal.plane.distance,
            });
        }

        if planes.is_empty() {
            return None;
        }

        Some(PortalFrustum { planes })
    }
}

/// The portal-based occlusion system for indoor environments.
///
/// Manages a set of rooms connected by portals and determines which rooms
/// are visible from a given camera position through recursive portal traversal.
#[derive(Debug)]
pub struct PortalOcclusion {
    /// All rooms in the level.
    rooms: Vec<Room>,
    /// All portals in the level.
    portals: Vec<Portal>,
    /// Maximum recursion depth for portal traversal.
    max_depth: u32,
    /// Cached visible rooms from the last query.
    last_visible_rooms: Vec<RoomId>,
}

/// Configuration for portal occlusion.
#[derive(Debug, Clone)]
pub struct PortalOcclusionConfig {
    /// Maximum recursion depth when traversing portals.
    pub max_depth: u32,
}

impl Default for PortalOcclusionConfig {
    fn default() -> Self {
        Self { max_depth: 8 }
    }
}

impl PortalOcclusion {
    /// Creates a new portal occlusion system.
    pub fn new(config: PortalOcclusionConfig) -> Self {
        Self {
            rooms: Vec::new(),
            portals: Vec::new(),
            max_depth: config.max_depth,
            last_visible_rooms: Vec::new(),
        }
    }

    /// Creates a system with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(PortalOcclusionConfig::default())
    }

    /// Adds a room to the system.
    pub fn add_room(&mut self, room: Room) {
        self.rooms.push(room);
    }

    /// Adds a portal to the system and links it to its rooms.
    pub fn add_portal(&mut self, portal: Portal) {
        let portal_index = self.portals.len();

        // Link portal to its front room.
        if let Some(room) = self.rooms.iter_mut().find(|r| r.id == portal.room_front) {
            room.portal_indices.push(portal_index);
        }

        // Link portal to its back room.
        if let Some(room) = self.rooms.iter_mut().find(|r| r.id == portal.room_back) {
            room.portal_indices.push(portal_index);
        }

        self.portals.push(portal);
    }

    /// Returns a reference to all rooms.
    pub fn rooms(&self) -> &[Room] {
        &self.rooms
    }

    /// Returns a reference to all portals.
    pub fn portals(&self) -> &[Portal] {
        &self.portals
    }

    /// Returns the visible rooms from the last query.
    pub fn last_visible_rooms(&self) -> &[RoomId] {
        &self.last_visible_rooms
    }

    /// Finds the room containing the given point.
    ///
    /// If the point is in multiple rooms (overlapping bounds), the smallest
    /// room is returned.
    pub fn find_room(&self, point: Vec3) -> Option<RoomId> {
        let mut best_room: Option<(RoomId, f32)> = None;

        for room in &self.rooms {
            if room.contains_point(point) {
                let volume = room.bounds.volume();
                match best_room {
                    Some((_, best_vol)) if volume < best_vol => {
                        best_room = Some((room.id, volume));
                    }
                    None => {
                        best_room = Some((room.id, volume));
                    }
                    _ => {}
                }
            }
        }

        best_room.map(|(id, _)| id)
    }

    /// Determines which rooms are visible from the camera.
    ///
    /// Starts from the room containing the camera and recursively traverses
    /// portals, narrowing the frustum at each step. Returns the set of
    /// visible room IDs.
    ///
    /// # Arguments
    /// * `camera_pos` - World-space camera position.
    /// * `view_projection` - The camera's view-projection matrix.
    ///
    /// # Returns
    /// A sorted, deduplicated list of visible room IDs.
    pub fn determine_visible_rooms(
        &mut self,
        camera_pos: Vec3,
        view_projection: &Mat4,
    ) -> Vec<RoomId> {
        let camera_room = match self.find_room(camera_pos) {
            Some(id) => id,
            None => {
                // Camera is not in any room -- mark all rooms as visible.
                self.last_visible_rooms = self.rooms.iter().map(|r| r.id).collect();
                return self.last_visible_rooms.clone();
            }
        };

        let initial_frustum = PortalFrustum::from_view_projection(view_projection);

        let mut visible = Vec::new();
        let mut visited = vec![false; self.rooms.len()];

        self.traverse_portals(
            camera_room,
            camera_pos,
            &initial_frustum,
            0,
            &mut visible,
            &mut visited,
        );

        visible.sort();
        visible.dedup();

        self.last_visible_rooms = visible.clone();
        visible
    }

    /// Determines visible rooms starting from a known room ID.
    pub fn determine_visible_rooms_from(
        &mut self,
        camera_room: RoomId,
        camera_pos: Vec3,
        view_projection: &Mat4,
    ) -> Vec<RoomId> {
        let initial_frustum = PortalFrustum::from_view_projection(view_projection);

        let mut visible = Vec::new();
        let mut visited = vec![false; self.rooms.len()];

        self.traverse_portals(
            camera_room,
            camera_pos,
            &initial_frustum,
            0,
            &mut visible,
            &mut visited,
        );

        visible.sort();
        visible.dedup();

        self.last_visible_rooms = visible.clone();
        visible
    }

    /// Recursively traverses portals to find visible rooms.
    fn traverse_portals(
        &self,
        current_room_id: RoomId,
        eye: Vec3,
        frustum: &PortalFrustum,
        depth: u32,
        visible: &mut Vec<RoomId>,
        visited: &mut Vec<bool>,
    ) {
        if depth > self.max_depth {
            return;
        }

        // Find the room index.
        let room_index = match self.rooms.iter().position(|r| r.id == current_room_id) {
            Some(i) => i,
            None => return,
        };

        // Mark as visited to prevent infinite loops.
        if visited[room_index] {
            return;
        }
        visited[room_index] = true;

        // This room is visible.
        visible.push(current_room_id);

        // Traverse each portal of this room.
        let portal_indices = self.rooms[room_index].portal_indices.clone();
        for &portal_idx in &portal_indices {
            if portal_idx >= self.portals.len() {
                continue;
            }

            let portal = &self.portals[portal_idx];

            // Skip closed portals.
            if !portal.open {
                continue;
            }

            // Determine the destination room.
            let dest_room = portal.other_room(current_room_id);

            // Check if the portal is visible within the current frustum.
            let portal_aabb = portal.aabb();
            if !frustum.contains_aabb(&portal_aabb) {
                continue;
            }

            // Narrow the frustum through the portal.
            if let Some(narrowed_frustum) = frustum.narrow_through_portal(eye, portal) {
                self.traverse_portals(
                    dest_room,
                    eye,
                    &narrowed_frustum,
                    depth + 1,
                    visible,
                    visited,
                );
            }
        }
    }

    /// Opens or closes a portal by ID.
    pub fn set_portal_open(&mut self, portal_id: u32, open: bool) {
        if let Some(portal) = self.portals.iter_mut().find(|p| p.id == portal_id) {
            portal.open = open;
        }
    }

    /// Toggles a portal's open state.
    pub fn toggle_portal(&mut self, portal_id: u32) {
        if let Some(portal) = self.portals.iter_mut().find(|p| p.id == portal_id) {
            portal.open = !portal.open;
        }
    }
}

// ===========================================================================
// HierarchicalZBuffer -- multi-resolution depth testing
// ===========================================================================

/// A hierarchical Z-buffer (Hi-Z) for accelerated occlusion testing.
///
/// Built from an `OcclusionBuffer`, each mip level stores the *maximum* depth
/// of its 2x2 parent texels. Testing an AABB can start at a coarse mip and
/// only drill down if the test is inconclusive.
#[derive(Debug, Clone)]
pub struct HierarchicalZBuffer {
    /// Mip levels, from finest (index 0 = full resolution) to coarsest.
    mips: Vec<ZBufferMip>,
    /// Number of mip levels.
    mip_count: usize,
}

/// A single mip level of the hierarchical Z-buffer.
#[derive(Debug, Clone)]
struct ZBufferMip {
    /// Depth values for this mip level (row-major).
    data: Vec<f32>,
    /// Width of this mip level.
    width: usize,
    /// Height of this mip level.
    height: usize,
}

impl HierarchicalZBuffer {
    /// Builds a hierarchical Z-buffer from an `OcclusionBuffer`.
    pub fn build(buffer: &OcclusionBuffer) -> Self {
        let mut mips = Vec::new();

        // Level 0: copy from the occlusion buffer (maximum of 2x2 blocks
        // for some initial reduction, or just use as-is).
        let w0 = buffer.width();
        let h0 = buffer.height();
        let mut data0 = Vec::with_capacity(w0 * h0);
        for y in 0..h0 {
            for x in 0..w0 {
                data0.push(buffer.sample(x, y));
            }
        }
        mips.push(ZBufferMip {
            data: data0,
            width: w0,
            height: h0,
        });

        // Build subsequent mip levels by taking the max of each 2x2 block.
        let mut prev_w = w0;
        let mut prev_h = h0;

        while prev_w > 1 || prev_h > 1 {
            let new_w = (prev_w + 1) / 2;
            let new_h = (prev_h + 1) / 2;
            let prev_data = &mips.last().unwrap().data;
            let mut new_data = Vec::with_capacity(new_w * new_h);

            for y in 0..new_h {
                for x in 0..new_w {
                    let x0 = x * 2;
                    let y0 = y * 2;
                    let x1 = (x0 + 1).min(prev_w - 1);
                    let y1 = (y0 + 1).min(prev_h - 1);

                    let d00 = prev_data[y0 * prev_w + x0];
                    let d10 = prev_data[y0 * prev_w + x1];
                    let d01 = prev_data[y1 * prev_w + x0];
                    let d11 = prev_data[y1 * prev_w + x1];

                    // Max: the farthest depth in this block. If the test depth
                    // is behind the max, the block fully occludes.
                    new_data.push(d00.max(d10).max(d01).max(d11));
                }
            }

            mips.push(ZBufferMip {
                data: new_data,
                width: new_w,
                height: new_h,
            });

            prev_w = new_w;
            prev_h = new_h;
        }

        let mip_count = mips.len();
        Self { mips, mip_count }
    }

    /// Returns the number of mip levels.
    pub fn mip_count(&self) -> usize {
        self.mip_count
    }

    /// Tests whether a screen-space rectangle is occluded, using the Hi-Z
    /// hierarchy for early-out.
    ///
    /// Returns `true` if the rectangle is *visible* (not fully occluded).
    pub fn test_rect(
        &self,
        left: usize,
        top: usize,
        right: usize,
        bottom: usize,
        test_depth: f32,
    ) -> bool {
        if self.mips.is_empty() {
            return true;
        }

        // Choose the coarsest mip level that still covers the rect with at
        // least a few texels for accuracy.
        let rect_w = right.saturating_sub(left).max(1);
        let rect_h = bottom.saturating_sub(top).max(1);
        let base_w = self.mips[0].width;
        let base_h = self.mips[0].height;

        // Start from the coarsest level and refine.
        let target_texels = 2; // at least 2 texels across for the test
        let mut mip = 0;
        for i in (0..self.mip_count).rev() {
            let scale = 1usize << i;
            let tw = (rect_w + scale - 1) / scale;
            let th = (rect_h + scale - 1) / scale;
            if tw >= target_texels || th >= target_texels || i == 0 {
                mip = i;
                break;
            }
        }

        let ref_mip = &self.mips[mip];
        let scale = 1usize << mip;

        let ml = left / scale;
        let mt = top / scale;
        let mr = ((right + scale - 1) / scale).min(ref_mip.width);
        let mb = ((bottom + scale - 1) / scale).min(ref_mip.height);

        // Test: if any texel in the coarse rect has depth >= test_depth,
        // the object might be visible.
        for y in mt..mb {
            for x in ml..mr {
                if ref_mip.data[y * ref_mip.width + x] >= test_depth {
                    return true; // visible
                }
            }
        }

        false // all texels are closer -- occluded
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // OcclusionBuffer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_buffer_clear() {
        let mut buf = OcclusionBuffer::new(16, 16);
        buf.write_depth(0, 0, 0.5);
        assert!(buf.sample(0, 0) <= 0.5);
        buf.clear();
        assert!((buf.sample(0, 0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_depth_write_closer() {
        let mut buf = OcclusionBuffer::new(8, 8);
        buf.write_depth(3, 3, 0.8);
        assert!((buf.sample(3, 3) - 0.8).abs() < EPSILON);
        buf.write_depth(3, 3, 0.3);
        assert!((buf.sample(3, 3) - 0.3).abs() < EPSILON);
        // Writing a farther value should not overwrite.
        buf.write_depth(3, 3, 0.9);
        assert!((buf.sample(3, 3) - 0.3).abs() < EPSILON);
    }

    #[test]
    fn test_rasterize_simple_triangle() {
        let mut buf = OcclusionBuffer::new(32, 32);
        buf.rasterize_triangle(
            Vec3::new(8.0, 4.0, 0.5),
            Vec3::new(24.0, 4.0, 0.5),
            Vec3::new(16.0, 28.0, 0.5),
        );
        // Center of the triangle should have been written to.
        let center_depth = buf.sample(16, 16);
        assert!(center_depth <= 0.5 + EPSILON, "center depth = {center_depth}");
        // Corner outside the triangle should still be 1.0.
        assert!((buf.sample(0, 0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_aabb_test_visible_in_front() {
        let buf = OcclusionBuffer::new(64, 64);
        // An empty buffer (all 1.0) should report everything as visible.
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -5.0), Vec3::new(1.0, 1.0, -3.0));
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let mvp = proj * view;
        assert!(buf.test_aabb(&aabb, &mvp));
    }

    #[test]
    fn test_aabb_test_behind_camera() {
        let buf = OcclusionBuffer::new(64, 64);
        // AABB behind the camera should still be visible (conservative).
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, 3.0), Vec3::new(1.0, 1.0, 5.0));
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let mvp = proj * view;
        // Behind-camera AABBs are conservatively marked visible.
        assert!(buf.test_aabb(&aabb, &mvp));
    }

    // -----------------------------------------------------------------------
    // OcclusionCuller tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_culler_no_occluders() {
        let mut culler = OcclusionCuller::new(32, 32);
        let camera = OcclusionCamera::new(
            Vec3::ZERO,
            Vec3::NEG_Z,
            Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y),
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            0.1,
            100.0,
        );
        let renderables = vec![
            RenderableEntry {
                world_aabb: AABB::new(Vec3::new(-1.0, -1.0, -5.0), Vec3::new(1.0, 1.0, -3.0)),
                frustum_cull: false,
            },
        ];
        let visible = culler.cull(&camera, &[], &renderables);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0], 0);
    }

    #[test]
    fn test_culler_disabled() {
        let mut culler = OcclusionCuller::new(32, 32);
        culler.set_enabled(false);
        let camera = OcclusionCamera::new(
            Vec3::ZERO,
            Vec3::NEG_Z,
            Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y),
            Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
            0.1,
            100.0,
        );
        let renderables = vec![
            RenderableEntry {
                world_aabb: AABB::new(Vec3::splat(-1.0), Vec3::splat(1.0)),
                frustum_cull: false,
            },
            RenderableEntry {
                world_aabb: AABB::new(Vec3::splat(10.0), Vec3::splat(12.0)),
                frustum_cull: false,
            },
        ];
        let visible = culler.cull(&camera, &[], &renderables);
        assert_eq!(visible.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Portal tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_portal_other_room() {
        let portal = Portal::new(
            0,
            vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y],
            1,
            2,
        );
        assert_eq!(portal.other_room(1), 2);
        assert_eq!(portal.other_room(2), 1);
    }

    #[test]
    fn test_portal_area() {
        // A 2x2 square portal should have area 4.
        let portal = Portal::new(
            0,
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
            0,
            1,
        );
        assert!((portal.area() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_room_contains_point() {
        let room = Room::new(
            0,
            "test",
            AABB::new(Vec3::new(-5.0, -3.0, -5.0), Vec3::new(5.0, 3.0, 5.0)),
        );
        assert!(room.contains_point(Vec3::ZERO));
        assert!(!room.contains_point(Vec3::new(10.0, 0.0, 0.0)));
    }

    #[test]
    fn test_portal_occlusion_find_room() {
        let mut po = PortalOcclusion::with_defaults();
        po.add_room(Room::new(
            0,
            "room_a",
            AABB::new(Vec3::new(-10.0, -3.0, -10.0), Vec3::new(0.0, 3.0, 0.0)),
        ));
        po.add_room(Room::new(
            1,
            "room_b",
            AABB::new(Vec3::new(0.0, -3.0, -10.0), Vec3::new(10.0, 3.0, 0.0)),
        ));

        assert_eq!(po.find_room(Vec3::new(-5.0, 0.0, -5.0)), Some(0));
        assert_eq!(po.find_room(Vec3::new(5.0, 0.0, -5.0)), Some(1));
        assert_eq!(po.find_room(Vec3::new(50.0, 0.0, 0.0)), None);
    }

    #[test]
    fn test_portal_traversal_basic() {
        let mut po = PortalOcclusion::with_defaults();

        // Two rooms connected by a portal.
        po.add_room(Room::new(
            0,
            "room_a",
            AABB::new(Vec3::new(-10.0, -3.0, -10.0), Vec3::new(0.0, 3.0, 0.0)),
        ));
        po.add_room(Room::new(
            1,
            "room_b",
            AABB::new(Vec3::new(0.0, -3.0, -10.0), Vec3::new(10.0, 3.0, 0.0)),
        ));

        // Portal at x=0, a 2x2 square.
        po.add_portal(Portal::new(
            0,
            vec![
                Vec3::new(0.0, -1.0, -6.0),
                Vec3::new(0.0, -1.0, -4.0),
                Vec3::new(0.0, 1.0, -4.0),
                Vec3::new(0.0, 1.0, -6.0),
            ],
            0,
            1,
        ));

        // Camera in room_a, looking towards the portal.
        let eye = Vec3::new(-5.0, 0.0, -5.0);
        let view = Mat4::look_at_rh(eye, Vec3::new(0.0, 0.0, -5.0), Vec3::Y);
        let proj = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let vp = proj * view;

        let visible = po.determine_visible_rooms(eye, &vp);
        // Both rooms should be visible through the portal.
        assert!(visible.contains(&0));
        assert!(visible.contains(&1));
    }

    #[test]
    fn test_closed_portal_blocks_visibility() {
        let mut po = PortalOcclusion::with_defaults();

        po.add_room(Room::new(
            0,
            "room_a",
            AABB::new(Vec3::new(-10.0, -3.0, -10.0), Vec3::new(0.0, 3.0, 0.0)),
        ));
        po.add_room(Room::new(
            1,
            "room_b",
            AABB::new(Vec3::new(0.0, -3.0, -10.0), Vec3::new(10.0, 3.0, 0.0)),
        ));

        po.add_portal(Portal::new(
            0,
            vec![
                Vec3::new(0.0, -1.0, -6.0),
                Vec3::new(0.0, -1.0, -4.0),
                Vec3::new(0.0, 1.0, -4.0),
                Vec3::new(0.0, 1.0, -6.0),
            ],
            0,
            1,
        ));

        // Close the portal (simulating a closed door).
        po.set_portal_open(0, false);

        let eye = Vec3::new(-5.0, 0.0, -5.0);
        let view = Mat4::look_at_rh(eye, Vec3::new(0.0, 0.0, -5.0), Vec3::Y);
        let proj = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let vp = proj * view;

        let visible = po.determine_visible_rooms(eye, &vp);
        assert!(visible.contains(&0));
        // Room B should NOT be visible through a closed portal.
        assert!(!visible.contains(&1));
    }

    // -----------------------------------------------------------------------
    // HierarchicalZBuffer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hiz_build() {
        let mut buf = OcclusionBuffer::new(8, 8);
        // Write some depth values.
        buf.write_depth(0, 0, 0.2);
        buf.write_depth(1, 0, 0.3);
        buf.write_depth(0, 1, 0.4);
        buf.write_depth(1, 1, 0.1);

        let hiz = HierarchicalZBuffer::build(&buf);
        assert!(hiz.mip_count() >= 2);

        // Mip 1 should have max of the 2x2 block = 0.4.
        let mip1 = &hiz.mips[1];
        assert!((mip1.data[0] - 0.4).abs() < EPSILON);
    }

    #[test]
    fn test_hiz_test_visible() {
        let buf = OcclusionBuffer::new(16, 16);
        // Empty buffer (all 1.0) -- everything should be visible.
        let hiz = HierarchicalZBuffer::build(&buf);
        assert!(hiz.test_rect(2, 2, 10, 10, 0.5));
    }

    #[test]
    fn test_hiz_test_occluded() {
        let mut buf = OcclusionBuffer::new(8, 8);
        // Fill entire buffer with very close depth.
        for y in 0..8 {
            for x in 0..8 {
                buf.write_depth(x, y, 0.1);
            }
        }

        let hiz = HierarchicalZBuffer::build(&buf);
        // An object at depth 0.5 should be occluded.
        assert!(!hiz.test_rect(1, 1, 6, 6, 0.5));
        // An object at depth 0.05 should be visible (closer than buffer).
        assert!(hiz.test_rect(1, 1, 6, 6, 0.05));
    }
}
