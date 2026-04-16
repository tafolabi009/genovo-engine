//! Immediate-mode debug shape renderer for the Genovo engine.
//!
//! The [`DebugRenderer`] collects lines, shapes, and text draw requests
//! during the frame and batches them into vertex buffers for a single draw
//! call. All shapes are rendered as wireframe lines.
//!
//! # Drawing modes
//!
//! - **Single-frame** (default): shapes are cleared at the start of each
//!   frame.
//! - **Persistent**: shapes remain visible for a configurable duration.
//!
//! - **Depth-tested**: shapes are occluded by scene geometry.
//! - **Always-visible**: shapes render on top of everything (overlay).

use std::f32::consts::PI;
use std::time::{Duration, Instant};

use glam::{Mat4, Quat, Vec3};

use genovo_core::AABB;

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

/// RGBA color for debug drawing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    /// Create a color from RGBA floats (0.0 - 1.0).
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Create a color from RGBA bytes (0 - 255).
    pub fn from_bytes(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    /// Convert to RGBA bytes.
    pub fn to_bytes(&self) -> [u8; 4] {
        [
            (self.r * 255.0).clamp(0.0, 255.0) as u8,
            (self.g * 255.0).clamp(0.0, 255.0) as u8,
            (self.b * 255.0).clamp(0.0, 255.0) as u8,
            (self.a * 255.0).clamp(0.0, 255.0) as u8,
        ]
    }

    /// Linear interpolation between two colors.
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: a.r + (b.r - a.r) * t,
            g: a.g + (b.g - a.g) * t,
            b: a.b + (b.b - a.b) * t,
            a: a.a + (b.a - a.a) * t,
        }
    }

    /// Create a color with modified alpha.
    pub fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    // Named color constants.
    pub const RED: Self = Self::new(1.0, 0.0, 0.0, 1.0);
    pub const GREEN: Self = Self::new(0.0, 1.0, 0.0, 1.0);
    pub const BLUE: Self = Self::new(0.0, 0.0, 1.0, 1.0);
    pub const YELLOW: Self = Self::new(1.0, 1.0, 0.0, 1.0);
    pub const CYAN: Self = Self::new(0.0, 1.0, 1.0, 1.0);
    pub const MAGENTA: Self = Self::new(1.0, 0.0, 1.0, 1.0);
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    pub const ORANGE: Self = Self::new(1.0, 0.5, 0.0, 1.0);
    pub const PURPLE: Self = Self::new(0.5, 0.0, 1.0, 1.0);
    pub const PINK: Self = Self::new(1.0, 0.4, 0.7, 1.0);
    pub const GRAY: Self = Self::new(0.5, 0.5, 0.5, 1.0);
    pub const LIGHT_GRAY: Self = Self::new(0.75, 0.75, 0.75, 1.0);
    pub const DARK_GRAY: Self = Self::new(0.25, 0.25, 0.25, 1.0);
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

// ---------------------------------------------------------------------------
// DebugVertex
// ---------------------------------------------------------------------------

/// Vertex for debug line rendering.
#[derive(Debug, Clone, Copy)]
pub struct DebugVertex {
    /// World-space position.
    pub position: Vec3,
    /// RGBA color.
    pub color: Color,
}

impl DebugVertex {
    /// Create a new debug vertex.
    pub fn new(position: Vec3, color: Color) -> Self {
        Self { position, color }
    }
}

// ---------------------------------------------------------------------------
// Draw mode flags
// ---------------------------------------------------------------------------

/// Controls how debug shapes are rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthMode {
    /// Standard depth testing — shapes are occluded by scene geometry.
    DepthTested,
    /// Rendered on top of everything (overlay mode).
    AlwaysVisible,
}

impl Default for DepthMode {
    fn default() -> Self {
        Self::DepthTested
    }
}

// ---------------------------------------------------------------------------
// DebugLine
// ---------------------------------------------------------------------------

/// A single debug line segment.
#[derive(Debug, Clone, Copy)]
struct DebugLine {
    start: Vec3,
    end: Vec3,
    color: Color,
    depth_mode: DepthMode,
}

// ---------------------------------------------------------------------------
// PersistentShape
// ---------------------------------------------------------------------------

/// A debug shape with a limited lifetime.
#[derive(Debug, Clone)]
struct PersistentShape {
    /// Lines making up the shape.
    lines: Vec<DebugLine>,
    /// When the shape expires.
    expire_time: Instant,
}

// ---------------------------------------------------------------------------
// DebugText3D
// ---------------------------------------------------------------------------

/// A text label positioned in 3D space.
#[derive(Debug, Clone)]
pub struct DebugText3D {
    /// World-space position.
    pub position: Vec3,
    /// Text content.
    pub text: String,
    /// Text color.
    pub color: Color,
    /// Whether this text persists.
    pub persistent: bool,
    /// Expiry time (for persistent text).
    pub expire_time: Option<Instant>,
}

// ---------------------------------------------------------------------------
// BoneInfo (for skeleton drawing)
// ---------------------------------------------------------------------------

/// Minimal bone information for debug skeleton rendering.
#[derive(Debug, Clone)]
pub struct BoneInfo {
    /// Index of the parent bone (-1 for root).
    pub parent_index: i32,
    /// Local-to-world transform for this bone.
    pub world_transform: Mat4,
}

// ---------------------------------------------------------------------------
// DebugRenderer
// ---------------------------------------------------------------------------

/// Immediate-mode debug shape renderer.
///
/// Collects draw requests during the frame and produces batched vertex data.
///
/// # Example
///
/// ```ignore
/// let mut dbg = DebugRenderer::new();
///
/// // At frame start:
/// dbg.begin_frame();
///
/// // During the frame:
/// dbg.draw_line(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Color::RED);
/// dbg.draw_sphere(Vec3::new(0.0, 2.0, 0.0), 1.0, Color::GREEN, 16);
/// dbg.draw_box(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, Color::BLUE);
///
/// // At frame end: get the vertex buffer.
/// let (depth_tested, overlay) = dbg.build_vertex_buffers();
/// ```
pub struct DebugRenderer {
    /// Single-frame lines (depth-tested).
    lines_depth: Vec<DebugLine>,
    /// Single-frame lines (always visible).
    lines_overlay: Vec<DebugLine>,
    /// Persistent shapes with expiry times.
    persistent: Vec<PersistentShape>,
    /// 3D text labels.
    texts: Vec<DebugText3D>,
    /// Whether the debug renderer is enabled.
    enabled: bool,
    /// Default depth mode for convenience methods.
    default_depth_mode: DepthMode,
    /// Maximum number of lines per frame (budget).
    max_lines: usize,
}

impl DebugRenderer {
    /// Create a new debug renderer.
    pub fn new() -> Self {
        Self {
            lines_depth: Vec::with_capacity(4096),
            lines_overlay: Vec::with_capacity(1024),
            persistent: Vec::new(),
            texts: Vec::new(),
            enabled: true,
            default_depth_mode: DepthMode::DepthTested,
            max_lines: 100_000,
        }
    }

    /// Enable or disable debug rendering.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if debug rendering is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set the default depth mode.
    pub fn set_default_depth_mode(&mut self, mode: DepthMode) {
        self.default_depth_mode = mode;
    }

    /// Set the maximum line budget per frame.
    pub fn set_max_lines(&mut self, max: usize) {
        self.max_lines = max;
    }

    /// Clear single-frame shapes and expire persistent shapes. Call at frame
    /// start.
    pub fn begin_frame(&mut self) {
        self.lines_depth.clear();
        self.lines_overlay.clear();
        self.texts.retain(|t| {
            if let Some(expire) = t.expire_time {
                Instant::now() < expire
            } else {
                !t.persistent
            }
        });
        // Expire persistent shapes.
        let now = Instant::now();
        self.persistent.retain(|s| now < s.expire_time);
    }

    /// Total number of lines queued this frame (including persistent).
    pub fn line_count(&self) -> usize {
        let persistent_count: usize = self.persistent.iter().map(|s| s.lines.len()).sum();
        self.lines_depth.len() + self.lines_overlay.len() + persistent_count
    }

    /// Get all 3D text labels.
    pub fn texts(&self) -> &[DebugText3D] {
        &self.texts
    }

    // -- Internal line pushing ---------------------------------------------

    fn push_line(&mut self, start: Vec3, end: Vec3, color: Color, depth_mode: DepthMode) {
        if !self.enabled {
            return;
        }
        let total = self.lines_depth.len() + self.lines_overlay.len();
        if total >= self.max_lines {
            return;
        }
        let line = DebugLine {
            start,
            end,
            color,
            depth_mode,
        };
        match depth_mode {
            DepthMode::DepthTested => self.lines_depth.push(line),
            DepthMode::AlwaysVisible => self.lines_overlay.push(line),
        }
    }

    fn push_line_default(&mut self, start: Vec3, end: Vec3, color: Color) {
        self.push_line(start, end, color, self.default_depth_mode);
    }

    fn push_persistent_line(
        &mut self,
        start: Vec3,
        end: Vec3,
        color: Color,
        depth_mode: DepthMode,
        duration: Duration,
    ) {
        let line = DebugLine {
            start,
            end,
            color,
            depth_mode,
        };
        self.persistent.push(PersistentShape {
            lines: vec![line],
            expire_time: Instant::now() + duration,
        });
    }

    // -- Drawing methods ---------------------------------------------------

    /// Draw a line segment.
    pub fn draw_line(&mut self, start: Vec3, end: Vec3, color: Color) {
        self.push_line_default(start, end, color);
    }

    /// Draw a line with explicit depth mode.
    pub fn draw_line_ex(
        &mut self,
        start: Vec3,
        end: Vec3,
        color: Color,
        depth_mode: DepthMode,
    ) {
        self.push_line(start, end, color, depth_mode);
    }

    /// Draw a persistent line that remains for `duration`.
    pub fn draw_line_persistent(
        &mut self,
        start: Vec3,
        end: Vec3,
        color: Color,
        duration: Duration,
    ) {
        self.push_persistent_line(start, end, color, self.default_depth_mode, duration);
    }

    /// Draw a ray from `origin` in `direction` with a given `length`.
    pub fn draw_ray(&mut self, origin: Vec3, direction: Vec3, length: f32, color: Color) {
        let end = origin + direction.normalize_or_zero() * length;
        self.push_line_default(origin, end, color);
    }

    /// Draw a wireframe sphere.
    pub fn draw_sphere(&mut self, center: Vec3, radius: f32, color: Color, segments: u32) {
        let segments = segments.max(4);
        // Draw three orthogonal circles.
        self.draw_circle(center, Vec3::Y, radius, color, segments);
        self.draw_circle(center, Vec3::X, radius, color, segments);
        self.draw_circle(center, Vec3::Z, radius, color, segments);
    }

    /// Draw a wireframe axis-aligned box.
    pub fn draw_box(&mut self, center: Vec3, half_extents: Vec3, rotation: Quat, color: Color) {
        // Compute the 8 corners.
        let corners_local = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];

        let corners: Vec<Vec3> = corners_local
            .iter()
            .map(|c| center + rotation * (*c * half_extents))
            .collect();

        // 12 edges of a box.
        let edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), // bottom face
            (4, 5), (5, 6), (6, 7), (7, 4), // top face
            (0, 4), (1, 5), (2, 6), (3, 7), // vertical edges
        ];

        for (a, b) in &edges {
            self.push_line_default(corners[*a], corners[*b], color);
        }
    }

    /// Draw a wireframe capsule between two points.
    pub fn draw_capsule(
        &mut self,
        start: Vec3,
        end: Vec3,
        radius: f32,
        color: Color,
    ) {
        let dir = end - start;
        let length = dir.length();
        if length < 1e-6 {
            self.draw_sphere(start, radius, color, 12);
            return;
        }

        let axis = dir / length;
        let segments = 16u32;

        // Find perpendicular vectors.
        let (perp1, perp2) = perpendicular_pair(axis);

        // Draw the cylinder body (4 lines along the length).
        for i in 0..4 {
            let angle = (i as f32) * PI * 0.5;
            let offset = (perp1 * angle.cos() + perp2 * angle.sin()) * radius;
            self.push_line_default(start + offset, end + offset, color);
        }

        // Draw hemisphere caps.
        self.draw_half_sphere(start, -axis, perp1, perp2, radius, segments, color);
        self.draw_half_sphere(end, axis, perp1, perp2, radius, segments, color);

        // Draw circles at the joins.
        self.draw_circle_oriented(start, axis, perp1, perp2, radius, color, segments);
        self.draw_circle_oriented(end, axis, perp1, perp2, radius, color, segments);
    }

    /// Draw a wireframe cylinder.
    pub fn draw_cylinder(
        &mut self,
        start: Vec3,
        end: Vec3,
        radius: f32,
        color: Color,
    ) {
        let dir = end - start;
        let length = dir.length();
        if length < 1e-6 {
            return;
        }
        let axis = dir / length;
        let (perp1, perp2) = perpendicular_pair(axis);
        let segments = 16u32;

        // Draw circles at top and bottom.
        self.draw_circle_oriented(start, axis, perp1, perp2, radius, color, segments);
        self.draw_circle_oriented(end, axis, perp1, perp2, radius, color, segments);

        // Draw vertical lines.
        for i in 0..8 {
            let angle = (i as f32) * PI * 0.25;
            let offset = (perp1 * angle.cos() + perp2 * angle.sin()) * radius;
            self.push_line_default(start + offset, end + offset, color);
        }
    }

    /// Draw a wireframe cone.
    pub fn draw_cone(
        &mut self,
        apex: Vec3,
        direction: Vec3,
        height: f32,
        angle: f32,
        color: Color,
    ) {
        let dir = direction.normalize_or_zero();
        let base_center = apex + dir * height;
        let base_radius = height * angle.tan();
        let (perp1, perp2) = perpendicular_pair(dir);
        let segments = 16u32;

        // Draw base circle.
        self.draw_circle_oriented(base_center, dir, perp1, perp2, base_radius, color, segments);

        // Draw lines from apex to base.
        for i in 0..8 {
            let a = (i as f32) * PI * 0.25;
            let base_pt =
                base_center + (perp1 * a.cos() + perp2 * a.sin()) * base_radius;
            self.push_line_default(apex, base_pt, color);
        }
    }

    /// Draw a wireframe view frustum from a view-projection matrix.
    pub fn draw_frustum(&mut self, view_proj: Mat4, color: Color) {
        let inv = view_proj.inverse();

        // NDC corners.
        let ndc_corners = [
            Vec3::new(-1.0, -1.0, 0.0), // near bottom-left
            Vec3::new(1.0, -1.0, 0.0),  // near bottom-right
            Vec3::new(1.0, 1.0, 0.0),   // near top-right
            Vec3::new(-1.0, 1.0, 0.0),  // near top-left
            Vec3::new(-1.0, -1.0, 1.0), // far bottom-left
            Vec3::new(1.0, -1.0, 1.0),  // far bottom-right
            Vec3::new(1.0, 1.0, 1.0),   // far top-right
            Vec3::new(-1.0, 1.0, 1.0),  // far top-left
        ];

        let world_corners: Vec<Vec3> = ndc_corners
            .iter()
            .map(|ndc| {
                let clip = inv * ndc.extend(1.0);
                Vec3::new(clip.x, clip.y, clip.z) / clip.w
            })
            .collect();

        // Near face.
        for i in 0..4 {
            self.push_line_default(world_corners[i], world_corners[(i + 1) % 4], color);
        }
        // Far face.
        for i in 4..8 {
            let next = 4 + (i - 4 + 1) % 4;
            self.push_line_default(world_corners[i], world_corners[next], color);
        }
        // Connecting edges.
        for i in 0..4 {
            self.push_line_default(world_corners[i], world_corners[i + 4], color);
        }
    }

    /// Draw a wireframe AABB.
    pub fn draw_aabb(&mut self, aabb: &AABB, color: Color) {
        let center = aabb.center();
        let half = aabb.half_extents();
        self.draw_box(center, half, Quat::IDENTITY, color);
    }

    /// Draw a grid on the XZ plane.
    pub fn draw_grid(&mut self, center: Vec3, size: f32, divisions: u32, color: Color) {
        let half = size * 0.5;
        let step = size / divisions as f32;

        for i in 0..=divisions {
            let offset = -half + step * i as f32;

            // Lines along Z.
            self.push_line_default(
                center + Vec3::new(offset, 0.0, -half),
                center + Vec3::new(offset, 0.0, half),
                color,
            );
            // Lines along X.
            self.push_line_default(
                center + Vec3::new(-half, 0.0, offset),
                center + Vec3::new(half, 0.0, offset),
                color,
            );
        }
    }

    /// Draw RGB-colored XYZ axes.
    pub fn draw_axis(&mut self, position: Vec3, rotation: Quat, size: f32) {
        let x = rotation * Vec3::X * size;
        let y = rotation * Vec3::Y * size;
        let z = rotation * Vec3::Z * size;

        self.push_line_default(position, position + x, Color::RED);
        self.push_line_default(position, position + y, Color::GREEN);
        self.push_line_default(position, position + z, Color::BLUE);
    }

    /// Draw a 3D text label.
    pub fn draw_text_3d(&mut self, position: Vec3, text: &str, color: Color) {
        self.texts.push(DebugText3D {
            position,
            text: text.to_string(),
            color,
            persistent: false,
            expire_time: None,
        });
    }

    /// Draw a persistent 3D text label.
    pub fn draw_text_3d_persistent(
        &mut self,
        position: Vec3,
        text: &str,
        color: Color,
        duration: Duration,
    ) {
        self.texts.push(DebugText3D {
            position,
            text: text.to_string(),
            color,
            persistent: true,
            expire_time: Some(Instant::now() + duration),
        });
    }

    /// Draw an arrow from `start` to `end` with an arrowhead.
    pub fn draw_arrow(
        &mut self,
        start: Vec3,
        end: Vec3,
        head_size: f32,
        color: Color,
    ) {
        let dir = end - start;
        let length = dir.length();
        if length < 1e-6 {
            return;
        }
        let axis = dir / length;

        // Main line.
        self.push_line_default(start, end, color);

        // Arrowhead.
        let (perp1, perp2) = perpendicular_pair(axis);
        let base = end - axis * head_size;
        let head_radius = head_size * 0.4;

        for i in 0..4 {
            let angle = (i as f32) * PI * 0.5;
            let offset = (perp1 * angle.cos() + perp2 * angle.sin()) * head_radius;
            self.push_line_default(end, base + offset, color);
        }
    }

    /// Draw a circle in 3D space.
    pub fn draw_circle(
        &mut self,
        center: Vec3,
        normal: Vec3,
        radius: f32,
        color: Color,
        segments: u32,
    ) {
        let normal = normal.normalize_or_zero();
        let (perp1, perp2) = perpendicular_pair(normal);
        self.draw_circle_oriented(center, normal, perp1, perp2, radius, color, segments);
    }

    /// Draw a circle (with explicit basis vectors).
    fn draw_circle_oriented(
        &mut self,
        center: Vec3,
        _normal: Vec3,
        perp1: Vec3,
        perp2: Vec3,
        radius: f32,
        color: Color,
        segments: u32,
    ) {
        let step = 2.0 * PI / segments as f32;
        let mut prev = center + perp1 * radius;
        for i in 1..=segments {
            let angle = step * i as f32;
            let pt = center + (perp1 * angle.cos() + perp2 * angle.sin()) * radius;
            self.push_line_default(prev, pt, color);
            prev = pt;
        }
    }

    /// Draw a half-sphere (hemisphere) cap.
    fn draw_half_sphere(
        &mut self,
        center: Vec3,
        pole: Vec3,
        perp1: Vec3,
        perp2: Vec3,
        radius: f32,
        segments: u32,
        color: Color,
    ) {
        let half_segments = segments / 2;
        for ring in 1..=half_segments {
            let phi = (ring as f32 / half_segments as f32) * PI * 0.5;
            let r = radius * phi.cos();
            let y_offset = pole * radius * phi.sin();
            let ring_center = center + y_offset;

            let step = 2.0 * PI / segments as f32;
            let mut prev = ring_center + perp1 * r;
            for i in 1..=segments {
                let angle = step * i as f32;
                let pt = ring_center + (perp1 * angle.cos() + perp2 * angle.sin()) * r;
                self.push_line_default(prev, pt, color);
                prev = pt;
            }
        }
    }

    /// Draw a Bezier curve through control points.
    pub fn draw_bezier(
        &mut self,
        points: &[Vec3],
        color: Color,
        segments: u32,
    ) {
        if points.len() < 2 {
            return;
        }
        let segments = segments.max(2);

        // Cubic Bezier: only use first 4 points.
        match points.len() {
            2 => {
                self.push_line_default(points[0], points[1], color);
            }
            3 => {
                // Quadratic Bezier.
                let mut prev = points[0];
                for i in 1..=segments {
                    let t = i as f32 / segments as f32;
                    let omt = 1.0 - t;
                    let pt = points[0] * omt * omt
                        + points[1] * 2.0 * omt * t
                        + points[2] * t * t;
                    self.push_line_default(prev, pt, color);
                    prev = pt;
                }
            }
            _ => {
                // Cubic Bezier using first 4 points.
                let p0 = points[0];
                let p1 = points[1];
                let p2 = points[2];
                let p3 = points[3.min(points.len() - 1)];
                let mut prev = p0;
                for i in 1..=segments {
                    let t = i as f32 / segments as f32;
                    let omt = 1.0 - t;
                    let pt = p0 * omt * omt * omt
                        + p1 * 3.0 * omt * omt * t
                        + p2 * 3.0 * omt * t * t
                        + p3 * t * t * t;
                    self.push_line_default(prev, pt, color);
                    prev = pt;
                }
            }
        }
    }

    /// Draw a bone as an octahedron shape between two joints.
    pub fn draw_bone(&mut self, start: Vec3, end: Vec3, color: Color) {
        let dir = end - start;
        let length = dir.length();
        if length < 1e-6 {
            return;
        }
        let axis = dir / length;
        let (perp1, perp2) = perpendicular_pair(axis);
        let mid = start + dir * 0.2;
        let width = length * 0.1;

        // Octahedron points.
        let top = mid + perp1 * width;
        let bottom = mid - perp1 * width;
        let left = mid + perp2 * width;
        let right = mid - perp2 * width;

        // From start.
        self.push_line_default(start, top, color);
        self.push_line_default(start, bottom, color);
        self.push_line_default(start, left, color);
        self.push_line_default(start, right, color);

        // Middle ring.
        self.push_line_default(top, left, color);
        self.push_line_default(left, bottom, color);
        self.push_line_default(bottom, right, color);
        self.push_line_default(right, top, color);

        // To end.
        self.push_line_default(top, end, color);
        self.push_line_default(bottom, end, color);
        self.push_line_default(left, end, color);
        self.push_line_default(right, end, color);
    }

    /// Draw a polyline path through a sequence of points.
    pub fn draw_path(&mut self, points: &[Vec3], color: Color) {
        if points.len() < 2 {
            return;
        }
        for window in points.windows(2) {
            self.push_line_default(window[0], window[1], color);
        }
    }

    /// Draw a skeleton (a collection of bones with parent indices).
    pub fn draw_skeleton(
        &mut self,
        bones: &[BoneInfo],
        color: Color,
    ) {
        for (i, bone) in bones.iter().enumerate() {
            if bone.parent_index >= 0 && (bone.parent_index as usize) < bones.len() {
                let parent = &bones[bone.parent_index as usize];
                let start = Vec3::new(
                    parent.world_transform.w_axis.x,
                    parent.world_transform.w_axis.y,
                    parent.world_transform.w_axis.z,
                );
                let end = Vec3::new(
                    bone.world_transform.w_axis.x,
                    bone.world_transform.w_axis.y,
                    bone.world_transform.w_axis.z,
                );
                self.draw_bone(start, end, color);
            } else {
                // Root bone: draw a small axis.
                let pos = Vec3::new(
                    bone.world_transform.w_axis.x,
                    bone.world_transform.w_axis.y,
                    bone.world_transform.w_axis.z,
                );
                self.draw_axis(pos, Quat::IDENTITY, 0.1);
            }
            let _ = i;
        }
    }

    // -- Vertex buffer building --------------------------------------------

    /// Build vertex buffers for rendering.
    ///
    /// Returns `(depth_tested_vertices, overlay_vertices)`, each a list of
    /// vertex pairs (start, end) for line rendering.
    pub fn build_vertex_buffers(&self) -> (Vec<DebugVertex>, Vec<DebugVertex>) {
        let now = Instant::now();

        // Depth-tested lines.
        let mut depth_verts = Vec::with_capacity(self.lines_depth.len() * 2);
        for line in &self.lines_depth {
            depth_verts.push(DebugVertex::new(line.start, line.color));
            depth_verts.push(DebugVertex::new(line.end, line.color));
        }

        // Overlay lines.
        let mut overlay_verts = Vec::with_capacity(self.lines_overlay.len() * 2);
        for line in &self.lines_overlay {
            overlay_verts.push(DebugVertex::new(line.start, line.color));
            overlay_verts.push(DebugVertex::new(line.end, line.color));
        }

        // Persistent shapes go into whichever buffer matches their depth mode.
        for shape in &self.persistent {
            if now >= shape.expire_time {
                continue;
            }
            for line in &shape.lines {
                let vert_start = DebugVertex::new(line.start, line.color);
                let vert_end = DebugVertex::new(line.end, line.color);
                match line.depth_mode {
                    DepthMode::DepthTested => {
                        depth_verts.push(vert_start);
                        depth_verts.push(vert_end);
                    }
                    DepthMode::AlwaysVisible => {
                        overlay_verts.push(vert_start);
                        overlay_verts.push(vert_end);
                    }
                }
            }
        }

        (depth_verts, overlay_verts)
    }

    /// Clear all shapes (single-frame and persistent).
    pub fn clear_all(&mut self) {
        self.lines_depth.clear();
        self.lines_overlay.clear();
        self.persistent.clear();
        self.texts.clear();
    }
}

impl Default for DebugRenderer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Compute a pair of perpendicular vectors to the given axis.
fn perpendicular_pair(axis: Vec3) -> (Vec3, Vec3) {
    let a = axis.normalize_or_zero();
    let candidate = if a.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let perp1 = a.cross(candidate).normalize_or_zero();
    let perp2 = a.cross(perp1).normalize_or_zero();
    (perp1, perp2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_constants() {
        assert_eq!(Color::RED, Color::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(Color::GREEN, Color::new(0.0, 1.0, 0.0, 1.0));
        assert_eq!(Color::BLUE, Color::new(0.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn color_lerp() {
        let c = Color::lerp(Color::BLACK, Color::WHITE, 0.5);
        assert!((c.r - 0.5).abs() < 1e-5);
        assert!((c.g - 0.5).abs() < 1e-5);
    }

    #[test]
    fn color_byte_conversion() {
        let c = Color::from_bytes(255, 128, 0, 255);
        let bytes = c.to_bytes();
        assert_eq!(bytes[0], 255);
        assert_eq!(bytes[2], 0);
    }

    #[test]
    fn draw_line() {
        let mut r = DebugRenderer::new();
        r.draw_line(Vec3::ZERO, Vec3::X, Color::RED);
        assert_eq!(r.line_count(), 1);
    }

    #[test]
    fn draw_sphere_generates_lines() {
        let mut r = DebugRenderer::new();
        r.draw_sphere(Vec3::ZERO, 1.0, Color::GREEN, 8);
        assert!(r.line_count() > 0);
    }

    #[test]
    fn draw_box_generates_12_edges() {
        let mut r = DebugRenderer::new();
        r.draw_box(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, Color::BLUE);
        assert_eq!(r.line_count(), 12);
    }

    #[test]
    fn draw_grid() {
        let mut r = DebugRenderer::new();
        r.draw_grid(Vec3::ZERO, 10.0, 10, Color::GRAY);
        // 11 lines in each direction = 22 lines.
        assert_eq!(r.line_count(), 22);
    }

    #[test]
    fn draw_axis_generates_3_lines() {
        let mut r = DebugRenderer::new();
        r.draw_axis(Vec3::ZERO, Quat::IDENTITY, 1.0);
        assert_eq!(r.line_count(), 3);
    }

    #[test]
    fn draw_arrow() {
        let mut r = DebugRenderer::new();
        r.draw_arrow(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.2, Color::YELLOW);
        assert!(r.line_count() >= 5); // 1 shaft + 4 head
    }

    #[test]
    fn draw_path() {
        let mut r = DebugRenderer::new();
        let points = vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y];
        r.draw_path(&points, Color::WHITE);
        assert_eq!(r.line_count(), 3);
    }

    #[test]
    fn draw_bone() {
        let mut r = DebugRenderer::new();
        r.draw_bone(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), Color::WHITE);
        assert_eq!(r.line_count(), 12); // 4 + 4 + 4
    }

    #[test]
    fn build_vertex_buffers() {
        let mut r = DebugRenderer::new();
        r.draw_line(Vec3::ZERO, Vec3::X, Color::RED);
        r.draw_line_ex(Vec3::ZERO, Vec3::Y, Color::GREEN, DepthMode::AlwaysVisible);

        let (depth, overlay) = r.build_vertex_buffers();
        assert_eq!(depth.len(), 2);
        assert_eq!(overlay.len(), 2);
    }

    #[test]
    fn begin_frame_clears_single_frame() {
        let mut r = DebugRenderer::new();
        r.draw_line(Vec3::ZERO, Vec3::X, Color::RED);
        assert_eq!(r.line_count(), 1);
        r.begin_frame();
        assert_eq!(r.line_count(), 0);
    }

    #[test]
    fn disabled_renderer_skips() {
        let mut r = DebugRenderer::new();
        r.set_enabled(false);
        r.draw_line(Vec3::ZERO, Vec3::X, Color::RED);
        assert_eq!(r.line_count(), 0);
    }

    #[test]
    fn draw_aabb() {
        let mut r = DebugRenderer::new();
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        r.draw_aabb(&aabb, Color::CYAN);
        assert_eq!(r.line_count(), 12);
    }

    #[test]
    fn draw_frustum() {
        let mut r = DebugRenderer::new();
        let vp = Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 100.0);
        r.draw_frustum(vp, Color::WHITE);
        assert_eq!(r.line_count(), 12); // 4 near + 4 far + 4 connecting
    }

    #[test]
    fn draw_text_3d() {
        let mut r = DebugRenderer::new();
        r.draw_text_3d(Vec3::ZERO, "Hello", Color::WHITE);
        assert_eq!(r.texts().len(), 1);
        assert_eq!(r.texts()[0].text, "Hello");
    }

    #[test]
    fn draw_bezier_cubic() {
        let mut r = DebugRenderer::new();
        let points = [Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y];
        r.draw_bezier(&points, Color::MAGENTA, 10);
        assert_eq!(r.line_count(), 10);
    }

    #[test]
    fn draw_cylinder() {
        let mut r = DebugRenderer::new();
        r.draw_cylinder(Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0), 0.5, Color::ORANGE);
        assert!(r.line_count() > 0);
    }

    #[test]
    fn draw_cone() {
        let mut r = DebugRenderer::new();
        r.draw_cone(Vec3::ZERO, Vec3::Y, 2.0, 0.5, Color::PURPLE);
        assert!(r.line_count() > 0);
    }

    #[test]
    fn perpendicular_pair_orthogonal() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let (p1, p2) = perpendicular_pair(axis);
        assert!(p1.dot(axis).abs() < 1e-5);
        assert!(p2.dot(axis).abs() < 1e-5);
        assert!(p1.dot(p2).abs() < 1e-5);
    }
}
