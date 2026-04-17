// engine/render/src/primitive_batch.rs
//
// Efficient batched primitive rendering for the Genovo engine.
//
// Provides instanced rendering of wireframe boxes, spheres, lines, arrows,
// circles, cones, and cylinders for debug visualization. All primitives are
// collected into batches per-type, sorted by depth test state, and rendered
// with minimal draw calls.
//
// # Architecture
//
// `PrimitiveBatch` collects draw commands during a frame. At flush time it
// sorts by render state (depth test on/off, line width) and issues
// instanced draws. Each primitive type has its own vertex layout and
// instance data format.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    #[inline]
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const RIGHT: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
    pub const FORWARD: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    #[inline]
    pub fn add(self, o: Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) }

    #[inline]
    pub fn sub(self, o: Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) }

    #[inline]
    pub fn scale(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }

    #[inline]
    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }

    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    #[inline]
    pub fn length(self) -> f32 { self.dot(self).sqrt() }

    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l > 1e-7 { self.scale(1.0 / l) } else { Self::ZERO }
    }

    #[inline]
    pub fn lerp(self, o: Self, t: f32) -> Self {
        self.scale(1.0 - t).add(o.scale(t))
    }
}

/// RGBA color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Self = Self { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Self = Self { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const CYAN: Self = Self { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const MAGENTA: Self = Self { r: 1.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const ORANGE: Self = Self { r: 1.0, g: 0.5, b: 0.0, a: 1.0 };
    pub const TRANSPARENT: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

    #[inline]
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self { r, g, b, a } }

    #[inline]
    pub fn rgb(r: f32, g: f32, b: f32) -> Self { Self { r, g, b, a: 1.0 } }

    #[inline]
    pub fn with_alpha(self, a: f32) -> Self { Self { a, ..self } }

    #[inline]
    pub fn to_array(self) -> [f32; 4] { [self.r, self.g, self.b, self.a] }

    /// Pack to u32 (RGBA8).
    pub fn to_u32(self) -> u32 {
        let r = (self.r.clamp(0.0, 1.0) * 255.0) as u32;
        let g = (self.g.clamp(0.0, 1.0) * 255.0) as u32;
        let b = (self.b.clamp(0.0, 1.0) * 255.0) as u32;
        let a = (self.a.clamp(0.0, 1.0) * 255.0) as u32;
        (r << 24) | (g << 16) | (b << 8) | a
    }

    /// Lerp between two colors.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of lines per batch.
pub const MAX_LINES_PER_BATCH: usize = 65536;

/// Maximum number of instances per draw call.
pub const MAX_INSTANCES_PER_DRAW: usize = 4096;

/// Default line width in pixels.
pub const DEFAULT_LINE_WIDTH: f32 = 1.0;

/// Number of segments for wireframe sphere.
pub const SPHERE_SEGMENTS: u32 = 24;

/// Number of segments for wireframe circle.
pub const CIRCLE_SEGMENTS: u32 = 32;

/// Arrow head length ratio (fraction of total arrow length).
pub const ARROW_HEAD_RATIO: f32 = 0.15;

/// Arrow head width ratio.
pub const ARROW_HEAD_WIDTH: f32 = 0.08;

/// Cone segments for arrow heads.
pub const ARROW_CONE_SEGMENTS: u32 = 8;

// ---------------------------------------------------------------------------
// Render state
// ---------------------------------------------------------------------------

/// Render state for a batch of primitives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrimitiveRenderState {
    /// Whether depth testing is enabled.
    pub depth_test: bool,
    /// Whether depth writing is enabled.
    pub depth_write: bool,
    /// Line width in pixels.
    pub line_width: f32,
    /// Whether to use screen-space line width (pixel) or world-space.
    pub screen_space_width: bool,
    /// Whether to enable alpha blending.
    pub alpha_blend: bool,
    /// Duration this primitive persists (0 = one frame).
    pub duration: f32,
}

impl Default for PrimitiveRenderState {
    fn default() -> Self {
        Self {
            depth_test: true,
            depth_write: false,
            line_width: DEFAULT_LINE_WIDTH,
            screen_space_width: true,
            alpha_blend: true,
            duration: 0.0,
        }
    }
}

/// State key for batching (primitives with same key are batched together).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BatchKey {
    depth_test: bool,
    depth_write: bool,
    line_width_bits: u32,
    alpha_blend: bool,
}

impl From<&PrimitiveRenderState> for BatchKey {
    fn from(state: &PrimitiveRenderState) -> Self {
        Self {
            depth_test: state.depth_test,
            depth_write: state.depth_write,
            line_width_bits: state.line_width.to_bits(),
            alpha_blend: state.alpha_blend,
        }
    }
}

// ---------------------------------------------------------------------------
// Line primitive
// ---------------------------------------------------------------------------

/// A single line segment.
#[derive(Debug, Clone, Copy)]
pub struct LinePrimitive {
    pub start: Vec3,
    pub end: Vec3,
    pub start_color: Color,
    pub end_color: Color,
}

impl LinePrimitive {
    /// Create a line with uniform color.
    pub fn new(start: Vec3, end: Vec3, color: Color) -> Self {
        Self {
            start,
            end,
            start_color: color,
            end_color: color,
        }
    }

    /// Create a line with gradient color.
    pub fn gradient(start: Vec3, end: Vec3, start_color: Color, end_color: Color) -> Self {
        Self {
            start,
            end,
            start_color,
            end_color,
        }
    }
}

// ---------------------------------------------------------------------------
// Box primitive
// ---------------------------------------------------------------------------

/// A wireframe box.
#[derive(Debug, Clone, Copy)]
pub struct BoxPrimitive {
    pub center: Vec3,
    pub half_extents: Vec3,
    pub color: Color,
    /// Rotation as axis-angle (if zero, no rotation).
    pub rotation_axis: Vec3,
    pub rotation_angle: f32,
}

impl BoxPrimitive {
    /// Create an axis-aligned wireframe box.
    pub fn new(center: Vec3, half_extents: Vec3, color: Color) -> Self {
        Self {
            center,
            half_extents,
            color,
            rotation_axis: Vec3::ZERO,
            rotation_angle: 0.0,
        }
    }

    /// Create a rotated wireframe box.
    pub fn rotated(center: Vec3, half_extents: Vec3, color: Color, axis: Vec3, angle: f32) -> Self {
        Self {
            center,
            half_extents,
            color,
            rotation_axis: axis.normalize(),
            rotation_angle: angle,
        }
    }

    /// Generate the 12 edges of this box as lines.
    pub fn to_lines(&self) -> Vec<LinePrimitive> {
        let he = self.half_extents;
        let corners = [
            Vec3::new(-he.x, -he.y, -he.z),
            Vec3::new(he.x, -he.y, -he.z),
            Vec3::new(he.x, he.y, -he.z),
            Vec3::new(-he.x, he.y, -he.z),
            Vec3::new(-he.x, -he.y, he.z),
            Vec3::new(he.x, -he.y, he.z),
            Vec3::new(he.x, he.y, he.z),
            Vec3::new(-he.x, he.y, he.z),
        ];

        // Apply rotation if needed.
        let transformed: Vec<Vec3> = if self.rotation_angle.abs() > 1e-6 {
            corners
                .iter()
                .map(|c| self.rotate_point(*c).add(self.center))
                .collect()
        } else {
            corners.iter().map(|c| c.add(self.center)).collect()
        };

        let edges: [(usize, usize); 12] = [
            (0, 1), (1, 2), (2, 3), (3, 0), // front face
            (4, 5), (5, 6), (6, 7), (7, 4), // back face
            (0, 4), (1, 5), (2, 6), (3, 7), // connections
        ];

        edges
            .iter()
            .map(|&(a, b)| LinePrimitive::new(transformed[a], transformed[b], self.color))
            .collect()
    }

    fn rotate_point(&self, p: Vec3) -> Vec3 {
        // Rodrigues' rotation formula.
        let k = self.rotation_axis;
        let cos_a = self.rotation_angle.cos();
        let sin_a = self.rotation_angle.sin();
        let dot = k.dot(p);
        let cross = k.cross(p);
        p.scale(cos_a).add(cross.scale(sin_a)).add(k.scale(dot * (1.0 - cos_a)))
    }
}

// ---------------------------------------------------------------------------
// Sphere primitive
// ---------------------------------------------------------------------------

/// A wireframe sphere.
#[derive(Debug, Clone, Copy)]
pub struct SpherePrimitive {
    pub center: Vec3,
    pub radius: f32,
    pub color: Color,
    pub segments: u32,
}

impl SpherePrimitive {
    pub fn new(center: Vec3, radius: f32, color: Color) -> Self {
        Self {
            center,
            radius,
            color,
            segments: SPHERE_SEGMENTS,
        }
    }

    /// Generate lines for 3 great circles (XY, XZ, YZ planes).
    pub fn to_lines(&self) -> Vec<LinePrimitive> {
        let mut lines = Vec::new();
        let n = self.segments;

        // XY circle.
        for i in 0..n {
            let a0 = (i as f32 / n as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / n as f32) * std::f32::consts::TAU;
            let p0 = self.center.add(Vec3::new(a0.cos() * self.radius, a0.sin() * self.radius, 0.0));
            let p1 = self.center.add(Vec3::new(a1.cos() * self.radius, a1.sin() * self.radius, 0.0));
            lines.push(LinePrimitive::new(p0, p1, self.color));
        }

        // XZ circle.
        for i in 0..n {
            let a0 = (i as f32 / n as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / n as f32) * std::f32::consts::TAU;
            let p0 = self.center.add(Vec3::new(a0.cos() * self.radius, 0.0, a0.sin() * self.radius));
            let p1 = self.center.add(Vec3::new(a1.cos() * self.radius, 0.0, a1.sin() * self.radius));
            lines.push(LinePrimitive::new(p0, p1, self.color));
        }

        // YZ circle.
        for i in 0..n {
            let a0 = (i as f32 / n as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / n as f32) * std::f32::consts::TAU;
            let p0 = self.center.add(Vec3::new(0.0, a0.cos() * self.radius, a0.sin() * self.radius));
            let p1 = self.center.add(Vec3::new(0.0, a1.cos() * self.radius, a1.sin() * self.radius));
            lines.push(LinePrimitive::new(p0, p1, self.color));
        }

        lines
    }
}

// ---------------------------------------------------------------------------
// Arrow primitive
// ---------------------------------------------------------------------------

/// A wireframe arrow (line + cone head).
#[derive(Debug, Clone, Copy)]
pub struct ArrowPrimitive {
    pub start: Vec3,
    pub end: Vec3,
    pub color: Color,
    pub head_size: f32,
}

impl ArrowPrimitive {
    pub fn new(start: Vec3, end: Vec3, color: Color) -> Self {
        let len = start.sub(end).length();
        Self {
            start,
            end,
            color,
            head_size: len * ARROW_HEAD_RATIO,
        }
    }

    pub fn with_head_size(start: Vec3, end: Vec3, color: Color, head_size: f32) -> Self {
        Self {
            start,
            end,
            color,
            head_size,
        }
    }

    /// Generate lines for this arrow.
    pub fn to_lines(&self) -> Vec<LinePrimitive> {
        let mut lines = Vec::new();

        let dir = self.end.sub(self.start);
        let len = dir.length();
        if len < 1e-6 {
            return lines;
        }
        let dir_n = dir.scale(1.0 / len);

        // Shaft line.
        let shaft_end = self.start.add(dir_n.scale(len - self.head_size));
        lines.push(LinePrimitive::new(self.start, shaft_end, self.color));

        // Arrow head: find perpendicular vectors.
        let up = if dir_n.dot(Vec3::UP).abs() < 0.99 {
            Vec3::UP
        } else {
            Vec3::RIGHT
        };
        let perp1 = dir_n.cross(up).normalize();
        let perp2 = dir_n.cross(perp1).normalize();

        let head_radius = self.head_size * 0.4;
        let n = ARROW_CONE_SEGMENTS;

        for i in 0..n {
            let a0 = (i as f32 / n as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / n as f32) * std::f32::consts::TAU;

            let p0 = shaft_end
                .add(perp1.scale(a0.cos() * head_radius))
                .add(perp2.scale(a0.sin() * head_radius));
            let p1 = shaft_end
                .add(perp1.scale(a1.cos() * head_radius))
                .add(perp2.scale(a1.sin() * head_radius));

            // Ring of the cone base.
            lines.push(LinePrimitive::new(p0, p1, self.color));
            // Lines from base to tip.
            lines.push(LinePrimitive::new(p0, self.end, self.color));
        }

        lines
    }
}

// ---------------------------------------------------------------------------
// Circle primitive
// ---------------------------------------------------------------------------

/// A wireframe circle.
#[derive(Debug, Clone, Copy)]
pub struct CirclePrimitive {
    pub center: Vec3,
    pub radius: f32,
    pub normal: Vec3,
    pub color: Color,
    pub segments: u32,
}

impl CirclePrimitive {
    pub fn new(center: Vec3, radius: f32, normal: Vec3, color: Color) -> Self {
        Self {
            center,
            radius,
            normal: normal.normalize(),
            color,
            segments: CIRCLE_SEGMENTS,
        }
    }

    /// Generate lines for this circle.
    pub fn to_lines(&self) -> Vec<LinePrimitive> {
        let mut lines = Vec::new();

        let up = if self.normal.dot(Vec3::UP).abs() < 0.99 {
            Vec3::UP
        } else {
            Vec3::RIGHT
        };
        let tangent = self.normal.cross(up).normalize();
        let bitangent = self.normal.cross(tangent).normalize();

        for i in 0..self.segments {
            let a0 = (i as f32 / self.segments as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / self.segments as f32) * std::f32::consts::TAU;

            let p0 = self.center
                .add(tangent.scale(a0.cos() * self.radius))
                .add(bitangent.scale(a0.sin() * self.radius));
            let p1 = self.center
                .add(tangent.scale(a1.cos() * self.radius))
                .add(bitangent.scale(a1.sin() * self.radius));

            lines.push(LinePrimitive::new(p0, p1, self.color));
        }

        lines
    }
}

// ---------------------------------------------------------------------------
// Grid primitive
// ---------------------------------------------------------------------------

/// A wireframe grid on a plane.
#[derive(Debug, Clone)]
pub struct GridPrimitive {
    pub center: Vec3,
    pub normal: Vec3,
    pub size: f32,
    pub divisions: u32,
    pub color: Color,
    pub subdivisions_color: Color,
    pub subdivisions: u32,
}

impl GridPrimitive {
    pub fn new(center: Vec3, normal: Vec3, size: f32, divisions: u32, color: Color) -> Self {
        Self {
            center,
            normal: normal.normalize(),
            size,
            divisions,
            color,
            subdivisions_color: color.with_alpha(color.a * 0.3),
            subdivisions: 0,
        }
    }

    /// Generate lines for this grid.
    pub fn to_lines(&self) -> Vec<LinePrimitive> {
        let mut lines = Vec::new();

        let up = if self.normal.dot(Vec3::UP).abs() < 0.99 {
            Vec3::UP
        } else {
            Vec3::RIGHT
        };
        let tangent = self.normal.cross(up).normalize();
        let bitangent = self.normal.cross(tangent).normalize();

        let half = self.size * 0.5;
        let step = self.size / self.divisions as f32;

        // Major grid lines.
        for i in 0..=self.divisions {
            let offset = -half + i as f32 * step;

            let start_a = self.center.add(tangent.scale(offset)).sub(bitangent.scale(half));
            let end_a = self.center.add(tangent.scale(offset)).add(bitangent.scale(half));
            lines.push(LinePrimitive::new(start_a, end_a, self.color));

            let start_b = self.center.add(bitangent.scale(offset)).sub(tangent.scale(half));
            let end_b = self.center.add(bitangent.scale(offset)).add(tangent.scale(half));
            lines.push(LinePrimitive::new(start_b, end_b, self.color));
        }

        // Subdivision lines.
        if self.subdivisions > 0 {
            let sub_step = step / (self.subdivisions + 1) as f32;
            for i in 0..self.divisions {
                for j in 1..=self.subdivisions {
                    let offset = -half + i as f32 * step + j as f32 * sub_step;

                    let start_a = self.center.add(tangent.scale(offset)).sub(bitangent.scale(half));
                    let end_a = self.center.add(tangent.scale(offset)).add(bitangent.scale(half));
                    lines.push(LinePrimitive::new(start_a, end_a, self.subdivisions_color));

                    let start_b = self.center.add(bitangent.scale(offset)).sub(tangent.scale(half));
                    let end_b = self.center.add(bitangent.scale(offset)).add(tangent.scale(half));
                    lines.push(LinePrimitive::new(start_b, end_b, self.subdivisions_color));
                }
            }
        }

        lines
    }
}

// ---------------------------------------------------------------------------
// Coordinate axes
// ---------------------------------------------------------------------------

/// Draw a set of coordinate axes (RGB for XYZ).
pub struct AxesPrimitive {
    pub origin: Vec3,
    pub length: f32,
}

impl AxesPrimitive {
    pub fn new(origin: Vec3, length: f32) -> Self {
        Self { origin, length }
    }

    pub fn to_arrows(&self) -> Vec<ArrowPrimitive> {
        vec![
            ArrowPrimitive::new(self.origin, self.origin.add(Vec3::RIGHT.scale(self.length)), Color::RED),
            ArrowPrimitive::new(self.origin, self.origin.add(Vec3::UP.scale(self.length)), Color::GREEN),
            ArrowPrimitive::new(self.origin, self.origin.add(Vec3::FORWARD.scale(self.length)), Color::BLUE),
        ]
    }
}

// ---------------------------------------------------------------------------
// Primitive batch
// ---------------------------------------------------------------------------

/// Collected primitive for deferred rendering.
#[derive(Debug, Clone)]
struct PrimitiveEntry {
    lines: Vec<LinePrimitive>,
    state: PrimitiveRenderState,
    remaining_duration: f32,
}

/// Statistics for the primitive batch.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrimitiveBatchStats {
    /// Total lines submitted this frame.
    pub total_lines: u32,
    /// Total draw calls issued.
    pub draw_calls: u32,
    /// Total instances drawn.
    pub total_instances: u32,
    /// Number of distinct batch keys.
    pub batch_count: u32,
    /// Number of persistent primitives.
    pub persistent_count: u32,
}

/// Batched primitive renderer.
///
/// Collects debug primitives during a frame and flushes them as batched
/// draw calls.
pub struct PrimitiveBatch {
    /// One-frame primitives (cleared every frame).
    frame_entries: Vec<PrimitiveEntry>,
    /// Persistent primitives (with duration > 0).
    persistent_entries: Vec<PrimitiveEntry>,
    /// Stats for the last flush.
    stats: PrimitiveBatchStats,
    /// Whether the batch is currently recording.
    recording: bool,
    /// Default render state for convenience methods.
    default_state: PrimitiveRenderState,
}

impl PrimitiveBatch {
    /// Create a new primitive batch.
    pub fn new() -> Self {
        Self {
            frame_entries: Vec::new(),
            persistent_entries: Vec::new(),
            stats: PrimitiveBatchStats::default(),
            recording: false,
            default_state: PrimitiveRenderState::default(),
        }
    }

    /// Begin recording primitives for a new frame.
    pub fn begin(&mut self) {
        self.frame_entries.clear();
        self.recording = true;
        self.stats = PrimitiveBatchStats::default();
    }

    /// Set the default render state for subsequent draw calls.
    pub fn set_default_state(&mut self, state: PrimitiveRenderState) {
        self.default_state = state;
    }

    /// Draw a line.
    pub fn draw_line(&mut self, start: Vec3, end: Vec3, color: Color) {
        self.draw_line_ex(start, end, color, &self.default_state.clone());
    }

    /// Draw a line with custom render state.
    pub fn draw_line_ex(&mut self, start: Vec3, end: Vec3, color: Color, state: &PrimitiveRenderState) {
        let entry = PrimitiveEntry {
            lines: vec![LinePrimitive::new(start, end, color)],
            state: *state,
            remaining_duration: state.duration,
        };
        self.add_entry(entry);
    }

    /// Draw a gradient line.
    pub fn draw_gradient_line(&mut self, start: Vec3, end: Vec3, start_color: Color, end_color: Color) {
        let entry = PrimitiveEntry {
            lines: vec![LinePrimitive::gradient(start, end, start_color, end_color)],
            state: self.default_state,
            remaining_duration: 0.0,
        };
        self.add_entry(entry);
    }

    /// Draw a wireframe box.
    pub fn draw_box(&mut self, center: Vec3, half_extents: Vec3, color: Color) {
        let prim = BoxPrimitive::new(center, half_extents, color);
        let entry = PrimitiveEntry {
            lines: prim.to_lines(),
            state: self.default_state,
            remaining_duration: 0.0,
        };
        self.add_entry(entry);
    }

    /// Draw a wireframe box with custom state.
    pub fn draw_box_ex(&mut self, center: Vec3, half_extents: Vec3, color: Color, state: &PrimitiveRenderState) {
        let prim = BoxPrimitive::new(center, half_extents, color);
        let entry = PrimitiveEntry {
            lines: prim.to_lines(),
            state: *state,
            remaining_duration: state.duration,
        };
        self.add_entry(entry);
    }

    /// Draw a rotated wireframe box.
    pub fn draw_box_rotated(&mut self, center: Vec3, half_extents: Vec3, color: Color, axis: Vec3, angle: f32) {
        let prim = BoxPrimitive::rotated(center, half_extents, color, axis, angle);
        let entry = PrimitiveEntry {
            lines: prim.to_lines(),
            state: self.default_state,
            remaining_duration: 0.0,
        };
        self.add_entry(entry);
    }

    /// Draw a wireframe sphere.
    pub fn draw_sphere(&mut self, center: Vec3, radius: f32, color: Color) {
        let prim = SpherePrimitive::new(center, radius, color);
        let entry = PrimitiveEntry {
            lines: prim.to_lines(),
            state: self.default_state,
            remaining_duration: 0.0,
        };
        self.add_entry(entry);
    }

    /// Draw an arrow.
    pub fn draw_arrow(&mut self, start: Vec3, end: Vec3, color: Color) {
        let prim = ArrowPrimitive::new(start, end, color);
        let entry = PrimitiveEntry {
            lines: prim.to_lines(),
            state: self.default_state,
            remaining_duration: 0.0,
        };
        self.add_entry(entry);
    }

    /// Draw a circle.
    pub fn draw_circle(&mut self, center: Vec3, radius: f32, normal: Vec3, color: Color) {
        let prim = CirclePrimitive::new(center, radius, normal, color);
        let entry = PrimitiveEntry {
            lines: prim.to_lines(),
            state: self.default_state,
            remaining_duration: 0.0,
        };
        self.add_entry(entry);
    }

    /// Draw coordinate axes.
    pub fn draw_axes(&mut self, origin: Vec3, length: f32) {
        let axes = AxesPrimitive::new(origin, length);
        for arrow in axes.to_arrows() {
            let entry = PrimitiveEntry {
                lines: arrow.to_lines(),
                state: self.default_state,
                remaining_duration: 0.0,
            };
            self.add_entry(entry);
        }
    }

    /// Draw a grid.
    pub fn draw_grid(&mut self, center: Vec3, normal: Vec3, size: f32, divisions: u32, color: Color) {
        let grid = GridPrimitive::new(center, normal, size, divisions, color);
        let entry = PrimitiveEntry {
            lines: grid.to_lines(),
            state: self.default_state,
            remaining_duration: 0.0,
        };
        self.add_entry(entry);
    }

    /// Draw a polyline (connected line segments).
    pub fn draw_polyline(&mut self, points: &[Vec3], color: Color, closed: bool) {
        if points.len() < 2 {
            return;
        }

        let mut lines = Vec::with_capacity(points.len());
        for i in 0..points.len() - 1 {
            lines.push(LinePrimitive::new(points[i], points[i + 1], color));
        }
        if closed && points.len() >= 3 {
            lines.push(LinePrimitive::new(points[points.len() - 1], points[0], color));
        }

        let entry = PrimitiveEntry {
            lines,
            state: self.default_state,
            remaining_duration: 0.0,
        };
        self.add_entry(entry);
    }

    /// Draw a frustum wireframe.
    pub fn draw_frustum(&mut self, corners: &[Vec3; 8], color: Color) {
        // corners: 0-3 near plane, 4-7 far plane.
        let mut lines = Vec::with_capacity(12);

        // Near face.
        for i in 0..4 {
            lines.push(LinePrimitive::new(corners[i], corners[(i + 1) % 4], color));
        }
        // Far face.
        for i in 0..4 {
            lines.push(LinePrimitive::new(corners[4 + i], corners[4 + (i + 1) % 4], color));
        }
        // Connections.
        for i in 0..4 {
            lines.push(LinePrimitive::new(corners[i], corners[i + 4], color));
        }

        let entry = PrimitiveEntry {
            lines,
            state: self.default_state,
            remaining_duration: 0.0,
        };
        self.add_entry(entry);
    }

    /// Add an entry, routing to frame or persistent storage.
    fn add_entry(&mut self, entry: PrimitiveEntry) {
        if entry.remaining_duration > 0.0 {
            self.persistent_entries.push(entry);
        } else {
            self.frame_entries.push(entry);
        }
    }

    /// Flush all collected primitives into sorted batches for rendering.
    /// Returns the batched line data ready for GPU upload.
    pub fn flush(&mut self) -> Vec<PrimitiveBatchData> {
        self.recording = false;

        // Combine frame and persistent entries.
        let mut all_entries: Vec<&PrimitiveEntry> = Vec::new();
        all_entries.extend(self.frame_entries.iter());
        all_entries.extend(self.persistent_entries.iter());

        // Group by batch key.
        let mut batches: HashMap<BatchKey, Vec<LinePrimitive>> = HashMap::new();
        for entry in &all_entries {
            let key = BatchKey::from(&entry.state);
            let batch = batches.entry(key).or_default();
            batch.extend_from_slice(&entry.lines);
        }

        let mut result = Vec::new();
        for (key, lines) in &batches {
            if lines.is_empty() {
                continue;
            }

            self.stats.batch_count += 1;
            self.stats.total_lines += lines.len() as u32;

            // Split into sub-batches if needed.
            for chunk in lines.chunks(MAX_LINES_PER_BATCH) {
                self.stats.draw_calls += 1;
                self.stats.total_instances += chunk.len() as u32;

                result.push(PrimitiveBatchData {
                    lines: chunk.to_vec(),
                    depth_test: key.depth_test,
                    depth_write: key.depth_write,
                    line_width: f32::from_bits(key.line_width_bits),
                    alpha_blend: key.alpha_blend,
                });
            }
        }

        self.stats.persistent_count = self.persistent_entries.len() as u32;

        result
    }

    /// Update persistent primitives (call with delta time each frame).
    pub fn update(&mut self, dt: f32) {
        self.persistent_entries.retain_mut(|entry| {
            entry.remaining_duration -= dt;
            entry.remaining_duration > 0.0
        });
    }

    /// Clear all primitives (both frame and persistent).
    pub fn clear(&mut self) {
        self.frame_entries.clear();
        self.persistent_entries.clear();
    }

    /// Get the stats from the last flush.
    pub fn stats(&self) -> &PrimitiveBatchStats {
        &self.stats
    }

    /// Whether the batch is currently recording.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Get the number of pending frame entries.
    pub fn frame_entry_count(&self) -> usize {
        self.frame_entries.len()
    }

    /// Get the number of persistent entries.
    pub fn persistent_entry_count(&self) -> usize {
        self.persistent_entries.len()
    }
}

/// Batch data ready for GPU upload.
#[derive(Debug, Clone)]
pub struct PrimitiveBatchData {
    pub lines: Vec<LinePrimitive>,
    pub depth_test: bool,
    pub depth_write: bool,
    pub line_width: f32,
    pub alpha_blend: bool,
}

impl PrimitiveBatchData {
    /// Get the number of lines in this batch.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Convert lines to vertex data (position + color interleaved).
    pub fn to_vertex_data(&self) -> Vec<f32> {
        let mut data = Vec::with_capacity(self.lines.len() * 2 * 7);
        for line in &self.lines {
            // Start vertex: pos(3) + color(4).
            data.push(line.start.x);
            data.push(line.start.y);
            data.push(line.start.z);
            data.push(line.start_color.r);
            data.push(line.start_color.g);
            data.push(line.start_color.b);
            data.push(line.start_color.a);

            // End vertex.
            data.push(line.end.x);
            data.push(line.end.y);
            data.push(line.end.z);
            data.push(line.end_color.r);
            data.push(line.end_color.g);
            data.push(line.end_color.b);
            data.push(line.end_color.a);
        }
        data
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_to_lines() {
        let b = BoxPrimitive::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), Color::RED);
        let lines = b.to_lines();
        assert_eq!(lines.len(), 12);
    }

    #[test]
    fn test_sphere_to_lines() {
        let s = SpherePrimitive::new(Vec3::ZERO, 1.0, Color::GREEN);
        let lines = s.to_lines();
        assert_eq!(lines.len(), SPHERE_SEGMENTS as usize * 3);
    }

    #[test]
    fn test_arrow_to_lines() {
        let a = ArrowPrimitive::new(Vec3::ZERO, Vec3::new(0.0, 10.0, 0.0), Color::BLUE);
        let lines = a.to_lines();
        // 1 shaft + ARROW_CONE_SEGMENTS * 2 (ring + tip lines).
        assert_eq!(lines.len(), 1 + ARROW_CONE_SEGMENTS as usize * 2);
    }

    #[test]
    fn test_primitive_batch() {
        let mut batch = PrimitiveBatch::new();
        batch.begin();
        batch.draw_line(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Color::RED);
        batch.draw_box(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), Color::GREEN);
        batch.draw_sphere(Vec3::ZERO, 1.0, Color::BLUE);

        let batches = batch.flush();
        assert!(!batches.is_empty());
        assert!(batch.stats().total_lines > 0);
    }

    #[test]
    fn test_color_u32() {
        let c = Color::new(1.0, 0.0, 0.0, 1.0);
        let packed = c.to_u32();
        assert_eq!(packed, 0xFF0000FF);
    }

    #[test]
    fn test_batch_data_vertex_conversion() {
        let data = PrimitiveBatchData {
            lines: vec![LinePrimitive::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Color::WHITE)],
            depth_test: true,
            depth_write: false,
            line_width: 1.0,
            alpha_blend: true,
        };
        let verts = data.to_vertex_data();
        // 1 line * 2 vertices * 7 floats.
        assert_eq!(verts.len(), 14);
    }

    #[test]
    fn test_persistent_primitives() {
        let mut batch = PrimitiveBatch::new();
        batch.begin();

        let state = PrimitiveRenderState {
            duration: 2.0,
            ..Default::default()
        };
        batch.draw_line_ex(Vec3::ZERO, Vec3::UP, Color::RED, &state);
        assert_eq!(batch.persistent_entry_count(), 1);

        batch.update(1.0);
        assert_eq!(batch.persistent_entry_count(), 1);

        batch.update(1.5);
        assert_eq!(batch.persistent_entry_count(), 0);
    }
}
