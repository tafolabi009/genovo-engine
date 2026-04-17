// engine/render/src/line_renderer.rs
//
// Anti-aliased line rendering system for the Genovo engine.
//
// Provides GPU-friendly thick line rendering with various styles:
//
// - **Thick lines** — Configurable width with anti-aliased edges.
// - **Dashed/dotted lines** — Parametric dash patterns along the line.
// - **Line caps** — Butt, round, and square end caps.
// - **Line joins** — Miter, bevel, and round joins at polyline vertices.
// - **Polylines** — Connected sequences of line segments with proper joins.
// - **Bezier curves** — Quadratic and cubic Bezier curve rendering via
//   adaptive tessellation.
// - **Screen-space line width** — Width specified in pixels, independent of
//   camera distance.
// - **Depth-tested and overlay modes** — Lines can participate in depth
//   testing or render on top of everything.
//
// # Pipeline integration
//
// Lines are rendered as triangle strips (two triangles per segment, expanded
// perpendicular to the line direction). The expansion happens in the vertex
// shader for screen-space width, or on the CPU for world-space width.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Line style types
// ---------------------------------------------------------------------------

/// Line cap style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineCap {
    /// Flat cap at the line endpoint (default).
    Butt,
    /// Semicircular cap extending beyond the endpoint by half the line width.
    Round,
    /// Rectangular cap extending beyond the endpoint by half the line width.
    Square,
}

/// Line join style for polylines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineJoin {
    /// Sharp corner (may be clamped to miter limit).
    Miter,
    /// Bevelled corner (flat cut across the join).
    Bevel,
    /// Rounded corner (arc between the two edges).
    Round,
}

/// Dash pattern for dashed/dotted lines.
#[derive(Debug, Clone)]
pub struct DashPattern {
    /// Alternating dash and gap lengths (in world or screen units).
    /// E.g. [10.0, 5.0] = 10 units dash, 5 units gap, repeating.
    pub pattern: Vec<f32>,
    /// Offset along the pattern (for animation).
    pub offset: f32,
}

impl DashPattern {
    /// Solid line (no dashes).
    pub fn solid() -> Self {
        Self { pattern: Vec::new(), offset: 0.0 }
    }

    /// Simple dashed pattern.
    pub fn dashed(dash: f32, gap: f32) -> Self {
        Self { pattern: vec![dash, gap], offset: 0.0 }
    }

    /// Dotted pattern (short dash, equal gap).
    pub fn dotted(size: f32) -> Self {
        Self { pattern: vec![size, size], offset: 0.0 }
    }

    /// Dash-dot pattern.
    pub fn dash_dot(dash: f32, dot: f32, gap: f32) -> Self {
        Self { pattern: vec![dash, gap, dot, gap], offset: 0.0 }
    }

    /// Set the offset.
    pub fn with_offset(mut self, offset: f32) -> Self {
        self.offset = offset;
        self
    }

    /// Total pattern length.
    pub fn total_length(&self) -> f32 {
        self.pattern.iter().sum()
    }

    /// Check whether a point at `dist` along the line is visible (dash) or hidden (gap).
    pub fn is_visible(&self, dist: f32) -> bool {
        if self.pattern.is_empty() {
            return true; // Solid line.
        }

        let total = self.total_length();
        if total <= 0.0 {
            return true;
        }

        let d = ((dist + self.offset) % total + total) % total;
        let mut accum = 0.0;
        for (i, &seg) in self.pattern.iter().enumerate() {
            accum += seg;
            if d < accum {
                return i % 2 == 0; // Even indices are dashes.
            }
        }
        true
    }

    /// Compute the dash visibility factor with anti-aliasing (soft edges).
    ///
    /// Returns a value in [0, 1] where 0 = gap, 1 = dash.
    pub fn visibility_factor(&self, dist: f32, aa_width: f32) -> f32 {
        if self.pattern.is_empty() {
            return 1.0;
        }

        let total = self.total_length();
        if total <= 0.0 {
            return 1.0;
        }

        let d = ((dist + self.offset) % total + total) % total;
        let mut accum = 0.0;
        let half_aa = aa_width * 0.5;

        for (i, &seg) in self.pattern.iter().enumerate() {
            let prev_accum = accum;
            accum += seg;

            if d < accum {
                let is_dash = i % 2 == 0;
                if is_dash {
                    // Smooth transition at the end of the dash.
                    let to_end = accum - d;
                    let from_start = d - prev_accum;
                    let edge = to_end.min(from_start);
                    return (edge / half_aa).clamp(0.0, 1.0);
                } else {
                    // In a gap.
                    let to_end = accum - d;
                    let from_start = d - prev_accum;
                    let edge = to_end.min(from_start);
                    return 1.0 - (edge / half_aa).clamp(0.0, 1.0);
                }
            }
        }
        1.0
    }
}

/// Rendering mode for lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineDepthMode {
    /// Lines participate in depth testing.
    DepthTested,
    /// Lines always render on top (overlay/debug).
    Overlay,
    /// Lines render on top but with transparency based on depth.
    OverlayFaded,
}

// ---------------------------------------------------------------------------
// Line style configuration
// ---------------------------------------------------------------------------

/// Complete line style specification.
#[derive(Debug, Clone)]
pub struct LineStyle {
    /// Line width in pixels (screen space) or world units (world space).
    pub width: f32,
    /// Line colour (linear RGBA).
    pub color: [f32; 4],
    /// Line cap style.
    pub cap: LineCap,
    /// Line join style.
    pub join: LineJoin,
    /// Miter limit (angle in radians; joints sharper than this become bevels).
    pub miter_limit: f32,
    /// Dash pattern.
    pub dash: DashPattern,
    /// Depth mode.
    pub depth_mode: LineDepthMode,
    /// Whether the width is in screen pixels (true) or world units (false).
    pub screen_space_width: bool,
    /// Anti-aliasing width in pixels.
    pub aa_width: f32,
    /// Opacity (multiplied with colour alpha).
    pub opacity: f32,
}

impl Default for LineStyle {
    fn default() -> Self {
        Self {
            width: 2.0,
            color: [1.0, 1.0, 1.0, 1.0],
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: 0.5, // ~28.9 degrees.
            dash: DashPattern::solid(),
            depth_mode: LineDepthMode::DepthTested,
            screen_space_width: true,
            aa_width: 1.5,
            opacity: 1.0,
        }
    }
}

impl LineStyle {
    /// Create a style with the given width and colour.
    pub fn new(width: f32, color: [f32; 4]) -> Self {
        Self { width, color, ..Self::default() }
    }

    /// Set the cap style.
    pub fn with_cap(mut self, cap: LineCap) -> Self {
        self.cap = cap;
        self
    }

    /// Set the join style.
    pub fn with_join(mut self, join: LineJoin) -> Self {
        self.join = join;
        self
    }

    /// Set the dash pattern.
    pub fn with_dash(mut self, dash: DashPattern) -> Self {
        self.dash = dash;
        self
    }

    /// Set overlay mode.
    pub fn overlay(mut self) -> Self {
        self.depth_mode = LineDepthMode::Overlay;
        self
    }
}

// ---------------------------------------------------------------------------
// Line vertex
// ---------------------------------------------------------------------------

/// Vertex layout for line rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LineVertex {
    /// Position (x, y, z).
    pub position: [f32; 3],
    /// Normal for extrusion (x, y, z).
    pub normal: [f32; 3],
    /// Colour (r, g, b, a).
    pub color: [f32; 4],
    /// Line distance along the path (for dash patterns).
    pub line_dist: f32,
    /// Signed distance from the line centre (for AA, -1 to +1).
    pub edge_dist: f32,
}

// ---------------------------------------------------------------------------
// Line segment and polyline
// ---------------------------------------------------------------------------

/// A single line segment between two 3D points.
#[derive(Debug, Clone, Copy)]
pub struct LineSegment {
    pub start: [f32; 3],
    pub end: [f32; 3],
}

impl LineSegment {
    pub fn new(start: [f32; 3], end: [f32; 3]) -> Self {
        Self { start, end }
    }

    /// Length of the segment.
    pub fn length(&self) -> f32 {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        let dz = self.end[2] - self.start[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Direction vector (normalised).
    pub fn direction(&self) -> [f32; 3] {
        let len = self.length();
        if len < 1e-10 {
            return [1.0, 0.0, 0.0];
        }
        let inv = 1.0 / len;
        [
            (self.end[0] - self.start[0]) * inv,
            (self.end[1] - self.start[1]) * inv,
            (self.end[2] - self.start[2]) * inv,
        ]
    }

    /// Point at parameter t [0, 1].
    pub fn lerp(&self, t: f32) -> [f32; 3] {
        [
            self.start[0] + (self.end[0] - self.start[0]) * t,
            self.start[1] + (self.end[1] - self.start[1]) * t,
            self.start[2] + (self.end[2] - self.start[2]) * t,
        ]
    }
}

/// A polyline: connected sequence of points.
#[derive(Debug, Clone)]
pub struct Polyline {
    /// Points of the polyline.
    pub points: Vec<[f32; 3]>,
    /// Whether the polyline is closed (last point connects to first).
    pub closed: bool,
}

impl Polyline {
    /// Create an open polyline.
    pub fn new(points: Vec<[f32; 3]>) -> Self {
        Self { points, closed: false }
    }

    /// Create a closed polyline.
    pub fn closed(points: Vec<[f32; 3]>) -> Self {
        Self { points, closed: true }
    }

    /// Total path length.
    pub fn total_length(&self) -> f32 {
        if self.points.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        for i in 0..self.points.len() - 1 {
            let seg = LineSegment::new(self.points[i], self.points[i + 1]);
            total += seg.length();
        }
        if self.closed && self.points.len() > 2 {
            let seg = LineSegment::new(*self.points.last().unwrap(), self.points[0]);
            total += seg.length();
        }
        total
    }

    /// Number of segments.
    pub fn segment_count(&self) -> usize {
        if self.points.len() < 2 {
            return 0;
        }
        let n = self.points.len() - 1;
        if self.closed { n + 1 } else { n }
    }

    /// Get a segment by index.
    pub fn segment(&self, index: usize) -> Option<LineSegment> {
        let count = self.segment_count();
        if index >= count {
            return None;
        }
        let next = if index + 1 >= self.points.len() {
            0
        } else {
            index + 1
        };
        Some(LineSegment::new(self.points[index], self.points[next]))
    }
}

// ---------------------------------------------------------------------------
// Bezier curves
// ---------------------------------------------------------------------------

/// A quadratic Bezier curve (3 control points).
#[derive(Debug, Clone, Copy)]
pub struct QuadBezier {
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub p2: [f32; 3],
}

impl QuadBezier {
    pub fn new(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> Self {
        Self { p0, p1, p2 }
    }

    /// Evaluate the curve at parameter t [0, 1].
    pub fn evaluate(&self, t: f32) -> [f32; 3] {
        let omt = 1.0 - t;
        let omt2 = omt * omt;
        let t2 = t * t;
        [
            omt2 * self.p0[0] + 2.0 * omt * t * self.p1[0] + t2 * self.p2[0],
            omt2 * self.p0[1] + 2.0 * omt * t * self.p1[1] + t2 * self.p2[1],
            omt2 * self.p0[2] + 2.0 * omt * t * self.p1[2] + t2 * self.p2[2],
        ]
    }

    /// Evaluate the tangent at parameter t.
    pub fn tangent(&self, t: f32) -> [f32; 3] {
        let omt = 1.0 - t;
        [
            2.0 * omt * (self.p1[0] - self.p0[0]) + 2.0 * t * (self.p2[0] - self.p1[0]),
            2.0 * omt * (self.p1[1] - self.p0[1]) + 2.0 * t * (self.p2[1] - self.p1[1]),
            2.0 * omt * (self.p1[2] - self.p0[2]) + 2.0 * t * (self.p2[2] - self.p1[2]),
        ]
    }

    /// Tessellate the curve into a polyline with adaptive subdivision.
    pub fn tessellate(&self, max_segments: u32) -> Polyline {
        let segments = max_segments.max(2);
        let mut points = Vec::with_capacity(segments as usize + 1);
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            points.push(self.evaluate(t));
        }
        Polyline::new(points)
    }

    /// Approximate arc length.
    pub fn approx_length(&self, samples: u32) -> f32 {
        let mut total = 0.0;
        let mut prev = self.evaluate(0.0);
        for i in 1..=samples {
            let t = i as f32 / samples as f32;
            let curr = self.evaluate(t);
            let dx = curr[0] - prev[0];
            let dy = curr[1] - prev[1];
            let dz = curr[2] - prev[2];
            total += (dx * dx + dy * dy + dz * dz).sqrt();
            prev = curr;
        }
        total
    }
}

/// A cubic Bezier curve (4 control points).
#[derive(Debug, Clone, Copy)]
pub struct CubicBezier {
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub p2: [f32; 3],
    pub p3: [f32; 3],
}

impl CubicBezier {
    pub fn new(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Evaluate the curve at parameter t [0, 1].
    pub fn evaluate(&self, t: f32) -> [f32; 3] {
        let omt = 1.0 - t;
        let omt2 = omt * omt;
        let omt3 = omt2 * omt;
        let t2 = t * t;
        let t3 = t2 * t;
        [
            omt3 * self.p0[0] + 3.0 * omt2 * t * self.p1[0] + 3.0 * omt * t2 * self.p2[0] + t3 * self.p3[0],
            omt3 * self.p0[1] + 3.0 * omt2 * t * self.p1[1] + 3.0 * omt * t2 * self.p2[1] + t3 * self.p3[1],
            omt3 * self.p0[2] + 3.0 * omt2 * t * self.p1[2] + 3.0 * omt * t2 * self.p2[2] + t3 * self.p3[2],
        ]
    }

    /// Evaluate the tangent at parameter t.
    pub fn tangent(&self, t: f32) -> [f32; 3] {
        let omt = 1.0 - t;
        let omt2 = omt * omt;
        let t2 = t * t;
        [
            3.0 * omt2 * (self.p1[0] - self.p0[0]) + 6.0 * omt * t * (self.p2[0] - self.p1[0]) + 3.0 * t2 * (self.p3[0] - self.p2[0]),
            3.0 * omt2 * (self.p1[1] - self.p0[1]) + 6.0 * omt * t * (self.p2[1] - self.p1[1]) + 3.0 * t2 * (self.p3[1] - self.p2[1]),
            3.0 * omt2 * (self.p1[2] - self.p0[2]) + 6.0 * omt * t * (self.p2[2] - self.p1[2]) + 3.0 * t2 * (self.p3[2] - self.p2[2]),
        ]
    }

    /// Tessellate the curve.
    pub fn tessellate(&self, max_segments: u32) -> Polyline {
        let segments = max_segments.max(2);
        let mut points = Vec::with_capacity(segments as usize + 1);
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            points.push(self.evaluate(t));
        }
        Polyline::new(points)
    }

    /// Approximate arc length.
    pub fn approx_length(&self, samples: u32) -> f32 {
        let mut total = 0.0;
        let mut prev = self.evaluate(0.0);
        for i in 1..=samples {
            let t = i as f32 / samples as f32;
            let curr = self.evaluate(t);
            let dx = curr[0] - prev[0];
            let dy = curr[1] - prev[1];
            let dz = curr[2] - prev[2];
            total += (dx * dx + dy * dy + dz * dz).sqrt();
            prev = curr;
        }
        total
    }

    /// Split the curve at parameter t into two cubic Beziers (de Casteljau).
    pub fn split(&self, t: f32) -> (CubicBezier, CubicBezier) {
        let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };

        let m01 = lerp3(self.p0, self.p1, t);
        let m12 = lerp3(self.p1, self.p2, t);
        let m23 = lerp3(self.p2, self.p3, t);
        let m012 = lerp3(m01, m12, t);
        let m123 = lerp3(m12, m23, t);
        let m0123 = lerp3(m012, m123, t);

        (
            CubicBezier::new(self.p0, m01, m012, m0123),
            CubicBezier::new(m0123, m123, m23, self.p3),
        )
    }
}

// ---------------------------------------------------------------------------
// Line mesh generation
// ---------------------------------------------------------------------------

/// Generate triangle-strip vertices for a polyline with the given style.
///
/// # Arguments
/// * `polyline` — The polyline to render.
/// * `style` — Line style.
/// * `camera_pos` — Camera position (for billboard-style width computation).
///
/// # Returns
/// (vertices, indices) for rendering.
pub fn generate_line_mesh(
    polyline: &Polyline,
    style: &LineStyle,
    camera_pos: [f32; 3],
) -> (Vec<LineVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    if polyline.points.len() < 2 {
        return (vertices, indices);
    }

    let half_width = style.width * 0.5;
    let mut cumulative_dist = 0.0_f32;

    let num_points = polyline.points.len();
    let segment_count = polyline.segment_count();

    for i in 0..num_points {
        let p = polyline.points[i];

        // Compute tangent at this point.
        let tangent = if i == 0 && !polyline.closed {
            // First point: use the direction of the first segment.
            let next = polyline.points[1];
            normalize3([next[0] - p[0], next[1] - p[1], next[2] - p[2]])
        } else if i == num_points - 1 && !polyline.closed {
            // Last point: use the direction of the last segment.
            let prev = polyline.points[i - 1];
            normalize3([p[0] - prev[0], p[1] - prev[1], p[2] - prev[2]])
        } else {
            // Interior point (or closed polyline): average of adjacent segments.
            let prev_idx = if i == 0 { num_points - 1 } else { i - 1 };
            let next_idx = if i == num_points - 1 { 0 } else { i + 1 };
            let prev = polyline.points[prev_idx];
            let next = polyline.points[next_idx];
            let t1 = normalize3([p[0] - prev[0], p[1] - prev[1], p[2] - prev[2]]);
            let t2 = normalize3([next[0] - p[0], next[1] - p[1], next[2] - p[2]]);
            normalize3([t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2]])
        };

        // Compute a perpendicular direction for width expansion.
        // Use the direction to the camera for billboard-style lines.
        let to_cam = normalize3([
            camera_pos[0] - p[0],
            camera_pos[1] - p[1],
            camera_pos[2] - p[2],
        ]);
        let right = cross3(tangent, to_cam);
        let right = normalize3(right);

        // Handle miter correction.
        let miter_scale = compute_miter_scale(i, polyline, &style);

        let effective_width = half_width * miter_scale;

        // Cumulative distance for dash pattern.
        if i > 0 {
            let prev = polyline.points[i - 1];
            let dx = p[0] - prev[0];
            let dy = p[1] - prev[1];
            let dz = p[2] - prev[2];
            cumulative_dist += (dx * dx + dy * dy + dz * dz).sqrt();
        }

        let color = style.color;

        // Two vertices: left and right of the line centre.
        let base_idx = vertices.len() as u32;

        vertices.push(LineVertex {
            position: [
                p[0] - right[0] * effective_width,
                p[1] - right[1] * effective_width,
                p[2] - right[2] * effective_width,
            ],
            normal: [-right[0], -right[1], -right[2]],
            color,
            line_dist: cumulative_dist,
            edge_dist: -1.0,
        });

        vertices.push(LineVertex {
            position: [
                p[0] + right[0] * effective_width,
                p[1] + right[1] * effective_width,
                p[2] + right[2] * effective_width,
            ],
            normal: [right[0], right[1], right[2]],
            color,
            line_dist: cumulative_dist,
            edge_dist: 1.0,
        });

        // Generate indices for triangle strip.
        if i > 0 {
            let prev_base = base_idx - 2;
            indices.push(prev_base);
            indices.push(prev_base + 1);
            indices.push(base_idx);

            indices.push(prev_base + 1);
            indices.push(base_idx + 1);
            indices.push(base_idx);
        }
    }

    // Close the polyline if needed.
    if polyline.closed && num_points > 2 {
        let last_base = (vertices.len() - 2) as u32;
        indices.push(last_base);
        indices.push(last_base + 1);
        indices.push(0);

        indices.push(last_base + 1);
        indices.push(1);
        indices.push(0);
    }

    (vertices, indices)
}

/// Generate cap vertices for a line endpoint.
pub fn generate_cap_vertices(
    point: [f32; 3],
    direction: [f32; 3],
    right: [f32; 3],
    half_width: f32,
    cap: LineCap,
    is_start: bool,
    color: [f32; 4],
    line_dist: f32,
) -> Vec<LineVertex> {
    let mut verts = Vec::new();

    match cap {
        LineCap::Butt => {
            // No extra vertices needed.
        }
        LineCap::Square => {
            // Extend the line by half_width in the direction.
            let ext = if is_start { -1.0 } else { 1.0 };
            let tip = [
                point[0] + direction[0] * half_width * ext,
                point[1] + direction[1] * half_width * ext,
                point[2] + direction[2] * half_width * ext,
            ];

            verts.push(LineVertex {
                position: [
                    tip[0] - right[0] * half_width,
                    tip[1] - right[1] * half_width,
                    tip[2] - right[2] * half_width,
                ],
                normal: [-right[0], -right[1], -right[2]],
                color,
                line_dist,
                edge_dist: -1.0,
            });
            verts.push(LineVertex {
                position: [
                    tip[0] + right[0] * half_width,
                    tip[1] + right[1] * half_width,
                    tip[2] + right[2] * half_width,
                ],
                normal: [right[0], right[1], right[2]],
                color,
                line_dist,
                edge_dist: 1.0,
            });
        }
        LineCap::Round => {
            // Generate a semicircle of vertices.
            let segments = 8;
            let ext = if is_start { -1.0 } else { 1.0 };
            let up = cross3(direction, right);

            for i in 0..=segments {
                let angle = PI * i as f32 / segments as f32;
                let cos_a = angle.cos() * ext;
                let sin_a = angle.sin();

                let pos = [
                    point[0] + (direction[0] * cos_a + up[0] * sin_a) * half_width,
                    point[1] + (direction[1] * cos_a + up[1] * sin_a) * half_width,
                    point[2] + (direction[2] * cos_a + up[2] * sin_a) * half_width,
                ];

                verts.push(LineVertex {
                    position: pos,
                    normal: normalize3([
                        pos[0] - point[0],
                        pos[1] - point[1],
                        pos[2] - point[2],
                    ]),
                    color,
                    line_dist,
                    edge_dist: 0.0,
                });
            }
        }
    }

    verts
}

/// Compute the miter scale factor at a polyline vertex.
fn compute_miter_scale(index: usize, polyline: &Polyline, style: &LineStyle) -> f32 {
    let num_points = polyline.points.len();
    if num_points < 3 {
        return 1.0;
    }

    let is_endpoint = !polyline.closed && (index == 0 || index == num_points - 1);
    if is_endpoint {
        return 1.0;
    }

    let prev_idx = if index == 0 { num_points - 1 } else { index - 1 };
    let next_idx = if index == num_points - 1 { 0 } else { index + 1 };

    let p = polyline.points[index];
    let prev = polyline.points[prev_idx];
    let next = polyline.points[next_idx];

    let t1 = normalize3([p[0] - prev[0], p[1] - prev[1], p[2] - prev[2]]);
    let t2 = normalize3([next[0] - p[0], next[1] - p[1], next[2] - p[2]]);

    let dot = t1[0] * t2[0] + t1[1] * t2[1] + t1[2] * t2[2];
    let half_angle = ((1.0 + dot) * 0.5).sqrt();

    if half_angle < 1e-6 {
        return 1.0;
    }

    let miter = 1.0 / half_angle;

    match style.join {
        LineJoin::Miter => {
            if half_angle < style.miter_limit.cos() {
                1.0 // Fall back to bevel.
            } else {
                miter.min(4.0) // Clamp miter to prevent extreme spikes.
            }
        }
        LineJoin::Bevel => 1.0,
        LineJoin::Round => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Line renderer
// ---------------------------------------------------------------------------

/// Submission entry for the line renderer.
#[derive(Debug, Clone)]
pub struct LineDrawCall {
    /// Vertices.
    pub vertices: Vec<LineVertex>,
    /// Indices.
    pub indices: Vec<u32>,
    /// Depth mode.
    pub depth_mode: LineDepthMode,
    /// Atlas/pattern texture (0 = none).
    pub texture: u64,
}

/// The line renderer collects line draw calls and sorts them for rendering.
#[derive(Debug)]
pub struct LineRenderer {
    /// Pending draw calls.
    draw_calls: Vec<LineDrawCall>,
    /// Camera position for billboard computation.
    pub camera_pos: [f32; 3],
}

impl LineRenderer {
    /// Create a new line renderer.
    pub fn new() -> Self {
        Self {
            draw_calls: Vec::new(),
            camera_pos: [0.0; 3],
        }
    }

    /// Submit a single line segment.
    pub fn draw_line(&mut self, start: [f32; 3], end: [f32; 3], style: &LineStyle) {
        let polyline = Polyline::new(vec![start, end]);
        let (vertices, indices) = generate_line_mesh(&polyline, style, self.camera_pos);
        self.draw_calls.push(LineDrawCall {
            vertices,
            indices,
            depth_mode: style.depth_mode,
            texture: 0,
        });
    }

    /// Submit a polyline.
    pub fn draw_polyline(&mut self, polyline: &Polyline, style: &LineStyle) {
        let (vertices, indices) = generate_line_mesh(polyline, style, self.camera_pos);
        self.draw_calls.push(LineDrawCall {
            vertices,
            indices,
            depth_mode: style.depth_mode,
            texture: 0,
        });
    }

    /// Submit a quadratic Bezier curve.
    pub fn draw_quad_bezier(&mut self, bezier: &QuadBezier, style: &LineStyle, segments: u32) {
        let polyline = bezier.tessellate(segments);
        self.draw_polyline(&polyline, style);
    }

    /// Submit a cubic Bezier curve.
    pub fn draw_cubic_bezier(&mut self, bezier: &CubicBezier, style: &LineStyle, segments: u32) {
        let polyline = bezier.tessellate(segments);
        self.draw_polyline(&polyline, style);
    }

    /// Submit a circle.
    pub fn draw_circle(&mut self, center: [f32; 3], radius: f32, segments: u32, style: &LineStyle) {
        let mut points = Vec::with_capacity(segments as usize);
        for i in 0..segments {
            let angle = 2.0 * PI * i as f32 / segments as f32;
            points.push([
                center[0] + angle.cos() * radius,
                center[1],
                center[2] + angle.sin() * radius,
            ]);
        }
        let polyline = Polyline::closed(points);
        self.draw_polyline(&polyline, style);
    }

    /// Submit an arc.
    pub fn draw_arc(
        &mut self,
        center: [f32; 3],
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        segments: u32,
        style: &LineStyle,
    ) {
        let mut points = Vec::with_capacity(segments as usize + 1);
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            let angle = start_angle + (end_angle - start_angle) * t;
            points.push([
                center[0] + angle.cos() * radius,
                center[1],
                center[2] + angle.sin() * radius,
            ]);
        }
        let polyline = Polyline::new(points);
        self.draw_polyline(&polyline, style);
    }

    /// Get all pending draw calls.
    pub fn draw_calls(&self) -> &[LineDrawCall] {
        &self.draw_calls
    }

    /// Clear all draw calls.
    pub fn clear(&mut self) {
        self.draw_calls.clear();
    }

    /// Number of pending draw calls.
    pub fn draw_call_count(&self) -> usize {
        self.draw_calls.len()
    }

    /// Total vertex count across all draw calls.
    pub fn total_vertices(&self) -> usize {
        self.draw_calls.iter().map(|dc| dc.vertices.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        [0.0; 3]
    } else {
        let inv = 1.0 / len;
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_segment() {
        let seg = LineSegment::new([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        assert!((seg.length() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dash_pattern() {
        let dash = DashPattern::dashed(10.0, 5.0);
        assert!(dash.is_visible(5.0));   // In the dash.
        assert!(!dash.is_visible(12.0)); // In the gap.
        assert!(dash.is_visible(16.0));  // In the next dash.
    }

    #[test]
    fn test_cubic_bezier() {
        let bezier = CubicBezier::new(
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
        );
        let start = bezier.evaluate(0.0);
        let end = bezier.evaluate(1.0);
        assert!((start[0] - 0.0).abs() < 1e-6);
        assert!((end[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_polyline_length() {
        let polyline = Polyline::new(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        assert!((polyline.total_length() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_line_mesh_generation() {
        let polyline = Polyline::new(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        let style = LineStyle::default();
        let (verts, idxs) = generate_line_mesh(&polyline, &style, [0.0, 1.0, 0.0]);
        assert_eq!(verts.len(), 4); // 2 points * 2 vertices each.
        assert_eq!(idxs.len(), 6);  // 2 triangles.
    }
}
