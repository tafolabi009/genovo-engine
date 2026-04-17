// engine/render/src/debug_renderer_v2.rs
//
// Enhanced debug rendering: persistent shapes with configurable duration,
// batched lines/triangles, screen-space text labels, wire mesh visualization,
// navigation mesh visualization, and gizmo drawing.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 { pub x: f32, pub y: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

impl Vec2 { pub fn new(x: f32, y: f32) -> Self { Self { x, y } } }

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self { x: self.y*r.z - self.z*r.y, y: self.z*r.x - self.x*r.z, z: self.x*r.y - self.y*r.x } }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn normalize(self) -> Self { let l = self.length(); if l < 1e-12 { Self::ZERO } else { Self { x:self.x/l, y:self.y/l, z:self.z/l } } }
    pub fn scale(self, s: f32) -> Self { Self { x:self.x*s, y:self.y*s, z:self.z*s } }
    pub fn add(self, r: Self) -> Self { Self { x:self.x+r.x, y:self.y+r.y, z:self.z+r.z } }
    pub fn sub(self, r: Self) -> Self { Self { x:self.x-r.x, y:self.y-r.y, z:self.z-r.z } }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Self = Self { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Self = Self { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const CYAN: Self = Self { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const MAGENTA: Self = Self { r: 1.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const GRAY: Self = Self { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
    pub const ORANGE: Self = Self { r: 1.0, g: 0.5, b: 0.0, a: 1.0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self { r, g, b, a } }

    pub fn with_alpha(self, a: f32) -> Self { Self { a, ..self } }

    pub fn to_array(self) -> [f32; 4] { [self.r, self.g, self.b, self.a] }

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
// Debug line vertex
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct DebugVertex {
    pub position: Vec3,
    pub color: Color,
}

impl DebugVertex {
    pub fn new(position: Vec3, color: Color) -> Self { Self { position, color } }
}

// ---------------------------------------------------------------------------
// Shape primitives
// ---------------------------------------------------------------------------

/// Duration for a debug shape. Either single-frame or persistent for N seconds.
#[derive(Debug, Clone, Copy)]
pub enum Duration {
    /// Draw for one frame only.
    SingleFrame,
    /// Draw for the given number of seconds.
    Seconds(f32),
    /// Draw indefinitely until explicitly removed.
    Infinite,
}

/// Depth test mode for debug rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthMode {
    /// Normal depth testing - hidden behind geometry.
    DepthTested,
    /// Always draw on top (overlay).
    AlwaysVisible,
    /// Draw with depth test but also draw a dimmed version on top when occluded.
    XRay,
}

/// A persistent debug shape with lifetime tracking.
#[derive(Debug, Clone)]
pub struct DebugShape {
    pub primitive: DebugPrimitive,
    pub color: Color,
    pub depth_mode: DepthMode,
    pub remaining_time: f32,
    pub is_infinite: bool,
    pub group: u32,
    pub label: Option<String>,
}

/// Debug primitive types.
#[derive(Debug, Clone)]
pub enum DebugPrimitive {
    Line { start: Vec3, end: Vec3, thickness: f32 },
    Arrow { start: Vec3, end: Vec3, head_size: f32 },
    Box { center: Vec3, half_extents: Vec3, rotation: [f32; 9] },
    WireBox { center: Vec3, half_extents: Vec3, rotation: [f32; 9] },
    Sphere { center: Vec3, radius: f32 },
    WireSphere { center: Vec3, radius: f32, segments: u32 },
    Capsule { start: Vec3, end: Vec3, radius: f32 },
    Cylinder { start: Vec3, end: Vec3, radius: f32, segments: u32 },
    Cone { apex: Vec3, base_center: Vec3, radius: f32, segments: u32 },
    Circle { center: Vec3, normal: Vec3, radius: f32, segments: u32 },
    Arc { center: Vec3, normal: Vec3, start_dir: Vec3, radius: f32, angle: f32, segments: u32 },
    Triangle { v0: Vec3, v1: Vec3, v2: Vec3, filled: bool },
    Quad { corners: [Vec3; 4], filled: bool },
    Frustum { corners_near: [Vec3; 4], corners_far: [Vec3; 4] },
    Point { position: Vec3, size: f32 },
    Axis { center: Vec3, size: f32 },
    Grid { center: Vec3, normal: Vec3, size: f32, divisions: u32 },
    Path { points: Vec<Vec3>, closed: bool },
    Text3D { position: Vec3, text: String, size: f32 },
}

// ---------------------------------------------------------------------------
// Screen-space text label
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ScreenLabel {
    pub world_position: Vec3,
    pub text: String,
    pub color: Color,
    pub background_color: Option<Color>,
    pub font_size: f32,
    pub offset: Vec2, // screen-space offset in pixels
    pub remaining_time: f32,
    pub is_infinite: bool,
}

// ---------------------------------------------------------------------------
// Debug renderer
// ---------------------------------------------------------------------------

/// Enhanced debug rendering system with batching and persistence.
pub struct DebugRendererV2 {
    // Immediate mode batches (rebuilt each frame)
    pub line_vertices: Vec<DebugVertex>,
    pub solid_vertices: Vec<DebugVertex>,
    pub solid_indices: Vec<u32>,

    // Persistent shapes
    shapes: Vec<DebugShape>,
    labels: Vec<ScreenLabel>,

    // Configuration
    pub enabled: bool,
    pub default_depth_mode: DepthMode,
    pub default_color: Color,
    pub max_shapes: usize,
    pub max_labels: usize,
    pub max_line_vertices: usize,
    pub groups_visible: Vec<bool>, // per-group visibility

    // Statistics
    pub stats: DebugRenderStats,
}

#[derive(Debug, Clone, Default)]
pub struct DebugRenderStats {
    pub line_count: u32,
    pub solid_triangle_count: u32,
    pub persistent_shape_count: u32,
    pub label_count: u32,
    pub total_vertices: u32,
}

impl DebugRendererV2 {
    pub fn new() -> Self {
        Self {
            line_vertices: Vec::with_capacity(65536),
            solid_vertices: Vec::with_capacity(65536),
            solid_indices: Vec::with_capacity(65536),
            shapes: Vec::new(),
            labels: Vec::new(),
            enabled: true,
            default_depth_mode: DepthMode::DepthTested,
            default_color: Color::WHITE,
            max_shapes: 10000,
            max_labels: 1000,
            max_line_vertices: 500000,
            groups_visible: vec![true; 32],
            stats: DebugRenderStats::default(),
        }
    }

    /// Begin a new frame. Clears immediate-mode buffers and updates lifetimes.
    pub fn begin_frame(&mut self, dt: f32) {
        self.line_vertices.clear();
        self.solid_vertices.clear();
        self.solid_indices.clear();

        // Update persistent shapes
        self.shapes.retain_mut(|shape| {
            if shape.is_infinite { return true; }
            shape.remaining_time -= dt;
            shape.remaining_time > 0.0
        });

        self.labels.retain_mut(|label| {
            if label.is_infinite { return true; }
            label.remaining_time -= dt;
            label.remaining_time > 0.0
        });
    }

    /// Flush persistent shapes into the immediate-mode buffers.
    pub fn flush_persistent(&mut self) {
        let shapes: Vec<DebugShape> = self.shapes.iter()
            .filter(|s| {
                let group = s.group as usize;
                group < self.groups_visible.len() && self.groups_visible[group]
            })
            .cloned()
            .collect();

        for shape in &shapes {
            self.render_primitive(&shape.primitive, shape.color);
        }

        // Update stats
        self.stats = DebugRenderStats {
            line_count: (self.line_vertices.len() / 2) as u32,
            solid_triangle_count: (self.solid_indices.len() / 3) as u32,
            persistent_shape_count: self.shapes.len() as u32,
            label_count: self.labels.len() as u32,
            total_vertices: (self.line_vertices.len() + self.solid_vertices.len()) as u32,
        };
    }

    // -----------------------------------------------------------------------
    // Immediate mode drawing
    // -----------------------------------------------------------------------

    /// Draw a line.
    pub fn draw_line(&mut self, start: Vec3, end: Vec3, color: Color) {
        if !self.enabled { return; }
        self.line_vertices.push(DebugVertex::new(start, color));
        self.line_vertices.push(DebugVertex::new(end, color));
    }

    /// Draw an arrow.
    pub fn draw_arrow(&mut self, start: Vec3, end: Vec3, color: Color, head_size: f32) {
        if !self.enabled { return; }
        self.draw_line(start, end, color);

        let dir = end.sub(start).normalize();
        let len = start.sub(end).length();
        let head_len = (head_size * len).min(len * 0.3);

        // Create arrowhead
        let up = if dir.dot(Vec3::new(0.0, 1.0, 0.0)).abs() > 0.99 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let right = dir.cross(up).normalize();
        let up2 = right.cross(dir).normalize();

        let base = end.sub(dir.scale(head_len));
        let offset = head_len * 0.3;

        self.draw_line(end, base.add(right.scale(offset)), color);
        self.draw_line(end, base.sub(right.scale(offset)), color);
        self.draw_line(end, base.add(up2.scale(offset)), color);
        self.draw_line(end, base.sub(up2.scale(offset)), color);
    }

    /// Draw a wire box.
    pub fn draw_wire_box(&mut self, center: Vec3, half_extents: Vec3, color: Color) {
        if !self.enabled { return; }
        let h = half_extents;
        let corners = [
            center.add(Vec3::new(-h.x, -h.y, -h.z)),
            center.add(Vec3::new( h.x, -h.y, -h.z)),
            center.add(Vec3::new( h.x,  h.y, -h.z)),
            center.add(Vec3::new(-h.x,  h.y, -h.z)),
            center.add(Vec3::new(-h.x, -h.y,  h.z)),
            center.add(Vec3::new( h.x, -h.y,  h.z)),
            center.add(Vec3::new( h.x,  h.y,  h.z)),
            center.add(Vec3::new(-h.x,  h.y,  h.z)),
        ];

        // Bottom face
        self.draw_line(corners[0], corners[1], color);
        self.draw_line(corners[1], corners[2], color);
        self.draw_line(corners[2], corners[3], color);
        self.draw_line(corners[3], corners[0], color);
        // Top face
        self.draw_line(corners[4], corners[5], color);
        self.draw_line(corners[5], corners[6], color);
        self.draw_line(corners[6], corners[7], color);
        self.draw_line(corners[7], corners[4], color);
        // Vertical edges
        self.draw_line(corners[0], corners[4], color);
        self.draw_line(corners[1], corners[5], color);
        self.draw_line(corners[2], corners[6], color);
        self.draw_line(corners[3], corners[7], color);
    }

    /// Draw a wire sphere.
    pub fn draw_wire_sphere(&mut self, center: Vec3, radius: f32, color: Color) {
        self.draw_wire_sphere_segments(center, radius, color, 24);
    }

    pub fn draw_wire_sphere_segments(&mut self, center: Vec3, radius: f32, color: Color, segments: u32) {
        if !self.enabled { return; }
        // Three orthogonal circles
        self.draw_circle(center, Vec3::new(0.0, 1.0, 0.0), radius, color, segments);
        self.draw_circle(center, Vec3::new(1.0, 0.0, 0.0), radius, color, segments);
        self.draw_circle(center, Vec3::new(0.0, 0.0, 1.0), radius, color, segments);
    }

    /// Draw a circle.
    pub fn draw_circle(&mut self, center: Vec3, normal: Vec3, radius: f32, color: Color, segments: u32) {
        if !self.enabled { return; }
        let up = if normal.dot(Vec3::new(0.0, 1.0, 0.0)).abs() > 0.99 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let right = normal.cross(up).normalize();
        let forward = right.cross(normal).normalize();

        let step = std::f32::consts::TAU / segments as f32;
        for i in 0..segments {
            let a1 = i as f32 * step;
            let a2 = (i + 1) as f32 * step;
            let p1 = center.add(right.scale(a1.cos() * radius)).add(forward.scale(a1.sin() * radius));
            let p2 = center.add(right.scale(a2.cos() * radius)).add(forward.scale(a2.sin() * radius));
            self.draw_line(p1, p2, color);
        }
    }

    /// Draw a capsule.
    pub fn draw_capsule(&mut self, start: Vec3, end: Vec3, radius: f32, color: Color) {
        if !self.enabled { return; }
        let dir = end.sub(start).normalize();
        let segments = 16;

        // Side lines
        let up = if dir.dot(Vec3::new(0.0, 1.0, 0.0)).abs() > 0.99 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let right = dir.cross(up).normalize();
        let forward = right.cross(dir).normalize();

        // Draw 4 side lines
        for i in 0..4 {
            let angle = i as f32 * std::f32::consts::TAU / 4.0;
            let offset = right.scale(angle.cos() * radius).add(forward.scale(angle.sin() * radius));
            self.draw_line(start.add(offset), end.add(offset), color);
        }

        // End caps (circles)
        self.draw_circle(start, dir, radius, color, segments);
        self.draw_circle(end, dir, radius, color, segments);
    }

    /// Draw an axis gizmo (RGB = XYZ).
    pub fn draw_axis(&mut self, center: Vec3, size: f32) {
        if !self.enabled { return; }
        self.draw_arrow(center, center.add(Vec3::new(size, 0.0, 0.0)), Color::RED, 0.15);
        self.draw_arrow(center, center.add(Vec3::new(0.0, size, 0.0)), Color::GREEN, 0.15);
        self.draw_arrow(center, center.add(Vec3::new(0.0, 0.0, size)), Color::BLUE, 0.15);
    }

    /// Draw a grid on the XZ plane.
    pub fn draw_grid(&mut self, center: Vec3, size: f32, divisions: u32, color: Color) {
        if !self.enabled { return; }
        let half = size * 0.5;
        let step = size / divisions as f32;

        for i in 0..=divisions {
            let t = -half + i as f32 * step;
            // Lines along X
            self.draw_line(
                center.add(Vec3::new(-half, 0.0, t)),
                center.add(Vec3::new(half, 0.0, t)),
                color,
            );
            // Lines along Z
            self.draw_line(
                center.add(Vec3::new(t, 0.0, -half)),
                center.add(Vec3::new(t, 0.0, half)),
                color,
            );
        }
    }

    /// Draw a frustum from 8 corner points.
    pub fn draw_frustum(&mut self, near_corners: [Vec3; 4], far_corners: [Vec3; 4], color: Color) {
        if !self.enabled { return; }
        // Near plane
        for i in 0..4 { self.draw_line(near_corners[i], near_corners[(i+1)%4], color); }
        // Far plane
        for i in 0..4 { self.draw_line(far_corners[i], far_corners[(i+1)%4], color); }
        // Connecting edges
        for i in 0..4 { self.draw_line(near_corners[i], far_corners[i], color); }
    }

    /// Draw a filled triangle.
    pub fn draw_solid_triangle(&mut self, v0: Vec3, v1: Vec3, v2: Vec3, color: Color) {
        if !self.enabled { return; }
        let base = self.solid_vertices.len() as u32;
        let normal = v1.sub(v0).cross(v2.sub(v0)).normalize();
        // Shade both sides
        let _ = normal;
        self.solid_vertices.push(DebugVertex::new(v0, color));
        self.solid_vertices.push(DebugVertex::new(v1, color));
        self.solid_vertices.push(DebugVertex::new(v2, color));
        self.solid_indices.push(base);
        self.solid_indices.push(base + 1);
        self.solid_indices.push(base + 2);
    }

    /// Draw a wire mesh (all edges of a triangle mesh).
    pub fn draw_wire_mesh(&mut self, vertices: &[Vec3], indices: &[u32], color: Color) {
        if !self.enabled { return; }
        let tri_count = indices.len() / 3;
        let mut drawn_edges: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();

        for t in 0..tri_count {
            let idx = [indices[t*3], indices[t*3+1], indices[t*3+2]];
            for e in 0..3 {
                let v0 = idx[e];
                let v1 = idx[(e+1) % 3];
                let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                if drawn_edges.insert(key) {
                    if (v0 as usize) < vertices.len() && (v1 as usize) < vertices.len() {
                        self.draw_line(vertices[v0 as usize], vertices[v1 as usize], color);
                    }
                }
            }
        }
    }

    /// Draw a path (connected line segments).
    pub fn draw_path(&mut self, points: &[Vec3], color: Color, closed: bool) {
        if !self.enabled || points.len() < 2 { return; }
        for i in 0..(points.len() - 1) {
            self.draw_line(points[i], points[i + 1], color);
        }
        if closed && points.len() >= 3 {
            self.draw_line(points[points.len() - 1], points[0], color);
        }
    }

    /// Draw a cone wireframe.
    pub fn draw_cone(&mut self, apex: Vec3, base_center: Vec3, radius: f32, color: Color, segments: u32) {
        if !self.enabled { return; }
        let dir = base_center.sub(apex).normalize();
        self.draw_circle(base_center, dir, radius, color, segments);

        let up = if dir.dot(Vec3::new(0.0, 1.0, 0.0)).abs() > 0.99 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let right = dir.cross(up).normalize();
        let forward = right.cross(dir).normalize();

        let step = std::f32::consts::TAU / 4.0;
        for i in 0..4 {
            let angle = i as f32 * step;
            let base_point = base_center
                .add(right.scale(angle.cos() * radius))
                .add(forward.scale(angle.sin() * radius));
            self.draw_line(apex, base_point, color);
        }
    }

    // -----------------------------------------------------------------------
    // Persistent shapes
    // -----------------------------------------------------------------------

    /// Add a persistent shape that lasts for a given duration.
    pub fn add_shape(&mut self, primitive: DebugPrimitive, color: Color, duration: Duration) {
        self.add_shape_full(primitive, color, duration, self.default_depth_mode, 0, None);
    }

    /// Add a persistent shape with full options.
    pub fn add_shape_full(
        &mut self,
        primitive: DebugPrimitive,
        color: Color,
        duration: Duration,
        depth_mode: DepthMode,
        group: u32,
        label: Option<String>,
    ) {
        if self.shapes.len() >= self.max_shapes {
            // Remove oldest non-infinite shape
            if let Some(idx) = self.shapes.iter().position(|s| !s.is_infinite) {
                self.shapes.remove(idx);
            } else {
                return;
            }
        }

        let (remaining, is_infinite) = match duration {
            Duration::SingleFrame => (0.016, false),
            Duration::Seconds(s) => (s, false),
            Duration::Infinite => (0.0, true),
        };

        self.shapes.push(DebugShape {
            primitive,
            color,
            depth_mode,
            remaining_time: remaining,
            is_infinite,
            group,
            label,
        });
    }

    /// Add a screen-space text label attached to a world position.
    pub fn add_label(&mut self, world_pos: Vec3, text: &str, color: Color, duration: Duration) {
        if self.labels.len() >= self.max_labels { return; }

        let (remaining, is_infinite) = match duration {
            Duration::SingleFrame => (0.016, false),
            Duration::Seconds(s) => (s, false),
            Duration::Infinite => (0.0, true),
        };

        self.labels.push(ScreenLabel {
            world_position: world_pos,
            text: text.to_string(),
            color,
            background_color: Some(Color::new(0.0, 0.0, 0.0, 0.7)),
            font_size: 14.0,
            offset: Vec2::new(0.0, -20.0),
            remaining_time: remaining,
            is_infinite,
        });
    }

    /// Remove all shapes in a group.
    pub fn clear_group(&mut self, group: u32) {
        self.shapes.retain(|s| s.group != group);
    }

    /// Remove all shapes and labels.
    pub fn clear_all(&mut self) {
        self.shapes.clear();
        self.labels.clear();
    }

    /// Set group visibility.
    pub fn set_group_visible(&mut self, group: u32, visible: bool) {
        if (group as usize) < self.groups_visible.len() {
            self.groups_visible[group as usize] = visible;
        }
    }

    /// Get all labels for rendering.
    pub fn get_labels(&self) -> &[ScreenLabel] {
        &self.labels
    }

    // -----------------------------------------------------------------------
    // Render primitives from DebugPrimitive
    // -----------------------------------------------------------------------

    fn render_primitive(&mut self, prim: &DebugPrimitive, color: Color) {
        match prim {
            DebugPrimitive::Line { start, end, .. } => {
                self.draw_line(*start, *end, color);
            }
            DebugPrimitive::Arrow { start, end, head_size } => {
                self.draw_arrow(*start, *end, color, *head_size);
            }
            DebugPrimitive::WireBox { center, half_extents, .. } => {
                self.draw_wire_box(*center, *half_extents, color);
            }
            DebugPrimitive::Sphere { center, radius } | DebugPrimitive::WireSphere { center, radius, .. } => {
                self.draw_wire_sphere(*center, *radius, color);
            }
            DebugPrimitive::Capsule { start, end, radius } => {
                self.draw_capsule(*start, *end, *radius, color);
            }
            DebugPrimitive::Circle { center, normal, radius, segments } => {
                self.draw_circle(*center, *normal, *radius, color, *segments);
            }
            DebugPrimitive::Axis { center, size } => {
                self.draw_axis(*center, *size);
            }
            DebugPrimitive::Grid { center, normal: _, size, divisions } => {
                self.draw_grid(*center, *size, *divisions, color);
            }
            DebugPrimitive::Path { points, closed } => {
                self.draw_path(points, color, *closed);
            }
            DebugPrimitive::Frustum { corners_near, corners_far } => {
                self.draw_frustum(*corners_near, *corners_far, color);
            }
            DebugPrimitive::Triangle { v0, v1, v2, filled } => {
                if *filled {
                    self.draw_solid_triangle(*v0, *v1, *v2, color);
                } else {
                    self.draw_line(*v0, *v1, color);
                    self.draw_line(*v1, *v2, color);
                    self.draw_line(*v2, *v0, color);
                }
            }
            DebugPrimitive::Point { position, size } => {
                let s = *size;
                self.draw_line(position.add(Vec3::new(-s, 0.0, 0.0)), position.add(Vec3::new(s, 0.0, 0.0)), color);
                self.draw_line(position.add(Vec3::new(0.0, -s, 0.0)), position.add(Vec3::new(0.0, s, 0.0)), color);
                self.draw_line(position.add(Vec3::new(0.0, 0.0, -s)), position.add(Vec3::new(0.0, 0.0, s)), color);
            }
            DebugPrimitive::Cone { apex, base_center, radius, segments } => {
                self.draw_cone(*apex, *base_center, *radius, color, *segments);
            }
            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // Navigation mesh visualization
    // -----------------------------------------------------------------------

    /// Draw a navigation mesh with polygon edges and optionally filled polygons.
    pub fn draw_navmesh(
        &mut self,
        vertices: &[Vec3],
        polygons: &[Vec<u32>],
        fill: bool,
        edge_color: Color,
        fill_color: Color,
    ) {
        for poly in polygons {
            // Draw edges
            for i in 0..poly.len() {
                let v0 = poly[i] as usize;
                let v1 = poly[(i + 1) % poly.len()] as usize;
                if v0 < vertices.len() && v1 < vertices.len() {
                    self.draw_line(vertices[v0], vertices[v1], edge_color);
                }
            }

            // Optionally fill (fan triangulation)
            if fill && poly.len() >= 3 {
                for i in 1..(poly.len() - 1) {
                    let v0 = poly[0] as usize;
                    let v1 = poly[i] as usize;
                    let v2 = poly[i + 1] as usize;
                    if v0 < vertices.len() && v1 < vertices.len() && v2 < vertices.len() {
                        self.draw_solid_triangle(vertices[v0], vertices[v1], vertices[v2], fill_color);
                    }
                }
            }
        }
    }

    /// Draw a navigation path.
    pub fn draw_nav_path(&mut self, path: &[Vec3], color: Color, node_size: f32) {
        if path.len() < 2 { return; }

        for i in 0..(path.len() - 1) {
            self.draw_arrow(path[i], path[i + 1], color, 0.2);
        }

        // Draw nodes
        for pt in path {
            self.draw_wire_sphere(*pt, node_size, color);
        }
    }

    /// Draw velocity vector.
    pub fn draw_velocity(&mut self, position: Vec3, velocity: Vec3, scale: f32, color: Color) {
        let end = position.add(velocity.scale(scale));
        self.draw_arrow(position, end, color, 0.15);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_renderer_creation() {
        let renderer = DebugRendererV2::new();
        assert!(renderer.enabled);
        assert_eq!(renderer.line_vertices.len(), 0);
    }

    #[test]
    fn test_draw_line() {
        let mut renderer = DebugRendererV2::new();
        renderer.draw_line(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Color::RED);
        assert_eq!(renderer.line_vertices.len(), 2);
    }

    #[test]
    fn test_persistent_shape() {
        let mut renderer = DebugRendererV2::new();
        renderer.add_shape(
            DebugPrimitive::Sphere { center: Vec3::ZERO, radius: 1.0 },
            Color::GREEN,
            Duration::Seconds(5.0),
        );
        assert_eq!(renderer.shapes.len(), 1);

        // Advance time
        renderer.begin_frame(6.0);
        assert_eq!(renderer.shapes.len(), 0); // should have expired
    }

    #[test]
    fn test_infinite_shape() {
        let mut renderer = DebugRendererV2::new();
        renderer.add_shape(
            DebugPrimitive::Line { start: Vec3::ZERO, end: Vec3::new(1.0, 0.0, 0.0), thickness: 1.0 },
            Color::WHITE,
            Duration::Infinite,
        );
        renderer.begin_frame(100.0);
        assert_eq!(renderer.shapes.len(), 1); // should persist
    }

    #[test]
    fn test_clear_group() {
        let mut renderer = DebugRendererV2::new();
        renderer.add_shape_full(
            DebugPrimitive::Point { position: Vec3::ZERO, size: 0.1 },
            Color::RED, Duration::Infinite, DepthMode::DepthTested, 5, None,
        );
        renderer.add_shape_full(
            DebugPrimitive::Point { position: Vec3::ZERO, size: 0.1 },
            Color::GREEN, Duration::Infinite, DepthMode::DepthTested, 7, None,
        );
        renderer.clear_group(5);
        assert_eq!(renderer.shapes.len(), 1);
    }

    #[test]
    fn test_label() {
        let mut renderer = DebugRendererV2::new();
        renderer.add_label(Vec3::ZERO, "Test", Color::WHITE, Duration::Seconds(1.0));
        assert_eq!(renderer.labels.len(), 1);
    }
}
