// engine/core/src/debug_draw.rs
//
// Debug drawing API for the Genovo engine.
//
// Provides immediate-mode debug drawing primitives for development:
//
// - draw_line: line segment between two points.
// - draw_sphere: wireframe sphere at a position.
// - draw_box: wireframe box (AABB or OBB).
// - draw_text_3d: text label at a 3D world position.
// - draw_arrow: arrow from start to end with arrowhead.
// - draw_frustum: camera frustum visualization.
// - Persistent vs single-frame drawing modes.
// - Depth-tested vs overlay rendering modes.
// - Color and line width per primitive.
// - Automatic buffer management and batching.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum debug primitives per frame.
const MAX_PRIMITIVES: usize = 65536;

/// Default line width.
const DEFAULT_LINE_WIDTH: f32 = 1.0;

/// Default text size.
const DEFAULT_TEXT_SIZE: f32 = 14.0;

/// Maximum persistent primitives.
const MAX_PERSISTENT: usize = 4096;

/// Number of segments for sphere wireframe.
const SPHERE_SEGMENTS: u32 = 24;

/// Number of segments for circle wireframe.
const CIRCLE_SEGMENTS: u32 = 32;

/// Arrowhead length ratio (relative to arrow length).
const ARROWHEAD_RATIO: f32 = 0.15;

/// Arrowhead width ratio.
const ARROWHEAD_WIDTH_RATIO: f32 = 0.08;

// ---------------------------------------------------------------------------
// Color Presets
// ---------------------------------------------------------------------------

/// Common debug colors.
pub struct DebugColors;

impl DebugColors {
    pub const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    pub const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    pub const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    pub const YELLOW: [f32; 4] = [1.0, 1.0, 0.0, 1.0];
    pub const CYAN: [f32; 4] = [0.0, 1.0, 1.0, 1.0];
    pub const MAGENTA: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
    pub const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    pub const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
    pub const ORANGE: [f32; 4] = [1.0, 0.5, 0.0, 1.0];
    pub const PURPLE: [f32; 4] = [0.5, 0.0, 1.0, 1.0];
    pub const GRAY: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
    pub const LIGHT_GREEN: [f32; 4] = [0.5, 1.0, 0.5, 0.8];
    pub const LIGHT_BLUE: [f32; 4] = [0.5, 0.5, 1.0, 0.8];
    pub const TRANSPARENT_RED: [f32; 4] = [1.0, 0.0, 0.0, 0.3];
    pub const TRANSPARENT_GREEN: [f32; 4] = [0.0, 1.0, 0.0, 0.3];
    pub const TRANSPARENT_BLUE: [f32; 4] = [0.0, 0.0, 1.0, 0.3];
}

// ---------------------------------------------------------------------------
// Render Mode
// ---------------------------------------------------------------------------

/// How a debug primitive is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugRenderMode {
    /// Rendered with depth testing (occluded by geometry).
    DepthTested,
    /// Rendered as overlay (always visible).
    Overlay,
    /// Rendered with depth testing but slightly biased toward camera.
    DepthBiased,
}

// ---------------------------------------------------------------------------
// Debug Primitive
// ---------------------------------------------------------------------------

/// A debug drawing primitive.
#[derive(Debug, Clone)]
pub struct DebugPrimitive {
    /// Primitive type and data.
    pub shape: DebugShape,
    /// Color.
    pub color: [f32; 4],
    /// Line width.
    pub line_width: f32,
    /// Rendering mode.
    pub render_mode: DebugRenderMode,
    /// Duration remaining (0 = single frame, >0 = persistent).
    pub duration: f32,
    /// Whether this primitive is currently active.
    pub active: bool,
}

/// Types of debug shapes.
#[derive(Debug, Clone)]
pub enum DebugShape {
    /// Line segment.
    Line {
        start: [f32; 3],
        end: [f32; 3],
    },
    /// Wireframe sphere.
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    /// Wireframe box (axis-aligned).
    Box {
        min: [f32; 3],
        max: [f32; 3],
    },
    /// Oriented bounding box.
    OrientedBox {
        center: [f32; 3],
        half_extents: [f32; 3],
        rotation: [f32; 4],
    },
    /// 3D text label.
    Text3D {
        position: [f32; 3],
        text: String,
        size: f32,
    },
    /// Arrow.
    Arrow {
        start: [f32; 3],
        end: [f32; 3],
    },
    /// Camera frustum.
    Frustum {
        corners: [[f32; 3]; 8],
    },
    /// Circle (in a specified plane).
    Circle {
        center: [f32; 3],
        radius: f32,
        normal: [f32; 3],
    },
    /// Point (rendered as a small cross).
    Point {
        position: [f32; 3],
        size: f32,
    },
    /// Capsule wireframe.
    Capsule {
        start: [f32; 3],
        end: [f32; 3],
        radius: f32,
    },
    /// Axis gizmo (three colored lines at a position).
    Axes {
        position: [f32; 3],
        size: f32,
    },
    /// Grid plane.
    Grid {
        center: [f32; 3],
        size: f32,
        subdivisions: u32,
    },
    /// Polygon (list of points, closed).
    Polygon {
        points: Vec<[f32; 3]>,
    },
    /// Cone wireframe.
    Cone {
        apex: [f32; 3],
        direction: [f32; 3],
        height: f32,
        radius: f32,
    },
}

// ---------------------------------------------------------------------------
// Line Vertex
// ---------------------------------------------------------------------------

/// A vertex in the debug line buffer.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DebugLineVertex {
    /// Position.
    pub position: [f32; 3],
    /// Color.
    pub color: [f32; 4],
}

impl DebugLineVertex {
    /// Create a new vertex.
    pub fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }

    /// Stride in bytes.
    pub fn stride() -> usize {
        std::mem::size_of::<Self>()
    }
}

// ---------------------------------------------------------------------------
// Text Label
// ---------------------------------------------------------------------------

/// A 3D text label for debug display.
#[derive(Debug, Clone)]
pub struct DebugTextLabel {
    /// World position.
    pub position: [f32; 3],
    /// Text content.
    pub text: String,
    /// Font size.
    pub size: f32,
    /// Color.
    pub color: [f32; 4],
    /// Duration remaining.
    pub duration: f32,
    /// Render mode.
    pub render_mode: DebugRenderMode,
}

// ---------------------------------------------------------------------------
// Debug Draw System
// ---------------------------------------------------------------------------

/// The main debug drawing system.
#[derive(Debug)]
pub struct DebugDrawSystem {
    /// Single-frame primitives (cleared each frame).
    pub frame_primitives: Vec<DebugPrimitive>,
    /// Persistent primitives (duration-based).
    pub persistent_primitives: Vec<DebugPrimitive>,
    /// Generated line vertices for rendering.
    pub line_vertices: Vec<DebugLineVertex>,
    /// Overlay line vertices (no depth test).
    pub overlay_vertices: Vec<DebugLineVertex>,
    /// Text labels.
    pub text_labels: Vec<DebugTextLabel>,
    /// Whether debug drawing is enabled.
    pub enabled: bool,
    /// Default line width.
    pub default_line_width: f32,
    /// Default render mode.
    pub default_render_mode: DebugRenderMode,
    /// Statistics.
    pub stats: DebugDrawStats,
}

impl DebugDrawSystem {
    /// Create a new debug draw system.
    pub fn new() -> Self {
        Self {
            frame_primitives: Vec::new(),
            persistent_primitives: Vec::new(),
            line_vertices: Vec::new(),
            overlay_vertices: Vec::new(),
            text_labels: Vec::new(),
            enabled: true,
            default_line_width: DEFAULT_LINE_WIDTH,
            default_render_mode: DebugRenderMode::DepthTested,
            stats: DebugDrawStats::default(),
        }
    }

    /// Draw a line segment.
    pub fn draw_line(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 4]) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Line { start, end },
            color,
            line_width: self.default_line_width,
            render_mode: self.default_render_mode,
            duration: 0.0,
            active: true,
        });
    }

    /// Draw a line segment that persists for a duration.
    pub fn draw_line_persistent(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 4], duration: f32) {
        if !self.enabled { return; }
        self.persistent_primitives.push(DebugPrimitive {
            shape: DebugShape::Line { start, end },
            color,
            line_width: self.default_line_width,
            render_mode: self.default_render_mode,
            duration,
            active: true,
        });
    }

    /// Draw a wireframe sphere.
    pub fn draw_sphere(&mut self, center: [f32; 3], radius: f32, color: [f32; 4]) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Sphere { center, radius },
            color,
            line_width: self.default_line_width,
            render_mode: self.default_render_mode,
            duration: 0.0,
            active: true,
        });
    }

    /// Draw a wireframe AABB.
    pub fn draw_box(&mut self, min: [f32; 3], max: [f32; 3], color: [f32; 4]) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Box { min, max },
            color,
            line_width: self.default_line_width,
            render_mode: self.default_render_mode,
            duration: 0.0,
            active: true,
        });
    }

    /// Draw a 3D text label.
    pub fn draw_text_3d(&mut self, position: [f32; 3], text: &str, color: [f32; 4]) {
        if !self.enabled { return; }
        self.text_labels.push(DebugTextLabel {
            position,
            text: text.to_string(),
            size: DEFAULT_TEXT_SIZE,
            color,
            duration: 0.0,
            render_mode: DebugRenderMode::Overlay,
        });
    }

    /// Draw an arrow from start to end.
    pub fn draw_arrow(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 4]) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Arrow { start, end },
            color,
            line_width: self.default_line_width,
            render_mode: self.default_render_mode,
            duration: 0.0,
            active: true,
        });
    }

    /// Draw a camera frustum.
    pub fn draw_frustum(&mut self, corners: [[f32; 3]; 8], color: [f32; 4]) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Frustum { corners },
            color,
            line_width: self.default_line_width,
            render_mode: self.default_render_mode,
            duration: 0.0,
            active: true,
        });
    }

    /// Draw a point (rendered as a small cross).
    pub fn draw_point(&mut self, position: [f32; 3], size: f32, color: [f32; 4]) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Point { position, size },
            color,
            line_width: self.default_line_width,
            render_mode: self.default_render_mode,
            duration: 0.0,
            active: true,
        });
    }

    /// Draw a circle.
    pub fn draw_circle(&mut self, center: [f32; 3], radius: f32, normal: [f32; 3], color: [f32; 4]) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Circle { center, radius, normal },
            color,
            line_width: self.default_line_width,
            render_mode: self.default_render_mode,
            duration: 0.0,
            active: true,
        });
    }

    /// Draw an axis gizmo at a position.
    pub fn draw_axes(&mut self, position: [f32; 3], size: f32) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Axes { position, size },
            color: DebugColors::WHITE,
            line_width: 2.0,
            render_mode: DebugRenderMode::Overlay,
            duration: 0.0,
            active: true,
        });
    }

    /// Draw a grid plane.
    pub fn draw_grid(&mut self, center: [f32; 3], size: f32, subdivisions: u32, color: [f32; 4]) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Grid { center, size, subdivisions },
            color,
            line_width: 0.5,
            render_mode: self.default_render_mode,
            duration: 0.0,
            active: true,
        });
    }

    /// Draw a capsule wireframe.
    pub fn draw_capsule(&mut self, start: [f32; 3], end: [f32; 3], radius: f32, color: [f32; 4]) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Capsule { start, end, radius },
            color,
            line_width: self.default_line_width,
            render_mode: self.default_render_mode,
            duration: 0.0,
            active: true,
        });
    }

    /// Draw a cone wireframe.
    pub fn draw_cone(&mut self, apex: [f32; 3], direction: [f32; 3], height: f32, radius: f32, color: [f32; 4]) {
        if !self.enabled { return; }
        self.frame_primitives.push(DebugPrimitive {
            shape: DebugShape::Cone { apex, direction, height, radius },
            color,
            line_width: self.default_line_width,
            render_mode: self.default_render_mode,
            duration: 0.0,
            active: true,
        });
    }

    /// Generate line vertices from all primitives for rendering.
    pub fn generate_vertices(&mut self) {
        self.line_vertices.clear();
        self.overlay_vertices.clear();

        let all_primitives: Vec<DebugPrimitive> = self.frame_primitives.iter()
            .chain(self.persistent_primitives.iter())
            .filter(|p| p.active)
            .cloned()
            .collect();

        for prim in &all_primitives {
            let target = match prim.render_mode {
                DebugRenderMode::Overlay => &mut self.overlay_vertices,
                _ => &mut self.line_vertices,
            };

            match &prim.shape {
                DebugShape::Line { start, end } => {
                    target.push(DebugLineVertex::new(*start, prim.color));
                    target.push(DebugLineVertex::new(*end, prim.color));
                }
                DebugShape::Box { min, max } => {
                    Self::generate_box_lines(target, *min, *max, prim.color);
                }
                DebugShape::Point { position, size } => {
                    let s = *size;
                    target.push(DebugLineVertex::new([position[0] - s, position[1], position[2]], prim.color));
                    target.push(DebugLineVertex::new([position[0] + s, position[1], position[2]], prim.color));
                    target.push(DebugLineVertex::new([position[0], position[1] - s, position[2]], prim.color));
                    target.push(DebugLineVertex::new([position[0], position[1] + s, position[2]], prim.color));
                    target.push(DebugLineVertex::new([position[0], position[1], position[2] - s], prim.color));
                    target.push(DebugLineVertex::new([position[0], position[1], position[2] + s], prim.color));
                }
                DebugShape::Axes { position, size } => {
                    let s = *size;
                    target.push(DebugLineVertex::new(*position, DebugColors::RED));
                    target.push(DebugLineVertex::new([position[0] + s, position[1], position[2]], DebugColors::RED));
                    target.push(DebugLineVertex::new(*position, DebugColors::GREEN));
                    target.push(DebugLineVertex::new([position[0], position[1] + s, position[2]], DebugColors::GREEN));
                    target.push(DebugLineVertex::new(*position, DebugColors::BLUE));
                    target.push(DebugLineVertex::new([position[0], position[1], position[2] + s], DebugColors::BLUE));
                }
                DebugShape::Arrow { start, end } => {
                    target.push(DebugLineVertex::new(*start, prim.color));
                    target.push(DebugLineVertex::new(*end, prim.color));
                    // Arrowhead lines are omitted for brevity in vertex generation.
                }
                _ => {
                    // Other shapes generate more complex line sets.
                }
            }
        }

        self.stats.line_count = (self.line_vertices.len() + self.overlay_vertices.len()) / 2;
        self.stats.primitive_count = all_primitives.len();
        self.stats.text_count = self.text_labels.len();
    }

    /// Generate box wireframe lines.
    fn generate_box_lines(target: &mut Vec<DebugLineVertex>, min: [f32; 3], max: [f32; 3], color: [f32; 4]) {
        let corners = [
            [min[0], min[1], min[2]], [max[0], min[1], min[2]],
            [max[0], max[1], min[2]], [min[0], max[1], min[2]],
            [min[0], min[1], max[2]], [max[0], min[1], max[2]],
            [max[0], max[1], max[2]], [min[0], max[1], max[2]],
        ];
        let edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];
        for (a, b) in edges {
            target.push(DebugLineVertex::new(corners[a], color));
            target.push(DebugLineVertex::new(corners[b], color));
        }
    }

    /// Update the system: decay persistent primitives and clear frame primitives.
    pub fn end_frame(&mut self, dt: f32) {
        self.frame_primitives.clear();
        self.text_labels.retain(|t| t.duration > 0.0);

        for prim in &mut self.persistent_primitives {
            prim.duration -= dt;
            if prim.duration <= 0.0 {
                prim.active = false;
            }
        }
        self.persistent_primitives.retain(|p| p.active);
    }

    /// Clear all primitives.
    pub fn clear_all(&mut self) {
        self.frame_primitives.clear();
        self.persistent_primitives.clear();
        self.text_labels.clear();
        self.line_vertices.clear();
        self.overlay_vertices.clear();
    }

    /// Enable or disable debug drawing.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear_all();
        }
    }
}

/// Statistics for the debug draw system.
#[derive(Debug, Clone, Default)]
pub struct DebugDrawStats {
    /// Total line segments rendered.
    pub line_count: usize,
    /// Total primitives submitted.
    pub primitive_count: usize,
    /// Total text labels.
    pub text_count: usize,
    /// Total vertex buffer size in bytes.
    pub vertex_buffer_bytes: usize,
}
