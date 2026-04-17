// engine/debug/src/gizmos.rs
//
// Runtime gizmo drawing for the Genovo engine.
//
// Provides an immediate-mode API for drawing debug shapes in the 3D scene:
// axis arrows, collider wireframes, AABBs, skeletons, paths, velocity arrows,
// springs, and 3D-anchored text labels.
//
// All gizmos are batched into vertex buffers per frame and rendered in a
// single pass. Two modes are supported:
//
// - **Depth-tested**: gizmos are occluded by scene geometry.
// - **Overlay**: gizmos render on top of everything.

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// GizmoColor
// ---------------------------------------------------------------------------

/// RGBA color for gizmo drawing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GizmoColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl GizmoColor {
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Self = Self { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Self = Self { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const CYAN: Self = Self { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const MAGENTA: Self = Self { r: 1.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const ORANGE: Self = Self { r: 1.0, g: 0.5, b: 0.0, a: 1.0 };
    pub const GRAY: Self = Self { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };

    /// Creates a new color.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Creates a color from RGB with full alpha.
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Returns the color with modified alpha.
    pub const fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    /// Converts to a Vec4 (r, g, b, a).
    pub fn to_vec4(self) -> Vec4 {
        Vec4::new(self.r, self.g, self.b, self.a)
    }

    /// Converts to a `[f32; 4]` array.
    pub fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

impl From<Vec4> for GizmoColor {
    fn from(v: Vec4) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }
}

impl From<GizmoColor> for Vec4 {
    fn from(c: GizmoColor) -> Self {
        Vec4::new(c.r, c.g, c.b, c.a)
    }
}

// ---------------------------------------------------------------------------
// GizmoVertex
// ---------------------------------------------------------------------------

/// A single gizmo vertex with position and color.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GizmoVertex {
    pub position: Vec3,
    pub color: GizmoColor,
}

impl GizmoVertex {
    pub fn new(position: Vec3, color: GizmoColor) -> Self {
        Self { position, color }
    }
}

// ---------------------------------------------------------------------------
// GizmoDepthMode
// ---------------------------------------------------------------------------

/// Whether gizmos are depth-tested or drawn as overlays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GizmoDepthMode {
    /// Gizmos are depth-tested against the scene depth buffer.
    DepthTested,
    /// Gizmos render on top of everything.
    Overlay,
}

impl Default for GizmoDepthMode {
    fn default() -> Self {
        GizmoDepthMode::DepthTested
    }
}

// ---------------------------------------------------------------------------
// ColliderShape
// ---------------------------------------------------------------------------

/// Collider shapes that gizmos can visualize.
#[derive(Debug, Clone, Copy)]
pub enum ColliderShape {
    /// Sphere with a radius.
    Sphere { radius: f32 },
    /// Axis-aligned box with half-extents.
    Box { half_extents: Vec3 },
    /// Capsule along the Y axis.
    Capsule { radius: f32, half_height: f32 },
    /// Cylinder along the Y axis.
    Cylinder { radius: f32, half_height: f32 },
    /// Cone with base radius and height along Y.
    Cone { radius: f32, height: f32 },
}

// ---------------------------------------------------------------------------
// GizmoLabel
// ---------------------------------------------------------------------------

/// A text label anchored at a 3D world position.
#[derive(Debug, Clone)]
pub struct GizmoLabel {
    /// World-space position of the label anchor.
    pub position: Vec3,
    /// Text to display.
    pub text: String,
    /// Text color.
    pub color: GizmoColor,
    /// Offset in screen pixels from the projected position.
    pub screen_offset: Vec2,
    /// Font size in pixels.
    pub font_size: f32,
}

// ---------------------------------------------------------------------------
// GizmoBatch
// ---------------------------------------------------------------------------

/// Accumulated gizmo geometry for a single frame and depth mode.
#[derive(Debug, Default)]
pub struct GizmoBatch {
    /// Line segments: pairs of vertices.
    pub line_vertices: Vec<GizmoVertex>,
    /// 3D text labels.
    pub labels: Vec<GizmoLabel>,
}

impl GizmoBatch {
    /// Clears all accumulated geometry.
    pub fn clear(&mut self) {
        self.line_vertices.clear();
        self.labels.clear();
    }

    /// Returns the number of line segments.
    pub fn line_count(&self) -> usize {
        self.line_vertices.len() / 2
    }

    /// Returns true if the batch has no geometry.
    pub fn is_empty(&self) -> bool {
        self.line_vertices.is_empty() && self.labels.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Gizmos
// ---------------------------------------------------------------------------

/// Immediate-mode gizmo drawing API.
///
/// Call draw methods during your frame update. At the end of the frame,
/// retrieve the batches via [`take_batches`] and render them.
///
/// All draw methods accept a [`GizmoDepthMode`] parameter to choose
/// between depth-tested and overlay rendering.
#[derive(Debug, Default)]
pub struct Gizmos {
    /// Depth-tested gizmo batch.
    depth_tested: GizmoBatch,
    /// Overlay gizmo batch.
    overlay: GizmoBatch,
    /// Whether gizmo drawing is enabled.
    enabled: bool,
    /// Default depth mode for convenience methods.
    default_depth_mode: GizmoDepthMode,
}

impl Gizmos {
    /// Creates a new gizmo context with drawing enabled.
    pub fn new() -> Self {
        Self {
            depth_tested: GizmoBatch::default(),
            overlay: GizmoBatch::default(),
            enabled: true,
            default_depth_mode: GizmoDepthMode::DepthTested,
        }
    }

    /// Sets whether gizmo drawing is enabled.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether gizmo drawing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Sets the default depth mode.
    pub fn set_default_depth_mode(&mut self, mode: GizmoDepthMode) {
        self.default_depth_mode = mode;
    }

    /// Returns a mutable reference to the batch for the given depth mode.
    fn batch_mut(&mut self, mode: GizmoDepthMode) -> &mut GizmoBatch {
        match mode {
            GizmoDepthMode::DepthTested => &mut self.depth_tested,
            GizmoDepthMode::Overlay => &mut self.overlay,
        }
    }

    /// Takes the accumulated batches, leaving empty batches in their place.
    pub fn take_batches(&mut self) -> (GizmoBatch, GizmoBatch) {
        let depth = std::mem::take(&mut self.depth_tested);
        let overlay = std::mem::take(&mut self.overlay);
        (depth, overlay)
    }

    /// Clears all accumulated gizmo geometry.
    pub fn clear(&mut self) {
        self.depth_tested.clear();
        self.overlay.clear();
    }

    // -----------------------------------------------------------------------
    // Primitive drawing
    // -----------------------------------------------------------------------

    /// Draws a line segment between two points.
    pub fn draw_line(
        &mut self,
        start: Vec3,
        end: Vec3,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }
        let batch = self.batch_mut(mode);
        batch.line_vertices.push(GizmoVertex::new(start, color));
        batch.line_vertices.push(GizmoVertex::new(end, color));
    }

    /// Draws a line using the default depth mode.
    pub fn line(&mut self, start: Vec3, end: Vec3, color: GizmoColor) {
        let mode = self.default_depth_mode;
        self.draw_line(start, end, color, mode);
    }

    /// Draws an arrow from `start` to `end` with an arrowhead.
    pub fn draw_arrow(
        &mut self,
        start: Vec3,
        end: Vec3,
        color: GizmoColor,
        head_size: f32,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        self.draw_line(start, end, color, mode);

        let dir = (end - start).normalize_or_zero();
        if dir.length_squared() < 0.001 {
            return;
        }

        // Build perpendicular vectors for the arrowhead
        let up = if dir.y.abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let right = dir.cross(up).normalize() * head_size;
        let up_perp = dir.cross(right).normalize() * head_size;

        let tip = end;
        let base = end - dir * head_size * 2.0;

        self.draw_line(tip, base + right, color, mode);
        self.draw_line(tip, base - right, color, mode);
        self.draw_line(tip, base + up_perp, color, mode);
        self.draw_line(tip, base - up_perp, color, mode);
    }

    /// Draws a circle in the XZ plane.
    pub fn draw_circle_xz(
        &mut self,
        center: Vec3,
        radius: f32,
        segments: u32,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        let step = 2.0 * PI / segments as f32;
        for i in 0..segments {
            let a0 = step * i as f32;
            let a1 = step * (i + 1) as f32;
            let p0 = center + Vec3::new(a0.cos() * radius, 0.0, a0.sin() * radius);
            let p1 = center + Vec3::new(a1.cos() * radius, 0.0, a1.sin() * radius);
            self.draw_line(p0, p1, color, mode);
        }
    }

    /// Draws a circle in the XY plane.
    pub fn draw_circle_xy(
        &mut self,
        center: Vec3,
        radius: f32,
        segments: u32,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        let step = 2.0 * PI / segments as f32;
        for i in 0..segments {
            let a0 = step * i as f32;
            let a1 = step * (i + 1) as f32;
            let p0 = center + Vec3::new(a0.cos() * radius, a0.sin() * radius, 0.0);
            let p1 = center + Vec3::new(a1.cos() * radius, a1.sin() * radius, 0.0);
            self.draw_line(p0, p1, color, mode);
        }
    }

    /// Draws a circle in the YZ plane.
    pub fn draw_circle_yz(
        &mut self,
        center: Vec3,
        radius: f32,
        segments: u32,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        let step = 2.0 * PI / segments as f32;
        for i in 0..segments {
            let a0 = step * i as f32;
            let a1 = step * (i + 1) as f32;
            let p0 = center + Vec3::new(0.0, a0.cos() * radius, a0.sin() * radius);
            let p1 = center + Vec3::new(0.0, a1.cos() * radius, a1.sin() * radius);
            self.draw_line(p0, p1, color, mode);
        }
    }

    /// Draws a wireframe sphere.
    pub fn draw_wire_sphere(
        &mut self,
        center: Vec3,
        radius: f32,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        let segments = 24;
        self.draw_circle_xz(center, radius, segments, color, mode);
        self.draw_circle_xy(center, radius, segments, color, mode);
        self.draw_circle_yz(center, radius, segments, color, mode);
    }

    // -----------------------------------------------------------------------
    // High-level gizmo drawing
    // -----------------------------------------------------------------------

    /// Draws a transform gizmo: RGB axis arrows showing position, rotation,
    /// and scale.
    ///
    /// - Red arrow = X axis
    /// - Green arrow = Y axis
    /// - Blue arrow = Z axis
    pub fn draw_transform(
        &mut self,
        position: Vec3,
        rotation: Quat,
        scale: f32,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        let length = scale;
        let head = scale * 0.15;

        let x_axis = rotation * Vec3::X * length;
        let y_axis = rotation * Vec3::Y * length;
        let z_axis = rotation * Vec3::Z * length;

        self.draw_arrow(position, position + x_axis, GizmoColor::RED, head, mode);
        self.draw_arrow(position, position + y_axis, GizmoColor::GREEN, head, mode);
        self.draw_arrow(position, position + z_axis, GizmoColor::BLUE, head, mode);
    }

    /// Draws a wireframe collider shape at the given transform.
    pub fn draw_collider(
        &mut self,
        shape: &ColliderShape,
        position: Vec3,
        rotation: Quat,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        match *shape {
            ColliderShape::Sphere { radius } => {
                self.draw_wire_sphere(position, radius, color, mode);
            }
            ColliderShape::Box { half_extents } => {
                self.draw_wireframe_box(position, rotation, half_extents, color, mode);
            }
            ColliderShape::Capsule {
                radius,
                half_height,
            } => {
                self.draw_capsule(position, rotation, radius, half_height, color, mode);
            }
            ColliderShape::Cylinder {
                radius,
                half_height,
            } => {
                self.draw_cylinder(position, rotation, radius, half_height, color, mode);
            }
            ColliderShape::Cone { radius, height } => {
                self.draw_cone(position, rotation, radius, height, color, mode);
            }
        }
    }

    /// Draws a wireframe AABB.
    pub fn draw_bounds(
        &mut self,
        min: Vec3,
        max: Vec3,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        // 8 corners of the AABB
        let corners = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];

        // 12 edges
        let edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), // front face
            (4, 5), (5, 6), (6, 7), (7, 4), // back face
            (0, 4), (1, 5), (2, 6), (3, 7), // connecting edges
        ];

        for (a, b) in &edges {
            self.draw_line(corners[*a], corners[*b], color, mode);
        }
    }

    /// Draws a skeleton visualization from an array of bone transforms.
    ///
    /// `bone_transforms`: world-space transforms for each bone.
    /// `parent_indices`: for each bone, the index of its parent (-1 for roots).
    pub fn draw_skeleton(
        &mut self,
        bone_transforms: &[Mat4],
        parent_indices: &[i32],
        color: GizmoColor,
        joint_size: f32,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        for (i, transform) in bone_transforms.iter().enumerate() {
            let pos = transform.col(3).truncate();

            // Draw a small cross at each joint
            let half = joint_size * 0.5;
            self.draw_line(
                pos - Vec3::X * half,
                pos + Vec3::X * half,
                color,
                mode,
            );
            self.draw_line(
                pos - Vec3::Y * half,
                pos + Vec3::Y * half,
                color,
                mode,
            );
            self.draw_line(
                pos - Vec3::Z * half,
                pos + Vec3::Z * half,
                color,
                mode,
            );

            // Draw bone connecting to parent
            if i < parent_indices.len() && parent_indices[i] >= 0 {
                let parent_idx = parent_indices[i] as usize;
                if parent_idx < bone_transforms.len() {
                    let parent_pos = bone_transforms[parent_idx].col(3).truncate();
                    self.draw_line(parent_pos, pos, color, mode);

                    // Draw a diamond shape for the bone
                    let mid = (parent_pos + pos) * 0.5;
                    let dir = (pos - parent_pos).normalize_or_zero();
                    let bone_len = (pos - parent_pos).length();
                    let thickness = (bone_len * 0.1).min(joint_size);

                    let up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
                    let right = dir.cross(up).normalize() * thickness;
                    let up_perp = dir.cross(right).normalize() * thickness;

                    let diamond_color = GizmoColor::new(
                        color.r * 0.7,
                        color.g * 0.7,
                        color.b * 0.7,
                        color.a,
                    );
                    self.draw_line(parent_pos, mid + right, diamond_color, mode);
                    self.draw_line(parent_pos, mid - right, diamond_color, mode);
                    self.draw_line(parent_pos, mid + up_perp, diamond_color, mode);
                    self.draw_line(parent_pos, mid - up_perp, diamond_color, mode);
                    self.draw_line(pos, mid + right, diamond_color, mode);
                    self.draw_line(pos, mid - right, diamond_color, mode);
                    self.draw_line(pos, mid + up_perp, diamond_color, mode);
                    self.draw_line(pos, mid - up_perp, diamond_color, mode);
                }
            }
        }
    }

    /// Draws a path as a polyline with directional arrows.
    pub fn draw_path(
        &mut self,
        points: &[Vec3],
        color: GizmoColor,
        arrow_interval: usize,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled || points.len() < 2 {
            return;
        }

        for i in 0..points.len() - 1 {
            self.draw_line(points[i], points[i + 1], color, mode);

            // Draw an arrowhead at regular intervals
            if arrow_interval > 0 && i % arrow_interval == 0 {
                let dir = (points[i + 1] - points[i]).normalize_or_zero();
                let mid = (points[i] + points[i + 1]) * 0.5;
                let segment_len = (points[i + 1] - points[i]).length();
                let head_size = (segment_len * 0.15).min(0.2);
                self.draw_arrow(mid - dir * head_size, mid + dir * head_size, color, head_size, mode);
            }
        }
    }

    /// Draws a velocity arrow showing the direction and magnitude of a
    /// velocity vector.
    pub fn draw_velocity(
        &mut self,
        position: Vec3,
        velocity: Vec3,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        let speed = velocity.length();
        if speed < 0.001 {
            return;
        }

        let head_size = (speed * 0.1).clamp(0.05, 0.3);
        self.draw_arrow(position, position + velocity, color, head_size, mode);
    }

    /// Draws a spring coil visualization between two points.
    pub fn draw_spring(
        &mut self,
        start: Vec3,
        end: Vec3,
        coils: u32,
        radius: f32,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        let dir = end - start;
        let length = dir.length();
        if length < 0.001 {
            return;
        }
        let norm_dir = dir / length;

        // Build a local frame
        let up = if norm_dir.y.abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let right = norm_dir.cross(up).normalize() * radius;
        let up_perp = norm_dir.cross(right).normalize() * radius;

        // Straight lead-in
        let lead_fraction = 0.1;
        let lead_start = start;
        let coil_start = start + dir * lead_fraction;
        let coil_end = start + dir * (1.0 - lead_fraction);
        let lead_end = end;

        self.draw_line(lead_start, coil_start, color, mode);

        // Coil section
        let total_segments = coils * 16;
        let coil_length = (1.0 - 2.0 * lead_fraction) * length;

        let mut prev = coil_start;
        for i in 1..=total_segments {
            let t = i as f32 / total_segments as f32;
            let angle = t * coils as f32 * 2.0 * PI;
            let along = coil_start + norm_dir * (t * coil_length);
            let offset = right * angle.cos() + up_perp * angle.sin();
            let point = along + offset;
            self.draw_line(prev, point, color, mode);
            prev = point;
        }

        self.draw_line(prev, lead_end, color, mode);
    }

    /// Draws a text label anchored at a 3D world position.
    pub fn draw_label_3d(
        &mut self,
        position: Vec3,
        text: impl Into<String>,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        let label = GizmoLabel {
            position,
            text: text.into(),
            color,
            screen_offset: Vec2::ZERO,
            font_size: 14.0,
        };

        let batch = self.batch_mut(mode);
        batch.labels.push(label);
    }

    /// Draws a text label with a screen-space offset and custom font size.
    pub fn draw_label_3d_ext(
        &mut self,
        position: Vec3,
        text: impl Into<String>,
        color: GizmoColor,
        screen_offset: Vec2,
        font_size: f32,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        let label = GizmoLabel {
            position,
            text: text.into(),
            color,
            screen_offset,
            font_size,
        };

        let batch = self.batch_mut(mode);
        batch.labels.push(label);
    }

    /// Draws a wireframe grid in the XZ plane.
    pub fn draw_grid(
        &mut self,
        center: Vec3,
        size: f32,
        divisions: u32,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        let half = size * 0.5;
        let step = size / divisions as f32;

        for i in 0..=divisions {
            let offset = -half + step * i as f32;

            // Lines along Z
            self.draw_line(
                center + Vec3::new(offset, 0.0, -half),
                center + Vec3::new(offset, 0.0, half),
                color,
                mode,
            );

            // Lines along X
            self.draw_line(
                center + Vec3::new(-half, 0.0, offset),
                center + Vec3::new(half, 0.0, offset),
                color,
                mode,
            );
        }
    }

    /// Draws a wireframe frustum from a view-projection matrix.
    pub fn draw_frustum(
        &mut self,
        view_proj: Mat4,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        if !self.enabled {
            return;
        }

        let inv_vp = view_proj.inverse();

        // NDC corners (near z=-1, far z=1 in OpenGL convention)
        let ndc_corners = [
            Vec4::new(-1.0, -1.0, -1.0, 1.0),
            Vec4::new(1.0, -1.0, -1.0, 1.0),
            Vec4::new(1.0, 1.0, -1.0, 1.0),
            Vec4::new(-1.0, 1.0, -1.0, 1.0),
            Vec4::new(-1.0, -1.0, 1.0, 1.0),
            Vec4::new(1.0, -1.0, 1.0, 1.0),
            Vec4::new(1.0, 1.0, 1.0, 1.0),
            Vec4::new(-1.0, 1.0, 1.0, 1.0),
        ];

        let world_corners: Vec<Vec3> = ndc_corners
            .iter()
            .map(|&ndc| {
                let world = inv_vp * ndc;
                Vec3::new(world.x / world.w, world.y / world.w, world.z / world.w)
            })
            .collect();

        // Near plane
        for i in 0..4 {
            self.draw_line(world_corners[i], world_corners[(i + 1) % 4], color, mode);
        }
        // Far plane
        for i in 4..8 {
            self.draw_line(world_corners[i], world_corners[4 + (i - 4 + 1) % 4], color, mode);
        }
        // Connecting edges
        for i in 0..4 {
            self.draw_line(world_corners[i], world_corners[i + 4], color, mode);
        }
    }

    // -----------------------------------------------------------------------
    // Private shape helpers
    // -----------------------------------------------------------------------

    /// Draws a wireframe box.
    fn draw_wireframe_box(
        &mut self,
        center: Vec3,
        rotation: Quat,
        half_extents: Vec3,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        let axes = [
            rotation * Vec3::X * half_extents.x,
            rotation * Vec3::Y * half_extents.y,
            rotation * Vec3::Z * half_extents.z,
        ];

        let corners = [
            center - axes[0] - axes[1] - axes[2],
            center + axes[0] - axes[1] - axes[2],
            center + axes[0] + axes[1] - axes[2],
            center - axes[0] + axes[1] - axes[2],
            center - axes[0] - axes[1] + axes[2],
            center + axes[0] - axes[1] + axes[2],
            center + axes[0] + axes[1] + axes[2],
            center - axes[0] + axes[1] + axes[2],
        ];

        let edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];

        for (a, b) in &edges {
            self.draw_line(corners[*a], corners[*b], color, mode);
        }
    }

    /// Draws a wireframe capsule.
    fn draw_capsule(
        &mut self,
        center: Vec3,
        rotation: Quat,
        radius: f32,
        half_height: f32,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        let up = rotation * Vec3::Y;
        let top = center + up * half_height;
        let bottom = center - up * half_height;

        // Circles at top and bottom of the cylinder part
        let segments = 16;
        let right = rotation * Vec3::X;
        let forward = rotation * Vec3::Z;

        for i in 0..segments {
            let a0 = 2.0 * PI * i as f32 / segments as f32;
            let a1 = 2.0 * PI * (i + 1) as f32 / segments as f32;

            let p0 = right * a0.cos() * radius + forward * a0.sin() * radius;
            let p1 = right * a1.cos() * radius + forward * a1.sin() * radius;

            // Top circle
            self.draw_line(top + p0, top + p1, color, mode);
            // Bottom circle
            self.draw_line(bottom + p0, bottom + p1, color, mode);

            // Vertical lines at cardinal directions
            if i % (segments / 4) == 0 {
                self.draw_line(top + p0, bottom + p0, color, mode);
            }
        }

        // Semicircles for the hemisphere caps
        let half_segments = segments / 2;
        for i in 0..half_segments {
            let a0 = PI * i as f32 / half_segments as f32;
            let a1 = PI * (i + 1) as f32 / half_segments as f32;

            // Top cap (XY plane)
            let t0 = top + right * a0.sin() * radius + up * a0.cos() * radius;
            let t1 = top + right * a1.sin() * radius + up * a1.cos() * radius;
            self.draw_line(t0, t1, color, mode);

            // Top cap (ZY plane)
            let t0 = top + forward * a0.sin() * radius + up * a0.cos() * radius;
            let t1 = top + forward * a1.sin() * radius + up * a1.cos() * radius;
            self.draw_line(t0, t1, color, mode);

            // Bottom cap (XY plane)
            let b0 = bottom + right * a0.sin() * radius - up * a0.cos() * radius;
            let b1 = bottom + right * a1.sin() * radius - up * a1.cos() * radius;
            self.draw_line(b0, b1, color, mode);

            // Bottom cap (ZY plane)
            let b0 = bottom + forward * a0.sin() * radius - up * a0.cos() * radius;
            let b1 = bottom + forward * a1.sin() * radius - up * a1.cos() * radius;
            self.draw_line(b0, b1, color, mode);
        }
    }

    /// Draws a wireframe cylinder.
    fn draw_cylinder(
        &mut self,
        center: Vec3,
        rotation: Quat,
        radius: f32,
        half_height: f32,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        let up = rotation * Vec3::Y;
        let top = center + up * half_height;
        let bottom = center - up * half_height;

        let segments = 16;
        let right = rotation * Vec3::X;
        let forward = rotation * Vec3::Z;

        for i in 0..segments {
            let a0 = 2.0 * PI * i as f32 / segments as f32;
            let a1 = 2.0 * PI * (i + 1) as f32 / segments as f32;

            let p0 = right * a0.cos() * radius + forward * a0.sin() * radius;
            let p1 = right * a1.cos() * radius + forward * a1.sin() * radius;

            self.draw_line(top + p0, top + p1, color, mode);
            self.draw_line(bottom + p0, bottom + p1, color, mode);

            if i % (segments / 4) == 0 {
                self.draw_line(top + p0, bottom + p0, color, mode);
            }
        }
    }

    /// Draws a wireframe cone.
    fn draw_cone(
        &mut self,
        center: Vec3,
        rotation: Quat,
        radius: f32,
        height: f32,
        color: GizmoColor,
        mode: GizmoDepthMode,
    ) {
        let up = rotation * Vec3::Y;
        let tip = center + up * height;
        let base_center = center;

        let segments = 16;
        let right = rotation * Vec3::X;
        let forward = rotation * Vec3::Z;

        for i in 0..segments {
            let a0 = 2.0 * PI * i as f32 / segments as f32;
            let a1 = 2.0 * PI * (i + 1) as f32 / segments as f32;

            let p0 = base_center + right * a0.cos() * radius + forward * a0.sin() * radius;
            let p1 = base_center + right * a1.cos() * radius + forward * a1.sin() * radius;

            self.draw_line(p0, p1, color, mode);

            if i % (segments / 4) == 0 {
                self.draw_line(tip, p0, color, mode);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GizmosComponent
// ---------------------------------------------------------------------------

/// ECS component that provides per-entity gizmo drawing configuration.
#[derive(Debug, Clone)]
pub struct GizmosComponent {
    /// Whether to draw the transform gizmo for this entity.
    pub show_transform: bool,
    /// Scale of the transform gizmo arrows.
    pub transform_scale: f32,
    /// Whether to draw the entity's collider.
    pub show_collider: bool,
    /// Whether to draw the entity's AABB.
    pub show_bounds: bool,
    /// Whether to draw velocity vector.
    pub show_velocity: bool,
    /// Color override (if set, all gizmos use this color).
    pub color_override: Option<GizmoColor>,
    /// Depth mode for this entity's gizmos.
    pub depth_mode: GizmoDepthMode,
}

impl Default for GizmosComponent {
    fn default() -> Self {
        Self {
            show_transform: true,
            transform_scale: 1.0,
            show_collider: false,
            show_bounds: false,
            show_velocity: false,
            color_override: None,
            depth_mode: GizmoDepthMode::DepthTested,
        }
    }
}

impl GizmosComponent {
    /// Creates a new component with transform gizmo enabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables collider visualization.
    pub fn with_collider(mut self) -> Self {
        self.show_collider = true;
        self
    }

    /// Enables AABB visualization.
    pub fn with_bounds(mut self) -> Self {
        self.show_bounds = true;
        self
    }

    /// Enables velocity visualization.
    pub fn with_velocity(mut self) -> Self {
        self.show_velocity = true;
        self
    }

    /// Sets the depth mode.
    pub fn with_depth_mode(mut self, mode: GizmoDepthMode) -> Self {
        self.depth_mode = mode;
        self
    }

    /// Sets a color override.
    pub fn with_color(mut self, color: GizmoColor) -> Self {
        self.color_override = Some(color);
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gizmo_color() {
        let c = GizmoColor::RED.with_alpha(0.5);
        assert_eq!(c.r, 1.0);
        assert_eq!(c.a, 0.5);

        let v: Vec4 = c.into();
        assert_eq!(v, Vec4::new(1.0, 0.0, 0.0, 0.5));

        let back: GizmoColor = v.into();
        assert_eq!(back, c);
    }

    #[test]
    fn test_draw_line() {
        let mut gizmos = Gizmos::new();
        gizmos.draw_line(Vec3::ZERO, Vec3::X, GizmoColor::RED, GizmoDepthMode::DepthTested);
        let (depth, overlay) = gizmos.take_batches();
        assert_eq!(depth.line_count(), 1);
        assert!(overlay.is_empty());
    }

    #[test]
    fn test_draw_line_overlay() {
        let mut gizmos = Gizmos::new();
        gizmos.draw_line(Vec3::ZERO, Vec3::X, GizmoColor::RED, GizmoDepthMode::Overlay);
        let (depth, overlay) = gizmos.take_batches();
        assert!(depth.is_empty());
        assert_eq!(overlay.line_count(), 1);
    }

    #[test]
    fn test_disabled_drawing() {
        let mut gizmos = Gizmos::new();
        gizmos.set_enabled(false);
        gizmos.draw_line(Vec3::ZERO, Vec3::X, GizmoColor::RED, GizmoDepthMode::DepthTested);
        let (depth, _) = gizmos.take_batches();
        assert!(depth.is_empty());
    }

    #[test]
    fn test_draw_transform() {
        let mut gizmos = Gizmos::new();
        gizmos.draw_transform(Vec3::ZERO, Quat::IDENTITY, 1.0, GizmoDepthMode::DepthTested);
        let (depth, _) = gizmos.take_batches();
        // 3 arrows, each with 1 shaft + 4 head lines = 15 lines
        assert_eq!(depth.line_count(), 15);
    }

    #[test]
    fn test_draw_bounds() {
        let mut gizmos = Gizmos::new();
        gizmos.draw_bounds(Vec3::ZERO, Vec3::ONE, GizmoColor::YELLOW, GizmoDepthMode::DepthTested);
        let (depth, _) = gizmos.take_batches();
        assert_eq!(depth.line_count(), 12); // 12 edges of a box
    }

    #[test]
    fn test_draw_path() {
        let mut gizmos = Gizmos::new();
        let points = vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)];
        gizmos.draw_path(&points, GizmoColor::GREEN, 1, GizmoDepthMode::DepthTested);
        let (depth, _) = gizmos.take_batches();
        // 2 line segments + arrows at intervals
        assert!(depth.line_count() >= 2);
    }

    #[test]
    fn test_draw_spring() {
        let mut gizmos = Gizmos::new();
        gizmos.draw_spring(
            Vec3::ZERO,
            Vec3::new(0.0, 5.0, 0.0),
            5,
            0.3,
            GizmoColor::CYAN,
            GizmoDepthMode::DepthTested,
        );
        let (depth, _) = gizmos.take_batches();
        // Lead-in + coil segments + lead-out
        assert!(depth.line_count() > 10);
    }

    #[test]
    fn test_draw_label() {
        let mut gizmos = Gizmos::new();
        gizmos.draw_label_3d(Vec3::ZERO, "Hello", GizmoColor::WHITE, GizmoDepthMode::Overlay);
        let (_, overlay) = gizmos.take_batches();
        assert_eq!(overlay.labels.len(), 1);
        assert_eq!(overlay.labels[0].text, "Hello");
    }

    #[test]
    fn test_draw_grid() {
        let mut gizmos = Gizmos::new();
        gizmos.draw_grid(Vec3::ZERO, 10.0, 10, GizmoColor::GRAY, GizmoDepthMode::DepthTested);
        let (depth, _) = gizmos.take_batches();
        // 11 lines in each direction = 22 lines
        assert_eq!(depth.line_count(), 22);
    }

    #[test]
    fn test_clear() {
        let mut gizmos = Gizmos::new();
        gizmos.draw_line(Vec3::ZERO, Vec3::X, GizmoColor::RED, GizmoDepthMode::DepthTested);
        gizmos.clear();
        let (depth, _) = gizmos.take_batches();
        assert!(depth.is_empty());
    }

    #[test]
    fn test_component_builder() {
        let comp = GizmosComponent::new()
            .with_collider()
            .with_bounds()
            .with_velocity()
            .with_color(GizmoColor::MAGENTA)
            .with_depth_mode(GizmoDepthMode::Overlay);

        assert!(comp.show_collider);
        assert!(comp.show_bounds);
        assert!(comp.show_velocity);
        assert_eq!(comp.color_override, Some(GizmoColor::MAGENTA));
        assert_eq!(comp.depth_mode, GizmoDepthMode::Overlay);
    }

    #[test]
    fn test_wire_sphere() {
        let mut gizmos = Gizmos::new();
        gizmos.draw_wire_sphere(Vec3::ZERO, 1.0, GizmoColor::WHITE, GizmoDepthMode::DepthTested);
        let (depth, _) = gizmos.take_batches();
        // 3 circles * 24 segments = 72 lines
        assert_eq!(depth.line_count(), 72);
    }

    #[test]
    fn test_default_depth_mode() {
        let mut gizmos = Gizmos::new();
        gizmos.set_default_depth_mode(GizmoDepthMode::Overlay);
        gizmos.line(Vec3::ZERO, Vec3::X, GizmoColor::RED);
        let (depth, overlay) = gizmos.take_batches();
        assert!(depth.is_empty());
        assert_eq!(overlay.line_count(), 1);
    }
}
