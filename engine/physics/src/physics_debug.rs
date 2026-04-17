//! Physics debug visualization for rendering collision shapes, contacts,
//! joints, and simulation state.
//!
//! Provides:
//! - Draw collision shapes (sphere, box, capsule, mesh outline)
//! - Contact points and contact normals visualization
//! - Joint limit arcs and joint axes
//! - Sleeping body indicator
//! - Velocity and angular velocity arrows
//! - Broadphase grid visualization
//! - Constraint error display (color-coded)
//! - Configurable per-category enable/disable and colors
//! - Output as a list of debug primitives for any renderer

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default color for collision shape wireframes (green).
const COLOR_COLLISION_SHAPE: Vec4 = Vec4::new(0.2, 0.9, 0.2, 0.8);
/// Default color for contact points (red).
const COLOR_CONTACT_POINT: Vec4 = Vec4::new(1.0, 0.2, 0.2, 1.0);
/// Default color for contact normals (yellow).
const COLOR_CONTACT_NORMAL: Vec4 = Vec4::new(1.0, 1.0, 0.2, 1.0);
/// Default color for joint axes (cyan).
const COLOR_JOINT_AXIS: Vec4 = Vec4::new(0.2, 0.9, 0.9, 0.9);
/// Default color for joint limits (orange).
const COLOR_JOINT_LIMIT: Vec4 = Vec4::new(1.0, 0.6, 0.1, 0.7);
/// Default color for velocity arrows (blue).
const COLOR_VELOCITY: Vec4 = Vec4::new(0.3, 0.3, 1.0, 0.8);
/// Default color for angular velocity (magenta).
const COLOR_ANGULAR_VELOCITY: Vec4 = Vec4::new(0.9, 0.2, 0.9, 0.8);
/// Default color for sleeping bodies (gray).
const COLOR_SLEEPING: Vec4 = Vec4::new(0.5, 0.5, 0.5, 0.4);
/// Default color for broadphase grid (dim green).
const COLOR_BROADPHASE: Vec4 = Vec4::new(0.1, 0.4, 0.1, 0.2);
/// Default color for AABB wireframes (white).
const COLOR_AABB: Vec4 = Vec4::new(1.0, 1.0, 1.0, 0.3);
/// Default contact normal display length.
const CONTACT_NORMAL_LENGTH: f32 = 0.3;
/// Default velocity arrow scale.
const VELOCITY_ARROW_SCALE: f32 = 0.2;
/// Default angular velocity arrow scale.
const ANGULAR_VELOCITY_ARROW_SCALE: f32 = 0.3;
/// Number of segments for circle/arc approximations.
const CIRCLE_SEGMENTS: usize = 24;
/// Number of segments for sphere wireframe rings.
const SPHERE_RING_COUNT: usize = 3;
/// Epsilon for numerical stability.
const EPSILON: f32 = 1e-7;
/// Default contact point radius.
const CONTACT_POINT_RADIUS: f32 = 0.02;

// ---------------------------------------------------------------------------
// Debug primitive types
// ---------------------------------------------------------------------------

/// A debug draw primitive to be rendered by the application.
#[derive(Debug, Clone)]
pub enum DebugPrimitive {
    /// A line segment.
    Line {
        start: Vec3,
        end: Vec3,
        color: Vec4,
    },
    /// A point (rendered as a small sphere or dot).
    Point {
        position: Vec3,
        radius: f32,
        color: Vec4,
    },
    /// A wireframe sphere.
    WireSphere {
        center: Vec3,
        radius: f32,
        color: Vec4,
    },
    /// A wireframe box (oriented).
    WireBox {
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
        color: Vec4,
    },
    /// A wireframe capsule.
    WireCapsule {
        start: Vec3,
        end: Vec3,
        radius: f32,
        color: Vec4,
    },
    /// An arrow (line with arrowhead).
    Arrow {
        start: Vec3,
        end: Vec3,
        color: Vec4,
        head_size: f32,
    },
    /// A wireframe circle/arc.
    Arc {
        center: Vec3,
        normal: Vec3,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        color: Vec4,
    },
    /// A text label at a world position.
    Text {
        position: Vec3,
        text: String,
        color: Vec4,
    },
    /// An axis-aligned bounding box.
    WireAABB {
        min: Vec3,
        max: Vec3,
        color: Vec4,
    },
    /// A triangle (wireframe).
    WireTriangle {
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        color: Vec4,
    },
    /// A filled circle (for indicators).
    FilledCircle {
        center: Vec3,
        normal: Vec3,
        radius: f32,
        color: Vec4,
    },
}

impl DebugPrimitive {
    /// Create a line.
    pub fn line(start: Vec3, end: Vec3, color: Vec4) -> Self {
        DebugPrimitive::Line { start, end, color }
    }

    /// Create a point.
    pub fn point(position: Vec3, color: Vec4) -> Self {
        DebugPrimitive::Point {
            position,
            radius: CONTACT_POINT_RADIUS,
            color,
        }
    }

    /// Create an arrow.
    pub fn arrow(start: Vec3, end: Vec3, color: Vec4) -> Self {
        let length = (end - start).length();
        DebugPrimitive::Arrow {
            start,
            end,
            color,
            head_size: length * 0.15,
        }
    }

    /// Create a text label.
    pub fn text(position: Vec3, text: &str, color: Vec4) -> Self {
        DebugPrimitive::Text {
            position,
            text: text.to_string(),
            color,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug draw settings
// ---------------------------------------------------------------------------

/// Per-category color overrides.
#[derive(Debug, Clone)]
pub struct DebugColors {
    /// Collision shape wireframe color.
    pub collision_shape: Vec4,
    /// Contact point color.
    pub contact_point: Vec4,
    /// Contact normal color.
    pub contact_normal: Vec4,
    /// Joint axis color.
    pub joint_axis: Vec4,
    /// Joint limit color.
    pub joint_limit: Vec4,
    /// Velocity arrow color.
    pub velocity: Vec4,
    /// Angular velocity arrow color.
    pub angular_velocity: Vec4,
    /// Sleeping body indicator color.
    pub sleeping: Vec4,
    /// Broadphase grid color.
    pub broadphase: Vec4,
    /// AABB wireframe color.
    pub aabb: Vec4,
    /// Constraint error (low) color.
    pub constraint_error_low: Vec4,
    /// Constraint error (high) color.
    pub constraint_error_high: Vec4,
}

impl Default for DebugColors {
    fn default() -> Self {
        Self {
            collision_shape: COLOR_COLLISION_SHAPE,
            contact_point: COLOR_CONTACT_POINT,
            contact_normal: COLOR_CONTACT_NORMAL,
            joint_axis: COLOR_JOINT_AXIS,
            joint_limit: COLOR_JOINT_LIMIT,
            velocity: COLOR_VELOCITY,
            angular_velocity: COLOR_ANGULAR_VELOCITY,
            sleeping: COLOR_SLEEPING,
            broadphase: COLOR_BROADPHASE,
            aabb: COLOR_AABB,
            constraint_error_low: Vec4::new(0.2, 0.8, 0.2, 0.8),
            constraint_error_high: Vec4::new(1.0, 0.1, 0.1, 0.9),
        }
    }
}

/// Settings controlling which debug categories are rendered.
#[derive(Debug, Clone)]
pub struct PhysicsDebugSettings {
    /// Master enable for all debug rendering.
    pub enabled: bool,
    /// Draw collision shape wireframes.
    pub draw_collision_shapes: bool,
    /// Draw contact points.
    pub draw_contact_points: bool,
    /// Draw contact normals.
    pub draw_contact_normals: bool,
    /// Draw joint axes.
    pub draw_joint_axes: bool,
    /// Draw joint limits.
    pub draw_joint_limits: bool,
    /// Draw velocity arrows.
    pub draw_velocity: bool,
    /// Draw angular velocity indicators.
    pub draw_angular_velocity: bool,
    /// Draw sleeping body indicators.
    pub draw_sleeping: bool,
    /// Draw broadphase grid.
    pub draw_broadphase: bool,
    /// Draw AABBs.
    pub draw_aabbs: bool,
    /// Draw constraint error indicators.
    pub draw_constraint_errors: bool,
    /// Draw center of mass markers.
    pub draw_center_of_mass: bool,
    /// Draw body axes (local coordinate frames).
    pub draw_body_axes: bool,
    /// Draw body IDs as text labels.
    pub draw_body_ids: bool,
    /// Minimum velocity magnitude to draw arrows (avoids clutter for near-stationary).
    pub min_velocity_display: f32,
    /// Maximum number of debug primitives to generate per frame (performance limit).
    pub max_primitives: usize,
    /// Color overrides.
    pub colors: DebugColors,
    /// Scale factor for all debug visualizations.
    pub scale: f32,
}

impl Default for PhysicsDebugSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            draw_collision_shapes: true,
            draw_contact_points: true,
            draw_contact_normals: true,
            draw_joint_axes: true,
            draw_joint_limits: true,
            draw_velocity: true,
            draw_angular_velocity: true,
            draw_sleeping: true,
            draw_broadphase: false,
            draw_aabbs: false,
            draw_constraint_errors: false,
            draw_center_of_mass: false,
            draw_body_axes: false,
            draw_body_ids: false,
            min_velocity_display: 0.1,
            max_primitives: 50_000,
            colors: DebugColors::default(),
            scale: 1.0,
        }
    }
}

impl PhysicsDebugSettings {
    /// Enable all debug categories.
    pub fn enable_all(&mut self) {
        self.enabled = true;
        self.draw_collision_shapes = true;
        self.draw_contact_points = true;
        self.draw_contact_normals = true;
        self.draw_joint_axes = true;
        self.draw_joint_limits = true;
        self.draw_velocity = true;
        self.draw_angular_velocity = true;
        self.draw_sleeping = true;
        self.draw_broadphase = true;
        self.draw_aabbs = true;
        self.draw_constraint_errors = true;
        self.draw_center_of_mass = true;
        self.draw_body_axes = true;
        self.draw_body_ids = true;
    }

    /// Disable all debug categories.
    pub fn disable_all(&mut self) {
        self.draw_collision_shapes = false;
        self.draw_contact_points = false;
        self.draw_contact_normals = false;
        self.draw_joint_axes = false;
        self.draw_joint_limits = false;
        self.draw_velocity = false;
        self.draw_angular_velocity = false;
        self.draw_sleeping = false;
        self.draw_broadphase = false;
        self.draw_aabbs = false;
        self.draw_constraint_errors = false;
        self.draw_center_of_mass = false;
        self.draw_body_axes = false;
        self.draw_body_ids = false;
    }

    /// Enable only collision-related categories.
    pub fn collision_only(&mut self) {
        self.disable_all();
        self.enabled = true;
        self.draw_collision_shapes = true;
        self.draw_contact_points = true;
        self.draw_contact_normals = true;
        self.draw_aabbs = true;
    }

    /// Enable only joint-related categories.
    pub fn joints_only(&mut self) {
        self.disable_all();
        self.enabled = true;
        self.draw_joint_axes = true;
        self.draw_joint_limits = true;
    }

    /// Enable only dynamics-related categories.
    pub fn dynamics_only(&mut self) {
        self.disable_all();
        self.enabled = true;
        self.draw_velocity = true;
        self.draw_angular_velocity = true;
        self.draw_sleeping = true;
        self.draw_center_of_mass = true;
    }
}

// ---------------------------------------------------------------------------
// Input data structures (physics state to visualize)
// ---------------------------------------------------------------------------

/// Collision shape data for debug rendering.
#[derive(Debug, Clone)]
pub enum DebugCollisionShape {
    /// Sphere.
    Sphere {
        center: Vec3,
        radius: f32,
    },
    /// Oriented box.
    Box {
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
    },
    /// Capsule.
    Capsule {
        start: Vec3,
        end: Vec3,
        radius: f32,
    },
    /// Convex hull (set of vertices).
    ConvexHull {
        center: Vec3,
        vertices: Vec<Vec3>,
    },
    /// Triangle mesh (wireframe).
    TriMesh {
        vertices: Vec<Vec3>,
        indices: Vec<u32>,
    },
    /// Heightfield.
    Heightfield {
        origin: Vec3,
        cell_size: f32,
        heights: Vec<Vec<f32>>,
    },
}

/// A contact point from the physics simulation.
#[derive(Debug, Clone)]
pub struct DebugContactPoint {
    /// World-space position of the contact.
    pub position: Vec3,
    /// Contact normal (from body A to body B).
    pub normal: Vec3,
    /// Penetration depth (positive = penetrating).
    pub depth: f32,
    /// Impulse applied at this contact.
    pub impulse: f32,
}

/// Joint debug information.
#[derive(Debug, Clone)]
pub struct DebugJointInfo {
    /// Anchor point on body A (world space).
    pub anchor_a: Vec3,
    /// Anchor point on body B (world space).
    pub anchor_b: Vec3,
    /// Joint axis in world space.
    pub axis: Vec3,
    /// Current angle (for hinge joints).
    pub angle: Option<f32>,
    /// Min angle limit (for hinge joints).
    pub min_limit: Option<f32>,
    /// Max angle limit (for hinge joints).
    pub max_limit: Option<f32>,
    /// Secondary axis (for cone-twist joints).
    pub secondary_axis: Option<Vec3>,
    /// Joint type name for labeling.
    pub joint_type: String,
    /// Current constraint error.
    pub error: f32,
}

/// Body debug information.
#[derive(Debug, Clone)]
pub struct DebugBodyInfo {
    /// Body ID.
    pub id: u64,
    /// World-space position (center of mass).
    pub position: Vec3,
    /// Orientation.
    pub rotation: Quat,
    /// Linear velocity.
    pub velocity: Vec3,
    /// Angular velocity.
    pub angular_velocity: Vec3,
    /// Whether the body is sleeping.
    pub sleeping: bool,
    /// Whether the body is static.
    pub is_static: bool,
    /// AABB min.
    pub aabb_min: Vec3,
    /// AABB max.
    pub aabb_max: Vec3,
    /// Mass (0 for static).
    pub mass: f32,
}

/// Broadphase grid information.
#[derive(Debug, Clone)]
pub struct DebugBroadphaseGrid {
    /// Origin of the grid.
    pub origin: Vec3,
    /// Cell size.
    pub cell_size: f32,
    /// Active cells (min corner of each cell).
    pub active_cells: Vec<Vec3>,
    /// Number of pairs in each cell.
    pub cell_pair_counts: Vec<usize>,
}

/// Constraint error information.
#[derive(Debug, Clone)]
pub struct DebugConstraintError {
    /// Position to display the error at.
    pub position: Vec3,
    /// Error magnitude.
    pub error: f32,
    /// Maximum expected error (for normalization).
    pub max_expected_error: f32,
    /// Constraint type name.
    pub constraint_type: String,
}

// ---------------------------------------------------------------------------
// Physics debug renderer
// ---------------------------------------------------------------------------

/// The main physics debug renderer that converts physics state into
/// debug draw primitives.
#[derive(Debug)]
pub struct PhysicsDebugRenderer {
    /// Debug rendering settings.
    pub settings: PhysicsDebugSettings,
    /// Generated primitives for the current frame.
    primitives: Vec<DebugPrimitive>,
    /// Statistics from the last frame.
    pub stats: DebugRenderStats,
}

/// Statistics about debug rendering.
#[derive(Debug, Clone, Default)]
pub struct DebugRenderStats {
    /// Total primitives generated.
    pub total_primitives: usize,
    /// Number of collision shape primitives.
    pub shape_primitives: usize,
    /// Number of contact primitives.
    pub contact_primitives: usize,
    /// Number of joint primitives.
    pub joint_primitives: usize,
    /// Number of body primitives.
    pub body_primitives: usize,
    /// Number of broadphase primitives.
    pub broadphase_primitives: usize,
    /// Whether the primitive limit was hit.
    pub limit_reached: bool,
}

impl PhysicsDebugRenderer {
    /// Create a new debug renderer with default settings.
    pub fn new() -> Self {
        Self {
            settings: PhysicsDebugSettings::default(),
            primitives: Vec::new(),
            stats: DebugRenderStats::default(),
        }
    }

    /// Create with custom settings.
    pub fn with_settings(settings: PhysicsDebugSettings) -> Self {
        Self {
            settings,
            primitives: Vec::new(),
            stats: DebugRenderStats::default(),
        }
    }

    /// Check if we can add more primitives.
    fn can_add(&self) -> bool {
        self.primitives.len() < self.settings.max_primitives
    }

    /// Add a primitive to the output buffer.
    fn push(&mut self, primitive: DebugPrimitive) {
        if self.can_add() {
            self.primitives.push(primitive);
        }
    }

    /// Begin a new frame (clear previous primitives).
    pub fn begin_frame(&mut self) {
        self.primitives.clear();
        self.stats = DebugRenderStats::default();
    }

    /// Draw a collision shape.
    pub fn draw_collision_shape(&mut self, shape: &DebugCollisionShape) {
        if !self.settings.enabled || !self.settings.draw_collision_shapes {
            return;
        }
        let color = self.settings.colors.collision_shape;
        let before = self.primitives.len();

        match shape {
            DebugCollisionShape::Sphere { center, radius } => {
                self.push(DebugPrimitive::WireSphere {
                    center: *center,
                    radius: *radius * self.settings.scale,
                    color,
                });
            }
            DebugCollisionShape::Box {
                center,
                half_extents,
                rotation,
            } => {
                self.push(DebugPrimitive::WireBox {
                    center: *center,
                    half_extents: *half_extents * self.settings.scale,
                    rotation: *rotation,
                    color,
                });
            }
            DebugCollisionShape::Capsule { start, end, radius } => {
                self.push(DebugPrimitive::WireCapsule {
                    start: *start,
                    end: *end,
                    radius: *radius * self.settings.scale,
                    color,
                });
            }
            DebugCollisionShape::ConvexHull { vertices, .. } => {
                // Draw edges of the convex hull
                for i in 0..vertices.len() {
                    for j in (i + 1)..vertices.len() {
                        if !self.can_add() {
                            break;
                        }
                        self.push(DebugPrimitive::Line {
                            start: vertices[i],
                            end: vertices[j],
                            color,
                        });
                    }
                }
            }
            DebugCollisionShape::TriMesh { vertices, indices } => {
                let tri_count = indices.len() / 3;
                for t in 0..tri_count {
                    if !self.can_add() {
                        break;
                    }
                    let i0 = indices[t * 3] as usize;
                    let i1 = indices[t * 3 + 1] as usize;
                    let i2 = indices[t * 3 + 2] as usize;
                    if i0 < vertices.len() && i1 < vertices.len() && i2 < vertices.len() {
                        self.push(DebugPrimitive::WireTriangle {
                            v0: vertices[i0],
                            v1: vertices[i1],
                            v2: vertices[i2],
                            color,
                        });
                    }
                }
            }
            DebugCollisionShape::Heightfield {
                origin,
                cell_size,
                heights,
            } => {
                let rows = heights.len();
                if rows == 0 {
                    return;
                }
                let cols = heights[0].len();

                for r in 0..rows {
                    for c in 0..cols {
                        if !self.can_add() {
                            break;
                        }
                        let pos = *origin
                            + Vec3::new(c as f32 * cell_size, heights[r][c], r as f32 * cell_size);
                        // Draw lines to right and down neighbors
                        if c + 1 < cols {
                            let next = *origin
                                + Vec3::new(
                                    (c + 1) as f32 * cell_size,
                                    heights[r][c + 1],
                                    r as f32 * cell_size,
                                );
                            self.push(DebugPrimitive::Line {
                                start: pos,
                                end: next,
                                color,
                            });
                        }
                        if r + 1 < rows {
                            let next = *origin
                                + Vec3::new(
                                    c as f32 * cell_size,
                                    heights[r + 1][c],
                                    (r + 1) as f32 * cell_size,
                                );
                            self.push(DebugPrimitive::Line {
                                start: pos,
                                end: next,
                                color,
                            });
                        }
                    }
                }
            }
        }

        self.stats.shape_primitives += self.primitives.len() - before;
    }

    /// Draw a contact point with optional normal.
    pub fn draw_contact(&mut self, contact: &DebugContactPoint) {
        if !self.settings.enabled {
            return;
        }
        let before = self.primitives.len();

        if self.settings.draw_contact_points {
            self.push(DebugPrimitive::Point {
                position: contact.position,
                radius: CONTACT_POINT_RADIUS * self.settings.scale,
                color: self.settings.colors.contact_point,
            });
        }

        if self.settings.draw_contact_normals {
            let normal_end =
                contact.position + contact.normal * CONTACT_NORMAL_LENGTH * self.settings.scale;
            self.push(DebugPrimitive::Arrow {
                start: contact.position,
                end: normal_end,
                color: self.settings.colors.contact_normal,
                head_size: 0.02 * self.settings.scale,
            });

            // Draw penetration depth indicator
            if contact.depth > 0.0 {
                let pen_end = contact.position - contact.normal * contact.depth;
                let depth_factor = (contact.depth / 0.1).clamp(0.0, 1.0);
                let pen_color = Vec4::new(1.0, 1.0 - depth_factor, 0.0, 0.8);
                self.push(DebugPrimitive::Line {
                    start: contact.position,
                    end: pen_end,
                    color: pen_color,
                });
            }
        }

        self.stats.contact_primitives += self.primitives.len() - before;
    }

    /// Draw joint debug information.
    pub fn draw_joint(&mut self, joint: &DebugJointInfo) {
        if !self.settings.enabled {
            return;
        }
        let before = self.primitives.len();

        // Draw line between anchors
        self.push(DebugPrimitive::Line {
            start: joint.anchor_a,
            end: joint.anchor_b,
            color: self.settings.colors.joint_axis,
        });

        // Draw anchor points
        self.push(DebugPrimitive::Point {
            position: joint.anchor_a,
            radius: 0.03 * self.settings.scale,
            color: self.settings.colors.joint_axis,
        });
        self.push(DebugPrimitive::Point {
            position: joint.anchor_b,
            radius: 0.03 * self.settings.scale,
            color: self.settings.colors.joint_axis,
        });

        // Draw joint axis
        if self.settings.draw_joint_axes {
            let axis_length = 0.2 * self.settings.scale;
            let mid = (joint.anchor_a + joint.anchor_b) * 0.5;
            self.push(DebugPrimitive::Arrow {
                start: mid,
                end: mid + joint.axis * axis_length,
                color: self.settings.colors.joint_axis,
                head_size: 0.02 * self.settings.scale,
            });
        }

        // Draw joint limits (arc around the axis)
        if self.settings.draw_joint_limits {
            if let (Some(min_limit), Some(max_limit)) = (joint.min_limit, joint.max_limit) {
                let mid = (joint.anchor_a + joint.anchor_b) * 0.5;
                let limit_radius = 0.15 * self.settings.scale;

                self.push(DebugPrimitive::Arc {
                    center: mid,
                    normal: joint.axis,
                    radius: limit_radius,
                    start_angle: min_limit,
                    end_angle: max_limit,
                    color: self.settings.colors.joint_limit,
                });

                // Draw current angle indicator
                if let Some(angle) = joint.angle {
                    let perp = if joint.axis.y.abs() < 0.999 {
                        Vec3::Y.cross(joint.axis).normalize_or_zero()
                    } else {
                        Vec3::X.cross(joint.axis).normalize_or_zero()
                    };
                    let indicator_dir = Quat::from_axis_angle(joint.axis, angle) * perp;
                    self.push(DebugPrimitive::Line {
                        start: mid,
                        end: mid + indicator_dir * limit_radius * 1.2,
                        color: Vec4::new(1.0, 1.0, 1.0, 0.9),
                    });
                }
            }
        }

        // Draw constraint error
        if self.settings.draw_constraint_errors && joint.error > EPSILON {
            let error_normalized = (joint.error / 0.01).clamp(0.0, 1.0);
            let error_color = lerp_color(
                self.settings.colors.constraint_error_low,
                self.settings.colors.constraint_error_high,
                error_normalized,
            );
            let mid = (joint.anchor_a + joint.anchor_b) * 0.5;
            self.push(DebugPrimitive::Text {
                position: mid + Vec3::Y * 0.1,
                text: format!("err: {:.4}", joint.error),
                color: error_color,
            });
        }

        self.stats.joint_primitives += self.primitives.len() - before;
    }

    /// Draw body debug information (velocity, sleeping, etc.).
    pub fn draw_body(&mut self, body: &DebugBodyInfo) {
        if !self.settings.enabled {
            return;
        }
        let before = self.primitives.len();

        // Sleeping indicator
        if self.settings.draw_sleeping && body.sleeping {
            self.push(DebugPrimitive::FilledCircle {
                center: body.position + Vec3::Y * 0.3,
                normal: Vec3::Y,
                radius: 0.05 * self.settings.scale,
                color: self.settings.colors.sleeping,
            });
            self.push(DebugPrimitive::Text {
                position: body.position + Vec3::Y * 0.4,
                text: "Zzz".to_string(),
                color: self.settings.colors.sleeping,
            });
        }

        // Velocity arrow
        if self.settings.draw_velocity && !body.is_static {
            let speed = body.velocity.length();
            if speed > self.settings.min_velocity_display {
                let arrow_end =
                    body.position + body.velocity * VELOCITY_ARROW_SCALE * self.settings.scale;
                self.push(DebugPrimitive::Arrow {
                    start: body.position,
                    end: arrow_end,
                    color: self.settings.colors.velocity,
                    head_size: 0.03 * self.settings.scale,
                });
            }
        }

        // Angular velocity indicator
        if self.settings.draw_angular_velocity && !body.is_static {
            let ang_speed = body.angular_velocity.length();
            if ang_speed > self.settings.min_velocity_display {
                let ang_dir = body.angular_velocity.normalize_or_zero();
                let arrow_end = body.position
                    + ang_dir * ang_speed * ANGULAR_VELOCITY_ARROW_SCALE * self.settings.scale;
                self.push(DebugPrimitive::Arrow {
                    start: body.position,
                    end: arrow_end,
                    color: self.settings.colors.angular_velocity,
                    head_size: 0.02 * self.settings.scale,
                });

                // Draw rotation circle
                self.push(DebugPrimitive::Arc {
                    center: body.position,
                    normal: ang_dir,
                    radius: 0.15 * self.settings.scale,
                    start_angle: 0.0,
                    end_angle: (ang_speed * 0.5).min(std::f32::consts::PI * 2.0),
                    color: self.settings.colors.angular_velocity,
                });
            }
        }

        // AABB
        if self.settings.draw_aabbs && !body.is_static {
            self.push(DebugPrimitive::WireAABB {
                min: body.aabb_min,
                max: body.aabb_max,
                color: self.settings.colors.aabb,
            });
        }

        // Center of mass
        if self.settings.draw_center_of_mass {
            let com_color = Vec4::new(1.0, 1.0, 0.0, 1.0);
            self.push(DebugPrimitive::Point {
                position: body.position,
                radius: 0.03 * self.settings.scale,
                color: com_color,
            });
        }

        // Body axes
        if self.settings.draw_body_axes && !body.is_static {
            let axis_length = 0.15 * self.settings.scale;
            let x_axis = body.rotation * Vec3::X;
            let y_axis = body.rotation * Vec3::Y;
            let z_axis = body.rotation * Vec3::Z;

            self.push(DebugPrimitive::Arrow {
                start: body.position,
                end: body.position + x_axis * axis_length,
                color: Vec4::new(1.0, 0.0, 0.0, 0.9),
                head_size: 0.01 * self.settings.scale,
            });
            self.push(DebugPrimitive::Arrow {
                start: body.position,
                end: body.position + y_axis * axis_length,
                color: Vec4::new(0.0, 1.0, 0.0, 0.9),
                head_size: 0.01 * self.settings.scale,
            });
            self.push(DebugPrimitive::Arrow {
                start: body.position,
                end: body.position + z_axis * axis_length,
                color: Vec4::new(0.0, 0.0, 1.0, 0.9),
                head_size: 0.01 * self.settings.scale,
            });
        }

        // Body ID label
        if self.settings.draw_body_ids {
            self.push(DebugPrimitive::Text {
                position: body.position + Vec3::Y * 0.2,
                text: format!("#{} m={:.1}", body.id, body.mass),
                color: Vec4::new(1.0, 1.0, 1.0, 0.7),
            });
        }

        self.stats.body_primitives += self.primitives.len() - before;
    }

    /// Draw broadphase grid.
    pub fn draw_broadphase(&mut self, grid: &DebugBroadphaseGrid) {
        if !self.settings.enabled || !self.settings.draw_broadphase {
            return;
        }
        let before = self.primitives.len();
        let color = self.settings.colors.broadphase;

        for (i, cell_min) in grid.active_cells.iter().enumerate() {
            if !self.can_add() {
                break;
            }
            let cell_max = *cell_min + Vec3::splat(grid.cell_size);

            // Color intensity based on pair count
            let intensity = if i < grid.cell_pair_counts.len() {
                (grid.cell_pair_counts[i] as f32 / 10.0).clamp(0.1, 1.0)
            } else {
                0.3
            };

            let cell_color = Vec4::new(
                color.x * intensity,
                color.y * intensity,
                color.z * intensity,
                color.w,
            );

            self.push(DebugPrimitive::WireAABB {
                min: *cell_min,
                max: cell_max,
                color: cell_color,
            });

            // Show pair count
            if self.settings.draw_body_ids {
                if let Some(&count) = grid.cell_pair_counts.get(i) {
                    if count > 0 {
                        let center = (*cell_min + cell_max) * 0.5;
                        self.push(DebugPrimitive::Text {
                            position: center,
                            text: format!("{}", count),
                            color: cell_color,
                        });
                    }
                }
            }
        }

        self.stats.broadphase_primitives += self.primitives.len() - before;
    }

    /// Draw constraint error indicators.
    pub fn draw_constraint_error(&mut self, error: &DebugConstraintError) {
        if !self.settings.enabled || !self.settings.draw_constraint_errors {
            return;
        }

        let normalized = (error.error / error.max_expected_error.max(EPSILON)).clamp(0.0, 1.0);
        let color = lerp_color(
            self.settings.colors.constraint_error_low,
            self.settings.colors.constraint_error_high,
            normalized,
        );

        // Visual indicator size based on error magnitude
        let indicator_size = 0.02 + 0.08 * normalized;

        self.push(DebugPrimitive::Point {
            position: error.position,
            radius: indicator_size * self.settings.scale,
            color,
        });

        self.push(DebugPrimitive::Text {
            position: error.position + Vec3::Y * (indicator_size + 0.05),
            text: format!("{}: {:.4}", error.constraint_type, error.error),
            color,
        });
    }

    /// Finalize the frame and return the generated primitives.
    pub fn end_frame(&mut self) -> &[DebugPrimitive] {
        self.stats.total_primitives = self.primitives.len();
        self.stats.limit_reached = self.primitives.len() >= self.settings.max_primitives;
        &self.primitives
    }

    /// Get the primitives generated so far.
    pub fn primitives(&self) -> &[DebugPrimitive] {
        &self.primitives
    }

    /// Get the number of primitives generated.
    pub fn primitive_count(&self) -> usize {
        self.primitives.len()
    }

    /// Take ownership of the primitives (consuming for rendering).
    pub fn take_primitives(&mut self) -> Vec<DebugPrimitive> {
        std::mem::take(&mut self.primitives)
    }

    /// Convenience: draw all bodies from a slice.
    pub fn draw_bodies(&mut self, bodies: &[DebugBodyInfo]) {
        for body in bodies {
            self.draw_body(body);
        }
    }

    /// Convenience: draw all contacts from a slice.
    pub fn draw_contacts(&mut self, contacts: &[DebugContactPoint]) {
        for contact in contacts {
            self.draw_contact(contact);
        }
    }

    /// Convenience: draw all joints from a slice.
    pub fn draw_joints(&mut self, joints: &[DebugJointInfo]) {
        for joint in joints {
            self.draw_joint(joint);
        }
    }

    /// Convenience: draw all shapes from a slice.
    pub fn draw_shapes(&mut self, shapes: &[DebugCollisionShape]) {
        for shape in shapes {
            self.draw_collision_shape(shape);
        }
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Linearly interpolate between two colors.
fn lerp_color(a: Vec4, b: Vec4, t: f32) -> Vec4 {
    a + (b - a) * t
}

/// Generate wireframe points for a circle.
pub fn generate_circle_points(
    center: Vec3,
    normal: Vec3,
    radius: f32,
    segments: usize,
) -> Vec<Vec3> {
    let segments = segments.max(3);
    let normal = normal.normalize_or_zero();
    let up = if normal.y.abs() < 0.999 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let right = normal.cross(up).normalize_or_zero();
    let forward = right.cross(normal).normalize_or_zero();

    let mut points = Vec::with_capacity(segments);
    for i in 0..segments {
        let angle = (i as f32 / segments as f32) * 2.0 * std::f32::consts::PI;
        let point = center + (right * angle.cos() + forward * angle.sin()) * radius;
        points.push(point);
    }
    points
}

/// Generate wireframe lines for a box.
pub fn generate_box_lines(
    center: Vec3,
    half_extents: Vec3,
    rotation: Quat,
) -> Vec<(Vec3, Vec3)> {
    let corners = [
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
    ];

    let world_corners: Vec<Vec3> = corners
        .iter()
        .map(|c| center + rotation * (*c * half_extents))
        .collect();

    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), // front face
        (4, 5), (5, 6), (6, 7), (7, 4), // back face
        (0, 4), (1, 5), (2, 6), (3, 7), // connecting edges
    ];

    edges
        .iter()
        .map(|&(a, b)| (world_corners[a], world_corners[b]))
        .collect()
}

/// Convert debug primitives to a flat list of line segments for simple renderers.
pub fn primitives_to_lines(primitives: &[DebugPrimitive]) -> Vec<(Vec3, Vec3, Vec4)> {
    let mut lines = Vec::new();

    for prim in primitives {
        match prim {
            DebugPrimitive::Line { start, end, color } => {
                lines.push((*start, *end, *color));
            }
            DebugPrimitive::Arrow {
                start,
                end,
                color,
                head_size,
            } => {
                lines.push((*start, *end, *color));
                // Simple arrowhead
                let dir = (*end - *start).normalize_or_zero();
                let perp = if dir.y.abs() < 0.999 {
                    Vec3::Y.cross(dir).normalize_or_zero()
                } else {
                    Vec3::X.cross(dir).normalize_or_zero()
                };
                let h1 = *end - dir * *head_size + perp * *head_size * 0.5;
                let h2 = *end - dir * *head_size - perp * *head_size * 0.5;
                lines.push((*end, h1, *color));
                lines.push((*end, h2, *color));
            }
            DebugPrimitive::WireAABB { min, max, color } => {
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
                let edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ];
                for (a, b) in edges {
                    lines.push((corners[a], corners[b], *color));
                }
            }
            DebugPrimitive::WireTriangle { v0, v1, v2, color } => {
                lines.push((*v0, *v1, *color));
                lines.push((*v1, *v2, *color));
                lines.push((*v2, *v0, *color));
            }
            _ => {} // Other primitives need more complex handling
        }
    }

    lines
}
