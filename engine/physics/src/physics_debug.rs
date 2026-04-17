// engine/physics/src/physics_debug_v2.rs
//
// Physics debug rendering v2: contact points with normals, constraint limits,
// body velocity arrows, sleep state indicators, broadphase cell visualization.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }
impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn add(self, r: Self) -> Self { Self { x:self.x+r.x, y:self.y+r.y, z:self.z+r.z } }
    pub fn sub(self, r: Self) -> Self { Self { x:self.x-r.x, y:self.y-r.y, z:self.z-r.z } }
    pub fn scale(self, s: f32) -> Self { Self { x:self.x*s, y:self.y*s, z:self.z*s } }
    pub fn length(self) -> f32 { (self.x*self.x+self.y*self.y+self.z*self.z).sqrt() }
    pub fn normalize(self) -> Self { let l=self.length(); if l<1e-12{Self::ZERO}else{self.scale(1.0/l)} }
    pub fn cross(self, r: Self) -> Self { Self{x:self.y*r.z-self.z*r.y,y:self.z*r.x-self.x*r.z,z:self.x*r.y-self.y*r.x} }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x+self.y*r.y+self.z*r.z }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color { pub r: f32, pub g: f32, pub b: f32, pub a: f32 }
impl Color {
    pub const RED: Self = Self{r:1.0,g:0.0,b:0.0,a:1.0};
    pub const GREEN: Self = Self{r:0.0,g:1.0,b:0.0,a:1.0};
    pub const BLUE: Self = Self{r:0.0,g:0.0,b:1.0,a:1.0};
    pub const YELLOW: Self = Self{r:1.0,g:1.0,b:0.0,a:1.0};
    pub const CYAN: Self = Self{r:0.0,g:1.0,b:1.0,a:1.0};
    pub const MAGENTA: Self = Self{r:1.0,g:0.0,b:1.0,a:1.0};
    pub const WHITE: Self = Self{r:1.0,g:1.0,b:1.0,a:1.0};
    pub const GRAY: Self = Self{r:0.5,g:0.5,b:0.5,a:1.0};
    pub const ORANGE: Self = Self{r:1.0,g:0.5,b:0.0,a:1.0};
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self{r,g,b,a} }
    pub fn with_alpha(self, a: f32) -> Self { Self{a,..self} }
}

#[derive(Debug, Clone)]
pub struct DebugLine { pub start: Vec3, pub end: Vec3, pub color: Color }
#[derive(Debug, Clone)]
pub struct DebugPoint { pub position: Vec3, pub color: Color, pub size: f32 }
#[derive(Debug, Clone)]
pub struct DebugText { pub position: Vec3, pub text: String, pub color: Color }

/// Settings for physics debug visualization.
#[derive(Debug, Clone)]
pub struct PhysicsDebugSettingsV2 {
    pub draw_collision_shapes: bool,
    pub draw_contact_points: bool,
    pub draw_contact_normals: bool,
    pub draw_contact_impulses: bool,
    pub draw_velocity_arrows: bool,
    pub draw_angular_velocity: bool,
    pub draw_center_of_mass: bool,
    pub draw_aabbs: bool,
    pub draw_broadphase_grid: bool,
    pub draw_joint_axes: bool,
    pub draw_joint_limits: bool,
    pub draw_sleep_state: bool,
    pub draw_island_colors: bool,
    pub draw_body_ids: bool,
    pub draw_constraint_errors: bool,
    pub contact_normal_length: f32,
    pub velocity_arrow_scale: f32,
    pub impulse_arrow_scale: f32,
    pub active_body_color: Color,
    pub sleeping_body_color: Color,
    pub static_body_color: Color,
    pub kinematic_body_color: Color,
    pub contact_point_color: Color,
    pub contact_normal_color: Color,
    pub joint_color: Color,
    pub aabb_color: Color,
    pub broadphase_color: Color,
    pub constraint_error_threshold: f32,
}

impl Default for PhysicsDebugSettingsV2 {
    fn default() -> Self {
        Self {
            draw_collision_shapes: true, draw_contact_points: true,
            draw_contact_normals: true, draw_contact_impulses: false,
            draw_velocity_arrows: false, draw_angular_velocity: false,
            draw_center_of_mass: false, draw_aabbs: false,
            draw_broadphase_grid: false, draw_joint_axes: true,
            draw_joint_limits: true, draw_sleep_state: true,
            draw_island_colors: false, draw_body_ids: false,
            draw_constraint_errors: false,
            contact_normal_length: 0.3, velocity_arrow_scale: 0.1,
            impulse_arrow_scale: 0.05,
            active_body_color: Color::GREEN, sleeping_body_color: Color::GRAY,
            static_body_color: Color::BLUE, kinematic_body_color: Color::CYAN,
            contact_point_color: Color::RED, contact_normal_color: Color::YELLOW,
            joint_color: Color::MAGENTA,
            aabb_color: Color::new(0.3,0.8,0.3,0.3),
            broadphase_color: Color::new(0.2,0.2,0.8,0.15),
            constraint_error_threshold: 0.01,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyTypeDbg { Static, Dynamic, Kinematic }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepStateDbg { Awake, Sleeping, WantsSleep }

#[derive(Debug, Clone)]
pub enum ShapeDataDbg {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { radius: f32, half_height: f32 },
    Cylinder { radius: f32, half_height: f32 },
    ConvexHull { vertices: Vec<Vec3> },
    TriMesh { vertex_count: u32 },
}

#[derive(Debug, Clone)]
pub struct PhysicsBodyInfoDbg {
    pub id: u32,
    pub body_type: BodyTypeDbg,
    pub position: Vec3,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub center_of_mass: Vec3,
    pub sleep_state: SleepStateDbg,
    pub island_id: u32,
    pub shape: ShapeDataDbg,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
}

#[derive(Debug, Clone)]
pub struct ContactInfoDbg {
    pub position: Vec3,
    pub normal: Vec3,
    pub depth: f32,
    pub normal_impulse: f32,
    pub friction_impulse: f32,
    pub body_a: u32,
    pub body_b: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointTypeDbg { Fixed, Hinge, Ball, Slider, Cone, Distance }

#[derive(Debug, Clone)]
pub struct JointInfoDbg {
    pub joint_type: JointTypeDbg,
    pub body_a: u32,
    pub body_b: u32,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub axis: Vec3,
    pub min_limit: f32,
    pub max_limit: f32,
    pub current_value: f32,
    pub error: f32,
}

#[derive(Debug, Clone)]
pub struct BroadphaseCellDbg {
    pub min: Vec3,
    pub max: Vec3,
    pub body_count: u32,
}

#[derive(Debug, Clone, Default)]
pub struct PhysicsDebugStatsV2 {
    pub bodies_drawn: u32,
    pub contacts_drawn: u32,
    pub joints_drawn: u32,
    pub lines_generated: u32,
    pub points_generated: u32,
    pub texts_generated: u32,
}

/// Physics debug renderer with batched primitives and configurable visualization.
pub struct PhysicsDebugRendererV2 {
    pub settings: PhysicsDebugSettingsV2,
    pub lines: Vec<DebugLine>,
    pub points: Vec<DebugPoint>,
    pub texts: Vec<DebugText>,
    island_colors: Vec<Color>,
    pub stats: PhysicsDebugStatsV2,
}

impl PhysicsDebugRendererV2 {
    pub fn new() -> Self {
        let mut ic = Vec::with_capacity(64);
        for i in 0..64 {
            let h = (i as f32 / 64.0) * 360.0;
            let x = 1.0 - (h / 60.0 % 2.0 - 1.0).abs();
            let (r,g,b) = match (h/60.0) as u32 {
                0=>(1.0,x,0.0), 1=>(x,1.0,0.0), 2=>(0.0,1.0,x),
                3=>(0.0,x,1.0), 4=>(x,0.0,1.0), _=>(1.0,0.0,x),
            };
            ic.push(Color::new(r,g,b,0.8));
        }
        Self {
            settings: PhysicsDebugSettingsV2::default(),
            lines: Vec::with_capacity(65536),
            points: Vec::with_capacity(8192),
            texts: Vec::with_capacity(1024),
            island_colors: ic,
            stats: PhysicsDebugStatsV2::default(),
        }
    }

    pub fn clear(&mut self) {
        self.lines.clear();
        self.points.clear();
        self.texts.clear();
        self.stats = PhysicsDebugStatsV2::default();
    }

    /// Draw all physics bodies with collision shapes, velocities, and state.
    pub fn draw_bodies(&mut self, bodies: &[PhysicsBodyInfoDbg]) {
        for body in bodies {
            let color = match body.body_type {
                BodyTypeDbg::Static => self.settings.static_body_color,
                BodyTypeDbg::Kinematic => self.settings.kinematic_body_color,
                BodyTypeDbg::Dynamic => {
                    if self.settings.draw_island_colors {
                        self.island_colors[body.island_id as usize % self.island_colors.len()]
                    } else {
                        match body.sleep_state {
                            SleepStateDbg::Sleeping => self.settings.sleeping_body_color,
                            SleepStateDbg::WantsSleep => Color::YELLOW,
                            SleepStateDbg::Awake => self.settings.active_body_color,
                        }
                    }
                }
            };

            if self.settings.draw_collision_shapes {
                self.draw_shape_internal(&body.shape, body.position, color);
                self.stats.bodies_drawn += 1;
            }

            if self.settings.draw_aabbs {
                self.draw_wire_box_mm(body.aabb_min, body.aabb_max, self.settings.aabb_color);
            }

            if self.settings.draw_velocity_arrows && body.body_type == BodyTypeDbg::Dynamic {
                let end = body.position.add(body.linear_velocity.scale(self.settings.velocity_arrow_scale));
                self.draw_arrow_internal(body.position, end, Color::CYAN);
            }

            if self.settings.draw_angular_velocity && body.body_type == BodyTypeDbg::Dynamic {
                let end = body.position.add(body.angular_velocity.scale(self.settings.velocity_arrow_scale));
                self.draw_arrow_internal(body.position, end, Color::MAGENTA);
            }

            if self.settings.draw_center_of_mass {
                self.draw_cross_internal(body.center_of_mass, 0.05, Color::WHITE);
            }

            if self.settings.draw_sleep_state && body.sleep_state == SleepStateDbg::Sleeping {
                self.texts.push(DebugText {
                    position: body.position.add(Vec3::new(0.0, 0.3, 0.0)),
                    text: "Zzz".into(),
                    color: Color::GRAY,
                });
            }

            if self.settings.draw_body_ids {
                self.texts.push(DebugText {
                    position: body.position.add(Vec3::new(0.0, 0.2, 0.0)),
                    text: format!("#{}", body.id),
                    color: Color::WHITE,
                });
            }
        }
    }

    /// Draw contact points with normals and impulse visualization.
    pub fn draw_contacts(&mut self, contacts: &[ContactInfoDbg]) {
        for c in contacts {
            if self.settings.draw_contact_points {
                self.points.push(DebugPoint {
                    position: c.position,
                    color: self.settings.contact_point_color,
                    size: 4.0,
                });
                self.stats.contacts_drawn += 1;
            }

            if self.settings.draw_contact_normals {
                let end = c.position.add(c.normal.scale(self.settings.contact_normal_length));
                self.draw_arrow_internal(c.position, end, self.settings.contact_normal_color);
            }

            if self.settings.draw_contact_impulses && c.normal_impulse > 0.01 {
                let end = c.position.add(c.normal.scale(c.normal_impulse * self.settings.impulse_arrow_scale));
                self.draw_arrow_internal(c.position, end, Color::ORANGE);
            }
        }
    }

    /// Draw joint constraints with axes, limits, and error visualization.
    pub fn draw_joints(&mut self, joints: &[JointInfoDbg]) {
        for j in joints {
            let c = self.settings.joint_color;
            self.lines.push(DebugLine { start: j.anchor_a, end: j.anchor_b, color: c });
            self.stats.joints_drawn += 1;

            if self.settings.draw_joint_axes {
                let end = j.anchor_a.add(j.axis.scale(0.2));
                self.draw_arrow_internal(j.anchor_a, end, Color::CYAN);
            }

            if self.settings.draw_joint_limits {
                match j.joint_type {
                    JointTypeDbg::Hinge => {
                        self.draw_arc_internal(j.anchor_a, j.axis, j.min_limit, j.max_limit, 0.15, Color::YELLOW);
                        // Current angle indicator
                        self.draw_arc_internal(j.anchor_a, j.axis, j.current_value - 0.05, j.current_value + 0.05, 0.18, Color::GREEN);
                    }
                    JointTypeDbg::Slider => {
                        let mn = j.anchor_a.add(j.axis.scale(j.min_limit));
                        let mx = j.anchor_a.add(j.axis.scale(j.max_limit));
                        self.lines.push(DebugLine { start: mn, end: mx, color: Color::YELLOW });
                        let cur = j.anchor_a.add(j.axis.scale(j.current_value));
                        self.draw_cross_internal(cur, 0.03, Color::GREEN);
                    }
                    JointTypeDbg::Distance => {
                        self.draw_wire_sphere_internal(j.anchor_a, j.min_limit, Color::YELLOW.with_alpha(0.3));
                    }
                    _ => {}
                }
            }

            if self.settings.draw_constraint_errors && j.error > self.settings.constraint_error_threshold {
                let ec = Color::new(1.0, 0.0, 0.0, (j.error * 10.0).min(1.0));
                self.draw_cross_internal(j.anchor_a, 0.05, ec);
                self.texts.push(DebugText {
                    position: j.anchor_a.add(Vec3::new(0.0, 0.15, 0.0)),
                    text: format!("err:{:.3}", j.error),
                    color: ec,
                });
            }
        }
    }

    /// Draw broadphase grid cells colored by occupancy.
    pub fn draw_broadphase(&mut self, cells: &[BroadphaseCellDbg]) {
        if !self.settings.draw_broadphase_grid { return; }
        for cell in cells {
            if cell.body_count > 0 {
                let intensity = (cell.body_count as f32 / 10.0).min(1.0);
                let c = Color::new(intensity, 0.2, 1.0 - intensity, 0.1 + intensity * 0.2);
                self.draw_wire_box_mm(cell.min, cell.max, c);
            }
        }
    }

    // ------- Internal drawing helpers -------

    fn draw_shape_internal(&mut self, shape: &ShapeDataDbg, pos: Vec3, color: Color) {
        match shape {
            ShapeDataDbg::Sphere { radius } => self.draw_wire_sphere_internal(pos, *radius, color),
            ShapeDataDbg::Box { half_extents } => {
                let h = *half_extents;
                self.draw_wire_box_mm(pos.sub(h), pos.add(h), color);
            }
            ShapeDataDbg::Capsule { radius, half_height } => {
                let top = pos.add(Vec3::new(0.0, *half_height, 0.0));
                let bot = pos.sub(Vec3::new(0.0, *half_height, 0.0));
                self.draw_wire_sphere_internal(top, *radius, color);
                self.draw_wire_sphere_internal(bot, *radius, color);
                for i in 0..4 {
                    let a = i as f32 * std::f32::consts::FRAC_PI_2;
                    let o = Vec3::new(a.cos() * radius, 0.0, a.sin() * radius);
                    self.lines.push(DebugLine { start: top.add(o), end: bot.add(o), color });
                }
            }
            ShapeDataDbg::Cylinder { radius, half_height } => {
                let top = pos.add(Vec3::new(0.0, *half_height, 0.0));
                let bot = pos.sub(Vec3::new(0.0, *half_height, 0.0));
                self.draw_circle_internal(top, Vec3::new(0.0,1.0,0.0), *radius, color, 16);
                self.draw_circle_internal(bot, Vec3::new(0.0,1.0,0.0), *radius, color, 16);
                for i in 0..4 {
                    let a = i as f32 * std::f32::consts::FRAC_PI_2;
                    let o = Vec3::new(a.cos() * radius, 0.0, a.sin() * radius);
                    self.lines.push(DebugLine { start: top.add(o), end: bot.add(o), color });
                }
            }
            ShapeDataDbg::ConvexHull { vertices } => {
                // Draw edges of convex hull (simplified: draw nearest-neighbor pairs)
                for i in 0..vertices.len().min(32) {
                    for j in (i+1)..vertices.len().min(32) {
                        if vertices[i].sub(vertices[j]).length() < 2.0 {
                            self.lines.push(DebugLine {
                                start: pos.add(vertices[i]),
                                end: pos.add(vertices[j]),
                                color,
                            });
                        }
                    }
                }
            }
            ShapeDataDbg::TriMesh { .. } => {
                self.draw_cross_internal(pos, 0.1, color);
            }
        }
    }

    fn draw_wire_sphere_internal(&mut self, center: Vec3, radius: f32, color: Color) {
        self.draw_circle_internal(center, Vec3::new(0.0,1.0,0.0), radius, color, 16);
        self.draw_circle_internal(center, Vec3::new(1.0,0.0,0.0), radius, color, 16);
        self.draw_circle_internal(center, Vec3::new(0.0,0.0,1.0), radius, color, 16);
    }

    fn draw_circle_internal(&mut self, center: Vec3, normal: Vec3, radius: f32, color: Color, segs: u32) {
        let up = if normal.dot(Vec3::new(0.0,1.0,0.0)).abs() > 0.99 {
            Vec3::new(1.0,0.0,0.0)
        } else {
            Vec3::new(0.0,1.0,0.0)
        };
        let right = normal.cross(up).normalize();
        let fwd = right.cross(normal).normalize();
        let step = std::f32::consts::TAU / segs as f32;
        for i in 0..segs {
            let a1 = i as f32 * step;
            let a2 = (i + 1) as f32 * step;
            let p1 = center.add(right.scale(a1.cos() * radius)).add(fwd.scale(a1.sin() * radius));
            let p2 = center.add(right.scale(a2.cos() * radius)).add(fwd.scale(a2.sin() * radius));
            self.lines.push(DebugLine { start: p1, end: p2, color });
        }
    }

    fn draw_wire_box_mm(&mut self, min: Vec3, max: Vec3, color: Color) {
        let corners = [
            Vec3::new(min.x,min.y,min.z), Vec3::new(max.x,min.y,min.z),
            Vec3::new(max.x,max.y,min.z), Vec3::new(min.x,max.y,min.z),
            Vec3::new(min.x,min.y,max.z), Vec3::new(max.x,min.y,max.z),
            Vec3::new(max.x,max.y,max.z), Vec3::new(min.x,max.y,max.z),
        ];
        let edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)];
        for &(a,b) in &edges {
            self.lines.push(DebugLine { start: corners[a], end: corners[b], color });
        }
    }

    fn draw_arrow_internal(&mut self, start: Vec3, end: Vec3, color: Color) {
        self.lines.push(DebugLine { start, end, color });
        let d = end.sub(start);
        let l = d.length();
        if l < 1e-6 { return; }
        let dn = d.scale(1.0 / l);
        let up = if dn.dot(Vec3::new(0.0,1.0,0.0)).abs() > 0.99 {
            Vec3::new(1.0,0.0,0.0)
        } else {
            Vec3::new(0.0,1.0,0.0)
        };
        let right = dn.cross(up).normalize();
        let h = l * 0.2;
        let base = end.sub(dn.scale(h));
        self.lines.push(DebugLine { start: end, end: base.add(right.scale(h * 0.3)), color });
        self.lines.push(DebugLine { start: end, end: base.sub(right.scale(h * 0.3)), color });
    }

    fn draw_cross_internal(&mut self, pos: Vec3, size: f32, color: Color) {
        self.lines.push(DebugLine { start: pos.add(Vec3::new(-size,0.0,0.0)), end: pos.add(Vec3::new(size,0.0,0.0)), color });
        self.lines.push(DebugLine { start: pos.add(Vec3::new(0.0,-size,0.0)), end: pos.add(Vec3::new(0.0,size,0.0)), color });
        self.lines.push(DebugLine { start: pos.add(Vec3::new(0.0,0.0,-size)), end: pos.add(Vec3::new(0.0,0.0,size)), color });
    }

    fn draw_arc_internal(&mut self, center: Vec3, axis: Vec3, min_a: f32, max_a: f32, radius: f32, color: Color) {
        let up = if axis.dot(Vec3::new(0.0,1.0,0.0)).abs() > 0.99 {
            Vec3::new(1.0,0.0,0.0)
        } else {
            Vec3::new(0.0,1.0,0.0)
        };
        let r = axis.cross(up).normalize();
        let f = r.cross(axis).normalize();
        let segs = 16u32;
        let step = (max_a - min_a) / segs as f32;
        for i in 0..segs {
            let a1 = min_a + i as f32 * step;
            let a2 = min_a + (i + 1) as f32 * step;
            let p1 = center.add(r.scale(a1.cos() * radius)).add(f.scale(a1.sin() * radius));
            let p2 = center.add(r.scale(a2.cos() * radius)).add(f.scale(a2.sin() * radius));
            self.lines.push(DebugLine { start: p1, end: p2, color });
        }
    }

    /// Finalize frame, update statistics.
    pub fn finalize(&mut self) {
        self.stats.lines_generated = self.lines.len() as u32;
        self.stats.points_generated = self.points.len() as u32;
        self.stats.texts_generated = self.texts.len() as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_renderer_creation() {
        let r = PhysicsDebugRendererV2::new();
        assert!(r.island_colors.len() > 0);
        assert!(r.settings.draw_collision_shapes);
    }

    #[test]
    fn test_draw_contacts() {
        let mut r = PhysicsDebugRendererV2::new();
        r.draw_contacts(&[ContactInfoDbg {
            position: Vec3::ZERO, normal: Vec3::new(0.0,1.0,0.0),
            depth: 0.01, normal_impulse: 1.0, friction_impulse: 0.1,
            body_a: 0, body_b: 1,
        }]);
        r.finalize();
        assert!(r.stats.contacts_drawn > 0);
        assert!(r.stats.lines_generated > 0);
    }

    #[test]
    fn test_draw_bodies() {
        let mut r = PhysicsDebugRendererV2::new();
        r.draw_bodies(&[PhysicsBodyInfoDbg {
            id: 0, body_type: BodyTypeDbg::Dynamic,
            position: Vec3::ZERO, linear_velocity: Vec3::new(1.0,0.0,0.0),
            angular_velocity: Vec3::ZERO, center_of_mass: Vec3::ZERO,
            sleep_state: SleepStateDbg::Awake, island_id: 0,
            shape: ShapeDataDbg::Sphere { radius: 1.0 },
            aabb_min: Vec3::new(-1.0,-1.0,-1.0), aabb_max: Vec3::new(1.0,1.0,1.0),
        }]);
        r.finalize();
        assert!(r.stats.bodies_drawn > 0);
    }

    #[test]
    fn test_draw_joints() {
        let mut r = PhysicsDebugRendererV2::new();
        r.draw_joints(&[JointInfoDbg {
            joint_type: JointTypeDbg::Hinge, body_a: 0, body_b: 1,
            anchor_a: Vec3::ZERO, anchor_b: Vec3::new(1.0,0.0,0.0),
            axis: Vec3::new(0.0,1.0,0.0), min_limit: -1.0, max_limit: 1.0,
            current_value: 0.5, error: 0.001,
        }]);
        r.finalize();
        assert!(r.stats.joints_drawn > 0);
    }
}
