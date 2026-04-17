// engine/editor/src/gizmo_renderer.rs
//
// 3D transform gizmo rendering for the Genovo editor.
//
// Provides translate, rotate, and scale gizmo handles with axis/plane
// picking, delta computation from mouse drag, and WGSL shader rendering.

use glam::{Mat4, Vec2, Vec3, Vec4};

pub const GIZMO_HANDLE_SIZE: f32 = 0.15;
pub const GIZMO_AXIS_LENGTH: f32 = 1.5;
pub const GIZMO_PLANE_SIZE: f32 = 0.4;
pub const GIZMO_RING_RADIUS: f32 = 1.2;
pub const GIZMO_PICK_THRESHOLD: f32 = 12.0;

pub const GIZMO_WGSL: &str = r#"
struct GizmoUniforms { view_proj: mat4x4<f32>, color: vec4<f32>, model: mat4x4<f32> };
@group(0) @binding(0) var<uniform> gizmo: GizmoUniforms;
struct VsIn { @location(0) position: vec3<f32> };
struct VsOut { @builtin(position) clip_pos: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs_main(input: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = gizmo.view_proj * gizmo.model * vec4<f32>(input.position, 1.0);
    out.color = gizmo.color;
    return out;
}
@fragment fn fs_main(input: VsOut) -> @location(0) vec4<f32> { return input.color; }
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GizmoType { Translate, Rotate, Scale }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GizmoAxis { None, X, Y, Z, XY, XZ, YZ, XYZ, ViewPlane }

impl GizmoAxis {
    pub fn color(&self) -> Vec4 {
        match self { Self::X => Vec4::new(1.0, 0.2, 0.2, 1.0), Self::Y => Vec4::new(0.2, 1.0, 0.2, 1.0), Self::Z => Vec4::new(0.2, 0.2, 1.0, 1.0),
            Self::XY => Vec4::new(1.0, 1.0, 0.2, 0.5), Self::XZ => Vec4::new(1.0, 0.2, 1.0, 0.5), Self::YZ => Vec4::new(0.2, 1.0, 1.0, 0.5),
            Self::XYZ | Self::ViewPlane => Vec4::new(1.0, 1.0, 1.0, 0.7), Self::None => Vec4::new(0.5, 0.5, 0.5, 0.5) }
    }
    pub fn highlight_color(&self) -> Vec4 { let c = self.color(); Vec4::new(c.x.min(1.0) + 0.3, c.y.min(1.0) + 0.3, c.z.min(1.0) + 0.3, 1.0) }
    pub fn direction(&self) -> Vec3 { match self { Self::X => Vec3::X, Self::Y => Vec3::Y, Self::Z => Vec3::Z, _ => Vec3::ZERO } }
    pub fn plane_normal(&self) -> Vec3 { match self { Self::XY => Vec3::Z, Self::XZ => Vec3::Y, Self::YZ => Vec3::X, _ => Vec3::ZERO } }
}

#[derive(Debug, Clone)]
pub struct GizmoHandle { pub axis: GizmoAxis, pub gizmo_type: GizmoType, pub start: Vec3, pub end: Vec3, pub is_hovered: bool }

#[derive(Debug, Clone)]
pub struct GizmoState {
    pub gizmo_type: GizmoType,
    pub position: Vec3,
    pub rotation: glam::Quat,
    pub scale: f32,
    pub active_axis: GizmoAxis,
    pub hovered_axis: GizmoAxis,
    pub is_dragging: bool,
    pub drag_start_screen: Vec2,
    pub drag_start_world: Vec3,
    pub drag_start_value: Vec3,
    pub accumulated_delta: Vec3,
    pub use_local_space: bool,
    pub use_snapping: bool,
    pub snap_translate: f32,
    pub snap_rotate: f32,
    pub snap_scale: f32,
}

impl GizmoState {
    pub fn new() -> Self {
        Self { gizmo_type: GizmoType::Translate, position: Vec3::ZERO, rotation: glam::Quat::IDENTITY, scale: 1.0,
            active_axis: GizmoAxis::None, hovered_axis: GizmoAxis::None, is_dragging: false,
            drag_start_screen: Vec2::ZERO, drag_start_world: Vec3::ZERO, drag_start_value: Vec3::ZERO,
            accumulated_delta: Vec3::ZERO, use_local_space: false, use_snapping: false,
            snap_translate: 0.5, snap_rotate: 15.0, snap_scale: 0.1 }
    }

    pub fn set_mode(&mut self, mode: GizmoType) { self.gizmo_type = mode; self.cancel_drag(); }
    pub fn toggle_local_space(&mut self) { self.use_local_space = !self.use_local_space; }
    pub fn toggle_snapping(&mut self) { self.use_snapping = !self.use_snapping; }
    pub fn cancel_drag(&mut self) { self.is_dragging = false; self.active_axis = GizmoAxis::None; self.accumulated_delta = Vec3::ZERO; }
}

#[derive(Debug, Clone)]
pub struct GizmoDelta { pub translation: Vec3, pub rotation: Vec3, pub scale: Vec3, pub axis: GizmoAxis }

/// The gizmo renderer and interaction handler.
#[derive(Debug)]
pub struct GizmoRenderer {
    pub state: GizmoState,
    pub handles: Vec<GizmoHandle>,
    pub visible: bool,
    pub screen_scale_factor: f32,
}

impl GizmoRenderer {
    pub fn new() -> Self { Self { state: GizmoState::new(), handles: Vec::new(), visible: true, screen_scale_factor: 1.0 } }

    pub fn update_scale(&mut self, camera_pos: Vec3) {
        let dist = (self.state.position - camera_pos).length();
        self.screen_scale_factor = (dist * 0.15).clamp(0.5, 5.0);
    }

    pub fn build_handles(&mut self) {
        self.handles.clear();
        let pos = self.state.position;
        let s = self.screen_scale_factor;
        match self.state.gizmo_type {
            GizmoType::Translate => {
                self.handles.push(GizmoHandle { axis: GizmoAxis::X, gizmo_type: GizmoType::Translate, start: pos, end: pos + Vec3::X * GIZMO_AXIS_LENGTH * s, is_hovered: false });
                self.handles.push(GizmoHandle { axis: GizmoAxis::Y, gizmo_type: GizmoType::Translate, start: pos, end: pos + Vec3::Y * GIZMO_AXIS_LENGTH * s, is_hovered: false });
                self.handles.push(GizmoHandle { axis: GizmoAxis::Z, gizmo_type: GizmoType::Translate, start: pos, end: pos + Vec3::Z * GIZMO_AXIS_LENGTH * s, is_hovered: false });
                // Plane handles.
                let ps = GIZMO_PLANE_SIZE * s;
                self.handles.push(GizmoHandle { axis: GizmoAxis::XY, gizmo_type: GizmoType::Translate, start: pos + Vec3::new(ps, 0.0, 0.0), end: pos + Vec3::new(ps, ps, 0.0), is_hovered: false });
                self.handles.push(GizmoHandle { axis: GizmoAxis::XZ, gizmo_type: GizmoType::Translate, start: pos + Vec3::new(ps, 0.0, 0.0), end: pos + Vec3::new(ps, 0.0, ps), is_hovered: false });
                self.handles.push(GizmoHandle { axis: GizmoAxis::YZ, gizmo_type: GizmoType::Translate, start: pos + Vec3::new(0.0, ps, 0.0), end: pos + Vec3::new(0.0, ps, ps), is_hovered: false });
            }
            GizmoType::Rotate => {
                let r = GIZMO_RING_RADIUS * s;
                self.handles.push(GizmoHandle { axis: GizmoAxis::X, gizmo_type: GizmoType::Rotate, start: pos, end: pos + Vec3::X * r, is_hovered: false });
                self.handles.push(GizmoHandle { axis: GizmoAxis::Y, gizmo_type: GizmoType::Rotate, start: pos, end: pos + Vec3::Y * r, is_hovered: false });
                self.handles.push(GizmoHandle { axis: GizmoAxis::Z, gizmo_type: GizmoType::Rotate, start: pos, end: pos + Vec3::Z * r, is_hovered: false });
            }
            GizmoType::Scale => {
                self.handles.push(GizmoHandle { axis: GizmoAxis::X, gizmo_type: GizmoType::Scale, start: pos, end: pos + Vec3::X * GIZMO_AXIS_LENGTH * s * 0.8, is_hovered: false });
                self.handles.push(GizmoHandle { axis: GizmoAxis::Y, gizmo_type: GizmoType::Scale, start: pos, end: pos + Vec3::Y * GIZMO_AXIS_LENGTH * s * 0.8, is_hovered: false });
                self.handles.push(GizmoHandle { axis: GizmoAxis::Z, gizmo_type: GizmoType::Scale, start: pos, end: pos + Vec3::Z * GIZMO_AXIS_LENGTH * s * 0.8, is_hovered: false });
                self.handles.push(GizmoHandle { axis: GizmoAxis::XYZ, gizmo_type: GizmoType::Scale, start: pos, end: pos + Vec3::ONE.normalize() * s * 0.3, is_hovered: false });
            }
        }
    }

    pub fn pick(&mut self, screen_pos: Vec2, view_proj: &Mat4, screen_size: Vec2) -> GizmoAxis {
        let mut best = GizmoAxis::None;
        let mut best_dist = GIZMO_PICK_THRESHOLD;
        for handle in &mut self.handles {
            let a = world_to_screen(handle.start, view_proj, screen_size);
            let b = world_to_screen(handle.end, view_proj, screen_size);
            let d = point_to_segment_distance(screen_pos, a, b);
            handle.is_hovered = d < GIZMO_PICK_THRESHOLD;
            if d < best_dist { best_dist = d; best = handle.axis; }
        }
        self.state.hovered_axis = best;
        best
    }

    pub fn begin_drag(&mut self, screen_pos: Vec2, world_pos: Vec3) {
        if self.state.hovered_axis == GizmoAxis::None { return; }
        self.state.is_dragging = true;
        self.state.active_axis = self.state.hovered_axis;
        self.state.drag_start_screen = screen_pos;
        self.state.drag_start_world = world_pos;
        self.state.drag_start_value = self.state.position;
        self.state.accumulated_delta = Vec3::ZERO;
    }

    pub fn update_drag(&mut self, screen_pos: Vec2, view_proj: &Mat4, screen_size: Vec2) -> Option<GizmoDelta> {
        if !self.state.is_dragging { return None; }
        let screen_delta = screen_pos - self.state.drag_start_screen;
        let axis = self.state.active_axis;

        match self.state.gizmo_type {
            GizmoType::Translate => {
                let dir = axis.direction();
                if dir.length() > 0.0 {
                    let proj_dir = world_to_screen(self.state.position + dir, view_proj, screen_size)
                        - world_to_screen(self.state.position, view_proj, screen_size);
                    let proj_len = proj_dir.length();
                    if proj_len > 0.01 {
                        let t = screen_delta.dot(proj_dir) / (proj_len * proj_len);
                        let mut delta = dir * t * self.screen_scale_factor;
                        if self.state.use_snapping { delta = snap_vec3(delta, self.state.snap_translate); }
                        return Some(GizmoDelta { translation: delta, rotation: Vec3::ZERO, scale: Vec3::ONE, axis });
                    }
                }
            }
            GizmoType::Rotate => {
                let sensitivity = 0.5;
                let angle = screen_delta.x * sensitivity;
                let mut rot = axis.direction() * angle.to_radians();
                if self.state.use_snapping { rot = snap_vec3(rot, self.state.snap_rotate.to_radians()); }
                return Some(GizmoDelta { translation: Vec3::ZERO, rotation: rot, scale: Vec3::ONE, axis });
            }
            GizmoType::Scale => {
                let sensitivity = 0.01;
                let factor = 1.0 + screen_delta.x * sensitivity;
                let mut s = if axis == GizmoAxis::XYZ { Vec3::splat(factor) }
                    else { let d = axis.direction(); Vec3::ONE + d * (factor - 1.0) };
                if self.state.use_snapping { s = snap_vec3(s, self.state.snap_scale); }
                return Some(GizmoDelta { translation: Vec3::ZERO, rotation: Vec3::ZERO, scale: s, axis });
            }
        }
        None
    }

    pub fn end_drag(&mut self) -> Option<GizmoDelta> { self.state.cancel_drag(); None }
    pub fn shader_source(&self) -> &'static str { GIZMO_WGSL }
}

fn world_to_screen(pos: Vec3, vp: &Mat4, screen: Vec2) -> Vec2 {
    let clip = *vp * Vec4::new(pos.x, pos.y, pos.z, 1.0);
    if clip.w < 0.001 { return Vec2::new(-10000.0, -10000.0); }
    let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
    Vec2::new((ndc.x * 0.5 + 0.5) * screen.x, (0.5 - ndc.y * 0.5) * screen.y)
}

fn point_to_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a; let ap = p - a;
    let t = ap.dot(ab) / ab.dot(ab).max(0.0001);
    let t = t.clamp(0.0, 1.0);
    let closest = a + ab * t;
    (p - closest).length()
}

fn snap_vec3(v: Vec3, snap: f32) -> Vec3 {
    if snap < 0.001 { return v; }
    Vec3::new((v.x / snap).round() * snap, (v.y / snap).round() * snap, (v.z / snap).round() * snap)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_gizmo_state() { let s = GizmoState::new(); assert_eq!(s.gizmo_type, GizmoType::Translate); }
    #[test] fn test_axis_colors() { let c = GizmoAxis::X.color(); assert!(c.x > 0.5); }
    #[test] fn test_pick_threshold() { let d = point_to_segment_distance(Vec2::new(5.0, 5.0), Vec2::ZERO, Vec2::new(10.0, 0.0)); assert!(d > 0.0); }
    #[test] fn test_snap() { let v = snap_vec3(Vec3::new(0.7, 1.3, 2.8), 0.5); assert!((v.x - 0.5).abs() < 0.01); assert!((v.y - 1.5).abs() < 0.01); }
}
