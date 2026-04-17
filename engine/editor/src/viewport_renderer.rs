// engine/editor/src/viewport_renderer.rs
//
// Editor viewport integration for the Genovo engine.
//
// Connects the scene renderer to the editor viewport panel with camera
// controls, selection highlighting, grid overlay, and debug visualization.

use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::HashSet;

pub const DEFAULT_GRID_SIZE: f32 = 100.0;
pub const DEFAULT_GRID_SPACING: f32 = 1.0;
pub const DEFAULT_GRID_SUBDIVISIONS: u32 = 10;
pub const DEFAULT_NEAR_PLANE: f32 = 0.1;
pub const DEFAULT_FAR_PLANE: f32 = 5000.0;

pub const GRID_WGSL: &str = r#"
struct GridUniforms { view_proj: mat4x4<f32>, grid_color: vec4<f32>, grid_params: vec4<f32>, camera_pos: vec3<f32>, fade_distance: f32 };
@group(0) @binding(0) var<uniform> grid: GridUniforms;
struct VsIn { @location(0) position: vec3<f32> };
struct VsOut { @builtin(position) clip_pos: vec4<f32>, @location(0) world_pos: vec3<f32>, @location(1) @interpolate(flat) color: vec4<f32> };
@vertex fn vs_main(input: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = grid.view_proj * vec4<f32>(input.position, 1.0);
    out.world_pos = input.position;
    out.color = grid.grid_color;
    return out;
}
@fragment fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
    let dist = length(input.world_pos.xz - grid.camera_pos.xz);
    let fade = 1.0 - smoothstep(grid.fade_distance * 0.7, grid.fade_distance, dist);
    return vec4<f32>(input.color.rgb, input.color.a * fade);
}
"#;

pub const SELECTION_OUTLINE_WGSL: &str = r#"
struct SelectionUniforms { view_proj: mat4x4<f32>, model: mat4x4<f32>, outline_color: vec4<f32>, outline_width: f32, _pad: vec3<f32> };
@group(0) @binding(0) var<uniform> sel: SelectionUniforms;
struct VsIn { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32> };
struct VsOut { @builtin(position) clip_pos: vec4<f32> };
@vertex fn vs_main(input: VsIn) -> VsOut {
    var out: VsOut;
    let expanded = input.position + input.normal * sel.outline_width;
    out.clip_pos = sel.view_proj * sel.model * vec4<f32>(expanded, 1.0);
    return out;
}
@fragment fn fs_main(input: VsOut) -> @location(0) vec4<f32> { return sel.outline_color; }
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewportMode { Perspective, Top, Front, Right, Left, Back, Bottom }

impl ViewportMode {
    pub fn is_orthographic(&self) -> bool { !matches!(self, Self::Perspective) }
    pub fn camera_direction(&self) -> Vec3 {
        match self { Self::Perspective => Vec3::NEG_Z, Self::Top => Vec3::NEG_Y, Self::Bottom => Vec3::Y,
            Self::Front => Vec3::NEG_Z, Self::Back => Vec3::Z, Self::Right => Vec3::NEG_X, Self::Left => Vec3::X }
    }
    pub fn camera_up(&self) -> Vec3 {
        match self { Self::Top | Self::Bottom => Vec3::NEG_Z, _ => Vec3::Y }
    }
}

#[derive(Debug, Clone)]
pub struct ViewportCamera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub ortho_size: f32,
    pub mode: ViewportMode,
    pub orbit_distance: f32,
    pub orbit_yaw: f32,
    pub orbit_pitch: f32,
    pub pan_speed: f32,
    pub orbit_speed: f32,
    pub zoom_speed: f32,
    pub fly_speed: f32,
    pub is_flying: bool,
}

impl Default for ViewportCamera {
    fn default() -> Self {
        Self { position: Vec3::new(10.0, 10.0, 10.0), target: Vec3::ZERO, up: Vec3::Y, fov: 60.0,
            near: DEFAULT_NEAR_PLANE, far: DEFAULT_FAR_PLANE, ortho_size: 20.0, mode: ViewportMode::Perspective,
            orbit_distance: 15.0, orbit_yaw: 45.0f32.to_radians(), orbit_pitch: 30.0f32.to_radians(),
            pan_speed: 0.01, orbit_speed: 0.005, zoom_speed: 1.0, fly_speed: 10.0, is_flying: false }
    }
}

impl ViewportCamera {
    pub fn view_matrix(&self) -> Mat4 { Mat4::look_at_rh(self.position, self.target, self.up) }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        if self.mode.is_orthographic() {
            let half = self.ortho_size * 0.5;
            Mat4::orthographic_rh(-half * aspect, half * aspect, -half, half, self.near, self.far)
        } else {
            Mat4::perspective_rh(self.fov.to_radians(), aspect, self.near, self.far)
        }
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 { self.projection_matrix(aspect) * self.view_matrix() }

    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.orbit_yaw += delta_x * self.orbit_speed;
        self.orbit_pitch = (self.orbit_pitch + delta_y * self.orbit_speed).clamp(-1.5, 1.5);
        self.update_orbit_position();
    }

    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let view = self.view_matrix();
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let offset = right * -delta_x * self.pan_speed * self.orbit_distance + up * delta_y * self.pan_speed * self.orbit_distance;
        self.target += offset;
        self.update_orbit_position();
    }

    pub fn zoom(&mut self, delta: f32) {
        self.orbit_distance = (self.orbit_distance - delta * self.zoom_speed).clamp(0.5, 500.0);
        if self.mode.is_orthographic() { self.ortho_size = self.orbit_distance; }
        self.update_orbit_position();
    }

    pub fn focus_on(&mut self, center: Vec3, radius: f32) {
        self.target = center;
        self.orbit_distance = radius * 2.5;
        self.update_orbit_position();
    }

    fn update_orbit_position(&mut self) {
        let x = self.orbit_distance * self.orbit_pitch.cos() * self.orbit_yaw.sin();
        let y = self.orbit_distance * self.orbit_pitch.sin();
        let z = self.orbit_distance * self.orbit_pitch.cos() * self.orbit_yaw.cos();
        self.position = self.target + Vec3::new(x, y, z);
    }

    pub fn set_mode(&mut self, mode: ViewportMode) {
        self.mode = mode;
        if mode.is_orthographic() {
            self.position = self.target - mode.camera_direction() * self.orbit_distance;
            self.up = mode.camera_up();
        }
    }

    pub fn screen_to_ray(&self, screen_pos: Vec2, screen_size: Vec2, aspect: f32) -> (Vec3, Vec3) {
        let ndc_x = (screen_pos.x / screen_size.x) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_pos.y / screen_size.y) * 2.0;
        let inv_vp = self.view_projection(aspect).inverse();
        let near_ndc = Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far_ndc = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near_world = inv_vp * near_ndc;
        let far_world = inv_vp * far_ndc;
        let origin = Vec3::new(near_world.x / near_world.w, near_world.y / near_world.w, near_world.z / near_world.w);
        let far_pt = Vec3::new(far_world.x / far_world.w, far_world.y / far_world.w, far_world.z / far_world.w);
        let direction = (far_pt - origin).normalize();
        (origin, direction)
    }
}

#[derive(Debug, Clone)]
pub struct GridConfig {
    pub visible: bool,
    pub size: f32,
    pub spacing: f32,
    pub subdivisions: u32,
    pub color: Vec4,
    pub sub_color: Vec4,
    pub axis_x_color: Vec4,
    pub axis_z_color: Vec4,
    pub fade_distance: f32,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self { visible: true, size: DEFAULT_GRID_SIZE, spacing: DEFAULT_GRID_SPACING, subdivisions: DEFAULT_GRID_SUBDIVISIONS,
            color: Vec4::new(0.4, 0.4, 0.4, 0.5), sub_color: Vec4::new(0.3, 0.3, 0.3, 0.3),
            axis_x_color: Vec4::new(0.8, 0.2, 0.2, 0.6), axis_z_color: Vec4::new(0.2, 0.2, 0.8, 0.6),
            fade_distance: DEFAULT_GRID_SIZE * 0.8 }
    }
}

impl GridConfig {
    pub fn generate_lines(&self) -> Vec<[f32; 3]> {
        let mut verts = Vec::new();
        let half = self.size * 0.5;
        let count = (self.size / self.spacing) as i32;
        for i in -count..=count {
            let x = i as f32 * self.spacing;
            verts.push([x, 0.0, -half]); verts.push([x, 0.0, half]);
            verts.push([-half, 0.0, x]); verts.push([half, 0.0, x]);
        }
        verts
    }
}

#[derive(Debug, Clone)]
pub struct SelectionHighlight { pub entity_id: u64, pub outline_color: Vec4, pub outline_width: f32, pub fill_color: Option<Vec4> }

impl Default for SelectionHighlight {
    fn default() -> Self { Self { entity_id: 0, outline_color: Vec4::new(1.0, 0.6, 0.0, 1.0), outline_width: 0.02, fill_color: None } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DebugOverlay { None, Wireframe, Normals, UVs, Overdraw, Lightmap, Collision }

#[derive(Debug)]
pub struct ViewportRenderer {
    pub camera: ViewportCamera,
    pub grid: GridConfig,
    pub selected_entities: HashSet<u64>,
    pub selection_highlight: SelectionHighlight,
    pub debug_overlay: DebugOverlay,
    pub viewport_size: Vec2,
    pub show_gizmos: bool,
    pub show_icons: bool,
    pub show_bounds: bool,
    pub show_stats: bool,
    pub background_color: Vec4,
    pub render_handle: u64,
}

impl ViewportRenderer {
    pub fn new(width: f32, height: f32) -> Self {
        Self { camera: ViewportCamera::default(), grid: GridConfig::default(), selected_entities: HashSet::new(),
            selection_highlight: SelectionHighlight::default(), debug_overlay: DebugOverlay::None,
            viewport_size: Vec2::new(width, height), show_gizmos: true, show_icons: true,
            show_bounds: false, show_stats: true, background_color: Vec4::new(0.15, 0.15, 0.2, 1.0), render_handle: 0 }
    }

    pub fn aspect_ratio(&self) -> f32 { self.viewport_size.x / self.viewport_size.y.max(1.0) }

    pub fn resize(&mut self, width: f32, height: f32) { self.viewport_size = Vec2::new(width, height); }

    pub fn select_entity(&mut self, id: u64) { self.selected_entities.insert(id); }
    pub fn deselect_entity(&mut self, id: u64) { self.selected_entities.remove(&id); }
    pub fn clear_selection(&mut self) { self.selected_entities.clear(); }
    pub fn is_selected(&self, id: u64) -> bool { self.selected_entities.contains(&id) }

    pub fn pick_ray(&self, screen_pos: Vec2) -> (Vec3, Vec3) { self.camera.screen_to_ray(screen_pos, self.viewport_size, self.aspect_ratio()) }

    pub fn view_projection(&self) -> Mat4 { self.camera.view_projection(self.aspect_ratio()) }

    pub fn grid_shader_source(&self) -> &'static str { GRID_WGSL }
    pub fn selection_shader_source(&self) -> &'static str { SELECTION_OUTLINE_WGSL }

    pub fn frame_selection(&mut self, center: Vec3, radius: f32) { self.camera.focus_on(center, radius); }

    pub fn handle_orbit_input(&mut self, dx: f32, dy: f32) { self.camera.orbit(dx, dy); }
    pub fn handle_pan_input(&mut self, dx: f32, dy: f32) { self.camera.pan(dx, dy); }
    pub fn handle_zoom_input(&mut self, delta: f32) { self.camera.zoom(delta); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_viewport_creation() { let vr = ViewportRenderer::new(1920.0, 1080.0); assert!((vr.aspect_ratio() - 16.0/9.0).abs() < 0.01); }
    #[test] fn test_camera_orbit() { let mut c = ViewportCamera::default(); c.orbit(0.1, 0.1); assert!((c.position - Vec3::new(10.0, 10.0, 10.0)).length() > 0.01); }
    #[test] fn test_selection() { let mut vr = ViewportRenderer::new(800.0, 600.0); vr.select_entity(42); assert!(vr.is_selected(42)); vr.deselect_entity(42); assert!(!vr.is_selected(42)); }
    #[test] fn test_grid_lines() { let g = GridConfig::default(); let lines = g.generate_lines(); assert!(lines.len() > 100); }
    #[test] fn test_viewport_mode() { assert!(ViewportMode::Top.is_orthographic()); assert!(!ViewportMode::Perspective.is_orthographic()); }
}
