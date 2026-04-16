// engine/render/src/camera_system.rs
//
// Camera system for the Genovo engine. Provides multiple camera management,
// view/projection matrix computation, frustum culling integration, and TAA
// jitter support using Halton sequences.

use crate::lighting::light_culling::Frustum;
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

// ---------------------------------------------------------------------------
// Projection types
// ---------------------------------------------------------------------------

/// Camera projection mode.
#[derive(Debug, Clone, Copy)]
pub enum ProjectionMode {
    /// Perspective projection with vertical FOV.
    Perspective {
        /// Vertical field of view in radians.
        fov_y: f32,
        /// Near clip plane distance.
        near: f32,
        /// Far clip plane distance.
        far: f32,
    },
    /// Orthographic projection.
    Orthographic {
        /// Left edge of the view volume.
        left: f32,
        /// Right edge of the view volume.
        right: f32,
        /// Bottom edge of the view volume.
        bottom: f32,
        /// Top edge of the view volume.
        top: f32,
        /// Near clip plane distance.
        near: f32,
        /// Far clip plane distance.
        far: f32,
    },
}

impl Default for ProjectionMode {
    fn default() -> Self {
        Self::Perspective {
            fov_y: 60.0_f32.to_radians(),
            near: 0.1,
            far: 1000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// CameraComponent
// ---------------------------------------------------------------------------

/// A camera component representing a viewpoint in the scene.
#[derive(Debug, Clone)]
pub struct CameraComponent {
    /// World-space position.
    pub position: Vec3,
    /// Rotation (orientation).
    pub rotation: Quat,
    /// Projection mode (perspective or orthographic).
    pub projection: ProjectionMode,
    /// Viewport aspect ratio (width / height).
    pub aspect_ratio: f32,
    /// Clear colour for this camera's render target.
    pub clear_color: Vec4,
    /// Priority (highest-priority camera is the active one).
    pub priority: i32,
    /// Whether this camera is active.
    pub active: bool,
    /// Optional render target override (None = swapchain).
    pub render_target: Option<u64>,
    /// Viewport rectangle (normalised 0..1): x, y, width, height.
    pub viewport: [f32; 4],
    /// TAA jitter offset in pixels (applied to projection matrix).
    pub jitter: Vec2,
    /// Current frame index (for TAA jitter sequence).
    pub frame_index: u64,
    /// Exposure override (0.0 = use auto-exposure).
    pub exposure: f32,
}

impl Default for CameraComponent {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 1.0, 5.0),
            rotation: Quat::IDENTITY,
            projection: ProjectionMode::default(),
            aspect_ratio: 16.0 / 9.0,
            clear_color: Vec4::new(0.1, 0.1, 0.15, 1.0),
            priority: 0,
            active: true,
            render_target: None,
            viewport: [0.0, 0.0, 1.0, 1.0],
            jitter: Vec2::ZERO,
            frame_index: 0,
            exposure: 1.0,
        }
    }
}

impl CameraComponent {
    /// Create a new perspective camera.
    pub fn perspective(position: Vec3, fov_y: f32, near: f32, far: f32) -> Self {
        Self {
            position,
            projection: ProjectionMode::Perspective { fov_y, near, far },
            ..Default::default()
        }
    }

    /// Create a new orthographic camera.
    pub fn orthographic(position: Vec3, half_height: f32, near: f32, far: f32) -> Self {
        Self {
            position,
            projection: ProjectionMode::Orthographic {
                left: -half_height,
                right: half_height,
                bottom: -half_height,
                top: half_height,
                near,
                far,
            },
            ..Default::default()
        }
    }

    /// Switch to perspective projection.
    pub fn set_perspective(&mut self, fov_y: f32, near: f32, far: f32) {
        self.projection = ProjectionMode::Perspective { fov_y, near, far };
    }

    /// Switch to orthographic projection.
    pub fn set_orthographic(&mut self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) {
        self.projection = ProjectionMode::Orthographic { left, right, bottom, top, near, far };
    }

    /// Compute the view matrix from position and rotation.
    pub fn view_matrix(&self) -> Mat4 {
        compute_view_matrix(self.position, self.rotation)
    }

    /// Compute the projection matrix (without jitter).
    pub fn projection_matrix(&self) -> Mat4 {
        match self.projection {
            ProjectionMode::Perspective { fov_y, near, far } => {
                compute_projection_matrix(fov_y, self.aspect_ratio, near, far)
            }
            ProjectionMode::Orthographic { left, right, bottom, top, near, far } => {
                compute_ortho_matrix(left, right, bottom, top, near, far)
            }
        }
    }

    /// Compute the projection matrix with TAA jitter applied.
    pub fn jittered_projection_matrix(&self, viewport_width: u32, viewport_height: u32) -> Mat4 {
        let mut proj = self.projection_matrix();

        if self.jitter.x.abs() > 1e-8 || self.jitter.y.abs() > 1e-8 {
            // Apply sub-pixel jitter offset. The jitter is in pixel units;
            // convert to clip space by dividing by viewport dimensions.
            let jitter_x = self.jitter.x / viewport_width as f32;
            let jitter_y = self.jitter.y / viewport_height as f32;

            // Offset the projection matrix by adding to the translation
            // components of the last column.
            // For proper sub-pixel jitter, modify the projection matrix
            // elements [2][0] and [2][1] to shift the clip-space origin.
            proj.col_mut(2).x += jitter_x * 2.0;
            proj.col_mut(2).y += jitter_y * 2.0;
        }

        proj
    }

    /// Compute the view-projection matrix.
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Compute the jittered view-projection matrix.
    pub fn jittered_view_projection(&self, viewport_width: u32, viewport_height: u32) -> Mat4 {
        self.jittered_projection_matrix(viewport_width, viewport_height) * self.view_matrix()
    }

    /// Compute the inverse view-projection matrix.
    pub fn inv_view_projection_matrix(&self) -> Mat4 {
        self.view_projection_matrix().inverse()
    }

    /// Get the forward direction (-Z in camera space).
    pub fn forward(&self) -> Vec3 {
        self.rotation * -Vec3::Z
    }

    /// Get the right direction (+X in camera space).
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Get the up direction (+Y in camera space).
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Look at a target position.
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let view = Mat4::look_at_rh(self.position, target, up);
        // Extract rotation from the view matrix.
        let inv_view = view.inverse();
        let (_, rot, _) = inv_view.to_scale_rotation_translation();
        self.rotation = rot;
    }

    /// Extract the frustum planes from the current view-projection.
    pub fn frustum(&self) -> Frustum {
        let vp = self.view_projection_matrix();
        Frustum::from_view_projection(&vp)
    }

    /// Get the near clip distance.
    pub fn near_plane(&self) -> f32 {
        match self.projection {
            ProjectionMode::Perspective { near, .. } => near,
            ProjectionMode::Orthographic { near, .. } => near,
        }
    }

    /// Get the far clip distance.
    pub fn far_plane(&self) -> f32 {
        match self.projection {
            ProjectionMode::Perspective { far, .. } => far,
            ProjectionMode::Orthographic { far, .. } => far,
        }
    }

    /// Get the vertical field of view (perspective only, returns 0 for ortho).
    pub fn fov_y(&self) -> f32 {
        match self.projection {
            ProjectionMode::Perspective { fov_y, .. } => fov_y,
            ProjectionMode::Orthographic { .. } => 0.0,
        }
    }

    /// Whether this camera uses perspective projection.
    pub fn is_perspective(&self) -> bool {
        matches!(self.projection, ProjectionMode::Perspective { .. })
    }

    /// Update TAA jitter for this frame using a Halton sequence.
    pub fn update_jitter(&mut self, frame_index: u64, sequence_length: u32) {
        self.frame_index = frame_index;
        let sample_index = (frame_index % sequence_length as u64) as u32;
        self.jitter = halton_jitter(sample_index, sequence_length);
    }

    /// Clear the jitter offset.
    pub fn clear_jitter(&mut self) {
        self.jitter = Vec2::ZERO;
    }

    /// Compute a ray from screen coordinates (normalised [0,1]).
    ///
    /// Returns (ray_origin, ray_direction) in world space.
    pub fn screen_to_ray(&self, screen_x: f32, screen_y: f32) -> (Vec3, Vec3) {
        let inv_vp = self.inv_view_projection_matrix();

        let clip_x = screen_x * 2.0 - 1.0;
        let clip_y = (1.0 - screen_y) * 2.0 - 1.0;

        let near_point = inv_vp * Vec4::new(clip_x, clip_y, 0.0, 1.0);
        let far_point = inv_vp * Vec4::new(clip_x, clip_y, 1.0, 1.0);

        let near_world = near_point.truncate() / near_point.w;
        let far_world = far_point.truncate() / far_point.w;

        let direction = (far_world - near_world).normalize_or_zero();
        (near_world, direction)
    }
}

// ---------------------------------------------------------------------------
// Matrix computation functions
// ---------------------------------------------------------------------------

/// Compute a view matrix from position and rotation quaternion.
///
/// Uses right-handed coordinate system with -Z as forward.
pub fn compute_view_matrix(position: Vec3, rotation: Quat) -> Mat4 {
    let rot_matrix = Mat4::from_quat(rotation);
    let translation = Mat4::from_translation(-position);
    // View = inverse(T * R) = R_inv * T_inv = R^T * T(-pos)
    rot_matrix.transpose() * translation
}

/// Compute a perspective projection matrix.
///
/// Right-handed, depth range [0, 1] (Vulkan/wgpu convention).
pub fn compute_projection_matrix(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    Mat4::perspective_rh(fov_y, aspect, near, far)
}

/// Compute an orthographic projection matrix.
///
/// Right-handed, depth range [0, 1].
pub fn compute_ortho_matrix(
    left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32,
) -> Mat4 {
    Mat4::orthographic_rh(left, right, bottom, top, near, far)
}

/// Compute a view matrix from position, target, and up vector.
pub fn compute_look_at_matrix(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    Mat4::look_at_rh(eye, target, up)
}

/// Compute a reverse-Z infinite perspective projection.
///
/// Used for improved depth precision. Maps [near, inf) to [1, 0].
pub fn compute_infinite_reverse_z_projection(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, near, 0.0),
    )
}

// ---------------------------------------------------------------------------
// TAA Jitter (Halton sequence)
// ---------------------------------------------------------------------------

/// Compute a 2D Halton sequence point for TAA sub-pixel jitter.
///
/// Returns the jitter offset in pixel units, centred around zero.
pub fn halton_jitter(sample_index: u32, _sequence_length: u32) -> Vec2 {
    let x = halton_sequence(sample_index, 2);
    let y = halton_sequence(sample_index, 3);
    // Map from [0,1] to [-0.5, 0.5] pixel offset.
    Vec2::new(x - 0.5, y - 0.5)
}

/// Compute the n-th element of a Halton sequence with the given base.
fn halton_sequence(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0f32;
    let mut fraction = 1.0f32;
    let base_f = base as f32;

    while index > 0 {
        fraction /= base_f;
        result += fraction * (index % base) as f32;
        index /= base;
    }

    result
}

/// Generate a full Halton jitter sequence for TAA.
///
/// Returns `count` 2D jitter offsets in [-0.5, 0.5] pixel range.
pub fn generate_halton_sequence(count: u32) -> Vec<Vec2> {
    (0..count).map(|i| halton_jitter(i, count)).collect()
}

/// R2 quasi-random sequence (Martin Roberts' method).
///
/// Alternative to Halton that produces better 2D coverage.
pub fn r2_jitter(sample_index: u32) -> Vec2 {
    let g = 1.324_717_957_244_746; // plastic constant
    let a1 = 1.0 / g;
    let a2 = 1.0 / (g * g);
    let x = (0.5 + a1 * sample_index as f64) % 1.0;
    let y = (0.5 + a2 * sample_index as f64) % 1.0;
    Vec2::new(x as f32 - 0.5, y as f32 - 0.5)
}

// ---------------------------------------------------------------------------
// CameraManager
// ---------------------------------------------------------------------------

/// Manages multiple cameras and selects the active one for rendering.
pub struct CameraManager {
    /// All registered cameras.
    cameras: Vec<CameraComponent>,
    /// Index of the currently active camera (cached).
    active_index: Option<usize>,
    /// Whether the active index needs recomputation.
    dirty: bool,
}

impl CameraManager {
    /// Create a new camera manager.
    pub fn new() -> Self {
        Self { cameras: Vec::new(), active_index: None, dirty: true }
    }

    /// Add a camera. Returns its index.
    pub fn add_camera(&mut self, camera: CameraComponent) -> usize {
        let idx = self.cameras.len();
        self.cameras.push(camera);
        self.dirty = true;
        idx
    }

    /// Remove a camera by index.
    pub fn remove_camera(&mut self, index: usize) -> Option<CameraComponent> {
        if index < self.cameras.len() {
            self.dirty = true;
            Some(self.cameras.remove(index))
        } else {
            None
        }
    }

    /// Get a camera by index.
    pub fn get_camera(&self, index: usize) -> Option<&CameraComponent> {
        self.cameras.get(index)
    }

    /// Get a mutable camera by index.
    pub fn get_camera_mut(&mut self, index: usize) -> Option<&mut CameraComponent> {
        self.cameras.get_mut(index)
    }

    /// Number of cameras.
    pub fn camera_count(&self) -> usize { self.cameras.len() }

    /// Find and return a reference to the active camera (highest priority + active).
    pub fn active_camera(&mut self) -> Option<&CameraComponent> {
        self.update_active();
        self.active_index.map(|i| &self.cameras[i])
    }

    /// Get the active camera mutably.
    pub fn active_camera_mut(&mut self) -> Option<&mut CameraComponent> {
        self.update_active();
        self.active_index.map(move |i| &mut self.cameras[i])
    }

    /// Get the index of the active camera.
    pub fn active_index(&mut self) -> Option<usize> {
        self.update_active();
        self.active_index
    }

    /// Set a camera's priority.
    pub fn set_priority(&mut self, index: usize, priority: i32) {
        if let Some(cam) = self.cameras.get_mut(index) {
            cam.priority = priority;
            self.dirty = true;
        }
    }

    /// Set a camera's active state.
    pub fn set_active(&mut self, index: usize, active: bool) {
        if let Some(cam) = self.cameras.get_mut(index) {
            cam.active = active;
            self.dirty = true;
        }
    }

    /// Update the aspect ratio for all cameras.
    pub fn set_aspect_ratio(&mut self, aspect: f32) {
        for cam in &mut self.cameras {
            cam.aspect_ratio = aspect;
        }
    }

    /// Notify all cameras of a viewport resize.
    pub fn on_resize(&mut self, width: u32, height: u32) {
        if height > 0 {
            let aspect = width as f32 / height as f32;
            self.set_aspect_ratio(aspect);
        }
    }

    /// Update TAA jitter for the active camera.
    pub fn update_jitter(&mut self, frame_index: u64, sequence_length: u32) {
        self.update_active();
        if let Some(idx) = self.active_index {
            self.cameras[idx].update_jitter(frame_index, sequence_length);
        }
    }

    /// Get all cameras that need rendering (active cameras, sorted by priority).
    pub fn cameras_to_render(&mut self) -> Vec<(usize, &CameraComponent)> {
        let mut active: Vec<(usize, &CameraComponent)> = self.cameras
            .iter()
            .enumerate()
            .filter(|(_, c)| c.active)
            .collect();
        active.sort_by(|a, b| b.1.priority.cmp(&a.1.priority));
        active
    }

    /// Recompute the active camera index.
    fn update_active(&mut self) {
        if !self.dirty { return; }
        self.active_index = self.cameras
            .iter()
            .enumerate()
            .filter(|(_, c)| c.active)
            .max_by_key(|(_, c)| c.priority)
            .map(|(i, _)| i);
        self.dirty = false;
    }

    /// Clear all cameras.
    pub fn clear(&mut self) {
        self.cameras.clear();
        self.active_index = None;
        self.dirty = false;
    }
}

impl Default for CameraManager {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Frustum culling helpers
// ---------------------------------------------------------------------------

/// Cull a list of bounding spheres against the camera frustum.
///
/// Returns a bitmask (as a Vec<bool>) indicating which objects are visible.
pub fn frustum_cull_spheres(
    frustum: &Frustum,
    positions: &[Vec3],
    radii: &[f32],
) -> Vec<bool> {
    assert_eq!(positions.len(), radii.len());
    positions.iter().zip(radii.iter())
        .map(|(pos, radius)| frustum.intersects_sphere(*pos, *radius))
        .collect()
}

/// Cull AABBs against the camera frustum.
pub fn frustum_cull_aabbs(
    frustum: &Frustum,
    mins: &[Vec3],
    maxs: &[Vec3],
) -> Vec<bool> {
    assert_eq!(mins.len(), maxs.len());
    mins.iter().zip(maxs.iter())
        .map(|(min, max)| frustum.intersects_aabb(*min, *max))
        .collect()
}

/// Compute the screen-space projected size of a sphere.
///
/// Returns the approximate diameter in pixels.
pub fn projected_sphere_diameter(
    center: Vec3,
    radius: f32,
    view: &Mat4,
    projection: &Mat4,
    viewport_height: f32,
) -> f32 {
    let view_pos = (*view * center.extend(1.0)).truncate();
    let dist = -view_pos.z; // camera looks down -Z
    if dist <= 0.0 { return 0.0; }

    // The projected size depends on the projection matrix.
    // For perspective: proj_size = (radius / distance) * viewport_height / tan(fov/2)
    // We can extract tan(fov/2) from the projection matrix element [1][1] = 1/tan(fov/2).
    let proj_scale = projection.col(1).y; // 1/tan(fov/2) for perspective
    let projected_radius = radius * proj_scale / dist;
    projected_radius * viewport_height
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn camera_component_default() {
        let cam = CameraComponent::default();
        assert!(cam.active);
        assert!(cam.is_perspective());
        assert_eq!(cam.priority, 0);
    }

    #[test]
    fn camera_perspective_creation() {
        let cam = CameraComponent::perspective(
            Vec3::new(0.0, 5.0, 10.0),
            60.0_f32.to_radians(),
            0.1,
            100.0,
        );
        assert!(cam.is_perspective());
        assert!((cam.fov_y() - 60.0_f32.to_radians()).abs() < 1e-5);
    }

    #[test]
    fn camera_orthographic_creation() {
        let cam = CameraComponent::orthographic(Vec3::ZERO, 5.0, 0.1, 100.0);
        assert!(!cam.is_perspective());
        assert_eq!(cam.fov_y(), 0.0);
    }

    #[test]
    fn camera_projection_switch() {
        let mut cam = CameraComponent::default();
        assert!(cam.is_perspective());
        cam.set_orthographic(-5.0, 5.0, -5.0, 5.0, 0.1, 100.0);
        assert!(!cam.is_perspective());
        cam.set_perspective(90.0_f32.to_radians(), 0.1, 1000.0);
        assert!(cam.is_perspective());
    }

    #[test]
    fn view_matrix_identity_at_origin() {
        let view = compute_view_matrix(Vec3::ZERO, Quat::IDENTITY);
        // At origin with identity rotation, view should be identity.
        let id = Mat4::IDENTITY;
        for i in 0..4 {
            for j in 0..4 {
                assert!((view.col(i)[j] - id.col(i)[j]).abs() < 1e-5,
                    "View matrix should be identity at origin");
            }
        }
    }

    #[test]
    fn projection_matrix_not_zero() {
        let proj = compute_projection_matrix(
            60.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0,
        );
        assert!(proj.col(0).x != 0.0);
        assert!(proj.col(1).y != 0.0);
    }

    #[test]
    fn ortho_matrix_computation() {
        let ortho = compute_ortho_matrix(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
        // Orthographic should map the centre to clip (0,0,z).
        let center = ortho * Vec4::new(0.0, 0.0, -50.0, 1.0);
        assert!((center.x).abs() < 1e-5);
        assert!((center.y).abs() < 1e-5);
    }

    #[test]
    fn camera_forward_direction() {
        let cam = CameraComponent::default();
        let fwd = cam.forward();
        // Default rotation (identity) should have forward = -Z.
        assert!((fwd.z + 1.0).abs() < 1e-5);
    }

    #[test]
    fn camera_look_at() {
        let mut cam = CameraComponent::default();
        cam.position = Vec3::new(0.0, 5.0, 10.0);
        cam.look_at(Vec3::ZERO, Vec3::Y);
        let fwd = cam.forward();
        // Should point roughly toward the origin.
        let expected = (Vec3::ZERO - cam.position).normalize();
        assert!((fwd - expected).length() < 0.1);
    }

    #[test]
    fn camera_screen_to_ray() {
        let mut cam = CameraComponent::perspective(
            Vec3::new(0.0, 0.0, 5.0), 60.0_f32.to_radians(), 0.1, 100.0,
        );
        cam.look_at(Vec3::ZERO, Vec3::Y);
        let (origin, dir) = cam.screen_to_ray(0.5, 0.5);
        // Centre of screen should produce a ray going roughly toward -Z.
        assert!(dir.z < 0.0, "Ray should point toward -Z");
    }

    #[test]
    fn halton_sequence_values() {
        // Base 2: 1/2, 1/4, 3/4, 1/8, ...
        let h1 = halton_sequence(1, 2);
        assert!((h1 - 0.5).abs() < 1e-5);

        let h2 = halton_sequence(2, 2);
        assert!((h2 - 0.25).abs() < 1e-5);

        // Base 3: 1/3, 2/3, 1/9, ...
        let h1_b3 = halton_sequence(1, 3);
        assert!((h1_b3 - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn halton_jitter_range() {
        for i in 0..16 {
            let jitter = halton_jitter(i, 16);
            assert!(jitter.x >= -0.5 && jitter.x <= 0.5);
            assert!(jitter.y >= -0.5 && jitter.y <= 0.5);
        }
    }

    #[test]
    fn r2_jitter_range() {
        for i in 0..16 {
            let jitter = r2_jitter(i);
            assert!(jitter.x >= -0.5 && jitter.x <= 0.5,
                "R2 jitter x out of range: {}", jitter.x);
            assert!(jitter.y >= -0.5 && jitter.y <= 0.5,
                "R2 jitter y out of range: {}", jitter.y);
        }
    }

    #[test]
    fn halton_sequence_generation() {
        let seq = generate_halton_sequence(8);
        assert_eq!(seq.len(), 8);
    }

    #[test]
    fn camera_manager_basics() {
        let mut mgr = CameraManager::new();
        let cam = CameraComponent::perspective(Vec3::ZERO, 1.0, 0.1, 100.0);
        let idx = mgr.add_camera(cam);
        assert_eq!(idx, 0);
        assert_eq!(mgr.camera_count(), 1);
        assert!(mgr.active_camera().is_some());
    }

    #[test]
    fn camera_manager_priority() {
        let mut mgr = CameraManager::new();
        let mut cam1 = CameraComponent::default();
        cam1.priority = 0;
        let mut cam2 = CameraComponent::default();
        cam2.priority = 10;

        mgr.add_camera(cam1);
        mgr.add_camera(cam2);

        let active_idx = mgr.active_index().unwrap();
        assert_eq!(active_idx, 1, "Higher priority camera should be active");
    }

    #[test]
    fn camera_manager_deactivation() {
        let mut mgr = CameraManager::new();
        let cam = CameraComponent::default();
        let idx = mgr.add_camera(cam);

        mgr.set_active(idx, false);
        assert!(mgr.active_camera().is_none());
    }

    #[test]
    fn camera_manager_resize() {
        let mut mgr = CameraManager::new();
        mgr.add_camera(CameraComponent::default());
        mgr.on_resize(2560, 1440);
        let cam = mgr.get_camera(0).unwrap();
        assert!((cam.aspect_ratio - 2560.0 / 1440.0).abs() < 0.01);
    }

    #[test]
    fn camera_manager_jitter_update() {
        let mut mgr = CameraManager::new();
        mgr.add_camera(CameraComponent::default());
        mgr.update_jitter(5, 16);
        let cam = mgr.active_camera().unwrap();
        assert_eq!(cam.frame_index, 5);
        // Jitter should be non-zero for frame 5.
        assert!(cam.jitter.x != 0.0 || cam.jitter.y != 0.0);
    }

    #[test]
    fn camera_manager_cameras_to_render() {
        let mut mgr = CameraManager::new();
        let mut cam1 = CameraComponent::default();
        cam1.priority = 1;
        cam1.active = true;
        let mut cam2 = CameraComponent::default();
        cam2.priority = 5;
        cam2.active = true;
        let mut cam3 = CameraComponent::default();
        cam3.active = false;

        mgr.add_camera(cam1);
        mgr.add_camera(cam2);
        mgr.add_camera(cam3);

        let to_render = mgr.cameras_to_render();
        assert_eq!(to_render.len(), 2); // only 2 active cameras
        assert_eq!(to_render[0].0, 1); // highest priority first
    }

    #[test]
    fn frustum_cull_spheres_test() {
        let vp = Mat4::perspective_rh(PI / 2.0, 1.0, 0.1, 100.0);
        let frustum = Frustum::from_view_projection(&vp);

        let positions = vec![
            Vec3::new(0.0, 0.0, -5.0), // inside
            Vec3::new(200.0, 0.0, -5.0), // outside
        ];
        let radii = vec![1.0, 1.0];

        let visible = frustum_cull_spheres(&frustum, &positions, &radii);
        assert!(visible[0]);
        // The second one is far outside.
    }

    #[test]
    fn projected_sphere_size() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(PI / 4.0, 1.0, 0.1, 100.0);
        let diameter = projected_sphere_diameter(Vec3::ZERO, 1.0, &view, &proj, 1080.0);
        assert!(diameter > 0.0);
    }

    #[test]
    fn infinite_reverse_z_projection() {
        let proj = compute_infinite_reverse_z_projection(PI / 4.0, 16.0 / 9.0, 0.1);
        // Near plane should map to depth=1, far to depth=0.
        let near_point = proj * Vec4::new(0.0, 0.0, -0.1, 1.0);
        let near_ndc_z = near_point.z / near_point.w;
        // With reverse-Z, near should map to approx 1.0.
        assert!((near_ndc_z - 1.0).abs() < 0.1);
    }

    #[test]
    fn camera_jittered_projection() {
        let mut cam = CameraComponent::default();
        cam.jitter = Vec2::new(0.5, -0.3);
        let proj = cam.projection_matrix();
        let jittered = cam.jittered_projection_matrix(1920, 1080);
        // Jittered should differ from un-jittered.
        assert!((proj.col(2).x - jittered.col(2).x).abs() > 1e-6);
    }
}
