// engine/render/src/shadows/shadow_map.rs
//
// Core shadow map types: depth textures, atlas packing, light-space matrix
// computation, and settings. Provides the foundation for all shadow
// techniques (cascaded, PCF, VSM, etc.).

use glam::{Mat4, Vec3, Vec4};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ShadowSettings
// ---------------------------------------------------------------------------

/// Global shadow settings for the renderer.
#[derive(Debug, Clone)]
pub struct ShadowSettings {
    /// Shadow map resolution (width = height for square maps).
    pub resolution: u32,
    /// Atlas resolution (the large texture that holds all shadow maps).
    pub atlas_resolution: u32,
    /// Constant depth bias to prevent shadow acne.
    pub depth_bias: f32,
    /// Normal-offset bias (pushes sample along surface normal).
    pub normal_bias: f32,
    /// Near plane for shadow cameras.
    pub near_plane: f32,
    /// Far plane for directional light shadow cameras.
    pub far_plane: f32,
    /// Maximum number of shadow maps (limits atlas slots).
    pub max_shadow_maps: u32,
    /// Whether to use 32-bit depth (otherwise 16-bit).
    pub use_32bit_depth: bool,
    /// Global shadow distance (objects beyond this are unshadowed).
    pub shadow_distance: f32,
    /// Fade-out range at the edge of the shadow distance.
    pub fade_distance: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            resolution: 2048,
            atlas_resolution: 8192,
            depth_bias: 0.005,
            normal_bias: 0.02,
            near_plane: 0.1,
            far_plane: 500.0,
            max_shadow_maps: 32,
            use_32bit_depth: true,
            shadow_distance: 200.0,
            fade_distance: 20.0,
        }
    }
}

impl ShadowSettings {
    /// High-quality shadow settings.
    pub fn high_quality() -> Self {
        Self {
            resolution: 4096,
            atlas_resolution: 16384,
            depth_bias: 0.002,
            normal_bias: 0.01,
            max_shadow_maps: 64,
            ..Default::default()
        }
    }

    /// Low-quality settings for mobile / performance.
    pub fn low_quality() -> Self {
        Self {
            resolution: 1024,
            atlas_resolution: 4096,
            depth_bias: 0.01,
            normal_bias: 0.04,
            max_shadow_maps: 16,
            use_32bit_depth: false,
            shadow_distance: 100.0,
            ..Default::default()
        }
    }

    /// Compute the shadow fade factor at a given distance from the camera.
    ///
    /// Returns 1.0 for fully shadowed, 0.0 for fully faded.
    pub fn shadow_fade(&self, distance: f32) -> f32 {
        if distance >= self.shadow_distance {
            return 0.0;
        }
        let fade_start = self.shadow_distance - self.fade_distance;
        if distance <= fade_start {
            return 1.0;
        }
        1.0 - (distance - fade_start) / self.fade_distance
    }
}

// ---------------------------------------------------------------------------
// ShadowMap
// ---------------------------------------------------------------------------

/// A single shadow map (depth texture) associated with a light source.
#[derive(Debug, Clone)]
pub struct ShadowMap {
    /// Unique identifier.
    pub id: u32,
    /// Resolution (width = height for square maps).
    pub resolution: u32,
    /// Light-space view-projection matrix.
    pub light_matrix: Mat4,
    /// Depth bias for this shadow map.
    pub depth_bias: f32,
    /// Normal bias for this shadow map.
    pub normal_bias: f32,
    /// Which face of a cube map this represents (0..5 for point lights,
    /// or 0 for directional/spot).
    pub face_index: u32,
    /// Index into the shadow atlas (set by the atlas packer).
    pub atlas_slot: Option<u32>,
    /// UV offset within the atlas.
    pub atlas_uv_offset: [f32; 2],
    /// UV scale within the atlas.
    pub atlas_uv_scale: [f32; 2],
}

impl ShadowMap {
    /// Create a new shadow map.
    pub fn new(id: u32, resolution: u32, light_matrix: Mat4) -> Self {
        Self {
            id,
            resolution,
            light_matrix,
            depth_bias: 0.005,
            normal_bias: 0.02,
            face_index: 0,
            atlas_slot: None,
            atlas_uv_offset: [0.0, 0.0],
            atlas_uv_scale: [1.0, 1.0],
        }
    }

    /// Compute the shadow matrix that transforms world-space positions to
    /// shadow-map UV + depth coordinates.
    ///
    /// The result maps from world space to [0,1]^3 where xy = UV and z = depth.
    pub fn world_to_shadow_uv(&self) -> Mat4 {
        // The light matrix maps to clip space [-1,1]^3. We need to remap
        // to [0,1]^3, then apply the atlas offset/scale.
        let bias_matrix = Mat4::from_cols_array(&[
            0.5, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0,
        ]);

        let atlas_matrix = Mat4::from_cols_array(&[
            self.atlas_uv_scale[0],
            0.0,
            0.0,
            0.0,
            0.0,
            self.atlas_uv_scale[1],
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            self.atlas_uv_offset[0],
            self.atlas_uv_offset[1],
            0.0,
            1.0,
        ]);

        atlas_matrix * bias_matrix * self.light_matrix
    }
}

// ---------------------------------------------------------------------------
// ShadowMapEntry (atlas slot)
// ---------------------------------------------------------------------------

/// An entry in the shadow map atlas, tracking the position and size of
/// a shadow map within the atlas texture.
#[derive(Debug, Clone, Copy)]
pub struct ShadowMapEntry {
    /// X offset in pixels within the atlas.
    pub x: u32,
    /// Y offset in pixels within the atlas.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Which shadow map this slot is assigned to.
    pub shadow_map_id: u32,
}

impl ShadowMapEntry {
    /// UV offset (normalised to [0,1] based on atlas size).
    pub fn uv_offset(&self, atlas_size: u32) -> [f32; 2] {
        [
            self.x as f32 / atlas_size as f32,
            self.y as f32 / atlas_size as f32,
        ]
    }

    /// UV scale (normalised to [0,1] based on atlas size).
    pub fn uv_scale(&self, atlas_size: u32) -> [f32; 2] {
        [
            self.width as f32 / atlas_size as f32,
            self.height as f32 / atlas_size as f32,
        ]
    }
}

// ---------------------------------------------------------------------------
// ShadowMapAtlas
// ---------------------------------------------------------------------------

/// Packs multiple shadow maps into a single large depth texture (atlas).
///
/// This avoids the overhead of switching render targets for each shadow
/// map. The atlas uses a simple shelf-packing algorithm.
pub struct ShadowMapAtlas {
    /// Atlas texture size (square: width = height).
    pub atlas_size: u32,
    /// Allocated entries.
    entries: Vec<ShadowMapEntry>,
    /// Current shelf positions for packing.
    shelf_x: u32,
    shelf_y: u32,
    shelf_height: u32,
    /// Map from shadow_map_id to entry index.
    id_to_entry: HashMap<u32, usize>,
    /// Next shadow map ID.
    next_id: u32,
}

impl ShadowMapAtlas {
    /// Create a new atlas with the given size.
    pub fn new(atlas_size: u32) -> Self {
        Self {
            atlas_size,
            entries: Vec::new(),
            shelf_x: 0,
            shelf_y: 0,
            shelf_height: 0,
            id_to_entry: HashMap::new(),
            next_id: 0,
        }
    }

    /// Clear all allocations.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.id_to_entry.clear();
        self.shelf_x = 0;
        self.shelf_y = 0;
        self.shelf_height = 0;
    }

    /// Allocate a slot for a shadow map of the given resolution.
    ///
    /// Uses a simple shelf-packing algorithm. Returns `None` if the atlas
    /// is full.
    pub fn allocate(&mut self, resolution: u32) -> Option<ShadowMapEntry> {
        if resolution > self.atlas_size {
            return None;
        }

        // Check if the current shelf has room.
        if self.shelf_x + resolution > self.atlas_size {
            // Move to next shelf.
            self.shelf_y += self.shelf_height;
            self.shelf_x = 0;
            self.shelf_height = 0;
        }

        if self.shelf_y + resolution > self.atlas_size {
            return None; // Atlas full.
        }

        let id = self.next_id;
        self.next_id += 1;

        let entry = ShadowMapEntry {
            x: self.shelf_x,
            y: self.shelf_y,
            width: resolution,
            height: resolution,
            shadow_map_id: id,
        };

        let idx = self.entries.len();
        self.entries.push(entry);
        self.id_to_entry.insert(id, idx);

        self.shelf_x += resolution;
        self.shelf_height = self.shelf_height.max(resolution);

        Some(entry)
    }

    /// Allocate a slot and update the shadow map's atlas information.
    pub fn allocate_for(&mut self, shadow_map: &mut ShadowMap) -> bool {
        if let Some(entry) = self.allocate(shadow_map.resolution) {
            shadow_map.atlas_slot = Some(entry.shadow_map_id);
            shadow_map.atlas_uv_offset = entry.uv_offset(self.atlas_size);
            shadow_map.atlas_uv_scale = entry.uv_scale(self.atlas_size);
            true
        } else {
            false
        }
    }

    /// Look up an entry by shadow map ID.
    pub fn get_entry(&self, id: u32) -> Option<&ShadowMapEntry> {
        self.id_to_entry.get(&id).map(|&idx| &self.entries[idx])
    }

    /// Number of allocated slots.
    pub fn slot_count(&self) -> usize {
        self.entries.len()
    }

    /// Remaining area in the atlas (approximate).
    pub fn remaining_area(&self) -> u32 {
        let total = self.atlas_size * self.atlas_size;
        let used: u32 = self.entries.iter().map(|e| e.width * e.height).sum();
        total.saturating_sub(used)
    }

    /// All entries.
    pub fn entries(&self) -> &[ShadowMapEntry] {
        &self.entries
    }
}

// ---------------------------------------------------------------------------
// Light-space matrix computation
// ---------------------------------------------------------------------------

/// Compute an orthographic light-space view-projection matrix for a
/// directional light.
///
/// The matrix maps a region of the scene (defined by the frustum corners)
/// into the light's clip space.
///
/// # Arguments
/// - `light_dir` — direction *toward* the light (normalised).
/// - `frustum_corners` — 8 corners of the camera frustum in world space.
///
/// # Returns
/// The combined light view-projection matrix.
pub fn compute_directional_light_matrix(
    light_dir: Vec3,
    frustum_corners: &[Vec3; 8],
) -> Mat4 {
    // Look-at matrix for the light.
    let light_pos = Vec3::ZERO; // We'll translate later.
    let up = if light_dir.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let light_view = Mat4::look_at_rh(light_pos, light_pos - light_dir, up);

    // Transform frustum corners to light space and compute the tight AABB.
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for &corner in frustum_corners {
        let ls = light_view.transform_point3(corner);
        min = min.min(ls);
        max = max.max(ls);
    }

    // Orthographic projection.
    let light_proj = Mat4::orthographic_rh(min.x, max.x, min.y, max.y, min.z, max.z);

    light_proj * light_view
}

/// Compute a perspective light-space matrix for a spot light.
pub fn compute_spot_light_matrix(
    position: Vec3,
    direction: Vec3,
    outer_angle: f32,
    range: f32,
) -> Mat4 {
    let target = position + direction;
    let up = if direction.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(position, target, up);
    let proj = Mat4::perspective_rh(outer_angle * 2.0, 1.0, 0.1, range);
    proj * view
}

/// Compute the 6 face view-projection matrices for a point light cube map.
///
/// Returns 6 matrices for +X, -X, +Y, -Y, +Z, -Z faces.
pub fn compute_point_light_cube_matrices(
    position: Vec3,
    near: f32,
    far: f32,
) -> [Mat4; 6] {
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far);

    let targets = [
        position + Vec3::X,  // +X
        position - Vec3::X,  // -X
        position + Vec3::Y,  // +Y
        position - Vec3::Y,  // -Y
        position + Vec3::Z,  // +Z
        position - Vec3::Z,  // -Z
    ];

    let ups = [
        -Vec3::Y, // +X
        -Vec3::Y, // -X
        Vec3::Z,  // +Y
        -Vec3::Z, // -Y
        -Vec3::Y, // +Z
        -Vec3::Y, // -Z
    ];

    let mut matrices = [Mat4::IDENTITY; 6];
    for i in 0..6 {
        let view = Mat4::look_at_rh(position, targets[i], ups[i]);
        matrices[i] = proj * view;
    }
    matrices
}

/// Compute the 8 corners of a camera frustum from its inverse
/// view-projection matrix.
pub fn frustum_corners_world_space(inv_view_proj: &Mat4) -> [Vec3; 8] {
    let ndc_corners = [
        Vec4::new(-1.0, -1.0, 0.0, 1.0), // near-bottom-left
        Vec4::new(1.0, -1.0, 0.0, 1.0),  // near-bottom-right
        Vec4::new(-1.0, 1.0, 0.0, 1.0),  // near-top-left
        Vec4::new(1.0, 1.0, 0.0, 1.0),   // near-top-right
        Vec4::new(-1.0, -1.0, 1.0, 1.0), // far-bottom-left
        Vec4::new(1.0, -1.0, 1.0, 1.0),  // far-bottom-right
        Vec4::new(-1.0, 1.0, 1.0, 1.0),  // far-top-left
        Vec4::new(1.0, 1.0, 1.0, 1.0),   // far-top-right
    ];

    let mut corners = [Vec3::ZERO; 8];
    for (i, &ndc) in ndc_corners.iter().enumerate() {
        let world = *inv_view_proj * ndc;
        corners[i] = world.truncate() / world.w;
    }
    corners
}

/// Compute the 8 corners of a sub-frustum (for cascade splits).
///
/// Linearly interpolates between the near and far frustum corners.
///
/// # Arguments
/// - `near_corners` — 4 corners of the near plane.
/// - `far_corners` — 4 corners of the far plane.
/// - `near_t` — interpolation parameter for the near split (0..1).
/// - `far_t` — interpolation parameter for the far split (0..1).
pub fn sub_frustum_corners(
    near_corners: &[Vec3; 4],
    far_corners: &[Vec3; 4],
    near_t: f32,
    far_t: f32,
) -> [Vec3; 8] {
    let mut corners = [Vec3::ZERO; 8];
    for i in 0..4 {
        let ray = far_corners[i] - near_corners[i];
        corners[i] = near_corners[i] + ray * near_t;     // near plane
        corners[i + 4] = near_corners[i] + ray * far_t;  // far plane
    }
    corners
}

// ---------------------------------------------------------------------------
// GPU shadow data
// ---------------------------------------------------------------------------

/// GPU-compatible shadow data for a single shadow map.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowDataGpu {
    /// Light-space view-projection matrix.
    pub light_matrix: [[f32; 4]; 4],
    /// xy = UV offset in atlas, zw = UV scale.
    pub atlas_uv: [f32; 4],
    /// x = depth_bias, y = normal_bias, z = shadow_map_index, w = padding.
    pub bias_params: [f32; 4],
}

impl ShadowDataGpu {
    /// Build from a `ShadowMap`.
    pub fn from_shadow_map(sm: &ShadowMap) -> Self {
        Self {
            light_matrix: sm.light_matrix.to_cols_array_2d(),
            atlas_uv: [
                sm.atlas_uv_offset[0],
                sm.atlas_uv_offset[1],
                sm.atlas_uv_scale[0],
                sm.atlas_uv_scale[1],
            ],
            bias_params: [sm.depth_bias, sm.normal_bias, sm.face_index as f32, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_settings_defaults() {
        let s = ShadowSettings::default();
        assert_eq!(s.resolution, 2048);
        assert!(s.depth_bias > 0.0);
    }

    #[test]
    fn shadow_fade_factor() {
        let s = ShadowSettings {
            shadow_distance: 100.0,
            fade_distance: 20.0,
            ..Default::default()
        };
        assert!((s.shadow_fade(50.0) - 1.0).abs() < 0.001);
        assert!((s.shadow_fade(100.0)).abs() < 0.001);
        assert!((s.shadow_fade(90.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn atlas_packing() {
        let mut atlas = ShadowMapAtlas::new(4096);
        // Should fit 4 maps of 2048x2048 in a 4096x4096 atlas.
        assert!(atlas.allocate(2048).is_some());
        assert!(atlas.allocate(2048).is_some());
        assert!(atlas.allocate(2048).is_some());
        assert!(atlas.allocate(2048).is_some());
        // The 5th should fail.
        assert!(atlas.allocate(2048).is_none());
    }

    #[test]
    fn atlas_clear_resets() {
        let mut atlas = ShadowMapAtlas::new(2048);
        atlas.allocate(1024);
        atlas.allocate(1024);
        assert_eq!(atlas.slot_count(), 2);
        atlas.clear();
        assert_eq!(atlas.slot_count(), 0);
    }

    #[test]
    fn point_light_cube_matrices() {
        let matrices = compute_point_light_cube_matrices(Vec3::ZERO, 0.1, 100.0);
        for m in &matrices {
            // Each matrix should be non-identity.
            assert_ne!(*m, Mat4::IDENTITY);
        }
    }

    #[test]
    fn frustum_corners_near_far() {
        let proj = Mat4::perspective_rh(1.0, 1.0, 1.0, 100.0);
        let inv = proj.inverse();
        let corners = frustum_corners_world_space(&inv);
        // Near corners should have smaller Z magnitude than far corners.
        assert!(corners[0].z.abs() < corners[4].z.abs());
    }
}
