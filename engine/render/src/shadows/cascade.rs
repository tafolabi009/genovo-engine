// engine/render/src/shadows/cascade.rs
//
// Cascaded Shadow Maps (CSM) for directional lights. Splits the camera
// frustum into depth slices, each with its own shadow map, providing
// high-resolution shadows close to the camera and lower resolution far away.
//
// Implements logarithmic, uniform, and PSSM (Practical Split Scheme Method)
// split schemes, as well as cascade stabilisation to prevent shadow swimming.

use super::shadow_map::{
    compute_directional_light_matrix, frustum_corners_world_space, sub_frustum_corners, ShadowMap,
};
use glam::{Mat4, Vec3, Vec4};

// ---------------------------------------------------------------------------
// CascadeSplitScheme
// ---------------------------------------------------------------------------

/// Method for dividing the camera frustum into cascade depth splits.
#[derive(Debug, Clone, PartialEq)]
pub enum CascadeSplitScheme {
    /// Uniform (linear) splits. Simple but wastes resolution near the camera.
    Uniform,
    /// Logarithmic splits. Gives more resolution near the camera but can
    /// create very thin near-slices.
    Logarithmic,
    /// Practical Split Scheme Method (PSSM). A blend of uniform and
    /// logarithmic controlled by a lambda parameter (0 = uniform, 1 = log).
    Pssm { lambda: f32 },
    /// Manual split distances (normalised 0..1).
    Manual { splits: Vec<f32> },
}

impl Default for CascadeSplitScheme {
    fn default() -> Self {
        Self::Pssm { lambda: 0.75 }
    }
}

// ---------------------------------------------------------------------------
// CascadeSplit
// ---------------------------------------------------------------------------

/// The computed split distances for a set of cascades.
#[derive(Debug, Clone)]
pub struct CascadeSplit {
    /// Split distances in view-space depth. The i-th cascade covers
    /// [splits[i], splits[i+1]).
    pub splits: Vec<f32>,
    /// Number of cascades.
    pub cascade_count: u32,
}

impl CascadeSplit {
    /// Compute cascade splits using the given scheme.
    ///
    /// # Arguments
    /// - `scheme` — the split method.
    /// - `cascade_count` — number of cascades (1..8).
    /// - `near` — camera near plane.
    /// - `far` — shadow distance (may be less than camera far).
    pub fn compute(scheme: &CascadeSplitScheme, cascade_count: u32, near: f32, far: f32) -> Self {
        let count = cascade_count.max(1).min(8);
        let mut splits = Vec::with_capacity(count as usize + 1);
        splits.push(near);

        match scheme {
            CascadeSplitScheme::Uniform => {
                for i in 1..=count {
                    let t = i as f32 / count as f32;
                    splits.push(near + (far - near) * t);
                }
            }
            CascadeSplitScheme::Logarithmic => {
                for i in 1..=count {
                    let t = i as f32 / count as f32;
                    splits.push(near * (far / near).powf(t));
                }
            }
            CascadeSplitScheme::Pssm { lambda } => {
                let lam = lambda.clamp(0.0, 1.0);
                for i in 1..=count {
                    let t = i as f32 / count as f32;
                    let log_split = near * (far / near).powf(t);
                    let uniform_split = near + (far - near) * t;
                    let split = lam * log_split + (1.0 - lam) * uniform_split;
                    splits.push(split);
                }
            }
            CascadeSplitScheme::Manual { splits: manual } => {
                for (i, &s) in manual.iter().enumerate() {
                    if i >= count as usize {
                        break;
                    }
                    splits.push(near + (far - near) * s.clamp(0.0, 1.0));
                }
                // Pad if not enough manual splits.
                while splits.len() <= count as usize {
                    splits.push(far);
                }
            }
        }

        Self {
            splits,
            cascade_count: count,
        }
    }

    /// Get the near depth of cascade `i`.
    pub fn cascade_near(&self, i: u32) -> f32 {
        self.splits[i as usize]
    }

    /// Get the far depth of cascade `i`.
    pub fn cascade_far(&self, i: u32) -> f32 {
        self.splits[(i + 1) as usize]
    }

    /// Returns the splits as a Vec4 (for up to 4 cascades).
    /// Unused slots are set to the far distance.
    pub fn splits_vec4(&self) -> Vec4 {
        let far = *self.splits.last().unwrap_or(&100.0);
        Vec4::new(
            self.splits.get(1).copied().unwrap_or(far),
            self.splits.get(2).copied().unwrap_or(far),
            self.splits.get(3).copied().unwrap_or(far),
            self.splits.get(4).copied().unwrap_or(far),
        )
    }

    /// Select the cascade index for a given view-space depth.
    pub fn select_cascade(&self, view_depth: f32) -> u32 {
        for i in 0..self.cascade_count {
            if view_depth < self.cascade_far(i) {
                return i;
            }
        }
        self.cascade_count - 1
    }

    /// Compute a cross-cascade blend factor for smooth transitions.
    ///
    /// Returns (cascade_index, blend_factor) where blend_factor is 0.0
    /// at the center of the cascade and approaches 1.0 at the boundary.
    pub fn cascade_blend(&self, view_depth: f32, blend_distance: f32) -> (u32, f32) {
        let cascade = self.select_cascade(view_depth);
        let far = self.cascade_far(cascade);
        let dist_to_boundary = far - view_depth;

        if dist_to_boundary < blend_distance && cascade + 1 < self.cascade_count {
            let t = 1.0 - (dist_to_boundary / blend_distance);
            (cascade, t)
        } else {
            (cascade, 0.0)
        }
    }
}

// ---------------------------------------------------------------------------
// CascadedShadowMap
// ---------------------------------------------------------------------------

/// A set of cascaded shadow maps for a single directional light.
pub struct CascadedShadowMap {
    /// The cascade split distances.
    pub splits: CascadeSplit,
    /// Per-cascade shadow maps.
    pub cascade_maps: Vec<ShadowMap>,
    /// Per-cascade light-space matrices.
    pub cascade_matrices: Vec<Mat4>,
    /// Light direction (toward the light).
    pub light_direction: Vec3,
    /// Shadow map resolution per cascade.
    pub resolution: u32,
    /// Whether stabilisation is enabled.
    pub stabilise: bool,
    /// Texel size for stabilisation (computed from resolution).
    _texel_size: f32,
}

impl CascadedShadowMap {
    /// Create a new cascaded shadow map.
    ///
    /// # Arguments
    /// - `light_direction` — direction toward the light (normalised).
    /// - `cascade_count` — number of cascades (1..8).
    /// - `resolution` — shadow map resolution per cascade.
    /// - `scheme` — cascade split scheme.
    /// - `near` — camera near plane.
    /// - `far` — shadow distance.
    pub fn new(
        light_direction: Vec3,
        cascade_count: u32,
        resolution: u32,
        scheme: &CascadeSplitScheme,
        near: f32,
        far: f32,
    ) -> Self {
        let splits = CascadeSplit::compute(scheme, cascade_count, near, far);
        let cascade_maps = (0..cascade_count)
            .map(|i| ShadowMap::new(i, resolution, Mat4::IDENTITY))
            .collect();
        let cascade_matrices = vec![Mat4::IDENTITY; cascade_count as usize];

        Self {
            splits,
            cascade_maps,
            cascade_matrices,
            light_direction: light_direction.normalize_or_zero(),
            resolution,
            stabilise: true,
            _texel_size: 0.0,
        }
    }

    /// Update the cascade matrices for the current camera.
    ///
    /// # Arguments
    /// - `camera_view` — camera view matrix.
    /// - `camera_proj` — camera projection matrix.
    pub fn update(&mut self, camera_view: &Mat4, camera_proj: &Mat4) {
        let inv_view_proj = (*camera_proj * *camera_view).inverse();
        let full_corners = frustum_corners_world_space(&inv_view_proj);

        // Separate near and far plane corners.
        let near_corners = [full_corners[0], full_corners[1], full_corners[2], full_corners[3]];
        let far_corners = [full_corners[4], full_corners[5], full_corners[6], full_corners[7]];

        let camera_near = self.splits.cascade_near(0);
        let camera_far = *self.splits.splits.last().unwrap();
        let total_range = camera_far - camera_near;

        for i in 0..self.splits.cascade_count {
            let near_t = (self.splits.cascade_near(i) - camera_near) / total_range;
            let far_t = (self.splits.cascade_far(i) - camera_near) / total_range;

            let cascade_corners =
                sub_frustum_corners(&near_corners, &far_corners, near_t, far_t);

            let mut matrix =
                compute_directional_light_matrix(self.light_direction, &cascade_corners);

            if self.stabilise {
                matrix = self.stabilise_matrix(matrix, &cascade_corners);
            }

            self.cascade_matrices[i as usize] = matrix;
            self.cascade_maps[i as usize].light_matrix = matrix;
        }
    }

    /// Stabilise a cascade matrix to prevent shadow swimming.
    ///
    /// Quantises the light-space position to the shadow map texel grid,
    /// ensuring that the shadow map only shifts by whole texels as the
    /// camera moves.
    fn stabilise_matrix(&self, matrix: Mat4, corners: &[Vec3; 8]) -> Mat4 {
        // Compute the size of the projection in light space.
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for &corner in corners {
            let ls = matrix.transform_point3(corner);
            min = min.min(ls);
            max = max.max(ls);
        }

        let extent_x = max.x - min.x;
        let extent_y = max.y - min.y;

        let texel_size_x = extent_x / self.resolution as f32;
        let texel_size_y = extent_y / self.resolution as f32;

        if texel_size_x < 1e-7 || texel_size_y < 1e-7 {
            return matrix;
        }

        // Extract the translation from the matrix.
        let mut cols = matrix.to_cols_array_2d();

        // Quantise the X and Y translation to texel boundaries.
        cols[3][0] = (cols[3][0] / texel_size_x).floor() * texel_size_x;
        cols[3][1] = (cols[3][1] / texel_size_y).floor() * texel_size_y;

        Mat4::from_cols_array_2d(&cols)
    }

    /// Get the cascade matrices as an array suitable for GPU upload.
    pub fn matrices_array(&self) -> Vec<[[f32; 4]; 4]> {
        self.cascade_matrices
            .iter()
            .map(|m| m.to_cols_array_2d())
            .collect()
    }

    /// Get the cascade count.
    pub fn cascade_count(&self) -> u32 {
        self.splits.cascade_count
    }
}

// ---------------------------------------------------------------------------
// compute_cascade_matrices (standalone function)
// ---------------------------------------------------------------------------

/// Compute cascade light-space matrices for a directional light.
///
/// This is the main entry point for cascade computation. It performs:
/// 1. Frustum extraction from the camera matrices.
/// 2. Split computation using the given scheme.
/// 3. Per-cascade matrix computation with optional stabilisation.
///
/// # Arguments
/// - `camera_view` — camera view matrix.
/// - `camera_proj` — camera projection matrix.
/// - `light_dir` — direction toward the light source.
/// - `cascade_count` — number of cascades.
/// - `scheme` — split scheme.
/// - `shadow_distance` — maximum shadow distance.
/// - `near` — camera near plane.
/// - `resolution` — shadow map resolution (for stabilisation).
/// - `stabilise` — whether to quantise to texel grid.
///
/// # Returns
/// A tuple of (cascade_matrices, split_depths).
pub fn compute_cascade_matrices(
    camera_view: &Mat4,
    camera_proj: &Mat4,
    light_dir: Vec3,
    cascade_count: u32,
    scheme: &CascadeSplitScheme,
    shadow_distance: f32,
    near: f32,
    resolution: u32,
    stabilise: bool,
) -> (Vec<Mat4>, CascadeSplit) {
    let splits = CascadeSplit::compute(scheme, cascade_count, near, shadow_distance);

    let inv_view_proj = (*camera_proj * *camera_view).inverse();
    let full_corners = frustum_corners_world_space(&inv_view_proj);

    let near_corners = [full_corners[0], full_corners[1], full_corners[2], full_corners[3]];
    let far_corners = [full_corners[4], full_corners[5], full_corners[6], full_corners[7]];

    let total_range = shadow_distance - near;

    let mut matrices = Vec::with_capacity(cascade_count as usize);

    for i in 0..cascade_count {
        let near_t = (splits.cascade_near(i) - near) / total_range;
        let far_t = (splits.cascade_far(i) - near) / total_range;

        let cascade_corners = sub_frustum_corners(&near_corners, &far_corners, near_t, far_t);
        let mut matrix = compute_directional_light_matrix(light_dir, &cascade_corners);

        if stabilise {
            matrix = stabilise_cascade_matrix(matrix, &cascade_corners, resolution);
        }

        matrices.push(matrix);
    }

    (matrices, splits)
}

/// Stabilise a cascade matrix by quantising to the texel grid.
fn stabilise_cascade_matrix(matrix: Mat4, corners: &[Vec3; 8], resolution: u32) -> Mat4 {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for &corner in corners {
        let ls = matrix.transform_point3(corner);
        min = min.min(ls);
        max = max.max(ls);
    }

    let extent_x = max.x - min.x;
    let extent_y = max.y - min.y;

    let texel_size_x = extent_x / resolution as f32;
    let texel_size_y = extent_y / resolution as f32;

    if texel_size_x < 1e-7 || texel_size_y < 1e-7 {
        return matrix;
    }

    let mut cols = matrix.to_cols_array_2d();
    cols[3][0] = (cols[3][0] / texel_size_x).floor() * texel_size_x;
    cols[3][1] = (cols[3][1] / texel_size_y).floor() * texel_size_y;

    Mat4::from_cols_array_2d(&cols)
}

// ---------------------------------------------------------------------------
// GPU cascade data
// ---------------------------------------------------------------------------

/// GPU-compatible cascade shadow data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CascadeDataGpu {
    /// Up to 4 cascade matrices.
    pub matrices: [[[f32; 4]; 4]; 4],
    /// Split depths (view-space).
    pub split_depths: [f32; 4],
    /// Bias parameters per cascade.
    pub bias: [f32; 4],
    /// Padding.
    pub _pad: [f32; 4],
}

impl CascadeDataGpu {
    /// Build from a cascaded shadow map.
    pub fn from_csm(csm: &CascadedShadowMap, depth_bias: f32) -> Self {
        let mut matrices = [[[0.0f32; 4]; 4]; 4];
        for (i, m) in csm.cascade_matrices.iter().enumerate() {
            if i < 4 {
                matrices[i] = m.to_cols_array_2d();
            }
        }

        let splits = csm.splits.splits_vec4();

        Self {
            matrices,
            split_depths: splits.to_array(),
            bias: [depth_bias; 4],
            _pad: [0.0; 4],
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
    fn uniform_splits_are_evenly_spaced() {
        let split = CascadeSplit::compute(&CascadeSplitScheme::Uniform, 4, 0.1, 100.0);
        assert_eq!(split.splits.len(), 5);
        let d1 = split.splits[1] - split.splits[0];
        let d2 = split.splits[2] - split.splits[1];
        assert!((d1 - d2).abs() < 0.1);
    }

    #[test]
    fn log_splits_closer_near() {
        let split = CascadeSplit::compute(&CascadeSplitScheme::Logarithmic, 4, 0.1, 100.0);
        // First split should be much smaller than last.
        let d_first = split.cascade_far(0) - split.cascade_near(0);
        let d_last =
            split.cascade_far(split.cascade_count - 1) - split.cascade_near(split.cascade_count - 1);
        assert!(d_first < d_last);
    }

    #[test]
    fn pssm_is_between_uniform_and_log() {
        let uniform = CascadeSplit::compute(&CascadeSplitScheme::Uniform, 4, 0.1, 100.0);
        let log = CascadeSplit::compute(&CascadeSplitScheme::Logarithmic, 4, 0.1, 100.0);
        let pssm =
            CascadeSplit::compute(&CascadeSplitScheme::Pssm { lambda: 0.5 }, 4, 0.1, 100.0);

        // PSSM first split should be between uniform and log.
        assert!(pssm.splits[1] >= log.splits[1] - 0.1);
        assert!(pssm.splits[1] <= uniform.splits[1] + 0.1);
    }

    #[test]
    fn cascade_selection() {
        let split = CascadeSplit::compute(&CascadeSplitScheme::Uniform, 4, 1.0, 100.0);
        assert_eq!(split.select_cascade(5.0), 0);
        assert_eq!(split.select_cascade(99.0), 3);
    }

    #[test]
    fn cascade_blend_at_boundary() {
        let split = CascadeSplit::compute(&CascadeSplitScheme::Uniform, 4, 1.0, 100.0);
        let far = split.cascade_far(0);
        let (cascade, blend) = split.cascade_blend(far - 0.5, 2.0);
        assert_eq!(cascade, 0);
        assert!(blend > 0.0);
    }

    #[test]
    fn compute_cascade_matrices_produces_correct_count() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let (matrices, splits) = compute_cascade_matrices(
            &view,
            &proj,
            Vec3::new(0.2, 1.0, 0.3).normalize(),
            4,
            &CascadeSplitScheme::default(),
            100.0,
            0.1,
            2048,
            true,
        );
        assert_eq!(matrices.len(), 4);
        assert_eq!(splits.cascade_count, 4);
    }
}
