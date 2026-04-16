// engine/render/src/lighting/light_clustering.rs
//
// Clustered forward rendering: the view frustum is divided into a 3D grid of
// clusters, and each cluster stores the indices of lights that overlap it.
// During the fragment shader, the pixel determines its cluster from its
// screen-space position and depth, then only evaluates the lights assigned
// to that cluster.

use super::light_culling::Aabb;
use super::light_types::Light;
use glam::{Mat4, Vec3, Vec4};
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// ClusterSettings
// ---------------------------------------------------------------------------

/// Configuration for the cluster grid.
#[derive(Debug, Clone, Copy)]
pub struct ClusterSettings {
    /// Number of clusters along the X axis (screen width).
    pub x_count: u32,
    /// Number of clusters along the Y axis (screen height).
    pub y_count: u32,
    /// Number of clusters along the Z axis (depth, logarithmic).
    pub z_count: u32,
    /// Near plane of the camera.
    pub near: f32,
    /// Far plane of the camera.
    pub far: f32,
    /// Maximum number of lights per cluster.
    pub max_lights_per_cluster: u32,
}

impl Default for ClusterSettings {
    fn default() -> Self {
        Self {
            x_count: 16,
            y_count: 9,
            z_count: 24,
            near: 0.1,
            far: 100.0,
            max_lights_per_cluster: 128,
        }
    }
}

impl ClusterSettings {
    /// Total number of clusters.
    pub fn total_clusters(&self) -> u32 {
        self.x_count * self.y_count * self.z_count
    }

    /// Compute the Z slice index for a given view-space depth.
    ///
    /// Uses logarithmic slicing: z_slice = log(depth/near) / log(far/near) * z_count
    pub fn z_slice_for_depth(&self, view_depth: f32) -> u32 {
        if view_depth <= self.near {
            return 0;
        }
        if view_depth >= self.far {
            return self.z_count - 1;
        }
        let log_ratio = (view_depth / self.near).ln() / (self.far / self.near).ln();
        let slice = (log_ratio * self.z_count as f32).floor() as u32;
        slice.min(self.z_count - 1)
    }

    /// Compute the near depth of a given Z slice.
    pub fn slice_near_depth(&self, slice: u32) -> f32 {
        self.near * (self.far / self.near).powf(slice as f32 / self.z_count as f32)
    }

    /// Compute the far depth of a given Z slice.
    pub fn slice_far_depth(&self, slice: u32) -> f32 {
        self.near * (self.far / self.near).powf((slice + 1) as f32 / self.z_count as f32)
    }

    /// Convert a 3D cluster index to a linear index.
    #[inline]
    pub fn to_linear_index(&self, x: u32, y: u32, z: u32) -> u32 {
        z * self.x_count * self.y_count + y * self.x_count + x
    }

    /// Convert a linear index to a 3D cluster index.
    #[inline]
    pub fn from_linear_index(&self, index: u32) -> (u32, u32, u32) {
        let z = index / (self.x_count * self.y_count);
        let rem = index % (self.x_count * self.y_count);
        let y = rem / self.x_count;
        let x = rem % self.x_count;
        (x, y, z)
    }
}

// ---------------------------------------------------------------------------
// ClusterGrid
// ---------------------------------------------------------------------------

/// A 3D grid of clusters, each containing a list of light indices that
/// overlap with that cluster.
pub struct ClusterGrid {
    /// Cluster settings.
    pub settings: ClusterSettings,
    /// Per-cluster light lists. Index with `settings.to_linear_index()`.
    pub clusters: Vec<SmallVec<[u16; 32]>>,
    /// AABB for each cluster in view space.
    pub cluster_aabbs: Vec<Aabb>,
    /// Total number of light-cluster assignments.
    pub total_assignments: u32,
}

impl ClusterGrid {
    /// Create an empty cluster grid.
    pub fn new(settings: ClusterSettings) -> Self {
        let total = settings.total_clusters() as usize;
        let clusters = vec![SmallVec::new(); total];
        let cluster_aabbs = vec![
            Aabb::from_min_max(Vec3::ZERO, Vec3::ZERO);
            total
        ];
        Self {
            settings,
            clusters,
            cluster_aabbs,
            total_assignments: 0,
        }
    }

    /// Clear all cluster light lists for a new frame.
    pub fn clear(&mut self) {
        for cluster in &mut self.clusters {
            cluster.clear();
        }
        self.total_assignments = 0;
    }

    /// Get the light list for a cluster.
    pub fn get_lights(&self, x: u32, y: u32, z: u32) -> &[u16] {
        let idx = self.settings.to_linear_index(x, y, z) as usize;
        &self.clusters[idx]
    }

    /// Get the light list for a cluster by linear index.
    pub fn get_lights_linear(&self, linear_idx: u32) -> &[u16] {
        &self.clusters[linear_idx as usize]
    }

    /// Average number of lights per non-empty cluster.
    pub fn avg_lights_per_cluster(&self) -> f32 {
        let non_empty: usize = self.clusters.iter().filter(|c| !c.is_empty()).count();
        if non_empty == 0 {
            return 0.0;
        }
        self.total_assignments as f32 / non_empty as f32
    }

    /// Maximum number of lights in any single cluster.
    pub fn max_lights_in_cluster(&self) -> usize {
        self.clusters.iter().map(|c| c.len()).max().unwrap_or(0)
    }

    /// Number of non-empty clusters.
    pub fn non_empty_clusters(&self) -> usize {
        self.clusters.iter().filter(|c| !c.is_empty()).count()
    }

    /// Build GPU buffer data for the cluster grid.
    ///
    /// Returns two buffers:
    /// 1. Cluster offset/count table: for each cluster, the offset into the
    ///    light index buffer and the number of lights.
    /// 2. Light index buffer: packed light indices referenced by clusters.
    pub fn build_gpu_buffers(&self) -> (Vec<u32>, Vec<u32>) {
        let total = self.settings.total_clusters() as usize;

        // Offset/count table: 2 u32 per cluster (offset, count).
        let mut offset_count = Vec::with_capacity(total * 2);
        // Light index buffer.
        let mut light_indices = Vec::new();

        let mut current_offset = 0u32;
        for cluster in &self.clusters {
            offset_count.push(current_offset);
            offset_count.push(cluster.len() as u32);
            for &light_idx in cluster.iter() {
                light_indices.push(light_idx as u32);
            }
            current_offset += cluster.len() as u32;
        }

        (offset_count, light_indices)
    }
}

// ---------------------------------------------------------------------------
// Cluster AABB computation
// ---------------------------------------------------------------------------

/// Compute the view-space AABB for a cluster.
///
/// # Arguments
/// - `x`, `y`, `z` — cluster indices.
/// - `settings` — cluster grid settings.
/// - `inv_proj` — inverse projection matrix.
/// - `screen_width`, `screen_height` — viewport dimensions.
fn compute_cluster_aabb(
    x: u32,
    y: u32,
    z: u32,
    settings: &ClusterSettings,
    inv_proj: &Mat4,
    screen_width: f32,
    screen_height: f32,
) -> Aabb {
    let tile_w = screen_width / settings.x_count as f32;
    let tile_h = screen_height / settings.y_count as f32;

    // Screen-space corners of the cluster tile.
    let ss_min_x = x as f32 * tile_w;
    let ss_max_x = (x + 1) as f32 * tile_w;
    let ss_min_y = y as f32 * tile_h;
    let ss_max_y = (y + 1) as f32 * tile_h;

    // Depth range of the cluster (logarithmic).
    let z_near = settings.slice_near_depth(z);
    let z_far = settings.slice_far_depth(z);

    // Convert screen-space to NDC (-1..1).
    let ndc_min_x = (ss_min_x / screen_width) * 2.0 - 1.0;
    let ndc_max_x = (ss_max_x / screen_width) * 2.0 - 1.0;
    let ndc_min_y = 1.0 - (ss_max_y / screen_height) * 2.0; // Y flipped
    let ndc_max_y = 1.0 - (ss_min_y / screen_height) * 2.0;

    // Unproject the 8 corners of the cluster box from NDC to view space.
    let corners = [
        unproject_point(inv_proj, ndc_min_x, ndc_min_y, z_near),
        unproject_point(inv_proj, ndc_max_x, ndc_min_y, z_near),
        unproject_point(inv_proj, ndc_min_x, ndc_max_y, z_near),
        unproject_point(inv_proj, ndc_max_x, ndc_max_y, z_near),
        unproject_point(inv_proj, ndc_min_x, ndc_min_y, z_far),
        unproject_point(inv_proj, ndc_max_x, ndc_min_y, z_far),
        unproject_point(inv_proj, ndc_min_x, ndc_max_y, z_far),
        unproject_point(inv_proj, ndc_max_x, ndc_max_y, z_far),
    ];

    let mut aabb_min = corners[0];
    let mut aabb_max = corners[0];
    for &c in &corners[1..] {
        aabb_min = aabb_min.min(c);
        aabb_max = aabb_max.max(c);
    }

    Aabb::from_min_max(aabb_min, aabb_max)
}

/// Unproject a point from NDC to view space.
///
/// For a perspective projection, this maps (ndc_x, ndc_y, view_depth) to
/// a view-space position by scaling ndc by the depth.
fn unproject_point(inv_proj: &Mat4, ndc_x: f32, ndc_y: f32, view_depth: f32) -> Vec3 {
    // Map view_depth to NDC Z. For RH: ndc_z = (far*near / depth - far) / (near - far)
    // Simpler: project a point at the desired depth and read NDC Z.
    // For an approximation, we use the inverse projection.
    let clip = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let view = *inv_proj * clip;
    let view_dir = Vec3::new(view.x / view.w, view.y / view.w, view.z / view.w);

    // Scale so that the Z component matches the desired depth.
    if view_dir.z.abs() < 1e-7 {
        return Vec3::new(0.0, 0.0, -view_depth);
    }
    let t = -view_depth / view_dir.z;
    view_dir * t
}

// ---------------------------------------------------------------------------
// Build cluster grid
// ---------------------------------------------------------------------------

/// Build a cluster grid by assigning lights to clusters.
///
/// This is the main entry point for the clustered lighting system. It:
/// 1. Computes the view-space AABB for each cluster.
/// 2. Transforms each light to view space.
/// 3. Tests each light against each cluster for overlap.
///
/// # Arguments
/// - `view_matrix` — the camera view matrix.
/// - `projection` — the camera projection matrix.
/// - `lights` — the set of lights to assign.
/// - `screen_width`, `screen_height` — viewport dimensions.
/// - `settings` — cluster grid configuration.
pub fn build_cluster_grid(
    view_matrix: &Mat4,
    projection: &Mat4,
    lights: &[Light],
    screen_width: f32,
    screen_height: f32,
    settings: ClusterSettings,
) -> ClusterGrid {
    let inv_proj = projection.inverse();
    let total = settings.total_clusters() as usize;

    let mut grid = ClusterGrid::new(settings);

    // Pre-compute cluster AABBs.
    for z in 0..settings.z_count {
        for y in 0..settings.y_count {
            for x in 0..settings.x_count {
                let idx = settings.to_linear_index(x, y, z) as usize;
                grid.cluster_aabbs[idx] = compute_cluster_aabb(
                    x,
                    y,
                    z,
                    &settings,
                    &inv_proj,
                    screen_width,
                    screen_height,
                );
            }
        }
    }

    // Transform lights to view space and assign to clusters.
    for (light_idx, light) in lights.iter().enumerate() {
        if light_idx > u16::MAX as usize {
            break; // Can't store more than 65535 light indices.
        }

        match light {
            Light::Directional(_) => {
                // Directional lights affect all clusters.
                for cluster in &mut grid.clusters {
                    if cluster.len() < settings.max_lights_per_cluster as usize {
                        cluster.push(light_idx as u16);
                    }
                }
                grid.total_assignments += total as u32;
            }
            Light::Point(p) => {
                // Transform point light to view space.
                let view_pos = (*view_matrix * Vec4::new(p.position.x, p.position.y, p.position.z, 1.0)).truncate();

                // Find the Z range of the light's bounding sphere.
                let z_min = (-view_pos.z - p.radius).max(settings.near);
                let z_max = (-view_pos.z + p.radius).min(settings.far);
                if z_min > settings.far || z_max < settings.near {
                    continue;
                }

                let z_start = settings.z_slice_for_depth(z_min);
                let z_end = settings.z_slice_for_depth(z_max);

                for z in z_start..=z_end {
                    for y in 0..settings.y_count {
                        for x in 0..settings.x_count {
                            let idx = settings.to_linear_index(x, y, z) as usize;
                            let aabb = &grid.cluster_aabbs[idx];
                            if aabb.intersects_sphere(view_pos, p.radius) {
                                if grid.clusters[idx].len()
                                    < settings.max_lights_per_cluster as usize
                                {
                                    grid.clusters[idx].push(light_idx as u16);
                                    grid.total_assignments += 1;
                                }
                            }
                        }
                    }
                }
            }
            Light::Spot(s) => {
                // Transform spot light to view space.
                let view_pos = (*view_matrix * Vec4::new(s.position.x, s.position.y, s.position.z, 1.0)).truncate();

                // Conservative: use the spot's bounding sphere.
                let bounding_radius = s.range;

                let z_min = (-view_pos.z - bounding_radius).max(settings.near);
                let z_max = (-view_pos.z + bounding_radius).min(settings.far);
                if z_min > settings.far || z_max < settings.near {
                    continue;
                }

                let z_start = settings.z_slice_for_depth(z_min);
                let z_end = settings.z_slice_for_depth(z_max);

                for z in z_start..=z_end {
                    for y in 0..settings.y_count {
                        for x in 0..settings.x_count {
                            let idx = settings.to_linear_index(x, y, z) as usize;
                            let aabb = &grid.cluster_aabbs[idx];
                            if aabb.intersects_sphere(view_pos, bounding_radius) {
                                if grid.clusters[idx].len()
                                    < settings.max_lights_per_cluster as usize
                                {
                                    grid.clusters[idx].push(light_idx as u16);
                                    grid.total_assignments += 1;
                                }
                            }
                        }
                    }
                }
            }
            Light::Area(a) => {
                // Area lights use their bounding sphere.
                let view_pos = (*view_matrix * Vec4::new(a.position.x, a.position.y, a.position.z, 1.0)).truncate();
                let bounding_radius = a.range;

                let z_min = (-view_pos.z - bounding_radius).max(settings.near);
                let z_max = (-view_pos.z + bounding_radius).min(settings.far);
                if z_min > settings.far || z_max < settings.near {
                    continue;
                }

                let z_start = settings.z_slice_for_depth(z_min);
                let z_end = settings.z_slice_for_depth(z_max);

                for z in z_start..=z_end {
                    for y in 0..settings.y_count {
                        for x in 0..settings.x_count {
                            let idx = settings.to_linear_index(x, y, z) as usize;
                            let aabb = &grid.cluster_aabbs[idx];
                            if aabb.intersects_sphere(view_pos, bounding_radius) {
                                if grid.clusters[idx].len()
                                    < settings.max_lights_per_cluster as usize
                                {
                                    grid.clusters[idx].push(light_idx as u16);
                                    grid.total_assignments += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    grid
}

// ---------------------------------------------------------------------------
// GPU buffer layouts
// ---------------------------------------------------------------------------

/// GPU-compatible cluster info for a single cluster.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterInfoGpu {
    /// Offset into the light index buffer.
    pub offset: u32,
    /// Number of lights in this cluster.
    pub count: u32,
}

/// GPU-compatible header for the cluster grid.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterGridHeader {
    /// xyz = cluster counts, w = total clusters.
    pub dimensions: [u32; 4],
    /// x = near, y = far, z = log(far/near), w = 1.0/log(far/near).
    pub depth_params: [f32; 4],
    /// x = screen_width, y = screen_height, z = tile_width, w = tile_height.
    pub screen_params: [f32; 4],
}

impl ClusterGridHeader {
    /// Build the header from settings and screen dimensions.
    pub fn from_settings(settings: &ClusterSettings, screen_width: f32, screen_height: f32) -> Self {
        let log_ratio = (settings.far / settings.near).ln();
        Self {
            dimensions: [
                settings.x_count,
                settings.y_count,
                settings.z_count,
                settings.total_clusters(),
            ],
            depth_params: [settings.near, settings.far, log_ratio, 1.0 / log_ratio],
            screen_params: [
                screen_width,
                screen_height,
                screen_width / settings.x_count as f32,
                screen_height / settings.y_count as f32,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighting::light_types::DirectionalLight;

    #[test]
    fn cluster_index_roundtrip() {
        let settings = ClusterSettings::default();
        for z in 0..settings.z_count {
            for y in 0..settings.y_count {
                for x in 0..settings.x_count {
                    let linear = settings.to_linear_index(x, y, z);
                    let (rx, ry, rz) = settings.from_linear_index(linear);
                    assert_eq!((x, y, z), (rx, ry, rz));
                }
            }
        }
    }

    #[test]
    fn z_slice_near_far() {
        let settings = ClusterSettings {
            near: 0.1,
            far: 100.0,
            z_count: 24,
            ..Default::default()
        };
        assert_eq!(settings.z_slice_for_depth(0.05), 0);
        assert_eq!(settings.z_slice_for_depth(100.0), 23);
        assert_eq!(settings.z_slice_for_depth(200.0), 23);
    }

    #[test]
    fn z_slices_are_monotonic() {
        let settings = ClusterSettings::default();
        let mut prev = 0u32;
        for i in 1..100 {
            let depth = settings.near + (settings.far - settings.near) * (i as f32 / 100.0);
            let slice = settings.z_slice_for_depth(depth);
            assert!(slice >= prev);
            prev = slice;
        }
    }

    #[test]
    fn directional_light_fills_all_clusters() {
        let settings = ClusterSettings {
            x_count: 2,
            y_count: 2,
            z_count: 2,
            ..Default::default()
        };
        let view = Mat4::IDENTITY;
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let lights = vec![DirectionalLight::sun().to_light()];
        let grid = build_cluster_grid(&view, &proj, &lights, 800.0, 600.0, settings);
        // Every cluster should have the directional light.
        for cluster in &grid.clusters {
            assert_eq!(cluster.len(), 1);
        }
    }

    #[test]
    fn gpu_buffer_roundtrip() {
        let settings = ClusterSettings {
            x_count: 2,
            y_count: 2,
            z_count: 2,
            ..Default::default()
        };
        let grid = ClusterGrid::new(settings);
        let (offsets, indices) = grid.build_gpu_buffers();
        assert_eq!(offsets.len(), settings.total_clusters() as usize * 2);
        assert!(indices.is_empty());
    }
}
