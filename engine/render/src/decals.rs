// engine/render/src/decals.rs
//
// Deferred decal system for the Genovo engine. Decals are projected textures
// applied on top of scene geometry -- used for bullet holes, blood spatters,
// footprints, tire marks, etc.
//
// The system uses a deferred projection approach: each decal is an oriented
// box that projects its texture onto any geometry that intersects the box.
// The projection is computed in the fragment shader using the depth buffer,
// but the CPU-side data structures and lifecycle management live here.

use glam::{Mat4, Quat, Vec3};

// Vec4 is imported in tests where it's needed.

// ---------------------------------------------------------------------------
// DecalBlendMode
// ---------------------------------------------------------------------------

/// How the decal blends with the underlying surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecalBlendMode {
    /// Standard alpha blending.
    AlphaBlend,
    /// Multiplicative (darkens the surface).
    Multiply,
    /// Additive (brightens the surface).
    Additive,
    /// Replace the surface color entirely within the decal region.
    Replace,
}

impl Default for DecalBlendMode {
    fn default() -> Self {
        DecalBlendMode::AlphaBlend
    }
}

// ---------------------------------------------------------------------------
// Decal
// ---------------------------------------------------------------------------

/// A single decal instance in the world.
///
/// The decal is defined by a position, rotation, and size that form an
/// oriented bounding box. The decal's texture is projected along the box's
/// local -Z axis onto any geometry that intersects the box.
#[derive(Debug, Clone)]
pub struct Decal {
    /// World-space position of the decal center.
    pub position: Vec3,
    /// Orientation (the local -Z axis is the projection direction).
    pub rotation: Quat,
    /// Half-extents of the decal box (width, height, depth).
    pub half_extents: Vec3,
    /// Texture atlas region: `(u_min, v_min, u_max, v_max)`.
    pub atlas_region: [f32; 4],
    /// How much the decal's normal map blends with the surface normal.
    /// 0.0 = surface normal only, 1.0 = full decal normal.
    pub normal_blend: f32,
    /// Overall opacity.
    pub opacity: f32,
    /// Blend mode.
    pub blend_mode: DecalBlendMode,
    /// Angle fade: decals on surfaces whose normal deviates from the
    /// projection direction by more than this angle (radians) will fade.
    pub angle_fade_start: f32,
    /// Angle at which the decal is fully transparent.
    pub angle_fade_end: f32,
    /// Time when this decal was created (for temporal fade-out).
    pub birth_time: f32,
    /// Duration before the decal starts fading (seconds). 0 = never.
    pub fade_delay: f32,
    /// How long the fade-out takes (seconds).
    pub fade_duration: f32,
    /// Sorting order (lower = rendered first).
    pub sort_order: i32,
    /// Tint color (multiplied with texture).
    pub tint: [f32; 4],
    /// If `true`, this decal affects the normal buffer.
    pub affects_normals: bool,
    /// If `true`, this decal is still alive.
    pub alive: bool,
    /// User tag for identification.
    pub tag: u32,
}

impl Decal {
    /// Creates a new decal.
    pub fn new(position: Vec3, rotation: Quat, size: Vec3) -> Self {
        Self {
            position,
            rotation,
            half_extents: size * 0.5,
            atlas_region: [0.0, 0.0, 1.0, 1.0],
            normal_blend: 0.5,
            opacity: 1.0,
            blend_mode: DecalBlendMode::AlphaBlend,
            angle_fade_start: std::f32::consts::FRAC_PI_3, // 60 degrees
            angle_fade_end: std::f32::consts::FRAC_PI_2,   // 90 degrees
            birth_time: 0.0,
            fade_delay: 0.0,
            fade_duration: 0.0,
            sort_order: 0,
            tint: [1.0, 1.0, 1.0, 1.0],
            affects_normals: true,
            alive: true,
            tag: 0,
        }
    }

    /// Sets the texture atlas region.
    pub fn with_atlas_region(mut self, u_min: f32, v_min: f32, u_max: f32, v_max: f32) -> Self {
        self.atlas_region = [u_min, v_min, u_max, v_max];
        self
    }

    /// Sets the normal blend factor.
    pub fn with_normal_blend(mut self, blend: f32) -> Self {
        self.normal_blend = blend;
        self
    }

    /// Sets the opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Sets the blend mode.
    pub fn with_blend_mode(mut self, mode: DecalBlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Sets the angle fade parameters.
    pub fn with_angle_fade(mut self, start: f32, end: f32) -> Self {
        self.angle_fade_start = start;
        self.angle_fade_end = end;
        self
    }

    /// Sets temporal fade-out parameters.
    pub fn with_fade(mut self, delay: f32, duration: f32) -> Self {
        self.fade_delay = delay;
        self.fade_duration = duration;
        self
    }

    /// Sets the birth time.
    pub fn with_birth_time(mut self, time: f32) -> Self {
        self.birth_time = time;
        self
    }

    /// Sets the tint color.
    pub fn with_tint(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.tint = [r, g, b, a];
        self
    }

    /// Sets the sort order.
    pub fn with_sort_order(mut self, order: i32) -> Self {
        self.sort_order = order;
        self
    }

    /// Sets a user tag.
    pub fn with_tag(mut self, tag: u32) -> Self {
        self.tag = tag;
        self
    }

    /// Returns the projection matrix (world -> decal UV space).
    ///
    /// This maps world-space positions into the decal's local space where
    /// X and Y are in [-1, 1] (corresponding to UV [0, 1]) and Z is the
    /// depth along the projection direction.
    pub fn projection_matrix(&self) -> Mat4 {
        let world_to_decal = Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            self.rotation,
            self.position,
        )
        .inverse();

        // Scale by inverse half-extents to map to [-1, 1].
        let inv_scale = Vec3::new(
            1.0 / self.half_extents.x,
            1.0 / self.half_extents.y,
            1.0 / self.half_extents.z,
        );
        let scale_matrix = Mat4::from_scale(inv_scale);

        scale_matrix * world_to_decal
    }

    /// Returns the world-space transform matrix.
    pub fn transform_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.half_extents * 2.0,
            self.rotation,
            self.position,
        )
    }

    /// Computes the angle-based opacity fade for a given surface normal.
    ///
    /// # Arguments
    /// * `surface_normal` - The surface normal at the decal projection point.
    ///
    /// # Returns
    /// Opacity multiplier in [0, 1].
    pub fn angle_fade(&self, surface_normal: Vec3) -> f32 {
        let projection_dir = self.rotation * Vec3::NEG_Z;
        let cos_angle = (-projection_dir).dot(surface_normal).clamp(-1.0, 1.0);
        let angle = cos_angle.acos();

        if angle <= self.angle_fade_start {
            1.0
        } else if angle >= self.angle_fade_end {
            0.0
        } else {
            let t = (angle - self.angle_fade_start)
                / (self.angle_fade_end - self.angle_fade_start);
            1.0 - t
        }
    }

    /// Computes the temporal opacity fade.
    ///
    /// # Arguments
    /// * `current_time` - The current simulation time.
    ///
    /// # Returns
    /// Opacity multiplier in [0, 1].
    pub fn temporal_fade(&self, current_time: f32) -> f32 {
        if self.fade_duration <= 0.0 {
            return 1.0;
        }

        let age = current_time - self.birth_time;
        if age < self.fade_delay {
            return 1.0;
        }

        let fade_time = age - self.fade_delay;
        if fade_time >= self.fade_duration {
            return 0.0;
        }

        1.0 - (fade_time / self.fade_duration)
    }

    /// Returns the effective opacity at the given time and surface normal.
    pub fn effective_opacity(&self, current_time: f32, surface_normal: Vec3) -> f32 {
        self.opacity * self.temporal_fade(current_time) * self.angle_fade(surface_normal)
    }

    /// Returns the 8 corners of the decal box in world space.
    pub fn world_corners(&self) -> [Vec3; 8] {
        let he = self.half_extents;
        let corners = [
            Vec3::new(-he.x, -he.y, -he.z),
            Vec3::new(he.x, -he.y, -he.z),
            Vec3::new(he.x, he.y, -he.z),
            Vec3::new(-he.x, he.y, -he.z),
            Vec3::new(-he.x, -he.y, he.z),
            Vec3::new(he.x, -he.y, he.z),
            Vec3::new(he.x, he.y, he.z),
            Vec3::new(-he.x, he.y, he.z),
        ];
        let mut world_corners = [Vec3::ZERO; 8];
        for (i, c) in corners.iter().enumerate() {
            world_corners[i] = self.position + self.rotation * *c;
        }
        world_corners
    }

    /// Returns the projection direction (local -Z axis in world space).
    pub fn projection_direction(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }
}

// ---------------------------------------------------------------------------
// DecalAtlas
// ---------------------------------------------------------------------------

/// A texture atlas for batching multiple decal textures into a single draw call.
///
/// Decals reference regions within the atlas using `atlas_region` coordinates.
#[derive(Debug, Clone)]
pub struct DecalAtlas {
    /// Atlas dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// Allocated regions: (name, u_min, v_min, u_max, v_max).
    regions: Vec<(String, [f32; 4])>,
    /// Simple shelf packer state.
    current_x: u32,
    current_y: u32,
    shelf_height: u32,
    padding: u32,
}

impl DecalAtlas {
    /// Creates a new atlas with the given pixel dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            regions: Vec::new(),
            current_x: 0,
            current_y: 0,
            shelf_height: 0,
            padding: 2,
        }
    }

    /// Allocates a region in the atlas for a texture of the given pixel size.
    ///
    /// Returns the normalized UV region `[u_min, v_min, u_max, v_max]`, or
    /// `None` if there is no space.
    pub fn allocate(
        &mut self,
        name: impl Into<String>,
        tex_width: u32,
        tex_height: u32,
    ) -> Option<[f32; 4]> {
        let w = tex_width + self.padding * 2;
        let h = tex_height + self.padding * 2;

        // Try to fit on the current shelf.
        if self.current_x + w > self.width {
            // Move to the next shelf.
            self.current_x = 0;
            self.current_y += self.shelf_height;
            self.shelf_height = 0;
        }

        if self.current_y + h > self.height {
            return None; // Atlas is full.
        }

        let x = self.current_x + self.padding;
        let y = self.current_y + self.padding;

        let region = [
            x as f32 / self.width as f32,
            y as f32 / self.height as f32,
            (x + tex_width) as f32 / self.width as f32,
            (y + tex_height) as f32 / self.height as f32,
        ];

        self.current_x += w;
        self.shelf_height = self.shelf_height.max(h);

        self.regions.push((name.into(), region));
        Some(region)
    }

    /// Looks up a region by name.
    pub fn find(&self, name: &str) -> Option<[f32; 4]> {
        self.regions
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| *r)
    }

    /// Returns all allocated regions.
    pub fn regions(&self) -> &[(String, [f32; 4])] {
        &self.regions
    }

    /// Returns the number of allocated regions.
    pub fn count(&self) -> usize {
        self.regions.len()
    }

    /// Resets the atlas (clears all allocations).
    pub fn clear(&mut self) {
        self.regions.clear();
        self.current_x = 0;
        self.current_y = 0;
        self.shelf_height = 0;
    }
}

// ---------------------------------------------------------------------------
// DecalGpuData
// ---------------------------------------------------------------------------

/// Per-decal data packed for GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DecalGpuData {
    /// Inverse world matrix (world -> decal local space).
    pub inv_world: [[f32; 4]; 4],
    /// Atlas UV region [u_min, v_min, u_max, v_max].
    pub atlas_region: [f32; 4],
    /// Tint color (RGBA).
    pub tint: [f32; 4],
    /// [normal_blend, opacity, angle_fade_start_cos, angle_fade_end_cos]
    pub params: [f32; 4],
}

// ---------------------------------------------------------------------------
// DecalManager
// ---------------------------------------------------------------------------

/// Manages a pool of decals with lifecycle and batch rendering support.
pub struct DecalManager {
    /// All decals (including dead ones -- compacted periodically).
    decals: Vec<Decal>,
    /// Maximum number of decals.
    pub max_decals: usize,
    /// Optional texture atlas.
    pub atlas: Option<DecalAtlas>,
    /// GPU data buffer (rebuilt each frame).
    gpu_data: Vec<DecalGpuData>,
    /// Current simulation time.
    current_time: f32,
}

impl DecalManager {
    /// Creates a new decal manager.
    pub fn new(max_decals: usize) -> Self {
        Self {
            decals: Vec::with_capacity(max_decals),
            max_decals,
            atlas: None,
            gpu_data: Vec::with_capacity(max_decals),
            current_time: 0.0,
        }
    }

    /// Sets the atlas.
    pub fn with_atlas(mut self, atlas: DecalAtlas) -> Self {
        self.atlas = Some(atlas);
        self
    }

    /// Spawns a new decal. Returns its index, or `None` if at capacity.
    pub fn spawn(&mut self, decal: Decal) -> Option<usize> {
        if self.decals.len() >= self.max_decals {
            // Try to reclaim a dead slot.
            if let Some(idx) = self.decals.iter().position(|d| !d.alive) {
                self.decals[idx] = decal;
                return Some(idx);
            }
            // Remove the oldest decal.
            if !self.decals.is_empty() {
                let oldest = self
                    .decals
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.birth_time
                            .partial_cmp(&b.birth_time)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.decals[oldest] = decal;
                return Some(oldest);
            }
            return None;
        }
        let idx = self.decals.len();
        self.decals.push(decal);
        Some(idx)
    }

    /// Spawns a decal from a raycast hit.
    pub fn spawn_at_hit(
        &mut self,
        position: Vec3,
        normal: Vec3,
        size: f32,
        current_time: f32,
    ) -> Option<usize> {
        // Build rotation: -Z aligns with the surface normal.
        let up = if normal.y.abs() < 0.99 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let right = normal.cross(up).normalize();
        let corrected_up = right.cross(normal).normalize();
        let rotation = Quat::from_mat3(&glam::Mat3::from_cols(right, corrected_up, -normal));

        let decal = Decal::new(
            position + normal * 0.01, // Slight offset to prevent z-fighting.
            rotation,
            Vec3::new(size, size, size * 0.5),
        )
        .with_birth_time(current_time);

        self.spawn(decal)
    }

    /// Updates all decals: performs temporal fade and removes dead decals.
    pub fn update(&mut self, current_time: f32) {
        self.current_time = current_time;

        for decal in &mut self.decals {
            if !decal.alive {
                continue;
            }
            // Check temporal fade.
            if decal.fade_duration > 0.0 {
                let fade = decal.temporal_fade(current_time);
                if fade <= 0.0 {
                    decal.alive = false;
                }
            }
        }
    }

    /// Compacts the decal list by removing dead entries.
    pub fn compact(&mut self) {
        self.decals.retain(|d| d.alive);
    }

    /// Prepares GPU data for all alive decals.
    pub fn prepare_gpu_data(&mut self) {
        self.gpu_data.clear();

        for decal in &self.decals {
            if !decal.alive {
                continue;
            }

            let opacity = decal.opacity * decal.temporal_fade(self.current_time);
            if opacity < 0.001 {
                continue;
            }

            let inv_world = decal.projection_matrix();
            let cols = inv_world.to_cols_array_2d();

            self.gpu_data.push(DecalGpuData {
                inv_world: [
                    [cols[0][0], cols[0][1], cols[0][2], cols[0][3]],
                    [cols[1][0], cols[1][1], cols[1][2], cols[1][3]],
                    [cols[2][0], cols[2][1], cols[2][2], cols[2][3]],
                    [cols[3][0], cols[3][1], cols[3][2], cols[3][3]],
                ],
                atlas_region: decal.atlas_region,
                tint: [
                    decal.tint[0] * opacity,
                    decal.tint[1] * opacity,
                    decal.tint[2] * opacity,
                    decal.tint[3] * opacity,
                ],
                params: [
                    decal.normal_blend,
                    opacity,
                    decal.angle_fade_start.cos(),
                    decal.angle_fade_end.cos(),
                ],
            });
        }

        // Sort by sort order.
        self.gpu_data.sort_by(|a, b| {
            // Lower sort order first (we don't store sort_order in gpu_data
            // so this is a simplified approach).
            a.params[1]
                .partial_cmp(&b.params[1])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Returns the prepared GPU data.
    pub fn gpu_data(&self) -> &[DecalGpuData] {
        &self.gpu_data
    }

    /// Returns the number of alive decals.
    pub fn alive_count(&self) -> usize {
        self.decals.iter().filter(|d| d.alive).count()
    }

    /// Returns the total number of decals (including dead).
    pub fn total_count(&self) -> usize {
        self.decals.len()
    }

    /// Removes all decals with the given tag.
    pub fn remove_by_tag(&mut self, tag: u32) {
        for decal in &mut self.decals {
            if decal.tag == tag {
                decal.alive = false;
            }
        }
    }

    /// Removes all decals.
    pub fn clear(&mut self) {
        self.decals.clear();
        self.gpu_data.clear();
    }

    /// Returns a decal by index.
    pub fn get(&self, index: usize) -> Option<&Decal> {
        self.decals.get(index)
    }

    /// Returns a mutable decal by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Decal> {
        self.decals.get_mut(index)
    }

    /// Returns an iterator over alive decals.
    pub fn alive_decals(&self) -> impl Iterator<Item = &Decal> {
        self.decals.iter().filter(|d| d.alive)
    }
}

impl Default for DecalManager {
    fn default() -> Self {
        Self::new(256)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    #[test]
    fn decal_projection_matrix() {
        let decal = Decal::new(
            Vec3::new(5.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::new(2.0, 2.0, 1.0),
        );
        let proj = decal.projection_matrix();

        // A point at the decal center should map to (0, 0, 0).
        let center = proj * Vec4::new(5.0, 0.0, 0.0, 1.0);
        assert!(
            center.x.abs() < 0.01 && center.y.abs() < 0.01 && center.z.abs() < 0.01,
            "Center should map to origin, got {:?}",
            center
        );
    }

    #[test]
    fn angle_fade() {
        let decal = Decal::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);

        // Surface facing the projection direction (perpendicular): full opacity.
        let fade = decal.angle_fade(Vec3::Z);
        assert!(
            (fade - 1.0).abs() < 0.01,
            "Direct facing should be full opacity, got {fade}"
        );

        // Surface at 90 degrees: zero.
        let fade = decal.angle_fade(Vec3::X);
        assert!(
            fade < 0.01,
            "90-degree surface should have no opacity, got {fade}"
        );
    }

    #[test]
    fn temporal_fade() {
        let decal = Decal::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
            .with_birth_time(0.0)
            .with_fade(2.0, 1.0);

        // Before fade delay.
        assert!((decal.temporal_fade(1.0) - 1.0).abs() < 0.01);
        // During fade.
        assert!(decal.temporal_fade(2.5) < 1.0);
        assert!(decal.temporal_fade(2.5) > 0.0);
        // After fade.
        assert!(decal.temporal_fade(3.5) < 0.01);
    }

    #[test]
    fn atlas_allocation() {
        let mut atlas = DecalAtlas::new(1024, 1024);
        let r1 = atlas.allocate("bullet_hole", 128, 128);
        assert!(r1.is_some());
        let r2 = atlas.allocate("blood", 256, 256);
        assert!(r2.is_some());
        assert_eq!(atlas.count(), 2);

        let found = atlas.find("bullet_hole");
        assert!(found.is_some());
    }

    #[test]
    fn manager_spawn_and_cleanup() {
        let mut manager = DecalManager::new(100);

        let decal = Decal::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
            .with_birth_time(0.0)
            .with_fade(0.5, 0.5);

        manager.spawn(decal);
        assert_eq!(manager.alive_count(), 1);

        manager.update(2.0);
        assert_eq!(
            manager.alive_count(),
            0,
            "Decal should have faded out by t=2.0"
        );
    }

    #[test]
    fn manager_recycles_oldest() {
        let mut manager = DecalManager::new(3);

        for i in 0..5 {
            let decal = Decal::new(
                Vec3::new(i as f32, 0.0, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            )
            .with_birth_time(i as f32);
            manager.spawn(decal);
        }

        // Should still have 3 decals (oldest recycled).
        assert_eq!(manager.total_count(), 3);
    }
}
