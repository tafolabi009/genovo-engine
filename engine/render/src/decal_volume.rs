// engine/render/src/decal_volume.rs
//
// Deferred decal volume projection system for the Genovo engine.
//
// Unlike the basic decal system in `decals.rs` (which tracks individual decals
// with atlas regions), this module implements a full deferred decal projection
// pipeline using oriented bounding boxes (OBBs) rendered as box geometry in a
// deferred pass.
//
// # How it works
//
// 1. Each decal is an OBB defined by a world matrix and half-extents.
// 2. During the deferred pass, the OBB is rasterized as box geometry.
// 3. For each fragment, the world position is reconstructed from the depth
//    buffer.
// 4. The world position is transformed into decal local space. If the point
//    lies within [-1,1]^3, it is inside the decal volume.
// 5. The XY coordinates in decal space are used as UVs to sample the decal's
//    albedo and normal textures.
// 6. An angle-based fade is applied: `dot(surface_normal, decal_forward)` is
//    used to fade the decal on steep surfaces.
// 7. The result is blended into the G-buffer (albedo, normal, roughness).
//
// # Architecture
//
// ```text
//  DecalVolumeManager
//    |-- pool of DecalVolume instances (max capacity, oldest recycled)
//    |-- spawn / despawn / age / frustum cull
//    |
//    +-> DecalVolumeRenderer
//          |-- sort by priority
//          |-- build GPU data (inverse world matrix, UVs, params)
//          |-- render OBB box geometry per decal
//          |-- fragment shader: reconstruct world pos, project, sample, blend
// ```

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of decal volumes that can exist simultaneously.
pub const MAX_DECAL_VOLUMES: usize = 512;

/// Maximum number of decal volumes rendered per frame (GPU budget).
pub const MAX_RENDERED_PER_FRAME: usize = 256;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-7;

// ---------------------------------------------------------------------------
// NormalBlendMode
// ---------------------------------------------------------------------------

/// How the decal volume's normal map blends with the surface normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormalBlendMode {
    /// No normal blending -- decal does not affect normals.
    None,
    /// Reoriented normal mapping (RNM). The decal normal is reoriented to
    /// the surface tangent frame. This is the most physically correct method.
    Reoriented,
    /// Partial derivative blending (UDN). Cheaper, works well for subtle
    /// normal details.
    PartialDerivative,
    /// Simple linear blend between surface and decal normals.
    Linear,
    /// Full replacement -- the decal normal completely overrides the surface.
    Replace,
}

impl Default for NormalBlendMode {
    fn default() -> Self {
        NormalBlendMode::Reoriented
    }
}

// ---------------------------------------------------------------------------
// DecalVolumeId
// ---------------------------------------------------------------------------

/// Opaque identifier for a decal volume in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecalVolumeId {
    /// Index into the pool.
    pub index: u32,
    /// Generation counter for ABA prevention.
    pub generation: u32,
}

impl DecalVolumeId {
    /// Creates a new ID.
    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Returns an invalid ID sentinel.
    pub fn invalid() -> Self {
        Self {
            index: u32::MAX,
            generation: 0,
        }
    }

    /// Returns true if this ID is the invalid sentinel.
    pub fn is_invalid(&self) -> bool {
        self.index == u32::MAX
    }
}

// ---------------------------------------------------------------------------
// DecalVolumeMaterial
// ---------------------------------------------------------------------------

/// Material data for a decal volume.
#[derive(Debug, Clone)]
pub struct DecalVolumeMaterial {
    /// Handle to the albedo texture (opaque asset ID).
    pub albedo_texture: u64,
    /// Handle to the normal map texture.
    pub normal_texture: u64,
    /// Handle to the roughness/metalness texture (optional).
    pub orm_texture: u64,
    /// Handle to the emissive texture (optional, 0 = none).
    pub emissive_texture: u64,
    /// Base color tint (RGBA).
    pub base_color: Vec4,
    /// Roughness override (used if orm_texture is 0).
    pub roughness: f32,
    /// Metalness override (used if orm_texture is 0).
    pub metalness: f32,
    /// Emissive color multiplier.
    pub emissive_color: Vec3,
    /// UV scale (tiling within the decal volume).
    pub uv_scale: Vec2,
    /// UV offset.
    pub uv_offset: Vec2,
}

impl Default for DecalVolumeMaterial {
    fn default() -> Self {
        Self {
            albedo_texture: 0,
            normal_texture: 0,
            orm_texture: 0,
            emissive_texture: 0,
            base_color: Vec4::ONE,
            roughness: 0.5,
            metalness: 0.0,
            emissive_color: Vec3::ZERO,
            uv_scale: Vec2::ONE,
            uv_offset: Vec2::ZERO,
        }
    }
}

impl DecalVolumeMaterial {
    /// Creates a simple material with just an albedo texture.
    pub fn with_albedo(texture: u64) -> Self {
        Self {
            albedo_texture: texture,
            ..Default::default()
        }
    }

    /// Creates a material with albedo and normal textures.
    pub fn with_albedo_and_normal(albedo: u64, normal: u64) -> Self {
        Self {
            albedo_texture: albedo,
            normal_texture: normal,
            ..Default::default()
        }
    }

    /// Sets the base color tint.
    pub fn with_color(mut self, color: Vec4) -> Self {
        self.base_color = color;
        self
    }

    /// Sets the UV scale.
    pub fn with_uv_scale(mut self, scale: Vec2) -> Self {
        self.uv_scale = scale;
        self
    }
}

// ---------------------------------------------------------------------------
// DecalVolume
// ---------------------------------------------------------------------------

/// A deferred decal volume defined by an oriented bounding box (OBB).
///
/// The decal projects along its local -Z axis. The local XY plane corresponds
/// to the decal's UV space: local X maps to U, local Y maps to V.
#[derive(Debug, Clone)]
pub struct DecalVolume {
    /// World-space position of the decal center.
    pub position: Vec3,
    /// Orientation quaternion.
    pub rotation: Quat,
    /// Half-extents of the OBB (width/2, height/2, depth/2).
    pub half_extents: Vec3,
    /// Material data.
    pub material: DecalVolumeMaterial,
    /// How normals are blended.
    pub normal_blend_mode: NormalBlendMode,
    /// Normal blend strength [0, 1].
    pub normal_blend_strength: f32,
    /// Angle (radians) at which the decal starts fading on steep surfaces.
    pub angle_fade_start: f32,
    /// Angle (radians) at which the decal is fully transparent.
    pub angle_fade_end: f32,
    /// Overall opacity [0, 1].
    pub opacity: f32,
    /// Remaining lifetime in seconds. Negative or zero means immortal.
    pub lifetime: f32,
    /// Current age in seconds.
    pub age: f32,
    /// Fade-out duration at end of life (seconds).
    pub fade_out_duration: f32,
    /// Sort priority: lower values are rendered first (behind higher values).
    pub sort_priority: i32,
    /// Whether this decal affects the albedo channel.
    pub affects_albedo: bool,
    /// Whether this decal affects the normal channel.
    pub affects_normals: bool,
    /// Whether this decal affects the roughness/metalness channels.
    pub affects_orm: bool,
    /// Whether this decal is currently alive.
    pub alive: bool,
    /// Generation counter for ID validation.
    pub(crate) generation: u32,
    /// Optional user tag for filtering/identification.
    pub tag: u32,
}

impl DecalVolume {
    /// Creates a new decal volume.
    pub fn new(position: Vec3, rotation: Quat, half_extents: Vec3) -> Self {
        Self {
            position,
            rotation,
            half_extents,
            material: DecalVolumeMaterial::default(),
            normal_blend_mode: NormalBlendMode::default(),
            normal_blend_strength: 1.0,
            angle_fade_start: std::f32::consts::FRAC_PI_3, // 60 degrees
            angle_fade_end: std::f32::consts::FRAC_PI_2,   // 90 degrees
            opacity: 1.0,
            lifetime: -1.0,
            age: 0.0,
            fade_out_duration: 0.5,
            sort_priority: 0,
            affects_albedo: true,
            affects_normals: true,
            affects_orm: false,
            alive: true,
            generation: 0,
            tag: 0,
        }
    }

    /// Builder: sets the material.
    pub fn with_material(mut self, material: DecalVolumeMaterial) -> Self {
        self.material = material;
        self
    }

    /// Builder: sets the normal blend mode.
    pub fn with_normal_blend(mut self, mode: NormalBlendMode, strength: f32) -> Self {
        self.normal_blend_mode = mode;
        self.normal_blend_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Builder: sets angle fade parameters.
    pub fn with_angle_fade(mut self, start_radians: f32, end_radians: f32) -> Self {
        self.angle_fade_start = start_radians;
        self.angle_fade_end = end_radians;
        self
    }

    /// Builder: sets the opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Builder: sets the lifetime in seconds (0 or negative = immortal).
    pub fn with_lifetime(mut self, seconds: f32) -> Self {
        self.lifetime = seconds;
        self
    }

    /// Builder: sets the fade-out duration.
    pub fn with_fade_out(mut self, duration: f32) -> Self {
        self.fade_out_duration = duration.max(0.0);
        self
    }

    /// Builder: sets the sort priority.
    pub fn with_sort_priority(mut self, priority: i32) -> Self {
        self.sort_priority = priority;
        self
    }

    /// Builder: sets the tag.
    pub fn with_tag(mut self, tag: u32) -> Self {
        self.tag = tag;
        self
    }

    /// Builder: sets which G-buffer channels are affected.
    pub fn with_channels(mut self, albedo: bool, normals: bool, orm: bool) -> Self {
        self.affects_albedo = albedo;
        self.affects_normals = normals;
        self.affects_orm = orm;
        self
    }

    // -----------------------------------------------------------------------
    // Transform helpers
    // -----------------------------------------------------------------------

    /// Returns the world-space transform matrix (decal local -> world).
    pub fn world_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.half_extents * 2.0,
            self.rotation,
            self.position,
        )
    }

    /// Returns the inverse world matrix (world -> decal local space, normalized
    /// to [-1,1]^3).
    ///
    /// A world-space point `p` is inside the decal volume if and only if all
    /// three components of `inv * p` are in the range [-1, 1].
    pub fn inverse_world_matrix(&self) -> Mat4 {
        // Build world matrix without scale, then apply inverse scale separately
        // for better numerical stability.
        let unscaled = Mat4::from_rotation_translation(self.rotation, self.position);
        let inv_unscaled = unscaled.inverse();

        // Inverse scale maps half-extents to [-1, 1].
        let inv_scale = Vec3::new(
            1.0 / self.half_extents.x.max(EPSILON),
            1.0 / self.half_extents.y.max(EPSILON),
            1.0 / self.half_extents.z.max(EPSILON),
        );
        let scale_mat = Mat4::from_scale(inv_scale);

        scale_mat * inv_unscaled
    }

    /// Returns the decal's forward direction (projection direction = local -Z).
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// Returns the decal's right direction (local +X).
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Returns the decal's up direction (local +Y).
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Returns the 8 corners of the OBB in world space.
    pub fn world_corners(&self) -> [Vec3; 8] {
        let he = self.half_extents;
        let local_corners = [
            Vec3::new(-he.x, -he.y, -he.z),
            Vec3::new(he.x, -he.y, -he.z),
            Vec3::new(he.x, he.y, -he.z),
            Vec3::new(-he.x, he.y, -he.z),
            Vec3::new(-he.x, -he.y, he.z),
            Vec3::new(he.x, -he.y, he.z),
            Vec3::new(he.x, he.y, he.z),
            Vec3::new(-he.x, he.y, he.z),
        ];

        let mut result = [Vec3::ZERO; 8];
        for (i, lc) in local_corners.iter().enumerate() {
            result[i] = self.position + self.rotation * *lc;
        }
        result
    }

    /// Returns an axis-aligned bounding box that encloses the OBB (conservative).
    pub fn world_aabb(&self) -> (Vec3, Vec3) {
        let corners = self.world_corners();
        let mut aabb_min = Vec3::splat(f32::INFINITY);
        let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);
        for c in &corners {
            aabb_min = aabb_min.min(*c);
            aabb_max = aabb_max.max(*c);
        }
        (aabb_min, aabb_max)
    }

    // -----------------------------------------------------------------------
    // Projection math
    // -----------------------------------------------------------------------

    /// Computes the effective opacity of the decal at the current time,
    /// accounting for lifetime fade-out.
    pub fn effective_opacity(&self) -> f32 {
        if self.lifetime <= 0.0 {
            return self.opacity;
        }

        let remaining = self.lifetime - self.age;
        if remaining <= 0.0 {
            return 0.0;
        }

        if remaining < self.fade_out_duration && self.fade_out_duration > EPSILON {
            self.opacity * (remaining / self.fade_out_duration)
        } else {
            self.opacity
        }
    }

    /// Returns true if this decal has expired.
    pub fn is_expired(&self) -> bool {
        self.lifetime > 0.0 && self.age >= self.lifetime
    }
}

// ---------------------------------------------------------------------------
// Projection utility functions
// ---------------------------------------------------------------------------

/// Projects a world-space position into decal UV space.
///
/// Returns `Some((u, v))` if the point is inside the decal volume (all three
/// local-space coordinates are within [-1, 1]). The UV coordinates are mapped
/// from [-1, 1] to [0, 1].
///
/// Returns `None` if the point is outside the volume.
pub fn project_point_to_decal_uv(
    world_pos: Vec3,
    decal_inverse_matrix: &Mat4,
) -> Option<Vec2> {
    let local = *decal_inverse_matrix * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);

    // Check bounds: all components must be in [-1, 1].
    if local.x.abs() > 1.0 || local.y.abs() > 1.0 || local.z.abs() > 1.0 {
        return None;
    }

    // Map from [-1, 1] to [0, 1].
    let u = local.x * 0.5 + 0.5;
    let v = local.y * 0.5 + 0.5;

    Some(Vec2::new(u, v))
}

/// Projects a world-space position and returns the full local-space coordinates.
///
/// Returns `Some((u, v, depth))` where depth is the normalized projection depth
/// in [0, 1] (0 = front face, 1 = back face of the decal volume).
pub fn project_point_to_decal_uvw(
    world_pos: Vec3,
    decal_inverse_matrix: &Mat4,
) -> Option<Vec3> {
    let local = *decal_inverse_matrix * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);

    if local.x.abs() > 1.0 || local.y.abs() > 1.0 || local.z.abs() > 1.0 {
        return None;
    }

    let u = local.x * 0.5 + 0.5;
    let v = local.y * 0.5 + 0.5;
    let w = local.z * 0.5 + 0.5;

    Some(Vec3::new(u, v, w))
}

/// Computes the angle-based fade factor for decal projection.
///
/// The decal fades as the angle between the surface normal and the decal's
/// projection direction increases.
///
/// # Arguments
/// * `surface_normal` - Unit normal of the surface at the projection point.
/// * `decal_forward` - The decal's forward (projection) direction (unit).
/// * `fade_start` - Angle in radians where fading begins.
/// * `fade_end` - Angle in radians where the decal is fully transparent.
///
/// # Returns
/// Opacity multiplier in [0, 1].
pub fn compute_angle_fade(
    surface_normal: Vec3,
    decal_forward: Vec3,
    fade_start: f32,
    fade_end: f32,
) -> f32 {
    // The decal projects along -Z, so the "ideal" surface faces opposite.
    let cos_angle = (-decal_forward).dot(surface_normal).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();

    if angle <= fade_start {
        1.0
    } else if angle >= fade_end {
        0.0
    } else {
        let range = fade_end - fade_start;
        if range < EPSILON {
            return 0.0;
        }
        1.0 - (angle - fade_start) / range
    }
}

/// Reconstructs a world-space position from a depth buffer value and
/// inverse view-projection matrix.
///
/// This is the CPU-side equivalent of the fragment shader operation:
/// ```text
///   vec4 ndc = vec4(screen_uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
///   vec4 world = inv_view_proj * ndc;
///   world.xyz /= world.w;
/// ```
///
/// # Arguments
/// * `screen_uv` - Screen-space UV coordinates in [0, 1].
/// * `depth` - Depth buffer value in [0, 1].
/// * `inv_view_proj` - Inverse view-projection matrix.
///
/// # Returns
/// The reconstructed world-space position.
pub fn reconstruct_world_position(
    screen_uv: Vec2,
    depth: f32,
    inv_view_proj: &Mat4,
) -> Vec3 {
    let ndc_x = screen_uv.x * 2.0 - 1.0;
    let ndc_y = screen_uv.y * 2.0 - 1.0;
    let ndc_z = depth * 2.0 - 1.0;

    let clip = *inv_view_proj * Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);

    if clip.w.abs() < EPSILON {
        return Vec3::ZERO;
    }

    Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
}

/// Blends a decal normal with a surface normal using the reoriented normal
/// mapping (RNM) technique.
///
/// # Arguments
/// * `surface_normal` - The base surface normal (must be unit length).
/// * `decal_normal` - The decal normal-map sample (in tangent space, Z-up).
/// * `decal_forward` - The decal's forward direction (world space).
/// * `decal_right` - The decal's right direction (world space).
/// * `strength` - Blend strength [0, 1].
///
/// # Returns
/// The blended world-space normal (unit length).
pub fn blend_normals_rnm(
    surface_normal: Vec3,
    decal_normal: Vec3,
    decal_forward: Vec3,
    decal_right: Vec3,
    strength: f32,
) -> Vec3 {
    if strength < EPSILON {
        return surface_normal;
    }

    // Build a tangent frame from the decal orientation.
    let decal_up = decal_forward.cross(decal_right).normalize();

    // Transform decal normal from tangent space to world space.
    let world_decal_normal = (decal_right * decal_normal.x
        + decal_up * decal_normal.y
        + (-decal_forward) * decal_normal.z)
        .normalize();

    // Reoriented normal mapping: blend using quaternion composition.
    // Simplified version using linear interpolation + normalize.
    let t1 = Vec3::new(surface_normal.x, surface_normal.y, surface_normal.z + 1.0);
    let t2 = Vec3::new(
        -world_decal_normal.x * strength,
        -world_decal_normal.y * strength,
        world_decal_normal.z,
    );

    let result = t1 * t2.dot(t1) / t1.dot(t1).max(EPSILON) - t2;
    let blended = result.normalize();

    // Lerp based on strength for a smoother result.
    let final_normal = surface_normal.lerp(blended, strength);
    final_normal.normalize()
}

/// Blends normals using the partial derivative (UDN) method.
pub fn blend_normals_partial_derivative(
    surface_normal: Vec3,
    decal_normal: Vec3,
    decal_forward: Vec3,
    decal_right: Vec3,
    strength: f32,
) -> Vec3 {
    if strength < EPSILON {
        return surface_normal;
    }

    let decal_up = decal_forward.cross(decal_right).normalize();

    // Transform decal normal from tangent space to world space.
    let dn = decal_right * decal_normal.x + decal_up * decal_normal.y;

    // UDN: add the XY of the decal normal to the surface normal.
    let result = surface_normal + dn * strength;
    result.normalize()
}

// ---------------------------------------------------------------------------
// DecalVolumeGpuData
// ---------------------------------------------------------------------------

/// Per-decal-volume data packed for GPU upload.
///
/// 128 bytes per decal (two cache lines).
///
/// ```text
/// Offset   Size   Field
///   0      64     inverse_world (4x4 float)
///  64      16     base_color (RGBA float)
///  80      16     params0: [opacity, normal_blend, fade_cos_start, fade_cos_end]
///  96      16     params1: [uv_scale.x, uv_scale.y, uv_offset.x, uv_offset.y]
/// 112      16     params2: [roughness, metalness, channel_mask_bits, sort_priority_f]
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DecalVolumeGpuData {
    /// Inverse world matrix (world -> decal local space [-1,1]^3).
    pub inverse_world: [[f32; 4]; 4],
    /// Base color tint (RGBA).
    pub base_color: [f32; 4],
    /// [opacity, normal_blend_strength, cos(angle_fade_start), cos(angle_fade_end)]
    pub params0: [f32; 4],
    /// [uv_scale.x, uv_scale.y, uv_offset.x, uv_offset.y]
    pub params1: [f32; 4],
    /// [roughness, metalness, channel_mask_bits, sort_priority as f32]
    pub params2: [f32; 4],
}

impl DecalVolumeGpuData {
    /// Returns the byte size of one decal GPU data entry.
    pub const fn byte_size() -> usize {
        std::mem::size_of::<Self>()
    }

    /// Returns the data as a byte slice (for GPU upload).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    /// Converts a slice of GPU data to bytes.
    pub fn slice_as_bytes(data: &[Self]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<Self>(),
            )
        }
    }

    /// Builds GPU data from a `DecalVolume`.
    pub fn from_decal(decal: &DecalVolume) -> Self {
        let inv = decal.inverse_world_matrix();
        let cols = inv.to_cols_array_2d();

        let channel_mask: u32 = (decal.affects_albedo as u32)
            | ((decal.affects_normals as u32) << 1)
            | ((decal.affects_orm as u32) << 2);

        Self {
            inverse_world: [
                [cols[0][0], cols[0][1], cols[0][2], cols[0][3]],
                [cols[1][0], cols[1][1], cols[1][2], cols[1][3]],
                [cols[2][0], cols[2][1], cols[2][2], cols[2][3]],
                [cols[3][0], cols[3][1], cols[3][2], cols[3][3]],
            ],
            base_color: [
                decal.material.base_color.x,
                decal.material.base_color.y,
                decal.material.base_color.z,
                decal.material.base_color.w,
            ],
            params0: [
                decal.effective_opacity(),
                decal.normal_blend_strength,
                decal.angle_fade_start.cos(),
                decal.angle_fade_end.cos(),
            ],
            params1: [
                decal.material.uv_scale.x,
                decal.material.uv_scale.y,
                decal.material.uv_offset.x,
                decal.material.uv_offset.y,
            ],
            params2: [
                decal.material.roughness,
                decal.material.metalness,
                f32::from_bits(channel_mask),
                decal.sort_priority as f32,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// DecalVolumeRenderer
// ---------------------------------------------------------------------------

/// Renders decal volumes in the deferred pass.
///
/// Each decal is rendered as box geometry. The fragment shader:
/// 1. Samples the depth buffer and reconstructs world position.
/// 2. Transforms world position to decal local space.
/// 3. If inside [-1,1]^3, computes UVs and samples decal textures.
/// 4. Applies angle fade based on surface normal vs decal forward.
/// 5. Blends result into the G-buffer.
#[derive(Debug)]
pub struct DecalVolumeRenderer {
    /// Prepared GPU data for this frame.
    gpu_data: Vec<DecalVolumeGpuData>,
    /// World matrices for box geometry rendering.
    world_matrices: Vec<Mat4>,
    /// Material handles for texture binding.
    material_bindings: Vec<DecalVolumeMaterialBinding>,
    /// Statistics from the last frame.
    last_stats: DecalVolumeRenderStats,
}

/// Material binding info for a single decal draw call.
#[derive(Debug, Clone)]
pub struct DecalVolumeMaterialBinding {
    /// Albedo texture handle.
    pub albedo_texture: u64,
    /// Normal texture handle.
    pub normal_texture: u64,
    /// ORM texture handle.
    pub orm_texture: u64,
    /// Emissive texture handle.
    pub emissive_texture: u64,
    /// Normal blend mode (encoded as integer for shader).
    pub normal_blend_mode: u32,
}

/// Rendering statistics for one frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecalVolumeRenderStats {
    /// Number of decals submitted for rendering.
    pub decals_rendered: u32,
    /// Number of decals culled by frustum.
    pub decals_frustum_culled: u32,
    /// Number of decals skipped due to zero opacity.
    pub decals_faded_out: u32,
    /// Total bytes of GPU data uploaded.
    pub gpu_bytes: u64,
}

/// Frustum planes for view frustum culling.
#[derive(Debug, Clone, Copy)]
struct FrustumPlane {
    normal: Vec3,
    distance: f32,
}

impl DecalVolumeRenderer {
    /// Creates a new decal volume renderer.
    pub fn new() -> Self {
        Self {
            gpu_data: Vec::with_capacity(MAX_RENDERED_PER_FRAME),
            world_matrices: Vec::with_capacity(MAX_RENDERED_PER_FRAME),
            material_bindings: Vec::with_capacity(MAX_RENDERED_PER_FRAME),
            last_stats: DecalVolumeRenderStats::default(),
        }
    }

    /// Returns the prepared GPU data for the current frame.
    pub fn gpu_data(&self) -> &[DecalVolumeGpuData] {
        &self.gpu_data
    }

    /// Returns the world matrices for box geometry instancing.
    pub fn world_matrices(&self) -> &[Mat4] {
        &self.world_matrices
    }

    /// Returns the material bindings for each decal draw call.
    pub fn material_bindings(&self) -> &[DecalVolumeMaterialBinding] {
        &self.material_bindings
    }

    /// Returns statistics from the last frame.
    pub fn last_stats(&self) -> &DecalVolumeRenderStats {
        &self.last_stats
    }

    /// Returns the GPU data buffer as bytes (for upload).
    pub fn gpu_data_bytes(&self) -> &[u8] {
        DecalVolumeGpuData::slice_as_bytes(&self.gpu_data)
    }

    /// Prepares render data from a list of alive decal volumes.
    ///
    /// This performs frustum culling, opacity filtering, sorting by priority,
    /// and packing GPU data.
    pub fn prepare(
        &mut self,
        decals: &[DecalVolume],
        view_projection: &Mat4,
    ) {
        self.gpu_data.clear();
        self.world_matrices.clear();
        self.material_bindings.clear();

        let mut stats = DecalVolumeRenderStats::default();

        // Extract frustum planes for culling.
        let frustum = Self::extract_frustum_planes(view_projection);

        // Collect visible decals with their sort keys.
        let mut visible: Vec<(usize, i32)> = Vec::new();

        for (i, decal) in decals.iter().enumerate() {
            if !decal.alive {
                continue;
            }

            // Skip fully faded decals.
            let opacity = decal.effective_opacity();
            if opacity < 0.001 {
                stats.decals_faded_out += 1;
                continue;
            }

            // Frustum cull using the OBB's AABB.
            let (aabb_min, aabb_max) = decal.world_aabb();
            if !Self::test_aabb_frustum(&aabb_min, &aabb_max, &frustum) {
                stats.decals_frustum_culled += 1;
                continue;
            }

            visible.push((i, decal.sort_priority));
        }

        // Sort by priority (lower first).
        visible.sort_by_key(|&(_, priority)| priority);

        // Limit to GPU budget.
        let count = visible.len().min(MAX_RENDERED_PER_FRAME);

        for &(idx, _) in visible.iter().take(count) {
            let decal = &decals[idx];

            self.gpu_data.push(DecalVolumeGpuData::from_decal(decal));
            self.world_matrices.push(decal.world_matrix());
            self.material_bindings.push(DecalVolumeMaterialBinding {
                albedo_texture: decal.material.albedo_texture,
                normal_texture: decal.material.normal_texture,
                orm_texture: decal.material.orm_texture,
                emissive_texture: decal.material.emissive_texture,
                normal_blend_mode: decal.normal_blend_mode as u32,
            });

            stats.decals_rendered += 1;
        }

        stats.gpu_bytes = (self.gpu_data.len() * DecalVolumeGpuData::byte_size()) as u64;
        self.last_stats = stats;
    }

    /// Extracts six frustum planes from a view-projection matrix.
    fn extract_frustum_planes(vp: &Mat4) -> [FrustumPlane; 6] {
        let cols = vp.to_cols_array_2d();
        let row = |r: usize| -> Vec4 {
            Vec4::new(cols[0][r], cols[1][r], cols[2][r], cols[3][r])
        };

        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let normalize_plane = |v: Vec4| -> FrustumPlane {
            let len = Vec3::new(v.x, v.y, v.z).length();
            if len < EPSILON {
                return FrustumPlane {
                    normal: Vec3::Y,
                    distance: 0.0,
                };
            }
            FrustumPlane {
                normal: Vec3::new(v.x, v.y, v.z) / len,
                distance: -v.w / len,
            }
        };

        [
            normalize_plane(r3 + r0), // left
            normalize_plane(r3 - r0), // right
            normalize_plane(r3 + r1), // bottom
            normalize_plane(r3 - r1), // top
            normalize_plane(r3 + r2), // near
            normalize_plane(r3 - r2), // far
        ]
    }

    /// Tests an AABB against frustum planes.
    fn test_aabb_frustum(
        aabb_min: &Vec3,
        aabb_max: &Vec3,
        planes: &[FrustumPlane; 6],
    ) -> bool {
        for plane in planes {
            let p = Vec3::new(
                if plane.normal.x >= 0.0 { aabb_max.x } else { aabb_min.x },
                if plane.normal.y >= 0.0 { aabb_max.y } else { aabb_min.y },
                if plane.normal.z >= 0.0 { aabb_max.z } else { aabb_min.z },
            );
            if plane.normal.dot(p) - plane.distance < 0.0 {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// DecalVolumeManager
// ---------------------------------------------------------------------------

/// High-level decal volume manager with pooled allocation and lifecycle
/// management.
///
/// Decals are stored in a flat pool with generational IDs. When the pool is
/// full, the oldest decal is recycled. Each frame, the manager ages decals,
/// removes expired ones, and prepares render data.
#[derive(Debug)]
pub struct DecalVolumeManager {
    /// Pool of decal volumes.
    decals: Vec<DecalVolume>,
    /// Generation counters for each pool slot (for ID validation).
    generations: Vec<u32>,
    /// Maximum pool capacity.
    max_capacity: usize,
    /// Number of alive decals.
    alive_count: usize,
    /// The renderer.
    renderer: DecalVolumeRenderer,
    /// Stats from the last update.
    last_stats: DecalVolumeManagerStats,
}

/// Statistics from the decal volume manager.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecalVolumeManagerStats {
    /// Total pool capacity.
    pub capacity: u32,
    /// Number of alive decals.
    pub alive: u32,
    /// Number of decals spawned this frame.
    pub spawned_this_frame: u32,
    /// Number of decals expired this frame.
    pub expired_this_frame: u32,
    /// Number of decals recycled this frame (oldest replaced).
    pub recycled_this_frame: u32,
    /// Render stats.
    pub render_stats: DecalVolumeRenderStats,
}

impl DecalVolumeManager {
    /// Creates a new decal volume manager with the given pool capacity.
    pub fn new(max_capacity: usize) -> Self {
        Self {
            decals: Vec::with_capacity(max_capacity),
            generations: Vec::with_capacity(max_capacity),
            max_capacity,
            alive_count: 0,
            renderer: DecalVolumeRenderer::new(),
            last_stats: DecalVolumeManagerStats::default(),
        }
    }

    /// Creates a manager with the default capacity.
    pub fn with_defaults() -> Self {
        Self::new(MAX_DECAL_VOLUMES)
    }

    /// Returns a reference to the renderer.
    pub fn renderer(&self) -> &DecalVolumeRenderer {
        &self.renderer
    }

    /// Returns the last frame's statistics.
    pub fn last_stats(&self) -> &DecalVolumeManagerStats {
        &self.last_stats
    }

    /// Returns the number of alive decals.
    pub fn alive_count(&self) -> usize {
        self.alive_count
    }

    /// Returns the pool capacity.
    pub fn capacity(&self) -> usize {
        self.max_capacity
    }

    /// Returns a reference to a decal by ID, if it is still valid.
    pub fn get(&self, id: DecalVolumeId) -> Option<&DecalVolume> {
        let idx = id.index as usize;
        if idx < self.decals.len()
            && self.generations[idx] == id.generation
            && self.decals[idx].alive
        {
            Some(&self.decals[idx])
        } else {
            None
        }
    }

    /// Returns a mutable reference to a decal by ID.
    pub fn get_mut(&mut self, id: DecalVolumeId) -> Option<&mut DecalVolume> {
        let idx = id.index as usize;
        if idx < self.decals.len()
            && self.generations[idx] == id.generation
            && self.decals[idx].alive
        {
            Some(&mut self.decals[idx])
        } else {
            None
        }
    }

    /// Spawns a new decal volume.
    ///
    /// Convenience method that builds a `DecalVolume` from basic parameters.
    ///
    /// # Returns
    /// The ID of the spawned decal, or `None` if the pool is full and no
    /// slot could be recycled (should not happen in practice since oldest
    /// is always recycled).
    pub fn spawn(
        &mut self,
        position: Vec3,
        rotation: Quat,
        size: Vec3,
        material: DecalVolumeMaterial,
        lifetime: f32,
    ) -> DecalVolumeId {
        let half_extents = size * 0.5;
        let decal = DecalVolume::new(position, rotation, half_extents)
            .with_material(material)
            .with_lifetime(lifetime);

        self.spawn_decal(decal)
    }

    /// Spawns a pre-built decal volume into the pool.
    pub fn spawn_decal(&mut self, mut decal: DecalVolume) -> DecalVolumeId {
        // Try to find a dead slot.
        if let Some(idx) = self.decals.iter().position(|d| !d.alive) {
            self.generations[idx] = self.generations[idx].wrapping_add(1);
            decal.generation = self.generations[idx];
            decal.alive = true;
            self.decals[idx] = decal;
            self.alive_count += 1;
            return DecalVolumeId::new(idx as u32, self.generations[idx]);
        }

        // Pool not full yet -- append.
        if self.decals.len() < self.max_capacity {
            let idx = self.decals.len();
            let generation = 1u32;
            decal.generation = generation;
            decal.alive = true;
            self.decals.push(decal);
            self.generations.push(generation);
            self.alive_count += 1;
            return DecalVolumeId::new(idx as u32, generation);
        }

        // Pool is full -- recycle the oldest alive decal.
        let oldest_idx = self.find_oldest_alive();
        self.generations[oldest_idx] = self.generations[oldest_idx].wrapping_add(1);
        decal.generation = self.generations[oldest_idx];
        decal.alive = true;
        self.decals[oldest_idx] = decal;
        // alive_count stays the same (replaced one alive with another).
        DecalVolumeId::new(oldest_idx as u32, self.generations[oldest_idx])
    }

    /// Kills a decal by ID.
    pub fn despawn(&mut self, id: DecalVolumeId) {
        if let Some(decal) = self.get_mut(id) {
            decal.alive = false;
            self.alive_count = self.alive_count.saturating_sub(1);
        }
    }

    /// Kills all decals with the given tag.
    pub fn despawn_by_tag(&mut self, tag: u32) {
        for decal in &mut self.decals {
            if decal.alive && decal.tag == tag {
                decal.alive = false;
                self.alive_count = self.alive_count.saturating_sub(1);
            }
        }
    }

    /// Kills all decals.
    pub fn clear(&mut self) {
        for decal in &mut self.decals {
            decal.alive = false;
        }
        self.alive_count = 0;
    }

    /// Per-frame update: age decals, remove expired, and prepare render data.
    pub fn update(&mut self, dt: f32, view_projection: &Mat4) {
        let mut stats = DecalVolumeManagerStats {
            capacity: self.max_capacity as u32,
            ..Default::default()
        };

        // Age and expire.
        for decal in &mut self.decals {
            if !decal.alive {
                continue;
            }

            if decal.lifetime > 0.0 {
                decal.age += dt;
                if decal.age >= decal.lifetime {
                    decal.alive = false;
                    self.alive_count = self.alive_count.saturating_sub(1);
                    stats.expired_this_frame += 1;
                }
            }
        }

        stats.alive = self.alive_count as u32;

        // Collect alive decals for rendering.
        let alive_decals: Vec<DecalVolume> = self
            .decals
            .iter()
            .filter(|d| d.alive)
            .cloned()
            .collect();

        self.renderer.prepare(&alive_decals, view_projection);

        stats.render_stats = *self.renderer.last_stats();
        self.last_stats = stats;
    }

    /// Returns an iterator over alive decals.
    pub fn alive_decals(&self) -> impl Iterator<Item = &DecalVolume> {
        self.decals.iter().filter(|d| d.alive)
    }

    /// Finds the index of the oldest alive decal (by age).
    fn find_oldest_alive(&self) -> usize {
        self.decals
            .iter()
            .enumerate()
            .filter(|(_, d)| d.alive)
            .max_by(|(_, a), (_, b)| {
                a.age
                    .partial_cmp(&b.age)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // UV projection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_project_point_center() {
        let decal = DecalVolume::new(
            Vec3::new(5.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::new(1.0, 1.0, 0.5),
        );
        let inv = decal.inverse_world_matrix();

        // Point at the decal center should project to (0.5, 0.5).
        let uv = project_point_to_decal_uv(Vec3::new(5.0, 0.0, 0.0), &inv);
        assert!(uv.is_some());
        let uv = uv.unwrap();
        assert!((uv.x - 0.5).abs() < 0.01, "u = {}, expected 0.5", uv.x);
        assert!((uv.y - 0.5).abs() < 0.01, "v = {}, expected 0.5", uv.y);
    }

    #[test]
    fn test_project_point_corner() {
        let decal = DecalVolume::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(1.0, 1.0, 1.0),
        );
        let inv = decal.inverse_world_matrix();

        // Point at corner (-1, -1, 0) in decal space = world (-1, -1, 0).
        let uv = project_point_to_decal_uv(Vec3::new(-1.0, -1.0, 0.0), &inv);
        assert!(uv.is_some());
        let uv = uv.unwrap();
        assert!((uv.x - 0.0).abs() < 0.01, "u = {}, expected 0.0", uv.x);
        assert!((uv.y - 0.0).abs() < 0.01, "v = {}, expected 0.0", uv.y);
    }

    #[test]
    fn test_project_point_outside() {
        let decal = DecalVolume::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(1.0, 1.0, 1.0),
        );
        let inv = decal.inverse_world_matrix();

        // Point far outside the decal volume.
        let uv = project_point_to_decal_uv(Vec3::new(10.0, 0.0, 0.0), &inv);
        assert!(uv.is_none());
    }

    #[test]
    fn test_project_point_rotated_decal() {
        // Decal rotated 90 degrees around Y.
        let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let decal = DecalVolume::new(Vec3::ZERO, rotation, Vec3::new(2.0, 2.0, 1.0));
        let inv = decal.inverse_world_matrix();

        // Center should still project to (0.5, 0.5).
        let uv = project_point_to_decal_uv(Vec3::ZERO, &inv);
        assert!(uv.is_some());
        let uv = uv.unwrap();
        assert!((uv.x - 0.5).abs() < 0.01);
        assert!((uv.y - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_project_uvw() {
        let decal = DecalVolume::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(1.0, 1.0, 1.0),
        );
        let inv = decal.inverse_world_matrix();

        let uvw = project_point_to_decal_uvw(Vec3::ZERO, &inv);
        assert!(uvw.is_some());
        let uvw = uvw.unwrap();
        assert!((uvw.x - 0.5).abs() < 0.01);
        assert!((uvw.y - 0.5).abs() < 0.01);
        assert!((uvw.z - 0.5).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Angle fade tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_angle_fade_direct() {
        // Surface facing directly opposite to the decal forward: full opacity.
        let fade = compute_angle_fade(
            Vec3::Z,
            Vec3::NEG_Z,
            std::f32::consts::FRAC_PI_3,
            std::f32::consts::FRAC_PI_2,
        );
        assert!(
            (fade - 1.0).abs() < 0.01,
            "Direct facing should be 1.0, got {}",
            fade
        );
    }

    #[test]
    fn test_angle_fade_perpendicular() {
        // Surface perpendicular to decal forward (90 degrees).
        let fade = compute_angle_fade(
            Vec3::X,
            Vec3::NEG_Z,
            std::f32::consts::FRAC_PI_3,
            std::f32::consts::FRAC_PI_2,
        );
        assert!(
            fade < 0.01,
            "Perpendicular should be ~0.0, got {}",
            fade
        );
    }

    #[test]
    fn test_angle_fade_midpoint() {
        // Surface at 75 degrees (halfway between 60 and 90).
        let angle = 75.0_f32.to_radians();
        let surface_normal = Vec3::new(angle.sin(), 0.0, angle.cos());
        let fade = compute_angle_fade(
            surface_normal,
            Vec3::NEG_Z,
            60.0_f32.to_radians(),
            90.0_f32.to_radians(),
        );
        assert!(
            fade > 0.1 && fade < 0.9,
            "Midpoint should be ~0.5, got {}",
            fade
        );
    }

    // -----------------------------------------------------------------------
    // Lifetime tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lifetime_immortal() {
        let decal = DecalVolume::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        // Negative lifetime = immortal.
        assert!((decal.effective_opacity() - 1.0).abs() < EPSILON);
        assert!(!decal.is_expired());
    }

    #[test]
    fn test_lifetime_expired() {
        let mut decal = DecalVolume::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
            .with_lifetime(5.0);
        decal.age = 6.0;
        assert!(decal.is_expired());
        assert!((decal.effective_opacity() - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_lifetime_fade_out() {
        let mut decal = DecalVolume::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)
            .with_lifetime(5.0)
            .with_fade_out(1.0)
            .with_opacity(1.0);

        // At age 4.0, remaining = 1.0 = fade_out_duration -> start of fade.
        decal.age = 4.0;
        assert!((decal.effective_opacity() - 1.0).abs() < EPSILON);

        // At age 4.5, remaining = 0.5 -> 50% fade.
        decal.age = 4.5;
        assert!((decal.effective_opacity() - 0.5).abs() < 0.01);

        // At age 5.0, remaining = 0 -> fully faded.
        decal.age = 5.0;
        assert!((decal.effective_opacity() - 0.0).abs() < EPSILON);
    }

    // -----------------------------------------------------------------------
    // World position reconstruction test
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconstruct_world_position() {
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let proj = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let vp = proj * view;
        let inv_vp = vp.inverse();

        // Project a known world point to screen, then reconstruct.
        let world_point = Vec3::new(1.0, 2.0, -10.0);
        let clip = vp * Vec4::new(world_point.x, world_point.y, world_point.z, 1.0);
        let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        let screen_uv = Vec2::new(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5);
        let depth = ndc.z * 0.5 + 0.5;

        let reconstructed = reconstruct_world_position(screen_uv, depth, &inv_vp);
        assert!(
            (reconstructed - world_point).length() < 0.01,
            "Reconstructed {:?}, expected {:?}",
            reconstructed,
            world_point
        );
    }

    // -----------------------------------------------------------------------
    // Normal blending test
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_normals_rnm_zero_strength() {
        let surface = Vec3::Y;
        let decal_n = Vec3::new(0.3, 0.3, 0.9).normalize();
        let result = blend_normals_rnm(surface, decal_n, Vec3::NEG_Z, Vec3::X, 0.0);
        // Zero strength should return the original surface normal.
        assert!((result - surface).length() < 0.01);
    }

    #[test]
    fn test_blend_normals_partial_derivative() {
        let surface = Vec3::Y;
        let decal_n = Vec3::new(0.0, 0.0, 1.0); // flat normal map
        let result =
            blend_normals_partial_derivative(surface, decal_n, Vec3::NEG_Z, Vec3::X, 1.0);
        // Flat decal normal should not significantly perturb the surface.
        assert!(
            result.dot(surface) > 0.9,
            "Flat decal normal should not perturb much, got {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // Manager tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_manager_spawn_and_despawn() {
        let mut mgr = DecalVolumeManager::new(10);

        let id = mgr.spawn(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
            DecalVolumeMaterial::default(),
            -1.0,
        );

        assert_eq!(mgr.alive_count(), 1);
        assert!(mgr.get(id).is_some());

        mgr.despawn(id);
        assert_eq!(mgr.alive_count(), 0);
        assert!(mgr.get(id).is_none());
    }

    #[test]
    fn test_manager_recycles_oldest() {
        let mut mgr = DecalVolumeManager::new(3);

        // Spawn 3 decals.
        let mut ids = Vec::new();
        for i in 0..3 {
            let mut decal = DecalVolume::new(
                Vec3::new(i as f32, 0.0, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            )
            .with_lifetime(-1.0);
            decal.age = i as f32; // older age for earlier spawned
            ids.push(mgr.spawn_decal(decal));
        }
        assert_eq!(mgr.alive_count(), 3);

        // Spawn a 4th -- should recycle the oldest (index 2 has highest age).
        let new_decal = DecalVolume::new(Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE)
            .with_lifetime(-1.0);
        let new_id = mgr.spawn_decal(new_decal);

        // Still 3 alive (one was recycled).
        assert_eq!(mgr.alive_count(), 3);

        // The oldest (id with age=2.0) should be gone.
        assert!(mgr.get(ids[2]).is_none());
        // The new one should be alive.
        assert!(mgr.get(new_id).is_some());
    }

    #[test]
    fn test_manager_expiry() {
        let mut mgr = DecalVolumeManager::new(10);

        let id = mgr.spawn(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            DecalVolumeMaterial::default(),
            1.0, // 1 second lifetime
        );
        assert_eq!(mgr.alive_count(), 1);

        let vp = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);

        // Update past the lifetime.
        mgr.update(2.0, &vp);

        assert_eq!(mgr.alive_count(), 0);
        assert!(mgr.get(id).is_none());
    }

    #[test]
    fn test_manager_despawn_by_tag() {
        let mut mgr = DecalVolumeManager::new(10);

        let d1 = DecalVolume::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE).with_tag(42);
        let d2 = DecalVolume::new(Vec3::X, Quat::IDENTITY, Vec3::ONE).with_tag(99);
        let d3 = DecalVolume::new(Vec3::Y, Quat::IDENTITY, Vec3::ONE).with_tag(42);

        mgr.spawn_decal(d1);
        let id2 = mgr.spawn_decal(d2);
        mgr.spawn_decal(d3);

        assert_eq!(mgr.alive_count(), 3);

        mgr.despawn_by_tag(42);
        assert_eq!(mgr.alive_count(), 1);
        assert!(mgr.get(id2).is_some());
    }

    // -----------------------------------------------------------------------
    // GPU data tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_data_size() {
        assert_eq!(
            DecalVolumeGpuData::byte_size(),
            128,
            "DecalVolumeGpuData should be 128 bytes"
        );
    }

    #[test]
    fn test_gpu_data_from_decal() {
        let decal = DecalVolume::new(
            Vec3::new(5.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::new(2.0, 2.0, 1.0),
        )
        .with_opacity(0.8)
        .with_sort_priority(3);

        let gpu = DecalVolumeGpuData::from_decal(&decal);

        // Check opacity.
        assert!((gpu.params0[0] - 0.8).abs() < 0.01);
        // Check sort priority.
        assert!((gpu.params2[3] - 3.0).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Renderer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_renderer_prepare() {
        let mut renderer = DecalVolumeRenderer::new();

        let decals = vec![
            DecalVolume::new(Vec3::new(0.0, 0.0, -5.0), Quat::IDENTITY, Vec3::ONE)
                .with_opacity(1.0),
            DecalVolume::new(Vec3::new(0.0, 0.0, -10.0), Quat::IDENTITY, Vec3::ONE)
                .with_opacity(1.0),
            DecalVolume::new(Vec3::new(0.0, 0.0, -15.0), Quat::IDENTITY, Vec3::ONE)
                .with_opacity(0.0), // fully faded -- should be skipped
        ];

        let vp = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);

        renderer.prepare(&decals, &vp);

        let stats = renderer.last_stats();
        assert_eq!(stats.decals_rendered, 2);
        assert_eq!(stats.decals_faded_out, 1);
    }

    #[test]
    fn test_renderer_sort_priority() {
        let mut renderer = DecalVolumeRenderer::new();

        let decals = vec![
            DecalVolume::new(Vec3::new(0.0, 0.0, -5.0), Quat::IDENTITY, Vec3::ONE)
                .with_sort_priority(10),
            DecalVolume::new(Vec3::new(0.0, 0.0, -5.0), Quat::IDENTITY, Vec3::ONE)
                .with_sort_priority(1),
            DecalVolume::new(Vec3::new(0.0, 0.0, -5.0), Quat::IDENTITY, Vec3::ONE)
                .with_sort_priority(5),
        ];

        let vp = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 100.0)
            * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);

        renderer.prepare(&decals, &vp);

        // Should be sorted by priority: 1, 5, 10.
        assert_eq!(renderer.gpu_data().len(), 3);
        assert!((renderer.gpu_data()[0].params2[3] - 1.0).abs() < 0.01);
        assert!((renderer.gpu_data()[1].params2[3] - 5.0).abs() < 0.01);
        assert!((renderer.gpu_data()[2].params2[3] - 10.0).abs() < 0.01);
    }
}
