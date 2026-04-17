// engine/render/src/deferred_v2.rs
//
// Enhanced deferred rendering pipeline with thin G-buffer packing, stencil-based
// light volumes, light pre-pass (deferred lighting), tiled deferred shading,
// and cluster debug visualization.
//
// This module extends the basic deferred pipeline with bandwidth-efficient
// G-buffer layouts, advanced light culling strategies, and diagnostic tools
// for inspecting the tile/cluster grid.

use crate::interface::resource::TextureFormat;
use glam::{Mat4, UVec2, UVec3, Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of lights supported by the tiled/clustered deferred path.
pub const MAX_TILED_LIGHTS: usize = 4096;

/// Default tile size in pixels for tiled deferred.
pub const DEFAULT_TILE_SIZE: u32 = 16;

/// Default number of depth slices for clustered shading.
pub const DEFAULT_CLUSTER_DEPTH_SLICES: u32 = 24;

/// Maximum number of lights per tile before overflow handling kicks in.
pub const MAX_LIGHTS_PER_TILE: usize = 256;

/// Maximum number of lights per cluster.
pub const MAX_LIGHTS_PER_CLUSTER: usize = 128;

/// Stencil reference value used for marking light volume pixels.
pub const STENCIL_LIGHT_VOLUME_REF: u8 = 0x01;

/// Small epsilon for numerical stability.
const EPSILON: f32 = 1e-7;

// ---------------------------------------------------------------------------
// Thin G-Buffer layout
// ---------------------------------------------------------------------------

/// Describes a bandwidth-efficient "thin" G-buffer that packs albedo+metallic
/// into a single render target and normal+roughness into another.
///
/// Layout:
///   RT0: albedo.rgb (R8G8B8) + metallic (A8)     -> RGBA8Unorm
///   RT1: normal.xy  (R16G16) + roughness (B8) + flags (A8)  -> RGBA16Float
///   Depth: 32-bit float depth buffer
#[derive(Debug, Clone)]
pub struct ThinGBufferLayout {
    /// Format for the albedo+metallic render target.
    pub albedo_metallic_format: TextureFormat,
    /// Format for the normal+roughness render target.
    pub normal_roughness_format: TextureFormat,
    /// Depth-stencil format.
    pub depth_stencil_format: TextureFormat,
    /// Optional emissive render target (only when emissive objects are present).
    pub emissive_format: Option<TextureFormat>,
    /// Width of the G-buffer in pixels.
    pub width: u32,
    /// Height of the G-buffer in pixels.
    pub height: u32,
}

impl Default for ThinGBufferLayout {
    fn default() -> Self {
        Self {
            albedo_metallic_format: TextureFormat::Rgba8Unorm,
            normal_roughness_format: TextureFormat::Rgba16Float,
            depth_stencil_format: TextureFormat::Depth32Float,
            emissive_format: None,
            width: 1920,
            height: 1080,
        }
    }
}

impl ThinGBufferLayout {
    /// Create a thin G-buffer layout for a given resolution.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// High-precision variant using 16-bit float for albedo as well.
    pub fn high_precision(width: u32, height: u32) -> Self {
        Self {
            albedo_metallic_format: TextureFormat::Rgba16Float,
            normal_roughness_format: TextureFormat::Rgba16Float,
            depth_stencil_format: TextureFormat::Depth32Float,
            emissive_format: Some(TextureFormat::Rgba16Float),
            width,
            height,
        }
    }

    /// Minimal layout using RGB10A2 for normals (10-bit precision per axis).
    pub fn minimal(width: u32, height: u32) -> Self {
        Self {
            albedo_metallic_format: TextureFormat::Rgba8Unorm,
            normal_roughness_format: TextureFormat::Rgb10A2Unorm,
            depth_stencil_format: TextureFormat::Depth32Float,
            emissive_format: None,
            width,
            height,
        }
    }

    /// Returns the total colour attachment count.
    pub fn color_attachment_count(&self) -> usize {
        if self.emissive_format.is_some() { 3 } else { 2 }
    }

    /// Returns the total estimated memory usage in bytes.
    pub fn estimated_memory_bytes(&self) -> u64 {
        let pixels = self.width as u64 * self.height as u64;
        let albedo_bpp = format_bytes_per_pixel(self.albedo_metallic_format);
        let normal_bpp = format_bytes_per_pixel(self.normal_roughness_format);
        let depth_bpp = 4u64; // 32-bit depth
        let emissive_bpp = self
            .emissive_format
            .map(format_bytes_per_pixel)
            .unwrap_or(0);
        pixels * (albedo_bpp + normal_bpp + depth_bpp + emissive_bpp)
    }
}

/// Returns the bytes per pixel for a given texture format (approximate).
fn format_bytes_per_pixel(format: TextureFormat) -> u64 {
    match format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8Snorm => 4,
        TextureFormat::Rgba16Float => 8,
        TextureFormat::Rg16Float => 4,
        TextureFormat::Rgb10A2Unorm => 4,
        TextureFormat::Rg11B10Float => 4,
        TextureFormat::Depth32Float => 4,
        _ => 4,
    }
}

// ---------------------------------------------------------------------------
// Thin G-Buffer runtime data
// ---------------------------------------------------------------------------

/// Holds the actual render target handles for a thin G-buffer.
#[derive(Debug, Clone)]
pub struct ThinGBuffer {
    /// Albedo+metallic packed render target.
    pub albedo_metallic_rt: u64,
    /// Normal+roughness packed render target.
    pub normal_roughness_rt: u64,
    /// Depth-stencil render target.
    pub depth_stencil_rt: u64,
    /// Optional emissive render target.
    pub emissive_rt: Option<u64>,
    /// Layout descriptor.
    pub layout: ThinGBufferLayout,
    /// Whether the G-buffer has been filled this frame.
    pub is_valid: bool,
}

impl ThinGBuffer {
    /// Create a new thin G-buffer from a layout descriptor.
    pub fn new(layout: ThinGBufferLayout) -> Self {
        Self {
            albedo_metallic_rt: 0,
            normal_roughness_rt: 0,
            depth_stencil_rt: 0,
            emissive_rt: if layout.emissive_format.is_some() {
                Some(0)
            } else {
                None
            },
            layout,
            is_valid: false,
        }
    }

    /// Invalidate the G-buffer (e.g. on resize).
    pub fn invalidate(&mut self) {
        self.is_valid = false;
    }

    /// Mark the G-buffer as valid after a geometry pass.
    pub fn mark_valid(&mut self) {
        self.is_valid = true;
    }

    /// Resize the G-buffer.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.layout.width = width;
        self.layout.height = height;
        self.invalidate();
    }

    /// Get resolution as a 2D vector.
    pub fn resolution(&self) -> UVec2 {
        UVec2::new(self.layout.width, self.layout.height)
    }
}

// ---------------------------------------------------------------------------
// G-Buffer packing / unpacking helpers
// ---------------------------------------------------------------------------

/// Encode albedo and metallic into a single RGBA8 value.
#[derive(Debug, Clone, Copy)]
pub struct PackedAlbedoMetallic {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub metallic: u8,
}

impl PackedAlbedoMetallic {
    /// Pack an albedo colour (linear 0-1 range) and metallic value.
    pub fn pack(albedo: Vec3, metallic: f32) -> Self {
        Self {
            r: (albedo.x.clamp(0.0, 1.0) * 255.0) as u8,
            g: (albedo.y.clamp(0.0, 1.0) * 255.0) as u8,
            b: (albedo.z.clamp(0.0, 1.0) * 255.0) as u8,
            metallic: (metallic.clamp(0.0, 1.0) * 255.0) as u8,
        }
    }

    /// Unpack to linear albedo and metallic.
    pub fn unpack(&self) -> (Vec3, f32) {
        let albedo = Vec3::new(
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
        );
        let metallic = self.metallic as f32 / 255.0;
        (albedo, metallic)
    }

    /// Encode to a u32 (RGBA8 packed).
    pub fn to_u32(&self) -> u32 {
        (self.r as u32) | ((self.g as u32) << 8) | ((self.b as u32) << 16) | ((self.metallic as u32) << 24)
    }

    /// Decode from a u32 (RGBA8 packed).
    pub fn from_u32(v: u32) -> Self {
        Self {
            r: (v & 0xFF) as u8,
            g: ((v >> 8) & 0xFF) as u8,
            b: ((v >> 16) & 0xFF) as u8,
            metallic: ((v >> 24) & 0xFF) as u8,
        }
    }
}

/// Encode world-space normal (octahedron mapping) and roughness.
#[derive(Debug, Clone, Copy)]
pub struct PackedNormalRoughness {
    /// Octahedron-encoded normal X component (16-bit float).
    pub oct_x: f32,
    /// Octahedron-encoded normal Y component (16-bit float).
    pub oct_y: f32,
    /// Roughness value (0-1).
    pub roughness: f32,
    /// Material flags (ambient occlusion, subsurface, etc.)
    pub flags: u8,
}

impl PackedNormalRoughness {
    /// Encode a world-space normal and roughness.
    pub fn pack(normal: Vec3, roughness: f32, flags: u8) -> Self {
        let n = normal.normalize_or_zero();
        let inv_l1 = 1.0 / (n.x.abs() + n.y.abs() + n.z.abs() + EPSILON);
        let mut oct_x = n.x * inv_l1;
        let mut oct_y = n.y * inv_l1;
        if n.z < 0.0 {
            let tmp_x = (1.0 - oct_y.abs()) * oct_x.signum();
            let tmp_y = (1.0 - oct_x.abs()) * oct_y.signum();
            oct_x = tmp_x;
            oct_y = tmp_y;
        }
        Self {
            oct_x,
            oct_y,
            roughness: roughness.clamp(0.0, 1.0),
            flags,
        }
    }

    /// Decode octahedron normal.
    pub fn unpack_normal(&self) -> Vec3 {
        let mut n = Vec3::new(self.oct_x, self.oct_y, 1.0 - self.oct_x.abs() - self.oct_y.abs());
        if n.z < 0.0 {
            let tmp_x = (1.0 - n.y.abs()) * n.x.signum();
            let tmp_y = (1.0 - n.x.abs()) * n.y.signum();
            n.x = tmp_x;
            n.y = tmp_y;
        }
        n.normalize_or_zero()
    }

    /// Get roughness value.
    pub fn unpack_roughness(&self) -> f32 {
        self.roughness
    }
}

// ---------------------------------------------------------------------------
// Stencil-based light volumes
// ---------------------------------------------------------------------------

/// A light volume for stencil-based deferred lighting.
///
/// The stencil technique works in two passes:
/// 1. Render the back-faces of the light volume with depth test, incrementing stencil.
/// 2. Render the front-faces with stencil test, applying lighting where stencil != 0.
///
/// This avoids lighting pixels outside the light's influence.
#[derive(Debug, Clone)]
pub struct StencilLightVolume {
    /// Light index into the global light array.
    pub light_index: u32,
    /// Type of volume geometry.
    pub volume_type: LightVolumeType,
    /// World-space centre of the light.
    pub position: Vec3,
    /// Radius of the light (for sphere volumes).
    pub radius: f32,
    /// Direction (for cone/spot volumes).
    pub direction: Vec3,
    /// Outer angle in radians (for spot lights).
    pub outer_angle: f32,
    /// Inner angle in radians (for spot lights).
    pub inner_angle: f32,
    /// World transform of the volume mesh.
    pub world_transform: Mat4,
    /// Whether the camera is inside this volume.
    pub camera_inside: bool,
    /// Light colour (linear HDR).
    pub color: Vec3,
    /// Light intensity.
    pub intensity: f32,
}

/// Types of light volume geometry used for stencil marking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightVolumeType {
    /// Full-screen quad (for directional lights).
    FullScreenQuad,
    /// Sphere (for point lights).
    Sphere,
    /// Cone (for spot lights).
    Cone,
    /// Oriented box (for area/rect lights).
    OrientedBox,
    /// Custom mesh (for unusual light shapes).
    CustomMesh,
}

impl StencilLightVolume {
    /// Create a point light volume.
    pub fn point_light(index: u32, position: Vec3, radius: f32, color: Vec3, intensity: f32) -> Self {
        let scale = Mat4::from_scale(Vec3::splat(radius));
        let translation = Mat4::from_translation(position);
        Self {
            light_index: index,
            volume_type: LightVolumeType::Sphere,
            position,
            radius,
            direction: Vec3::NEG_Z,
            outer_angle: PI,
            inner_angle: PI,
            world_transform: translation * scale,
            camera_inside: false,
            color,
            intensity,
        }
    }

    /// Create a spot light volume.
    pub fn spot_light(
        index: u32,
        position: Vec3,
        direction: Vec3,
        range: f32,
        outer_angle: f32,
        inner_angle: f32,
        color: Vec3,
        intensity: f32,
    ) -> Self {
        let cone_radius = range * outer_angle.tan();
        let scale = Mat4::from_scale(Vec3::new(cone_radius, cone_radius, range));
        let translation = Mat4::from_translation(position);
        Self {
            light_index: index,
            volume_type: LightVolumeType::Cone,
            position,
            radius: range,
            direction: direction.normalize_or_zero(),
            outer_angle,
            inner_angle,
            world_transform: translation * scale,
            camera_inside: false,
            color,
            intensity,
        }
    }

    /// Create a directional light (full-screen quad, no volume).
    pub fn directional_light(index: u32, direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            light_index: index,
            volume_type: LightVolumeType::FullScreenQuad,
            position: Vec3::ZERO,
            radius: f32::MAX,
            direction: direction.normalize_or_zero(),
            outer_angle: PI,
            inner_angle: PI,
            world_transform: Mat4::IDENTITY,
            camera_inside: true,
            color,
            intensity,
        }
    }

    /// Create an area light volume (oriented box).
    pub fn area_light(
        index: u32,
        position: Vec3,
        half_extents: Vec3,
        orientation: Mat4,
        color: Vec3,
        intensity: f32,
    ) -> Self {
        let scale = Mat4::from_scale(half_extents);
        let translation = Mat4::from_translation(position);
        Self {
            light_index: index,
            volume_type: LightVolumeType::OrientedBox,
            position,
            radius: half_extents.length(),
            direction: Vec3::NEG_Z,
            outer_angle: PI,
            inner_angle: PI,
            world_transform: translation * orientation * scale,
            camera_inside: false,
            color,
            intensity,
        }
    }

    /// Test whether the camera is inside this light volume.
    pub fn test_camera_inside(&mut self, camera_pos: Vec3) {
        self.camera_inside = match self.volume_type {
            LightVolumeType::FullScreenQuad => true,
            LightVolumeType::Sphere => {
                (camera_pos - self.position).length_squared() < self.radius * self.radius
            }
            LightVolumeType::Cone => {
                let to_cam = camera_pos - self.position;
                let dist_along = to_cam.dot(self.direction);
                if dist_along < 0.0 || dist_along > self.radius {
                    false
                } else {
                    let perp_dist = (to_cam - self.direction * dist_along).length();
                    let cone_radius_at_dist = dist_along * self.outer_angle.tan();
                    perp_dist < cone_radius_at_dist
                }
            }
            LightVolumeType::OrientedBox | LightVolumeType::CustomMesh => {
                // Conservative: assume inside if close enough
                (camera_pos - self.position).length_squared() < self.radius * self.radius
            }
        };
    }

    /// Get the stencil render state for the back-face pass.
    pub fn back_face_stencil_state(&self) -> StencilState {
        if self.camera_inside {
            StencilState {
                stencil_test: false,
                stencil_op_fail: StencilOp::Keep,
                stencil_op_depth_fail: StencilOp::Keep,
                stencil_op_pass: StencilOp::Keep,
                reference: 0,
                read_mask: 0xFF,
                write_mask: 0xFF,
                depth_test: false,
                cull_mode: CullFace::None,
            }
        } else {
            StencilState {
                stencil_test: false,
                stencil_op_fail: StencilOp::Keep,
                stencil_op_depth_fail: StencilOp::IncrementWrap,
                stencil_op_pass: StencilOp::Keep,
                reference: 0,
                read_mask: 0xFF,
                write_mask: 0xFF,
                depth_test: true,
                cull_mode: CullFace::Front,
            }
        }
    }

    /// Get the stencil render state for the front-face lighting pass.
    pub fn front_face_stencil_state(&self) -> StencilState {
        if self.camera_inside {
            StencilState {
                stencil_test: false,
                stencil_op_fail: StencilOp::Keep,
                stencil_op_depth_fail: StencilOp::Keep,
                stencil_op_pass: StencilOp::Keep,
                reference: 0,
                read_mask: 0xFF,
                write_mask: 0x00,
                depth_test: false,
                cull_mode: CullFace::Front,
            }
        } else {
            StencilState {
                stencil_test: true,
                stencil_op_fail: StencilOp::Keep,
                stencil_op_depth_fail: StencilOp::Keep,
                stencil_op_pass: StencilOp::DecrementWrap,
                reference: STENCIL_LIGHT_VOLUME_REF,
                read_mask: 0xFF,
                write_mask: 0xFF,
                depth_test: true,
                cull_mode: CullFace::Back,
            }
        }
    }
}

/// Stencil operations for GPU state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
    Invert,
}

/// Cull face mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullFace {
    None,
    Front,
    Back,
}

/// Combined stencil/depth state for rendering light volumes.
#[derive(Debug, Clone, Copy)]
pub struct StencilState {
    pub stencil_test: bool,
    pub stencil_op_fail: StencilOp,
    pub stencil_op_depth_fail: StencilOp,
    pub stencil_op_pass: StencilOp,
    pub reference: u8,
    pub read_mask: u8,
    pub write_mask: u8,
    pub depth_test: bool,
    pub cull_mode: CullFace,
}

/// Manages a list of stencil light volumes for rendering.
#[derive(Debug)]
pub struct StencilLightVolumeManager {
    /// All light volumes for the current frame.
    pub volumes: Vec<StencilLightVolume>,
    /// Number of directional lights (rendered as full-screen quads).
    pub directional_count: u32,
    /// Number of point lights.
    pub point_count: u32,
    /// Number of spot lights.
    pub spot_count: u32,
    /// Number of area lights.
    pub area_count: u32,
    /// Sphere mesh vertex count for point light volumes.
    pub sphere_vertex_count: u32,
    /// Cone mesh vertex count for spot light volumes.
    pub cone_vertex_count: u32,
}

impl StencilLightVolumeManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self {
            volumes: Vec::new(),
            directional_count: 0,
            point_count: 0,
            spot_count: 0,
            area_count: 0,
            sphere_vertex_count: 0,
            cone_vertex_count: 0,
        }
    }

    /// Clear all volumes for a new frame.
    pub fn clear(&mut self) {
        self.volumes.clear();
        self.directional_count = 0;
        self.point_count = 0;
        self.spot_count = 0;
        self.area_count = 0;
    }

    /// Add a light volume.
    pub fn add_volume(&mut self, volume: StencilLightVolume) {
        match volume.volume_type {
            LightVolumeType::FullScreenQuad => self.directional_count += 1,
            LightVolumeType::Sphere => self.point_count += 1,
            LightVolumeType::Cone => self.spot_count += 1,
            LightVolumeType::OrientedBox => self.area_count += 1,
            LightVolumeType::CustomMesh => {}
        }
        self.volumes.push(volume);
    }

    /// Update camera-inside tests for all volumes.
    pub fn update_camera_tests(&mut self, camera_pos: Vec3) {
        for volume in &mut self.volumes {
            volume.test_camera_inside(camera_pos);
        }
    }

    /// Sort volumes: directionals first, then by distance to camera.
    pub fn sort_for_rendering(&mut self, camera_pos: Vec3) {
        self.volumes.sort_by(|a, b| {
            let a_order = match a.volume_type {
                LightVolumeType::FullScreenQuad => 0u32,
                _ => 1,
            };
            let b_order = match b.volume_type {
                LightVolumeType::FullScreenQuad => 0u32,
                _ => 1,
            };
            if a_order != b_order {
                return a_order.cmp(&b_order);
            }
            let a_dist = (a.position - camera_pos).length_squared();
            let b_dist = (b.position - camera_pos).length_squared();
            a_dist.partial_cmp(&b_dist).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get volumes that need stencil marking (non-fullscreen, camera outside).
    pub fn stencil_mark_volumes(&self) -> Vec<&StencilLightVolume> {
        self.volumes
            .iter()
            .filter(|v| {
                v.volume_type != LightVolumeType::FullScreenQuad && !v.camera_inside
            })
            .collect()
    }

    /// Total volume count.
    pub fn total_count(&self) -> usize {
        self.volumes.len()
    }
}

impl Default for StencilLightVolumeManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Light pre-pass (deferred lighting)
// ---------------------------------------------------------------------------

/// Configuration for the light pre-pass technique.
///
/// In a light pre-pass, the lighting equation is split:
/// 1. Geometry pass: write normals + depth only (minimal G-buffer).
/// 2. Light pass: accumulate diffuse + specular lighting into a light buffer.
/// 3. Material pass: re-render geometry, sampling the light buffer and applying materials.
///
/// This allows more material variety than standard deferred since material data
/// doesn't need to fit in the G-buffer.
#[derive(Debug, Clone)]
pub struct LightPrePassConfig {
    /// Normal buffer format.
    pub normal_format: TextureFormat,
    /// Depth format.
    pub depth_format: TextureFormat,
    /// Diffuse light accumulation format.
    pub diffuse_light_format: TextureFormat,
    /// Specular light accumulation format.
    pub specular_light_format: TextureFormat,
    /// Whether to use half-resolution light pass.
    pub half_res_lighting: bool,
    /// Whether to reconstruct position from depth (saves a G-buffer target).
    pub reconstruct_position: bool,
    /// Shininess/specular power encoding mode.
    pub specular_power_mode: SpecularPowerMode,
}

/// How specular power is encoded in the light pre-pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecularPowerMode {
    /// Store specular power in the normal buffer alpha channel.
    InNormalAlpha,
    /// Use a fixed specular power for all surfaces.
    Fixed,
    /// Store in a separate 8-bit channel.
    Separate,
}

impl Default for LightPrePassConfig {
    fn default() -> Self {
        Self {
            normal_format: TextureFormat::Rgba16Float,
            depth_format: TextureFormat::Depth32Float,
            diffuse_light_format: TextureFormat::Rgba16Float,
            specular_light_format: TextureFormat::Rgba16Float,
            half_res_lighting: false,
            reconstruct_position: true,
            specular_power_mode: SpecularPowerMode::InNormalAlpha,
        }
    }
}

impl LightPrePassConfig {
    /// Mobile-friendly config with lower precision.
    pub fn mobile() -> Self {
        Self {
            normal_format: TextureFormat::Rgb10A2Unorm,
            depth_format: TextureFormat::Depth32Float,
            diffuse_light_format: TextureFormat::Rg11B10Float,
            specular_light_format: TextureFormat::Rg11B10Float,
            half_res_lighting: true,
            reconstruct_position: true,
            specular_power_mode: SpecularPowerMode::Fixed,
        }
    }

    /// High-quality config with full precision.
    pub fn high_quality() -> Self {
        Self {
            normal_format: TextureFormat::Rgba16Float,
            depth_format: TextureFormat::Depth32Float,
            diffuse_light_format: TextureFormat::Rgba16Float,
            specular_light_format: TextureFormat::Rgba16Float,
            half_res_lighting: false,
            reconstruct_position: true,
            specular_power_mode: SpecularPowerMode::InNormalAlpha,
        }
    }
}

/// The light pre-pass pipeline state and buffers.
#[derive(Debug)]
pub struct LightPrePass {
    /// Configuration.
    pub config: LightPrePassConfig,
    /// Normal render target handle.
    pub normal_rt: u64,
    /// Depth render target handle.
    pub depth_rt: u64,
    /// Diffuse light accumulation render target.
    pub diffuse_light_rt: u64,
    /// Specular light accumulation render target.
    pub specular_light_rt: u64,
    /// Resolution of the light buffers.
    pub light_buffer_resolution: UVec2,
    /// Resolution of the full-res geometry pass.
    pub geometry_resolution: UVec2,
    /// Inverse projection matrix for position reconstruction.
    pub inverse_projection: Mat4,
    /// Inverse view matrix for position reconstruction.
    pub inverse_view: Mat4,
    /// Number of lights processed in the last frame.
    pub lights_processed: u32,
    /// Current frame index.
    pub frame_index: u64,
}

impl LightPrePass {
    /// Create a new light pre-pass pipeline.
    pub fn new(config: LightPrePassConfig, width: u32, height: u32) -> Self {
        let light_res = if config.half_res_lighting {
            UVec2::new(width / 2, height / 2)
        } else {
            UVec2::new(width, height)
        };
        Self {
            config,
            normal_rt: 0,
            depth_rt: 0,
            diffuse_light_rt: 0,
            specular_light_rt: 0,
            light_buffer_resolution: light_res,
            geometry_resolution: UVec2::new(width, height),
            inverse_projection: Mat4::IDENTITY,
            inverse_view: Mat4::IDENTITY,
            lights_processed: 0,
            frame_index: 0,
        }
    }

    /// Resize the pipeline.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.geometry_resolution = UVec2::new(width, height);
        self.light_buffer_resolution = if self.config.half_res_lighting {
            UVec2::new(width / 2, height / 2)
        } else {
            UVec2::new(width, height)
        };
    }

    /// Update camera matrices for position reconstruction.
    pub fn update_matrices(&mut self, view: Mat4, projection: Mat4) {
        self.inverse_view = view.inverse();
        self.inverse_projection = projection.inverse();
    }

    /// Reconstruct world position from depth and UV.
    pub fn reconstruct_position(&self, uv: Vec2, depth: f32) -> Vec3 {
        let ndc = Vec4::new(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
        let view_pos = self.inverse_projection * ndc;
        let view_pos = view_pos / view_pos.w;
        let world_pos = self.inverse_view * view_pos;
        world_pos.truncate()
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.lights_processed = 0;
        self.frame_index += 1;
    }

    /// Record that a light was processed.
    pub fn record_light(&mut self) {
        self.lights_processed += 1;
    }
}

// ---------------------------------------------------------------------------
// Tiled deferred shading
// ---------------------------------------------------------------------------

/// A screen-space tile that contains a list of affecting lights.
#[derive(Debug, Clone)]
pub struct Tile {
    /// Tile X index in the grid.
    pub x: u32,
    /// Tile Y index in the grid.
    pub y: u32,
    /// Screen-space bounds of this tile (min_x, min_y, max_x, max_y).
    pub bounds: [u32; 4],
    /// Indices of lights affecting this tile.
    pub light_indices: Vec<u16>,
    /// Min depth in this tile.
    pub min_depth: f32,
    /// Max depth in this tile.
    pub max_depth: f32,
}

impl Tile {
    /// Create a new empty tile.
    pub fn new(x: u32, y: u32, min_x: u32, min_y: u32, max_x: u32, max_y: u32) -> Self {
        Self {
            x,
            y,
            bounds: [min_x, min_y, max_x, max_y],
            light_indices: Vec::new(),
            min_depth: 1.0,
            max_depth: 0.0,
        }
    }

    /// Clear the tile for a new frame.
    pub fn clear(&mut self) {
        self.light_indices.clear();
        self.min_depth = 1.0;
        self.max_depth = 0.0;
    }

    /// Add a light index to this tile.
    pub fn add_light(&mut self, index: u16) {
        if self.light_indices.len() < MAX_LIGHTS_PER_TILE {
            self.light_indices.push(index);
        }
    }

    /// Returns the number of lights in this tile.
    pub fn light_count(&self) -> usize {
        self.light_indices.len()
    }

    /// Update the depth range from a depth sample.
    pub fn update_depth(&mut self, depth: f32) {
        if depth < self.min_depth {
            self.min_depth = depth;
        }
        if depth > self.max_depth {
            self.max_depth = depth;
        }
    }

    /// Check if a sphere (light volume) intersects this tile's frustum.
    pub fn intersects_sphere_screen(&self, center_screen: Vec2, radius_screen: f32) -> bool {
        let tile_min = Vec2::new(self.bounds[0] as f32, self.bounds[1] as f32);
        let tile_max = Vec2::new(self.bounds[2] as f32, self.bounds[3] as f32);
        let closest = Vec2::new(
            center_screen.x.clamp(tile_min.x, tile_max.x),
            center_screen.y.clamp(tile_min.y, tile_max.y),
        );
        let dist_sq = (center_screen - closest).length_squared();
        dist_sq < radius_screen * radius_screen
    }
}

/// GPU-friendly light data for tiled/clustered shading.
#[derive(Debug, Clone, Copy)]
pub struct TiledLight {
    /// Position (XYZ) and radius (W) in view space.
    pub position_radius: Vec4,
    /// Color (RGB) and intensity (A).
    pub color_intensity: Vec4,
    /// Direction (XYZ) and cos(outer_angle) (W) -- for spot lights.
    pub direction_angle: Vec4,
    /// Light type (0=point, 1=spot, 2=directional) and additional flags.
    pub type_flags: u32,
    /// Shadow map index (-1 if no shadow).
    pub shadow_index: i32,
    /// Attenuation parameters.
    pub attenuation: Vec2,
}

impl TiledLight {
    /// Create a point light for tiled shading.
    pub fn point(position: Vec3, radius: f32, color: Vec3, intensity: f32) -> Self {
        Self {
            position_radius: Vec4::new(position.x, position.y, position.z, radius),
            color_intensity: Vec4::new(color.x, color.y, color.z, intensity),
            direction_angle: Vec4::ZERO,
            type_flags: 0,
            shadow_index: -1,
            attenuation: Vec2::new(1.0, radius * radius),
        }
    }

    /// Create a spot light for tiled shading.
    pub fn spot(
        position: Vec3,
        direction: Vec3,
        radius: f32,
        outer_angle: f32,
        inner_angle: f32,
        color: Vec3,
        intensity: f32,
    ) -> Self {
        Self {
            position_radius: Vec4::new(position.x, position.y, position.z, radius),
            color_intensity: Vec4::new(color.x, color.y, color.z, intensity),
            direction_angle: Vec4::new(direction.x, direction.y, direction.z, outer_angle.cos()),
            type_flags: 1,
            shadow_index: -1,
            attenuation: Vec2::new(inner_angle.cos(), outer_angle.cos()),
        }
    }

    /// Create a directional light for tiled shading.
    pub fn directional(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            position_radius: Vec4::new(0.0, 0.0, 0.0, f32::MAX),
            color_intensity: Vec4::new(color.x, color.y, color.z, intensity),
            direction_angle: Vec4::new(direction.x, direction.y, direction.z, -1.0),
            type_flags: 2,
            shadow_index: -1,
            attenuation: Vec2::ZERO,
        }
    }

    /// Set the shadow map index.
    pub fn with_shadow(mut self, index: i32) -> Self {
        self.shadow_index = index;
        self
    }

    /// Check if this light is a directional light.
    pub fn is_directional(&self) -> bool {
        self.type_flags == 2
    }

    /// Get the light radius.
    pub fn radius(&self) -> f32 {
        self.position_radius.w
    }

    /// Get the light position (view space).
    pub fn position(&self) -> Vec3 {
        self.position_radius.truncate()
    }
}

/// Configuration for the tiled deferred pipeline.
#[derive(Debug, Clone)]
pub struct TiledDeferredConfig {
    /// Tile size in pixels (usually 16 or 32).
    pub tile_size: u32,
    /// Whether to use depth bounds for tighter culling.
    pub use_depth_bounds: bool,
    /// Whether to use 2.5D culling (per-tile depth range).
    pub use_depth_range_culling: bool,
    /// Maximum lights per tile before fallback.
    pub max_lights_per_tile: usize,
    /// Whether to use compute shader for light culling.
    pub compute_culling: bool,
    /// Whether to output a debug heatmap of light counts.
    pub debug_heatmap: bool,
    /// Debug heatmap maximum light count for colour scale.
    pub debug_heatmap_max: u32,
}

impl Default for TiledDeferredConfig {
    fn default() -> Self {
        Self {
            tile_size: DEFAULT_TILE_SIZE,
            use_depth_bounds: true,
            use_depth_range_culling: true,
            max_lights_per_tile: MAX_LIGHTS_PER_TILE,
            compute_culling: true,
            debug_heatmap: false,
            debug_heatmap_max: 64,
        }
    }
}

/// The tiled deferred shading pipeline.
#[derive(Debug)]
pub struct TiledDeferredPipeline {
    /// Configuration.
    pub config: TiledDeferredConfig,
    /// Screen resolution.
    pub resolution: UVec2,
    /// Grid dimensions in tiles.
    pub grid_dims: UVec2,
    /// All tiles in row-major order.
    pub tiles: Vec<Tile>,
    /// All lights for the current frame.
    pub lights: Vec<TiledLight>,
    /// Flat light index list (GPU buffer data).
    /// Format: for each tile, a contiguous run of u16 light indices.
    pub light_index_list: Vec<u16>,
    /// Per-tile offset+count into the light_index_list.
    /// Each entry: (offset, count).
    pub tile_light_table: Vec<(u32, u32)>,
    /// Statistics for the last frame.
    pub stats: TiledDeferredStats,
}

/// Statistics for the tiled deferred pipeline.
#[derive(Debug, Clone, Default)]
pub struct TiledDeferredStats {
    /// Total number of tiles.
    pub total_tiles: u32,
    /// Number of tiles with at least one light.
    pub lit_tiles: u32,
    /// Maximum lights in any single tile.
    pub max_lights_in_tile: u32,
    /// Average lights per lit tile.
    pub avg_lights_per_tile: f32,
    /// Total light-tile pairs (sum of all per-tile light counts).
    pub total_light_tile_pairs: u32,
    /// Number of lights culled by depth range.
    pub depth_culled_count: u32,
    /// Time in microseconds for light culling.
    pub culling_time_us: f64,
}

impl TiledDeferredPipeline {
    /// Create a new tiled deferred pipeline.
    pub fn new(config: TiledDeferredConfig, width: u32, height: u32) -> Self {
        let grid_x = (width + config.tile_size - 1) / config.tile_size;
        let grid_y = (height + config.tile_size - 1) / config.tile_size;
        let tile_count = (grid_x * grid_y) as usize;

        let mut tiles = Vec::with_capacity(tile_count);
        for ty in 0..grid_y {
            for tx in 0..grid_x {
                let min_x = tx * config.tile_size;
                let min_y = ty * config.tile_size;
                let max_x = ((tx + 1) * config.tile_size).min(width);
                let max_y = ((ty + 1) * config.tile_size).min(height);
                tiles.push(Tile::new(tx, ty, min_x, min_y, max_x, max_y));
            }
        }

        Self {
            config,
            resolution: UVec2::new(width, height),
            grid_dims: UVec2::new(grid_x, grid_y),
            tiles,
            lights: Vec::new(),
            light_index_list: Vec::new(),
            tile_light_table: vec![(0, 0); tile_count],
            stats: TiledDeferredStats::default(),
        }
    }

    /// Resize the pipeline for a new screen resolution.
    pub fn resize(&mut self, width: u32, height: u32) {
        let grid_x = (width + self.config.tile_size - 1) / self.config.tile_size;
        let grid_y = (height + self.config.tile_size - 1) / self.config.tile_size;
        let tile_count = (grid_x * grid_y) as usize;

        self.resolution = UVec2::new(width, height);
        self.grid_dims = UVec2::new(grid_x, grid_y);
        self.tiles.clear();
        for ty in 0..grid_y {
            for tx in 0..grid_x {
                let min_x = tx * self.config.tile_size;
                let min_y = ty * self.config.tile_size;
                let max_x = ((tx + 1) * self.config.tile_size).min(width);
                let max_y = ((ty + 1) * self.config.tile_size).min(height);
                self.tiles.push(Tile::new(tx, ty, min_x, min_y, max_x, max_y));
            }
        }
        self.tile_light_table.resize(tile_count, (0, 0));
    }

    /// Begin a new frame: clear tiles and lights.
    pub fn begin_frame(&mut self) {
        for tile in &mut self.tiles {
            tile.clear();
        }
        self.lights.clear();
        self.light_index_list.clear();
        for entry in &mut self.tile_light_table {
            *entry = (0, 0);
        }
        self.stats = TiledDeferredStats::default();
        self.stats.total_tiles = self.tiles.len() as u32;
    }

    /// Add a light for this frame.
    pub fn add_light(&mut self, light: TiledLight) {
        if self.lights.len() < MAX_TILED_LIGHTS {
            self.lights.push(light);
        }
    }

    /// Perform CPU-side light culling against all tiles.
    ///
    /// This is the fallback path when compute culling is not available.
    /// For each light, test against each tile's screen-space bounds.
    pub fn cull_lights_cpu(&mut self, view: Mat4, projection: Mat4) {
        let start = std::time::Instant::now();
        let vp = projection * view;
        let half_res = Vec2::new(self.resolution.x as f32 * 0.5, self.resolution.y as f32 * 0.5);

        for (light_idx, light) in self.lights.iter().enumerate() {
            if light.is_directional() {
                // Directional lights affect all tiles
                for tile in &mut self.tiles {
                    tile.add_light(light_idx as u16);
                }
                continue;
            }

            let pos = light.position();
            let radius = light.radius();

            // Project light sphere center to screen space
            let clip = vp * Vec4::new(pos.x, pos.y, pos.z, 1.0);
            if clip.w <= EPSILON {
                continue; // Behind camera
            }
            let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
            if ndc.z < 0.0 || ndc.z > 1.0 {
                continue; // Outside depth range (conservative)
            }

            let screen_center = Vec2::new(
                (ndc.x + 1.0) * half_res.x,
                (1.0 - ndc.y) * half_res.y,
            );

            // Approximate screen-space radius
            let dist = (view * Vec4::new(pos.x, pos.y, pos.z, 1.0)).z.abs();
            let screen_radius = if dist > EPSILON {
                (radius / dist) * half_res.x
            } else {
                half_res.x * 2.0
            };

            let mut depth_culled = 0u32;
            for tile in &mut self.tiles {
                if tile.intersects_sphere_screen(screen_center, screen_radius) {
                    // Depth range culling
                    if self.config.use_depth_range_culling {
                        let light_z = ndc.z;
                        let light_z_min = (light_z - radius * 0.01).max(0.0);
                        let light_z_max = (light_z + radius * 0.01).min(1.0);
                        if light_z_min > tile.max_depth || light_z_max < tile.min_depth {
                            depth_culled += 1;
                            continue;
                        }
                    }
                    tile.add_light(light_idx as u16);
                }
            }
            self.stats.depth_culled_count += depth_culled;
        }

        // Build the flat light index list and per-tile table.
        self.build_light_index_list();

        let elapsed = start.elapsed();
        self.stats.culling_time_us = elapsed.as_secs_f64() * 1_000_000.0;
    }

    /// Build the flat GPU-friendly light index list from per-tile light lists.
    fn build_light_index_list(&mut self) {
        self.light_index_list.clear();
        let mut lit_tiles = 0u32;
        let mut max_lights = 0u32;
        let mut total_pairs = 0u32;

        for (tile_idx, tile) in self.tiles.iter().enumerate() {
            let offset = self.light_index_list.len() as u32;
            let count = tile.light_count() as u32;
            if tile_idx < self.tile_light_table.len() {
                self.tile_light_table[tile_idx] = (offset, count);
            }
            self.light_index_list.extend_from_slice(&tile.light_indices);
            if count > 0 {
                lit_tiles += 1;
                if count > max_lights {
                    max_lights = count;
                }
                total_pairs += count;
            }
        }

        self.stats.lit_tiles = lit_tiles;
        self.stats.max_lights_in_tile = max_lights;
        self.stats.total_light_tile_pairs = total_pairs;
        self.stats.avg_lights_per_tile = if lit_tiles > 0 {
            total_pairs as f32 / lit_tiles as f32
        } else {
            0.0
        };
    }

    /// Get the tile at the given pixel coordinate.
    pub fn tile_at_pixel(&self, x: u32, y: u32) -> Option<&Tile> {
        let tx = x / self.config.tile_size;
        let ty = y / self.config.tile_size;
        if tx < self.grid_dims.x && ty < self.grid_dims.y {
            let idx = (ty * self.grid_dims.x + tx) as usize;
            self.tiles.get(idx)
        } else {
            None
        }
    }

    /// Get the number of lights affecting a specific pixel.
    pub fn light_count_at_pixel(&self, x: u32, y: u32) -> usize {
        self.tile_at_pixel(x, y)
            .map(|t| t.light_count())
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Clustered shading
// ---------------------------------------------------------------------------

/// A cluster in the 3D grid (tile + depth slice).
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Tile X index.
    pub tile_x: u32,
    /// Tile Y index.
    pub tile_y: u32,
    /// Depth slice index.
    pub slice: u32,
    /// AABB min in view space.
    pub aabb_min: Vec3,
    /// AABB max in view space.
    pub aabb_max: Vec3,
    /// Indices of lights affecting this cluster.
    pub light_indices: Vec<u16>,
}

impl Cluster {
    /// Create a new empty cluster.
    pub fn new(tile_x: u32, tile_y: u32, slice: u32) -> Self {
        Self {
            tile_x,
            tile_y,
            slice,
            aabb_min: Vec3::ZERO,
            aabb_max: Vec3::ZERO,
            light_indices: Vec::new(),
        }
    }

    /// Clear lights for a new frame.
    pub fn clear(&mut self) {
        self.light_indices.clear();
    }

    /// Add a light to this cluster.
    pub fn add_light(&mut self, index: u16) {
        if self.light_indices.len() < MAX_LIGHTS_PER_CLUSTER {
            self.light_indices.push(index);
        }
    }

    /// Test if a view-space sphere intersects this cluster's AABB.
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        let closest = Vec3::new(
            center.x.clamp(self.aabb_min.x, self.aabb_max.x),
            center.y.clamp(self.aabb_min.y, self.aabb_max.y),
            center.z.clamp(self.aabb_min.z, self.aabb_max.z),
        );
        (center - closest).length_squared() < radius * radius
    }

    /// Get the number of lights.
    pub fn light_count(&self) -> usize {
        self.light_indices.len()
    }
}

/// Configuration for clustered shading.
#[derive(Debug, Clone)]
pub struct ClusteredShadingConfig {
    /// Tile size in pixels.
    pub tile_size: u32,
    /// Number of depth slices.
    pub depth_slices: u32,
    /// Near plane distance.
    pub near_plane: f32,
    /// Far plane distance.
    pub far_plane: f32,
    /// Whether to use logarithmic depth slicing (better distribution).
    pub log_depth_slicing: bool,
    /// Maximum lights per cluster.
    pub max_lights_per_cluster: usize,
    /// Whether to enable debug visualization.
    pub debug_visualization: bool,
}

impl Default for ClusteredShadingConfig {
    fn default() -> Self {
        Self {
            tile_size: DEFAULT_TILE_SIZE,
            depth_slices: DEFAULT_CLUSTER_DEPTH_SLICES,
            near_plane: 0.1,
            far_plane: 1000.0,
            log_depth_slicing: true,
            max_lights_per_cluster: MAX_LIGHTS_PER_CLUSTER,
            debug_visualization: false,
        }
    }
}

impl ClusteredShadingConfig {
    /// Compute the depth at a given slice index.
    pub fn slice_depth(&self, slice: u32) -> f32 {
        let t = slice as f32 / self.depth_slices as f32;
        if self.log_depth_slicing {
            self.near_plane * (self.far_plane / self.near_plane).powf(t)
        } else {
            self.near_plane + (self.far_plane - self.near_plane) * t
        }
    }

    /// Find the slice index for a given view-space depth.
    pub fn depth_to_slice(&self, depth: f32) -> u32 {
        if depth <= self.near_plane {
            return 0;
        }
        if depth >= self.far_plane {
            return self.depth_slices - 1;
        }
        let t = if self.log_depth_slicing {
            (depth / self.near_plane).ln() / (self.far_plane / self.near_plane).ln()
        } else {
            (depth - self.near_plane) / (self.far_plane - self.near_plane)
        };
        ((t * self.depth_slices as f32) as u32).min(self.depth_slices - 1)
    }

    /// Total number of clusters.
    pub fn total_clusters(&self, grid_x: u32, grid_y: u32) -> u32 {
        grid_x * grid_y * self.depth_slices
    }
}

/// The clustered shading pipeline.
#[derive(Debug)]
pub struct ClusteredShadingPipeline {
    /// Configuration.
    pub config: ClusteredShadingConfig,
    /// Screen resolution.
    pub resolution: UVec2,
    /// Grid dimensions in tiles (X, Y).
    pub grid_dims: UVec2,
    /// All clusters in a flat array: [tile_y][tile_x][slice].
    pub clusters: Vec<Cluster>,
    /// All lights for the current frame.
    pub lights: Vec<TiledLight>,
    /// Statistics for the last frame.
    pub stats: ClusteredShadingStats,
}

/// Statistics for clustered shading.
#[derive(Debug, Clone, Default)]
pub struct ClusteredShadingStats {
    /// Total number of clusters.
    pub total_clusters: u32,
    /// Active clusters (with at least one light).
    pub active_clusters: u32,
    /// Maximum lights in any cluster.
    pub max_lights_in_cluster: u32,
    /// Average lights per active cluster.
    pub avg_lights_per_cluster: f32,
    /// Total light-cluster assignments.
    pub total_assignments: u32,
    /// Culling time in microseconds.
    pub culling_time_us: f64,
}

impl ClusteredShadingPipeline {
    /// Create a new clustered shading pipeline.
    pub fn new(config: ClusteredShadingConfig, width: u32, height: u32) -> Self {
        let grid_x = (width + config.tile_size - 1) / config.tile_size;
        let grid_y = (height + config.tile_size - 1) / config.tile_size;
        let total = config.total_clusters(grid_x, grid_y) as usize;

        let mut clusters = Vec::with_capacity(total);
        for ty in 0..grid_y {
            for tx in 0..grid_x {
                for s in 0..config.depth_slices {
                    clusters.push(Cluster::new(tx, ty, s));
                }
            }
        }

        Self {
            config,
            resolution: UVec2::new(width, height),
            grid_dims: UVec2::new(grid_x, grid_y),
            clusters,
            lights: Vec::new(),
            stats: ClusteredShadingStats::default(),
        }
    }

    /// Resize the pipeline.
    pub fn resize(&mut self, width: u32, height: u32) {
        let grid_x = (width + self.config.tile_size - 1) / self.config.tile_size;
        let grid_y = (height + self.config.tile_size - 1) / self.config.tile_size;

        self.resolution = UVec2::new(width, height);
        self.grid_dims = UVec2::new(grid_x, grid_y);

        let total = self.config.total_clusters(grid_x, grid_y) as usize;
        self.clusters.clear();
        self.clusters.reserve(total);
        for ty in 0..grid_y {
            for tx in 0..grid_x {
                for s in 0..self.config.depth_slices {
                    self.clusters.push(Cluster::new(tx, ty, s));
                }
            }
        }
    }

    /// Build cluster AABBs from the camera's inverse projection.
    pub fn build_cluster_aabbs(&mut self, inv_projection: Mat4) {
        let tile_size_f = self.config.tile_size as f32;
        let res_x = self.resolution.x as f32;
        let res_y = self.resolution.y as f32;

        for cluster in &mut self.clusters {
            let tx = cluster.tile_x;
            let ty = cluster.tile_y;
            let s = cluster.slice;

            let min_x = tx as f32 * tile_size_f / res_x * 2.0 - 1.0;
            let max_x = ((tx + 1) as f32 * tile_size_f).min(res_x) / res_x * 2.0 - 1.0;
            let min_y = ty as f32 * tile_size_f / res_y * 2.0 - 1.0;
            let max_y = ((ty + 1) as f32 * tile_size_f).min(res_y) / res_y * 2.0 - 1.0;

            let near_depth = self.config.slice_depth(s);
            let far_depth = self.config.slice_depth(s + 1);

            // Convert NDC corners to view space
            let corners = [
                unproject_point(inv_projection, Vec3::new(min_x, min_y, 0.0)),
                unproject_point(inv_projection, Vec3::new(max_x, min_y, 0.0)),
                unproject_point(inv_projection, Vec3::new(min_x, max_y, 0.0)),
                unproject_point(inv_projection, Vec3::new(max_x, max_y, 0.0)),
            ];

            let mut aabb_min = Vec3::splat(f32::MAX);
            let mut aabb_max = Vec3::splat(f32::MIN);

            for corner in &corners {
                let dir = corner.normalize_or_zero();
                let near_pt = dir * near_depth;
                let far_pt = dir * far_depth;
                aabb_min = aabb_min.min(near_pt).min(far_pt);
                aabb_max = aabb_max.max(near_pt).max(far_pt);
            }

            cluster.aabb_min = aabb_min;
            cluster.aabb_max = aabb_max;
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        for cluster in &mut self.clusters {
            cluster.clear();
        }
        self.lights.clear();
        self.stats = ClusteredShadingStats::default();
        self.stats.total_clusters = self.clusters.len() as u32;
    }

    /// Add a light.
    pub fn add_light(&mut self, light: TiledLight) {
        if self.lights.len() < MAX_TILED_LIGHTS {
            self.lights.push(light);
        }
    }

    /// Perform CPU-side light-to-cluster assignment.
    pub fn assign_lights_cpu(&mut self, view: Mat4) {
        let start = std::time::Instant::now();

        for (light_idx, light) in self.lights.iter().enumerate() {
            if light.is_directional() {
                for cluster in &mut self.clusters {
                    cluster.add_light(light_idx as u16);
                }
                continue;
            }

            let pos_world = light.position();
            let pos_view = (view * Vec4::new(pos_world.x, pos_world.y, pos_world.z, 1.0)).truncate();
            let radius = light.radius();

            for cluster in &mut self.clusters {
                if cluster.intersects_sphere(pos_view, radius) {
                    cluster.add_light(light_idx as u16);
                }
            }
        }

        // Compute statistics
        let mut active = 0u32;
        let mut max_lights = 0u32;
        let mut total = 0u32;
        for cluster in &self.clusters {
            let count = cluster.light_count() as u32;
            if count > 0 {
                active += 1;
                total += count;
                if count > max_lights {
                    max_lights = count;
                }
            }
        }
        self.stats.active_clusters = active;
        self.stats.max_lights_in_cluster = max_lights;
        self.stats.total_assignments = total;
        self.stats.avg_lights_per_cluster = if active > 0 {
            total as f32 / active as f32
        } else {
            0.0
        };

        let elapsed = start.elapsed();
        self.stats.culling_time_us = elapsed.as_secs_f64() * 1_000_000.0;
    }

    /// Get the cluster for a given screen pixel and view-space depth.
    pub fn cluster_at(&self, pixel_x: u32, pixel_y: u32, depth: f32) -> Option<&Cluster> {
        let tx = pixel_x / self.config.tile_size;
        let ty = pixel_y / self.config.tile_size;
        let s = self.config.depth_to_slice(depth);
        if tx < self.grid_dims.x && ty < self.grid_dims.y {
            let idx = ((ty * self.grid_dims.x + tx) * self.config.depth_slices + s) as usize;
            self.clusters.get(idx)
        } else {
            None
        }
    }

    /// Get the cluster index for a given tile and slice.
    pub fn cluster_index(&self, tile_x: u32, tile_y: u32, slice: u32) -> usize {
        ((tile_y * self.grid_dims.x + tile_x) * self.config.depth_slices + slice) as usize
    }
}

/// Unproject a point from NDC to view space.
fn unproject_point(inv_projection: Mat4, ndc: Vec3) -> Vec3 {
    let p = inv_projection * Vec4::new(ndc.x, ndc.y, ndc.z, 1.0);
    if p.w.abs() > EPSILON {
        Vec3::new(p.x / p.w, p.y / p.w, p.z / p.w)
    } else {
        Vec3::ZERO
    }
}

// ---------------------------------------------------------------------------
// Cluster debug visualization
// ---------------------------------------------------------------------------

/// Debug visualization mode for the cluster grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterDebugMode {
    /// No debug visualization.
    Off,
    /// Show light count per cluster as a heatmap overlay.
    LightCountHeatmap,
    /// Show cluster slice boundaries as colour bands.
    SliceBoundaries,
    /// Show active vs inactive clusters.
    ActiveClusters,
    /// Show lights per tile (2D) ignoring depth.
    TileLightCount,
    /// Show the depth slice index as a gradient.
    DepthSliceGradient,
    /// Show cluster AABBs as wireframes (3D).
    WireframeAABB,
}

/// A colour used for debug visualization (linear RGBA).
#[derive(Debug, Clone, Copy)]
pub struct DebugColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl DebugColor {
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Self = Self { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Self = Self { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const CYAN: Self = Self { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const MAGENTA: Self = Self { r: 1.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const TRANSPARENT: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

    /// Create a new debug colour.
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Lerp between two colours.
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: a.r + (b.r - a.r) * t,
            g: a.g + (b.g - a.g) * t,
            b: a.b + (b.b - a.b) * t,
            a: a.a + (b.a - a.a) * t,
        }
    }

    /// Generate a heatmap colour from a 0-1 value (blue -> green -> yellow -> red).
    pub fn heatmap(value: f32) -> Self {
        let v = value.clamp(0.0, 1.0);
        if v < 0.25 {
            Self::lerp(Self::BLUE, Self::CYAN, v * 4.0)
        } else if v < 0.5 {
            Self::lerp(Self::CYAN, Self::GREEN, (v - 0.25) * 4.0)
        } else if v < 0.75 {
            Self::lerp(Self::GREEN, Self::YELLOW, (v - 0.5) * 4.0)
        } else {
            Self::lerp(Self::YELLOW, Self::RED, (v - 0.75) * 4.0)
        }
    }

    /// Convert to Vec4.
    pub fn to_vec4(self) -> Vec4 {
        Vec4::new(self.r, self.g, self.b, self.a)
    }
}

/// Cluster debug visualizer that generates a colour per pixel.
#[derive(Debug)]
pub struct ClusterDebugVisualizer {
    /// Current debug mode.
    pub mode: ClusterDebugMode,
    /// Maximum light count for heatmap normalisation.
    pub heatmap_max: u32,
    /// Overlay opacity.
    pub overlay_alpha: f32,
    /// Whether to show grid lines between tiles.
    pub show_grid_lines: bool,
    /// Grid line width in pixels.
    pub grid_line_width: u32,
    /// Grid line colour.
    pub grid_line_color: DebugColor,
    /// Whether to show a text label per tile.
    pub show_labels: bool,
    /// Slice colours for the SliceBoundaries mode.
    pub slice_colors: Vec<DebugColor>,
}

impl Default for ClusterDebugVisualizer {
    fn default() -> Self {
        let mut slice_colors = Vec::with_capacity(DEFAULT_CLUSTER_DEPTH_SLICES as usize);
        for i in 0..DEFAULT_CLUSTER_DEPTH_SLICES {
            let hue = i as f32 / DEFAULT_CLUSTER_DEPTH_SLICES as f32;
            slice_colors.push(DebugColor::heatmap(hue));
        }
        Self {
            mode: ClusterDebugMode::Off,
            heatmap_max: 64,
            overlay_alpha: 0.5,
            show_grid_lines: true,
            grid_line_width: 1,
            grid_line_color: DebugColor::WHITE,
            show_labels: false,
            slice_colors,
        }
    }
}

impl ClusterDebugVisualizer {
    /// Create a new visualizer with the given mode.
    pub fn new(mode: ClusterDebugMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Get the debug colour for a pixel given the cluster information.
    pub fn pixel_color(
        &self,
        pixel_x: u32,
        pixel_y: u32,
        tile_x: u32,
        tile_y: u32,
        slice: u32,
        light_count: u32,
        tile_size: u32,
    ) -> DebugColor {
        // Grid line check
        if self.show_grid_lines {
            let lx = pixel_x % tile_size;
            let ly = pixel_y % tile_size;
            if lx < self.grid_line_width || ly < self.grid_line_width {
                return self.grid_line_color;
            }
        }

        match self.mode {
            ClusterDebugMode::Off => DebugColor::TRANSPARENT,
            ClusterDebugMode::LightCountHeatmap => {
                let v = light_count as f32 / self.heatmap_max as f32;
                let mut c = DebugColor::heatmap(v);
                c.a = self.overlay_alpha;
                c
            }
            ClusterDebugMode::SliceBoundaries => {
                let idx = (slice as usize).min(self.slice_colors.len().saturating_sub(1));
                let mut c = if idx < self.slice_colors.len() {
                    self.slice_colors[idx]
                } else {
                    DebugColor::WHITE
                };
                c.a = self.overlay_alpha;
                c
            }
            ClusterDebugMode::ActiveClusters => {
                let mut c = if light_count > 0 {
                    DebugColor::GREEN
                } else {
                    DebugColor::RED
                };
                c.a = self.overlay_alpha * 0.3;
                c
            }
            ClusterDebugMode::TileLightCount => {
                let v = light_count as f32 / self.heatmap_max as f32;
                let mut c = DebugColor::heatmap(v);
                c.a = self.overlay_alpha;
                c
            }
            ClusterDebugMode::DepthSliceGradient => {
                let v = slice as f32 / DEFAULT_CLUSTER_DEPTH_SLICES as f32;
                DebugColor::new(v, v, 1.0 - v, self.overlay_alpha)
            }
            ClusterDebugMode::WireframeAABB => DebugColor::TRANSPARENT,
        }
    }

    /// Generate wireframe lines for cluster AABBs (for 3D debug drawing).
    pub fn generate_wireframe_lines(
        &self,
        clusters: &[Cluster],
        max_clusters: usize,
    ) -> Vec<(Vec3, Vec3, DebugColor)> {
        let mut lines = Vec::new();
        let count = clusters.len().min(max_clusters);
        for cluster in clusters.iter().take(count) {
            if cluster.light_count() == 0 {
                continue;
            }
            let min = cluster.aabb_min;
            let max = cluster.aabb_max;
            let color = DebugColor::heatmap(
                cluster.light_count() as f32 / self.heatmap_max as f32,
            );

            // 12 edges of the AABB
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

            let edges: [(usize, usize); 12] = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ];

            for (a, b) in edges {
                lines.push((corners[a], corners[b], color));
            }
        }
        lines
    }

    /// Is the visualizer currently active?
    pub fn is_active(&self) -> bool {
        self.mode != ClusterDebugMode::Off
    }

    /// Toggle the mode to the next one.
    pub fn cycle_mode(&mut self) {
        self.mode = match self.mode {
            ClusterDebugMode::Off => ClusterDebugMode::LightCountHeatmap,
            ClusterDebugMode::LightCountHeatmap => ClusterDebugMode::SliceBoundaries,
            ClusterDebugMode::SliceBoundaries => ClusterDebugMode::ActiveClusters,
            ClusterDebugMode::ActiveClusters => ClusterDebugMode::TileLightCount,
            ClusterDebugMode::TileLightCount => ClusterDebugMode::DepthSliceGradient,
            ClusterDebugMode::DepthSliceGradient => ClusterDebugMode::WireframeAABB,
            ClusterDebugMode::WireframeAABB => ClusterDebugMode::Off,
        };
    }
}

// ---------------------------------------------------------------------------
// Enhanced deferred rendering system (ECS integration)
// ---------------------------------------------------------------------------

/// Quality preset for the deferred V2 pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferredQualityPreset {
    /// Minimal: thin G-buffer, no tiling, stencil volumes only.
    Low,
    /// Standard: thin G-buffer with tiled deferred.
    Medium,
    /// High: full G-buffer with clustered shading.
    High,
    /// Ultra: full G-buffer, clustered shading, debug features available.
    Ultra,
}

/// Main configuration for the enhanced deferred renderer.
#[derive(Debug, Clone)]
pub struct DeferredV2Config {
    /// Quality preset.
    pub quality: DeferredQualityPreset,
    /// Whether to use the thin G-buffer layout.
    pub use_thin_gbuffer: bool,
    /// Whether to use the light pre-pass technique.
    pub use_light_prepass: bool,
    /// Whether to use tiled deferred.
    pub use_tiled_deferred: bool,
    /// Whether to use clustered shading.
    pub use_clustered_shading: bool,
    /// Whether to use stencil light volumes.
    pub use_stencil_volumes: bool,
    /// Tile size for tiled/clustered paths.
    pub tile_size: u32,
    /// Number of depth slices for clustered path.
    pub depth_slices: u32,
    /// Near plane.
    pub near_plane: f32,
    /// Far plane.
    pub far_plane: f32,
    /// Whether to enable debug visualization.
    pub debug_mode: ClusterDebugMode,
}

impl Default for DeferredV2Config {
    fn default() -> Self {
        Self {
            quality: DeferredQualityPreset::High,
            use_thin_gbuffer: true,
            use_light_prepass: false,
            use_tiled_deferred: false,
            use_clustered_shading: true,
            use_stencil_volumes: true,
            tile_size: DEFAULT_TILE_SIZE,
            depth_slices: DEFAULT_CLUSTER_DEPTH_SLICES,
            near_plane: 0.1,
            far_plane: 1000.0,
            debug_mode: ClusterDebugMode::Off,
        }
    }
}

impl DeferredV2Config {
    /// Create a config from a quality preset.
    pub fn from_preset(preset: DeferredQualityPreset) -> Self {
        match preset {
            DeferredQualityPreset::Low => Self {
                quality: preset,
                use_thin_gbuffer: true,
                use_light_prepass: false,
                use_tiled_deferred: false,
                use_clustered_shading: false,
                use_stencil_volumes: true,
                tile_size: 32,
                depth_slices: 8,
                ..Default::default()
            },
            DeferredQualityPreset::Medium => Self {
                quality: preset,
                use_thin_gbuffer: true,
                use_light_prepass: false,
                use_tiled_deferred: true,
                use_clustered_shading: false,
                use_stencil_volumes: true,
                tile_size: 16,
                depth_slices: 16,
                ..Default::default()
            },
            DeferredQualityPreset::High => Self {
                quality: preset,
                use_thin_gbuffer: true,
                use_light_prepass: false,
                use_tiled_deferred: false,
                use_clustered_shading: true,
                use_stencil_volumes: true,
                tile_size: 16,
                depth_slices: 24,
                ..Default::default()
            },
            DeferredQualityPreset::Ultra => Self {
                quality: preset,
                use_thin_gbuffer: false,
                use_light_prepass: false,
                use_tiled_deferred: false,
                use_clustered_shading: true,
                use_stencil_volumes: true,
                tile_size: 8,
                depth_slices: 32,
                ..Default::default()
            },
        }
    }
}

/// The main enhanced deferred renderer component.
#[derive(Debug)]
pub struct DeferredRendererV2 {
    /// Configuration.
    pub config: DeferredV2Config,
    /// Thin G-buffer (if enabled).
    pub thin_gbuffer: Option<ThinGBuffer>,
    /// Light pre-pass pipeline (if enabled).
    pub light_prepass: Option<LightPrePass>,
    /// Tiled deferred pipeline (if enabled).
    pub tiled_pipeline: Option<TiledDeferredPipeline>,
    /// Clustered shading pipeline (if enabled).
    pub clustered_pipeline: Option<ClusteredShadingPipeline>,
    /// Stencil light volume manager.
    pub stencil_volumes: StencilLightVolumeManager,
    /// Debug visualizer.
    pub debug_visualizer: ClusterDebugVisualizer,
    /// Current frame index.
    pub frame_index: u64,
    /// Screen resolution.
    pub resolution: UVec2,
}

impl DeferredRendererV2 {
    /// Create a new deferred renderer with the given configuration.
    pub fn new(config: DeferredV2Config, width: u32, height: u32) -> Self {
        let thin_gbuffer = if config.use_thin_gbuffer {
            Some(ThinGBuffer::new(ThinGBufferLayout::new(width, height)))
        } else {
            None
        };

        let light_prepass = if config.use_light_prepass {
            Some(LightPrePass::new(LightPrePassConfig::default(), width, height))
        } else {
            None
        };

        let tiled_pipeline = if config.use_tiled_deferred {
            Some(TiledDeferredPipeline::new(
                TiledDeferredConfig {
                    tile_size: config.tile_size,
                    ..Default::default()
                },
                width,
                height,
            ))
        } else {
            None
        };

        let clustered_pipeline = if config.use_clustered_shading {
            Some(ClusteredShadingPipeline::new(
                ClusteredShadingConfig {
                    tile_size: config.tile_size,
                    depth_slices: config.depth_slices,
                    near_plane: config.near_plane,
                    far_plane: config.far_plane,
                    ..Default::default()
                },
                width,
                height,
            ))
        } else {
            None
        };

        Self {
            config,
            thin_gbuffer,
            light_prepass,
            tiled_pipeline,
            clustered_pipeline,
            stencil_volumes: StencilLightVolumeManager::new(),
            debug_visualizer: ClusterDebugVisualizer::default(),
            frame_index: 0,
            resolution: UVec2::new(width, height),
        }
    }

    /// Resize all internal buffers.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.resolution = UVec2::new(width, height);
        if let Some(ref mut gb) = self.thin_gbuffer {
            gb.resize(width, height);
        }
        if let Some(ref mut lp) = self.light_prepass {
            lp.resize(width, height);
        }
        if let Some(ref mut tp) = self.tiled_pipeline {
            tp.resize(width, height);
        }
        if let Some(ref mut cp) = self.clustered_pipeline {
            cp.resize(width, height);
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.frame_index += 1;
        self.stencil_volumes.clear();
        if let Some(ref mut tp) = self.tiled_pipeline {
            tp.begin_frame();
        }
        if let Some(ref mut cp) = self.clustered_pipeline {
            cp.begin_frame();
        }
        if let Some(ref mut lp) = self.light_prepass {
            lp.begin_frame();
        }
    }

    /// Add a point light.
    pub fn add_point_light(&mut self, position: Vec3, radius: f32, color: Vec3, intensity: f32) {
        let idx = self.stencil_volumes.volumes.len() as u32;
        self.stencil_volumes
            .add_volume(StencilLightVolume::point_light(idx, position, radius, color, intensity));

        let tiled_light = TiledLight::point(position, radius, color, intensity);
        if let Some(ref mut tp) = self.tiled_pipeline {
            tp.add_light(tiled_light);
        }
        if let Some(ref mut cp) = self.clustered_pipeline {
            cp.add_light(tiled_light);
        }
    }

    /// Add a spot light.
    pub fn add_spot_light(
        &mut self,
        position: Vec3,
        direction: Vec3,
        range: f32,
        outer_angle: f32,
        inner_angle: f32,
        color: Vec3,
        intensity: f32,
    ) {
        let idx = self.stencil_volumes.volumes.len() as u32;
        self.stencil_volumes.add_volume(StencilLightVolume::spot_light(
            idx, position, direction, range, outer_angle, inner_angle, color, intensity,
        ));

        let tiled_light = TiledLight::spot(position, direction, range, outer_angle, inner_angle, color, intensity);
        if let Some(ref mut tp) = self.tiled_pipeline {
            tp.add_light(tiled_light);
        }
        if let Some(ref mut cp) = self.clustered_pipeline {
            cp.add_light(tiled_light);
        }
    }

    /// Add a directional light.
    pub fn add_directional_light(&mut self, direction: Vec3, color: Vec3, intensity: f32) {
        let idx = self.stencil_volumes.volumes.len() as u32;
        self.stencil_volumes
            .add_volume(StencilLightVolume::directional_light(idx, direction, color, intensity));

        let tiled_light = TiledLight::directional(direction, color, intensity);
        if let Some(ref mut tp) = self.tiled_pipeline {
            tp.add_light(tiled_light);
        }
        if let Some(ref mut cp) = self.clustered_pipeline {
            cp.add_light(tiled_light);
        }
    }

    /// Perform light culling for the current frame.
    pub fn cull_lights(&mut self, view: Mat4, projection: Mat4, camera_pos: Vec3) {
        self.stencil_volumes.update_camera_tests(camera_pos);
        self.stencil_volumes.sort_for_rendering(camera_pos);

        if let Some(ref mut tp) = self.tiled_pipeline {
            tp.cull_lights_cpu(view, projection);
        }
        if let Some(ref mut cp) = self.clustered_pipeline {
            let inv_proj = projection.inverse();
            cp.build_cluster_aabbs(inv_proj);
            cp.assign_lights_cpu(view);
        }
        if let Some(ref mut lp) = self.light_prepass {
            lp.update_matrices(view, projection);
        }
    }

    /// Get statistics summary.
    pub fn stats_summary(&self) -> DeferredV2Stats {
        DeferredV2Stats {
            frame_index: self.frame_index,
            total_lights: self.stencil_volumes.total_count() as u32,
            directional_lights: self.stencil_volumes.directional_count,
            point_lights: self.stencil_volumes.point_count,
            spot_lights: self.stencil_volumes.spot_count,
            area_lights: self.stencil_volumes.area_count,
            tiled_stats: self.tiled_pipeline.as_ref().map(|tp| tp.stats.clone()),
            clustered_stats: self.clustered_pipeline.as_ref().map(|cp| cp.stats.clone()),
        }
    }

    /// Toggle debug visualization.
    pub fn toggle_debug(&mut self) {
        self.debug_visualizer.cycle_mode();
    }
}

/// Summary statistics for the deferred V2 pipeline.
#[derive(Debug, Clone)]
pub struct DeferredV2Stats {
    pub frame_index: u64,
    pub total_lights: u32,
    pub directional_lights: u32,
    pub point_lights: u32,
    pub spot_lights: u32,
    pub area_lights: u32,
    pub tiled_stats: Option<TiledDeferredStats>,
    pub clustered_stats: Option<ClusteredShadingStats>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thin_gbuffer_layout_default() {
        let layout = ThinGBufferLayout::default();
        assert_eq!(layout.color_attachment_count(), 2);
        assert!(layout.estimated_memory_bytes() > 0);
    }

    #[test]
    fn test_packed_albedo_metallic_roundtrip() {
        let packed = PackedAlbedoMetallic::pack(Vec3::new(0.5, 0.3, 0.8), 0.7);
        let (albedo, metallic) = packed.unpack();
        assert!((albedo.x - 0.5).abs() < 0.01);
        assert!((albedo.y - 0.3).abs() < 0.01);
        assert!((albedo.z - 0.8).abs() < 0.01);
        assert!((metallic - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_packed_normal_roundtrip() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let packed = PackedNormalRoughness::pack(normal, 0.5, 0);
        let decoded = packed.unpack_normal();
        assert!((decoded - normal).length() < 0.02);
        assert!((packed.unpack_roughness() - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_stencil_volume_camera_inside() {
        let mut vol = StencilLightVolume::point_light(0, Vec3::ZERO, 10.0, Vec3::ONE, 1.0);
        vol.test_camera_inside(Vec3::new(1.0, 0.0, 0.0));
        assert!(vol.camera_inside);
        vol.test_camera_inside(Vec3::new(20.0, 0.0, 0.0));
        assert!(!vol.camera_inside);
    }

    #[test]
    fn test_tile_sphere_intersection() {
        let tile = Tile::new(0, 0, 0, 0, 16, 16);
        assert!(tile.intersects_sphere_screen(Vec2::new(8.0, 8.0), 5.0));
        assert!(!tile.intersects_sphere_screen(Vec2::new(100.0, 100.0), 5.0));
    }

    #[test]
    fn test_clustered_depth_slicing() {
        let config = ClusteredShadingConfig::default();
        let near_depth = config.slice_depth(0);
        let far_depth = config.slice_depth(config.depth_slices);
        assert!((near_depth - config.near_plane).abs() < 0.01);
        assert!((far_depth - config.far_plane).abs() < 1.0);
    }

    #[test]
    fn test_heatmap_color() {
        let c0 = DebugColor::heatmap(0.0);
        assert!(c0.b > 0.5); // Blue end
        let c1 = DebugColor::heatmap(1.0);
        assert!(c1.r > 0.5); // Red end
    }

    #[test]
    fn test_deferred_renderer_v2_lifecycle() {
        let config = DeferredV2Config::from_preset(DeferredQualityPreset::High);
        let mut renderer = DeferredRendererV2::new(config, 800, 600);
        renderer.begin_frame();
        renderer.add_point_light(Vec3::new(5.0, 3.0, 0.0), 10.0, Vec3::ONE, 100.0);
        renderer.add_directional_light(Vec3::NEG_Y, Vec3::ONE, 1.0);
        let stats = renderer.stats_summary();
        assert_eq!(stats.total_lights, 2);
        assert_eq!(stats.point_lights, 1);
        assert_eq!(stats.directional_lights, 1);
    }

    #[test]
    fn test_deferred_v2_config_presets() {
        let low = DeferredV2Config::from_preset(DeferredQualityPreset::Low);
        assert!(!low.use_clustered_shading);
        assert!(low.use_stencil_volumes);

        let ultra = DeferredV2Config::from_preset(DeferredQualityPreset::Ultra);
        assert!(ultra.use_clustered_shading);
        assert_eq!(ultra.tile_size, 8);
    }
}
