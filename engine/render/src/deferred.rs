// engine/render/src/deferred.rs
//
// Deferred rendering pipeline for the Genovo engine. Implements a full
// G-Buffer with multiple render targets, geometry pass, lighting pass with
// per-light-type volumes, composition pass, and a forward transparency pass.
//
// The deferred pipeline separates geometry rasterisation from lighting
// evaluation, enabling efficient rendering of many lights without
// re-drawing geometry.

use crate::interface::resource::TextureFormat;
use crate::lighting::light_types::{Light, LightType, MAX_LIGHTS};
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// GBuffer layout and configuration
// ---------------------------------------------------------------------------

/// Describes the format and binding of each G-Buffer render target.
#[derive(Debug, Clone)]
pub struct GBufferLayout {
    /// Albedo (base colour): RGBA8 or RGBA16Float.
    pub albedo_format: TextureFormat,
    /// World-space or view-space normals: RGB10A2 or RGBA16Float.
    pub normal_format: TextureFormat,
    /// Metallic (R), Roughness (G), AO (B), material ID (A).
    pub metallic_roughness_format: TextureFormat,
    /// Depth buffer format.
    pub depth_format: TextureFormat,
    /// Emissive: RGBA16Float.
    pub emissive_format: TextureFormat,
    /// Velocity / motion vectors: RG16Float.
    pub velocity_format: TextureFormat,
}

impl Default for GBufferLayout {
    fn default() -> Self {
        Self {
            albedo_format: TextureFormat::Rgba8Unorm,
            normal_format: TextureFormat::Rgba16Float,
            metallic_roughness_format: TextureFormat::Rgba8Unorm,
            depth_format: TextureFormat::Depth32Float,
            emissive_format: TextureFormat::Rgba16Float,
            velocity_format: TextureFormat::Rg16Float,
        }
    }
}

impl GBufferLayout {
    /// High-precision layout using 16-bit float for most targets.
    pub fn high_precision() -> Self {
        Self {
            albedo_format: TextureFormat::Rgba16Float,
            normal_format: TextureFormat::Rgba16Float,
            metallic_roughness_format: TextureFormat::Rgba16Float,
            depth_format: TextureFormat::Depth32Float,
            emissive_format: TextureFormat::Rgba16Float,
            velocity_format: TextureFormat::Rg16Float,
        }
    }

    /// Compact layout for lower memory bandwidth usage.
    pub fn compact() -> Self {
        Self {
            albedo_format: TextureFormat::Rgba8Unorm,
            normal_format: TextureFormat::Rgb10A2Unorm,
            metallic_roughness_format: TextureFormat::Rgba8Unorm,
            depth_format: TextureFormat::Depth32Float,
            emissive_format: TextureFormat::Rg11B10Float,
            velocity_format: TextureFormat::Rg16Float,
        }
    }

    /// Return all colour attachment formats in order.
    pub fn color_formats(&self) -> Vec<TextureFormat> {
        vec![
            self.albedo_format,
            self.normal_format,
            self.metallic_roughness_format,
            self.emissive_format,
            self.velocity_format,
        ]
    }
}

// ---------------------------------------------------------------------------
// GBuffer
// ---------------------------------------------------------------------------

/// Holds the G-Buffer render target handles and dimensions.
#[derive(Debug, Clone)]
pub struct GBuffer {
    /// Albedo / base colour render target.
    pub albedo_rt: u64,
    /// Normal render target (world-space normals, encoded).
    pub normal_rt: u64,
    /// Metallic-roughness-AO render target.
    pub metallic_roughness_rt: u64,
    /// Depth render target (hardware depth buffer).
    pub depth_rt: u64,
    /// Emissive render target.
    pub emissive_rt: u64,
    /// Velocity / motion vector render target.
    pub velocity_rt: u64,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Layout describing formats.
    pub layout: GBufferLayout,
}

impl GBuffer {
    /// Create a new G-Buffer with the given dimensions and layout.
    pub fn new(width: u32, height: u32, layout: GBufferLayout) -> Self {
        Self {
            albedo_rt: 0,
            normal_rt: 0,
            metallic_roughness_rt: 0,
            depth_rt: 0,
            emissive_rt: 0,
            velocity_rt: 0,
            width,
            height,
            layout,
        }
    }

    /// Create with default layout.
    pub fn with_default_layout(width: u32, height: u32) -> Self {
        Self::new(width, height, GBufferLayout::default())
    }

    /// Resize the G-Buffer (invalidates all handles).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        // Handles would need to be recreated by the renderer.
        self.albedo_rt = 0;
        self.normal_rt = 0;
        self.metallic_roughness_rt = 0;
        self.depth_rt = 0;
        self.emissive_rt = 0;
        self.velocity_rt = 0;
    }

    /// Check whether all render targets have valid handles.
    pub fn is_valid(&self) -> bool {
        self.albedo_rt != 0
            && self.normal_rt != 0
            && self.metallic_roughness_rt != 0
            && self.depth_rt != 0
            && self.emissive_rt != 0
            && self.velocity_rt != 0
    }

    /// Total number of colour attachments (excludes depth).
    pub fn color_attachment_count(&self) -> usize {
        5 // albedo, normal, metallic_roughness, emissive, velocity
    }

    /// Compute the total memory footprint estimate in bytes.
    pub fn estimated_memory_bytes(&self) -> u64 {
        let pixels = self.width as u64 * self.height as u64;
        let albedo_bpp = format_bytes_per_pixel(self.layout.albedo_format);
        let normal_bpp = format_bytes_per_pixel(self.layout.normal_format);
        let mr_bpp = format_bytes_per_pixel(self.layout.metallic_roughness_format);
        let depth_bpp = 4u64; // Depth32Float = 4 bytes
        let emissive_bpp = format_bytes_per_pixel(self.layout.emissive_format);
        let velocity_bpp = format_bytes_per_pixel(self.layout.velocity_format);

        pixels * (albedo_bpp + normal_bpp + mr_bpp + depth_bpp + emissive_bpp + velocity_bpp)
    }
}

/// Approximate bytes per pixel for a texture format.
fn format_bytes_per_pixel(format: TextureFormat) -> u64 {
    match format {
        TextureFormat::R8Unorm | TextureFormat::R8Snorm | TextureFormat::R8Uint | TextureFormat::R8Sint => 1,
        TextureFormat::Rg8Unorm | TextureFormat::Rg8Snorm | TextureFormat::R16Float | TextureFormat::R16Uint | TextureFormat::R16Sint => 2,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Snorm
        | TextureFormat::Rgba8Uint | TextureFormat::Rgba8Sint
        | TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::Rg16Float | TextureFormat::Rgb10A2Unorm | TextureFormat::Rg11B10Float
        | TextureFormat::R32Float | TextureFormat::R32Uint | TextureFormat::R32Sint
        | TextureFormat::Rg16Uint | TextureFormat::Rg16Sint
        | TextureFormat::Depth32Float | TextureFormat::Depth24PlusStencil8 => 4,
        TextureFormat::Rgba16Float | TextureFormat::Rgba16Uint | TextureFormat::Rgba16Sint
        | TextureFormat::Rg32Float | TextureFormat::Rg32Uint | TextureFormat::Rg32Sint => 8,
        TextureFormat::Rgba32Float | TextureFormat::Rgba32Uint | TextureFormat::Rgba32Sint => 16,
        _ => 4, // default estimate
    }
}

// ---------------------------------------------------------------------------
// Normal encoding/decoding
// ---------------------------------------------------------------------------

/// Encode a world-space normal to octahedron encoding (2 floats).
///
/// Octahedron encoding maps unit sphere normals to a 2D [-1,1]^2 square,
/// which can be stored in RG16Float with minimal loss.
pub fn encode_normal_octahedron(normal: Vec3) -> Vec2 {
    let n = normal / (normal.x.abs() + normal.y.abs() + normal.z.abs());
    if n.z >= 0.0 {
        Vec2::new(n.x, n.y)
    } else {
        Vec2::new(
            (1.0 - n.y.abs()) * if n.x >= 0.0 { 1.0 } else { -1.0 },
            (1.0 - n.x.abs()) * if n.y >= 0.0 { 1.0 } else { -1.0 },
        )
    }
}

/// Decode an octahedron-encoded normal back to a unit vector.
pub fn decode_normal_octahedron(encoded: Vec2) -> Vec3 {
    let mut n = Vec3::new(encoded.x, encoded.y, 1.0 - encoded.x.abs() - encoded.y.abs());
    if n.z < 0.0 {
        let old_x = n.x;
        let old_y = n.y;
        n.x = (1.0 - old_y.abs()) * if old_x >= 0.0 { 1.0 } else { -1.0 };
        n.y = (1.0 - old_x.abs()) * if old_y >= 0.0 { 1.0 } else { -1.0 };
    }
    n.normalize_or_zero()
}

/// Encode a normal to spheremap (Lambert azimuthal equal-area) encoding.
pub fn encode_normal_spheremap(normal: Vec3) -> Vec2 {
    let f = (8.0 * normal.z + 8.0).sqrt();
    Vec2::new(normal.x / f + 0.5, normal.y / f + 0.5)
}

/// Decode a spheremap-encoded normal.
pub fn decode_normal_spheremap(encoded: Vec2) -> Vec3 {
    let fn_val = Vec2::new(encoded.x * 4.0 - 2.0, encoded.y * 4.0 - 2.0);
    let f = fn_val.dot(fn_val);
    let g = (1.0 - f / 4.0).sqrt();
    Vec3::new(fn_val.x * g, fn_val.y * g, 1.0 - f / 2.0).normalize_or_zero()
}

// ---------------------------------------------------------------------------
// Light volume meshes
// ---------------------------------------------------------------------------

/// A simple vertex for light volume geometry (position only).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightVolumeVertex {
    pub position: [f32; 3],
}

/// Generate a unit sphere mesh for point light volumes.
///
/// Returns (vertices, indices) for a UV sphere with the given tessellation.
pub fn generate_sphere_volume(segments: u32, rings: u32) -> (Vec<LightVolumeVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Generate vertices.
    for ring in 0..=rings {
        let theta = PI * ring as f32 / rings as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for seg in 0..=segments {
            let phi = 2.0 * PI * seg as f32 / segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            vertices.push(LightVolumeVertex {
                position: [
                    sin_theta * cos_phi,
                    cos_theta,
                    sin_theta * sin_phi,
                ],
            });
        }
    }

    // Generate indices.
    for ring in 0..rings {
        for seg in 0..segments {
            let current = ring * (segments + 1) + seg;
            let next = current + segments + 1;

            indices.push(current);
            indices.push(next);
            indices.push(current + 1);

            indices.push(current + 1);
            indices.push(next);
            indices.push(next + 1);
        }
    }

    (vertices, indices)
}

/// Generate a cone mesh for spot light volumes.
///
/// The cone apex is at the origin, pointing along +Z with the given
/// half-angle and height.
pub fn generate_cone_volume(segments: u32, half_angle: f32, height: f32) -> (Vec<LightVolumeVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let base_radius = height * half_angle.tan();

    // Apex vertex.
    vertices.push(LightVolumeVertex { position: [0.0, 0.0, 0.0] });

    // Base circle vertices.
    for seg in 0..=segments {
        let phi = 2.0 * PI * seg as f32 / segments as f32;
        vertices.push(LightVolumeVertex {
            position: [
                base_radius * phi.cos(),
                base_radius * phi.sin(),
                height,
            ],
        });
    }

    // Base center vertex.
    let center_idx = vertices.len() as u32;
    vertices.push(LightVolumeVertex { position: [0.0, 0.0, height] });

    // Side triangles (apex to base edge).
    for seg in 0..segments {
        indices.push(0); // apex
        indices.push(1 + seg);
        indices.push(1 + seg + 1);
    }

    // Base cap triangles.
    for seg in 0..segments {
        indices.push(center_idx);
        indices.push(1 + seg + 1);
        indices.push(1 + seg);
    }

    (vertices, indices)
}

/// Generate a fullscreen triangle (used for directional light pass and
/// composition). The triangle covers the entire viewport when rendered
/// without a projection matrix.
pub fn generate_fullscreen_triangle() -> (Vec<LightVolumeVertex>, Vec<u32>) {
    let vertices = vec![
        LightVolumeVertex { position: [-1.0, -1.0, 0.0] },
        LightVolumeVertex { position: [3.0, -1.0, 0.0] },
        LightVolumeVertex { position: [-1.0, 3.0, 0.0] },
    ];
    let indices = vec![0, 1, 2];
    (vertices, indices)
}

// ---------------------------------------------------------------------------
// GPU uniform structures
// ---------------------------------------------------------------------------

/// Per-frame camera data for the deferred pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeferredCameraUniforms {
    /// View matrix.
    pub view: [[f32; 4]; 4],
    /// Projection matrix.
    pub projection: [[f32; 4]; 4],
    /// Inverse view matrix (for reconstructing world position from depth).
    pub inv_view: [[f32; 4]; 4],
    /// Inverse projection matrix.
    pub inv_projection: [[f32; 4]; 4],
    /// View-projection matrix.
    pub view_projection: [[f32; 4]; 4],
    /// Previous frame view-projection (for velocity computation).
    pub prev_view_projection: [[f32; 4]; 4],
    /// Camera position in world space (xyz) + near plane (w).
    pub camera_pos_near: [f32; 4],
    /// Viewport size (xy) + far plane (z) + padding (w).
    pub viewport_far: [f32; 4],
}

impl DeferredCameraUniforms {
    /// Build from matrices and camera parameters.
    pub fn from_matrices(
        view: Mat4,
        projection: Mat4,
        prev_vp: Mat4,
        camera_pos: Vec3,
        near: f32,
        far: f32,
        width: u32,
        height: u32,
    ) -> Self {
        let vp = projection * view;
        let inv_view = view.inverse();
        let inv_proj = projection.inverse();

        Self {
            view: view.to_cols_array_2d(),
            projection: projection.to_cols_array_2d(),
            inv_view: inv_view.to_cols_array_2d(),
            inv_projection: inv_proj.to_cols_array_2d(),
            view_projection: vp.to_cols_array_2d(),
            prev_view_projection: prev_vp.to_cols_array_2d(),
            camera_pos_near: [camera_pos.x, camera_pos.y, camera_pos.z, near],
            viewport_far: [width as f32, height as f32, far, 0.0],
        }
    }
}

/// Per-object data for the geometry pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GeometryPassUniforms {
    /// Model (world) matrix.
    pub model: [[f32; 4]; 4],
    /// Normal matrix (transpose of inverse of upper-left 3x3 of model).
    pub normal_matrix: [[f32; 4]; 4],
    /// Previous frame model matrix (for velocity computation).
    pub prev_model: [[f32; 4]; 4],
    /// Material parameters: x=metallic, y=roughness, z=ao, w=emissive_intensity.
    pub material_params: [f32; 4],
    /// Base colour (albedo) tint.
    pub base_color: [f32; 4],
    /// Emissive colour (RGB) + padding.
    pub emissive_color: [f32; 4],
}

impl GeometryPassUniforms {
    /// Build from model matrix and material properties.
    pub fn from_model(
        model: Mat4,
        prev_model: Mat4,
        base_color: Vec4,
        metallic: f32,
        roughness: f32,
        ao: f32,
        emissive_color: Vec3,
        emissive_intensity: f32,
    ) -> Self {
        let normal_matrix = model.inverse().transpose();

        Self {
            model: model.to_cols_array_2d(),
            normal_matrix: normal_matrix.to_cols_array_2d(),
            prev_model: prev_model.to_cols_array_2d(),
            material_params: [metallic, roughness, ao, emissive_intensity],
            base_color: base_color.to_array(),
            emissive_color: [emissive_color.x, emissive_color.y, emissive_color.z, 0.0],
        }
    }
}

/// Per-light data for the lighting pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightPassUniforms {
    /// Light world matrix (for light volume positioning/scaling).
    pub light_world: [[f32; 4]; 4],
    /// Light position (xyz) + type (w).
    pub position_type: [f32; 4],
    /// Light colour (rgb) + intensity (a).
    pub color_intensity: [f32; 4],
    /// Light direction (xyz) + range (w).
    pub direction_range: [f32; 4],
    /// Spot parameters: inner_cos (x), outer_cos (y), shadow_index (z), padding (w).
    pub spot_params: [f32; 4],
}

impl LightPassUniforms {
    /// Build from a Light and a world matrix for the light volume.
    pub fn from_light(light: &Light, light_world: Mat4) -> Self {
        let gpu_data = light.to_gpu_data();
        Self {
            light_world: light_world.to_cols_array_2d(),
            position_type: gpu_data.position_type.to_array(),
            color_intensity: gpu_data.color_intensity.to_array(),
            direction_range: gpu_data.direction_range.to_array(),
            spot_params: gpu_data.spot_params.to_array(),
        }
    }

    /// Build a directional light uniform (fullscreen pass, identity world).
    pub fn from_directional(light: &Light) -> Self {
        Self::from_light(light, Mat4::IDENTITY)
    }

    /// Build for a point light with a sphere volume.
    pub fn from_point_light(light: &Light, position: Vec3, radius: f32) -> Self {
        let world = Mat4::from_scale_rotation_translation(
            Vec3::splat(radius),
            glam::Quat::IDENTITY,
            position,
        );
        Self::from_light(light, world)
    }

    /// Build for a spot light with a cone volume.
    pub fn from_spot_light(
        light: &Light,
        position: Vec3,
        direction: Vec3,
        range: f32,
        outer_angle: f32,
    ) -> Self {
        // Build a rotation that orients +Z towards the spot direction.
        let up = if direction.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
        let right = direction.cross(up).normalize_or_zero();
        let up = right.cross(direction).normalize_or_zero();
        let rotation = Mat4::from_cols(
            right.extend(0.0),
            up.extend(0.0),
            direction.extend(0.0),
            Vec4::W,
        );

        let scale_factor = range * outer_angle.tan();
        let scale = Mat4::from_scale(Vec3::new(scale_factor, scale_factor, range));
        let translation = Mat4::from_translation(position);
        let world = translation * rotation * scale;

        Self::from_light(light, world)
    }
}

/// Composition pass uniforms.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompositionUniforms {
    /// Ambient colour (rgb) + AO strength (a).
    pub ambient_ao: [f32; 4],
    /// Exposure (x), gamma (y), padding (zw).
    pub exposure_gamma: [f32; 4],
}

// ---------------------------------------------------------------------------
// Stencil optimisation helpers
// ---------------------------------------------------------------------------

/// Stencil reference values for the deferred pipeline.
pub mod stencil {
    /// Stencil value written during the geometry pass to mark pixels with
    /// geometry (non-sky pixels).
    pub const GEOMETRY_BIT: u32 = 0x01;

    /// Stencil value for lit pixels (used by light volume stencil pass).
    pub const LIT_BIT: u32 = 0x02;

    /// Stencil function: equal to reference.
    pub const FUNC_EQUAL: u32 = 0;
    /// Stencil function: not equal to reference.
    pub const FUNC_NOT_EQUAL: u32 = 1;
    /// Stencil function: always pass.
    pub const FUNC_ALWAYS: u32 = 2;

    /// Stencil op: keep.
    pub const OP_KEEP: u32 = 0;
    /// Stencil op: replace with reference.
    pub const OP_REPLACE: u32 = 1;
    /// Stencil op: increment and clamp.
    pub const OP_INCR: u32 = 2;
    /// Stencil op: decrement and clamp.
    pub const OP_DECR: u32 = 3;
    /// Stencil op: invert.
    pub const OP_INVERT: u32 = 4;
}

// ---------------------------------------------------------------------------
// DeferredRenderer
// ---------------------------------------------------------------------------

/// Render pass descriptor for the deferred pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferredPass {
    /// Geometry pass: render opaque meshes into G-Buffer.
    Geometry,
    /// Lighting pass: evaluate per-light contribution.
    Lighting,
    /// Composition pass: combine lighting + ambient + emissive + AO.
    Composition,
    /// Forward pass: render transparent objects on top.
    Forward,
}

/// Configuration for the deferred renderer.
#[derive(Debug, Clone)]
pub struct DeferredRendererConfig {
    /// G-Buffer layout to use.
    pub gbuffer_layout: GBufferLayout,
    /// Whether to use stencil optimisation for light volumes.
    pub use_stencil_optimisation: bool,
    /// Maximum number of lights to process per frame.
    pub max_lights: usize,
    /// Whether to enable emissive in the composition pass.
    pub enable_emissive: bool,
    /// Ambient colour (linear RGB).
    pub ambient_color: Vec3,
    /// Ambient intensity.
    pub ambient_intensity: f32,
    /// Whether to render a forward pass for transparent objects.
    pub enable_forward_pass: bool,
}

impl Default for DeferredRendererConfig {
    fn default() -> Self {
        Self {
            gbuffer_layout: GBufferLayout::default(),
            use_stencil_optimisation: true,
            max_lights: MAX_LIGHTS,
            enable_emissive: true,
            ambient_color: Vec3::new(0.03, 0.03, 0.04),
            ambient_intensity: 1.0,
            enable_forward_pass: true,
        }
    }
}

/// The deferred renderer orchestrates the multi-pass deferred pipeline.
pub struct DeferredRenderer {
    /// Configuration.
    pub config: DeferredRendererConfig,
    /// G-Buffer.
    pub gbuffer: GBuffer,
    /// Light accumulation buffer handle.
    pub light_accumulation_rt: u64,
    /// Sphere mesh for point light volumes.
    sphere_vertices: Vec<LightVolumeVertex>,
    sphere_indices: Vec<u32>,
    /// Cone mesh for spot light volumes.
    cone_vertices: Vec<LightVolumeVertex>,
    cone_indices: Vec<u32>,
    /// Fullscreen triangle for directional lights and composition.
    fullscreen_vertices: Vec<LightVolumeVertex>,
    fullscreen_indices: Vec<u32>,
    /// Current width.
    width: u32,
    /// Current height.
    height: u32,
    /// Frame counter.
    frame_index: u64,
    /// Previous frame view-projection matrix (for motion vectors).
    prev_view_projection: Mat4,
}

impl DeferredRenderer {
    /// Create a new deferred renderer.
    pub fn new(width: u32, height: u32, config: DeferredRendererConfig) -> Self {
        let gbuffer = GBuffer::new(width, height, config.gbuffer_layout.clone());

        let (sphere_v, sphere_i) = generate_sphere_volume(16, 12);
        let (cone_v, cone_i) = generate_cone_volume(16, std::f32::consts::FRAC_PI_4, 1.0);
        let (fs_v, fs_i) = generate_fullscreen_triangle();

        Self {
            config,
            gbuffer,
            light_accumulation_rt: 0,
            sphere_vertices: sphere_v,
            sphere_indices: sphere_i,
            cone_vertices: cone_v,
            cone_indices: cone_i,
            fullscreen_vertices: fs_v,
            fullscreen_indices: fs_i,
            width,
            height,
            frame_index: 0,
            prev_view_projection: Mat4::IDENTITY,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(width: u32, height: u32) -> Self {
        Self::new(width, height, DeferredRendererConfig::default())
    }

    /// Resize render targets.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.gbuffer.resize(width, height);
        self.light_accumulation_rt = 0;
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// End the frame, saving the current VP for next frame's motion vectors.
    pub fn end_frame(&mut self, current_view_projection: Mat4) {
        self.prev_view_projection = current_view_projection;
    }

    /// Get the previous frame's view-projection matrix.
    pub fn prev_view_projection(&self) -> Mat4 {
        self.prev_view_projection
    }

    /// Current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Width.
    pub fn width(&self) -> u32 { self.width }
    /// Height.
    pub fn height(&self) -> u32 { self.height }

    /// Prepare geometry pass uniforms for an object.
    pub fn prepare_geometry_uniforms(
        &self,
        model: Mat4,
        prev_model: Mat4,
        base_color: Vec4,
        metallic: f32,
        roughness: f32,
        ao: f32,
        emissive_color: Vec3,
        emissive_intensity: f32,
    ) -> GeometryPassUniforms {
        GeometryPassUniforms::from_model(
            model, prev_model, base_color,
            metallic, roughness, ao,
            emissive_color, emissive_intensity,
        )
    }

    /// Prepare camera uniforms for the current frame.
    pub fn prepare_camera_uniforms(
        &self,
        view: Mat4,
        projection: Mat4,
        camera_pos: Vec3,
        near: f32,
        far: f32,
    ) -> DeferredCameraUniforms {
        DeferredCameraUniforms::from_matrices(
            view, projection,
            self.prev_view_projection,
            camera_pos, near, far,
            self.width, self.height,
        )
    }

    /// Classify lights and prepare uniforms for the lighting pass.
    ///
    /// Returns a list of (DeferredPass type, LightPassUniforms) for each
    /// light, grouped by their rendering strategy.
    pub fn prepare_light_uniforms(&self, lights: &[Light]) -> Vec<(LightType, LightPassUniforms)> {
        let mut result = Vec::with_capacity(lights.len().min(self.config.max_lights));

        for light in lights.iter().take(self.config.max_lights) {
            let uniforms = match light {
                Light::Directional(_) => {
                    LightPassUniforms::from_directional(light)
                }
                Light::Point(p) => {
                    LightPassUniforms::from_point_light(light, p.position, p.radius)
                }
                Light::Spot(s) => {
                    LightPassUniforms::from_spot_light(
                        light, s.position, s.direction, s.range, s.outer_angle,
                    )
                }
                Light::Area(a) => {
                    // Area lights rendered as point lights with large radius.
                    LightPassUniforms::from_point_light(light, a.position, a.range)
                }
            };

            result.push((light.light_type(), uniforms));
        }

        result
    }

    /// Prepare composition uniforms.
    pub fn prepare_composition_uniforms(
        &self,
        exposure: f32,
        gamma: f32,
        ao_strength: f32,
    ) -> CompositionUniforms {
        let ambient = self.config.ambient_color * self.config.ambient_intensity;
        CompositionUniforms {
            ambient_ao: [ambient.x, ambient.y, ambient.z, ao_strength],
            exposure_gamma: [exposure, gamma, 0.0, 0.0],
        }
    }

    /// Get the sphere volume mesh data (for point lights).
    pub fn sphere_volume(&self) -> (&[LightVolumeVertex], &[u32]) {
        (&self.sphere_vertices, &self.sphere_indices)
    }

    /// Get the cone volume mesh data (for spot lights).
    pub fn cone_volume(&self) -> (&[LightVolumeVertex], &[u32]) {
        (&self.cone_vertices, &self.cone_indices)
    }

    /// Get the fullscreen triangle data.
    pub fn fullscreen_triangle(&self) -> (&[LightVolumeVertex], &[u32]) {
        (&self.fullscreen_vertices, &self.fullscreen_indices)
    }

    /// Estimated GPU memory usage of the G-Buffer.
    pub fn gbuffer_memory_bytes(&self) -> u64 {
        self.gbuffer.estimated_memory_bytes()
    }
}

// ---------------------------------------------------------------------------
// Depth reconstruction helpers
// ---------------------------------------------------------------------------

/// Reconstruct view-space position from depth buffer value and screen UV.
///
/// # Arguments
/// - `screen_uv` -- screen coordinates in [0,1]^2.
/// - `depth` -- hardware depth value [0,1].
/// - `inv_projection` -- inverse of the projection matrix.
pub fn reconstruct_view_position(screen_uv: Vec2, depth: f32, inv_projection: &Mat4) -> Vec3 {
    // Convert screen UV to clip space.
    let clip_x = screen_uv.x * 2.0 - 1.0;
    let clip_y = (1.0 - screen_uv.y) * 2.0 - 1.0; // flip Y for RH
    let clip_pos = Vec4::new(clip_x, clip_y, depth, 1.0);

    // Transform to view space.
    let view_pos = *inv_projection * clip_pos;
    Vec3::new(
        view_pos.x / view_pos.w,
        view_pos.y / view_pos.w,
        view_pos.z / view_pos.w,
    )
}

/// Reconstruct world-space position from depth buffer.
pub fn reconstruct_world_position(
    screen_uv: Vec2,
    depth: f32,
    inv_view_projection: &Mat4,
) -> Vec3 {
    let clip_x = screen_uv.x * 2.0 - 1.0;
    let clip_y = (1.0 - screen_uv.y) * 2.0 - 1.0;
    let clip_pos = Vec4::new(clip_x, clip_y, depth, 1.0);
    let world_pos = *inv_view_projection * clip_pos;
    Vec3::new(
        world_pos.x / world_pos.w,
        world_pos.y / world_pos.w,
        world_pos.z / world_pos.w,
    )
}

/// Linearise a hardware depth value (reverse-Z or standard).
///
/// For a perspective projection with near and far planes:
///   linear_depth = near * far / (far - depth * (far - near))
pub fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    near * far / (far - depth * (far - near))
}

/// Linearise depth for a reverse-Z projection.
pub fn linearize_depth_reverse_z(depth: f32, near: f32, far: f32) -> f32 {
    near * far / (near + depth * (far - near))
}

// ---------------------------------------------------------------------------
// WGSL shader source for the deferred pipeline
// ---------------------------------------------------------------------------

/// WGSL shader for the G-Buffer geometry pass.
pub const GBUFFER_WRITE_WGSL: &str = r#"
// ---- G-Buffer Geometry Pass Shader ----

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    prev_view_projection: mat4x4<f32>,
    camera_pos_near: vec4<f32>,
    viewport_far: vec4<f32>,
};

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    prev_model: mat4x4<f32>,
    material_params: vec4<f32>,  // metallic, roughness, ao, emissive_intensity
    base_color: vec4<f32>,
    emissive_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) prev_clip_position: vec4<f32>,
    @location(4) world_tangent: vec3<f32>,
    @location(5) tangent_sign: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_pos = object.model * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.view_projection * world_pos;

    // Normal in world space.
    out.world_normal = normalize((object.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);

    // Tangent.
    out.world_tangent = normalize((object.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
    out.tangent_sign = in.tangent.w;

    out.uv = in.uv;

    // Previous clip position for velocity.
    let prev_world_pos = object.prev_model * vec4<f32>(in.position, 1.0);
    out.prev_clip_position = camera.prev_view_projection * prev_world_pos;

    return out;
}

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) metallic_roughness: vec4<f32>,
    @location(3) emissive: vec4<f32>,
    @location(4) velocity: vec2<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    var out: GBufferOutput;

    // Albedo.
    out.albedo = object.base_color;

    // World-space normal (pack into [0,1] range).
    let n = normalize(in.world_normal);
    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);

    // Metallic-roughness-AO.
    out.metallic_roughness = vec4<f32>(
        object.material_params.x,  // metallic
        object.material_params.y,  // roughness
        object.material_params.z,  // AO
        1.0,  // material ID / flags
    );

    // Emissive.
    out.emissive = vec4<f32>(
        object.emissive_color.xyz * object.material_params.w,
        1.0,
    );

    // Velocity (screen-space motion vector).
    let current_ndc = in.clip_position.xy / in.clip_position.w;
    let prev_ndc = in.prev_clip_position.xy / in.prev_clip_position.w;
    out.velocity = (current_ndc - prev_ndc) * 0.5;

    return out;
}
"#;

/// WGSL shader for the deferred lighting resolve pass.
pub const LIGHTING_RESOLVE_WGSL: &str = r#"
// ---- Deferred Lighting Resolve Shader ----
// Reads the G-Buffer textures and evaluates PBR lighting for a single light.

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    prev_view_projection: mat4x4<f32>,
    camera_pos_near: vec4<f32>,
    viewport_far: vec4<f32>,
};

struct LightUniforms {
    light_world: mat4x4<f32>,
    position_type: vec4<f32>,
    color_intensity: vec4<f32>,
    direction_range: vec4<f32>,
    spot_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> light: LightUniforms;

@group(1) @binding(0) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_metallic_roughness: texture_2d<f32>;
@group(1) @binding(3) var gbuffer_depth: texture_depth_2d;
@group(1) @binding(4) var gbuffer_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_uv: vec2<f32>,
};

// Fullscreen triangle vertex shader.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    var out: VertexOutput;
    let pos = positions[vertex_index];
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.screen_uv = pos * 0.5 + 0.5;
    out.screen_uv.y = 1.0 - out.screen_uv.y;
    return out;
}

// Light volume vertex shader (for point/spot lights).
@vertex
fn vs_light_volume(@location(0) position: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = light.light_world * vec4<f32>(position, 1.0);
    out.position = camera.view_projection * world_pos;
    out.screen_uv = out.position.xy / out.position.w * 0.5 + 0.5;
    out.screen_uv.y = 1.0 - out.screen_uv.y;
    return out;
}

// Reconstruct world position from depth.
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip_x = uv.x * 2.0 - 1.0;
    let clip_y = (1.0 - uv.y) * 2.0 - 1.0;
    let clip_pos = vec4<f32>(clip_x, clip_y, depth, 1.0);
    let inv_vp = camera.inv_view * camera.inv_projection;
    let world_pos = inv_vp * clip_pos;
    return world_pos.xyz / world_pos.w;
}

// GGX Normal Distribution Function.
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (3.14159265 * denom * denom + 0.0001);
}

// Schlick-GGX geometry function.
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) *
           geometry_schlick_ggx(n_dot_l, roughness);
}

// Schlick Fresnel approximation.
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let t = pow(max(1.0 - cos_theta, 0.0), 5.0);
    return f0 + (vec3<f32>(1.0) - f0) * t;
}

// Distance attenuation (inverse-square with smooth windowing).
fn distance_attenuation(distance: f32, range: f32) -> f32 {
    let d2 = distance * distance;
    let r2 = range * range;
    let ratio = d2 / r2;
    let factor = max(1.0 - ratio * ratio, 0.0);
    return factor * factor / max(d2, 0.0001);
}

// Spot angle attenuation.
fn spot_attenuation(cos_angle: f32, inner_cos: f32, outer_cos: f32) -> f32 {
    let t = clamp((cos_angle - outer_cos) / max(inner_cos - outer_cos, 0.0001), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@fragment
fn fs_lighting(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.screen_uv;
    let tex_coord = vec2<i32>(vec2<f32>(textureDimensions(gbuffer_albedo)) * uv);

    // Sample G-Buffer.
    let albedo = textureLoad(gbuffer_albedo, tex_coord, 0).rgb;
    let normal_packed = textureLoad(gbuffer_normal, tex_coord, 0).rgb;
    let mr = textureLoad(gbuffer_metallic_roughness, tex_coord, 0);
    let depth = textureLoad(gbuffer_depth, tex_coord, 0);

    // Decode normal from [0,1] to [-1,1].
    let normal = normalize(normal_packed * 2.0 - 1.0);
    let metallic = mr.r;
    let roughness = max(mr.g, 0.04);
    let ao = mr.b;

    // Reconstruct world position.
    let world_pos = reconstruct_world_pos(uv, depth);

    // View direction.
    let camera_pos = camera.camera_pos_near.xyz;
    let v = normalize(camera_pos - world_pos);

    // F0 (reflectance at normal incidence).
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let light_type = u32(light.position_type.w);
    var radiance = vec3<f32>(0.0);

    // Compute light direction and attenuation based on type.
    var l = vec3<f32>(0.0);
    var attenuation = 1.0f;

    if light_type == 0u {
        // Directional light.
        l = normalize(light.position_type.xyz);
        attenuation = 1.0;
    } else if light_type == 1u {
        // Point light.
        let light_pos = light.position_type.xyz;
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = to_light / max(dist, 0.0001);
        let range = light.direction_range.w;
        attenuation = distance_attenuation(dist, range);
    } else if light_type == 2u {
        // Spot light.
        let light_pos = light.position_type.xyz;
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = to_light / max(dist, 0.0001);
        let range = light.direction_range.w;
        attenuation = distance_attenuation(dist, range);
        let spot_dir = normalize(light.direction_range.xyz);
        let cos_angle = dot(l, spot_dir);
        attenuation *= spot_attenuation(cos_angle, light.spot_params.x, light.spot_params.y);
    }

    let light_color = light.color_intensity.rgb * light.color_intensity.a;

    // Cook-Torrance BRDF.
    let h = normalize(v + l);
    let n_dot_l = max(dot(normal, l), 0.0);
    let n_dot_v = max(dot(normal, v), 0.001);
    let n_dot_h = max(dot(normal, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);

    let specular = d * g * f / max(4.0 * n_dot_v * n_dot_l, 0.001);

    let k_d = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = k_d * albedo / 3.14159265;

    radiance = (diffuse + specular) * light_color * n_dot_l * attenuation;

    return vec4<f32>(radiance, 1.0);
}
"#;

/// WGSL shader for the composition pass.
pub const COMPOSITION_WGSL: &str = r#"
// ---- Deferred Composition Pass Shader ----
// Combines lighting accumulation with ambient + emissive + AO.

struct CompositionUniforms {
    ambient_ao: vec4<f32>,     // rgb = ambient colour, a = AO strength
    exposure_gamma: vec4<f32>, // x = exposure, y = gamma
};

@group(0) @binding(0) var<uniform> comp: CompositionUniforms;
@group(0) @binding(1) var lighting_texture: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_metallic_roughness: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_emissive: texture_2d<f32>;
@group(0) @binding(5) var comp_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    var out: VertexOutput;
    let pos = positions[vertex_index];
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_coord = vec2<i32>(vec2<f32>(textureDimensions(lighting_texture)) * in.uv);

    let lighting = textureLoad(lighting_texture, tex_coord, 0).rgb;
    let albedo = textureLoad(gbuffer_albedo, tex_coord, 0).rgb;
    let mr = textureLoad(gbuffer_metallic_roughness, tex_coord, 0);
    let emissive = textureLoad(gbuffer_emissive, tex_coord, 0).rgb;

    let ao = mr.b;
    let ao_strength = comp.ambient_ao.a;
    let effective_ao = mix(1.0, ao, ao_strength);

    // Ambient contribution.
    let ambient = comp.ambient_ao.rgb * albedo * effective_ao;

    // Combine: direct lighting + ambient + emissive.
    var final_color = lighting + ambient + emissive;

    // Exposure tone mapping (simple Reinhard).
    let exposure = comp.exposure_gamma.x;
    final_color = final_color * exposure;
    final_color = final_color / (final_color + vec3<f32>(1.0));

    // Gamma correction.
    let gamma = comp.exposure_gamma.y;
    let inv_gamma = 1.0 / gamma;
    final_color = pow(max(final_color, vec3<f32>(0.0)), vec3<f32>(inv_gamma));

    return vec4<f32>(final_color, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbuffer_layout_default() {
        let layout = GBufferLayout::default();
        assert_eq!(layout.color_formats().len(), 5);
    }

    #[test]
    fn gbuffer_creation() {
        let gb = GBuffer::with_default_layout(1920, 1080);
        assert_eq!(gb.width, 1920);
        assert_eq!(gb.height, 1080);
        assert_eq!(gb.color_attachment_count(), 5);
    }

    #[test]
    fn gbuffer_resize() {
        let mut gb = GBuffer::with_default_layout(1920, 1080);
        gb.albedo_rt = 1;
        gb.resize(2560, 1440);
        assert_eq!(gb.width, 2560);
        assert_eq!(gb.albedo_rt, 0); // invalidated
    }

    #[test]
    fn gbuffer_memory_estimate() {
        let gb = GBuffer::with_default_layout(1920, 1080);
        let mem = gb.estimated_memory_bytes();
        // Should be > 0 and reasonable (roughly 1920*1080 * ~20 bytes)
        assert!(mem > 1_000_000);
        assert!(mem < 1_000_000_000);
    }

    #[test]
    fn normal_octahedron_roundtrip() {
        let normals = [
            Vec3::X, Vec3::Y, Vec3::Z,
            -Vec3::X, -Vec3::Y, -Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(-0.5, 0.3, 0.8).normalize(),
        ];
        for n in &normals {
            let encoded = encode_normal_octahedron(*n);
            let decoded = decode_normal_octahedron(encoded);
            let error = (*n - decoded).length();
            assert!(error < 0.01, "Octahedron encoding error too large: {error} for {n}");
        }
    }

    #[test]
    fn normal_spheremap_roundtrip() {
        let normal = Vec3::new(0.5, 0.5, 0.707).normalize();
        let encoded = encode_normal_spheremap(normal);
        let decoded = decode_normal_spheremap(encoded);
        let error = (normal - decoded).length();
        assert!(error < 0.05, "Spheremap encoding error too large: {error}");
    }

    #[test]
    fn sphere_volume_generation() {
        let (verts, indices) = generate_sphere_volume(8, 6);
        assert!(!verts.is_empty());
        assert!(!indices.is_empty());
        assert_eq!(indices.len() % 3, 0, "Index count should be a multiple of 3");
    }

    #[test]
    fn cone_volume_generation() {
        let (verts, indices) = generate_cone_volume(8, 0.5, 2.0);
        assert!(!verts.is_empty());
        assert!(!indices.is_empty());
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn fullscreen_triangle_generation() {
        let (verts, indices) = generate_fullscreen_triangle();
        assert_eq!(verts.len(), 3);
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn depth_linearisation() {
        let linear = linearize_depth(0.5, 0.1, 100.0);
        assert!(linear > 0.0 && linear < 100.0);
    }

    #[test]
    fn camera_uniforms_construction() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
        let uniforms = DeferredCameraUniforms::from_matrices(
            view, proj, Mat4::IDENTITY, Vec3::new(0.0, 5.0, 10.0), 0.1, 100.0, 1920, 1080,
        );
        assert_eq!(uniforms.viewport_far[0], 1920.0);
        assert_eq!(uniforms.camera_pos_near[3], 0.1);
    }

    #[test]
    fn geometry_uniforms_normal_matrix() {
        let model = Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0));
        let uniforms = GeometryPassUniforms::from_model(
            model, model, Vec4::ONE, 0.0, 0.5, 1.0, Vec3::ZERO, 0.0,
        );
        // Normal matrix should exist (non-zero).
        assert!(uniforms.normal_matrix[0][0] != 0.0 || uniforms.normal_matrix[1][1] != 0.0);
    }

    #[test]
    fn light_uniforms_directional() {
        let light = crate::lighting::light_types::DirectionalLight::sun().to_light();
        let uniforms = LightPassUniforms::from_directional(&light);
        assert_eq!(uniforms.position_type[3], 0.0); // directional type = 0
    }

    #[test]
    fn deferred_renderer_creation() {
        let renderer = DeferredRenderer::with_defaults(1920, 1080);
        assert_eq!(renderer.width(), 1920);
        assert_eq!(renderer.height(), 1080);
        assert!(!renderer.sphere_volume().0.is_empty());
        assert!(!renderer.cone_volume().0.is_empty());
    }

    #[test]
    fn deferred_renderer_resize() {
        let mut renderer = DeferredRenderer::with_defaults(1920, 1080);
        renderer.resize(2560, 1440);
        assert_eq!(renderer.width(), 2560);
        assert_eq!(renderer.gbuffer.width, 2560);
    }

    #[test]
    fn world_position_reconstruction() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let inv_vp = (proj * view).inverse();
        // Center of screen at some depth should reconstruct to a point on the view axis.
        let world = reconstruct_world_position(Vec2::new(0.5, 0.5), 0.5, &inv_vp);
        // The z should be somewhere between near and far along the view direction.
        assert!(world.z.is_finite());
    }

    #[test]
    fn composition_uniforms() {
        let renderer = DeferredRenderer::with_defaults(1920, 1080);
        let comp = renderer.prepare_composition_uniforms(1.0, 2.2, 0.5);
        assert_eq!(comp.exposure_gamma[0], 1.0);
        assert_eq!(comp.exposure_gamma[1], 2.2);
    }
}
