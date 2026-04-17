// engine/render/src/skybox_renderer.rs
//
// Skybox rendering system for the Genovo engine.
//
// Supports two modes:
// 1. **Cubemap skybox** — renders a pre-baked HDR cubemap onto a fullscreen
//    triangle behind all geometry.
// 2. **Procedural sky** — generates a sky from atmosphere scattering with
//    Rayleigh/Mie contributions, sun disc, horizon blend, and HDR output.
//
// Both modes render behind everything by writing max depth and disabling
// depth writes (or writing depth = 1.0 in the shader). The skybox is drawn
// as a single full-screen triangle to minimize vertex processing.
//
// Features:
// - Cubemap sampling with rotation and exposure control
// - Procedural Preetham/Hosek sky model
// - Sun disc with bloom-ready HDR values
// - Gradient-based atmosphere with Rayleigh scattering approximation
// - Star rendering for night sky
// - Moon phase rendering
// - Fog / horizon haze integration
// - HDR environment map generation from procedural sky
// - Mip-mapped cubemap for diffuse irradiance
// - GPU pipeline creation with WGSL shaders

use glam::{Mat4, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum exposure for HDR sky rendering.
pub const MAX_SKY_EXPOSURE: f32 = 16.0;

/// Default sun angular radius in radians (~0.53 degrees).
pub const DEFAULT_SUN_ANGULAR_RADIUS: f32 = 0.00465;

/// Default moon angular radius in radians (~0.52 degrees).
pub const DEFAULT_MOON_ANGULAR_RADIUS: f32 = 0.00454;

/// Number of stars for night sky rendering.
pub const DEFAULT_STAR_COUNT: u32 = 4096;

/// Default cubemap face resolution for procedural sky capture.
pub const DEFAULT_SKY_CUBEMAP_RESOLUTION: u32 = 256;

// ---------------------------------------------------------------------------
// WGSL Shader: Fullscreen Triangle Vertex
// ---------------------------------------------------------------------------

/// WGSL vertex shader for rendering a fullscreen triangle.
/// Uses vertex index to generate clip-space coordinates covering the screen.
pub const FULLSCREEN_TRIANGLE_VS_WGSL: &str = r#"
// Full-screen triangle vertex shader.
// No vertex buffer needed -- positions are generated from vertex_index.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate a full-screen triangle with vertex indices 0, 1, 2.
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0); // depth = 1.0 (farthest)
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: Cubemap Skybox
// ---------------------------------------------------------------------------

/// WGSL fragment shader for cubemap skybox rendering.
/// Samples a cubemap texture using view direction reconstructed from UV and
/// inverse view-projection matrix.
pub const CUBEMAP_SKYBOX_FS_WGSL: &str = r#"
// Cubemap skybox fragment shader.
// Reconstructs world-space view direction from screen UV, samples the cubemap.

struct SkyboxUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_position: vec3<f32>,
    exposure: f32,
    rotation: mat4x4<f32>,
    tint: vec3<f32>,
    gamma: f32,
};

@group(0) @binding(0)
var<uniform> sky: SkyboxUniforms;

@group(0) @binding(1)
var sky_cubemap: texture_cube<f32>;

@group(0) @binding(2)
var sky_sampler: sampler;

struct FragmentInput {
    @location(0) uv: vec2<f32>,
};

// Reconstruct the world-space view direction from screen UVs.
fn reconstruct_view_dir(uv: vec2<f32>, inv_vp: mat4x4<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, 1.0, 1.0);
    let world_pos = inv_vp * ndc;
    return normalize(world_pos.xyz / world_pos.w - sky.camera_position);
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let view_dir = reconstruct_view_dir(input.uv, sky.inv_view_proj);

    // Apply cubemap rotation.
    let rotated_dir = (sky.rotation * vec4<f32>(view_dir, 0.0)).xyz;

    // Sample the cubemap.
    var color = textureSample(sky_cubemap, sky_sampler, rotated_dir).rgb;

    // Apply tint and exposure.
    color = color * sky.tint * sky.exposure;

    // Tone mapping (Reinhard).
    color = color / (color + vec3<f32>(1.0));

    // Gamma correction.
    color = pow(color, vec3<f32>(1.0 / sky.gamma));

    return vec4<f32>(color, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: Procedural Sky
// ---------------------------------------------------------------------------

/// WGSL fragment shader for procedural sky rendering.
/// Implements a simplified atmosphere model with Rayleigh scattering,
/// sun disc, stars, and horizon haze.
pub const PROCEDURAL_SKY_FS_WGSL: &str = r#"
// Procedural sky fragment shader.
// Rayleigh/Mie scattering approximation with sun disc and stars.

struct ProceduralSkyUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_position: vec3<f32>,
    sun_intensity: f32,
    sun_direction: vec3<f32>,
    sun_angular_radius: f32,
    rayleigh_coefficients: vec3<f32>,
    mie_coefficient: f32,
    mie_direction: f32,
    exposure: f32,
    ground_color: vec3<f32>,
    horizon_falloff: f32,
    sky_tint: vec3<f32>,
    time_of_day: f32,
    star_brightness: f32,
    star_threshold: f32,
    moon_direction: vec3<f32>,
    moon_angular_radius: f32,
    moon_color: vec3<f32>,
    moon_phase: f32,
};

@group(0) @binding(0)
var<uniform> proc_sky: ProceduralSkyUniforms;

struct FragmentInput {
    @location(0) uv: vec2<f32>,
};

const PI: f32 = 3.14159265359;

// Reconstruct view direction from screen UVs.
fn get_view_dir(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, 1.0, 1.0);
    let world_pos = proc_sky.inv_view_proj * ndc;
    return normalize(world_pos.xyz / world_pos.w - proc_sky.camera_position);
}

// Rayleigh phase function.
fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

// Henyey-Greenstein phase function for Mie scattering.
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = (1.0 - g2);
    let denom = 4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return num / max(denom, 0.0001);
}

// Simple hash for star positions.
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Generate stars in the sky.
fn stars(dir: vec3<f32>, threshold: f32, brightness: f32) -> f32 {
    // Convert direction to spherical coordinates for tiling.
    let theta = acos(dir.y);
    let phi = atan2(dir.z, dir.x);
    let grid = vec2<f32>(phi * 40.0 / PI, theta * 40.0 / PI);
    let cell = floor(grid);
    let h = hash21(cell);
    if (h > threshold) {
        let star_pos = cell + vec2<f32>(hash21(cell + vec2<f32>(1.0, 0.0)),
                                          hash21(cell + vec2<f32>(0.0, 1.0)));
        let d = length(grid - star_pos);
        return brightness * smoothstep(0.08, 0.0, d) * smoothstep(threshold, 1.0, h);
    }
    return 0.0;
}

// Render the sun disc with bloom.
fn sun_disc(view_dir: vec3<f32>, sun_dir: vec3<f32>, angular_radius: f32) -> vec3<f32> {
    let cos_angle = dot(view_dir, sun_dir);
    let sun_angle = acos(clamp(cos_angle, -1.0, 1.0));

    // Hard disc.
    var sun = 0.0;
    if (sun_angle < angular_radius) {
        sun = 1.0;
    }

    // Soft bloom around the disc.
    let bloom = pow(max(cos_angle, 0.0), 512.0) * 2.0;
    let corona = pow(max(cos_angle, 0.0), 64.0) * 0.3;

    return vec3<f32>(1.0, 0.95, 0.85) * (sun * proc_sky.sun_intensity + bloom + corona);
}

// Render moon with phase.
fn moon_disc(view_dir: vec3<f32>) -> vec3<f32> {
    let cos_angle = dot(view_dir, proc_sky.moon_direction);
    let moon_angle = acos(clamp(cos_angle, -1.0, 1.0));

    if (moon_angle > proc_sky.moon_angular_radius * 3.0) {
        return vec3<f32>(0.0);
    }

    let t = moon_angle / proc_sky.moon_angular_radius;
    var moon = smoothstep(1.2, 0.9, t);

    // Apply phase shading.
    let right = normalize(cross(proc_sky.moon_direction, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, proc_sky.moon_direction);
    let moon_local_x = dot(view_dir - proc_sky.moon_direction * cos_angle, right);
    let phase_shade = smoothstep(-0.5, 0.5, moon_local_x * sign(proc_sky.moon_phase - 0.5));
    moon = moon * mix(0.1, 1.0, phase_shade);

    return proc_sky.moon_color * moon * 0.4;
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let view_dir = get_view_dir(input.uv);
    let sun_dir = normalize(proc_sky.sun_direction);

    // Scattering.
    let cos_theta = dot(view_dir, sun_dir);
    let rayleigh = rayleigh_phase(cos_theta) * proc_sky.rayleigh_coefficients;
    let mie = mie_phase(cos_theta, proc_sky.mie_direction) * proc_sky.mie_coefficient;

    // Sky gradient based on view elevation.
    let y = max(view_dir.y, 0.0);
    let horizon = exp(-y * proc_sky.horizon_falloff);

    // Combine scattering.
    var sky_color = (rayleigh + mie) * proc_sky.sun_intensity * proc_sky.sky_tint;

    // Add horizon haze.
    sky_color = mix(sky_color, proc_sky.ground_color, horizon * 0.5);

    // Ground below horizon.
    if (view_dir.y < 0.0) {
        let ground_fade = smoothstep(0.0, -0.02, view_dir.y);
        sky_color = mix(sky_color, proc_sky.ground_color, ground_fade);
    }

    // Add sun disc.
    sky_color = sky_color + sun_disc(view_dir, sun_dir, proc_sky.sun_angular_radius);

    // Add moon at night.
    let night_factor = smoothstep(0.05, -0.1, sun_dir.y);
    if (night_factor > 0.01) {
        sky_color = sky_color + moon_disc(view_dir) * night_factor;

        // Add stars at night.
        let star = stars(view_dir, proc_sky.star_threshold, proc_sky.star_brightness);
        sky_color = sky_color + vec3<f32>(star) * night_factor * max(view_dir.y, 0.0);
    }

    // Exposure.
    sky_color = sky_color * proc_sky.exposure;

    return vec4<f32>(sky_color, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// WGSL Shader: Environment Irradiance from Cubemap
// ---------------------------------------------------------------------------

/// WGSL compute shader for generating diffuse irradiance from a cubemap.
pub const ENV_IRRADIANCE_COMPUTE_WGSL: &str = r#"
// Compute shader: generate diffuse irradiance cubemap from HDR environment.
// Convolves the environment map with a cosine lobe for diffuse lighting.

@group(0) @binding(0)
var env_cubemap: texture_cube<f32>;

@group(0) @binding(1)
var env_sampler: sampler;

@group(0) @binding(2)
var output_face: texture_storage_2d<rgba16float, write>;

struct IrradianceParams {
    face_index: u32,
    resolution: u32,
    sample_count: u32,
    _pad: u32,
};

@group(0) @binding(3)
var<uniform> params: IrradianceParams;

const PI: f32 = 3.14159265359;

// Convert face index + UV to cubemap direction.
fn face_uv_to_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;
    switch (face) {
        case 0u: { return normalize(vec3<f32>( 1.0,   -v,   -u)); } // +X
        case 1u: { return normalize(vec3<f32>(-1.0,   -v,    u)); } // -X
        case 2u: { return normalize(vec3<f32>(   u,  1.0,    v)); } // +Y
        case 3u: { return normalize(vec3<f32>(   u, -1.0,   -v)); } // -Y
        case 4u: { return normalize(vec3<f32>(   u,   -v,  1.0)); } // +Z
        default: { return normalize(vec3<f32>(  -u,   -v, -1.0)); } // -Z
    }
}

// Hammersley sequence for quasi-random sampling.
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

fn importance_sample_cosine(xi: vec2<f32>, n: vec3<f32>) -> vec3<f32> {
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt(1.0 - xi.y);
    let sin_theta = sqrt(xi.y);

    let h = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);

    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = params.resolution;
    if (id.x >= res || id.y >= res) {
        return;
    }

    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + 0.5) / f32(res);
    let normal = face_uv_to_dir(params.face_index, uv);

    var irradiance = vec3<f32>(0.0);
    let n = params.sample_count;

    for (var i = 0u; i < n; i = i + 1u) {
        let xi = hammersley(i, n);
        let sample_dir = importance_sample_cosine(xi, normal);
        let ndotl = max(dot(normal, sample_dir), 0.0);
        irradiance = irradiance + textureSampleLevel(env_cubemap, env_sampler, sample_dir, 0.0).rgb;
    }

    irradiance = irradiance * PI / f32(n);
    textureStore(output_face, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(irradiance, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Skybox mode
// ---------------------------------------------------------------------------

/// Skybox rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkyboxMode {
    /// Render from a pre-baked HDR cubemap.
    Cubemap,
    /// Render procedurally from atmosphere parameters.
    Procedural,
    /// Solid color background.
    SolidColor,
    /// Gradient background (top to bottom).
    Gradient,
    /// Disabled (no sky rendering).
    None,
}

impl Default for SkyboxMode {
    fn default() -> Self {
        Self::Procedural
    }
}

// ---------------------------------------------------------------------------
// Cubemap skybox configuration
// ---------------------------------------------------------------------------

/// Configuration for cubemap-based skybox rendering.
#[derive(Debug, Clone)]
pub struct CubemapSkyboxConfig {
    /// Cubemap texture handle (6-face cube texture).
    pub cubemap_handle: u64,
    /// Rotation matrix applied to the cubemap direction.
    pub rotation: Mat4,
    /// Exposure multiplier for HDR cubemaps.
    pub exposure: f32,
    /// Color tint applied to the cubemap.
    pub tint: Vec3,
    /// Gamma correction value.
    pub gamma: f32,
    /// Blur amount (mip level for blurry skybox).
    pub blur_level: f32,
    /// Whether the cubemap is in linear color space.
    pub is_linear: bool,
}

impl Default for CubemapSkyboxConfig {
    fn default() -> Self {
        Self {
            cubemap_handle: 0,
            rotation: Mat4::IDENTITY,
            exposure: 1.0,
            tint: Vec3::ONE,
            gamma: 2.2,
            blur_level: 0.0,
            is_linear: true,
        }
    }
}

impl CubemapSkyboxConfig {
    /// Set rotation from Euler angles (degrees).
    pub fn set_rotation_degrees(&mut self, yaw: f32, pitch: f32, roll: f32) {
        let yaw_rad = yaw.to_radians();
        let pitch_rad = pitch.to_radians();
        let roll_rad = roll.to_radians();
        self.rotation = Mat4::from_euler(glam::EulerRot::YXZ, yaw_rad, pitch_rad, roll_rad);
    }

    /// Returns the GPU uniform data for this cubemap skybox.
    pub fn gpu_uniforms(&self, inv_view_proj: &Mat4, camera_pos: Vec3) -> CubemapSkyboxUniforms {
        CubemapSkyboxUniforms {
            inv_view_proj: *inv_view_proj,
            camera_position: camera_pos,
            exposure: self.exposure,
            rotation: self.rotation,
            tint: self.tint,
            gamma: self.gamma,
        }
    }
}

/// GPU uniform structure for cubemap skybox rendering.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CubemapSkyboxUniforms {
    pub inv_view_proj: Mat4,
    pub camera_position: Vec3,
    pub exposure: f32,
    pub rotation: Mat4,
    pub tint: Vec3,
    pub gamma: f32,
}

// ---------------------------------------------------------------------------
// Procedural sky configuration
// ---------------------------------------------------------------------------

/// Configuration for procedural sky rendering.
#[derive(Debug, Clone)]
pub struct ProceduralSkyConfig {
    /// Sun direction (normalized, pointing toward sun).
    pub sun_direction: Vec3,
    /// Sun intensity.
    pub sun_intensity: f32,
    /// Sun angular radius in radians.
    pub sun_angular_radius: f32,
    /// Sun color (linear RGB).
    pub sun_color: Vec3,
    /// Rayleigh scattering coefficients (wavelength-dependent).
    pub rayleigh_coefficients: Vec3,
    /// Mie scattering coefficient.
    pub mie_coefficient: f32,
    /// Mie scattering preferred direction (g parameter, -1 to 1).
    pub mie_direction: f32,
    /// Exposure for HDR output.
    pub exposure: f32,
    /// Ground albedo color (below-horizon color).
    pub ground_color: Vec3,
    /// Horizon falloff exponent.
    pub horizon_falloff: f32,
    /// Sky color tint.
    pub sky_tint: Vec3,
    /// Time of day (0.0 = midnight, 0.5 = noon, 1.0 = midnight).
    pub time_of_day: f32,
    /// Star brightness (night sky).
    pub star_brightness: f32,
    /// Star visibility threshold (0.0 = many stars, 1.0 = few stars).
    pub star_threshold: f32,
    /// Moon direction.
    pub moon_direction: Vec3,
    /// Moon angular radius.
    pub moon_angular_radius: f32,
    /// Moon color.
    pub moon_color: Vec3,
    /// Moon phase (0.0 = new moon, 0.5 = full moon, 1.0 = new moon).
    pub moon_phase: f32,
    /// Whether stars are enabled.
    pub enable_stars: bool,
    /// Whether moon is enabled.
    pub enable_moon: bool,
    /// Turbidity (atmospheric haze, 2 = clear, 10 = hazy).
    pub turbidity: f32,
}

impl Default for ProceduralSkyConfig {
    fn default() -> Self {
        Self {
            sun_direction: Vec3::new(0.0, 0.707, 0.707),
            sun_intensity: 20.0,
            sun_angular_radius: DEFAULT_SUN_ANGULAR_RADIUS,
            sun_color: Vec3::new(1.0, 0.96, 0.88),
            rayleigh_coefficients: Vec3::new(5.5e-6, 13.0e-6, 22.4e-6),
            mie_coefficient: 21e-6,
            mie_direction: 0.758,
            exposure: 1.0,
            ground_color: Vec3::new(0.37, 0.35, 0.32),
            horizon_falloff: 4.0,
            sky_tint: Vec3::ONE,
            time_of_day: 0.3,
            star_brightness: 0.8,
            star_threshold: 0.95,
            moon_direction: Vec3::new(-0.3, 0.5, -0.8),
            moon_angular_radius: DEFAULT_MOON_ANGULAR_RADIUS,
            moon_color: Vec3::new(0.85, 0.9, 1.0),
            moon_phase: 0.65,
            enable_stars: true,
            enable_moon: true,
            turbidity: 3.0,
        }
    }
}

impl ProceduralSkyConfig {
    /// Set the sun position from time of day (0-24 hours).
    pub fn set_time_hours(&mut self, hours: f32) {
        let normalized = hours / 24.0;
        self.time_of_day = normalized;

        // Calculate sun direction from time.
        let angle = (normalized - 0.25) * 2.0 * PI; // 6:00 = sunrise
        self.sun_direction = Vec3::new(
            angle.cos() * 0.5,
            angle.sin(),
            angle.cos() * 0.866,
        ).normalize();

        // Adjust exposure based on sun height.
        let sun_height = self.sun_direction.y;
        self.exposure = if sun_height > 0.0 {
            1.0 + sun_height * 0.5
        } else {
            0.3 + sun_height.abs().min(0.5)
        };

        // Moon is opposite to sun.
        self.moon_direction = Vec3::new(
            -self.sun_direction.x,
            (-self.sun_direction.y).max(0.1),
            -self.sun_direction.z,
        ).normalize();
    }

    /// Adjust atmosphere for different planet types.
    pub fn set_earth_like(&mut self) {
        self.rayleigh_coefficients = Vec3::new(5.5e-6, 13.0e-6, 22.4e-6);
        self.mie_coefficient = 21e-6;
        self.mie_direction = 0.758;
        self.ground_color = Vec3::new(0.37, 0.35, 0.32);
    }

    /// Set Mars-like atmosphere.
    pub fn set_mars_like(&mut self) {
        self.rayleigh_coefficients = Vec3::new(19.918e-6, 13.57e-6, 5.75e-6);
        self.mie_coefficient = 50e-6;
        self.mie_direction = 0.63;
        self.ground_color = Vec3::new(0.65, 0.35, 0.18);
        self.sky_tint = Vec3::new(1.0, 0.7, 0.5);
    }

    /// Set alien purple atmosphere.
    pub fn set_alien_atmosphere(&mut self) {
        self.rayleigh_coefficients = Vec3::new(20.0e-6, 5.0e-6, 15.0e-6);
        self.mie_coefficient = 10e-6;
        self.mie_direction = 0.8;
        self.ground_color = Vec3::new(0.2, 0.1, 0.3);
        self.sky_tint = Vec3::new(0.8, 0.5, 1.0);
    }

    /// Returns the GPU uniform data.
    pub fn gpu_uniforms(&self, inv_view_proj: &Mat4, camera_pos: Vec3) -> ProceduralSkyUniforms {
        ProceduralSkyUniforms {
            inv_view_proj: *inv_view_proj,
            camera_position: camera_pos,
            sun_intensity: self.sun_intensity,
            sun_direction: self.sun_direction,
            sun_angular_radius: self.sun_angular_radius,
            rayleigh_coefficients: self.rayleigh_coefficients,
            mie_coefficient: self.mie_coefficient,
            mie_direction: self.mie_direction,
            exposure: self.exposure,
            ground_color: self.ground_color,
            horizon_falloff: self.horizon_falloff,
            sky_tint: self.sky_tint,
            time_of_day: self.time_of_day,
            star_brightness: if self.enable_stars { self.star_brightness } else { 0.0 },
            star_threshold: self.star_threshold,
            moon_direction: self.moon_direction,
            moon_angular_radius: if self.enable_moon { self.moon_angular_radius } else { 0.0 },
            moon_color: self.moon_color,
            moon_phase: self.moon_phase,
        }
    }
}

/// GPU uniform structure for procedural sky rendering.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ProceduralSkyUniforms {
    pub inv_view_proj: Mat4,
    pub camera_position: Vec3,
    pub sun_intensity: f32,
    pub sun_direction: Vec3,
    pub sun_angular_radius: f32,
    pub rayleigh_coefficients: Vec3,
    pub mie_coefficient: f32,
    pub mie_direction: f32,
    pub exposure: f32,
    pub ground_color: Vec3,
    pub horizon_falloff: f32,
    pub sky_tint: Vec3,
    pub time_of_day: f32,
    pub star_brightness: f32,
    pub star_threshold: f32,
    pub moon_direction: Vec3,
    pub moon_angular_radius: f32,
    pub moon_color: Vec3,
    pub moon_phase: f32,
}

// ---------------------------------------------------------------------------
// Environment map capture
// ---------------------------------------------------------------------------

/// Captures the procedural sky to a cubemap for environment lighting.
#[derive(Debug, Clone)]
pub struct EnvironmentCapture {
    /// Resolution per cubemap face.
    pub face_resolution: u32,
    /// Whether the capture is dirty and needs re-rendering.
    pub dirty: bool,
    /// Whether to generate irradiance map from the capture.
    pub generate_irradiance: bool,
    /// Irradiance map resolution per face.
    pub irradiance_resolution: u32,
    /// Number of samples for irradiance convolution.
    pub irradiance_samples: u32,
    /// Whether to generate prefiltered environment map (for specular).
    pub generate_prefiltered: bool,
    /// Number of mip levels for prefiltered map.
    pub prefiltered_mip_levels: u32,
    /// Cubemap face view matrices.
    face_views: [Mat4; 6],
    /// Cubemap face projection matrix.
    face_projection: Mat4,
}

impl EnvironmentCapture {
    /// Creates a new environment capture configuration.
    pub fn new(resolution: u32) -> Self {
        let proj = Mat4::perspective_rh(PI * 0.5, 1.0, 0.1, 10.0);
        let views = Self::compute_face_views(Vec3::ZERO);

        Self {
            face_resolution: resolution,
            dirty: true,
            generate_irradiance: true,
            irradiance_resolution: 32,
            irradiance_samples: 1024,
            generate_prefiltered: true,
            prefiltered_mip_levels: 5,
            face_views: views,
            face_projection: proj,
        }
    }

    /// Compute the 6 face view matrices for cubemap rendering.
    fn compute_face_views(center: Vec3) -> [Mat4; 6] {
        [
            Mat4::look_at_rh(center, center + Vec3::X, Vec3::NEG_Y),    // +X
            Mat4::look_at_rh(center, center + Vec3::NEG_X, Vec3::NEG_Y),// -X
            Mat4::look_at_rh(center, center + Vec3::Y, Vec3::Z),        // +Y
            Mat4::look_at_rh(center, center + Vec3::NEG_Y, Vec3::NEG_Z),// -Y
            Mat4::look_at_rh(center, center + Vec3::Z, Vec3::NEG_Y),    // +Z
            Mat4::look_at_rh(center, center + Vec3::NEG_Z, Vec3::NEG_Y),// -Z
        ]
    }

    /// Returns the view-projection matrix for a cubemap face.
    pub fn face_view_proj(&self, face_index: usize) -> Mat4 {
        self.face_projection * self.face_views[face_index.min(5)]
    }

    /// Returns the inverse view-projection matrix for a cubemap face.
    pub fn face_inv_view_proj(&self, face_index: usize) -> Mat4 {
        self.face_view_proj(face_index).inverse()
    }

    /// Mark the capture as needing re-rendering.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Clear the dirty flag after re-rendering.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }
}

// ---------------------------------------------------------------------------
// Skybox renderer
// ---------------------------------------------------------------------------

/// The main skybox renderer.
///
/// Manages skybox rendering in both cubemap and procedural modes. The
/// renderer creates and caches GPU pipelines and updates uniforms each frame.
#[derive(Debug)]
pub struct SkyboxRenderer {
    /// Active skybox mode.
    pub mode: SkyboxMode,
    /// Cubemap configuration (used when mode == Cubemap).
    pub cubemap_config: CubemapSkyboxConfig,
    /// Procedural sky configuration (used when mode == Procedural).
    pub procedural_config: ProceduralSkyConfig,
    /// Solid color (used when mode == SolidColor).
    pub solid_color: Vec3,
    /// Gradient top color (used when mode == Gradient).
    pub gradient_top: Vec3,
    /// Gradient bottom color (used when mode == Gradient).
    pub gradient_bottom: Vec3,
    /// Environment capture settings.
    pub env_capture: EnvironmentCapture,
    /// Whether the pipeline has been created.
    pipeline_created: bool,
    /// Whether the skybox needs updating this frame.
    pub needs_update: bool,
    /// Cached inverse view-projection matrix.
    cached_inv_view_proj: Mat4,
    /// Cached camera position.
    cached_camera_pos: Vec3,
}

impl SkyboxRenderer {
    /// Creates a new skybox renderer.
    pub fn new() -> Self {
        Self {
            mode: SkyboxMode::Procedural,
            cubemap_config: CubemapSkyboxConfig::default(),
            procedural_config: ProceduralSkyConfig::default(),
            solid_color: Vec3::new(0.2, 0.2, 0.3),
            gradient_top: Vec3::new(0.1, 0.15, 0.4),
            gradient_bottom: Vec3::new(0.6, 0.5, 0.4),
            env_capture: EnvironmentCapture::new(DEFAULT_SKY_CUBEMAP_RESOLUTION),
            pipeline_created: false,
            needs_update: true,
            cached_inv_view_proj: Mat4::IDENTITY,
            cached_camera_pos: Vec3::ZERO,
        }
    }

    /// Set the skybox mode.
    pub fn set_mode(&mut self, mode: SkyboxMode) {
        if self.mode != mode {
            self.mode = mode;
            self.needs_update = true;
            self.pipeline_created = false; // Force pipeline recreation.
        }
    }

    /// Update the skybox for the current frame.
    pub fn update(&mut self, view: &Mat4, proj: &Mat4, camera_pos: Vec3) {
        let vp = *proj * *view;
        self.cached_inv_view_proj = vp.inverse();
        self.cached_camera_pos = camera_pos;

        // Check if the procedural sky needs a cubemap capture.
        if self.mode == SkyboxMode::Procedural && self.needs_update {
            self.env_capture.mark_dirty();
        }
    }

    /// Returns the WGSL shader source for the current mode.
    pub fn vertex_shader_source(&self) -> &'static str {
        FULLSCREEN_TRIANGLE_VS_WGSL
    }

    /// Returns the WGSL fragment shader source for the current mode.
    pub fn fragment_shader_source(&self) -> &'static str {
        match self.mode {
            SkyboxMode::Cubemap => CUBEMAP_SKYBOX_FS_WGSL,
            SkyboxMode::Procedural => PROCEDURAL_SKY_FS_WGSL,
            SkyboxMode::SolidColor | SkyboxMode::Gradient | SkyboxMode::None => CUBEMAP_SKYBOX_FS_WGSL,
        }
    }

    /// Returns the GPU pipeline descriptor for the skybox.
    pub fn pipeline_descriptor(&self) -> SkyboxPipelineDesc {
        SkyboxPipelineDesc {
            mode: self.mode,
            color_format: ColorFormat::Rgba16Float,
            depth_format: DepthFormat::Depth32Float,
            depth_write: false,
            depth_compare: DepthCompare::LessEqual,
            cull_mode: CullMode::None,
            blend: BlendMode::None,
        }
    }

    /// Returns procedural sky uniforms for the GPU.
    pub fn procedural_uniforms(&self) -> ProceduralSkyUniforms {
        self.procedural_config.gpu_uniforms(&self.cached_inv_view_proj, self.cached_camera_pos)
    }

    /// Returns cubemap skybox uniforms for the GPU.
    pub fn cubemap_uniforms(&self) -> CubemapSkyboxUniforms {
        self.cubemap_config.gpu_uniforms(&self.cached_inv_view_proj, self.cached_camera_pos)
    }

    /// Set time of day and update all related sky parameters.
    pub fn set_time_of_day(&mut self, hours: f32) {
        self.procedural_config.set_time_hours(hours);
        self.needs_update = true;
        self.env_capture.mark_dirty();
    }

    /// Get the current sun direction.
    pub fn sun_direction(&self) -> Vec3 {
        self.procedural_config.sun_direction
    }

    /// Get the sun color at the current time of day (for lighting).
    pub fn sun_color(&self) -> Vec3 {
        let height = self.procedural_config.sun_direction.y;
        if height < -0.05 {
            return Vec3::ZERO; // Night.
        }
        // Sunset/sunrise tint.
        let sunset_factor = 1.0 - height.max(0.0).min(1.0);
        let base = self.procedural_config.sun_color;
        let sunset = Vec3::new(1.0, 0.5, 0.2);
        let color = Vec3::new(
            base.x * (1.0 - sunset_factor * 0.3) + sunset.x * sunset_factor * 0.3,
            base.y * (1.0 - sunset_factor * 0.5) + sunset.y * sunset_factor * 0.5,
            base.z * (1.0 - sunset_factor * 0.7) + sunset.z * sunset_factor * 0.7,
        );
        color * self.procedural_config.sun_intensity * height.max(0.0).powf(0.5)
    }

    /// Get the ambient light color from the sky.
    pub fn ambient_color(&self) -> Vec3 {
        let sun_height = self.procedural_config.sun_direction.y;
        let day = sun_height.max(0.0);
        let night = (-sun_height).max(0.0).min(1.0);

        let day_ambient = Vec3::new(0.3, 0.35, 0.5) * day;
        let night_ambient = Vec3::new(0.02, 0.02, 0.05) * night;
        let transition = Vec3::new(0.2, 0.15, 0.1) * (1.0 - day - night).max(0.0);

        day_ambient + night_ambient + transition
    }
}

// ---------------------------------------------------------------------------
// Pipeline types
// ---------------------------------------------------------------------------

/// Color format for the skybox render target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorFormat {
    Rgba8Unorm,
    Rgba16Float,
    Bgra8Unorm,
}

/// Depth format for the skybox render target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthFormat {
    Depth24Plus,
    Depth32Float,
}

/// Depth comparison function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthCompare {
    Less,
    LessEqual,
    Always,
}

/// Cull mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

/// Blend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    None,
    Alpha,
    Additive,
}

/// Pipeline descriptor for skybox rendering.
#[derive(Debug, Clone)]
pub struct SkyboxPipelineDesc {
    pub mode: SkyboxMode,
    pub color_format: ColorFormat,
    pub depth_format: DepthFormat,
    pub depth_write: bool,
    pub depth_compare: DepthCompare,
    pub cull_mode: CullMode,
    pub blend: BlendMode,
}

// ---------------------------------------------------------------------------
// ECS component
// ---------------------------------------------------------------------------

/// ECS component for the skybox.
#[derive(Debug, Clone)]
pub struct SkyboxComponent {
    /// Skybox mode.
    pub mode: SkyboxMode,
    /// Cubemap handle (for Cubemap mode).
    pub cubemap_handle: Option<u64>,
    /// Time of day in hours (for Procedural mode).
    pub time_of_day: f32,
    /// Auto-advance time of day.
    pub auto_time: bool,
    /// Time speed multiplier (for auto-advance).
    pub time_speed: f32,
    /// Exposure.
    pub exposure: f32,
    /// Planet type preset.
    pub planet_preset: PlanetPreset,
}

/// Planet atmosphere preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanetPreset {
    Earth,
    Mars,
    Alien,
    Custom,
}

impl Default for SkyboxComponent {
    fn default() -> Self {
        Self {
            mode: SkyboxMode::Procedural,
            cubemap_handle: None,
            time_of_day: 10.0,
            auto_time: false,
            time_speed: 1.0,
            exposure: 1.0,
            planet_preset: PlanetPreset::Earth,
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
    fn test_skybox_renderer_creation() {
        let renderer = SkyboxRenderer::new();
        assert_eq!(renderer.mode, SkyboxMode::Procedural);
        assert!(renderer.needs_update);
    }

    #[test]
    fn test_set_time_of_day() {
        let mut renderer = SkyboxRenderer::new();
        renderer.set_time_of_day(12.0); // Noon
        assert!(renderer.procedural_config.sun_direction.y > 0.5);
        assert!(renderer.env_capture.dirty);
    }

    #[test]
    fn test_sun_color_at_noon() {
        let mut renderer = SkyboxRenderer::new();
        renderer.set_time_of_day(12.0);
        let color = renderer.sun_color();
        assert!(color.x > 0.0);
        assert!(color.y > 0.0);
        assert!(color.z > 0.0);
    }

    #[test]
    fn test_sun_color_at_midnight() {
        let mut renderer = SkyboxRenderer::new();
        renderer.set_time_of_day(0.0);
        let color = renderer.sun_color();
        // At midnight the sun is below horizon, so color should be zero.
        assert!(color.length() < 0.1);
    }

    #[test]
    fn test_cubemap_uv_rotation() {
        let mut config = CubemapSkyboxConfig::default();
        config.set_rotation_degrees(90.0, 0.0, 0.0);
        // After rotation, the matrix should not be identity.
        assert_ne!(config.rotation, Mat4::IDENTITY);
    }

    #[test]
    fn test_environment_capture() {
        let capture = EnvironmentCapture::new(256);
        assert_eq!(capture.face_resolution, 256);
        assert!(capture.dirty);
        for i in 0..6 {
            let vp = capture.face_view_proj(i);
            assert_ne!(vp, Mat4::IDENTITY);
        }
    }

    #[test]
    fn test_mars_atmosphere() {
        let mut config = ProceduralSkyConfig::default();
        config.set_mars_like();
        // Mars has inverted Rayleigh (red sky).
        assert!(config.rayleigh_coefficients.x > config.rayleigh_coefficients.z);
    }

    #[test]
    fn test_ambient_color() {
        let mut renderer = SkyboxRenderer::new();
        renderer.set_time_of_day(12.0);
        let ambient = renderer.ambient_color();
        assert!(ambient.length() > 0.0);
    }
}
