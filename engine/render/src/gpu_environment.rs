// engine/render/src/gpu_environment.rs
//
// Environment Maps and Sky Rendering for the Genovo engine.
//
// # Features
//
// - `EnvironmentMapRenderer`: cubemap texture for reflections
// - WGSL sky shader: sample cubemap for background
// - WGSL PBR update: sample environment map for specular IBL
// - Irradiance map: prefiltered diffuse from cubemap
// - Prefilter map: roughness-based mip selection for specular
// - Procedural sky: atmospheric scattering as fallback when no cubemap
// - Sky rendering: fullscreen triangle with reverse depth, sample sky direction

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

// ============================================================================
// Constants
// ============================================================================

/// Default cubemap face resolution.
pub const CUBEMAP_FACE_SIZE: u32 = 512;

/// Irradiance map resolution.
pub const IRRADIANCE_MAP_SIZE: u32 = 64;

/// Prefilter map resolution (per mip level).
pub const PREFILTER_MAP_SIZE: u32 = 256;

/// Number of roughness mip levels in the prefilter map.
pub const PREFILTER_MIP_LEVELS: u32 = 6;

/// BRDF LUT resolution.
pub const BRDF_LUT_SIZE: u32 = 512;

// ============================================================================
// Environment map configuration
// ============================================================================

/// Configuration for environment map rendering.
#[derive(Debug, Clone)]
pub struct EnvironmentConfig {
    /// Cubemap face resolution.
    pub cubemap_size: u32,
    /// Irradiance map resolution.
    pub irradiance_size: u32,
    /// Prefilter map resolution.
    pub prefilter_size: u32,
    /// Number of prefilter mip levels.
    pub prefilter_mip_levels: u32,
    /// Sky colour (top).
    pub sky_color_top: Vec3,
    /// Sky colour (bottom / horizon).
    pub sky_color_bottom: Vec3,
    /// Sun direction (normalized).
    pub sun_direction: Vec3,
    /// Sun colour.
    pub sun_color: Vec3,
    /// Sun intensity.
    pub sun_intensity: f32,
    /// Sun angular radius (radians).
    pub sun_radius: f32,
    /// Atmospheric density.
    pub atmosphere_density: f32,
    /// Use procedural sky (true) or cubemap (false).
    pub use_procedural_sky: bool,
    /// Environment map intensity multiplier.
    pub environment_intensity: f32,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            cubemap_size: CUBEMAP_FACE_SIZE,
            irradiance_size: IRRADIANCE_MAP_SIZE,
            prefilter_size: PREFILTER_MAP_SIZE,
            prefilter_mip_levels: PREFILTER_MIP_LEVELS,
            sky_color_top: Vec3::new(0.1, 0.3, 0.8),
            sky_color_bottom: Vec3::new(0.6, 0.7, 0.9),
            sun_direction: Vec3::new(0.4, 0.8, 0.4).normalize(),
            sun_color: Vec3::new(1.0, 0.95, 0.85),
            sun_intensity: 20.0,
            sun_radius: 0.01,
            atmosphere_density: 1.0,
            use_procedural_sky: true,
            environment_intensity: 1.0,
        }
    }
}

// ============================================================================
// Environment uniform
// ============================================================================

/// Environment rendering parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct EnvironmentUniform {
    /// Inverse view-projection matrix (for ray direction from screen UV).
    pub inv_view_proj: [[f32; 4]; 4],
    /// Camera position.
    pub camera_position: [f32; 4],
    /// Sky colour top (RGB) + unused.
    pub sky_color_top: [f32; 4],
    /// Sky colour bottom (RGB) + unused.
    pub sky_color_bottom: [f32; 4],
    /// Sun direction (XYZ) + sun_radius.
    pub sun_direction: [f32; 4],
    /// Sun colour (RGB) + intensity.
    pub sun_color_intensity: [f32; 4],
    /// .x = atmosphere_density, .y = env_intensity, .z = use_procedural, .w = time.
    pub env_params: [f32; 4],
}

// ============================================================================
// Procedural sky shader
// ============================================================================

/// WGSL shader for procedural sky rendering.
pub const PROCEDURAL_SKY_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Procedural Sky Shader
// ============================================================================
//
// Renders a procedural sky using atmospheric scattering approximation.
// Includes sun disc, Rayleigh scattering gradient, and Mie scattering halo.
//
// Rendered as a fullscreen triangle at the far plane.

struct EnvironmentParams {
    inv_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sky_color_top: vec4<f32>,
    sky_color_bottom: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color_intensity: vec4<f32>,
    env_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> env: EnvironmentParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    // Fullscreen triangle at far depth.
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);

    return output;
}

// Reconstruct view ray direction from UV.
fn get_ray_direction(uv: vec2<f32>, inv_vp: mat4x4<f32>) -> vec3<f32> {
    // NDC coordinates.
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    // Unproject near and far points.
    let near_point = inv_vp * vec4<f32>(ndc, 0.0, 1.0);
    let far_point = inv_vp * vec4<f32>(ndc, 1.0, 1.0);

    let near_world = near_point.xyz / near_point.w;
    let far_world = far_point.xyz / far_point.w;

    return normalize(far_world - near_world);
}

// Simple Rayleigh scattering approximation.
fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 0.75 * (1.0 + cos_theta * cos_theta);
}

// Mie scattering phase function (Henyey-Greenstein).
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = 1.0 - g2;
    let denom = 4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return num / denom;
}

// Atmospheric scattering sky color.
fn atmosphere_color(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sky_top = env.sky_color_top.xyz;
    let sky_bottom = env.sky_color_bottom.xyz;

    // Height gradient.
    let y = ray_dir.y;
    let t = clamp(y * 0.5 + 0.5, 0.0, 1.0);

    // Base sky colour (gradient from bottom to top).
    var sky = mix(sky_bottom, sky_top, pow(t, 0.6));

    // Darken below horizon.
    if y < 0.0 {
        let below = clamp(-y * 3.0, 0.0, 1.0);
        sky = mix(sky, sky_bottom * 0.15, below);
    }

    // Rayleigh scattering (blue sky).
    let cos_sun = dot(ray_dir, sun_dir);
    let rayleigh = rayleigh_phase(cos_sun);
    let rayleigh_color = vec3<f32>(0.2, 0.4, 0.8);
    sky = sky + rayleigh_color * rayleigh * 0.05 * env.env_params.x;

    // Mie scattering (halo around sun).
    let mie = mie_phase(cos_sun, 0.76);
    let mie_color = env.sun_color_intensity.xyz * 0.1;
    sky = sky + mie_color * mie * 0.15 * env.env_params.x;

    // Sun disc.
    let sun_radius = env.sun_direction.w;
    let sun_intensity = env.sun_color_intensity.w;
    let sun_angle = acos(clamp(cos_sun, -1.0, 1.0));

    if sun_angle < sun_radius * 3.0 {
        // Soft edge sun disc.
        let sun_edge = smoothstep(sun_radius, sun_radius * 0.5, sun_angle);
        let sun_glow = exp(-sun_angle * sun_angle / (sun_radius * sun_radius * 4.0));
        sky = sky + env.sun_color_intensity.xyz * (sun_edge * sun_intensity + sun_glow * sun_intensity * 0.3);
    }

    // Sunset/sunrise colouring near horizon.
    let horizon_factor = exp(-abs(y) * 5.0);
    let sunset_dir = abs(cos_sun);
    let sunset_color = vec3<f32>(1.0, 0.4, 0.1) * sunset_dir;
    sky = sky + sunset_color * horizon_factor * 0.3 * env.env_params.x;

    return sky;
}

@fragment
fn fs_sky(input: VertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = get_ray_direction(input.uv, env.inv_view_proj);
    let sun_dir = normalize(env.sun_direction.xyz);

    var color = atmosphere_color(ray_dir, sun_dir);

    // Simple exposure / tone mapping.
    color = color * env.env_params.y; // environment intensity
    color = color / (color + vec3<f32>(1.0)); // Reinhard

    return vec4<f32>(color, 1.0);
}
"#;

// ============================================================================
// Cubemap sky shader
// ============================================================================

/// WGSL shader for cubemap sky rendering.
pub const CUBEMAP_SKY_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Cubemap Sky Shader
// ============================================================================
//
// Renders the sky by sampling a cubemap texture.

struct EnvironmentParams {
    inv_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sky_color_top: vec4<f32>,
    sky_color_bottom: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color_intensity: vec4<f32>,
    env_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> env: EnvironmentParams;
@group(0) @binding(1) var sky_cubemap: texture_cube<f32>;
@group(0) @binding(2) var sky_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_cubemap_sky(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

fn get_ray_dir_cubemap(uv: vec2<f32>, inv_vp: mat4x4<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let near_point = inv_vp * vec4<f32>(ndc, 0.0, 1.0);
    let far_point = inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    return normalize(far_point.xyz / far_point.w - near_point.xyz / near_point.w);
}

@fragment
fn fs_cubemap_sky(input: VertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = get_ray_dir_cubemap(input.uv, env.inv_view_proj);
    var color = textureSample(sky_cubemap, sky_sampler, ray_dir).xyz;

    color = color * env.env_params.y; // environment intensity

    return vec4<f32>(color, 1.0);
}
"#;

// ============================================================================
// IBL (Image-Based Lighting) shaders
// ============================================================================

/// WGSL shader for irradiance map convolution.
pub const IRRADIANCE_CONVOLUTION_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Irradiance Map Convolution
// ============================================================================
//
// Convolves the environment cubemap to compute the diffuse irradiance map.
// Integrates the incoming light over the hemisphere around each normal direction.

struct ConvolutionParams {
    face_index: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: ConvolutionParams;
@group(0) @binding(1) var env_cubemap: texture_cube<f32>;
@group(0) @binding(2) var env_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_irradiance(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

const PI: f32 = 3.14159265359;

// Convert face UV + face index to a cubemap direction.
fn face_uv_to_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;

    switch face {
        case 0u: { return normalize(vec3<f32>( 1.0, -v,  -u)); } // +X
        case 1u: { return normalize(vec3<f32>(-1.0, -v,   u)); } // -X
        case 2u: { return normalize(vec3<f32>(   u, 1.0,   v)); } // +Y
        case 3u: { return normalize(vec3<f32>(   u,-1.0,  -v)); } // -Y
        case 4u: { return normalize(vec3<f32>(   u, -v,  1.0)); } // +Z
        case 5u: { return normalize(vec3<f32>(  -u, -v, -1.0)); } // -Z
        default: { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

@fragment
fn fs_irradiance(input: VertexOutput) -> @location(0) vec4<f32> {
    let face = u32(params.face_index.x);
    let normal = face_uv_to_direction(face, input.uv);

    // Build TBN from normal.
    var up_vec = vec3<f32>(0.0, 1.0, 0.0);
    if abs(normal.y) > 0.999 {
        up_vec = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(up_vec, normal));
    let up = cross(normal, right);

    // Monte Carlo integration over the hemisphere.
    var irradiance = vec3<f32>(0.0);
    let sample_delta = 0.05;
    var num_samples = 0.0;

    var phi = 0.0;
    loop {
        if phi >= 2.0 * PI { break; }

        var theta = 0.0;
        loop {
            if theta >= 0.5 * PI { break; }

            // Spherical to Cartesian (tangent space).
            let tangent_sample = vec3<f32>(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
            );

            // Tangent space to world space.
            let sample_dir = tangent_sample.x * right +
                             tangent_sample.y * up +
                             tangent_sample.z * normal;

            // Sample the cubemap.
            let env_color = textureSample(env_cubemap, env_sampler, sample_dir).xyz;

            // cos(theta) * sin(theta) is the solid angle weight.
            irradiance = irradiance + env_color * cos(theta) * sin(theta);
            num_samples = num_samples + 1.0;

            theta = theta + sample_delta;
        }

        phi = phi + sample_delta;
    }

    irradiance = PI * irradiance / num_samples;

    return vec4<f32>(irradiance, 1.0);
}
"#;

/// WGSL shader for specular prefilter map generation.
pub const PREFILTER_MAP_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- Specular Prefilter Map
// ============================================================================
//
// Generates a prefiltered environment map for specular IBL.
// Each mip level corresponds to a roughness value.

struct PrefilterParams {
    face_roughness: vec4<f32>,  // .x = face, .y = roughness, .z = resolution, .w = unused
};

@group(0) @binding(0) var<uniform> params: PrefilterParams;
@group(0) @binding(1) var env_cubemap: texture_cube<f32>;
@group(0) @binding(2) var env_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_prefilter(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

const PI: f32 = 3.14159265359;

fn face_uv_to_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;
    switch face {
        case 0u: { return normalize(vec3<f32>( 1.0, -v, -u)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -v,  u)); }
        case 2u: { return normalize(vec3<f32>(   u, 1.0,  v)); }
        case 3u: { return normalize(vec3<f32>(   u,-1.0, -v)); }
        case 4u: { return normalize(vec3<f32>(   u, -v, 1.0)); }
        case 5u: { return normalize(vec3<f32>(  -u, -v,-1.0)); }
        default: { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

// Hammersley quasi-random sequence.
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

// GGX importance sampling.
fn importance_sample_ggx(xi: vec2<f32>, roughness: f32, n: vec3<f32>) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Spherical to Cartesian (tangent space).
    let h = vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );

    // Tangent-space to world-space.
    var up_v = vec3<f32>(0.0, 1.0, 0.0);
    if abs(n.y) > 0.999 { up_v = vec3<f32>(1.0, 0.0, 0.0); }
    let tangent = normalize(cross(up_v, n));
    let bitangent = cross(n, tangent);

    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

@fragment
fn fs_prefilter(input: VertexOutput) -> @location(0) vec4<f32> {
    let face = u32(params.face_roughness.x);
    let roughness = params.face_roughness.y;
    let resolution = params.face_roughness.z;

    let normal = face_uv_to_dir(face, input.uv);
    let r = normal;
    let v = r;

    let sample_count = 1024u;
    var prefiltered = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < sample_count; i = i + 1u) {
        let xi = hammersley(i, sample_count);
        let h = importance_sample_ggx(xi, roughness, normal);
        let l = normalize(2.0 * dot(v, h) * h - v);

        let n_dot_l = max(dot(normal, l), 0.0);
        if n_dot_l > 0.0 {
            let env_color = textureSample(env_cubemap, env_sampler, l).xyz;
            prefiltered = prefiltered + env_color * n_dot_l;
            total_weight = total_weight + n_dot_l;
        }
    }

    prefiltered = prefiltered / total_weight;

    return vec4<f32>(prefiltered, 1.0);
}
"#;

/// WGSL shader for BRDF integration LUT.
pub const BRDF_LUT_SHADER_WGSL: &str = r#"
// ============================================================================
// Genovo Engine -- BRDF Integration LUT
// ============================================================================
//
// Precomputes the split-sum BRDF integration lookup table.
// Input: n_dot_v (x-axis), roughness (y-axis).
// Output: (scale, bias) for the split-sum approximation.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_brdf_lut(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

const PI: f32 = 3.14159265359;

fn radical_inverse_vdc_brdf(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley_brdf(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc_brdf(i));
}

fn importance_sample_ggx_brdf(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn geometry_schlick_ggx_brdf(n_dot_v: f32, roughness: f32) -> f32 {
    let a = roughness;
    let k = (a * a) / 2.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith_brdf(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx_brdf(n_dot_v, roughness) *
           geometry_schlick_ggx_brdf(n_dot_l, roughness);
}

@fragment
fn fs_brdf_lut(input: VertexOutput) -> @location(0) vec4<f32> {
    let n_dot_v = max(input.uv.x, 0.001);
    let roughness = input.uv.y;

    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let n = vec3<f32>(0.0, 0.0, 1.0);

    var a = 0.0;
    var b = 0.0;

    let sample_count = 1024u;

    for (var i = 0u; i < sample_count; i = i + 1u) {
        let xi = hammersley_brdf(i, sample_count);
        let h = importance_sample_ggx_brdf(xi, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);

        let n_dot_l = max(l.z, 0.0);
        let n_dot_h = max(h.z, 0.0);
        let v_dot_h = max(dot(v, h), 0.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith_brdf(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v);
            let fc = pow(1.0 - v_dot_h, 5.0);

            a = a + (1.0 - fc) * g_vis;
            b = b + fc * g_vis;
        }
    }

    a = a / f32(sample_count);
    b = b / f32(sample_count);

    return vec4<f32>(a, b, 0.0, 1.0);
}
"#;

// ============================================================================
// Environment map renderer
// ============================================================================

/// Renders the sky and manages environment cubemaps for IBL.
pub struct EnvironmentMapRenderer {
    /// Configuration.
    pub config: EnvironmentConfig,
    /// Procedural sky pipeline.
    procedural_sky_pipeline: wgpu::RenderPipeline,
    /// Cubemap sky pipeline.
    cubemap_sky_pipeline: wgpu::RenderPipeline,
    /// BRDF LUT generation pipeline.
    brdf_lut_pipeline: wgpu::RenderPipeline,
    /// Irradiance convolution pipeline.
    irradiance_pipeline: wgpu::RenderPipeline,
    /// Prefilter pipeline.
    prefilter_pipeline: wgpu::RenderPipeline,

    /// Environment uniform buffer.
    env_uniform_buffer: wgpu::Buffer,
    /// Procedural sky bind group layout.
    procedural_bgl: wgpu::BindGroupLayout,
    /// Cubemap sky bind group layout.
    cubemap_bgl: wgpu::BindGroupLayout,
    /// IBL convolution bind group layout.
    convolution_bgl: wgpu::BindGroupLayout,
    /// Procedural sky bind group.
    procedural_bind_group: wgpu::BindGroup,

    /// BRDF integration LUT texture.
    pub brdf_lut_texture: Option<wgpu::Texture>,
    pub brdf_lut_view: Option<wgpu::TextureView>,

    /// Irradiance map (cubemap, 6 faces).
    pub irradiance_texture: Option<wgpu::Texture>,
    pub irradiance_view: Option<wgpu::TextureView>,

    /// Prefilter map (cubemap with mips).
    pub prefilter_texture: Option<wgpu::Texture>,
    pub prefilter_view: Option<wgpu::TextureView>,

    /// Cubemap sampler.
    pub cubemap_sampler: wgpu::Sampler,
    /// BRDF LUT sampler.
    pub lut_sampler: wgpu::Sampler,

    /// Convolution params buffer (for face index, roughness).
    convolution_params_buffer: wgpu::Buffer,
}

impl EnvironmentMapRenderer {
    /// Create a new environment map renderer.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        config: EnvironmentConfig,
    ) -> Self {
        // --- Bind group layouts ---
        let procedural_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("procedural_sky_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let cubemap_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cubemap_sky_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let convolution_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ibl_convolution_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // --- Pipelines ---
        let create_sky_pipeline =
            |shader_src: &str, vs: &str, fs: &str, bgl: &wgpu::BindGroupLayout, label: &str| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{}_layout", label)),
                    bind_group_layouts: &[bgl],
                    push_constant_ranges: &[],
                });
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some(vs),
                        buffers: &[],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(fs),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: false,
                        depth_compare: wgpu::CompareFunction::LessEqual,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
            };

        let procedural_sky_pipeline = create_sky_pipeline(
            PROCEDURAL_SKY_SHADER_WGSL,
            "vs_sky",
            "fs_sky",
            &procedural_bgl,
            "procedural_sky_pipeline",
        );

        let cubemap_sky_pipeline = create_sky_pipeline(
            CUBEMAP_SKY_SHADER_WGSL,
            "vs_cubemap_sky",
            "fs_cubemap_sky",
            &cubemap_bgl,
            "cubemap_sky_pipeline",
        );

        // BRDF LUT pipeline (no depth).
        let brdf_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brdf_lut_shader"),
            source: wgpu::ShaderSource::Wgsl(BRDF_LUT_SHADER_WGSL.into()),
        });
        let brdf_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brdf_lut_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let brdf_lut_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("brdf_lut_pipeline"),
            layout: Some(&brdf_layout),
            vertex: wgpu::VertexState {
                module: &brdf_module,
                entry_point: Some("vs_brdf_lut"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &brdf_module,
                entry_point: Some("fs_brdf_lut"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Irradiance pipeline.
        let irradiance_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("irradiance_shader"),
            source: wgpu::ShaderSource::Wgsl(IRRADIANCE_CONVOLUTION_WGSL.into()),
        });
        let irradiance_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("irradiance_layout"),
            bind_group_layouts: &[&convolution_bgl],
            push_constant_ranges: &[],
        });
        let irradiance_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("irradiance_pipeline"),
            layout: Some(&irradiance_layout),
            vertex: wgpu::VertexState {
                module: &irradiance_module,
                entry_point: Some("vs_irradiance"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &irradiance_module,
                entry_point: Some("fs_irradiance"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Prefilter pipeline.
        let prefilter_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefilter_shader"),
            source: wgpu::ShaderSource::Wgsl(PREFILTER_MAP_WGSL.into()),
        });
        let prefilter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prefilter_layout"),
            bind_group_layouts: &[&convolution_bgl],
            push_constant_ranges: &[],
        });
        let prefilter_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prefilter_pipeline"),
            layout: Some(&prefilter_layout),
            vertex: wgpu::VertexState {
                module: &prefilter_module,
                entry_point: Some("vs_prefilter"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &prefilter_module,
                entry_point: Some("fs_prefilter"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Uniform buffers ---
        let env_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("env_uniform_buffer"),
            size: std::mem::size_of::<EnvironmentUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let convolution_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convolution_params_buffer"),
            size: 16, // vec4<f32>
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Procedural sky bind group ---
        let procedural_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("procedural_sky_bg"),
            layout: &procedural_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: env_uniform_buffer.as_entire_binding(),
            }],
        });

        // --- Samplers ---
        let cubemap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cubemap_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("brdf_lut_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            config,
            procedural_sky_pipeline,
            cubemap_sky_pipeline,
            brdf_lut_pipeline,
            irradiance_pipeline,
            prefilter_pipeline,
            env_uniform_buffer,
            procedural_bgl,
            cubemap_bgl,
            convolution_bgl,
            procedural_bind_group,
            brdf_lut_texture: None,
            brdf_lut_view: None,
            irradiance_texture: None,
            irradiance_view: None,
            prefilter_texture: None,
            prefilter_view: None,
            cubemap_sampler,
            lut_sampler,
            convolution_params_buffer,
        }
    }

    /// Update environment parameters.
    pub fn update_params(
        &self,
        queue: &wgpu::Queue,
        camera_view: Mat4,
        camera_proj: Mat4,
        camera_pos: Vec3,
        time: f32,
    ) {
        let vp = camera_proj * camera_view;
        let inv_vp = vp.inverse();

        let uniform = EnvironmentUniform {
            inv_view_proj: inv_vp.to_cols_array_2d(),
            camera_position: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            sky_color_top: [
                self.config.sky_color_top.x,
                self.config.sky_color_top.y,
                self.config.sky_color_top.z,
                0.0,
            ],
            sky_color_bottom: [
                self.config.sky_color_bottom.x,
                self.config.sky_color_bottom.y,
                self.config.sky_color_bottom.z,
                0.0,
            ],
            sun_direction: [
                self.config.sun_direction.x,
                self.config.sun_direction.y,
                self.config.sun_direction.z,
                self.config.sun_radius,
            ],
            sun_color_intensity: [
                self.config.sun_color.x,
                self.config.sun_color.y,
                self.config.sun_color.z,
                self.config.sun_intensity,
            ],
            env_params: [
                self.config.atmosphere_density,
                self.config.environment_intensity,
                if self.config.use_procedural_sky { 1.0 } else { 0.0 },
                time,
            ],
        };

        queue.write_buffer(
            &self.env_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniform),
        );
    }

    /// Generate the BRDF integration LUT.
    pub fn generate_brdf_lut(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brdf_lut"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_SIZE,
                height: BRDF_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brdf_lut_encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("brdf_lut_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.brdf_lut_pipeline);
            pass.draw(0..3, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        self.brdf_lut_texture = Some(texture);
        self.brdf_lut_view = Some(view);
    }

    /// Render the procedural sky.
    pub fn render_procedural_sky<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        pass.set_pipeline(&self.procedural_sky_pipeline);
        pass.set_bind_group(0, &self.procedural_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Access the procedural sky bind group layout.
    pub fn procedural_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.procedural_bgl
    }

    /// Access the cubemap sky bind group layout.
    pub fn cubemap_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.cubemap_bgl
    }
}
