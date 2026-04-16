// engine/render/src/pbr/shader_gen.rs
//
// Dynamic WGSL shader generation for PBR materials. Builds vertex and fragment
// shaders from a set of feature flags, allowing the engine to generate optimal
// shader permutations without a combinatorial explosion of handwritten variants.

use std::fmt::Write;

// ---------------------------------------------------------------------------
// MaterialFeatures
// ---------------------------------------------------------------------------

bitflags::bitflags! {
    /// Feature flags controlling which PBR shader features are enabled.
    ///
    /// Each flag gates a block of shader code, allowing dead-code elimination
    /// at the source level before compilation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MaterialFeatures: u32 {
        /// Material has an albedo texture map.
        const HAS_ALBEDO_MAP             = 1 << 0;
        /// Material has a tangent-space normal map.
        const HAS_NORMAL_MAP             = 1 << 1;
        /// Material has a packed metallic-roughness map.
        const HAS_METALLIC_ROUGHNESS_MAP = 1 << 2;
        /// Material has an ambient-occlusion map.
        const HAS_AO_MAP                 = 1 << 3;
        /// Material has an emissive colour map.
        const HAS_EMISSIVE_MAP           = 1 << 4;
        /// Material uses alpha-mask transparency.
        const ALPHA_MASK                 = 1 << 5;
        /// Material uses alpha-blend transparency.
        const ALPHA_BLEND                = 1 << 6;
        /// Material is double-sided (no backface culling).
        const DOUBLE_SIDED               = 1 << 7;
        /// Material has a height/parallax map.
        const HAS_HEIGHT_MAP             = 1 << 8;
        /// Material has a detail normal map.
        const HAS_DETAIL_NORMAL          = 1 << 9;
        /// Material has a detail albedo map.
        const HAS_DETAIL_ALBEDO          = 1 << 10;
        /// Material has a clearcoat layer.
        const HAS_CLEARCOAT              = 1 << 11;
        /// Material uses anisotropic reflections.
        const HAS_ANISOTROPY             = 1 << 12;
        /// Material has subsurface scattering.
        const HAS_SUBSURFACE             = 1 << 13;
        /// Material has a sheen layer (fabric).
        const HAS_SHEEN                  = 1 << 14;
        /// Enable shadow receiving.
        const RECEIVE_SHADOWS            = 1 << 15;
        /// Enable IBL (image-based lighting).
        const USE_IBL                    = 1 << 16;
        /// Use vertex colours.
        const HAS_VERTEX_COLORS          = 1 << 17;
        /// Has a second UV channel.
        const HAS_UV1                    = 1 << 18;
        /// Enable skinned mesh (bone transforms).
        const SKINNED                    = 1 << 19;
        /// Enable instanced rendering.
        const INSTANCED                  = 1 << 20;
        /// Has a lightmap.
        const HAS_LIGHTMAP               = 1 << 21;
    }
}

impl MaterialFeatures {
    /// Compute a cache key suitable for a shader permutation cache.
    pub fn cache_key(self) -> u32 {
        self.bits()
    }

    /// A minimal feature set for a basic opaque PBR material.
    pub fn basic_opaque() -> Self {
        Self::empty()
    }

    /// A typical textured PBR material with albedo, normal, and
    /// metallic-roughness maps.
    pub fn standard_textured() -> Self {
        Self::HAS_ALBEDO_MAP | Self::HAS_NORMAL_MAP | Self::HAS_METALLIC_ROUGHNESS_MAP
    }

    /// Full-featured PBR with IBL, shadows, and AO.
    pub fn full_pbr() -> Self {
        Self::standard_textured()
            | Self::HAS_AO_MAP
            | Self::HAS_EMISSIVE_MAP
            | Self::RECEIVE_SHADOWS
            | Self::USE_IBL
    }
}

// ---------------------------------------------------------------------------
// Shader code constants — building blocks
// ---------------------------------------------------------------------------

/// Camera / scene uniform struct (bind group 0, binding 0).
const WGSL_SCENE_UNIFORMS: &str = r#"
struct SceneUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    time: f32,
    screen_size: vec2<f32>,
    near_plane: f32,
    far_plane: f32,
};

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
"#;

/// Material uniform struct (bind group 1, binding 0).
const WGSL_MATERIAL_UNIFORMS: &str = r#"
struct MaterialUniforms {
    albedo_color: vec4<f32>,
    emissive: vec4<f32>,          // rgb = color, a = strength
    metallic_roughness: vec4<f32>, // x=metallic, y=roughness, z=reflectance, w=normal_scale
    ao_alpha_clearcoat: vec4<f32>, // x=ao_strength, y=alpha_cutoff, z=clearcoat, w=clearcoat_roughness
    aniso_subsurface_sheen: vec4<f32>, // x=anisotropy, y=aniso_rotation, z=subsurface, w=sheen_roughness
    subsurface_color: vec4<f32>,  // rgb = color, a = padding
    sheen_color_flags: vec4<f32>, // rgb = sheen, a = feature flags (as bits)
    _reserved: vec4<f32>,
};

@group(1) @binding(0) var<uniform> material: MaterialUniforms;
"#;

/// Model / object uniform struct (bind group 2, binding 0).
const WGSL_MODEL_UNIFORMS: &str = r#"
struct ModelUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
};

@group(2) @binding(0) var<uniform> model: ModelUniforms;
"#;

/// Light data struct and light buffer.
const WGSL_LIGHT_BUFFER: &str = r#"
struct LightData {
    position_type: vec4<f32>,     // xyz = position/direction, w = light type
    color_intensity: vec4<f32>,   // rgb = color, a = intensity
    direction_range: vec4<f32>,   // xyz = direction (spot), w = range
    spot_params: vec4<f32>,       // x = inner_angle_cos, y = outer_angle_cos, z = shadow_idx, w = padding
};

struct LightBuffer {
    count: vec4<u32>,             // x = light count
    lights: array<LightData, 128>,
};

@group(0) @binding(1) var<storage, read> light_buffer: LightBuffer;
"#;

/// Shadow map sampler and cascade matrices.
const WGSL_SHADOW_BINDINGS: &str = r#"
struct ShadowUniforms {
    cascade_matrices: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    shadow_bias: vec4<f32>,       // x=depth_bias, y=normal_bias, z=pcf_radius, w=padding
};

@group(0) @binding(2) var<uniform> shadow_uniforms: ShadowUniforms;
@group(0) @binding(3) var shadow_map: texture_depth_2d_array;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;
"#;

/// IBL texture bindings.
const WGSL_IBL_BINDINGS: &str = r#"
@group(0) @binding(5) var irradiance_map: texture_cube<f32>;
@group(0) @binding(6) var prefilter_map: texture_cube<f32>;
@group(0) @binding(7) var brdf_lut: texture_2d<f32>;
@group(0) @binding(8) var env_sampler: sampler;
"#;

/// Vertex output struct.
const WGSL_VERTEX_OUTPUT_BASE: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
"#;

/// PBR math functions.
const WGSL_PBR_FUNCTIONS: &str = r#"
// ---------------------------------------------------------------------------
// PBR Math Functions
// ---------------------------------------------------------------------------

const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067;

// GGX Normal Distribution Function
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denom * denom + 0.0000001);
}

// Smith-GGX Geometry (correlated)
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let lambda_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2);
    let lambda_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2);
    return 0.5 / max(lambda_v + lambda_l, 0.0000001);
}

// Schlick Fresnel approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let factor = pow(max(1.0 - cos_theta, 0.0), 5.0);
    return f0 + (vec3<f32>(1.0) - f0) * factor;
}

// Schlick Fresnel with roughness (for IBL)
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let factor = pow(max(1.0 - cos_theta, 0.0), 5.0);
    let max_f0 = max(vec3<f32>(1.0 - roughness), f0);
    return f0 + (max_f0 - f0) * factor;
}

// Lambertian diffuse BRDF
fn diffuse_lambertian(albedo: vec3<f32>) -> vec3<f32> {
    return albedo * INV_PI;
}

// Compute F0 from material parameters
fn compute_f0(albedo: vec3<f32>, metallic: f32, reflectance: f32) -> vec3<f32> {
    let dielectric_f0 = vec3<f32>(0.16 * reflectance * reflectance);
    return dielectric_f0 * (1.0 - metallic) + albedo * metallic;
}

// Light attenuation (inverse-square with smooth windowing)
fn light_attenuation(distance: f32, range: f32) -> f32 {
    let dist2 = distance * distance;
    let range2 = range * range;
    let factor = dist2 / range2;
    let smooth = saturate(1.0 - factor * factor);
    return (smooth * smooth) / max(dist2, 0.0001);
}

// Spot light angle falloff
fn spot_falloff(cos_angle: f32, inner_cos: f32, outer_cos: f32) -> f32 {
    return saturate((cos_angle - outer_cos) / max(inner_cos - outer_cos, 0.0001));
}

// Reinhard tone mapping
fn tone_map_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

// ACES tone mapping approximation (Narkowicz 2015)
fn tone_map_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Linear to sRGB
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(color.x <= 0.0031308, color.y <= 0.0031308, color.z <= 0.0031308);
    let higher = vec3<f32>(
        1.055 * pow(color.x, 1.0 / 2.4) - 0.055,
        1.055 * pow(color.y, 1.0 / 2.4) - 0.055,
        1.055 * pow(color.z, 1.0 / 2.4) - 0.055,
    );
    let lower = color * 12.92;
    return mix(higher, lower, cutoff);
}
"#;

/// Shadow sampling functions.
const WGSL_SHADOW_FUNCTIONS: &str = r#"
// ---------------------------------------------------------------------------
// Shadow Functions
// ---------------------------------------------------------------------------

// Select cascade index based on view-space depth
fn select_cascade(view_depth: f32) -> u32 {
    var cascade: u32 = 0u;
    if view_depth > shadow_uniforms.cascade_splits.x {
        cascade = 1u;
    }
    if view_depth > shadow_uniforms.cascade_splits.y {
        cascade = 2u;
    }
    if view_depth > shadow_uniforms.cascade_splits.z {
        cascade = 3u;
    }
    return cascade;
}

// Sample shadow map with PCF (3x3 kernel)
fn sample_shadow_pcf(world_pos: vec3<f32>, cascade: u32) -> f32 {
    let light_space = shadow_uniforms.cascade_matrices[cascade] * vec4<f32>(world_pos, 1.0);
    let proj = light_space.xyz / light_space.w;
    let shadow_uv = proj.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);

    if shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0 {
        return 1.0;
    }

    let depth = proj.z - shadow_uniforms.shadow_bias.x;
    let texel_size = 1.0 / f32(textureDimensions(shadow_map).x);
    var shadow = 0.0;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow += textureSampleCompareLevel(
                shadow_map, shadow_sampler,
                shadow_uv + offset, cascade, depth
            );
        }
    }

    return shadow / 9.0;
}
"#;

/// Normal mapping helper.
const WGSL_NORMAL_MAP_FUNCTION: &str = r#"
// Perturb normal using normal map and TBN matrix
fn perturb_normal(normal_sample: vec3<f32>, world_normal: vec3<f32>, world_tangent: vec3<f32>, world_bitangent: vec3<f32>, normal_scale: f32) -> vec3<f32> {
    var mapped = normal_sample * 2.0 - vec3<f32>(1.0);
    mapped.x *= normal_scale;
    mapped.y *= normal_scale;
    let tbn = mat3x3<f32>(
        normalize(world_tangent),
        normalize(world_bitangent),
        normalize(world_normal),
    );
    return normalize(tbn * mapped);
}
"#;

/// Clearcoat helper functions.
const WGSL_CLEARCOAT_FUNCTIONS: &str = r#"
// Clearcoat lobe: GGX with fixed IOR 1.5 (F0 = 0.04)
fn evaluate_clearcoat(n_dot_h: f32, l_dot_h: f32, clearcoat: f32, clearcoat_roughness: f32) -> vec2<f32> {
    let d_cc = distribution_ggx(n_dot_h, clearcoat_roughness);
    let f_cc = 0.04 + (1.0 - 0.04) * pow(1.0 - l_dot_h, 5.0);
    let g_cc = 0.25 / max(l_dot_h * l_dot_h, 0.0001);
    let specular_cc = d_cc * f_cc * g_cc * clearcoat;
    let attenuation = 1.0 - f_cc * clearcoat;
    return vec2<f32>(specular_cc, attenuation);
}
"#;

// ---------------------------------------------------------------------------
// Shader generation
// ---------------------------------------------------------------------------

/// Generate a complete WGSL PBR shader for the given feature set.
///
/// The generated shader contains both vertex and fragment entry points
/// (`vs_main` and `fs_main`) with all necessary struct definitions,
/// uniforms, and PBR lighting calculations.
pub fn generate_pbr_shader(features: MaterialFeatures) -> String {
    let mut s = String::with_capacity(8192);

    // Header comment.
    writeln!(s, "// Auto-generated PBR shader").unwrap();
    writeln!(s, "// Features: {features:?}").unwrap();
    writeln!(s).unwrap();

    // Scene uniforms.
    s.push_str(WGSL_SCENE_UNIFORMS);

    // Light buffer.
    s.push_str(WGSL_LIGHT_BUFFER);

    // Shadow bindings (conditional).
    if features.contains(MaterialFeatures::RECEIVE_SHADOWS) {
        s.push_str(WGSL_SHADOW_BINDINGS);
    }

    // IBL bindings (conditional).
    if features.contains(MaterialFeatures::USE_IBL) {
        s.push_str(WGSL_IBL_BINDINGS);
    }

    // Material uniforms.
    s.push_str(WGSL_MATERIAL_UNIFORMS);

    // Material texture bindings.
    generate_texture_bindings(&mut s, features);

    // Model uniforms.
    s.push_str(WGSL_MODEL_UNIFORMS);

    // Vertex output struct.
    generate_vertex_output_struct(&mut s, features);

    // Vertex input struct.
    generate_vertex_input_struct(&mut s, features);

    // PBR math functions.
    s.push_str(WGSL_PBR_FUNCTIONS);

    // Normal mapping.
    if features.contains(MaterialFeatures::HAS_NORMAL_MAP) {
        s.push_str(WGSL_NORMAL_MAP_FUNCTION);
    }

    // Shadow functions.
    if features.contains(MaterialFeatures::RECEIVE_SHADOWS) {
        s.push_str(WGSL_SHADOW_FUNCTIONS);
    }

    // Clearcoat functions.
    if features.contains(MaterialFeatures::HAS_CLEARCOAT) {
        s.push_str(WGSL_CLEARCOAT_FUNCTIONS);
    }

    // Vertex shader.
    generate_vertex_shader(&mut s, features);

    // Fragment shader.
    generate_fragment_shader(&mut s, features);

    s
}

/// Generate texture sampler bindings for material textures.
fn generate_texture_bindings(s: &mut String, features: MaterialFeatures) {
    writeln!(s).unwrap();
    writeln!(s, "// Material texture bindings (group 1)").unwrap();

    let mut binding = 1u32; // binding 0 is material uniforms

    if features.contains(MaterialFeatures::HAS_ALBEDO_MAP) {
        writeln!(s, "@group(1) @binding({binding}) var albedo_texture: texture_2d<f32>;").unwrap();
        binding += 1;
        writeln!(s, "@group(1) @binding({binding}) var albedo_sampler: sampler;").unwrap();
        binding += 1;
    }
    if features.contains(MaterialFeatures::HAS_NORMAL_MAP) {
        writeln!(s, "@group(1) @binding({binding}) var normal_texture: texture_2d<f32>;").unwrap();
        binding += 1;
        writeln!(s, "@group(1) @binding({binding}) var normal_sampler: sampler;").unwrap();
        binding += 1;
    }
    if features.contains(MaterialFeatures::HAS_METALLIC_ROUGHNESS_MAP) {
        writeln!(
            s,
            "@group(1) @binding({binding}) var metallic_roughness_texture: texture_2d<f32>;"
        )
        .unwrap();
        binding += 1;
        writeln!(
            s,
            "@group(1) @binding({binding}) var metallic_roughness_sampler: sampler;"
        )
        .unwrap();
        binding += 1;
    }
    if features.contains(MaterialFeatures::HAS_AO_MAP) {
        writeln!(s, "@group(1) @binding({binding}) var ao_texture: texture_2d<f32>;").unwrap();
        binding += 1;
        writeln!(s, "@group(1) @binding({binding}) var ao_sampler: sampler;").unwrap();
        binding += 1;
    }
    if features.contains(MaterialFeatures::HAS_EMISSIVE_MAP) {
        writeln!(
            s,
            "@group(1) @binding({binding}) var emissive_texture: texture_2d<f32>;"
        )
        .unwrap();
        binding += 1;
        writeln!(s, "@group(1) @binding({binding}) var emissive_sampler: sampler;").unwrap();
        binding += 1;
    }
    if features.contains(MaterialFeatures::HAS_HEIGHT_MAP) {
        writeln!(s, "@group(1) @binding({binding}) var height_texture: texture_2d<f32>;").unwrap();
        binding += 1;
        writeln!(s, "@group(1) @binding({binding}) var height_sampler: sampler;").unwrap();
        binding += 1;
    }
    let _ = binding;
}

/// Generate the vertex output struct.
fn generate_vertex_output_struct(s: &mut String, features: MaterialFeatures) {
    s.push_str(WGSL_VERTEX_OUTPUT_BASE);

    let mut loc = 3u32;

    if features.contains(MaterialFeatures::HAS_NORMAL_MAP) {
        writeln!(s, "    @location({loc}) world_tangent: vec3<f32>,").unwrap();
        loc += 1;
        writeln!(s, "    @location({loc}) world_bitangent: vec3<f32>,").unwrap();
        loc += 1;
    }
    if features.contains(MaterialFeatures::HAS_UV1) {
        writeln!(s, "    @location({loc}) uv1: vec2<f32>,").unwrap();
        loc += 1;
    }
    if features.contains(MaterialFeatures::HAS_VERTEX_COLORS) {
        writeln!(s, "    @location({loc}) vertex_color: vec4<f32>,").unwrap();
        loc += 1;
    }
    if features.contains(MaterialFeatures::RECEIVE_SHADOWS) {
        writeln!(s, "    @location({loc}) view_depth: f32,").unwrap();
        loc += 1;
    }
    let _ = loc;

    writeln!(s, "}};").unwrap();
}

/// Generate the vertex input struct.
fn generate_vertex_input_struct(s: &mut String, features: MaterialFeatures) {
    writeln!(s).unwrap();
    writeln!(s, "struct VertexInput {{").unwrap();
    writeln!(s, "    @location(0) position: vec3<f32>,").unwrap();
    writeln!(s, "    @location(1) normal: vec3<f32>,").unwrap();
    writeln!(s, "    @location(2) uv0: vec2<f32>,").unwrap();

    let mut loc = 3u32;
    if features.contains(MaterialFeatures::HAS_NORMAL_MAP) {
        writeln!(s, "    @location({loc}) tangent: vec4<f32>,").unwrap();
        loc += 1;
    }
    if features.contains(MaterialFeatures::HAS_UV1) {
        writeln!(s, "    @location({loc}) uv1: vec2<f32>,").unwrap();
        loc += 1;
    }
    if features.contains(MaterialFeatures::HAS_VERTEX_COLORS) {
        writeln!(s, "    @location({loc}) color: vec4<f32>,").unwrap();
        loc += 1;
    }
    if features.contains(MaterialFeatures::SKINNED) {
        writeln!(s, "    @location({loc}) joint_indices: vec4<u32>,").unwrap();
        loc += 1;
        writeln!(s, "    @location({loc}) joint_weights: vec4<f32>,").unwrap();
        loc += 1;
    }
    let _ = loc;

    writeln!(s, "}};").unwrap();
}

/// Generate the vertex shader entry point.
fn generate_vertex_shader(s: &mut String, features: MaterialFeatures) {
    writeln!(s).unwrap();
    writeln!(s, "// ---------------------------------------------------------------------------").unwrap();
    writeln!(s, "// Vertex Shader").unwrap();
    writeln!(s, "// ---------------------------------------------------------------------------").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "@vertex").unwrap();
    writeln!(s, "fn vs_main(input: VertexInput) -> VertexOutput {{").unwrap();
    writeln!(s, "    var out: VertexOutput;").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "    let world_pos = model.model * vec4<f32>(input.position, 1.0);").unwrap();
    writeln!(s, "    out.world_position = world_pos.xyz;").unwrap();
    writeln!(s, "    out.clip_position = scene.view_projection * world_pos;").unwrap();
    writeln!(s, "    out.world_normal = normalize(model.normal_matrix * input.normal);").unwrap();
    writeln!(s, "    out.uv0 = input.uv0;").unwrap();

    if features.contains(MaterialFeatures::HAS_NORMAL_MAP) {
        writeln!(s).unwrap();
        writeln!(s, "    // TBN vectors for normal mapping").unwrap();
        writeln!(s, "    let tangent = normalize(model.normal_matrix * input.tangent.xyz);").unwrap();
        writeln!(s, "    let bitangent = cross(out.world_normal, tangent) * input.tangent.w;").unwrap();
        writeln!(s, "    out.world_tangent = tangent;").unwrap();
        writeln!(s, "    out.world_bitangent = bitangent;").unwrap();
    }

    if features.contains(MaterialFeatures::HAS_UV1) {
        writeln!(s, "    out.uv1 = input.uv1;").unwrap();
    }
    if features.contains(MaterialFeatures::HAS_VERTEX_COLORS) {
        writeln!(s, "    out.vertex_color = input.color;").unwrap();
    }
    if features.contains(MaterialFeatures::RECEIVE_SHADOWS) {
        writeln!(s, "    // View-space depth for cascade selection").unwrap();
        writeln!(s, "    let view_pos = scene.view * world_pos;").unwrap();
        writeln!(s, "    out.view_depth = -view_pos.z;").unwrap();
    }

    writeln!(s).unwrap();
    writeln!(s, "    return out;").unwrap();
    writeln!(s, "}}").unwrap();
}

/// Generate the fragment shader entry point.
fn generate_fragment_shader(s: &mut String, features: MaterialFeatures) {
    writeln!(s).unwrap();
    writeln!(s, "// ---------------------------------------------------------------------------").unwrap();
    writeln!(s, "// Fragment Shader").unwrap();
    writeln!(s, "// ---------------------------------------------------------------------------").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "@fragment").unwrap();
    writeln!(s, "fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{").unwrap();

    // --- Base colour ---
    writeln!(s).unwrap();
    writeln!(s, "    // Base colour").unwrap();
    if features.contains(MaterialFeatures::HAS_ALBEDO_MAP) {
        writeln!(s, "    var base_color = textureSample(albedo_texture, albedo_sampler, in.uv0) * material.albedo_color;").unwrap();
    } else {
        writeln!(s, "    var base_color = material.albedo_color;").unwrap();
    }
    if features.contains(MaterialFeatures::HAS_VERTEX_COLORS) {
        writeln!(s, "    base_color *= in.vertex_color;").unwrap();
    }
    if features.contains(MaterialFeatures::HAS_DETAIL_ALBEDO) {
        writeln!(s, "    // Detail albedo overlay (multiply blend)").unwrap();
        writeln!(s, "    // let detail_albedo = textureSample(detail_albedo_texture, detail_albedo_sampler, in.uv0 * 4.0);").unwrap();
        writeln!(s, "    // base_color = vec4<f32>(base_color.rgb * detail_albedo.rgb, base_color.a);").unwrap();
    }

    // --- Alpha test ---
    if features.contains(MaterialFeatures::ALPHA_MASK) {
        writeln!(s).unwrap();
        writeln!(s, "    // Alpha masking").unwrap();
        writeln!(s, "    if base_color.a < material.ao_alpha_clearcoat.y {{").unwrap();
        writeln!(s, "        discard;").unwrap();
        writeln!(s, "    }}").unwrap();
    }

    // --- Metallic / roughness ---
    writeln!(s).unwrap();
    writeln!(s, "    // Metallic-roughness").unwrap();
    writeln!(s, "    var metallic = material.metallic_roughness.x;").unwrap();
    writeln!(s, "    var roughness = material.metallic_roughness.y;").unwrap();
    writeln!(s, "    let reflectance = material.metallic_roughness.z;").unwrap();
    if features.contains(MaterialFeatures::HAS_METALLIC_ROUGHNESS_MAP) {
        writeln!(s, "    let mr_sample = textureSample(metallic_roughness_texture, metallic_roughness_sampler, in.uv0);").unwrap();
        writeln!(s, "    roughness *= mr_sample.g;").unwrap();
        writeln!(s, "    metallic *= mr_sample.b;").unwrap();
    }
    writeln!(s, "    roughness = clamp(roughness, 0.04, 1.0);").unwrap();

    // --- Normal ---
    writeln!(s).unwrap();
    writeln!(s, "    // Surface normal").unwrap();
    if features.contains(MaterialFeatures::HAS_NORMAL_MAP) {
        writeln!(s, "    let normal_sample = textureSample(normal_texture, normal_sampler, in.uv0).xyz;").unwrap();
        writeln!(s, "    let N = perturb_normal(normal_sample, in.world_normal, in.world_tangent, in.world_bitangent, material.metallic_roughness.w);").unwrap();
    } else if features.contains(MaterialFeatures::DOUBLE_SIDED) {
        writeln!(s, "    var N = normalize(in.world_normal);").unwrap();
        writeln!(s, "    if !in.front_facing {{ N = -N; }}").unwrap();
    } else {
        writeln!(s, "    let N = normalize(in.world_normal);").unwrap();
    }

    // --- View vector ---
    writeln!(s).unwrap();
    writeln!(s, "    // View direction").unwrap();
    writeln!(s, "    let V = normalize(scene.camera_position - in.world_position);").unwrap();
    writeln!(s, "    let n_dot_v = max(dot(N, V), 0.0001);").unwrap();

    // --- F0 ---
    writeln!(s).unwrap();
    writeln!(s, "    // Compute F0").unwrap();
    writeln!(s, "    let albedo = base_color.rgb;").unwrap();
    writeln!(s, "    let f0 = compute_f0(albedo, metallic, reflectance);").unwrap();
    writeln!(s, "    let diffuse_color = albedo * (1.0 - metallic);").unwrap();

    // --- Lighting loop ---
    writeln!(s).unwrap();
    writeln!(s, "    // Direct lighting").unwrap();
    writeln!(s, "    var lo = vec3<f32>(0.0);").unwrap();
    writeln!(s, "    let light_count = light_buffer.count.x;").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "    for (var i = 0u; i < light_count; i++) {{").unwrap();
    writeln!(s, "        let light = light_buffer.lights[i];").unwrap();
    writeln!(s, "        let light_type = u32(light.position_type.w);").unwrap();
    writeln!(s, "        let light_color = light.color_intensity.rgb * light.color_intensity.a;").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "        var L: vec3<f32>;").unwrap();
    writeln!(s, "        var attenuation = 1.0;").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "        if light_type == 0u {{").unwrap();
    writeln!(s, "            // Directional light").unwrap();
    writeln!(s, "            L = normalize(-light.position_type.xyz);").unwrap();
    writeln!(s, "        }} else if light_type == 1u {{").unwrap();
    writeln!(s, "            // Point light").unwrap();
    writeln!(s, "            let to_light = light.position_type.xyz - in.world_position;").unwrap();
    writeln!(s, "            let dist = length(to_light);").unwrap();
    writeln!(s, "            L = to_light / max(dist, 0.0001);").unwrap();
    writeln!(s, "            attenuation = light_attenuation(dist, light.direction_range.w);").unwrap();
    writeln!(s, "        }} else {{").unwrap();
    writeln!(s, "            // Spot light").unwrap();
    writeln!(s, "            let to_light = light.position_type.xyz - in.world_position;").unwrap();
    writeln!(s, "            let dist = length(to_light);").unwrap();
    writeln!(s, "            L = to_light / max(dist, 0.0001);").unwrap();
    writeln!(s, "            attenuation = light_attenuation(dist, light.direction_range.w);").unwrap();
    writeln!(s, "            let cos_angle = dot(-L, normalize(light.direction_range.xyz));").unwrap();
    writeln!(s, "            attenuation *= spot_falloff(cos_angle, light.spot_params.x, light.spot_params.y);").unwrap();
    writeln!(s, "        }}").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "        let H = normalize(V + L);").unwrap();
    writeln!(s, "        let n_dot_l = max(dot(N, L), 0.0);").unwrap();
    writeln!(s, "        let n_dot_h = max(dot(N, H), 0.0);").unwrap();
    writeln!(s, "        let v_dot_h = max(dot(V, H), 0.0);").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "        // Cook-Torrance specular BRDF").unwrap();
    writeln!(s, "        let D = distribution_ggx(n_dot_h, roughness);").unwrap();
    writeln!(s, "        let G = geometry_smith(n_dot_v, n_dot_l, roughness);").unwrap();
    writeln!(s, "        let F = fresnel_schlick(v_dot_h, f0);").unwrap();
    writeln!(s, "        let specular = D * G * F;").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "        // Energy-conserving diffuse").unwrap();
    writeln!(s, "        let k_d = (vec3<f32>(1.0) - F) * (1.0 - metallic);").unwrap();
    writeln!(s, "        let diffuse = diffuse_lambertian(diffuse_color);").unwrap();

    // --- Clearcoat ---
    if features.contains(MaterialFeatures::HAS_CLEARCOAT) {
        writeln!(s).unwrap();
        writeln!(s, "        // Clearcoat lobe").unwrap();
        writeln!(s, "        let l_dot_h = max(dot(L, H), 0.0);").unwrap();
        writeln!(s, "        let cc = evaluate_clearcoat(n_dot_h, l_dot_h, material.ao_alpha_clearcoat.z, material.ao_alpha_clearcoat.w);").unwrap();
        writeln!(s, "        let cc_specular = cc.x;").unwrap();
        writeln!(s, "        let cc_attenuation = cc.y;").unwrap();
        writeln!(s, "        lo += (k_d * diffuse * cc_attenuation + specular * cc_attenuation + cc_specular) * light_color * attenuation * n_dot_l;").unwrap();
    } else {
        writeln!(s).unwrap();
        writeln!(s, "        lo += (k_d * diffuse + specular) * light_color * attenuation * n_dot_l;").unwrap();
    }

    writeln!(s, "    }}").unwrap();

    // --- Shadows ---
    if features.contains(MaterialFeatures::RECEIVE_SHADOWS) {
        writeln!(s).unwrap();
        writeln!(s, "    // Shadow").unwrap();
        writeln!(s, "    let cascade = select_cascade(in.view_depth);").unwrap();
        writeln!(s, "    let shadow = sample_shadow_pcf(in.world_position, cascade);").unwrap();
        writeln!(s, "    lo *= shadow;").unwrap();
    }

    // --- IBL ---
    if features.contains(MaterialFeatures::USE_IBL) {
        writeln!(s).unwrap();
        writeln!(s, "    // Image-based lighting (IBL)").unwrap();
        writeln!(s, "    let F_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);").unwrap();
        writeln!(s, "    let k_d_ibl = (vec3<f32>(1.0) - F_ibl) * (1.0 - metallic);").unwrap();
        writeln!(s).unwrap();
        writeln!(s, "    // Diffuse IBL").unwrap();
        writeln!(s, "    let irradiance = textureSample(irradiance_map, env_sampler, N).rgb;").unwrap();
        writeln!(s, "    let diffuse_ibl = k_d_ibl * irradiance * diffuse_color;").unwrap();
        writeln!(s).unwrap();
        writeln!(s, "    // Specular IBL").unwrap();
        writeln!(s, "    let R = reflect(-V, N);").unwrap();
        writeln!(s, "    let max_mip = f32(textureNumLevels(prefilter_map) - 1u);").unwrap();
        writeln!(s, "    let prefiltered = textureSampleLevel(prefilter_map, env_sampler, R, roughness * max_mip).rgb;").unwrap();
        writeln!(s, "    let brdf = textureSample(brdf_lut, env_sampler, vec2<f32>(n_dot_v, roughness)).rg;").unwrap();
        writeln!(s, "    let specular_ibl = prefiltered * (F_ibl * brdf.x + brdf.y);").unwrap();
        writeln!(s).unwrap();
        writeln!(s, "    lo += diffuse_ibl + specular_ibl;").unwrap();
    } else {
        // Basic ambient.
        writeln!(s).unwrap();
        writeln!(s, "    // Simple ambient term (no IBL)").unwrap();
        writeln!(s, "    let ambient = vec3<f32>(0.03) * diffuse_color;").unwrap();
        writeln!(s, "    lo += ambient;").unwrap();
    }

    // --- AO ---
    if features.contains(MaterialFeatures::HAS_AO_MAP) {
        writeln!(s).unwrap();
        writeln!(s, "    // Ambient occlusion").unwrap();
        writeln!(s, "    let ao_sample = textureSample(ao_texture, ao_sampler, in.uv0).r;").unwrap();
        writeln!(s, "    let ao = mix(1.0, ao_sample, material.ao_alpha_clearcoat.x);").unwrap();
        writeln!(s, "    lo *= ao;").unwrap();
    }

    // --- Emissive ---
    writeln!(s).unwrap();
    writeln!(s, "    // Emissive").unwrap();
    if features.contains(MaterialFeatures::HAS_EMISSIVE_MAP) {
        writeln!(s, "    let emissive_sample = textureSample(emissive_texture, emissive_sampler, in.uv0).rgb;").unwrap();
        writeln!(s, "    lo += emissive_sample * material.emissive.rgb * material.emissive.a;").unwrap();
    } else {
        writeln!(s, "    lo += material.emissive.rgb * material.emissive.a;").unwrap();
    }

    // --- Final output ---
    writeln!(s).unwrap();
    writeln!(s, "    // Tone mapping and output").unwrap();
    writeln!(s, "    let mapped = tone_map_aces(lo);").unwrap();

    if features.contains(MaterialFeatures::ALPHA_BLEND) {
        writeln!(s, "    return vec4<f32>(mapped, base_color.a);").unwrap();
    } else {
        writeln!(s, "    return vec4<f32>(mapped, 1.0);").unwrap();
    }

    writeln!(s, "}}").unwrap();
}

// ---------------------------------------------------------------------------
// Depth-only shader (for shadow maps)
// ---------------------------------------------------------------------------

/// Generate a depth-only vertex shader for shadow map rendering.
pub fn generate_depth_only_shader(skinned: bool) -> String {
    let mut s = String::with_capacity(1024);

    writeln!(s, "// Auto-generated depth-only shader for shadow maps").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "struct DepthUniforms {{").unwrap();
    writeln!(s, "    light_view_projection: mat4x4<f32>,").unwrap();
    writeln!(s, "}};").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "@group(0) @binding(0) var<uniform> depth_uniforms: DepthUniforms;").unwrap();
    writeln!(s).unwrap();

    writeln!(s, "struct ModelUniforms {{").unwrap();
    writeln!(s, "    model: mat4x4<f32>,").unwrap();
    writeln!(s, "    normal_matrix: mat3x3<f32>,").unwrap();
    writeln!(s, "}};").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "@group(1) @binding(0) var<uniform> model: ModelUniforms;").unwrap();
    writeln!(s).unwrap();

    writeln!(s, "struct VertexInput {{").unwrap();
    writeln!(s, "    @location(0) position: vec3<f32>,").unwrap();
    if skinned {
        writeln!(s, "    @location(1) joint_indices: vec4<u32>,").unwrap();
        writeln!(s, "    @location(2) joint_weights: vec4<f32>,").unwrap();
    }
    writeln!(s, "}};").unwrap();
    writeln!(s).unwrap();

    writeln!(s, "@vertex").unwrap();
    writeln!(s, "fn vs_main(input: VertexInput) -> @builtin(position) vec4<f32> {{").unwrap();
    writeln!(s, "    let world_pos = model.model * vec4<f32>(input.position, 1.0);").unwrap();
    writeln!(s, "    return depth_uniforms.light_view_projection * world_pos;").unwrap();
    writeln!(s, "}}").unwrap();

    s
}

/// Generate a full-screen triangle shader for post-processing passes.
pub fn generate_fullscreen_triangle_shader() -> String {
    let mut s = String::with_capacity(512);

    writeln!(s, "// Auto-generated full-screen triangle shader").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "struct VertexOutput {{").unwrap();
    writeln!(s, "    @builtin(position) position: vec4<f32>,").unwrap();
    writeln!(s, "    @location(0) uv: vec2<f32>,").unwrap();
    writeln!(s, "}};").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "@vertex").unwrap();
    writeln!(s, "fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {{").unwrap();
    writeln!(s, "    var out: VertexOutput;").unwrap();
    writeln!(s, "    // Full-screen triangle trick: 3 vertices, no buffer needed").unwrap();
    writeln!(s, "    let x = f32(i32(vertex_index) / 2) * 4.0 - 1.0;").unwrap();
    writeln!(s, "    let y = f32(i32(vertex_index) % 2) * 4.0 - 1.0;").unwrap();
    writeln!(s, "    out.position = vec4<f32>(x, y, 0.0, 1.0);").unwrap();
    writeln!(s, "    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);").unwrap();
    writeln!(s, "    return out;").unwrap();
    writeln!(s, "}}").unwrap();

    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_shader_generation() {
        let shader = generate_pbr_shader(MaterialFeatures::empty());
        assert!(shader.contains("fn vs_main"));
        assert!(shader.contains("fn fs_main"));
        assert!(shader.contains("SceneUniforms"));
        assert!(shader.contains("MaterialUniforms"));
    }

    #[test]
    fn shader_with_normal_map() {
        let features = MaterialFeatures::HAS_NORMAL_MAP;
        let shader = generate_pbr_shader(features);
        assert!(shader.contains("normal_texture"));
        assert!(shader.contains("perturb_normal"));
        assert!(shader.contains("world_tangent"));
    }

    #[test]
    fn shader_with_shadows() {
        let features = MaterialFeatures::RECEIVE_SHADOWS;
        let shader = generate_pbr_shader(features);
        assert!(shader.contains("shadow_map"));
        assert!(shader.contains("sample_shadow_pcf"));
        assert!(shader.contains("select_cascade"));
    }

    #[test]
    fn shader_with_ibl() {
        let features = MaterialFeatures::USE_IBL;
        let shader = generate_pbr_shader(features);
        assert!(shader.contains("irradiance_map"));
        assert!(shader.contains("prefilter_map"));
        assert!(shader.contains("brdf_lut"));
    }

    #[test]
    fn full_pbr_shader() {
        let features = MaterialFeatures::full_pbr();
        let shader = generate_pbr_shader(features);
        // Should contain all major sections.
        assert!(shader.contains("distribution_ggx"));
        assert!(shader.contains("geometry_smith"));
        assert!(shader.contains("fresnel_schlick"));
        assert!(shader.contains("compute_f0"));
        assert!(shader.len() > 3000); // Should be substantial
    }

    #[test]
    fn depth_only_shader() {
        let shader = generate_depth_only_shader(false);
        assert!(shader.contains("light_view_projection"));
        assert!(shader.contains("fn vs_main"));
    }

    #[test]
    fn feature_cache_key_is_unique() {
        let a = MaterialFeatures::HAS_ALBEDO_MAP;
        let b = MaterialFeatures::HAS_NORMAL_MAP;
        assert_ne!(a.cache_key(), b.cache_key());
    }
}
