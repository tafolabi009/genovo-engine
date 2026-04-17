// engine/terrain/src/terrain_renderer_gpu.rs
//
// GPU terrain rendering for the Genovo engine.
//
// Heightmap texture, clipmap vertex buffer, tessellation parameters,
// and material splatting in shader.

use glam::{Mat4, Vec2, Vec3, Vec4};

pub const MAX_TERRAIN_LAYERS: usize = 8;
pub const DEFAULT_TILE_SIZE: u32 = 256;

pub const TERRAIN_VS_WGSL: &str = r#"
struct TerrainUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    terrain_scale: f32,
    heightmap_size: vec2<f32>,
    texel_size: vec2<f32>,
    lod_distances: vec4<f32>,
    terrain_offset: vec3<f32>,
    height_scale: f32,
};
@group(0) @binding(0) var<uniform> terrain: TerrainUniforms;
@group(0) @binding(1) var heightmap: texture_2d<f32>;
@group(0) @binding(2) var height_sampler: sampler;

struct VertexInput { @location(0) grid_pos: vec2<f32>, @location(1) lod: f32 };
struct VertexOutput { @builtin(position) clip_pos: vec4<f32>, @location(0) world_pos: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) normal: vec3<f32> };

@vertex fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let uv = input.grid_pos / terrain.heightmap_size;
    let height = textureSampleLevel(heightmap, height_sampler, uv, 0.0).r * terrain.height_scale;
    let world_pos = vec3<f32>(input.grid_pos.x * terrain.terrain_scale + terrain.terrain_offset.x, height + terrain.terrain_offset.y, input.grid_pos.y * terrain.terrain_scale + terrain.terrain_offset.z);

    let ts = terrain.texel_size;
    let h_l = textureSampleLevel(heightmap, height_sampler, uv + vec2<f32>(-ts.x, 0.0), 0.0).r * terrain.height_scale;
    let h_r = textureSampleLevel(heightmap, height_sampler, uv + vec2<f32>(ts.x, 0.0), 0.0).r * terrain.height_scale;
    let h_d = textureSampleLevel(heightmap, height_sampler, uv + vec2<f32>(0.0, -ts.y), 0.0).r * terrain.height_scale;
    let h_u = textureSampleLevel(heightmap, height_sampler, uv + vec2<f32>(0.0, ts.y), 0.0).r * terrain.height_scale;
    let normal = normalize(vec3<f32>(h_l - h_r, 2.0 * terrain.terrain_scale, h_d - h_u));

    out.clip_pos = terrain.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.uv = uv;
    out.normal = normal;
    return out;
}
"#;

pub const TERRAIN_FS_WGSL: &str = r#"
struct MaterialParams {
    layer_scales: array<vec4<f32>, 2>,
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    ambient_color: vec3<f32>,
    fog_density: f32,
    fog_color: vec3<f32>,
    fog_start: f32,
    camera_pos: vec3<f32>,
    _pad: f32,
};
@group(1) @binding(0) var<uniform> material: MaterialParams;
@group(1) @binding(1) var splatmap: texture_2d<f32>;
@group(1) @binding(2) var splat_sampler: sampler;
@group(1) @binding(3) var layer0_albedo: texture_2d<f32>;
@group(1) @binding(4) var layer1_albedo: texture_2d<f32>;
@group(1) @binding(5) var layer2_albedo: texture_2d<f32>;
@group(1) @binding(6) var layer3_albedo: texture_2d<f32>;
@group(1) @binding(7) var layer_sampler: sampler;

struct FragInput { @location(0) world_pos: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) normal: vec3<f32> };

@fragment fn fs_main(input: FragInput) -> @location(0) vec4<f32> {
    let splat = textureSample(splatmap, splat_sampler, input.uv);
    let scale0 = material.layer_scales[0].x; let scale1 = material.layer_scales[0].y;
    let scale2 = material.layer_scales[0].z; let scale3 = material.layer_scales[0].w;
    let tiled_uv0 = input.uv * scale0; let tiled_uv1 = input.uv * scale1;
    let tiled_uv2 = input.uv * scale2; let tiled_uv3 = input.uv * scale3;
    let c0 = textureSample(layer0_albedo, layer_sampler, tiled_uv0).rgb;
    let c1 = textureSample(layer1_albedo, layer_sampler, tiled_uv1).rgb;
    let c2 = textureSample(layer2_albedo, layer_sampler, tiled_uv2).rgb;
    let c3 = textureSample(layer3_albedo, layer_sampler, tiled_uv3).rgb;
    let albedo = c0 * splat.r + c1 * splat.g + c2 * splat.b + c3 * splat.a;
    let n = normalize(input.normal);
    let ndotl = max(dot(n, normalize(material.sun_direction)), 0.0);
    let diffuse = albedo * (material.ambient_color + vec3<f32>(ndotl) * material.sun_intensity);
    let dist = length(input.world_pos - material.camera_pos);
    let fog = 1.0 - exp(-material.fog_density * max(dist - material.fog_start, 0.0));
    let color = mix(diffuse, material.fog_color, clamp(fog, 0.0, 1.0));
    return vec4<f32>(color, 1.0);
}
"#;

#[derive(Debug, Clone)]
pub struct TerrainGpuConfig {
    pub heightmap_size: u32,
    pub terrain_scale: f32,
    pub height_scale: f32,
    pub offset: Vec3,
    pub layer_count: usize,
    pub layer_scales: [f32; MAX_TERRAIN_LAYERS],
    pub lod_distances: [f32; 4],
    pub fog_density: f32,
    pub fog_color: Vec3,
    pub fog_start: f32,
    pub sun_direction: Vec3,
    pub sun_intensity: f32,
    pub ambient_color: Vec3,
    pub wireframe: bool,
}

impl Default for TerrainGpuConfig {
    fn default() -> Self {
        Self { heightmap_size: DEFAULT_TILE_SIZE, terrain_scale: 1.0, height_scale: 50.0, offset: Vec3::ZERO,
            layer_count: 4, layer_scales: [32.0, 32.0, 64.0, 16.0, 32.0, 32.0, 32.0, 32.0],
            lod_distances: [50.0, 100.0, 200.0, 400.0], fog_density: 0.002, fog_color: Vec3::new(0.7, 0.8, 0.9),
            fog_start: 100.0, sun_direction: Vec3::new(0.5, 0.8, 0.3).normalize(), sun_intensity: 1.2,
            ambient_color: Vec3::new(0.15, 0.18, 0.25), wireframe: false }
    }
}

#[derive(Debug, Clone)]
pub struct ClipmapRing { pub lod: u32, pub grid_size: u32, pub cell_size: f32, pub origin: Vec2, pub vertex_count: u32, pub index_count: u32 }

impl ClipmapRing {
    pub fn new(lod: u32, grid_size: u32, base_cell_size: f32) -> Self {
        let cell_size = base_cell_size * (1u32 << lod) as f32;
        let vc = (grid_size + 1) * (grid_size + 1);
        let ic = grid_size * grid_size * 6;
        Self { lod, grid_size, cell_size, origin: Vec2::ZERO, vertex_count: vc, index_count: ic }
    }

    pub fn update_origin(&mut self, camera_x: f32, camera_z: f32) {
        self.origin.x = (camera_x / self.cell_size).floor() * self.cell_size;
        self.origin.y = (camera_z / self.cell_size).floor() * self.cell_size;
    }

    pub fn generate_vertices(&self) -> Vec<[f32; 3]> {
        let mut verts = Vec::with_capacity(self.vertex_count as usize);
        let half = self.grid_size as f32 * 0.5;
        for z in 0..=self.grid_size {
            for x in 0..=self.grid_size {
                let wx = self.origin.x + (x as f32 - half) * self.cell_size;
                let wz = self.origin.y + (z as f32 - half) * self.cell_size;
                verts.push([wx, 0.0, wz]);
            }
        }
        verts
    }

    pub fn generate_indices(&self) -> Vec<u32> {
        let mut indices = Vec::with_capacity(self.index_count as usize);
        let stride = self.grid_size + 1;
        for z in 0..self.grid_size {
            for x in 0..self.grid_size {
                let tl = z * stride + x;
                let tr = tl + 1;
                let bl = (z + 1) * stride + x;
                let br = bl + 1;
                indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
            }
        }
        indices
    }
}

#[derive(Debug)]
pub struct TerrainGpuRenderer {
    pub config: TerrainGpuConfig,
    pub clipmap_rings: Vec<ClipmapRing>,
    pub total_vertices: u32,
    pub total_indices: u32,
}

impl TerrainGpuRenderer {
    pub fn new(config: TerrainGpuConfig, lod_count: u32, grid_size: u32) -> Self {
        let mut rings = Vec::new();
        let mut tv = 0; let mut ti = 0;
        for lod in 0..lod_count { let r = ClipmapRing::new(lod, grid_size, config.terrain_scale); tv += r.vertex_count; ti += r.index_count; rings.push(r); }
        Self { config, clipmap_rings: rings, total_vertices: tv, total_indices: ti }
    }

    pub fn update(&mut self, camera_pos: Vec3) { for ring in &mut self.clipmap_rings { ring.update_origin(camera_pos.x, camera_pos.z); } }

    pub fn vertex_shader_source(&self) -> &'static str { TERRAIN_VS_WGSL }
    pub fn fragment_shader_source(&self) -> &'static str { TERRAIN_FS_WGSL }

    pub fn gpu_uniforms(&self, view_proj: &Mat4, camera_pos: Vec3) -> TerrainGpuUniforms {
        let hs = self.config.heightmap_size as f32;
        TerrainGpuUniforms { view_proj: *view_proj, camera_pos, terrain_scale: self.config.terrain_scale, heightmap_size: Vec2::new(hs, hs), texel_size: Vec2::new(1.0/hs, 1.0/hs), lod_distances: Vec4::from_array(self.config.lod_distances), terrain_offset: self.config.offset, height_scale: self.config.height_scale }
    }
}

#[derive(Debug, Clone, Copy)] #[repr(C)]
pub struct TerrainGpuUniforms { pub view_proj: Mat4, pub camera_pos: Vec3, pub terrain_scale: f32, pub heightmap_size: Vec2, pub texel_size: Vec2, pub lod_distances: Vec4, pub terrain_offset: Vec3, pub height_scale: f32 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_clipmap_ring() { let r = ClipmapRing::new(0, 32, 1.0); assert_eq!(r.vertex_count, 33*33); }
    #[test] fn test_renderer() { let r = TerrainGpuRenderer::new(TerrainGpuConfig::default(), 4, 32); assert_eq!(r.clipmap_rings.len(), 4); }
}
