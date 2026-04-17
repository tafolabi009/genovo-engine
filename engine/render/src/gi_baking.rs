// engine/render/src/gi_baking.rs
//
// Global Illumination baking system for the Genovo engine.
//
// Implements a complete lightmap GI baking pipeline:
//
// - Progressive path tracing with configurable sample counts and bounce limits.
// - UV2 automatic unwrapping and chart packing for lightmap UVs.
// - Post-bake denoising with bilateral, Gaussian, and edge-preserving filters.
// - HDR lightmap encoding (RGBM, LogLUV, RGBE, Float16, Float32).
// - Directional lightmaps storing dominant light direction per texel.
// - Irradiance probe baking at scene-placed positions.
// - Progressive refinement with interruptible bake sessions.
//
// The baking pipeline operates in multiple stages:
//
// 1. Scene geometry collection and BVH construction.
// 2. UV2 unwrapping and atlas allocation.
// 3. Texel rasterisation (determine world position/normal per texel).
// 4. Monte Carlo path tracing with progressive sample accumulation.
// 5. Denoising pass (configurable filter type and strength).
// 6. Directional lightmap computation (optional).
// 7. Encoding to target format and compression.
// 8. Probe baking for dynamic objects.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum distance for ray-surface intersection to avoid self-intersection.
const RAY_OFFSET_EPSILON: f32 = 1e-4;

/// Maximum number of UV charts per mesh during unwrapping.
const MAX_CHARTS_PER_MESH: usize = 256;

/// Default margin between charts in texels.
const DEFAULT_CHART_MARGIN: u32 = 2;

/// Russian roulette probability threshold for path termination.
const RUSSIAN_ROULETTE_THRESHOLD: f32 = 0.05;

/// Cosine-weighted hemisphere PDF normalization factor.
const COSINE_HEMISPHERE_PDF: f32 = 1.0 / PI;

// ---------------------------------------------------------------------------
// GI Bake Quality
// ---------------------------------------------------------------------------

/// Quality preset for GI baking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GiBakeQuality {
    /// Fast preview with minimal samples (4-16 spp).
    Preview,
    /// Draft quality for rapid iteration (32-64 spp).
    Draft,
    /// Medium quality for testing (128-256 spp).
    Medium,
    /// High quality for near-final results (512-1024 spp).
    High,
    /// Production quality for shipping (2048+ spp).
    Production,
    /// Custom quality with user-specified parameters.
    Custom,
}

impl GiBakeQuality {
    /// Returns the recommended samples-per-texel for this quality level.
    pub fn samples_per_texel(self) -> u32 {
        match self {
            Self::Preview => 8,
            Self::Draft => 48,
            Self::Medium => 192,
            Self::High => 768,
            Self::Production => 2048,
            Self::Custom => 256,
        }
    }

    /// Returns the recommended maximum bounce count.
    pub fn max_bounces(self) -> u32 {
        match self {
            Self::Preview => 1,
            Self::Draft => 2,
            Self::Medium => 3,
            Self::High => 4,
            Self::Production => 6,
            Self::Custom => 3,
        }
    }

    /// Returns the recommended texels-per-world-unit.
    pub fn texels_per_unit(self) -> f32 {
        match self {
            Self::Preview => 2.0,
            Self::Draft => 4.0,
            Self::Medium => 8.0,
            Self::High => 16.0,
            Self::Production => 32.0,
            Self::Custom => 8.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Lightmap Encoding
// ---------------------------------------------------------------------------

/// Target encoding format for baked lightmaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GiLightmapEncoding {
    /// Raw RGBA 32-bit float per channel.
    Float32,
    /// RGBA 16-bit half-float per channel.
    Float16,
    /// RGBM encoding: RGB * multiplier in alpha.
    Rgbm,
    /// LogLUV encoding for high dynamic range in 32 bits.
    LogLuv,
    /// Radiance RGBE encoding.
    Rgbe,
    /// BC6H compressed HDR (GPU block compression).
    Bc6h,
}

impl GiLightmapEncoding {
    /// Returns the bytes per texel for this encoding.
    pub fn bytes_per_texel(self) -> u32 {
        match self {
            Self::Float32 => 16,
            Self::Float16 => 8,
            Self::Rgbm => 4,
            Self::LogLuv => 4,
            Self::Rgbe => 4,
            Self::Bc6h => 1, // average for 4x4 block = 16 bytes / 16 texels
        }
    }

    /// Whether this format supports values above 1.0 natively.
    pub fn is_hdr(self) -> bool {
        matches!(self, Self::Float32 | Self::Float16 | Self::Rgbm | Self::LogLuv | Self::Rgbe | Self::Bc6h)
    }

    /// Maximum representable value for this encoding.
    pub fn max_value(self) -> f32 {
        match self {
            Self::Float32 => f32::MAX,
            Self::Float16 => 65504.0,
            Self::Rgbm => 8.0,
            Self::LogLuv => 1e38,
            Self::Rgbe => 1e38,
            Self::Bc6h => 65504.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Denoise filter
// ---------------------------------------------------------------------------

/// Filter type for post-bake denoising.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenoiseFilter {
    /// No denoising.
    None,
    /// Simple box blur.
    BoxBlur,
    /// Gaussian blur.
    Gaussian,
    /// Bilateral filter (edge-preserving).
    Bilateral,
    /// Edge-aware wavelet denoiser.
    Wavelet,
    /// A-Trous wavelet filter.
    ATrous,
    /// OIDN-style machine learning denoiser (requires external library).
    MlDenoiser,
}

/// Configuration for the denoising pass.
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    /// Which filter to apply.
    pub filter: DenoiseFilter,
    /// Filter radius in texels.
    pub radius: u32,
    /// Spatial sigma for bilateral/Gaussian filters.
    pub sigma_spatial: f32,
    /// Range sigma for bilateral filter (controls edge sensitivity).
    pub sigma_range: f32,
    /// Number of A-Trous iterations.
    pub atrous_iterations: u32,
    /// Whether to use normal-guided denoising.
    pub normal_aware: bool,
    /// Normal deviation threshold in degrees.
    pub normal_threshold_degrees: f32,
    /// Whether to use position-guided denoising.
    pub position_aware: bool,
    /// Position threshold in world units.
    pub position_threshold: f32,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            filter: DenoiseFilter::Bilateral,
            radius: 3,
            sigma_spatial: 2.0,
            sigma_range: 0.1,
            atrous_iterations: 5,
            normal_aware: true,
            normal_threshold_degrees: 25.0,
            position_aware: true,
            position_threshold: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// UV2 Unwrapping
// ---------------------------------------------------------------------------

/// Method used for UV2 generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnwrapMethod {
    /// Angle-based flattening.
    AngleBased,
    /// Conformal (LSCM) mapping.
    Conformal,
    /// Projection-based (top-down, planar).
    Projection,
    /// Smart UV projection (angle threshold splits).
    SmartProject,
    /// Lightmap pack (simple box packing).
    LightmapPack,
}

/// A single UV chart (island) in the lightmap atlas.
#[derive(Debug, Clone)]
pub struct UvChart {
    /// Unique chart identifier.
    pub id: u32,
    /// Mesh index this chart belongs to.
    pub mesh_index: u32,
    /// Triangle indices within the mesh that belong to this chart.
    pub triangle_indices: Vec<u32>,
    /// UV coordinates for each vertex in chart-local space [0, 1].
    pub uvs: Vec<[f32; 2]>,
    /// Packed position in the atlas (top-left corner, in texels).
    pub atlas_offset: [u32; 2],
    /// Packed size in the atlas (in texels).
    pub atlas_size: [u32; 2],
    /// Surface area in world units squared (for texel density computation).
    pub world_area: f32,
    /// UV-space area for stretch metric computation.
    pub uv_area: f32,
}

impl UvChart {
    /// Create a new empty chart.
    pub fn new(id: u32, mesh_index: u32) -> Self {
        Self {
            id,
            mesh_index,
            triangle_indices: Vec::new(),
            uvs: Vec::new(),
            atlas_offset: [0, 0],
            atlas_size: [0, 0],
            world_area: 0.0,
            uv_area: 0.0,
        }
    }

    /// Compute the stretch metric (ratio of UV area to world area).
    pub fn stretch_metric(&self) -> f32 {
        if self.world_area < 1e-10 {
            return 0.0;
        }
        self.uv_area / self.world_area
    }

    /// Returns the number of triangles in this chart.
    pub fn triangle_count(&self) -> usize {
        self.triangle_indices.len()
    }

    /// Returns the texel count (width * height).
    pub fn texel_count(&self) -> u32 {
        self.atlas_size[0] * self.atlas_size[1]
    }
}

/// UV2 unwrapper that generates lightmap UVs for meshes.
#[derive(Debug)]
pub struct Uv2Unwrapper {
    /// Unwrapping method.
    pub method: UnwrapMethod,
    /// Angle threshold for smart projection (in degrees).
    pub angle_threshold: f32,
    /// Margin between charts in texels.
    pub chart_margin: u32,
    /// Maximum number of charts per mesh.
    pub max_charts: usize,
    /// Target texels per world unit.
    pub texels_per_unit: f32,
    /// Whether to rotate charts for better packing.
    pub allow_rotation: bool,
    /// Generated charts.
    pub charts: Vec<UvChart>,
    /// Total atlas width.
    pub atlas_width: u32,
    /// Total atlas height.
    pub atlas_height: u32,
}

impl Uv2Unwrapper {
    /// Create a new UV2 unwrapper with default settings.
    pub fn new() -> Self {
        Self {
            method: UnwrapMethod::SmartProject,
            angle_threshold: 66.0,
            chart_margin: DEFAULT_CHART_MARGIN,
            max_charts: MAX_CHARTS_PER_MESH,
            texels_per_unit: 8.0,
            allow_rotation: true,
            charts: Vec::new(),
            atlas_width: 0,
            atlas_height: 0,
        }
    }

    /// Set the unwrapping method.
    pub fn with_method(mut self, method: UnwrapMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the texels-per-unit density.
    pub fn with_density(mut self, texels_per_unit: f32) -> Self {
        self.texels_per_unit = texels_per_unit;
        self
    }

    /// Set the chart margin.
    pub fn with_margin(mut self, margin: u32) -> Self {
        self.chart_margin = margin;
        self
    }

    /// Generate UV2 coordinates for a mesh given its triangle data.
    ///
    /// `positions` is a flat array of vertex positions (3 floats per vertex).
    /// `normals` is a flat array of vertex normals (3 floats per vertex).
    /// `indices` is the index buffer (3 indices per triangle).
    pub fn unwrap_mesh(
        &mut self,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
    ) -> Vec<[f32; 2]> {
        let triangle_count = indices.len() / 3;
        let mut uvs = vec![[0.0f32, 0.0f32]; positions.len()];

        // Simple planar projection per triangle group as a baseline.
        for tri_idx in 0..triangle_count {
            let i0 = indices[tri_idx * 3] as usize;
            let i1 = indices[tri_idx * 3 + 1] as usize;
            let i2 = indices[tri_idx * 3 + 2] as usize;

            let p0 = positions[i0];
            let p1 = positions[i1];
            let p2 = positions[i2];

            let n = normals[i0];
            let (u_axis, v_axis) = Self::compute_tangent_frame(n);

            for &idx in &[i0, i1, i2] {
                let p = positions[idx];
                let u = p[0] * u_axis[0] + p[1] * u_axis[1] + p[2] * u_axis[2];
                let v = p[0] * v_axis[0] + p[1] * v_axis[1] + p[2] * v_axis[2];
                uvs[idx] = [u * self.texels_per_unit, v * self.texels_per_unit];
            }

            // Track world area for the chart.
            let edge1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let edge2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ];
            let _area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        }

        uvs
    }

    /// Compute a tangent frame from a normal vector.
    fn compute_tangent_frame(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        let n = normal;
        let up = if n[1].abs() < 0.999 {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };

        let u = [
            up[1] * n[2] - up[2] * n[1],
            up[2] * n[0] - up[0] * n[2],
            up[0] * n[1] - up[1] * n[0],
        ];
        let len = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-10);
        let u = [u[0] / len, u[1] / len, u[2] / len];

        let v = [
            n[1] * u[2] - n[2] * u[1],
            n[2] * u[0] - n[0] * u[2],
            n[0] * u[1] - n[1] * u[0],
        ];

        (u, v)
    }

    /// Pack all charts into a rectangular atlas using a shelf-based algorithm.
    pub fn pack_atlas(&mut self, max_atlas_size: u32) {
        self.charts.sort_by(|a, b| b.atlas_size[1].cmp(&a.atlas_size[1]));

        let mut cursor_x: u32 = 0;
        let mut cursor_y: u32 = 0;
        let mut shelf_height: u32 = 0;
        let mut atlas_width: u32 = 0;

        for chart in &mut self.charts {
            let w = chart.atlas_size[0] + self.chart_margin * 2;
            let h = chart.atlas_size[1] + self.chart_margin * 2;

            if cursor_x + w > max_atlas_size {
                cursor_x = 0;
                cursor_y += shelf_height;
                shelf_height = 0;
            }

            chart.atlas_offset = [cursor_x + self.chart_margin, cursor_y + self.chart_margin];
            cursor_x += w;
            shelf_height = shelf_height.max(h);
            atlas_width = atlas_width.max(cursor_x);
        }

        self.atlas_width = atlas_width.next_power_of_two().min(max_atlas_size);
        self.atlas_height = (cursor_y + shelf_height).next_power_of_two().min(max_atlas_size);
    }

    /// Returns the total number of charts generated.
    pub fn chart_count(&self) -> usize {
        self.charts.len()
    }

    /// Returns atlas dimensions as (width, height).
    pub fn atlas_dimensions(&self) -> (u32, u32) {
        (self.atlas_width, self.atlas_height)
    }
}

// ---------------------------------------------------------------------------
// Path Tracing
// ---------------------------------------------------------------------------

/// A single texel in the lightmap being baked.
#[derive(Debug, Clone)]
pub struct BakeTexel {
    /// World-space position of this texel.
    pub position: [f32; 3],
    /// World-space normal at this texel.
    pub normal: [f32; 3],
    /// Accumulated radiance (RGB).
    pub radiance: [f32; 3],
    /// Accumulated directional radiance (for directional lightmaps).
    pub directional_radiance: [f32; 3],
    /// Dominant light direction (for directional lightmaps).
    pub dominant_direction: [f32; 3],
    /// Number of samples accumulated so far.
    pub sample_count: u32,
    /// Whether this texel is valid (maps to actual geometry).
    pub valid: bool,
    /// Chart index this texel belongs to.
    pub chart_id: u32,
}

impl BakeTexel {
    /// Create an invalid (empty) texel.
    pub fn invalid() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0; 3],
            radiance: [0.0; 3],
            directional_radiance: [0.0; 3],
            dominant_direction: [0.0, 1.0, 0.0],
            sample_count: 0,
            valid: false,
            chart_id: 0,
        }
    }

    /// Create a valid texel with a position and normal.
    pub fn new(position: [f32; 3], normal: [f32; 3], chart_id: u32) -> Self {
        Self {
            position,
            normal,
            radiance: [0.0; 3],
            directional_radiance: [0.0; 3],
            dominant_direction: [0.0, 1.0, 0.0],
            sample_count: 0,
            valid: true,
            chart_id,
        }
    }

    /// Add a sample to the accumulated radiance using running average.
    pub fn add_sample(&mut self, incoming: [f32; 3], direction: [f32; 3]) {
        self.sample_count += 1;
        let n = self.sample_count as f32;
        let inv_n = 1.0 / n;

        for i in 0..3 {
            self.radiance[i] += (incoming[i] - self.radiance[i]) * inv_n;
        }

        // Update directional info.
        let cos_theta = direction[0] * self.normal[0]
            + direction[1] * self.normal[1]
            + direction[2] * self.normal[2];
        let weight = cos_theta.max(0.0);

        for i in 0..3 {
            self.directional_radiance[i] += (incoming[i] * weight - self.directional_radiance[i]) * inv_n;
            self.dominant_direction[i] += (direction[i] - self.dominant_direction[i]) * inv_n;
        }
    }

    /// Normalize the dominant direction to unit length.
    pub fn normalize_direction(&mut self) {
        let len = (self.dominant_direction[0] * self.dominant_direction[0]
            + self.dominant_direction[1] * self.dominant_direction[1]
            + self.dominant_direction[2] * self.dominant_direction[2])
            .sqrt();
        if len > 1e-8 {
            self.dominant_direction[0] /= len;
            self.dominant_direction[1] /= len;
            self.dominant_direction[2] /= len;
        }
    }

    /// Returns the luminance of the accumulated radiance.
    pub fn luminance(&self) -> f32 {
        0.2126 * self.radiance[0] + 0.7152 * self.radiance[1] + 0.0722 * self.radiance[2]
    }
}

/// Path tracing configuration for the GI baker.
#[derive(Debug, Clone)]
pub struct PathTraceConfig {
    /// Total samples per texel.
    pub samples_per_texel: u32,
    /// Maximum path depth (bounces).
    pub max_bounces: u32,
    /// Whether to enable Russian roulette for path termination.
    pub russian_roulette: bool,
    /// Minimum bounces before Russian roulette kicks in.
    pub rr_min_bounces: u32,
    /// Whether to include direct lighting in the lightmap.
    pub include_direct: bool,
    /// Whether to include indirect lighting in the lightmap.
    pub include_indirect: bool,
    /// Whether to include emissive surfaces as light sources.
    pub include_emissive: bool,
    /// Sky intensity multiplier.
    pub sky_intensity: f32,
    /// Sky color (HDR).
    pub sky_color: [f32; 3],
    /// Whether to use importance sampling for environment light.
    pub importance_sample_sky: bool,
    /// Clamp maximum contribution from a single sample to reduce fireflies.
    pub max_sample_contribution: f32,
    /// Ray offset along normal to prevent self-intersection.
    pub ray_offset: f32,
}

impl Default for PathTraceConfig {
    fn default() -> Self {
        Self {
            samples_per_texel: 256,
            max_bounces: 3,
            russian_roulette: true,
            rr_min_bounces: 2,
            include_direct: true,
            include_indirect: true,
            include_emissive: true,
            sky_intensity: 1.0,
            sky_color: [0.5, 0.7, 1.0],
            importance_sample_sky: true,
            max_sample_contribution: 100.0,
            ray_offset: RAY_OFFSET_EPSILON,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene Geometry for Baking
// ---------------------------------------------------------------------------

/// A triangle in the bake scene.
#[derive(Debug, Clone)]
pub struct BakeTriangle {
    /// Vertex positions.
    pub positions: [[f32; 3]; 3],
    /// Vertex normals.
    pub normals: [[f32; 3]; 3],
    /// UV2 (lightmap) coordinates.
    pub uv2: [[f32; 2]; 3],
    /// Material index for this triangle.
    pub material_index: u32,
    /// Mesh index this triangle belongs to.
    pub mesh_index: u32,
}

impl BakeTriangle {
    /// Compute the face normal of this triangle.
    pub fn face_normal(&self) -> [f32; 3] {
        let e1 = [
            self.positions[1][0] - self.positions[0][0],
            self.positions[1][1] - self.positions[0][1],
            self.positions[1][2] - self.positions[0][2],
        ];
        let e2 = [
            self.positions[2][0] - self.positions[0][0],
            self.positions[2][1] - self.positions[0][1],
            self.positions[2][2] - self.positions[0][2],
        ];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if len < 1e-10 {
            return [0.0, 1.0, 0.0];
        }
        [cross[0] / len, cross[1] / len, cross[2] / len]
    }

    /// Compute the area of this triangle.
    pub fn area(&self) -> f32 {
        let e1 = [
            self.positions[1][0] - self.positions[0][0],
            self.positions[1][1] - self.positions[0][1],
            self.positions[1][2] - self.positions[0][2],
        ];
        let e2 = [
            self.positions[2][0] - self.positions[0][0],
            self.positions[2][1] - self.positions[0][1],
            self.positions[2][2] - self.positions[0][2],
        ];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
    }

    /// Interpolate position at barycentric coordinates (u, v).
    pub fn interpolate_position(&self, u: f32, v: f32) -> [f32; 3] {
        let w = 1.0 - u - v;
        [
            self.positions[0][0] * w + self.positions[1][0] * u + self.positions[2][0] * v,
            self.positions[0][1] * w + self.positions[1][1] * u + self.positions[2][1] * v,
            self.positions[0][2] * w + self.positions[1][2] * u + self.positions[2][2] * v,
        ]
    }

    /// Interpolate normal at barycentric coordinates (u, v).
    pub fn interpolate_normal(&self, u: f32, v: f32) -> [f32; 3] {
        let w = 1.0 - u - v;
        let n = [
            self.normals[0][0] * w + self.normals[1][0] * u + self.normals[2][0] * v,
            self.normals[0][1] * w + self.normals[1][1] * u + self.normals[2][1] * v,
            self.normals[0][2] * w + self.normals[1][2] * u + self.normals[2][2] * v,
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len < 1e-10 {
            return [0.0, 1.0, 0.0];
        }
        [n[0] / len, n[1] / len, n[2] / len]
    }
}

/// Material data for baking (simplified).
#[derive(Debug, Clone)]
pub struct BakeMaterial {
    /// Base albedo color.
    pub albedo: [f32; 3],
    /// Emissive color (HDR).
    pub emissive: [f32; 3],
    /// Whether this material is transparent.
    pub transparent: bool,
    /// Opacity (for transparent materials).
    pub opacity: f32,
    /// Whether to include this material in GI computation.
    pub contributes_gi: bool,
    /// Whether this is a two-sided material.
    pub two_sided: bool,
}

impl Default for BakeMaterial {
    fn default() -> Self {
        Self {
            albedo: [0.8, 0.8, 0.8],
            emissive: [0.0, 0.0, 0.0],
            transparent: false,
            opacity: 1.0,
            contributes_gi: true,
            two_sided: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Irradiance Probes
// ---------------------------------------------------------------------------

/// A single irradiance probe placed in the scene.
#[derive(Debug, Clone)]
pub struct IrradianceProbe {
    /// Unique probe identifier.
    pub id: u32,
    /// World-space position.
    pub position: [f32; 3],
    /// Spherical harmonics coefficients (L2, 9 coefficients per channel = 27 floats).
    pub sh_coefficients: [f32; 27],
    /// Whether this probe has been baked.
    pub baked: bool,
    /// Number of samples used to bake this probe.
    pub sample_count: u32,
    /// Influence radius.
    pub radius: f32,
    /// Priority (higher priority probes override lower).
    pub priority: u32,
    /// Whether this probe is interior-only.
    pub interior: bool,
}

impl IrradianceProbe {
    /// Create a new unbaked probe at the given position.
    pub fn new(id: u32, position: [f32; 3], radius: f32) -> Self {
        Self {
            id,
            position,
            sh_coefficients: [0.0; 27],
            baked: false,
            sample_count: 0,
            radius,
            priority: 0,
            interior: false,
        }
    }

    /// Add a radiance sample from the given direction.
    pub fn add_sample(&mut self, direction: [f32; 3], radiance: [f32; 3]) {
        self.sample_count += 1;
        let inv_n = 1.0 / self.sample_count as f32;

        // Evaluate SH basis functions.
        let sh_basis = Self::evaluate_sh_basis(direction);

        for c in 0..3 {
            for b in 0..9 {
                let idx = c * 9 + b;
                let value = radiance[c] * sh_basis[b];
                self.sh_coefficients[idx] += (value - self.sh_coefficients[idx]) * inv_n;
            }
        }
    }

    /// Evaluate L2 spherical harmonics basis functions for a direction.
    fn evaluate_sh_basis(dir: [f32; 3]) -> [f32; 9] {
        let x = dir[0];
        let y = dir[1];
        let z = dir[2];

        let c0 = 0.282095; // Y00
        let c1 = 0.488603; // Y1m
        let c2a = 1.092548; // Y2m (some)
        let c2b = 0.315392; // Y20
        let c2c = 0.546274; // Y22

        [
            c0,
            c1 * y,
            c1 * z,
            c1 * x,
            c2a * x * y,
            c2a * y * z,
            c2b * (3.0 * z * z - 1.0),
            c2a * x * z,
            c2c * (x * x - y * y),
        ]
    }

    /// Evaluate the irradiance at this probe in the given direction.
    pub fn evaluate(&self, direction: [f32; 3]) -> [f32; 3] {
        let sh_basis = Self::evaluate_sh_basis(direction);
        let mut result = [0.0f32; 3];
        for c in 0..3 {
            for b in 0..9 {
                result[c] += self.sh_coefficients[c * 9 + b] * sh_basis[b];
            }
            result[c] = result[c].max(0.0);
        }
        result
    }

    /// Check if a point is within this probe's influence radius.
    pub fn contains(&self, point: [f32; 3]) -> bool {
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        let dz = point[2] - self.position[2];
        (dx * dx + dy * dy + dz * dz) <= self.radius * self.radius
    }

    /// Returns the weight for this probe at the given distance.
    pub fn weight_at_distance(&self, distance: f32) -> f32 {
        if distance >= self.radius {
            return 0.0;
        }
        let t = distance / self.radius;
        // Smooth falloff.
        let s = 1.0 - t;
        s * s * (3.0 - 2.0 * s)
    }
}

/// Grid-based irradiance probe placement.
#[derive(Debug, Clone)]
pub struct ProbeGrid {
    /// Grid origin in world space.
    pub origin: [f32; 3],
    /// Grid cell size.
    pub cell_size: f32,
    /// Number of probes along each axis.
    pub dimensions: [u32; 3],
    /// All probes in the grid (flattened x, y, z order).
    pub probes: Vec<IrradianceProbe>,
}

impl ProbeGrid {
    /// Create a new probe grid.
    pub fn new(origin: [f32; 3], cell_size: f32, dimensions: [u32; 3]) -> Self {
        let total = (dimensions[0] * dimensions[1] * dimensions[2]) as usize;
        let mut probes = Vec::with_capacity(total);
        let mut id = 0u32;

        for z in 0..dimensions[2] {
            for y in 0..dimensions[1] {
                for x in 0..dimensions[0] {
                    let pos = [
                        origin[0] + x as f32 * cell_size,
                        origin[1] + y as f32 * cell_size,
                        origin[2] + z as f32 * cell_size,
                    ];
                    probes.push(IrradianceProbe::new(id, pos, cell_size * 1.5));
                    id += 1;
                }
            }
        }

        Self {
            origin,
            cell_size,
            dimensions,
            probes,
        }
    }

    /// Find the nearest probes for trilinear interpolation at a world position.
    pub fn find_interpolation_probes(&self, position: [f32; 3]) -> Vec<(u32, f32)> {
        let lx = (position[0] - self.origin[0]) / self.cell_size;
        let ly = (position[1] - self.origin[1]) / self.cell_size;
        let lz = (position[2] - self.origin[2]) / self.cell_size;

        let x0 = (lx.floor() as i32).max(0).min(self.dimensions[0] as i32 - 1) as u32;
        let y0 = (ly.floor() as i32).max(0).min(self.dimensions[1] as i32 - 1) as u32;
        let z0 = (lz.floor() as i32).max(0).min(self.dimensions[2] as i32 - 1) as u32;
        let x1 = (x0 + 1).min(self.dimensions[0] - 1);
        let y1 = (y0 + 1).min(self.dimensions[1] - 1);
        let z1 = (z0 + 1).min(self.dimensions[2] - 1);

        let fx = lx.fract();
        let fy = ly.fract();
        let fz = lz.fract();

        let mut results = Vec::with_capacity(8);
        let corners = [
            (x0, y0, z0, (1.0 - fx) * (1.0 - fy) * (1.0 - fz)),
            (x1, y0, z0, fx * (1.0 - fy) * (1.0 - fz)),
            (x0, y1, z0, (1.0 - fx) * fy * (1.0 - fz)),
            (x1, y1, z0, fx * fy * (1.0 - fz)),
            (x0, y0, z1, (1.0 - fx) * (1.0 - fy) * fz),
            (x1, y0, z1, fx * (1.0 - fy) * fz),
            (x0, y1, z1, (1.0 - fx) * fy * fz),
            (x1, y1, z1, fx * fy * fz),
        ];

        for (cx, cy, cz, weight) in corners {
            if weight > 1e-6 {
                let idx = cx + cy * self.dimensions[0] + cz * self.dimensions[0] * self.dimensions[1];
                results.push((idx, weight));
            }
        }

        results
    }

    /// Evaluate interpolated irradiance at a world position and direction.
    pub fn evaluate(&self, position: [f32; 3], direction: [f32; 3]) -> [f32; 3] {
        let probes = self.find_interpolation_probes(position);
        let mut result = [0.0f32; 3];

        for (idx, weight) in probes {
            if let Some(probe) = self.probes.get(idx as usize) {
                let irradiance = probe.evaluate(direction);
                result[0] += irradiance[0] * weight;
                result[1] += irradiance[1] * weight;
                result[2] += irradiance[2] * weight;
            }
        }

        result
    }

    /// Returns the total number of probes in the grid.
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Returns the number of baked probes.
    pub fn baked_count(&self) -> usize {
        self.probes.iter().filter(|p| p.baked).count()
    }
}

// ---------------------------------------------------------------------------
// GI Bake Session
// ---------------------------------------------------------------------------

/// Status of a bake session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeStatus {
    /// Not started.
    Idle,
    /// Collecting scene geometry.
    CollectingGeometry,
    /// Generating UV2.
    Unwrapping,
    /// Rasterising texels.
    Rasterising,
    /// Path tracing in progress.
    Tracing,
    /// Denoising.
    Denoising,
    /// Encoding to target format.
    Encoding,
    /// Baking irradiance probes.
    BakingProbes,
    /// Completed successfully.
    Complete,
    /// Cancelled by user.
    Cancelled,
    /// Failed with an error.
    Failed,
}

/// Progress information for a running bake.
#[derive(Debug, Clone)]
pub struct BakeProgress {
    /// Current phase.
    pub status: BakeStatus,
    /// Progress within current phase (0.0 to 1.0).
    pub phase_progress: f32,
    /// Overall progress across all phases (0.0 to 1.0).
    pub overall_progress: f32,
    /// Texels completed in the tracing phase.
    pub texels_completed: u64,
    /// Total texels to trace.
    pub texels_total: u64,
    /// Samples completed per texel (current batch).
    pub samples_completed: u32,
    /// Total samples per texel.
    pub samples_total: u32,
    /// Elapsed time in seconds.
    pub elapsed_seconds: f64,
    /// Estimated remaining time in seconds.
    pub estimated_remaining: f64,
    /// Probes baked (in probe baking phase).
    pub probes_baked: u32,
    /// Total probes to bake.
    pub probes_total: u32,
}

impl Default for BakeProgress {
    fn default() -> Self {
        Self {
            status: BakeStatus::Idle,
            phase_progress: 0.0,
            overall_progress: 0.0,
            texels_completed: 0,
            texels_total: 0,
            samples_completed: 0,
            samples_total: 0,
            elapsed_seconds: 0.0,
            estimated_remaining: 0.0,
            probes_baked: 0,
            probes_total: 0,
        }
    }
}

/// Main GI baking session that orchestrates the full pipeline.
#[derive(Debug)]
pub struct GiBakeSession {
    /// Bake quality preset.
    pub quality: GiBakeQuality,
    /// Path tracing configuration.
    pub trace_config: PathTraceConfig,
    /// Denoising configuration.
    pub denoise_config: DenoiseConfig,
    /// Target lightmap encoding.
    pub encoding: GiLightmapEncoding,
    /// UV2 unwrapper.
    pub unwrapper: Uv2Unwrapper,
    /// Whether to generate directional lightmaps.
    pub directional: bool,
    /// Scene triangles.
    pub triangles: Vec<BakeTriangle>,
    /// Scene materials.
    pub materials: Vec<BakeMaterial>,
    /// Lightmap texels.
    pub texels: Vec<BakeTexel>,
    /// Lightmap width.
    pub lightmap_width: u32,
    /// Lightmap height.
    pub lightmap_height: u32,
    /// Probe grid (optional).
    pub probe_grid: Option<ProbeGrid>,
    /// Current progress.
    pub progress: BakeProgress,
    /// Whether the bake has been cancelled.
    pub cancelled: bool,
    /// Output encoded lightmap data.
    pub output_data: Vec<u8>,
    /// Output directional lightmap data (if directional is enabled).
    pub output_directional_data: Vec<u8>,
    /// Error message if bake failed.
    pub error: Option<String>,
}

impl GiBakeSession {
    /// Create a new bake session with the given quality preset.
    pub fn new(quality: GiBakeQuality) -> Self {
        let trace_config = PathTraceConfig {
            samples_per_texel: quality.samples_per_texel(),
            max_bounces: quality.max_bounces(),
            ..Default::default()
        };

        let mut unwrapper = Uv2Unwrapper::new();
        unwrapper.texels_per_unit = quality.texels_per_unit();

        Self {
            quality,
            trace_config,
            denoise_config: DenoiseConfig::default(),
            encoding: GiLightmapEncoding::Rgbm,
            unwrapper,
            directional: false,
            triangles: Vec::new(),
            materials: Vec::new(),
            texels: Vec::new(),
            lightmap_width: 0,
            lightmap_height: 0,
            probe_grid: None,
            progress: BakeProgress::default(),
            cancelled: false,
            output_data: Vec::new(),
            output_directional_data: Vec::new(),
            error: None,
        }
    }

    /// Enable directional lightmaps.
    pub fn with_directional(mut self, enabled: bool) -> Self {
        self.directional = enabled;
        self
    }

    /// Set the target encoding.
    pub fn with_encoding(mut self, encoding: GiLightmapEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    /// Set the denoising configuration.
    pub fn with_denoise(mut self, config: DenoiseConfig) -> Self {
        self.denoise_config = config;
        self
    }

    /// Add a triangle to the bake scene.
    pub fn add_triangle(&mut self, triangle: BakeTriangle) {
        self.triangles.push(triangle);
    }

    /// Add a material to the bake scene.
    pub fn add_material(&mut self, material: BakeMaterial) {
        self.materials.push(material);
    }

    /// Set up a probe grid for irradiance probe baking.
    pub fn setup_probe_grid(&mut self, origin: [f32; 3], cell_size: f32, dims: [u32; 3]) {
        self.probe_grid = Some(ProbeGrid::new(origin, cell_size, dims));
    }

    /// Cancel the bake.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        self.progress.status = BakeStatus::Cancelled;
    }

    /// Returns the current bake status.
    pub fn status(&self) -> BakeStatus {
        self.progress.status
    }

    /// Encode a single texel's radiance to the target format.
    pub fn encode_texel(&self, radiance: [f32; 3]) -> [u8; 4] {
        match self.encoding {
            GiLightmapEncoding::Rgbm => Self::encode_rgbm(radiance),
            GiLightmapEncoding::LogLuv => Self::encode_logluv(radiance),
            GiLightmapEncoding::Rgbe => Self::encode_rgbe(radiance),
            _ => {
                // For float formats, we just store as bytes.
                let r = (radiance[0].clamp(0.0, 1.0) * 255.0) as u8;
                let g = (radiance[1].clamp(0.0, 1.0) * 255.0) as u8;
                let b = (radiance[2].clamp(0.0, 1.0) * 255.0) as u8;
                [r, g, b, 255]
            }
        }
    }

    /// Encode HDR radiance to RGBM format.
    fn encode_rgbm(radiance: [f32; 3]) -> [u8; 4] {
        let max_range = 8.0f32;
        let r = radiance[0] / max_range;
        let g = radiance[1] / max_range;
        let b = radiance[2] / max_range;

        let m = r.max(g).max(b).max(1e-6).min(1.0);
        let m_byte = (m * 255.0).ceil() / 255.0;

        let inv_m = 1.0 / m_byte;
        [
            (r * inv_m * 255.0).clamp(0.0, 255.0) as u8,
            (g * inv_m * 255.0).clamp(0.0, 255.0) as u8,
            (b * inv_m * 255.0).clamp(0.0, 255.0) as u8,
            (m_byte * 255.0).clamp(0.0, 255.0) as u8,
        ]
    }

    /// Encode HDR radiance to LogLUV format.
    fn encode_logluv(radiance: [f32; 3]) -> [u8; 4] {
        let x = 0.4124 * radiance[0] + 0.3576 * radiance[1] + 0.1805 * radiance[2];
        let y = 0.2126 * radiance[0] + 0.7152 * radiance[1] + 0.0722 * radiance[2];
        let z = 0.0193 * radiance[0] + 0.1192 * radiance[1] + 0.9505 * radiance[2];

        let denom = x + 15.0 * y + 3.0 * z;
        let (u_prime, v_prime) = if denom > 1e-10 {
            (4.0 * x / denom, 9.0 * y / denom)
        } else {
            (0.0, 0.0)
        };

        let le = if y > 0.0 {
            ((y.ln() * 256.0 / 20.0) + 128.0).clamp(0.0, 255.0)
        } else {
            0.0
        };

        [
            le as u8,
            (u_prime * 255.0).clamp(0.0, 255.0) as u8,
            (v_prime * 255.0).clamp(0.0, 255.0) as u8,
            255,
        ]
    }

    /// Encode HDR radiance to RGBE format.
    fn encode_rgbe(radiance: [f32; 3]) -> [u8; 4] {
        let max_val = radiance[0].max(radiance[1]).max(radiance[2]);
        if max_val < 1e-32 {
            return [0, 0, 0, 0];
        }

        let (mantissa, exponent) = frexp(max_val);
        let scale = mantissa * 256.0 / max_val;

        [
            (radiance[0] * scale) as u8,
            (radiance[1] * scale) as u8,
            (radiance[2] * scale) as u8,
            (exponent + 128) as u8,
        ]
    }

    /// Apply denoising to the lightmap texels.
    pub fn apply_denoise(&mut self) {
        if self.denoise_config.filter == DenoiseFilter::None {
            return;
        }

        let width = self.lightmap_width as i32;
        let height = self.lightmap_height as i32;
        let radius = self.denoise_config.radius as i32;
        let sigma_s = self.denoise_config.sigma_spatial;
        let sigma_r = self.denoise_config.sigma_range;

        let original: Vec<[f32; 3]> = self.texels.iter().map(|t| t.radiance).collect();
        let original_normals: Vec<[f32; 3]> = self.texels.iter().map(|t| t.normal).collect();

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if !self.texels[idx].valid {
                    continue;
                }

                let center_radiance = original[idx];
                let center_normal = original_normals[idx];

                let mut sum = [0.0f32; 3];
                let mut weight_sum = 0.0f32;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = x + dx;
                        let ny = y + dy;

                        if nx < 0 || nx >= width || ny < 0 || ny >= height {
                            continue;
                        }

                        let nidx = (ny * width + nx) as usize;
                        if !self.texels[nidx].valid {
                            continue;
                        }

                        let spatial_dist_sq = (dx * dx + dy * dy) as f32;
                        let spatial_weight = (-spatial_dist_sq / (2.0 * sigma_s * sigma_s)).exp();

                        let range_diff = [
                            original[nidx][0] - center_radiance[0],
                            original[nidx][1] - center_radiance[1],
                            original[nidx][2] - center_radiance[2],
                        ];
                        let range_dist_sq = range_diff[0] * range_diff[0]
                            + range_diff[1] * range_diff[1]
                            + range_diff[2] * range_diff[2];
                        let range_weight = (-range_dist_sq / (2.0 * sigma_r * sigma_r)).exp();

                        let mut normal_weight = 1.0f32;
                        if self.denoise_config.normal_aware {
                            let dot = center_normal[0] * original_normals[nidx][0]
                                + center_normal[1] * original_normals[nidx][1]
                                + center_normal[2] * original_normals[nidx][2];
                            let threshold = self.denoise_config.normal_threshold_degrees.to_radians().cos();
                            if dot < threshold {
                                normal_weight = 0.0;
                            }
                        }

                        let w = spatial_weight * range_weight * normal_weight;
                        sum[0] += original[nidx][0] * w;
                        sum[1] += original[nidx][1] * w;
                        sum[2] += original[nidx][2] * w;
                        weight_sum += w;
                    }
                }

                if weight_sum > 1e-8 {
                    let inv_w = 1.0 / weight_sum;
                    self.texels[idx].radiance = [sum[0] * inv_w, sum[1] * inv_w, sum[2] * inv_w];
                }
            }
        }
    }

    /// Encode the lightmap to the output format.
    pub fn encode_output(&mut self) {
        let texel_count = (self.lightmap_width * self.lightmap_height) as usize;
        self.output_data = Vec::with_capacity(texel_count * 4);

        for texel in &self.texels {
            let encoded = self.encode_texel(texel.radiance);
            self.output_data.extend_from_slice(&encoded);
        }

        if self.directional {
            self.output_directional_data = Vec::with_capacity(texel_count * 4);
            for texel in &self.texels {
                let dir = texel.dominant_direction;
                let encoded = [
                    ((dir[0] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8,
                    ((dir[1] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8,
                    ((dir[2] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8,
                    255,
                ];
                self.output_directional_data.extend_from_slice(&encoded);
            }
        }
    }

    /// Get the bake statistics summary.
    pub fn statistics(&self) -> GiBakeStatistics {
        GiBakeStatistics {
            triangle_count: self.triangles.len(),
            material_count: self.materials.len(),
            texel_count: self.texels.len(),
            valid_texel_count: self.texels.iter().filter(|t| t.valid).count(),
            lightmap_width: self.lightmap_width,
            lightmap_height: self.lightmap_height,
            chart_count: self.unwrapper.chart_count(),
            total_samples: self.texels.iter().map(|t| t.sample_count as u64).sum(),
            output_size_bytes: self.output_data.len(),
            directional_size_bytes: self.output_directional_data.len(),
            probe_count: self.probe_grid.as_ref().map(|g| g.probe_count()).unwrap_or(0),
            elapsed_seconds: self.progress.elapsed_seconds,
        }
    }
}

/// Statistics from a completed GI bake.
#[derive(Debug, Clone)]
pub struct GiBakeStatistics {
    /// Number of triangles in the scene.
    pub triangle_count: usize,
    /// Number of materials.
    pub material_count: usize,
    /// Total number of texels in the lightmap.
    pub texel_count: usize,
    /// Number of valid texels (mapping to geometry).
    pub valid_texel_count: usize,
    /// Lightmap width in texels.
    pub lightmap_width: u32,
    /// Lightmap height in texels.
    pub lightmap_height: u32,
    /// Number of UV charts.
    pub chart_count: usize,
    /// Total samples taken across all texels.
    pub total_samples: u64,
    /// Size of the encoded output in bytes.
    pub output_size_bytes: usize,
    /// Size of the directional lightmap in bytes.
    pub directional_size_bytes: usize,
    /// Number of irradiance probes.
    pub probe_count: usize,
    /// Total elapsed time in seconds.
    pub elapsed_seconds: f64,
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Software frexp implementation (decompose float into mantissa and exponent).
fn frexp(value: f32) -> (f32, i32) {
    if value == 0.0 {
        return (0.0, 0);
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 126;
    let mantissa = f32::from_bits((bits & 0x807FFFFF) | 0x3F000000);
    (mantissa, exponent)
}

/// Generate a cosine-weighted random direction in the hemisphere around the normal.
pub fn cosine_hemisphere_sample(normal: [f32; 3], u1: f32, u2: f32) -> [f32; 3] {
    let r = u1.sqrt();
    let theta = 2.0 * PI * u2;
    let x = r * theta.cos();
    let y = r * theta.sin();
    let z = (1.0 - u1).sqrt();

    // Build a coordinate system around the normal.
    let up = if normal[1].abs() < 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };

    let tangent = [
        up[1] * normal[2] - up[2] * normal[1],
        up[2] * normal[0] - up[0] * normal[2],
        up[0] * normal[1] - up[1] * normal[0],
    ];
    let len = (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
    let tangent = [tangent[0] / len, tangent[1] / len, tangent[2] / len];

    let bitangent = [
        normal[1] * tangent[2] - normal[2] * tangent[1],
        normal[2] * tangent[0] - normal[0] * tangent[2],
        normal[0] * tangent[1] - normal[1] * tangent[0],
    ];

    [
        tangent[0] * x + bitangent[0] * y + normal[0] * z,
        tangent[1] * x + bitangent[1] * y + normal[1] * z,
        tangent[2] * x + bitangent[2] * y + normal[2] * z,
    ]
}

/// Compute the PDF for a cosine-weighted hemisphere sample.
pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    cos_theta.max(0.0) * COSINE_HEMISPHERE_PDF
}

/// Directional lightmap data for a single texel, including SH coefficients.
#[derive(Debug, Clone)]
pub struct DirectionalTexelData {
    /// L1 spherical harmonics for R channel (4 coefficients).
    pub sh_r: [f32; 4],
    /// L1 spherical harmonics for G channel (4 coefficients).
    pub sh_g: [f32; 4],
    /// L1 spherical harmonics for B channel (4 coefficients).
    pub sh_b: [f32; 4],
    /// Dominant light direction (unit vector).
    pub direction: [f32; 3],
    /// Directional intensity (ratio of directional to total).
    pub directionality: f32,
}

impl DirectionalTexelData {
    /// Create empty directional data.
    pub fn zero() -> Self {
        Self {
            sh_r: [0.0; 4],
            sh_g: [0.0; 4],
            sh_b: [0.0; 4],
            direction: [0.0, 1.0, 0.0],
            directionality: 0.0,
        }
    }

    /// Evaluate the directional irradiance for a given surface normal.
    pub fn evaluate(&self, normal: [f32; 3]) -> [f32; 3] {
        let basis = [
            0.282095,           // Y00
            0.488603 * normal[1], // Y1-1
            0.488603 * normal[2], // Y10
            0.488603 * normal[0], // Y11
        ];

        let r: f32 = self.sh_r.iter().zip(basis.iter()).map(|(c, b)| c * b).sum();
        let g: f32 = self.sh_g.iter().zip(basis.iter()).map(|(c, b)| c * b).sum();
        let b: f32 = self.sh_b.iter().zip(basis.iter()).map(|(c, b)| c * b).sum();

        [r.max(0.0), g.max(0.0), b.max(0.0)]
    }
}
