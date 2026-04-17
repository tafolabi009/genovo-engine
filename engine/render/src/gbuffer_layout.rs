// engine/render/src/gbuffer_layout.rs
//
// G-Buffer configuration and encode/decode functions for deferred rendering.
//
// The G-Buffer stores geometric and material attributes per-pixel in multiple
// render targets. This module defines configurable layouts (thin, standard,
// extended) and provides encode/decode functions for packing/unpacking data.
//
// Standard layout:
//   RT0: albedo.rgb (R8G8B8) + metallic (A8)       -- RGBA8 unorm
//   RT1: normal.xy  (R16G16)  + roughness (B8) + flags (A8) -- RGBA16 float
//   RT2: emissive.rgb (R11G11B10) + AO (packed)    -- R11G11B10 float
//   RT3: velocity.xy (R16G16)                       -- RG16 float
//   Depth: D32 float
//
// Thin layout (bandwidth-optimized):
//   RT0: albedo.rgb (R8G8B8) + metallic (A8)
//   RT1: normal.xy (octahedron R8G8) + roughness (B8) + AO (A8)
//   Depth: D32 float
//
// Extended layout (maximum quality):
//   RT0-RT3: same as standard
//   RT4: subsurface color (R8G8B8) + subsurface amount (A8)
//   RT5: clearcoat roughness (R8) + anisotropy (G8) + aniso direction (B8A8)

use std::fmt;

// ---------------------------------------------------------------------------
// G-Buffer format enum
// ---------------------------------------------------------------------------

/// Pixel format for G-Buffer render targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GBufferFormat {
    RGBA8Unorm,
    RGBA8Snorm,
    RGBA16Float,
    RG16Float,
    RG16Snorm,
    R11G11B10Float,
    RGB10A2Unorm,
    R8Unorm,
    RG8Unorm,
    R16Float,
    R32Float,
    RGBA32Float,
    D16Unorm,
    D24UnormS8Uint,
    D32Float,
    D32FloatS8Uint,
}

impl GBufferFormat {
    /// Bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::RG8Unorm => 2,
            Self::R16Float => 2,
            Self::D16Unorm => 2,
            Self::RGBA8Unorm | Self::RGBA8Snorm | Self::RG16Float | Self::RG16Snorm
            | Self::R11G11B10Float | Self::RGB10A2Unorm | Self::R32Float
            | Self::D24UnormS8Uint | Self::D32Float => 4,
            Self::RGBA16Float | Self::D32FloatS8Uint => 8,
            Self::RGBA32Float => 16,
        }
    }

    /// Whether this is a depth format.
    pub fn is_depth(&self) -> bool {
        matches!(self, Self::D16Unorm | Self::D24UnormS8Uint | Self::D32Float | Self::D32FloatS8Uint)
    }

    /// Whether this is a floating-point format.
    pub fn is_float(&self) -> bool {
        matches!(self, Self::RGBA16Float | Self::RG16Float | Self::R11G11B10Float
            | Self::R16Float | Self::R32Float | Self::RGBA32Float
            | Self::D32Float | Self::D32FloatS8Uint)
    }

    /// Number of components.
    pub fn component_count(&self) -> u32 {
        match self {
            Self::R8Unorm | Self::R16Float | Self::R32Float
            | Self::D16Unorm | Self::D24UnormS8Uint | Self::D32Float | Self::D32FloatS8Uint => 1,
            Self::RG8Unorm | Self::RG16Float | Self::RG16Snorm => 2,
            Self::R11G11B10Float => 3,
            Self::RGBA8Unorm | Self::RGBA8Snorm | Self::RGBA16Float
            | Self::RGB10A2Unorm | Self::RGBA32Float => 4,
        }
    }
}

impl fmt::Display for GBufferFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Render target descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a single G-Buffer render target.
#[derive(Debug, Clone)]
pub struct RenderTargetDesc {
    /// Render target name (e.g. "RT0_AlbedoMetallic").
    pub name: String,
    /// Pixel format.
    pub format: GBufferFormat,
    /// What data channels are stored (for documentation and debug display).
    pub channels: Vec<ChannelDesc>,
    /// Whether this RT is required (false = optional, can be disabled).
    pub required: bool,
    /// Clear color.
    pub clear_color: [f32; 4],
}

/// Description of a single channel within a render target.
#[derive(Debug, Clone)]
pub struct ChannelDesc {
    /// Channel name (e.g. "albedo.r", "normal.x").
    pub name: String,
    /// Which RGBA component this occupies.
    pub component: Component,
    /// Data type.
    pub data_type: ChannelDataType,
    /// Value range [min, max].
    pub range: [f32; 2],
}

/// RGBA component selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    R,
    G,
    B,
    A,
    RG,
    RGB,
    RGBA,
}

/// Data type for a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelDataType {
    Unorm,
    Snorm,
    Float,
    Uint,
    Sint,
}

// ---------------------------------------------------------------------------
// G-Buffer layout presets
// ---------------------------------------------------------------------------

/// G-Buffer layout preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GBufferPreset {
    /// Minimal layout: 2 RTs + depth. Best for bandwidth-limited GPUs.
    Thin,
    /// Standard layout: 4 RTs + depth. Good balance of quality and bandwidth.
    Standard,
    /// Extended layout: 6 RTs + depth. Maximum quality for advanced shading.
    Extended,
    /// Custom layout defined by the user.
    Custom,
}

/// Complete G-Buffer layout configuration.
#[derive(Debug, Clone)]
pub struct GBufferLayout {
    /// Preset name.
    pub preset: GBufferPreset,
    /// Render target descriptors.
    pub render_targets: Vec<RenderTargetDesc>,
    /// Depth buffer format.
    pub depth_format: GBufferFormat,
    /// Whether to use stencil (requires D24S8 or D32FS8).
    pub use_stencil: bool,
    /// Resolution width.
    pub width: u32,
    /// Resolution height.
    pub height: u32,
    /// Resolution scale (0.5 = half-res, 1.0 = full-res).
    pub resolution_scale: f32,
    /// Whether normal encoding uses octahedron mapping (more accurate, 2 components).
    pub octahedron_normals: bool,
    /// Whether to store velocity in the G-Buffer (for TAA/motion blur).
    pub store_velocity: bool,
}

impl GBufferLayout {
    /// Create a thin G-Buffer layout.
    pub fn thin(width: u32, height: u32) -> Self {
        let rt0 = RenderTargetDesc {
            name: "RT0_AlbedoMetallic".to_string(),
            format: GBufferFormat::RGBA8Unorm,
            channels: vec![
                ChannelDesc {
                    name: "albedo.r".to_string(),
                    component: Component::R,
                    data_type: ChannelDataType::Unorm,
                    range: [0.0, 1.0],
                },
                ChannelDesc {
                    name: "albedo.g".to_string(),
                    component: Component::G,
                    data_type: ChannelDataType::Unorm,
                    range: [0.0, 1.0],
                },
                ChannelDesc {
                    name: "albedo.b".to_string(),
                    component: Component::B,
                    data_type: ChannelDataType::Unorm,
                    range: [0.0, 1.0],
                },
                ChannelDesc {
                    name: "metallic".to_string(),
                    component: Component::A,
                    data_type: ChannelDataType::Unorm,
                    range: [0.0, 1.0],
                },
            ],
            required: true,
            clear_color: [0.0, 0.0, 0.0, 0.0],
        };
        let rt1 = RenderTargetDesc {
            name: "RT1_NormalRoughnessAO".to_string(),
            format: GBufferFormat::RGBA8Unorm,
            channels: vec![
                ChannelDesc {
                    name: "normal.x (oct)".to_string(),
                    component: Component::R,
                    data_type: ChannelDataType::Unorm,
                    range: [0.0, 1.0],
                },
                ChannelDesc {
                    name: "normal.y (oct)".to_string(),
                    component: Component::G,
                    data_type: ChannelDataType::Unorm,
                    range: [0.0, 1.0],
                },
                ChannelDesc {
                    name: "roughness".to_string(),
                    component: Component::B,
                    data_type: ChannelDataType::Unorm,
                    range: [0.0, 1.0],
                },
                ChannelDesc {
                    name: "ao".to_string(),
                    component: Component::A,
                    data_type: ChannelDataType::Unorm,
                    range: [0.0, 1.0],
                },
            ],
            required: true,
            clear_color: [0.5, 0.5, 0.5, 1.0],
        };

        Self {
            preset: GBufferPreset::Thin,
            render_targets: vec![rt0, rt1],
            depth_format: GBufferFormat::D32Float,
            use_stencil: false,
            width,
            height,
            resolution_scale: 1.0,
            octahedron_normals: true,
            store_velocity: false,
        }
    }

    /// Create a standard G-Buffer layout.
    pub fn standard(width: u32, height: u32) -> Self {
        let rt0 = RenderTargetDesc {
            name: "RT0_AlbedoMetallic".to_string(),
            format: GBufferFormat::RGBA8Unorm,
            channels: vec![
                ChannelDesc { name: "albedo.r".into(), component: Component::R, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
                ChannelDesc { name: "albedo.g".into(), component: Component::G, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
                ChannelDesc { name: "albedo.b".into(), component: Component::B, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
                ChannelDesc { name: "metallic".into(), component: Component::A, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
            ],
            required: true,
            clear_color: [0.0, 0.0, 0.0, 0.0],
        };
        let rt1 = RenderTargetDesc {
            name: "RT1_NormalRoughnessFlags".to_string(),
            format: GBufferFormat::RGBA16Float,
            channels: vec![
                ChannelDesc { name: "normal.x (oct)".into(), component: Component::R, data_type: ChannelDataType::Float, range: [-1.0, 1.0] },
                ChannelDesc { name: "normal.y (oct)".into(), component: Component::G, data_type: ChannelDataType::Float, range: [-1.0, 1.0] },
                ChannelDesc { name: "roughness".into(), component: Component::B, data_type: ChannelDataType::Float, range: [0.0, 1.0] },
                ChannelDesc { name: "flags".into(), component: Component::A, data_type: ChannelDataType::Float, range: [0.0, 255.0] },
            ],
            required: true,
            clear_color: [0.0, 0.0, 0.5, 0.0],
        };
        let rt2 = RenderTargetDesc {
            name: "RT2_EmissiveAO".to_string(),
            format: GBufferFormat::R11G11B10Float,
            channels: vec![
                ChannelDesc { name: "emissive.r".into(), component: Component::R, data_type: ChannelDataType::Float, range: [0.0, 100.0] },
                ChannelDesc { name: "emissive.g".into(), component: Component::G, data_type: ChannelDataType::Float, range: [0.0, 100.0] },
                ChannelDesc { name: "emissive.b + ao".into(), component: Component::B, data_type: ChannelDataType::Float, range: [0.0, 100.0] },
            ],
            required: true,
            clear_color: [0.0, 0.0, 0.0, 0.0],
        };
        let rt3 = RenderTargetDesc {
            name: "RT3_Velocity".to_string(),
            format: GBufferFormat::RG16Float,
            channels: vec![
                ChannelDesc { name: "velocity.x".into(), component: Component::R, data_type: ChannelDataType::Float, range: [-1.0, 1.0] },
                ChannelDesc { name: "velocity.y".into(), component: Component::G, data_type: ChannelDataType::Float, range: [-1.0, 1.0] },
            ],
            required: false,
            clear_color: [0.0, 0.0, 0.0, 0.0],
        };

        Self {
            preset: GBufferPreset::Standard,
            render_targets: vec![rt0, rt1, rt2, rt3],
            depth_format: GBufferFormat::D32Float,
            use_stencil: false,
            width,
            height,
            resolution_scale: 1.0,
            octahedron_normals: true,
            store_velocity: true,
        }
    }

    /// Create an extended G-Buffer layout.
    pub fn extended(width: u32, height: u32) -> Self {
        let mut layout = Self::standard(width, height);
        layout.preset = GBufferPreset::Extended;

        layout.render_targets.push(RenderTargetDesc {
            name: "RT4_SubsurfaceColor".to_string(),
            format: GBufferFormat::RGBA8Unorm,
            channels: vec![
                ChannelDesc { name: "sss_color.r".into(), component: Component::R, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
                ChannelDesc { name: "sss_color.g".into(), component: Component::G, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
                ChannelDesc { name: "sss_color.b".into(), component: Component::B, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
                ChannelDesc { name: "sss_amount".into(), component: Component::A, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
            ],
            required: false,
            clear_color: [0.0, 0.0, 0.0, 0.0],
        });
        layout.render_targets.push(RenderTargetDesc {
            name: "RT5_Clearcoat_Anisotropy".to_string(),
            format: GBufferFormat::RGBA8Unorm,
            channels: vec![
                ChannelDesc { name: "clearcoat_roughness".into(), component: Component::R, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
                ChannelDesc { name: "anisotropy".into(), component: Component::G, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
                ChannelDesc { name: "aniso_dir.x".into(), component: Component::B, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
                ChannelDesc { name: "aniso_dir.y".into(), component: Component::A, data_type: ChannelDataType::Unorm, range: [0.0, 1.0] },
            ],
            required: false,
            clear_color: [0.0, 0.0, 0.5, 0.5],
        });

        layout
    }

    /// Total bandwidth per pixel in bytes.
    pub fn bandwidth_per_pixel(&self) -> u32 {
        let mut total = self.depth_format.bytes_per_pixel();
        for rt in &self.render_targets {
            total += rt.format.bytes_per_pixel();
        }
        total
    }

    /// Total bandwidth per frame in bytes.
    pub fn bandwidth_per_frame(&self) -> u64 {
        let effective_width = (self.width as f32 * self.resolution_scale) as u64;
        let effective_height = (self.height as f32 * self.resolution_scale) as u64;
        let pixels = effective_width * effective_height;
        pixels * self.bandwidth_per_pixel() as u64
    }

    /// Total GPU memory used by the G-Buffer in bytes.
    pub fn total_memory(&self) -> u64 {
        self.bandwidth_per_frame()
    }

    /// Number of render targets.
    pub fn render_target_count(&self) -> usize {
        self.render_targets.len()
    }

    /// Effective resolution after scale.
    pub fn effective_resolution(&self) -> (u32, u32) {
        let w = (self.width as f32 * self.resolution_scale) as u32;
        let h = (self.height as f32 * self.resolution_scale) as u32;
        (w.max(1), h.max(1))
    }

    /// Resize the G-Buffer.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

// ---------------------------------------------------------------------------
// Octahedron normal encoding/decoding
// ---------------------------------------------------------------------------

/// Encode a unit normal vector using octahedron mapping.
///
/// Maps a unit vector on the sphere to a point in the [-1, 1]^2 square.
/// This is more accurate than spherical coordinates and uses only 2 components.
pub fn octahedron_encode(normal: [f32; 3]) -> [f32; 2] {
    let n = normal;
    let abs_sum = n[0].abs() + n[1].abs() + n[2].abs();
    let mut oct = [n[0] / abs_sum, n[1] / abs_sum];

    if n[2] < 0.0 {
        let sign_x = if oct[0] >= 0.0 { 1.0 } else { -1.0 };
        let sign_y = if oct[1] >= 0.0 { 1.0 } else { -1.0 };
        oct = [
            (1.0 - oct[1].abs()) * sign_x,
            (1.0 - oct[0].abs()) * sign_y,
        ];
    }

    oct
}

/// Decode a unit normal vector from octahedron mapping.
pub fn octahedron_decode(oct: [f32; 2]) -> [f32; 3] {
    let mut n = [oct[0], oct[1], 1.0 - oct[0].abs() - oct[1].abs()];

    if n[2] < 0.0 {
        let sign_x = if n[0] >= 0.0 { 1.0 } else { -1.0 };
        let sign_y = if n[1] >= 0.0 { 1.0 } else { -1.0 };
        let ox = n[0];
        let oy = n[1];
        n[0] = (1.0 - oy.abs()) * sign_x;
        n[1] = (1.0 - ox.abs()) * sign_y;
    }

    // Normalize.
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 {
        n[0] /= len;
        n[1] /= len;
        n[2] /= len;
    }

    n
}

/// Pack octahedron-encoded normal into two 16-bit unorm values (stored in a u32).
pub fn octahedron_pack_u32(normal: [f32; 3]) -> u32 {
    let oct = octahedron_encode(normal);
    let x = ((oct[0] * 0.5 + 0.5).clamp(0.0, 1.0) * 65535.0) as u32;
    let y = ((oct[1] * 0.5 + 0.5).clamp(0.0, 1.0) * 65535.0) as u32;
    (x << 16) | y
}

/// Unpack a u32 into an octahedron-encoded normal.
pub fn octahedron_unpack_u32(packed: u32) -> [f32; 3] {
    let x = ((packed >> 16) & 0xFFFF) as f32 / 65535.0 * 2.0 - 1.0;
    let y = (packed & 0xFFFF) as f32 / 65535.0 * 2.0 - 1.0;
    octahedron_decode([x, y])
}

/// Pack octahedron-encoded normal into two 8-bit unorm values (stored in a u16).
pub fn octahedron_pack_u16(normal: [f32; 3]) -> u16 {
    let oct = octahedron_encode(normal);
    let x = ((oct[0] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u16;
    let y = ((oct[1] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u16;
    (x << 8) | y
}

/// Unpack a u16 into an octahedron-encoded normal.
pub fn octahedron_unpack_u16(packed: u16) -> [f32; 3] {
    let x = ((packed >> 8) & 0xFF) as f32 / 255.0 * 2.0 - 1.0;
    let y = (packed & 0xFF) as f32 / 255.0 * 2.0 - 1.0;
    octahedron_decode([x, y])
}

// ---------------------------------------------------------------------------
// G-Buffer encode/decode functions
// ---------------------------------------------------------------------------

/// Encoded G-Buffer data for a single pixel (CPU-side representation).
#[derive(Debug, Clone, Copy)]
pub struct GBufferPixel {
    /// RT0: albedo.rgb + metallic.
    pub rt0: [f32; 4],
    /// RT1: normal.xy + roughness + flags.
    pub rt1: [f32; 4],
    /// RT2: emissive.rgb (+ ao packed in blue).
    pub rt2: [f32; 3],
    /// RT3: velocity.xy.
    pub rt3: [f32; 2],
    /// Depth value.
    pub depth: f32,
}

/// Surface data decoded from the G-Buffer.
#[derive(Debug, Clone, Copy)]
pub struct GBufferSurface {
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub normal: [f32; 3],
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub ao: f32,
    pub velocity: [f32; 2],
    pub depth: f32,
    pub flags: u32,
}

/// Material flags stored in the G-Buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GBufferFlags {
    /// Default PBR material.
    Default = 0,
    /// Subsurface scattering material.
    Subsurface = 1,
    /// Hair material.
    Hair = 2,
    /// Cloth material.
    Cloth = 3,
    /// Eye material.
    Eye = 4,
    /// Clearcoat material.
    Clearcoat = 5,
    /// Unlit (emissive only).
    Unlit = 6,
    /// Foliage (two-sided, SSS).
    Foliage = 7,
}

/// Encode surface data into G-Buffer pixel format.
pub fn encode_gbuffer(
    albedo: [f32; 3],
    metallic: f32,
    normal: [f32; 3],
    roughness: f32,
    emissive: [f32; 3],
    ao: f32,
    velocity: [f32; 2],
    depth: f32,
    flags: GBufferFlags,
) -> GBufferPixel {
    let oct_normal = octahedron_encode(normal);

    GBufferPixel {
        rt0: [albedo[0], albedo[1], albedo[2], metallic],
        rt1: [oct_normal[0], oct_normal[1], roughness, flags as u32 as f32 / 255.0],
        rt2: [
            emissive[0],
            emissive[1],
            emissive[2] + ao * 0.001, // Pack AO into emissive blue LSB.
        ],
        rt3: velocity,
        depth,
    }
}

/// Decode G-Buffer pixel data back into surface attributes.
pub fn decode_gbuffer(pixel: &GBufferPixel) -> GBufferSurface {
    let albedo = [pixel.rt0[0], pixel.rt0[1], pixel.rt0[2]];
    let metallic = pixel.rt0[3];
    let normal = octahedron_decode([pixel.rt1[0], pixel.rt1[1]]);
    let roughness = pixel.rt1[2];
    let flags = (pixel.rt1[3] * 255.0) as u32;
    let emissive = [pixel.rt2[0], pixel.rt2[1], pixel.rt2[2]];
    let ao = (pixel.rt2[2].fract() * 1000.0).clamp(0.0, 1.0);
    let velocity = pixel.rt3;
    let depth = pixel.depth;

    GBufferSurface {
        albedo,
        metallic,
        normal,
        roughness,
        emissive,
        ao,
        velocity,
        depth,
        flags,
    }
}

// ---------------------------------------------------------------------------
// Bandwidth analysis
// ---------------------------------------------------------------------------

/// Bandwidth analysis result for a G-Buffer configuration.
#[derive(Debug, Clone)]
pub struct BandwidthAnalysis {
    /// Layout preset name.
    pub preset: GBufferPreset,
    /// Resolution.
    pub width: u32,
    pub height: u32,
    /// Bytes per pixel.
    pub bytes_per_pixel: u32,
    /// Total memory in bytes.
    pub total_memory_bytes: u64,
    /// Total memory in megabytes.
    pub total_memory_mb: f64,
    /// Estimated bandwidth per frame at a given FPS.
    pub bandwidth_mbps_at_60fps: f64,
    pub bandwidth_mbps_at_120fps: f64,
    /// Per-RT breakdown.
    pub rt_breakdown: Vec<(String, u32, f64)>,
}

/// Analyze bandwidth for a G-Buffer layout.
pub fn analyze_bandwidth(layout: &GBufferLayout) -> BandwidthAnalysis {
    let (w, h) = layout.effective_resolution();
    let pixels = w as u64 * h as u64;
    let bpp = layout.bandwidth_per_pixel();
    let total = pixels * bpp as u64;
    let total_mb = total as f64 / (1024.0 * 1024.0);

    // Read + write per frame (deferred pass reads, lighting pass writes back).
    let rw_total_mb = total_mb * 2.0;
    let bw_60 = rw_total_mb * 60.0;
    let bw_120 = rw_total_mb * 120.0;

    let mut rt_breakdown = Vec::new();
    for rt in &layout.render_targets {
        let rt_size = pixels * rt.format.bytes_per_pixel() as u64;
        let rt_mb = rt_size as f64 / (1024.0 * 1024.0);
        let rt_pct = rt_mb / total_mb * 100.0;
        rt_breakdown.push((rt.name.clone(), rt.format.bytes_per_pixel(), rt_pct));
    }
    // Depth.
    let depth_size = pixels * layout.depth_format.bytes_per_pixel() as u64;
    let depth_mb = depth_size as f64 / (1024.0 * 1024.0);
    let depth_pct = depth_mb / total_mb * 100.0;
    rt_breakdown.push(("Depth".to_string(), layout.depth_format.bytes_per_pixel(), depth_pct));

    BandwidthAnalysis {
        preset: layout.preset,
        width: w,
        height: h,
        bytes_per_pixel: bpp,
        total_memory_bytes: total,
        total_memory_mb: total_mb,
        bandwidth_mbps_at_60fps: bw_60,
        bandwidth_mbps_at_120fps: bw_120,
        rt_breakdown,
    }
}

/// Print a human-readable bandwidth analysis.
pub fn format_bandwidth_analysis(analysis: &BandwidthAnalysis) -> String {
    let mut s = String::new();
    s.push_str(&format!("G-Buffer Bandwidth Analysis ({:?})\n", analysis.preset));
    s.push_str(&format!("  Resolution: {}x{}\n", analysis.width, analysis.height));
    s.push_str(&format!("  Bytes/pixel: {}\n", analysis.bytes_per_pixel));
    s.push_str(&format!("  Total memory: {:.2} MB\n", analysis.total_memory_mb));
    s.push_str(&format!("  Bandwidth @60fps: {:.1} MB/s\n", analysis.bandwidth_mbps_at_60fps));
    s.push_str(&format!("  Bandwidth @120fps: {:.1} MB/s\n", analysis.bandwidth_mbps_at_120fps));
    s.push_str("  RT breakdown:\n");
    for (name, bpp, pct) in &analysis.rt_breakdown {
        s.push_str(&format!("    {} ({}B/px): {:.1}%\n", name, bpp, pct));
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octahedron_roundtrip() {
        let normals = [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.577, 0.577, 0.577], // normalized (1,1,1)
        ];
        for n in &normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let normalized = [n[0] / len, n[1] / len, n[2] / len];
            let encoded = octahedron_encode(normalized);
            let decoded = octahedron_decode(encoded);
            for i in 0..3 {
                assert!((decoded[i] - normalized[i]).abs() < 0.01,
                    "Component {} mismatch: {} vs {}", i, decoded[i], normalized[i]);
            }
        }
    }

    #[test]
    fn test_octahedron_u32_roundtrip() {
        let normal = [0.0f32, 1.0, 0.0];
        let packed = octahedron_pack_u32(normal);
        let unpacked = octahedron_unpack_u32(packed);
        assert!((unpacked[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_thin_layout_bandwidth() {
        let layout = GBufferLayout::thin(1920, 1080);
        assert_eq!(layout.render_target_count(), 2);
        let bpp = layout.bandwidth_per_pixel();
        // RT0: 4 + RT1: 4 + Depth: 4 = 12 bytes/pixel.
        assert_eq!(bpp, 12);
    }

    #[test]
    fn test_standard_layout() {
        let layout = GBufferLayout::standard(1920, 1080);
        assert_eq!(layout.render_target_count(), 4);
        assert!(layout.store_velocity);
    }

    #[test]
    fn test_extended_layout() {
        let layout = GBufferLayout::extended(1920, 1080);
        assert_eq!(layout.render_target_count(), 6);
    }

    #[test]
    fn test_gbuffer_encode_decode() {
        let pixel = encode_gbuffer(
            [0.8, 0.2, 0.1],
            0.5,
            [0.0, 1.0, 0.0],
            0.3,
            [0.0, 0.0, 0.0],
            1.0,
            [0.01, -0.02],
            0.5,
            GBufferFlags::Default,
        );
        let surface = decode_gbuffer(&pixel);
        assert!((surface.albedo[0] - 0.8).abs() < 0.01);
        assert!((surface.metallic - 0.5).abs() < 0.01);
        assert!((surface.roughness - 0.3).abs() < 0.01);
        assert!((surface.normal[1] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_bandwidth_analysis() {
        let layout = GBufferLayout::standard(1920, 1080);
        let analysis = analyze_bandwidth(&layout);
        assert!(analysis.total_memory_mb > 0.0);
        assert!(analysis.bandwidth_mbps_at_60fps > 0.0);
        let report = format_bandwidth_analysis(&analysis);
        assert!(report.contains("Standard"));
    }
}
