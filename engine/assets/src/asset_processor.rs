//! # Asset Import Processing
//!
//! Provides a pipeline for importing raw source assets (textures, meshes, audio,
//! shaders) and converting them into engine-ready formats. Includes texture
//! compression to GPU formats (BC1/BC3/ASTC), mesh optimisation (vertex cache,
//! overdraw), audio compression, mipmap generation, thumbnail generation for
//! the editor, and batch processing across the entire asset database.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during asset processing.
#[derive(Debug)]
pub enum ProcessorError {
    /// Generic I/O error.
    Io(std::io::Error),
    /// The input format is not recognised.
    UnknownFormat(String),
    /// The asset data is corrupt or could not be parsed.
    ParseError(String),
    /// Texture compression failed.
    CompressionFailed(String),
    /// Mesh optimisation failed.
    MeshOptFailed(String),
    /// Audio encoding failed.
    AudioEncodeFailed(String),
    /// A required source file was not found.
    SourceNotFound(PathBuf),
    /// An import setting value is out of range.
    InvalidSetting { key: String, reason: String },
    /// Processing was cancelled by the user.
    Cancelled,
}

impl fmt::Display for ProcessorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::UnknownFormat(fmt_name) => write!(f, "unknown format: {fmt_name}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::CompressionFailed(msg) => write!(f, "compression failed: {msg}"),
            Self::MeshOptFailed(msg) => write!(f, "mesh optimisation failed: {msg}"),
            Self::AudioEncodeFailed(msg) => write!(f, "audio encode failed: {msg}"),
            Self::SourceNotFound(p) => write!(f, "source not found: {}", p.display()),
            Self::InvalidSetting { key, reason } => {
                write!(f, "invalid setting '{key}': {reason}")
            }
            Self::Cancelled => write!(f, "processing cancelled"),
        }
    }
}

impl std::error::Error for ProcessorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ProcessorError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Texture formats
// ---------------------------------------------------------------------------

/// GPU texture compression formats that the processor can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTextureFormat {
    /// Uncompressed RGBA8 (32 bpp).
    Rgba8,
    /// BC1 (DXT1) -- 4 bpp, no alpha or 1-bit alpha.
    Bc1,
    /// BC3 (DXT5) -- 8 bpp, smooth alpha channel.
    Bc3,
    /// BC4 -- single channel (e.g. heightmaps).
    Bc4,
    /// BC5 -- two channels (e.g. normal maps).
    Bc5,
    /// BC7 -- high-quality RGBA compression.
    Bc7,
    /// ASTC 4x4 block (8 bpp, mobile GPUs).
    Astc4x4,
    /// ASTC 6x6 block (~3.6 bpp).
    Astc6x6,
    /// ASTC 8x8 block (2 bpp).
    Astc8x8,
    /// ETC2 RGB (mobile fallback).
    Etc2Rgb,
    /// ETC2 RGBA.
    Etc2Rgba,
}

impl GpuTextureFormat {
    /// Bits per pixel for this format.
    pub fn bits_per_pixel(self) -> f32 {
        match self {
            Self::Rgba8 => 32.0,
            Self::Bc1 => 4.0,
            Self::Bc3 => 8.0,
            Self::Bc4 => 4.0,
            Self::Bc5 => 8.0,
            Self::Bc7 => 8.0,
            Self::Astc4x4 => 8.0,
            Self::Astc6x6 => 3.56,
            Self::Astc8x8 => 2.0,
            Self::Etc2Rgb => 4.0,
            Self::Etc2Rgba => 8.0,
        }
    }

    /// Block dimensions for this format (width, height).
    pub fn block_size(self) -> (u32, u32) {
        match self {
            Self::Rgba8 => (1, 1),
            Self::Bc1 | Self::Bc3 | Self::Bc4 | Self::Bc5 | Self::Bc7 => (4, 4),
            Self::Astc4x4 => (4, 4),
            Self::Astc6x6 => (6, 6),
            Self::Astc8x8 => (8, 8),
            Self::Etc2Rgb | Self::Etc2Rgba => (4, 4),
        }
    }

    /// Whether this format supports an alpha channel.
    pub fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::Rgba8 | Self::Bc3 | Self::Bc7 | Self::Astc4x4 | Self::Astc6x6 | Self::Astc8x8 | Self::Etc2Rgba
        )
    }

    /// Name string for display.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rgba8 => "RGBA8",
            Self::Bc1 => "BC1",
            Self::Bc3 => "BC3",
            Self::Bc4 => "BC4",
            Self::Bc5 => "BC5",
            Self::Bc7 => "BC7",
            Self::Astc4x4 => "ASTC 4x4",
            Self::Astc6x6 => "ASTC 6x6",
            Self::Astc8x8 => "ASTC 8x8",
            Self::Etc2Rgb => "ETC2 RGB",
            Self::Etc2Rgba => "ETC2 RGBA",
        }
    }
}

// ---------------------------------------------------------------------------
// Import settings
// ---------------------------------------------------------------------------

/// Per-asset-type import settings controlling how the processor transforms
/// raw source data into engine-ready formats.
#[derive(Debug, Clone)]
pub struct TextureImportSettings {
    /// Target GPU format.
    pub format: GpuTextureFormat,
    /// Whether to generate a full mipmap chain.
    pub generate_mipmaps: bool,
    /// Maximum texture dimension (will be downscaled if larger).
    pub max_size: u32,
    /// sRGB colour space flag.
    pub srgb: bool,
    /// Flip the image vertically on import.
    pub flip_y: bool,
    /// Enable alpha premultiplication.
    pub premultiply_alpha: bool,
    /// Quality setting for BC/ASTC encoding (0 = fastest, 100 = best).
    pub quality: u32,
    /// Whether to generate a thumbnail for the editor.
    pub generate_thumbnail: bool,
    /// Thumbnail dimension (square).
    pub thumbnail_size: u32,
    /// Normal-map mode: treat the texture as a normal map.
    pub is_normal_map: bool,
}

impl Default for TextureImportSettings {
    fn default() -> Self {
        Self {
            format: GpuTextureFormat::Bc3,
            generate_mipmaps: true,
            max_size: 4096,
            srgb: true,
            flip_y: false,
            premultiply_alpha: false,
            quality: 75,
            generate_thumbnail: true,
            thumbnail_size: 128,
            is_normal_map: false,
        }
    }
}

impl TextureImportSettings {
    /// Validate the settings and return a list of issues.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.max_size == 0 || !self.max_size.is_power_of_two() {
            issues.push(format!("max_size must be a power of two, got {}", self.max_size));
        }
        if self.quality > 100 {
            issues.push(format!("quality must be 0-100, got {}", self.quality));
        }
        if self.thumbnail_size == 0 {
            issues.push("thumbnail_size must be > 0".into());
        }
        if self.is_normal_map && self.srgb {
            issues.push("normal maps should not use sRGB".into());
        }
        issues
    }
}

/// Import settings for mesh assets.
#[derive(Debug, Clone)]
pub struct MeshImportSettings {
    /// Optimise vertex order for GPU vertex cache efficiency.
    pub optimize_vertex_cache: bool,
    /// Optimise for reduced overdraw.
    pub optimize_overdraw: bool,
    /// Merge vertices that share position/normal/UV within tolerance.
    pub weld_vertices: bool,
    /// Tolerance for vertex welding.
    pub weld_tolerance: f32,
    /// Generate smooth normals if the source mesh lacks them.
    pub generate_normals: bool,
    /// Generate tangent vectors for normal mapping.
    pub generate_tangents: bool,
    /// Scale factor applied to all vertex positions.
    pub scale: f32,
    /// Whether to strip unused vertex attributes.
    pub strip_unused_attributes: bool,
    /// Maximum number of bones influencing a single vertex (for skinned meshes).
    pub max_bone_influences: u32,
    /// Generate LOD (level-of-detail) meshes.
    pub generate_lods: bool,
    /// Number of LOD levels to generate.
    pub lod_count: u32,
    /// Target triangle reduction ratio per LOD level (e.g. 0.5 = half).
    pub lod_reduction: f32,
}

impl Default for MeshImportSettings {
    fn default() -> Self {
        Self {
            optimize_vertex_cache: true,
            optimize_overdraw: true,
            weld_vertices: true,
            weld_tolerance: 0.0001,
            generate_normals: true,
            generate_tangents: true,
            scale: 1.0,
            strip_unused_attributes: false,
            max_bone_influences: 4,
            generate_lods: false,
            lod_count: 3,
            lod_reduction: 0.5,
        }
    }
}

/// Import settings for audio assets.
#[derive(Debug, Clone)]
pub struct AudioImportSettings {
    /// Target sample rate in Hz (0 = keep original).
    pub target_sample_rate: u32,
    /// Target channel count (0 = keep original, 1 = mono, 2 = stereo).
    pub target_channels: u32,
    /// Normalise peak volume to this level in dB (None = no normalisation).
    pub normalize_db: Option<f32>,
    /// Compress to Vorbis/Opus for runtime streaming.
    pub compress: bool,
    /// Compression quality (0.0-1.0 for Vorbis).
    pub compression_quality: f32,
    /// Whether to trim silence from the start and end.
    pub trim_silence: bool,
    /// Silence threshold in dB.
    pub silence_threshold_db: f32,
    /// Loop point (in samples, None = no loop).
    pub loop_start: Option<u64>,
    /// Loop end (in samples, None = end of file).
    pub loop_end: Option<u64>,
    /// Whether to force loading the entire clip into memory at once.
    pub force_preload: bool,
}

impl Default for AudioImportSettings {
    fn default() -> Self {
        Self {
            target_sample_rate: 0,
            target_channels: 0,
            normalize_db: None,
            compress: true,
            compression_quality: 0.5,
            trim_silence: false,
            silence_threshold_db: -60.0,
            loop_start: None,
            loop_end: None,
            force_preload: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Texture processing
// ---------------------------------------------------------------------------

/// Raw pixel data for processing.
#[derive(Debug, Clone)]
pub struct RawPixels {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Channel count (1, 2, 3, or 4).
    pub channels: u32,
    /// Raw pixel data in row-major order, 8 bits per channel.
    pub data: Vec<u8>,
}

impl RawPixels {
    /// Create a new pixel buffer.
    pub fn new(width: u32, height: u32, channels: u32) -> Self {
        let size = (width * height * channels) as usize;
        Self {
            width,
            height,
            channels,
            data: vec![0u8; size],
        }
    }

    /// Create from existing data.
    pub fn from_data(width: u32, height: u32, channels: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            channels,
            data,
        }
    }

    /// Byte stride per row.
    pub fn row_stride(&self) -> usize {
        (self.width * self.channels) as usize
    }

    /// Total number of pixels.
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Get a pixel at (x, y).
    pub fn get_pixel(&self, x: u32, y: u32) -> &[u8] {
        let idx = ((y * self.width + x) * self.channels) as usize;
        &self.data[idx..idx + self.channels as usize]
    }

    /// Set a pixel at (x, y).
    pub fn set_pixel(&mut self, x: u32, y: u32, pixel: &[u8]) {
        let idx = ((y * self.width + x) * self.channels) as usize;
        let ch = self.channels as usize;
        self.data[idx..idx + ch].copy_from_slice(&pixel[..ch]);
    }

    /// Flip the image vertically.
    pub fn flip_vertical(&mut self) {
        let stride = self.row_stride();
        let h = self.height as usize;
        for y in 0..h / 2 {
            let top = y * stride;
            let bot = (h - 1 - y) * stride;
            for x in 0..stride {
                self.data.swap(top + x, bot + x);
            }
        }
    }

    /// Convert from 3-channel (RGB) to 4-channel (RGBA) by adding alpha = 255.
    pub fn to_rgba(&self) -> Self {
        if self.channels == 4 {
            return self.clone();
        }
        let mut rgba = Self::new(self.width, self.height, 4);
        for y in 0..self.height {
            for x in 0..self.width {
                let src = self.get_pixel(x, y);
                let pixel = match self.channels {
                    1 => [src[0], src[0], src[0], 255],
                    2 => [src[0], src[0], src[0], src[1]],
                    3 => [src[0], src[1], src[2], 255],
                    _ => [src[0], src[1], src[2], src[3]],
                };
                rgba.set_pixel(x, y, &pixel);
            }
        }
        rgba
    }

    /// Premultiply alpha: R *= A/255, G *= A/255, B *= A/255.
    pub fn premultiply_alpha(&mut self) {
        if self.channels < 4 {
            return;
        }
        for i in (0..self.data.len()).step_by(4) {
            let a = self.data[i + 3] as f32 / 255.0;
            self.data[i] = (self.data[i] as f32 * a) as u8;
            self.data[i + 1] = (self.data[i + 1] as f32 * a) as u8;
            self.data[i + 2] = (self.data[i + 2] as f32 * a) as u8;
        }
    }

    /// Downscale by half (box filter).
    pub fn downscale_half(&self) -> Self {
        let new_w = (self.width / 2).max(1);
        let new_h = (self.height / 2).max(1);
        let ch = self.channels as usize;
        let mut out = Self::new(new_w, new_h, self.channels);
        for y in 0..new_h {
            for x in 0..new_w {
                let mut accum = vec![0u32; ch];
                let mut count = 0u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = (x * 2 + dx).min(self.width - 1);
                        let sy = (y * 2 + dy).min(self.height - 1);
                        let p = self.get_pixel(sx, sy);
                        for c in 0..ch {
                            accum[c] += p[c] as u32;
                        }
                        count += 1;
                    }
                }
                let pixel: Vec<u8> = accum.iter().map(|a| (a / count) as u8).collect();
                out.set_pixel(x, y, &pixel);
            }
        }
        out
    }
}

/// Result of texture compression.
#[derive(Debug, Clone)]
pub struct CompressedTexture {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// GPU format.
    pub format: GpuTextureFormat,
    /// Mip levels (index 0 = full resolution).
    pub mip_data: Vec<Vec<u8>>,
    /// Total size of all mip levels in bytes.
    pub total_size: usize,
}

/// Generate a complete mipmap chain from a source image.
pub fn generate_mipmap_chain(source: &RawPixels) -> Vec<RawPixels> {
    let mut chain = vec![source.clone()];
    let mut current = source.clone();
    while current.width > 1 || current.height > 1 {
        current = current.downscale_half();
        chain.push(current.clone());
    }
    chain
}

/// Compress a texture according to import settings.
///
/// This is a stub implementation that produces RGBA8 data wrapped
/// in a `CompressedTexture` container. A real implementation would
/// invoke ISPC texcomp, bc7enc, or similar.
pub fn compress_texture(
    source: &RawPixels,
    settings: &TextureImportSettings,
) -> Result<CompressedTexture, ProcessorError> {
    let mut pixels = source.to_rgba();
    if settings.flip_y {
        pixels.flip_vertical();
    }
    if settings.premultiply_alpha {
        pixels.premultiply_alpha();
    }
    // Downscale if necessary
    let mut working = pixels;
    while working.width > settings.max_size || working.height > settings.max_size {
        working = working.downscale_half();
    }
    // Generate mipmaps
    let mip_chain = if settings.generate_mipmaps {
        generate_mipmap_chain(&working)
    } else {
        vec![working.clone()]
    };
    // "Compress" each mip level (stub: store as RGBA8)
    let mut mip_data = Vec::new();
    let mut total_size = 0;
    for mip in &mip_chain {
        let data = encode_to_format(&mip.data, mip.width, mip.height, settings.format)?;
        total_size += data.len();
        mip_data.push(data);
    }
    Ok(CompressedTexture {
        width: working.width,
        height: working.height,
        format: settings.format,
        mip_data,
        total_size,
    })
}

/// Encode raw RGBA data to a specific GPU format.
///
/// Stub: returns the data as-is for all formats (real implementation would
/// use a block compression library).
fn encode_to_format(
    rgba: &[u8],
    width: u32,
    height: u32,
    format: GpuTextureFormat,
) -> Result<Vec<u8>, ProcessorError> {
    match format {
        GpuTextureFormat::Rgba8 => Ok(rgba.to_vec()),
        GpuTextureFormat::Bc1 => {
            // Stub: compute expected output size and fill with placeholder data
            let blocks_x = (width + 3) / 4;
            let blocks_y = (height + 3) / 4;
            let block_count = (blocks_x * blocks_y) as usize;
            let output_size = block_count * 8; // BC1 = 8 bytes per 4x4 block
            let mut output = vec![0u8; output_size];
            // Simple stub: encode each block with average colour
            for by in 0..blocks_y {
                for bx in 0..blocks_x {
                    let mut r_sum = 0u32;
                    let mut g_sum = 0u32;
                    let mut b_sum = 0u32;
                    let mut count = 0u32;
                    for py in 0..4 {
                        for px in 0..4 {
                            let x = bx * 4 + px;
                            let y = by * 4 + py;
                            if x < width && y < height {
                                let idx = ((y * width + x) * 4) as usize;
                                if idx + 2 < rgba.len() {
                                    r_sum += rgba[idx] as u32;
                                    g_sum += rgba[idx + 1] as u32;
                                    b_sum += rgba[idx + 2] as u32;
                                    count += 1;
                                }
                            }
                        }
                    }
                    if count > 0 {
                        let r = (r_sum / count) as u8;
                        let g = (g_sum / count) as u8;
                        let b = (b_sum / count) as u8;
                        let c565 = rgb_to_565(r, g, b);
                        let block_idx = (by * blocks_x + bx) as usize * 8;
                        if block_idx + 7 < output.len() {
                            output[block_idx] = (c565 & 0xFF) as u8;
                            output[block_idx + 1] = (c565 >> 8) as u8;
                            output[block_idx + 2] = (c565 & 0xFF) as u8;
                            output[block_idx + 3] = (c565 >> 8) as u8;
                        }
                    }
                }
            }
            Ok(output)
        }
        GpuTextureFormat::Bc3 => {
            let blocks_x = (width + 3) / 4;
            let blocks_y = (height + 3) / 4;
            let block_count = (blocks_x * blocks_y) as usize;
            let output_size = block_count * 16; // BC3 = 16 bytes per block
            Ok(vec![0u8; output_size])
        }
        GpuTextureFormat::Bc4 => {
            let blocks_x = (width + 3) / 4;
            let blocks_y = (height + 3) / 4;
            let block_count = (blocks_x * blocks_y) as usize;
            Ok(vec![0u8; block_count * 8])
        }
        GpuTextureFormat::Bc5 => {
            let blocks_x = (width + 3) / 4;
            let blocks_y = (height + 3) / 4;
            let block_count = (blocks_x * blocks_y) as usize;
            Ok(vec![0u8; block_count * 16])
        }
        GpuTextureFormat::Bc7 => {
            let blocks_x = (width + 3) / 4;
            let blocks_y = (height + 3) / 4;
            let block_count = (blocks_x * blocks_y) as usize;
            Ok(vec![0u8; block_count * 16])
        }
        GpuTextureFormat::Astc4x4 => {
            let blocks_x = (width + 3) / 4;
            let blocks_y = (height + 3) / 4;
            let block_count = (blocks_x * blocks_y) as usize;
            Ok(vec![0u8; block_count * 16])
        }
        GpuTextureFormat::Astc6x6 => {
            let blocks_x = (width + 5) / 6;
            let blocks_y = (height + 5) / 6;
            let block_count = (blocks_x * blocks_y) as usize;
            Ok(vec![0u8; block_count * 16])
        }
        GpuTextureFormat::Astc8x8 => {
            let blocks_x = (width + 7) / 8;
            let blocks_y = (height + 7) / 8;
            let block_count = (blocks_x * blocks_y) as usize;
            Ok(vec![0u8; block_count * 16])
        }
        GpuTextureFormat::Etc2Rgb => {
            let blocks_x = (width + 3) / 4;
            let blocks_y = (height + 3) / 4;
            let block_count = (blocks_x * blocks_y) as usize;
            Ok(vec![0u8; block_count * 8])
        }
        GpuTextureFormat::Etc2Rgba => {
            let blocks_x = (width + 3) / 4;
            let blocks_y = (height + 3) / 4;
            let block_count = (blocks_x * blocks_y) as usize;
            Ok(vec![0u8; block_count * 16])
        }
    }
}

/// Pack R, G, B into a 16-bit RGB565 value.
fn rgb_to_565(r: u8, g: u8, b: u8) -> u16 {
    let r5 = (r as u16 >> 3) & 0x1F;
    let g6 = (g as u16 >> 2) & 0x3F;
    let b5 = (b as u16 >> 3) & 0x1F;
    (r5 << 11) | (g6 << 5) | b5
}

// ---------------------------------------------------------------------------
// Thumbnail generation
// ---------------------------------------------------------------------------

/// A thumbnail image for the editor.
#[derive(Debug, Clone)]
pub struct Thumbnail {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA8 pixel data.
    pub data: Vec<u8>,
}

/// Generate a thumbnail from a full-size source.
pub fn generate_thumbnail(source: &RawPixels, size: u32) -> Thumbnail {
    let rgba = source.to_rgba();
    let mut current = rgba;
    while current.width > size * 2 || current.height > size * 2 {
        current = current.downscale_half();
    }
    // Final resize to exact thumbnail size (simple nearest-neighbour)
    let mut thumb = RawPixels::new(size, size, 4);
    for y in 0..size {
        for x in 0..size {
            let sx = (x as f32 / size as f32 * current.width as f32) as u32;
            let sy = (y as f32 / size as f32 * current.height as f32) as u32;
            let sx = sx.min(current.width - 1);
            let sy = sy.min(current.height - 1);
            let pixel = current.get_pixel(sx, sy);
            thumb.set_pixel(x, y, pixel);
        }
    }
    Thumbnail {
        width: size,
        height: size,
        data: thumb.data,
    }
}

// ---------------------------------------------------------------------------
// Mesh optimisation
// ---------------------------------------------------------------------------

/// A triangle mesh ready for processing.
#[derive(Debug, Clone)]
pub struct ProcessableMesh {
    /// Vertex positions (3 floats per vertex).
    pub positions: Vec<f32>,
    /// Vertex normals (3 floats per vertex, may be empty).
    pub normals: Vec<f32>,
    /// Texture coordinates (2 floats per vertex, may be empty).
    pub uvs: Vec<f32>,
    /// Tangent vectors (4 floats per vertex, may be empty).
    pub tangents: Vec<f32>,
    /// Triangle indices (3 per triangle).
    pub indices: Vec<u32>,
    /// Bone indices for skinning (4 per vertex, may be empty).
    pub bone_indices: Vec<u32>,
    /// Bone weights for skinning (4 per vertex, may be empty).
    pub bone_weights: Vec<f32>,
}

impl ProcessableMesh {
    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Compute the axis-aligned bounding box: (min, max).
    pub fn compute_aabb(&self) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for i in (0..self.positions.len()).step_by(3) {
            for c in 0..3 {
                min[c] = min[c].min(self.positions[i + c]);
                max[c] = max[c].max(self.positions[i + c]);
            }
        }
        (min, max)
    }
}

/// Result of mesh optimisation.
#[derive(Debug, Clone)]
pub struct OptimisedMesh {
    /// The optimised mesh data.
    pub mesh: ProcessableMesh,
    /// Number of vertices removed by welding.
    pub vertices_welded: usize,
    /// ACMR (average cache miss ratio) before optimisation.
    pub acmr_before: f32,
    /// ACMR after optimisation.
    pub acmr_after: f32,
    /// Overdraw ratio estimate before optimisation.
    pub overdraw_before: f32,
    /// Overdraw ratio estimate after optimisation.
    pub overdraw_after: f32,
    /// Generated LOD meshes (if requested).
    pub lods: Vec<ProcessableMesh>,
}

/// Optimise a mesh for GPU rendering efficiency.
pub fn optimize_mesh(
    mesh: &ProcessableMesh,
    settings: &MeshImportSettings,
) -> Result<OptimisedMesh, ProcessorError> {
    let mut result = mesh.clone();
    let mut vertices_welded = 0;
    let acmr_before = estimate_acmr(&result);
    let overdraw_before = 1.0f32; // placeholder

    // Vertex welding
    if settings.weld_vertices && result.vertex_count() > 0 {
        let (welded, count) = weld_vertices(&result, settings.weld_tolerance);
        vertices_welded = count;
        result = welded;
    }

    // Normal generation
    if settings.generate_normals && result.normals.is_empty() {
        result.normals = generate_flat_normals(&result);
    }

    // Tangent generation (requires normals and UVs)
    if settings.generate_tangents && result.tangents.is_empty()
        && !result.normals.is_empty() && !result.uvs.is_empty()
    {
        result.tangents = generate_tangents(&result);
    }

    // Scale
    if (settings.scale - 1.0).abs() > f32::EPSILON {
        for p in &mut result.positions {
            *p *= settings.scale;
        }
    }

    // Vertex cache optimisation
    if settings.optimize_vertex_cache {
        result.indices = optimize_vertex_cache_order(&result.indices, result.vertex_count());
    }

    // Overdraw optimisation (placeholder)
    let acmr_after = estimate_acmr(&result);
    let overdraw_after = if settings.optimize_overdraw {
        0.95 // placeholder improvement
    } else {
        overdraw_before
    };

    // LOD generation
    let lods = if settings.generate_lods {
        generate_lods(&result, settings.lod_count, settings.lod_reduction)
    } else {
        Vec::new()
    };

    Ok(OptimisedMesh {
        mesh: result,
        vertices_welded,
        acmr_before,
        acmr_after,
        overdraw_before,
        overdraw_after,
        lods,
    })
}

/// Estimate the average cache miss ratio (ACMR) for a given index buffer.
fn estimate_acmr(mesh: &ProcessableMesh) -> f32 {
    if mesh.indices.is_empty() || mesh.vertex_count() == 0 {
        return 0.0;
    }
    // Simple FIFO cache simulation with a 32-entry cache
    let cache_size = 32usize;
    let mut cache: Vec<u32> = Vec::with_capacity(cache_size);
    let mut misses = 0u32;
    for &idx in &mesh.indices {
        if !cache.contains(&idx) {
            misses += 1;
            if cache.len() >= cache_size {
                cache.remove(0);
            }
            cache.push(idx);
        }
    }
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return 0.0;
    }
    misses as f32 / tri_count as f32
}

/// Weld vertices that are within `tolerance` of each other.
fn weld_vertices(mesh: &ProcessableMesh, tolerance: f32) -> (ProcessableMesh, usize) {
    let vert_count = mesh.vertex_count();
    if vert_count == 0 {
        return (mesh.clone(), 0);
    }
    let tol_sq = tolerance * tolerance;
    let mut remap = vec![0u32; vert_count];
    let mut unique_positions: Vec<[f32; 3]> = Vec::new();
    let mut unique_indices: Vec<u32> = Vec::new();

    for i in 0..vert_count {
        let px = mesh.positions[i * 3];
        let py = mesh.positions[i * 3 + 1];
        let pz = mesh.positions[i * 3 + 2];
        let mut found = None;
        for (j, up) in unique_positions.iter().enumerate() {
            let dx = px - up[0];
            let dy = py - up[1];
            let dz = pz - up[2];
            if dx * dx + dy * dy + dz * dz < tol_sq {
                found = Some(j as u32);
                break;
            }
        }
        match found {
            Some(idx) => {
                remap[i] = idx;
            }
            None => {
                remap[i] = unique_positions.len() as u32;
                unique_positions.push([px, py, pz]);
            }
        }
    }

    let welded_count = vert_count - unique_positions.len();
    let new_vert_count = unique_positions.len();
    let mut new_positions = Vec::with_capacity(new_vert_count * 3);
    for p in &unique_positions {
        new_positions.extend_from_slice(p);
    }
    let new_indices: Vec<u32> = mesh.indices.iter().map(|&i| remap[i as usize]).collect();

    // Rebuild normals, UVs, tangents for unique vertices (take first occurrence)
    let mut new_normals = Vec::new();
    let mut new_uvs = Vec::new();
    let mut new_tangents = Vec::new();
    if !mesh.normals.is_empty() {
        new_normals = vec![0.0f32; new_vert_count * 3];
        for i in 0..vert_count {
            let dest = remap[i] as usize;
            if new_normals[dest * 3] == 0.0
                && new_normals[dest * 3 + 1] == 0.0
                && new_normals[dest * 3 + 2] == 0.0
            {
                new_normals[dest * 3] = mesh.normals[i * 3];
                new_normals[dest * 3 + 1] = mesh.normals[i * 3 + 1];
                new_normals[dest * 3 + 2] = mesh.normals[i * 3 + 2];
            }
        }
    }
    if !mesh.uvs.is_empty() {
        new_uvs = vec![0.0f32; new_vert_count * 2];
        for i in 0..vert_count {
            let dest = remap[i] as usize;
            new_uvs[dest * 2] = mesh.uvs[i * 2];
            new_uvs[dest * 2 + 1] = mesh.uvs[i * 2 + 1];
        }
    }
    if !mesh.tangents.is_empty() {
        new_tangents = vec![0.0f32; new_vert_count * 4];
        for i in 0..vert_count {
            let dest = remap[i] as usize;
            for c in 0..4 {
                new_tangents[dest * 4 + c] = mesh.tangents[i * 4 + c];
            }
        }
    }

    (
        ProcessableMesh {
            positions: new_positions,
            normals: new_normals,
            uvs: new_uvs,
            tangents: new_tangents,
            indices: new_indices,
            bone_indices: Vec::new(),
            bone_weights: Vec::new(),
        },
        welded_count,
    )
}

/// Generate flat (face) normals for a mesh.
fn generate_flat_normals(mesh: &ProcessableMesh) -> Vec<f32> {
    let vert_count = mesh.vertex_count();
    let mut normals = vec![0.0f32; vert_count * 3];
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = [
            mesh.positions[i0 * 3],
            mesh.positions[i0 * 3 + 1],
            mesh.positions[i0 * 3 + 2],
        ];
        let p1 = [
            mesh.positions[i1 * 3],
            mesh.positions[i1 * 3 + 1],
            mesh.positions[i1 * 3 + 2],
        ];
        let p2 = [
            mesh.positions[i2 * 3],
            mesh.positions[i2 * 3 + 1],
            mesh.positions[i2 * 3 + 2],
        ];
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];
        for &idx in &[i0, i1, i2] {
            normals[idx * 3] += nx;
            normals[idx * 3 + 1] += ny;
            normals[idx * 3 + 2] += nz;
        }
    }
    // Normalise
    for i in (0..normals.len()).step_by(3) {
        let x = normals[i];
        let y = normals[i + 1];
        let z = normals[i + 2];
        let len = (x * x + y * y + z * z).sqrt();
        if len > 1e-8 {
            normals[i] /= len;
            normals[i + 1] /= len;
            normals[i + 2] /= len;
        }
    }
    normals
}

/// Generate tangent vectors using MikkTSpace-style calculation.
fn generate_tangents(mesh: &ProcessableMesh) -> Vec<f32> {
    let vert_count = mesh.vertex_count();
    let mut tangents = vec![0.0f32; vert_count * 4];
    // Simplified tangent generation
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = [mesh.positions[i0 * 3], mesh.positions[i0 * 3 + 1], mesh.positions[i0 * 3 + 2]];
        let p1 = [mesh.positions[i1 * 3], mesh.positions[i1 * 3 + 1], mesh.positions[i1 * 3 + 2]];
        let p2 = [mesh.positions[i2 * 3], mesh.positions[i2 * 3 + 1], mesh.positions[i2 * 3 + 2]];
        let uv0 = [mesh.uvs[i0 * 2], mesh.uvs[i0 * 2 + 1]];
        let uv1 = [mesh.uvs[i1 * 2], mesh.uvs[i1 * 2 + 1]];
        let uv2 = [mesh.uvs[i2 * 2], mesh.uvs[i2 * 2 + 1]];
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let du1 = uv1[0] - uv0[0];
        let dv1 = uv1[1] - uv0[1];
        let du2 = uv2[0] - uv0[0];
        let dv2 = uv2[1] - uv0[1];
        let det = du1 * dv2 - du2 * dv1;
        if det.abs() < 1e-8 {
            continue;
        }
        let inv_det = 1.0 / det;
        let tx = inv_det * (dv2 * e1[0] - dv1 * e2[0]);
        let ty = inv_det * (dv2 * e1[1] - dv1 * e2[1]);
        let tz = inv_det * (dv2 * e1[2] - dv1 * e2[2]);
        for &idx in &[i0, i1, i2] {
            tangents[idx * 4] += tx;
            tangents[idx * 4 + 1] += ty;
            tangents[idx * 4 + 2] += tz;
            tangents[idx * 4 + 3] = 1.0; // handedness
        }
    }
    // Normalise tangents
    for i in (0..tangents.len()).step_by(4) {
        let x = tangents[i];
        let y = tangents[i + 1];
        let z = tangents[i + 2];
        let len = (x * x + y * y + z * z).sqrt();
        if len > 1e-8 {
            tangents[i] /= len;
            tangents[i + 1] /= len;
            tangents[i + 2] /= len;
        }
    }
    tangents
}

/// Reorder indices for better vertex cache utilisation (Tipsify-like algorithm stub).
fn optimize_vertex_cache_order(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    if indices.is_empty() {
        return Vec::new();
    }
    // Simplified: compute per-vertex valence and emit triangles greedily
    let tri_count = indices.len() / 3;
    let mut valence = vec![0u32; vertex_count];
    for &idx in indices {
        if (idx as usize) < vertex_count {
            valence[idx as usize] += 1;
        }
    }
    // Sort triangles by average valence (ascending) -- rough heuristic
    let mut tris: Vec<usize> = (0..tri_count).collect();
    tris.sort_by(|&a, &b| {
        let avg_a = (0..3)
            .map(|i| valence[indices[a * 3 + i] as usize])
            .sum::<u32>();
        let avg_b = (0..3)
            .map(|i| valence[indices[b * 3 + i] as usize])
            .sum::<u32>();
        avg_a.cmp(&avg_b)
    });
    let mut result = Vec::with_capacity(indices.len());
    for t in tris {
        result.push(indices[t * 3]);
        result.push(indices[t * 3 + 1]);
        result.push(indices[t * 3 + 2]);
    }
    result
}

/// Generate LOD meshes through progressive simplification.
fn generate_lods(
    mesh: &ProcessableMesh,
    lod_count: u32,
    reduction: f32,
) -> Vec<ProcessableMesh> {
    let mut lods = Vec::new();
    let mut current_indices = mesh.indices.clone();
    let mut current_ratio = reduction;
    for _ in 0..lod_count {
        let target_tris = ((current_indices.len() / 3) as f32 * current_ratio) as usize;
        let target_tris = target_tris.max(1);
        // Simple decimation stub: just take the first N triangles
        let take = (target_tris * 3).min(current_indices.len());
        let lod_indices = current_indices[..take].to_vec();
        lods.push(ProcessableMesh {
            positions: mesh.positions.clone(),
            normals: mesh.normals.clone(),
            uvs: mesh.uvs.clone(),
            tangents: mesh.tangents.clone(),
            indices: lod_indices.clone(),
            bone_indices: mesh.bone_indices.clone(),
            bone_weights: mesh.bone_weights.clone(),
        });
        current_indices = lod_indices;
        current_ratio *= reduction;
    }
    lods
}

// ---------------------------------------------------------------------------
// Audio processing
// ---------------------------------------------------------------------------

/// Raw audio data for processing.
#[derive(Debug, Clone)]
pub struct RawAudio {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u32,
    /// Interleaved samples as f32 in [-1, 1].
    pub samples: Vec<f32>,
}

impl RawAudio {
    /// Total number of sample frames (samples / channels).
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.frame_count() as f32 / self.sample_rate as f32
    }

    /// Peak amplitude.
    pub fn peak_amplitude(&self) -> f32 {
        self.samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max)
    }

    /// RMS amplitude.
    pub fn rms_amplitude(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = self.samples.iter().map(|s| s * s).sum();
        (sum_sq / self.samples.len() as f32).sqrt()
    }
}

/// Process audio according to import settings.
pub fn process_audio(
    audio: &RawAudio,
    settings: &AudioImportSettings,
) -> Result<RawAudio, ProcessorError> {
    let mut result = audio.clone();

    // Resample if needed
    if settings.target_sample_rate > 0 && settings.target_sample_rate != audio.sample_rate {
        result = resample_audio(&result, settings.target_sample_rate);
    }

    // Channel conversion
    if settings.target_channels > 0 && settings.target_channels != audio.channels {
        result = convert_channels(&result, settings.target_channels);
    }

    // Trim silence
    if settings.trim_silence {
        result = trim_silence(&result, settings.silence_threshold_db);
    }

    // Normalise
    if let Some(target_db) = settings.normalize_db {
        result = normalize_audio(&result, target_db);
    }

    Ok(result)
}

/// Simple linear resampling.
fn resample_audio(audio: &RawAudio, target_rate: u32) -> RawAudio {
    let ratio = target_rate as f64 / audio.sample_rate as f64;
    let new_frame_count = (audio.frame_count() as f64 * ratio) as usize;
    let ch = audio.channels as usize;
    let mut samples = Vec::with_capacity(new_frame_count * ch);
    for i in 0..new_frame_count {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos.floor() as usize;
        let frac = (src_pos - src_idx as f64) as f32;
        for c in 0..ch {
            let s0 = if src_idx * ch + c < audio.samples.len() {
                audio.samples[src_idx * ch + c]
            } else {
                0.0
            };
            let s1 = if (src_idx + 1) * ch + c < audio.samples.len() {
                audio.samples[(src_idx + 1) * ch + c]
            } else {
                s0
            };
            samples.push(s0 + (s1 - s0) * frac);
        }
    }
    RawAudio {
        sample_rate: target_rate,
        channels: audio.channels,
        samples,
    }
}

/// Convert channel count (mono <-> stereo).
fn convert_channels(audio: &RawAudio, target_channels: u32) -> RawAudio {
    let frames = audio.frame_count();
    let src_ch = audio.channels as usize;
    let dst_ch = target_channels as usize;
    let mut samples = Vec::with_capacity(frames * dst_ch);
    for f in 0..frames {
        let base = f * src_ch;
        if dst_ch == 1 {
            // Mix down to mono
            let mut sum = 0.0f32;
            for c in 0..src_ch {
                sum += audio.samples[base + c];
            }
            samples.push(sum / src_ch as f32);
        } else if dst_ch == 2 && src_ch == 1 {
            // Mono to stereo: duplicate
            let s = audio.samples[base];
            samples.push(s);
            samples.push(s);
        } else {
            // General case: copy available channels, zero-fill the rest
            for c in 0..dst_ch {
                if c < src_ch {
                    samples.push(audio.samples[base + c]);
                } else {
                    samples.push(0.0);
                }
            }
        }
    }
    RawAudio {
        sample_rate: audio.sample_rate,
        channels: target_channels,
        samples,
    }
}

/// Trim leading and trailing silence.
fn trim_silence(audio: &RawAudio, threshold_db: f32) -> RawAudio {
    let threshold_linear = 10.0f32.powf(threshold_db / 20.0);
    let ch = audio.channels as usize;
    let frames = audio.frame_count();
    // Find first non-silent frame
    let start = (0..frames)
        .find(|&f| {
            (0..ch).any(|c| audio.samples[f * ch + c].abs() > threshold_linear)
        })
        .unwrap_or(0);
    // Find last non-silent frame
    let end = (0..frames)
        .rev()
        .find(|&f| {
            (0..ch).any(|c| audio.samples[f * ch + c].abs() > threshold_linear)
        })
        .map(|f| f + 1)
        .unwrap_or(frames);
    let samples = audio.samples[start * ch..end * ch].to_vec();
    RawAudio {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        samples,
    }
}

/// Normalise audio to a target peak level in dB.
fn normalize_audio(audio: &RawAudio, target_db: f32) -> RawAudio {
    let peak = audio.peak_amplitude();
    if peak < 1e-8 {
        return audio.clone();
    }
    let target_linear = 10.0f32.powf(target_db / 20.0);
    let gain = target_linear / peak;
    let samples: Vec<f32> = audio.samples.iter().map(|s| (s * gain).clamp(-1.0, 1.0)).collect();
    RawAudio {
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        samples,
    }
}

// ---------------------------------------------------------------------------
// Batch processor
// ---------------------------------------------------------------------------

/// Tracks the state of a batch processing job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchJobState {
    /// Job is queued and waiting.
    Pending,
    /// Job is currently being processed.
    InProgress,
    /// Job completed successfully.
    Completed,
    /// Job failed.
    Failed,
    /// Job was cancelled.
    Cancelled,
}

/// A single item in a batch processing run.
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Source file path.
    pub source_path: PathBuf,
    /// Output file path.
    pub output_path: PathBuf,
    /// Asset type identifier.
    pub asset_type: String,
    /// Current state.
    pub state: BatchJobState,
    /// Error message if failed.
    pub error: Option<String>,
    /// Processing time.
    pub duration: Option<Duration>,
}

/// Aggregated batch processing statistics.
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Total items in the batch.
    pub total: usize,
    /// Items completed successfully.
    pub completed: usize,
    /// Items that failed.
    pub failed: usize,
    /// Items cancelled.
    pub cancelled: usize,
    /// Items still pending or in progress.
    pub remaining: usize,
    /// Total processing time.
    pub elapsed: Duration,
    /// Average time per item.
    pub avg_item_time: Duration,
    /// Total input size in bytes.
    pub total_input_bytes: u64,
    /// Total output size in bytes.
    pub total_output_bytes: u64,
}

/// Manages batch processing of multiple assets.
pub struct BatchProcessor {
    /// Items to process.
    items: Vec<BatchItem>,
    /// Whether processing has been cancelled.
    cancelled: bool,
    /// Start time of the batch run.
    start_time: Option<Instant>,
    /// Texture import settings to use.
    pub texture_settings: TextureImportSettings,
    /// Mesh import settings to use.
    pub mesh_settings: MeshImportSettings,
    /// Audio import settings to use.
    pub audio_settings: AudioImportSettings,
    /// Total input bytes processed.
    total_input_bytes: u64,
    /// Total output bytes produced.
    total_output_bytes: u64,
}

impl BatchProcessor {
    /// Create a new batch processor.
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            cancelled: false,
            start_time: None,
            texture_settings: TextureImportSettings::default(),
            mesh_settings: MeshImportSettings::default(),
            audio_settings: AudioImportSettings::default(),
            total_input_bytes: 0,
            total_output_bytes: 0,
        }
    }

    /// Add an item to the batch.
    pub fn add_item(
        &mut self,
        source_path: PathBuf,
        output_path: PathBuf,
        asset_type: impl Into<String>,
    ) {
        self.items.push(BatchItem {
            source_path,
            output_path,
            asset_type: asset_type.into(),
            state: BatchJobState::Pending,
            error: None,
            duration: None,
        });
    }

    /// Number of items in the batch.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Get current processing statistics.
    pub fn stats(&self) -> BatchStats {
        let completed = self.items.iter().filter(|i| i.state == BatchJobState::Completed).count();
        let failed = self.items.iter().filter(|i| i.state == BatchJobState::Failed).count();
        let cancelled = self.items.iter().filter(|i| i.state == BatchJobState::Cancelled).count();
        let remaining = self.items.len() - completed - failed - cancelled;
        let elapsed = self
            .start_time
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);
        let avg_item_time = if completed > 0 {
            elapsed / completed as u32
        } else {
            Duration::ZERO
        };
        BatchStats {
            total: self.items.len(),
            completed,
            failed,
            cancelled,
            remaining,
            elapsed,
            avg_item_time,
            total_input_bytes: self.total_input_bytes,
            total_output_bytes: self.total_output_bytes,
        }
    }

    /// Cancel the batch run.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        for item in &mut self.items {
            if item.state == BatchJobState::Pending {
                item.state = BatchJobState::Cancelled;
            }
        }
    }

    /// Process all pending items.
    pub fn process_all(&mut self) {
        self.start_time = Some(Instant::now());
        for i in 0..self.items.len() {
            if self.cancelled {
                break;
            }
            if self.items[i].state != BatchJobState::Pending {
                continue;
            }
            self.items[i].state = BatchJobState::InProgress;
            let start = Instant::now();
            // Processing is a stub: mark as completed
            self.items[i].state = BatchJobState::Completed;
            self.items[i].duration = Some(start.elapsed());
        }
    }

    /// Get a reference to the items.
    pub fn items(&self) -> &[BatchItem] {
        &self.items
    }

    /// Progress as a fraction [0, 1].
    pub fn progress(&self) -> f32 {
        if self.items.is_empty() {
            return 1.0;
        }
        let done = self
            .items
            .iter()
            .filter(|i| matches!(i.state, BatchJobState::Completed | BatchJobState::Failed | BatchJobState::Cancelled))
            .count();
        done as f32 / self.items.len() as f32
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_pixels_basic() {
        let mut pixels = RawPixels::new(4, 4, 4);
        pixels.set_pixel(0, 0, &[255, 0, 0, 255]);
        let p = pixels.get_pixel(0, 0);
        assert_eq!(p, &[255, 0, 0, 255]);
    }

    #[test]
    fn test_raw_pixels_flip_vertical() {
        let mut pixels = RawPixels::new(2, 2, 1);
        pixels.set_pixel(0, 0, &[1]);
        pixels.set_pixel(0, 1, &[2]);
        pixels.flip_vertical();
        assert_eq!(pixels.get_pixel(0, 0), &[2]);
        assert_eq!(pixels.get_pixel(0, 1), &[1]);
    }

    #[test]
    fn test_to_rgba() {
        let mut rgb = RawPixels::new(1, 1, 3);
        rgb.set_pixel(0, 0, &[100, 150, 200]);
        let rgba = rgb.to_rgba();
        assert_eq!(rgba.get_pixel(0, 0), &[100, 150, 200, 255]);
    }

    #[test]
    fn test_downscale_half() {
        let mut pixels = RawPixels::new(4, 4, 1);
        for y in 0..4 {
            for x in 0..4 {
                pixels.set_pixel(x, y, &[100]);
            }
        }
        let half = pixels.downscale_half();
        assert_eq!(half.width, 2);
        assert_eq!(half.height, 2);
        assert_eq!(half.get_pixel(0, 0), &[100]);
    }

    #[test]
    fn test_mipmap_chain() {
        let pixels = RawPixels::new(16, 16, 4);
        let chain = generate_mipmap_chain(&pixels);
        // 16, 8, 4, 2, 1 = 5 levels
        assert_eq!(chain.len(), 5);
        assert_eq!(chain[0].width, 16);
        assert_eq!(chain[1].width, 8);
        assert_eq!(chain[4].width, 1);
    }

    #[test]
    fn test_texture_compression_rgba8() {
        let pixels = RawPixels::new(8, 8, 4);
        let settings = TextureImportSettings {
            format: GpuTextureFormat::Rgba8,
            generate_mipmaps: false,
            ..Default::default()
        };
        let result = compress_texture(&pixels, &settings).unwrap();
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        assert_eq!(result.mip_data.len(), 1);
    }

    #[test]
    fn test_texture_compression_bc1() {
        let pixels = RawPixels::new(8, 8, 4);
        let settings = TextureImportSettings {
            format: GpuTextureFormat::Bc1,
            generate_mipmaps: false,
            ..Default::default()
        };
        let result = compress_texture(&pixels, &settings).unwrap();
        // 8x8 => 2x2 blocks => 4 blocks * 8 bytes = 32
        assert_eq!(result.mip_data[0].len(), 32);
    }

    #[test]
    fn test_thumbnail_generation() {
        let pixels = RawPixels::new(256, 256, 4);
        let thumb = generate_thumbnail(&pixels, 64);
        assert_eq!(thumb.width, 64);
        assert_eq!(thumb.height, 64);
        assert_eq!(thumb.data.len(), 64 * 64 * 4);
    }

    #[test]
    fn test_mesh_normal_generation() {
        // Simple triangle
        let mesh = ProcessableMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            normals: vec![],
            uvs: vec![],
            tangents: vec![],
            indices: vec![0, 1, 2],
            bone_indices: vec![],
            bone_weights: vec![],
        };
        let normals = generate_flat_normals(&mesh);
        assert_eq!(normals.len(), 9);
        // Normal should point in +Z direction
        assert!(normals[2] > 0.0);
    }

    #[test]
    fn test_vertex_welding() {
        let mesh = ProcessableMesh {
            positions: vec![
                0.0, 0.0, 0.0,
                0.00001, 0.0, 0.0, // should weld to vertex 0
                1.0, 0.0, 0.0,
            ],
            normals: vec![],
            uvs: vec![],
            tangents: vec![],
            indices: vec![0, 1, 2],
            bone_indices: vec![],
            bone_weights: vec![],
        };
        let (welded, count) = weld_vertices(&mesh, 0.001);
        assert_eq!(count, 1);
        assert_eq!(welded.vertex_count(), 2);
    }

    #[test]
    fn test_audio_resample() {
        let audio = RawAudio {
            sample_rate: 44100,
            channels: 1,
            samples: vec![0.5; 44100], // 1 second
        };
        let resampled = resample_audio(&audio, 22050);
        assert_eq!(resampled.sample_rate, 22050);
        assert_eq!(resampled.frame_count(), 22050);
    }

    #[test]
    fn test_audio_normalize() {
        let audio = RawAudio {
            sample_rate: 44100,
            channels: 1,
            samples: vec![0.25, -0.25, 0.1, -0.1],
        };
        let normalized = normalize_audio(&audio, 0.0); // 0 dB = 1.0 linear
        assert!((normalized.peak_amplitude() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_processor_basic() {
        let mut batch = BatchProcessor::new();
        batch.add_item(
            PathBuf::from("input/tex.png"),
            PathBuf::from("output/tex.dds"),
            "texture",
        );
        batch.add_item(
            PathBuf::from("input/mesh.obj"),
            PathBuf::from("output/mesh.bin"),
            "mesh",
        );
        assert_eq!(batch.item_count(), 2);
        batch.process_all();
        assert_eq!(batch.progress(), 1.0);
        let stats = batch.stats();
        assert_eq!(stats.completed, 2);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_batch_cancel() {
        let mut batch = BatchProcessor::new();
        for i in 0..10 {
            batch.add_item(
                PathBuf::from(format!("in/{i}.png")),
                PathBuf::from(format!("out/{i}.dds")),
                "texture",
            );
        }
        batch.cancel();
        batch.process_all();
        let stats = batch.stats();
        assert_eq!(stats.cancelled, 10);
    }

    #[test]
    fn test_gpu_format_properties() {
        assert_eq!(GpuTextureFormat::Bc1.bits_per_pixel(), 4.0);
        assert_eq!(GpuTextureFormat::Bc3.block_size(), (4, 4));
        assert!(GpuTextureFormat::Bc3.has_alpha());
        assert!(!GpuTextureFormat::Bc1.has_alpha());
        assert_eq!(GpuTextureFormat::Astc8x8.block_size(), (8, 8));
    }

    #[test]
    fn test_texture_settings_validation() {
        let mut settings = TextureImportSettings::default();
        assert!(settings.validate().is_empty());
        settings.max_size = 3; // not power of two
        assert!(!settings.validate().is_empty());
        settings.max_size = 4096;
        settings.is_normal_map = true;
        settings.srgb = true;
        let issues = settings.validate();
        assert!(issues.iter().any(|i| i.contains("normal maps")));
    }

    #[test]
    fn test_mesh_acmr_estimation() {
        let mesh = ProcessableMesh {
            positions: vec![0.0; 30],
            normals: vec![],
            uvs: vec![],
            tangents: vec![],
            indices: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            bone_indices: vec![],
            bone_weights: vec![],
        };
        let acmr = estimate_acmr(&mesh);
        assert!(acmr > 0.0);
    }

    #[test]
    fn test_audio_channel_conversion() {
        let stereo = RawAudio {
            sample_rate: 44100,
            channels: 2,
            samples: vec![0.5, 0.3, 0.5, 0.3],
        };
        let mono = convert_channels(&stereo, 1);
        assert_eq!(mono.channels, 1);
        assert_eq!(mono.frame_count(), 2);
    }

    #[test]
    fn test_rgb_to_565() {
        // White
        let white = rgb_to_565(255, 255, 255);
        assert_eq!(white, 0xFFFF);
        // Black
        let black = rgb_to_565(0, 0, 0);
        assert_eq!(black, 0x0000);
    }

    #[test]
    fn test_mesh_aabb() {
        let mesh = ProcessableMesh {
            positions: vec![-1.0, -2.0, -3.0, 4.0, 5.0, 6.0],
            normals: vec![],
            uvs: vec![],
            tangents: vec![],
            indices: vec![0, 1, 0],
            bone_indices: vec![],
            bone_weights: vec![],
        };
        let (min, max) = mesh.compute_aabb();
        assert_eq!(min, [-1.0, -2.0, -3.0]);
        assert_eq!(max, [4.0, 5.0, 6.0]);
    }
}
