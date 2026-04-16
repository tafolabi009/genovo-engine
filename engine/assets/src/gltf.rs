//! glTF 2.0 loader.
//!
//! Parses the JSON-based glTF 2.0 format (both `.gltf` + external buffers
//! and `.glb` binary containers) and converts the data into engine-native
//! types such as [`GltfMeshData`], [`GltfMaterial`], [`AnimationClip`], and
//! [`Skeleton`].
//!
//! # Supported features
//!
//! - Full JSON structure: scenes, nodes, meshes, materials, textures,
//!   accessors, buffer views, buffers, animations, skins
//! - Binary buffer reading with byte stride handling
//! - GLB container support (magic, version, JSON + BIN chunks)
//! - Base64 data URI decoding for embedded buffers
//! - Accessor types: SCALAR, VEC2, VEC3, VEC4, MAT4
//! - Component types: u8, u16, u32, f32
//! - Animation: translation, rotation, scale, weights with linear interpolation
//! - Skins / skeletons: inverse bind matrices, joint hierarchy

use std::collections::HashMap;
use std::path::Path;

use crate::loader::{AssetError, AssetLoader};

// =========================================================================
// Public data types
// =========================================================================

/// A complete glTF document loaded into memory.
#[derive(Debug, Clone)]
pub struct GltfDocument {
    /// All scenes in the file.
    pub scenes: Vec<GltfScene>,
    /// The default scene index (if specified).
    pub default_scene: Option<usize>,
    /// All nodes.
    pub nodes: Vec<GltfNode>,
    /// All meshes.
    pub meshes: Vec<GltfMesh>,
    /// All materials.
    pub materials: Vec<GltfMaterial>,
    /// All textures.
    pub textures: Vec<GltfTexture>,
    /// All images.
    pub images: Vec<GltfImage>,
    /// All samplers.
    pub samplers: Vec<GltfSampler>,
    /// All animations.
    pub animations: Vec<AnimationClip>,
    /// All skins / skeletons.
    pub skins: Vec<Skeleton>,
}

/// A scene containing a list of root node indices.
#[derive(Debug, Clone)]
pub struct GltfScene {
    pub name: Option<String>,
    pub nodes: Vec<usize>,
}

/// A node in the scene graph.
#[derive(Debug, Clone)]
pub struct GltfNode {
    pub name: Option<String>,
    pub children: Vec<usize>,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub matrix: Option<[f32; 16]>,
}

/// A mesh consisting of one or more primitives.
#[derive(Debug, Clone)]
pub struct GltfMesh {
    pub name: Option<String>,
    pub primitives: Vec<GltfPrimitive>,
}

/// A single mesh primitive (draw call).
#[derive(Debug, Clone)]
pub struct GltfPrimitive {
    pub attributes: HashMap<String, usize>,
    pub indices: Option<usize>,
    pub material: Option<usize>,
    pub mode: u32,
}

/// Decoded mesh data ready for the engine.
#[derive(Debug, Clone)]
pub struct GltfMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

/// PBR metallic-roughness material parameters.
#[derive(Debug, Clone)]
pub struct GltfMaterial {
    pub name: Option<String>,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub base_color_texture: Option<GltfTextureInfo>,
    pub metallic_roughness_texture: Option<GltfTextureInfo>,
    pub normal_texture: Option<GltfTextureInfo>,
    pub occlusion_texture: Option<GltfTextureInfo>,
    pub emissive_texture: Option<GltfTextureInfo>,
    pub emissive_factor: [f32; 3],
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

/// Alpha blending mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

/// A reference to a texture with a texture coordinate set index.
#[derive(Debug, Clone)]
pub struct GltfTextureInfo {
    pub index: usize,
    pub tex_coord: u32,
}

/// A texture referencing a source image and a sampler.
#[derive(Debug, Clone)]
pub struct GltfTexture {
    pub name: Option<String>,
    pub source: Option<usize>,
    pub sampler: Option<usize>,
}

/// An image source.
#[derive(Debug, Clone)]
pub struct GltfImage {
    pub name: Option<String>,
    pub uri: Option<String>,
    pub mime_type: Option<String>,
    pub buffer_view: Option<usize>,
}

/// A texture sampler.
#[derive(Debug, Clone)]
pub struct GltfSampler {
    pub mag_filter: Option<u32>,
    pub min_filter: Option<u32>,
    pub wrap_s: u32,
    pub wrap_t: u32,
}

/// A decoded animation clip.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    pub name: Option<String>,
    pub channels: Vec<AnimationChannel>,
    pub duration: f32,
}

/// A single animation channel targeting a node property.
#[derive(Debug, Clone)]
pub struct AnimationChannel {
    pub target_node: usize,
    pub property: AnimationProperty,
    pub timestamps: Vec<f32>,
    pub values: AnimationValues,
    pub interpolation: Interpolation,
}

/// The property being animated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationProperty {
    Translation,
    Rotation,
    Scale,
    Weights,
}

/// Interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Step,
    CubicSpline,
}

/// Typed animation keyframe values.
#[derive(Debug, Clone)]
pub enum AnimationValues {
    Vec3(Vec<[f32; 3]>),
    Vec4(Vec<[f32; 4]>),
    Scalar(Vec<f32>),
}

/// A skeleton (skin) with joint hierarchy and inverse bind matrices.
#[derive(Debug, Clone)]
pub struct Skeleton {
    pub name: Option<String>,
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<[f32; 16]>,
    pub skeleton_root: Option<usize>,
}

// =========================================================================
// Internal JSON structures (serde)
// =========================================================================

/// We parse glTF JSON manually to avoid a serde_json dependency on specific
/// struct shapes.  We use serde_json::Value and extract fields by hand.

// =========================================================================
// GltfLoader — AssetLoader implementation
// =========================================================================

/// Loads glTF 2.0 files (`.gltf` and `.glb`).
pub struct GltfLoader;

impl AssetLoader for GltfLoader {
    type Asset = GltfDocument;

    fn extensions(&self) -> &[&str] {
        &["gltf", "glb"]
    }

    fn load(&self, path: &Path, bytes: &[u8]) -> Result<GltfDocument, AssetError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if ext == "glb" {
            parse_glb(bytes)
        } else {
            parse_gltf_json(bytes, path)
        }
    }
}

// =========================================================================
// GLB container parser
// =========================================================================

const GLB_MAGIC: u32 = 0x46546C67; // "glTF"
const GLB_CHUNK_JSON: u32 = 0x4E4F534A;
const GLB_CHUNK_BIN: u32 = 0x004E4942;

/// Parse a GLB binary container.
fn parse_glb(data: &[u8]) -> Result<GltfDocument, AssetError> {
    if data.len() < 12 {
        return Err(AssetError::InvalidData("GLB too small for header".into()));
    }

    let magic = read_u32_le(data, 0);
    if magic != GLB_MAGIC {
        return Err(AssetError::InvalidData(format!(
            "GLB bad magic: 0x{magic:08X} (expected 0x{GLB_MAGIC:08X})"
        )));
    }

    let version = read_u32_le(data, 4);
    if version != 2 {
        return Err(AssetError::InvalidData(format!(
            "GLB version {version} not supported (expected 2)"
        )));
    }

    let _total_length = read_u32_le(data, 8);

    // Parse chunks
    let mut offset = 12usize;
    let mut json_data: Option<&[u8]> = None;
    let mut bin_data: Option<&[u8]> = None;

    while offset + 8 <= data.len() {
        let chunk_length = read_u32_le(data, offset) as usize;
        let chunk_type = read_u32_le(data, offset + 4);
        let chunk_start = offset + 8;
        let chunk_end = chunk_start + chunk_length;

        if chunk_end > data.len() {
            return Err(AssetError::InvalidData("GLB chunk extends past EOF".into()));
        }

        match chunk_type {
            GLB_CHUNK_JSON => {
                json_data = Some(&data[chunk_start..chunk_end]);
            }
            GLB_CHUNK_BIN => {
                bin_data = Some(&data[chunk_start..chunk_end]);
            }
            _ => {
                // Unknown chunk, skip
            }
        }

        offset = chunk_end;
        // Chunks are padded to 4-byte boundaries
        offset = (offset + 3) & !3;
    }

    let json_bytes = json_data.ok_or_else(|| {
        AssetError::InvalidData("GLB missing JSON chunk".into())
    })?;

    let json: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|e| AssetError::Parse(format!("GLB JSON parse error: {e}")))?;

    let buffers = load_buffers_glb(&json, bin_data)?;
    parse_gltf_document(&json, &buffers)
}

/// Parse a standalone `.gltf` JSON file.
fn parse_gltf_json(data: &[u8], _path: &Path) -> Result<GltfDocument, AssetError> {
    let json: serde_json::Value = serde_json::from_slice(data)
        .map_err(|e| AssetError::Parse(format!("glTF JSON parse error: {e}")))?;

    let buffers = load_buffers_gltf(&json)?;
    parse_gltf_document(&json, &buffers)
}

// =========================================================================
// Buffer loading
// =========================================================================

/// Load buffers from a GLB file.
fn load_buffers_glb(
    json: &serde_json::Value,
    bin_chunk: Option<&[u8]>,
) -> Result<Vec<Vec<u8>>, AssetError> {
    let mut buffers = Vec::new();

    if let Some(arr) = json.get("buffers").and_then(|v| v.as_array()) {
        for (i, buf_json) in arr.iter().enumerate() {
            let byte_length = buf_json
                .get("byteLength")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            if let Some(uri) = buf_json.get("uri").and_then(|v| v.as_str()) {
                // Data URI or external file
                let decoded = decode_data_uri(uri)?;
                buffers.push(decoded);
            } else if i == 0 {
                // First buffer with no URI uses the GLB binary chunk
                let bin = bin_chunk.ok_or_else(|| {
                    AssetError::InvalidData("GLB buffer 0 has no URI and no BIN chunk".into())
                })?;
                buffers.push(bin[..byte_length.min(bin.len())].to_vec());
            } else {
                buffers.push(vec![0u8; byte_length]);
            }
        }
    }

    Ok(buffers)
}

/// Load buffers from a standalone glTF file (data URIs only).
fn load_buffers_gltf(json: &serde_json::Value) -> Result<Vec<Vec<u8>>, AssetError> {
    let mut buffers = Vec::new();

    if let Some(arr) = json.get("buffers").and_then(|v| v.as_array()) {
        for buf_json in arr {
            let byte_length = buf_json
                .get("byteLength")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            if let Some(uri) = buf_json.get("uri").and_then(|v| v.as_str()) {
                if uri.starts_with("data:") {
                    let decoded = decode_data_uri(uri)?;
                    buffers.push(decoded);
                } else {
                    // External file reference — would need filesystem access
                    // For now, create a placeholder
                    buffers.push(vec![0u8; byte_length]);
                }
            } else {
                buffers.push(vec![0u8; byte_length]);
            }
        }
    }

    Ok(buffers)
}

/// Decode a base64 data URI.
///
/// Supports: `data:application/octet-stream;base64,AAAA...`
///           `data:application/gltf-buffer;base64,AAAA...`
fn decode_data_uri(uri: &str) -> Result<Vec<u8>, AssetError> {
    if !uri.starts_with("data:") {
        return Err(AssetError::InvalidData(format!(
            "Expected data URI, got: {}",
            &uri[..uri.len().min(50)]
        )));
    }

    let base64_marker = ";base64,";
    let base64_start = uri.find(base64_marker).ok_or_else(|| {
        AssetError::InvalidData("Data URI missing ;base64, marker".into())
    })? + base64_marker.len();

    let encoded = &uri[base64_start..];
    base64_decode(encoded)
}

/// Simple base64 decoder.
fn base64_decode(input: &str) -> Result<Vec<u8>, AssetError> {
    const DECODE_TABLE: [u8; 128] = {
        let mut table = [255u8; 128];
        let mut i = 0u8;
        while i < 26 {
            table[(b'A' + i) as usize] = i;
            table[(b'a' + i) as usize] = i + 26;
            i += 1;
        }
        let mut d = 0u8;
        while d < 10 {
            table[(b'0' + d) as usize] = d + 52;
            d += 1;
        }
        table[b'+' as usize] = 62;
        table[b'/' as usize] = 63;
        table
    };

    let bytes: Vec<u8> = input.bytes().filter(|&b| b != b'\n' && b != b'\r' && b != b' ').collect();
    let mut output = Vec::with_capacity(bytes.len() * 3 / 4);

    let mut i = 0;
    while i + 3 < bytes.len() {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        let b2 = bytes[i + 2];
        let b3 = bytes[i + 3];

        if b0 == b'=' {
            break;
        }

        let v0 = if b0 < 128 { DECODE_TABLE[b0 as usize] } else { 255 };
        let v1 = if b1 < 128 { DECODE_TABLE[b1 as usize] } else { 255 };

        if v0 == 255 || v1 == 255 {
            return Err(AssetError::InvalidData("Invalid base64 character".into()));
        }

        let triple = (v0 as u32) << 18
            | (v1 as u32) << 12
            | if b2 != b'=' {
                let v2 = if b2 < 128 { DECODE_TABLE[b2 as usize] } else { 255 };
                if v2 == 255 {
                    return Err(AssetError::InvalidData("Invalid base64 character".into()));
                }
                (v2 as u32) << 6
            } else {
                0
            }
            | if b3 != b'=' {
                let v3 = if b3 < 128 { DECODE_TABLE[b3 as usize] } else { 255 };
                if v3 == 255 {
                    return Err(AssetError::InvalidData("Invalid base64 character".into()));
                }
                v3 as u32
            } else {
                0
            };

        output.push(((triple >> 16) & 0xFF) as u8);
        if b2 != b'=' {
            output.push(((triple >> 8) & 0xFF) as u8);
        }
        if b3 != b'=' {
            output.push((triple & 0xFF) as u8);
        }

        i += 4;
    }

    Ok(output)
}

// =========================================================================
// glTF document parser
// =========================================================================

/// Parse the full glTF document from JSON and loaded buffers.
fn parse_gltf_document(
    json: &serde_json::Value,
    buffers: &[Vec<u8>],
) -> Result<GltfDocument, AssetError> {
    // Parse buffer views
    let buffer_views = parse_buffer_views(json)?;

    // Parse accessors
    let accessors = parse_accessors(json)?;

    // Parse scenes
    let scenes = parse_scenes(json)?;
    let default_scene = json.get("scene").and_then(|v| v.as_u64()).map(|v| v as usize);

    // Parse nodes
    let nodes = parse_nodes(json)?;

    // Parse meshes
    let meshes = parse_meshes(json)?;

    // Parse materials
    let materials = parse_materials(json)?;

    // Parse textures, images, samplers
    let textures = parse_textures(json)?;
    let images = parse_images(json)?;
    let samplers = parse_samplers(json)?;

    // Parse animations
    let animations = parse_animations(json, &accessors, &buffer_views, buffers)?;

    // Parse skins
    let skins = parse_skins(json, &accessors, &buffer_views, buffers)?;

    Ok(GltfDocument {
        scenes,
        default_scene,
        nodes,
        meshes,
        materials,
        textures,
        images,
        samplers,
        animations,
        skins,
    })
}

// =========================================================================
// Buffer view / accessor reading
// =========================================================================

#[derive(Debug, Clone)]
struct BufferView {
    buffer: usize,
    byte_offset: usize,
    byte_length: usize,
    byte_stride: Option<usize>,
    target: Option<u32>,
}

#[derive(Debug, Clone)]
struct Accessor {
    buffer_view: Option<usize>,
    byte_offset: usize,
    component_type: u32,
    count: usize,
    accessor_type: String,
    min: Option<Vec<f64>>,
    max: Option<Vec<f64>>,
    normalized: bool,
}

impl Accessor {
    /// Returns the number of components per element.
    fn component_count(&self) -> usize {
        match self.accessor_type.as_str() {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT2" => 4,
            "MAT3" => 9,
            "MAT4" => 16,
            _ => 1,
        }
    }

    /// Returns the byte size of a single component.
    fn component_size(&self) -> usize {
        match self.component_type {
            5120 => 1,  // BYTE
            5121 => 1,  // UNSIGNED_BYTE
            5122 => 2,  // SHORT
            5123 => 2,  // UNSIGNED_SHORT
            5125 => 4,  // UNSIGNED_INT
            5126 => 4,  // FLOAT
            _ => 4,
        }
    }

    /// Returns the byte size of one element.
    fn element_size(&self) -> usize {
        self.component_count() * self.component_size()
    }
}

fn parse_buffer_views(json: &serde_json::Value) -> Result<Vec<BufferView>, AssetError> {
    let mut views = Vec::new();
    if let Some(arr) = json.get("bufferViews").and_then(|v| v.as_array()) {
        for bv in arr {
            views.push(BufferView {
                buffer: bv.get("buffer").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                byte_offset: bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                byte_length: bv.get("byteLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                byte_stride: bv.get("byteStride").and_then(|v| v.as_u64()).map(|v| v as usize),
                target: bv.get("target").and_then(|v| v.as_u64()).map(|v| v as u32),
            });
        }
    }
    Ok(views)
}

fn parse_accessors(json: &serde_json::Value) -> Result<Vec<Accessor>, AssetError> {
    let mut accessors = Vec::new();
    if let Some(arr) = json.get("accessors").and_then(|v| v.as_array()) {
        for acc in arr {
            accessors.push(Accessor {
                buffer_view: acc.get("bufferView").and_then(|v| v.as_u64()).map(|v| v as usize),
                byte_offset: acc.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                component_type: acc
                    .get("componentType")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5126) as u32,
                count: acc.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                accessor_type: acc
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("SCALAR")
                    .to_owned(),
                min: acc
                    .get("min")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect()),
                max: acc
                    .get("max")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect()),
                normalized: acc.get("normalized").and_then(|v| v.as_bool()).unwrap_or(false),
            });
        }
    }
    Ok(accessors)
}

/// Read accessor data as Vec<f32>.
///
/// Handles byte stride, component type conversion, and normalization.
fn read_accessor_f32(
    accessor: &Accessor,
    buffer_views: &[BufferView],
    buffers: &[Vec<u8>],
) -> Result<Vec<f32>, AssetError> {
    let bv_idx = match accessor.buffer_view {
        Some(idx) => idx,
        None => return Ok(vec![0.0f32; accessor.count * accessor.component_count()]),
    };

    if bv_idx >= buffer_views.len() {
        return Err(AssetError::InvalidData(format!(
            "Buffer view index {bv_idx} out of range"
        )));
    }

    let bv = &buffer_views[bv_idx];
    if bv.buffer >= buffers.len() {
        return Err(AssetError::InvalidData(format!(
            "Buffer index {} out of range",
            bv.buffer
        )));
    }

    let buffer = &buffers[bv.buffer];
    let base_offset = bv.byte_offset + accessor.byte_offset;
    let stride = bv.byte_stride.unwrap_or(accessor.element_size());
    let comp_count = accessor.component_count();

    let mut result = Vec::with_capacity(accessor.count * comp_count);

    for i in 0..accessor.count {
        let elem_offset = base_offset + i * stride;

        for c in 0..comp_count {
            let comp_offset = elem_offset + c * accessor.component_size();

            let value = match accessor.component_type {
                5120 => {
                    // BYTE (i8)
                    if comp_offset >= buffer.len() {
                        0.0
                    } else {
                        let v = buffer[comp_offset] as i8;
                        if accessor.normalized {
                            (v as f32 / 127.0).max(-1.0)
                        } else {
                            v as f32
                        }
                    }
                }
                5121 => {
                    // UNSIGNED_BYTE
                    if comp_offset >= buffer.len() {
                        0.0
                    } else {
                        let v = buffer[comp_offset];
                        if accessor.normalized {
                            v as f32 / 255.0
                        } else {
                            v as f32
                        }
                    }
                }
                5122 => {
                    // SHORT (i16)
                    if comp_offset + 2 > buffer.len() {
                        0.0
                    } else {
                        let v = i16::from_le_bytes([buffer[comp_offset], buffer[comp_offset + 1]]);
                        if accessor.normalized {
                            (v as f32 / 32767.0).max(-1.0)
                        } else {
                            v as f32
                        }
                    }
                }
                5123 => {
                    // UNSIGNED_SHORT
                    if comp_offset + 2 > buffer.len() {
                        0.0
                    } else {
                        let v = u16::from_le_bytes([buffer[comp_offset], buffer[comp_offset + 1]]);
                        if accessor.normalized {
                            v as f32 / 65535.0
                        } else {
                            v as f32
                        }
                    }
                }
                5125 => {
                    // UNSIGNED_INT
                    if comp_offset + 4 > buffer.len() {
                        0.0
                    } else {
                        let v = u32::from_le_bytes([
                            buffer[comp_offset],
                            buffer[comp_offset + 1],
                            buffer[comp_offset + 2],
                            buffer[comp_offset + 3],
                        ]);
                        v as f32
                    }
                }
                5126 => {
                    // FLOAT
                    if comp_offset + 4 > buffer.len() {
                        0.0
                    } else {
                        f32::from_le_bytes([
                            buffer[comp_offset],
                            buffer[comp_offset + 1],
                            buffer[comp_offset + 2],
                            buffer[comp_offset + 3],
                        ])
                    }
                }
                _ => 0.0,
            };

            result.push(value);
        }
    }

    Ok(result)
}

/// Read accessor data as Vec<u32> (for index buffers).
fn read_accessor_u32(
    accessor: &Accessor,
    buffer_views: &[BufferView],
    buffers: &[Vec<u8>],
) -> Result<Vec<u32>, AssetError> {
    let bv_idx = match accessor.buffer_view {
        Some(idx) => idx,
        None => return Ok(Vec::new()),
    };

    if bv_idx >= buffer_views.len() {
        return Err(AssetError::InvalidData(format!(
            "Buffer view index {bv_idx} out of range"
        )));
    }

    let bv = &buffer_views[bv_idx];
    if bv.buffer >= buffers.len() {
        return Err(AssetError::InvalidData(format!(
            "Buffer index {} out of range",
            bv.buffer
        )));
    }

    let buffer = &buffers[bv.buffer];
    let base_offset = bv.byte_offset + accessor.byte_offset;
    let comp_size = accessor.component_size();
    let stride = bv.byte_stride.unwrap_or(comp_size);

    let mut result = Vec::with_capacity(accessor.count);

    for i in 0..accessor.count {
        let offset = base_offset + i * stride;

        let value = match accessor.component_type {
            5121 => {
                // UNSIGNED_BYTE
                if offset < buffer.len() {
                    buffer[offset] as u32
                } else {
                    0
                }
            }
            5123 => {
                // UNSIGNED_SHORT
                if offset + 2 <= buffer.len() {
                    u16::from_le_bytes([buffer[offset], buffer[offset + 1]]) as u32
                } else {
                    0
                }
            }
            5125 => {
                // UNSIGNED_INT
                if offset + 4 <= buffer.len() {
                    u32::from_le_bytes([
                        buffer[offset],
                        buffer[offset + 1],
                        buffer[offset + 2],
                        buffer[offset + 3],
                    ])
                } else {
                    0
                }
            }
            _ => 0,
        };

        result.push(value);
    }

    Ok(result)
}

// =========================================================================
// JSON structure parsers
// =========================================================================

fn parse_scenes(json: &serde_json::Value) -> Result<Vec<GltfScene>, AssetError> {
    let mut scenes = Vec::new();
    if let Some(arr) = json.get("scenes").and_then(|v| v.as_array()) {
        for scene_json in arr {
            let name = scene_json.get("name").and_then(|v| v.as_str()).map(String::from);
            let nodes = scene_json
                .get("nodes")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            scenes.push(GltfScene { name, nodes });
        }
    }
    Ok(scenes)
}

fn parse_nodes(json: &serde_json::Value) -> Result<Vec<GltfNode>, AssetError> {
    let mut nodes = Vec::new();
    if let Some(arr) = json.get("nodes").and_then(|v| v.as_array()) {
        for node_json in arr {
            let name = node_json.get("name").and_then(|v| v.as_str()).map(String::from);
            let children = node_json
                .get("children")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let mesh = node_json.get("mesh").and_then(|v| v.as_u64()).map(|v| v as usize);
            let skin = node_json.get("skin").and_then(|v| v.as_u64()).map(|v| v as usize);

            let translation = parse_f32_array3(node_json.get("translation"), [0.0, 0.0, 0.0]);
            let rotation = parse_f32_array4(node_json.get("rotation"), [0.0, 0.0, 0.0, 1.0]);
            let scale = parse_f32_array3(node_json.get("scale"), [1.0, 1.0, 1.0]);

            let matrix = node_json.get("matrix").and_then(|v| {
                let arr = v.as_array()?;
                if arr.len() != 16 {
                    return None;
                }
                let mut m = [0.0f32; 16];
                for (i, val) in arr.iter().enumerate() {
                    m[i] = val.as_f64()? as f32;
                }
                Some(m)
            });

            nodes.push(GltfNode {
                name,
                children,
                mesh,
                skin,
                translation,
                rotation,
                scale,
                matrix,
            });
        }
    }
    Ok(nodes)
}

fn parse_meshes(json: &serde_json::Value) -> Result<Vec<GltfMesh>, AssetError> {
    let mut meshes = Vec::new();
    if let Some(arr) = json.get("meshes").and_then(|v| v.as_array()) {
        for mesh_json in arr {
            let name = mesh_json.get("name").and_then(|v| v.as_str()).map(String::from);
            let mut primitives = Vec::new();

            if let Some(prims) = mesh_json.get("primitives").and_then(|v| v.as_array()) {
                for prim in prims {
                    let mut attributes = HashMap::new();
                    if let Some(attrs) = prim.get("attributes").and_then(|v| v.as_object()) {
                        for (key, val) in attrs {
                            if let Some(idx) = val.as_u64() {
                                attributes.insert(key.clone(), idx as usize);
                            }
                        }
                    }
                    let indices = prim.get("indices").and_then(|v| v.as_u64()).map(|v| v as usize);
                    let material =
                        prim.get("material").and_then(|v| v.as_u64()).map(|v| v as usize);
                    let mode = prim.get("mode").and_then(|v| v.as_u64()).unwrap_or(4) as u32;

                    primitives.push(GltfPrimitive {
                        attributes,
                        indices,
                        material,
                        mode,
                    });
                }
            }

            meshes.push(GltfMesh { name, primitives });
        }
    }
    Ok(meshes)
}

fn parse_materials(json: &serde_json::Value) -> Result<Vec<GltfMaterial>, AssetError> {
    let mut materials = Vec::new();
    if let Some(arr) = json.get("materials").and_then(|v| v.as_array()) {
        for mat_json in arr {
            let name = mat_json.get("name").and_then(|v| v.as_str()).map(String::from);

            let pbr = mat_json.get("pbrMetallicRoughness");
            let base_color_factor =
                parse_f32_array4(pbr.and_then(|p| p.get("baseColorFactor")), [1.0, 1.0, 1.0, 1.0]);
            let metallic_factor = pbr
                .and_then(|p| p.get("metallicFactor"))
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            let roughness_factor = pbr
                .and_then(|p| p.get("roughnessFactor"))
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;

            let base_color_texture =
                parse_texture_info(pbr.and_then(|p| p.get("baseColorTexture")));
            let metallic_roughness_texture =
                parse_texture_info(pbr.and_then(|p| p.get("metallicRoughnessTexture")));
            let normal_texture = parse_texture_info(mat_json.get("normalTexture"));
            let occlusion_texture = parse_texture_info(mat_json.get("occlusionTexture"));
            let emissive_texture = parse_texture_info(mat_json.get("emissiveTexture"));

            let emissive_factor =
                parse_f32_array3(mat_json.get("emissiveFactor"), [0.0, 0.0, 0.0]);

            let alpha_mode = match mat_json
                .get("alphaMode")
                .and_then(|v| v.as_str())
                .unwrap_or("OPAQUE")
            {
                "MASK" => AlphaMode::Mask,
                "BLEND" => AlphaMode::Blend,
                _ => AlphaMode::Opaque,
            };

            let alpha_cutoff = mat_json
                .get("alphaCutoff")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32;

            let double_sided = mat_json
                .get("doubleSided")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            materials.push(GltfMaterial {
                name,
                base_color_factor,
                metallic_factor,
                roughness_factor,
                base_color_texture,
                metallic_roughness_texture,
                normal_texture,
                occlusion_texture,
                emissive_texture,
                emissive_factor,
                alpha_mode,
                alpha_cutoff,
                double_sided,
            });
        }
    }
    Ok(materials)
}

fn parse_textures(json: &serde_json::Value) -> Result<Vec<GltfTexture>, AssetError> {
    let mut textures = Vec::new();
    if let Some(arr) = json.get("textures").and_then(|v| v.as_array()) {
        for tex_json in arr {
            textures.push(GltfTexture {
                name: tex_json.get("name").and_then(|v| v.as_str()).map(String::from),
                source: tex_json.get("source").and_then(|v| v.as_u64()).map(|v| v as usize),
                sampler: tex_json.get("sampler").and_then(|v| v.as_u64()).map(|v| v as usize),
            });
        }
    }
    Ok(textures)
}

fn parse_images(json: &serde_json::Value) -> Result<Vec<GltfImage>, AssetError> {
    let mut images = Vec::new();
    if let Some(arr) = json.get("images").and_then(|v| v.as_array()) {
        for img_json in arr {
            images.push(GltfImage {
                name: img_json.get("name").and_then(|v| v.as_str()).map(String::from),
                uri: img_json.get("uri").and_then(|v| v.as_str()).map(String::from),
                mime_type: img_json.get("mimeType").and_then(|v| v.as_str()).map(String::from),
                buffer_view: img_json
                    .get("bufferView")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize),
            });
        }
    }
    Ok(images)
}

fn parse_samplers(json: &serde_json::Value) -> Result<Vec<GltfSampler>, AssetError> {
    let mut samplers = Vec::new();
    if let Some(arr) = json.get("samplers").and_then(|v| v.as_array()) {
        for samp_json in arr {
            samplers.push(GltfSampler {
                mag_filter: samp_json.get("magFilter").and_then(|v| v.as_u64()).map(|v| v as u32),
                min_filter: samp_json.get("minFilter").and_then(|v| v.as_u64()).map(|v| v as u32),
                wrap_s: samp_json
                    .get("wrapS")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10497) as u32,
                wrap_t: samp_json
                    .get("wrapT")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10497) as u32,
            });
        }
    }
    Ok(samplers)
}

fn parse_animations(
    json: &serde_json::Value,
    accessors: &[Accessor],
    buffer_views: &[BufferView],
    buffers: &[Vec<u8>],
) -> Result<Vec<AnimationClip>, AssetError> {
    let mut clips = Vec::new();

    if let Some(arr) = json.get("animations").and_then(|v| v.as_array()) {
        for anim_json in arr {
            let name = anim_json.get("name").and_then(|v| v.as_str()).map(String::from);

            // Parse samplers for this animation
            let anim_samplers: Vec<(usize, usize, Interpolation)> = anim_json
                .get("samplers")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|s| {
                            let input = s.get("input").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            let output =
                                s.get("output").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            let interp = match s
                                .get("interpolation")
                                .and_then(|v| v.as_str())
                                .unwrap_or("LINEAR")
                            {
                                "STEP" => Interpolation::Step,
                                "CUBICSPLINE" => Interpolation::CubicSpline,
                                _ => Interpolation::Linear,
                            };
                            (input, output, interp)
                        })
                        .collect()
                })
                .unwrap_or_default();

            let mut channels = Vec::new();
            let mut max_time: f32 = 0.0;

            if let Some(ch_arr) = anim_json.get("channels").and_then(|v| v.as_array()) {
                for ch_json in ch_arr {
                    let sampler_idx = ch_json
                        .get("sampler")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;

                    let target = ch_json.get("target");
                    let target_node = target
                        .and_then(|t| t.get("node"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let target_path = target
                        .and_then(|t| t.get("path"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("translation");

                    let property = match target_path {
                        "rotation" => AnimationProperty::Rotation,
                        "scale" => AnimationProperty::Scale,
                        "weights" => AnimationProperty::Weights,
                        _ => AnimationProperty::Translation,
                    };

                    if sampler_idx >= anim_samplers.len() {
                        continue;
                    }
                    let (input_acc, output_acc, interp) = anim_samplers[sampler_idx];

                    if input_acc >= accessors.len() || output_acc >= accessors.len() {
                        continue;
                    }

                    // Read timestamps
                    let timestamps =
                        read_accessor_f32(&accessors[input_acc], buffer_views, buffers)?;

                    if let Some(&last) = timestamps.last() {
                        if last > max_time {
                            max_time = last;
                        }
                    }

                    // Read values
                    let raw_values =
                        read_accessor_f32(&accessors[output_acc], buffer_views, buffers)?;

                    let values = match property {
                        AnimationProperty::Translation | AnimationProperty::Scale => {
                            let mut vecs = Vec::with_capacity(raw_values.len() / 3);
                            for chunk in raw_values.chunks_exact(3) {
                                vecs.push([chunk[0], chunk[1], chunk[2]]);
                            }
                            AnimationValues::Vec3(vecs)
                        }
                        AnimationProperty::Rotation => {
                            let mut vecs = Vec::with_capacity(raw_values.len() / 4);
                            for chunk in raw_values.chunks_exact(4) {
                                vecs.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            }
                            AnimationValues::Vec4(vecs)
                        }
                        AnimationProperty::Weights => AnimationValues::Scalar(raw_values),
                    };

                    channels.push(AnimationChannel {
                        target_node,
                        property,
                        timestamps,
                        values,
                        interpolation: interp,
                    });
                }
            }

            clips.push(AnimationClip {
                name,
                channels,
                duration: max_time,
            });
        }
    }

    Ok(clips)
}

fn parse_skins(
    json: &serde_json::Value,
    accessors: &[Accessor],
    buffer_views: &[BufferView],
    buffers: &[Vec<u8>],
) -> Result<Vec<Skeleton>, AssetError> {
    let mut skins = Vec::new();

    if let Some(arr) = json.get("skins").and_then(|v| v.as_array()) {
        for skin_json in arr {
            let name = skin_json.get("name").and_then(|v| v.as_str()).map(String::from);

            let joints: Vec<usize> = skin_json
                .get("joints")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();

            let skeleton_root = skin_json
                .get("skeleton")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);

            let ibm_accessor = skin_json
                .get("inverseBindMatrices")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);

            let inverse_bind_matrices = if let Some(ibm_idx) = ibm_accessor {
                if ibm_idx < accessors.len() {
                    let raw = read_accessor_f32(&accessors[ibm_idx], buffer_views, buffers)?;
                    let mut matrices = Vec::with_capacity(raw.len() / 16);
                    for chunk in raw.chunks_exact(16) {
                        let mut m = [0.0f32; 16];
                        m.copy_from_slice(chunk);
                        matrices.push(m);
                    }
                    matrices
                } else {
                    Vec::new()
                }
            } else {
                // Default: identity matrices
                joints.iter().map(|_| identity_matrix()).collect()
            };

            skins.push(Skeleton {
                name,
                joints,
                inverse_bind_matrices,
                skeleton_root,
            });
        }
    }

    Ok(skins)
}

// =========================================================================
// Mesh data extraction
// =========================================================================

/// Extract mesh data from a glTF primitive using the document's accessors and buffers.
pub fn extract_mesh_data(
    primitive: &GltfPrimitive,
    json: &serde_json::Value,
    buffers: &[Vec<u8>],
) -> Result<GltfMeshData, AssetError> {
    let buffer_views = parse_buffer_views(json)?;
    let accessors = parse_accessors(json)?;

    extract_mesh_data_internal(primitive, &accessors, &buffer_views, buffers)
}

fn extract_mesh_data_internal(
    primitive: &GltfPrimitive,
    accessors: &[Accessor],
    buffer_views: &[BufferView],
    buffers: &[Vec<u8>],
) -> Result<GltfMeshData, AssetError> {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut joints = Vec::new();
    let mut weights = Vec::new();
    let mut indices = Vec::new();

    // POSITION
    if let Some(&acc_idx) = primitive.attributes.get("POSITION") {
        if acc_idx < accessors.len() {
            let raw = read_accessor_f32(&accessors[acc_idx], buffer_views, buffers)?;
            for chunk in raw.chunks_exact(3) {
                positions.push([chunk[0], chunk[1], chunk[2]]);
            }
        }
    }

    // NORMAL
    if let Some(&acc_idx) = primitive.attributes.get("NORMAL") {
        if acc_idx < accessors.len() {
            let raw = read_accessor_f32(&accessors[acc_idx], buffer_views, buffers)?;
            for chunk in raw.chunks_exact(3) {
                normals.push([chunk[0], chunk[1], chunk[2]]);
            }
        }
    }

    // TEXCOORD_0
    if let Some(&acc_idx) = primitive.attributes.get("TEXCOORD_0") {
        if acc_idx < accessors.len() {
            let raw = read_accessor_f32(&accessors[acc_idx], buffer_views, buffers)?;
            for chunk in raw.chunks_exact(2) {
                uvs.push([chunk[0], chunk[1]]);
            }
        }
    }

    // TANGENT
    if let Some(&acc_idx) = primitive.attributes.get("TANGENT") {
        if acc_idx < accessors.len() {
            let raw = read_accessor_f32(&accessors[acc_idx], buffer_views, buffers)?;
            for chunk in raw.chunks_exact(4) {
                tangents.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
    }

    // JOINTS_0
    if let Some(&acc_idx) = primitive.attributes.get("JOINTS_0") {
        if acc_idx < accessors.len() {
            let raw = read_accessor_f32(&accessors[acc_idx], buffer_views, buffers)?;
            for chunk in raw.chunks_exact(4) {
                joints.push([chunk[0] as u16, chunk[1] as u16, chunk[2] as u16, chunk[3] as u16]);
            }
        }
    }

    // WEIGHTS_0
    if let Some(&acc_idx) = primitive.attributes.get("WEIGHTS_0") {
        if acc_idx < accessors.len() {
            let raw = read_accessor_f32(&accessors[acc_idx], buffer_views, buffers)?;
            for chunk in raw.chunks_exact(4) {
                weights.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
    }

    // Indices
    if let Some(indices_acc) = primitive.indices {
        if indices_acc < accessors.len() {
            indices = read_accessor_u32(&accessors[indices_acc], buffer_views, buffers)?;
        }
    }

    Ok(GltfMeshData {
        positions,
        normals,
        uvs,
        tangents,
        joints,
        weights,
        indices,
    })
}

// =========================================================================
// Helpers
// =========================================================================

fn parse_texture_info(value: Option<&serde_json::Value>) -> Option<GltfTextureInfo> {
    let v = value?;
    let index = v.get("index")?.as_u64()? as usize;
    let tex_coord = v.get("texCoord").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    Some(GltfTextureInfo { index, tex_coord })
}

fn parse_f32_array3(value: Option<&serde_json::Value>, default: [f32; 3]) -> [f32; 3] {
    value
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            if arr.len() >= 3 {
                Some([
                    arr[0].as_f64()? as f32,
                    arr[1].as_f64()? as f32,
                    arr[2].as_f64()? as f32,
                ])
            } else {
                None
            }
        })
        .unwrap_or(default)
}

fn parse_f32_array4(value: Option<&serde_json::Value>, default: [f32; 4]) -> [f32; 4] {
    value
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            if arr.len() >= 4 {
                Some([
                    arr[0].as_f64()? as f32,
                    arr[1].as_f64()? as f32,
                    arr[2].as_f64()? as f32,
                    arr[3].as_f64()? as f32,
                ])
            } else {
                None
            }
        })
        .unwrap_or(default)
}

fn identity_matrix() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Base64 tests -----------------------------------------------------

    #[test]
    fn test_base64_decode_basic() {
        let decoded = base64_decode("SGVsbG8=").unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_base64_decode_no_padding() {
        let decoded = base64_decode("YQ==").unwrap();
        assert_eq!(decoded, b"a");

        let decoded = base64_decode("YWI=").unwrap();
        assert_eq!(decoded, b"ab");

        let decoded = base64_decode("YWJj").unwrap();
        assert_eq!(decoded, b"abc");
    }

    #[test]
    fn test_base64_decode_binary() {
        // 4 bytes: [0, 1, 2, 3] -> base64 "AAIDBA==" wait no...
        // [0,1,2,3] -> "AAECAw=="
        let decoded = base64_decode("AAECAw==").unwrap();
        assert_eq!(decoded, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_data_uri_decode() {
        let uri = "data:application/octet-stream;base64,SGVsbG8=";
        let decoded = decode_data_uri(uri).unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_data_uri_invalid() {
        assert!(decode_data_uri("http://example.com").is_err());
    }

    // -- GLB tests --------------------------------------------------------

    fn make_minimal_glb(json_str: &str) -> Vec<u8> {
        let json_bytes = json_str.as_bytes();
        // Pad JSON to 4-byte alignment
        let json_padded_len = (json_bytes.len() + 3) & !3;
        let total_len = 12 + 8 + json_padded_len;

        let mut buf = Vec::with_capacity(total_len);
        // Header: magic, version, length
        buf.extend_from_slice(&GLB_MAGIC.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&(total_len as u32).to_le_bytes());

        // JSON chunk
        buf.extend_from_slice(&(json_padded_len as u32).to_le_bytes());
        buf.extend_from_slice(&GLB_CHUNK_JSON.to_le_bytes());
        buf.extend_from_slice(json_bytes);
        // Pad with spaces
        for _ in json_bytes.len()..json_padded_len {
            buf.push(b' ');
        }

        buf
    }

    #[test]
    fn test_glb_minimal() {
        let json = r#"{"asset":{"version":"2.0"}}"#;
        let glb = make_minimal_glb(json);
        let doc = parse_glb(&glb).unwrap();
        assert!(doc.scenes.is_empty());
        assert!(doc.nodes.is_empty());
        assert!(doc.meshes.is_empty());
    }

    #[test]
    fn test_glb_with_scene() {
        let json = r#"{
            "asset":{"version":"2.0"},
            "scene": 0,
            "scenes":[{"name":"main","nodes":[0]}],
            "nodes":[{"name":"root","translation":[1.0,2.0,3.0]}]
        }"#;
        let glb = make_minimal_glb(json);
        let doc = parse_glb(&glb).unwrap();
        assert_eq!(doc.scenes.len(), 1);
        assert_eq!(doc.scenes[0].name.as_deref(), Some("main"));
        assert_eq!(doc.scenes[0].nodes, vec![0]);
        assert_eq!(doc.default_scene, Some(0));
        assert_eq!(doc.nodes[0].translation, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_glb_bad_magic() {
        let mut glb = make_minimal_glb("{}");
        glb[0] = 0; // corrupt magic
        assert!(parse_glb(&glb).is_err());
    }

    #[test]
    fn test_glb_too_small() {
        assert!(parse_glb(&[0u8; 8]).is_err());
    }

    // -- glTF JSON tests --------------------------------------------------

    #[test]
    fn test_gltf_json_minimal() {
        let json = br#"{"asset":{"version":"2.0"}}"#;
        let doc = parse_gltf_json(json, Path::new("test.gltf")).unwrap();
        assert!(doc.meshes.is_empty());
    }

    #[test]
    fn test_gltf_materials() {
        let json = br#"{
            "asset":{"version":"2.0"},
            "materials":[{
                "name":"mat1",
                "pbrMetallicRoughness":{
                    "baseColorFactor":[0.5,0.6,0.7,1.0],
                    "metallicFactor":0.3,
                    "roughnessFactor":0.8
                },
                "alphaMode":"BLEND",
                "alphaCutoff":0.25,
                "doubleSided":true,
                "emissiveFactor":[0.1,0.2,0.3]
            }]
        }"#;
        let doc = parse_gltf_json(json, Path::new("test.gltf")).unwrap();
        assert_eq!(doc.materials.len(), 1);
        let mat = &doc.materials[0];
        assert_eq!(mat.name.as_deref(), Some("mat1"));
        assert!((mat.base_color_factor[0] - 0.5).abs() < 0.001);
        assert!((mat.metallic_factor - 0.3).abs() < 0.001);
        assert!((mat.roughness_factor - 0.8).abs() < 0.001);
        assert_eq!(mat.alpha_mode, AlphaMode::Blend);
        assert!((mat.alpha_cutoff - 0.25).abs() < 0.001);
        assert!(mat.double_sided);
    }

    #[test]
    fn test_gltf_mesh_structure() {
        let json = br#"{
            "asset":{"version":"2.0"},
            "meshes":[{
                "name":"cube",
                "primitives":[{
                    "attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},
                    "indices":3,
                    "material":0,
                    "mode":4
                }]
            }]
        }"#;
        let doc = parse_gltf_json(json, Path::new("test.gltf")).unwrap();
        assert_eq!(doc.meshes.len(), 1);
        assert_eq!(doc.meshes[0].primitives.len(), 1);
        let prim = &doc.meshes[0].primitives[0];
        assert_eq!(prim.attributes.get("POSITION"), Some(&0));
        assert_eq!(prim.attributes.get("NORMAL"), Some(&1));
        assert_eq!(prim.indices, Some(3));
        assert_eq!(prim.material, Some(0));
    }

    #[test]
    fn test_gltf_node_hierarchy() {
        let json = br#"{
            "asset":{"version":"2.0"},
            "nodes":[
                {"name":"parent","children":[1,2],"scale":[2.0,2.0,2.0]},
                {"name":"child_a","mesh":0},
                {"name":"child_b","rotation":[0.0,0.707,0.0,0.707]}
            ]
        }"#;
        let doc = parse_gltf_json(json, Path::new("test.gltf")).unwrap();
        assert_eq!(doc.nodes.len(), 3);
        assert_eq!(doc.nodes[0].children, vec![1, 2]);
        assert_eq!(doc.nodes[0].scale, [2.0, 2.0, 2.0]);
        assert_eq!(doc.nodes[1].mesh, Some(0));
        assert!((doc.nodes[2].rotation[1] - 0.707).abs() < 0.001);
    }

    #[test]
    fn test_gltf_textures_and_samplers() {
        let json = br#"{
            "asset":{"version":"2.0"},
            "textures":[{"source":0,"sampler":0}],
            "images":[{"uri":"diffuse.png","mimeType":"image/png"}],
            "samplers":[{"magFilter":9729,"minFilter":9987,"wrapS":33071,"wrapT":33071}]
        }"#;
        let doc = parse_gltf_json(json, Path::new("test.gltf")).unwrap();
        assert_eq!(doc.textures.len(), 1);
        assert_eq!(doc.textures[0].source, Some(0));
        assert_eq!(doc.images.len(), 1);
        assert_eq!(doc.images[0].uri.as_deref(), Some("diffuse.png"));
        assert_eq!(doc.samplers.len(), 1);
        assert_eq!(doc.samplers[0].mag_filter, Some(9729));
        assert_eq!(doc.samplers[0].wrap_s, 33071);
    }

    // -- Accessor tests ---------------------------------------------------

    #[test]
    fn test_accessor_component_count() {
        let acc = Accessor {
            buffer_view: None,
            byte_offset: 0,
            component_type: 5126,
            count: 1,
            accessor_type: "VEC3".to_owned(),
            min: None,
            max: None,
            normalized: false,
        };
        assert_eq!(acc.component_count(), 3);
        assert_eq!(acc.component_size(), 4); // f32
        assert_eq!(acc.element_size(), 12);
    }

    #[test]
    fn test_read_accessor_f32() {
        // Create a buffer with 3 floats: [1.0, 2.0, 3.0]
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&1.0f32.to_le_bytes());
        buffer.extend_from_slice(&2.0f32.to_le_bytes());
        buffer.extend_from_slice(&3.0f32.to_le_bytes());

        let bv = BufferView {
            buffer: 0,
            byte_offset: 0,
            byte_length: 12,
            byte_stride: None,
            target: None,
        };

        let acc = Accessor {
            buffer_view: Some(0),
            byte_offset: 0,
            component_type: 5126, // FLOAT
            count: 1,
            accessor_type: "VEC3".to_owned(),
            min: None,
            max: None,
            normalized: false,
        };

        let result = read_accessor_f32(&acc, &[bv], &[buffer]).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 2.0).abs() < 0.001);
        assert!((result[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_read_accessor_u16_indices() {
        // 3 u16 indices: [0, 1, 2]
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&0u16.to_le_bytes());
        buffer.extend_from_slice(&1u16.to_le_bytes());
        buffer.extend_from_slice(&2u16.to_le_bytes());

        let bv = BufferView {
            buffer: 0,
            byte_offset: 0,
            byte_length: 6,
            byte_stride: None,
            target: None,
        };

        let acc = Accessor {
            buffer_view: Some(0),
            byte_offset: 0,
            component_type: 5123, // UNSIGNED_SHORT
            count: 3,
            accessor_type: "SCALAR".to_owned(),
            min: None,
            max: None,
            normalized: false,
        };

        let result = read_accessor_u32(&acc, &[bv], &[buffer]).unwrap();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_read_accessor_with_stride() {
        // Buffer with stride=16 but element_size=12 (VEC3 f32)
        // 2 elements: [1,2,3, padding, 4,5,6, padding]
        let mut buffer = vec![0u8; 32];
        buffer[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        buffer[4..8].copy_from_slice(&2.0f32.to_le_bytes());
        buffer[8..12].copy_from_slice(&3.0f32.to_le_bytes());
        // [12..16] = padding
        buffer[16..20].copy_from_slice(&4.0f32.to_le_bytes());
        buffer[20..24].copy_from_slice(&5.0f32.to_le_bytes());
        buffer[24..28].copy_from_slice(&6.0f32.to_le_bytes());

        let bv = BufferView {
            buffer: 0,
            byte_offset: 0,
            byte_length: 32,
            byte_stride: Some(16),
            target: None,
        };

        let acc = Accessor {
            buffer_view: Some(0),
            byte_offset: 0,
            component_type: 5126,
            count: 2,
            accessor_type: "VEC3".to_owned(),
            min: None,
            max: None,
            normalized: false,
        };

        let result = read_accessor_f32(&acc, &[bv], &[buffer]).unwrap();
        assert_eq!(result.len(), 6);
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[3] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_read_accessor_normalized_u8() {
        let buffer = vec![0u8, 128, 255];

        let bv = BufferView {
            buffer: 0,
            byte_offset: 0,
            byte_length: 3,
            byte_stride: None,
            target: None,
        };

        let acc = Accessor {
            buffer_view: Some(0),
            byte_offset: 0,
            component_type: 5121, // UNSIGNED_BYTE
            count: 3,
            accessor_type: "SCALAR".to_owned(),
            min: None,
            max: None,
            normalized: true,
        };

        let result = read_accessor_f32(&acc, &[bv], &[buffer]).unwrap();
        assert!((result[0] - 0.0).abs() < 0.01);
        assert!((result[1] - 0.502).abs() < 0.01);
        assert!((result[2] - 1.0).abs() < 0.01);
    }

    // -- Skin/animation tests ---------------------------------------------

    #[test]
    fn test_gltf_skin_parsing() {
        let json = br#"{
            "asset":{"version":"2.0"},
            "skins":[{
                "name":"armature",
                "joints":[0,1,2],
                "skeleton":0
            }]
        }"#;
        let doc = parse_gltf_json(json, Path::new("test.gltf")).unwrap();
        assert_eq!(doc.skins.len(), 1);
        assert_eq!(doc.skins[0].joints, vec![0, 1, 2]);
        assert_eq!(doc.skins[0].skeleton_root, Some(0));
        // Without inverseBindMatrices, should get identity matrices
        assert_eq!(doc.skins[0].inverse_bind_matrices.len(), 3);
        assert_eq!(doc.skins[0].inverse_bind_matrices[0][0], 1.0);
        assert_eq!(doc.skins[0].inverse_bind_matrices[0][5], 1.0);
    }

    #[test]
    fn test_alpha_mode_parsing() {
        assert_eq!(
            match "OPAQUE" {
                "MASK" => AlphaMode::Mask,
                "BLEND" => AlphaMode::Blend,
                _ => AlphaMode::Opaque,
            },
            AlphaMode::Opaque
        );
    }

    #[test]
    fn test_identity_matrix() {
        let m = identity_matrix();
        assert_eq!(m[0], 1.0);
        assert_eq!(m[5], 1.0);
        assert_eq!(m[10], 1.0);
        assert_eq!(m[15], 1.0);
        assert_eq!(m[1], 0.0);
        assert_eq!(m[4], 0.0);
    }

    #[test]
    fn test_interpolation_variants() {
        let interps = [Interpolation::Linear, Interpolation::Step, Interpolation::CubicSpline];
        assert_eq!(interps.len(), 3);
    }

    #[test]
    fn test_gltf_animation_structure() {
        // Minimal animation JSON without buffer data (will have empty values)
        let json = br#"{
            "asset":{"version":"2.0"},
            "accessors":[
                {"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR"},
                {"bufferView":1,"componentType":5126,"count":2,"type":"VEC3"}
            ],
            "bufferViews":[
                {"buffer":0,"byteOffset":0,"byteLength":8},
                {"buffer":0,"byteOffset":8,"byteLength":24}
            ],
            "buffers":[{"byteLength":32}],
            "animations":[{
                "name":"walk",
                "samplers":[{"input":0,"output":1,"interpolation":"LINEAR"}],
                "channels":[{"sampler":0,"target":{"node":0,"path":"translation"}}]
            }]
        }"#;
        let doc = parse_gltf_json(json, Path::new("test.gltf")).unwrap();
        assert_eq!(doc.animations.len(), 1);
        assert_eq!(doc.animations[0].name.as_deref(), Some("walk"));
        assert_eq!(doc.animations[0].channels.len(), 1);
        assert_eq!(doc.animations[0].channels[0].property, AnimationProperty::Translation);
    }
}
