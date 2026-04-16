//! Format-specific asset loaders.
//!
//! Provides loaders for common file formats:
//! - [`TextureLoader`] — BMP image files
//! - [`WavLoader`] — WAV audio files
//! - [`ObjLoader`] — Wavefront OBJ mesh files
//! - [`TextLoader`] — plain text / string files
//! - [`BytesLoader`] — raw bytes (catch-all)

use std::path::Path;

use crate::loader::{AssetError, AssetLoader};

// =========================================================================
// Data types
// =========================================================================

/// Pixel format of a loaded texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// 3 bytes per pixel: red, green, blue.
    Rgb8,
    /// 4 bytes per pixel: red, green, blue, alpha.
    Rgba8,
}

/// Decoded texture / image data.
#[derive(Debug, Clone)]
pub struct TextureData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Raw pixel bytes in row-major order (top-to-bottom, left-to-right).
    pub pixels: Vec<u8>,
    /// Pixel format.
    pub format: TextureFormat,
}

/// Decoded PCM audio data from a WAV file.
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Samples per second.
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Bits per sample in the source file.
    pub bits_per_sample: u16,
    /// Interleaved PCM samples normalised to `f32` in [-1, 1].
    pub samples: Vec<f32>,
}

/// Decoded mesh data from a Wavefront OBJ file.
#[derive(Debug, Clone)]
pub struct MeshData {
    /// Vertex positions, 3 floats each `[x, y, z]`.
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals (may be empty if the OBJ has no `vn` lines).
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates (may be empty if the OBJ has no `vt` lines).
    pub uvs: Vec<[f32; 2]>,
    /// Triangle indices into the above attribute arrays.  The arrays are
    /// de-indexed (i.e. each unique pos/normal/uv combination gets its own
    /// index) so all attribute arrays have the same length and the indices
    /// reference into them uniformly.
    pub indices: Vec<u32>,
}

// =========================================================================
// TextureLoader — BMP parser
// =========================================================================

/// Loads BMP image files.
///
/// Supports uncompressed 24-bit (RGB) and 32-bit (RGBA) BMP files with a
/// standard 40-byte BITMAPINFOHEADER.
pub struct TextureLoader;

impl AssetLoader for TextureLoader {
    type Asset = TextureData;

    fn extensions(&self) -> &[&str] {
        &["bmp"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<TextureData, AssetError> {
        parse_bmp(bytes)
    }
}

/// Parse a BMP file from raw bytes.
fn parse_bmp(data: &[u8]) -> Result<TextureData, AssetError> {
    // -- BMP file header (14 bytes) ------------------------------------------
    if data.len() < 54 {
        return Err(AssetError::InvalidData(
            "BMP file too small (need at least 54 bytes for headers)".into(),
        ));
    }

    if data[0] != b'B' || data[1] != b'M' {
        return Err(AssetError::InvalidData("Not a BMP file (missing BM magic)".into()));
    }

    let pixel_data_offset = read_u32_le(data, 10) as usize;

    // -- DIB header (BITMAPINFOHEADER, 40 bytes) -----------------------------
    let dib_header_size = read_u32_le(data, 14);
    if dib_header_size < 40 {
        return Err(AssetError::InvalidData(format!(
            "Unsupported DIB header size: {dib_header_size} (expected >= 40)"
        )));
    }

    let width = read_i32_le(data, 18);
    let height = read_i32_le(data, 22);
    let _planes = read_u16_le(data, 26);
    let bits_per_pixel = read_u16_le(data, 28);
    let compression = read_u32_le(data, 30);

    if compression != 0 {
        return Err(AssetError::InvalidData(format!(
            "Compressed BMP not supported (compression = {compression})"
        )));
    }

    if bits_per_pixel != 24 && bits_per_pixel != 32 {
        return Err(AssetError::InvalidData(format!(
            "Unsupported bits per pixel: {bits_per_pixel} (only 24 and 32 supported)"
        )));
    }

    let abs_width = width.unsigned_abs();
    let abs_height = height.unsigned_abs();
    // BMP rows are bottom-up when height > 0, top-down when height < 0.
    let bottom_up = height > 0;

    let bytes_per_pixel = (bits_per_pixel / 8) as usize;
    // BMP rows are padded to multiples of 4 bytes.
    let row_stride = ((abs_width as usize * bytes_per_pixel + 3) / 4) * 4;

    let pixel_data = &data[pixel_data_offset..];
    let needed = row_stride * abs_height as usize;
    if pixel_data.len() < needed {
        return Err(AssetError::InvalidData(format!(
            "BMP pixel data too short: have {} bytes, need {needed}",
            pixel_data.len()
        )));
    }

    // Output is always RGB (3 bytes/pixel), top-to-bottom.
    let out_bpp = 3usize;
    let mut pixels = vec![0u8; abs_width as usize * abs_height as usize * out_bpp];

    for row in 0..abs_height as usize {
        let src_row = if bottom_up {
            abs_height as usize - 1 - row
        } else {
            row
        };
        let src_offset = src_row * row_stride;

        for col in 0..abs_width as usize {
            let src_px = src_offset + col * bytes_per_pixel;
            let dst_px = (row * abs_width as usize + col) * out_bpp;

            // BMP stores BGR(A); convert to RGB.
            let b = pixel_data[src_px];
            let g = pixel_data[src_px + 1];
            let r = pixel_data[src_px + 2];

            pixels[dst_px] = r;
            pixels[dst_px + 1] = g;
            pixels[dst_px + 2] = b;
        }
    }

    Ok(TextureData {
        width: abs_width,
        height: abs_height,
        pixels,
        format: TextureFormat::Rgb8,
    })
}

// =========================================================================
// WavLoader
// =========================================================================

/// Loads WAV audio files (PCM 8-bit, 16-bit, and 24-bit).
pub struct WavLoader;

impl AssetLoader for WavLoader {
    type Asset = AudioData;

    fn extensions(&self) -> &[&str] {
        &["wav"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<AudioData, AssetError> {
        parse_wav(bytes)
    }
}

/// Parse a RIFF/WAVE file from raw bytes.
fn parse_wav(data: &[u8]) -> Result<AudioData, AssetError> {
    if data.len() < 44 {
        return Err(AssetError::InvalidData("WAV file too small".into()));
    }

    // RIFF header
    if &data[0..4] != b"RIFF" {
        return Err(AssetError::InvalidData("Missing RIFF header".into()));
    }
    if &data[8..12] != b"WAVE" {
        return Err(AssetError::InvalidData("Missing WAVE identifier".into()));
    }

    // Walk chunks to find "fmt " and "data".
    let mut pos = 12usize;
    let mut fmt_found = false;
    let mut audio_format: u16 = 0;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut pcm_data: &[u8] = &[];

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = read_u32_le(data, pos + 4) as usize;
        let chunk_body = pos + 8;

        if chunk_id == b"fmt " {
            if chunk_size < 16 || chunk_body + 16 > data.len() {
                return Err(AssetError::InvalidData("fmt chunk too small".into()));
            }
            audio_format = read_u16_le(data, chunk_body);
            channels = read_u16_le(data, chunk_body + 2);
            sample_rate = read_u32_le(data, chunk_body + 4);
            // bytes_per_sec at +8, block_align at +12
            bits_per_sample = read_u16_le(data, chunk_body + 14);
            fmt_found = true;
        } else if chunk_id == b"data" {
            let end = (chunk_body + chunk_size).min(data.len());
            pcm_data = &data[chunk_body..end];
        }

        // Advance to next chunk (chunks are word-aligned).
        pos = chunk_body + ((chunk_size + 1) & !1);
    }

    if !fmt_found {
        return Err(AssetError::InvalidData("No fmt chunk found in WAV".into()));
    }
    if audio_format != 1 {
        return Err(AssetError::InvalidData(format!(
            "Only PCM (format=1) WAV supported, got format={audio_format}"
        )));
    }
    if pcm_data.is_empty() {
        return Err(AssetError::InvalidData("No data chunk found in WAV".into()));
    }

    // Convert raw PCM to f32 samples.
    let samples = match bits_per_sample {
        8 => pcm_data.iter().map(|&b| (b as f32 / 128.0) - 1.0).collect(),
        16 => {
            let mut out = Vec::with_capacity(pcm_data.len() / 2);
            for chunk in pcm_data.chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(s as f32 / 32768.0);
            }
            out
        }
        24 => {
            let mut out = Vec::with_capacity(pcm_data.len() / 3);
            for chunk in pcm_data.chunks_exact(3) {
                // Sign-extend 24-bit to 32-bit.
                let val = (chunk[0] as i32) | ((chunk[1] as i32) << 8) | ((chunk[2] as i32) << 16);
                let signed = if val & 0x80_0000 != 0 {
                    val | !0xFF_FFFF
                } else {
                    val
                };
                out.push(signed as f32 / 8_388_608.0);
            }
            out
        }
        _ => {
            return Err(AssetError::InvalidData(format!(
                "Unsupported bits per sample: {bits_per_sample}"
            )));
        }
    };

    Ok(AudioData {
        sample_rate,
        channels,
        bits_per_sample,
        samples,
    })
}

// =========================================================================
// ObjLoader — Wavefront OBJ
// =========================================================================

/// Loads Wavefront OBJ mesh files.
///
/// Supports `v`, `vn`, `vt`, and `f` directives.  Polygonal faces with more
/// than 3 vertices are triangulated using a simple fan method.
pub struct ObjLoader;

impl AssetLoader for ObjLoader {
    type Asset = MeshData;

    fn extensions(&self) -> &[&str] {
        &["obj"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<MeshData, AssetError> {
        let text =
            std::str::from_utf8(bytes).map_err(|e| AssetError::Parse(e.to_string()))?;
        parse_obj(text)
    }
}

/// A single vertex index triplet from an OBJ face: `v/vt/vn`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ObjVertex {
    pos: usize,   // 0-based index into positions
    uv: usize,    // 0-based; usize::MAX means absent
    norm: usize,  // 0-based; usize::MAX means absent
}

fn parse_obj(text: &str) -> Result<MeshData, AssetError> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();

    // The de-indexed output arrays.
    let mut out_positions: Vec<[f32; 3]> = Vec::new();
    let mut out_normals: Vec<[f32; 3]> = Vec::new();
    let mut out_uvs: Vec<[f32; 2]> = Vec::new();
    let mut out_indices: Vec<u32> = Vec::new();

    // Map from ObjVertex -> output index for de-duplication.
    let mut vertex_map: std::collections::HashMap<ObjVertex, u32> =
        std::collections::HashMap::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_whitespace();
        let keyword = match parts.next() {
            Some(k) => k,
            None => continue,
        };

        match keyword {
            "v" => {
                let coords = parse_floats(&mut parts, 3, "v")?;
                positions.push([coords[0], coords[1], coords[2]]);
            }
            "vn" => {
                let coords = parse_floats(&mut parts, 3, "vn")?;
                normals.push([coords[0], coords[1], coords[2]]);
            }
            "vt" => {
                let coords = parse_floats(&mut parts, 2, "vt")?;
                uvs.push([coords[0], coords[1]]);
            }
            "f" => {
                let verts: Vec<ObjVertex> = parts
                    .map(|tok| parse_face_vertex(tok, positions.len(), uvs.len(), normals.len()))
                    .collect::<Result<Vec<_>, _>>()?;

                if verts.len() < 3 {
                    return Err(AssetError::Parse(
                        "Face with fewer than 3 vertices".into(),
                    ));
                }

                // Fan triangulation: (0,1,2), (0,2,3), (0,3,4), ...
                for i in 1..verts.len() - 1 {
                    for &vi in &[verts[0], verts[i], verts[i + 1]] {
                        let idx = if let Some(&existing) = vertex_map.get(&vi) {
                            existing
                        } else {
                            let idx = out_positions.len() as u32;
                            out_positions.push(positions[vi.pos]);
                            out_normals.push(if vi.norm != usize::MAX {
                                normals[vi.norm]
                            } else {
                                [0.0, 0.0, 0.0]
                            });
                            out_uvs.push(if vi.uv != usize::MAX {
                                uvs[vi.uv]
                            } else {
                                [0.0, 0.0]
                            });
                            vertex_map.insert(vi, idx);
                            idx
                        };
                        out_indices.push(idx);
                    }
                }
            }
            // Silently skip unsupported directives (mtllib, usemtl, o, g, s, ...).
            _ => {}
        }
    }

    Ok(MeshData {
        positions: out_positions,
        normals: out_normals,
        uvs: out_uvs,
        indices: out_indices,
    })
}

/// Parse a face vertex token like `1/2/3`, `1//3`, `1/2`, or `1`.
fn parse_face_vertex(
    token: &str,
    num_pos: usize,
    num_uv: usize,
    num_norm: usize,
) -> Result<ObjVertex, AssetError> {
    let parts: Vec<&str> = token.split('/').collect();

    let parse_idx = |s: &str, count: usize, name: &str| -> Result<usize, AssetError> {
        let idx: i64 = s
            .parse()
            .map_err(|_| AssetError::Parse(format!("invalid {name} index: '{s}'")))?;
        if idx > 0 {
            Ok((idx - 1) as usize)
        } else if idx < 0 {
            // Negative indices are relative to the end.
            let abs = (-idx) as usize;
            if abs > count {
                return Err(AssetError::Parse(format!(
                    "negative {name} index {idx} out of range (count={count})"
                )));
            }
            Ok(count - abs)
        } else {
            Err(AssetError::Parse(format!("{name} index cannot be 0")))
        }
    };

    let pos = parse_idx(
        parts.first().ok_or_else(|| AssetError::Parse("empty face vertex".into()))?,
        num_pos,
        "position",
    )?;

    let uv = if parts.len() > 1 && !parts[1].is_empty() {
        parse_idx(parts[1], num_uv, "uv")?
    } else {
        usize::MAX
    };

    let norm = if parts.len() > 2 && !parts[2].is_empty() {
        parse_idx(parts[2], num_norm, "normal")?
    } else {
        usize::MAX
    };

    Ok(ObjVertex { pos, uv, norm })
}

/// Parse `count` floats from a whitespace-delimited iterator.
fn parse_floats<'a>(
    iter: &mut impl Iterator<Item = &'a str>,
    count: usize,
    directive: &str,
) -> Result<Vec<f32>, AssetError> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let tok = iter.next().ok_or_else(|| {
            AssetError::Parse(format!("{directive}: expected {count} floats, got {i}"))
        })?;
        let val: f32 = tok
            .parse()
            .map_err(|_| AssetError::Parse(format!("{directive}: invalid float '{tok}'")))?;
        out.push(val);
    }
    Ok(out)
}

// =========================================================================
// TextLoader
// =========================================================================

/// Loads any file as a UTF-8 [`String`].
pub struct TextLoader;

impl AssetLoader for TextLoader {
    type Asset = String;

    fn extensions(&self) -> &[&str] {
        &["txt", "json", "ron", "toml", "yaml", "yml", "xml", "csv", "cfg", "ini", "log"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<String, AssetError> {
        String::from_utf8(bytes.to_vec()).map_err(|e| AssetError::Parse(e.to_string()))
    }
}

// =========================================================================
// BytesLoader
// =========================================================================

/// Loads any file as raw `Vec<u8>` bytes.  Useful as a fallback loader.
pub struct BytesLoader;

impl AssetLoader for BytesLoader {
    type Asset = Vec<u8>;

    fn extensions(&self) -> &[&str] {
        &["bin", "dat", "raw"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<Vec<u8>, AssetError> {
        Ok(bytes.to_vec())
    }
}

// =========================================================================
// Little-endian read helpers
// =========================================================================

fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn read_i32_le(data: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([
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

    // -- BMP tests --------------------------------------------------------

    /// Builds a minimal valid 24-bit BMP file in memory.
    fn make_test_bmp(width: u32, height: u32, pixel_rgb: &[u8]) -> Vec<u8> {
        let row_stride = ((width as usize * 3 + 3) / 4) * 4;
        let pixel_data_size = row_stride * height as usize;
        let file_size = 54 + pixel_data_size;

        let mut buf = vec![0u8; file_size];

        // File header
        buf[0] = b'B';
        buf[1] = b'M';
        buf[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
        buf[10..14].copy_from_slice(&54u32.to_le_bytes()); // pixel data offset

        // DIB header (BITMAPINFOHEADER)
        buf[14..18].copy_from_slice(&40u32.to_le_bytes()); // header size
        buf[18..22].copy_from_slice(&(width as i32).to_le_bytes());
        buf[22..26].copy_from_slice(&(height as i32).to_le_bytes()); // positive = bottom-up
        buf[26..28].copy_from_slice(&1u16.to_le_bytes()); // planes
        buf[28..30].copy_from_slice(&24u16.to_le_bytes()); // bpp
        buf[30..34].copy_from_slice(&0u32.to_le_bytes()); // compression = 0

        // Write pixel data bottom-up, BGR
        for row in 0..height as usize {
            for col in 0..width as usize {
                let src = (row * width as usize + col) * 3;
                // BMP is bottom-up, so row 0 in our input is the last row in the file.
                let bmp_row = height as usize - 1 - row;
                let dst = 54 + bmp_row * row_stride + col * 3;
                buf[dst] = pixel_rgb[src + 2]; // B
                buf[dst + 1] = pixel_rgb[src + 1]; // G
                buf[dst + 2] = pixel_rgb[src]; // R
            }
        }

        buf
    }

    #[test]
    fn test_bmp_parse_2x2() {
        // 2x2 image: red, green, blue, white
        #[rustfmt::skip]
        let pixels: &[u8] = &[
            255, 0, 0,    0, 255, 0,   // row 0 (top): red, green
            0, 0, 255,    255, 255, 255, // row 1 (bottom): blue, white
        ];
        let bmp = make_test_bmp(2, 2, pixels);
        let result = parse_bmp(&bmp).unwrap();

        assert_eq!(result.width, 2);
        assert_eq!(result.height, 2);
        assert_eq!(result.format, TextureFormat::Rgb8);
        assert_eq!(result.pixels.len(), 2 * 2 * 3);

        // Top-left should be red.
        assert_eq!(&result.pixels[0..3], &[255, 0, 0]);
        // Top-right should be green.
        assert_eq!(&result.pixels[3..6], &[0, 255, 0]);
        // Bottom-left should be blue.
        assert_eq!(&result.pixels[6..9], &[0, 0, 255]);
        // Bottom-right should be white.
        assert_eq!(&result.pixels[9..12], &[255, 255, 255]);
    }

    #[test]
    fn test_bmp_invalid_magic() {
        let data = vec![0u8; 100];
        assert!(parse_bmp(&data).is_err());
    }

    #[test]
    fn test_bmp_too_small() {
        let data = vec![b'B', b'M'];
        assert!(parse_bmp(&data).is_err());
    }

    // -- WAV tests --------------------------------------------------------

    /// Builds a minimal valid WAV file with 16-bit PCM data.
    fn make_test_wav(sample_rate: u32, channels: u16, samples_i16: &[i16]) -> Vec<u8> {
        let bits_per_sample: u16 = 16;
        let block_align = channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * block_align as u32;
        let data_size = (samples_i16.len() * 2) as u32;
        let file_size = 36 + data_size;

        let mut buf = Vec::with_capacity(file_size as usize + 8);

        // RIFF header
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");

        // fmt  chunk
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples_i16 {
            buf.extend_from_slice(&s.to_le_bytes());
        }

        buf
    }

    #[test]
    fn test_wav_parse_basic() {
        let samples = [0i16, 16384, -16384, 32767];
        let wav = make_test_wav(44100, 1, &samples);
        let result = parse_wav(&wav).unwrap();

        assert_eq!(result.sample_rate, 44100);
        assert_eq!(result.channels, 1);
        assert_eq!(result.bits_per_sample, 16);
        assert_eq!(result.samples.len(), 4);

        // Check approximate values.
        assert!((result.samples[0]).abs() < 0.001);
        assert!((result.samples[1] - 0.5).abs() < 0.01);
        assert!((result.samples[2] + 0.5).abs() < 0.01);
    }

    #[test]
    fn test_wav_invalid_header() {
        let data = vec![0u8; 100];
        assert!(parse_wav(&data).is_err());
    }

    // -- OBJ tests --------------------------------------------------------

    #[test]
    fn test_obj_triangle() {
        let obj = "\
# Simple triangle
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
f 1//1 2//1 3//1
";
        let mesh = parse_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.normals.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
        assert_eq!(mesh.positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(mesh.positions[1], [1.0, 0.0, 0.0]);
        assert_eq!(mesh.positions[2], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_obj_quad_triangulation() {
        let obj = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
";
        let mesh = parse_obj(obj).unwrap();
        // Quad -> 2 triangles -> 6 indices.
        assert_eq!(mesh.indices.len(), 6);
        // 4 unique vertices.
        assert_eq!(mesh.positions.len(), 4);
    }

    #[test]
    fn test_obj_with_uvs_and_normals() {
        let obj = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.5 1.0
vn 0.0 0.0 1.0
f 1/1/1 2/2/1 3/3/1
";
        let mesh = parse_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.uvs.len(), 3);
        assert_eq!(mesh.normals.len(), 3);
        assert_eq!(mesh.uvs[0], [0.0, 0.0]);
        assert_eq!(mesh.uvs[1], [1.0, 0.0]);
        assert_eq!(mesh.uvs[2], [0.5, 1.0]);
    }

    #[test]
    fn test_obj_negative_indices() {
        let obj = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
f -3 -2 -1
";
        let mesh = parse_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
        // -3 -> index 0, -2 -> index 1, -1 -> index 2
        assert_eq!(mesh.positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(mesh.positions[1], [1.0, 0.0, 0.0]);
        assert_eq!(mesh.positions[2], [0.5, 1.0, 0.0]);
    }

    #[test]
    fn test_obj_empty() {
        let mesh = parse_obj("# empty file\n").unwrap();
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_obj_cube() {
        let obj = "\
# Cube
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
f 1 2 3 4
f 5 8 7 6
f 1 4 8 5
f 2 6 7 3
f 4 3 7 8
f 1 5 6 2
";
        let mesh = parse_obj(obj).unwrap();
        // 6 quads -> 12 triangles -> 36 indices.
        assert_eq!(mesh.indices.len(), 36);
        assert_eq!(mesh.positions.len(), 8);
    }

    // -- TextLoader / BytesLoader tests -----------------------------------

    #[test]
    fn test_text_loader() {
        let loader = TextLoader;
        let result = loader.load(Path::new("test.txt"), b"hello world").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_bytes_loader() {
        let loader = BytesLoader;
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let result = loader.load(Path::new("test.bin"), &data).unwrap();
        assert_eq!(result, data);
    }

    // -- TextureLoader integration ----------------------------------------

    #[test]
    fn test_texture_loader_extensions() {
        let loader = TextureLoader;
        assert_eq!(loader.extensions(), &["bmp"]);
    }

    #[test]
    fn test_texture_loader_load() {
        let bmp = make_test_bmp(1, 1, &[255, 128, 64]);
        let loader = TextureLoader;
        let tex = loader.load(Path::new("test.bmp"), &bmp).unwrap();
        assert_eq!(tex.width, 1);
        assert_eq!(tex.height, 1);
        assert_eq!(&tex.pixels, &[255, 128, 64]);
    }

    // -- Full disk round-trip with AssetServer ----------------------------

    #[test]
    fn test_load_obj_from_disk() {
        let dir = std::env::temp_dir().join("genovo_format_obj_test");
        let _ = std::fs::create_dir_all(&dir);
        let obj_path = dir.join("tri.obj");
        std::fs::write(
            &obj_path,
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        )
        .unwrap();

        let server = crate::loader::AssetServer::new(dir.to_str().unwrap());
        server.register_loader(ObjLoader);

        let handle: crate::loader::AssetHandle<MeshData> =
            server.load_sync("tri.obj").unwrap();
        let mesh = server.get_cloned(&handle).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices.len(), 3);

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_load_bmp_from_disk() {
        let dir = std::env::temp_dir().join("genovo_format_bmp_test");
        let _ = std::fs::create_dir_all(&dir);
        let bmp_path = dir.join("test.bmp");
        let pixels: Vec<u8> = (0..16).flat_map(|_| [100u8, 150, 200]).collect();
        let bmp = make_test_bmp(4, 4, &pixels);
        std::fs::write(&bmp_path, &bmp).unwrap();

        let server = crate::loader::AssetServer::new(dir.to_str().unwrap());
        server.register_loader(TextureLoader);

        let handle: crate::loader::AssetHandle<TextureData> =
            server.load_sync("test.bmp").unwrap();
        let tex = server.get_cloned(&handle).unwrap();
        assert_eq!(tex.width, 4);
        assert_eq!(tex.height, 4);

        let _ = std::fs::remove_file(&bmp_path);
        let _ = std::fs::remove_dir(&dir);
    }
}
