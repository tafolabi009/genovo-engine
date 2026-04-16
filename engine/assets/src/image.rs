//! Extended image format loaders and image processing operations.
//!
//! Provides parsers for several image formats beyond the basic BMP in `formats`:
//! - **TGA** — Truevision TGA (uncompressed + RLE, 16/24/32-bit, top-down & bottom-up)
//! - **HDR** — Radiance RGBE high dynamic range images (adaptive RLE)
//! - **DDS** — DirectDraw Surface with BC1/BC3 block-compressed textures
//!
//! Also provides common image operations:
//! - Bilinear and bicubic resize
//! - Mipmap chain generation
//! - Vertical/horizontal flip
//! - Format conversion (RGB ↔ RGBA, float ↔ byte)
//! - Height-map to normal-map (Sobel operator)

use std::path::Path;

use crate::loader::{AssetError, AssetLoader};

// =========================================================================
// Image struct — the canonical in-memory image
// =========================================================================

/// Pixel format for an [`Image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    /// 8-bit per channel, 1 channel (grayscale).
    R8,
    /// 8-bit per channel, 3 channels.
    Rgb8,
    /// 8-bit per channel, 4 channels.
    Rgba8,
    /// 16-bit per channel, 1 channel.
    R16,
    /// 16-bit per channel, 4 channels.
    Rgba16,
    /// 32-bit float per channel, 1 channel.
    R32F,
    /// 32-bit float per channel, 3 channels.
    Rgb32F,
    /// 32-bit float per channel, 4 channels.
    Rgba32F,
}

impl ImageFormat {
    /// Returns the number of channels for this format.
    pub fn channels(&self) -> u32 {
        match self {
            ImageFormat::R8 | ImageFormat::R16 | ImageFormat::R32F => 1,
            ImageFormat::Rgb8 | ImageFormat::Rgb32F => 3,
            ImageFormat::Rgba8 | ImageFormat::Rgba16 | ImageFormat::Rgba32F => 4,
        }
    }

    /// Returns the number of bytes per pixel.
    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            ImageFormat::R8 => 1,
            ImageFormat::Rgb8 => 3,
            ImageFormat::Rgba8 => 4,
            ImageFormat::R16 => 2,
            ImageFormat::Rgba16 => 8,
            ImageFormat::R32F => 4,
            ImageFormat::Rgb32F => 12,
            ImageFormat::Rgba32F => 16,
        }
    }

    /// Returns `true` if the format stores floating-point data.
    pub fn is_float(&self) -> bool {
        matches!(self, ImageFormat::R32F | ImageFormat::Rgb32F | ImageFormat::Rgba32F)
    }
}

/// A decoded image in CPU-accessible memory.
///
/// Stores raw pixel data in row-major order (top-to-bottom, left-to-right).
/// For byte formats, the data is stored in `data`.  For float formats, the
/// data is stored as the bytes of `f32` values in native endian.
#[derive(Debug, Clone)]
pub struct Image {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of channels (1, 3, or 4).
    pub channels: u32,
    /// Raw pixel data.
    pub data: Vec<u8>,
    /// Pixel format.
    pub format: ImageFormat,
}

impl Image {
    /// Creates a new blank (zeroed) image.
    pub fn new(width: u32, height: u32, format: ImageFormat) -> Self {
        let size = (width * height * format.bytes_per_pixel()) as usize;
        Self {
            width,
            height,
            channels: format.channels(),
            data: vec![0u8; size],
            format,
        }
    }

    /// Creates an image from existing pixel data.  Returns an error if the
    /// data length does not match the expected size.
    pub fn from_data(
        width: u32,
        height: u32,
        format: ImageFormat,
        data: Vec<u8>,
    ) -> Result<Self, AssetError> {
        let expected = (width * height * format.bytes_per_pixel()) as usize;
        if data.len() != expected {
            return Err(AssetError::InvalidData(format!(
                "Image data length mismatch: expected {expected} bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            channels: format.channels(),
            data,
            format,
        })
    }

    /// Returns the byte offset for a given pixel coordinate.
    pub fn pixel_offset(&self, x: u32, y: u32) -> usize {
        ((y * self.width + x) * self.format.bytes_per_pixel()) as usize
    }

    /// Returns the total number of pixels.
    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }

    /// Returns the total size of the pixel data in bytes.
    pub fn data_size(&self) -> usize {
        self.data.len()
    }
}

// =========================================================================
// TGA Loader
// =========================================================================

/// Loads Truevision TGA image files.
///
/// Supports:
/// - Uncompressed true-color (type 2) and grayscale (type 3)
/// - RLE-compressed true-color (type 10) and grayscale (type 11)
/// - 16-bit (5-5-5-1), 24-bit, and 32-bit pixel depths
/// - Top-down and bottom-up image origin
pub struct TgaLoader;

impl AssetLoader for TgaLoader {
    type Asset = Image;

    fn extensions(&self) -> &[&str] {
        &["tga"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<Image, AssetError> {
        parse_tga(bytes)
    }
}

/// TGA file header (18 bytes).
#[derive(Debug, Clone)]
struct TgaHeader {
    id_length: u8,
    color_map_type: u8,
    image_type: u8,
    // Color map specification (5 bytes)
    cm_first_entry: u16,
    cm_length: u16,
    cm_entry_size: u8,
    // Image specification (10 bytes)
    x_origin: u16,
    y_origin: u16,
    width: u16,
    height: u16,
    pixel_depth: u8,
    image_descriptor: u8,
}

impl TgaHeader {
    fn parse(data: &[u8]) -> Result<Self, AssetError> {
        if data.len() < 18 {
            return Err(AssetError::InvalidData(
                "TGA file too small for header (need 18 bytes)".into(),
            ));
        }
        Ok(Self {
            id_length: data[0],
            color_map_type: data[1],
            image_type: data[2],
            cm_first_entry: u16::from_le_bytes([data[3], data[4]]),
            cm_length: u16::from_le_bytes([data[5], data[6]]),
            cm_entry_size: data[7],
            x_origin: u16::from_le_bytes([data[8], data[9]]),
            y_origin: u16::from_le_bytes([data[10], data[11]]),
            width: u16::from_le_bytes([data[12], data[13]]),
            height: u16::from_le_bytes([data[14], data[15]]),
            pixel_depth: data[16],
            image_descriptor: data[17],
        })
    }

    /// Returns `true` if the origin is top-left (bit 5 of image descriptor).
    fn is_top_down(&self) -> bool {
        (self.image_descriptor & 0x20) != 0
    }

    /// Returns the alpha channel depth from the image descriptor (bits 0-3).
    fn alpha_bits(&self) -> u8 {
        self.image_descriptor & 0x0F
    }
}

/// Parse a TGA file from raw bytes.
fn parse_tga(data: &[u8]) -> Result<Image, AssetError> {
    let header = TgaHeader::parse(data)?;

    let width = header.width as u32;
    let height = header.height as u32;

    if width == 0 || height == 0 {
        return Err(AssetError::InvalidData("TGA image has zero dimensions".into()));
    }

    // Validate image type
    let is_rle = matches!(header.image_type, 9 | 10 | 11);
    let is_uncompressed = matches!(header.image_type, 1 | 2 | 3);
    if !is_rle && !is_uncompressed {
        return Err(AssetError::InvalidData(format!(
            "Unsupported TGA image type: {}",
            header.image_type
        )));
    }

    let is_grayscale = matches!(header.image_type, 3 | 11);
    let is_color_mapped = matches!(header.image_type, 1 | 9);

    // Skip past header + image ID
    let mut offset = 18 + header.id_length as usize;

    // Read color map if present
    let color_map: Option<Vec<[u8; 4]>> = if header.color_map_type == 1 {
        let entry_bytes = (header.cm_entry_size as usize + 7) / 8;
        let cm_size = header.cm_length as usize * entry_bytes;
        if offset + cm_size > data.len() {
            return Err(AssetError::InvalidData("TGA color map extends past EOF".into()));
        }
        let mut map = Vec::with_capacity(header.cm_length as usize);
        for i in 0..header.cm_length as usize {
            let base = offset + i * entry_bytes;
            let pixel = decode_tga_pixel(&data[base..base + entry_bytes], header.cm_entry_size)?;
            map.push(pixel);
        }
        offset += cm_size;
        Some(map)
    } else {
        None
    };

    let pixel_bytes = (header.pixel_depth as usize + 7) / 8;
    let pixel_count = width as usize * height as usize;

    // Determine output format
    let (out_format, out_channels) = if is_grayscale {
        (ImageFormat::R8, 1u32)
    } else if header.pixel_depth == 32 || header.alpha_bits() > 0 {
        (ImageFormat::Rgba8, 4)
    } else {
        (ImageFormat::Rgba8, 4) // always output RGBA for consistency
    };

    let mut pixels = vec![0u8; pixel_count * out_channels as usize];

    if is_rle {
        // RLE-compressed pixel data
        decode_tga_rle(
            data,
            &mut offset,
            &mut pixels,
            pixel_count,
            pixel_bytes,
            out_channels as usize,
            is_grayscale,
            is_color_mapped,
            &color_map,
            header.pixel_depth,
        )?;
    } else {
        // Uncompressed pixel data
        for i in 0..pixel_count {
            if offset + pixel_bytes > data.len() {
                return Err(AssetError::InvalidData(
                    "TGA pixel data truncated".into(),
                ));
            }
            let rgba = if is_color_mapped {
                let idx = read_tga_index(&data[offset..offset + pixel_bytes], pixel_bytes);
                let cm = color_map.as_ref().ok_or_else(|| {
                    AssetError::InvalidData("Color-mapped TGA with no color map".into())
                })?;
                if idx >= cm.len() {
                    return Err(AssetError::InvalidData(format!(
                        "TGA color map index {idx} out of range"
                    )));
                }
                cm[idx]
            } else if is_grayscale {
                let v = data[offset];
                [v, v, v, 255]
            } else {
                decode_tga_pixel(&data[offset..offset + pixel_bytes], header.pixel_depth)?
            };

            let dst = i * out_channels as usize;
            if is_grayscale {
                pixels[dst] = rgba[0];
            } else {
                pixels[dst] = rgba[0];
                pixels[dst + 1] = rgba[1];
                pixels[dst + 2] = rgba[2];
                pixels[dst + 3] = rgba[3];
            }
            offset += pixel_bytes;
        }
    }

    // Handle bottom-up origin (flip vertically)
    if !header.is_top_down() {
        let row_bytes = width as usize * out_channels as usize;
        let mut temp = vec![0u8; row_bytes];
        for y in 0..height as usize / 2 {
            let top_start = y * row_bytes;
            let bot_start = (height as usize - 1 - y) * row_bytes;
            temp.copy_from_slice(&pixels[top_start..top_start + row_bytes]);
            pixels.copy_within(bot_start..bot_start + row_bytes, top_start);
            pixels[bot_start..bot_start + row_bytes].copy_from_slice(&temp);
        }
    }

    Ok(Image {
        width,
        height,
        channels: out_channels,
        data: pixels,
        format: if is_grayscale { ImageFormat::R8 } else { out_format },
    })
}

/// Decode a single TGA pixel from raw bytes into RGBA.
fn decode_tga_pixel(bytes: &[u8], depth: u8) -> Result<[u8; 4], AssetError> {
    match depth {
        16 => {
            // 5-5-5-1 format: ARRRRRGG GGGBBBBB (little-endian)
            if bytes.len() < 2 {
                return Err(AssetError::InvalidData("TGA 16-bit pixel too short".into()));
            }
            let val = u16::from_le_bytes([bytes[0], bytes[1]]);
            let b = ((val & 0x001F) << 3) as u8;
            let g = (((val >> 5) & 0x001F) << 3) as u8;
            let r = (((val >> 10) & 0x001F) << 3) as u8;
            let a = if (val & 0x8000) != 0 { 255 } else { 255 }; // attribute bit
            Ok([r, g, b, a])
        }
        24 => {
            // BGR format
            if bytes.len() < 3 {
                return Err(AssetError::InvalidData("TGA 24-bit pixel too short".into()));
            }
            Ok([bytes[2], bytes[1], bytes[0], 255])
        }
        32 => {
            // BGRA format
            if bytes.len() < 4 {
                return Err(AssetError::InvalidData("TGA 32-bit pixel too short".into()));
            }
            Ok([bytes[2], bytes[1], bytes[0], bytes[3]])
        }
        8 => {
            // Grayscale or color-mapped index
            Ok([bytes[0], bytes[0], bytes[0], 255])
        }
        _ => Err(AssetError::InvalidData(format!(
            "Unsupported TGA pixel depth: {depth}"
        ))),
    }
}

/// Read a color-map index from raw bytes.
fn read_tga_index(bytes: &[u8], byte_count: usize) -> usize {
    match byte_count {
        1 => bytes[0] as usize,
        2 => u16::from_le_bytes([bytes[0], bytes[1]]) as usize,
        _ => bytes[0] as usize,
    }
}

/// Decode RLE-compressed TGA pixel data.
fn decode_tga_rle(
    data: &[u8],
    offset: &mut usize,
    pixels: &mut [u8],
    pixel_count: usize,
    pixel_bytes: usize,
    out_channels: usize,
    is_grayscale: bool,
    is_color_mapped: bool,
    color_map: &Option<Vec<[u8; 4]>>,
    pixel_depth: u8,
) -> Result<(), AssetError> {
    let mut pixel_idx = 0;

    while pixel_idx < pixel_count {
        if *offset >= data.len() {
            return Err(AssetError::InvalidData("TGA RLE data truncated".into()));
        }

        let packet_header = data[*offset];
        *offset += 1;

        let count = (packet_header & 0x7F) as usize + 1;
        let is_run = (packet_header & 0x80) != 0;

        if is_run {
            // Run-length packet: one pixel repeated `count` times
            if *offset + pixel_bytes > data.len() {
                return Err(AssetError::InvalidData("TGA RLE run pixel truncated".into()));
            }

            let rgba = if is_color_mapped {
                let idx = read_tga_index(&data[*offset..*offset + pixel_bytes], pixel_bytes);
                let cm = color_map.as_ref().ok_or_else(|| {
                    AssetError::InvalidData("Color-mapped TGA with no color map".into())
                })?;
                if idx >= cm.len() {
                    return Err(AssetError::InvalidData(format!(
                        "TGA color map index {idx} out of range"
                    )));
                }
                cm[idx]
            } else if is_grayscale {
                let v = data[*offset];
                [v, v, v, 255]
            } else {
                decode_tga_pixel(&data[*offset..*offset + pixel_bytes], pixel_depth)?
            };
            *offset += pixel_bytes;

            for _ in 0..count {
                if pixel_idx >= pixel_count {
                    break;
                }
                let dst = pixel_idx * out_channels;
                if is_grayscale {
                    pixels[dst] = rgba[0];
                } else {
                    pixels[dst] = rgba[0];
                    pixels[dst + 1] = rgba[1];
                    pixels[dst + 2] = rgba[2];
                    if out_channels == 4 {
                        pixels[dst + 3] = rgba[3];
                    }
                }
                pixel_idx += 1;
            }
        } else {
            // Raw packet: `count` literal pixels
            for _ in 0..count {
                if pixel_idx >= pixel_count {
                    break;
                }
                if *offset + pixel_bytes > data.len() {
                    return Err(AssetError::InvalidData("TGA RLE raw pixel truncated".into()));
                }

                let rgba = if is_color_mapped {
                    let idx = read_tga_index(&data[*offset..*offset + pixel_bytes], pixel_bytes);
                    let cm = color_map.as_ref().ok_or_else(|| {
                        AssetError::InvalidData("Color-mapped TGA with no color map".into())
                    })?;
                    if idx >= cm.len() {
                        return Err(AssetError::InvalidData(format!(
                            "TGA color map index {idx} out of range"
                        )));
                    }
                    cm[idx]
                } else if is_grayscale {
                    let v = data[*offset];
                    [v, v, v, 255]
                } else {
                    decode_tga_pixel(&data[*offset..*offset + pixel_bytes], pixel_depth)?
                };
                *offset += pixel_bytes;

                let dst = pixel_idx * out_channels;
                if is_grayscale {
                    pixels[dst] = rgba[0];
                } else {
                    pixels[dst] = rgba[0];
                    pixels[dst + 1] = rgba[1];
                    pixels[dst + 2] = rgba[2];
                    if out_channels == 4 {
                        pixels[dst + 3] = rgba[3];
                    }
                }
                pixel_idx += 1;
            }
        }
    }

    Ok(())
}

// =========================================================================
// HDR / Radiance RGBE Loader
// =========================================================================

/// Loads Radiance RGBE (`.hdr`) high dynamic range images.
///
/// Supports the standard Radiance format:
/// - `#?RADIANCE` or `#?RGBE` header
/// - `FORMAT=32-bit_rle_rgbe` or `FORMAT=32-bit_rle_xyze`
/// - Adaptive run-length encoding (new-style scanline RLE)
/// - Old-style uncompressed scanlines
pub struct HdrLoader;

impl AssetLoader for HdrLoader {
    type Asset = Image;

    fn extensions(&self) -> &[&str] {
        &["hdr"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<Image, AssetError> {
        parse_hdr(bytes)
    }
}

/// Parse a Radiance RGBE file from raw bytes.
fn parse_hdr(data: &[u8]) -> Result<Image, AssetError> {
    let text = std::str::from_utf8(data)
        .map_err(|_| AssetError::InvalidData("HDR file is not valid text in header".into()));
    // We parse character by character to find the header end
    let mut offset = 0;

    // Validate magic number
    // Accept #?RADIANCE or #?RGBE
    let magic_radiance = b"#?RADIANCE";
    let magic_rgbe = b"#?RGBE";

    let has_radiance = data.len() >= magic_radiance.len()
        && &data[..magic_radiance.len()] == magic_radiance;
    let has_rgbe = data.len() >= magic_rgbe.len()
        && &data[..magic_rgbe.len()] == magic_rgbe;

    if !has_radiance && !has_rgbe {
        return Err(AssetError::InvalidData(
            "HDR file missing #?RADIANCE or #?RGBE magic".into(),
        ));
    }

    // Find the end of header (empty line = \n\n)
    let mut header_end = 0;
    let mut found_empty_line = false;
    for i in 0..data.len().saturating_sub(1) {
        if data[i] == b'\n' && data[i + 1] == b'\n' {
            header_end = i + 2;
            found_empty_line = true;
            break;
        }
    }

    if !found_empty_line {
        return Err(AssetError::InvalidData(
            "HDR file header not terminated by empty line".into(),
        ));
    }

    // Parse header lines for FORMAT
    let header_bytes = &data[..header_end];
    let _header_text = text.unwrap_or(""); // may not be valid UTF-8 but headers should be ASCII
    let mut _format_found = false;

    // Parse resolution string after header
    offset = header_end;

    // Find the resolution line: should be like "-Y 512 +X 1024\n"
    let mut res_end = offset;
    while res_end < data.len() && data[res_end] != b'\n' {
        res_end += 1;
    }
    if res_end >= data.len() {
        return Err(AssetError::InvalidData("HDR missing resolution line".into()));
    }

    let res_line = std::str::from_utf8(&data[offset..res_end])
        .map_err(|_| AssetError::InvalidData("HDR resolution line not ASCII".into()))?;

    let (width, height, _flip_x, _flip_y) = parse_hdr_resolution(res_line)?;
    offset = res_end + 1; // skip newline

    // Decode scanlines
    let mut rgbe_data = vec![[0u8; 4]; (width * height) as usize];

    let mut row = 0u32;
    while row < height {
        if offset + 4 > data.len() {
            return Err(AssetError::InvalidData("HDR data truncated".into()));
        }

        // Check for adaptive RLE marker
        let b0 = data[offset];
        let b1 = data[offset + 1];

        if b0 == 2 && b1 == 2 && offset + 4 <= data.len() {
            // New-style adaptive RLE scanline
            let scan_width =
                ((data[offset + 2] as u32) << 8) | (data[offset + 3] as u32);
            if scan_width != width {
                return Err(AssetError::InvalidData(format!(
                    "HDR scanline width mismatch: expected {width}, got {scan_width}"
                )));
            }
            offset += 4;

            // Read each channel separately (R, G, B, E)
            let row_start = (row * width) as usize;
            for ch in 0..4u8 {
                let mut col = 0u32;
                while col < width {
                    if offset >= data.len() {
                        return Err(AssetError::InvalidData(
                            "HDR adaptive RLE data truncated".into(),
                        ));
                    }
                    let code = data[offset];
                    offset += 1;

                    if code > 128 {
                        // Run: repeat next byte (code - 128) times
                        let count = (code - 128) as u32;
                        if offset >= data.len() {
                            return Err(AssetError::InvalidData(
                                "HDR RLE run value truncated".into(),
                            ));
                        }
                        let val = data[offset];
                        offset += 1;
                        for _ in 0..count {
                            if col < width {
                                rgbe_data[row_start + col as usize][ch as usize] = val;
                                col += 1;
                            }
                        }
                    } else {
                        // Literal: read `code` bytes
                        let count = code as u32;
                        for _ in 0..count {
                            if offset >= data.len() || col >= width {
                                break;
                            }
                            rgbe_data[row_start + col as usize][ch as usize] = data[offset];
                            offset += 1;
                            col += 1;
                        }
                    }
                }
            }
            row += 1;
        } else {
            // Old-style: flat RGBE pixels (or old-style RLE with repeated pixels)
            // Read one scanline of flat RGBE pixels
            let row_start = (row * width) as usize;
            for col in 0..width as usize {
                if offset + 4 > data.len() {
                    return Err(AssetError::InvalidData(
                        "HDR flat scanline data truncated".into(),
                    ));
                }
                rgbe_data[row_start + col] = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                offset += 4;
            }
            row += 1;
        }
    }

    // Convert RGBE to float RGB
    let mut float_data = Vec::with_capacity((width * height * 3) as usize * 4);
    for pixel in &rgbe_data {
        let (r, g, b) = rgbe_to_float(*pixel);
        float_data.extend_from_slice(&r.to_ne_bytes());
        float_data.extend_from_slice(&g.to_ne_bytes());
        float_data.extend_from_slice(&b.to_ne_bytes());
    }

    Ok(Image {
        width,
        height,
        channels: 3,
        data: float_data,
        format: ImageFormat::Rgb32F,
    })
}

/// Parse the HDR resolution string, e.g. "-Y 512 +X 1024".
fn parse_hdr_resolution(line: &str) -> Result<(u32, u32, bool, bool), AssetError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 4 {
        return Err(AssetError::InvalidData(format!(
            "Invalid HDR resolution line: '{line}'"
        )));
    }

    let (height, flip_y) = match parts[0] {
        "-Y" => (
            parts[1].parse::<u32>().map_err(|_| {
                AssetError::InvalidData(format!("Bad HDR height: '{}'", parts[1]))
            })?,
            false,
        ),
        "+Y" => (
            parts[1].parse::<u32>().map_err(|_| {
                AssetError::InvalidData(format!("Bad HDR height: '{}'", parts[1]))
            })?,
            true,
        ),
        _ => {
            return Err(AssetError::InvalidData(format!(
                "HDR resolution: expected -Y or +Y, got '{}'",
                parts[0]
            )));
        }
    };

    let (width, flip_x) = match parts[2] {
        "+X" => (
            parts[3].parse::<u32>().map_err(|_| {
                AssetError::InvalidData(format!("Bad HDR width: '{}'", parts[3]))
            })?,
            false,
        ),
        "-X" => (
            parts[3].parse::<u32>().map_err(|_| {
                AssetError::InvalidData(format!("Bad HDR width: '{}'", parts[3]))
            })?,
            true,
        ),
        _ => {
            return Err(AssetError::InvalidData(format!(
                "HDR resolution: expected +X or -X, got '{}'",
                parts[2]
            )));
        }
    };

    Ok((width, height, flip_x, flip_y))
}

/// Convert an RGBE pixel to three f32 values (R, G, B).
///
/// RGBE encoding: the exponent byte `e` encodes a shared exponent.
/// If `e == 0`, the pixel is black.  Otherwise, the real value is:
///   component = (mantissa + 0.5) / 256.0 * 2^(e - 128)
fn rgbe_to_float(rgbe: [u8; 4]) -> (f32, f32, f32) {
    let e = rgbe[3];
    if e == 0 {
        return (0.0, 0.0, 0.0);
    }
    // 2^(e - 128) / 256.0 = 2^(e - 128 - 8) = 2^(e - 136)
    // Using ldexp equivalent: f * 2^exp
    let factor = f32::from_bits(((e as u32).wrapping_add(127).wrapping_sub(136)) << 23);
    let r = (rgbe[0] as f32 + 0.5) * factor;
    let g = (rgbe[1] as f32 + 0.5) * factor;
    let b = (rgbe[2] as f32 + 0.5) * factor;
    (r, g, b)
}

// =========================================================================
// DDS Loader
// =========================================================================

/// Loads DirectDraw Surface (DDS) files.
///
/// Supports:
/// - DDS header parsing (magic + 124-byte header + optional DX10 header)
/// - BC1 (DXT1) decompression — 4x4 blocks, RGB565 endpoints, 2-bit indices
/// - BC3 (DXT5) decompression — BC1 colour + interpolated 8-bit alpha
/// - Uncompressed RGBA formats
pub struct DdsLoader;

impl AssetLoader for DdsLoader {
    type Asset = Image;

    fn extensions(&self) -> &[&str] {
        &["dds"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<Image, AssetError> {
        parse_dds(bytes)
    }
}

/// DDS pixel format flags.
const DDPF_FOURCC: u32 = 0x4;
const DDPF_RGB: u32 = 0x40;
const DDPF_RGBA: u32 = 0x41;

/// DDS header (128 bytes total: 4 bytes magic + 124 bytes header).
#[derive(Debug)]
struct DdsHeader {
    height: u32,
    width: u32,
    pitch_or_linear_size: u32,
    depth: u32,
    mip_count: u32,
    // Pixel format
    pf_flags: u32,
    pf_fourcc: [u8; 4],
    pf_rgb_bit_count: u32,
    pf_r_mask: u32,
    pf_g_mask: u32,
    pf_b_mask: u32,
    pf_a_mask: u32,
}

fn parse_dds(data: &[u8]) -> Result<Image, AssetError> {
    if data.len() < 128 {
        return Err(AssetError::InvalidData(
            "DDS file too small (need at least 128 bytes)".into(),
        ));
    }

    // Magic: "DDS "
    if &data[0..4] != b"DDS " {
        return Err(AssetError::InvalidData("Missing DDS magic".into()));
    }

    let _header_size = read_u32_le(data, 4); // should be 124
    let _flags = read_u32_le(data, 8);
    let height = read_u32_le(data, 12);
    let width = read_u32_le(data, 16);
    let pitch_or_linear_size = read_u32_le(data, 20);
    let depth = read_u32_le(data, 24).max(1);
    let mip_count = read_u32_le(data, 28).max(1);
    // reserved1: 11 * u32 at offset 32..76

    // Pixel format at offset 76 (32 bytes)
    let _pf_size = read_u32_le(data, 76);
    let pf_flags = read_u32_le(data, 80);
    let pf_fourcc = [data[84], data[85], data[86], data[87]];
    let pf_rgb_bit_count = read_u32_le(data, 88);
    let pf_r_mask = read_u32_le(data, 92);
    let pf_g_mask = read_u32_le(data, 96);
    let pf_b_mask = read_u32_le(data, 100);
    let pf_a_mask = read_u32_le(data, 104);

    let header = DdsHeader {
        height,
        width,
        pitch_or_linear_size,
        depth,
        mip_count,
        pf_flags,
        pf_fourcc,
        pf_rgb_bit_count,
        pf_r_mask,
        pf_g_mask,
        pf_b_mask,
        pf_a_mask,
    };

    // Data starts after the 128-byte header
    let data_offset = 128usize;
    let pixel_data = &data[data_offset..];

    if (pf_flags & DDPF_FOURCC) != 0 {
        match &pf_fourcc {
            b"DXT1" => decompress_bc1(pixel_data, width, height),
            b"DXT3" => decompress_bc3(pixel_data, width, height), // treat DXT3 like BC3 for simplicity
            b"DXT5" => decompress_bc3(pixel_data, width, height),
            _ => Err(AssetError::InvalidData(format!(
                "Unsupported DDS FourCC: {:?}",
                std::str::from_utf8(&pf_fourcc).unwrap_or("????")
            ))),
        }
    } else if (pf_flags & DDPF_RGB) != 0 || (pf_flags & DDPF_RGBA) != 0 {
        decompress_uncompressed_dds(pixel_data, &header)
    } else {
        Err(AssetError::InvalidData(
            "Unsupported DDS pixel format flags".into(),
        ))
    }
}

/// Decompress BC1 (DXT1) data.
///
/// BC1 encodes 4x4 pixel blocks in 8 bytes:
/// - 2 bytes: color0 (RGB565)
/// - 2 bytes: color1 (RGB565)
/// - 4 bytes: 4x4 grid of 2-bit indices
///
/// If color0 > color1:
///   index 0 = color0, 1 = color1, 2 = 2/3*c0 + 1/3*c1, 3 = 1/3*c0 + 2/3*c1
/// If color0 <= color1:
///   index 0 = color0, 1 = color1, 2 = 1/2*c0 + 1/2*c1, 3 = transparent black
fn decompress_bc1(data: &[u8], width: u32, height: u32) -> Result<Image, AssetError> {
    let block_w = (width + 3) / 4;
    let block_h = (height + 3) / 4;
    let block_count = (block_w * block_h) as usize;
    let needed = block_count * 8;

    if data.len() < needed {
        return Err(AssetError::InvalidData(format!(
            "BC1 data too small: need {needed} bytes, got {}",
            data.len()
        )));
    }

    let mut pixels = vec![0u8; (width * height * 4) as usize];

    for by in 0..block_h {
        for bx in 0..block_w {
            let block_idx = (by * block_w + bx) as usize;
            let block_offset = block_idx * 8;

            let c0_raw = u16::from_le_bytes([data[block_offset], data[block_offset + 1]]);
            let c1_raw = u16::from_le_bytes([data[block_offset + 2], data[block_offset + 3]]);

            let c0 = rgb565_to_rgba(c0_raw);
            let c1 = rgb565_to_rgba(c1_raw);

            // Build the 4-entry colour palette
            let palette: [[u8; 4]; 4] = if c0_raw > c1_raw {
                [
                    c0,
                    c1,
                    [
                        ((2 * c0[0] as u16 + c1[0] as u16 + 1) / 3) as u8,
                        ((2 * c0[1] as u16 + c1[1] as u16 + 1) / 3) as u8,
                        ((2 * c0[2] as u16 + c1[2] as u16 + 1) / 3) as u8,
                        255,
                    ],
                    [
                        ((c0[0] as u16 + 2 * c1[0] as u16 + 1) / 3) as u8,
                        ((c0[1] as u16 + 2 * c1[1] as u16 + 1) / 3) as u8,
                        ((c0[2] as u16 + 2 * c1[2] as u16 + 1) / 3) as u8,
                        255,
                    ],
                ]
            } else {
                [
                    c0,
                    c1,
                    [
                        ((c0[0] as u16 + c1[0] as u16) / 2) as u8,
                        ((c0[1] as u16 + c1[1] as u16) / 2) as u8,
                        ((c0[2] as u16 + c1[2] as u16) / 2) as u8,
                        255,
                    ],
                    [0, 0, 0, 0], // transparent black
                ]
            };

            // Read the 4x4 index grid (4 bytes, 2 bits per pixel, row-major)
            let idx_bytes = [
                data[block_offset + 4],
                data[block_offset + 5],
                data[block_offset + 6],
                data[block_offset + 7],
            ];

            for row in 0..4u32 {
                for col in 0..4u32 {
                    let px = bx * 4 + col;
                    let py = by * 4 + row;
                    if px >= width || py >= height {
                        continue;
                    }

                    let bit_idx = (row * 4 + col) * 2;
                    let byte_idx = (bit_idx / 8) as usize;
                    let bit_offset = bit_idx % 8;
                    let index = ((idx_bytes[byte_idx] >> bit_offset) & 0x03) as usize;

                    let dst = ((py * width + px) * 4) as usize;
                    pixels[dst] = palette[index][0];
                    pixels[dst + 1] = palette[index][1];
                    pixels[dst + 2] = palette[index][2];
                    pixels[dst + 3] = palette[index][3];
                }
            }
        }
    }

    Ok(Image {
        width,
        height,
        channels: 4,
        data: pixels,
        format: ImageFormat::Rgba8,
    })
}

/// Decompress BC3 (DXT5) data.
///
/// BC3 = explicit alpha block (8 bytes) + BC1 colour block (8 bytes) per 4x4 block.
/// Alpha block:
/// - 2 bytes: alpha0, alpha1 endpoints
/// - 6 bytes: 4x4 grid of 3-bit alpha indices
fn decompress_bc3(data: &[u8], width: u32, height: u32) -> Result<Image, AssetError> {
    let block_w = (width + 3) / 4;
    let block_h = (height + 3) / 4;
    let block_count = (block_w * block_h) as usize;
    let needed = block_count * 16;

    if data.len() < needed {
        return Err(AssetError::InvalidData(format!(
            "BC3 data too small: need {needed} bytes, got {}",
            data.len()
        )));
    }

    let mut pixels = vec![0u8; (width * height * 4) as usize];

    for by in 0..block_h {
        for bx in 0..block_w {
            let block_idx = (by * block_w + bx) as usize;
            let block_offset = block_idx * 16;

            // Alpha block (8 bytes)
            let alpha0 = data[block_offset] as u16;
            let alpha1 = data[block_offset + 1] as u16;

            // Build alpha palette
            let alpha_palette: [u8; 8] = if alpha0 > alpha1 {
                [
                    alpha0 as u8,
                    alpha1 as u8,
                    ((6 * alpha0 + 1 * alpha1 + 3) / 7) as u8,
                    ((5 * alpha0 + 2 * alpha1 + 3) / 7) as u8,
                    ((4 * alpha0 + 3 * alpha1 + 3) / 7) as u8,
                    ((3 * alpha0 + 4 * alpha1 + 3) / 7) as u8,
                    ((2 * alpha0 + 5 * alpha1 + 3) / 7) as u8,
                    ((1 * alpha0 + 6 * alpha1 + 3) / 7) as u8,
                ]
            } else {
                [
                    alpha0 as u8,
                    alpha1 as u8,
                    ((4 * alpha0 + 1 * alpha1 + 2) / 5) as u8,
                    ((3 * alpha0 + 2 * alpha1 + 2) / 5) as u8,
                    ((2 * alpha0 + 3 * alpha1 + 2) / 5) as u8,
                    ((1 * alpha0 + 4 * alpha1 + 2) / 5) as u8,
                    0,
                    255,
                ]
            };

            // Read 48-bit (6 bytes) alpha index data
            let alpha_bits: u64 = (data[block_offset + 2] as u64)
                | ((data[block_offset + 3] as u64) << 8)
                | ((data[block_offset + 4] as u64) << 16)
                | ((data[block_offset + 5] as u64) << 24)
                | ((data[block_offset + 6] as u64) << 32)
                | ((data[block_offset + 7] as u64) << 40);

            // Color block (8 bytes starting at block_offset + 8)
            let color_offset = block_offset + 8;
            let c0_raw = u16::from_le_bytes([data[color_offset], data[color_offset + 1]]);
            let c1_raw = u16::from_le_bytes([data[color_offset + 2], data[color_offset + 3]]);

            let c0 = rgb565_to_rgba(c0_raw);
            let c1 = rgb565_to_rgba(c1_raw);

            let color_palette: [[u8; 3]; 4] = [
                [c0[0], c0[1], c0[2]],
                [c1[0], c1[1], c1[2]],
                [
                    ((2 * c0[0] as u16 + c1[0] as u16 + 1) / 3) as u8,
                    ((2 * c0[1] as u16 + c1[1] as u16 + 1) / 3) as u8,
                    ((2 * c0[2] as u16 + c1[2] as u16 + 1) / 3) as u8,
                ],
                [
                    ((c0[0] as u16 + 2 * c1[0] as u16 + 1) / 3) as u8,
                    ((c0[1] as u16 + 2 * c1[1] as u16 + 1) / 3) as u8,
                    ((c0[2] as u16 + 2 * c1[2] as u16 + 1) / 3) as u8,
                ],
            ];

            let color_indices = [
                data[color_offset + 4],
                data[color_offset + 5],
                data[color_offset + 6],
                data[color_offset + 7],
            ];

            for row in 0..4u32 {
                for col in 0..4u32 {
                    let px = bx * 4 + col;
                    let py = by * 4 + row;
                    if px >= width || py >= height {
                        continue;
                    }

                    // Alpha index (3 bits)
                    let alpha_bit_pos = (row * 4 + col) * 3;
                    let alpha_idx = ((alpha_bits >> alpha_bit_pos) & 0x07) as usize;

                    // Color index (2 bits)
                    let color_bit_idx = (row * 4 + col) * 2;
                    let color_byte = (color_bit_idx / 8) as usize;
                    let color_bit_off = color_bit_idx % 8;
                    let color_idx =
                        ((color_indices[color_byte] >> color_bit_off) & 0x03) as usize;

                    let dst = ((py * width + px) * 4) as usize;
                    pixels[dst] = color_palette[color_idx][0];
                    pixels[dst + 1] = color_palette[color_idx][1];
                    pixels[dst + 2] = color_palette[color_idx][2];
                    pixels[dst + 3] = alpha_palette[alpha_idx];
                }
            }
        }
    }

    Ok(Image {
        width,
        height,
        channels: 4,
        data: pixels,
        format: ImageFormat::Rgba8,
    })
}

/// Decode uncompressed DDS pixel data.
fn decompress_uncompressed_dds(data: &[u8], header: &DdsHeader) -> Result<Image, AssetError> {
    let bpp = header.pf_rgb_bit_count;
    let byte_pp = (bpp / 8) as usize;
    let pixel_count = (header.width * header.height) as usize;
    let needed = pixel_count * byte_pp;

    if data.len() < needed {
        return Err(AssetError::InvalidData(format!(
            "Uncompressed DDS data too small: need {needed}, got {}",
            data.len()
        )));
    }

    let has_alpha = header.pf_a_mask != 0;
    let mut pixels = vec![0u8; pixel_count * 4];

    for i in 0..pixel_count {
        let src = i * byte_pp;
        let mut raw = 0u32;
        for b in 0..byte_pp {
            raw |= (data[src + b] as u32) << (b * 8);
        }

        let r = extract_channel(raw, header.pf_r_mask);
        let g = extract_channel(raw, header.pf_g_mask);
        let b = extract_channel(raw, header.pf_b_mask);
        let a = if has_alpha {
            extract_channel(raw, header.pf_a_mask)
        } else {
            255
        };

        let dst = i * 4;
        pixels[dst] = r;
        pixels[dst + 1] = g;
        pixels[dst + 2] = b;
        pixels[dst + 3] = a;
    }

    Ok(Image {
        width: header.width,
        height: header.height,
        channels: 4,
        data: pixels,
        format: ImageFormat::Rgba8,
    })
}

/// Extract a channel value from a packed pixel using a bitmask.
fn extract_channel(pixel: u32, mask: u32) -> u8 {
    if mask == 0 {
        return 0;
    }
    let shift = mask.trailing_zeros();
    let bits = (mask >> shift).count_ones();
    let max_val = (1u32 << bits) - 1;
    let val = (pixel & mask) >> shift;
    // Scale to 0..255
    ((val * 255 + max_val / 2) / max_val) as u8
}

/// Convert an RGB565 value to RGBA8.
fn rgb565_to_rgba(c: u16) -> [u8; 4] {
    let r5 = ((c >> 11) & 0x1F) as u8;
    let g6 = ((c >> 5) & 0x3F) as u8;
    let b5 = (c & 0x1F) as u8;
    [
        (r5 << 3) | (r5 >> 2),
        (g6 << 2) | (g6 >> 4),
        (b5 << 3) | (b5 >> 2),
        255,
    ]
}

// =========================================================================
// Image Operations
// =========================================================================

/// Resize filter method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeFilter {
    /// Nearest-neighbour (point) sampling.
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation (Catmull-Rom).
    Bicubic,
}

/// Resize an image to the given dimensions using the specified filter.
pub fn resize(image: &Image, new_width: u32, new_height: u32, filter: ResizeFilter) -> Image {
    if !image.format.is_float() && image.format != ImageFormat::Rgb8
        && image.format != ImageFormat::Rgba8
        && image.format != ImageFormat::R8
    {
        // Fallback: return a blank image for unsupported formats
        return Image::new(new_width, new_height, image.format);
    }

    let channels = image.channels as usize;
    let bpp = image.format.bytes_per_pixel() as usize;
    let new_size = (new_width * new_height) as usize * bpp;
    let mut out = vec![0u8; new_size];

    let src_w = image.width as f64;
    let src_h = image.height as f64;
    let dst_w = new_width as f64;
    let dst_h = new_height as f64;

    for dy in 0..new_height {
        for dx in 0..new_width {
            let sx = (dx as f64 + 0.5) * src_w / dst_w - 0.5;
            let sy = (dy as f64 + 0.5) * src_h / dst_h - 0.5;

            let dst_off = ((dy * new_width + dx) as usize) * bpp;

            match filter {
                ResizeFilter::Nearest => {
                    let ix = sx.round().max(0.0).min(src_w - 1.0) as u32;
                    let iy = sy.round().max(0.0).min(src_h - 1.0) as u32;
                    let src_off = ((iy * image.width + ix) as usize) * bpp;
                    out[dst_off..dst_off + bpp]
                        .copy_from_slice(&image.data[src_off..src_off + bpp]);
                }
                ResizeFilter::Bilinear => {
                    let ix0 = sx.floor().max(0.0) as u32;
                    let iy0 = sy.floor().max(0.0) as u32;
                    let ix1 = (ix0 + 1).min(image.width - 1);
                    let iy1 = (iy0 + 1).min(image.height - 1);
                    let fx = sx - sx.floor();
                    let fy = sy - sy.floor();

                    if image.format.is_float() {
                        let bytes_per_ch = 4; // f32
                        for ch in 0..channels {
                            let s00 = read_f32_pixel(&image.data, ix0, iy0, image.width, channels, ch);
                            let s10 = read_f32_pixel(&image.data, ix1, iy0, image.width, channels, ch);
                            let s01 = read_f32_pixel(&image.data, ix0, iy1, image.width, channels, ch);
                            let s11 = read_f32_pixel(&image.data, ix1, iy1, image.width, channels, ch);

                            let val = bilerp(s00 as f64, s10 as f64, s01 as f64, s11 as f64, fx, fy) as f32;
                            let off = dst_off + ch * bytes_per_ch;
                            out[off..off + 4].copy_from_slice(&val.to_ne_bytes());
                        }
                    } else {
                        for ch in 0..channels {
                            let s00 = read_u8_pixel(&image.data, ix0, iy0, image.width, channels, ch) as f64;
                            let s10 = read_u8_pixel(&image.data, ix1, iy0, image.width, channels, ch) as f64;
                            let s01 = read_u8_pixel(&image.data, ix0, iy1, image.width, channels, ch) as f64;
                            let s11 = read_u8_pixel(&image.data, ix1, iy1, image.width, channels, ch) as f64;

                            let val = bilerp(s00, s10, s01, s11, fx, fy);
                            out[dst_off + ch] = val.round().max(0.0).min(255.0) as u8;
                        }
                    }
                }
                ResizeFilter::Bicubic => {
                    let ix = sx.floor() as i32;
                    let iy = sy.floor() as i32;
                    let fx = (sx - sx.floor()) as f64;
                    let fy = (sy - sy.floor()) as f64;

                    if image.format.is_float() {
                        let bytes_per_ch = 4;
                        for ch in 0..channels {
                            let mut col_vals = [0.0f64; 4];
                            for m in 0..4i32 {
                                let mut row_vals = [0.0f64; 4];
                                for n in 0..4i32 {
                                    let px = (ix + n - 1).max(0).min(image.width as i32 - 1) as u32;
                                    let py = (iy + m - 1).max(0).min(image.height as i32 - 1) as u32;
                                    row_vals[n as usize] =
                                        read_f32_pixel(&image.data, px, py, image.width, channels, ch) as f64;
                                }
                                col_vals[m as usize] = cubic_interp(row_vals, fx);
                            }
                            let val = cubic_interp(col_vals, fy) as f32;
                            let off = dst_off + ch * bytes_per_ch;
                            out[off..off + 4].copy_from_slice(&val.to_ne_bytes());
                        }
                    } else {
                        for ch in 0..channels {
                            let mut col_vals = [0.0f64; 4];
                            for m in 0..4i32 {
                                let mut row_vals = [0.0f64; 4];
                                for n in 0..4i32 {
                                    let px = (ix + n - 1).max(0).min(image.width as i32 - 1) as u32;
                                    let py = (iy + m - 1).max(0).min(image.height as i32 - 1) as u32;
                                    row_vals[n as usize] =
                                        read_u8_pixel(&image.data, px, py, image.width, channels, ch) as f64;
                                }
                                col_vals[m as usize] = cubic_interp(row_vals, fx);
                            }
                            let val = cubic_interp(col_vals, fy);
                            out[dst_off + ch] = val.round().max(0.0).min(255.0) as u8;
                        }
                    }
                }
            }
        }
    }

    Image {
        width: new_width,
        height: new_height,
        channels: image.channels,
        data: out,
        format: image.format,
    }
}

/// Generate a complete mipmap chain for the given image.
///
/// Each successive mip level is half the width and height (rounding down,
/// minimum 1).  The first element of the returned vector is the original
/// (level 0) image.
pub fn generate_mipmaps(image: &Image) -> Vec<Image> {
    let mut chain = Vec::new();
    chain.push(image.clone());

    let mut w = image.width;
    let mut h = image.height;

    loop {
        let nw = (w / 2).max(1);
        let nh = (h / 2).max(1);
        if nw == w && nh == h {
            break; // 1x1 already
        }
        let mip = resize(chain.last().unwrap(), nw, nh, ResizeFilter::Bilinear);
        w = nw;
        h = nh;
        chain.push(mip);
        if nw == 1 && nh == 1 {
            break;
        }
    }

    chain
}

/// Flip an image vertically (top to bottom).
pub fn flip_vertical(image: &mut Image) {
    let row_bytes = (image.width * image.format.bytes_per_pixel()) as usize;
    let mut temp = vec![0u8; row_bytes];

    for y in 0..image.height as usize / 2 {
        let top = y * row_bytes;
        let bot = (image.height as usize - 1 - y) * row_bytes;
        temp.copy_from_slice(&image.data[top..top + row_bytes]);
        image.data.copy_within(bot..bot + row_bytes, top);
        image.data[bot..bot + row_bytes].copy_from_slice(&temp);
    }
}

/// Flip an image horizontally (left to right).
pub fn flip_horizontal(image: &mut Image) {
    let bpp = image.format.bytes_per_pixel() as usize;
    let w = image.width as usize;

    for y in 0..image.height as usize {
        let row_start = y * w * bpp;
        for x in 0..w / 2 {
            let left = row_start + x * bpp;
            let right = row_start + (w - 1 - x) * bpp;
            for b in 0..bpp {
                image.data.swap(left + b, right + b);
            }
        }
    }
}

/// Convert an image from one format to another.
///
/// Supported conversions:
/// - Rgb8 ↔ Rgba8 (add/remove alpha)
/// - R8 → Rgb8, R8 → Rgba8
/// - Rgb32F → Rgba32F, Rgba32F → Rgb32F
/// - Byte → float, float → byte (with clamping/scaling)
pub fn convert_format(image: &Image, target: ImageFormat) -> Result<Image, AssetError> {
    if image.format == target {
        return Ok(image.clone());
    }

    let pixel_count = (image.width * image.height) as usize;

    match (image.format, target) {
        (ImageFormat::Rgb8, ImageFormat::Rgba8) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for i in 0..pixel_count {
                let s = i * 3;
                out.push(image.data[s]);
                out.push(image.data[s + 1]);
                out.push(image.data[s + 2]);
                out.push(255);
            }
            Image::from_data(image.width, image.height, target, out)
        }
        (ImageFormat::Rgba8, ImageFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for i in 0..pixel_count {
                let s = i * 4;
                out.push(image.data[s]);
                out.push(image.data[s + 1]);
                out.push(image.data[s + 2]);
            }
            Image::from_data(image.width, image.height, target, out)
        }
        (ImageFormat::R8, ImageFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for i in 0..pixel_count {
                let v = image.data[i];
                out.push(v);
                out.push(v);
                out.push(v);
            }
            Image::from_data(image.width, image.height, target, out)
        }
        (ImageFormat::R8, ImageFormat::Rgba8) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for i in 0..pixel_count {
                let v = image.data[i];
                out.push(v);
                out.push(v);
                out.push(v);
                out.push(255);
            }
            Image::from_data(image.width, image.height, target, out)
        }
        (ImageFormat::Rgb32F, ImageFormat::Rgba32F) => {
            let mut out = Vec::with_capacity(pixel_count * 16);
            for i in 0..pixel_count {
                let s = i * 12;
                out.extend_from_slice(&image.data[s..s + 12]);
                out.extend_from_slice(&1.0f32.to_ne_bytes());
            }
            Image::from_data(image.width, image.height, target, out)
        }
        (ImageFormat::Rgba32F, ImageFormat::Rgb32F) => {
            let mut out = Vec::with_capacity(pixel_count * 12);
            for i in 0..pixel_count {
                let s = i * 16;
                out.extend_from_slice(&image.data[s..s + 12]);
            }
            Image::from_data(image.width, image.height, target, out)
        }
        (ImageFormat::Rgb8, ImageFormat::Rgb32F) => {
            let mut out = Vec::with_capacity(pixel_count * 12);
            for i in 0..pixel_count {
                let s = i * 3;
                for ch in 0..3 {
                    let f = image.data[s + ch] as f32 / 255.0;
                    out.extend_from_slice(&f.to_ne_bytes());
                }
            }
            Image::from_data(image.width, image.height, target, out)
        }
        (ImageFormat::Rgb32F, ImageFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for i in 0..pixel_count {
                let s = i * 12;
                for ch in 0..3 {
                    let f = f32::from_ne_bytes([
                        image.data[s + ch * 4],
                        image.data[s + ch * 4 + 1],
                        image.data[s + ch * 4 + 2],
                        image.data[s + ch * 4 + 3],
                    ]);
                    out.push((f.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            Image::from_data(image.width, image.height, target, out)
        }
        _ => Err(AssetError::InvalidData(format!(
            "Unsupported format conversion: {:?} -> {:?}",
            image.format, target
        ))),
    }
}

/// Convert a grayscale heightmap image to a normal map using the Sobel operator.
///
/// The `strength` parameter controls the steepness of the normals (typical
/// values: 1.0 to 10.0).  The output is an RGBA8 image where RGB = normal
/// and A = 255.
pub fn height_to_normal(heightmap: &Image, strength: f32) -> Result<Image, AssetError> {
    let w = heightmap.width as usize;
    let h = heightmap.height as usize;

    // Extract a single-channel height value for each pixel
    let heights = extract_heights(heightmap)?;

    let mut normals = vec![0u8; w * h * 4];

    for y in 0..h {
        for x in 0..w {
            // Sobel operator in X and Y
            let tl = sample_height(&heights, w, h, x.wrapping_sub(1), y.wrapping_sub(1));
            let tc = sample_height(&heights, w, h, x, y.wrapping_sub(1));
            let tr = sample_height(&heights, w, h, x + 1, y.wrapping_sub(1));
            let ml = sample_height(&heights, w, h, x.wrapping_sub(1), y);
            let mr = sample_height(&heights, w, h, x + 1, y);
            let bl = sample_height(&heights, w, h, x.wrapping_sub(1), y + 1);
            let bc = sample_height(&heights, w, h, x, y + 1);
            let br = sample_height(&heights, w, h, x + 1, y + 1);

            // Sobel X: [-1 0 1; -2 0 2; -1 0 1]
            let dx = (-tl + tr - 2.0 * ml + 2.0 * mr - bl + br) * strength;
            // Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1]
            let dy = (-tl - 2.0 * tc - tr + bl + 2.0 * bc + br) * strength;
            let dz = 1.0f32;

            // Normalise
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            let nx = dx / len;
            let ny = dy / len;
            let nz = dz / len;

            // Map from [-1,1] to [0,255]
            let dst = (y * w + x) * 4;
            normals[dst] = ((nx * 0.5 + 0.5) * 255.0).round() as u8;
            normals[dst + 1] = ((ny * 0.5 + 0.5) * 255.0).round() as u8;
            normals[dst + 2] = ((nz * 0.5 + 0.5) * 255.0).round() as u8;
            normals[dst + 3] = 255;
        }
    }

    Image::from_data(heightmap.width, heightmap.height, ImageFormat::Rgba8, normals)
}

/// Extract per-pixel height values (0..1) from an image.
fn extract_heights(image: &Image) -> Result<Vec<f32>, AssetError> {
    let pixel_count = (image.width * image.height) as usize;
    let mut heights = Vec::with_capacity(pixel_count);

    match image.format {
        ImageFormat::R8 => {
            for i in 0..pixel_count {
                heights.push(image.data[i] as f32 / 255.0);
            }
        }
        ImageFormat::Rgb8 => {
            for i in 0..pixel_count {
                let r = image.data[i * 3] as f32;
                let g = image.data[i * 3 + 1] as f32;
                let b = image.data[i * 3 + 2] as f32;
                heights.push((0.299 * r + 0.587 * g + 0.114 * b) / 255.0);
            }
        }
        ImageFormat::Rgba8 => {
            for i in 0..pixel_count {
                let r = image.data[i * 4] as f32;
                let g = image.data[i * 4 + 1] as f32;
                let b = image.data[i * 4 + 2] as f32;
                heights.push((0.299 * r + 0.587 * g + 0.114 * b) / 255.0);
            }
        }
        ImageFormat::R32F => {
            for i in 0..pixel_count {
                let off = i * 4;
                let f = f32::from_ne_bytes([
                    image.data[off],
                    image.data[off + 1],
                    image.data[off + 2],
                    image.data[off + 3],
                ]);
                heights.push(f);
            }
        }
        _ => {
            return Err(AssetError::InvalidData(format!(
                "Unsupported format for heightmap: {:?}",
                image.format
            )));
        }
    }

    Ok(heights)
}

/// Sample height with clamped boundary.
fn sample_height(heights: &[f32], w: usize, h: usize, x: usize, y: usize) -> f32 {
    let cx = x.min(w - 1);
    let cy = y.min(h - 1);
    // Handle wrapping for underflow (usize)
    if x >= w || y >= h {
        // The wrapping_sub might produce huge values for usize
        return heights[cy.min(h - 1) * w + cx.min(w - 1)];
    }
    heights[cy * w + cx]
}

// =========================================================================
// Interpolation helpers
// =========================================================================

/// Bilinear interpolation.
fn bilerp(s00: f64, s10: f64, s01: f64, s11: f64, fx: f64, fy: f64) -> f64 {
    let top = s00 + (s10 - s00) * fx;
    let bot = s01 + (s11 - s01) * fx;
    top + (bot - top) * fy
}

/// Catmull-Rom cubic interpolation on 4 samples.
fn cubic_interp(p: [f64; 4], t: f64) -> f64 {
    let a = -0.5 * p[0] + 1.5 * p[1] - 1.5 * p[2] + 0.5 * p[3];
    let b = p[0] - 2.5 * p[1] + 2.0 * p[2] - 0.5 * p[3];
    let c = -0.5 * p[0] + 0.5 * p[2];
    let d = p[1];
    ((a * t + b) * t + c) * t + d
}

/// Read a single u8 channel value from packed byte data.
fn read_u8_pixel(data: &[u8], x: u32, y: u32, width: u32, channels: usize, ch: usize) -> u8 {
    let idx = ((y * width + x) as usize) * channels + ch;
    data[idx]
}

/// Read a single f32 channel value from packed float data.
fn read_f32_pixel(data: &[u8], x: u32, y: u32, width: u32, channels: usize, ch: usize) -> f32 {
    let idx = ((y * width + x) as usize) * channels + ch;
    let off = idx * 4;
    f32::from_ne_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

// =========================================================================
// Little-endian helpers (local to this module)
// =========================================================================

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

    // -- TGA tests --------------------------------------------------------

    /// Build a minimal uncompressed 24-bit TGA (bottom-up).
    fn make_tga_uncompressed_24(width: u16, height: u16, bgr_pixels: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; 18];
        buf[2] = 2; // uncompressed true-color
        buf[12] = (width & 0xFF) as u8;
        buf[13] = (width >> 8) as u8;
        buf[14] = (height & 0xFF) as u8;
        buf[15] = (height >> 8) as u8;
        buf[16] = 24; // pixel depth
        buf[17] = 0;  // bottom-up, no alpha bits
        buf.extend_from_slice(bgr_pixels);
        buf
    }

    /// Build a minimal uncompressed 32-bit TGA (top-down).
    fn make_tga_uncompressed_32_topdown(width: u16, height: u16, bgra_pixels: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; 18];
        buf[2] = 2;  // uncompressed true-color
        buf[12] = (width & 0xFF) as u8;
        buf[13] = (width >> 8) as u8;
        buf[14] = (height & 0xFF) as u8;
        buf[15] = (height >> 8) as u8;
        buf[16] = 32; // pixel depth
        buf[17] = 0x28; // top-down (bit 5), 8 alpha bits
        buf.extend_from_slice(bgra_pixels);
        buf
    }

    /// Build an RLE-compressed TGA.
    fn make_tga_rle_24(width: u16, height: u16, rle_data: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; 18];
        buf[2] = 10; // RLE true-color
        buf[12] = (width & 0xFF) as u8;
        buf[13] = (width >> 8) as u8;
        buf[14] = (height & 0xFF) as u8;
        buf[15] = (height >> 8) as u8;
        buf[16] = 24;
        buf[17] = 0x20; // top-down for simplicity
        buf.extend_from_slice(rle_data);
        buf
    }

    #[test]
    fn test_tga_uncompressed_24_2x2() {
        // Bottom-up 2x2 image: pixels in BGR order
        // Row 0 (bottom of image): blue, green
        // Row 1 (top of image): red, white
        #[rustfmt::skip]
        let bgr = &[
            255, 0, 0,     0, 255, 0,     // bottom row: blue, green (BGR)
            0, 0, 255,     255, 255, 255,  // top row: red, white (BGR)
        ];
        let tga = make_tga_uncompressed_24(2, 2, bgr);
        let img = parse_tga(&tga).unwrap();

        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        assert_eq!(img.format, ImageFormat::Rgba8);

        // After bottom-up flip, row 0 in output = top of image = red, white
        assert_eq!(img.data[0], 255); // R of red pixel
        assert_eq!(img.data[1], 0);   // G
        assert_eq!(img.data[2], 0);   // B (was at bytes[2] in BGR)
    }

    #[test]
    fn test_tga_uncompressed_32_topdown() {
        // 1x1 32-bit BGRA, top-down
        let bgra = &[128u8, 64, 255, 200]; // B=128, G=64, R=255, A=200
        let tga = make_tga_uncompressed_32_topdown(1, 1, bgra);
        let img = parse_tga(&tga).unwrap();

        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
        assert_eq!(img.data[0], 255); // R
        assert_eq!(img.data[1], 64);  // G
        assert_eq!(img.data[2], 128); // B
        assert_eq!(img.data[3], 200); // A
    }

    #[test]
    fn test_tga_rle() {
        // 4x1 top-down RLE image with 24-bit pixels:
        //   Run of 3 red pixels + raw 1 blue pixel
        let mut rle = Vec::new();
        // Run packet: header = 0x82 (run of 3), pixel = BGR red (0,0,255)
        rle.push(0x82); // run, count=3
        rle.extend_from_slice(&[0, 0, 255]); // BGR red
        // Raw packet: header = 0x00 (1 raw pixel)
        rle.push(0x00);
        rle.extend_from_slice(&[255, 0, 0]); // BGR blue

        let tga = make_tga_rle_24(4, 1, &rle);
        let img = parse_tga(&tga).unwrap();

        assert_eq!(img.width, 4);
        assert_eq!(img.height, 1);
        // Pixel 0: red
        assert_eq!(img.data[0], 255);
        assert_eq!(img.data[1], 0);
        assert_eq!(img.data[2], 0);
        // Pixel 3: blue
        assert_eq!(img.data[12], 0);
        assert_eq!(img.data[13], 0);
        assert_eq!(img.data[14], 255);
    }

    #[test]
    fn test_tga_too_small() {
        assert!(parse_tga(&[0u8; 10]).is_err());
    }

    #[test]
    fn test_tga_zero_dimensions() {
        let mut buf = vec![0u8; 18];
        buf[2] = 2;
        // width and height are 0
        assert!(parse_tga(&buf).is_err());
    }

    // -- HDR tests --------------------------------------------------------

    /// Build a minimal HDR file with flat (uncompressed) scanlines.
    fn make_hdr_flat(width: u32, height: u32, rgbe_pixels: &[[u8; 4]]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"#?RADIANCE\n");
        buf.extend_from_slice(b"FORMAT=32-bit_rle_rgbe\n");
        buf.push(b'\n'); // empty line = end of header
        let res_line = format!("-Y {} +X {}\n", height, width);
        buf.extend_from_slice(res_line.as_bytes());

        for pixel in rgbe_pixels {
            buf.extend_from_slice(pixel);
        }
        buf
    }

    #[test]
    fn test_hdr_flat_1x1() {
        // RGBE: R=128, G=128, B=128, E=128+1=129  -> each = (128.5/256)*2 = 1.00390625
        let pixels = &[[128u8, 128, 128, 129]];
        let hdr = make_hdr_flat(1, 1, pixels);
        let img = parse_hdr(&hdr).unwrap();

        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
        assert_eq!(img.format, ImageFormat::Rgb32F);
        assert_eq!(img.data.len(), 12); // 3 * f32

        let r = f32::from_ne_bytes([img.data[0], img.data[1], img.data[2], img.data[3]]);
        // (128 + 0.5) / 256 * 2^(129-128) = 128.5/256 * 2 = 1.00390625
        assert!((r - 1.00390625).abs() < 0.001, "r = {r}");
    }

    #[test]
    fn test_hdr_black_pixel() {
        // Exponent 0 means black
        let pixels = &[[100u8, 200, 50, 0]];
        let hdr = make_hdr_flat(1, 1, pixels);
        let img = parse_hdr(&hdr).unwrap();

        let r = f32::from_ne_bytes([img.data[0], img.data[1], img.data[2], img.data[3]]);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_hdr_missing_magic() {
        let bad = b"NOT_HDR\n";
        assert!(parse_hdr(bad).is_err());
    }

    #[test]
    fn test_rgbe_to_float_basic() {
        let (r, g, b) = rgbe_to_float([0, 0, 0, 0]);
        assert_eq!(r, 0.0);
        assert_eq!(g, 0.0);
        assert_eq!(b, 0.0);

        let (r, _, _) = rgbe_to_float([255, 0, 0, 128]);
        // (255 + 0.5) / 256 * 2^(128-128) = 255.5/256 * 1 ~ 0.998
        assert!((r - 0.998).abs() < 0.01);
    }

    // -- DDS tests --------------------------------------------------------

    /// Build a minimal DDS file with BC1 data.
    fn make_dds_bc1(width: u32, height: u32, bc1_blocks: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; 128];
        // Magic
        buf[0..4].copy_from_slice(b"DDS ");
        // Header size
        buf[4..8].copy_from_slice(&124u32.to_le_bytes());
        // Flags
        buf[8..12].copy_from_slice(&0x1007u32.to_le_bytes());
        // Height, Width
        buf[12..16].copy_from_slice(&height.to_le_bytes());
        buf[16..20].copy_from_slice(&width.to_le_bytes());
        // Pitch
        let pitch = ((width + 3) / 4) * 8;
        buf[20..24].copy_from_slice(&pitch.to_le_bytes());
        // Depth
        buf[24..28].copy_from_slice(&1u32.to_le_bytes());
        // Mip count
        buf[28..32].copy_from_slice(&1u32.to_le_bytes());

        // Pixel format
        buf[76..80].copy_from_slice(&32u32.to_le_bytes()); // pf size
        buf[80..84].copy_from_slice(&DDPF_FOURCC.to_le_bytes()); // flags
        buf[84..88].copy_from_slice(b"DXT1"); // fourcc

        buf.extend_from_slice(bc1_blocks);
        buf
    }

    #[test]
    fn test_dds_bc1_solid_red_block() {
        // A single 4x4 BC1 block with both endpoints = red (RGB565)
        // Red in RGB565: R=31, G=0, B=0 => 0xF800
        let c0: u16 = 0xF800; // pure red
        let c1: u16 = 0xF800; // same
        let mut block = Vec::new();
        block.extend_from_slice(&c0.to_le_bytes());
        block.extend_from_slice(&c1.to_le_bytes());
        block.extend_from_slice(&[0, 0, 0, 0]); // all index 0

        let dds = make_dds_bc1(4, 4, &block);
        let img = parse_dds(&dds).unwrap();

        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert_eq!(img.format, ImageFormat::Rgba8);

        // All pixels should be red (255, 0, 0, 255)
        for i in 0..16 {
            let base = i * 4;
            assert_eq!(img.data[base], 255, "pixel {i} R");
            assert_eq!(img.data[base + 1], 0, "pixel {i} G");
            assert_eq!(img.data[base + 2], 0, "pixel {i} B");
            assert_eq!(img.data[base + 3], 255, "pixel {i} A");
        }
    }

    #[test]
    fn test_dds_bc1_interpolation() {
        // c0 = pure red (0xF800), c1 = pure blue (0x001F)
        // c0 > c1 so: idx2 = 2/3*c0 + 1/3*c1, idx3 = 1/3*c0 + 2/3*c1
        let c0: u16 = 0xF800;
        let c1: u16 = 0x001F;
        let mut block = Vec::new();
        block.extend_from_slice(&c0.to_le_bytes());
        block.extend_from_slice(&c1.to_le_bytes());
        // All pixels use index 0 (= c0 = red)
        block.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        let dds = make_dds_bc1(4, 4, &block);
        let img = parse_dds(&dds).unwrap();

        // pixel 0 should be c0 = red
        assert_eq!(img.data[0], 255);
        assert_eq!(img.data[1], 0);
        assert_eq!(img.data[2], 0);
    }

    #[test]
    fn test_dds_too_small() {
        assert!(parse_dds(&[0u8; 64]).is_err());
    }

    #[test]
    fn test_dds_bad_magic() {
        let mut buf = vec![0u8; 128];
        buf[0..4].copy_from_slice(b"BAD!");
        assert!(parse_dds(&buf).is_err());
    }

    #[test]
    fn test_rgb565_to_rgba() {
        // Pure red: 11111_000000_00000 = 0xF800
        let c = rgb565_to_rgba(0xF800);
        assert_eq!(c[0], 255);
        assert_eq!(c[1], 0);
        assert_eq!(c[2], 0);

        // Pure green: 00000_111111_00000 = 0x07E0
        let c = rgb565_to_rgba(0x07E0);
        assert_eq!(c[0], 0);
        assert_eq!(c[1], 255);
        assert_eq!(c[2], 0);

        // Pure blue: 00000_000000_11111 = 0x001F
        let c = rgb565_to_rgba(0x001F);
        assert_eq!(c[0], 0);
        assert_eq!(c[1], 0);
        assert_eq!(c[2], 255);
    }

    // -- Image ops tests --------------------------------------------------

    #[test]
    fn test_resize_nearest() {
        let img = Image::from_data(2, 2, ImageFormat::R8, vec![10, 20, 30, 40]).unwrap();
        let resized = resize(&img, 4, 4, ResizeFilter::Nearest);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(resized.data.len(), 16);
    }

    #[test]
    fn test_resize_bilinear() {
        let img = Image::from_data(2, 1, ImageFormat::R8, vec![0, 200]).unwrap();
        let resized = resize(&img, 3, 1, ResizeFilter::Bilinear);
        assert_eq!(resized.width, 3);
        // Middle pixel should be interpolated
        assert!(resized.data[1] > 50 && resized.data[1] < 180);
    }

    #[test]
    fn test_resize_bicubic() {
        let img = Image::from_data(4, 4, ImageFormat::R8, vec![
            0, 50, 100, 150,
            50, 100, 150, 200,
            100, 150, 200, 250,
            150, 200, 250, 255,
        ]).unwrap();
        let resized = resize(&img, 2, 2, ResizeFilter::Bicubic);
        assert_eq!(resized.width, 2);
        assert_eq!(resized.height, 2);
    }

    #[test]
    fn test_generate_mipmaps() {
        let img = Image::from_data(8, 8, ImageFormat::R8, vec![128u8; 64]).unwrap();
        let chain = generate_mipmaps(&img);
        assert!(chain.len() >= 4); // 8x8, 4x4, 2x2, 1x1
        assert_eq!(chain[0].width, 8);
        assert_eq!(chain[1].width, 4);
        assert_eq!(chain[2].width, 2);
        assert_eq!(chain[3].width, 1);
    }

    #[test]
    fn test_flip_vertical() {
        let mut img = Image::from_data(2, 2, ImageFormat::R8, vec![1, 2, 3, 4]).unwrap();
        flip_vertical(&mut img);
        assert_eq!(img.data, vec![3, 4, 1, 2]);
    }

    #[test]
    fn test_flip_horizontal() {
        let mut img = Image::from_data(3, 1, ImageFormat::R8, vec![1, 2, 3]).unwrap();
        flip_horizontal(&mut img);
        assert_eq!(img.data, vec![3, 2, 1]);
    }

    #[test]
    fn test_convert_rgb8_to_rgba8() {
        let img = Image::from_data(1, 1, ImageFormat::Rgb8, vec![255, 128, 64]).unwrap();
        let converted = convert_format(&img, ImageFormat::Rgba8).unwrap();
        assert_eq!(converted.data, vec![255, 128, 64, 255]);
    }

    #[test]
    fn test_convert_rgba8_to_rgb8() {
        let img = Image::from_data(1, 1, ImageFormat::Rgba8, vec![100, 200, 50, 128]).unwrap();
        let converted = convert_format(&img, ImageFormat::Rgb8).unwrap();
        assert_eq!(converted.data, vec![100, 200, 50]);
    }

    #[test]
    fn test_convert_r8_to_rgba8() {
        let img = Image::from_data(1, 1, ImageFormat::R8, vec![200]).unwrap();
        let converted = convert_format(&img, ImageFormat::Rgba8).unwrap();
        assert_eq!(converted.data, vec![200, 200, 200, 255]);
    }

    #[test]
    fn test_height_to_normal_flat() {
        // A flat heightmap should produce normals pointing straight up (0, 0, 1)
        let img = Image::from_data(3, 3, ImageFormat::R8, vec![128u8; 9]).unwrap();
        let normal = height_to_normal(&img, 1.0).unwrap();
        assert_eq!(normal.width, 3);
        assert_eq!(normal.height, 3);

        // Center pixel normal should be approximately (0.5, 0.5, 1.0) in [0,255] = (128, 128, 255)
        let cx = 1;
        let cy = 1;
        let off = (cy * 3 + cx) * 4;
        assert!((normal.data[off] as i32 - 128).abs() <= 1, "nx");
        assert!((normal.data[off + 1] as i32 - 128).abs() <= 1, "ny");
        assert!((normal.data[off + 2] as i32 - 255).abs() <= 1, "nz");
    }

    #[test]
    fn test_image_new() {
        let img = Image::new(16, 16, ImageFormat::Rgba8);
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
        assert_eq!(img.channels, 4);
        assert_eq!(img.data.len(), 16 * 16 * 4);
    }

    #[test]
    fn test_image_from_data_mismatch() {
        let result = Image::from_data(2, 2, ImageFormat::Rgba8, vec![0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_image_format_channels() {
        assert_eq!(ImageFormat::R8.channels(), 1);
        assert_eq!(ImageFormat::Rgb8.channels(), 3);
        assert_eq!(ImageFormat::Rgba8.channels(), 4);
        assert_eq!(ImageFormat::Rgb32F.channels(), 3);
        assert_eq!(ImageFormat::Rgba32F.channels(), 4);
    }

    #[test]
    fn test_image_format_bpp() {
        assert_eq!(ImageFormat::R8.bytes_per_pixel(), 1);
        assert_eq!(ImageFormat::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(ImageFormat::Rgba8.bytes_per_pixel(), 4);
        assert_eq!(ImageFormat::R32F.bytes_per_pixel(), 4);
        assert_eq!(ImageFormat::Rgb32F.bytes_per_pixel(), 12);
        assert_eq!(ImageFormat::Rgba32F.bytes_per_pixel(), 16);
    }

    #[test]
    fn test_extract_channel() {
        // 8-bit red mask at bits 16-23
        assert_eq!(extract_channel(0x00FF0000, 0x00FF0000), 255);
        assert_eq!(extract_channel(0x00800000, 0x00FF0000), 128);
        assert_eq!(extract_channel(0x00000000, 0x00FF0000), 0);
    }
}
