// engine/render/src/screen_capture.rs
//
// Screenshot and video frame capture system for the Genovo engine.
//
// Provides utilities for capturing rendered frames:
//
// - **Capture render target to image** — Read back the final colour buffer
//   to CPU memory.
// - **PNG encoding** — Simple uncompressed PNG writer for screenshots.
// - **Timestamp-based filenames** — Automatic filename generation with date
//   and time.
// - **Capture region** — Full screen or a sub-rectangle.
// - **Video frame capture** — Save frame sequences for offline video encoding.
// - **Async GPU readback** — Non-blocking capture using staging buffers.
//
// # Pipeline integration
//
// Screen capture hooks into the end of the render pipeline, after all post-
// processing. The readback is asynchronous: the GPU copies the render target
// into a staging buffer, and the CPU reads the data one or more frames later.

use std::io::Write;

// ---------------------------------------------------------------------------
// Capture configuration
// ---------------------------------------------------------------------------

/// Image format for saving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureFormat {
    /// PNG (lossless, using a simple uncompressed encoder).
    Png,
    /// TGA (uncompressed Targa).
    Tga,
    /// BMP (uncompressed bitmap).
    Bmp,
    /// Raw RGBA bytes (no header).
    Raw,
    /// PPM (Portable PixMap, text-based).
    Ppm,
}

/// Capture region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRegion {
    /// Top-left X.
    pub x: u32,
    /// Top-left Y.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

impl CaptureRegion {
    /// Full-screen region.
    pub fn full(width: u32, height: u32) -> Self {
        Self { x: 0, y: 0, width, height }
    }

    /// Custom sub-region.
    pub fn rect(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }

    /// Number of pixels in the region.
    pub fn pixel_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Clamp to screen bounds.
    pub fn clamp_to(&self, screen_w: u32, screen_h: u32) -> Self {
        let x = self.x.min(screen_w);
        let y = self.y.min(screen_h);
        let w = self.width.min(screen_w - x);
        let h = self.height.min(screen_h - y);
        Self { x, y, width: w, height: h }
    }
}

/// Screenshot request configuration.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// Output directory (relative or absolute).
    pub output_dir: String,
    /// Filename prefix (e.g. "screenshot").
    pub filename_prefix: String,
    /// File format.
    pub format: CaptureFormat,
    /// Capture region (None = full screen).
    pub region: Option<CaptureRegion>,
    /// Include alpha channel (if format supports it).
    pub include_alpha: bool,
    /// Apply gamma correction before saving (sRGB).
    pub apply_gamma: bool,
    /// Supersampling factor (1 = native, 2 = 2x, etc.).
    pub supersample: u32,
    /// JPEG quality (only for JPEG format, not currently implemented).
    pub quality: u32,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            output_dir: "screenshots".to_string(),
            filename_prefix: "screenshot".to_string(),
            format: CaptureFormat::Png,
            region: None,
            include_alpha: false,
            apply_gamma: true,
            supersample: 1,
            quality: 90,
        }
    }
}

impl CaptureConfig {
    /// Generate a filename with timestamp.
    pub fn generate_filename(&self, frame: u64) -> String {
        let ext = match self.format {
            CaptureFormat::Png => "png",
            CaptureFormat::Tga => "tga",
            CaptureFormat::Bmp => "bmp",
            CaptureFormat::Raw => "raw",
            CaptureFormat::Ppm => "ppm",
        };
        format!(
            "{}/{}_{:06}.{}",
            self.output_dir, self.filename_prefix, frame, ext
        )
    }

    /// Generate a filename with a custom timestamp string.
    pub fn generate_filename_with_timestamp(&self, timestamp: &str) -> String {
        let ext = match self.format {
            CaptureFormat::Png => "png",
            CaptureFormat::Tga => "tga",
            CaptureFormat::Bmp => "bmp",
            CaptureFormat::Raw => "raw",
            CaptureFormat::Ppm => "ppm",
        };
        format!(
            "{}/{}_{}.{}",
            self.output_dir, self.filename_prefix, timestamp, ext
        )
    }
}

// ---------------------------------------------------------------------------
// Captured frame data
// ---------------------------------------------------------------------------

/// A captured frame ready for encoding/saving.
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    /// Pixel data (RGBA, u8 per channel, row-major, top-to-bottom).
    pub pixels: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Channels per pixel (3 = RGB, 4 = RGBA).
    pub channels: u32,
    /// Frame number.
    pub frame: u64,
    /// Whether gamma correction has been applied.
    pub gamma_corrected: bool,
}

impl CapturedFrame {
    /// Create a new captured frame from raw RGBA data.
    pub fn from_rgba(pixels: Vec<u8>, width: u32, height: u32, frame: u64) -> Self {
        Self {
            pixels,
            width,
            height,
            channels: 4,
            frame,
            gamma_corrected: false,
        }
    }

    /// Create from float RGB data.
    pub fn from_float_rgb(data: &[[f32; 3]], width: u32, height: u32, frame: u64, apply_gamma: bool) -> Self {
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        for (i, pixel) in data.iter().enumerate() {
            let idx = i * 4;
            if idx + 3 < pixels.len() {
                let (r, g, b) = if apply_gamma {
                    (
                        linear_to_srgb(pixel[0]),
                        linear_to_srgb(pixel[1]),
                        linear_to_srgb(pixel[2]),
                    )
                } else {
                    (pixel[0], pixel[1], pixel[2])
                };
                pixels[idx] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[idx + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[idx + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[idx + 3] = 255;
            }
        }

        Self {
            pixels,
            width,
            height,
            channels: 4,
            frame,
            gamma_corrected: apply_gamma,
        }
    }

    /// Create from float RGBA data.
    pub fn from_float_rgba(data: &[[f32; 4]], width: u32, height: u32, frame: u64, apply_gamma: bool) -> Self {
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        for (i, pixel) in data.iter().enumerate() {
            let idx = i * 4;
            if idx + 3 < pixels.len() {
                let (r, g, b) = if apply_gamma {
                    (
                        linear_to_srgb(pixel[0]),
                        linear_to_srgb(pixel[1]),
                        linear_to_srgb(pixel[2]),
                    )
                } else {
                    (pixel[0], pixel[1], pixel[2])
                };
                pixels[idx] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[idx + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[idx + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[idx + 3] = (pixel[3].clamp(0.0, 1.0) * 255.0) as u8;
            }
        }

        Self {
            pixels,
            width,
            height,
            channels: 4,
            frame,
            gamma_corrected: apply_gamma,
        }
    }

    /// Extract a sub-region.
    pub fn extract_region(&self, region: &CaptureRegion) -> CapturedFrame {
        let region = region.clamp_to(self.width, self.height);
        let mut pixels = vec![0u8; (region.width * region.height * self.channels) as usize];
        let src_stride = (self.width * self.channels) as usize;
        let dst_stride = (region.width * self.channels) as usize;

        for y in 0..region.height {
            let src_offset = ((region.y + y) * self.width + region.x) as usize * self.channels as usize;
            let dst_offset = y as usize * dst_stride;
            let src_end = src_offset + dst_stride;
            if src_end <= self.pixels.len() && dst_offset + dst_stride <= pixels.len() {
                pixels[dst_offset..dst_offset + dst_stride]
                    .copy_from_slice(&self.pixels[src_offset..src_end]);
            }
        }

        CapturedFrame {
            pixels,
            width: region.width,
            height: region.height,
            channels: self.channels,
            frame: self.frame,
            gamma_corrected: self.gamma_corrected,
        }
    }

    /// Flip vertically (some APIs return bottom-to-top).
    pub fn flip_vertical(&mut self) {
        let stride = (self.width * self.channels) as usize;
        let half = self.height / 2;
        for y in 0..half {
            let top = y as usize * stride;
            let bot = (self.height - 1 - y) as usize * stride;
            for i in 0..stride {
                self.pixels.swap(top + i, bot + i);
            }
        }
    }

    /// Get pixel data size in bytes.
    pub fn data_size(&self) -> usize {
        self.pixels.len()
    }

    /// Get a pixel at (x, y).
    pub fn pixel_at(&self, x: u32, y: u32) -> Option<&[u8]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) as usize * self.channels as usize;
        let end = idx + self.channels as usize;
        if end <= self.pixels.len() {
            Some(&self.pixels[idx..end])
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// PNG encoding (simple uncompressed)
// ---------------------------------------------------------------------------

/// Encode a captured frame to PNG format.
///
/// This is a simple, non-compressed PNG encoder. It produces valid PNG files
/// but without deflate compression, so file sizes are large.
pub fn encode_png(frame: &CapturedFrame) -> Vec<u8> {
    let mut output = Vec::new();

    // PNG signature.
    output.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    // IHDR chunk.
    let channels = if frame.channels >= 4 { 4u8 } else { 3u8 };
    let color_type: u8 = if channels == 4 { 6 } else { 2 }; // 6=RGBA, 2=RGB
    let bit_depth: u8 = 8;

    let mut ihdr_data = Vec::new();
    ihdr_data.extend_from_slice(&frame.width.to_be_bytes());
    ihdr_data.extend_from_slice(&frame.height.to_be_bytes());
    ihdr_data.push(bit_depth);
    ihdr_data.push(color_type);
    ihdr_data.push(0); // Compression method.
    ihdr_data.push(0); // Filter method.
    ihdr_data.push(0); // Interlace method.

    write_png_chunk(&mut output, b"IHDR", &ihdr_data);

    // IDAT chunk (uncompressed deflate).
    let row_size = (frame.width * channels as u32) as usize;
    let mut raw_data = Vec::new();

    // Build the raw pixel data with filter bytes.
    for y in 0..frame.height {
        raw_data.push(0); // Filter type: None.
        let row_start = y as usize * (frame.width * frame.channels) as usize;
        for x in 0..frame.width as usize {
            let src_idx = row_start + x * frame.channels as usize;
            for c in 0..channels as usize {
                if src_idx + c < frame.pixels.len() {
                    raw_data.push(frame.pixels[src_idx + c]);
                } else {
                    raw_data.push(0);
                }
            }
        }
    }

    // Wrap in uncompressed deflate blocks.
    let deflate_data = wrap_uncompressed_deflate(&raw_data);

    // Zlib wrapper: CMF, FLG, data, ADLER32.
    let mut zlib_data = Vec::new();
    zlib_data.push(0x78); // CMF (deflate, 32K window).
    zlib_data.push(0x01); // FLG (no preset dict, check bits).
    zlib_data.extend_from_slice(&deflate_data);

    let adler = adler32(&raw_data);
    zlib_data.extend_from_slice(&adler.to_be_bytes());

    write_png_chunk(&mut output, b"IDAT", &zlib_data);

    // IEND chunk.
    write_png_chunk(&mut output, b"IEND", &[]);

    output
}

/// Write a PNG chunk.
fn write_png_chunk(output: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    let length = data.len() as u32;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(chunk_type);
    output.extend_from_slice(data);

    // CRC32 over type + data.
    let mut crc_data = Vec::with_capacity(4 + data.len());
    crc_data.extend_from_slice(chunk_type);
    crc_data.extend_from_slice(data);
    let crc = crc32(&crc_data);
    output.extend_from_slice(&crc.to_be_bytes());
}

/// Wrap data in uncompressed deflate blocks (max 65535 bytes per block).
fn wrap_uncompressed_deflate(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let max_block = 65535usize;
    let mut offset = 0;

    while offset < data.len() {
        let remaining = data.len() - offset;
        let block_size = remaining.min(max_block);
        let is_last = offset + block_size >= data.len();

        result.push(if is_last { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00
        let len = block_size as u16;
        let nlen = !len;
        result.extend_from_slice(&len.to_le_bytes());
        result.extend_from_slice(&nlen.to_le_bytes());
        result.extend_from_slice(&data[offset..offset + block_size]);

        offset += block_size;
    }

    result
}

/// Compute CRC32 (PNG uses ISO 3309 / ITU-T V.42).
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

/// Compute Adler-32 checksum.
fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;

    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }

    (b << 16) | a
}

// ---------------------------------------------------------------------------
// TGA encoding
// ---------------------------------------------------------------------------

/// Encode a captured frame to uncompressed TGA format.
pub fn encode_tga(frame: &CapturedFrame) -> Vec<u8> {
    let channels = frame.channels.min(4) as u8;
    let bpp = channels * 8;
    let image_type: u8 = 2; // Uncompressed true-colour.

    let mut output = Vec::new();

    // TGA header (18 bytes).
    output.push(0); // ID length.
    output.push(0); // Colour map type.
    output.push(image_type);
    output.extend_from_slice(&[0u8; 5]); // Colour map spec.
    output.extend_from_slice(&0u16.to_le_bytes()); // X origin.
    output.extend_from_slice(&0u16.to_le_bytes()); // Y origin.
    output.extend_from_slice(&(frame.width as u16).to_le_bytes());
    output.extend_from_slice(&(frame.height as u16).to_le_bytes());
    output.push(bpp);
    output.push(if channels == 4 { 0x28 } else { 0x20 }); // Image descriptor (top-left origin + alpha bits).

    // Pixel data (TGA uses BGRA order).
    for y in 0..frame.height {
        for x in 0..frame.width {
            let idx = (y * frame.width + x) as usize * frame.channels as usize;
            let r = if idx < frame.pixels.len() { frame.pixels[idx] } else { 0 };
            let g = if idx + 1 < frame.pixels.len() { frame.pixels[idx + 1] } else { 0 };
            let b = if idx + 2 < frame.pixels.len() { frame.pixels[idx + 2] } else { 0 };

            output.push(b);
            output.push(g);
            output.push(r);

            if channels == 4 {
                let a = if idx + 3 < frame.pixels.len() { frame.pixels[idx + 3] } else { 255 };
                output.push(a);
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// BMP encoding
// ---------------------------------------------------------------------------

/// Encode a captured frame to BMP format.
pub fn encode_bmp(frame: &CapturedFrame) -> Vec<u8> {
    let channels = 3u32; // BMP uses 24-bit.
    let row_stride = ((frame.width * channels + 3) & !3) as usize; // Rows padded to 4 bytes.
    let data_size = row_stride * frame.height as usize;
    let file_size = 54 + data_size;

    let mut output = Vec::with_capacity(file_size);

    // BMP file header (14 bytes).
    output.push(b'B');
    output.push(b'M');
    output.extend_from_slice(&(file_size as u32).to_le_bytes());
    output.extend_from_slice(&0u16.to_le_bytes()); // Reserved.
    output.extend_from_slice(&0u16.to_le_bytes()); // Reserved.
    output.extend_from_slice(&54u32.to_le_bytes()); // Pixel data offset.

    // DIB header (BITMAPINFOHEADER, 40 bytes).
    output.extend_from_slice(&40u32.to_le_bytes()); // Header size.
    output.extend_from_slice(&(frame.width as i32).to_le_bytes());
    output.extend_from_slice(&(-(frame.height as i32)).to_le_bytes()); // Top-down.
    output.extend_from_slice(&1u16.to_le_bytes()); // Planes.
    output.extend_from_slice(&24u16.to_le_bytes()); // Bits per pixel.
    output.extend_from_slice(&0u32.to_le_bytes()); // Compression (none).
    output.extend_from_slice(&(data_size as u32).to_le_bytes());
    output.extend_from_slice(&2835u32.to_le_bytes()); // X pixels/meter.
    output.extend_from_slice(&2835u32.to_le_bytes()); // Y pixels/meter.
    output.extend_from_slice(&0u32.to_le_bytes()); // Colours used.
    output.extend_from_slice(&0u32.to_le_bytes()); // Important colours.

    // Pixel data (BGR, bottom-to-top with padding).
    for y in 0..frame.height {
        for x in 0..frame.width {
            let idx = (y * frame.width + x) as usize * frame.channels as usize;
            let r = if idx < frame.pixels.len() { frame.pixels[idx] } else { 0 };
            let g = if idx + 1 < frame.pixels.len() { frame.pixels[idx + 1] } else { 0 };
            let b = if idx + 2 < frame.pixels.len() { frame.pixels[idx + 2] } else { 0 };

            output.push(b);
            output.push(g);
            output.push(r);
        }

        // Padding to 4-byte boundary.
        let padding = row_stride - (frame.width * channels) as usize;
        for _ in 0..padding {
            output.push(0);
        }
    }

    output
}

// ---------------------------------------------------------------------------
// PPM encoding
// ---------------------------------------------------------------------------

/// Encode a captured frame to PPM format (P6, binary).
pub fn encode_ppm(frame: &CapturedFrame) -> Vec<u8> {
    let header = format!("P6\n{} {}\n255\n", frame.width, frame.height);
    let mut output = Vec::with_capacity(header.len() + (frame.width * frame.height * 3) as usize);
    output.extend_from_slice(header.as_bytes());

    for y in 0..frame.height {
        for x in 0..frame.width {
            let idx = (y * frame.width + x) as usize * frame.channels as usize;
            let r = if idx < frame.pixels.len() { frame.pixels[idx] } else { 0 };
            let g = if idx + 1 < frame.pixels.len() { frame.pixels[idx + 1] } else { 0 };
            let b = if idx + 2 < frame.pixels.len() { frame.pixels[idx + 2] } else { 0 };
            output.push(r);
            output.push(g);
            output.push(b);
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Encode dispatcher
// ---------------------------------------------------------------------------

/// Encode a captured frame to the specified format.
pub fn encode_frame(frame: &CapturedFrame, format: CaptureFormat) -> Vec<u8> {
    match format {
        CaptureFormat::Png => encode_png(frame),
        CaptureFormat::Tga => encode_tga(frame),
        CaptureFormat::Bmp => encode_bmp(frame),
        CaptureFormat::Ppm => encode_ppm(frame),
        CaptureFormat::Raw => frame.pixels.clone(),
    }
}

// ---------------------------------------------------------------------------
// Video frame capture
// ---------------------------------------------------------------------------

/// Video frame capture session.
#[derive(Debug)]
pub struct VideoCaptureSession {
    /// Base output directory.
    pub output_dir: String,
    /// Frame filename prefix.
    pub prefix: String,
    /// Frame format.
    pub format: CaptureFormat,
    /// Target frame rate.
    pub target_fps: f32,
    /// Capture region (None = full screen).
    pub region: Option<CaptureRegion>,
    /// Number of frames captured so far.
    pub frame_count: u64,
    /// Whether the session is actively recording.
    pub recording: bool,
    /// Maximum number of frames (0 = unlimited).
    pub max_frames: u64,
    /// Apply gamma correction.
    pub apply_gamma: bool,
    /// Accumulated time for frame rate control.
    accumulated_time: f64,
    /// Time between frames.
    frame_interval: f64,
}

impl VideoCaptureSession {
    /// Create a new video capture session.
    pub fn new(output_dir: impl Into<String>, fps: f32) -> Self {
        Self {
            output_dir: output_dir.into(),
            prefix: "frame".to_string(),
            format: CaptureFormat::Png,
            target_fps: fps,
            region: None,
            frame_count: 0,
            recording: false,
            max_frames: 0,
            apply_gamma: true,
            accumulated_time: 0.0,
            frame_interval: 1.0 / fps as f64,
        }
    }

    /// Start recording.
    pub fn start(&mut self) {
        self.recording = true;
        self.frame_count = 0;
        self.accumulated_time = 0.0;
    }

    /// Stop recording.
    pub fn stop(&mut self) {
        self.recording = false;
    }

    /// Check if a frame should be captured this tick.
    ///
    /// # Arguments
    /// * `dt` — Delta time in seconds.
    pub fn should_capture(&mut self, dt: f64) -> bool {
        if !self.recording {
            return false;
        }

        if self.max_frames > 0 && self.frame_count >= self.max_frames {
            self.recording = false;
            return false;
        }

        self.accumulated_time += dt;
        if self.accumulated_time >= self.frame_interval {
            self.accumulated_time -= self.frame_interval;
            true
        } else {
            false
        }
    }

    /// Generate the filename for the next frame.
    pub fn next_filename(&mut self) -> String {
        let name = format!(
            "{}/{}_{:06}.{}",
            self.output_dir,
            self.prefix,
            self.frame_count,
            match self.format {
                CaptureFormat::Png => "png",
                CaptureFormat::Tga => "tga",
                CaptureFormat::Bmp => "bmp",
                CaptureFormat::Raw => "raw",
                CaptureFormat::Ppm => "ppm",
            }
        );
        self.frame_count += 1;
        name
    }

    /// Whether the session has reached its frame limit.
    pub fn is_complete(&self) -> bool {
        self.max_frames > 0 && self.frame_count >= self.max_frames
    }

    /// Estimated video duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        if self.target_fps > 0.0 {
            self.frame_count as f64 / self.target_fps as f64
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Async capture manager
// ---------------------------------------------------------------------------

/// Status of an async capture request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureStatus {
    /// Waiting for GPU readback.
    Pending,
    /// Data is ready.
    Ready,
    /// Capture failed.
    Failed,
    /// Data has been saved/consumed.
    Consumed,
}

/// An async capture request.
#[derive(Debug)]
pub struct CaptureRequest {
    /// Unique request ID.
    pub id: u64,
    /// Status.
    pub status: CaptureStatus,
    /// Filename to save to.
    pub filename: String,
    /// Format.
    pub format: CaptureFormat,
    /// Captured frame data (filled when Ready).
    pub frame: Option<CapturedFrame>,
    /// Region.
    pub region: Option<CaptureRegion>,
    /// Frame number when requested.
    pub request_frame: u64,
}

/// Manages async screen captures.
#[derive(Debug)]
pub struct CaptureManager {
    /// Pending capture requests.
    requests: Vec<CaptureRequest>,
    /// Next request ID.
    next_id: u64,
    /// Default configuration.
    pub config: CaptureConfig,
    /// Video capture session (if active).
    pub video_session: Option<VideoCaptureSession>,
    /// Current frame number.
    pub frame: u64,
}

impl CaptureManager {
    /// Create a new capture manager.
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
            next_id: 1,
            config: CaptureConfig::default(),
            video_session: None,
            frame: 0,
        }
    }

    /// Request a screenshot.
    pub fn capture_screenshot(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let filename = self.config.generate_filename(self.frame);

        self.requests.push(CaptureRequest {
            id,
            status: CaptureStatus::Pending,
            filename,
            format: self.config.format,
            frame: None,
            region: self.config.region,
            request_frame: self.frame,
        });

        id
    }

    /// Check capture status.
    pub fn status(&self, id: u64) -> CaptureStatus {
        self.requests.iter()
            .find(|r| r.id == id)
            .map(|r| r.status)
            .unwrap_or(CaptureStatus::Failed)
    }

    /// Complete a pending capture with frame data.
    pub fn complete_capture(&mut self, id: u64, frame: CapturedFrame) {
        if let Some(req) = self.requests.iter_mut().find(|r| r.id == id) {
            req.frame = Some(frame);
            req.status = CaptureStatus::Ready;
        }
    }

    /// Get all ready captures for saving.
    pub fn take_ready(&mut self) -> Vec<(String, Vec<u8>)> {
        let mut results = Vec::new();

        for req in &mut self.requests {
            if req.status == CaptureStatus::Ready {
                if let Some(frame) = req.frame.take() {
                    let mut final_frame = frame;

                    // Extract region if needed.
                    if let Some(region) = &req.region {
                        final_frame = final_frame.extract_region(region);
                    }

                    let encoded = encode_frame(&final_frame, req.format);
                    results.push((req.filename.clone(), encoded));
                    req.status = CaptureStatus::Consumed;
                }
            }
        }

        // Clean up consumed requests.
        self.requests.retain(|r| r.status != CaptureStatus::Consumed);

        results
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }
}

impl Default for CaptureManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// sRGB gamma conversion.
#[inline]
fn linear_to_srgb(l: f32) -> f32 {
    if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_region() {
        let region = CaptureRegion::full(1920, 1080);
        assert_eq!(region.pixel_count(), 1920 * 1080);

        let clamped = CaptureRegion::rect(1900, 1060, 100, 100).clamp_to(1920, 1080);
        assert_eq!(clamped.width, 20);
        assert_eq!(clamped.height, 20);
    }

    #[test]
    fn test_png_encode() {
        let pixels = vec![255u8; 4 * 4 * 4]; // 4x4 white RGBA.
        let frame = CapturedFrame::from_rgba(pixels, 4, 4, 0);
        let png = encode_png(&frame);

        // Check PNG signature.
        assert_eq!(&png[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        assert!(png.len() > 50); // Has data.
    }

    #[test]
    fn test_tga_encode() {
        let pixels = vec![128u8; 4 * 4 * 4];
        let frame = CapturedFrame::from_rgba(pixels, 4, 4, 0);
        let tga = encode_tga(&frame);
        assert_eq!(tga[2], 2); // Uncompressed true-colour.
    }

    #[test]
    fn test_bmp_encode() {
        let pixels = vec![128u8; 4 * 4 * 4];
        let frame = CapturedFrame::from_rgba(pixels, 4, 4, 0);
        let bmp = encode_bmp(&frame);
        assert_eq!(bmp[0], b'B');
        assert_eq!(bmp[1], b'M');
    }

    #[test]
    fn test_video_session() {
        let mut session = VideoCaptureSession::new("output", 30.0);
        session.start();

        assert!(session.should_capture(1.0 / 30.0));
        let name = session.next_filename();
        assert!(name.contains("000000"));
    }

    #[test]
    fn test_crc32() {
        let data = b"IEND";
        let crc = crc32(data);
        assert_eq!(crc, 0xAE426082);
    }

    #[test]
    fn test_adler32() {
        let data = b"Wikipedia";
        let adler = adler32(data);
        assert_eq!(adler, 0x11E60398);
    }

    #[test]
    fn test_float_frame() {
        let data = vec![[0.5f32, 0.5, 0.5]; 4];
        let frame = CapturedFrame::from_float_rgb(&data, 2, 2, 0, false);
        assert_eq!(frame.channels, 4);
        assert_eq!(frame.pixels[0], 128); // 0.5 * 255 ≈ 128.
    }
}
