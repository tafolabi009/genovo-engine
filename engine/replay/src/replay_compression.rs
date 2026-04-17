//! # Replay Data Compression
//!
//! Provides specialised compression algorithms for replay data: delta encoding
//! between frames, run-length encoding for static entities, quantized float
//! encoding, input compression (only store changes), variable-length integer
//! encoding, and replay file size analysis.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from replay compression operations.
#[derive(Debug, Clone)]
pub enum CompressionError {
    /// Input buffer too short.
    BufferTooShort,
    /// Invalid magic / header.
    InvalidHeader,
    /// Decompression produced unexpected output size.
    SizeMismatch { expected: usize, actual: usize },
    /// Corrupt data detected.
    CorruptData(String),
    /// Unsupported compression version.
    UnsupportedVersion(u32),
}

impl fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooShort => write!(f, "buffer too short"),
            Self::InvalidHeader => write!(f, "invalid header"),
            Self::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {expected}, got {actual}")
            }
            Self::CorruptData(msg) => write!(f, "corrupt data: {msg}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
        }
    }
}

impl std::error::Error for CompressionError {}

// ---------------------------------------------------------------------------
// Variable-length integer encoding (VarInt)
// ---------------------------------------------------------------------------

/// Encode a `u32` as a variable-length integer (1-5 bytes).
///
/// Uses a continuation bit in each byte (bit 7): if set, more bytes follow.
pub fn encode_varint_u32(value: u32, out: &mut Vec<u8>) {
    let mut v = value;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        } else {
            out.push(byte | 0x80);
        }
    }
}

/// Decode a `u32` from a variable-length integer.
///
/// Returns `(value, bytes_consumed)`.
pub fn decode_varint_u32(data: &[u8]) -> Result<(u32, usize), CompressionError> {
    let mut result: u32 = 0;
    let mut shift = 0;
    for (i, &byte) in data.iter().enumerate() {
        if i >= 5 {
            return Err(CompressionError::CorruptData(
                "varint too long".into(),
            ));
        }
        result |= ((byte & 0x7F) as u32) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
    }
    Err(CompressionError::BufferTooShort)
}

/// Encode a `u64` as a variable-length integer (1-10 bytes).
pub fn encode_varint_u64(value: u64, out: &mut Vec<u8>) {
    let mut v = value;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        } else {
            out.push(byte | 0x80);
        }
    }
}

/// Decode a `u64` from a variable-length integer.
pub fn decode_varint_u64(data: &[u8]) -> Result<(u64, usize), CompressionError> {
    let mut result: u64 = 0;
    let mut shift = 0;
    for (i, &byte) in data.iter().enumerate() {
        if i >= 10 {
            return Err(CompressionError::CorruptData(
                "varint too long".into(),
            ));
        }
        result |= ((byte & 0x7F) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
    }
    Err(CompressionError::BufferTooShort)
}

/// Encode a signed `i32` as a zigzag-encoded varint.
///
/// Zigzag maps negative values to positive (e.g. 0->0, -1->1, 1->2, -2->3)
/// for efficient varint representation.
pub fn encode_varint_i32(value: i32, out: &mut Vec<u8>) {
    let zigzag = ((value << 1) ^ (value >> 31)) as u32;
    encode_varint_u32(zigzag, out);
}

/// Decode a zigzag-encoded `i32` varint.
pub fn decode_varint_i32(data: &[u8]) -> Result<(i32, usize), CompressionError> {
    let (zigzag, consumed) = decode_varint_u32(data)?;
    let value = ((zigzag >> 1) as i32) ^ -((zigzag & 1) as i32);
    Ok((value, consumed))
}

// ---------------------------------------------------------------------------
// Quantized float encoding
// ---------------------------------------------------------------------------

/// Precision levels for float quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationPrecision {
    /// 8-bit quantization (256 levels).
    Low,
    /// 16-bit quantization (65536 levels).
    Medium,
    /// 24-bit quantization.
    High,
    /// No quantization (full 32-bit float).
    Full,
}

/// Quantize a float value to a reduced-precision integer.
///
/// Maps `value` from `[min, max]` to `[0, levels-1]`.
pub fn quantize_float(value: f32, min: f32, max: f32, precision: QuantizationPrecision) -> u32 {
    let range = max - min;
    if range.abs() < f32::EPSILON {
        return 0;
    }
    let levels = match precision {
        QuantizationPrecision::Low => 255,
        QuantizationPrecision::Medium => 65535,
        QuantizationPrecision::High => 16_777_215,
        QuantizationPrecision::Full => return value.to_bits(),
    };
    let normalized = ((value - min) / range).clamp(0.0, 1.0);
    (normalized * levels as f32).round() as u32
}

/// Dequantize an integer back to a float value.
pub fn dequantize_float(
    quantized: u32,
    min: f32,
    max: f32,
    precision: QuantizationPrecision,
) -> f32 {
    let range = max - min;
    match precision {
        QuantizationPrecision::Full => f32::from_bits(quantized),
        _ => {
            let levels = match precision {
                QuantizationPrecision::Low => 255,
                QuantizationPrecision::Medium => 65535,
                QuantizationPrecision::High => 16_777_215,
                QuantizationPrecision::Full => unreachable!(),
            };
            let normalized = quantized as f32 / levels as f32;
            min + normalized * range
        }
    }
}

/// Quantize a 3D position.
pub fn quantize_position(
    x: f32,
    y: f32,
    z: f32,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    precision: QuantizationPrecision,
) -> [u32; 3] {
    [
        quantize_float(x, bounds_min[0], bounds_max[0], precision),
        quantize_float(y, bounds_min[1], bounds_max[1], precision),
        quantize_float(z, bounds_min[2], bounds_max[2], precision),
    ]
}

/// Dequantize a 3D position.
pub fn dequantize_position(
    q: [u32; 3],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    precision: QuantizationPrecision,
) -> [f32; 3] {
    [
        dequantize_float(q[0], bounds_min[0], bounds_max[0], precision),
        dequantize_float(q[1], bounds_min[1], bounds_max[1], precision),
        dequantize_float(q[2], bounds_min[2], bounds_max[2], precision),
    ]
}

/// Quantize a quaternion rotation using the "smallest three" method.
///
/// Stores only three components and the index of the largest component.
/// Each stored component is quantized to the given precision.
pub fn quantize_quaternion(
    x: f32,
    y: f32,
    z: f32,
    w: f32,
    precision: QuantizationPrecision,
) -> (u8, [u32; 3]) {
    let components = [x, y, z, w];
    let mut max_idx = 0;
    let mut max_val = components[0].abs();
    for i in 1..4 {
        if components[i].abs() > max_val {
            max_val = components[i].abs();
            max_idx = i;
        }
    }
    // The sign of the dropped component determines whether we negate
    let sign = if components[max_idx] < 0.0 { -1.0 } else { 1.0 };
    let mut stored = [0u32; 3];
    let mut j = 0;
    for i in 0..4 {
        if i != max_idx {
            // Each component is in [-1/sqrt(2), 1/sqrt(2)] ~= [-0.707, 0.707]
            let val = components[i] * sign;
            stored[j] = quantize_float(val, -0.7072, 0.7072, precision);
            j += 1;
        }
    }
    (max_idx as u8, stored)
}

/// Dequantize a quaternion from smallest-three representation.
pub fn dequantize_quaternion(
    max_idx: u8,
    stored: [u32; 3],
    precision: QuantizationPrecision,
) -> (f32, f32, f32, f32) {
    let mut components = [0.0f32; 4];
    let mut sum_sq = 0.0f32;
    let mut j = 0;
    for i in 0..4 {
        if i != max_idx as usize {
            components[i] = dequantize_float(stored[j], -0.7072, 0.7072, precision);
            sum_sq += components[i] * components[i];
            j += 1;
        }
    }
    components[max_idx as usize] = (1.0 - sum_sq).max(0.0).sqrt();
    (components[0], components[1], components[2], components[3])
}

// ---------------------------------------------------------------------------
// Delta encoding
// ---------------------------------------------------------------------------

/// Delta-encode a sequence of `u32` values.
///
/// Stores the first value verbatim, then each subsequent value as the
/// difference from the previous.
pub fn delta_encode_u32(values: &[u32]) -> Vec<i32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(values.len());
    result.push(values[0] as i32);
    for i in 1..values.len() {
        result.push(values[i] as i32 - values[i - 1] as i32);
    }
    result
}

/// Delta-decode a sequence back to original values.
pub fn delta_decode_u32(deltas: &[i32]) -> Vec<u32> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(deltas.len());
    result.push(deltas[0] as u32);
    for i in 1..deltas.len() {
        let prev = *result.last().unwrap() as i32;
        result.push((prev + deltas[i]) as u32);
    }
    result
}

/// Delta-encode a sequence of f32 values (quantized to integers first).
pub fn delta_encode_f32(
    values: &[f32],
    min: f32,
    max: f32,
    precision: QuantizationPrecision,
) -> Vec<i32> {
    let quantized: Vec<u32> = values
        .iter()
        .map(|&v| quantize_float(v, min, max, precision))
        .collect();
    delta_encode_u32(&quantized)
}

/// Delta-decode f32 values.
pub fn delta_decode_f32(
    deltas: &[i32],
    min: f32,
    max: f32,
    precision: QuantizationPrecision,
) -> Vec<f32> {
    let quantized = delta_decode_u32(deltas);
    quantized
        .iter()
        .map(|&q| dequantize_float(q, min, max, precision))
        .collect()
}

/// Delta-encode entity transforms between two frames.
///
/// Only stores data for entities whose transform has changed.
#[derive(Debug, Clone)]
pub struct FrameDelta {
    /// Frame number this delta applies to.
    pub frame: u64,
    /// Entity IDs that changed.
    pub changed_entities: Vec<u32>,
    /// Delta-encoded position components (x, y, z interleaved).
    pub position_deltas: Vec<i32>,
    /// Rotation data (smallest-three, per changed entity).
    pub rotation_data: Vec<(u8, [u32; 3])>,
    /// Whether this is a keyframe (full state, not delta).
    pub is_keyframe: bool,
}

impl FrameDelta {
    /// Create a keyframe (full state snapshot).
    pub fn keyframe(frame: u64) -> Self {
        Self {
            frame,
            changed_entities: Vec::new(),
            position_deltas: Vec::new(),
            rotation_data: Vec::new(),
            is_keyframe: true,
        }
    }

    /// Create a delta frame.
    pub fn delta(frame: u64) -> Self {
        Self {
            frame,
            changed_entities: Vec::new(),
            position_deltas: Vec::new(),
            rotation_data: Vec::new(),
            is_keyframe: false,
        }
    }

    /// Approximate byte size of this delta.
    pub fn byte_size(&self) -> usize {
        8 // frame number
        + self.changed_entities.len() * 4
        + self.position_deltas.len() * 4
        + self.rotation_data.len() * 13 // 1 + 3*4 bytes
        + 1 // is_keyframe flag
    }
}

// ---------------------------------------------------------------------------
// Run-length encoding for static entities
// ---------------------------------------------------------------------------

/// Run-length encode a sequence of entity state flags.
///
/// Each run is `(value, count)` meaning `count` consecutive entries all
/// have the same `value`.
pub fn rle_encode(data: &[u8]) -> Vec<(u8, u32)> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut current = data[0];
    let mut count = 1u32;
    for &byte in &data[1..] {
        if byte == current && count < u32::MAX {
            count += 1;
        } else {
            runs.push((current, count));
            current = byte;
            count = 1;
        }
    }
    runs.push((current, count));
    runs
}

/// Decode a run-length encoded sequence.
pub fn rle_decode(runs: &[(u8, u32)]) -> Vec<u8> {
    let total: usize = runs.iter().map(|(_, c)| *c as usize).sum();
    let mut data = Vec::with_capacity(total);
    for &(value, count) in runs {
        for _ in 0..count {
            data.push(value);
        }
    }
    data
}

/// Compress an RLE-encoded sequence to bytes using varint.
pub fn rle_to_bytes(runs: &[(u8, u32)]) -> Vec<u8> {
    let mut out = Vec::new();
    encode_varint_u32(runs.len() as u32, &mut out);
    for &(value, count) in runs {
        out.push(value);
        encode_varint_u32(count, &mut out);
    }
    out
}

/// Decompress bytes back to RLE runs.
pub fn rle_from_bytes(data: &[u8]) -> Result<Vec<(u8, u32)>, CompressionError> {
    let (run_count, mut pos) = decode_varint_u32(data)?;
    let mut runs = Vec::with_capacity(run_count as usize);
    for _ in 0..run_count {
        if pos >= data.len() {
            return Err(CompressionError::BufferTooShort);
        }
        let value = data[pos];
        pos += 1;
        let (count, consumed) = decode_varint_u32(&data[pos..])?;
        pos += consumed;
        runs.push((value, count));
    }
    Ok(runs)
}

// ---------------------------------------------------------------------------
// Input compression
// ---------------------------------------------------------------------------

/// A single input event in a replay stream.
#[derive(Debug, Clone, PartialEq)]
pub struct InputDelta {
    /// Frame offset from the previous input event.
    pub frame_offset: u32,
    /// Input channel (e.g. button index, axis index).
    pub channel: u16,
    /// New value for this channel (quantized).
    pub value: i32,
}

/// Compress a sequence of input events.
///
/// Only stores changes (events where the value differs from the previous
/// known value for that channel).
pub fn compress_inputs(events: &[InputDelta]) -> Vec<u8> {
    let mut out = Vec::new();
    encode_varint_u32(events.len() as u32, &mut out);
    for event in events {
        encode_varint_u32(event.frame_offset, &mut out);
        out.extend_from_slice(&event.channel.to_le_bytes());
        encode_varint_i32(event.value, &mut out);
    }
    out
}

/// Decompress a sequence of input events.
pub fn decompress_inputs(data: &[u8]) -> Result<Vec<InputDelta>, CompressionError> {
    let (count, mut pos) = decode_varint_u32(data)?;
    let mut events = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let (frame_offset, consumed) = decode_varint_u32(&data[pos..])?;
        pos += consumed;
        if pos + 2 > data.len() {
            return Err(CompressionError::BufferTooShort);
        }
        let channel = u16::from_le_bytes([data[pos], data[pos + 1]]);
        pos += 2;
        let (value, consumed) = decode_varint_i32(&data[pos..])?;
        pos += consumed;
        events.push(InputDelta {
            frame_offset,
            channel,
            value,
        });
    }
    Ok(events)
}

/// Track input channels to detect changes.
pub struct InputChangeTracker {
    /// Last known value for each channel.
    last_values: HashMap<u16, i32>,
    /// Accumulated delta events.
    deltas: Vec<InputDelta>,
    /// Current frame.
    current_frame: u64,
    /// Frame of the last recorded delta.
    last_delta_frame: u64,
}

impl InputChangeTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self {
            last_values: HashMap::new(),
            deltas: Vec::new(),
            current_frame: 0,
            last_delta_frame: 0,
        }
    }

    /// Set the current frame.
    pub fn set_frame(&mut self, frame: u64) {
        self.current_frame = frame;
    }

    /// Record an input value. Only creates a delta if the value changed.
    pub fn record(&mut self, channel: u16, value: i32) {
        let changed = self
            .last_values
            .get(&channel)
            .map_or(true, |&last| last != value);
        if changed {
            let frame_offset = (self.current_frame - self.last_delta_frame) as u32;
            self.deltas.push(InputDelta {
                frame_offset,
                channel,
                value,
            });
            self.last_values.insert(channel, value);
            self.last_delta_frame = self.current_frame;
        }
    }

    /// Take accumulated deltas.
    pub fn take_deltas(&mut self) -> Vec<InputDelta> {
        std::mem::take(&mut self.deltas)
    }

    /// Number of accumulated deltas.
    pub fn delta_count(&self) -> usize {
        self.deltas.len()
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.last_values.clear();
        self.deltas.clear();
        self.current_frame = 0;
        self.last_delta_frame = 0;
    }
}

impl Default for InputChangeTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Replay file size analysis
// ---------------------------------------------------------------------------

/// Breakdown of replay file size by component.
#[derive(Debug, Clone)]
pub struct ReplaySizeAnalysis {
    /// Total file size in bytes.
    pub total_bytes: u64,
    /// Header size.
    pub header_bytes: u64,
    /// Input data size.
    pub input_bytes: u64,
    /// Entity state (transforms) size.
    pub entity_bytes: u64,
    /// Checkpoint (keyframe) size.
    pub checkpoint_bytes: u64,
    /// Event data size.
    pub event_bytes: u64,
    /// Number of frames.
    pub frame_count: u64,
    /// Number of keyframes.
    pub keyframe_count: u64,
    /// Average bytes per frame.
    pub avg_bytes_per_frame: f64,
    /// Average bytes per entity per frame.
    pub avg_bytes_per_entity_per_frame: f64,
    /// Number of entities tracked.
    pub entity_count: u32,
    /// Compression ratio (compressed / uncompressed).
    pub compression_ratio: f64,
    /// Duration of the replay in seconds.
    pub duration_seconds: f64,
    /// Bytes per second of replay.
    pub bytes_per_second: f64,
}

impl ReplaySizeAnalysis {
    /// Create a new analysis from component sizes.
    pub fn new(
        header_bytes: u64,
        input_bytes: u64,
        entity_bytes: u64,
        checkpoint_bytes: u64,
        event_bytes: u64,
        frame_count: u64,
        keyframe_count: u64,
        entity_count: u32,
        uncompressed_size: u64,
        duration_seconds: f64,
    ) -> Self {
        let total_bytes = header_bytes + input_bytes + entity_bytes + checkpoint_bytes + event_bytes;
        let avg_bytes_per_frame = if frame_count > 0 {
            total_bytes as f64 / frame_count as f64
        } else {
            0.0
        };
        let avg_bytes_per_entity_per_frame = if frame_count > 0 && entity_count > 0 {
            entity_bytes as f64 / (frame_count as f64 * entity_count as f64)
        } else {
            0.0
        };
        let compression_ratio = if uncompressed_size > 0 {
            total_bytes as f64 / uncompressed_size as f64
        } else {
            1.0
        };
        let bytes_per_second = if duration_seconds > 0.0 {
            total_bytes as f64 / duration_seconds
        } else {
            0.0
        };
        Self {
            total_bytes,
            header_bytes,
            input_bytes,
            entity_bytes,
            checkpoint_bytes,
            event_bytes,
            frame_count,
            keyframe_count,
            avg_bytes_per_frame,
            avg_bytes_per_entity_per_frame,
            entity_count,
            compression_ratio,
            duration_seconds,
            bytes_per_second,
        }
    }

    /// Format as a human-readable report.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Replay Size Analysis\n"));
        s.push_str(&format!("====================\n"));
        s.push_str(&format!("Total:        {} bytes ({:.1} KB)\n", self.total_bytes, self.total_bytes as f64 / 1024.0));
        s.push_str(&format!("  Header:     {} bytes\n", self.header_bytes));
        s.push_str(&format!("  Input:      {} bytes ({:.1}%)\n", self.input_bytes, self.input_bytes as f64 / self.total_bytes.max(1) as f64 * 100.0));
        s.push_str(&format!("  Entities:   {} bytes ({:.1}%)\n", self.entity_bytes, self.entity_bytes as f64 / self.total_bytes.max(1) as f64 * 100.0));
        s.push_str(&format!("  Checkpoints:{} bytes ({:.1}%)\n", self.checkpoint_bytes, self.checkpoint_bytes as f64 / self.total_bytes.max(1) as f64 * 100.0));
        s.push_str(&format!("  Events:     {} bytes ({:.1}%)\n", self.event_bytes, self.event_bytes as f64 / self.total_bytes.max(1) as f64 * 100.0));
        s.push_str(&format!("Frames:       {}\n", self.frame_count));
        s.push_str(&format!("Keyframes:    {}\n", self.keyframe_count));
        s.push_str(&format!("Entities:     {}\n", self.entity_count));
        s.push_str(&format!("Avg/frame:    {:.1} bytes\n", self.avg_bytes_per_frame));
        s.push_str(&format!("Avg/entity/f: {:.1} bytes\n", self.avg_bytes_per_entity_per_frame));
        s.push_str(&format!("Compression:  {:.1}%\n", self.compression_ratio * 100.0));
        s.push_str(&format!("Duration:     {:.1}s\n", self.duration_seconds));
        s.push_str(&format!("Rate:         {:.0} bytes/s\n", self.bytes_per_second));
        s
    }
}

// ---------------------------------------------------------------------------
// Compressed replay stream
// ---------------------------------------------------------------------------

/// A compressed replay data stream that combines all compression techniques.
pub struct CompressedReplayStream {
    /// Compressed frame deltas.
    pub frame_deltas: Vec<FrameDelta>,
    /// Compressed input events.
    pub compressed_inputs: Vec<u8>,
    /// RLE-encoded static entity flags.
    pub static_entity_rle: Vec<u8>,
    /// Quantization precision used.
    pub precision: QuantizationPrecision,
    /// Bounds for position quantization.
    pub position_bounds_min: [f32; 3],
    pub position_bounds_max: [f32; 3],
    /// Keyframe interval (every N frames).
    pub keyframe_interval: u32,
    /// Total frames.
    pub total_frames: u64,
}

impl CompressedReplayStream {
    /// Create a new stream with default settings.
    pub fn new(keyframe_interval: u32, precision: QuantizationPrecision) -> Self {
        Self {
            frame_deltas: Vec::new(),
            compressed_inputs: Vec::new(),
            static_entity_rle: Vec::new(),
            precision,
            position_bounds_min: [-1000.0, -100.0, -1000.0],
            position_bounds_max: [1000.0, 500.0, 1000.0],
            keyframe_interval,
            total_frames: 0,
        }
    }

    /// Set the position bounds.
    pub fn set_position_bounds(&mut self, min: [f32; 3], max: [f32; 3]) {
        self.position_bounds_min = min;
        self.position_bounds_max = max;
    }

    /// Add a frame delta.
    pub fn add_frame_delta(&mut self, delta: FrameDelta) {
        self.total_frames = delta.frame + 1;
        self.frame_deltas.push(delta);
    }

    /// Approximate total size in bytes.
    pub fn approximate_size(&self) -> usize {
        let frame_size: usize = self.frame_deltas.iter().map(|d| d.byte_size()).sum();
        frame_size + self.compressed_inputs.len() + self.static_entity_rle.len()
    }

    /// Number of keyframes.
    pub fn keyframe_count(&self) -> usize {
        self.frame_deltas.iter().filter(|d| d.is_keyframe).count()
    }
}

// ---------------------------------------------------------------------------
// Bit-packing utilities
// ---------------------------------------------------------------------------

/// A simple bitstream writer for packing bits.
pub struct BitWriter {
    data: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
}

impl BitWriter {
    /// Create a new bit writer.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    /// Write `count` bits from `value`.
    pub fn write_bits(&mut self, value: u32, count: u8) {
        let mut remaining = count;
        let mut val = value;
        while remaining > 0 {
            let available = 8 - self.bit_pos;
            let bits_to_write = remaining.min(available);
            let mask = (1u32 << bits_to_write) - 1;
            self.current_byte |= ((val & mask) as u8) << self.bit_pos;
            val >>= bits_to_write;
            self.bit_pos += bits_to_write;
            remaining -= bits_to_write;
            if self.bit_pos == 8 {
                self.data.push(self.current_byte);
                self.current_byte = 0;
                self.bit_pos = 0;
            }
        }
    }

    /// Flush remaining bits (padded with zeros).
    pub fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.data.push(self.current_byte);
        }
        self.data
    }

    /// Total bits written.
    pub fn bits_written(&self) -> usize {
        self.data.len() * 8 + self.bit_pos as usize
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple bitstream reader.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read `count` bits as a `u32`.
    pub fn read_bits(&mut self, count: u8) -> Result<u32, CompressionError> {
        let mut result: u32 = 0;
        let mut remaining = count;
        let mut shift = 0;
        while remaining > 0 {
            if self.byte_pos >= self.data.len() {
                return Err(CompressionError::BufferTooShort);
            }
            let available = 8 - self.bit_pos;
            let bits_to_read = remaining.min(available);
            let mask = (1u32 << bits_to_read) - 1;
            let bits = ((self.data[self.byte_pos] >> self.bit_pos) as u32) & mask;
            result |= bits << shift;
            shift += bits_to_read;
            self.bit_pos += bits_to_read;
            remaining -= bits_to_read;
            if self.bit_pos == 8 {
                self.byte_pos += 1;
                self.bit_pos = 0;
            }
        }
        Ok(result)
    }

    /// Total bits remaining.
    pub fn bits_remaining(&self) -> usize {
        if self.byte_pos >= self.data.len() {
            return 0;
        }
        (self.data.len() - self.byte_pos) * 8 - self.bit_pos as usize
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_u32_small() {
        let mut buf = Vec::new();
        encode_varint_u32(42, &mut buf);
        let (val, consumed) = decode_varint_u32(&buf).unwrap();
        assert_eq!(val, 42);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_varint_u32_medium() {
        let mut buf = Vec::new();
        encode_varint_u32(300, &mut buf);
        let (val, consumed) = decode_varint_u32(&buf).unwrap();
        assert_eq!(val, 300);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_varint_u32_large() {
        let mut buf = Vec::new();
        encode_varint_u32(u32::MAX, &mut buf);
        let (val, _) = decode_varint_u32(&buf).unwrap();
        assert_eq!(val, u32::MAX);
    }

    #[test]
    fn test_varint_u64() {
        let mut buf = Vec::new();
        encode_varint_u64(1_000_000_000_000u64, &mut buf);
        let (val, _) = decode_varint_u64(&buf).unwrap();
        assert_eq!(val, 1_000_000_000_000u64);
    }

    #[test]
    fn test_varint_i32_positive() {
        let mut buf = Vec::new();
        encode_varint_i32(100, &mut buf);
        let (val, _) = decode_varint_i32(&buf).unwrap();
        assert_eq!(val, 100);
    }

    #[test]
    fn test_varint_i32_negative() {
        let mut buf = Vec::new();
        encode_varint_i32(-100, &mut buf);
        let (val, _) = decode_varint_i32(&buf).unwrap();
        assert_eq!(val, -100);
    }

    #[test]
    fn test_quantize_float_roundtrip() {
        let original = 0.75f32;
        let q = quantize_float(original, 0.0, 1.0, QuantizationPrecision::Medium);
        let decoded = dequantize_float(q, 0.0, 1.0, QuantizationPrecision::Medium);
        assert!((decoded - original).abs() < 0.001);
    }

    #[test]
    fn test_quantize_position_roundtrip() {
        let min = [-100.0, -50.0, -100.0];
        let max = [100.0, 50.0, 100.0];
        let q = quantize_position(25.0, -10.0, 50.0, min, max, QuantizationPrecision::Medium);
        let decoded = dequantize_position(q, min, max, QuantizationPrecision::Medium);
        assert!((decoded[0] - 25.0).abs() < 0.1);
        assert!((decoded[1] - (-10.0)).abs() < 0.1);
        assert!((decoded[2] - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_quantize_quaternion_roundtrip() {
        let (qx, qy, qz, qw) = (0.0, 0.0, 0.0, 1.0);
        let (max_idx, stored) = quantize_quaternion(qx, qy, qz, qw, QuantizationPrecision::Medium);
        let (dx, dy, dz, dw) = dequantize_quaternion(max_idx, stored, QuantizationPrecision::Medium);
        assert!((dx - qx).abs() < 0.01);
        assert!((dy - qy).abs() < 0.01);
        assert!((dz - qz).abs() < 0.01);
        assert!((dw - qw).abs() < 0.05);
    }

    #[test]
    fn test_delta_encode_decode_u32() {
        let values = vec![100, 105, 103, 110, 110];
        let deltas = delta_encode_u32(&values);
        let decoded = delta_decode_u32(&deltas);
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_rle_encode_decode() {
        let data = vec![0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2];
        let runs = rle_encode(&data);
        let decoded = rle_decode(&runs);
        assert_eq!(decoded, data);
        assert_eq!(runs.len(), 4); // 3 zeros, 2 ones, 5 zeros, 1 two
    }

    #[test]
    fn test_rle_bytes_roundtrip() {
        let data = vec![1, 1, 1, 0, 0, 2, 2, 2, 2];
        let runs = rle_encode(&data);
        let bytes = rle_to_bytes(&runs);
        let decoded_runs = rle_from_bytes(&bytes).unwrap();
        let decoded = rle_decode(&decoded_runs);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_input_compression_roundtrip() {
        let events = vec![
            InputDelta { frame_offset: 0, channel: 0, value: 100 },
            InputDelta { frame_offset: 5, channel: 1, value: -50 },
            InputDelta { frame_offset: 10, channel: 0, value: 0 },
        ];
        let compressed = compress_inputs(&events);
        let decompressed = decompress_inputs(&compressed).unwrap();
        assert_eq!(decompressed, events);
    }

    #[test]
    fn test_input_change_tracker() {
        let mut tracker = InputChangeTracker::new();
        tracker.set_frame(0);
        tracker.record(0, 100);
        tracker.record(1, 200);
        tracker.set_frame(1);
        tracker.record(0, 100); // no change
        tracker.record(1, 250); // changed
        assert_eq!(tracker.delta_count(), 3); // initial 0, initial 1, change 1
    }

    #[test]
    fn test_bit_writer_reader() {
        let mut writer = BitWriter::new();
        writer.write_bits(5, 3);  // 101
        writer.write_bits(3, 2);  // 11
        writer.write_bits(10, 4); // 1010
        let data = writer.finish();

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(3).unwrap(), 5);
        assert_eq!(reader.read_bits(2).unwrap(), 3);
        assert_eq!(reader.read_bits(4).unwrap(), 10);
    }

    #[test]
    fn test_bit_writer_cross_byte() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xFF, 8);
        writer.write_bits(0xAB, 8);
        let data = writer.finish();
        assert_eq!(data, vec![0xFF, 0xAB]);
    }

    #[test]
    fn test_replay_size_analysis() {
        let analysis = ReplaySizeAnalysis::new(
            64,     // header
            1024,   // input
            8192,   // entities
            2048,   // checkpoints
            512,    // events
            1800,   // frames (30 fps * 60 seconds)
            6,      // keyframes
            50,     // entities
            100_000, // uncompressed
            60.0,    // duration
        );
        assert_eq!(analysis.total_bytes, 11840);
        assert!(analysis.avg_bytes_per_frame > 0.0);
        assert!(analysis.compression_ratio < 1.0);
        assert!(analysis.bytes_per_second > 0.0);
        let report = analysis.report();
        assert!(report.contains("Replay Size Analysis"));
    }

    #[test]
    fn test_compressed_replay_stream() {
        let mut stream = CompressedReplayStream::new(60, QuantizationPrecision::Medium);
        stream.add_frame_delta(FrameDelta::keyframe(0));
        stream.add_frame_delta(FrameDelta::delta(1));
        stream.add_frame_delta(FrameDelta::delta(2));
        assert_eq!(stream.total_frames, 3);
        assert_eq!(stream.keyframe_count(), 1);
    }

    #[test]
    fn test_delta_encode_f32() {
        let values = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let deltas = delta_encode_f32(&values, 0.0, 10.0, QuantizationPrecision::Medium);
        let decoded = delta_decode_f32(&deltas, 0.0, 10.0, QuantizationPrecision::Medium);
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01);
        }
    }

    #[test]
    fn test_frame_delta_byte_size() {
        let mut delta = FrameDelta::delta(100);
        delta.changed_entities = vec![1, 2, 3];
        delta.position_deltas = vec![0, 1, -1, 2, 0, 0, -2, 1, 0];
        assert!(delta.byte_size() > 0);
    }
}
