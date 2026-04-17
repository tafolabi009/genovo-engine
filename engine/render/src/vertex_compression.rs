// engine/render/src/vertex_compression.rs
//
// Vertex data compression for the Genovo engine.
//
// Provides utilities for compressing vertex attributes to reduce GPU memory
// bandwidth and storage:
//
// - **Position quantization** — Quantize float positions to 16-bit integers
//   within a bounding box.
// - **Normal encoding** — Octahedron-encode unit normals to 2x 8-bit or
//   2x 16-bit integers.
// - **UV quantization** — Quantize texture coordinates to 16-bit integers.
// - **Tangent frame as quaternion** — Pack the full TBN frame into a single
//   quaternion (4x 8-bit).
// - **Packed vertex formats** — Pre-defined compressed vertex layouts.
// - **Compression ratio analysis** — Compare uncompressed and compressed
//   sizes.
//
// # GPU pipeline
//
// Compressed vertices are decoded in the vertex shader. The decode functions
// are designed to have minimal ALU cost:
// - Positions: `world_pos = compressed * scale + offset`
// - Normals: octahedron decode (a few MAD + normalize)
// - UVs: `uv = compressed * uv_scale + uv_offset`
// - TBN: quaternion-to-matrix

// ---------------------------------------------------------------------------
// Position quantization (16-bit)
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box for position quantization.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationBounds {
    /// Minimum corner.
    pub min: [f32; 3],
    /// Maximum corner.
    pub max: [f32; 3],
}

impl QuantizationBounds {
    /// Create bounds from min and max.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Compute bounds from a set of positions.
    pub fn from_positions(positions: &[[f32; 3]]) -> Self {
        if positions.is_empty() {
            return Self {
                min: [0.0; 3],
                max: [1.0; 3],
            };
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for pos in positions {
            for i in 0..3 {
                min[i] = min[i].min(pos[i]);
                max[i] = max[i].max(pos[i]);
            }
        }

        // Add a small epsilon to avoid degenerate axes.
        for i in 0..3 {
            if (max[i] - min[i]).abs() < 1e-6 {
                max[i] = min[i] + 1e-6;
            }
        }

        Self { min, max }
    }

    /// Get the scale factor for quantization.
    pub fn scale(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Get the inverse scale for decoding.
    pub fn inv_scale(&self) -> [f32; 3] {
        let s = self.scale();
        [
            if s[0].abs() > 1e-10 { 1.0 / s[0] } else { 0.0 },
            if s[1].abs() > 1e-10 { 1.0 / s[1] } else { 0.0 },
            if s[2].abs() > 1e-10 { 1.0 / s[2] } else { 0.0 },
        ]
    }

    /// Expand bounds by a margin.
    pub fn expand(&mut self, margin: f32) {
        for i in 0..3 {
            self.min[i] -= margin;
            self.max[i] += margin;
        }
    }

    /// Get the center.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Get the diagonal length.
    pub fn diagonal(&self) -> f32 {
        let s = self.scale();
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
    }
}

/// Quantize a float position to 16-bit integers within the given bounds.
///
/// Each component is mapped from [min, max] to [0, 65535].
pub fn quantize_position_16(pos: [f32; 3], bounds: &QuantizationBounds) -> [u16; 3] {
    let inv_scale = bounds.inv_scale();
    [
        ((pos[0] - bounds.min[0]) * inv_scale[0] * 65535.0).round().clamp(0.0, 65535.0) as u16,
        ((pos[1] - bounds.min[1]) * inv_scale[1] * 65535.0).round().clamp(0.0, 65535.0) as u16,
        ((pos[2] - bounds.min[2]) * inv_scale[2] * 65535.0).round().clamp(0.0, 65535.0) as u16,
    ]
}

/// Decode a 16-bit quantized position back to float.
pub fn dequantize_position_16(quantized: [u16; 3], bounds: &QuantizationBounds) -> [f32; 3] {
    let scale = bounds.scale();
    [
        quantized[0] as f32 / 65535.0 * scale[0] + bounds.min[0],
        quantized[1] as f32 / 65535.0 * scale[1] + bounds.min[1],
        quantized[2] as f32 / 65535.0 * scale[2] + bounds.min[2],
    ]
}

/// Batch quantize positions.
pub fn quantize_positions(positions: &[[f32; 3]], bounds: &QuantizationBounds) -> Vec<[u16; 3]> {
    positions.iter().map(|p| quantize_position_16(*p, bounds)).collect()
}

/// Batch dequantize positions.
pub fn dequantize_positions(quantized: &[[u16; 3]], bounds: &QuantizationBounds) -> Vec<[f32; 3]> {
    quantized.iter().map(|q| dequantize_position_16(*q, bounds)).collect()
}

/// Compute the maximum quantization error for a set of positions.
pub fn position_quantization_error(
    original: &[[f32; 3]],
    bounds: &QuantizationBounds,
) -> QuantizationError {
    let mut max_error = 0.0f32;
    let mut avg_error = 0.0f32;

    for pos in original {
        let q = quantize_position_16(*pos, bounds);
        let dq = dequantize_position_16(q, bounds);

        let err = ((pos[0] - dq[0]).powi(2) + (pos[1] - dq[1]).powi(2) + (pos[2] - dq[2]).powi(2)).sqrt();
        max_error = max_error.max(err);
        avg_error += err;
    }

    if !original.is_empty() {
        avg_error /= original.len() as f32;
    }

    QuantizationError {
        max_error,
        avg_error,
        relative_error: max_error / bounds.diagonal().max(1e-10),
    }
}

/// Quantization error metrics.
#[derive(Debug, Clone)]
pub struct QuantizationError {
    /// Maximum absolute error (world units).
    pub max_error: f32,
    /// Average absolute error (world units).
    pub avg_error: f32,
    /// Maximum error relative to the bounding box diagonal.
    pub relative_error: f32,
}

// ---------------------------------------------------------------------------
// Normal encoding (octahedron)
// ---------------------------------------------------------------------------

/// Encode a unit normal to octahedron mapping (float [-1, 1]).
///
/// Maps a unit vector on S^2 to the unit square [-1, 1]^2 using the
/// octahedron encoding by Meyer et al.
pub fn octahedron_encode(normal: [f32; 3]) -> [f32; 2] {
    let n = normal;
    let abs_sum = n[0].abs() + n[1].abs() + n[2].abs();
    if abs_sum < 1e-10 {
        return [0.0, 0.0];
    }

    // Project onto the octahedron.
    let mut oct = [n[0] / abs_sum, n[1] / abs_sum];

    // Reflect the folds of the lower hemisphere.
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

/// Decode an octahedron-encoded normal back to a unit vector.
pub fn octahedron_decode(oct: [f32; 2]) -> [f32; 3] {
    let mut n = [oct[0], oct[1], 1.0 - oct[0].abs() - oct[1].abs()];

    if n[2] < 0.0 {
        let sign_x = if n[0] >= 0.0 { 1.0 } else { -1.0 };
        let sign_y = if n[1] >= 0.0 { 1.0 } else { -1.0 };
        n[0] = (1.0 - n[1].abs()) * sign_x;
        n[1] = (1.0 - oct[0].abs()) * sign_y;
    }

    // Normalise.
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 1e-10 {
        let inv = 1.0 / len;
        [n[0] * inv, n[1] * inv, n[2] * inv]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Encode a normal to 2x 8-bit octahedron encoding.
pub fn octahedron_encode_8bit(normal: [f32; 3]) -> [u8; 2] {
    let oct = octahedron_encode(normal);
    [
        ((oct[0] * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((oct[1] * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

/// Decode a 2x 8-bit octahedron-encoded normal.
pub fn octahedron_decode_8bit(encoded: [u8; 2]) -> [f32; 3] {
    let oct = [
        encoded[0] as f32 / 255.0 * 2.0 - 1.0,
        encoded[1] as f32 / 255.0 * 2.0 - 1.0,
    ];
    octahedron_decode(oct)
}

/// Encode a normal to 2x 16-bit octahedron encoding.
pub fn octahedron_encode_16bit(normal: [f32; 3]) -> [u16; 2] {
    let oct = octahedron_encode(normal);
    [
        ((oct[0] * 0.5 + 0.5) * 65535.0).round().clamp(0.0, 65535.0) as u16,
        ((oct[1] * 0.5 + 0.5) * 65535.0).round().clamp(0.0, 65535.0) as u16,
    ]
}

/// Decode a 2x 16-bit octahedron-encoded normal.
pub fn octahedron_decode_16bit(encoded: [u16; 2]) -> [f32; 3] {
    let oct = [
        encoded[0] as f32 / 65535.0 * 2.0 - 1.0,
        encoded[1] as f32 / 65535.0 * 2.0 - 1.0,
    ];
    octahedron_decode(oct)
}

/// Batch encode normals to 8-bit octahedron.
pub fn octahedron_encode_normals_8bit(normals: &[[f32; 3]]) -> Vec<[u8; 2]> {
    normals.iter().map(|n| octahedron_encode_8bit(*n)).collect()
}

/// Compute the maximum angular error of octahedron encoding for a set of normals.
pub fn normal_encoding_error(normals: &[[f32; 3]], bits: u32) -> f32 {
    let mut max_error = 0.0f32;

    for normal in normals {
        let decoded = if bits <= 8 {
            let enc = octahedron_encode_8bit(*normal);
            octahedron_decode_8bit(enc)
        } else {
            let enc = octahedron_encode_16bit(*normal);
            octahedron_decode_16bit(enc)
        };

        // Angular error.
        let dot = normal[0] * decoded[0] + normal[1] * decoded[1] + normal[2] * decoded[2];
        let angle = dot.clamp(-1.0, 1.0).acos();
        max_error = max_error.max(angle);
    }

    max_error
}

// ---------------------------------------------------------------------------
// UV quantization (16-bit)
// ---------------------------------------------------------------------------

/// UV quantization bounds.
#[derive(Debug, Clone, Copy)]
pub struct UvBounds {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl UvBounds {
    /// Standard [0, 1] range.
    pub fn standard() -> Self {
        Self { min: [0.0, 0.0], max: [1.0, 1.0] }
    }

    /// Compute bounds from a set of UVs.
    pub fn from_uvs(uvs: &[[f32; 2]]) -> Self {
        if uvs.is_empty() {
            return Self::standard();
        }

        let mut min = [f32::MAX; 2];
        let mut max = [f32::MIN; 2];

        for uv in uvs {
            for i in 0..2 {
                min[i] = min[i].min(uv[i]);
                max[i] = max[i].max(uv[i]);
            }
        }

        for i in 0..2 {
            if (max[i] - min[i]).abs() < 1e-6 {
                max[i] = min[i] + 1e-6;
            }
        }

        Self { min, max }
    }

    /// Scale for encoding.
    pub fn scale(&self) -> [f32; 2] {
        [self.max[0] - self.min[0], self.max[1] - self.min[1]]
    }

    /// Inverse scale for encoding.
    pub fn inv_scale(&self) -> [f32; 2] {
        let s = self.scale();
        [
            if s[0].abs() > 1e-10 { 1.0 / s[0] } else { 0.0 },
            if s[1].abs() > 1e-10 { 1.0 / s[1] } else { 0.0 },
        ]
    }
}

/// Quantize a UV to 16-bit.
pub fn quantize_uv_16(uv: [f32; 2], bounds: &UvBounds) -> [u16; 2] {
    let inv = bounds.inv_scale();
    [
        ((uv[0] - bounds.min[0]) * inv[0] * 65535.0).round().clamp(0.0, 65535.0) as u16,
        ((uv[1] - bounds.min[1]) * inv[1] * 65535.0).round().clamp(0.0, 65535.0) as u16,
    ]
}

/// Decode a 16-bit quantized UV.
pub fn dequantize_uv_16(quantized: [u16; 2], bounds: &UvBounds) -> [f32; 2] {
    let scale = bounds.scale();
    [
        quantized[0] as f32 / 65535.0 * scale[0] + bounds.min[0],
        quantized[1] as f32 / 65535.0 * scale[1] + bounds.min[1],
    ]
}

/// Batch quantize UVs.
pub fn quantize_uvs(uvs: &[[f32; 2]], bounds: &UvBounds) -> Vec<[u16; 2]> {
    uvs.iter().map(|uv| quantize_uv_16(*uv, bounds)).collect()
}

// ---------------------------------------------------------------------------
// Tangent frame as quaternion
// ---------------------------------------------------------------------------

/// Encode a TBN frame (tangent, bitangent, normal) as a quaternion (4x f32).
///
/// The quaternion represents the rotation from the default TBN basis
/// ([1,0,0], [0,1,0], [0,0,1]) to the given TBN.
pub fn encode_tbn_quaternion(
    tangent: [f32; 3],
    bitangent: [f32; 3],
    normal: [f32; 3],
) -> [f32; 4] {
    // Build the rotation matrix (column-major: columns are T, B, N).
    let m = [
        [tangent[0], bitangent[0], normal[0]],
        [tangent[1], bitangent[1], normal[1]],
        [tangent[2], bitangent[2], normal[2]],
    ];

    // Convert rotation matrix to quaternion.
    let trace = m[0][0] + m[1][1] + m[2][2];

    let (w, x, y, z) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (m[2][1] - m[1][2]) / s;
        let y = (m[0][2] - m[2][0]) / s;
        let z = (m[1][0] - m[0][1]) / s;
        (w, x, y, z)
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        let w = (m[2][1] - m[1][2]) / s;
        let x = 0.25 * s;
        let y = (m[0][1] + m[1][0]) / s;
        let z = (m[0][2] + m[2][0]) / s;
        (w, x, y, z)
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        let w = (m[0][2] - m[2][0]) / s;
        let x = (m[0][1] + m[1][0]) / s;
        let y = 0.25 * s;
        let z = (m[1][2] + m[2][1]) / s;
        (w, x, y, z)
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        let w = (m[1][0] - m[0][1]) / s;
        let x = (m[0][2] + m[2][0]) / s;
        let y = (m[1][2] + m[2][1]) / s;
        let z = 0.25 * s;
        (w, x, y, z)
    };

    // Ensure w is positive (canonical form).
    if w < 0.0 {
        [-x, -y, -z, -w]
    } else {
        [x, y, z, w]
    }
}

/// Decode a quaternion back to TBN vectors.
pub fn decode_tbn_quaternion(q: [f32; 4]) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);

    let tangent = [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ];

    let bitangent = [
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ];

    let normal = [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ];

    (tangent, bitangent, normal)
}

/// Pack a TBN quaternion to 4x 8-bit.
pub fn pack_tbn_8bit(q: [f32; 4]) -> [u8; 4] {
    [
        ((q[0] * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((q[1] * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((q[2] * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((q[3] * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

/// Unpack a 4x 8-bit TBN quaternion.
pub fn unpack_tbn_8bit(packed: [u8; 4]) -> [f32; 4] {
    let q = [
        packed[0] as f32 / 255.0 * 2.0 - 1.0,
        packed[1] as f32 / 255.0 * 2.0 - 1.0,
        packed[2] as f32 / 255.0 * 2.0 - 1.0,
        packed[3] as f32 / 255.0 * 2.0 - 1.0,
    ];

    // Renormalise.
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 1e-10 {
        let inv = 1.0 / len;
        [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// Pack a TBN quaternion to 4x 16-bit.
pub fn pack_tbn_16bit(q: [f32; 4]) -> [u16; 4] {
    [
        ((q[0] * 0.5 + 0.5) * 65535.0).round().clamp(0.0, 65535.0) as u16,
        ((q[1] * 0.5 + 0.5) * 65535.0).round().clamp(0.0, 65535.0) as u16,
        ((q[2] * 0.5 + 0.5) * 65535.0).round().clamp(0.0, 65535.0) as u16,
        ((q[3] * 0.5 + 0.5) * 65535.0).round().clamp(0.0, 65535.0) as u16,
    ]
}

/// Unpack a 4x 16-bit TBN quaternion.
pub fn unpack_tbn_16bit(packed: [u16; 4]) -> [f32; 4] {
    let q = [
        packed[0] as f32 / 65535.0 * 2.0 - 1.0,
        packed[1] as f32 / 65535.0 * 2.0 - 1.0,
        packed[2] as f32 / 65535.0 * 2.0 - 1.0,
        packed[3] as f32 / 65535.0 * 2.0 - 1.0,
    ];

    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 1e-10 {
        let inv = 1.0 / len;
        [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

// ---------------------------------------------------------------------------
// Packed vertex formats
// ---------------------------------------------------------------------------

/// Uncompressed vertex (full float).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexFull {
    /// Position (x, y, z).
    pub position: [f32; 3],
    /// Normal (x, y, z).
    pub normal: [f32; 3],
    /// Tangent (x, y, z, w — w encodes bitangent sign).
    pub tangent: [f32; 4],
    /// Texture coordinate 0.
    pub uv0: [f32; 2],
    /// Texture coordinate 1.
    pub uv1: [f32; 2],
    /// Vertex colour.
    pub color: [u8; 4],
}

impl VertexFull {
    /// Size in bytes.
    pub const SIZE: usize = 3 * 4 + 3 * 4 + 4 * 4 + 2 * 4 + 2 * 4 + 4; // 56 bytes.
}

/// Compressed vertex (position quantized, normal octahedron-encoded).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexCompressed {
    /// Quantized position (16-bit per component).
    pub position: [u16; 3],
    /// Padding to align.
    pub _pad0: u16,
    /// Octahedron-encoded normal (8-bit per component).
    pub normal: [u8; 2],
    /// Tangent sign (1 bit encoded in MSB of a byte).
    pub tangent_sign: u8,
    /// Padding.
    pub _pad1: u8,
    /// Quantized UV0 (16-bit per component).
    pub uv0: [u16; 2],
    /// Vertex colour (RGBA).
    pub color: [u8; 4],
}

impl VertexCompressed {
    /// Size in bytes.
    pub const SIZE: usize = 6 + 2 + 2 + 1 + 1 + 4 + 4; // 20 bytes.
}

/// Minimally compressed vertex (position + normal only).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexMinimal {
    /// Quantized position (16-bit per component).
    pub position: [u16; 3],
    /// Octahedron-encoded normal (8-bit per component).
    pub normal: [u8; 2],
}

impl VertexMinimal {
    /// Size in bytes.
    pub const SIZE: usize = 6 + 2; // 8 bytes.
}

/// Full TBN compressed vertex.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexTbnCompressed {
    /// Quantized position (16-bit per component).
    pub position: [u16; 3],
    /// Padding.
    pub _pad: u16,
    /// TBN as quaternion (8-bit per component).
    pub tbn: [u8; 4],
    /// Quantized UV0 (16-bit per component).
    pub uv0: [u16; 2],
    /// Vertex colour (RGBA).
    pub color: [u8; 4],
}

impl VertexTbnCompressed {
    /// Size in bytes.
    pub const SIZE: usize = 8 + 4 + 4 + 4; // 20 bytes.
}

// ---------------------------------------------------------------------------
// Compression ratio analysis
// ---------------------------------------------------------------------------

/// Compression analysis report.
#[derive(Debug, Clone)]
pub struct CompressionReport {
    /// Number of vertices.
    pub vertex_count: u32,
    /// Uncompressed size in bytes.
    pub uncompressed_bytes: u64,
    /// Compressed size in bytes.
    pub compressed_bytes: u64,
    /// Compression ratio (uncompressed / compressed).
    pub ratio: f32,
    /// Space savings percentage.
    pub savings_percent: f32,
    /// Position quantization error.
    pub position_error: Option<QuantizationError>,
    /// Normal encoding max angular error (radians).
    pub normal_max_angle_error: Option<f32>,
    /// UV quantization max error.
    pub uv_max_error: Option<f32>,
}

impl CompressionReport {
    /// Compute compression statistics.
    pub fn analyse(
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        uvs: &[[f32; 2]],
        format: CompressionFormat,
    ) -> Self {
        let vertex_count = positions.len() as u32;

        let uncompressed_bytes = vertex_count as u64 * VertexFull::SIZE as u64;
        let compressed_bytes = match format {
            CompressionFormat::Compressed => vertex_count as u64 * VertexCompressed::SIZE as u64,
            CompressionFormat::Minimal => vertex_count as u64 * VertexMinimal::SIZE as u64,
            CompressionFormat::TbnCompressed => vertex_count as u64 * VertexTbnCompressed::SIZE as u64,
        };

        let ratio = if compressed_bytes > 0 {
            uncompressed_bytes as f32 / compressed_bytes as f32
        } else {
            0.0
        };

        let savings_percent = if uncompressed_bytes > 0 {
            (1.0 - compressed_bytes as f32 / uncompressed_bytes as f32) * 100.0
        } else {
            0.0
        };

        let position_error = if !positions.is_empty() {
            let bounds = QuantizationBounds::from_positions(positions);
            Some(position_quantization_error(positions, &bounds))
        } else {
            None
        };

        let normal_max_angle_error = if !normals.is_empty() {
            Some(normal_encoding_error(normals, 8))
        } else {
            None
        };

        let uv_max_error = if !uvs.is_empty() {
            let bounds = UvBounds::from_uvs(uvs);
            let mut max_err = 0.0f32;
            for uv in uvs {
                let q = quantize_uv_16(*uv, &bounds);
                let dq = dequantize_uv_16(q, &bounds);
                let err = ((uv[0] - dq[0]).powi(2) + (uv[1] - dq[1]).powi(2)).sqrt();
                max_err = max_err.max(err);
            }
            Some(max_err)
        } else {
            None
        };

        Self {
            vertex_count,
            uncompressed_bytes,
            compressed_bytes,
            ratio,
            savings_percent,
            position_error,
            normal_max_angle_error,
            uv_max_error,
        }
    }

    /// Format a summary string.
    pub fn summary(&self) -> String {
        format!(
            "Vertices: {} | Uncompressed: {} bytes | Compressed: {} bytes | \
             Ratio: {:.2}x | Savings: {:.1}%",
            self.vertex_count,
            self.uncompressed_bytes,
            self.compressed_bytes,
            self.ratio,
            self.savings_percent,
        )
    }
}

/// Compression format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionFormat {
    /// Standard compressed (position 16-bit, normal oct-8, UV 16-bit, colour).
    Compressed,
    /// Minimal (position 16-bit, normal oct-8 only).
    Minimal,
    /// TBN compressed (position 16-bit, TBN quat-8, UV 16-bit, colour).
    TbnCompressed,
}

// ---------------------------------------------------------------------------
// GPU decode constants
// ---------------------------------------------------------------------------

/// Constants needed on the GPU to decode compressed vertices.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexDecodeConstants {
    /// Position offset (bounds min, xyz) + padding (w).
    pub position_offset: [f32; 4],
    /// Position scale (bounds range, xyz) + padding (w).
    pub position_scale: [f32; 4],
    /// UV offset (bounds min, xy) + UV scale (bounds range, zw).
    pub uv_offset_scale: [f32; 4],
}

impl VertexDecodeConstants {
    /// Build decode constants from quantization bounds.
    pub fn from_bounds(
        pos_bounds: &QuantizationBounds,
        uv_bounds: &UvBounds,
    ) -> Self {
        let pos_scale = pos_bounds.scale();
        let uv_scale = uv_bounds.scale();

        Self {
            position_offset: [pos_bounds.min[0], pos_bounds.min[1], pos_bounds.min[2], 0.0],
            position_scale: [
                pos_scale[0] / 65535.0,
                pos_scale[1] / 65535.0,
                pos_scale[2] / 65535.0,
                0.0,
            ],
            uv_offset_scale: [
                uv_bounds.min[0],
                uv_bounds.min[1],
                uv_scale[0] / 65535.0,
                uv_scale[1] / 65535.0,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_quantize_roundtrip() {
        let bounds = QuantizationBounds::new([-10.0, -20.0, -30.0], [10.0, 20.0, 30.0]);
        let pos = [5.0, -10.0, 15.0];
        let q = quantize_position_16(pos, &bounds);
        let dq = dequantize_position_16(q, &bounds);

        for i in 0..3 {
            assert!((pos[i] - dq[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_octahedron_normals() {
        let test_normals = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ];

        for normal in &test_normals {
            let enc = octahedron_encode(*normal);
            let dec = octahedron_decode(enc);

            let dot = normal[0] * dec[0] + normal[1] * dec[1] + normal[2] * dec[2];
            assert!((dot - 1.0).abs() < 0.01, "Normal {:?} -> {:?} -> {:?}, dot={}", normal, enc, dec, dot);
        }
    }

    #[test]
    fn test_octahedron_8bit() {
        let normal = [0.577, 0.577, 0.577]; // ~normalised (1,1,1).
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let n = [normal[0] / len, normal[1] / len, normal[2] / len];

        let enc = octahedron_encode_8bit(n);
        let dec = octahedron_decode_8bit(enc);

        let dot = n[0] * dec[0] + n[1] * dec[1] + n[2] * dec[2];
        assert!(dot > 0.95);
    }

    #[test]
    fn test_uv_quantize() {
        let bounds = UvBounds::standard();
        let uv = [0.5, 0.75];
        let q = quantize_uv_16(uv, &bounds);
        let dq = dequantize_uv_16(q, &bounds);

        assert!((uv[0] - dq[0]).abs() < 0.001);
        assert!((uv[1] - dq[1]).abs() < 0.001);
    }

    #[test]
    fn test_tbn_quaternion() {
        let tangent = [1.0, 0.0, 0.0];
        let bitangent = [0.0, 1.0, 0.0];
        let normal = [0.0, 0.0, 1.0];

        let q = encode_tbn_quaternion(tangent, bitangent, normal);
        let (t, b, n) = decode_tbn_quaternion(q);

        let dot_t = tangent[0] * t[0] + tangent[1] * t[1] + tangent[2] * t[2];
        let dot_n = normal[0] * n[0] + normal[1] * n[1] + normal[2] * n[2];

        assert!(dot_t > 0.99, "Tangent mismatch: {:?} vs {:?}", tangent, t);
        assert!(dot_n > 0.99, "Normal mismatch: {:?} vs {:?}", normal, n);
    }

    #[test]
    fn test_compression_ratio() {
        let positions = vec![[1.0f32, 2.0, 3.0]; 1000];
        let normals = vec![[0.0f32, 1.0, 0.0]; 1000];
        let uvs = vec![[0.5f32, 0.5]; 1000];

        let report = CompressionReport::analyse(
            &positions, &normals, &uvs,
            CompressionFormat::Compressed,
        );

        assert!(report.ratio > 2.0); // Should be at least 2x compression.
        assert!(report.savings_percent > 50.0);
    }

    #[test]
    fn test_vertex_sizes() {
        assert_eq!(VertexFull::SIZE, 56);
        assert_eq!(VertexCompressed::SIZE, 20);
        assert_eq!(VertexMinimal::SIZE, 8);
    }
}
