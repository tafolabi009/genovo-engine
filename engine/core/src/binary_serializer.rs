// engine/core/src/binary_serializer.rs
//
// Binary serialization system for the Genovo engine.
//
// Provides a compact binary serialization format with schema versioning:
//
// - Compact binary encoding for primitive and composite types.
// - Schema versioning with forward/backward compatibility.
// - Field tags for selective deserialization.
// - Support for nested structs, arrays, maps.
// - Endian-aware encoding (little-endian default).
// - Variable-length integer encoding (varint) for compact sizes.
// - Stream-based writer and reader with position tracking.
// - CRC32 checksum for data integrity.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic number for binary format identification.
const BINARY_MAGIC: u32 = 0x474E5642; // "GNVB"

/// Current format version.
const FORMAT_VERSION: u16 = 1;

/// Maximum nesting depth.
const MAX_NESTING_DEPTH: u32 = 64;

/// Maximum string length.
const MAX_STRING_LENGTH: usize = 65536;

/// Maximum array length.
const MAX_ARRAY_LENGTH: usize = 1 << 24;

/// Maximum field tag value.
const MAX_FIELD_TAG: u32 = 65535;

// ---------------------------------------------------------------------------
// Type Tags
// ---------------------------------------------------------------------------

/// Type tag for identifying value types in the binary stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TypeTag {
    /// No value (null/none).
    Null = 0,
    /// Boolean.
    Bool = 1,
    /// 8-bit unsigned integer.
    U8 = 2,
    /// 16-bit unsigned integer.
    U16 = 3,
    /// 32-bit unsigned integer.
    U32 = 4,
    /// 64-bit unsigned integer.
    U64 = 5,
    /// 8-bit signed integer.
    I8 = 6,
    /// 16-bit signed integer.
    I16 = 7,
    /// 32-bit signed integer.
    I32 = 8,
    /// 64-bit signed integer.
    I64 = 9,
    /// 32-bit float.
    F32 = 10,
    /// 64-bit float.
    F64 = 11,
    /// UTF-8 string (length-prefixed).
    String = 12,
    /// Raw byte buffer (length-prefixed).
    Bytes = 13,
    /// Array of uniform type.
    Array = 14,
    /// Map (key-value pairs).
    Map = 15,
    /// Struct (tagged fields).
    Struct = 16,
    /// Variable-length integer.
    Varint = 17,
    /// Enum variant.
    Enum = 18,
    /// Optional value.
    Optional = 19,
}

impl TypeTag {
    /// Convert from u8.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Null),
            1 => Some(Self::Bool),
            2 => Some(Self::U8),
            3 => Some(Self::U16),
            4 => Some(Self::U32),
            5 => Some(Self::U64),
            6 => Some(Self::I8),
            7 => Some(Self::I16),
            8 => Some(Self::I32),
            9 => Some(Self::I64),
            10 => Some(Self::F32),
            11 => Some(Self::F64),
            12 => Some(Self::String),
            13 => Some(Self::Bytes),
            14 => Some(Self::Array),
            15 => Some(Self::Map),
            16 => Some(Self::Struct),
            17 => Some(Self::Varint),
            18 => Some(Self::Enum),
            19 => Some(Self::Optional),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Varint Encoding
// ---------------------------------------------------------------------------

/// Encode a u64 as a variable-length integer.
pub fn encode_varint(value: u64, buffer: &mut Vec<u8>) {
    let mut v = value;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buffer.push(byte);
            break;
        } else {
            buffer.push(byte | 0x80);
        }
    }
}

/// Decode a variable-length integer from a buffer.
pub fn decode_varint(buffer: &[u8], pos: &mut usize) -> Result<u64, BinaryError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    loop {
        if *pos >= buffer.len() {
            return Err(BinaryError::UnexpectedEnd);
        }
        let byte = buffer[*pos];
        *pos += 1;

        result |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(BinaryError::VarintOverflow);
        }
    }

    Ok(result)
}

/// Compute the encoded size of a varint.
pub fn varint_size(value: u64) -> usize {
    if value == 0 { return 1; }
    let bits = 64 - value.leading_zeros() as usize;
    (bits + 6) / 7
}

// ---------------------------------------------------------------------------
// Binary Writer
// ---------------------------------------------------------------------------

/// Writer for binary serialization.
#[derive(Debug)]
pub struct BinaryWriter {
    /// Output buffer.
    pub buffer: Vec<u8>,
    /// Current nesting depth.
    pub depth: u32,
    /// Schema version.
    pub schema_version: u16,
    /// Whether to write type tags.
    pub write_tags: bool,
    /// Whether to compute checksum.
    pub compute_checksum: bool,
}

impl BinaryWriter {
    /// Create a new binary writer.
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024),
            depth: 0,
            schema_version: 1,
            write_tags: true,
            compute_checksum: true,
        }
    }

    /// Create a writer with a pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            depth: 0,
            schema_version: 1,
            write_tags: true,
            compute_checksum: true,
        }
    }

    /// Write the file header.
    pub fn write_header(&mut self) {
        self.write_u32_raw(BINARY_MAGIC);
        self.write_u16_raw(FORMAT_VERSION);
        self.write_u16_raw(self.schema_version);
    }

    /// Write a raw u8.
    pub fn write_u8_raw(&mut self, value: u8) {
        self.buffer.push(value);
    }

    /// Write a raw u16 (little-endian).
    pub fn write_u16_raw(&mut self, value: u16) {
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a raw u32 (little-endian).
    pub fn write_u32_raw(&mut self, value: u32) {
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a raw u64 (little-endian).
    pub fn write_u64_raw(&mut self, value: u64) {
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a raw f32 (little-endian).
    pub fn write_f32_raw(&mut self, value: f32) {
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a raw f64 (little-endian).
    pub fn write_f64_raw(&mut self, value: f64) {
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a type tag.
    fn write_tag(&mut self, tag: TypeTag) {
        if self.write_tags {
            self.buffer.push(tag as u8);
        }
    }

    /// Write a boolean value.
    pub fn write_bool(&mut self, value: bool) {
        self.write_tag(TypeTag::Bool);
        self.buffer.push(if value { 1 } else { 0 });
    }

    /// Write a u8 value.
    pub fn write_u8(&mut self, value: u8) {
        self.write_tag(TypeTag::U8);
        self.write_u8_raw(value);
    }

    /// Write a u16 value.
    pub fn write_u16(&mut self, value: u16) {
        self.write_tag(TypeTag::U16);
        self.write_u16_raw(value);
    }

    /// Write a u32 value.
    pub fn write_u32(&mut self, value: u32) {
        self.write_tag(TypeTag::U32);
        self.write_u32_raw(value);
    }

    /// Write a u64 value.
    pub fn write_u64(&mut self, value: u64) {
        self.write_tag(TypeTag::U64);
        self.write_u64_raw(value);
    }

    /// Write an i32 value.
    pub fn write_i32(&mut self, value: i32) {
        self.write_tag(TypeTag::I32);
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    /// Write an i64 value.
    pub fn write_i64(&mut self, value: i64) {
        self.write_tag(TypeTag::I64);
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    /// Write an f32 value.
    pub fn write_f32(&mut self, value: f32) {
        self.write_tag(TypeTag::F32);
        self.write_f32_raw(value);
    }

    /// Write an f64 value.
    pub fn write_f64(&mut self, value: f64) {
        self.write_tag(TypeTag::F64);
        self.write_f64_raw(value);
    }

    /// Write a string value.
    pub fn write_string(&mut self, value: &str) {
        self.write_tag(TypeTag::String);
        let bytes = value.as_bytes();
        encode_varint(bytes.len() as u64, &mut self.buffer);
        self.buffer.extend_from_slice(bytes);
    }

    /// Write a byte buffer.
    pub fn write_bytes(&mut self, value: &[u8]) {
        self.write_tag(TypeTag::Bytes);
        encode_varint(value.len() as u64, &mut self.buffer);
        self.buffer.extend_from_slice(value);
    }

    /// Write a varint.
    pub fn write_varint(&mut self, value: u64) {
        self.write_tag(TypeTag::Varint);
        encode_varint(value, &mut self.buffer);
    }

    /// Begin an array. Returns the position where the length will be written.
    pub fn begin_array(&mut self, length: u32) {
        self.write_tag(TypeTag::Array);
        encode_varint(length as u64, &mut self.buffer);
    }

    /// Begin a struct with a field count.
    pub fn begin_struct(&mut self, field_count: u32) -> Result<(), BinaryError> {
        if self.depth >= MAX_NESTING_DEPTH {
            return Err(BinaryError::MaxDepthExceeded);
        }
        self.write_tag(TypeTag::Struct);
        encode_varint(field_count as u64, &mut self.buffer);
        self.depth += 1;
        Ok(())
    }

    /// Write a field tag within a struct.
    pub fn write_field_tag(&mut self, tag: u32) {
        encode_varint(tag as u64, &mut self.buffer);
    }

    /// End a struct.
    pub fn end_struct(&mut self) {
        if self.depth > 0 {
            self.depth -= 1;
        }
    }

    /// Write an optional value (None).
    pub fn write_none(&mut self) {
        self.write_tag(TypeTag::Optional);
        self.buffer.push(0);
    }

    /// Write an optional value (Some marker -- caller writes the actual value).
    pub fn write_some(&mut self) {
        self.write_tag(TypeTag::Optional);
        self.buffer.push(1);
    }

    /// Begin a map with a given entry count.
    pub fn begin_map(&mut self, entry_count: u32) {
        self.write_tag(TypeTag::Map);
        encode_varint(entry_count as u64, &mut self.buffer);
    }

    /// Finalize the buffer and append checksum.
    pub fn finalize(&mut self) -> &[u8] {
        if self.compute_checksum {
            let crc = crc32_compute(&self.buffer);
            self.write_u32_raw(crc);
        }
        &self.buffer
    }

    /// Get the current buffer size.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Get the buffer as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Take ownership of the buffer.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }

    /// Reset the writer for reuse.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.depth = 0;
    }
}

// ---------------------------------------------------------------------------
// Binary Reader
// ---------------------------------------------------------------------------

/// Reader for binary deserialization.
#[derive(Debug)]
pub struct BinaryReader<'a> {
    /// Input buffer.
    pub data: &'a [u8],
    /// Current read position.
    pub pos: usize,
    /// Current nesting depth.
    pub depth: u32,
    /// Whether to read type tags.
    pub read_tags: bool,
    /// Schema version from header.
    pub schema_version: u16,
}

impl<'a> BinaryReader<'a> {
    /// Create a new binary reader.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            depth: 0,
            read_tags: true,
            schema_version: 0,
        }
    }

    /// Read and validate the file header.
    pub fn read_header(&mut self) -> Result<(), BinaryError> {
        let magic = self.read_u32_raw()?;
        if magic != BINARY_MAGIC {
            return Err(BinaryError::InvalidMagic(magic));
        }
        let format_version = self.read_u16_raw()?;
        if format_version > FORMAT_VERSION {
            return Err(BinaryError::UnsupportedVersion(format_version));
        }
        self.schema_version = self.read_u16_raw()?;
        Ok(())
    }

    /// Read a raw u8.
    pub fn read_u8_raw(&mut self) -> Result<u8, BinaryError> {
        if self.pos >= self.data.len() {
            return Err(BinaryError::UnexpectedEnd);
        }
        let value = self.data[self.pos];
        self.pos += 1;
        Ok(value)
    }

    /// Read a raw u16.
    pub fn read_u16_raw(&mut self) -> Result<u16, BinaryError> {
        if self.pos + 2 > self.data.len() {
            return Err(BinaryError::UnexpectedEnd);
        }
        let value = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(value)
    }

    /// Read a raw u32.
    pub fn read_u32_raw(&mut self) -> Result<u32, BinaryError> {
        if self.pos + 4 > self.data.len() {
            return Err(BinaryError::UnexpectedEnd);
        }
        let bytes: [u8; 4] = [self.data[self.pos], self.data[self.pos + 1],
            self.data[self.pos + 2], self.data[self.pos + 3]];
        self.pos += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Read a raw u64.
    pub fn read_u64_raw(&mut self) -> Result<u64, BinaryError> {
        if self.pos + 8 > self.data.len() {
            return Err(BinaryError::UnexpectedEnd);
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_le_bytes(bytes))
    }

    /// Read a raw f32.
    pub fn read_f32_raw(&mut self) -> Result<f32, BinaryError> {
        let bits = self.read_u32_raw()?;
        Ok(f32::from_bits(bits))
    }

    /// Read a raw f64.
    pub fn read_f64_raw(&mut self) -> Result<f64, BinaryError> {
        let bits = self.read_u64_raw()?;
        Ok(f64::from_bits(bits))
    }

    /// Read a type tag.
    pub fn read_type_tag(&mut self) -> Result<TypeTag, BinaryError> {
        let byte = self.read_u8_raw()?;
        TypeTag::from_u8(byte).ok_or(BinaryError::UnknownTypeTag(byte))
    }

    /// Read a boolean.
    pub fn read_bool(&mut self) -> Result<bool, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::Bool)?; }
        Ok(self.read_u8_raw()? != 0)
    }

    /// Read a u32.
    pub fn read_u32(&mut self) -> Result<u32, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::U32)?; }
        self.read_u32_raw()
    }

    /// Read a u64.
    pub fn read_u64(&mut self) -> Result<u64, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::U64)?; }
        self.read_u64_raw()
    }

    /// Read an i32.
    pub fn read_i32(&mut self) -> Result<i32, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::I32)?; }
        let bits = self.read_u32_raw()?;
        Ok(bits as i32)
    }

    /// Read an f32.
    pub fn read_f32(&mut self) -> Result<f32, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::F32)?; }
        self.read_f32_raw()
    }

    /// Read an f64.
    pub fn read_f64(&mut self) -> Result<f64, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::F64)?; }
        self.read_f64_raw()
    }

    /// Read a string.
    pub fn read_string(&mut self) -> Result<String, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::String)?; }
        let len = decode_varint(self.data, &mut self.pos)? as usize;
        if len > MAX_STRING_LENGTH {
            return Err(BinaryError::StringTooLong(len));
        }
        if self.pos + len > self.data.len() {
            return Err(BinaryError::UnexpectedEnd);
        }
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|_| BinaryError::InvalidUtf8)?;
        self.pos += len;
        Ok(s.to_string())
    }

    /// Read a byte buffer.
    pub fn read_bytes(&mut self) -> Result<Vec<u8>, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::Bytes)?; }
        let len = decode_varint(self.data, &mut self.pos)? as usize;
        if self.pos + len > self.data.len() {
            return Err(BinaryError::UnexpectedEnd);
        }
        let bytes = self.data[self.pos..self.pos + len].to_vec();
        self.pos += len;
        Ok(bytes)
    }

    /// Read a varint.
    pub fn read_varint(&mut self) -> Result<u64, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::Varint)?; }
        decode_varint(self.data, &mut self.pos)
    }

    /// Read array header (returns length).
    pub fn read_array_header(&mut self) -> Result<u32, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::Array)?; }
        let len = decode_varint(self.data, &mut self.pos)?;
        Ok(len as u32)
    }

    /// Read struct header (returns field count).
    pub fn read_struct_header(&mut self) -> Result<u32, BinaryError> {
        if self.depth >= MAX_NESTING_DEPTH {
            return Err(BinaryError::MaxDepthExceeded);
        }
        if self.read_tags { self.expect_tag(TypeTag::Struct)?; }
        self.depth += 1;
        let count = decode_varint(self.data, &mut self.pos)?;
        Ok(count as u32)
    }

    /// Read a field tag.
    pub fn read_field_tag(&mut self) -> Result<u32, BinaryError> {
        let tag = decode_varint(self.data, &mut self.pos)?;
        Ok(tag as u32)
    }

    /// End struct reading.
    pub fn end_struct(&mut self) {
        if self.depth > 0 {
            self.depth -= 1;
        }
    }

    /// Read map header (returns entry count).
    pub fn read_map_header(&mut self) -> Result<u32, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::Map)?; }
        let count = decode_varint(self.data, &mut self.pos)?;
        Ok(count as u32)
    }

    /// Read optional header (returns true if Some).
    pub fn read_optional(&mut self) -> Result<bool, BinaryError> {
        if self.read_tags { self.expect_tag(TypeTag::Optional)?; }
        Ok(self.read_u8_raw()? != 0)
    }

    /// Expect a specific type tag.
    fn expect_tag(&mut self, expected: TypeTag) -> Result<(), BinaryError> {
        let actual = self.read_type_tag()?;
        if actual != expected {
            Err(BinaryError::TypeMismatch {
                expected: expected as u8,
                actual: actual as u8,
            })
        } else {
            Ok(())
        }
    }

    /// Remaining bytes.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Whether the reader has reached the end.
    pub fn is_at_end(&self) -> bool {
        self.pos >= self.data.len()
    }
}

// ---------------------------------------------------------------------------
// CRC32
// ---------------------------------------------------------------------------

/// Compute CRC32 of a byte slice.
pub fn crc32_compute(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
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
    !crc
}

/// Verify CRC32 checksum at the end of a buffer.
pub fn verify_checksum(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }
    let payload = &data[..data.len() - 4];
    let stored = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    crc32_compute(payload) == stored
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from binary serialization/deserialization.
#[derive(Debug, Clone)]
pub enum BinaryError {
    /// Unexpected end of data.
    UnexpectedEnd,
    /// Invalid magic number.
    InvalidMagic(u32),
    /// Unsupported format version.
    UnsupportedVersion(u16),
    /// Unknown type tag.
    UnknownTypeTag(u8),
    /// Type mismatch during reading.
    TypeMismatch { expected: u8, actual: u8 },
    /// String too long.
    StringTooLong(usize),
    /// Invalid UTF-8 string.
    InvalidUtf8,
    /// Varint overflow.
    VarintOverflow,
    /// Maximum nesting depth exceeded.
    MaxDepthExceeded,
    /// Checksum verification failed.
    ChecksumMismatch,
    /// Unknown field tag (for forward compatibility, can be skipped).
    UnknownField(u32),
}

impl std::fmt::Display for BinaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "Unexpected end of binary data"),
            Self::InvalidMagic(m) => write!(f, "Invalid magic number: 0x{:08X}", m),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported format version: {}", v),
            Self::UnknownTypeTag(t) => write!(f, "Unknown type tag: {}", t),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "Type mismatch: expected {}, got {}", expected, actual)
            }
            Self::StringTooLong(len) => write!(f, "String too long: {} bytes", len),
            Self::InvalidUtf8 => write!(f, "Invalid UTF-8 string"),
            Self::VarintOverflow => write!(f, "Varint overflow"),
            Self::MaxDepthExceeded => write!(f, "Maximum nesting depth exceeded"),
            Self::ChecksumMismatch => write!(f, "Checksum mismatch"),
            Self::UnknownField(tag) => write!(f, "Unknown field tag: {}", tag),
        }
    }
}
