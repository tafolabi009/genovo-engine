//! Engine serialization system for the Genovo engine.
//!
//! Provides a format-agnostic serialization interface with multiple concrete
//! backends: binary, JSON, and RON. Includes versioned data wrappers and a
//! migration registry for forward/backward compatibility.
//!
//! # Architecture
//!
//! The system is built around two trait pairs:
//!
//! - [`SerializeWriter`] / [`SerializeReader`] — low-level read/write of
//!   primitive types and structural markers (structs, arrays).
//! - [`Serialize`] — high-level trait for engine types that know how to
//!   serialize and deserialize themselves.
//!
//! Three concrete backends are provided:
//!
//! - [`BinaryWriter`] / [`BinaryReader`] — compact little-endian binary format
//!   with a version header and length-prefixed strings.
//! - [`JsonWriter`] / [`JsonReader`] — JSON text format backed by serde_json.
//! - [`RonWriter`] / [`RonReader`] — RON (Rusty Object Notation) format.
//!
//! # Versioning
//!
//! [`VersionedData`] wraps serialized data with a version number. The
//! [`MigrationRegistry`] stores migration functions that upgrade data from
//! version N to N+1, enabling forward compatibility.

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Write};

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during serialization/deserialization.
#[derive(Debug, Error)]
pub enum SerializeError {
    /// I/O error during read/write.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Data format is invalid or corrupt.
    #[error("Format error: {0}")]
    Format(String),

    /// Unexpected end of data.
    #[error("Unexpected end of data")]
    UnexpectedEof,

    /// Version mismatch.
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },

    /// Type mismatch during deserialization.
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        expected: String,
        actual: String,
    },

    /// Missing field during deserialization.
    #[error("Missing field: {0}")]
    MissingField(String),

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(String),

    /// Generic custom error.
    #[error("{0}")]
    Custom(String),
}

/// Result type for serialization operations.
pub type SerializeResult<T> = Result<T, SerializeError>;

// ---------------------------------------------------------------------------
// SerializeWriter trait
// ---------------------------------------------------------------------------

/// Trait for writing serialized data.
///
/// Implementations produce data in a specific format (binary, JSON, etc.)
/// while exposing a uniform API to [`Serialize`] implementations.
pub trait SerializeWriter {
    /// Write a boolean value.
    fn write_bool(&mut self, value: bool) -> SerializeResult<()>;
    /// Write an unsigned 8-bit integer.
    fn write_u8(&mut self, value: u8) -> SerializeResult<()>;
    /// Write an unsigned 16-bit integer.
    fn write_u16(&mut self, value: u16) -> SerializeResult<()>;
    /// Write an unsigned 32-bit integer.
    fn write_u32(&mut self, value: u32) -> SerializeResult<()>;
    /// Write an unsigned 64-bit integer.
    fn write_u64(&mut self, value: u64) -> SerializeResult<()>;
    /// Write a signed 32-bit integer.
    fn write_i32(&mut self, value: i32) -> SerializeResult<()>;
    /// Write a signed 64-bit integer.
    fn write_i64(&mut self, value: i64) -> SerializeResult<()>;
    /// Write a 32-bit float.
    fn write_f32(&mut self, value: f32) -> SerializeResult<()>;
    /// Write a 64-bit float.
    fn write_f64(&mut self, value: f64) -> SerializeResult<()>;
    /// Write a UTF-8 string.
    fn write_string(&mut self, value: &str) -> SerializeResult<()>;
    /// Write a byte slice.
    fn write_bytes(&mut self, value: &[u8]) -> SerializeResult<()>;
    /// Begin a struct with the given name.
    fn begin_struct(&mut self, name: &str) -> SerializeResult<()>;
    /// End the current struct.
    fn end_struct(&mut self) -> SerializeResult<()>;
    /// Begin an array of the given length.
    fn begin_array(&mut self, len: usize) -> SerializeResult<()>;
    /// End the current array.
    fn end_array(&mut self) -> SerializeResult<()>;
    /// Begin a named field within a struct.
    fn begin_field(&mut self, name: &str) -> SerializeResult<()>;
    /// End the current field.
    fn end_field(&mut self) -> SerializeResult<()>;
}

// ---------------------------------------------------------------------------
// SerializeReader trait
// ---------------------------------------------------------------------------

/// Trait for reading serialized data.
pub trait SerializeReader {
    /// Read a boolean value.
    fn read_bool(&mut self) -> SerializeResult<bool>;
    /// Read an unsigned 8-bit integer.
    fn read_u8(&mut self) -> SerializeResult<u8>;
    /// Read an unsigned 16-bit integer.
    fn read_u16(&mut self) -> SerializeResult<u16>;
    /// Read an unsigned 32-bit integer.
    fn read_u32(&mut self) -> SerializeResult<u32>;
    /// Read an unsigned 64-bit integer.
    fn read_u64(&mut self) -> SerializeResult<u64>;
    /// Read a signed 32-bit integer.
    fn read_i32(&mut self) -> SerializeResult<i32>;
    /// Read a signed 64-bit integer.
    fn read_i64(&mut self) -> SerializeResult<i64>;
    /// Read a 32-bit float.
    fn read_f32(&mut self) -> SerializeResult<f32>;
    /// Read a 64-bit float.
    fn read_f64(&mut self) -> SerializeResult<f64>;
    /// Read a UTF-8 string.
    fn read_string(&mut self) -> SerializeResult<String>;
    /// Read a byte slice.
    fn read_bytes(&mut self) -> SerializeResult<Vec<u8>>;
    /// Begin reading a struct, returning its name.
    fn begin_struct(&mut self) -> SerializeResult<String>;
    /// End reading the current struct.
    fn end_struct(&mut self) -> SerializeResult<()>;
    /// Begin reading an array, returning its length.
    fn begin_array(&mut self) -> SerializeResult<usize>;
    /// End reading the current array.
    fn end_array(&mut self) -> SerializeResult<()>;
    /// Begin reading a named field, returning its name.
    fn begin_field(&mut self) -> SerializeResult<String>;
    /// End reading the current field.
    fn end_field(&mut self) -> SerializeResult<()>;
}

// ---------------------------------------------------------------------------
// Serialize trait
// ---------------------------------------------------------------------------

/// Engine serialization interface.
///
/// Types implementing this trait can be serialized to and deserialized from
/// any format supported by the engine.
pub trait Serialize: Sized {
    /// Serialize this value to the given writer.
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()>;
    /// Deserialize a value from the given reader.
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self>;
}

// ---------------------------------------------------------------------------
// Serialize implementations for primitive types
// ---------------------------------------------------------------------------

impl Serialize for bool {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.write_bool(*self)
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.read_bool()
    }
}

impl Serialize for u8 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.write_u8(*self)
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.read_u8()
    }
}

impl Serialize for u16 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.write_u16(*self)
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.read_u16()
    }
}

impl Serialize for u32 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.write_u32(*self)
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.read_u32()
    }
}

impl Serialize for u64 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.write_u64(*self)
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.read_u64()
    }
}

impl Serialize for i32 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.write_i32(*self)
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.read_i32()
    }
}

impl Serialize for i64 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.write_i64(*self)
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.read_i64()
    }
}

impl Serialize for f32 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.write_f32(*self)
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.read_f32()
    }
}

impl Serialize for f64 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.write_f64(*self)
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.read_f64()
    }
}

impl Serialize for String {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.write_string(self)
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.read_string()
    }
}

impl<T: Serialize> Serialize for Vec<T> {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.begin_array(self.len())?;
        for item in self {
            item.serialize(writer)?;
        }
        writer.end_array()
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        let len = reader.begin_array()?;
        let mut items = Vec::with_capacity(len);
        for _ in 0..len {
            items.push(T::deserialize(reader)?);
        }
        reader.end_array()?;
        Ok(items)
    }
}

impl<T: Serialize> Serialize for Option<T> {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        match self {
            Some(val) => {
                writer.write_bool(true)?;
                val.serialize(writer)
            }
            None => writer.write_bool(false),
        }
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        if reader.read_bool()? {
            Ok(Some(T::deserialize(reader)?))
        } else {
            Ok(None)
        }
    }
}

// ---------------------------------------------------------------------------
// Serialize implementations for glam types
// ---------------------------------------------------------------------------

impl Serialize for Vec2 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.begin_struct("Vec2")?;
        writer.begin_field("x")?;
        writer.write_f32(self.x)?;
        writer.end_field()?;
        writer.begin_field("y")?;
        writer.write_f32(self.y)?;
        writer.end_field()?;
        writer.end_struct()
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.begin_struct()?;
        reader.begin_field()?;
        let x = reader.read_f32()?;
        reader.end_field()?;
        reader.begin_field()?;
        let y = reader.read_f32()?;
        reader.end_field()?;
        reader.end_struct()?;
        Ok(Vec2::new(x, y))
    }
}

impl Serialize for Vec3 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.begin_struct("Vec3")?;
        writer.begin_field("x")?;
        writer.write_f32(self.x)?;
        writer.end_field()?;
        writer.begin_field("y")?;
        writer.write_f32(self.y)?;
        writer.end_field()?;
        writer.begin_field("z")?;
        writer.write_f32(self.z)?;
        writer.end_field()?;
        writer.end_struct()
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.begin_struct()?;
        reader.begin_field()?;
        let x = reader.read_f32()?;
        reader.end_field()?;
        reader.begin_field()?;
        let y = reader.read_f32()?;
        reader.end_field()?;
        reader.begin_field()?;
        let z = reader.read_f32()?;
        reader.end_field()?;
        reader.end_struct()?;
        Ok(Vec3::new(x, y, z))
    }
}

impl Serialize for Vec4 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.begin_struct("Vec4")?;
        writer.begin_field("x")?;
        writer.write_f32(self.x)?;
        writer.end_field()?;
        writer.begin_field("y")?;
        writer.write_f32(self.y)?;
        writer.end_field()?;
        writer.begin_field("z")?;
        writer.write_f32(self.z)?;
        writer.end_field()?;
        writer.begin_field("w")?;
        writer.write_f32(self.w)?;
        writer.end_field()?;
        writer.end_struct()
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.begin_struct()?;
        reader.begin_field()?;
        let x = reader.read_f32()?;
        reader.end_field()?;
        reader.begin_field()?;
        let y = reader.read_f32()?;
        reader.end_field()?;
        reader.begin_field()?;
        let z = reader.read_f32()?;
        reader.end_field()?;
        reader.begin_field()?;
        let w = reader.read_f32()?;
        reader.end_field()?;
        reader.end_struct()?;
        Ok(Vec4::new(x, y, z, w))
    }
}

impl Serialize for Quat {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.begin_struct("Quat")?;
        writer.begin_field("x")?;
        writer.write_f32(self.x)?;
        writer.end_field()?;
        writer.begin_field("y")?;
        writer.write_f32(self.y)?;
        writer.end_field()?;
        writer.begin_field("z")?;
        writer.write_f32(self.z)?;
        writer.end_field()?;
        writer.begin_field("w")?;
        writer.write_f32(self.w)?;
        writer.end_field()?;
        writer.end_struct()
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.begin_struct()?;
        reader.begin_field()?;
        let x = reader.read_f32()?;
        reader.end_field()?;
        reader.begin_field()?;
        let y = reader.read_f32()?;
        reader.end_field()?;
        reader.begin_field()?;
        let z = reader.read_f32()?;
        reader.end_field()?;
        reader.begin_field()?;
        let w = reader.read_f32()?;
        reader.end_field()?;
        reader.end_struct()?;
        Ok(Quat::from_xyzw(x, y, z, w))
    }
}

impl Serialize for Mat4 {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        let cols = self.to_cols_array();
        writer.begin_struct("Mat4")?;
        writer.begin_array(16)?;
        for val in &cols {
            writer.write_f32(*val)?;
        }
        writer.end_array()?;
        writer.end_struct()
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.begin_struct()?;
        let len = reader.begin_array()?;
        if len != 16 {
            return Err(SerializeError::Format(format!(
                "Mat4 expects 16 floats, got {}",
                len
            )));
        }
        let mut cols = [0.0f32; 16];
        for col in &mut cols {
            *col = reader.read_f32()?;
        }
        reader.end_array()?;
        reader.end_struct()?;
        Ok(Mat4::from_cols_array(&cols))
    }
}

impl Serialize for crate::math::Transform {
    fn serialize(&self, writer: &mut dyn SerializeWriter) -> SerializeResult<()> {
        writer.begin_struct("Transform")?;
        writer.begin_field("position")?;
        self.position.serialize(writer)?;
        writer.end_field()?;
        writer.begin_field("rotation")?;
        self.rotation.serialize(writer)?;
        writer.end_field()?;
        writer.begin_field("scale")?;
        self.scale.serialize(writer)?;
        writer.end_field()?;
        writer.end_struct()
    }
    fn deserialize(reader: &mut dyn SerializeReader) -> SerializeResult<Self> {
        reader.begin_struct()?;
        reader.begin_field()?;
        let position = Vec3::deserialize(reader)?;
        reader.end_field()?;
        reader.begin_field()?;
        let rotation = Quat::deserialize(reader)?;
        reader.end_field()?;
        reader.begin_field()?;
        let scale = Vec3::deserialize(reader)?;
        reader.end_field()?;
        reader.end_struct()?;
        Ok(crate::math::Transform::new(position, rotation, scale))
    }
}

// ---------------------------------------------------------------------------
// BinaryWriter
// ---------------------------------------------------------------------------

/// Binary format magic number.
const BINARY_MAGIC: u32 = 0x474E564F; // "GNVO"

/// Current binary format version.
const BINARY_VERSION: u32 = 1;

/// Compact little-endian binary serialization writer.
///
/// Format:
/// - Header: 4-byte magic + 4-byte version
/// - Primitives: little-endian encoding
/// - Strings: 4-byte length prefix + UTF-8 bytes
/// - Bytes: 4-byte length prefix + raw bytes
/// - Structs: 4-byte name length + name bytes (begin) / no data (end)
/// - Arrays: 4-byte element count (begin) / no data (end)
pub struct BinaryWriter {
    buffer: Vec<u8>,
    header_written: bool,
}

impl BinaryWriter {
    /// Create a new binary writer.
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(4096),
            header_written: false,
        }
    }

    /// Ensure the header has been written.
    fn ensure_header(&mut self) {
        if !self.header_written {
            self.buffer.extend_from_slice(&BINARY_MAGIC.to_le_bytes());
            self.buffer
                .extend_from_slice(&BINARY_VERSION.to_le_bytes());
            self.header_written = true;
        }
    }

    /// Get the serialized bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }

    /// Get a reference to the serialized bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Get the current size of the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Write the buffer to a `std::io::Write`.
    pub fn write_to(&self, writer: &mut dyn Write) -> io::Result<()> {
        writer.write_all(&self.buffer)
    }
}

impl Default for BinaryWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl SerializeWriter for BinaryWriter {
    fn write_bool(&mut self, value: bool) -> SerializeResult<()> {
        self.ensure_header();
        self.buffer.push(if value { 1 } else { 0 });
        Ok(())
    }

    fn write_u8(&mut self, value: u8) -> SerializeResult<()> {
        self.ensure_header();
        self.buffer.push(value);
        Ok(())
    }

    fn write_u16(&mut self, value: u16) -> SerializeResult<()> {
        self.ensure_header();
        self.buffer.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn write_u32(&mut self, value: u32) -> SerializeResult<()> {
        self.ensure_header();
        self.buffer.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn write_u64(&mut self, value: u64) -> SerializeResult<()> {
        self.ensure_header();
        self.buffer.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn write_i32(&mut self, value: i32) -> SerializeResult<()> {
        self.ensure_header();
        self.buffer.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn write_i64(&mut self, value: i64) -> SerializeResult<()> {
        self.ensure_header();
        self.buffer.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn write_f32(&mut self, value: f32) -> SerializeResult<()> {
        self.ensure_header();
        self.buffer.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn write_f64(&mut self, value: f64) -> SerializeResult<()> {
        self.ensure_header();
        self.buffer.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn write_string(&mut self, value: &str) -> SerializeResult<()> {
        self.ensure_header();
        let bytes = value.as_bytes();
        self.buffer
            .extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        self.buffer.extend_from_slice(bytes);
        Ok(())
    }

    fn write_bytes(&mut self, value: &[u8]) -> SerializeResult<()> {
        self.ensure_header();
        self.buffer
            .extend_from_slice(&(value.len() as u32).to_le_bytes());
        self.buffer.extend_from_slice(value);
        Ok(())
    }

    fn begin_struct(&mut self, name: &str) -> SerializeResult<()> {
        self.write_string(name)
    }

    fn end_struct(&mut self) -> SerializeResult<()> {
        Ok(())
    }

    fn begin_array(&mut self, len: usize) -> SerializeResult<()> {
        self.write_u32(len as u32)
    }

    fn end_array(&mut self) -> SerializeResult<()> {
        Ok(())
    }

    fn begin_field(&mut self, name: &str) -> SerializeResult<()> {
        self.write_string(name)
    }

    fn end_field(&mut self) -> SerializeResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BinaryReader
// ---------------------------------------------------------------------------

/// Compact little-endian binary deserialization reader.
pub struct BinaryReader {
    data: Vec<u8>,
    pos: usize,
    version: u32,
    header_read: bool,
}

impl BinaryReader {
    /// Create a reader from a byte buffer.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            pos: 0,
            version: 0,
            header_read: false,
        }
    }

    /// Create a reader from a `std::io::Read`.
    pub fn from_reader(reader: &mut dyn Read) -> io::Result<Self> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        Ok(Self::new(data))
    }

    /// Ensure the header has been read and validated.
    fn ensure_header(&mut self) -> SerializeResult<()> {
        if self.header_read {
            return Ok(());
        }
        if self.data.len() < 8 {
            return Err(SerializeError::UnexpectedEof);
        }
        let magic = u32::from_le_bytes([
            self.data[0],
            self.data[1],
            self.data[2],
            self.data[3],
        ]);
        if magic != BINARY_MAGIC {
            return Err(SerializeError::Format(format!(
                "Invalid magic number: 0x{:08X}",
                magic
            )));
        }
        self.version = u32::from_le_bytes([
            self.data[4],
            self.data[5],
            self.data[6],
            self.data[7],
        ]);
        self.pos = 8;
        self.header_read = true;
        Ok(())
    }

    /// Read `n` bytes, advancing the position.
    fn read_n(&mut self, n: usize) -> SerializeResult<&[u8]> {
        self.ensure_header()?;
        if self.pos + n > self.data.len() {
            return Err(SerializeError::UnexpectedEof);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Get the data format version.
    pub fn version(&mut self) -> SerializeResult<u32> {
        self.ensure_header()?;
        Ok(self.version)
    }

    /// Get the current read position.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Check if we've reached the end of the data.
    pub fn is_at_end(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Remaining bytes to read.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

impl SerializeReader for BinaryReader {
    fn read_bool(&mut self) -> SerializeResult<bool> {
        let b = self.read_n(1)?;
        Ok(b[0] != 0)
    }

    fn read_u8(&mut self) -> SerializeResult<u8> {
        let b = self.read_n(1)?;
        Ok(b[0])
    }

    fn read_u16(&mut self) -> SerializeResult<u16> {
        let b = self.read_n(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> SerializeResult<u32> {
        let b = self.read_n(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> SerializeResult<u64> {
        let b = self.read_n(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i32(&mut self) -> SerializeResult<i32> {
        let b = self.read_n(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i64(&mut self) -> SerializeResult<i64> {
        let b = self.read_n(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f32(&mut self) -> SerializeResult<f32> {
        let b = self.read_n(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> SerializeResult<f64> {
        let b = self.read_n(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_string(&mut self) -> SerializeResult<String> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_n(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| SerializeError::Format(format!("Invalid UTF-8: {}", e)))
    }

    fn read_bytes(&mut self) -> SerializeResult<Vec<u8>> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_n(len)?;
        Ok(bytes.to_vec())
    }

    fn begin_struct(&mut self) -> SerializeResult<String> {
        self.read_string()
    }

    fn end_struct(&mut self) -> SerializeResult<()> {
        Ok(())
    }

    fn begin_array(&mut self) -> SerializeResult<usize> {
        Ok(self.read_u32()? as usize)
    }

    fn end_array(&mut self) -> SerializeResult<()> {
        Ok(())
    }

    fn begin_field(&mut self) -> SerializeResult<String> {
        self.read_string()
    }

    fn end_field(&mut self) -> SerializeResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JsonWriter
// ---------------------------------------------------------------------------

/// JSON text format serialization writer.
///
/// Produces a JSON structure using serde_json. The output can be
/// pretty-printed or compact.
pub struct JsonWriter {
    /// Stack of serde_json::Value being built.
    stack: Vec<JsonStackFrame>,
    /// Whether to pretty-print output.
    pretty: bool,
}

enum JsonStackFrame {
    Object {
        name: String,
        fields: Vec<(String, serde_json::Value)>,
        current_field: Option<String>,
    },
    Array {
        items: Vec<serde_json::Value>,
    },
}

impl JsonWriter {
    /// Create a new JSON writer (pretty-printed by default).
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            pretty: true,
        }
    }

    /// Create a compact (non-pretty-printed) JSON writer.
    pub fn compact() -> Self {
        Self {
            stack: Vec::new(),
            pretty: false,
        }
    }

    /// Convert the accumulated data to a JSON string.
    pub fn to_string(&self) -> SerializeResult<String> {
        if self.stack.len() != 1 {
            return Err(SerializeError::Format(
                "JSON writer has unfinished structures".into(),
            ));
        }
        let value = self.top_value()?;
        if self.pretty {
            Ok(serde_json::to_string_pretty(&value)
                .map_err(|e| SerializeError::Json(e.to_string()))?)
        } else {
            Ok(serde_json::to_string(&value)
                .map_err(|e| SerializeError::Json(e.to_string()))?)
        }
    }

    fn top_value(&self) -> SerializeResult<serde_json::Value> {
        match self.stack.last() {
            Some(JsonStackFrame::Object { fields, .. }) => {
                let mut map = serde_json::Map::new();
                for (k, v) in fields {
                    map.insert(k.clone(), v.clone());
                }
                Ok(serde_json::Value::Object(map))
            }
            Some(JsonStackFrame::Array { items }) => {
                Ok(serde_json::Value::Array(items.clone()))
            }
            None => Err(SerializeError::Format("No data written".into())),
        }
    }

    fn push_value(&mut self, value: serde_json::Value) -> SerializeResult<()> {
        match self.stack.last_mut() {
            Some(JsonStackFrame::Object {
                fields,
                current_field,
                ..
            }) => {
                if let Some(field_name) = current_field.take() {
                    fields.push((field_name, value));
                } else {
                    // Value directly on struct (e.g. array inside struct)
                    fields.push(("_value".into(), value));
                }
            }
            Some(JsonStackFrame::Array { items }) => {
                items.push(value);
            }
            None => {
                // Root-level value: push a wrapper.
                self.stack.push(JsonStackFrame::Object {
                    name: "root".into(),
                    fields: vec![("_root".into(), value)],
                    current_field: None,
                });
            }
        }
        Ok(())
    }
}

impl Default for JsonWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl SerializeWriter for JsonWriter {
    fn write_bool(&mut self, value: bool) -> SerializeResult<()> {
        self.push_value(serde_json::Value::Bool(value))
    }

    fn write_u8(&mut self, value: u8) -> SerializeResult<()> {
        self.push_value(serde_json::json!(value))
    }

    fn write_u16(&mut self, value: u16) -> SerializeResult<()> {
        self.push_value(serde_json::json!(value))
    }

    fn write_u32(&mut self, value: u32) -> SerializeResult<()> {
        self.push_value(serde_json::json!(value))
    }

    fn write_u64(&mut self, value: u64) -> SerializeResult<()> {
        self.push_value(serde_json::json!(value))
    }

    fn write_i32(&mut self, value: i32) -> SerializeResult<()> {
        self.push_value(serde_json::json!(value))
    }

    fn write_i64(&mut self, value: i64) -> SerializeResult<()> {
        self.push_value(serde_json::json!(value))
    }

    fn write_f32(&mut self, value: f32) -> SerializeResult<()> {
        self.push_value(serde_json::json!(value))
    }

    fn write_f64(&mut self, value: f64) -> SerializeResult<()> {
        self.push_value(serde_json::json!(value))
    }

    fn write_string(&mut self, value: &str) -> SerializeResult<()> {
        self.push_value(serde_json::Value::String(value.to_string()))
    }

    fn write_bytes(&mut self, value: &[u8]) -> SerializeResult<()> {
        // Encode bytes as base64 string.
        use serde_json::Value;
        let encoded: Vec<Value> = value.iter().map(|b| Value::from(*b)).collect();
        self.push_value(Value::Array(encoded))
    }

    fn begin_struct(&mut self, name: &str) -> SerializeResult<()> {
        self.stack.push(JsonStackFrame::Object {
            name: name.to_string(),
            fields: Vec::new(),
            current_field: None,
        });
        Ok(())
    }

    fn end_struct(&mut self) -> SerializeResult<()> {
        let frame = self.stack.pop().ok_or_else(|| {
            SerializeError::Format("end_struct without matching begin".into())
        })?;
        match frame {
            JsonStackFrame::Object { fields, name, .. } => {
                let mut map = serde_json::Map::new();
                map.insert("_type".into(), serde_json::Value::String(name));
                for (k, v) in fields {
                    map.insert(k, v);
                }
                let value = serde_json::Value::Object(map);
                self.push_value(value)
            }
            _ => Err(SerializeError::Format(
                "end_struct called on non-struct frame".into(),
            )),
        }
    }

    fn begin_array(&mut self, _len: usize) -> SerializeResult<()> {
        self.stack.push(JsonStackFrame::Array {
            items: Vec::new(),
        });
        Ok(())
    }

    fn end_array(&mut self) -> SerializeResult<()> {
        let frame = self.stack.pop().ok_or_else(|| {
            SerializeError::Format("end_array without matching begin".into())
        })?;
        match frame {
            JsonStackFrame::Array { items } => {
                self.push_value(serde_json::Value::Array(items))
            }
            _ => Err(SerializeError::Format(
                "end_array called on non-array frame".into(),
            )),
        }
    }

    fn begin_field(&mut self, name: &str) -> SerializeResult<()> {
        if let Some(JsonStackFrame::Object {
            current_field, ..
        }) = self.stack.last_mut()
        {
            *current_field = Some(name.to_string());
            Ok(())
        } else {
            Err(SerializeError::Format(
                "begin_field called outside of struct".into(),
            ))
        }
    }

    fn end_field(&mut self) -> SerializeResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JsonReader
// ---------------------------------------------------------------------------

/// JSON text format deserialization reader.
pub struct JsonReader {
    /// Stack of values being read.
    stack: Vec<JsonReadFrame>,
}

enum JsonReadFrame {
    Object {
        map: serde_json::Map<String, serde_json::Value>,
        field_iter: Vec<(String, serde_json::Value)>,
        current_field: Option<String>,
    },
    Array {
        items: Vec<serde_json::Value>,
        index: usize,
    },
    Value(serde_json::Value),
}

impl JsonReader {
    /// Create a reader from a JSON string.
    pub fn from_str(json: &str) -> SerializeResult<Self> {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| SerializeError::Json(e.to_string()))?;
        Ok(Self {
            stack: vec![JsonReadFrame::Value(value)],
        })
    }

    fn current_value(&mut self) -> SerializeResult<serde_json::Value> {
        match self.stack.last_mut() {
            Some(JsonReadFrame::Value(v)) => Ok(v.clone()),
            Some(JsonReadFrame::Object {
                current_field,
                map,
                ..
            }) => {
                if let Some(field_name) = current_field.as_ref() {
                    map.get(field_name)
                        .cloned()
                        .ok_or_else(|| SerializeError::MissingField(field_name.clone()))
                } else {
                    Err(SerializeError::Format(
                        "No field selected in object".into(),
                    ))
                }
            }
            Some(JsonReadFrame::Array { items, index }) => {
                if *index < items.len() {
                    let val = items[*index].clone();
                    *index += 1;
                    Ok(val)
                } else {
                    Err(SerializeError::UnexpectedEof)
                }
            }
            None => Err(SerializeError::UnexpectedEof),
        }
    }
}

impl SerializeReader for JsonReader {
    fn read_bool(&mut self) -> SerializeResult<bool> {
        let v = self.current_value()?;
        v.as_bool()
            .ok_or_else(|| SerializeError::TypeMismatch {
                expected: "bool".into(),
                actual: format!("{}", v),
            })
    }

    fn read_u8(&mut self) -> SerializeResult<u8> {
        let v = self.current_value()?;
        v.as_u64()
            .map(|n| n as u8)
            .ok_or_else(|| SerializeError::TypeMismatch {
                expected: "u8".into(),
                actual: format!("{}", v),
            })
    }

    fn read_u16(&mut self) -> SerializeResult<u16> {
        let v = self.current_value()?;
        v.as_u64()
            .map(|n| n as u16)
            .ok_or_else(|| SerializeError::TypeMismatch {
                expected: "u16".into(),
                actual: format!("{}", v),
            })
    }

    fn read_u32(&mut self) -> SerializeResult<u32> {
        let v = self.current_value()?;
        v.as_u64()
            .map(|n| n as u32)
            .ok_or_else(|| SerializeError::TypeMismatch {
                expected: "u32".into(),
                actual: format!("{}", v),
            })
    }

    fn read_u64(&mut self) -> SerializeResult<u64> {
        let v = self.current_value()?;
        v.as_u64()
            .ok_or_else(|| SerializeError::TypeMismatch {
                expected: "u64".into(),
                actual: format!("{}", v),
            })
    }

    fn read_i32(&mut self) -> SerializeResult<i32> {
        let v = self.current_value()?;
        v.as_i64()
            .map(|n| n as i32)
            .ok_or_else(|| SerializeError::TypeMismatch {
                expected: "i32".into(),
                actual: format!("{}", v),
            })
    }

    fn read_i64(&mut self) -> SerializeResult<i64> {
        let v = self.current_value()?;
        v.as_i64()
            .ok_or_else(|| SerializeError::TypeMismatch {
                expected: "i64".into(),
                actual: format!("{}", v),
            })
    }

    fn read_f32(&mut self) -> SerializeResult<f32> {
        let v = self.current_value()?;
        v.as_f64()
            .map(|n| n as f32)
            .ok_or_else(|| SerializeError::TypeMismatch {
                expected: "f32".into(),
                actual: format!("{}", v),
            })
    }

    fn read_f64(&mut self) -> SerializeResult<f64> {
        let v = self.current_value()?;
        v.as_f64()
            .ok_or_else(|| SerializeError::TypeMismatch {
                expected: "f64".into(),
                actual: format!("{}", v),
            })
    }

    fn read_string(&mut self) -> SerializeResult<String> {
        let v = self.current_value()?;
        v.as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| SerializeError::TypeMismatch {
                expected: "string".into(),
                actual: format!("{}", v),
            })
    }

    fn read_bytes(&mut self) -> SerializeResult<Vec<u8>> {
        let v = self.current_value()?;
        match v.as_array() {
            Some(arr) => {
                let mut bytes = Vec::with_capacity(arr.len());
                for item in arr {
                    bytes.push(
                        item.as_u64()
                            .map(|n| n as u8)
                            .ok_or_else(|| {
                                SerializeError::TypeMismatch {
                                    expected: "byte".into(),
                                    actual: format!("{}", item),
                                }
                            })?,
                    );
                }
                Ok(bytes)
            }
            None => Err(SerializeError::TypeMismatch {
                expected: "byte array".into(),
                actual: format!("{}", v),
            }),
        }
    }

    fn begin_struct(&mut self) -> SerializeResult<String> {
        let v = self.current_value()?;
        match v {
            serde_json::Value::Object(map) => {
                let name = map
                    .get("_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let field_iter: Vec<(String, serde_json::Value)> = map
                    .iter()
                    .filter(|(k, _)| k.as_str() != "_type")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                self.stack.push(JsonReadFrame::Object {
                    map,
                    field_iter,
                    current_field: None,
                });
                Ok(name)
            }
            _ => Err(SerializeError::TypeMismatch {
                expected: "object".into(),
                actual: format!("{}", v),
            }),
        }
    }

    fn end_struct(&mut self) -> SerializeResult<()> {
        match self.stack.pop() {
            Some(JsonReadFrame::Object { .. }) => Ok(()),
            _ => Err(SerializeError::Format(
                "end_struct without matching begin".into(),
            )),
        }
    }

    fn begin_array(&mut self) -> SerializeResult<usize> {
        let v = self.current_value()?;
        match v {
            serde_json::Value::Array(items) => {
                let len = items.len();
                self.stack.push(JsonReadFrame::Array { items, index: 0 });
                Ok(len)
            }
            _ => Err(SerializeError::TypeMismatch {
                expected: "array".into(),
                actual: format!("{}", v),
            }),
        }
    }

    fn end_array(&mut self) -> SerializeResult<()> {
        match self.stack.pop() {
            Some(JsonReadFrame::Array { .. }) => Ok(()),
            _ => Err(SerializeError::Format(
                "end_array without matching begin".into(),
            )),
        }
    }

    fn begin_field(&mut self) -> SerializeResult<String> {
        match self.stack.last_mut() {
            Some(JsonReadFrame::Object {
                field_iter,
                current_field,
                ..
            }) => {
                if let Some((name, _)) = field_iter.first() {
                    let name = name.clone();
                    *current_field = Some(name.clone());
                    field_iter.remove(0);
                    Ok(name)
                } else {
                    Err(SerializeError::UnexpectedEof)
                }
            }
            _ => Err(SerializeError::Format(
                "begin_field called outside object".into(),
            )),
        }
    }

    fn end_field(&mut self) -> SerializeResult<()> {
        if let Some(JsonReadFrame::Object {
            current_field, ..
        }) = self.stack.last_mut()
        {
            *current_field = None;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RonWriter / RonReader (using ron crate via serde)
// ---------------------------------------------------------------------------

/// RON (Rusty Object Notation) serialization writer.
///
/// Wraps `ron::ser` via serde for convenience.
pub struct RonWriter {
    /// Accumulated serde_json::Value that will be converted to RON at the end.
    inner: JsonWriter,
}

impl RonWriter {
    /// Create a new RON writer.
    pub fn new() -> Self {
        Self {
            inner: JsonWriter::new(),
        }
    }

    /// Convert the accumulated data to a RON string.
    pub fn to_string(&self) -> SerializeResult<String> {
        let json_str = self.inner.to_string()?;
        let value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| SerializeError::Json(e.to_string()))?;
        ron::ser::to_string_pretty(&value, ron::ser::PrettyConfig::default())
            .map_err(|e| SerializeError::Custom(format!("RON error: {}", e)))
    }
}

impl Default for RonWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl SerializeWriter for RonWriter {
    fn write_bool(&mut self, value: bool) -> SerializeResult<()> {
        self.inner.write_bool(value)
    }
    fn write_u8(&mut self, value: u8) -> SerializeResult<()> {
        self.inner.write_u8(value)
    }
    fn write_u16(&mut self, value: u16) -> SerializeResult<()> {
        self.inner.write_u16(value)
    }
    fn write_u32(&mut self, value: u32) -> SerializeResult<()> {
        self.inner.write_u32(value)
    }
    fn write_u64(&mut self, value: u64) -> SerializeResult<()> {
        self.inner.write_u64(value)
    }
    fn write_i32(&mut self, value: i32) -> SerializeResult<()> {
        self.inner.write_i32(value)
    }
    fn write_i64(&mut self, value: i64) -> SerializeResult<()> {
        self.inner.write_i64(value)
    }
    fn write_f32(&mut self, value: f32) -> SerializeResult<()> {
        self.inner.write_f32(value)
    }
    fn write_f64(&mut self, value: f64) -> SerializeResult<()> {
        self.inner.write_f64(value)
    }
    fn write_string(&mut self, value: &str) -> SerializeResult<()> {
        self.inner.write_string(value)
    }
    fn write_bytes(&mut self, value: &[u8]) -> SerializeResult<()> {
        self.inner.write_bytes(value)
    }
    fn begin_struct(&mut self, name: &str) -> SerializeResult<()> {
        self.inner.begin_struct(name)
    }
    fn end_struct(&mut self) -> SerializeResult<()> {
        self.inner.end_struct()
    }
    fn begin_array(&mut self, len: usize) -> SerializeResult<()> {
        self.inner.begin_array(len)
    }
    fn end_array(&mut self) -> SerializeResult<()> {
        self.inner.end_array()
    }
    fn begin_field(&mut self, name: &str) -> SerializeResult<()> {
        self.inner.begin_field(name)
    }
    fn end_field(&mut self) -> SerializeResult<()> {
        self.inner.end_field()
    }
}

/// RON (Rusty Object Notation) deserialization reader.
///
/// Wraps `ron::de` via serde for convenience. RON data is first parsed to
/// a serde_json::Value and then read through [`JsonReader`].
pub struct RonReader {
    inner: JsonReader,
}

impl RonReader {
    /// Create a reader from a RON string.
    pub fn from_str(ron_text: &str) -> SerializeResult<Self> {
        let value: serde_json::Value =
            ron::de::from_str(ron_text)
                .map_err(|e| SerializeError::Custom(format!("RON parse error: {}", e)))?;
        let json_str = serde_json::to_string(&value)
            .map_err(|e| SerializeError::Json(e.to_string()))?;
        Ok(Self {
            inner: JsonReader::from_str(&json_str)?,
        })
    }
}

impl SerializeReader for RonReader {
    fn read_bool(&mut self) -> SerializeResult<bool> {
        self.inner.read_bool()
    }
    fn read_u8(&mut self) -> SerializeResult<u8> {
        self.inner.read_u8()
    }
    fn read_u16(&mut self) -> SerializeResult<u16> {
        self.inner.read_u16()
    }
    fn read_u32(&mut self) -> SerializeResult<u32> {
        self.inner.read_u32()
    }
    fn read_u64(&mut self) -> SerializeResult<u64> {
        self.inner.read_u64()
    }
    fn read_i32(&mut self) -> SerializeResult<i32> {
        self.inner.read_i32()
    }
    fn read_i64(&mut self) -> SerializeResult<i64> {
        self.inner.read_i64()
    }
    fn read_f32(&mut self) -> SerializeResult<f32> {
        self.inner.read_f32()
    }
    fn read_f64(&mut self) -> SerializeResult<f64> {
        self.inner.read_f64()
    }
    fn read_string(&mut self) -> SerializeResult<String> {
        self.inner.read_string()
    }
    fn read_bytes(&mut self) -> SerializeResult<Vec<u8>> {
        self.inner.read_bytes()
    }
    fn begin_struct(&mut self) -> SerializeResult<String> {
        self.inner.begin_struct()
    }
    fn end_struct(&mut self) -> SerializeResult<()> {
        self.inner.end_struct()
    }
    fn begin_array(&mut self) -> SerializeResult<usize> {
        self.inner.begin_array()
    }
    fn end_array(&mut self) -> SerializeResult<()> {
        self.inner.end_array()
    }
    fn begin_field(&mut self) -> SerializeResult<String> {
        self.inner.begin_field()
    }
    fn end_field(&mut self) -> SerializeResult<()> {
        self.inner.end_field()
    }
}

// ---------------------------------------------------------------------------
// VersionedData
// ---------------------------------------------------------------------------

/// Wrapper that associates a version number with serialized data for
/// forward/backward compatibility.
///
/// When reading versioned data, the [`MigrationRegistry`] can automatically
/// upgrade older versions to the current version.
#[derive(Debug, Clone)]
pub struct VersionedData {
    /// Version number of the serialized data.
    pub version: u32,
    /// The serialized payload (format-specific).
    pub data: Vec<u8>,
}

impl VersionedData {
    /// Create a new versioned data wrapper.
    pub fn new(version: u32, data: Vec<u8>) -> Self {
        Self { version, data }
    }

    /// Serialize to bytes (includes version header).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + self.data.len());
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> SerializeResult<Self> {
        if bytes.len() < 4 {
            return Err(SerializeError::UnexpectedEof);
        }
        let version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let data = bytes[4..].to_vec();
        Ok(Self { version, data })
    }
}

impl fmt::Display for VersionedData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VersionedData(v{}, {} bytes)",
            self.version,
            self.data.len()
        )
    }
}

// ---------------------------------------------------------------------------
// MigrationRegistry
// ---------------------------------------------------------------------------

/// Type alias for a migration function.
type MigrationFn = Box<dyn Fn(&[u8]) -> SerializeResult<Vec<u8>> + Send + Sync>;

/// Registry of migration functions for upgrading versioned data.
///
/// Migration functions transform data from version N to version N+1.
/// When loading data at version V, the registry applies all migrations
/// from V to the current version sequentially.
pub struct MigrationRegistry {
    /// Type name this registry handles.
    type_name: String,
    /// Current version.
    current_version: u32,
    /// Migration functions indexed by source version.
    /// `migrations[v]` upgrades data from version v to version v+1.
    migrations: HashMap<u32, MigrationFn>,
}

impl MigrationRegistry {
    /// Create a new migration registry.
    pub fn new(type_name: impl Into<String>, current_version: u32) -> Self {
        Self {
            type_name: type_name.into(),
            current_version,
            migrations: HashMap::new(),
        }
    }

    /// Register a migration from `from_version` to `from_version + 1`.
    pub fn register_migration<F>(&mut self, from_version: u32, migration: F)
    where
        F: Fn(&[u8]) -> SerializeResult<Vec<u8>> + Send + Sync + 'static,
    {
        self.migrations
            .insert(from_version, Box::new(migration));
    }

    /// Migrate versioned data to the current version.
    ///
    /// Applies all registered migrations sequentially from `data.version`
    /// to `current_version`.
    pub fn migrate(&self, data: &VersionedData) -> SerializeResult<VersionedData> {
        if data.version == self.current_version {
            return Ok(data.clone());
        }

        if data.version > self.current_version {
            return Err(SerializeError::VersionMismatch {
                expected: self.current_version,
                actual: data.version,
            });
        }

        let mut current_data = data.data.clone();
        let mut version = data.version;

        while version < self.current_version {
            let migration = self.migrations.get(&version).ok_or_else(|| {
                SerializeError::Custom(format!(
                    "No migration registered for {} v{} -> v{}",
                    self.type_name,
                    version,
                    version + 1,
                ))
            })?;
            current_data = migration(&current_data)?;
            version += 1;
        }

        Ok(VersionedData {
            version: self.current_version,
            data: current_data,
        })
    }

    /// Get the current version.
    pub fn current_version(&self) -> u32 {
        self.current_version
    }

    /// Get the type name.
    pub fn type_name(&self) -> &str {
        &self.type_name
    }

    /// Check if a migration path exists from the given version to current.
    pub fn can_migrate_from(&self, version: u32) -> bool {
        if version >= self.current_version {
            return version == self.current_version;
        }
        let mut v = version;
        while v < self.current_version {
            if !self.migrations.contains_key(&v) {
                return false;
            }
            v += 1;
        }
        true
    }
}

impl fmt::Debug for MigrationRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MigrationRegistry")
            .field("type_name", &self.type_name)
            .field("current_version", &self.current_version)
            .field("migration_count", &self.migrations.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_round_trip_primitives() {
        let mut writer = BinaryWriter::new();
        writer.write_bool(true).unwrap();
        writer.write_u8(42).unwrap();
        writer.write_u16(1234).unwrap();
        writer.write_u32(99999).unwrap();
        writer.write_u64(123456789).unwrap();
        writer.write_i32(-42).unwrap();
        writer.write_i64(-99999).unwrap();
        writer.write_f32(3.14).unwrap();
        writer.write_f64(2.71828).unwrap();
        writer.write_string("hello").unwrap();

        let bytes = writer.into_bytes();
        let mut reader = BinaryReader::new(bytes);

        assert!(reader.read_bool().unwrap());
        assert_eq!(reader.read_u8().unwrap(), 42);
        assert_eq!(reader.read_u16().unwrap(), 1234);
        assert_eq!(reader.read_u32().unwrap(), 99999);
        assert_eq!(reader.read_u64().unwrap(), 123456789);
        assert_eq!(reader.read_i32().unwrap(), -42);
        assert_eq!(reader.read_i64().unwrap(), -99999);
        assert!((reader.read_f32().unwrap() - 3.14).abs() < 1e-5);
        assert!((reader.read_f64().unwrap() - 2.71828).abs() < 1e-10);
        assert_eq!(reader.read_string().unwrap(), "hello");
    }

    #[test]
    fn binary_version_header() {
        let mut writer = BinaryWriter::new();
        writer.write_u8(0).unwrap();
        let bytes = writer.into_bytes();

        let mut reader = BinaryReader::new(bytes);
        assert_eq!(reader.version().unwrap(), BINARY_VERSION);
    }

    #[test]
    fn binary_invalid_magic() {
        let data = vec![0, 0, 0, 0, 1, 0, 0, 0];
        let mut reader = BinaryReader::new(data);
        assert!(reader.read_bool().is_err());
    }

    #[test]
    fn binary_round_trip_vec3() {
        let original = Vec3::new(1.0, 2.0, 3.0);
        let mut writer = BinaryWriter::new();
        original.serialize(&mut writer).unwrap();

        let bytes = writer.into_bytes();
        let mut reader = BinaryReader::new(bytes);
        let restored = Vec3::deserialize(&mut reader).unwrap();

        assert_eq!(original, restored);
    }

    #[test]
    fn binary_round_trip_quat() {
        let original = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let mut writer = BinaryWriter::new();
        original.serialize(&mut writer).unwrap();

        let bytes = writer.into_bytes();
        let mut reader = BinaryReader::new(bytes);
        let restored = Quat::deserialize(&mut reader).unwrap();

        assert!((original.x - restored.x).abs() < 1e-5);
        assert!((original.y - restored.y).abs() < 1e-5);
        assert!((original.z - restored.z).abs() < 1e-5);
        assert!((original.w - restored.w).abs() < 1e-5);
    }

    #[test]
    fn binary_round_trip_transform() {
        let original = crate::math::Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::IDENTITY,
            Vec3::new(1.0, 1.0, 1.0),
        );
        let mut writer = BinaryWriter::new();
        original.serialize(&mut writer).unwrap();

        let bytes = writer.into_bytes();
        let mut reader = BinaryReader::new(bytes);
        let restored = crate::math::Transform::deserialize(&mut reader).unwrap();

        assert_eq!(original.position, restored.position);
        assert_eq!(original.scale, restored.scale);
    }

    #[test]
    fn binary_round_trip_mat4() {
        let original = Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::IDENTITY,
            Vec3::new(5.0, 10.0, 15.0),
        );
        let mut writer = BinaryWriter::new();
        original.serialize(&mut writer).unwrap();

        let bytes = writer.into_bytes();
        let mut reader = BinaryReader::new(bytes);
        let restored = Mat4::deserialize(&mut reader).unwrap();

        let orig_cols = original.to_cols_array();
        let rest_cols = restored.to_cols_array();
        for i in 0..16 {
            assert!((orig_cols[i] - rest_cols[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn binary_round_trip_vec() {
        let original = vec![1u32, 2, 3, 4, 5];
        let mut writer = BinaryWriter::new();
        original.serialize(&mut writer).unwrap();

        let bytes = writer.into_bytes();
        let mut reader = BinaryReader::new(bytes);
        let restored = Vec::<u32>::deserialize(&mut reader).unwrap();

        assert_eq!(original, restored);
    }

    #[test]
    fn binary_round_trip_option() {
        let some_val: Option<u32> = Some(42);
        let none_val: Option<u32> = None;

        let mut writer = BinaryWriter::new();
        some_val.serialize(&mut writer).unwrap();
        none_val.serialize(&mut writer).unwrap();

        let bytes = writer.into_bytes();
        let mut reader = BinaryReader::new(bytes);
        let restored_some = Option::<u32>::deserialize(&mut reader).unwrap();
        let restored_none = Option::<u32>::deserialize(&mut reader).unwrap();

        assert_eq!(restored_some, Some(42));
        assert_eq!(restored_none, None);
    }

    #[test]
    fn versioned_data_round_trip() {
        let data = VersionedData::new(3, vec![1, 2, 3, 4]);
        let bytes = data.to_bytes();
        let restored = VersionedData::from_bytes(&bytes).unwrap();

        assert_eq!(restored.version, 3);
        assert_eq!(restored.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn migration_registry_sequential() {
        let mut registry = MigrationRegistry::new("TestType", 3);

        // v1 -> v2: append a byte.
        registry.register_migration(1, |data| {
            let mut new_data = data.to_vec();
            new_data.push(0xAA);
            Ok(new_data)
        });

        // v2 -> v3: append another byte.
        registry.register_migration(2, |data| {
            let mut new_data = data.to_vec();
            new_data.push(0xBB);
            Ok(new_data)
        });

        let v1_data = VersionedData::new(1, vec![0x01]);
        let result = registry.migrate(&v1_data).unwrap();
        assert_eq!(result.version, 3);
        assert_eq!(result.data, vec![0x01, 0xAA, 0xBB]);
    }

    #[test]
    fn migration_registry_already_current() {
        let registry = MigrationRegistry::new("TestType", 1);
        let data = VersionedData::new(1, vec![1, 2, 3]);
        let result = registry.migrate(&data).unwrap();
        assert_eq!(result.version, 1);
        assert_eq!(result.data, vec![1, 2, 3]);
    }

    #[test]
    fn migration_registry_future_version_error() {
        let registry = MigrationRegistry::new("TestType", 2);
        let data = VersionedData::new(5, vec![]);
        assert!(registry.migrate(&data).is_err());
    }

    #[test]
    fn migration_can_migrate_from() {
        let mut registry = MigrationRegistry::new("Test", 3);
        registry.register_migration(1, |data| Ok(data.to_vec()));
        registry.register_migration(2, |data| Ok(data.to_vec()));

        assert!(registry.can_migrate_from(1));
        assert!(registry.can_migrate_from(2));
        assert!(registry.can_migrate_from(3));
        assert!(!registry.can_migrate_from(0));
        assert!(!registry.can_migrate_from(4));
    }

    #[test]
    fn binary_bytes_round_trip() {
        let original_bytes = vec![0u8, 1, 2, 3, 255, 128];
        let mut writer = BinaryWriter::new();
        writer.write_bytes(&original_bytes).unwrap();

        let data = writer.into_bytes();
        let mut reader = BinaryReader::new(data);
        let restored = reader.read_bytes().unwrap();

        assert_eq!(original_bytes, restored);
    }
}
