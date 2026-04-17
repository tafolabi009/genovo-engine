// engine/networking/src/packet_serializer.rs
//
// Efficient packet serialization for the Genovo engine.
//
// Provides compact binary serialization for network packets:
//
// - **Bit-level packing** -- Pack values at the bit level for minimal overhead.
// - **Variable-length integers** -- Encode small values in fewer bytes.
// - **Quantized floats** -- Compress floats to reduced precision.
// - **String table** -- Send string indices instead of full strings.
// - **Schema versioning** -- Version packet schemas for backward compatibility.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_PACKET_SIZE: usize = 1400; // MTU-safe
const STRING_TABLE_MAGIC: u16 = 0x5354; // "ST"
const SCHEMA_VERSION_CURRENT: u16 = 1;

// ---------------------------------------------------------------------------
// Bit writer
// ---------------------------------------------------------------------------

/// Writes data at the bit level into a byte buffer.
pub struct BitWriter {
    buffer: Vec<u8>,
    bit_position: usize,
}

impl BitWriter {
    /// Create a new bit writer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            bit_position: 0,
        }
    }

    /// Write a boolean (1 bit).
    pub fn write_bool(&mut self, value: bool) {
        self.write_bits(if value { 1 } else { 0 }, 1);
    }

    /// Write N bits from a u32 value.
    pub fn write_bits(&mut self, value: u32, bit_count: u32) {
        assert!(bit_count <= 32);
        for i in 0..bit_count {
            let byte_idx = self.bit_position / 8;
            let bit_idx = self.bit_position % 8;

            if byte_idx >= self.buffer.len() {
                self.buffer.push(0);
            }

            let bit = (value >> i) & 1;
            self.buffer[byte_idx] |= (bit as u8) << bit_idx;
            self.bit_position += 1;
        }
    }

    /// Write a u8.
    pub fn write_u8(&mut self, value: u8) {
        self.write_bits(value as u32, 8);
    }

    /// Write a u16.
    pub fn write_u16(&mut self, value: u16) {
        self.write_bits(value as u32, 16);
    }

    /// Write a u32.
    pub fn write_u32(&mut self, value: u32) {
        self.write_bits(value, 32);
    }

    /// Write a variable-length integer (1-5 bytes).
    pub fn write_varint(&mut self, mut value: u32) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;
            if value > 0 {
                byte |= 0x80;
            }
            self.write_u8(byte);
            if value == 0 {
                break;
            }
        }
    }

    /// Write a quantized float (reduced precision).
    /// Compresses a float in [min, max] to `bits` bits of precision.
    pub fn write_quantized_float(&mut self, value: f32, min: f32, max: f32, bits: u32) {
        let range = max - min;
        if range <= 0.0 {
            self.write_bits(0, bits);
            return;
        }
        let normalized = ((value - min) / range).clamp(0.0, 1.0);
        let max_val = (1u32 << bits) - 1;
        let quantized = (normalized * max_val as f32).round() as u32;
        self.write_bits(quantized.min(max_val), bits);
    }

    /// Write a full f32 (32 bits).
    pub fn write_f32(&mut self, value: f32) {
        self.write_u32(value.to_bits());
    }

    /// Write a string (length-prefixed).
    pub fn write_string(&mut self, s: &str) {
        let bytes = s.as_bytes();
        self.write_varint(bytes.len() as u32);
        for &b in bytes {
            self.write_u8(b);
        }
    }

    /// Write a string table index.
    pub fn write_string_index(&mut self, index: u16) {
        self.write_u16(index);
    }

    /// Get the number of bits written.
    pub fn bits_written(&self) -> usize {
        self.bit_position
    }

    /// Get the number of bytes used (rounded up).
    pub fn bytes_used(&self) -> usize {
        (self.bit_position + 7) / 8
    }

    /// Get the buffer as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer[..self.bytes_used()]
    }

    /// Consume and return the buffer.
    pub fn into_bytes(self) -> Vec<u8> {
        let used = self.bytes_used();
        let mut buf = self.buffer;
        buf.truncate(used);
        buf
    }

    /// Reset the writer.
    pub fn reset(&mut self) {
        self.buffer.fill(0);
        self.bit_position = 0;
    }
}

impl fmt::Debug for BitWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitWriter({} bits, {} bytes)", self.bit_position, self.bytes_used())
    }
}

// ---------------------------------------------------------------------------
// Bit reader
// ---------------------------------------------------------------------------

/// Reads data at the bit level from a byte buffer.
pub struct BitReader<'a> {
    buffer: &'a [u8],
    bit_position: usize,
    total_bits: usize,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader.
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            bit_position: 0,
            total_bits: buffer.len() * 8,
        }
    }

    /// Read a boolean (1 bit).
    pub fn read_bool(&mut self) -> Option<bool> {
        Some(self.read_bits(1)? != 0)
    }

    /// Read N bits as a u32.
    pub fn read_bits(&mut self, bit_count: u32) -> Option<u32> {
        if self.bit_position + bit_count as usize > self.total_bits {
            return None;
        }
        let mut value: u32 = 0;
        for i in 0..bit_count {
            let byte_idx = self.bit_position / 8;
            let bit_idx = self.bit_position % 8;
            let bit = ((self.buffer[byte_idx] >> bit_idx) & 1) as u32;
            value |= bit << i;
            self.bit_position += 1;
        }
        Some(value)
    }

    /// Read a u8.
    pub fn read_u8(&mut self) -> Option<u8> {
        Some(self.read_bits(8)? as u8)
    }

    /// Read a u16.
    pub fn read_u16(&mut self) -> Option<u16> {
        Some(self.read_bits(16)? as u16)
    }

    /// Read a u32.
    pub fn read_u32(&mut self) -> Option<u32> {
        self.read_bits(32)
    }

    /// Read a variable-length integer.
    pub fn read_varint(&mut self) -> Option<u32> {
        let mut value: u32 = 0;
        let mut shift: u32 = 0;
        loop {
            let byte = self.read_u8()?;
            value |= ((byte & 0x7F) as u32) << shift;
            shift += 7;
            if byte & 0x80 == 0 || shift >= 35 {
                break;
            }
        }
        Some(value)
    }

    /// Read a quantized float.
    pub fn read_quantized_float(&mut self, min: f32, max: f32, bits: u32) -> Option<f32> {
        let quantized = self.read_bits(bits)?;
        let max_val = (1u32 << bits) - 1;
        let normalized = quantized as f32 / max_val as f32;
        Some(min + normalized * (max - min))
    }

    /// Read a full f32.
    pub fn read_f32(&mut self) -> Option<f32> {
        Some(f32::from_bits(self.read_u32()?))
    }

    /// Read a string.
    pub fn read_string(&mut self) -> Option<String> {
        let len = self.read_varint()? as usize;
        let mut bytes = Vec::with_capacity(len);
        for _ in 0..len {
            bytes.push(self.read_u8()?);
        }
        String::from_utf8(bytes).ok()
    }

    /// Read a string table index.
    pub fn read_string_index(&mut self) -> Option<u16> {
        self.read_u16()
    }

    /// Bits remaining.
    pub fn bits_remaining(&self) -> usize {
        self.total_bits.saturating_sub(self.bit_position)
    }

    /// Current bit position.
    pub fn position(&self) -> usize {
        self.bit_position
    }
}

// ---------------------------------------------------------------------------
// String table
// ---------------------------------------------------------------------------

/// A string table for network string deduplication.
#[derive(Debug, Clone)]
pub struct StringTable {
    /// Index to string mapping.
    strings: Vec<String>,
    /// String to index mapping.
    lookup: HashMap<String, u16>,
    /// Whether the table has been modified since last sync.
    dirty: bool,
}

impl StringTable {
    /// Create a new empty string table.
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            lookup: HashMap::new(),
            dirty: false,
        }
    }

    /// Add a string and return its index.
    pub fn intern(&mut self, s: &str) -> u16 {
        if let Some(&idx) = self.lookup.get(s) {
            return idx;
        }
        let idx = self.strings.len() as u16;
        self.strings.push(s.to_string());
        self.lookup.insert(s.to_string(), idx);
        self.dirty = true;
        idx
    }

    /// Look up a string by index.
    pub fn get(&self, index: u16) -> Option<&str> {
        self.strings.get(index as usize).map(|s| s.as_str())
    }

    /// Number of strings in the table.
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Whether the table has been modified.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark as clean (after syncing).
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Serialize the table for network transmission.
    pub fn serialize(&self) -> Vec<u8> {
        let mut writer = BitWriter::new(4096);
        writer.write_u16(STRING_TABLE_MAGIC);
        writer.write_varint(self.strings.len() as u32);
        for s in &self.strings {
            writer.write_string(s);
        }
        writer.into_bytes()
    }

    /// Deserialize a string table.
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        let mut reader = BitReader::new(data);
        let magic = reader.read_u16()?;
        if magic != STRING_TABLE_MAGIC {
            return None;
        }
        let count = reader.read_varint()? as usize;
        let mut table = StringTable::new();
        for _ in 0..count {
            let s = reader.read_string()?;
            table.intern(&s);
        }
        Some(table)
    }

    /// Pre-populate with common strings.
    pub fn with_common_strings() -> Self {
        let mut table = Self::new();
        // Common game-related strings.
        for s in &[
            "position", "rotation", "velocity", "health", "damage",
            "player", "npc", "item", "weapon", "projectile",
            "spawn", "despawn", "hit", "miss", "death",
            "chat", "team", "score", "time", "round",
        ] {
            table.intern(s);
        }
        table.mark_clean();
        table
    }
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Schema version
// ---------------------------------------------------------------------------

/// Packet schema version for backward compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaVersion {
    pub version: u16,
    pub min_compatible: u16,
}

impl SchemaVersion {
    pub fn new(version: u16) -> Self {
        Self {
            version,
            min_compatible: version,
        }
    }

    pub fn with_min_compatible(mut self, min: u16) -> Self {
        self.min_compatible = min;
        self
    }

    pub fn is_compatible_with(&self, other: &SchemaVersion) -> bool {
        self.version >= other.min_compatible && other.version >= self.min_compatible
    }

    pub fn current() -> Self {
        Self::new(SCHEMA_VERSION_CURRENT)
    }
}

// ---------------------------------------------------------------------------
// Packet serializer
// ---------------------------------------------------------------------------

/// High-level packet serializer with string table and quantization support.
pub struct PacketSerializer {
    writer: BitWriter,
    string_table: StringTable,
    schema: SchemaVersion,
    stats: SerializerStats,
}

/// Statistics for the serializer.
#[derive(Debug, Clone, Default)]
pub struct SerializerStats {
    pub packets_written: u64,
    pub total_bits: u64,
    pub total_bytes: u64,
    pub strings_interned: u64,
    pub quantized_floats: u64,
    pub varints_written: u64,
    pub avg_packet_bits: f32,
}

impl PacketSerializer {
    /// Create a new packet serializer.
    pub fn new() -> Self {
        Self {
            writer: BitWriter::new(MAX_PACKET_SIZE),
            string_table: StringTable::with_common_strings(),
            schema: SchemaVersion::current(),
            stats: SerializerStats::default(),
        }
    }

    /// Create with a custom string table.
    pub fn with_string_table(table: StringTable) -> Self {
        Self {
            writer: BitWriter::new(MAX_PACKET_SIZE),
            string_table: table,
            schema: SchemaVersion::current(),
            stats: SerializerStats::default(),
        }
    }

    /// Begin a new packet.
    pub fn begin_packet(&mut self) {
        self.writer.reset();
        self.writer.write_u16(self.schema.version);
    }

    /// Write a boolean.
    pub fn write_bool(&mut self, value: bool) {
        self.writer.write_bool(value);
    }

    /// Write a u8.
    pub fn write_u8(&mut self, value: u8) {
        self.writer.write_u8(value);
    }

    /// Write a u16.
    pub fn write_u16(&mut self, value: u16) {
        self.writer.write_u16(value);
    }

    /// Write a u32.
    pub fn write_u32(&mut self, value: u32) {
        self.writer.write_u32(value);
    }

    /// Write a variable-length integer.
    pub fn write_varint(&mut self, value: u32) {
        self.writer.write_varint(value);
        self.stats.varints_written += 1;
    }

    /// Write a full-precision float.
    pub fn write_f32(&mut self, value: f32) {
        self.writer.write_f32(value);
    }

    /// Write a quantized float.
    pub fn write_quantized(&mut self, value: f32, min: f32, max: f32, bits: u32) {
        self.writer.write_quantized_float(value, min, max, bits);
        self.stats.quantized_floats += 1;
    }

    /// Write a position vector (quantized).
    pub fn write_position(&mut self, pos: [f32; 3], bounds: f32, bits: u32) {
        for &v in &pos {
            self.write_quantized(v, -bounds, bounds, bits);
        }
    }

    /// Write a normalized direction (quantized to unit sphere).
    pub fn write_direction(&mut self, dir: [f32; 3], bits: u32) {
        for &v in &dir {
            self.write_quantized(v, -1.0, 1.0, bits);
        }
    }

    /// Write a string using the string table.
    pub fn write_string_interned(&mut self, s: &str) {
        let idx = self.string_table.intern(s);
        self.writer.write_string_index(idx);
        self.stats.strings_interned += 1;
    }

    /// Write a raw string (no interning).
    pub fn write_string_raw(&mut self, s: &str) {
        self.writer.write_string(s);
    }

    /// Finish the packet and return the bytes.
    pub fn finish_packet(&mut self) -> Vec<u8> {
        let bits = self.writer.bits_written();
        self.stats.packets_written += 1;
        self.stats.total_bits += bits as u64;
        let bytes = self.writer.as_bytes().to_vec();
        self.stats.total_bytes += bytes.len() as u64;
        self.stats.avg_packet_bits = self.stats.total_bits as f32 / self.stats.packets_written as f32;
        bytes
    }

    /// Get the string table.
    pub fn string_table(&self) -> &StringTable {
        &self.string_table
    }

    /// Get mutable string table.
    pub fn string_table_mut(&mut self) -> &mut StringTable {
        &mut self.string_table
    }

    /// Get statistics.
    pub fn stats(&self) -> &SerializerStats {
        &self.stats
    }
}

impl Default for PacketSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level packet deserializer.
pub struct PacketDeserializer<'a> {
    reader: BitReader<'a>,
    string_table: &'a StringTable,
    schema: SchemaVersion,
}

impl<'a> PacketDeserializer<'a> {
    /// Create from packet bytes and a string table.
    pub fn new(data: &'a [u8], string_table: &'a StringTable) -> Option<Self> {
        let mut reader = BitReader::new(data);
        let version = reader.read_u16()?;
        Some(Self {
            reader,
            string_table,
            schema: SchemaVersion::new(version),
        })
    }

    pub fn schema_version(&self) -> u16 { self.schema.version }
    pub fn read_bool(&mut self) -> Option<bool> { self.reader.read_bool() }
    pub fn read_u8(&mut self) -> Option<u8> { self.reader.read_u8() }
    pub fn read_u16(&mut self) -> Option<u16> { self.reader.read_u16() }
    pub fn read_u32(&mut self) -> Option<u32> { self.reader.read_u32() }
    pub fn read_varint(&mut self) -> Option<u32> { self.reader.read_varint() }
    pub fn read_f32(&mut self) -> Option<f32> { self.reader.read_f32() }

    pub fn read_quantized(&mut self, min: f32, max: f32, bits: u32) -> Option<f32> {
        self.reader.read_quantized_float(min, max, bits)
    }

    pub fn read_position(&mut self, bounds: f32, bits: u32) -> Option<[f32; 3]> {
        let x = self.read_quantized(-bounds, bounds, bits)?;
        let y = self.read_quantized(-bounds, bounds, bits)?;
        let z = self.read_quantized(-bounds, bounds, bits)?;
        Some([x, y, z])
    }

    pub fn read_direction(&mut self, bits: u32) -> Option<[f32; 3]> {
        let x = self.read_quantized(-1.0, 1.0, bits)?;
        let y = self.read_quantized(-1.0, 1.0, bits)?;
        let z = self.read_quantized(-1.0, 1.0, bits)?;
        Some([x, y, z])
    }

    pub fn read_string_interned(&mut self) -> Option<&str> {
        let idx = self.reader.read_string_index()?;
        self.string_table.get(idx)
    }

    pub fn read_string_raw(&mut self) -> Option<String> {
        self.reader.read_string()
    }

    pub fn bits_remaining(&self) -> usize { self.reader.bits_remaining() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_roundtrip() {
        let mut writer = BitWriter::new(64);
        writer.write_bool(true);
        writer.write_bits(42, 7);
        writer.write_u8(255);
        writer.write_u16(1234);

        let bytes = writer.as_bytes();
        let mut reader = BitReader::new(bytes);
        assert_eq!(reader.read_bool(), Some(true));
        assert_eq!(reader.read_bits(7), Some(42));
        assert_eq!(reader.read_u8(), Some(255));
        assert_eq!(reader.read_u16(), Some(1234));
    }

    #[test]
    fn test_varint() {
        let mut writer = BitWriter::new(64);
        writer.write_varint(0);
        writer.write_varint(127);
        writer.write_varint(128);
        writer.write_varint(100000);

        let mut reader = BitReader::new(writer.as_bytes());
        assert_eq!(reader.read_varint(), Some(0));
        assert_eq!(reader.read_varint(), Some(127));
        assert_eq!(reader.read_varint(), Some(128));
        assert_eq!(reader.read_varint(), Some(100000));
    }

    #[test]
    fn test_quantized_float() {
        let mut writer = BitWriter::new(64);
        writer.write_quantized_float(0.5, 0.0, 1.0, 16);

        let mut reader = BitReader::new(writer.as_bytes());
        let v = reader.read_quantized_float(0.0, 1.0, 16).unwrap();
        assert!((v - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_string_table() {
        let mut table = StringTable::new();
        let idx1 = table.intern("hello");
        let idx2 = table.intern("world");
        let idx3 = table.intern("hello"); // Duplicate.
        assert_eq!(idx1, idx3);
        assert_ne!(idx1, idx2);
        assert_eq!(table.get(idx1), Some("hello"));
    }

    #[test]
    fn test_string_table_serialization() {
        let mut table = StringTable::new();
        table.intern("alpha");
        table.intern("beta");
        let bytes = table.serialize();
        let restored = StringTable::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.get(0), Some("alpha"));
        assert_eq!(restored.get(1), Some("beta"));
    }

    #[test]
    fn test_packet_serializer() {
        let mut ser = PacketSerializer::new();
        ser.begin_packet();
        ser.write_bool(true);
        ser.write_varint(42);
        ser.write_position([1.0, 2.0, 3.0], 1000.0, 16);
        ser.write_string_interned("health");
        let bytes = ser.finish_packet();

        let deser = PacketDeserializer::new(&bytes, ser.string_table()).unwrap();
        assert_eq!(deser.schema_version(), SCHEMA_VERSION_CURRENT);
    }

    #[test]
    fn test_schema_compatibility() {
        let v1 = SchemaVersion::new(1).with_min_compatible(1);
        let v2 = SchemaVersion::new(2).with_min_compatible(1);
        assert!(v1.is_compatible_with(&v2));
        assert!(v2.is_compatible_with(&v1));
    }
}
