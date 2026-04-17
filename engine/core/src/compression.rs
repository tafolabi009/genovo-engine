//! # Data Compression
//!
//! Compression algorithms for game asset pipelines, network packets, and
//! save-file storage. All implementations are self-contained with no external
//! dependencies.
//!
//! - [`LZ4Compressor`] — Fast LZ4-style byte-level compression
//! - [`RunLengthEncoder`] — Simple RLE for homogeneous data
//! - [`HuffmanCoder`] — Entropy coding with canonical Huffman trees
//! - [`DeltaEncoder`] — Delta encoding for time-series / animation data

use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::fmt;

// ---------------------------------------------------------------------------
// LZ4Compressor — fast LZ4-style compression
// ---------------------------------------------------------------------------

/// Fast LZ4-style compressor/decompressor.
///
/// The format uses a sequence of **tokens**, each consisting of:
///
/// 1. A **token byte**: high nibble = literal length, low nibble = match length (minus 4).
/// 2. Optional literal-length extension bytes (if literal length >= 15).
/// 3. Literal bytes.
/// 4. A 2-byte little-endian **match offset** (if match length > 0).
/// 5. Optional match-length extension bytes (if match length >= 19, i.e. nibble == 15).
///
/// The hash table uses 4-byte windows for match finding with a 12-bit hash.
///
/// # Example
/// ```
/// use genovo_core::compression::LZ4Compressor;
///
/// let data = b"AAAAABBBBBCCCCCAAAAABBBBB";
/// let compressed = LZ4Compressor::compress(data);
/// let decompressed = LZ4Compressor::decompress(&compressed, data.len()).unwrap();
/// assert_eq!(&decompressed, data);
/// ```
pub struct LZ4Compressor;

impl LZ4Compressor {
    /// Size of the hash table (4096 entries = 2^12).
    const HASH_TABLE_SIZE: usize = 4096;
    /// Minimum match length (LZ4 standard).
    const MIN_MATCH: usize = 4;
    /// Maximum offset for a back-reference (64 KiB window).
    const MAX_OFFSET: usize = 65535;

    /// Hashes 4 bytes starting at `data[pos]` into a 12-bit index.
    #[inline]
    fn hash4(data: &[u8], pos: usize) -> usize {
        if pos + 4 > data.len() {
            return 0;
        }
        let val = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        ((val.wrapping_mul(2654435761)) >> 20) as usize & (Self::HASH_TABLE_SIZE - 1)
    }

    /// Writes a literal-length or match-length extension using the LZ4 variable
    /// encoding: successive 255-bytes followed by a remainder.
    fn write_length_extension(output: &mut Vec<u8>, mut length: usize) {
        while length >= 255 {
            output.push(255);
            length -= 255;
        }
        output.push(length as u8);
    }

    /// Reads a variable-length extension value from the compressed stream.
    fn read_length_extension(input: &[u8], pos: &mut usize) -> usize {
        let mut total: usize = 0;
        loop {
            if *pos >= input.len() {
                break;
            }
            let byte = input[*pos];
            *pos += 1;
            total += byte as usize;
            if byte != 255 {
                break;
            }
        }
        total
    }

    /// Compresses `input` using LZ4-style algorithm.
    pub fn compress(input: &[u8]) -> Vec<u8> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(input.len());
        let mut hash_table = vec![0usize; Self::HASH_TABLE_SIZE];
        let mut anchor = 0usize; // start of un-emitted literals
        let mut pos = 0usize;
        let input_end = if input.len() > 5 {
            input.len() - 5
        } else {
            0
        };

        while pos < input_end {
            // Look up the hash for the 4 bytes at `pos`.
            let h = Self::hash4(input, pos);
            let candidate = hash_table[h];
            hash_table[h] = pos;

            // Check if we have a valid match.
            let offset = pos - candidate;
            if offset == 0
                || offset > Self::MAX_OFFSET
                || candidate >= pos
                || pos + 4 > input.len()
                || candidate + 4 > input.len()
                || input[candidate] != input[pos]
                || input[candidate + 1] != input[pos + 1]
                || input[candidate + 2] != input[pos + 2]
                || input[candidate + 3] != input[pos + 3]
            {
                pos += 1;
                continue;
            }

            // Extend the match forward.
            let mut match_len = Self::MIN_MATCH;
            while pos + match_len < input.len()
                && candidate + match_len < input.len()
                && input[pos + match_len] == input[candidate + match_len]
            {
                match_len += 1;
            }

            // Emit the token.
            let literal_len = pos - anchor;
            let ml_base = match_len - Self::MIN_MATCH;

            let lit_nibble = if literal_len >= 15 { 15u8 } else { literal_len as u8 };
            let match_nibble = if ml_base >= 15 { 15u8 } else { ml_base as u8 };
            output.push((lit_nibble << 4) | match_nibble);

            // Literal length extension.
            if literal_len >= 15 {
                Self::write_length_extension(&mut output, literal_len - 15);
            }

            // Copy literal bytes.
            output.extend_from_slice(&input[anchor..anchor + literal_len]);

            // Match offset (2 bytes, little-endian).
            output.push((offset & 0xFF) as u8);
            output.push(((offset >> 8) & 0xFF) as u8);

            // Match length extension.
            if ml_base >= 15 {
                Self::write_length_extension(&mut output, ml_base - 15);
            }

            // Advance past the match.
            pos += match_len;
            anchor = pos;
        }

        // Emit remaining literals (the last 5+ bytes are always literals in LZ4).
        let remaining_literals = input.len() - anchor;
        if remaining_literals > 0 {
            let lit_nibble = if remaining_literals >= 15 {
                15u8
            } else {
                remaining_literals as u8
            };
            // No match, so match nibble = 0.
            output.push(lit_nibble << 4);

            if remaining_literals >= 15 {
                Self::write_length_extension(&mut output, remaining_literals - 15);
            }

            output.extend_from_slice(&input[anchor..]);
        }

        output
    }

    /// Decompresses an LZ4-compressed stream. `max_output` is the maximum
    /// decompressed size (for safety).
    pub fn decompress(input: &[u8], max_output: usize) -> Option<Vec<u8>> {
        if input.is_empty() {
            return Some(Vec::new());
        }

        let mut output = Vec::with_capacity(max_output.min(input.len() * 4));
        let mut pos = 0usize;

        while pos < input.len() {
            if output.len() > max_output {
                return None; // exceeded safety limit
            }

            let token = input[pos];
            pos += 1;

            // Decode literal length.
            let mut literal_len = ((token >> 4) & 0x0F) as usize;
            if literal_len == 15 {
                literal_len += Self::read_length_extension(input, &mut pos);
            }

            // Copy literals.
            if pos + literal_len > input.len() {
                return None; // malformed
            }
            output.extend_from_slice(&input[pos..pos + literal_len]);
            pos += literal_len;

            // If we've consumed the entire input, we're done (last sequence has
            // no match).
            if pos >= input.len() {
                break;
            }

            // Decode match offset.
            if pos + 2 > input.len() {
                return None;
            }
            let offset = input[pos] as usize | ((input[pos + 1] as usize) << 8);
            pos += 2;

            if offset == 0 {
                return None; // invalid offset
            }

            // Decode match length.
            let mut match_len = ((token & 0x0F) as usize) + Self::MIN_MATCH;
            if (token & 0x0F) == 15 {
                match_len += Self::read_length_extension(input, &mut pos);
            }

            // Copy from already-decoded output (may overlap for RLE-like runs).
            let match_start = output.len().checked_sub(offset)?;
            for i in 0..match_len {
                let byte = output[match_start + (i % offset)];
                output.push(byte);
            }
        }

        Some(output)
    }
}

// ---------------------------------------------------------------------------
// RunLengthEncoder
// ---------------------------------------------------------------------------

/// Run-length encoding: compresses runs of identical bytes into
/// `(count, value)` pairs. Highly effective for tile maps, sprite masks,
/// and height-field data with large uniform regions.
///
/// The encoded format uses a variable-length count:
/// - If count <= 127: single byte `count`
/// - If count > 127: byte `0x80 | (count >> 8)` followed by byte `count & 0xFF`
///
/// For simplicity, this implementation uses fixed 2-byte `(count_hi, count_lo, value)`
/// encoding to support runs up to 65535.
///
/// # Example
/// ```
/// use genovo_core::compression::RunLengthEncoder;
///
/// let data = vec![0u8; 100];
/// let encoded = RunLengthEncoder::encode(&data);
/// let decoded = RunLengthEncoder::decode(&encoded);
/// assert_eq!(decoded, data);
/// assert!(encoded.len() < data.len());
/// ```
pub struct RunLengthEncoder;

impl RunLengthEncoder {
    /// Maximum run length per encoded triplet.
    const MAX_RUN: usize = 65535;

    /// Encodes `data` into run-length encoded form.
    ///
    /// Output format: repeated groups of `[count_hi, count_lo, value]`.
    pub fn encode(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let value = data[i];
            let mut run_len = 1usize;

            while i + run_len < data.len()
                && data[i + run_len] == value
                && run_len < Self::MAX_RUN
            {
                run_len += 1;
            }

            // Write count as big-endian u16 + value byte.
            output.push((run_len >> 8) as u8);
            output.push((run_len & 0xFF) as u8);
            output.push(value);

            i += run_len;
        }

        output
    }

    /// Decodes run-length encoded data back to the original byte stream.
    pub fn decode(data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let mut i = 0;

        while i + 2 < data.len() {
            let count = ((data[i] as usize) << 8) | (data[i + 1] as usize);
            let value = data[i + 2];
            for _ in 0..count {
                output.push(value);
            }
            i += 3;
        }

        output
    }

    /// Returns the compression ratio (compressed_size / original_size).
    pub fn compression_ratio(original: &[u8], compressed: &[u8]) -> f64 {
        if original.is_empty() {
            return 1.0;
        }
        compressed.len() as f64 / original.len() as f64
    }
}

// ---------------------------------------------------------------------------
// HuffmanCoder — entropy coding
// ---------------------------------------------------------------------------

/// A Huffman code table entry: the variable-length bit pattern and its length.
#[derive(Debug, Clone, Copy, Default)]
pub struct HuffmanCode {
    /// The bit pattern (right-aligned, up to 32 bits).
    pub bits: u32,
    /// Number of valid bits.
    pub length: u8,
}

/// A Huffman coding table mapping each byte value to its prefix code.
/// Used for both encoding and decoding.
#[derive(Clone)]
pub struct HuffmanTable {
    /// Code for each possible byte value (256 entries).
    codes: [HuffmanCode; 256],
    /// Frequency of each symbol (for reference / serialization).
    frequencies: [u32; 256],
    /// Maximum code length in the table.
    max_code_length: u8,
}

impl HuffmanTable {
    /// Returns the code for a given byte value.
    pub fn code_for(&self, byte: u8) -> HuffmanCode {
        self.codes[byte as usize]
    }

    /// Returns the frequency array.
    pub fn frequencies(&self) -> &[u32; 256] {
        &self.frequencies
    }

    /// Returns the maximum code length.
    pub fn max_code_length(&self) -> u8 {
        self.max_code_length
    }
}

impl fmt::Debug for HuffmanTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let active = self.codes.iter().filter(|c| c.length > 0).count();
        f.debug_struct("HuffmanTable")
            .field("active_symbols", &active)
            .field("max_code_length", &self.max_code_length)
            .finish()
    }
}

/// Internal node for building the Huffman tree.
#[derive(Debug, Clone)]
enum HuffmanNode {
    Leaf {
        symbol: u8,
        frequency: u32,
    },
    Internal {
        frequency: u32,
        left: Box<HuffmanNode>,
        right: Box<HuffmanNode>,
    },
}

impl HuffmanNode {
    fn frequency(&self) -> u32 {
        match self {
            HuffmanNode::Leaf { frequency, .. } => *frequency,
            HuffmanNode::Internal { frequency, .. } => *frequency,
        }
    }
}

impl PartialEq for HuffmanNode {
    fn eq(&self, other: &Self) -> bool {
        self.frequency() == other.frequency()
    }
}

impl Eq for HuffmanNode {}

impl PartialOrd for HuffmanNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HuffmanNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.frequency().cmp(&other.frequency())
    }
}

/// Huffman entropy coder. Builds an optimal prefix-free code from symbol
/// frequencies and uses it for encoding / decoding.
///
/// # Example
/// ```
/// use genovo_core::compression::HuffmanCoder;
///
/// let data = b"AAAAABBBCCDDDDDDDD";
/// let (encoded, table) = HuffmanCoder::encode(data);
/// let decoded = HuffmanCoder::decode(&encoded, &table, data.len());
/// assert_eq!(decoded, data);
/// ```
pub struct HuffmanCoder;

impl HuffmanCoder {
    /// Builds a frequency table from the input data.
    pub fn build_frequency_table(data: &[u8]) -> [u32; 256] {
        let mut freq = [0u32; 256];
        for &byte in data {
            freq[byte as usize] += 1;
        }
        freq
    }

    /// Builds a Huffman tree from a frequency table using a min-heap.
    fn build_tree(freq: &[u32; 256]) -> Option<HuffmanNode> {
        let mut heap: BinaryHeap<Reverse<HuffmanNode>> = BinaryHeap::new();

        for (symbol, &f) in freq.iter().enumerate() {
            if f > 0 {
                heap.push(Reverse(HuffmanNode::Leaf {
                    symbol: symbol as u8,
                    frequency: f,
                }));
            }
        }

        if heap.is_empty() {
            return None;
        }

        // Special case: only one symbol.
        if heap.len() == 1 {
            let node = heap.pop().unwrap().0;
            // Wrap in an internal node so it gets a code of length 1.
            return Some(HuffmanNode::Internal {
                frequency: node.frequency(),
                left: Box::new(node),
                right: Box::new(HuffmanNode::Leaf {
                    symbol: 0,
                    frequency: 0,
                }),
            });
        }

        // Standard Huffman tree construction: repeatedly merge the two
        // lowest-frequency nodes.
        while heap.len() > 1 {
            let left = heap.pop().unwrap().0;
            let right = heap.pop().unwrap().0;
            let merged = HuffmanNode::Internal {
                frequency: left.frequency() + right.frequency(),
                left: Box::new(left),
                right: Box::new(right),
            };
            heap.push(Reverse(merged));
        }

        Some(heap.pop().unwrap().0)
    }

    /// Recursively generates prefix codes from the Huffman tree.
    fn generate_codes(
        node: &HuffmanNode,
        code: u32,
        length: u8,
        table: &mut [HuffmanCode; 256],
        max_len: &mut u8,
    ) {
        match node {
            HuffmanNode::Leaf { symbol, .. } => {
                table[*symbol as usize] = HuffmanCode { bits: code, length };
                if length > *max_len {
                    *max_len = length;
                }
            }
            HuffmanNode::Internal { left, right, .. } => {
                // Left child gets a 0 bit, right child gets a 1 bit.
                Self::generate_codes(left, code << 1, length + 1, table, max_len);
                Self::generate_codes(right, (code << 1) | 1, length + 1, table, max_len);
            }
        }
    }

    /// Builds a [`HuffmanTable`] from the input data.
    pub fn build_table(data: &[u8]) -> HuffmanTable {
        let freq = Self::build_frequency_table(data);
        let mut codes = [HuffmanCode::default(); 256];
        let mut max_len = 0u8;

        if let Some(tree) = Self::build_tree(&freq) {
            Self::generate_codes(&tree, 0, 0, &mut codes, &mut max_len);
        }

        HuffmanTable {
            codes,
            frequencies: freq,
            max_code_length: max_len,
        }
    }

    /// Encodes data using Huffman coding. Returns the packed bit stream
    /// and the code table needed for decoding.
    pub fn encode(data: &[u8]) -> (Vec<u8>, HuffmanTable) {
        let table = Self::build_table(data);

        if data.is_empty() {
            return (Vec::new(), table);
        }

        let mut output = Vec::new();
        let mut current_byte: u8 = 0;
        let mut bits_in_byte: u8 = 0;

        for &byte in data {
            let code = table.codes[byte as usize];
            // Write code bits MSB first.
            for i in (0..code.length).rev() {
                let bit = (code.bits >> i) & 1;
                current_byte = (current_byte << 1) | (bit as u8);
                bits_in_byte += 1;
                if bits_in_byte == 8 {
                    output.push(current_byte);
                    current_byte = 0;
                    bits_in_byte = 0;
                }
            }
        }

        // Flush remaining bits (pad with zeros on the right).
        if bits_in_byte > 0 {
            current_byte <<= 8 - bits_in_byte;
            output.push(current_byte);
        }

        (output, table)
    }

    /// Decodes a Huffman-encoded bit stream back to the original data.
    /// `output_len` is required to know when to stop.
    pub fn decode(bits: &[u8], table: &HuffmanTable, output_len: usize) -> Vec<u8> {
        if output_len == 0 {
            return Vec::new();
        }

        // Rebuild a decode lookup: we walk the bit stream and match against
        // all codes. For a real engine we'd build a lookup table or tree,
        // but for correctness this brute-force approach works.
        //
        // Build a mapping from (bits, length) -> symbol for all active codes.
        let mut decode_map: Vec<(u32, u8, u8)> = Vec::new(); // (bits, length, symbol)
        for (sym, &code) in table.codes.iter().enumerate() {
            if code.length > 0 {
                decode_map.push((code.bits, code.length, sym as u8));
            }
        }
        // Sort by code length for fast matching (shortest codes first).
        decode_map.sort_by_key(|&(_, len, _)| len);

        let mut output = Vec::with_capacity(output_len);
        let mut bit_pos: usize = 0; // global bit position in `bits`
        let total_bits = bits.len() * 8;

        while output.len() < output_len && bit_pos < total_bits {
            let mut matched = false;

            for &(code_bits, code_len, symbol) in &decode_map {
                let cl = code_len as usize;
                if bit_pos + cl > total_bits {
                    continue;
                }

                // Extract `cl` bits starting at `bit_pos`.
                let mut extracted: u32 = 0;
                for j in 0..cl {
                    let byte_idx = (bit_pos + j) / 8;
                    let bit_idx = 7 - ((bit_pos + j) % 8);
                    let bit_val = ((bits[byte_idx] >> bit_idx) & 1) as u32;
                    extracted = (extracted << 1) | bit_val;
                }

                if extracted == code_bits {
                    output.push(symbol);
                    bit_pos += cl;
                    matched = true;
                    break;
                }
            }

            if !matched {
                break; // padding bits or corruption
            }
        }

        output
    }

    /// Computes the theoretical entropy of the data in bits per symbol.
    pub fn entropy(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let freq = Self::build_frequency_table(data);
        let total = data.len() as f64;
        let mut entropy = 0.0f64;
        for &f in &freq {
            if f > 0 {
                let p = f as f64 / total;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

// ---------------------------------------------------------------------------
// DeltaEncoder — for time-series / animation data
// ---------------------------------------------------------------------------

/// Delta encoding for integer time-series data (positions, animation keys,
/// telemetry). Stores first-order differences, which typically compress much
/// better with a subsequent entropy or RLE pass.
///
/// # Example
/// ```
/// use genovo_core::compression::DeltaEncoder;
///
/// let values = vec![100, 102, 105, 103, 110];
/// let deltas = DeltaEncoder::encode_deltas(&values);
/// assert_eq!(deltas, vec![100, 2, 3, -2, 7]);
/// let restored = DeltaEncoder::decode_deltas(&deltas);
/// assert_eq!(restored, values);
/// ```
pub struct DeltaEncoder;

impl DeltaEncoder {
    /// Encodes a sequence of values as deltas: `[v0, v1-v0, v2-v1, ...]`.
    pub fn encode_deltas(values: &[i32]) -> Vec<i32> {
        if values.is_empty() {
            return Vec::new();
        }
        let mut deltas = Vec::with_capacity(values.len());
        deltas.push(values[0]);
        for i in 1..values.len() {
            deltas.push(values[i] - values[i - 1]);
        }
        deltas
    }

    /// Decodes deltas back to absolute values.
    pub fn decode_deltas(deltas: &[i32]) -> Vec<i32> {
        if deltas.is_empty() {
            return Vec::new();
        }
        let mut values = Vec::with_capacity(deltas.len());
        values.push(deltas[0]);
        for i in 1..deltas.len() {
            values.push(values[i - 1] + deltas[i]);
        }
        values
    }

    /// Encodes second-order deltas (delta of deltas) for smoother curves.
    pub fn encode_double_deltas(values: &[i32]) -> Vec<i32> {
        let first = Self::encode_deltas(values);
        Self::encode_deltas(&first)
    }

    /// Decodes second-order deltas.
    pub fn decode_double_deltas(double_deltas: &[i32]) -> Vec<i32> {
        let first = Self::decode_deltas(double_deltas);
        Self::decode_deltas(&first)
    }

    /// Encodes unsigned values using ZigZag encoding so that small negative
    /// numbers map to small positive numbers (better for subsequent compression).
    pub fn zigzag_encode(value: i32) -> u32 {
        ((value << 1) ^ (value >> 31)) as u32
    }

    /// Decodes a ZigZag-encoded value.
    pub fn zigzag_decode(value: u32) -> i32 {
        ((value >> 1) as i32) ^ (-((value & 1) as i32))
    }

    /// Encodes a slice of i32 deltas with ZigZag encoding for unsigned output.
    pub fn encode_zigzag(values: &[i32]) -> Vec<u32> {
        values.iter().map(|&v| Self::zigzag_encode(v)).collect()
    }

    /// Decodes a ZigZag-encoded slice back to i32.
    pub fn decode_zigzag(values: &[u32]) -> Vec<i32> {
        values.iter().map(|&v| Self::zigzag_decode(v)).collect()
    }

    /// Combines delta + zigzag for optimal compression preparation.
    pub fn encode_deltas_zigzag(values: &[i32]) -> Vec<u32> {
        let deltas = Self::encode_deltas(values);
        Self::encode_zigzag(&deltas)
    }

    /// Decodes combined delta + zigzag.
    pub fn decode_deltas_zigzag(encoded: &[u32]) -> Vec<i32> {
        let deltas = Self::decode_zigzag(encoded);
        Self::decode_deltas(&deltas)
    }

    /// Variable-length encodes a u32 value (1-5 bytes, 7 bits per byte).
    pub fn varint_encode(mut value: u32, output: &mut Vec<u8>) {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            if value == 0 {
                output.push(byte);
                break;
            } else {
                output.push(byte | 0x80);
            }
        }
    }

    /// Decodes a varint from the byte stream, advancing `pos`.
    pub fn varint_decode(input: &[u8], pos: &mut usize) -> Option<u32> {
        let mut result: u32 = 0;
        let mut shift = 0u32;
        loop {
            if *pos >= input.len() {
                return None;
            }
            let byte = input[*pos];
            *pos += 1;
            result |= ((byte & 0x7F) as u32) << shift;
            if byte & 0x80 == 0 {
                return Some(result);
            }
            shift += 7;
            if shift >= 35 {
                return None; // overflow protection
            }
        }
    }

    /// Encodes deltas + zigzag + varint into a compact byte stream.
    pub fn encode_compact(values: &[i32]) -> Vec<u8> {
        let zigzag = Self::encode_deltas_zigzag(values);
        let mut output = Vec::new();
        // Store count as varint first.
        Self::varint_encode(zigzag.len() as u32, &mut output);
        for &v in &zigzag {
            Self::varint_encode(v, &mut output);
        }
        output
    }

    /// Decodes a compact-encoded byte stream back to i32 values.
    pub fn decode_compact(data: &[u8]) -> Vec<i32> {
        let mut pos = 0;
        let count = match Self::varint_decode(data, &mut pos) {
            Some(c) => c as usize,
            None => return Vec::new(),
        };
        let mut zigzag = Vec::with_capacity(count);
        for _ in 0..count {
            match Self::varint_decode(data, &mut pos) {
                Some(v) => zigzag.push(v),
                None => break,
            }
        }
        Self::decode_deltas_zigzag(&zigzag)
    }
}

// ---------------------------------------------------------------------------
// Compression utilities
// ---------------------------------------------------------------------------

/// Identifies the best compression strategy for a data block by sampling.
pub struct CompressionAnalyzer;

impl CompressionAnalyzer {
    /// Estimates the byte entropy (0.0 = uniform, 8.0 = maximum entropy).
    pub fn byte_entropy(data: &[u8]) -> f64 {
        HuffmanCoder::entropy(data)
    }

    /// Estimates the run-length compressibility: average run length.
    pub fn average_run_length(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut runs = 0usize;
        let mut i = 0;
        while i < data.len() {
            let v = data[i];
            while i < data.len() && data[i] == v {
                i += 1;
            }
            runs += 1;
        }
        data.len() as f64 / runs as f64
    }

    /// Returns `true` if the data likely benefits from LZ4 compression
    /// (has repeated patterns beyond simple runs).
    pub fn is_lz4_beneficial(data: &[u8]) -> bool {
        if data.len() < 64 {
            return false;
        }
        let compressed = LZ4Compressor::compress(data);
        compressed.len() < (data.len() * 9 / 10) // at least 10% savings
    }

    /// Returns `true` if RLE is likely effective (long average runs).
    pub fn is_rle_beneficial(data: &[u8]) -> bool {
        Self::average_run_length(data) >= 4.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- LZ4 tests ------------------------------------------------------------

    #[test]
    fn lz4_empty() {
        let compressed = LZ4Compressor::compress(b"");
        let decompressed = LZ4Compressor::decompress(&compressed, 0).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn lz4_short() {
        let data = b"Hi!";
        let compressed = LZ4Compressor::compress(data);
        let decompressed = LZ4Compressor::decompress(&compressed, data.len() + 100).unwrap();
        assert_eq!(&decompressed, data);
    }

    #[test]
    fn lz4_repeated() {
        let data = b"ABCABCABCABCABCABCABCABC";
        let compressed = LZ4Compressor::compress(data);
        let decompressed = LZ4Compressor::decompress(&compressed, data.len() + 100).unwrap();
        assert_eq!(&decompressed[..], &data[..]);
    }

    #[test]
    fn lz4_large_repeated() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(pattern);
        }
        let compressed = LZ4Compressor::compress(&data);
        assert!(compressed.len() < data.len(), "LZ4 should compress repeated data");
        let decompressed = LZ4Compressor::decompress(&compressed, data.len() + 100).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_all_same() {
        let data = vec![0xAA; 10000];
        let compressed = LZ4Compressor::compress(&data);
        assert!(compressed.len() < 200, "all-same data should compress extremely well");
        let decompressed = LZ4Compressor::decompress(&compressed, data.len() + 100).unwrap();
        assert_eq!(decompressed, data);
    }

    // -- RLE tests ------------------------------------------------------------

    #[test]
    fn rle_basic() {
        let data = b"AAABBBCCCC";
        let encoded = RunLengthEncoder::encode(data);
        let decoded = RunLengthEncoder::decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn rle_uniform() {
        let data = vec![42u8; 1000];
        let encoded = RunLengthEncoder::encode(&data);
        let decoded = RunLengthEncoder::decode(&encoded);
        assert_eq!(decoded, data);
        assert!(encoded.len() < data.len());
    }

    #[test]
    fn rle_empty() {
        assert!(RunLengthEncoder::encode(b"").is_empty());
        assert!(RunLengthEncoder::decode(b"").is_empty());
    }

    #[test]
    fn rle_no_runs() {
        let data = b"ABCDEF";
        let encoded = RunLengthEncoder::encode(data);
        let decoded = RunLengthEncoder::decode(&encoded);
        assert_eq!(decoded, data);
    }

    // -- Huffman tests --------------------------------------------------------

    #[test]
    fn huffman_basic() {
        let data = b"AAAAABBBCCDDDDDDDD";
        let (encoded, table) = HuffmanCoder::encode(data);
        let decoded = HuffmanCoder::decode(&encoded, &table, data.len());
        assert_eq!(decoded, data);
    }

    #[test]
    fn huffman_single_symbol() {
        let data = b"AAAAAAA";
        let (encoded, table) = HuffmanCoder::encode(data);
        let decoded = HuffmanCoder::decode(&encoded, &table, data.len());
        assert_eq!(decoded, data);
    }

    #[test]
    fn huffman_all_256() {
        let data: Vec<u8> = (0..=255).collect();
        let (encoded, table) = HuffmanCoder::encode(&data);
        let decoded = HuffmanCoder::decode(&encoded, &table, data.len());
        assert_eq!(decoded, data);
    }

    #[test]
    fn huffman_frequency_table() {
        let data = b"AABBBCCCC";
        let freq = HuffmanCoder::build_frequency_table(data);
        assert_eq!(freq[b'A' as usize], 2);
        assert_eq!(freq[b'B' as usize], 3);
        assert_eq!(freq[b'C' as usize], 4);
    }

    #[test]
    fn huffman_entropy() {
        let uniform = vec![42u8; 100];
        assert!(HuffmanCoder::entropy(&uniform) < 0.01);

        let diverse: Vec<u8> = (0..=255).cycle().take(2560).collect();
        let entropy = HuffmanCoder::entropy(&diverse);
        assert!(entropy > 7.9); // near-maximum entropy
    }

    // -- Delta tests ----------------------------------------------------------

    #[test]
    fn delta_roundtrip() {
        let values = vec![100, 102, 105, 103, 110];
        let deltas = DeltaEncoder::encode_deltas(&values);
        assert_eq!(deltas, vec![100, 2, 3, -2, 7]);
        let restored = DeltaEncoder::decode_deltas(&deltas);
        assert_eq!(restored, values);
    }

    #[test]
    fn delta_double_roundtrip() {
        let values = vec![0, 1, 4, 9, 16, 25];
        let dd = DeltaEncoder::encode_double_deltas(&values);
        let restored = DeltaEncoder::decode_double_deltas(&dd);
        assert_eq!(restored, values);
    }

    #[test]
    fn delta_zigzag() {
        assert_eq!(DeltaEncoder::zigzag_encode(0), 0);
        assert_eq!(DeltaEncoder::zigzag_encode(-1), 1);
        assert_eq!(DeltaEncoder::zigzag_encode(1), 2);
        assert_eq!(DeltaEncoder::zigzag_encode(-2), 3);

        for v in -1000..=1000 {
            assert_eq!(DeltaEncoder::zigzag_decode(DeltaEncoder::zigzag_encode(v)), v);
        }
    }

    #[test]
    fn delta_compact_roundtrip() {
        let values = vec![100, 102, 105, 103, 110, 115, 112, 108, 100];
        let compact = DeltaEncoder::encode_compact(&values);
        let restored = DeltaEncoder::decode_compact(&compact);
        assert_eq!(restored, values);
        assert!(compact.len() < values.len() * 4, "compact should be smaller than raw i32s");
    }

    #[test]
    fn varint_roundtrip() {
        let test_values = [0u32, 1, 127, 128, 16383, 16384, 2097151, u32::MAX];
        for &val in &test_values {
            let mut buf = Vec::new();
            DeltaEncoder::varint_encode(val, &mut buf);
            let mut pos = 0;
            let decoded = DeltaEncoder::varint_decode(&buf, &mut pos).unwrap();
            assert_eq!(decoded, val, "varint roundtrip failed for {}", val);
            assert_eq!(pos, buf.len(), "varint didn't consume all bytes for {}", val);
        }
    }

    // -- Analyzer tests -------------------------------------------------------

    #[test]
    fn analyzer_entropy() {
        let uniform = vec![0u8; 100];
        assert!(CompressionAnalyzer::byte_entropy(&uniform) < 0.01);
    }

    #[test]
    fn analyzer_run_length() {
        let data = vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
        let avg = CompressionAnalyzer::average_run_length(&data);
        assert!((avg - 4.0).abs() < 0.01);
    }
}
