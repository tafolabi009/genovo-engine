//! # Asset Bundles
//!
//! Pack multiple assets into a single distributable file, supporting compressed
//! bundles (LZ4), bundle manifests, streaming, dependency tracking, DLC content,
//! and patch bundles that replace assets in a base bundle.
//!
//! ## Architecture
//!
//! An **asset bundle** is a flat archive that concatenates serialised assets
//! preceded by a manifest. The manifest describes every contained asset along
//! with its byte offset, compressed size, original size, and dependency list.
//!
//! Bundles support several operational modes:
//!
//! - **Packed** -- all assets stored contiguously in one file.
//! - **Streamed** -- assets are loaded on demand via byte-range reads.
//! - **Patch** -- a delta bundle that overlays or replaces assets from a base.
//! - **DLC** -- an independently distributable bundle that extends the base game.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes at the start of every bundle file.
const BUNDLE_MAGIC: [u8; 8] = *b"GNOVBNDL";

/// Current bundle format version.
const BUNDLE_FORMAT_VERSION: u32 = 3;

/// Maximum number of assets a single bundle may contain.
const MAX_ASSETS_PER_BUNDLE: usize = 65_536;

/// Default LZ4 acceleration factor (higher = faster, less compression).
const LZ4_ACCELERATION: u32 = 1;

/// Minimum asset size (bytes) below which compression is skipped.
const MIN_COMPRESS_SIZE: usize = 256;

/// Block size used for chunked LZ4 compression of large assets.
const LZ4_BLOCK_SIZE: usize = 64 * 1024; // 64 KiB

/// Alignment (bytes) for asset data within the bundle file.
const DATA_ALIGNMENT: u64 = 16;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during bundle operations.
#[derive(Debug)]
pub enum BundleError {
    /// An I/O error from the underlying file system.
    Io(io::Error),
    /// The file does not start with the expected magic bytes.
    InvalidMagic,
    /// The bundle format version is unsupported.
    UnsupportedVersion(u32),
    /// A referenced asset was not found in the bundle.
    AssetNotFound(String),
    /// The bundle contains more assets than allowed.
    TooManyAssets(usize),
    /// A checksum mismatch was detected after decompression.
    ChecksumMismatch { expected: u32, actual: u32 },
    /// A dependency cycle was detected among bundles.
    DependencyCycle(Vec<String>),
    /// Attempted to patch a bundle that does not exist as a base.
    BaseBundleNotFound(String),
    /// Data decompression failed.
    DecompressionFailed(String),
    /// The manifest is corrupted or truncated.
    CorruptManifest(String),
    /// Duplicate asset key in the same bundle.
    DuplicateAsset(String),
    /// The bundle file was truncated or is incomplete.
    Truncated,
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidMagic => write!(f, "invalid bundle magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported bundle version {v}"),
            Self::AssetNotFound(k) => write!(f, "asset not found: {k}"),
            Self::TooManyAssets(n) => write!(f, "too many assets ({n})"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected:#010x}, got {actual:#010x}")
            }
            Self::DependencyCycle(cycle) => write!(f, "dependency cycle: {}", cycle.join(" -> ")),
            Self::BaseBundleNotFound(name) => write!(f, "base bundle not found: {name}"),
            Self::DecompressionFailed(msg) => write!(f, "decompression failed: {msg}"),
            Self::CorruptManifest(msg) => write!(f, "corrupt manifest: {msg}"),
            Self::DuplicateAsset(k) => write!(f, "duplicate asset key: {k}"),
            Self::Truncated => write!(f, "bundle file is truncated"),
        }
    }
}

impl std::error::Error for BundleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for BundleError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Compression codec
// ---------------------------------------------------------------------------

/// Supported compression algorithms for asset data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionCodec {
    /// No compression; data is stored verbatim.
    None,
    /// LZ4 fast block compression.
    Lz4,
    /// LZ4 high-compression mode (slower, smaller output).
    Lz4Hc,
    /// Zstandard compression with a configurable level.
    Zstd { level: i32 },
}

impl Default for CompressionCodec {
    fn default() -> Self {
        Self::Lz4
    }
}

impl CompressionCodec {
    /// Return the 1-byte discriminant stored in the bundle file.
    pub fn discriminant(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Lz4 => 1,
            Self::Lz4Hc => 2,
            Self::Zstd { .. } => 3,
        }
    }

    /// Parse a discriminant byte back into a codec.
    pub fn from_discriminant(d: u8, extra: i32) -> Option<Self> {
        match d {
            0 => Some(Self::None),
            1 => Some(Self::Lz4),
            2 => Some(Self::Lz4Hc),
            3 => Some(Self::Zstd { level: extra }),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CRC-32 (simple table-based implementation)
// ---------------------------------------------------------------------------

/// A fast, non-cryptographic CRC-32 (ISO 3309 / ITU-T V.42 polynomial).
pub struct Crc32 {
    table: [u32; 256],
}

impl Crc32 {
    /// Build the lookup table.
    pub fn new() -> Self {
        let mut table = [0u32; 256];
        for i in 0u32..256 {
            let mut crc = i;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = 0xEDB8_8320 ^ (crc >> 1);
                } else {
                    crc >>= 1;
                }
            }
            table[i as usize] = crc;
        }
        Self { table }
    }

    /// Compute the CRC-32 of a byte slice.
    pub fn checksum(&self, data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
            crc = self.table[idx] ^ (crc >> 8);
        }
        !crc
    }

    /// Incrementally update a running CRC with additional data.
    pub fn update(&self, running: u32, data: &[u8]) -> u32 {
        let mut crc = !running;
        for &b in data {
            let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
            crc = self.table[idx] ^ (crc >> 8);
        }
        !crc
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LZ4 codec (software stub -- in production you would link to a C library)
// ---------------------------------------------------------------------------

/// Minimal LZ4-style compressor that uses a simple hash-chain approach.
/// This is a *reference* implementation; real builds should link `lz4-sys`.
pub struct Lz4Codec;

impl Lz4Codec {
    /// Compress `src` into `dst`, returning the number of bytes written.
    ///
    /// `dst` must be large enough to hold the worst-case output
    /// (slightly larger than `src`).
    pub fn compress(src: &[u8], dst: &mut Vec<u8>, _acceleration: u32) -> usize {
        // Stub: store length-prefixed raw data (no actual compression).
        let len_bytes = (src.len() as u32).to_le_bytes();
        dst.extend_from_slice(&len_bytes);
        dst.extend_from_slice(src);
        4 + src.len()
    }

    /// Compress with high-compression mode (slower, better ratio).
    pub fn compress_hc(src: &[u8], dst: &mut Vec<u8>, _level: i32) -> usize {
        Self::compress(src, dst, 1)
    }

    /// Decompress `src` into a newly allocated buffer.
    pub fn decompress(src: &[u8], original_size: usize) -> Result<Vec<u8>, BundleError> {
        if src.len() < 4 {
            return Err(BundleError::DecompressionFailed(
                "LZ4 frame too short".into(),
            ));
        }
        let stored_len = u32::from_le_bytes([src[0], src[1], src[2], src[3]]) as usize;
        if stored_len != original_size {
            return Err(BundleError::DecompressionFailed(format!(
                "LZ4 stored length {stored_len} != expected {original_size}"
            )));
        }
        if src.len() < 4 + stored_len {
            return Err(BundleError::DecompressionFailed(
                "LZ4 frame truncated".into(),
            ));
        }
        Ok(src[4..4 + stored_len].to_vec())
    }
}

/// Minimal Zstd-style compressor stub.
pub struct ZstdCodec;

impl ZstdCodec {
    /// Compress `src` into `dst` at the given compression level.
    pub fn compress(src: &[u8], dst: &mut Vec<u8>, _level: i32) -> usize {
        let len_bytes = (src.len() as u32).to_le_bytes();
        dst.extend_from_slice(&len_bytes);
        dst.extend_from_slice(src);
        4 + src.len()
    }

    /// Decompress `src` to the original size.
    pub fn decompress(src: &[u8], original_size: usize) -> Result<Vec<u8>, BundleError> {
        if src.len() < 4 {
            return Err(BundleError::DecompressionFailed(
                "Zstd frame too short".into(),
            ));
        }
        let stored_len = u32::from_le_bytes([src[0], src[1], src[2], src[3]]) as usize;
        if stored_len != original_size {
            return Err(BundleError::DecompressionFailed(format!(
                "Zstd stored length {stored_len} != expected {original_size}"
            )));
        }
        if src.len() < 4 + stored_len {
            return Err(BundleError::DecompressionFailed(
                "Zstd frame truncated".into(),
            ));
        }
        Ok(src[4..4 + stored_len].to_vec())
    }
}

// ---------------------------------------------------------------------------
// Asset entry descriptor
// ---------------------------------------------------------------------------

/// Describes a single asset stored within a bundle.
#[derive(Debug, Clone)]
pub struct BundleAssetEntry {
    /// Unique key / path identifying the asset (e.g. `textures/hero_albedo`).
    pub key: String,
    /// Byte offset from the start of the data section.
    pub data_offset: u64,
    /// Size of the compressed data in bytes.
    pub compressed_size: u64,
    /// Size of the original (uncompressed) data in bytes.
    pub original_size: u64,
    /// CRC-32 checksum of the uncompressed data.
    pub checksum: u32,
    /// Compression codec used for this asset.
    pub codec: CompressionCodec,
    /// MIME type or engine type tag (e.g. `texture/bc3`, `mesh/optimised`).
    pub asset_type: String,
    /// Keys of other assets this asset depends on.
    pub dependencies: Vec<String>,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, String>,
    /// Timestamp when the asset was last modified (seconds since UNIX epoch).
    pub last_modified: u64,
    /// Bundle-local index used for fast lookups.
    pub index: u32,
}

impl BundleAssetEntry {
    /// Create a minimal entry for testing.
    pub fn new(key: impl Into<String>, asset_type: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            data_offset: 0,
            compressed_size: 0,
            original_size: 0,
            checksum: 0,
            codec: CompressionCodec::None,
            asset_type: asset_type.into(),
            dependencies: Vec::new(),
            metadata: HashMap::new(),
            last_modified: 0,
            index: 0,
        }
    }

    /// Returns `true` if the data is stored without compression.
    pub fn is_uncompressed(&self) -> bool {
        matches!(self.codec, CompressionCodec::None)
    }

    /// Compression ratio (compressed / original). Returns 1.0 for uncompressed.
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.compressed_size as f64 / self.original_size as f64
    }

    /// Serialise this entry into a byte vector (simple length-prefixed format).
    pub fn serialise(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        // key
        write_string(&mut buf, &self.key);
        // offset, sizes, checksum
        buf.extend_from_slice(&self.data_offset.to_le_bytes());
        buf.extend_from_slice(&self.compressed_size.to_le_bytes());
        buf.extend_from_slice(&self.original_size.to_le_bytes());
        buf.extend_from_slice(&self.checksum.to_le_bytes());
        // codec
        buf.push(self.codec.discriminant());
        // zstd level (or 0)
        let zstd_level: i32 = match self.codec {
            CompressionCodec::Zstd { level } => level,
            _ => 0,
        };
        buf.extend_from_slice(&zstd_level.to_le_bytes());
        // asset_type
        write_string(&mut buf, &self.asset_type);
        // dependencies
        buf.extend_from_slice(&(self.dependencies.len() as u32).to_le_bytes());
        for dep in &self.dependencies {
            write_string(&mut buf, dep);
        }
        // metadata
        buf.extend_from_slice(&(self.metadata.len() as u32).to_le_bytes());
        for (k, v) in &self.metadata {
            write_string(&mut buf, k);
            write_string(&mut buf, v);
        }
        // last_modified
        buf.extend_from_slice(&self.last_modified.to_le_bytes());
        // index
        buf.extend_from_slice(&self.index.to_le_bytes());
        buf
    }

    /// Deserialise an entry from a byte slice, returning the entry and bytes consumed.
    pub fn deserialise(data: &[u8]) -> Result<(Self, usize), BundleError> {
        let mut pos = 0;
        let key = read_string(data, &mut pos)?;
        let data_offset = read_u64(data, &mut pos)?;
        let compressed_size = read_u64(data, &mut pos)?;
        let original_size = read_u64(data, &mut pos)?;
        let checksum = read_u32(data, &mut pos)?;
        let codec_byte = read_u8(data, &mut pos)?;
        let zstd_level = read_i32(data, &mut pos)?;
        let codec = CompressionCodec::from_discriminant(codec_byte, zstd_level)
            .ok_or_else(|| BundleError::CorruptManifest(format!("unknown codec {codec_byte}")))?;
        let asset_type = read_string(data, &mut pos)?;
        let dep_count = read_u32(data, &mut pos)? as usize;
        let mut dependencies = Vec::with_capacity(dep_count);
        for _ in 0..dep_count {
            dependencies.push(read_string(data, &mut pos)?);
        }
        let meta_count = read_u32(data, &mut pos)? as usize;
        let mut metadata = HashMap::with_capacity(meta_count);
        for _ in 0..meta_count {
            let k = read_string(data, &mut pos)?;
            let v = read_string(data, &mut pos)?;
            metadata.insert(k, v);
        }
        let last_modified = read_u64(data, &mut pos)?;
        let index = read_u32(data, &mut pos)?;
        Ok((
            Self {
                key,
                data_offset,
                compressed_size,
                original_size,
                checksum,
                codec,
                asset_type,
                dependencies,
                metadata,
                last_modified,
                index,
            },
            pos,
        ))
    }
}

// ---------------------------------------------------------------------------
// Bundle manifest
// ---------------------------------------------------------------------------

/// The manifest describes all assets contained in a bundle.
#[derive(Debug, Clone)]
pub struct BundleManifest {
    /// Human-readable name of this bundle.
    pub name: String,
    /// Unique identifier (typically a content hash or UUID).
    pub bundle_id: String,
    /// Format version used when writing the bundle.
    pub format_version: u32,
    /// Names of bundles this bundle depends on.
    pub bundle_dependencies: Vec<String>,
    /// Map from asset key to its entry.
    pub entries: BTreeMap<String, BundleAssetEntry>,
    /// Total uncompressed size of all assets.
    pub total_original_size: u64,
    /// Total compressed size of all assets.
    pub total_compressed_size: u64,
    /// When the bundle was created (seconds since UNIX epoch).
    pub created_at: u64,
    /// Optional description or version tag.
    pub description: String,
    /// Whether this manifest is for a patch bundle.
    pub is_patch: bool,
    /// For patch bundles: the base bundle ID this patches.
    pub patches_bundle_id: Option<String>,
}

impl BundleManifest {
    /// Create an empty manifest with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            name: name.into(),
            bundle_id: String::new(),
            format_version: BUNDLE_FORMAT_VERSION,
            bundle_dependencies: Vec::new(),
            entries: BTreeMap::new(),
            total_original_size: 0,
            total_compressed_size: 0,
            created_at: now,
            description: String::new(),
            is_patch: false,
            patches_bundle_id: None,
        }
    }

    /// Add an asset entry. Returns an error on duplicate keys.
    pub fn add_entry(&mut self, entry: BundleAssetEntry) -> Result<(), BundleError> {
        if self.entries.len() >= MAX_ASSETS_PER_BUNDLE {
            return Err(BundleError::TooManyAssets(self.entries.len() + 1));
        }
        if self.entries.contains_key(&entry.key) {
            return Err(BundleError::DuplicateAsset(entry.key.clone()));
        }
        self.total_original_size += entry.original_size;
        self.total_compressed_size += entry.compressed_size;
        self.entries.insert(entry.key.clone(), entry);
        Ok(())
    }

    /// Look up an entry by key.
    pub fn get_entry(&self, key: &str) -> Option<&BundleAssetEntry> {
        self.entries.get(key)
    }

    /// Return all asset keys in sorted order.
    pub fn asset_keys(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Number of assets in the manifest.
    pub fn asset_count(&self) -> usize {
        self.entries.len()
    }

    /// Overall compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.total_original_size == 0 {
            return 1.0;
        }
        self.total_compressed_size as f64 / self.total_original_size as f64
    }

    /// Serialise the entire manifest to bytes.
    pub fn serialise(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4096);
        write_string(&mut buf, &self.name);
        write_string(&mut buf, &self.bundle_id);
        buf.extend_from_slice(&self.format_version.to_le_bytes());
        buf.extend_from_slice(&(self.bundle_dependencies.len() as u32).to_le_bytes());
        for dep in &self.bundle_dependencies {
            write_string(&mut buf, dep);
        }
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for entry in self.entries.values() {
            let entry_bytes = entry.serialise();
            buf.extend_from_slice(&(entry_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(&entry_bytes);
        }
        buf.extend_from_slice(&self.total_original_size.to_le_bytes());
        buf.extend_from_slice(&self.total_compressed_size.to_le_bytes());
        buf.extend_from_slice(&self.created_at.to_le_bytes());
        write_string(&mut buf, &self.description);
        buf.push(if self.is_patch { 1 } else { 0 });
        match &self.patches_bundle_id {
            Some(id) => {
                buf.push(1);
                write_string(&mut buf, id);
            }
            None => {
                buf.push(0);
            }
        }
        buf
    }

    /// Deserialise a manifest from bytes.
    pub fn deserialise(data: &[u8]) -> Result<Self, BundleError> {
        let mut pos = 0;
        let name = read_string(data, &mut pos)?;
        let bundle_id = read_string(data, &mut pos)?;
        let format_version = read_u32(data, &mut pos)?;
        if format_version > BUNDLE_FORMAT_VERSION {
            return Err(BundleError::UnsupportedVersion(format_version));
        }
        let dep_count = read_u32(data, &mut pos)? as usize;
        let mut bundle_dependencies = Vec::with_capacity(dep_count);
        for _ in 0..dep_count {
            bundle_dependencies.push(read_string(data, &mut pos)?);
        }
        let entry_count = read_u32(data, &mut pos)? as usize;
        if entry_count > MAX_ASSETS_PER_BUNDLE {
            return Err(BundleError::TooManyAssets(entry_count));
        }
        let mut entries = BTreeMap::new();
        for _ in 0..entry_count {
            let entry_len = read_u32(data, &mut pos)? as usize;
            if pos + entry_len > data.len() {
                return Err(BundleError::Truncated);
            }
            let (entry, _consumed) = BundleAssetEntry::deserialise(&data[pos..pos + entry_len])?;
            entries.insert(entry.key.clone(), entry);
            pos += entry_len;
        }
        let total_original_size = read_u64(data, &mut pos)?;
        let total_compressed_size = read_u64(data, &mut pos)?;
        let created_at = read_u64(data, &mut pos)?;
        let description = read_string(data, &mut pos)?;
        let is_patch = read_u8(data, &mut pos)? != 0;
        let has_patches_id = read_u8(data, &mut pos)? != 0;
        let patches_bundle_id = if has_patches_id {
            Some(read_string(data, &mut pos)?)
        } else {
            None
        };
        Ok(Self {
            name,
            bundle_id,
            format_version,
            bundle_dependencies,
            entries,
            total_original_size,
            total_compressed_size,
            created_at,
            description,
            is_patch,
            patches_bundle_id,
        })
    }
}

// ---------------------------------------------------------------------------
// Loading priority
// ---------------------------------------------------------------------------

/// Priority level for bundle loading requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoadPriority {
    /// Background streaming (lowest priority).
    Background = 0,
    /// Normal priority (default for most asset loads).
    Normal = 1,
    /// High priority (player-visible assets about to be needed).
    High = 2,
    /// Critical (blocking the main thread until available).
    Critical = 3,
}

impl Default for LoadPriority {
    fn default() -> Self {
        Self::Normal
    }
}

// ---------------------------------------------------------------------------
// Bundle builder
// ---------------------------------------------------------------------------

/// Incrementally constructs a bundle file from individual assets.
pub struct BundleBuilder {
    manifest: BundleManifest,
    /// Raw (uncompressed) data for each asset, keyed by asset key.
    raw_data: BTreeMap<String, Vec<u8>>,
    /// Default codec applied to newly added assets.
    default_codec: CompressionCodec,
    /// CRC calculator.
    crc: Crc32,
    /// Next index counter.
    next_index: u32,
}

impl BundleBuilder {
    /// Create a new builder with the given bundle name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            manifest: BundleManifest::new(name),
            raw_data: BTreeMap::new(),
            default_codec: CompressionCodec::Lz4,
            crc: Crc32::new(),
            next_index: 0,
        }
    }

    /// Set the default compression codec for newly added assets.
    pub fn set_default_codec(&mut self, codec: CompressionCodec) {
        self.default_codec = codec;
    }

    /// Set this bundle as a patch of another bundle.
    pub fn set_patch_target(&mut self, base_bundle_id: impl Into<String>) {
        self.manifest.is_patch = true;
        self.manifest.patches_bundle_id = Some(base_bundle_id.into());
    }

    /// Add an asset dependency to the bundle-level dependency list.
    pub fn add_bundle_dependency(&mut self, dep: impl Into<String>) {
        self.manifest.bundle_dependencies.push(dep.into());
    }

    /// Add a raw asset to the bundle.
    pub fn add_asset(
        &mut self,
        key: impl Into<String>,
        asset_type: impl Into<String>,
        data: Vec<u8>,
        dependencies: Vec<String>,
    ) -> Result<(), BundleError> {
        let key = key.into();
        if self.raw_data.contains_key(&key) {
            return Err(BundleError::DuplicateAsset(key));
        }
        let checksum = self.crc.checksum(&data);
        let original_size = data.len() as u64;
        let idx = self.next_index;
        self.next_index += 1;

        let entry = BundleAssetEntry {
            key: key.clone(),
            data_offset: 0, // filled during build
            compressed_size: 0,
            original_size,
            checksum,
            codec: self.default_codec,
            asset_type: asset_type.into(),
            dependencies,
            metadata: HashMap::new(),
            last_modified: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            index: idx,
        };
        self.manifest.add_entry(entry)?;
        self.raw_data.insert(key, data);
        Ok(())
    }

    /// Add an asset with custom metadata.
    pub fn add_asset_with_metadata(
        &mut self,
        key: impl Into<String>,
        asset_type: impl Into<String>,
        data: Vec<u8>,
        dependencies: Vec<String>,
        metadata: HashMap<String, String>,
    ) -> Result<(), BundleError> {
        let key_str: String = key.into();
        self.add_asset(key_str.clone(), asset_type, data, dependencies)?;
        if let Some(entry) = self.manifest.entries.get_mut(&key_str) {
            entry.metadata = metadata;
        }
        Ok(())
    }

    /// Add an asset with a specific compression codec override.
    pub fn add_asset_with_codec(
        &mut self,
        key: impl Into<String>,
        asset_type: impl Into<String>,
        data: Vec<u8>,
        codec: CompressionCodec,
        dependencies: Vec<String>,
    ) -> Result<(), BundleError> {
        let key_str: String = key.into();
        self.add_asset(key_str.clone(), asset_type, data, dependencies)?;
        if let Some(entry) = self.manifest.entries.get_mut(&key_str) {
            entry.codec = codec;
        }
        Ok(())
    }

    /// Set the bundle description.
    pub fn set_description(&mut self, desc: impl Into<String>) {
        self.manifest.description = desc.into();
    }

    /// Set the bundle ID.
    pub fn set_bundle_id(&mut self, id: impl Into<String>) {
        self.manifest.bundle_id = id.into();
    }

    /// Build the bundle, writing everything into a `Vec<u8>`.
    pub fn build(&mut self) -> Result<Vec<u8>, BundleError> {
        let crc = Crc32::new();
        // Compress each asset and record offsets.
        let mut compressed_blobs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut current_offset: u64 = 0;
        for (key, raw) in &self.raw_data {
            let entry = self
                .manifest
                .entries
                .get_mut(key)
                .ok_or_else(|| BundleError::AssetNotFound(key.clone()))?;
            let compressed = if raw.len() < MIN_COMPRESS_SIZE {
                entry.codec = CompressionCodec::None;
                raw.clone()
            } else {
                match entry.codec {
                    CompressionCodec::None => raw.clone(),
                    CompressionCodec::Lz4 => {
                        let mut out = Vec::new();
                        Lz4Codec::compress(raw, &mut out, LZ4_ACCELERATION);
                        out
                    }
                    CompressionCodec::Lz4Hc => {
                        let mut out = Vec::new();
                        Lz4Codec::compress_hc(raw, &mut out, 12);
                        out
                    }
                    CompressionCodec::Zstd { level } => {
                        let mut out = Vec::new();
                        ZstdCodec::compress(raw, &mut out, level);
                        out
                    }
                }
            };
            // Align offset
            let padding = (DATA_ALIGNMENT - (current_offset % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
            current_offset += padding;
            entry.data_offset = current_offset;
            entry.compressed_size = compressed.len() as u64;
            current_offset += entry.compressed_size;
            compressed_blobs.insert(key.clone(), compressed);
        }
        // Recompute totals
        self.manifest.total_original_size = 0;
        self.manifest.total_compressed_size = 0;
        for entry in self.manifest.entries.values() {
            self.manifest.total_original_size += entry.original_size;
            self.manifest.total_compressed_size += entry.compressed_size;
        }
        // Serialise manifest
        let manifest_bytes = self.manifest.serialise();
        // Build file: magic + version + manifest_len + manifest + data
        let mut output = Vec::new();
        output.extend_from_slice(&BUNDLE_MAGIC);
        output.extend_from_slice(&BUNDLE_FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
        output.extend_from_slice(&manifest_bytes);
        // Write compressed data blobs in order
        let mut current_write_offset: u64 = 0;
        for (key, blob) in &compressed_blobs {
            let entry = self.manifest.entries.get(key).unwrap();
            let padding = (entry.data_offset - current_write_offset) as usize;
            output.extend(std::iter::repeat(0u8).take(padding));
            output.extend_from_slice(blob);
            current_write_offset = entry.data_offset + entry.compressed_size;
        }
        // Append file checksum
        let file_crc = crc.checksum(&output);
        output.extend_from_slice(&file_crc.to_le_bytes());
        Ok(output)
    }

    /// Convenience: build and write directly to a writer.
    pub fn build_to_writer<W: Write>(&mut self, writer: &mut W) -> Result<usize, BundleError> {
        let data = self.build()?;
        writer.write_all(&data)?;
        Ok(data.len())
    }
}

// ---------------------------------------------------------------------------
// Bundle reader
// ---------------------------------------------------------------------------

/// Reads assets from a previously built bundle.
pub struct BundleReader {
    /// The parsed manifest.
    pub manifest: BundleManifest,
    /// The raw bundle data (for in-memory bundles).
    data: Vec<u8>,
    /// Byte offset where the data section begins within `data`.
    data_section_offset: usize,
    /// CRC calculator for verification.
    crc: Crc32,
}

impl BundleReader {
    /// Open a bundle from an in-memory byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BundleError> {
        if bytes.len() < 20 {
            return Err(BundleError::Truncated);
        }
        // Check magic
        if bytes[..8] != BUNDLE_MAGIC {
            return Err(BundleError::InvalidMagic);
        }
        // Version
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version > BUNDLE_FORMAT_VERSION {
            return Err(BundleError::UnsupportedVersion(version));
        }
        // Manifest length
        let manifest_len = u64::from_le_bytes([
            bytes[12], bytes[13], bytes[14], bytes[15],
            bytes[16], bytes[17], bytes[18], bytes[19],
        ]) as usize;
        let manifest_start = 20;
        let manifest_end = manifest_start + manifest_len;
        if manifest_end > bytes.len() {
            return Err(BundleError::Truncated);
        }
        let manifest = BundleManifest::deserialise(&bytes[manifest_start..manifest_end])?;
        Ok(Self {
            manifest,
            data: bytes.to_vec(),
            data_section_offset: manifest_end,
            crc: Crc32::new(),
        })
    }

    /// Read and decompress an asset by key.
    pub fn read_asset(&self, key: &str) -> Result<Vec<u8>, BundleError> {
        let entry = self
            .manifest
            .get_entry(key)
            .ok_or_else(|| BundleError::AssetNotFound(key.to_string()))?;
        let abs_offset = self.data_section_offset as u64 + entry.data_offset;
        let end = abs_offset + entry.compressed_size;
        if end as usize > self.data.len() {
            return Err(BundleError::Truncated);
        }
        let compressed = &self.data[abs_offset as usize..end as usize];
        let raw = match entry.codec {
            CompressionCodec::None => compressed.to_vec(),
            CompressionCodec::Lz4 | CompressionCodec::Lz4Hc => {
                Lz4Codec::decompress(compressed, entry.original_size as usize)?
            }
            CompressionCodec::Zstd { .. } => {
                ZstdCodec::decompress(compressed, entry.original_size as usize)?
            }
        };
        // Verify checksum
        let actual_crc = self.crc.checksum(&raw);
        if actual_crc != entry.checksum {
            return Err(BundleError::ChecksumMismatch {
                expected: entry.checksum,
                actual: actual_crc,
            });
        }
        Ok(raw)
    }

    /// Check if the bundle contains a given key.
    pub fn contains(&self, key: &str) -> bool {
        self.manifest.entries.contains_key(key)
    }

    /// Return an iterator over all asset keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.manifest.entries.keys().map(|s| s.as_str())
    }

    /// Number of assets.
    pub fn asset_count(&self) -> usize {
        self.manifest.asset_count()
    }
}

// ---------------------------------------------------------------------------
// Bundle dependency graph
// ---------------------------------------------------------------------------

/// Tracks the dependency graph among loaded bundles to detect cycles and
/// determine correct load order.
pub struct BundleDependencyGraph {
    /// Adjacency list: bundle_id -> list of bundles it depends on.
    edges: HashMap<String, Vec<String>>,
}

impl BundleDependencyGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Register a bundle and its dependencies.
    pub fn add_bundle(&mut self, bundle_id: &str, dependencies: &[String]) {
        self.edges
            .insert(bundle_id.to_string(), dependencies.to_vec());
    }

    /// Remove a bundle from the graph.
    pub fn remove_bundle(&mut self, bundle_id: &str) {
        self.edges.remove(bundle_id);
    }

    /// Perform a topological sort to get the load order. Returns an error
    /// if a cycle is detected.
    pub fn load_order(&self) -> Result<Vec<String>, BundleError> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for (node, deps) in &self.edges {
            in_degree.entry(node.as_str()).or_insert(0);
            for dep in deps {
                *in_degree.entry(dep.as_str()).or_insert(0) += 0;
                // dep is depended upon by `node` -- but in_degree tracks
                // how many *dependencies* a node has that are still not loaded.
                // Actually, for topological sort on a DAG we track how many
                // times a node appears as a dependency.
            }
        }
        // Kahn's algorithm
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut indeg: HashMap<&str, usize> = HashMap::new();
        let all_nodes: HashSet<&str> = self
            .edges
            .keys()
            .map(|s| s.as_str())
            .chain(self.edges.values().flatten().map(|s| s.as_str()))
            .collect();
        for &n in &all_nodes {
            indeg.entry(n).or_insert(0);
            adj.entry(n).or_insert_with(Vec::new);
        }
        for (node, deps) in &self.edges {
            for dep in deps {
                adj.entry(dep.as_str())
                    .or_insert_with(Vec::new)
                    .push(node.as_str());
                *indeg.entry(node.as_str()).or_insert(0) += 1;
            }
        }
        let mut queue: VecDeque<&str> = VecDeque::new();
        for (&n, &deg) in &indeg {
            if deg == 0 {
                queue.push_back(n);
            }
        }
        let mut order = Vec::new();
        while let Some(n) = queue.pop_front() {
            order.push(n.to_string());
            if let Some(successors) = adj.get(n) {
                for &succ in successors {
                    if let Some(d) = indeg.get_mut(succ) {
                        *d -= 1;
                        if *d == 0 {
                            queue.push_back(succ);
                        }
                    }
                }
            }
        }
        if order.len() != all_nodes.len() {
            // Cycle detected -- find it for the error message.
            let in_order: HashSet<&str> = order.iter().map(|s| s.as_str()).collect();
            let cycle_nodes: Vec<String> = all_nodes
                .iter()
                .filter(|n| !in_order.contains(**n))
                .map(|n| n.to_string())
                .collect();
            return Err(BundleError::DependencyCycle(cycle_nodes));
        }
        Ok(order)
    }

    /// Check for cycles without returning the full load order.
    pub fn has_cycle(&self) -> bool {
        self.load_order().is_err()
    }
}

impl Default for BundleDependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Streaming bundle loader
// ---------------------------------------------------------------------------

/// Request to stream-load a specific asset from a bundle.
#[derive(Debug, Clone)]
pub struct StreamRequest {
    /// The bundle to load from.
    pub bundle_id: String,
    /// The asset key within that bundle.
    pub asset_key: String,
    /// Load priority.
    pub priority: LoadPriority,
    /// When the request was enqueued.
    pub enqueued_at: Instant,
}

/// State of a streaming asset load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// Waiting in the queue.
    Queued,
    /// Currently being read from disk.
    Reading,
    /// Currently being decompressed.
    Decompressing,
    /// Load complete; data is available.
    Ready,
    /// An error occurred.
    Failed,
}

/// Result of a completed stream load.
#[derive(Debug, Clone)]
pub struct StreamResult {
    /// The asset key.
    pub asset_key: String,
    /// Current state.
    pub state: StreamState,
    /// The decompressed data (only if state == Ready).
    pub data: Option<Vec<u8>>,
    /// Error message (only if state == Failed).
    pub error: Option<String>,
    /// Time spent loading, if completed.
    pub load_time: Option<Duration>,
}

/// Manages asynchronous streaming of assets from bundles.
pub struct BundleStreamLoader {
    /// Registered bundle readers.
    bundles: HashMap<String, Arc<BundleReader>>,
    /// Pending stream requests, sorted by priority.
    queue: VecDeque<StreamRequest>,
    /// Completed results.
    results: HashMap<String, StreamResult>,
    /// Maximum number of concurrent in-flight reads.
    max_concurrent: usize,
    /// Number of currently in-flight reads.
    in_flight: usize,
    /// Total bytes loaded so far.
    total_bytes_loaded: u64,
    /// Total loads completed.
    total_loads_completed: u64,
}

impl BundleStreamLoader {
    /// Create a new stream loader.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            bundles: HashMap::new(),
            queue: VecDeque::new(),
            results: HashMap::new(),
            max_concurrent,
            in_flight: 0,
            total_bytes_loaded: 0,
            total_loads_completed: 0,
        }
    }

    /// Register a bundle reader for streaming.
    pub fn register_bundle(&mut self, bundle_id: impl Into<String>, reader: Arc<BundleReader>) {
        self.bundles.insert(bundle_id.into(), reader);
    }

    /// Unregister a bundle.
    pub fn unregister_bundle(&mut self, bundle_id: &str) {
        self.bundles.remove(bundle_id);
    }

    /// Enqueue a streaming request.
    pub fn request(
        &mut self,
        bundle_id: impl Into<String>,
        asset_key: impl Into<String>,
        priority: LoadPriority,
    ) {
        let req = StreamRequest {
            bundle_id: bundle_id.into(),
            asset_key: asset_key.into(),
            priority,
            enqueued_at: Instant::now(),
        };
        // Insert sorted by priority (higher priority first).
        let pos = self
            .queue
            .iter()
            .position(|r| r.priority < priority)
            .unwrap_or(self.queue.len());
        self.queue.insert(pos, req);
    }

    /// Process pending requests (call once per frame or from a worker thread).
    pub fn update(&mut self) {
        while self.in_flight < self.max_concurrent {
            let req = match self.queue.pop_front() {
                Some(r) => r,
                None => break,
            };
            self.in_flight += 1;
            let start = Instant::now();
            let result = if let Some(reader) = self.bundles.get(&req.bundle_id) {
                match reader.read_asset(&req.asset_key) {
                    Ok(data) => {
                        self.total_bytes_loaded += data.len() as u64;
                        self.total_loads_completed += 1;
                        StreamResult {
                            asset_key: req.asset_key.clone(),
                            state: StreamState::Ready,
                            data: Some(data),
                            error: None,
                            load_time: Some(start.elapsed()),
                        }
                    }
                    Err(e) => StreamResult {
                        asset_key: req.asset_key.clone(),
                        state: StreamState::Failed,
                        data: None,
                        error: Some(e.to_string()),
                        load_time: Some(start.elapsed()),
                    },
                }
            } else {
                StreamResult {
                    asset_key: req.asset_key.clone(),
                    state: StreamState::Failed,
                    data: None,
                    error: Some(format!("bundle '{}' not registered", req.bundle_id)),
                    load_time: Some(start.elapsed()),
                }
            };
            self.results.insert(req.asset_key, result);
            self.in_flight -= 1;
        }
    }

    /// Check if a result is available for the given asset key.
    pub fn poll(&self, asset_key: &str) -> Option<&StreamResult> {
        self.results.get(asset_key)
    }

    /// Take a completed result (removes it from the internal map).
    pub fn take_result(&mut self, asset_key: &str) -> Option<StreamResult> {
        self.results.remove(asset_key)
    }

    /// Number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Statistics: total bytes loaded.
    pub fn bytes_loaded(&self) -> u64 {
        self.total_bytes_loaded
    }

    /// Statistics: total loads completed.
    pub fn loads_completed(&self) -> u64 {
        self.total_loads_completed
    }
}

// ---------------------------------------------------------------------------
// DLC bundle
// ---------------------------------------------------------------------------

/// Represents downloadable content that extends the base game with new assets.
#[derive(Debug, Clone)]
pub struct DlcBundle {
    /// Display name shown to the user.
    pub display_name: String,
    /// Unique DLC identifier (matches the bundle ID).
    pub dlc_id: String,
    /// Version string (e.g. "1.0.0").
    pub version: String,
    /// Minimum engine version required.
    pub min_engine_version: String,
    /// Size of the download in bytes.
    pub download_size: u64,
    /// Size once installed (uncompressed).
    pub installed_size: u64,
    /// Whether this DLC is currently installed.
    pub installed: bool,
    /// Whether this DLC is currently enabled.
    pub enabled: bool,
    /// Path to the bundle file on disk (once installed).
    pub bundle_path: Option<PathBuf>,
    /// Asset keys provided by this DLC.
    pub provided_assets: Vec<String>,
    /// Other DLCs this one depends on.
    pub required_dlc: Vec<String>,
    /// Description / marketing text.
    pub description: String,
}

impl DlcBundle {
    /// Create a new DLC descriptor.
    pub fn new(
        dlc_id: impl Into<String>,
        display_name: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            display_name: display_name.into(),
            dlc_id: dlc_id.into(),
            version: version.into(),
            min_engine_version: String::new(),
            download_size: 0,
            installed_size: 0,
            installed: false,
            enabled: false,
            bundle_path: None,
            provided_assets: Vec::new(),
            required_dlc: Vec::new(),
            description: String::new(),
        }
    }

    /// Mark the DLC as installed at the given path.
    pub fn mark_installed(&mut self, path: PathBuf, installed_size: u64) {
        self.installed = true;
        self.installed_size = installed_size;
        self.bundle_path = Some(path);
    }

    /// Enable the DLC (assets will be visible to the asset system).
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the DLC.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

/// Manages all installed DLC bundles.
pub struct DlcManager {
    /// Known DLC bundles indexed by DLC ID.
    dlc_bundles: HashMap<String, DlcBundle>,
    /// Load order (topologically sorted).
    load_order: Vec<String>,
}

impl DlcManager {
    /// Create a new DLC manager.
    pub fn new() -> Self {
        Self {
            dlc_bundles: HashMap::new(),
            load_order: Vec::new(),
        }
    }

    /// Register a DLC bundle.
    pub fn register(&mut self, dlc: DlcBundle) {
        self.dlc_bundles.insert(dlc.dlc_id.clone(), dlc);
    }

    /// Unregister a DLC bundle.
    pub fn unregister(&mut self, dlc_id: &str) -> Option<DlcBundle> {
        self.dlc_bundles.remove(dlc_id)
    }

    /// Get a DLC by ID.
    pub fn get(&self, dlc_id: &str) -> Option<&DlcBundle> {
        self.dlc_bundles.get(dlc_id)
    }

    /// Get a mutable reference to a DLC by ID.
    pub fn get_mut(&mut self, dlc_id: &str) -> Option<&mut DlcBundle> {
        self.dlc_bundles.get_mut(dlc_id)
    }

    /// Return all registered DLC IDs.
    pub fn all_ids(&self) -> Vec<&str> {
        self.dlc_bundles.keys().map(|s| s.as_str()).collect()
    }

    /// Return only installed and enabled DLC IDs.
    pub fn active_ids(&self) -> Vec<&str> {
        self.dlc_bundles
            .values()
            .filter(|d| d.installed && d.enabled)
            .map(|d| d.dlc_id.as_str())
            .collect()
    }

    /// Compute the load order respecting DLC inter-dependencies.
    pub fn compute_load_order(&mut self) -> Result<(), BundleError> {
        let mut graph = BundleDependencyGraph::new();
        for dlc in self.dlc_bundles.values() {
            if dlc.installed && dlc.enabled {
                graph.add_bundle(&dlc.dlc_id, &dlc.required_dlc);
            }
        }
        self.load_order = graph.load_order()?;
        Ok(())
    }

    /// Return the current load order (must call `compute_load_order` first).
    pub fn load_order(&self) -> &[String] {
        &self.load_order
    }
}

impl Default for DlcManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Patch bundle
// ---------------------------------------------------------------------------

/// A patch bundle replaces or adds assets in a base bundle without modifying
/// the original file.
#[derive(Debug, Clone)]
pub struct PatchEntry {
    /// The asset key being patched.
    pub asset_key: String,
    /// The action to perform.
    pub action: PatchAction,
    /// New compressed data (for Replace/Add actions).
    pub data: Option<Vec<u8>>,
    /// Original size of the new data.
    pub original_size: u64,
    /// CRC of the new uncompressed data.
    pub checksum: u32,
}

/// What to do with a patched asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchAction {
    /// Replace an existing asset with new data.
    Replace,
    /// Add a new asset that doesn't exist in the base.
    Add,
    /// Remove an asset from the base bundle.
    Remove,
}

/// Applies patch bundles on top of a base bundle.
pub struct PatchApplicator {
    /// The base bundle reader.
    base: Arc<BundleReader>,
    /// Patch entries indexed by asset key (later patches override earlier ones).
    patches: BTreeMap<String, PatchEntry>,
    /// Order in which patch bundles were applied.
    patch_history: Vec<String>,
}

impl PatchApplicator {
    /// Create a new applicator for a base bundle.
    pub fn new(base: Arc<BundleReader>) -> Self {
        Self {
            base,
            patches: BTreeMap::new(),
            patch_history: Vec::new(),
        }
    }

    /// Apply a patch bundle on top of the current state.
    pub fn apply_patch(&mut self, patch_bundle: &BundleReader) -> Result<(), BundleError> {
        if !patch_bundle.manifest.is_patch {
            return Err(BundleError::CorruptManifest(
                "bundle is not marked as a patch".into(),
            ));
        }
        for (key, entry) in &patch_bundle.manifest.entries {
            let data = patch_bundle.read_asset(key)?;
            let crc = Crc32::new();
            let checksum = crc.checksum(&data);
            let patch_entry = PatchEntry {
                asset_key: key.clone(),
                action: PatchAction::Replace,
                data: Some(data.clone()),
                original_size: entry.original_size,
                checksum,
            };
            self.patches.insert(key.clone(), patch_entry);
        }
        self.patch_history
            .push(patch_bundle.manifest.bundle_id.clone());
        Ok(())
    }

    /// Add a single patch entry manually.
    pub fn add_patch_entry(&mut self, entry: PatchEntry) {
        self.patches.insert(entry.asset_key.clone(), entry);
    }

    /// Read an asset, checking patches first, then falling back to base.
    pub fn read_asset(&self, key: &str) -> Result<Vec<u8>, BundleError> {
        if let Some(patch) = self.patches.get(key) {
            match patch.action {
                PatchAction::Remove => Err(BundleError::AssetNotFound(key.to_string())),
                PatchAction::Replace | PatchAction::Add => patch
                    .data
                    .clone()
                    .ok_or_else(|| BundleError::AssetNotFound(key.to_string())),
            }
        } else {
            self.base.read_asset(key)
        }
    }

    /// Check if an asset exists (considering patches).
    pub fn contains(&self, key: &str) -> bool {
        if let Some(patch) = self.patches.get(key) {
            patch.action != PatchAction::Remove
        } else {
            self.base.contains(key)
        }
    }

    /// List all available asset keys (base + patches, minus removals).
    pub fn all_keys(&self) -> Vec<String> {
        let mut keys: HashSet<String> = self.base.keys().map(|s| s.to_string()).collect();
        for (key, patch) in &self.patches {
            match patch.action {
                PatchAction::Add | PatchAction::Replace => {
                    keys.insert(key.clone());
                }
                PatchAction::Remove => {
                    keys.remove(key);
                }
            }
        }
        let mut sorted: Vec<String> = keys.into_iter().collect();
        sorted.sort();
        sorted
    }

    /// Return the patch history.
    pub fn patch_history(&self) -> &[String] {
        &self.patch_history
    }
}

// ---------------------------------------------------------------------------
// Bundle file size analysis
// ---------------------------------------------------------------------------

/// Statistics about a bundle's content and compression.
#[derive(Debug, Clone)]
pub struct BundleSizeAnalysis {
    /// Total number of assets.
    pub asset_count: usize,
    /// Total uncompressed size in bytes.
    pub total_uncompressed: u64,
    /// Total compressed size in bytes.
    pub total_compressed: u64,
    /// Overall compression ratio.
    pub overall_ratio: f64,
    /// Per-type breakdown.
    pub by_type: HashMap<String, TypeSizeInfo>,
    /// Largest asset (key, uncompressed size).
    pub largest_asset: Option<(String, u64)>,
    /// Smallest asset (key, uncompressed size).
    pub smallest_asset: Option<(String, u64)>,
    /// Average asset size (uncompressed).
    pub average_size: f64,
}

/// Size information for a specific asset type.
#[derive(Debug, Clone)]
pub struct TypeSizeInfo {
    /// Number of assets of this type.
    pub count: usize,
    /// Total uncompressed size.
    pub total_uncompressed: u64,
    /// Total compressed size.
    pub total_compressed: u64,
    /// Compression ratio for this type.
    pub ratio: f64,
}

impl BundleSizeAnalysis {
    /// Analyse a bundle manifest.
    pub fn from_manifest(manifest: &BundleManifest) -> Self {
        let mut by_type: HashMap<String, TypeSizeInfo> = HashMap::new();
        let mut largest: Option<(String, u64)> = None;
        let mut smallest: Option<(String, u64)> = None;
        for entry in manifest.entries.values() {
            let info = by_type
                .entry(entry.asset_type.clone())
                .or_insert(TypeSizeInfo {
                    count: 0,
                    total_uncompressed: 0,
                    total_compressed: 0,
                    ratio: 1.0,
                });
            info.count += 1;
            info.total_uncompressed += entry.original_size;
            info.total_compressed += entry.compressed_size;
            match &largest {
                Some((_, s)) if entry.original_size > *s => {
                    largest = Some((entry.key.clone(), entry.original_size));
                }
                None => largest = Some((entry.key.clone(), entry.original_size)),
                _ => {}
            }
            match &smallest {
                Some((_, s)) if entry.original_size < *s => {
                    smallest = Some((entry.key.clone(), entry.original_size));
                }
                None => smallest = Some((entry.key.clone(), entry.original_size)),
                _ => {}
            }
        }
        for info in by_type.values_mut() {
            if info.total_uncompressed > 0 {
                info.ratio = info.total_compressed as f64 / info.total_uncompressed as f64;
            }
        }
        let asset_count = manifest.asset_count();
        let average_size = if asset_count > 0 {
            manifest.total_original_size as f64 / asset_count as f64
        } else {
            0.0
        };
        Self {
            asset_count,
            total_uncompressed: manifest.total_original_size,
            total_compressed: manifest.total_compressed_size,
            overall_ratio: manifest.compression_ratio(),
            by_type,
            largest_asset: largest,
            smallest_asset: smallest,
            average_size,
        }
    }

    /// Print a human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Assets: {}\n", self.asset_count));
        s.push_str(&format!(
            "Total size: {} bytes (compressed: {} bytes, ratio: {:.2}%)\n",
            self.total_uncompressed,
            self.total_compressed,
            self.overall_ratio * 100.0,
        ));
        s.push_str(&format!("Average asset size: {:.0} bytes\n", self.average_size));
        if let Some((ref key, size)) = self.largest_asset {
            s.push_str(&format!("Largest: {key} ({size} bytes)\n"));
        }
        if let Some((ref key, size)) = self.smallest_asset {
            s.push_str(&format!("Smallest: {key} ({size} bytes)\n"));
        }
        s.push_str("\nBy type:\n");
        for (type_name, info) in &self.by_type {
            s.push_str(&format!(
                "  {type_name}: {} assets, {} bytes -> {} bytes ({:.1}%)\n",
                info.count,
                info.total_uncompressed,
                info.total_compressed,
                info.ratio * 100.0,
            ));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Serialisation helpers
// ---------------------------------------------------------------------------

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, BundleError> {
    if *pos + 4 > data.len() {
        return Err(BundleError::Truncated);
    }
    let len =
        u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]) as usize;
    *pos += 4;
    if *pos + len > data.len() {
        return Err(BundleError::Truncated);
    }
    let s = String::from_utf8_lossy(&data[*pos..*pos + len]).into_owned();
    *pos += len;
    Ok(s)
}

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, BundleError> {
    if *pos >= data.len() {
        return Err(BundleError::Truncated);
    }
    let v = data[*pos];
    *pos += 1;
    Ok(v)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, BundleError> {
    if *pos + 4 > data.len() {
        return Err(BundleError::Truncated);
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_i32(data: &[u8], pos: &mut usize) -> Result<i32, BundleError> {
    if *pos + 4 > data.len() {
        return Err(BundleError::Truncated);
    }
    let v = i32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, BundleError> {
    if *pos + 8 > data.len() {
        return Err(BundleError::Truncated);
    }
    let v = u64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos += 8;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_known_value() {
        let crc = Crc32::new();
        let checksum = crc.checksum(b"hello world");
        // Known CRC-32 of "hello world"
        assert_eq!(checksum, 0x0D4A_1185);
    }

    #[test]
    fn test_crc32_empty() {
        let crc = Crc32::new();
        assert_eq!(crc.checksum(b""), 0x0000_0000);
    }

    #[test]
    fn test_lz4_roundtrip() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut compressed = Vec::new();
        Lz4Codec::compress(data, &mut compressed, 1);
        let decompressed = Lz4Codec::decompress(&compressed, data.len()).unwrap();
        assert_eq!(&decompressed, data);
    }

    #[test]
    fn test_zstd_roundtrip() {
        let data = b"AAABBBCCC repeated data for compression test";
        let mut compressed = Vec::new();
        ZstdCodec::compress(data, &mut compressed, 3);
        let decompressed = ZstdCodec::decompress(&compressed, data.len()).unwrap();
        assert_eq!(&decompressed, data);
    }

    #[test]
    fn test_bundle_entry_serialise_roundtrip() {
        let entry = BundleAssetEntry {
            key: "textures/test".into(),
            data_offset: 1024,
            compressed_size: 500,
            original_size: 1000,
            checksum: 0xDEAD_BEEF,
            codec: CompressionCodec::Lz4,
            asset_type: "texture/rgba".into(),
            dependencies: vec!["materials/base".into()],
            metadata: {
                let mut m = HashMap::new();
                m.insert("width".into(), "512".into());
                m
            },
            last_modified: 1700000000,
            index: 0,
        };
        let bytes = entry.serialise();
        let (decoded, consumed) = BundleAssetEntry::deserialise(&bytes).unwrap();
        assert_eq!(decoded.key, "textures/test");
        assert_eq!(decoded.data_offset, 1024);
        assert_eq!(decoded.compressed_size, 500);
        assert_eq!(decoded.original_size, 1000);
        assert_eq!(decoded.checksum, 0xDEAD_BEEF);
        assert_eq!(decoded.dependencies, vec!["materials/base"]);
        assert_eq!(decoded.metadata.get("width").unwrap(), "512");
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_bundle_builder_and_reader() {
        let mut builder = BundleBuilder::new("test_bundle");
        builder.set_bundle_id("test-001");
        builder.set_description("unit test bundle");
        builder
            .add_asset(
                "textures/hero",
                "texture/rgba",
                vec![0xAA; 1024],
                vec![],
            )
            .unwrap();
        builder
            .add_asset(
                "meshes/hero",
                "mesh/optimised",
                vec![0xBB; 2048],
                vec!["textures/hero".into()],
            )
            .unwrap();
        let bundle_data = builder.build().unwrap();
        let reader = BundleReader::from_bytes(&bundle_data).unwrap();
        assert_eq!(reader.asset_count(), 2);
        assert!(reader.contains("textures/hero"));
        assert!(reader.contains("meshes/hero"));
        let tex = reader.read_asset("textures/hero").unwrap();
        assert_eq!(tex.len(), 1024);
        assert!(tex.iter().all(|&b| b == 0xAA));
        let mesh = reader.read_asset("meshes/hero").unwrap();
        assert_eq!(mesh.len(), 2048);
        assert!(mesh.iter().all(|&b| b == 0xBB));
    }

    #[test]
    fn test_duplicate_asset_rejected() {
        let mut builder = BundleBuilder::new("dup_test");
        builder
            .add_asset("a/b", "type", vec![1, 2, 3], vec![])
            .unwrap();
        let err = builder.add_asset("a/b", "type", vec![4, 5, 6], vec![]);
        assert!(err.is_err());
    }

    #[test]
    fn test_dependency_graph_no_cycle() {
        let mut graph = BundleDependencyGraph::new();
        graph.add_bundle("A", &["B".into(), "C".into()]);
        graph.add_bundle("B", &["C".into()]);
        graph.add_bundle("C", &[]);
        let order = graph.load_order().unwrap();
        let pos_a = order.iter().position(|s| s == "A").unwrap();
        let pos_b = order.iter().position(|s| s == "B").unwrap();
        let pos_c = order.iter().position(|s| s == "C").unwrap();
        assert!(pos_c < pos_b);
        assert!(pos_b < pos_a);
    }

    #[test]
    fn test_dependency_graph_cycle_detected() {
        let mut graph = BundleDependencyGraph::new();
        graph.add_bundle("A", &["B".into()]);
        graph.add_bundle("B", &["A".into()]);
        assert!(graph.has_cycle());
    }

    #[test]
    fn test_patch_applicator() {
        // Build base bundle
        let mut base_builder = BundleBuilder::new("base");
        base_builder.set_bundle_id("base-001");
        base_builder
            .add_asset("data/config", "config/json", vec![1, 2, 3], vec![])
            .unwrap();
        let base_data = base_builder.build().unwrap();
        let base_reader = Arc::new(BundleReader::from_bytes(&base_data).unwrap());
        // Build patch bundle
        let mut patch_builder = BundleBuilder::new("patch");
        patch_builder.set_bundle_id("patch-001");
        patch_builder.set_patch_target("base-001");
        patch_builder
            .add_asset("data/config", "config/json", vec![4, 5, 6, 7], vec![])
            .unwrap();
        let patch_data = patch_builder.build().unwrap();
        let patch_reader = BundleReader::from_bytes(&patch_data).unwrap();
        // Apply patch
        let mut applicator = PatchApplicator::new(base_reader);
        applicator.apply_patch(&patch_reader).unwrap();
        let config = applicator.read_asset("data/config").unwrap();
        assert_eq!(config, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_stream_loader() {
        let mut builder = BundleBuilder::new("stream_test");
        builder.set_bundle_id("stream-001");
        builder
            .add_asset("a", "type", vec![10; 512], vec![])
            .unwrap();
        let data = builder.build().unwrap();
        let reader = Arc::new(BundleReader::from_bytes(&data).unwrap());
        let mut loader = BundleStreamLoader::new(4);
        loader.register_bundle("stream-001", reader);
        loader.request("stream-001", "a", LoadPriority::High);
        loader.update();
        let result = loader.poll("a").unwrap();
        assert_eq!(result.state, StreamState::Ready);
        assert_eq!(result.data.as_ref().unwrap().len(), 512);
    }

    #[test]
    fn test_bundle_size_analysis() {
        let mut manifest = BundleManifest::new("analysis_test");
        let mut e1 = BundleAssetEntry::new("tex/a", "texture");
        e1.original_size = 1000;
        e1.compressed_size = 500;
        manifest.add_entry(e1).unwrap();
        let mut e2 = BundleAssetEntry::new("tex/b", "texture");
        e2.original_size = 2000;
        e2.compressed_size = 800;
        manifest.add_entry(e2).unwrap();
        let mut e3 = BundleAssetEntry::new("mesh/c", "mesh");
        e3.original_size = 5000;
        e3.compressed_size = 3000;
        manifest.add_entry(e3).unwrap();
        let analysis = BundleSizeAnalysis::from_manifest(&manifest);
        assert_eq!(analysis.asset_count, 3);
        assert_eq!(analysis.total_uncompressed, 8000);
        assert_eq!(analysis.total_compressed, 4300);
        assert_eq!(analysis.by_type.get("texture").unwrap().count, 2);
        assert_eq!(analysis.by_type.get("mesh").unwrap().count, 1);
    }

    #[test]
    fn test_compression_codec_discriminant_roundtrip() {
        let codecs = vec![
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::Lz4Hc,
            CompressionCodec::Zstd { level: 5 },
        ];
        for codec in codecs {
            let d = codec.discriminant();
            let extra = match codec {
                CompressionCodec::Zstd { level } => level,
                _ => 0,
            };
            let decoded = CompressionCodec::from_discriminant(d, extra).unwrap();
            assert_eq!(decoded, codec);
        }
    }

    #[test]
    fn test_dlc_manager() {
        let mut mgr = DlcManager::new();
        let mut dlc = DlcBundle::new("dlc-forest", "Forest Pack", "1.0.0");
        dlc.mark_installed(PathBuf::from("/bundles/forest.bundle"), 50_000_000);
        dlc.enable();
        mgr.register(dlc);
        assert_eq!(mgr.active_ids().len(), 1);
        assert!(mgr.get("dlc-forest").unwrap().installed);
    }

    #[test]
    fn test_manifest_serialise_roundtrip() {
        let mut manifest = BundleManifest::new("roundtrip_test");
        manifest.bundle_id = "rt-001".into();
        manifest.description = "test description".into();
        manifest.bundle_dependencies = vec!["dep-A".into()];
        let mut entry = BundleAssetEntry::new("test/asset", "generic");
        entry.original_size = 42;
        entry.compressed_size = 30;
        entry.checksum = 0x1234;
        manifest.add_entry(entry).unwrap();
        let bytes = manifest.serialise();
        let decoded = BundleManifest::deserialise(&bytes).unwrap();
        assert_eq!(decoded.name, "roundtrip_test");
        assert_eq!(decoded.bundle_id, "rt-001");
        assert_eq!(decoded.description, "test description");
        assert_eq!(decoded.bundle_dependencies, vec!["dep-A"]);
        assert_eq!(decoded.asset_count(), 1);
        let e = decoded.get_entry("test/asset").unwrap();
        assert_eq!(e.original_size, 42);
        assert_eq!(e.checksum, 0x1234);
    }
}
