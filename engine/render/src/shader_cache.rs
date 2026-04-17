// engine/render/src/shader_cache.rs
//
// Compiled shader cache: hash shader source + defines, store compiled modules
// on disk, load from cache if hash matches, cache invalidation, cache size
// management, warm cache on startup.
//
// This module provides a persistent disk cache for compiled shader modules.
// Rather than recompiling every shader on startup or when materials change,
// the cache stores compiled SPIR-V / DXIL / MSL bytecode keyed by a hash of
// the original source text and all preprocessor defines. On subsequent loads,
// the cache is consulted first, and the compiled module is returned directly
// if the hash matches.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes at the start of every cache entry file.
pub const CACHE_MAGIC: [u8; 4] = [b'S', b'H', b'C', 0x01];

/// Current cache format version.
pub const CACHE_VERSION: u32 = 1;

/// Default maximum cache size on disk in bytes (256 MB).
pub const DEFAULT_MAX_CACHE_SIZE: u64 = 256 * 1024 * 1024;

/// Default maximum age for cache entries before eviction (30 days).
pub const DEFAULT_MAX_AGE_SECS: u64 = 30 * 24 * 3600;

/// File extension for cache entries.
pub const CACHE_EXTENSION: &str = "shcache";

// ---------------------------------------------------------------------------
// Hash utilities
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Compute a combined hash of shader source and a sorted list of defines.
pub fn compute_shader_hash(source: &str, defines: &[(String, String)]) -> u64 {
    let mut hasher_input = source.to_string();
    hasher_input.push('\0');

    // Sort defines for deterministic hashing.
    let mut sorted_defines: Vec<(&str, &str)> = defines.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    sorted_defines.sort();
    for (key, val) in &sorted_defines {
        hasher_input.push_str(key);
        hasher_input.push('=');
        hasher_input.push_str(val);
        hasher_input.push('\0');
    }

    fnv1a_64(hasher_input.as_bytes())
}

/// Compute a hash from raw bytes (for binary shader sources like SPIR-V).
pub fn compute_binary_hash(data: &[u8], defines: &[(String, String)]) -> u64 {
    let mut combined = Vec::with_capacity(data.len() + 256);
    combined.extend_from_slice(data);
    combined.push(0);
    let mut sorted_defines: Vec<(&str, &str)> = defines.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    sorted_defines.sort();
    for (key, val) in &sorted_defines {
        combined.extend_from_slice(key.as_bytes());
        combined.push(b'=');
        combined.extend_from_slice(val.as_bytes());
        combined.push(0);
    }
    fnv1a_64(&combined)
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// Metadata for a cached shader entry.
#[derive(Debug, Clone)]
pub struct ShaderCacheEntry {
    /// Hash of the source + defines that produced this entry.
    pub source_hash: u64,
    /// The compiled shader bytecode.
    pub bytecode: Vec<u8>,
    /// Shader stage (vertex, fragment, compute, etc.).
    pub stage: ShaderStageKind,
    /// Target backend (SPIRV, DXIL, MSL, WGSL).
    pub target: ShaderTarget,
    /// Entry point name.
    pub entry_point: String,
    /// Time the entry was created.
    pub created_at: SystemTime,
    /// Time the entry was last accessed.
    pub last_accessed: SystemTime,
    /// Number of times this entry has been loaded from cache.
    pub access_count: u32,
    /// Size of the bytecode in bytes.
    pub bytecode_size: u64,
    /// Original source file path (for diagnostics).
    pub source_path: Option<String>,
    /// Defines used to compile this variant.
    pub defines: Vec<(String, String)>,
}

/// Shader stage types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStageKind {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    TessControl,
    TessEval,
    Mesh,
    Task,
    RayGen,
    RayClosestHit,
    RayAnyHit,
    RayMiss,
    RayIntersection,
}

impl ShaderStageKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Compute => "compute",
            Self::Geometry => "geometry",
            Self::TessControl => "tess_control",
            Self::TessEval => "tess_eval",
            Self::Mesh => "mesh",
            Self::Task => "task",
            Self::RayGen => "ray_gen",
            Self::RayClosestHit => "ray_closest_hit",
            Self::RayAnyHit => "ray_any_hit",
            Self::RayMiss => "ray_miss",
            Self::RayIntersection => "ray_intersection",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "vertex" => Some(Self::Vertex),
            "fragment" => Some(Self::Fragment),
            "compute" => Some(Self::Compute),
            "geometry" => Some(Self::Geometry),
            "tess_control" => Some(Self::TessControl),
            "tess_eval" => Some(Self::TessEval),
            "mesh" => Some(Self::Mesh),
            "task" => Some(Self::Task),
            "ray_gen" => Some(Self::RayGen),
            "ray_closest_hit" => Some(Self::RayClosestHit),
            "ray_any_hit" => Some(Self::RayAnyHit),
            "ray_miss" => Some(Self::RayMiss),
            "ray_intersection" => Some(Self::RayIntersection),
            _ => None,
        }
    }
}

/// Target shader compilation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderTarget {
    SpirV,
    Dxil,
    Msl,
    Wgsl,
    Glsl,
}

impl ShaderTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SpirV => "spirv",
            Self::Dxil => "dxil",
            Self::Msl => "msl",
            Self::Wgsl => "wgsl",
            Self::Glsl => "glsl",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "spirv" => Some(Self::SpirV),
            "dxil" => Some(Self::Dxil),
            "msl" => Some(Self::Msl),
            "wgsl" => Some(Self::Wgsl),
            "glsl" => Some(Self::Glsl),
            _ => None,
        }
    }
}

impl ShaderCacheEntry {
    /// Serialize the entry to bytes for disk storage.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256 + self.bytecode.len());
        // Magic.
        buf.extend_from_slice(&CACHE_MAGIC);
        // Version.
        buf.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        // Source hash.
        buf.extend_from_slice(&self.source_hash.to_le_bytes());
        // Stage.
        let stage_str = self.stage.as_str();
        buf.extend_from_slice(&(stage_str.len() as u32).to_le_bytes());
        buf.extend_from_slice(stage_str.as_bytes());
        // Target.
        let target_str = self.target.as_str();
        buf.extend_from_slice(&(target_str.len() as u32).to_le_bytes());
        buf.extend_from_slice(target_str.as_bytes());
        // Entry point.
        buf.extend_from_slice(&(self.entry_point.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.entry_point.as_bytes());
        // Access count.
        buf.extend_from_slice(&self.access_count.to_le_bytes());
        // Bytecode length.
        buf.extend_from_slice(&(self.bytecode.len() as u64).to_le_bytes());
        // Bytecode.
        buf.extend_from_slice(&self.bytecode);
        // Source path.
        let sp = self.source_path.as_deref().unwrap_or("");
        buf.extend_from_slice(&(sp.len() as u32).to_le_bytes());
        buf.extend_from_slice(sp.as_bytes());
        // Defines count.
        buf.extend_from_slice(&(self.defines.len() as u32).to_le_bytes());
        for (k, v) in &self.defines {
            buf.extend_from_slice(&(k.len() as u32).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        }
        // Checksum of everything so far.
        let checksum = fnv1a_64(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Deserialize an entry from bytes.
    pub fn deserialize(data: &[u8]) -> Result<Self, ShaderCacheError> {
        let mut pos = 0usize;

        let read_u32 = |data: &[u8], pos: &mut usize| -> Result<u32, ShaderCacheError> {
            if *pos + 4 > data.len() { return Err(ShaderCacheError::CorruptEntry); }
            let val = u32::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3]]);
            *pos += 4;
            Ok(val)
        };
        let read_u64 = |data: &[u8], pos: &mut usize| -> Result<u64, ShaderCacheError> {
            if *pos + 8 > data.len() { return Err(ShaderCacheError::CorruptEntry); }
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&data[*pos..*pos+8]);
            *pos += 8;
            Ok(u64::from_le_bytes(bytes))
        };
        let read_string = |data: &[u8], pos: &mut usize| -> Result<String, ShaderCacheError> {
            let len = read_u32(data, pos)? as usize;
            if *pos + len > data.len() { return Err(ShaderCacheError::CorruptEntry); }
            let s = std::str::from_utf8(&data[*pos..*pos+len])
                .map_err(|_| ShaderCacheError::CorruptEntry)?
                .to_string();
            *pos += len;
            Ok(s)
        };

        // Magic.
        if data.len() < 4 || data[0..4] != CACHE_MAGIC {
            return Err(ShaderCacheError::InvalidMagic);
        }
        pos += 4;

        // Version.
        let version = read_u32(data, &mut pos)?;
        if version != CACHE_VERSION {
            return Err(ShaderCacheError::VersionMismatch { expected: CACHE_VERSION, found: version });
        }

        // Source hash.
        let source_hash = read_u64(data, &mut pos)?;
        // Stage.
        let stage_str = read_string(data, &mut pos)?;
        let stage = ShaderStageKind::from_str(&stage_str).ok_or(ShaderCacheError::CorruptEntry)?;
        // Target.
        let target_str = read_string(data, &mut pos)?;
        let target = ShaderTarget::from_str(&target_str).ok_or(ShaderCacheError::CorruptEntry)?;
        // Entry point.
        let entry_point = read_string(data, &mut pos)?;
        // Access count.
        let access_count = read_u32(data, &mut pos)?;
        // Bytecode.
        let bytecode_len = read_u64(data, &mut pos)? as usize;
        if pos + bytecode_len > data.len() { return Err(ShaderCacheError::CorruptEntry); }
        let bytecode = data[pos..pos + bytecode_len].to_vec();
        pos += bytecode_len;
        // Source path.
        let sp = read_string(data, &mut pos)?;
        let source_path = if sp.is_empty() { None } else { Some(sp) };
        // Defines.
        let def_count = read_u32(data, &mut pos)? as usize;
        let mut defines = Vec::with_capacity(def_count);
        for _ in 0..def_count {
            let k = read_string(data, &mut pos)?;
            let v = read_string(data, &mut pos)?;
            defines.push((k, v));
        }
        // Checksum.
        if pos + 8 > data.len() { return Err(ShaderCacheError::CorruptEntry); }
        let stored_checksum = read_u64(data, &mut pos)?;
        let computed_checksum = fnv1a_64(&data[..pos - 8]);
        if stored_checksum != computed_checksum {
            return Err(ShaderCacheError::ChecksumMismatch);
        }

        Ok(Self {
            source_hash,
            bytecode,
            stage,
            target,
            entry_point,
            created_at: SystemTime::UNIX_EPOCH,
            last_accessed: SystemTime::now(),
            access_count,
            bytecode_size: bytecode_len as u64,
            source_path,
            defines,
        })
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the shader cache.
#[derive(Debug, Clone)]
pub enum ShaderCacheError {
    /// File I/O error.
    IoError(String),
    /// Cache entry has invalid magic bytes.
    InvalidMagic,
    /// Cache version mismatch.
    VersionMismatch { expected: u32, found: u32 },
    /// Entry data is corrupt.
    CorruptEntry,
    /// Checksum does not match.
    ChecksumMismatch,
    /// Entry not found in cache.
    NotFound,
    /// Cache is full (size limit reached).
    CacheFull,
}

impl std::fmt::Display for ShaderCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
            Self::InvalidMagic => write!(f, "Invalid cache magic bytes"),
            Self::VersionMismatch { expected, found } => write!(f, "Version mismatch: expected {}, found {}", expected, found),
            Self::CorruptEntry => write!(f, "Corrupt cache entry"),
            Self::ChecksumMismatch => write!(f, "Checksum mismatch"),
            Self::NotFound => write!(f, "Entry not found"),
            Self::CacheFull => write!(f, "Cache full"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shader cache
// ---------------------------------------------------------------------------

/// Configuration for the shader cache.
#[derive(Debug, Clone)]
pub struct ShaderCacheConfig {
    /// Root directory for the cache on disk.
    pub cache_dir: PathBuf,
    /// Maximum total size of the cache in bytes.
    pub max_cache_size: u64,
    /// Maximum age for cache entries.
    pub max_entry_age: Duration,
    /// Whether to enable the disk cache.
    pub disk_cache_enabled: bool,
    /// Whether to enable the in-memory cache.
    pub memory_cache_enabled: bool,
    /// Maximum number of entries in the in-memory cache.
    pub max_memory_entries: usize,
    /// Whether to validate checksums on load.
    pub validate_checksums: bool,
    /// Whether to compress cache entries on disk.
    pub compress_entries: bool,
}

impl Default for ShaderCacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from("shader_cache"),
            max_cache_size: DEFAULT_MAX_CACHE_SIZE,
            max_entry_age: Duration::from_secs(DEFAULT_MAX_AGE_SECS),
            disk_cache_enabled: true,
            memory_cache_enabled: true,
            max_memory_entries: 1024,
            validate_checksums: true,
            compress_entries: false,
        }
    }
}

/// Statistics about the shader cache.
#[derive(Debug, Clone, Default)]
pub struct ShaderCacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries in memory.
    pub memory_entries: usize,
    /// Number of entries on disk.
    pub disk_entries: usize,
    /// Total size of the disk cache in bytes.
    pub disk_size: u64,
    /// Number of evictions.
    pub evictions: u64,
    /// Number of invalidations.
    pub invalidations: u64,
    /// Total bytes loaded from cache.
    pub bytes_loaded: u64,
    /// Total bytes stored to cache.
    pub bytes_stored: u64,
    /// Number of checksum failures.
    pub checksum_failures: u64,
}

impl ShaderCacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// The shader cache.
///
/// Stores compiled shader bytecode keyed by a hash of the source and defines.
/// Supports both in-memory and disk caching.
pub struct ShaderCache {
    config: ShaderCacheConfig,
    /// In-memory cache: hash -> entry.
    memory_cache: HashMap<u64, ShaderCacheEntry>,
    /// LRU order for memory cache eviction: most recently used at the back.
    lru_order: Vec<u64>,
    /// Known disk entries: hash -> file path.
    disk_index: HashMap<u64, PathBuf>,
    /// Running statistics.
    pub stats: ShaderCacheStats,
}

impl ShaderCache {
    /// Create a new shader cache with the given configuration.
    pub fn new(config: ShaderCacheConfig) -> Self {
        Self {
            config,
            memory_cache: HashMap::new(),
            lru_order: Vec::new(),
            disk_index: HashMap::new(),
            stats: ShaderCacheStats::default(),
        }
    }

    /// Initialize the cache: scan the disk cache directory and build the index.
    pub fn initialize(&mut self) -> Result<(), ShaderCacheError> {
        if !self.config.disk_cache_enabled { return Ok(()); }

        // Create the cache directory if it doesn't exist.
        if !self.config.cache_dir.exists() {
            std::fs::create_dir_all(&self.config.cache_dir)
                .map_err(|e| ShaderCacheError::IoError(e.to_string()))?;
        }

        // Scan for existing cache files.
        let entries = std::fs::read_dir(&self.config.cache_dir)
            .map_err(|e| ShaderCacheError::IoError(e.to_string()))?;

        let mut total_size = 0u64;
        for entry in entries {
            let entry = entry.map_err(|e| ShaderCacheError::IoError(e.to_string()))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some(CACHE_EXTENSION) {
                // Extract hash from filename.
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(hash) = u64::from_str_radix(stem, 16) {
                        let metadata = std::fs::metadata(&path)
                            .map_err(|e| ShaderCacheError::IoError(e.to_string()))?;
                        total_size += metadata.len();
                        self.disk_index.insert(hash, path);
                    }
                }
            }
        }
        self.stats.disk_entries = self.disk_index.len();
        self.stats.disk_size = total_size;
        Ok(())
    }

    /// Look up a cached shader by source hash.
    pub fn get(&mut self, source_hash: u64) -> Result<ShaderCacheEntry, ShaderCacheError> {
        // Check memory cache first.
        if self.config.memory_cache_enabled {
            if let Some(entry) = self.memory_cache.get_mut(&source_hash) {
                entry.last_accessed = SystemTime::now();
                entry.access_count += 1;
                let result = entry.clone();
                let bytecode_size = result.bytecode_size;
                self.stats.hits += 1;
                self.stats.bytes_loaded += bytecode_size;
                self.touch_lru(source_hash);
                return Ok(result);
            }
        }

        // Check disk cache.
        if self.config.disk_cache_enabled {
            if let Some(path) = self.disk_index.get(&source_hash).cloned() {
                let data = std::fs::read(&path)
                    .map_err(|e| ShaderCacheError::IoError(e.to_string()))?;
                let mut entry = ShaderCacheEntry::deserialize(&data)?;
                entry.last_accessed = SystemTime::now();
                entry.access_count += 1;
                self.stats.hits += 1;
                self.stats.bytes_loaded += entry.bytecode_size;

                // Promote to memory cache.
                if self.config.memory_cache_enabled {
                    self.insert_memory(source_hash, entry.clone());
                }
                return Ok(entry);
            }
        }

        self.stats.misses += 1;
        Err(ShaderCacheError::NotFound)
    }

    /// Store a compiled shader in the cache.
    pub fn store(&mut self, entry: ShaderCacheEntry) -> Result<(), ShaderCacheError> {
        let hash = entry.source_hash;
        let bytecode_size = entry.bytecode_size;

        // Store in memory cache.
        if self.config.memory_cache_enabled {
            self.insert_memory(hash, entry.clone());
        }

        // Store on disk.
        if self.config.disk_cache_enabled {
            let data = entry.serialize();
            let filename = format!("{:016x}.{}", hash, CACHE_EXTENSION);
            let path = self.config.cache_dir.join(&filename);
            std::fs::write(&path, &data)
                .map_err(|e| ShaderCacheError::IoError(e.to_string()))?;
            self.disk_index.insert(hash, path);
            self.stats.disk_entries = self.disk_index.len();
            self.stats.disk_size += data.len() as u64;
            self.stats.bytes_stored += bytecode_size;
        }

        // Check if cache size exceeded.
        if self.stats.disk_size > self.config.max_cache_size {
            self.evict_oldest_disk_entries()?;
        }

        Ok(())
    }

    /// Invalidate a specific entry.
    pub fn invalidate(&mut self, source_hash: u64) -> Result<(), ShaderCacheError> {
        self.memory_cache.remove(&source_hash);
        self.lru_order.retain(|&h| h != source_hash);

        if let Some(path) = self.disk_index.remove(&source_hash) {
            if path.exists() {
                let metadata = std::fs::metadata(&path)
                    .map_err(|e| ShaderCacheError::IoError(e.to_string()))?;
                self.stats.disk_size = self.stats.disk_size.saturating_sub(metadata.len());
                std::fs::remove_file(&path)
                    .map_err(|e| ShaderCacheError::IoError(e.to_string()))?;
            }
        }
        self.stats.invalidations += 1;
        self.stats.disk_entries = self.disk_index.len();
        self.stats.memory_entries = self.memory_cache.len();
        Ok(())
    }

    /// Invalidate all entries.
    pub fn invalidate_all(&mut self) -> Result<(), ShaderCacheError> {
        self.memory_cache.clear();
        self.lru_order.clear();
        let hashes: Vec<u64> = self.disk_index.keys().copied().collect();
        for hash in hashes {
            if let Some(path) = self.disk_index.remove(&hash) {
                if path.exists() {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        self.stats.invalidations += 1;
        self.stats.disk_entries = 0;
        self.stats.disk_size = 0;
        self.stats.memory_entries = 0;
        Ok(())
    }

    /// Warm the memory cache by loading all disk entries.
    pub fn warm_cache(&mut self) -> Result<u32, ShaderCacheError> {
        if !self.config.memory_cache_enabled || !self.config.disk_cache_enabled {
            return Ok(0);
        }

        let paths: Vec<(u64, PathBuf)> = self.disk_index.iter().map(|(h, p)| (*h, p.clone())).collect();
        let mut loaded = 0u32;
        for (hash, path) in paths {
            if self.memory_cache.contains_key(&hash) { continue; }
            if let Ok(data) = std::fs::read(&path) {
                if let Ok(entry) = ShaderCacheEntry::deserialize(&data) {
                    self.insert_memory(hash, entry);
                    loaded += 1;
                }
            }
        }
        Ok(loaded)
    }

    /// Evict expired entries from the disk cache.
    pub fn evict_expired(&mut self) -> Result<u32, ShaderCacheError> {
        let now = SystemTime::now();
        let max_age = self.config.max_entry_age;
        let mut to_remove = Vec::new();

        for (hash, path) in &self.disk_index {
            if let Ok(metadata) = std::fs::metadata(path) {
                if let Ok(modified) = metadata.modified() {
                    if let Ok(age) = now.duration_since(modified) {
                        if age > max_age {
                            to_remove.push(*hash);
                        }
                    }
                }
            }
        }

        let count = to_remove.len() as u32;
        for hash in to_remove {
            self.invalidate(hash)?;
            self.stats.evictions += 1;
        }
        Ok(count)
    }

    /// Check whether a particular shader is cached.
    pub fn contains(&self, source_hash: u64) -> bool {
        self.memory_cache.contains_key(&source_hash) || self.disk_index.contains_key(&source_hash)
    }

    /// Get the number of entries in the cache (memory + disk).
    pub fn entry_count(&self) -> usize {
        self.disk_index.len()
    }

    /// Get the total disk size of the cache.
    pub fn disk_size(&self) -> u64 {
        self.stats.disk_size
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn insert_memory(&mut self, hash: u64, entry: ShaderCacheEntry) {
        // Evict from memory if at capacity.
        while self.memory_cache.len() >= self.config.max_memory_entries {
            if let Some(oldest) = self.lru_order.first().copied() {
                self.lru_order.remove(0);
                self.memory_cache.remove(&oldest);
            } else {
                break;
            }
        }
        self.memory_cache.insert(hash, entry);
        self.lru_order.retain(|&h| h != hash);
        self.lru_order.push(hash);
        self.stats.memory_entries = self.memory_cache.len();
    }

    fn touch_lru(&mut self, hash: u64) {
        self.lru_order.retain(|&h| h != hash);
        self.lru_order.push(hash);
    }

    fn evict_oldest_disk_entries(&mut self) -> Result<(), ShaderCacheError> {
        // Sort disk entries by modification time and remove oldest until under limit.
        let mut entries: Vec<(u64, PathBuf, SystemTime)> = Vec::new();
        for (hash, path) in &self.disk_index {
            let mtime = std::fs::metadata(path)
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            entries.push((*hash, path.clone(), mtime));
        }
        entries.sort_by(|a, b| a.2.cmp(&b.2));

        while self.stats.disk_size > self.config.max_cache_size {
            if let Some((hash, _, _)) = entries.first() {
                let h = *hash;
                entries.remove(0);
                self.invalidate(h)?;
                self.stats.evictions += 1;
            } else {
                break;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shader compile request
// ---------------------------------------------------------------------------

/// A request to compile a shader (or look up from cache).
#[derive(Debug, Clone)]
pub struct ShaderCompileRequest {
    /// Shader source code.
    pub source: String,
    /// Preprocessor defines.
    pub defines: Vec<(String, String)>,
    /// Shader stage.
    pub stage: ShaderStageKind,
    /// Target backend.
    pub target: ShaderTarget,
    /// Entry point name.
    pub entry_point: String,
    /// Source file path (optional, for diagnostics).
    pub source_path: Option<String>,
}

impl ShaderCompileRequest {
    pub fn source_hash(&self) -> u64 {
        compute_shader_hash(&self.source, &self.defines)
    }
}

/// Result of a shader compilation.
#[derive(Debug)]
pub struct ShaderCompileResult {
    pub request: ShaderCompileRequest,
    pub bytecode: Vec<u8>,
    pub was_cached: bool,
    pub compile_time_ms: f64,
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Shader cache manager (high-level API)
// ---------------------------------------------------------------------------

/// High-level shader cache manager that wraps the cache with a compile callback.
pub struct ShaderCacheManager {
    pub cache: ShaderCache,
    /// Callback for compiling shaders when not cached.
    compile_fn: Option<Box<dyn Fn(&ShaderCompileRequest) -> Result<Vec<u8>, String>>>,
    /// Queue of shaders to warm on startup.
    warm_queue: Vec<ShaderCompileRequest>,
}

impl ShaderCacheManager {
    pub fn new(config: ShaderCacheConfig) -> Self {
        Self {
            cache: ShaderCache::new(config),
            compile_fn: None,
            warm_queue: Vec::new(),
        }
    }

    /// Set the shader compile callback.
    pub fn set_compile_fn<F>(&mut self, f: F)
    where
        F: Fn(&ShaderCompileRequest) -> Result<Vec<u8>, String> + 'static,
    {
        self.compile_fn = Some(Box::new(f));
    }

    /// Add a shader to the warm-up queue.
    pub fn add_to_warm_queue(&mut self, request: ShaderCompileRequest) {
        self.warm_queue.push(request);
    }

    /// Compile or retrieve a shader from cache.
    pub fn get_or_compile(&mut self, request: &ShaderCompileRequest) -> Result<ShaderCompileResult, String> {
        let hash = request.source_hash();

        // Try cache first.
        if let Ok(entry) = self.cache.get(hash) {
            return Ok(ShaderCompileResult {
                request: request.clone(),
                bytecode: entry.bytecode,
                was_cached: true,
                compile_time_ms: 0.0,
                warnings: Vec::new(),
            });
        }

        // Compile.
        let compile_fn = self.compile_fn.as_ref().ok_or("No compile function set")?;
        let start = std::time::Instant::now();
        let bytecode = compile_fn(request)?;
        let compile_time = start.elapsed().as_secs_f64() * 1000.0;

        // Store in cache.
        let entry = ShaderCacheEntry {
            source_hash: hash,
            bytecode: bytecode.clone(),
            stage: request.stage,
            target: request.target,
            entry_point: request.entry_point.clone(),
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            bytecode_size: bytecode.len() as u64,
            source_path: request.source_path.clone(),
            defines: request.defines.clone(),
        };
        let _ = self.cache.store(entry);

        Ok(ShaderCompileResult {
            request: request.clone(),
            bytecode,
            was_cached: false,
            compile_time_ms: compile_time,
            warnings: Vec::new(),
        })
    }

    /// Process the warm-up queue: compile/cache all queued shaders.
    pub fn process_warm_queue(&mut self) -> Vec<Result<ShaderCompileResult, String>> {
        let queue = std::mem::take(&mut self.warm_queue);
        let mut results = Vec::with_capacity(queue.len());
        for req in &queue {
            results.push(self.get_or_compile(req));
        }
        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_shader_hash_deterministic() {
        let source = "void main() { }";
        let defines = vec![("USE_FOG".into(), "1".into()), ("MAX_LIGHTS".into(), "16".into())];
        let h1 = compute_shader_hash(source, &defines);
        let h2 = compute_shader_hash(source, &defines);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_compute_shader_hash_define_order_independent() {
        let source = "void main() { }";
        let d1 = vec![("A".into(), "1".into()), ("B".into(), "2".into())];
        let d2 = vec![("B".into(), "2".into()), ("A".into(), "1".into())];
        assert_eq!(compute_shader_hash(source, &d1), compute_shader_hash(source, &d2));
    }

    #[test]
    fn test_entry_serialize_deserialize() {
        let entry = ShaderCacheEntry {
            source_hash: 0xDEADBEEFCAFEBABE,
            bytecode: vec![0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00],
            stage: ShaderStageKind::Fragment,
            target: ShaderTarget::SpirV,
            entry_point: "main".to_string(),
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 5,
            bytecode_size: 8,
            source_path: Some("shaders/test.frag".to_string()),
            defines: vec![("FOO".into(), "bar".into())],
        };
        let data = entry.serialize();
        let restored = ShaderCacheEntry::deserialize(&data).expect("deserialize should succeed");
        assert_eq!(restored.source_hash, entry.source_hash);
        assert_eq!(restored.bytecode, entry.bytecode);
        assert_eq!(restored.stage, entry.stage);
        assert_eq!(restored.target, entry.target);
        assert_eq!(restored.entry_point, entry.entry_point);
        assert_eq!(restored.defines, entry.defines);
    }

    #[test]
    fn test_cache_memory_hit() {
        let config = ShaderCacheConfig {
            disk_cache_enabled: false,
            ..Default::default()
        };
        let mut cache = ShaderCache::new(config);
        let entry = ShaderCacheEntry {
            source_hash: 42,
            bytecode: vec![1, 2, 3],
            stage: ShaderStageKind::Vertex,
            target: ShaderTarget::Wgsl,
            entry_point: "vs_main".into(),
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            bytecode_size: 3,
            source_path: None,
            defines: Vec::new(),
        };
        cache.store(entry.clone()).unwrap();
        let result = cache.get(42).unwrap();
        assert_eq!(result.bytecode, vec![1, 2, 3]);
        assert_eq!(cache.stats.hits, 1);
    }

    #[test]
    fn test_cache_miss() {
        let config = ShaderCacheConfig {
            disk_cache_enabled: false,
            ..Default::default()
        };
        let mut cache = ShaderCache::new(config);
        assert!(cache.get(999).is_err());
        assert_eq!(cache.stats.misses, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let config = ShaderCacheConfig {
            disk_cache_enabled: false,
            max_memory_entries: 3,
            ..Default::default()
        };
        let mut cache = ShaderCache::new(config);
        for i in 0..5 {
            let entry = ShaderCacheEntry {
                source_hash: i,
                bytecode: vec![i as u8],
                stage: ShaderStageKind::Compute,
                target: ShaderTarget::SpirV,
                entry_point: "main".into(),
                created_at: SystemTime::now(),
                last_accessed: SystemTime::now(),
                access_count: 0,
                bytecode_size: 1,
                source_path: None,
                defines: Vec::new(),
            };
            cache.store(entry).unwrap();
        }
        // Only 3 entries should remain in memory.
        assert_eq!(cache.memory_cache.len(), 3);
        // The oldest (0, 1) should have been evicted.
        assert!(cache.memory_cache.get(&0).is_none());
        assert!(cache.memory_cache.get(&1).is_none());
        assert!(cache.memory_cache.get(&4).is_some());
    }
}
