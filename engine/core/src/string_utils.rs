//! # String Utilities
//!
//! High-performance string manipulation utilities for the Genovo engine.
//!
//! ## Features
//!
//! - **StringBuilder** — Efficient string builder with pre-allocated capacity
//!   and fluent API for constructing strings without repeated allocations.
//! - **String hashing** — FNV-1a, Murmur3, and CityHash implementations for
//!   fast, deterministic string hashing.
//! - **Named-argument formatting** — String formatting with `{name}` placeholders
//!   replaced from a key-value map.
//! - **Wildcard matching** — Glob-style pattern matching with `*` and `?`.
//! - **Fuzzy matching** — Levenshtein distance for approximate string comparison.
//! - **String pool** — Deduplicated string storage with O(1) equality checks.
//! - **Path manipulation** — Join, normalize, extract extension/stem/parent.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// StringBuilder
// ---------------------------------------------------------------------------

/// An efficient string builder optimized for constructing strings incrementally.
///
/// Pre-allocates capacity to avoid repeated reallocations. Provides a fluent
/// API for chaining append operations.
///
/// # Example
///
/// ```ignore
/// use genovo_core::string_utils::StringBuilder;
///
/// let result = StringBuilder::new()
///     .push_str("Hello, ")
///     .push_str("world")
///     .push_char('!')
///     .build();
///
/// assert_eq!(result, "Hello, world!");
/// ```
pub struct StringBuilder {
    buffer: String,
}

impl StringBuilder {
    /// Create a new empty `StringBuilder`.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Create a `StringBuilder` with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: String::with_capacity(capacity),
        }
    }

    /// Append a string slice.
    pub fn push_str(mut self, s: &str) -> Self {
        self.buffer.push_str(s);
        self
    }

    /// Append a single character.
    pub fn push_char(mut self, c: char) -> Self {
        self.buffer.push(c);
        self
    }

    /// Append a formatted value.
    pub fn push_fmt(mut self, args: fmt::Arguments<'_>) -> Self {
        fmt::Write::write_fmt(&mut self.buffer, args).unwrap();
        self
    }

    /// Append an integer.
    pub fn push_int(self, value: i64) -> Self {
        self.push_str(&value.to_string())
    }

    /// Append a float with the given decimal precision.
    pub fn push_float(self, value: f64, decimals: usize) -> Self {
        self.push_str(&format!("{:.prec$}", value, prec = decimals))
    }

    /// Append a newline.
    pub fn push_newline(self) -> Self {
        self.push_char('\n')
    }

    /// Append a space.
    pub fn push_space(self) -> Self {
        self.push_char(' ')
    }

    /// Append the contents of another `StringBuilder`.
    pub fn append(mut self, other: &StringBuilder) -> Self {
        self.buffer.push_str(&other.buffer);
        self
    }

    /// Append a string repeated `count` times.
    pub fn push_repeated(mut self, s: &str, count: usize) -> Self {
        for _ in 0..count {
            self.buffer.push_str(s);
        }
        self
    }

    /// Append items from an iterator, separated by `separator`.
    pub fn push_join<I, T>(mut self, separator: &str, items: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: fmt::Display,
    {
        let mut first = true;
        for item in items {
            if !first {
                self.buffer.push_str(separator);
            }
            self.buffer.push_str(&item.to_string());
            first = false;
        }
        self
    }

    /// Returns the current length in bytes.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if the builder is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the builder without deallocating.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Returns the current capacity.
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Consume the builder and return the built string.
    pub fn build(self) -> String {
        self.buffer
    }

    /// Returns a reference to the current buffer.
    pub fn as_str(&self) -> &str {
        &self.buffer
    }

    /// Indent: prepend each line with `count` spaces.
    pub fn indent(self, count: usize) -> Self {
        let prefix = " ".repeat(count);
        let indented = self
            .buffer
            .lines()
            .map(|line| format!("{}{}", prefix, line))
            .collect::<Vec<_>>()
            .join("\n");
        Self { buffer: indented }
    }

    /// Trim whitespace from both ends.
    pub fn trim(mut self) -> Self {
        let trimmed = self.buffer.trim().to_string();
        self.buffer = trimmed;
        self
    }
}

impl Default for StringBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StringBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.buffer)
    }
}

impl From<StringBuilder> for String {
    fn from(sb: StringBuilder) -> String {
        sb.buffer
    }
}

impl From<&str> for StringBuilder {
    fn from(s: &str) -> Self {
        Self {
            buffer: s.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// String Hashing — FNV-1a
// ---------------------------------------------------------------------------

/// FNV-1a hash parameters for 32-bit hashes.
const FNV1A_32_OFFSET: u32 = 2_166_136_261;
const FNV1A_32_PRIME: u32 = 16_777_619;

/// FNV-1a hash parameters for 64-bit hashes.
const FNV1A_64_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV1A_64_PRIME: u64 = 1_099_511_628_211;

/// Compute a 32-bit FNV-1a hash of the given bytes.
///
/// FNV-1a is extremely fast for short keys (like component names, tags)
/// and has good distribution for hash tables.
pub fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash = FNV1A_32_OFFSET;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(FNV1A_32_PRIME);
    }
    hash
}

/// Compute a 64-bit FNV-1a hash of the given bytes.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV1A_64_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV1A_64_PRIME);
    }
    hash
}

/// Compute a 32-bit FNV-1a hash of a string.
pub fn fnv1a_str(s: &str) -> u32 {
    fnv1a_32(s.as_bytes())
}

/// Compute a 64-bit FNV-1a hash of a string.
pub fn fnv1a_str_64(s: &str) -> u64 {
    fnv1a_64(s.as_bytes())
}

// ---------------------------------------------------------------------------
// String Hashing — Murmur3
// ---------------------------------------------------------------------------

/// Compute a 32-bit MurmurHash3 of the given bytes.
///
/// Murmur3 provides excellent distribution and is widely used for hash-table
/// keys, bloom filters, and checksum validation.
pub fn murmur3_32(data: &[u8], seed: u32) -> u32 {
    let c1: u32 = 0xcc9e2d51;
    let c2: u32 = 0x1b873593;
    let len = data.len();
    let nblocks = len / 4;

    let mut h1 = seed;

    // Body: process 4-byte blocks.
    for i in 0..nblocks {
        let offset = i * 4;
        let mut k1 = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);

        k1 = k1.wrapping_mul(c1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(c2);

        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe6546b64);
    }

    // Tail: process remaining bytes.
    let tail = &data[nblocks * 4..];
    let mut k1: u32 = 0;

    if tail.len() >= 3 {
        k1 ^= (tail[2] as u32) << 16;
    }
    if tail.len() >= 2 {
        k1 ^= (tail[1] as u32) << 8;
    }
    if !tail.is_empty() {
        k1 ^= tail[0] as u32;
        k1 = k1.wrapping_mul(c1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(c2);
        h1 ^= k1;
    }

    // Finalization mix.
    h1 ^= len as u32;
    h1 = fmix32(h1);

    h1
}

/// Murmur3 finalization mix for 32-bit values.
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

/// Compute a 32-bit MurmurHash3 of a string.
pub fn murmur3_str(s: &str, seed: u32) -> u32 {
    murmur3_32(s.as_bytes(), seed)
}

// ---------------------------------------------------------------------------
// String Hashing — CityHash (simplified 64-bit)
// ---------------------------------------------------------------------------

/// CityHash-inspired 64-bit hash.
///
/// A simplified version of Google's CityHash providing good distribution
/// for strings of any length.
pub fn city_hash_64(data: &[u8]) -> u64 {
    let len = data.len() as u64;

    if data.len() <= 16 {
        return city_hash_short(data, len);
    }

    // For longer strings, process in 32-byte chunks.
    let mut a: u64 = fnv1a_64(data);
    let mut b: u64 = len.wrapping_mul(0x9e3779b97f4a7c15);
    let mut c: u64 = b.rotate_left(37).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(a);

    let chunks = data.len() / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let word = if offset + 8 <= data.len() {
            u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ])
        } else {
            0
        };

        a = a.wrapping_add(word);
        b = b.rotate_left(31).wrapping_mul(a);
        c ^= b;
        a = a.rotate_left(27).wrapping_add(c);
    }

    a = a.wrapping_add(b);
    a ^= a >> 33;
    a = a.wrapping_mul(0xff51afd7ed558ccd);
    a ^= a >> 33;
    a = a.wrapping_mul(0xc4ceb9fe1a85ec53);
    a ^= a >> 33;

    a
}

/// Short-string CityHash fallback for strings of 16 bytes or less.
fn city_hash_short(data: &[u8], len: u64) -> u64 {
    let mut h = len.wrapping_mul(0x9e3779b97f4a7c15);
    for &b in data {
        h = h.wrapping_add(b as u64);
        h = h.wrapping_mul(0x517cc1b727220a95);
        h = h.rotate_left(31);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h
}

/// Compute a 64-bit CityHash of a string.
pub fn city_hash_str(s: &str) -> u64 {
    city_hash_64(s.as_bytes())
}

// ---------------------------------------------------------------------------
// Named-argument formatting
// ---------------------------------------------------------------------------

/// Format a template string by replacing `{name}` placeholders with values
/// from a key-value map.
///
/// # Example
///
/// ```ignore
/// use genovo_core::string_utils::format_named;
///
/// let mut args = std::collections::HashMap::new();
/// args.insert("name".to_string(), "World".to_string());
/// args.insert("count".to_string(), "42".to_string());
///
/// let result = format_named("Hello, {name}! You have {count} items.", &args);
/// assert_eq!(result, "Hello, World! You have 42 items.");
/// ```
pub fn format_named(template: &str, args: &HashMap<String, String>) -> String {
    let mut result = String::with_capacity(template.len() * 2);
    let mut chars = template.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '{' {
            // Check for escaped brace `{{`.
            if chars.peek() == Some(&'{') {
                chars.next();
                result.push('{');
                continue;
            }

            // Read the placeholder name.
            let mut name = String::new();
            let mut found_close = false;
            for inner in chars.by_ref() {
                if inner == '}' {
                    found_close = true;
                    break;
                }
                name.push(inner);
            }

            if found_close {
                if let Some(value) = args.get(&name) {
                    result.push_str(value);
                } else {
                    // Leave the placeholder as-is if not found.
                    result.push('{');
                    result.push_str(&name);
                    result.push('}');
                }
            } else {
                result.push('{');
                result.push_str(&name);
            }
        } else if c == '}' {
            // Escaped closing brace `}}`.
            if chars.peek() == Some(&'}') {
                chars.next();
                result.push('}');
            } else {
                result.push(c);
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Helper to build a named argument map.
pub struct NamedArgs {
    map: HashMap<String, String>,
}

impl NamedArgs {
    /// Create a new empty argument map.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Insert a key-value pair.
    pub fn arg(mut self, key: impl Into<String>, value: impl fmt::Display) -> Self {
        self.map.insert(key.into(), value.to_string());
        self
    }

    /// Build the HashMap.
    pub fn build(self) -> HashMap<String, String> {
        self.map
    }

    /// Format a template with these arguments.
    pub fn format(&self, template: &str) -> String {
        format_named(template, &self.map)
    }
}

impl Default for NamedArgs {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Wildcard Matching
// ---------------------------------------------------------------------------

/// Check if a string matches a wildcard pattern.
///
/// Supports:
/// - `*` — matches any sequence of characters (including empty).
/// - `?` — matches exactly one character.
///
/// # Example
///
/// ```ignore
/// use genovo_core::string_utils::wildcard_match;
///
/// assert!(wildcard_match("hello_world", "hello_*"));
/// assert!(wildcard_match("test.rs", "*.rs"));
/// assert!(wildcard_match("a", "?"));
/// assert!(!wildcard_match("ab", "?"));
/// ```
pub fn wildcard_match(text: &str, pattern: &str) -> bool {
    let text_bytes = text.as_bytes();
    let pattern_bytes = pattern.as_bytes();

    let m = text_bytes.len();
    let n = pattern_bytes.len();

    // DP approach: dp[i][j] = whether text[0..i] matches pattern[0..j].
    // Use two rows to save memory.
    let mut prev = vec![false; n + 1];
    let mut curr = vec![false; n + 1];

    prev[0] = true;

    // Initialize: pattern `*` at the start can match empty string.
    for j in 1..=n {
        if pattern_bytes[j - 1] == b'*' {
            prev[j] = prev[j - 1];
        }
    }

    for i in 1..=m {
        curr[0] = false;
        for j in 1..=n {
            let pc = pattern_bytes[j - 1];
            if pc == b'*' {
                // '*' matches zero chars (curr[j-1]) or one more char (prev[j]).
                curr[j] = curr[j - 1] || prev[j];
            } else if pc == b'?' || pc == text_bytes[i - 1] {
                curr[j] = prev[j - 1];
            } else {
                curr[j] = false;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Case-insensitive wildcard match.
pub fn wildcard_match_ci(text: &str, pattern: &str) -> bool {
    wildcard_match(&text.to_lowercase(), &pattern.to_lowercase())
}

// ---------------------------------------------------------------------------
// Fuzzy Matching — Levenshtein Distance
// ---------------------------------------------------------------------------

/// Compute the Levenshtein edit distance between two strings.
///
/// The Levenshtein distance is the minimum number of single-character edits
/// (insertions, deletions, substitutions) needed to transform one string
/// into another.
///
/// # Example
///
/// ```ignore
/// use genovo_core::string_utils::levenshtein_distance;
///
/// assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
/// assert_eq!(levenshtein_distance("", "hello"), 5);
/// assert_eq!(levenshtein_distance("same", "same"), 0);
/// ```
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let m = a_bytes.len();
    let n = b_bytes.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use a single row for space efficiency.
    let mut prev_row: Vec<usize> = (0..=n).collect();
    let mut curr_row = vec![0usize; n + 1];

    for i in 1..=m {
        curr_row[0] = i;
        for j in 1..=n {
            let cost = if a_bytes[i - 1] == b_bytes[j - 1] {
                0
            } else {
                1
            };

            curr_row[j] = (prev_row[j] + 1) // deletion
                .min(curr_row[j - 1] + 1) // insertion
                .min(prev_row[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
}

/// Compute normalized Levenshtein similarity in [0, 1].
///
/// Returns 1.0 for identical strings and 0.0 for completely different strings.
pub fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein_distance(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

/// Compute the Damerau-Levenshtein distance (allows transpositions).
pub fn damerau_levenshtein_distance(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let m = a_bytes.len();
    let n = b_bytes.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut matrix = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        matrix[i][0] = i;
    }
    for j in 0..=n {
        matrix[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_bytes[i - 1] == b_bytes[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);

            // Transposition.
            if i > 1
                && j > 1
                && a_bytes[i - 1] == b_bytes[j - 2]
                && a_bytes[i - 2] == b_bytes[j - 1]
            {
                matrix[i][j] = matrix[i][j].min(matrix[i - 2][j - 2] + cost);
            }
        }
    }

    matrix[m][n]
}

/// Find the best fuzzy matches for `query` among `candidates`.
///
/// Returns candidates sorted by similarity (best first), filtered by
/// a minimum similarity threshold.
pub fn fuzzy_find<'a>(
    query: &str,
    candidates: &'a [&str],
    min_similarity: f64,
) -> Vec<(&'a str, f64)> {
    let mut results: Vec<(&str, f64)> = candidates
        .iter()
        .map(|&c| (c, levenshtein_similarity(query, c)))
        .filter(|(_, sim)| *sim >= min_similarity)
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ---------------------------------------------------------------------------
// String Pool
// ---------------------------------------------------------------------------

/// A string pool for deduplicating string storage.
///
/// Stores each unique string once and returns a `StringId` handle.
/// Equality checks between pooled strings are O(1) (just compare IDs).
///
/// # Example
///
/// ```ignore
/// use genovo_core::string_utils::StringPool;
///
/// let mut pool = StringPool::new();
/// let id1 = pool.intern("hello");
/// let id2 = pool.intern("hello");
/// let id3 = pool.intern("world");
///
/// assert_eq!(id1, id2);
/// assert_ne!(id1, id3);
/// assert_eq!(pool.resolve(id1), Some("hello"));
/// ```
pub struct StringPool {
    /// Stored strings.
    strings: Vec<String>,
    /// Map from string to index for deduplication.
    index: HashMap<String, StringId>,
}

/// Handle to a string in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StringId(pub u32);

impl StringPool {
    /// Create a new empty string pool.
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Create a pool with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            strings: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
        }
    }

    /// Intern a string, returning its ID.
    ///
    /// If the string is already in the pool, returns the existing ID.
    pub fn intern(&mut self, s: &str) -> StringId {
        if let Some(&id) = self.index.get(s) {
            return id;
        }

        let id = StringId(self.strings.len() as u32);
        self.strings.push(s.to_string());
        self.index.insert(s.to_string(), id);
        id
    }

    /// Resolve a `StringId` back to its string.
    pub fn resolve(&self, id: StringId) -> Option<&str> {
        self.strings.get(id.0 as usize).map(|s| s.as_str())
    }

    /// Returns the number of unique strings in the pool.
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Check if a string is already in the pool.
    pub fn contains(&self, s: &str) -> bool {
        self.index.contains_key(s)
    }

    /// Get the ID of a string without interning it.
    pub fn get(&self, s: &str) -> Option<StringId> {
        self.index.get(s).copied()
    }

    /// Returns all interned strings.
    pub fn all_strings(&self) -> &[String] {
        &self.strings
    }

    /// Total bytes used by interned strings.
    pub fn total_bytes(&self) -> usize {
        self.strings.iter().map(|s| s.len()).sum()
    }

    /// Clear the pool.
    pub fn clear(&mut self) {
        self.strings.clear();
        self.index.clear();
    }
}

impl Default for StringPool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Path Manipulation
// ---------------------------------------------------------------------------

/// Join two path components with a separator.
///
/// Handles trailing/leading separators to avoid double-separators.
pub fn path_join(base: &str, child: &str) -> String {
    if base.is_empty() {
        return child.to_string();
    }
    if child.is_empty() {
        return base.to_string();
    }

    let base_trimmed = base.trim_end_matches('/').trim_end_matches('\\');
    let child_trimmed = child.trim_start_matches('/').trim_start_matches('\\');

    format!("{}/{}", base_trimmed, child_trimmed)
}

/// Join multiple path components.
pub fn path_join_all(parts: &[&str]) -> String {
    let mut result = String::new();
    for part in parts {
        result = path_join(&result, part);
    }
    result
}

/// Normalize a path by resolving `.` and `..` components and converting
/// backslashes to forward slashes.
pub fn path_normalize(path: &str) -> String {
    let path = path.replace('\\', "/");
    let mut parts: Vec<&str> = Vec::new();

    for component in path.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if !parts.is_empty() && *parts.last().unwrap() != ".." {
                    parts.pop();
                } else {
                    parts.push("..");
                }
            }
            other => parts.push(other),
        }
    }

    let result = parts.join("/");
    if path.starts_with('/') {
        format!("/{}", result)
    } else {
        result
    }
}

/// Extract the file extension from a path (without the dot).
///
/// Returns `None` if there is no extension.
pub fn path_extension(path: &str) -> Option<&str> {
    let filename = path_filename(path)?;
    let dot_pos = filename.rfind('.')?;
    if dot_pos == 0 || dot_pos == filename.len() - 1 {
        None
    } else {
        Some(&filename[dot_pos + 1..])
    }
}

/// Extract the file stem (filename without extension).
pub fn path_stem(path: &str) -> Option<&str> {
    let filename = path_filename(path)?;
    if let Some(dot_pos) = filename.rfind('.') {
        if dot_pos > 0 {
            Some(&filename[..dot_pos])
        } else {
            Some(filename)
        }
    } else {
        Some(filename)
    }
}

/// Extract the filename from a path (last component).
pub fn path_filename(path: &str) -> Option<&str> {
    let trimmed = path.trim_end_matches('/').trim_end_matches('\\');
    if trimmed.is_empty() {
        return None;
    }
    let last_sep = trimmed
        .rfind('/')
        .or_else(|| trimmed.rfind('\\'));
    match last_sep {
        Some(pos) => Some(&trimmed[pos + 1..]),
        None => Some(trimmed),
    }
}

/// Extract the parent directory from a path.
pub fn path_parent(path: &str) -> Option<&str> {
    let trimmed = path.trim_end_matches('/').trim_end_matches('\\');
    let last_sep = trimmed
        .rfind('/')
        .or_else(|| trimmed.rfind('\\'));
    match last_sep {
        Some(pos) => {
            if pos == 0 {
                Some("/")
            } else {
                Some(&trimmed[..pos])
            }
        }
        None => None,
    }
}

/// Check if a path has a specific extension (case-insensitive).
pub fn path_has_extension(path: &str, ext: &str) -> bool {
    path_extension(path)
        .map(|e| e.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

/// Replace the extension of a path.
pub fn path_with_extension(path: &str, new_ext: &str) -> String {
    if let Some(dot_pos) = path.rfind('.') {
        let before_dot = &path[..dot_pos];
        if new_ext.is_empty() {
            before_dot.to_string()
        } else {
            format!("{}.{}", before_dot, new_ext)
        }
    } else if new_ext.is_empty() {
        path.to_string()
    } else {
        format!("{}.{}", path, new_ext)
    }
}

/// Make a path relative to a base directory.
pub fn path_relative(path: &str, base: &str) -> String {
    let norm_path = path_normalize(path);
    let norm_base = path_normalize(base);

    if norm_path.starts_with(&norm_base) {
        let relative = &norm_path[norm_base.len()..];
        let relative = relative.trim_start_matches('/');
        if relative.is_empty() {
            ".".to_string()
        } else {
            relative.to_string()
        }
    } else {
        norm_path
    }
}

// ---------------------------------------------------------------------------
// Additional string utilities
// ---------------------------------------------------------------------------

/// Split a string by a delimiter, trimming whitespace from each part.
pub fn split_trim(s: &str, delimiter: char) -> Vec<&str> {
    s.split(delimiter).map(|part| part.trim()).collect()
}

/// Check if a string contains only ASCII alphanumeric characters and underscores.
pub fn is_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !s.chars().next().unwrap().is_ascii_digit()
}

/// Convert a string to camelCase.
pub fn to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;

    for (i, c) in s.chars().enumerate() {
        if c == '_' || c == '-' || c == ' ' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else if i == 0 {
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }

    result
}

/// Convert a string to PascalCase.
pub fn to_pascal_case(s: &str) -> String {
    let camel = to_camel_case(s);
    let mut chars = camel.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let upper: String = first.to_uppercase().collect();
            format!("{}{}", upper, chars.collect::<String>())
        }
    }
}

/// Convert a string to snake_case.
pub fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    let mut prev_was_upper = false;

    for (i, c) in s.chars().enumerate() {
        if c == '-' || c == ' ' {
            result.push('_');
            prev_was_upper = false;
        } else if c.is_ascii_uppercase() {
            if i > 0 && !prev_was_upper && !result.ends_with('_') {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
            prev_was_upper = true;
        } else {
            result.push(c);
            prev_was_upper = false;
        }
    }

    result
}

/// Truncate a string to `max_len` characters, appending `suffix` if truncated.
pub fn truncate(s: &str, max_len: usize, suffix: &str) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let truncated = &s[..max_len.saturating_sub(suffix.len())];
        format!("{}{}", truncated, suffix)
    }
}

/// Repeat a string `count` times.
pub fn repeat(s: &str, count: usize) -> String {
    s.repeat(count)
}

/// Count occurrences of a substring.
pub fn count_occurrences(text: &str, pattern: &str) -> usize {
    if pattern.is_empty() {
        return 0;
    }
    text.matches(pattern).count()
}

/// Replace the first occurrence of `from` with `to`.
pub fn replace_first(text: &str, from: &str, to: &str) -> String {
    if let Some(pos) = text.find(from) {
        format!("{}{}{}", &text[..pos], to, &text[pos + from.len()..])
    } else {
        text.to_string()
    }
}

/// Pad a string on the left to reach `width`.
pub fn pad_left(s: &str, width: usize, pad_char: char) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        let padding: String = std::iter::repeat(pad_char).take(width - s.len()).collect();
        format!("{}{}", padding, s)
    }
}

/// Pad a string on the right to reach `width`.
pub fn pad_right(s: &str, width: usize, pad_char: char) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        let padding: String = std::iter::repeat(pad_char).take(width - s.len()).collect();
        format!("{}{}", s, padding)
    }
}

/// Center a string within `width`, padding with `pad_char`.
pub fn center(s: &str, width: usize, pad_char: char) -> String {
    if s.len() >= width {
        return s.to_string();
    }
    let total_pad = width - s.len();
    let left_pad = total_pad / 2;
    let right_pad = total_pad - left_pad;
    let left: String = std::iter::repeat(pad_char).take(left_pad).collect();
    let right: String = std::iter::repeat(pad_char).take(right_pad).collect();
    format!("{}{}{}", left, s, right)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // StringBuilder tests
    #[test]
    fn test_string_builder_basic() {
        let result = StringBuilder::new()
            .push_str("Hello")
            .push_str(", ")
            .push_str("World")
            .push_char('!')
            .build();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_string_builder_int_float() {
        let result = StringBuilder::new()
            .push_str("x=")
            .push_int(42)
            .push_str(" y=")
            .push_float(3.14, 2)
            .build();
        assert_eq!(result, "x=42 y=3.14");
    }

    #[test]
    fn test_string_builder_join() {
        let result = StringBuilder::new()
            .push_join(", ", vec!["a", "b", "c"])
            .build();
        assert_eq!(result, "a, b, c");
    }

    #[test]
    fn test_string_builder_repeated() {
        let result = StringBuilder::new().push_repeated("ab", 3).build();
        assert_eq!(result, "ababab");
    }

    // FNV-1a tests
    #[test]
    fn test_fnv1a_deterministic() {
        let h1 = fnv1a_str("hello");
        let h2 = fnv1a_str("hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_strings() {
        let h1 = fnv1a_str("hello");
        let h2 = fnv1a_str("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_fnv1a_64_deterministic() {
        let h1 = fnv1a_str_64("test");
        let h2 = fnv1a_str_64("test");
        assert_eq!(h1, h2);
    }

    // Murmur3 tests
    #[test]
    fn test_murmur3_deterministic() {
        let h1 = murmur3_str("hello", 0);
        let h2 = murmur3_str("hello", 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_murmur3_different_seeds() {
        let h1 = murmur3_str("hello", 0);
        let h2 = murmur3_str("hello", 42);
        assert_ne!(h1, h2);
    }

    // CityHash tests
    #[test]
    fn test_city_hash_deterministic() {
        let h1 = city_hash_str("hello world");
        let h2 = city_hash_str("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_city_hash_different_strings() {
        let h1 = city_hash_str("abc");
        let h2 = city_hash_str("def");
        assert_ne!(h1, h2);
    }

    // Named args tests
    #[test]
    fn test_format_named() {
        let args = NamedArgs::new()
            .arg("name", "World")
            .arg("count", 42)
            .build();
        let result = format_named("Hello, {name}! {count} items.", &args);
        assert_eq!(result, "Hello, World! 42 items.");
    }

    #[test]
    fn test_format_named_missing_key() {
        let args = HashMap::new();
        let result = format_named("Hello, {missing}!", &args);
        assert_eq!(result, "Hello, {missing}!");
    }

    #[test]
    fn test_format_named_escaped_braces() {
        let args = HashMap::new();
        let result = format_named("{{literal}}", &args);
        assert_eq!(result, "{literal}");
    }

    // Wildcard tests
    #[test]
    fn test_wildcard_star() {
        assert!(wildcard_match("hello_world", "hello_*"));
        assert!(wildcard_match("test.rs", "*.rs"));
        assert!(wildcard_match("anything", "*"));
        assert!(wildcard_match("", "*"));
    }

    #[test]
    fn test_wildcard_question() {
        assert!(wildcard_match("a", "?"));
        assert!(!wildcard_match("ab", "?"));
        assert!(wildcard_match("abc", "a?c"));
    }

    #[test]
    fn test_wildcard_combined() {
        assert!(wildcard_match("foo_bar_baz.rs", "foo_*_*.rs"));
        assert!(wildcard_match("test_1.txt", "test_?.txt"));
    }

    #[test]
    fn test_wildcard_no_match() {
        assert!(!wildcard_match("hello", "world"));
        assert!(!wildcard_match("test.rs", "*.txt"));
    }

    // Levenshtein tests
    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein_distance("same", "same"), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein_distance("", "hello"), 5);
        assert_eq!(levenshtein_distance("hello", ""), 5);
    }

    #[test]
    fn test_levenshtein_kitten_sitting() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_levenshtein_similarity_score() {
        let sim = levenshtein_similarity("hello", "hallo");
        assert!(sim > 0.7);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_damerau_levenshtein() {
        // Transposition: "ab" -> "ba" should be 1.
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
    }

    #[test]
    fn test_fuzzy_find() {
        let candidates = vec!["hello", "hallo", "world", "help", "helicopter"];
        let results = fuzzy_find("hello", &candidates, 0.5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "hello");
    }

    // String Pool tests
    #[test]
    fn test_string_pool_intern() {
        let mut pool = StringPool::new();
        let id1 = pool.intern("hello");
        let id2 = pool.intern("hello");
        let id3 = pool.intern("world");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_string_pool_resolve() {
        let mut pool = StringPool::new();
        let id = pool.intern("test");
        assert_eq!(pool.resolve(id), Some("test"));
    }

    // Path tests
    #[test]
    fn test_path_join() {
        assert_eq!(path_join("a", "b"), "a/b");
        assert_eq!(path_join("a/", "b"), "a/b");
        assert_eq!(path_join("a", "/b"), "a/b");
        assert_eq!(path_join("", "b"), "b");
    }

    #[test]
    fn test_path_normalize() {
        assert_eq!(path_normalize("a/b/../c"), "a/c");
        assert_eq!(path_normalize("a/./b"), "a/b");
        assert_eq!(path_normalize("a\\b\\c"), "a/b/c");
    }

    #[test]
    fn test_path_extension() {
        assert_eq!(path_extension("test.rs"), Some("rs"));
        assert_eq!(path_extension("no_ext"), None);
        assert_eq!(path_extension("a/b/c.txt"), Some("txt"));
    }

    #[test]
    fn test_path_stem() {
        assert_eq!(path_stem("test.rs"), Some("test"));
        assert_eq!(path_stem("a/b/file.txt"), Some("file"));
    }

    #[test]
    fn test_path_filename() {
        assert_eq!(path_filename("a/b/c.txt"), Some("c.txt"));
        assert_eq!(path_filename("file.txt"), Some("file.txt"));
    }

    #[test]
    fn test_path_parent() {
        assert_eq!(path_parent("a/b/c.txt"), Some("a/b"));
        assert_eq!(path_parent("file.txt"), None);
    }

    #[test]
    fn test_path_with_extension() {
        assert_eq!(path_with_extension("file.txt", "rs"), "file.rs");
        assert_eq!(path_with_extension("file", "rs"), "file.rs");
    }

    // Case conversion tests
    #[test]
    fn test_to_camel_case() {
        assert_eq!(to_camel_case("hello_world"), "helloWorld");
        assert_eq!(to_camel_case("some-thing"), "someThing");
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("HelloWorld"), "hello_world");
        assert_eq!(to_snake_case("camelCase"), "camel_case");
    }

    // Utility tests
    #[test]
    fn test_is_identifier() {
        assert!(is_identifier("hello_world"));
        assert!(is_identifier("_private"));
        assert!(!is_identifier("123abc"));
        assert!(!is_identifier(""));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello world", 8, "..."), "hello...");
        assert_eq!(truncate("short", 10, "..."), "short");
    }

    #[test]
    fn test_pad_left_right() {
        assert_eq!(pad_left("42", 5, '0'), "00042");
        assert_eq!(pad_right("hi", 5, '.'), "hi...");
    }

    #[test]
    fn test_center() {
        assert_eq!(center("hi", 6, '-'), "--hi--");
    }

    #[test]
    fn test_count_occurrences() {
        assert_eq!(count_occurrences("abcabc", "abc"), 2);
        assert_eq!(count_occurrences("hello", "xyz"), 0);
    }

    #[test]
    fn test_replace_first() {
        assert_eq!(replace_first("aaa", "a", "b"), "baa");
    }

    #[test]
    fn test_split_trim() {
        let parts = split_trim("a , b , c", ',');
        assert_eq!(parts, vec!["a", "b", "c"]);
    }
}
