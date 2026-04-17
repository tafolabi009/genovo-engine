// engine/render/src/shader_defines.rs
//
// Shader preprocessing and define management for the Genovo engine.
//
// Provides a system for managing shader preprocessor directives:
//
// - **#define/#undef management** — Set and clear preprocessor defines with
//   typed values.
// - **Feature flag combinations** — Generate all valid combinations of feature
//   flags for shader variant compilation.
// - **Shader variant compilation** — Compile shader permutations from a set
//   of active keywords.
// - **Shader keyword sets** — Named sets of keywords that can be toggled
//   together.
// - **Global/per-material keywords** — Two-level keyword system: global
//   keywords affect all shaders, per-material keywords are specific.
// - **Shader complexity analysis** — Estimate shader instruction count and
//   resource usage from the source.
//
// # Architecture
//
// The system produces a list of `#define` directives to prepend to shader
// source code before compilation. Shader variants are identified by a hash
// of their active keywords, enabling caching and reuse.

use std::collections::{HashMap, HashSet, BTreeSet};

// ---------------------------------------------------------------------------
// Define value
// ---------------------------------------------------------------------------

/// A value assigned to a preprocessor define.
#[derive(Debug, Clone, PartialEq)]
pub enum DefineValue {
    /// Boolean define (present or absent, no value).
    Flag,
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// String value.
    Str(String),
}

impl DefineValue {
    /// Format the value for a #define directive.
    pub fn to_define_string(&self) -> String {
        match self {
            Self::Flag => String::new(),
            Self::Int(v) => v.to_string(),
            Self::Float(v) => format!("{:.8}", v),
            Self::Str(v) => v.clone(),
        }
    }
}

impl std::fmt::Display for DefineValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flag => Ok(()),
            Self::Int(v) => write!(f, "{}", v),
            Self::Float(v) => write!(f, "{:.8}", v),
            Self::Str(v) => write!(f, "{}", v),
        }
    }
}

// ---------------------------------------------------------------------------
// Shader define set
// ---------------------------------------------------------------------------

/// A set of preprocessor defines.
#[derive(Debug, Clone)]
pub struct DefineSet {
    /// Active defines: name → value.
    defines: HashMap<String, DefineValue>,
}

impl DefineSet {
    /// Create an empty define set.
    pub fn new() -> Self {
        Self { defines: HashMap::new() }
    }

    /// Set a flag define (no value, just present).
    pub fn set_flag(&mut self, name: impl Into<String>) {
        self.defines.insert(name.into(), DefineValue::Flag);
    }

    /// Set an integer define.
    pub fn set_int(&mut self, name: impl Into<String>, value: i64) {
        self.defines.insert(name.into(), DefineValue::Int(value));
    }

    /// Set a float define.
    pub fn set_float(&mut self, name: impl Into<String>, value: f64) {
        self.defines.insert(name.into(), DefineValue::Float(value));
    }

    /// Set a string define.
    pub fn set_str(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.defines.insert(name.into(), DefineValue::Str(value.into()));
    }

    /// Remove (undefine) a define.
    pub fn undefine(&mut self, name: &str) {
        self.defines.remove(name);
    }

    /// Check if a define is set.
    pub fn is_defined(&self, name: &str) -> bool {
        self.defines.contains_key(name)
    }

    /// Get the value of a define.
    pub fn get(&self, name: &str) -> Option<&DefineValue> {
        self.defines.get(name)
    }

    /// Get all define names (sorted for deterministic output).
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.defines.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Number of defines.
    pub fn len(&self) -> usize {
        self.defines.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.defines.is_empty()
    }

    /// Merge another define set into this one (overwrites on conflict).
    pub fn merge(&mut self, other: &DefineSet) {
        for (name, value) in &other.defines {
            self.defines.insert(name.clone(), value.clone());
        }
    }

    /// Generate the preprocessor preamble text.
    ///
    /// Returns a string of `#define` directives to prepend to shader source.
    pub fn to_preamble(&self) -> String {
        let mut lines = Vec::new();
        let mut sorted: Vec<(&String, &DefineValue)> = self.defines.iter().collect();
        sorted.sort_by_key(|(k, _)| k.as_str());

        for (name, value) in sorted {
            match value {
                DefineValue::Flag => lines.push(format!("#define {}", name)),
                DefineValue::Int(v) => lines.push(format!("#define {} {}", name, v)),
                DefineValue::Float(v) => lines.push(format!("#define {} {:.8}", name, v)),
                DefineValue::Str(v) => lines.push(format!("#define {} {}", name, v)),
            }
        }

        lines.join("\n") + "\n"
    }

    /// Compute a hash of the active defines for variant identification.
    pub fn compute_hash(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64; // FNV-1a offset basis.
        let mut sorted: Vec<(&String, &DefineValue)> = self.defines.iter().collect();
        sorted.sort_by_key(|(k, _)| k.as_str());

        for (name, value) in sorted {
            for byte in name.bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash ^= b'=' as u64;
            hash = hash.wrapping_mul(0x100000001b3);

            let value_str = value.to_define_string();
            for byte in value_str.bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash ^= b'\n' as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }

        hash
    }

    /// Clear all defines.
    pub fn clear(&mut self) {
        self.defines.clear();
    }
}

impl Default for DefineSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shader keywords
// ---------------------------------------------------------------------------

/// A shader keyword: a named toggle that maps to a preprocessor define.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderKeyword {
    /// Keyword name (e.g. "USE_NORMAL_MAP").
    pub name: String,
    /// Whether this keyword is enabled by default.
    pub default_enabled: bool,
    /// Optional: the preprocessor define name (defaults to the keyword name).
    pub define_name: Option<String>,
    /// Description for editor UI.
    pub description: String,
}

impl ShaderKeyword {
    /// Create a new keyword.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            name,
            default_enabled: false,
            define_name: None,
            description: String::new(),
        }
    }

    /// Set the default state.
    pub fn with_default(mut self, enabled: bool) -> Self {
        self.default_enabled = enabled;
        self
    }

    /// Set a custom define name.
    pub fn with_define(mut self, define: impl Into<String>) -> Self {
        self.define_name = Some(define.into());
        self
    }

    /// Set a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Get the preprocessor define name for this keyword.
    pub fn effective_define(&self) -> &str {
        self.define_name.as_deref().unwrap_or(&self.name)
    }
}

/// A mutually exclusive keyword group (e.g. "SHADOW_QUALITY" → LOW, MEDIUM, HIGH).
#[derive(Debug, Clone)]
pub struct KeywordGroup {
    /// Group name.
    pub name: String,
    /// Keywords in this group (only one can be active at a time).
    pub keywords: Vec<ShaderKeyword>,
    /// Index of the default keyword.
    pub default_index: usize,
}

impl KeywordGroup {
    /// Create a new keyword group.
    pub fn new(name: impl Into<String>, keywords: Vec<ShaderKeyword>, default_index: usize) -> Self {
        Self {
            name: name.into(),
            keywords,
            default_index,
        }
    }

    /// Get the currently active keyword name from a set of enabled keywords.
    pub fn active_keyword<'a>(&'a self, enabled: &HashSet<String>) -> &'a ShaderKeyword {
        for kw in &self.keywords {
            if enabled.contains(&kw.name) {
                return kw;
            }
        }
        &self.keywords[self.default_index.min(self.keywords.len() - 1)]
    }
}

// ---------------------------------------------------------------------------
// Keyword state
// ---------------------------------------------------------------------------

/// Manages the state of shader keywords (global + per-material).
#[derive(Debug, Clone)]
pub struct KeywordState {
    /// Global keywords (affect all shaders).
    pub global: HashSet<String>,
    /// Per-material keywords.
    pub material: HashSet<String>,
    /// Available keywords.
    pub available: Vec<ShaderKeyword>,
    /// Keyword groups.
    pub groups: Vec<KeywordGroup>,
}

impl KeywordState {
    /// Create a new keyword state.
    pub fn new() -> Self {
        Self {
            global: HashSet::new(),
            material: HashSet::new(),
            available: Vec::new(),
            groups: Vec::new(),
        }
    }

    /// Register a keyword.
    pub fn register(&mut self, keyword: ShaderKeyword) {
        if keyword.default_enabled {
            self.global.insert(keyword.name.clone());
        }
        self.available.push(keyword);
    }

    /// Register a keyword group.
    pub fn register_group(&mut self, group: KeywordGroup) {
        // Enable the default keyword.
        if let Some(kw) = group.keywords.get(group.default_index) {
            self.global.insert(kw.name.clone());
        }
        for kw in &group.keywords {
            self.available.push(kw.clone());
        }
        self.groups.push(group);
    }

    /// Enable a global keyword.
    pub fn enable_global(&mut self, name: &str) {
        // Disable other keywords in the same group.
        for group in &self.groups {
            if group.keywords.iter().any(|k| k.name == name) {
                for kw in &group.keywords {
                    self.global.remove(&kw.name);
                }
            }
        }
        self.global.insert(name.to_string());
    }

    /// Disable a global keyword.
    pub fn disable_global(&mut self, name: &str) {
        self.global.remove(name);
    }

    /// Enable a per-material keyword.
    pub fn enable_material(&mut self, name: &str) {
        for group in &self.groups {
            if group.keywords.iter().any(|k| k.name == name) {
                for kw in &group.keywords {
                    self.material.remove(&kw.name);
                }
            }
        }
        self.material.insert(name.to_string());
    }

    /// Disable a per-material keyword.
    pub fn disable_material(&mut self, name: &str) {
        self.material.remove(name);
    }

    /// Build the combined define set from global + material keywords.
    pub fn build_defines(&self) -> DefineSet {
        let mut defines = DefineSet::new();

        let combined: HashSet<&str> = self.global.iter()
            .chain(self.material.iter())
            .map(|s| s.as_str())
            .collect();

        for kw in &self.available {
            if combined.contains(kw.name.as_str()) {
                defines.set_flag(kw.effective_define());
            }
        }

        defines
    }

    /// Get the set of all active keyword names.
    pub fn active_keywords(&self) -> BTreeSet<String> {
        self.global.iter().chain(self.material.iter()).cloned().collect()
    }
}

impl Default for KeywordState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shader variant
// ---------------------------------------------------------------------------

/// A compiled shader variant identified by its keyword hash.
#[derive(Debug, Clone)]
pub struct ShaderVariant {
    /// Hash of the active keywords.
    pub keyword_hash: u64,
    /// Active keywords for this variant.
    pub keywords: BTreeSet<String>,
    /// Preprocessor preamble.
    pub preamble: String,
    /// Compiled shader handle (backend-specific).
    pub compiled_handle: u64,
    /// Compilation time in milliseconds.
    pub compile_time_ms: f64,
    /// Whether compilation was successful.
    pub compiled: bool,
    /// Compilation error message (if any).
    pub error: Option<String>,
}

/// Manages shader variant compilation.
#[derive(Debug)]
pub struct ShaderVariantManager {
    /// Cached variants by keyword hash.
    variants: HashMap<u64, ShaderVariant>,
    /// Shader source code.
    pub source: String,
    /// Shader name/path.
    pub name: String,
    /// Available keywords for this shader.
    pub keywords: Vec<ShaderKeyword>,
    /// Total number of possible variants.
    pub total_variants: u64,
}

impl ShaderVariantManager {
    /// Create a new variant manager.
    pub fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            variants: HashMap::new(),
            source: source.into(),
            name: name.into(),
            keywords: Vec::new(),
            total_variants: 1,
        }
    }

    /// Register a keyword for this shader.
    pub fn add_keyword(&mut self, keyword: ShaderKeyword) {
        self.keywords.push(keyword);
        self.total_variants = 1 << self.keywords.len().min(20);
    }

    /// Check if a variant is cached.
    pub fn has_variant(&self, defines: &DefineSet) -> bool {
        let hash = defines.compute_hash();
        self.variants.contains_key(&hash)
    }

    /// Get a cached variant.
    pub fn get_variant(&self, defines: &DefineSet) -> Option<&ShaderVariant> {
        let hash = defines.compute_hash();
        self.variants.get(&hash)
    }

    /// Record a compiled variant.
    pub fn register_variant(
        &mut self,
        defines: &DefineSet,
        compiled_handle: u64,
        compile_time_ms: f64,
    ) {
        let hash = defines.compute_hash();
        let keywords = defines.names().iter().map(|s| s.to_string()).collect();

        self.variants.insert(hash, ShaderVariant {
            keyword_hash: hash,
            keywords,
            preamble: defines.to_preamble(),
            compiled_handle,
            compile_time_ms,
            compiled: true,
            error: None,
        });
    }

    /// Record a failed compilation.
    pub fn register_error(&mut self, defines: &DefineSet, error: String) {
        let hash = defines.compute_hash();
        let keywords = defines.names().iter().map(|s| s.to_string()).collect();

        self.variants.insert(hash, ShaderVariant {
            keyword_hash: hash,
            keywords,
            preamble: defines.to_preamble(),
            compiled_handle: 0,
            compile_time_ms: 0.0,
            compiled: false,
            error: Some(error),
        });
    }

    /// Number of cached variants.
    pub fn cached_count(&self) -> usize {
        self.variants.len()
    }

    /// Generate all possible keyword combinations.
    pub fn all_combinations(&self) -> Vec<DefineSet> {
        let n = self.keywords.len();
        let count = 1u64 << n;
        let mut result = Vec::with_capacity(count as usize);

        for mask in 0..count {
            let mut defines = DefineSet::new();
            for (i, kw) in self.keywords.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    defines.set_flag(kw.effective_define());
                }
            }
            result.push(defines);
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Feature flag combinations
// ---------------------------------------------------------------------------

/// Generate all valid combinations of feature flags, respecting groups.
///
/// # Arguments
/// * `toggles` — Independent toggle keywords.
/// * `groups` — Mutually exclusive keyword groups.
///
/// # Returns
/// A list of define sets, one per valid combination.
pub fn generate_feature_combinations(
    toggles: &[ShaderKeyword],
    groups: &[KeywordGroup],
) -> Vec<DefineSet> {
    // Start with toggle combinations.
    let toggle_count = toggles.len();
    let toggle_combos = 1u64 << toggle_count;

    // Group combinations: product of each group's keyword count.
    let mut group_sizes: Vec<usize> = Vec::new();
    for group in groups {
        group_sizes.push(group.keywords.len().max(1));
    }
    let group_total: u64 = group_sizes.iter().map(|&s| s as u64).product();

    let total = toggle_combos * group_total;
    let mut result = Vec::with_capacity(total.min(65536) as usize);

    for toggle_mask in 0..toggle_combos {
        for group_idx in 0..group_total {
            let mut defines = DefineSet::new();

            // Set toggle defines.
            for (i, kw) in toggles.iter().enumerate() {
                if toggle_mask & (1 << i) != 0 {
                    defines.set_flag(kw.effective_define());
                }
            }

            // Set group defines.
            let mut remaining = group_idx;
            for (g, group) in groups.iter().enumerate() {
                let size = group_sizes[g] as u64;
                let selected = (remaining % size) as usize;
                remaining /= size;

                if selected < group.keywords.len() {
                    defines.set_flag(group.keywords[selected].effective_define());
                }
            }

            result.push(defines);

            if result.len() >= 65536 {
                return result; // Safety limit.
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Shader complexity analysis
// ---------------------------------------------------------------------------

/// Result of shader complexity analysis.
#[derive(Debug, Clone)]
pub struct ShaderComplexity {
    /// Estimated ALU instruction count.
    pub alu_ops: u32,
    /// Estimated texture sample count.
    pub texture_samples: u32,
    /// Number of branches.
    pub branches: u32,
    /// Number of loops.
    pub loops: u32,
    /// Number of uniform/buffer reads.
    pub uniform_reads: u32,
    /// Number of output writes.
    pub output_writes: u32,
    /// Estimated register pressure (number of temporaries).
    pub registers: u32,
    /// Number of #define directives.
    pub defines: u32,
    /// Source line count.
    pub source_lines: u32,
    /// Estimated complexity tier (1=simple, 2=medium, 3=complex, 4=very complex).
    pub tier: u32,
}

/// Analyse shader source code complexity (heuristic, not a real compiler).
///
/// This is a rough estimate based on pattern matching common shader operations.
pub fn analyse_shader_complexity(source: &str) -> ShaderComplexity {
    let mut alu_ops = 0u32;
    let mut texture_samples = 0u32;
    let mut branches = 0u32;
    let mut loops = 0u32;
    let mut uniform_reads = 0u32;
    let mut output_writes = 0u32;
    let mut defines = 0u32;
    let mut registers = 0u32;

    let lines: Vec<&str> = source.lines().collect();
    let source_lines = lines.len() as u32;

    for line in &lines {
        let trimmed = line.trim();

        // Count ALU ops (arithmetic).
        alu_ops += count_occurrences(trimmed, '+') as u32;
        alu_ops += count_occurrences(trimmed, '-') as u32;
        alu_ops += count_occurrences(trimmed, '*') as u32;
        alu_ops += count_occurrences(trimmed, '/') as u32;

        // Count built-in math functions.
        for func in &[
            "sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "pow",
            "exp", "log", "abs", "floor", "ceil", "fract", "mod",
            "min", "max", "clamp", "mix", "step", "smoothstep",
            "normalize", "dot", "cross", "length", "distance", "reflect",
            "refract", "faceforward", "inversesqrt",
        ] {
            alu_ops += count_word_occurrences(trimmed, func) as u32;
        }

        // Count texture samples.
        for func in &[
            "texture", "textureSample", "textureLod", "textureGrad",
            "textureSampleLevel", "textureSampleGrad", "textureSampleCompare",
            "textureLoad", "textureStore", "texelFetch", "textureCube",
        ] {
            texture_samples += count_word_occurrences(trimmed, func) as u32;
        }

        // Count branches.
        if trimmed.starts_with("if") || trimmed.starts_with("} else") {
            branches += 1;
        }

        // Count loops.
        if trimmed.starts_with("for") || trimmed.starts_with("while") || trimmed.starts_with("loop") {
            loops += 1;
        }

        // Count defines.
        if trimmed.starts_with("#define") {
            defines += 1;
        }

        // Count uniform reads (heuristic).
        if trimmed.contains("uniform") || trimmed.contains("@group") || trimmed.contains("cbuffer") {
            uniform_reads += 1;
        }

        // Count output writes.
        if trimmed.contains("gl_Position") || trimmed.contains("gl_FragColor")
            || trimmed.contains("@location") || trimmed.contains("SV_Target")
        {
            output_writes += 1;
        }

        // Count variable declarations (register pressure estimate).
        if trimmed.starts_with("let ") || trimmed.starts_with("var ") || trimmed.starts_with("float ") || trimmed.starts_with("vec") {
            registers += 1;
        }
    }

    // Compute tier.
    let score = alu_ops + texture_samples * 4 + branches * 2 + loops * 8;
    let tier = if score < 50 {
        1
    } else if score < 200 {
        2
    } else if score < 500 {
        3
    } else {
        4
    };

    ShaderComplexity {
        alu_ops,
        texture_samples,
        branches,
        loops,
        uniform_reads,
        output_writes,
        registers,
        defines,
        source_lines,
        tier,
    }
}

/// Count occurrences of a character in a string.
fn count_occurrences(s: &str, c: char) -> usize {
    s.chars().filter(|&ch| ch == c).count()
}

/// Count occurrences of a word in a string (word boundary aware).
fn count_word_occurrences(s: &str, word: &str) -> usize {
    let mut count = 0;
    let mut remaining = s;
    while let Some(pos) = remaining.find(word) {
        // Check word boundaries.
        let before_ok = if pos == 0 {
            true
        } else {
            let ch = remaining.as_bytes()[pos - 1];
            !ch.is_ascii_alphanumeric() && ch != b'_'
        };

        let after_pos = pos + word.len();
        let after_ok = if after_pos >= remaining.len() {
            true
        } else {
            let ch = remaining.as_bytes()[after_pos];
            !ch.is_ascii_alphanumeric() && ch != b'_'
        };

        if before_ok && after_ok {
            count += 1;
        }

        remaining = &remaining[pos + word.len()..];
    }
    count
}

// ---------------------------------------------------------------------------
// Shader source preprocessing
// ---------------------------------------------------------------------------

/// Preprocess shader source by applying defines and resolving #ifdef/#ifndef/#else/#endif.
///
/// This is a simple preprocessor (not a full C preprocessor). It handles:
/// - `#define NAME` / `#define NAME VALUE`
/// - `#undef NAME`
/// - `#ifdef NAME` / `#ifndef NAME`
/// - `#else`
/// - `#endif`
/// - `#if defined(NAME)`
///
/// Does NOT handle `#include` or expression evaluation.
pub fn preprocess_shader(source: &str, external_defines: &DefineSet) -> String {
    let mut defines: HashMap<String, String> = HashMap::new();

    // Copy external defines.
    for name in external_defines.names() {
        let value = external_defines.get(name).unwrap();
        defines.insert(name.to_string(), value.to_define_string());
    }

    let mut output = String::new();
    let mut condition_stack: Vec<bool> = Vec::new(); // true = current block is active.
    let mut else_stack: Vec<bool> = Vec::new(); // tracks whether we've seen #else.

    for line in source.lines() {
        let trimmed = line.trim();
        let all_active = condition_stack.iter().all(|&b| b);

        if trimmed.starts_with("#ifdef ") {
            let name = trimmed[7..].trim();
            let active = defines.contains_key(name) && all_active;
            condition_stack.push(active);
            else_stack.push(false);
            continue;
        }

        if trimmed.starts_with("#ifndef ") {
            let name = trimmed[8..].trim();
            let active = !defines.contains_key(name) && all_active;
            condition_stack.push(active);
            else_stack.push(false);
            continue;
        }

        if trimmed.starts_with("#if defined(") {
            if let Some(end) = trimmed.find(')') {
                let name = &trimmed[12..end];
                let active = defines.contains_key(name) && all_active;
                condition_stack.push(active);
                else_stack.push(false);
            }
            continue;
        }

        if trimmed == "#else" {
            let stack_len = condition_stack.len();
            if stack_len > 0 {
                let parent_active = condition_stack[..stack_len - 1]
                    .iter()
                    .all(|&b| b);
                if !*else_stack.last().unwrap_or(&false) {
                    if parent_active {
                        let last = condition_stack.last_mut().unwrap();
                        *last = !*last;
                    }
                    if let Some(e) = else_stack.last_mut() {
                        *e = true;
                    }
                }
            }
            continue;
        }

        if trimmed == "#endif" {
            condition_stack.pop();
            else_stack.pop();
            continue;
        }

        if !all_active {
            continue;
        }

        if trimmed.starts_with("#define ") {
            let rest = &trimmed[8..];
            let parts: Vec<&str> = rest.splitn(2, char::is_whitespace).collect();
            let name = parts[0].to_string();
            let value = if parts.len() > 1 { parts[1].to_string() } else { String::new() };
            defines.insert(name, value);
            output.push_str(line);
            output.push('\n');
            continue;
        }

        if trimmed.starts_with("#undef ") {
            let name = trimmed[7..].trim();
            defines.remove(name);
            output.push_str(line);
            output.push('\n');
            continue;
        }

        output.push_str(line);
        output.push('\n');
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_define_set() {
        let mut ds = DefineSet::new();
        ds.set_flag("USE_NORMAL_MAP");
        ds.set_int("MAX_LIGHTS", 16);

        assert!(ds.is_defined("USE_NORMAL_MAP"));
        assert_eq!(ds.get("MAX_LIGHTS"), Some(&DefineValue::Int(16)));

        let preamble = ds.to_preamble();
        assert!(preamble.contains("#define MAX_LIGHTS 16"));
        assert!(preamble.contains("#define USE_NORMAL_MAP"));
    }

    #[test]
    fn test_hash_determinism() {
        let mut a = DefineSet::new();
        a.set_flag("A");
        a.set_flag("B");

        let mut b = DefineSet::new();
        b.set_flag("B");
        b.set_flag("A");

        assert_eq!(a.compute_hash(), b.compute_hash());
    }

    #[test]
    fn test_keyword_group() {
        let group = KeywordGroup::new(
            "QUALITY",
            vec![
                ShaderKeyword::new("QUALITY_LOW"),
                ShaderKeyword::new("QUALITY_MEDIUM"),
                ShaderKeyword::new("QUALITY_HIGH"),
            ],
            1, // Default to MEDIUM.
        );

        let mut enabled = HashSet::new();
        enabled.insert("QUALITY_HIGH".to_string());
        let active = group.active_keyword(&enabled);
        assert_eq!(active.name, "QUALITY_HIGH");
    }

    #[test]
    fn test_preprocess_ifdef() {
        let source = r#"
before
#ifdef FEATURE_A
feature_a_code
#else
no_feature_a
#endif
after
"#;
        let mut defines = DefineSet::new();
        defines.set_flag("FEATURE_A");

        let result = preprocess_shader(source, &defines);
        assert!(result.contains("feature_a_code"));
        assert!(!result.contains("no_feature_a"));
        assert!(result.contains("before"));
        assert!(result.contains("after"));
    }

    #[test]
    fn test_shader_complexity() {
        let source = r#"
let color = textureSample(t, s, uv);
let n = normalize(normal);
let d = dot(n, light_dir);
if (d > 0.0) {
    color = color * d;
}
"#;
        let complexity = analyse_shader_complexity(source);
        assert!(complexity.texture_samples >= 1);
        assert!(complexity.branches >= 1);
        assert!(complexity.alu_ops > 0);
    }

    #[test]
    fn test_feature_combinations() {
        let toggles = vec![
            ShaderKeyword::new("A"),
            ShaderKeyword::new("B"),
        ];
        let combos = generate_feature_combinations(&toggles, &[]);
        assert_eq!(combos.len(), 4); // 2^2 = 4.
    }
}
