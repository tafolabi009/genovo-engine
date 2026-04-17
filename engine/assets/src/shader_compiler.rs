//! # Shader Compilation Pipeline (v2)
//!
//! Provides WGSL shader validation, include/define preprocessing,
//! conditional compilation (`#ifdef`), shader caching (skip recompilation
//! when source is unchanged), cross-compilation from WGSL to SPIR-V stubs,
//! and shader reflection to extract uniform/texture bindings.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors raised during shader compilation.
#[derive(Debug, Clone)]
pub enum ShaderError {
    /// A syntax error in the WGSL source.
    SyntaxError { line: usize, col: usize, message: String },
    /// An `#include` directive references a file that does not exist.
    IncludeNotFound(String),
    /// Circular include detected.
    CircularInclude(Vec<String>),
    /// An `#ifdef` / `#endif` mismatch.
    PreprocessorMismatch(String),
    /// SPIR-V cross-compilation failed.
    SpirVError(String),
    /// A validation error in the compiled shader.
    ValidationError(String),
    /// I/O error loading a shader file.
    IoError(String),
    /// Unknown or invalid entry point.
    UnknownEntryPoint(String),
    /// Cache corruption.
    CacheCorrupted(String),
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SyntaxError { line, col, message } => {
                write!(f, "syntax error at {line}:{col}: {message}")
            }
            Self::IncludeNotFound(path) => write!(f, "include not found: {path}"),
            Self::CircularInclude(chain) => {
                write!(f, "circular include: {}", chain.join(" -> "))
            }
            Self::PreprocessorMismatch(msg) => write!(f, "preprocessor: {msg}"),
            Self::SpirVError(msg) => write!(f, "SPIR-V error: {msg}"),
            Self::ValidationError(msg) => write!(f, "validation error: {msg}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::UnknownEntryPoint(name) => write!(f, "unknown entry point: {name}"),
            Self::CacheCorrupted(msg) => write!(f, "cache corrupted: {msg}"),
        }
    }
}

impl std::error::Error for ShaderError {}

// ---------------------------------------------------------------------------
// Shader stage
// ---------------------------------------------------------------------------

/// GPU pipeline stage a shader program runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

impl ShaderStage {
    /// SPIR-V execution model enum value.
    pub fn spirv_execution_model(self) -> u32 {
        match self {
            Self::Vertex => 0,
            Self::Fragment => 4,
            Self::Compute => 5,
        }
    }

    /// WGSL annotation name.
    pub fn wgsl_annotation(self) -> &'static str {
        match self {
            Self::Vertex => "@vertex",
            Self::Fragment => "@fragment",
            Self::Compute => "@compute",
        }
    }
}

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vertex => write!(f, "vertex"),
            Self::Fragment => write!(f, "fragment"),
            Self::Compute => write!(f, "compute"),
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocessor
// ---------------------------------------------------------------------------

/// Preprocessor state for handling `#ifdef`, `#ifndef`, `#else`, `#endif`,
/// `#define`, `#include`, and simple text substitution.
pub struct ShaderPreprocessor {
    /// Active defines (key -> optional replacement text).
    defines: HashMap<String, Option<String>>,
    /// Include search paths.
    include_paths: Vec<PathBuf>,
    /// Resolved include file contents (path -> source text).
    include_cache: HashMap<String, String>,
    /// Track includes currently being processed to detect cycles.
    include_stack: Vec<String>,
}

impl ShaderPreprocessor {
    /// Create a new preprocessor.
    pub fn new() -> Self {
        Self {
            defines: HashMap::new(),
            include_paths: Vec::new(),
            include_cache: HashMap::new(),
            include_stack: Vec::new(),
        }
    }

    /// Add a preprocessor define.
    pub fn define(&mut self, name: impl Into<String>, value: Option<String>) {
        self.defines.insert(name.into(), value);
    }

    /// Remove a preprocessor define.
    pub fn undefine(&mut self, name: &str) {
        self.defines.remove(name);
    }

    /// Add an include search path.
    pub fn add_include_path(&mut self, path: impl Into<PathBuf>) {
        self.include_paths.push(path.into());
    }

    /// Register an include file's content directly (for virtual file systems).
    pub fn register_include(&mut self, name: impl Into<String>, source: impl Into<String>) {
        self.include_cache.insert(name.into(), source.into());
    }

    /// Check if a symbol is defined.
    pub fn is_defined(&self, name: &str) -> bool {
        self.defines.contains_key(name)
    }

    /// Process a shader source string, resolving all directives.
    pub fn process(&mut self, source: &str, file_name: &str) -> Result<String, ShaderError> {
        // Check for circular includes
        if self.include_stack.contains(&file_name.to_string()) {
            let mut chain = self.include_stack.clone();
            chain.push(file_name.to_string());
            return Err(ShaderError::CircularInclude(chain));
        }
        self.include_stack.push(file_name.to_string());

        let mut output = String::with_capacity(source.len());
        let mut if_stack: Vec<IfState> = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        for (line_no, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Is the current section active?
            let active = if_stack.iter().all(|s| s.active);

            if trimmed.starts_with("#define ") {
                if active {
                    let rest = trimmed.strip_prefix("#define ").unwrap().trim();
                    let mut parts = rest.splitn(2, char::is_whitespace);
                    let name = parts.next().unwrap_or("").to_string();
                    let value = parts.next().map(|s| s.to_string());
                    if !name.is_empty() {
                        self.defines.insert(name, value);
                    }
                }
            } else if trimmed.starts_with("#undef ") {
                if active {
                    let name = trimmed.strip_prefix("#undef ").unwrap().trim();
                    self.defines.remove(name);
                }
            } else if trimmed.starts_with("#ifdef ") {
                let name = trimmed.strip_prefix("#ifdef ").unwrap().trim();
                let defined = self.defines.contains_key(name);
                if_stack.push(IfState {
                    active: active && defined,
                    else_seen: false,
                    parent_active: active,
                });
            } else if trimmed.starts_with("#ifndef ") {
                let name = trimmed.strip_prefix("#ifndef ").unwrap().trim();
                let defined = self.defines.contains_key(name);
                if_stack.push(IfState {
                    active: active && !defined,
                    else_seen: false,
                    parent_active: active,
                });
            } else if trimmed == "#else" {
                if let Some(state) = if_stack.last_mut() {
                    if state.else_seen {
                        return Err(ShaderError::PreprocessorMismatch(
                            format!("duplicate #else at line {}", line_no + 1),
                        ));
                    }
                    state.else_seen = true;
                    state.active = state.parent_active && !state.active;
                } else {
                    return Err(ShaderError::PreprocessorMismatch(
                        format!("#else without #ifdef at line {}", line_no + 1),
                    ));
                }
            } else if trimmed == "#endif" {
                if if_stack.pop().is_none() {
                    return Err(ShaderError::PreprocessorMismatch(
                        format!("#endif without #ifdef at line {}", line_no + 1),
                    ));
                }
            } else if trimmed.starts_with("#include ") {
                if active {
                    let path = trimmed
                        .strip_prefix("#include ")
                        .unwrap()
                        .trim()
                        .trim_matches('"')
                        .trim_matches('<')
                        .trim_matches('>')
                        .to_string();
                    let include_source = self.resolve_include(&path)?;
                    let processed = self.process(&include_source, &path)?;
                    output.push_str(&processed);
                    output.push('\n');
                }
            } else if active {
                // Perform define substitution
                let mut expanded = line.to_string();
                for (name, value) in &self.defines {
                    if let Some(replacement) = value {
                        expanded = expanded.replace(name.as_str(), replacement);
                    }
                }
                output.push_str(&expanded);
                output.push('\n');
            }
        }

        if !if_stack.is_empty() {
            return Err(ShaderError::PreprocessorMismatch(format!(
                "{} unclosed #ifdef directive(s) in {file_name}",
                if_stack.len()
            )));
        }

        self.include_stack.pop();
        Ok(output)
    }

    /// Resolve an include file name to its source content.
    fn resolve_include(&self, name: &str) -> Result<String, ShaderError> {
        // Check the in-memory cache first
        if let Some(src) = self.include_cache.get(name) {
            return Ok(src.clone());
        }
        // Search include paths
        for dir in &self.include_paths {
            let full_path = dir.join(name);
            if full_path.exists() {
                return std::fs::read_to_string(&full_path)
                    .map_err(|e| ShaderError::IoError(format!("{}: {e}", full_path.display())));
            }
        }
        Err(ShaderError::IncludeNotFound(name.to_string()))
    }
}

impl Default for ShaderPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal state for conditional compilation.
struct IfState {
    active: bool,
    else_seen: bool,
    parent_active: bool,
}

// ---------------------------------------------------------------------------
// WGSL validator
// ---------------------------------------------------------------------------

/// Result of WGSL validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the shader is valid.
    pub valid: bool,
    /// Warnings (non-fatal).
    pub warnings: Vec<ShaderDiagnostic>,
    /// Errors (fatal).
    pub errors: Vec<ShaderDiagnostic>,
    /// Detected entry points.
    pub entry_points: Vec<EntryPointInfo>,
}

/// A diagnostic message from the validator.
#[derive(Debug, Clone)]
pub struct ShaderDiagnostic {
    /// Severity level.
    pub severity: DiagnosticSeverity,
    /// Line number (1-based).
    pub line: usize,
    /// Column number (1-based).
    pub col: usize,
    /// Message text.
    pub message: String,
}

/// Severity of a shader diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

/// Information about a shader entry point.
#[derive(Debug, Clone)]
pub struct EntryPointInfo {
    /// Name of the entry point function.
    pub name: String,
    /// Pipeline stage.
    pub stage: ShaderStage,
    /// Workgroup size (only for compute shaders).
    pub workgroup_size: Option<[u32; 3]>,
}

/// Validate a WGSL shader source string.
///
/// This is a simplified validator that checks for common issues. A production
/// implementation would use `naga` for full validation.
pub fn validate_wgsl(source: &str) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut entry_points = Vec::new();

    let lines: Vec<&str> = source.lines().collect();
    let mut brace_depth: i32 = 0;
    let mut paren_depth: i32 = 0;

    for (line_no, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Count braces
        for ch in trimmed.chars() {
            match ch {
                '{' => brace_depth += 1,
                '}' => brace_depth -= 1,
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                _ => {}
            }
        }

        if brace_depth < 0 {
            errors.push(ShaderDiagnostic {
                severity: DiagnosticSeverity::Error,
                line: line_no + 1,
                col: 1,
                message: "unexpected closing brace '}'".into(),
            });
        }

        // Detect entry points
        if trimmed.contains("@vertex") {
            if let Some(fn_name) = extract_fn_name(trimmed, &lines, line_no) {
                entry_points.push(EntryPointInfo {
                    name: fn_name,
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                });
            }
        }
        if trimmed.contains("@fragment") {
            if let Some(fn_name) = extract_fn_name(trimmed, &lines, line_no) {
                entry_points.push(EntryPointInfo {
                    name: fn_name,
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                });
            }
        }
        if trimmed.contains("@compute") {
            let wg_size = extract_workgroup_size(trimmed);
            if let Some(fn_name) = extract_fn_name(trimmed, &lines, line_no) {
                entry_points.push(EntryPointInfo {
                    name: fn_name,
                    stage: ShaderStage::Compute,
                    workgroup_size: wg_size,
                });
            }
        }

        // Warn about common issues
        if trimmed.contains("discard") && !trimmed.starts_with("//") {
            warnings.push(ShaderDiagnostic {
                severity: DiagnosticSeverity::Warning,
                line: line_no + 1,
                col: 1,
                message: "'discard' can cause performance issues on some GPUs".into(),
            });
        }
    }

    if brace_depth != 0 {
        errors.push(ShaderDiagnostic {
            severity: DiagnosticSeverity::Error,
            line: lines.len(),
            col: 1,
            message: format!("unmatched braces: depth is {brace_depth} at end of file"),
        });
    }
    if paren_depth != 0 {
        errors.push(ShaderDiagnostic {
            severity: DiagnosticSeverity::Error,
            line: lines.len(),
            col: 1,
            message: format!("unmatched parentheses: depth is {paren_depth} at end of file"),
        });
    }

    let valid = errors.is_empty();
    ValidationResult {
        valid,
        warnings,
        errors,
        entry_points,
    }
}

/// Extract a function name from a line containing a stage annotation.
fn extract_fn_name(line: &str, all_lines: &[&str], line_idx: usize) -> Option<String> {
    // Look for "fn <name>" on this line or the next
    let search_text = if line.contains("fn ") {
        line.to_string()
    } else if line_idx + 1 < all_lines.len() {
        format!("{} {}", line, all_lines[line_idx + 1])
    } else {
        return None;
    };
    let fn_pos = search_text.find("fn ")?;
    let after_fn = &search_text[fn_pos + 3..];
    let name: String = after_fn
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Extract workgroup_size from a compute shader annotation.
fn extract_workgroup_size(line: &str) -> Option<[u32; 3]> {
    let start = line.find("@workgroup_size(")?;
    let after = &line[start + 16..];
    let end = after.find(')')?;
    let args: Vec<&str> = after[..end].split(',').collect();
    let mut size = [1u32; 3];
    for (i, arg) in args.iter().enumerate().take(3) {
        if let Ok(v) = arg.trim().parse::<u32>() {
            size[i] = v;
        }
    }
    Some(size)
}

// ---------------------------------------------------------------------------
// Shader reflection
// ---------------------------------------------------------------------------

/// A binding extracted from a shader through reflection.
#[derive(Debug, Clone)]
pub struct ShaderBinding {
    /// Binding group index.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Name of the binding variable.
    pub name: String,
    /// Type of binding.
    pub binding_type: BindingType,
    /// Size in bytes (for uniform buffers).
    pub size: Option<u32>,
}

/// Type of a shader resource binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingType {
    /// Uniform buffer.
    UniformBuffer,
    /// Storage buffer (read-only).
    StorageBufferReadOnly,
    /// Storage buffer (read-write).
    StorageBufferReadWrite,
    /// Sampled texture (2D).
    Texture2D,
    /// Sampled texture (3D).
    Texture3D,
    /// Sampled texture (cube).
    TextureCube,
    /// Texture with multisampling.
    TextureMultisampled,
    /// Depth texture.
    DepthTexture,
    /// Sampler.
    Sampler,
    /// Comparison sampler.
    ComparisonSampler,
    /// Storage texture (write).
    StorageTexture,
}

/// Full reflection data for a compiled shader.
#[derive(Debug, Clone)]
pub struct ShaderReflection {
    /// All resource bindings.
    pub bindings: Vec<ShaderBinding>,
    /// Entry points.
    pub entry_points: Vec<EntryPointInfo>,
    /// Push constant ranges (offset, size).
    pub push_constants: Vec<(u32, u32)>,
    /// Required features.
    pub required_features: Vec<String>,
}

/// Perform simple reflection on WGSL source to extract bindings.
pub fn reflect_wgsl(source: &str) -> ShaderReflection {
    let mut bindings = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut current_group: Option<u32> = None;
    let mut current_binding: Option<u32> = None;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Parse @group(N) @binding(M)
        if let Some(g) = extract_attribute_value(trimmed, "@group") {
            current_group = Some(g);
        }
        if let Some(b) = extract_attribute_value(trimmed, "@binding") {
            current_binding = Some(b);
        }

        // If we have group and binding, try to parse the var declaration
        if let (Some(group), Some(binding)) = (current_group, current_binding) {
            if trimmed.starts_with("var") {
                let bt = infer_binding_type(trimmed);
                let name = extract_var_name(trimmed).unwrap_or_default();
                bindings.push(ShaderBinding {
                    group,
                    binding,
                    name,
                    binding_type: bt,
                    size: None,
                });
                current_group = None;
                current_binding = None;
            }
        }

        // Reset if we hit a line without annotations
        if !trimmed.starts_with('@') && !trimmed.starts_with("var") && !trimmed.is_empty() {
            if current_group.is_some() || current_binding.is_some() {
                // only reset if we didn't just parse a binding
            }
        }
    }

    let validation = validate_wgsl(source);
    ShaderReflection {
        bindings,
        entry_points: validation.entry_points,
        push_constants: Vec::new(),
        required_features: Vec::new(),
    }
}

/// Extract a numeric value from an attribute like `@group(0)`.
fn extract_attribute_value(line: &str, attr: &str) -> Option<u32> {
    let pos = line.find(attr)?;
    let after = &line[pos + attr.len()..];
    let paren_start = after.find('(')?;
    let paren_end = after.find(')')?;
    after[paren_start + 1..paren_end].trim().parse().ok()
}

/// Extract the variable name from a `var<...> name : type` declaration.
fn extract_var_name(line: &str) -> Option<String> {
    // Handle: var<uniform> my_uniform : MyType;
    let var_pos = line.find("var")?;
    let rest = &line[var_pos + 3..];
    let after_angle = if let Some(close) = rest.find('>') {
        &rest[close + 1..]
    } else {
        rest
    };
    let name: String = after_angle
        .trim()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Infer the binding type from a WGSL variable declaration.
fn infer_binding_type(line: &str) -> BindingType {
    if line.contains("uniform") {
        BindingType::UniformBuffer
    } else if line.contains("storage") && line.contains("read_write") {
        BindingType::StorageBufferReadWrite
    } else if line.contains("storage") {
        BindingType::StorageBufferReadOnly
    } else if line.contains("texture_2d") {
        BindingType::Texture2D
    } else if line.contains("texture_3d") {
        BindingType::Texture3D
    } else if line.contains("texture_cube") {
        BindingType::TextureCube
    } else if line.contains("texture_multisampled") {
        BindingType::TextureMultisampled
    } else if line.contains("texture_depth") {
        BindingType::DepthTexture
    } else if line.contains("texture_storage") {
        BindingType::StorageTexture
    } else if line.contains("sampler_comparison") {
        BindingType::ComparisonSampler
    } else if line.contains("sampler") {
        BindingType::Sampler
    } else {
        BindingType::UniformBuffer
    }
}

// ---------------------------------------------------------------------------
// SPIR-V cross-compilation (stub)
// ---------------------------------------------------------------------------

/// SPIR-V output from cross-compilation.
#[derive(Debug, Clone)]
pub struct SpirVModule {
    /// Raw SPIR-V bytecode.
    pub bytecode: Vec<u32>,
    /// SPIR-V version (e.g. 0x00010500 for 1.5).
    pub version: u32,
    /// Entry points in this module.
    pub entry_points: Vec<EntryPointInfo>,
}

impl SpirVModule {
    /// Size in bytes.
    pub fn byte_size(&self) -> usize {
        self.bytecode.len() * 4
    }

    /// Serialise to raw bytes (little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.bytecode.len() * 4);
        for &word in &self.bytecode {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// Validate the SPIR-V magic number.
    pub fn validate_magic(&self) -> bool {
        if self.bytecode.is_empty() {
            return false;
        }
        self.bytecode[0] == 0x0723_0203
    }
}

/// Cross-compile WGSL source to SPIR-V.
///
/// This is a stub that generates a minimal valid SPIR-V module header.
/// A real implementation would use `naga` for cross-compilation.
pub fn compile_wgsl_to_spirv(
    source: &str,
    stage: ShaderStage,
    entry_point: &str,
) -> Result<SpirVModule, ShaderError> {
    // First validate the WGSL
    let validation = validate_wgsl(source);
    if !validation.valid {
        let first_error = validation
            .errors
            .first()
            .map(|e| e.message.clone())
            .unwrap_or_else(|| "unknown error".into());
        return Err(ShaderError::ValidationError(first_error));
    }

    // Check that the entry point exists
    let ep = validation
        .entry_points
        .iter()
        .find(|ep| ep.name == entry_point)
        .cloned()
        .ok_or_else(|| ShaderError::UnknownEntryPoint(entry_point.to_string()))?;

    // Generate a minimal SPIR-V module (stub)
    let mut bytecode = Vec::new();
    // Magic number
    bytecode.push(0x0723_0203);
    // Version 1.5
    let version = 0x0001_0500u32;
    bytecode.push(version);
    // Generator magic
    bytecode.push(0x474E_4F56); // "GNOV"
    // Bound (max ID + 1)
    bytecode.push(16);
    // Reserved
    bytecode.push(0);
    // OpCapability Shader (17, 1 word operand)
    bytecode.push((2 << 16) | 17);
    bytecode.push(1); // Shader capability
    // OpMemoryModel (14, Logical, GLSL450)
    bytecode.push((3 << 16) | 14);
    bytecode.push(0); // Logical
    bytecode.push(1); // GLSL450
    // OpEntryPoint
    let exec_model = stage.spirv_execution_model();
    let name_words = string_to_spirv_words(entry_point);
    let word_count = 3 + name_words.len() as u32;
    bytecode.push((word_count << 16) | 15);
    bytecode.push(exec_model);
    bytecode.push(1); // entry point ID
    bytecode.extend_from_slice(&name_words);

    Ok(SpirVModule {
        bytecode,
        version,
        entry_points: vec![ep],
    })
}

/// Encode a string as SPIR-V literal words (null-terminated, padded to 4 bytes).
fn string_to_spirv_words(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let padded_len = ((bytes.len() + 1) + 3) / 4 * 4;
    let mut padded = vec![0u8; padded_len];
    padded[..bytes.len()].copy_from_slice(bytes);
    padded
        .chunks(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Shader cache
// ---------------------------------------------------------------------------

/// Represents a cached compilation result.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Hash of the source (after preprocessing).
    pub source_hash: u64,
    /// Hash of the define set used.
    pub defines_hash: u64,
    /// Cached SPIR-V bytecode.
    pub spirv: Vec<u32>,
    /// When the entry was created.
    pub timestamp: u64,
    /// Whether this entry is still valid.
    pub valid: bool,
}

/// A simple in-memory shader compilation cache.
pub struct ShaderCache {
    /// Map from (file_key, stage, entry_point) to cached result.
    entries: HashMap<String, CacheEntry>,
    /// Maximum number of entries.
    max_entries: usize,
    /// Cache hits since creation.
    hits: u64,
    /// Cache misses since creation.
    misses: u64,
}

impl ShaderCache {
    /// Create a new shader cache.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            hits: 0,
            misses: 0,
        }
    }

    /// Build a cache key from file name, stage, and entry point.
    pub fn make_key(file: &str, stage: ShaderStage, entry_point: &str) -> String {
        format!("{file}::{stage}::{entry_point}")
    }

    /// Look up a cached entry.
    pub fn get(&mut self, key: &str, source_hash: u64, defines_hash: u64) -> Option<&CacheEntry> {
        if let Some(entry) = self.entries.get(key) {
            if entry.valid && entry.source_hash == source_hash && entry.defines_hash == defines_hash
            {
                self.hits += 1;
                return Some(entry);
            }
        }
        self.misses += 1;
        None
    }

    /// Store a compiled result in the cache.
    pub fn put(&mut self, key: String, source_hash: u64, defines_hash: u64, spirv: Vec<u32>) {
        // Evict if full (simple: remove the oldest entry)
        if self.entries.len() >= self.max_entries {
            if let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, v)| v.timestamp)
                .map(|(k, _)| k.clone())
            {
                self.entries.remove(&oldest_key);
            }
        }
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.entries.insert(
            key,
            CacheEntry {
                source_hash,
                defines_hash,
                spirv,
                timestamp: now,
                valid: true,
            },
        );
    }

    /// Invalidate a specific cache entry.
    pub fn invalidate(&mut self, key: &str) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.valid = false;
        }
    }

    /// Invalidate all entries.
    pub fn invalidate_all(&mut self) {
        for entry in self.entries.values_mut() {
            entry.valid = false;
        }
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hit rate as a fraction.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Total number of hits.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Total number of misses.
    pub fn misses(&self) -> u64 {
        self.misses
    }
}

impl Default for ShaderCache {
    fn default() -> Self {
        Self::new(1024)
    }
}

// ---------------------------------------------------------------------------
// Simple FNV-1a hash for strings
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash for content hashing.
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Hash a set of defines for cache keying.
pub fn hash_defines(defines: &HashMap<String, Option<String>>) -> u64 {
    let mut sorted: Vec<(&String, &Option<String>)> = defines.iter().collect();
    sorted.sort_by_key(|(k, _)| *k);
    let mut combined = String::new();
    for (k, v) in sorted {
        combined.push_str(k);
        combined.push('=');
        if let Some(val) = v {
            combined.push_str(val);
        }
        combined.push(';');
    }
    fnv1a_hash(combined.as_bytes())
}

// ---------------------------------------------------------------------------
// Compiler orchestrator
// ---------------------------------------------------------------------------

/// High-level shader compiler that ties together preprocessing, validation,
/// cross-compilation, and caching.
pub struct ShaderCompiler {
    /// Preprocessor instance.
    pub preprocessor: ShaderPreprocessor,
    /// Compilation cache.
    pub cache: ShaderCache,
    /// Whether to perform full validation.
    pub validate: bool,
    /// Target SPIR-V version.
    pub spirv_version: u32,
    /// Compilation statistics.
    pub stats: CompilerStats,
}

/// Statistics about shader compilation.
#[derive(Debug, Clone, Default)]
pub struct CompilerStats {
    /// Total shaders compiled (not cached).
    pub compilations: u64,
    /// Total shaders served from cache.
    pub cache_hits: u64,
    /// Total compilation time.
    pub total_compile_time: Duration,
    /// Total validation time.
    pub total_validation_time: Duration,
    /// Total preprocessing time.
    pub total_preprocess_time: Duration,
}

impl ShaderCompiler {
    /// Create a new shader compiler with default settings.
    pub fn new() -> Self {
        Self {
            preprocessor: ShaderPreprocessor::new(),
            cache: ShaderCache::new(1024),
            validate: true,
            spirv_version: 0x0001_0500,
            stats: CompilerStats::default(),
        }
    }

    /// Compile a WGSL shader to SPIR-V, using the cache when possible.
    pub fn compile(
        &mut self,
        source: &str,
        file_name: &str,
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<SpirVModule, ShaderError> {
        // Preprocess
        let preprocess_start = Instant::now();
        let processed = self.preprocessor.process(source, file_name)?;
        self.stats.total_preprocess_time += preprocess_start.elapsed();

        // Check cache
        let source_hash = fnv1a_hash(processed.as_bytes());
        let defines_hash = hash_defines(&self.preprocessor.defines);
        let cache_key = ShaderCache::make_key(file_name, stage, entry_point);

        if let Some(cached) = self.cache.get(&cache_key, source_hash, defines_hash) {
            self.stats.cache_hits += 1;
            return Ok(SpirVModule {
                bytecode: cached.spirv.clone(),
                version: self.spirv_version,
                entry_points: vec![],
            });
        }

        // Validate
        if self.validate {
            let val_start = Instant::now();
            let result = validate_wgsl(&processed);
            self.stats.total_validation_time += val_start.elapsed();
            if !result.valid {
                let msg = result
                    .errors
                    .first()
                    .map(|e| format!("line {}:{}: {}", e.line, e.col, e.message))
                    .unwrap_or_else(|| "unknown error".into());
                return Err(ShaderError::ValidationError(msg));
            }
        }

        // Compile
        let compile_start = Instant::now();
        let module = compile_wgsl_to_spirv(&processed, stage, entry_point)?;
        self.stats.total_compile_time += compile_start.elapsed();
        self.stats.compilations += 1;

        // Cache the result
        self.cache
            .put(cache_key, source_hash, defines_hash, module.bytecode.clone());

        Ok(module)
    }

    /// Add a preprocessor define.
    pub fn define(&mut self, name: impl Into<String>, value: Option<String>) {
        self.preprocessor.define(name, value);
    }

    /// Register an include file.
    pub fn register_include(&mut self, name: impl Into<String>, source: impl Into<String>) {
        self.preprocessor.register_include(name, source);
    }
}

impl Default for ShaderCompiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_ifdef_true() {
        let mut pp = ShaderPreprocessor::new();
        pp.define("FEATURE_A", None);
        let source = r#"
#ifdef FEATURE_A
float x = 1.0;
#endif
"#;
        let result = pp.process(source, "test.wgsl").unwrap();
        assert!(result.contains("float x = 1.0;"));
    }

    #[test]
    fn test_preprocessor_ifdef_false() {
        let mut pp = ShaderPreprocessor::new();
        let source = r#"
#ifdef FEATURE_A
float x = 1.0;
#endif
"#;
        let result = pp.process(source, "test.wgsl").unwrap();
        assert!(!result.contains("float x = 1.0;"));
    }

    #[test]
    fn test_preprocessor_ifdef_else() {
        let mut pp = ShaderPreprocessor::new();
        let source = r#"
#ifdef HAS_ALPHA
var alpha = true;
#else
var alpha = false;
#endif
"#;
        let result = pp.process(source, "test.wgsl").unwrap();
        assert!(result.contains("var alpha = false;"));
        assert!(!result.contains("var alpha = true;"));
    }

    #[test]
    fn test_preprocessor_ifndef() {
        let mut pp = ShaderPreprocessor::new();
        let source = r#"
#ifndef GUARD
var included = true;
#endif
"#;
        let result = pp.process(source, "test.wgsl").unwrap();
        assert!(result.contains("var included = true;"));
    }

    #[test]
    fn test_preprocessor_include() {
        let mut pp = ShaderPreprocessor::new();
        pp.register_include("common.wgsl", "fn common() -> f32 { return 1.0; }");
        let source = r#"
#include "common.wgsl"
fn main() {}
"#;
        let result = pp.process(source, "test.wgsl").unwrap();
        assert!(result.contains("fn common()"));
        assert!(result.contains("fn main()"));
    }

    #[test]
    fn test_preprocessor_circular_include() {
        let mut pp = ShaderPreprocessor::new();
        pp.register_include("a.wgsl", "#include \"b.wgsl\"");
        pp.register_include("b.wgsl", "#include \"a.wgsl\"");
        let result = pp.process("#include \"a.wgsl\"", "main.wgsl");
        assert!(matches!(result, Err(ShaderError::CircularInclude(_))));
    }

    #[test]
    fn test_preprocessor_define_substitution() {
        let mut pp = ShaderPreprocessor::new();
        pp.define("MAX_LIGHTS", Some("16".into()));
        let source = "var lights: array<Light, MAX_LIGHTS>;";
        let result = pp.process(source, "test.wgsl").unwrap();
        assert!(result.contains("array<Light, 16>"));
    }

    #[test]
    fn test_preprocessor_unmatched_endif() {
        let mut pp = ShaderPreprocessor::new();
        let source = "#endif\n";
        let result = pp.process(source, "test.wgsl");
        assert!(matches!(result, Err(ShaderError::PreprocessorMismatch(_))));
    }

    #[test]
    fn test_preprocessor_unclosed_ifdef() {
        let mut pp = ShaderPreprocessor::new();
        let source = "#ifdef SOMETHING\nfoo\n";
        let result = pp.process(source, "test.wgsl");
        assert!(matches!(result, Err(ShaderError::PreprocessorMismatch(_))));
    }

    #[test]
    fn test_wgsl_validation_valid() {
        let source = r#"
@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;
        let result = validate_wgsl(source);
        assert!(result.valid);
        assert_eq!(result.entry_points.len(), 1);
        assert_eq!(result.entry_points[0].name, "vs_main");
        assert_eq!(result.entry_points[0].stage, ShaderStage::Vertex);
    }

    #[test]
    fn test_wgsl_validation_unmatched_braces() {
        let source = "fn foo() { { }";
        let result = validate_wgsl(source);
        assert!(!result.valid);
    }

    #[test]
    fn test_shader_reflection() {
        let source = r#"
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var diffuse_texture: texture_2d<f32>;

@group(0) @binding(2)
var diffuse_sampler: sampler;

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;
        let reflection = reflect_wgsl(source);
        assert_eq!(reflection.bindings.len(), 3);
        assert_eq!(reflection.bindings[0].name, "camera");
        assert_eq!(reflection.bindings[0].binding_type, BindingType::UniformBuffer);
        assert_eq!(reflection.bindings[1].name, "diffuse_texture");
        assert_eq!(reflection.bindings[1].binding_type, BindingType::Texture2D);
        assert_eq!(reflection.bindings[2].name, "diffuse_sampler");
        assert_eq!(reflection.bindings[2].binding_type, BindingType::Sampler);
    }

    #[test]
    fn test_spirv_compilation() {
        let source = r#"
@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;
        let module = compile_wgsl_to_spirv(source, ShaderStage::Vertex, "vs_main").unwrap();
        assert!(module.validate_magic());
        assert!(!module.bytecode.is_empty());
    }

    #[test]
    fn test_spirv_unknown_entry_point() {
        let source = r#"
@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;
        let result = compile_wgsl_to_spirv(source, ShaderStage::Vertex, "nonexistent");
        assert!(matches!(result, Err(ShaderError::UnknownEntryPoint(_))));
    }

    #[test]
    fn test_shader_cache() {
        let mut cache = ShaderCache::new(10);
        assert!(cache.is_empty());
        cache.put("test_key".into(), 123, 456, vec![0x0723_0203, 0]);
        assert_eq!(cache.len(), 1);
        let hit = cache.get("test_key", 123, 456);
        assert!(hit.is_some());
        assert_eq!(cache.hits(), 1);

        // Different hash should miss
        let miss = cache.get("test_key", 999, 456);
        assert!(miss.is_none());
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_shader_cache_invalidation() {
        let mut cache = ShaderCache::new(10);
        cache.put("k1".into(), 1, 1, vec![]);
        cache.invalidate("k1");
        let result = cache.get("k1", 1, 1);
        assert!(result.is_none());
    }

    #[test]
    fn test_fnv1a_hash() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        let h3 = fnv1a_hash(b"world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_shader_compiler_full_pipeline() {
        let mut compiler = ShaderCompiler::new();
        compiler.define("USE_NORMAL_MAP", None);
        compiler.register_include(
            "common.wgsl",
            "struct CameraUniform { view_proj: mat4x4<f32> }",
        );
        let source = r#"
#include "common.wgsl"

#ifdef USE_NORMAL_MAP
var has_normal_map: bool = true;
#else
var has_normal_map: bool = false;
#endif

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;
        let module = compiler
            .compile(source, "test.wgsl", ShaderStage::Vertex, "vs_main")
            .unwrap();
        assert!(module.validate_magic());
        assert_eq!(compiler.stats.compilations, 1);

        // Second compilation should hit cache
        let _module2 = compiler
            .compile(source, "test.wgsl", ShaderStage::Vertex, "vs_main")
            .unwrap();
        assert_eq!(compiler.stats.cache_hits, 1);
    }

    #[test]
    fn test_compute_shader_workgroup_size() {
        let source = r#"
@compute @workgroup_size(8, 8, 1)
fn cs_main() {
}
"#;
        let result = validate_wgsl(source);
        assert!(result.valid);
        assert_eq!(result.entry_points.len(), 1);
        assert_eq!(result.entry_points[0].stage, ShaderStage::Compute);
        assert_eq!(result.entry_points[0].workgroup_size, Some([8, 8, 1]));
    }

    #[test]
    fn test_shader_stage_display() {
        assert_eq!(ShaderStage::Vertex.to_string(), "vertex");
        assert_eq!(ShaderStage::Fragment.to_string(), "fragment");
        assert_eq!(ShaderStage::Compute.to_string(), "compute");
    }

    #[test]
    fn test_spirv_module_to_bytes() {
        let module = SpirVModule {
            bytecode: vec![0x0723_0203, 0x0001_0500],
            version: 0x0001_0500,
            entry_points: vec![],
        };
        let bytes = module.to_bytes();
        assert_eq!(bytes.len(), 8);
        assert_eq!(module.byte_size(), 8);
    }
}
