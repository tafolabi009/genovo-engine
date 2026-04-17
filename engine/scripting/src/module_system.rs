//! Module and import system for the Genovo scripting VM.
//!
//! Provides a module system that allows scripts to organize code into
//! reusable, namespaced units. Modules can export functions and variables,
//! and other scripts can import them.
//!
//! # Architecture
//!
//! ```text
//! import "math"        // Triggers module resolution
//!     |
//!     v
//! ModuleResolver       // Locates the module source
//!     |
//!     v
//! Compile & Execute    // Module body runs once (exports are captured)
//!     |
//!     v
//! ModuleRegistry       // Caches the loaded module for reuse
//! ```
//!
//! # Standard modules
//!
//! - `math`: trigonometry, rounding, random, clamping
//! - `string`: length, substring, find, replace, split, trim
//! - `array`: push, pop, map, filter, reduce, sort, reverse
//! - `io`: print, read_file (sandboxed), write_file (sandboxed)
//! - `engine`: entity creation, component access, world queries
//! - `physics`: raycast, overlap, apply_force
//! - `debug`: log, assert, breakpoint

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum depth of nested imports before circular import detection kicks in.
pub const MAX_IMPORT_DEPTH: usize = 64;

/// Maximum number of modules that can be loaded simultaneously.
pub const MAX_LOADED_MODULES: usize = 256;

/// Maximum number of search paths.
pub const MAX_SEARCH_PATHS: usize = 32;

/// Standard module names.
pub const STD_MODULE_MATH: &str = "math";
pub const STD_MODULE_STRING: &str = "string";
pub const STD_MODULE_ARRAY: &str = "array";
pub const STD_MODULE_IO: &str = "io";
pub const STD_MODULE_ENGINE: &str = "engine";
pub const STD_MODULE_PHYSICS: &str = "physics";
pub const STD_MODULE_DEBUG: &str = "debug";

// ---------------------------------------------------------------------------
// ModuleError
// ---------------------------------------------------------------------------

/// Errors that can occur during module loading.
#[derive(Debug, Clone)]
pub enum ModuleError {
    /// Module not found in any search path.
    NotFound(String),
    /// Circular import detected.
    CircularImport {
        module: String,
        chain: Vec<String>,
    },
    /// Compilation error in the module source.
    CompileError {
        module: String,
        message: String,
        line: usize,
    },
    /// Runtime error during module initialization.
    RuntimeError {
        module: String,
        message: String,
    },
    /// Too many modules loaded.
    ModuleLimitExceeded,
    /// An exported symbol was not found.
    ExportNotFound {
        module: String,
        symbol: String,
    },
    /// I/O error reading module file.
    IoError(String),
    /// Import depth exceeded (likely circular).
    MaxDepthExceeded(String),
}

impl fmt::Display for ModuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModuleError::NotFound(name) => write!(f, "module '{}' not found", name),
            ModuleError::CircularImport { module, chain } => {
                write!(
                    f,
                    "circular import detected for '{}': {}",
                    module,
                    chain.join(" -> ")
                )
            }
            ModuleError::CompileError {
                module,
                message,
                line,
            } => write!(
                f,
                "compile error in module '{}' at line {}: {}",
                module, line, message
            ),
            ModuleError::RuntimeError { module, message } => {
                write!(f, "runtime error in module '{}': {}", module, message)
            }
            ModuleError::ModuleLimitExceeded => {
                write!(f, "maximum number of loaded modules exceeded")
            }
            ModuleError::ExportNotFound { module, symbol } => {
                write!(f, "module '{}' does not export '{}'", module, symbol)
            }
            ModuleError::IoError(msg) => write!(f, "I/O error: {}", msg),
            ModuleError::MaxDepthExceeded(name) => {
                write!(f, "import depth exceeded for '{}' (circular?)", name)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ExportedValue
// ---------------------------------------------------------------------------

/// A value exported by a module.
#[derive(Debug, Clone)]
pub enum ExportedValue {
    /// An exported function (identified by name and arity).
    Function {
        name: String,
        arity: u8,
        /// Compiled bytecode index in the module's chunk.
        bytecode_offset: usize,
    },
    /// An exported variable with its current value.
    Variable {
        name: String,
        value: ModuleValue,
    },
    /// An exported constant.
    Constant {
        name: String,
        value: ModuleValue,
    },
}

impl ExportedValue {
    /// Returns the name of this export.
    pub fn name(&self) -> &str {
        match self {
            ExportedValue::Function { name, .. } => name,
            ExportedValue::Variable { name, .. } => name,
            ExportedValue::Constant { name, .. } => name,
        }
    }

    /// Returns `true` if this export is a function.
    pub fn is_function(&self) -> bool {
        matches!(self, ExportedValue::Function { .. })
    }

    /// Returns `true` if this export is a variable.
    pub fn is_variable(&self) -> bool {
        matches!(self, ExportedValue::Variable { .. })
    }
}

// ---------------------------------------------------------------------------
// ModuleValue — simplified value for module exports
// ---------------------------------------------------------------------------

/// A simplified value type for module-level exports.
///
/// This mirrors the main `ScriptValue` but is self-contained so the module
/// system does not need to depend directly on the VM's value type.
#[derive(Debug, Clone)]
pub enum ModuleValue {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Arc<str>),
    Vec3(f32, f32, f32),
    Array(Vec<ModuleValue>),
    Map(HashMap<String, ModuleValue>),
}

impl ModuleValue {
    /// Returns `true` if this value is nil.
    pub fn is_nil(&self) -> bool {
        matches!(self, ModuleValue::Nil)
    }

    /// Try to convert to f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ModuleValue::Float(v) => Some(*v),
            ModuleValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to convert to i64.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ModuleValue::Int(v) => Some(*v),
            ModuleValue::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Try to convert to string slice.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ModuleValue::String(s) => Some(s),
            _ => None,
        }
    }
}

impl fmt::Display for ModuleValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModuleValue::Nil => write!(f, "nil"),
            ModuleValue::Bool(v) => write!(f, "{}", v),
            ModuleValue::Int(v) => write!(f, "{}", v),
            ModuleValue::Float(v) => write!(f, "{}", v),
            ModuleValue::String(s) => write!(f, "\"{}\"", s),
            ModuleValue::Vec3(x, y, z) => write!(f, "vec3({}, {}, {})", x, y, z),
            ModuleValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            ModuleValue::Map(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CompiledChunk (placeholder for module bytecode)
// ---------------------------------------------------------------------------

/// Compiled bytecode chunk for a module.
///
/// This is a simplified representation. The actual bytecode lives in the
/// VM's `Chunk` type; this struct holds the serialized form for caching.
#[derive(Debug, Clone)]
pub struct CompiledChunk {
    /// Raw bytecode instructions.
    pub bytecode: Vec<u8>,
    /// Constant pool (serialized values).
    pub constants: Vec<ModuleValue>,
    /// Source filename (for error reporting).
    pub source_name: String,
    /// Number of local variables used.
    pub local_count: usize,
}

impl CompiledChunk {
    /// Create an empty chunk.
    pub fn empty(source_name: impl Into<String>) -> Self {
        Self {
            bytecode: Vec::new(),
            constants: Vec::new(),
            source_name: source_name.into(),
            local_count: 0,
        }
    }

    /// Returns `true` if this chunk has no bytecode.
    pub fn is_empty(&self) -> bool {
        self.bytecode.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ScriptModule
// ---------------------------------------------------------------------------

/// A loaded script module.
///
/// Contains the module's compiled bytecode, exported functions and
/// variables, and metadata.
#[derive(Debug, Clone)]
pub struct ScriptModule {
    /// Module name (e.g., "math", "game.player").
    pub name: String,
    /// Resolved file path (None for built-in/standard modules).
    pub file_path: Option<PathBuf>,
    /// Compiled bytecode chunk.
    pub chunk: CompiledChunk,
    /// Exported symbols.
    pub exports: HashMap<String, ExportedValue>,
    /// Dependencies (names of modules this module imports).
    pub dependencies: Vec<String>,
    /// Whether this is a standard library module.
    pub is_stdlib: bool,
    /// Whether this module has been fully initialized (body executed).
    pub initialized: bool,
    /// Version or hash for hot-reload detection.
    pub content_hash: u64,
}

impl ScriptModule {
    /// Create a new uninitialized module.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            file_path: None,
            chunk: CompiledChunk::empty(""),
            exports: HashMap::new(),
            dependencies: Vec::new(),
            is_stdlib: false,
            initialized: false,
            content_hash: 0,
        }
    }

    /// Mark this module as a standard library module.
    pub fn as_stdlib(mut self) -> Self {
        self.is_stdlib = true;
        self.initialized = true;
        self
    }

    /// Set the file path.
    pub fn with_file_path(mut self, path: PathBuf) -> Self {
        self.file_path = Some(path);
        self
    }

    /// Add an exported function.
    pub fn export_function(
        &mut self,
        name: impl Into<String>,
        arity: u8,
        offset: usize,
    ) {
        let name = name.into();
        self.exports.insert(
            name.clone(),
            ExportedValue::Function {
                name,
                arity,
                bytecode_offset: offset,
            },
        );
    }

    /// Add an exported variable.
    pub fn export_variable(&mut self, name: impl Into<String>, value: ModuleValue) {
        let name = name.into();
        self.exports.insert(
            name.clone(),
            ExportedValue::Variable {
                name,
                value,
            },
        );
    }

    /// Add an exported constant.
    pub fn export_constant(&mut self, name: impl Into<String>, value: ModuleValue) {
        let name = name.into();
        self.exports.insert(
            name.clone(),
            ExportedValue::Constant {
                name,
                value,
            },
        );
    }

    /// Get an export by name.
    pub fn get_export(&self, name: &str) -> Option<&ExportedValue> {
        self.exports.get(name)
    }

    /// Get all exported function names.
    pub fn exported_functions(&self) -> Vec<&str> {
        self.exports
            .values()
            .filter_map(|e| {
                if e.is_function() {
                    Some(e.name())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all exported variable names.
    pub fn exported_variables(&self) -> Vec<&str> {
        self.exports
            .values()
            .filter_map(|e| {
                if e.is_variable() {
                    Some(e.name())
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ModuleResolver
// ---------------------------------------------------------------------------

/// Resolves module names to source code.
///
/// Searches a list of paths for module files and supports standard
/// library modules that are built-in.
#[derive(Debug, Clone)]
pub struct ModuleResolver {
    /// Ordered list of directories to search for module files.
    pub search_paths: Vec<PathBuf>,
    /// File extension for script module files (default: ".gvs").
    pub file_extension: String,
    /// Standard library module names that are built-in.
    pub stdlib_modules: HashSet<String>,
}

impl ModuleResolver {
    /// Create a new resolver with default settings.
    pub fn new() -> Self {
        let mut stdlib = HashSet::new();
        stdlib.insert(STD_MODULE_MATH.to_string());
        stdlib.insert(STD_MODULE_STRING.to_string());
        stdlib.insert(STD_MODULE_ARRAY.to_string());
        stdlib.insert(STD_MODULE_IO.to_string());
        stdlib.insert(STD_MODULE_ENGINE.to_string());
        stdlib.insert(STD_MODULE_PHYSICS.to_string());
        stdlib.insert(STD_MODULE_DEBUG.to_string());

        Self {
            search_paths: Vec::new(),
            file_extension: ".gvs".to_string(),
            stdlib_modules: stdlib,
        }
    }

    /// Add a search path.
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        let path = path.into();
        if self.search_paths.len() < MAX_SEARCH_PATHS && !self.search_paths.contains(&path) {
            self.search_paths.push(path);
        }
    }

    /// Check if a module name refers to a standard library module.
    pub fn is_stdlib(&self, name: &str) -> bool {
        self.stdlib_modules.contains(name)
    }

    /// Resolve a module name to a file path.
    ///
    /// For standard modules, returns `None` (they are built-in).
    /// For user modules, searches the search paths in order.
    pub fn resolve(&self, name: &str) -> Result<Option<PathBuf>, ModuleError> {
        if self.is_stdlib(name) {
            return Ok(None);
        }

        // Convert dotted module names to path separators.
        let relative = name.replace('.', "/");
        let filename = format!("{}{}", relative, self.file_extension);

        for search_dir in &self.search_paths {
            let candidate = search_dir.join(&filename);
            if candidate.exists() {
                return Ok(Some(candidate));
            }

            // Also try as a directory with mod.gvs.
            let dir_candidate = search_dir.join(&relative).join(format!("mod{}", self.file_extension));
            if dir_candidate.exists() {
                return Ok(Some(dir_candidate));
            }
        }

        Err(ModuleError::NotFound(name.to_string()))
    }
}

impl Default for ModuleResolver {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ImportStatement
// ---------------------------------------------------------------------------

/// A parsed import statement.
#[derive(Debug, Clone)]
pub struct ImportStatement {
    /// The module being imported (e.g., "math").
    pub module_name: String,
    /// Specific symbols to import (empty = import all).
    pub symbols: Vec<String>,
    /// Optional alias for the module (e.g., `import "math" as m`).
    pub alias: Option<String>,
}

impl ImportStatement {
    /// Create a simple import (import all exports).
    pub fn all(module_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            symbols: Vec::new(),
            alias: None,
        }
    }

    /// Create a selective import.
    pub fn symbols(module_name: impl Into<String>, symbols: Vec<String>) -> Self {
        Self {
            module_name: module_name.into(),
            symbols,
            alias: None,
        }
    }

    /// Create an aliased import.
    pub fn aliased(module_name: impl Into<String>, alias: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            symbols: Vec::new(),
            alias: Some(alias.into()),
        }
    }

    /// Returns `true` if this imports all exports.
    pub fn is_wildcard(&self) -> bool {
        self.symbols.is_empty() && self.alias.is_none()
    }
}

// ---------------------------------------------------------------------------
// Standard Module Builders
// ---------------------------------------------------------------------------

/// Build the standard `math` module with exported functions and constants.
pub fn build_math_module() -> ScriptModule {
    let mut module = ScriptModule::new(STD_MODULE_MATH).as_stdlib();
    module.chunk = CompiledChunk::empty("std:math");

    // Constants.
    module.export_constant("PI", ModuleValue::Float(std::f64::consts::PI));
    module.export_constant("E", ModuleValue::Float(std::f64::consts::E));
    module.export_constant("TAU", ModuleValue::Float(std::f64::consts::TAU));
    module.export_constant("INF", ModuleValue::Float(f64::INFINITY));
    module.export_constant("NAN", ModuleValue::Float(f64::NAN));

    // Functions (arity, offset is symbolic for builtins).
    module.export_function("sin", 1, 0);
    module.export_function("cos", 1, 0);
    module.export_function("tan", 1, 0);
    module.export_function("asin", 1, 0);
    module.export_function("acos", 1, 0);
    module.export_function("atan", 1, 0);
    module.export_function("atan2", 2, 0);
    module.export_function("sqrt", 1, 0);
    module.export_function("abs", 1, 0);
    module.export_function("floor", 1, 0);
    module.export_function("ceil", 1, 0);
    module.export_function("round", 1, 0);
    module.export_function("min", 2, 0);
    module.export_function("max", 2, 0);
    module.export_function("clamp", 3, 0);
    module.export_function("lerp", 3, 0);
    module.export_function("pow", 2, 0);
    module.export_function("log", 1, 0);
    module.export_function("log2", 1, 0);
    module.export_function("exp", 1, 0);
    module.export_function("sign", 1, 0);
    module.export_function("fract", 1, 0);
    module.export_function("random", 0, 0);
    module.export_function("random_range", 2, 0);

    module
}

/// Build the standard `string` module.
pub fn build_string_module() -> ScriptModule {
    let mut module = ScriptModule::new(STD_MODULE_STRING).as_stdlib();
    module.chunk = CompiledChunk::empty("std:string");

    module.export_function("length", 1, 0);
    module.export_function("substring", 3, 0);
    module.export_function("find", 2, 0);
    module.export_function("replace", 3, 0);
    module.export_function("split", 2, 0);
    module.export_function("trim", 1, 0);
    module.export_function("trim_start", 1, 0);
    module.export_function("trim_end", 1, 0);
    module.export_function("to_upper", 1, 0);
    module.export_function("to_lower", 1, 0);
    module.export_function("starts_with", 2, 0);
    module.export_function("ends_with", 2, 0);
    module.export_function("contains", 2, 0);
    module.export_function("char_at", 2, 0);
    module.export_function("repeat", 2, 0);
    module.export_function("reverse", 1, 0);
    module.export_function("format", 2, 0);
    module.export_function("parse_int", 1, 0);
    module.export_function("parse_float", 1, 0);
    module.export_function("to_string", 1, 0);

    module
}

/// Build the standard `array` module.
pub fn build_array_module() -> ScriptModule {
    let mut module = ScriptModule::new(STD_MODULE_ARRAY).as_stdlib();
    module.chunk = CompiledChunk::empty("std:array");

    module.export_function("push", 2, 0);
    module.export_function("pop", 1, 0);
    module.export_function("length", 1, 0);
    module.export_function("get", 2, 0);
    module.export_function("set", 3, 0);
    module.export_function("insert", 3, 0);
    module.export_function("remove", 2, 0);
    module.export_function("clear", 1, 0);
    module.export_function("contains", 2, 0);
    module.export_function("index_of", 2, 0);
    module.export_function("sort", 1, 0);
    module.export_function("reverse", 1, 0);
    module.export_function("map", 2, 0);
    module.export_function("filter", 2, 0);
    module.export_function("reduce", 3, 0);
    module.export_function("flatten", 1, 0);
    module.export_function("slice", 3, 0);
    module.export_function("join", 2, 0);
    module.export_function("zip", 2, 0);
    module.export_function("enumerate", 1, 0);

    module
}

/// Build the standard `io` module (sandboxed).
pub fn build_io_module() -> ScriptModule {
    let mut module = ScriptModule::new(STD_MODULE_IO).as_stdlib();
    module.chunk = CompiledChunk::empty("std:io");

    module.export_function("print", 1, 0);
    module.export_function("println", 1, 0);
    module.export_function("read_file", 1, 0);
    module.export_function("write_file", 2, 0);
    module.export_function("file_exists", 1, 0);
    module.export_function("input", 1, 0);

    module
}

/// Build the standard `engine` module.
pub fn build_engine_module() -> ScriptModule {
    let mut module = ScriptModule::new(STD_MODULE_ENGINE).as_stdlib();
    module.chunk = CompiledChunk::empty("std:engine");

    module.export_function("spawn_entity", 0, 0);
    module.export_function("destroy_entity", 1, 0);
    module.export_function("get_component", 2, 0);
    module.export_function("set_component", 3, 0);
    module.export_function("has_component", 2, 0);
    module.export_function("add_component", 2, 0);
    module.export_function("remove_component", 2, 0);
    module.export_function("get_position", 1, 0);
    module.export_function("set_position", 2, 0);
    module.export_function("get_rotation", 1, 0);
    module.export_function("set_rotation", 2, 0);
    module.export_function("get_scale", 1, 0);
    module.export_function("set_scale", 2, 0);
    module.export_function("find_entity", 1, 0);
    module.export_function("query_entities", 1, 0);
    module.export_function("delta_time", 0, 0);
    module.export_function("total_time", 0, 0);

    module
}

/// Build the standard `physics` module.
pub fn build_physics_module() -> ScriptModule {
    let mut module = ScriptModule::new(STD_MODULE_PHYSICS).as_stdlib();
    module.chunk = CompiledChunk::empty("std:physics");

    module.export_function("raycast", 3, 0);
    module.export_function("overlap_sphere", 2, 0);
    module.export_function("overlap_box", 3, 0);
    module.export_function("apply_force", 2, 0);
    module.export_function("apply_impulse", 2, 0);
    module.export_function("set_velocity", 2, 0);
    module.export_function("get_velocity", 1, 0);
    module.export_function("set_gravity", 1, 0);

    module
}

/// Build the standard `debug` module.
pub fn build_debug_module() -> ScriptModule {
    let mut module = ScriptModule::new(STD_MODULE_DEBUG).as_stdlib();
    module.chunk = CompiledChunk::empty("std:debug");

    module.export_function("log", 1, 0);
    module.export_function("warn", 1, 0);
    module.export_function("error", 1, 0);
    module.export_function("assert", 2, 0);
    module.export_function("assert_eq", 3, 0);
    module.export_function("breakpoint", 0, 0);
    module.export_function("type_of", 1, 0);
    module.export_function("dump", 1, 0);
    module.export_function("time", 0, 0);
    module.export_function("trace", 0, 0);

    module
}

/// Build and return all standard library modules.
pub fn build_all_stdlib() -> Vec<ScriptModule> {
    vec![
        build_math_module(),
        build_string_module(),
        build_array_module(),
        build_io_module(),
        build_engine_module(),
        build_physics_module(),
        build_debug_module(),
    ]
}

// ---------------------------------------------------------------------------
// ModuleRegistry
// ---------------------------------------------------------------------------

/// Central registry of loaded modules.
///
/// Manages module caching, dependency tracking, and provides lookups
/// by name. Modules are loaded once and reused for all subsequent
/// import statements.
pub struct ModuleRegistry {
    /// Loaded modules, keyed by module name.
    modules: HashMap<String, ScriptModule>,
    /// Module resolver for finding source files.
    resolver: ModuleResolver,
    /// Set of module names currently being loaded (for circular import
    /// detection).
    loading_stack: Vec<String>,
    /// Whether standard library modules have been pre-loaded.
    stdlib_loaded: bool,
}

impl ModuleRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            resolver: ModuleResolver::new(),
            loading_stack: Vec::new(),
            stdlib_loaded: false,
        }
    }

    /// Create a registry with a custom resolver.
    pub fn with_resolver(resolver: ModuleResolver) -> Self {
        Self {
            modules: HashMap::new(),
            resolver,
            loading_stack: Vec::new(),
            stdlib_loaded: false,
        }
    }

    /// Pre-load all standard library modules.
    pub fn load_stdlib(&mut self) {
        if self.stdlib_loaded {
            return;
        }
        for module in build_all_stdlib() {
            self.modules.insert(module.name.clone(), module);
        }
        self.stdlib_loaded = true;
    }

    /// Add a search path for module resolution.
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        self.resolver.add_search_path(path);
    }

    /// Register a pre-built module (e.g., from native Rust code).
    pub fn register_module(&mut self, module: ScriptModule) -> Result<(), ModuleError> {
        if self.modules.len() >= MAX_LOADED_MODULES {
            return Err(ModuleError::ModuleLimitExceeded);
        }
        self.modules.insert(module.name.clone(), module);
        Ok(())
    }

    /// Get a loaded module by name.
    pub fn get_module(&self, name: &str) -> Option<&ScriptModule> {
        self.modules.get(name)
    }

    /// Get a mutable reference to a loaded module.
    pub fn get_module_mut(&mut self, name: &str) -> Option<&mut ScriptModule> {
        self.modules.get_mut(name)
    }

    /// Check if a module is loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// Begin loading a module (for circular import detection).
    /// Returns an error if a circular import is detected.
    pub fn begin_loading(&mut self, name: &str) -> Result<(), ModuleError> {
        if self.loading_stack.contains(&name.to_string()) {
            let mut chain = self.loading_stack.clone();
            chain.push(name.to_string());
            return Err(ModuleError::CircularImport {
                module: name.to_string(),
                chain,
            });
        }
        if self.loading_stack.len() >= MAX_IMPORT_DEPTH {
            return Err(ModuleError::MaxDepthExceeded(name.to_string()));
        }
        self.loading_stack.push(name.to_string());
        Ok(())
    }

    /// Finish loading a module (pop from the loading stack).
    pub fn finish_loading(&mut self, name: &str) {
        if let Some(pos) = self.loading_stack.iter().position(|n| n == name) {
            self.loading_stack.remove(pos);
        }
    }

    /// Import a module by name.
    ///
    /// If the module is already loaded, returns a reference to the cached
    /// version. Otherwise, resolves and loads the module.
    ///
    /// For file-based modules, this performs:
    /// 1. Resolution (find the file)
    /// 2. Loading (read the source)
    /// 3. Compilation (compile to bytecode)
    /// 4. Initialization (execute the module body)
    /// 5. Caching (store for future imports)
    pub fn import(&mut self, name: &str) -> Result<&ScriptModule, ModuleError> {
        // Return cached module if already loaded.
        if self.modules.contains_key(name) {
            return Ok(self.modules.get(name).unwrap());
        }

        // Check for circular imports.
        self.begin_loading(name)?;

        // Resolve the module.
        let resolved = self.resolver.resolve(name)?;

        match resolved {
            None => {
                // Standard library module — should have been pre-loaded.
                self.finish_loading(name);
                self.modules
                    .get(name)
                    .ok_or_else(|| ModuleError::NotFound(name.to_string()))
            }
            Some(path) => {
                // File-based module: read source.
                let source = std::fs::read_to_string(&path)
                    .map_err(|e| ModuleError::IoError(format!("{}: {}", path.display(), e)))?;

                // Create a new module with the source.
                let mut module = ScriptModule::new(name);
                module.file_path = Some(path);

                // Compute a simple content hash.
                module.content_hash = simple_hash(&source);

                // In a real implementation, we would compile the source here.
                // For now, we create the module with an empty chunk.
                module.chunk = CompiledChunk::empty(name);
                module.initialized = true;

                if self.modules.len() >= MAX_LOADED_MODULES {
                    self.finish_loading(name);
                    return Err(ModuleError::ModuleLimitExceeded);
                }

                self.modules.insert(name.to_string(), module);
                self.finish_loading(name);

                Ok(self.modules.get(name).unwrap())
            }
        }
    }

    /// Unload a module by name.
    pub fn unload(&mut self, name: &str) -> Option<ScriptModule> {
        self.modules.remove(name)
    }

    /// Unload all non-stdlib modules.
    pub fn unload_user_modules(&mut self) {
        self.modules.retain(|_, m| m.is_stdlib);
    }

    /// Returns the number of loaded modules.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// List all loaded module names.
    pub fn loaded_module_names(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }

    /// Get the resolver.
    pub fn resolver(&self) -> &ModuleResolver {
        &self.resolver
    }

    /// Get a mutable reference to the resolver.
    pub fn resolver_mut(&mut self) -> &mut ModuleResolver {
        &mut self.resolver
    }

    /// Look up an exported value from a module.
    pub fn get_export(
        &self,
        module_name: &str,
        symbol: &str,
    ) -> Result<&ExportedValue, ModuleError> {
        let module = self
            .modules
            .get(module_name)
            .ok_or_else(|| ModuleError::NotFound(module_name.to_string()))?;

        module.get_export(symbol).ok_or_else(|| ModuleError::ExportNotFound {
            module: module_name.to_string(),
            symbol: symbol.to_string(),
        })
    }

    /// Clear all modules and reset.
    pub fn clear(&mut self) {
        self.modules.clear();
        self.loading_stack.clear();
        self.stdlib_loaded = false;
    }
}

impl Default for ModuleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple FNV-1a-style hash for content change detection.
fn simple_hash(data: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_value_display() {
        assert_eq!(format!("{}", ModuleValue::Nil), "nil");
        assert_eq!(format!("{}", ModuleValue::Bool(true)), "true");
        assert_eq!(format!("{}", ModuleValue::Int(42)), "42");
        assert_eq!(format!("{}", ModuleValue::Float(3.14)), "3.14");
        assert_eq!(
            format!("{}", ModuleValue::String(Arc::from("hello"))),
            "\"hello\""
        );
    }

    #[test]
    fn test_module_value_conversions() {
        assert_eq!(ModuleValue::Int(10).as_float(), Some(10.0));
        assert_eq!(ModuleValue::Float(3.14).as_int(), Some(3));
        assert_eq!(ModuleValue::String(Arc::from("hi")).as_str(), Some("hi"));
        assert_eq!(ModuleValue::Nil.as_float(), None);
    }

    #[test]
    fn test_build_math_module() {
        let module = build_math_module();
        assert!(module.is_stdlib);
        assert!(module.initialized);
        assert!(module.get_export("sin").is_some());
        assert!(module.get_export("PI").is_some());
        assert!(module.get_export("nonexistent").is_none());
    }

    #[test]
    fn test_build_all_stdlib() {
        let modules = build_all_stdlib();
        assert_eq!(modules.len(), 7);
        let names: Vec<&str> = modules.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"math"));
        assert!(names.contains(&"string"));
        assert!(names.contains(&"array"));
        assert!(names.contains(&"io"));
        assert!(names.contains(&"engine"));
        assert!(names.contains(&"physics"));
        assert!(names.contains(&"debug"));
    }

    #[test]
    fn test_module_exports() {
        let mut module = ScriptModule::new("test");
        module.export_function("foo", 2, 0);
        module.export_variable("bar", ModuleValue::Int(42));

        assert_eq!(module.exported_functions(), vec!["foo"]);
        assert_eq!(module.exported_variables(), vec!["bar"]);
    }

    #[test]
    fn test_resolver_stdlib() {
        let resolver = ModuleResolver::new();
        assert!(resolver.is_stdlib("math"));
        assert!(resolver.is_stdlib("string"));
        assert!(!resolver.is_stdlib("my_game_module"));

        // Stdlib resolves to None path.
        let result = resolver.resolve("math");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_resolver_not_found() {
        let resolver = ModuleResolver::new();
        let result = resolver.resolve("nonexistent_user_module");
        assert!(matches!(result, Err(ModuleError::NotFound(_))));
    }

    #[test]
    fn test_import_statement() {
        let import = ImportStatement::all("math");
        assert!(import.is_wildcard());
        assert_eq!(import.module_name, "math");

        let import = ImportStatement::symbols("math", vec!["sin".into(), "cos".into()]);
        assert!(!import.is_wildcard());
        assert_eq!(import.symbols.len(), 2);

        let import = ImportStatement::aliased("math", "m");
        assert!(!import.is_wildcard());
        assert_eq!(import.alias, Some("m".to_string()));
    }

    #[test]
    fn test_registry_load_stdlib() {
        let mut registry = ModuleRegistry::new();
        assert_eq!(registry.module_count(), 0);

        registry.load_stdlib();
        assert_eq!(registry.module_count(), 7);
        assert!(registry.is_loaded("math"));
        assert!(registry.is_loaded("debug"));
    }

    #[test]
    fn test_registry_register_custom() {
        let mut registry = ModuleRegistry::new();
        let module = ScriptModule::new("custom");
        registry.register_module(module).unwrap();
        assert!(registry.is_loaded("custom"));
    }

    #[test]
    fn test_registry_circular_import_detection() {
        let mut registry = ModuleRegistry::new();
        registry.begin_loading("a").unwrap();
        registry.begin_loading("b").unwrap();

        let result = registry.begin_loading("a");
        assert!(matches!(result, Err(ModuleError::CircularImport { .. })));
    }

    #[test]
    fn test_registry_import_cached() {
        let mut registry = ModuleRegistry::new();
        registry.load_stdlib();

        // Importing "math" should return the cached module.
        let module = registry.import("math").unwrap();
        assert_eq!(module.name, "math");
        assert!(module.is_stdlib);
    }

    #[test]
    fn test_registry_get_export() {
        let mut registry = ModuleRegistry::new();
        registry.load_stdlib();

        let export = registry.get_export("math", "sin");
        assert!(export.is_ok());

        let export = registry.get_export("math", "nonexistent");
        assert!(matches!(export, Err(ModuleError::ExportNotFound { .. })));

        let export = registry.get_export("nonexistent", "sin");
        assert!(matches!(export, Err(ModuleError::NotFound(_))));
    }

    #[test]
    fn test_registry_unload() {
        let mut registry = ModuleRegistry::new();
        registry.load_stdlib();

        let custom = ScriptModule::new("custom");
        registry.register_module(custom).unwrap();
        assert_eq!(registry.module_count(), 8);

        registry.unload_user_modules();
        assert_eq!(registry.module_count(), 7);
        assert!(!registry.is_loaded("custom"));
    }

    #[test]
    fn test_simple_hash() {
        let h1 = simple_hash("hello");
        let h2 = simple_hash("hello");
        let h3 = simple_hash("world");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_compiled_chunk() {
        let chunk = CompiledChunk::empty("test");
        assert!(chunk.is_empty());
        assert_eq!(chunk.source_name, "test");
    }
}
