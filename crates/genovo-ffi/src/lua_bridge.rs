//! # Lua Language Bridge
//!
//! Provides a bridge between the Genovo engine and Lua scripts: Lua state
//! management, pushing/popping values, calling Lua functions from Rust and
//! Rust functions from Lua, table manipulation, coroutine support, error
//! handling, metatables for engine types, and a sandbox to restrict
//! dangerous operations.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur in the Lua bridge.
#[derive(Debug, Clone)]
pub enum LuaError {
    /// A Lua runtime error (message from pcall).
    RuntimeError(String),
    /// A Lua syntax error in source code.
    SyntaxError(String),
    /// A Lua memory allocation error.
    MemoryError,
    /// Attempt to call a non-function value.
    NotCallable(String),
    /// Stack overflow.
    StackOverflow,
    /// Type mismatch when reading a value.
    TypeError { expected: String, got: String },
    /// A Rust callback panicked.
    CallbackPanic(String),
    /// Sandbox violation (blocked operation).
    SandboxViolation(String),
    /// The Lua state has been closed.
    StateClosed,
    /// A coroutine error.
    CoroutineError(String),
    /// A table key was not found.
    KeyNotFound(String),
    /// An argument count mismatch.
    ArgCountMismatch { expected: usize, got: usize },
}

impl fmt::Display for LuaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeError(msg) => write!(f, "Lua runtime error: {msg}"),
            Self::SyntaxError(msg) => write!(f, "Lua syntax error: {msg}"),
            Self::MemoryError => write!(f, "Lua memory error"),
            Self::NotCallable(name) => write!(f, "'{name}' is not callable"),
            Self::StackOverflow => write!(f, "Lua stack overflow"),
            Self::TypeError { expected, got } => {
                write!(f, "type error: expected {expected}, got {got}")
            }
            Self::CallbackPanic(msg) => write!(f, "callback panic: {msg}"),
            Self::SandboxViolation(msg) => write!(f, "sandbox violation: {msg}"),
            Self::StateClosed => write!(f, "Lua state is closed"),
            Self::CoroutineError(msg) => write!(f, "coroutine error: {msg}"),
            Self::KeyNotFound(key) => write!(f, "key not found: {key}"),
            Self::ArgCountMismatch { expected, got } => {
                write!(f, "expected {expected} args, got {got}")
            }
        }
    }
}

impl std::error::Error for LuaError {}

/// Result type for Lua operations.
pub type LuaResult<T> = Result<T, LuaError>;

// ---------------------------------------------------------------------------
// Lua value types
// ---------------------------------------------------------------------------

/// A value that can be passed to/from Lua.
#[derive(Debug, Clone)]
pub enum LuaValue {
    Nil,
    Boolean(bool),
    Integer(i64),
    Number(f64),
    String(String),
    Table(LuaTable),
    Function(LuaFunction),
    UserData(LuaUserData),
    LightUserData(usize),
}

impl LuaValue {
    /// Type name as a string.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Nil => "nil",
            Self::Boolean(_) => "boolean",
            Self::Integer(_) => "number",
            Self::Number(_) => "number",
            Self::String(_) => "string",
            Self::Table(_) => "table",
            Self::Function(_) => "function",
            Self::UserData(_) => "userdata",
            Self::LightUserData(_) => "lightuserdata",
        }
    }

    /// Try to convert to bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            Self::Nil => Some(false),
            _ => None,
        }
    }

    /// Try to convert to i64.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            Self::Number(n) => Some(*n as i64),
            _ => None,
        }
    }

    /// Try to convert to f64.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            Self::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to convert to string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to convert to table.
    pub fn as_table(&self) -> Option<&LuaTable> {
        match self {
            Self::Table(t) => Some(t),
            _ => None,
        }
    }

    /// Check if the value is nil.
    pub fn is_nil(&self) -> bool {
        matches!(self, Self::Nil)
    }

    /// Truthiness (Lua rules: nil and false are falsy, everything else is truthy).
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Nil => false,
            Self::Boolean(b) => *b,
            _ => true,
        }
    }

    /// Convert to a display string.
    pub fn to_display_string(&self) -> String {
        match self {
            Self::Nil => "nil".to_string(),
            Self::Boolean(b) => b.to_string(),
            Self::Integer(i) => i.to_string(),
            Self::Number(n) => format!("{n}"),
            Self::String(s) => format!("\"{s}\""),
            Self::Table(_) => "table".to_string(),
            Self::Function(_) => "function".to_string(),
            Self::UserData(ud) => format!("userdata({})", ud.type_name),
            Self::LightUserData(ptr) => format!("lightuserdata({ptr:#x})"),
        }
    }
}

impl Default for LuaValue {
    fn default() -> Self {
        Self::Nil
    }
}

impl From<bool> for LuaValue {
    fn from(b: bool) -> Self { Self::Boolean(b) }
}

impl From<i32> for LuaValue {
    fn from(i: i32) -> Self { Self::Integer(i as i64) }
}

impl From<i64> for LuaValue {
    fn from(i: i64) -> Self { Self::Integer(i) }
}

impl From<f32> for LuaValue {
    fn from(f: f32) -> Self { Self::Number(f as f64) }
}

impl From<f64> for LuaValue {
    fn from(f: f64) -> Self { Self::Number(f) }
}

impl From<&str> for LuaValue {
    fn from(s: &str) -> Self { Self::String(s.to_string()) }
}

impl From<String> for LuaValue {
    fn from(s: String) -> Self { Self::String(s) }
}

// ---------------------------------------------------------------------------
// Lua table
// ---------------------------------------------------------------------------

/// A Lua table (hash part + array part).
#[derive(Debug, Clone)]
pub struct LuaTable {
    /// Array part (1-based indexing in Lua).
    array: Vec<LuaValue>,
    /// Hash part.
    hash: HashMap<String, LuaValue>,
    /// Metatable name (if any).
    metatable: Option<String>,
}

impl LuaTable {
    /// Create an empty table.
    pub fn new() -> Self {
        Self {
            array: Vec::new(),
            hash: HashMap::new(),
            metatable: None,
        }
    }

    /// Create a table with pre-allocated capacity.
    pub fn with_capacity(array_cap: usize, hash_cap: usize) -> Self {
        Self {
            array: Vec::with_capacity(array_cap),
            hash: HashMap::with_capacity(hash_cap),
            metatable: None,
        }
    }

    /// Set a string-keyed value.
    pub fn set(&mut self, key: impl Into<String>, value: LuaValue) {
        self.hash.insert(key.into(), value);
    }

    /// Get a string-keyed value.
    pub fn get(&self, key: &str) -> Option<&LuaValue> {
        self.hash.get(key)
    }

    /// Remove a string-keyed value.
    pub fn remove(&mut self, key: &str) -> Option<LuaValue> {
        self.hash.remove(key)
    }

    /// Set an integer-keyed value (array part, 1-based).
    pub fn set_index(&mut self, index: usize, value: LuaValue) {
        if index == 0 {
            return; // Lua uses 1-based indexing
        }
        let idx = index - 1;
        if idx >= self.array.len() {
            self.array.resize(idx + 1, LuaValue::Nil);
        }
        self.array[idx] = value;
    }

    /// Get an integer-keyed value (1-based).
    pub fn get_index(&self, index: usize) -> Option<&LuaValue> {
        if index == 0 {
            return None;
        }
        self.array.get(index - 1)
    }

    /// Append a value to the array part.
    pub fn push(&mut self, value: LuaValue) {
        self.array.push(value);
    }

    /// Pop from the array part.
    pub fn pop(&mut self) -> Option<LuaValue> {
        self.array.pop()
    }

    /// Length of the array part (Lua `#` operator).
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.array.is_empty() && self.hash.is_empty()
    }

    /// Number of hash entries.
    pub fn hash_len(&self) -> usize {
        self.hash.len()
    }

    /// Total number of entries.
    pub fn total_len(&self) -> usize {
        self.array.len() + self.hash.len()
    }

    /// Set the metatable name.
    pub fn set_metatable(&mut self, name: impl Into<String>) {
        self.metatable = Some(name.into());
    }

    /// Get the metatable name.
    pub fn metatable(&self) -> Option<&str> {
        self.metatable.as_deref()
    }

    /// Iterate over hash entries.
    pub fn iter_hash(&self) -> impl Iterator<Item = (&str, &LuaValue)> {
        self.hash.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Iterate over array entries (1-based index, value).
    pub fn iter_array(&self) -> impl Iterator<Item = (usize, &LuaValue)> {
        self.array.iter().enumerate().map(|(i, v)| (i + 1, v))
    }

    /// Get all keys.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = (1..=self.array.len())
            .map(|i| i.to_string())
            .collect();
        keys.extend(self.hash.keys().cloned());
        keys
    }
}

impl Default for LuaTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Lua function
// ---------------------------------------------------------------------------

/// A reference to a Lua function.
#[derive(Debug, Clone)]
pub struct LuaFunction {
    /// Name (for debugging).
    pub name: String,
    /// Whether this is a Rust callback.
    pub is_rust: bool,
    /// Registry index (in a real implementation).
    pub registry_index: i32,
}

/// Type alias for Rust callbacks registered with Lua.
pub type RustCallback = Arc<dyn Fn(&[LuaValue]) -> LuaResult<Vec<LuaValue>> + Send + Sync>;

// ---------------------------------------------------------------------------
// User data
// ---------------------------------------------------------------------------

/// A Lua userdata object wrapping an engine type.
#[derive(Debug, Clone)]
pub struct LuaUserData {
    /// Type name (metatable name).
    pub type_name: String,
    /// Opaque identifier for the underlying object.
    pub id: u64,
    /// Whether the GC should free this object.
    pub gc_owned: bool,
}

impl LuaUserData {
    /// Create a new userdata.
    pub fn new(type_name: impl Into<String>, id: u64) -> Self {
        Self {
            type_name: type_name.into(),
            id,
            gc_owned: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Sandbox configuration
// ---------------------------------------------------------------------------

/// Configuration for the Lua sandbox.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Whether to allow `os` library functions.
    pub allow_os: bool,
    /// Whether to allow `io` library functions.
    pub allow_io: bool,
    /// Whether to allow `debug` library functions.
    pub allow_debug: bool,
    /// Whether to allow `require` / `dofile` / `loadfile`.
    pub allow_load: bool,
    /// Whether to allow raw string.rep (can be used for memory DoS).
    pub allow_string_rep: bool,
    /// Maximum memory usage in bytes (0 = unlimited).
    pub max_memory: usize,
    /// Maximum execution steps (0 = unlimited).
    pub max_instructions: u64,
    /// Allowed module names for require().
    pub allowed_modules: Vec<String>,
    /// Blocked global names.
    pub blocked_globals: Vec<String>,
    /// Whether to allow creating coroutines.
    pub allow_coroutines: bool,
    /// Maximum string length allowed.
    pub max_string_length: usize,
    /// Maximum table nesting depth.
    pub max_table_depth: u32,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            allow_os: false,
            allow_io: false,
            allow_debug: false,
            allow_load: false,
            allow_string_rep: false,
            max_memory: 64 * 1024 * 1024, // 64 MB
            max_instructions: 10_000_000,
            allowed_modules: Vec::new(),
            blocked_globals: vec![
                "os".into(),
                "io".into(),
                "debug".into(),
                "dofile".into(),
                "loadfile".into(),
                "load".into(),
                "rawset".into(),
                "rawget".into(),
            ],
            allow_coroutines: true,
            max_string_length: 1024 * 1024, // 1 MB
            max_table_depth: 100,
        }
    }
}

impl SandboxConfig {
    /// Create a fully permissive configuration (no restrictions).
    pub fn permissive() -> Self {
        Self {
            allow_os: true,
            allow_io: true,
            allow_debug: true,
            allow_load: true,
            allow_string_rep: true,
            max_memory: 0,
            max_instructions: 0,
            allowed_modules: Vec::new(),
            blocked_globals: Vec::new(),
            allow_coroutines: true,
            max_string_length: 0,
            max_table_depth: 0,
        }
    }

    /// Create a minimal sandbox (very restrictive).
    pub fn minimal() -> Self {
        Self {
            max_memory: 4 * 1024 * 1024, // 4 MB
            max_instructions: 100_000,
            ..Default::default()
        }
    }

    /// Check if a global name is allowed.
    pub fn is_global_allowed(&self, name: &str) -> bool {
        !self.blocked_globals.iter().any(|b| b == name)
    }

    /// Check if a module is allowed.
    pub fn is_module_allowed(&self, name: &str) -> bool {
        if !self.allow_load {
            return false;
        }
        if self.allowed_modules.is_empty() {
            return true; // All modules allowed if no whitelist
        }
        self.allowed_modules.iter().any(|m| m == name)
    }
}

// ---------------------------------------------------------------------------
// Coroutine
// ---------------------------------------------------------------------------

/// State of a Lua coroutine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoroutineState {
    /// Created but not yet started.
    Created,
    /// Running (currently executing).
    Running,
    /// Suspended (yielded).
    Suspended,
    /// Finished (returned).
    Dead,
    /// An error occurred.
    Error,
}

impl fmt::Display for CoroutineState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Created => write!(f, "created"),
            Self::Running => write!(f, "running"),
            Self::Suspended => write!(f, "suspended"),
            Self::Dead => write!(f, "dead"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// A Lua coroutine handle.
#[derive(Debug, Clone)]
pub struct LuaCoroutine {
    /// Unique identifier.
    pub id: u32,
    /// Current state.
    pub state: CoroutineState,
    /// Name (for debugging).
    pub name: String,
    /// Values yielded by the coroutine.
    pub yielded_values: Vec<LuaValue>,
    /// Return values (when finished).
    pub return_values: Vec<LuaValue>,
    /// Error message (if state == Error).
    pub error: Option<String>,
}

impl LuaCoroutine {
    /// Create a new coroutine.
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            state: CoroutineState::Created,
            name: name.into(),
            yielded_values: Vec::new(),
            return_values: Vec::new(),
            error: None,
        }
    }

    /// Whether the coroutine can be resumed.
    pub fn is_resumable(&self) -> bool {
        matches!(self.state, CoroutineState::Created | CoroutineState::Suspended)
    }

    /// Whether the coroutine has finished.
    pub fn is_dead(&self) -> bool {
        matches!(self.state, CoroutineState::Dead | CoroutineState::Error)
    }
}

// ---------------------------------------------------------------------------
// Metatable registry
// ---------------------------------------------------------------------------

/// Describes a metatable for an engine type exposed to Lua.
#[derive(Debug, Clone)]
pub struct MetatableDesc {
    /// Type name (used as the metatable name in the Lua registry).
    pub type_name: String,
    /// Methods available on this type.
    pub methods: HashMap<String, MethodDesc>,
    /// Metamethods (__index, __newindex, __tostring, etc.).
    pub metamethods: HashMap<String, String>,
    /// Whether instances are garbage collected by Lua.
    pub gc_managed: bool,
}

/// Describes a method on a Lua-exposed type.
#[derive(Debug, Clone)]
pub struct MethodDesc {
    /// Method name.
    pub name: String,
    /// Number of expected arguments (not counting self).
    pub arg_count: usize,
    /// Return count.
    pub return_count: usize,
    /// Brief description (for documentation).
    pub description: String,
}

impl MetatableDesc {
    /// Create a new metatable descriptor.
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            methods: HashMap::new(),
            metamethods: HashMap::new(),
            gc_managed: true,
        }
    }

    /// Add a method.
    pub fn add_method(
        &mut self,
        name: impl Into<String>,
        arg_count: usize,
        return_count: usize,
        description: impl Into<String>,
    ) {
        let name = name.into();
        self.methods.insert(
            name.clone(),
            MethodDesc {
                name,
                arg_count,
                return_count,
                description: description.into(),
            },
        );
    }

    /// Add a metamethod.
    pub fn add_metamethod(&mut self, name: impl Into<String>, description: impl Into<String>) {
        self.metamethods.insert(name.into(), description.into());
    }
}

// ---------------------------------------------------------------------------
// Lua state (software implementation)
// ---------------------------------------------------------------------------

/// A software Lua state that simulates Lua operations for testing and
/// development. In production, this would wrap a real Lua C state.
pub struct LuaState {
    /// Global variables.
    globals: HashMap<String, LuaValue>,
    /// Stack (simulated).
    stack: Vec<LuaValue>,
    /// Registered Rust callbacks.
    callbacks: HashMap<String, RustCallback>,
    /// Registered metatables.
    metatables: HashMap<String, MetatableDesc>,
    /// Active coroutines.
    coroutines: HashMap<u32, LuaCoroutine>,
    /// Next coroutine ID.
    next_coroutine_id: u32,
    /// Sandbox configuration.
    pub sandbox: SandboxConfig,
    /// Memory usage tracking (bytes).
    memory_used: usize,
    /// Instruction counter.
    instructions_executed: u64,
    /// Whether the state is open.
    is_open: bool,
    /// Error handler callback name.
    error_handler: Option<String>,
    /// Output capture (for print() etc).
    pub output: Vec<String>,
}

impl LuaState {
    /// Create a new Lua state with standard libraries.
    pub fn new() -> Self {
        let mut state = Self {
            globals: HashMap::new(),
            stack: Vec::new(),
            callbacks: HashMap::new(),
            metatables: HashMap::new(),
            coroutines: HashMap::new(),
            next_coroutine_id: 1,
            sandbox: SandboxConfig::default(),
            memory_used: 0,
            instructions_executed: 0,
            is_open: true,
            error_handler: None,
            output: Vec::new(),
        };
        state.register_standard_globals();
        state
    }

    /// Create a sandboxed Lua state.
    pub fn new_sandboxed(config: SandboxConfig) -> Self {
        let mut state = Self::new();
        state.sandbox = config;
        state.apply_sandbox();
        state
    }

    /// Register standard global values.
    fn register_standard_globals(&mut self) {
        self.globals
            .insert("_VERSION".into(), LuaValue::String("Lua 5.4 (genovo)".into()));
        self.globals
            .insert("math".into(), LuaValue::Table(self.create_math_table()));
        self.globals
            .insert("string".into(), LuaValue::Table(self.create_string_table()));
        self.globals
            .insert("table".into(), LuaValue::Table(self.create_table_table()));
    }

    /// Create the math standard library table.
    fn create_math_table(&self) -> LuaTable {
        let mut t = LuaTable::new();
        t.set("pi", LuaValue::Number(std::f64::consts::PI));
        t.set("huge", LuaValue::Number(f64::INFINITY));
        t.set("maxinteger", LuaValue::Integer(i64::MAX));
        t.set("mininteger", LuaValue::Integer(i64::MIN));
        t
    }

    /// Create the string standard library table.
    fn create_string_table(&self) -> LuaTable {
        LuaTable::new()
    }

    /// Create the table standard library table.
    fn create_table_table(&self) -> LuaTable {
        LuaTable::new()
    }

    /// Apply sandbox restrictions.
    fn apply_sandbox(&mut self) {
        for name in &self.sandbox.blocked_globals.clone() {
            self.globals.remove(name);
        }
    }

    // -- State management --

    /// Check if the state is open.
    pub fn is_open(&self) -> bool {
        self.is_open
    }

    /// Close the state.
    pub fn close(&mut self) {
        self.is_open = false;
        self.globals.clear();
        self.stack.clear();
        self.callbacks.clear();
        self.coroutines.clear();
    }

    /// Reset the state to a clean configuration.
    pub fn reset(&mut self) {
        self.globals.clear();
        self.stack.clear();
        self.coroutines.clear();
        self.memory_used = 0;
        self.instructions_executed = 0;
        self.output.clear();
        self.register_standard_globals();
        self.apply_sandbox();
    }

    // -- Global variables --

    /// Set a global variable.
    pub fn set_global(&mut self, name: impl Into<String>, value: LuaValue) -> LuaResult<()> {
        self.check_open()?;
        let name = name.into();
        if !self.sandbox.is_global_allowed(&name) {
            return Err(LuaError::SandboxViolation(format!(
                "cannot set blocked global '{name}'"
            )));
        }
        self.globals.insert(name, value);
        Ok(())
    }

    /// Get a global variable.
    pub fn get_global(&self, name: &str) -> LuaResult<&LuaValue> {
        self.check_open()?;
        self.globals.get(name).ok_or(LuaError::KeyNotFound(name.into()))
    }

    /// Check if a global exists.
    pub fn has_global(&self, name: &str) -> bool {
        self.globals.contains_key(name)
    }

    // -- Stack operations --

    /// Push a value onto the stack.
    pub fn push(&mut self, value: LuaValue) -> LuaResult<()> {
        self.check_open()?;
        if self.stack.len() >= 1_000_000 {
            return Err(LuaError::StackOverflow);
        }
        self.stack.push(value);
        Ok(())
    }

    /// Pop a value from the stack.
    pub fn pop(&mut self) -> LuaResult<LuaValue> {
        self.check_open()?;
        self.stack.pop().ok_or(LuaError::RuntimeError("stack underflow".into()))
    }

    /// Get the current stack size.
    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }

    /// Peek at the top of the stack.
    pub fn top(&self) -> Option<&LuaValue> {
        self.stack.last()
    }

    /// Clear the stack.
    pub fn clear_stack(&mut self) {
        self.stack.clear();
    }

    // -- Function calls --

    /// Register a Rust function as a Lua global.
    pub fn register_function(
        &mut self,
        name: impl Into<String>,
        callback: RustCallback,
    ) -> LuaResult<()> {
        self.check_open()?;
        let name = name.into();
        let func = LuaFunction {
            name: name.clone(),
            is_rust: true,
            registry_index: self.callbacks.len() as i32,
        };
        self.callbacks.insert(name.clone(), callback);
        self.globals.insert(name, LuaValue::Function(func));
        Ok(())
    }

    /// Call a Lua function by name with arguments.
    pub fn call_function(
        &mut self,
        name: &str,
        args: &[LuaValue],
    ) -> LuaResult<Vec<LuaValue>> {
        self.check_open()?;
        // Check if it's a registered Rust callback
        if let Some(callback) = self.callbacks.get(name).cloned() {
            self.instructions_executed += 1;
            self.check_instruction_limit()?;
            return callback(args);
        }
        // Check if it's a Lua function in globals
        match self.globals.get(name) {
            Some(LuaValue::Function(_)) => {
                // In a real implementation, this would call pcall.
                // For our stub, we just return an empty result.
                self.instructions_executed += 1;
                Ok(Vec::new())
            }
            Some(other) => Err(LuaError::NotCallable(format!(
                "{name} is a {}, not a function",
                other.type_name()
            ))),
            None => Err(LuaError::KeyNotFound(name.into())),
        }
    }

    /// Execute a Lua source string.
    pub fn exec(&mut self, source: &str) -> LuaResult<Vec<LuaValue>> {
        self.check_open()?;
        self.instructions_executed += source.len() as u64 / 10; // rough estimate
        self.check_instruction_limit()?;

        // Very simplified: just look for simple assignments and print calls
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("--") {
                continue;
            }
            // Handle simple variable assignments: name = value
            if let Some(eq_pos) = trimmed.find('=') {
                if !trimmed.starts_with("if")
                    && !trimmed.starts_with("for")
                    && !trimmed.starts_with("while")
                    && !trimmed.contains("==")
                    && !trimmed.contains("~=")
                    && !trimmed.contains("<=")
                    && !trimmed.contains(">=")
                {
                    let var_name = trimmed[..eq_pos].trim();
                    let value_str = trimmed[eq_pos + 1..].trim();
                    if !var_name.contains(' ') && !var_name.contains('.') {
                        let value = self.parse_simple_value(value_str);
                        self.globals.insert(var_name.to_string(), value);
                    }
                }
            }
        }
        Ok(Vec::new())
    }

    /// Parse a simple literal value from Lua source.
    fn parse_simple_value(&self, s: &str) -> LuaValue {
        let s = s.trim().trim_end_matches(';');
        if s == "true" {
            LuaValue::Boolean(true)
        } else if s == "false" {
            LuaValue::Boolean(false)
        } else if s == "nil" {
            LuaValue::Nil
        } else if let Ok(i) = s.parse::<i64>() {
            LuaValue::Integer(i)
        } else if let Ok(f) = s.parse::<f64>() {
            LuaValue::Number(f)
        } else if (s.starts_with('"') && s.ends_with('"'))
            || (s.starts_with('\'') && s.ends_with('\''))
        {
            LuaValue::String(s[1..s.len() - 1].to_string())
        } else {
            // Treat as a reference to another global
            self.globals
                .get(s)
                .cloned()
                .unwrap_or(LuaValue::Nil)
        }
    }

    // -- Table operations --

    /// Create a new empty table and push it onto the stack.
    pub fn new_table(&mut self) -> LuaResult<LuaTable> {
        self.check_open()?;
        Ok(LuaTable::new())
    }

    /// Get a field from a table value.
    pub fn table_get(table: &LuaTable, key: &str) -> LuaValue {
        table.get(key).cloned().unwrap_or(LuaValue::Nil)
    }

    /// Set a field on a table value.
    pub fn table_set(table: &mut LuaTable, key: impl Into<String>, value: LuaValue) {
        table.set(key, value);
    }

    // -- Metatable operations --

    /// Register a metatable for an engine type.
    pub fn register_metatable(&mut self, desc: MetatableDesc) -> LuaResult<()> {
        self.check_open()?;
        self.metatables.insert(desc.type_name.clone(), desc);
        Ok(())
    }

    /// Get a registered metatable descriptor.
    pub fn get_metatable(&self, type_name: &str) -> Option<&MetatableDesc> {
        self.metatables.get(type_name)
    }

    /// Create a userdata with a metatable.
    pub fn create_userdata(&self, type_name: &str, id: u64) -> LuaResult<LuaUserData> {
        if !self.metatables.contains_key(type_name) {
            return Err(LuaError::KeyNotFound(format!(
                "metatable '{type_name}' not registered"
            )));
        }
        Ok(LuaUserData::new(type_name, id))
    }

    // -- Coroutine operations --

    /// Create a new coroutine.
    pub fn create_coroutine(&mut self, name: impl Into<String>) -> LuaResult<u32> {
        self.check_open()?;
        if !self.sandbox.allow_coroutines {
            return Err(LuaError::SandboxViolation(
                "coroutines are disabled".into(),
            ));
        }
        let id = self.next_coroutine_id;
        self.next_coroutine_id += 1;
        let co = LuaCoroutine::new(id, name);
        self.coroutines.insert(id, co);
        Ok(id)
    }

    /// Resume a coroutine.
    pub fn resume_coroutine(
        &mut self,
        id: u32,
        args: &[LuaValue],
    ) -> LuaResult<Vec<LuaValue>> {
        self.check_open()?;
        let co = self
            .coroutines
            .get_mut(&id)
            .ok_or(LuaError::CoroutineError(format!("coroutine {id} not found")))?;
        if !co.is_resumable() {
            return Err(LuaError::CoroutineError(format!(
                "coroutine {id} is not resumable (state: {})",
                co.state
            )));
        }
        co.state = CoroutineState::Running;
        // In a real implementation, this would resume the Lua coroutine.
        // For our stub, mark it as dead immediately.
        co.state = CoroutineState::Dead;
        co.return_values = args.to_vec();
        Ok(co.return_values.clone())
    }

    /// Get coroutine state.
    pub fn coroutine_state(&self, id: u32) -> Option<CoroutineState> {
        self.coroutines.get(&id).map(|co| co.state)
    }

    // -- Memory / instruction tracking --

    /// Get current memory usage.
    pub fn memory_used(&self) -> usize {
        self.memory_used
    }

    /// Get total instructions executed.
    pub fn instructions_executed(&self) -> u64 {
        self.instructions_executed
    }

    // -- Internal helpers --

    fn check_open(&self) -> LuaResult<()> {
        if self.is_open {
            Ok(())
        } else {
            Err(LuaError::StateClosed)
        }
    }

    fn check_instruction_limit(&self) -> LuaResult<()> {
        if self.sandbox.max_instructions > 0
            && self.instructions_executed > self.sandbox.max_instructions
        {
            Err(LuaError::SandboxViolation(
                "instruction limit exceeded".into(),
            ))
        } else {
            Ok(())
        }
    }
}

impl Default for LuaState {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LuaState {
    fn drop(&mut self) {
        if self.is_open {
            self.close();
        }
    }
}

// ---------------------------------------------------------------------------
// Engine type bindings (helpers for common engine types)
// ---------------------------------------------------------------------------

/// Create a Vec3 metatable descriptor.
pub fn vec3_metatable() -> MetatableDesc {
    let mut desc = MetatableDesc::new("Vec3");
    desc.add_method("length", 0, 1, "Returns the length of the vector");
    desc.add_method("normalize", 0, 1, "Returns a normalized copy");
    desc.add_method("dot", 1, 1, "Dot product with another Vec3");
    desc.add_method("cross", 1, 1, "Cross product with another Vec3");
    desc.add_method("lerp", 2, 1, "Linear interpolation toward another Vec3");
    desc.add_method("distance", 1, 1, "Distance to another Vec3");
    desc.add_metamethod("__add", "Vector addition");
    desc.add_metamethod("__sub", "Vector subtraction");
    desc.add_metamethod("__mul", "Scalar multiplication");
    desc.add_metamethod("__unm", "Negation");
    desc.add_metamethod("__tostring", "String representation");
    desc.add_metamethod("__eq", "Equality comparison");
    desc.add_metamethod("__index", "Field access (x, y, z)");
    desc.add_metamethod("__newindex", "Field assignment");
    desc
}

/// Create a Transform metatable descriptor.
pub fn transform_metatable() -> MetatableDesc {
    let mut desc = MetatableDesc::new("Transform");
    desc.add_method("get_position", 0, 1, "Get the position as Vec3");
    desc.add_method("set_position", 1, 0, "Set the position from Vec3");
    desc.add_method("get_rotation", 0, 1, "Get the rotation as Quat");
    desc.add_method("set_rotation", 1, 0, "Set the rotation from Quat");
    desc.add_method("get_scale", 0, 1, "Get the scale as Vec3");
    desc.add_method("set_scale", 1, 0, "Set the scale from Vec3");
    desc.add_method("translate", 1, 0, "Translate by a Vec3");
    desc.add_method("rotate", 1, 0, "Rotate by a Quat");
    desc.add_method("look_at", 1, 0, "Rotate to look at a position");
    desc.add_metamethod("__tostring", "String representation");
    desc
}

/// Create an Entity metatable descriptor.
pub fn entity_metatable() -> MetatableDesc {
    let mut desc = MetatableDesc::new("Entity");
    desc.add_method("id", 0, 1, "Get the entity ID");
    desc.add_method("name", 0, 1, "Get the entity name");
    desc.add_method("set_name", 1, 0, "Set the entity name");
    desc.add_method("is_alive", 0, 1, "Check if the entity is alive");
    desc.add_method("destroy", 0, 0, "Destroy the entity");
    desc.add_method("get_component", 1, 1, "Get a component by name");
    desc.add_method("add_component", 1, 1, "Add a component by name");
    desc.add_method("remove_component", 1, 0, "Remove a component by name");
    desc.add_method("has_component", 1, 1, "Check for a component");
    desc.add_metamethod("__tostring", "String representation");
    desc.add_metamethod("__eq", "Equality comparison");
    desc
}

/// Register all standard engine metatables.
pub fn register_engine_metatables(state: &mut LuaState) -> LuaResult<()> {
    state.register_metatable(vec3_metatable())?;
    state.register_metatable(transform_metatable())?;
    state.register_metatable(entity_metatable())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lua_state_creation() {
        let state = LuaState::new();
        assert!(state.is_open());
        assert!(state.has_global("_VERSION"));
    }

    #[test]
    fn test_lua_state_close() {
        let mut state = LuaState::new();
        state.close();
        assert!(!state.is_open());
        assert!(state.set_global("x", LuaValue::Nil).is_err());
    }

    #[test]
    fn test_set_get_global() {
        let mut state = LuaState::new();
        state
            .set_global("health", LuaValue::Integer(100))
            .unwrap();
        let val = state.get_global("health").unwrap();
        assert_eq!(val.as_integer(), Some(100));
    }

    #[test]
    fn test_stack_operations() {
        let mut state = LuaState::new();
        state.push(LuaValue::Integer(42)).unwrap();
        state.push(LuaValue::String("hello".into())).unwrap();
        assert_eq!(state.stack_size(), 2);
        let val = state.pop().unwrap();
        assert_eq!(val.as_str(), Some("hello"));
        assert_eq!(state.stack_size(), 1);
    }

    #[test]
    fn test_rust_callback() {
        let mut state = LuaState::new();
        let callback: RustCallback = Arc::new(|args| {
            let sum: f64 = args
                .iter()
                .filter_map(|v| v.as_number())
                .sum();
            Ok(vec![LuaValue::Number(sum)])
        });
        state.register_function("add", callback).unwrap();
        let result = state
            .call_function("add", &[LuaValue::Number(3.0), LuaValue::Number(4.0)])
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_number(), Some(7.0));
    }

    #[test]
    fn test_lua_table() {
        let mut table = LuaTable::new();
        table.set("name", LuaValue::String("hero".into()));
        table.push(LuaValue::Integer(1));
        table.push(LuaValue::Integer(2));
        assert_eq!(table.len(), 2);
        assert_eq!(table.hash_len(), 1);
        assert_eq!(
            table.get("name").unwrap().as_str(),
            Some("hero")
        );
        assert_eq!(
            table.get_index(1).unwrap().as_integer(),
            Some(1)
        );
    }

    #[test]
    fn test_lua_table_set_index() {
        let mut table = LuaTable::new();
        table.set_index(3, LuaValue::String("third".into()));
        assert_eq!(table.len(), 3); // Nil-padded
        assert!(table.get_index(1).unwrap().is_nil());
        assert_eq!(table.get_index(3).unwrap().as_str(), Some("third"));
    }

    #[test]
    fn test_lua_value_types() {
        assert_eq!(LuaValue::Nil.type_name(), "nil");
        assert_eq!(LuaValue::Boolean(true).type_name(), "boolean");
        assert_eq!(LuaValue::Integer(42).type_name(), "number");
        assert_eq!(LuaValue::String("hi".into()).type_name(), "string");
    }

    #[test]
    fn test_lua_value_truthiness() {
        assert!(!LuaValue::Nil.is_truthy());
        assert!(!LuaValue::Boolean(false).is_truthy());
        assert!(LuaValue::Boolean(true).is_truthy());
        assert!(LuaValue::Integer(0).is_truthy()); // In Lua, 0 is truthy
        assert!(LuaValue::String("".into()).is_truthy());
    }

    #[test]
    fn test_lua_value_conversions() {
        let v: LuaValue = 42i32.into();
        assert_eq!(v.as_integer(), Some(42));
        let v: LuaValue = 3.14f64.into();
        assert_eq!(v.as_number(), Some(3.14));
        let v: LuaValue = "hello".into();
        assert_eq!(v.as_str(), Some("hello"));
    }

    #[test]
    fn test_exec_simple() {
        let mut state = LuaState::new();
        state.exec("x = 42").unwrap();
        assert_eq!(state.get_global("x").unwrap().as_integer(), Some(42));
    }

    #[test]
    fn test_exec_string() {
        let mut state = LuaState::new();
        state.exec("name = \"hero\"").unwrap();
        assert_eq!(state.get_global("name").unwrap().as_str(), Some("hero"));
    }

    #[test]
    fn test_exec_bool() {
        let mut state = LuaState::new();
        state.exec("flag = true").unwrap();
        assert_eq!(
            state.get_global("flag").unwrap().as_bool(),
            Some(true)
        );
    }

    #[test]
    fn test_sandbox_blocks_globals() {
        let mut state = LuaState::new_sandboxed(SandboxConfig::default());
        assert!(!state.has_global("os"));
        assert!(!state.has_global("io"));
    }

    #[test]
    fn test_sandbox_blocks_set() {
        let mut state = LuaState::new_sandboxed(SandboxConfig::default());
        let result = state.set_global("os", LuaValue::Table(LuaTable::new()));
        assert!(result.is_err());
    }

    #[test]
    fn test_sandbox_permissive() {
        let config = SandboxConfig::permissive();
        assert!(config.allow_os);
        assert!(config.allow_io);
        assert!(config.is_global_allowed("os"));
    }

    #[test]
    fn test_sandbox_module_check() {
        let mut config = SandboxConfig::default();
        config.allow_load = true;
        config.allowed_modules = vec!["json".into(), "math".into()];
        assert!(config.is_module_allowed("json"));
        assert!(!config.is_module_allowed("socket"));
    }

    #[test]
    fn test_coroutine_lifecycle() {
        let mut state = LuaState::new();
        let id = state.create_coroutine("test_co").unwrap();
        assert_eq!(state.coroutine_state(id), Some(CoroutineState::Created));
        let result = state.resume_coroutine(id, &[]).unwrap();
        assert_eq!(state.coroutine_state(id), Some(CoroutineState::Dead));
    }

    #[test]
    fn test_coroutine_not_resumable_when_dead() {
        let mut state = LuaState::new();
        let id = state.create_coroutine("co").unwrap();
        state.resume_coroutine(id, &[]).unwrap();
        let result = state.resume_coroutine(id, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_metatable_registration() {
        let mut state = LuaState::new();
        register_engine_metatables(&mut state).unwrap();
        assert!(state.get_metatable("Vec3").is_some());
        assert!(state.get_metatable("Transform").is_some());
        assert!(state.get_metatable("Entity").is_some());
    }

    #[test]
    fn test_userdata_creation() {
        let mut state = LuaState::new();
        state.register_metatable(vec3_metatable()).unwrap();
        let ud = state.create_userdata("Vec3", 42).unwrap();
        assert_eq!(ud.type_name, "Vec3");
        assert_eq!(ud.id, 42);
    }

    #[test]
    fn test_userdata_invalid_type() {
        let state = LuaState::new();
        let result = state.create_userdata("NonExistent", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_metatable_methods() {
        let mt = vec3_metatable();
        assert!(mt.methods.contains_key("length"));
        assert!(mt.methods.contains_key("dot"));
        assert!(mt.methods.contains_key("cross"));
        assert!(mt.metamethods.contains_key("__add"));
    }

    #[test]
    fn test_lua_value_display() {
        assert_eq!(LuaValue::Nil.to_display_string(), "nil");
        assert_eq!(LuaValue::Boolean(true).to_display_string(), "true");
        assert_eq!(LuaValue::Integer(42).to_display_string(), "42");
        assert_eq!(LuaValue::String("hi".into()).to_display_string(), "\"hi\"");
    }

    #[test]
    fn test_lua_table_keys() {
        let mut table = LuaTable::new();
        table.push(LuaValue::Integer(10));
        table.set("name", LuaValue::String("test".into()));
        let keys = table.keys();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_lua_table_pop() {
        let mut table = LuaTable::new();
        table.push(LuaValue::Integer(1));
        table.push(LuaValue::Integer(2));
        let val = table.pop().unwrap();
        assert_eq!(val.as_integer(), Some(2));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_lua_state_reset() {
        let mut state = LuaState::new();
        state.set_global("x", LuaValue::Integer(42)).unwrap();
        state.reset();
        assert!(state.get_global("x").is_err());
        assert!(state.has_global("_VERSION"));
    }

    #[test]
    fn test_not_callable_error() {
        let mut state = LuaState::new();
        state
            .set_global("x", LuaValue::Integer(42))
            .unwrap();
        let result = state.call_function("x", &[]);
        assert!(matches!(result, Err(LuaError::NotCallable(_))));
    }

    #[test]
    fn test_error_display() {
        let err = LuaError::TypeError {
            expected: "number".into(),
            got: "string".into(),
        };
        assert!(err.to_string().contains("number"));
        assert!(err.to_string().contains("string"));
    }
}
