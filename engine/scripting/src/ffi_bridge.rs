//! Script-to-native FFI bridge for the Genovo scripting module.
//!
//! Provides a mechanism to register Rust functions that can be called from
//! scripts, with automatic type marshalling between [`ScriptValue`] and Rust
//! types. Also supports callbacks from native code back into script functions.
//!
//! # Features
//!
//! - Register Rust functions callable from scripts
//! - Type marshalling: ScriptValue <-> Rust types (bool, i64, f64, String, Vec3)
//! - Error propagation across the native boundary
//! - Async native calls with completion callbacks
//! - Native-to-script callbacks (call script functions from Rust)
//! - Module-based function namespacing
//! - Argument validation and coercion
//! - Native object wrapping with opaque handles
//!
//! # Example
//!
//! ```ignore
//! let mut bridge = FfiBridge::new();
//!
//! bridge.register("math", "add", |args| {
//!     let a = args[0].as_float().unwrap_or(0.0);
//!     let b = args[1].as_float().unwrap_or(0.0);
//!     Ok(FfiValue::Float(a + b))
//! });
//!
//! let result = bridge.call("math", "add", &[
//!     FfiValue::Float(3.0),
//!     FfiValue::Float(4.0),
//! ]).unwrap();
//! assert_eq!(result.as_float(), Some(7.0));
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// FFI Value (marshalling type)
// ---------------------------------------------------------------------------

/// A value that can be passed across the script-native boundary.
///
/// This mirrors [`ScriptValue`] but is designed for ergonomic use from Rust
/// code, with helper methods for conversion.
#[derive(Debug, Clone)]
pub enum FfiValue {
    /// Nil/null value.
    Nil,
    /// Boolean.
    Bool(bool),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit floating point.
    Float(f64),
    /// String.
    String(String),
    /// 3-component vector.
    Vec3(f32, f32, f32),
    /// Entity handle (opaque u64).
    Entity(u64),
    /// Array of values.
    Array(Vec<FfiValue>),
    /// Map of string keys to values.
    Map(Vec<(String, FfiValue)>),
    /// An opaque native handle.
    NativeHandle(NativeHandle),
}

impl FfiValue {
    // --- Type checking ---

    /// Returns `true` if this value is nil.
    pub fn is_nil(&self) -> bool {
        matches!(self, FfiValue::Nil)
    }

    /// Returns `true` if this value is truthy (non-nil and non-false).
    pub fn is_truthy(&self) -> bool {
        !matches!(self, FfiValue::Nil | FfiValue::Bool(false))
    }

    // --- Extraction ---

    /// Try to extract a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            FfiValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to extract an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            FfiValue::Int(i) => Some(*i),
            FfiValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Try to extract a float (integers are promoted).
    pub fn as_float(&self) -> Option<f64> {
        match self {
            FfiValue::Float(f) => Some(*f),
            FfiValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to extract a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            FfiValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract a Vec3.
    pub fn as_vec3(&self) -> Option<(f32, f32, f32)> {
        match self {
            FfiValue::Vec3(x, y, z) => Some((*x, *y, *z)),
            _ => None,
        }
    }

    /// Try to extract an entity handle.
    pub fn as_entity(&self) -> Option<u64> {
        match self {
            FfiValue::Entity(e) => Some(*e),
            _ => None,
        }
    }

    /// Try to extract an array.
    pub fn as_array(&self) -> Option<&[FfiValue]> {
        match self {
            FfiValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Try to extract a map.
    pub fn as_map(&self) -> Option<&[(String, FfiValue)]> {
        match self {
            FfiValue::Map(entries) => Some(entries),
            _ => None,
        }
    }

    /// Try to extract a native handle.
    pub fn as_native_handle(&self) -> Option<&NativeHandle> {
        match self {
            FfiValue::NativeHandle(h) => Some(h),
            _ => None,
        }
    }

    /// Returns a type name for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            FfiValue::Nil => "nil",
            FfiValue::Bool(_) => "bool",
            FfiValue::Int(_) => "int",
            FfiValue::Float(_) => "float",
            FfiValue::String(_) => "string",
            FfiValue::Vec3(_, _, _) => "vec3",
            FfiValue::Entity(_) => "entity",
            FfiValue::Array(_) => "array",
            FfiValue::Map(_) => "map",
            FfiValue::NativeHandle(_) => "native_handle",
        }
    }

    /// Coerce to float, returning an error if not possible.
    pub fn coerce_float(&self) -> Result<f64, FfiError> {
        self.as_float().ok_or_else(|| {
            FfiError::TypeMismatch {
                expected: "float",
                got: self.type_name(),
            }
        })
    }

    /// Coerce to int, returning an error if not possible.
    pub fn coerce_int(&self) -> Result<i64, FfiError> {
        self.as_int().ok_or_else(|| {
            FfiError::TypeMismatch {
                expected: "int",
                got: self.type_name(),
            }
        })
    }

    /// Coerce to string, returning an error if not possible.
    pub fn coerce_string(&self) -> Result<String, FfiError> {
        match self {
            FfiValue::String(s) => Ok(s.clone()),
            FfiValue::Int(i) => Ok(i.to_string()),
            FfiValue::Float(f) => Ok(f.to_string()),
            FfiValue::Bool(b) => Ok(b.to_string()),
            FfiValue::Nil => Ok("nil".to_string()),
            other => Err(FfiError::TypeMismatch {
                expected: "string",
                got: other.type_name(),
            }),
        }
    }
}

impl fmt::Display for FfiValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FfiValue::Nil => write!(f, "nil"),
            FfiValue::Bool(b) => write!(f, "{b}"),
            FfiValue::Int(i) => write!(f, "{i}"),
            FfiValue::Float(v) => write!(f, "{v}"),
            FfiValue::String(s) => write!(f, "\"{s}\""),
            FfiValue::Vec3(x, y, z) => write!(f, "vec3({x}, {y}, {z})"),
            FfiValue::Entity(e) => write!(f, "entity({e})"),
            FfiValue::Array(arr) => write!(f, "[{} elements]", arr.len()),
            FfiValue::Map(entries) => write!(f, "{{{} entries}}", entries.len()),
            FfiValue::NativeHandle(h) => write!(f, "native_handle({}:{})", h.type_id, h.id),
        }
    }
}

impl PartialEq for FfiValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FfiValue::Nil, FfiValue::Nil) => true,
            (FfiValue::Bool(a), FfiValue::Bool(b)) => a == b,
            (FfiValue::Int(a), FfiValue::Int(b)) => a == b,
            (FfiValue::Float(a), FfiValue::Float(b)) => a == b,
            (FfiValue::String(a), FfiValue::String(b)) => a == b,
            (FfiValue::Vec3(ax, ay, az), FfiValue::Vec3(bx, by, bz)) => {
                ax == bx && ay == by && az == bz
            }
            (FfiValue::Entity(a), FfiValue::Entity(b)) => a == b,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Native Handle
// ---------------------------------------------------------------------------

/// An opaque handle to a native (Rust) object accessible from scripts.
#[derive(Debug, Clone)]
pub struct NativeHandle {
    /// Unique identifier for this handle.
    pub id: u64,
    /// Type identifier string.
    pub type_id: String,
}

impl NativeHandle {
    /// Create a new native handle.
    pub fn new(id: u64, type_id: &str) -> Self {
        Self {
            id,
            type_id: type_id.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// FFI Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during FFI operations.
#[derive(Debug, Clone)]
pub enum FfiError {
    /// A type mismatch occurred during marshalling.
    TypeMismatch {
        /// The expected type.
        expected: &'static str,
        /// The actual type received.
        got: &'static str,
    },
    /// The wrong number of arguments was provided.
    ArgumentCount {
        /// The function name.
        function: String,
        /// Expected number of arguments.
        expected: usize,
        /// Actual number of arguments.
        got: usize,
    },
    /// The function was not found.
    FunctionNotFound {
        /// The module name.
        module: String,
        /// The function name.
        function: String,
    },
    /// The module was not found.
    ModuleNotFound {
        /// The module name.
        module: String,
    },
    /// A native function returned an error.
    NativeError(String),
    /// A script callback failed.
    CallbackError(String),
    /// An async call timed out.
    Timeout(String),
    /// The bridge has been shut down.
    BridgeShutdown,
    /// A native handle was invalid or expired.
    InvalidHandle(u64),
    /// An argument validation error.
    ValidationError(String),
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {expected}, got {got}")
            }
            Self::ArgumentCount {
                function,
                expected,
                got,
            } => {
                write!(
                    f,
                    "wrong argument count for '{function}': expected {expected}, got {got}"
                )
            }
            Self::FunctionNotFound { module, function } => {
                write!(f, "function not found: {module}::{function}")
            }
            Self::ModuleNotFound { module } => {
                write!(f, "module not found: {module}")
            }
            Self::NativeError(msg) => write!(f, "native error: {msg}"),
            Self::CallbackError(msg) => write!(f, "callback error: {msg}"),
            Self::Timeout(msg) => write!(f, "async call timed out: {msg}"),
            Self::BridgeShutdown => write!(f, "FFI bridge has been shut down"),
            Self::InvalidHandle(id) => write!(f, "invalid native handle: {id}"),
            Self::ValidationError(msg) => write!(f, "validation error: {msg}"),
        }
    }
}

/// Result type for FFI operations.
pub type FfiResult<T> = Result<T, FfiError>;

// ---------------------------------------------------------------------------
// Function signatures and descriptors
// ---------------------------------------------------------------------------

/// Describes an argument of a native function.
#[derive(Debug, Clone)]
pub struct ArgDescriptor {
    /// Argument name.
    pub name: String,
    /// Expected type name.
    pub type_name: String,
    /// Whether this argument is optional.
    pub optional: bool,
    /// Default value (if optional).
    pub default: Option<FfiValue>,
}

impl ArgDescriptor {
    /// Create a required argument descriptor.
    pub fn required(name: &str, type_name: &str) -> Self {
        Self {
            name: name.to_string(),
            type_name: type_name.to_string(),
            optional: false,
            default: None,
        }
    }

    /// Create an optional argument descriptor with a default value.
    pub fn optional(name: &str, type_name: &str, default: FfiValue) -> Self {
        Self {
            name: name.to_string(),
            type_name: type_name.to_string(),
            optional: true,
            default: Some(default),
        }
    }
}

/// Describes a native function registered with the bridge.
#[derive(Debug, Clone)]
pub struct FunctionDescriptor {
    /// Module name.
    pub module: String,
    /// Function name.
    pub name: String,
    /// Documentation string.
    pub doc: Option<String>,
    /// Argument descriptors.
    pub args: Vec<ArgDescriptor>,
    /// Return type name.
    pub return_type: String,
    /// Whether the function is async.
    pub is_async: bool,
}

impl FunctionDescriptor {
    /// Create a new function descriptor.
    pub fn new(module: &str, name: &str) -> Self {
        Self {
            module: module.to_string(),
            name: name.to_string(),
            doc: None,
            args: Vec::new(),
            return_type: "any".to_string(),
            is_async: false,
        }
    }

    /// Set the documentation string.
    pub fn with_doc(mut self, doc: &str) -> Self {
        self.doc = Some(doc.to_string());
        self
    }

    /// Add an argument descriptor.
    pub fn with_arg(mut self, arg: ArgDescriptor) -> Self {
        self.args.push(arg);
        self
    }

    /// Set the return type.
    pub fn with_return_type(mut self, type_name: &str) -> Self {
        self.return_type = type_name.to_string();
        self
    }

    /// Mark the function as async.
    pub fn as_async(mut self) -> Self {
        self.is_async = true;
        self
    }

    /// Fully qualified name (module::function).
    pub fn qualified_name(&self) -> String {
        format!("{}::{}", self.module, self.name)
    }

    /// Returns the required argument count.
    pub fn required_arg_count(&self) -> usize {
        self.args.iter().filter(|a| !a.optional).count()
    }
}

impl fmt::Display for FunctionDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let args: Vec<String> = self
            .args
            .iter()
            .map(|a| {
                if a.optional {
                    format!("{}?: {}", a.name, a.type_name)
                } else {
                    format!("{}: {}", a.name, a.type_name)
                }
            })
            .collect();
        write!(
            f,
            "fn {}::{}({}) -> {}",
            self.module,
            self.name,
            args.join(", "),
            self.return_type
        )
    }
}

// ---------------------------------------------------------------------------
// Native function trait
// ---------------------------------------------------------------------------

/// Type alias for a synchronous native function.
pub type NativeFn = Arc<dyn Fn(&[FfiValue]) -> FfiResult<FfiValue> + Send + Sync>;

/// A registered native function with its descriptor.
struct RegisteredFunction {
    descriptor: FunctionDescriptor,
    implementation: NativeFn,
}

// ---------------------------------------------------------------------------
// Async call handle
// ---------------------------------------------------------------------------

/// Handle for tracking an async native call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncCallId(pub u64);

impl fmt::Display for AsyncCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AsyncCall({})", self.0)
    }
}

/// The status of an async call.
#[derive(Debug, Clone)]
pub enum AsyncCallStatus {
    /// The call is still pending.
    Pending,
    /// The call completed successfully.
    Completed(FfiValue),
    /// The call failed with an error.
    Failed(FfiError),
    /// The call was cancelled.
    Cancelled,
}

impl AsyncCallStatus {
    /// Returns `true` if the call is still pending.
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }

    /// Returns `true` if the call is complete (success or failure).
    pub fn is_done(&self) -> bool {
        !matches!(self, Self::Pending)
    }
}

// ---------------------------------------------------------------------------
// Script callback
// ---------------------------------------------------------------------------

/// A handle to a script function that can be called from native code.
#[derive(Debug, Clone)]
pub struct ScriptCallback {
    /// The name of the script function.
    pub function_name: String,
    /// The module the function belongs to.
    pub module_name: Option<String>,
    /// Opaque identifier for the function in the VM.
    pub function_id: u32,
}

impl ScriptCallback {
    /// Create a new script callback.
    pub fn new(function_name: &str, function_id: u32) -> Self {
        Self {
            function_name: function_name.to_string(),
            module_name: None,
            function_id,
        }
    }

    /// Create a callback in a specific module.
    pub fn in_module(module: &str, function_name: &str, function_id: u32) -> Self {
        Self {
            function_name: function_name.to_string(),
            module_name: Some(module.to_string()),
            function_id,
        }
    }
}

impl fmt::Display for ScriptCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref module) = self.module_name {
            write!(f, "{}::{}", module, self.function_name)
        } else {
            write!(f, "{}", self.function_name)
        }
    }
}

/// A queued callback invocation from native to script.
#[derive(Debug, Clone)]
pub struct PendingCallback {
    /// The callback to invoke.
    pub callback: ScriptCallback,
    /// Arguments to pass to the script function.
    pub args: Vec<FfiValue>,
    /// Priority for ordering (higher = sooner).
    pub priority: u32,
}

// ---------------------------------------------------------------------------
// FFI Module
// ---------------------------------------------------------------------------

/// A namespace for grouping related native functions.
struct FfiModule {
    /// Module name.
    name: String,
    /// Registered functions in this module.
    functions: HashMap<String, RegisteredFunction>,
    /// Documentation string.
    doc: Option<String>,
}

impl FfiModule {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            functions: HashMap::new(),
            doc: None,
        }
    }
}

// ---------------------------------------------------------------------------
// FFI Bridge
// ---------------------------------------------------------------------------

/// The main FFI bridge for script-to-native interop.
pub struct FfiBridge {
    /// Registered modules.
    modules: HashMap<String, FfiModule>,
    /// Async call tracking.
    async_calls: HashMap<AsyncCallId, AsyncCallStatus>,
    /// Next async call ID.
    next_async_id: u64,
    /// Pending callbacks from native to script.
    pending_callbacks: Vec<PendingCallback>,
    /// Native object registry.
    native_objects: HashMap<u64, NativeObjectEntry>,
    /// Next native object ID.
    next_native_id: u64,
    /// Whether the bridge is active.
    active: bool,
    /// Statistics.
    stats: FfiBridgeStats,
}

/// An entry in the native object registry.
struct NativeObjectEntry {
    type_id: String,
    data: Vec<u8>,
    ref_count: u32,
}

/// Statistics for the FFI bridge.
#[derive(Debug, Clone, Default)]
pub struct FfiBridgeStats {
    /// Total number of native function calls.
    pub total_calls: u64,
    /// Total number of successful calls.
    pub successful_calls: u64,
    /// Total number of failed calls.
    pub failed_calls: u64,
    /// Total number of async calls initiated.
    pub async_calls_initiated: u64,
    /// Total number of async calls completed.
    pub async_calls_completed: u64,
    /// Total number of callbacks dispatched to script.
    pub callbacks_dispatched: u64,
    /// Total number of native objects created.
    pub native_objects_created: u64,
    /// Total number of native objects destroyed.
    pub native_objects_destroyed: u64,
}

impl fmt::Display for FfiBridgeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FFI[calls={}/{} ok, async={}/{}, callbacks={}, objects={}/{}]",
            self.total_calls,
            self.successful_calls,
            self.async_calls_initiated,
            self.async_calls_completed,
            self.callbacks_dispatched,
            self.native_objects_created,
            self.native_objects_destroyed,
        )
    }
}

impl FfiBridge {
    /// Create a new FFI bridge.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            async_calls: HashMap::new(),
            next_async_id: 1,
            pending_callbacks: Vec::new(),
            native_objects: HashMap::new(),
            next_native_id: 1,
            active: true,
            stats: FfiBridgeStats::default(),
        }
    }

    /// Register a module (creates it if it doesn't exist).
    pub fn register_module(&mut self, name: &str, doc: Option<&str>) {
        let module = self
            .modules
            .entry(name.to_string())
            .or_insert_with(|| FfiModule::new(name));
        if let Some(d) = doc {
            module.doc = Some(d.to_string());
        }
    }

    /// Register a native function in a module.
    pub fn register<F>(
        &mut self,
        module: &str,
        name: &str,
        f: F,
    ) where
        F: Fn(&[FfiValue]) -> FfiResult<FfiValue> + Send + Sync + 'static,
    {
        let descriptor = FunctionDescriptor::new(module, name);
        self.register_with_descriptor(descriptor, f);
    }

    /// Register a native function with a full descriptor.
    pub fn register_with_descriptor<F>(
        &mut self,
        descriptor: FunctionDescriptor,
        f: F,
    ) where
        F: Fn(&[FfiValue]) -> FfiResult<FfiValue> + Send + Sync + 'static,
    {
        let module_name = descriptor.module.clone();
        let func_name = descriptor.name.clone();

        let module = self
            .modules
            .entry(module_name)
            .or_insert_with(|| FfiModule::new(&descriptor.module));

        module.functions.insert(
            func_name,
            RegisteredFunction {
                descriptor,
                implementation: Arc::new(f),
            },
        );
    }

    /// Call a registered native function.
    pub fn call(
        &mut self,
        module: &str,
        function: &str,
        args: &[FfiValue],
    ) -> FfiResult<FfiValue> {
        if !self.active {
            return Err(FfiError::BridgeShutdown);
        }

        self.stats.total_calls += 1;

        let module_obj = self.modules.get(module).ok_or_else(|| {
            FfiError::ModuleNotFound {
                module: module.to_string(),
            }
        })?;

        let registered = module_obj.functions.get(function).ok_or_else(|| {
            FfiError::FunctionNotFound {
                module: module.to_string(),
                function: function.to_string(),
            }
        })?;

        // Validate argument count.
        let required = registered.descriptor.required_arg_count();
        let max = registered.descriptor.args.len();
        if !registered.descriptor.args.is_empty() {
            if args.len() < required {
                self.stats.failed_calls += 1;
                return Err(FfiError::ArgumentCount {
                    function: registered.descriptor.qualified_name(),
                    expected: required,
                    got: args.len(),
                });
            }
            if args.len() > max && max > 0 {
                self.stats.failed_calls += 1;
                return Err(FfiError::ArgumentCount {
                    function: registered.descriptor.qualified_name(),
                    expected: max,
                    got: args.len(),
                });
            }
        }

        // Fill in defaults for missing optional args.
        let mut effective_args: Vec<FfiValue>;
        if args.len() < registered.descriptor.args.len() {
            effective_args = args.to_vec();
            for i in args.len()..registered.descriptor.args.len() {
                if let Some(ref default) = registered.descriptor.args[i].default {
                    effective_args.push(default.clone());
                }
            }
        } else {
            effective_args = args.to_vec();
        }

        // Clone the implementation Arc to avoid borrow conflict.
        let implementation = registered.implementation.clone();

        match implementation(&effective_args) {
            Ok(result) => {
                self.stats.successful_calls += 1;
                Ok(result)
            }
            Err(e) => {
                self.stats.failed_calls += 1;
                Err(e)
            }
        }
    }

    /// Initiate an async native call. Returns an ID to poll for completion.
    pub fn call_async(
        &mut self,
        module: &str,
        function: &str,
        args: &[FfiValue],
    ) -> FfiResult<AsyncCallId> {
        if !self.active {
            return Err(FfiError::BridgeShutdown);
        }

        // Verify the function exists.
        let _ = self
            .modules
            .get(module)
            .ok_or_else(|| FfiError::ModuleNotFound {
                module: module.to_string(),
            })?
            .functions
            .get(function)
            .ok_or_else(|| FfiError::FunctionNotFound {
                module: module.to_string(),
                function: function.to_string(),
            })?;

        let id = AsyncCallId(self.next_async_id);
        self.next_async_id += 1;
        self.async_calls.insert(id, AsyncCallStatus::Pending);
        self.stats.async_calls_initiated += 1;

        // In a real implementation, this would spawn the call on a thread pool.
        // For now, we execute synchronously and store the result.
        let result = self.call(module, function, args);
        match result {
            Ok(value) => {
                self.async_calls
                    .insert(id, AsyncCallStatus::Completed(value));
                self.stats.async_calls_completed += 1;
            }
            Err(e) => {
                self.async_calls.insert(id, AsyncCallStatus::Failed(e));
                self.stats.async_calls_completed += 1;
            }
        }

        Ok(id)
    }

    /// Poll an async call for completion.
    pub fn poll_async(&self, id: AsyncCallId) -> Option<&AsyncCallStatus> {
        self.async_calls.get(&id)
    }

    /// Take the result of a completed async call (removes it from tracking).
    pub fn take_async_result(&mut self, id: AsyncCallId) -> Option<AsyncCallStatus> {
        self.async_calls.remove(&id)
    }

    /// Cancel an async call.
    pub fn cancel_async(&mut self, id: AsyncCallId) {
        self.async_calls.insert(id, AsyncCallStatus::Cancelled);
    }

    // --- Callbacks (native -> script) ---

    /// Queue a callback to be invoked in the script VM.
    pub fn queue_callback(
        &mut self,
        callback: ScriptCallback,
        args: Vec<FfiValue>,
        priority: u32,
    ) {
        self.pending_callbacks.push(PendingCallback {
            callback,
            args,
            priority,
        });
    }

    /// Drain all pending callbacks, sorted by priority (highest first).
    pub fn drain_callbacks(&mut self) -> Vec<PendingCallback> {
        let mut callbacks = std::mem::take(&mut self.pending_callbacks);
        callbacks.sort_by(|a, b| b.priority.cmp(&a.priority));
        self.stats.callbacks_dispatched += callbacks.len() as u64;
        callbacks
    }

    /// Returns the number of pending callbacks.
    pub fn pending_callback_count(&self) -> usize {
        self.pending_callbacks.len()
    }

    // --- Native object management ---

    /// Register a native object and get a handle for script access.
    pub fn register_native_object(&mut self, type_id: &str, data: Vec<u8>) -> NativeHandle {
        let id = self.next_native_id;
        self.next_native_id += 1;

        self.native_objects.insert(
            id,
            NativeObjectEntry {
                type_id: type_id.to_string(),
                data,
                ref_count: 1,
            },
        );

        self.stats.native_objects_created += 1;
        NativeHandle::new(id, type_id)
    }

    /// Get the data of a native object.
    pub fn get_native_object(&self, id: u64) -> FfiResult<&[u8]> {
        self.native_objects
            .get(&id)
            .map(|entry| entry.data.as_slice())
            .ok_or(FfiError::InvalidHandle(id))
    }

    /// Get the type ID of a native object.
    pub fn get_native_object_type(&self, id: u64) -> FfiResult<&str> {
        self.native_objects
            .get(&id)
            .map(|entry| entry.type_id.as_str())
            .ok_or(FfiError::InvalidHandle(id))
    }

    /// Increment the reference count of a native object.
    pub fn add_ref(&mut self, id: u64) -> FfiResult<()> {
        self.native_objects
            .get_mut(&id)
            .map(|entry| {
                entry.ref_count += 1;
            })
            .ok_or(FfiError::InvalidHandle(id))
    }

    /// Decrement the reference count. Destroys the object if it reaches zero.
    pub fn release(&mut self, id: u64) -> FfiResult<()> {
        let should_remove = {
            let entry = self
                .native_objects
                .get_mut(&id)
                .ok_or(FfiError::InvalidHandle(id))?;
            entry.ref_count = entry.ref_count.saturating_sub(1);
            entry.ref_count == 0
        };
        if should_remove {
            self.native_objects.remove(&id);
            self.stats.native_objects_destroyed += 1;
        }
        Ok(())
    }

    // --- Introspection ---

    /// List all registered modules.
    pub fn modules(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }

    /// List all functions in a module.
    pub fn functions_in_module(&self, module: &str) -> Option<Vec<&FunctionDescriptor>> {
        self.modules.get(module).map(|m| {
            m.functions.values().map(|f| &f.descriptor).collect()
        })
    }

    /// Get the descriptor for a specific function.
    pub fn function_descriptor(
        &self,
        module: &str,
        function: &str,
    ) -> Option<&FunctionDescriptor> {
        self.modules
            .get(module)
            .and_then(|m| m.functions.get(function))
            .map(|f| &f.descriptor)
    }

    /// Returns the total number of registered functions.
    pub fn function_count(&self) -> usize {
        self.modules
            .values()
            .map(|m| m.functions.len())
            .sum()
    }

    /// Returns the bridge statistics.
    pub fn stats(&self) -> &FfiBridgeStats {
        &self.stats
    }

    /// Shut down the bridge (no more calls accepted).
    pub fn shutdown(&mut self) {
        self.active = false;
        self.pending_callbacks.clear();
        self.async_calls.clear();
    }

    /// Returns `true` if the bridge is active.
    pub fn is_active(&self) -> bool {
        self.active
    }
}

impl Default for FfiBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for FfiBridge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FfiBridge[modules={}, functions={}, active={}]",
            self.modules.len(),
            self.function_count(),
            self.active
        )
    }
}

// ---------------------------------------------------------------------------
// Argument validation helpers
// ---------------------------------------------------------------------------

/// Validate that exactly N arguments are provided.
pub fn expect_args(args: &[FfiValue], count: usize, func: &str) -> FfiResult<()> {
    if args.len() != count {
        Err(FfiError::ArgumentCount {
            function: func.to_string(),
            expected: count,
            got: args.len(),
        })
    } else {
        Ok(())
    }
}

/// Validate that at least N arguments are provided.
pub fn expect_min_args(args: &[FfiValue], min: usize, func: &str) -> FfiResult<()> {
    if args.len() < min {
        Err(FfiError::ArgumentCount {
            function: func.to_string(),
            expected: min,
            got: args.len(),
        })
    } else {
        Ok(())
    }
}

/// Validate that between min and max arguments are provided.
pub fn expect_args_range(
    args: &[FfiValue],
    min: usize,
    max: usize,
    func: &str,
) -> FfiResult<()> {
    if args.len() < min || args.len() > max {
        Err(FfiError::ArgumentCount {
            function: func.to_string(),
            expected: min,
            got: args.len(),
        })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_value_types() {
        assert!(FfiValue::Nil.is_nil());
        assert!(!FfiValue::Bool(true).is_nil());
        assert!(FfiValue::Bool(true).is_truthy());
        assert!(!FfiValue::Bool(false).is_truthy());
        assert!(!FfiValue::Nil.is_truthy());
    }

    #[test]
    fn test_ffi_value_extraction() {
        assert_eq!(FfiValue::Bool(true).as_bool(), Some(true));
        assert_eq!(FfiValue::Int(42).as_int(), Some(42));
        assert_eq!(FfiValue::Float(3.14).as_float(), Some(3.14));
        assert_eq!(FfiValue::Int(5).as_float(), Some(5.0));
        assert_eq!(FfiValue::String("hello".into()).as_str(), Some("hello"));
        assert_eq!(FfiValue::Vec3(1.0, 2.0, 3.0).as_vec3(), Some((1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_ffi_value_coercion() {
        assert_eq!(FfiValue::Float(3.14).coerce_float().unwrap(), 3.14);
        assert_eq!(FfiValue::Int(5).coerce_float().unwrap(), 5.0);
        assert!(FfiValue::String("hello".into()).coerce_float().is_err());

        assert_eq!(FfiValue::Int(42).coerce_string().unwrap(), "42");
        assert_eq!(FfiValue::Bool(true).coerce_string().unwrap(), "true");
    }

    #[test]
    fn test_register_and_call() {
        let mut bridge = FfiBridge::new();

        bridge.register("math", "add", |args| {
            let a = args[0].coerce_float()?;
            let b = args[1].coerce_float()?;
            Ok(FfiValue::Float(a + b))
        });

        let result = bridge
            .call("math", "add", &[FfiValue::Float(3.0), FfiValue::Float(4.0)])
            .unwrap();
        assert_eq!(result.as_float(), Some(7.0));
    }

    #[test]
    fn test_call_nonexistent_module() {
        let mut bridge = FfiBridge::new();
        let result = bridge.call("nope", "func", &[]);
        assert!(matches!(result, Err(FfiError::ModuleNotFound { .. })));
    }

    #[test]
    fn test_call_nonexistent_function() {
        let mut bridge = FfiBridge::new();
        bridge.register_module("test", None);
        let result = bridge.call("test", "nope", &[]);
        assert!(matches!(result, Err(FfiError::FunctionNotFound { .. })));
    }

    #[test]
    fn test_function_descriptor() {
        let desc = FunctionDescriptor::new("math", "lerp")
            .with_doc("Linear interpolation")
            .with_arg(ArgDescriptor::required("a", "float"))
            .with_arg(ArgDescriptor::required("b", "float"))
            .with_arg(ArgDescriptor::optional("t", "float", FfiValue::Float(0.5)))
            .with_return_type("float");

        assert_eq!(desc.qualified_name(), "math::lerp");
        assert_eq!(desc.required_arg_count(), 2);
        assert_eq!(desc.args.len(), 3);
    }

    #[test]
    fn test_native_error_propagation() {
        let mut bridge = FfiBridge::new();

        bridge.register("test", "fail", |_| {
            Err(FfiError::NativeError("something broke".to_string()))
        });

        let result = bridge.call("test", "fail", &[]);
        assert!(matches!(result, Err(FfiError::NativeError(_))));
    }

    #[test]
    fn test_async_call() {
        let mut bridge = FfiBridge::new();

        bridge.register("math", "double", |args| {
            let n = args[0].coerce_float()?;
            Ok(FfiValue::Float(n * 2.0))
        });

        let id = bridge
            .call_async("math", "double", &[FfiValue::Float(21.0)])
            .unwrap();
        let status = bridge.poll_async(id).unwrap();
        match status {
            AsyncCallStatus::Completed(val) => {
                assert_eq!(val.as_float(), Some(42.0));
            }
            _ => panic!("expected completed"),
        }
    }

    #[test]
    fn test_callbacks() {
        let mut bridge = FfiBridge::new();

        bridge.queue_callback(
            ScriptCallback::new("on_damage", 1),
            vec![FfiValue::Int(50)],
            1,
        );
        bridge.queue_callback(
            ScriptCallback::new("on_heal", 2),
            vec![FfiValue::Int(25)],
            10,
        );

        assert_eq!(bridge.pending_callback_count(), 2);

        let callbacks = bridge.drain_callbacks();
        assert_eq!(callbacks.len(), 2);
        // Higher priority first.
        assert_eq!(callbacks[0].callback.function_name, "on_heal");
        assert_eq!(callbacks[1].callback.function_name, "on_damage");
    }

    #[test]
    fn test_native_objects() {
        let mut bridge = FfiBridge::new();

        let handle = bridge.register_native_object("texture", vec![1, 2, 3, 4]);
        assert_eq!(handle.type_id, "texture");

        let data = bridge.get_native_object(handle.id).unwrap();
        assert_eq!(data, &[1, 2, 3, 4]);

        bridge.add_ref(handle.id).unwrap();
        bridge.release(handle.id).unwrap(); // ref_count = 1
        assert!(bridge.get_native_object(handle.id).is_ok());

        bridge.release(handle.id).unwrap(); // ref_count = 0, removed
        assert!(bridge.get_native_object(handle.id).is_err());
    }

    #[test]
    fn test_bridge_shutdown() {
        let mut bridge = FfiBridge::new();
        bridge.register("test", "fn1", |_| Ok(FfiValue::Nil));

        assert!(bridge.is_active());
        bridge.shutdown();
        assert!(!bridge.is_active());

        let result = bridge.call("test", "fn1", &[]);
        assert!(matches!(result, Err(FfiError::BridgeShutdown)));
    }

    #[test]
    fn test_bridge_introspection() {
        let mut bridge = FfiBridge::new();
        bridge.register("math", "sin", |_| Ok(FfiValue::Float(0.0)));
        bridge.register("math", "cos", |_| Ok(FfiValue::Float(0.0)));
        bridge.register("io", "read", |_| Ok(FfiValue::Nil));

        assert_eq!(bridge.function_count(), 3);
        let modules = bridge.modules();
        assert!(modules.contains(&"math"));
        assert!(modules.contains(&"io"));

        let math_fns = bridge.functions_in_module("math").unwrap();
        assert_eq!(math_fns.len(), 2);
    }

    #[test]
    fn test_stats_tracking() {
        let mut bridge = FfiBridge::new();
        bridge.register("test", "ok", |_| Ok(FfiValue::Nil));
        bridge.register("test", "fail", |_| {
            Err(FfiError::NativeError("err".into()))
        });

        let _ = bridge.call("test", "ok", &[]);
        let _ = bridge.call("test", "ok", &[]);
        let _ = bridge.call("test", "fail", &[]);

        assert_eq!(bridge.stats().total_calls, 3);
        assert_eq!(bridge.stats().successful_calls, 2);
        assert_eq!(bridge.stats().failed_calls, 1);
    }

    #[test]
    fn test_expect_args() {
        let args = vec![FfiValue::Int(1), FfiValue::Int(2)];
        assert!(expect_args(&args, 2, "test").is_ok());
        assert!(expect_args(&args, 3, "test").is_err());
        assert!(expect_min_args(&args, 1, "test").is_ok());
        assert!(expect_min_args(&args, 3, "test").is_err());
    }

    #[test]
    fn test_ffi_value_equality() {
        assert_eq!(FfiValue::Int(42), FfiValue::Int(42));
        assert_eq!(FfiValue::String("hello".into()), FfiValue::String("hello".into()));
        assert_ne!(FfiValue::Int(1), FfiValue::Int(2));
        assert_ne!(FfiValue::Int(1), FfiValue::Float(1.0));
    }
}
