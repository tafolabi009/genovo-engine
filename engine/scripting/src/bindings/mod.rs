//! Script binding infrastructure for the Genovo engine.
//!
//! Provides traits and registries for exposing Rust engine types and functions
//! to scripting languages. The [`ScriptBindable`] trait marks types that can
//! be reflected into script VMs, and [`BindingRegistry`] manages the set of
//! all registered bindings. Built-in native functions cover vector math,
//! entity management, and utility operations.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use genovo_core::{EngineError, EngineResult};
use genovo_ecs::Component;

use crate::vm::{
    GenovoVM, ScriptContext, ScriptError, ScriptValue,
    ScriptVM,
};

// ---------------------------------------------------------------------------
// ScriptBindable
// ---------------------------------------------------------------------------

/// Trait for Rust types that can be exposed to scripting languages.
///
/// Implementing this trait allows a type's methods and properties to be
/// accessed from Lua, WASM, or the Genovo custom VM via the binding registry.
pub trait ScriptBindable: Send + Sync + 'static {
    /// Returns the name under which this type is exposed to scripts.
    fn script_type_name() -> &'static str
    where
        Self: Sized;

    /// Registers this type's methods and properties with the given VM.
    fn register_bindings(vm: &mut dyn ScriptVM) -> EngineResult<()>
    where
        Self: Sized;

    /// Converts an instance of this type into a [`ScriptValue`].
    fn to_script_value(&self) -> ScriptValue;

    /// Attempts to reconstruct an instance from a [`ScriptValue`].
    fn from_script_value(value: &ScriptValue) -> EngineResult<Self>
    where
        Self: Sized;
}

// ---------------------------------------------------------------------------
// BindingDescriptor
// ---------------------------------------------------------------------------

/// Describes a single bindable type and how to register it.
struct BindingDescriptor {
    /// The script-visible type name.
    type_name: &'static str,
    /// Function that registers this type's bindings with a VM.
    register_fn: fn(&mut dyn ScriptVM) -> EngineResult<()>,
}

// ---------------------------------------------------------------------------
// BindingRegistry
// ---------------------------------------------------------------------------

/// Central registry for all engine API bindings exposed to scripts.
///
/// Types are registered at startup, then bulk-applied to each script VM
/// instance that is created. The registry also holds standalone native
/// functions that are not tied to a specific type.
pub struct BindingRegistry {
    /// All registered type bindings.
    descriptors: Vec<BindingDescriptor>,
    /// Standalone native functions (not attached to a type).
    standalone_functions: HashMap<String, Arc<dyn Fn(&[ScriptValue]) -> Result<ScriptValue, ScriptError> + Send + Sync>>,
}

impl BindingRegistry {
    /// Creates a new empty binding registry.
    pub fn new() -> Self {
        Self {
            descriptors: Vec::new(),
            standalone_functions: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with all built-in functions.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        reg.register_builtins();
        reg
    }

    /// Registers a [`ScriptBindable`] type so its bindings will be applied
    /// to all VMs.
    pub fn register_type<T: ScriptBindable>(&mut self) {
        self.descriptors.push(BindingDescriptor {
            type_name: T::script_type_name(),
            register_fn: T::register_bindings,
        });
        log::debug!("Registered script binding: {}", T::script_type_name());
    }

    /// Registers a standalone native function.
    pub fn register_function(
        &mut self,
        name: impl Into<String>,
        func: Arc<dyn Fn(&[ScriptValue]) -> Result<ScriptValue, ScriptError> + Send + Sync>,
    ) {
        let name = name.into();
        log::debug!("Registered standalone script function: {name}");
        self.standalone_functions.insert(name, func);
    }

    /// Register all built-in native functions.
    pub fn register_builtins(&mut self) {
        // -- Vector math --
        self.register_function("vec3", Arc::new(native_vec3));
        self.register_function("vec3_add", Arc::new(native_vec3_add));
        self.register_function("vec3_sub", Arc::new(native_vec3_sub));
        self.register_function("vec3_scale", Arc::new(native_vec3_scale));
        self.register_function("vec3_length", Arc::new(native_vec3_length));
        self.register_function("vec3_normalize", Arc::new(native_vec3_normalize));
        self.register_function("vec3_dot", Arc::new(native_vec3_dot));
        self.register_function("vec3_cross", Arc::new(native_vec3_cross));
        self.register_function("vec3_x", Arc::new(native_vec3_x));
        self.register_function("vec3_y", Arc::new(native_vec3_y));
        self.register_function("vec3_z", Arc::new(native_vec3_z));

        // -- Entity management --
        self.register_function("entity_spawn", Arc::new(native_entity_spawn));
        self.register_function("entity_destroy", Arc::new(native_entity_destroy));
        self.register_function("entity_is_valid", Arc::new(native_entity_is_valid));

        // -- Position (simulated) --
        self.register_function("get_position", Arc::new(native_get_position));
        self.register_function("set_position", Arc::new(native_set_position));

        // -- Utilities --
        self.register_function("log", Arc::new(native_log));
        self.register_function("time", Arc::new(native_time));
        self.register_function("random", Arc::new(native_random));
        self.register_function("to_int", Arc::new(native_to_int));
        self.register_function("to_float", Arc::new(native_to_float));
        self.register_function("to_string", Arc::new(native_to_string));
        self.register_function("type_of", Arc::new(native_type_of));
        self.register_function("abs", Arc::new(native_abs));
        self.register_function("min", Arc::new(native_min));
        self.register_function("max", Arc::new(native_max));
        self.register_function("clamp", Arc::new(native_clamp));
        self.register_function("sqrt", Arc::new(native_sqrt));
        self.register_function("floor", Arc::new(native_floor));
        self.register_function("ceil", Arc::new(native_ceil));
        self.register_function("sin", Arc::new(native_sin));
        self.register_function("cos", Arc::new(native_cos));
        self.register_function("len", Arc::new(native_len));
        self.register_function("push", Arc::new(native_push));

        log::debug!(
            "Registered {} built-in script functions",
            self.standalone_functions.len()
        );
    }

    /// Applies all registered bindings and standalone functions to a VM.
    pub fn apply_to_vm(&self, vm: &mut dyn ScriptVM) -> EngineResult<()> {
        // Register type bindings.
        for desc in &self.descriptors {
            (desc.register_fn)(vm).map_err(|e| {
                EngineError::Other(format!(
                    "Failed to register bindings for '{}': {e}",
                    desc.type_name
                ))
            })?;
        }

        // Register standalone functions.
        for (name, func) in &self.standalone_functions {
            let func_clone = Arc::clone(func);
            vm.register_function(
                name,
                Box::new(move |args| func_clone(args)),
            )?;
        }

        Ok(())
    }

    /// Returns the number of registered type bindings.
    pub fn type_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Returns the number of registered standalone functions.
    pub fn function_count(&self) -> usize {
        self.standalone_functions.len()
    }

    /// Returns a list of all registered function names.
    pub fn function_names(&self) -> Vec<String> {
        self.standalone_functions.keys().cloned().collect()
    }
}

impl Default for BindingRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Built-in native functions
// ===========================================================================

// -- Vector math ------------------------------------------------------------

/// `vec3(x, y, z)` — construct a Vec3.
fn native_vec3(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 3 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3".into(),
            expected: 3,
            got: args.len() as u8,
        });
    }
    let x = to_f32(&args[0], "vec3", "x")?;
    let y = to_f32(&args[1], "vec3", "y")?;
    let z = to_f32(&args[2], "vec3", "z")?;
    Ok(ScriptValue::Vec3(x, y, z))
}

/// `vec3_add(a, b)` — add two Vec3 values.
fn native_vec3_add(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 2 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3_add".into(),
            expected: 2,
            got: args.len() as u8,
        });
    }
    let (ax, ay, az) = extract_vec3(&args[0], "vec3_add", "a")?;
    let (bx, by, bz) = extract_vec3(&args[1], "vec3_add", "b")?;
    Ok(ScriptValue::Vec3(ax + bx, ay + by, az + bz))
}

/// `vec3_sub(a, b)` — subtract two Vec3 values.
fn native_vec3_sub(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 2 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3_sub".into(),
            expected: 2,
            got: args.len() as u8,
        });
    }
    let (ax, ay, az) = extract_vec3(&args[0], "vec3_sub", "a")?;
    let (bx, by, bz) = extract_vec3(&args[1], "vec3_sub", "b")?;
    Ok(ScriptValue::Vec3(ax - bx, ay - by, az - bz))
}

/// `vec3_scale(v, scalar)` — scale a Vec3 by a scalar.
fn native_vec3_scale(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 2 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3_scale".into(),
            expected: 2,
            got: args.len() as u8,
        });
    }
    let (x, y, z) = extract_vec3(&args[0], "vec3_scale", "v")?;
    let s = to_f32(&args[1], "vec3_scale", "scalar")?;
    Ok(ScriptValue::Vec3(x * s, y * s, z * s))
}

/// `vec3_length(v)` — return the length (magnitude) of a Vec3.
fn native_vec3_length(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3_length".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    let (x, y, z) = extract_vec3(&args[0], "vec3_length", "v")?;
    let len = (x * x + y * y + z * z).sqrt();
    Ok(ScriptValue::Float(len as f64))
}

/// `vec3_normalize(v)` — return a unit-length Vec3 in the same direction.
fn native_vec3_normalize(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3_normalize".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    let (x, y, z) = extract_vec3(&args[0], "vec3_normalize", "v")?;
    let len = (x * x + y * y + z * z).sqrt();
    if len == 0.0 {
        return Err(ScriptError::RuntimeError(
            "cannot normalize zero-length vector".into(),
        ));
    }
    Ok(ScriptValue::Vec3(x / len, y / len, z / len))
}

/// `vec3_dot(a, b)` — dot product.
fn native_vec3_dot(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 2 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3_dot".into(),
            expected: 2,
            got: args.len() as u8,
        });
    }
    let (ax, ay, az) = extract_vec3(&args[0], "vec3_dot", "a")?;
    let (bx, by, bz) = extract_vec3(&args[1], "vec3_dot", "b")?;
    let dot = ax * bx + ay * by + az * bz;
    Ok(ScriptValue::Float(dot as f64))
}

/// `vec3_cross(a, b)` — cross product.
fn native_vec3_cross(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 2 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3_cross".into(),
            expected: 2,
            got: args.len() as u8,
        });
    }
    let (ax, ay, az) = extract_vec3(&args[0], "vec3_cross", "a")?;
    let (bx, by, bz) = extract_vec3(&args[1], "vec3_cross", "b")?;
    Ok(ScriptValue::Vec3(
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ))
}

/// `vec3_x(v)` — extract x component.
fn native_vec3_x(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3_x".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    let (x, _, _) = extract_vec3(&args[0], "vec3_x", "v")?;
    Ok(ScriptValue::Float(x as f64))
}

/// `vec3_y(v)` — extract y component.
fn native_vec3_y(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3_y".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    let (_, y, _) = extract_vec3(&args[0], "vec3_y", "v")?;
    Ok(ScriptValue::Float(y as f64))
}

/// `vec3_z(v)` — extract z component.
fn native_vec3_z(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "vec3_z".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    let (_, _, z) = extract_vec3(&args[0], "vec3_z", "v")?;
    Ok(ScriptValue::Float(z as f64))
}

// -- Entity management ------------------------------------------------------

/// Global entity counter for script-spawned entities.
static NEXT_ENTITY_ID: AtomicU64 = AtomicU64::new(1000);

/// `entity_spawn()` — spawn a new entity, returning its id.
fn native_entity_spawn(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if !args.is_empty() {
        return Err(ScriptError::ArityMismatch {
            function: "entity_spawn".into(),
            expected: 0,
            got: args.len() as u8,
        });
    }
    let id = NEXT_ENTITY_ID.fetch_add(1, Ordering::Relaxed);
    log::debug!("script: spawned entity {id}");
    Ok(ScriptValue::Entity(id))
}

/// `entity_destroy(id)` — destroy an entity.
fn native_entity_destroy(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "entity_destroy".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    match &args[0] {
        ScriptValue::Entity(id) => {
            log::debug!("script: destroyed entity {id}");
            Ok(ScriptValue::Nil)
        }
        _ => Err(ScriptError::TypeError(format!(
            "entity_destroy expected entity, got {}",
            args[0].type_name()
        ))),
    }
}

/// `entity_is_valid(id)` — check if an entity id is valid.
fn native_entity_is_valid(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "entity_is_valid".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    match &args[0] {
        ScriptValue::Entity(id) => {
            // In a real engine, this would check the ECS world.
            Ok(ScriptValue::Bool(*id > 0))
        }
        _ => Ok(ScriptValue::Bool(false)),
    }
}

// -- Position ---------------------------------------------------------------

/// Thread-local simulated position storage.
/// In a real engine, these would access the ECS world.
use std::sync::Mutex;
static POSITIONS: Mutex<Option<HashMap<u64, (f32, f32, f32)>>> = Mutex::new(None);

fn get_positions() -> std::sync::MutexGuard<'static, Option<HashMap<u64, (f32, f32, f32)>>> {
    let mut guard = POSITIONS.lock().unwrap();
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    guard
}

/// `get_position(entity)` — get the position of an entity as a Vec3.
fn native_get_position(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "get_position".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    match &args[0] {
        ScriptValue::Entity(id) => {
            let positions = get_positions();
            let pos = positions
                .as_ref()
                .unwrap()
                .get(id)
                .copied()
                .unwrap_or((0.0, 0.0, 0.0));
            Ok(ScriptValue::Vec3(pos.0, pos.1, pos.2))
        }
        _ => Err(ScriptError::TypeError(format!(
            "get_position expected entity, got {}",
            args[0].type_name()
        ))),
    }
}

/// `set_position(entity, x, y, z)` — set the position of an entity.
fn native_set_position(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 4 {
        return Err(ScriptError::ArityMismatch {
            function: "set_position".into(),
            expected: 4,
            got: args.len() as u8,
        });
    }
    match &args[0] {
        ScriptValue::Entity(id) => {
            let x = to_f32(&args[1], "set_position", "x")?;
            let y = to_f32(&args[2], "set_position", "y")?;
            let z = to_f32(&args[3], "set_position", "z")?;
            let mut positions = get_positions();
            positions.as_mut().unwrap().insert(*id, (x, y, z));
            Ok(ScriptValue::Nil)
        }
        _ => Err(ScriptError::TypeError(format!(
            "set_position expected entity as first arg, got {}",
            args[0].type_name()
        ))),
    }
}

// -- Utilities --------------------------------------------------------------

/// `log(msg)` — log a message.
fn native_log(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.is_empty() {
        return Err(ScriptError::ArityMismatch {
            function: "log".into(),
            expected: 1,
            got: 0,
        });
    }
    let msg: Vec<String> = args.iter().map(|v| format!("{v}")).collect();
    log::info!("[script:log] {}", msg.join(" "));
    Ok(ScriptValue::Nil)
}

/// `time()` — return current time in seconds since epoch.
fn native_time(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if !args.is_empty() {
        return Err(ScriptError::ArityMismatch {
            function: "time".into(),
            expected: 0,
            got: args.len() as u8,
        });
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Ok(ScriptValue::Float(now.as_secs_f64()))
}

/// `random()` — return a pseudo-random float in [0, 1).
fn native_random(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if !args.is_empty() {
        return Err(ScriptError::ArityMismatch {
            function: "random".into(),
            expected: 0,
            got: args.len() as u8,
        });
    }
    // Simple xorshift-based PRNG seeded from time.
    static SEED: AtomicU64 = AtomicU64::new(0);
    let mut s = SEED.load(Ordering::Relaxed);
    if s == 0 {
        s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        if s == 0 {
            s = 1;
        }
    }
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    SEED.store(s, Ordering::Relaxed);
    let val = (s as f64) / (u64::MAX as f64);
    Ok(ScriptValue::Float(val))
}

/// `to_int(v)` — convert a value to integer.
fn native_to_int(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "to_int".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    match &args[0] {
        ScriptValue::Int(i) => Ok(ScriptValue::Int(*i)),
        ScriptValue::Float(f) => Ok(ScriptValue::Int(*f as i64)),
        ScriptValue::Bool(b) => Ok(ScriptValue::Int(if *b { 1 } else { 0 })),
        ScriptValue::String(s) => s
            .parse::<i64>()
            .map(ScriptValue::Int)
            .map_err(|_| {
                ScriptError::TypeError(format!("cannot convert '{s}' to int"))
            }),
        _ => Err(ScriptError::TypeError(format!(
            "cannot convert {} to int",
            args[0].type_name()
        ))),
    }
}

/// `to_float(v)` — convert a value to float.
fn native_to_float(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "to_float".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    match &args[0] {
        ScriptValue::Int(i) => Ok(ScriptValue::Float(*i as f64)),
        ScriptValue::Float(f) => Ok(ScriptValue::Float(*f)),
        ScriptValue::String(s) => s
            .parse::<f64>()
            .map(ScriptValue::Float)
            .map_err(|_| {
                ScriptError::TypeError(format!("cannot convert '{s}' to float"))
            }),
        _ => Err(ScriptError::TypeError(format!(
            "cannot convert {} to float",
            args[0].type_name()
        ))),
    }
}

/// `to_string(v)` — convert a value to string.
fn native_to_string(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "to_string".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    Ok(ScriptValue::from_string(format!("{}", args[0])))
}

/// `type_of(v)` — return the type name of a value as a string.
fn native_type_of(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "type_of".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    Ok(ScriptValue::from_string(args[0].type_name()))
}

/// `abs(x)` — absolute value.
fn native_abs(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "abs".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    match &args[0] {
        ScriptValue::Int(i) => Ok(ScriptValue::Int(i.abs())),
        ScriptValue::Float(f) => Ok(ScriptValue::Float(f.abs())),
        _ => Err(ScriptError::TypeError(format!(
            "abs expected number, got {}",
            args[0].type_name()
        ))),
    }
}

/// `min(a, b)` — minimum of two values.
fn native_min(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 2 {
        return Err(ScriptError::ArityMismatch {
            function: "min".into(),
            expected: 2,
            got: args.len() as u8,
        });
    }
    match (&args[0], &args[1]) {
        (ScriptValue::Int(a), ScriptValue::Int(b)) => Ok(ScriptValue::Int(*a.min(b))),
        (ScriptValue::Float(a), ScriptValue::Float(b)) => Ok(ScriptValue::Float(a.min(*b))),
        (ScriptValue::Int(a), ScriptValue::Float(b)) => {
            Ok(ScriptValue::Float((*a as f64).min(*b)))
        }
        (ScriptValue::Float(a), ScriptValue::Int(b)) => {
            Ok(ScriptValue::Float(a.min(*b as f64)))
        }
        _ => Err(ScriptError::TypeError(format!(
            "min expected numbers, got {} and {}",
            args[0].type_name(),
            args[1].type_name()
        ))),
    }
}

/// `max(a, b)` — maximum of two values.
fn native_max(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 2 {
        return Err(ScriptError::ArityMismatch {
            function: "max".into(),
            expected: 2,
            got: args.len() as u8,
        });
    }
    match (&args[0], &args[1]) {
        (ScriptValue::Int(a), ScriptValue::Int(b)) => Ok(ScriptValue::Int(*a.max(b))),
        (ScriptValue::Float(a), ScriptValue::Float(b)) => Ok(ScriptValue::Float(a.max(*b))),
        (ScriptValue::Int(a), ScriptValue::Float(b)) => {
            Ok(ScriptValue::Float((*a as f64).max(*b)))
        }
        (ScriptValue::Float(a), ScriptValue::Int(b)) => {
            Ok(ScriptValue::Float(a.max(*b as f64)))
        }
        _ => Err(ScriptError::TypeError(format!(
            "max expected numbers, got {} and {}",
            args[0].type_name(),
            args[1].type_name()
        ))),
    }
}

/// `clamp(x, min, max)` — clamp a value between min and max.
fn native_clamp(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 3 {
        return Err(ScriptError::ArityMismatch {
            function: "clamp".into(),
            expected: 3,
            got: args.len() as u8,
        });
    }
    let x = args[0]
        .as_float()
        .ok_or_else(|| ScriptError::TypeError("clamp: x must be a number".into()))?;
    let lo = args[1]
        .as_float()
        .ok_or_else(|| ScriptError::TypeError("clamp: min must be a number".into()))?;
    let hi = args[2]
        .as_float()
        .ok_or_else(|| ScriptError::TypeError("clamp: max must be a number".into()))?;
    Ok(ScriptValue::Float(x.max(lo).min(hi)))
}

/// `sqrt(x)` — square root.
fn native_sqrt(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "sqrt".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    let x = args[0]
        .as_float()
        .ok_or_else(|| ScriptError::TypeError("sqrt: expected number".into()))?;
    if x < 0.0 {
        return Err(ScriptError::RuntimeError(
            "sqrt of negative number".into(),
        ));
    }
    Ok(ScriptValue::Float(x.sqrt()))
}

/// `floor(x)` — floor of a float.
fn native_floor(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "floor".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    let x = args[0]
        .as_float()
        .ok_or_else(|| ScriptError::TypeError("floor: expected number".into()))?;
    Ok(ScriptValue::Float(x.floor()))
}

/// `ceil(x)` — ceiling of a float.
fn native_ceil(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "ceil".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    let x = args[0]
        .as_float()
        .ok_or_else(|| ScriptError::TypeError("ceil: expected number".into()))?;
    Ok(ScriptValue::Float(x.ceil()))
}

/// `sin(x)` — sine (radians).
fn native_sin(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "sin".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    let x = args[0]
        .as_float()
        .ok_or_else(|| ScriptError::TypeError("sin: expected number".into()))?;
    Ok(ScriptValue::Float(x.sin()))
}

/// `cos(x)` — cosine (radians).
fn native_cos(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "cos".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    let x = args[0]
        .as_float()
        .ok_or_else(|| ScriptError::TypeError("cos: expected number".into()))?;
    Ok(ScriptValue::Float(x.cos()))
}

/// `len(v)` — length of an array or string.
fn native_len(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 1 {
        return Err(ScriptError::ArityMismatch {
            function: "len".into(),
            expected: 1,
            got: args.len() as u8,
        });
    }
    match &args[0] {
        ScriptValue::Array(a) => Ok(ScriptValue::Int(a.len() as i64)),
        ScriptValue::String(s) => Ok(ScriptValue::Int(s.len() as i64)),
        ScriptValue::Map(m) => Ok(ScriptValue::Int(m.len() as i64)),
        _ => Err(ScriptError::TypeError(format!(
            "len: expected array, string, or map; got {}",
            args[0].type_name()
        ))),
    }
}

/// `push(array, value)` — push a value onto an array, returning the new array.
fn native_push(args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
    if args.len() != 2 {
        return Err(ScriptError::ArityMismatch {
            function: "push".into(),
            expected: 2,
            got: args.len() as u8,
        });
    }
    match &args[0] {
        ScriptValue::Array(a) => {
            let mut new_arr = a.clone();
            new_arr.push(args[1].clone());
            Ok(ScriptValue::Array(new_arr))
        }
        _ => Err(ScriptError::TypeError(format!(
            "push: expected array as first argument, got {}",
            args[0].type_name()
        ))),
    }
}

// -- Helpers ----------------------------------------------------------------

/// Extract a Vec3's components.
fn extract_vec3(val: &ScriptValue, func: &str, param: &str) -> Result<(f32, f32, f32), ScriptError> {
    match val {
        ScriptValue::Vec3(x, y, z) => Ok((*x, *y, *z)),
        _ => Err(ScriptError::TypeError(format!(
            "{func}: {param} must be a vec3, got {}",
            val.type_name()
        ))),
    }
}

/// Convert a ScriptValue to f32.
fn to_f32(val: &ScriptValue, func: &str, param: &str) -> Result<f32, ScriptError> {
    match val {
        ScriptValue::Float(f) => Ok(*f as f32),
        ScriptValue::Int(i) => Ok(*i as f32),
        _ => Err(ScriptError::TypeError(format!(
            "{func}: {param} must be a number, got {}",
            val.type_name()
        ))),
    }
}

// ===========================================================================
// ScriptComponent
// ===========================================================================

/// ECS component that attaches a compiled script to an entity.
///
/// When the [`ScriptSystem`] runs, it iterates over all entities with a
/// `ScriptComponent` and executes the associated script, passing the entity
/// handle and delta time.
#[derive(Debug, Clone)]
pub struct ScriptComponent {
    /// Name of the script to execute (must be loaded in the VM).
    pub script_name: String,
    /// Name of the VM language to use (e.g., `"Genovo"`).
    pub vm_language: String,
    /// Whether this script is currently enabled.
    pub enabled: bool,
    /// Per-instance properties that override script defaults.
    pub properties: HashMap<String, ScriptValue>,
    /// Name of the function to call each frame (default: `"update"`).
    pub update_function: String,
    /// Name of the initialization function (default: `"init"`).
    pub init_function: String,
    /// Whether `init` has been called.
    pub initialized: bool,
}

impl Component for ScriptComponent {}

impl ScriptComponent {
    /// Creates a new script component for the given script name.
    pub fn new(script_name: impl Into<String>, vm_language: impl Into<String>) -> Self {
        Self {
            script_name: script_name.into(),
            vm_language: vm_language.into(),
            enabled: true,
            properties: HashMap::new(),
            update_function: "update".to_string(),
            init_function: "init".to_string(),
            initialized: false,
        }
    }

    /// Sets the update function name.
    pub fn with_update_function(mut self, name: impl Into<String>) -> Self {
        self.update_function = name.into();
        self
    }

    /// Sets the init function name.
    pub fn with_init_function(mut self, name: impl Into<String>) -> Self {
        self.init_function = name.into();
        self
    }

    /// Sets a property override on this script instance.
    pub fn set_property(&mut self, name: impl Into<String>, value: ScriptValue) {
        self.properties.insert(name.into(), value);
    }

    /// Gets a property value, or `Nil` if not set.
    pub fn get_property(&self, name: &str) -> ScriptValue {
        self.properties
            .get(name)
            .cloned()
            .unwrap_or(ScriptValue::Nil)
    }
}

// ===========================================================================
// ScriptSystem
// ===========================================================================

/// System that executes all entity scripts each frame.
///
/// In a full engine integration, this system would query the ECS world for
/// all entities with a `ScriptComponent`, load/compile scripts as needed,
/// and call the per-entity update function. This implementation provides
/// the interface and logic, but entity iteration is stubbed pending ECS
/// integration.
pub struct ScriptSystem {
    /// The scripting VM.
    vm: GenovoVM,
    /// The binding registry (owned by the system, retained for re-application).
    #[allow(dead_code)]
    bindings: BindingRegistry,
    /// Script context for each frame.
    context: ScriptContext,
    /// Scripts that have been loaded.
    loaded_scripts: Vec<String>,
}

impl ScriptSystem {
    /// Create a new script system with the default binding registry.
    pub fn new() -> Self {
        let bindings = BindingRegistry::with_builtins();
        let mut vm = GenovoVM::new();

        // Apply all bindings to the VM.
        if let Err(e) = bindings.apply_to_vm(&mut vm) {
            log::error!("Failed to apply script bindings: {e}");
        }

        Self {
            vm,
            bindings,
            context: ScriptContext::new(),
            loaded_scripts: Vec::new(),
        }
    }

    /// Create a script system with a custom binding registry.
    pub fn with_bindings(bindings: BindingRegistry) -> Self {
        let mut vm = GenovoVM::new();
        if let Err(e) = bindings.apply_to_vm(&mut vm) {
            log::error!("Failed to apply script bindings: {e}");
        }

        Self {
            vm,
            bindings,
            context: ScriptContext::new(),
            loaded_scripts: Vec::new(),
        }
    }

    /// Load a script from source code.
    pub fn load_script(&mut self, name: &str, source: &str) -> EngineResult<()> {
        self.vm.load_script(name, source)?;
        self.loaded_scripts.push(name.to_string());
        log::debug!("ScriptSystem: loaded script '{name}'");
        Ok(())
    }

    /// Execute a named script.
    pub fn execute_script(
        &mut self,
        name: &str,
        delta_time: f64,
        total_time: f64,
    ) -> EngineResult<ScriptValue> {
        self.context.delta_time = delta_time;
        self.context.total_time = total_time;
        self.vm.execute(name, &mut self.context)
    }

    /// Call a function in a loaded script.
    pub fn call_function(
        &mut self,
        function_name: &str,
        args: &[ScriptValue],
    ) -> EngineResult<ScriptValue> {
        self.vm
            .call_function(function_name, args, &mut self.context)
    }

    /// Update all scripts for a single frame.
    ///
    /// In a full ECS integration, this would iterate over all entities
    /// with `ScriptComponent` and call their update functions.
    pub fn update(&mut self, delta_time: f64, total_time: f64) {
        self.context.delta_time = delta_time;
        self.context.total_time = total_time;

        for script_name in &self.loaded_scripts.clone() {
            match self.vm.execute(script_name, &mut self.context) {
                Ok(_) => {}
                Err(e) => {
                    log::error!("Script '{script_name}' error: {e}");
                }
            }
        }
    }

    /// Run a single script component for an entity.
    ///
    /// Calls the init function if not yet initialized, then the update
    /// function with the given delta_time.
    pub fn run_component(
        &mut self,
        component: &mut ScriptComponent,
        entity_id: u64,
        delta_time: f64,
    ) -> EngineResult<()> {
        if !component.enabled {
            return Ok(());
        }

        // Set per-entity context.
        self.vm
            .set_global("self_entity", ScriptValue::Entity(entity_id))?;
        self.vm
            .set_global("dt", ScriptValue::Float(delta_time))?;

        // Set properties as globals.
        for (name, value) in &component.properties {
            self.vm.set_global(name, value.clone())?;
        }

        // Call init if not yet done.
        if !component.initialized {
            if let Ok(_) = self.vm.call_function(
                &component.init_function,
                &[],
                &mut self.context,
            ) {
                component.initialized = true;
            }
            // If init function doesn't exist, that's fine — mark as initialized.
            component.initialized = true;
        }

        // Call update.
        if let Err(e) = self.vm.call_function(
            &component.update_function,
            &[ScriptValue::Float(delta_time)],
            &mut self.context,
        ) {
            log::warn!(
                "Script component '{}' update error: {e}",
                component.script_name
            );
        }

        Ok(())
    }

    /// Direct access to the VM.
    pub fn vm(&self) -> &GenovoVM {
        &self.vm
    }

    /// Mutable access to the VM.
    pub fn vm_mut(&mut self) -> &mut GenovoVM {
        &mut self.vm
    }

    /// Get all collected output from print statements.
    pub fn output(&self) -> &[String] {
        self.vm.output()
    }

    /// Set a global variable in the scripting context.
    pub fn set_global(
        &mut self,
        name: impl Into<String>,
        value: ScriptValue,
    ) -> EngineResult<()> {
        let name = name.into();
        self.vm.set_global(&name, value)
    }

    /// Get a global variable from the scripting context.
    pub fn get_global(&self, name: &str) -> EngineResult<ScriptValue> {
        self.vm.get_global(name)
    }
}

impl Default for ScriptSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_registry_creation() {
        let reg = BindingRegistry::new();
        assert_eq!(reg.type_count(), 0);
        assert_eq!(reg.function_count(), 0);
    }

    #[test]
    fn test_binding_registry_with_builtins() {
        let reg = BindingRegistry::with_builtins();
        assert!(reg.function_count() > 0);
        let names = reg.function_names();
        assert!(names.contains(&"vec3".to_string()));
        assert!(names.contains(&"log".to_string()));
        assert!(names.contains(&"time".to_string()));
        assert!(names.contains(&"random".to_string()));
        assert!(names.contains(&"entity_spawn".to_string()));
    }

    #[test]
    fn test_native_vec3() {
        let result = native_vec3(&[
            ScriptValue::Float(1.0),
            ScriptValue::Float(2.0),
            ScriptValue::Float(3.0),
        ])
        .unwrap();
        assert_eq!(result, ScriptValue::Vec3(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_native_vec3_add() {
        let a = ScriptValue::Vec3(1.0, 2.0, 3.0);
        let b = ScriptValue::Vec3(4.0, 5.0, 6.0);
        let result = native_vec3_add(&[a, b]).unwrap();
        assert_eq!(result, ScriptValue::Vec3(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_native_vec3_length() {
        let v = ScriptValue::Vec3(3.0, 4.0, 0.0);
        let result = native_vec3_length(&[v]).unwrap();
        assert_eq!(result, ScriptValue::Float(5.0));
    }

    #[test]
    fn test_native_vec3_normalize() {
        let v = ScriptValue::Vec3(0.0, 0.0, 5.0);
        let result = native_vec3_normalize(&[v]).unwrap();
        assert_eq!(result, ScriptValue::Vec3(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_native_vec3_dot() {
        let a = ScriptValue::Vec3(1.0, 2.0, 3.0);
        let b = ScriptValue::Vec3(4.0, 5.0, 6.0);
        let result = native_vec3_dot(&[a, b]).unwrap();
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result, ScriptValue::Float(32.0));
    }

    #[test]
    fn test_native_vec3_cross() {
        let a = ScriptValue::Vec3(1.0, 0.0, 0.0);
        let b = ScriptValue::Vec3(0.0, 1.0, 0.0);
        let result = native_vec3_cross(&[a, b]).unwrap();
        assert_eq!(result, ScriptValue::Vec3(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_native_entity_spawn() {
        let result = native_entity_spawn(&[]).unwrap();
        match result {
            ScriptValue::Entity(id) => assert!(id > 0),
            _ => panic!("expected Entity"),
        }
    }

    #[test]
    fn test_native_entity_destroy() {
        let entity = ScriptValue::Entity(42);
        let result = native_entity_destroy(&[entity]).unwrap();
        assert_eq!(result, ScriptValue::Nil);
    }

    #[test]
    fn test_native_abs() {
        assert_eq!(
            native_abs(&[ScriptValue::Int(-5)]).unwrap(),
            ScriptValue::Int(5)
        );
        assert_eq!(
            native_abs(&[ScriptValue::Float(-3.14)]).unwrap(),
            ScriptValue::Float(3.14)
        );
    }

    #[test]
    fn test_native_min_max() {
        assert_eq!(
            native_min(&[ScriptValue::Int(3), ScriptValue::Int(7)]).unwrap(),
            ScriptValue::Int(3)
        );
        assert_eq!(
            native_max(&[ScriptValue::Int(3), ScriptValue::Int(7)]).unwrap(),
            ScriptValue::Int(7)
        );
    }

    #[test]
    fn test_native_clamp() {
        let result = native_clamp(&[
            ScriptValue::Float(15.0),
            ScriptValue::Float(0.0),
            ScriptValue::Float(10.0),
        ])
        .unwrap();
        assert_eq!(result, ScriptValue::Float(10.0));
    }

    #[test]
    fn test_native_sqrt() {
        assert_eq!(
            native_sqrt(&[ScriptValue::Float(9.0)]).unwrap(),
            ScriptValue::Float(3.0)
        );
    }

    #[test]
    fn test_native_floor_ceil() {
        assert_eq!(
            native_floor(&[ScriptValue::Float(3.7)]).unwrap(),
            ScriptValue::Float(3.0)
        );
        assert_eq!(
            native_ceil(&[ScriptValue::Float(3.2)]).unwrap(),
            ScriptValue::Float(4.0)
        );
    }

    #[test]
    fn test_native_sin_cos() {
        let sin_0 = native_sin(&[ScriptValue::Float(0.0)]).unwrap();
        assert_eq!(sin_0, ScriptValue::Float(0.0));

        let cos_0 = native_cos(&[ScriptValue::Float(0.0)]).unwrap();
        assert_eq!(cos_0, ScriptValue::Float(1.0));
    }

    #[test]
    fn test_native_to_int() {
        assert_eq!(
            native_to_int(&[ScriptValue::Float(3.7)]).unwrap(),
            ScriptValue::Int(3)
        );
        assert_eq!(
            native_to_int(&[ScriptValue::Bool(true)]).unwrap(),
            ScriptValue::Int(1)
        );
    }

    #[test]
    fn test_native_to_float() {
        assert_eq!(
            native_to_float(&[ScriptValue::Int(42)]).unwrap(),
            ScriptValue::Float(42.0)
        );
    }

    #[test]
    fn test_native_to_string() {
        let result = native_to_string(&[ScriptValue::Int(42)]).unwrap();
        assert_eq!(result, ScriptValue::from_string("42"));
    }

    #[test]
    fn test_native_type_of() {
        assert_eq!(
            native_type_of(&[ScriptValue::Int(1)]).unwrap(),
            ScriptValue::from_string("int")
        );
        assert_eq!(
            native_type_of(&[ScriptValue::Nil]).unwrap(),
            ScriptValue::from_string("nil")
        );
    }

    #[test]
    fn test_native_len() {
        let arr = ScriptValue::Array(vec![
            ScriptValue::Int(1),
            ScriptValue::Int(2),
            ScriptValue::Int(3),
        ]);
        assert_eq!(native_len(&[arr]).unwrap(), ScriptValue::Int(3));

        let s = ScriptValue::from_string("hello");
        assert_eq!(native_len(&[s]).unwrap(), ScriptValue::Int(5));
    }

    #[test]
    fn test_native_push() {
        let arr = ScriptValue::Array(vec![ScriptValue::Int(1)]);
        let result = native_push(&[arr, ScriptValue::Int(2)]).unwrap();
        assert_eq!(
            result,
            ScriptValue::Array(vec![ScriptValue::Int(1), ScriptValue::Int(2)])
        );
    }

    #[test]
    fn test_native_time() {
        let result = native_time(&[]).unwrap();
        match result {
            ScriptValue::Float(t) => assert!(t > 0.0),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_native_random() {
        let result = native_random(&[]).unwrap();
        match result {
            ScriptValue::Float(r) => {
                assert!(r >= 0.0);
                assert!(r < 1.0);
            }
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_native_arity_errors() {
        assert!(native_vec3(&[]).is_err());
        assert!(native_vec3_add(&[ScriptValue::Vec3(0.0, 0.0, 0.0)]).is_err());
        assert!(native_sqrt(&[]).is_err());
        assert!(native_clamp(&[ScriptValue::Float(1.0)]).is_err());
    }

    #[test]
    fn test_script_component() {
        let mut comp = ScriptComponent::new("player", "Genovo");
        assert!(comp.enabled);
        assert_eq!(comp.script_name, "player");
        assert_eq!(comp.vm_language, "Genovo");
        assert_eq!(comp.update_function, "update");
        assert!(!comp.initialized);

        comp.set_property("speed", ScriptValue::Float(5.0));
        assert_eq!(comp.get_property("speed"), ScriptValue::Float(5.0));
        assert_eq!(comp.get_property("nonexistent"), ScriptValue::Nil);
    }

    #[test]
    fn test_script_component_builder() {
        let comp = ScriptComponent::new("enemy", "Genovo")
            .with_update_function("tick")
            .with_init_function("on_spawn");

        assert_eq!(comp.update_function, "tick");
        assert_eq!(comp.init_function, "on_spawn");
    }

    #[test]
    fn test_script_system_creation() {
        let system = ScriptSystem::new();
        assert!(system.output().is_empty());
    }

    #[test]
    fn test_script_system_load_and_execute() {
        let mut system = ScriptSystem::new();
        system
            .load_script(
                "test",
                r#"
                let x = 42
                print(x)
                "#,
            )
            .unwrap();
        system.execute_script("test", 0.016, 1.0).unwrap();
        assert_eq!(system.output(), &["42"]);
    }

    #[test]
    fn test_script_system_with_native_functions() {
        let mut system = ScriptSystem::new();

        // The built-in sqrt should be available via native function mechanism.
        // For this test, let's just verify the system can execute basic scripts.
        system
            .load_script(
                "math_test",
                r#"
                let x = 3 * 4
                let y = x + 1
                print(y)
                "#,
            )
            .unwrap();
        system.execute_script("math_test", 0.016, 1.0).unwrap();
        assert_eq!(system.output(), &["13"]);
    }

    #[test]
    fn test_script_system_globals() {
        let mut system = ScriptSystem::new();
        system
            .set_global("player_health", ScriptValue::Int(100))
            .unwrap();
        let val = system.get_global("player_health").unwrap();
        assert_eq!(val, ScriptValue::Int(100));
    }

    #[test]
    fn test_binding_registry_apply() {
        let reg = BindingRegistry::with_builtins();
        let mut vm = GenovoVM::new();
        reg.apply_to_vm(&mut vm).unwrap();
        // Verify at least some functions were registered.
        // We can't directly inspect the native map, but we can check it
        // doesn't error.
    }

    #[test]
    fn test_vec3_components() {
        let v = ScriptValue::Vec3(1.5, 2.5, 3.5);
        assert_eq!(
            native_vec3_x(&[v.clone()]).unwrap(),
            ScriptValue::Float(1.5)
        );
        assert_eq!(
            native_vec3_y(&[v.clone()]).unwrap(),
            ScriptValue::Float(2.5)
        );
        assert_eq!(
            native_vec3_z(&[v]).unwrap(),
            ScriptValue::Float(3.5)
        );
    }

    #[test]
    fn test_vec3_scale() {
        let v = ScriptValue::Vec3(1.0, 2.0, 3.0);
        let s = ScriptValue::Float(2.0);
        let result = native_vec3_scale(&[v, s]).unwrap();
        assert_eq!(result, ScriptValue::Vec3(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_native_normalize_zero_vector() {
        let v = ScriptValue::Vec3(0.0, 0.0, 0.0);
        assert!(native_vec3_normalize(&[v]).is_err());
    }

    #[test]
    fn test_native_sqrt_negative() {
        assert!(native_sqrt(&[ScriptValue::Float(-1.0)]).is_err());
    }

    #[test]
    fn test_entity_is_valid() {
        assert_eq!(
            native_entity_is_valid(&[ScriptValue::Entity(1)]).unwrap(),
            ScriptValue::Bool(true)
        );
        assert_eq!(
            native_entity_is_valid(&[ScriptValue::Entity(0)]).unwrap(),
            ScriptValue::Bool(false)
        );
        assert_eq!(
            native_entity_is_valid(&[ScriptValue::Int(5)]).unwrap(),
            ScriptValue::Bool(false)
        );
    }

    #[test]
    fn test_position_get_set() {
        let entity = ScriptValue::Entity(9999);

        // Default position is (0, 0, 0).
        let pos = native_get_position(&[entity.clone()]).unwrap();
        assert_eq!(pos, ScriptValue::Vec3(0.0, 0.0, 0.0));

        // Set position.
        native_set_position(&[
            entity.clone(),
            ScriptValue::Float(1.0),
            ScriptValue::Float(2.0),
            ScriptValue::Float(3.0),
        ])
        .unwrap();

        // Get position back.
        let pos = native_get_position(&[entity]).unwrap();
        assert_eq!(pos, ScriptValue::Vec3(1.0, 2.0, 3.0));
    }
}
