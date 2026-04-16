//! Genovo scripting bytecode virtual machine.
//!
//! A complete stack-based VM that executes compiled bytecode for the Genovo
//! custom scripting language. Supports dynamic typing, closures, native function
//! interop, and engine entity/component access.

pub mod compiler;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use genovo_core::{EngineError, EngineResult};

// ---------------------------------------------------------------------------
// FunctionId
// ---------------------------------------------------------------------------

/// Identifies a function within the VM's function table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(pub u32);

impl fmt::Display for FunctionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fn#{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ScriptValue — dynamic value type
// ---------------------------------------------------------------------------

/// A dynamically-typed value used throughout the scripting VM.
///
/// All operations on the operand stack, in variables, and across function
/// boundaries use `ScriptValue`. The type implements coercion rules for
/// arithmetic and comparison, and a truthiness convention where `Nil` and
/// `Bool(false)` are falsy — everything else is truthy.
#[derive(Debug, Clone)]
pub enum ScriptValue {
    /// No value / null.
    Nil,
    /// Boolean.
    Bool(bool),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit floating-point.
    Float(f64),
    /// Immutable, reference-counted string.
    String(Arc<str>),
    /// 3-component vector (x, y, z).
    Vec3(f32, f32, f32),
    /// Entity handle (opaque u64 encoding id+generation).
    Entity(u64),
    /// Ordered array of values.
    Array(Vec<ScriptValue>),
    /// String-keyed map of values.
    Map(HashMap<String, ScriptValue>),
    /// Reference to a function in the VM's function table.
    Function(FunctionId),
}

impl ScriptValue {
    /// Returns `true` if this value is [`Nil`](ScriptValue::Nil).
    #[inline]
    pub fn is_nil(&self) -> bool {
        matches!(self, ScriptValue::Nil)
    }

    /// Truthiness: `Nil` and `Bool(false)` are falsy, everything else truthy.
    #[inline]
    pub fn is_truthy(&self) -> bool {
        !matches!(self, ScriptValue::Nil | ScriptValue::Bool(false))
    }

    /// Returns `true` if this value is falsy.
    #[inline]
    pub fn is_falsy(&self) -> bool {
        !self.is_truthy()
    }

    /// Attempts to extract a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ScriptValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Attempts to extract an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ScriptValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Attempts to extract a float (integers are promoted).
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ScriptValue::Float(f) => Some(*f),
            ScriptValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Attempts to extract a string slice.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ScriptValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the runtime type name (for error messages).
    pub fn type_name(&self) -> &'static str {
        match self {
            ScriptValue::Nil => "nil",
            ScriptValue::Bool(_) => "bool",
            ScriptValue::Int(_) => "int",
            ScriptValue::Float(_) => "float",
            ScriptValue::String(_) => "string",
            ScriptValue::Vec3(..) => "vec3",
            ScriptValue::Entity(_) => "entity",
            ScriptValue::Array(_) => "array",
            ScriptValue::Map(_) => "map",
            ScriptValue::Function(_) => "function",
        }
    }

    /// Construct a ScriptValue::String from anything Into<String>.
    pub fn from_string(s: impl Into<String>) -> Self {
        let s: String = s.into();
        ScriptValue::String(Arc::from(s.as_str()))
    }

    // -- Arithmetic helpers -------------------------------------------------

    /// Add two values with type coercion.
    pub fn add(&self, other: &ScriptValue) -> Result<ScriptValue, ScriptError> {
        match (self, other) {
            (ScriptValue::Int(a), ScriptValue::Int(b)) => Ok(ScriptValue::Int(a.wrapping_add(*b))),
            (ScriptValue::Float(a), ScriptValue::Float(b)) => Ok(ScriptValue::Float(a + b)),
            (ScriptValue::Int(a), ScriptValue::Float(b)) => Ok(ScriptValue::Float(*a as f64 + b)),
            (ScriptValue::Float(a), ScriptValue::Int(b)) => Ok(ScriptValue::Float(a + *b as f64)),
            (ScriptValue::String(a), ScriptValue::String(b)) => {
                let mut s = a.to_string();
                s.push_str(b);
                Ok(ScriptValue::from_string(s))
            }
            (ScriptValue::Vec3(ax, ay, az), ScriptValue::Vec3(bx, by, bz)) => {
                Ok(ScriptValue::Vec3(ax + bx, ay + by, az + bz))
            }
            _ => Err(ScriptError::TypeError(format!(
                "cannot add {} and {}",
                self.type_name(),
                other.type_name()
            ))),
        }
    }

    /// Subtract two values with type coercion.
    pub fn sub(&self, other: &ScriptValue) -> Result<ScriptValue, ScriptError> {
        match (self, other) {
            (ScriptValue::Int(a), ScriptValue::Int(b)) => Ok(ScriptValue::Int(a.wrapping_sub(*b))),
            (ScriptValue::Float(a), ScriptValue::Float(b)) => Ok(ScriptValue::Float(a - b)),
            (ScriptValue::Int(a), ScriptValue::Float(b)) => Ok(ScriptValue::Float(*a as f64 - b)),
            (ScriptValue::Float(a), ScriptValue::Int(b)) => Ok(ScriptValue::Float(a - *b as f64)),
            (ScriptValue::Vec3(ax, ay, az), ScriptValue::Vec3(bx, by, bz)) => {
                Ok(ScriptValue::Vec3(ax - bx, ay - by, az - bz))
            }
            _ => Err(ScriptError::TypeError(format!(
                "cannot subtract {} from {}",
                other.type_name(),
                self.type_name()
            ))),
        }
    }

    /// Multiply two values with type coercion.
    pub fn mul(&self, other: &ScriptValue) -> Result<ScriptValue, ScriptError> {
        match (self, other) {
            (ScriptValue::Int(a), ScriptValue::Int(b)) => Ok(ScriptValue::Int(a.wrapping_mul(*b))),
            (ScriptValue::Float(a), ScriptValue::Float(b)) => Ok(ScriptValue::Float(a * b)),
            (ScriptValue::Int(a), ScriptValue::Float(b)) => Ok(ScriptValue::Float(*a as f64 * b)),
            (ScriptValue::Float(a), ScriptValue::Int(b)) => Ok(ScriptValue::Float(a * *b as f64)),
            // scalar * vec3
            (ScriptValue::Float(s), ScriptValue::Vec3(x, y, z)) => {
                let s = *s as f32;
                Ok(ScriptValue::Vec3(x * s, y * s, z * s))
            }
            (ScriptValue::Vec3(x, y, z), ScriptValue::Float(s)) => {
                let s = *s as f32;
                Ok(ScriptValue::Vec3(x * s, y * s, z * s))
            }
            (ScriptValue::Int(s), ScriptValue::Vec3(x, y, z)) => {
                let s = *s as f32;
                Ok(ScriptValue::Vec3(x * s, y * s, z * s))
            }
            (ScriptValue::Vec3(x, y, z), ScriptValue::Int(s)) => {
                let s = *s as f32;
                Ok(ScriptValue::Vec3(x * s, y * s, z * s))
            }
            _ => Err(ScriptError::TypeError(format!(
                "cannot multiply {} and {}",
                self.type_name(),
                other.type_name()
            ))),
        }
    }

    /// Divide two values with type coercion.
    pub fn div(&self, other: &ScriptValue) -> Result<ScriptValue, ScriptError> {
        match (self, other) {
            (ScriptValue::Int(a), ScriptValue::Int(b)) => {
                if *b == 0 {
                    return Err(ScriptError::RuntimeError("division by zero".into()));
                }
                Ok(ScriptValue::Int(a / b))
            }
            (ScriptValue::Float(a), ScriptValue::Float(b)) => {
                if *b == 0.0 {
                    return Err(ScriptError::RuntimeError("division by zero".into()));
                }
                Ok(ScriptValue::Float(a / b))
            }
            (ScriptValue::Int(a), ScriptValue::Float(b)) => {
                if *b == 0.0 {
                    return Err(ScriptError::RuntimeError("division by zero".into()));
                }
                Ok(ScriptValue::Float(*a as f64 / b))
            }
            (ScriptValue::Float(a), ScriptValue::Int(b)) => {
                if *b == 0 {
                    return Err(ScriptError::RuntimeError("division by zero".into()));
                }
                Ok(ScriptValue::Float(a / *b as f64))
            }
            _ => Err(ScriptError::TypeError(format!(
                "cannot divide {} by {}",
                self.type_name(),
                other.type_name()
            ))),
        }
    }

    /// Modulo two values with type coercion.
    pub fn modulo(&self, other: &ScriptValue) -> Result<ScriptValue, ScriptError> {
        match (self, other) {
            (ScriptValue::Int(a), ScriptValue::Int(b)) => {
                if *b == 0 {
                    return Err(ScriptError::RuntimeError("modulo by zero".into()));
                }
                Ok(ScriptValue::Int(a % b))
            }
            (ScriptValue::Float(a), ScriptValue::Float(b)) => {
                if *b == 0.0 {
                    return Err(ScriptError::RuntimeError("modulo by zero".into()));
                }
                Ok(ScriptValue::Float(a % b))
            }
            (ScriptValue::Int(a), ScriptValue::Float(b)) => {
                if *b == 0.0 {
                    return Err(ScriptError::RuntimeError("modulo by zero".into()));
                }
                Ok(ScriptValue::Float(*a as f64 % b))
            }
            (ScriptValue::Float(a), ScriptValue::Int(b)) => {
                if *b == 0 {
                    return Err(ScriptError::RuntimeError("modulo by zero".into()));
                }
                Ok(ScriptValue::Float(a % *b as f64))
            }
            _ => Err(ScriptError::TypeError(format!(
                "cannot modulo {} by {}",
                self.type_name(),
                other.type_name()
            ))),
        }
    }

    /// Negate a value.
    pub fn negate(&self) -> Result<ScriptValue, ScriptError> {
        match self {
            ScriptValue::Int(i) => Ok(ScriptValue::Int(-i)),
            ScriptValue::Float(f) => Ok(ScriptValue::Float(-f)),
            ScriptValue::Vec3(x, y, z) => Ok(ScriptValue::Vec3(-x, -y, -z)),
            _ => Err(ScriptError::TypeError(format!(
                "cannot negate {}",
                self.type_name()
            ))),
        }
    }

    // -- Comparison helpers -------------------------------------------------

    /// Equality comparison.
    pub fn equals(&self, other: &ScriptValue) -> bool {
        match (self, other) {
            (ScriptValue::Nil, ScriptValue::Nil) => true,
            (ScriptValue::Bool(a), ScriptValue::Bool(b)) => a == b,
            (ScriptValue::Int(a), ScriptValue::Int(b)) => a == b,
            (ScriptValue::Float(a), ScriptValue::Float(b)) => a == b,
            (ScriptValue::Int(a), ScriptValue::Float(b)) => (*a as f64) == *b,
            (ScriptValue::Float(a), ScriptValue::Int(b)) => *a == (*b as f64),
            (ScriptValue::String(a), ScriptValue::String(b)) => a == b,
            (ScriptValue::Vec3(ax, ay, az), ScriptValue::Vec3(bx, by, bz)) => {
                ax == bx && ay == by && az == bz
            }
            (ScriptValue::Entity(a), ScriptValue::Entity(b)) => a == b,
            (ScriptValue::Function(a), ScriptValue::Function(b)) => a == b,
            (ScriptValue::Array(a), ScriptValue::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.equals(y))
            }
            (ScriptValue::Map(a), ScriptValue::Map(b)) => {
                a.len() == b.len()
                    && a.iter().all(|(k, v)| {
                        b.get(k).is_some_and(|bv| v.equals(bv))
                    })
            }
            _ => false,
        }
    }

    /// Less-than comparison.
    pub fn less_than(&self, other: &ScriptValue) -> Result<bool, ScriptError> {
        match (self, other) {
            (ScriptValue::Int(a), ScriptValue::Int(b)) => Ok(a < b),
            (ScriptValue::Float(a), ScriptValue::Float(b)) => Ok(a < b),
            (ScriptValue::Int(a), ScriptValue::Float(b)) => Ok((*a as f64) < *b),
            (ScriptValue::Float(a), ScriptValue::Int(b)) => Ok(*a < (*b as f64)),
            (ScriptValue::String(a), ScriptValue::String(b)) => Ok(*a < *b),
            _ => Err(ScriptError::TypeError(format!(
                "cannot compare {} and {} with '<'",
                self.type_name(),
                other.type_name()
            ))),
        }
    }

    /// Greater-than comparison.
    pub fn greater_than(&self, other: &ScriptValue) -> Result<bool, ScriptError> {
        match (self, other) {
            (ScriptValue::Int(a), ScriptValue::Int(b)) => Ok(a > b),
            (ScriptValue::Float(a), ScriptValue::Float(b)) => Ok(a > b),
            (ScriptValue::Int(a), ScriptValue::Float(b)) => Ok((*a as f64) > *b),
            (ScriptValue::Float(a), ScriptValue::Int(b)) => Ok(*a > (*b as f64)),
            (ScriptValue::String(a), ScriptValue::String(b)) => Ok(*a > *b),
            _ => Err(ScriptError::TypeError(format!(
                "cannot compare {} and {} with '>'",
                self.type_name(),
                other.type_name()
            ))),
        }
    }
}

impl fmt::Display for ScriptValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScriptValue::Nil => write!(f, "nil"),
            ScriptValue::Bool(b) => write!(f, "{b}"),
            ScriptValue::Int(i) => write!(f, "{i}"),
            ScriptValue::Float(v) => {
                if v.fract() == 0.0 && v.is_finite() {
                    write!(f, "{v:.1}")
                } else {
                    write!(f, "{v}")
                }
            }
            ScriptValue::String(s) => write!(f, "{s}"),
            ScriptValue::Vec3(x, y, z) => write!(f, "vec3({x}, {y}, {z})"),
            ScriptValue::Entity(id) => write!(f, "entity({id})"),
            ScriptValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            ScriptValue::Map(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
            ScriptValue::Function(id) => write!(f, "<function {id}>"),
        }
    }
}

impl PartialEq for ScriptValue {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

// ---------------------------------------------------------------------------
// ScriptError
// ---------------------------------------------------------------------------

/// Errors produced by the scripting VM during compilation or execution.
#[derive(Debug, Clone)]
pub enum ScriptError {
    /// A compile-time error (syntax, unknown variable, etc.).
    CompileError(String),
    /// A runtime type mismatch.
    TypeError(String),
    /// A generic runtime error.
    RuntimeError(String),
    /// Stack overflow.
    StackOverflow,
    /// Stack underflow (pop from empty stack).
    StackUnderflow,
    /// Division / modulo by zero — also covered by RuntimeError but offered
    /// as a distinct variant for pattern-matching convenience.
    DivisionByZero,
    /// An undefined variable was referenced.
    UndefinedVariable(String),
    /// A function was called with the wrong number of arguments.
    ArityMismatch {
        function: String,
        expected: u8,
        got: u8,
    },
    /// An assertion failed at runtime.
    AssertionFailed(String),
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScriptError::CompileError(msg) => write!(f, "compile error: {msg}"),
            ScriptError::TypeError(msg) => write!(f, "type error: {msg}"),
            ScriptError::RuntimeError(msg) => write!(f, "runtime error: {msg}"),
            ScriptError::StackOverflow => write!(f, "stack overflow"),
            ScriptError::StackUnderflow => write!(f, "stack underflow"),
            ScriptError::DivisionByZero => write!(f, "division by zero"),
            ScriptError::UndefinedVariable(name) => {
                write!(f, "undefined variable '{name}'")
            }
            ScriptError::ArityMismatch {
                function,
                expected,
                got,
            } => {
                write!(
                    f,
                    "function '{function}' expected {expected} arguments, got {got}"
                )
            }
            ScriptError::AssertionFailed(msg) => write!(f, "assertion failed: {msg}"),
        }
    }
}

impl std::error::Error for ScriptError {}

impl From<ScriptError> for EngineError {
    fn from(e: ScriptError) -> Self {
        EngineError::Other(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// OpCode — bytecode instructions
// ---------------------------------------------------------------------------

/// Individual bytecode instructions for the stack-based VM.
///
/// Each variant is encoded into a `Vec<u8>` stream by the compiler and decoded
/// by the VM's main dispatch loop. Operands are stored inline after the opcode
/// tag byte.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum OpCode {
    // -- Stack manipulation --
    /// Push constant from the constant pool at `index`.
    Push = 0,
    /// Pop and discard the top of stack.
    Pop = 1,
    /// Duplicate the top of stack.
    Dup = 2,
    /// Swap the top two stack entries.
    Swap = 3,

    // -- Arithmetic --
    /// Pop two values, push their sum.
    Add = 10,
    /// Pop two values, push (a - b) where a was pushed first.
    Sub = 11,
    /// Pop two values, push their product.
    Mul = 12,
    /// Pop two values, push (a / b).
    Div = 13,
    /// Pop two values, push (a % b).
    Mod = 14,
    /// Pop one value, push its negation.
    Neg = 15,

    // -- Comparison --
    /// Pop two values, push Bool(a == b).
    Eq = 20,
    /// Pop two values, push Bool(a != b).
    Ne = 21,
    /// Pop two values, push Bool(a < b).
    Lt = 22,
    /// Pop two values, push Bool(a > b).
    Gt = 23,
    /// Pop two values, push Bool(a <= b).
    Le = 24,
    /// Pop two values, push Bool(a >= b).
    Ge = 25,

    // -- Logic --
    /// Pop one value, push Bool(!truthy).
    Not = 30,
    /// Pop two values, push Bool(a.truthy && b.truthy).
    And = 31,
    /// Pop two values, push Bool(a.truthy || b.truthy).
    Or = 32,

    // -- Variables --
    /// Push the value of local variable at `slot`.
    GetLocal = 40,
    /// Pop and store into local variable `slot`.
    SetLocal = 41,
    /// Push the value of global variable (name from constant pool at `name_idx`).
    GetGlobal = 42,
    /// Pop and store into global variable (name from constant pool).
    SetGlobal = 43,

    // -- Control flow --
    /// Unconditional jump forward by `offset` bytes.
    Jump = 50,
    /// Pop value; if falsy, jump forward by `offset` bytes.
    JumpIfFalse = 51,
    /// Pop value; if truthy, jump forward by `offset` bytes.
    JumpIfTrue = 52,
    /// Unconditional jump backward by `offset` bytes (for loops).
    Loop = 53,

    // -- Functions --
    /// Pop `arg_count` args + the function value, invoke it, push result.
    Call = 60,
    /// Pop a value from the stack and return from current call frame.
    Return = 61,

    // -- Objects / fields --
    /// Pop an object, push the named field (name from constant pool).
    GetField = 70,
    /// Pop a value and an object, set the named field.
    SetField = 71,

    // -- Arrays --
    /// Pop `size` values from the stack, push a new array.
    ArrayNew = 80,
    /// Pop index then array, push array[index].
    ArrayGet = 81,
    /// Pop value, index, array — set array[index] = value.
    ArraySet = 82,
    /// Pop array, push its length as Int.
    ArrayLen = 83,

    // -- Special --
    /// Pop and print a value to the output buffer.
    Print = 90,
    /// Pop a value; if falsy, trigger an assertion error.
    Assert = 91,
    /// Halt execution immediately.
    Halt = 92,

    // -- Engine integration --
    /// Spawn a new entity, push its id.
    SpawnEntity = 100,
    /// Pop entity, push component value for `type_idx`.
    GetComponent = 101,
    /// Pop value then entity, set component `type_idx`.
    SetComponent = 102,
    /// Call a native function by name index with `arg_count` arguments.
    CallNative = 110,
    /// Push Nil onto the stack.
    PushNil = 111,
    /// Push Bool(true).
    PushTrue = 112,
    /// Push Bool(false).
    PushFalse = 113,

    // -- Optimizer --
    /// No operation (placeholder used by the bytecode optimizer).
    Nop = 120,
}

impl OpCode {
    /// Decode an opcode from a byte, returning `None` for unknown tags.
    pub fn from_byte(byte: u8) -> Option<OpCode> {
        match byte {
            0 => Some(OpCode::Push),
            1 => Some(OpCode::Pop),
            2 => Some(OpCode::Dup),
            3 => Some(OpCode::Swap),
            10 => Some(OpCode::Add),
            11 => Some(OpCode::Sub),
            12 => Some(OpCode::Mul),
            13 => Some(OpCode::Div),
            14 => Some(OpCode::Mod),
            15 => Some(OpCode::Neg),
            20 => Some(OpCode::Eq),
            21 => Some(OpCode::Ne),
            22 => Some(OpCode::Lt),
            23 => Some(OpCode::Gt),
            24 => Some(OpCode::Le),
            25 => Some(OpCode::Ge),
            30 => Some(OpCode::Not),
            31 => Some(OpCode::And),
            32 => Some(OpCode::Or),
            40 => Some(OpCode::GetLocal),
            41 => Some(OpCode::SetLocal),
            42 => Some(OpCode::GetGlobal),
            43 => Some(OpCode::SetGlobal),
            50 => Some(OpCode::Jump),
            51 => Some(OpCode::JumpIfFalse),
            52 => Some(OpCode::JumpIfTrue),
            53 => Some(OpCode::Loop),
            60 => Some(OpCode::Call),
            61 => Some(OpCode::Return),
            70 => Some(OpCode::GetField),
            71 => Some(OpCode::SetField),
            80 => Some(OpCode::ArrayNew),
            81 => Some(OpCode::ArrayGet),
            82 => Some(OpCode::ArraySet),
            83 => Some(OpCode::ArrayLen),
            90 => Some(OpCode::Print),
            91 => Some(OpCode::Assert),
            92 => Some(OpCode::Halt),
            100 => Some(OpCode::SpawnEntity),
            101 => Some(OpCode::GetComponent),
            102 => Some(OpCode::SetComponent),
            110 => Some(OpCode::CallNative),
            111 => Some(OpCode::PushNil),
            112 => Some(OpCode::PushTrue),
            113 => Some(OpCode::PushFalse),
            120 => Some(OpCode::Nop),
            _ => None,
        }
    }

    /// Encode this opcode as its tag byte.
    #[inline]
    pub fn to_byte(self) -> u8 {
        self as u8
    }
}

// ---------------------------------------------------------------------------
// Chunk — compiled bytecode container
// ---------------------------------------------------------------------------

/// A chunk of compiled bytecode with its associated constant pool and debug
/// information.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The raw bytecode stream. Each instruction is a 1-byte opcode tag
    /// optionally followed by 2-byte (u16 LE) operands.
    pub code: Vec<u8>,
    /// The constant pool. Operand indices in Push / GetGlobal etc. index
    /// into this vector.
    pub constants: Vec<ScriptValue>,
    /// Source line numbers, one entry per byte in `code` (for error messages).
    pub line_numbers: Vec<u32>,
}

impl Chunk {
    /// Create a new empty chunk.
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            constants: Vec::new(),
            line_numbers: Vec::new(),
        }
    }

    /// Emit an opcode with no operand.
    pub fn emit_op(&mut self, op: OpCode, line: u32) {
        self.code.push(op.to_byte());
        self.line_numbers.push(line);
    }

    /// Emit an opcode followed by a u16 operand (little-endian).
    pub fn emit_op_u16(&mut self, op: OpCode, operand: u16, line: u32) {
        self.code.push(op.to_byte());
        self.line_numbers.push(line);
        let bytes = operand.to_le_bytes();
        self.code.push(bytes[0]);
        self.line_numbers.push(line);
        self.code.push(bytes[1]);
        self.line_numbers.push(line);
    }

    /// Add a constant to the pool and return its index.
    pub fn add_constant(&mut self, value: ScriptValue) -> u16 {
        let idx = self.constants.len();
        assert!(
            idx <= u16::MAX as usize,
            "constant pool overflow (max 65535)"
        );
        self.constants.push(value);
        idx as u16
    }

    /// Emit a Push instruction for a constant.
    pub fn emit_constant(&mut self, value: ScriptValue, line: u32) {
        let idx = self.add_constant(value);
        self.emit_op_u16(OpCode::Push, idx, line);
    }

    /// Read a u16 operand at the given byte offset (little-endian).
    #[inline]
    pub fn read_u16(&self, offset: usize) -> u16 {
        u16::from_le_bytes([self.code[offset], self.code[offset + 1]])
    }

    /// Current length of the code buffer (used for jump patching).
    #[inline]
    pub fn len(&self) -> usize {
        self.code.len()
    }

    /// Whether the code buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// Emit a placeholder jump instruction, returning the offset to patch.
    pub fn emit_jump(&mut self, op: OpCode, line: u32) -> usize {
        self.emit_op_u16(op, 0xFFFF, line);
        // return offset of the u16 operand (the byte after the opcode)
        self.code.len() - 2
    }

    /// Patch a previously emitted jump's u16 operand.
    pub fn patch_jump(&mut self, offset: usize) -> Result<(), ScriptError> {
        // Jump target = current position. The operand is the number of bytes
        // to skip *after* reading the operand (i.e., from offset+2 to here).
        let jump = self.code.len() - offset - 2;
        if jump > u16::MAX as usize {
            return Err(ScriptError::CompileError(
                "jump offset too large".into(),
            ));
        }
        let bytes = (jump as u16).to_le_bytes();
        self.code[offset] = bytes[0];
        self.code[offset + 1] = bytes[1];
        Ok(())
    }

    /// Emit a Loop instruction that jumps backwards.
    pub fn emit_loop(&mut self, loop_start: usize, line: u32) -> Result<(), ScriptError> {
        // +3 accounts for the Loop opcode byte + 2-byte operand we're about
        // to emit.
        let offset = self.code.len() - loop_start + 3;
        if offset > u16::MAX as usize {
            return Err(ScriptError::CompileError(
                "loop body too large".into(),
            ));
        }
        self.emit_op_u16(OpCode::Loop, offset as u16, line);
        Ok(())
    }

    /// Return the line number associated with a bytecode offset.
    pub fn get_line(&self, offset: usize) -> u32 {
        if offset < self.line_numbers.len() {
            self.line_numbers[offset]
        } else {
            0
        }
    }

    /// Disassemble this chunk into a human-readable string for debugging.
    pub fn disassemble(&self, name: &str) -> String {
        let mut out = format!("== {name} ==\n");
        let mut offset = 0;
        while offset < self.code.len() {
            let line = self.get_line(offset);
            let byte = self.code[offset];
            let op = OpCode::from_byte(byte);
            match op {
                Some(op) => {
                    let op_name = format!("{:?}", op);
                    match op {
                        OpCode::Push
                        | OpCode::GetLocal
                        | OpCode::SetLocal
                        | OpCode::GetGlobal
                        | OpCode::SetGlobal
                        | OpCode::Jump
                        | OpCode::JumpIfFalse
                        | OpCode::JumpIfTrue
                        | OpCode::Loop
                        | OpCode::Call
                        | OpCode::GetField
                        | OpCode::SetField
                        | OpCode::ArrayNew
                        | OpCode::CallNative
                        | OpCode::GetComponent
                        | OpCode::SetComponent => {
                            let operand = self.read_u16(offset + 1);
                            out.push_str(&format!(
                                "{offset:04}  L{line:<4}  {op_name:<16} {operand}\n"
                            ));
                            offset += 3;
                        }
                        _ => {
                            out.push_str(&format!(
                                "{offset:04}  L{line:<4}  {op_name}\n"
                            ));
                            offset += 1;
                        }
                    }
                }
                None => {
                    out.push_str(&format!(
                        "{offset:04}  L{line:<4}  <unknown {byte:#04x}>\n"
                    ));
                    offset += 1;
                }
            }
        }
        out
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ScriptFunction
// ---------------------------------------------------------------------------

/// A compiled function ready for execution by the VM.
#[derive(Debug, Clone)]
pub struct ScriptFunction {
    /// The function name (for stack traces and debugging).
    pub name: String,
    /// Number of parameters this function expects.
    pub arity: u8,
    /// Compiled bytecode for the function body.
    pub chunk: Chunk,
    /// Number of local variable slots required (including parameters).
    pub local_count: u8,
}

impl ScriptFunction {
    /// Create a new script function.
    pub fn new(name: impl Into<String>, arity: u8) -> Self {
        Self {
            name: name.into(),
            arity,
            chunk: Chunk::new(),
            local_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// CallFrame
// ---------------------------------------------------------------------------

/// A single activation record on the call stack.
///
/// Each function call creates a new frame that remembers the caller's
/// instruction pointer and the base of the callee's local-variable window on
/// the operand stack.
#[derive(Debug, Clone)]
pub struct CallFrame {
    /// Index into the VM's function table identifying the executing function.
    pub function_id: usize,
    /// Saved instruction pointer (byte offset into the function's chunk).
    pub ip: usize,
    /// Offset in the operand stack where this frame's locals begin.
    pub stack_base: usize,
}

// ---------------------------------------------------------------------------
// NativeFn
// ---------------------------------------------------------------------------

/// Type alias for Rust functions callable from scripts.
///
/// Receives a slice of `ScriptValue` arguments and returns either a value
/// or a `ScriptError`.
pub type NativeFn = Box<dyn Fn(&[ScriptValue]) -> Result<ScriptValue, ScriptError> + Send + Sync>;

// ---------------------------------------------------------------------------
// ScriptContext — bridge between VM and engine
// ---------------------------------------------------------------------------

/// Execution context passed to the VM when running scripts.
///
/// Gives scripts access to engine timing and global data. The VM reads
/// `delta_time` / `total_time` and exposes them as built-in globals.
pub struct ScriptContext {
    /// Per-instance global variables shared with all scripts in this context.
    pub globals: HashMap<String, ScriptValue>,
    /// Delta time for the current frame (seconds).
    pub delta_time: f64,
    /// Total elapsed time since engine start (seconds).
    pub total_time: f64,
}

impl ScriptContext {
    /// Creates a new, empty context.
    pub fn new() -> Self {
        Self {
            globals: HashMap::new(),
            delta_time: 0.0,
            total_time: 0.0,
        }
    }

    /// Sets a context-level global variable.
    pub fn set_global(&mut self, name: impl Into<String>, value: ScriptValue) {
        self.globals.insert(name.into(), value);
    }

    /// Gets a context-level global, returning `Nil` if absent.
    pub fn get_global(&self, name: &str) -> ScriptValue {
        self.globals
            .get(name)
            .cloned()
            .unwrap_or(ScriptValue::Nil)
    }
}

impl Default for ScriptContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VM — the virtual machine
// ---------------------------------------------------------------------------

/// Maximum operand stack depth.
const MAX_STACK: usize = 256;
/// Maximum call-frame depth.
const MAX_FRAMES: usize = 64;
/// Maximum number of instructions before the VM force-halts (prevents infinite loops).
const MAX_INSTRUCTIONS: usize = 10_000_000;

/// The Genovo bytecode virtual machine.
///
/// Executes compiled `ScriptFunction` bytecode on a value stack, maintaining
/// a call stack of `CallFrame`s. Native Rust functions can be registered and
/// are callable from scripts. Global variables are shared across all function
/// invocations within a single `run()` call.
pub struct VM {
    /// Operand stack.
    stack: Vec<ScriptValue>,
    /// Call stack.
    frames: Vec<CallFrame>,
    /// Global variables.
    globals: HashMap<String, ScriptValue>,
    /// Registered native functions (name -> closure).
    native_functions: HashMap<String, NativeFn>,
    /// All compiled functions. Index 0 is always the top-level script.
    functions: Vec<ScriptFunction>,
    /// Instruction pointer into the *current frame's* function chunk.
    ip: usize,
    /// Output buffer for `print` statements (captured for testing / logging).
    output: Vec<String>,
    /// Next entity ID to assign (simple counter for script-spawned entities).
    next_entity_id: u64,
}

impl VM {
    /// Create a new, empty VM.
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(MAX_STACK),
            frames: Vec::with_capacity(MAX_FRAMES),
            globals: HashMap::new(),
            native_functions: HashMap::new(),
            functions: Vec::new(),
            ip: 0,
            output: Vec::new(),
            next_entity_id: 1,
        }
    }

    /// Register a native Rust function that scripts can call by name.
    pub fn register_native(&mut self, name: &str, func: NativeFn) {
        self.native_functions.insert(name.to_string(), func);
    }

    /// Set a global variable before or between executions.
    pub fn set_global(&mut self, name: impl Into<String>, value: ScriptValue) {
        self.globals.insert(name.into(), value);
    }

    /// Get a global variable.
    pub fn get_global(&self, name: &str) -> Option<&ScriptValue> {
        self.globals.get(name)
    }

    /// Load a compiled function into the VM's function table and return its
    /// index.
    pub fn load_function(&mut self, func: ScriptFunction) -> usize {
        let id = self.functions.len();
        self.functions.push(func);
        id
    }

    /// Return captured print output.
    pub fn output(&self) -> &[String] {
        &self.output
    }

    /// Clear captured output.
    pub fn clear_output(&mut self) {
        self.output.clear();
    }

    /// Return the number of loaded functions.
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// Execute the function at index `func_id` with the given arguments.
    pub fn call(&mut self, func_id: usize, args: &[ScriptValue]) -> Result<ScriptValue, ScriptError> {
        if func_id >= self.functions.len() {
            return Err(ScriptError::RuntimeError(format!(
                "function index {func_id} out of range"
            )));
        }

        let arity = self.functions[func_id].arity;
        if args.len() != arity as usize {
            return Err(ScriptError::ArityMismatch {
                function: self.functions[func_id].name.clone(),
                expected: arity,
                got: args.len() as u8,
            });
        }

        // Push a sentinel nil for the function slot, then arguments.
        let stack_base = self.stack.len();
        self.stack.push(ScriptValue::Nil); // slot 0 = function reference
        for arg in args {
            self.push(arg.clone())?;
        }

        // Don't pre-allocate remaining locals. The compiler emits pushes
        // for each `let` declaration as part of the bytecode.

        self.frames.push(CallFrame {
            function_id: func_id,
            ip: 0,
            stack_base,
        });
        self.ip = 0;

        self.run()
    }

    /// Execute the top-level script (function index 0).
    pub fn run_script(&mut self) -> Result<ScriptValue, ScriptError> {
        if self.functions.is_empty() {
            return Err(ScriptError::RuntimeError(
                "no functions loaded".into(),
            ));
        }
        self.stack.clear();
        self.frames.clear();
        self.ip = 0;

        // Don't pre-allocate locals for the top-level script.
        // The compiler emits code that pushes values onto the stack as local
        // variables are declared (via `let`), so the stack grows naturally.

        self.frames.push(CallFrame {
            function_id: 0,
            ip: 0,
            stack_base: 0,
        });

        self.run()
    }

    // -- Stack operations ---------------------------------------------------

    #[inline]
    fn push(&mut self, value: ScriptValue) -> Result<(), ScriptError> {
        if self.stack.len() >= MAX_STACK {
            return Err(ScriptError::StackOverflow);
        }
        self.stack.push(value);
        Ok(())
    }

    #[inline]
    fn pop(&mut self) -> Result<ScriptValue, ScriptError> {
        self.stack.pop().ok_or(ScriptError::StackUnderflow)
    }

    #[inline]
    fn peek(&self, distance: usize) -> Result<&ScriptValue, ScriptError> {
        if self.stack.len() <= distance {
            return Err(ScriptError::StackUnderflow);
        }
        Ok(&self.stack[self.stack.len() - 1 - distance])
    }

    // -- Helper to get current chunk ----------------------------------------

    #[inline]
    fn current_frame(&self) -> &CallFrame {
        self.frames.last().expect("call stack is empty")
    }

    #[inline]
    fn current_function_id(&self) -> usize {
        self.current_frame().function_id
    }

    #[inline]
    fn read_byte(&mut self) -> u8 {
        let fid = self.current_function_id();
        let byte = self.functions[fid].chunk.code[self.ip];
        self.ip += 1;
        byte
    }

    #[inline]
    fn read_u16(&mut self) -> u16 {
        let fid = self.current_function_id();
        let val = self.functions[fid].chunk.read_u16(self.ip);
        self.ip += 2;
        val
    }

    #[inline]
    fn get_constant(&self, idx: u16) -> ScriptValue {
        let fid = self.current_function_id();
        self.functions[fid].chunk.constants[idx as usize].clone()
    }

    #[inline]
    fn get_line(&self) -> u32 {
        let fid = self.current_function_id();
        self.functions[fid].chunk.get_line(self.ip.saturating_sub(1))
    }

    /// Build a stack trace string for error reporting.
    fn stack_trace(&self) -> String {
        let mut trace = String::from("stack trace:\n");
        for (i, frame) in self.frames.iter().rev().enumerate() {
            let func = &self.functions[frame.function_id];
            let line = func.chunk.get_line(frame.ip.saturating_sub(1));
            trace.push_str(&format!(
                "  [{i}] in {}() at line {line}\n",
                func.name
            ));
        }
        trace
    }

    /// Wrap a ScriptError with stack trace information.
    fn runtime_error(&self, err: ScriptError) -> ScriptError {
        let trace = self.stack_trace();
        ScriptError::RuntimeError(format!("{err}\n{trace}"))
    }

    // -- Main execution loop ------------------------------------------------

    /// Execute bytecode starting from the current instruction pointer until
    /// a `Return` or `Halt` is encountered, or an error occurs.
    fn run(&mut self) -> Result<ScriptValue, ScriptError> {
        let mut instruction_count: usize = 0;

        loop {
            instruction_count += 1;
            if instruction_count > MAX_INSTRUCTIONS {
                return Err(self.runtime_error(ScriptError::RuntimeError(
                    "maximum instruction count exceeded (possible infinite loop)".into(),
                )));
            }

            let fid = self.current_function_id();
            if self.ip >= self.functions[fid].chunk.code.len() {
                // Reached end of function without explicit return — return Nil.
                if self.frames.len() <= 1 {
                    return Ok(ScriptValue::Nil);
                }
                let frame = self.frames.pop().unwrap();
                self.stack.truncate(frame.stack_base);
                self.push(ScriptValue::Nil)?;
                self.ip = self.frames.last().unwrap().ip;
                continue;
            }

            let byte = self.read_byte();
            let op = match OpCode::from_byte(byte) {
                Some(op) => op,
                None => {
                    return Err(self.runtime_error(ScriptError::RuntimeError(
                        format!("unknown opcode {byte:#04x} at ip {}", self.ip - 1),
                    )));
                }
            };

            match op {
                // -- Stack manipulation ------------------------------------

                OpCode::Push => {
                    let idx = self.read_u16();
                    let val = self.get_constant(idx);
                    self.push(val)?;
                }

                OpCode::PushNil => {
                    self.push(ScriptValue::Nil)?;
                }

                OpCode::PushTrue => {
                    self.push(ScriptValue::Bool(true))?;
                }

                OpCode::PushFalse => {
                    self.push(ScriptValue::Bool(false))?;
                }

                OpCode::Pop => {
                    self.pop()?;
                }

                OpCode::Dup => {
                    let val = self.peek(0)?.clone();
                    self.push(val)?;
                }

                OpCode::Swap => {
                    let len = self.stack.len();
                    if len < 2 {
                        return Err(ScriptError::StackUnderflow);
                    }
                    self.stack.swap(len - 1, len - 2);
                }

                // -- Arithmetic --------------------------------------------

                OpCode::Add => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = a.add(&b).map_err(|e| self.runtime_error(e))?;
                    self.push(result)?;
                }

                OpCode::Sub => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = a.sub(&b).map_err(|e| self.runtime_error(e))?;
                    self.push(result)?;
                }

                OpCode::Mul => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = a.mul(&b).map_err(|e| self.runtime_error(e))?;
                    self.push(result)?;
                }

                OpCode::Div => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = a.div(&b).map_err(|e| self.runtime_error(e))?;
                    self.push(result)?;
                }

                OpCode::Mod => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = a.modulo(&b).map_err(|e| self.runtime_error(e))?;
                    self.push(result)?;
                }

                OpCode::Neg => {
                    let a = self.pop()?;
                    let result = a.negate().map_err(|e| self.runtime_error(e))?;
                    self.push(result)?;
                }

                // -- Comparison --------------------------------------------

                OpCode::Eq => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(ScriptValue::Bool(a.equals(&b)))?;
                }

                OpCode::Ne => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(ScriptValue::Bool(!a.equals(&b)))?;
                }

                OpCode::Lt => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = a.less_than(&b).map_err(|e| self.runtime_error(e))?;
                    self.push(ScriptValue::Bool(result))?;
                }

                OpCode::Gt => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = a.greater_than(&b).map_err(|e| self.runtime_error(e))?;
                    self.push(ScriptValue::Bool(result))?;
                }

                OpCode::Le => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let gt = a.greater_than(&b).map_err(|e| self.runtime_error(e))?;
                    self.push(ScriptValue::Bool(!gt))?;
                }

                OpCode::Ge => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let lt = a.less_than(&b).map_err(|e| self.runtime_error(e))?;
                    self.push(ScriptValue::Bool(!lt))?;
                }

                // -- Logic -------------------------------------------------

                OpCode::Not => {
                    let a = self.pop()?;
                    self.push(ScriptValue::Bool(a.is_falsy()))?;
                }

                OpCode::And => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    // Short-circuit semantics: return last evaluated value.
                    if a.is_falsy() {
                        self.push(a)?;
                    } else {
                        self.push(b)?;
                    }
                }

                OpCode::Or => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    if a.is_truthy() {
                        self.push(a)?;
                    } else {
                        self.push(b)?;
                    }
                }

                // -- Variables ---------------------------------------------

                OpCode::GetLocal => {
                    let slot = self.read_u16() as usize;
                    let base = self.current_frame().stack_base;
                    let idx = base + slot;
                    if idx >= self.stack.len() {
                        return Err(self.runtime_error(ScriptError::RuntimeError(
                            format!("local slot {slot} out of range"),
                        )));
                    }
                    let val = self.stack[idx].clone();
                    self.push(val)?;
                }

                OpCode::SetLocal => {
                    let slot = self.read_u16() as usize;
                    let base = self.current_frame().stack_base;
                    let idx = base + slot;
                    let val = self.peek(0)?.clone();
                    if idx >= self.stack.len() {
                        return Err(self.runtime_error(ScriptError::RuntimeError(
                            format!("local slot {slot} out of range"),
                        )));
                    }
                    self.stack[idx] = val;
                }

                OpCode::GetGlobal => {
                    let name_idx = self.read_u16();
                    let name_val = self.get_constant(name_idx);
                    let name = match &name_val {
                        ScriptValue::String(s) => s.to_string(),
                        _ => {
                            return Err(self.runtime_error(ScriptError::RuntimeError(
                                "global name is not a string".into(),
                            )));
                        }
                    };
                    let val = self
                        .globals
                        .get(&name)
                        .cloned()
                        .ok_or_else(|| {
                            self.runtime_error(ScriptError::UndefinedVariable(name.clone()))
                        })?;
                    self.push(val)?;
                }

                OpCode::SetGlobal => {
                    let name_idx = self.read_u16();
                    let name_val = self.get_constant(name_idx);
                    let name = match &name_val {
                        ScriptValue::String(s) => s.to_string(),
                        _ => {
                            return Err(self.runtime_error(ScriptError::RuntimeError(
                                "global name is not a string".into(),
                            )));
                        }
                    };
                    let val = self.peek(0)?.clone();
                    self.globals.insert(name, val);
                }

                // -- Control flow ------------------------------------------

                OpCode::Jump => {
                    let offset = self.read_u16() as usize;
                    self.ip += offset;
                }

                OpCode::JumpIfFalse => {
                    let offset = self.read_u16() as usize;
                    let condition = self.pop()?;
                    if condition.is_falsy() {
                        self.ip += offset;
                    }
                }

                OpCode::JumpIfTrue => {
                    let offset = self.read_u16() as usize;
                    let condition = self.pop()?;
                    if condition.is_truthy() {
                        self.ip += offset;
                    }
                }

                OpCode::Loop => {
                    let offset = self.read_u16() as usize;
                    self.ip -= offset;
                }

                // -- Functions ---------------------------------------------

                OpCode::Call => {
                    let arg_count = self.read_u16() as u8;
                    // The function value is below the arguments on the stack.
                    let func_val = self
                        .peek(arg_count as usize)?
                        .clone();

                    match func_val {
                        ScriptValue::Function(FunctionId(fid)) => {
                            let fid = fid as usize;
                            if fid >= self.functions.len() {
                                return Err(self.runtime_error(ScriptError::RuntimeError(
                                    format!("invalid function id {fid}"),
                                )));
                            }

                            let expected = self.functions[fid].arity;
                            if arg_count != expected {
                                return Err(self.runtime_error(ScriptError::ArityMismatch {
                                    function: self.functions[fid].name.clone(),
                                    expected,
                                    got: arg_count,
                                }));
                            }

                            let stack_base =
                                self.stack.len() - arg_count as usize - 1;

                            // Don't pre-allocate locals. The compiler
                            // emits code that pushes values onto the stack
                            // as locals are declared via `let`.

                            // Save current ip in the calling frame.
                            if let Some(frame) = self.frames.last_mut() {
                                frame.ip = self.ip;
                            }

                            if self.frames.len() >= MAX_FRAMES {
                                return Err(self.runtime_error(ScriptError::StackOverflow));
                            }

                            self.frames.push(CallFrame {
                                function_id: fid,
                                ip: 0,
                                stack_base,
                            });
                            self.ip = 0;
                        }
                        _ => {
                            return Err(self.runtime_error(ScriptError::TypeError(
                                format!("cannot call value of type {}", func_val.type_name()),
                            )));
                        }
                    }
                }

                OpCode::CallNative => {
                    let name_idx = self.read_u16();
                    let arg_count_byte = self.read_byte();
                    let arg_count = arg_count_byte as usize;

                    let name_val = self.get_constant(name_idx);
                    let name = match &name_val {
                        ScriptValue::String(s) => s.to_string(),
                        _ => {
                            return Err(self.runtime_error(ScriptError::RuntimeError(
                                "native function name is not a string".into(),
                            )));
                        }
                    };

                    // Collect arguments from the stack.
                    let mut args = Vec::with_capacity(arg_count);
                    for _ in 0..arg_count {
                        args.push(self.pop()?);
                    }
                    args.reverse();

                    // Look up and call the native function.
                    // We need to use unsafe to split the borrow. This is safe
                    // because the native function only accesses its arguments
                    // and doesn't touch the VM's native_functions map.
                    let func_ptr = self.native_functions.get(&name).map(|f| {
                        f as *const (dyn Fn(&[ScriptValue]) -> Result<ScriptValue, ScriptError>
                                  + Send
                                  + Sync)
                    });

                    match func_ptr {
                        Some(ptr) => {
                            // SAFETY: We hold an immutable reference to the
                            // function for the duration of the call. The call
                            // cannot mutate the map.
                            let result = unsafe { (*ptr)(&args) }
                                .map_err(|e| self.runtime_error(e))?;
                            self.push(result)?;
                        }
                        None => {
                            return Err(self.runtime_error(ScriptError::UndefinedVariable(
                                format!("native function '{name}' not found"),
                            )));
                        }
                    }
                }

                OpCode::Return => {
                    let result = self.pop()?;

                    if self.frames.len() <= 1 {
                        // Returning from top-level — done.
                        return Ok(result);
                    }

                    let frame = self.frames.pop().unwrap();
                    self.stack.truncate(frame.stack_base);
                    self.push(result)?;

                    // Restore caller's IP.
                    self.ip = self.frames.last().unwrap().ip;
                }

                // -- Objects / fields --------------------------------------

                OpCode::GetField => {
                    let name_idx = self.read_u16();
                    let name_val = self.get_constant(name_idx);
                    let name = match &name_val {
                        ScriptValue::String(s) => s.to_string(),
                        _ => {
                            return Err(self.runtime_error(ScriptError::RuntimeError(
                                "field name is not a string".into(),
                            )));
                        }
                    };

                    let obj = self.pop()?;
                    match obj {
                        ScriptValue::Map(map) => {
                            let val = map
                                .get(&name)
                                .cloned()
                                .unwrap_or(ScriptValue::Nil);
                            self.push(val)?;
                        }
                        _ => {
                            return Err(self.runtime_error(ScriptError::TypeError(
                                format!(
                                    "cannot access field '{name}' on {}",
                                    obj.type_name()
                                ),
                            )));
                        }
                    }
                }

                OpCode::SetField => {
                    let name_idx = self.read_u16();
                    let name_val = self.get_constant(name_idx);
                    let name = match &name_val {
                        ScriptValue::String(s) => s.to_string(),
                        _ => {
                            return Err(self.runtime_error(ScriptError::RuntimeError(
                                "field name is not a string".into(),
                            )));
                        }
                    };

                    let value = self.pop()?;
                    let obj = self.pop()?;
                    match obj {
                        ScriptValue::Map(mut map) => {
                            map.insert(name, value);
                            self.push(ScriptValue::Map(map))?;
                        }
                        _ => {
                            return Err(self.runtime_error(ScriptError::TypeError(
                                format!(
                                    "cannot set field '{name}' on {}",
                                    obj.type_name()
                                ),
                            )));
                        }
                    }
                }

                // -- Arrays ------------------------------------------------

                OpCode::ArrayNew => {
                    let size = self.read_u16() as usize;
                    let mut arr = Vec::with_capacity(size);
                    for _ in 0..size {
                        arr.push(self.pop()?);
                    }
                    arr.reverse();
                    self.push(ScriptValue::Array(arr))?;
                }

                OpCode::ArrayGet => {
                    let index = self.pop()?;
                    let arr = self.pop()?;
                    match (&arr, &index) {
                        (ScriptValue::Array(a), ScriptValue::Int(i)) => {
                            let idx = *i as usize;
                            if idx >= a.len() {
                                return Err(self.runtime_error(ScriptError::RuntimeError(
                                    format!(
                                        "array index {idx} out of bounds (len {})",
                                        a.len()
                                    ),
                                )));
                            }
                            self.push(a[idx].clone())?;
                        }
                        _ => {
                            return Err(self.runtime_error(ScriptError::TypeError(
                                format!(
                                    "cannot index {} with {}",
                                    arr.type_name(),
                                    index.type_name()
                                ),
                            )));
                        }
                    }
                }

                OpCode::ArraySet => {
                    let value = self.pop()?;
                    let index = self.pop()?;
                    let arr = self.pop()?;
                    match (arr, &index) {
                        (ScriptValue::Array(mut a), ScriptValue::Int(i)) => {
                            let idx = *i as usize;
                            if idx >= a.len() {
                                return Err(self.runtime_error(ScriptError::RuntimeError(
                                    format!(
                                        "array index {idx} out of bounds (len {})",
                                        a.len()
                                    ),
                                )));
                            }
                            a[idx] = value;
                            self.push(ScriptValue::Array(a))?;
                        }
                        (other, _) => {
                            return Err(self.runtime_error(ScriptError::TypeError(
                                format!(
                                    "cannot index-assign {} with {}",
                                    other.type_name(),
                                    index.type_name()
                                ),
                            )));
                        }
                    }
                }

                OpCode::ArrayLen => {
                    let arr = self.pop()?;
                    match &arr {
                        ScriptValue::Array(a) => {
                            self.push(ScriptValue::Int(a.len() as i64))?;
                        }
                        ScriptValue::String(s) => {
                            self.push(ScriptValue::Int(s.len() as i64))?;
                        }
                        _ => {
                            return Err(self.runtime_error(ScriptError::TypeError(
                                format!("cannot get length of {}", arr.type_name()),
                            )));
                        }
                    }
                }

                // -- Special -----------------------------------------------

                OpCode::Print => {
                    let val = self.pop()?;
                    let text = format!("{val}");
                    log::info!("[script] {text}");
                    self.output.push(text);
                }

                OpCode::Assert => {
                    let val = self.pop()?;
                    if val.is_falsy() {
                        let line = self.get_line();
                        return Err(self.runtime_error(ScriptError::AssertionFailed(
                            format!("assertion failed at line {line}"),
                        )));
                    }
                }

                OpCode::Halt => {
                    let result = if self.stack.is_empty() {
                        ScriptValue::Nil
                    } else {
                        self.pop()?
                    };
                    return Ok(result);
                }

                // -- Engine integration ------------------------------------

                OpCode::SpawnEntity => {
                    let id = self.next_entity_id;
                    self.next_entity_id += 1;
                    self.push(ScriptValue::Entity(id))?;
                }

                OpCode::GetComponent => {
                    let _type_idx = self.read_u16();
                    let _entity = self.pop()?;
                    // In a real engine, this would look up the component via ECS.
                    // For now, push Nil.
                    self.push(ScriptValue::Nil)?;
                }

                OpCode::SetComponent => {
                    let _type_idx = self.read_u16();
                    let _value = self.pop()?;
                    let _entity = self.pop()?;
                    // In a real engine, this would set the component via ECS.
                }

                OpCode::Nop => {
                    // No operation — do nothing.
                }
            }
        }
    }
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ScriptVM trait (compatibility bridge)
// ---------------------------------------------------------------------------

/// Trait defining the interface for a scripting language virtual machine.
///
/// The Genovo bytecode VM implements this trait so it can be used
/// interchangeably with other potential scripting backends.
pub trait ScriptVM: Send + Sync {
    /// Returns the name of the scripting language.
    fn language_name(&self) -> &str;

    /// Loads and compiles a script from source code.
    fn load_script(&mut self, name: &str, source: &str) -> EngineResult<()>;

    /// Executes a previously loaded script by name.
    fn execute(
        &mut self,
        name: &str,
        ctx: &mut ScriptContext,
    ) -> EngineResult<ScriptValue>;

    /// Calls a named function with arguments.
    fn call_function(
        &mut self,
        function_name: &str,
        args: &[ScriptValue],
        ctx: &mut ScriptContext,
    ) -> EngineResult<ScriptValue>;

    /// Gets a global variable.
    fn get_global(&self, name: &str) -> EngineResult<ScriptValue>;

    /// Sets a global variable.
    fn set_global(&mut self, name: &str, value: ScriptValue) -> EngineResult<()>;

    /// Registers a native function.
    fn register_function(&mut self, name: &str, func: NativeFn) -> EngineResult<()>;

    /// Garbage collection (no-op for this VM).
    fn collect_garbage(&mut self) {}

    /// Approximate memory usage.
    fn memory_usage(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// GenovoVM — ScriptVM adapter for the bytecode VM
// ---------------------------------------------------------------------------

/// Adapter that wraps the bytecode [`VM`] behind the [`ScriptVM`] trait,
/// providing the `load_script` / `execute` / `call_function` interface
/// expected by the rest of the engine.
pub struct GenovoVM {
    /// The underlying bytecode VM.
    vm: VM,
    /// Compiled scripts, keyed by name. Each entry maps to the top-level
    /// function index in the VM's function table.
    scripts: HashMap<String, usize>,
    /// Map from user-visible function name to function index.
    function_map: HashMap<String, usize>,
}

impl GenovoVM {
    /// Create a new Genovo scripting VM.
    pub fn new() -> Self {
        Self {
            vm: VM::new(),
            scripts: HashMap::new(),
            function_map: HashMap::new(),
        }
    }

    /// Direct access to the underlying bytecode VM.
    pub fn inner(&self) -> &VM {
        &self.vm
    }

    /// Mutable access to the underlying bytecode VM.
    pub fn inner_mut(&mut self) -> &mut VM {
        &mut self.vm
    }

    /// Return captured output lines.
    pub fn output(&self) -> &[String] {
        self.vm.output()
    }
}

impl Default for GenovoVM {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptVM for GenovoVM {
    fn language_name(&self) -> &str {
        "Genovo"
    }

    fn load_script(&mut self, name: &str, source: &str) -> EngineResult<()> {
        use compiler::Compiler;

        let mut compiler = Compiler::new(source, name);
        let compiled = compiler.compile().map_err(|e| {
            EngineError::Other(format!("compilation failed for '{name}': {e}"))
        })?;

        // Load all compiled functions into the VM.
        let base_id = self.vm.functions.len();

        // Load the top-level script function.
        let script_func_id = self.vm.load_function(compiled.script);

        // Load sub-functions and register them by name.
        for func in compiled.functions {
            let fname = func.name.clone();
            let fid = self.vm.load_function(func);
            self.function_map.insert(fname, fid);
        }

        // Also register function values as globals so scripts can call them.
        for (fname, &fid) in &self.function_map {
            self.vm.set_global(
                fname.clone(),
                ScriptValue::Function(FunctionId(fid as u32)),
            );
        }

        self.scripts.insert(name.to_string(), script_func_id);

        log::debug!(
            "GenovoVM: loaded script '{name}' (func_id={script_func_id}, \
             {} sub-functions, base_id={base_id})",
            self.function_map.len()
        );

        Ok(())
    }

    fn execute(
        &mut self,
        name: &str,
        ctx: &mut ScriptContext,
    ) -> EngineResult<ScriptValue> {
        let &func_id = self.scripts.get(name).ok_or_else(|| {
            EngineError::NotFound(format!("script '{name}' not loaded"))
        })?;

        // Import context globals into VM.
        self.vm.set_global("dt", ScriptValue::Float(ctx.delta_time));
        self.vm
            .set_global("total_time", ScriptValue::Float(ctx.total_time));
        for (k, v) in &ctx.globals {
            self.vm.set_global(k.clone(), v.clone());
        }

        // Reset VM state for a fresh execution.
        self.vm.stack.clear();
        self.vm.frames.clear();
        self.vm.ip = 0;

        // Don't pre-allocate locals. The compiler generates pushes for each
        // let declaration.

        self.vm.frames.push(CallFrame {
            function_id: func_id,
            ip: 0,
            stack_base: 0,
        });

        let result = self.vm.run().map_err(|e| {
            EngineError::Other(format!("script '{name}' runtime error: {e}"))
        })?;

        // Export modified globals back to context.
        for (k, v) in &self.vm.globals {
            ctx.globals.insert(k.clone(), v.clone());
        }

        Ok(result)
    }

    fn call_function(
        &mut self,
        function_name: &str,
        args: &[ScriptValue],
        _ctx: &mut ScriptContext,
    ) -> EngineResult<ScriptValue> {
        let &func_id = self.function_map.get(function_name).ok_or_else(|| {
            EngineError::NotFound(format!("function '{function_name}' not found"))
        })?;

        // Reset VM state.
        self.vm.stack.clear();
        self.vm.frames.clear();
        self.vm.ip = 0;

        self.vm.call(func_id, args).map_err(|e| {
            EngineError::Other(format!(
                "error calling '{function_name}': {e}"
            ))
        })
    }

    fn get_global(&self, name: &str) -> EngineResult<ScriptValue> {
        Ok(self
            .vm
            .globals
            .get(name)
            .cloned()
            .unwrap_or(ScriptValue::Nil))
    }

    fn set_global(&mut self, name: &str, value: ScriptValue) -> EngineResult<()> {
        self.vm.set_global(name, value);
        Ok(())
    }

    fn register_function(&mut self, name: &str, func: NativeFn) -> EngineResult<()> {
        self.vm.register_native(name, func);
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        self.vm.stack.capacity() * std::mem::size_of::<ScriptValue>()
            + self.vm.functions.len() * 256 // rough estimate
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ScriptValue tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_display() {
        assert_eq!(format!("{}", ScriptValue::Nil), "nil");
        assert_eq!(format!("{}", ScriptValue::Bool(true)), "true");
        assert_eq!(format!("{}", ScriptValue::Int(42)), "42");
        assert_eq!(format!("{}", ScriptValue::Float(3.14)), "3.14");
        assert_eq!(
            format!("{}", ScriptValue::from_string("hello")),
            "hello"
        );
        assert_eq!(
            format!("{}", ScriptValue::Vec3(1.0, 2.0, 3.0)),
            "vec3(1, 2, 3)"
        );
    }

    #[test]
    fn test_value_truthiness() {
        assert!(!ScriptValue::Nil.is_truthy());
        assert!(!ScriptValue::Bool(false).is_truthy());
        assert!(ScriptValue::Bool(true).is_truthy());
        assert!(ScriptValue::Int(0).is_truthy());
        assert!(ScriptValue::Int(1).is_truthy());
        assert!(ScriptValue::Float(0.0).is_truthy());
        assert!(ScriptValue::from_string("").is_truthy());
    }

    #[test]
    fn test_value_equality() {
        assert_eq!(ScriptValue::Nil, ScriptValue::Nil);
        assert_eq!(ScriptValue::Bool(true), ScriptValue::Bool(true));
        assert_eq!(ScriptValue::Int(42), ScriptValue::Int(42));
        assert_eq!(ScriptValue::Int(5), ScriptValue::Float(5.0));
        assert_ne!(ScriptValue::Int(1), ScriptValue::Bool(true));
        assert_ne!(ScriptValue::Nil, ScriptValue::Bool(false));
    }

    #[test]
    fn test_value_arithmetic() {
        let a = ScriptValue::Int(10);
        let b = ScriptValue::Int(3);

        assert_eq!(a.add(&b).unwrap(), ScriptValue::Int(13));
        assert_eq!(a.sub(&b).unwrap(), ScriptValue::Int(7));
        assert_eq!(a.mul(&b).unwrap(), ScriptValue::Int(30));
        assert_eq!(a.div(&b).unwrap(), ScriptValue::Int(3));
        assert_eq!(a.modulo(&b).unwrap(), ScriptValue::Int(1));
    }

    #[test]
    fn test_value_float_arithmetic() {
        let a = ScriptValue::Float(10.5);
        let b = ScriptValue::Float(3.0);

        assert_eq!(a.add(&b).unwrap(), ScriptValue::Float(13.5));
        assert_eq!(a.sub(&b).unwrap(), ScriptValue::Float(7.5));
    }

    #[test]
    fn test_value_mixed_arithmetic() {
        let a = ScriptValue::Int(10);
        let b = ScriptValue::Float(2.5);

        assert_eq!(a.add(&b).unwrap(), ScriptValue::Float(12.5));
        assert_eq!(a.mul(&b).unwrap(), ScriptValue::Float(25.0));
    }

    #[test]
    fn test_value_string_concat() {
        let a = ScriptValue::from_string("hello ");
        let b = ScriptValue::from_string("world");
        let result = a.add(&b).unwrap();
        assert_eq!(result, ScriptValue::from_string("hello world"));
    }

    #[test]
    fn test_value_vec3_arithmetic() {
        let a = ScriptValue::Vec3(1.0, 2.0, 3.0);
        let b = ScriptValue::Vec3(4.0, 5.0, 6.0);

        assert_eq!(a.add(&b).unwrap(), ScriptValue::Vec3(5.0, 7.0, 9.0));
        assert_eq!(a.sub(&b).unwrap(), ScriptValue::Vec3(-3.0, -3.0, -3.0));
    }

    #[test]
    fn test_value_negate() {
        assert_eq!(
            ScriptValue::Int(5).negate().unwrap(),
            ScriptValue::Int(-5)
        );
        assert_eq!(
            ScriptValue::Float(3.14).negate().unwrap(),
            ScriptValue::Float(-3.14)
        );
    }

    #[test]
    fn test_value_comparison() {
        let a = ScriptValue::Int(5);
        let b = ScriptValue::Int(10);

        assert!(a.less_than(&b).unwrap());
        assert!(!b.less_than(&a).unwrap());
        assert!(b.greater_than(&a).unwrap());
        assert!(!a.greater_than(&b).unwrap());
    }

    #[test]
    fn test_division_by_zero() {
        let a = ScriptValue::Int(10);
        let b = ScriptValue::Int(0);
        assert!(a.div(&b).is_err());
        assert!(a.modulo(&b).is_err());
    }

    // -----------------------------------------------------------------------
    // Chunk tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chunk_emit_and_read() {
        let mut chunk = Chunk::new();
        chunk.emit_constant(ScriptValue::Int(42), 1);
        chunk.emit_constant(ScriptValue::Int(8), 1);
        chunk.emit_op(OpCode::Add, 1);
        chunk.emit_op(OpCode::Halt, 1);

        assert_eq!(chunk.constants.len(), 2);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn test_chunk_jump_patching() {
        let mut chunk = Chunk::new();
        let offset = chunk.emit_jump(OpCode::JumpIfFalse, 1);
        chunk.emit_op(OpCode::Pop, 2);
        chunk.emit_op(OpCode::Pop, 3);
        chunk.patch_jump(offset).unwrap();

        let patched = chunk.read_u16(offset);
        assert_eq!(patched, 2); // skip 2 bytes (two Pop ops)
    }

    #[test]
    fn test_chunk_disassemble() {
        let mut chunk = Chunk::new();
        chunk.emit_constant(ScriptValue::Int(42), 1);
        chunk.emit_op(OpCode::Print, 1);
        chunk.emit_op(OpCode::Halt, 1);
        let dis = chunk.disassemble("test");
        assert!(dis.contains("Push"));
        assert!(dis.contains("Print"));
        assert!(dis.contains("Halt"));
    }

    // -----------------------------------------------------------------------
    // VM tests (hand-compiled bytecode)
    // -----------------------------------------------------------------------

    #[test]
    fn test_vm_simple_add() {
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_constant(ScriptValue::Int(10), 1);
        func.chunk.emit_constant(ScriptValue::Int(20), 1);
        func.chunk.emit_op(OpCode::Add, 1);
        func.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(30));
    }

    #[test]
    fn test_vm_arithmetic_chain() {
        // (3 + 4) * 2 - 1 = 13
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_constant(ScriptValue::Int(3), 1);
        func.chunk.emit_constant(ScriptValue::Int(4), 1);
        func.chunk.emit_op(OpCode::Add, 1);
        func.chunk.emit_constant(ScriptValue::Int(2), 1);
        func.chunk.emit_op(OpCode::Mul, 1);
        func.chunk.emit_constant(ScriptValue::Int(1), 1);
        func.chunk.emit_op(OpCode::Sub, 1);
        func.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(13));
    }

    #[test]
    fn test_vm_comparison() {
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_constant(ScriptValue::Int(5), 1);
        func.chunk.emit_constant(ScriptValue::Int(10), 1);
        func.chunk.emit_op(OpCode::Lt, 1);
        func.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Bool(true));
    }

    #[test]
    fn test_vm_globals() {
        let mut func = ScriptFunction::new("main", 0);
        // Set global "x" = 42
        func.chunk.emit_constant(ScriptValue::Int(42), 1);
        let name_idx = func
            .chunk
            .add_constant(ScriptValue::from_string("x"));
        func.chunk
            .emit_op_u16(OpCode::SetGlobal, name_idx, 1);
        func.chunk.emit_op(OpCode::Pop, 1);
        // Get global "x"
        let name_idx2 = func
            .chunk
            .add_constant(ScriptValue::from_string("x"));
        func.chunk
            .emit_op_u16(OpCode::GetGlobal, name_idx2, 2);
        func.chunk.emit_op(OpCode::Return, 2);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(42));
    }

    #[test]
    fn test_vm_locals() {
        // Emulate what the compiler does: `let` pushes a value onto
        // the stack and the stack position becomes the local slot.
        let mut func = ScriptFunction::new("main", 0);

        // local[0] = 10  (push 10 — it occupies stack slot 0)
        func.chunk.emit_constant(ScriptValue::Int(10), 1);
        // local[1] = 20  (push 20 — it occupies stack slot 1)
        func.chunk.emit_constant(ScriptValue::Int(20), 2);
        // push local[0] + local[1]
        func.chunk.emit_op_u16(OpCode::GetLocal, 0, 3);
        func.chunk.emit_op_u16(OpCode::GetLocal, 1, 3);
        func.chunk.emit_op(OpCode::Add, 3);
        func.chunk.emit_op(OpCode::Return, 3);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(30));
    }

    #[test]
    fn test_vm_conditional_jump() {
        // if (true) push 42 else push 99
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_op(OpCode::PushTrue, 1);
        let else_jump = func.chunk.emit_jump(OpCode::JumpIfFalse, 1);
        func.chunk.emit_constant(ScriptValue::Int(42), 2);
        let end_jump = func.chunk.emit_jump(OpCode::Jump, 2);
        func.chunk.patch_jump(else_jump).unwrap();
        func.chunk.emit_constant(ScriptValue::Int(99), 4);
        func.chunk.patch_jump(end_jump).unwrap();
        func.chunk.emit_op(OpCode::Return, 5);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(42));
    }

    #[test]
    fn test_vm_loop() {
        // sum = 0; i = 0; while (i < 5) { sum = sum + i; i = i + 1; } return sum
        let mut func = ScriptFunction::new("main", 0);

        // Push locals onto the stack (the way the compiler does it).
        // slot 0 = sum = 0 (pushed first)
        func.chunk.emit_constant(ScriptValue::Int(0), 1);
        // slot 1 = i = 0 (pushed second)
        func.chunk.emit_constant(ScriptValue::Int(0), 2);

        let loop_start = func.chunk.len();

        // condition: i < 5
        func.chunk.emit_op_u16(OpCode::GetLocal, 1, 3);
        func.chunk.emit_constant(ScriptValue::Int(5), 3);
        func.chunk.emit_op(OpCode::Lt, 3);
        let exit_jump = func.chunk.emit_jump(OpCode::JumpIfFalse, 3);

        // body: sum = sum + i
        func.chunk.emit_op_u16(OpCode::GetLocal, 0, 4);
        func.chunk.emit_op_u16(OpCode::GetLocal, 1, 4);
        func.chunk.emit_op(OpCode::Add, 4);
        func.chunk.emit_op_u16(OpCode::SetLocal, 0, 4);
        func.chunk.emit_op(OpCode::Pop, 4);

        // i = i + 1
        func.chunk.emit_op_u16(OpCode::GetLocal, 1, 5);
        func.chunk.emit_constant(ScriptValue::Int(1), 5);
        func.chunk.emit_op(OpCode::Add, 5);
        func.chunk.emit_op_u16(OpCode::SetLocal, 1, 5);
        func.chunk.emit_op(OpCode::Pop, 5);

        func.chunk.emit_loop(loop_start, 6).unwrap();
        func.chunk.patch_jump(exit_jump).unwrap();

        // return sum
        func.chunk.emit_op_u16(OpCode::GetLocal, 0, 7);
        func.chunk.emit_op(OpCode::Return, 7);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        // 0 + 1 + 2 + 3 + 4 = 10
        assert_eq!(result, ScriptValue::Int(10));
    }

    #[test]
    fn test_vm_print() {
        let mut func = ScriptFunction::new("main", 0);
        func.chunk
            .emit_constant(ScriptValue::from_string("hello world"), 1);
        func.chunk.emit_op(OpCode::Print, 1);
        func.chunk.emit_op(OpCode::Halt, 2);

        let mut vm = VM::new();
        vm.load_function(func);
        vm.run_script().unwrap();
        assert_eq!(vm.output(), &["hello world"]);
    }

    #[test]
    fn test_vm_native_function() {
        let mut vm = VM::new();
        vm.register_native(
            "add_native",
            Box::new(|args| {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(ScriptValue::Int(a + b))
            }),
        );

        let mut func = ScriptFunction::new("main", 0);
        // Push arguments.
        func.chunk.emit_constant(ScriptValue::Int(100), 1);
        func.chunk.emit_constant(ScriptValue::Int(200), 1);
        // CallNative: name_idx, arg_count
        let name_idx = func
            .chunk
            .add_constant(ScriptValue::from_string("add_native"));
        func.chunk
            .emit_op_u16(OpCode::CallNative, name_idx, 1);
        // emit arg_count byte
        func.chunk.code.push(2u8);
        func.chunk.line_numbers.push(1);

        func.chunk.emit_op(OpCode::Return, 1);

        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(300));
    }

    #[test]
    fn test_vm_function_call() {
        // Define function "double" that takes 1 arg and returns arg * 2.
        let mut double_fn = ScriptFunction::new("double", 1);
        // slot 0 = fn ref (pushed by Call), slot 1 = param (pushed by caller)
        double_fn.chunk.emit_op_u16(OpCode::GetLocal, 1, 1);
        double_fn.chunk.emit_constant(ScriptValue::Int(2), 1);
        double_fn.chunk.emit_op(OpCode::Mul, 1);
        double_fn.chunk.emit_op(OpCode::Return, 1);

        // Main function: call double(21).
        let mut main_fn = ScriptFunction::new("main", 0);
        main_fn.local_count = 0;
        // Push the function reference.
        main_fn
            .chunk
            .emit_constant(ScriptValue::Function(FunctionId(1)), 1);
        // Push argument.
        main_fn
            .chunk
            .emit_constant(ScriptValue::Int(21), 1);
        // Call with 1 argument.
        main_fn.chunk.emit_op_u16(OpCode::Call, 1, 1);
        main_fn.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(main_fn); // index 0
        vm.load_function(double_fn); // index 1
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(42));
    }

    #[test]
    fn test_vm_array_operations() {
        let mut func = ScriptFunction::new("main", 0);
        // Create array [10, 20, 30]
        func.chunk.emit_constant(ScriptValue::Int(10), 1);
        func.chunk.emit_constant(ScriptValue::Int(20), 1);
        func.chunk.emit_constant(ScriptValue::Int(30), 1);
        func.chunk.emit_op_u16(OpCode::ArrayNew, 3, 1);
        // Dup array, get element at index 1
        func.chunk.emit_op(OpCode::Dup, 2);
        func.chunk.emit_constant(ScriptValue::Int(1), 2);
        func.chunk.emit_op(OpCode::ArrayGet, 2);
        func.chunk.emit_op(OpCode::Return, 2);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(20));
    }

    #[test]
    fn test_vm_not_logic() {
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_op(OpCode::PushFalse, 1);
        func.chunk.emit_op(OpCode::Not, 1);
        func.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Bool(true));
    }

    #[test]
    fn test_vm_dup_swap() {
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_constant(ScriptValue::Int(1), 1);
        func.chunk.emit_constant(ScriptValue::Int(2), 1);
        func.chunk.emit_op(OpCode::Swap, 1);
        // After swap: stack = [2, 1]. Pop removes top (1), leaving 2.
        func.chunk.emit_op(OpCode::Pop, 1);
        func.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(2));
    }

    #[test]
    fn test_vm_spawn_entity() {
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_op(OpCode::SpawnEntity, 1);
        func.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Entity(1));
    }

    #[test]
    fn test_vm_stack_overflow() {
        let mut func = ScriptFunction::new("main", 0);
        for _ in 0..300 {
            func.chunk.emit_constant(ScriptValue::Int(1), 1);
        }
        func.chunk.emit_op(OpCode::Halt, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script();
        assert!(result.is_err());
    }

    #[test]
    fn test_vm_negation() {
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_constant(ScriptValue::Int(42), 1);
        func.chunk.emit_op(OpCode::Neg, 1);
        func.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(-42));
    }

    #[test]
    fn test_vm_modulo() {
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_constant(ScriptValue::Int(17), 1);
        func.chunk.emit_constant(ScriptValue::Int(5), 1);
        func.chunk.emit_op(OpCode::Mod, 1);
        func.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(2));
    }

    #[test]
    fn test_vm_array_len() {
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_constant(ScriptValue::Int(1), 1);
        func.chunk.emit_constant(ScriptValue::Int(2), 1);
        func.chunk.emit_constant(ScriptValue::Int(3), 1);
        func.chunk.emit_op_u16(OpCode::ArrayNew, 3, 1);
        func.chunk.emit_op(OpCode::ArrayLen, 1);
        func.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(3));
    }

    #[test]
    fn test_vm_string_len() {
        let mut func = ScriptFunction::new("main", 0);
        func.chunk.emit_constant(ScriptValue::from_string("hello"), 1);
        func.chunk.emit_op(OpCode::ArrayLen, 1);
        func.chunk.emit_op(OpCode::Return, 1);

        let mut vm = VM::new();
        vm.load_function(func);
        let result = vm.run_script().unwrap();
        assert_eq!(result, ScriptValue::Int(5));
    }

    #[test]
    fn test_genovo_vm_trait() {
        let mut gvm = GenovoVM::new();
        assert_eq!(gvm.language_name(), "Genovo");
        gvm.set_global("test", ScriptValue::Int(42)).unwrap();
        let val = gvm.get_global("test").unwrap();
        assert_eq!(val, ScriptValue::Int(42));
    }
}
