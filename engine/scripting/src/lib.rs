//! Genovo Engine - Scripting Module
//!
//! Provides a complete custom scripting runtime with a stack-based bytecode VM,
//! a lexer/parser/compiler pipeline, and engine binding infrastructure.
//!
//! # Architecture
//!
//! ```text
//! Source Text
//!     |
//!     v
//! [Lexer] --> Token stream
//!     |
//!     v
//! [Parser + Compiler] --> Bytecode (Chunk)
//!     |
//!     v
//! [VM] --> Executes bytecodes on a value stack
//! ```
//!
//! The scripting language supports:
//! - Dynamic typing with nil, bool, int, float, string, vec3, entity, array, map
//! - Arithmetic, comparison, and logic operators
//! - Variables (let bindings, assignment)
//! - Control flow (if/else, while, for)
//! - Functions with parameters and return values
//! - Native (Rust) function interop
//! - Print statements for debugging
//!
//! # Example
//!
//! ```ignore
//! use genovo_scripting::prelude::*;
//!
//! let mut system = ScriptSystem::new();
//! system.load_script("game", r#"
//!     let speed = 5.0
//!     fn update(dt) {
//!         print(speed * dt)
//!     }
//!     update(0.016)
//! "#).unwrap();
//! system.execute_script("game", 0.016, 0.0).unwrap();
//! ```

pub mod bindings;
pub mod coroutines;
pub mod debugger;
pub mod ffi_bridge;
pub mod gc;
pub mod module_system;
pub mod optimizer;
pub mod stdlib;
pub mod type_system;
pub mod vm;

pub use bindings::{BindingRegistry, ScriptBindable, ScriptComponent, ScriptSystem};
pub use coroutines::{
    Coroutine, CoroutineContext, CoroutineError, CoroutineEvent, CoroutineId,
    CoroutineManager, CoroutineResult, CoroutineState, CoroutineValue, WaitFor,
};
pub use debugger::{
    Breakpoint, BreakpointCondition, DebugEvent, ExecutionState, PauseReason,
    ScriptDebugger, StackFrame, WatchExpression,
};
pub use module_system::{
    CompiledChunk, ExportedValue, ImportStatement, ModuleError, ModuleRegistry,
    ModuleResolver, ModuleValue, ScriptModule,
};
pub use optimizer::{BytecodeOptimizer, OptLevel, OptimizationStats};
pub use stdlib::{get_stdlib, stdlib_function_names};
pub use vm::{
    Chunk, FunctionId, GenovoVM, NativeFn, OpCode, ScriptContext, ScriptError,
    ScriptFunction, ScriptValue, ScriptVM, VM,
};
pub use vm::compiler::Compiler;
pub use type_system::{
    BinaryOp, EnumDef, EnumPayload, EnumVariant, FunctionParam, FunctionType,
    SourceLocation, StructDef, StructField, Type, TypeChecker, TypeError,
    TypeErrorKind, UnaryOp, is_assignable, common_type,
};
pub use gc::{
    ClosureData, CollectionType, GarbageCollector, GcCollectionStats, GcColor,
    GcConfig, GcHandle, GcObjectData, GcStats, GcValue, Generation,
    NativeObject, StructInstance,
};
pub use ffi_bridge::{
    ArgDescriptor, AsyncCallId, AsyncCallStatus, FfiBridge, FfiBridgeStats,
    FfiError, FfiResult, FfiValue, FunctionDescriptor, NativeHandle,
    PendingCallback, ScriptCallback,
};

/// Convenience prelude for common imports.
pub mod prelude {
    pub use crate::bindings::{BindingRegistry, ScriptBindable, ScriptComponent, ScriptSystem};
    pub use crate::coroutines::{
        Coroutine, CoroutineId, CoroutineManager, CoroutineState, CoroutineValue, WaitFor,
    };
    pub use crate::debugger::{ScriptDebugger, StackFrame, BreakpointCondition};
    pub use crate::ffi_bridge::{FfiBridge, FfiValue, FunctionDescriptor};
    pub use crate::gc::{GarbageCollector, GcConfig, GcHandle, GcObjectData, GcValue};
    pub use crate::module_system::{ModuleRegistry, ScriptModule};
    pub use crate::optimizer::{BytecodeOptimizer, OptLevel};
    pub use crate::stdlib::get_stdlib;
    pub use crate::type_system::{Type, TypeChecker, TypeError};
    pub use crate::vm::{
        GenovoVM, NativeFn, ScriptContext, ScriptError, ScriptFunction, ScriptValue,
        ScriptVM, VM,
    };
    pub use crate::vm::compiler::Compiler;
}
