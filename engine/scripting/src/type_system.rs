//! Static type system for the Genovo scripting language.
//!
//! Provides compile-time type checking and inference for scripts before they
//! are executed by the VM. This module enables optional gradual typing — scripts
//! can start fully dynamic and progressively add type annotations for improved
//! safety and performance.
//!
//! # Features
//!
//! - Type checking pass over the AST
//! - Type inference for variables and expressions
//! - Typed variable declarations (`let x: Int = 5`)
//! - Function signatures with parameter types and return type
//! - Struct definitions with named fields
//! - Enum definitions with variant payloads
//! - Array types (`[Int]`, `[String]`)
//! - Optional types (`Int?`, `String?`)
//! - Union types (`Int | String`)
//! - Type errors with source location
//! - Type compatibility and coercion rules
//! - Generic type parameters (basic)
//!
//! # Example
//!
//! ```ignore
//! let mut checker = TypeChecker::new();
//! checker.define_struct("Vector3", &[
//!     ("x", Type::Float),
//!     ("y", Type::Float),
//!     ("z", Type::Float),
//! ]);
//!
//! let errors = checker.check_program(&ast);
//! for err in &errors {
//!     eprintln!("{}", err);
//! }
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Source Location
// ---------------------------------------------------------------------------

/// A location in source code for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SourceLocation {
    /// Line number (1-based).
    pub line: u32,
    /// Column number (1-based).
    pub column: u32,
    /// Byte offset from start of source.
    pub offset: u32,
}

impl SourceLocation {
    /// Create a new source location.
    pub fn new(line: u32, column: u32, offset: u32) -> Self {
        Self { line, column, offset }
    }

    /// A sentinel "unknown" location.
    pub fn unknown() -> Self {
        Self { line: 0, column: 0, offset: 0 }
    }

    /// Returns `true` if this is an unknown/unset location.
    pub fn is_unknown(&self) -> bool {
        self.line == 0 && self.column == 0
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unknown() {
            write!(f, "<unknown>")
        } else {
            write!(f, "{}:{}", self.line, self.column)
        }
    }
}

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

/// Represents a type in the scripting language's type system.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// No type / void return.
    Void,
    /// The `nil` type (unit type).
    Nil,
    /// Boolean type.
    Bool,
    /// 64-bit signed integer.
    Int,
    /// 64-bit floating point.
    Float,
    /// String type.
    String,
    /// 3-component vector.
    Vec3,
    /// Entity handle.
    Entity,
    /// Array of a specific element type.
    Array(Box<Type>),
    /// Map from String keys to a value type.
    Map(Box<Type>),
    /// Optional type (can be nil or the inner type).
    Optional(Box<Type>),
    /// Union of multiple types.
    Union(Vec<Type>),
    /// A reference to a user-defined struct.
    Struct(String),
    /// A reference to a user-defined enum.
    Enum(String),
    /// A function type: parameter types and return type.
    Function(FunctionType),
    /// A generic type parameter (e.g. `T`).
    Generic(String),
    /// A type that has not been inferred yet.
    Inferred,
    /// A type that could not be determined (type error occurred).
    Error,
    /// Dynamic / any type (opt out of type checking).
    Any,
}

impl Type {
    /// Returns `true` if this is a numeric type (Int or Float).
    pub fn is_numeric(&self) -> bool {
        matches!(self, Type::Int | Type::Float)
    }

    /// Returns `true` if this is a primitive type.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            Type::Bool | Type::Int | Type::Float | Type::String | Type::Vec3 | Type::Nil
        )
    }

    /// Returns `true` if this type can be nil.
    pub fn is_nullable(&self) -> bool {
        matches!(self, Type::Optional(_) | Type::Nil | Type::Any)
    }

    /// Returns `true` if this is a composite type (struct, enum, array, map).
    pub fn is_composite(&self) -> bool {
        matches!(
            self,
            Type::Array(_) | Type::Map(_) | Type::Struct(_) | Type::Enum(_)
        )
    }

    /// Returns `true` if this is the error sentinel type.
    pub fn is_error(&self) -> bool {
        matches!(self, Type::Error)
    }

    /// Wrap this type in an Optional.
    pub fn optional(self) -> Type {
        if matches!(self, Type::Optional(_)) {
            self // Already optional.
        } else {
            Type::Optional(Box::new(self))
        }
    }

    /// Create an array type with the given element type.
    pub fn array_of(element: Type) -> Type {
        Type::Array(Box::new(element))
    }

    /// Create a map type with the given value type (keys are always String).
    pub fn map_of(value: Type) -> Type {
        Type::Map(Box::new(value))
    }

    /// Create a union type.
    pub fn union(types: Vec<Type>) -> Type {
        if types.len() == 1 {
            return types.into_iter().next().unwrap();
        }
        // Flatten nested unions.
        let mut flat = Vec::new();
        for t in types {
            match t {
                Type::Union(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        // Deduplicate.
        flat.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        flat.dedup();
        if flat.len() == 1 {
            flat.into_iter().next().unwrap()
        } else {
            Type::Union(flat)
        }
    }

    /// Returns a human-readable type name.
    pub fn display_name(&self) -> String {
        match self {
            Type::Void => "Void".to_string(),
            Type::Nil => "Nil".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::Int => "Int".to_string(),
            Type::Float => "Float".to_string(),
            Type::String => "String".to_string(),
            Type::Vec3 => "Vec3".to_string(),
            Type::Entity => "Entity".to_string(),
            Type::Array(elem) => format!("[{}]", elem.display_name()),
            Type::Map(val) => format!("Map<{}>", val.display_name()),
            Type::Optional(inner) => format!("{}?", inner.display_name()),
            Type::Union(types) => types
                .iter()
                .map(|t| t.display_name())
                .collect::<Vec<_>>()
                .join(" | "),
            Type::Struct(name) => name.clone(),
            Type::Enum(name) => name.clone(),
            Type::Function(ft) => ft.display_name(),
            Type::Generic(name) => name.clone(),
            Type::Inferred => "<inferred>".to_string(),
            Type::Error => "<error>".to_string(),
            Type::Any => "Any".to_string(),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ---------------------------------------------------------------------------
// Function type
// ---------------------------------------------------------------------------

/// A function type with parameter types and return type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType {
    /// Parameter types (in order).
    pub params: Vec<FunctionParam>,
    /// Return type.
    pub return_type: Box<Type>,
    /// Whether this function is variadic (last param accepts multiple values).
    pub variadic: bool,
}

impl FunctionType {
    /// Create a new function type.
    pub fn new(params: Vec<FunctionParam>, return_type: Type) -> Self {
        Self {
            params,
            return_type: Box::new(return_type),
            variadic: false,
        }
    }

    /// Create a variadic function type.
    pub fn variadic(params: Vec<FunctionParam>, return_type: Type) -> Self {
        Self {
            params,
            return_type: Box::new(return_type),
            variadic: true,
        }
    }

    /// Returns a display name for this function type.
    pub fn display_name(&self) -> String {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                if let Some(ref name) = p.name {
                    format!("{}: {}", name, p.ty.display_name())
                } else {
                    p.ty.display_name()
                }
            })
            .collect();
        let variadic = if self.variadic { "..." } else { "" };
        format!(
            "fn({}{}) -> {}",
            params.join(", "),
            variadic,
            self.return_type.display_name()
        )
    }
}

/// A parameter in a function type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionParam {
    /// Parameter name (optional in type declarations).
    pub name: Option<String>,
    /// Parameter type.
    pub ty: Type,
    /// Whether this parameter has a default value.
    pub has_default: bool,
}

impl FunctionParam {
    /// Create a named parameter.
    pub fn named(name: &str, ty: Type) -> Self {
        Self {
            name: Some(name.to_string()),
            ty,
            has_default: false,
        }
    }

    /// Create an unnamed parameter.
    pub fn unnamed(ty: Type) -> Self {
        Self {
            name: None,
            ty,
            has_default: false,
        }
    }

    /// Create a parameter with a default value.
    pub fn with_default(name: &str, ty: Type) -> Self {
        Self {
            name: Some(name.to_string()),
            ty,
            has_default: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Struct definition
// ---------------------------------------------------------------------------

/// A user-defined struct type.
#[derive(Debug, Clone)]
pub struct StructDef {
    /// Struct name.
    pub name: String,
    /// Fields in definition order.
    pub fields: Vec<StructField>,
    /// Generic type parameters (e.g. `<T, U>`).
    pub generics: Vec<String>,
    /// Where this struct was defined.
    pub location: SourceLocation,
    /// Documentation comment.
    pub doc: Option<String>,
}

impl StructDef {
    /// Create a new struct definition.
    pub fn new(name: &str, fields: Vec<StructField>) -> Self {
        Self {
            name: name.to_string(),
            fields,
            generics: Vec::new(),
            location: SourceLocation::unknown(),
            doc: None,
        }
    }

    /// Look up a field by name.
    pub fn field(&self, name: &str) -> Option<&StructField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Returns `true` if this struct has a field with the given name.
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.iter().any(|f| f.name == name)
    }

    /// Returns the number of fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Returns `true` if this struct has generic type parameters.
    pub fn is_generic(&self) -> bool {
        !self.generics.is_empty()
    }
}

/// A field in a struct definition.
#[derive(Debug, Clone)]
pub struct StructField {
    /// Field name.
    pub name: String,
    /// Field type.
    pub ty: Type,
    /// Whether this field has a default value.
    pub has_default: bool,
    /// Whether this field is mutable.
    pub mutable: bool,
    /// Documentation comment.
    pub doc: Option<String>,
}

impl StructField {
    /// Create a new struct field.
    pub fn new(name: &str, ty: Type) -> Self {
        Self {
            name: name.to_string(),
            ty,
            has_default: false,
            mutable: true,
            doc: None,
        }
    }

    /// Create an immutable field.
    pub fn immutable(name: &str, ty: Type) -> Self {
        Self {
            name: name.to_string(),
            ty,
            has_default: false,
            mutable: false,
            doc: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Enum definition
// ---------------------------------------------------------------------------

/// A user-defined enum type.
#[derive(Debug, Clone)]
pub struct EnumDef {
    /// Enum name.
    pub name: String,
    /// Variants.
    pub variants: Vec<EnumVariant>,
    /// Generic type parameters.
    pub generics: Vec<String>,
    /// Where this enum was defined.
    pub location: SourceLocation,
    /// Documentation comment.
    pub doc: Option<String>,
}

impl EnumDef {
    /// Create a new enum definition.
    pub fn new(name: &str, variants: Vec<EnumVariant>) -> Self {
        Self {
            name: name.to_string(),
            variants,
            generics: Vec::new(),
            location: SourceLocation::unknown(),
            doc: None,
        }
    }

    /// Look up a variant by name.
    pub fn variant(&self, name: &str) -> Option<&EnumVariant> {
        self.variants.iter().find(|v| v.name == name)
    }

    /// Returns `true` if this enum has a variant with the given name.
    pub fn has_variant(&self, name: &str) -> bool {
        self.variants.iter().any(|v| v.name == name)
    }

    /// Returns the number of variants.
    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }
}

/// A variant of an enum type.
#[derive(Debug, Clone)]
pub struct EnumVariant {
    /// Variant name.
    pub name: String,
    /// Payload type (if any). None means a unit variant.
    pub payload: Option<EnumPayload>,
    /// Documentation comment.
    pub doc: Option<String>,
}

impl EnumVariant {
    /// Create a unit variant (no payload).
    pub fn unit(name: &str) -> Self {
        Self {
            name: name.to_string(),
            payload: None,
            doc: None,
        }
    }

    /// Create a tuple variant (positional fields).
    pub fn tuple(name: &str, types: Vec<Type>) -> Self {
        Self {
            name: name.to_string(),
            payload: Some(EnumPayload::Tuple(types)),
            doc: None,
        }
    }

    /// Create a struct variant (named fields).
    pub fn record(name: &str, fields: Vec<(String, Type)>) -> Self {
        Self {
            name: name.to_string(),
            payload: Some(EnumPayload::Struct(fields)),
            doc: None,
        }
    }

    /// Returns `true` if this variant has a payload.
    pub fn has_payload(&self) -> bool {
        self.payload.is_some()
    }
}

/// The payload of an enum variant.
#[derive(Debug, Clone)]
pub enum EnumPayload {
    /// Positional fields (tuple variant).
    Tuple(Vec<Type>),
    /// Named fields (struct variant).
    Struct(Vec<(String, Type)>),
}

// ---------------------------------------------------------------------------
// Type errors
// ---------------------------------------------------------------------------

/// A type error discovered during type checking.
#[derive(Debug, Clone)]
pub struct TypeError {
    /// Error message.
    pub message: String,
    /// Where the error occurred.
    pub location: SourceLocation,
    /// The kind of type error.
    pub kind: TypeErrorKind,
    /// Optional expected type.
    pub expected: Option<Type>,
    /// Optional actual type found.
    pub actual: Option<Type>,
}

impl TypeError {
    /// Create a new type error.
    pub fn new(kind: TypeErrorKind, message: &str, location: SourceLocation) -> Self {
        Self {
            message: message.to_string(),
            location,
            kind,
            expected: None,
            actual: None,
        }
    }

    /// Create a type mismatch error.
    pub fn mismatch(expected: Type, actual: Type, location: SourceLocation) -> Self {
        Self {
            message: format!(
                "type mismatch: expected '{}', found '{}'",
                expected.display_name(),
                actual.display_name()
            ),
            location,
            kind: TypeErrorKind::TypeMismatch,
            expected: Some(expected),
            actual: Some(actual),
        }
    }

    /// Create an "undefined variable" error.
    pub fn undefined_variable(name: &str, location: SourceLocation) -> Self {
        Self {
            message: format!("undefined variable '{name}'"),
            location,
            kind: TypeErrorKind::UndefinedVariable,
            expected: None,
            actual: None,
        }
    }

    /// Create an "undefined type" error.
    pub fn undefined_type(name: &str, location: SourceLocation) -> Self {
        Self {
            message: format!("undefined type '{name}'"),
            location,
            kind: TypeErrorKind::UndefinedType,
            expected: None,
            actual: None,
        }
    }

    /// Create an "undefined field" error.
    pub fn undefined_field(struct_name: &str, field: &str, location: SourceLocation) -> Self {
        Self {
            message: format!("struct '{struct_name}' has no field '{field}'"),
            location,
            kind: TypeErrorKind::UndefinedField,
            expected: None,
            actual: None,
        }
    }

    /// Set the expected and actual types.
    pub fn with_types(mut self, expected: Type, actual: Type) -> Self {
        self.expected = Some(expected);
        self.actual = Some(actual);
        self
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] error: {}", self.location, self.message)
    }
}

/// Categories of type errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeErrorKind {
    /// Expected one type, got another.
    TypeMismatch,
    /// A variable was used before being defined.
    UndefinedVariable,
    /// A type name was not found.
    UndefinedType,
    /// A field was not found on a struct.
    UndefinedField,
    /// A function was called with the wrong number of arguments.
    ArgumentCountMismatch,
    /// A function was not found.
    UndefinedFunction,
    /// Attempted to call a non-function value.
    NotCallable,
    /// Attempted to index a non-indexable value.
    NotIndexable,
    /// Attempted field access on a non-struct value.
    NotAStruct,
    /// Binary operator applied to incompatible types.
    InvalidOperator,
    /// A return type does not match the function signature.
    ReturnTypeMismatch,
    /// A variable was assigned an incompatible type.
    AssignmentTypeMismatch,
    /// An enum variant was not found.
    UndefinedVariant,
    /// Duplicate definition.
    DuplicateDefinition,
    /// Generic type parameter error.
    GenericError,
    /// Cannot infer type.
    InferenceFailure,
}

impl fmt::Display for TypeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch => write!(f, "type mismatch"),
            Self::UndefinedVariable => write!(f, "undefined variable"),
            Self::UndefinedType => write!(f, "undefined type"),
            Self::UndefinedField => write!(f, "undefined field"),
            Self::ArgumentCountMismatch => write!(f, "argument count mismatch"),
            Self::UndefinedFunction => write!(f, "undefined function"),
            Self::NotCallable => write!(f, "not callable"),
            Self::NotIndexable => write!(f, "not indexable"),
            Self::NotAStruct => write!(f, "not a struct"),
            Self::InvalidOperator => write!(f, "invalid operator"),
            Self::ReturnTypeMismatch => write!(f, "return type mismatch"),
            Self::AssignmentTypeMismatch => write!(f, "assignment type mismatch"),
            Self::UndefinedVariant => write!(f, "undefined variant"),
            Self::DuplicateDefinition => write!(f, "duplicate definition"),
            Self::GenericError => write!(f, "generic type error"),
            Self::InferenceFailure => write!(f, "inference failure"),
        }
    }
}

// ---------------------------------------------------------------------------
// Type environment (scope)
// ---------------------------------------------------------------------------

/// A scope containing variable type bindings.
#[derive(Debug, Clone)]
struct TypeScope {
    /// Variable name -> type.
    variables: HashMap<String, Type>,
    /// Whether this is a function scope (affects return type checking).
    is_function_scope: bool,
    /// The expected return type for this function scope.
    return_type: Option<Type>,
}

impl TypeScope {
    fn new() -> Self {
        Self {
            variables: HashMap::new(),
            is_function_scope: false,
            return_type: None,
        }
    }

    fn function_scope(return_type: Type) -> Self {
        Self {
            variables: HashMap::new(),
            is_function_scope: true,
            return_type: Some(return_type),
        }
    }
}

// ---------------------------------------------------------------------------
// Type compatibility rules
// ---------------------------------------------------------------------------

/// Check if `source` type is assignable to `target` type.
pub fn is_assignable(target: &Type, source: &Type) -> bool {
    // Same type is always assignable.
    if target == source {
        return true;
    }

    // Any accepts everything.
    if matches!(target, Type::Any) || matches!(source, Type::Any) {
        return true;
    }

    // Error type is compatible with everything (to avoid cascading errors).
    if matches!(target, Type::Error) || matches!(source, Type::Error) {
        return true;
    }

    // Inferred type is compatible with everything.
    if matches!(target, Type::Inferred) || matches!(source, Type::Inferred) {
        return true;
    }

    // Nil is assignable to Optional types.
    if matches!(source, Type::Nil) && matches!(target, Type::Optional(_)) {
        return true;
    }

    // T is assignable to Optional<T>.
    if let Type::Optional(inner) = target {
        if is_assignable(inner, source) {
            return true;
        }
    }

    // Int is promotable to Float.
    if matches!(target, Type::Float) && matches!(source, Type::Int) {
        return true;
    }

    // A type is assignable to a Union if it's assignable to any member.
    if let Type::Union(members) = target {
        if members.iter().any(|m| is_assignable(m, source)) {
            return true;
        }
    }

    // A Union source is assignable if all members are assignable to the target.
    if let Type::Union(members) = source {
        if members.iter().all(|m| is_assignable(target, m)) {
            return true;
        }
    }

    // Array covariance.
    if let (Type::Array(target_elem), Type::Array(source_elem)) = (target, source) {
        return is_assignable(target_elem, source_elem);
    }

    // Map covariance.
    if let (Type::Map(target_val), Type::Map(source_val)) = (target, source) {
        return is_assignable(target_val, source_val);
    }

    false
}

/// Find the common supertype of two types.
pub fn common_type(a: &Type, b: &Type) -> Type {
    if a == b {
        return a.clone();
    }

    // Int + Float -> Float
    if (matches!(a, Type::Int) && matches!(b, Type::Float))
        || (matches!(a, Type::Float) && matches!(b, Type::Int))
    {
        return Type::Float;
    }

    // Optional widening.
    if let Type::Optional(inner_a) = a {
        if is_assignable(inner_a, b) {
            return a.clone();
        }
    }
    if let Type::Optional(inner_b) = b {
        if is_assignable(inner_b, a) {
            return b.clone();
        }
    }

    // Nil + T -> Optional<T>
    if matches!(a, Type::Nil) {
        return Type::Optional(Box::new(b.clone()));
    }
    if matches!(b, Type::Nil) {
        return Type::Optional(Box::new(a.clone()));
    }

    // Fall back to Union.
    Type::union(vec![a.clone(), b.clone()])
}

/// Determine the result type of a binary operation.
pub fn binary_op_result(op: BinaryOp, left: &Type, right: &Type) -> Option<Type> {
    match op {
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
            match (left, right) {
                (Type::Int, Type::Int) => Some(Type::Int),
                (Type::Float, Type::Float) => Some(Type::Float),
                (Type::Int, Type::Float) | (Type::Float, Type::Int) => Some(Type::Float),
                (Type::Vec3, Type::Vec3) => {
                    if matches!(op, BinaryOp::Add | BinaryOp::Sub) {
                        Some(Type::Vec3)
                    } else {
                        None
                    }
                }
                (Type::Vec3, Type::Float) | (Type::Float, Type::Vec3) => {
                    if matches!(op, BinaryOp::Mul | BinaryOp::Div) {
                        Some(Type::Vec3)
                    } else {
                        None
                    }
                }
                (Type::String, Type::String) if matches!(op, BinaryOp::Add) => {
                    Some(Type::String)
                }
                _ => None,
            }
        }
        BinaryOp::Eq | BinaryOp::Ne => Some(Type::Bool),
        BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
            if left.is_numeric() && right.is_numeric() {
                Some(Type::Bool)
            } else if matches!(left, Type::String) && matches!(right, Type::String) {
                Some(Type::Bool)
            } else {
                None
            }
        }
        BinaryOp::And | BinaryOp::Or => Some(Type::Bool),
    }
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
            Self::Mod => write!(f, "%"),
            Self::Eq => write!(f, "=="),
            Self::Ne => write!(f, "!="),
            Self::Lt => write!(f, "<"),
            Self::Le => write!(f, "<="),
            Self::Gt => write!(f, ">"),
            Self::Ge => write!(f, ">="),
            Self::And => write!(f, "&&"),
            Self::Or => write!(f, "||"),
        }
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Not,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Neg => write!(f, "-"),
            Self::Not => write!(f, "!"),
        }
    }
}

/// Determine the result type of a unary operation.
pub fn unary_op_result(op: UnaryOp, operand: &Type) -> Option<Type> {
    match op {
        UnaryOp::Neg => match operand {
            Type::Int => Some(Type::Int),
            Type::Float => Some(Type::Float),
            Type::Vec3 => Some(Type::Vec3),
            _ => None,
        },
        UnaryOp::Not => Some(Type::Bool),
    }
}

// ---------------------------------------------------------------------------
// Type Checker
// ---------------------------------------------------------------------------

/// The type checker for the scripting language.
pub struct TypeChecker {
    /// Struct definitions.
    structs: HashMap<String, StructDef>,
    /// Enum definitions.
    enums: HashMap<String, EnumDef>,
    /// Type aliases.
    aliases: HashMap<String, Type>,
    /// Function signatures (name -> type).
    functions: HashMap<String, FunctionType>,
    /// Scope stack.
    scopes: Vec<TypeScope>,
    /// Accumulated type errors.
    errors: Vec<TypeError>,
    /// Whether to treat type errors as warnings (lenient mode).
    lenient: bool,
    /// Generic type parameter bindings during instantiation.
    generic_bindings: HashMap<String, Type>,
}

impl TypeChecker {
    /// Create a new type checker.
    pub fn new() -> Self {
        let mut checker = Self {
            structs: HashMap::new(),
            enums: HashMap::new(),
            aliases: HashMap::new(),
            functions: HashMap::new(),
            scopes: vec![TypeScope::new()], // Global scope.
            errors: Vec::new(),
            lenient: false,
            generic_bindings: HashMap::new(),
        };
        checker.register_builtins();
        checker
    }

    /// Create a type checker in lenient mode (type errors are warnings).
    pub fn lenient() -> Self {
        let mut checker = Self::new();
        checker.lenient = true;
        checker
    }

    /// Register built-in types and functions.
    fn register_builtins(&mut self) {
        // Built-in functions.
        self.define_function(
            "print",
            FunctionType::variadic(
                vec![FunctionParam::unnamed(Type::Any)],
                Type::Void,
            ),
        );
        self.define_function(
            "len",
            FunctionType::new(
                vec![FunctionParam::unnamed(Type::Any)],
                Type::Int,
            ),
        );
        self.define_function(
            "str",
            FunctionType::new(
                vec![FunctionParam::unnamed(Type::Any)],
                Type::String,
            ),
        );
        self.define_function(
            "int",
            FunctionType::new(
                vec![FunctionParam::unnamed(Type::Any)],
                Type::Int,
            ),
        );
        self.define_function(
            "float",
            FunctionType::new(
                vec![FunctionParam::unnamed(Type::Any)],
                Type::Float,
            ),
        );
        self.define_function(
            "abs",
            FunctionType::new(
                vec![FunctionParam::unnamed(Type::Float)],
                Type::Float,
            ),
        );
        self.define_function(
            "sqrt",
            FunctionType::new(
                vec![FunctionParam::unnamed(Type::Float)],
                Type::Float,
            ),
        );
        self.define_function(
            "sin",
            FunctionType::new(
                vec![FunctionParam::unnamed(Type::Float)],
                Type::Float,
            ),
        );
        self.define_function(
            "cos",
            FunctionType::new(
                vec![FunctionParam::unnamed(Type::Float)],
                Type::Float,
            ),
        );
        self.define_function(
            "vec3",
            FunctionType::new(
                vec![
                    FunctionParam::named("x", Type::Float),
                    FunctionParam::named("y", Type::Float),
                    FunctionParam::named("z", Type::Float),
                ],
                Type::Vec3,
            ),
        );
    }

    // --- Definition registration ---

    /// Define a struct type.
    pub fn define_struct(&mut self, name: &str, fields: &[(&str, Type)]) {
        let struct_fields: Vec<StructField> = fields
            .iter()
            .map(|(n, t)| StructField::new(n, t.clone()))
            .collect();
        self.structs.insert(
            name.to_string(),
            StructDef::new(name, struct_fields),
        );
    }

    /// Define a struct with a full StructDef.
    pub fn define_struct_def(&mut self, def: StructDef) {
        self.structs.insert(def.name.clone(), def);
    }

    /// Define an enum type.
    pub fn define_enum(&mut self, name: &str, variants: Vec<EnumVariant>) {
        self.enums.insert(
            name.to_string(),
            EnumDef::new(name, variants),
        );
    }

    /// Define an enum with a full EnumDef.
    pub fn define_enum_def(&mut self, def: EnumDef) {
        self.enums.insert(def.name.clone(), def);
    }

    /// Define a function signature.
    pub fn define_function(&mut self, name: &str, sig: FunctionType) {
        self.functions.insert(name.to_string(), sig);
    }

    /// Define a type alias.
    pub fn define_alias(&mut self, name: &str, target: Type) {
        self.aliases.insert(name.to_string(), target);
    }

    // --- Scope management ---

    /// Push a new scope.
    pub fn push_scope(&mut self) {
        self.scopes.push(TypeScope::new());
    }

    /// Push a function scope with a return type.
    pub fn push_function_scope(&mut self, return_type: Type) {
        self.scopes.push(TypeScope::function_scope(return_type));
    }

    /// Pop the current scope.
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Define a variable in the current scope.
    pub fn define_variable(&mut self, name: &str, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.variables.insert(name.to_string(), ty);
        }
    }

    /// Look up a variable's type, searching from inner to outer scope.
    pub fn lookup_variable(&self, name: &str) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.variables.get(name) {
                return Some(ty);
            }
        }
        None
    }

    /// Look up the current function's return type.
    pub fn current_return_type(&self) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if scope.is_function_scope {
                return scope.return_type.as_ref();
            }
        }
        None
    }

    // --- Type resolution ---

    /// Resolve a type name to its definition.
    pub fn resolve_type(&self, name: &str) -> Option<Type> {
        // Check aliases first.
        if let Some(aliased) = self.aliases.get(name) {
            return Some(aliased.clone());
        }
        // Check structs.
        if self.structs.contains_key(name) {
            return Some(Type::Struct(name.to_string()));
        }
        // Check enums.
        if self.enums.contains_key(name) {
            return Some(Type::Enum(name.to_string()));
        }
        // Check generic bindings.
        if let Some(bound) = self.generic_bindings.get(name) {
            return Some(bound.clone());
        }
        // Built-in type names.
        match name {
            "Void" => Some(Type::Void),
            "Nil" => Some(Type::Nil),
            "Bool" => Some(Type::Bool),
            "Int" => Some(Type::Int),
            "Float" => Some(Type::Float),
            "String" => Some(Type::String),
            "Vec3" => Some(Type::Vec3),
            "Entity" => Some(Type::Entity),
            "Any" => Some(Type::Any),
            _ => None,
        }
    }

    /// Resolve a struct definition.
    pub fn resolve_struct(&self, name: &str) -> Option<&StructDef> {
        self.structs.get(name)
    }

    /// Resolve an enum definition.
    pub fn resolve_enum(&self, name: &str) -> Option<&EnumDef> {
        self.enums.get(name)
    }

    /// Resolve a function signature.
    pub fn resolve_function(&self, name: &str) -> Option<&FunctionType> {
        self.functions.get(name)
    }

    // --- Type checking ---

    /// Check a variable declaration and record it.
    pub fn check_variable_decl(
        &mut self,
        name: &str,
        declared_type: Option<&Type>,
        init_type: &Type,
        location: SourceLocation,
    ) -> Type {
        let resolved_type = if let Some(decl_ty) = declared_type {
            // Explicit type annotation.
            if !is_assignable(decl_ty, init_type) {
                self.report_error(TypeError::mismatch(
                    decl_ty.clone(),
                    init_type.clone(),
                    location,
                ));
                decl_ty.clone()
            } else {
                decl_ty.clone()
            }
        } else {
            // Infer type from initializer.
            if matches!(init_type, Type::Inferred) {
                self.report_error(TypeError::new(
                    TypeErrorKind::InferenceFailure,
                    &format!("cannot infer type for variable '{name}'"),
                    location,
                ));
                Type::Any
            } else {
                init_type.clone()
            }
        };

        self.define_variable(name, resolved_type.clone());
        resolved_type
    }

    /// Check an assignment to a variable.
    pub fn check_assignment(
        &mut self,
        name: &str,
        value_type: &Type,
        location: SourceLocation,
    ) -> Type {
        if let Some(var_type) = self.lookup_variable(name).cloned() {
            if !is_assignable(&var_type, value_type) {
                self.report_error(TypeError {
                    message: format!(
                        "cannot assign '{}' to variable '{}' of type '{}'",
                        value_type.display_name(),
                        name,
                        var_type.display_name()
                    ),
                    location,
                    kind: TypeErrorKind::AssignmentTypeMismatch,
                    expected: Some(var_type.clone()),
                    actual: Some(value_type.clone()),
                });
            }
            var_type
        } else {
            self.report_error(TypeError::undefined_variable(name, location));
            Type::Error
        }
    }

    /// Check a function call.
    pub fn check_function_call(
        &mut self,
        name: &str,
        arg_types: &[Type],
        location: SourceLocation,
    ) -> Type {
        if let Some(sig) = self.functions.get(name).cloned() {
            // Check argument count.
            let required = sig.params.iter().filter(|p| !p.has_default).count();
            let max_args = if sig.variadic {
                usize::MAX
            } else {
                sig.params.len()
            };

            if arg_types.len() < required || arg_types.len() > max_args {
                self.report_error(TypeError::new(
                    TypeErrorKind::ArgumentCountMismatch,
                    &format!(
                        "function '{}' expects {} argument(s), got {}",
                        name,
                        if sig.variadic {
                            format!("at least {required}")
                        } else {
                            format!("{}", sig.params.len())
                        },
                        arg_types.len()
                    ),
                    location,
                ));
                return *sig.return_type;
            }

            // Check argument types.
            for (i, (param, arg)) in sig.params.iter().zip(arg_types.iter()).enumerate() {
                if !is_assignable(&param.ty, arg) {
                    let param_name = param
                        .name
                        .as_deref()
                        .unwrap_or(&format!("argument {}", i + 1));
                    self.report_error(TypeError {
                        message: format!(
                            "argument '{}' of function '{}': expected '{}', got '{}'",
                            param_name,
                            name,
                            param.ty.display_name(),
                            arg.display_name()
                        ),
                        location,
                        kind: TypeErrorKind::TypeMismatch,
                        expected: Some(param.ty.clone()),
                        actual: Some(arg.clone()),
                    });
                }
            }

            *sig.return_type
        } else {
            self.report_error(TypeError::new(
                TypeErrorKind::UndefinedFunction,
                &format!("undefined function '{name}'"),
                location,
            ));
            Type::Error
        }
    }

    /// Check a field access on a struct.
    pub fn check_field_access(
        &mut self,
        struct_type: &Type,
        field_name: &str,
        location: SourceLocation,
    ) -> Type {
        match struct_type {
            Type::Struct(name) => {
                if let Some(def) = self.structs.get(name).cloned() {
                    if let Some(field) = def.field(field_name) {
                        field.ty.clone()
                    } else {
                        self.report_error(TypeError::undefined_field(name, field_name, location));
                        Type::Error
                    }
                } else {
                    self.report_error(TypeError::undefined_type(name, location));
                    Type::Error
                }
            }
            Type::Vec3 => {
                match field_name {
                    "x" | "y" | "z" => Type::Float,
                    _ => {
                        self.report_error(TypeError::new(
                            TypeErrorKind::UndefinedField,
                            &format!("Vec3 has no field '{field_name}'"),
                            location,
                        ));
                        Type::Error
                    }
                }
            }
            Type::Array(_) => {
                match field_name {
                    "length" | "len" => Type::Int,
                    _ => {
                        self.report_error(TypeError::new(
                            TypeErrorKind::UndefinedField,
                            &format!("Array has no field '{field_name}'"),
                            location,
                        ));
                        Type::Error
                    }
                }
            }
            Type::String => {
                match field_name {
                    "length" | "len" => Type::Int,
                    _ => {
                        self.report_error(TypeError::new(
                            TypeErrorKind::UndefinedField,
                            &format!("String has no field '{field_name}'"),
                            location,
                        ));
                        Type::Error
                    }
                }
            }
            Type::Any => Type::Any,
            Type::Error => Type::Error,
            _ => {
                self.report_error(TypeError::new(
                    TypeErrorKind::NotAStruct,
                    &format!(
                        "cannot access field '{field_name}' on type '{}'",
                        struct_type.display_name()
                    ),
                    location,
                ));
                Type::Error
            }
        }
    }

    /// Check an index access on an array or map.
    pub fn check_index_access(
        &mut self,
        container: &Type,
        index: &Type,
        location: SourceLocation,
    ) -> Type {
        match container {
            Type::Array(elem) => {
                if !is_assignable(&Type::Int, index) {
                    self.report_error(TypeError {
                        message: format!(
                            "array index must be Int, got '{}'",
                            index.display_name()
                        ),
                        location,
                        kind: TypeErrorKind::TypeMismatch,
                        expected: Some(Type::Int),
                        actual: Some(index.clone()),
                    });
                }
                *elem.clone()
            }
            Type::Map(val) => {
                if !is_assignable(&Type::String, index) {
                    self.report_error(TypeError {
                        message: format!(
                            "map key must be String, got '{}'",
                            index.display_name()
                        ),
                        location,
                        kind: TypeErrorKind::TypeMismatch,
                        expected: Some(Type::String),
                        actual: Some(index.clone()),
                    });
                }
                *val.clone()
            }
            Type::String => {
                if !is_assignable(&Type::Int, index) {
                    self.report_error(TypeError {
                        message: format!(
                            "string index must be Int, got '{}'",
                            index.display_name()
                        ),
                        location,
                        kind: TypeErrorKind::TypeMismatch,
                        expected: Some(Type::Int),
                        actual: Some(index.clone()),
                    });
                }
                Type::String
            }
            Type::Any => Type::Any,
            Type::Error => Type::Error,
            _ => {
                self.report_error(TypeError::new(
                    TypeErrorKind::NotIndexable,
                    &format!(
                        "type '{}' is not indexable",
                        container.display_name()
                    ),
                    location,
                ));
                Type::Error
            }
        }
    }

    /// Check a return statement against the function's return type.
    pub fn check_return(
        &mut self,
        value_type: &Type,
        location: SourceLocation,
    ) {
        if let Some(return_type) = self.current_return_type().cloned() {
            if !is_assignable(&return_type, value_type) {
                self.report_error(TypeError {
                    message: format!(
                        "return type mismatch: expected '{}', got '{}'",
                        return_type.display_name(),
                        value_type.display_name()
                    ),
                    location,
                    kind: TypeErrorKind::ReturnTypeMismatch,
                    expected: Some(return_type),
                    actual: Some(value_type.clone()),
                });
            }
        }
    }

    /// Check a binary operation and return the result type.
    pub fn check_binary_op(
        &mut self,
        op: BinaryOp,
        left: &Type,
        right: &Type,
        location: SourceLocation,
    ) -> Type {
        if matches!(left, Type::Any) || matches!(right, Type::Any) {
            return Type::Any;
        }
        if matches!(left, Type::Error) || matches!(right, Type::Error) {
            return Type::Error;
        }
        match binary_op_result(op, left, right) {
            Some(result) => result,
            None => {
                self.report_error(TypeError::new(
                    TypeErrorKind::InvalidOperator,
                    &format!(
                        "operator '{}' cannot be applied to '{}' and '{}'",
                        op,
                        left.display_name(),
                        right.display_name()
                    ),
                    location,
                ));
                Type::Error
            }
        }
    }

    /// Check a unary operation and return the result type.
    pub fn check_unary_op(
        &mut self,
        op: UnaryOp,
        operand: &Type,
        location: SourceLocation,
    ) -> Type {
        if matches!(operand, Type::Any) {
            return Type::Any;
        }
        if matches!(operand, Type::Error) {
            return Type::Error;
        }
        match unary_op_result(op, operand) {
            Some(result) => result,
            None => {
                self.report_error(TypeError::new(
                    TypeErrorKind::InvalidOperator,
                    &format!(
                        "operator '{}' cannot be applied to '{}'",
                        op,
                        operand.display_name()
                    ),
                    location,
                ));
                Type::Error
            }
        }
    }

    // --- Error handling ---

    /// Report a type error.
    fn report_error(&mut self, error: TypeError) {
        self.errors.push(error);
    }

    /// Returns all accumulated type errors.
    pub fn errors(&self) -> &[TypeError] {
        &self.errors
    }

    /// Returns `true` if any type errors were found.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Returns the number of errors.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Clear all errors.
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }

    /// Returns all registered struct definitions.
    pub fn struct_definitions(&self) -> &HashMap<String, StructDef> {
        &self.structs
    }

    /// Returns all registered enum definitions.
    pub fn enum_definitions(&self) -> &HashMap<String, EnumDef> {
        &self.enums
    }

    /// Returns all registered function signatures.
    pub fn function_signatures(&self) -> &HashMap<String, FunctionType> {
        &self.functions
    }
}

impl Default for TypeChecker {
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
    fn test_type_display() {
        assert_eq!(Type::Int.display_name(), "Int");
        assert_eq!(Type::array_of(Type::String).display_name(), "[String]");
        assert_eq!(Type::Float.optional().display_name(), "Float?");
        assert_eq!(
            Type::union(vec![Type::Int, Type::String]).display_name(),
            "Int | String"
        );
    }

    #[test]
    fn test_assignability_same_type() {
        assert!(is_assignable(&Type::Int, &Type::Int));
        assert!(is_assignable(&Type::String, &Type::String));
    }

    #[test]
    fn test_assignability_int_to_float() {
        assert!(is_assignable(&Type::Float, &Type::Int));
        assert!(!is_assignable(&Type::Int, &Type::Float));
    }

    #[test]
    fn test_assignability_nil_to_optional() {
        assert!(is_assignable(&Type::Optional(Box::new(Type::Int)), &Type::Nil));
        assert!(!is_assignable(&Type::Int, &Type::Nil));
    }

    #[test]
    fn test_assignability_to_optional() {
        assert!(is_assignable(
            &Type::Optional(Box::new(Type::Int)),
            &Type::Int
        ));
    }

    #[test]
    fn test_assignability_union() {
        let union_type = Type::union(vec![Type::Int, Type::String]);
        assert!(is_assignable(&union_type, &Type::Int));
        assert!(is_assignable(&union_type, &Type::String));
        assert!(!is_assignable(&union_type, &Type::Bool));
    }

    #[test]
    fn test_assignability_any() {
        assert!(is_assignable(&Type::Any, &Type::Int));
        assert!(is_assignable(&Type::Int, &Type::Any));
    }

    #[test]
    fn test_common_type_same() {
        assert_eq!(common_type(&Type::Int, &Type::Int), Type::Int);
    }

    #[test]
    fn test_common_type_int_float() {
        assert_eq!(common_type(&Type::Int, &Type::Float), Type::Float);
    }

    #[test]
    fn test_common_type_nil_creates_optional() {
        let result = common_type(&Type::Nil, &Type::Int);
        assert_eq!(result, Type::Optional(Box::new(Type::Int)));
    }

    #[test]
    fn test_binary_op_types() {
        assert_eq!(binary_op_result(BinaryOp::Add, &Type::Int, &Type::Int), Some(Type::Int));
        assert_eq!(
            binary_op_result(BinaryOp::Add, &Type::Int, &Type::Float),
            Some(Type::Float)
        );
        assert_eq!(
            binary_op_result(BinaryOp::Add, &Type::String, &Type::String),
            Some(Type::String)
        );
        assert_eq!(binary_op_result(BinaryOp::Eq, &Type::Int, &Type::Int), Some(Type::Bool));
        assert_eq!(binary_op_result(BinaryOp::Add, &Type::Bool, &Type::Int), None);
    }

    #[test]
    fn test_unary_op_types() {
        assert_eq!(unary_op_result(UnaryOp::Neg, &Type::Int), Some(Type::Int));
        assert_eq!(unary_op_result(UnaryOp::Neg, &Type::Float), Some(Type::Float));
        assert_eq!(unary_op_result(UnaryOp::Not, &Type::Bool), Some(Type::Bool));
        assert_eq!(unary_op_result(UnaryOp::Neg, &Type::String), None);
    }

    #[test]
    fn test_struct_def() {
        let def = StructDef::new("Player", vec![
            StructField::new("name", Type::String),
            StructField::new("health", Type::Int),
        ]);
        assert_eq!(def.field_count(), 2);
        assert!(def.has_field("name"));
        assert!(!def.has_field("mana"));
        assert_eq!(def.field("health").unwrap().ty, Type::Int);
    }

    #[test]
    fn test_enum_def() {
        let def = EnumDef::new("Direction", vec![
            EnumVariant::unit("North"),
            EnumVariant::unit("South"),
            EnumVariant::unit("East"),
            EnumVariant::unit("West"),
        ]);
        assert_eq!(def.variant_count(), 4);
        assert!(def.has_variant("North"));
        assert!(!def.has_variant("Up"));
    }

    #[test]
    fn test_enum_variant_with_payload() {
        let variant = EnumVariant::tuple("Some", vec![Type::Int]);
        assert!(variant.has_payload());

        let unit = EnumVariant::unit("None");
        assert!(!unit.has_payload());
    }

    #[test]
    fn test_type_checker_variable() {
        let mut checker = TypeChecker::new();
        let loc = SourceLocation::new(1, 1, 0);

        let ty = checker.check_variable_decl("x", Some(&Type::Int), &Type::Int, loc);
        assert_eq!(ty, Type::Int);
        assert!(!checker.has_errors());

        assert_eq!(checker.lookup_variable("x"), Some(&Type::Int));
    }

    #[test]
    fn test_type_checker_variable_mismatch() {
        let mut checker = TypeChecker::new();
        let loc = SourceLocation::new(1, 1, 0);

        checker.check_variable_decl("x", Some(&Type::Int), &Type::String, loc);
        assert!(checker.has_errors());
        assert_eq!(checker.errors()[0].kind, TypeErrorKind::TypeMismatch);
    }

    #[test]
    fn test_type_checker_infer() {
        let mut checker = TypeChecker::new();
        let loc = SourceLocation::new(1, 1, 0);

        let ty = checker.check_variable_decl("x", None, &Type::Float, loc);
        assert_eq!(ty, Type::Float);
        assert!(!checker.has_errors());
    }

    #[test]
    fn test_type_checker_assignment() {
        let mut checker = TypeChecker::new();
        let loc = SourceLocation::new(1, 1, 0);

        checker.define_variable("x", Type::Int);
        checker.check_assignment("x", &Type::Int, loc);
        assert!(!checker.has_errors());

        checker.check_assignment("x", &Type::String, loc);
        assert!(checker.has_errors());
    }

    #[test]
    fn test_type_checker_function_call() {
        let mut checker = TypeChecker::new();
        let loc = SourceLocation::new(1, 1, 0);

        let ret = checker.check_function_call("sqrt", &[Type::Float], loc);
        assert_eq!(ret, Type::Float);
        assert!(!checker.has_errors());
    }

    #[test]
    fn test_type_checker_function_call_wrong_args() {
        let mut checker = TypeChecker::new();
        let loc = SourceLocation::new(1, 1, 0);

        checker.check_function_call("sqrt", &[], loc);
        assert!(checker.has_errors());
    }

    #[test]
    fn test_type_checker_field_access() {
        let mut checker = TypeChecker::new();
        checker.define_struct("Player", &[("name", Type::String), ("hp", Type::Int)]);
        let loc = SourceLocation::new(1, 1, 0);

        let ty = checker.check_field_access(&Type::Struct("Player".to_string()), "name", loc);
        assert_eq!(ty, Type::String);
        assert!(!checker.has_errors());

        checker.check_field_access(&Type::Struct("Player".to_string()), "mana", loc);
        assert!(checker.has_errors());
    }

    #[test]
    fn test_type_checker_index_access() {
        let mut checker = TypeChecker::new();
        let loc = SourceLocation::new(1, 1, 0);

        let ty = checker.check_index_access(
            &Type::array_of(Type::Int),
            &Type::Int,
            loc,
        );
        assert_eq!(ty, Type::Int);
        assert!(!checker.has_errors());
    }

    #[test]
    fn test_type_checker_scopes() {
        let mut checker = TypeChecker::new();
        checker.define_variable("outer", Type::Int);
        checker.push_scope();
        checker.define_variable("inner", Type::String);

        assert_eq!(checker.lookup_variable("outer"), Some(&Type::Int));
        assert_eq!(checker.lookup_variable("inner"), Some(&Type::String));

        checker.pop_scope();
        assert_eq!(checker.lookup_variable("outer"), Some(&Type::Int));
        assert_eq!(checker.lookup_variable("inner"), None);
    }

    #[test]
    fn test_type_checker_binary_op() {
        let mut checker = TypeChecker::new();
        let loc = SourceLocation::new(1, 1, 0);

        let ty = checker.check_binary_op(BinaryOp::Add, &Type::Int, &Type::Int, loc);
        assert_eq!(ty, Type::Int);
        assert!(!checker.has_errors());

        let ty = checker.check_binary_op(BinaryOp::Add, &Type::Bool, &Type::Int, loc);
        assert_eq!(ty, Type::Error);
        assert!(checker.has_errors());
    }

    #[test]
    fn test_function_type_display() {
        let ft = FunctionType::new(
            vec![
                FunctionParam::named("x", Type::Float),
                FunctionParam::named("y", Type::Float),
            ],
            Type::Float,
        );
        assert_eq!(ft.display_name(), "fn(x: Float, y: Float) -> Float");
    }

    #[test]
    fn test_type_union_flatten() {
        let t = Type::union(vec![
            Type::Int,
            Type::union(vec![Type::String, Type::Bool]),
        ]);
        match t {
            Type::Union(members) => {
                assert_eq!(members.len(), 3);
            }
            _ => panic!("expected union type"),
        }
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new(10, 5, 200);
        assert_eq!(format!("{loc}"), "10:5");
        assert!(!loc.is_unknown());

        let unknown = SourceLocation::unknown();
        assert!(unknown.is_unknown());
    }

    #[test]
    fn test_type_error_display() {
        let err = TypeError::mismatch(
            Type::Int,
            Type::String,
            SourceLocation::new(5, 10, 0),
        );
        assert!(err.to_string().contains("type mismatch"));
        assert!(err.to_string().contains("5:10"));
    }
}
