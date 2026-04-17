//! Read-Eval-Print Loop (REPL) for the Genovo scripting runtime.
//!
//! Provides an interactive script console for debugging, expression evaluation,
//! variable inspection, command history, tab completion for globals/functions,
//! multi-line input support, error recovery, and output formatting.
//!
//! # Example
//!
//! ```ignore
//! let mut repl = Repl::new();
//! repl.register_global("player_health", ReplValue::Float(100.0));
//! repl.register_function("heal", vec!["amount"], |args| {
//!     Ok(ReplValue::String("healed!".to_string()))
//! });
//!
//! let result = repl.eval("player_health + 10");
//! assert_eq!(result.unwrap().to_string(), "110");
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ReplValue
// ---------------------------------------------------------------------------

/// A value in the REPL environment.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplValue {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<ReplValue>),
    Map(Vec<(String, ReplValue)>),
    Function { name: String, params: Vec<String> },
    Error(String),
}

impl ReplValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            ReplValue::Nil => "nil",
            ReplValue::Bool(_) => "bool",
            ReplValue::Int(_) => "int",
            ReplValue::Float(_) => "float",
            ReplValue::String(_) => "string",
            ReplValue::Array(_) => "array",
            ReplValue::Map(_) => "map",
            ReplValue::Function { .. } => "function",
            ReplValue::Error(_) => "error",
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            ReplValue::Nil => false,
            ReplValue::Bool(b) => *b,
            ReplValue::Int(i) => *i != 0,
            ReplValue::Float(f) => *f != 0.0,
            ReplValue::String(s) => !s.is_empty(),
            ReplValue::Array(a) => !a.is_empty(),
            ReplValue::Map(m) => !m.is_empty(),
            ReplValue::Function { .. } => true,
            ReplValue::Error(_) => false,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            ReplValue::Int(i) => Some(*i),
            ReplValue::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            ReplValue::Float(f) => Some(*f),
            ReplValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            ReplValue::String(s) => Some(s),
            _ => None,
        }
    }
}

impl fmt::Display for ReplValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplValue::Nil => write!(f, "nil"),
            ReplValue::Bool(b) => write!(f, "{}", b),
            ReplValue::Int(i) => write!(f, "{}", i),
            ReplValue::Float(v) => {
                if v.fract() == 0.0 {
                    write!(f, "{:.1}", v)
                } else {
                    write!(f, "{}", v)
                }
            }
            ReplValue::String(s) => write!(f, "\"{}\"", s),
            ReplValue::Array(a) => {
                write!(f, "[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            ReplValue::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            ReplValue::Function { name, params } => {
                write!(f, "fn {}({})", name, params.join(", "))
            }
            ReplValue::Error(msg) => write!(f, "Error: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// ReplError
// ---------------------------------------------------------------------------

/// Errors from the REPL.
#[derive(Debug, Clone)]
pub enum ReplError {
    /// Syntax error in the input.
    SyntaxError(String),
    /// Runtime error during evaluation.
    RuntimeError(String),
    /// Undefined variable or function.
    UndefinedName(String),
    /// Type mismatch.
    TypeError(String),
    /// Division by zero.
    DivisionByZero,
    /// Incomplete input (needs more lines).
    IncompleteInput,
    /// Unknown command.
    UnknownCommand(String),
}

impl fmt::Display for ReplError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplError::SyntaxError(msg) => write!(f, "Syntax error: {}", msg),
            ReplError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            ReplError::UndefinedName(name) => write!(f, "Undefined: '{}'", name),
            ReplError::TypeError(msg) => write!(f, "Type error: {}", msg),
            ReplError::DivisionByZero => write!(f, "Division by zero"),
            ReplError::IncompleteInput => write!(f, "Incomplete input"),
            ReplError::UnknownCommand(cmd) => write!(f, "Unknown command: {}", cmd),
        }
    }
}

pub type ReplResult<T> = Result<T, ReplError>;

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

/// Command history for the REPL.
#[derive(Debug, Clone)]
pub struct History {
    entries: VecDeque<HistoryEntry>,
    max_entries: usize,
    cursor: usize,
}

/// A single history entry.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub input: String,
    pub result: Option<String>,
    pub is_error: bool,
    pub timestamp: Instant,
    pub duration: Duration,
}

impl History {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries),
            max_entries,
            cursor: 0,
        }
    }

    /// Add an entry.
    pub fn push(&mut self, entry: HistoryEntry) {
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        self.cursor = self.entries.len();
    }

    /// Navigate backward in history.
    pub fn prev(&mut self) -> Option<&str> {
        if self.cursor > 0 {
            self.cursor -= 1;
            Some(&self.entries[self.cursor].input)
        } else {
            None
        }
    }

    /// Navigate forward in history.
    pub fn next(&mut self) -> Option<&str> {
        if self.cursor < self.entries.len().saturating_sub(1) {
            self.cursor += 1;
            Some(&self.entries[self.cursor].input)
        } else {
            self.cursor = self.entries.len();
            None
        }
    }

    /// Reset cursor to the end.
    pub fn reset_cursor(&mut self) {
        self.cursor = self.entries.len();
    }

    /// Get all entries.
    pub fn entries(&self) -> &VecDeque<HistoryEntry> {
        &self.entries
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear history.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.cursor = 0;
    }

    /// Search history for entries containing a substring.
    pub fn search(&self, query: &str) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.input.contains(query))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tab Completion
// ---------------------------------------------------------------------------

/// Tab completion provider.
pub struct CompletionProvider {
    /// Available completions keyed by prefix.
    globals: Vec<String>,
    functions: Vec<String>,
    keywords: Vec<String>,
    commands: Vec<String>,
}

impl CompletionProvider {
    pub fn new() -> Self {
        Self {
            globals: Vec::new(),
            functions: Vec::new(),
            keywords: vec![
                "let".to_string(),
                "if".to_string(),
                "else".to_string(),
                "while".to_string(),
                "for".to_string(),
                "fn".to_string(),
                "return".to_string(),
                "true".to_string(),
                "false".to_string(),
                "nil".to_string(),
                "print".to_string(),
            ],
            commands: vec![
                ":help".to_string(),
                ":quit".to_string(),
                ":clear".to_string(),
                ":vars".to_string(),
                ":funcs".to_string(),
                ":history".to_string(),
                ":reset".to_string(),
                ":type".to_string(),
                ":inspect".to_string(),
                ":time".to_string(),
            ],
        }
    }

    /// Add a global variable name for completion.
    pub fn add_global(&mut self, name: &str) {
        if !self.globals.contains(&name.to_string()) {
            self.globals.push(name.to_string());
        }
    }

    /// Add a function name for completion.
    pub fn add_function(&mut self, name: &str) {
        if !self.functions.contains(&name.to_string()) {
            self.functions.push(name.to_string());
        }
    }

    /// Remove a global from completions.
    pub fn remove_global(&mut self, name: &str) {
        self.globals.retain(|g| g != name);
    }

    /// Get completions for a partial input.
    pub fn complete(&self, partial: &str) -> Vec<String> {
        let prefix = partial.split_whitespace().last().unwrap_or(partial);
        let mut results = Vec::new();

        if prefix.starts_with(':') {
            // Command completion.
            for cmd in &self.commands {
                if cmd.starts_with(prefix) {
                    results.push(cmd.clone());
                }
            }
        } else {
            // Variable/function/keyword completion.
            for name in &self.globals {
                if name.starts_with(prefix) {
                    results.push(name.clone());
                }
            }
            for name in &self.functions {
                if name.starts_with(prefix) {
                    results.push(format!("{}(", name));
                }
            }
            for kw in &self.keywords {
                if kw.starts_with(prefix) {
                    results.push(kw.clone());
                }
            }
        }

        results.sort();
        results.dedup();
        results
    }

    /// Get the longest common prefix among completions.
    pub fn common_prefix(&self, partial: &str) -> String {
        let completions = self.complete(partial);
        if completions.is_empty() {
            return partial.to_string();
        }
        if completions.len() == 1 {
            return completions[0].clone();
        }

        let first = &completions[0];
        let mut prefix_len = first.len();
        for c in &completions[1..] {
            let common = first
                .chars()
                .zip(c.chars())
                .take_while(|(a, b)| a == b)
                .count();
            if common < prefix_len {
                prefix_len = common;
            }
        }
        first[..prefix_len].to_string()
    }
}

impl Default for CompletionProvider {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Multi-line Input
// ---------------------------------------------------------------------------

/// Multi-line input handler.
#[derive(Debug, Clone)]
pub struct MultiLineInput {
    /// Accumulated lines.
    lines: Vec<String>,
    /// Whether we're currently accumulating multi-line input.
    active: bool,
    /// Current nesting depth (open braces/parens).
    depth: i32,
}

impl MultiLineInput {
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            active: false,
            depth: 0,
        }
    }

    /// Feed a line of input. Returns `true` if the input is complete.
    pub fn feed(&mut self, line: &str) -> bool {
        // Count open/close braces and parens.
        for ch in line.chars() {
            match ch {
                '{' | '(' | '[' => self.depth += 1,
                '}' | ')' | ']' => self.depth -= 1,
                _ => {}
            }
        }

        self.lines.push(line.to_string());

        if self.depth <= 0 {
            // Input is complete.
            self.active = false;
            self.depth = 0;
            true
        } else {
            self.active = true;
            false
        }
    }

    /// Get the accumulated input.
    pub fn take(&mut self) -> String {
        let result = self.lines.join("\n");
        self.lines.clear();
        self.depth = 0;
        self.active = false;
        result
    }

    /// Whether we're in multi-line mode.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Current nesting depth.
    pub fn depth(&self) -> i32 {
        self.depth
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.lines.clear();
        self.depth = 0;
        self.active = false;
    }
}

impl Default for MultiLineInput {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Output Formatter
// ---------------------------------------------------------------------------

/// Formats REPL output for display.
pub struct OutputFormatter {
    /// Maximum array elements to display.
    pub max_array_elements: usize,
    /// Maximum string length before truncation.
    pub max_string_length: usize,
    /// Maximum nesting depth for display.
    pub max_depth: usize,
    /// Whether to show type annotations.
    pub show_types: bool,
    /// Color output (ANSI escape codes).
    pub use_colors: bool,
}

impl OutputFormatter {
    pub fn new() -> Self {
        Self {
            max_array_elements: 20,
            max_string_length: 200,
            max_depth: 5,
            show_types: false,
            use_colors: false,
        }
    }

    /// Format a value for display.
    pub fn format(&self, value: &ReplValue) -> String {
        self.format_depth(value, 0)
    }

    fn format_depth(&self, value: &ReplValue, depth: usize) -> String {
        if depth > self.max_depth {
            return "...".to_string();
        }

        let formatted = match value {
            ReplValue::Nil => "nil".to_string(),
            ReplValue::Bool(b) => b.to_string(),
            ReplValue::Int(i) => i.to_string(),
            ReplValue::Float(f) => {
                if f.fract() == 0.0 {
                    format!("{:.1}", f)
                } else {
                    format!("{}", f)
                }
            }
            ReplValue::String(s) => {
                if s.len() > self.max_string_length {
                    format!("\"{}...\" ({} chars)", &s[..self.max_string_length], s.len())
                } else {
                    format!("\"{}\"", s)
                }
            }
            ReplValue::Array(a) => {
                if a.is_empty() {
                    "[]".to_string()
                } else if a.len() <= self.max_array_elements {
                    let items: Vec<String> = a
                        .iter()
                        .map(|v| self.format_depth(v, depth + 1))
                        .collect();
                    format!("[{}]", items.join(", "))
                } else {
                    let items: Vec<String> = a
                        .iter()
                        .take(self.max_array_elements)
                        .map(|v| self.format_depth(v, depth + 1))
                        .collect();
                    format!("[{}, ... ({} more)]", items.join(", "), a.len() - self.max_array_elements)
                }
            }
            ReplValue::Map(m) => {
                if m.is_empty() {
                    "{}".to_string()
                } else {
                    let items: Vec<String> = m
                        .iter()
                        .take(self.max_array_elements)
                        .map(|(k, v)| format!("{}: {}", k, self.format_depth(v, depth + 1)))
                        .collect();
                    format!("{{{}}}", items.join(", "))
                }
            }
            ReplValue::Function { name, params } => {
                format!("fn {}({})", name, params.join(", "))
            }
            ReplValue::Error(msg) => format!("Error: {}", msg),
        };

        if self.show_types {
            format!("{} : {}", formatted, value.type_name())
        } else {
            formatted
        }
    }

    /// Format an error for display.
    pub fn format_error(&self, error: &ReplError) -> String {
        format!("Error: {}", error)
    }
}

impl Default for OutputFormatter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ReplCommand
// ---------------------------------------------------------------------------

/// A built-in REPL command (prefixed with `:`)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplCommand {
    Help,
    Quit,
    Clear,
    ListVars,
    ListFunctions,
    ShowHistory,
    Reset,
    TypeOf(String),
    Inspect(String),
    Time(String),
}

impl ReplCommand {
    /// Parse a command string.
    pub fn parse(input: &str) -> ReplResult<Self> {
        let parts: Vec<&str> = input.trim().splitn(2, ' ').collect();
        match parts[0] {
            ":help" | ":h" | ":?" => Ok(ReplCommand::Help),
            ":quit" | ":q" | ":exit" => Ok(ReplCommand::Quit),
            ":clear" | ":cls" => Ok(ReplCommand::Clear),
            ":vars" | ":v" => Ok(ReplCommand::ListVars),
            ":funcs" | ":f" => Ok(ReplCommand::ListFunctions),
            ":history" | ":hist" => Ok(ReplCommand::ShowHistory),
            ":reset" => Ok(ReplCommand::Reset),
            ":type" | ":t" => {
                let arg = parts.get(1).unwrap_or(&"").to_string();
                Ok(ReplCommand::TypeOf(arg))
            }
            ":inspect" | ":i" => {
                let arg = parts.get(1).unwrap_or(&"").to_string();
                Ok(ReplCommand::Inspect(arg))
            }
            ":time" => {
                let arg = parts.get(1).unwrap_or(&"").to_string();
                Ok(ReplCommand::Time(arg))
            }
            _ => Err(ReplError::UnknownCommand(parts[0].to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// Repl
// ---------------------------------------------------------------------------

/// Native function type for the REPL.
pub type NativeReplFn = Box<dyn Fn(&[ReplValue]) -> ReplResult<ReplValue> + Send + Sync>;

/// The Read-Eval-Print Loop environment.
pub struct Repl {
    /// Global variables.
    globals: HashMap<String, ReplValue>,
    /// Registered native functions.
    functions: HashMap<String, (Vec<String>, NativeReplFn)>,
    /// Command history.
    history: History,
    /// Tab completion provider.
    completion: CompletionProvider,
    /// Multi-line input handler.
    multi_line: MultiLineInput,
    /// Output formatter.
    formatter: OutputFormatter,
    /// Whether the REPL is running.
    running: bool,
    /// Prompt string.
    prompt: String,
    /// Continuation prompt (for multi-line).
    continuation_prompt: String,
    /// Total evaluations performed.
    eval_count: u64,
    /// Last result (stored as `_`).
    last_result: ReplValue,
}

impl Repl {
    /// Create a new REPL environment.
    pub fn new() -> Self {
        let mut repl = Self {
            globals: HashMap::new(),
            functions: HashMap::new(),
            history: History::new(1000),
            completion: CompletionProvider::new(),
            multi_line: MultiLineInput::new(),
            formatter: OutputFormatter::new(),
            running: true,
            prompt: ">>> ".to_string(),
            continuation_prompt: "... ".to_string(),
            eval_count: 0,
            last_result: ReplValue::Nil,
        };

        // Register built-in globals.
        repl.register_global("_", ReplValue::Nil);
        repl.register_global("PI", ReplValue::Float(std::f64::consts::PI));
        repl.register_global("E", ReplValue::Float(std::f64::consts::E));
        repl.register_global("TAU", ReplValue::Float(std::f64::consts::TAU));

        repl
    }

    /// Register a global variable.
    pub fn register_global(&mut self, name: &str, value: ReplValue) {
        self.globals.insert(name.to_string(), value);
        self.completion.add_global(name);
    }

    /// Register a native function.
    pub fn register_function<F>(
        &mut self,
        name: &str,
        params: Vec<&str>,
        func: F,
    ) where
        F: Fn(&[ReplValue]) -> ReplResult<ReplValue> + Send + Sync + 'static,
    {
        let param_strings: Vec<String> = params.iter().map(|p| p.to_string()).collect();
        self.functions
            .insert(name.to_string(), (param_strings, Box::new(func)));
        self.completion.add_function(name);
    }

    /// Process a line of input.
    pub fn process_line(&mut self, line: &str) -> ReplLineResult {
        let trimmed = line.trim();

        // Empty line.
        if trimmed.is_empty() {
            return ReplLineResult::Empty;
        }

        // Multi-line mode.
        if self.multi_line.is_active() {
            if self.multi_line.feed(trimmed) {
                let full_input = self.multi_line.take();
                return self.evaluate_and_record(&full_input);
            } else {
                return ReplLineResult::NeedMore;
            }
        }

        // Commands.
        if trimmed.starts_with(':') {
            return match ReplCommand::parse(trimmed) {
                Ok(cmd) => self.execute_command(cmd),
                Err(e) => ReplLineResult::Error(self.formatter.format_error(&e)),
            };
        }

        // Check for multi-line start.
        if !self.multi_line.feed(trimmed) {
            return ReplLineResult::NeedMore;
        }
        let full_input = self.multi_line.take();

        self.evaluate_and_record(&full_input)
    }

    fn evaluate_and_record(&mut self, input: &str) -> ReplLineResult {
        let start = Instant::now();
        let result = self.eval(input);
        let duration = start.elapsed();

        match result {
            Ok(value) => {
                let formatted = self.formatter.format(&value);
                self.last_result = value;
                self.globals.insert("_".to_string(), self.last_result.clone());

                self.history.push(HistoryEntry {
                    input: input.to_string(),
                    result: Some(formatted.clone()),
                    is_error: false,
                    timestamp: Instant::now(),
                    duration,
                });

                self.eval_count += 1;
                ReplLineResult::Value(formatted)
            }
            Err(e) => {
                let error_msg = self.formatter.format_error(&e);
                self.history.push(HistoryEntry {
                    input: input.to_string(),
                    result: Some(error_msg.clone()),
                    is_error: true,
                    timestamp: Instant::now(),
                    duration,
                });
                ReplLineResult::Error(error_msg)
            }
        }
    }

    /// Evaluate an expression.
    pub fn eval(&mut self, input: &str) -> ReplResult<ReplValue> {
        let trimmed = input.trim();

        // Variable assignment: let x = expr
        if trimmed.starts_with("let ") {
            return self.eval_assignment(&trimmed[4..]);
        }

        // Assignment: x = expr
        if let Some(eq_pos) = trimmed.find('=') {
            if eq_pos > 0
                && !trimmed[..eq_pos].contains(' ')
                && !trimmed[eq_pos..].starts_with("==")
            {
                let name = trimmed[..eq_pos].trim();
                let expr = trimmed[eq_pos + 1..].trim();
                let value = self.eval_expression(expr)?;
                self.register_global(name, value.clone());
                return Ok(value);
            }
        }

        // Function call: name(args)
        if let Some(paren_pos) = trimmed.find('(') {
            if trimmed.ends_with(')') {
                let name = trimmed[..paren_pos].trim();
                if self.functions.contains_key(name) {
                    let args_str = &trimmed[paren_pos + 1..trimmed.len() - 1];
                    return self.eval_function_call(name, args_str);
                }
            }
        }

        // Expression.
        self.eval_expression(trimmed)
    }

    fn eval_assignment(&mut self, rest: &str) -> ReplResult<ReplValue> {
        let parts: Vec<&str> = rest.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(ReplError::SyntaxError("expected 'let name = value'".to_string()));
        }
        let name = parts[0].trim();
        let expr = parts[1].trim();
        let value = self.eval_expression(expr)?;
        self.register_global(name, value.clone());
        Ok(value)
    }

    fn eval_expression(&self, expr: &str) -> ReplResult<ReplValue> {
        let trimmed = expr.trim();

        // Nil.
        if trimmed == "nil" || trimmed == "null" {
            return Ok(ReplValue::Nil);
        }

        // Boolean.
        if trimmed == "true" {
            return Ok(ReplValue::Bool(true));
        }
        if trimmed == "false" {
            return Ok(ReplValue::Bool(false));
        }

        // String literal.
        if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
            return Ok(ReplValue::String(trimmed[1..trimmed.len() - 1].to_string()));
        }

        // Integer literal.
        if let Ok(i) = trimmed.parse::<i64>() {
            return Ok(ReplValue::Int(i));
        }

        // Float literal.
        if let Ok(f) = trimmed.parse::<f64>() {
            return Ok(ReplValue::Float(f));
        }

        // Array literal.
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let inner = &trimmed[1..trimmed.len() - 1].trim();
            if inner.is_empty() {
                return Ok(ReplValue::Array(Vec::new()));
            }
            let elements: ReplResult<Vec<ReplValue>> = inner
                .split(',')
                .map(|e| self.eval_expression(e.trim()))
                .collect();
            return Ok(ReplValue::Array(elements?));
        }

        // Simple binary operations.
        if let Some(result) = self.try_binary_op(trimmed) {
            return result;
        }

        // Unary negation.
        if trimmed.starts_with('-') && trimmed.len() > 1 {
            let inner = &trimmed[1..];
            match self.eval_expression(inner)? {
                ReplValue::Int(i) => return Ok(ReplValue::Int(-i)),
                ReplValue::Float(f) => return Ok(ReplValue::Float(-f)),
                _ => return Err(ReplError::TypeError("cannot negate non-numeric value".to_string())),
            }
        }

        // Variable lookup.
        if let Some(value) = self.globals.get(trimmed) {
            return Ok(value.clone());
        }

        Err(ReplError::UndefinedName(trimmed.to_string()))
    }

    fn try_binary_op(&self, expr: &str) -> Option<ReplResult<ReplValue>> {
        // Try operators in precedence order (lowest first for left-to-right).
        let operators = [
            ("+", BinOp::Add),
            ("-", BinOp::Sub),
            ("*", BinOp::Mul),
            ("/", BinOp::Div),
            ("%", BinOp::Mod),
            ("==", BinOp::Eq),
            ("!=", BinOp::Neq),
            (">=", BinOp::Gte),
            ("<=", BinOp::Lte),
            (">", BinOp::Gt),
            ("<", BinOp::Lt),
        ];

        for (op_str, op) in &operators {
            // Find the operator, but skip if it's the first character (unary).
            if let Some(pos) = expr[1..].find(op_str) {
                let pos = pos + 1; // Adjust for the skip.
                let left = &expr[..pos];
                let right = &expr[pos + op_str.len()..];

                if left.trim().is_empty() || right.trim().is_empty() {
                    continue;
                }

                let left_val = match self.eval_expression(left.trim()) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                };
                let right_val = match self.eval_expression(right.trim()) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                };

                return Some(self.apply_binary_op(&left_val, &right_val, op));
            }
        }

        None
    }

    fn apply_binary_op(
        &self,
        left: &ReplValue,
        right: &ReplValue,
        op: &BinOp,
    ) -> ReplResult<ReplValue> {
        match (left, right, op) {
            // Integer arithmetic.
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Add) => Ok(ReplValue::Int(a + b)),
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Sub) => Ok(ReplValue::Int(a - b)),
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Mul) => Ok(ReplValue::Int(a * b)),
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Div) => {
                if *b == 0 {
                    Err(ReplError::DivisionByZero)
                } else {
                    Ok(ReplValue::Int(a / b))
                }
            }
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Mod) => {
                if *b == 0 {
                    Err(ReplError::DivisionByZero)
                } else {
                    Ok(ReplValue::Int(a % b))
                }
            }

            // Float arithmetic.
            (ReplValue::Float(a), ReplValue::Float(b), BinOp::Add) => Ok(ReplValue::Float(a + b)),
            (ReplValue::Float(a), ReplValue::Float(b), BinOp::Sub) => Ok(ReplValue::Float(a - b)),
            (ReplValue::Float(a), ReplValue::Float(b), BinOp::Mul) => Ok(ReplValue::Float(a * b)),
            (ReplValue::Float(a), ReplValue::Float(b), BinOp::Div) => {
                if *b == 0.0 {
                    Err(ReplError::DivisionByZero)
                } else {
                    Ok(ReplValue::Float(a / b))
                }
            }

            // Mixed int/float.
            (ReplValue::Int(a), ReplValue::Float(b), op) => {
                self.apply_binary_op(&ReplValue::Float(*a as f64), &ReplValue::Float(*b), op)
            }
            (ReplValue::Float(a), ReplValue::Int(b), op) => {
                self.apply_binary_op(&ReplValue::Float(*a), &ReplValue::Float(*b as f64), op)
            }

            // String concatenation.
            (ReplValue::String(a), ReplValue::String(b), BinOp::Add) => {
                Ok(ReplValue::String(format!("{}{}", a, b)))
            }

            // Comparison.
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Eq) => Ok(ReplValue::Bool(a == b)),
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Neq) => Ok(ReplValue::Bool(a != b)),
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Lt) => Ok(ReplValue::Bool(a < b)),
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Gt) => Ok(ReplValue::Bool(a > b)),
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Lte) => Ok(ReplValue::Bool(a <= b)),
            (ReplValue::Int(a), ReplValue::Int(b), BinOp::Gte) => Ok(ReplValue::Bool(a >= b)),

            _ => Err(ReplError::TypeError(format!(
                "unsupported operation {} {:?} {}",
                left.type_name(),
                op,
                right.type_name()
            ))),
        }
    }

    fn eval_function_call(&self, name: &str, args_str: &str) -> ReplResult<ReplValue> {
        let args: ReplResult<Vec<ReplValue>> = if args_str.trim().is_empty() {
            Ok(Vec::new())
        } else {
            args_str
                .split(',')
                .map(|a| self.eval_expression(a.trim()))
                .collect()
        };
        let args = args?;

        if let Some((_, func)) = self.functions.get(name) {
            func(&args)
        } else {
            Err(ReplError::UndefinedName(name.to_string()))
        }
    }

    /// Execute a built-in command.
    fn execute_command(&mut self, cmd: ReplCommand) -> ReplLineResult {
        match cmd {
            ReplCommand::Help => {
                let help = r#"REPL Commands:
  :help     Show this help message
  :quit     Exit the REPL
  :clear    Clear the screen
  :vars     List all variables
  :funcs    List all functions
  :history  Show command history
  :reset    Reset the environment
  :type X   Show the type of expression X
  :inspect X Show detailed info about X
  :time X   Measure execution time of X"#;
                ReplLineResult::Output(help.to_string())
            }
            ReplCommand::Quit => {
                self.running = false;
                ReplLineResult::Quit
            }
            ReplCommand::Clear => ReplLineResult::Clear,
            ReplCommand::ListVars => {
                let mut lines = Vec::new();
                for (name, value) in &self.globals {
                    lines.push(format!("  {} = {} : {}", name, self.formatter.format(value), value.type_name()));
                }
                lines.sort();
                ReplLineResult::Output(lines.join("\n"))
            }
            ReplCommand::ListFunctions => {
                let mut lines = Vec::new();
                for (name, (params, _)) in &self.functions {
                    lines.push(format!("  fn {}({})", name, params.join(", ")));
                }
                lines.sort();
                ReplLineResult::Output(lines.join("\n"))
            }
            ReplCommand::ShowHistory => {
                let mut lines = Vec::new();
                for (i, entry) in self.history.entries().iter().enumerate() {
                    let marker = if entry.is_error { "!" } else { " " };
                    lines.push(format!("{}{}: {}", marker, i + 1, entry.input));
                }
                ReplLineResult::Output(lines.join("\n"))
            }
            ReplCommand::Reset => {
                self.globals.clear();
                self.register_global("_", ReplValue::Nil);
                self.register_global("PI", ReplValue::Float(std::f64::consts::PI));
                self.register_global("E", ReplValue::Float(std::f64::consts::E));
                ReplLineResult::Output("Environment reset.".to_string())
            }
            ReplCommand::TypeOf(expr) => {
                match self.eval_expression(&expr) {
                    Ok(val) => ReplLineResult::Output(val.type_name().to_string()),
                    Err(e) => ReplLineResult::Error(self.formatter.format_error(&e)),
                }
            }
            ReplCommand::Inspect(expr) => {
                match self.eval_expression(&expr) {
                    Ok(val) => {
                        let info = format!(
                            "Value: {}\nType:  {}\nDebug: {:?}",
                            self.formatter.format(&val),
                            val.type_name(),
                            val,
                        );
                        ReplLineResult::Output(info)
                    }
                    Err(e) => ReplLineResult::Error(self.formatter.format_error(&e)),
                }
            }
            ReplCommand::Time(expr) => {
                let start = Instant::now();
                let result = self.eval(&expr);
                let elapsed = start.elapsed();
                match result {
                    Ok(val) => {
                        let msg = format!(
                            "{}\n(executed in {:.3}ms)",
                            self.formatter.format(&val),
                            elapsed.as_secs_f64() * 1000.0
                        );
                        ReplLineResult::Output(msg)
                    }
                    Err(e) => ReplLineResult::Error(self.formatter.format_error(&e)),
                }
            }
        }
    }

    /// Get tab completions for partial input.
    pub fn complete(&self, partial: &str) -> Vec<String> {
        self.completion.complete(partial)
    }

    /// Get the current prompt.
    pub fn prompt(&self) -> &str {
        if self.multi_line.is_active() {
            &self.continuation_prompt
        } else {
            &self.prompt
        }
    }

    /// Whether the REPL is still running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get the history.
    pub fn history(&self) -> &History {
        &self.history
    }

    /// Get the history mutably.
    pub fn history_mut(&mut self) -> &mut History {
        &mut self.history
    }

    /// Get the output formatter.
    pub fn formatter(&self) -> &OutputFormatter {
        &self.formatter
    }

    /// Get the output formatter mutably.
    pub fn formatter_mut(&mut self) -> &mut OutputFormatter {
        &mut self.formatter
    }

    /// Number of evaluations performed.
    pub fn eval_count(&self) -> u64 {
        self.eval_count
    }

    /// Get all global variable names.
    pub fn global_names(&self) -> Vec<&str> {
        self.globals.keys().map(|s| s.as_str()).collect()
    }

    /// Get a global value.
    pub fn get_global(&self, name: &str) -> Option<&ReplValue> {
        self.globals.get(name)
    }
}

impl Default for Repl {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of processing a line of REPL input.
#[derive(Debug, Clone)]
pub enum ReplLineResult {
    /// A value was produced.
    Value(String),
    /// An error occurred.
    Error(String),
    /// Output text (from a command).
    Output(String),
    /// Need more input (multi-line).
    NeedMore,
    /// Empty input.
    Empty,
    /// Clear the screen.
    Clear,
    /// Quit the REPL.
    Quit,
}

/// Binary operation type.
#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_literals() {
        let repl = Repl::new();
        assert_eq!(repl.eval("42").unwrap(), ReplValue::Int(42));
        assert_eq!(repl.eval("3.14").unwrap(), ReplValue::Float(3.14));
        assert_eq!(repl.eval("true").unwrap(), ReplValue::Bool(true));
        assert_eq!(repl.eval("nil").unwrap(), ReplValue::Nil);
        assert_eq!(
            repl.eval("\"hello\"").unwrap(),
            ReplValue::String("hello".to_string())
        );
    }

    #[test]
    fn test_eval_arithmetic() {
        let repl = Repl::new();
        assert_eq!(repl.eval("2+3").unwrap(), ReplValue::Int(5));
        assert_eq!(repl.eval("10-4").unwrap(), ReplValue::Int(6));
        assert_eq!(repl.eval("3*4").unwrap(), ReplValue::Int(12));
        assert_eq!(repl.eval("10/3").unwrap(), ReplValue::Int(3));
    }

    #[test]
    fn test_variable_assignment() {
        let mut repl = Repl::new();
        repl.eval("let x = 42").unwrap();
        assert_eq!(repl.eval("x").unwrap(), ReplValue::Int(42));
    }

    #[test]
    fn test_function_call() {
        let mut repl = Repl::new();
        repl.register_function("double", vec!["x"], |args| {
            match args.first() {
                Some(ReplValue::Int(x)) => Ok(ReplValue::Int(x * 2)),
                _ => Err(ReplError::TypeError("expected int".to_string())),
            }
        });

        assert_eq!(repl.eval("double(21)").unwrap(), ReplValue::Int(42));
    }

    #[test]
    fn test_undefined_variable() {
        let repl = Repl::new();
        assert!(matches!(
            repl.eval("undefined_var"),
            Err(ReplError::UndefinedName(_))
        ));
    }

    #[test]
    fn test_division_by_zero() {
        let repl = Repl::new();
        assert!(matches!(repl.eval("10/0"), Err(ReplError::DivisionByZero)));
    }

    #[test]
    fn test_history() {
        let mut repl = Repl::new();
        repl.process_line("42");
        repl.process_line("true");

        assert_eq!(repl.history().len(), 2);
    }

    #[test]
    fn test_tab_completion() {
        let mut repl = Repl::new();
        repl.register_global("player_health", ReplValue::Int(100));
        repl.register_global("player_name", ReplValue::String("Hero".to_string()));

        let completions = repl.complete("player");
        assert!(completions.len() >= 2);
    }

    #[test]
    fn test_commands() {
        let mut repl = Repl::new();

        let result = repl.process_line(":help");
        assert!(matches!(result, ReplLineResult::Output(_)));

        let result = repl.process_line(":vars");
        assert!(matches!(result, ReplLineResult::Output(_)));
    }

    #[test]
    fn test_multi_line() {
        let mut input = MultiLineInput::new();
        assert!(!input.feed("{"));
        assert!(input.is_active());
        assert!(input.feed("}"));
        assert!(!input.is_active());
    }

    #[test]
    fn test_output_formatter() {
        let fmt = OutputFormatter::new();
        assert_eq!(fmt.format(&ReplValue::Int(42)), "42");
        assert_eq!(fmt.format(&ReplValue::String("hi".to_string())), "\"hi\"");
        assert_eq!(fmt.format(&ReplValue::Array(vec![ReplValue::Int(1), ReplValue::Int(2)])), "[1, 2]");
    }
}
