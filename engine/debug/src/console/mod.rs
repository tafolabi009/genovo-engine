//! In-game developer console for the Genovo engine.
//!
//! The [`Console`] provides a command-line interface for runtime debugging,
//! variable tweaking, and command execution. Commands can be registered
//! dynamically and support tab completion, history navigation, and
//! colored output.
//!
//! # Built-in Commands
//!
//! - `help [command]` — display help for all commands or a specific one
//! - `clear` — clear the console output buffer
//! - `echo <text>` — print text to the console
//! - `set <var> <value>` — set a console variable
//! - `get <var>` — get a console variable's value
//! - `list_commands` — list all registered commands
//! - `exec <file>` — execute a script file (one command per line)
//! - `alias <name> <command>` — create a command alias
//!
//! # Console Variables
//!
//! [`ConsoleVar<T>`] wraps a typed value with get/set access, optional change
//! callbacks, and integration with the `set`/`get` commands.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// Output categories and colors
// ---------------------------------------------------------------------------

/// Output line categories for coloring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleCategory {
    /// General informational output.
    Info,
    /// Warning messages.
    Warning,
    /// Error messages.
    Error,
    /// Debug-level output.
    Debug,
    /// System / engine messages.
    System,
    /// Command echo (the command itself).
    Command,
    /// Success messages.
    Success,
}

impl ConsoleCategory {
    /// ANSI-like color code for terminal rendering.
    pub fn color_code(&self) -> [u8; 4] {
        match self {
            ConsoleCategory::Info => [255, 255, 255, 255],
            ConsoleCategory::Warning => [255, 200, 50, 255],
            ConsoleCategory::Error => [255, 80, 80, 255],
            ConsoleCategory::Debug => [150, 150, 150, 255],
            ConsoleCategory::System => [100, 200, 255, 255],
            ConsoleCategory::Command => [200, 200, 200, 255],
            ConsoleCategory::Success => [80, 255, 80, 255],
        }
    }

    /// Category label for formatting.
    pub fn label(&self) -> &'static str {
        match self {
            ConsoleCategory::Info => "INFO",
            ConsoleCategory::Warning => "WARN",
            ConsoleCategory::Error => "ERROR",
            ConsoleCategory::Debug => "DEBUG",
            ConsoleCategory::System => "SYS",
            ConsoleCategory::Command => "CMD",
            ConsoleCategory::Success => "OK",
        }
    }
}

// ---------------------------------------------------------------------------
// ConsoleLine
// ---------------------------------------------------------------------------

/// A single line of console output.
#[derive(Debug, Clone)]
pub struct ConsoleLine {
    /// The text content of the line.
    pub text: String,
    /// RGBA color for rendering.
    pub color: [u8; 4],
    /// Timestamp when the line was created.
    pub timestamp: Instant,
    /// Category of the line.
    pub category: ConsoleCategory,
}

impl ConsoleLine {
    /// Create a new console line with the given text and category.
    pub fn new(text: impl Into<String>, category: ConsoleCategory) -> Self {
        Self {
            text: text.into(),
            color: category.color_code(),
            timestamp: Instant::now(),
            category,
        }
    }

    /// Create an info line.
    pub fn info(text: impl Into<String>) -> Self {
        Self::new(text, ConsoleCategory::Info)
    }

    /// Create a warning line.
    pub fn warning(text: impl Into<String>) -> Self {
        Self::new(text, ConsoleCategory::Warning)
    }

    /// Create an error line.
    pub fn error(text: impl Into<String>) -> Self {
        Self::new(text, ConsoleCategory::Error)
    }

    /// Create a debug line.
    pub fn debug(text: impl Into<String>) -> Self {
        Self::new(text, ConsoleCategory::Debug)
    }

    /// Create a system line.
    pub fn system(text: impl Into<String>) -> Self {
        Self::new(text, ConsoleCategory::System)
    }
}

impl fmt::Display for ConsoleLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.category.label(), self.text)
    }
}

// ---------------------------------------------------------------------------
// Argument types
// ---------------------------------------------------------------------------

/// Supported argument types for command registration.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgType {
    /// A string argument.
    String,
    /// An integer argument.
    Integer,
    /// A floating-point argument.
    Float,
    /// A boolean argument.
    Bool,
    /// An optional argument of the given type.
    Optional(Box<ArgType>),
    /// Variadic: consumes all remaining arguments as strings.
    Variadic,
}

impl fmt::Display for ArgType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArgType::String => write!(f, "string"),
            ArgType::Integer => write!(f, "int"),
            ArgType::Float => write!(f, "float"),
            ArgType::Bool => write!(f, "bool"),
            ArgType::Optional(inner) => write!(f, "[{}]", inner),
            ArgType::Variadic => write!(f, "..."),
        }
    }
}

// ---------------------------------------------------------------------------
// ConsoleCommand
// ---------------------------------------------------------------------------

/// A registered console command.
pub struct ConsoleCommand {
    /// Command name (lowercase, no spaces).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Expected argument types.
    pub arg_types: Vec<ArgType>,
    /// The callback invoked when the command is executed.
    callback: Box<dyn Fn(&[String], &mut Console) + Send + Sync>,
}

impl ConsoleCommand {
    /// Create a new console command.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        arg_types: Vec<ArgType>,
        callback: impl Fn(&[String], &mut Console) + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            arg_types,
            callback: Box::new(callback),
        }
    }

    /// Format a usage string for this command.
    pub fn usage(&self) -> String {
        let args: Vec<String> = self.arg_types.iter().map(|a| format!("<{}>", a)).collect();
        if args.is_empty() {
            self.name.clone()
        } else {
            format!("{} {}", self.name, args.join(" "))
        }
    }
}

impl fmt::Debug for ConsoleCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConsoleCommand")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("arg_types", &self.arg_types)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ConsoleVar<T>
// ---------------------------------------------------------------------------

/// A typed console variable with optional change callback.
///
/// Console variables can be read and written via the `get`/`set` commands.
/// They are stored as strings internally and parsed on access.
///
/// # Example
///
/// ```ignore
/// let mut console = Console::new();
/// let gravity = ConsoleVar::new("sv_gravity", 9.81_f32, "World gravity");
/// console.register_var(gravity);
///
/// console.execute("set sv_gravity 20.0");
/// ```
pub struct ConsoleVar<T: ConsoleVarValue> {
    /// Variable name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Current value.
    value: T,
    /// Default value.
    default_value: T,
    /// Optional change callback.
    on_change: Option<Box<dyn Fn(&T, &T) + Send + Sync>>,
}

impl<T: ConsoleVarValue> ConsoleVar<T> {
    /// Create a new console variable.
    pub fn new(name: impl Into<String>, default: T, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            value: default.clone(),
            default_value: default,
            on_change: None,
        }
    }

    /// Set a change callback that is invoked whenever the value changes.
    pub fn with_on_change(
        mut self,
        callback: impl Fn(&T, &T) + Send + Sync + 'static,
    ) -> Self {
        self.on_change = Some(Box::new(callback));
        self
    }

    /// Get the current value.
    pub fn get(&self) -> &T {
        &self.value
    }

    /// Set the value, invoking the change callback if registered.
    pub fn set(&mut self, new_value: T) {
        let old = std::mem::replace(&mut self.value, new_value);
        if let Some(ref cb) = self.on_change {
            cb(&old, &self.value);
        }
    }

    /// Reset to the default value.
    pub fn reset(&mut self) {
        self.set(self.default_value.clone());
    }

    /// Get the default value.
    pub fn default_value(&self) -> &T {
        &self.default_value
    }
}

impl<T: ConsoleVarValue> fmt::Debug for ConsoleVar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConsoleVar")
            .field("name", &self.name)
            .field("value", &self.value.to_string())
            .field("default", &self.default_value.to_string())
            .finish()
    }
}

/// Trait for types that can be used as console variable values.
pub trait ConsoleVarValue: Clone + fmt::Display + Send + Sync + 'static {
    /// Parse a string into this type.
    fn from_str(s: &str) -> Option<Self>;
    /// Type name for display purposes.
    fn type_name() -> &'static str;
}

impl ConsoleVarValue for f32 {
    fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
    fn type_name() -> &'static str {
        "float"
    }
}

impl ConsoleVarValue for f64 {
    fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
    fn type_name() -> &'static str {
        "double"
    }
}

impl ConsoleVarValue for i32 {
    fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
    fn type_name() -> &'static str {
        "int"
    }
}

impl ConsoleVarValue for i64 {
    fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
    fn type_name() -> &'static str {
        "long"
    }
}

impl ConsoleVarValue for u32 {
    fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
    fn type_name() -> &'static str {
        "uint"
    }
}

impl ConsoleVarValue for bool {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" => Some(false),
            _ => None,
        }
    }
    fn type_name() -> &'static str {
        "bool"
    }
}

impl ConsoleVarValue for String {
    fn from_str(s: &str) -> Option<Self> {
        Some(s.to_string())
    }
    fn type_name() -> &'static str {
        "string"
    }
}

// ---------------------------------------------------------------------------
// ConsoleVarEntry — type-erased variable storage
// ---------------------------------------------------------------------------

/// Type-erased console variable entry.
struct ConsoleVarEntry {
    /// Variable name.
    name: String,
    /// Description.
    description: String,
    /// Type name.
    type_name: String,
    /// Current value as string.
    value_str: String,
    /// Default value as string.
    default_str: String,
    /// Setter: parse and apply a string value.
    setter: Box<dyn Fn(&str) -> Result<String, String> + Send + Sync>,
}

impl fmt::Debug for ConsoleVarEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConsoleVarEntry")
            .field("name", &self.name)
            .field("value", &self.value_str)
            .field("type", &self.type_name)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Console
// ---------------------------------------------------------------------------

/// In-game developer console with command registration, history, tab
/// completion, and colored output.
pub struct Console {
    /// Registered commands, keyed by name.
    commands: HashMap<String, ConsoleCommand>,
    /// Command aliases (alias_name -> target_command).
    aliases: HashMap<String, String>,
    /// Console output buffer.
    output: Vec<ConsoleLine>,
    /// Maximum output buffer size.
    max_output_lines: usize,
    /// Command history (most recent last).
    history: Vec<String>,
    /// Maximum history size.
    max_history: usize,
    /// Current history navigation index (for up/down arrow).
    history_index: Option<usize>,
    /// Current input line.
    input: String,
    /// Cursor position in the input.
    cursor_pos: usize,
    /// Console variables (type-erased).
    variables: HashMap<String, Arc<Mutex<ConsoleVarEntry>>>,
    /// Whether the console is visible.
    visible: bool,
    /// Tab completion state.
    tab_completions: Vec<String>,
    /// Current tab completion index.
    tab_index: usize,
}

impl Console {
    /// Create a new console with default settings and built-in commands.
    pub fn new() -> Self {
        let mut console = Self {
            commands: HashMap::new(),
            aliases: HashMap::new(),
            output: Vec::new(),
            max_output_lines: 1000,
            history: Vec::new(),
            max_history: 200,
            history_index: None,
            input: String::new(),
            cursor_pos: 0,
            variables: HashMap::new(),
            visible: false,
            tab_completions: Vec::new(),
            tab_index: 0,
        };
        console.register_builtins();
        console
    }

    /// Register all built-in commands.
    fn register_builtins(&mut self) {
        // help
        self.register_command(ConsoleCommand::new(
            "help",
            "Display help. Usage: help [command]",
            vec![ArgType::Optional(Box::new(ArgType::String))],
            |args, console| {
                if let Some(cmd_name) = args.first() {
                    if let Some(cmd) = console.commands.get(cmd_name.as_str()) {
                        console.print_system(&format!("{} — {}", cmd.usage(), cmd.description));
                    } else {
                        console.print_error(&format!("Unknown command: {}", cmd_name));
                    }
                } else {
                    console.print_system("Available commands:");
                    let mut names: Vec<String> =
                        console.commands.keys().cloned().collect();
                    names.sort();
                    for name in &names {
                        if let Some(cmd) = console.commands.get(name.as_str()) {
                            console.print_info(&format!("  {} — {}", name, cmd.description));
                        }
                    }
                }
            },
        ));

        // clear
        self.register_command(ConsoleCommand::new(
            "clear",
            "Clear the console output",
            vec![],
            |_args, console| {
                console.clear_output();
            },
        ));

        // echo
        self.register_command(ConsoleCommand::new(
            "echo",
            "Print text to the console",
            vec![ArgType::Variadic],
            |args, console| {
                let text = args.join(" ");
                console.print_info(&text);
            },
        ));

        // set
        self.register_command(ConsoleCommand::new(
            "set",
            "Set a console variable. Usage: set <name> <value>",
            vec![ArgType::String, ArgType::String],
            |args, console| {
                if args.len() < 2 {
                    console.print_error("Usage: set <name> <value>");
                    return;
                }
                let name = &args[0];
                let value = &args[1];
                if let Some(var) = console.variables.get(name).cloned() {
                    let mut entry = var.lock();
                    match (entry.setter)(value) {
                        Ok(new_val) => {
                            entry.value_str = new_val.clone();
                            // We need to print outside the lock
                            drop(entry);
                            console.print_success(&format!("{} = {}", name, new_val));
                        }
                        Err(e) => {
                            drop(entry);
                            console.print_error(&format!("Failed to set {}: {}", name, e));
                        }
                    }
                } else {
                    console.print_error(&format!("Unknown variable: {}", name));
                }
            },
        ));

        // get
        self.register_command(ConsoleCommand::new(
            "get",
            "Get a console variable's value. Usage: get <name>",
            vec![ArgType::String],
            |args, console| {
                if args.is_empty() {
                    console.print_error("Usage: get <name>");
                    return;
                }
                let name = &args[0];
                if let Some(var) = console.variables.get(name).cloned() {
                    let entry = var.lock();
                    console.print_info(&format!(
                        "{} = {} (type: {}, default: {})",
                        entry.name, entry.value_str, entry.type_name, entry.default_str
                    ));
                } else {
                    console.print_error(&format!("Unknown variable: {}", name));
                }
            },
        ));

        // list_commands
        self.register_command(ConsoleCommand::new(
            "list_commands",
            "List all registered commands",
            vec![],
            |_args, console| {
                let mut names: Vec<String> = console.commands.keys().cloned().collect();
                names.sort();
                console.print_system(&format!("Registered commands ({}):", names.len()));
                for name in &names {
                    console.print_info(&format!("  {}", name));
                }
            },
        ));

        // list_vars
        self.register_command(ConsoleCommand::new(
            "list_vars",
            "List all console variables",
            vec![],
            |_args, console| {
                let mut names: Vec<String> = console.variables.keys().cloned().collect();
                names.sort();
                console.print_system(&format!("Console variables ({}):", names.len()));
                let lines: Vec<String> = names
                    .iter()
                    .filter_map(|name| {
                        console.variables.get(name.as_str()).map(|var| {
                            let entry = var.lock();
                            format!("  {} = {} [{}]", entry.name, entry.value_str, entry.type_name)
                        })
                    })
                    .collect();
                for line in &lines {
                    console.print_info(line);
                }
            },
        ));

        // exec
        self.register_command(ConsoleCommand::new(
            "exec",
            "Execute a script file (one command per line). Usage: exec <file>",
            vec![ArgType::String],
            |args, console| {
                if args.is_empty() {
                    console.print_error("Usage: exec <file>");
                    return;
                }
                let path = &args[0];
                match std::fs::read_to_string(path) {
                    Ok(contents) => {
                        console.print_system(&format!("Executing {}...", path));
                        let lines: Vec<String> = contents
                            .lines()
                            .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
                            .map(|l| l.to_string())
                            .collect();
                        for line in lines {
                            console.execute(&line);
                        }
                        console.print_success(&format!("Finished executing {}", path));
                    }
                    Err(e) => {
                        console.print_error(&format!("Failed to read {}: {}", path, e));
                    }
                }
            },
        ));

        // alias
        self.register_command(ConsoleCommand::new(
            "alias",
            "Create a command alias. Usage: alias <name> <command>",
            vec![ArgType::String, ArgType::Variadic],
            |args, console| {
                if args.len() < 2 {
                    console.print_error("Usage: alias <name> <command>");
                    return;
                }
                let alias_name = args[0].clone();
                let target = args[1..].join(" ");
                console.aliases.insert(alias_name.clone(), target.clone());
                console.print_success(&format!("Alias: {} -> {}", alias_name, target));
            },
        ));

        // reset_var
        self.register_command(ConsoleCommand::new(
            "reset_var",
            "Reset a console variable to its default. Usage: reset_var <name>",
            vec![ArgType::String],
            |args, console| {
                if args.is_empty() {
                    console.print_error("Usage: reset_var <name>");
                    return;
                }
                let name = &args[0];
                if let Some(var) = console.variables.get(name).cloned() {
                    let mut entry = var.lock();
                    let default = entry.default_str.clone();
                    match (entry.setter)(&default) {
                        Ok(new_val) => {
                            entry.value_str = new_val;
                            drop(entry);
                            console.print_success(&format!("{} reset to {}", name, default));
                        }
                        Err(e) => {
                            drop(entry);
                            console.print_error(&format!("Failed to reset {}: {}", name, e));
                        }
                    }
                } else {
                    console.print_error(&format!("Unknown variable: {}", name));
                }
            },
        ));

        // version
        self.register_command(ConsoleCommand::new(
            "version",
            "Display engine version",
            vec![],
            |_args, console| {
                console.print_system("Genovo Engine v0.1.0");
            },
        ));
    }

    // -- Command registration -----------------------------------------------

    /// Register a command with the console.
    pub fn register_command(&mut self, command: ConsoleCommand) {
        self.commands.insert(command.name.clone(), command);
    }

    /// Unregister a command by name.
    pub fn remove_command(&mut self, name: &str) -> bool {
        self.commands.remove(name).is_some()
    }

    /// Check if a command exists.
    pub fn has_command(&self, name: &str) -> bool {
        self.commands.contains_key(name)
    }

    // -- Variable registration -----------------------------------------------

    /// Register a console variable.
    pub fn register_var<T: ConsoleVarValue>(&mut self, var: ConsoleVar<T>) {
        let name = var.name.clone();
        let description = var.description.clone();
        let type_name = T::type_name().to_string();
        let value_str = var.value.to_string();
        let default_str = var.default_value.to_string();

        // Wrap the var in an Arc<Mutex> so the setter closure can mutate it.
        let var = Arc::new(Mutex::new(var));
        let var_clone = Arc::clone(&var);

        let setter = Box::new(move |s: &str| -> Result<String, String> {
            match T::from_str(s) {
                Some(val) => {
                    let mut v = var_clone.lock();
                    v.set(val);
                    Ok(v.get().to_string())
                }
                None => Err(format!(
                    "Cannot parse '{}' as {}",
                    s,
                    T::type_name()
                )),
            }
        });

        let entry = ConsoleVarEntry {
            name: name.clone(),
            description,
            type_name,
            value_str,
            default_str,
            setter,
        };

        self.variables
            .insert(name, Arc::new(Mutex::new(entry)));
    }

    /// Get the string value of a console variable.
    pub fn get_var_string(&self, name: &str) -> Option<String> {
        self.variables
            .get(name)
            .map(|v| v.lock().value_str.clone())
    }

    /// Set a console variable from a string.
    pub fn set_var_string(&mut self, name: &str, value: &str) -> Result<(), String> {
        if let Some(var) = self.variables.get(name).cloned() {
            let mut entry = var.lock();
            match (entry.setter)(value) {
                Ok(new_val) => {
                    entry.value_str = new_val;
                    Ok(())
                }
                Err(e) => Err(e),
            }
        } else {
            Err(format!("Unknown variable: {}", name))
        }
    }

    // -- Command execution --------------------------------------------------

    /// Parse and execute a command string.
    ///
    /// Handles quoted strings, aliases, and command lookup.
    pub fn execute(&mut self, input: &str) {
        let input = input.trim();
        if input.is_empty() {
            return;
        }

        // Add to history.
        if self.history.last().map_or(true, |last| last != input) {
            self.history.push(input.to_string());
            if self.history.len() > self.max_history {
                self.history.remove(0);
            }
        }
        self.history_index = None;

        // Echo the command.
        self.output.push(ConsoleLine::new(
            format!("> {}", input),
            ConsoleCategory::Command,
        ));

        // Parse the command line.
        let tokens = Self::parse_tokens(input);
        if tokens.is_empty() {
            return;
        }

        let cmd_name = &tokens[0];
        let args: Vec<String> = tokens[1..].to_vec();

        // Check aliases first.
        if let Some(target) = self.aliases.get(cmd_name).cloned() {
            let expanded = if args.is_empty() {
                target
            } else {
                format!("{} {}", target, args.join(" "))
            };
            // Recursive execution of the alias target.
            self.execute(&expanded);
            return;
        }

        // Execute the command. We need to temporarily remove it from the map
        // to avoid borrowing issues (the callback takes `&mut Console`).
        if let Some(cmd) = self.commands.remove(cmd_name) {
            (cmd.callback)(&args, self);
            self.commands.insert(cmd.name.clone(), cmd);
        } else {
            self.print_error(&format!("Unknown command: {}", cmd_name));
        }

        // Trim output buffer.
        while self.output.len() > self.max_output_lines {
            self.output.remove(0);
        }
    }

    /// Parse a command string into tokens, handling quoted strings.
    fn parse_tokens(input: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut quote_char = '"';
        let mut escape_next = false;

        for ch in input.chars() {
            if escape_next {
                current.push(ch);
                escape_next = false;
                continue;
            }

            if ch == '\\' {
                escape_next = true;
                continue;
            }

            if in_quotes {
                if ch == quote_char {
                    in_quotes = false;
                } else {
                    current.push(ch);
                }
            } else {
                match ch {
                    '"' | '\'' => {
                        in_quotes = true;
                        quote_char = ch;
                    }
                    ' ' | '\t' => {
                        if !current.is_empty() {
                            tokens.push(std::mem::take(&mut current));
                        }
                    }
                    _ => {
                        current.push(ch);
                    }
                }
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    // -- Tab completion -----------------------------------------------------

    /// Compute tab completions for the current input.
    pub fn tab_complete(&mut self) -> Vec<String> {
        let input = self.input.trim().to_lowercase();
        if input.is_empty() {
            return Vec::new();
        }

        let tokens = Self::parse_tokens(&input);
        if tokens.is_empty() {
            return Vec::new();
        }

        // If we're still on the first token, complete command names.
        if tokens.len() <= 1 {
            let prefix = &tokens[0];
            let mut completions: Vec<String> = self
                .commands
                .keys()
                .filter(|name| name.starts_with(prefix))
                .cloned()
                .collect();
            completions.extend(
                self.aliases
                    .keys()
                    .filter(|name| name.starts_with(prefix))
                    .cloned(),
            );
            completions.sort();
            completions.dedup();
            self.tab_completions = completions.clone();
            self.tab_index = 0;
            return completions;
        }

        // If the command is "set" or "get", complete variable names.
        let cmd = &tokens[0];
        if (cmd == "set" || cmd == "get" || cmd == "reset_var") && tokens.len() == 2 {
            let prefix = &tokens[1];
            let mut completions: Vec<String> = self
                .variables
                .keys()
                .filter(|name| name.starts_with(prefix))
                .cloned()
                .collect();
            completions.sort();
            self.tab_completions = completions.clone();
            self.tab_index = 0;
            return completions;
        }

        Vec::new()
    }

    /// Cycle to the next tab completion and apply it to the input.
    pub fn next_tab_completion(&mut self) -> Option<String> {
        if self.tab_completions.is_empty() {
            self.tab_complete();
        }
        if self.tab_completions.is_empty() {
            return None;
        }

        let completion = self.tab_completions[self.tab_index % self.tab_completions.len()].clone();
        self.tab_index += 1;

        // Replace the last token with the completion.
        let tokens = Self::parse_tokens(&self.input);
        if tokens.len() <= 1 {
            self.input = completion.clone();
        } else {
            let prefix: Vec<&str> = tokens[..tokens.len() - 1].iter().map(|s| s.as_str()).collect();
            self.input = format!("{} {}", prefix.join(" "), completion);
        }
        self.cursor_pos = self.input.len();

        Some(completion)
    }

    // -- History navigation -------------------------------------------------

    /// Navigate to the previous command in history.
    pub fn history_previous(&mut self) -> Option<&str> {
        if self.history.is_empty() {
            return None;
        }
        let idx = match self.history_index {
            Some(0) => 0,
            Some(i) => i - 1,
            None => self.history.len() - 1,
        };
        self.history_index = Some(idx);
        self.input = self.history[idx].clone();
        self.cursor_pos = self.input.len();
        Some(&self.history[idx])
    }

    /// Navigate to the next command in history.
    pub fn history_next(&mut self) -> Option<&str> {
        match self.history_index {
            Some(i) if i + 1 < self.history.len() => {
                let next = i + 1;
                self.history_index = Some(next);
                self.input = self.history[next].clone();
                self.cursor_pos = self.input.len();
                Some(&self.history[next])
            }
            Some(_) => {
                self.history_index = None;
                self.input.clear();
                self.cursor_pos = 0;
                None
            }
            None => None,
        }
    }

    // -- Output methods -----------------------------------------------------

    /// Print an info-level message.
    pub fn print_info(&mut self, text: &str) {
        self.output.push(ConsoleLine::info(text));
    }

    /// Print a warning message.
    pub fn print_warning(&mut self, text: &str) {
        self.output.push(ConsoleLine::warning(text));
    }

    /// Print an error message.
    pub fn print_error(&mut self, text: &str) {
        self.output.push(ConsoleLine::error(text));
    }

    /// Print a debug message.
    pub fn print_debug(&mut self, text: &str) {
        self.output.push(ConsoleLine::debug(text));
    }

    /// Print a system message.
    pub fn print_system(&mut self, text: &str) {
        self.output.push(ConsoleLine::system(text));
    }

    /// Print a success message.
    pub fn print_success(&mut self, text: &str) {
        self.output
            .push(ConsoleLine::new(text, ConsoleCategory::Success));
    }

    /// Clear all output.
    pub fn clear_output(&mut self) {
        self.output.clear();
    }

    /// Get the output buffer.
    pub fn output(&self) -> &[ConsoleLine] {
        &self.output
    }

    /// Get the last N lines of output.
    pub fn last_n_lines(&self, n: usize) -> &[ConsoleLine] {
        let start = self.output.len().saturating_sub(n);
        &self.output[start..]
    }

    /// Get the number of output lines.
    pub fn output_line_count(&self) -> usize {
        self.output.len()
    }

    // -- Input management ---------------------------------------------------

    /// Get the current input string.
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Set the input string.
    pub fn set_input(&mut self, input: String) {
        self.cursor_pos = input.len();
        self.input = input;
        self.tab_completions.clear();
    }

    /// Insert a character at the cursor position.
    pub fn insert_char(&mut self, ch: char) {
        self.input.insert(self.cursor_pos, ch);
        self.cursor_pos += ch.len_utf8();
        self.tab_completions.clear();
    }

    /// Delete the character before the cursor (backspace).
    pub fn backspace(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
            self.input.remove(self.cursor_pos);
            self.tab_completions.clear();
        }
    }

    /// Delete the character at the cursor (delete key).
    pub fn delete_char(&mut self) {
        if self.cursor_pos < self.input.len() {
            self.input.remove(self.cursor_pos);
            self.tab_completions.clear();
        }
    }

    /// Move the cursor left.
    pub fn cursor_left(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
        }
    }

    /// Move the cursor right.
    pub fn cursor_right(&mut self) {
        if self.cursor_pos < self.input.len() {
            self.cursor_pos += 1;
        }
    }

    /// Move the cursor to the start of the line.
    pub fn cursor_home(&mut self) {
        self.cursor_pos = 0;
    }

    /// Move the cursor to the end of the line.
    pub fn cursor_end(&mut self) {
        self.cursor_pos = self.input.len();
    }

    /// Submit the current input (execute and clear).
    pub fn submit(&mut self) {
        let input = std::mem::take(&mut self.input);
        self.cursor_pos = 0;
        self.execute(&input);
    }

    // -- Visibility ---------------------------------------------------------

    /// Toggle console visibility.
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Set console visibility.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Check if the console is visible.
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    // -- Log integration ----------------------------------------------------

    /// Push a log message into the console (for log crate integration).
    pub fn log_message(&mut self, level: log::Level, message: &str) {
        let category = match level {
            log::Level::Error => ConsoleCategory::Error,
            log::Level::Warn => ConsoleCategory::Warning,
            log::Level::Info => ConsoleCategory::Info,
            log::Level::Debug => ConsoleCategory::Debug,
            log::Level::Trace => ConsoleCategory::Debug,
        };
        self.output.push(ConsoleLine::new(message, category));
    }

    // -- Command count / info -----------------------------------------------

    /// Get the number of registered commands.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Get the number of registered variables.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Get the history.
    pub fn command_history(&self) -> &[String] {
        &self.history
    }

    /// Get the number of aliases.
    pub fn alias_count(&self) -> usize {
        self.aliases.len()
    }
}

impl Default for Console {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Console {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Console")
            .field("commands", &self.commands.len())
            .field("variables", &self.variables.len())
            .field("output_lines", &self.output.len())
            .field("history_len", &self.history.len())
            .field("visible", &self.visible)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ConsoleLogger — log crate integration
// ---------------------------------------------------------------------------

/// A `log::Log` implementation that forwards messages to a shared [`Console`].
pub struct ConsoleLogger {
    console: Arc<Mutex<Console>>,
    max_level: log::LevelFilter,
}

impl ConsoleLogger {
    /// Create a new console logger wrapping the given console.
    pub fn new(console: Arc<Mutex<Console>>, max_level: log::LevelFilter) -> Self {
        Self { console, max_level }
    }
}

impl log::Log for ConsoleLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= self.max_level
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let msg = format!("[{}] {}", record.target(), record.args());
            self.console.lock().log_message(record.level(), &msg);
        }
    }

    fn flush(&self) {}
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_commands_registered() {
        let console = Console::new();
        assert!(console.has_command("help"));
        assert!(console.has_command("clear"));
        assert!(console.has_command("echo"));
        assert!(console.has_command("set"));
        assert!(console.has_command("get"));
        assert!(console.has_command("list_commands"));
        assert!(console.has_command("exec"));
        assert!(console.has_command("alias"));
        assert!(console.has_command("version"));
    }

    #[test]
    fn echo_command() {
        let mut console = Console::new();
        console.execute("echo Hello World");
        // Output should contain the echo'd line and the command echo.
        let lines: Vec<&str> = console.output().iter().map(|l| l.text.as_str()).collect();
        assert!(lines.iter().any(|l| *l == "Hello World"));
    }

    #[test]
    fn unknown_command_error() {
        let mut console = Console::new();
        console.execute("nonexistent_cmd");
        let last = console.output().last().unwrap();
        assert_eq!(last.category, ConsoleCategory::Error);
        assert!(last.text.contains("Unknown command"));
    }

    #[test]
    fn command_history() {
        let mut console = Console::new();
        console.execute("echo one");
        console.execute("echo two");
        console.execute("echo three");
        assert_eq!(console.command_history().len(), 3);
    }

    #[test]
    fn parse_quoted_strings() {
        let tokens = Console::parse_tokens(r#"echo "hello world" foo"#);
        assert_eq!(tokens, vec!["echo", "hello world", "foo"]);
    }

    #[test]
    fn parse_escaped_characters() {
        let tokens = Console::parse_tokens(r#"echo hello\ world"#);
        assert_eq!(tokens, vec!["echo", "hello world"]);
    }

    #[test]
    fn console_variable_registration() {
        let mut console = Console::new();
        let var = ConsoleVar::new("sv_gravity", 9.81_f32, "World gravity");
        console.register_var(var);
        assert_eq!(console.variable_count(), 1);
        assert_eq!(console.get_var_string("sv_gravity"), Some("9.81".into()));
    }

    #[test]
    fn set_variable_via_command() {
        let mut console = Console::new();
        let var = ConsoleVar::new("sv_gravity", 9.81_f32, "World gravity");
        console.register_var(var);
        console.execute("set sv_gravity 20.0");
        assert_eq!(console.get_var_string("sv_gravity"), Some("20".into()));
    }

    #[test]
    fn tab_completion() {
        let mut console = Console::new();
        console.set_input("he".into());
        let completions = console.tab_complete();
        assert!(completions.contains(&"help".to_string()));
    }

    #[test]
    fn alias_creation_and_execution() {
        let mut console = Console::new();
        console.execute("alias hi echo Hello");
        assert_eq!(console.alias_count(), 1);
        console.execute("hi");
        let lines: Vec<&str> = console.output().iter().map(|l| l.text.as_str()).collect();
        assert!(lines.iter().any(|l| *l == "Hello"));
    }

    #[test]
    fn clear_output() {
        let mut console = Console::new();
        console.print_info("test line");
        assert!(!console.output().is_empty());
        console.execute("clear");
        assert!(console.output().is_empty());
    }

    #[test]
    fn input_editing() {
        let mut console = Console::new();
        console.insert_char('a');
        console.insert_char('b');
        console.insert_char('c');
        assert_eq!(console.input(), "abc");
        console.backspace();
        assert_eq!(console.input(), "ab");
        console.cursor_left();
        console.insert_char('x');
        assert_eq!(console.input(), "axb");
    }

    #[test]
    fn custom_command_registration() {
        let mut console = Console::new();
        console.register_command(ConsoleCommand::new(
            "greet",
            "Say hello",
            vec![ArgType::String],
            |args, console| {
                let name = args.first().map(|s| s.as_str()).unwrap_or("World");
                console.print_info(&format!("Hello, {}!", name));
            },
        ));
        console.execute("greet Claude");
        let lines: Vec<&str> = console.output().iter().map(|l| l.text.as_str()).collect();
        assert!(lines.iter().any(|l| *l == "Hello, Claude!"));
    }

    #[test]
    fn history_navigation() {
        let mut console = Console::new();
        console.execute("echo one");
        console.execute("echo two");

        console.history_previous();
        assert_eq!(console.input(), "echo two");
        console.history_previous();
        assert_eq!(console.input(), "echo one");
        console.history_next();
        assert_eq!(console.input(), "echo two");
    }

    #[test]
    fn toggle_visibility() {
        let mut console = Console::new();
        assert!(!console.is_visible());
        console.toggle();
        assert!(console.is_visible());
        console.toggle();
        assert!(!console.is_visible());
    }

    #[test]
    fn log_integration() {
        let mut console = Console::new();
        console.log_message(log::Level::Error, "Something went wrong");
        let last = console.output().last().unwrap();
        assert_eq!(last.category, ConsoleCategory::Error);
        assert!(last.text.contains("Something went wrong"));
    }
}
