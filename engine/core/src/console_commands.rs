// engine/core/src/console_commands.rs
//
// Engine console commands: cvar system (typed console variables), command
// registration with help text, argument parsing, autocomplete, cvar persistence
// to config file, cvar change callbacks, built-in engine cvars.
//
// The console system provides an in-game developer console similar to Quake/
// Source engine consoles. Console variables (cvars) store named typed values
// that can be modified at runtime, and console commands execute registered
// callbacks with parsed arguments.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// CVar value
// ---------------------------------------------------------------------------

/// A typed console variable value.
#[derive(Debug, Clone, PartialEq)]
pub enum CvarValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl CvarValue {
    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(v) => *v,
            Self::Int(v) => *v != 0,
            Self::Float(v) => *v != 0.0,
            Self::String(v) => !v.is_empty() && v != "0" && v != "false",
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            Self::Bool(v) => if *v { 1 } else { 0 },
            Self::Int(v) => *v,
            Self::Float(v) => *v as i64,
            Self::String(v) => v.parse().unwrap_or(0),
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Self::Bool(v) => if *v { 1.0 } else { 0.0 },
            Self::Int(v) => *v as f64,
            Self::Float(v) => *v,
            Self::String(v) => v.parse().unwrap_or(0.0),
        }
    }

    pub fn as_string(&self) -> String {
        match self {
            Self::Bool(v) => v.to_string(),
            Self::Int(v) => v.to_string(),
            Self::Float(v) => format!("{:.6}", v),
            Self::String(v) => v.clone(),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::String(_) => "string",
        }
    }

    /// Parse a string into a CvarValue matching the type of this value.
    pub fn parse_like(&self, input: &str) -> Result<CvarValue, String> {
        match self {
            Self::Bool(_) => {
                match input.to_lowercase().as_str() {
                    "true" | "1" | "yes" | "on" => Ok(CvarValue::Bool(true)),
                    "false" | "0" | "no" | "off" => Ok(CvarValue::Bool(false)),
                    _ => Err(format!("Cannot parse '{}' as bool", input)),
                }
            }
            Self::Int(_) => {
                input.parse::<i64>()
                    .map(CvarValue::Int)
                    .map_err(|e| format!("Cannot parse '{}' as int: {}", input, e))
            }
            Self::Float(_) => {
                input.parse::<f64>()
                    .map(CvarValue::Float)
                    .map_err(|e| format!("Cannot parse '{}' as float: {}", input, e))
            }
            Self::String(_) => Ok(CvarValue::String(input.to_string())),
        }
    }
}

impl std::fmt::Display for CvarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_string())
    }
}

// ---------------------------------------------------------------------------
// CVar flags
// ---------------------------------------------------------------------------

/// Flags that control cvar behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CvarFlags {
    bits: u32,
}

impl CvarFlags {
    pub const NONE: Self = Self { bits: 0 };
    /// This cvar is saved to the config file.
    pub const ARCHIVE: Self = Self { bits: 1 << 0 };
    /// This cvar requires a restart to take effect.
    pub const REQUIRES_RESTART: Self = Self { bits: 1 << 1 };
    /// This cvar is read-only (cannot be changed by the user).
    pub const READ_ONLY: Self = Self { bits: 1 << 2 };
    /// This cvar is hidden from autocomplete and help.
    pub const HIDDEN: Self = Self { bits: 1 << 3 };
    /// This cvar is a cheat (only works in debug/dev mode).
    pub const CHEAT: Self = Self { bits: 1 << 4 };
    /// This cvar is replicated to clients in multiplayer.
    pub const REPLICATED: Self = Self { bits: 1 << 5 };
    /// This cvar is user-facing (shown in settings UI).
    pub const USER_SETTING: Self = Self { bits: 1 << 6 };

    pub fn contains(self, other: Self) -> bool {
        self.bits & other.bits == other.bits
    }

    pub fn union(self, other: Self) -> Self {
        Self { bits: self.bits | other.bits }
    }
}

impl std::ops::BitOr for CvarFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

// ---------------------------------------------------------------------------
// CVar
// ---------------------------------------------------------------------------

/// A console variable.
#[derive(Debug, Clone)]
pub struct Cvar {
    /// Name of the cvar.
    pub name: String,
    /// Description / help text.
    pub description: String,
    /// Current value.
    pub value: CvarValue,
    /// Default value.
    pub default_value: CvarValue,
    /// Minimum value (for numeric types).
    pub min_value: Option<f64>,
    /// Maximum value (for numeric types).
    pub max_value: Option<f64>,
    /// Allowed string values (for enum-like cvars).
    pub allowed_values: Vec<String>,
    /// Flags.
    pub flags: CvarFlags,
    /// Category for grouping in UI.
    pub category: String,
    /// Whether the value has been modified from default.
    pub modified: bool,
    /// Number of times the value has been changed.
    pub change_count: u32,
}

impl Cvar {
    pub fn new(name: &str, description: &str, default: CvarValue) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            value: default.clone(),
            default_value: default,
            min_value: None,
            max_value: None,
            allowed_values: Vec::new(),
            flags: CvarFlags::NONE,
            category: "General".to_string(),
            modified: false,
            change_count: 0,
        }
    }

    pub fn with_flags(mut self, flags: CvarFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    pub fn with_category(mut self, category: &str) -> Self {
        self.category = category.to_string();
        self
    }

    pub fn with_allowed_values(mut self, values: Vec<String>) -> Self {
        self.allowed_values = values;
        self
    }

    /// Set the value, clamping to range if applicable.
    pub fn set(&mut self, value: CvarValue) -> Result<CvarValue, String> {
        if self.flags.contains(CvarFlags::READ_ONLY) {
            return Err(format!("'{}' is read-only", self.name));
        }

        // Validate allowed values.
        if !self.allowed_values.is_empty() {
            let s = value.as_string();
            if !self.allowed_values.contains(&s) {
                return Err(format!("'{}' is not a valid value for '{}'. Allowed: {:?}",
                    s, self.name, self.allowed_values));
            }
        }

        // Clamp numeric values.
        let clamped = match &value {
            CvarValue::Int(v) => {
                let mut val = *v as f64;
                if let Some(min) = self.min_value { val = val.max(min); }
                if let Some(max) = self.max_value { val = val.min(max); }
                CvarValue::Int(val as i64)
            }
            CvarValue::Float(v) => {
                let mut val = *v;
                if let Some(min) = self.min_value { val = val.max(min); }
                if let Some(max) = self.max_value { val = val.min(max); }
                CvarValue::Float(val)
            }
            other => other.clone(),
        };

        let old_value = self.value.clone();
        self.value = clamped.clone();
        self.modified = self.value != self.default_value;
        self.change_count += 1;

        Ok(old_value)
    }

    /// Reset to default value.
    pub fn reset(&mut self) {
        self.value = self.default_value.clone();
        self.modified = false;
    }

    /// Set from a string representation.
    pub fn set_from_string(&mut self, input: &str) -> Result<CvarValue, String> {
        let parsed = self.default_value.parse_like(input)?;
        self.set(parsed)
    }
}

// ---------------------------------------------------------------------------
// Console command
// ---------------------------------------------------------------------------

/// A parsed command argument.
#[derive(Debug, Clone)]
pub enum CommandArg {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl CommandArg {
    pub fn as_str(&self) -> &str {
        match self {
            Self::String(s) => s.as_str(),
            _ => "",
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            Self::Int(v) => *v,
            Self::Float(v) => *v as i64,
            Self::String(s) => s.parse().unwrap_or(0),
            Self::Bool(v) => if *v { 1 } else { 0 },
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Self::Float(v) => *v,
            Self::Int(v) => *v as f64,
            Self::String(s) => s.parse().unwrap_or(0.0),
            Self::Bool(v) => if *v { 1.0 } else { 0.0 },
        }
    }
}

/// Parse command line arguments from a string.
pub fn parse_args(input: &str) -> Vec<CommandArg> {
    let mut args = Vec::new();
    let mut in_quotes = false;
    let mut current = String::new();

    for ch in input.chars() {
        if ch == '"' {
            in_quotes = !in_quotes;
            continue;
        }
        if ch == ' ' && !in_quotes {
            if !current.is_empty() {
                args.push(classify_arg(&current));
                current.clear();
            }
            continue;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        args.push(classify_arg(&current));
    }

    args
}

fn classify_arg(s: &str) -> CommandArg {
    if let Ok(v) = s.parse::<i64>() { return CommandArg::Int(v); }
    if let Ok(v) = s.parse::<f64>() { return CommandArg::Float(v); }
    match s.to_lowercase().as_str() {
        "true" | "yes" | "on" => CommandArg::Bool(true),
        "false" | "no" | "off" => CommandArg::Bool(false),
        _ => CommandArg::String(s.to_string()),
    }
}

/// The execution context for a console command.
pub struct CommandContext<'a> {
    pub args: &'a [CommandArg],
    pub raw_input: &'a str,
    pub output: Vec<String>,
}

impl<'a> CommandContext<'a> {
    pub fn new(args: &'a [CommandArg], raw_input: &'a str) -> Self {
        Self { args, raw_input, output: Vec::new() }
    }

    pub fn print(&mut self, msg: &str) { self.output.push(msg.to_string()); }
    pub fn arg_count(&self) -> usize { self.args.len() }

    pub fn arg_str(&self, index: usize) -> Option<&str> {
        self.args.get(index).map(|a| match a {
            CommandArg::String(s) => s.as_str(),
            _ => "",
        })
    }

    pub fn arg_int(&self, index: usize) -> Option<i64> {
        self.args.get(index).map(|a| a.as_int())
    }

    pub fn arg_float(&self, index: usize) -> Option<f64> {
        self.args.get(index).map(|a| a.as_float())
    }
}

/// A registered console command.
pub struct ConsoleCommand {
    pub name: String,
    pub description: String,
    pub usage: String,
    pub min_args: usize,
    pub max_args: usize,
    pub flags: CvarFlags,
    pub callback: Box<dyn Fn(&mut CommandContext) + Send + Sync>,
}

// ---------------------------------------------------------------------------
// Console system
// ---------------------------------------------------------------------------

/// Output from executing a console command.
#[derive(Debug, Clone)]
pub struct ConsoleOutput {
    pub lines: Vec<String>,
    pub is_error: bool,
}

/// The console system: manages cvars, commands, and command execution.
pub struct ConsoleSystem {
    /// Registered cvars.
    cvars: HashMap<String, Cvar>,
    /// Registered commands.
    commands: HashMap<String, ConsoleCommand>,
    /// Command history.
    history: Vec<String>,
    /// Maximum history size.
    pub max_history: usize,
    /// Output log.
    output_log: Vec<ConsoleOutput>,
    /// Maximum output log size.
    pub max_output_log: usize,
    /// CVar change callbacks: cvar_name -> callbacks.
    change_callbacks: HashMap<String, Vec<Box<dyn Fn(&Cvar, &CvarValue) + Send + Sync>>>,
    /// Config file path for cvar persistence.
    pub config_path: Option<PathBuf>,
    /// Whether cheats are enabled.
    pub cheats_enabled: bool,
    /// Command aliases.
    aliases: HashMap<String, String>,
}

impl ConsoleSystem {
    pub fn new() -> Self {
        let mut system = Self {
            cvars: HashMap::new(),
            commands: HashMap::new(),
            history: Vec::new(),
            max_history: 100,
            output_log: Vec::new(),
            max_output_log: 500,
            change_callbacks: HashMap::new(),
            config_path: None,
            cheats_enabled: cfg!(debug_assertions),
            aliases: HashMap::new(),
        };
        system.register_builtin_commands();
        system.register_builtin_cvars();
        system
    }

    /// Register a console variable.
    pub fn register_cvar(&mut self, cvar: Cvar) {
        self.cvars.insert(cvar.name.clone(), cvar);
    }

    /// Register a console command.
    pub fn register_command(&mut self, cmd: ConsoleCommand) {
        self.commands.insert(cmd.name.clone(), cmd);
    }

    /// Register a cvar change callback.
    pub fn on_cvar_changed<F>(&mut self, name: &str, callback: F)
    where
        F: Fn(&Cvar, &CvarValue) + Send + Sync + 'static,
    {
        self.change_callbacks.entry(name.to_string())
            .or_default()
            .push(Box::new(callback));
    }

    /// Get a cvar value.
    pub fn get_cvar(&self, name: &str) -> Option<&Cvar> {
        self.cvars.get(name)
    }

    /// Get a cvar value as a specific type.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.cvars.get(name).map(|c| c.value.as_bool())
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.cvars.get(name).map(|c| c.value.as_int())
    }

    pub fn get_float(&self, name: &str) -> Option<f64> {
        self.cvars.get(name).map(|c| c.value.as_float())
    }

    pub fn get_string(&self, name: &str) -> Option<String> {
        self.cvars.get(name).map(|c| c.value.as_string())
    }

    /// Set a cvar value.
    pub fn set_cvar(&mut self, name: &str, value: CvarValue) -> Result<(), String> {
        let cvar = self.cvars.get_mut(name).ok_or(format!("Unknown cvar: {}", name))?;

        if cvar.flags.contains(CvarFlags::CHEAT) && !self.cheats_enabled {
            return Err(format!("'{}' is a cheat cvar (enable cheats first)", name));
        }

        let old_value = cvar.set(value.clone())?;
        let _ = old_value;

        // Fire change callbacks.
        if let Some(callbacks) = self.change_callbacks.get(name) {
            let cvar_ref = self.cvars.get(name).unwrap();
            for cb in callbacks {
                cb(cvar_ref, &value);
            }
        }

        Ok(())
    }

    /// Set a cvar from a string.
    pub fn set_cvar_string(&mut self, name: &str, input: &str) -> Result<(), String> {
        let cvar = self.cvars.get(name).ok_or(format!("Unknown cvar: {}", name))?;
        let parsed = cvar.default_value.parse_like(input)?;
        self.set_cvar(name, parsed)
    }

    /// Execute a console command string.
    pub fn execute(&mut self, input: &str) -> ConsoleOutput {
        let input = input.trim();
        if input.is_empty() {
            return ConsoleOutput { lines: Vec::new(), is_error: false };
        }

        // Add to history.
        self.history.push(input.to_string());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        // Check for alias.
        let resolved = if let Some(alias) = self.aliases.get(input.split_whitespace().next().unwrap_or("")) {
            let rest = input.splitn(2, ' ').nth(1).unwrap_or("");
            if rest.is_empty() { alias.clone() } else { format!("{} {}", alias, rest) }
        } else {
            input.to_string()
        };

        // Split into command name and arguments.
        let parts: Vec<&str> = resolved.splitn(2, ' ').collect();
        let cmd_name = parts[0];
        let arg_str = if parts.len() > 1 { parts[1] } else { "" };

        // Check if it's a cvar query/set.
        if let Some(cvar) = self.cvars.get(cmd_name) {
            if arg_str.is_empty() {
                // Query: print current value.
                let output = ConsoleOutput {
                    lines: vec![
                        format!("\"{}\" = \"{}\" (default: \"{}\") - {}",
                            cvar.name, cvar.value, cvar.default_value, cvar.description),
                    ],
                    is_error: false,
                };
                self.push_output(&output);
                return output;
            } else {
                // Set value.
                match self.set_cvar_string(cmd_name, arg_str) {
                    Ok(()) => {
                        let cvar = self.cvars.get(cmd_name).unwrap();
                        let output = ConsoleOutput {
                            lines: vec![format!("\"{}\" set to \"{}\"", cvar.name, cvar.value)],
                            is_error: false,
                        };
                        self.push_output(&output);
                        return output;
                    }
                    Err(e) => {
                        let output = ConsoleOutput { lines: vec![e], is_error: true };
                        self.push_output(&output);
                        return output;
                    }
                }
            }
        }

        // Check if it's a registered command.
        let args = parse_args(arg_str);
        if self.commands.contains_key(cmd_name) {
            let cmd = self.commands.get(cmd_name).unwrap();

            if args.len() < cmd.min_args {
                let output = ConsoleOutput {
                    lines: vec![
                        format!("Too few arguments for '{}'. Usage: {}", cmd_name, cmd.usage),
                    ],
                    is_error: true,
                };
                self.push_output(&output);
                return output;
            }

            let mut ctx = CommandContext::new(&args, &resolved);
            (cmd.callback)(&mut ctx);

            let output = ConsoleOutput {
                lines: ctx.output,
                is_error: false,
            };
            self.push_output(&output);
            return output;
        }

        // Unknown command.
        let output = ConsoleOutput {
            lines: vec![format!("Unknown command or cvar: '{}'", cmd_name)],
            is_error: true,
        };
        self.push_output(&output);
        output
    }

    /// Autocomplete a partial input string.
    pub fn autocomplete(&self, partial: &str) -> Vec<String> {
        let partial_lower = partial.to_lowercase();
        let mut results = Vec::new();

        // Match cvars.
        for name in self.cvars.keys() {
            if name.to_lowercase().starts_with(&partial_lower) {
                results.push(name.clone());
            }
        }

        // Match commands.
        for name in self.commands.keys() {
            if name.to_lowercase().starts_with(&partial_lower) {
                results.push(name.clone());
            }
        }

        // Match aliases.
        for name in self.aliases.keys() {
            if name.to_lowercase().starts_with(&partial_lower) {
                results.push(name.clone());
            }
        }

        results.sort();
        results
    }

    /// Get the command history.
    pub fn history(&self) -> &[String] { &self.history }

    /// Get the output log.
    pub fn output_log(&self) -> &[ConsoleOutput] { &self.output_log }

    /// Clear the output log.
    pub fn clear_output(&mut self) { self.output_log.clear(); }

    /// Register an alias.
    pub fn register_alias(&mut self, name: &str, expansion: &str) {
        self.aliases.insert(name.to_string(), expansion.to_string());
    }

    /// Save all ARCHIVE cvars to a config file.
    pub fn save_config(&self) -> Result<(), String> {
        let path = self.config_path.as_ref().ok_or("No config path set")?;
        let mut content = String::new();
        content.push_str("// Engine console variables\n");
        content.push_str("// Auto-generated -- do not edit manually\n\n");

        let mut archive_cvars: Vec<&Cvar> = self.cvars.values()
            .filter(|c| c.flags.contains(CvarFlags::ARCHIVE))
            .collect();
        archive_cvars.sort_by(|a, b| a.name.cmp(&b.name));

        for cvar in archive_cvars {
            content.push_str(&format!("{} \"{}\"\n", cvar.name, cvar.value.as_string()));
        }

        std::fs::write(path, &content).map_err(|e| e.to_string())
    }

    /// Load cvars from a config file.
    pub fn load_config(&mut self) -> Result<u32, String> {
        let path = self.config_path.as_ref().ok_or("No config path set")?.clone();
        let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let mut loaded = 0u32;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") { continue; }
            self.execute(line);
            loaded += 1;
        }

        Ok(loaded)
    }

    /// Get all cvars in a category.
    pub fn cvars_in_category(&self, category: &str) -> Vec<&Cvar> {
        self.cvars.values().filter(|c| c.category == category).collect()
    }

    /// Get all cvar categories.
    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self.cvars.values()
            .map(|c| c.category.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        cats.sort();
        cats
    }

    // -----------------------------------------------------------------------
    // Internal: built-in registrations
    // -----------------------------------------------------------------------

    fn register_builtin_commands(&mut self) {
        self.register_command(ConsoleCommand {
            name: "help".to_string(),
            description: "List all commands and cvars".to_string(),
            usage: "help [command_name]".to_string(),
            min_args: 0,
            max_args: 1,
            flags: CvarFlags::NONE,
            callback: Box::new(|ctx| {
                ctx.print("Available commands:");
                ctx.print("  help - List all commands");
                ctx.print("  echo <text> - Print text");
                ctx.print("  clear - Clear console output");
                ctx.print("  quit - Quit the engine");
                ctx.print("  cvarlist - List all cvars");
            }),
        });

        self.register_command(ConsoleCommand {
            name: "echo".to_string(),
            description: "Print text to the console".to_string(),
            usage: "echo <text>".to_string(),
            min_args: 1,
            max_args: 100,
            flags: CvarFlags::NONE,
            callback: Box::new(|ctx| {
                ctx.print(ctx.raw_input.trim_start_matches("echo "));
            }),
        });

        self.register_command(ConsoleCommand {
            name: "clear".to_string(),
            description: "Clear the console output".to_string(),
            usage: "clear".to_string(),
            min_args: 0,
            max_args: 0,
            flags: CvarFlags::NONE,
            callback: Box::new(|ctx| {
                ctx.print("[Console cleared]");
            }),
        });

        self.register_command(ConsoleCommand {
            name: "cvarlist".to_string(),
            description: "List all console variables".to_string(),
            usage: "cvarlist [filter]".to_string(),
            min_args: 0,
            max_args: 1,
            flags: CvarFlags::NONE,
            callback: Box::new(|ctx| {
                ctx.print("Use 'help' for command listing.");
            }),
        });
    }

    fn register_builtin_cvars(&mut self) {
        self.register_cvar(
            Cvar::new("r_resolution_scale", "Render resolution scale", CvarValue::Float(1.0))
                .with_range(0.25, 2.0)
                .with_flags(CvarFlags::ARCHIVE | CvarFlags::USER_SETTING)
                .with_category("Rendering"),
        );
        self.register_cvar(
            Cvar::new("r_vsync", "Enable vertical sync", CvarValue::Bool(true))
                .with_flags(CvarFlags::ARCHIVE | CvarFlags::USER_SETTING)
                .with_category("Rendering"),
        );
        self.register_cvar(
            Cvar::new("r_max_fps", "Maximum frame rate (0 = unlimited)", CvarValue::Int(0))
                .with_range(0.0, 1000.0)
                .with_flags(CvarFlags::ARCHIVE | CvarFlags::USER_SETTING)
                .with_category("Rendering"),
        );
        self.register_cvar(
            Cvar::new("r_shadow_quality", "Shadow map quality", CvarValue::Int(2))
                .with_range(0.0, 4.0)
                .with_flags(CvarFlags::ARCHIVE | CvarFlags::USER_SETTING)
                .with_category("Rendering"),
        );
        self.register_cvar(
            Cvar::new("s_master_volume", "Master audio volume", CvarValue::Float(1.0))
                .with_range(0.0, 1.0)
                .with_flags(CvarFlags::ARCHIVE | CvarFlags::USER_SETTING)
                .with_category("Audio"),
        );
        self.register_cvar(
            Cvar::new("s_music_volume", "Music volume", CvarValue::Float(0.7))
                .with_range(0.0, 1.0)
                .with_flags(CvarFlags::ARCHIVE | CvarFlags::USER_SETTING)
                .with_category("Audio"),
        );
        self.register_cvar(
            Cvar::new("s_sfx_volume", "Sound effects volume", CvarValue::Float(1.0))
                .with_range(0.0, 1.0)
                .with_flags(CvarFlags::ARCHIVE | CvarFlags::USER_SETTING)
                .with_category("Audio"),
        );
        self.register_cvar(
            Cvar::new("g_fov", "Field of view (degrees)", CvarValue::Float(90.0))
                .with_range(60.0, 120.0)
                .with_flags(CvarFlags::ARCHIVE | CvarFlags::USER_SETTING)
                .with_category("Gameplay"),
        );
        self.register_cvar(
            Cvar::new("g_mouse_sensitivity", "Mouse sensitivity", CvarValue::Float(1.0))
                .with_range(0.1, 10.0)
                .with_flags(CvarFlags::ARCHIVE | CvarFlags::USER_SETTING)
                .with_category("Gameplay"),
        );
        self.register_cvar(
            Cvar::new("net_tickrate", "Server tick rate", CvarValue::Int(60))
                .with_range(10.0, 128.0)
                .with_flags(CvarFlags::ARCHIVE)
                .with_category("Network"),
        );
        self.register_cvar(
            Cvar::new("debug_show_fps", "Show FPS counter", CvarValue::Bool(false))
                .with_flags(CvarFlags::ARCHIVE)
                .with_category("Debug"),
        );
        self.register_cvar(
            Cvar::new("debug_physics", "Show physics debug visualization", CvarValue::Bool(false))
                .with_flags(CvarFlags::CHEAT)
                .with_category("Debug"),
        );
        self.register_cvar(
            Cvar::new("engine_name", "Engine name", CvarValue::String("Genovo".to_string()))
                .with_flags(CvarFlags::READ_ONLY)
                .with_category("Engine"),
        );
        self.register_cvar(
            Cvar::new("engine_version", "Engine version", CvarValue::String("0.1.0".to_string()))
                .with_flags(CvarFlags::READ_ONLY)
                .with_category("Engine"),
        );
    }

    fn push_output(&mut self, output: &ConsoleOutput) {
        self.output_log.push(output.clone());
        while self.output_log.len() > self.max_output_log {
            self.output_log.remove(0);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvar_value_types() {
        assert_eq!(CvarValue::Bool(true).as_bool(), true);
        assert_eq!(CvarValue::Int(42).as_int(), 42);
        assert!((CvarValue::Float(3.14).as_float() - 3.14).abs() < 1e-6);
        assert_eq!(CvarValue::String("hello".into()).as_string(), "hello");
    }

    #[test]
    fn test_cvar_range_clamping() {
        let mut cvar = Cvar::new("test", "test", CvarValue::Float(0.5)).with_range(0.0, 1.0);
        cvar.set(CvarValue::Float(5.0)).unwrap();
        assert!((cvar.value.as_float() - 1.0).abs() < 1e-6);
        cvar.set(CvarValue::Float(-1.0)).unwrap();
        assert!((cvar.value.as_float() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cvar_read_only() {
        let mut cvar = Cvar::new("ro", "read only", CvarValue::Int(1)).with_flags(CvarFlags::READ_ONLY);
        assert!(cvar.set(CvarValue::Int(2)).is_err());
    }

    #[test]
    fn test_parse_args() {
        let args = parse_args("42 3.14 hello \"world of war\"");
        assert_eq!(args.len(), 4);
        assert_eq!(args[0].as_int(), 42);
        assert!((args[1].as_float() - 3.14).abs() < 0.01);
    }

    #[test]
    fn test_console_execute_cvar() {
        let mut console = ConsoleSystem::new();
        // Query a built-in cvar.
        let output = console.execute("r_vsync");
        assert!(!output.is_error);
        assert!(!output.lines.is_empty());

        // Set a cvar.
        let output = console.execute("r_vsync false");
        assert!(!output.is_error);
        assert_eq!(console.get_bool("r_vsync"), Some(false));
    }

    #[test]
    fn test_autocomplete() {
        let console = ConsoleSystem::new();
        let results = console.autocomplete("r_");
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.starts_with("r_"));
        }
    }

    #[test]
    fn test_execute_command() {
        let mut console = ConsoleSystem::new();
        let output = console.execute("help");
        assert!(!output.is_error);
        assert!(!output.lines.is_empty());
    }

    #[test]
    fn test_unknown_command() {
        let mut console = ConsoleSystem::new();
        let output = console.execute("nonexistent_command");
        assert!(output.is_error);
    }

    #[test]
    fn test_aliases() {
        let mut console = ConsoleSystem::new();
        console.register_alias("fps", "debug_show_fps true");
        let output = console.execute("fps");
        assert_eq!(console.get_bool("debug_show_fps"), Some(true));
    }
}
