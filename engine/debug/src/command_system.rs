//! Extended Console Command System
//!
//! Provides a comprehensive command framework for the developer console,
//! including:
//!
//! - Command registration with categories and aliases
//! - Built-in cheat commands (god mode, noclip, fly, give, teleport)
//! - Key-to-command binding
//! - Command macros (execute multiple commands in sequence)
//! - Config file execution (`exec autoexec.cfg`)
//! - Command history and auto-completion
//!
//! # Architecture
//!
//! ```text
//! CommandSystem
//!   +-- CommandRegistry  (name -> CommandDef)
//!   +-- AliasRegistry    (alias -> command)
//!   +-- KeyBindings      (key -> command string)
//!   +-- MacroRegistry    (name -> [commands])
//!   +-- CommandHistory    (ring buffer)
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// CommandId
// ---------------------------------------------------------------------------

/// Unique identifier for a registered command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommandId {
    pub index: u32,
}

impl CommandId {
    pub fn from_raw(index: u32) -> Self {
        Self { index }
    }
}

impl fmt::Display for CommandId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CommandId({})", self.index)
    }
}

// ---------------------------------------------------------------------------
// CommandCategory
// ---------------------------------------------------------------------------

/// Category for organizing commands in the console.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CommandCategory {
    /// General-purpose commands.
    General,
    /// Debug and development commands.
    Debug,
    /// Cheat/testing commands.
    Cheat,
    /// Rendering/graphics commands.
    Rendering,
    /// Audio commands.
    Audio,
    /// Physics commands.
    Physics,
    /// Networking commands.
    Network,
    /// UI commands.
    UI,
    /// Editor commands.
    Editor,
    /// System/engine commands.
    System,
    /// Custom category.
    Custom(String),
}

impl fmt::Display for CommandCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::General => write!(f, "General"),
            Self::Debug => write!(f, "Debug"),
            Self::Cheat => write!(f, "Cheat"),
            Self::Rendering => write!(f, "Rendering"),
            Self::Audio => write!(f, "Audio"),
            Self::Physics => write!(f, "Physics"),
            Self::Network => write!(f, "Network"),
            Self::UI => write!(f, "UI"),
            Self::Editor => write!(f, "Editor"),
            Self::System => write!(f, "System"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

impl Default for CommandCategory {
    fn default() -> Self {
        Self::General
    }
}

// ---------------------------------------------------------------------------
// CommandArg
// ---------------------------------------------------------------------------

/// A parsed command argument.
#[derive(Debug, Clone)]
pub enum CommandArg {
    /// A string argument.
    String(String),
    /// An integer argument.
    Int(i64),
    /// A floating-point argument.
    Float(f64),
    /// A boolean argument.
    Bool(bool),
}

impl CommandArg {
    /// Attempts to parse the argument as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Attempts to parse the argument as an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            Self::String(s) => s.parse().ok(),
            Self::Bool(b) => Some(if *b { 1 } else { 0 }),
        }
    }

    /// Attempts to parse the argument as a float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            Self::String(s) => s.parse().ok(),
            Self::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        }
    }

    /// Attempts to parse the argument as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::Int(v) => Some(*v != 0),
            Self::Float(v) => Some(*v != 0.0),
            Self::String(s) => match s.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Some(true),
                "false" | "0" | "no" | "off" => Some(false),
                _ => None,
            },
        }
    }

    /// Returns the argument as a display string.
    pub fn to_display_string(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Int(v) => v.to_string(),
            Self::Float(v) => format!("{:.4}", v),
            Self::Bool(v) => v.to_string(),
        }
    }
}

impl fmt::Display for CommandArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_display_string())
    }
}

/// Parses a raw string argument into the appropriate typed argument.
fn parse_arg(raw: &str) -> CommandArg {
    // Try boolean.
    match raw.to_lowercase().as_str() {
        "true" | "yes" | "on" => return CommandArg::Bool(true),
        "false" | "no" | "off" => return CommandArg::Bool(false),
        _ => {}
    }

    // Try integer.
    if let Ok(v) = raw.parse::<i64>() {
        return CommandArg::Int(v);
    }

    // Try float.
    if let Ok(v) = raw.parse::<f64>() {
        return CommandArg::Float(v);
    }

    // Default to string.
    CommandArg::String(raw.to_string())
}

// ---------------------------------------------------------------------------
// CommandArgDef
// ---------------------------------------------------------------------------

/// Definition of a command argument for help/validation.
#[derive(Debug, Clone)]
pub struct CommandArgDef {
    /// Name of the argument.
    pub name: String,
    /// Description of the argument.
    pub description: String,
    /// Whether this argument is required.
    pub required: bool,
    /// Default value if not provided.
    pub default_value: Option<String>,
    /// Expected type hint.
    pub type_hint: ArgTypeHint,
}

/// Type hint for command argument validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgTypeHint {
    String,
    Int,
    Float,
    Bool,
    EntityId,
    FilePath,
    Any,
}

impl fmt::Display for ArgTypeHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::Int => write!(f, "int"),
            Self::Float => write!(f, "float"),
            Self::Bool => write!(f, "bool"),
            Self::EntityId => write!(f, "entity_id"),
            Self::FilePath => write!(f, "file_path"),
            Self::Any => write!(f, "any"),
        }
    }
}

impl CommandArgDef {
    /// Creates a required argument definition.
    pub fn required(name: impl Into<String>, description: impl Into<String>, type_hint: ArgTypeHint) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: true,
            default_value: None,
            type_hint,
        }
    }

    /// Creates an optional argument definition with a default value.
    pub fn optional(
        name: impl Into<String>,
        description: impl Into<String>,
        type_hint: ArgTypeHint,
        default: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: false,
            default_value: Some(default.into()),
            type_hint,
        }
    }
}

// ---------------------------------------------------------------------------
// CommandResult
// ---------------------------------------------------------------------------

/// Result of executing a command.
#[derive(Debug, Clone)]
pub enum CommandResult {
    /// Command executed successfully with an optional output message.
    Success(Option<String>),
    /// Command failed with an error message.
    Error(String),
    /// Command produced multiple lines of output.
    Output(Vec<String>),
    /// Command requests the console to print help text.
    Help(String),
}

impl CommandResult {
    /// Creates a success result with no message.
    pub fn ok() -> Self {
        Self::Success(None)
    }

    /// Creates a success result with a message.
    pub fn message(msg: impl Into<String>) -> Self {
        Self::Success(Some(msg.into()))
    }

    /// Creates an error result.
    pub fn error(msg: impl Into<String>) -> Self {
        Self::Error(msg.into())
    }

    /// Returns `true` if the command succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_) | Self::Output(_) | Self::Help(_))
    }

    /// Returns `true` if the command failed.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Returns the output lines.
    pub fn lines(&self) -> Vec<String> {
        match self {
            Self::Success(Some(msg)) => vec![msg.clone()],
            Self::Success(None) => Vec::new(),
            Self::Error(msg) => vec![format!("ERROR: {}", msg)],
            Self::Output(lines) => lines.clone(),
            Self::Help(text) => text.lines().map(|l| l.to_string()).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// CommandDef
// ---------------------------------------------------------------------------

/// Definition of a registered command.
pub struct CommandDef {
    /// Unique command identifier.
    pub id: CommandId,
    /// The primary name of the command.
    pub name: String,
    /// Short description.
    pub description: String,
    /// Long help text.
    pub help_text: String,
    /// Category for organization.
    pub category: CommandCategory,
    /// Whether this command requires cheats to be enabled.
    pub requires_cheats: bool,
    /// Whether this command is hidden from help/autocomplete.
    pub hidden: bool,
    /// Argument definitions.
    pub args: Vec<CommandArgDef>,
    /// The command handler function.
    handler: Box<dyn Fn(&[CommandArg]) -> CommandResult + Send + Sync>,
}

impl CommandDef {
    /// Creates a new command definition.
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        category: CommandCategory,
        handler: F,
    ) -> Self
    where
        F: Fn(&[CommandArg]) -> CommandResult + Send + Sync + 'static,
    {
        static NEXT_CMD_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
        Self {
            id: CommandId::from_raw(NEXT_CMD_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
            name: name.into(),
            description: description.into(),
            help_text: String::new(),
            category,
            requires_cheats: false,
            hidden: false,
            args: Vec::new(),
            handler: Box::new(handler),
        }
    }

    /// Sets the help text.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help_text = help.into();
        self
    }

    /// Marks this command as requiring cheats.
    pub fn with_cheats_required(mut self) -> Self {
        self.requires_cheats = true;
        self
    }

    /// Marks this command as hidden.
    pub fn with_hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Adds an argument definition.
    pub fn with_arg(mut self, arg: CommandArgDef) -> Self {
        self.args.push(arg);
        self
    }

    /// Executes the command with the given arguments.
    pub fn execute(&self, args: &[CommandArg]) -> CommandResult {
        (self.handler)(args)
    }

    /// Returns the usage string for this command.
    pub fn usage(&self) -> String {
        let mut parts = vec![self.name.clone()];
        for arg in &self.args {
            if arg.required {
                parts.push(format!("<{}>", arg.name));
            } else {
                parts.push(format!("[{}]", arg.name));
            }
        }
        parts.join(" ")
    }
}

impl fmt::Debug for CommandDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandDef")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("category", &self.category)
            .field("requires_cheats", &self.requires_cheats)
            .field("hidden", &self.hidden)
            .field("args", &self.args)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// KeyBinding
// ---------------------------------------------------------------------------

/// A key-to-command binding.
#[derive(Debug, Clone)]
pub struct KeyBinding {
    /// The key code or key name.
    pub key: String,
    /// Modifier keys required (shift, ctrl, alt).
    pub modifiers: KeyModifiers,
    /// The command string to execute when the key is pressed.
    pub command: String,
    /// Whether this binding is active.
    pub active: bool,
    /// Optional description for the binding.
    pub description: String,
}

/// Modifier key flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KeyModifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

impl KeyModifiers {
    pub const NONE: Self = Self {
        shift: false,
        ctrl: false,
        alt: false,
    };

    pub const SHIFT: Self = Self {
        shift: true,
        ctrl: false,
        alt: false,
    };

    pub const CTRL: Self = Self {
        shift: false,
        ctrl: true,
        alt: false,
    };

    pub const ALT: Self = Self {
        shift: false,
        ctrl: false,
        alt: true,
    };

    pub const CTRL_SHIFT: Self = Self {
        shift: true,
        ctrl: true,
        alt: false,
    };

    /// Returns `true` if no modifiers are pressed.
    pub fn is_none(&self) -> bool {
        !self.shift && !self.ctrl && !self.alt
    }

    /// Checks whether the given modifiers match.
    pub fn matches(&self, other: &KeyModifiers) -> bool {
        self.shift == other.shift && self.ctrl == other.ctrl && self.alt == other.alt
    }
}

impl fmt::Display for KeyModifiers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.ctrl {
            parts.push("Ctrl");
        }
        if self.shift {
            parts.push("Shift");
        }
        if self.alt {
            parts.push("Alt");
        }
        write!(f, "{}", parts.join("+"))
    }
}

impl KeyBinding {
    /// Creates a new key binding.
    pub fn new(
        key: impl Into<String>,
        modifiers: KeyModifiers,
        command: impl Into<String>,
    ) -> Self {
        Self {
            key: key.into(),
            modifiers,
            command: command.into(),
            active: true,
            description: String::new(),
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Returns the display string for this binding (e.g., "Ctrl+Shift+F5").
    pub fn display_string(&self) -> String {
        let mods = self.modifiers.to_string();
        if mods.is_empty() {
            self.key.clone()
        } else {
            format!("{}+{}", mods, self.key)
        }
    }
}

// ---------------------------------------------------------------------------
// CommandMacro
// ---------------------------------------------------------------------------

/// A macro that executes multiple commands in sequence.
#[derive(Debug, Clone)]
pub struct CommandMacro {
    /// Name of the macro.
    pub name: String,
    /// Description.
    pub description: String,
    /// Commands to execute in order.
    pub commands: Vec<String>,
    /// Whether to stop on first error.
    pub stop_on_error: bool,
}

impl CommandMacro {
    /// Creates a new command macro.
    pub fn new(name: impl Into<String>, commands: Vec<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            commands,
            stop_on_error: true,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Sets whether to stop on the first error.
    pub fn with_stop_on_error(mut self, stop: bool) -> Self {
        self.stop_on_error = stop;
        self
    }

    /// Returns the number of commands in this macro.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Returns `true` if the macro is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

// ---------------------------------------------------------------------------
// CommandHistoryEntry
// ---------------------------------------------------------------------------

/// An entry in the command history.
#[derive(Debug, Clone)]
pub struct CommandHistoryEntry {
    /// The raw command string that was executed.
    pub command_string: String,
    /// The result of execution.
    pub result: CommandResult,
    /// Timestamp (frame number).
    pub frame: u64,
    /// Timestamp (seconds since engine start).
    pub timestamp: f64,
}

// ---------------------------------------------------------------------------
// CheatFlags
// ---------------------------------------------------------------------------

/// Flags tracking which cheat modes are currently active.
#[derive(Debug, Clone, Default)]
pub struct CheatFlags {
    /// God mode: player cannot take damage.
    pub god_mode: bool,
    /// Noclip: player can pass through walls and fly freely.
    pub noclip: bool,
    /// Fly mode: player can fly but still collides.
    pub fly: bool,
    /// Infinite ammo.
    pub infinite_ammo: bool,
    /// Time scale override (1.0 = normal).
    pub time_scale: f32,
    /// Whether AI is frozen.
    pub ai_frozen: bool,
    /// Whether the HUD is hidden.
    pub hud_hidden: bool,
    /// Free camera mode.
    pub free_camera: bool,
    /// Wireframe rendering.
    pub wireframe: bool,
    /// Ghost mode (invisible to AI).
    pub ghost: bool,
}

impl CheatFlags {
    /// Creates default cheat flags with everything disabled.
    pub fn new() -> Self {
        Self {
            time_scale: 1.0,
            ..Default::default()
        }
    }

    /// Resets all cheat flags to their defaults.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns `true` if any cheat is active.
    pub fn any_active(&self) -> bool {
        self.god_mode
            || self.noclip
            || self.fly
            || self.infinite_ammo
            || (self.time_scale - 1.0).abs() > 0.001
            || self.ai_frozen
            || self.hud_hidden
            || self.free_camera
            || self.wireframe
            || self.ghost
    }

    /// Returns a human-readable summary of active cheats.
    pub fn active_summary(&self) -> Vec<String> {
        let mut active = Vec::new();
        if self.god_mode { active.push("God Mode".to_string()); }
        if self.noclip { active.push("NoClip".to_string()); }
        if self.fly { active.push("Fly".to_string()); }
        if self.infinite_ammo { active.push("Infinite Ammo".to_string()); }
        if (self.time_scale - 1.0).abs() > 0.001 {
            active.push(format!("Time Scale: {:.2}x", self.time_scale));
        }
        if self.ai_frozen { active.push("AI Frozen".to_string()); }
        if self.hud_hidden { active.push("HUD Hidden".to_string()); }
        if self.free_camera { active.push("Free Camera".to_string()); }
        if self.wireframe { active.push("Wireframe".to_string()); }
        if self.ghost { active.push("Ghost".to_string()); }
        active
    }
}

// ---------------------------------------------------------------------------
// CommandSystem
// ---------------------------------------------------------------------------

/// The central command system that manages command registration, execution,
/// key bindings, macros, and config file loading.
pub struct CommandSystem {
    /// Registered commands by name (lowercase).
    commands: HashMap<String, CommandDef>,
    /// Command aliases (alias -> primary command name).
    aliases: HashMap<String, String>,
    /// Key bindings.
    key_bindings: Vec<KeyBinding>,
    /// Command macros.
    macros: HashMap<String, CommandMacro>,
    /// Command history.
    history: Vec<CommandHistoryEntry>,
    /// Maximum history size.
    max_history: usize,
    /// Whether cheats are enabled.
    cheats_enabled: bool,
    /// Current cheat flags.
    cheat_flags: CheatFlags,
    /// Current frame number (for history timestamps).
    current_frame: u64,
    /// Current time (for history timestamps).
    current_time: f64,
    /// Config search paths.
    config_paths: Vec<PathBuf>,
    /// Echo mode: print commands as they are executed.
    echo: bool,
}

impl CommandSystem {
    /// Creates a new command system.
    pub fn new() -> Self {
        let mut system = Self {
            commands: HashMap::new(),
            aliases: HashMap::new(),
            key_bindings: Vec::new(),
            macros: HashMap::new(),
            history: Vec::new(),
            max_history: 1000,
            cheats_enabled: false,
            cheat_flags: CheatFlags::new(),
            current_frame: 0,
            current_time: 0.0,
            config_paths: Vec::new(),
            echo: false,
        };

        system.register_builtin_commands();
        system
    }

    /// Enables or disables cheats.
    pub fn set_cheats_enabled(&mut self, enabled: bool) {
        self.cheats_enabled = enabled;
        if !enabled {
            self.cheat_flags.reset();
        }
    }

    /// Returns whether cheats are enabled.
    pub fn cheats_enabled(&self) -> bool {
        self.cheats_enabled
    }

    /// Returns a reference to the current cheat flags.
    pub fn cheat_flags(&self) -> &CheatFlags {
        &self.cheat_flags
    }

    /// Returns a mutable reference to the cheat flags.
    pub fn cheat_flags_mut(&mut self) -> &mut CheatFlags {
        &mut self.cheat_flags
    }

    /// Registers a command.
    pub fn register(&mut self, command: CommandDef) {
        let name = command.name.to_lowercase();
        self.commands.insert(name, command);
    }

    /// Unregisters a command by name.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.commands.remove(&name.to_lowercase()).is_some()
    }

    /// Registers a command alias.
    pub fn register_alias(&mut self, alias: impl Into<String>, command_name: impl Into<String>) {
        self.aliases
            .insert(alias.into().to_lowercase(), command_name.into().to_lowercase());
    }

    /// Removes a command alias.
    pub fn remove_alias(&mut self, alias: &str) -> bool {
        self.aliases.remove(&alias.to_lowercase()).is_some()
    }

    /// Adds a key binding.
    pub fn bind_key(&mut self, binding: KeyBinding) {
        // Remove existing binding for the same key+modifiers.
        self.key_bindings
            .retain(|b| b.key != binding.key || !b.modifiers.matches(&binding.modifiers));
        self.key_bindings.push(binding);
    }

    /// Removes a key binding.
    pub fn unbind_key(&mut self, key: &str, modifiers: &KeyModifiers) {
        self.key_bindings
            .retain(|b| b.key != key || !b.modifiers.matches(modifiers));
    }

    /// Returns the command bound to a key, if any.
    pub fn get_key_binding(&self, key: &str, modifiers: &KeyModifiers) -> Option<&str> {
        self.key_bindings
            .iter()
            .find(|b| b.active && b.key == key && b.modifiers.matches(modifiers))
            .map(|b| b.command.as_str())
    }

    /// Handles a key press by executing the bound command, if any.
    pub fn handle_key_press(&mut self, key: &str, modifiers: &KeyModifiers) -> Option<CommandResult> {
        let command = self.get_key_binding(key, modifiers)?.to_string();
        Some(self.execute(&command))
    }

    /// Registers a command macro.
    pub fn register_macro(&mut self, macro_def: CommandMacro) {
        self.macros.insert(macro_def.name.to_lowercase(), macro_def);
    }

    /// Removes a command macro.
    pub fn remove_macro(&mut self, name: &str) -> bool {
        self.macros.remove(&name.to_lowercase()).is_some()
    }

    /// Adds a config search path.
    pub fn add_config_path(&mut self, path: impl Into<PathBuf>) {
        self.config_paths.push(path.into());
    }

    /// Executes a command string.
    ///
    /// The string is parsed into a command name and arguments. Aliases are
    /// resolved, and the appropriate handler is called.
    pub fn execute(&mut self, command_string: &str) -> CommandResult {
        let command_string = command_string.trim();
        if command_string.is_empty() {
            return CommandResult::ok();
        }

        // Handle comments.
        if command_string.starts_with("//") || command_string.starts_with('#') {
            return CommandResult::ok();
        }

        // Handle multiple commands separated by semicolons.
        if command_string.contains(';') {
            let parts: Vec<&str> = command_string.split(';').collect();
            let mut results = Vec::new();
            for part in parts {
                let result = self.execute(part.trim());
                let is_error = result.is_error();
                results.extend(result.lines());
                if is_error {
                    break;
                }
            }
            return CommandResult::Output(results);
        }

        // Parse command and arguments.
        let tokens = tokenize(command_string);
        if tokens.is_empty() {
            return CommandResult::ok();
        }

        let cmd_name = tokens[0].to_lowercase();
        let raw_args: Vec<CommandArg> = tokens[1..].iter().map(|t| parse_arg(t)).collect();

        // Resolve aliases.
        let resolved_name = self
            .aliases
            .get(&cmd_name)
            .cloned()
            .unwrap_or(cmd_name.clone());

        // Check for macro.
        if let Some(macro_def) = self.macros.get(&resolved_name).cloned() {
            return self.execute_macro(&macro_def);
        }

        // Look up command.
        let command = match self.commands.get(&resolved_name) {
            Some(cmd) => cmd,
            None => {
                return CommandResult::error(format!("Unknown command: '{}'", tokens[0]));
            }
        };

        // Check cheat requirement.
        if command.requires_cheats && !self.cheats_enabled {
            return CommandResult::error(format!(
                "'{}' requires cheats to be enabled. Use 'sv_cheats 1' first.",
                command.name
            ));
        }

        // Execute.
        let result = command.execute(&raw_args);

        // Record history.
        self.history.push(CommandHistoryEntry {
            command_string: command_string.to_string(),
            result: result.clone(),
            frame: self.current_frame,
            timestamp: self.current_time,
        });

        // Trim history.
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        result
    }

    /// Executes a command macro.
    fn execute_macro(&mut self, macro_def: &CommandMacro) -> CommandResult {
        let mut results = Vec::new();
        for cmd in &macro_def.commands {
            let result = self.execute(cmd);
            let is_error = result.is_error();
            results.extend(result.lines());
            if is_error && macro_def.stop_on_error {
                break;
            }
        }
        CommandResult::Output(results)
    }

    /// Executes commands from a config file.
    ///
    /// Lines starting with `//` or `#` are treated as comments. Empty lines
    /// are skipped.
    pub fn exec_config(&mut self, filename: &str) -> CommandResult {
        let path = self.find_config_file(filename);
        let content = match path {
            Some(ref p) => match std::fs::read_to_string(p) {
                Ok(c) => c,
                Err(e) => return CommandResult::error(format!("Failed to read '{}': {}", filename, e)),
            },
            None => return CommandResult::error(format!("Config file not found: '{}'", filename)),
        };

        let mut results = Vec::new();
        results.push(format!("Executing '{}'...", filename));

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
                continue;
            }
            let result = self.execute(line);
            results.extend(result.lines());
        }

        results.push(format!("Finished executing '{}'", filename));
        CommandResult::Output(results)
    }

    /// Searches config paths for a config file.
    fn find_config_file(&self, filename: &str) -> Option<PathBuf> {
        // Check absolute path first.
        let path = Path::new(filename);
        if path.is_absolute() && path.exists() {
            return Some(path.to_path_buf());
        }

        // Search config paths.
        for config_path in &self.config_paths {
            let full_path = config_path.join(filename);
            if full_path.exists() {
                return Some(full_path);
            }
        }

        None
    }

    /// Returns autocomplete suggestions for a partial command string.
    pub fn autocomplete(&self, partial: &str) -> Vec<String> {
        let partial_lower = partial.to_lowercase();
        let mut suggestions = Vec::new();

        // Match command names.
        for (name, cmd) in &self.commands {
            if !cmd.hidden && name.starts_with(&partial_lower) {
                suggestions.push(cmd.name.clone());
            }
        }

        // Match aliases.
        for alias in self.aliases.keys() {
            if alias.starts_with(&partial_lower) {
                suggestions.push(alias.clone());
            }
        }

        // Match macros.
        for name in self.macros.keys() {
            if name.starts_with(&partial_lower) {
                suggestions.push(name.clone());
            }
        }

        suggestions.sort();
        suggestions.dedup();
        suggestions
    }

    /// Returns the command history.
    pub fn history(&self) -> &[CommandHistoryEntry] {
        &self.history
    }

    /// Clears the command history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Returns all registered commands, optionally filtered by category.
    pub fn list_commands(&self, category: Option<&CommandCategory>) -> Vec<&CommandDef> {
        let mut cmds: Vec<&CommandDef> = self
            .commands
            .values()
            .filter(|cmd| !cmd.hidden)
            .filter(|cmd| category.map_or(true, |cat| &cmd.category == cat))
            .collect();
        cmds.sort_by(|a, b| a.name.cmp(&b.name));
        cmds
    }

    /// Returns all key bindings.
    pub fn key_bindings(&self) -> &[KeyBinding] {
        &self.key_bindings
    }

    /// Returns all macros.
    pub fn macros(&self) -> &HashMap<String, CommandMacro> {
        &self.macros
    }

    /// Returns all aliases.
    pub fn aliases(&self) -> &HashMap<String, String> {
        &self.aliases
    }

    /// Updates the frame counter and time.
    pub fn update(&mut self, frame: u64, time: f64) {
        self.current_frame = frame;
        self.current_time = time;
    }

    /// Returns the number of registered commands.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Sets the echo mode.
    pub fn set_echo(&mut self, echo: bool) {
        self.echo = echo;
    }

    // -----------------------------------------------------------------------
    // Built-in command registration
    // -----------------------------------------------------------------------

    fn register_builtin_commands(&mut self) {
        // -- help --
        self.register(
            CommandDef::new("help", "Display help for commands", CommandCategory::General, |args| {
                if args.is_empty() {
                    CommandResult::Help("Type 'help <command>' for specific help.\nType 'cmdlist' to list all commands.".to_string())
                } else {
                    CommandResult::Help(format!("Help for '{}'", args[0].to_display_string()))
                }
            })
            .with_help("Usage: help [command]\nDisplays help information for the specified command.")
            .with_arg(CommandArgDef::optional("command", "Command to get help for", ArgTypeHint::String, ""))
        );

        // -- echo --
        self.register(
            CommandDef::new("echo", "Print a message to the console", CommandCategory::General, |args| {
                let msg: Vec<String> = args.iter().map(|a| a.to_display_string()).collect();
                CommandResult::message(msg.join(" "))
            })
        );

        // -- clear --
        self.register(
            CommandDef::new("clear", "Clear the console output", CommandCategory::General, |_| {
                CommandResult::ok()
            })
        );

        // -- cmdlist --
        self.register(
            CommandDef::new("cmdlist", "List all available commands", CommandCategory::General, |_| {
                CommandResult::message("Use CommandSystem::list_commands() to enumerate commands")
            })
        );

        // -- sv_cheats --
        self.register(
            CommandDef::new("sv_cheats", "Enable or disable cheats", CommandCategory::System, |args| {
                if args.is_empty() {
                    CommandResult::message("Usage: sv_cheats <0|1>")
                } else {
                    let enabled = args[0].as_bool().unwrap_or(false);
                    CommandResult::message(format!("Cheats {}", if enabled { "enabled" } else { "disabled" }))
                }
            })
        );

        // -- god --
        self.register(
            CommandDef::new("god", "Toggle god mode", CommandCategory::Cheat, |_| {
                CommandResult::message("God mode toggled")
            })
            .with_cheats_required()
            .with_help("Toggles god mode on/off. Player becomes invulnerable.")
        );

        // -- noclip --
        self.register(
            CommandDef::new("noclip", "Toggle noclip mode", CommandCategory::Cheat, |_| {
                CommandResult::message("Noclip toggled")
            })
            .with_cheats_required()
            .with_help("Toggles noclip mode. Player can pass through walls and fly.")
        );

        // -- fly --
        self.register(
            CommandDef::new("fly", "Toggle fly mode", CommandCategory::Cheat, |_| {
                CommandResult::message("Fly mode toggled")
            })
            .with_cheats_required()
        );

        // -- give --
        self.register(
            CommandDef::new("give", "Give an item to the player", CommandCategory::Cheat, |args| {
                if args.is_empty() {
                    CommandResult::error("Usage: give <item_id> [count]")
                } else {
                    let item = args[0].to_display_string();
                    let count = args.get(1).and_then(|a| a.as_int()).unwrap_or(1);
                    CommandResult::message(format!("Gave {} x{}", item, count))
                }
            })
            .with_cheats_required()
            .with_arg(CommandArgDef::required("item_id", "Item identifier", ArgTypeHint::String))
            .with_arg(CommandArgDef::optional("count", "Number of items", ArgTypeHint::Int, "1"))
        );

        // -- teleport --
        self.register(
            CommandDef::new("teleport", "Teleport to coordinates", CommandCategory::Cheat, |args| {
                if args.len() < 3 {
                    CommandResult::error("Usage: teleport <x> <y> <z>")
                } else {
                    let x = args[0].as_float().unwrap_or(0.0);
                    let y = args[1].as_float().unwrap_or(0.0);
                    let z = args[2].as_float().unwrap_or(0.0);
                    CommandResult::message(format!("Teleported to ({:.1}, {:.1}, {:.1})", x, y, z))
                }
            })
            .with_cheats_required()
            .with_arg(CommandArgDef::required("x", "X coordinate", ArgTypeHint::Float))
            .with_arg(CommandArgDef::required("y", "Y coordinate", ArgTypeHint::Float))
            .with_arg(CommandArgDef::required("z", "Z coordinate", ArgTypeHint::Float))
        );

        // -- timescale --
        self.register(
            CommandDef::new("timescale", "Set time scale", CommandCategory::Cheat, |args| {
                if args.is_empty() {
                    CommandResult::error("Usage: timescale <scale>")
                } else {
                    let scale = args[0].as_float().unwrap_or(1.0);
                    CommandResult::message(format!("Time scale set to {:.2}", scale))
                }
            })
            .with_cheats_required()
            .with_arg(CommandArgDef::required("scale", "Time scale multiplier", ArgTypeHint::Float))
        );

        // -- bind --
        self.register(
            CommandDef::new("bind", "Bind a key to a command", CommandCategory::System, |args| {
                if args.len() < 2 {
                    CommandResult::error("Usage: bind <key> <command>")
                } else {
                    let key = args[0].to_display_string();
                    let cmd: Vec<String> = args[1..].iter().map(|a| a.to_display_string()).collect();
                    CommandResult::message(format!("Bound '{}' to '{}'", key, cmd.join(" ")))
                }
            })
            .with_arg(CommandArgDef::required("key", "Key to bind", ArgTypeHint::String))
            .with_arg(CommandArgDef::required("command", "Command to execute", ArgTypeHint::String))
        );

        // -- unbind --
        self.register(
            CommandDef::new("unbind", "Remove a key binding", CommandCategory::System, |args| {
                if args.is_empty() {
                    CommandResult::error("Usage: unbind <key>")
                } else {
                    CommandResult::message(format!("Unbound '{}'", args[0].to_display_string()))
                }
            })
        );

        // -- exec --
        self.register(
            CommandDef::new("exec", "Execute a config file", CommandCategory::System, |args| {
                if args.is_empty() {
                    CommandResult::error("Usage: exec <filename>")
                } else {
                    CommandResult::message(format!("exec: use CommandSystem::exec_config() for '{}'", args[0].to_display_string()))
                }
            })
            .with_arg(CommandArgDef::required("filename", "Config file to execute", ArgTypeHint::FilePath))
        );

        // -- alias --
        self.register(
            CommandDef::new("alias", "Create or list command aliases", CommandCategory::System, |args| {
                if args.is_empty() {
                    CommandResult::message("Usage: alias <name> <command>")
                } else if args.len() == 1 {
                    CommandResult::message(format!("Alias '{}' not defined", args[0].to_display_string()))
                } else {
                    let name = args[0].to_display_string();
                    let cmd: Vec<String> = args[1..].iter().map(|a| a.to_display_string()).collect();
                    CommandResult::message(format!("Alias '{}' = '{}'", name, cmd.join(" ")))
                }
            })
        );

        // -- freeze_ai --
        self.register(
            CommandDef::new("freeze_ai", "Freeze all AI", CommandCategory::Cheat, |_| {
                CommandResult::message("AI frozen toggled")
            })
            .with_cheats_required()
        );

        // -- ghost --
        self.register(
            CommandDef::new("ghost", "Toggle ghost mode (invisible to AI)", CommandCategory::Cheat, |_| {
                CommandResult::message("Ghost mode toggled")
            })
            .with_cheats_required()
        );

        // -- wireframe --
        self.register(
            CommandDef::new("wireframe", "Toggle wireframe rendering", CommandCategory::Rendering, |_| {
                CommandResult::message("Wireframe toggled")
            })
            .with_cheats_required()
        );

        // -- quit --
        self.register(
            CommandDef::new("quit", "Quit the application", CommandCategory::System, |_| {
                CommandResult::message("Quit requested")
            })
        );

        // Register aliases.
        self.register_alias("tp", "teleport");
        self.register_alias("godmode", "god");
        self.register_alias("nc", "noclip");
        self.register_alias("exit", "quit");
        self.register_alias("cls", "clear");
    }
}

impl Default for CommandSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CommandSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandSystem")
            .field("command_count", &self.commands.len())
            .field("alias_count", &self.aliases.len())
            .field("key_binding_count", &self.key_bindings.len())
            .field("macro_count", &self.macros.len())
            .field("history_size", &self.history.len())
            .field("cheats_enabled", &self.cheats_enabled)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Tokenizes a command string, handling quoted strings.
fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut quote_char = ' ';

    for ch in input.chars() {
        if in_quotes {
            if ch == quote_char {
                in_quotes = false;
            } else {
                current.push(ch);
            }
        } else if ch == '"' || ch == '\'' {
            in_quotes = true;
            quote_char = ch;
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_simple() {
        let tokens = tokenize("echo hello world");
        assert_eq!(tokens, vec!["echo", "hello", "world"]);
    }

    #[test]
    fn test_tokenize_quoted() {
        let tokens = tokenize("echo \"hello world\"");
        assert_eq!(tokens, vec!["echo", "hello world"]);
    }

    #[test]
    fn test_parse_arg_types() {
        assert!(matches!(parse_arg("42"), CommandArg::Int(42)));
        assert!(matches!(parse_arg("3.14"), CommandArg::Float(_)));
        assert!(matches!(parse_arg("true"), CommandArg::Bool(true)));
        assert!(matches!(parse_arg("hello"), CommandArg::String(_)));
    }

    #[test]
    fn test_command_system_execute() {
        let mut system = CommandSystem::new();
        let result = system.execute("echo hello");
        assert!(result.is_success());
    }

    #[test]
    fn test_command_system_unknown_command() {
        let mut system = CommandSystem::new();
        let result = system.execute("nonexistent_command");
        assert!(result.is_error());
    }

    #[test]
    fn test_command_alias() {
        let mut system = CommandSystem::new();
        let result = system.execute("cls");
        assert!(result.is_success());
    }

    #[test]
    fn test_cheat_requires_enablement() {
        let mut system = CommandSystem::new();
        let result = system.execute("god");
        assert!(result.is_error());

        system.set_cheats_enabled(true);
        let result = system.execute("god");
        assert!(result.is_success());
    }

    #[test]
    fn test_autocomplete() {
        let system = CommandSystem::new();
        let suggestions = system.autocomplete("ec");
        assert!(suggestions.contains(&"echo".to_string()));
    }

    #[test]
    fn test_key_binding() {
        let mut system = CommandSystem::new();
        system.bind_key(KeyBinding::new("F5", KeyModifiers::NONE, "god"));
        let cmd = system.get_key_binding("F5", &KeyModifiers::NONE);
        assert_eq!(cmd, Some("god"));
    }

    #[test]
    fn test_command_macro() {
        let mut system = CommandSystem::new();
        system.set_cheats_enabled(true);
        let macro_def = CommandMacro::new("reset_cheats", vec![
            "echo Resetting cheats...".to_string(),
        ]);
        system.register_macro(macro_def);
        let result = system.execute("reset_cheats");
        assert!(result.is_success());
    }

    #[test]
    fn test_cheat_flags() {
        let mut flags = CheatFlags::new();
        assert!(!flags.any_active());
        flags.god_mode = true;
        assert!(flags.any_active());
        let summary = flags.active_summary();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0], "God Mode");
    }

    #[test]
    fn test_semicolon_commands() {
        let mut system = CommandSystem::new();
        let result = system.execute("echo hello; echo world");
        assert!(result.is_success());
    }
}
