//! Configuration management for the Genovo engine.
//!
//! Supports INI/TOML-style config parsing, typed config values, config
//! inheritance (base + override), command-line argument parsing, environment
//! variable fallback, config hot-reload, config validation, and serialization.
//!
//! # Format
//!
//! The configuration format supports sections, key-value pairs, comments,
//! and nested values:
//!
//! ```text
//! # This is a comment
//! [graphics]
//! resolution_x = 1920
//! resolution_y = 1080
//! fullscreen = true
//! vsync = false
//! render_scale = 1.0
//!
//! [audio]
//! master_volume = 0.8
//! music_volume = 0.6
//! sfx_volume = 1.0
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// ConfigValue
// ---------------------------------------------------------------------------

/// A typed configuration value.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    /// Boolean value.
    Bool(bool),
    /// Signed 64-bit integer.
    Int(i64),
    /// 64-bit floating point.
    Float(f64),
    /// String value.
    String(String),
    /// Array of config values.
    Array(Vec<ConfigValue>),
    /// A table/section of key-value pairs.
    Table(BTreeMap<String, ConfigValue>),
    /// Null / not set.
    Null,
}

impl ConfigValue {
    /// Try to interpret this value as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Bool(b) => Some(*b),
            ConfigValue::Int(i) => Some(*i != 0),
            ConfigValue::String(s) => match s.to_lowercase().as_str() {
                "true" | "yes" | "on" | "1" => Some(true),
                "false" | "no" | "off" | "0" => Some(false),
                _ => None,
            },
            _ => None,
        }
    }

    /// Try to interpret this value as an i64.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ConfigValue::Int(i) => Some(*i),
            ConfigValue::Float(f) => Some(*f as i64),
            ConfigValue::Bool(b) => Some(if *b { 1 } else { 0 }),
            ConfigValue::String(s) => s.parse::<i64>().ok(),
            _ => None,
        }
    }

    /// Try to interpret this value as an f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConfigValue::Float(f) => Some(*f),
            ConfigValue::Int(i) => Some(*i as f64),
            ConfigValue::String(s) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Try to interpret this value as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ConfigValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to get the value as a string, converting other types.
    pub fn to_string_lossy(&self) -> String {
        match self {
            ConfigValue::Bool(b) => b.to_string(),
            ConfigValue::Int(i) => i.to_string(),
            ConfigValue::Float(f) => f.to_string(),
            ConfigValue::String(s) => s.clone(),
            ConfigValue::Array(a) => format!("{:?}", a),
            ConfigValue::Table(t) => format!("{:?}", t),
            ConfigValue::Null => "null".to_string(),
        }
    }

    /// Try to interpret this value as an array.
    pub fn as_array(&self) -> Option<&[ConfigValue]> {
        match self {
            ConfigValue::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    /// Try to interpret this value as a table.
    pub fn as_table(&self) -> Option<&BTreeMap<String, ConfigValue>> {
        match self {
            ConfigValue::Table(t) => Some(t),
            _ => None,
        }
    }

    /// Returns `true` if the value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, ConfigValue::Null)
    }

    /// Returns the type name of this value.
    pub fn type_name(&self) -> &'static str {
        match self {
            ConfigValue::Bool(_) => "bool",
            ConfigValue::Int(_) => "int",
            ConfigValue::Float(_) => "float",
            ConfigValue::String(_) => "string",
            ConfigValue::Array(_) => "array",
            ConfigValue::Table(_) => "table",
            ConfigValue::Null => "null",
        }
    }
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigValue::Bool(b) => write!(f, "{}", b),
            ConfigValue::Int(i) => write!(f, "{}", i),
            ConfigValue::Float(v) => write!(f, "{}", v),
            ConfigValue::String(s) => write!(f, "{}", s),
            ConfigValue::Array(a) => {
                write!(f, "[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            ConfigValue::Table(t) => {
                write!(f, "{{")?;
                for (i, (k, v)) in t.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} = {}", k, v)?;
                }
                write!(f, "}}")
            }
            ConfigValue::Null => write!(f, "null"),
        }
    }
}

impl From<bool> for ConfigValue {
    fn from(v: bool) -> Self {
        ConfigValue::Bool(v)
    }
}

impl From<i64> for ConfigValue {
    fn from(v: i64) -> Self {
        ConfigValue::Int(v)
    }
}

impl From<i32> for ConfigValue {
    fn from(v: i32) -> Self {
        ConfigValue::Int(v as i64)
    }
}

impl From<f64> for ConfigValue {
    fn from(v: f64) -> Self {
        ConfigValue::Float(v)
    }
}

impl From<f32> for ConfigValue {
    fn from(v: f32) -> Self {
        ConfigValue::Float(v as f64)
    }
}

impl From<&str> for ConfigValue {
    fn from(v: &str) -> Self {
        ConfigValue::String(v.to_string())
    }
}

impl From<String> for ConfigValue {
    fn from(v: String) -> Self {
        ConfigValue::String(v)
    }
}

// ---------------------------------------------------------------------------
// Config Parse Error
// ---------------------------------------------------------------------------

/// Errors that can occur during config parsing or validation.
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Syntax error in the config file.
    ParseError { line: usize, message: String },
    /// A required key is missing.
    MissingKey(String),
    /// A value has the wrong type.
    TypeError { key: String, expected: String, got: String },
    /// A value is outside the allowed range.
    RangeError { key: String, message: String },
    /// File I/O error.
    IoError(String),
    /// Validation error.
    ValidationError(String),
    /// Circular inheritance detected.
    CircularInheritance(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::ParseError { line, message } => {
                write!(f, "parse error at line {}: {}", line, message)
            }
            ConfigError::MissingKey(key) => write!(f, "missing required key: {}", key),
            ConfigError::TypeError { key, expected, got } => {
                write!(f, "type error for '{}': expected {}, got {}", key, expected, got)
            }
            ConfigError::RangeError { key, message } => {
                write!(f, "range error for '{}': {}", key, message)
            }
            ConfigError::IoError(msg) => write!(f, "I/O error: {}", msg),
            ConfigError::ValidationError(msg) => write!(f, "validation error: {}", msg),
            ConfigError::CircularInheritance(msg) => {
                write!(f, "circular inheritance: {}", msg)
            }
        }
    }
}

pub type ConfigResult<T> = Result<T, ConfigError>;

// ---------------------------------------------------------------------------
// Config Parser (INI-style)
// ---------------------------------------------------------------------------

/// Parses an INI/TOML-style configuration string into a nested map.
pub struct ConfigParser;

impl ConfigParser {
    /// Parse a configuration string into sections and key-value pairs.
    pub fn parse(source: &str) -> ConfigResult<BTreeMap<String, ConfigValue>> {
        let mut root = BTreeMap::new();
        let mut current_section: Option<String> = None;

        for (line_num, raw_line) in source.lines().enumerate() {
            let line = raw_line.trim();

            // Skip empty lines and comments.
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }

            // Section header.
            if line.starts_with('[') && line.ends_with(']') {
                let section_name = line[1..line.len() - 1].trim().to_string();
                if section_name.is_empty() {
                    return Err(ConfigError::ParseError {
                        line: line_num + 1,
                        message: "empty section name".to_string(),
                    });
                }
                current_section = Some(section_name);
                continue;
            }

            // Key = value pair.
            let (key, value_str) = if let Some(eq_pos) = line.find('=') {
                let k = line[..eq_pos].trim();
                let v = line[eq_pos + 1..].trim();
                (k.to_string(), v.to_string())
            } else {
                return Err(ConfigError::ParseError {
                    line: line_num + 1,
                    message: format!("expected 'key = value', got: {}", line),
                });
            };

            // Strip inline comments.
            let value_str = if let Some(comment_pos) = value_str.find(" #") {
                value_str[..comment_pos].trim().to_string()
            } else {
                value_str
            };

            let value = Self::parse_value(&value_str);

            if let Some(ref section) = current_section {
                let section_table = root
                    .entry(section.clone())
                    .or_insert_with(|| ConfigValue::Table(BTreeMap::new()));
                if let ConfigValue::Table(table) = section_table {
                    table.insert(key, value);
                }
            } else {
                root.insert(key, value);
            }
        }

        Ok(root)
    }

    /// Parse a value string into a typed ConfigValue.
    fn parse_value(s: &str) -> ConfigValue {
        // Boolean.
        match s.to_lowercase().as_str() {
            "true" | "yes" | "on" => return ConfigValue::Bool(true),
            "false" | "no" | "off" => return ConfigValue::Bool(false),
            "null" | "nil" | "none" => return ConfigValue::Null,
            _ => {}
        }

        // Integer.
        if let Ok(i) = s.parse::<i64>() {
            return ConfigValue::Int(i);
        }

        // Hex integer.
        if s.starts_with("0x") || s.starts_with("0X") {
            if let Ok(i) = i64::from_str_radix(&s[2..], 16) {
                return ConfigValue::Int(i);
            }
        }

        // Float.
        if let Ok(f) = s.parse::<f64>() {
            return ConfigValue::Float(f);
        }

        // Quoted string.
        if (s.starts_with('"') && s.ends_with('"'))
            || (s.starts_with('\'') && s.ends_with('\''))
        {
            return ConfigValue::String(s[1..s.len() - 1].to_string());
        }

        // Array: [a, b, c].
        if s.starts_with('[') && s.ends_with(']') {
            let inner = s[1..s.len() - 1].trim();
            if inner.is_empty() {
                return ConfigValue::Array(Vec::new());
            }
            let items: Vec<ConfigValue> = inner
                .split(',')
                .map(|item| Self::parse_value(item.trim()))
                .collect();
            return ConfigValue::Array(items);
        }

        // Fallback: treat as string.
        ConfigValue::String(s.to_string())
    }

    /// Serialize a config map back to INI format.
    pub fn serialize(config: &BTreeMap<String, ConfigValue>) -> String {
        let mut output = String::new();

        // First, write top-level non-table values.
        for (key, value) in config {
            if !matches!(value, ConfigValue::Table(_)) {
                output.push_str(&format!("{} = {}\n", key, Self::serialize_value(value)));
            }
        }

        // Then write sections.
        for (key, value) in config {
            if let ConfigValue::Table(table) = value {
                output.push_str(&format!("\n[{}]\n", key));
                for (k, v) in table {
                    output.push_str(&format!("{} = {}\n", k, Self::serialize_value(v)));
                }
            }
        }

        output
    }

    fn serialize_value(value: &ConfigValue) -> String {
        match value {
            ConfigValue::Bool(b) => b.to_string(),
            ConfigValue::Int(i) => i.to_string(),
            ConfigValue::Float(f) => {
                if f.fract() == 0.0 {
                    format!("{:.1}", f)
                } else {
                    f.to_string()
                }
            }
            ConfigValue::String(s) => {
                if s.contains(' ') || s.contains('=') || s.contains('#') {
                    format!("\"{}\"", s)
                } else {
                    s.clone()
                }
            }
            ConfigValue::Array(a) => {
                let items: Vec<String> = a.iter().map(Self::serialize_value).collect();
                format!("[{}]", items.join(", "))
            }
            ConfigValue::Table(t) => {
                // Inline table.
                let items: Vec<String> = t
                    .iter()
                    .map(|(k, v)| format!("{} = {}", k, Self::serialize_value(v)))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
            ConfigValue::Null => "null".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Command Line Argument Parser
// ---------------------------------------------------------------------------

/// Parsed command-line arguments.
#[derive(Debug, Clone, Default)]
pub struct CommandLineArgs {
    /// Positional arguments (not prefixed with - or --).
    pub positional: Vec<String>,
    /// Named flags and values: `--key=value` or `--flag` (value is "true").
    pub named: HashMap<String, String>,
    /// Short flags: `-v`, `-d` etc.
    pub short_flags: Vec<char>,
}

impl CommandLineArgs {
    /// Parse from a list of argument strings.
    pub fn parse(args: &[String]) -> Self {
        let mut result = CommandLineArgs::default();

        let mut iter = args.iter().peekable();
        while let Some(arg) = iter.next() {
            if arg.starts_with("--") {
                let rest = &arg[2..];
                if let Some(eq_pos) = rest.find('=') {
                    let key = rest[..eq_pos].to_string();
                    let value = rest[eq_pos + 1..].to_string();
                    result.named.insert(key, value);
                } else {
                    // Check if next arg is the value.
                    if let Some(next) = iter.peek() {
                        if !next.starts_with('-') {
                            result.named.insert(rest.to_string(), iter.next().unwrap().clone());
                        } else {
                            result.named.insert(rest.to_string(), "true".to_string());
                        }
                    } else {
                        result.named.insert(rest.to_string(), "true".to_string());
                    }
                }
            } else if arg.starts_with('-') && arg.len() > 1 {
                for c in arg[1..].chars() {
                    result.short_flags.push(c);
                }
            } else {
                result.positional.push(arg.clone());
            }
        }

        result
    }

    /// Get a named argument value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.named.get(key).map(|s| s.as_str())
    }

    /// Check if a short flag is set.
    pub fn has_flag(&self, flag: char) -> bool {
        self.short_flags.contains(&flag)
    }

    /// Check if a named argument exists.
    pub fn has(&self, key: &str) -> bool {
        self.named.contains_key(key)
    }

    /// Get a named argument as a typed value.
    pub fn get_as<T: std::str::FromStr>(&self, key: &str) -> Option<T> {
        self.named.get(key).and_then(|s| s.parse().ok())
    }
}

// ---------------------------------------------------------------------------
// Validation Rule
// ---------------------------------------------------------------------------

/// A rule that validates a single configuration value.
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// The dotted key path (e.g., "graphics.resolution_x").
    pub key: String,
    /// Whether this key is required.
    pub required: bool,
    /// Expected type name (for error messages).
    pub expected_type: Option<String>,
    /// Minimum value (for numeric types).
    pub min: Option<f64>,
    /// Maximum value (for numeric types).
    pub max: Option<f64>,
    /// Allowed string values (enum-like).
    pub allowed_values: Vec<String>,
    /// Default value to use if missing.
    pub default: Option<ConfigValue>,
    /// Human-readable description.
    pub description: String,
}

impl ValidationRule {
    /// Create a new validation rule for the given key.
    pub fn new(key: &str) -> Self {
        Self {
            key: key.to_string(),
            required: false,
            expected_type: None,
            min: None,
            max: None,
            allowed_values: Vec::new(),
            default: None,
            description: String::new(),
        }
    }

    /// Mark this key as required.
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Set the expected type.
    pub fn expected_type(mut self, t: &str) -> Self {
        self.expected_type = Some(t.to_string());
        self
    }

    /// Set a minimum numeric value.
    pub fn min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    /// Set a maximum numeric value.
    pub fn max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Set allowed string values.
    pub fn allowed(mut self, values: &[&str]) -> Self {
        self.allowed_values = values.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set a default value.
    pub fn default_value(mut self, value: ConfigValue) -> Self {
        self.default = Some(value);
        self
    }

    /// Set a description.
    pub fn description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }
}

// ---------------------------------------------------------------------------
// ConfigSchema
// ---------------------------------------------------------------------------

/// Schema that describes the expected structure and constraints of a config.
#[derive(Debug, Clone, Default)]
pub struct ConfigSchema {
    /// Validation rules keyed by dotted path.
    pub rules: Vec<ValidationRule>,
}

impl ConfigSchema {
    /// Create an empty schema.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a validation rule.
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    /// Builder-style: add a rule and return self.
    pub fn with_rule(mut self, rule: ValidationRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Validate a config map against this schema.
    pub fn validate(&self, config: &ConfigSystem) -> Vec<ConfigError> {
        let mut errors = Vec::new();

        for rule in &self.rules {
            let value = config.get(&rule.key);

            match value {
                None | Some(ConfigValue::Null) => {
                    if rule.required && rule.default.is_none() {
                        errors.push(ConfigError::MissingKey(rule.key.clone()));
                    }
                }
                Some(val) => {
                    // Type check.
                    if let Some(ref expected) = rule.expected_type {
                        let actual = val.type_name();
                        if actual != expected.as_str() {
                            errors.push(ConfigError::TypeError {
                                key: rule.key.clone(),
                                expected: expected.clone(),
                                got: actual.to_string(),
                            });
                        }
                    }

                    // Range check.
                    if let Some(num) = val.as_float() {
                        if let Some(min) = rule.min {
                            if num < min {
                                errors.push(ConfigError::RangeError {
                                    key: rule.key.clone(),
                                    message: format!("value {} is below minimum {}", num, min),
                                });
                            }
                        }
                        if let Some(max) = rule.max {
                            if num > max {
                                errors.push(ConfigError::RangeError {
                                    key: rule.key.clone(),
                                    message: format!("value {} is above maximum {}", num, max),
                                });
                            }
                        }
                    }

                    // Allowed values check.
                    if !rule.allowed_values.is_empty() {
                        let s = val.to_string_lossy();
                        if !rule.allowed_values.contains(&s) {
                            errors.push(ConfigError::ValidationError(format!(
                                "key '{}': value '{}' is not in allowed set {:?}",
                                rule.key, s, rule.allowed_values
                            )));
                        }
                    }
                }
            }
        }

        errors
    }

    /// Apply defaults from the schema to a config system.
    pub fn apply_defaults(&self, config: &mut ConfigSystem) {
        for rule in &self.rules {
            if config.get(&rule.key).is_none() {
                if let Some(ref default) = rule.default {
                    config.set(&rule.key, default.clone());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Environment Variable Resolver
// ---------------------------------------------------------------------------

/// Resolves configuration values from environment variables.
#[derive(Debug, Clone)]
pub struct EnvResolver {
    /// Prefix for environment variables (e.g., "GENOVO_").
    prefix: String,
    /// Separator between section and key in env var name (e.g., "__").
    separator: String,
}

impl EnvResolver {
    /// Create a new environment variable resolver.
    pub fn new(prefix: &str, separator: &str) -> Self {
        Self {
            prefix: prefix.to_uppercase(),
            separator: separator.to_string(),
        }
    }

    /// Resolve a dotted config key to an environment variable value.
    pub fn resolve(&self, key: &str) -> Option<ConfigValue> {
        let env_key = format!(
            "{}{}",
            self.prefix,
            key.replace('.', &self.separator).to_uppercase()
        );
        std::env::var(&env_key).ok().map(|v| ConfigParser::parse_value_public(&v))
    }

    /// Resolve all matching environment variables.
    pub fn resolve_all(&self) -> BTreeMap<String, ConfigValue> {
        let mut result = BTreeMap::new();
        for (key, value) in std::env::vars() {
            if key.starts_with(&self.prefix) {
                let config_key = key[self.prefix.len()..]
                    .replace(&self.separator, ".")
                    .to_lowercase();
                result.insert(config_key, ConfigParser::parse_value_public(&value));
            }
        }
        result
    }
}

impl ConfigParser {
    /// Public wrapper for value parsing (used by EnvResolver).
    pub fn parse_value_public(s: &str) -> ConfigValue {
        Self::parse_value(s)
    }
}

// ---------------------------------------------------------------------------
// Hot-Reload Watcher
// ---------------------------------------------------------------------------

/// Tracks file modification times for hot-reload.
#[derive(Debug, Clone)]
pub struct ConfigWatcher {
    /// Path to the watched config file.
    pub path: String,
    /// Last known modification time (as a SystemTime serialized to nanos).
    pub last_modified: u64,
    /// Last time we checked for changes.
    pub last_check: Instant,
    /// Minimum interval between checks.
    pub check_interval: std::time::Duration,
    /// Whether a reload is pending.
    pub dirty: bool,
}

impl ConfigWatcher {
    /// Create a watcher for a config file path.
    pub fn new(path: &str, check_interval: std::time::Duration) -> Self {
        let last_modified = Self::get_file_mtime(path).unwrap_or(0);
        Self {
            path: path.to_string(),
            last_modified,
            last_check: Instant::now(),
            check_interval,
            dirty: false,
        }
    }

    /// Check whether the file has been modified since last check.
    pub fn check(&mut self) -> bool {
        if self.last_check.elapsed() < self.check_interval {
            return self.dirty;
        }
        self.last_check = Instant::now();

        if let Some(mtime) = Self::get_file_mtime(&self.path) {
            if mtime != self.last_modified {
                self.last_modified = mtime;
                self.dirty = true;
            }
        }
        self.dirty
    }

    /// Clear the dirty flag after a reload.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    fn get_file_mtime(path: &str) -> Option<u64> {
        std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as u64)
    }
}

// ---------------------------------------------------------------------------
// ConfigSystem
// ---------------------------------------------------------------------------

/// The main configuration system, combining parsing, inheritance, environment
/// fallback, validation, hot-reload, and serialization.
pub struct ConfigSystem {
    /// The resolved configuration data.
    data: BTreeMap<String, ConfigValue>,
    /// Stack of config layers (base -> overrides). Index 0 is lowest priority.
    layers: Vec<(String, BTreeMap<String, ConfigValue>)>,
    /// Environment variable resolver.
    env_resolver: Option<EnvResolver>,
    /// File watchers for hot-reload.
    watchers: Vec<ConfigWatcher>,
    /// Schema for validation.
    schema: Option<ConfigSchema>,
    /// Listeners notified on config changes.
    change_listeners: Vec<Box<dyn Fn(&str, &ConfigValue) + Send + Sync>>,
    /// History of config changes for debugging.
    change_history: Vec<ConfigChange>,
    /// Maximum history entries.
    max_history: usize,
}

/// A record of a configuration change.
#[derive(Debug, Clone)]
pub struct ConfigChange {
    /// The key that was changed.
    pub key: String,
    /// The old value (None if newly added).
    pub old_value: Option<ConfigValue>,
    /// The new value.
    pub new_value: ConfigValue,
    /// When the change occurred.
    pub timestamp: Instant,
    /// Source of the change (e.g., "file", "env", "cli", "api").
    pub source: String,
}

impl ConfigSystem {
    /// Create a new, empty config system.
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
            layers: Vec::new(),
            env_resolver: None,
            watchers: Vec::new(),
            schema: None,
            change_listeners: Vec::new(),
            change_history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Set the environment variable resolver.
    pub fn set_env_resolver(&mut self, resolver: EnvResolver) {
        self.env_resolver = Some(resolver);
    }

    /// Set the config schema.
    pub fn set_schema(&mut self, schema: ConfigSchema) {
        self.schema = Some(schema);
    }

    /// Load a config layer from a string.
    pub fn load_str(&mut self, name: &str, source: &str) -> ConfigResult<()> {
        let parsed = ConfigParser::parse(source)?;
        self.layers.push((name.to_string(), parsed));
        self.rebuild();
        Ok(())
    }

    /// Load a config layer from a file path.
    pub fn load_file(&mut self, path: &str) -> ConfigResult<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        self.load_str(path, &content)?;

        // Register a watcher for hot-reload.
        let watcher = ConfigWatcher::new(path, std::time::Duration::from_secs(2));
        self.watchers.push(watcher);

        Ok(())
    }

    /// Load command-line arguments as a config layer.
    pub fn load_args(&mut self, args: &[String]) {
        let parsed = CommandLineArgs::parse(args);
        let mut map = BTreeMap::new();
        for (key, value) in &parsed.named {
            let config_key = key.replace('-', "_");
            map.insert(config_key, ConfigParser::parse_value_public(value));
        }
        self.layers.push(("cli".to_string(), map));
        self.rebuild();
    }

    /// Rebuild the resolved config by flattening layers in order.
    fn rebuild(&mut self) {
        let mut merged = BTreeMap::new();
        for (_, layer) in &self.layers {
            Self::merge_into(&mut merged, layer);
        }

        // Apply environment variable overrides.
        if let Some(ref resolver) = self.env_resolver {
            let env_map = resolver.resolve_all();
            Self::merge_into(&mut merged, &env_map);
        }

        self.data = merged;
    }

    /// Deep-merge source into target.
    fn merge_into(target: &mut BTreeMap<String, ConfigValue>, source: &BTreeMap<String, ConfigValue>) {
        for (key, value) in source {
            match (target.get_mut(key), value) {
                (Some(ConfigValue::Table(existing)), ConfigValue::Table(incoming)) => {
                    for (k, v) in incoming {
                        existing.insert(k.clone(), v.clone());
                    }
                }
                _ => {
                    target.insert(key.clone(), value.clone());
                }
            }
        }
    }

    /// Get a config value by dotted key path (e.g., "graphics.resolution_x").
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        let parts: Vec<&str> = key.splitn(2, '.').collect();
        if parts.len() == 1 {
            self.data.get(parts[0])
        } else {
            let section_val = self.data.get(parts[0])?;
            if let ConfigValue::Table(table) = section_val {
                table.get(parts[1])
            } else {
                None
            }
        }
    }

    /// Get a typed boolean value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }

    /// Get a typed integer value.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.as_int())
    }

    /// Get a typed float value.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_float())
    }

    /// Get a typed string value.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }

    /// Get a value or a fallback default.
    pub fn get_or(&self, key: &str, default: ConfigValue) -> ConfigValue {
        self.get(key).cloned().unwrap_or(default)
    }

    /// Get a boolean with a default.
    pub fn get_bool_or(&self, key: &str, default: bool) -> bool {
        self.get_bool(key).unwrap_or(default)
    }

    /// Get an integer with a default.
    pub fn get_int_or(&self, key: &str, default: i64) -> i64 {
        self.get_int(key).unwrap_or(default)
    }

    /// Get a float with a default.
    pub fn get_float_or(&self, key: &str, default: f64) -> f64 {
        self.get_float(key).unwrap_or(default)
    }

    /// Set a value at a dotted key path.
    pub fn set(&mut self, key: &str, value: ConfigValue) {
        let old = self.get(key).cloned();

        let parts: Vec<&str> = key.splitn(2, '.').collect();
        if parts.len() == 1 {
            self.data.insert(parts[0].to_string(), value.clone());
        } else {
            let section = self
                .data
                .entry(parts[0].to_string())
                .or_insert_with(|| ConfigValue::Table(BTreeMap::new()));
            if let ConfigValue::Table(table) = section {
                table.insert(parts[1].to_string(), value.clone());
            }
        }

        // Record change.
        self.change_history.push(ConfigChange {
            key: key.to_string(),
            old_value: old,
            new_value: value.clone(),
            timestamp: Instant::now(),
            source: "api".to_string(),
        });
        if self.change_history.len() > self.max_history {
            self.change_history.remove(0);
        }

        // Notify listeners.
        for listener in &self.change_listeners {
            listener(key, &value);
        }
    }

    /// Remove a key.
    pub fn remove(&mut self, key: &str) -> Option<ConfigValue> {
        let parts: Vec<&str> = key.splitn(2, '.').collect();
        if parts.len() == 1 {
            self.data.remove(parts[0])
        } else {
            if let Some(ConfigValue::Table(table)) = self.data.get_mut(parts[0]) {
                table.remove(parts[1])
            } else {
                None
            }
        }
    }

    /// Validate the current configuration against the schema.
    pub fn validate(&self) -> Vec<ConfigError> {
        if let Some(ref schema) = self.schema {
            schema.validate(self)
        } else {
            Vec::new()
        }
    }

    /// Apply schema defaults.
    pub fn apply_defaults(&mut self) {
        if let Some(schema) = self.schema.clone() {
            schema.apply_defaults(self);
        }
    }

    /// Check file watchers and reload modified configs.
    pub fn check_hot_reload(&mut self) -> bool {
        let mut any_reloaded = false;
        let watcher_paths: Vec<String> = self.watchers.iter().map(|w| w.path.clone()).collect();

        for i in 0..self.watchers.len() {
            if self.watchers[i].check() {
                self.watchers[i].clear_dirty();
                // Reload the file.
                if let Ok(content) = std::fs::read_to_string(&watcher_paths[i]) {
                    if let Ok(parsed) = ConfigParser::parse(&content) {
                        // Find and replace the layer.
                        let path = &watcher_paths[i];
                        if let Some(layer) = self.layers.iter_mut().find(|(name, _)| name == path) {
                            layer.1 = parsed;
                        }
                        self.rebuild();
                        any_reloaded = true;
                    }
                }
            }
        }

        any_reloaded
    }

    /// Serialize the current config back to INI format.
    pub fn serialize(&self) -> String {
        ConfigParser::serialize(&self.data)
    }

    /// Returns a reference to the raw data map.
    pub fn raw_data(&self) -> &BTreeMap<String, ConfigValue> {
        &self.data
    }

    /// Returns the change history.
    pub fn change_history(&self) -> &[ConfigChange] {
        &self.change_history
    }

    /// Number of config layers loaded.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get all keys (flattened with dotted notation).
    pub fn keys(&self) -> Vec<String> {
        let mut result = Vec::new();
        for (key, value) in &self.data {
            match value {
                ConfigValue::Table(table) => {
                    for sub_key in table.keys() {
                        result.push(format!("{}.{}", key, sub_key));
                    }
                }
                _ => {
                    result.push(key.clone());
                }
            }
        }
        result
    }

    /// Add a change listener.
    pub fn add_change_listener<F>(&mut self, listener: F)
    where
        F: Fn(&str, &ConfigValue) + Send + Sync + 'static,
    {
        self.change_listeners.push(Box::new(listener));
    }

    /// Clear all data and layers.
    pub fn clear(&mut self) {
        self.data.clear();
        self.layers.clear();
        self.watchers.clear();
        self.change_history.clear();
    }
}

impl Default for ConfigSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ConfigSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConfigSystem")
            .field("layers", &self.layers.len())
            .field("keys", &self.data.len())
            .field("watchers", &self.watchers.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let source = r#"
# Graphics settings
[graphics]
resolution_x = 1920
resolution_y = 1080
fullscreen = true
render_scale = 1.5

[audio]
volume = 0.8
"#;
        let mut config = ConfigSystem::new();
        config.load_str("test", source).unwrap();

        assert_eq!(config.get_int("graphics.resolution_x"), Some(1920));
        assert_eq!(config.get_bool("graphics.fullscreen"), Some(true));
        assert_eq!(config.get_float("graphics.render_scale"), Some(1.5));
        assert_eq!(config.get_float("audio.volume"), Some(0.8));
    }

    #[test]
    fn test_layer_override() {
        let base = "[video]\nvsync = true\nfps_cap = 60\n";
        let override_cfg = "[video]\nfps_cap = 144\n";

        let mut config = ConfigSystem::new();
        config.load_str("base", base).unwrap();
        config.load_str("override", override_cfg).unwrap();

        assert_eq!(config.get_bool("video.vsync"), Some(true));
        assert_eq!(config.get_int("video.fps_cap"), Some(144));
    }

    #[test]
    fn test_cli_args() {
        let args = vec![
            "--resolution".to_string(),
            "1080".to_string(),
            "--fullscreen".to_string(),
            "-vd".to_string(),
        ];
        let parsed = CommandLineArgs::parse(&args);
        assert_eq!(parsed.get("resolution"), Some("1080"));
        assert_eq!(parsed.get("fullscreen"), Some("true"));
        assert!(parsed.has_flag('v'));
        assert!(parsed.has_flag('d'));
    }

    #[test]
    fn test_validation() {
        let schema = ConfigSchema::new()
            .with_rule(
                ValidationRule::new("graphics.resolution_x")
                    .required()
                    .expected_type("int")
                    .min(640.0)
                    .max(7680.0),
            )
            .with_rule(
                ValidationRule::new("graphics.fullscreen")
                    .default_value(ConfigValue::Bool(false)),
            );

        let mut config = ConfigSystem::new();
        config.load_str("test", "[graphics]\nresolution_x = 320\n").unwrap();
        config.set_schema(schema);

        let errors = config.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_set_and_get() {
        let mut config = ConfigSystem::new();
        config.set("player.name", ConfigValue::String("Hero".to_string()));
        assert_eq!(config.get_str("player.name"), Some("Hero"));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let source = "[test]\nvalue = 42\nname = hello\n";
        let mut config = ConfigSystem::new();
        config.load_str("test", source).unwrap();
        let serialized = config.serialize();
        assert!(serialized.contains("value = 42"));
        assert!(serialized.contains("name = hello"));
    }
}
