//! Save File Versioning and Migration
//!
//! Provides a framework for managing save file versions over the lifetime
//! of a game, including:
//!
//! - Version number in save headers
//! - Migration functions for upgrading between versions (v1->v2->v3...)
//! - Backward compatibility
//! - Save file validation
//! - Corrupt save recovery
//! - Save file diff (comparing two saves)
//!
//! # Architecture
//!
//! ```text
//! SaveMigrationManager
//!   +-- VersionRegistry      (version metadata)
//!   +-- MigrationChain       (v1->v2->v3 functions)
//!   +-- SaveValidator        (structural validation)
//!   +-- SaveRecovery         (corrupt save handling)
//!   +-- SaveDiffer           (comparison between saves)
//! ```
//!
//! # Version Chain
//!
//! ```text
//! v1 ──migration_fn──> v2 ──migration_fn──> v3 ──migration_fn──> v4
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// SaveVersion
// ---------------------------------------------------------------------------

/// A save file version number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SaveVersion {
    /// Major version (breaking changes).
    pub major: u32,
    /// Minor version (additive changes).
    pub minor: u32,
    /// Patch version (bug fixes in migration code).
    pub patch: u32,
}

impl SaveVersion {
    /// Creates a new save version.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Creates a version from just a major number.
    pub const fn major_only(major: u32) -> Self {
        Self {
            major,
            minor: 0,
            patch: 0,
        }
    }

    /// Returns a numeric representation for comparison (major*10000 + minor*100 + patch).
    pub fn numeric(&self) -> u64 {
        self.major as u64 * 10000 + self.minor as u64 * 100 + self.patch as u64
    }

    /// Returns `true` if this version is compatible with the given version.
    /// A version is compatible if the major version matches.
    pub fn is_compatible_with(&self, other: &SaveVersion) -> bool {
        self.major == other.major
    }

    /// Returns `true` if this version is newer than the other.
    pub fn is_newer_than(&self, other: &SaveVersion) -> bool {
        self.numeric() > other.numeric()
    }
}

impl fmt::Display for SaveVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Default for SaveVersion {
    fn default() -> Self {
        Self::new(1, 0, 0)
    }
}

// ---------------------------------------------------------------------------
// SaveHeader
// ---------------------------------------------------------------------------

/// Header data extracted from a save file for version checking.
#[derive(Debug, Clone)]
pub struct SaveHeader {
    /// Magic bytes identifier.
    pub magic: [u8; 4],
    /// Save file version.
    pub version: SaveVersion,
    /// Game identifier (to prevent loading saves from different games).
    pub game_id: String,
    /// Build number or commit hash.
    pub build_info: String,
    /// Checksum of the save data.
    pub checksum: u64,
    /// Timestamp when the save was created (Unix epoch seconds).
    pub created_at: u64,
    /// Total size of the save data in bytes.
    pub data_size: u64,
    /// Whether the save data is compressed.
    pub compressed: bool,
    /// Compression algorithm identifier.
    pub compression_algo: Option<String>,
    /// Additional metadata flags.
    pub flags: u32,
}

impl SaveHeader {
    /// Expected magic bytes.
    pub const MAGIC: [u8; 4] = *b"GNSV";

    /// Creates a new save header with the given version.
    pub fn new(version: SaveVersion) -> Self {
        Self {
            magic: Self::MAGIC,
            version,
            game_id: String::new(),
            build_info: String::new(),
            checksum: 0,
            created_at: 0,
            data_size: 0,
            compressed: false,
            compression_algo: None,
            flags: 0,
        }
    }

    /// Validates the magic bytes.
    pub fn validate_magic(&self) -> bool {
        self.magic == Self::MAGIC
    }

    /// Sets the game identifier.
    pub fn with_game_id(mut self, id: impl Into<String>) -> Self {
        self.game_id = id.into();
        self
    }

    /// Sets the build info.
    pub fn with_build_info(mut self, info: impl Into<String>) -> Self {
        self.build_info = info.into();
        self
    }
}

// ---------------------------------------------------------------------------
// SaveData
// ---------------------------------------------------------------------------

/// Type-erased save data represented as a nested key-value structure.
///
/// This is the intermediate representation used during migration. Real
/// save data (entities, components) is serialized into this generic
/// format for migration transforms.
#[derive(Debug, Clone)]
pub enum SaveValue {
    /// A null/empty value.
    Null,
    /// A boolean value.
    Bool(bool),
    /// An integer value.
    Int(i64),
    /// A floating-point value.
    Float(f64),
    /// A string value.
    String(String),
    /// A byte array.
    Bytes(Vec<u8>),
    /// An ordered list of values.
    Array(Vec<SaveValue>),
    /// A map of string keys to values.
    Object(HashMap<String, SaveValue>),
}

impl SaveValue {
    /// Returns `true` if this value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Attempts to get a boolean value.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempts to get an integer value.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Attempts to get a float value.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Attempts to get a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    /// Attempts to get a mutable reference to an array.
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<SaveValue>> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Attempts to get a reference to an object map.
    pub fn as_object(&self) -> Option<&HashMap<String, SaveValue>> {
        match self {
            Self::Object(v) => Some(v),
            _ => None,
        }
    }

    /// Attempts to get a mutable reference to an object map.
    pub fn as_object_mut(&mut self) -> Option<&mut HashMap<String, SaveValue>> {
        match self {
            Self::Object(v) => Some(v),
            _ => None,
        }
    }

    /// Gets a nested value by dot-separated path.
    pub fn get_path(&self, path: &str) -> Option<&SaveValue> {
        let segments: Vec<&str> = path.split('.').collect();
        let mut current = self;
        for segment in segments {
            match current {
                Self::Object(map) => {
                    current = map.get(segment)?;
                }
                Self::Array(arr) => {
                    let index: usize = segment.parse().ok()?;
                    current = arr.get(index)?;
                }
                _ => return None,
            }
        }
        Some(current)
    }

    /// Sets a nested value by dot-separated path, creating intermediate objects.
    pub fn set_path(&mut self, path: &str, value: SaveValue) {
        let segments: Vec<&str> = path.split('.').collect();
        if segments.is_empty() {
            return;
        }

        let mut current = self;
        for (i, segment) in segments.iter().enumerate() {
            if i == segments.len() - 1 {
                // Last segment: set the value.
                if let Self::Object(map) = current {
                    map.insert(segment.to_string(), value);
                }
                return;
            }

            // Navigate or create intermediate objects.
            if let Self::Object(map) = current {
                if !map.contains_key(*segment) {
                    map.insert(segment.to_string(), SaveValue::Object(HashMap::new()));
                }
                current = map.get_mut(*segment).unwrap();
            } else {
                return;
            }
        }
    }

    /// Removes a value at the given path.
    pub fn remove_path(&mut self, path: &str) -> Option<SaveValue> {
        let segments: Vec<&str> = path.split('.').collect();
        if segments.is_empty() {
            return None;
        }

        if segments.len() == 1 {
            if let Self::Object(map) = self {
                return map.remove(segments[0]);
            }
            return None;
        }

        let mut current = self;
        for segment in &segments[..segments.len() - 1] {
            if let Self::Object(map) = current {
                current = map.get_mut(*segment)?;
            } else {
                return None;
            }
        }

        if let Self::Object(map) = current {
            map.remove(segments.last().unwrap().to_owned())
        } else {
            None
        }
    }

    /// Renames a key in an object.
    pub fn rename_key(&mut self, old_key: &str, new_key: &str) -> bool {
        if let Self::Object(map) = self {
            if let Some(value) = map.remove(old_key) {
                map.insert(new_key.to_string(), value);
                return true;
            }
        }
        false
    }

    /// Returns the approximate memory size of this value.
    pub fn estimated_size(&self) -> usize {
        match self {
            Self::Null => 8,
            Self::Bool(_) => 8,
            Self::Int(_) => 8,
            Self::Float(_) => 8,
            Self::String(s) => 24 + s.len(),
            Self::Bytes(b) => 24 + b.len(),
            Self::Array(arr) => 24 + arr.iter().map(|v| v.estimated_size()).sum::<usize>(),
            Self::Object(map) => {
                24 + map
                    .iter()
                    .map(|(k, v)| 24 + k.len() + v.estimated_size())
                    .sum::<usize>()
            }
        }
    }

    /// Returns a type name string for debugging.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::String(_) => "string",
            Self::Bytes(_) => "bytes",
            Self::Array(_) => "array",
            Self::Object(_) => "object",
        }
    }
}

impl Default for SaveValue {
    fn default() -> Self {
        Self::Null
    }
}

// ---------------------------------------------------------------------------
// MigrationError
// ---------------------------------------------------------------------------

/// Errors that can occur during save migration.
#[derive(Debug, Clone)]
pub enum MigrationError {
    /// No migration path exists between the two versions.
    NoMigrationPath {
        from: SaveVersion,
        to: SaveVersion,
    },
    /// A migration step failed.
    MigrationFailed {
        from: SaveVersion,
        to: SaveVersion,
        reason: String,
    },
    /// The save file header is invalid.
    InvalidHeader(String),
    /// The save data is structurally invalid.
    ValidationFailed(Vec<ValidationIssue>),
    /// The save data is corrupt beyond recovery.
    Unrecoverable(String),
    /// Version is too old and cannot be migrated.
    VersionTooOld(SaveVersion),
    /// Version is newer than the current engine supports.
    VersionTooNew(SaveVersion),
}

impl fmt::Display for MigrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMigrationPath { from, to } => {
                write!(f, "No migration path from {} to {}", from, to)
            }
            Self::MigrationFailed { from, to, reason } => {
                write!(f, "Migration {} -> {} failed: {}", from, to, reason)
            }
            Self::InvalidHeader(msg) => write!(f, "Invalid save header: {}", msg),
            Self::ValidationFailed(issues) => {
                write!(f, "Validation failed with {} issues", issues.len())
            }
            Self::Unrecoverable(msg) => write!(f, "Unrecoverable: {}", msg),
            Self::VersionTooOld(v) => write!(f, "Version {} is too old to migrate", v),
            Self::VersionTooNew(v) => write!(f, "Version {} is newer than supported", v),
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationIssue
// ---------------------------------------------------------------------------

/// A single validation issue found in a save file.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Path to the problematic data.
    pub path: String,
    /// Severity of the issue.
    pub severity: ValidationSeverity,
    /// Description of the issue.
    pub message: String,
    /// Whether the issue can be auto-repaired.
    pub auto_repairable: bool,
}

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity {
    /// Informational: data is valid but unusual.
    Info,
    /// Warning: data may cause minor issues.
    Warning,
    /// Error: data is invalid and needs repair.
    Error,
    /// Critical: data corruption detected.
    Critical,
}

impl fmt::Display for ValidationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl ValidationIssue {
    /// Creates a new validation issue.
    pub fn new(
        path: impl Into<String>,
        severity: ValidationSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            severity,
            message: message.into(),
            auto_repairable: false,
        }
    }

    /// Marks this issue as auto-repairable.
    pub fn with_auto_repair(mut self) -> Self {
        self.auto_repairable = true;
        self
    }
}

// ---------------------------------------------------------------------------
// ValidationRule
// ---------------------------------------------------------------------------

/// A rule for validating save data structure.
pub struct ValidationRule {
    /// Human-readable name for this rule.
    pub name: String,
    /// Description of what this rule checks.
    pub description: String,
    /// The validation function.
    check_fn: Box<dyn Fn(&SaveValue) -> Vec<ValidationIssue> + Send + Sync>,
}

impl ValidationRule {
    /// Creates a new validation rule.
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        check_fn: F,
    ) -> Self
    where
        F: Fn(&SaveValue) -> Vec<ValidationIssue> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            check_fn: Box::new(check_fn),
        }
    }

    /// Creates a rule that checks for the existence of a required field.
    pub fn required_field(path: &str) -> Self {
        let path_owned = path.to_string();
        Self::new(
            format!("required_{}", path.replace('.', "_")),
            format!("Checks that '{}' exists", path),
            move |data: &SaveValue| {
                if data.get_path(&path_owned).is_none() {
                    vec![ValidationIssue::new(
                        &path_owned,
                        ValidationSeverity::Error,
                        format!("Required field '{}' is missing", path_owned),
                    )]
                } else {
                    Vec::new()
                }
            },
        )
    }

    /// Creates a rule that checks a field has the expected type.
    pub fn field_type(path: &str, expected_type: &str) -> Self {
        let path_owned = path.to_string();
        let type_owned = expected_type.to_string();
        Self::new(
            format!("type_{}_{}", path.replace('.', "_"), expected_type),
            format!("Checks that '{}' is of type '{}'", path, expected_type),
            move |data: &SaveValue| {
                if let Some(value) = data.get_path(&path_owned) {
                    if value.type_name() != type_owned {
                        vec![ValidationIssue::new(
                            &path_owned,
                            ValidationSeverity::Error,
                            format!(
                                "Field '{}' has type '{}', expected '{}'",
                                path_owned,
                                value.type_name(),
                                type_owned
                            ),
                        )]
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            },
        )
    }

    /// Runs this rule against save data.
    pub fn check(&self, data: &SaveValue) -> Vec<ValidationIssue> {
        (self.check_fn)(data)
    }
}

impl fmt::Debug for ValidationRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidationRule")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SaveValidator
// ---------------------------------------------------------------------------

/// Validates save data against a set of rules.
pub struct SaveValidator {
    /// Validation rules.
    rules: Vec<ValidationRule>,
    /// Whether to stop on the first critical issue.
    pub stop_on_critical: bool,
}

impl SaveValidator {
    /// Creates a new save validator.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            stop_on_critical: false,
        }
    }

    /// Adds a validation rule.
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    /// Validates save data against all rules.
    pub fn validate(&self, data: &SaveValue) -> Vec<ValidationIssue> {
        let mut all_issues = Vec::new();
        for rule in &self.rules {
            let issues = rule.check(data);
            let has_critical = issues
                .iter()
                .any(|i| i.severity == ValidationSeverity::Critical);
            all_issues.extend(issues);
            if self.stop_on_critical && has_critical {
                break;
            }
        }
        all_issues
    }

    /// Returns the number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

impl Default for SaveValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MigrationStep
// ---------------------------------------------------------------------------

/// A single migration step that transforms save data from one version to the next.
pub struct MigrationStep {
    /// Source version.
    pub from_version: SaveVersion,
    /// Target version.
    pub to_version: SaveVersion,
    /// Description of what this migration does.
    pub description: String,
    /// The migration function.
    migrate_fn: Box<dyn Fn(&mut SaveValue) -> Result<(), String> + Send + Sync>,
}

impl MigrationStep {
    /// Creates a new migration step.
    pub fn new<F>(
        from: SaveVersion,
        to: SaveVersion,
        description: impl Into<String>,
        migrate_fn: F,
    ) -> Self
    where
        F: Fn(&mut SaveValue) -> Result<(), String> + Send + Sync + 'static,
    {
        Self {
            from_version: from,
            to_version: to,
            description: description.into(),
            migrate_fn: Box::new(migrate_fn),
        }
    }

    /// Applies this migration step to the save data.
    pub fn apply(&self, data: &mut SaveValue) -> Result<(), String> {
        (self.migrate_fn)(data)
    }
}

impl fmt::Debug for MigrationStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MigrationStep")
            .field("from", &self.from_version)
            .field("to", &self.to_version)
            .field("description", &self.description)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RecoveryStrategy
// ---------------------------------------------------------------------------

/// Strategy for recovering from corrupt save data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Use default values for missing or corrupt fields.
    UseDefaults,
    /// Attempt to parse as much data as possible, ignoring errors.
    BestEffort,
    /// Load the most recent valid backup.
    LoadBackup,
    /// Reject the save and report the error.
    Reject,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self::BestEffort
    }
}

// ---------------------------------------------------------------------------
// RecoveryResult
// ---------------------------------------------------------------------------

/// Result of a save recovery attempt.
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Whether recovery was successful.
    pub success: bool,
    /// The recovered data, if successful.
    pub recovered_data: Option<SaveValue>,
    /// Issues found during recovery.
    pub issues: Vec<ValidationIssue>,
    /// Fields that were reset to defaults.
    pub defaulted_fields: Vec<String>,
    /// Strategy that was used.
    pub strategy_used: RecoveryStrategy,
    /// Human-readable summary.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// SaveDiffEntry
// ---------------------------------------------------------------------------

/// A single difference between two save files.
#[derive(Debug, Clone)]
pub struct SaveDiffEntry {
    /// Path to the differing value.
    pub path: String,
    /// The type of difference.
    pub diff_type: DiffType,
    /// Value in the first save (as string representation).
    pub left_value: Option<String>,
    /// Value in the second save (as string representation).
    pub right_value: Option<String>,
}

/// Type of difference between two values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffType {
    /// Value exists only in the left save.
    LeftOnly,
    /// Value exists only in the right save.
    RightOnly,
    /// Value exists in both but differs.
    Changed,
    /// Type of value differs.
    TypeChanged,
}

impl fmt::Display for DiffType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LeftOnly => write!(f, "LEFT_ONLY"),
            Self::RightOnly => write!(f, "RIGHT_ONLY"),
            Self::Changed => write!(f, "CHANGED"),
            Self::TypeChanged => write!(f, "TYPE_CHANGED"),
        }
    }
}

impl fmt::Display for SaveDiffEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: ", self.diff_type, self.path)?;
        match self.diff_type {
            DiffType::LeftOnly => write!(f, "{}", self.left_value.as_deref().unwrap_or("?")),
            DiffType::RightOnly => write!(f, "{}", self.right_value.as_deref().unwrap_or("?")),
            DiffType::Changed | DiffType::TypeChanged => write!(
                f,
                "{} -> {}",
                self.left_value.as_deref().unwrap_or("?"),
                self.right_value.as_deref().unwrap_or("?")
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// SaveDiffer
// ---------------------------------------------------------------------------

/// Compares two save files and produces a diff report.
pub struct SaveDiffer {
    /// Paths to ignore during comparison.
    pub ignore_paths: Vec<String>,
    /// Maximum depth for recursive comparison.
    pub max_depth: usize,
}

impl SaveDiffer {
    /// Creates a new save differ.
    pub fn new() -> Self {
        Self {
            ignore_paths: Vec::new(),
            max_depth: 32,
        }
    }

    /// Adds a path to ignore during comparison.
    pub fn ignore_path(&mut self, path: impl Into<String>) {
        self.ignore_paths.push(path.into());
    }

    /// Compares two save values and returns the differences.
    pub fn diff(&self, left: &SaveValue, right: &SaveValue) -> Vec<SaveDiffEntry> {
        let mut entries = Vec::new();
        self.diff_recursive(left, right, "", 0, &mut entries);
        entries
    }

    /// Produces a human-readable diff summary.
    pub fn diff_summary(&self, left: &SaveValue, right: &SaveValue) -> DiffSummary {
        let entries = self.diff(left, right);
        let mut added = 0;
        let mut removed = 0;
        let mut changed = 0;

        for entry in &entries {
            match entry.diff_type {
                DiffType::RightOnly => added += 1,
                DiffType::LeftOnly => removed += 1,
                DiffType::Changed | DiffType::TypeChanged => changed += 1,
            }
        }

        DiffSummary {
            total_differences: entries.len(),
            added,
            removed,
            changed,
            entries,
        }
    }

    /// Recursively compares two values.
    fn diff_recursive(
        &self,
        left: &SaveValue,
        right: &SaveValue,
        path: &str,
        depth: usize,
        entries: &mut Vec<SaveDiffEntry>,
    ) {
        if depth > self.max_depth {
            return;
        }

        // Check if this path should be ignored.
        if self.ignore_paths.iter().any(|p| path.starts_with(p)) {
            return;
        }

        // Check type mismatch.
        if std::mem::discriminant(left) != std::mem::discriminant(right) {
            entries.push(SaveDiffEntry {
                path: path.to_string(),
                diff_type: DiffType::TypeChanged,
                left_value: Some(format!("({}) {:?}", left.type_name(), value_preview(left))),
                right_value: Some(format!("({}) {:?}", right.type_name(), value_preview(right))),
            });
            return;
        }

        match (left, right) {
            (SaveValue::Object(left_map), SaveValue::Object(right_map)) => {
                // Check keys in left but not right.
                for key in left_map.keys() {
                    let child_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", path, key)
                    };
                    if !right_map.contains_key(key) {
                        entries.push(SaveDiffEntry {
                            path: child_path,
                            diff_type: DiffType::LeftOnly,
                            left_value: Some(value_preview(left_map.get(key).unwrap())),
                            right_value: None,
                        });
                    }
                }
                // Check keys in right but not left.
                for key in right_map.keys() {
                    let child_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", path, key)
                    };
                    if !left_map.contains_key(key) {
                        entries.push(SaveDiffEntry {
                            path: child_path,
                            diff_type: DiffType::RightOnly,
                            left_value: None,
                            right_value: Some(value_preview(right_map.get(key).unwrap())),
                        });
                    }
                }
                // Recursively compare shared keys.
                for key in left_map.keys() {
                    if let Some(right_val) = right_map.get(key) {
                        let child_path = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{}.{}", path, key)
                        };
                        self.diff_recursive(
                            left_map.get(key).unwrap(),
                            right_val,
                            &child_path,
                            depth + 1,
                            entries,
                        );
                    }
                }
            }
            (SaveValue::Array(left_arr), SaveValue::Array(right_arr)) => {
                let max_len = left_arr.len().max(right_arr.len());
                for i in 0..max_len {
                    let child_path = format!("{}[{}]", path, i);
                    match (left_arr.get(i), right_arr.get(i)) {
                        (Some(l), Some(r)) => {
                            self.diff_recursive(l, r, &child_path, depth + 1, entries);
                        }
                        (Some(l), None) => {
                            entries.push(SaveDiffEntry {
                                path: child_path,
                                diff_type: DiffType::LeftOnly,
                                left_value: Some(value_preview(l)),
                                right_value: None,
                            });
                        }
                        (None, Some(r)) => {
                            entries.push(SaveDiffEntry {
                                path: child_path,
                                diff_type: DiffType::RightOnly,
                                left_value: None,
                                right_value: Some(value_preview(r)),
                            });
                        }
                        (None, None) => {}
                    }
                }
            }
            _ => {
                let left_str = value_preview(left);
                let right_str = value_preview(right);
                if left_str != right_str {
                    entries.push(SaveDiffEntry {
                        path: path.to_string(),
                        diff_type: DiffType::Changed,
                        left_value: Some(left_str),
                        right_value: Some(right_str),
                    });
                }
            }
        }
    }
}

impl Default for SaveDiffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of differences between two saves.
#[derive(Debug, Clone)]
pub struct DiffSummary {
    /// Total number of differences.
    pub total_differences: usize,
    /// Number of values added (exist only in right).
    pub added: usize,
    /// Number of values removed (exist only in left).
    pub removed: usize,
    /// Number of values changed.
    pub changed: usize,
    /// All diff entries.
    pub entries: Vec<SaveDiffEntry>,
}

impl fmt::Display for DiffSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Diff: {} total ({} added, {} removed, {} changed)",
            self.total_differences, self.added, self.removed, self.changed
        )
    }
}

/// Returns a short string preview of a SaveValue for diff display.
fn value_preview(value: &SaveValue) -> String {
    match value {
        SaveValue::Null => "null".to_string(),
        SaveValue::Bool(v) => v.to_string(),
        SaveValue::Int(v) => v.to_string(),
        SaveValue::Float(v) => format!("{:.4}", v),
        SaveValue::String(v) => {
            if v.len() > 50 {
                format!("\"{}...\"", &v[..47])
            } else {
                format!("\"{}\"", v)
            }
        }
        SaveValue::Bytes(b) => format!("<{} bytes>", b.len()),
        SaveValue::Array(a) => format!("[{} items]", a.len()),
        SaveValue::Object(o) => format!("{{{} keys}}", o.len()),
    }
}

// ---------------------------------------------------------------------------
// SaveMigrationManager
// ---------------------------------------------------------------------------

/// Central manager for save file versioning and migration.
///
/// The migration manager owns the version registry, migration chain,
/// validator, and recovery logic. It provides a one-call `migrate()`
/// method that handles the full upgrade path from any supported version
/// to the current version.
pub struct SaveMigrationManager {
    /// The current (latest) save version.
    pub current_version: SaveVersion,
    /// Minimum supported save version (older saves are rejected).
    pub minimum_version: SaveVersion,
    /// Registered migration steps.
    migrations: Vec<MigrationStep>,
    /// Save data validator.
    pub validator: SaveValidator,
    /// Recovery strategy for corrupt saves.
    pub recovery_strategy: RecoveryStrategy,
    /// Save file differ.
    pub differ: SaveDiffer,
    /// Version descriptions.
    version_descriptions: HashMap<SaveVersion, String>,
    /// Default values for recovery.
    default_values: HashMap<String, SaveValue>,
}

impl SaveMigrationManager {
    /// Creates a new migration manager.
    pub fn new(current_version: SaveVersion) -> Self {
        Self {
            current_version,
            minimum_version: SaveVersion::new(1, 0, 0),
            migrations: Vec::new(),
            validator: SaveValidator::new(),
            recovery_strategy: RecoveryStrategy::BestEffort,
            differ: SaveDiffer::new(),
            version_descriptions: HashMap::new(),
            default_values: HashMap::new(),
        }
    }

    /// Sets the minimum supported version.
    pub fn set_minimum_version(&mut self, version: SaveVersion) {
        self.minimum_version = version;
    }

    /// Registers a version description.
    pub fn register_version(&mut self, version: SaveVersion, description: impl Into<String>) {
        self.version_descriptions
            .insert(version, description.into());
    }

    /// Registers a migration step.
    pub fn register_migration(&mut self, step: MigrationStep) {
        self.migrations.push(step);
        // Sort migrations by source version.
        self.migrations
            .sort_by(|a, b| a.from_version.numeric().cmp(&b.from_version.numeric()));
    }

    /// Registers a default value for recovery.
    pub fn register_default(&mut self, path: impl Into<String>, value: SaveValue) {
        self.default_values.insert(path.into(), value);
    }

    /// Validates a save header.
    pub fn validate_header(&self, header: &SaveHeader) -> Result<(), MigrationError> {
        if !header.validate_magic() {
            return Err(MigrationError::InvalidHeader(
                "Invalid magic bytes".to_string(),
            ));
        }

        if header.version.is_newer_than(&self.current_version) {
            return Err(MigrationError::VersionTooNew(header.version));
        }

        if self.minimum_version.is_newer_than(&header.version) {
            return Err(MigrationError::VersionTooOld(header.version));
        }

        Ok(())
    }

    /// Migrates save data from the given version to the current version.
    ///
    /// Applies migration steps in sequence: v1->v2->v3->...->current.
    pub fn migrate(
        &self,
        data: &mut SaveValue,
        from_version: SaveVersion,
    ) -> Result<MigrationReport, MigrationError> {
        if from_version == self.current_version {
            return Ok(MigrationReport {
                from_version,
                to_version: self.current_version,
                steps_applied: Vec::new(),
                warnings: Vec::new(),
                success: true,
            });
        }

        if self.minimum_version.is_newer_than(&from_version) {
            return Err(MigrationError::VersionTooOld(from_version));
        }

        // Build the migration chain.
        let chain = self.build_chain(from_version, self.current_version)?;

        let mut report = MigrationReport {
            from_version,
            to_version: self.current_version,
            steps_applied: Vec::new(),
            warnings: Vec::new(),
            success: true,
        };

        for step in &chain {
            match step.apply(data) {
                Ok(()) => {
                    report.steps_applied.push(format!(
                        "{} -> {}: {}",
                        step.from_version, step.to_version, step.description
                    ));
                }
                Err(reason) => {
                    return Err(MigrationError::MigrationFailed {
                        from: step.from_version,
                        to: step.to_version,
                        reason,
                    });
                }
            }
        }

        Ok(report)
    }

    /// Validates save data against all registered rules.
    pub fn validate(&self, data: &SaveValue) -> Vec<ValidationIssue> {
        self.validator.validate(data)
    }

    /// Attempts to recover corrupt save data.
    pub fn recover(
        &self,
        data: &mut SaveValue,
        strategy: RecoveryStrategy,
    ) -> RecoveryResult {
        let issues = self.validate(data);
        let critical_count = issues
            .iter()
            .filter(|i| i.severity >= ValidationSeverity::Error)
            .count();

        if critical_count == 0 {
            return RecoveryResult {
                success: true,
                recovered_data: None,
                issues,
                defaulted_fields: Vec::new(),
                strategy_used: strategy,
                summary: "No recovery needed".to_string(),
            };
        }

        match strategy {
            RecoveryStrategy::UseDefaults => {
                let mut defaulted = Vec::new();
                for (path, default_value) in &self.default_values {
                    if data.get_path(path).is_none() {
                        data.set_path(path, default_value.clone());
                        defaulted.push(path.clone());
                    }
                }
                RecoveryResult {
                    success: true,
                    recovered_data: Some(data.clone()),
                    issues,
                    defaulted_fields: defaulted,
                    strategy_used: strategy,
                    summary: "Recovery applied using default values".to_string(),
                }
            }
            RecoveryStrategy::BestEffort => {
                // Auto-repair any auto-repairable issues.
                let mut repaired = Vec::new();
                for issue in &issues {
                    if issue.auto_repairable {
                        if let Some(default) = self.default_values.get(&issue.path) {
                            data.set_path(&issue.path, default.clone());
                            repaired.push(issue.path.clone());
                        }
                    }
                }
                RecoveryResult {
                    success: true,
                    recovered_data: Some(data.clone()),
                    issues,
                    defaulted_fields: repaired,
                    strategy_used: strategy,
                    summary: "Best-effort recovery applied".to_string(),
                }
            }
            RecoveryStrategy::LoadBackup => {
                RecoveryResult {
                    success: false,
                    recovered_data: None,
                    issues,
                    defaulted_fields: Vec::new(),
                    strategy_used: strategy,
                    summary: "Backup loading not implemented at this level".to_string(),
                }
            }
            RecoveryStrategy::Reject => {
                RecoveryResult {
                    success: false,
                    recovered_data: None,
                    issues,
                    defaulted_fields: Vec::new(),
                    strategy_used: strategy,
                    summary: "Save rejected due to corruption".to_string(),
                }
            }
        }
    }

    /// Compares two save values.
    pub fn diff(&self, left: &SaveValue, right: &SaveValue) -> DiffSummary {
        self.differ.diff_summary(left, right)
    }

    /// Returns all registered version descriptions.
    pub fn version_history(&self) -> Vec<(SaveVersion, &str)> {
        let mut versions: Vec<_> = self
            .version_descriptions
            .iter()
            .map(|(v, d)| (*v, d.as_str()))
            .collect();
        versions.sort_by(|a, b| a.0.numeric().cmp(&b.0.numeric()));
        versions
    }

    /// Returns the number of registered migration steps.
    pub fn migration_count(&self) -> usize {
        self.migrations.len()
    }

    /// Builds the migration chain from source to target version.
    fn build_chain(
        &self,
        from: SaveVersion,
        to: SaveVersion,
    ) -> Result<Vec<&MigrationStep>, MigrationError> {
        let mut chain = Vec::new();
        let mut current = from;

        while current != to {
            let step = self
                .migrations
                .iter()
                .find(|s| s.from_version == current);

            match step {
                Some(s) => {
                    chain.push(s);
                    current = s.to_version;
                }
                None => {
                    return Err(MigrationError::NoMigrationPath { from: current, to });
                }
            }
        }

        Ok(chain)
    }
}

impl Default for SaveMigrationManager {
    fn default() -> Self {
        Self::new(SaveVersion::new(1, 0, 0))
    }
}

impl fmt::Debug for SaveMigrationManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SaveMigrationManager")
            .field("current_version", &self.current_version)
            .field("minimum_version", &self.minimum_version)
            .field("migration_count", &self.migrations.len())
            .field("rule_count", &self.validator.rule_count())
            .finish()
    }
}

/// Report of a completed migration.
#[derive(Debug, Clone)]
pub struct MigrationReport {
    /// Source version.
    pub from_version: SaveVersion,
    /// Target version.
    pub to_version: SaveVersion,
    /// Steps that were applied.
    pub steps_applied: Vec<String>,
    /// Warnings generated during migration.
    pub warnings: Vec<String>,
    /// Whether migration was successful.
    pub success: bool,
}

impl fmt::Display for MigrationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Migration {} -> {}: {} ({} steps)",
            self.from_version,
            self.to_version,
            if self.success { "SUCCESS" } else { "FAILED" },
            self.steps_applied.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_version_ordering() {
        let v1 = SaveVersion::new(1, 0, 0);
        let v2 = SaveVersion::new(2, 0, 0);
        let v1_1 = SaveVersion::new(1, 1, 0);
        assert!(v2.is_newer_than(&v1));
        assert!(v1_1.is_newer_than(&v1));
        assert!(!v1.is_newer_than(&v2));
    }

    #[test]
    fn test_save_value_path() {
        let mut data = SaveValue::Object(HashMap::new());
        data.set_path("player.health", SaveValue::Float(100.0));
        assert_eq!(
            data.get_path("player.health")
                .and_then(|v| v.as_float()),
            Some(100.0)
        );
    }

    #[test]
    fn test_save_value_rename() {
        let mut data = SaveValue::Object(HashMap::new());
        if let SaveValue::Object(ref mut map) = data {
            map.insert("old_key".to_string(), SaveValue::Int(42));
        }
        assert!(data.rename_key("old_key", "new_key"));
        assert!(data.get_path("new_key").is_some());
        assert!(data.get_path("old_key").is_none());
    }

    #[test]
    fn test_migration_chain() {
        let mut manager = SaveMigrationManager::new(SaveVersion::new(3, 0, 0));
        manager.register_migration(MigrationStep::new(
            SaveVersion::new(1, 0, 0),
            SaveVersion::new(2, 0, 0),
            "Add player inventory",
            |data| {
                data.set_path("player.inventory", SaveValue::Array(Vec::new()));
                Ok(())
            },
        ));
        manager.register_migration(MigrationStep::new(
            SaveVersion::new(2, 0, 0),
            SaveVersion::new(3, 0, 0),
            "Add quest log",
            |data| {
                data.set_path("quests", SaveValue::Array(Vec::new()));
                Ok(())
            },
        ));

        let mut data = SaveValue::Object(HashMap::new());
        let report = manager.migrate(&mut data, SaveVersion::new(1, 0, 0)).unwrap();
        assert!(report.success);
        assert_eq!(report.steps_applied.len(), 2);
    }

    #[test]
    fn test_validation() {
        let mut validator = SaveValidator::new();
        validator.add_rule(ValidationRule::required_field("player"));
        validator.add_rule(ValidationRule::required_field("player.health"));

        let mut data = SaveValue::Object(HashMap::new());
        data.set_path("player.health", SaveValue::Float(100.0));

        let issues = validator.validate(&data);
        assert!(issues.is_empty());

        let empty_data = SaveValue::Object(HashMap::new());
        let issues = validator.validate(&empty_data);
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_save_diff() {
        let differ = SaveDiffer::new();

        let mut left = SaveValue::Object(HashMap::new());
        left.set_path("health", SaveValue::Float(100.0));
        left.set_path("name", SaveValue::String("Player1".to_string()));

        let mut right = SaveValue::Object(HashMap::new());
        right.set_path("health", SaveValue::Float(75.0));
        right.set_path("name", SaveValue::String("Player1".to_string()));
        right.set_path("level", SaveValue::Int(5));

        let summary = differ.diff_summary(&left, &right);
        assert!(summary.total_differences > 0);
        assert_eq!(summary.changed, 1); // health
        assert_eq!(summary.added, 1);   // level
    }

    #[test]
    fn test_header_validation() {
        let manager = SaveMigrationManager::new(SaveVersion::new(3, 0, 0));
        let header = SaveHeader::new(SaveVersion::new(2, 0, 0));
        assert!(manager.validate_header(&header).is_ok());

        let future_header = SaveHeader::new(SaveVersion::new(5, 0, 0));
        assert!(manager.validate_header(&future_header).is_err());
    }

    #[test]
    fn test_no_migration_needed() {
        let manager = SaveMigrationManager::new(SaveVersion::new(1, 0, 0));
        let mut data = SaveValue::Object(HashMap::new());
        let report = manager.migrate(&mut data, SaveVersion::new(1, 0, 0)).unwrap();
        assert!(report.success);
        assert_eq!(report.steps_applied.len(), 0);
    }
}
