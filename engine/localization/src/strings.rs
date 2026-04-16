//! String tables with template substitution, pluralization, and gender selection.
//!
//! Supports:
//! - Simple parameter substitution: `"Hello {name}!"`
//! - ICU-style pluralization: `"{count, plural, one{# item} other{# items}}"`
//! - Gender selection: `"{gender, select, male{He} female{She} other{They}}"`
//! - Missing string detection and fallback chains.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::formatting::PluralRules;
use crate::locale::LocaleId;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from string table operations.
#[derive(Debug, thiserror::Error)]
pub enum StringError {
    #[error("String key not found: {0}")]
    KeyNotFound(String),
    #[error("Template parse error in \"{key}\": {message}")]
    TemplateError { key: String, message: String },
    #[error("JSON parse error: {0}")]
    JsonError(String),
    #[error("Missing parameter \"{0}\" in template")]
    MissingParameter(String),
}

// ---------------------------------------------------------------------------
// StringTable
// ---------------------------------------------------------------------------

/// Key-value string storage for a single locale.
///
/// Strings can be simple text or templates containing parameter placeholders,
/// plural forms, and gender-select expressions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringTable {
    /// Locale this table belongs to.
    pub locale: LocaleId,
    /// Raw string entries keyed by string ID.
    entries: HashMap<String, String>,
    /// Plural rules for this locale.
    #[serde(skip)]
    plural_rules: Option<PluralRules>,
}

impl StringTable {
    /// Create a new empty string table for a locale.
    pub fn new() -> Self {
        Self {
            locale: LocaleId::EnUS,
            entries: HashMap::new(),
            plural_rules: None,
        }
    }

    /// Create a string table for a specific locale.
    pub fn for_locale(locale: LocaleId) -> Self {
        let plural_rules = Some(PluralRules::for_locale(&locale));
        Self {
            locale,
            entries: HashMap::new(),
            plural_rules,
        }
    }

    /// Insert or update a string entry.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.entries.insert(key.into(), value.into());
    }

    /// Remove a string entry.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.entries.remove(key)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Get the raw template string for a key.
    pub fn get_raw(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(|s| s.as_str())
    }

    /// Get a localized string by key.
    ///
    /// If the key is not found, returns the key itself and logs a warning.
    pub fn get<'a>(&'a self, key: &'a str) -> &'a str {
        match self.entries.get(key) {
            Some(s) => s.as_str(),
            None => {
                log::warn!(
                    "[L10N] Missing string key \"{}\" for locale {}",
                    key,
                    self.locale
                );
                key
            }
        }
    }

    /// Get a formatted string with parameter substitution.
    ///
    /// Supports:
    /// - `{name}` -- simple substitution
    /// - `{count, plural, one{...} other{...}}` -- pluralization
    /// - `{gender, select, male{...} female{...} other{...}}` -- selection
    pub fn get_formatted(&self, key: &str, args: &[(&str, &str)]) -> String {
        let template = match self.entries.get(key) {
            Some(s) => s.as_str(),
            None => {
                log::warn!(
                    "[L10N] Missing string key \"{}\" for locale {}",
                    key,
                    self.locale
                );
                return key.to_string();
            }
        };

        let arg_map: HashMap<&str, &str> = args.iter().copied().collect();
        self.expand_template(template, &arg_map)
    }

    /// Expand a template string with the given arguments.
    fn expand_template(&self, template: &str, args: &HashMap<&str, &str>) -> String {
        let mut result = String::with_capacity(template.len());
        let chars: Vec<char> = template.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i] == '{' {
                // Find matching closing brace (handling nesting).
                if let Some(end) = find_matching_brace(&chars, i) {
                    let inner: String = chars[i + 1..end].iter().collect();
                    let expanded = self.expand_expression(&inner, args);
                    result.push_str(&expanded);
                    i = end + 1;
                } else {
                    // Unmatched brace -- emit literally.
                    result.push('{');
                    i += 1;
                }
            } else {
                result.push(chars[i]);
                i += 1;
            }
        }

        result
    }

    /// Expand a single `{...}` expression.
    fn expand_expression(&self, expr: &str, args: &HashMap<&str, &str>) -> String {
        let trimmed = expr.trim();

        // Check for ICU-style: "count, plural, ..." or "gender, select, ..."
        if let Some(comma_pos) = trimmed.find(',') {
            let var_name = trimmed[..comma_pos].trim();
            let rest = trimmed[comma_pos + 1..].trim();

            if rest.starts_with("plural,") || rest.starts_with("plural ") {
                let plural_body = rest.splitn(2, |c: char| c == ',' || c == ' ').nth(1).unwrap_or("").trim();
                return self.expand_plural(var_name, plural_body, args);
            }
            if rest.starts_with("select,") || rest.starts_with("select ") {
                let select_body = rest.splitn(2, |c: char| c == ',' || c == ' ').nth(1).unwrap_or("").trim();
                return self.expand_select(var_name, select_body, args);
            }
        }

        // Simple parameter substitution.
        match args.get(trimmed) {
            Some(val) => val.to_string(),
            None => {
                log::warn!("[L10N] Missing parameter \"{}\" in template", trimmed);
                format!("{{{trimmed}}}")
            }
        }
    }

    /// Expand a plural expression.
    ///
    /// Format: `one{# item} few{# items-few} other{# items}`
    /// The `#` character is replaced with the actual count.
    fn expand_plural(
        &self,
        var_name: &str,
        body: &str,
        args: &HashMap<&str, &str>,
    ) -> String {
        let count_str = args.get(var_name).copied().unwrap_or("0");
        let count: f64 = count_str.parse().unwrap_or(0.0);

        let rules = self
            .plural_rules
            .as_ref()
            .cloned()
            .unwrap_or_else(|| PluralRules::for_locale(&self.locale));
        let category = rules.select(count);
        let category_name = category.as_str();

        // Parse the plural body: "one{...} other{...}"
        let cases = parse_icu_cases(body);

        // Look up the category, falling back to "other".
        let template = cases
            .get(category_name)
            .or_else(|| cases.get("other"))
            .map(|s| s.as_str())
            .unwrap_or(count_str);

        // Replace # with the count value.
        template.replace('#', count_str)
    }

    /// Expand a select expression.
    ///
    /// Format: `male{He} female{She} other{They}`
    fn expand_select(
        &self,
        var_name: &str,
        body: &str,
        args: &HashMap<&str, &str>,
    ) -> String {
        let value = args.get(var_name).copied().unwrap_or("other");

        let cases = parse_icu_cases(body);

        cases
            .get(value)
            .or_else(|| cases.get("other"))
            .cloned()
            .unwrap_or_else(|| value.to_string())
    }

    /// Get all keys in this table.
    pub fn keys(&self) -> Vec<&str> {
        self.entries.keys().map(|k| k.as_str()).collect()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Merge another string table into this one.
    ///
    /// Existing keys are overwritten.
    pub fn merge(&mut self, other: &StringTable) {
        for (key, value) in &other.entries {
            self.entries.insert(key.clone(), value.clone());
        }
    }
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StringTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StringTable({}, {} entries)",
            self.locale,
            self.entries.len()
        )
    }
}

// ---------------------------------------------------------------------------
// ICU case parser
// ---------------------------------------------------------------------------

/// Parse ICU-style case expressions: `one{text one} other{text other}`
///
/// Returns a map of category name -> text.
fn parse_icu_cases(body: &str) -> HashMap<String, String> {
    let mut cases = HashMap::new();
    let chars: Vec<char> = body.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Skip whitespace.
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }

        // Read category name (until '{').
        let name_start = i;
        while i < chars.len() && chars[i] != '{' && !chars[i].is_whitespace() {
            i += 1;
        }
        let name: String = chars[name_start..i].iter().collect();
        if name.is_empty() {
            i += 1;
            continue;
        }

        // Skip whitespace between name and '{'.
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }

        // Expect '{'.
        if i >= chars.len() || chars[i] != '{' {
            break;
        }

        // Find matching '}'.
        if let Some(end) = find_matching_brace(&chars, i) {
            let content: String = chars[i + 1..end].iter().collect();
            cases.insert(name, content);
            i = end + 1;
        } else {
            break;
        }
    }

    cases
}

/// Find the matching closing brace for an opening brace at `start`.
/// Handles nested braces.
fn find_matching_brace(chars: &[char], start: usize) -> Option<usize> {
    if start >= chars.len() || chars[start] != '{' {
        return None;
    }
    let mut depth = 1;
    let mut i = start + 1;
    while i < chars.len() {
        match chars[i] {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// StringTableLoader
// ---------------------------------------------------------------------------

/// Loads string tables from JSON files.
///
/// Expected JSON format:
/// ```json
/// {
///   "greeting": "Hello {name}!",
///   "items": "{count, plural, one{# item} other{# items}}",
///   "quit_confirm": "Are you sure you want to quit?"
/// }
/// ```
pub struct StringTableLoader;

impl StringTableLoader {
    /// Load a string table from a JSON string.
    pub fn from_json(json: &str, locale: LocaleId) -> Result<StringTable, StringError> {
        let map: HashMap<String, String> =
            serde_json::from_str(json).map_err(|e| StringError::JsonError(e.to_string()))?;

        let mut table = StringTable::for_locale(locale);
        for (key, value) in map {
            table.insert(key, value);
        }

        log::info!("[L10N] Loaded {} strings for {}", table.len(), table.locale);
        Ok(table)
    }

    /// Load a string table from a JSON file path.
    pub fn from_file(path: &std::path::Path, locale: LocaleId) -> Result<StringTable, StringError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| StringError::JsonError(format!("Failed to read {}: {}", path.display(), e)))?;
        Self::from_json(&contents, locale)
    }

    /// Serialize a string table to JSON.
    pub fn to_json(table: &StringTable) -> Result<String, StringError> {
        let map: HashMap<&str, &str> = table.iter().collect();
        serde_json::to_string_pretty(&map).map_err(|e| StringError::JsonError(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// StringValidation
// ---------------------------------------------------------------------------

/// Result of validating string tables across locales.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// The string key with an issue.
    pub key: String,
    /// Description of the issue.
    pub message: String,
    /// Locale where the issue was found.
    pub locale: LocaleId,
    /// Severity level.
    pub severity: IssueSeverity,
}

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Missing translation -- string exists in reference but not in target.
    Error,
    /// Possible issue (e.g., parameter mismatch).
    Warning,
    /// Informational (e.g., unused key).
    Info,
}

/// String table validation utilities.
pub struct StringValidation;

impl StringValidation {
    /// Find keys present in the reference table but missing in the target.
    pub fn find_missing_keys(
        reference: &StringTable,
        target: &StringTable,
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        for key in reference.keys() {
            if !target.contains_key(key) {
                issues.push(ValidationIssue {
                    key: key.to_string(),
                    message: format!(
                        "Key \"{}\" exists in {} but is missing in {}",
                        key, reference.locale, target.locale
                    ),
                    locale: target.locale.clone(),
                    severity: IssueSeverity::Error,
                });
            }
        }
        issues
    }

    /// Find keys present in the target but not in the reference (unused).
    pub fn find_unused_keys(
        reference: &StringTable,
        target: &StringTable,
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        for key in target.keys() {
            if !reference.contains_key(key) {
                issues.push(ValidationIssue {
                    key: key.to_string(),
                    message: format!(
                        "Key \"{}\" exists in {} but not in reference {}",
                        key, target.locale, reference.locale
                    ),
                    locale: target.locale.clone(),
                    severity: IssueSeverity::Info,
                });
            }
        }
        issues
    }

    /// Extract parameter names from a template string.
    pub fn extract_parameters(template: &str) -> Vec<String> {
        let mut params = Vec::new();
        let chars: Vec<char> = template.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i] == '{' {
                if let Some(end) = find_matching_brace(&chars, i) {
                    let inner: String = chars[i + 1..end].iter().collect();
                    let trimmed = inner.trim();

                    // Check if it's a simple parameter or ICU expression.
                    if let Some(comma_pos) = trimmed.find(',') {
                        let param = trimmed[..comma_pos].trim().to_string();
                        if !params.contains(&param) {
                            params.push(param);
                        }
                    } else {
                        let param = trimmed.to_string();
                        if !params.contains(&param) {
                            params.push(param);
                        }
                    }
                    i = end + 1;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        params
    }

    /// Check that two templates have the same parameters.
    pub fn check_parameter_consistency(
        key: &str,
        reference_template: &str,
        target_template: &str,
        target_locale: &LocaleId,
    ) -> Vec<ValidationIssue> {
        let ref_params = Self::extract_parameters(reference_template);
        let target_params = Self::extract_parameters(target_template);

        let mut issues = Vec::new();

        for param in &ref_params {
            if !target_params.contains(param) {
                issues.push(ValidationIssue {
                    key: key.to_string(),
                    message: format!(
                        "Parameter \"{}\" in reference but missing in {} translation",
                        param, target_locale
                    ),
                    locale: target_locale.clone(),
                    severity: IssueSeverity::Warning,
                });
            }
        }

        for param in &target_params {
            if !ref_params.contains(param) {
                issues.push(ValidationIssue {
                    key: key.to_string(),
                    message: format!(
                        "Parameter \"{}\" in {} translation but not in reference",
                        param, target_locale
                    ),
                    locale: target_locale.clone(),
                    severity: IssueSeverity::Warning,
                });
            }
        }

        issues
    }

    /// Validate a target table against a reference table.
    pub fn validate(
        reference: &StringTable,
        target: &StringTable,
    ) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Missing keys.
        issues.extend(Self::find_missing_keys(reference, target));

        // Unused keys.
        issues.extend(Self::find_unused_keys(reference, target));

        // Parameter consistency.
        for key in reference.keys() {
            if let (Some(ref_tmpl), Some(tgt_tmpl)) = (
                reference.get_raw(key),
                target.get_raw(key),
            ) {
                issues.extend(Self::check_parameter_consistency(
                    key,
                    ref_tmpl,
                    tgt_tmpl,
                    &target.locale,
                ));
            }
        }

        issues
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_get() {
        let mut table = StringTable::for_locale(LocaleId::EnUS);
        table.insert("hello", "Hello World!");
        assert_eq!(table.get("hello"), "Hello World!");
    }

    #[test]
    fn missing_key_returns_key() {
        let table = StringTable::for_locale(LocaleId::EnUS);
        assert_eq!(table.get("nonexistent"), "nonexistent");
    }

    #[test]
    fn simple_substitution() {
        let mut table = StringTable::for_locale(LocaleId::EnUS);
        table.insert("greeting", "Hello {name}!");
        let result = table.get_formatted("greeting", &[("name", "Alice")]);
        assert_eq!(result, "Hello Alice!");
    }

    #[test]
    fn multiple_substitutions() {
        let mut table = StringTable::for_locale(LocaleId::EnUS);
        table.insert("welcome", "Welcome {name} to {place}!");
        let result = table.get_formatted("welcome", &[("name", "Bob"), ("place", "Genovo")]);
        assert_eq!(result, "Welcome Bob to Genovo!");
    }

    #[test]
    fn plural_one_other() {
        let mut table = StringTable::for_locale(LocaleId::EnUS);
        table.insert("items", "{count, plural, one{# item} other{# items}}");

        let one = table.get_formatted("items", &[("count", "1")]);
        assert_eq!(one, "1 item");

        let many = table.get_formatted("items", &[("count", "5")]);
        assert_eq!(many, "5 items");

        let zero = table.get_formatted("items", &[("count", "0")]);
        assert_eq!(zero, "0 items");
    }

    #[test]
    fn gender_select() {
        let mut table = StringTable::for_locale(LocaleId::EnUS);
        table.insert(
            "pronoun",
            "{gender, select, male{He} female{She} other{They}}",
        );

        let male = table.get_formatted("pronoun", &[("gender", "male")]);
        assert_eq!(male, "He");

        let female = table.get_formatted("pronoun", &[("gender", "female")]);
        assert_eq!(female, "She");

        let other = table.get_formatted("pronoun", &[("gender", "unknown")]);
        assert_eq!(other, "They");
    }

    #[test]
    fn nested_expression() {
        let mut table = StringTable::for_locale(LocaleId::EnUS);
        table.insert("msg", "You have {count, plural, one{# new message} other{# new messages}}");

        let result = table.get_formatted("msg", &[("count", "3")]);
        assert_eq!(result, "You have 3 new messages");
    }

    #[test]
    fn json_loading() {
        let json = r#"{
            "hello": "Hello!",
            "goodbye": "Goodbye!",
            "items": "{count, plural, one{# thing} other{# things}}"
        }"#;

        let table = StringTableLoader::from_json(json, LocaleId::EnUS).unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(table.get("hello"), "Hello!");
    }

    #[test]
    fn parameter_extraction() {
        let params = StringValidation::extract_parameters("Hello {name}, you have {count, plural, one{# item} other{# items}}");
        assert_eq!(params, vec!["name".to_string(), "count".to_string()]);
    }

    #[test]
    fn missing_key_validation() {
        let mut reference = StringTable::for_locale(LocaleId::EnUS);
        reference.insert("a", "Alpha");
        reference.insert("b", "Beta");

        let mut target = StringTable::for_locale(LocaleId::FrFR);
        target.insert("a", "Alpha-FR");
        // "b" is missing.

        let issues = StringValidation::find_missing_keys(&reference, &target);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].key, "b");
    }

    #[test]
    fn parameter_consistency() {
        let issues = StringValidation::check_parameter_consistency(
            "test",
            "Hello {name}!",
            "Bonjour!",
            &LocaleId::FrFR,
        );
        assert!(!issues.is_empty());
        assert_eq!(issues[0].severity, IssueSeverity::Warning);
    }

    #[test]
    fn merge_tables() {
        let mut base = StringTable::for_locale(LocaleId::EnUS);
        base.insert("a", "Alpha");
        base.insert("b", "Beta");

        let mut overlay = StringTable::for_locale(LocaleId::EnUS);
        overlay.insert("b", "Beta-Override");
        overlay.insert("c", "Charlie");

        base.merge(&overlay);
        assert_eq!(base.get("a"), "Alpha");
        assert_eq!(base.get("b"), "Beta-Override");
        assert_eq!(base.get("c"), "Charlie");
    }
}
