//! Enhanced localization: dynamic string loading, font switching per locale,
//! text direction (LTR/RTL), locale-specific number/date, string validation.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextDirectionV2 { LeftToRight, RightToLeft }

#[derive(Debug, Clone)]
pub struct LocaleInfoV2 {
    pub code: String, pub name: String, pub native_name: String,
    pub text_direction: TextDirectionV2, pub font_family: String,
    pub fallback_font: Option<String>, pub decimal_separator: char,
    pub thousands_separator: char, pub date_format: String,
    pub time_format: String, pub currency_symbol: String,
    pub currency_position: CurrencyPosition, pub plural_rules: PluralRuleSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurrencyPosition { Before, After }

#[derive(Debug, Clone)]
pub struct PluralRuleSet { pub zero: Option<String>, pub one: String, pub two: Option<String>, pub few: Option<String>, pub many: Option<String>, pub other: String }
impl Default for PluralRuleSet { fn default() -> Self { Self { zero: None, one: "one".to_string(), two: None, few: None, many: None, other: "other".to_string() } } }

impl LocaleInfoV2 {
    pub fn english() -> Self { Self { code: "en-US".to_string(), name: "English (US)".to_string(), native_name: "English (US)".to_string(), text_direction: TextDirectionV2::LeftToRight, font_family: "Roboto".to_string(), fallback_font: None, decimal_separator: '.', thousands_separator: ',', date_format: "MM/DD/YYYY".to_string(), time_format: "hh:mm A".to_string(), currency_symbol: "$".to_string(), currency_position: CurrencyPosition::Before, plural_rules: PluralRuleSet::default() } }
    pub fn arabic() -> Self { Self { code: "ar-SA".to_string(), name: "Arabic".to_string(), native_name: "Arabic".to_string(), text_direction: TextDirectionV2::RightToLeft, font_family: "Noto Sans Arabic".to_string(), fallback_font: Some("Arial".to_string()), decimal_separator: '.', thousands_separator: ',', date_format: "DD/MM/YYYY".to_string(), time_format: "HH:mm".to_string(), currency_symbol: "SAR".to_string(), currency_position: CurrencyPosition::After, plural_rules: PluralRuleSet::default() } }
    pub fn japanese() -> Self { Self { code: "ja-JP".to_string(), name: "Japanese".to_string(), native_name: "Japanese".to_string(), text_direction: TextDirectionV2::LeftToRight, font_family: "Noto Sans JP".to_string(), fallback_font: None, decimal_separator: '.', thousands_separator: ',', date_format: "YYYY/MM/DD".to_string(), time_format: "HH:mm".to_string(), currency_symbol: "Y".to_string(), currency_position: CurrencyPosition::Before, plural_rules: PluralRuleSet::default() } }
    pub fn is_rtl(&self) -> bool { self.text_direction == TextDirectionV2::RightToLeft }
}

#[derive(Debug, Clone)]
pub struct StringEntry { pub key: String, pub value: String, pub context: Option<String>, pub max_length: Option<usize>, pub validated: bool }
impl StringEntry { pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self { Self { key: key.into(), value: value.into(), context: None, max_length: None, validated: false } } }

#[derive(Debug, Clone)]
pub struct StringTable { pub locale: String, pub entries: HashMap<String, StringEntry>, pub loaded: bool, pub source_path: Option<String> }
impl StringTable {
    pub fn new(locale: impl Into<String>) -> Self { Self { locale: locale.into(), entries: HashMap::new(), loaded: false, source_path: None } }
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) { let k = key.into(); self.entries.insert(k.clone(), StringEntry::new(k, value)); }
    pub fn get(&self, key: &str) -> Option<&str> { self.entries.get(key).map(|e| e.value.as_str()) }
    pub fn get_formatted(&self, key: &str, args: &[(&str, &str)]) -> Option<String> {
        let template = self.get(key)?;
        let mut result = template.to_string();
        for (name, value) in args { result = result.replace(&format!("{{{}}}", name), value); }
        Some(result)
    }
    pub fn entry_count(&self) -> usize { self.entries.len() }
}

#[derive(Debug, Clone)]
pub enum ValidationIssue { MissingKey(String), ExceedsMaxLength(String, usize, usize), PlaceholderMismatch(String, Vec<String>, Vec<String>), EmptyValue(String) }
impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingKey(k) => write!(f, "Missing key: {}", k),
            Self::ExceedsMaxLength(k, actual, max) => write!(f, "{}: length {} exceeds max {}", k, actual, max),
            Self::PlaceholderMismatch(k, expected, found) => write!(f, "{}: expected placeholders {:?}, found {:?}", k, expected, found),
            Self::EmptyValue(k) => write!(f, "{}: empty value", k),
        }
    }
}

pub fn format_number(value: f64, locale: &LocaleInfoV2) -> String {
    let is_negative = value < 0.0;
    let abs = value.abs();
    let integer = abs as u64;
    let frac = ((abs - integer as f64) * 100.0).round() as u64;
    let int_str = integer.to_string();
    let mut formatted = String::new();
    for (i, c) in int_str.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { formatted.push(locale.thousands_separator); }
        formatted.push(c);
    }
    let formatted: String = formatted.chars().rev().collect();
    let result = if frac > 0 { format!("{}{}{:02}", formatted, locale.decimal_separator, frac) } else { formatted };
    if is_negative { format!("-{}", result) } else { result }
}

pub fn format_currency(value: f64, locale: &LocaleInfoV2) -> String {
    let num = format_number(value, locale);
    match locale.currency_position {
        CurrencyPosition::Before => format!("{}{}", locale.currency_symbol, num),
        CurrencyPosition::After => format!("{} {}", num, locale.currency_symbol),
    }
}

pub fn validate_table(reference: &StringTable, target: &StringTable) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    for (key, ref_entry) in &reference.entries {
        match target.entries.get(key) {
            None => issues.push(ValidationIssue::MissingKey(key.clone())),
            Some(entry) => {
                if entry.value.is_empty() { issues.push(ValidationIssue::EmptyValue(key.clone())); }
                if let Some(max) = ref_entry.max_length { if entry.value.len() > max { issues.push(ValidationIssue::ExceedsMaxLength(key.clone(), entry.value.len(), max)); } }
            }
        }
    }
    issues
}

#[derive(Debug, Clone)]
pub enum LocaleEvent { LocaleChanged(String), TableLoaded(String), TableUnloaded(String), ValidationCompleted(usize) }

pub struct LocaleManagerV2 {
    pub current_locale: String, pub locales: HashMap<String, LocaleInfoV2>,
    pub tables: HashMap<String, StringTable>, pub fallback_locale: String,
    pub events: Vec<LocaleEvent>, pub auto_load: bool,
}

impl LocaleManagerV2 {
    pub fn new(default_locale: impl Into<String>) -> Self {
        let locale = default_locale.into();
        let mut locales = HashMap::new();
        locales.insert("en-US".to_string(), LocaleInfoV2::english());
        locales.insert("ar-SA".to_string(), LocaleInfoV2::arabic());
        locales.insert("ja-JP".to_string(), LocaleInfoV2::japanese());
        Self { current_locale: locale.clone(), locales, tables: HashMap::new(), fallback_locale: "en-US".to_string(), events: Vec::new(), auto_load: true }
    }

    pub fn set_locale(&mut self, locale: impl Into<String>) {
        let l = locale.into();
        self.current_locale = l.clone();
        self.events.push(LocaleEvent::LocaleChanged(l));
    }

    pub fn current_info(&self) -> Option<&LocaleInfoV2> { self.locales.get(&self.current_locale) }
    pub fn is_rtl(&self) -> bool { self.current_info().map(|i| i.is_rtl()).unwrap_or(false) }
    pub fn current_font(&self) -> &str { self.current_info().map(|i| i.font_family.as_str()).unwrap_or("default") }

    pub fn load_table(&mut self, table: StringTable) {
        let locale = table.locale.clone();
        self.tables.insert(locale.clone(), table);
        self.events.push(LocaleEvent::TableLoaded(locale));
    }

    pub fn get_string(&self, key: &str) -> &str {
        if let Some(table) = self.tables.get(&self.current_locale) { if let Some(v) = table.get(key) { return v; } }
        if let Some(table) = self.tables.get(&self.fallback_locale) { if let Some(v) = table.get(key) { return v; } }
        key
    }

    pub fn get_formatted(&self, key: &str, args: &[(&str, &str)]) -> String {
        if let Some(table) = self.tables.get(&self.current_locale) { if let Some(v) = table.get_formatted(key, args) { return v; } }
        if let Some(table) = self.tables.get(&self.fallback_locale) { if let Some(v) = table.get_formatted(key, args) { return v; } }
        key.to_string()
    }

    pub fn format_number(&self, value: f64) -> String { let info = self.current_info().cloned().unwrap_or_else(LocaleInfoV2::english); format_number(value, &info) }
    pub fn format_currency(&self, value: f64) -> String { let info = self.current_info().cloned().unwrap_or_else(LocaleInfoV2::english); format_currency(value, &info) }
    pub fn locale_count(&self) -> usize { self.locales.len() }
    pub fn drain_events(&mut self) -> Vec<LocaleEvent> { std::mem::take(&mut self.events) }
}
impl Default for LocaleManagerV2 { fn default() -> Self { Self::new("en-US") } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn string_lookup() {
        let mut mgr = LocaleManagerV2::new("en-US");
        let mut table = StringTable::new("en-US");
        table.insert("greeting", "Hello {name}!");
        mgr.load_table(table);
        assert_eq!(mgr.get_string("greeting"), "Hello {name}!");
        assert_eq!(mgr.get_formatted("greeting", &[("name", "World")]), "Hello World!");
    }
    #[test]
    fn number_formatting() {
        let info = LocaleInfoV2::english();
        assert_eq!(format_number(1234567.89, &info), "1,234,567.89");
    }
}
