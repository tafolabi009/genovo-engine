//! Translation Quality Analysis
//!
//! Provides tools for analyzing translation quality, coverage, and
//! consistency across locales. Features include:
//!
//! - Find untranslated strings
//! - Find unused translation keys
//! - Translation coverage report per locale
//! - Context annotations for translators
//! - Screenshot reference system
//! - Translation memory (reuse similar translations)
//!
//! # Usage
//!
//! ```ignore
//! let analyzer = TextAnalyzer::new();
//! let report = analyzer.coverage_report(&string_tables, &["en_US", "fr_FR", "de_DE"]);
//! println!("{}", report);
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// TranslationKey
// ---------------------------------------------------------------------------

/// A localization key with metadata for translators.
#[derive(Debug, Clone)]
pub struct TranslationKey {
    /// The string key (e.g., "ui.menu.start_game").
    pub key: String,
    /// The source (reference) text, typically English.
    pub source_text: String,
    /// Context annotation to help translators understand usage.
    pub context: Option<String>,
    /// Maximum character length for the translated string (if constrained).
    pub max_length: Option<usize>,
    /// Screenshot reference showing where this string appears in the UI.
    pub screenshot_ref: Option<ScreenshotReference>,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Whether this key is marked as "do not translate" (e.g., proper nouns).
    pub do_not_translate: bool,
    /// Priority/importance level.
    pub priority: TranslationPriority,
    /// The module or system that uses this key.
    pub module: Option<String>,
    /// Comments from developers.
    pub developer_notes: Vec<String>,
    /// Whether this key has been deprecated.
    pub deprecated: bool,
}

/// Priority level for a translation key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TranslationPriority {
    /// Critical: visible in main gameplay or menus.
    Critical,
    /// High: important UI text.
    High,
    /// Medium: secondary text.
    Medium,
    /// Low: rarely seen text, debug messages.
    Low,
}

impl Default for TranslationPriority {
    fn default() -> Self {
        Self::Medium
    }
}

impl fmt::Display for TranslationPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::High => write!(f, "HIGH"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::Low => write!(f, "LOW"),
        }
    }
}

impl TranslationKey {
    /// Creates a new translation key.
    pub fn new(key: impl Into<String>, source_text: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            source_text: source_text.into(),
            context: None,
            max_length: None,
            screenshot_ref: None,
            tags: Vec::new(),
            do_not_translate: false,
            priority: TranslationPriority::Medium,
            module: None,
            developer_notes: Vec::new(),
            deprecated: false,
        }
    }

    /// Sets the context annotation.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Sets the maximum character length.
    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = Some(max);
        self
    }

    /// Adds a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Sets the module.
    pub fn with_module(mut self, module: impl Into<String>) -> Self {
        self.module = Some(module.into());
        self
    }

    /// Sets the priority.
    pub fn with_priority(mut self, priority: TranslationPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Adds a developer note.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.developer_notes.push(note.into());
        self
    }

    /// Marks as "do not translate".
    pub fn as_do_not_translate(mut self) -> Self {
        self.do_not_translate = true;
        self
    }

    /// Sets a screenshot reference.
    pub fn with_screenshot(mut self, screenshot: ScreenshotReference) -> Self {
        self.screenshot_ref = Some(screenshot);
        self
    }
}

// ---------------------------------------------------------------------------
// ScreenshotReference
// ---------------------------------------------------------------------------

/// A screenshot showing where a translation string appears in the UI.
#[derive(Debug, Clone)]
pub struct ScreenshotReference {
    /// Path to the screenshot image file.
    pub image_path: String,
    /// Description of the context in the screenshot.
    pub description: String,
    /// Bounding box of the text in the screenshot (x, y, width, height).
    pub highlight_rect: Option<[f32; 4]>,
    /// The game state or screen where this appears.
    pub screen_context: String,
    /// Timestamp when the screenshot was taken.
    pub timestamp: String,
    /// Resolution of the screenshot.
    pub resolution: (u32, u32),
}

impl ScreenshotReference {
    /// Creates a new screenshot reference.
    pub fn new(image_path: impl Into<String>, screen_context: impl Into<String>) -> Self {
        Self {
            image_path: image_path.into(),
            description: String::new(),
            highlight_rect: None,
            screen_context: screen_context.into(),
            timestamp: String::new(),
            resolution: (1920, 1080),
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Sets the highlight rectangle.
    pub fn with_highlight(mut self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.highlight_rect = Some([x, y, w, h]);
        self
    }
}

// ---------------------------------------------------------------------------
// TranslationEntry
// ---------------------------------------------------------------------------

/// A translation of a key in a specific locale.
#[derive(Debug, Clone)]
pub struct TranslationEntry {
    /// The locale code (e.g., "fr_FR").
    pub locale: String,
    /// The translated text.
    pub text: String,
    /// Whether this translation has been reviewed/approved.
    pub reviewed: bool,
    /// Who translated this entry.
    pub translator: Option<String>,
    /// When this was last updated.
    pub updated_at: Option<String>,
    /// Whether this translation was generated from translation memory.
    pub from_memory: bool,
    /// Confidence score if generated from memory (0.0 to 1.0).
    pub memory_confidence: f32,
    /// Whether this translation needs re-review (source text changed).
    pub needs_review: bool,
}

impl TranslationEntry {
    /// Creates a new translation entry.
    pub fn new(locale: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            locale: locale.into(),
            text: text.into(),
            reviewed: false,
            translator: None,
            updated_at: None,
            from_memory: false,
            memory_confidence: 0.0,
            needs_review: false,
        }
    }

    /// Marks as reviewed.
    pub fn with_reviewed(mut self) -> Self {
        self.reviewed = true;
        self
    }

    /// Sets the translator.
    pub fn with_translator(mut self, translator: impl Into<String>) -> Self {
        self.translator = Some(translator.into());
        self
    }
}

// ---------------------------------------------------------------------------
// TranslationDatabase
// ---------------------------------------------------------------------------

/// In-memory database of all translation keys and their translations.
pub struct TranslationDatabase {
    /// All translation keys.
    keys: HashMap<String, TranslationKey>,
    /// Translations indexed by (key, locale).
    translations: HashMap<(String, String), TranslationEntry>,
    /// All known locales.
    locales: Vec<String>,
    /// The source locale (reference language).
    source_locale: String,
    /// Keys actually referenced in source code.
    used_keys: HashSet<String>,
}

impl TranslationDatabase {
    /// Creates a new translation database.
    pub fn new(source_locale: impl Into<String>) -> Self {
        let source = source_locale.into();
        Self {
            keys: HashMap::new(),
            translations: HashMap::new(),
            locales: vec![source.clone()],
            source_locale: source,
            used_keys: HashSet::new(),
        }
    }

    /// Adds a locale to the database.
    pub fn add_locale(&mut self, locale: impl Into<String>) {
        let locale_str = locale.into();
        if !self.locales.contains(&locale_str) {
            self.locales.push(locale_str);
        }
    }

    /// Registers a translation key.
    pub fn register_key(&mut self, key: TranslationKey) {
        self.keys.insert(key.key.clone(), key);
    }

    /// Adds a translation for a key in a specific locale.
    pub fn add_translation(&mut self, key: &str, entry: TranslationEntry) {
        let locale = entry.locale.clone();
        self.add_locale(&locale);
        self.translations
            .insert((key.to_string(), locale), entry);
    }

    /// Marks a key as used in source code.
    pub fn mark_used(&mut self, key: &str) {
        self.used_keys.insert(key.to_string());
    }

    /// Marks multiple keys as used.
    pub fn mark_used_batch(&mut self, keys: &[&str]) {
        for key in keys {
            self.used_keys.insert(key.to_string());
        }
    }

    /// Returns a reference to a translation key.
    pub fn get_key(&self, key: &str) -> Option<&TranslationKey> {
        self.keys.get(key)
    }

    /// Returns a translation for a key in a specific locale.
    pub fn get_translation(&self, key: &str, locale: &str) -> Option<&TranslationEntry> {
        self.translations.get(&(key.to_string(), locale.to_string()))
    }

    /// Returns all registered key names.
    pub fn all_keys(&self) -> Vec<&str> {
        self.keys.keys().map(|k| k.as_str()).collect()
    }

    /// Returns all registered locales.
    pub fn all_locales(&self) -> &[String] {
        &self.locales
    }

    /// Returns the number of registered keys.
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Returns the number of translations.
    pub fn translation_count(&self) -> usize {
        self.translations.len()
    }
}

impl fmt::Debug for TranslationDatabase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TranslationDatabase")
            .field("key_count", &self.keys.len())
            .field("translation_count", &self.translations.len())
            .field("locale_count", &self.locales.len())
            .field("source_locale", &self.source_locale)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// TranslationMemory
// ---------------------------------------------------------------------------

/// Translation memory for reusing similar translations.
///
/// When a new string needs translation, the memory is searched for similar
/// previously translated strings. If a close match is found, the existing
/// translation is suggested as a starting point.
pub struct TranslationMemory {
    /// Stored translation pairs indexed by source text.
    entries: HashMap<String, Vec<TranslationMemoryEntry>>,
    /// Minimum similarity threshold for suggestions (0.0 to 1.0).
    pub similarity_threshold: f32,
    /// Maximum number of suggestions to return.
    pub max_suggestions: usize,
}

/// A single translation memory entry.
#[derive(Debug, Clone)]
pub struct TranslationMemoryEntry {
    /// The source (reference language) text.
    pub source_text: String,
    /// The translated text.
    pub translated_text: String,
    /// The target locale.
    pub locale: String,
    /// Number of times this translation has been used.
    pub usage_count: u32,
    /// Last used timestamp.
    pub last_used: Option<String>,
    /// Quality rating (0.0 to 1.0).
    pub quality: f32,
}

impl TranslationMemory {
    /// Creates a new translation memory.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            similarity_threshold: 0.7,
            max_suggestions: 5,
        }
    }

    /// Adds a translation pair to the memory.
    pub fn add(
        &mut self,
        source_text: impl Into<String>,
        translated_text: impl Into<String>,
        locale: impl Into<String>,
    ) {
        let source = source_text.into();
        let entry = TranslationMemoryEntry {
            source_text: source.clone(),
            translated_text: translated_text.into(),
            locale: locale.into(),
            usage_count: 1,
            last_used: None,
            quality: 1.0,
        };
        self.entries.entry(source).or_default().push(entry);
    }

    /// Searches for similar translations.
    pub fn find_similar(
        &self,
        source_text: &str,
        locale: &str,
    ) -> Vec<TranslationSuggestion> {
        let mut suggestions = Vec::new();

        for (stored_source, entries) in &self.entries {
            let similarity = self.calculate_similarity(source_text, stored_source);
            if similarity >= self.similarity_threshold {
                for entry in entries {
                    if entry.locale == locale {
                        suggestions.push(TranslationSuggestion {
                            source_text: entry.source_text.clone(),
                            suggested_text: entry.translated_text.clone(),
                            similarity,
                            quality: entry.quality,
                            usage_count: entry.usage_count,
                        });
                    }
                }
            }
        }

        suggestions.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.truncate(self.max_suggestions);
        suggestions
    }

    /// Returns the total number of entries in memory.
    pub fn entry_count(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Calculates the similarity between two strings using a simple
    /// token-based Jaccard similarity.
    fn calculate_similarity(&self, a: &str, b: &str) -> f32 {
        let tokens_a: HashSet<&str> = a.split_whitespace().collect();
        let tokens_b: HashSet<&str> = b.split_whitespace().collect();

        if tokens_a.is_empty() && tokens_b.is_empty() {
            return 1.0;
        }

        let intersection = tokens_a.intersection(&tokens_b).count();
        let union = tokens_a.union(&tokens_b).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }
}

impl Default for TranslationMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// A translation suggestion from translation memory.
#[derive(Debug, Clone)]
pub struct TranslationSuggestion {
    /// The source text that matched.
    pub source_text: String,
    /// The suggested translation.
    pub suggested_text: String,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f32,
    /// Quality rating of the source translation.
    pub quality: f32,
    /// Number of times the source translation has been used.
    pub usage_count: u32,
}

// ---------------------------------------------------------------------------
// TextAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes translation quality, coverage, and consistency.
pub struct TextAnalyzer {
    /// Translation memory for similarity suggestions.
    pub memory: TranslationMemory,
    /// Patterns to ignore during analysis (e.g., format specifiers).
    pub ignore_patterns: Vec<String>,
    /// Known locale codes for validation.
    pub known_locales: Vec<String>,
}

impl TextAnalyzer {
    /// Creates a new text analyzer.
    pub fn new() -> Self {
        Self {
            memory: TranslationMemory::new(),
            ignore_patterns: Vec::new(),
            known_locales: vec![
                "en_US".to_string(),
                "en_GB".to_string(),
                "fr_FR".to_string(),
                "de_DE".to_string(),
                "es_ES".to_string(),
                "it_IT".to_string(),
                "pt_BR".to_string(),
                "ja_JP".to_string(),
                "ko_KR".to_string(),
                "zh_CN".to_string(),
                "zh_TW".to_string(),
                "ru_RU".to_string(),
                "pl_PL".to_string(),
                "ar_SA".to_string(),
                "tr_TR".to_string(),
            ],
        }
    }

    /// Finds all untranslated strings for a given locale.
    pub fn find_untranslated(
        &self,
        db: &TranslationDatabase,
        locale: &str,
    ) -> Vec<UntranslatedEntry> {
        let mut untranslated = Vec::new();

        for (key_name, key_def) in &db.keys {
            if key_def.do_not_translate || key_def.deprecated {
                continue;
            }

            let has_translation = db.get_translation(key_name, locale).is_some();
            if !has_translation {
                untranslated.push(UntranslatedEntry {
                    key: key_name.clone(),
                    source_text: key_def.source_text.clone(),
                    priority: key_def.priority,
                    module: key_def.module.clone(),
                    has_context: key_def.context.is_some(),
                    has_screenshot: key_def.screenshot_ref.is_some(),
                });
            }
        }

        // Sort by priority (critical first).
        untranslated.sort_by(|a, b| a.priority.cmp(&b.priority));
        untranslated
    }

    /// Finds translation keys that exist in the database but are not
    /// referenced in source code.
    pub fn find_unused_keys(&self, db: &TranslationDatabase) -> Vec<UnusedKeyEntry> {
        let mut unused = Vec::new();

        for (key_name, key_def) in &db.keys {
            if key_def.deprecated {
                continue;
            }

            if !db.used_keys.contains(key_name) {
                // Count how many locales have translations for this key.
                let translation_count = db
                    .locales
                    .iter()
                    .filter(|locale| db.get_translation(key_name, locale).is_some())
                    .count();

                unused.push(UnusedKeyEntry {
                    key: key_name.clone(),
                    source_text: key_def.source_text.clone(),
                    module: key_def.module.clone(),
                    translation_count,
                    total_locales: db.locales.len(),
                });
            }
        }

        unused
    }

    /// Generates a translation coverage report for all locales.
    pub fn coverage_report(&self, db: &TranslationDatabase) -> CoverageReport {
        let total_keys = db.keys.len();
        let translatable_keys = db
            .keys
            .values()
            .filter(|k| !k.do_not_translate && !k.deprecated)
            .count();

        let mut locale_coverage = Vec::new();

        for locale in &db.locales {
            if *locale == db.source_locale {
                continue;
            }

            let mut translated = 0;
            let mut reviewed = 0;
            let mut needs_review = 0;
            let mut from_memory = 0;

            for (key_name, key_def) in &db.keys {
                if key_def.do_not_translate || key_def.deprecated {
                    continue;
                }

                if let Some(entry) = db.get_translation(key_name, locale) {
                    translated += 1;
                    if entry.reviewed {
                        reviewed += 1;
                    }
                    if entry.needs_review {
                        needs_review += 1;
                    }
                    if entry.from_memory {
                        from_memory += 1;
                    }
                }
            }

            let coverage_pct = if translatable_keys > 0 {
                (translated as f64 / translatable_keys as f64) * 100.0
            } else {
                100.0
            };

            let reviewed_pct = if translated > 0 {
                (reviewed as f64 / translated as f64) * 100.0
            } else {
                0.0
            };

            locale_coverage.push(LocaleCoverage {
                locale: locale.clone(),
                total_keys: translatable_keys,
                translated,
                reviewed,
                needs_review,
                from_memory,
                coverage_percentage: coverage_pct,
                review_percentage: reviewed_pct,
            });
        }

        // Sort by coverage (lowest first for attention).
        locale_coverage.sort_by(|a, b| {
            a.coverage_percentage
                .partial_cmp(&b.coverage_percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        CoverageReport {
            total_keys,
            translatable_keys,
            source_locale: db.source_locale.clone(),
            locale_coverage,
        }
    }

    /// Checks for potential translation issues (length violations,
    /// missing format specifiers, etc.).
    pub fn check_quality(
        &self,
        db: &TranslationDatabase,
        locale: &str,
    ) -> Vec<QualityIssue> {
        let mut issues = Vec::new();

        for (key_name, key_def) in &db.keys {
            if key_def.do_not_translate || key_def.deprecated {
                continue;
            }

            if let Some(entry) = db.get_translation(key_name, locale) {
                // Check max length.
                if let Some(max_len) = key_def.max_length {
                    if entry.text.len() > max_len {
                        issues.push(QualityIssue {
                            key: key_name.clone(),
                            locale: locale.to_string(),
                            issue_type: QualityIssueType::LengthExceeded,
                            message: format!(
                                "Translation length {} exceeds maximum {}",
                                entry.text.len(),
                                max_len
                            ),
                            severity: IssueSeverity::Warning,
                        });
                    }
                }

                // Check for missing format specifiers.
                let source_specifiers = extract_format_specifiers(&key_def.source_text);
                let translated_specifiers = extract_format_specifiers(&entry.text);
                if source_specifiers != translated_specifiers {
                    issues.push(QualityIssue {
                        key: key_name.clone(),
                        locale: locale.to_string(),
                        issue_type: QualityIssueType::MissingFormatSpecifiers,
                        message: format!(
                            "Format specifiers mismatch: source has {:?}, translation has {:?}",
                            source_specifiers, translated_specifiers
                        ),
                        severity: IssueSeverity::Error,
                    });
                }

                // Check for empty translation.
                if entry.text.trim().is_empty() {
                    issues.push(QualityIssue {
                        key: key_name.clone(),
                        locale: locale.to_string(),
                        issue_type: QualityIssueType::EmptyTranslation,
                        message: "Translation is empty".to_string(),
                        severity: IssueSeverity::Error,
                    });
                }

                // Check for untranslated (same as source).
                if entry.text == key_def.source_text
                    && locale != db.source_locale.as_str()
                {
                    issues.push(QualityIssue {
                        key: key_name.clone(),
                        locale: locale.to_string(),
                        issue_type: QualityIssueType::IdenticalToSource,
                        message: "Translation is identical to source text".to_string(),
                        severity: IssueSeverity::Info,
                    });
                }
            }
        }

        issues
    }

    /// Populates translation memory from the database.
    pub fn build_memory(&mut self, db: &TranslationDatabase) {
        for (key_name, key_def) in &db.keys {
            for locale in &db.locales {
                if *locale == db.source_locale {
                    continue;
                }
                if let Some(entry) = db.get_translation(key_name, locale) {
                    if entry.reviewed {
                        self.memory
                            .add(&key_def.source_text, &entry.text, locale);
                    }
                }
            }
        }
    }

    /// Generates translation suggestions from memory for untranslated keys.
    pub fn suggest_translations(
        &self,
        db: &TranslationDatabase,
        locale: &str,
    ) -> Vec<TranslationSuggestionReport> {
        let mut reports = Vec::new();

        let untranslated = self.find_untranslated(db, locale);
        for entry in &untranslated {
            let suggestions = self.memory.find_similar(&entry.source_text, locale);
            if !suggestions.is_empty() {
                reports.push(TranslationSuggestionReport {
                    key: entry.key.clone(),
                    source_text: entry.source_text.clone(),
                    suggestions,
                });
            }
        }

        reports
    }
}

impl Default for TextAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

/// Information about an untranslated string.
#[derive(Debug, Clone)]
pub struct UntranslatedEntry {
    /// The translation key.
    pub key: String,
    /// The source text.
    pub source_text: String,
    /// Priority level.
    pub priority: TranslationPriority,
    /// Module that uses this key.
    pub module: Option<String>,
    /// Whether context is available.
    pub has_context: bool,
    /// Whether a screenshot reference exists.
    pub has_screenshot: bool,
}

/// Information about an unused translation key.
#[derive(Debug, Clone)]
pub struct UnusedKeyEntry {
    /// The translation key.
    pub key: String,
    /// The source text.
    pub source_text: String,
    /// Module that defines this key.
    pub module: Option<String>,
    /// Number of locales that have translations for this key.
    pub translation_count: usize,
    /// Total number of locales.
    pub total_locales: usize,
}

/// Translation coverage for a single locale.
#[derive(Debug, Clone)]
pub struct LocaleCoverage {
    /// Locale code.
    pub locale: String,
    /// Total number of translatable keys.
    pub total_keys: usize,
    /// Number of translated keys.
    pub translated: usize,
    /// Number of reviewed translations.
    pub reviewed: usize,
    /// Number of translations that need re-review.
    pub needs_review: usize,
    /// Number of translations from translation memory.
    pub from_memory: usize,
    /// Coverage percentage.
    pub coverage_percentage: f64,
    /// Review percentage.
    pub review_percentage: f64,
}

impl fmt::Display for LocaleCoverage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.1}% translated ({}/{}), {:.1}% reviewed",
            self.locale,
            self.coverage_percentage,
            self.translated,
            self.total_keys,
            self.review_percentage
        )
    }
}

/// Complete coverage report across all locales.
#[derive(Debug, Clone)]
pub struct CoverageReport {
    /// Total keys in the database.
    pub total_keys: usize,
    /// Keys that need translation.
    pub translatable_keys: usize,
    /// Source locale.
    pub source_locale: String,
    /// Per-locale coverage data.
    pub locale_coverage: Vec<LocaleCoverage>,
}

impl fmt::Display for CoverageReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Translation Coverage Report (source: {})",
            self.source_locale
        )?;
        writeln!(
            f,
            "Total keys: {} ({} translatable)",
            self.total_keys, self.translatable_keys
        )?;
        writeln!(f, "---")?;
        for lc in &self.locale_coverage {
            writeln!(f, "  {}", lc)?;
        }
        Ok(())
    }
}

/// A quality issue found in a translation.
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// The translation key.
    pub key: String,
    /// The locale.
    pub locale: String,
    /// Type of issue.
    pub issue_type: QualityIssueType,
    /// Description.
    pub message: String,
    /// Severity.
    pub severity: IssueSeverity,
}

/// Types of quality issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityIssueType {
    /// Translation exceeds maximum allowed length.
    LengthExceeded,
    /// Format specifiers don't match between source and translation.
    MissingFormatSpecifiers,
    /// Translation is empty.
    EmptyTranslation,
    /// Translation is identical to the source text.
    IdenticalToSource,
    /// Trailing/leading whitespace issues.
    WhitespaceIssue,
    /// Inconsistent terminology.
    InconsistentTerminology,
}

/// Severity of a quality issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
}

/// Suggestions report for a single untranslated key.
#[derive(Debug, Clone)]
pub struct TranslationSuggestionReport {
    /// The untranslated key.
    pub key: String,
    /// The source text.
    pub source_text: String,
    /// Translation suggestions from memory.
    pub suggestions: Vec<TranslationSuggestion>,
}

/// Extracts format specifiers from a string (e.g., `{name}`, `{0}`).
fn extract_format_specifiers(text: &str) -> Vec<String> {
    let mut specifiers = Vec::new();
    let mut in_brace = false;
    let mut current = String::new();

    for ch in text.chars() {
        match ch {
            '{' if !in_brace => {
                in_brace = true;
                current.clear();
            }
            '}' if in_brace => {
                in_brace = false;
                if !current.is_empty() {
                    specifiers.push(current.clone());
                }
            }
            _ if in_brace => {
                current.push(ch);
            }
            _ => {}
        }
    }

    specifiers.sort();
    specifiers
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_db() -> TranslationDatabase {
        let mut db = TranslationDatabase::new("en_US");
        db.add_locale("fr_FR");
        db.add_locale("de_DE");

        db.register_key(TranslationKey::new("greeting", "Hello, {name}!"));
        db.register_key(TranslationKey::new("farewell", "Goodbye!"));
        db.register_key(
            TranslationKey::new("ui.start", "Start Game")
                .with_priority(TranslationPriority::Critical),
        );

        db.add_translation(
            "greeting",
            TranslationEntry::new("fr_FR", "Bonjour, {name}!").with_reviewed(),
        );
        // farewell is not translated to fr_FR
        db.add_translation(
            "ui.start",
            TranslationEntry::new("fr_FR", "Commencer"),
        );

        db.mark_used("greeting");
        db.mark_used("ui.start");
        // farewell is not marked as used

        db
    }

    #[test]
    fn test_find_untranslated() {
        let db = make_test_db();
        let analyzer = TextAnalyzer::new();
        let untranslated = analyzer.find_untranslated(&db, "fr_FR");
        assert_eq!(untranslated.len(), 1);
        assert_eq!(untranslated[0].key, "farewell");
    }

    #[test]
    fn test_find_unused_keys() {
        let db = make_test_db();
        let analyzer = TextAnalyzer::new();
        let unused = analyzer.find_unused_keys(&db);
        assert_eq!(unused.len(), 1);
        assert_eq!(unused[0].key, "farewell");
    }

    #[test]
    fn test_coverage_report() {
        let db = make_test_db();
        let analyzer = TextAnalyzer::new();
        let report = analyzer.coverage_report(&db);
        assert_eq!(report.total_keys, 3);

        let fr_coverage = report.locale_coverage.iter().find(|c| c.locale == "fr_FR");
        assert!(fr_coverage.is_some());
        let fr = fr_coverage.unwrap();
        assert_eq!(fr.translated, 2);
    }

    #[test]
    fn test_quality_check_format_specifiers() {
        let mut db = TranslationDatabase::new("en_US");
        db.add_locale("de_DE");
        db.register_key(TranslationKey::new("msg", "Hello {name}, you have {count} items"));
        db.add_translation(
            "msg",
            TranslationEntry::new("de_DE", "Hallo {name}, Sie haben Artikel"),
        );

        let analyzer = TextAnalyzer::new();
        let issues = analyzer.check_quality(&db, "de_DE");
        assert!(issues.iter().any(|i| i.issue_type == QualityIssueType::MissingFormatSpecifiers));
    }

    #[test]
    fn test_translation_memory() {
        let mut memory = TranslationMemory::new();
        memory.add("Start Game", "Jeu de demarrage", "fr_FR");
        memory.add("Start Level", "Niveau de demarrage", "fr_FR");

        let suggestions = memory.find_similar("Start Game", "fr_FR");
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].similarity > 0.5);
    }

    #[test]
    fn test_extract_format_specifiers() {
        let specs = extract_format_specifiers("Hello {name}, you have {count} items");
        assert_eq!(specs.len(), 2);
        assert!(specs.contains(&"count".to_string()));
        assert!(specs.contains(&"name".to_string()));
    }

    #[test]
    fn test_screenshot_reference() {
        let key = TranslationKey::new("ui.start", "Start Game")
            .with_screenshot(ScreenshotReference::new(
                "screenshots/main_menu.png",
                "Main Menu",
            ).with_highlight(100.0, 200.0, 200.0, 40.0));
        assert!(key.screenshot_ref.is_some());
    }
}
