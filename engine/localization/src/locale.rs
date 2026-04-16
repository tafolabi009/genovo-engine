//! Locale management and identification.
//!
//! Provides [`Locale`] (language/region/script), [`LocaleId`] (well-known
//! locale identifiers), and [`LocaleManager`] (runtime locale switching with
//! fallback chains).

use std::fmt;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by locale operations.
#[derive(Debug, thiserror::Error)]
pub enum LocaleError {
    #[error("Unknown locale identifier: {0}")]
    UnknownLocale(String),
    #[error("Invalid BCP 47 tag: {0}")]
    InvalidTag(String),
    #[error("No fallback available for locale: {0}")]
    NoFallback(String),
}

// ---------------------------------------------------------------------------
// TextDirection
// ---------------------------------------------------------------------------

/// Text direction for a locale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextDirection {
    /// Left-to-right (English, French, German, etc.)
    LeftToRight,
    /// Right-to-left (Arabic, Hebrew, etc.)
    RightToLeft,
}

impl Default for TextDirection {
    fn default() -> Self {
        Self::LeftToRight
    }
}

// ---------------------------------------------------------------------------
// Locale
// ---------------------------------------------------------------------------

/// A locale definition following BCP 47 conventions.
///
/// # Fields
/// - `language` -- ISO 639-1 two-letter language code (e.g., "en", "fr").
/// - `region` -- ISO 3166-1 two-letter region code (e.g., "US", "FR").
/// - `script` -- ISO 15924 four-letter script code (e.g., "Latn", "Hans").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Locale {
    /// ISO 639-1 language code.
    pub language: String,
    /// ISO 3166-1 region code (optional).
    pub region: Option<String>,
    /// ISO 15924 script code (optional).
    pub script: Option<String>,
}

impl Locale {
    /// Create a locale with language only.
    pub fn from_language(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            region: None,
            script: None,
        }
    }

    /// Create a locale with language and region.
    pub fn from_language_region(language: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            region: Some(region.into()),
            script: None,
        }
    }

    /// Create a locale with language, region, and script.
    pub fn new(
        language: impl Into<String>,
        region: Option<String>,
        script: Option<String>,
    ) -> Self {
        Self {
            language: language.into(),
            region,
            script,
        }
    }

    /// Parse a BCP 47 tag (e.g., "en-US", "zh-Hans-CN").
    pub fn from_bcp47(tag: &str) -> Result<Self, LocaleError> {
        let parts: Vec<&str> = tag.split('-').collect();
        if parts.is_empty() {
            return Err(LocaleError::InvalidTag(tag.to_string()));
        }

        let language = parts[0].to_lowercase();
        if language.len() < 2 || language.len() > 3 {
            return Err(LocaleError::InvalidTag(format!(
                "Invalid language code: {language}"
            )));
        }

        let mut region = None;
        let mut script = None;

        for &part in &parts[1..] {
            if part.len() == 4 && part.chars().next().map_or(false, |c| c.is_uppercase()) {
                // Script code (4 letters, title case)
                script = Some(part.to_string());
            } else if part.len() == 2 && part.chars().all(|c| c.is_uppercase()) {
                // Region code (2 uppercase letters)
                region = Some(part.to_string());
            } else if part.len() == 2 {
                // Might be a region in lowercase
                region = Some(part.to_uppercase());
            }
        }

        Ok(Self {
            language,
            region,
            script,
        })
    }

    /// Produce the BCP 47 tag string.
    pub fn to_bcp47(&self) -> String {
        let mut tag = self.language.clone();
        if let Some(ref script) = self.script {
            tag.push('-');
            tag.push_str(script);
        }
        if let Some(ref region) = self.region {
            tag.push('-');
            tag.push_str(region);
        }
        tag
    }

    /// Determine the text direction for this locale.
    pub fn text_direction(&self) -> TextDirection {
        match self.language.as_str() {
            "ar" | "he" | "fa" | "ur" | "yi" | "ps" | "sd" | "ku" => TextDirection::RightToLeft,
            _ => TextDirection::LeftToRight,
        }
    }

    /// Whether this locale is right-to-left.
    pub fn is_rtl(&self) -> bool {
        self.text_direction() == TextDirection::RightToLeft
    }

    /// Check if this locale matches another (language must match; region and
    /// script match if both are present).
    pub fn matches(&self, other: &Locale) -> bool {
        if self.language != other.language {
            return false;
        }
        if let (Some(a), Some(b)) = (&self.region, &other.region) {
            if a != b {
                return false;
            }
        }
        if let (Some(a), Some(b)) = (&self.script, &other.script) {
            if a != b {
                return false;
            }
        }
        true
    }
}

impl fmt::Display for Locale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_bcp47())
    }
}

// ---------------------------------------------------------------------------
// LocaleId
// ---------------------------------------------------------------------------

/// Well-known locale identifiers.
///
/// Provides a typed enum for the most common locales used in games, plus
/// a `Custom` variant for anything not listed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LocaleId {
    EnUS,
    EnGB,
    FrFR,
    FrCA,
    DeDE,
    JaJP,
    ZhCN,
    ZhTW,
    KoKR,
    EsES,
    EsMX,
    PtBR,
    PtPT,
    RuRU,
    ItIT,
    ArSA,
    ThTH,
    HiIN,
    PlPL,
    NlNL,
    SvSE,
    TrTR,
    /// Any locale not in the predefined list.
    Custom(String),
}

impl LocaleId {
    /// Convert to a [`Locale`].
    pub fn to_locale(&self) -> Locale {
        match self {
            Self::EnUS => Locale::from_language_region("en", "US"),
            Self::EnGB => Locale::from_language_region("en", "GB"),
            Self::FrFR => Locale::from_language_region("fr", "FR"),
            Self::FrCA => Locale::from_language_region("fr", "CA"),
            Self::DeDE => Locale::from_language_region("de", "DE"),
            Self::JaJP => Locale::from_language_region("ja", "JP"),
            Self::ZhCN => Locale::new("zh", Some("CN".into()), Some("Hans".into())),
            Self::ZhTW => Locale::new("zh", Some("TW".into()), Some("Hant".into())),
            Self::KoKR => Locale::from_language_region("ko", "KR"),
            Self::EsES => Locale::from_language_region("es", "ES"),
            Self::EsMX => Locale::from_language_region("es", "MX"),
            Self::PtBR => Locale::from_language_region("pt", "BR"),
            Self::PtPT => Locale::from_language_region("pt", "PT"),
            Self::RuRU => Locale::from_language_region("ru", "RU"),
            Self::ItIT => Locale::from_language_region("it", "IT"),
            Self::ArSA => Locale::from_language_region("ar", "SA"),
            Self::ThTH => Locale::from_language_region("th", "TH"),
            Self::HiIN => Locale::from_language_region("hi", "IN"),
            Self::PlPL => Locale::from_language_region("pl", "PL"),
            Self::NlNL => Locale::from_language_region("nl", "NL"),
            Self::SvSE => Locale::from_language_region("sv", "SE"),
            Self::TrTR => Locale::from_language_region("tr", "TR"),
            Self::Custom(tag) => {
                Locale::from_bcp47(tag).unwrap_or_else(|_| Locale::from_language("en"))
            }
        }
    }

    /// Get the BCP 47 tag string.
    pub fn as_str(&self) -> String {
        self.to_locale().to_bcp47()
    }

    /// Parse from a BCP 47 string.
    pub fn from_str(s: &str) -> Self {
        match s {
            "en-US" => Self::EnUS,
            "en-GB" => Self::EnGB,
            "fr-FR" => Self::FrFR,
            "fr-CA" => Self::FrCA,
            "de-DE" => Self::DeDE,
            "ja-JP" => Self::JaJP,
            "zh-CN" | "zh-Hans-CN" => Self::ZhCN,
            "zh-TW" | "zh-Hant-TW" => Self::ZhTW,
            "ko-KR" => Self::KoKR,
            "es-ES" => Self::EsES,
            "es-MX" => Self::EsMX,
            "pt-BR" => Self::PtBR,
            "pt-PT" => Self::PtPT,
            "ru-RU" => Self::RuRU,
            "it-IT" => Self::ItIT,
            "ar-SA" => Self::ArSA,
            "th-TH" => Self::ThTH,
            "hi-IN" => Self::HiIN,
            "pl-PL" => Self::PlPL,
            "nl-NL" => Self::NlNL,
            "sv-SE" => Self::SvSE,
            "tr-TR" => Self::TrTR,
            other => Self::Custom(other.to_string()),
        }
    }

    /// Default fallback chain for this locale.
    ///
    /// For example, `fr-CA` falls back to `fr-FR`, then `en-US`.
    pub fn fallback_chain(&self) -> Vec<LocaleId> {
        match self {
            Self::EnUS => vec![],
            Self::EnGB => vec![Self::EnUS],
            Self::FrFR => vec![Self::EnUS],
            Self::FrCA => vec![Self::FrFR, Self::EnUS],
            Self::DeDE => vec![Self::EnUS],
            Self::JaJP => vec![Self::EnUS],
            Self::ZhCN => vec![Self::EnUS],
            Self::ZhTW => vec![Self::ZhCN, Self::EnUS],
            Self::KoKR => vec![Self::EnUS],
            Self::EsES => vec![Self::EnUS],
            Self::EsMX => vec![Self::EsES, Self::EnUS],
            Self::PtBR => vec![Self::PtPT, Self::EnUS],
            Self::PtPT => vec![Self::EnUS],
            Self::RuRU => vec![Self::EnUS],
            Self::ItIT => vec![Self::EnUS],
            Self::ArSA => vec![Self::EnUS],
            Self::ThTH => vec![Self::EnUS],
            Self::HiIN => vec![Self::EnUS],
            Self::PlPL => vec![Self::EnUS],
            Self::NlNL => vec![Self::EnUS],
            Self::SvSE => vec![Self::EnUS],
            Self::TrTR => vec![Self::EnUS],
            Self::Custom(_) => vec![Self::EnUS],
        }
    }
}

impl fmt::Display for LocaleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Default for LocaleId {
    fn default() -> Self {
        Self::EnUS
    }
}

// ---------------------------------------------------------------------------
// LocaleManager
// ---------------------------------------------------------------------------

/// Runtime locale manager.
///
/// Manages the current locale, provides fallback resolution, and detects
/// system locale settings. Thread-safe via `parking_lot::RwLock`.
pub struct LocaleManager {
    /// Current active locale.
    current: RwLock<LocaleId>,
    /// Custom fallback chain override (if set, replaces the default).
    custom_fallback: RwLock<Option<Vec<LocaleId>>>,
    /// Available locales (those that have string tables loaded).
    available: RwLock<Vec<LocaleId>>,
    /// Callbacks to invoke when the locale changes.
    on_change_callbacks: RwLock<Vec<Box<dyn Fn(&LocaleId) + Send + Sync>>>,
}

impl LocaleManager {
    /// Create a new locale manager with the given initial locale.
    pub fn new(initial: LocaleId) -> Self {
        Self {
            current: RwLock::new(initial),
            custom_fallback: RwLock::new(None),
            available: RwLock::new(Vec::new()),
            on_change_callbacks: RwLock::new(Vec::new()),
        }
    }

    /// Create a locale manager that auto-detects the system locale.
    pub fn with_system_locale() -> Self {
        let detected = Self::detect_system_locale();
        Self::new(detected)
    }

    /// Get the current locale.
    pub fn get_locale(&self) -> LocaleId {
        self.current.read().clone()
    }

    /// Get the current locale as a [`Locale`].
    pub fn get_locale_info(&self) -> Locale {
        self.current.read().to_locale()
    }

    /// Set the active locale.
    ///
    /// Invokes all registered change callbacks after switching.
    pub fn set_locale(&self, locale: LocaleId) {
        {
            let mut current = self.current.write();
            *current = locale.clone();
        }
        log::info!("Locale changed to: {}", locale);
        // Fire callbacks.
        let callbacks = self.on_change_callbacks.read();
        for cb in callbacks.iter() {
            cb(&locale);
        }
    }

    /// Register a callback that fires when the locale changes.
    pub fn on_change(&self, callback: impl Fn(&LocaleId) + Send + Sync + 'static) {
        self.on_change_callbacks.write().push(Box::new(callback));
    }

    /// Get the effective fallback chain for the current locale.
    ///
    /// Returns the custom chain if set, otherwise the locale's default chain.
    pub fn fallback_chain(&self) -> Vec<LocaleId> {
        if let Some(ref chain) = *self.custom_fallback.read() {
            return chain.clone();
        }
        self.current.read().fallback_chain()
    }

    /// Set a custom fallback chain (overrides the locale's default).
    pub fn set_fallback_chain(&self, chain: Vec<LocaleId>) {
        *self.custom_fallback.write() = Some(chain);
    }

    /// Clear the custom fallback chain (reverts to locale default).
    pub fn clear_custom_fallback(&self) {
        *self.custom_fallback.write() = None;
    }

    /// Register an available locale.
    pub fn add_available(&self, locale: LocaleId) {
        let mut avail = self.available.write();
        if !avail.contains(&locale) {
            avail.push(locale);
        }
    }

    /// Get all available locales.
    pub fn available_locales(&self) -> Vec<LocaleId> {
        self.available.read().clone()
    }

    /// Whether the current locale is RTL.
    pub fn is_rtl(&self) -> bool {
        self.get_locale_info().is_rtl()
    }

    /// Text direction of the current locale.
    pub fn text_direction(&self) -> TextDirection {
        self.get_locale_info().text_direction()
    }

    /// Detect the system locale.
    ///
    /// On most platforms this reads environment variables (LANG, LC_ALL)
    /// or platform-specific APIs. This simplified implementation uses
    /// environment variables.
    pub fn detect_system_locale() -> LocaleId {
        // Try common environment variables.
        for var in &["LANG", "LC_ALL", "LC_MESSAGES", "LANGUAGE"] {
            if let Ok(val) = std::env::var(var) {
                // Parse "en_US.UTF-8" or "en_US" format.
                let cleaned = val
                    .split('.')
                    .next()
                    .unwrap_or("")
                    .replace('_', "-");
                if !cleaned.is_empty() {
                    return LocaleId::from_str(&cleaned);
                }
            }
        }
        // Default to en-US.
        LocaleId::EnUS
    }
}

impl fmt::Debug for LocaleManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocaleManager")
            .field("current", &*self.current.read())
            .field("available", &*self.available.read())
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
    fn locale_bcp47_parse() {
        let locale = Locale::from_bcp47("en-US").unwrap();
        assert_eq!(locale.language, "en");
        assert_eq!(locale.region.as_deref(), Some("US"));
        assert!(locale.script.is_none());
    }

    #[test]
    fn locale_bcp47_with_script() {
        let locale = Locale::from_bcp47("zh-Hans-CN").unwrap();
        assert_eq!(locale.language, "zh");
        assert_eq!(locale.script.as_deref(), Some("Hans"));
        assert_eq!(locale.region.as_deref(), Some("CN"));
    }

    #[test]
    fn locale_bcp47_roundtrip() {
        let locale = Locale::from_language_region("fr", "FR");
        assert_eq!(locale.to_bcp47(), "fr-FR");
    }

    #[test]
    fn locale_rtl_detection() {
        assert!(Locale::from_language("ar").is_rtl());
        assert!(Locale::from_language("he").is_rtl());
        assert!(!Locale::from_language("en").is_rtl());
        assert!(!Locale::from_language("ja").is_rtl());
    }

    #[test]
    fn locale_id_conversions() {
        let id = LocaleId::JaJP;
        let locale = id.to_locale();
        assert_eq!(locale.language, "ja");
        assert_eq!(locale.region.as_deref(), Some("JP"));
    }

    #[test]
    fn locale_id_fallback_chain() {
        let chain = LocaleId::FrCA.fallback_chain();
        assert_eq!(chain, vec![LocaleId::FrFR, LocaleId::EnUS]);
    }

    #[test]
    fn locale_id_from_str() {
        assert_eq!(LocaleId::from_str("en-US"), LocaleId::EnUS);
        assert_eq!(LocaleId::from_str("ja-JP"), LocaleId::JaJP);
        assert!(matches!(LocaleId::from_str("xx-YY"), LocaleId::Custom(_)));
    }

    #[test]
    fn locale_manager_switch() {
        let manager = LocaleManager::new(LocaleId::EnUS);
        assert_eq!(manager.get_locale(), LocaleId::EnUS);

        manager.set_locale(LocaleId::JaJP);
        assert_eq!(manager.get_locale(), LocaleId::JaJP);
    }

    #[test]
    fn locale_manager_fallback() {
        let manager = LocaleManager::new(LocaleId::FrCA);
        let chain = manager.fallback_chain();
        assert_eq!(chain, vec![LocaleId::FrFR, LocaleId::EnUS]);

        // Override with custom chain.
        manager.set_fallback_chain(vec![LocaleId::EnGB, LocaleId::EnUS]);
        let chain = manager.fallback_chain();
        assert_eq!(chain, vec![LocaleId::EnGB, LocaleId::EnUS]);
    }

    #[test]
    fn locale_matches() {
        let en_us = Locale::from_language_region("en", "US");
        let en_gb = Locale::from_language_region("en", "GB");
        let en = Locale::from_language("en");

        assert!(!en_us.matches(&en_gb)); // Different region
        assert!(en.matches(&en_us)); // en matches en-US (region only in other)
        assert!(en_us.matches(&en)); // en-US matches en (region only in self)
    }
}
