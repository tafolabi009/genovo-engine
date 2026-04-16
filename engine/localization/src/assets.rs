//! Localized asset resolution.
//!
//! Provides [`LocalizedAsset`] for assets that vary by locale (voice lines,
//! textures with baked text), [`LocalizedAssetPath`] for resolving file paths,
//! and [`FontSelector`] for choosing appropriate fonts per locale.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::locale::{LocaleId, LocaleManager};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from localized asset operations.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("No asset variant found for locale {0} or any fallback")]
    NoVariant(String),
    #[error("Font not found for locale {0}")]
    FontNotFound(String),
    #[error("Asset path resolution failed: {0}")]
    PathError(String),
}

// ---------------------------------------------------------------------------
// LocalizedAsset<T>
// ---------------------------------------------------------------------------

/// An asset that has different variants per locale.
///
/// For example, a texture with baked-in text might have an English version,
/// a Japanese version, etc. The runtime selects the correct variant based
/// on the active locale and falls back through the locale chain.
///
/// `T` is typically a handle or path to the actual asset data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizedAsset<T: Clone + fmt::Debug> {
    /// Asset variants keyed by locale.
    variants: HashMap<String, T>,
    /// Optional default variant (used when no locale-specific variant exists).
    default: Option<T>,
    /// Description for tooling/editor display.
    pub description: String,
}

impl<T: Clone + fmt::Debug> LocalizedAsset<T> {
    /// Create a new localized asset with no variants.
    pub fn new() -> Self {
        Self {
            variants: HashMap::new(),
            default: None,
            description: String::new(),
        }
    }

    /// Create a localized asset with a default value.
    pub fn with_default(default: T) -> Self {
        Self {
            variants: HashMap::new(),
            default: Some(default),
            description: String::new(),
        }
    }

    /// Add a variant for a specific locale.
    pub fn add_variant(&mut self, locale: &LocaleId, asset: T) {
        self.variants.insert(locale.as_str(), asset);
    }

    /// Set the default variant.
    pub fn set_default(&mut self, asset: T) {
        self.default = Some(asset);
    }

    /// Resolve the asset for a locale with fallback.
    ///
    /// Tries the exact locale, then each fallback in the chain, then the
    /// default. Returns `None` if nothing matches.
    pub fn resolve(&self, locale: &LocaleId) -> Option<&T> {
        // Try exact match.
        if let Some(v) = self.variants.get(&locale.as_str()) {
            return Some(v);
        }

        // Try fallback chain.
        for fallback in locale.fallback_chain() {
            if let Some(v) = self.variants.get(&fallback.as_str()) {
                return Some(v);
            }
        }

        // Try default.
        self.default.as_ref()
    }

    /// Resolve the asset using a [`LocaleManager`].
    pub fn resolve_with_manager(&self, manager: &LocaleManager) -> Option<&T> {
        let locale = manager.get_locale();
        self.resolve(&locale)
    }

    /// Number of locale-specific variants.
    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }

    /// Get all locale tags that have variants.
    pub fn available_locales(&self) -> Vec<String> {
        self.variants.keys().cloned().collect()
    }

    /// Check if a variant exists for the given locale.
    pub fn has_variant(&self, locale: &LocaleId) -> bool {
        self.variants.contains_key(&locale.as_str())
    }
}

impl<T: Clone + fmt::Debug> Default for LocalizedAsset<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LocalizedAssetPath
// ---------------------------------------------------------------------------

/// Resolves asset file paths based on the active locale.
///
/// Uses a directory-based convention:
///
/// ```text
/// assets/audio/voices/en-US/greeting.ogg
/// assets/audio/voices/ja-JP/greeting.ogg
/// assets/textures/ui/en-US/main_menu.png
/// ```
///
/// The locale tag is inserted into the path between a base directory and the
/// asset filename.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizedAssetPath {
    /// Base directory (e.g., "assets/audio/voices").
    pub base_dir: PathBuf,
    /// Asset filename (e.g., "greeting.ogg").
    pub filename: String,
    /// Path pattern. Use `{locale}` as a placeholder.
    /// Example: "assets/audio/voices/{locale}/greeting.ogg"
    pub pattern: Option<String>,
}

impl LocalizedAssetPath {
    /// Create a path resolver from base directory and filename.
    pub fn new(base_dir: impl Into<PathBuf>, filename: impl Into<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
            filename: filename.into(),
            pattern: None,
        }
    }

    /// Create a path resolver from a pattern string.
    ///
    /// The pattern should contain `{locale}` which will be replaced with the
    /// BCP 47 tag.
    pub fn from_pattern(pattern: impl Into<String>) -> Self {
        Self {
            base_dir: PathBuf::new(),
            filename: String::new(),
            pattern: Some(pattern.into()),
        }
    }

    /// Resolve the path for a specific locale.
    pub fn resolve(&self, locale: &LocaleId) -> PathBuf {
        if let Some(ref pattern) = self.pattern {
            let resolved = pattern.replace("{locale}", &locale.as_str());
            PathBuf::from(resolved)
        } else {
            self.base_dir.join(locale.as_str()).join(&self.filename)
        }
    }

    /// Resolve the path for a locale with fallback, checking file existence.
    ///
    /// Tries each locale in the fallback chain until a file is found.
    pub fn resolve_with_fallback(&self, locale: &LocaleId) -> PathBuf {
        // Try exact locale.
        let path = self.resolve(locale);
        if path.exists() {
            return path;
        }

        // Try fallback chain.
        for fallback in locale.fallback_chain() {
            let fb_path = self.resolve(&fallback);
            if fb_path.exists() {
                log::debug!(
                    "[L10N] Asset fallback: {} -> {} (locale {} -> {})",
                    path.display(),
                    fb_path.display(),
                    locale,
                    fallback
                );
                return fb_path;
            }
        }

        // Return the original path even if it doesn't exist.
        log::warn!(
            "[L10N] No localized asset found for {} at {}",
            locale,
            path.display()
        );
        path
    }

    /// List all locale variants that exist on disk.
    pub fn find_available_locales(&self) -> Vec<LocaleId> {
        let mut result = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        let locale = LocaleId::from_str(name);
                        let full_path = entry.path().join(&self.filename);
                        if full_path.exists() {
                            result.push(locale);
                        }
                    }
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// FontSelector
// ---------------------------------------------------------------------------

/// A font configuration for a specific locale or language family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontConfig {
    /// Primary font file path.
    pub primary: String,
    /// Fallback font file paths (tried in order if glyphs are missing).
    pub fallbacks: Vec<String>,
    /// Font size adjustment factor (1.0 = default, >1.0 = larger).
    pub size_scale: f32,
    /// Line height multiplier.
    pub line_height: f32,
    /// Letter spacing adjustment in pixels.
    pub letter_spacing: f32,
    /// Whether to enable font hinting.
    pub hinting: bool,
}

impl FontConfig {
    /// Create a basic font config.
    pub fn new(primary: impl Into<String>) -> Self {
        Self {
            primary: primary.into(),
            fallbacks: Vec::new(),
            size_scale: 1.0,
            line_height: 1.2,
            letter_spacing: 0.0,
            hinting: true,
        }
    }

    /// Add a fallback font.
    pub fn with_fallback(mut self, fallback: impl Into<String>) -> Self {
        self.fallbacks.push(fallback.into());
        self
    }

    /// Set the size scale.
    pub fn with_size_scale(mut self, scale: f32) -> Self {
        self.size_scale = scale;
        self
    }

    /// Set the line height.
    pub fn with_line_height(mut self, height: f32) -> Self {
        self.line_height = height;
        self
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self::new("fonts/default.ttf")
    }
}

/// Selects appropriate fonts based on the active locale.
///
/// Different scripts require different fonts (Latin, CJK, Arabic, Devanagari,
/// Thai). The font selector maps locale language codes to font configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontSelector {
    /// Font configurations per language code.
    configs: HashMap<String, FontConfig>,
    /// Default font configuration.
    default_config: FontConfig,
}

impl FontSelector {
    /// Create a font selector with a default font.
    pub fn new(default: FontConfig) -> Self {
        Self {
            configs: HashMap::new(),
            default_config: default,
        }
    }

    /// Create a font selector with sensible defaults for common languages.
    pub fn with_defaults() -> Self {
        let mut selector = Self::new(
            FontConfig::new("fonts/NotoSans-Regular.ttf")
                .with_fallback("fonts/NotoSansCJK-Regular.ttf"),
        );

        // CJK fonts
        selector.set_font(
            "ja",
            FontConfig::new("fonts/NotoSansJP-Regular.ttf")
                .with_fallback("fonts/NotoSansCJK-Regular.ttf")
                .with_size_scale(1.0)
                .with_line_height(1.4),
        );
        selector.set_font(
            "zh",
            FontConfig::new("fonts/NotoSansSC-Regular.ttf")
                .with_fallback("fonts/NotoSansCJK-Regular.ttf")
                .with_line_height(1.4),
        );
        selector.set_font(
            "ko",
            FontConfig::new("fonts/NotoSansKR-Regular.ttf")
                .with_fallback("fonts/NotoSansCJK-Regular.ttf")
                .with_line_height(1.4),
        );

        // Arabic
        selector.set_font(
            "ar",
            FontConfig::new("fonts/NotoSansArabic-Regular.ttf")
                .with_size_scale(1.1)
                .with_line_height(1.5),
        );

        // Hindi / Devanagari
        selector.set_font(
            "hi",
            FontConfig::new("fonts/NotoSansDevanagari-Regular.ttf")
                .with_line_height(1.5),
        );

        // Thai
        selector.set_font(
            "th",
            FontConfig::new("fonts/NotoSansThai-Regular.ttf")
                .with_line_height(1.6),
        );

        selector
    }

    /// Set the font configuration for a language.
    pub fn set_font(&mut self, language: &str, config: FontConfig) {
        self.configs.insert(language.to_string(), config);
    }

    /// Get the font configuration for a locale.
    ///
    /// Looks up by language code, falling back to the default.
    pub fn get_font(&self, locale: &LocaleId) -> &FontConfig {
        let lang = locale.to_locale().language;
        self.configs.get(&lang).unwrap_or(&self.default_config)
    }

    /// Get the font configuration using a locale manager.
    pub fn get_font_for_current(&self, manager: &LocaleManager) -> &FontConfig {
        self.get_font(&manager.get_locale())
    }

    /// Get the default font configuration.
    pub fn default_font(&self) -> &FontConfig {
        &self.default_config
    }

    /// List all languages with custom font configurations.
    pub fn configured_languages(&self) -> Vec<&str> {
        self.configs.keys().map(|k| k.as_str()).collect()
    }
}

impl Default for FontSelector {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn localized_asset_resolve() {
        let mut asset = LocalizedAsset::with_default("default.png".to_string());
        asset.add_variant(&LocaleId::JaJP, "ja_JP.png".to_string());
        asset.add_variant(&LocaleId::EnUS, "en_US.png".to_string());

        assert_eq!(asset.resolve(&LocaleId::JaJP), Some(&"ja_JP.png".to_string()));
        assert_eq!(asset.resolve(&LocaleId::EnUS), Some(&"en_US.png".to_string()));
        // Unknown locale falls back through chain to en-US.
        assert_eq!(asset.resolve(&LocaleId::DeDE), Some(&"en_US.png".to_string()));
    }

    #[test]
    fn localized_asset_default_fallback() {
        let mut asset = LocalizedAsset::with_default("fallback.ogg".to_string());
        asset.add_variant(&LocaleId::EnUS, "en.ogg".to_string());

        // fr-FR -> en-US via fallback chain
        assert_eq!(asset.resolve(&LocaleId::FrFR), Some(&"en.ogg".to_string()));
    }

    #[test]
    fn localized_asset_no_variant() {
        let asset: LocalizedAsset<String> = LocalizedAsset::new();
        assert_eq!(asset.resolve(&LocaleId::EnUS), None);
    }

    #[test]
    fn asset_path_resolve() {
        let path = LocalizedAssetPath::new("assets/audio", "greeting.ogg");
        let resolved = path.resolve(&LocaleId::JaJP);
        assert!(resolved.to_str().unwrap().contains("ja-JP"));
        assert!(resolved.to_str().unwrap().contains("greeting.ogg"));
    }

    #[test]
    fn asset_path_pattern() {
        let path = LocalizedAssetPath::from_pattern("assets/voices/{locale}/intro.ogg");
        let resolved = path.resolve(&LocaleId::FrFR);
        assert_eq!(
            resolved.to_str().unwrap(),
            "assets/voices/fr-FR/intro.ogg"
        );
    }

    #[test]
    fn font_selector_lookup() {
        let selector = FontSelector::with_defaults();
        let ja_font = selector.get_font(&LocaleId::JaJP);
        assert!(ja_font.primary.contains("JP"));

        let en_font = selector.get_font(&LocaleId::EnUS);
        // Should return default font.
        assert!(en_font.primary.contains("NotoSans"));
    }

    #[test]
    fn font_selector_arabic() {
        let selector = FontSelector::with_defaults();
        let ar_font = selector.get_font(&LocaleId::ArSA);
        assert!(ar_font.primary.contains("Arabic"));
        assert!(ar_font.size_scale > 1.0);
    }

    #[test]
    fn localized_asset_variant_count() {
        let mut asset = LocalizedAsset::new();
        asset.add_variant(&LocaleId::EnUS, "en.png".to_string());
        asset.add_variant(&LocaleId::JaJP, "ja.png".to_string());
        assert_eq!(asset.variant_count(), 2);
        assert!(asset.has_variant(&LocaleId::EnUS));
        assert!(!asset.has_variant(&LocaleId::FrFR));
    }
}
