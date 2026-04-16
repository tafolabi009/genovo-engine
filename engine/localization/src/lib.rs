//! # Genovo Localization
//!
//! Internationalization (i18n) and localization (l10n) support for the Genovo
//! game engine. Provides locale management, string tables with template
//! substitution and pluralization, locale-aware number/date formatting, and
//! localized asset resolution.
//!
//! ## Quick start
//!
//! ```ignore
//! use genovo_localization::{LocaleId, LocaleManager, StringTable};
//!
//! let mut manager = LocaleManager::new(LocaleId::EnUS);
//! let mut table = StringTable::new();
//! table.insert("greeting", "Hello {name}!");
//! let result = table.get_formatted("greeting", &[("name", "World")]);
//! assert_eq!(result, "Hello World!");
//! ```

pub mod assets;
pub mod formatting;
pub mod locale;
pub mod strings;

// Re-exports for ergonomic access.
pub use assets::{FontSelector, LocalizedAsset, LocalizedAssetPath};
pub use formatting::{DateFormat, DatePattern, NumberFormat, PluralCategory, PluralRules};
pub use locale::{Locale, LocaleId, LocaleManager, TextDirection};
pub use strings::{StringTable, StringTableLoader, StringValidation};
