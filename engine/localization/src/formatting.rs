//! Locale-aware number, date, and plural formatting.
//!
//! Provides:
//! - [`NumberFormat`] -- decimal/thousands separators, currency, percentages.
//! - [`DateFormat`] -- date patterns, 12/24h time, relative time.
//! - [`PluralRules`] -- CLDR-based plural categories for many locales.
//! - [`PluralCategory`] -- the six CLDR plural categories.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::locale::LocaleId;

// ---------------------------------------------------------------------------
// PluralCategory
// ---------------------------------------------------------------------------

/// CLDR plural categories.
///
/// Not all locales use all categories. English uses `One` and `Other`;
/// Arabic uses all six; Japanese uses only `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluralCategory {
    /// Used for 0 in some languages (e.g., Arabic).
    Zero,
    /// Singular (e.g., English "1 item").
    One,
    /// Dual form (e.g., Arabic "2 items").
    Two,
    /// Few (e.g., Polish 2-4, Czech 2-4).
    Few,
    /// Many (e.g., Polish 5-21, Russian 5-20).
    Many,
    /// General plural / default.
    Other,
}

impl PluralCategory {
    /// Get the string name of this category.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::One => "one",
            Self::Two => "two",
            Self::Few => "few",
            Self::Many => "many",
            Self::Other => "other",
        }
    }

    /// Parse from a string name.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "zero" => Some(Self::Zero),
            "one" => Some(Self::One),
            "two" => Some(Self::Two),
            "few" => Some(Self::Few),
            "many" => Some(Self::Many),
            "other" => Some(Self::Other),
            _ => None,
        }
    }
}

impl fmt::Display for PluralCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// PluralRules
// ---------------------------------------------------------------------------

/// CLDR-based plural rule selector.
///
/// Given a locale and a numeric count, selects the appropriate
/// [`PluralCategory`]. The rules are hardcoded for common locales
/// following the CLDR specification.
#[derive(Debug, Clone)]
pub struct PluralRules {
    /// Language code this rule set applies to.
    language: String,
}

impl PluralRules {
    /// Create plural rules for a locale.
    pub fn for_locale(locale: &LocaleId) -> Self {
        Self {
            language: locale.to_locale().language,
        }
    }

    /// Create plural rules from a language code.
    pub fn for_language(language: &str) -> Self {
        Self {
            language: language.to_string(),
        }
    }

    /// Select the plural category for a given count.
    ///
    /// Implements CLDR plural rules for many languages. The count can be
    /// a float (some languages distinguish e.g., "1.0" from "1").
    pub fn select(&self, count: f64) -> PluralCategory {
        let n = count.abs();
        let i = n as u64; // integer part
        let v = decimal_places(count); // number of visible fraction digits
        let f = visible_fraction(count, v); // visible fraction digits as integer
        let _t = trailing_fraction(count, v); // fraction digits without trailing zeros

        match self.language.as_str() {
            // -- English, German, Dutch, Swedish, Danish, Norwegian, Italian,
            //    Portuguese, Spanish, Greek, Finnish, Estonian, Hungarian,
            //    Turkish (one/other) --
            "en" | "de" | "nl" | "sv" | "da" | "nb" | "nn" | "it" | "pt" | "es" | "el" | "fi"
            | "et" | "hu" | "tr" | "hi" | "bn" | "gu" | "kn" | "mr" | "te" | "ml" | "si"
            | "ca" | "gl" | "eu" | "af" | "bg" | "ka" | "sq" | "az" | "mn" | "ur" | "sw"
            | "zu" | "xh" | "is" => {
                // one: i = 1 and v = 0
                if i == 1 && v == 0 {
                    PluralCategory::One
                } else {
                    PluralCategory::Other
                }
            }

            // -- French (one: i = 0 or 1) --
            "fr" => {
                if i == 0 || i == 1 {
                    PluralCategory::One
                } else {
                    PluralCategory::Other
                }
            }

            // -- Japanese, Chinese, Korean, Thai, Vietnamese, Indonesian,
            //    Malay, Burmese, Lao, Khmer (other only) --
            "ja" | "zh" | "ko" | "th" | "vi" | "id" | "ms" | "my" | "lo" | "km" => {
                PluralCategory::Other
            }

            // -- Arabic (zero/one/two/few/many/other) --
            "ar" => {
                if n == 0.0 {
                    PluralCategory::Zero
                } else if n == 1.0 {
                    PluralCategory::One
                } else if n == 2.0 {
                    PluralCategory::Two
                } else {
                    let mod100 = i % 100;
                    if mod100 >= 3 && mod100 <= 10 {
                        PluralCategory::Few
                    } else if mod100 >= 11 && mod100 <= 99 {
                        PluralCategory::Many
                    } else {
                        PluralCategory::Other
                    }
                }
            }

            // -- Hebrew (one/two/other; with many for 11+) --
            "he" | "iw" => {
                if i == 1 && v == 0 {
                    PluralCategory::One
                } else if i == 2 && v == 0 {
                    PluralCategory::Two
                } else {
                    PluralCategory::Other
                }
            }

            // -- Polish (one/few/many/other) --
            "pl" => {
                if i == 1 && v == 0 {
                    PluralCategory::One
                } else if v == 0 {
                    let mod10 = i % 10;
                    let mod100 = i % 100;
                    if mod10 >= 2 && mod10 <= 4 && !(mod100 >= 12 && mod100 <= 14) {
                        PluralCategory::Few
                    } else if (mod10 == 0 || mod10 == 1)
                        || (mod10 >= 5 && mod10 <= 9)
                        || (mod100 >= 12 && mod100 <= 14)
                    {
                        PluralCategory::Many
                    } else {
                        PluralCategory::Other
                    }
                } else {
                    PluralCategory::Other
                }
            }

            // -- Russian, Ukrainian, Belarusian (one/few/many/other) --
            "ru" | "uk" | "be" => {
                if v == 0 {
                    let mod10 = i % 10;
                    let mod100 = i % 100;
                    if mod10 == 1 && mod100 != 11 {
                        PluralCategory::One
                    } else if mod10 >= 2 && mod10 <= 4 && !(mod100 >= 12 && mod100 <= 14) {
                        PluralCategory::Few
                    } else if mod10 == 0
                        || (mod10 >= 5 && mod10 <= 9)
                        || (mod100 >= 11 && mod100 <= 14)
                    {
                        PluralCategory::Many
                    } else {
                        PluralCategory::Other
                    }
                } else {
                    PluralCategory::Other
                }
            }

            // -- Czech, Slovak (one/few/other; many for fractions) --
            "cs" | "sk" => {
                if i == 1 && v == 0 {
                    PluralCategory::One
                } else if i >= 2 && i <= 4 && v == 0 {
                    PluralCategory::Few
                } else if v != 0 {
                    PluralCategory::Many
                } else {
                    PluralCategory::Other
                }
            }

            // -- Croatian, Serbian, Bosnian (one/few/other) --
            "hr" | "sr" | "bs" => {
                if v == 0 {
                    let mod10 = i % 10;
                    let mod100 = i % 100;
                    if mod10 == 1 && mod100 != 11 {
                        PluralCategory::One
                    } else if mod10 >= 2 && mod10 <= 4 && !(mod100 >= 12 && mod100 <= 14) {
                        PluralCategory::Few
                    } else {
                        PluralCategory::Other
                    }
                } else {
                    PluralCategory::Other
                }
            }

            // -- Lithuanian (one/few/many/other) --
            "lt" => {
                let mod10 = i % 10;
                let mod100 = i % 100;
                if v == 0 {
                    if mod10 == 1 && mod100 != 11 {
                        PluralCategory::One
                    } else if mod10 >= 2 && mod10 <= 9 && !(mod100 >= 12 && mod100 <= 19) {
                        PluralCategory::Few
                    } else {
                        PluralCategory::Other
                    }
                } else {
                    PluralCategory::Many
                }
            }

            // -- Latvian (zero/one/other) --
            "lv" => {
                let mod10 = i % 10;
                let mod100 = i % 100;
                if n == 0.0 {
                    PluralCategory::Zero
                } else if (mod10 == 1 && mod100 != 11) || (v != 0 && {
                    let fmod10 = f % 10;
                    let fmod100 = f % 100;
                    fmod10 == 1 && fmod100 != 11
                }) {
                    PluralCategory::One
                } else {
                    PluralCategory::Other
                }
            }

            // -- Romanian (one/few/other) --
            "ro" | "mo" => {
                if i == 1 && v == 0 {
                    PluralCategory::One
                } else {
                    let mod100 = i % 100;
                    if v != 0 || n == 0.0 || (mod100 >= 2 && mod100 <= 19) {
                        PluralCategory::Few
                    } else {
                        PluralCategory::Other
                    }
                }
            }

            // -- Slovenian (one/two/few/other) --
            "sl" => {
                if v == 0 {
                    let mod100 = i % 100;
                    if mod100 == 1 {
                        PluralCategory::One
                    } else if mod100 == 2 {
                        PluralCategory::Two
                    } else if mod100 == 3 || mod100 == 4 {
                        PluralCategory::Few
                    } else {
                        PluralCategory::Other
                    }
                } else {
                    PluralCategory::Other
                }
            }

            // -- Irish (one/two/few/many/other) --
            "ga" => {
                if n == 1.0 {
                    PluralCategory::One
                } else if n == 2.0 {
                    PluralCategory::Two
                } else if i >= 3 && i <= 6 {
                    PluralCategory::Few
                } else if i >= 7 && i <= 10 {
                    PluralCategory::Many
                } else {
                    PluralCategory::Other
                }
            }

            // -- Welsh (zero/one/two/few/many/other) --
            "cy" => {
                if n == 0.0 {
                    PluralCategory::Zero
                } else if n == 1.0 {
                    PluralCategory::One
                } else if n == 2.0 {
                    PluralCategory::Two
                } else if n == 3.0 {
                    PluralCategory::Few
                } else if n == 6.0 {
                    PluralCategory::Many
                } else {
                    PluralCategory::Other
                }
            }

            // -- Maltese (one/few/many/other) --
            "mt" => {
                if n == 1.0 {
                    PluralCategory::One
                } else {
                    let mod100 = i % 100;
                    if n == 0.0 || (mod100 >= 2 && mod100 <= 10) {
                        PluralCategory::Few
                    } else if mod100 >= 11 && mod100 <= 19 {
                        PluralCategory::Many
                    } else {
                        PluralCategory::Other
                    }
                }
            }

            // -- Default: one/other --
            _ => {
                if i == 1 && v == 0 {
                    PluralCategory::One
                } else {
                    PluralCategory::Other
                }
            }
        }
    }

    /// Get the available categories for this locale.
    pub fn categories(&self) -> Vec<PluralCategory> {
        match self.language.as_str() {
            "ja" | "zh" | "ko" | "th" | "vi" | "id" | "ms" => {
                vec![PluralCategory::Other]
            }
            "en" | "de" | "nl" | "sv" | "it" | "pt" | "es" | "hi" | "tr" | "fr" | "fi"
            | "et" | "hu" | "da" | "nb" | "nn" => {
                vec![PluralCategory::One, PluralCategory::Other]
            }
            "ar" => vec![
                PluralCategory::Zero,
                PluralCategory::One,
                PluralCategory::Two,
                PluralCategory::Few,
                PluralCategory::Many,
                PluralCategory::Other,
            ],
            "pl" | "ru" | "uk" | "lt" => vec![
                PluralCategory::One,
                PluralCategory::Few,
                PluralCategory::Many,
                PluralCategory::Other,
            ],
            "he" => vec![
                PluralCategory::One,
                PluralCategory::Two,
                PluralCategory::Other,
            ],
            "cs" | "sk" => vec![
                PluralCategory::One,
                PluralCategory::Few,
                PluralCategory::Many,
                PluralCategory::Other,
            ],
            "cy" => vec![
                PluralCategory::Zero,
                PluralCategory::One,
                PluralCategory::Two,
                PluralCategory::Few,
                PluralCategory::Many,
                PluralCategory::Other,
            ],
            _ => vec![PluralCategory::One, PluralCategory::Other],
        }
    }
}

/// Count the number of visible decimal places in a formatted number.
fn decimal_places(n: f64) -> u32 {
    let s = format!("{}", n);
    match s.find('.') {
        Some(pos) => (s.len() - pos - 1) as u32,
        None => 0,
    }
}

/// Get the visible fraction digits as an integer.
fn visible_fraction(n: f64, v: u32) -> u64 {
    if v == 0 {
        return 0;
    }
    let s = format!("{}", n);
    match s.find('.') {
        Some(pos) => {
            let frac = &s[pos + 1..];
            frac.parse().unwrap_or(0)
        }
        None => 0,
    }
}

/// Get the fraction digits without trailing zeros.
fn trailing_fraction(n: f64, v: u32) -> u64 {
    if v == 0 {
        return 0;
    }
    let f = visible_fraction(n, v);
    let mut t = f;
    while t > 0 && t % 10 == 0 {
        t /= 10;
    }
    t
}

// ---------------------------------------------------------------------------
// NumberFormat
// ---------------------------------------------------------------------------

/// Locale-aware number formatting configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberFormat {
    /// Decimal separator character.
    pub decimal_separator: char,
    /// Thousands grouping separator.
    pub thousands_separator: char,
    /// Grouping size (typically 3).
    pub grouping_size: u32,
    /// Whether to use grouping (thousands separators).
    pub use_grouping: bool,
    /// Currency symbol (e.g., "$", "EUR", "JPY").
    pub currency_symbol: String,
    /// Whether the currency symbol comes before the number.
    pub currency_prefix: bool,
    /// Space between currency symbol and number.
    pub currency_space: bool,
    /// Percent sign character.
    pub percent_sign: char,
    /// Minus sign character.
    pub minus_sign: char,
    /// Plus sign character.
    pub plus_sign: char,
}

impl NumberFormat {
    /// Create a number format for a locale.
    pub fn for_locale(locale: &LocaleId) -> Self {
        let lang = locale.to_locale().language;
        let _region = locale.to_locale().region.unwrap_or_default();

        match lang.as_str() {
            // Comma decimal, period thousands: most European languages
            "de" | "fr" | "it" | "es" | "pt" | "nl" | "pl" | "sv" | "da" | "nb" | "nn"
            | "fi" | "cs" | "sk" | "hr" | "sl" | "ro" | "bg" | "el" | "ru" | "uk" | "tr" => {
                Self {
                    decimal_separator: ',',
                    thousands_separator: '.',
                    grouping_size: 3,
                    use_grouping: true,
                    currency_symbol: Self::currency_for_locale(locale),
                    currency_prefix: false,
                    currency_space: true,
                    percent_sign: '%',
                    minus_sign: '-',
                    plus_sign: '+',
                }
            }
            // Space thousands, comma decimal: French, some others
            // (Already handled above; French uses period or space depending on region)

            // Arabic: uses Arabic-Indic digits in some regions, but we'll use Western Arabic
            "ar" => Self {
                decimal_separator: '.',
                thousands_separator: ',',
                grouping_size: 3,
                use_grouping: true,
                currency_symbol: Self::currency_for_locale(locale),
                currency_prefix: false,
                currency_space: true,
                percent_sign: '%',
                minus_sign: '-',
                plus_sign: '+',
            },

            // Japanese, Chinese, Korean: no thousands separator typically
            "ja" | "zh" | "ko" => Self {
                decimal_separator: '.',
                thousands_separator: ',',
                grouping_size: 3,
                use_grouping: true,
                currency_symbol: Self::currency_for_locale(locale),
                currency_prefix: true,
                currency_space: false,
                percent_sign: '%',
                minus_sign: '-',
                plus_sign: '+',
            },

            // Hindi: lakh grouping (2, 2, 3)
            "hi" => Self {
                decimal_separator: '.',
                thousands_separator: ',',
                grouping_size: 3, // First group is 3, then 2. Simplified to 3 here.
                use_grouping: true,
                currency_symbol: Self::currency_for_locale(locale),
                currency_prefix: true,
                currency_space: false,
                percent_sign: '%',
                minus_sign: '-',
                plus_sign: '+',
            },

            // Default: English-style
            _ => Self {
                decimal_separator: '.',
                thousands_separator: ',',
                grouping_size: 3,
                use_grouping: true,
                currency_symbol: Self::currency_for_locale(locale),
                currency_prefix: true,
                currency_space: false,
                percent_sign: '%',
                minus_sign: '-',
                plus_sign: '+',
            },
        }
    }

    /// Get the default currency symbol for a locale.
    fn currency_for_locale(locale: &LocaleId) -> String {
        match locale {
            LocaleId::EnUS => "$".to_string(),
            LocaleId::EnGB => "\u{00A3}".to_string(), // GBP
            LocaleId::JaJP => "\u{00A5}".to_string(), // Yen
            LocaleId::ZhCN => "\u{00A5}".to_string(), // Yuan
            LocaleId::KoKR => "\u{20A9}".to_string(), // Won
            LocaleId::RuRU => "\u{20BD}".to_string(), // Ruble
            LocaleId::ThTH => "\u{0E3F}".to_string(), // Baht
            LocaleId::HiIN => "\u{20B9}".to_string(), // Rupee
            LocaleId::TrTR => "\u{20BA}".to_string(), // Lira
            LocaleId::PtBR => "R$".to_string(),
            // Most European countries use Euro
            LocaleId::FrFR | LocaleId::DeDE | LocaleId::EsES | LocaleId::ItIT | LocaleId::NlNL
            | LocaleId::FrCA => "\u{20AC}".to_string(), // Euro
            _ => "$".to_string(),
        }
    }

    /// Format a number with the configured separators.
    pub fn format_number(&self, value: f64, decimals: u32) -> String {
        let negative = value < 0.0;
        let abs_val = value.abs();

        // Format the fractional part.
        let frac_str = if decimals > 0 {
            let factor = 10f64.powi(decimals as i32);
            let frac = ((abs_val * factor).round() as u64) % (factor as u64);
            format!("{:0>width$}", frac, width = decimals as usize)
        } else {
            String::new()
        };

        // Format the integer part with grouping.
        let int_part = abs_val.trunc() as u64;
        let int_str = self.format_integer(int_part);

        let mut result = String::new();
        if negative {
            result.push(self.minus_sign);
        }
        result.push_str(&int_str);
        if !frac_str.is_empty() {
            result.push(self.decimal_separator);
            result.push_str(&frac_str);
        }
        result
    }

    /// Format an integer with thousands grouping.
    fn format_integer(&self, value: u64) -> String {
        let digits = value.to_string();
        if !self.use_grouping || digits.len() <= self.grouping_size as usize {
            return digits;
        }

        let mut result = String::with_capacity(digits.len() + digits.len() / 3);
        let chars: Vec<char> = digits.chars().collect();
        let len = chars.len();

        for (i, ch) in chars.iter().enumerate() {
            result.push(*ch);
            let remaining = len - i - 1;
            if remaining > 0 && remaining % self.grouping_size as usize == 0 {
                result.push(self.thousands_separator);
            }
        }
        result
    }

    /// Format a currency value.
    pub fn format_currency(&self, value: f64, currency: Option<&str>) -> String {
        let symbol = currency.unwrap_or(&self.currency_symbol);
        let number = self.format_number(value, 2);

        if self.currency_prefix {
            if self.currency_space {
                format!("{} {}", symbol, number)
            } else {
                format!("{}{}", symbol, number)
            }
        } else if self.currency_space {
            format!("{} {}", number, symbol)
        } else {
            format!("{}{}", number, symbol)
        }
    }

    /// Format a percentage value.
    pub fn format_percent(&self, value: f64) -> String {
        let number = self.format_number(value * 100.0, 1);
        format!("{}{}", number, self.percent_sign)
    }
}

impl Default for NumberFormat {
    fn default() -> Self {
        Self::for_locale(&LocaleId::EnUS)
    }
}

// ---------------------------------------------------------------------------
// DateFormat
// ---------------------------------------------------------------------------

/// Date format pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatePattern {
    /// ISO 8601: 2026-04-16
    YearMonthDay,
    /// European: 16/04/2026
    DayMonthYear,
    /// US: 04/16/2026
    MonthDayYear,
    /// Long format: April 16, 2026
    Long,
    /// Short format: Apr 16
    Short,
}

impl Default for DatePattern {
    fn default() -> Self {
        Self::YearMonthDay
    }
}

/// Time format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeFormat {
    /// 24-hour clock: 14:30
    Hour24,
    /// 12-hour clock: 2:30 PM
    Hour12,
}

impl Default for TimeFormat {
    fn default() -> Self {
        Self::Hour24
    }
}

/// Locale-aware date and time formatting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateFormat {
    /// Date pattern.
    pub date_pattern: DatePattern,
    /// Time format.
    pub time_format: TimeFormat,
    /// Date separator character.
    pub date_separator: char,
    /// Month names (January..December).
    pub month_names: Vec<String>,
    /// Abbreviated month names (Jan..Dec).
    pub month_abbrevs: Vec<String>,
    /// Day names (Sunday..Saturday).
    pub day_names: Vec<String>,
    /// AM/PM labels.
    pub am_pm: (String, String),
    /// Relative time strings.
    pub relative_time: RelativeTimeStrings,
}

/// Strings for relative time formatting ("5 minutes ago", "in 2 days").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativeTimeStrings {
    pub now: String,
    pub seconds_ago: String,
    pub minute_ago: String,
    pub minutes_ago: String,
    pub hour_ago: String,
    pub hours_ago: String,
    pub day_ago: String,
    pub days_ago: String,
    pub in_seconds: String,
    pub in_minute: String,
    pub in_minutes: String,
    pub in_hour: String,
    pub in_hours: String,
    pub in_day: String,
    pub in_days: String,
}

impl Default for RelativeTimeStrings {
    fn default() -> Self {
        Self {
            now: "just now".to_string(),
            seconds_ago: "{n} seconds ago".to_string(),
            minute_ago: "1 minute ago".to_string(),
            minutes_ago: "{n} minutes ago".to_string(),
            hour_ago: "1 hour ago".to_string(),
            hours_ago: "{n} hours ago".to_string(),
            day_ago: "yesterday".to_string(),
            days_ago: "{n} days ago".to_string(),
            in_seconds: "in {n} seconds".to_string(),
            in_minute: "in 1 minute".to_string(),
            in_minutes: "in {n} minutes".to_string(),
            in_hour: "in 1 hour".to_string(),
            in_hours: "in {n} hours".to_string(),
            in_day: "tomorrow".to_string(),
            in_days: "in {n} days".to_string(),
        }
    }
}

impl DateFormat {
    /// Create date format for a locale.
    pub fn for_locale(locale: &LocaleId) -> Self {
        let lang = locale.to_locale().language;
        match lang.as_str() {
            "en" => Self::english(locale),
            "ja" => Self::japanese(),
            "de" => Self::german(),
            "fr" => Self::french(),
            "es" => Self::spanish(),
            "ru" => Self::russian(),
            "ar" => Self::arabic(),
            "zh" => Self::chinese(),
            "ko" => Self::korean(),
            _ => Self::english(locale),
        }
    }

    fn english(locale: &LocaleId) -> Self {
        let pattern = match locale {
            LocaleId::EnUS => DatePattern::MonthDayYear,
            _ => DatePattern::DayMonthYear,
        };
        Self {
            date_pattern: pattern,
            time_format: TimeFormat::Hour12,
            date_separator: '/',
            month_names: vec![
                "January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            month_abbrevs: vec![
                "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                "Dec",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            day_names: vec![
                "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            am_pm: ("AM".to_string(), "PM".to_string()),
            relative_time: RelativeTimeStrings::default(),
        }
    }

    fn japanese() -> Self {
        Self {
            date_pattern: DatePattern::YearMonthDay,
            time_format: TimeFormat::Hour24,
            date_separator: '/',
            month_names: (1..=12).map(|m| format!("{}月", m)).collect(),
            month_abbrevs: (1..=12).map(|m| format!("{}月", m)).collect(),
            day_names: vec![
                "日曜日", "月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            am_pm: ("午前".to_string(), "午後".to_string()),
            relative_time: RelativeTimeStrings {
                now: "たった今".to_string(),
                seconds_ago: "{n}秒前".to_string(),
                minute_ago: "1分前".to_string(),
                minutes_ago: "{n}分前".to_string(),
                hour_ago: "1時間前".to_string(),
                hours_ago: "{n}時間前".to_string(),
                day_ago: "昨日".to_string(),
                days_ago: "{n}日前".to_string(),
                in_seconds: "{n}秒後".to_string(),
                in_minute: "1分後".to_string(),
                in_minutes: "{n}分後".to_string(),
                in_hour: "1時間後".to_string(),
                in_hours: "{n}時間後".to_string(),
                in_day: "明日".to_string(),
                in_days: "{n}日後".to_string(),
            },
        }
    }

    fn german() -> Self {
        Self {
            date_pattern: DatePattern::DayMonthYear,
            time_format: TimeFormat::Hour24,
            date_separator: '.',
            month_names: vec![
                "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
                "September", "Oktober", "November", "Dezember",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            month_abbrevs: vec![
                "Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov",
                "Dez",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            day_names: vec![
                "Sonntag", "Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            am_pm: ("AM".to_string(), "PM".to_string()),
            relative_time: RelativeTimeStrings {
                now: "gerade eben".to_string(),
                seconds_ago: "vor {n} Sekunden".to_string(),
                minute_ago: "vor 1 Minute".to_string(),
                minutes_ago: "vor {n} Minuten".to_string(),
                hour_ago: "vor 1 Stunde".to_string(),
                hours_ago: "vor {n} Stunden".to_string(),
                day_ago: "gestern".to_string(),
                days_ago: "vor {n} Tagen".to_string(),
                in_seconds: "in {n} Sekunden".to_string(),
                in_minute: "in 1 Minute".to_string(),
                in_minutes: "in {n} Minuten".to_string(),
                in_hour: "in 1 Stunde".to_string(),
                in_hours: "in {n} Stunden".to_string(),
                in_day: "morgen".to_string(),
                in_days: "in {n} Tagen".to_string(),
            },
        }
    }

    fn french() -> Self {
        Self {
            date_pattern: DatePattern::DayMonthYear,
            time_format: TimeFormat::Hour24,
            date_separator: '/',
            month_names: vec![
                "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
                "septembre", "octobre", "novembre", "décembre",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            month_abbrevs: vec![
                "janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.",
                "nov.", "déc.",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            day_names: vec![
                "dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            am_pm: ("AM".to_string(), "PM".to_string()),
            relative_time: RelativeTimeStrings {
                now: "à l'instant".to_string(),
                seconds_ago: "il y a {n} secondes".to_string(),
                minute_ago: "il y a 1 minute".to_string(),
                minutes_ago: "il y a {n} minutes".to_string(),
                hour_ago: "il y a 1 heure".to_string(),
                hours_ago: "il y a {n} heures".to_string(),
                day_ago: "hier".to_string(),
                days_ago: "il y a {n} jours".to_string(),
                in_seconds: "dans {n} secondes".to_string(),
                in_minute: "dans 1 minute".to_string(),
                in_minutes: "dans {n} minutes".to_string(),
                in_hour: "dans 1 heure".to_string(),
                in_hours: "dans {n} heures".to_string(),
                in_day: "demain".to_string(),
                in_days: "dans {n} jours".to_string(),
            },
        }
    }

    fn spanish() -> Self {
        Self {
            date_pattern: DatePattern::DayMonthYear,
            time_format: TimeFormat::Hour24,
            date_separator: '/',
            month_names: vec![
                "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                "septiembre", "octubre", "noviembre", "diciembre",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            month_abbrevs: vec![
                "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov",
                "dic",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            day_names: vec![
                "domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            am_pm: ("a.\u{00A0}m.".to_string(), "p.\u{00A0}m.".to_string()),
            relative_time: RelativeTimeStrings {
                now: "ahora mismo".to_string(),
                seconds_ago: "hace {n} segundos".to_string(),
                minute_ago: "hace 1 minuto".to_string(),
                minutes_ago: "hace {n} minutos".to_string(),
                hour_ago: "hace 1 hora".to_string(),
                hours_ago: "hace {n} horas".to_string(),
                day_ago: "ayer".to_string(),
                days_ago: "hace {n} días".to_string(),
                in_seconds: "en {n} segundos".to_string(),
                in_minute: "en 1 minuto".to_string(),
                in_minutes: "en {n} minutos".to_string(),
                in_hour: "en 1 hora".to_string(),
                in_hours: "en {n} horas".to_string(),
                in_day: "mañana".to_string(),
                in_days: "en {n} días".to_string(),
            },
        }
    }

    fn russian() -> Self {
        Self {
            date_pattern: DatePattern::DayMonthYear,
            time_format: TimeFormat::Hour24,
            date_separator: '.',
            month_names: vec![
                "январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август",
                "сентябрь", "октябрь", "ноябрь", "декабрь",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            month_abbrevs: vec![
                "янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя",
                "дек",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            day_names: vec![
                "воскресенье",
                "понедельник",
                "вторник",
                "среда",
                "четверг",
                "пятница",
                "суббота",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            am_pm: ("AM".to_string(), "PM".to_string()),
            relative_time: RelativeTimeStrings {
                now: "только что".to_string(),
                seconds_ago: "{n} секунд назад".to_string(),
                minute_ago: "1 минуту назад".to_string(),
                minutes_ago: "{n} минут назад".to_string(),
                hour_ago: "1 час назад".to_string(),
                hours_ago: "{n} часов назад".to_string(),
                day_ago: "вчера".to_string(),
                days_ago: "{n} дней назад".to_string(),
                in_seconds: "через {n} секунд".to_string(),
                in_minute: "через 1 минуту".to_string(),
                in_minutes: "через {n} минут".to_string(),
                in_hour: "через 1 час".to_string(),
                in_hours: "через {n} часов".to_string(),
                in_day: "завтра".to_string(),
                in_days: "через {n} дней".to_string(),
            },
        }
    }

    fn arabic() -> Self {
        Self {
            date_pattern: DatePattern::DayMonthYear,
            time_format: TimeFormat::Hour12,
            date_separator: '/',
            month_names: vec![
                "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس",
                "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            month_abbrevs: vec![
                "ينا", "فبر", "مار", "أبر", "ماي", "يون", "يول", "أغس", "سبت", "أكت", "نوف",
                "ديس",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            day_names: vec![
                "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            am_pm: ("ص".to_string(), "م".to_string()),
            relative_time: RelativeTimeStrings::default(),
        }
    }

    fn chinese() -> Self {
        Self {
            date_pattern: DatePattern::YearMonthDay,
            time_format: TimeFormat::Hour24,
            date_separator: '-',
            month_names: (1..=12).map(|m| format!("{}月", m)).collect(),
            month_abbrevs: (1..=12).map(|m| format!("{}月", m)).collect(),
            day_names: vec![
                "星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            am_pm: ("上午".to_string(), "下午".to_string()),
            relative_time: RelativeTimeStrings {
                now: "刚刚".to_string(),
                seconds_ago: "{n}秒前".to_string(),
                minute_ago: "1分钟前".to_string(),
                minutes_ago: "{n}分钟前".to_string(),
                hour_ago: "1小时前".to_string(),
                hours_ago: "{n}小时前".to_string(),
                day_ago: "昨天".to_string(),
                days_ago: "{n}天前".to_string(),
                in_seconds: "{n}秒后".to_string(),
                in_minute: "1分钟后".to_string(),
                in_minutes: "{n}分钟后".to_string(),
                in_hour: "1小时后".to_string(),
                in_hours: "{n}小时后".to_string(),
                in_day: "明天".to_string(),
                in_days: "{n}天后".to_string(),
            },
        }
    }

    fn korean() -> Self {
        Self {
            date_pattern: DatePattern::YearMonthDay,
            time_format: TimeFormat::Hour12,
            date_separator: '.',
            month_names: (1..=12).map(|m| format!("{}월", m)).collect(),
            month_abbrevs: (1..=12).map(|m| format!("{}월", m)).collect(),
            day_names: vec![
                "일요일", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            am_pm: ("오전".to_string(), "오후".to_string()),
            relative_time: RelativeTimeStrings {
                now: "방금".to_string(),
                seconds_ago: "{n}초 전".to_string(),
                minute_ago: "1분 전".to_string(),
                minutes_ago: "{n}분 전".to_string(),
                hour_ago: "1시간 전".to_string(),
                hours_ago: "{n}시간 전".to_string(),
                day_ago: "어제".to_string(),
                days_ago: "{n}일 전".to_string(),
                in_seconds: "{n}초 후".to_string(),
                in_minute: "1분 후".to_string(),
                in_minutes: "{n}분 후".to_string(),
                in_hour: "1시간 후".to_string(),
                in_hours: "{n}시간 후".to_string(),
                in_day: "내일".to_string(),
                in_days: "{n}일 후".to_string(),
            },
        }
    }

    /// Format a date from year/month/day components.
    pub fn format_date(&self, year: i32, month: u32, day: u32) -> String {
        let m = month.clamp(1, 12);
        let d = day.clamp(1, 31);
        let sep = self.date_separator;

        match self.date_pattern {
            DatePattern::YearMonthDay => format!("{:04}{}{:02}{}{:02}", year, sep, m, sep, d),
            DatePattern::DayMonthYear => format!("{:02}{}{:02}{}{:04}", d, sep, m, sep, year),
            DatePattern::MonthDayYear => format!("{:02}{}{:02}{}{:04}", m, sep, d, sep, year),
            DatePattern::Long => {
                let month_name = &self.month_names[(m - 1) as usize];
                format!("{} {}, {}", month_name, d, year)
            }
            DatePattern::Short => {
                let month_abbrev = &self.month_abbrevs[(m - 1) as usize];
                format!("{} {}", month_abbrev, d)
            }
        }
    }

    /// Format a time from hour/minute/second components.
    pub fn format_time(&self, hour: u32, minute: u32, second: u32) -> String {
        let h = hour.clamp(0, 23);
        let m = minute.clamp(0, 59);
        let s = second.clamp(0, 59);

        match self.time_format {
            TimeFormat::Hour24 => format!("{:02}:{:02}:{:02}", h, m, s),
            TimeFormat::Hour12 => {
                let (h12, period) = if h == 0 {
                    (12, &self.am_pm.0)
                } else if h < 12 {
                    (h, &self.am_pm.0)
                } else if h == 12 {
                    (12, &self.am_pm.1)
                } else {
                    (h - 12, &self.am_pm.1)
                };
                format!("{:02}:{:02}:{:02} {}", h12, m, s, period)
            }
        }
    }

    /// Format a relative time from a delta in seconds.
    ///
    /// Positive delta = past ("5 minutes ago"), negative = future ("in 5 minutes").
    pub fn format_relative(&self, delta_seconds: i64) -> String {
        let abs = delta_seconds.unsigned_abs();
        let rt = &self.relative_time;

        if abs < 10 {
            return rt.now.clone();
        }

        let (template, n) = if delta_seconds > 0 {
            // Past
            if abs < 60 {
                (&rt.seconds_ago, abs)
            } else if abs < 120 {
                (&rt.minute_ago, 1)
            } else if abs < 3600 {
                (&rt.minutes_ago, abs / 60)
            } else if abs < 7200 {
                (&rt.hour_ago, 1)
            } else if abs < 86400 {
                (&rt.hours_ago, abs / 3600)
            } else if abs < 172800 {
                (&rt.day_ago, 1)
            } else {
                (&rt.days_ago, abs / 86400)
            }
        } else {
            // Future
            if abs < 60 {
                (&rt.in_seconds, abs)
            } else if abs < 120 {
                (&rt.in_minute, 1)
            } else if abs < 3600 {
                (&rt.in_minutes, abs / 60)
            } else if abs < 7200 {
                (&rt.in_hour, 1)
            } else if abs < 86400 {
                (&rt.in_hours, abs / 3600)
            } else if abs < 172800 {
                (&rt.in_day, 1)
            } else {
                (&rt.in_days, abs / 86400)
            }
        };

        template.replace("{n}", &n.to_string())
    }

    /// Get the month name for a 1-based month index.
    pub fn month_name(&self, month: u32) -> &str {
        let idx = (month.clamp(1, 12) - 1) as usize;
        &self.month_names[idx]
    }

    /// Get the abbreviated month name.
    pub fn month_abbrev(&self, month: u32) -> &str {
        let idx = (month.clamp(1, 12) - 1) as usize;
        &self.month_abbrevs[idx]
    }

    /// Get the day name for a 0-based day index (0=Sunday).
    pub fn day_name(&self, day: u32) -> &str {
        let idx = (day % 7) as usize;
        &self.day_names[idx]
    }
}

impl Default for DateFormat {
    fn default() -> Self {
        Self::for_locale(&LocaleId::EnUS)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Plural rules tests --

    #[test]
    fn english_plurals() {
        let rules = PluralRules::for_locale(&LocaleId::EnUS);
        assert_eq!(rules.select(0.0), PluralCategory::Other);
        assert_eq!(rules.select(1.0), PluralCategory::One);
        assert_eq!(rules.select(2.0), PluralCategory::Other);
        assert_eq!(rules.select(100.0), PluralCategory::Other);
    }

    #[test]
    fn french_plurals() {
        let rules = PluralRules::for_locale(&LocaleId::FrFR);
        assert_eq!(rules.select(0.0), PluralCategory::One);
        assert_eq!(rules.select(1.0), PluralCategory::One);
        assert_eq!(rules.select(2.0), PluralCategory::Other);
    }

    #[test]
    fn japanese_plurals() {
        let rules = PluralRules::for_locale(&LocaleId::JaJP);
        assert_eq!(rules.select(0.0), PluralCategory::Other);
        assert_eq!(rules.select(1.0), PluralCategory::Other);
        assert_eq!(rules.select(100.0), PluralCategory::Other);
    }

    #[test]
    fn arabic_plurals() {
        let rules = PluralRules::for_locale(&LocaleId::ArSA);
        assert_eq!(rules.select(0.0), PluralCategory::Zero);
        assert_eq!(rules.select(1.0), PluralCategory::One);
        assert_eq!(rules.select(2.0), PluralCategory::Two);
        assert_eq!(rules.select(3.0), PluralCategory::Few);
        assert_eq!(rules.select(10.0), PluralCategory::Few);
        assert_eq!(rules.select(11.0), PluralCategory::Many);
        assert_eq!(rules.select(99.0), PluralCategory::Many);
        assert_eq!(rules.select(100.0), PluralCategory::Other);
    }

    #[test]
    fn polish_plurals() {
        let rules = PluralRules::for_locale(&LocaleId::PlPL);
        assert_eq!(rules.select(1.0), PluralCategory::One);
        assert_eq!(rules.select(2.0), PluralCategory::Few);
        assert_eq!(rules.select(3.0), PluralCategory::Few);
        assert_eq!(rules.select(4.0), PluralCategory::Few);
        assert_eq!(rules.select(5.0), PluralCategory::Many);
        assert_eq!(rules.select(12.0), PluralCategory::Many);
        assert_eq!(rules.select(21.0), PluralCategory::Many);
        assert_eq!(rules.select(22.0), PluralCategory::Few);
        assert_eq!(rules.select(25.0), PluralCategory::Many);
    }

    #[test]
    fn russian_plurals() {
        let rules = PluralRules::for_locale(&LocaleId::RuRU);
        assert_eq!(rules.select(1.0), PluralCategory::One);
        assert_eq!(rules.select(2.0), PluralCategory::Few);
        assert_eq!(rules.select(5.0), PluralCategory::Many);
        assert_eq!(rules.select(11.0), PluralCategory::Many);
        assert_eq!(rules.select(21.0), PluralCategory::One);
        assert_eq!(rules.select(22.0), PluralCategory::Few);
        assert_eq!(rules.select(25.0), PluralCategory::Many);
    }

    // -- Number format tests --

    #[test]
    fn number_format_en_us() {
        let fmt = NumberFormat::for_locale(&LocaleId::EnUS);
        assert_eq!(fmt.format_number(1234.56, 2), "1,234.56");
        assert_eq!(fmt.format_number(0.5, 1), "0.5");
        assert_eq!(fmt.format_number(-42.0, 0), "-42");
    }

    #[test]
    fn number_format_de_de() {
        let fmt = NumberFormat::for_locale(&LocaleId::DeDE);
        assert_eq!(fmt.format_number(1234.56, 2), "1.234,56");
    }

    #[test]
    fn currency_format_en_us() {
        let fmt = NumberFormat::for_locale(&LocaleId::EnUS);
        assert_eq!(fmt.format_currency(9.99, None), "$9.99");
    }

    #[test]
    fn currency_format_de_de() {
        let fmt = NumberFormat::for_locale(&LocaleId::DeDE);
        let result = fmt.format_currency(9.99, None);
        assert!(result.contains("9,99"), "Expected comma decimal: {}", result);
        assert!(result.contains('\u{20AC}'), "Expected Euro sign: {}", result);
    }

    #[test]
    fn percent_format() {
        let fmt = NumberFormat::for_locale(&LocaleId::EnUS);
        assert_eq!(fmt.format_percent(0.75), "75.0%");
    }

    // -- Date format tests --

    #[test]
    fn date_format_en_us() {
        let fmt = DateFormat::for_locale(&LocaleId::EnUS);
        let result = fmt.format_date(2026, 4, 16);
        assert_eq!(result, "04/16/2026");
    }

    #[test]
    fn date_format_de_de() {
        let fmt = DateFormat::for_locale(&LocaleId::DeDE);
        let result = fmt.format_date(2026, 4, 16);
        assert_eq!(result, "16.04.2026");
    }

    #[test]
    fn date_format_ja_jp() {
        let fmt = DateFormat::for_locale(&LocaleId::JaJP);
        let result = fmt.format_date(2026, 4, 16);
        assert_eq!(result, "2026/04/16");
    }

    #[test]
    fn time_format_24h() {
        let fmt = DateFormat::for_locale(&LocaleId::DeDE);
        assert_eq!(fmt.format_time(14, 30, 0), "14:30:00");
    }

    #[test]
    fn time_format_12h() {
        let fmt = DateFormat::for_locale(&LocaleId::EnUS);
        assert_eq!(fmt.format_time(14, 30, 0), "02:30:00 PM");
        assert_eq!(fmt.format_time(0, 0, 0), "12:00:00 AM");
    }

    #[test]
    fn relative_time_past() {
        let fmt = DateFormat::for_locale(&LocaleId::EnUS);
        assert_eq!(fmt.format_relative(5), "just now");
        assert_eq!(fmt.format_relative(30), "30 seconds ago");
        assert_eq!(fmt.format_relative(90), "1 minute ago");
        assert_eq!(fmt.format_relative(300), "5 minutes ago");
        assert_eq!(fmt.format_relative(7200), "2 hours ago");
        assert_eq!(fmt.format_relative(90000), "yesterday");
        assert_eq!(fmt.format_relative(259200), "3 days ago");
    }

    #[test]
    fn relative_time_future() {
        let fmt = DateFormat::for_locale(&LocaleId::EnUS);
        assert_eq!(fmt.format_relative(-300), "in 5 minutes");
        assert_eq!(fmt.format_relative(-7200), "in 2 hours");
        assert_eq!(fmt.format_relative(-90000), "tomorrow");
    }

    #[test]
    fn relative_time_japanese() {
        let fmt = DateFormat::for_locale(&LocaleId::JaJP);
        assert_eq!(fmt.format_relative(300), "5分前");
        assert_eq!(fmt.format_relative(-300), "5分後");
        assert_eq!(fmt.format_relative(90000), "昨日");
    }

    #[test]
    fn month_names() {
        let fmt = DateFormat::for_locale(&LocaleId::FrFR);
        assert_eq!(fmt.month_name(1), "janvier");
        assert_eq!(fmt.month_name(12), "décembre");
    }

    #[test]
    fn plural_categories_returned() {
        let en = PluralRules::for_locale(&LocaleId::EnUS);
        assert_eq!(en.categories().len(), 2);

        let ar = PluralRules::for_locale(&LocaleId::ArSA);
        assert_eq!(ar.categories().len(), 6);

        let ja = PluralRules::for_locale(&LocaleId::JaJP);
        assert_eq!(ja.categories().len(), 1);
    }
}
