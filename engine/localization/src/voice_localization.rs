//! Voice Localization System
//!
//! Manages voice-over assets across multiple locales, providing:
//!
//! - Voice line registry per locale
//! - Voice file path resolution
//! - Fallback chain (dialect -> language -> default)
//! - Lip sync timing data per locale
//! - Voice volume normalization
//! - Voice casting notes for voice directors
//!
//! # Architecture
//!
//! ```text
//! VoiceLocalizationManager
//!   +-- VoiceLineRegistry       (id -> voice line metadata)
//!   +-- LocaleVoiceBank[]       (per-locale voice asset paths)
//!   +-- FallbackChain           (locale resolution order)
//!   +-- LipSyncDatabase         (per-locale timing data)
//!   +-- VolumeNormalizer        (loudness normalization)
//!   +-- CastingNoteDatabase     (voice director notes)
//! ```
//!
//! # Fallback Example
//!
//! ```text
//! Request: "es_MX" (Mexican Spanish)
//!   1. Try es_MX voice bank
//!   2. Try es_ES voice bank (Castilian Spanish)
//!   3. Try en_US voice bank (default)
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// VoiceLineId
// ---------------------------------------------------------------------------

/// Unique identifier for a voice line.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VoiceLineId(pub String);

impl VoiceLineId {
    /// Creates a new voice line ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for VoiceLineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// VoiceLineMeta
// ---------------------------------------------------------------------------

/// Metadata for a voice line, independent of locale.
#[derive(Debug, Clone)]
pub struct VoiceLineMeta {
    /// Unique identifier.
    pub id: VoiceLineId,
    /// Human-readable description.
    pub description: String,
    /// The character/speaker for this line.
    pub speaker: String,
    /// The script text (reference language).
    pub script_text: String,
    /// The emotion or delivery direction.
    pub emotion: Option<String>,
    /// Context notes (what is happening in the scene).
    pub context: Option<String>,
    /// Whether this line is critical for gameplay (must be present).
    pub critical: bool,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Estimated duration in seconds (for scheduling).
    pub estimated_duration: f32,
    /// Related voice line IDs (for dialogue chains).
    pub related_lines: Vec<VoiceLineId>,
    /// Whether this line should be played as a one-shot or can loop.
    pub loop_allowed: bool,
}

impl VoiceLineMeta {
    /// Creates new voice line metadata.
    pub fn new(
        id: impl Into<String>,
        speaker: impl Into<String>,
        script_text: impl Into<String>,
    ) -> Self {
        Self {
            id: VoiceLineId::new(id),
            description: String::new(),
            speaker: speaker.into(),
            script_text: script_text.into(),
            emotion: None,
            context: None,
            critical: false,
            tags: Vec::new(),
            estimated_duration: 0.0,
            related_lines: Vec::new(),
            loop_allowed: false,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Sets the emotion/delivery direction.
    pub fn with_emotion(mut self, emotion: impl Into<String>) -> Self {
        self.emotion = Some(emotion.into());
        self
    }

    /// Sets the context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Marks as critical.
    pub fn with_critical(mut self) -> Self {
        self.critical = true;
        self
    }

    /// Adds a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Sets the estimated duration.
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.estimated_duration = duration;
        self
    }

    /// Adds a related voice line.
    pub fn with_related(mut self, related_id: impl Into<String>) -> Self {
        self.related_lines.push(VoiceLineId::new(related_id));
        self
    }
}

// ---------------------------------------------------------------------------
// VoiceAssetInfo
// ---------------------------------------------------------------------------

/// Information about a voice audio file for a specific locale.
#[derive(Debug, Clone)]
pub struct VoiceAssetInfo {
    /// Path to the audio file (relative to the voice assets root).
    pub file_path: String,
    /// Audio format.
    pub format: AudioFormat,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
    /// Duration in seconds.
    pub duration: f32,
    /// File size in bytes.
    pub file_size: u64,
    /// Peak amplitude (for normalization).
    pub peak_amplitude: f32,
    /// RMS loudness (for normalization).
    pub rms_loudness: f32,
    /// Whether this asset has been validated (file exists, correct format).
    pub validated: bool,
    /// Voice actor name.
    pub voice_actor: Option<String>,
    /// Recording session identifier.
    pub session_id: Option<String>,
    /// Take number.
    pub take_number: u32,
}

/// Audio file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Ogg,
    Mp3,
    Flac,
    Aac,
    Opus,
}

impl AudioFormat {
    /// Returns the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Ogg => "ogg",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
            Self::Aac => "aac",
            Self::Opus => "opus",
        }
    }
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::Ogg
    }
}

impl VoiceAssetInfo {
    /// Creates new voice asset info.
    pub fn new(file_path: impl Into<String>, format: AudioFormat, duration: f32) -> Self {
        Self {
            file_path: file_path.into(),
            format,
            sample_rate: 44100,
            channels: 1,
            duration,
            file_size: 0,
            peak_amplitude: 1.0,
            rms_loudness: -20.0,
            validated: false,
            voice_actor: None,
            session_id: None,
            take_number: 1,
        }
    }

    /// Sets the voice actor name.
    pub fn with_voice_actor(mut self, actor: impl Into<String>) -> Self {
        self.voice_actor = Some(actor.into());
        self
    }

    /// Sets the RMS loudness.
    pub fn with_loudness(mut self, rms: f32) -> Self {
        self.rms_loudness = rms;
        self
    }

    /// Returns the full file path given a root directory.
    pub fn full_path(&self, root: &Path) -> PathBuf {
        root.join(&self.file_path)
    }
}

// ---------------------------------------------------------------------------
// LipSyncData
// ---------------------------------------------------------------------------

/// Lip synchronization timing data for a voice line.
#[derive(Debug, Clone)]
pub struct LipSyncData {
    /// The voice line this data corresponds to.
    pub voice_line_id: VoiceLineId,
    /// The locale this lip sync data is for.
    pub locale: String,
    /// Phoneme events sorted by time.
    pub phonemes: Vec<PhonemeEvent>,
    /// Viseme events sorted by time (mapped from phonemes).
    pub visemes: Vec<VisemeEvent>,
    /// Total duration of the lip sync data.
    pub duration: f32,
    /// Whether this data was auto-generated or manually authored.
    pub auto_generated: bool,
}

/// A phoneme event at a specific time.
#[derive(Debug, Clone)]
pub struct PhonemeEvent {
    /// Time in seconds.
    pub time: f32,
    /// Duration of this phoneme.
    pub duration: f32,
    /// Phoneme identifier (IPA or engine-specific).
    pub phoneme: String,
    /// Intensity/weight (0.0 to 1.0).
    pub intensity: f32,
}

/// A viseme (visual phoneme) event.
#[derive(Debug, Clone)]
pub struct VisemeEvent {
    /// Time in seconds.
    pub time: f32,
    /// Duration.
    pub duration: f32,
    /// Viseme identifier (mapped from phoneme).
    pub viseme: VisemeType,
    /// Blend weight (0.0 to 1.0).
    pub weight: f32,
}

/// Standard viseme types for facial animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VisemeType {
    /// Silence / neutral mouth.
    Silent,
    /// "AH" sound.
    Aa,
    /// "EE" sound.
    Ee,
    /// "IH" sound.
    Ih,
    /// "OH" sound.
    Oh,
    /// "OO" sound.
    Oo,
    /// "SS" / "SH" sound.
    Ss,
    /// "FF" / "VV" sound.
    Ff,
    /// "TH" sound.
    Th,
    /// "PP" / "BB" / "MM" sound (lips together).
    Pp,
    /// "DD" / "TT" / "NN" sound.
    Dd,
    /// "KK" / "GG" sound.
    Kk,
    /// "CH" / "JJ" sound.
    Ch,
    /// "RR" sound.
    Rr,
}

impl Default for VisemeType {
    fn default() -> Self {
        Self::Silent
    }
}

impl fmt::Display for VisemeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Silent => write!(f, "SIL"),
            Self::Aa => write!(f, "AA"),
            Self::Ee => write!(f, "EE"),
            Self::Ih => write!(f, "IH"),
            Self::Oh => write!(f, "OH"),
            Self::Oo => write!(f, "OO"),
            Self::Ss => write!(f, "SS"),
            Self::Ff => write!(f, "FF"),
            Self::Th => write!(f, "TH"),
            Self::Pp => write!(f, "PP"),
            Self::Dd => write!(f, "DD"),
            Self::Kk => write!(f, "KK"),
            Self::Ch => write!(f, "CH"),
            Self::Rr => write!(f, "RR"),
        }
    }
}

impl LipSyncData {
    /// Creates new lip sync data.
    pub fn new(voice_line_id: VoiceLineId, locale: impl Into<String>) -> Self {
        Self {
            voice_line_id,
            locale: locale.into(),
            phonemes: Vec::new(),
            visemes: Vec::new(),
            duration: 0.0,
            auto_generated: true,
        }
    }

    /// Adds a phoneme event.
    pub fn add_phoneme(&mut self, event: PhonemeEvent) {
        let end = event.time + event.duration;
        if end > self.duration {
            self.duration = end;
        }
        self.phonemes.push(event);
        self.phonemes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Adds a viseme event.
    pub fn add_viseme(&mut self, event: VisemeEvent) {
        let end = event.time + event.duration;
        if end > self.duration {
            self.duration = end;
        }
        self.visemes.push(event);
        self.visemes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Returns the active viseme at the given time.
    pub fn viseme_at(&self, time: f32) -> Option<&VisemeEvent> {
        self.visemes
            .iter()
            .rev()
            .find(|v| time >= v.time && time < v.time + v.duration)
    }

    /// Returns the number of phoneme events.
    pub fn phoneme_count(&self) -> usize {
        self.phonemes.len()
    }

    /// Returns the number of viseme events.
    pub fn viseme_count(&self) -> usize {
        self.visemes.len()
    }
}

// ---------------------------------------------------------------------------
// FallbackChain
// ---------------------------------------------------------------------------

/// Defines the locale fallback order for voice resolution.
#[derive(Debug, Clone)]
pub struct FallbackChain {
    /// Ordered list of fallback rules.
    rules: Vec<FallbackRule>,
    /// The ultimate default locale.
    pub default_locale: String,
}

/// A fallback rule mapping a locale to its fallback.
#[derive(Debug, Clone)]
pub struct FallbackRule {
    /// The locale this rule applies to.
    pub locale: String,
    /// The fallback locale to try.
    pub fallback: String,
}

impl FallbackChain {
    /// Creates a new fallback chain with the given default.
    pub fn new(default_locale: impl Into<String>) -> Self {
        Self {
            rules: Vec::new(),
            default_locale: default_locale.into(),
        }
    }

    /// Adds a fallback rule.
    pub fn add_rule(&mut self, locale: impl Into<String>, fallback: impl Into<String>) {
        self.rules.push(FallbackRule {
            locale: locale.into(),
            fallback: fallback.into(),
        });
    }

    /// Resolves the fallback chain for a given locale.
    ///
    /// Returns a list of locales to try in order, ending with the
    /// default locale.
    pub fn resolve(&self, locale: &str) -> Vec<String> {
        let mut chain = vec![locale.to_string()];
        let mut current = locale.to_string();
        let mut visited = std::collections::HashSet::new();
        visited.insert(current.clone());

        loop {
            // Find explicit rule.
            let fallback = self
                .rules
                .iter()
                .find(|r| r.locale == current)
                .map(|r| r.fallback.clone());

            if let Some(fb) = fallback {
                if visited.contains(&fb) {
                    break;
                }
                visited.insert(fb.clone());
                chain.push(fb.clone());
                current = fb;
            } else {
                // Try language-only fallback (e.g., es_MX -> es).
                if let Some(underscore_pos) = current.find('_') {
                    let language_only = &current[..underscore_pos];
                    let lang_str = language_only.to_string();
                    if !visited.contains(&lang_str) {
                        visited.insert(lang_str.clone());
                        chain.push(lang_str.clone());
                        current = lang_str;
                        continue;
                    }
                }
                break;
            }
        }

        // Ensure default is in the chain.
        if !chain.contains(&self.default_locale) {
            chain.push(self.default_locale.clone());
        }

        chain
    }
}

// ---------------------------------------------------------------------------
// VolumeNormalization
// ---------------------------------------------------------------------------

/// Configuration for voice volume normalization.
#[derive(Debug, Clone)]
pub struct VolumeNormalization {
    /// Target RMS loudness in dBFS.
    pub target_loudness: f32,
    /// Maximum allowed peak amplitude.
    pub max_peak: f32,
    /// Whether normalization is enabled.
    pub enabled: bool,
    /// Per-speaker volume adjustments.
    pub speaker_adjustments: HashMap<String, f32>,
}

impl VolumeNormalization {
    /// Creates default normalization settings.
    pub fn new() -> Self {
        Self {
            target_loudness: -18.0,
            max_peak: -1.0,
            enabled: true,
            speaker_adjustments: HashMap::new(),
        }
    }

    /// Sets a per-speaker volume adjustment (in dB).
    pub fn set_speaker_adjustment(&mut self, speaker: impl Into<String>, adjustment_db: f32) {
        self.speaker_adjustments.insert(speaker.into(), adjustment_db);
    }

    /// Calculates the gain to apply to a voice asset for normalization.
    pub fn calculate_gain(&self, asset: &VoiceAssetInfo, speaker: &str) -> f32 {
        if !self.enabled {
            return 1.0;
        }

        let speaker_adj = self.speaker_adjustments.get(speaker).copied().unwrap_or(0.0);
        let target = self.target_loudness + speaker_adj;
        let diff_db = target - asset.rms_loudness;
        db_to_linear(diff_db)
    }
}

impl Default for VolumeNormalization {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts decibels to linear gain.
fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Converts linear gain to decibels.
fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        return -120.0;
    }
    20.0 * linear.log10()
}

// ---------------------------------------------------------------------------
// CastingNote
// ---------------------------------------------------------------------------

/// Notes for voice directors about a character's voice casting.
#[derive(Debug, Clone)]
pub struct CastingNote {
    /// Character/speaker name.
    pub speaker: String,
    /// The locale this note applies to.
    pub locale: String,
    /// Voice qualities to aim for.
    pub voice_qualities: Vec<String>,
    /// Reference actors or characters for the voice style.
    pub references: Vec<String>,
    /// Age range description.
    pub age_range: Option<String>,
    /// Gender description.
    pub gender: Option<String>,
    /// Accent or dialect notes.
    pub accent_notes: Option<String>,
    /// General direction notes.
    pub direction_notes: Vec<String>,
    /// Lines to use for auditions.
    pub audition_lines: Vec<VoiceLineId>,
    /// Whether casting is finalized.
    pub casting_finalized: bool,
    /// Name of the cast voice actor.
    pub cast_actor: Option<String>,
}

impl CastingNote {
    /// Creates a new casting note.
    pub fn new(speaker: impl Into<String>, locale: impl Into<String>) -> Self {
        Self {
            speaker: speaker.into(),
            locale: locale.into(),
            voice_qualities: Vec::new(),
            references: Vec::new(),
            age_range: None,
            gender: None,
            accent_notes: None,
            direction_notes: Vec::new(),
            audition_lines: Vec::new(),
            casting_finalized: false,
            cast_actor: None,
        }
    }

    /// Adds a voice quality descriptor.
    pub fn with_quality(mut self, quality: impl Into<String>) -> Self {
        self.voice_qualities.push(quality.into());
        self
    }

    /// Adds a reference actor/character.
    pub fn with_reference(mut self, reference: impl Into<String>) -> Self {
        self.references.push(reference.into());
        self
    }

    /// Sets the age range.
    pub fn with_age(mut self, age_range: impl Into<String>) -> Self {
        self.age_range = Some(age_range.into());
        self
    }

    /// Sets the gender.
    pub fn with_gender(mut self, gender: impl Into<String>) -> Self {
        self.gender = Some(gender.into());
        self
    }

    /// Adds a direction note.
    pub fn with_direction(mut self, note: impl Into<String>) -> Self {
        self.direction_notes.push(note.into());
        self
    }

    /// Finalizes casting with the given actor.
    pub fn finalize_casting(&mut self, actor: impl Into<String>) {
        self.cast_actor = Some(actor.into());
        self.casting_finalized = true;
    }
}

// ---------------------------------------------------------------------------
// VoiceLocalizationManager
// ---------------------------------------------------------------------------

/// Central manager for voice localization across all locales.
///
/// Handles voice line registration, locale-specific asset resolution,
/// fallback chain traversal, lip sync data lookup, volume normalization,
/// and casting note management.
pub struct VoiceLocalizationManager {
    /// Voice line metadata registry.
    voice_lines: HashMap<VoiceLineId, VoiceLineMeta>,
    /// Per-locale voice asset banks: (voice_line_id, locale) -> VoiceAssetInfo.
    voice_banks: HashMap<(VoiceLineId, String), VoiceAssetInfo>,
    /// Fallback chain for locale resolution.
    pub fallback_chain: FallbackChain,
    /// Lip sync database: (voice_line_id, locale) -> LipSyncData.
    lip_sync: HashMap<(VoiceLineId, String), LipSyncData>,
    /// Volume normalization settings.
    pub normalization: VolumeNormalization,
    /// Casting notes per (speaker, locale).
    casting_notes: HashMap<(String, String), CastingNote>,
    /// Root directory for voice assets.
    pub voice_assets_root: PathBuf,
    /// Default audio format preference.
    pub preferred_format: AudioFormat,
}

impl VoiceLocalizationManager {
    /// Creates a new voice localization manager.
    pub fn new(default_locale: impl Into<String>, voice_root: impl Into<PathBuf>) -> Self {
        let default = default_locale.into();
        Self {
            voice_lines: HashMap::new(),
            voice_banks: HashMap::new(),
            fallback_chain: FallbackChain::new(&default),
            lip_sync: HashMap::new(),
            normalization: VolumeNormalization::new(),
            casting_notes: HashMap::new(),
            voice_assets_root: voice_root.into(),
            preferred_format: AudioFormat::Ogg,
        }
    }

    /// Registers a voice line.
    pub fn register_voice_line(&mut self, meta: VoiceLineMeta) {
        self.voice_lines.insert(meta.id.clone(), meta);
    }

    /// Adds a voice asset for a specific line and locale.
    pub fn add_voice_asset(
        &mut self,
        voice_line_id: &VoiceLineId,
        locale: impl Into<String>,
        asset: VoiceAssetInfo,
    ) {
        self.voice_banks
            .insert((voice_line_id.clone(), locale.into()), asset);
    }

    /// Adds lip sync data for a voice line in a specific locale.
    pub fn add_lip_sync(&mut self, data: LipSyncData) {
        self.lip_sync.insert(
            (data.voice_line_id.clone(), data.locale.clone()),
            data,
        );
    }

    /// Adds a casting note.
    pub fn add_casting_note(&mut self, note: CastingNote) {
        self.casting_notes.insert(
            (note.speaker.clone(), note.locale.clone()),
            note,
        );
    }

    /// Resolves a voice asset for a given line and locale, traversing
    /// the fallback chain if the primary locale is not available.
    pub fn resolve_voice_asset(
        &self,
        voice_line_id: &VoiceLineId,
        locale: &str,
    ) -> Option<ResolvedVoiceAsset> {
        let chain = self.fallback_chain.resolve(locale);

        for try_locale in &chain {
            let key = (voice_line_id.clone(), try_locale.clone());
            if let Some(asset) = self.voice_banks.get(&key) {
                let is_fallback = try_locale != locale;
                let meta = self.voice_lines.get(voice_line_id);

                let gain = meta
                    .map(|m| self.normalization.calculate_gain(asset, &m.speaker))
                    .unwrap_or(1.0);

                return Some(ResolvedVoiceAsset {
                    voice_line_id: voice_line_id.clone(),
                    requested_locale: locale.to_string(),
                    resolved_locale: try_locale.clone(),
                    asset: asset.clone(),
                    is_fallback,
                    volume_gain: gain,
                });
            }
        }

        None
    }

    /// Resolves lip sync data for a voice line, using fallback chain.
    pub fn resolve_lip_sync(
        &self,
        voice_line_id: &VoiceLineId,
        locale: &str,
    ) -> Option<&LipSyncData> {
        let chain = self.fallback_chain.resolve(locale);
        for try_locale in &chain {
            let key = (voice_line_id.clone(), try_locale.clone());
            if let Some(data) = self.lip_sync.get(&key) {
                return Some(data);
            }
        }
        None
    }

    /// Returns voice line metadata.
    pub fn get_voice_line(&self, id: &VoiceLineId) -> Option<&VoiceLineMeta> {
        self.voice_lines.get(id)
    }

    /// Returns casting notes for a speaker and locale.
    pub fn get_casting_note(&self, speaker: &str, locale: &str) -> Option<&CastingNote> {
        self.casting_notes
            .get(&(speaker.to_string(), locale.to_string()))
    }

    /// Returns a list of all voice lines for a given speaker.
    pub fn lines_for_speaker(&self, speaker: &str) -> Vec<&VoiceLineMeta> {
        self.voice_lines
            .values()
            .filter(|vl| vl.speaker == speaker)
            .collect()
    }

    /// Returns all unique speakers.
    pub fn all_speakers(&self) -> Vec<&str> {
        let mut speakers: Vec<&str> = self
            .voice_lines
            .values()
            .map(|vl| vl.speaker.as_str())
            .collect();
        speakers.sort();
        speakers.dedup();
        speakers
    }

    /// Returns the number of registered voice lines.
    pub fn voice_line_count(&self) -> usize {
        self.voice_lines.len()
    }

    /// Returns the total number of voice assets across all locales.
    pub fn total_asset_count(&self) -> usize {
        self.voice_banks.len()
    }

    /// Generates a coverage report for voice assets.
    pub fn voice_coverage_report(&self, locales: &[&str]) -> VoiceCoverageReport {
        let total_lines = self.voice_lines.len();
        let mut locale_reports = Vec::new();

        for locale in locales {
            let mut present = 0;
            let mut missing_critical = 0;

            for (id, meta) in &self.voice_lines {
                let key = (id.clone(), locale.to_string());
                if self.voice_banks.contains_key(&key) {
                    present += 1;
                } else if meta.critical {
                    missing_critical += 1;
                }
            }

            let coverage = if total_lines > 0 {
                (present as f64 / total_lines as f64) * 100.0
            } else {
                100.0
            };

            locale_reports.push(VoiceLocaleCoverage {
                locale: locale.to_string(),
                total_lines,
                present,
                missing: total_lines - present,
                missing_critical,
                coverage_percentage: coverage,
            });
        }

        VoiceCoverageReport {
            total_voice_lines: total_lines,
            locale_reports,
        }
    }

    /// Validates all voice assets (checks file paths, format consistency).
    pub fn validate_assets(&self) -> Vec<VoiceValidationIssue> {
        let mut issues = Vec::new();

        for ((voice_id, locale), asset) in &self.voice_banks {
            // Check for zero duration.
            if asset.duration <= 0.0 {
                issues.push(VoiceValidationIssue {
                    voice_line_id: voice_id.clone(),
                    locale: locale.clone(),
                    issue: format!("Zero or negative duration: {}", asset.duration),
                    severity: VoiceIssueSeverity::Error,
                });
            }

            // Check for unreasonably loud audio.
            if asset.rms_loudness > -3.0 {
                issues.push(VoiceValidationIssue {
                    voice_line_id: voice_id.clone(),
                    locale: locale.clone(),
                    issue: format!(
                        "Audio may be clipping: RMS loudness {:.1} dBFS",
                        asset.rms_loudness
                    ),
                    severity: VoiceIssueSeverity::Warning,
                });
            }

            // Check for very quiet audio.
            if asset.rms_loudness < -40.0 {
                issues.push(VoiceValidationIssue {
                    voice_line_id: voice_id.clone(),
                    locale: locale.clone(),
                    issue: format!(
                        "Audio is very quiet: RMS loudness {:.1} dBFS",
                        asset.rms_loudness
                    ),
                    severity: VoiceIssueSeverity::Warning,
                });
            }
        }

        issues
    }
}

impl fmt::Debug for VoiceLocalizationManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VoiceLocalizationManager")
            .field("voice_line_count", &self.voice_lines.len())
            .field("voice_bank_count", &self.voice_banks.len())
            .field("lip_sync_count", &self.lip_sync.len())
            .field("casting_note_count", &self.casting_notes.len())
            .finish()
    }
}

/// A resolved voice asset with fallback information.
#[derive(Debug, Clone)]
pub struct ResolvedVoiceAsset {
    /// The voice line ID.
    pub voice_line_id: VoiceLineId,
    /// The locale that was requested.
    pub requested_locale: String,
    /// The locale that was actually resolved (may differ if fallback was used).
    pub resolved_locale: String,
    /// The voice asset info.
    pub asset: VoiceAssetInfo,
    /// Whether a fallback was used.
    pub is_fallback: bool,
    /// Volume gain to apply for normalization.
    pub volume_gain: f32,
}

/// Voice coverage for a single locale.
#[derive(Debug, Clone)]
pub struct VoiceLocaleCoverage {
    /// Locale code.
    pub locale: String,
    /// Total voice lines registered.
    pub total_lines: usize,
    /// Lines with voice assets present.
    pub present: usize,
    /// Lines missing voice assets.
    pub missing: usize,
    /// Critical lines missing voice assets.
    pub missing_critical: usize,
    /// Coverage percentage.
    pub coverage_percentage: f64,
}

/// Voice coverage report across locales.
#[derive(Debug, Clone)]
pub struct VoiceCoverageReport {
    /// Total voice lines.
    pub total_voice_lines: usize,
    /// Per-locale coverage data.
    pub locale_reports: Vec<VoiceLocaleCoverage>,
}

/// A voice asset validation issue.
#[derive(Debug, Clone)]
pub struct VoiceValidationIssue {
    /// Voice line ID.
    pub voice_line_id: VoiceLineId,
    /// Locale.
    pub locale: String,
    /// Description of the issue.
    pub issue: String,
    /// Severity.
    pub severity: VoiceIssueSeverity,
}

/// Severity of a voice validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceIssueSeverity {
    Info,
    Warning,
    Error,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_chain_basic() {
        let mut chain = FallbackChain::new("en_US");
        chain.add_rule("es_MX", "es_ES");
        chain.add_rule("es_ES", "en_US");

        let resolved = chain.resolve("es_MX");
        assert_eq!(resolved, vec!["es_MX", "es_ES", "en_US"]);
    }

    #[test]
    fn test_fallback_chain_language_only() {
        let chain = FallbackChain::new("en_US");
        let resolved = chain.resolve("fr_CA");
        assert_eq!(resolved, vec!["fr_CA", "fr", "en_US"]);
    }

    #[test]
    fn test_voice_asset_resolution() {
        let mut manager = VoiceLocalizationManager::new("en_US", "./voices");

        let meta = VoiceLineMeta::new("greeting_01", "Hero", "Hello there!");
        manager.register_voice_line(meta);

        let line_id = VoiceLineId::new("greeting_01");

        // Add English asset.
        manager.add_voice_asset(
            &line_id,
            "en_US",
            VoiceAssetInfo::new("en_US/greeting_01.ogg", AudioFormat::Ogg, 2.5),
        );

        // Request English: should resolve directly.
        let resolved = manager.resolve_voice_asset(&line_id, "en_US");
        assert!(resolved.is_some());
        assert!(!resolved.unwrap().is_fallback);

        // Request French: should fall back to English.
        let resolved = manager.resolve_voice_asset(&line_id, "fr_FR");
        assert!(resolved.is_some());
        assert!(resolved.unwrap().is_fallback);
    }

    #[test]
    fn test_volume_normalization() {
        let norm = VolumeNormalization::new();
        let asset = VoiceAssetInfo::new("test.ogg", AudioFormat::Ogg, 1.0)
            .with_loudness(-24.0);
        let gain = norm.calculate_gain(&asset, "TestSpeaker");
        // Target is -18, asset is -24, so gain should boost by 6 dB.
        assert!(gain > 1.0);
    }

    #[test]
    fn test_lip_sync_viseme() {
        let mut lip_sync = LipSyncData::new(VoiceLineId::new("test"), "en_US");
        lip_sync.add_viseme(VisemeEvent {
            time: 0.0,
            duration: 0.2,
            viseme: VisemeType::Aa,
            weight: 1.0,
        });
        lip_sync.add_viseme(VisemeEvent {
            time: 0.2,
            duration: 0.15,
            viseme: VisemeType::Ee,
            weight: 0.8,
        });

        let v = lip_sync.viseme_at(0.1);
        assert!(v.is_some());
        assert_eq!(v.unwrap().viseme, VisemeType::Aa);
    }

    #[test]
    fn test_casting_note() {
        let note = CastingNote::new("Hero", "en_US")
            .with_quality("Deep baritone")
            .with_reference("Geralt of Rivia")
            .with_age("35-45")
            .with_gender("Male")
            .with_direction("Stoic, world-weary, occasionally sarcastic");
        assert_eq!(note.voice_qualities.len(), 1);
        assert!(!note.casting_finalized);
    }

    #[test]
    fn test_voice_coverage_report() {
        let mut manager = VoiceLocalizationManager::new("en_US", "./voices");
        let line_id = VoiceLineId::new("test_line");
        manager.register_voice_line(
            VoiceLineMeta::new("test_line", "Hero", "Hello").with_critical(),
        );
        manager.add_voice_asset(
            &line_id,
            "en_US",
            VoiceAssetInfo::new("en_US/test.ogg", AudioFormat::Ogg, 1.0),
        );

        let report = manager.voice_coverage_report(&["en_US", "fr_FR"]);
        assert_eq!(report.total_voice_lines, 1);
        let en = report.locale_reports.iter().find(|r| r.locale == "en_US").unwrap();
        assert_eq!(en.present, 1);
        let fr = report.locale_reports.iter().find(|r| r.locale == "fr_FR").unwrap();
        assert_eq!(fr.missing_critical, 1);
    }

    #[test]
    fn test_db_to_linear() {
        let gain = db_to_linear(6.0);
        assert!((gain - 2.0).abs() < 0.1);
        let gain = db_to_linear(0.0);
        assert!((gain - 1.0).abs() < 0.001);
    }
}
