//! Subtitle Rendering System
//!
//! Provides a complete subtitle system for cinematics and dialogue, including:
//!
//! - Timed subtitles with speaker name and color
//! - Word-by-word reveal effect
//! - Configurable positioning (bottom, top, custom)
//! - Speaker portrait display
//! - Multiple simultaneous subtitles
//! - Localization integration
//! - Style presets for different subtitle types
//!
//! # Architecture
//!
//! ```text
//! SubtitleManager
//!   +-- SubtitleQueue        (pending subtitles)
//!   +-- ActiveSubtitleSlot[] (currently displayed)
//!   +-- SubtitleStyleSheet   (style presets)
//!   +-- WordRevealAnimator   (per-subtitle animation)
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// SubtitleId
// ---------------------------------------------------------------------------

/// Unique identifier for a subtitle entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubtitleId(pub u64);

impl SubtitleId {
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

impl fmt::Display for SubtitleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Subtitle({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// SubtitlePosition
// ---------------------------------------------------------------------------

/// Position where subtitles are displayed on screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SubtitlePosition {
    /// Bottom center of the screen (default for most subtitles).
    BottomCenter,
    /// Top center of the screen.
    TopCenter,
    /// Bottom left.
    BottomLeft,
    /// Bottom right.
    BottomRight,
    /// Top left (useful for narration).
    TopLeft,
    /// Top right.
    TopRight,
    /// Center of the screen.
    Center,
    /// Custom position in normalized screen coordinates (0.0 to 1.0).
    Custom {
        /// Horizontal position (0.0 = left, 1.0 = right).
        x: f32,
        /// Vertical position (0.0 = top, 1.0 = bottom).
        y: f32,
    },
}

impl Default for SubtitlePosition {
    fn default() -> Self {
        Self::BottomCenter
    }
}

impl SubtitlePosition {
    /// Returns the normalized screen coordinates for this position.
    pub fn to_normalized(&self) -> (f32, f32) {
        match self {
            Self::BottomCenter => (0.5, 0.9),
            Self::TopCenter => (0.5, 0.1),
            Self::BottomLeft => (0.2, 0.9),
            Self::BottomRight => (0.8, 0.9),
            Self::TopLeft => (0.2, 0.1),
            Self::TopRight => (0.8, 0.1),
            Self::Center => (0.5, 0.5),
            Self::Custom { x, y } => (*x, *y),
        }
    }
}

// ---------------------------------------------------------------------------
// SubtitleStyle
// ---------------------------------------------------------------------------

/// Visual style for a subtitle.
#[derive(Debug, Clone)]
pub struct SubtitleStyle {
    /// Font family name.
    pub font_family: String,
    /// Font size in logical pixels.
    pub font_size: f32,
    /// Text color (RGBA).
    pub text_color: [f32; 4],
    /// Background panel color (RGBA, [0] = transparent).
    pub background_color: [f32; 4],
    /// Whether to show a background panel behind the text.
    pub show_background: bool,
    /// Background panel padding (in logical pixels).
    pub background_padding: f32,
    /// Background panel corner radius.
    pub background_corner_radius: f32,
    /// Text outline/stroke color.
    pub outline_color: [f32; 4],
    /// Text outline width (0 = no outline).
    pub outline_width: f32,
    /// Text shadow color.
    pub shadow_color: [f32; 4],
    /// Shadow offset X.
    pub shadow_offset_x: f32,
    /// Shadow offset Y.
    pub shadow_offset_y: f32,
    /// Speaker name font size multiplier relative to text.
    pub speaker_name_scale: f32,
    /// Speaker name color (if not overridden per-subtitle).
    pub speaker_name_color: [f32; 4],
    /// Whether to show the speaker name.
    pub show_speaker_name: bool,
    /// Maximum width of the subtitle text as a fraction of screen width.
    pub max_width_fraction: f32,
    /// Text alignment.
    pub text_alignment: SubtitleTextAlignment,
    /// Line spacing multiplier.
    pub line_spacing: f32,
    /// Position on screen.
    pub position: SubtitlePosition,
    /// Margin from screen edges (in logical pixels).
    pub margin: f32,
    /// Fade-in duration (seconds).
    pub fade_in: f32,
    /// Fade-out duration (seconds).
    pub fade_out: f32,
}

/// Text alignment for subtitles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleTextAlignment {
    Left,
    Center,
    Right,
}

impl Default for SubtitleTextAlignment {
    fn default() -> Self {
        Self::Center
    }
}

impl SubtitleStyle {
    /// Creates a default subtitle style.
    pub fn new() -> Self {
        Self {
            font_family: "default".to_string(),
            font_size: 24.0,
            text_color: [1.0, 1.0, 1.0, 1.0],
            background_color: [0.0, 0.0, 0.0, 0.6],
            show_background: true,
            background_padding: 8.0,
            background_corner_radius: 4.0,
            outline_color: [0.0, 0.0, 0.0, 1.0],
            outline_width: 1.5,
            shadow_color: [0.0, 0.0, 0.0, 0.5],
            shadow_offset_x: 1.0,
            shadow_offset_y: 1.0,
            speaker_name_scale: 0.8,
            speaker_name_color: [0.8, 0.8, 0.8, 1.0],
            show_speaker_name: true,
            max_width_fraction: 0.7,
            text_alignment: SubtitleTextAlignment::Center,
            line_spacing: 1.2,
            position: SubtitlePosition::BottomCenter,
            margin: 40.0,
            fade_in: 0.2,
            fade_out: 0.3,
        }
    }

    /// Sets the font size.
    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Sets the text color.
    pub fn with_text_color(mut self, color: [f32; 4]) -> Self {
        self.text_color = color;
        self
    }

    /// Sets the position.
    pub fn with_position(mut self, position: SubtitlePosition) -> Self {
        self.position = position;
        self
    }

    /// Disables the background panel.
    pub fn without_background(mut self) -> Self {
        self.show_background = false;
        self
    }

    /// Sets the outline width.
    pub fn with_outline(mut self, width: f32, color: [f32; 4]) -> Self {
        self.outline_width = width;
        self.outline_color = color;
        self
    }

    /// Hides the speaker name.
    pub fn without_speaker_name(mut self) -> Self {
        self.show_speaker_name = false;
        self
    }
}

impl Default for SubtitleStyle {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SubtitlePreset
// ---------------------------------------------------------------------------

/// Named style presets for different subtitle contexts.
#[derive(Debug, Clone)]
pub struct SubtitlePreset {
    /// Preset name.
    pub name: String,
    /// The style for this preset.
    pub style: SubtitleStyle,
    /// Description of when to use this preset.
    pub description: String,
}

impl SubtitlePreset {
    /// Creates a new preset.
    pub fn new(name: impl Into<String>, style: SubtitleStyle) -> Self {
        Self {
            name: name.into(),
            style,
            description: String::new(),
        }
    }

    /// Standard dialogue preset.
    pub fn dialogue() -> Self {
        Self {
            name: "dialogue".to_string(),
            style: SubtitleStyle::new(),
            description: "Standard dialogue subtitles".to_string(),
        }
    }

    /// Narration preset (top of screen, italic feel).
    pub fn narration() -> Self {
        Self {
            name: "narration".to_string(),
            style: SubtitleStyle::new()
                .with_position(SubtitlePosition::TopCenter)
                .with_text_color([0.9, 0.9, 0.7, 1.0])
                .without_speaker_name(),
            description: "Narration/voiceover subtitles".to_string(),
        }
    }

    /// System/tutorial text preset (center of screen).
    pub fn system() -> Self {
        Self {
            name: "system".to_string(),
            style: SubtitleStyle::new()
                .with_position(SubtitlePosition::Center)
                .with_text_color([1.0, 1.0, 0.5, 1.0])
                .with_font_size(28.0)
                .without_speaker_name()
                .without_background(),
            description: "System/tutorial messages".to_string(),
        }
    }

    /// Thought/internal monologue preset (italic, lighter color).
    pub fn thought() -> Self {
        Self {
            name: "thought".to_string(),
            style: SubtitleStyle::new()
                .with_text_color([0.7, 0.8, 1.0, 0.9]),
            description: "Internal thoughts/monologue".to_string(),
        }
    }

    /// Radio/intercom preset (slightly different styling).
    pub fn radio() -> Self {
        Self {
            name: "radio".to_string(),
            style: SubtitleStyle::new()
                .with_text_color([0.5, 1.0, 0.5, 1.0])
                .with_position(SubtitlePosition::TopRight),
            description: "Radio/intercom communications".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// SpeakerPortrait
// ---------------------------------------------------------------------------

/// Portrait image displayed alongside a subtitle.
#[derive(Debug, Clone)]
pub struct SpeakerPortrait {
    /// Asset ID for the portrait image.
    pub asset_id: String,
    /// Display width in logical pixels.
    pub width: f32,
    /// Display height in logical pixels.
    pub height: f32,
    /// Whether to flip the portrait horizontally.
    pub flip_horizontal: bool,
    /// Border color for the portrait.
    pub border_color: [f32; 4],
    /// Border width.
    pub border_width: f32,
    /// Position relative to the subtitle text.
    pub alignment: PortraitAlignment,
    /// Optional animation (e.g., breathing, talking).
    pub animation: Option<PortraitAnimation>,
}

/// Alignment of the speaker portrait relative to subtitle text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortraitAlignment {
    /// Portrait on the left side.
    Left,
    /// Portrait on the right side.
    Right,
}

impl Default for PortraitAlignment {
    fn default() -> Self {
        Self::Left
    }
}

/// Animation applied to a speaker portrait.
#[derive(Debug, Clone)]
pub enum PortraitAnimation {
    /// No animation (static image).
    None,
    /// Simple breathing animation (subtle scale oscillation).
    Breathing { speed: f32, amplitude: f32 },
    /// Talking animation (alternate between mouth-open/closed frames).
    Talking { open_frame: String, closed_frame: String, speed: f32 },
    /// Emotion-based expression change.
    Expression { expression_asset: String },
}

impl SpeakerPortrait {
    /// Creates a new speaker portrait.
    pub fn new(asset_id: impl Into<String>) -> Self {
        Self {
            asset_id: asset_id.into(),
            width: 128.0,
            height: 128.0,
            flip_horizontal: false,
            border_color: [1.0, 1.0, 1.0, 0.5],
            border_width: 2.0,
            alignment: PortraitAlignment::Left,
            animation: None,
        }
    }

    /// Sets the display size.
    pub fn with_size(mut self, width: f32, height: f32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Sets the alignment.
    pub fn with_alignment(mut self, alignment: PortraitAlignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Sets a breathing animation.
    pub fn with_breathing(mut self, speed: f32, amplitude: f32) -> Self {
        self.animation = Some(PortraitAnimation::Breathing { speed, amplitude });
        self
    }
}

// ---------------------------------------------------------------------------
// SubtitleEntry
// ---------------------------------------------------------------------------

/// A single subtitle to be displayed.
#[derive(Debug, Clone)]
pub struct SubtitleEntry {
    /// Unique identifier.
    pub id: SubtitleId,
    /// The subtitle text.
    pub text: String,
    /// Speaker name (if any).
    pub speaker: Option<String>,
    /// Speaker-specific color override.
    pub speaker_color: Option<[f32; 4]>,
    /// Start time (relative to cutscene or absolute engine time).
    pub start_time: f64,
    /// Duration in seconds.
    pub duration: f64,
    /// Style preset name (looked up in the style sheet).
    pub style_preset: String,
    /// Speaker portrait configuration.
    pub portrait: Option<SpeakerPortrait>,
    /// Localization key for looking up translated text.
    pub localization_key: Option<String>,
    /// Whether to use word-by-word reveal.
    pub word_reveal: bool,
    /// Word reveal speed (words per second, if word_reveal is true).
    pub word_reveal_speed: f32,
    /// Audio asset for the voice line associated with this subtitle.
    pub voice_asset: Option<String>,
    /// Priority (higher priority subtitles may replace lower ones).
    pub priority: i32,
    /// Whether this subtitle should interrupt/replace currently displayed ones.
    pub interrupt: bool,
}

impl SubtitleEntry {
    /// Creates a new subtitle entry.
    pub fn new(text: impl Into<String>, start_time: f64, duration: f64) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            id: SubtitleId(NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
            text: text.into(),
            speaker: None,
            speaker_color: None,
            start_time,
            duration,
            style_preset: "dialogue".to_string(),
            portrait: None,
            localization_key: None,
            word_reveal: false,
            word_reveal_speed: 8.0,
            voice_asset: None,
            priority: 0,
            interrupt: false,
        }
    }

    /// Sets the speaker name.
    pub fn with_speaker(mut self, name: impl Into<String>) -> Self {
        self.speaker = Some(name.into());
        self
    }

    /// Sets the speaker color.
    pub fn with_speaker_color(mut self, color: [f32; 4]) -> Self {
        self.speaker_color = Some(color);
        self
    }

    /// Sets the style preset.
    pub fn with_style(mut self, preset: impl Into<String>) -> Self {
        self.style_preset = preset.into();
        self
    }

    /// Sets the speaker portrait.
    pub fn with_portrait(mut self, portrait: SpeakerPortrait) -> Self {
        self.portrait = Some(portrait);
        self
    }

    /// Sets the localization key.
    pub fn with_localization_key(mut self, key: impl Into<String>) -> Self {
        self.localization_key = Some(key.into());
        self
    }

    /// Enables word-by-word reveal.
    pub fn with_word_reveal(mut self, speed: f32) -> Self {
        self.word_reveal = true;
        self.word_reveal_speed = speed;
        self
    }

    /// Sets the voice asset.
    pub fn with_voice(mut self, asset: impl Into<String>) -> Self {
        self.voice_asset = Some(asset.into());
        self
    }

    /// Returns the end time of this subtitle.
    pub fn end_time(&self) -> f64 {
        self.start_time + self.duration
    }

    /// Returns the word count of the subtitle text.
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

// ---------------------------------------------------------------------------
// WordRevealState
// ---------------------------------------------------------------------------

/// State for word-by-word subtitle reveal animation.
#[derive(Debug, Clone)]
pub struct WordRevealState {
    /// Total number of words.
    pub total_words: usize,
    /// Number of currently visible words.
    pub visible_words: usize,
    /// Words per second.
    pub speed: f32,
    /// Elapsed time since reveal started.
    pub elapsed: f32,
    /// Whether the reveal is complete.
    pub complete: bool,
    /// The individual words.
    pub words: Vec<String>,
}

impl WordRevealState {
    /// Creates a new word reveal state from a text string.
    pub fn new(text: &str, speed: f32) -> Self {
        let words: Vec<String> = text.split_whitespace().map(|w| w.to_string()).collect();
        let total = words.len();
        Self {
            total_words: total,
            visible_words: 0,
            speed,
            elapsed: 0.0,
            complete: total == 0,
            words,
        }
    }

    /// Updates the reveal state.
    pub fn update(&mut self, dt: f32) {
        if self.complete {
            return;
        }
        self.elapsed += dt;
        let target = (self.elapsed * self.speed).floor() as usize;
        self.visible_words = target.min(self.total_words);
        if self.visible_words >= self.total_words {
            self.complete = true;
        }
    }

    /// Skips to reveal all words immediately.
    pub fn skip(&mut self) {
        self.visible_words = self.total_words;
        self.complete = true;
    }

    /// Returns the currently visible text.
    pub fn visible_text(&self) -> String {
        self.words[..self.visible_words].join(" ")
    }

    /// Returns the reveal progress (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        if self.total_words == 0 {
            return 1.0;
        }
        self.visible_words as f32 / self.total_words as f32
    }

    /// Resets the reveal state.
    pub fn reset(&mut self) {
        self.visible_words = 0;
        self.elapsed = 0.0;
        self.complete = self.total_words == 0;
    }
}

// ---------------------------------------------------------------------------
// ActiveSubtitleSlot
// ---------------------------------------------------------------------------

/// Represents a currently displayed subtitle with its rendering state.
#[derive(Debug, Clone)]
pub struct ActiveSubtitleSlot {
    /// The subtitle entry.
    pub entry: SubtitleEntry,
    /// Resolved style (from preset lookup).
    pub style: SubtitleStyle,
    /// Current opacity (for fade-in/out).
    pub opacity: f32,
    /// Word reveal state (if word reveal is enabled).
    pub word_reveal: Option<WordRevealState>,
    /// Time elapsed since this subtitle became active.
    pub elapsed: f64,
    /// Whether this slot is being faded out.
    pub fading_out: bool,
    /// Slot index for vertical stacking.
    pub slot_index: usize,
}

impl ActiveSubtitleSlot {
    /// Creates a new active subtitle slot.
    pub fn new(entry: SubtitleEntry, style: SubtitleStyle, slot_index: usize) -> Self {
        let word_reveal = if entry.word_reveal {
            Some(WordRevealState::new(&entry.text, entry.word_reveal_speed))
        } else {
            None
        };
        Self {
            entry,
            style,
            opacity: 0.0,
            word_reveal,
            elapsed: 0.0,
            fading_out: false,
            slot_index,
        }
    }

    /// Updates the subtitle slot state.
    pub fn update(&mut self, dt: f64) {
        self.elapsed += dt;

        // Update word reveal.
        if let Some(ref mut reveal) = self.word_reveal {
            reveal.update(dt as f32);
        }

        // Update opacity (fade in/out).
        let fade_in = self.style.fade_in;
        let fade_out = self.style.fade_out;
        let duration = self.entry.duration;

        if self.elapsed < fade_in as f64 {
            self.opacity = (self.elapsed as f32 / fade_in).min(1.0);
        } else if self.elapsed > duration - fade_out as f64 {
            let fade_time = (self.elapsed - (duration - fade_out as f64)) as f32;
            self.opacity = (1.0 - fade_time / fade_out).max(0.0);
            self.fading_out = true;
        } else {
            self.opacity = 1.0;
        }
    }

    /// Returns the currently displayed text (accounting for word reveal).
    pub fn displayed_text(&self) -> &str {
        // If word reveal is active, we would return the partial text.
        // For simplicity, return the full text and let the renderer handle
        // the word reveal via the WordRevealState.
        &self.entry.text
    }

    /// Returns `true` if this subtitle has expired.
    pub fn is_expired(&self) -> bool {
        self.elapsed >= self.entry.duration
    }

    /// Returns the screen position for this subtitle, adjusted for stacking.
    pub fn screen_position(&self, screen_height: f32) -> (f32, f32) {
        let (base_x, base_y) = self.style.position.to_normalized();
        let stack_offset = self.slot_index as f32 * (self.style.font_size * self.style.line_spacing + 8.0) / screen_height;
        // Stack upward from the base position for bottom-aligned subtitles.
        let y = if base_y > 0.5 {
            base_y - stack_offset
        } else {
            base_y + stack_offset
        };
        (base_x, y)
    }
}

// ---------------------------------------------------------------------------
// SubtitleManager
// ---------------------------------------------------------------------------

/// Central manager for all subtitle rendering.
///
/// The subtitle manager queues subtitles, resolves styles from presets,
/// manages active display slots, handles word reveal animations, and
/// provides the data needed for rendering.
pub struct SubtitleManager {
    /// Style presets by name.
    presets: HashMap<String, SubtitlePreset>,
    /// Queued subtitles waiting to be displayed.
    queue: Vec<SubtitleEntry>,
    /// Currently active/displayed subtitles.
    active: Vec<ActiveSubtitleSlot>,
    /// Maximum number of simultaneous subtitles.
    pub max_simultaneous: usize,
    /// Global subtitle visibility toggle.
    pub visible: bool,
    /// Global subtitle scale factor.
    pub global_scale: f32,
    /// Current engine time.
    current_time: f64,
    /// Whether subtitles are enabled.
    pub enabled: bool,
    /// Localization callback for resolving localization keys.
    /// In a real engine, this would be a trait object or function pointer.
    localization_locale: Option<String>,
    /// Speaker color overrides.
    speaker_colors: HashMap<String, [f32; 4]>,
    /// Speaker portrait overrides.
    speaker_portraits: HashMap<String, SpeakerPortrait>,
    /// Next slot index for stacking.
    next_slot_index: usize,
}

impl SubtitleManager {
    /// Creates a new subtitle manager with default presets.
    pub fn new() -> Self {
        let mut presets = HashMap::new();
        presets.insert("dialogue".to_string(), SubtitlePreset::dialogue());
        presets.insert("narration".to_string(), SubtitlePreset::narration());
        presets.insert("system".to_string(), SubtitlePreset::system());
        presets.insert("thought".to_string(), SubtitlePreset::thought());
        presets.insert("radio".to_string(), SubtitlePreset::radio());

        Self {
            presets,
            queue: Vec::new(),
            active: Vec::new(),
            max_simultaneous: 3,
            visible: true,
            global_scale: 1.0,
            current_time: 0.0,
            enabled: true,
            localization_locale: None,
            speaker_colors: HashMap::new(),
            speaker_portraits: HashMap::new(),
            next_slot_index: 0,
        }
    }

    /// Registers a style preset.
    pub fn register_preset(&mut self, preset: SubtitlePreset) {
        self.presets.insert(preset.name.clone(), preset);
    }

    /// Removes a style preset.
    pub fn remove_preset(&mut self, name: &str) -> bool {
        self.presets.remove(name).is_some()
    }

    /// Returns a reference to a preset by name.
    pub fn get_preset(&self, name: &str) -> Option<&SubtitlePreset> {
        self.presets.get(name)
    }

    /// Registers a speaker color override.
    pub fn set_speaker_color(&mut self, speaker: impl Into<String>, color: [f32; 4]) {
        self.speaker_colors.insert(speaker.into(), color);
    }

    /// Registers a speaker portrait override.
    pub fn set_speaker_portrait(&mut self, speaker: impl Into<String>, portrait: SpeakerPortrait) {
        self.speaker_portraits.insert(speaker.into(), portrait);
    }

    /// Sets the localization locale.
    pub fn set_locale(&mut self, locale: impl Into<String>) {
        self.localization_locale = Some(locale.into());
    }

    /// Queues a subtitle for display.
    pub fn queue_subtitle(&mut self, entry: SubtitleEntry) {
        self.queue.push(entry);
        self.queue.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());
    }

    /// Immediately displays a subtitle, bypassing the queue.
    pub fn show_immediate(&mut self, entry: SubtitleEntry) {
        if !self.enabled || !self.visible {
            return;
        }

        // Handle interrupt.
        if entry.interrupt {
            self.active.clear();
            self.next_slot_index = 0;
        }

        // Resolve style.
        let style = self.resolve_style(&entry);
        let slot_index = self.next_slot_index;
        self.next_slot_index += 1;

        let slot = ActiveSubtitleSlot::new(entry, style, slot_index);
        self.active.push(slot);

        // Enforce max simultaneous.
        while self.active.len() > self.max_simultaneous {
            self.active.remove(0);
        }
    }

    /// Clears all active and queued subtitles.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.active.clear();
        self.next_slot_index = 0;
    }

    /// Updates the subtitle manager (call once per frame).
    pub fn update(&mut self, dt: f64, current_time: f64) {
        self.current_time = current_time;

        if !self.enabled {
            return;
        }

        // Activate queued subtitles.
        while let Some(entry) = self.queue.first() {
            if entry.start_time <= current_time {
                let entry = self.queue.remove(0);
                self.show_immediate(entry);
            } else {
                break;
            }
        }

        // Update active subtitles.
        for slot in &mut self.active {
            slot.update(dt);
        }

        // Remove expired subtitles.
        self.active.retain(|slot| !slot.is_expired());

        // Reassign slot indices.
        for (i, slot) in self.active.iter_mut().enumerate() {
            slot.slot_index = i;
        }
        self.next_slot_index = self.active.len();
    }

    /// Skips word reveal for all active subtitles.
    pub fn skip_word_reveal(&mut self) {
        for slot in &mut self.active {
            if let Some(ref mut reveal) = slot.word_reveal {
                reveal.skip();
            }
        }
    }

    /// Returns the currently active subtitles for rendering.
    pub fn active_subtitles(&self) -> &[ActiveSubtitleSlot] {
        &self.active
    }

    /// Returns the number of active subtitles.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Returns the number of queued subtitles.
    pub fn queued_count(&self) -> usize {
        self.queue.len()
    }

    /// Returns the number of registered presets.
    pub fn preset_count(&self) -> usize {
        self.presets.len()
    }

    /// Resolves the style for a subtitle entry by looking up its preset
    /// and applying speaker overrides.
    fn resolve_style(&self, entry: &SubtitleEntry) -> SubtitleStyle {
        let base_style = self
            .presets
            .get(&entry.style_preset)
            .map(|p| p.style.clone())
            .unwrap_or_default();

        // Apply global scale.
        let mut style = base_style;
        style.font_size *= self.global_scale;

        // Apply speaker color override.
        if let Some(ref speaker) = entry.speaker {
            if let Some(color) = self.speaker_colors.get(speaker) {
                style.speaker_name_color = *color;
            }
        }

        style
    }
}

impl Default for SubtitleManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SubtitleManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubtitleManager")
            .field("preset_count", &self.presets.len())
            .field("queued", &self.queue.len())
            .field("active", &self.active.len())
            .field("enabled", &self.enabled)
            .field("visible", &self.visible)
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
    fn test_subtitle_entry() {
        let entry = SubtitleEntry::new("Hello, world!", 1.0, 3.0)
            .with_speaker("Narrator")
            .with_speaker_color([1.0, 0.5, 0.0, 1.0]);
        assert_eq!(entry.end_time(), 4.0);
        assert_eq!(entry.word_count(), 2);
    }

    #[test]
    fn test_word_reveal() {
        let mut reveal = WordRevealState::new("Hello beautiful world", 2.0);
        assert_eq!(reveal.total_words, 3);
        assert_eq!(reveal.visible_words, 0);
        reveal.update(0.5);
        assert_eq!(reveal.visible_words, 1);
        reveal.update(0.5);
        assert_eq!(reveal.visible_words, 2);
        assert_eq!(reveal.visible_text(), "Hello beautiful");
        reveal.skip();
        assert!(reveal.complete);
    }

    #[test]
    fn test_subtitle_manager_default_presets() {
        let manager = SubtitleManager::new();
        assert!(manager.get_preset("dialogue").is_some());
        assert!(manager.get_preset("narration").is_some());
        assert!(manager.get_preset("system").is_some());
        assert!(manager.get_preset("thought").is_some());
        assert!(manager.get_preset("radio").is_some());
    }

    #[test]
    fn test_show_immediate() {
        let mut manager = SubtitleManager::new();
        let entry = SubtitleEntry::new("Test subtitle", 0.0, 3.0);
        manager.show_immediate(entry);
        assert_eq!(manager.active_count(), 1);
    }

    #[test]
    fn test_queue_and_activate() {
        let mut manager = SubtitleManager::new();
        let entry = SubtitleEntry::new("Queued subtitle", 1.0, 3.0);
        manager.queue_subtitle(entry);
        assert_eq!(manager.queued_count(), 1);
        assert_eq!(manager.active_count(), 0);

        manager.update(0.0, 1.5);
        assert_eq!(manager.queued_count(), 0);
        assert_eq!(manager.active_count(), 1);
    }

    #[test]
    fn test_subtitle_expiration() {
        let mut manager = SubtitleManager::new();
        let entry = SubtitleEntry::new("Short subtitle", 0.0, 1.0);
        manager.show_immediate(entry);
        assert_eq!(manager.active_count(), 1);

        manager.update(2.0, 2.0);
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_max_simultaneous() {
        let mut manager = SubtitleManager::new();
        manager.max_simultaneous = 2;

        for i in 0..5 {
            let entry = SubtitleEntry::new(format!("Subtitle {}", i), 0.0, 10.0);
            manager.show_immediate(entry);
        }

        assert_eq!(manager.active_count(), 2);
    }

    #[test]
    fn test_subtitle_position() {
        let pos = SubtitlePosition::BottomCenter;
        let (x, y) = pos.to_normalized();
        assert!((x - 0.5).abs() < 0.01);
        assert!(y > 0.5);
    }

    #[test]
    fn test_speaker_color_override() {
        let mut manager = SubtitleManager::new();
        manager.set_speaker_color("Hero", [0.0, 1.0, 0.0, 1.0]);
        let entry = SubtitleEntry::new("Test", 0.0, 3.0).with_speaker("Hero");
        let style = manager.resolve_style(&entry);
        assert_eq!(style.speaker_name_color, [0.0, 1.0, 0.0, 1.0]);
    }
}
