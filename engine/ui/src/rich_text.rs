//! Rich Text Rendering
//!
//! Provides inline formatting, styled text spans, clickable links, inline
//! images, text effects (outline, shadow, gradient), and animated text
//! effects (typewriter reveal, wave, shake).
//!
//! # Markup Syntax
//!
//! Rich text uses an HTML-like tag syntax:
//!
//! - `<b>bold</b>`
//! - `<i>italic</i>`
//! - `<u>underline</u>`
//! - `<s>strikethrough</s>`
//! - `<color=#FF0000>red text</color>`
//! - `<size=24>larger text</size>`
//! - `<link=url>clickable</link>`
//! - `<img=asset_id/>` (inline image)
//! - `<outline color=#000000 width=2>outlined</outline>`
//! - `<shadow color=#000000 offset=2,2>shadowed</shadow>`
//! - `<gradient from=#FF0000 to=#0000FF>gradient text</gradient>`
//! - `<wave amp=4 freq=2>wavy text</wave>`
//! - `<shake amp=2 freq=10>shaky text</shake>`
//! - `<typewriter speed=20>revealed text</typewriter>`

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

/// RGBA color value with components in the range [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RichTextColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl RichTextColor {
    /// Creates a new color from RGBA components.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Pure white.
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);
    /// Pure black.
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    /// Fully transparent.
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    /// Red.
    pub const RED: Self = Self::new(1.0, 0.0, 0.0, 1.0);
    /// Green.
    pub const GREEN: Self = Self::new(0.0, 1.0, 0.0, 1.0);
    /// Blue.
    pub const BLUE: Self = Self::new(0.0, 0.0, 1.0, 1.0);
    /// Yellow.
    pub const YELLOW: Self = Self::new(1.0, 1.0, 0.0, 1.0);

    /// Parses a hex color string like "#FF0000" or "#FF000080".
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        match hex.len() {
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()? as f32 / 255.0;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()? as f32 / 255.0;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()? as f32 / 255.0;
                Some(Self::new(r, g, b, 1.0))
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()? as f32 / 255.0;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()? as f32 / 255.0;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()? as f32 / 255.0;
                let a = u8::from_str_radix(&hex[6..8], 16).ok()? as f32 / 255.0;
                Some(Self::new(r, g, b, a))
            }
            _ => None,
        }
    }

    /// Linearly interpolates between two colors.
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: a.r + (b.r - a.r) * t,
            g: a.g + (b.g - a.g) * t,
            b: a.b + (b.b - a.b) * t,
            a: a.a + (b.a - a.a) * t,
        }
    }

    /// Returns this color with the alpha component replaced.
    pub fn with_alpha(self, alpha: f32) -> Self {
        Self { a: alpha, ..self }
    }

    /// Converts to a hex string like "#RRGGBB" or "#RRGGBBAA".
    pub fn to_hex(&self) -> String {
        let r = (self.r * 255.0).round() as u8;
        let g = (self.g * 255.0).round() as u8;
        let b = (self.b * 255.0).round() as u8;
        let a = (self.a * 255.0).round() as u8;
        if a == 255 {
            format!("#{:02X}{:02X}{:02X}", r, g, b)
        } else {
            format!("#{:02X}{:02X}{:02X}{:02X}", r, g, b, a)
        }
    }
}

impl Default for RichTextColor {
    fn default() -> Self {
        Self::WHITE
    }
}

impl fmt::Display for RichTextColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// ---------------------------------------------------------------------------
// TextStyle
// ---------------------------------------------------------------------------

/// Describes the visual style of a text span.
#[derive(Debug, Clone, PartialEq)]
pub struct TextStyle {
    /// Font family name.
    pub font_family: String,
    /// Font size in logical pixels.
    pub font_size: f32,
    /// Whether the text is bold.
    pub bold: bool,
    /// Whether the text is italic.
    pub italic: bool,
    /// Whether the text has an underline.
    pub underline: bool,
    /// Whether the text has a strikethrough line.
    pub strikethrough: bool,
    /// Text color.
    pub color: RichTextColor,
    /// Outline parameters, if any.
    pub outline: Option<OutlineParams>,
    /// Shadow parameters, if any.
    pub shadow: Option<ShadowParams>,
    /// Gradient parameters, if any.
    pub gradient: Option<GradientParams>,
    /// Letter spacing adjustment in logical pixels.
    pub letter_spacing: f32,
    /// Line height multiplier (1.0 = normal).
    pub line_height: f32,
}

impl TextStyle {
    /// Creates a default text style.
    pub fn new() -> Self {
        Self {
            font_family: "default".to_string(),
            font_size: 16.0,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
            color: RichTextColor::WHITE,
            outline: None,
            shadow: None,
            gradient: None,
            letter_spacing: 0.0,
            line_height: 1.2,
        }
    }

    /// Returns a new style with bold enabled.
    pub fn with_bold(mut self) -> Self {
        self.bold = true;
        self
    }

    /// Returns a new style with italic enabled.
    pub fn with_italic(mut self) -> Self {
        self.italic = true;
        self
    }

    /// Returns a new style with the given font size.
    pub fn with_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Returns a new style with the given color.
    pub fn with_color(mut self, color: RichTextColor) -> Self {
        self.color = color;
        self
    }

    /// Returns a new style with underline enabled.
    pub fn with_underline(mut self) -> Self {
        self.underline = true;
        self
    }

    /// Returns a new style with strikethrough enabled.
    pub fn with_strikethrough(mut self) -> Self {
        self.strikethrough = true;
        self
    }

    /// Returns a new style with outline parameters.
    pub fn with_outline(mut self, outline: OutlineParams) -> Self {
        self.outline = Some(outline);
        self
    }

    /// Returns a new style with shadow parameters.
    pub fn with_shadow(mut self, shadow: ShadowParams) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Returns a new style with gradient parameters.
    pub fn with_gradient(mut self, gradient: GradientParams) -> Self {
        self.gradient = Some(gradient);
        self
    }
}

impl Default for TextStyle {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Outline / Shadow / Gradient params
// ---------------------------------------------------------------------------

/// Parameters for text outline rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct OutlineParams {
    /// Outline color.
    pub color: RichTextColor,
    /// Outline width in logical pixels.
    pub width: f32,
}

impl OutlineParams {
    /// Creates new outline parameters.
    pub fn new(color: RichTextColor, width: f32) -> Self {
        Self { color, width }
    }
}

/// Parameters for text drop shadow rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowParams {
    /// Shadow color.
    pub color: RichTextColor,
    /// Horizontal offset in logical pixels.
    pub offset_x: f32,
    /// Vertical offset in logical pixels.
    pub offset_y: f32,
    /// Blur radius in logical pixels (0 = hard shadow).
    pub blur_radius: f32,
}

impl ShadowParams {
    /// Creates new shadow parameters.
    pub fn new(color: RichTextColor, offset_x: f32, offset_y: f32) -> Self {
        Self {
            color,
            offset_x,
            offset_y,
            blur_radius: 0.0,
        }
    }

    /// Returns new shadow parameters with blur.
    pub fn with_blur(mut self, radius: f32) -> Self {
        self.blur_radius = radius;
        self
    }
}

/// Parameters for text gradient rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientParams {
    /// Starting color.
    pub from: RichTextColor,
    /// Ending color.
    pub to: RichTextColor,
    /// Gradient direction.
    pub direction: GradientDirection,
}

/// Direction of a text gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientDirection {
    /// Left to right.
    Horizontal,
    /// Top to bottom.
    Vertical,
    /// Top-left to bottom-right.
    Diagonal,
}

impl Default for GradientDirection {
    fn default() -> Self {
        Self::Horizontal
    }
}

impl GradientParams {
    /// Creates a horizontal gradient.
    pub fn horizontal(from: RichTextColor, to: RichTextColor) -> Self {
        Self {
            from,
            to,
            direction: GradientDirection::Horizontal,
        }
    }

    /// Creates a vertical gradient.
    pub fn vertical(from: RichTextColor, to: RichTextColor) -> Self {
        Self {
            from,
            to,
            direction: GradientDirection::Vertical,
        }
    }

    /// Returns the interpolated color at position `t` (0.0 to 1.0).
    pub fn sample(&self, t: f32) -> RichTextColor {
        RichTextColor::lerp(self.from, self.to, t)
    }
}

// ---------------------------------------------------------------------------
// RichTextSpan
// ---------------------------------------------------------------------------

/// A contiguous run of text with uniform style.
#[derive(Debug, Clone)]
pub struct RichTextSpan {
    /// The text content of this span.
    pub text: String,
    /// The visual style for this span.
    pub style: TextStyle,
    /// Optional link URL or action ID.
    pub link: Option<String>,
    /// Optional text effect applied to this span.
    pub effect: Option<TextEffect>,
}

impl RichTextSpan {
    /// Creates a new plain text span.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            style: TextStyle::default(),
            link: None,
            effect: None,
        }
    }

    /// Creates a new span with a specific style.
    pub fn styled(text: impl Into<String>, style: TextStyle) -> Self {
        Self {
            text: text.into(),
            style,
            link: None,
            effect: None,
        }
    }

    /// Creates a link span.
    pub fn link(text: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            style: TextStyle::default().with_color(RichTextColor::BLUE).with_underline(),
            link: Some(url.into()),
            effect: None,
        }
    }

    /// Returns the character count of this span.
    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    /// Returns `true` if this span is a link.
    pub fn is_link(&self) -> bool {
        self.link.is_some()
    }

    /// Returns `true` if this span has an animated effect.
    pub fn has_effect(&self) -> bool {
        self.effect.is_some()
    }
}

// ---------------------------------------------------------------------------
// InlineImage
// ---------------------------------------------------------------------------

/// An inline image embedded within rich text.
#[derive(Debug, Clone)]
pub struct InlineImage {
    /// Asset identifier for the image.
    pub asset_id: String,
    /// Display width in logical pixels (0 = use natural width).
    pub width: f32,
    /// Display height in logical pixels (0 = use natural height).
    pub height: f32,
    /// Vertical alignment relative to the text baseline.
    pub alignment: ImageAlignment,
    /// Optional tint color applied to the image.
    pub tint: Option<RichTextColor>,
}

/// Vertical alignment for inline images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageAlignment {
    /// Align the bottom of the image with the text baseline.
    Baseline,
    /// Center the image vertically within the line.
    Middle,
    /// Align the top of the image with the top of the line.
    Top,
    /// Align the bottom of the image with the bottom of the line.
    Bottom,
}

impl Default for ImageAlignment {
    fn default() -> Self {
        Self::Baseline
    }
}

impl InlineImage {
    /// Creates a new inline image with default sizing.
    pub fn new(asset_id: impl Into<String>) -> Self {
        Self {
            asset_id: asset_id.into(),
            width: 0.0,
            height: 0.0,
            alignment: ImageAlignment::Baseline,
            tint: None,
        }
    }

    /// Sets the display size.
    pub fn with_size(mut self, width: f32, height: f32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Sets the vertical alignment.
    pub fn with_alignment(mut self, alignment: ImageAlignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Sets a tint color.
    pub fn with_tint(mut self, tint: RichTextColor) -> Self {
        self.tint = Some(tint);
        self
    }
}

// ---------------------------------------------------------------------------
// TextEffect
// ---------------------------------------------------------------------------

/// Animated text effects that modify character positions or appearance.
#[derive(Debug, Clone)]
pub enum TextEffect {
    /// Typewriter effect: reveal characters one at a time.
    Typewriter(TypewriterEffect),
    /// Wave effect: characters oscillate vertically.
    Wave(WaveEffect),
    /// Shake effect: characters jitter randomly.
    Shake(ShakeEffect),
    /// Fade-in effect: characters fade from transparent to opaque.
    FadeIn(FadeEffect),
    /// Rainbow effect: color cycles through the spectrum.
    Rainbow(RainbowEffect),
    /// Pulse effect: text size oscillates.
    Pulse(PulseEffect),
}

/// Typewriter (character reveal) effect parameters.
#[derive(Debug, Clone)]
pub struct TypewriterEffect {
    /// Characters revealed per second.
    pub speed: f32,
    /// Delay before starting the reveal (in seconds).
    pub initial_delay: f32,
    /// Whether to play a sound for each character.
    pub play_sound: bool,
    /// Sound asset to play per character (if `play_sound` is true).
    pub sound_asset: Option<String>,
    /// Characters that trigger a brief pause (e.g., punctuation).
    pub pause_characters: Vec<char>,
    /// Duration of the pause in seconds.
    pub pause_duration: f32,
}

impl TypewriterEffect {
    /// Creates a new typewriter effect with the given speed.
    pub fn new(characters_per_second: f32) -> Self {
        Self {
            speed: characters_per_second,
            initial_delay: 0.0,
            play_sound: false,
            sound_asset: None,
            pause_characters: vec!['.', '!', '?', ',', ';', ':'],
            pause_duration: 0.15,
        }
    }

    /// Sets the initial delay before starting.
    pub fn with_delay(mut self, delay: f32) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Enables per-character sound playback.
    pub fn with_sound(mut self, asset: impl Into<String>) -> Self {
        self.play_sound = true;
        self.sound_asset = Some(asset.into());
        self
    }

    /// Calculates the number of visible characters at the given elapsed time.
    pub fn visible_chars(&self, elapsed: f32, total_chars: usize) -> usize {
        if elapsed < self.initial_delay {
            return 0;
        }
        let effective_time = elapsed - self.initial_delay;
        let chars = (effective_time * self.speed).floor() as usize;
        chars.min(total_chars)
    }

    /// Returns `true` if the typewriter effect is complete.
    pub fn is_complete(&self, elapsed: f32, total_chars: usize) -> bool {
        self.visible_chars(elapsed, total_chars) >= total_chars
    }
}

/// Wave (vertical oscillation) effect parameters.
#[derive(Debug, Clone)]
pub struct WaveEffect {
    /// Amplitude of the wave in logical pixels.
    pub amplitude: f32,
    /// Frequency of the wave in cycles per second.
    pub frequency: f32,
    /// Phase offset per character (creates a traveling wave).
    pub phase_per_char: f32,
}

impl WaveEffect {
    /// Creates a new wave effect.
    pub fn new(amplitude: f32, frequency: f32) -> Self {
        Self {
            amplitude,
            frequency,
            phase_per_char: 0.3,
        }
    }

    /// Calculates the vertical offset for a character at the given time.
    pub fn offset_y(&self, char_index: usize, time: f32) -> f32 {
        let phase = time * self.frequency * std::f32::consts::TAU
            + char_index as f32 * self.phase_per_char;
        phase.sin() * self.amplitude
    }
}

/// Shake (random jitter) effect parameters.
#[derive(Debug, Clone)]
pub struct ShakeEffect {
    /// Maximum horizontal displacement in logical pixels.
    pub amplitude_x: f32,
    /// Maximum vertical displacement in logical pixels.
    pub amplitude_y: f32,
    /// How often the shake updates (times per second).
    pub frequency: f32,
    /// Seed for the pseudo-random number generator.
    pub seed: u32,
}

impl ShakeEffect {
    /// Creates a new shake effect with uniform amplitude.
    pub fn new(amplitude: f32, frequency: f32) -> Self {
        Self {
            amplitude_x: amplitude,
            amplitude_y: amplitude,
            frequency,
            seed: 42,
        }
    }

    /// Creates a shake effect with separate X and Y amplitudes.
    pub fn with_separate_axes(amplitude_x: f32, amplitude_y: f32, frequency: f32) -> Self {
        Self {
            amplitude_x,
            amplitude_y,
            frequency,
            seed: 42,
        }
    }

    /// Calculates the displacement for a character at the given time.
    pub fn displacement(&self, char_index: usize, time: f32) -> (f32, f32) {
        let frame = (time * self.frequency) as u32;
        let hash = self.hash(char_index as u32, frame);
        let dx = (hash as f32 / u32::MAX as f32 - 0.5) * 2.0 * self.amplitude_x;
        let hash2 = self.hash(char_index as u32 + 1000, frame);
        let dy = (hash2 as f32 / u32::MAX as f32 - 0.5) * 2.0 * self.amplitude_y;
        (dx, dy)
    }

    /// Simple hash function for deterministic pseudo-random displacement.
    fn hash(&self, a: u32, b: u32) -> u32 {
        let mut h = self.seed;
        h = h.wrapping_mul(31).wrapping_add(a);
        h = h.wrapping_mul(31).wrapping_add(b);
        h ^= h >> 16;
        h = h.wrapping_mul(0x45d9f3b);
        h ^= h >> 16;
        h
    }
}

/// Fade-in effect parameters.
#[derive(Debug, Clone)]
pub struct FadeEffect {
    /// Duration of the fade in seconds.
    pub duration: f32,
    /// Delay between each character starting to fade.
    pub stagger: f32,
}

impl FadeEffect {
    /// Creates a new fade effect.
    pub fn new(duration: f32) -> Self {
        Self {
            duration,
            stagger: 0.05,
        }
    }

    /// Calculates the alpha for a character at the given time.
    pub fn alpha(&self, char_index: usize, time: f32) -> f32 {
        let char_start = char_index as f32 * self.stagger;
        let t = (time - char_start) / self.duration;
        t.clamp(0.0, 1.0)
    }
}

/// Rainbow color cycling effect parameters.
#[derive(Debug, Clone)]
pub struct RainbowEffect {
    /// Speed of color cycling (cycles per second).
    pub speed: f32,
    /// Hue offset per character.
    pub hue_per_char: f32,
    /// Saturation (0.0 to 1.0).
    pub saturation: f32,
    /// Value/brightness (0.0 to 1.0).
    pub value: f32,
}

impl RainbowEffect {
    /// Creates a new rainbow effect.
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            hue_per_char: 0.05,
            saturation: 1.0,
            value: 1.0,
        }
    }

    /// Calculates the color for a character at the given time.
    pub fn color(&self, char_index: usize, time: f32) -> RichTextColor {
        let hue = (time * self.speed + char_index as f32 * self.hue_per_char) % 1.0;
        hsv_to_rgb(hue, self.saturation, self.value)
    }
}

/// Pulse (size oscillation) effect parameters.
#[derive(Debug, Clone)]
pub struct PulseEffect {
    /// Minimum scale factor.
    pub min_scale: f32,
    /// Maximum scale factor.
    pub max_scale: f32,
    /// Oscillation speed (cycles per second).
    pub speed: f32,
}

impl PulseEffect {
    /// Creates a new pulse effect.
    pub fn new(min_scale: f32, max_scale: f32, speed: f32) -> Self {
        Self {
            min_scale,
            max_scale,
            speed,
        }
    }

    /// Calculates the scale factor at the given time.
    pub fn scale(&self, time: f32) -> f32 {
        let t = (time * self.speed * std::f32::consts::TAU).sin() * 0.5 + 0.5;
        self.min_scale + (self.max_scale - self.min_scale) * t
    }
}

// ---------------------------------------------------------------------------
// RichTextElement
// ---------------------------------------------------------------------------

/// A single element in a rich text document, either a text span or an
/// inline image.
#[derive(Debug, Clone)]
pub enum RichTextElement {
    /// A styled text span.
    Text(RichTextSpan),
    /// An inline image.
    Image(InlineImage),
    /// A line break.
    LineBreak,
}

// ---------------------------------------------------------------------------
// RichTextDocument
// ---------------------------------------------------------------------------

/// A complete rich text document composed of styled elements.
///
/// The document is the top-level container for rich text content. It holds
/// a sequence of elements (text spans, images, line breaks) and provides
/// methods for building, querying, and rendering the document.
#[derive(Debug, Clone)]
pub struct RichTextDocument {
    /// The elements in this document.
    pub elements: Vec<RichTextElement>,
    /// The default text style applied when no explicit style is set.
    pub default_style: TextStyle,
    /// Maximum width for word-wrapping (0 = no wrapping).
    pub max_width: f32,
    /// Horizontal text alignment.
    pub alignment: TextAlignment,
}

/// Horizontal text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlignment {
    Left,
    Center,
    Right,
    Justify,
}

impl Default for TextAlignment {
    fn default() -> Self {
        Self::Left
    }
}

impl RichTextDocument {
    /// Creates a new empty rich text document.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            default_style: TextStyle::default(),
            max_width: 0.0,
            alignment: TextAlignment::Left,
        }
    }

    /// Creates a document with the given default style.
    pub fn with_default_style(style: TextStyle) -> Self {
        Self {
            elements: Vec::new(),
            default_style: style,
            max_width: 0.0,
            alignment: TextAlignment::Left,
        }
    }

    /// Sets the maximum width for word-wrapping.
    pub fn set_max_width(&mut self, width: f32) {
        self.max_width = width;
    }

    /// Sets the text alignment.
    pub fn set_alignment(&mut self, alignment: TextAlignment) {
        self.alignment = alignment;
    }

    /// Appends a text span.
    pub fn push_text(&mut self, span: RichTextSpan) {
        self.elements.push(RichTextElement::Text(span));
    }

    /// Appends plain text with the default style.
    pub fn push_plain(&mut self, text: impl Into<String>) {
        self.elements
            .push(RichTextElement::Text(RichTextSpan::new(text)));
    }

    /// Appends an inline image.
    pub fn push_image(&mut self, image: InlineImage) {
        self.elements.push(RichTextElement::Image(image));
    }

    /// Appends a line break.
    pub fn push_line_break(&mut self) {
        self.elements.push(RichTextElement::LineBreak);
    }

    /// Returns the total character count across all text spans.
    pub fn total_char_count(&self) -> usize {
        self.elements
            .iter()
            .map(|e| match e {
                RichTextElement::Text(span) => span.char_count(),
                _ => 0,
            })
            .sum()
    }

    /// Returns the number of elements.
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Returns all links in the document with their character ranges.
    pub fn links(&self) -> Vec<LinkInfo> {
        let mut links = Vec::new();
        let mut char_offset = 0;
        for element in &self.elements {
            if let RichTextElement::Text(span) = element {
                if let Some(ref url) = span.link {
                    links.push(LinkInfo {
                        url: url.clone(),
                        text: span.text.clone(),
                        char_start: char_offset,
                        char_end: char_offset + span.char_count(),
                    });
                }
                char_offset += span.char_count();
            }
        }
        links
    }

    /// Clears all elements.
    pub fn clear(&mut self) {
        self.elements.clear();
    }
}

impl Default for RichTextDocument {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a clickable link within a rich text document.
#[derive(Debug, Clone)]
pub struct LinkInfo {
    /// The URL or action ID.
    pub url: String,
    /// The display text of the link.
    pub text: String,
    /// Character index where the link starts.
    pub char_start: usize,
    /// Character index where the link ends (exclusive).
    pub char_end: usize,
}

// ---------------------------------------------------------------------------
// RichTextParser
// ---------------------------------------------------------------------------

/// Parses a markup string into a [`RichTextDocument`].
///
/// The parser processes HTML-like tags and produces a sequence of styled
/// text spans, inline images, and line breaks.
pub struct RichTextParser {
    /// The default style to use as the base.
    default_style: TextStyle,
    /// Custom tag handlers.
    custom_tags: HashMap<String, CustomTagHandler>,
}

/// A custom tag handler function.
pub struct CustomTagHandler {
    /// Name of the custom tag.
    pub tag_name: String,
    /// Handler function that takes tag attributes and inner text.
    handler_fn: Box<dyn Fn(&HashMap<String, String>, &str) -> Vec<RichTextElement> + Send + Sync>,
}

impl fmt::Debug for CustomTagHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomTagHandler")
            .field("tag_name", &self.tag_name)
            .finish()
    }
}

impl RichTextParser {
    /// Creates a new parser with the default text style.
    pub fn new() -> Self {
        Self {
            default_style: TextStyle::default(),
            custom_tags: HashMap::new(),
        }
    }

    /// Creates a new parser with the given default text style.
    pub fn with_default_style(style: TextStyle) -> Self {
        Self {
            default_style: style,
            custom_tags: HashMap::new(),
        }
    }

    /// Registers a custom tag handler.
    pub fn register_custom_tag<F>(
        &mut self,
        tag_name: impl Into<String>,
        handler: F,
    ) where
        F: Fn(&HashMap<String, String>, &str) -> Vec<RichTextElement> + Send + Sync + 'static,
    {
        let name = tag_name.into();
        self.custom_tags.insert(
            name.clone(),
            CustomTagHandler {
                tag_name: name,
                handler_fn: Box::new(handler),
            },
        );
    }

    /// Parses a markup string into a rich text document.
    pub fn parse(&self, markup: &str) -> RichTextDocument {
        let mut doc = RichTextDocument::with_default_style(self.default_style.clone());
        let mut style_stack: Vec<TextStyle> = vec![self.default_style.clone()];
        let mut link_stack: Vec<Option<String>> = vec![None];
        let mut effect_stack: Vec<Option<TextEffect>> = vec![None];

        let mut chars = markup.chars().peekable();
        let mut current_text = String::new();

        while let Some(ch) = chars.next() {
            if ch == '<' {
                // Flush any accumulated plain text.
                if !current_text.is_empty() {
                    let style = style_stack.last().cloned().unwrap_or_default();
                    let link = link_stack.last().cloned().unwrap_or(None);
                    let effect = effect_stack.last().cloned().unwrap_or(None);
                    doc.push_text(RichTextSpan {
                        text: std::mem::take(&mut current_text),
                        style,
                        link,
                        effect,
                    });
                }

                // Read the tag.
                let mut tag_content = String::new();
                let mut found_close = false;
                for tag_ch in chars.by_ref() {
                    if tag_ch == '>' {
                        found_close = true;
                        break;
                    }
                    tag_content.push(tag_ch);
                }

                if !found_close {
                    // Malformed tag, treat as literal text.
                    current_text.push('<');
                    current_text.push_str(&tag_content);
                    continue;
                }

                let is_closing = tag_content.starts_with('/');
                let is_self_closing = tag_content.ends_with('/');
                let tag_body = if is_closing {
                    &tag_content[1..]
                } else if is_self_closing {
                    &tag_content[..tag_content.len() - 1]
                } else {
                    &tag_content
                };

                let (tag_name, attrs) = parse_tag_attributes(tag_body);

                if is_closing {
                    // Pop style stack.
                    if style_stack.len() > 1 {
                        style_stack.pop();
                    }
                    if link_stack.len() > 1 {
                        link_stack.pop();
                    }
                    if effect_stack.len() > 1 {
                        effect_stack.pop();
                    }
                } else if is_self_closing {
                    // Handle self-closing tags (e.g., <img=asset_id/>).
                    match tag_name.to_lowercase().as_str() {
                        "img" => {
                            let asset_id = attrs
                                .get("img")
                                .or_else(|| attrs.get("src"))
                                .cloned()
                                .unwrap_or_default();
                            let mut image = InlineImage::new(asset_id);
                            if let Some(w) = attrs.get("width") {
                                image.width = w.parse().unwrap_or(0.0);
                            }
                            if let Some(h) = attrs.get("height") {
                                image.height = h.parse().unwrap_or(0.0);
                            }
                            doc.push_image(image);
                        }
                        "br" => {
                            doc.push_line_break();
                        }
                        _ => {
                            // Check custom tags.
                            if let Some(handler) = self.custom_tags.get(&tag_name.to_lowercase()) {
                                let elements = (handler.handler_fn)(&attrs, "");
                                for element in elements {
                                    doc.elements.push(element);
                                }
                            }
                        }
                    }
                } else {
                    // Opening tag -- push new style.
                    let mut new_style = style_stack.last().cloned().unwrap_or_default();
                    let mut new_link = link_stack.last().cloned().unwrap_or(None);
                    let mut new_effect = effect_stack.last().cloned().unwrap_or(None);

                    match tag_name.to_lowercase().as_str() {
                        "b" | "bold" | "strong" => {
                            new_style.bold = true;
                        }
                        "i" | "italic" | "em" => {
                            new_style.italic = true;
                        }
                        "u" | "underline" => {
                            new_style.underline = true;
                        }
                        "s" | "strikethrough" | "strike" => {
                            new_style.strikethrough = true;
                        }
                        "color" | "colour" => {
                            if let Some(color_str) = attrs.get("color")
                                .or_else(|| attrs.get("colour"))
                                .or_else(|| attrs.get(&tag_name.to_lowercase()))
                            {
                                if let Some(color) = RichTextColor::from_hex(color_str) {
                                    new_style.color = color;
                                }
                            }
                        }
                        "size" => {
                            if let Some(size_str) = attrs.get("size")
                                .or_else(|| attrs.get(&tag_name.to_lowercase()))
                            {
                                if let Ok(size) = size_str.parse::<f32>() {
                                    new_style.font_size = size;
                                }
                            }
                        }
                        "link" | "a" => {
                            let url = attrs
                                .get("link")
                                .or_else(|| attrs.get("href"))
                                .or_else(|| attrs.get("a"))
                                .cloned()
                                .unwrap_or_default();
                            new_link = Some(url);
                            new_style.color = RichTextColor::BLUE;
                            new_style.underline = true;
                        }
                        "outline" => {
                            let color = attrs
                                .get("color")
                                .and_then(|s| RichTextColor::from_hex(s))
                                .unwrap_or(RichTextColor::BLACK);
                            let width = attrs
                                .get("width")
                                .and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(1.0);
                            new_style.outline = Some(OutlineParams::new(color, width));
                        }
                        "shadow" => {
                            let color = attrs
                                .get("color")
                                .and_then(|s| RichTextColor::from_hex(s))
                                .unwrap_or(RichTextColor::BLACK);
                            let offset = attrs
                                .get("offset")
                                .map(|s| parse_offset(s))
                                .unwrap_or((2.0, 2.0));
                            new_style.shadow =
                                Some(ShadowParams::new(color, offset.0, offset.1));
                        }
                        "gradient" => {
                            let from = attrs
                                .get("from")
                                .and_then(|s| RichTextColor::from_hex(s))
                                .unwrap_or(RichTextColor::RED);
                            let to = attrs
                                .get("to")
                                .and_then(|s| RichTextColor::from_hex(s))
                                .unwrap_or(RichTextColor::BLUE);
                            new_style.gradient =
                                Some(GradientParams::horizontal(from, to));
                        }
                        "wave" => {
                            let amp = attrs
                                .get("amp")
                                .and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(4.0);
                            let freq = attrs
                                .get("freq")
                                .and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(2.0);
                            new_effect = Some(TextEffect::Wave(WaveEffect::new(amp, freq)));
                        }
                        "shake" => {
                            let amp = attrs
                                .get("amp")
                                .and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(2.0);
                            let freq = attrs
                                .get("freq")
                                .and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(10.0);
                            new_effect = Some(TextEffect::Shake(ShakeEffect::new(amp, freq)));
                        }
                        "typewriter" => {
                            let speed = attrs
                                .get("speed")
                                .and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(20.0);
                            new_effect =
                                Some(TextEffect::Typewriter(TypewriterEffect::new(speed)));
                        }
                        "rainbow" => {
                            let speed = attrs
                                .get("speed")
                                .and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(1.0);
                            new_effect =
                                Some(TextEffect::Rainbow(RainbowEffect::new(speed)));
                        }
                        "pulse" => {
                            let min_scale = attrs
                                .get("min")
                                .and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(0.8);
                            let max_scale = attrs
                                .get("max")
                                .and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(1.2);
                            let speed = attrs
                                .get("speed")
                                .and_then(|s| s.parse::<f32>().ok())
                                .unwrap_or(2.0);
                            new_effect = Some(TextEffect::Pulse(PulseEffect::new(
                                min_scale, max_scale, speed,
                            )));
                        }
                        _ => {
                            // Check custom tags.
                            if let Some(handler) = self.custom_tags.get(&tag_name.to_lowercase()) {
                                // For custom tags with content, we would need a more
                                // sophisticated parser. For now, the content is handled
                                // inline.
                                let _ = handler;
                            }
                        }
                    }

                    style_stack.push(new_style);
                    link_stack.push(new_link);
                    effect_stack.push(new_effect);
                }
            } else if ch == '\n' {
                // Flush text before the line break.
                if !current_text.is_empty() {
                    let style = style_stack.last().cloned().unwrap_or_default();
                    let link = link_stack.last().cloned().unwrap_or(None);
                    let effect = effect_stack.last().cloned().unwrap_or(None);
                    doc.push_text(RichTextSpan {
                        text: std::mem::take(&mut current_text),
                        style,
                        link,
                        effect,
                    });
                }
                doc.push_line_break();
            } else {
                current_text.push(ch);
            }
        }

        // Flush remaining text.
        if !current_text.is_empty() {
            let style = style_stack.last().cloned().unwrap_or_default();
            let link = link_stack.last().cloned().unwrap_or(None);
            let effect = effect_stack.last().cloned().unwrap_or(None);
            doc.push_text(RichTextSpan {
                text: current_text,
                style,
                link,
                effect,
            });
        }

        doc
    }
}

impl Default for RichTextParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TypewriterAnimator
// ---------------------------------------------------------------------------

/// Manages the state of a typewriter animation applied to a rich text
/// document.
#[derive(Debug, Clone)]
pub struct TypewriterAnimator {
    /// Total number of characters in the document.
    pub total_chars: usize,
    /// Characters per second.
    pub speed: f32,
    /// Elapsed time in seconds.
    pub elapsed: f32,
    /// Whether the animation is paused.
    pub paused: bool,
    /// Whether the animation has completed.
    pub completed: bool,
    /// Pause accumulator (for punctuation pauses).
    pub pause_timer: f32,
    /// Characters that trigger pauses.
    pub pause_chars: Vec<char>,
    /// Duration of each pause in seconds.
    pub pause_duration: f32,
    /// Current visible character count.
    pub visible_chars: usize,
}

impl TypewriterAnimator {
    /// Creates a new typewriter animator.
    pub fn new(total_chars: usize, speed: f32) -> Self {
        Self {
            total_chars,
            speed,
            elapsed: 0.0,
            paused: false,
            completed: false,
            pause_timer: 0.0,
            pause_chars: vec!['.', '!', '?', ','],
            pause_duration: 0.15,
            visible_chars: 0,
        }
    }

    /// Advances the animation by the given delta time.
    pub fn update(&mut self, dt: f32) {
        if self.completed || self.paused {
            return;
        }

        if self.pause_timer > 0.0 {
            self.pause_timer -= dt;
            return;
        }

        self.elapsed += dt;
        let new_visible = (self.elapsed * self.speed).floor() as usize;
        let new_visible = new_visible.min(self.total_chars);

        if new_visible > self.visible_chars {
            self.visible_chars = new_visible;
            if self.visible_chars >= self.total_chars {
                self.completed = true;
            }
        }
    }

    /// Skips the animation, revealing all characters immediately.
    pub fn skip(&mut self) {
        self.visible_chars = self.total_chars;
        self.completed = true;
    }

    /// Pauses or resumes the animation.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Resets the animation to the beginning.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.visible_chars = 0;
        self.completed = false;
        self.paused = false;
        self.pause_timer = 0.0;
    }

    /// Returns the fraction of the animation that has completed (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        if self.total_chars == 0 {
            return 1.0;
        }
        self.visible_chars as f32 / self.total_chars as f32
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Parses tag attributes from a tag body string.
///
/// Handles both `tag_name=value` (for the primary attribute) and
/// `key=value` pairs. Values may be quoted or unquoted.
fn parse_tag_attributes(tag_body: &str) -> (String, HashMap<String, String>) {
    let mut attrs = HashMap::new();
    let parts: Vec<&str> = tag_body.split_whitespace().collect();

    if parts.is_empty() {
        return (String::new(), attrs);
    }

    // The first part may be "tagname" or "tagname=value".
    let first = parts[0];
    let (tag_name, primary_value) = if let Some(eq_pos) = first.find('=') {
        let name = &first[..eq_pos];
        let value = first[eq_pos + 1..].trim_matches('"').trim_matches('\'');
        (name.to_string(), Some(value.to_string()))
    } else {
        (first.to_string(), None)
    };

    if let Some(value) = primary_value {
        attrs.insert(tag_name.to_lowercase(), value);
    }

    // Parse remaining key=value pairs.
    for part in &parts[1..] {
        if let Some(eq_pos) = part.find('=') {
            let key = &part[..eq_pos];
            let value = part[eq_pos + 1..].trim_matches('"').trim_matches('\'');
            attrs.insert(key.to_lowercase(), value.to_string());
        }
    }

    (tag_name, attrs)
}

/// Parses an offset string like "2,2" into (x, y).
fn parse_offset(s: &str) -> (f32, f32) {
    let parts: Vec<&str> = s.split(',').collect();
    let x = parts.first().and_then(|p| p.trim().parse().ok()).unwrap_or(0.0);
    let y = parts.get(1).and_then(|p| p.trim().parse().ok()).unwrap_or(x);
    (x, y)
}

/// Converts HSV to RGB color.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> RichTextColor {
    let h = h.fract();
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 1.0 / 6.0 {
        (c, x, 0.0)
    } else if h < 2.0 / 6.0 {
        (x, c, 0.0)
    } else if h < 3.0 / 6.0 {
        (0.0, c, x)
    } else if h < 4.0 / 6.0 {
        (0.0, x, c)
    } else if h < 5.0 / 6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    RichTextColor::new(r + m, g + m, b + m, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex() {
        let color = RichTextColor::from_hex("#FF0000").unwrap();
        assert!((color.r - 1.0).abs() < 0.01);
        assert!(color.g.abs() < 0.01);
        assert!(color.b.abs() < 0.01);
    }

    #[test]
    fn test_color_lerp() {
        let a = RichTextColor::RED;
        let b = RichTextColor::BLUE;
        let mid = RichTextColor::lerp(a, b, 0.5);
        assert!((mid.r - 0.5).abs() < 0.01);
        assert!((mid.b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_plain_text() {
        let parser = RichTextParser::new();
        let doc = parser.parse("Hello, world!");
        assert_eq!(doc.element_count(), 1);
        assert_eq!(doc.total_char_count(), 13);
    }

    #[test]
    fn test_parse_bold() {
        let parser = RichTextParser::new();
        let doc = parser.parse("Hello <b>bold</b> world");
        assert_eq!(doc.element_count(), 3);
    }

    #[test]
    fn test_parse_color() {
        let parser = RichTextParser::new();
        let doc = parser.parse("Normal <color=#FF0000>red</color> text");
        assert_eq!(doc.element_count(), 3);
    }

    #[test]
    fn test_typewriter_effect() {
        let effect = TypewriterEffect::new(10.0);
        assert_eq!(effect.visible_chars(0.0, 20), 0);
        assert_eq!(effect.visible_chars(1.0, 20), 10);
        assert_eq!(effect.visible_chars(3.0, 20), 20);
    }

    #[test]
    fn test_wave_effect() {
        let effect = WaveEffect::new(4.0, 2.0);
        let y = effect.offset_y(0, 0.0);
        assert!(y.abs() <= 4.0);
    }

    #[test]
    fn test_typewriter_animator() {
        let mut animator = TypewriterAnimator::new(100, 50.0);
        animator.update(1.0);
        assert_eq!(animator.visible_chars, 50);
        assert!(!animator.completed);
        animator.update(1.0);
        assert!(animator.completed);
    }

    #[test]
    fn test_inline_image() {
        let parser = RichTextParser::new();
        let doc = parser.parse("Text <img=icon_health/> more text");
        assert_eq!(doc.element_count(), 3);
    }

    #[test]
    fn test_link_extraction() {
        let parser = RichTextParser::new();
        let doc = parser.parse("Click <link=https://example.com>here</link> please");
        let links = doc.links();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].url, "https://example.com");
    }
}
