// engine/render/src/text_renderer.rs
//
// GPU text rendering system for the Genovo engine.
//
// Implements high-quality text rendering using signed-distance-field (SDF) fonts:
//
// - **SDF font rendering** — Single-channel SDF glyph textures allow sharp text
//   at any scale with minimal GPU cost.
// - **Glyph atlas management** — Dynamic packing of glyphs into an atlas texture
//   with LRU eviction.
// - **Text batching** — Groups text draws by font/atlas to minimise draw calls.
// - **Text alignment** — Left, centre, right, and justified horizontal alignment.
// - **Word wrapping** — Automatic line breaking at word boundaries or manual line
//   breaks.
// - **Rich text** — Per-character styling: bold, italic, colour, size overrides.
// - **Text outline** — Configurable outline width and colour using the SDF.
// - **Text shadow** — Offset drop shadow with configurable colour and softness.
// - **3D text** — Billboarded text in world space with configurable facing.
//
// # Pipeline integration
//
// Text is rendered as quads (two triangles per glyph) using an alpha-test or
// smoothstep on the SDF value in the fragment shader.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Glyph and font data
// ---------------------------------------------------------------------------

/// Metrics for a single glyph.
#[derive(Debug, Clone, Copy)]
pub struct GlyphMetrics {
    /// Unicode codepoint.
    pub codepoint: char,
    /// Advance width (horizontal distance to the next glyph, in font units).
    pub advance: f32,
    /// Horizontal bearing (offset from the pen position to the left edge).
    pub bearing_x: f32,
    /// Vertical bearing (offset from the baseline to the top edge).
    pub bearing_y: f32,
    /// Glyph bounding box width (in font units).
    pub width: f32,
    /// Glyph bounding box height (in font units).
    pub height: f32,
}

/// UV rect within the glyph atlas.
#[derive(Debug, Clone, Copy)]
pub struct GlyphAtlasRect {
    /// Top-left U.
    pub u: f32,
    /// Top-left V.
    pub v: f32,
    /// Width in UV space.
    pub w: f32,
    /// Height in UV space.
    pub h: f32,
}

impl GlyphAtlasRect {
    pub fn new(u: f32, v: f32, w: f32, h: f32) -> Self {
        Self { u, v, w, h }
    }
}

/// An entry in the glyph atlas.
#[derive(Debug, Clone, Copy)]
pub struct GlyphEntry {
    /// Glyph metrics.
    pub metrics: GlyphMetrics,
    /// Position in the atlas.
    pub atlas_rect: GlyphAtlasRect,
    /// Font size this glyph was rasterised at.
    pub font_size: f32,
    /// SDF spread (in pixels in the atlas).
    pub sdf_spread: f32,
    /// Last frame this glyph was used (for LRU eviction).
    pub last_used_frame: u64,
}

/// A font with SDF glyph data.
#[derive(Debug, Clone)]
pub struct SdfFont {
    /// Font name/identifier.
    pub name: String,
    /// Glyph entries indexed by codepoint.
    pub glyphs: HashMap<char, GlyphEntry>,
    /// Font-wide line height (distance between baselines, in font units).
    pub line_height: f32,
    /// Ascender (above baseline).
    pub ascender: f32,
    /// Descender (below baseline, typically negative).
    pub descender: f32,
    /// Default font size for this SDF font.
    pub base_size: f32,
    /// SDF spread used during atlas generation.
    pub sdf_spread: f32,
    /// Kerning pairs: (left_char, right_char) → horizontal adjustment.
    pub kerning: HashMap<(char, char), f32>,
    /// Atlas texture handle.
    pub atlas_handle: u64,
    /// Atlas texture dimensions (width, height).
    pub atlas_size: (u32, u32),
    /// Whether bold variant is available.
    pub has_bold: bool,
    /// Whether italic variant is available.
    pub has_italic: bool,
}

impl SdfFont {
    /// Create a new SDF font.
    pub fn new(name: impl Into<String>, base_size: f32, atlas_handle: u64, atlas_size: (u32, u32)) -> Self {
        Self {
            name: name.into(),
            glyphs: HashMap::new(),
            line_height: base_size * 1.2,
            ascender: base_size * 0.8,
            descender: base_size * -0.2,
            base_size,
            sdf_spread: 4.0,
            kerning: HashMap::new(),
            atlas_handle,
            atlas_size,
            has_bold: false,
            has_italic: false,
        }
    }

    /// Register a glyph.
    pub fn add_glyph(&mut self, entry: GlyphEntry) {
        self.glyphs.insert(entry.metrics.codepoint, entry);
    }

    /// Register a kerning pair.
    pub fn add_kerning(&mut self, left: char, right: char, amount: f32) {
        self.kerning.insert((left, right), amount);
    }

    /// Look up a glyph by codepoint.
    pub fn glyph(&self, ch: char) -> Option<&GlyphEntry> {
        self.glyphs.get(&ch)
    }

    /// Get kerning between two characters.
    pub fn get_kerning(&self, left: char, right: char) -> f32 {
        self.kerning.get(&(left, right)).copied().unwrap_or(0.0)
    }

    /// Compute the width of a text string at a given font size.
    pub fn measure_text(&self, text: &str, font_size: f32) -> f32 {
        let scale = font_size / self.base_size;
        let mut width = 0.0_f32;
        let mut prev_char: Option<char> = None;

        for ch in text.chars() {
            if let Some(prev) = prev_char {
                width += self.get_kerning(prev, ch) * scale;
            }

            if let Some(glyph) = self.glyph(ch) {
                width += glyph.metrics.advance * scale;
            }

            prev_char = Some(ch);
        }

        width
    }

    /// Compute the width and height of wrapped text.
    pub fn measure_text_wrapped(&self, text: &str, font_size: f32, max_width: f32) -> (f32, f32) {
        let lines = word_wrap(text, self, font_size, max_width);
        let scale = font_size / self.base_size;
        let line_h = self.line_height * scale;

        let mut max_w = 0.0_f32;
        for line in &lines {
            let w = self.measure_text(line, font_size);
            max_w = max_w.max(w);
        }

        (max_w, lines.len() as f32 * line_h)
    }
}

// ---------------------------------------------------------------------------
// Glyph atlas packing
// ---------------------------------------------------------------------------

/// Simple row-based atlas packer.
#[derive(Debug)]
pub struct GlyphAtlasPacker {
    /// Atlas width in pixels.
    pub width: u32,
    /// Atlas height in pixels.
    pub height: u32,
    /// Current X cursor.
    cursor_x: u32,
    /// Current Y cursor.
    cursor_y: u32,
    /// Current row height.
    row_height: u32,
    /// Padding between glyphs.
    pub padding: u32,
    /// Number of glyphs packed.
    pub glyph_count: u32,
}

impl GlyphAtlasPacker {
    /// Create a new packer.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            padding: 2,
            glyph_count: 0,
        }
    }

    /// Try to pack a glyph of the given size. Returns the position if successful.
    pub fn pack(&mut self, glyph_w: u32, glyph_h: u32) -> Option<(u32, u32)> {
        let padded_w = glyph_w + self.padding;
        let padded_h = glyph_h + self.padding;

        // Check if the glyph fits in the current row.
        if self.cursor_x + padded_w > self.width {
            // Move to next row.
            self.cursor_x = 0;
            self.cursor_y += self.row_height;
            self.row_height = 0;
        }

        // Check if it fits vertically.
        if self.cursor_y + padded_h > self.height {
            return None; // Atlas full.
        }

        let pos = (self.cursor_x, self.cursor_y);
        self.cursor_x += padded_w;
        self.row_height = self.row_height.max(padded_h);
        self.glyph_count += 1;

        Some(pos)
    }

    /// Reset the packer.
    pub fn reset(&mut self) {
        self.cursor_x = 0;
        self.cursor_y = 0;
        self.row_height = 0;
        self.glyph_count = 0;
    }

    /// Compute the utilisation ratio.
    pub fn utilisation(&self) -> f32 {
        let used = self.cursor_y * self.width + self.cursor_x * self.row_height;
        used as f32 / (self.width * self.height) as f32
    }
}

// ---------------------------------------------------------------------------
// Text alignment and wrapping
// ---------------------------------------------------------------------------

/// Horizontal text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
    Justify,
}

/// Vertical text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextVAlign {
    Top,
    Middle,
    Bottom,
    Baseline,
}

/// Word-wrap text into lines that fit within `max_width`.
pub fn word_wrap(text: &str, font: &SdfFont, font_size: f32, max_width: f32) -> Vec<String> {
    let mut lines = Vec::new();
    let scale = font_size / font.base_size;

    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            lines.push(String::new());
            continue;
        }

        let words: Vec<&str> = paragraph.split_whitespace().collect();
        if words.is_empty() {
            lines.push(String::new());
            continue;
        }

        let mut current_line = String::new();
        let mut current_width = 0.0_f32;

        let space_width = font.glyph(' ').map(|g| g.metrics.advance * scale).unwrap_or(font_size * 0.25);

        for word in &words {
            let word_width = font.measure_text(word, font_size);

            if current_line.is_empty() {
                // First word on the line.
                current_line.push_str(word);
                current_width = word_width;
            } else if current_width + space_width + word_width <= max_width {
                // Word fits on the current line.
                current_line.push(' ');
                current_line.push_str(word);
                current_width += space_width + word_width;
            } else {
                // Word doesn't fit; start a new line.
                lines.push(current_line);
                current_line = word.to_string();
                current_width = word_width;
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

// ---------------------------------------------------------------------------
// Rich text
// ---------------------------------------------------------------------------

/// Style for a span of rich text.
#[derive(Debug, Clone)]
pub struct RichTextStyle {
    /// Font size override (None = use default).
    pub font_size: Option<f32>,
    /// Colour override (linear RGBA).
    pub color: Option<[f32; 4]>,
    /// Bold flag.
    pub bold: bool,
    /// Italic flag.
    pub italic: bool,
    /// Underline flag.
    pub underline: bool,
    /// Strikethrough flag.
    pub strikethrough: bool,
    /// Outline override.
    pub outline: Option<TextOutline>,
    /// Shadow override.
    pub shadow: Option<TextShadow>,
    /// Additional letter spacing.
    pub letter_spacing: f32,
}

impl Default for RichTextStyle {
    fn default() -> Self {
        Self {
            font_size: None,
            color: None,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
            outline: None,
            shadow: None,
            letter_spacing: 0.0,
        }
    }
}

/// A span of rich text with a consistent style.
#[derive(Debug, Clone)]
pub struct RichTextSpan {
    /// The text content.
    pub text: String,
    /// Style for this span.
    pub style: RichTextStyle,
}

/// Rich text: multiple styled spans.
#[derive(Debug, Clone)]
pub struct RichText {
    /// Spans of styled text.
    pub spans: Vec<RichTextSpan>,
}

impl RichText {
    /// Create rich text from a single plain string.
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            spans: vec![RichTextSpan {
                text: text.into(),
                style: RichTextStyle::default(),
            }],
        }
    }

    /// Add a styled span.
    pub fn add_span(mut self, text: impl Into<String>, style: RichTextStyle) -> Self {
        self.spans.push(RichTextSpan { text: text.into(), style });
        self
    }

    /// Get the total text content (all spans concatenated).
    pub fn plain_text(&self) -> String {
        self.spans.iter().map(|s| s.text.as_str()).collect()
    }

    /// Get the total character count.
    pub fn char_count(&self) -> usize {
        self.spans.iter().map(|s| s.text.chars().count()).sum()
    }
}

// ---------------------------------------------------------------------------
// Text outline and shadow
// ---------------------------------------------------------------------------

/// Text outline configuration.
#[derive(Debug, Clone, Copy)]
pub struct TextOutline {
    /// Outline width (in SDF distance units, typically 0.0-0.3).
    pub width: f32,
    /// Outline colour (linear RGBA).
    pub color: [f32; 4],
    /// Softness (0 = hard, higher = softer edge).
    pub softness: f32,
}

impl Default for TextOutline {
    fn default() -> Self {
        Self {
            width: 0.15,
            color: [0.0, 0.0, 0.0, 1.0],
            softness: 0.05,
        }
    }
}

/// Text shadow configuration.
#[derive(Debug, Clone, Copy)]
pub struct TextShadow {
    /// Offset in pixels (x, y).
    pub offset: [f32; 2],
    /// Shadow colour (linear RGBA).
    pub color: [f32; 4],
    /// Softness (blur radius in SDF distance units).
    pub softness: f32,
}

impl Default for TextShadow {
    fn default() -> Self {
        Self {
            offset: [2.0, 2.0],
            color: [0.0, 0.0, 0.0, 0.5],
            softness: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Text vertex
// ---------------------------------------------------------------------------

/// Vertex layout for text rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TextVertex {
    /// Position (x, y).
    pub position: [f32; 2],
    /// Texture coordinates (u, v) in the glyph atlas.
    pub uv: [f32; 2],
    /// Colour (r, g, b, a).
    pub color: [f32; 4],
}

// ---------------------------------------------------------------------------
// Text draw parameters
// ---------------------------------------------------------------------------

/// Parameters for a text draw call.
#[derive(Debug, Clone)]
pub struct TextDrawParams {
    /// Position (screen or world space).
    pub position: [f32; 2],
    /// Font size in pixels.
    pub font_size: f32,
    /// Text colour (linear RGBA).
    pub color: [f32; 4],
    /// Horizontal alignment.
    pub align: TextAlign,
    /// Vertical alignment.
    pub valign: TextVAlign,
    /// Maximum width for wrapping (0 = no wrapping).
    pub max_width: f32,
    /// Line spacing multiplier (1.0 = default).
    pub line_spacing: f32,
    /// Letter spacing in pixels.
    pub letter_spacing: f32,
    /// Text outline (None = no outline).
    pub outline: Option<TextOutline>,
    /// Text shadow (None = no shadow).
    pub shadow: Option<TextShadow>,
    /// Rotation in radians (around the text origin).
    pub rotation: f32,
    /// Opacity.
    pub opacity: f32,
    /// Sort layer.
    pub layer: i32,
}

impl Default for TextDrawParams {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            font_size: 16.0,
            color: [1.0, 1.0, 1.0, 1.0],
            align: TextAlign::Left,
            valign: TextVAlign::Top,
            max_width: 0.0,
            line_spacing: 1.0,
            letter_spacing: 0.0,
            outline: None,
            shadow: None,
            rotation: 0.0,
            opacity: 1.0,
            layer: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// 3D text (billboarded)
// ---------------------------------------------------------------------------

/// Billboard mode for 3D text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BillboardMode {
    /// Face the camera fully (screen-aligned).
    ScreenAligned,
    /// Face the camera but locked to the Y axis (cylindrical).
    YAxisLocked,
    /// No billboarding (fixed orientation in world space).
    None,
}

/// 3D text draw parameters.
#[derive(Debug, Clone)]
pub struct Text3DParams {
    /// World-space position.
    pub position: [f32; 3],
    /// The 2D text parameters.
    pub text_params: TextDrawParams,
    /// Billboard mode.
    pub billboard: BillboardMode,
    /// World-space scale (pixels per world unit).
    pub scale: f32,
    /// Depth test enabled.
    pub depth_test: bool,
    /// Depth write enabled.
    pub depth_write: bool,
    /// Face culling enabled.
    pub cull: bool,
    /// Maximum view distance for the text.
    pub max_distance: f32,
    /// Distance-based scale (text stays constant screen size).
    pub constant_screen_size: bool,
}

impl Default for Text3DParams {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            text_params: TextDrawParams::default(),
            billboard: BillboardMode::ScreenAligned,
            scale: 0.01,
            depth_test: true,
            depth_write: false,
            cull: false,
            max_distance: 100.0,
            constant_screen_size: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Text batch
// ---------------------------------------------------------------------------

/// A batch of text glyphs for a single font/atlas.
#[derive(Debug, Clone)]
pub struct TextBatch {
    /// Font atlas texture handle.
    pub atlas_handle: u64,
    /// Vertices.
    pub vertices: Vec<TextVertex>,
    /// Indices.
    pub indices: Vec<u32>,
    /// Number of glyphs.
    pub glyph_count: u32,
    /// Sort layer.
    pub layer: i32,
    /// Outline parameters (if any glyphs have outlines).
    pub outline: Option<TextOutline>,
    /// Shadow parameters (if any glyphs have shadows).
    pub shadow: Option<TextShadow>,
}

impl TextBatch {
    /// Create a new empty batch.
    pub fn new(atlas_handle: u64, layer: i32) -> Self {
        Self {
            atlas_handle,
            vertices: Vec::new(),
            indices: Vec::new(),
            glyph_count: 0,
            layer,
            outline: None,
            shadow: None,
        }
    }

    /// Add a glyph quad to the batch.
    pub fn add_glyph(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        atlas_rect: &GlyphAtlasRect,
        color: [f32; 4],
    ) {
        let base = self.vertices.len() as u32;

        self.vertices.push(TextVertex {
            position: [x, y],
            uv: [atlas_rect.u, atlas_rect.v],
            color,
        });
        self.vertices.push(TextVertex {
            position: [x + w, y],
            uv: [atlas_rect.u + atlas_rect.w, atlas_rect.v],
            color,
        });
        self.vertices.push(TextVertex {
            position: [x + w, y + h],
            uv: [atlas_rect.u + atlas_rect.w, atlas_rect.v + atlas_rect.h],
            color,
        });
        self.vertices.push(TextVertex {
            position: [x, y + h],
            uv: [atlas_rect.u, atlas_rect.v + atlas_rect.h],
            color,
        });

        self.indices.push(base);
        self.indices.push(base + 1);
        self.indices.push(base + 2);
        self.indices.push(base);
        self.indices.push(base + 2);
        self.indices.push(base + 3);

        self.glyph_count += 1;
    }

    /// Clear the batch.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.glyph_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Text layout engine
// ---------------------------------------------------------------------------

/// Positioned glyph result from text layout.
#[derive(Debug, Clone)]
pub struct PositionedGlyph {
    /// Glyph codepoint.
    pub codepoint: char,
    /// Screen position (x, y) — top-left of the glyph quad.
    pub x: f32,
    pub y: f32,
    /// Glyph size (scaled).
    pub width: f32,
    pub height: f32,
    /// Atlas UV rect.
    pub atlas_rect: GlyphAtlasRect,
    /// Colour for this glyph.
    pub color: [f32; 4],
    /// Line index.
    pub line: usize,
}

/// Lay out text into positioned glyphs.
///
/// # Arguments
/// * `text` — The text to lay out.
/// * `font` — The SDF font.
/// * `params` — Draw parameters.
///
/// # Returns
/// List of positioned glyphs ready for batching.
pub fn layout_text(
    text: &str,
    font: &SdfFont,
    params: &TextDrawParams,
) -> Vec<PositionedGlyph> {
    let scale = params.font_size / font.base_size;
    let line_h = font.line_height * scale * params.line_spacing;

    // Word wrap.
    let lines = if params.max_width > 0.0 {
        word_wrap(text, font, params.font_size, params.max_width)
    } else {
        text.split('\n').map(|s| s.to_string()).collect()
    };

    let total_height = lines.len() as f32 * line_h;

    // Vertical offset based on alignment.
    let y_offset = match params.valign {
        TextVAlign::Top => 0.0,
        TextVAlign::Middle => -total_height * 0.5,
        TextVAlign::Bottom => -total_height,
        TextVAlign::Baseline => -font.ascender * scale,
    };

    let mut result = Vec::new();

    for (line_idx, line) in lines.iter().enumerate() {
        let line_width = font.measure_text(line, params.font_size);

        // Horizontal offset based on alignment.
        let x_offset = match params.align {
            TextAlign::Left => 0.0,
            TextAlign::Center => -line_width * 0.5,
            TextAlign::Right => -line_width,
            TextAlign::Justify => 0.0, // Handled below.
        };

        let mut pen_x = params.position[0] + x_offset;
        let pen_y = params.position[1] + y_offset + line_idx as f32 * line_h;

        // Justify: compute extra space per word gap.
        let justify_extra = if params.align == TextAlign::Justify && line_idx < lines.len() - 1 {
            let word_count = line.split_whitespace().count();
            if word_count > 1 && params.max_width > 0.0 {
                (params.max_width - line_width) / (word_count - 1) as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        let mut prev_char: Option<char> = None;
        let mut in_word_gap = false;

        for ch in line.chars() {
            if ch == ' ' && justify_extra > 0.0 {
                in_word_gap = true;
            }

            // Kerning.
            if let Some(prev) = prev_char {
                pen_x += font.get_kerning(prev, ch) * scale;
            }

            if let Some(glyph) = font.glyph(ch) {
                let gx = pen_x + glyph.metrics.bearing_x * scale;
                let gy = pen_y + (font.ascender - glyph.metrics.bearing_y) * scale;
                let gw = glyph.metrics.width * scale;
                let gh = glyph.metrics.height * scale;

                if ch != ' ' {
                    result.push(PositionedGlyph {
                        codepoint: ch,
                        x: gx,
                        y: gy,
                        width: gw,
                        height: gh,
                        atlas_rect: glyph.atlas_rect,
                        color: params.color,
                        line: line_idx,
                    });
                }

                pen_x += glyph.metrics.advance * scale + params.letter_spacing;

                if in_word_gap {
                    pen_x += justify_extra;
                    in_word_gap = false;
                }
            }

            prev_char = Some(ch);
        }
    }

    result
}

/// Lay out rich text into positioned glyphs with per-character styling.
pub fn layout_rich_text(
    rich: &RichText,
    font: &SdfFont,
    params: &TextDrawParams,
) -> Vec<PositionedGlyph> {
    let plain = rich.plain_text();
    let mut glyphs = layout_text(&plain, font, params);

    // Apply per-span colours.
    let mut char_idx = 0;
    for span in &rich.spans {
        for _ in span.text.chars() {
            if char_idx < glyphs.len() {
                if let Some(color) = span.style.color {
                    glyphs[char_idx].color = color;
                }
            }
            char_idx += 1;
        }
    }

    glyphs
}

// ---------------------------------------------------------------------------
// Text renderer
// ---------------------------------------------------------------------------

/// The text renderer manages batching and rendering of text.
#[derive(Debug)]
pub struct TextRenderer {
    /// Registered fonts.
    fonts: HashMap<String, SdfFont>,
    /// Pending text batches.
    batches: Vec<TextBatch>,
    /// Current frame number (for LRU tracking).
    frame: u64,
    /// Screen size.
    pub screen_size: (u32, u32),
    /// Default font name.
    pub default_font: String,
    /// SDF threshold for edge detection (typically 0.5).
    pub sdf_threshold: f32,
    /// SDF smoothing width (for anti-aliasing).
    pub sdf_smoothing: f32,
}

impl TextRenderer {
    /// Create a new text renderer.
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            fonts: HashMap::new(),
            batches: Vec::new(),
            frame: 0,
            screen_size: (screen_width, screen_height),
            default_font: String::new(),
            sdf_threshold: 0.5,
            sdf_smoothing: 0.1,
        }
    }

    /// Register a font.
    pub fn add_font(&mut self, font: SdfFont) {
        if self.default_font.is_empty() {
            self.default_font = font.name.clone();
        }
        self.fonts.insert(font.name.clone(), font);
    }

    /// Get a font by name.
    pub fn font(&self, name: &str) -> Option<&SdfFont> {
        self.fonts.get(name)
    }

    /// Get the default font.
    pub fn default_font(&self) -> Option<&SdfFont> {
        self.fonts.get(&self.default_font)
    }

    /// Draw text.
    pub fn draw_text(&mut self, text: &str, font_name: &str, params: &TextDrawParams) {
        let (glyphs, atlas_handle) = {
            let font = match self.fonts.get(font_name) {
                Some(f) => f,
                None => return,
            };
            (layout_text(text, font, params), font.atlas_handle)
        };
        self.batch_glyphs_direct(&glyphs, atlas_handle, params);
    }

    /// Draw rich text.
    pub fn draw_rich_text(&mut self, rich: &RichText, font_name: &str, params: &TextDrawParams) {
        let (glyphs, atlas_handle) = {
            let font = match self.fonts.get(font_name) {
                Some(f) => f,
                None => return,
            };
            (layout_rich_text(rich, font, params), font.atlas_handle)
        };
        self.batch_glyphs_direct(&glyphs, atlas_handle, params);
    }

    /// Internal: batch positioned glyphs (takes atlas handle directly to avoid borrow conflicts).
    fn batch_glyphs_direct(&mut self, glyphs: &[PositionedGlyph], atlas_handle: u64, params: &TextDrawParams) {
        let mut batch = TextBatch::new(atlas_handle, params.layer);
        batch.outline = params.outline;
        batch.shadow = params.shadow;

        for glyph in glyphs {
            let color = [
                glyph.color[0],
                glyph.color[1],
                glyph.color[2],
                glyph.color[3] * params.opacity,
            ];
            batch.add_glyph(
                glyph.x,
                glyph.y,
                glyph.width,
                glyph.height,
                &glyph.atlas_rect,
                color,
            );
        }

        if batch.glyph_count > 0 {
            self.batches.push(batch);
        }
    }

    /// Flush and get the compiled batches.
    pub fn flush(&mut self) -> Vec<TextBatch> {
        self.frame += 1;
        let mut batches = Vec::new();
        std::mem::swap(&mut batches, &mut self.batches);

        // Sort by layer.
        batches.sort_by_key(|b| b.layer);
        batches
    }

    /// Get the orthographic projection matrix for 2D text.
    pub fn projection_matrix(&self) -> [f32; 16] {
        let w = self.screen_size.0 as f32;
        let h = self.screen_size.1 as f32;

        [
            2.0 / w, 0.0, 0.0, 0.0,
            0.0, -2.0 / h, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
        ]
    }
}

// ---------------------------------------------------------------------------
// GPU uniform data
// ---------------------------------------------------------------------------

/// Packed SDF text shader parameters for GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TextSdfUniforms {
    /// sdf_threshold (x), sdf_smoothing (y), outline_width (z), outline_softness (w).
    pub sdf_params: [f32; 4],
    /// outline_color (rgba).
    pub outline_color: [f32; 4],
    /// shadow_offset (xy), shadow_softness (z), pad (w).
    pub shadow_params: [f32; 4],
    /// shadow_color (rgba).
    pub shadow_color: [f32; 4],
}

impl TextSdfUniforms {
    /// Build from text renderer settings and optional outline/shadow.
    pub fn from_params(
        threshold: f32,
        smoothing: f32,
        outline: Option<&TextOutline>,
        shadow: Option<&TextShadow>,
    ) -> Self {
        let (ow, os, oc) = outline
            .map(|o| (o.width, o.softness, o.color))
            .unwrap_or((0.0, 0.0, [0.0; 4]));

        let (so, ss, sc) = shadow
            .map(|s| (s.offset, s.softness, s.color))
            .unwrap_or(([0.0; 2], 0.0, [0.0; 4]));

        Self {
            sdf_params: [threshold, smoothing, ow, os],
            outline_color: oc,
            shadow_params: [so[0], so[1], ss, 0.0],
            shadow_color: sc,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_font() -> SdfFont {
        let mut font = SdfFont::new("test", 32.0, 1, (512, 512));

        // Add some test glyphs.
        for ch in "Hello world!".chars() {
            let entry = GlyphEntry {
                metrics: GlyphMetrics {
                    codepoint: ch,
                    advance: 16.0,
                    bearing_x: 1.0,
                    bearing_y: 24.0,
                    width: 14.0,
                    height: 28.0,
                },
                atlas_rect: GlyphAtlasRect::new(0.0, 0.0, 0.05, 0.05),
                font_size: 32.0,
                sdf_spread: 4.0,
                last_used_frame: 0,
            };
            font.add_glyph(entry);
        }

        font
    }

    #[test]
    fn test_measure_text() {
        let font = make_test_font();
        let width = font.measure_text("Hello", 32.0);
        assert!(width > 0.0);
    }

    #[test]
    fn test_word_wrap() {
        let font = make_test_font();
        let lines = word_wrap("Hello world", &font, 32.0, 100.0);
        // With 16px advance per glyph at 32px font size, "Hello" = 5*16 = 80px.
        // "Hello world" = 11*16 = 176px > 100px, so it should wrap.
        assert!(lines.len() >= 2);
    }

    #[test]
    fn test_text_layout() {
        let font = make_test_font();
        let params = TextDrawParams {
            position: [10.0, 10.0],
            font_size: 32.0,
            ..TextDrawParams::default()
        };
        let glyphs = layout_text("Hello", &font, &params);
        assert_eq!(glyphs.len(), 5);
    }

    #[test]
    fn test_atlas_packer() {
        let mut packer = GlyphAtlasPacker::new(256, 256);
        let pos = packer.pack(32, 32);
        assert!(pos.is_some());
        assert_eq!(packer.glyph_count, 1);
    }

    #[test]
    fn test_rich_text() {
        let rich = RichText::plain("Hello")
            .add_span(" World", RichTextStyle {
                color: Some([1.0, 0.0, 0.0, 1.0]),
                bold: true,
                ..RichTextStyle::default()
            });
        assert_eq!(rich.char_count(), 11);
    }
}
