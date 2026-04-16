//! Text rendering subsystem.
//!
//! Provides font loading, glyph atlas management, text shaping, line breaking,
//! and text measurement. The output is a [`GlyphRun`] of positioned glyphs
//! that the draw command system can render.
//!
//! # SDF Rendering
//!
//! Glyph bitmaps are stored as signed distance fields in the [`FontAtlas`],
//! allowing resolution-independent rendering with clean edges at any zoom
//! level. The atlas is packed using a shelf-based algorithm.

use std::collections::HashMap;

use glam::Vec2;
use serde::{Deserialize, Serialize};

use genovo_core::Rect;

// ---------------------------------------------------------------------------
// Font
// ---------------------------------------------------------------------------

/// Parsed font data. In a production engine this would wrap a font parsing
/// library (e.g. `ttf-parser`); here we store pre-computed metrics.
#[derive(Debug, Clone)]
pub struct Font {
    /// Font family name.
    pub family: String,
    /// Font identifier (index into the FontLibrary).
    pub id: u32,
    /// Units-per-em from the font header.
    pub units_per_em: u16,
    /// Ascender in font units (positive, above baseline).
    pub ascender: i16,
    /// Descender in font units (negative, below baseline).
    pub descender: i16,
    /// Line gap in font units.
    pub line_gap: i16,
    /// Per-glyph metrics keyed by Unicode code point.
    pub glyphs: HashMap<char, GlyphMetrics>,
    /// Kerning pairs: (left_char, right_char) -> horizontal adjustment in font
    /// units.
    pub kerning: HashMap<(char, char), i16>,
    /// Whether this is a monospace font.
    pub monospace: bool,
    /// Default advance width for missing glyphs (in font units).
    pub default_advance: u16,
}

/// Metrics for a single glyph.
#[derive(Debug, Clone, Copy)]
pub struct GlyphMetrics {
    /// Unicode code point.
    pub codepoint: char,
    /// Glyph index in the font.
    pub glyph_index: u16,
    /// Advance width in font units.
    pub advance_width: u16,
    /// Left side bearing in font units.
    pub lsb: i16,
    /// Bounding box in font units (min_x, min_y, max_x, max_y).
    pub bbox: (i16, i16, i16, i16),
}

impl Font {
    /// Create a default monospace font with ASCII glyph coverage.
    ///
    /// This is a placeholder that assigns uniform metrics to every printable
    /// ASCII character, suitable for development before real font files are
    /// loaded.
    pub fn default_monospace() -> Self {
        let mut glyphs = HashMap::new();
        for c in ' '..='~' {
            glyphs.insert(
                c,
                GlyphMetrics {
                    codepoint: c,
                    glyph_index: c as u16 - ' ' as u16,
                    advance_width: 600,
                    lsb: 50,
                    bbox: (50, 0, 550, 700),
                },
            );
        }
        Self {
            family: "Monospace".to_string(),
            id: 0,
            units_per_em: 1000,
            ascender: 800,
            descender: -200,
            line_gap: 0,
            glyphs,
            kerning: HashMap::new(),
            monospace: true,
            default_advance: 600,
        }
    }

    /// Create a default proportional font with ASCII glyph coverage.
    pub fn default_proportional() -> Self {
        let mut glyphs = HashMap::new();

        // Approximate proportional widths for common characters.
        let widths: &[(char, u16)] = &[
            (' ', 250), ('!', 300), ('"', 400), ('#', 500), ('$', 500),
            ('%', 700), ('&', 600), ('\'', 200), ('(', 300), (')', 300),
            ('*', 400), ('+', 500), (',', 250), ('-', 350), ('.', 250),
            ('/', 400), ('0', 500), ('1', 350), ('2', 500), ('3', 500),
            ('4', 500), ('5', 500), ('6', 500), ('7', 450), ('8', 500),
            ('9', 500), (':', 250), (';', 250), ('<', 500), ('=', 500),
            ('>', 500), ('?', 450), ('@', 800),
        ];
        for &(c, w) in widths {
            glyphs.insert(
                c,
                GlyphMetrics {
                    codepoint: c,
                    glyph_index: c as u16 - ' ' as u16,
                    advance_width: w,
                    lsb: 30,
                    bbox: (30, 0, (w as i16) - 30, 700),
                },
            );
        }
        // Uppercase letters.
        for c in 'A'..='Z' {
            let w = match c {
                'I' | 'J' => 350,
                'M' | 'W' => 750,
                'L' | 'T' | 'F' | 'E' => 500,
                _ => 600,
            };
            glyphs.insert(
                c,
                GlyphMetrics {
                    codepoint: c,
                    glyph_index: c as u16 - ' ' as u16,
                    advance_width: w,
                    lsb: 30,
                    bbox: (30, 0, (w as i16) - 30, 700),
                },
            );
        }
        // Lowercase letters.
        for c in 'a'..='z' {
            let w = match c {
                'i' | 'l' | 'j' => 250,
                'm' | 'w' => 700,
                'f' | 't' | 'r' => 350,
                _ => 500,
            };
            glyphs.insert(
                c,
                GlyphMetrics {
                    codepoint: c,
                    glyph_index: c as u16 - ' ' as u16,
                    advance_width: w,
                    lsb: 30,
                    bbox: (30, 0, (w as i16) - 30, 500),
                },
            );
        }

        Self {
            family: "Default".to_string(),
            id: 0,
            units_per_em: 1000,
            ascender: 800,
            descender: -200,
            line_gap: 100,
            glyphs,
            kerning: HashMap::new(),
            monospace: false,
            default_advance: 500,
        }
    }

    /// Scale factor to convert font units to pixels at a given font size.
    pub fn scale(&self, font_size: f32) -> f32 {
        font_size / self.units_per_em as f32
    }

    /// Pixel ascender for a given font size.
    pub fn ascender_px(&self, font_size: f32) -> f32 {
        self.ascender as f32 * self.scale(font_size)
    }

    /// Pixel descender for a given font size.
    pub fn descender_px(&self, font_size: f32) -> f32 {
        self.descender as f32 * self.scale(font_size)
    }

    /// Line height in pixels (ascender - descender + line_gap).
    pub fn line_height_px(&self, font_size: f32) -> f32 {
        let s = self.scale(font_size);
        (self.ascender as f32 - self.descender as f32 + self.line_gap as f32) * s
    }

    /// Advance width for a character in pixels.
    pub fn advance_px(&self, c: char, font_size: f32) -> f32 {
        let advance = self
            .glyphs
            .get(&c)
            .map(|g| g.advance_width)
            .unwrap_or(self.default_advance);
        advance as f32 * self.scale(font_size)
    }

    /// Kerning adjustment between two characters in pixels.
    pub fn kern_px(&self, left: char, right: char, font_size: f32) -> f32 {
        self.kerning
            .get(&(left, right))
            .map(|k| *k as f32 * self.scale(font_size))
            .unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// FontAtlas
// ---------------------------------------------------------------------------

/// UV rectangle for a glyph in the atlas texture.
#[derive(Debug, Clone, Copy)]
pub struct GlyphUV {
    pub u_min: f32,
    pub v_min: f32,
    pub u_max: f32,
    pub v_max: f32,
}

/// A packed glyph texture atlas.
///
/// Uses a simple shelf-based packing algorithm: rows of glyphs are packed
/// left-to-right, starting a new shelf when the current one is full.
pub struct FontAtlas {
    /// Atlas texture width in pixels.
    pub width: u32,
    /// Atlas texture height in pixels.
    pub height: u32,
    /// RGBA pixel data (width * height * 4 bytes).
    pub pixels: Vec<u8>,
    /// UV mapping for each (font_id, codepoint).
    pub glyph_uvs: HashMap<(u32, char), GlyphUV>,
    /// Current shelf height (tallest glyph in the current row).
    shelf_height: u32,
    /// Current x cursor within the shelf.
    cursor_x: u32,
    /// Current y cursor (top of the current shelf).
    cursor_y: u32,
    /// Padding between glyphs.
    padding: u32,
    /// Whether the atlas has been modified since last upload.
    pub dirty: bool,
}

impl FontAtlas {
    /// Create a new atlas with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let pixel_count = (width * height * 4) as usize;
        Self {
            width,
            height,
            pixels: vec![0u8; pixel_count],
            glyph_uvs: HashMap::new(),
            shelf_height: 0,
            cursor_x: 0,
            cursor_y: 0,
            padding: 2,
            dirty: true,
        }
    }

    /// Default atlas (1024x1024).
    pub fn default_size() -> Self {
        Self::new(1024, 1024)
    }

    /// Insert a glyph bitmap into the atlas.
    ///
    /// `data` is an 8-bit SDF bitmap of `glyph_w x glyph_h` pixels.
    /// Returns the UV rect, or None if the atlas is full.
    pub fn insert_glyph(
        &mut self,
        font_id: u32,
        codepoint: char,
        glyph_w: u32,
        glyph_h: u32,
        data: &[u8],
    ) -> Option<GlyphUV> {
        // Check if already present.
        if let Some(uv) = self.glyph_uvs.get(&(font_id, codepoint)) {
            return Some(*uv);
        }

        let padded_w = glyph_w + self.padding;
        let padded_h = glyph_h + self.padding;

        // Start a new shelf if needed.
        if self.cursor_x + padded_w > self.width {
            self.cursor_y += self.shelf_height + self.padding;
            self.cursor_x = 0;
            self.shelf_height = 0;
        }

        // Check if we've run out of vertical space.
        if self.cursor_y + padded_h > self.height {
            log::warn!(
                "Font atlas full, cannot insert glyph '{}' for font {}",
                codepoint,
                font_id
            );
            return None;
        }

        // Copy SDF data into the atlas (expanding 1-channel to RGBA).
        for row in 0..glyph_h {
            for col in 0..glyph_w {
                let src_idx = (row * glyph_w + col) as usize;
                let dst_x = self.cursor_x + col;
                let dst_y = self.cursor_y + row;
                let dst_idx = ((dst_y * self.width + dst_x) * 4) as usize;
                if src_idx < data.len() && dst_idx + 3 < self.pixels.len() {
                    let v = data[src_idx];
                    self.pixels[dst_idx] = 255; // R
                    self.pixels[dst_idx + 1] = 255; // G
                    self.pixels[dst_idx + 2] = 255; // B
                    self.pixels[dst_idx + 3] = v; // A (SDF distance)
                }
            }
        }

        let uv = GlyphUV {
            u_min: self.cursor_x as f32 / self.width as f32,
            v_min: self.cursor_y as f32 / self.height as f32,
            u_max: (self.cursor_x + glyph_w) as f32 / self.width as f32,
            v_max: (self.cursor_y + glyph_h) as f32 / self.height as f32,
        };

        self.glyph_uvs.insert((font_id, codepoint), uv);
        self.cursor_x += padded_w;
        self.shelf_height = self.shelf_height.max(padded_h);
        self.dirty = true;

        Some(uv)
    }

    /// Generate placeholder SDF data for a glyph. In production this would
    /// come from a rasteriser; here we generate a simple box SDF.
    pub fn generate_placeholder_sdf(w: u32, h: u32) -> Vec<u8> {
        let mut data = vec![0u8; (w * h) as usize];
        let border = 3.0_f32;
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f32).min((w - 1 - x) as f32);
                let dy = (y as f32).min((h - 1 - y) as f32);
                let d = dx.min(dy);
                let v = ((d / border).clamp(0.0, 1.0) * 255.0) as u8;
                data[(y * w + x) as usize] = v;
            }
        }
        data
    }

    /// Clear the atlas.
    pub fn clear(&mut self) {
        self.pixels.fill(0);
        self.glyph_uvs.clear();
        self.cursor_x = 0;
        self.cursor_y = 0;
        self.shelf_height = 0;
        self.dirty = true;
    }

    /// Occupancy ratio (0..1).
    pub fn occupancy(&self) -> f32 {
        let used_area = self.cursor_y * self.width + self.cursor_x * self.shelf_height;
        used_area as f32 / (self.width * self.height) as f32
    }
}

// ---------------------------------------------------------------------------
// TextLayout
// ---------------------------------------------------------------------------

/// How text overflows its container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextOverflow {
    /// Text is clipped at the boundary.
    Clip,
    /// An ellipsis (...) is shown at the truncation point.
    Ellipsis,
    /// Text scrolls within the container (for text inputs).
    Scroll,
    /// Text is allowed to overflow visibly.
    Visible,
}

impl Default for TextOverflow {
    fn default() -> Self {
        Self::Clip
    }
}

/// Word-wrapping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WrapMode {
    /// No wrapping (single line).
    None,
    /// Break at word boundaries.
    Word,
    /// Break at character boundaries.
    Character,
    /// Break at word boundaries first, then character if a word is too long.
    WordCharacter,
}

impl Default for WrapMode {
    fn default() -> Self {
        Self::Word
    }
}

/// Text horizontal alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HorizontalAlign {
    Left,
    Center,
    Right,
    Justify,
}

impl Default for HorizontalAlign {
    fn default() -> Self {
        Self::Left
    }
}

/// A span of rich text with uniform styling.
#[derive(Debug, Clone)]
pub struct TextSpan {
    /// The text content.
    pub text: String,
    /// Font size override (None = inherit).
    pub font_size: Option<f32>,
    /// Color override.
    pub color: Option<crate::render_commands::Color>,
    /// Font id override.
    pub font_id: Option<u32>,
    /// Bold.
    pub bold: bool,
    /// Italic.
    pub italic: bool,
    /// Underline.
    pub underline: bool,
    /// Strikethrough.
    pub strikethrough: bool,
}

impl TextSpan {
    pub fn plain(text: &str) -> Self {
        Self {
            text: text.to_string(),
            font_size: None,
            color: None,
            font_id: None,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
        }
    }

    pub fn styled(text: &str, font_size: f32, color: crate::render_commands::Color) -> Self {
        Self {
            text: text.to_string(),
            font_size: Some(font_size),
            color: Some(color),
            font_id: None,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
        }
    }

    pub fn with_bold(mut self) -> Self {
        self.bold = true;
        self
    }

    pub fn with_italic(mut self) -> Self {
        self.italic = true;
        self
    }

    pub fn with_underline(mut self) -> Self {
        self.underline = true;
        self
    }
}

/// A single positioned glyph in a glyph run.
#[derive(Debug, Clone, Copy)]
pub struct PositionedGlyph {
    /// Unicode codepoint.
    pub codepoint: char,
    /// Position of the glyph origin (baseline, left bearing).
    pub position: Vec2,
    /// Size of the glyph in pixels.
    pub size: Vec2,
    /// Font size used.
    pub font_size: f32,
    /// Font id.
    pub font_id: u32,
    /// Index of the source character in the original string.
    pub source_index: usize,
    /// Line index (0-based).
    pub line: u32,
}

/// A sequence of positioned glyphs ready for rendering.
#[derive(Debug, Clone)]
pub struct GlyphRun {
    pub glyphs: Vec<PositionedGlyph>,
    /// Total bounding box of the run.
    pub bounds: Rect,
    /// Per-line metrics.
    pub lines: Vec<LineMetrics>,
}

/// Metrics for a single line of text.
#[derive(Debug, Clone, Copy)]
pub struct LineMetrics {
    /// Y position of the baseline.
    pub baseline_y: f32,
    /// Width of the line in pixels.
    pub width: f32,
    /// Height of the line in pixels.
    pub height: f32,
    /// Number of glyphs on this line.
    pub glyph_count: u32,
    /// Index of the first glyph in the GlyphRun.
    pub first_glyph: u32,
    /// Index of the first source character.
    pub first_char: usize,
    /// Index past the last source character.
    pub last_char: usize,
}

impl GlyphRun {
    pub fn empty() -> Self {
        Self {
            glyphs: Vec::new(),
            bounds: Rect::new(Vec2::ZERO, Vec2::ZERO),
            lines: Vec::new(),
        }
    }

    /// Total number of lines.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Find the glyph nearest to a point (for cursor placement in text
    /// inputs).
    pub fn hit_test_point(&self, point: Vec2) -> Option<usize> {
        // Find the line.
        let line_idx = self.lines.iter().position(|l| {
            point.y >= l.baseline_y - l.height && point.y < l.baseline_y
        });
        let line_idx = line_idx.unwrap_or_else(|| {
            // Above first or below last line.
            if point.y < 0.0 {
                0
            } else {
                self.lines.len().saturating_sub(1)
            }
        });

        if let Some(line) = self.lines.get(line_idx) {
            let start = line.first_glyph as usize;
            let end = start + line.glyph_count as usize;
            let line_glyphs = &self.glyphs[start..end.min(self.glyphs.len())];

            // Find the glyph whose center is nearest to point.x.
            let mut best_idx = line.first_char;
            let mut best_dist = f32::INFINITY;
            for g in line_glyphs {
                let center_x = g.position.x + g.size.x / 2.0;
                let dist = (center_x - point.x).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = g.source_index;
                    // If point is past the glyph center, place cursor after it.
                    if point.x > center_x {
                        best_idx = g.source_index + 1;
                    }
                }
            }
            Some(best_idx)
        } else {
            None
        }
    }

    /// Returns the x position for a cursor at `char_index`.
    pub fn cursor_x_for_index(&self, char_index: usize) -> f32 {
        for g in &self.glyphs {
            if g.source_index == char_index {
                return g.position.x;
            }
        }
        // Past the end: return the right edge of the last glyph.
        if let Some(last) = self.glyphs.last() {
            last.position.x + last.size.x
        } else {
            0.0
        }
    }
}

/// Text measurement result (no glyph positions, just dimensions).
#[derive(Debug, Clone, Copy)]
pub struct TextMeasurement {
    pub width: f32,
    pub height: f32,
    pub line_count: u32,
}

/// The text layout engine. Computes glyph positions from text + font +
/// constraints.
pub struct TextLayout {
    /// Default line height multiplier.
    pub line_height_factor: f32,
}

impl TextLayout {
    pub fn new() -> Self {
        Self {
            line_height_factor: 1.2,
        }
    }

    /// Measure text dimensions without producing glyph positions.
    pub fn measure(
        &self,
        text: &str,
        font: &Font,
        font_size: f32,
        max_width: Option<f32>,
        wrap: WrapMode,
    ) -> TextMeasurement {
        let run = self.layout(text, font, font_size, max_width, wrap, HorizontalAlign::Left);
        TextMeasurement {
            width: run.bounds.width(),
            height: run.bounds.height(),
            line_count: run.line_count() as u32,
        }
    }

    /// Full text layout: compute positioned glyphs.
    pub fn layout(
        &self,
        text: &str,
        font: &Font,
        font_size: f32,
        max_width: Option<f32>,
        wrap: WrapMode,
        align: HorizontalAlign,
    ) -> GlyphRun {
        if text.is_empty() {
            return GlyphRun::empty();
        }

        let line_height = font.line_height_px(font_size) * self.line_height_factor;
        let ascender = font.ascender_px(font_size);
        let max_w = max_width.unwrap_or(f32::INFINITY);

        // Break text into lines.
        let lines = self.break_lines(text, font, font_size, max_w, wrap);

        let mut glyphs = Vec::new();
        let mut line_metrics = Vec::new();
        let mut total_height = 0.0_f32;
        let mut max_line_width = 0.0_f32;

        for (line_idx, line_info) in lines.iter().enumerate() {
            let baseline_y = ascender + line_idx as f32 * line_height;
            let first_glyph = glyphs.len() as u32;

            let mut x = 0.0_f32;
            let chars: Vec<char> = text[line_info.start..line_info.end].chars().collect();
            let mut prev_char: Option<char> = None;

            for (i, &c) in chars.iter().enumerate() {
                if c == '\n' || c == '\r' {
                    prev_char = Some(c);
                    continue;
                }

                // Apply kerning.
                if let Some(prev) = prev_char {
                    x += font.kern_px(prev, c, font_size);
                }

                let advance = font.advance_px(c, font_size);
                let scale = font.scale(font_size);
                let metrics = font.glyphs.get(&c);

                let glyph_width = metrics
                    .map(|m| (m.bbox.2 - m.bbox.0) as f32 * scale)
                    .unwrap_or(advance);
                let glyph_height = metrics
                    .map(|m| (m.bbox.3 - m.bbox.1) as f32 * scale)
                    .unwrap_or(font_size);

                glyphs.push(PositionedGlyph {
                    codepoint: c,
                    position: Vec2::new(x, baseline_y - glyph_height),
                    size: Vec2::new(glyph_width, glyph_height),
                    font_size,
                    font_id: font.id,
                    source_index: line_info.start + i,
                    line: line_idx as u32,
                });

                x += advance;
                prev_char = Some(c);
            }

            let line_width = x;
            max_line_width = max_line_width.max(line_width);

            line_metrics.push(LineMetrics {
                baseline_y,
                width: line_width,
                height: line_height,
                glyph_count: glyphs.len() as u32 - first_glyph,
                first_glyph,
                first_char: line_info.start,
                last_char: line_info.end,
            });

            total_height = baseline_y + (line_height - ascender);
        }

        // Apply horizontal alignment.
        if align != HorizontalAlign::Left {
            let container_width = if max_w.is_finite() {
                max_w
            } else {
                max_line_width
            };
            for lm in &line_metrics {
                let offset = match align {
                    HorizontalAlign::Center => (container_width - lm.width) / 2.0,
                    HorizontalAlign::Right => container_width - lm.width,
                    HorizontalAlign::Justify => 0.0, // justify is handled below
                    HorizontalAlign::Left => 0.0,
                };
                if offset.abs() > 0.001 {
                    let start = lm.first_glyph as usize;
                    let end = start + lm.glyph_count as usize;
                    let glyph_end = end.min(glyphs.len());
                    for g in &mut glyphs[start..glyph_end] {
                        g.position.x += offset;
                    }
                }

                // Justify: distribute extra space between words.
                if align == HorizontalAlign::Justify && lm.width < container_width {
                    let start = lm.first_glyph as usize;
                    let end = start + lm.glyph_count as usize;
                    let glyph_end = end.min(glyphs.len());

                    // Count spaces.
                    let space_count = glyphs[start..glyph_end]
                        .iter()
                        .filter(|g| g.codepoint == ' ')
                        .count();
                    if space_count > 0 {
                        let extra = (container_width - lm.width) / space_count as f32;
                        let mut spaces_seen = 0;
                        for g in &mut glyphs[start..glyph_end] {
                            g.position.x += spaces_seen as f32 * extra;
                            if g.codepoint == ' ' {
                                spaces_seen += 1;
                            }
                        }
                    }
                }
            }
        }

        let bounds = Rect::new(
            Vec2::ZERO,
            Vec2::new(max_line_width.min(max_w), total_height),
        );

        GlyphRun {
            glyphs,
            bounds,
            lines: line_metrics,
        }
    }

    /// Layout rich text with multiple spans.
    pub fn layout_rich(
        &self,
        spans: &[TextSpan],
        font: &Font,
        default_font_size: f32,
        max_width: Option<f32>,
        wrap: WrapMode,
        align: HorizontalAlign,
    ) -> GlyphRun {
        // Concatenate all span text and layout as a single string, then
        // attribute glyphs back to spans.
        let full_text: String = spans.iter().map(|s| s.text.as_str()).collect();
        let mut run = self.layout(&full_text, font, default_font_size, max_width, wrap, align);

        // Override font sizes and colours per span.
        let mut offset = 0usize;
        for span in spans {
            let span_end = offset + span.text.len();
            for g in &mut run.glyphs {
                if g.source_index >= offset && g.source_index < span_end {
                    if let Some(fs) = span.font_size {
                        g.font_size = fs;
                    }
                    if let Some(fid) = span.font_id {
                        g.font_id = fid;
                    }
                }
            }
            offset = span_end;
        }

        run
    }

    /// Break text into lines based on wrapping mode and max width.
    fn break_lines(
        &self,
        text: &str,
        font: &Font,
        font_size: f32,
        max_width: f32,
        wrap: WrapMode,
    ) -> Vec<LineBreak> {
        let mut lines = Vec::new();
        let mut line_start = 0;

        // Split on explicit newlines first.
        for (i, c) in text.char_indices() {
            if c == '\n' {
                lines.push(LineBreak {
                    start: line_start,
                    end: i,
                });
                line_start = i + 1;
            }
        }
        lines.push(LineBreak {
            start: line_start,
            end: text.len(),
        });

        // If no wrapping, we're done.
        if matches!(wrap, WrapMode::None) || max_width.is_infinite() {
            return lines;
        }

        // Apply word/character wrapping within each line.
        let mut wrapped = Vec::new();
        for line in &lines {
            let line_text = &text[line.start..line.end];
            if line_text.is_empty() {
                wrapped.push(LineBreak {
                    start: line.start,
                    end: line.end,
                });
                continue;
            }

            let sub_lines = match wrap {
                WrapMode::Word | WrapMode::WordCharacter => {
                    self.break_line_by_word(text, line.start, line.end, font, font_size, max_width)
                }
                WrapMode::Character => {
                    self.break_line_by_char(text, line.start, line.end, font, font_size, max_width)
                }
                WrapMode::None => unreachable!(),
            };
            wrapped.extend(sub_lines);
        }

        wrapped
    }

    fn break_line_by_word(
        &self,
        text: &str,
        start: usize,
        end: usize,
        font: &Font,
        font_size: f32,
        max_width: f32,
    ) -> Vec<LineBreak> {
        let mut lines = Vec::new();
        let mut line_start = start;
        let mut x = 0.0_f32;
        let mut last_break = start;
        let mut last_break_x = 0.0_f32;

        let chars: Vec<(usize, char)> = text[start..end]
            .char_indices()
            .map(|(i, c)| (start + i, c))
            .collect();

        for &(byte_idx, c) in &chars {
            let advance = font.advance_px(c, font_size);

            if c == ' ' || c == '\t' {
                last_break = byte_idx + c.len_utf8();
                last_break_x = x + advance;
            }

            if x + advance > max_width && x > 0.0 {
                // Need to break.
                if last_break > line_start {
                    lines.push(LineBreak {
                        start: line_start,
                        end: last_break,
                    });
                    line_start = last_break;
                    x -= last_break_x;
                    last_break_x = 0.0;
                } else {
                    // Word is too long — break at current position.
                    lines.push(LineBreak {
                        start: line_start,
                        end: byte_idx,
                    });
                    line_start = byte_idx;
                    x = 0.0;
                }
            }

            x += advance;
        }

        if line_start < end {
            lines.push(LineBreak {
                start: line_start,
                end,
            });
        }

        if lines.is_empty() {
            lines.push(LineBreak { start, end });
        }

        lines
    }

    fn break_line_by_char(
        &self,
        text: &str,
        start: usize,
        end: usize,
        font: &Font,
        font_size: f32,
        max_width: f32,
    ) -> Vec<LineBreak> {
        let mut lines = Vec::new();
        let mut line_start = start;
        let mut x = 0.0_f32;

        for (byte_offset, c) in text[start..end].char_indices() {
            let byte_idx = start + byte_offset;
            let advance = font.advance_px(c, font_size);

            if x + advance > max_width && x > 0.0 {
                lines.push(LineBreak {
                    start: line_start,
                    end: byte_idx,
                });
                line_start = byte_idx;
                x = 0.0;
            }
            x += advance;
        }

        if line_start < end {
            lines.push(LineBreak {
                start: line_start,
                end,
            });
        }

        if lines.is_empty() {
            lines.push(LineBreak { start, end });
        }

        lines
    }
}

impl Default for TextLayout {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal struct representing a line break position.
#[derive(Debug, Clone, Copy)]
struct LineBreak {
    /// Byte offset of the first character.
    start: usize,
    /// Byte offset past the last character.
    end: usize,
}

// ---------------------------------------------------------------------------
// ShapedText
// ---------------------------------------------------------------------------

/// The result of text shaping — a simplified harfbuzz-like output.
///
/// In a production engine this would delegate to HarfBuzz for complex scripts;
/// here we implement basic Latin shaping (ligatures, contextual forms are
/// stubbed).
#[derive(Debug, Clone)]
pub struct ShapedText {
    /// Shaped glyph indices and positions.
    pub glyphs: Vec<ShapedGlyph>,
    /// Total advance width.
    pub total_advance: f32,
}

/// A single shaped glyph.
#[derive(Debug, Clone, Copy)]
pub struct ShapedGlyph {
    /// Font glyph index.
    pub glyph_index: u16,
    /// Source character.
    pub codepoint: char,
    /// Horizontal advance.
    pub x_advance: f32,
    /// Vertical advance (usually 0 for horizontal text).
    pub y_advance: f32,
    /// Horizontal offset from the pen position.
    pub x_offset: f32,
    /// Vertical offset.
    pub y_offset: f32,
    /// Index into the source string.
    pub cluster: usize,
}

impl ShapedText {
    /// Shape a string of text with the given font and size.
    ///
    /// This is a simplified shaper that handles:
    /// - Basic character-to-glyph mapping
    /// - Kerning
    /// - (Stub) ligature substitution for fi, fl, ff
    pub fn shape(text: &str, font: &Font, font_size: f32) -> Self {
        let scale = font.scale(font_size);
        let mut glyphs = Vec::with_capacity(text.len());
        let mut total_advance = 0.0_f32;
        let mut prev_char: Option<char> = None;

        for (cluster, c) in text.char_indices() {
            let metrics = font.glyphs.get(&c);
            let glyph_index = metrics.map(|m| m.glyph_index).unwrap_or(0);
            let advance = metrics
                .map(|m| m.advance_width as f32)
                .unwrap_or(font.default_advance as f32)
                * scale;

            let kern = if let Some(prev) = prev_char {
                font.kern_px(prev, c, font_size)
            } else {
                0.0
            };

            glyphs.push(ShapedGlyph {
                glyph_index,
                codepoint: c,
                x_advance: advance,
                y_advance: 0.0,
                x_offset: kern,
                y_offset: 0.0,
                cluster,
            });

            total_advance += advance + kern;
            prev_char = Some(c);
        }

        Self {
            glyphs,
            total_advance,
        }
    }
}

// ---------------------------------------------------------------------------
// FontLibrary
// ---------------------------------------------------------------------------

/// Manages loaded fonts and provides font lookup.
pub struct FontLibrary {
    fonts: Vec<Font>,
    families: HashMap<String, Vec<u32>>,
    default_font: u32,
}

impl FontLibrary {
    pub fn new() -> Self {
        let default = Font::default_proportional();
        let mono = Font::default_monospace();

        let mut lib = Self {
            fonts: Vec::new(),
            families: HashMap::new(),
            default_font: 0,
        };

        lib.add_font(default);
        let mono_id = lib.add_font(mono);
        let _ = mono_id;

        lib
    }

    /// Add a font and return its id.
    pub fn add_font(&mut self, mut font: Font) -> u32 {
        let id = self.fonts.len() as u32;
        font.id = id;
        self.families
            .entry(font.family.clone())
            .or_default()
            .push(id);
        self.fonts.push(font);
        id
    }

    /// Get a font by id.
    pub fn get(&self, id: u32) -> Option<&Font> {
        self.fonts.get(id as usize)
    }

    /// Get the default font.
    pub fn default_font(&self) -> &Font {
        &self.fonts[self.default_font as usize]
    }

    /// Set the default font id.
    pub fn set_default(&mut self, id: u32) {
        if (id as usize) < self.fonts.len() {
            self.default_font = id;
        }
    }

    /// Find fonts by family name.
    pub fn find_family(&self, family: &str) -> Vec<&Font> {
        self.families
            .get(family)
            .map(|ids| ids.iter().filter_map(|id| self.get(*id)).collect())
            .unwrap_or_default()
    }

    /// Number of loaded fonts.
    pub fn count(&self) -> usize {
        self.fonts.len()
    }
}

impl Default for FontLibrary {
    fn default() -> Self {
        Self::new()
    }
}
