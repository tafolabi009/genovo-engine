//! Complete text system for the Slate UI.
//!
//! Provides font atlas management, text shaping, word wrapping, text
//! measurement, text selection, cursor management, rich text decorators,
//! syntax highlighting, clipboard integration, and per-field undo.

use std::collections::HashMap;
use std::ops::Range;

use glam::Vec2;
use genovo_core::Rect;

use crate::render_commands::Color;

// =========================================================================
// 1. Font Atlas
// =========================================================================

/// Metrics for a single rasterised glyph in the atlas.
#[derive(Debug, Clone, Copy)]
pub struct AtlasGlyph {
    /// Unicode codepoint.
    pub codepoint: char,
    /// Advance width in pixels at the atlas font size.
    pub advance: f32,
    /// Left-side bearing in pixels.
    pub bearing_x: f32,
    /// Top bearing (distance from baseline to top of glyph) in pixels.
    pub bearing_y: f32,
    /// Glyph bitmap width in pixels.
    pub width: f32,
    /// Glyph bitmap height in pixels.
    pub height: f32,
    /// UV rectangle in the atlas texture: (u_min, v_min, u_max, v_max).
    pub uv: (f32, f32, f32, f32),
}

/// A rasterised glyph atlas with multiple sizes.
///
/// In a production engine this would be backed by an actual texture atlas
/// with SDF glyphs. Here we store pre-computed glyph metrics for a clean
/// sans-serif font approximation.
#[derive(Debug, Clone)]
pub struct SlateFontAtlas {
    /// Font family name.
    pub family: String,
    /// Font id.
    pub font_id: u32,
    /// Atlas texture width.
    pub atlas_width: u32,
    /// Atlas texture height.
    pub atlas_height: u32,
    /// Base font size that glyphs were rasterised at.
    pub base_size: f32,
    /// Ascender (pixels above baseline at base size).
    pub ascender: f32,
    /// Descender (pixels below baseline -- negative).
    pub descender: f32,
    /// Line gap.
    pub line_gap: f32,
    /// Glyph data keyed by codepoint.
    pub glyphs: HashMap<char, AtlasGlyph>,
    /// Kerning pairs: (left, right) -> horizontal adjustment in pixels.
    pub kerning: HashMap<(char, char), f32>,
    /// Whether this is a monospace font.
    pub monospace: bool,
    /// Fallback glyph (used for missing characters).
    pub fallback_glyph: AtlasGlyph,
}

impl SlateFontAtlas {
    /// Build a default sans-serif font atlas with ASCII coverage and common
    /// Unicode characters. Glyph metrics approximate a standard proportional
    /// font.
    pub fn default_proportional() -> Self {
        let base_size = 32.0;
        let ascender = 24.0;
        let descender = -8.0;
        let line_gap = 4.0;

        let mut glyphs = HashMap::new();
        let atlas_cols = 16u32;
        let glyph_cell = 32.0;

        // Proportional width table for ASCII printable characters.
        // Values are normalised to the base size (1.0 = full width).
        let width_map: HashMap<char, f32> = [
            (' ', 0.30), ('!', 0.30), ('"', 0.38), ('#', 0.60), ('$', 0.55),
            ('%', 0.72), ('&', 0.68), ('\'', 0.22), ('(', 0.33), (')', 0.33),
            ('*', 0.44), ('+', 0.60), (',', 0.28), ('-', 0.36), ('.', 0.28),
            ('/', 0.36), ('0', 0.55), ('1', 0.40), ('2', 0.55), ('3', 0.55),
            ('4', 0.55), ('5', 0.55), ('6', 0.55), ('7', 0.50), ('8', 0.55),
            ('9', 0.55), (':', 0.28), (';', 0.28), ('<', 0.55), ('=', 0.55),
            ('>', 0.55), ('?', 0.50), ('@', 0.80),
            ('A', 0.65), ('B', 0.62), ('C', 0.62), ('D', 0.66), ('E', 0.56),
            ('F', 0.54), ('G', 0.68), ('H', 0.66), ('I', 0.28), ('J', 0.44),
            ('K', 0.62), ('L', 0.52), ('M', 0.78), ('N', 0.66), ('O', 0.70),
            ('P', 0.58), ('Q', 0.70), ('R', 0.62), ('S', 0.56), ('T', 0.56),
            ('U', 0.66), ('V', 0.62), ('W', 0.84), ('X', 0.60), ('Y', 0.58),
            ('Z', 0.58),
            ('[', 0.30), ('\\', 0.36), (']', 0.30), ('^', 0.48), ('_', 0.50),
            ('`', 0.30),
            ('a', 0.52), ('b', 0.55), ('c', 0.48), ('d', 0.55), ('e', 0.52),
            ('f', 0.33), ('g', 0.55), ('h', 0.55), ('i', 0.24), ('j', 0.24),
            ('k', 0.50), ('l', 0.24), ('m', 0.82), ('n', 0.55), ('o', 0.55),
            ('p', 0.55), ('q', 0.55), ('r', 0.36), ('s', 0.46), ('t', 0.34),
            ('u', 0.55), ('v', 0.50), ('w', 0.72), ('x', 0.50), ('y', 0.50),
            ('z', 0.46),
            ('{', 0.34), ('|', 0.26), ('}', 0.34), ('~', 0.60),
        ]
        .iter()
        .copied()
        .collect();

        let mut idx = 0u32;
        for cp in ' '..='~' {
            let col = idx % atlas_cols;
            let row = idx / atlas_cols;
            let w = width_map.get(&cp).copied().unwrap_or(0.55);
            let advance = w * base_size;
            let glyph_w = advance;
            let glyph_h = if cp == ' ' { 0.0 } else { ascender - descender };

            let u_min = col as f32 * glyph_cell / (atlas_cols as f32 * glyph_cell);
            let v_min = row as f32 * glyph_cell / (16.0 * glyph_cell);
            let u_max = u_min + glyph_cell / (atlas_cols as f32 * glyph_cell);
            let v_max = v_min + glyph_cell / (16.0 * glyph_cell);

            glyphs.insert(
                cp,
                AtlasGlyph {
                    codepoint: cp,
                    advance,
                    bearing_x: 0.0,
                    bearing_y: ascender,
                    width: glyph_w,
                    height: glyph_h,
                    uv: (u_min, v_min, u_max, v_max),
                },
            );
            idx += 1;
        }

        // Add some common Unicode characters with estimated widths.
        let extra_chars: Vec<(char, f32)> = vec![
            ('\u{00C0}', 0.65), // A-grave
            ('\u{00C9}', 0.56), // E-acute
            ('\u{00D1}', 0.66), // N-tilde
            ('\u{00DC}', 0.66), // U-umlaut
            ('\u{00E0}', 0.52), // a-grave
            ('\u{00E9}', 0.52), // e-acute
            ('\u{00F1}', 0.55), // n-tilde
            ('\u{00FC}', 0.55), // u-umlaut
            ('\u{2013}', 0.55), // en-dash
            ('\u{2014}', 0.80), // em-dash
            ('\u{2018}', 0.22), // left single quote
            ('\u{2019}', 0.22), // right single quote
            ('\u{201C}', 0.38), // left double quote
            ('\u{201D}', 0.38), // right double quote
            ('\u{2022}', 0.36), // bullet
            ('\u{2026}', 0.80), // ellipsis
            ('\u{20AC}', 0.55), // Euro sign
            ('\u{00A3}', 0.55), // Pound sign
            ('\u{00A5}', 0.55), // Yen sign
            ('\u{00A9}', 0.72), // Copyright
            ('\u{00AE}', 0.72), // Registered
            ('\u{2122}', 0.72), // Trademark
        ];

        for (cp, w) in extra_chars {
            let col = idx % atlas_cols;
            let row = idx / atlas_cols;
            let advance = w * base_size;
            let glyph_h = ascender - descender;
            let u_min = col as f32 * glyph_cell / (atlas_cols as f32 * glyph_cell);
            let v_min = row as f32 * glyph_cell / (16.0 * glyph_cell);
            let u_max = u_min + glyph_cell / (atlas_cols as f32 * glyph_cell);
            let v_max = v_min + glyph_cell / (16.0 * glyph_cell);

            glyphs.insert(
                cp,
                AtlasGlyph {
                    codepoint: cp,
                    advance,
                    bearing_x: 0.0,
                    bearing_y: ascender,
                    width: advance,
                    height: glyph_h,
                    uv: (u_min, v_min, u_max, v_max),
                },
            );
            idx += 1;
        }

        // Kerning pairs for common character combinations.
        let mut kerning = HashMap::new();
        let kern_pairs: Vec<((char, char), f32)> = vec![
            (('A', 'V'), -1.5), (('A', 'W'), -1.0), (('A', 'T'), -1.5),
            (('A', 'Y'), -1.5), (('A', 'v'), -0.8), (('A', 'w'), -0.6),
            (('F', 'a'), -0.8), (('F', '.'), -1.5), (('F', ','), -1.5),
            (('L', 'T'), -1.5), (('L', 'V'), -1.5), (('L', 'W'), -1.0),
            (('L', 'Y'), -1.5), (('L', 'y'), -0.8),
            (('P', '.'), -1.5), (('P', ','), -1.5), (('P', 'a'), -0.8),
            (('T', 'a'), -1.2), (('T', 'e'), -1.2), (('T', 'i'), -0.6),
            (('T', 'o'), -1.2), (('T', 'r'), -0.6), (('T', '.'), -1.5),
            (('T', ','), -1.5), (('T', ':'), -1.0), (('T', ';'), -1.0),
            (('V', 'a'), -1.0), (('V', 'e'), -1.0), (('V', 'o'), -1.0),
            (('V', '.'), -1.5), (('V', ','), -1.5),
            (('W', 'a'), -0.6), (('W', 'e'), -0.6), (('W', 'o'), -0.6),
            (('W', '.'), -1.0), (('W', ','), -1.0),
            (('Y', 'a'), -1.2), (('Y', 'e'), -1.2), (('Y', 'o'), -1.2),
            (('Y', '.'), -1.5), (('Y', ','), -1.5),
            (('f', '.'), -0.5), (('f', ','), -0.5),
            (('r', '.'), -0.5), (('r', ','), -0.5),
            (('v', '.'), -0.6), (('v', ','), -0.6),
            (('w', '.'), -0.4), (('w', ','), -0.4),
            (('y', '.'), -0.6), (('y', ','), -0.6),
        ];
        for ((a, b), kern) in kern_pairs {
            kerning.insert((a, b), kern);
        }

        let fallback = AtlasGlyph {
            codepoint: '\u{FFFD}',
            advance: 0.55 * base_size,
            bearing_x: 0.0,
            bearing_y: ascender,
            width: 0.55 * base_size,
            height: ascender - descender,
            uv: (0.0, 0.0, 0.0, 0.0),
        };

        Self {
            family: "Default Sans".to_string(),
            font_id: 0,
            atlas_width: atlas_cols * glyph_cell as u32,
            atlas_height: 16 * glyph_cell as u32,
            base_size,
            ascender,
            descender,
            line_gap,
            glyphs,
            kerning,
            monospace: false,
            fallback_glyph: fallback,
        }
    }

    /// Build a default monospace font atlas.
    pub fn default_monospace() -> Self {
        let base_size = 32.0;
        let ascender = 24.0;
        let descender = -8.0;
        let line_gap = 4.0;
        let mono_advance = 0.6 * base_size;

        let mut glyphs = HashMap::new();
        let atlas_cols = 16u32;
        let glyph_cell = 32.0;

        let mut idx = 0u32;
        for cp in ' '..='~' {
            let col = idx % atlas_cols;
            let row = idx / atlas_cols;
            let glyph_h = if cp == ' ' { 0.0 } else { ascender - descender };

            let u_min = col as f32 * glyph_cell / (atlas_cols as f32 * glyph_cell);
            let v_min = row as f32 * glyph_cell / (16.0 * glyph_cell);
            let u_max = u_min + glyph_cell / (atlas_cols as f32 * glyph_cell);
            let v_max = v_min + glyph_cell / (16.0 * glyph_cell);

            glyphs.insert(
                cp,
                AtlasGlyph {
                    codepoint: cp,
                    advance: mono_advance,
                    bearing_x: 0.0,
                    bearing_y: ascender,
                    width: mono_advance,
                    height: glyph_h,
                    uv: (u_min, v_min, u_max, v_max),
                },
            );
            idx += 1;
        }

        let fallback = AtlasGlyph {
            codepoint: '\u{FFFD}',
            advance: mono_advance,
            bearing_x: 0.0,
            bearing_y: ascender,
            width: mono_advance,
            height: ascender - descender,
            uv: (0.0, 0.0, 0.0, 0.0),
        };

        Self {
            family: "Default Mono".to_string(),
            font_id: 1,
            atlas_width: atlas_cols * glyph_cell as u32,
            atlas_height: 16 * glyph_cell as u32,
            base_size,
            ascender,
            descender,
            line_gap,
            glyphs,
            kerning: HashMap::new(),
            monospace: true,
            fallback_glyph: fallback,
        }
    }

    /// Get glyph metrics for a character, falling back to the fallback glyph.
    pub fn glyph(&self, c: char) -> &AtlasGlyph {
        self.glyphs.get(&c).unwrap_or(&self.fallback_glyph)
    }

    /// Get the advance width for a character at a given font size.
    pub fn advance(&self, c: char, font_size: f32) -> f32 {
        let scale = font_size / self.base_size;
        self.glyph(c).advance * scale
    }

    /// Get kerning between two characters at a given font size.
    pub fn kern(&self, left: char, right: char, font_size: f32) -> f32 {
        let scale = font_size / self.base_size;
        self.kerning
            .get(&(left, right))
            .copied()
            .unwrap_or(0.0)
            * scale
    }

    /// Line height for a given font size.
    pub fn line_height(&self, font_size: f32) -> f32 {
        let scale = font_size / self.base_size;
        (self.ascender - self.descender + self.line_gap) * scale
    }

    /// Ascender height for a given font size.
    pub fn ascender_at(&self, font_size: f32) -> f32 {
        self.ascender * (font_size / self.base_size)
    }

    /// Descender depth for a given font size.
    pub fn descender_at(&self, font_size: f32) -> f32 {
        self.descender * (font_size / self.base_size)
    }
}

// =========================================================================
// 2. Text Shaper
// =========================================================================

/// A positioned glyph in the output of text shaping.
#[derive(Debug, Clone, Copy)]
pub struct PositionedGlyph {
    pub codepoint: char,
    /// X position relative to the text origin.
    pub x: f32,
    /// Y position (baseline) relative to the text origin.
    pub y: f32,
    /// Advance width.
    pub advance: f32,
    /// UV coordinates in the atlas.
    pub uv: (f32, f32, f32, f32),
    /// Glyph bitmap size.
    pub size: Vec2,
    /// Bearing.
    pub bearing: Vec2,
    /// Line index (0-based).
    pub line: usize,
}

/// Text alignment for the shaper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeTextAlign {
    Left,
    Center,
    Right,
    Justify,
}

impl Default for ShapeTextAlign {
    fn default() -> Self {
        ShapeTextAlign::Left
    }
}

/// Text shaper: computes glyph positions for a string.
#[derive(Debug)]
pub struct TextShaper {
    atlas: SlateFontAtlas,
}

impl TextShaper {
    pub fn new(atlas: SlateFontAtlas) -> Self {
        Self { atlas }
    }

    /// Shape a single line of text (no wrapping).
    pub fn shape_text(&self, text: &str, font_size: f32) -> Vec<PositionedGlyph> {
        let scale = font_size / self.atlas.base_size;
        let mut glyphs = Vec::with_capacity(text.len());
        let mut x = 0.0f32;
        let mut prev_char: Option<char> = None;

        for ch in text.chars() {
            // Apply kerning.
            if let Some(prev) = prev_char {
                x += self.atlas.kern(prev, ch, font_size);
            }

            let glyph = self.atlas.glyph(ch);
            let advance = glyph.advance * scale;

            glyphs.push(PositionedGlyph {
                codepoint: ch,
                x,
                y: 0.0,
                advance,
                uv: glyph.uv,
                size: Vec2::new(glyph.width * scale, glyph.height * scale),
                bearing: Vec2::new(glyph.bearing_x * scale, glyph.bearing_y * scale),
                line: 0,
            });

            x += advance;
            prev_char = Some(ch);
        }

        glyphs
    }

    /// Shape text with word wrapping. Returns positioned glyphs with line
    /// information.
    pub fn shape_text_wrapped(
        &self,
        text: &str,
        font_size: f32,
        max_width: f32,
        align: ShapeTextAlign,
    ) -> Vec<PositionedGlyph> {
        let lines = self.compute_line_breaks(text, font_size, max_width);
        let line_height = self.atlas.line_height(font_size);

        let mut all_glyphs = Vec::new();

        for (line_idx, line_text) in lines.iter().enumerate() {
            let mut line_glyphs = self.shape_text(line_text, font_size);
            let line_width = line_glyphs.last().map(|g| g.x + g.advance).unwrap_or(0.0);

            // Apply alignment.
            let x_offset = match align {
                ShapeTextAlign::Left => 0.0,
                ShapeTextAlign::Center => (max_width - line_width) * 0.5,
                ShapeTextAlign::Right => max_width - line_width,
                ShapeTextAlign::Justify => {
                    // Justify by distributing extra space between words.
                    if line_idx < lines.len() - 1 && !line_text.is_empty() {
                        let spaces = line_text.chars().filter(|c| *c == ' ').count();
                        if spaces > 0 {
                            let extra = max_width - line_width;
                            let per_space = extra / spaces as f32;
                            let mut space_count = 0usize;
                            for glyph in &mut line_glyphs {
                                glyph.x += space_count as f32 * per_space;
                                if glyph.codepoint == ' ' {
                                    space_count += 1;
                                }
                            }
                        }
                    }
                    0.0
                }
            };

            let y_offset = line_idx as f32 * line_height;

            for glyph in &mut line_glyphs {
                glyph.x += x_offset;
                glyph.y = y_offset;
                glyph.line = line_idx;
            }

            all_glyphs.extend(line_glyphs);
        }

        all_glyphs
    }

    /// Compute line breaks for word wrapping.
    pub fn compute_line_breaks(
        &self,
        text: &str,
        font_size: f32,
        max_width: f32,
    ) -> Vec<String> {
        if text.is_empty() {
            return vec![String::new()];
        }

        let mut lines = Vec::new();

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
            let mut current_width = 0.0f32;
            let space_width = self.atlas.advance(' ', font_size);

            for word in &words {
                let word_width = self.measure_word(word, font_size);

                if current_line.is_empty() {
                    // First word on the line.
                    if word_width > max_width {
                        // Word is too long; break it character by character.
                        let mut chars = word.chars().peekable();
                        let mut fragment = String::new();
                        let mut frag_width = 0.0;
                        while let Some(ch) = chars.next() {
                            let ch_width = self.atlas.advance(ch, font_size);
                            if frag_width + ch_width > max_width && !fragment.is_empty() {
                                lines.push(fragment);
                                fragment = String::new();
                                frag_width = 0.0;
                            }
                            fragment.push(ch);
                            frag_width += ch_width;
                        }
                        current_line = fragment;
                        current_width = frag_width;
                    } else {
                        current_line = word.to_string();
                        current_width = word_width;
                    }
                } else {
                    let test_width = current_width + space_width + word_width;
                    if test_width <= max_width {
                        current_line.push(' ');
                        current_line.push_str(word);
                        current_width = test_width;
                    } else {
                        lines.push(current_line);
                        current_line = word.to_string();
                        current_width = word_width;
                    }
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

    fn measure_word(&self, word: &str, font_size: f32) -> f32 {
        let mut width = 0.0;
        let mut prev: Option<char> = None;
        for ch in word.chars() {
            if let Some(p) = prev {
                width += self.atlas.kern(p, ch, font_size);
            }
            width += self.atlas.advance(ch, font_size);
            prev = Some(ch);
        }
        width
    }

    /// Get the underlying atlas reference.
    pub fn atlas(&self) -> &SlateFontAtlas {
        &self.atlas
    }
}

// =========================================================================
// 3. Text Measurement
// =========================================================================

/// Measure text dimensions without rendering.
#[derive(Debug)]
pub struct SlateTextMeasurement {
    atlas: SlateFontAtlas,
}

impl SlateTextMeasurement {
    pub fn new(atlas: SlateFontAtlas) -> Self {
        Self { atlas }
    }

    /// Measure the bounding box of text.
    pub fn measure(&self, text: &str, font_size: f32, max_width: Option<f32>) -> Vec2 {
        if text.is_empty() {
            return Vec2::new(0.0, self.atlas.line_height(font_size));
        }

        match max_width {
            Some(mw) => {
                let shaper = TextShaper::new(self.atlas.clone());
                let lines = shaper.compute_line_breaks(text, font_size, mw);
                let line_height = self.atlas.line_height(font_size);
                let max_line_w = lines
                    .iter()
                    .map(|l| self.measure_line_width(l, font_size))
                    .fold(0.0f32, f32::max);
                Vec2::new(max_line_w, lines.len() as f32 * line_height)
            }
            None => {
                // Single line.
                let width = self.measure_line_width(text, font_size);
                Vec2::new(width, self.atlas.line_height(font_size))
            }
        }
    }

    /// Measure just the width of a single line.
    pub fn measure_line_width(&self, text: &str, font_size: f32) -> f32 {
        let mut width = 0.0;
        let mut prev: Option<char> = None;
        for ch in text.chars() {
            if let Some(p) = prev {
                width += self.atlas.kern(p, ch, font_size);
            }
            width += self.atlas.advance(ch, font_size);
            prev = Some(ch);
        }
        width
    }

    /// Count the number of lines when wrapping text.
    pub fn line_count(&self, text: &str, font_size: f32, max_width: f32) -> usize {
        let shaper = TextShaper::new(self.atlas.clone());
        shaper.compute_line_breaks(text, font_size, max_width).len()
    }

    /// Get the character index at a given x offset within a single line.
    pub fn char_index_at_x(&self, text: &str, font_size: f32, x: f32) -> usize {
        let mut acc = 0.0;
        let mut prev: Option<char> = None;
        for (i, ch) in text.chars().enumerate() {
            if let Some(p) = prev {
                acc += self.atlas.kern(p, ch, font_size);
            }
            let advance = self.atlas.advance(ch, font_size);
            if x < acc + advance * 0.5 {
                return i;
            }
            acc += advance;
            prev = Some(ch);
        }
        text.chars().count()
    }

    /// Get the x offset for a character index.
    pub fn x_at_char_index(&self, text: &str, font_size: f32, index: usize) -> f32 {
        let mut x = 0.0;
        let mut prev: Option<char> = None;
        for (i, ch) in text.chars().enumerate() {
            if i == index {
                return x;
            }
            if let Some(p) = prev {
                x += self.atlas.kern(p, ch, font_size);
            }
            x += self.atlas.advance(ch, font_size);
            prev = Some(ch);
        }
        x
    }
}

// =========================================================================
// 4. Text Selection
// =========================================================================

/// A text selection range (start and end cursor positions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextSelection {
    /// Selection anchor (where the selection started).
    pub anchor: TextPosition,
    /// Selection head (current cursor position -- the moving end).
    pub head: TextPosition,
}

/// A position within text (line and column).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextPosition {
    pub line: usize,
    pub column: usize,
}

impl TextPosition {
    pub fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }

    pub fn zero() -> Self {
        Self { line: 0, column: 0 }
    }

    /// Convert to a linear offset within a string.
    pub fn to_offset(&self, text: &str) -> usize {
        let mut offset = 0;
        for (i, line) in text.lines().enumerate() {
            if i == self.line {
                return offset + self.column.min(line.len());
            }
            offset += line.len() + 1; // +1 for newline.
        }
        text.len()
    }

    /// Create from a linear offset.
    pub fn from_offset(text: &str, offset: usize) -> Self {
        let offset = offset.min(text.len());
        let mut line = 0;
        let mut col = 0;
        let mut current_offset = 0;

        for (i, ch) in text.chars().enumerate() {
            if current_offset >= offset {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
            current_offset += ch.len_utf8();
        }

        Self { line, column: col }
    }
}

impl PartialOrd for TextPosition {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TextPosition {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.line
            .cmp(&other.line)
            .then_with(|| self.column.cmp(&other.column))
    }
}

impl TextSelection {
    pub fn new(anchor: TextPosition, head: TextPosition) -> Self {
        Self { anchor, head }
    }

    pub fn caret(position: TextPosition) -> Self {
        Self {
            anchor: position,
            head: position,
        }
    }

    /// Whether this is a caret (no actual selection).
    pub fn is_caret(&self) -> bool {
        self.anchor == self.head
    }

    /// Get the start (leftmost) position.
    pub fn start(&self) -> TextPosition {
        if self.anchor <= self.head {
            self.anchor
        } else {
            self.head
        }
    }

    /// Get the end (rightmost) position.
    pub fn end(&self) -> TextPosition {
        if self.anchor >= self.head {
            self.anchor
        } else {
            self.head
        }
    }

    /// Get the selected text from a string.
    pub fn selected_text<'a>(&self, text: &'a str) -> &'a str {
        let start = self.start().to_offset(text);
        let end = self.end().to_offset(text);
        &text[start..end]
    }

    /// Select the word at a given position (double-click behaviour).
    pub fn select_word_at(text: &str, position: TextPosition) -> Self {
        let offset = position.to_offset(text);
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();

        if offset >= len {
            return Self::caret(position);
        }

        let mut start = offset;
        while start > 0 && chars[start - 1].is_alphanumeric() {
            start -= 1;
        }
        let mut end = offset;
        while end < len && chars[end].is_alphanumeric() {
            end += 1;
        }

        Self::new(
            TextPosition::from_offset(text, start),
            TextPosition::from_offset(text, end),
        )
    }

    /// Select the entire line at a given position (triple-click behaviour).
    pub fn select_line_at(text: &str, position: TextPosition) -> Self {
        let lines: Vec<&str> = text.lines().collect();
        let line = position.line.min(lines.len().saturating_sub(1));

        let start = TextPosition::new(line, 0);
        let end = TextPosition::new(line, lines.get(line).map_or(0, |l| l.len()));
        Self::new(start, end)
    }

    /// Select all text.
    pub fn select_all(text: &str) -> Self {
        let lines: Vec<&str> = text.lines().collect();
        let last_line = lines.len().saturating_sub(1);
        let last_col = lines.last().map_or(0, |l| l.len());
        Self::new(
            TextPosition::zero(),
            TextPosition::new(last_line, last_col),
        )
    }
}

// =========================================================================
// 5. Text Cursor
// =========================================================================

/// Blinking text cursor with position tracking.
#[derive(Debug, Clone)]
pub struct TextCursor {
    /// Current position.
    pub position: TextPosition,
    /// Selection anchor (for shift-click / shift-arrow selection).
    pub selection_anchor: Option<TextPosition>,
    /// Blink timer (seconds).
    pub blink_timer: f32,
    /// Whether the cursor is currently visible (blink state).
    pub visible: bool,
    /// Blink on duration (seconds).
    pub blink_on: f32,
    /// Blink off duration (seconds).
    pub blink_off: f32,
    /// Preferred column (for up/down navigation through lines of different
    /// lengths).
    pub preferred_column: Option<usize>,
}

impl TextCursor {
    pub fn new() -> Self {
        Self {
            position: TextPosition::zero(),
            selection_anchor: None,
            blink_timer: 0.0,
            visible: true,
            blink_on: 0.53,
            blink_off: 0.53,
            preferred_column: None,
        }
    }

    /// Update the blink timer.
    pub fn tick(&mut self, dt: f32) {
        self.blink_timer += dt;
        let cycle = self.blink_on + self.blink_off;
        let phase = self.blink_timer % cycle;
        self.visible = phase < self.blink_on;
    }

    /// Reset the blink timer (called when the cursor moves).
    pub fn reset_blink(&mut self) {
        self.blink_timer = 0.0;
        self.visible = true;
    }

    /// Move the cursor left by one character.
    pub fn move_left(&mut self, text: &str) {
        let offset = self.position.to_offset(text);
        if offset > 0 {
            self.position = TextPosition::from_offset(text, offset - 1);
        }
        self.preferred_column = None;
        self.reset_blink();
    }

    /// Move the cursor right by one character.
    pub fn move_right(&mut self, text: &str) {
        let offset = self.position.to_offset(text);
        if offset < text.len() {
            self.position = TextPosition::from_offset(text, offset + 1);
        }
        self.preferred_column = None;
        self.reset_blink();
    }

    /// Move the cursor up by one line.
    pub fn move_up(&mut self, text: &str) {
        if self.position.line > 0 {
            let col = self.preferred_column.unwrap_or(self.position.column);
            self.position.line -= 1;
            let lines: Vec<&str> = text.lines().collect();
            let line_len = lines.get(self.position.line).map_or(0, |l| l.len());
            self.position.column = col.min(line_len);
            if self.preferred_column.is_none() {
                self.preferred_column = Some(col);
            }
        }
        self.reset_blink();
    }

    /// Move the cursor down by one line.
    pub fn move_down(&mut self, text: &str) {
        let line_count = text.lines().count().max(1);
        if self.position.line + 1 < line_count {
            let col = self.preferred_column.unwrap_or(self.position.column);
            self.position.line += 1;
            let lines: Vec<&str> = text.lines().collect();
            let line_len = lines.get(self.position.line).map_or(0, |l| l.len());
            self.position.column = col.min(line_len);
            if self.preferred_column.is_none() {
                self.preferred_column = Some(col);
            }
        }
        self.reset_blink();
    }

    /// Move to the start of the current line.
    pub fn move_home(&mut self) {
        self.position.column = 0;
        self.preferred_column = None;
        self.reset_blink();
    }

    /// Move to the end of the current line.
    pub fn move_end(&mut self, text: &str) {
        let lines: Vec<&str> = text.lines().collect();
        let line_len = lines.get(self.position.line).map_or(0, |l| l.len());
        self.position.column = line_len;
        self.preferred_column = None;
        self.reset_blink();
    }

    /// Move one word to the left (Ctrl+Left).
    pub fn move_word_left(&mut self, text: &str) {
        let offset = self.position.to_offset(text);
        let chars: Vec<char> = text.chars().collect();
        let mut p = offset;

        // Skip whitespace/punctuation.
        while p > 0 && !chars[p - 1].is_alphanumeric() {
            p -= 1;
        }
        // Skip word characters.
        while p > 0 && chars[p - 1].is_alphanumeric() {
            p -= 1;
        }

        self.position = TextPosition::from_offset(text, p);
        self.preferred_column = None;
        self.reset_blink();
    }

    /// Move one word to the right (Ctrl+Right).
    pub fn move_word_right(&mut self, text: &str) {
        let offset = self.position.to_offset(text);
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut p = offset;

        // Skip word characters.
        while p < len && chars[p].is_alphanumeric() {
            p += 1;
        }
        // Skip whitespace/punctuation.
        while p < len && !chars[p].is_alphanumeric() {
            p += 1;
        }

        self.position = TextPosition::from_offset(text, p);
        self.preferred_column = None;
        self.reset_blink();
    }

    /// Set position directly.
    pub fn set_position(&mut self, pos: TextPosition) {
        self.position = pos;
        self.preferred_column = None;
        self.reset_blink();
    }

    /// Get the current selection (if any).
    pub fn selection(&self) -> Option<TextSelection> {
        self.selection_anchor.map(|anchor| {
            TextSelection::new(anchor, self.position)
        })
    }

    /// Begin selection from the current position.
    pub fn begin_selection(&mut self) {
        if self.selection_anchor.is_none() {
            self.selection_anchor = Some(self.position);
        }
    }

    /// End selection.
    pub fn clear_selection(&mut self) {
        self.selection_anchor = None;
    }
}

impl Default for TextCursor {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// 6. Rich Text Decorator Trait
// =========================================================================

/// Token produced by a rich text decorator.
#[derive(Debug, Clone)]
pub struct DecoratorToken {
    pub range: Range<usize>,
    pub color: Option<Color>,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub background: Option<Color>,
}

/// Trait for custom rich-text decorators (inline rendering modifiers).
pub trait SlateRichTextDecorator: std::fmt::Debug {
    fn name(&self) -> &str;
    fn create_tokens(&self, text: &str) -> Vec<DecoratorToken>;
}

/// Bold decorator.
#[derive(Debug)]
pub struct SlateBoldDecorator;
impl SlateRichTextDecorator for SlateBoldDecorator {
    fn name(&self) -> &str { "bold" }
    fn create_tokens(&self, text: &str) -> Vec<DecoratorToken> {
        // Find text between ** markers.
        let mut tokens = Vec::new();
        let mut start = None;
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len().saturating_sub(1) {
            if chars[i] == '*' && chars[i + 1] == '*' {
                if let Some(s) = start {
                    tokens.push(DecoratorToken {
                        range: s..i,
                        color: None,
                        bold: true,
                        italic: false,
                        underline: false,
                        background: None,
                    });
                    start = None;
                    i += 2;
                    continue;
                } else {
                    start = Some(i + 2);
                    i += 2;
                    continue;
                }
            }
            i += 1;
        }
        tokens
    }
}

/// Italic decorator.
#[derive(Debug)]
pub struct SlateItalicDecorator;
impl SlateRichTextDecorator for SlateItalicDecorator {
    fn name(&self) -> &str { "italic" }
    fn create_tokens(&self, text: &str) -> Vec<DecoratorToken> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = None;
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '_' {
                if let Some(s) = start {
                    tokens.push(DecoratorToken {
                        range: s..i,
                        color: None,
                        bold: false,
                        italic: true,
                        underline: false,
                        background: None,
                    });
                    start = None;
                } else {
                    start = Some(i + 1);
                }
            }
            i += 1;
        }
        tokens
    }
}

/// Color decorator (inline color markup).
#[derive(Debug)]
pub struct SlateColorDecorator;
impl SlateRichTextDecorator for SlateColorDecorator {
    fn name(&self) -> &str { "color" }
    fn create_tokens(&self, _text: &str) -> Vec<DecoratorToken> {
        // Placeholder: real implementation would parse color tags.
        Vec::new()
    }
}

/// Link decorator.
#[derive(Debug)]
pub struct SlateLinkDecorator;
impl SlateRichTextDecorator for SlateLinkDecorator {
    fn name(&self) -> &str { "link" }
    fn create_tokens(&self, _text: &str) -> Vec<DecoratorToken> {
        Vec::new()
    }
}

/// Inline image decorator.
#[derive(Debug)]
pub struct SlateImageDecorator;
impl SlateRichTextDecorator for SlateImageDecorator {
    fn name(&self) -> &str { "image" }
    fn create_tokens(&self, _text: &str) -> Vec<DecoratorToken> {
        Vec::new()
    }
}

// =========================================================================
// 7. Syntax Highlighter
// =========================================================================

/// Token types for syntax highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    Keyword,
    String,
    Number,
    Comment,
    Operator,
    Identifier,
    Punctuation,
    Type,
    Function,
    Macro,
    Attribute,
    Whitespace,
    Unknown,
}

impl TokenType {
    /// Default colour for this token type.
    pub fn default_color(&self) -> Color {
        match self {
            TokenType::Keyword => Color::from_hex("#569CD6"),
            TokenType::String => Color::from_hex("#CE9178"),
            TokenType::Number => Color::from_hex("#B5CEA8"),
            TokenType::Comment => Color::from_hex("#6A9955"),
            TokenType::Operator => Color::from_hex("#D4D4D4"),
            TokenType::Identifier => Color::from_hex("#9CDCFE"),
            TokenType::Punctuation => Color::from_hex("#D4D4D4"),
            TokenType::Type => Color::from_hex("#4EC9B0"),
            TokenType::Function => Color::from_hex("#DCDCAA"),
            TokenType::Macro => Color::from_hex("#4FC1FF"),
            TokenType::Attribute => Color::from_hex("#C586C0"),
            TokenType::Whitespace => Color::TRANSPARENT,
            TokenType::Unknown => Color::from_hex("#D4D4D4"),
        }
    }
}

/// A highlighted token with range and type.
#[derive(Debug, Clone)]
pub struct HighlightToken {
    pub range: Range<usize>,
    pub token_type: TokenType,
}

/// Language for syntax highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HighlightLanguage {
    /// Rust-like syntax.
    RustLike,
    /// WGSL shader language.
    Wgsl,
    /// JSON.
    Json,
    /// Generic script (C-like).
    GenericScript,
}

/// Syntax highlighter: tokenizes text for colored display.
#[derive(Debug)]
pub struct SyntaxHighlighter {
    /// Keyword sets per language.
    keywords: HashMap<HighlightLanguage, Vec<String>>,
    /// Type names per language.
    type_names: HashMap<HighlightLanguage, Vec<String>>,
}

impl SyntaxHighlighter {
    pub fn new() -> Self {
        let mut keywords = HashMap::new();
        let mut type_names = HashMap::new();

        // Rust-like keywords.
        keywords.insert(
            HighlightLanguage::RustLike,
            vec![
                "fn", "let", "mut", "const", "static", "pub", "mod", "use", "struct", "enum",
                "impl", "trait", "type", "where", "as", "if", "else", "match", "for", "while",
                "loop", "break", "continue", "return", "self", "Self", "super", "crate",
                "true", "false", "in", "ref", "move", "async", "await", "unsafe", "extern",
                "dyn", "box",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );
        type_names.insert(
            HighlightLanguage::RustLike,
            vec![
                "i8", "i16", "i32", "i64", "i128", "isize",
                "u8", "u16", "u32", "u64", "u128", "usize",
                "f32", "f64", "bool", "char", "str", "String",
                "Vec", "Option", "Result", "Box", "Rc", "Arc",
                "HashMap", "HashSet", "BTreeMap", "BTreeSet",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // WGSL keywords.
        keywords.insert(
            HighlightLanguage::Wgsl,
            vec![
                "fn", "var", "let", "const", "struct", "if", "else", "for", "while", "loop",
                "break", "continue", "return", "switch", "case", "default", "discard",
                "enable", "alias", "override",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );
        type_names.insert(
            HighlightLanguage::Wgsl,
            vec![
                "f32", "f16", "i32", "u32", "bool",
                "vec2", "vec3", "vec4", "mat2x2", "mat3x3", "mat4x4",
                "sampler", "texture_2d", "texture_3d", "texture_cube",
                "array", "ptr",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // JSON keywords.
        keywords.insert(
            HighlightLanguage::Json,
            vec!["true", "false", "null"]
                .into_iter()
                .map(String::from)
                .collect(),
        );
        type_names.insert(HighlightLanguage::Json, Vec::new());

        // Generic script.
        keywords.insert(
            HighlightLanguage::GenericScript,
            vec![
                "function", "var", "let", "const", "if", "else", "for", "while", "do",
                "switch", "case", "default", "break", "continue", "return", "new", "delete",
                "typeof", "instanceof", "in", "of", "class", "extends", "import", "export",
                "from", "try", "catch", "finally", "throw", "async", "await", "yield",
                "true", "false", "null", "undefined", "this", "super",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );
        type_names.insert(HighlightLanguage::GenericScript, Vec::new());

        Self {
            keywords,
            type_names,
        }
    }

    /// Tokenize text into highlighted tokens.
    pub fn highlight(&self, text: &str, language: HighlightLanguage) -> Vec<HighlightToken> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        let keywords = self.keywords.get(&language).cloned().unwrap_or_default();
        let types = self.type_names.get(&language).cloned().unwrap_or_default();

        while i < len {
            let ch = chars[i];

            // Whitespace.
            if ch.is_whitespace() {
                let start = i;
                while i < len && chars[i].is_whitespace() {
                    i += 1;
                }
                tokens.push(HighlightToken {
                    range: start..i,
                    token_type: TokenType::Whitespace,
                });
                continue;
            }

            // Line comment.
            if i + 1 < len && ch == '/' && chars[i + 1] == '/' {
                let start = i;
                while i < len && chars[i] != '\n' {
                    i += 1;
                }
                tokens.push(HighlightToken {
                    range: start..i,
                    token_type: TokenType::Comment,
                });
                continue;
            }

            // Block comment.
            if i + 1 < len && ch == '/' && chars[i + 1] == '*' {
                let start = i;
                i += 2;
                while i + 1 < len {
                    if chars[i] == '*' && chars[i + 1] == '/' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
                tokens.push(HighlightToken {
                    range: start..i,
                    token_type: TokenType::Comment,
                });
                continue;
            }

            // String literal.
            if ch == '"' {
                let start = i;
                i += 1;
                while i < len && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < len {
                        i += 1; // Skip escaped character.
                    }
                    i += 1;
                }
                if i < len {
                    i += 1; // Closing quote.
                }
                tokens.push(HighlightToken {
                    range: start..i,
                    token_type: TokenType::String,
                });
                continue;
            }

            // Single-quoted string.
            if ch == '\'' {
                let start = i;
                i += 1;
                while i < len && chars[i] != '\'' {
                    if chars[i] == '\\' && i + 1 < len {
                        i += 1;
                    }
                    i += 1;
                }
                if i < len {
                    i += 1;
                }
                tokens.push(HighlightToken {
                    range: start..i,
                    token_type: TokenType::String,
                });
                continue;
            }

            // Number.
            if ch.is_ascii_digit() || (ch == '.' && i + 1 < len && chars[i + 1].is_ascii_digit()) {
                let start = i;
                // Hex prefix.
                if ch == '0' && i + 1 < len && (chars[i + 1] == 'x' || chars[i + 1] == 'X') {
                    i += 2;
                    while i < len && chars[i].is_ascii_hexdigit() {
                        i += 1;
                    }
                } else {
                    while i < len && (chars[i].is_ascii_digit() || chars[i] == '.') {
                        i += 1;
                    }
                    // Exponent.
                    if i < len && (chars[i] == 'e' || chars[i] == 'E') {
                        i += 1;
                        if i < len && (chars[i] == '+' || chars[i] == '-') {
                            i += 1;
                        }
                        while i < len && chars[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                    // Type suffix (f32, u64, etc.).
                    if i < len && chars[i].is_ascii_alphabetic() {
                        while i < len && chars[i].is_ascii_alphanumeric() {
                            i += 1;
                        }
                    }
                }
                tokens.push(HighlightToken {
                    range: start..i,
                    token_type: TokenType::Number,
                });
                continue;
            }

            // Identifier / keyword.
            if ch.is_alphabetic() || ch == '_' {
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();

                // Check if followed by '(' (function call).
                let is_fn = i < len && chars[i] == '(';
                // Check if followed by '!' (macro).
                let is_macro = i < len && chars[i] == '!';

                let token_type = if keywords.iter().any(|k| k == &word) {
                    TokenType::Keyword
                } else if types.iter().any(|t| t == &word) {
                    TokenType::Type
                } else if is_macro {
                    TokenType::Macro
                } else if is_fn {
                    TokenType::Function
                } else if word.chars().next().map_or(false, |c| c.is_uppercase()) {
                    TokenType::Type
                } else {
                    TokenType::Identifier
                };

                tokens.push(HighlightToken {
                    range: start..i,
                    token_type,
                });
                continue;
            }

            // Attribute (e.g., #[...] in Rust, @... in WGSL).
            if ch == '#' || ch == '@' {
                let start = i;
                i += 1;
                if i < len && chars[i] == '[' {
                    while i < len && chars[i] != ']' {
                        i += 1;
                    }
                    if i < len {
                        i += 1;
                    }
                } else {
                    while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                        i += 1;
                    }
                }
                tokens.push(HighlightToken {
                    range: start..i,
                    token_type: TokenType::Attribute,
                });
                continue;
            }

            // Operator.
            if "+-*/%=<>!&|^~?".contains(ch) {
                let start = i;
                i += 1;
                // Two-character operators.
                if i < len {
                    let two: String = [ch, chars[i]].iter().collect();
                    if ["==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "+=", "-=",
                        "*=", "/=", "%=", "&=", "|=", "^=", "->", "=>"]
                        .contains(&two.as_str())
                    {
                        i += 1;
                    }
                }
                tokens.push(HighlightToken {
                    range: start..i,
                    token_type: TokenType::Operator,
                });
                continue;
            }

            // Punctuation.
            if "(){}[];:,.".contains(ch) {
                tokens.push(HighlightToken {
                    range: i..i + 1,
                    token_type: TokenType::Punctuation,
                });
                i += 1;
                continue;
            }

            // Unknown character.
            tokens.push(HighlightToken {
                range: i..i + 1,
                token_type: TokenType::Unknown,
            });
            i += 1;
        }

        tokens
    }

    /// Highlight text and return (range, color) pairs.
    pub fn highlight_colored(
        &self,
        text: &str,
        language: HighlightLanguage,
    ) -> Vec<(Range<usize>, Color)> {
        self.highlight(text, language)
            .into_iter()
            .filter(|t| t.token_type != TokenType::Whitespace)
            .map(|t| (t.range, t.token_type.default_color()))
            .collect()
    }
}

impl Default for SyntaxHighlighter {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// 8. Clipboard Integration
// =========================================================================

/// Clipboard integration for copy/paste text.
///
/// The actual platform clipboard is accessed via callbacks; this struct
/// provides the API and a fallback in-process clipboard.
#[derive(Debug, Clone)]
pub struct ClipboardIntegration {
    /// Internal fallback clipboard (used when no platform callback is set).
    internal_clipboard: String,
    /// Whether platform clipboard access is available.
    pub platform_available: bool,
}

impl ClipboardIntegration {
    pub fn new() -> Self {
        Self {
            internal_clipboard: String::new(),
            platform_available: false,
        }
    }

    /// Copy text to the clipboard.
    pub fn copy(&mut self, text: &str) {
        self.internal_clipboard = text.to_string();
        // In production, this would call the platform clipboard API.
    }

    /// Get text from the clipboard.
    pub fn paste(&self) -> String {
        // In production, this would call the platform clipboard API.
        self.internal_clipboard.clone()
    }

    /// Check if the clipboard has text.
    pub fn has_text(&self) -> bool {
        !self.internal_clipboard.is_empty()
    }

    /// Clear the clipboard.
    pub fn clear(&mut self) {
        self.internal_clipboard.clear();
    }

    /// Set the internal clipboard directly (for testing / fallback).
    pub fn set_internal(&mut self, text: &str) {
        self.internal_clipboard = text.to_string();
    }
}

impl Default for ClipboardIntegration {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// 9. Undo Stack for Text Editing
// =========================================================================

/// Type of text edit operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextEditOperation {
    /// Insert text at a position.
    Insert {
        offset: usize,
        text: String,
    },
    /// Delete text at a range.
    Delete {
        offset: usize,
        text: String,
    },
    /// Replace text at a range.
    Replace {
        offset: usize,
        old_text: String,
        new_text: String,
    },
}

impl TextEditOperation {
    /// Reverse the operation (for undo).
    pub fn inverse(&self) -> Self {
        match self {
            TextEditOperation::Insert { offset, text } => TextEditOperation::Delete {
                offset: *offset,
                text: text.clone(),
            },
            TextEditOperation::Delete { offset, text } => TextEditOperation::Insert {
                offset: *offset,
                text: text.clone(),
            },
            TextEditOperation::Replace {
                offset,
                old_text,
                new_text,
            } => TextEditOperation::Replace {
                offset: *offset,
                old_text: new_text.clone(),
                new_text: old_text.clone(),
            },
        }
    }

    /// Check if this operation can be merged with another (for consecutive
    /// typing).
    pub fn can_merge(&self, other: &TextEditOperation) -> bool {
        match (self, other) {
            (
                TextEditOperation::Insert {
                    offset: o1,
                    text: t1,
                },
                TextEditOperation::Insert {
                    offset: o2,
                    text: t2,
                },
            ) => {
                // Merge consecutive single-character inserts.
                *o2 == *o1 + t1.len()
                    && t2.len() == 1
                    && t1.len() < 100
                    && !t2.starts_with(' ')
                    && !t2.starts_with('\n')
            }
            (
                TextEditOperation::Delete {
                    offset: o1,
                    text: t1,
                },
                TextEditOperation::Delete {
                    offset: o2,
                    text: t2,
                },
            ) => {
                // Merge consecutive backspace deletes.
                (*o2 + t2.len() == *o1 || *o2 == *o1) && t2.len() == 1 && t1.len() < 100
            }
            _ => false,
        }
    }

    /// Merge another operation into this one.
    pub fn merge(&mut self, other: &TextEditOperation) {
        match (self, other) {
            (
                TextEditOperation::Insert { text, .. },
                TextEditOperation::Insert { text: t2, .. },
            ) => {
                text.push_str(t2);
            }
            (
                TextEditOperation::Delete { offset, text },
                TextEditOperation::Delete {
                    offset: o2,
                    text: t2,
                },
            ) => {
                if *o2 + t2.len() == *offset {
                    // Backspace: prepend.
                    let mut new_text = t2.clone();
                    new_text.push_str(text);
                    *text = new_text;
                    *offset = *o2;
                } else {
                    // Forward delete: append.
                    text.push_str(t2);
                }
            }
            _ => {}
        }
    }
}

/// Per-field undo stack with merge support for consecutive typing.
#[derive(Debug, Clone)]
pub struct TextUndoStack {
    /// Undo history.
    pub undo_stack: Vec<TextEditOperation>,
    /// Redo history.
    pub redo_stack: Vec<TextEditOperation>,
    /// Maximum number of undo levels.
    pub max_levels: usize,
    /// Whether the document is considered "dirty" (has unsaved changes).
    pub dirty: bool,
    /// The undo level at which the document was last saved.
    pub save_point: Option<usize>,
}

impl TextUndoStack {
    pub fn new() -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_levels: 100,
            dirty: false,
            save_point: Some(0),
        }
    }

    /// Push a new edit operation.
    pub fn push(&mut self, op: TextEditOperation) {
        // Try to merge with the last operation.
        if let Some(last) = self.undo_stack.last_mut() {
            if last.can_merge(&op) {
                last.merge(&op);
                self.redo_stack.clear();
                self.dirty = true;
                return;
            }
        }

        self.undo_stack.push(op);
        self.redo_stack.clear();
        self.dirty = true;

        // Enforce max levels.
        while self.undo_stack.len() > self.max_levels {
            self.undo_stack.remove(0);
        }
    }

    /// Undo the last operation. Returns the inverse operation to apply to
    /// the text.
    pub fn undo(&mut self) -> Option<TextEditOperation> {
        if let Some(op) = self.undo_stack.pop() {
            let inverse = op.inverse();
            self.redo_stack.push(op);
            self.dirty = self.save_point != Some(self.undo_stack.len());
            Some(inverse)
        } else {
            None
        }
    }

    /// Redo the last undone operation.
    pub fn redo(&mut self) -> Option<TextEditOperation> {
        if let Some(op) = self.redo_stack.pop() {
            let result = op.clone();
            self.undo_stack.push(op);
            self.dirty = self.save_point != Some(self.undo_stack.len());
            Some(result)
        } else {
            None
        }
    }

    /// Mark the current state as the save point.
    pub fn mark_saved(&mut self) {
        self.save_point = Some(self.undo_stack.len());
        self.dirty = false;
    }

    /// Check if undo is available.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Check if redo is available.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Clear all undo/redo history.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.save_point = Some(0);
        self.dirty = false;
    }

    /// Apply an operation to a text string and return the new text.
    pub fn apply_operation(text: &str, op: &TextEditOperation) -> String {
        match op {
            TextEditOperation::Insert { offset, text: ins } => {
                let offset = (*offset).min(text.len());
                let mut result = text.to_string();
                result.insert_str(offset, ins);
                result
            }
            TextEditOperation::Delete { offset, text: del } => {
                let offset = (*offset).min(text.len());
                let end = (offset + del.len()).min(text.len());
                let mut result = text.to_string();
                result.drain(offset..end);
                result
            }
            TextEditOperation::Replace {
                offset,
                old_text,
                new_text,
            } => {
                let offset = (*offset).min(text.len());
                let end = (offset + old_text.len()).min(text.len());
                let mut result = text.to_string();
                result.replace_range(offset..end, new_text);
                result
            }
        }
    }
}

impl Default for TextUndoStack {
    fn default() -> Self {
        Self::new()
    }
}
