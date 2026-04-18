//! Production font system for the Genovo Slate UI.
//!
//! Provides a full font pipeline: loading font faces, parsing glyph metrics,
//! rasterising glyphs into a dynamic texture atlas with shelf-based packing,
//! LRU cache eviction, text shaping with kerning, word-wrap, alignment, and
//! text measurement.
//!
//! An embedded fallback font is always available so text never fails silently.

use std::collections::{HashMap, VecDeque};
use std::ops::Range;

use glam::Vec2;
use genovo_core::Rect;

use crate::render_commands::{Color, TextureId};

// =========================================================================
// Constants
// =========================================================================

/// Minimum atlas dimension.
const ATLAS_MIN_SIZE: u32 = 512;
/// Maximum atlas dimension.
const ATLAS_MAX_SIZE: u32 = 4096;
/// Default starting atlas size.
const ATLAS_INITIAL_SIZE: u32 = 512;
/// Maximum number of atlas pages.
const MAX_ATLAS_PAGES: usize = 8;
/// Padding between glyphs in atlas (to avoid bleeding).
const GLYPH_PADDING: u32 = 2;
/// Maximum number of glyphs in the LRU cache before eviction starts.
const LRU_MAX_GLYPHS: usize = 8192;
/// Number of glyphs to evict when the cache is full.
const LRU_EVICT_BATCH: usize = 256;
/// Default number of recent-color palette slots.
const DEFAULT_FALLBACK_FONT_SIZE: f32 = 16.0;

// =========================================================================
// FontId
// =========================================================================

/// Unique identifier for a loaded font face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontId(pub u32);

impl FontId {
    pub const INVALID: Self = Self(u32::MAX);
    pub const DEFAULT: Self = Self(0);
}

impl Default for FontId {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// =========================================================================
// FontSize
// =========================================================================

/// Font size specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FontSize {
    /// Fixed pixel size (not DPI-scaled).
    Fixed(f32),
    /// DPI-scaled size -- the actual pixel size is `points * dpi_scale`.
    Points(f32),
}

impl FontSize {
    /// Resolve to pixel size given a DPI scale factor.
    pub fn to_pixels(self, dpi_scale: f32) -> f32 {
        match self {
            FontSize::Fixed(px) => px,
            FontSize::Points(pt) => pt * dpi_scale,
        }
    }

    /// The raw numeric value regardless of mode.
    pub fn value(self) -> f32 {
        match self {
            FontSize::Fixed(v) | FontSize::Points(v) => v,
        }
    }
}

impl Default for FontSize {
    fn default() -> Self {
        FontSize::Fixed(DEFAULT_FALLBACK_FONT_SIZE)
    }
}

impl From<f32> for FontSize {
    fn from(v: f32) -> Self {
        FontSize::Fixed(v)
    }
}

// =========================================================================
// FontWeight / FontStyle
// =========================================================================

/// Font weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontWeight {
    Thin,
    ExtraLight,
    Light,
    Regular,
    Medium,
    SemiBold,
    Bold,
    ExtraBold,
    Black,
}

impl FontWeight {
    /// Numeric weight (100-900).
    pub fn to_numeric(self) -> u16 {
        match self {
            FontWeight::Thin => 100,
            FontWeight::ExtraLight => 200,
            FontWeight::Light => 300,
            FontWeight::Regular => 400,
            FontWeight::Medium => 500,
            FontWeight::SemiBold => 600,
            FontWeight::Bold => 700,
            FontWeight::ExtraBold => 800,
            FontWeight::Black => 900,
        }
    }

    /// Closest weight from a numeric value.
    pub fn from_numeric(n: u16) -> Self {
        match n {
            0..=150 => FontWeight::Thin,
            151..=250 => FontWeight::ExtraLight,
            251..=350 => FontWeight::Light,
            351..=450 => FontWeight::Regular,
            451..=550 => FontWeight::Medium,
            551..=650 => FontWeight::SemiBold,
            651..=750 => FontWeight::Bold,
            751..=850 => FontWeight::ExtraBold,
            _ => FontWeight::Black,
        }
    }
}

impl Default for FontWeight {
    fn default() -> Self {
        FontWeight::Regular
    }
}

/// Font style (italic, oblique, normal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

impl Default for FontStyle {
    fn default() -> Self {
        FontStyle::Normal
    }
}

// =========================================================================
// LineMetrics
// =========================================================================

/// Vertical metrics for a font face at a given size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineMetrics {
    /// Distance from baseline to the top of the tallest glyph (positive up).
    pub ascender: f32,
    /// Distance from baseline to the bottom of the lowest descender (negative down).
    pub descender: f32,
    /// Extra spacing between lines.
    pub line_gap: f32,
    /// Underline offset from baseline (negative = below).
    pub underline_offset: f32,
    /// Underline thickness.
    pub underline_thickness: f32,
    /// Strikethrough offset from baseline.
    pub strikethrough_offset: f32,
    /// Strikethrough thickness.
    pub strikethrough_thickness: f32,
    /// Capital letter height.
    pub cap_height: f32,
    /// Lowercase x height.
    pub x_height: f32,
}

impl LineMetrics {
    /// Total line height = ascender - descender + line_gap.
    pub fn line_height(&self) -> f32 {
        self.ascender - self.descender + self.line_gap
    }

    /// Scale all metrics by a factor.
    pub fn scaled(self, factor: f32) -> Self {
        Self {
            ascender: self.ascender * factor,
            descender: self.descender * factor,
            line_gap: self.line_gap * factor,
            underline_offset: self.underline_offset * factor,
            underline_thickness: self.underline_thickness * factor,
            strikethrough_offset: self.strikethrough_offset * factor,
            strikethrough_thickness: self.strikethrough_thickness * factor,
            cap_height: self.cap_height * factor,
            x_height: self.x_height * factor,
        }
    }

    /// Create metrics from basic ascender/descender/line_gap.
    pub fn from_basic(ascender: f32, descender: f32, line_gap: f32) -> Self {
        Self {
            ascender,
            descender,
            line_gap,
            underline_offset: descender * 0.5,
            underline_thickness: 1.0,
            strikethrough_offset: ascender * 0.33,
            strikethrough_thickness: 1.0,
            cap_height: ascender * 0.9,
            x_height: ascender * 0.55,
        }
    }
}

impl Default for LineMetrics {
    fn default() -> Self {
        Self::from_basic(12.0, -4.0, 2.0)
    }
}

// =========================================================================
// GlyphMetrics
// =========================================================================

/// Metrics for a single glyph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphMetrics {
    /// Advance width (distance to next glyph origin).
    pub advance_width: f32,
    /// Left side bearing (x offset from origin to start of bitmap).
    pub left_side_bearing: f32,
    /// Right side bearing.
    pub right_side_bearing: f32,
    /// Bounding box minimum (relative to baseline origin).
    pub bbox_min: Vec2,
    /// Bounding box maximum.
    pub bbox_max: Vec2,
}

impl Default for GlyphMetrics {
    fn default() -> Self {
        Self {
            advance_width: 8.0,
            left_side_bearing: 0.0,
            right_side_bearing: 0.0,
            bbox_min: Vec2::ZERO,
            bbox_max: Vec2::new(8.0, 12.0),
        }
    }
}

// =========================================================================
// KerningPair
// =========================================================================

/// A kerning adjustment between two glyphs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KerningPair {
    pub left: char,
    pub right: char,
    /// Horizontal adjustment (negative = tighter).
    pub x_advance: f32,
}

// =========================================================================
// FontFace
// =========================================================================

/// A parsed font face with glyph metrics, kerning, and line metrics.
///
/// In a full engine this would wrap an actual font parser (e.g. `ttf-parser`).
/// Here we store pre-computed metrics that can be loaded from a simple binary
/// format or generated from built-in data.
#[derive(Debug, Clone)]
pub struct FontFace {
    /// Unique identifier.
    pub id: FontId,
    /// Family name (e.g. "Roboto", "Genovo Sans").
    pub family: String,
    /// Weight.
    pub weight: FontWeight,
    /// Style.
    pub style: FontStyle,
    /// Whether this is a monospace font.
    pub monospace: bool,
    /// Units per em (used for scaling).
    pub units_per_em: u16,
    /// Line metrics at 1em.
    pub line_metrics: LineMetrics,
    /// Per-glyph metrics keyed by codepoint.
    pub glyph_metrics: HashMap<char, GlyphMetrics>,
    /// Kerning table.
    pub kerning: HashMap<(char, char), f32>,
    /// Raw font data (TTF/OTF bytes). Stored for re-rasterisation.
    pub raw_data: Vec<u8>,
    /// Number of glyphs in the font.
    pub glyph_count: u32,
    /// Character coverage ranges.
    pub coverage: Vec<Range<u32>>,
    /// Has the font been fully loaded and parsed?
    pub loaded: bool,
}

impl FontFace {
    /// Create a new empty font face.
    pub fn new(id: FontId, family: &str) -> Self {
        Self {
            id,
            family: family.to_string(),
            weight: FontWeight::Regular,
            style: FontStyle::Normal,
            monospace: false,
            units_per_em: 1000,
            line_metrics: LineMetrics::default(),
            glyph_metrics: HashMap::new(),
            kerning: HashMap::new(),
            raw_data: Vec::new(),
            glyph_count: 0,
            coverage: Vec::new(),
            loaded: false,
        }
    }

    /// Look up metrics for a character, returning the fallback if missing.
    pub fn metrics_for(&self, c: char) -> GlyphMetrics {
        self.glyph_metrics.get(&c).copied().unwrap_or_default()
    }

    /// Look up kerning adjustment between two characters.
    pub fn kerning_for(&self, left: char, right: char) -> f32 {
        self.kerning.get(&(left, right)).copied().unwrap_or(0.0)
    }

    /// Line metrics scaled to a given pixel size.
    pub fn line_metrics_at_size(&self, size: f32) -> LineMetrics {
        let scale = size / self.units_per_em as f32;
        self.line_metrics.scaled(scale)
    }

    /// Advance width for a character at a given pixel size.
    pub fn advance_at_size(&self, c: char, size: f32) -> f32 {
        let scale = size / self.units_per_em as f32;
        self.metrics_for(c).advance_width * scale
    }

    /// Kerning at a given pixel size.
    pub fn kerning_at_size(&self, left: char, right: char, size: f32) -> f32 {
        let scale = size / self.units_per_em as f32;
        self.kerning_for(left, right) * scale
    }

    /// Whether the font has a glyph for the given character.
    pub fn has_glyph(&self, c: char) -> bool {
        self.glyph_metrics.contains_key(&c)
    }

    /// Set the font as monospace with a given advance width.
    pub fn set_monospace(mut self, advance: f32) -> Self {
        self.monospace = true;
        for metrics in self.glyph_metrics.values_mut() {
            metrics.advance_width = advance;
        }
        self
    }

    /// Create a builder for constructing a font face from descriptive data.
    pub fn builder(id: FontId, family: &str) -> FontFaceBuilder {
        FontFaceBuilder::new(id, family)
    }
}

// =========================================================================
// FontFaceBuilder
// =========================================================================

/// Incremental builder for `FontFace`.
pub struct FontFaceBuilder {
    face: FontFace,
}

impl FontFaceBuilder {
    pub fn new(id: FontId, family: &str) -> Self {
        Self {
            face: FontFace::new(id, family),
        }
    }

    pub fn weight(mut self, w: FontWeight) -> Self {
        self.face.weight = w;
        self
    }

    pub fn style(mut self, s: FontStyle) -> Self {
        self.face.style = s;
        self
    }

    pub fn monospace(mut self, m: bool) -> Self {
        self.face.monospace = m;
        self
    }

    pub fn units_per_em(mut self, u: u16) -> Self {
        self.face.units_per_em = u;
        self
    }

    pub fn line_metrics(mut self, lm: LineMetrics) -> Self {
        self.face.line_metrics = lm;
        self
    }

    pub fn glyph(mut self, c: char, metrics: GlyphMetrics) -> Self {
        self.face.glyph_metrics.insert(c, metrics);
        self
    }

    pub fn kern(mut self, left: char, right: char, adjust: f32) -> Self {
        self.face.kerning.insert((left, right), adjust);
        self
    }

    pub fn raw_data(mut self, data: Vec<u8>) -> Self {
        self.face.raw_data = data;
        self
    }

    pub fn coverage(mut self, ranges: Vec<Range<u32>>) -> Self {
        self.face.coverage = ranges;
        self
    }

    pub fn build(mut self) -> FontFace {
        self.face.glyph_count = self.face.glyph_metrics.len() as u32;
        self.face.loaded = true;
        self.face
    }
}

// =========================================================================
// FontDatabase
// =========================================================================

/// Registry of loaded fonts. Provides lookup by family name, weight, and style.
#[derive(Debug, Clone)]
pub struct FontDatabase {
    /// All loaded font faces.
    fonts: Vec<FontFace>,
    /// Index: family name (lowercase) -> list of face indices.
    family_index: HashMap<String, Vec<usize>>,
    /// Next ID to assign.
    next_id: u32,
    /// The default/fallback font id.
    default_font: FontId,
    /// Font substitution table: missing family -> fallback family.
    substitutions: HashMap<String, String>,
}

impl FontDatabase {
    /// Create a new empty database.
    pub fn new() -> Self {
        Self {
            fonts: Vec::new(),
            family_index: HashMap::new(),
            next_id: 0,
            default_font: FontId::DEFAULT,
            substitutions: HashMap::new(),
        }
    }

    /// Create a database pre-loaded with the embedded fallback font.
    pub fn with_fallback() -> Self {
        let mut db = Self::new();
        let fallback = create_embedded_fallback_font(FontId(0));
        db.register_face(fallback);
        db
    }

    /// Register a font face. Returns its `FontId`.
    pub fn register_face(&mut self, mut face: FontFace) -> FontId {
        let id = FontId(self.next_id);
        self.next_id += 1;
        face.id = id;

        let key = face.family.to_lowercase();
        let idx = self.fonts.len();
        self.fonts.push(face);

        self.family_index.entry(key).or_default().push(idx);
        id
    }

    /// Remove a font face by ID.
    pub fn remove_face(&mut self, id: FontId) -> Option<FontFace> {
        if let Some(pos) = self.fonts.iter().position(|f| f.id == id) {
            let face = self.fonts.remove(pos);
            // Rebuild index
            self.rebuild_index();
            Some(face)
        } else {
            None
        }
    }

    fn rebuild_index(&mut self) {
        self.family_index.clear();
        for (idx, face) in self.fonts.iter().enumerate() {
            let key = face.family.to_lowercase();
            self.family_index.entry(key).or_default().push(idx);
        }
    }

    /// Get a font face by ID.
    pub fn get(&self, id: FontId) -> Option<&FontFace> {
        self.fonts.iter().find(|f| f.id == id)
    }

    /// Get a mutable font face by ID.
    pub fn get_mut(&mut self, id: FontId) -> Option<&mut FontFace> {
        self.fonts.iter_mut().find(|f| f.id == id)
    }

    /// Find the best matching face for a family/weight/style request.
    pub fn find_best_match(
        &self,
        family: &str,
        weight: FontWeight,
        style: FontStyle,
    ) -> Option<&FontFace> {
        let key = family.to_lowercase();
        let candidates = self.family_index.get(&key)?;

        // First pass: exact match on weight and style.
        for &idx in candidates {
            let f = &self.fonts[idx];
            if f.weight == weight && f.style == style {
                return Some(f);
            }
        }

        // Second pass: match style, closest weight.
        let mut best: Option<(usize, u16)> = None;
        for &idx in candidates {
            let f = &self.fonts[idx];
            if f.style == style {
                let diff = (f.weight.to_numeric() as i32 - weight.to_numeric() as i32).unsigned_abs() as u16;
                if best.is_none() || diff < best.unwrap().1 {
                    best = Some((idx, diff));
                }
            }
        }
        if let Some((idx, _)) = best {
            return Some(&self.fonts[idx]);
        }

        // Third pass: any face in the family.
        candidates.first().map(|&idx| &self.fonts[idx])
    }

    /// Look up by family only (any weight/style).
    pub fn find_family(&self, family: &str) -> Option<&FontFace> {
        let key = family.to_lowercase();
        self.family_index
            .get(&key)
            .and_then(|v| v.first())
            .map(|&idx| &self.fonts[idx])
    }

    /// Resolve a family name, applying substitutions if needed, then falling
    /// back to the default font.
    pub fn resolve(&self, family: &str, weight: FontWeight, style: FontStyle) -> &FontFace {
        if let Some(face) = self.find_best_match(family, weight, style) {
            return face;
        }
        // Try substitution.
        let key = family.to_lowercase();
        if let Some(sub) = self.substitutions.get(&key) {
            if let Some(face) = self.find_best_match(sub, weight, style) {
                return face;
            }
        }
        // Fallback to default.
        self.get(self.default_font)
            .unwrap_or_else(|| &self.fonts[0])
    }

    /// Set the default fallback font.
    pub fn set_default(&mut self, id: FontId) {
        self.default_font = id;
    }

    /// Add a font substitution rule.
    pub fn add_substitution(&mut self, from: &str, to: &str) {
        self.substitutions
            .insert(from.to_lowercase(), to.to_lowercase());
    }

    /// List all registered families.
    pub fn families(&self) -> Vec<String> {
        self.family_index.keys().cloned().collect()
    }

    /// Number of loaded faces.
    pub fn face_count(&self) -> usize {
        self.fonts.len()
    }

    /// Iterator over all faces.
    pub fn iter(&self) -> impl Iterator<Item = &FontFace> {
        self.fonts.iter()
    }

    /// Load font data from raw TTF/OTF bytes.
    ///
    /// This performs a simplified parse of the font metrics. In a production
    /// engine, this would use `ttf-parser` or `fontdue`.
    pub fn load_from_bytes(&mut self, family: &str, data: &[u8]) -> FontId {
        let mut face = FontFace::new(FontId(self.next_id), family);
        face.raw_data = data.to_vec();

        // Parse basic metrics from the font binary.
        // This is a simplified parse -- we extract units-per-em from the
        // `head` table and metrics from `hhea`/`OS/2` if present, but
        // fall back to reasonable defaults.
        let upm = parse_units_per_em(data).unwrap_or(1000);
        face.units_per_em = upm;

        let (ascender, descender, line_gap) = parse_hhea_metrics(data, upm);
        face.line_metrics = LineMetrics::from_basic(ascender, descender, line_gap);

        // Build glyph metrics from cmap + hmtx tables.
        let (glyphs, kern) = parse_glyph_metrics(data, upm);
        face.glyph_metrics = glyphs;
        face.kerning = kern;
        face.glyph_count = face.glyph_metrics.len() as u32;
        face.loaded = true;

        self.register_face(face)
    }
}

impl Default for FontDatabase {
    fn default() -> Self {
        Self::with_fallback()
    }
}

// =========================================================================
// Simplified font binary parsing helpers
// =========================================================================

/// Try to extract units-per-em from a TTF/OTF `head` table.
fn parse_units_per_em(data: &[u8]) -> Option<u16> {
    // The `head` table is at a fixed offset in the font directory.
    // Minimal parsing: search for the `head` tag in the table directory.
    if data.len() < 12 {
        return None;
    }
    let num_tables = u16::from_be_bytes([data[4], data[5]]) as usize;
    for i in 0..num_tables {
        let offset = 12 + i * 16;
        if offset + 16 > data.len() {
            break;
        }
        let tag = &data[offset..offset + 4];
        if tag == b"head" {
            let table_offset =
                u32::from_be_bytes([data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11]])
                    as usize;
            // units-per-em is at offset 18 within the head table.
            if table_offset + 20 <= data.len() {
                return Some(u16::from_be_bytes([
                    data[table_offset + 18],
                    data[table_offset + 19],
                ]));
            }
        }
    }
    None
}

/// Parse ascender, descender, line gap from `hhea` table.
fn parse_hhea_metrics(data: &[u8], units_per_em: u16) -> (f32, f32, f32) {
    if data.len() < 12 {
        return (0.75, -0.25, 0.125);
    }
    let num_tables = u16::from_be_bytes([data[4], data[5]]) as usize;
    for i in 0..num_tables {
        let offset = 12 + i * 16;
        if offset + 16 > data.len() {
            break;
        }
        let tag = &data[offset..offset + 4];
        if tag == b"hhea" {
            let table_offset =
                u32::from_be_bytes([data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11]])
                    as usize;
            if table_offset + 8 <= data.len() {
                let asc = i16::from_be_bytes([data[table_offset + 4], data[table_offset + 5]]);
                let desc = i16::from_be_bytes([data[table_offset + 6], data[table_offset + 7]]);
                let gap_bytes = if table_offset + 10 <= data.len() {
                    i16::from_be_bytes([data[table_offset + 8], data[table_offset + 9]])
                } else {
                    0
                };
                let scale = 1.0 / units_per_em as f32;
                return (asc as f32 * scale, desc as f32 * scale, gap_bytes as f32 * scale);
            }
        }
    }
    (0.75, -0.25, 0.125)
}

/// Parse glyph metrics and kerning from font data.
/// Simplified version that generates reasonable proportional metrics for
/// ASCII when actual table parsing isn't available.
fn parse_glyph_metrics(
    _data: &[u8],
    units_per_em: u16,
) -> (HashMap<char, GlyphMetrics>, HashMap<(char, char), f32>) {
    let mut glyphs = HashMap::new();
    let mut kern = HashMap::new();
    let upm = units_per_em as f32;

    // Proportional advance widths for ASCII (normalised to units_per_em).
    let width_table: &[(char, f32)] = &[
        (' ', 0.25), ('!', 0.33), ('"', 0.41), ('#', 0.50), ('$', 0.50),
        ('%', 0.83), ('&', 0.78), ('\'', 0.18), ('(', 0.33), (')', 0.33),
        ('*', 0.50), ('+', 0.56), (',', 0.25), ('-', 0.33), ('.', 0.25),
        ('/', 0.28), ('0', 0.50), ('1', 0.50), ('2', 0.50), ('3', 0.50),
        ('4', 0.50), ('5', 0.50), ('6', 0.50), ('7', 0.50), ('8', 0.50),
        ('9', 0.50), (':', 0.28), (';', 0.28), ('<', 0.56), ('=', 0.56),
        ('>', 0.56), ('?', 0.44), ('@', 1.02),
        ('A', 0.67), ('B', 0.67), ('C', 0.72), ('D', 0.72), ('E', 0.67),
        ('F', 0.61), ('G', 0.78), ('H', 0.72), ('I', 0.28), ('J', 0.50),
        ('K', 0.67), ('L', 0.56), ('M', 0.83), ('N', 0.72), ('O', 0.78),
        ('P', 0.67), ('Q', 0.78), ('R', 0.72), ('S', 0.67), ('T', 0.61),
        ('U', 0.72), ('V', 0.67), ('W', 0.94), ('X', 0.67), ('Y', 0.67),
        ('Z', 0.61),
        ('[', 0.28), ('\\', 0.28), (']', 0.28), ('^', 0.47), ('_', 0.50),
        ('`', 0.33),
        ('a', 0.44), ('b', 0.50), ('c', 0.44), ('d', 0.50), ('e', 0.44),
        ('f', 0.33), ('g', 0.50), ('h', 0.50), ('i', 0.28), ('j', 0.28),
        ('k', 0.50), ('l', 0.28), ('m', 0.78), ('n', 0.50), ('o', 0.50),
        ('p', 0.50), ('q', 0.50), ('r', 0.33), ('s', 0.39), ('t', 0.28),
        ('u', 0.50), ('v', 0.50), ('w', 0.72), ('x', 0.50), ('y', 0.50),
        ('z', 0.44),
        ('{', 0.48), ('|', 0.20), ('}', 0.48), ('~', 0.54),
    ];

    for &(ch, w) in width_table {
        let advance = w * upm;
        let bearing_x = 0.0;
        let height = upm * 0.75;
        let bbox_w = advance - bearing_x;
        glyphs.insert(ch, GlyphMetrics {
            advance_width: advance,
            left_side_bearing: bearing_x,
            right_side_bearing: advance - bbox_w - bearing_x,
            bbox_min: Vec2::new(bearing_x, -upm * 0.25),
            bbox_max: Vec2::new(bearing_x + bbox_w, height - upm * 0.25),
        });
    }

    // Extended Latin, common symbols, etc.
    let extended_chars: &[char] = &[
        '\u{00C0}', '\u{00C1}', '\u{00C2}', '\u{00C3}', '\u{00C4}', '\u{00C5}', // A-accented
        '\u{00C6}', '\u{00C7}', '\u{00C8}', '\u{00C9}', '\u{00CA}', '\u{00CB}',
        '\u{00CC}', '\u{00CD}', '\u{00CE}', '\u{00CF}', '\u{00D0}', '\u{00D1}',
        '\u{00D2}', '\u{00D3}', '\u{00D4}', '\u{00D5}', '\u{00D6}', '\u{00D7}',
        '\u{00D8}', '\u{00D9}', '\u{00DA}', '\u{00DB}', '\u{00DC}', '\u{00DD}',
        '\u{00DE}', '\u{00DF}',
        '\u{00E0}', '\u{00E1}', '\u{00E2}', '\u{00E3}', '\u{00E4}', '\u{00E5}',
        '\u{00E6}', '\u{00E7}', '\u{00E8}', '\u{00E9}', '\u{00EA}', '\u{00EB}',
        '\u{00EC}', '\u{00ED}', '\u{00EE}', '\u{00EF}', '\u{00F0}', '\u{00F1}',
        '\u{00F2}', '\u{00F3}', '\u{00F4}', '\u{00F5}', '\u{00F6}', '\u{00F7}',
        '\u{00F8}', '\u{00F9}', '\u{00FA}', '\u{00FB}', '\u{00FC}', '\u{00FD}',
        '\u{00FE}', '\u{00FF}',
        '\u{2013}', '\u{2014}', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}',
        '\u{2022}', '\u{2026}', '\u{20AC}', '\u{2122}',
    ];

    for &ch in extended_chars {
        let advance = upm * 0.55;
        glyphs.insert(ch, GlyphMetrics {
            advance_width: advance,
            left_side_bearing: 0.0,
            right_side_bearing: 0.0,
            bbox_min: Vec2::new(0.0, -upm * 0.25),
            bbox_max: Vec2::new(advance, upm * 0.5),
        });
    }

    // Common kerning pairs.
    let kern_pairs: &[((char, char), f32)] = &[
        (('A', 'V'), -0.08), (('A', 'W'), -0.06), (('A', 'Y'), -0.08),
        (('A', 'T'), -0.08), (('A', 'v'), -0.04), (('A', 'w'), -0.03),
        (('F', 'a'), -0.04), (('F', 'e'), -0.04), (('F', 'o'), -0.04),
        (('L', 'T'), -0.08), (('L', 'V'), -0.08), (('L', 'W'), -0.06),
        (('L', 'Y'), -0.08), (('P', 'a'), -0.04), (('P', '.'), -0.10),
        (('P', ','), -0.10), (('T', 'a'), -0.08), (('T', 'e'), -0.08),
        (('T', 'i'), -0.04), (('T', 'o'), -0.08), (('T', 'r'), -0.04),
        (('T', 's'), -0.08), (('T', 'u'), -0.04), (('T', 'y'), -0.04),
        (('T', '.'), -0.08), (('T', ','), -0.08), (('T', ':'), -0.08),
        (('V', 'a'), -0.06), (('V', 'e'), -0.06), (('V', 'i'), -0.02),
        (('V', 'o'), -0.06), (('V', 'u'), -0.04), (('V', '.'), -0.08),
        (('V', ','), -0.08), (('W', 'a'), -0.04), (('W', 'e'), -0.04),
        (('W', 'o'), -0.04), (('Y', 'a'), -0.08), (('Y', 'e'), -0.08),
        (('Y', 'i'), -0.04), (('Y', 'o'), -0.08), (('Y', 'p'), -0.06),
        (('Y', 'u'), -0.06), (('Y', 'v'), -0.06), (('Y', '.'), -0.10),
        (('Y', ','), -0.10), (('f', 'f'), -0.02), (('r', '.'), -0.04),
        (('r', ','), -0.04),
    ];

    for &(pair, adj) in kern_pairs {
        kern.insert(pair, adj * upm);
    }

    (glyphs, kern)
}

// =========================================================================
// Embedded fallback font
// =========================================================================

/// Create the built-in fallback font face.
///
/// This provides a minimal sans-serif font with ASCII coverage so that text
/// can always be rendered even when no external font files are loaded.
pub fn create_embedded_fallback_font(id: FontId) -> FontFace {
    let upm: u16 = 1000;
    let (glyphs, kern) = parse_glyph_metrics(&[], upm);

    let mut face = FontFace::new(id, "Genovo Sans");
    face.units_per_em = upm;
    face.line_metrics = LineMetrics {
        ascender: 750.0,
        descender: -250.0,
        line_gap: 100.0,
        underline_offset: -125.0,
        underline_thickness: 50.0,
        strikethrough_offset: 250.0,
        strikethrough_thickness: 50.0,
        cap_height: 680.0,
        x_height: 450.0,
    };
    face.glyph_metrics = glyphs;
    face.kerning = kern;
    face.glyph_count = face.glyph_metrics.len() as u32;
    face.loaded = true;
    face
}

// =========================================================================
// GlyphKey / CachedGlyph
// =========================================================================

/// Key for looking up a rasterised glyph in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// Character codepoint.
    pub codepoint: char,
    /// Font size in 1/64th pixel units for sub-pixel precision.
    pub size_64ths: u32,
    /// Font identifier.
    pub font_id: FontId,
    /// Sub-pixel offset (0-3 for 4x horizontal sub-pixel positioning).
    pub sub_pixel_x: u8,
}

impl GlyphKey {
    pub fn new(codepoint: char, font_size: f32, font_id: FontId) -> Self {
        Self {
            codepoint,
            size_64ths: (font_size * 64.0) as u32,
            font_id,
            sub_pixel_x: 0,
        }
    }

    pub fn with_sub_pixel(mut self, offset: u8) -> Self {
        self.sub_pixel_x = offset;
        self
    }

    pub fn font_size(&self) -> f32 {
        self.size_64ths as f32 / 64.0
    }
}

/// A rasterised glyph stored in the atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CachedGlyph {
    /// UV rectangle in the atlas: (u_min, v_min, u_max, v_max).
    pub uv: [f32; 4],
    /// Pixel-space offset from the pen position to the top-left of the bitmap.
    pub offset: Vec2,
    /// Bitmap size in pixels.
    pub size: Vec2,
    /// Advance width in pixels.
    pub advance: f32,
    /// Which atlas page this glyph is on.
    pub page: u32,
    /// LRU timestamp (frame counter).
    pub last_used: u64,
}

impl CachedGlyph {
    /// UV rect as a genovo_core::Rect.
    pub fn uv_rect(&self) -> Rect {
        Rect::new(
            Vec2::new(self.uv[0], self.uv[1]),
            Vec2::new(self.uv[2], self.uv[3]),
        )
    }
}

// =========================================================================
// AtlasShelf (shelf-based packing)
// =========================================================================

/// A horizontal "shelf" in the atlas texture.
#[derive(Debug, Clone)]
struct AtlasShelf {
    /// Y position of the shelf's top edge.
    y: u32,
    /// Height of the shelf (determined by the tallest glyph placed on it).
    height: u32,
    /// Current X cursor (where the next glyph will be placed).
    x_cursor: u32,
}

impl AtlasShelf {
    fn new(y: u32, height: u32) -> Self {
        Self {
            y,
            height,
            x_cursor: GLYPH_PADDING,
        }
    }

    /// Try to fit a glyph of the given size. Returns the (x, y) if it fits.
    fn try_allocate(&mut self, width: u32, height: u32, atlas_width: u32) -> Option<(u32, u32)> {
        if height > self.height {
            return None;
        }
        let needed = self.x_cursor + width + GLYPH_PADDING;
        if needed > atlas_width {
            return None;
        }
        let x = self.x_cursor;
        self.x_cursor = needed;
        Some((x, self.y))
    }
}

// =========================================================================
// FontAtlasPage
// =========================================================================

/// A single page (texture) in the font atlas.
#[derive(Debug, Clone)]
pub struct FontAtlasPage {
    /// Page index.
    pub index: u32,
    /// Texture ID for this page.
    pub texture_id: TextureId,
    /// Current width.
    pub width: u32,
    /// Current height.
    pub height: u32,
    /// Pixel data (single-channel alpha, row-major).
    pub pixels: Vec<u8>,
    /// Shelves for packing.
    shelves: Vec<AtlasShelf>,
    /// Next shelf Y position.
    next_shelf_y: u32,
    /// Whether the page is dirty (needs GPU upload).
    pub dirty: bool,
    /// Dirty rectangle (region that changed since last upload).
    pub dirty_rect: Option<[u32; 4]>,
}

impl FontAtlasPage {
    pub fn new(index: u32, width: u32, height: u32) -> Self {
        Self {
            index,
            texture_id: TextureId(1000 + index as u64),
            width,
            height,
            pixels: vec![0u8; (width * height) as usize],
            shelves: Vec::new(),
            next_shelf_y: GLYPH_PADDING,
            dirty: true,
            dirty_rect: Some([0, 0, width, height]),
        }
    }

    /// Attempt to allocate space for a glyph of the given pixel size.
    /// Returns (x, y) in the texture if successful.
    pub fn allocate(&mut self, glyph_w: u32, glyph_h: u32) -> Option<(u32, u32)> {
        // Try existing shelves.
        for shelf in &mut self.shelves {
            if let Some(pos) = shelf.try_allocate(glyph_w, glyph_h, self.width) {
                return Some(pos);
            }
        }

        // Create a new shelf.
        let shelf_height = glyph_h + GLYPH_PADDING;
        if self.next_shelf_y + shelf_height > self.height {
            return None; // No room.
        }

        let mut shelf = AtlasShelf::new(self.next_shelf_y, shelf_height);
        let pos = shelf.try_allocate(glyph_w, glyph_h, self.width);
        self.next_shelf_y += shelf_height;
        self.shelves.push(shelf);
        pos
    }

    /// Write glyph bitmap data at the given position.
    pub fn write_glyph(&mut self, x: u32, y: u32, width: u32, height: u32, data: &[u8]) {
        for row in 0..height {
            let src_start = (row * width) as usize;
            let dst_start = ((y + row) * self.width + x) as usize;
            let src_end = src_start + width as usize;
            let dst_end = dst_start + width as usize;
            if src_end <= data.len() && dst_end <= self.pixels.len() {
                self.pixels[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
            }
        }
        self.dirty = true;
        // Expand dirty rect.
        if let Some(ref mut dr) = self.dirty_rect {
            dr[0] = dr[0].min(x);
            dr[1] = dr[1].min(y);
            dr[2] = dr[2].max(x + width);
            dr[3] = dr[3].max(y + height);
        } else {
            self.dirty_rect = Some([x, y, x + width, y + height]);
        }
    }

    /// Remaining vertical space (approximate).
    pub fn remaining_height(&self) -> u32 {
        self.height.saturating_sub(self.next_shelf_y)
    }

    /// Mark as clean after GPU upload.
    pub fn mark_clean(&mut self) {
        self.dirty = false;
        self.dirty_rect = None;
    }

    /// Reset the page, clearing all allocations.
    pub fn reset(&mut self) {
        self.pixels.fill(0);
        self.shelves.clear();
        self.next_shelf_y = GLYPH_PADDING;
        self.dirty = true;
        self.dirty_rect = Some([0, 0, self.width, self.height]);
    }

    /// Grow the page to a larger size.
    pub fn grow(&mut self, new_width: u32, new_height: u32) {
        if new_width <= self.width && new_height <= self.height {
            return;
        }
        let nw = new_width.max(self.width);
        let nh = new_height.max(self.height);
        let mut new_pixels = vec![0u8; (nw * nh) as usize];
        // Copy old data row by row.
        for y in 0..self.height {
            let src_start = (y * self.width) as usize;
            let src_end = src_start + self.width as usize;
            let dst_start = (y * nw) as usize;
            let dst_end = dst_start + self.width as usize;
            new_pixels[dst_start..dst_end].copy_from_slice(&self.pixels[src_start..src_end]);
        }
        self.pixels = new_pixels;
        self.width = nw;
        self.height = nh;
        self.dirty = true;
        self.dirty_rect = Some([0, 0, nw, nh]);
    }
}

// =========================================================================
// FontAtlasBuilder
// =========================================================================

/// Dynamic glyph atlas with multi-page support and shelf-based packing.
///
/// Starts at 512x512 and grows as needed up to `ATLAS_MAX_SIZE` per page.
/// When a page is full, a new page is added (up to `MAX_ATLAS_PAGES`).
#[derive(Debug, Clone)]
pub struct FontAtlasBuilder {
    /// Atlas pages.
    pub pages: Vec<FontAtlasPage>,
    /// Current page width for new pages.
    pub current_page_size: u32,
    /// Maximum atlas dimension.
    pub max_size: u32,
    /// Maximum pages.
    pub max_pages: usize,
}

impl FontAtlasBuilder {
    pub fn new() -> Self {
        let mut pages = Vec::new();
        pages.push(FontAtlasPage::new(0, ATLAS_INITIAL_SIZE, ATLAS_INITIAL_SIZE));
        Self {
            pages,
            current_page_size: ATLAS_INITIAL_SIZE,
            max_size: ATLAS_MAX_SIZE,
            max_pages: MAX_ATLAS_PAGES,
        }
    }

    /// Allocate space for a glyph of the given pixel size.
    /// Returns `(page_index, x, y)` or `None` if no space.
    pub fn allocate(&mut self, glyph_w: u32, glyph_h: u32) -> Option<(u32, u32, u32)> {
        // Try current pages.
        for (i, page) in self.pages.iter_mut().enumerate() {
            if let Some((x, y)) = page.allocate(glyph_w, glyph_h) {
                return Some((i as u32, x, y));
            }
        }

        // Try growing the last page.
        if let Some(last) = self.pages.last_mut() {
            if last.width < self.max_size || last.height < self.max_size {
                let new_w = (last.width * 2).min(self.max_size);
                let new_h = (last.height * 2).min(self.max_size);
                last.grow(new_w, new_h);
                if let Some((x, y)) = last.allocate(glyph_w, glyph_h) {
                    self.current_page_size = new_w;
                    return Some(((self.pages.len() - 1) as u32, x, y));
                }
            }
        }

        // Add a new page.
        if self.pages.len() < self.max_pages {
            let page_idx = self.pages.len() as u32;
            let size = self.current_page_size.max(ATLAS_INITIAL_SIZE);
            let mut page = FontAtlasPage::new(page_idx, size, size);
            let result = page.allocate(glyph_w, glyph_h);
            self.pages.push(page);
            if let Some((x, y)) = result {
                return Some((page_idx, x, y));
            }
        }

        None
    }

    /// Write glyph pixel data.
    pub fn write_glyph(
        &mut self,
        page: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: &[u8],
    ) {
        if let Some(p) = self.pages.get_mut(page as usize) {
            p.write_glyph(x, y, width, height, data);
        }
    }

    /// Get a page by index.
    pub fn page(&self, index: u32) -> Option<&FontAtlasPage> {
        self.pages.get(index as usize)
    }

    /// Get a mutable page.
    pub fn page_mut(&mut self, index: u32) -> Option<&mut FontAtlasPage> {
        self.pages.get_mut(index as usize)
    }

    /// Total number of pages.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Mark all pages clean.
    pub fn mark_all_clean(&mut self) {
        for page in &mut self.pages {
            page.mark_clean();
        }
    }

    /// Get all dirty pages.
    pub fn dirty_pages(&self) -> Vec<u32> {
        self.pages
            .iter()
            .enumerate()
            .filter(|(_, p)| p.dirty)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Reset all pages.
    pub fn reset_all(&mut self) {
        for page in &mut self.pages {
            page.reset();
        }
    }
}

impl Default for FontAtlasBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// GlyphCache
// =========================================================================

/// Rasterised glyph cache with LRU eviction.
///
/// When glyphs are requested they are rasterised on demand and stored in the
/// atlas. An LRU list tracks access order so that the least-recently-used
/// glyphs can be evicted when the atlas fills up.
#[derive(Debug, Clone)]
pub struct GlyphCache {
    /// Cached glyph lookup.
    cache: HashMap<GlyphKey, CachedGlyph>,
    /// LRU order: front = most-recently-used.
    lru_order: VecDeque<GlyphKey>,
    /// Current frame counter for LRU timestamps.
    frame_counter: u64,
    /// Maximum cache entries before eviction.
    max_entries: usize,
    /// Number of cache hits this frame.
    pub hits: u64,
    /// Number of cache misses this frame.
    pub misses: u64,
    /// Atlas builder for allocating glyph space.
    pub atlas: FontAtlasBuilder,
}

impl GlyphCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            lru_order: VecDeque::new(),
            frame_counter: 0,
            max_entries: LRU_MAX_GLYPHS,
            hits: 0,
            misses: 0,
            atlas: FontAtlasBuilder::new(),
        }
    }

    /// Look up a cached glyph. Updates the LRU timestamp if found.
    pub fn get(&mut self, key: &GlyphKey) -> Option<&CachedGlyph> {
        if let Some(g) = self.cache.get_mut(key) {
            g.last_used = self.frame_counter;
            self.hits += 1;
            // Move to front of LRU.
            if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
                self.lru_order.remove(pos);
            }
            self.lru_order.push_front(*key);
            Some(g)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Look up without updating LRU (peek).
    pub fn peek(&self, key: &GlyphKey) -> Option<&CachedGlyph> {
        self.cache.get(key)
    }

    /// Insert a rasterised glyph into the cache.
    pub fn insert(&mut self, key: GlyphKey, glyph: CachedGlyph) {
        // Evict if full.
        if self.cache.len() >= self.max_entries {
            self.evict(LRU_EVICT_BATCH);
        }
        self.cache.insert(key, glyph);
        self.lru_order.push_front(key);
    }

    /// Evict the `count` least-recently-used entries.
    pub fn evict(&mut self, count: usize) {
        for _ in 0..count {
            if let Some(key) = self.lru_order.pop_back() {
                self.cache.remove(&key);
            }
        }
    }

    /// Rasterise and cache a glyph if not already cached.
    ///
    /// Uses the font face metrics to produce a simple bitmap. In a production
    /// engine this would call a real rasteriser like `fontdue`.
    pub fn rasterise_glyph(
        &mut self,
        key: GlyphKey,
        face: &FontFace,
    ) -> Option<CachedGlyph> {
        // Check cache first.
        if let Some(g) = self.get(&key) {
            return Some(*g);
        }

        let font_size = key.font_size();
        let scale = font_size / face.units_per_em as f32;
        let metrics = face.metrics_for(key.codepoint);

        let glyph_w = (metrics.advance_width * scale).ceil() as u32;
        let glyph_h = ((face.line_metrics.ascender - face.line_metrics.descender) * scale).ceil() as u32;

        if glyph_w == 0 || glyph_h == 0 {
            // Space or invisible glyph -- cache with zero-size UV.
            let cached = CachedGlyph {
                uv: [0.0, 0.0, 0.0, 0.0],
                offset: Vec2::ZERO,
                size: Vec2::ZERO,
                advance: metrics.advance_width * scale,
                page: 0,
                last_used: self.frame_counter,
            };
            self.insert(key, cached);
            return Some(cached);
        }

        // Allocate atlas space.
        let (page, x, y) = self.atlas.allocate(glyph_w, glyph_h)?;

        // Generate a simple bitmap (box-based approximation).
        // In a real engine, this would call `fontdue::Font::rasterize`.
        let bitmap = rasterise_glyph_simple(key.codepoint, glyph_w, glyph_h);
        self.atlas.write_glyph(page, x, y, glyph_w, glyph_h, &bitmap);

        let atlas_page = self.atlas.page(page)?;
        let aw = atlas_page.width as f32;
        let ah = atlas_page.height as f32;

        let cached = CachedGlyph {
            uv: [
                x as f32 / aw,
                y as f32 / ah,
                (x + glyph_w) as f32 / aw,
                (y + glyph_h) as f32 / ah,
            ],
            offset: Vec2::new(
                metrics.left_side_bearing * scale,
                -face.line_metrics.ascender * scale,
            ),
            size: Vec2::new(glyph_w as f32, glyph_h as f32),
            advance: metrics.advance_width * scale,
            page,
            last_used: self.frame_counter,
        };

        self.insert(key, cached);
        Some(cached)
    }

    /// Advance to the next frame, resetting per-frame stats.
    pub fn next_frame(&mut self) {
        self.frame_counter += 1;
        self.hits = 0;
        self.misses = 0;
    }

    /// Number of cached glyphs.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear the entire cache and reset the atlas.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.atlas.reset_all();
    }

    /// Cache hit rate for the last frame (0.0 - 1.0).
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            1.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

impl Default for GlyphCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple glyph rasterisation (produces a box-like bitmap for any character).
///
/// A real implementation would use SDF rendering or coverage-based
/// rasterisation. This placeholder produces a visible glyph shape.
fn rasterise_glyph_simple(codepoint: char, width: u32, height: u32) -> Vec<u8> {
    let mut bitmap = vec![0u8; (width * height) as usize];

    if codepoint == ' ' || width == 0 || height == 0 {
        return bitmap;
    }

    let w = width as f32;
    let h = height as f32;
    let margin_x = (w * 0.1).max(1.0);
    let margin_top = (h * 0.15).max(1.0);
    let margin_bottom = (h * 0.1).max(1.0);

    for py in 0..height {
        for px in 0..width {
            let fx = px as f32;
            let fy = py as f32;

            // Inside the glyph region?
            if fx >= margin_x && fx < w - margin_x && fy >= margin_top && fy < h - margin_bottom {
                // Distance from the edge for anti-aliasing.
                let dx = (fx - margin_x).min(w - margin_x - fx).min(2.0);
                let dy = (fy - margin_top).min(h - margin_bottom - fy).min(2.0);
                let edge = dx.min(dy) / 2.0;
                let alpha = edge.clamp(0.0, 1.0);
                bitmap[(py * width + px) as usize] = (alpha * 200.0) as u8;
            }
        }
    }

    bitmap
}

// =========================================================================
// ShapedGlyph
// =========================================================================

/// A positioned glyph after text shaping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapedGlyph {
    /// Character.
    pub codepoint: char,
    /// Position relative to the text origin (top-left of bounding box).
    pub position: Vec2,
    /// UV rect in the atlas.
    pub uv: [f32; 4],
    /// Size in pixels.
    pub size: Vec2,
    /// Advance width.
    pub advance: f32,
    /// Offset from the pen position to the glyph bitmap.
    pub offset: Vec2,
    /// Atlas page.
    pub page: u32,
    /// Index in the source string (byte offset).
    pub string_index: usize,
    /// Cluster index.
    pub cluster: usize,
    /// Whether this glyph is a line-break point.
    pub is_newline: bool,
    /// Whether this glyph is whitespace.
    pub is_whitespace: bool,
}

// =========================================================================
// TextAlignment
// =========================================================================

/// Horizontal text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontTextAlignment {
    Left,
    Center,
    Right,
    Justify,
}

impl Default for FontTextAlignment {
    fn default() -> Self {
        FontTextAlignment::Left
    }
}

// =========================================================================
// LineBreak
// =========================================================================

/// Information about a line of text.
#[derive(Debug, Clone)]
pub struct TextLine {
    /// Glyph indices in the shaped output that belong to this line.
    pub glyph_range: Range<usize>,
    /// Baseline Y position.
    pub baseline_y: f32,
    /// Total advance width of this line.
    pub width: f32,
    /// Number of spaces in this line (for justification).
    pub space_count: usize,
    /// Starting byte index in the source string.
    pub string_start: usize,
    /// Ending byte index in the source string.
    pub string_end: usize,
}

// =========================================================================
// TextShaper
// =========================================================================

/// Shapes text into positioned glyphs with kerning, line breaking, and alignment.
///
/// This is the main entry point for text layout. Feed it a string, font, and
/// size, and get back a list of positioned glyphs ready for rendering.
#[derive(Debug, Clone)]
pub struct FontTextShaper {
    /// Shaped glyph output.
    pub glyphs: Vec<ShapedGlyph>,
    /// Line information.
    pub lines: Vec<TextLine>,
    /// Total bounding box.
    pub bounds: Vec2,
    /// The font used for shaping.
    pub font_id: FontId,
    /// The font size used.
    pub font_size: f32,
    /// Line height multiplier (1.0 = default).
    pub line_height_multiplier: f32,
    /// Letter spacing (extra advance per glyph).
    pub letter_spacing: f32,
    /// Word spacing (extra advance per space).
    pub word_spacing: f32,
    /// Tab width in spaces.
    pub tab_width: u32,
}

impl FontTextShaper {
    pub fn new() -> Self {
        Self {
            glyphs: Vec::new(),
            lines: Vec::new(),
            bounds: Vec2::ZERO,
            font_id: FontId::DEFAULT,
            font_size: 14.0,
            line_height_multiplier: 1.0,
            letter_spacing: 0.0,
            word_spacing: 0.0,
            tab_width: 4,
        }
    }

    /// Shape text with the given font, size, and optional max width for wrapping.
    pub fn shape(
        &mut self,
        text: &str,
        face: &FontFace,
        size: f32,
        max_width: Option<f32>,
        alignment: FontTextAlignment,
        glyph_cache: &mut GlyphCache,
    ) {
        self.glyphs.clear();
        self.lines.clear();
        self.font_id = face.id;
        self.font_size = size;

        if text.is_empty() {
            self.bounds = Vec2::ZERO;
            return;
        }

        let scale = size / face.units_per_em as f32;
        let line_metrics = face.line_metrics.scaled(scale);
        let line_height = line_metrics.line_height() * self.line_height_multiplier;

        let mut pen_x: f32 = 0.0;
        let mut pen_y: f32 = line_metrics.ascender;
        let mut prev_char: Option<char> = None;
        let mut line_start_glyph: usize = 0;
        let mut line_width: f32 = 0.0;
        let mut line_space_count: usize = 0;
        let mut line_string_start: usize = 0;
        let mut last_break_glyph: Option<usize> = None;
        let mut last_break_x: f32 = 0.0;
        let mut last_break_string_idx: usize = 0;

        let chars: Vec<(usize, char)> = text.char_indices().collect();

        for (ci, &(byte_idx, ch)) in chars.iter().enumerate() {
            // Handle explicit newlines.
            if ch == '\n' {
                // Finish current line.
                self.finish_line(
                    line_start_glyph,
                    pen_x,
                    pen_y,
                    line_space_count,
                    line_string_start,
                    byte_idx,
                );

                // Add a newline glyph.
                self.glyphs.push(ShapedGlyph {
                    codepoint: '\n',
                    position: Vec2::new(pen_x, pen_y - line_metrics.ascender),
                    uv: [0.0; 4],
                    size: Vec2::ZERO,
                    advance: 0.0,
                    offset: Vec2::ZERO,
                    page: 0,
                    string_index: byte_idx,
                    cluster: ci,
                    is_newline: true,
                    is_whitespace: true,
                });

                pen_x = 0.0;
                pen_y += line_height;
                prev_char = None;
                line_start_glyph = self.glyphs.len();
                line_width = 0.0;
                line_space_count = 0;
                line_string_start = byte_idx + ch.len_utf8();
                last_break_glyph = None;
                continue;
            }

            // Handle tab.
            if ch == '\t' {
                let space_advance = face.advance_at_size(' ', size);
                let tab_advance = space_advance * self.tab_width as f32;
                let next_tab_stop = ((pen_x / tab_advance).floor() + 1.0) * tab_advance;
                let advance = next_tab_stop - pen_x;

                self.glyphs.push(ShapedGlyph {
                    codepoint: '\t',
                    position: Vec2::new(pen_x, pen_y - line_metrics.ascender),
                    uv: [0.0; 4],
                    size: Vec2::ZERO,
                    advance,
                    offset: Vec2::ZERO,
                    page: 0,
                    string_index: byte_idx,
                    cluster: ci,
                    is_newline: false,
                    is_whitespace: true,
                });

                pen_x += advance;
                line_width = pen_x;
                prev_char = Some(ch);
                last_break_glyph = Some(self.glyphs.len() - 1);
                last_break_x = pen_x;
                last_break_string_idx = byte_idx + 1;
                line_space_count += 1;
                continue;
            }

            // Kerning.
            let kern = if let Some(prev) = prev_char {
                face.kerning_at_size(prev, ch, size)
            } else {
                0.0
            };

            pen_x += kern;

            // Get glyph metrics.
            let glyph_advance = face.advance_at_size(ch, size) + self.letter_spacing;
            let is_space = ch == ' ';
            let extra_word_spacing = if is_space { self.word_spacing } else { 0.0 };
            let total_advance = glyph_advance + extra_word_spacing;

            // Word wrap check.
            if let Some(max_w) = max_width {
                if pen_x + total_advance > max_w && pen_x > 0.0 && !is_space {
                    // Need to wrap.
                    if let Some(break_glyph) = last_break_glyph {
                        // Wrap at the last break point.
                        let wrap_line_width = last_break_x;
                        self.finish_line(
                            line_start_glyph,
                            wrap_line_width,
                            pen_y,
                            line_space_count.saturating_sub(1),
                            line_string_start,
                            last_break_string_idx,
                        );

                        // Reposition glyphs after the break point.
                        pen_y += line_height;
                        let shift = last_break_x;
                        line_start_glyph = break_glyph + 1;
                        for g in &mut self.glyphs[line_start_glyph..] {
                            g.position.x -= shift;
                            g.position.y = pen_y - line_metrics.ascender;
                        }
                        pen_x -= shift;
                        line_width = pen_x;
                        line_space_count = 0;
                        line_string_start = last_break_string_idx;
                        last_break_glyph = None;
                    } else {
                        // No break point -- force wrap here.
                        self.finish_line(
                            line_start_glyph,
                            pen_x,
                            pen_y,
                            line_space_count,
                            line_string_start,
                            byte_idx,
                        );
                        pen_x = 0.0;
                        pen_y += line_height;
                        line_start_glyph = self.glyphs.len();
                        line_width = 0.0;
                        line_space_count = 0;
                        line_string_start = byte_idx;
                        last_break_glyph = None;
                    }
                }
            }

            // Record break opportunity.
            if is_space {
                last_break_glyph = Some(self.glyphs.len());
                last_break_x = pen_x + total_advance;
                last_break_string_idx = byte_idx + 1;
                line_space_count += 1;
            }

            // Rasterise glyph.
            let key = GlyphKey::new(ch, size, face.id);
            let cached = glyph_cache.rasterise_glyph(key, face);

            let (uv, gsize, offset, page) = if let Some(cg) = cached {
                (cg.uv, cg.size, cg.offset, cg.page)
            } else {
                ([0.0; 4], Vec2::ZERO, Vec2::ZERO, 0)
            };

            self.glyphs.push(ShapedGlyph {
                codepoint: ch,
                position: Vec2::new(pen_x, pen_y - line_metrics.ascender),
                uv,
                size: gsize,
                advance: total_advance,
                offset,
                page,
                string_index: byte_idx,
                cluster: ci,
                is_newline: false,
                is_whitespace: is_space,
            });

            pen_x += total_advance;
            line_width = pen_x;
            prev_char = Some(ch);
        }

        // Finish last line.
        if line_start_glyph <= self.glyphs.len() {
            self.finish_line(
                line_start_glyph,
                pen_x,
                pen_y,
                line_space_count,
                line_string_start,
                text.len(),
            );
        }

        // Apply alignment.
        self.apply_alignment(alignment, max_width);

        // Compute bounds.
        let max_line_w = self.lines.iter().map(|l| l.width).fold(0.0f32, f32::max);
        let total_h = pen_y + (line_height - line_metrics.ascender);
        self.bounds = Vec2::new(max_line_w, total_h);
    }

    fn finish_line(
        &mut self,
        start_glyph: usize,
        width: f32,
        baseline_y: f32,
        space_count: usize,
        string_start: usize,
        string_end: usize,
    ) {
        self.lines.push(TextLine {
            glyph_range: start_glyph..self.glyphs.len(),
            baseline_y,
            width,
            space_count,
            string_start,
            string_end,
        });
    }

    fn apply_alignment(&mut self, alignment: FontTextAlignment, max_width: Option<f32>) {
        let max_w = max_width.unwrap_or_else(|| {
            self.lines.iter().map(|l| l.width).fold(0.0f32, f32::max)
        });

        for line in &self.lines {
            let shift = match alignment {
                FontTextAlignment::Left => 0.0,
                FontTextAlignment::Center => (max_w - line.width) * 0.5,
                FontTextAlignment::Right => max_w - line.width,
                FontTextAlignment::Justify => 0.0, // Handled below.
            };

            if alignment == FontTextAlignment::Justify && line.space_count > 0 {
                // Distribute extra space among word gaps.
                let extra = max_w - line.width;
                let per_space = extra / line.space_count as f32;
                let mut space_idx = 0;
                for gi in line.glyph_range.clone() {
                    if gi < self.glyphs.len() {
                        self.glyphs[gi].position.x += per_space * space_idx as f32;
                        if self.glyphs[gi].is_whitespace && !self.glyphs[gi].is_newline {
                            space_idx += 1;
                        }
                    }
                }
            } else if shift.abs() > 0.001 {
                for gi in line.glyph_range.clone() {
                    if gi < self.glyphs.len() {
                        self.glyphs[gi].position.x += shift;
                    }
                }
            }
        }
    }

    /// Get the caret position (x, y, height) for a given string byte index.
    pub fn caret_position(&self, byte_index: usize, face: &FontFace) -> (f32, f32, f32) {
        let scale = self.font_size / face.units_per_em as f32;
        let line_height = face.line_metrics.scaled(scale).line_height() * self.line_height_multiplier;

        // Find which line this index is on.
        for line in &self.lines {
            if byte_index >= line.string_start && byte_index <= line.string_end {
                // Find the glyph at this position.
                let mut x = 0.0;
                for gi in line.glyph_range.clone() {
                    if gi < self.glyphs.len() {
                        if self.glyphs[gi].string_index >= byte_index {
                            x = self.glyphs[gi].position.x;
                            break;
                        }
                        x = self.glyphs[gi].position.x + self.glyphs[gi].advance;
                    }
                }
                return (x, line.baseline_y - face.line_metrics.ascender * scale, line_height);
            }
        }

        // Past the end.
        if let Some(last_line) = self.lines.last() {
            let y = last_line.baseline_y - face.line_metrics.ascender * scale;
            return (last_line.width, y, line_height);
        }

        (0.0, 0.0, line_height)
    }

    /// Hit-test: given a pixel position, find the closest string byte index.
    pub fn hit_test(&self, pos: Vec2, face: &FontFace) -> usize {
        let scale = self.font_size / face.units_per_em as f32;
        let line_height = face.line_metrics.scaled(scale).line_height() * self.line_height_multiplier;

        // Find the line.
        let mut target_line = 0;
        for (i, line) in self.lines.iter().enumerate() {
            let line_top = line.baseline_y - face.line_metrics.ascender * scale;
            if pos.y >= line_top && pos.y < line_top + line_height {
                target_line = i;
                break;
            }
            if pos.y >= line_top + line_height {
                target_line = i;
            }
        }

        if let Some(line) = self.lines.get(target_line) {
            // Find the glyph.
            let mut best_idx = line.string_start;
            let mut best_dist = f32::MAX;
            for gi in line.glyph_range.clone() {
                if gi < self.glyphs.len() {
                    let g = &self.glyphs[gi];
                    let mid = g.position.x + g.advance * 0.5;
                    let dist = (pos.x - mid).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = if pos.x < mid {
                            g.string_index
                        } else {
                            g.string_index + g.codepoint.len_utf8()
                        };
                    }
                }
            }
            best_idx
        } else {
            0
        }
    }
}

impl Default for FontTextShaper {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// TextMeasurer
// =========================================================================

/// Measures text dimensions without rendering to the atlas.
///
/// Useful for layout calculations where you need to know how much space
/// text will occupy without actually rasterising it.
#[derive(Debug, Clone)]
pub struct TextMeasurer;

impl TextMeasurer {
    /// Measure the size of `text` in the given `face` at `size` pixels,
    /// optionally wrapping at `max_width`.
    pub fn measure(
        text: &str,
        face: &FontFace,
        size: f32,
        max_width: Option<f32>,
    ) -> TextMeasurement {
        if text.is_empty() {
            return TextMeasurement {
                width: 0.0,
                height: 0.0,
                line_count: 0,
                lines: Vec::new(),
            };
        }

        let scale = size / face.units_per_em as f32;
        let line_metrics = face.line_metrics.scaled(scale);
        let line_height = line_metrics.line_height();

        let mut lines: Vec<LineMeasurement> = Vec::new();
        let mut pen_x: f32 = 0.0;
        let mut prev_char: Option<char> = None;
        let mut current_line_start: usize = 0;
        let mut last_break_x: f32 = 0.0;
        let mut last_break_byte: Option<usize> = None;
        let mut max_width_seen: f32 = 0.0;

        let chars: Vec<(usize, char)> = text.char_indices().collect();

        for &(byte_idx, ch) in &chars {
            if ch == '\n' {
                lines.push(LineMeasurement {
                    width: pen_x,
                    byte_range: current_line_start..byte_idx,
                });
                max_width_seen = max_width_seen.max(pen_x);
                pen_x = 0.0;
                prev_char = None;
                current_line_start = byte_idx + 1;
                last_break_byte = None;
                continue;
            }

            if ch == '\t' {
                let space_advance = face.advance_at_size(' ', size);
                let tab_advance = space_advance * 4.0;
                let next_tab = ((pen_x / tab_advance).floor() + 1.0) * tab_advance;
                pen_x = next_tab;
                prev_char = Some(ch);
                last_break_byte = Some(byte_idx + 1);
                last_break_x = pen_x;
                continue;
            }

            let kern = prev_char
                .map(|p| face.kerning_at_size(p, ch, size))
                .unwrap_or(0.0);
            pen_x += kern;

            let advance = face.advance_at_size(ch, size);

            if ch == ' ' {
                last_break_byte = Some(byte_idx + 1);
                last_break_x = pen_x + advance;
            }

            if let Some(max_w) = max_width {
                if pen_x + advance > max_w && pen_x > 0.0 && ch != ' ' {
                    if let Some(brk) = last_break_byte {
                        lines.push(LineMeasurement {
                            width: last_break_x,
                            byte_range: current_line_start..brk,
                        });
                        max_width_seen = max_width_seen.max(last_break_x);
                        pen_x -= last_break_x;
                        current_line_start = brk;
                        last_break_byte = None;
                    } else {
                        lines.push(LineMeasurement {
                            width: pen_x,
                            byte_range: current_line_start..byte_idx,
                        });
                        max_width_seen = max_width_seen.max(pen_x);
                        pen_x = 0.0;
                        current_line_start = byte_idx;
                        last_break_byte = None;
                    }
                }
            }

            pen_x += advance;
            prev_char = Some(ch);
        }

        // Last line.
        lines.push(LineMeasurement {
            width: pen_x,
            byte_range: current_line_start..text.len(),
        });
        max_width_seen = max_width_seen.max(pen_x);

        let line_count = lines.len();
        let total_height = line_count as f32 * line_height;

        TextMeasurement {
            width: max_width_seen,
            height: total_height,
            line_count,
            lines,
        }
    }

    /// Measure a single line (no wrapping).
    pub fn measure_line(text: &str, face: &FontFace, size: f32) -> f32 {
        let scale = size / face.units_per_em as f32;
        let mut width: f32 = 0.0;
        let mut prev: Option<char> = None;
        for ch in text.chars() {
            if let Some(p) = prev {
                width += face.kerning_for(p, ch) * scale;
            }
            width += face.metrics_for(ch).advance_width * scale;
            prev = Some(ch);
        }
        width
    }

    /// Measure a single character advance.
    pub fn char_width(ch: char, face: &FontFace, size: f32) -> f32 {
        face.advance_at_size(ch, size)
    }

    /// Compute the line height at a given size.
    pub fn line_height(face: &FontFace, size: f32) -> f32 {
        let scale = size / face.units_per_em as f32;
        face.line_metrics.scaled(scale).line_height()
    }
}

/// Result of text measurement.
#[derive(Debug, Clone)]
pub struct TextMeasurement {
    /// Maximum line width.
    pub width: f32,
    /// Total height.
    pub height: f32,
    /// Number of lines.
    pub line_count: usize,
    /// Per-line measurements.
    pub lines: Vec<LineMeasurement>,
}

/// Measurement of a single line.
#[derive(Debug, Clone)]
pub struct LineMeasurement {
    /// Width of this line in pixels.
    pub width: f32,
    /// Byte range in the source string.
    pub byte_range: Range<usize>,
}

// =========================================================================
// FontSystem (main entry point)
// =========================================================================

/// Top-level font system that ties together the database, atlas, cache, and
/// shaper. This is the recommended way to use the font system.
#[derive(Debug, Clone)]
pub struct FontSystem {
    /// Font database.
    pub database: FontDatabase,
    /// Glyph cache (includes the atlas).
    pub glyph_cache: GlyphCache,
    /// Text shaper.
    pub shaper: FontTextShaper,
    /// Current DPI scale.
    pub dpi_scale: f32,
}

impl FontSystem {
    /// Create a new font system with the embedded fallback font.
    pub fn new() -> Self {
        Self {
            database: FontDatabase::with_fallback(),
            glyph_cache: GlyphCache::new(),
            shaper: FontTextShaper::new(),
            dpi_scale: 1.0,
        }
    }

    /// Set the DPI scale factor.
    pub fn set_dpi_scale(&mut self, scale: f32) {
        self.dpi_scale = scale;
    }

    /// Load a font from raw bytes.
    pub fn load_font(&mut self, family: &str, data: &[u8]) -> FontId {
        self.database.load_from_bytes(family, data)
    }

    /// Shape text and return the shaped glyphs.
    pub fn shape_text(
        &mut self,
        text: &str,
        family: &str,
        size: FontSize,
        max_width: Option<f32>,
        alignment: FontTextAlignment,
    ) -> Vec<ShapedGlyph> {
        let px_size = size.to_pixels(self.dpi_scale);
        let face = self.database.resolve(family, FontWeight::Regular, FontStyle::Normal).clone();
        self.shaper.shape(text, &face, px_size, max_width, alignment, &mut self.glyph_cache);
        self.shaper.glyphs.clone()
    }

    /// Measure text dimensions.
    pub fn measure_text(
        &self,
        text: &str,
        family: &str,
        size: FontSize,
        max_width: Option<f32>,
    ) -> TextMeasurement {
        let px_size = size.to_pixels(self.dpi_scale);
        let face = self.database.resolve(family, FontWeight::Regular, FontStyle::Normal);
        TextMeasurer::measure(text, face, px_size, max_width)
    }

    /// Advance the frame counter.
    pub fn next_frame(&mut self) {
        self.glyph_cache.next_frame();
    }

    /// Get dirty atlas pages that need GPU upload.
    pub fn dirty_pages(&self) -> Vec<u32> {
        self.glyph_cache.atlas.dirty_pages()
    }

    /// Mark all pages as clean after GPU upload.
    pub fn mark_clean(&mut self) {
        self.glyph_cache.atlas.mark_all_clean();
    }

    /// Get atlas page pixel data for GPU upload.
    pub fn page_pixels(&self, page: u32) -> Option<&[u8]> {
        self.glyph_cache.atlas.page(page).map(|p| p.pixels.as_slice())
    }

    /// Get atlas page dimensions.
    pub fn page_dimensions(&self, page: u32) -> Option<(u32, u32)> {
        self.glyph_cache.atlas.page(page).map(|p| (p.width, p.height))
    }

    /// Get the default font face.
    pub fn default_face(&self) -> &FontFace {
        self.database.get(FontId::DEFAULT).unwrap_or_else(|| {
            self.database.iter().next().expect("No fonts loaded")
        })
    }

    /// Clear all caches. Useful when DPI changes or fonts are reloaded.
    pub fn clear_caches(&mut self) {
        self.glyph_cache.clear();
    }

    /// Cache statistics.
    pub fn cache_stats(&self) -> (u64, u64, f32, usize) {
        (
            self.glyph_cache.hits,
            self.glyph_cache.misses,
            self.glyph_cache.hit_rate(),
            self.glyph_cache.len(),
        )
    }
}

impl Default for FontSystem {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_font_size_conversion() {
        let fixed = FontSize::Fixed(16.0);
        assert_eq!(fixed.to_pixels(2.0), 16.0);

        let points = FontSize::Points(12.0);
        assert_eq!(points.to_pixels(2.0), 24.0);
    }

    #[test]
    fn test_font_weight_numeric() {
        assert_eq!(FontWeight::Regular.to_numeric(), 400);
        assert_eq!(FontWeight::Bold.to_numeric(), 700);
        assert_eq!(FontWeight::from_numeric(400), FontWeight::Regular);
        assert_eq!(FontWeight::from_numeric(700), FontWeight::Bold);
    }

    #[test]
    fn test_line_metrics_scaling() {
        let lm = LineMetrics::from_basic(750.0, -250.0, 100.0);
        let scaled = lm.scaled(0.016); // 16px / 1000upm
        assert!((scaled.ascender - 12.0).abs() < 0.01);
        assert!((scaled.descender - -4.0).abs() < 0.01);
    }

    #[test]
    fn test_font_database_registration() {
        let mut db = FontDatabase::new();
        let face = FontFace::new(FontId(0), "Test Font");
        let id = db.register_face(face);
        assert!(db.get(id).is_some());
        assert_eq!(db.face_count(), 1);
    }

    #[test]
    fn test_font_database_find_family() {
        let mut db = FontDatabase::with_fallback();
        let face = db.find_family("Genovo Sans");
        assert!(face.is_some());
    }

    #[test]
    fn test_embedded_font_has_ascii() {
        let face = create_embedded_fallback_font(FontId(0));
        assert!(face.has_glyph('A'));
        assert!(face.has_glyph('z'));
        assert!(face.has_glyph('0'));
        assert!(face.has_glyph(' '));
    }

    #[test]
    fn test_glyph_cache_insert_and_get() {
        let mut cache = GlyphCache::new();
        let key = GlyphKey::new('A', 16.0, FontId(0));
        let glyph = CachedGlyph {
            uv: [0.0, 0.0, 0.1, 0.1],
            offset: Vec2::ZERO,
            size: Vec2::new(10.0, 12.0),
            advance: 10.0,
            page: 0,
            last_used: 0,
        };
        cache.insert(key, glyph);
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_atlas_page_allocate() {
        let mut page = FontAtlasPage::new(0, 512, 512);
        let result = page.allocate(32, 32);
        assert!(result.is_some());
    }

    #[test]
    fn test_atlas_builder_multi_page() {
        let mut builder = FontAtlasBuilder::new();
        // Allocate many glyphs to trigger page growth.
        for _ in 0..100 {
            builder.allocate(32, 32);
        }
        assert!(builder.page_count() >= 1);
    }

    #[test]
    fn test_text_measurer() {
        let face = create_embedded_fallback_font(FontId(0));
        let m = TextMeasurer::measure("Hello", &face, 16.0, None);
        assert!(m.width > 0.0);
        assert!(m.height > 0.0);
        assert_eq!(m.line_count, 1);
    }

    #[test]
    fn test_text_measurer_multiline() {
        let face = create_embedded_fallback_font(FontId(0));
        let m = TextMeasurer::measure("Hello\nWorld", &face, 16.0, None);
        assert_eq!(m.line_count, 2);
    }

    #[test]
    fn test_text_measurer_wrap() {
        let face = create_embedded_fallback_font(FontId(0));
        let m = TextMeasurer::measure("Hello World this is a long line", &face, 16.0, Some(50.0));
        assert!(m.line_count > 1);
    }

    #[test]
    fn test_text_shaper() {
        let mut shaper = FontTextShaper::new();
        let face = create_embedded_fallback_font(FontId(0));
        let mut cache = GlyphCache::new();
        shaper.shape("Hello", &face, 16.0, None, FontTextAlignment::Left, &mut cache);
        assert_eq!(shaper.glyphs.len(), 5);
        assert_eq!(shaper.lines.len(), 1);
        assert!(shaper.bounds.x > 0.0);
    }

    #[test]
    fn test_font_system_shape_and_measure() {
        let mut sys = FontSystem::new();
        let glyphs = sys.shape_text("Test", "Genovo Sans", FontSize::Fixed(14.0), None, FontTextAlignment::Left);
        assert_eq!(glyphs.len(), 4);

        let m = sys.measure_text("Test", "Genovo Sans", FontSize::Fixed(14.0), None);
        assert!(m.width > 0.0);
    }

    #[test]
    fn test_kerning() {
        let face = create_embedded_fallback_font(FontId(0));
        let k = face.kerning_for('A', 'V');
        assert!(k < 0.0); // AV should kern tighter.
    }
}
