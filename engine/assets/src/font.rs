//! TrueType font parser.
//!
//! Parses `.ttf` files to extract glyph outlines, metrics, and kerning
//! information.  Supports:
//!
//! - Table directory parsing (head, cmap, hhea, hmtx, loca, glyf, kern, maxp)
//! - Character → glyph ID lookup via cmap format 4 (BMP Unicode)
//! - Simple and compound glyph outlines with delta-encoded coordinates
//! - Horizontal metrics and kerning pairs
//! - Basic text layout (string → positioned glyph list)
//! - Signed-distance-field (SDF) rasterisation

use std::collections::HashMap;
use std::path::Path;

use crate::loader::{AssetError, AssetLoader};

// =========================================================================
// Public data types
// =========================================================================

/// A loaded TrueType font.
#[derive(Debug, Clone)]
pub struct FontData {
    /// Font units per em-square.
    pub units_per_em: u16,
    /// Ascent in font units (from baseline to top of tallest glyph).
    pub ascent: i16,
    /// Descent in font units (negative, from baseline down).
    pub descent: i16,
    /// Line gap in font units.
    pub line_gap: i16,
    /// Number of glyphs.
    pub num_glyphs: u16,
    /// Index-to-location format (0 = short offsets, 1 = long offsets).
    pub index_to_loc_format: i16,
    /// Character → glyph index mapping.
    pub cmap: HashMap<u32, u16>,
    /// Glyph outlines indexed by glyph ID.
    pub glyphs: Vec<GlyphOutline>,
    /// Horizontal metrics per glyph.
    pub h_metrics: Vec<HMetric>,
    /// Kerning pairs: `(left_glyph, right_glyph) -> adjustment`.
    pub kern_pairs: HashMap<(u16, u16), i16>,
}

/// Horizontal metric for a single glyph.
#[derive(Debug, Clone, Copy)]
pub struct HMetric {
    /// Advance width in font units.
    pub advance_width: u16,
    /// Left side bearing in font units.
    pub lsb: i16,
}

/// A glyph outline.
#[derive(Debug, Clone)]
pub struct GlyphOutline {
    /// Glyph index.
    pub glyph_id: u16,
    /// Minimum X coordinate in font units.
    pub x_min: i16,
    /// Minimum Y coordinate.
    pub y_min: i16,
    /// Maximum X coordinate.
    pub x_max: i16,
    /// Maximum Y coordinate.
    pub y_max: i16,
    /// Contours making up this glyph (list of point lists).
    pub contours: Vec<Vec<GlyphPoint>>,
}

/// A point on a glyph contour.
#[derive(Debug, Clone, Copy)]
pub struct GlyphPoint {
    /// X coordinate in font units.
    pub x: i16,
    /// Y coordinate in font units.
    pub y: i16,
    /// Whether this is an on-curve point (true) or a control point (false).
    pub on_curve: bool,
}

/// A positioned glyph for text layout output.
#[derive(Debug, Clone)]
pub struct PositionedGlyph {
    /// Glyph index.
    pub glyph_id: u16,
    /// X position in font units.
    pub x: f32,
    /// Y position in font units.
    pub y: f32,
}

/// A rasterised SDF glyph bitmap.
#[derive(Debug, Clone)]
pub struct SdfGlyph {
    /// Glyph index.
    pub glyph_id: u16,
    /// Width of the bitmap in pixels.
    pub width: u32,
    /// Height of the bitmap in pixels.
    pub height: u32,
    /// Signed distance values, one per pixel (positive = outside, negative = inside).
    pub distances: Vec<f32>,
    /// Horizontal bearing X in pixels.
    pub bearing_x: f32,
    /// Vertical bearing Y in pixels.
    pub bearing_y: f32,
    /// Advance width in pixels.
    pub advance: f32,
}

// =========================================================================
// FontLoader
// =========================================================================

/// Loads TrueType font files (`.ttf`).
pub struct FontLoader;

impl AssetLoader for FontLoader {
    type Asset = FontData;

    fn extensions(&self) -> &[&str] {
        &["ttf"]
    }

    fn load(&self, _path: &Path, bytes: &[u8]) -> Result<FontData, AssetError> {
        parse_ttf(bytes)
    }
}

// =========================================================================
// Table directory
// =========================================================================

/// An entry in the TrueType offset table.
#[derive(Debug, Clone)]
struct TableEntry {
    tag: [u8; 4],
    checksum: u32,
    offset: u32,
    length: u32,
}

/// Find a table by its 4-byte tag.
fn find_table<'a>(tables: &'a [TableEntry], tag: &[u8; 4]) -> Option<&'a TableEntry> {
    tables.iter().find(|t| &t.tag == tag)
}

/// Parse the offset table and table directory.
fn parse_table_directory(data: &[u8]) -> Result<Vec<TableEntry>, AssetError> {
    if data.len() < 12 {
        return Err(AssetError::InvalidData("TTF too small for offset table".into()));
    }

    // Offset table: sfVersion(4), numTables(2), searchRange(2), entrySelector(2), rangeShift(2)
    let sf_version = read_u32_be(data, 0);
    // Accept 0x00010000 (TrueType) or 0x4F54544F (CFF/OpenType)
    if sf_version != 0x00010000 && sf_version != 0x4F54544F {
        return Err(AssetError::InvalidData(format!(
            "Not a TrueType font (sfVersion = 0x{sf_version:08X})"
        )));
    }

    let num_tables = read_u16_be(data, 4) as usize;
    let directory_start = 12;
    let directory_end = directory_start + num_tables * 16;

    if data.len() < directory_end {
        return Err(AssetError::InvalidData(
            "TTF file too small for table directory".into(),
        ));
    }

    let mut tables = Vec::with_capacity(num_tables);
    for i in 0..num_tables {
        let off = directory_start + i * 16;
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&data[off..off + 4]);
        tables.push(TableEntry {
            tag,
            checksum: read_u32_be(data, off + 4),
            offset: read_u32_be(data, off + 8),
            length: read_u32_be(data, off + 12),
        });
    }

    Ok(tables)
}

// =========================================================================
// TTF parser
// =========================================================================

/// Parse a TrueType font from raw bytes.
fn parse_ttf(data: &[u8]) -> Result<FontData, AssetError> {
    let tables = parse_table_directory(data)?;

    // Parse 'head' table
    let head = find_table(&tables, b"head")
        .ok_or_else(|| AssetError::InvalidData("Missing 'head' table".into()))?;
    let (units_per_em, index_to_loc_format) = parse_head(data, head)?;

    // Parse 'maxp' table
    let maxp = find_table(&tables, b"maxp")
        .ok_or_else(|| AssetError::InvalidData("Missing 'maxp' table".into()))?;
    let num_glyphs = parse_maxp(data, maxp)?;

    // Parse 'hhea' table
    let hhea = find_table(&tables, b"hhea")
        .ok_or_else(|| AssetError::InvalidData("Missing 'hhea' table".into()))?;
    let (ascent, descent, line_gap, num_h_metrics) = parse_hhea(data, hhea)?;

    // Parse 'hmtx' table
    let hmtx = find_table(&tables, b"hmtx")
        .ok_or_else(|| AssetError::InvalidData("Missing 'hmtx' table".into()))?;
    let h_metrics = parse_hmtx(data, hmtx, num_h_metrics, num_glyphs)?;

    // Parse 'cmap' table
    let cmap_table = find_table(&tables, b"cmap")
        .ok_or_else(|| AssetError::InvalidData("Missing 'cmap' table".into()))?;
    let cmap = parse_cmap(data, cmap_table)?;

    // Parse 'loca' table
    let loca = find_table(&tables, b"loca")
        .ok_or_else(|| AssetError::InvalidData("Missing 'loca' table".into()))?;
    let glyph_offsets = parse_loca(data, loca, index_to_loc_format, num_glyphs)?;

    // Parse 'glyf' table
    let glyf = find_table(&tables, b"glyf")
        .ok_or_else(|| AssetError::InvalidData("Missing 'glyf' table".into()))?;
    let glyphs = parse_all_glyphs(data, glyf, &glyph_offsets, num_glyphs)?;

    // Parse 'kern' table (optional)
    let kern_pairs = if let Some(kern) = find_table(&tables, b"kern") {
        parse_kern(data, kern)?
    } else {
        HashMap::new()
    };

    Ok(FontData {
        units_per_em,
        ascent,
        descent,
        line_gap,
        num_glyphs,
        index_to_loc_format,
        cmap,
        glyphs,
        h_metrics,
        kern_pairs,
    })
}

// =========================================================================
// 'head' table
// =========================================================================

fn parse_head(data: &[u8], table: &TableEntry) -> Result<(u16, i16), AssetError> {
    let off = table.offset as usize;
    if table.length < 54 {
        return Err(AssetError::InvalidData("'head' table too small".into()));
    }

    // unitsPerEm at offset 18
    let units_per_em = read_u16_be(data, off + 18);
    // indexToLocFormat at offset 50
    let index_to_loc_format = read_i16_be(data, off + 50);

    Ok((units_per_em, index_to_loc_format))
}

// =========================================================================
// 'maxp' table
// =========================================================================

fn parse_maxp(data: &[u8], table: &TableEntry) -> Result<u16, AssetError> {
    let off = table.offset as usize;
    if table.length < 6 {
        return Err(AssetError::InvalidData("'maxp' table too small".into()));
    }

    // numGlyphs at offset 4
    let num_glyphs = read_u16_be(data, off + 4);
    Ok(num_glyphs)
}

// =========================================================================
// 'hhea' table
// =========================================================================

fn parse_hhea(data: &[u8], table: &TableEntry) -> Result<(i16, i16, i16, u16), AssetError> {
    let off = table.offset as usize;
    if table.length < 36 {
        return Err(AssetError::InvalidData("'hhea' table too small".into()));
    }

    let ascent = read_i16_be(data, off + 4);
    let descent = read_i16_be(data, off + 6);
    let line_gap = read_i16_be(data, off + 8);
    let num_h_metrics = read_u16_be(data, off + 34);

    Ok((ascent, descent, line_gap, num_h_metrics))
}

// =========================================================================
// 'hmtx' table
// =========================================================================

fn parse_hmtx(
    data: &[u8],
    table: &TableEntry,
    num_h_metrics: u16,
    num_glyphs: u16,
) -> Result<Vec<HMetric>, AssetError> {
    let off = table.offset as usize;
    let mut metrics = Vec::with_capacity(num_glyphs as usize);

    let mut last_advance = 0u16;

    for i in 0..num_h_metrics as usize {
        let entry_off = off + i * 4;
        if entry_off + 4 > data.len() {
            break;
        }
        let advance_width = read_u16_be(data, entry_off);
        let lsb = read_i16_be(data, entry_off + 2);
        last_advance = advance_width;
        metrics.push(HMetric {
            advance_width,
            lsb,
        });
    }

    // Remaining glyphs share the last advance width
    let lsb_start = off + (num_h_metrics as usize) * 4;
    for i in num_h_metrics..num_glyphs {
        let lsb_off = lsb_start + (i - num_h_metrics) as usize * 2;
        let lsb = if lsb_off + 2 <= data.len() {
            read_i16_be(data, lsb_off)
        } else {
            0
        };
        metrics.push(HMetric {
            advance_width: last_advance,
            lsb,
        });
    }

    Ok(metrics)
}

// =========================================================================
// 'cmap' table — format 4 parser
// =========================================================================

fn parse_cmap(data: &[u8], table: &TableEntry) -> Result<HashMap<u32, u16>, AssetError> {
    let base = table.offset as usize;
    if table.length < 4 {
        return Err(AssetError::InvalidData("'cmap' table too small".into()));
    }

    let _version = read_u16_be(data, base);
    let num_subtables = read_u16_be(data, base + 2) as usize;

    // Find a format 4 subtable (platform 3 / encoding 1 = Windows Unicode BMP)
    // or platform 0 (Unicode)
    let mut fmt4_offset: Option<usize> = None;

    for i in 0..num_subtables {
        let rec_off = base + 4 + i * 8;
        if rec_off + 8 > data.len() {
            break;
        }
        let platform = read_u16_be(data, rec_off);
        let _encoding = read_u16_be(data, rec_off + 2);
        let subtable_offset = read_u32_be(data, rec_off + 4) as usize;

        let abs_off = base + subtable_offset;
        if abs_off + 2 > data.len() {
            continue;
        }
        let format = read_u16_be(data, abs_off);

        // Prefer format 4 (Unicode BMP)
        if format == 4 && (platform == 0 || platform == 3) {
            fmt4_offset = Some(abs_off);
            break;
        }
    }

    let fmt4_off = match fmt4_offset {
        Some(off) => off,
        None => {
            // No format 4 found; return empty mapping
            return Ok(HashMap::new());
        }
    };

    parse_cmap_format4(data, fmt4_off)
}

/// Parse cmap format 4 subtable.
///
/// Format 4 uses segment arrays to map contiguous character code ranges to
/// glyph IDs.  Each segment has: endCode, startCode, idDelta, idRangeOffset.
fn parse_cmap_format4(data: &[u8], offset: usize) -> Result<HashMap<u32, u16>, AssetError> {
    if offset + 14 > data.len() {
        return Err(AssetError::InvalidData("cmap format 4 header truncated".into()));
    }

    let _format = read_u16_be(data, offset); // should be 4
    let length = read_u16_be(data, offset + 2) as usize;
    let _language = read_u16_be(data, offset + 4);
    let seg_count_x2 = read_u16_be(data, offset + 6) as usize;
    let seg_count = seg_count_x2 / 2;

    // Validate we have enough data
    let header_size = 14;
    let arrays_size = seg_count * 2 * 4 + 2; // endCode, reservedPad, startCode, idDelta, idRangeOffset
    if offset + header_size + arrays_size > data.len() {
        return Err(AssetError::InvalidData("cmap format 4 arrays truncated".into()));
    }

    let end_code_off = offset + 14;
    let _reserved_pad_off = end_code_off + seg_count * 2; // should be 0
    let start_code_off = _reserved_pad_off + 2;
    let id_delta_off = start_code_off + seg_count * 2;
    let id_range_offset_off = id_delta_off + seg_count * 2;

    let mut cmap = HashMap::new();

    for seg in 0..seg_count {
        let end_code = read_u16_be(data, end_code_off + seg * 2) as u32;
        let start_code = read_u16_be(data, start_code_off + seg * 2) as u32;
        let id_delta = read_i16_be(data, id_delta_off + seg * 2) as i32;
        let id_range_offset = read_u16_be(data, id_range_offset_off + seg * 2) as usize;

        if start_code == 0xFFFF {
            break;
        }

        for code in start_code..=end_code {
            let glyph_id = if id_range_offset == 0 {
                // Glyph ID = characterCode + idDelta (mod 65536)
                ((code as i32 + id_delta) & 0xFFFF) as u16
            } else {
                // Glyph ID is looked up from the glyphIdArray
                let range_off_pos = id_range_offset_off + seg * 2;
                let glyph_addr = range_off_pos
                    + id_range_offset
                    + ((code - start_code) as usize) * 2;

                if glyph_addr + 2 > data.len() {
                    0
                } else {
                    let glyph = read_u16_be(data, glyph_addr);
                    if glyph == 0 {
                        0
                    } else {
                        ((glyph as i32 + id_delta) & 0xFFFF) as u16
                    }
                }
            };

            if glyph_id != 0 {
                cmap.insert(code, glyph_id);
            }
        }
    }

    Ok(cmap)
}

// =========================================================================
// 'loca' table
// =========================================================================

fn parse_loca(
    data: &[u8],
    table: &TableEntry,
    format: i16,
    num_glyphs: u16,
) -> Result<Vec<u32>, AssetError> {
    let off = table.offset as usize;
    let count = num_glyphs as usize + 1; // loca has numGlyphs + 1 entries

    let mut offsets = Vec::with_capacity(count);

    match format {
        0 => {
            // Short format: offsets are u16, actual offset = value * 2
            for i in 0..count {
                let entry_off = off + i * 2;
                if entry_off + 2 > data.len() {
                    offsets.push(0);
                } else {
                    offsets.push(read_u16_be(data, entry_off) as u32 * 2);
                }
            }
        }
        1 => {
            // Long format: offsets are u32
            for i in 0..count {
                let entry_off = off + i * 4;
                if entry_off + 4 > data.len() {
                    offsets.push(0);
                } else {
                    offsets.push(read_u32_be(data, entry_off));
                }
            }
        }
        _ => {
            return Err(AssetError::InvalidData(format!(
                "Unknown indexToLocFormat: {format}"
            )));
        }
    }

    Ok(offsets)
}

// =========================================================================
// 'glyf' table — glyph outlines
// =========================================================================

/// Parse all glyph outlines from the glyf table.
fn parse_all_glyphs(
    data: &[u8],
    glyf_table: &TableEntry,
    offsets: &[u32],
    num_glyphs: u16,
) -> Result<Vec<GlyphOutline>, AssetError> {
    let glyf_base = glyf_table.offset as usize;
    let mut glyphs = Vec::with_capacity(num_glyphs as usize);

    for glyph_id in 0..num_glyphs {
        let idx = glyph_id as usize;
        if idx + 1 >= offsets.len() {
            glyphs.push(GlyphOutline {
                glyph_id,
                x_min: 0,
                y_min: 0,
                x_max: 0,
                y_max: 0,
                contours: Vec::new(),
            });
            continue;
        }

        let glyph_offset = offsets[idx] as usize;
        let next_offset = offsets[idx + 1] as usize;

        if glyph_offset == next_offset {
            // Empty glyph (e.g. space)
            glyphs.push(GlyphOutline {
                glyph_id,
                x_min: 0,
                y_min: 0,
                x_max: 0,
                y_max: 0,
                contours: Vec::new(),
            });
            continue;
        }

        let abs_offset = glyf_base + glyph_offset;
        if abs_offset + 10 > data.len() {
            glyphs.push(GlyphOutline {
                glyph_id,
                x_min: 0,
                y_min: 0,
                x_max: 0,
                y_max: 0,
                contours: Vec::new(),
            });
            continue;
        }

        let num_contours = read_i16_be(data, abs_offset);
        let x_min = read_i16_be(data, abs_offset + 2);
        let y_min = read_i16_be(data, abs_offset + 4);
        let x_max = read_i16_be(data, abs_offset + 6);
        let y_max = read_i16_be(data, abs_offset + 8);

        let contours = if num_contours >= 0 {
            parse_simple_glyph(data, abs_offset + 10, num_contours as usize)?
        } else {
            parse_compound_glyph(data, abs_offset + 10, glyf_base, offsets)?
        };

        glyphs.push(GlyphOutline {
            glyph_id,
            x_min,
            y_min,
            x_max,
            y_max,
            contours,
        });
    }

    Ok(glyphs)
}

/// Parse a simple glyph (positive numberOfContours).
fn parse_simple_glyph(
    data: &[u8],
    offset: usize,
    num_contours: usize,
) -> Result<Vec<Vec<GlyphPoint>>, AssetError> {
    if num_contours == 0 {
        return Ok(Vec::new());
    }

    let mut pos = offset;

    // Read endPtsOfContours
    let mut end_pts = Vec::with_capacity(num_contours);
    for _ in 0..num_contours {
        if pos + 2 > data.len() {
            return Ok(Vec::new());
        }
        end_pts.push(read_u16_be(data, pos));
        pos += 2;
    }

    let num_points = *end_pts.last().unwrap_or(&0) as usize + 1;

    // Skip instructions
    if pos + 2 > data.len() {
        return Ok(Vec::new());
    }
    let instruction_len = read_u16_be(data, pos) as usize;
    pos += 2 + instruction_len;

    // Read flags
    let mut flags = Vec::with_capacity(num_points);
    while flags.len() < num_points {
        if pos >= data.len() {
            break;
        }
        let flag = data[pos];
        pos += 1;
        flags.push(flag);

        // Check repeat bit (bit 3)
        if (flag & 0x08) != 0 {
            if pos >= data.len() {
                break;
            }
            let repeat_count = data[pos] as usize;
            pos += 1;
            for _ in 0..repeat_count {
                if flags.len() >= num_points {
                    break;
                }
                flags.push(flag);
            }
        }
    }

    // Read X coordinates (delta-encoded)
    let mut x_coords = Vec::with_capacity(num_points);
    let mut x: i16 = 0;
    for i in 0..num_points.min(flags.len()) {
        let flag = flags[i];
        let x_short = (flag & 0x02) != 0; // bit 1: x is 1 byte
        let x_same_or_pos = (flag & 0x10) != 0; // bit 4: meaning depends on x_short

        if x_short {
            if pos >= data.len() {
                break;
            }
            let dx = data[pos] as i16;
            pos += 1;
            if x_same_or_pos {
                x += dx;
            } else {
                x -= dx;
            }
        } else if x_same_or_pos {
            // x is the same as previous (delta = 0)
        } else {
            if pos + 2 > data.len() {
                break;
            }
            let dx = read_i16_be(data, pos);
            pos += 2;
            x += dx;
        }
        x_coords.push(x);
    }

    // Read Y coordinates (delta-encoded)
    let mut y_coords = Vec::with_capacity(num_points);
    let mut y: i16 = 0;
    for i in 0..num_points.min(flags.len()) {
        let flag = flags[i];
        let y_short = (flag & 0x04) != 0; // bit 2
        let y_same_or_pos = (flag & 0x20) != 0; // bit 5

        if y_short {
            if pos >= data.len() {
                break;
            }
            let dy = data[pos] as i16;
            pos += 1;
            if y_same_or_pos {
                y += dy;
            } else {
                y -= dy;
            }
        } else if y_same_or_pos {
            // y same
        } else {
            if pos + 2 > data.len() {
                break;
            }
            let dy = read_i16_be(data, pos);
            pos += 2;
            y += dy;
        }
        y_coords.push(y);
    }

    // Build contours
    let actual_points = x_coords.len().min(y_coords.len()).min(flags.len());
    let mut contours = Vec::with_capacity(num_contours);
    let mut start = 0usize;

    for &end in &end_pts {
        let end_idx = (end as usize + 1).min(actual_points);
        let mut contour = Vec::new();
        for i in start..end_idx {
            contour.push(GlyphPoint {
                x: x_coords[i],
                y: y_coords[i],
                on_curve: (flags[i] & 0x01) != 0,
            });
        }
        contours.push(contour);
        start = end_idx;
    }

    Ok(contours)
}

/// Parse a compound glyph (negative numberOfContours).
///
/// Compound glyphs reference other glyphs with transformation.
fn parse_compound_glyph(
    data: &[u8],
    offset: usize,
    glyf_base: usize,
    offsets: &[u32],
) -> Result<Vec<Vec<GlyphPoint>>, AssetError> {
    let mut all_contours = Vec::new();
    let mut pos = offset;

    // Component flags
    const ARG_1_AND_2_ARE_WORDS: u16 = 0x0001;
    const ARGS_ARE_XY_VALUES: u16 = 0x0002;
    const WE_HAVE_A_SCALE: u16 = 0x0008;
    const MORE_COMPONENTS: u16 = 0x0020;
    const WE_HAVE_AN_X_AND_Y_SCALE: u16 = 0x0040;
    const WE_HAVE_A_TWO_BY_TWO: u16 = 0x0080;

    loop {
        if pos + 4 > data.len() {
            break;
        }

        let comp_flags = read_u16_be(data, pos);
        let glyph_index = read_u16_be(data, pos + 2);
        pos += 4;

        // Read arguments (offsets or point indices)
        let (dx, dy): (i16, i16);
        if (comp_flags & ARG_1_AND_2_ARE_WORDS) != 0 {
            if pos + 4 > data.len() {
                break;
            }
            if (comp_flags & ARGS_ARE_XY_VALUES) != 0 {
                dx = read_i16_be(data, pos);
                dy = read_i16_be(data, pos + 2);
            } else {
                dx = 0;
                dy = 0;
            }
            pos += 4;
        } else {
            if pos + 2 > data.len() {
                break;
            }
            if (comp_flags & ARGS_ARE_XY_VALUES) != 0 {
                dx = data[pos] as i8 as i16;
                dy = data[pos + 1] as i8 as i16;
            } else {
                dx = 0;
                dy = 0;
            }
            pos += 2;
        }

        // Read scale
        let mut scale_x: f32 = 1.0;
        let mut scale_y: f32 = 1.0;
        let mut _scale_01: f32 = 0.0;
        let mut _scale_10: f32 = 0.0;

        if (comp_flags & WE_HAVE_A_SCALE) != 0 {
            if pos + 2 > data.len() {
                break;
            }
            let s = read_i16_be(data, pos) as f32 / 16384.0; // 2.14 fixed point
            scale_x = s;
            scale_y = s;
            pos += 2;
        } else if (comp_flags & WE_HAVE_AN_X_AND_Y_SCALE) != 0 {
            if pos + 4 > data.len() {
                break;
            }
            scale_x = read_i16_be(data, pos) as f32 / 16384.0;
            scale_y = read_i16_be(data, pos + 2) as f32 / 16384.0;
            pos += 4;
        } else if (comp_flags & WE_HAVE_A_TWO_BY_TWO) != 0 {
            if pos + 8 > data.len() {
                break;
            }
            scale_x = read_i16_be(data, pos) as f32 / 16384.0;
            _scale_01 = read_i16_be(data, pos + 2) as f32 / 16384.0;
            _scale_10 = read_i16_be(data, pos + 4) as f32 / 16384.0;
            scale_y = read_i16_be(data, pos + 6) as f32 / 16384.0;
            pos += 8;
        }

        // Load referenced glyph
        let glyph_idx = glyph_index as usize;
        if glyph_idx + 1 < offsets.len() {
            let ref_offset = offsets[glyph_idx] as usize;
            let ref_next = offsets[glyph_idx + 1] as usize;

            if ref_offset != ref_next {
                let abs_ref = glyf_base + ref_offset;
                if abs_ref + 10 <= data.len() {
                    let ref_num_contours = read_i16_be(data, abs_ref);
                    if ref_num_contours >= 0 {
                        if let Ok(ref_contours) =
                            parse_simple_glyph(data, abs_ref + 10, ref_num_contours as usize)
                        {
                            for contour in ref_contours {
                                let transformed: Vec<GlyphPoint> = contour
                                    .iter()
                                    .map(|p| GlyphPoint {
                                        x: (p.x as f32 * scale_x + dx as f32).round() as i16,
                                        y: (p.y as f32 * scale_y + dy as f32).round() as i16,
                                        on_curve: p.on_curve,
                                    })
                                    .collect();
                                all_contours.push(transformed);
                            }
                        }
                    }
                }
            }
        }

        if (comp_flags & MORE_COMPONENTS) == 0 {
            break;
        }
    }

    Ok(all_contours)
}

// =========================================================================
// 'kern' table
// =========================================================================

fn parse_kern(data: &[u8], table: &TableEntry) -> Result<HashMap<(u16, u16), i16>, AssetError> {
    let base = table.offset as usize;
    if table.length < 4 {
        return Ok(HashMap::new());
    }

    let _version = read_u16_be(data, base);
    let num_tables = read_u16_be(data, base + 2) as usize;

    let mut pairs = HashMap::new();
    let mut offset = base + 4;

    for _ in 0..num_tables {
        if offset + 6 > data.len() {
            break;
        }

        let _sub_version = read_u16_be(data, offset);
        let sub_length = read_u16_be(data, offset + 2) as usize;
        let coverage = read_u16_be(data, offset + 4);

        let format = coverage >> 8;

        if format == 0 {
            // Format 0: ordered list of kerning pairs
            if offset + 14 > data.len() {
                break;
            }
            let num_pairs = read_u16_be(data, offset + 6) as usize;
            let pair_start = offset + 14;

            for p in 0..num_pairs {
                let pair_off = pair_start + p * 6;
                if pair_off + 6 > data.len() {
                    break;
                }
                let left = read_u16_be(data, pair_off);
                let right = read_u16_be(data, pair_off + 2);
                let value = read_i16_be(data, pair_off + 4);
                pairs.insert((left, right), value);
            }
        }

        offset += sub_length.max(6);
    }

    Ok(pairs)
}

// =========================================================================
// Text layout
// =========================================================================

/// Lay out a string of text using the font, returning positioned glyphs.
///
/// Text is laid out left-to-right on a single line starting at (0, 0).
/// The `font_size` is in pixels; coordinates are in pixel units.
pub fn layout_text(font: &FontData, text: &str, font_size: f32) -> Vec<PositionedGlyph> {
    let scale = font_size / font.units_per_em as f32;
    let mut positioned = Vec::with_capacity(text.len());
    let mut cursor_x: f32 = 0.0;
    let cursor_y: f32 = 0.0;

    let mut prev_glyph: Option<u16> = None;

    for ch in text.chars() {
        let code_point = ch as u32;
        let glyph_id = font.cmap.get(&code_point).copied().unwrap_or(0);

        // Apply kerning
        if let Some(prev) = prev_glyph {
            if let Some(&kern_val) = font.kern_pairs.get(&(prev, glyph_id)) {
                cursor_x += kern_val as f32 * scale;
            }
        }

        positioned.push(PositionedGlyph {
            glyph_id,
            x: cursor_x,
            y: cursor_y,
        });

        // Advance cursor
        if (glyph_id as usize) < font.h_metrics.len() {
            cursor_x += font.h_metrics[glyph_id as usize].advance_width as f32 * scale;
        }

        prev_glyph = Some(glyph_id);
    }

    positioned
}

// =========================================================================
// SDF Rasterisation
// =========================================================================

/// Rasterise a glyph as a signed distance field.
///
/// The SDF bitmap is `size x size` pixels.  The glyph is scaled and centred
/// within the bitmap with a `padding` pixel margin on each side.
/// `spread` controls how many em-space units of distance are represented
/// in the output (larger spread = more visible outline).
pub fn rasterize_sdf(
    font: &FontData,
    glyph_id: u16,
    size: u32,
    padding: u32,
    spread: f32,
) -> SdfGlyph {
    let glyph_idx = glyph_id as usize;
    let outline = if glyph_idx < font.glyphs.len() {
        &font.glyphs[glyph_idx]
    } else {
        return SdfGlyph {
            glyph_id,
            width: size,
            height: size,
            distances: vec![1.0; (size * size) as usize],
            bearing_x: 0.0,
            bearing_y: 0.0,
            advance: 0.0,
        };
    };

    let glyph_w = (outline.x_max - outline.x_min).max(1) as f32;
    let glyph_h = (outline.y_max - outline.y_min).max(1) as f32;
    let inner_size = (size - 2 * padding) as f32;
    let scale = inner_size / glyph_w.max(glyph_h);

    let offset_x = padding as f32 + (inner_size - glyph_w * scale) * 0.5;
    let offset_y = padding as f32 + (inner_size - glyph_h * scale) * 0.5;

    // Collect all line segments from the outline
    let mut segments: Vec<(f32, f32, f32, f32)> = Vec::new();
    for contour in &outline.contours {
        if contour.len() < 2 {
            continue;
        }
        for i in 0..contour.len() {
            let p0 = &contour[i];
            let p1 = &contour[(i + 1) % contour.len()];

            let x0 = (p0.x - outline.x_min) as f32 * scale + offset_x;
            let y0 = (glyph_h - (p0.y - outline.y_min) as f32) * scale + offset_y;
            let x1 = (p1.x - outline.x_min) as f32 * scale + offset_x;
            let y1 = (glyph_h - (p1.y - outline.y_min) as f32) * scale + offset_y;

            segments.push((x0, y0, x1, y1));
        }
    }

    let max_dist = spread * scale;
    let mut distances = Vec::with_capacity((size * size) as usize);

    for py in 0..size {
        for px in 0..size {
            let x = px as f32 + 0.5;
            let y = py as f32 + 0.5;

            let mut min_dist = f32::MAX;
            let mut winding = 0i32;

            for &(x0, y0, x1, y1) in &segments {
                // Distance from point to line segment
                let dist = point_to_segment_distance(x, y, x0, y0, x1, y1);
                if dist < min_dist {
                    min_dist = dist;
                }

                // Winding number contribution
                if y0 <= y {
                    if y1 > y {
                        let cross = (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0);
                        if cross > 0.0 {
                            winding += 1;
                        }
                    }
                } else if y1 <= y {
                    let cross = (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0);
                    if cross < 0.0 {
                        winding -= 1;
                    }
                }
            }

            let inside = winding != 0;
            let signed_dist = if inside { -min_dist } else { min_dist };
            let normalized = signed_dist / max_dist * 0.5 + 0.5;
            distances.push(normalized.clamp(0.0, 1.0));
        }
    }

    let h_metric = if glyph_idx < font.h_metrics.len() {
        &font.h_metrics[glyph_idx]
    } else {
        &HMetric {
            advance_width: 0,
            lsb: 0,
        }
    };

    SdfGlyph {
        glyph_id,
        width: size,
        height: size,
        distances,
        bearing_x: offset_x,
        bearing_y: offset_y,
        advance: h_metric.advance_width as f32 * scale,
    }
}

/// Minimum distance from point (px, py) to line segment (x0,y0)-(x1,y1).
fn point_to_segment_distance(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len_sq = dx * dx + dy * dy;

    if len_sq < 1e-10 {
        let ddx = px - x0;
        let ddy = py - y0;
        return (ddx * ddx + ddy * ddy).sqrt();
    }

    let t = ((px - x0) * dx + (py - y0) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);

    let proj_x = x0 + t * dx;
    let proj_y = y0 + t * dy;

    let ddx = px - proj_x;
    let ddy = py - proj_y;
    (ddx * ddx + ddy * ddy).sqrt()
}

// =========================================================================
// Big-endian read helpers
// =========================================================================

fn read_u16_be(data: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes([data[offset], data[offset + 1]])
}

fn read_i16_be(data: &[u8], offset: usize) -> i16 {
    i16::from_be_bytes([data[offset], data[offset + 1]])
}

fn read_u32_be(data: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_directory_bad_magic() {
        let data = vec![0u8; 64];
        assert!(parse_table_directory(&data).is_err());
    }

    #[test]
    fn test_table_directory_too_small() {
        assert!(parse_table_directory(&[0u8; 8]).is_err());
    }

    #[test]
    fn test_cmap_format4_empty() {
        // Build a minimal cmap format 4 with just the sentinel segment (0xFFFF)
        let mut data = Vec::new();
        // format, length, language
        data.extend_from_slice(&4u16.to_be_bytes());
        data.extend_from_slice(&((14 + 2 * 4 + 2) as u16).to_be_bytes()); // length
        data.extend_from_slice(&0u16.to_be_bytes()); // language
        data.extend_from_slice(&2u16.to_be_bytes()); // segCountX2 = 2 (1 segment)
        data.extend_from_slice(&2u16.to_be_bytes()); // searchRange
        data.extend_from_slice(&0u16.to_be_bytes()); // entrySelector
        data.extend_from_slice(&0u16.to_be_bytes()); // rangeShift

        // endCode[0] = 0xFFFF
        data.extend_from_slice(&0xFFFFu16.to_be_bytes());
        // reservedPad
        data.extend_from_slice(&0u16.to_be_bytes());
        // startCode[0] = 0xFFFF
        data.extend_from_slice(&0xFFFFu16.to_be_bytes());
        // idDelta[0] = 1
        data.extend_from_slice(&1u16.to_be_bytes());
        // idRangeOffset[0] = 0
        data.extend_from_slice(&0u16.to_be_bytes());

        let result = parse_cmap_format4(&data, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cmap_format4_simple_range() {
        // Map characters 65-67 (A-C) to glyphs 10-12 using idDelta
        let mut data = Vec::new();
        // format=4, length, language=0
        data.extend_from_slice(&4u16.to_be_bytes());
        data.extend_from_slice(&((14 + 4 * 4 + 2) as u16).to_be_bytes()); // estimate
        data.extend_from_slice(&0u16.to_be_bytes());
        data.extend_from_slice(&4u16.to_be_bytes()); // segCountX2 = 4 (2 segments)
        data.extend_from_slice(&4u16.to_be_bytes()); // searchRange
        data.extend_from_slice(&1u16.to_be_bytes()); // entrySelector
        data.extend_from_slice(&0u16.to_be_bytes()); // rangeShift

        // endCode: [67, 0xFFFF]
        data.extend_from_slice(&67u16.to_be_bytes());
        data.extend_from_slice(&0xFFFFu16.to_be_bytes());
        // reservedPad
        data.extend_from_slice(&0u16.to_be_bytes());
        // startCode: [65, 0xFFFF]
        data.extend_from_slice(&65u16.to_be_bytes());
        data.extend_from_slice(&0xFFFFu16.to_be_bytes());
        // idDelta: [10-65=-55, 1]
        let delta = (10i16 - 65) as u16;
        data.extend_from_slice(&delta.to_be_bytes());
        data.extend_from_slice(&1u16.to_be_bytes());
        // idRangeOffset: [0, 0]
        data.extend_from_slice(&0u16.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes());

        let result = parse_cmap_format4(&data, 0).unwrap();
        assert_eq!(result.get(&65), Some(&10u16)); // 'A' -> 10
        assert_eq!(result.get(&66), Some(&11u16)); // 'B' -> 11
        assert_eq!(result.get(&67), Some(&12u16)); // 'C' -> 12
        assert_eq!(result.get(&68), None);          // 'D' not mapped
    }

    #[test]
    fn test_simple_glyph_minimal() {
        // A trivial glyph with 1 contour and 3 points (triangle)
        let mut data = Vec::new();

        // endPtsOfContours: [2] (3 points, indices 0..2)
        data.extend_from_slice(&2u16.to_be_bytes());

        // instructionLength: 0
        data.extend_from_slice(&0u16.to_be_bytes());

        // Flags for 3 points: all on-curve, x and y are 2-byte
        // flag = 0x01 (on-curve) for each
        data.push(0x01);
        data.push(0x01);
        data.push(0x01);

        // X coordinates (delta-encoded, 2 bytes each since flags say so):
        // Point 0: x=100, Point 1: x=200 (delta=100), Point 2: x=150 (delta=-50)
        data.extend_from_slice(&100i16.to_be_bytes());
        data.extend_from_slice(&100i16.to_be_bytes()); // delta
        data.extend_from_slice(&(-50i16).to_be_bytes()); // delta

        // Y coordinates:
        // Point 0: y=0, Point 1: y=0 (delta=0), Point 2: y=200 (delta=200)
        data.extend_from_slice(&0i16.to_be_bytes());
        data.extend_from_slice(&0i16.to_be_bytes());
        data.extend_from_slice(&200i16.to_be_bytes());

        let contours = parse_simple_glyph(&data, 0, 1).unwrap();
        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0].len(), 3);
        assert_eq!(contours[0][0].x, 100);
        assert_eq!(contours[0][0].y, 0);
        assert_eq!(contours[0][1].x, 200);
        assert_eq!(contours[0][1].y, 0);
        assert_eq!(contours[0][2].x, 150);
        assert_eq!(contours[0][2].y, 200);
        assert!(contours[0][0].on_curve);
    }

    #[test]
    fn test_point_to_segment_distance() {
        // Point (0, 0) to segment (1, 0)-(1, 1) -> distance = 1.0
        let d = point_to_segment_distance(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);
        assert!((d - 1.0).abs() < 0.001);

        // Point (0.5, 0.5) to segment (0, 0)-(1, 0) -> distance = 0.5
        let d = point_to_segment_distance(0.5, 0.5, 0.0, 0.0, 1.0, 0.0);
        assert!((d - 0.5).abs() < 0.001);

        // Point on segment
        let d = point_to_segment_distance(0.5, 0.0, 0.0, 0.0, 1.0, 0.0);
        assert!(d < 0.001);
    }

    #[test]
    fn test_layout_text_empty() {
        let font = FontData {
            units_per_em: 1000,
            ascent: 800,
            descent: -200,
            line_gap: 0,
            num_glyphs: 0,
            index_to_loc_format: 0,
            cmap: HashMap::new(),
            glyphs: Vec::new(),
            h_metrics: Vec::new(),
            kern_pairs: HashMap::new(),
        };

        let positioned = layout_text(&font, "", 16.0);
        assert!(positioned.is_empty());
    }

    #[test]
    fn test_layout_text_with_metrics() {
        let mut cmap = HashMap::new();
        cmap.insert('A' as u32, 1u16);
        cmap.insert('B' as u32, 2u16);

        let font = FontData {
            units_per_em: 1000,
            ascent: 800,
            descent: -200,
            line_gap: 0,
            num_glyphs: 3,
            index_to_loc_format: 0,
            cmap,
            glyphs: Vec::new(),
            h_metrics: vec![
                HMetric { advance_width: 0, lsb: 0 },     // glyph 0 (.notdef)
                HMetric { advance_width: 600, lsb: 50 },   // glyph 1 (A)
                HMetric { advance_width: 700, lsb: 60 },   // glyph 2 (B)
            ],
            kern_pairs: HashMap::new(),
        };

        let positioned = layout_text(&font, "AB", 16.0);
        assert_eq!(positioned.len(), 2);
        assert_eq!(positioned[0].glyph_id, 1);
        assert!((positioned[0].x - 0.0).abs() < 0.001);
        assert_eq!(positioned[1].glyph_id, 2);
        // A advance = 600 * (16/1000) = 9.6
        assert!((positioned[1].x - 9.6).abs() < 0.01);
    }

    #[test]
    fn test_layout_text_with_kerning() {
        let mut cmap = HashMap::new();
        cmap.insert('A' as u32, 1u16);
        cmap.insert('V' as u32, 2u16);

        let mut kern_pairs = HashMap::new();
        kern_pairs.insert((1u16, 2u16), -50i16); // AV kern = -50

        let font = FontData {
            units_per_em: 1000,
            ascent: 800,
            descent: -200,
            line_gap: 0,
            num_glyphs: 3,
            index_to_loc_format: 0,
            cmap,
            glyphs: Vec::new(),
            h_metrics: vec![
                HMetric { advance_width: 0, lsb: 0 },
                HMetric { advance_width: 600, lsb: 50 },
                HMetric { advance_width: 600, lsb: 50 },
            ],
            kern_pairs,
        };

        let positioned = layout_text(&font, "AV", 20.0);
        assert_eq!(positioned.len(), 2);
        // V position = A_advance + kern = 600 * 0.02 + (-50 * 0.02) = 12.0 - 1.0 = 11.0
        let scale = 20.0 / 1000.0;
        let expected = 600.0 * scale - 50.0 * scale;
        assert!((positioned[1].x - expected).abs() < 0.01);
    }

    #[test]
    fn test_sdf_rasterize_empty_glyph() {
        let font = FontData {
            units_per_em: 1000,
            ascent: 800,
            descent: -200,
            line_gap: 0,
            num_glyphs: 1,
            index_to_loc_format: 0,
            cmap: HashMap::new(),
            glyphs: vec![GlyphOutline {
                glyph_id: 0,
                x_min: 0,
                y_min: 0,
                x_max: 100,
                y_max: 100,
                contours: Vec::new(),
            }],
            h_metrics: vec![HMetric { advance_width: 500, lsb: 0 }],
            kern_pairs: HashMap::new(),
        };

        let sdf = rasterize_sdf(&font, 0, 32, 2, 10.0);
        assert_eq!(sdf.width, 32);
        assert_eq!(sdf.height, 32);
        assert_eq!(sdf.distances.len(), 32 * 32);
    }

    #[test]
    fn test_read_helpers_be() {
        let data = [0x00, 0x42, 0x00, 0x00, 0x10, 0x00];
        assert_eq!(read_u16_be(&data, 0), 0x0042);
        assert_eq!(read_i16_be(&data, 0), 0x0042);
        assert_eq!(read_u32_be(&data, 0), 0x00420000);
    }

    #[test]
    fn test_hmetric_struct() {
        let m = HMetric { advance_width: 500, lsb: -10 };
        assert_eq!(m.advance_width, 500);
        assert_eq!(m.lsb, -10);
    }

    #[test]
    fn test_glyph_point() {
        let p = GlyphPoint { x: 100, y: 200, on_curve: true };
        assert_eq!(p.x, 100);
        assert_eq!(p.y, 200);
        assert!(p.on_curve);
    }
}
