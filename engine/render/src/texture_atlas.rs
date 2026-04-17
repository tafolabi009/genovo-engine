// engine/render/src/texture_atlas.rs
//
// Automatic texture atlas packing. Provides multiple rectangle packing
// algorithms (shelf, guillotine, max-rects), multi-page atlas support,
// UV coordinate remapping, and sprite atlas integration for 2D UI and
// particle textures.
//
// # Algorithms
//
// - **Shelf packing**: fast O(n) algorithm that maintains horizontal shelves.
//   Best for similarly-sized rectangles (sprite sheets, font glyphs).
// - **Guillotine packing**: binary subdivision of free rectangles. Good
//   balance of speed and density for mixed-size inputs.
// - **Max-rects packing**: tracks all maximal free rectangles and uses
//   best-short-side-fit heuristic. Highest density but slower for many items.

use std::fmt;

// ---------------------------------------------------------------------------
// Rect types
// ---------------------------------------------------------------------------

/// A rectangle with position and size (in pixels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackRect {
    /// X position in the atlas (set by the packer).
    pub x: u32,
    /// Y position in the atlas (set by the packer).
    pub y: u32,
    /// Width.
    pub w: u32,
    /// Height.
    pub h: u32,
}

impl PackRect {
    /// Create a new rectangle at the origin.
    pub fn new(w: u32, h: u32) -> Self {
        Self { x: 0, y: 0, w, h }
    }

    /// Create a positioned rectangle.
    pub fn positioned(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// Area in pixels.
    pub fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    /// Right edge (exclusive).
    pub fn right(&self) -> u32 {
        self.x + self.w
    }

    /// Bottom edge (exclusive).
    pub fn bottom(&self) -> u32 {
        self.y + self.h
    }

    /// Check if this rectangle overlaps another.
    pub fn overlaps(&self, other: &PackRect) -> bool {
        self.x < other.right()
            && self.right() > other.x
            && self.y < other.bottom()
            && self.bottom() > other.y
    }

    /// Check if this rectangle contains a point.
    pub fn contains_point(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    /// Check if this rectangle fully contains another.
    pub fn contains_rect(&self, other: &PackRect) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }
}

impl fmt::Display for PackRect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}x{})", self.x, self.y, self.w, self.h)
    }
}

// ---------------------------------------------------------------------------
// Atlas entry
// ---------------------------------------------------------------------------

/// An entry in the atlas: maps an original texture to its location.
#[derive(Debug, Clone)]
pub struct AtlasEntry {
    /// Index of the original texture in the input list.
    pub source_index: usize,
    /// Page index (for multi-page atlases).
    pub page: u32,
    /// Position and size within the atlas page (excluding padding).
    pub rect: PackRect,
    /// UV offset: top-left corner in normalised [0, 1] coordinates.
    pub uv_offset: [f32; 2],
    /// UV scale: size in normalised [0, 1] coordinates.
    pub uv_scale: [f32; 2],
    /// Original width before packing (may differ if rotated).
    pub original_width: u32,
    /// Original height before packing.
    pub original_height: u32,
    /// Whether the rectangle was rotated 90 degrees to fit.
    pub rotated: bool,
    /// Padding in pixels added around this entry.
    pub padding: u32,
}

impl AtlasEntry {
    /// Compute UV coordinates for a point within the original texture.
    /// `u` and `v` are in [0, 1] relative to the original texture.
    pub fn map_uv(&self, u: f32, v: f32) -> [f32; 2] {
        if self.rotated {
            [
                self.uv_offset[0] + v * self.uv_scale[0],
                self.uv_offset[1] + (1.0 - u) * self.uv_scale[1],
            ]
        } else {
            [
                self.uv_offset[0] + u * self.uv_scale[0],
                self.uv_offset[1] + v * self.uv_scale[1],
            ]
        }
    }
}

// ---------------------------------------------------------------------------
// Atlas layout result
// ---------------------------------------------------------------------------

/// Result of packing textures into one or more atlas pages.
#[derive(Debug, Clone)]
pub struct AtlasLayout {
    /// Width of each atlas page.
    pub page_width: u32,
    /// Height of each atlas page.
    pub page_height: u32,
    /// Number of pages used.
    pub page_count: u32,
    /// All placed entries.
    pub entries: Vec<AtlasEntry>,
    /// Packing efficiency (0..1) across all pages.
    pub efficiency: f32,
    /// Total wasted pixels across all pages.
    pub wasted_pixels: u64,
}

impl AtlasLayout {
    /// Get entries for a specific page.
    pub fn entries_for_page(&self, page: u32) -> Vec<&AtlasEntry> {
        self.entries.iter().filter(|e| e.page == page).collect()
    }

    /// Get the entry for a given source index.
    pub fn entry_for_source(&self, source_index: usize) -> Option<&AtlasEntry> {
        self.entries.iter().find(|e| e.source_index == source_index)
    }

    /// Compute the efficiency for a single page.
    pub fn page_efficiency(&self, page: u32) -> f32 {
        let page_area = self.page_width as u64 * self.page_height as u64;
        if page_area == 0 {
            return 0.0;
        }
        let used: u64 = self
            .entries
            .iter()
            .filter(|e| e.page == page)
            .map(|e| e.rect.area())
            .sum();
        used as f32 / page_area as f32
    }
}

impl fmt::Display for AtlasLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "AtlasLayout: {}x{}, {} pages, {} entries, {:.1}% efficient",
            self.page_width,
            self.page_height,
            self.page_count,
            self.entries.len(),
            self.efficiency * 100.0,
        )?;
        for page in 0..self.page_count {
            let count = self.entries.iter().filter(|e| e.page == page).count();
            writeln!(
                f,
                "  Page {}: {} entries, {:.1}% fill",
                page,
                count,
                self.page_efficiency(page) * 100.0,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Packing heuristic
// ---------------------------------------------------------------------------

/// Heuristic for choosing where to place a rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingHeuristic {
    /// Best short side fit: minimise the shorter leftover side.
    BestShortSideFit,
    /// Best long side fit: minimise the longer leftover side.
    BestLongSideFit,
    /// Best area fit: minimise leftover area.
    BestAreaFit,
    /// Bottom-left: place as low and as left as possible.
    BottomLeft,
}

/// Whether to allow 90-degree rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationMode {
    /// Never rotate.
    None,
    /// Allow 90-degree rotation if it helps.
    Allow90,
}

// ---------------------------------------------------------------------------
// Shelf packer
// ---------------------------------------------------------------------------

/// A horizontal shelf within the atlas.
#[derive(Debug, Clone)]
struct Shelf {
    /// Y position of this shelf's top edge.
    y: u32,
    /// Height of this shelf.
    height: u32,
    /// Current X position (rightmost placed rect's right edge).
    cursor_x: u32,
}

/// Fast shelf-based rectangle packing.
///
/// Maintains a list of horizontal shelves. Each new rectangle is placed on
/// the first shelf where it fits (height-wise), or a new shelf is created.
/// Rects are placed left-to-right on each shelf.
pub struct ShelfPacker {
    /// Atlas page width.
    width: u32,
    /// Atlas page height.
    height: u32,
    /// Active shelves.
    shelves: Vec<Shelf>,
    /// Placed rectangles.
    placed: Vec<PackRect>,
    /// Current Y for the next shelf.
    next_shelf_y: u32,
    /// Padding between entries.
    padding: u32,
}

impl ShelfPacker {
    /// Create a new shelf packer for the given atlas dimensions.
    pub fn new(width: u32, height: u32, padding: u32) -> Self {
        Self {
            width,
            height,
            shelves: Vec::new(),
            placed: Vec::new(),
            next_shelf_y: 0,
            padding,
        }
    }

    /// Try to pack a rectangle. Returns the placed position or None if it
    /// doesn't fit on this page.
    pub fn pack(&mut self, w: u32, h: u32) -> Option<PackRect> {
        let pw = w + self.padding * 2;
        let ph = h + self.padding * 2;

        // Try each existing shelf.
        for shelf in &mut self.shelves {
            if ph <= shelf.height && shelf.cursor_x + pw <= self.width {
                let rect = PackRect::positioned(
                    shelf.cursor_x + self.padding,
                    shelf.y + self.padding,
                    w,
                    h,
                );
                shelf.cursor_x += pw;
                self.placed.push(rect);
                return Some(rect);
            }
        }

        // Create a new shelf if there's room.
        if self.next_shelf_y + ph <= self.height {
            let rect = PackRect::positioned(self.padding, self.next_shelf_y + self.padding, w, h);
            self.shelves.push(Shelf {
                y: self.next_shelf_y,
                height: ph,
                cursor_x: pw,
            });
            self.next_shelf_y += ph;
            self.placed.push(rect);
            Some(rect)
        } else {
            None
        }
    }

    /// Reset the packer for reuse.
    pub fn reset(&mut self) {
        self.shelves.clear();
        self.placed.clear();
        self.next_shelf_y = 0;
    }

    /// Current fill ratio.
    pub fn fill_ratio(&self) -> f32 {
        let used: u64 = self.placed.iter().map(|r| r.area()).sum();
        let total = self.width as u64 * self.height as u64;
        if total == 0 {
            return 0.0;
        }
        used as f32 / total as f32
    }
}

// ---------------------------------------------------------------------------
// Guillotine packer
// ---------------------------------------------------------------------------

/// Rectangle packing using the guillotine (binary subdivision) algorithm.
///
/// Maintains a list of free rectangles. When a rect is placed in a free
/// rectangle, the remaining space is split into two new free rectangles
/// (either horizontally or vertically along the placed rect's edges).
pub struct GuillotinePacker {
    /// Atlas width.
    width: u32,
    /// Atlas height.
    height: u32,
    /// Free rectangles.
    free_rects: Vec<PackRect>,
    /// Placed rectangles.
    placed: Vec<PackRect>,
    /// Padding.
    padding: u32,
}

/// How to split the remaining space after placing a rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuillotineSplit {
    /// Split along the shorter remaining side.
    ShorterAxis,
    /// Split along the longer remaining side.
    LongerAxis,
    /// Split horizontally (create left/right and top free rects).
    Horizontal,
    /// Split vertically (create top/bottom and right free rects).
    Vertical,
    /// Minimise area of the smaller resulting free rect.
    MinimiseArea,
}

impl GuillotinePacker {
    /// Create a new guillotine packer.
    pub fn new(width: u32, height: u32, padding: u32) -> Self {
        Self {
            width,
            height,
            free_rects: vec![PackRect::positioned(0, 0, width, height)],
            placed: Vec::new(),
            padding,
        }
    }

    /// Find the best free rectangle for a given size using BSSF heuristic.
    fn find_best_fit(&self, w: u32, h: u32) -> Option<(usize, bool)> {
        let pw = w + self.padding * 2;
        let ph = h + self.padding * 2;

        let mut best_idx = None;
        let mut best_short_side = u32::MAX;
        let mut best_rotated = false;

        for (i, free) in self.free_rects.iter().enumerate() {
            // Try normal orientation.
            if pw <= free.w && ph <= free.h {
                let short_side = (free.w - pw).min(free.h - ph);
                if short_side < best_short_side {
                    best_short_side = short_side;
                    best_idx = Some(i);
                    best_rotated = false;
                }
            }
            // Try rotated.
            if ph <= free.w && pw <= free.h && pw != ph {
                let short_side = (free.w - ph).min(free.h - pw);
                if short_side < best_short_side {
                    best_short_side = short_side;
                    best_idx = Some(i);
                    best_rotated = true;
                }
            }
        }

        best_idx.map(|i| (i, best_rotated))
    }

    /// Split a free rectangle after placing a rect in it.
    fn split_free_rect(
        &mut self,
        free_idx: usize,
        placed_w: u32,
        placed_h: u32,
        split: GuillotineSplit,
    ) {
        let free = self.free_rects[free_idx];
        let pw = placed_w + self.padding * 2;
        let ph = placed_h + self.padding * 2;

        let right_w = free.w - pw;
        let bottom_h = free.h - ph;

        // Decide split direction.
        let split_horizontal = match split {
            GuillotineSplit::Horizontal => true,
            GuillotineSplit::Vertical => false,
            GuillotineSplit::ShorterAxis => right_w <= bottom_h,
            GuillotineSplit::LongerAxis => right_w > bottom_h,
            GuillotineSplit::MinimiseArea => {
                let area_h = right_w as u64 * ph as u64 + free.w as u64 * bottom_h as u64;
                let area_v = right_w as u64 * free.h as u64 + pw as u64 * bottom_h as u64;
                area_h.min(area_v) == area_h
            }
        };

        // Remove the used free rect.
        self.free_rects.swap_remove(free_idx);

        if split_horizontal {
            // Right of placed rect, same height as placed.
            if right_w > 0 && ph > 0 {
                self.free_rects.push(PackRect::positioned(
                    free.x + pw,
                    free.y,
                    right_w,
                    ph,
                ));
            }
            // Below placed rect, full width of original free rect.
            if bottom_h > 0 && free.w > 0 {
                self.free_rects.push(PackRect::positioned(
                    free.x,
                    free.y + ph,
                    free.w,
                    bottom_h,
                ));
            }
        } else {
            // Right of placed rect, full height of original free rect.
            if right_w > 0 && free.h > 0 {
                self.free_rects.push(PackRect::positioned(
                    free.x + pw,
                    free.y,
                    right_w,
                    free.h,
                ));
            }
            // Below placed rect, same width as placed.
            if bottom_h > 0 && pw > 0 {
                self.free_rects.push(PackRect::positioned(
                    free.x,
                    free.y + ph,
                    pw,
                    bottom_h,
                ));
            }
        }
    }

    /// Try to pack a rectangle. Returns the placed position or None.
    pub fn pack(&mut self, w: u32, h: u32) -> Option<(PackRect, bool)> {
        let (free_idx, rotated) = self.find_best_fit(w, h)?;
        let free = self.free_rects[free_idx];

        let (pw, ph) = if rotated { (h, w) } else { (w, h) };
        let rect = PackRect::positioned(
            free.x + self.padding,
            free.y + self.padding,
            pw,
            ph,
        );

        self.split_free_rect(free_idx, pw, ph, GuillotineSplit::ShorterAxis);
        self.placed.push(rect);

        Some((rect, rotated))
    }

    /// Merge adjacent free rectangles to reduce fragmentation.
    pub fn merge_free_rects(&mut self) {
        let mut i = 0;
        while i < self.free_rects.len() {
            let mut j = i + 1;
            while j < self.free_rects.len() {
                let a = self.free_rects[i];
                let b = self.free_rects[j];

                // Try horizontal merge: same y, same height, adjacent x.
                if a.y == b.y && a.h == b.h && a.right() == b.x {
                    self.free_rects[i] = PackRect::positioned(a.x, a.y, a.w + b.w, a.h);
                    self.free_rects.swap_remove(j);
                    continue;
                }
                if a.y == b.y && a.h == b.h && b.right() == a.x {
                    self.free_rects[i] = PackRect::positioned(b.x, b.y, a.w + b.w, a.h);
                    self.free_rects.swap_remove(j);
                    continue;
                }

                // Try vertical merge: same x, same width, adjacent y.
                if a.x == b.x && a.w == b.w && a.bottom() == b.y {
                    self.free_rects[i] = PackRect::positioned(a.x, a.y, a.w, a.h + b.h);
                    self.free_rects.swap_remove(j);
                    continue;
                }
                if a.x == b.x && a.w == b.w && b.bottom() == a.y {
                    self.free_rects[i] = PackRect::positioned(b.x, b.y, a.w, a.h + b.h);
                    self.free_rects.swap_remove(j);
                    continue;
                }

                j += 1;
            }
            i += 1;
        }
    }

    /// Reset for reuse.
    pub fn reset(&mut self) {
        self.free_rects.clear();
        self.free_rects
            .push(PackRect::positioned(0, 0, self.width, self.height));
        self.placed.clear();
    }

    /// Current fill ratio.
    pub fn fill_ratio(&self) -> f32 {
        let used: u64 = self.placed.iter().map(|r| r.area()).sum();
        let total = self.width as u64 * self.height as u64;
        if total == 0 {
            return 0.0;
        }
        used as f32 / total as f32
    }
}

// ---------------------------------------------------------------------------
// Max-rects packer
// ---------------------------------------------------------------------------

/// Rectangle packing using the max-rects algorithm.
///
/// Maintains a set of maximal free rectangles. When a rect is placed, all
/// overlapping free rectangles are split, and any free rect that is fully
/// contained by another is pruned. This produces optimal density but is
/// O(n^2) in the number of free rectangles.
pub struct MaxRectsPacker {
    /// Atlas width.
    width: u32,
    /// Atlas height.
    height: u32,
    /// Set of maximal free rectangles.
    free_rects: Vec<PackRect>,
    /// Placed rectangles.
    placed: Vec<PackRect>,
    /// Padding.
    padding: u32,
    /// Heuristic.
    heuristic: PackingHeuristic,
    /// Rotation mode.
    rotation: RotationMode,
}

impl MaxRectsPacker {
    /// Create a new max-rects packer.
    pub fn new(
        width: u32,
        height: u32,
        padding: u32,
        heuristic: PackingHeuristic,
        rotation: RotationMode,
    ) -> Self {
        Self {
            width,
            height,
            free_rects: vec![PackRect::positioned(0, 0, width, height)],
            placed: Vec::new(),
            padding,
            heuristic,
            rotation,
        }
    }

    /// Score a potential placement using the configured heuristic.
    /// Returns (primary_score, secondary_score). Lower is better.
    fn score_placement(&self, free: &PackRect, w: u32, h: u32) -> (i64, i64) {
        match self.heuristic {
            PackingHeuristic::BestShortSideFit => {
                let leftover_x = (free.w as i64) - (w as i64);
                let leftover_y = (free.h as i64) - (h as i64);
                let short = leftover_x.min(leftover_y);
                let long = leftover_x.max(leftover_y);
                (short, long)
            }
            PackingHeuristic::BestLongSideFit => {
                let leftover_x = (free.w as i64) - (w as i64);
                let leftover_y = (free.h as i64) - (h as i64);
                let short = leftover_x.min(leftover_y);
                let long = leftover_x.max(leftover_y);
                (long, short)
            }
            PackingHeuristic::BestAreaFit => {
                let area_leftover = (free.w as i64 * free.h as i64) - (w as i64 * h as i64);
                let short_side = ((free.w as i64) - (w as i64))
                    .min((free.h as i64) - (h as i64));
                (area_leftover, short_side)
            }
            PackingHeuristic::BottomLeft => {
                let top_y = free.y as i64 + h as i64;
                let left_x = free.x as i64;
                (top_y, left_x)
            }
        }
    }

    /// Find the best placement for a rectangle of the given size.
    /// Returns (free_rect_index, placed_rect, rotated, score).
    fn find_best_placement(
        &self,
        w: u32,
        h: u32,
    ) -> Option<(usize, PackRect, bool, (i64, i64))> {
        let pw = w + self.padding * 2;
        let ph = h + self.padding * 2;

        let mut best: Option<(usize, PackRect, bool, (i64, i64))> = None;

        for (i, free) in self.free_rects.iter().enumerate() {
            // Normal orientation.
            if pw <= free.w && ph <= free.h {
                let score = self.score_placement(free, pw, ph);
                let rect = PackRect::positioned(
                    free.x + self.padding,
                    free.y + self.padding,
                    w,
                    h,
                );
                if best
                    .as_ref()
                    .map_or(true, |(_, _, _, best_score)| score < *best_score)
                {
                    best = Some((i, rect, false, score));
                }
            }

            // Rotated orientation.
            if self.rotation == RotationMode::Allow90 && ph <= free.w && pw <= free.h && pw != ph {
                let score = self.score_placement(free, ph, pw);
                let rect = PackRect::positioned(
                    free.x + self.padding,
                    free.y + self.padding,
                    h,
                    w,
                );
                if best
                    .as_ref()
                    .map_or(true, |(_, _, _, best_score)| score < *best_score)
                {
                    best = Some((i, rect, true, score));
                }
            }
        }

        best
    }

    /// Split all free rectangles that overlap with the placed rectangle.
    fn split_free_rects(&mut self, placed: &PackRect) {
        // The placed rect with padding.
        let padded = PackRect::positioned(
            placed.x.saturating_sub(self.padding),
            placed.y.saturating_sub(self.padding),
            placed.w + self.padding * 2,
            placed.h + self.padding * 2,
        );

        let mut new_rects = Vec::new();
        let mut i = 0;

        while i < self.free_rects.len() {
            let free = self.free_rects[i];

            if !free.overlaps(&padded) {
                i += 1;
                continue;
            }

            // This free rect overlaps the placed rect. Split it into up to
            // 4 new free rects (one for each side of the placed rect).

            // Left portion.
            if padded.x > free.x {
                new_rects.push(PackRect::positioned(
                    free.x,
                    free.y,
                    padded.x - free.x,
                    free.h,
                ));
            }

            // Right portion.
            if padded.right() < free.right() {
                new_rects.push(PackRect::positioned(
                    padded.right(),
                    free.y,
                    free.right() - padded.right(),
                    free.h,
                ));
            }

            // Top portion.
            if padded.y > free.y {
                new_rects.push(PackRect::positioned(
                    free.x,
                    free.y,
                    free.w,
                    padded.y - free.y,
                ));
            }

            // Bottom portion.
            if padded.bottom() < free.bottom() {
                new_rects.push(PackRect::positioned(
                    free.x,
                    padded.bottom(),
                    free.w,
                    free.bottom() - padded.bottom(),
                ));
            }

            // Remove the overlapping free rect.
            self.free_rects.swap_remove(i);
            // Don't increment i since swap_remove moved the last element here.
        }

        // Add new split rects.
        self.free_rects.extend(new_rects);
    }

    /// Remove free rectangles that are fully contained by another.
    fn prune_contained(&mut self) {
        let mut i = 0;
        while i < self.free_rects.len() {
            let mut j = i + 1;
            let mut remove_i = false;
            while j < self.free_rects.len() {
                let a = self.free_rects[i];
                let b = self.free_rects[j];

                if a.contains_rect(&b) {
                    // b is contained by a -- remove b.
                    self.free_rects.swap_remove(j);
                    continue;
                }
                if b.contains_rect(&a) {
                    // a is contained by b -- mark a for removal.
                    remove_i = true;
                    break;
                }

                j += 1;
            }
            if remove_i {
                self.free_rects.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Try to pack a rectangle. Returns (placed_rect, rotated) or None.
    pub fn pack(&mut self, w: u32, h: u32) -> Option<(PackRect, bool)> {
        let (_idx, rect, rotated, _score) = self.find_best_placement(w, h)?;

        self.split_free_rects(&rect);
        self.prune_contained();
        self.placed.push(rect);

        Some((rect, rotated))
    }

    /// Reset for reuse.
    pub fn reset(&mut self) {
        self.free_rects.clear();
        self.free_rects
            .push(PackRect::positioned(0, 0, self.width, self.height));
        self.placed.clear();
    }

    /// Current fill ratio.
    pub fn fill_ratio(&self) -> f32 {
        let used: u64 = self.placed.iter().map(|r| r.area()).sum();
        let total = self.width as u64 * self.height as u64;
        if total == 0 {
            return 0.0;
        }
        used as f32 / total as f32
    }

    /// Number of free rectangles (for diagnostics).
    pub fn free_rect_count(&self) -> usize {
        self.free_rects.len()
    }
}

// ---------------------------------------------------------------------------
// Packing algorithm selection
// ---------------------------------------------------------------------------

/// Which packing algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingAlgorithm {
    /// Shelf packing (fastest, lowest density).
    Shelf,
    /// Guillotine packing (medium speed and density).
    Guillotine,
    /// Max-rects packing (slowest, highest density).
    MaxRects,
}

// ---------------------------------------------------------------------------
// TextureAtlas
// ---------------------------------------------------------------------------

/// Configuration for building a texture atlas.
#[derive(Debug, Clone)]
pub struct AtlasConfig {
    /// Maximum width of a single atlas page.
    pub max_page_width: u32,
    /// Maximum height of a single atlas page.
    pub max_page_height: u32,
    /// Padding between entries (in pixels).
    pub padding: u32,
    /// Packing algorithm.
    pub algorithm: PackingAlgorithm,
    /// Packing heuristic (for max-rects).
    pub heuristic: PackingHeuristic,
    /// Rotation mode.
    pub rotation: RotationMode,
    /// Maximum number of pages.
    pub max_pages: u32,
    /// Whether to power-of-two align the final atlas dimensions.
    pub power_of_two: bool,
    /// Whether to try multiple sort orders and pick the best.
    pub try_multiple_orderings: bool,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            max_page_width: 4096,
            max_page_height: 4096,
            padding: 1,
            algorithm: PackingAlgorithm::MaxRects,
            heuristic: PackingHeuristic::BestShortSideFit,
            rotation: RotationMode::Allow90,
            max_pages: 16,
            power_of_two: true,
            try_multiple_orderings: true,
        }
    }
}

/// A texture atlas that can pack multiple sub-textures into one or more pages.
pub struct TextureAtlas {
    config: AtlasConfig,
}

/// Error during atlas packing.
#[derive(Debug, Clone)]
pub enum AtlasError {
    /// A texture is too large to fit on a single page.
    TextureTooLarge {
        index: usize,
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },
    /// All pages are full.
    OutOfPages {
        remaining: usize,
    },
    /// No textures to pack.
    EmptyInput,
}

impl fmt::Display for AtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TextureTooLarge {
                index,
                width,
                height,
                max_width,
                max_height,
            } => write!(
                f,
                "Texture {} ({}x{}) exceeds max page size ({}x{})",
                index, width, height, max_width, max_height
            ),
            Self::OutOfPages { remaining } => {
                write!(f, "Ran out of atlas pages, {} textures remaining", remaining)
            }
            Self::EmptyInput => write!(f, "No textures to pack"),
        }
    }
}

/// Round up to the next power of two.
fn next_power_of_two(v: u32) -> u32 {
    if v == 0 {
        return 1;
    }
    let mut n = v - 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n + 1
}

impl TextureAtlas {
    /// Create a new atlas with the given configuration.
    pub fn new(config: AtlasConfig) -> Self {
        Self { config }
    }

    /// Create an atlas with default settings.
    pub fn with_defaults() -> Self {
        Self::new(AtlasConfig::default())
    }

    /// Pack a list of textures given as (width, height) pairs.
    ///
    /// Returns an `AtlasLayout` describing where each texture was placed.
    pub fn pack(&self, textures: &[(u32, u32)]) -> Result<AtlasLayout, AtlasError> {
        if textures.is_empty() {
            return Err(AtlasError::EmptyInput);
        }

        let max_w = self.config.max_page_width;
        let max_h = self.config.max_page_height;
        let pad = self.config.padding;

        // Validate all textures fit on a page.
        for (i, &(w, h)) in textures.iter().enumerate() {
            let pw = w + pad * 2;
            let ph = h + pad * 2;
            if pw > max_w || ph > max_h {
                // Check rotated.
                if ph > max_w || pw > max_h {
                    return Err(AtlasError::TextureTooLarge {
                        index: i,
                        width: w,
                        height: h,
                        max_width: max_w,
                        max_height: max_h,
                    });
                }
            }
        }

        // Sort textures by area (descending) for better packing.
        let mut sorted_indices: Vec<usize> = (0..textures.len()).collect();
        if self.config.try_multiple_orderings {
            // Try multiple orderings and keep the best.
            let orderings: Vec<Vec<usize>> = self.generate_orderings(textures);
            let mut best_layout: Option<AtlasLayout> = None;
            for ordering in &orderings {
                match self.pack_with_ordering(textures, ordering) {
                    Ok(layout) => {
                        let is_better = best_layout
                            .as_ref()
                            .map_or(true, |best| {
                                layout.page_count < best.page_count
                                    || (layout.page_count == best.page_count
                                        && layout.efficiency > best.efficiency)
                            });
                        if is_better {
                            best_layout = Some(layout);
                        }
                    }
                    Err(_) => continue,
                }
            }
            return best_layout.ok_or(AtlasError::OutOfPages {
                remaining: textures.len(),
            });
        }

        // Default: sort by area descending.
        sorted_indices.sort_by(|&a, &b| {
            let area_a = textures[a].0 as u64 * textures[a].1 as u64;
            let area_b = textures[b].0 as u64 * textures[b].1 as u64;
            area_b.cmp(&area_a)
        });

        self.pack_with_ordering(textures, &sorted_indices)
    }

    /// Generate multiple sort orderings to try.
    fn generate_orderings(&self, textures: &[(u32, u32)]) -> Vec<Vec<usize>> {
        let mut orderings = Vec::new();
        let indices: Vec<usize> = (0..textures.len()).collect();

        // 1. Sort by area descending.
        let mut by_area = indices.clone();
        by_area.sort_by(|&a, &b| {
            let area_a = textures[a].0 as u64 * textures[a].1 as u64;
            let area_b = textures[b].0 as u64 * textures[b].1 as u64;
            area_b.cmp(&area_a)
        });
        orderings.push(by_area);

        // 2. Sort by height descending.
        let mut by_height = indices.clone();
        by_height.sort_by(|&a, &b| textures[b].1.cmp(&textures[a].1));
        orderings.push(by_height);

        // 3. Sort by width descending.
        let mut by_width = indices.clone();
        by_width.sort_by(|&a, &b| textures[b].0.cmp(&textures[a].0));
        orderings.push(by_width);

        // 4. Sort by max side descending.
        let mut by_max_side = indices.clone();
        by_max_side.sort_by(|&a, &b| {
            let max_a = textures[a].0.max(textures[a].1);
            let max_b = textures[b].0.max(textures[b].1);
            max_b.cmp(&max_a)
        });
        orderings.push(by_max_side);

        // 5. Sort by perimeter descending.
        let mut by_perimeter = indices;
        by_perimeter.sort_by(|&a, &b| {
            let p_a = textures[a].0 + textures[a].1;
            let p_b = textures[b].0 + textures[b].1;
            p_b.cmp(&p_a)
        });
        orderings.push(by_perimeter);

        orderings
    }

    /// Pack with a specific ordering of texture indices.
    fn pack_with_ordering(
        &self,
        textures: &[(u32, u32)],
        order: &[usize],
    ) -> Result<AtlasLayout, AtlasError> {
        let max_w = self.config.max_page_width;
        let max_h = self.config.max_page_height;

        let mut entries: Vec<AtlasEntry> = Vec::with_capacity(textures.len());
        let mut pages: Vec<PagePacker> = Vec::new();

        for &src_idx in order {
            let (w, h) = textures[src_idx];
            let mut placed = false;

            // Try existing pages.
            for (page_idx, page) in pages.iter_mut().enumerate() {
                if let Some((rect, rotated)) = page.try_pack(w, h, &self.config) {
                    let (ow, oh) = if rotated { (h, w) } else { (w, h) };
                    entries.push(AtlasEntry {
                        source_index: src_idx,
                        page: page_idx as u32,
                        rect,
                        uv_offset: [
                            rect.x as f32 / max_w as f32,
                            rect.y as f32 / max_h as f32,
                        ],
                        uv_scale: [
                            ow as f32 / max_w as f32,
                            oh as f32 / max_h as f32,
                        ],
                        original_width: w,
                        original_height: h,
                        rotated,
                        padding: self.config.padding,
                    });
                    placed = true;
                    break;
                }
            }

            if !placed {
                // Start a new page.
                if pages.len() as u32 >= self.config.max_pages {
                    return Err(AtlasError::OutOfPages {
                        remaining: order.len() - entries.len(),
                    });
                }
                let mut page = PagePacker::new(max_w, max_h, &self.config);
                if let Some((rect, rotated)) = page.try_pack(w, h, &self.config) {
                    let page_idx = pages.len();
                    let (ow, oh) = if rotated { (h, w) } else { (w, h) };
                    entries.push(AtlasEntry {
                        source_index: src_idx,
                        page: page_idx as u32,
                        rect,
                        uv_offset: [
                            rect.x as f32 / max_w as f32,
                            rect.y as f32 / max_h as f32,
                        ],
                        uv_scale: [
                            ow as f32 / max_w as f32,
                            oh as f32 / max_h as f32,
                        ],
                        original_width: w,
                        original_height: h,
                        rotated,
                        padding: self.config.padding,
                    });
                    pages.push(page);
                } else {
                    return Err(AtlasError::TextureTooLarge {
                        index: src_idx,
                        width: w,
                        height: h,
                        max_width: max_w,
                        max_height: max_h,
                    });
                }
            }
        }

        // Compute final dimensions.
        let page_count = pages.len().max(1) as u32;
        let mut final_w = max_w;
        let mut final_h = max_h;

        if self.config.power_of_two {
            // Find the minimum power-of-two dimensions that contain all entries.
            let actual_max_x = entries
                .iter()
                .map(|e| e.rect.right() + self.config.padding)
                .max()
                .unwrap_or(0);
            let actual_max_y = entries
                .iter()
                .map(|e| e.rect.bottom() + self.config.padding)
                .max()
                .unwrap_or(0);
            final_w = next_power_of_two(actual_max_x).min(max_w);
            final_h = next_power_of_two(actual_max_y).min(max_h);
        }

        // Recompute UVs with final dimensions.
        for entry in &mut entries {
            entry.uv_offset = [
                entry.rect.x as f32 / final_w as f32,
                entry.rect.y as f32 / final_h as f32,
            ];
            let (ow, oh) = if entry.rotated {
                (entry.original_height, entry.original_width)
            } else {
                (entry.original_width, entry.original_height)
            };
            entry.uv_scale = [
                ow as f32 / final_w as f32,
                oh as f32 / final_h as f32,
            ];
        }

        let total_page_area = final_w as u64 * final_h as u64 * page_count as u64;
        let total_used: u64 = entries.iter().map(|e| e.rect.area()).sum();
        let efficiency = if total_page_area > 0 {
            total_used as f32 / total_page_area as f32
        } else {
            0.0
        };
        let wasted = total_page_area.saturating_sub(total_used);

        Ok(AtlasLayout {
            page_width: final_w,
            page_height: final_h,
            page_count,
            entries,
            efficiency,
            wasted_pixels: wasted,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal page packer
// ---------------------------------------------------------------------------

/// Wraps a packer for a single atlas page.
enum PagePacker {
    Shelf(ShelfPacker),
    Guillotine(GuillotinePacker),
    MaxRects(MaxRectsPacker),
}

impl PagePacker {
    fn new(width: u32, height: u32, config: &AtlasConfig) -> Self {
        match config.algorithm {
            PackingAlgorithm::Shelf => {
                PagePacker::Shelf(ShelfPacker::new(width, height, config.padding))
            }
            PackingAlgorithm::Guillotine => {
                PagePacker::Guillotine(GuillotinePacker::new(width, height, config.padding))
            }
            PackingAlgorithm::MaxRects => PagePacker::MaxRects(MaxRectsPacker::new(
                width,
                height,
                config.padding,
                config.heuristic,
                config.rotation,
            )),
        }
    }

    fn try_pack(&mut self, w: u32, h: u32, _config: &AtlasConfig) -> Option<(PackRect, bool)> {
        match self {
            PagePacker::Shelf(packer) => packer.pack(w, h).map(|r| (r, false)),
            PagePacker::Guillotine(packer) => packer.pack(w, h),
            PagePacker::MaxRects(packer) => packer.pack(w, h),
        }
    }
}

// ---------------------------------------------------------------------------
// AtlasBuilder
// ---------------------------------------------------------------------------

/// High-level builder for creating texture atlases.
///
/// Collects textures with associated data, packs them, and produces
/// a lookup table for UV remapping.
pub struct AtlasBuilder<T> {
    /// Collected textures: (width, height, user data).
    textures: Vec<(u32, u32, T)>,
    /// Atlas configuration.
    config: AtlasConfig,
}

impl<T> AtlasBuilder<T> {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            config: AtlasConfig::default(),
        }
    }

    /// Create a new builder with custom configuration.
    pub fn with_config(config: AtlasConfig) -> Self {
        Self {
            textures: Vec::new(),
            config,
        }
    }

    /// Add a texture with associated user data.
    pub fn add(&mut self, width: u32, height: u32, data: T) -> usize {
        let idx = self.textures.len();
        self.textures.push((width, height, data));
        idx
    }

    /// Number of textures added.
    pub fn len(&self) -> usize {
        self.textures.len()
    }

    /// Whether the builder has any textures.
    pub fn is_empty(&self) -> bool {
        self.textures.is_empty()
    }

    /// Build the atlas and return the layout plus the user data in atlas order.
    pub fn build(self) -> Result<(AtlasLayout, Vec<T>), AtlasError> {
        let sizes: Vec<(u32, u32)> = self.textures.iter().map(|(w, h, _)| (*w, *h)).collect();
        let atlas = TextureAtlas::new(self.config);
        let layout = atlas.pack(&sizes)?;

        // Reorder user data to match entry order. Each entry has source_index.
        let mut data_items: Vec<Option<T>> = self.textures.into_iter().map(|(_, _, d)| Some(d)).collect();
        let mut ordered_data = Vec::with_capacity(layout.entries.len());
        for entry in &layout.entries {
            ordered_data.push(
                data_items[entry.source_index]
                    .take()
                    .expect("Each source_index should be used exactly once"),
            );
        }

        Ok((layout, ordered_data))
    }
}

impl<T> Default for AtlasBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SpriteAtlas
// ---------------------------------------------------------------------------

/// A sprite atlas entry with animation metadata.
#[derive(Debug, Clone)]
pub struct SpriteEntry {
    /// Name / identifier of the sprite.
    pub name: String,
    /// Atlas entry with UV coordinates.
    pub atlas_entry: AtlasEntry,
    /// Pivot point in normalised coordinates (0..1).
    pub pivot: [f32; 2],
    /// Pixels per unit (for world-space sizing).
    pub pixels_per_unit: f32,
}

/// A sprite atlas optimised for 2D UI and particle textures.
pub struct SpriteAtlas {
    /// Layout of the atlas.
    pub layout: AtlasLayout,
    /// All sprite entries.
    pub sprites: Vec<SpriteEntry>,
    /// Name -> index lookup.
    name_map: std::collections::HashMap<String, usize>,
}

impl SpriteAtlas {
    /// Create a sprite atlas from an atlas layout and sprite metadata.
    pub fn new(layout: AtlasLayout, sprites: Vec<SpriteEntry>) -> Self {
        let mut name_map = std::collections::HashMap::new();
        for (i, sprite) in sprites.iter().enumerate() {
            name_map.insert(sprite.name.clone(), i);
        }
        Self {
            layout,
            sprites,
            name_map,
        }
    }

    /// Look up a sprite by name.
    pub fn get(&self, name: &str) -> Option<&SpriteEntry> {
        self.name_map.get(name).map(|&i| &self.sprites[i])
    }

    /// Get UV coordinates for a named sprite.
    pub fn uv(&self, name: &str) -> Option<([f32; 2], [f32; 2])> {
        self.get(name)
            .map(|s| (s.atlas_entry.uv_offset, s.atlas_entry.uv_scale))
    }

    /// Number of sprites.
    pub fn len(&self) -> usize {
        self.sprites.len()
    }

    /// Whether the atlas is empty.
    pub fn is_empty(&self) -> bool {
        self.sprites.is_empty()
    }

    /// Get all sprite names.
    pub fn names(&self) -> Vec<&str> {
        self.sprites.iter().map(|s| s.name.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Sprite animation
// ---------------------------------------------------------------------------

/// A sprite animation consisting of a sequence of frames from the atlas.
#[derive(Debug, Clone)]
pub struct SpriteAnimation {
    /// Name of the animation.
    pub name: String,
    /// Indices into the `SpriteAtlas::sprites` array.
    pub frames: Vec<usize>,
    /// Duration of each frame in seconds.
    pub frame_duration: f32,
    /// Whether the animation loops.
    pub looping: bool,
}

impl SpriteAnimation {
    /// Get the frame index at a given time (in seconds).
    pub fn frame_at(&self, time: f32) -> usize {
        if self.frames.is_empty() {
            return 0;
        }
        let total_duration = self.frame_duration * self.frames.len() as f32;
        if total_duration <= 0.0 {
            return self.frames[0];
        }
        let t = if self.looping {
            time % total_duration
        } else {
            time.min(total_duration - 0.001)
        };
        let frame_idx = (t / self.frame_duration) as usize;
        let frame_idx = frame_idx.min(self.frames.len() - 1);
        self.frames[frame_idx]
    }

    /// Total duration of the animation.
    pub fn total_duration(&self) -> f32 {
        self.frame_duration * self.frames.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shelf_packer_basic() {
        let mut packer = ShelfPacker::new(256, 256, 0);
        let r1 = packer.pack(64, 32).expect("Should fit");
        assert_eq!(r1.x, 0);
        assert_eq!(r1.y, 0);
        assert_eq!(r1.w, 64);
        assert_eq!(r1.h, 32);

        let r2 = packer.pack(64, 32).expect("Should fit");
        assert_eq!(r2.x, 64); // same shelf
        assert_eq!(r2.y, 0);
    }

    #[test]
    fn test_shelf_packer_new_shelf() {
        let mut packer = ShelfPacker::new(128, 256, 0);
        packer.pack(128, 32).expect("Should fit");
        let r2 = packer.pack(64, 64).expect("Should fit on new shelf");
        assert_eq!(r2.y, 32); // new shelf below
    }

    #[test]
    fn test_shelf_packer_overflow() {
        let mut packer = ShelfPacker::new(64, 64, 0);
        packer.pack(64, 64).expect("Should fit");
        assert!(packer.pack(1, 1).is_none());
    }

    #[test]
    fn test_shelf_packer_padding() {
        let mut packer = ShelfPacker::new(256, 256, 2);
        let r = packer.pack(32, 32).expect("Should fit");
        assert_eq!(r.x, 2); // padding offset
        assert_eq!(r.y, 2);
        let r2 = packer.pack(32, 32).expect("Should fit");
        assert_eq!(r2.x, 2 + 32 + 2 + 2); // 32 + pad + pad + pad
    }

    #[test]
    fn test_guillotine_packer_basic() {
        let mut packer = GuillotinePacker::new(256, 256, 0);
        let (r1, _) = packer.pack(64, 64).expect("Should fit");
        assert_eq!(r1.x, 0);
        assert_eq!(r1.y, 0);

        let (r2, _) = packer.pack(128, 128).expect("Should fit");
        // Should be placed in the remaining space.
        assert!(r2.x >= 0);
        assert!(!r1.overlaps(&r2));
    }

    #[test]
    fn test_max_rects_basic() {
        let mut packer = MaxRectsPacker::new(
            256,
            256,
            0,
            PackingHeuristic::BestShortSideFit,
            RotationMode::None,
        );
        let (r1, _) = packer.pack(64, 64).expect("Should fit");
        let (r2, _) = packer.pack(64, 64).expect("Should fit");
        assert!(!r1.overlaps(&r2));
    }

    #[test]
    fn test_max_rects_rotation() {
        let mut packer = MaxRectsPacker::new(
            100,
            50,
            0,
            PackingHeuristic::BestShortSideFit,
            RotationMode::Allow90,
        );
        // 80x30 fits normally.
        let (r1, _rot1) = packer.pack(80, 30).expect("Should fit");
        assert_eq!(r1.w + r1.h, 80 + 30); // total dims preserved

        // 40x80 won't fit upright (height 80 > page height 50) but rotated
        // to 80x40 might if there's space. Let's test a simpler case.
        let mut packer2 = MaxRectsPacker::new(
            100,
            50,
            0,
            PackingHeuristic::BestShortSideFit,
            RotationMode::Allow90,
        );
        // 30x90 won't fit either way in 100x50 page.
        let result = packer2.pack(30, 90);
        assert!(result.is_none());

        // But 45x20 should fit rotated if needed.
        let mut packer3 = MaxRectsPacker::new(
            100,
            50,
            0,
            PackingHeuristic::BestShortSideFit,
            RotationMode::Allow90,
        );
        let result3 = packer3.pack(45, 20);
        assert!(result3.is_some());
    }

    #[test]
    fn test_max_rects_many_rects() {
        let mut packer = MaxRectsPacker::new(
            512,
            512,
            1,
            PackingHeuristic::BestShortSideFit,
            RotationMode::None,
        );
        let mut placed = 0;
        for _ in 0..100 {
            if packer.pack(32, 32).is_some() {
                placed += 1;
            }
        }
        // 512x512 with 1px padding, each rect occupies 34x34 effectively.
        // 512/34 = 15 per row, 15 per column = 225 possible. We placed 100.
        assert!(placed >= 50, "Should place at least 50 rects, got {}", placed);
    }

    #[test]
    fn test_atlas_packing() {
        let config = AtlasConfig {
            max_page_width: 512,
            max_page_height: 512,
            padding: 1,
            algorithm: PackingAlgorithm::MaxRects,
            try_multiple_orderings: false,
            ..Default::default()
        };
        let atlas = TextureAtlas::new(config);
        let textures = vec![
            (64, 64),
            (128, 32),
            (32, 128),
            (64, 64),
            (96, 96),
        ];
        let layout = atlas.pack(&textures).expect("Should pack");
        assert_eq!(layout.entries.len(), 5);
        assert!(layout.page_count >= 1);
        assert!(layout.efficiency > 0.0);

        // All entries should be within page bounds.
        for entry in &layout.entries {
            assert!(entry.rect.right() <= layout.page_width);
            assert!(entry.rect.bottom() <= layout.page_height);
        }

        // No overlaps.
        for i in 0..layout.entries.len() {
            for j in (i + 1)..layout.entries.len() {
                if layout.entries[i].page == layout.entries[j].page {
                    let a = &layout.entries[i].rect;
                    let b = &layout.entries[j].rect;
                    assert!(!a.overlaps(b), "Entries {} and {} overlap: {} vs {}", i, j, a, b);
                }
            }
        }
    }

    #[test]
    fn test_atlas_builder() {
        let mut builder: AtlasBuilder<&str> = AtlasBuilder::new();
        builder.add(64, 64, "texture_a");
        builder.add(32, 32, "texture_b");
        builder.add(128, 64, "texture_c");

        let (layout, data) = builder.build().expect("Should build");
        assert_eq!(layout.entries.len(), 3);
        assert_eq!(data.len(), 3);
    }

    #[test]
    fn test_multi_page_atlas() {
        let config = AtlasConfig {
            max_page_width: 128,
            max_page_height: 128,
            padding: 0,
            algorithm: PackingAlgorithm::Shelf,
            max_pages: 4,
            power_of_two: false,
            try_multiple_orderings: false,
            ..Default::default()
        };
        let atlas = TextureAtlas::new(config);
        // Each rect is 64x64, so 4 fit per page. We have 8, so we need 2 pages.
        let textures: Vec<(u32, u32)> = (0..8).map(|_| (64, 64)).collect();
        let layout = atlas.pack(&textures).expect("Should pack");
        assert!(layout.page_count >= 2, "Need at least 2 pages");
        assert_eq!(layout.entries.len(), 8);
    }

    #[test]
    fn test_texture_too_large() {
        let config = AtlasConfig {
            max_page_width: 128,
            max_page_height: 128,
            padding: 0,
            try_multiple_orderings: false,
            ..Default::default()
        };
        let atlas = TextureAtlas::new(config);
        let result = atlas.pack(&[(256, 256)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(255), 256);
        assert_eq!(next_power_of_two(256), 256);
        assert_eq!(next_power_of_two(257), 512);
    }

    #[test]
    fn test_pack_rect_methods() {
        let a = PackRect::positioned(10, 20, 30, 40);
        assert_eq!(a.right(), 40);
        assert_eq!(a.bottom(), 60);
        assert_eq!(a.area(), 1200);
        assert!(a.contains_point(10, 20));
        assert!(a.contains_point(39, 59));
        assert!(!a.contains_point(40, 60));
    }

    #[test]
    fn test_sprite_animation() {
        let anim = SpriteAnimation {
            name: "walk".to_string(),
            frames: vec![0, 1, 2, 3],
            frame_duration: 0.1,
            looping: true,
        };
        assert_eq!(anim.frame_at(0.0), 0);
        assert_eq!(anim.frame_at(0.05), 0);
        assert_eq!(anim.frame_at(0.1), 1);
        assert_eq!(anim.frame_at(0.25), 2);
        assert_eq!(anim.frame_at(0.35), 3);
        // Looping: wraps around.
        assert_eq!(anim.frame_at(0.4), 0);
        assert_eq!(anim.total_duration(), 0.4);
    }
}
