//! Terrain texture painting system for the Genovo engine.
//!
//! Provides a complete brush-based painting toolkit for terrain editing:
//!
//! - **Brush system** — circle, square, and custom shape brushes with
//!   configurable size, falloff, and opacity.
//! - **Splatmap painting** — paint texture layer weights onto the terrain
//!   splatmap for multi-texture blending.
//! - **Blend modes** — add, subtract, smooth, set, and multiply operations.
//! - **Vegetation density** — paint vegetation density maps.
//! - **Terrain holes** — paint holes (non-renderable/non-collidable cells).
//! - **Detail painting** — paint grass, rocks, and other detail layers.
//! - **Undo/redo** — per-stroke undo/redo with history management.

use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// Brush shape
// ---------------------------------------------------------------------------

/// The shape of a terrain painting brush.
#[derive(Debug, Clone)]
pub enum BrushShape {
    /// Circular brush.
    Circle,
    /// Square brush.
    Square,
    /// Diamond (rotated square) brush.
    Diamond,
    /// Custom brush shape defined by a 2D weight mask.
    Custom {
        /// Mask width.
        width: u32,
        /// Mask height.
        height: u32,
        /// Mask weights (row-major, 0.0 .. 1.0).
        mask: Vec<f32>,
    },
}

impl BrushShape {
    /// Creates a custom brush from a grayscale mask.
    pub fn custom(width: u32, height: u32, mask: Vec<f32>) -> Self {
        assert_eq!(
            mask.len(),
            (width * height) as usize,
            "Mask size must match width * height"
        );
        Self::Custom {
            width,
            height,
            mask,
        }
    }

    /// Samples the brush weight at a normalized position (-1..1, -1..1).
    pub fn sample(&self, nx: f32, ny: f32) -> f32 {
        match self {
            Self::Circle => {
                let dist = (nx * nx + ny * ny).sqrt();
                if dist <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Square => {
                if nx.abs() <= 1.0 && ny.abs() <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Diamond => {
                if nx.abs() + ny.abs() <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Custom {
                width,
                height,
                mask,
            } => {
                // Map normalized coords to mask coords.
                let mx = ((nx * 0.5 + 0.5) * (*width as f32 - 1.0)).round() as i32;
                let my = ((ny * 0.5 + 0.5) * (*height as f32 - 1.0)).round() as i32;
                if mx >= 0 && mx < *width as i32 && my >= 0 && my < *height as i32 {
                    mask[(my as u32 * width + mx as u32) as usize]
                } else {
                    0.0
                }
            }
        }
    }
}

impl Default for BrushShape {
    fn default() -> Self {
        Self::Circle
    }
}

// ---------------------------------------------------------------------------
// Brush falloff
// ---------------------------------------------------------------------------

/// Falloff curve for brush edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BrushFalloff {
    /// No falloff -- hard edge.
    None,
    /// Linear falloff from center to edge.
    Linear,
    /// Smooth (Hermite) falloff.
    Smooth,
    /// Quadratic falloff.
    Quadratic,
    /// Custom falloff with an exponent.
    Power(f32),
}

impl BrushFalloff {
    /// Evaluates the falloff for a distance factor (0 = center, 1 = edge).
    pub fn evaluate(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::None => 1.0,
            Self::Linear => 1.0 - t,
            Self::Smooth => {
                let t2 = t * t;
                let t3 = t2 * t;
                1.0 - (3.0 * t2 - 2.0 * t3)
            }
            Self::Quadratic => {
                let inv = 1.0 - t;
                inv * inv
            }
            Self::Power(exp) => (1.0 - t).powf(*exp),
        }
    }
}

impl Default for BrushFalloff {
    fn default() -> Self {
        Self::Smooth
    }
}

// ---------------------------------------------------------------------------
// Blend mode
// ---------------------------------------------------------------------------

/// How the brush affects the existing terrain data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Add brush value to existing value.
    Add,
    /// Subtract brush value from existing value.
    Subtract,
    /// Set to brush value directly.
    Set,
    /// Smooth (average) the area under the brush.
    Smooth,
    /// Multiply existing value by brush value.
    Multiply,
    /// Paint only where current value is below target.
    Max,
    /// Paint only where current value is above target.
    Min,
}

impl BlendMode {
    /// Applies the blend operation.
    pub fn apply(&self, current: f32, brush_value: f32, target: f32) -> f32 {
        match self {
            Self::Add => (current + brush_value).clamp(0.0, 1.0),
            Self::Subtract => (current - brush_value).clamp(0.0, 1.0),
            Self::Set => target * brush_value + current * (1.0 - brush_value),
            Self::Smooth => current, // Handled specially in the paint loop.
            Self::Multiply => (current * (1.0 + brush_value)).clamp(0.0, 1.0),
            Self::Max => current.max(brush_value * target),
            Self::Min => current.min(1.0 - brush_value * (1.0 - target)),
        }
    }
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::Add
    }
}

// ---------------------------------------------------------------------------
// Brush configuration
// ---------------------------------------------------------------------------

/// Complete configuration for a terrain painting brush.
#[derive(Debug, Clone)]
pub struct BrushConfig {
    /// Brush shape.
    pub shape: BrushShape,
    /// Brush radius in world units.
    pub radius: f32,
    /// Falloff curve.
    pub falloff: BrushFalloff,
    /// Falloff start (0..1, how far from center the falloff begins).
    pub falloff_start: f32,
    /// Opacity / strength (0..1).
    pub opacity: f32,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Target value for Set mode (0..1).
    pub target_value: f32,
    /// Rotation angle in degrees.
    pub rotation: f32,
    /// Jitter in position (randomness, 0..1).
    pub jitter: f32,
    /// Spacing between brush stamps (as fraction of radius).
    pub spacing: f32,
    /// Whether to accumulate on held stroke (vs single stamp).
    pub continuous: bool,
}

impl BrushConfig {
    /// Creates a default circular painting brush.
    pub fn new() -> Self {
        Self {
            shape: BrushShape::Circle,
            radius: 10.0,
            falloff: BrushFalloff::Smooth,
            falloff_start: 0.5,
            opacity: 0.5,
            blend_mode: BlendMode::Add,
            target_value: 1.0,
            rotation: 0.0,
            jitter: 0.0,
            spacing: 0.25,
            continuous: true,
        }
    }

    /// Creates a smooth/flatten brush.
    pub fn smooth_brush(radius: f32) -> Self {
        Self {
            shape: BrushShape::Circle,
            radius,
            falloff: BrushFalloff::Smooth,
            falloff_start: 0.3,
            opacity: 0.3,
            blend_mode: BlendMode::Smooth,
            target_value: 0.0,
            rotation: 0.0,
            jitter: 0.0,
            spacing: 0.25,
            continuous: true,
        }
    }

    /// Creates a sharp-edged stamp brush.
    pub fn stamp_brush(radius: f32) -> Self {
        Self {
            shape: BrushShape::Circle,
            radius,
            falloff: BrushFalloff::None,
            falloff_start: 0.0,
            opacity: 1.0,
            blend_mode: BlendMode::Set,
            target_value: 1.0,
            rotation: 0.0,
            jitter: 0.0,
            spacing: 1.0,
            continuous: false,
        }
    }

    /// Evaluates the brush weight at a world-space offset from the brush center.
    pub fn evaluate(&self, offset_x: f32, offset_z: f32) -> f32 {
        let dist = (offset_x * offset_x + offset_z * offset_z).sqrt();
        if dist > self.radius {
            return 0.0;
        }

        // Normalize to -1..1.
        let nx = offset_x / self.radius;
        let nz = offset_z / self.radius;

        // Apply rotation.
        let (nx, nz) = if self.rotation.abs() > 0.001 {
            let cos_r = self.rotation.to_radians().cos();
            let sin_r = self.rotation.to_radians().sin();
            (nx * cos_r - nz * sin_r, nx * sin_r + nz * cos_r)
        } else {
            (nx, nz)
        };

        // Shape mask.
        let shape_weight = self.shape.sample(nx, nz);
        if shape_weight <= 0.0 {
            return 0.0;
        }

        // Falloff.
        let t = dist / self.radius;
        let falloff_weight = if t < self.falloff_start {
            1.0
        } else {
            let falloff_t = (t - self.falloff_start) / (1.0 - self.falloff_start);
            self.falloff.evaluate(falloff_t)
        };

        shape_weight * falloff_weight * self.opacity
    }
}

impl Default for BrushConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Splatmap layer
// ---------------------------------------------------------------------------

/// A single splatmap texture layer (e.g. grass, rock, sand).
#[derive(Debug, Clone)]
pub struct SplatmapLayer {
    /// Layer name.
    pub name: String,
    /// Layer index in the splatmap.
    pub index: u8,
    /// Diffuse/albedo texture identifier.
    pub diffuse_texture: String,
    /// Normal map texture identifier.
    pub normal_texture: Option<String>,
    /// UV tiling scale.
    pub tiling: f32,
    /// Metallic value.
    pub metallic: f32,
    /// Smoothness/roughness value.
    pub smoothness: f32,
}

impl SplatmapLayer {
    /// Creates a new splatmap layer.
    pub fn new(name: &str, index: u8, diffuse: &str) -> Self {
        Self {
            name: name.to_string(),
            index,
            diffuse_texture: diffuse.to_string(),
            normal_texture: None,
            tiling: 1.0,
            metallic: 0.0,
            smoothness: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Terrain canvas (the editable data)
// ---------------------------------------------------------------------------

/// A 2D grid of floating-point values used for painting operations.
///
/// This is the generic "canvas" that splatmap channels, vegetation density
/// maps, hole masks, and detail maps are all painted onto.
#[derive(Debug, Clone)]
pub struct PaintCanvas {
    /// Width in cells.
    pub width: u32,
    /// Height in cells.
    pub height: u32,
    /// Pixel data (row-major, 0.0 .. 1.0 by convention).
    pub data: Vec<f32>,
}

impl PaintCanvas {
    /// Creates a new canvas filled with a value.
    pub fn new(width: u32, height: u32, fill: f32) -> Self {
        Self {
            width,
            height,
            data: vec![fill; (width * height) as usize],
        }
    }

    /// Creates a canvas filled with zeros.
    pub fn zeroed(width: u32, height: u32) -> Self {
        Self::new(width, height, 0.0)
    }

    /// Gets the value at (col, row).
    pub fn get(&self, col: u32, row: u32) -> f32 {
        if col < self.width && row < self.height {
            self.data[(row * self.width + col) as usize]
        } else {
            0.0
        }
    }

    /// Sets the value at (col, row).
    pub fn set(&mut self, col: u32, row: u32, value: f32) {
        if col < self.width && row < self.height {
            self.data[(row * self.width + col) as usize] = value;
        }
    }

    /// Bilinear sample at fractional coordinates.
    pub fn sample(&self, fx: f32, fy: f32) -> f32 {
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let safe_get = |x: i32, y: i32| -> f32 {
            if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
                self.data[(y as u32 * self.width + x as u32) as usize]
            } else {
                0.0
            }
        };

        let v00 = safe_get(x0, y0);
        let v10 = safe_get(x1, y0);
        let v01 = safe_get(x0, y1);
        let v11 = safe_get(x1, y1);

        let v0 = v00 + (v10 - v00) * tx;
        let v1 = v01 + (v11 - v01) * tx;

        v0 + (v1 - v0) * ty
    }

    /// Returns the average value of a rectangular region.
    pub fn average_rect(&self, col_min: u32, row_min: u32, col_max: u32, row_max: u32) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0u32;

        let col_max = col_max.min(self.width - 1);
        let row_max = row_max.min(self.height - 1);

        for row in row_min..=row_max {
            for col in col_min..=col_max {
                sum += self.get(col, row);
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Clears the canvas to a fill value.
    pub fn clear(&mut self, fill: f32) {
        self.data.fill(fill);
    }

    /// Returns the minimum value.
    pub fn min_value(&self) -> f32 {
        self.data.iter().copied().fold(f32::MAX, f32::min)
    }

    /// Returns the maximum value.
    pub fn max_value(&self) -> f32 {
        self.data.iter().copied().fold(f32::MIN, f32::max)
    }

    /// Normalises all values to the 0..1 range.
    pub fn normalize(&mut self) {
        let min = self.min_value();
        let max = self.max_value();
        let range = max - min;
        if range > 1e-9 {
            for v in &mut self.data {
                *v = (*v - min) / range;
            }
        }
    }

    /// Applies a clamp to all values.
    pub fn clamp_values(&mut self, min: f32, max: f32) {
        for v in &mut self.data {
            *v = v.clamp(min, max);
        }
    }

    /// Creates a snapshot of the data (for undo).
    pub fn snapshot(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Restores from a snapshot.
    pub fn restore(&mut self, snapshot: &[f32]) {
        if snapshot.len() == self.data.len() {
            self.data.copy_from_slice(snapshot);
        }
    }
}

// ---------------------------------------------------------------------------
// Paint target
// ---------------------------------------------------------------------------

/// What is being painted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintTarget {
    /// Painting a splatmap layer weight.
    SplatmapLayer(u8),
    /// Painting vegetation density.
    VegetationDensity(u8),
    /// Painting terrain holes.
    Holes,
    /// Painting a detail layer (grass, rocks).
    DetailLayer(u8),
    /// Painting heightmap (raise/lower terrain).
    Height,
}

impl fmt::Display for PaintTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SplatmapLayer(i) => write!(f, "Splatmap Layer {}", i),
            Self::VegetationDensity(i) => write!(f, "Vegetation Density {}", i),
            Self::Holes => write!(f, "Holes"),
            Self::DetailLayer(i) => write!(f, "Detail Layer {}", i),
            Self::Height => write!(f, "Height"),
        }
    }
}

// ---------------------------------------------------------------------------
// Undo / redo
// ---------------------------------------------------------------------------

/// A single undo-able paint stroke.
#[derive(Debug, Clone)]
pub struct PaintStroke {
    /// What was painted.
    pub target: PaintTarget,
    /// Snapshot of the canvas *before* the stroke.
    pub before: Vec<f32>,
    /// Snapshot of the canvas *after* the stroke.
    pub after: Vec<f32>,
    /// Description of the operation.
    pub description: String,
    /// Brush center positions during the stroke (world X, Z).
    pub positions: Vec<(f32, f32)>,
}

/// Undo/redo history for paint operations.
#[derive(Debug)]
pub struct PaintHistory {
    /// Completed strokes (undo stack).
    undo_stack: Vec<PaintStroke>,
    /// Redo stack (cleared on new stroke).
    redo_stack: Vec<PaintStroke>,
    /// Maximum number of undo entries.
    max_entries: usize,
}

impl PaintHistory {
    /// Creates a new history with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_entries,
        }
    }

    /// Records a completed stroke.
    pub fn push_stroke(&mut self, stroke: PaintStroke) {
        self.undo_stack.push(stroke);
        self.redo_stack.clear();

        // Trim to budget.
        while self.undo_stack.len() > self.max_entries {
            self.undo_stack.remove(0);
        }
    }

    /// Undoes the last stroke, returning it (caller must apply `before`).
    pub fn undo(&mut self) -> Option<&PaintStroke> {
        if let Some(stroke) = self.undo_stack.pop() {
            self.redo_stack.push(stroke);
            self.redo_stack.last()
        } else {
            None
        }
    }

    /// Redoes the last undone stroke, returning it (caller must apply `after`).
    pub fn redo(&mut self) -> Option<&PaintStroke> {
        if let Some(stroke) = self.redo_stack.pop() {
            self.undo_stack.push(stroke);
            self.undo_stack.last()
        } else {
            None
        }
    }

    /// Whether an undo is available.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Whether a redo is available.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Returns the number of undo entries.
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Returns the number of redo entries.
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Clears all history.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Returns descriptions of undoable operations.
    pub fn undo_descriptions(&self) -> Vec<&str> {
        self.undo_stack
            .iter()
            .rev()
            .map(|s| s.description.as_str())
            .collect()
    }

    /// Returns descriptions of redoable operations.
    pub fn redo_descriptions(&self) -> Vec<&str> {
        self.redo_stack
            .iter()
            .rev()
            .map(|s| s.description.as_str())
            .collect()
    }
}

impl Default for PaintHistory {
    fn default() -> Self {
        Self::new(50)
    }
}

// ---------------------------------------------------------------------------
// Terrain painter
// ---------------------------------------------------------------------------

/// The main terrain painting system.
///
/// Manages splatmap layers, vegetation density maps, hole masks, detail
/// layers, and provides brush-based painting with full undo/redo support.
pub struct TerrainPainter {
    /// Resolution of the splatmap (width in texels).
    splat_width: u32,
    /// Resolution of the splatmap (height in texels).
    splat_height: u32,
    /// Splatmap layer canvases (one per layer).
    splatmap_layers: Vec<PaintCanvas>,
    /// Splatmap layer definitions.
    layer_defs: Vec<SplatmapLayer>,
    /// Vegetation density canvases (one per vegetation type).
    vegetation_maps: Vec<PaintCanvas>,
    /// Hole mask canvas (>0.5 = hole).
    hole_canvas: PaintCanvas,
    /// Detail layer canvases.
    detail_layers: Vec<PaintCanvas>,
    /// Height canvas (for height painting).
    height_canvas: PaintCanvas,
    /// Current brush configuration.
    brush: BrushConfig,
    /// Undo/redo history.
    history: PaintHistory,
    /// Current stroke tracking.
    current_stroke: Option<StrokeState>,
    /// Terrain world size (for coordinate conversion).
    terrain_world_size: (f32, f32),
    /// Whether splatmap normalization is enforced (layer weights sum to 1).
    normalize_splatmap: bool,
}

/// Internal state for a stroke in progress.
#[derive(Debug, Clone)]
struct StrokeState {
    /// What we are painting.
    target: PaintTarget,
    /// Canvas snapshot before the stroke began.
    before_snapshot: Vec<f32>,
    /// Brush positions during this stroke.
    positions: Vec<(f32, f32)>,
    /// Description.
    description: String,
}

impl TerrainPainter {
    /// Creates a new terrain painter.
    pub fn new(
        splat_width: u32,
        splat_height: u32,
        terrain_world_size: (f32, f32),
    ) -> Self {
        Self {
            splat_width,
            splat_height,
            splatmap_layers: Vec::new(),
            layer_defs: Vec::new(),
            vegetation_maps: Vec::new(),
            hole_canvas: PaintCanvas::zeroed(splat_width, splat_height),
            detail_layers: Vec::new(),
            height_canvas: PaintCanvas::zeroed(splat_width, splat_height),
            brush: BrushConfig::new(),
            history: PaintHistory::new(50),
            current_stroke: None,
            terrain_world_size,
            normalize_splatmap: true,
        }
    }

    // -- Layer management -----------------------------------------------------

    /// Adds a splatmap layer. Returns the layer index.
    pub fn add_splatmap_layer(&mut self, layer: SplatmapLayer) -> u8 {
        let idx = self.splatmap_layers.len() as u8;
        let fill = if idx == 0 { 1.0 } else { 0.0 };
        self.splatmap_layers
            .push(PaintCanvas::new(self.splat_width, self.splat_height, fill));
        self.layer_defs.push(layer);
        idx
    }

    /// Adds a vegetation density layer. Returns the layer index.
    pub fn add_vegetation_layer(&mut self) -> u8 {
        let idx = self.vegetation_maps.len() as u8;
        self.vegetation_maps
            .push(PaintCanvas::zeroed(self.splat_width, self.splat_height));
        idx
    }

    /// Adds a detail layer. Returns the layer index.
    pub fn add_detail_layer(&mut self) -> u8 {
        let idx = self.detail_layers.len() as u8;
        self.detail_layers
            .push(PaintCanvas::zeroed(self.splat_width, self.splat_height));
        idx
    }

    /// Returns the number of splatmap layers.
    pub fn splatmap_layer_count(&self) -> usize {
        self.splatmap_layers.len()
    }

    /// Returns a splatmap layer definition.
    pub fn get_layer_def(&self, index: u8) -> Option<&SplatmapLayer> {
        self.layer_defs.get(index as usize)
    }

    /// Returns a splatmap canvas.
    pub fn get_splatmap(&self, index: u8) -> Option<&PaintCanvas> {
        self.splatmap_layers.get(index as usize)
    }

    /// Returns the hole canvas.
    pub fn get_hole_canvas(&self) -> &PaintCanvas {
        &self.hole_canvas
    }

    /// Returns the vegetation density canvas.
    pub fn get_vegetation_map(&self, index: u8) -> Option<&PaintCanvas> {
        self.vegetation_maps.get(index as usize)
    }

    /// Returns the height canvas.
    pub fn get_height_canvas(&self) -> &PaintCanvas {
        &self.height_canvas
    }

    // -- Brush management -----------------------------------------------------

    /// Sets the active brush configuration.
    pub fn set_brush(&mut self, brush: BrushConfig) {
        self.brush = brush;
    }

    /// Returns the current brush configuration.
    pub fn brush(&self) -> &BrushConfig {
        &self.brush
    }

    /// Returns a mutable reference to the brush configuration.
    pub fn brush_mut(&mut self) -> &mut BrushConfig {
        &mut self.brush
    }

    /// Sets the brush radius.
    pub fn set_brush_radius(&mut self, radius: f32) {
        self.brush.radius = radius.max(0.1);
    }

    /// Sets the brush opacity.
    pub fn set_brush_opacity(&mut self, opacity: f32) {
        self.brush.opacity = opacity.clamp(0.0, 1.0);
    }

    /// Sets the blend mode.
    pub fn set_blend_mode(&mut self, mode: BlendMode) {
        self.brush.blend_mode = mode;
    }

    // -- Coordinate conversion ------------------------------------------------

    /// Converts world-space XZ to canvas cell coordinates.
    fn world_to_canvas(&self, world_x: f32, world_z: f32) -> (f32, f32) {
        let nx = world_x / self.terrain_world_size.0;
        let nz = world_z / self.terrain_world_size.1;
        (nx * self.splat_width as f32, nz * self.splat_height as f32)
    }

    // -- Painting operations --------------------------------------------------

    /// Begins a new paint stroke.
    pub fn begin_stroke(&mut self, target: PaintTarget, description: &str) {
        let snapshot = match target {
            PaintTarget::SplatmapLayer(idx) => {
                self.splatmap_layers
                    .get(idx as usize)
                    .map(|c| c.snapshot())
                    .unwrap_or_default()
            }
            PaintTarget::VegetationDensity(idx) => {
                self.vegetation_maps
                    .get(idx as usize)
                    .map(|c| c.snapshot())
                    .unwrap_or_default()
            }
            PaintTarget::Holes => self.hole_canvas.snapshot(),
            PaintTarget::DetailLayer(idx) => {
                self.detail_layers
                    .get(idx as usize)
                    .map(|c| c.snapshot())
                    .unwrap_or_default()
            }
            PaintTarget::Height => self.height_canvas.snapshot(),
        };

        self.current_stroke = Some(StrokeState {
            target,
            before_snapshot: snapshot,
            positions: Vec::new(),
            description: description.to_string(),
        });
    }

    /// Applies a single brush stamp at a world position.
    pub fn paint_at(&mut self, world_x: f32, world_z: f32) {
        let target = match &self.current_stroke {
            Some(s) => s.target,
            None => return,
        };

        if let Some(ref mut stroke) = self.current_stroke {
            stroke.positions.push((world_x, world_z));
        }

        let (cx, cz) = self.world_to_canvas(world_x, world_z);
        let radius_cells_x = self.brush.radius / self.terrain_world_size.0 * self.splat_width as f32;
        let radius_cells_z = self.brush.radius / self.terrain_world_size.1 * self.splat_height as f32;

        let col_min = (cx - radius_cells_x).floor().max(0.0) as u32;
        let col_max = (cx + radius_cells_x).ceil().min(self.splat_width as f32 - 1.0) as u32;
        let row_min = (cz - radius_cells_z).floor().max(0.0) as u32;
        let row_max = (cz + radius_cells_z).ceil().min(self.splat_height as f32 - 1.0) as u32;

        let cell_size_x = self.terrain_world_size.0 / self.splat_width as f32;
        let cell_size_z = self.terrain_world_size.1 / self.splat_height as f32;

        // Compute smooth values if needed.
        let smooth_avg = if self.brush.blend_mode == BlendMode::Smooth {
            match target {
                PaintTarget::SplatmapLayer(idx) => {
                    self.splatmap_layers
                        .get(idx as usize)
                        .map(|c| c.average_rect(col_min, row_min, col_max, row_max))
                }
                PaintTarget::Height => {
                    Some(self.height_canvas.average_rect(col_min, row_min, col_max, row_max))
                }
                _ => None,
            }
        } else {
            None
        };

        for row in row_min..=row_max {
            for col in col_min..=col_max {
                let cell_world_x = col as f32 * cell_size_x;
                let cell_world_z = row as f32 * cell_size_z;
                let offset_x = cell_world_x - world_x;
                let offset_z = cell_world_z - world_z;

                let weight = self.brush.evaluate(offset_x, offset_z);
                if weight <= 0.001 {
                    continue;
                }

                match target {
                    PaintTarget::SplatmapLayer(idx) => {
                        if let Some(canvas) = self.splatmap_layers.get_mut(idx as usize) {
                            let current = canvas.get(col, row);
                            let new_val = if self.brush.blend_mode == BlendMode::Smooth {
                                let avg = smooth_avg.unwrap_or(current);
                                current + (avg - current) * weight
                            } else {
                                self.brush.blend_mode.apply(
                                    current,
                                    weight,
                                    self.brush.target_value,
                                )
                            };
                            canvas.set(col, row, new_val.clamp(0.0, 1.0));
                        }
                    }
                    PaintTarget::VegetationDensity(idx) => {
                        if let Some(canvas) = self.vegetation_maps.get_mut(idx as usize) {
                            let current = canvas.get(col, row);
                            let new_val = self.brush.blend_mode.apply(
                                current,
                                weight,
                                self.brush.target_value,
                            );
                            canvas.set(col, row, new_val.clamp(0.0, 1.0));
                        }
                    }
                    PaintTarget::Holes => {
                        let current = self.hole_canvas.get(col, row);
                        let new_val = self.brush.blend_mode.apply(
                            current,
                            weight,
                            self.brush.target_value,
                        );
                        self.hole_canvas.set(col, row, new_val.clamp(0.0, 1.0));
                    }
                    PaintTarget::DetailLayer(idx) => {
                        if let Some(canvas) = self.detail_layers.get_mut(idx as usize) {
                            let current = canvas.get(col, row);
                            let new_val = self.brush.blend_mode.apply(
                                current,
                                weight,
                                self.brush.target_value,
                            );
                            canvas.set(col, row, new_val.clamp(0.0, 1.0));
                        }
                    }
                    PaintTarget::Height => {
                        let current = self.height_canvas.get(col, row);
                        let new_val = if self.brush.blend_mode == BlendMode::Smooth {
                            let avg = smooth_avg.unwrap_or(current);
                            current + (avg - current) * weight
                        } else {
                            self.brush.blend_mode.apply(
                                current,
                                weight,
                                self.brush.target_value,
                            )
                        };
                        self.height_canvas.set(col, row, new_val);
                    }
                }
            }
        }

        // Normalize splatmap if needed.
        if self.normalize_splatmap {
            if let PaintTarget::SplatmapLayer(_) = target {
                self.normalize_splatmap_region(col_min, row_min, col_max, row_max);
            }
        }
    }

    /// Ends the current stroke and records it in the undo history.
    pub fn end_stroke(&mut self) {
        if let Some(stroke_state) = self.current_stroke.take() {
            let after = match stroke_state.target {
                PaintTarget::SplatmapLayer(idx) => {
                    self.splatmap_layers
                        .get(idx as usize)
                        .map(|c| c.snapshot())
                        .unwrap_or_default()
                }
                PaintTarget::VegetationDensity(idx) => {
                    self.vegetation_maps
                        .get(idx as usize)
                        .map(|c| c.snapshot())
                        .unwrap_or_default()
                }
                PaintTarget::Holes => self.hole_canvas.snapshot(),
                PaintTarget::DetailLayer(idx) => {
                    self.detail_layers
                        .get(idx as usize)
                        .map(|c| c.snapshot())
                        .unwrap_or_default()
                }
                PaintTarget::Height => self.height_canvas.snapshot(),
            };

            let stroke = PaintStroke {
                target: stroke_state.target,
                before: stroke_state.before_snapshot,
                after,
                description: stroke_state.description,
                positions: stroke_state.positions,
            };

            self.history.push_stroke(stroke);
        }
    }

    /// Normalises splatmap layers so all weights at each texel sum to 1.0.
    fn normalize_splatmap_region(
        &mut self,
        col_min: u32,
        row_min: u32,
        col_max: u32,
        row_max: u32,
    ) {
        let num_layers = self.splatmap_layers.len();
        if num_layers == 0 {
            return;
        }

        for row in row_min..=row_max {
            for col in col_min..=col_max {
                let mut total = 0.0f32;
                for layer in &self.splatmap_layers {
                    total += layer.get(col, row);
                }
                if total > 1e-6 {
                    for layer in &mut self.splatmap_layers {
                        let v = layer.get(col, row);
                        layer.set(col, row, v / total);
                    }
                }
            }
        }
    }

    /// Normalises the entire splatmap.
    pub fn normalize_splatmap_full(&mut self) {
        self.normalize_splatmap_region(0, 0, self.splat_width - 1, self.splat_height - 1);
    }

    // -- Undo / redo ----------------------------------------------------------

    /// Undoes the last stroke.
    pub fn undo(&mut self) -> bool {
        if let Some(stroke) = self.history.undo() {
            let target = stroke.target;
            let before = stroke.before.clone();
            self.apply_snapshot(target, &before);
            true
        } else {
            false
        }
    }

    /// Redoes the last undone stroke.
    pub fn redo(&mut self) -> bool {
        if let Some(stroke) = self.history.redo() {
            let target = stroke.target;
            let after = stroke.after.clone();
            self.apply_snapshot(target, &after);
            true
        } else {
            false
        }
    }

    /// Applies a data snapshot to the appropriate canvas.
    fn apply_snapshot(&mut self, target: PaintTarget, data: &[f32]) {
        match target {
            PaintTarget::SplatmapLayer(idx) => {
                if let Some(canvas) = self.splatmap_layers.get_mut(idx as usize) {
                    canvas.restore(data);
                }
            }
            PaintTarget::VegetationDensity(idx) => {
                if let Some(canvas) = self.vegetation_maps.get_mut(idx as usize) {
                    canvas.restore(data);
                }
            }
            PaintTarget::Holes => {
                self.hole_canvas.restore(data);
            }
            PaintTarget::DetailLayer(idx) => {
                if let Some(canvas) = self.detail_layers.get_mut(idx as usize) {
                    canvas.restore(data);
                }
            }
            PaintTarget::Height => {
                self.height_canvas.restore(data);
            }
        }
    }

    /// Whether an undo is available.
    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    /// Whether a redo is available.
    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Returns undo descriptions.
    pub fn undo_descriptions(&self) -> Vec<&str> {
        self.history.undo_descriptions()
    }

    // -- Queries --------------------------------------------------------------

    /// Returns the splatmap layer weights at a texel position.
    pub fn get_weights_at(&self, col: u32, row: u32) -> Vec<f32> {
        self.splatmap_layers
            .iter()
            .map(|c| c.get(col, row))
            .collect()
    }

    /// Returns the dominant splatmap layer index at a texel position.
    pub fn dominant_layer_at(&self, col: u32, row: u32) -> Option<u8> {
        let weights = self.get_weights_at(col, row);
        weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u8)
    }

    /// Whether a cell is a hole.
    pub fn is_hole(&self, col: u32, row: u32) -> bool {
        self.hole_canvas.get(col, row) > 0.5
    }

    /// Returns the resolution of the splatmap.
    pub fn resolution(&self) -> (u32, u32) {
        (self.splat_width, self.splat_height)
    }

    /// Whether splatmap normalisation is enabled.
    pub fn normalize_enabled(&self) -> bool {
        self.normalize_splatmap
    }

    /// Sets whether splatmap normalisation is enabled.
    pub fn set_normalize_enabled(&mut self, enabled: bool) {
        self.normalize_splatmap = enabled;
    }

    /// Returns the paint history.
    pub fn history(&self) -> &PaintHistory {
        &self.history
    }

    /// Clears all paint data.
    pub fn clear_all(&mut self) {
        for (i, canvas) in self.splatmap_layers.iter_mut().enumerate() {
            canvas.clear(if i == 0 { 1.0 } else { 0.0 });
        }
        for canvas in &mut self.vegetation_maps {
            canvas.clear(0.0);
        }
        self.hole_canvas.clear(0.0);
        for canvas in &mut self.detail_layers {
            canvas.clear(0.0);
        }
        self.height_canvas.clear(0.0);
        self.history.clear();
    }
}

impl fmt::Debug for TerrainPainter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TerrainPainter")
            .field("resolution", &(self.splat_width, self.splat_height))
            .field("splatmap_layers", &self.splatmap_layers.len())
            .field("vegetation_maps", &self.vegetation_maps.len())
            .field("detail_layers", &self.detail_layers.len())
            .field("undo_count", &self.history.undo_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_painter() -> TerrainPainter {
        let mut painter = TerrainPainter::new(64, 64, (100.0, 100.0));
        painter.add_splatmap_layer(SplatmapLayer::new("Grass", 0, "grass_diffuse"));
        painter.add_splatmap_layer(SplatmapLayer::new("Rock", 1, "rock_diffuse"));
        painter.add_splatmap_layer(SplatmapLayer::new("Sand", 2, "sand_diffuse"));
        painter
    }

    #[test]
    fn brush_shape_circle() {
        let shape = BrushShape::Circle;
        assert_eq!(shape.sample(0.0, 0.0), 1.0);
        assert_eq!(shape.sample(0.5, 0.0), 1.0);
        assert_eq!(shape.sample(1.5, 0.0), 0.0);
    }

    #[test]
    fn brush_shape_square() {
        let shape = BrushShape::Square;
        assert_eq!(shape.sample(0.9, 0.9), 1.0);
        assert_eq!(shape.sample(1.1, 0.0), 0.0);
    }

    #[test]
    fn brush_shape_diamond() {
        let shape = BrushShape::Diamond;
        assert_eq!(shape.sample(0.0, 0.0), 1.0);
        assert_eq!(shape.sample(0.5, 0.5), 1.0);
        assert_eq!(shape.sample(0.6, 0.6), 0.0);
    }

    #[test]
    fn brush_falloff_linear() {
        let falloff = BrushFalloff::Linear;
        assert!((falloff.evaluate(0.0) - 1.0).abs() < 0.001);
        assert!((falloff.evaluate(0.5) - 0.5).abs() < 0.001);
        assert!((falloff.evaluate(1.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn brush_falloff_smooth() {
        let falloff = BrushFalloff::Smooth;
        assert!((falloff.evaluate(0.0) - 1.0).abs() < 0.001);
        assert!((falloff.evaluate(1.0) - 0.0).abs() < 0.001);
        // Smooth should be S-shaped.
        let mid = falloff.evaluate(0.5);
        assert!(mid > 0.4 && mid < 0.6);
    }

    #[test]
    fn brush_evaluate() {
        let brush = BrushConfig {
            radius: 10.0,
            opacity: 1.0,
            falloff: BrushFalloff::None,
            falloff_start: 0.0,
            shape: BrushShape::Circle,
            ..BrushConfig::default()
        };

        // Center should be full strength.
        let w = brush.evaluate(0.0, 0.0);
        assert!((w - 1.0).abs() < 0.01);

        // Outside radius should be 0.
        let w = brush.evaluate(15.0, 0.0);
        assert!((w - 0.0).abs() < 0.01);
    }

    #[test]
    fn paint_canvas_basic() {
        let mut canvas = PaintCanvas::zeroed(16, 16);
        assert!((canvas.get(5, 5) - 0.0).abs() < 0.001);

        canvas.set(5, 5, 0.75);
        assert!((canvas.get(5, 5) - 0.75).abs() < 0.001);
    }

    #[test]
    fn paint_canvas_average() {
        let mut canvas = PaintCanvas::zeroed(4, 4);
        canvas.set(0, 0, 1.0);
        canvas.set(1, 0, 1.0);
        canvas.set(0, 1, 1.0);
        canvas.set(1, 1, 1.0);
        let avg = canvas.average_rect(0, 0, 1, 1);
        assert!((avg - 1.0).abs() < 0.001);
    }

    #[test]
    fn paint_canvas_snapshot_restore() {
        let mut canvas = PaintCanvas::zeroed(4, 4);
        let snapshot = canvas.snapshot();
        canvas.set(2, 2, 0.9);
        assert!((canvas.get(2, 2) - 0.9).abs() < 0.001);
        canvas.restore(&snapshot);
        assert!((canvas.get(2, 2) - 0.0).abs() < 0.001);
    }

    #[test]
    fn painter_add_layers() {
        let painter = make_painter();
        assert_eq!(painter.splatmap_layer_count(), 3);
        assert!(painter.get_layer_def(0).is_some());
        assert_eq!(painter.get_layer_def(0).unwrap().name, "Grass");
    }

    #[test]
    fn painter_paint_splatmap() {
        let mut painter = make_painter();

        painter.set_brush(BrushConfig {
            radius: 20.0,
            opacity: 1.0,
            blend_mode: BlendMode::Set,
            target_value: 1.0,
            falloff: BrushFalloff::None,
            falloff_start: 0.0,
            shape: BrushShape::Circle,
            ..BrushConfig::default()
        });

        painter.begin_stroke(PaintTarget::SplatmapLayer(1), "Paint rock");
        painter.paint_at(50.0, 50.0);
        painter.end_stroke();

        // Rock layer should have been painted near center.
        let weights = painter.get_weights_at(32, 32);
        assert!(weights[1] > 0.0);
    }

    #[test]
    fn painter_undo_redo() {
        let mut painter = make_painter();

        painter.set_brush(BrushConfig {
            radius: 10.0,
            opacity: 1.0,
            blend_mode: BlendMode::Set,
            target_value: 1.0,
            falloff: BrushFalloff::None,
            ..BrushConfig::default()
        });

        painter.begin_stroke(PaintTarget::SplatmapLayer(1), "Paint");
        painter.paint_at(50.0, 50.0);
        painter.end_stroke();

        assert!(painter.can_undo());

        // Undo.
        assert!(painter.undo());
        assert!(painter.can_redo());

        // Redo.
        assert!(painter.redo());
    }

    #[test]
    fn painter_paint_holes() {
        let mut painter = make_painter();

        painter.set_brush(BrushConfig {
            radius: 10.0,
            opacity: 1.0,
            blend_mode: BlendMode::Set,
            target_value: 1.0,
            falloff: BrushFalloff::None,
            ..BrushConfig::default()
        });

        painter.begin_stroke(PaintTarget::Holes, "Paint holes");
        painter.paint_at(50.0, 50.0);
        painter.end_stroke();

        // Center should be a hole.
        assert!(painter.is_hole(32, 32));
    }

    #[test]
    fn painter_vegetation() {
        let mut painter = make_painter();
        let veg_idx = painter.add_vegetation_layer();

        painter.set_brush(BrushConfig {
            radius: 15.0,
            opacity: 0.8,
            blend_mode: BlendMode::Add,
            ..BrushConfig::default()
        });

        painter.begin_stroke(PaintTarget::VegetationDensity(veg_idx), "Paint veg");
        painter.paint_at(50.0, 50.0);
        painter.end_stroke();

        let veg_map = painter.get_vegetation_map(veg_idx).unwrap();
        assert!(veg_map.get(32, 32) > 0.0);
    }

    #[test]
    fn blend_mode_add() {
        let result = BlendMode::Add.apply(0.3, 0.2, 1.0);
        assert!((result - 0.5).abs() < 0.01);
    }

    #[test]
    fn blend_mode_subtract() {
        let result = BlendMode::Subtract.apply(0.5, 0.2, 0.0);
        assert!((result - 0.3).abs() < 0.01);
    }

    #[test]
    fn blend_mode_set() {
        let result = BlendMode::Set.apply(0.0, 1.0, 0.8);
        assert!((result - 0.8).abs() < 0.01);
    }

    #[test]
    fn splatmap_normalization() {
        let mut painter = make_painter();
        painter.set_normalize_enabled(true);
        painter.normalize_splatmap_full();

        // After normalization, layer weights should sum to 1.0 at each texel.
        let weights = painter.get_weights_at(0, 0);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn dominant_layer() {
        let painter = make_painter();
        // Layer 0 (grass) is initialized to 1.0, others to 0.0.
        assert_eq!(painter.dominant_layer_at(0, 0), Some(0));
    }

    #[test]
    fn paint_history() {
        let mut history = PaintHistory::new(3);
        assert!(!history.can_undo());

        for i in 0..5 {
            history.push_stroke(PaintStroke {
                target: PaintTarget::Height,
                before: vec![0.0],
                after: vec![1.0],
                description: format!("Stroke {}", i),
                positions: vec![(0.0, 0.0)],
            });
        }

        // Should be capped at 3.
        assert_eq!(history.undo_count(), 3);
    }

    #[test]
    fn clear_all() {
        let mut painter = make_painter();
        painter.clear_all();
        let weights = painter.get_weights_at(0, 0);
        // Layer 0 should be 1.0, rest 0.0.
        assert!((weights[0] - 1.0).abs() < 0.001);
        assert!((weights[1]).abs() < 0.001);
    }
}
