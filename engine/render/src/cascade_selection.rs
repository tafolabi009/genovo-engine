// engine/render/src/cascade_selection.rs
//
// Shadow cascade management for the Genovo engine.
//
// Implements shadow map cascade selection and management for directional lights:
//
// - **Split schemes** -- Uniform, logarithmic, and Practical Split Scheme Maps
//   (PSSM) for distributing cascade boundaries across the view frustum.
// - **Per-cascade viewports** -- Each cascade has its own viewport with
//   configurable resolution and shadow map region.
// - **Cascade blending** -- Smooth blending at cascade boundaries to avoid
//   visible seams between shadow quality levels.
// - **Cascade stabilization** -- Eliminates shadow "swimming" by snapping
//   cascade projection matrices to texel-aligned boundaries.
// - **Visualization colors** -- Debug colors per cascade for visual debugging.
// - **Quality per cascade** -- Different filtering quality per cascade level
//   (e.g., PCF 5x5 for near, 3x3 for mid, 2x2 for far).

use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of shadow cascades supported.
pub const MAX_CASCADES: usize = 8;

/// Default cascade count for most scenes.
pub const DEFAULT_CASCADE_COUNT: usize = 4;

/// Default shadow map resolution per cascade.
pub const DEFAULT_CASCADE_RESOLUTION: u32 = 2048;

/// Default cascade blend width in world units.
const DEFAULT_BLEND_WIDTH: f32 = 2.0;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Split scheme
// ---------------------------------------------------------------------------

/// Strategy for distributing cascade split distances across the view frustum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitScheme {
    /// Uniform distribution: each cascade covers an equal range of depth.
    Uniform,
    /// Logarithmic distribution: each cascade covers an exponentially
    /// increasing depth range, giving more resolution to nearby geometry.
    Logarithmic,
    /// Practical Split Scheme Maps (PSSM): blends uniform and logarithmic
    /// using a lambda parameter (0.0 = uniform, 1.0 = logarithmic).
    Pssm { lambda: f32 },
    /// Manual split distances specified directly by the user.
    Manual,
}

impl Default for SplitScheme {
    fn default() -> Self {
        Self::Pssm { lambda: 0.5 }
    }
}

impl fmt::Display for SplitScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uniform => write!(f, "Uniform"),
            Self::Logarithmic => write!(f, "Logarithmic"),
            Self::Pssm { lambda } => write!(f, "PSSM(lambda={:.2})", lambda),
            Self::Manual => write!(f, "Manual"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow quality per cascade
// ---------------------------------------------------------------------------

/// Shadow filtering quality level applied per cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShadowFilterQuality {
    /// No filtering (hard shadows).
    Hard,
    /// 2x2 PCF.
    Pcf2x2,
    /// 3x3 PCF.
    Pcf3x3,
    /// 5x5 PCF.
    Pcf5x5,
    /// 7x7 PCF (expensive, highest quality).
    Pcf7x7,
    /// Percentage-closer soft shadows.
    Pcss,
    /// Variance shadow maps.
    Vsm,
}

impl Default for ShadowFilterQuality {
    fn default() -> Self {
        Self::Pcf3x3
    }
}

impl ShadowFilterQuality {
    /// Returns the number of shadow map samples for this quality level.
    pub fn sample_count(&self) -> u32 {
        match self {
            Self::Hard => 1,
            Self::Pcf2x2 => 4,
            Self::Pcf3x3 => 9,
            Self::Pcf5x5 => 25,
            Self::Pcf7x7 => 49,
            Self::Pcss => 32,
            Self::Vsm => 1,
        }
    }

    /// Returns the approximate relative GPU cost (1.0 = baseline).
    pub fn relative_cost(&self) -> f32 {
        match self {
            Self::Hard => 0.5,
            Self::Pcf2x2 => 1.0,
            Self::Pcf3x3 => 1.5,
            Self::Pcf5x5 => 3.0,
            Self::Pcf7x7 => 5.5,
            Self::Pcss => 4.0,
            Self::Vsm => 1.2,
        }
    }
}

// ---------------------------------------------------------------------------
// Cascade viewport
// ---------------------------------------------------------------------------

/// Viewport rectangle within a shadow atlas for a single cascade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CascadeViewport {
    /// X offset in the shadow atlas (pixels).
    pub x: u32,
    /// Y offset in the shadow atlas (pixels).
    pub y: u32,
    /// Width of this cascade's region in the atlas (pixels).
    pub width: u32,
    /// Height of this cascade's region in the atlas (pixels).
    pub height: u32,
}

impl CascadeViewport {
    /// Create a new cascade viewport.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }

    /// Returns the area in pixels.
    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Returns UV coordinates for this viewport within an atlas of the given size.
    pub fn uv_rect(&self, atlas_width: u32, atlas_height: u32) -> [f32; 4] {
        let inv_w = 1.0 / atlas_width as f32;
        let inv_h = 1.0 / atlas_height as f32;
        [
            self.x as f32 * inv_w,
            self.y as f32 * inv_h,
            self.width as f32 * inv_w,
            self.height as f32 * inv_h,
        ]
    }

    /// Check if a point (in atlas pixels) falls within this viewport.
    pub fn contains_pixel(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }
}

impl Default for CascadeViewport {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: DEFAULT_CASCADE_RESOLUTION,
            height: DEFAULT_CASCADE_RESOLUTION,
        }
    }
}

// ---------------------------------------------------------------------------
// Cascade data
// ---------------------------------------------------------------------------

/// Visualization color for a cascade (RGBA, 0..1).
#[derive(Debug, Clone, Copy)]
pub struct CascadeColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl CascadeColor {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

/// Default cascade debug colors (red, green, blue, yellow, cyan, magenta, orange, white).
const CASCADE_COLORS: [CascadeColor; MAX_CASCADES] = [
    CascadeColor::new(1.0, 0.0, 0.0, 0.3),
    CascadeColor::new(0.0, 1.0, 0.0, 0.3),
    CascadeColor::new(0.0, 0.0, 1.0, 0.3),
    CascadeColor::new(1.0, 1.0, 0.0, 0.3),
    CascadeColor::new(0.0, 1.0, 1.0, 0.3),
    CascadeColor::new(1.0, 0.0, 1.0, 0.3),
    CascadeColor::new(1.0, 0.5, 0.0, 0.3),
    CascadeColor::new(1.0, 1.0, 1.0, 0.3),
];

/// Per-cascade state and computed data.
#[derive(Debug, Clone)]
pub struct CascadeData {
    /// Index of this cascade (0 = nearest).
    pub index: usize,
    /// Near split distance (world units from the camera).
    pub near_distance: f32,
    /// Far split distance (world units from the camera).
    pub far_distance: f32,
    /// Viewport within the shadow atlas.
    pub viewport: CascadeViewport,
    /// Shadow filter quality for this cascade.
    pub filter_quality: ShadowFilterQuality,
    /// View-projection matrix for this cascade's light frustum.
    pub view_projection: [f32; 16],
    /// Cascade texel size in world units.
    pub texel_size: f32,
    /// Whether this cascade is active (can be disabled for LOD).
    pub active: bool,
    /// Debug visualization color.
    pub debug_color: CascadeColor,
    /// Stabilization offset used to snap projection to texel grid.
    pub stabilization_offset: [f32; 2],
    /// Depth bias for this cascade.
    pub depth_bias: f32,
    /// Normal bias for this cascade.
    pub normal_bias: f32,
}

impl CascadeData {
    /// Create a new cascade with default values.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            near_distance: 0.0,
            far_distance: 100.0,
            viewport: CascadeViewport::default(),
            filter_quality: ShadowFilterQuality::default(),
            view_projection: [0.0; 16],
            texel_size: 0.0,
            active: true,
            debug_color: CASCADE_COLORS[index % MAX_CASCADES],
            stabilization_offset: [0.0; 2],
            depth_bias: 0.005,
            normal_bias: 0.02,
        }
    }

    /// Returns the depth range covered by this cascade.
    pub fn depth_range(&self) -> f32 {
        self.far_distance - self.near_distance
    }

    /// Returns the ratio of far to near distance.
    pub fn depth_ratio(&self) -> f32 {
        if self.near_distance.abs() < EPSILON {
            self.far_distance / EPSILON
        } else {
            self.far_distance / self.near_distance
        }
    }

    /// Returns the center distance of this cascade.
    pub fn center_distance(&self) -> f32 {
        (self.near_distance + self.far_distance) * 0.5
    }

    /// Computes the blend factor at a given depth for cascade boundary blending.
    /// Returns 0.0 when fully inside this cascade, 1.0 at the far boundary.
    pub fn blend_factor(&self, depth: f32, blend_width: f32) -> f32 {
        if blend_width <= EPSILON {
            return 0.0;
        }
        let blend_start = self.far_distance - blend_width;
        if depth <= blend_start {
            0.0
        } else if depth >= self.far_distance {
            1.0
        } else {
            (depth - blend_start) / blend_width
        }
    }
}

// ---------------------------------------------------------------------------
// Cascade blend settings
// ---------------------------------------------------------------------------

/// Blending mode for cascade transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CascadeBlendMode {
    /// No blending -- hard cut between cascades.
    None,
    /// Linear blend in a region near cascade boundaries.
    Linear,
    /// Smooth (hermite) blend at cascade boundaries.
    Smooth,
    /// Dither-based blend (noise pattern to hide seams).
    Dither,
}

impl Default for CascadeBlendMode {
    fn default() -> Self {
        Self::Linear
    }
}

/// Configuration for cascade boundary blending.
#[derive(Debug, Clone, Copy)]
pub struct CascadeBlendConfig {
    /// Blend mode used at cascade boundaries.
    pub mode: CascadeBlendMode,
    /// Width of the blend region in world units.
    pub blend_width: f32,
    /// Dither pattern scale (only used with Dither mode).
    pub dither_scale: f32,
    /// Whether to visualize blend regions.
    pub visualize_blend: bool,
}

impl Default for CascadeBlendConfig {
    fn default() -> Self {
        Self {
            mode: CascadeBlendMode::Linear,
            blend_width: DEFAULT_BLEND_WIDTH,
            dither_scale: 1.0,
            visualize_blend: false,
        }
    }
}

impl CascadeBlendConfig {
    /// Compute the blend factor between two cascades at a given depth.
    pub fn compute_blend(&self, depth: f32, near_cascade: &CascadeData, _far_cascade: &CascadeData) -> f32 {
        let raw = near_cascade.blend_factor(depth, self.blend_width);
        match self.mode {
            CascadeBlendMode::None => 0.0,
            CascadeBlendMode::Linear => raw,
            CascadeBlendMode::Smooth => hermite_smooth(raw),
            CascadeBlendMode::Dither => {
                // Dither uses the raw factor to drive a threshold pattern
                raw
            }
        }
    }
}

/// Hermite smoothstep function for smooth cascade transitions.
fn hermite_smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Stabilization
// ---------------------------------------------------------------------------

/// Cascade stabilization settings to prevent shadow "swimming."
#[derive(Debug, Clone, Copy)]
pub struct StabilizationConfig {
    /// Enable texel-grid snapping for cascade projection.
    pub enabled: bool,
    /// Round projection origin to multiples of this value (usually texel size).
    pub snap_granularity: f32,
    /// Lock cascade bounds to avoid per-frame size changes.
    pub lock_bounds: bool,
    /// Padding factor applied to cascade bounds to reduce re-computation.
    pub bounds_padding: f32,
}

impl Default for StabilizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            snap_granularity: 0.0, // auto-computed from texel size
            lock_bounds: false,
            bounds_padding: 1.05,
        }
    }
}

impl StabilizationConfig {
    /// Snap a world-space position to the texel grid.
    pub fn snap_position(&self, x: f32, y: f32, texel_size: f32) -> (f32, f32) {
        if !self.enabled || texel_size <= EPSILON {
            return (x, y);
        }
        let snap = if self.snap_granularity > EPSILON {
            self.snap_granularity
        } else {
            texel_size
        };
        let snapped_x = (x / snap).floor() * snap;
        let snapped_y = (y / snap).floor() * snap;
        (snapped_x, snapped_y)
    }

    /// Compute the stabilization offset for a given origin and texel size.
    pub fn compute_offset(&self, origin_x: f32, origin_y: f32, texel_size: f32) -> [f32; 2] {
        let (sx, sy) = self.snap_position(origin_x, origin_y, texel_size);
        [origin_x - sx, origin_y - sy]
    }
}

// ---------------------------------------------------------------------------
// Cascade configuration
// ---------------------------------------------------------------------------

/// Top-level configuration for the shadow cascade system.
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    /// Number of active cascades (1..=MAX_CASCADES).
    pub cascade_count: usize,
    /// Split scheme for distributing cascade boundaries.
    pub split_scheme: SplitScheme,
    /// Manual split distances (used only when split_scheme is Manual).
    /// Must have `cascade_count + 1` entries (near, split1, split2, ..., far).
    pub manual_splits: Vec<f32>,
    /// Shadow map resolution per cascade (square).
    pub resolution: u32,
    /// Overall shadow distance (far plane for the last cascade).
    pub shadow_distance: f32,
    /// Near plane for the first cascade (usually the camera near plane).
    pub near_plane: f32,
    /// Cascade blend configuration.
    pub blend: CascadeBlendConfig,
    /// Stabilization configuration.
    pub stabilization: StabilizationConfig,
    /// Per-cascade filter quality overrides. If shorter than cascade_count,
    /// remaining cascades use the last specified quality.
    pub quality_per_cascade: Vec<ShadowFilterQuality>,
    /// Per-cascade resolution overrides (0 means use `resolution`).
    pub resolution_per_cascade: Vec<u32>,
    /// Per-cascade depth bias overrides.
    pub depth_bias_per_cascade: Vec<f32>,
    /// Per-cascade normal bias overrides.
    pub normal_bias_per_cascade: Vec<f32>,
    /// Whether to enable debug visualization colors.
    pub debug_visualization: bool,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            cascade_count: DEFAULT_CASCADE_COUNT,
            split_scheme: SplitScheme::default(),
            manual_splits: Vec::new(),
            resolution: DEFAULT_CASCADE_RESOLUTION,
            shadow_distance: 200.0,
            near_plane: 0.1,
            blend: CascadeBlendConfig::default(),
            stabilization: StabilizationConfig::default(),
            quality_per_cascade: vec![
                ShadowFilterQuality::Pcf5x5,
                ShadowFilterQuality::Pcf3x3,
                ShadowFilterQuality::Pcf2x2,
                ShadowFilterQuality::Hard,
            ],
            resolution_per_cascade: Vec::new(),
            depth_bias_per_cascade: Vec::new(),
            normal_bias_per_cascade: Vec::new(),
            debug_visualization: false,
        }
    }
}

impl CascadeConfig {
    /// Get the filter quality for a cascade index.
    pub fn quality_for(&self, index: usize) -> ShadowFilterQuality {
        if let Some(&q) = self.quality_per_cascade.get(index) {
            q
        } else {
            self.quality_per_cascade.last().copied().unwrap_or_default()
        }
    }

    /// Get the resolution for a cascade index.
    pub fn resolution_for(&self, index: usize) -> u32 {
        if let Some(&r) = self.resolution_per_cascade.get(index) {
            if r > 0 { r } else { self.resolution }
        } else {
            self.resolution
        }
    }

    /// Get the depth bias for a cascade index.
    pub fn depth_bias_for(&self, index: usize) -> f32 {
        self.depth_bias_per_cascade.get(index).copied().unwrap_or(0.005)
    }

    /// Get the normal bias for a cascade index.
    pub fn normal_bias_for(&self, index: usize) -> f32 {
        self.normal_bias_per_cascade.get(index).copied().unwrap_or(0.02)
    }

    /// Validate the configuration and return errors if any.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.cascade_count == 0 || self.cascade_count > MAX_CASCADES {
            errors.push(format!(
                "cascade_count must be 1..{}, got {}",
                MAX_CASCADES, self.cascade_count
            ));
        }
        if self.shadow_distance <= self.near_plane {
            errors.push("shadow_distance must be greater than near_plane".into());
        }
        if self.resolution == 0 {
            errors.push("resolution must be > 0".into());
        }
        if matches!(self.split_scheme, SplitScheme::Manual) {
            if self.manual_splits.len() != self.cascade_count + 1 {
                errors.push(format!(
                    "Manual splits must have {} entries, got {}",
                    self.cascade_count + 1,
                    self.manual_splits.len()
                ));
            }
            for window in self.manual_splits.windows(2) {
                if window[1] <= window[0] {
                    errors.push(format!(
                        "Manual splits must be strictly increasing, got {} >= {}",
                        window[0], window[1]
                    ));
                }
            }
        }
        errors
    }
}

// ---------------------------------------------------------------------------
// Split distance computation
// ---------------------------------------------------------------------------

/// Compute split distances for the given configuration.
///
/// Returns `cascade_count + 1` distances: [near, split1, ..., far].
pub fn compute_split_distances(config: &CascadeConfig) -> Vec<f32> {
    let n = config.cascade_count;
    let near = config.near_plane.max(EPSILON);
    let far = config.shadow_distance;

    match config.split_scheme {
        SplitScheme::Manual => {
            if config.manual_splits.len() == n + 1 {
                config.manual_splits.clone()
            } else {
                // Fallback to uniform if manual splits are invalid
                compute_uniform_splits(near, far, n)
            }
        }
        SplitScheme::Uniform => compute_uniform_splits(near, far, n),
        SplitScheme::Logarithmic => compute_log_splits(near, far, n),
        SplitScheme::Pssm { lambda } => compute_pssm_splits(near, far, n, lambda),
    }
}

/// Compute uniform split distances.
fn compute_uniform_splits(near: f32, far: f32, count: usize) -> Vec<f32> {
    let mut splits = Vec::with_capacity(count + 1);
    let range = far - near;
    for i in 0..=count {
        splits.push(near + range * (i as f32 / count as f32));
    }
    splits
}

/// Compute logarithmic split distances.
fn compute_log_splits(near: f32, far: f32, count: usize) -> Vec<f32> {
    let mut splits = Vec::with_capacity(count + 1);
    let near = near.max(EPSILON);
    let ratio = far / near;
    for i in 0..=count {
        let t = i as f32 / count as f32;
        splits.push(near * ratio.powf(t));
    }
    splits
}

/// Compute PSSM (Practical Split Scheme Maps) split distances.
///
/// Blends between uniform and logarithmic using `lambda`:
/// - lambda = 0.0: fully uniform
/// - lambda = 1.0: fully logarithmic
pub fn compute_pssm_splits(near: f32, far: f32, count: usize, lambda: f32) -> Vec<f32> {
    let near = near.max(EPSILON);
    let lambda = lambda.clamp(0.0, 1.0);
    let mut splits = Vec::with_capacity(count + 1);
    let ratio = far / near;

    for i in 0..=count {
        let t = i as f32 / count as f32;
        let log_split = near * ratio.powf(t);
        let uniform_split = near + (far - near) * t;
        splits.push(lambda * log_split + (1.0 - lambda) * uniform_split);
    }
    splits
}

// ---------------------------------------------------------------------------
// Cascade statistics
// ---------------------------------------------------------------------------

/// Statistics about the cascade system for profiling and debugging.
#[derive(Debug, Clone, Default)]
pub struct CascadeStats {
    /// Number of active cascades.
    pub active_cascade_count: usize,
    /// Total shadow map memory usage (estimated, in bytes).
    pub total_memory_bytes: u64,
    /// Per-cascade texel density (world units per texel).
    pub texel_densities: Vec<f32>,
    /// Per-cascade coverage area (square world units).
    pub coverage_areas: Vec<f32>,
    /// Number of times cascades were recalculated this frame.
    pub recalculation_count: u32,
    /// Whether stabilization was applied this frame.
    pub stabilization_applied: bool,
}

impl CascadeStats {
    /// Returns the total number of shadow samples per fragment (sum of all cascades).
    pub fn total_samples(&self, config: &CascadeConfig) -> u32 {
        (0..config.cascade_count)
            .map(|i| config.quality_for(i).sample_count())
            .sum()
    }

    /// Returns the average texel density across all cascades.
    pub fn average_texel_density(&self) -> f32 {
        if self.texel_densities.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.texel_densities.iter().sum();
        sum / self.texel_densities.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Cascade manager
// ---------------------------------------------------------------------------

/// Manages shadow cascades for a directional light.
///
/// Maintains cascade split distances, viewports, and per-cascade state.
/// Updated once per frame before shadow rendering.
#[derive(Debug, Clone)]
pub struct CascadeManager {
    /// Configuration.
    config: CascadeConfig,
    /// Per-cascade computed data.
    cascades: Vec<CascadeData>,
    /// Computed split distances.
    split_distances: Vec<f32>,
    /// Current frame statistics.
    stats: CascadeStats,
    /// Frame counter for temporal effects.
    frame_count: u64,
    /// Cached atlas layout dimensions.
    atlas_width: u32,
    atlas_height: u32,
}

impl CascadeManager {
    /// Create a new cascade manager with the given configuration.
    pub fn new(config: CascadeConfig) -> Self {
        let n = config.cascade_count.min(MAX_CASCADES);
        let split_distances = compute_split_distances(&config);
        let mut cascades = Vec::with_capacity(n);

        for i in 0..n {
            let mut cascade = CascadeData::new(i);
            cascade.near_distance = split_distances[i];
            cascade.far_distance = split_distances[i + 1];
            cascade.filter_quality = config.quality_for(i);
            cascade.depth_bias = config.depth_bias_for(i);
            cascade.normal_bias = config.normal_bias_for(i);
            cascades.push(cascade);
        }

        let (atlas_w, atlas_h) = Self::compute_atlas_layout(n, &config);

        Self {
            config,
            cascades,
            split_distances,
            stats: CascadeStats::default(),
            frame_count: 0,
            atlas_width: atlas_w,
            atlas_height: atlas_h,
        }
    }

    /// Compute the atlas layout dimensions for the given cascade count.
    fn compute_atlas_layout(count: usize, config: &CascadeConfig) -> (u32, u32) {
        match count {
            0 => (0, 0),
            1 => (config.resolution, config.resolution),
            2 => (config.resolution * 2, config.resolution),
            3 | 4 => (config.resolution * 2, config.resolution * 2),
            _ => {
                let cols = ((count as f32).sqrt().ceil()) as u32;
                let rows = ((count as f32 / cols as f32).ceil()) as u32;
                (config.resolution * cols, config.resolution * rows)
            }
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &CascadeConfig {
        &self.config
    }

    /// Update the configuration and recompute cascades.
    pub fn set_config(&mut self, config: CascadeConfig) {
        self.config = config;
        self.rebuild();
    }

    /// Rebuild cascades from the current configuration.
    pub fn rebuild(&mut self) {
        let n = self.config.cascade_count.min(MAX_CASCADES);
        self.split_distances = compute_split_distances(&self.config);
        self.cascades.clear();

        for i in 0..n {
            let mut cascade = CascadeData::new(i);
            cascade.near_distance = self.split_distances[i];
            cascade.far_distance = self.split_distances[i + 1];
            cascade.filter_quality = self.config.quality_for(i);
            cascade.depth_bias = self.config.depth_bias_for(i);
            cascade.normal_bias = self.config.normal_bias_for(i);
            self.cascades.push(cascade);
        }

        let (w, h) = Self::compute_atlas_layout(n, &self.config);
        self.atlas_width = w;
        self.atlas_height = h;

        self.assign_viewports();
    }

    /// Assign viewports to cascades based on atlas layout.
    fn assign_viewports(&mut self) {
        let count = self.cascades.len();
        if count == 0 {
            return;
        }
        let cols = ((count as f32).sqrt().ceil()) as u32;

        for (i, cascade) in self.cascades.iter_mut().enumerate() {
            let res = self.config.resolution_for(i);
            let col = (i as u32) % cols;
            let row = (i as u32) / cols;
            cascade.viewport = CascadeViewport::new(col * res, row * res, res, res);
        }
    }

    /// Update cascades for the current frame.
    ///
    /// This recomputes split distances, viewports, and stabilization offsets.
    pub fn update(
        &mut self,
        camera_near: f32,
        camera_far: f32,
        _camera_view: &[f32; 16],
        light_direction: [f32; 3],
    ) {
        self.frame_count += 1;
        self.stats.recalculation_count = 0;

        // Clamp shadow distance to camera far plane.
        let effective_far = self.config.shadow_distance.min(camera_far);
        let effective_near = camera_near.max(EPSILON);

        // Recompute split distances if needed.
        let old_splits = self.split_distances.clone();
        let mut temp_config = self.config.clone();
        temp_config.near_plane = effective_near;
        temp_config.shadow_distance = effective_far;
        self.split_distances = compute_split_distances(&temp_config);

        // Check if splits changed.
        let splits_changed = old_splits.len() != self.split_distances.len()
            || old_splits
                .iter()
                .zip(self.split_distances.iter())
                .any(|(a, b)| (a - b).abs() > EPSILON);

        if splits_changed {
            self.stats.recalculation_count += 1;
        }

        // Update per-cascade data.
        let light_len = (light_direction[0] * light_direction[0]
            + light_direction[1] * light_direction[1]
            + light_direction[2] * light_direction[2])
            .sqrt()
            .max(EPSILON);
        let _light_dir = [
            light_direction[0] / light_len,
            light_direction[1] / light_len,
            light_direction[2] / light_len,
        ];

        self.stats.texel_densities.clear();
        self.stats.coverage_areas.clear();
        self.stats.active_cascade_count = 0;

        for i in 0..self.cascades.len() {
            let cascade = &mut self.cascades[i];
            cascade.near_distance = self.split_distances[i];
            cascade.far_distance = self.split_distances[i + 1];

            // Compute texel size based on cascade coverage.
            let cascade_range = cascade.far_distance - cascade.near_distance;
            let res = self.config.resolution_for(i) as f32;
            cascade.texel_size = if res > 0.0 { cascade_range / res } else { 1.0 };

            // Apply stabilization.
            if self.config.stabilization.enabled {
                let offset = self.config.stabilization.compute_offset(0.0, 0.0, cascade.texel_size);
                cascade.stabilization_offset = offset;
                self.stats.stabilization_applied = true;
            }

            if cascade.active {
                self.stats.active_cascade_count += 1;
            }
            self.stats.texel_densities.push(cascade.texel_size);
            self.stats.coverage_areas.push(cascade_range * cascade_range);
        }

        // Compute total memory.
        self.stats.total_memory_bytes = self.cascades.iter().enumerate().map(|(i, c)| {
            if c.active {
                let res = self.config.resolution_for(i) as u64;
                res * res * 4 // 32-bit depth
            } else {
                0
            }
        }).sum();
    }

    /// Select which cascade should be used for a given view-space depth.
    pub fn select_cascade(&self, depth: f32) -> Option<usize> {
        for (i, cascade) in self.cascades.iter().enumerate() {
            if cascade.active && depth >= cascade.near_distance && depth < cascade.far_distance {
                return Some(i);
            }
        }
        // Check the last cascade's far boundary inclusively.
        if let Some(last) = self.cascades.last() {
            if last.active && (depth - last.far_distance).abs() < EPSILON {
                return Some(self.cascades.len() - 1);
            }
        }
        None
    }

    /// Select cascade and compute blend factor for smooth transitions.
    pub fn select_cascade_blended(&self, depth: f32) -> CascadeSelection {
        let primary = self.select_cascade(depth);
        match primary {
            None => CascadeSelection {
                primary_cascade: None,
                secondary_cascade: None,
                blend_factor: 0.0,
            },
            Some(idx) => {
                let cascade = &self.cascades[idx];
                let blend = self
                    .config
                    .blend
                    .compute_blend(depth, cascade, cascade);

                let secondary = if blend > EPSILON && idx + 1 < self.cascades.len() {
                    if self.cascades[idx + 1].active {
                        Some(idx + 1)
                    } else {
                        None
                    }
                } else {
                    None
                };

                CascadeSelection {
                    primary_cascade: Some(idx),
                    secondary_cascade: secondary,
                    blend_factor: blend,
                }
            }
        }
    }

    /// Get the cascade data for a given index.
    pub fn cascade(&self, index: usize) -> Option<&CascadeData> {
        self.cascades.get(index)
    }

    /// Get mutable cascade data for a given index.
    pub fn cascade_mut(&mut self, index: usize) -> Option<&mut CascadeData> {
        self.cascades.get_mut(index)
    }

    /// Get all cascades.
    pub fn cascades(&self) -> &[CascadeData] {
        &self.cascades
    }

    /// Get the split distances.
    pub fn split_distances(&self) -> &[f32] {
        &self.split_distances
    }

    /// Get the shadow atlas dimensions.
    pub fn atlas_size(&self) -> (u32, u32) {
        (self.atlas_width, self.atlas_height)
    }

    /// Get the current frame statistics.
    pub fn stats(&self) -> &CascadeStats {
        &self.stats
    }

    /// Get the debug color for a cascade index.
    pub fn debug_color(&self, index: usize) -> CascadeColor {
        CASCADE_COLORS[index % MAX_CASCADES]
    }

    /// Set the number of active cascades.
    pub fn set_cascade_count(&mut self, count: usize) {
        let count = count.clamp(1, MAX_CASCADES);
        self.config.cascade_count = count;
        self.rebuild();
    }

    /// Set the shadow distance.
    pub fn set_shadow_distance(&mut self, distance: f32) {
        self.config.shadow_distance = distance.max(EPSILON);
        self.rebuild();
    }

    /// Enable or disable a specific cascade.
    pub fn set_cascade_active(&mut self, index: usize, active: bool) {
        if let Some(cascade) = self.cascades.get_mut(index) {
            cascade.active = active;
        }
    }

    /// Get a GPU-friendly struct with cascade data for shader upload.
    pub fn gpu_data(&self) -> CascadeGpuData {
        let mut data = CascadeGpuData::default();
        data.cascade_count = self.cascades.len() as u32;
        data.blend_width = self.config.blend.blend_width;
        data.shadow_distance = self.config.shadow_distance;

        for (i, cascade) in self.cascades.iter().enumerate().take(MAX_CASCADES) {
            data.split_distances[i] = cascade.far_distance;
            data.view_projections[i] = cascade.view_projection;
            data.texel_sizes[i] = cascade.texel_size;
            data.depth_biases[i] = cascade.depth_bias;
            data.normal_biases[i] = cascade.normal_bias;
        }
        data
    }
}

// ---------------------------------------------------------------------------
// Cascade selection result
// ---------------------------------------------------------------------------

/// Result of cascade selection for a given depth.
#[derive(Debug, Clone, Copy)]
pub struct CascadeSelection {
    /// Primary cascade index (None if out of shadow range).
    pub primary_cascade: Option<usize>,
    /// Secondary cascade index for blending (None if no blend needed).
    pub secondary_cascade: Option<usize>,
    /// Blend factor between primary and secondary (0.0 = fully primary).
    pub blend_factor: f32,
}

impl CascadeSelection {
    /// Returns true if the depth is within shadow range.
    pub fn in_shadow_range(&self) -> bool {
        self.primary_cascade.is_some()
    }

    /// Returns true if blending is needed.
    pub fn needs_blend(&self) -> bool {
        self.secondary_cascade.is_some() && self.blend_factor > EPSILON
    }
}

// ---------------------------------------------------------------------------
// GPU data structure
// ---------------------------------------------------------------------------

/// GPU-friendly cascade data for shader constant buffer upload.
#[derive(Debug, Clone)]
pub struct CascadeGpuData {
    /// Number of active cascades.
    pub cascade_count: u32,
    /// Blend width for cascade transitions.
    pub blend_width: f32,
    /// Maximum shadow distance.
    pub shadow_distance: f32,
    /// Padding for alignment.
    pub _padding: f32,
    /// Per-cascade far split distances.
    pub split_distances: [f32; MAX_CASCADES],
    /// Per-cascade view-projection matrices.
    pub view_projections: [[f32; 16]; MAX_CASCADES],
    /// Per-cascade texel sizes.
    pub texel_sizes: [f32; MAX_CASCADES],
    /// Per-cascade depth biases.
    pub depth_biases: [f32; MAX_CASCADES],
    /// Per-cascade normal biases.
    pub normal_biases: [f32; MAX_CASCADES],
}

impl Default for CascadeGpuData {
    fn default() -> Self {
        Self {
            cascade_count: 0,
            blend_width: DEFAULT_BLEND_WIDTH,
            shadow_distance: 200.0,
            _padding: 0.0,
            split_distances: [0.0; MAX_CASCADES],
            view_projections: [[0.0; 16]; MAX_CASCADES],
            texel_sizes: [0.0; MAX_CASCADES],
            depth_biases: [0.0; MAX_CASCADES],
            normal_biases: [0.0; MAX_CASCADES],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_splits() {
        let splits = compute_uniform_splits(1.0, 100.0, 4);
        assert_eq!(splits.len(), 5);
        assert!((splits[0] - 1.0).abs() < EPSILON);
        assert!((splits[4] - 100.0).abs() < EPSILON);
        // Check uniform spacing
        let step = (100.0 - 1.0) / 4.0;
        for i in 1..5 {
            assert!((splits[i] - (1.0 + step * i as f32)).abs() < 0.01);
        }
    }

    #[test]
    fn test_log_splits() {
        let splits = compute_log_splits(1.0, 1000.0, 3);
        assert_eq!(splits.len(), 4);
        assert!((splits[0] - 1.0).abs() < EPSILON);
        assert!((splits[3] - 1000.0).abs() < 0.1);
        // Log splits should grow exponentially
        let ratio1 = splits[1] / splits[0];
        let ratio2 = splits[2] / splits[1];
        assert!((ratio1 - ratio2).abs() < 0.1);
    }

    #[test]
    fn test_pssm_splits_lambda_0_is_uniform() {
        let pssm = compute_pssm_splits(1.0, 100.0, 4, 0.0);
        let uniform = compute_uniform_splits(1.0, 100.0, 4);
        for (a, b) in pssm.iter().zip(uniform.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_pssm_splits_lambda_1_is_log() {
        let pssm = compute_pssm_splits(1.0, 100.0, 4, 1.0);
        let log = compute_log_splits(1.0, 100.0, 4);
        for (a, b) in pssm.iter().zip(log.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_cascade_selection() {
        let config = CascadeConfig {
            cascade_count: 4,
            shadow_distance: 100.0,
            near_plane: 0.1,
            split_scheme: SplitScheme::Uniform,
            ..Default::default()
        };
        let mgr = CascadeManager::new(config);

        assert_eq!(mgr.select_cascade(5.0), Some(0));
        assert_eq!(mgr.select_cascade(30.0), Some(1));
        assert_eq!(mgr.select_cascade(55.0), Some(2));
        assert_eq!(mgr.select_cascade(80.0), Some(3));
        assert_eq!(mgr.select_cascade(150.0), None);
    }

    #[test]
    fn test_blend_factor() {
        let cascade = CascadeData::new(0);
        let mut c = cascade;
        c.near_distance = 0.0;
        c.far_distance = 25.0;

        assert!((c.blend_factor(10.0, 5.0) - 0.0).abs() < EPSILON);
        assert!((c.blend_factor(22.5, 5.0) - 0.5).abs() < EPSILON);
        assert!((c.blend_factor(25.0, 5.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_stabilization_snap() {
        let stab = StabilizationConfig {
            enabled: true,
            snap_granularity: 0.0,
            lock_bounds: false,
            bounds_padding: 1.0,
        };
        let (sx, sy) = stab.snap_position(3.7, 5.2, 1.0);
        assert!((sx - 3.0).abs() < EPSILON);
        assert!((sy - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_config_validation() {
        let mut config = CascadeConfig::default();
        assert!(config.validate().is_empty());

        config.cascade_count = 0;
        assert!(!config.validate().is_empty());

        config.cascade_count = 4;
        config.shadow_distance = 0.01;
        config.near_plane = 1.0;
        assert!(!config.validate().is_empty());
    }
}
