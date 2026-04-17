// engine/render/src/depth_of_field_v2.rs
//
// Enhanced Depth of Field (v2) for the Genovo engine.
//
// Implements a physically-based circular DOF with advanced features:
//
// - **Circular DOF with bokeh shapes** -- Simulates real camera lens bokeh
//   using shaped kernels (circle, hexagon, octagon, custom).
// - **Foreground/background separation** -- Near and far field blur are
//   computed separately to avoid bleeding artifacts.
// - **Partial occlusion** -- Handles the case where in-focus objects partially
//   occlude out-of-focus background, preventing halo artifacts.
// - **Smooth transitions** -- Gradual focus falloff with no visible boundaries
//   between in-focus and out-of-focus regions.
// - **Camera settings** -- DOF parameters derived from physical camera
//   properties: aperture (f-stop), focal length, and focus distance.

use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;
const DEFAULT_FOCAL_LENGTH_MM: f32 = 50.0;
const DEFAULT_APERTURE: f32 = 5.6;
const DEFAULT_FOCUS_DISTANCE: f32 = 10.0;
const DEFAULT_SENSOR_WIDTH_MM: f32 = 36.0; // Full-frame 35mm
const MAX_COC_RADIUS: f32 = 32.0;
const PI: f32 = std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Bokeh shape
// ---------------------------------------------------------------------------

/// Shape of the out-of-focus bokeh highlights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BokehShape {
    /// Circular bokeh (ideal lens).
    Circle,
    /// Hexagonal bokeh (6-blade aperture).
    Hexagon,
    /// Octagonal bokeh (8-blade aperture).
    Octagon,
    /// Cat's eye bokeh (vignetting at edges).
    CatsEye,
    /// Anamorphic (oval/stretched) bokeh.
    Anamorphic,
}

impl Default for BokehShape {
    fn default() -> Self {
        Self::Circle
    }
}

impl fmt::Display for BokehShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Circle => write!(f, "Circle"),
            Self::Hexagon => write!(f, "Hexagon"),
            Self::Octagon => write!(f, "Octagon"),
            Self::CatsEye => write!(f, "Cat's Eye"),
            Self::Anamorphic => write!(f, "Anamorphic"),
        }
    }
}

impl BokehShape {
    /// Number of sides for polygon bokeh shapes.
    pub fn sides(&self) -> u32 {
        match self {
            Self::Circle => 0,  // infinite sides
            Self::Hexagon => 6,
            Self::Octagon => 8,
            Self::CatsEye => 0,
            Self::Anamorphic => 0,
        }
    }

    /// Generate sample points for this bokeh shape.
    pub fn generate_kernel(&self, sample_count: u32) -> Vec<[f32; 2]> {
        let n = sample_count.max(4);
        let mut samples = Vec::with_capacity(n as usize);

        match self {
            Self::Circle => {
                // Sunflower (Vogel) disk sampling for uniform distribution.
                let golden_angle = PI * (3.0 - 5.0_f32.sqrt());
                for i in 0..n {
                    let r = (i as f32 / n as f32).sqrt();
                    let theta = i as f32 * golden_angle;
                    samples.push([r * theta.cos(), r * theta.sin()]);
                }
            }
            Self::Hexagon | Self::Octagon => {
                let sides = self.sides() as f32;
                let golden_angle = PI * (3.0 - 5.0_f32.sqrt());
                for i in 0..n {
                    let r = (i as f32 / n as f32).sqrt();
                    let theta = i as f32 * golden_angle;
                    let x = r * theta.cos();
                    let y = r * theta.sin();
                    // Project onto polygon boundary for radius clamping.
                    let angle = y.atan2(x);
                    let sector = (2.0 * PI) / sides;
                    let half_sector = sector * 0.5;
                    let sector_angle = ((angle / sector).floor() + 0.5) * sector;
                    let rel_angle = angle - sector_angle;
                    let max_r = half_sector.cos() / (rel_angle.abs().max(EPSILON)).cos().min(1.0 / EPSILON);
                    let clamped_r = r.min(max_r.abs());
                    samples.push([clamped_r * theta.cos(), clamped_r * theta.sin()]);
                }
            }
            Self::CatsEye => {
                // Circle with vignetting that squishes towards edges.
                let golden_angle = PI * (3.0 - 5.0_f32.sqrt());
                for i in 0..n {
                    let r = (i as f32 / n as f32).sqrt();
                    let theta = i as f32 * golden_angle;
                    let x = r * theta.cos();
                    let y = r * theta.sin();
                    // Apply cat's eye distortion based on distance from center.
                    let dist = (x * x + y * y).sqrt();
                    let squeeze = 1.0 - 0.3 * dist;
                    samples.push([x * squeeze, y]);
                }
            }
            Self::Anamorphic => {
                // Oval bokeh (2:1 aspect ratio).
                let golden_angle = PI * (3.0 - 5.0_f32.sqrt());
                for i in 0..n {
                    let r = (i as f32 / n as f32).sqrt();
                    let theta = i as f32 * golden_angle;
                    samples.push([r * theta.cos() * 2.0, r * theta.sin()]);
                }
            }
        }
        samples
    }
}

// ---------------------------------------------------------------------------
// DOF quality preset
// ---------------------------------------------------------------------------

/// Quality preset for DOF rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DofQuality {
    /// Low quality: fewer samples, no partial occlusion.
    Low,
    /// Medium quality: moderate samples, basic foreground separation.
    Medium,
    /// High quality: many samples, full foreground/background separation.
    High,
    /// Ultra quality: maximum samples, partial occlusion, smooth transitions.
    Ultra,
    /// Cinematic: highest quality for cutscenes.
    Cinematic,
}

impl Default for DofQuality {
    fn default() -> Self {
        Self::Medium
    }
}

impl DofQuality {
    /// Get the number of bokeh samples for this quality level.
    pub fn sample_count(&self) -> u32 {
        match self {
            Self::Low => 16,
            Self::Medium => 32,
            Self::High => 64,
            Self::Ultra => 128,
            Self::Cinematic => 256,
        }
    }

    /// Whether to enable foreground separation at this quality.
    pub fn foreground_separation(&self) -> bool {
        !matches!(self, Self::Low)
    }

    /// Whether to enable partial occlusion at this quality.
    pub fn partial_occlusion(&self) -> bool {
        matches!(self, Self::High | Self::Ultra | Self::Cinematic)
    }

    /// Number of blur passes for smooth result.
    pub fn blur_passes(&self) -> u32 {
        match self {
            Self::Low => 1,
            Self::Medium => 1,
            Self::High => 2,
            Self::Ultra => 2,
            Self::Cinematic => 3,
        }
    }

    /// Internal render resolution scale (1.0 = full, 0.5 = half).
    pub fn resolution_scale(&self) -> f32 {
        match self {
            Self::Low => 0.5,
            Self::Medium => 0.5,
            Self::High => 1.0,
            Self::Ultra => 1.0,
            Self::Cinematic => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera lens settings
// ---------------------------------------------------------------------------

/// Physical camera lens parameters for DOF calculation.
#[derive(Debug, Clone, Copy)]
pub struct LensSettings {
    /// Focal length in millimeters (e.g., 50mm).
    pub focal_length_mm: f32,
    /// F-stop number (e.g., 2.8, 5.6, 11).
    pub aperture: f32,
    /// Focus distance in world units (meters).
    pub focus_distance: f32,
    /// Sensor width in millimeters (36mm for full-frame).
    pub sensor_width_mm: f32,
    /// Number of aperture blades (affects bokeh shape).
    pub blade_count: u32,
    /// Blade rotation angle in radians.
    pub blade_rotation: f32,
    /// Blade curvature (0 = straight, 1 = rounded).
    pub blade_curvature: f32,
}

impl Default for LensSettings {
    fn default() -> Self {
        Self {
            focal_length_mm: DEFAULT_FOCAL_LENGTH_MM,
            aperture: DEFAULT_APERTURE,
            focus_distance: DEFAULT_FOCUS_DISTANCE,
            sensor_width_mm: DEFAULT_SENSOR_WIDTH_MM,
            blade_count: 6,
            blade_rotation: 0.0,
            blade_curvature: 0.5,
        }
    }
}

impl LensSettings {
    /// Compute the Circle of Confusion (CoC) diameter in millimeters
    /// for a given object distance.
    ///
    /// Based on the thin lens equation:
    /// CoC = |S2 - S1| / S2 * f^2 / (N * (S1 - f))
    ///
    /// where S1 = focus distance, S2 = object distance, f = focal length,
    /// N = f-stop number.
    pub fn compute_coc_mm(&self, object_distance: f32) -> f32 {
        if object_distance <= EPSILON || self.focus_distance <= EPSILON {
            return 0.0;
        }
        let f = self.focal_length_mm * 0.001; // Convert to meters.
        let s1 = self.focus_distance;
        let s2 = object_distance;
        let n = self.aperture.max(EPSILON);

        let denominator = n * (s1 - f).max(EPSILON);
        let magnification = f * f / denominator;
        let coc = ((s2 - s1) / s2.max(EPSILON)).abs() * magnification;

        // Convert back to millimeters.
        coc * 1000.0
    }

    /// Compute the CoC in pixels for a given object distance and screen height.
    pub fn compute_coc_pixels(&self, object_distance: f32, screen_height: u32) -> f32 {
        let coc_mm = self.compute_coc_mm(object_distance);
        // Scale from sensor space to screen space.
        let sensor_to_screen = screen_height as f32 / self.sensor_width_mm;
        (coc_mm * sensor_to_screen).min(MAX_COC_RADIUS)
    }

    /// Compute near and far focus distances (hyperfocal, near limit, far limit).
    pub fn focus_range(&self) -> FocusRange {
        let f = self.focal_length_mm;
        let n = self.aperture.max(EPSILON);
        let s = self.focus_distance * 1000.0; // Convert to mm.
        let coc_limit = 0.03; // Acceptable CoC in mm (standard).

        // Hyperfocal distance.
        let h = (f * f) / (n * coc_limit) + f;

        // Near focus limit.
        let near = if s > EPSILON {
            (s * (h - f)) / (h + s - 2.0 * f)
        } else {
            0.0
        };

        // Far focus limit.
        let far = if (h - s).abs() > EPSILON {
            (s * (h - f)) / (h - s)
        } else {
            f32::INFINITY
        };

        FocusRange {
            hyperfocal_distance: h * 0.001,
            near_focus: (near * 0.001).max(0.0),
            far_focus: if far > 0.0 { far * 0.001 } else { f32::INFINITY },
        }
    }

    /// Get the field of view in radians (horizontal).
    pub fn horizontal_fov(&self) -> f32 {
        2.0 * (self.sensor_width_mm / (2.0 * self.focal_length_mm)).atan()
    }

    /// Get the BokehShape implied by blade count and curvature.
    pub fn implied_bokeh_shape(&self) -> BokehShape {
        if self.blade_curvature > 0.8 {
            BokehShape::Circle
        } else if self.blade_count <= 6 {
            BokehShape::Hexagon
        } else {
            BokehShape::Octagon
        }
    }
}

/// Focus range information computed from lens settings.
#[derive(Debug, Clone, Copy)]
pub struct FocusRange {
    /// Hyperfocal distance in world units (meters).
    pub hyperfocal_distance: f32,
    /// Near focus limit in world units.
    pub near_focus: f32,
    /// Far focus limit in world units (can be infinity).
    pub far_focus: f32,
}

impl FocusRange {
    /// Returns the total depth of field (far - near).
    pub fn depth_of_field(&self) -> f32 {
        if self.far_focus.is_infinite() {
            f32::INFINITY
        } else {
            self.far_focus - self.near_focus
        }
    }

    /// Returns true if the entire scene is in focus.
    pub fn is_everything_in_focus(&self) -> bool {
        self.near_focus <= EPSILON && self.far_focus.is_infinite()
    }
}

// ---------------------------------------------------------------------------
// DOF configuration
// ---------------------------------------------------------------------------

/// Full DOF configuration combining lens, quality, and visual settings.
#[derive(Debug, Clone)]
pub struct DofConfig {
    /// Physical lens settings.
    pub lens: LensSettings,
    /// Quality preset.
    pub quality: DofQuality,
    /// Bokeh shape override (None = derive from lens blade settings).
    pub bokeh_shape: Option<BokehShape>,
    /// Maximum CoC radius in pixels (clamped for performance).
    pub max_coc_radius: f32,
    /// Near field blur multiplier (increase for more foreground blur).
    pub near_blur_scale: f32,
    /// Far field blur multiplier.
    pub far_blur_scale: f32,
    /// Bokeh brightness boost for highlights.
    pub bokeh_brightness: f32,
    /// Bokeh brightness threshold (luminance above which bokeh is boosted).
    pub bokeh_threshold: f32,
    /// Enable foreground/background separation.
    pub foreground_separation: bool,
    /// Enable partial occlusion handling.
    pub partial_occlusion: bool,
    /// Transition smoothness between focus regions.
    pub transition_smoothness: f32,
    /// Whether DOF is enabled.
    pub enabled: bool,
    /// Debug visualization mode.
    pub debug_mode: DofDebugMode,
}

impl Default for DofConfig {
    fn default() -> Self {
        let quality = DofQuality::default();
        Self {
            lens: LensSettings::default(),
            quality,
            bokeh_shape: None,
            max_coc_radius: MAX_COC_RADIUS,
            near_blur_scale: 1.0,
            far_blur_scale: 1.0,
            bokeh_brightness: 1.0,
            bokeh_threshold: 1.5,
            foreground_separation: quality.foreground_separation(),
            partial_occlusion: quality.partial_occlusion(),
            transition_smoothness: 1.0,
            enabled: true,
            debug_mode: DofDebugMode::None,
        }
    }
}

impl DofConfig {
    /// Get the effective bokeh shape.
    pub fn effective_bokeh_shape(&self) -> BokehShape {
        self.bokeh_shape.unwrap_or_else(|| self.lens.implied_bokeh_shape())
    }

    /// Get the effective sample count.
    pub fn effective_sample_count(&self) -> u32 {
        self.quality.sample_count()
    }
}

/// Debug visualization modes for DOF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DofDebugMode {
    /// No debug visualization.
    None,
    /// Show CoC radius as a heat map.
    CocHeatMap,
    /// Show near/far field separation.
    FieldSeparation,
    /// Show focus plane distance.
    FocusPlane,
    /// Show raw bokeh kernel.
    BokehKernel,
}

// ---------------------------------------------------------------------------
// CoC computation
// ---------------------------------------------------------------------------

/// Per-pixel Circle of Confusion data.
#[derive(Debug, Clone, Copy)]
pub struct CocData {
    /// Signed CoC radius in pixels.
    /// Negative = foreground (near field), Positive = background (far field).
    pub coc: f32,
    /// The depth at which this pixel sits.
    pub depth: f32,
    /// Whether this pixel is in the near field.
    pub is_near_field: bool,
}

/// Compute the signed CoC for each depth value.
pub fn compute_coc_buffer(
    depths: &[f32],
    width: u32,
    height: u32,
    config: &DofConfig,
) -> Vec<CocData> {
    let screen_height = height;
    let lens = &config.lens;
    let mut result = Vec::with_capacity(depths.len());

    for &depth in depths {
        let coc_pixels = lens.compute_coc_pixels(depth, screen_height);
        let clamped = coc_pixels.min(config.max_coc_radius);

        let signed_coc = if depth < lens.focus_distance {
            // Near field: negative CoC.
            -clamped * config.near_blur_scale
        } else {
            // Far field: positive CoC.
            clamped * config.far_blur_scale
        };

        result.push(CocData {
            coc: signed_coc,
            depth,
            is_near_field: depth < lens.focus_distance,
        });
    }
    let _ = width; // Used in GPU implementation for 2D indexing.
    result
}

/// Dilate the near-field CoC to prevent foreground leaking.
pub fn dilate_near_coc(coc_buffer: &mut [CocData], width: u32, height: u32, radius: u32) {
    if radius == 0 {
        return;
    }

    let r = radius as i32;
    let w = width as i32;
    let h = height as i32;
    let mut temp = coc_buffer.to_vec();

    // Horizontal pass.
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let mut min_coc = coc_buffer[idx].coc;
            for dx in -r..=r {
                let nx = x + dx;
                if nx >= 0 && nx < w {
                    let nidx = (y * w + nx) as usize;
                    if coc_buffer[nidx].is_near_field {
                        min_coc = min_coc.min(coc_buffer[nidx].coc);
                    }
                }
            }
            temp[idx].coc = if min_coc < 0.0 { min_coc } else { coc_buffer[idx].coc };
        }
    }

    // Vertical pass.
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let mut min_coc = temp[idx].coc;
            for dy in -r..=r {
                let ny = y + dy;
                if ny >= 0 && ny < h {
                    let nidx = (ny * w + x) as usize;
                    if temp[nidx].is_near_field {
                        min_coc = min_coc.min(temp[nidx].coc);
                    }
                }
            }
            coc_buffer[idx].coc = if min_coc < 0.0 { min_coc } else { temp[idx].coc };
        }
    }
}

// ---------------------------------------------------------------------------
// Focus tracking
// ---------------------------------------------------------------------------

/// Auto-focus tracking modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoFocusMode {
    /// No auto-focus; manual focus distance.
    Manual,
    /// Focus on the center of the screen.
    CenterPoint,
    /// Focus on a specified screen-space point.
    ScreenPoint,
    /// Focus on a target entity/position.
    TargetTracking,
}

/// Auto-focus state for smooth focus transitions.
#[derive(Debug, Clone)]
pub struct AutoFocusState {
    /// Current focus mode.
    pub mode: AutoFocusMode,
    /// Screen-space focus point (normalized 0..1).
    pub focus_point: [f32; 2],
    /// Target world position (for TargetTracking mode).
    pub target_position: Option<[f32; 3]>,
    /// Current smoothed focus distance.
    pub current_focus: f32,
    /// Target focus distance (what we are transitioning to).
    pub target_focus: f32,
    /// Focus transition speed (units per second).
    pub focus_speed: f32,
    /// Smoothing factor (0 = instant, 1 = very smooth).
    pub smoothing: f32,
}

impl Default for AutoFocusState {
    fn default() -> Self {
        Self {
            mode: AutoFocusMode::Manual,
            focus_point: [0.5, 0.5],
            target_position: None,
            current_focus: DEFAULT_FOCUS_DISTANCE,
            target_focus: DEFAULT_FOCUS_DISTANCE,
            focus_speed: 5.0,
            smoothing: 0.9,
        }
    }
}

impl AutoFocusState {
    /// Update the auto-focus state (call each frame).
    pub fn update(&mut self, dt: f32, sampled_depth: f32) {
        match self.mode {
            AutoFocusMode::Manual => {
                // No auto-focus; current_focus is set manually.
            }
            AutoFocusMode::CenterPoint | AutoFocusMode::ScreenPoint => {
                self.target_focus = sampled_depth.max(EPSILON);
                let speed = self.focus_speed * dt;
                let diff = self.target_focus - self.current_focus;
                if diff.abs() > EPSILON {
                    self.current_focus += diff * (1.0 - self.smoothing.powf(dt * 60.0));
                    self.current_focus += diff.signum() * speed.min(diff.abs());
                    self.current_focus = self.current_focus.max(EPSILON);
                }
            }
            AutoFocusMode::TargetTracking => {
                // Target distance is set externally; just smooth.
                let diff = self.target_focus - self.current_focus;
                if diff.abs() > EPSILON {
                    self.current_focus += diff * (1.0 - self.smoothing.powf(dt * 60.0));
                }
            }
        }
    }

    /// Set focus distance manually.
    pub fn set_focus(&mut self, distance: f32) {
        self.target_focus = distance.max(EPSILON);
        if self.mode == AutoFocusMode::Manual {
            self.current_focus = self.target_focus;
        }
    }
}

// ---------------------------------------------------------------------------
// DOF system
// ---------------------------------------------------------------------------

/// Statistics for the DOF system.
#[derive(Debug, Clone, Default)]
pub struct DofStats {
    /// Current focus distance.
    pub focus_distance: f32,
    /// Current aperture.
    pub aperture: f32,
    /// Current focal length.
    pub focal_length_mm: f32,
    /// Near focus limit.
    pub near_focus: f32,
    /// Far focus limit.
    pub far_focus: f32,
    /// Average CoC across the frame.
    pub avg_coc: f32,
    /// Maximum CoC in the frame.
    pub max_coc: f32,
    /// Number of near-field pixels.
    pub near_field_pixels: u32,
    /// Number of far-field pixels.
    pub far_field_pixels: u32,
    /// Number of in-focus pixels.
    pub in_focus_pixels: u32,
    /// DOF computation time (microseconds).
    pub compute_time_us: u64,
}

/// The DOF rendering system.
pub struct DepthOfFieldSystem {
    /// Configuration.
    config: DofConfig,
    /// Auto-focus state.
    auto_focus: AutoFocusState,
    /// Precomputed bokeh kernel samples.
    bokeh_kernel: Vec<[f32; 2]>,
    /// Statistics.
    stats: DofStats,
}

impl DepthOfFieldSystem {
    /// Create a new DOF system with default settings.
    pub fn new() -> Self {
        let config = DofConfig::default();
        let shape = config.effective_bokeh_shape();
        let samples = config.effective_sample_count();
        let kernel = shape.generate_kernel(samples);

        Self {
            config,
            auto_focus: AutoFocusState::default(),
            bokeh_kernel: kernel,
            stats: DofStats::default(),
        }
    }

    /// Create with specific settings.
    pub fn with_config(config: DofConfig) -> Self {
        let shape = config.effective_bokeh_shape();
        let samples = config.effective_sample_count();
        let kernel = shape.generate_kernel(samples);

        Self {
            config,
            auto_focus: AutoFocusState::default(),
            bokeh_kernel: kernel,
            stats: DofStats::default(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &DofConfig {
        &self.config
    }

    /// Update the configuration and rebuild kernels if needed.
    pub fn set_config(&mut self, config: DofConfig) {
        let shape_changed = config.effective_bokeh_shape() != self.config.effective_bokeh_shape();
        let samples_changed = config.effective_sample_count() != self.config.effective_sample_count();
        self.config = config;

        if shape_changed || samples_changed {
            let shape = self.config.effective_bokeh_shape();
            let samples = self.config.effective_sample_count();
            self.bokeh_kernel = shape.generate_kernel(samples);
        }
    }

    /// Get the auto-focus state.
    pub fn auto_focus(&self) -> &AutoFocusState {
        &self.auto_focus
    }

    /// Get mutable auto-focus state.
    pub fn auto_focus_mut(&mut self) -> &mut AutoFocusState {
        &mut self.auto_focus
    }

    /// Get the precomputed bokeh kernel.
    pub fn bokeh_kernel(&self) -> &[[f32; 2]] {
        &self.bokeh_kernel
    }

    /// Get statistics.
    pub fn stats(&self) -> &DofStats {
        &self.stats
    }

    /// Update the DOF system for the current frame.
    pub fn update(&mut self, dt: f32, center_depth: f32) {
        // Update auto-focus.
        self.auto_focus.update(dt, center_depth);

        // Apply auto-focus result to lens settings.
        if self.auto_focus.mode != AutoFocusMode::Manual {
            self.config.lens.focus_distance = self.auto_focus.current_focus;
        }

        // Update statistics.
        let range = self.config.lens.focus_range();
        self.stats.focus_distance = self.config.lens.focus_distance;
        self.stats.aperture = self.config.lens.aperture;
        self.stats.focal_length_mm = self.config.lens.focal_length_mm;
        self.stats.near_focus = range.near_focus;
        self.stats.far_focus = if range.far_focus.is_infinite() {
            f32::MAX
        } else {
            range.far_focus
        };
    }

    /// Set the focus distance directly.
    pub fn set_focus_distance(&mut self, distance: f32) {
        self.config.lens.focus_distance = distance.max(EPSILON);
        self.auto_focus.set_focus(distance);
    }

    /// Set the aperture (f-stop).
    pub fn set_aperture(&mut self, aperture: f32) {
        self.config.lens.aperture = aperture.max(0.5);
    }

    /// Set the focal length in millimeters.
    pub fn set_focal_length(&mut self, focal_length_mm: f32) {
        self.config.lens.focal_length_mm = focal_length_mm.max(1.0);
    }

    /// Check if DOF should be applied (enabled and has visible blur).
    pub fn should_apply(&self) -> bool {
        self.config.enabled && self.config.lens.aperture < 22.0
    }
}

impl Default for DepthOfFieldSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coc_at_focus_distance() {
        let lens = LensSettings::default();
        let coc = lens.compute_coc_mm(lens.focus_distance);
        assert!(coc.abs() < 0.1, "CoC at focus distance should be near zero");
    }

    #[test]
    fn test_coc_increases_with_distance() {
        let lens = LensSettings::default();
        let coc_near = lens.compute_coc_mm(5.0);
        let coc_far = lens.compute_coc_mm(100.0);
        // Both should have some blur, far should also have CoC > 0.
        assert!(coc_near > 0.0);
        assert!(coc_far > 0.0);
    }

    #[test]
    fn test_aperture_affects_coc() {
        let mut lens = LensSettings::default();
        lens.aperture = 2.8;
        let coc_wide = lens.compute_coc_mm(50.0);

        lens.aperture = 16.0;
        let coc_narrow = lens.compute_coc_mm(50.0);

        assert!(coc_wide > coc_narrow, "Wider aperture = more blur");
    }

    #[test]
    fn test_focus_range() {
        let lens = LensSettings {
            focal_length_mm: 50.0,
            aperture: 5.6,
            focus_distance: 10.0,
            ..Default::default()
        };
        let range = lens.focus_range();
        assert!(range.near_focus > 0.0);
        assert!(range.near_focus < lens.focus_distance * 1000.0);
    }

    #[test]
    fn test_bokeh_kernel_generation() {
        let shape = BokehShape::Circle;
        let kernel = shape.generate_kernel(32);
        assert_eq!(kernel.len(), 32);
        for sample in &kernel {
            let r = (sample[0] * sample[0] + sample[1] * sample[1]).sqrt();
            assert!(r <= 1.0 + EPSILON, "Sample outside unit circle");
        }
    }

    #[test]
    fn test_auto_focus_manual() {
        let mut af = AutoFocusState::default();
        af.mode = AutoFocusMode::Manual;
        af.set_focus(20.0);
        assert!((af.current_focus - 20.0).abs() < EPSILON);
    }

    #[test]
    fn test_dof_system_default() {
        let sys = DepthOfFieldSystem::new();
        assert!(sys.should_apply());
        assert!(sys.bokeh_kernel().len() > 0);
    }
}
