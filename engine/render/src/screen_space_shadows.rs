// engine/render/src/screen_space_shadows.rs
//
// Screen-space / contact shadow rendering for the Genovo engine.
//
// Implements screen-space ray marching from the light direction to produce
// fine-detail contact shadows that complement shadow maps:
//
// - **Ray-march from light direction** -- For each pixel, march along the
//   light direction in screen space, checking depth buffer intersections.
// - **Thickness estimation** -- Infers shadow receiver thickness to avoid
//   self-shadowing on thin surfaces.
// - **Soft contact shadows** -- Variable penumbra based on distance from
//   occluder, producing realistic soft shadow edges.
// - **Temporal filtering** -- Accumulates shadow results across frames to
//   reduce noise and improve quality at lower sample counts.

use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;
const DEFAULT_MAX_STEPS: u32 = 16;
const DEFAULT_RAY_LENGTH: f32 = 0.1;
const DEFAULT_THICKNESS: f32 = 0.05;
const DEFAULT_SHADOW_INTENSITY: f32 = 1.0;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Quality preset for screen-space shadows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SSShadowQuality {
    /// Low quality: few steps, no temporal filtering.
    Low,
    /// Medium quality: moderate steps, basic temporal.
    Medium,
    /// High quality: many steps, full temporal filtering.
    High,
    /// Ultra quality: maximum quality for cinematic use.
    Ultra,
}

impl Default for SSShadowQuality {
    fn default() -> Self {
        Self::Medium
    }
}

impl SSShadowQuality {
    /// Get the max ray-march steps for this quality level.
    pub fn max_steps(&self) -> u32 {
        match self {
            Self::Low => 8,
            Self::Medium => 16,
            Self::High => 32,
            Self::Ultra => 64,
        }
    }

    /// Whether temporal filtering is enabled at this quality.
    pub fn temporal_filtering(&self) -> bool {
        !matches!(self, Self::Low)
    }

    /// Temporal accumulation weight.
    pub fn temporal_weight(&self) -> f32 {
        match self {
            Self::Low => 0.0,
            Self::Medium => 0.8,
            Self::High => 0.9,
            Self::Ultra => 0.95,
        }
    }

    /// Whether to use dithered ray start offset.
    pub fn dithered_offset(&self) -> bool {
        !matches!(self, Self::Low)
    }
}

/// Configuration for screen-space shadows.
#[derive(Debug, Clone)]
pub struct ScreenSpaceShadowConfig {
    /// Whether screen-space shadows are enabled.
    pub enabled: bool,
    /// Quality preset.
    pub quality: SSShadowQuality,
    /// Maximum number of ray-march steps (overrides quality if non-zero).
    pub max_steps_override: Option<u32>,
    /// Ray length in view space units.
    pub ray_length: f32,
    /// Thickness estimation value for receivers.
    pub thickness: f32,
    /// Shadow intensity (0.0 = no shadow, 1.0 = full black).
    pub intensity: f32,
    /// Soft shadow falloff distance.
    pub softness: f32,
    /// Distance fade start (world units from camera).
    pub fade_start: f32,
    /// Distance fade end (world units from camera).
    pub fade_end: f32,
    /// Enable temporal filtering.
    pub temporal_filtering: bool,
    /// Temporal accumulation weight (0.0..1.0).
    pub temporal_weight: f32,
    /// Depth bias to avoid self-shadowing.
    pub depth_bias: f32,
    /// Normal offset bias.
    pub normal_bias: f32,
    /// Dithered ray start for temporal anti-aliasing.
    pub dithered_start: bool,
    /// Maximum shadow distance.
    pub max_distance: f32,
    /// Light direction in view space (normalized).
    pub light_direction: [f32; 3],
}

impl Default for ScreenSpaceShadowConfig {
    fn default() -> Self {
        let quality = SSShadowQuality::default();
        Self {
            enabled: true,
            quality,
            max_steps_override: None,
            ray_length: DEFAULT_RAY_LENGTH,
            thickness: DEFAULT_THICKNESS,
            intensity: DEFAULT_SHADOW_INTENSITY,
            softness: 0.5,
            fade_start: 30.0,
            fade_end: 50.0,
            temporal_filtering: quality.temporal_filtering(),
            temporal_weight: quality.temporal_weight(),
            depth_bias: 0.001,
            normal_bias: 0.005,
            dithered_start: quality.dithered_offset(),
            max_distance: 50.0,
            light_direction: [0.0, -1.0, 0.5],
        }
    }
}

impl ScreenSpaceShadowConfig {
    /// Get the effective max steps.
    pub fn effective_max_steps(&self) -> u32 {
        self.max_steps_override.unwrap_or_else(|| self.quality.max_steps())
    }

    /// Apply a quality preset, overriding quality-dependent settings.
    pub fn apply_quality(&mut self, quality: SSShadowQuality) {
        self.quality = quality;
        self.max_steps_override = None;
        self.temporal_filtering = quality.temporal_filtering();
        self.temporal_weight = quality.temporal_weight();
        self.dithered_start = quality.dithered_offset();
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.ray_length <= 0.0 {
            errors.push("ray_length must be positive".into());
        }
        if self.thickness <= 0.0 {
            errors.push("thickness must be positive".into());
        }
        if self.fade_end <= self.fade_start {
            errors.push("fade_end must be greater than fade_start".into());
        }
        errors
    }
}

// ---------------------------------------------------------------------------
// Ray march result
// ---------------------------------------------------------------------------

/// Result of a single screen-space shadow ray march.
#[derive(Debug, Clone, Copy)]
pub struct RayMarchResult {
    /// Shadow value (0.0 = fully lit, 1.0 = fully shadowed).
    pub shadow: f32,
    /// Number of steps actually taken.
    pub steps_taken: u32,
    /// Whether the ray hit an occluder.
    pub hit: bool,
    /// Distance to the nearest occluder (in screen space).
    pub hit_distance: f32,
    /// Soft shadow penumbra factor.
    pub penumbra: f32,
}

impl RayMarchResult {
    /// No shadow (fully lit).
    pub fn lit() -> Self {
        Self {
            shadow: 0.0,
            steps_taken: 0,
            hit: false,
            hit_distance: 0.0,
            penumbra: 1.0,
        }
    }

    /// Full shadow.
    pub fn shadowed(steps: u32, distance: f32) -> Self {
        Self {
            shadow: 1.0,
            steps_taken: steps,
            hit: true,
            hit_distance: distance,
            penumbra: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Screen-space shadow renderer
// ---------------------------------------------------------------------------

/// Per-frame statistics.
#[derive(Debug, Clone, Default)]
pub struct ScreenSpaceShadowStats {
    /// Number of pixels processed.
    pub pixels_processed: u64,
    /// Number of pixels that are shadowed.
    pub shadowed_pixels: u64,
    /// Average steps per pixel.
    pub avg_steps: f32,
    /// Total ray-march steps this frame.
    pub total_steps: u64,
    /// Shadow computation time (microseconds).
    pub compute_time_us: u64,
    /// Percentage of shadowed pixels.
    pub shadow_coverage: f32,
    /// Whether temporal filtering was applied.
    pub temporal_applied: bool,
}

impl fmt::Display for ScreenSpaceShadowStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SS Shadows: {} pixels, {:.1}% shadowed, avg {:.1} steps, {} us",
            self.pixels_processed,
            self.shadow_coverage * 100.0,
            self.avg_steps,
            self.compute_time_us,
        )
    }
}

/// Screen-space shadow rendering system.
pub struct ScreenSpaceShadowRenderer {
    /// Configuration.
    config: ScreenSpaceShadowConfig,
    /// Current frame shadow buffer (one value per pixel).
    shadow_buffer: Vec<f32>,
    /// Previous frame shadow buffer (for temporal filtering).
    prev_shadow_buffer: Vec<f32>,
    /// Screen dimensions.
    width: u32,
    height: u32,
    /// Frame counter.
    frame_count: u64,
    /// Statistics.
    stats: ScreenSpaceShadowStats,
    /// Noise pattern for dithered ray starts (8x8 Bayer matrix).
    dither_pattern: [f32; 64],
}

impl ScreenSpaceShadowRenderer {
    /// Create a new renderer.
    pub fn new(config: ScreenSpaceShadowConfig) -> Self {
        Self {
            config,
            shadow_buffer: Vec::new(),
            prev_shadow_buffer: Vec::new(),
            width: 0,
            height: 0,
            frame_count: 0,
            stats: ScreenSpaceShadowStats::default(),
            dither_pattern: Self::generate_bayer_8x8(),
        }
    }

    /// Generate an 8x8 Bayer dither matrix normalized to [0, 1).
    fn generate_bayer_8x8() -> [f32; 64] {
        let bayer_4x4 = [
            0, 8, 2, 10,
            12, 4, 14, 6,
            3, 11, 1, 9,
            15, 7, 13, 5,
        ];
        let mut pattern = [0.0_f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                let bx = x % 4;
                let by = y % 4;
                let base = bayer_4x4[by * 4 + bx] as f32 / 16.0;
                let offset = ((x / 4) * 2 + (y / 4)) as f32 * 0.25 / 4.0;
                pattern[y * 8 + x] = (base + offset).fract();
            }
        }
        pattern
    }

    /// Get the dither value for a pixel coordinate.
    pub fn dither_value(&self, x: u32, y: u32) -> f32 {
        let dx = (x % 8) as usize;
        let dy = (y % 8) as usize;
        self.dither_pattern[dy * 8 + dx]
    }

    /// Get the configuration.
    pub fn config(&self) -> &ScreenSpaceShadowConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: ScreenSpaceShadowConfig) {
        self.config = config;
    }

    /// Resize the shadow buffers.
    pub fn resize(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        let size = (width * height) as usize;
        self.shadow_buffer = vec![0.0; size];
        self.prev_shadow_buffer = vec![0.0; size];
        self.width = width;
        self.height = height;
    }

    /// Perform screen-space shadow ray marching.
    ///
    /// This is a CPU reference implementation; the GPU version uses compute shaders.
    pub fn compute(
        &mut self,
        depth_buffer: &[f32],
        normal_buffer: Option<&[[f32; 3]]>,
        width: u32,
        height: u32,
        view_matrix: &[f32; 16],
        projection_matrix: &[f32; 16],
    ) {
        let start = std::time::Instant::now();
        self.resize(width, height);
        self.frame_count += 1;

        if !self.config.enabled {
            self.shadow_buffer.fill(0.0);
            return;
        }

        let max_steps = self.config.effective_max_steps();
        let pixel_count = (width * height) as usize;

        // Store previous frame for temporal filtering.
        std::mem::swap(&mut self.shadow_buffer, &mut self.prev_shadow_buffer);
        self.shadow_buffer.resize(pixel_count, 0.0);

        let mut total_steps: u64 = 0;
        let mut shadowed_count: u64 = 0;

        // Normalize light direction.
        let ld = self.config.light_direction;
        let ld_len = (ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2]).sqrt().max(EPSILON);
        let light_dir = [ld[0] / ld_len, ld[1] / ld_len, ld[2] / ld_len];

        // Project light direction into screen space.
        let ss_light = project_direction_to_screen(light_dir, view_matrix, projection_matrix, width, height);

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let depth = depth_buffer[idx];

                // Skip far plane / sky.
                if depth >= 1.0 - EPSILON || depth <= EPSILON {
                    self.shadow_buffer[idx] = 0.0;
                    continue;
                }

                // Apply normal bias if normal buffer is available.
                let _normal_offset = if let Some(normals) = normal_buffer {
                    let n = normals[idx];
                    let ndotl = n[0] * light_dir[0] + n[1] * light_dir[1] + n[2] * light_dir[2];
                    if ndotl >= 0.0 {
                        // Surface faces the light; no contact shadow needed.
                        self.shadow_buffer[idx] = 0.0;
                        continue;
                    }
                    ndotl
                } else {
                    0.0
                };

                // Dithered start offset for temporal stability.
                let start_offset = if self.config.dithered_start {
                    self.dither_value(x, y) + (self.frame_count % 2) as f32 * 0.5
                } else {
                    0.0
                };

                // Ray march.
                let result = self.ray_march_pixel(
                    x, y, depth,
                    ss_light,
                    depth_buffer,
                    width, height,
                    max_steps,
                    start_offset,
                );

                let mut shadow = result.shadow * self.config.intensity;

                // Apply distance fade.
                let view_depth = depth * self.config.max_distance;
                if view_depth > self.config.fade_start {
                    let fade = 1.0 - ((view_depth - self.config.fade_start)
                        / (self.config.fade_end - self.config.fade_start))
                        .clamp(0.0, 1.0);
                    shadow *= fade;
                }

                // Temporal filtering.
                if self.config.temporal_filtering && !self.prev_shadow_buffer.is_empty() {
                    let prev = self.prev_shadow_buffer[idx];
                    let weight = self.config.temporal_weight;
                    shadow = prev * weight + shadow * (1.0 - weight);
                }

                self.shadow_buffer[idx] = shadow;
                total_steps += result.steps_taken as u64;
                if shadow > 0.1 {
                    shadowed_count += 1;
                }
            }
        }

        // Update statistics.
        self.stats.pixels_processed = pixel_count as u64;
        self.stats.shadowed_pixels = shadowed_count;
        self.stats.total_steps = total_steps;
        self.stats.avg_steps = if pixel_count > 0 {
            total_steps as f32 / pixel_count as f32
        } else {
            0.0
        };
        self.stats.shadow_coverage = if pixel_count > 0 {
            shadowed_count as f32 / pixel_count as f32
        } else {
            0.0
        };
        self.stats.temporal_applied = self.config.temporal_filtering;
        self.stats.compute_time_us = start.elapsed().as_micros() as u64;
    }

    /// Ray march a single pixel.
    fn ray_march_pixel(
        &self,
        start_x: u32,
        start_y: u32,
        start_depth: f32,
        screen_light_dir: [f32; 3],
        depth_buffer: &[f32],
        width: u32,
        height: u32,
        max_steps: u32,
        start_offset: f32,
    ) -> RayMarchResult {
        let step_size = self.config.ray_length / max_steps as f32;
        let mut min_shadow = 1.0_f32;
        let mut hit = false;
        let mut hit_dist = 0.0_f32;
        let mut steps_taken = 0_u32;

        for step in 0..max_steps {
            let t = (step as f32 + start_offset) * step_size;
            let sx = start_x as f32 + screen_light_dir[0] * t * width as f32;
            let sy = start_y as f32 + screen_light_dir[1] * t * height as f32;
            let expected_depth = start_depth + screen_light_dir[2] * t;

            steps_taken = step + 1;

            // Bounds check.
            if sx < 0.0 || sx >= width as f32 || sy < 0.0 || sy >= height as f32 {
                break;
            }

            let px = sx as u32;
            let py = sy as u32;
            let sample_idx = (py * width + px) as usize;

            if sample_idx >= depth_buffer.len() {
                break;
            }

            let sampled_depth = depth_buffer[sample_idx];

            // Check intersection.
            let depth_diff = expected_depth - sampled_depth;
            if depth_diff > self.config.depth_bias && depth_diff < self.config.thickness {
                // Hit an occluder.
                hit = true;
                hit_dist = t;

                // Soft shadow: shadow strength based on distance.
                if self.config.softness > EPSILON {
                    let penumbra = (t / self.config.softness).min(1.0);
                    min_shadow = min_shadow.min(penumbra);
                } else {
                    min_shadow = 0.0;
                    break;
                }
            }
        }

        if hit {
            RayMarchResult {
                shadow: 1.0 - min_shadow,
                steps_taken,
                hit: true,
                hit_distance: hit_dist,
                penumbra: min_shadow,
            }
        } else {
            RayMarchResult::lit()
        }
    }

    /// Get the shadow buffer (one float per pixel, 0 = lit, 1 = shadowed).
    pub fn shadow_buffer(&self) -> &[f32] {
        &self.shadow_buffer
    }

    /// Get statistics.
    pub fn stats(&self) -> &ScreenSpaceShadowStats {
        &self.stats
    }

    /// Get the current frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

impl Default for ScreenSpaceShadowRenderer {
    fn default() -> Self {
        Self::new(ScreenSpaceShadowConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Project a 3D direction into screen-space direction.
fn project_direction_to_screen(
    direction: [f32; 3],
    view: &[f32; 16],
    _proj: &[f32; 16],
    _width: u32,
    _height: u32,
) -> [f32; 3] {
    // Transform direction by view matrix (rotation only, no translation).
    let vx = direction[0] * view[0] + direction[1] * view[4] + direction[2] * view[8];
    let vy = direction[0] * view[1] + direction[1] * view[5] + direction[2] * view[9];
    let vz = direction[0] * view[2] + direction[1] * view[6] + direction[2] * view[10];

    let len = (vx * vx + vy * vy + vz * vz).sqrt().max(EPSILON);
    [vx / len, vy / len, vz / len]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayer_matrix() {
        let renderer = ScreenSpaceShadowRenderer::default();
        for i in 0..64 {
            let v = renderer.dither_pattern[i];
            assert!(v >= 0.0 && v < 1.0, "Bayer value {} out of range: {}", i, v);
        }
    }

    #[test]
    fn test_quality_presets() {
        assert!(SSShadowQuality::Low.max_steps() < SSShadowQuality::High.max_steps());
        assert!(!SSShadowQuality::Low.temporal_filtering());
        assert!(SSShadowQuality::High.temporal_filtering());
    }

    #[test]
    fn test_ray_march_result() {
        let lit = RayMarchResult::lit();
        assert!(!lit.hit);
        assert!(lit.shadow < EPSILON);

        let shad = RayMarchResult::shadowed(10, 0.5);
        assert!(shad.hit);
        assert!((shad.shadow - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_config_validation() {
        let config = ScreenSpaceShadowConfig::default();
        assert!(config.validate().is_empty());

        let mut bad = config.clone();
        bad.ray_length = -1.0;
        assert!(!bad.validate().is_empty());
    }

    #[test]
    fn test_resize() {
        let mut renderer = ScreenSpaceShadowRenderer::default();
        renderer.resize(100, 100);
        assert_eq!(renderer.shadow_buffer.len(), 10000);
    }
}
