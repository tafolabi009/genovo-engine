// engine/render/src/tone_map_v2.rs
//
// Extended tone mapping and auto-exposure system for the Genovo engine (v2).
//
// Provides advanced tone mapping operators and automatic exposure control:
//
// - **Exposure compensation** — Manual EV adjustment.
// - **Local tone mapping** — Bilateral decomposition into base/detail layers
//   for local contrast enhancement.
// - **HDR histogram analysis** — Compute luminance histograms for exposure
//   decisions.
// - **Auto-exposure** — Adaptive exposure with multiple metering modes:
//   centre-weighted, spot, matrix (evaluative).
// - **Exposure adaptation speed** — Smooth transitions between exposure
//   levels with configurable bright/dark adaptation rates.
// - **EV100 computation** — Photography-standard exposure value.
// - **Luminance-based exposure** — Compute exposure from scene luminance
//   statistics (average, median, key value).
//
// # Pipeline integration
//
// The auto-exposure system runs as a compute pass that reads the HDR scene
// colour, builds a luminance histogram, and computes the target exposure.
// The exposure value is then used in the tone mapping pass.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// EV100 and exposure computation
// ---------------------------------------------------------------------------

/// Compute EV100 from camera settings.
///
/// EV100 = log2(N^2 / t) where N = f-number, t = shutter time.
///
/// # Arguments
/// * `aperture` — f-number (e.g. 2.8, 4.0, 5.6).
/// * `shutter_time` — Shutter time in seconds.
/// * `iso` — ISO sensitivity.
pub fn ev100_from_camera(aperture: f32, shutter_time: f32, iso: f32) -> f32 {
    let ev = (aperture * aperture / shutter_time.max(1e-10)).log2();
    ev - (iso / 100.0).log2()
}

/// Compute EV100 from average scene luminance.
///
/// Using the incident light metering equation.
///
/// # Arguments
/// * `avg_luminance` — Average scene luminance in cd/m^2.
pub fn ev100_from_luminance(avg_luminance: f32) -> f32 {
    if avg_luminance <= 0.0 {
        return 0.0;
    }
    (avg_luminance * 100.0 / 12.5).log2()
}

/// Compute the exposure factor from EV100.
///
/// The exposure factor converts scene luminance to [0, 1] range.
///
/// # Arguments
/// * `ev100` — Exposure value at ISO 100.
pub fn exposure_from_ev100(ev100: f32) -> f32 {
    1.0 / (2.0_f32.powf(ev100) * 1.2)
}

/// Compute the maximum luminance that will not be clipped at a given EV100.
pub fn max_luminance_from_ev100(ev100: f32) -> f32 {
    1.2 * 2.0_f32.powf(ev100)
}

/// Apply manual exposure compensation.
///
/// # Arguments
/// * `ev100` — Base exposure value.
/// * `compensation` — EV compensation (positive = brighter, negative = darker).
pub fn apply_exposure_compensation(ev100: f32, compensation: f32) -> f32 {
    ev100 - compensation
}

// ---------------------------------------------------------------------------
// Luminance computation
// ---------------------------------------------------------------------------

/// Compute Rec.709 luminance.
#[inline]
pub fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Compute log-average luminance of a buffer.
///
/// Uses the log-average (geometric mean) which is more robust to outliers.
pub fn log_average_luminance(pixels: &[[f32; 3]], delta: f32) -> f32 {
    if pixels.is_empty() {
        return 0.001;
    }

    let mut sum = 0.0_f64;
    for pixel in pixels {
        let l = luminance(pixel[0], pixel[1], pixel[2]) as f64;
        sum += (l + delta as f64).ln();
    }

    let avg = (sum / pixels.len() as f64).exp() as f32;
    avg.max(0.001)
}

/// Compute various luminance statistics for a buffer.
#[derive(Debug, Clone)]
pub struct LuminanceStats {
    /// Minimum luminance.
    pub min: f32,
    /// Maximum luminance.
    pub max: f32,
    /// Arithmetic mean luminance.
    pub mean: f32,
    /// Log-average (geometric mean) luminance.
    pub log_average: f32,
    /// Median luminance.
    pub median: f32,
    /// Standard deviation of luminance.
    pub std_dev: f32,
    /// 5th percentile.
    pub p05: f32,
    /// 95th percentile.
    pub p95: f32,
    /// Total pixel count.
    pub pixel_count: u32,
}

impl LuminanceStats {
    /// Compute luminance statistics from a pixel buffer.
    pub fn compute(pixels: &[[f32; 3]]) -> Self {
        if pixels.is_empty() {
            return Self {
                min: 0.0, max: 0.0, mean: 0.0, log_average: 0.0,
                median: 0.0, std_dev: 0.0, p05: 0.0, p95: 0.0,
                pixel_count: 0,
            };
        }

        let mut luminances: Vec<f32> = pixels
            .iter()
            .map(|p| luminance(p[0], p[1], p[2]).max(0.0))
            .collect();

        luminances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = luminances.len();
        let min = luminances[0];
        let max = luminances[n - 1];
        let mean: f32 = luminances.iter().sum::<f32>() / n as f32;
        let median = luminances[n / 2];
        let p05 = luminances[(n as f32 * 0.05) as usize];
        let p95 = luminances[(n as f32 * 0.95).min(n as f32 - 1.0) as usize];

        let log_avg = log_average_luminance(pixels, 0.001);

        let variance: f32 = luminances.iter()
            .map(|l| (l - mean) * (l - mean))
            .sum::<f32>() / n as f32;
        let std_dev = variance.sqrt();

        Self {
            min, max, mean, log_average: log_avg,
            median, std_dev, p05, p95,
            pixel_count: n as u32,
        }
    }

    /// Compute EV100 from the log-average luminance.
    pub fn ev100(&self) -> f32 {
        ev100_from_luminance(self.log_average)
    }

    /// Dynamic range in stops.
    pub fn dynamic_range(&self) -> f32 {
        if self.min > 0.0 {
            (self.max / self.min).log2()
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// HDR histogram
// ---------------------------------------------------------------------------

/// HDR luminance histogram with logarithmic binning.
#[derive(Debug, Clone)]
pub struct HdrHistogram {
    /// Histogram bins (log-luminance space).
    pub bins: Vec<u32>,
    /// Number of bins.
    pub bin_count: u32,
    /// Minimum log-luminance (left edge of first bin).
    pub log_min: f32,
    /// Maximum log-luminance (right edge of last bin).
    pub log_max: f32,
    /// Total number of samples.
    pub total_samples: u32,
}

impl HdrHistogram {
    /// Create a new histogram with the given range and bin count.
    pub fn new(bin_count: u32, log_min: f32, log_max: f32) -> Self {
        Self {
            bins: vec![0u32; bin_count as usize],
            bin_count,
            log_min,
            log_max,
            total_samples: 0,
        }
    }

    /// Default histogram for typical HDR scenes.
    pub fn default_hdr() -> Self {
        // Range: 10^-4 to 10^4 luminance ≈ -9.2 to 9.2 in log2.
        Self::new(256, -10.0, 10.0)
    }

    /// Add a luminance sample.
    pub fn add_sample(&mut self, lum: f32) {
        if lum <= 0.0 {
            return;
        }
        let log_lum = lum.log2();
        let t = (log_lum - self.log_min) / (self.log_max - self.log_min);
        let bin = (t * self.bin_count as f32) as i32;
        let bin = bin.clamp(0, self.bin_count as i32 - 1) as u32;
        self.bins[bin as usize] += 1;
        self.total_samples += 1;
    }

    /// Build the histogram from a pixel buffer.
    pub fn build(&mut self, pixels: &[[f32; 3]]) {
        self.clear();
        for pixel in pixels {
            let lum = luminance(pixel[0], pixel[1], pixel[2]);
            self.add_sample(lum);
        }
    }

    /// Clear all bins.
    pub fn clear(&mut self) {
        for bin in &mut self.bins {
            *bin = 0;
        }
        self.total_samples = 0;
    }

    /// Get the luminance at a given percentile.
    pub fn percentile(&self, p: f32) -> f32 {
        if self.total_samples == 0 {
            return 0.001;
        }

        let target = (self.total_samples as f32 * p) as u32;
        let mut acc = 0u32;

        for (i, &count) in self.bins.iter().enumerate() {
            acc += count;
            if acc >= target {
                let t = (i as f32 + 0.5) / self.bin_count as f32;
                let log_lum = self.log_min + t * (self.log_max - self.log_min);
                return 2.0_f32.powf(log_lum);
            }
        }

        2.0_f32.powf(self.log_max)
    }

    /// Average luminance from the histogram.
    pub fn average_luminance(&self) -> f32 {
        if self.total_samples == 0 {
            return 0.001;
        }

        let mut sum = 0.0_f64;
        for (i, &count) in self.bins.iter().enumerate() {
            if count > 0 {
                let t = (i as f32 + 0.5) / self.bin_count as f32;
                let log_lum = self.log_min + t * (self.log_max - self.log_min);
                let lum = 2.0_f64.powf(log_lum as f64);
                sum += lum * count as f64;
            }
        }

        (sum / self.total_samples as f64) as f32
    }

    /// Weighted average using a weighting function (e.g. for centre-weighted metering).
    pub fn weighted_average(&self, weights: &[f32]) -> f32 {
        if self.total_samples == 0 || weights.is_empty() {
            return 0.001;
        }

        let mut sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        for (i, &count) in self.bins.iter().enumerate() {
            if count > 0 {
                let w = weights.get(i).copied().unwrap_or(1.0) as f64;
                let t = (i as f32 + 0.5) / self.bin_count as f32;
                let log_lum = self.log_min + t * (self.log_max - self.log_min);
                let lum = 2.0_f64.powf(log_lum as f64);
                sum += lum * count as f64 * w;
                weight_sum += count as f64 * w;
            }
        }

        if weight_sum > 0.0 {
            (sum / weight_sum) as f32
        } else {
            0.001
        }
    }
}

// ---------------------------------------------------------------------------
// Metering modes
// ---------------------------------------------------------------------------

/// Auto-exposure metering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeteringMode {
    /// Average the entire screen equally.
    Average,
    /// Centre-weighted average (centre of the screen contributes more).
    CenterWeighted,
    /// Spot metering (small central region only).
    Spot,
    /// Matrix/evaluative (divide screen into zones, apply heuristics).
    Matrix,
}

/// Generate per-pixel metering weights for a given mode.
///
/// # Arguments
/// * `width`, `height` — Screen dimensions.
/// * `mode` — Metering mode.
///
/// # Returns
/// Weight map (width x height, float [0, 1]).
pub fn generate_metering_weights(width: u32, height: u32, mode: MeteringMode) -> Vec<f32> {
    let total = (width * height) as usize;
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    let max_r = (cx * cx + cy * cy).sqrt();

    let mut weights = vec![1.0f32; total];

    match mode {
        MeteringMode::Average => {
            // All pixels have equal weight (already 1.0).
        }
        MeteringMode::CenterWeighted => {
            for y in 0..height {
                for x in 0..width {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt() / max_r;
                    // Gaussian-like falloff.
                    let w = (-dist * dist * 4.0).exp();
                    weights[(y * width + x) as usize] = w.max(0.1);
                }
            }
        }
        MeteringMode::Spot => {
            let spot_radius = (width.min(height) as f32 * 0.05).max(1.0);
            for y in 0..height {
                for x in 0..width {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist <= spot_radius {
                        weights[(y * width + x) as usize] = 1.0;
                    } else {
                        weights[(y * width + x) as usize] = 0.0;
                    }
                }
            }
        }
        MeteringMode::Matrix => {
            // Divide into a 5x5 grid with different weights.
            let zone_weights = [
                [0.5, 0.6, 0.7, 0.6, 0.5],
                [0.6, 0.8, 0.9, 0.8, 0.6],
                [0.7, 0.9, 1.0, 0.9, 0.7],
                [0.6, 0.8, 0.9, 0.8, 0.6],
                [0.5, 0.6, 0.7, 0.6, 0.5],
            ];
            let zone_w = width as f32 / 5.0;
            let zone_h = height as f32 / 5.0;

            for y in 0..height {
                for x in 0..width {
                    let zx = ((x as f32 / zone_w) as usize).min(4);
                    let zy = ((y as f32 / zone_h) as usize).min(4);
                    weights[(y * width + x) as usize] = zone_weights[zy][zx];
                }
            }
        }
    }

    weights
}

// ---------------------------------------------------------------------------
// Auto-exposure
// ---------------------------------------------------------------------------

/// Auto-exposure configuration.
#[derive(Debug, Clone)]
pub struct AutoExposureConfig {
    /// Whether auto-exposure is enabled.
    pub enabled: bool,
    /// Metering mode.
    pub metering: MeteringMode,
    /// Minimum EV100.
    pub min_ev: f32,
    /// Maximum EV100.
    pub max_ev: f32,
    /// Manual exposure compensation (EV).
    pub compensation: f32,
    /// Adaptation speed for brightening (EV/second).
    pub bright_adapt_speed: f32,
    /// Adaptation speed for darkening (EV/second).
    pub dark_adapt_speed: f32,
    /// Target key value (mid-grey luminance fraction, typically 0.18).
    pub key_value: f32,
    /// Histogram low clip percentile (pixels below this are ignored).
    pub histogram_low_clip: f32,
    /// Histogram high clip percentile (pixels above this are ignored).
    pub histogram_high_clip: f32,
    /// Use histogram-based (true) or average-based (false) luminance.
    pub use_histogram: bool,
}

impl Default for AutoExposureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metering: MeteringMode::CenterWeighted,
            min_ev: -4.0,
            max_ev: 16.0,
            compensation: 0.0,
            bright_adapt_speed: 3.0,
            dark_adapt_speed: 1.0,
            key_value: 0.18,
            histogram_low_clip: 0.05,
            histogram_high_clip: 0.95,
            use_histogram: true,
        }
    }
}

/// Auto-exposure state.
#[derive(Debug, Clone)]
pub struct AutoExposureState {
    /// Current exposure EV100.
    pub current_ev: f32,
    /// Target exposure EV100.
    pub target_ev: f32,
    /// Current exposure factor.
    pub exposure: f32,
    /// Previous frame's average luminance.
    pub prev_luminance: f32,
    /// Frame counter.
    pub frame: u64,
    /// Whether the state has been initialised.
    pub initialised: bool,
}

impl AutoExposureState {
    /// Create a new auto-exposure state.
    pub fn new() -> Self {
        Self {
            current_ev: 0.0,
            target_ev: 0.0,
            exposure: 1.0,
            prev_luminance: 0.18,
            frame: 0,
            initialised: false,
        }
    }

    /// Update the auto-exposure state.
    ///
    /// # Arguments
    /// * `avg_luminance` — Measured average scene luminance.
    /// * `dt` — Delta time in seconds.
    /// * `config` — Auto-exposure configuration.
    pub fn update(&mut self, avg_luminance: f32, dt: f32, config: &AutoExposureConfig) {
        if !config.enabled {
            return;
        }

        let lum = avg_luminance.max(0.001);

        // Compute target EV.
        let key = config.key_value;
        self.target_ev = (lum / key).log2();
        self.target_ev = apply_exposure_compensation(self.target_ev, config.compensation);
        self.target_ev = self.target_ev.clamp(config.min_ev, config.max_ev);

        if !self.initialised {
            self.current_ev = self.target_ev;
            self.initialised = true;
        } else {
            // Smooth adaptation.
            let speed = if self.target_ev > self.current_ev {
                config.bright_adapt_speed
            } else {
                config.dark_adapt_speed
            };

            let diff = self.target_ev - self.current_ev;
            let max_change = speed * dt;
            if diff.abs() <= max_change {
                self.current_ev = self.target_ev;
            } else {
                self.current_ev += diff.signum() * max_change;
            }
        }

        self.exposure = exposure_from_ev100(self.current_ev);
        self.prev_luminance = lum;
        self.frame += 1;
    }

    /// Compute the exposure factor to apply to the scene.
    pub fn exposure_factor(&self) -> f32 {
        self.exposure
    }
}

impl Default for AutoExposureState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Local tone mapping
// ---------------------------------------------------------------------------

/// Local tone mapping using bilateral decomposition.
///
/// Separates the image into a base layer (low frequency) and detail layer
/// (high frequency), compresses the base layer's dynamic range, and
/// recombines.
///
/// # Arguments
/// * `pixels` — HDR pixel buffer (linear RGB).
/// * `width`, `height` — Dimensions.
/// * `spatial_sigma` — Bilateral spatial sigma.
/// * `range_sigma` — Bilateral range sigma.
/// * `base_contrast` — Compression factor for the base layer (0-1, lower = more compression).
/// * `detail_boost` — Detail layer multiplier (1 = neutral, >1 = enhance).
pub fn local_tone_map(
    pixels: &[[f32; 3]],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
    base_contrast: f32,
    detail_boost: f32,
) -> Vec<[f32; 3]> {
    // Step 1: Convert to log-luminance.
    let mut log_lum: Vec<f32> = pixels
        .iter()
        .map(|p| {
            let l = luminance(p[0], p[1], p[2]).max(1e-6);
            l.ln()
        })
        .collect();

    // Step 2: Bilateral filter the log-luminance to get the base layer.
    let base = bilateral_filter_1ch(
        &log_lum, width, height,
        spatial_sigma, range_sigma,
    );

    // Step 3: Detail layer = log_lum - base.
    let detail: Vec<f32> = log_lum.iter()
        .zip(base.iter())
        .map(|(l, b)| l - b)
        .collect();

    // Step 4: Compress the base layer.
    let base_min = base.iter().cloned().fold(f32::MAX, f32::min);
    let base_max = base.iter().cloned().fold(f32::MIN, f32::max);
    let base_range = (base_max - base_min).max(0.01);

    let target_range = base_range * base_contrast;
    let scale = target_range / base_range;

    let compressed_base: Vec<f32> = base.iter()
        .map(|b| base_min + (b - base_min) * scale)
        .collect();

    // Step 5: Recombine and convert back.
    let mut result = vec![[0.0f32; 3]; pixels.len()];

    for i in 0..pixels.len() {
        let new_log_lum = compressed_base[i] + detail[i] * detail_boost;
        let new_lum = new_log_lum.exp();
        let old_lum = luminance(pixels[i][0], pixels[i][1], pixels[i][2]).max(1e-6);
        let ratio = new_lum / old_lum;

        result[i] = [
            (pixels[i][0] * ratio).max(0.0),
            (pixels[i][1] * ratio).max(0.0),
            (pixels[i][2] * ratio).max(0.0),
        ];
    }

    result
}

/// Simple bilateral filter for a single-channel buffer.
fn bilateral_filter_1ch(
    src: &[f32],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
) -> Vec<f32> {
    let radius = (spatial_sigma * 2.0).ceil() as i32;
    let inv_spatial = -0.5 / (spatial_sigma * spatial_sigma);
    let inv_range = -0.5 / (range_sigma * range_sigma);

    let mut dst = vec![0.0f32; src.len()];

    for y in 0..height {
        for x in 0..width {
            let center = src[(y * width + x) as usize];
            let mut acc = 0.0f32;
            let mut weight_sum = 0.0f32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let sx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let sy = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                    let neighbour = src[(sy * width + sx) as usize];

                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let ws = (dist_sq * inv_spatial).exp();

                    let diff = center - neighbour;
                    let wr = (diff * diff * inv_range).exp();

                    let w = ws * wr;
                    acc += neighbour * w;
                    weight_sum += w;
                }
            }

            dst[(y * width + x) as usize] = if weight_sum > 0.0 {
                acc / weight_sum
            } else {
                center
            };
        }
    }

    dst
}

// ---------------------------------------------------------------------------
// GPU uniform data
// ---------------------------------------------------------------------------

/// Packed auto-exposure uniforms for GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExposureUniforms {
    /// exposure_factor (x), ev100 (y), min_ev (z), max_ev (w).
    pub params: [f32; 4],
    /// compensation (x), key_value (y), bright_speed (z), dark_speed (w).
    pub adaptation: [f32; 4],
    /// histogram_low_clip (x), histogram_high_clip (y), metering_mode (z), use_histogram (w).
    pub histogram_params: [f32; 4],
}

impl ExposureUniforms {
    /// Build from config and state.
    pub fn from_config_state(config: &AutoExposureConfig, state: &AutoExposureState) -> Self {
        Self {
            params: [
                state.exposure,
                state.current_ev,
                config.min_ev,
                config.max_ev,
            ],
            adaptation: [
                config.compensation,
                config.key_value,
                config.bright_adapt_speed,
                config.dark_adapt_speed,
            ],
            histogram_params: [
                config.histogram_low_clip,
                config.histogram_high_clip,
                match config.metering {
                    MeteringMode::Average => 0.0,
                    MeteringMode::CenterWeighted => 1.0,
                    MeteringMode::Spot => 2.0,
                    MeteringMode::Matrix => 3.0,
                },
                if config.use_histogram { 1.0 } else { 0.0 },
            ],
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
    fn test_ev100() {
        // Standard sunny-16 rule: f/16, 1/100s, ISO 100.
        let ev = ev100_from_camera(16.0, 1.0 / 100.0, 100.0);
        // Should be approximately 14.6.
        assert!((ev - 14.6).abs() < 0.5);
    }

    #[test]
    fn test_exposure_roundtrip() {
        let ev = 10.0;
        let exposure = exposure_from_ev100(ev);
        let max_lum = max_luminance_from_ev100(ev);
        // exposure * max_lum should be approximately 1.
        assert!((exposure * max_lum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_log_average_luminance() {
        let pixels = vec![[0.5f32; 3]; 100];
        let avg = log_average_luminance(&pixels, 0.001);
        let expected_lum = luminance(0.5, 0.5, 0.5);
        assert!((avg - expected_lum).abs() < 0.01);
    }

    #[test]
    fn test_hdr_histogram() {
        let mut hist = HdrHistogram::default_hdr();
        let pixels = vec![[0.5f32; 3]; 1000];
        hist.build(&pixels);
        assert_eq!(hist.total_samples, 1000);

        let median = hist.percentile(0.5);
        assert!(median > 0.0);
    }

    #[test]
    fn test_auto_exposure() {
        let config = AutoExposureConfig::default();
        let mut state = AutoExposureState::new();

        // First frame: instant adaptation.
        state.update(0.18, 1.0 / 60.0, &config);
        assert!(state.initialised);

        // Change scene brightness: should adapt over time.
        let prev_ev = state.current_ev;
        for _ in 0..60 {
            state.update(1.0, 1.0 / 60.0, &config);
        }
        // Should have moved towards the brighter target.
        assert!(state.current_ev != prev_ev);
    }

    #[test]
    fn test_metering_weights() {
        let weights = generate_metering_weights(100, 100, MeteringMode::CenterWeighted);
        assert_eq!(weights.len(), 10000);
        // Centre should have highest weight.
        let center = weights[50 * 100 + 50];
        let corner = weights[0];
        assert!(center > corner);
    }

    #[test]
    fn test_luminance_stats() {
        let pixels = vec![[0.5f32; 3]; 100];
        let stats = LuminanceStats::compute(&pixels);
        assert!(stats.mean > 0.0);
        assert!(stats.std_dev < 0.01); // Uniform.
    }
}
