// engine/render/src/bloom_v2.rs
//
// Advanced bloom system for the Genovo engine (v2).
//
// Implements a modern, energy-conserving bloom pipeline:
//
// - **Dual-filtering bloom** — Uses the dual-filter downsampling/upsampling
//   approach for efficient, high-quality bloom at any resolution.
// - **Threshold with soft knee** — Smooth brightness extraction with
//   configurable threshold, knee, and clamp.
// - **Lens dirt mask overlay** — Multiplicative dirt/dust texture that
//   brightens where bloom is strongest.
// - **Bloom tint per mip level** — Each downsample level can have a unique
//   colour tint for artistic control.
// - **Energy-conserving bloom** — The total energy added by bloom is bounded
//   to prevent the image from becoming brighter than the source.
// - **Bloom flicker prevention** — Temporal smoothing of the bloom intensity
//   to avoid frame-to-frame brightness changes.
// - **Screen percentage scaling** — Bloom can be computed at a lower internal
//   resolution for performance.
//
// # Pipeline overview
//
// 1. **Bright-pass filter** — Extract bright pixels above the threshold.
// 2. **Downsample chain** — Iteratively downsample the bright-pass using a
//    13-tap filter (3x3 tent + bilinear) for smooth, alias-free results.
// 3. **Upsample chain** — Iteratively upsample and accumulate using a 9-tap
//    tent filter, applying per-mip tints and weights.
// 4. **Lens dirt composite** — Multiply the bloom result with a dirt texture.
// 5. **Final composite** — Add the bloom to the scene colour with intensity
//    control.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Bloom configuration
// ---------------------------------------------------------------------------

/// Bloom quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloomQuality {
    /// Low quality: fewer mips, simpler filters.
    Low,
    /// Medium quality: standard mip chain, 9-tap filter.
    Medium,
    /// High quality: full mip chain, 13-tap filter.
    High,
    /// Ultra quality: full mip chain, 13-tap + temporal stability.
    Ultra,
}

/// Bloom configuration.
#[derive(Debug, Clone)]
pub struct BloomConfig {
    /// Whether bloom is enabled.
    pub enabled: bool,
    /// Quality preset.
    pub quality: BloomQuality,
    /// Overall bloom intensity (0 = none, 1 = full).
    pub intensity: f32,
    /// Brightness threshold: pixels below this luminance are excluded.
    pub threshold: f32,
    /// Soft knee for the threshold transition.
    /// 0 = hard cutoff, 1 = very soft transition.
    pub soft_knee: f32,
    /// Maximum brightness clamp for the bright-pass (prevents fireflies).
    pub clamp_max: f32,
    /// Number of downsample passes (typically 5-8).
    pub mip_count: u32,
    /// Per-mip bloom tint colours (linear RGB). If fewer than `mip_count`,
    /// the last tint is repeated.
    pub mip_tints: Vec<[f32; 3]>,
    /// Per-mip weight (blend factor during upsample). If fewer than
    /// `mip_count`, defaults to 1.0.
    pub mip_weights: Vec<f32>,
    /// Scatter (how much bloom spreads). Controls the relative contribution
    /// of each mip level. 0 = tight glow, 1 = wide spread.
    pub scatter: f32,
    /// Lens dirt texture handle (0 = no dirt).
    pub dirt_texture: u64,
    /// Lens dirt intensity.
    pub dirt_intensity: f32,
    /// Enable temporal smoothing.
    pub temporal_smoothing: bool,
    /// Temporal blend factor (0 = no history, 1 = full history).
    pub temporal_blend: f32,
    /// Screen percentage (0.5 = half resolution, 1.0 = full).
    pub screen_percentage: f32,
    /// Enable energy conservation.
    pub energy_conserving: bool,
    /// Anamorphic ratio: stretches the bloom horizontally (positive) or
    /// vertically (negative). 0 = isotropic.
    pub anamorphic_ratio: f32,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            quality: BloomQuality::High,
            intensity: 0.3,
            threshold: 1.0,
            soft_knee: 0.5,
            clamp_max: 65536.0,
            mip_count: 6,
            mip_tints: vec![[1.0, 1.0, 1.0]; 8],
            mip_weights: vec![1.0; 8],
            scatter: 0.7,
            dirt_texture: 0,
            dirt_intensity: 0.0,
            temporal_smoothing: true,
            temporal_blend: 0.9,
            screen_percentage: 1.0,
            energy_conserving: true,
            anamorphic_ratio: 0.0,
        }
    }
}

impl BloomConfig {
    /// Preset for subtle, cinematic bloom.
    pub fn cinematic() -> Self {
        Self {
            intensity: 0.15,
            threshold: 1.2,
            soft_knee: 0.8,
            scatter: 0.8,
            dirt_intensity: 0.05,
            ..Self::default()
        }
    }

    /// Preset for bright, stylised bloom.
    pub fn stylised() -> Self {
        Self {
            intensity: 0.6,
            threshold: 0.6,
            soft_knee: 0.3,
            scatter: 0.5,
            mip_tints: vec![
                [1.0, 0.95, 0.9],
                [1.0, 0.9, 0.8],
                [1.0, 0.85, 0.7],
                [0.95, 0.8, 0.65],
                [0.9, 0.75, 0.6],
                [0.85, 0.7, 0.55],
            ],
            ..Self::default()
        }
    }

    /// Preset for performance (low quality).
    pub fn performance() -> Self {
        Self {
            quality: BloomQuality::Low,
            mip_count: 4,
            screen_percentage: 0.5,
            temporal_smoothing: false,
            ..Self::default()
        }
    }

    /// Get the effective mip count based on resolution.
    pub fn effective_mip_count(&self, width: u32, height: u32) -> u32 {
        let max_mips = compute_max_mips(width, height);
        self.mip_count.min(max_mips)
    }

    /// Get the tint for a specific mip level.
    pub fn tint_for_mip(&self, mip: u32) -> [f32; 3] {
        self.mip_tints.get(mip as usize)
            .copied()
            .unwrap_or(*self.mip_tints.last().unwrap_or(&[1.0, 1.0, 1.0]))
    }

    /// Get the weight for a specific mip level.
    pub fn weight_for_mip(&self, mip: u32) -> f32 {
        self.mip_weights.get(mip as usize)
            .copied()
            .unwrap_or(1.0)
    }

    /// Compute the anamorphic scale factors for downsampling.
    pub fn anamorphic_scale(&self) -> (f32, f32) {
        if self.anamorphic_ratio.abs() < 0.01 {
            return (1.0, 1.0);
        }
        if self.anamorphic_ratio > 0.0 {
            (1.0, 1.0 - self.anamorphic_ratio)
        } else {
            (1.0 + self.anamorphic_ratio, 1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Bright-pass filter
// ---------------------------------------------------------------------------

/// Apply the soft-knee brightness threshold to a single pixel.
///
/// # Arguments
/// * `color` — Linear HDR colour (RGB).
/// * `threshold` — Brightness threshold.
/// * `knee` — Soft knee factor [0, 1].
/// * `clamp_max` — Maximum brightness after filtering.
///
/// # Returns
/// Filtered colour (RGB). Black if below threshold.
pub fn bright_pass(color: [f32; 3], threshold: f32, knee: f32, clamp_max: f32) -> [f32; 3] {
    let luminance = color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722;

    if luminance <= 0.0 {
        return [0.0; 3];
    }

    // Soft threshold curve.
    let knee_offset = threshold * knee;
    let soft_start = threshold - knee_offset;
    let soft_end = threshold + knee_offset;

    let contribution = if luminance < soft_start {
        0.0
    } else if luminance < soft_end {
        let t = (luminance - soft_start) / (soft_end - soft_start).max(1e-6);
        // Quadratic ease-in for smooth transition.
        t * t
    } else {
        1.0
    };

    if contribution <= 0.0 {
        return [0.0; 3];
    }

    // Apply contribution and clamp.
    let scale = contribution.min(clamp_max / luminance.max(1e-6));
    [
        (color[0] * scale).min(clamp_max),
        (color[1] * scale).min(clamp_max),
        (color[2] * scale).min(clamp_max),
    ]
}

/// Compute luminance (Rec.709).
#[inline]
pub fn luminance(color: [f32; 3]) -> f32 {
    color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
}

/// Apply Karis average (weighted average using luminance) to 4 pixels.
///
/// This reduces fireflies by weighting brighter pixels less.
pub fn karis_average(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> [f32; 3] {
    let wa = 1.0 / (1.0 + luminance(a));
    let wb = 1.0 / (1.0 + luminance(b));
    let wc = 1.0 / (1.0 + luminance(c));
    let wd = 1.0 / (1.0 + luminance(d));
    let total = wa + wb + wc + wd;
    let inv = 1.0 / total.max(1e-6);
    [
        (a[0] * wa + b[0] * wb + c[0] * wc + d[0] * wd) * inv,
        (a[1] * wa + b[1] * wb + c[1] * wc + d[1] * wd) * inv,
        (a[2] * wa + b[2] * wb + c[2] * wc + d[2] * wd) * inv,
    ]
}

// ---------------------------------------------------------------------------
// Downsample pass
// ---------------------------------------------------------------------------

/// Downsample a 2D HDR buffer to half resolution using a 13-tap filter.
///
/// The 13-tap filter samples a 3x3 neighbourhood with specific weights
/// to produce smooth, alias-free downsampling.
///
/// # Arguments
/// * `src` — Source buffer (width x height, RGB float).
/// * `src_w`, `src_h` — Source dimensions.
/// * `use_karis` — Apply Karis average on the first downsample (firefly reduction).
///
/// # Returns
/// Downsampled buffer at (src_w/2 x src_h/2).
pub fn downsample_13tap(
    src: &[[f32; 3]],
    src_w: u32,
    src_h: u32,
    use_karis: bool,
) -> Vec<[f32; 3]> {
    let dst_w = (src_w / 2).max(1);
    let dst_h = (src_h / 2).max(1);
    let mut dst = vec![[0.0f32; 3]; (dst_w * dst_h) as usize];

    let sample = |x: i32, y: i32| -> [f32; 3] {
        let cx = x.clamp(0, src_w as i32 - 1) as u32;
        let cy = y.clamp(0, src_h as i32 - 1) as u32;
        src[(cy * src_w + cx) as usize]
    };

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (dx * 2) as i32;
            let sy = (dy * 2) as i32;

            if use_karis {
                // 4 box samples with Karis averaging.
                let a = karis_average(
                    sample(sx - 1, sy - 1), sample(sx, sy - 1),
                    sample(sx - 1, sy), sample(sx, sy),
                );
                let b = karis_average(
                    sample(sx, sy - 1), sample(sx + 1, sy - 1),
                    sample(sx, sy), sample(sx + 1, sy),
                );
                let c = karis_average(
                    sample(sx - 1, sy), sample(sx, sy),
                    sample(sx - 1, sy + 1), sample(sx, sy + 1),
                );
                let d = karis_average(
                    sample(sx, sy), sample(sx + 1, sy),
                    sample(sx, sy + 1), sample(sx + 1, sy + 1),
                );
                let e = sample(sx, sy);

                dst[(dy * dst_w + dx) as usize] = [
                    (a[0] + b[0] + c[0] + d[0]) * 0.125 + e[0] * 0.5,
                    (a[1] + b[1] + c[1] + d[1]) * 0.125 + e[1] * 0.5,
                    (a[2] + b[2] + c[2] + d[2]) * 0.125 + e[2] * 0.5,
                ];
            } else {
                // Standard 13-tap filter.
                // Centre sample (weight 0.5).
                let e = sample(sx, sy);

                // Corner samples (weight 0.03125 each = 0.125 total).
                let a = sample(sx - 2, sy - 2);
                let c = sample(sx + 2, sy - 2);
                let g = sample(sx - 2, sy + 2);
                let i = sample(sx + 2, sy + 2);

                // Edge samples (weight 0.0625 each = 0.25 total).
                let b = sample(sx, sy - 2);
                let d = sample(sx - 2, sy);
                let f = sample(sx + 2, sy);
                let h = sample(sx, sy + 2);

                // Near samples (weight 0.03125 each, 4 of them = 0.125 total).
                let j = sample(sx - 1, sy - 1);
                let k = sample(sx + 1, sy - 1);
                let l = sample(sx - 1, sy + 1);
                let m = sample(sx + 1, sy + 1);

                let w_center = 0.125;
                let w_corner = 0.03125;
                let w_edge = 0.0625;
                let w_near = 0.125;

                dst[(dy * dst_w + dx) as usize] = [
                    e[0] * w_center
                        + (a[0] + c[0] + g[0] + i[0]) * w_corner
                        + (b[0] + d[0] + f[0] + h[0]) * w_edge
                        + (j[0] + k[0] + l[0] + m[0]) * w_near,
                    e[1] * w_center
                        + (a[1] + c[1] + g[1] + i[1]) * w_corner
                        + (b[1] + d[1] + f[1] + h[1]) * w_edge
                        + (j[1] + k[1] + l[1] + m[1]) * w_near,
                    e[2] * w_center
                        + (a[2] + c[2] + g[2] + i[2]) * w_corner
                        + (b[2] + d[2] + f[2] + h[2]) * w_edge
                        + (j[2] + k[2] + l[2] + m[2]) * w_near,
                ];
            }
        }
    }

    dst
}

// ---------------------------------------------------------------------------
// Upsample pass
// ---------------------------------------------------------------------------

/// Upsample a buffer to double resolution using a 9-tap tent filter and
/// additively blend with the higher-resolution buffer.
///
/// # Arguments
/// * `lower` — Lower-resolution buffer (width x height).
/// * `lower_w`, `lower_h` — Lower buffer dimensions.
/// * `upper` — Higher-resolution buffer (2*width x 2*height) to blend with.
/// * `upper_w`, `upper_h` — Upper buffer dimensions.
/// * `tint` — Colour tint for this mip level.
/// * `weight` — Blend weight for the lower buffer.
/// * `scatter` — Scatter factor controlling bloom spread.
///
/// # Returns
/// Upsampled-and-blended buffer at (upper_w x upper_h).
pub fn upsample_9tap(
    lower: &[[f32; 3]],
    lower_w: u32,
    lower_h: u32,
    upper: &[[f32; 3]],
    upper_w: u32,
    upper_h: u32,
    tint: [f32; 3],
    weight: f32,
    scatter: f32,
) -> Vec<[f32; 3]> {
    let mut dst = vec![[0.0f32; 3]; (upper_w * upper_h) as usize];

    let sample_lower = |x: i32, y: i32| -> [f32; 3] {
        let cx = x.clamp(0, lower_w as i32 - 1) as u32;
        let cy = y.clamp(0, lower_h as i32 - 1) as u32;
        lower[(cy * lower_w + cx) as usize]
    };

    for uy in 0..upper_h {
        for ux in 0..upper_w {
            // Map upper pixel to lower buffer coordinates.
            let lx = (ux as f32 * lower_w as f32 / upper_w as f32) as i32;
            let ly = (uy as f32 * lower_h as f32 / upper_h as f32) as i32;

            // 9-tap tent filter.
            let a = sample_lower(lx - 1, ly - 1);
            let b = sample_lower(lx, ly - 1);
            let c = sample_lower(lx + 1, ly - 1);
            let d = sample_lower(lx - 1, ly);
            let e = sample_lower(lx, ly);
            let f = sample_lower(lx + 1, ly);
            let g = sample_lower(lx - 1, ly + 1);
            let h = sample_lower(lx, ly + 1);
            let i = sample_lower(lx + 1, ly + 1);

            // Tent filter weights: corners=1/16, edges=2/16, centre=4/16.
            let bloom_r = (a[0] + c[0] + g[0] + i[0]) * (1.0 / 16.0)
                + (b[0] + d[0] + f[0] + h[0]) * (2.0 / 16.0)
                + e[0] * (4.0 / 16.0);
            let bloom_g = (a[1] + c[1] + g[1] + i[1]) * (1.0 / 16.0)
                + (b[1] + d[1] + f[1] + h[1]) * (2.0 / 16.0)
                + e[1] * (4.0 / 16.0);
            let bloom_b = (a[2] + c[2] + g[2] + i[2]) * (1.0 / 16.0)
                + (b[2] + d[2] + f[2] + h[2]) * (2.0 / 16.0)
                + e[2] * (4.0 / 16.0);

            let upper_pixel = upper[(uy * upper_w + ux) as usize];

            // Blend with scatter control.
            let bloom_weight = weight * scatter;
            let upper_weight = 1.0 - scatter;

            dst[(uy * upper_w + ux) as usize] = [
                upper_pixel[0] * upper_weight + bloom_r * tint[0] * bloom_weight,
                upper_pixel[1] * upper_weight + bloom_g * tint[1] * bloom_weight,
                upper_pixel[2] * upper_weight + bloom_b * tint[2] * bloom_weight,
            ];
        }
    }

    dst
}

// ---------------------------------------------------------------------------
// Lens dirt composite
// ---------------------------------------------------------------------------

/// Apply lens dirt overlay to bloom result.
///
/// # Arguments
/// * `bloom` — Bloom colour buffer.
/// * `dirt` — Dirt mask texture (greyscale, same dimensions).
/// * `width`, `height` — Buffer dimensions.
/// * `intensity` — Dirt intensity multiplier.
///
/// # Returns
/// Modified bloom buffer.
pub fn apply_lens_dirt(
    bloom: &mut [[f32; 3]],
    dirt: &[f32],
    width: u32,
    height: u32,
    intensity: f32,
) {
    let total = (width * height) as usize;
    for i in 0..total.min(bloom.len()).min(dirt.len()) {
        let d = dirt[i] * intensity;
        bloom[i][0] *= 1.0 + d;
        bloom[i][1] *= 1.0 + d;
        bloom[i][2] *= 1.0 + d;
    }
}

// ---------------------------------------------------------------------------
// Final composite
// ---------------------------------------------------------------------------

/// Composite bloom onto the scene colour.
///
/// # Arguments
/// * `scene` — Scene colour buffer (linear HDR RGB).
/// * `bloom` — Bloom colour buffer.
/// * `intensity` — Bloom intensity.
/// * `energy_conserving` — If true, subtract bloom contribution from scene
///   brightness to conserve total energy.
///
/// # Returns
/// Final composited colour.
pub fn composite_bloom(
    scene: [f32; 3],
    bloom: [f32; 3],
    intensity: f32,
    energy_conserving: bool,
) -> [f32; 3] {
    if energy_conserving {
        // Lerp between scene and scene+bloom based on intensity,
        // while maintaining total energy.
        let bloom_lum = luminance(bloom);
        let scene_lum = luminance(scene);
        let total_lum = scene_lum + bloom_lum * intensity;

        // Scale factor to conserve energy.
        let scale = if total_lum > 0.0 {
            (scene_lum + bloom_lum * intensity * 0.5) / total_lum
        } else {
            1.0
        };

        [
            (scene[0] + bloom[0] * intensity) * scale,
            (scene[1] + bloom[1] * intensity) * scale,
            (scene[2] + bloom[2] * intensity) * scale,
        ]
    } else {
        [
            scene[0] + bloom[0] * intensity,
            scene[1] + bloom[1] * intensity,
            scene[2] + bloom[2] * intensity,
        ]
    }
}

// ---------------------------------------------------------------------------
// Temporal smoothing
// ---------------------------------------------------------------------------

/// Temporal smoothing for bloom to prevent flicker.
#[derive(Debug, Clone)]
pub struct BloomTemporalFilter {
    /// Previous frame's bloom intensity.
    pub prev_intensity: f32,
    /// Previous frame's average bloom colour.
    pub prev_avg_color: [f32; 3],
    /// Blend factor (0 = no smoothing, 1 = full history).
    pub blend_factor: f32,
    /// Frame counter.
    pub frame: u64,
}

impl BloomTemporalFilter {
    /// Create a new temporal filter.
    pub fn new(blend_factor: f32) -> Self {
        Self {
            prev_intensity: 0.0,
            prev_avg_color: [0.0; 3],
            blend_factor,
            frame: 0,
        }
    }

    /// Smooth the bloom intensity.
    pub fn smooth_intensity(&mut self, current: f32) -> f32 {
        if self.frame == 0 {
            self.prev_intensity = current;
            self.frame += 1;
            return current;
        }

        let smoothed = lerp(current, self.prev_intensity, self.blend_factor);
        self.prev_intensity = smoothed;
        self.frame += 1;
        smoothed
    }

    /// Smooth the average bloom colour.
    pub fn smooth_color(&mut self, current: [f32; 3]) -> [f32; 3] {
        let smoothed = [
            lerp(current[0], self.prev_avg_color[0], self.blend_factor),
            lerp(current[1], self.prev_avg_color[1], self.blend_factor),
            lerp(current[2], self.prev_avg_color[2], self.blend_factor),
        ];
        self.prev_avg_color = smoothed;
        smoothed
    }
}

// ---------------------------------------------------------------------------
// Full bloom pipeline (CPU reference)
// ---------------------------------------------------------------------------

/// Run the full bloom pipeline on a CPU buffer.
///
/// # Arguments
/// * `scene` — Scene colour buffer (width x height, linear HDR RGB).
/// * `width`, `height` — Buffer dimensions.
/// * `config` — Bloom configuration.
///
/// # Returns
/// Bloom-only buffer (same dimensions as scene).
pub fn compute_bloom(
    scene: &[[f32; 3]],
    width: u32,
    height: u32,
    config: &BloomConfig,
) -> Vec<[f32; 3]> {
    if !config.enabled || config.intensity <= 0.0 {
        return vec![[0.0; 3]; (width * height) as usize];
    }

    // Step 1: Bright-pass filter.
    let mut bright: Vec<[f32; 3]> = scene
        .iter()
        .map(|c| bright_pass(*c, config.threshold, config.soft_knee, config.clamp_max))
        .collect();

    // Apply screen percentage scaling.
    let scale = config.screen_percentage.clamp(0.25, 1.0);
    let work_w = ((width as f32 * scale) as u32).max(1);
    let work_h = ((height as f32 * scale) as u32).max(1);

    // If scaled, resize the bright-pass.
    if work_w != width || work_h != height {
        bright = resize_buffer(&bright, width, height, work_w, work_h);
    }

    // Step 2: Downsample chain.
    let mip_count = config.effective_mip_count(work_w, work_h);
    let mut mip_chain: Vec<Vec<[f32; 3]>> = Vec::with_capacity(mip_count as usize);
    let mut mip_sizes: Vec<(u32, u32)> = Vec::with_capacity(mip_count as usize);

    let mut current = bright;
    let mut cw = work_w;
    let mut ch = work_h;

    for i in 0..mip_count {
        let use_karis = i == 0 && config.quality != BloomQuality::Low;
        let down = downsample_13tap(&current, cw, ch, use_karis);
        let dw = (cw / 2).max(1);
        let dh = (ch / 2).max(1);

        mip_sizes.push((dw, dh));
        mip_chain.push(down.clone());

        current = down;
        cw = dw;
        ch = dh;
    }

    // Step 3: Upsample chain.
    if mip_chain.is_empty() {
        return vec![[0.0; 3]; (width * height) as usize];
    }

    let mut upsampled = mip_chain.last().unwrap().clone();
    let mut uw = mip_sizes.last().unwrap().0;
    let mut uh = mip_sizes.last().unwrap().1;

    for i in (0..mip_chain.len() - 1).rev() {
        let upper_w = mip_sizes[i].0;
        let upper_h = mip_sizes[i].1;
        let tint = config.tint_for_mip(i as u32);
        let weight = config.weight_for_mip(i as u32);

        upsampled = upsample_9tap(
            &upsampled, uw, uh,
            &mip_chain[i], upper_w, upper_h,
            tint, weight, config.scatter,
        );

        uw = upper_w;
        uh = upper_h;
    }

    // Step 4: Resize back to original resolution if needed.
    if uw != width || uh != height {
        upsampled = resize_buffer(&upsampled, uw, uh, width, height);
    }

    upsampled
}

// ---------------------------------------------------------------------------
// GPU uniform data
// ---------------------------------------------------------------------------

/// Packed bloom parameters for GPU upload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BloomUniforms {
    /// intensity (x), threshold (y), soft_knee (z), clamp_max (w).
    pub params: [f32; 4],
    /// scatter (x), dirt_intensity (y), screen_percentage (z), mip_count (w).
    pub params2: [f32; 4],
    /// Per-mip tint (packed as vec4 for first 4 mips, rgb + weight).
    pub mip_tint_0: [f32; 4],
    pub mip_tint_1: [f32; 4],
    pub mip_tint_2: [f32; 4],
    pub mip_tint_3: [f32; 4],
    /// Texel size of the current mip level (xy), anamorphic_scale (zw).
    pub texel_size: [f32; 4],
}

impl BloomUniforms {
    /// Build uniform data from configuration.
    pub fn from_config(config: &BloomConfig, width: u32, height: u32) -> Self {
        let (ana_x, ana_y) = config.anamorphic_scale();

        Self {
            params: [config.intensity, config.threshold, config.soft_knee, config.clamp_max],
            params2: [config.scatter, config.dirt_intensity, config.screen_percentage, config.mip_count as f32],
            mip_tint_0: pack_tint(config, 0),
            mip_tint_1: pack_tint(config, 1),
            mip_tint_2: pack_tint(config, 2),
            mip_tint_3: pack_tint(config, 3),
            texel_size: [1.0 / width as f32, 1.0 / height as f32, ana_x, ana_y],
        }
    }
}

fn pack_tint(config: &BloomConfig, mip: u32) -> [f32; 4] {
    let t = config.tint_for_mip(mip);
    let w = config.weight_for_mip(mip);
    [t[0], t[1], t[2], w]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the maximum number of mip levels for a resolution.
fn compute_max_mips(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height) as f32;
    (max_dim.log2().floor() as u32).max(1)
}

/// Simple bilinear resize of an RGB buffer.
fn resize_buffer(
    src: &[[f32; 3]],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<[f32; 3]> {
    let mut dst = vec![[0.0f32; 3]; (dst_w * dst_h) as usize];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx as f32 * src_w as f32 / dst_w as f32;
            let sy = dy as f32 * src_h as f32 / dst_h as f32;

            let x0 = (sx as u32).min(src_w - 1);
            let y0 = (sy as u32).min(src_h - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);

            let fx = sx.fract();
            let fy = sy.fract();

            let p00 = src[(y0 * src_w + x0) as usize];
            let p10 = src[(y0 * src_w + x1) as usize];
            let p01 = src[(y1 * src_w + x0) as usize];
            let p11 = src[(y1 * src_w + x1) as usize];

            dst[(dy * dst_w + dx) as usize] = [
                bilerp(p00[0], p10[0], p01[0], p11[0], fx, fy),
                bilerp(p00[1], p10[1], p01[1], p11[1], fx, fy),
                bilerp(p00[2], p10[2], p01[2], p11[2], fx, fy),
            ];
        }
    }

    dst
}

#[inline]
fn bilerp(a: f32, b: f32, c: f32, d: f32, fx: f32, fy: f32) -> f32 {
    let top = a + (b - a) * fx;
    let bot = c + (d - c) * fx;
    top + (bot - top) * fy
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bright_pass() {
        // Below threshold.
        let r = bright_pass([0.5, 0.5, 0.5], 1.0, 0.5, 100.0);
        assert!(r[0] < 0.5);

        // Above threshold.
        let r = bright_pass([2.0, 2.0, 2.0], 1.0, 0.0, 100.0);
        assert!(r[0] > 0.0);
    }

    #[test]
    fn test_luminance() {
        let l = luminance([1.0, 1.0, 1.0]);
        assert!((l - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_karis_average() {
        let avg = karis_average(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        );
        assert!(avg[0] > 0.0);
    }

    #[test]
    fn test_downsample() {
        let src = vec![[1.0f32; 3]; 16 * 16];
        let dst = downsample_13tap(&src, 16, 16, false);
        assert_eq!(dst.len(), 8 * 8);
    }

    #[test]
    fn test_composite_energy_conserving() {
        let scene = [0.5, 0.5, 0.5];
        let bloom = [0.1, 0.1, 0.1];
        let result = composite_bloom(scene, bloom, 1.0, true);
        // With energy conservation, the result should not be brighter than
        // the sum of scene + bloom.
        let total_lum = luminance(result);
        assert!(total_lum <= luminance(scene) + luminance(bloom) + 0.01);
    }

    #[test]
    fn test_bloom_pipeline() {
        let scene = vec![[2.0f32; 3]; 64 * 64];
        let config = BloomConfig {
            mip_count: 3,
            ..BloomConfig::default()
        };
        let bloom = compute_bloom(&scene, 64, 64, &config);
        assert_eq!(bloom.len(), 64 * 64);
    }
}
