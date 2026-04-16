// engine/render/src/shadows/pcf.rs
//
// Percentage Closer Filtering (PCF) and Percentage Closer Soft Shadows
// (PCSS) for shadow map sampling. Provides pre-computed filter kernels,
// Poisson disk sample distributions, and PCSS blocker search logic.

use glam::Vec2;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// PCF Kernel sizes
// ---------------------------------------------------------------------------

/// Pre-defined PCF kernel sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcfKernel {
    /// 2x2 hardware PCF (the minimum).
    Pcf2x2,
    /// 3x3 uniform kernel (9 samples).
    Pcf3x3,
    /// 5x5 uniform kernel (25 samples).
    Pcf5x5,
    /// 7x7 uniform kernel (49 samples).
    Pcf7x7,
    /// Poisson disk with a specified number of samples.
    PoissonDisk { samples: u32 },
    /// Rotated Poisson disk (per-pixel rotation to reduce banding).
    RotatedPoissonDisk { samples: u32 },
}

impl PcfKernel {
    /// Number of texture samples required.
    pub fn sample_count(&self) -> u32 {
        match self {
            Self::Pcf2x2 => 4,
            Self::Pcf3x3 => 9,
            Self::Pcf5x5 => 25,
            Self::Pcf7x7 => 49,
            Self::PoissonDisk { samples } => *samples,
            Self::RotatedPoissonDisk { samples } => *samples,
        }
    }

    /// Half-size of the kernel in texels (radius).
    pub fn half_size(&self) -> f32 {
        match self {
            Self::Pcf2x2 => 0.5,
            Self::Pcf3x3 => 1.0,
            Self::Pcf5x5 => 2.0,
            Self::Pcf7x7 => 3.0,
            Self::PoissonDisk { .. } | Self::RotatedPoissonDisk { .. } => 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// PCF Settings
// ---------------------------------------------------------------------------

/// Configuration for PCF shadow filtering.
#[derive(Debug, Clone)]
pub struct PcfSettings {
    /// Which kernel to use.
    pub kernel: PcfKernel,
    /// Shadow softness (scales the filter radius in texels).
    pub softness: f32,
    /// Depth bias.
    pub depth_bias: f32,
    /// Normal-offset bias.
    pub normal_bias: f32,
    /// Whether to use receiver-plane depth bias.
    pub receiver_plane_bias: bool,
}

impl Default for PcfSettings {
    fn default() -> Self {
        Self {
            kernel: PcfKernel::Pcf3x3,
            softness: 1.0,
            depth_bias: 0.005,
            normal_bias: 0.02,
            receiver_plane_bias: false,
        }
    }
}

impl PcfSettings {
    /// High-quality settings with rotated Poisson disk.
    pub fn high_quality() -> Self {
        Self {
            kernel: PcfKernel::RotatedPoissonDisk { samples: 32 },
            softness: 1.5,
            depth_bias: 0.003,
            normal_bias: 0.015,
            receiver_plane_bias: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Poisson disk sample points
// ---------------------------------------------------------------------------

/// Pre-computed 16-sample Poisson disk distribution.
///
/// Points are in the unit disk [-1, 1]^2, distributed to minimise
/// clustering and provide good coverage for PCF.
pub const POISSON_DISK_16: [Vec2; 16] = [
    Vec2::new(-0.94201624, -0.39906216),
    Vec2::new(0.94558609, -0.76890725),
    Vec2::new(-0.09418410, -0.92938870),
    Vec2::new(0.34495938, 0.29387760),
    Vec2::new(-0.91588581, 0.45771432),
    Vec2::new(-0.81544232, -0.87912464),
    Vec2::new(-0.38277543, 0.27676845),
    Vec2::new(0.97484398, 0.75648379),
    Vec2::new(0.44323325, -0.97511554),
    Vec2::new(0.53742981, -0.47373420),
    Vec2::new(-0.26496911, -0.41893023),
    Vec2::new(0.79197514, 0.19090188),
    Vec2::new(-0.24188840, 0.99706507),
    Vec2::new(-0.81409955, 0.91437590),
    Vec2::new(0.19984126, 0.78641367),
    Vec2::new(0.14383161, -0.14100790),
];

/// Pre-computed 32-sample Poisson disk distribution.
pub const POISSON_DISK_32: [Vec2; 32] = [
    Vec2::new(-0.613392, 0.617481),
    Vec2::new(0.170019, -0.040254),
    Vec2::new(-0.299417, 0.791925),
    Vec2::new(0.645680, 0.493210),
    Vec2::new(-0.651784, 0.717887),
    Vec2::new(0.421003, 0.027070),
    Vec2::new(-0.817194, -0.271096),
    Vec2::new(-0.705374, -0.668203),
    Vec2::new(0.977050, -0.108615),
    Vec2::new(0.063326, 0.142369),
    Vec2::new(0.203528, 0.214331),
    Vec2::new(-0.667531, 0.326090),
    Vec2::new(-0.098422, -0.295755),
    Vec2::new(-0.885922, 0.215369),
    Vec2::new(0.566637, 0.605213),
    Vec2::new(0.039766, -0.396100),
    Vec2::new(0.751946, 0.453352),
    Vec2::new(0.078707, -0.715323),
    Vec2::new(-0.075838, -0.529344),
    Vec2::new(0.724479, -0.580798),
    Vec2::new(0.222999, -0.215125),
    Vec2::new(-0.467574, -0.405438),
    Vec2::new(-0.248268, -0.814753),
    Vec2::new(0.354411, -0.887570),
    Vec2::new(0.175817, 0.382366),
    Vec2::new(0.487472, -0.063082),
    Vec2::new(-0.084078, 0.898312),
    Vec2::new(0.488876, -0.783441),
    Vec2::new(0.470016, 0.217933),
    Vec2::new(-0.696890, -0.549791),
    Vec2::new(-0.149693, 0.605762),
    Vec2::new(0.034211, 0.979980),
];

/// Pre-computed 64-sample Poisson disk distribution.
pub fn poisson_disk_64() -> Vec<Vec2> {
    // Generate using a stratified jittered approach.
    let mut samples = Vec::with_capacity(64);
    let grid_size = 8u32;
    let cell_size = 2.0 / grid_size as f32;

    // Use a deterministic hash for jitter.
    for y in 0..grid_size {
        for x in 0..grid_size {
            let idx = y * grid_size + x;
            let jx = pseudo_random(idx * 2) * 0.8;
            let jy = pseudo_random(idx * 2 + 1) * 0.8;
            let px = -1.0 + (x as f32 + 0.5 + jx - 0.4) * cell_size;
            let py = -1.0 + (y as f32 + 0.5 + jy - 0.4) * cell_size;
            let p = Vec2::new(px.clamp(-1.0, 1.0), py.clamp(-1.0, 1.0));
            // Only keep points inside the unit disk.
            if p.length_squared() <= 1.0 {
                samples.push(p);
            } else {
                // Project onto disk.
                samples.push(p.normalize_or_zero() * 0.95);
            }
        }
    }
    samples
}

/// Simple deterministic pseudo-random [0,1) from an integer seed.
fn pseudo_random(seed: u32) -> f32 {
    let mut s = seed;
    s = s.wrapping_mul(747796405).wrapping_add(2891336453);
    s = ((s >> ((s >> 28).wrapping_add(4))) ^ s).wrapping_mul(277803737);
    s = (s >> 22) ^ s;
    s as f32 / u32::MAX as f32
}

// ---------------------------------------------------------------------------
// PCF sampling (CPU reference)
// ---------------------------------------------------------------------------

/// Generate uniform grid PCF sample offsets for an NxN kernel.
///
/// Returns offsets in texel space centered at origin.
pub fn generate_pcf_offsets(kernel_size: u32) -> Vec<Vec2> {
    let half = (kernel_size as f32 - 1.0) * 0.5;
    let mut offsets = Vec::with_capacity((kernel_size * kernel_size) as usize);
    for y in 0..kernel_size {
        for x in 0..kernel_size {
            offsets.push(Vec2::new(x as f32 - half, y as f32 - half));
        }
    }
    offsets
}

/// Generate Gaussian-weighted PCF sample weights for an NxN kernel.
///
/// The weights are normalised to sum to 1.0.
pub fn generate_pcf_weights(kernel_size: u32, sigma: f32) -> Vec<f32> {
    let half = (kernel_size as f32 - 1.0) * 0.5;
    let sigma2 = 2.0 * sigma * sigma;
    let mut weights = Vec::with_capacity((kernel_size * kernel_size) as usize);
    let mut total = 0.0f32;

    for y in 0..kernel_size {
        for x in 0..kernel_size {
            let dx = x as f32 - half;
            let dy = y as f32 - half;
            let w = (-(dx * dx + dy * dy) / sigma2).exp();
            weights.push(w);
            total += w;
        }
    }

    // Normalise.
    for w in &mut weights {
        *w /= total;
    }
    weights
}

/// Apply a per-pixel rotation to a set of Poisson disk samples.
///
/// Uses the pixel coordinates to generate a unique rotation angle,
/// reducing visible banding patterns in the shadow.
pub fn rotate_poisson_samples(samples: &[Vec2], pixel_x: u32, pixel_y: u32) -> Vec<Vec2> {
    let seed = pixel_x.wrapping_mul(1973).wrapping_add(pixel_y.wrapping_mul(9277));
    let angle = pseudo_random(seed) * 2.0 * PI;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    samples
        .iter()
        .map(|s| Vec2::new(s.x * cos_a - s.y * sin_a, s.x * sin_a + s.y * cos_a))
        .collect()
}

/// Evaluate PCF at a shadow map location (CPU reference implementation).
///
/// # Arguments
/// - `shadow_map` — shadow depth values (2D array, row-major).
/// - `map_width`, `map_height` — shadow map dimensions.
/// - `uv` — texture coordinates [0,1].
/// - `depth` — the reference depth to compare against.
/// - `kernel` — PCF kernel type.
/// - `softness` — filter radius scale.
///
/// # Returns
/// Shadow factor (0.0 = fully shadowed, 1.0 = fully lit).
pub fn evaluate_pcf(
    shadow_map: &[f32],
    map_width: u32,
    map_height: u32,
    uv: Vec2,
    depth: f32,
    kernel: &PcfKernel,
    softness: f32,
) -> f32 {
    match kernel {
        PcfKernel::Pcf2x2 | PcfKernel::Pcf3x3 | PcfKernel::Pcf5x5 | PcfKernel::Pcf7x7 => {
            let size = match kernel {
                PcfKernel::Pcf2x2 => 2,
                PcfKernel::Pcf3x3 => 3,
                PcfKernel::Pcf5x5 => 5,
                PcfKernel::Pcf7x7 => 7,
                _ => unreachable!(),
            };
            evaluate_pcf_uniform(shadow_map, map_width, map_height, uv, depth, size, softness)
        }
        PcfKernel::PoissonDisk { samples } => {
            let disk = if *samples <= 16 {
                POISSON_DISK_16[..(*samples as usize).min(16)].to_vec()
            } else {
                POISSON_DISK_32[..(*samples as usize).min(32)].to_vec()
            };
            evaluate_pcf_poisson(shadow_map, map_width, map_height, uv, depth, &disk, softness)
        }
        PcfKernel::RotatedPoissonDisk { samples } => {
            let disk = if *samples <= 16 {
                POISSON_DISK_16[..(*samples as usize).min(16)].to_vec()
            } else {
                POISSON_DISK_32[..(*samples as usize).min(32)].to_vec()
            };
            // Rotate based on UV for deterministic per-pixel rotation.
            let px = (uv.x * map_width as f32) as u32;
            let py = (uv.y * map_height as f32) as u32;
            let rotated = rotate_poisson_samples(&disk, px, py);
            evaluate_pcf_poisson(
                shadow_map,
                map_width,
                map_height,
                uv,
                depth,
                &rotated,
                softness,
            )
        }
    }
}

/// Uniform grid PCF.
fn evaluate_pcf_uniform(
    shadow_map: &[f32],
    map_width: u32,
    map_height: u32,
    uv: Vec2,
    depth: f32,
    kernel_size: u32,
    softness: f32,
) -> f32 {
    let texel_w = softness / map_width as f32;
    let texel_h = softness / map_height as f32;
    let half = (kernel_size as f32 - 1.0) * 0.5;
    let mut shadow = 0.0f32;
    let count = kernel_size * kernel_size;

    for ky in 0..kernel_size {
        for kx in 0..kernel_size {
            let offset_x = (kx as f32 - half) * texel_w;
            let offset_y = (ky as f32 - half) * texel_h;
            let sample_uv = Vec2::new(uv.x + offset_x, uv.y + offset_y);

            let sample_depth = sample_shadow_map(shadow_map, map_width, map_height, sample_uv);
            if depth <= sample_depth {
                shadow += 1.0;
            }
        }
    }

    shadow / count as f32
}

/// Poisson disk PCF.
fn evaluate_pcf_poisson(
    shadow_map: &[f32],
    map_width: u32,
    map_height: u32,
    uv: Vec2,
    depth: f32,
    samples: &[Vec2],
    softness: f32,
) -> f32 {
    let texel_w = softness / map_width as f32;
    let texel_h = softness / map_height as f32;
    let mut shadow = 0.0f32;

    for sample in samples {
        let sample_uv = Vec2::new(uv.x + sample.x * texel_w, uv.y + sample.y * texel_h);
        let sample_depth = sample_shadow_map(shadow_map, map_width, map_height, sample_uv);
        if depth <= sample_depth {
            shadow += 1.0;
        }
    }

    shadow / samples.len() as f32
}

/// Sample the shadow map at the given UV coordinates (nearest neighbour).
fn sample_shadow_map(map: &[f32], width: u32, height: u32, uv: Vec2) -> f32 {
    let x = (uv.x * width as f32).clamp(0.0, (width - 1) as f32) as u32;
    let y = (uv.y * height as f32).clamp(0.0, (height - 1) as f32) as u32;
    let idx = (y * width + x) as usize;
    map.get(idx).copied().unwrap_or(1.0)
}

// ---------------------------------------------------------------------------
// PCSS (Percentage Closer Soft Shadows)
// ---------------------------------------------------------------------------

/// Settings for PCSS.
#[derive(Debug, Clone)]
pub struct PcssSettings {
    /// Light size (in world units). Larger values produce softer shadows.
    pub light_size: f32,
    /// Number of blocker search samples.
    pub blocker_search_samples: u32,
    /// Number of PCF samples.
    pub pcf_samples: u32,
    /// Minimum filter radius (in texels).
    pub min_filter_radius: f32,
    /// Maximum filter radius (in texels).
    pub max_filter_radius: f32,
    /// Near plane of the light.
    pub light_near: f32,
}

impl Default for PcssSettings {
    fn default() -> Self {
        Self {
            light_size: 0.02,
            blocker_search_samples: 16,
            pcf_samples: 32,
            min_filter_radius: 1.0,
            max_filter_radius: 10.0,
            light_near: 0.1,
        }
    }
}

/// PCSS blocker search result.
#[derive(Debug, Clone, Copy)]
pub struct BlockerSearchResult {
    /// Average blocker depth.
    pub avg_depth: f32,
    /// Number of blocker samples found.
    pub blocker_count: u32,
    /// Whether any blockers were found.
    pub found: bool,
}

/// Perform the PCSS blocker search pass.
///
/// Searches the shadow map near the given UV for occluders closer than the
/// receiver and computes the average occluder depth.
///
/// # Arguments
/// - `shadow_map` — shadow depth values.
/// - `map_width`, `map_height` — shadow map dimensions.
/// - `uv` — shadow map UV coordinates of the receiver.
/// - `receiver_depth` — depth of the receiver in light space.
/// - `search_radius` — radius of the search area in texels.
/// - `sample_count` — number of samples for the search.
pub fn pcss_blocker_search(
    shadow_map: &[f32],
    map_width: u32,
    map_height: u32,
    uv: Vec2,
    receiver_depth: f32,
    search_radius: f32,
    sample_count: u32,
) -> BlockerSearchResult {
    let samples = if sample_count <= 16 {
        &POISSON_DISK_16[..sample_count as usize]
    } else {
        &POISSON_DISK_32[..(sample_count as usize).min(32)]
    };

    let texel_w = search_radius / map_width as f32;
    let texel_h = search_radius / map_height as f32;

    let mut blocker_sum = 0.0f32;
    let mut blocker_count = 0u32;

    for sample in samples {
        let sample_uv = Vec2::new(uv.x + sample.x * texel_w, uv.y + sample.y * texel_h);
        let sample_depth = sample_shadow_map(shadow_map, map_width, map_height, sample_uv);

        if sample_depth < receiver_depth {
            blocker_sum += sample_depth;
            blocker_count += 1;
        }
    }

    if blocker_count > 0 {
        BlockerSearchResult {
            avg_depth: blocker_sum / blocker_count as f32,
            blocker_count,
            found: true,
        }
    } else {
        BlockerSearchResult {
            avg_depth: 0.0,
            blocker_count: 0,
            found: false,
        }
    }
}

/// Estimate the penumbra size from blocker search results.
///
/// Uses the parallel planes assumption:
///   penumbra_width = light_size * (receiver_depth - blocker_depth) / blocker_depth
///
/// # Arguments
/// - `receiver_depth` — depth of the receiver.
/// - `blocker_depth` — average blocker depth.
/// - `light_size` — apparent size of the light source.
///
/// # Returns
/// The estimated penumbra width (in the same units as the shadow map).
pub fn estimate_penumbra_size(receiver_depth: f32, blocker_depth: f32, light_size: f32) -> f32 {
    let diff = (receiver_depth - blocker_depth).max(0.0);
    light_size * diff / blocker_depth.max(0.001)
}

/// Evaluate PCSS (complete pipeline: blocker search -> penumbra estimation -> PCF).
///
/// # Returns
/// Shadow factor (0.0 = fully shadowed, 1.0 = fully lit).
pub fn evaluate_pcss(
    shadow_map: &[f32],
    map_width: u32,
    map_height: u32,
    uv: Vec2,
    depth: f32,
    settings: &PcssSettings,
) -> f32 {
    // Step 1: Blocker search.
    let search_radius = settings.light_size * (depth - settings.light_near) / depth;
    let search_texels = (search_radius * map_width as f32)
        .clamp(settings.min_filter_radius, settings.max_filter_radius);

    let blocker =
        pcss_blocker_search(shadow_map, map_width, map_height, uv, depth, search_texels, settings.blocker_search_samples);

    if !blocker.found {
        // No blockers found -> fully lit.
        return 1.0;
    }

    // Step 2: Penumbra estimation.
    let penumbra = estimate_penumbra_size(depth, blocker.avg_depth, settings.light_size);
    let filter_radius = (penumbra * map_width as f32)
        .clamp(settings.min_filter_radius, settings.max_filter_radius);

    // Step 3: PCF with variable kernel size.
    let disk = if settings.pcf_samples <= 16 {
        POISSON_DISK_16[..(settings.pcf_samples as usize).min(16)].to_vec()
    } else {
        POISSON_DISK_32[..(settings.pcf_samples as usize).min(32)].to_vec()
    };

    evaluate_pcf_poisson(shadow_map, map_width, map_height, uv, depth, &disk, filter_radius)
}

// ---------------------------------------------------------------------------
// WGSL shader snippets for PCF/PCSS
// ---------------------------------------------------------------------------

/// Generate WGSL code for a PCF shadow sampling function.
pub fn generate_pcf_wgsl(kernel: &PcfKernel) -> String {
    match kernel {
        PcfKernel::Pcf3x3 => PCF_3X3_WGSL.to_string(),
        PcfKernel::Pcf5x5 => PCF_5X5_WGSL.to_string(),
        _ => PCF_3X3_WGSL.to_string(), // fallback
    }
}

const PCF_3X3_WGSL: &str = r#"
fn sample_shadow_pcf_3x3(shadow_map: texture_depth_2d_array, shadow_sampler: sampler_comparison, uv: vec2<f32>, layer: u32, depth: f32) -> f32 {
    let texel_size = 1.0 / f32(textureDimensions(shadow_map).x);
    var shadow = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, layer, depth);
        }
    }
    return shadow / 9.0;
}
"#;

const PCF_5X5_WGSL: &str = r#"
fn sample_shadow_pcf_5x5(shadow_map: texture_depth_2d_array, shadow_sampler: sampler_comparison, uv: vec2<f32>, layer: u32, depth: f32) -> f32 {
    let texel_size = 1.0 / f32(textureDimensions(shadow_map).x);
    var shadow = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, layer, depth);
        }
    }
    return shadow / 25.0;
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcf_kernel_sample_counts() {
        assert_eq!(PcfKernel::Pcf2x2.sample_count(), 4);
        assert_eq!(PcfKernel::Pcf3x3.sample_count(), 9);
        assert_eq!(PcfKernel::Pcf5x5.sample_count(), 25);
        assert_eq!(PcfKernel::Pcf7x7.sample_count(), 49);
    }

    #[test]
    fn uniform_pcf_offsets() {
        let offsets = generate_pcf_offsets(3);
        assert_eq!(offsets.len(), 9);
        // Center should be (0, 0).
        assert!((offsets[4].x).abs() < 0.01);
        assert!((offsets[4].y).abs() < 0.01);
    }

    #[test]
    fn gaussian_weights_sum_to_one() {
        let weights = generate_pcf_weights(5, 1.5);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn poisson_disk_points_in_unit_disk() {
        for p in &POISSON_DISK_16 {
            assert!(p.length() <= 1.5, "Point {p:?} outside unit disk");
        }
    }

    #[test]
    fn pcf_fully_lit() {
        // Shadow map with all 1.0 (far depth) should be fully lit.
        let map = vec![1.0f32; 64 * 64];
        let shadow = evaluate_pcf(
            &map,
            64,
            64,
            Vec2::new(0.5, 0.5),
            0.5,
            &PcfKernel::Pcf3x3,
            1.0,
        );
        assert!((shadow - 1.0).abs() < 0.01);
    }

    #[test]
    fn pcf_fully_shadowed() {
        // Shadow map with all 0.0 (close depth) should be fully shadowed.
        let map = vec![0.0f32; 64 * 64];
        let shadow = evaluate_pcf(
            &map,
            64,
            64,
            Vec2::new(0.5, 0.5),
            0.5,
            &PcfKernel::Pcf3x3,
            1.0,
        );
        assert!(shadow.abs() < 0.01);
    }

    #[test]
    fn blocker_search_no_blockers() {
        let map = vec![1.0f32; 64 * 64];
        let result = pcss_blocker_search(&map, 64, 64, Vec2::new(0.5, 0.5), 0.5, 3.0, 16);
        assert!(!result.found);
    }

    #[test]
    fn blocker_search_all_blockers() {
        let map = vec![0.1f32; 64 * 64];
        let result = pcss_blocker_search(&map, 64, 64, Vec2::new(0.5, 0.5), 0.5, 3.0, 16);
        assert!(result.found);
        assert!((result.avg_depth - 0.1).abs() < 0.01);
    }

    #[test]
    fn penumbra_size_increases_with_distance() {
        let p1 = estimate_penumbra_size(0.5, 0.4, 0.1);
        let p2 = estimate_penumbra_size(0.8, 0.4, 0.1);
        assert!(p2 > p1);
    }

    #[test]
    fn rotated_samples_differ() {
        let r1 = rotate_poisson_samples(&POISSON_DISK_16[..4], 10, 20);
        let r2 = rotate_poisson_samples(&POISSON_DISK_16[..4], 30, 40);
        // Different pixels should produce different rotations.
        assert!((r1[0].x - r2[0].x).abs() > 0.001 || (r1[0].y - r2[0].y).abs() > 0.001);
    }
}
