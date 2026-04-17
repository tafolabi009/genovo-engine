// engine/render/src/shadow_system_v2.rs
//
// Enhanced shadow system for the Genovo engine.
//
// Features:
// - Contact shadows (screen-space ray march from light direction)
// - Area light shadows (soft penumbra based on light size)
// - Shadow bias auto-tuning (based on surface angle and depth)
// - Shadow cache (static shadow maps that don't need re-rendering every frame)
// - Shadow importance scoring (skip shadows for distant or small lights)
//
// This module supplements the existing `shadows` module with more advanced
// shadow techniques for higher visual fidelity.

use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;

/// Default contact shadow ray march steps.
pub const DEFAULT_CONTACT_SHADOW_STEPS: u32 = 16;

/// Default contact shadow max distance (screen-space fraction).
pub const DEFAULT_CONTACT_SHADOW_MAX_DIST: f32 = 0.1;

/// Default contact shadow thickness.
pub const DEFAULT_CONTACT_SHADOW_THICKNESS: f32 = 0.01;

/// Maximum number of cached shadow maps.
pub const MAX_CACHED_SHADOW_MAPS: usize = 64;

/// Shadow importance threshold below which shadows are skipped.
pub const DEFAULT_IMPORTANCE_THRESHOLD: f32 = 0.01;

// ---------------------------------------------------------------------------
// Contact shadows
// ---------------------------------------------------------------------------

/// Configuration for screen-space contact shadows.
///
/// Contact shadows provide small-scale shadowing that cascaded shadow maps
/// miss due to resolution limitations. They work by ray-marching in screen
/// space from each pixel toward the light source, checking for occlusion
/// against the depth buffer.
#[derive(Debug, Clone)]
pub struct ContactShadowConfig {
    /// Whether contact shadows are enabled.
    pub enabled: bool,
    /// Number of ray march steps.
    pub step_count: u32,
    /// Maximum ray distance in screen-space (0-1).
    pub max_distance: f32,
    /// Thickness threshold for occlusion detection.
    pub thickness: f32,
    /// Intensity of the contact shadow (0 = none, 1 = full).
    pub intensity: f32,
    /// Whether to use dithered starting offset per pixel.
    pub dithered: bool,
    /// Whether to fade out contact shadows at the ray end.
    pub fade_at_end: bool,
    /// Fade distance at the end of the ray (fraction of max_distance).
    pub end_fade: f32,
    /// Maximum depth difference for a valid occlusion (in view-space units).
    pub max_depth_diff: f32,
    /// Whether to apply only to small-detail shadows (not large occluders).
    pub detail_only: bool,
    /// Length bias to avoid self-shadowing at the ray origin.
    pub start_bias: f32,
}

impl Default for ContactShadowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            step_count: DEFAULT_CONTACT_SHADOW_STEPS,
            max_distance: DEFAULT_CONTACT_SHADOW_MAX_DIST,
            thickness: DEFAULT_CONTACT_SHADOW_THICKNESS,
            intensity: 0.8,
            dithered: true,
            fade_at_end: true,
            end_fade: 0.2,
            max_depth_diff: 0.5,
            detail_only: false,
            start_bias: 0.002,
        }
    }
}

/// Contact shadow calculator.
#[derive(Debug)]
pub struct ContactShadowSystem {
    /// Configuration.
    pub config: ContactShadowConfig,
    /// Resolution of the depth buffer.
    pub resolution: UVec2,
    /// View matrix for the current frame.
    pub view: Mat4,
    /// Projection matrix for the current frame.
    pub projection: Mat4,
    /// Statistics.
    pub stats: ContactShadowStats,
}

/// Statistics for contact shadows.
#[derive(Debug, Clone, Default)]
pub struct ContactShadowStats {
    /// Number of pixels processed.
    pub pixels_processed: u64,
    /// Number of pixels that found occlusion.
    pub occluded_pixels: u64,
    /// Total ray march steps taken.
    pub total_steps: u64,
    /// Processing time in microseconds.
    pub time_us: f64,
}

impl ContactShadowSystem {
    /// Create a new contact shadow system.
    pub fn new(config: ContactShadowConfig, width: u32, height: u32) -> Self {
        Self {
            config,
            resolution: UVec2::new(width, height),
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
            stats: ContactShadowStats::default(),
        }
    }

    /// Update matrices for the current frame.
    pub fn update_matrices(&mut self, view: Mat4, projection: Mat4) {
        self.view = view;
        self.projection = projection;
    }

    /// Resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.resolution = UVec2::new(width, height);
    }

    /// Compute contact shadow for a single pixel (CPU reference).
    ///
    /// Returns 1.0 for fully lit, 0.0 for fully shadowed.
    pub fn compute_pixel(
        &self,
        uv: Vec2,
        view_pos: Vec3,
        light_dir_view: Vec3,
        depth_buffer: &dyn Fn(Vec2) -> f32,
    ) -> f32 {
        if !self.config.enabled {
            return 1.0;
        }

        let light_step = light_dir_view.normalize_or_zero() * self.config.max_distance
            / self.config.step_count as f32;

        let mut sample_pos = view_pos + light_dir_view.normalize_or_zero() * self.config.start_bias;
        let mut occlusion = 0.0f32;

        for step in 0..self.config.step_count {
            sample_pos += light_step;

            // Project to screen space
            let clip = self.projection * Vec4::new(sample_pos.x, sample_pos.y, sample_pos.z, 1.0);
            if clip.w <= EPSILON {
                break;
            }
            let ndc = clip / clip.w;
            let sample_uv = Vec2::new((ndc.x + 1.0) * 0.5, (1.0 - ndc.y) * 0.5);

            if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
                break;
            }

            let buffer_depth = depth_buffer(sample_uv);
            let sample_depth = ndc.z;
            let depth_diff = sample_depth - buffer_depth;

            // Check for occlusion
            if depth_diff > 0.0 && depth_diff < self.config.thickness {
                if !self.config.detail_only || depth_diff < self.config.max_depth_diff {
                    let mut shadow = self.config.intensity;

                    // Fade at end of ray
                    if self.config.fade_at_end {
                        let t = step as f32 / self.config.step_count as f32;
                        let fade_start = 1.0 - self.config.end_fade;
                        if t > fade_start {
                            shadow *= 1.0 - (t - fade_start) / self.config.end_fade;
                        }
                    }

                    occlusion = occlusion.max(shadow);
                    break; // Found occlusion, stop marching
                }
            }
        }

        1.0 - occlusion
    }
}

// ---------------------------------------------------------------------------
// Area light shadows
// ---------------------------------------------------------------------------

/// Configuration for area light soft shadows.
#[derive(Debug, Clone)]
pub struct AreaLightShadowConfig {
    /// Whether area light shadows are enabled.
    pub enabled: bool,
    /// Number of samples for penumbra estimation.
    pub sample_count: u32,
    /// Search radius for blocker estimation (in shadow map texels).
    pub search_radius: f32,
    /// Minimum penumbra width (prevents zero-width shadows).
    pub min_penumbra: f32,
    /// Maximum penumbra width.
    pub max_penumbra: f32,
    /// Light size in world units (controls shadow softness).
    pub light_size: f32,
    /// Near plane of the shadow map.
    pub shadow_near: f32,
    /// Whether to use Percentage Closer Soft Shadows (PCSS).
    pub use_pcss: bool,
    /// Whether to use Variance Shadow Maps (VSM).
    pub use_vsm: bool,
    /// Light bleeding reduction for VSM.
    pub vsm_light_bleed_reduction: f32,
}

impl Default for AreaLightShadowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_count: 16,
            search_radius: 8.0,
            min_penumbra: 0.5,
            max_penumbra: 20.0,
            light_size: 1.0,
            shadow_near: 0.1,
            use_pcss: true,
            use_vsm: false,
            vsm_light_bleed_reduction: 0.2,
        }
    }
}

/// PCSS shadow sample result.
#[derive(Debug, Clone, Copy)]
pub struct PcssSampleResult {
    /// Shadow factor (0 = fully shadowed, 1 = fully lit).
    pub shadow: f32,
    /// Average blocker depth.
    pub avg_blocker_depth: f32,
    /// Estimated penumbra width in texels.
    pub penumbra_width: f32,
    /// Number of blocking samples found.
    pub blocker_count: u32,
}

/// Area light shadow calculator.
#[derive(Debug)]
pub struct AreaLightShadowSystem {
    /// Configuration.
    pub config: AreaLightShadowConfig,
    /// Poisson disc samples for shadow filtering.
    pub poisson_samples: Vec<Vec2>,
}

impl AreaLightShadowSystem {
    /// Create a new area light shadow system.
    pub fn new(config: AreaLightShadowConfig) -> Self {
        let poisson_samples = Self::generate_poisson_disc(config.sample_count as usize);
        Self {
            config,
            poisson_samples,
        }
    }

    /// Generate a Poisson disc sample pattern.
    fn generate_poisson_disc(count: usize) -> Vec<Vec2> {
        let mut samples = Vec::with_capacity(count);
        let golden_angle = PI * (3.0 - 5.0f32.sqrt());
        for i in 0..count {
            let r = (i as f32 / count as f32).sqrt();
            let theta = i as f32 * golden_angle;
            samples.push(Vec2::new(r * theta.cos(), r * theta.sin()));
        }
        samples
    }

    /// Estimate the blocker depth for PCSS.
    pub fn estimate_blocker_depth(
        &self,
        shadow_uv: Vec2,
        receiver_depth: f32,
        shadow_map: &dyn Fn(Vec2) -> f32,
        texel_size: f32,
    ) -> (f32, u32) {
        let search_radius = self.config.search_radius * texel_size;
        let mut total_depth = 0.0f32;
        let mut count = 0u32;

        for sample in &self.poisson_samples {
            let sample_uv = shadow_uv + *sample * search_radius;
            let sample_depth = shadow_map(sample_uv);

            if sample_depth < receiver_depth {
                total_depth += sample_depth;
                count += 1;
            }
        }

        if count > 0 {
            (total_depth / count as f32, count)
        } else {
            (0.0, 0)
        }
    }

    /// Compute the penumbra width from blocker depth.
    pub fn compute_penumbra_width(
        &self,
        receiver_depth: f32,
        blocker_depth: f32,
    ) -> f32 {
        if blocker_depth < EPSILON {
            return self.config.min_penumbra;
        }

        let penumbra = self.config.light_size * (receiver_depth - blocker_depth)
            / (blocker_depth + EPSILON);

        penumbra.clamp(self.config.min_penumbra, self.config.max_penumbra)
    }

    /// Compute PCSS shadow for a point.
    pub fn compute_pcss(
        &self,
        shadow_uv: Vec2,
        receiver_depth: f32,
        shadow_map: &dyn Fn(Vec2) -> f32,
        texel_size: f32,
    ) -> PcssSampleResult {
        // Step 1: Blocker search
        let (avg_blocker_depth, blocker_count) = self.estimate_blocker_depth(
            shadow_uv, receiver_depth, shadow_map, texel_size,
        );

        if blocker_count == 0 {
            return PcssSampleResult {
                shadow: 1.0,
                avg_blocker_depth: 0.0,
                penumbra_width: 0.0,
                blocker_count: 0,
            };
        }

        // Step 2: Penumbra estimation
        let penumbra_width = self.compute_penumbra_width(receiver_depth, avg_blocker_depth);

        // Step 3: Filtering
        let filter_radius = penumbra_width * texel_size;
        let mut lit_count = 0u32;
        let total = self.poisson_samples.len() as u32;

        for sample in &self.poisson_samples {
            let sample_uv = shadow_uv + *sample * filter_radius;
            let sample_depth = shadow_map(sample_uv);

            if sample_depth >= receiver_depth {
                lit_count += 1;
            }
        }

        PcssSampleResult {
            shadow: lit_count as f32 / total as f32,
            avg_blocker_depth,
            penumbra_width,
            blocker_count,
        }
    }

    /// Compute Variance Shadow Map shadow.
    pub fn compute_vsm(&self, moments: Vec2, receiver_depth: f32) -> f32 {
        if receiver_depth <= moments.x {
            return 1.0;
        }

        let variance = moments.y - moments.x * moments.x;
        let variance = variance.max(0.0001);

        let d = receiver_depth - moments.x;
        let p_max = variance / (variance + d * d);

        // Light bleeding reduction
        let reduction = self.config.vsm_light_bleed_reduction;
        let p = ((p_max - reduction) / (1.0 - reduction)).clamp(0.0, 1.0);

        p
    }
}

// ---------------------------------------------------------------------------
// Shadow bias auto-tuning
// ---------------------------------------------------------------------------

/// Configuration for automatic shadow bias tuning.
#[derive(Debug, Clone)]
pub struct ShadowBiasConfig {
    /// Whether auto-tuning is enabled.
    pub enabled: bool,
    /// Base constant bias.
    pub constant_bias: f32,
    /// Slope-based bias scale.
    pub slope_bias_scale: f32,
    /// Maximum allowed bias.
    pub max_bias: f32,
    /// Minimum allowed bias.
    pub min_bias: f32,
    /// Normal offset bias in world units.
    pub normal_offset: f32,
    /// Whether to use receiver-plane depth bias.
    pub receiver_plane_bias: bool,
    /// Whether to adapt bias based on shadow map resolution.
    pub resolution_adaptive: bool,
    /// Reference shadow map resolution for bias scaling.
    pub reference_resolution: u32,
}

impl Default for ShadowBiasConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            constant_bias: 0.0005,
            slope_bias_scale: 2.0,
            max_bias: 0.01,
            min_bias: 0.0001,
            normal_offset: 0.02,
            receiver_plane_bias: true,
            resolution_adaptive: true,
            reference_resolution: 2048,
        }
    }
}

/// Shadow bias calculator.
#[derive(Debug)]
pub struct ShadowBiasCalculator {
    pub config: ShadowBiasConfig,
}

impl ShadowBiasCalculator {
    pub fn new(config: ShadowBiasConfig) -> Self {
        Self { config }
    }

    /// Compute the optimal bias for a surface.
    ///
    /// Parameters:
    /// - `surface_normal`: world-space surface normal.
    /// - `light_dir`: world-space direction toward the light.
    /// - `shadow_map_resolution`: resolution of the shadow map.
    /// - `shadow_map_depth_range`: depth range of the shadow map (far - near).
    pub fn compute_bias(
        &self,
        surface_normal: Vec3,
        light_dir: Vec3,
        shadow_map_resolution: u32,
        shadow_map_depth_range: f32,
    ) -> ShadowBias {
        if !self.config.enabled {
            return ShadowBias {
                depth_bias: self.config.constant_bias,
                normal_offset: self.config.normal_offset,
                slope_bias: 0.0,
            };
        }

        let n_dot_l = surface_normal.dot(light_dir).max(0.0);
        let sin_angle = (1.0 - n_dot_l * n_dot_l).sqrt();
        let tan_angle = if n_dot_l > EPSILON {
            sin_angle / n_dot_l
        } else {
            100.0
        };

        // Slope-based bias
        let slope_bias = tan_angle * self.config.slope_bias_scale;

        // Resolution-adaptive scaling
        let resolution_scale = if self.config.resolution_adaptive {
            self.config.reference_resolution as f32 / shadow_map_resolution as f32
        } else {
            1.0
        };

        // Depth range scaling
        let depth_scale = shadow_map_depth_range / 1000.0;

        let depth_bias = (self.config.constant_bias + slope_bias * 0.001)
            * resolution_scale
            * depth_scale;
        let depth_bias = depth_bias.clamp(self.config.min_bias, self.config.max_bias);

        // Normal offset (push surface along normal to reduce shadow acne)
        let normal_offset = self.config.normal_offset * (1.0 - n_dot_l) * resolution_scale;

        ShadowBias {
            depth_bias,
            normal_offset,
            slope_bias: slope_bias * 0.0001 * resolution_scale,
        }
    }
}

/// Computed shadow bias values.
#[derive(Debug, Clone, Copy)]
pub struct ShadowBias {
    /// Depth bias to add to the shadow comparison depth.
    pub depth_bias: f32,
    /// Normal offset to push the surface point along its normal.
    pub normal_offset: f32,
    /// Slope-dependent bias.
    pub slope_bias: f32,
}

impl ShadowBias {
    /// Apply the bias to a surface position for shadow map lookup.
    pub fn apply(&self, position: Vec3, normal: Vec3, light_dir: Vec3) -> Vec3 {
        position + normal * self.normal_offset
    }

    /// Apply the depth bias to a shadow map depth comparison.
    pub fn apply_depth(&self, depth: f32) -> f32 {
        depth - self.depth_bias - self.slope_bias
    }
}

// ---------------------------------------------------------------------------
// Shadow cache
// ---------------------------------------------------------------------------

/// Unique key for a cached shadow map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShadowCacheKey {
    /// Light ID.
    pub light_id: u64,
    /// Cascade index (for directional lights).
    pub cascade: u32,
    /// Face index (for point lights / cube maps).
    pub face: u32,
}

/// State of a cached shadow map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowCacheState {
    /// The cached shadow map is fresh and valid.
    Valid,
    /// The cached shadow map needs an update (some objects moved).
    NeedsPartialUpdate,
    /// The cached shadow map is invalid (needs full re-render).
    Invalid,
    /// The shadow map is being rendered this frame.
    Rendering,
}

/// Entry in the shadow cache.
#[derive(Debug, Clone)]
pub struct ShadowCacheEntry {
    /// Cache key.
    pub key: ShadowCacheKey,
    /// State of this entry.
    pub state: ShadowCacheState,
    /// Shadow map texture handle.
    pub texture_handle: u64,
    /// Shadow map resolution.
    pub resolution: u32,
    /// View-projection matrix used to render this shadow map.
    pub view_projection: Mat4,
    /// Frame number when this entry was last rendered.
    pub last_render_frame: u64,
    /// Frame number when this entry was last used.
    pub last_used_frame: u64,
    /// Number of dynamic objects in this shadow map's frustum.
    pub dynamic_object_count: u32,
    /// Whether this shadow map contains only static geometry.
    pub static_only: bool,
    /// Priority score for this entry (higher = more important).
    pub priority: f32,
}

/// Shadow map cache manager.
#[derive(Debug)]
pub struct ShadowCache {
    /// All cached entries.
    pub entries: HashMap<ShadowCacheKey, ShadowCacheEntry>,
    /// Maximum number of cached shadow maps.
    pub max_entries: usize,
    /// Current frame number.
    pub current_frame: u64,
    /// Number of shadow maps rendered this frame.
    pub renders_this_frame: u32,
    /// Maximum shadow map renders per frame (budget).
    pub max_renders_per_frame: u32,
    /// Statistics.
    pub stats: ShadowCacheStats,
}

/// Shadow cache statistics.
#[derive(Debug, Clone, Default)]
pub struct ShadowCacheStats {
    /// Total entries in the cache.
    pub total_entries: u32,
    /// Valid entries.
    pub valid_entries: u32,
    /// Cache hits this frame (used a cached shadow map).
    pub cache_hits: u32,
    /// Cache misses this frame (had to render a new shadow map).
    pub cache_misses: u32,
    /// Entries evicted this frame.
    pub evictions: u32,
    /// Total renders this frame.
    pub renders: u32,
}

impl ShadowCache {
    /// Create a new shadow cache.
    pub fn new(max_entries: usize, max_renders_per_frame: u32) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            current_frame: 0,
            renders_this_frame: 0,
            max_renders_per_frame,
            stats: ShadowCacheStats::default(),
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.current_frame += 1;
        self.renders_this_frame = 0;
        self.stats = ShadowCacheStats {
            total_entries: self.entries.len() as u32,
            ..Default::default()
        };
    }

    /// Request a shadow map, returning whether it needs rendering.
    pub fn request(&mut self, key: ShadowCacheKey) -> ShadowCacheState {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used_frame = self.current_frame;

            match entry.state {
                ShadowCacheState::Valid if entry.static_only => {
                    self.stats.cache_hits += 1;
                    ShadowCacheState::Valid
                }
                ShadowCacheState::Valid => {
                    // Dynamic objects may have moved -- check if update needed
                    if entry.dynamic_object_count > 0 {
                        entry.state = ShadowCacheState::NeedsPartialUpdate;
                        self.stats.cache_misses += 1;
                        ShadowCacheState::NeedsPartialUpdate
                    } else {
                        self.stats.cache_hits += 1;
                        ShadowCacheState::Valid
                    }
                }
                state => {
                    self.stats.cache_misses += 1;
                    state
                }
            }
        } else {
            self.stats.cache_misses += 1;
            ShadowCacheState::Invalid
        }
    }

    /// Register a newly rendered shadow map.
    pub fn insert(&mut self, entry: ShadowCacheEntry) {
        let key = entry.key;

        // Evict if at capacity
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            self.evict_lru();
        }

        self.entries.insert(key, entry);
        self.renders_this_frame += 1;
        self.stats.renders += 1;
    }

    /// Invalidate a specific entry (e.g. when an object in its frustum moves).
    pub fn invalidate(&mut self, key: &ShadowCacheKey) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.state = ShadowCacheState::Invalid;
        }
    }

    /// Invalidate all entries for a given light.
    pub fn invalidate_light(&mut self, light_id: u64) {
        for entry in self.entries.values_mut() {
            if entry.key.light_id == light_id {
                entry.state = ShadowCacheState::Invalid;
            }
        }
    }

    /// Invalidate all entries (e.g. on camera cut).
    pub fn invalidate_all(&mut self) {
        for entry in self.entries.values_mut() {
            entry.state = ShadowCacheState::Invalid;
        }
    }

    /// Whether the render budget allows another shadow map this frame.
    pub fn can_render(&self) -> bool {
        self.renders_this_frame < self.max_renders_per_frame
    }

    /// Evict the least-recently-used entry.
    fn evict_lru(&mut self) {
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_used_frame)
            .map(|(k, _)| *k);

        if let Some(key) = lru_key {
            self.entries.remove(&key);
            self.stats.evictions += 1;
        }
    }

    /// Remove entries not used for N frames.
    pub fn cleanup_stale(&mut self, max_age_frames: u64) {
        let threshold = self.current_frame.saturating_sub(max_age_frames);
        self.entries.retain(|_, e| e.last_used_frame >= threshold);
        self.stats.total_entries = self.entries.len() as u32;
    }

    /// Get the total memory usage estimate (based on resolution).
    pub fn estimated_memory_bytes(&self) -> u64 {
        self.entries
            .values()
            .map(|e| (e.resolution as u64 * e.resolution as u64 * 4)) // 32-bit depth
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Shadow importance scoring
// ---------------------------------------------------------------------------

/// Configuration for shadow importance scoring.
#[derive(Debug, Clone)]
pub struct ShadowImportanceConfig {
    /// Whether importance scoring is enabled.
    pub enabled: bool,
    /// Threshold below which shadows are skipped.
    pub threshold: f32,
    /// Weight for light intensity in the importance score.
    pub intensity_weight: f32,
    /// Weight for screen coverage in the importance score.
    pub coverage_weight: f32,
    /// Weight for distance from camera in the importance score.
    pub distance_weight: f32,
    /// Weight for light size (larger lights have more visible shadows).
    pub size_weight: f32,
    /// Whether to use a gradual quality reduction instead of hard cutoff.
    pub gradual_quality: bool,
    /// Minimum shadow quality level.
    pub min_quality: ShadowQualityLevel,
}

/// Shadow quality level (for gradual quality reduction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShadowQualityLevel {
    /// No shadow.
    None,
    /// Low quality (small shadow map, no filtering).
    Low,
    /// Medium quality (standard shadow map, PCF).
    Medium,
    /// High quality (large shadow map, PCSS).
    High,
    /// Ultra quality (cached, PCSS, contact shadows).
    Ultra,
}

impl Default for ShadowImportanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: DEFAULT_IMPORTANCE_THRESHOLD,
            intensity_weight: 1.0,
            coverage_weight: 1.5,
            distance_weight: 1.0,
            size_weight: 0.5,
            gradual_quality: true,
            min_quality: ShadowQualityLevel::Low,
        }
    }
}

/// Shadow importance scorer.
#[derive(Debug)]
pub struct ShadowImportanceScorer {
    pub config: ShadowImportanceConfig,
}

impl ShadowImportanceScorer {
    pub fn new(config: ShadowImportanceConfig) -> Self {
        Self { config }
    }

    /// Compute the importance score for a light.
    pub fn score(
        &self,
        light_intensity: f32,
        light_radius: f32,
        distance_to_camera: f32,
        screen_coverage: f32,
    ) -> f32 {
        if !self.config.enabled {
            return 1.0;
        }

        let intensity_score = (light_intensity / 100.0).min(1.0) * self.config.intensity_weight;
        let coverage_score = screen_coverage * self.config.coverage_weight;
        let distance_score = (1.0 / (1.0 + distance_to_camera * 0.01)) * self.config.distance_weight;
        let size_score = (light_radius / 10.0).min(1.0) * self.config.size_weight;

        let total_weight = self.config.intensity_weight
            + self.config.coverage_weight
            + self.config.distance_weight
            + self.config.size_weight;

        let score = (intensity_score + coverage_score + distance_score + size_score)
            / (total_weight + EPSILON);

        score
    }

    /// Determine the shadow quality level from an importance score.
    pub fn quality_from_score(&self, score: f32) -> ShadowQualityLevel {
        if !self.config.gradual_quality {
            return if score >= self.config.threshold {
                ShadowQualityLevel::High
            } else {
                ShadowQualityLevel::None
            };
        }

        if score < self.config.threshold * 0.5 {
            ShadowQualityLevel::None
        } else if score < self.config.threshold {
            self.config.min_quality
        } else if score < self.config.threshold * 3.0 {
            ShadowQualityLevel::Low
        } else if score < self.config.threshold * 10.0 {
            ShadowQualityLevel::Medium
        } else if score < self.config.threshold * 30.0 {
            ShadowQualityLevel::High
        } else {
            ShadowQualityLevel::Ultra
        }
    }

    /// Get the recommended shadow map resolution for a quality level.
    pub fn resolution_for_quality(quality: ShadowQualityLevel) -> u32 {
        match quality {
            ShadowQualityLevel::None => 0,
            ShadowQualityLevel::Low => 256,
            ShadowQualityLevel::Medium => 512,
            ShadowQualityLevel::High => 1024,
            ShadowQualityLevel::Ultra => 2048,
        }
    }
}

// ---------------------------------------------------------------------------
// Enhanced shadow system (combines all features)
// ---------------------------------------------------------------------------

/// The main enhanced shadow system component.
#[derive(Debug)]
pub struct ShadowSystemV2 {
    /// Contact shadow system.
    pub contact_shadows: ContactShadowSystem,
    /// Area light shadow system.
    pub area_shadows: AreaLightShadowSystem,
    /// Shadow bias calculator.
    pub bias_calculator: ShadowBiasCalculator,
    /// Shadow cache.
    pub cache: ShadowCache,
    /// Shadow importance scorer.
    pub importance: ShadowImportanceScorer,
    /// Frame index.
    pub frame_index: u64,
    /// Statistics.
    pub stats: ShadowSystemV2Stats,
}

/// Combined statistics.
#[derive(Debug, Clone, Default)]
pub struct ShadowSystemV2Stats {
    /// Total lights evaluated.
    pub lights_evaluated: u32,
    /// Lights with shadows rendered.
    pub lights_with_shadows: u32,
    /// Lights skipped due to low importance.
    pub lights_skipped: u32,
    /// Shadow maps rendered this frame.
    pub shadow_maps_rendered: u32,
    /// Shadow maps from cache.
    pub shadow_maps_cached: u32,
    /// Contact shadow pixels processed.
    pub contact_shadow_pixels: u64,
    /// Total shadow memory in bytes.
    pub total_shadow_memory: u64,
}

impl ShadowSystemV2 {
    /// Create a new enhanced shadow system.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            contact_shadows: ContactShadowSystem::new(ContactShadowConfig::default(), width, height),
            area_shadows: AreaLightShadowSystem::new(AreaLightShadowConfig::default()),
            bias_calculator: ShadowBiasCalculator::new(ShadowBiasConfig::default()),
            cache: ShadowCache::new(MAX_CACHED_SHADOW_MAPS, 4),
            importance: ShadowImportanceScorer::new(ShadowImportanceConfig::default()),
            frame_index: 0,
            stats: ShadowSystemV2Stats::default(),
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self, view: Mat4, projection: Mat4) {
        self.frame_index += 1;
        self.contact_shadows.update_matrices(view, projection);
        self.cache.begin_frame();
        self.stats = ShadowSystemV2Stats::default();
    }

    /// Evaluate whether a light should have shadows.
    pub fn evaluate_light(
        &mut self,
        light_intensity: f32,
        light_radius: f32,
        distance_to_camera: f32,
        screen_coverage: f32,
    ) -> ShadowQualityLevel {
        self.stats.lights_evaluated += 1;
        let score = self.importance.score(
            light_intensity,
            light_radius,
            distance_to_camera,
            screen_coverage,
        );
        let quality = self.importance.quality_from_score(score);
        if quality == ShadowQualityLevel::None {
            self.stats.lights_skipped += 1;
        } else {
            self.stats.lights_with_shadows += 1;
        }
        quality
    }

    /// Resize screen-space effects.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.contact_shadows.resize(width, height);
    }

    /// Compute the auto-tuned bias for a surface.
    pub fn compute_bias(
        &self,
        surface_normal: Vec3,
        light_dir: Vec3,
        shadow_map_resolution: u32,
        depth_range: f32,
    ) -> ShadowBias {
        self.bias_calculator.compute_bias(
            surface_normal,
            light_dir,
            shadow_map_resolution,
            depth_range,
        )
    }

    /// Cleanup stale cache entries.
    pub fn cleanup(&mut self) {
        self.cache.cleanup_stale(120); // Remove entries not used for 120 frames
        self.stats.total_shadow_memory = self.cache.estimated_memory_bytes();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_shadow_disabled() {
        let sys = ContactShadowSystem::new(
            ContactShadowConfig { enabled: false, ..Default::default() },
            800, 600,
        );
        let result = sys.compute_pixel(
            Vec2::new(0.5, 0.5),
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 1.0, 0.0),
            &|_| 0.5,
        );
        assert!((result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_pcss_no_blockers() {
        let sys = AreaLightShadowSystem::new(AreaLightShadowConfig::default());
        let result = sys.compute_pcss(
            Vec2::new(0.5, 0.5),
            0.5,
            &|_| 1.0, // All depths behind receiver
            0.001,
        );
        assert!((result.shadow - 1.0).abs() < EPSILON);
        assert_eq!(result.blocker_count, 0);
    }

    #[test]
    fn test_shadow_bias() {
        let calc = ShadowBiasCalculator::new(ShadowBiasConfig::default());
        let bias_face_on = calc.compute_bias(Vec3::Y, Vec3::Y, 2048, 100.0);
        let bias_grazing = calc.compute_bias(Vec3::Y, Vec3::X, 2048, 100.0);
        // Grazing angle should have more bias
        assert!(bias_grazing.depth_bias >= bias_face_on.depth_bias);
    }

    #[test]
    fn test_shadow_cache() {
        let mut cache = ShadowCache::new(4, 2);
        cache.begin_frame();
        let key = ShadowCacheKey { light_id: 1, cascade: 0, face: 0 };
        assert_eq!(cache.request(key), ShadowCacheState::Invalid);
        cache.insert(ShadowCacheEntry {
            key,
            state: ShadowCacheState::Valid,
            texture_handle: 42,
            resolution: 1024,
            view_projection: Mat4::IDENTITY,
            last_render_frame: 1,
            last_used_frame: 1,
            dynamic_object_count: 0,
            static_only: true,
            priority: 1.0,
        });
        assert_eq!(cache.request(key), ShadowCacheState::Valid);
        assert_eq!(cache.stats.cache_hits, 1);
    }

    #[test]
    fn test_shadow_importance() {
        let scorer = ShadowImportanceScorer::new(ShadowImportanceConfig::default());
        let bright_close = scorer.score(1000.0, 10.0, 5.0, 0.5);
        let dim_far = scorer.score(1.0, 0.1, 500.0, 0.001);
        assert!(bright_close > dim_far);
    }

    #[test]
    fn test_quality_levels() {
        assert!(ShadowQualityLevel::Ultra > ShadowQualityLevel::Low);
        assert_eq!(ShadowImportanceScorer::resolution_for_quality(ShadowQualityLevel::None), 0);
        assert_eq!(ShadowImportanceScorer::resolution_for_quality(ShadowQualityLevel::Ultra), 2048);
    }

    #[test]
    fn test_vsm() {
        let sys = AreaLightShadowSystem::new(AreaLightShadowConfig::default());
        // When receiver is in front of mean depth, should be fully lit
        let result = sys.compute_vsm(Vec2::new(0.5, 0.3), 0.3);
        assert!((result - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_shadow_system_v2() {
        let mut sys = ShadowSystemV2::new(800, 600);
        sys.begin_frame(Mat4::IDENTITY, Mat4::IDENTITY);
        let q = sys.evaluate_light(100.0, 5.0, 10.0, 0.3);
        assert!(q >= ShadowQualityLevel::Low);
        assert_eq!(sys.stats.lights_evaluated, 1);
    }
}
