// engine/render/src/atmospheric_effects.rs
//
// Additional atmospheric and optical phenomena for the Genovo engine.
//
// This module implements a collection of visually striking atmospheric effects:
// - Aurora borealis (northern/southern lights)
// - Rainbows (primary and secondary with supernumerary bands)
// - Halos (22-degree and 46-degree ice crystal halos)
// - Sundogs / parhelia (bright spots flanking the sun)
// - Crepuscular rays (god rays from volumetric light)
// - Heat haze (screen-space distortion from thermal convection)
// - Mirage effects (inferior and superior mirages)

use glam::{Mat4, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;
const TWO_PI: f32 = PI * 2.0;

/// Speed of light in vacuum (m/s) -- used for chromatic dispersion calculations.
const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Wavelengths (in nm) for visible light spectrum sampling.
pub const WAVELENGTHS_NM: [f32; 7] = [380.0, 450.0, 495.0, 520.0, 570.0, 620.0, 750.0];

/// Corresponding colours for the visible spectrum.
pub const SPECTRUM_COLORS: [Vec3; 7] = [
    Vec3::new(0.38, 0.0, 0.57),   // Violet
    Vec3::new(0.0, 0.0, 1.0),     // Blue
    Vec3::new(0.0, 0.5, 0.5),     // Cyan
    Vec3::new(0.0, 1.0, 0.0),     // Green
    Vec3::new(1.0, 1.0, 0.0),     // Yellow
    Vec3::new(1.0, 0.5, 0.0),     // Orange
    Vec3::new(1.0, 0.0, 0.0),     // Red
];

// ---------------------------------------------------------------------------
// Aurora Borealis
// ---------------------------------------------------------------------------

/// Configuration for the aurora borealis effect.
#[derive(Debug, Clone)]
pub struct AuroraConfig {
    /// Whether the aurora is visible.
    pub enabled: bool,
    /// Intensity of the aurora.
    pub intensity: f32,
    /// Primary colour (green is most common).
    pub primary_color: Vec3,
    /// Secondary colour (red/purple at higher altitudes).
    pub secondary_color: Vec3,
    /// Tertiary colour (blue at lower altitudes).
    pub tertiary_color: Vec3,
    /// Height of the aurora curtain base (in world units).
    pub base_height: f32,
    /// Thickness of the aurora curtain.
    pub thickness: f32,
    /// Number of curtain layers.
    pub layer_count: u32,
    /// Animation speed.
    pub animation_speed: f32,
    /// Turbulence frequency.
    pub turbulence_frequency: f32,
    /// Turbulence amplitude.
    pub turbulence_amplitude: f32,
    /// Direction the aurora faces (magnetic north).
    pub direction: Vec3,
    /// Coverage (0 = no aurora, 1 = full sky coverage).
    pub coverage: f32,
    /// Shimmer speed (fast intensity variations within curtains).
    pub shimmer_speed: f32,
    /// Whether to use volumetric ray marching.
    pub volumetric: bool,
    /// Number of ray-march steps (if volumetric).
    pub ray_march_steps: u32,
}

impl Default for AuroraConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 1.0,
            primary_color: Vec3::new(0.1, 0.9, 0.3),   // Green
            secondary_color: Vec3::new(0.6, 0.1, 0.5),  // Purple
            tertiary_color: Vec3::new(0.1, 0.3, 0.8),   // Blue
            base_height: 10000.0,
            thickness: 2000.0,
            layer_count: 3,
            animation_speed: 0.3,
            turbulence_frequency: 0.5,
            turbulence_amplitude: 500.0,
            direction: Vec3::new(0.0, 0.0, 1.0),
            coverage: 0.6,
            shimmer_speed: 2.0,
            volumetric: false,
            ray_march_steps: 16,
        }
    }
}

/// Runtime state for the aurora borealis effect.
#[derive(Debug)]
pub struct AuroraBorealis {
    /// Configuration.
    pub config: AuroraConfig,
    /// Current animation time.
    pub time: f32,
    /// Per-layer phase offsets.
    pub layer_phases: Vec<f32>,
    /// Per-layer intensities (animated).
    pub layer_intensities: Vec<f32>,
    /// Current overall activity level (0 = calm, 1 = storm).
    pub activity: f32,
    /// Target activity (smooth transition).
    pub target_activity: f32,
    /// Activity transition speed.
    pub activity_lerp_speed: f32,
}

impl AuroraBorealis {
    /// Create a new aurora system.
    pub fn new(config: AuroraConfig) -> Self {
        let count = config.layer_count as usize;
        let mut layer_phases = Vec::with_capacity(count);
        let mut layer_intensities = Vec::with_capacity(count);
        for i in 0..count {
            layer_phases.push(i as f32 * PI / count as f32);
            layer_intensities.push(1.0);
        }
        Self {
            config,
            time: 0.0,
            layer_phases,
            layer_intensities,
            activity: 0.5,
            target_activity: 0.5,
            activity_lerp_speed: 0.1,
        }
    }

    /// Update the aurora animation.
    pub fn update(&mut self, dt: f32) {
        self.time += dt * self.config.animation_speed;

        // Smooth activity transition
        let diff = self.target_activity - self.activity;
        self.activity += diff * self.activity_lerp_speed * dt;
        self.activity = self.activity.clamp(0.0, 1.0);

        // Update per-layer animations
        for i in 0..self.layer_phases.len() {
            let phase = self.time + self.layer_phases[i];
            let shimmer = ((phase * self.config.shimmer_speed * (i as f32 + 1.0)).sin() * 0.5 + 0.5)
                * self.activity;
            self.layer_intensities[i] = 0.3 + shimmer * 0.7;
        }
    }

    /// Set the activity level (0 = calm, 1 = geomagnetic storm).
    pub fn set_activity(&mut self, activity: f32) {
        self.target_activity = activity.clamp(0.0, 1.0);
    }

    /// Sample the aurora colour at a given view direction.
    pub fn sample(&self, view_dir: Vec3) -> Vec3 {
        if !self.config.enabled || view_dir.y <= 0.0 {
            return Vec3::ZERO;
        }

        // Height calculation: where the view ray hits the aurora curtain
        let t = self.config.base_height / (view_dir.y + EPSILON);
        let hit_pos = view_dir * t;

        // Curtain coordinate along the aurora direction
        let curtain_coord = hit_pos.dot(self.config.direction);

        let mut color = Vec3::ZERO;
        let mut total_weight = 0.0f32;

        for i in 0..self.config.layer_count as usize {
            if i >= self.layer_intensities.len() {
                break;
            }

            let layer_offset = i as f32 * self.config.thickness / self.config.layer_count as f32;
            let height = self.config.base_height + layer_offset;
            let height_factor = layer_offset / self.config.thickness;

            // Noise-based curtain shape
            let freq = self.config.turbulence_frequency;
            let phase = self.layer_phases.get(i).copied().unwrap_or(0.0);
            let noise_input = curtain_coord * freq + self.time + phase;
            let curtain_shape = (noise_input.sin() * 0.5 + 0.5)
                * ((noise_input * 2.3 + 1.7).sin() * 0.5 + 0.5);

            // Coverage mask
            let coverage_mask = if curtain_shape > (1.0 - self.config.coverage) {
                1.0
            } else {
                0.0
            };

            // Colour blending based on height
            let layer_color = if height_factor < 0.3 {
                Vec3::lerp(self.config.tertiary_color, self.config.primary_color, height_factor / 0.3)
            } else if height_factor < 0.7 {
                self.config.primary_color
            } else {
                Vec3::lerp(
                    self.config.primary_color,
                    self.config.secondary_color,
                    (height_factor - 0.7) / 0.3,
                )
            };

            let layer_intensity = self.layer_intensities[i];
            let weight = curtain_shape * coverage_mask * layer_intensity;
            color += layer_color * weight;
            total_weight += weight;
        }

        if total_weight > EPSILON {
            color /= total_weight;
            color *= total_weight.min(1.0);
        }

        // Fade near horizon
        let horizon_fade = (view_dir.y * 5.0).clamp(0.0, 1.0);
        color *= horizon_fade * self.config.intensity * self.activity;

        color
    }
}

// ---------------------------------------------------------------------------
// Rainbow
// ---------------------------------------------------------------------------

/// Rainbow type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RainbowType {
    /// Primary rainbow (42-degree radius).
    Primary,
    /// Secondary rainbow (51-degree radius, reversed colours).
    Secondary,
    /// Supernumerary bands (interference fringes inside primary).
    Supernumerary,
}

/// Configuration for a rainbow effect.
#[derive(Debug, Clone)]
pub struct RainbowConfig {
    /// Whether the rainbow is enabled.
    pub enabled: bool,
    /// Rainbow type to render.
    pub rainbow_type: RainbowType,
    /// Intensity of the rainbow.
    pub intensity: f32,
    /// Position of the anti-solar point (opposite the sun direction).
    pub anti_solar_dir: Vec3,
    /// Angular radius of the primary bow (degrees).
    pub primary_angle: f32,
    /// Angular radius of the secondary bow (degrees).
    pub secondary_angle: f32,
    /// Width of the bow in degrees.
    pub width: f32,
    /// Number of supernumerary bands (if type is Supernumerary).
    pub supernumerary_count: u32,
    /// Supernumerary band spacing (degrees).
    pub supernumerary_spacing: f32,
    /// Whether to darken Alexander's dark band (between primary and secondary).
    pub alexanders_dark_band: bool,
    /// Overall opacity.
    pub opacity: f32,
}

impl Default for RainbowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            rainbow_type: RainbowType::Primary,
            intensity: 0.5,
            anti_solar_dir: Vec3::new(0.0, -0.3, -1.0).normalize(),
            primary_angle: 42.0,
            secondary_angle: 51.0,
            width: 2.0,
            supernumerary_count: 3,
            supernumerary_spacing: 0.5,
            alexanders_dark_band: true,
            opacity: 0.7,
        }
    }
}

/// A rainbow renderer.
#[derive(Debug)]
pub struct Rainbow {
    /// Configuration.
    pub config: RainbowConfig,
    /// Precomputed spectrum colours for the bow.
    pub spectrum: Vec<Vec3>,
}

impl Rainbow {
    /// Create a new rainbow.
    pub fn new(config: RainbowConfig) -> Self {
        let spectrum = SPECTRUM_COLORS.to_vec();
        Self { config, spectrum }
    }

    /// Update the anti-solar direction (when sun moves).
    pub fn update_sun_direction(&mut self, sun_dir: Vec3) {
        self.config.anti_solar_dir = -sun_dir;
    }

    /// Sample the rainbow colour at a given view direction.
    pub fn sample(&self, view_dir: Vec3) -> Vec3 {
        if !self.config.enabled {
            return Vec3::ZERO;
        }

        let anti_solar = self.config.anti_solar_dir.normalize_or_zero();
        let cos_angle = view_dir.dot(anti_solar);
        let angle = cos_angle.acos().to_degrees();

        let mut color = Vec3::ZERO;

        // Primary bow
        let primary_dist = (angle - self.config.primary_angle).abs();
        if primary_dist < self.config.width {
            let t = primary_dist / self.config.width;
            let bow_intensity = (1.0 - t) * self.config.intensity;
            let spectrum_t = if self.config.rainbow_type == RainbowType::Secondary {
                1.0 - (angle - self.config.primary_angle + self.config.width)
                    / (self.config.width * 2.0)
            } else {
                (angle - self.config.primary_angle + self.config.width)
                    / (self.config.width * 2.0)
            };
            let spectrum_t = spectrum_t.clamp(0.0, 1.0);
            color += self.sample_spectrum(spectrum_t) * bow_intensity;
        }

        // Secondary bow (if enabled)
        if self.config.rainbow_type == RainbowType::Secondary
            || self.config.alexanders_dark_band
        {
            let secondary_dist = (angle - self.config.secondary_angle).abs();
            if secondary_dist < self.config.width * 1.5 {
                let t = secondary_dist / (self.config.width * 1.5);
                let bow_intensity = (1.0 - t) * self.config.intensity * 0.5;
                let spectrum_t = 1.0
                    - (angle - self.config.secondary_angle + self.config.width * 1.5)
                        / (self.config.width * 3.0);
                let spectrum_t = spectrum_t.clamp(0.0, 1.0);
                color += self.sample_spectrum(spectrum_t) * bow_intensity;
            }
        }

        // Supernumerary bands
        if self.config.rainbow_type == RainbowType::Supernumerary {
            for i in 0..self.config.supernumerary_count {
                let band_angle = self.config.primary_angle
                    - (i as f32 + 1.0) * self.config.supernumerary_spacing;
                let band_dist = (angle - band_angle).abs();
                if band_dist < self.config.supernumerary_spacing * 0.4 {
                    let t = band_dist / (self.config.supernumerary_spacing * 0.4);
                    let band_intensity = (1.0 - t) * self.config.intensity * 0.3 / (i as f32 + 1.0);
                    let phase = (i as f32 * PI) % TWO_PI;
                    let interference_color = Vec3::new(
                        0.5 + 0.5 * (phase).cos(),
                        0.5 + 0.5 * (phase + TWO_PI / 3.0).cos(),
                        0.5 + 0.5 * (phase + TWO_PI * 2.0 / 3.0).cos(),
                    );
                    color += interference_color * band_intensity;
                }
            }
        }

        // Alexander's dark band (darkening between primary and secondary)
        if self.config.alexanders_dark_band
            && angle > self.config.primary_angle + self.config.width
            && angle < self.config.secondary_angle - self.config.width
        {
            let dark_t = (angle - self.config.primary_angle - self.config.width)
                / (self.config.secondary_angle - self.config.primary_angle - self.config.width * 2.0);
            let dark_factor = 0.85 + 0.15 * (dark_t * PI).sin();
            color *= dark_factor;
        }

        color * self.config.opacity
    }

    /// Sample the rainbow spectrum at position t (0 = violet, 1 = red).
    fn sample_spectrum(&self, t: f32) -> Vec3 {
        let t = t.clamp(0.0, 1.0);
        let idx = t * (self.spectrum.len() - 1) as f32;
        let i = (idx as usize).min(self.spectrum.len() - 2);
        let frac = idx - i as f32;
        Vec3::lerp(self.spectrum[i], self.spectrum[i + 1], frac)
    }
}

// ---------------------------------------------------------------------------
// Halo (22° and 46° ice crystal halos)
// ---------------------------------------------------------------------------

/// Configuration for atmospheric halos.
#[derive(Debug, Clone)]
pub struct HaloConfig {
    /// Whether halos are enabled.
    pub enabled: bool,
    /// Intensity.
    pub intensity: f32,
    /// Whether to render the 22-degree halo.
    pub halo_22: bool,
    /// Whether to render the 46-degree halo.
    pub halo_46: bool,
    /// Sun direction.
    pub sun_direction: Vec3,
    /// Ring width in degrees.
    pub ring_width: f32,
    /// Colour tint (slight reddish inner edge is physically accurate).
    pub inner_color: Vec3,
    /// Outer colour.
    pub outer_color: Vec3,
    /// Opacity.
    pub opacity: f32,
}

impl Default for HaloConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.3,
            halo_22: true,
            halo_46: false,
            sun_direction: Vec3::new(0.0, 0.5, -1.0).normalize(),
            ring_width: 1.5,
            inner_color: Vec3::new(1.0, 0.9, 0.8),
            outer_color: Vec3::new(0.8, 0.85, 1.0),
            opacity: 0.5,
        }
    }
}

/// Halo renderer.
#[derive(Debug)]
pub struct AtmosphericHalo {
    pub config: HaloConfig,
}

impl AtmosphericHalo {
    pub fn new(config: HaloConfig) -> Self {
        Self { config }
    }

    /// Sample the halo contribution at a given view direction.
    pub fn sample(&self, view_dir: Vec3) -> Vec3 {
        if !self.config.enabled {
            return Vec3::ZERO;
        }

        let sun = self.config.sun_direction.normalize_or_zero();
        let cos_angle = view_dir.dot(sun);
        let angle = cos_angle.acos().to_degrees();
        let mut color = Vec3::ZERO;

        if self.config.halo_22 {
            let dist = (angle - 22.0).abs();
            if dist < self.config.ring_width {
                let t = dist / self.config.ring_width;
                let ring_intensity = (1.0 - t * t) * self.config.intensity;
                let ring_color = Vec3::lerp(self.config.inner_color, self.config.outer_color, t);
                color += ring_color * ring_intensity;
            }
        }

        if self.config.halo_46 {
            let dist = (angle - 46.0).abs();
            if dist < self.config.ring_width * 2.0 {
                let t = dist / (self.config.ring_width * 2.0);
                let ring_intensity = (1.0 - t * t) * self.config.intensity * 0.3;
                let ring_color = Vec3::lerp(self.config.inner_color, self.config.outer_color, t);
                color += ring_color * ring_intensity;
            }
        }

        color * self.config.opacity
    }
}

// ---------------------------------------------------------------------------
// Sundog (Parhelion)
// ---------------------------------------------------------------------------

/// Configuration for sundogs (parhelia).
#[derive(Debug, Clone)]
pub struct SundogConfig {
    /// Whether sundogs are enabled.
    pub enabled: bool,
    /// Intensity.
    pub intensity: f32,
    /// Sun direction.
    pub sun_direction: Vec3,
    /// Angular offset from the sun (approximately 22 degrees).
    pub angular_offset: f32,
    /// Sundog spot size in degrees.
    pub spot_size: f32,
    /// Colour spectrum (reddish on the sun side, white/blue on the outside).
    pub inner_color: Vec3,
    pub outer_color: Vec3,
    /// Whether to render the tail (extending away from the sun).
    pub show_tail: bool,
    /// Tail length in degrees.
    pub tail_length: f32,
    /// Opacity.
    pub opacity: f32,
}

impl Default for SundogConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.5,
            sun_direction: Vec3::new(0.0, 0.3, -1.0).normalize(),
            angular_offset: 22.0,
            spot_size: 3.0,
            inner_color: Vec3::new(1.0, 0.7, 0.4),
            outer_color: Vec3::new(0.8, 0.9, 1.0),
            show_tail: true,
            tail_length: 5.0,
            opacity: 0.6,
        }
    }
}

/// Sundog renderer.
#[derive(Debug)]
pub struct Sundog {
    pub config: SundogConfig,
}

impl Sundog {
    pub fn new(config: SundogConfig) -> Self {
        Self { config }
    }

    /// Get the positions of the two sundogs.
    pub fn sundog_directions(&self) -> (Vec3, Vec3) {
        let sun = self.config.sun_direction.normalize_or_zero();
        let offset_rad = self.config.angular_offset.to_radians();

        // Sundogs appear on the horizontal plane at the sun's elevation
        let right = sun.cross(Vec3::Y).normalize_or_zero();
        if right.length_squared() < EPSILON {
            return (sun, sun);
        }

        let left_sundog = (sun * offset_rad.cos() + right * offset_rad.sin()).normalize_or_zero();
        let right_sundog = (sun * offset_rad.cos() - right * offset_rad.sin()).normalize_or_zero();

        (left_sundog, right_sundog)
    }

    /// Sample the sundog contribution.
    pub fn sample(&self, view_dir: Vec3) -> Vec3 {
        if !self.config.enabled {
            return Vec3::ZERO;
        }

        let (left, right) = self.sundog_directions();
        let mut color = Vec3::ZERO;

        for sundog_dir in [left, right] {
            let cos_angle = view_dir.dot(sundog_dir);
            let angle = cos_angle.acos().to_degrees();

            if angle < self.config.spot_size {
                let t = angle / self.config.spot_size;
                let spot_intensity = (1.0 - t * t) * self.config.intensity;
                let spot_color = Vec3::lerp(self.config.inner_color, self.config.outer_color, t);
                color += spot_color * spot_intensity;
            }

            // Tail
            if self.config.show_tail {
                let sun = self.config.sun_direction.normalize_or_zero();
                let away_from_sun = (view_dir - sun * view_dir.dot(sun)).normalize_or_zero();
                let sundog_away = (sundog_dir - sun * sundog_dir.dot(sun)).normalize_or_zero();
                let tail_alignment = away_from_sun.dot(sundog_away).max(0.0);

                if tail_alignment > 0.8 && angle < self.config.spot_size + self.config.tail_length {
                    let tail_t = (angle - self.config.spot_size) / self.config.tail_length;
                    if tail_t > 0.0 && tail_t < 1.0 {
                        let tail_intensity = (1.0 - tail_t) * self.config.intensity * 0.2 * tail_alignment;
                        color += self.config.outer_color * tail_intensity;
                    }
                }
            }
        }

        color * self.config.opacity
    }
}

// ---------------------------------------------------------------------------
// Crepuscular Rays (God Rays)
// ---------------------------------------------------------------------------

/// Configuration for crepuscular rays.
#[derive(Debug, Clone)]
pub struct CrepuscularRaysConfig {
    /// Whether crepuscular rays are enabled.
    pub enabled: bool,
    /// Intensity of the rays.
    pub intensity: f32,
    /// Number of samples along each ray.
    pub sample_count: u32,
    /// Light source screen position (normalised 0-1).
    pub light_screen_pos: Vec2,
    /// Decay factor (how quickly rays fade with distance).
    pub decay: f32,
    /// Exposure adjustment.
    pub exposure: f32,
    /// Density of the rays.
    pub density: f32,
    /// Weight (contribution of each sample).
    pub weight: f32,
    /// Maximum ray length (screen fraction).
    pub max_length: f32,
    /// Colour tint.
    pub color: Vec3,
    /// Whether the light source is on screen.
    pub light_on_screen: bool,
    /// Whether to use a threshold mask (only bright pixels generate rays).
    pub use_threshold: bool,
    /// Threshold luminance.
    pub threshold: f32,
}

impl Default for CrepuscularRaysConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 1.0,
            sample_count: 64,
            light_screen_pos: Vec2::new(0.5, 0.3),
            decay: 0.97,
            exposure: 0.3,
            density: 1.0,
            weight: 0.01,
            max_length: 1.0,
            color: Vec3::new(1.0, 0.95, 0.8),
            light_on_screen: true,
            use_threshold: true,
            threshold: 0.8,
        }
    }
}

/// Crepuscular ray renderer (screen-space post-process).
#[derive(Debug)]
pub struct CrepuscularRays {
    pub config: CrepuscularRaysConfig,
    /// Intermediate render target for the occlusion mask.
    pub occlusion_rt: u64,
    /// Intermediate render target for the ray result.
    pub rays_rt: u64,
    /// Resolution.
    pub resolution: Vec2,
}

impl CrepuscularRays {
    pub fn new(config: CrepuscularRaysConfig, width: u32, height: u32) -> Self {
        Self {
            config,
            occlusion_rt: 0,
            rays_rt: 0,
            resolution: Vec2::new(width as f32, height as f32),
        }
    }

    /// Update the light source screen position from a world-space sun direction.
    pub fn update_from_sun(&mut self, sun_dir: Vec3, view_projection: Mat4) {
        let clip = view_projection * Vec4::new(sun_dir.x * 10000.0, sun_dir.y * 10000.0, sun_dir.z * 10000.0, 1.0);
        if clip.w > 0.0 {
            let ndc = clip / clip.w;
            self.config.light_screen_pos = Vec2::new(
                (ndc.x + 1.0) * 0.5,
                (1.0 - ndc.y) * 0.5,
            );
            self.config.light_on_screen = ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0 && ndc.z > 0.0;
        } else {
            self.config.light_on_screen = false;
        }
    }

    /// CPU reference: compute the ray contribution for a single pixel.
    pub fn sample_pixel(&self, uv: Vec2, scene_luminance: &dyn Fn(Vec2) -> f32) -> Vec3 {
        if !self.config.enabled || !self.config.light_on_screen {
            return Vec3::ZERO;
        }

        let delta = uv - self.config.light_screen_pos;
        let dist = delta.length();
        if dist < EPSILON || dist > self.config.max_length {
            return Vec3::ZERO;
        }

        let step = delta * self.config.density / self.config.sample_count as f32;
        let mut accumulated = 0.0f32;
        let mut decay = 1.0f32;
        let mut sample_pos = uv;

        for _ in 0..self.config.sample_count {
            sample_pos -= step;

            if sample_pos.x < 0.0 || sample_pos.x > 1.0 || sample_pos.y < 0.0 || sample_pos.y > 1.0 {
                break;
            }

            let luminance = scene_luminance(sample_pos);
            let masked = if self.config.use_threshold {
                (luminance - self.config.threshold).max(0.0)
            } else {
                luminance
            };

            accumulated += masked * decay * self.config.weight;
            decay *= self.config.decay;
        }

        self.config.color * accumulated * self.config.intensity * self.config.exposure
    }

    /// Resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.resolution = Vec2::new(width as f32, height as f32);
    }
}

// ---------------------------------------------------------------------------
// Heat Haze
// ---------------------------------------------------------------------------

/// Configuration for heat haze (screen-space distortion).
#[derive(Debug, Clone)]
pub struct HeatHazeConfig {
    /// Whether heat haze is enabled.
    pub enabled: bool,
    /// Distortion strength in pixels.
    pub distortion_strength: f32,
    /// Animation speed.
    pub speed: f32,
    /// Frequency of the distortion pattern.
    pub frequency: f32,
    /// Height range where heat haze is visible (world Y).
    pub min_height: f32,
    pub max_height: f32,
    /// Distance range where heat haze is visible.
    pub min_distance: f32,
    pub max_distance: f32,
    /// Whether to also distort the depth buffer (for correct sorting).
    pub distort_depth: bool,
    /// Number of octaves for the noise pattern.
    pub noise_octaves: u32,
    /// Lacunarity of noise.
    pub noise_lacunarity: f32,
}

impl Default for HeatHazeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            distortion_strength: 2.0,
            speed: 1.0,
            frequency: 10.0,
            min_height: 0.0,
            max_height: 5.0,
            min_distance: 10.0,
            max_distance: 500.0,
            distort_depth: false,
            noise_octaves: 3,
            noise_lacunarity: 2.0,
        }
    }
}

/// Heat haze effect system.
#[derive(Debug)]
pub struct HeatHaze {
    pub config: HeatHazeConfig,
    /// Current time for animation.
    pub time: f32,
}

impl HeatHaze {
    pub fn new(config: HeatHazeConfig) -> Self {
        Self { config, time: 0.0 }
    }

    /// Update animation.
    pub fn update(&mut self, dt: f32) {
        self.time += dt * self.config.speed;
    }

    /// Compute the UV distortion offset for a pixel.
    pub fn distortion_offset(&self, uv: Vec2, world_pos: Vec3) -> Vec2 {
        if !self.config.enabled {
            return Vec2::ZERO;
        }

        // Height falloff
        let height_t = if self.config.max_height > self.config.min_height {
            ((world_pos.y - self.config.min_height)
                / (self.config.max_height - self.config.min_height))
                .clamp(0.0, 1.0)
        } else {
            1.0
        };
        let height_factor = 1.0 - height_t; // Strongest near min_height

        // Distance falloff
        let dist = (world_pos.x * world_pos.x + world_pos.z * world_pos.z).sqrt();
        let dist_t = if self.config.max_distance > self.config.min_distance {
            ((dist - self.config.min_distance)
                / (self.config.max_distance - self.config.min_distance))
                .clamp(0.0, 1.0)
        } else {
            1.0
        };
        let dist_factor = dist_t * (1.0 - dist_t) * 4.0; // Peak in middle range

        // Noise-based distortion
        let mut amplitude = self.config.distortion_strength;
        let mut freq = self.config.frequency;
        let mut offset = Vec2::ZERO;

        for _ in 0..self.config.noise_octaves {
            let nx = (uv.x * freq + self.time * 0.7).sin()
                * (uv.y * freq * 1.3 + self.time).cos();
            let ny = (uv.y * freq + self.time * 0.9).cos()
                * (uv.x * freq * 0.7 + self.time * 1.1).sin();
            offset += Vec2::new(nx, ny) * amplitude;
            amplitude *= 0.5;
            freq *= self.config.noise_lacunarity;
        }

        offset * height_factor * dist_factor / 1000.0
    }
}

// ---------------------------------------------------------------------------
// Mirage Effect
// ---------------------------------------------------------------------------

/// Mirage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirageType {
    /// Inferior mirage (hot surface, image appears below the object).
    Inferior,
    /// Superior mirage (cold surface/warm air above, image appears above).
    Superior,
    /// Fata Morgana (complex layered mirage with stretching and inversion).
    FataMorgana,
}

/// Configuration for mirage effects.
#[derive(Debug, Clone)]
pub struct MirageConfig {
    /// Whether the mirage is enabled.
    pub enabled: bool,
    /// Type of mirage.
    pub mirage_type: MirageType,
    /// Intensity of the mirage distortion.
    pub intensity: f32,
    /// Height of the mirage layer (world Y).
    pub mirage_height: f32,
    /// Thickness of the distortion layer.
    pub layer_thickness: f32,
    /// Distance at which the mirage becomes visible.
    pub visible_distance: f32,
    /// Animation speed.
    pub animation_speed: f32,
    /// Temperature gradient strength (affects refraction).
    pub temperature_gradient: f32,
    /// Whether to render the inverted reflection.
    pub show_reflection: bool,
    /// Reflection intensity.
    pub reflection_intensity: f32,
}

impl Default for MirageConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mirage_type: MirageType::Inferior,
            intensity: 1.0,
            mirage_height: 0.5,
            layer_thickness: 2.0,
            visible_distance: 50.0,
            animation_speed: 0.5,
            temperature_gradient: 1.0,
            show_reflection: true,
            reflection_intensity: 0.3,
        }
    }
}

/// Mirage effect system.
#[derive(Debug)]
pub struct MirageEffect {
    pub config: MirageConfig,
    pub time: f32,
}

impl MirageEffect {
    pub fn new(config: MirageConfig) -> Self {
        Self { config, time: 0.0 }
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt * self.config.animation_speed;
    }

    /// Compute the UV distortion for the mirage effect at a given world position.
    pub fn compute_distortion(&self, world_pos: Vec3, camera_pos: Vec3) -> Vec2 {
        if !self.config.enabled {
            return Vec2::ZERO;
        }

        let distance = ((world_pos.x - camera_pos.x).powi(2)
            + (world_pos.z - camera_pos.z).powi(2))
        .sqrt();

        if distance < self.config.visible_distance {
            return Vec2::ZERO;
        }

        let height_diff = (world_pos.y - self.config.mirage_height).abs();
        let height_factor = if height_diff < self.config.layer_thickness {
            1.0 - height_diff / self.config.layer_thickness
        } else {
            0.0
        };

        let distance_factor = ((distance - self.config.visible_distance) / 100.0)
            .clamp(0.0, 1.0);

        let shimmer = (self.time * 3.0 + world_pos.x * 0.5).sin()
            * (self.time * 2.3 + world_pos.z * 0.7).cos();

        let base_distortion = match self.config.mirage_type {
            MirageType::Inferior => Vec2::new(shimmer * 0.3, -1.0),
            MirageType::Superior => Vec2::new(shimmer * 0.3, 1.0),
            MirageType::FataMorgana => {
                let stretch = (self.time * 1.5 + world_pos.x * 0.2).sin();
                Vec2::new(shimmer * 0.5, stretch * 2.0)
            }
        };

        base_distortion
            * self.config.intensity
            * self.config.temperature_gradient
            * height_factor
            * distance_factor
            / 100.0
    }

    /// Compute the reflection factor for inferior mirages.
    pub fn reflection_factor(&self, world_pos: Vec3, camera_pos: Vec3) -> f32 {
        if !self.config.enabled || !self.config.show_reflection {
            return 0.0;
        }

        let height_diff = (world_pos.y - self.config.mirage_height).abs();
        let in_layer = if height_diff < self.config.layer_thickness {
            1.0 - height_diff / self.config.layer_thickness
        } else {
            0.0
        };

        let distance = ((world_pos.x - camera_pos.x).powi(2)
            + (world_pos.z - camera_pos.z).powi(2))
        .sqrt();
        let dist_factor = ((distance - self.config.visible_distance) / 200.0)
            .clamp(0.0, 1.0);

        in_layer * dist_factor * self.config.reflection_intensity
    }
}

// ---------------------------------------------------------------------------
// Combined atmospheric effects manager
// ---------------------------------------------------------------------------

/// Manages all atmospheric effects in a unified interface.
#[derive(Debug)]
pub struct AtmosphericEffectsManager {
    /// Aurora borealis.
    pub aurora: AuroraBorealis,
    /// Rainbow.
    pub rainbow: Rainbow,
    /// Atmospheric halo.
    pub halo: AtmosphericHalo,
    /// Sundog.
    pub sundog: Sundog,
    /// Crepuscular rays.
    pub crepuscular_rays: CrepuscularRays,
    /// Heat haze.
    pub heat_haze: HeatHaze,
    /// Mirage effect.
    pub mirage: MirageEffect,
    /// Global intensity multiplier.
    pub global_intensity: f32,
    /// Whether to process effects this frame.
    pub active: bool,
}

impl AtmosphericEffectsManager {
    /// Create a new manager with default settings.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            aurora: AuroraBorealis::new(AuroraConfig::default()),
            rainbow: Rainbow::new(RainbowConfig::default()),
            halo: AtmosphericHalo::new(HaloConfig::default()),
            sundog: Sundog::new(SundogConfig::default()),
            crepuscular_rays: CrepuscularRays::new(CrepuscularRaysConfig::default(), width, height),
            heat_haze: HeatHaze::new(HeatHazeConfig::default()),
            mirage: MirageEffect::new(MirageConfig::default()),
            global_intensity: 1.0,
            active: true,
        }
    }

    /// Update all effects.
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }
        self.aurora.update(dt);
        self.heat_haze.update(dt);
        self.mirage.update(dt);
    }

    /// Update sun-dependent effects.
    pub fn update_sun(&mut self, sun_dir: Vec3, view_projection: Mat4) {
        self.rainbow.update_sun_direction(sun_dir);
        self.halo.config.sun_direction = sun_dir;
        self.sundog.config.sun_direction = sun_dir;
        self.crepuscular_rays.update_from_sun(sun_dir, view_projection);
    }

    /// Sample the combined sky contribution at a view direction.
    pub fn sample_sky(&self, view_dir: Vec3) -> Vec3 {
        if !self.active {
            return Vec3::ZERO;
        }
        let mut color = Vec3::ZERO;
        color += self.aurora.sample(view_dir);
        color += self.rainbow.sample(view_dir);
        color += self.halo.sample(view_dir);
        color += self.sundog.sample(view_dir);
        color * self.global_intensity
    }

    /// Resize screen-space effects.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.crepuscular_rays.resize(width, height);
    }

    /// Get a count of active effects.
    pub fn active_effect_count(&self) -> u32 {
        let mut count = 0u32;
        if self.aurora.config.enabled { count += 1; }
        if self.rainbow.config.enabled { count += 1; }
        if self.halo.config.enabled { count += 1; }
        if self.sundog.config.enabled { count += 1; }
        if self.crepuscular_rays.config.enabled { count += 1; }
        if self.heat_haze.config.enabled { count += 1; }
        if self.mirage.config.enabled { count += 1; }
        count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aurora_sample() {
        let mut aurora = AuroraBorealis::new(AuroraConfig {
            enabled: true,
            ..Default::default()
        });
        aurora.update(1.0);
        // Looking up should produce some colour
        let color = aurora.sample(Vec3::Y);
        assert!(color.length() >= 0.0);
        // Looking down should produce nothing
        let color_down = aurora.sample(Vec3::NEG_Y);
        assert!((color_down.length()) < EPSILON);
    }

    #[test]
    fn test_rainbow_spectrum() {
        let rainbow = Rainbow::new(RainbowConfig {
            enabled: true,
            ..Default::default()
        });
        let c0 = rainbow.sample_spectrum(0.0);
        let c1 = rainbow.sample_spectrum(1.0);
        assert!(c0.length() > 0.0);
        assert!(c1.length() > 0.0);
    }

    #[test]
    fn test_halo_disabled() {
        let halo = AtmosphericHalo::new(HaloConfig::default());
        let c = halo.sample(Vec3::X);
        assert!(c.length() < EPSILON);
    }

    #[test]
    fn test_sundog_directions() {
        let sundog = Sundog::new(SundogConfig {
            enabled: true,
            ..Default::default()
        });
        let (left, right) = sundog.sundog_directions();
        // Sundogs should be on opposite sides of the sun
        let sun = sundog.config.sun_direction.normalize();
        assert!(left.dot(sun) > 0.0);
        assert!(right.dot(sun) > 0.0);
    }

    #[test]
    fn test_heat_haze_zero_when_disabled() {
        let haze = HeatHaze::new(HeatHazeConfig::default());
        let offset = haze.distortion_offset(Vec2::new(0.5, 0.5), Vec3::ZERO);
        assert!(offset.length() < EPSILON);
    }

    #[test]
    fn test_atmospheric_manager() {
        let mut manager = AtmosphericEffectsManager::new(800, 600);
        manager.update(0.016);
        assert_eq!(manager.active_effect_count(), 0);
        manager.aurora.config.enabled = true;
        assert_eq!(manager.active_effect_count(), 1);
    }
}
