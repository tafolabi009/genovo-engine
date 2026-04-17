// engine/render/src/lens_effects.rs
//
// Camera lens simulation effects for the Genovo engine.
//
// Implements physically-inspired camera lens artefacts:
//
// - **Lens flare** — Ghost images from bright light sources scattered through
//   the lens system. Multiple elements with varying size, colour, and opacity.
// - **Anamorphic streak** — Horizontal streak from bright points, typical of
//   anamorphic cinema lenses.
// - **Lens dirt** — Dirt/smudge texture overlay that brightens in the presence
//   of flares.
// - **Bloom threshold** — Bright-pass extraction for bloom and flare sources.
// - **Star-burst** — Diffraction spikes from the aperture blades.
// - **Lens distortion** — Barrel and pincushion distortion from real lens
//   geometry.
//
// # Pipeline integration
//
// These effects are applied as post-processing passes after tone mapping:
//
// 1. Bright-pass filter to extract luminous pixels.
// 2. Generate lens flare ghosts from the bright-pass.
// 3. Generate anamorphic streaks.
// 4. Generate star-burst pattern.
// 5. Composite flares, streaks, star-burst, and dirt over the scene.
// 6. Apply lens distortion as the final step.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Lens flare
// ---------------------------------------------------------------------------

/// A single lens flare ghost element.
#[derive(Debug, Clone)]
pub struct FlareGhost {
    /// Position along the flare line (0 = light source, 1 = opposite side).
    /// Values > 1 or < 0 place ghosts outside the line.
    pub position: f32,
    /// Scale relative to the source brightness area.
    pub scale: f32,
    /// Ghost colour tint (linear RGB).
    pub color: [f32; 3],
    /// Opacity [0, 1].
    pub opacity: f32,
    /// Whether this ghost uses a radial gradient (disc) or a ring shape.
    pub ring: bool,
    /// Ring thickness (only used if `ring` is true) [0, 1].
    pub ring_thickness: f32,
    /// Chromatic aberration strength for this ghost.
    pub chromatic_aberration: f32,
}

impl FlareGhost {
    /// Creates a disc-shaped ghost.
    pub fn disc(position: f32, scale: f32, color: [f32; 3], opacity: f32) -> Self {
        Self {
            position,
            scale,
            color,
            opacity,
            ring: false,
            ring_thickness: 0.0,
            chromatic_aberration: 0.0,
        }
    }

    /// Creates a ring-shaped ghost.
    pub fn ring(position: f32, scale: f32, color: [f32; 3], opacity: f32, thickness: f32) -> Self {
        Self {
            position,
            scale,
            color,
            opacity,
            ring: true,
            ring_thickness: thickness,
            chromatic_aberration: 0.0,
        }
    }

    /// Sets chromatic aberration for this ghost.
    pub fn with_chromatic(mut self, strength: f32) -> Self {
        self.chromatic_aberration = strength;
        self
    }

    /// Evaluates the ghost intensity at a distance from its centre.
    ///
    /// # Arguments
    /// * `dist` — Normalised distance from the ghost centre [0, 1].
    ///
    /// # Returns
    /// Intensity [0, 1].
    pub fn evaluate(&self, dist: f32) -> f32 {
        if dist > 1.0 {
            return 0.0;
        }

        if self.ring {
            let inner = 1.0 - self.ring_thickness;
            if dist < inner {
                // Inside the ring: fade in.
                let t = dist / inner;
                (t - 0.8).max(0.0) / 0.2
            } else {
                // On the ring: fade out at the edge.
                let t = (dist - inner) / self.ring_thickness;
                (1.0 - t).max(0.0)
            }
        } else {
            // Disc: smooth radial falloff.
            let t = 1.0 - dist;
            t * t
        }
    }
}

/// Lens flare configuration.
#[derive(Debug, Clone)]
pub struct LensFlareConfig {
    /// Whether lens flare is enabled.
    pub enabled: bool,
    /// Ghost elements.
    pub ghosts: Vec<FlareGhost>,
    /// Overall intensity multiplier.
    pub intensity: f32,
    /// Threshold luminance for triggering flares.
    pub threshold: f32,
    /// Whether the flare fades near screen edges.
    pub edge_fade: bool,
    /// Edge fade distance (normalised, from screen edge).
    pub edge_fade_distance: f32,
    /// Halo radius (large ring around the source).
    pub halo_radius: f32,
    /// Halo colour.
    pub halo_color: [f32; 3],
    /// Halo intensity.
    pub halo_intensity: f32,
    /// Halo ring thickness.
    pub halo_thickness: f32,
    /// Global chromatic aberration for all ghosts.
    pub global_chromatic_aberration: f32,
}

impl LensFlareConfig {
    /// Creates a default physically-inspired lens flare.
    pub fn new() -> Self {
        Self {
            enabled: true,
            ghosts: vec![
                FlareGhost::disc(0.2, 0.5, [1.0, 0.8, 0.4], 0.3),
                FlareGhost::disc(0.4, 0.3, [0.4, 0.6, 1.0], 0.2),
                FlareGhost::ring(0.6, 0.8, [0.3, 1.0, 0.5], 0.15, 0.2),
                FlareGhost::disc(0.8, 0.15, [1.0, 0.4, 0.8], 0.1),
                FlareGhost::disc(1.0, 0.4, [0.8, 0.9, 1.0], 0.2),
                FlareGhost::ring(1.3, 1.2, [1.0, 0.7, 0.3], 0.1, 0.15),
                FlareGhost::disc(1.5, 0.2, [0.5, 0.5, 1.0], 0.08),
            ],
            intensity: 1.0,
            threshold: 1.0,
            edge_fade: true,
            edge_fade_distance: 0.15,
            halo_radius: 0.35,
            halo_color: [0.8, 0.9, 1.0],
            halo_intensity: 0.15,
            halo_thickness: 0.1,
            global_chromatic_aberration: 0.02,
        }
    }

    /// Creates a cinematic anamorphic flare.
    pub fn cinematic() -> Self {
        Self {
            enabled: true,
            ghosts: vec![
                FlareGhost::disc(0.15, 0.6, [0.7, 0.8, 1.0], 0.4),
                FlareGhost::ring(0.3, 0.4, [0.4, 0.5, 1.0], 0.25, 0.3),
                FlareGhost::disc(0.5, 0.2, [1.0, 0.6, 0.3], 0.15),
                FlareGhost::disc(0.7, 0.35, [0.3, 0.8, 1.0], 0.2),
                FlareGhost::ring(1.0, 1.0, [0.9, 0.95, 1.0], 0.15, 0.1),
            ],
            intensity: 1.5,
            threshold: 0.8,
            edge_fade: true,
            edge_fade_distance: 0.2,
            halo_radius: 0.4,
            halo_color: [0.6, 0.7, 1.0],
            halo_intensity: 0.2,
            halo_thickness: 0.08,
            global_chromatic_aberration: 0.04,
        }
    }

    /// Computes the edge fade factor for a given screen UV.
    pub fn edge_fade_factor(&self, uv: [f32; 2]) -> f32 {
        if !self.edge_fade {
            return 1.0;
        }
        let dx = (uv[0] - 0.5).abs() * 2.0; // [0, 1] from centre to edge.
        let dy = (uv[1] - 0.5).abs() * 2.0;
        let edge_dist = dx.max(dy);
        let fade_start = 1.0 - self.edge_fade_distance;
        if edge_dist < fade_start {
            1.0
        } else {
            1.0 - (edge_dist - fade_start) / self.edge_fade_distance
        }
    }
}

impl Default for LensFlareConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes lens flare ghosts for a single bright source.
///
/// # Arguments
/// * `source_uv` — Screen UV of the bright source.
/// * `source_brightness` — Brightness of the source.
/// * `config` — Lens flare configuration.
/// * `output` — Output buffer (RGB, same dimensions as screen).
/// * `width`, `height` — Screen dimensions.
pub fn compute_lens_flare(
    source_uv: [f32; 2],
    source_brightness: f32,
    config: &LensFlareConfig,
    output: &mut [[f32; 3]],
    width: u32,
    height: u32,
) {
    if !config.enabled || source_brightness < config.threshold {
        return;
    }

    let centre = [0.5f32, 0.5];
    let flare_dir = [
        centre[0] - source_uv[0],
        centre[1] - source_uv[1],
    ];

    let brightness_factor = (source_brightness - config.threshold).max(0.0) * config.intensity;

    for ghost in &config.ghosts {
        let ghost_centre = [
            source_uv[0] + flare_dir[0] * ghost.position * 2.0,
            source_uv[1] + flare_dir[1] * ghost.position * 2.0,
        ];

        let ghost_radius = ghost.scale * 0.1;

        for y in 0..height {
            for x in 0..width {
                let px_uv = [
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                ];

                let dx = px_uv[0] - ghost_centre[0];
                let dy = (px_uv[1] - ghost_centre[1]) * (width as f32 / height as f32);
                let dist = (dx * dx + dy * dy).sqrt() / ghost_radius;

                let intensity = ghost.evaluate(dist) * ghost.opacity * brightness_factor;

                if intensity > 0.001 {
                    let edge_fade = config.edge_fade_factor(px_uv);
                    let final_intensity = intensity * edge_fade;

                    let idx = (y * width + x) as usize;
                    output[idx][0] += ghost.color[0] * final_intensity;
                    output[idx][1] += ghost.color[1] * final_intensity;
                    output[idx][2] += ghost.color[2] * final_intensity;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Anamorphic streak
// ---------------------------------------------------------------------------

/// Anamorphic streak configuration.
#[derive(Debug, Clone)]
pub struct AnamorphicStreakConfig {
    /// Whether streaks are enabled.
    pub enabled: bool,
    /// Streak direction (0 = horizontal, PI/2 = vertical).
    pub angle: f32,
    /// Streak length (in UV units).
    pub length: f32,
    /// Streak colour tint.
    pub color: [f32; 3],
    /// Streak intensity.
    pub intensity: f32,
    /// Brightness threshold for generating streaks.
    pub threshold: f32,
    /// Number of blur passes for the streak.
    pub blur_passes: u32,
    /// Streak falloff exponent (higher = faster falloff).
    pub falloff: f32,
}

impl AnamorphicStreakConfig {
    /// Creates default anamorphic streak settings.
    pub fn new() -> Self {
        Self {
            enabled: true,
            angle: 0.0, // Horizontal.
            length: 0.3,
            color: [0.7, 0.8, 1.0],
            intensity: 0.5,
            threshold: 1.0,
            blur_passes: 3,
            falloff: 2.0,
        }
    }

    /// Creates a vertical streak.
    pub fn vertical() -> Self {
        Self {
            angle: PI / 2.0,
            ..Self::new()
        }
    }

    /// Returns the streak direction as a unit vector.
    pub fn direction(&self) -> [f32; 2] {
        [self.angle.cos(), self.angle.sin()]
    }
}

impl Default for AnamorphicStreakConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates anamorphic streaks from a bright-pass buffer.
///
/// # Arguments
/// * `bright_pass` — Bright pixels extracted from the scene.
/// * `width`, `height` — Buffer dimensions.
/// * `config` — Streak configuration.
///
/// # Returns
/// Streak colour buffer.
pub fn generate_anamorphic_streak(
    bright_pass: &[[f32; 3]],
    width: u32,
    height: u32,
    config: &AnamorphicStreakConfig,
) -> Vec<[f32; 3]> {
    let total = (width * height) as usize;
    let mut current = bright_pass.to_vec();
    let mut temp = vec![[0.0f32; 3]; total];

    let dir = config.direction();
    let step_pixels_x = (dir[0] * width as f32 * config.length / config.blur_passes as f32).round();
    let step_pixels_y = (dir[1] * height as f32 * config.length / config.blur_passes as f32).round();

    for pass in 0..config.blur_passes {
        let offset = (1 << pass) as f32;
        let ox = (step_pixels_x * offset) as i32;
        let oy = (step_pixels_y * offset) as i32;

        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let idx = (y * width as i32 + x) as usize;
                let mut sum = current[idx];

                // Sample in both directions along the streak.
                for &sign in &[-1i32, 1] {
                    let sx = x + ox * sign;
                    let sy = y + oy * sign;

                    if sx >= 0 && sx < width as i32 && sy >= 0 && sy < height as i32 {
                        let si = (sy * width as i32 + sx) as usize;
                        let weight = (1.0 / (1.0 + (offset * sign.abs() as f32).powf(config.falloff)))
                            * config.intensity;
                        sum[0] += current[si][0] * weight;
                        sum[1] += current[si][1] * weight;
                        sum[2] += current[si][2] * weight;
                    }
                }

                temp[idx] = [
                    sum[0] * config.color[0],
                    sum[1] * config.color[1],
                    sum[2] * config.color[2],
                ];
            }
        }

        std::mem::swap(&mut current, &mut temp);
    }

    current
}

// ---------------------------------------------------------------------------
// Lens dirt
// ---------------------------------------------------------------------------

/// Lens dirt overlay configuration.
#[derive(Debug, Clone)]
pub struct LensDirtConfig {
    /// Whether dirt is enabled.
    pub enabled: bool,
    /// Dirt intensity.
    pub intensity: f32,
    /// Dirt texture resolution.
    pub resolution: u32,
    /// Procedural dirt pattern (if no texture is loaded).
    pub procedural: bool,
    /// Number of smudge spots.
    pub smudge_count: u32,
    /// Smudge radius range [min, max].
    pub smudge_radius: [f32; 2],
    /// Smudge opacity.
    pub smudge_opacity: f32,
}

impl LensDirtConfig {
    /// Creates default dirt settings.
    pub fn new() -> Self {
        Self {
            enabled: true,
            intensity: 0.3,
            resolution: 512,
            procedural: true,
            smudge_count: 20,
            smudge_radius: [0.02, 0.08],
            smudge_opacity: 0.5,
        }
    }
}

impl Default for LensDirtConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates a procedural lens dirt texture.
///
/// Returns a single-channel intensity texture (higher = more dirt).
pub fn generate_dirt_texture(config: &LensDirtConfig) -> Vec<f32> {
    let res = config.resolution;
    let total = (res * res) as usize;
    let mut dirt = vec![0.0f32; total];

    // Generate random smudge spots.
    for i in 0..config.smudge_count {
        let seed = i * 12345 + 6789;
        let cx = pseudo_hash(seed) * 0.8 + 0.1;
        let cy = pseudo_hash(seed + 1) * 0.8 + 0.1;
        let radius = config.smudge_radius[0]
            + pseudo_hash(seed + 2) * (config.smudge_radius[1] - config.smudge_radius[0]);
        let opacity = config.smudge_opacity * (0.5 + pseudo_hash(seed + 3) * 0.5);

        for y in 0..res {
            for x in 0..res {
                let u = x as f32 / res as f32;
                let v = y as f32 / res as f32;
                let dx = u - cx;
                let dy = v - cy;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < radius {
                    let t = 1.0 - (dist / radius);
                    let value = t * t * opacity;
                    let idx = (y * res + x) as usize;
                    dirt[idx] = (dirt[idx] + value).min(1.0);
                }
            }
        }
    }

    // Add some uniform noise.
    for idx in 0..total {
        let noise = pseudo_hash(idx as u32) * 0.05;
        dirt[idx] = (dirt[idx] + noise).min(1.0);
    }

    dirt
}

/// Simple hash for procedural generation.
fn pseudo_hash(seed: u32) -> f32 {
    let n = seed.wrapping_mul(374761393);
    let n = n ^ (n >> 13);
    let n = n.wrapping_mul(n.wrapping_mul(60493).wrapping_add(19990303));
    (n & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF as f32
}

// ---------------------------------------------------------------------------
// Star-burst (diffraction spikes)
// ---------------------------------------------------------------------------

/// Star-burst / diffraction spike configuration.
#[derive(Debug, Clone)]
pub struct StarBurstConfig {
    /// Whether star-burst is enabled.
    pub enabled: bool,
    /// Number of aperture blades.
    pub blade_count: u32,
    /// Rotation angle of the aperture (radians).
    pub rotation: f32,
    /// Spike length.
    pub spike_length: f32,
    /// Spike intensity.
    pub intensity: f32,
    /// Spike colour.
    pub color: [f32; 3],
    /// Spike width (thickness).
    pub width: f32,
    /// Brightness threshold.
    pub threshold: f32,
    /// Chromatic separation.
    pub chromatic_separation: f32,
}

impl StarBurstConfig {
    /// Creates default star-burst settings for a 6-blade aperture.
    pub fn new() -> Self {
        Self {
            enabled: true,
            blade_count: 6,
            rotation: 0.0,
            spike_length: 0.15,
            intensity: 0.4,
            color: [1.0, 1.0, 1.0],
            width: 0.002,
            threshold: 1.5,
            chromatic_separation: 0.01,
        }
    }

    /// Returns the angles of the diffraction spikes.
    ///
    /// A star-burst produces spikes at angles determined by the aperture blades.
    /// An even number of blades produces N/2 spikes (opposite pairs merge).
    /// An odd number produces N spikes.
    pub fn spike_angles(&self) -> Vec<f32> {
        let num_spikes = if self.blade_count % 2 == 0 {
            self.blade_count / 2
        } else {
            self.blade_count
        };

        let angle_step = PI / num_spikes as f32;
        let mut angles = Vec::with_capacity(num_spikes as usize);

        for i in 0..num_spikes {
            angles.push(self.rotation + i as f32 * angle_step);
        }

        angles
    }

    /// Evaluates the star-burst intensity at a given direction from a bright point.
    pub fn evaluate(&self, angle: f32, distance: f32) -> f32 {
        if distance > self.spike_length || distance < 0.001 {
            return 0.0;
        }

        let spike_angles = self.spike_angles();
        let mut max_intensity = 0.0f32;

        for spike_angle in &spike_angles {
            // Distance from this spike's axis.
            let angle_diff = angle_distance(angle, *spike_angle);
            let angular_width = self.width / distance.max(0.001);
            let angular_factor = (-angle_diff.powi(2) / (2.0 * angular_width * angular_width)).exp();

            // Distance falloff.
            let dist_factor = (1.0 - distance / self.spike_length).max(0.0);
            let intensity = angular_factor * dist_factor * self.intensity;

            if intensity > max_intensity {
                max_intensity = intensity;
            }
        }

        max_intensity
    }
}

impl Default for StarBurstConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Shortest angular distance between two angles.
fn angle_distance(a: f32, b: f32) -> f32 {
    let diff = (a - b) % PI;
    if diff > PI / 2.0 {
        PI - diff
    } else if diff < -PI / 2.0 {
        PI + diff
    } else {
        diff.abs()
    }
}

// ---------------------------------------------------------------------------
// Lens distortion
// ---------------------------------------------------------------------------

/// Lens distortion model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionModel {
    /// Brown-Conrady radial distortion model.
    BrownConrady,
    /// Division model (simpler, invertible).
    Division,
    /// Polynomial model.
    Polynomial,
}

/// Lens distortion configuration.
#[derive(Debug, Clone)]
pub struct LensDistortion {
    /// Whether distortion is enabled.
    pub enabled: bool,
    /// Distortion model.
    pub model: DistortionModel,
    /// Radial distortion coefficient k1 (negative = barrel, positive = pincushion).
    pub k1: f32,
    /// Radial distortion coefficient k2.
    pub k2: f32,
    /// Radial distortion coefficient k3.
    pub k3: f32,
    /// Tangential distortion coefficients.
    pub p1: f32,
    pub p2: f32,
    /// Chromatic aberration strength (separate R/G/B distortion).
    pub chromatic_aberration: f32,
    /// Vignetting intensity [0, 1].
    pub vignette_intensity: f32,
    /// Vignetting falloff exponent.
    pub vignette_falloff: f32,
}

impl LensDistortion {
    /// Creates default lens distortion (slight barrel distortion).
    pub fn new() -> Self {
        Self {
            enabled: true,
            model: DistortionModel::BrownConrady,
            k1: -0.1,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            chromatic_aberration: 0.005,
            vignette_intensity: 0.3,
            vignette_falloff: 2.0,
        }
    }

    /// Creates no distortion.
    pub fn none() -> Self {
        Self {
            enabled: false,
            ..Self::new()
        }
    }

    /// Applies radial distortion to a UV coordinate.
    ///
    /// # Arguments
    /// * `uv` — Input UV [0, 1].
    ///
    /// # Returns
    /// Distorted UV.
    pub fn distort_uv(&self, uv: [f32; 2]) -> [f32; 2] {
        if !self.enabled {
            return uv;
        }

        // Centre UV at origin.
        let x = uv[0] * 2.0 - 1.0;
        let y = uv[1] * 2.0 - 1.0;

        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let (dx, dy) = match self.model {
            DistortionModel::BrownConrady => {
                let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
                let tx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
                let ty = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
                (x * radial + tx, y * radial + ty)
            }
            DistortionModel::Division => {
                let d = 1.0 + self.k1 * r2;
                if d.abs() > 1e-6 {
                    (x / d, y / d)
                } else {
                    (x, y)
                }
            }
            DistortionModel::Polynomial => {
                let radial = 1.0 + self.k1 * r2 + self.k2 * r4;
                (x * radial, y * radial)
            }
        };

        [(dx + 1.0) * 0.5, (dy + 1.0) * 0.5]
    }

    /// Computes the vignette factor at a given UV.
    pub fn vignette(&self, uv: [f32; 2]) -> f32 {
        if self.vignette_intensity <= 0.0 {
            return 1.0;
        }
        let x = uv[0] * 2.0 - 1.0;
        let y = uv[1] * 2.0 - 1.0;
        let r = (x * x + y * y).sqrt();
        let v = 1.0 - self.vignette_intensity * r.powf(self.vignette_falloff);
        v.clamp(0.0, 1.0)
    }

    /// Applies distortion and vignetting to a colour buffer.
    pub fn apply(
        &self,
        input: &[[f32; 3]],
        width: u32,
        height: u32,
    ) -> Vec<[f32; 3]> {
        let total = (width * height) as usize;
        let mut output = vec![[0.0f32; 3]; total];

        for y in 0..height {
            for x in 0..width {
                let uv = [
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                ];

                let distorted = self.distort_uv(uv);
                let vignette = self.vignette(uv);

                let idx = (y * width + x) as usize;

                // Sample with chromatic aberration.
                if self.chromatic_aberration > 0.0 {
                    let centre = [0.5f32, 0.5];
                    let dir = [distorted[0] - centre[0], distorted[1] - centre[1]];
                    let ca = self.chromatic_aberration;

                    let uv_r = [distorted[0] + dir[0] * ca, distorted[1] + dir[1] * ca];
                    let uv_g = distorted;
                    let uv_b = [distorted[0] - dir[0] * ca, distorted[1] - dir[1] * ca];

                    output[idx] = [
                        sample_channel(input, width, height, uv_r, 0) * vignette,
                        sample_channel(input, width, height, uv_g, 1) * vignette,
                        sample_channel(input, width, height, uv_b, 2) * vignette,
                    ];
                } else {
                    let sampled = sample_bilinear(input, width, height, distorted);
                    output[idx] = [
                        sampled[0] * vignette,
                        sampled[1] * vignette,
                        sampled[2] * vignette,
                    ];
                }
            }
        }

        output
    }
}

impl Default for LensDistortion {
    fn default() -> Self {
        Self::new()
    }
}

/// Samples a single colour channel with bilinear interpolation.
fn sample_channel(buffer: &[[f32; 3]], width: u32, height: u32, uv: [f32; 2], channel: usize) -> f32 {
    let u = uv[0].clamp(0.0, 1.0);
    let v = uv[1].clamp(0.0, 1.0);
    let fx = u * (width - 1) as f32;
    let fy = v * (height - 1) as f32;
    let x0 = (fx as u32).min(width - 2);
    let y0 = (fy as u32).min(height - 2);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let s00 = buffer[(y0 * width + x0) as usize][channel];
    let s10 = buffer[(y0 * width + x0 + 1) as usize][channel];
    let s01 = buffer[((y0 + 1) * width + x0) as usize][channel];
    let s11 = buffer[((y0 + 1) * width + x0 + 1) as usize][channel];

    let h0 = s00 + (s10 - s00) * tx;
    let h1 = s01 + (s11 - s01) * tx;
    h0 + (h1 - h0) * ty
}

/// Bilinear sampling of an RGB buffer.
fn sample_bilinear(buffer: &[[f32; 3]], width: u32, height: u32, uv: [f32; 2]) -> [f32; 3] {
    [
        sample_channel(buffer, width, height, uv, 0),
        sample_channel(buffer, width, height, uv, 1),
        sample_channel(buffer, width, height, uv, 2),
    ]
}

// ---------------------------------------------------------------------------
// Bloom threshold
// ---------------------------------------------------------------------------

/// Extracts bright pixels from a colour buffer.
///
/// # Arguments
/// * `input` — HDR colour buffer.
/// * `threshold` — Minimum luminance for extraction.
/// * `soft_knee` — Soft transition width around the threshold.
///
/// # Returns
/// Bright-pass buffer (pixels below threshold are black).
pub fn bright_pass(input: &[[f32; 3]], threshold: f32, soft_knee: f32) -> Vec<[f32; 3]> {
    input
        .iter()
        .map(|pixel| {
            let luminance = pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722;
            let contribution = if soft_knee > 0.0 {
                let knee = threshold - soft_knee;
                let soft = (luminance - knee).max(0.0);
                let soft = (soft * soft) / (4.0 * soft_knee + 1e-6);
                if luminance > threshold {
                    luminance - threshold
                } else {
                    soft
                }
            } else {
                (luminance - threshold).max(0.0)
            };

            if contribution > 0.0 {
                let scale = contribution / luminance.max(1e-6);
                [pixel[0] * scale, pixel[1] * scale, pixel[2] * scale]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LensEffects (top-level)
// ---------------------------------------------------------------------------

/// Top-level lens effects system.
#[derive(Debug, Clone)]
pub struct LensEffects {
    /// Lens flare configuration.
    pub flare: LensFlareConfig,
    /// Anamorphic streak configuration.
    pub streak: AnamorphicStreakConfig,
    /// Lens dirt configuration.
    pub dirt: LensDirtConfig,
    /// Star-burst configuration.
    pub starburst: StarBurstConfig,
    /// Lens distortion.
    pub distortion: LensDistortion,
    /// Bloom threshold.
    pub bloom_threshold: f32,
    /// Bloom soft knee.
    pub bloom_soft_knee: f32,
    /// Whether lens effects are enabled globally.
    pub enabled: bool,
}

impl LensEffects {
    /// Creates default lens effects.
    pub fn new() -> Self {
        Self {
            flare: LensFlareConfig::default(),
            streak: AnamorphicStreakConfig::default(),
            dirt: LensDirtConfig::default(),
            starburst: StarBurstConfig::default(),
            distortion: LensDistortion::default(),
            bloom_threshold: 1.0,
            bloom_soft_knee: 0.5,
            enabled: true,
        }
    }

    /// Creates cinematic lens effects.
    pub fn cinematic() -> Self {
        Self {
            flare: LensFlareConfig::cinematic(),
            streak: AnamorphicStreakConfig::new(),
            dirt: LensDirtConfig::new(),
            starburst: StarBurstConfig::new(),
            distortion: LensDistortion::new(),
            bloom_threshold: 0.8,
            bloom_soft_knee: 0.3,
            enabled: true,
        }
    }
}

impl Default for LensEffects {
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
    fn test_flare_ghost_disc() {
        let ghost = FlareGhost::disc(0.5, 1.0, [1.0, 1.0, 1.0], 1.0);
        assert!((ghost.evaluate(0.0) - 1.0).abs() < 0.01);
        assert!((ghost.evaluate(1.0) - 0.0).abs() < 0.01);
        assert!(ghost.evaluate(0.5) > 0.0);
    }

    #[test]
    fn test_flare_ghost_ring() {
        let ghost = FlareGhost::ring(0.5, 1.0, [1.0, 1.0, 1.0], 1.0, 0.2);
        // Centre of ring should have low intensity.
        assert!(ghost.evaluate(0.0) < 0.01);
        // Edge of ring (around 0.9) should have high intensity.
        let ring_val = ghost.evaluate(0.85);
        assert!(ring_val > 0.0, "Ring should have value near edge: {ring_val}");
    }

    #[test]
    fn test_edge_fade() {
        let config = LensFlareConfig::new();
        let centre_fade = config.edge_fade_factor([0.5, 0.5]);
        assert!((centre_fade - 1.0).abs() < 0.01, "Centre should be unfaded");

        let edge_fade = config.edge_fade_factor([0.99, 0.5]);
        assert!(edge_fade < 1.0, "Edge should be faded: {edge_fade}");
    }

    #[test]
    fn test_bright_pass() {
        let input = vec![
            [0.1, 0.1, 0.1], // Below threshold.
            [2.0, 2.0, 2.0], // Above threshold.
        ];
        let result = bright_pass(&input, 1.0, 0.0);
        assert!(result[0][0] < 0.01, "Below threshold should be black");
        assert!(result[1][0] > 0.0, "Above threshold should be bright");
    }

    #[test]
    fn test_bright_pass_soft_knee() {
        let input = vec![[0.9, 0.9, 0.9]]; // Near threshold.
        let result_hard = bright_pass(&input, 1.0, 0.0);
        let result_soft = bright_pass(&input, 1.0, 0.5);
        assert!(result_soft[0][0] > result_hard[0][0], "Soft knee should pass more");
    }

    #[test]
    fn test_lens_distortion_identity() {
        let dist = LensDistortion {
            enabled: true,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            ..LensDistortion::new()
        };
        let uv = dist.distort_uv([0.3, 0.7]);
        assert!((uv[0] - 0.3).abs() < 0.01, "Zero distortion should be identity");
        assert!((uv[1] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_lens_distortion_barrel() {
        let dist = LensDistortion::new(); // k1 = -0.1 (barrel).
        let centre = dist.distort_uv([0.5, 0.5]);
        assert!((centre[0] - 0.5).abs() < 0.01, "Centre should be unchanged");

        let corner = dist.distort_uv([0.0, 0.0]);
        // Barrel distortion moves corners inward.
        assert!(corner[0] > 0.0, "Corner should move inward with barrel");
    }

    #[test]
    fn test_vignette() {
        let dist = LensDistortion::new();
        let centre = dist.vignette([0.5, 0.5]);
        assert!((centre - 1.0).abs() < 0.1, "Centre should have no vignette");

        let corner = dist.vignette([0.0, 0.0]);
        assert!(corner < centre, "Corner should be darker");
    }

    #[test]
    fn test_starburst_spike_angles() {
        let sb = StarBurstConfig { blade_count: 6, ..StarBurstConfig::new() };
        let angles = sb.spike_angles();
        assert_eq!(angles.len(), 3, "6 blades should produce 3 spikes");

        let sb_odd = StarBurstConfig { blade_count: 5, ..StarBurstConfig::new() };
        let angles_odd = sb_odd.spike_angles();
        assert_eq!(angles_odd.len(), 5, "5 blades should produce 5 spikes");
    }

    #[test]
    fn test_anamorphic_streak_direction() {
        let streak = AnamorphicStreakConfig::new();
        let dir = streak.direction();
        assert!((dir[0] - 1.0).abs() < 0.01, "Horizontal streak");

        let vertical = AnamorphicStreakConfig::vertical();
        let dir_v = vertical.direction();
        assert!(dir_v[1].abs() > 0.9, "Vertical streak");
    }

    #[test]
    fn test_dirt_texture_generation() {
        let config = LensDirtConfig::new();
        let dirt = generate_dirt_texture(&config);
        assert_eq!(dirt.len(), (config.resolution * config.resolution) as usize);
        assert!(dirt.iter().all(|v| *v >= 0.0 && *v <= 1.0));
        // Should have some non-zero values.
        assert!(dirt.iter().any(|v| *v > 0.01));
    }

    #[test]
    fn test_lens_effects_creation() {
        let effects = LensEffects::new();
        assert!(effects.enabled);
        assert!(effects.flare.enabled);
        assert!(effects.streak.enabled);
    }

    #[test]
    fn test_cinematic_effects() {
        let effects = LensEffects::cinematic();
        assert!(effects.flare.intensity > 1.0);
    }

    #[test]
    fn test_hsv_to_rgb_red() {
        let r = super::hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((r[0] - 1.0).abs() < 0.01);
        assert!(r[1] < 0.01);
        assert!(r[2] < 0.01);
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    super::hsv_to_rgb(h, s, v)
}
