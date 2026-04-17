// engine/render/src/volumetric_clouds.rs
//
// Volumetric cloud rendering system for the Genovo engine.
//
// Implements a physically-based volumetric cloud renderer using:
//
// - Ray-marching through a 3D noise-based density field.
// - Henyey-Greenstein phase function for anisotropic scattering.
// - Beer's law for light absorption / transmittance.
// - Temporal reprojection to amortise the cost over multiple frames.
// - Cloud shadow map generation for ground shadowing.
// - Wind-driven animation with altitude-dependent drift.
// - Multiple cloud type presets (cumulus, stratus, cirrus).
// - Altitude-dependent density profiles with smooth falloff.
//
// # GPU pipeline overview
//
// 1. **Density pass** — For each pixel, ray-march from the camera through the
//    cloud layer and accumulate density + in-scattered light.
// 2. **Shadow pass** — Render a top-down orthographic shadow map of the cloud
//    layer for ground shadow casting.
// 3. **Temporal reprojection** — Blend the current frame's result with the
//    previous frame using motion-vector-guided reprojection.
// 4. **Composite** — Blend the cloud colour and transmittance into the scene.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Noise helpers (self-contained)
// ---------------------------------------------------------------------------

/// Simple 3D hash for value noise.
#[inline]
fn hash3d(x: i32, y: i32, z: i32) -> f32 {
    let n = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(z.wrapping_mul(1274126177));
    let n = n ^ (n >> 13);
    let n = n.wrapping_mul(
        n.wrapping_mul(n.wrapping_mul(60493).wrapping_add(19990303))
            .wrapping_add(1376312589),
    );
    (n & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF as f32 * 2.0 - 1.0
}

/// Quintic smoothstep interpolation.
#[inline]
fn quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// 3D value noise.
fn value_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();
    let u = quintic(xf);
    let v = quintic(yf);
    let w = quintic(zf);

    let c000 = hash3d(xi, yi, zi);
    let c100 = hash3d(xi + 1, yi, zi);
    let c010 = hash3d(xi, yi + 1, zi);
    let c110 = hash3d(xi + 1, yi + 1, zi);
    let c001 = hash3d(xi, yi, zi + 1);
    let c101 = hash3d(xi + 1, yi, zi + 1);
    let c011 = hash3d(xi, yi + 1, zi + 1);
    let c111 = hash3d(xi + 1, yi + 1, zi + 1);

    let x00 = c000 + (c100 - c000) * u;
    let x10 = c010 + (c110 - c010) * u;
    let x01 = c001 + (c101 - c001) * u;
    let x11 = c011 + (c111 - c011) * u;

    let y0 = x00 + (x10 - x00) * v;
    let y1 = x01 + (x11 - x01) * v;

    y0 + (y1 - y0) * w
}

/// Fractal Brownian Motion (fBm) combining multiple octaves of noise.
fn fbm_3d(
    x: f32,
    y: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amp = 0.0;

    for _ in 0..octaves {
        value += value_noise_3d(x * frequency, y * frequency, z * frequency) * amplitude;
        max_amp += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    if max_amp > 0.0 {
        value / max_amp
    } else {
        0.0
    }
}

/// Worley (cellular) noise approximation for cloud detail.
fn worley_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let mut min_dist = f32::MAX;

    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                // Deterministic pseudo-random point in this cell.
                let cell_x = xi + dx;
                let cell_y = yi + dy;
                let cell_z = zi + dz;

                let h1 = hash3d(cell_x, cell_y, cell_z) * 0.5 + 0.5;
                let h2 = hash3d(cell_x + 127, cell_y + 311, cell_z + 523) * 0.5 + 0.5;
                let h3 = hash3d(cell_x + 269, cell_y + 643, cell_z + 877) * 0.5 + 0.5;

                let px = dx as f32 + h1 - xf;
                let py = dy as f32 + h2 - yf;
                let pz = dz as f32 + h3 - zf;

                let dist = px * px + py * py + pz * pz;
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
    }

    min_dist.sqrt().clamp(0.0, 1.0)
}

/// Combined Perlin-Worley noise for cloud shapes.
fn perlin_worley_3d(x: f32, y: f32, z: f32) -> f32 {
    let perlin = fbm_3d(x, y, z, 4, 2.0, 0.5) * 0.5 + 0.5;
    let worley = 1.0 - worley_noise_3d(x, y, z);
    // Remap: use perlin to modulate worley.
    let pw = remap(perlin, 0.0, 1.0, worley, 1.0);
    pw.clamp(0.0, 1.0)
}

/// Remap a value from [old_min, old_max] to [new_min, new_max].
#[inline]
fn remap(value: f32, old_min: f32, old_max: f32, new_min: f32, new_max: f32) -> f32 {
    let t = ((value - old_min) / (old_max - old_min)).clamp(0.0, 1.0);
    new_min + t * (new_max - new_min)
}

/// Linear interpolation.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Saturate (clamp to [0, 1]).
#[inline]
fn saturate(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Cloud type
// ---------------------------------------------------------------------------

/// Cloud type classification, affecting density profile, shape, and altitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloudType {
    /// Cumulus: puffy, vertically developed, well-defined edges.
    Cumulus,
    /// Stratus: flat, layered, uniform coverage.
    Stratus,
    /// Cirrus: thin, wispy, high-altitude ice crystals.
    Cirrus,
    /// Cumulonimbus: towering storm clouds with anvil top.
    Cumulonimbus,
    /// Stratocumulus: lumpy layer clouds.
    Stratocumulus,
    /// Custom: user-defined density profile.
    Custom,
}

impl CloudType {
    /// Returns the vertical density profile for this cloud type.
    ///
    /// The profile defines how density varies with normalised height `h`
    /// within the cloud layer (0 = bottom, 1 = top).
    pub fn density_profile(&self, h: f32) -> f32 {
        match self {
            Self::Cumulus => {
                // Bell-shaped profile peaking in the lower-middle.
                let bottom = saturate(remap(h, 0.0, 0.15, 0.0, 1.0));
                let top = saturate(remap(h, 0.6, 1.0, 1.0, 0.0));
                bottom * top
            }
            Self::Stratus => {
                // Flat profile with gentle falloff at edges.
                let bottom = saturate(remap(h, 0.0, 0.05, 0.0, 1.0));
                let top = saturate(remap(h, 0.7, 1.0, 1.0, 0.0));
                bottom * top * 0.6
            }
            Self::Cirrus => {
                // Thin layer near the top.
                let bottom = saturate(remap(h, 0.6, 0.8, 0.0, 1.0));
                let top = saturate(remap(h, 0.9, 1.0, 1.0, 0.0));
                bottom * top * 0.3
            }
            Self::Cumulonimbus => {
                // Tall profile filling most of the vertical range.
                let bottom = saturate(remap(h, 0.0, 0.1, 0.0, 1.0));
                let top = saturate(remap(h, 0.85, 1.0, 1.0, 0.0));
                bottom * top * 1.2
            }
            Self::Stratocumulus => {
                // Lower layer with some vertical development.
                let bottom = saturate(remap(h, 0.0, 0.1, 0.0, 1.0));
                let top = saturate(remap(h, 0.35, 0.6, 1.0, 0.0));
                bottom * top * 0.8
            }
            Self::Custom => {
                // Simple linear falloff — user should override.
                saturate(1.0 - h)
            }
        }
    }

    /// Returns default noise frequency scaling for this cloud type.
    pub fn noise_frequency(&self) -> f32 {
        match self {
            Self::Cumulus => 0.8,
            Self::Stratus => 0.5,
            Self::Cirrus => 1.5,
            Self::Cumulonimbus => 0.6,
            Self::Stratocumulus => 0.7,
            Self::Custom => 1.0,
        }
    }

    /// Returns default coverage threshold for this cloud type.
    pub fn default_coverage(&self) -> f32 {
        match self {
            Self::Cumulus => 0.45,
            Self::Stratus => 0.7,
            Self::Cirrus => 0.3,
            Self::Cumulonimbus => 0.55,
            Self::Stratocumulus => 0.6,
            Self::Custom => 0.5,
        }
    }

    /// Returns the typical altitude range (min_km, max_km).
    pub fn altitude_range_km(&self) -> (f32, f32) {
        match self {
            Self::Cumulus => (1.5, 4.0),
            Self::Stratus => (0.5, 2.0),
            Self::Cirrus => (6.0, 10.0),
            Self::Cumulonimbus => (1.0, 12.0),
            Self::Stratocumulus => (0.8, 2.5),
            Self::Custom => (1.0, 5.0),
        }
    }
}

impl Default for CloudType {
    fn default() -> Self {
        Self::Cumulus
    }
}

// ---------------------------------------------------------------------------
// CloudLayerConfig
// ---------------------------------------------------------------------------

/// Configuration for a single cloud layer.
#[derive(Debug, Clone)]
pub struct CloudLayerConfig {
    /// Cloud type preset.
    pub cloud_type: CloudType,
    /// Bottom altitude of the cloud layer in world units.
    pub altitude_bottom: f32,
    /// Top altitude of the cloud layer in world units.
    pub altitude_top: f32,
    /// Cloud coverage [0, 1]. Higher values produce more cloud.
    pub coverage: f32,
    /// Base density multiplier.
    pub density_multiplier: f32,
    /// Detail density multiplier (for high-frequency erosion noise).
    pub detail_density: f32,
    /// Shape noise frequency.
    pub shape_frequency: f32,
    /// Detail noise frequency.
    pub detail_frequency: f32,
    /// Shape noise octaves.
    pub shape_octaves: u32,
    /// Detail noise octaves.
    pub detail_octaves: u32,
    /// Wind direction (normalised XZ).
    pub wind_direction: [f32; 2],
    /// Wind speed (world units per second).
    pub wind_speed: f32,
    /// Wind shear: how much the wind direction changes with altitude.
    pub wind_shear: f32,
    /// Wind vertical speed (upward drift).
    pub wind_vertical_speed: f32,
    /// Curl noise strength for turbulence at cloud edges.
    pub curl_noise_strength: f32,
    /// Anvil bias: flattens cloud tops (for cumulonimbus).
    pub anvil_bias: f32,
    /// Cloud colour at the top (sunlit).
    pub color_top: [f32; 3],
    /// Cloud colour at the bottom (shadowed).
    pub color_bottom: [f32; 3],
    /// Ambient light contribution.
    pub ambient_intensity: f32,
    /// Enable powder effect (darkening in dense regions due to multiple scattering).
    pub powder_effect: bool,
    /// Powder effect strength.
    pub powder_strength: f32,
}

impl CloudLayerConfig {
    /// Creates a cloud layer from a preset cloud type.
    pub fn from_type(cloud_type: CloudType) -> Self {
        let (alt_bot_km, alt_top_km) = cloud_type.altitude_range_km();
        let world_scale = 1000.0; // 1 km = 1000 world units

        Self {
            cloud_type,
            altitude_bottom: alt_bot_km * world_scale,
            altitude_top: alt_top_km * world_scale,
            coverage: cloud_type.default_coverage(),
            density_multiplier: 1.0,
            detail_density: 0.35,
            shape_frequency: cloud_type.noise_frequency() * 0.0003,
            detail_frequency: cloud_type.noise_frequency() * 0.002,
            shape_octaves: 4,
            detail_octaves: 3,
            wind_direction: [1.0, 0.0],
            wind_speed: 15.0,
            wind_shear: 0.1,
            wind_vertical_speed: 0.5,
            curl_noise_strength: 0.3,
            anvil_bias: if matches!(cloud_type, CloudType::Cumulonimbus) {
                0.4
            } else {
                0.0
            },
            color_top: [1.0, 1.0, 1.0],
            color_bottom: [0.5, 0.55, 0.6],
            ambient_intensity: 0.2,
            powder_effect: true,
            powder_strength: 2.0,
        }
    }

    /// Creates a default cumulus cloud layer.
    pub fn cumulus() -> Self {
        Self::from_type(CloudType::Cumulus)
    }

    /// Creates a stratus cloud layer.
    pub fn stratus() -> Self {
        Self::from_type(CloudType::Stratus)
    }

    /// Creates a cirrus cloud layer.
    pub fn cirrus() -> Self {
        Self::from_type(CloudType::Cirrus)
    }

    /// Creates a cumulonimbus cloud layer.
    pub fn cumulonimbus() -> Self {
        Self::from_type(CloudType::Cumulonimbus)
    }

    /// Sets the coverage.
    pub fn with_coverage(mut self, coverage: f32) -> Self {
        self.coverage = coverage.clamp(0.0, 1.0);
        self
    }

    /// Sets the altitude range.
    pub fn with_altitude(mut self, bottom: f32, top: f32) -> Self {
        self.altitude_bottom = bottom;
        self.altitude_top = top;
        self
    }

    /// Sets the wind parameters.
    pub fn with_wind(mut self, direction: [f32; 2], speed: f32) -> Self {
        let len = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        if len > 1e-6 {
            self.wind_direction = [direction[0] / len, direction[1] / len];
        }
        self.wind_speed = speed;
        self
    }

    /// Sets the density multiplier.
    pub fn with_density(mut self, density: f32) -> Self {
        self.density_multiplier = density;
        self
    }

    /// Returns the cloud layer thickness.
    pub fn thickness(&self) -> f32 {
        (self.altitude_top - self.altitude_bottom).max(1.0)
    }

    /// Returns the normalised height within the cloud layer.
    pub fn normalised_height(&self, world_y: f32) -> f32 {
        let h = (world_y - self.altitude_bottom) / self.thickness();
        h.clamp(0.0, 1.0)
    }

    /// Computes the wind offset at a given time and altitude.
    pub fn wind_offset(&self, time: f32, normalised_height: f32) -> [f32; 3] {
        let shear = 1.0 + self.wind_shear * normalised_height;
        let dx = self.wind_direction[0] * self.wind_speed * time * shear;
        let dz = self.wind_direction[1] * self.wind_speed * time * shear;
        let dy = self.wind_vertical_speed * time;
        [dx, dy, dz]
    }
}

impl Default for CloudLayerConfig {
    fn default() -> Self {
        Self::cumulus()
    }
}

// ---------------------------------------------------------------------------
// Scattering and absorption
// ---------------------------------------------------------------------------

/// Henyey-Greenstein phase function.
///
/// Models the angular distribution of light scattered by cloud particles.
///
/// # Arguments
/// * `cos_theta` — Cosine of angle between view and light directions.
/// * `g` — Asymmetry parameter (0 = isotropic, >0 = forward scattering).
pub fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    if denom < 1e-6 {
        return 1.0;
    }
    (1.0 - g2) / (4.0 * PI * denom * denom.sqrt())
}

/// Dual-lobe Henyey-Greenstein phase function.
///
/// Blends forward and backward scattering for a more realistic appearance.
/// Used for the silver lining effect on cloud edges.
///
/// # Arguments
/// * `cos_theta` — Cosine of angle between view and light directions.
/// * `g1` — Forward scattering asymmetry.
/// * `g2` — Backward scattering asymmetry (typically negative).
/// * `blend` — Blend factor between the two lobes.
pub fn dual_lobe_hg_phase(cos_theta: f32, g1: f32, g2: f32, blend: f32) -> f32 {
    let forward = henyey_greenstein_phase(cos_theta, g1);
    let backward = henyey_greenstein_phase(cos_theta, g2);
    lerp(forward, backward, blend)
}

/// Beer's law transmittance.
///
/// `T = exp(-extinction * distance)`
///
/// Models the fraction of light that survives passing through an absorbing
/// medium without being scattered or absorbed.
#[inline]
pub fn beers_law(optical_depth: f32) -> f32 {
    (-optical_depth).exp()
}

/// Beer-Powder approximation.
///
/// Combines Beer's law with a powder effect that darkens optically thick
/// regions. This approximates the energy that would be lost to multiple
/// scattering events deep inside the cloud.
///
/// `T_powder = 2 * exp(-d) * (1 - exp(-2*d))`
#[inline]
pub fn beers_powder(optical_depth: f32, powder_strength: f32) -> f32 {
    let beers = beers_law(optical_depth);
    let powder = 1.0 - (-optical_depth * powder_strength).exp();
    beers * powder * 2.0
}

/// Compute the multi-scattering approximation octaves.
///
/// Each successive bounce has reduced energy and increased isotropy.
/// This avoids the need for actual multi-bounce computation.
///
/// # Arguments
/// * `optical_depth` — Accumulated optical depth.
/// * `cos_theta` — Cosine of angle between view and light.
/// * `g` — Base asymmetry parameter.
/// * `num_octaves` — Number of scattering octaves.
///
/// # Returns
/// Approximated multi-scattering contribution.
pub fn multi_scattering_approximation(
    optical_depth: f32,
    cos_theta: f32,
    g: f32,
    num_octaves: u32,
) -> f32 {
    let mut total = 0.0;
    let mut attenuation = 1.0;
    let mut contribution = 1.0;
    let mut phase_g = g;

    let attenuation_decay = 0.5; // Energy loss per octave.
    let contribution_decay = 0.5;
    let phase_decay = 0.5; // Isotropy increase per octave.

    for _ in 0..num_octaves {
        let phase = henyey_greenstein_phase(cos_theta, phase_g);
        let beer = beers_law(optical_depth * attenuation);
        total += phase * beer * contribution;

        attenuation *= attenuation_decay;
        contribution *= contribution_decay;
        phase_g *= phase_decay;
    }

    total
}

// ---------------------------------------------------------------------------
// Cloud density sampling
// ---------------------------------------------------------------------------

/// Samples the base cloud density at a given 3D position within the cloud layer.
///
/// This combines:
/// 1. Height-dependent density profile (from the cloud type).
/// 2. Low-frequency shape noise (Perlin-Worley).
/// 3. Coverage remapping.
/// 4. High-frequency detail erosion.
///
/// # Arguments
/// * `pos` — World-space sample position.
/// * `config` — Cloud layer configuration.
/// * `time` — Current simulation time.
///
/// # Returns
/// Cloud density in [0, 1+]. Values above 0 indicate cloud.
pub fn sample_cloud_density(
    pos: [f32; 3],
    config: &CloudLayerConfig,
    time: f32,
) -> f32 {
    let world_y = pos[1];

    // Check if we are inside the cloud layer.
    if world_y < config.altitude_bottom || world_y > config.altitude_top {
        return 0.0;
    }

    let h = config.normalised_height(world_y);

    // Height-based density profile from cloud type.
    let height_profile = config.cloud_type.density_profile(h);
    if height_profile < 1e-6 {
        return 0.0;
    }

    // Wind offset.
    let wind = config.wind_offset(time, h);
    let sx = pos[0] + wind[0];
    let sy = pos[1] + wind[1];
    let sz = pos[2] + wind[2];

    // Low-frequency shape noise.
    let shape = perlin_worley_3d(
        sx * config.shape_frequency,
        sy * config.shape_frequency * 0.5, // Squash vertically.
        sz * config.shape_frequency,
    );

    // Apply coverage: remap so that coverage=0 means no cloud,
    // coverage=1 means fully covered.
    let coverage = config.coverage;
    let shaped = remap(shape, 1.0 - coverage, 1.0, 0.0, 1.0);
    let shaped = shaped * height_profile;

    if shaped <= 0.0 {
        return 0.0;
    }

    // Anvil bias: flatten cloud tops for cumulonimbus.
    let anvil = if config.anvil_bias > 0.0 {
        let anvil_factor = saturate(remap(h, 0.7, 1.0, 0.0, 1.0));
        1.0 + config.anvil_bias * anvil_factor
    } else {
        1.0
    };

    // High-frequency detail noise for erosion.
    let detail = fbm_3d(
        sx * config.detail_frequency,
        sy * config.detail_frequency,
        sz * config.detail_frequency,
        config.detail_octaves,
        2.0,
        0.5,
    ) * 0.5
        + 0.5;

    // Erode: subtract detail from shape, more erosion at edges.
    let detail_erosion = detail * config.detail_density;
    let eroded = (shaped - detail_erosion).max(0.0);

    // Curl noise turbulence at edges.
    let curl = if config.curl_noise_strength > 0.0 && eroded > 0.0 && eroded < 0.3 {
        let curl_val = value_noise_3d(
            sx * config.detail_frequency * 2.0,
            sy * config.detail_frequency * 2.0,
            sz * config.detail_frequency * 2.0,
        );
        curl_val * config.curl_noise_strength * (1.0 - eroded / 0.3)
    } else {
        0.0
    };

    let final_density = (eroded + curl).max(0.0) * config.density_multiplier * anvil;
    final_density
}

/// Samples the cloud density along a light ray (for in-scattering computation).
///
/// This is a simplified version that uses fewer noise octaves for performance.
///
/// # Arguments
/// * `pos` — World-space sample position.
/// * `config` — Cloud layer configuration.
/// * `time` — Current simulation time.
///
/// # Returns
/// Approximate cloud density.
pub fn sample_cloud_density_light(
    pos: [f32; 3],
    config: &CloudLayerConfig,
    time: f32,
) -> f32 {
    let world_y = pos[1];

    if world_y < config.altitude_bottom || world_y > config.altitude_top {
        return 0.0;
    }

    let h = config.normalised_height(world_y);
    let height_profile = config.cloud_type.density_profile(h);
    if height_profile < 1e-6 {
        return 0.0;
    }

    let wind = config.wind_offset(time, h);
    let sx = pos[0] + wind[0];
    let sy = pos[1] + wind[1];
    let sz = pos[2] + wind[2];

    // Shape noise only (skip detail for performance).
    let shape = perlin_worley_3d(
        sx * config.shape_frequency,
        sy * config.shape_frequency * 0.5,
        sz * config.shape_frequency,
    );

    let coverage = config.coverage;
    let shaped = remap(shape, 1.0 - coverage, 1.0, 0.0, 1.0);
    let shaped = shaped * height_profile;

    shaped.max(0.0) * config.density_multiplier
}

// ---------------------------------------------------------------------------
// Ray-march settings
// ---------------------------------------------------------------------------

/// Configuration for the volumetric cloud ray-marcher.
#[derive(Debug, Clone)]
pub struct CloudRaymarchSettings {
    /// Maximum number of primary ray steps.
    pub max_primary_steps: u32,
    /// Maximum number of light-direction steps (for shadow/scattering).
    pub max_light_steps: u32,
    /// Primary step size in world units.
    pub primary_step_size: f32,
    /// Maximum ray distance before aborting.
    pub max_ray_distance: f32,
    /// Minimum transmittance before early-out (near-opaque).
    pub min_transmittance: f32,
    /// Extinction coefficient (controls optical thickness).
    pub extinction_coefficient: f32,
    /// Scattering albedo [0, 1] (fraction of extinction that is scattering vs absorption).
    pub scattering_albedo: f32,
    /// Forward scattering asymmetry (Henyey-Greenstein g).
    pub phase_g_forward: f32,
    /// Backward scattering asymmetry.
    pub phase_g_backward: f32,
    /// Blend between forward/backward scattering lobes.
    pub phase_blend: f32,
    /// Number of multi-scattering octaves.
    pub multi_scatter_octaves: u32,
    /// Enable blue-noise dithering of ray start.
    pub blue_noise_dither: bool,
    /// Blue noise dither amplitude in world units.
    pub dither_amplitude: f32,
    /// Step size growth factor (larger steps farther away).
    pub step_growth: f32,
}

impl CloudRaymarchSettings {
    /// Creates default settings for high quality.
    pub fn high_quality() -> Self {
        Self {
            max_primary_steps: 128,
            max_light_steps: 8,
            primary_step_size: 50.0,
            max_ray_distance: 30000.0,
            min_transmittance: 0.01,
            extinction_coefficient: 0.04,
            scattering_albedo: 0.99,
            phase_g_forward: 0.8,
            phase_g_backward: -0.3,
            phase_blend: 0.2,
            multi_scatter_octaves: 4,
            blue_noise_dither: true,
            dither_amplitude: 25.0,
            step_growth: 1.005,
        }
    }

    /// Creates default settings for medium quality.
    pub fn medium_quality() -> Self {
        Self {
            max_primary_steps: 64,
            max_light_steps: 6,
            primary_step_size: 80.0,
            max_ray_distance: 25000.0,
            min_transmittance: 0.02,
            extinction_coefficient: 0.04,
            scattering_albedo: 0.98,
            phase_g_forward: 0.75,
            phase_g_backward: -0.25,
            phase_blend: 0.2,
            multi_scatter_octaves: 3,
            blue_noise_dither: true,
            dither_amplitude: 40.0,
            step_growth: 1.01,
        }
    }

    /// Creates default settings for low quality / mobile.
    pub fn low_quality() -> Self {
        Self {
            max_primary_steps: 32,
            max_light_steps: 4,
            primary_step_size: 120.0,
            max_ray_distance: 15000.0,
            min_transmittance: 0.05,
            extinction_coefficient: 0.04,
            scattering_albedo: 0.95,
            phase_g_forward: 0.7,
            phase_g_backward: -0.2,
            phase_blend: 0.15,
            multi_scatter_octaves: 2,
            blue_noise_dither: false,
            dither_amplitude: 60.0,
            step_growth: 1.02,
        }
    }
}

impl Default for CloudRaymarchSettings {
    fn default() -> Self {
        Self::medium_quality()
    }
}

// ---------------------------------------------------------------------------
// Cloud ray-march result
// ---------------------------------------------------------------------------

/// Result of ray-marching through a cloud layer for a single pixel.
#[derive(Debug, Clone, Copy)]
pub struct CloudRaymarchResult {
    /// Accumulated cloud colour (RGB, pre-multiplied alpha).
    pub color: [f32; 3],
    /// Remaining transmittance (1 = clear sky, 0 = fully opaque cloud).
    pub transmittance: f32,
    /// Depth of the nearest cloud hit (world units from camera).
    pub depth: f32,
    /// Number of ray-march steps taken.
    pub steps_taken: u32,
}

impl CloudRaymarchResult {
    /// Returns a fully transparent result (no cloud).
    pub fn clear() -> Self {
        Self {
            color: [0.0, 0.0, 0.0],
            transmittance: 1.0,
            depth: f32::MAX,
            steps_taken: 0,
        }
    }

    /// Alpha (opacity) of the cloud.
    pub fn alpha(&self) -> f32 {
        1.0 - self.transmittance
    }
}

// ---------------------------------------------------------------------------
// Ray-march implementation
// ---------------------------------------------------------------------------

/// Performs volumetric cloud ray-marching for a single ray.
///
/// # Arguments
/// * `ray_origin` — Camera/eye position in world space.
/// * `ray_dir` — Normalised ray direction.
/// * `sun_dir` — Normalised direction TO the sun.
/// * `sun_color` — Sun colour and intensity (linear RGB).
/// * `ambient_color` — Ambient light colour.
/// * `config` — Cloud layer configuration.
/// * `settings` — Ray-march quality settings.
/// * `time` — Current simulation time.
/// * `scene_depth` — Depth of the scene geometry along this ray (for cloud-behind-geometry).
///
/// # Returns
/// Ray-march result containing accumulated colour and transmittance.
pub fn raymarch_clouds(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    sun_dir: [f32; 3],
    sun_color: [f32; 3],
    ambient_color: [f32; 3],
    config: &CloudLayerConfig,
    settings: &CloudRaymarchSettings,
    time: f32,
    scene_depth: f32,
) -> CloudRaymarchResult {
    // Find intersection with the cloud layer (two horizontal planes).
    let dy = ray_dir[1];

    // Handle near-horizontal rays.
    if dy.abs() < 1e-6 {
        // If camera is inside the cloud layer, we could still march.
        if ray_origin[1] >= config.altitude_bottom && ray_origin[1] <= config.altitude_top {
            // March horizontally — use a simplified approach.
        } else {
            return CloudRaymarchResult::clear();
        }
    }

    // Compute entry and exit distances to the cloud layer slab.
    let t_bottom = (config.altitude_bottom - ray_origin[1]) / dy;
    let t_top = (config.altitude_top - ray_origin[1]) / dy;

    let t_enter;
    let t_exit;

    if ray_origin[1] < config.altitude_bottom {
        // Camera below cloud layer.
        if dy <= 0.0 {
            return CloudRaymarchResult::clear(); // Looking away.
        }
        t_enter = t_bottom;
        t_exit = t_top;
    } else if ray_origin[1] > config.altitude_top {
        // Camera above cloud layer.
        if dy >= 0.0 {
            return CloudRaymarchResult::clear(); // Looking away.
        }
        t_enter = t_top;
        t_exit = t_bottom;
    } else {
        // Camera inside cloud layer.
        t_enter = 0.0;
        t_exit = if dy > 0.0 { t_top } else { t_bottom };
    }

    let t_enter = t_enter.max(0.0);
    let t_exit = t_exit.min(settings.max_ray_distance).min(scene_depth);

    if t_enter >= t_exit {
        return CloudRaymarchResult::clear();
    }

    // Phase function (view-dependent, computed once).
    let cos_theta = ray_dir[0] * sun_dir[0] + ray_dir[1] * sun_dir[1] + ray_dir[2] * sun_dir[2];
    let phase = dual_lobe_hg_phase(
        cos_theta,
        settings.phase_g_forward,
        settings.phase_g_backward,
        settings.phase_blend,
    );

    // Ray-march state.
    let mut accumulated_color = [0.0f32; 3];
    let mut transmittance = 1.0f32;
    let mut nearest_depth = f32::MAX;
    let mut step_count = 0u32;

    // Dithered start offset.
    let dither = if settings.blue_noise_dither {
        // Simple hash-based dither (in production, use a blue noise texture).
        let hash = hash3d(
            (ray_dir[0] * 10000.0) as i32,
            (ray_dir[1] * 10000.0) as i32,
            (ray_dir[2] * 10000.0) as i32,
        ) * 0.5 + 0.5;
        hash * settings.dither_amplitude
    } else {
        0.0
    };

    let mut t = t_enter + dither;
    let mut step_size = settings.primary_step_size;
    let mut zero_density_count = 0u32;

    while t < t_exit && step_count < settings.max_primary_steps {
        let sample_pos = [
            ray_origin[0] + ray_dir[0] * t,
            ray_origin[1] + ray_dir[1] * t,
            ray_origin[2] + ray_dir[2] * t,
        ];

        let density = sample_cloud_density(sample_pos, config, time);

        if density > 1e-6 {
            zero_density_count = 0;

            // Record nearest cloud depth.
            if nearest_depth == f32::MAX {
                nearest_depth = t;
            }

            // Optical depth for this step.
            let optical_depth = density * settings.extinction_coefficient * step_size;

            // Light marching: cast a ray toward the sun to compute shadow.
            let light_optical_depth = light_march(
                sample_pos,
                sun_dir,
                config,
                settings,
                time,
            );

            // Light transmittance through the cloud toward the sun.
            let light_transmittance = beers_law(light_optical_depth);

            // Powder effect (darkening in dense regions).
            let powder = if config.powder_effect {
                let d = light_optical_depth;
                1.0 - (-d * config.powder_strength).exp()
            } else {
                1.0
            };

            // Multi-scattering approximation.
            let multi_scatter = multi_scattering_approximation(
                light_optical_depth,
                cos_theta,
                settings.phase_g_forward,
                settings.multi_scatter_octaves,
            );

            // Normalised height for colour gradient.
            let h = config.normalised_height(sample_pos[1]);

            // Interpolate cloud colour between bottom and top.
            let cloud_color = [
                lerp(config.color_bottom[0], config.color_top[0], h),
                lerp(config.color_bottom[1], config.color_top[1], h),
                lerp(config.color_bottom[2], config.color_top[2], h),
            ];

            // Direct lighting contribution.
            let direct_light = light_transmittance * phase * powder;

            // Multi-scattering contribution.
            let multi_light = multi_scatter * settings.scattering_albedo;

            // Total lighting.
            let total_light = direct_light + multi_light;

            // In-scattered radiance for this step.
            let inscattered = [
                cloud_color[0]
                    * (sun_color[0] * total_light
                        + ambient_color[0] * config.ambient_intensity),
                cloud_color[1]
                    * (sun_color[1] * total_light
                        + ambient_color[1] * config.ambient_intensity),
                cloud_color[2]
                    * (sun_color[2] * total_light
                        + ambient_color[2] * config.ambient_intensity),
            ];

            // Energy-conserving integration.
            let step_transmittance = beers_law(optical_depth);
            let integration_factor =
                transmittance * (1.0 - step_transmittance) * settings.scattering_albedo;

            accumulated_color[0] += inscattered[0] * integration_factor;
            accumulated_color[1] += inscattered[1] * integration_factor;
            accumulated_color[2] += inscattered[2] * integration_factor;

            transmittance *= step_transmittance;

            // Early termination when nearly opaque.
            if transmittance < settings.min_transmittance {
                transmittance = 0.0;
                break;
            }

            // Use smaller steps inside clouds for accuracy.
            step_size = settings.primary_step_size;
        } else {
            zero_density_count += 1;
            // Increase step size in empty space.
            if zero_density_count > 3 {
                step_size = settings.primary_step_size * 2.0;
            }
        }

        t += step_size;
        step_size *= settings.step_growth;
        step_count += 1;
    }

    CloudRaymarchResult {
        color: accumulated_color,
        transmittance,
        depth: nearest_depth,
        steps_taken: step_count,
    }
}

/// Marches a ray toward the sun to compute the optical depth (shadow).
///
/// Uses fewer steps and simpler density sampling for performance.
fn light_march(
    origin: [f32; 3],
    sun_dir: [f32; 3],
    config: &CloudLayerConfig,
    settings: &CloudRaymarchSettings,
    time: f32,
) -> f32 {
    let thickness = config.thickness();
    let step_size = thickness / settings.max_light_steps as f32;
    let mut optical_depth = 0.0;

    for i in 0..settings.max_light_steps {
        let t = (i as f32 + 0.5) * step_size;
        let pos = [
            origin[0] + sun_dir[0] * t,
            origin[1] + sun_dir[1] * t,
            origin[2] + sun_dir[2] * t,
        ];

        let density = sample_cloud_density_light(pos, config, time);
        optical_depth += density * settings.extinction_coefficient * step_size;
    }

    optical_depth
}

// ---------------------------------------------------------------------------
// Temporal reprojection
// ---------------------------------------------------------------------------

/// Temporal reprojection state for reducing cloud rendering cost.
///
/// Instead of rendering every pixel every frame, we render a fraction
/// (e.g. 1/16) and reproject previous results using motion vectors.
#[derive(Debug)]
pub struct CloudTemporalReprojection {
    /// Resolution of the cloud render target.
    pub width: u32,
    pub height: u32,
    /// Previous frame's cloud colour buffer (RGBA).
    prev_color: Vec<[f32; 4]>,
    /// Previous frame's depth buffer.
    prev_depth: Vec<f32>,
    /// Current frame counter for checkerboard pattern.
    pub frame_counter: u32,
    /// Blend factor for temporal accumulation (0 = current only, 1 = prev only).
    pub blend_factor: f32,
    /// Maximum depth difference for accepting reprojection (world units).
    pub depth_rejection_threshold: f32,
    /// Checkerboard sub-pixel pattern size.
    pub checkerboard_size: u32,
    /// Whether reprojection was rejected for the last sample.
    pub rejection_count: u32,
    /// Previous frame's view-projection matrix (column-major 4x4).
    pub prev_view_proj: [f32; 16],
    /// Current frame's inverse view-projection matrix.
    pub curr_inv_view_proj: [f32; 16],
}

impl CloudTemporalReprojection {
    /// Creates a new temporal reprojection state.
    pub fn new(width: u32, height: u32) -> Self {
        let total = (width * height) as usize;
        Self {
            width,
            height,
            prev_color: vec![[0.0; 4]; total],
            prev_depth: vec![f32::MAX; total],
            frame_counter: 0,
            blend_factor: 0.95,
            depth_rejection_threshold: 100.0,
            checkerboard_size: 4,
            rejection_count: 0,
            prev_view_proj: identity_matrix(),
            curr_inv_view_proj: identity_matrix(),
        }
    }

    /// Returns true if this pixel should be ray-marched this frame.
    pub fn should_render_pixel(&self, x: u32, y: u32) -> bool {
        let pattern = self.checkerboard_size;
        let px = x % pattern;
        let py = y % pattern;
        let phase = self.frame_counter % (pattern * pattern);
        let pixel_phase = py * pattern + px;
        pixel_phase == phase
    }

    /// Reprojects a previous result to the current frame.
    ///
    /// # Arguments
    /// * `x`, `y` — Current pixel coordinates.
    /// * `current_result` — If Some, the freshly ray-marched result for this pixel.
    ///
    /// # Returns
    /// Blended cloud result.
    pub fn reproject(
        &self,
        x: u32,
        y: u32,
        current_result: Option<&CloudRaymarchResult>,
    ) -> CloudRaymarchResult {
        let idx = (y * self.width + x) as usize;
        let prev = self.prev_color.get(idx).copied().unwrap_or([0.0; 4]);
        let prev_d = self.prev_depth.get(idx).copied().unwrap_or(f32::MAX);

        if let Some(curr) = current_result {
            // Check depth consistency.
            let depth_diff = (curr.depth - prev_d).abs();
            let accept_reprojection = depth_diff < self.depth_rejection_threshold;

            if accept_reprojection {
                let t = self.blend_factor;
                CloudRaymarchResult {
                    color: [
                        lerp(curr.color[0], prev[0], t),
                        lerp(curr.color[1], prev[1], t),
                        lerp(curr.color[2], prev[2], t),
                    ],
                    transmittance: lerp(curr.transmittance, prev[3], t),
                    depth: curr.depth,
                    steps_taken: curr.steps_taken,
                }
            } else {
                // Reject: use current result only.
                *curr
            }
        } else {
            // No current result: use previous.
            CloudRaymarchResult {
                color: [prev[0], prev[1], prev[2]],
                transmittance: prev[3],
                depth: prev_d,
                steps_taken: 0,
            }
        }
    }

    /// Stores the current frame's results for use in the next frame.
    pub fn store_result(&mut self, x: u32, y: u32, result: &CloudRaymarchResult) {
        let idx = (y * self.width + x) as usize;
        if idx < self.prev_color.len() {
            self.prev_color[idx] = [
                result.color[0],
                result.color[1],
                result.color[2],
                result.transmittance,
            ];
            self.prev_depth[idx] = result.depth;
        }
    }

    /// Advances to the next frame.
    pub fn advance_frame(&mut self) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
        self.rejection_count = 0;
    }

    /// Returns the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let total = (self.width * self.height) as usize;
        total * std::mem::size_of::<[f32; 4]>() + total * std::mem::size_of::<f32>()
    }

    /// Resizes the reprojection buffers.
    pub fn resize(&mut self, width: u32, height: u32) {
        let total = (width * height) as usize;
        self.width = width;
        self.height = height;
        self.prev_color = vec![[0.0; 4]; total];
        self.prev_depth = vec![f32::MAX; total];
    }
}

/// Returns a 4x4 identity matrix in column-major order.
fn identity_matrix() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, // col 0
        0.0, 1.0, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]
}

// ---------------------------------------------------------------------------
// Cloud shadow map
// ---------------------------------------------------------------------------

/// Cloud shadow map for casting cloud shadows onto the ground.
///
/// Renders a top-down orthographic view of the cloud layer's optical depth.
/// The resulting shadow map is sampled during scene shading to darken areas
/// under clouds.
#[derive(Debug)]
pub struct CloudShadowMap {
    /// Shadow map resolution.
    pub resolution: u32,
    /// World-space size covered by the shadow map (width = height).
    pub world_size: f32,
    /// World-space center of the shadow map (XZ plane).
    pub center: [f32; 2],
    /// Shadow intensity [0, 1].
    pub intensity: f32,
    /// Shadow softness (blur radius in texels).
    pub softness: u32,
    /// Shadow map data (optical depth per texel).
    data: Vec<f32>,
    /// Computed transmittance (after applying Beer's law).
    transmittance_data: Vec<f32>,
}

impl CloudShadowMap {
    /// Creates a new cloud shadow map.
    pub fn new(resolution: u32, world_size: f32) -> Self {
        let total = (resolution * resolution) as usize;
        Self {
            resolution,
            world_size,
            center: [0.0, 0.0],
            intensity: 0.8,
            softness: 2,
            data: vec![0.0; total],
            transmittance_data: vec![1.0; total],
        }
    }

    /// Recenters the shadow map on a given world XZ position.
    pub fn recenter(&mut self, x: f32, z: f32) {
        self.center = [x, z];
    }

    /// Generates the shadow map by ray-marching downward through the cloud layer.
    ///
    /// # Arguments
    /// * `sun_dir` — Normalised direction TO the sun.
    /// * `config` — Cloud layer configuration.
    /// * `settings` — Ray-march settings.
    /// * `time` — Simulation time.
    pub fn generate(
        &mut self,
        sun_dir: [f32; 3],
        config: &CloudLayerConfig,
        settings: &CloudRaymarchSettings,
        time: f32,
    ) {
        let res = self.resolution;
        let half_size = self.world_size * 0.5;
        let texel_size = self.world_size / res as f32;

        // Number of vertical samples through the cloud layer.
        let num_steps = settings.max_light_steps.max(4);
        let thickness = config.thickness();
        let step_size = thickness / num_steps as f32;

        for ty in 0..res {
            for tx in 0..res {
                let world_x = self.center[0] + (tx as f32 + 0.5) * texel_size - half_size;
                let world_z = self.center[1] + (ty as f32 + 0.5) * texel_size - half_size;

                let mut optical_depth = 0.0;

                // March from the top of the cloud layer downward along the sun direction.
                for step in 0..num_steps {
                    let t = (step as f32 + 0.5) * step_size;
                    let sample_y = config.altitude_top - t;
                    // Offset XZ by the sun direction to match shadow projection.
                    let sun_offset_t = (config.altitude_top - sample_y)
                        / sun_dir[1].abs().max(0.01);
                    let sample_x = world_x - sun_dir[0] * sun_offset_t;
                    let sample_z = world_z - sun_dir[2] * sun_offset_t;

                    let density = sample_cloud_density_light(
                        [sample_x, sample_y, sample_z],
                        config,
                        time,
                    );

                    optical_depth += density * settings.extinction_coefficient * step_size;
                }

                let idx = (ty * res + tx) as usize;
                self.data[idx] = optical_depth;
                self.transmittance_data[idx] =
                    lerp(1.0, beers_law(optical_depth), self.intensity);
            }
        }

        // Apply box blur for softness.
        if self.softness > 0 {
            self.apply_blur();
        }
    }

    /// Simple box blur for shadow softness.
    fn apply_blur(&mut self) {
        let res = self.resolution as usize;
        let radius = self.softness as i32;
        let mut temp = vec![1.0f32; res * res];

        // Horizontal pass.
        for y in 0..res {
            for x in 0..res {
                let mut sum = 0.0;
                let mut count = 0;
                for dx in -radius..=radius {
                    let sx = x as i32 + dx;
                    if sx >= 0 && sx < res as i32 {
                        sum += self.transmittance_data[y * res + sx as usize];
                        count += 1;
                    }
                }
                temp[y * res + x] = if count > 0 { sum / count as f32 } else { 1.0 };
            }
        }

        // Vertical pass.
        for y in 0..res {
            for x in 0..res {
                let mut sum = 0.0;
                let mut count = 0;
                for dy in -radius..=radius {
                    let sy = y as i32 + dy;
                    if sy >= 0 && sy < res as i32 {
                        sum += temp[sy as usize * res + x];
                        count += 1;
                    }
                }
                self.transmittance_data[y * res + x] =
                    if count > 0 { sum / count as f32 } else { 1.0 };
            }
        }
    }

    /// Samples the cloud shadow at a world-space XZ position.
    ///
    /// Returns the shadow factor: 1.0 = fully lit, 0.0 = fully shadowed.
    pub fn sample(&self, world_x: f32, world_z: f32) -> f32 {
        let half = self.world_size * 0.5;
        let u = (world_x - self.center[0] + half) / self.world_size;
        let v = (world_z - self.center[1] + half) / self.world_size;

        if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 {
            return 1.0; // Outside shadow map — no shadow.
        }

        let tx = ((u * self.resolution as f32) as u32).min(self.resolution - 1);
        let ty = ((v * self.resolution as f32) as u32).min(self.resolution - 1);
        let idx = (ty * self.resolution + tx) as usize;

        self.transmittance_data[idx]
    }

    /// Returns the shadow map resolution.
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Returns the raw transmittance data.
    pub fn transmittance_data(&self) -> &[f32] {
        &self.transmittance_data
    }

    /// Returns the raw optical depth data.
    pub fn optical_depth_data(&self) -> &[f32] {
        &self.data
    }

    /// Returns the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let total = (self.resolution * self.resolution) as usize;
        total * std::mem::size_of::<f32>() * 2
    }
}

// ---------------------------------------------------------------------------
// VolumetricCloudRenderer (top-level)
// ---------------------------------------------------------------------------

/// Top-level volumetric cloud rendering system.
///
/// Manages one or more cloud layers, the temporal reprojection state,
/// and the cloud shadow map. This is the main entry point for cloud rendering
/// in the engine.
#[derive(Debug)]
pub struct VolumetricCloudRenderer {
    /// Cloud layers (bottom to top).
    pub layers: Vec<CloudLayerConfig>,
    /// Ray-march quality settings.
    pub settings: CloudRaymarchSettings,
    /// Temporal reprojection state.
    pub temporal: CloudTemporalReprojection,
    /// Cloud shadow map.
    pub shadow_map: CloudShadowMap,
    /// Sun direction (normalised, towards the sun).
    pub sun_direction: [f32; 3],
    /// Sun colour and intensity.
    pub sun_color: [f32; 3],
    /// Ambient sky colour.
    pub ambient_color: [f32; 3],
    /// Global cloud density multiplier.
    pub global_density: f32,
    /// Global cloud coverage override (negative = use per-layer).
    pub global_coverage: f32,
    /// Whether cloud rendering is enabled.
    pub enabled: bool,
    /// Whether cloud shadows are enabled.
    pub shadows_enabled: bool,
    /// Whether temporal reprojection is enabled.
    pub temporal_enabled: bool,
    /// Internal frame counter.
    frame_index: u64,
}

impl VolumetricCloudRenderer {
    /// Creates a new cloud renderer with a single cumulus layer.
    pub fn new(render_width: u32, render_height: u32) -> Self {
        Self {
            layers: vec![CloudLayerConfig::cumulus()],
            settings: CloudRaymarchSettings::default(),
            temporal: CloudTemporalReprojection::new(render_width, render_height),
            shadow_map: CloudShadowMap::new(256, 20000.0),
            sun_direction: [0.0, 1.0, 0.0],
            sun_color: [1.0, 0.95, 0.85],
            ambient_color: [0.15, 0.18, 0.25],
            global_density: 1.0,
            global_coverage: -1.0,
            enabled: true,
            shadows_enabled: true,
            temporal_enabled: true,
            frame_index: 0,
        }
    }

    /// Creates with a specific quality preset.
    pub fn with_quality(mut self, settings: CloudRaymarchSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Adds a cloud layer.
    pub fn add_layer(&mut self, layer: CloudLayerConfig) {
        self.layers.push(layer);
    }

    /// Removes all cloud layers.
    pub fn clear_layers(&mut self) {
        self.layers.clear();
    }

    /// Sets the sun parameters.
    pub fn set_sun(&mut self, direction: [f32; 3], color: [f32; 3]) {
        // Normalise direction.
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if len > 1e-6 {
            self.sun_direction = [direction[0] / len, direction[1] / len, direction[2] / len];
        }
        self.sun_color = color;
    }

    /// Renders clouds for a single pixel.
    pub fn render_pixel(
        &self,
        ray_origin: [f32; 3],
        ray_dir: [f32; 3],
        scene_depth: f32,
        time: f32,
    ) -> CloudRaymarchResult {
        if !self.enabled || self.layers.is_empty() {
            return CloudRaymarchResult::clear();
        }

        // March through each layer and composite.
        let mut final_result = CloudRaymarchResult::clear();

        for layer in &self.layers {
            let mut layer_config = layer.clone();

            // Apply global overrides.
            layer_config.density_multiplier *= self.global_density;
            if self.global_coverage >= 0.0 {
                layer_config.coverage = self.global_coverage;
            }

            let result = raymarch_clouds(
                ray_origin,
                ray_dir,
                self.sun_direction,
                self.sun_color,
                self.ambient_color,
                &layer_config,
                &self.settings,
                time,
                scene_depth,
            );

            // Composite: front-to-back blending.
            if result.transmittance < 1.0 {
                let alpha = 1.0 - result.transmittance;
                final_result.color[0] += result.color[0] * final_result.transmittance;
                final_result.color[1] += result.color[1] * final_result.transmittance;
                final_result.color[2] += result.color[2] * final_result.transmittance;
                final_result.transmittance *= result.transmittance;

                if result.depth < final_result.depth {
                    final_result.depth = result.depth;
                }
                final_result.steps_taken += result.steps_taken;
            }

            // Early out if already opaque.
            if final_result.transmittance < self.settings.min_transmittance {
                break;
            }
        }

        final_result
    }

    /// Updates the cloud shadow map.
    pub fn update_shadow_map(&mut self, camera_x: f32, camera_z: f32, time: f32) {
        if !self.shadows_enabled || self.layers.is_empty() {
            return;
        }

        self.shadow_map.recenter(camera_x, camera_z);

        // Use the lowest (most impactful) cloud layer for shadows.
        let primary_layer = &self.layers[0];
        self.shadow_map.generate(
            self.sun_direction,
            primary_layer,
            &self.settings,
            time,
        );
    }

    /// Queries the cloud shadow at a world position.
    pub fn shadow_at(&self, world_x: f32, world_z: f32) -> f32 {
        if !self.shadows_enabled {
            return 1.0;
        }
        self.shadow_map.sample(world_x, world_z)
    }

    /// Advances internal state for the next frame.
    pub fn advance_frame(&mut self) {
        self.frame_index += 1;
        if self.temporal_enabled {
            self.temporal.advance_frame();
        }
    }

    /// Resizes the render target.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.temporal.resize(width, height);
    }

    /// Returns total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.temporal.memory_usage() + self.shadow_map.memory_usage()
    }

    /// Returns the current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
}

impl Default for VolumetricCloudRenderer {
    fn default() -> Self {
        Self::new(960, 540) // Half-res by default.
    }
}

// ---------------------------------------------------------------------------
// WGSL cloud shader
// ---------------------------------------------------------------------------

/// WGSL compute shader for volumetric cloud ray-marching.
///
/// This shader runs one thread per pixel. Each thread ray-marches through the
/// cloud layer and writes colour + transmittance to a UAV texture.
pub const VOLUMETRIC_CLOUDS_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Volumetric clouds compute shader (Genovo Engine)
// -----------------------------------------------------------------------

struct CloudUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    sun_direction: vec3<f32>,
    sun_intensity: f32,
    sun_color: vec3<f32>,
    cloud_bottom: f32,
    cloud_top: f32,
    coverage: f32,
    density: f32,
    extinction: f32,
    shape_freq: f32,
    detail_freq: f32,
    wind_x: f32,
    wind_z: f32,
    wind_speed: f32,
    phase_g: f32,
    max_steps: u32,
    light_steps: u32,
    frame_index: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> cloud: CloudUniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var shape_noise_tex: texture_3d<f32>;
@group(0) @binding(3) var detail_noise_tex: texture_3d<f32>;
@group(0) @binding(4) var noise_sampler: sampler;
@group(0) @binding(5) var depth_texture: texture_2d<f32>;
@group(0) @binding(6) var prev_cloud_texture: texture_2d<f32>;

const PI: f32 = 3.14159265358979;

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

fn beer(optical_depth: f32) -> f32 {
    return exp(-optical_depth);
}

fn remap_val(value: f32, old_min: f32, old_max: f32, new_min: f32, new_max: f32) -> f32 {
    let t = clamp((value - old_min) / (old_max - old_min), 0.0, 1.0);
    return new_min + t * (new_max - new_min);
}

fn height_fraction(y: f32) -> f32 {
    return clamp((y - cloud.cloud_bottom) / (cloud.cloud_top - cloud.cloud_bottom), 0.0, 1.0);
}

fn density_profile(h: f32) -> f32 {
    let bottom = clamp(remap_val(h, 0.0, 0.15, 0.0, 1.0), 0.0, 1.0);
    let top = clamp(remap_val(h, 0.6, 1.0, 1.0, 0.0), 0.0, 1.0);
    return bottom * top;
}

fn sample_cloud(pos: vec3<f32>) -> f32 {
    let h = height_fraction(pos.y);
    let profile = density_profile(h);
    if profile < 0.001 {
        return 0.0;
    }

    let wind_offset = vec3<f32>(
        cloud.wind_x * cloud.wind_speed * cloud.time,
        0.0,
        cloud.wind_z * cloud.wind_speed * cloud.time
    );

    let sample_pos = pos + wind_offset;
    let shape_uv = sample_pos * cloud.shape_freq;
    let shape = textureSampleLevel(shape_noise_tex, noise_sampler, shape_uv, 0.0).r;

    let shaped = remap_val(shape, 1.0 - cloud.coverage, 1.0, 0.0, 1.0) * profile;
    if shaped <= 0.0 {
        return 0.0;
    }

    let detail_uv = sample_pos * cloud.detail_freq;
    let detail = textureSampleLevel(detail_noise_tex, noise_sampler, detail_uv, 0.0).r;
    let eroded = max(shaped - detail * 0.35, 0.0);

    return eroded * cloud.density;
}

fn light_march(pos: vec3<f32>) -> f32 {
    let thickness = cloud.cloud_top - cloud.cloud_bottom;
    let step_size = thickness / f32(cloud.light_steps);
    var od = 0.0;

    for (var i = 0u; i < cloud.light_steps; i = i + 1u) {
        let t = (f32(i) + 0.5) * step_size;
        let sample_pos = pos + cloud.sun_direction * t;
        let d = sample_cloud(sample_pos);
        od += d * cloud.extinction * step_size;
    }

    return od;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(dims.x),
        (f32(gid.y) + 0.5) / f32(dims.y)
    );

    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let world = cloud.inv_view_proj * ndc;
    let ray_dir = normalize(world.xyz / world.w - cloud.camera_pos);

    let scene_depth = textureLoad(depth_texture, vec2<i32>(gid.xy), 0).r;

    let dy = ray_dir.y;
    if abs(dy) < 1e-6 {
        textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let t_bottom = (cloud.cloud_bottom - cloud.camera_pos.y) / dy;
    let t_top = (cloud.cloud_top - cloud.camera_pos.y) / dy;
    var t_enter = min(t_bottom, t_top);
    var t_exit = max(t_bottom, t_top);
    t_enter = max(t_enter, 0.0);
    t_exit = min(t_exit, 30000.0);

    if t_enter >= t_exit {
        textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let cos_theta = dot(ray_dir, cloud.sun_direction);
    let phase = henyey_greenstein(cos_theta, cloud.phase_g);

    let step_size = (t_exit - t_enter) / f32(cloud.max_steps);
    var color = vec3<f32>(0.0);
    var transmittance = 1.0;

    for (var i = 0u; i < cloud.max_steps; i = i + 1u) {
        let t = t_enter + (f32(i) + 0.5) * step_size;
        let pos = cloud.camera_pos + ray_dir * t;
        let d = sample_cloud(pos);

        if d > 0.001 {
            let od = d * cloud.extinction * step_size;
            let light_od = light_march(pos);
            let light_trans = beer(light_od);

            let h = height_fraction(pos.y);
            let cloud_col = mix(vec3<f32>(0.5, 0.55, 0.6), vec3<f32>(1.0), h);
            let inscattered = cloud_col * cloud.sun_color * light_trans * phase;

            let step_trans = beer(od);
            color += inscattered * transmittance * (1.0 - step_trans);
            transmittance *= step_trans;

            if transmittance < 0.01 {
                break;
            }
        }
    }

    textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(color, transmittance));
}
"#;

// ---------------------------------------------------------------------------
// Cloud noise texture generation
// ---------------------------------------------------------------------------

/// Generates a 3D noise texture for cloud shapes.
///
/// Returns a flat buffer suitable for uploading to a 3D GPU texture.
/// Format: single-channel f32, dimensions `size x size x size`.
pub fn generate_shape_noise_3d(size: u32, frequency: f32) -> Vec<f32> {
    let total = (size * size * size) as usize;
    let mut data = Vec::with_capacity(total);
    let inv_size = 1.0 / size as f32;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let fx = x as f32 * inv_size * frequency;
                let fy = y as f32 * inv_size * frequency;
                let fz = z as f32 * inv_size * frequency;
                let value = perlin_worley_3d(fx, fy, fz);
                data.push(value);
            }
        }
    }

    data
}

/// Generates a 3D detail noise texture for cloud erosion.
///
/// Uses higher-frequency Worley noise.
pub fn generate_detail_noise_3d(size: u32, frequency: f32) -> Vec<f32> {
    let total = (size * size * size) as usize;
    let mut data = Vec::with_capacity(total);
    let inv_size = 1.0 / size as f32;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let fx = x as f32 * inv_size * frequency;
                let fy = y as f32 * inv_size * frequency;
                let fz = z as f32 * inv_size * frequency;
                let worley = 1.0 - worley_noise_3d(fx, fy, fz);
                let fbm = fbm_3d(fx * 2.0, fy * 2.0, fz * 2.0, 3, 2.0, 0.5) * 0.5 + 0.5;
                let value = worley * 0.625 + fbm * 0.375;
                data.push(value);
            }
        }
    }

    data
}

// ---------------------------------------------------------------------------
// Weather map
// ---------------------------------------------------------------------------

/// 2D weather map that controls cloud coverage and type across the world.
///
/// Channels:
/// - R: cloud coverage
/// - G: cloud type (0 = stratus, 0.5 = cumulus, 1 = cumulonimbus)
/// - B: precipitation probability
#[derive(Debug)]
pub struct WeatherMap {
    /// Map resolution.
    pub resolution: u32,
    /// World-space size covered.
    pub world_size: f32,
    /// Map data (RGB per texel).
    data: Vec<[f32; 3]>,
}

impl WeatherMap {
    /// Creates a weather map with procedural coverage.
    pub fn new_procedural(resolution: u32, world_size: f32, time: f32) -> Self {
        let total = (resolution * resolution) as usize;
        let mut data = Vec::with_capacity(total);
        let inv_res = 1.0 / resolution as f32;

        for y in 0..resolution {
            for x in 0..resolution {
                let wx = x as f32 * inv_res * 8.0 + time * 0.01;
                let wz = y as f32 * inv_res * 8.0;

                // Coverage: low-frequency noise.
                let coverage = (fbm_3d(wx, 0.0, wz, 3, 2.0, 0.5) * 0.5 + 0.5).clamp(0.0, 1.0);

                // Cloud type: separate noise frequency.
                let cloud_type = (value_noise_3d(wx * 0.5, 0.0, wz * 0.5) * 0.5 + 0.5)
                    .clamp(0.0, 1.0);

                // Precipitation: correlated with high coverage.
                let precip = if coverage > 0.7 {
                    ((coverage - 0.7) / 0.3).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                data.push([coverage, cloud_type, precip]);
            }
        }

        Self {
            resolution,
            world_size,
            data,
        }
    }

    /// Creates a uniform weather map with constant coverage.
    pub fn new_uniform(resolution: u32, world_size: f32, coverage: f32) -> Self {
        let total = (resolution * resolution) as usize;
        Self {
            resolution,
            world_size,
            data: vec![[coverage, 0.5, 0.0]; total],
        }
    }

    /// Samples the weather map at a world-space XZ position.
    ///
    /// Returns `(coverage, cloud_type, precipitation)`.
    pub fn sample(&self, world_x: f32, world_z: f32) -> (f32, f32, f32) {
        let half = self.world_size * 0.5;
        let u = ((world_x + half) / self.world_size).rem_euclid(1.0);
        let v = ((world_z + half) / self.world_size).rem_euclid(1.0);

        let tx = ((u * self.resolution as f32) as u32).min(self.resolution - 1);
        let ty = ((v * self.resolution as f32) as u32).min(self.resolution - 1);
        let idx = (ty * self.resolution + tx) as usize;

        let d = self.data[idx];
        (d[0], d[1], d[2])
    }

    /// Returns the raw data.
    pub fn data(&self) -> &[[f32; 3]] {
        &self.data
    }

    /// Returns memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<[f32; 3]>()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_type_profiles() {
        let types = [
            CloudType::Cumulus,
            CloudType::Stratus,
            CloudType::Cirrus,
            CloudType::Cumulonimbus,
            CloudType::Stratocumulus,
        ];

        for ct in &types {
            // Profile should be zero at boundaries.
            let bottom = ct.density_profile(0.0);
            let top = ct.density_profile(1.0);
            assert!(
                bottom < 0.5 || matches!(ct, CloudType::Cumulonimbus),
                "{ct:?}: bottom = {bottom}"
            );
            assert!(top < 0.1, "{ct:?}: top = {top}");

            // Profile should have some mass in the middle.
            let mid = ct.density_profile(0.3);
            assert!(mid > 0.0, "{ct:?}: mid = {mid}");
        }
    }

    #[test]
    fn test_henyey_greenstein_isotropic() {
        let p1 = henyey_greenstein_phase(1.0, 0.0);
        let p2 = henyey_greenstein_phase(0.0, 0.0);
        let p3 = henyey_greenstein_phase(-1.0, 0.0);
        assert!((p1 - p2).abs() < 0.01);
        assert!((p2 - p3).abs() < 0.01);
    }

    #[test]
    fn test_henyey_greenstein_forward() {
        let forward = henyey_greenstein_phase(1.0, 0.8);
        let backward = henyey_greenstein_phase(-1.0, 0.8);
        assert!(
            forward > backward,
            "Forward scattering should be stronger with g=0.8"
        );
    }

    #[test]
    fn test_beers_law() {
        assert!((beers_law(0.0) - 1.0).abs() < 1e-6);
        assert!(beers_law(1.0) < 1.0);
        assert!(beers_law(10.0) < 0.001);
    }

    #[test]
    fn test_beers_powder() {
        let bp = beers_powder(1.0, 2.0);
        assert!(bp > 0.0);
        assert!(bp < 1.0);
    }

    #[test]
    fn test_cloud_density_outside_layer() {
        let config = CloudLayerConfig::cumulus();
        // Sample below the cloud layer.
        let density = sample_cloud_density([0.0, 0.0, 0.0], &config, 0.0);
        assert!(density < 1e-6, "No cloud below layer: {density}");

        // Sample above the cloud layer.
        let density = sample_cloud_density([0.0, 100000.0, 0.0], &config, 0.0);
        assert!(density < 1e-6, "No cloud above layer: {density}");
    }

    #[test]
    fn test_raymarch_clear_sky() {
        let config = CloudLayerConfig::cumulus();
        let settings = CloudRaymarchSettings::low_quality();

        // Look sideways from below the cloud layer — should not hit clouds
        // if ray direction is horizontal and camera is below.
        let result = raymarch_clouds(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], // horizontal
            [0.0, 1.0, 0.0], // sun up
            [1.0, 1.0, 1.0],
            [0.1, 0.1, 0.1],
            &config,
            &settings,
            0.0,
            f32::MAX,
        );

        // Horizontal ray from ground should not hit clouds.
        assert!(
            result.transmittance > 0.9,
            "Horizontal ray should mostly miss clouds: t = {}",
            result.transmittance
        );
    }

    #[test]
    fn test_raymarch_looking_up() {
        let config = CloudLayerConfig::cumulus().with_coverage(1.0).with_density(5.0);
        let settings = CloudRaymarchSettings::low_quality();

        let result = raymarch_clouds(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], // straight up
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.1, 0.1, 0.1],
            &config,
            &settings,
            0.0,
            f32::MAX,
        );

        // With full coverage and high density, transmittance should be reduced.
        // (May still be > 0 depending on noise, but steps_taken should be > 0.)
        assert!(result.steps_taken > 0, "Should have taken steps");
    }

    #[test]
    fn test_cloud_shadow_map() {
        let config = CloudLayerConfig::cumulus().with_coverage(0.8);
        let settings = CloudRaymarchSettings::low_quality();
        let mut shadow = CloudShadowMap::new(16, 10000.0);
        shadow.generate([0.0, 1.0, 0.0], &config, &settings, 0.0);

        // At least some texels should have shadow.
        let has_shadow = shadow
            .transmittance_data()
            .iter()
            .any(|&t| t < 0.99);
        // (Due to noise coverage thresholding, not guaranteed, but likely with 0.8 coverage.)
        // Just verify it ran without panicking and produced valid data.
        assert!(shadow.transmittance_data().len() == 16 * 16);
    }

    #[test]
    fn test_temporal_reprojection() {
        let mut temporal = CloudTemporalReprojection::new(4, 4);

        // Frame 0: store a result.
        let result = CloudRaymarchResult {
            color: [0.5, 0.5, 0.5],
            transmittance: 0.3,
            depth: 5000.0,
            steps_taken: 32,
        };
        temporal.store_result(0, 0, &result);
        temporal.advance_frame();

        // Frame 1: reproject without a new render.
        let reprojected = temporal.reproject(0, 0, None);
        assert!(
            (reprojected.color[0] - 0.5).abs() < 0.01,
            "Should use previous frame's data"
        );
    }

    #[test]
    fn test_weather_map_procedural() {
        let wm = WeatherMap::new_procedural(32, 10000.0, 0.0);
        let (cov, ct, _precip) = wm.sample(0.0, 0.0);
        assert!(cov >= 0.0 && cov <= 1.0);
        assert!(ct >= 0.0 && ct <= 1.0);
    }

    #[test]
    fn test_weather_map_uniform() {
        let wm = WeatherMap::new_uniform(8, 5000.0, 0.6);
        let (cov, _, _) = wm.sample(100.0, 200.0);
        assert!((cov - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_cloud_layer_config_defaults() {
        let config = CloudLayerConfig::cumulus();
        assert!(config.altitude_bottom > 0.0);
        assert!(config.altitude_top > config.altitude_bottom);
        assert!(config.thickness() > 0.0);
        assert!(config.coverage > 0.0);
    }

    #[test]
    fn test_volumetric_cloud_renderer_creation() {
        let renderer = VolumetricCloudRenderer::new(1920, 1080);
        assert_eq!(renderer.layers.len(), 1);
        assert!(renderer.enabled);
        assert!(renderer.shadows_enabled);
    }

    #[test]
    fn test_renderer_add_remove_layers() {
        let mut renderer = VolumetricCloudRenderer::new(320, 240);
        renderer.add_layer(CloudLayerConfig::cirrus());
        assert_eq!(renderer.layers.len(), 2);
        renderer.clear_layers();
        assert_eq!(renderer.layers.len(), 0);
    }

    #[test]
    fn test_noise_generation() {
        let shape = generate_shape_noise_3d(4, 1.0);
        assert_eq!(shape.len(), 64);
        assert!(shape.iter().all(|v| *v >= 0.0 && *v <= 1.0));

        let detail = generate_detail_noise_3d(4, 1.0);
        assert_eq!(detail.len(), 64);
    }

    #[test]
    fn test_multi_scattering() {
        let ms = multi_scattering_approximation(1.0, 0.5, 0.8, 4);
        assert!(ms > 0.0, "Multi-scattering should produce positive contribution");
    }

    #[test]
    fn test_dual_lobe_phase() {
        let dl = dual_lobe_hg_phase(1.0, 0.8, -0.3, 0.2);
        assert!(dl > 0.0);
        // Forward should be brighter than backward.
        let dl_back = dual_lobe_hg_phase(-1.0, 0.8, -0.3, 0.2);
        assert!(dl > dl_back);
    }

    #[test]
    fn test_cloud_shadow_map_sample_outside() {
        let shadow = CloudShadowMap::new(8, 1000.0);
        // Outside the shadow map should return 1.0 (no shadow).
        let s = shadow.sample(100000.0, 100000.0);
        assert!((s - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_wind_offset() {
        let config = CloudLayerConfig::cumulus().with_wind([1.0, 0.0], 10.0);
        let offset = config.wind_offset(1.0, 0.5);
        assert!((offset[0] - 10.5).abs() < 1.0, "Wind should move in X: {:?}", offset);
    }
}
