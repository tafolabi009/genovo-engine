// engine/render/src/atmosphere/mod.rs
//
// Physically-based atmospheric scattering for the Genovo engine.
//
// Implements a single-scattering Nishita-style sky model including:
//
// - Rayleigh scattering (small molecules — blue sky, red sunsets).
// - Mie scattering (aerosols/haze — bright halo around the sun).
// - Optical depth integration along view rays and sun rays.
// - Transmittance look-up table (altitude × zenith angle).
// - Sun disc rendering with physical limb darkening.
// - Time-of-day system that drives sun/moon position from geographic coords.
// - Procedural star field with magnitude-based brightness.
// - Moon rendering with phase calculation.
// - Simple noise-based cloud layer with single-scatter approximation.
// - WGSL compute shader source for GPU-side LUT generation.
//
// The module is designed to be consumed by [`crate::sky`] components and the
// render graph's sky pass.

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Speed of light in vacuum (m/s) — informational; not used in the shader math
/// but kept for reference.
#[allow(dead_code)]
const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Index of refraction for air at sea level (STP).
const AIR_IOR: f32 = 1.000_293;

/// Molecular number density of air at sea level (molecules/m^3).
const AIR_NUMBER_DENSITY: f64 = 2.545e25;

/// Default number of samples along the primary view ray.
const DEFAULT_VIEW_SAMPLES: u32 = 32;

/// Default number of samples along the sun (light) ray at each view sample.
const DEFAULT_LIGHT_SAMPLES: u32 = 8;

/// Minimum sun altitude (radians) below which we consider nighttime for star
/// rendering.
const STAR_FADE_ALTITUDE: f32 = -0.05;

/// Maximum sun altitude (radians) above which stars are fully invisible.
const STAR_VISIBLE_ALTITUDE: f32 = 0.1;

// ---------------------------------------------------------------------------
// AtmosphereParams
// ---------------------------------------------------------------------------

/// Complete parameter set for the atmospheric scattering model.
///
/// All distances are in **kilometres** to match the conventional scale used in
/// atmospheric rendering (planet radius ~6371 km). Internal math converts to a
/// consistent unit system where needed.
#[derive(Debug, Clone)]
pub struct AtmosphereParams {
    /// Planet radius in km (Earth ≈ 6371).
    pub planet_radius: f32,
    /// Thickness of the atmosphere shell in km (Earth ≈ 100).
    pub atmosphere_height: f32,
    /// Rayleigh scale height in km (Earth ≈ 8.5).
    pub rayleigh_scale_height: f32,
    /// Mie scale height in km (Earth ≈ 1.2).
    pub mie_scale_height: f32,
    /// Rayleigh scattering coefficients (1/km) at sea level for R, G, B.
    /// Earth default: `(5.5e-3, 13.0e-3, 22.4e-3)`.
    pub rayleigh_coefficients: Vec3,
    /// Mie scattering coefficient (1/km) at sea level.  Earth default ≈ 21e-3.
    pub mie_coefficient: f32,
    /// Mie absorption coefficient (1/km) at sea level.  Earth default ≈ 2.1e-3.
    pub mie_absorption: f32,
    /// Mie phase function asymmetry parameter *g* ∈ (−1, 1).
    /// Positive = forward scattering.  Earth default ≈ 0.758.
    pub mie_g: f32,
    /// Ozone absorption coefficients (1/km) at peak density for R, G, B.
    pub ozone_coefficients: Vec3,
    /// Centre of the ozone layer in km above sea level.
    pub ozone_centre_height: f32,
    /// Width of the ozone layer distribution in km.
    pub ozone_width: f32,
    /// Sun irradiance (colour × intensity).  Typically `(1, 1, 1) * intensity`.
    pub sun_intensity: Vec3,
    /// Normalised sun direction in world space (pointing *towards* the sun).
    pub sun_direction: Vec3,
    /// Angular diameter of the sun disc in radians (Earth-Sun ≈ 0.00935).
    pub sun_angular_diameter: f32,
    /// Ground albedo (Lambertian reflectance) used in multi-scattering approx.
    pub ground_albedo: Vec3,
    /// Number of integration samples along the view ray.
    pub view_samples: u32,
    /// Number of integration samples along the light ray at each view sample.
    pub light_samples: u32,
    /// Exposure multiplier applied to the final colour.
    pub exposure: f32,
}

impl AtmosphereParams {
    // -- Presets --

    /// Returns parameters matching Earth's atmosphere (standard clear-sky).
    pub fn earth() -> Self {
        Self {
            planet_radius: 6371.0,
            atmosphere_height: 100.0,
            rayleigh_scale_height: 8.5,
            mie_scale_height: 1.2,
            rayleigh_coefficients: Vec3::new(5.802e-3, 13.558e-3, 33.1e-3),
            mie_coefficient: 3.996e-3,
            mie_absorption: 4.4e-4,
            mie_g: 0.758,
            ozone_coefficients: Vec3::new(0.650e-3, 1.881e-3, 0.085e-3),
            ozone_centre_height: 25.0,
            ozone_width: 15.0,
            sun_intensity: Vec3::splat(20.0),
            sun_direction: Vec3::new(0.0, 1.0, 0.0),
            sun_angular_diameter: 0.009_35,
            ground_albedo: Vec3::splat(0.3),
            view_samples: DEFAULT_VIEW_SAMPLES,
            light_samples: DEFAULT_LIGHT_SAMPLES,
            exposure: 1.0,
        }
    }

    /// Returns a Mars-like atmosphere preset: thinner, mostly CO2.
    pub fn mars() -> Self {
        Self {
            planet_radius: 3389.5,
            atmosphere_height: 80.0,
            rayleigh_scale_height: 11.1,
            mie_scale_height: 2.0,
            // CO2-dominated Rayleigh scattering (weak blue, stronger red).
            rayleigh_coefficients: Vec3::new(19.918e-3, 13.57e-3, 5.75e-3),
            mie_coefficient: 21.0e-3,
            mie_absorption: 2.1e-3,
            mie_g: 0.65,
            ozone_coefficients: Vec3::ZERO,
            ozone_centre_height: 0.0,
            ozone_width: 1.0,
            sun_intensity: Vec3::splat(10.0),
            sun_direction: Vec3::new(0.0, 1.0, 0.0),
            sun_angular_diameter: 0.006_10, // farther from sun
            ground_albedo: Vec3::new(0.4, 0.26, 0.15),
            view_samples: DEFAULT_VIEW_SAMPLES,
            light_samples: DEFAULT_LIGHT_SAMPLES,
            exposure: 1.5,
        }
    }

    /// Returns a thick, hazy alien atmosphere preset.
    pub fn alien_haze() -> Self {
        Self {
            planet_radius: 8000.0,
            atmosphere_height: 200.0,
            rayleigh_scale_height: 12.0,
            mie_scale_height: 3.5,
            rayleigh_coefficients: Vec3::new(3.0e-3, 7.0e-3, 18.0e-3),
            mie_coefficient: 50.0e-3,
            mie_absorption: 5.0e-3,
            mie_g: 0.85,
            ozone_coefficients: Vec3::new(1.0e-3, 0.5e-3, 2.0e-3),
            ozone_centre_height: 40.0,
            ozone_width: 25.0,
            sun_intensity: Vec3::new(18.0, 14.0, 10.0),
            sun_direction: Vec3::new(0.0, 1.0, 0.0),
            sun_angular_diameter: 0.012,
            ground_albedo: Vec3::splat(0.2),
            view_samples: DEFAULT_VIEW_SAMPLES,
            light_samples: DEFAULT_LIGHT_SAMPLES,
            exposure: 1.2,
        }
    }

    /// Outer radius of the atmosphere (planet + atmosphere height).
    #[inline]
    pub fn atmosphere_radius(&self) -> f32 {
        self.planet_radius + self.atmosphere_height
    }
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self::earth()
    }
}

// ---------------------------------------------------------------------------
// Phase functions
// ---------------------------------------------------------------------------

/// Rayleigh phase function.
///
/// ```text
/// P_R(θ) = 3 / (16π) · (1 + cos²θ)
/// ```
#[inline]
pub fn rayleigh_phase(cos_theta: f32) -> f32 {
    (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta)
}

/// Cornette-Shanks Mie phase function (improved Henyey-Greenstein).
///
/// ```text
/// P_M(θ, g) = 3(1 − g²)(1 + cos²θ)
///             ─────────────────────────
///             8π(2 + g²)(1 + g² − 2g·cosθ)^(3/2)
/// ```
///
/// The Cornette-Shanks formulation normalises correctly over the sphere,
/// unlike the classic Henyey-Greenstein which does not include the
/// `(1 + cos²θ)` numerator term.
#[inline]
pub fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = 3.0 * (1.0 - g2) * (1.0 + cos_theta * cos_theta);
    let denom_base = 1.0 + g2 - 2.0 * g * cos_theta;
    let denom = 8.0 * PI * (2.0 + g2) * denom_base * denom_base.sqrt();
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    num / denom
}

/// Classic Henyey-Greenstein phase function (single-term).
///
/// ```text
/// P_HG(θ, g) = (1 − g²) / (4π (1 + g² − 2g·cosθ)^(3/2))
/// ```
#[inline]
pub fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom_base = 1.0 + g2 - 2.0 * g * cos_theta;
    let denom = 4.0 * PI * denom_base * denom_base.sqrt();
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (1.0 - g2) / denom
}

/// Schlick approximation to the Henyey-Greenstein phase function.
/// Cheaper to evaluate, close visual match.
///
/// ```text
/// P_S(θ, k) = (1 − k²) / (4π (1 − k·cosθ)²)
/// ```
///
/// where `k ≈ 1.55g − 0.55g³`.
#[inline]
pub fn schlick_phase(cos_theta: f32, g: f32) -> f32 {
    let k = 1.55 * g - 0.55 * g * g * g;
    let one_minus_k_cos = 1.0 - k * cos_theta;
    (1.0 - k * k) / (4.0 * PI * one_minus_k_cos * one_minus_k_cos)
}

// ---------------------------------------------------------------------------
// Density functions
// ---------------------------------------------------------------------------

/// Rayleigh density at a given height (km above sea level).
#[inline]
pub fn rayleigh_density(height_km: f32, scale_height: f32) -> f32 {
    (-height_km / scale_height).exp()
}

/// Mie density at a given height (km above sea level).
#[inline]
pub fn mie_density(height_km: f32, scale_height: f32) -> f32 {
    (-height_km / scale_height).exp()
}

/// Ozone density approximation: triangular profile centred at
/// `centre_height` with half-width `width`.
#[inline]
pub fn ozone_density(height_km: f32, centre: f32, width: f32) -> f32 {
    (1.0 - ((height_km - centre) / width).abs()).max(0.0)
}

// ---------------------------------------------------------------------------
// Ray-sphere intersection
// ---------------------------------------------------------------------------

/// Result of a ray-sphere intersection test.
#[derive(Debug, Clone, Copy)]
pub struct SphereIntersection {
    /// Whether the ray intersects the sphere at all.
    pub hit: bool,
    /// Distance to the near intersection (may be negative if inside sphere).
    pub t0: f32,
    /// Distance to the far intersection.
    pub t1: f32,
}

/// Tests intersection of a ray with a sphere centred at the origin.
///
/// Returns `(hit, t_near, t_far)`.  If the ray originates inside the sphere,
/// `t_near` will be negative and `t_far` gives the exit distance.
pub fn ray_sphere_intersect(
    ray_origin: Vec3,
    ray_dir: Vec3,
    sphere_radius: f32,
) -> SphereIntersection {
    let a = ray_dir.dot(ray_dir);
    let b = 2.0 * ray_origin.dot(ray_dir);
    let c = ray_origin.dot(ray_origin) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return SphereIntersection {
            hit: false,
            t0: 0.0,
            t1: 0.0,
        };
    }

    let sqrt_disc = discriminant.sqrt();
    let inv_2a = 0.5 / a;
    let t0 = (-b - sqrt_disc) * inv_2a;
    let t1 = (-b + sqrt_disc) * inv_2a;

    SphereIntersection {
        hit: true,
        t0,
        t1,
    }
}

// ---------------------------------------------------------------------------
// Optical depth
// ---------------------------------------------------------------------------

/// Computes the *optical depth* (total extinction) along a ray through the
/// atmosphere.
///
/// The returned `Vec3` encodes:
/// - `.x`: Rayleigh optical depth
/// - `.y`: Mie optical depth
/// - `.z`: Ozone optical depth (absorption only)
///
/// Integration uses the trapezoidal rule over `num_samples` segments.
pub fn compute_optical_depth(
    ray_origin: Vec3,
    ray_dir: Vec3,
    ray_length: f32,
    num_samples: u32,
    params: &AtmosphereParams,
) -> Vec3 {
    let ds = ray_length / num_samples as f32;
    let mut optical_depth = Vec3::ZERO;

    for i in 0..num_samples {
        let t = (i as f32 + 0.5) * ds;
        let sample_pos = ray_origin + ray_dir * t;
        let height = sample_pos.length() - params.planet_radius;

        if height < 0.0 {
            // Below surface — treat as opaque.
            return Vec3::splat(1e10);
        }

        let rho_r = rayleigh_density(height, params.rayleigh_scale_height);
        let rho_m = mie_density(height, params.mie_scale_height);
        let rho_o = ozone_density(height, params.ozone_centre_height, params.ozone_width);

        optical_depth.x += rho_r * ds;
        optical_depth.y += rho_m * ds;
        optical_depth.z += rho_o * ds;
    }

    optical_depth
}

/// Computes the combined extinction coefficient (Rayleigh + Mie + Ozone) from
/// an optical depth vector and the atmosphere parameters.
fn extinction_from_optical_depth(od: Vec3, params: &AtmosphereParams) -> Vec3 {
    let rayleigh = params.rayleigh_coefficients * od.x;
    let mie = Vec3::splat(params.mie_coefficient + params.mie_absorption) * od.y;
    let ozone = params.ozone_coefficients * od.z;
    rayleigh + mie + ozone
}

/// Computes the transmittance (e^-τ) from `ray_origin` along `ray_dir` for
/// `ray_length`.
pub fn compute_transmittance(
    ray_origin: Vec3,
    ray_dir: Vec3,
    ray_length: f32,
    num_samples: u32,
    params: &AtmosphereParams,
) -> Vec3 {
    let od = compute_optical_depth(ray_origin, ray_dir, ray_length, num_samples, params);
    let tau = extinction_from_optical_depth(od, params);
    Vec3::new((-tau.x).exp(), (-tau.y).exp(), (-tau.z).exp())
}

// ---------------------------------------------------------------------------
// Sky colour computation (single scattering)
// ---------------------------------------------------------------------------

/// Computes the sky colour for a single view direction using single-scattering.
///
/// The camera position should be specified in "atmosphere space" where the
/// planet centre is at the origin and distances are in km.  Typically
/// `camera_pos = Vec3::new(0, planet_radius + camera_altitude_km, 0)`.
///
/// Returns the *inscattered* radiance (linear HDR, pre-exposure).
pub fn compute_sky_color(
    camera_pos: Vec3,
    view_dir: Vec3,
    params: &AtmosphereParams,
) -> Vec3 {
    let sun_dir = params.sun_direction.normalize();

    // Intersect view ray with atmosphere sphere.
    let atmo_isect = ray_sphere_intersect(camera_pos, view_dir, params.atmosphere_radius());
    if !atmo_isect.hit || atmo_isect.t1 < 0.0 {
        return Vec3::ZERO; // looking away from atmosphere
    }

    // Also check if we hit the planet surface.
    let planet_isect = ray_sphere_intersect(camera_pos, view_dir, params.planet_radius);
    let planet_hit = planet_isect.hit && planet_isect.t0 > 0.0;

    let t_start = atmo_isect.t0.max(0.0);
    let t_end = if planet_hit {
        planet_isect.t0.min(atmo_isect.t1)
    } else {
        atmo_isect.t1
    };

    if t_end <= t_start {
        return Vec3::ZERO;
    }

    let ray_length = t_end - t_start;
    let ds = ray_length / params.view_samples as f32;

    let cos_theta = view_dir.dot(sun_dir);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, params.mie_g);

    let mut sum_r = Vec3::ZERO;
    let mut sum_m = Vec3::ZERO;
    let mut optical_depth_view = Vec3::ZERO;

    for i in 0..params.view_samples {
        let t = t_start + (i as f32 + 0.5) * ds;
        let sample_pos = camera_pos + view_dir * t;
        let height = sample_pos.length() - params.planet_radius;

        let rho_r = rayleigh_density(height, params.rayleigh_scale_height);
        let rho_m = mie_density(height, params.mie_scale_height);
        let rho_o = ozone_density(height, params.ozone_centre_height, params.ozone_width);

        // Accumulate optical depth along view ray.
        optical_depth_view.x += rho_r * ds;
        optical_depth_view.y += rho_m * ds;
        optical_depth_view.z += rho_o * ds;

        // Compute optical depth from this sample point towards the sun.
        let sun_isect = ray_sphere_intersect(sample_pos, sun_dir, params.atmosphere_radius());
        if !sun_isect.hit {
            continue;
        }
        let sun_ray_length = sun_isect.t1;
        if sun_ray_length <= 0.0 {
            continue;
        }

        // Check if sun ray hits planet (in shadow).
        let sun_planet_isect = ray_sphere_intersect(sample_pos, sun_dir, params.planet_radius);
        if sun_planet_isect.hit && sun_planet_isect.t0 > 0.0 && sun_planet_isect.t0 < sun_ray_length {
            continue; // sample point is in the planet's shadow
        }

        let od_sun = compute_optical_depth(
            sample_pos,
            sun_dir,
            sun_ray_length,
            params.light_samples,
            params,
        );

        let tau_combined = extinction_from_optical_depth(optical_depth_view, params)
            + extinction_from_optical_depth(od_sun, params);

        let attenuation = Vec3::new(
            (-tau_combined.x).exp(),
            (-tau_combined.y).exp(),
            (-tau_combined.z).exp(),
        );

        sum_r += attenuation * rho_r * ds;
        sum_m += attenuation * rho_m * ds;
    }

    let inscatter = params.sun_intensity
        * (sum_r * params.rayleigh_coefficients * phase_r
            + sum_m * Vec3::splat(params.mie_coefficient) * phase_m);

    // Ground contribution (simple Lambertian) when the ray hits the planet.
    let ground_color = if planet_hit {
        let ground_point = camera_pos + view_dir * planet_isect.t0;
        let ground_normal = ground_point.normalize();
        let n_dot_l = ground_normal.dot(sun_dir).max(0.0);

        // Transmittance from ground point to top of atmosphere towards sun.
        let ground_sun_isect =
            ray_sphere_intersect(ground_point, sun_dir, params.atmosphere_radius());
        let ground_sun_len = ground_sun_isect.t1.max(0.0);
        let od_ground_sun = compute_optical_depth(
            ground_point,
            sun_dir,
            ground_sun_len,
            params.light_samples,
            params,
        );
        let tau_ground_sun = extinction_from_optical_depth(od_ground_sun, params);
        let trans_ground_sun = Vec3::new(
            (-tau_ground_sun.x).exp(),
            (-tau_ground_sun.y).exp(),
            (-tau_ground_sun.z).exp(),
        );

        // Transmittance along view ray to ground.
        let tau_view = extinction_from_optical_depth(optical_depth_view, params);
        let trans_view = Vec3::new(
            (-tau_view.x).exp(),
            (-tau_view.y).exp(),
            (-tau_view.z).exp(),
        );

        params.ground_albedo * params.sun_intensity * trans_ground_sun * trans_view * n_dot_l
            / PI
    } else {
        Vec3::ZERO
    };

    (inscatter + ground_color) * params.exposure
}

/// Compute sky colour with multi-scattering approximation.
///
/// This adds a second-order scattering term estimated from the average
/// transmittance and isotropic scattering, producing more accurate results
/// at twilight and in thick atmospheres at low additional cost.
pub fn compute_sky_color_multiscatter(
    camera_pos: Vec3,
    view_dir: Vec3,
    params: &AtmosphereParams,
) -> Vec3 {
    // First compute single-scatter.
    let single = compute_sky_color(camera_pos, view_dir, params);

    // Estimate average multi-scatter contribution.
    let sun_dir = params.sun_direction.normalize();
    let atmo_isect = ray_sphere_intersect(camera_pos, view_dir, params.atmosphere_radius());
    if !atmo_isect.hit || atmo_isect.t1 < 0.0 {
        return single;
    }

    let t_start = atmo_isect.t0.max(0.0);
    let planet_isect = ray_sphere_intersect(camera_pos, view_dir, params.planet_radius);
    let planet_hit = planet_isect.hit && planet_isect.t0 > 0.0;
    let t_end = if planet_hit {
        planet_isect.t0.min(atmo_isect.t1)
    } else {
        atmo_isect.t1
    };

    if t_end <= t_start {
        return single;
    }

    let ray_length = t_end - t_start;
    let ds = ray_length / (params.view_samples / 2).max(4) as f32;
    let num_ms_samples = (params.view_samples / 2).max(4);

    let mut ms_sum = Vec3::ZERO;
    let mut od_view = Vec3::ZERO;

    for i in 0..num_ms_samples {
        let t = t_start + (i as f32 + 0.5) * ds;
        let sample_pos = camera_pos + view_dir * t;
        let height = sample_pos.length() - params.planet_radius;
        if height < 0.0 {
            break;
        }

        let rho_r = rayleigh_density(height, params.rayleigh_scale_height);
        let rho_m = mie_density(height, params.mie_scale_height);
        let rho_o = ozone_density(height, params.ozone_centre_height, params.ozone_width);

        od_view.x += rho_r * ds;
        od_view.y += rho_m * ds;
        od_view.z += rho_o * ds;

        // Isotropic scattering coefficient at sample.
        let scatter_coeff = params.rayleigh_coefficients * rho_r
            + Vec3::splat(params.mie_coefficient) * rho_m;

        // Approximate multi-scatter luminance: isotropic phase (1/4pi) ×
        // average inscattered radiance estimate.
        let tau_view = extinction_from_optical_depth(od_view, params);
        let trans = Vec3::new((-tau_view.x).exp(), (-tau_view.y).exp(), (-tau_view.z).exp());

        // Simple multi-scatter estimate: ground albedo feedback loop.
        let ms_factor = params.ground_albedo * 0.5; // crude approximation
        ms_sum += trans * scatter_coeff * ms_factor * ds / (4.0 * PI);
    }

    single + ms_sum * params.sun_intensity * params.exposure
}

// ---------------------------------------------------------------------------
// Transmittance LUT
// ---------------------------------------------------------------------------

/// Dimensions for the transmittance look-up table.
pub const TRANSMITTANCE_LUT_WIDTH: u32 = 256;
pub const TRANSMITTANCE_LUT_HEIGHT: u32 = 64;

/// A precomputed transmittance look-up table stored as a 2-D array of RGB
/// vectors.
///
/// - U axis: cosine of the zenith angle (`cos_zenith` mapped from −1 to 1).
/// - V axis: altitude above the planet surface (0 → atmosphere_height).
///
/// Each texel stores `transmittance(altitude, cos_zenith)`.
#[derive(Clone)]
pub struct TransmittanceLut {
    /// Width of the LUT in texels.
    pub width: u32,
    /// Height of the LUT in texels.
    pub height: u32,
    /// Pixel data in row-major order (width × height), RGB `Vec3`.
    pub data: Vec<Vec3>,
}

impl TransmittanceLut {
    /// Generates the transmittance LUT on the CPU.
    pub fn generate(params: &AtmosphereParams) -> Self {
        Self::generate_with_size(params, TRANSMITTANCE_LUT_WIDTH, TRANSMITTANCE_LUT_HEIGHT)
    }

    /// Generates the transmittance LUT with custom dimensions.
    pub fn generate_with_size(params: &AtmosphereParams, width: u32, height: u32) -> Self {
        let mut data = Vec::with_capacity((width * height) as usize);
        let atmo_radius = params.atmosphere_radius();

        for y in 0..height {
            // Map v ∈ [0,1] to altitude.
            let v = y as f32 / (height - 1).max(1) as f32;
            let altitude = v * params.atmosphere_height;
            let r = params.planet_radius + altitude;

            for x in 0..width {
                // Map u ∈ [0,1] to cos(zenith) ∈ [−1, 1].
                let u = x as f32 / (width - 1).max(1) as f32;
                let cos_zenith = u * 2.0 - 1.0;

                // Construct origin at this altitude on the "up" axis.
                let origin = Vec3::new(0.0, r, 0.0);
                // Zenith direction.
                let sin_zenith = (1.0 - cos_zenith * cos_zenith).max(0.0).sqrt();
                let dir = Vec3::new(sin_zenith, cos_zenith, 0.0).normalize();

                // Intersect with atmosphere outer sphere.
                let isect = ray_sphere_intersect(origin, dir, atmo_radius);
                if !isect.hit || isect.t1 <= 0.0 {
                    data.push(Vec3::ONE);
                    continue;
                }

                // Check ground intersection.
                let planet_isect = ray_sphere_intersect(origin, dir, params.planet_radius);
                let hits_ground = planet_isect.hit && planet_isect.t0 > 0.0;

                let ray_length = if hits_ground {
                    planet_isect.t0.min(isect.t1)
                } else {
                    isect.t1
                };

                let t_start = isect.t0.max(0.0);
                let effective_length = ray_length - t_start;
                if effective_length <= 0.0 {
                    data.push(Vec3::ONE);
                    continue;
                }

                let trans = compute_transmittance(origin, dir, effective_length, 40, params);
                data.push(trans);
            }
        }

        Self {
            width,
            height,
            data,
        }
    }

    /// Samples the LUT with bilinear interpolation.
    ///
    /// `altitude` in km, `cos_zenith` in [−1, 1].
    pub fn sample(&self, altitude: f32, cos_zenith: f32, atmosphere_height: f32) -> Vec3 {
        let u = (cos_zenith * 0.5 + 0.5).clamp(0.0, 1.0);
        let v = (altitude / atmosphere_height).clamp(0.0, 1.0);

        let fx = u * (self.width - 1) as f32;
        let fy = v * (self.height - 1) as f32;

        let x0 = (fx as u32).min(self.width - 2);
        let y0 = (fy as u32).min(self.height - 2);
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let sx = fx - x0 as f32;
        let sy = fy - y0 as f32;

        let idx = |x: u32, y: u32| (y * self.width + x) as usize;

        let c00 = self.data[idx(x0, y0)];
        let c10 = self.data[idx(x1, y0)];
        let c01 = self.data[idx(x0, y1)];
        let c11 = self.data[idx(x1, y1)];

        let top = c00 * (1.0 - sx) + c10 * sx;
        let bot = c01 * (1.0 - sx) + c11 * sx;
        top * (1.0 - sy) + bot * sy
    }

    /// Converts the LUT data to a flat `Vec<f32>` in RGBA format (A = 1.0)
    /// suitable for uploading to a GPU texture.
    pub fn to_rgba_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity((self.width * self.height * 4) as usize);
        for pixel in &self.data {
            out.push(pixel.x);
            out.push(pixel.y);
            out.push(pixel.z);
            out.push(1.0);
        }
        out
    }

    /// Converts the LUT data to a flat `Vec<u8>` in RGBA8 format (gamma-encoded).
    pub fn to_rgba8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity((self.width * self.height * 4) as usize);
        for pixel in &self.data {
            out.push(linear_to_srgb_u8(pixel.x));
            out.push(linear_to_srgb_u8(pixel.y));
            out.push(linear_to_srgb_u8(pixel.z));
            out.push(255);
        }
        out
    }
}

fn linear_to_srgb_u8(v: f32) -> u8 {
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

// ---------------------------------------------------------------------------
// Sky-view LUT (aerial perspective)
// ---------------------------------------------------------------------------

/// Dimensions for the sky-view LUT (azimuth × elevation).
pub const SKYVIEW_LUT_WIDTH: u32 = 192;
pub const SKYVIEW_LUT_HEIGHT: u32 = 108;

/// Precomputed sky-view radiance LUT parameterised by azimuth and elevation
/// relative to the camera position, enabling efficient real-time sky rendering
/// without per-pixel ray marching.
#[derive(Clone)]
pub struct SkyViewLut {
    pub width: u32,
    pub height: u32,
    pub data: Vec<Vec3>,
}

impl SkyViewLut {
    /// Generates the sky-view LUT for a given camera position and atmosphere.
    pub fn generate(camera_pos: Vec3, params: &AtmosphereParams) -> Self {
        Self::generate_with_size(camera_pos, params, SKYVIEW_LUT_WIDTH, SKYVIEW_LUT_HEIGHT)
    }

    /// Generates the sky-view LUT with custom dimensions.
    pub fn generate_with_size(
        camera_pos: Vec3,
        params: &AtmosphereParams,
        width: u32,
        height: u32,
    ) -> Self {
        let mut data = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            let v = y as f32 / (height - 1).max(1) as f32;
            // Map v to elevation angle: [0,1] → [−π/2, π/2].
            let elevation = (v * 2.0 - 1.0) * (PI * 0.5);

            for x in 0..width {
                let u = x as f32 / (width - 1).max(1) as f32;
                // Map u to azimuth: [0, 2π].
                let azimuth = u * 2.0 * PI;

                let cos_el = elevation.cos();
                let view_dir = Vec3::new(
                    cos_el * azimuth.cos(),
                    elevation.sin(),
                    cos_el * azimuth.sin(),
                )
                .normalize();

                let color = compute_sky_color(camera_pos, view_dir, params);
                data.push(color);
            }
        }

        Self {
            width,
            height,
            data,
        }
    }

    /// Samples the sky-view LUT with bilinear interpolation.
    pub fn sample(&self, azimuth: f32, elevation: f32) -> Vec3 {
        let u = (azimuth / (2.0 * PI)).rem_euclid(1.0);
        let v = ((elevation / (PI * 0.5)) * 0.5 + 0.5).clamp(0.0, 1.0);

        let fx = u * (self.width - 1) as f32;
        let fy = v * (self.height - 1) as f32;

        let x0 = (fx as u32).min(self.width - 2);
        let y0 = (fy as u32).min(self.height - 2);
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let sx = fx - x0 as f32;
        let sy = fy - y0 as f32;

        let idx = |x: u32, y: u32| (y * self.width + x) as usize;

        let c00 = self.data[idx(x0, y0)];
        let c10 = self.data[idx(x1, y0)];
        let c01 = self.data[idx(x0, y1)];
        let c11 = self.data[idx(x1, y1)];

        let top = c00 * (1.0 - sx) + c10 * sx;
        let bot = c01 * (1.0 - sx) + c11 * sx;
        top * (1.0 - sy) + bot * sy
    }

    /// Converts to RGBA f32 for GPU upload.
    pub fn to_rgba_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity((self.width * self.height * 4) as usize);
        for pixel in &self.data {
            out.push(pixel.x);
            out.push(pixel.y);
            out.push(pixel.z);
            out.push(1.0);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Aerial Perspective LUT
// ---------------------------------------------------------------------------

/// 3-D LUT for aerial perspective (in-scattering along view rays at various
/// depths).  Parameterised by (azimuth, elevation, depth).
pub const AERIAL_LUT_WIDTH: u32 = 32;
pub const AERIAL_LUT_HEIGHT: u32 = 32;
pub const AERIAL_LUT_DEPTH: u32 = 32;

/// Pre-integrated aerial perspective for distance-fogging geometry with
/// physically accurate atmospheric scattering.
#[derive(Clone)]
pub struct AerialPerspectiveLut {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// RGBA Vec4: RGB = inscattered light, A = transmittance (average of RGB).
    pub data: Vec<Vec4>,
}

impl AerialPerspectiveLut {
    /// Generates the aerial perspective LUT.
    pub fn generate(
        camera_pos: Vec3,
        max_distance: f32,
        params: &AtmosphereParams,
    ) -> Self {
        let width = AERIAL_LUT_WIDTH;
        let height = AERIAL_LUT_HEIGHT;
        let depth = AERIAL_LUT_DEPTH;
        let mut data = Vec::with_capacity((width * height * depth) as usize);

        for z in 0..depth {
            let w = (z as f32 + 0.5) / depth as f32;
            let distance = w * w * max_distance; // quadratic depth distribution

            for y in 0..height {
                let v = y as f32 / (height - 1).max(1) as f32;
                let elevation = (v * 2.0 - 1.0) * (PI * 0.5);

                for x in 0..width {
                    let u = x as f32 / (width - 1).max(1) as f32;
                    let azimuth = u * 2.0 * PI;

                    let cos_el = elevation.cos();
                    let view_dir = Vec3::new(
                        cos_el * azimuth.cos(),
                        elevation.sin(),
                        cos_el * azimuth.sin(),
                    )
                    .normalize();

                    // Compute inscattering and transmittance up to `distance`.
                    let (inscatter, trans) = compute_aerial_perspective_sample(
                        camera_pos, view_dir, distance, params,
                    );

                    let avg_trans = (trans.x + trans.y + trans.z) / 3.0;
                    data.push(Vec4::new(inscatter.x, inscatter.y, inscatter.z, avg_trans));
                }
            }
        }

        Self {
            width,
            height,
            depth,
            data,
        }
    }
}

/// Compute inscattered light and transmittance for a ray segment of given
/// length, used by aerial perspective.
fn compute_aerial_perspective_sample(
    camera_pos: Vec3,
    view_dir: Vec3,
    distance: f32,
    params: &AtmosphereParams,
) -> (Vec3, Vec3) {
    let sun_dir = params.sun_direction.normalize();
    let atmo_isect = ray_sphere_intersect(camera_pos, view_dir, params.atmosphere_radius());

    if !atmo_isect.hit || atmo_isect.t1 < 0.0 {
        return (Vec3::ZERO, Vec3::ONE);
    }

    let t_start = atmo_isect.t0.max(0.0);
    let t_end = distance.min(atmo_isect.t1);
    if t_end <= t_start {
        return (Vec3::ZERO, Vec3::ONE);
    }

    let ray_length = t_end - t_start;
    let num_samples = 16u32;
    let ds = ray_length / num_samples as f32;

    let cos_theta = view_dir.dot(sun_dir);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, params.mie_g);

    let mut sum_r = Vec3::ZERO;
    let mut sum_m = Vec3::ZERO;
    let mut od_view = Vec3::ZERO;

    for i in 0..num_samples {
        let t = t_start + (i as f32 + 0.5) * ds;
        let sample_pos = camera_pos + view_dir * t;
        let height = sample_pos.length() - params.planet_radius;
        if height < 0.0 {
            break;
        }

        let rho_r = rayleigh_density(height, params.rayleigh_scale_height);
        let rho_m = mie_density(height, params.mie_scale_height);
        let rho_o = ozone_density(height, params.ozone_centre_height, params.ozone_width);

        od_view.x += rho_r * ds;
        od_view.y += rho_m * ds;
        od_view.z += rho_o * ds;

        let sun_isect = ray_sphere_intersect(sample_pos, sun_dir, params.atmosphere_radius());
        if !sun_isect.hit || sun_isect.t1 <= 0.0 {
            continue;
        }

        let od_sun = compute_optical_depth(
            sample_pos,
            sun_dir,
            sun_isect.t1,
            8,
            params,
        );

        let tau = extinction_from_optical_depth(od_view, params)
            + extinction_from_optical_depth(od_sun, params);
        let atten = Vec3::new((-tau.x).exp(), (-tau.y).exp(), (-tau.z).exp());

        sum_r += atten * rho_r * ds;
        sum_m += atten * rho_m * ds;
    }

    let inscatter = params.sun_intensity
        * (sum_r * params.rayleigh_coefficients * phase_r
            + sum_m * Vec3::splat(params.mie_coefficient) * phase_m);

    let tau_view = extinction_from_optical_depth(od_view, params);
    let transmittance = Vec3::new(
        (-tau_view.x).exp(),
        (-tau_view.y).exp(),
        (-tau_view.z).exp(),
    );

    (inscatter * params.exposure, transmittance)
}

// ---------------------------------------------------------------------------
// Sun disc rendering
// ---------------------------------------------------------------------------

/// Computes the sun disc radiance with limb darkening.
///
/// Returns radiance to *add* to the sky colour when the sun is within the
/// pixel's solid angle.
pub fn compute_sun_disc(
    view_dir: Vec3,
    params: &AtmosphereParams,
) -> Vec3 {
    let sun_dir = params.sun_direction.normalize();
    let cos_angle = view_dir.dot(sun_dir);
    let half_angle = params.sun_angular_diameter * 0.5;
    let cos_half = half_angle.cos();

    if cos_angle < cos_half {
        return Vec3::ZERO;
    }

    // Normalised angular distance from centre of disc (0 = centre, 1 = limb).
    let angle = cos_angle.acos();
    let r = (angle / half_angle).clamp(0.0, 1.0);

    // Limb darkening using the Neckel-Labs 5-coefficient model for the Sun.
    let limb = limb_darkening(r);

    // Approximate the sun's black-body spectral radiance as white * intensity.
    // Attenuate by transmittance from observer to top of atmosphere towards sun.
    let sun_radiance = params.sun_intensity * limb * 50.0; // scale factor

    sun_radiance
}

/// Neckel-Labs solar limb darkening.
///
/// `r` is the fractional radius from the centre of the disc (0 at centre,
/// 1 at the limb).  Returns a brightness multiplier in [0, 1].
pub fn limb_darkening(r: f32) -> f32 {
    // Polynomial coefficients for the visible range.
    let mu = (1.0 - r * r).max(0.0).sqrt(); // cos of angle from disc centre
    let mu2 = mu * mu;
    let mu3 = mu2 * mu;
    let mu4 = mu3 * mu;
    let mu5 = mu4 * mu;

    // Neckel & Labs (1994) — wavelength-averaged.
    let a0 = 1.0;
    let a1 = -0.397;
    let a2 = 0.237;
    let a3 = -0.134;
    let a4 = 0.073;
    let a5 = -0.014;

    let ld = a0 + a1 * (1.0 - mu) + a2 * (1.0 - mu2) + a3 * (1.0 - mu3)
        + a4 * (1.0 - mu4) + a5 * (1.0 - mu5);

    ld.max(0.0)
}

// ---------------------------------------------------------------------------
// Time-of-day system
// ---------------------------------------------------------------------------

/// Geographic location for computing sun/moon positions.
#[derive(Debug, Clone, Copy)]
pub struct GeoLocation {
    /// Latitude in degrees (−90 to 90).  Positive = North.
    pub latitude: f32,
    /// Longitude in degrees (−180 to 180).  Positive = East.
    pub longitude: f32,
}

impl GeoLocation {
    pub fn new(latitude: f32, longitude: f32) -> Self {
        Self { latitude, longitude }
    }

    /// Returns a default location (roughly San Francisco).
    pub fn default_location() -> Self {
        Self {
            latitude: 37.77,
            longitude: -122.42,
        }
    }
}

/// Time-of-day controller.  Drives sun/moon directions, sky colour tinting,
/// and star visibility from a clock value.
#[derive(Debug, Clone)]
pub struct TimeOfDay {
    /// Current time in hours [0, 24).
    pub time_hours: f32,
    /// Day of the year [1, 365] — affects sun declination.
    pub day_of_year: u32,
    /// Geographic location.
    pub location: GeoLocation,
    /// Speed multiplier for time progression (1.0 = real-time, 60 = 1 min/sec).
    pub speed: f32,
    /// Whether the clock is ticking.
    pub paused: bool,
    /// Ambient light colour contribution at night (moonlight + starlight).
    pub night_ambient: Vec3,
    /// Ambient light colour contribution during the day.
    pub day_ambient: Vec3,
    /// Horizon reddening factor at dawn/dusk.
    pub horizon_reddening: f32,
}

impl TimeOfDay {
    pub fn new() -> Self {
        Self {
            time_hours: 12.0,
            day_of_year: 172, // Summer solstice (northern hemisphere).
            location: GeoLocation::default_location(),
            speed: 1.0,
            paused: false,
            night_ambient: Vec3::new(0.02, 0.03, 0.06),
            day_ambient: Vec3::new(0.15, 0.18, 0.25),
            horizon_reddening: 1.5,
        }
    }

    /// Sets the time directly (0–24).
    pub fn set_time(&mut self, hours: f32) {
        self.time_hours = hours.rem_euclid(24.0);
    }

    /// Advances the clock by `dt` seconds (real-time), scaled by `self.speed`.
    pub fn update(&mut self, dt_seconds: f32) {
        if self.paused {
            return;
        }
        let dt_hours = dt_seconds / 3600.0 * self.speed;
        self.time_hours = (self.time_hours + dt_hours).rem_euclid(24.0);
    }

    /// Computes the sun direction for the current time and location.
    pub fn sun_direction(&self) -> Vec3 {
        compute_sun_direction(self.time_hours, self.day_of_year, &self.location)
    }

    /// Computes the moon direction (roughly opposite the sun with a ~5° tilt).
    pub fn moon_direction(&self) -> Vec3 {
        let sun = self.sun_direction();
        // Approximate: moon roughly opposite sun, offset by orbital
        // inclination ~5.14°.
        let tilt_rad = 5.14_f32.to_radians();
        let rot = Mat3::from_rotation_z(tilt_rad);
        let moon = rot * (-sun);
        moon.normalize()
    }

    /// Returns the sun altitude in radians (negative = below horizon).
    pub fn sun_altitude(&self) -> f32 {
        let dir = self.sun_direction();
        dir.y.asin()
    }

    /// Returns the current ambient colour blended between day and night.
    pub fn ambient_color(&self) -> Vec3 {
        let alt = self.sun_altitude();
        let blend = remap_clamp(alt, -0.1, 0.2, 0.0, 1.0);
        self.night_ambient * (1.0 - blend) + self.day_ambient * blend
    }

    /// Returns a tint colour for the sky at the horizon that intensifies
    /// during dawn and dusk.
    pub fn horizon_tint(&self) -> Vec3 {
        let alt = self.sun_altitude();
        let dawn_dusk = 1.0 - remap_clamp(alt.abs(), 0.0, 0.2, 0.0, 1.0);
        let warm = Vec3::new(1.0, 0.45, 0.15) * dawn_dusk * self.horizon_reddening;
        warm
    }

    /// Returns star visibility factor (0 = invisible, 1 = full brightness).
    pub fn star_visibility(&self) -> f32 {
        let alt = self.sun_altitude();
        remap_clamp(alt, STAR_VISIBLE_ALTITUDE, STAR_FADE_ALTITUDE, 0.0, 1.0)
    }

    /// Applies the current time-of-day to an `AtmosphereParams`, setting the
    /// sun direction and adjusting exposure for twilight.
    pub fn apply_to_params(&self, params: &mut AtmosphereParams) {
        params.sun_direction = self.sun_direction();

        // Lower exposure during twilight/night for more natural look.
        let alt = self.sun_altitude();
        let day_factor = remap_clamp(alt, -0.05, 0.3, 0.0, 1.0);
        params.exposure = 0.5 + day_factor * 0.5;
    }
}

impl Default for TimeOfDay {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the sun direction vector for a given time, day, and location using
/// simplified solar position equations.
pub fn compute_sun_direction(time_hours: f32, day_of_year: u32, location: &GeoLocation) -> Vec3 {
    let lat = location.latitude.to_radians();

    // Solar declination (Spencer, 1971 approximation).
    let day_angle = 2.0 * PI * (day_of_year as f32 - 1.0) / 365.0;
    let declination = 0.006918
        - 0.399912 * day_angle.cos()
        + 0.070257 * day_angle.sin()
        - 0.006758 * (2.0 * day_angle).cos()
        + 0.000907 * (2.0 * day_angle).sin()
        - 0.002697 * (3.0 * day_angle).cos()
        + 0.001480 * (3.0 * day_angle).sin();

    // Hour angle: 0 at solar noon, positive in afternoon.
    // Approximate solar noon at 12:00 local time (ignoring equation of time
    // and longitude correction for simplicity in a game context).
    let hour_angle = ((time_hours - 12.0) * 15.0).to_radians();

    // Solar elevation.
    let sin_elev =
        lat.sin() * declination.sin() + lat.cos() * declination.cos() * hour_angle.cos();
    let elevation = sin_elev.asin();

    // Solar azimuth (from north, clockwise).
    let cos_azimuth = (declination.sin() - lat.sin() * sin_elev) / (lat.cos() * elevation.cos());
    let azimuth = if hour_angle > 0.0 {
        2.0 * PI - cos_azimuth.clamp(-1.0, 1.0).acos()
    } else {
        cos_azimuth.clamp(-1.0, 1.0).acos()
    };

    // Convert to direction vector (Y = up, X = east, Z = south).
    let cos_el = elevation.cos();
    Vec3::new(
        cos_el * azimuth.sin(),
        elevation.sin(),
        -cos_el * azimuth.cos(),
    )
    .normalize()
}

/// Utility: remap `value` from `[in_min, in_max]` to `[out_min, out_max]`,
/// clamped.
#[inline]
fn remap_clamp(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    let t = ((value - in_min) / (in_max - in_min)).clamp(0.0, 1.0);
    out_min + t * (out_max - out_min)
}

// ---------------------------------------------------------------------------
// Star field
// ---------------------------------------------------------------------------

/// A single star in the procedural star field.
#[derive(Debug, Clone, Copy)]
pub struct Star {
    /// Direction on the unit sphere (normalised).
    pub direction: Vec3,
    /// Apparent magnitude (lower = brighter; typically 0–6 for visible stars).
    pub magnitude: f32,
    /// Colour temperature in Kelvin (used to derive RGB tint).
    pub temperature: f32,
}

/// Procedural star field generator and renderer.
#[derive(Debug, Clone)]
pub struct StarField {
    /// Pre-generated stars.
    pub stars: Vec<Star>,
    /// Overall brightness multiplier.
    pub intensity: f32,
    /// Twinkle animation speed.
    pub twinkle_speed: f32,
    /// Twinkle amplitude (0 = no twinkle, 1 = full on/off).
    pub twinkle_amount: f32,
}

impl StarField {
    /// Generates a star field with `count` stars using a deterministic hash.
    pub fn generate(count: u32, seed: u64) -> Self {
        let mut stars = Vec::with_capacity(count as usize);
        let mut rng = SimpleRng::new(seed);

        for _ in 0..count {
            // Uniform distribution on the sphere via Marsaglia's method.
            let (x, y, z) = loop {
                let u = rng.next_f32() * 2.0 - 1.0;
                let v = rng.next_f32() * 2.0 - 1.0;
                let s = u * u + v * v;
                if s < 1.0 {
                    let factor = 2.0 * (1.0 - s).sqrt();
                    break (u * factor, v * factor, 1.0 - 2.0 * s);
                }
            };

            // Magnitude distribution: more faint stars.
            let mag = rng.next_f32() * 6.0;
            // Temperature: roughly 3000K–30000K, biased towards mid-range.
            let temp = 3000.0 + rng.next_f32().powf(0.5) * 27000.0;

            stars.push(Star {
                direction: Vec3::new(x, y, z).normalize(),
                magnitude: mag,
                temperature: temp,
            });
        }

        Self {
            stars,
            intensity: 1.0,
            twinkle_speed: 1.5,
            twinkle_amount: 0.3,
        }
    }

    /// Returns the colour (linear RGB) for a star given its magnitude and
    /// temperature, including an optional twinkle animation.
    pub fn star_color(&self, star: &Star, time: f32) -> Vec3 {
        // Brightness from magnitude: flux ∝ 10^(−0.4 × mag).
        let brightness = 10.0_f32.powf(-0.4 * star.magnitude) * self.intensity;

        // Twinkle.
        let phase = star.direction.x * 1234.5 + star.direction.z * 5678.9;
        let twinkle = 1.0
            - self.twinkle_amount
                * ((time * self.twinkle_speed + phase).sin() * 0.5 + 0.5);

        // Colour from temperature via simplified Planckian locus.
        let tint = temperature_to_rgb(star.temperature);

        tint * brightness * twinkle
    }

    /// Returns colours for all stars as a flat buffer (RGB f32), suitable
    /// for instanced rendering of point sprites.
    pub fn compute_all_colors(&self, time: f32) -> Vec<Vec3> {
        self.stars.iter().map(|s| self.star_color(s, time)).collect()
    }

    /// Returns vertex data for rendering stars as point sprites.
    /// Each star is `(position: Vec3, color: Vec3, size: f32)`.
    pub fn vertex_data(&self, time: f32) -> Vec<(Vec3, Vec3, f32)> {
        self.stars
            .iter()
            .map(|s| {
                let color = self.star_color(s, time);
                let size = (6.5 - s.magnitude).max(0.5) * 0.5;
                (s.direction, color, size)
            })
            .collect()
    }
}

/// Converts a colour temperature (Kelvin) to a linear RGB triplet using
/// Tanner Helland's approximation.
pub fn temperature_to_rgb(kelvin: f32) -> Vec3 {
    let temp = (kelvin / 100.0).clamp(10.0, 400.0);

    let r = if temp <= 66.0 {
        1.0
    } else {
        let x = temp - 60.0;
        (329.698_73 * x.powf(-0.133_204_76) / 255.0).clamp(0.0, 1.0)
    };

    let g = if temp <= 66.0 {
        let x = temp;
        (99.470_8 * x.ln() - 161.119_57) / 255.0
    } else {
        let x = temp - 60.0;
        288.122_17 * x.powf(-0.075_514_85) / 255.0
    }
    .clamp(0.0, 1.0);

    let b = if temp >= 66.0 {
        1.0
    } else if temp <= 19.0 {
        0.0
    } else {
        let x = temp - 10.0;
        (138.517_73 * x.ln() - 305.044_79) / 255.0
    }
    .clamp(0.0, 1.0);

    Vec3::new(r, g, b)
}

// ---------------------------------------------------------------------------
// Moon
// ---------------------------------------------------------------------------

/// Lunar phase identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LunarPhase {
    NewMoon,
    WaxingCrescent,
    FirstQuarter,
    WaxingGibbous,
    FullMoon,
    WaningGibbous,
    LastQuarter,
    WaningCrescent,
}

/// Moon rendering data.
#[derive(Debug, Clone)]
pub struct Moon {
    /// Direction towards the moon (normalised).
    pub direction: Vec3,
    /// Angular diameter in radians (Earth-Moon ≈ 0.00907).
    pub angular_diameter: f32,
    /// Surface brightness multiplier (full moon ≈ 1.0).
    pub brightness: f32,
    /// Phase angle in radians [0, 2π).
    pub phase_angle: f32,
    /// Albedo/colour tint.
    pub tint: Vec3,
}

impl Moon {
    pub fn new() -> Self {
        Self {
            direction: Vec3::new(0.0, 0.5, 0.5).normalize(),
            angular_diameter: 0.009_07,
            brightness: 0.15,
            phase_angle: 0.0,
            tint: Vec3::new(0.9, 0.92, 1.0),
        }
    }

    /// Computes the lunar phase from the phase angle.
    pub fn phase(&self) -> LunarPhase {
        let normalised = (self.phase_angle / (2.0 * PI)).rem_euclid(1.0);
        match (normalised * 8.0) as u32 {
            0 => LunarPhase::NewMoon,
            1 => LunarPhase::WaxingCrescent,
            2 => LunarPhase::FirstQuarter,
            3 => LunarPhase::WaxingGibbous,
            4 => LunarPhase::FullMoon,
            5 => LunarPhase::WaningGibbous,
            6 => LunarPhase::LastQuarter,
            _ => LunarPhase::WaningCrescent,
        }
    }

    /// Illumination fraction [0, 1] (0 = new moon, 1 = full moon).
    pub fn illumination_fraction(&self) -> f32 {
        (1.0 - self.phase_angle.cos()) * 0.5
    }

    /// Updates moon position and phase for the given day-of-year and time.
    ///
    /// Uses a very simplified synodic model (29.53 day cycle).
    pub fn update(&mut self, day_of_year: u32, time_hours: f32, sun_dir: Vec3) {
        let synodic_period = 29.530_59;
        let fractional_day = day_of_year as f32 + time_hours / 24.0;
        // Assume new moon at day 0 (arbitrary epoch).
        self.phase_angle = (fractional_day / synodic_period * 2.0 * PI).rem_euclid(2.0 * PI);

        // Place moon direction based on phase.
        let moon_angle_from_sun = self.phase_angle;
        let up = Vec3::Y;
        let sun_n = sun_dir.normalize();
        let right = sun_n.cross(up).normalize();
        let orbital_up = right.cross(sun_n).normalize();

        // Moon orbits in the plane defined by sun direction, with 5° tilt.
        let tilt = 5.14_f32.to_radians();
        self.direction = (sun_n * moon_angle_from_sun.cos()
            + right * moon_angle_from_sun.sin() * tilt.cos()
            + orbital_up * moon_angle_from_sun.sin() * tilt.sin())
        .normalize();

        // Brightness varies with phase.
        self.brightness = 0.05 + 0.15 * self.illumination_fraction();
    }

    /// Returns the moon disc colour at a given angular distance from its
    /// centre direction, accounting for limb darkening and phase.
    pub fn disc_color(&self, view_dir: Vec3) -> Vec3 {
        let cos_angle = view_dir.dot(self.direction);
        let half_angle = self.angular_diameter * 0.5;
        let cos_half = half_angle.cos();

        if cos_angle < cos_half {
            return Vec3::ZERO;
        }

        let angle = cos_angle.acos();
        let r = (angle / half_angle).clamp(0.0, 1.0);

        // Simple lunar limb darkening.
        let mu = (1.0 - r * r).max(0.0).sqrt();
        let ld = 0.6 + 0.4 * mu;

        // Phase shadow: darken part of the disc based on phase angle.
        // Project the sample point onto the phase terminator axis.
        // For a 2D approximation on the disc face:
        let x_on_disc = if half_angle > 1e-6 { angle / half_angle } else { 0.0 };
        let illuminated = phase_illumination(x_on_disc, r, self.phase_angle);

        self.tint * self.brightness * ld * illuminated
    }
}

impl Default for Moon {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple phase-based illumination for moon disc rendering.
/// Returns 0.0 (shadowed) to 1.0 (fully lit).
fn phase_illumination(x_norm: f32, r_norm: f32, phase_angle: f32) -> f32 {
    // Phase determines the terminator position on the disc.
    let illum_frac = (1.0 - phase_angle.cos()) * 0.5;

    // If r_norm is close to 1 (edge), apply softer blending.
    let terminator = (illum_frac * 2.0 - 1.0).clamp(-1.0, 1.0);

    // Smooth step at the terminator line.
    let edge = x_norm * (if phase_angle < PI { 1.0 } else { -1.0 });
    let alpha = smoothstep(-0.1, 0.1, edge - terminator);

    alpha.clamp(0.0, 1.0)
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Cloud layer
// ---------------------------------------------------------------------------

/// Simple single-layer cloud system driven by noise.
#[derive(Debug, Clone)]
pub struct CloudLayer {
    /// Altitude of the cloud layer in km above sea level.
    pub altitude: f32,
    /// Thickness of the cloud layer in km.
    pub thickness: f32,
    /// Overall cloud coverage [0, 1].
    pub coverage: f32,
    /// Cloud density multiplier.
    pub density: f32,
    /// Cloud base colour.
    pub base_color: Vec3,
    /// Cloud scatter colour (lit side).
    pub scatter_color: Vec3,
    /// Noise frequency for cloud shape.
    pub noise_frequency: f32,
    /// Wind direction (XZ plane) for cloud movement.
    pub wind_direction: Vec2,
    /// Wind speed (km/s game time).
    pub wind_speed: f32,
    /// Accumulated wind offset (updated each frame).
    pub wind_offset: Vec2,
    /// Number of ray march steps through the cloud layer.
    pub march_steps: u32,
    /// Absorption coefficient.
    pub absorption: f32,
    /// Forward scattering asymmetry for the cloud phase function.
    pub phase_g: f32,
}

impl CloudLayer {
    pub fn new() -> Self {
        Self {
            altitude: 5.0,
            thickness: 1.0,
            coverage: 0.5,
            density: 0.8,
            base_color: Vec3::new(0.85, 0.87, 0.9),
            scatter_color: Vec3::new(1.0, 0.98, 0.95),
            noise_frequency: 0.5,
            wind_direction: Vec2::new(1.0, 0.0),
            wind_speed: 0.01,
            wind_offset: Vec2::ZERO,
            march_steps: 16,
            absorption: 0.3,
            phase_g: 0.6,
        }
    }

    /// Advances the wind offset.
    pub fn update(&mut self, dt: f32) {
        self.wind_offset += self.wind_direction.normalize() * self.wind_speed * dt;
    }

    /// Samples the cloud density at a given 3D position (in km, planet-space).
    ///
    /// Uses a combination of value noise octaves and a coverage threshold.
    pub fn sample_density(&self, position: Vec3) -> f32 {
        let height = position.length() - (self.altitude + 6371.0); // relative to cloud base
        if height < 0.0 || height > self.thickness {
            return 0.0;
        }

        // Height gradient: densest at bottom, thinning at top.
        let height_fraction = height / self.thickness;
        let height_gradient = smoothstep(0.0, 0.1, height_fraction)
            * smoothstep(1.0, 0.6, height_fraction);

        // Noise sampling.
        let xz = Vec2::new(position.x, position.z) + self.wind_offset;
        let noise_val = fbm_value_noise_2d(
            xz.x * self.noise_frequency,
            xz.y * self.noise_frequency,
            4, // octaves
        );

        // Coverage remapping.
        let raw_density = remap_clamp(noise_val, 1.0 - self.coverage, 1.0, 0.0, 1.0);

        raw_density * height_gradient * self.density
    }

    /// Computes the cloud colour and opacity for a given view ray using
    /// a simplified single-scattering ray march.
    pub fn ray_march(
        &self,
        ray_origin: Vec3,
        ray_dir: Vec3,
        sun_dir: Vec3,
        planet_radius: f32,
    ) -> (Vec3, f32) {
        // Intersect ray with cloud layer spherical shells.
        let inner_radius = planet_radius + self.altitude;
        let outer_radius = inner_radius + self.thickness;

        let inner_isect = ray_sphere_intersect(ray_origin, ray_dir, inner_radius);
        let outer_isect = ray_sphere_intersect(ray_origin, ray_dir, outer_radius);

        if !outer_isect.hit {
            return (Vec3::ZERO, 0.0);
        }

        // Determine ray segment through the cloud layer.
        let t_start;
        let t_end;

        let cam_height = ray_origin.length() - planet_radius;

        if cam_height < self.altitude {
            // Below clouds: enter at inner shell, exit at outer shell.
            if !inner_isect.hit {
                return (Vec3::ZERO, 0.0);
            }
            t_start = inner_isect.t0.max(0.0);
            t_end = outer_isect.t1;
        } else if cam_height > self.altitude + self.thickness {
            // Above clouds: enter at outer shell, exit at outer shell (or inner).
            t_start = outer_isect.t0.max(0.0);
            t_end = if inner_isect.hit && inner_isect.t0 > 0.0 {
                inner_isect.t0
            } else {
                outer_isect.t1
            };
        } else {
            // Inside cloud layer.
            t_start = 0.0;
            t_end = outer_isect.t1;
        }

        if t_end <= t_start {
            return (Vec3::ZERO, 0.0);
        }

        let ds = (t_end - t_start) / self.march_steps as f32;
        let cos_theta = ray_dir.dot(sun_dir);
        let phase = henyey_greenstein_phase(cos_theta, self.phase_g);

        let mut accumulated_color = Vec3::ZERO;
        let mut transmittance = 1.0_f32;

        for i in 0..self.march_steps {
            if transmittance < 0.01 {
                break;
            }

            let t = t_start + (i as f32 + 0.5) * ds;
            let pos = ray_origin + ray_dir * t;
            let density = self.sample_density(pos);

            if density <= 0.001 {
                continue;
            }

            // Light march towards sun (simplified: 4 steps).
            let light_atten = self.light_march(pos, sun_dir, planet_radius, 4);

            // Scattering.
            let scatter = self.scatter_color * light_atten * phase
                + self.base_color * 0.05; // ambient

            let extinction = density * self.absorption * ds;
            let sample_trans = (-extinction).exp();

            // Energy-conserving integration.
            let integrated_scatter = scatter * (1.0 - sample_trans) / self.absorption.max(0.001);
            accumulated_color += transmittance * integrated_scatter;
            transmittance *= sample_trans;
        }

        let opacity = 1.0 - transmittance;
        (accumulated_color, opacity)
    }

    /// Simplified light-march from a point towards the sun through the cloud layer.
    /// Returns an attenuation factor [0, 1].
    fn light_march(&self, origin: Vec3, sun_dir: Vec3, planet_radius: f32, steps: u32) -> f32 {
        let outer_radius = planet_radius + self.altitude + self.thickness;
        let isect = ray_sphere_intersect(origin, sun_dir, outer_radius);
        if !isect.hit {
            return 1.0;
        }

        let march_length = isect.t1.max(0.0).min(self.thickness * 2.0);
        let ds = march_length / steps as f32;
        let mut optical_depth = 0.0_f32;

        for i in 0..steps {
            let t = (i as f32 + 0.5) * ds;
            let pos = origin + sun_dir * t;
            optical_depth += self.sample_density(pos) * ds;
        }

        (-optical_depth * self.absorption).exp()
    }
}

impl Default for CloudLayer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Noise utilities (used by clouds and stars)
// ---------------------------------------------------------------------------

/// Simple deterministic pseudo-random number generator (xorshift64).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
    }
}

/// 2-D value noise with smooth (quintic) interpolation.
fn value_noise_2d(x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - x.floor();
    let fy = y - y.floor();

    let sx = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let sy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

    let n00 = hash_2d(ix, iy);
    let n10 = hash_2d(ix + 1, iy);
    let n01 = hash_2d(ix, iy + 1);
    let n11 = hash_2d(ix + 1, iy + 1);

    let nx0 = n00 * (1.0 - sx) + n10 * sx;
    let nx1 = n01 * (1.0 - sx) + n11 * sx;
    nx0 * (1.0 - sy) + nx1 * sy
}

/// Hashed value for 2-D noise lattice points.  Returns [0, 1].
fn hash_2d(x: i32, y: i32) -> f32 {
    let n = (x.wrapping_mul(374761393))
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(1376312589);
    let n = n.wrapping_mul(n).wrapping_mul(n);
    let n = (n >> 13) ^ n;
    let n = n.wrapping_mul(n.wrapping_mul(n.wrapping_mul(15731).wrapping_add(789221)).wrapping_add(1376312589));
    ((n & 0x7fff_ffff) as f32) / 0x7fff_ffff as f32
}

/// Fractal Brownian motion built on `value_noise_2d`.
fn fbm_value_noise_2d(x: f32, y: f32, octaves: u32) -> f32 {
    let mut sum = 0.0_f32;
    let mut amplitude = 0.5;
    let mut frequency = 1.0_f32;
    let mut max_val = 0.0_f32;

    for _ in 0..octaves {
        sum += value_noise_2d(x * frequency, y * frequency) * amplitude;
        max_val += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    sum / max_val
}

// ---------------------------------------------------------------------------
// SkyBox component
// ---------------------------------------------------------------------------

/// Source type for a sky box.
#[derive(Debug, Clone)]
pub enum SkyboxSource {
    /// Six-face cubemap texture (array of 6 texture handles/paths).
    Cubemap {
        /// Paths or handles for +X, −X, +Y, −Y, +Z, −Z.
        faces: [String; 6],
    },
    /// Equirectangular HDR environment map.
    Equirectangular {
        /// Path to the HDR image.
        path: String,
    },
    /// Procedurally generated from the atmosphere model.
    Procedural,
}

/// Sky component attachable to a scene/entity.
#[derive(Debug, Clone)]
pub struct SkyBox {
    /// The source of sky imagery.
    pub source: SkyboxSource,
    /// Rotation (Euler angles in radians) applied to the sky before rendering.
    pub rotation: Vec3,
    /// Brightness/intensity multiplier.
    pub intensity: f32,
    /// Whether this sky box contributes to ambient/environment lighting.
    pub affects_lighting: bool,
    /// LOD bias for cubemap sampling.
    pub lod_bias: f32,
    /// Tint colour multiplied into the sky.
    pub tint: Vec3,
}

impl SkyBox {
    /// Creates a procedural sky box.
    pub fn procedural() -> Self {
        Self {
            source: SkyboxSource::Procedural,
            rotation: Vec3::ZERO,
            intensity: 1.0,
            affects_lighting: true,
            lod_bias: 0.0,
            tint: Vec3::ONE,
        }
    }

    /// Creates a cubemap sky box.
    pub fn cubemap(faces: [String; 6]) -> Self {
        Self {
            source: SkyboxSource::Cubemap { faces },
            rotation: Vec3::ZERO,
            intensity: 1.0,
            affects_lighting: true,
            lod_bias: 0.0,
            tint: Vec3::ONE,
        }
    }

    /// Creates an equirectangular HDR sky box.
    pub fn equirectangular(path: impl Into<String>) -> Self {
        Self {
            source: SkyboxSource::Equirectangular { path: path.into() },
            rotation: Vec3::ZERO,
            intensity: 1.0,
            affects_lighting: true,
            lod_bias: 0.0,
            tint: Vec3::ONE,
        }
    }
}

impl Default for SkyBox {
    fn default() -> Self {
        Self::procedural()
    }
}

// ---------------------------------------------------------------------------
// Atmosphere render state (ties everything together for a frame)
// ---------------------------------------------------------------------------

/// Complete atmosphere/sky rendering state for a single frame.
#[derive(Clone)]
pub struct AtmosphereRenderState {
    pub params: AtmosphereParams,
    pub time_of_day: TimeOfDay,
    pub star_field: StarField,
    pub moon: Moon,
    pub cloud_layer: CloudLayer,
    pub transmittance_lut: Option<TransmittanceLut>,
    pub sky_view_lut: Option<SkyViewLut>,
    pub lut_dirty: bool,
}

impl AtmosphereRenderState {
    /// Creates a new render state with Earth defaults.
    pub fn new() -> Self {
        Self {
            params: AtmosphereParams::earth(),
            time_of_day: TimeOfDay::new(),
            star_field: StarField::generate(4000, 42),
            moon: Moon::new(),
            cloud_layer: CloudLayer::new(),
            transmittance_lut: None,
            sky_view_lut: None,
            lut_dirty: true,
        }
    }

    /// Updates time, moon, clouds, and marks LUTs as needing regeneration.
    pub fn update(&mut self, dt: f32) {
        self.time_of_day.update(dt);
        self.time_of_day.apply_to_params(&mut self.params);

        self.moon.update(
            self.time_of_day.day_of_year,
            self.time_of_day.time_hours,
            self.params.sun_direction,
        );

        self.cloud_layer.update(dt);
        self.lut_dirty = true;
    }

    /// Regenerates LUTs if dirty.  In a real engine this would be dispatched
    /// as a GPU compute pass; here we show the CPU fallback.
    pub fn rebuild_luts(&mut self, camera_pos: Vec3) {
        if !self.lut_dirty {
            return;
        }
        self.transmittance_lut = Some(TransmittanceLut::generate(&self.params));
        self.sky_view_lut = Some(SkyViewLut::generate(camera_pos, &self.params));
        self.lut_dirty = false;
    }

    /// Computes the final sky colour for a single view direction, compositing
    /// atmosphere + sun disc + stars + moon + clouds.
    pub fn compute_pixel(
        &self,
        camera_pos: Vec3,
        view_dir: Vec3,
    ) -> Vec3 {
        // Base sky from atmosphere.
        let mut color = compute_sky_color(camera_pos, view_dir, &self.params);

        // Sun disc.
        let sun_disc = compute_sun_disc(view_dir, &self.params);
        // Attenuate sun disc by transmittance along view ray.
        let sun_transmittance = {
            let atmo = ray_sphere_intersect(camera_pos, view_dir, self.params.atmosphere_radius());
            if atmo.hit && atmo.t1 > 0.0 {
                compute_transmittance(
                    camera_pos,
                    view_dir,
                    atmo.t1,
                    8,
                    &self.params,
                )
            } else {
                Vec3::ONE
            }
        };
        color += sun_disc * sun_transmittance;

        // Moon.
        let moon_color = self.moon.disc_color(view_dir);
        color += moon_color;

        // Stars (fade with sun altitude).
        let star_vis = self.time_of_day.star_visibility();
        if star_vis > 0.0 {
            // Find closest star and add its contribution.
            for star in &self.star_field.stars {
                let cos_angle = view_dir.dot(star.direction);
                if cos_angle > 0.9999 {
                    let star_c =
                        self.star_field.star_color(star, self.time_of_day.time_hours * 360.0);
                    color += star_c * star_vis;
                }
            }
        }

        // Clouds.
        let (cloud_color, cloud_opacity) = self.cloud_layer.ray_march(
            camera_pos,
            view_dir,
            self.params.sun_direction,
            self.params.planet_radius,
        );
        color = color * (1.0 - cloud_opacity) + cloud_color;

        // Horizon reddening.
        let horizon_tint = self.time_of_day.horizon_tint();
        let horizon_factor = 1.0 - view_dir.y.abs();
        let horizon_factor = horizon_factor * horizon_factor * horizon_factor;
        color += horizon_tint * horizon_factor * 0.1;

        color
    }
}

impl Default for AtmosphereRenderState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WGSL compute shader for sky LUT generation
// ---------------------------------------------------------------------------

/// WGSL compute shader source for generating the transmittance LUT on the GPU.
pub const TRANSMITTANCE_LUT_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Transmittance LUT compute shader (Genovo Engine)
// -----------------------------------------------------------------------
// Dispatched as (LUT_WIDTH / 8, LUT_HEIGHT / 8, 1).
// Each invocation computes one texel of the transmittance LUT.

struct AtmosphereUniforms {
    planet_radius: f32,
    atmosphere_radius: f32,
    rayleigh_scale_height: f32,
    mie_scale_height: f32,
    rayleigh_coefficients: vec3<f32>,
    mie_coefficient: f32,
    mie_absorption: f32,
    ozone_coefficients: vec3<f32>,
    ozone_centre_height: f32,
    ozone_width: f32,
    sun_direction: vec3<f32>,
    sun_intensity: vec3<f32>,
    mie_g: f32,
    lut_width: u32,
    lut_height: u32,
};

@group(0) @binding(0) var<uniform> atmo: AtmosphereUniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;

const NUM_SAMPLES: u32 = 40u;
const PI: f32 = 3.141592653589793;

fn ray_sphere_intersect(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let a = dot(dir, dir);
    let b = 2.0 * dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return vec2<f32>(-1.0, -1.0);
    }
    let sq = sqrt(disc);
    let inv = 0.5 / a;
    return vec2<f32>((-b - sq) * inv, (-b + sq) * inv);
}

fn rayleigh_density(height: f32) -> f32 {
    return exp(-height / atmo.rayleigh_scale_height);
}

fn mie_density(height: f32) -> f32 {
    return exp(-height / atmo.mie_scale_height);
}

fn ozone_density(height: f32) -> f32 {
    return max(0.0, 1.0 - abs(height - atmo.ozone_centre_height) / atmo.ozone_width);
}

fn compute_optical_depth_local(origin: vec3<f32>, dir: vec3<f32>, ray_len: f32) -> vec3<f32> {
    let ds = ray_len / f32(NUM_SAMPLES);
    var od = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < NUM_SAMPLES; i = i + 1u) {
        let t = (f32(i) + 0.5) * ds;
        let pos = origin + dir * t;
        let h = length(pos) - atmo.planet_radius;
        if h < 0.0 { return vec3<f32>(1e10); }
        od.x += rayleigh_density(h) * ds;
        od.y += mie_density(h) * ds;
        od.z += ozone_density(h) * ds;
    }
    return od;
}

fn extinction_from_od(od: vec3<f32>) -> vec3<f32> {
    return atmo.rayleigh_coefficients * od.x
         + vec3<f32>(atmo.mie_coefficient + atmo.mie_absorption) * od.y
         + atmo.ozone_coefficients * od.z;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= atmo.lut_width || gid.y >= atmo.lut_height {
        return;
    }

    let u = f32(gid.x) / f32(atmo.lut_width - 1u);
    let v = f32(gid.y) / f32(atmo.lut_height - 1u);

    // Map u -> cos(zenith) in [-1, 1], v -> altitude in [0, atmo_height].
    let cos_zenith = u * 2.0 - 1.0;
    let altitude = v * (atmo.atmosphere_radius - atmo.planet_radius);
    let r = atmo.planet_radius + altitude;

    let origin = vec3<f32>(0.0, r, 0.0);
    let sin_zenith = sqrt(max(0.0, 1.0 - cos_zenith * cos_zenith));
    let dir = normalize(vec3<f32>(sin_zenith, cos_zenith, 0.0));

    let isect = ray_sphere_intersect(origin, dir, atmo.atmosphere_radius);
    if isect.y <= 0.0 {
        textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)),
                     vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }

    let planet_isect = ray_sphere_intersect(origin, dir, atmo.planet_radius);
    var ray_len = isect.y;
    if planet_isect.x > 0.0 && planet_isect.x < ray_len {
        ray_len = planet_isect.x;
    }
    let t_start = max(isect.x, 0.0);
    ray_len = ray_len - t_start;
    if ray_len <= 0.0 {
        textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)),
                     vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }

    let od = compute_optical_depth_local(origin, dir, ray_len);
    let tau = extinction_from_od(od);
    let trans = exp(-tau);

    textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(trans, 1.0));
}
"#;

/// WGSL compute shader source for generating the sky-view LUT on the GPU.
pub const SKY_VIEW_LUT_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Sky-View LUT compute shader (Genovo Engine)
// -----------------------------------------------------------------------
// Dispatched as (LUT_WIDTH / 8, LUT_HEIGHT / 8, 1).

struct AtmosphereUniforms {
    planet_radius: f32,
    atmosphere_radius: f32,
    rayleigh_scale_height: f32,
    mie_scale_height: f32,
    rayleigh_coefficients: vec3<f32>,
    mie_coefficient: f32,
    mie_absorption: f32,
    ozone_coefficients: vec3<f32>,
    ozone_centre_height: f32,
    ozone_width: f32,
    sun_direction: vec3<f32>,
    sun_intensity: vec3<f32>,
    mie_g: f32,
    lut_width: u32,
    lut_height: u32,
    camera_pos: vec3<f32>,
};

@group(0) @binding(0) var<uniform> atmo: AtmosphereUniforms;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var transmittance_sampler: sampler;
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba16float, write>;

const VIEW_SAMPLES: u32 = 32u;
const LIGHT_SAMPLES: u32 = 8u;
const PI: f32 = 3.141592653589793;

fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let a = dot(dir, dir);
    let b = 2.0 * dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 { return vec2<f32>(-1.0, -1.0); }
    let sq = sqrt(disc);
    let inv = 0.5 / a;
    return vec2<f32>((-b - sq) * inv, (-b + sq) * inv);
}

fn rayleigh_phase_fn(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn mie_phase_fn(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = 3.0 * (1.0 - g2) * (1.0 + cos_theta * cos_theta);
    let denom_base = 1.0 + g2 - 2.0 * g * cos_theta;
    return num / (8.0 * PI * (2.0 + g2) * denom_base * sqrt(denom_base));
}

fn sample_transmittance(altitude: f32, cos_z: f32) -> vec3<f32> {
    let u = cos_z * 0.5 + 0.5;
    let v = altitude / (atmo.atmosphere_radius - atmo.planet_radius);
    return textureSampleLevel(transmittance_lut, transmittance_sampler,
                              vec2<f32>(u, v), 0.0).rgb;
}

fn rho_rayleigh(h: f32) -> f32 { return exp(-h / atmo.rayleigh_scale_height); }
fn rho_mie(h: f32) -> f32 { return exp(-h / atmo.mie_scale_height); }
fn rho_ozone(h: f32) -> f32 {
    return max(0.0, 1.0 - abs(h - atmo.ozone_centre_height) / atmo.ozone_width);
}

fn extinction(od: vec3<f32>) -> vec3<f32> {
    return atmo.rayleigh_coefficients * od.x
         + vec3<f32>(atmo.mie_coefficient + atmo.mie_absorption) * od.y
         + atmo.ozone_coefficients * od.z;
}

fn optical_depth_segment(origin: vec3<f32>, dir: vec3<f32>, len: f32, samples: u32) -> vec3<f32> {
    let ds = len / f32(samples);
    var od = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < samples; i++) {
        let t = (f32(i) + 0.5) * ds;
        let p = origin + dir * t;
        let h = length(p) - atmo.planet_radius;
        if h < 0.0 { return vec3<f32>(1e10); }
        od.x += rho_rayleigh(h) * ds;
        od.y += rho_mie(h) * ds;
        od.z += rho_ozone(h) * ds;
    }
    return od;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= atmo.lut_width || gid.y >= atmo.lut_height { return; }

    let u = f32(gid.x) / f32(atmo.lut_width - 1u);
    let v = f32(gid.y) / f32(atmo.lut_height - 1u);

    let azimuth = u * 2.0 * PI;
    let elevation = (v * 2.0 - 1.0) * PI * 0.5;

    let ce = cos(elevation);
    let view_dir = normalize(vec3<f32>(ce * cos(azimuth), sin(elevation), ce * sin(azimuth)));
    let sun_dir = normalize(atmo.sun_direction);

    let atmo_isect = ray_sphere(atmo.camera_pos, view_dir, atmo.atmosphere_radius);
    if atmo_isect.y <= 0.0 {
        textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)),
                     vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let planet_isect = ray_sphere(atmo.camera_pos, view_dir, atmo.planet_radius);
    let planet_hit = planet_isect.x > 0.0;

    let t_start = max(atmo_isect.x, 0.0);
    var t_end = atmo_isect.y;
    if planet_hit { t_end = min(planet_isect.x, t_end); }

    let ray_len = t_end - t_start;
    if ray_len <= 0.0 {
        textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)),
                     vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let ds = ray_len / f32(VIEW_SAMPLES);
    let cos_theta = dot(view_dir, sun_dir);
    let ph_r = rayleigh_phase_fn(cos_theta);
    let ph_m = mie_phase_fn(cos_theta, atmo.mie_g);

    var sum_r = vec3<f32>(0.0);
    var sum_m = vec3<f32>(0.0);
    var od_view = vec3<f32>(0.0);

    for (var i: u32 = 0u; i < VIEW_SAMPLES; i++) {
        let t = t_start + (f32(i) + 0.5) * ds;
        let pos = atmo.camera_pos + view_dir * t;
        let h = length(pos) - atmo.planet_radius;
        if h < 0.0 { break; }

        let rr = rho_rayleigh(h);
        let rm = rho_mie(h);
        let ro = rho_ozone(h);

        od_view.x += rr * ds;
        od_view.y += rm * ds;
        od_view.z += ro * ds;

        let sun_isect = ray_sphere(pos, sun_dir, atmo.atmosphere_radius);
        if sun_isect.y <= 0.0 { continue; }

        // Shadow check.
        let sun_planet = ray_sphere(pos, sun_dir, atmo.planet_radius);
        if sun_planet.x > 0.0 && sun_planet.x < sun_isect.y { continue; }

        let od_sun = optical_depth_segment(pos, sun_dir, sun_isect.y, LIGHT_SAMPLES);
        let tau = extinction(od_view) + extinction(od_sun);
        let atten = exp(-tau);

        sum_r += atten * rr * ds;
        sum_m += atten * rm * ds;
    }

    let inscatter = atmo.sun_intensity * (
        sum_r * atmo.rayleigh_coefficients * ph_r +
        sum_m * vec3<f32>(atmo.mie_coefficient) * ph_m
    );

    textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(inscatter, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rayleigh_phase_normalisation() {
        // Integral of Rayleigh phase over sphere should be 1.
        let n = 1000;
        let mut sum = 0.0_f64;
        for i in 0..n {
            let cos_theta = -1.0 + 2.0 * i as f64 / n as f64;
            let p = rayleigh_phase(cos_theta as f32) as f64;
            sum += p * 2.0 * std::f64::consts::PI * 2.0 / n as f64;
        }
        assert!((sum - 1.0).abs() < 0.05, "Rayleigh phase integral = {sum}");
    }

    #[test]
    fn mie_phase_forward_peak() {
        let forward = mie_phase(1.0, 0.76);
        let backward = mie_phase(-1.0, 0.76);
        assert!(forward > backward * 5.0, "Mie should have strong forward peak");
    }

    #[test]
    fn ray_sphere_inside() {
        let isect = ray_sphere_intersect(Vec3::ZERO, Vec3::Y, 100.0);
        assert!(isect.hit);
        assert!(isect.t0 < 0.0); // behind origin
        assert!(isect.t1 > 0.0); // ahead
    }

    #[test]
    fn ray_sphere_miss() {
        let isect = ray_sphere_intersect(Vec3::new(200.0, 0.0, 0.0), Vec3::Y, 100.0);
        assert!(!isect.hit);
    }

    #[test]
    fn sky_color_not_black_at_noon() {
        let params = AtmosphereParams::earth();
        let camera = Vec3::new(0.0, params.planet_radius + 0.001, 0.0);
        let color = compute_sky_color(camera, Vec3::new(0.0, 0.5, 0.5).normalize(), &params);
        assert!(color.x > 0.0 || color.y > 0.0 || color.z > 0.0, "Sky should not be black");
    }

    #[test]
    fn transmittance_lut_generation() {
        let params = AtmosphereParams::earth();
        let lut = TransmittanceLut::generate_with_size(&params, 16, 8);
        assert_eq!(lut.data.len(), 128);
        // At zenith, transmittance should be high.
        let t = lut.sample(0.0, 1.0, params.atmosphere_height);
        assert!(t.x > 0.5, "Transmittance at zenith should be high");
    }

    #[test]
    fn time_of_day_sun_direction() {
        let mut tod = TimeOfDay::new();
        tod.set_time(12.0);
        let dir = tod.sun_direction();
        // At noon the sun should be roughly above the horizon.
        assert!(dir.y > 0.0, "Sun should be above horizon at noon");

        tod.set_time(0.0);
        let dir_midnight = tod.sun_direction();
        // At midnight the sun should be below the horizon.
        assert!(dir_midnight.y < 0.0, "Sun should be below horizon at midnight");
    }

    #[test]
    fn star_field_generation() {
        let sf = StarField::generate(100, 12345);
        assert_eq!(sf.stars.len(), 100);
        // All directions should be normalised.
        for star in &sf.stars {
            let len = star.direction.length();
            assert!((len - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn moon_phase_cycle() {
        let mut moon = Moon::new();
        moon.phase_angle = 0.0;
        assert_eq!(moon.phase(), LunarPhase::NewMoon);
        moon.phase_angle = PI;
        assert_eq!(moon.phase(), LunarPhase::FullMoon);
    }

    #[test]
    fn limb_darkening_centre_bright() {
        let centre = limb_darkening(0.0);
        let edge = limb_darkening(0.99);
        assert!(centre > edge, "Centre should be brighter than edge");
    }

    #[test]
    fn temperature_to_rgb_sanity() {
        let warm = temperature_to_rgb(3000.0);
        let cool = temperature_to_rgb(10000.0);
        // Warm stars should be redder.
        assert!(warm.x > warm.z, "3000K should be reddish");
        // Cool stars should be bluer.
        assert!(cool.z > cool.x * 0.5, "10000K should have strong blue");
    }

    #[test]
    fn cloud_density_outside_layer() {
        let cloud = CloudLayer::new();
        // Well below cloud layer.
        let pos = Vec3::new(0.0, 6371.0 + 1.0, 0.0);
        assert_eq!(cloud.sample_density(pos), 0.0);
    }
}
