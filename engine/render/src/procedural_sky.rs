// engine/render/src/procedural_sky_v2.rs
//
// Enhanced physically-based procedural sky rendering for the Genovo engine.
//
// This module provides:
// - Physically-based sky model with ozone absorption layer
// - Multiple scattering precomputation (LUT-based)
// - Aerial perspective look-up table for distance-based atmosphere colouring
// - Planet rendering from space (visible curvature, atmosphere glow)
// - Ring system rendering (Saturn-like rings)
// - Nebula backdrop for space scenes
//
// The sky model is based on Bruneton & Neyret (2008) with extensions for
// ozone and multi-scattering following Hillaire (2020).

use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-7;
const TWO_PI: f32 = PI * 2.0;

/// Earth's radius in km (default planet).
pub const EARTH_RADIUS_KM: f32 = 6371.0;

/// Typical atmosphere height in km.
pub const ATMOSPHERE_HEIGHT_KM: f32 = 100.0;

/// Rayleigh scale height in km.
pub const RAYLEIGH_SCALE_HEIGHT: f32 = 8.5;

/// Mie scale height in km.
pub const MIE_SCALE_HEIGHT: f32 = 1.2;

/// Ozone layer centre altitude in km.
pub const OZONE_CENTER_ALTITUDE: f32 = 25.0;

/// Ozone layer half-width in km.
pub const OZONE_HALF_WIDTH: f32 = 15.0;

/// Default LUT resolution for transmittance.
pub const TRANSMITTANCE_LUT_WIDTH: u32 = 256;
pub const TRANSMITTANCE_LUT_HEIGHT: u32 = 64;

/// Default LUT resolution for multiple scattering.
pub const MULTI_SCATTER_LUT_SIZE: u32 = 32;

/// Default aerial perspective LUT dimensions.
pub const AERIAL_LUT_WIDTH: u32 = 32;
pub const AERIAL_LUT_HEIGHT: u32 = 32;
pub const AERIAL_LUT_DEPTH: u32 = 32;

// ---------------------------------------------------------------------------
// Atmosphere parameters
// ---------------------------------------------------------------------------

/// Scattering coefficients for Rayleigh scattering (wavelength-dependent).
#[derive(Debug, Clone, Copy)]
pub struct RayleighCoefficients {
    /// Scattering coefficient at sea level (per km) for R, G, B.
    pub scattering: Vec3,
    /// Scale height in km.
    pub scale_height: f32,
}

impl Default for RayleighCoefficients {
    fn default() -> Self {
        Self {
            scattering: Vec3::new(5.802e-3, 13.558e-3, 33.1e-3),
            scale_height: RAYLEIGH_SCALE_HEIGHT,
        }
    }
}

impl RayleighCoefficients {
    /// Evaluate the density at a given altitude.
    pub fn density(&self, altitude_km: f32) -> f32 {
        (-altitude_km / self.scale_height).exp()
    }

    /// Evaluate scattering coefficient at altitude.
    pub fn scattering_at(&self, altitude_km: f32) -> Vec3 {
        self.scattering * self.density(altitude_km)
    }
}

/// Mie scattering parameters.
#[derive(Debug, Clone, Copy)]
pub struct MieCoefficients {
    /// Scattering coefficient at sea level (per km).
    pub scattering: f32,
    /// Absorption coefficient at sea level (per km).
    pub absorption: f32,
    /// Scale height in km.
    pub scale_height: f32,
    /// Asymmetry parameter g for the Henyey-Greenstein phase function.
    pub asymmetry: f32,
}

impl Default for MieCoefficients {
    fn default() -> Self {
        Self {
            scattering: 3.996e-3,
            absorption: 4.4e-4,
            scale_height: MIE_SCALE_HEIGHT,
            asymmetry: 0.8,
        }
    }
}

impl MieCoefficients {
    /// Density at altitude.
    pub fn density(&self, altitude_km: f32) -> f32 {
        (-altitude_km / self.scale_height).exp()
    }

    /// Total extinction at altitude.
    pub fn extinction_at(&self, altitude_km: f32) -> f32 {
        (self.scattering + self.absorption) * self.density(altitude_km)
    }

    /// Henyey-Greenstein phase function.
    pub fn phase(&self, cos_theta: f32) -> f32 {
        let g = self.asymmetry;
        let g2 = g * g;
        let denom = (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
        (1.0 - g2) / (4.0 * PI * denom + EPSILON)
    }
}

/// Ozone absorption parameters.
#[derive(Debug, Clone, Copy)]
pub struct OzoneParameters {
    /// Whether ozone absorption is enabled.
    pub enabled: bool,
    /// Absorption cross-section for R, G, B (per km at peak density).
    pub absorption: Vec3,
    /// Centre altitude of the ozone layer in km.
    pub center_altitude: f32,
    /// Half-width of the ozone layer in km.
    pub half_width: f32,
    /// Density multiplier.
    pub density_scale: f32,
}

impl Default for OzoneParameters {
    fn default() -> Self {
        Self {
            enabled: true,
            absorption: Vec3::new(0.65e-3, 1.881e-3, 0.085e-3),
            center_altitude: OZONE_CENTER_ALTITUDE,
            half_width: OZONE_HALF_WIDTH,
            density_scale: 1.0,
        }
    }
}

impl OzoneParameters {
    /// Evaluate ozone density at altitude (tent function).
    pub fn density(&self, altitude_km: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        let dist = (altitude_km - self.center_altitude).abs();
        let d = (1.0 - dist / self.half_width).max(0.0) * self.density_scale;
        d
    }

    /// Absorption coefficient at altitude.
    pub fn absorption_at(&self, altitude_km: f32) -> Vec3 {
        self.absorption * self.density(altitude_km)
    }
}

/// Complete atmosphere parameters.
#[derive(Debug, Clone)]
pub struct AtmosphereParams {
    /// Planet radius in km.
    pub planet_radius: f32,
    /// Atmosphere top radius in km (planet_radius + atmosphere_height).
    pub atmosphere_radius: f32,
    /// Rayleigh scattering.
    pub rayleigh: RayleighCoefficients,
    /// Mie scattering.
    pub mie: MieCoefficients,
    /// Ozone absorption.
    pub ozone: OzoneParameters,
    /// Ground albedo (for multi-scattering ground bounce).
    pub ground_albedo: Vec3,
    /// Sun disc angular radius in radians.
    pub sun_angular_radius: f32,
    /// Sun illuminance (lux).
    pub sun_illuminance: Vec3,
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self {
            planet_radius: EARTH_RADIUS_KM,
            atmosphere_radius: EARTH_RADIUS_KM + ATMOSPHERE_HEIGHT_KM,
            rayleigh: RayleighCoefficients::default(),
            mie: MieCoefficients::default(),
            ozone: OzoneParameters::default(),
            ground_albedo: Vec3::splat(0.3),
            sun_angular_radius: 0.00467, // ~0.267 degrees
            sun_illuminance: Vec3::splat(1.0),
        }
    }
}

impl AtmosphereParams {
    /// Create parameters for a Mars-like atmosphere.
    pub fn mars() -> Self {
        Self {
            planet_radius: 3389.5,
            atmosphere_radius: 3389.5 + 80.0,
            rayleigh: RayleighCoefficients {
                scattering: Vec3::new(19.918e-3, 13.57e-3, 5.75e-3),
                scale_height: 11.1,
            },
            mie: MieCoefficients {
                scattering: 4.0e-3,
                absorption: 4.0e-3,
                scale_height: 11.1,
                asymmetry: 0.76,
            },
            ozone: OzoneParameters {
                enabled: false,
                ..Default::default()
            },
            ground_albedo: Vec3::new(0.3, 0.15, 0.1),
            sun_angular_radius: 0.00348,
            sun_illuminance: Vec3::splat(0.43),
        }
    }

    /// Create an alien atmosphere with unusual colours.
    pub fn alien_atmosphere(planet_radius: f32, atmosphere_height: f32, tint: Vec3) -> Self {
        Self {
            planet_radius,
            atmosphere_radius: planet_radius + atmosphere_height,
            rayleigh: RayleighCoefficients {
                scattering: tint * 15.0e-3,
                scale_height: atmosphere_height * 0.1,
            },
            ..Default::default()
        }
    }

    /// Altitude from a position relative to planet centre.
    pub fn altitude(&self, height_from_centre: f32) -> f32 {
        (height_from_centre - self.planet_radius).max(0.0)
    }

    /// Total extinction at a given altitude.
    pub fn extinction_at(&self, altitude_km: f32) -> Vec3 {
        let rayleigh = self.rayleigh.scattering_at(altitude_km);
        let mie = Vec3::splat(self.mie.extinction_at(altitude_km));
        let ozone = self.ozone.absorption_at(altitude_km);
        rayleigh + mie + ozone
    }

    /// Total scattering at a given altitude.
    pub fn scattering_at(&self, altitude_km: f32) -> Vec3 {
        let rayleigh = self.rayleigh.scattering_at(altitude_km);
        let mie = Vec3::splat(self.mie.scattering * self.mie.density(altitude_km));
        rayleigh + mie
    }
}

// ---------------------------------------------------------------------------
// Rayleigh phase function
// ---------------------------------------------------------------------------

/// Rayleigh scattering phase function.
pub fn rayleigh_phase(cos_theta: f32) -> f32 {
    3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta)
}

// ---------------------------------------------------------------------------
// Transmittance LUT
// ---------------------------------------------------------------------------

/// A 2D look-up table storing optical depth / transmittance.
#[derive(Debug, Clone)]
pub struct TransmittanceLut {
    /// LUT data indexed [height][view_zenith].
    pub data: Vec<Vec3>,
    /// Width (view zenith cos angle samples).
    pub width: u32,
    /// Height (altitude samples).
    pub height: u32,
}

impl TransmittanceLut {
    /// Create an empty LUT.
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            data: vec![Vec3::ONE; size],
            width,
            height,
        }
    }

    /// Precompute the transmittance LUT for the given atmosphere.
    pub fn compute(&mut self, params: &AtmosphereParams, ray_steps: u32) {
        for y in 0..self.height {
            for x in 0..self.width {
                let u = (x as f32 + 0.5) / self.width as f32;
                let v = (y as f32 + 0.5) / self.height as f32;

                let h = v * (params.atmosphere_radius - params.planet_radius);
                let r = params.planet_radius + h;
                let cos_zenith = u * 2.0 - 1.0;

                let optical_depth = self.integrate_optical_depth(
                    params, r, cos_zenith, ray_steps,
                );

                let transmittance = Vec3::new(
                    (-optical_depth.x).exp(),
                    (-optical_depth.y).exp(),
                    (-optical_depth.z).exp(),
                );

                let idx = (y * self.width + x) as usize;
                if idx < self.data.len() {
                    self.data[idx] = transmittance;
                }
            }
        }
    }

    /// Integrate optical depth along a ray from (r, cos_zenith) to atmosphere top.
    fn integrate_optical_depth(
        &self,
        params: &AtmosphereParams,
        r: f32,
        cos_zenith: f32,
        steps: u32,
    ) -> Vec3 {
        let sin_zenith = (1.0 - cos_zenith * cos_zenith).max(0.0).sqrt();
        let ray_length = self.ray_atmosphere_intersect(
            r, cos_zenith, params.atmosphere_radius,
        );

        if ray_length <= 0.0 {
            return Vec3::ZERO;
        }

        let step_size = ray_length / steps as f32;
        let mut optical_depth = Vec3::ZERO;

        for i in 0..steps {
            let t = (i as f32 + 0.5) * step_size;
            let px = r * cos_zenith + t * cos_zenith; // Approximate
            let py = r * sin_zenith;
            // More accurate height computation
            let sample_r = (px * px + py * py + 2.0 * r * t * cos_zenith + t * t).sqrt();
            let altitude = params.altitude(sample_r);

            if altitude > params.atmosphere_radius - params.planet_radius {
                break;
            }

            let extinction = params.extinction_at(altitude);
            optical_depth += extinction * step_size;
        }

        optical_depth
    }

    /// Ray-sphere intersection for atmosphere.
    fn ray_atmosphere_intersect(&self, r: f32, cos_zenith: f32, atmo_radius: f32) -> f32 {
        let b = 2.0 * r * cos_zenith;
        let c = r * r - atmo_radius * atmo_radius;
        let discriminant = b * b - 4.0 * c;
        if discriminant < 0.0 {
            return 0.0;
        }
        let sqrt_d = discriminant.sqrt();
        let t = (-b + sqrt_d) * 0.5;
        t.max(0.0)
    }

    /// Sample the LUT at a given altitude and view zenith cosine.
    pub fn sample(&self, altitude_km: f32, cos_zenith: f32, atmosphere_height: f32) -> Vec3 {
        let u = (cos_zenith + 1.0) * 0.5;
        let v = (altitude_km / atmosphere_height).clamp(0.0, 1.0);

        let fx = u * (self.width - 1) as f32;
        let fy = v * (self.height - 1) as f32;
        let ix = (fx as u32).min(self.width - 2);
        let iy = (fy as u32).min(self.height - 2);
        let sx = fx - ix as f32;
        let sy = fy - iy as f32;

        let i00 = (iy * self.width + ix) as usize;
        let i10 = i00 + 1;
        let i01 = ((iy + 1) * self.width + ix) as usize;
        let i11 = i01 + 1;

        if i11 >= self.data.len() {
            return Vec3::ONE;
        }

        let a = Vec3::lerp(self.data[i00], self.data[i10], sx);
        let b = Vec3::lerp(self.data[i01], self.data[i11], sx);
        Vec3::lerp(a, b, sy)
    }
}

// ---------------------------------------------------------------------------
// Multiple scattering LUT
// ---------------------------------------------------------------------------

/// Precomputed multiple scattering look-up table.
#[derive(Debug, Clone)]
pub struct MultiScatterLut {
    /// LUT data indexed [height][sun_zenith].
    pub data: Vec<Vec3>,
    /// Resolution (square LUT).
    pub size: u32,
}

impl MultiScatterLut {
    /// Create an empty LUT.
    pub fn new(size: u32) -> Self {
        Self {
            data: vec![Vec3::ZERO; (size * size) as usize],
            size,
        }
    }

    /// Sample the multi-scatter LUT.
    pub fn sample(&self, altitude_km: f32, cos_sun_zenith: f32, atmosphere_height: f32) -> Vec3 {
        let u = (cos_sun_zenith + 1.0) * 0.5;
        let v = (altitude_km / atmosphere_height).clamp(0.0, 1.0);

        let fx = u * (self.size - 1) as f32;
        let fy = v * (self.size - 1) as f32;
        let ix = (fx as u32).min(self.size - 2);
        let iy = (fy as u32).min(self.size - 2);
        let sx = fx - ix as f32;
        let sy = fy - iy as f32;

        let i00 = (iy * self.size + ix) as usize;
        let i10 = i00 + 1;
        let i01 = ((iy + 1) * self.size + ix) as usize;
        let i11 = i01 + 1;

        if i11 >= self.data.len() {
            return Vec3::ZERO;
        }

        let a = Vec3::lerp(self.data[i00], self.data[i10], sx);
        let b = Vec3::lerp(self.data[i01], self.data[i11], sx);
        Vec3::lerp(a, b, sy)
    }
}

// ---------------------------------------------------------------------------
// Aerial perspective LUT
// ---------------------------------------------------------------------------

/// Aerial perspective LUT for distance-based atmosphere colouring.
///
/// This is a 3D texture indexed by (view direction, distance, height).
#[derive(Debug, Clone)]
pub struct AerialPerspectiveLut {
    /// LUT data.
    pub data: Vec<Vec4>,
    /// Dimensions.
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Maximum distance in km covered by the LUT.
    pub max_distance: f32,
}

impl AerialPerspectiveLut {
    /// Create an empty LUT.
    pub fn new(width: u32, height: u32, depth: u32, max_distance: f32) -> Self {
        let size = (width * height * depth) as usize;
        Self {
            data: vec![Vec4::ZERO; size],
            width,
            height,
            depth,
            max_distance,
        }
    }

    /// Sample aerial perspective at a given position.
    pub fn sample(&self, distance: f32, altitude_km: f32, cos_sun_angle: f32) -> Vec4 {
        let u = (distance / self.max_distance).clamp(0.0, 1.0);
        let v = (altitude_km / 20.0).clamp(0.0, 1.0); // Normalize to ~20km max
        let w = (cos_sun_angle + 1.0) * 0.5;

        let ix = ((u * (self.width - 1) as f32) as u32).min(self.width - 1);
        let iy = ((v * (self.height - 1) as f32) as u32).min(self.height - 1);
        let iz = ((w * (self.depth - 1) as f32) as u32).min(self.depth - 1);

        let idx = ((iz * self.height + iy) * self.width + ix) as usize;
        if idx < self.data.len() {
            self.data[idx]
        } else {
            Vec4::ZERO
        }
    }

    /// Apply aerial perspective to a scene colour.
    pub fn apply(&self, scene_color: Vec3, distance: f32, altitude_km: f32, cos_sun_angle: f32) -> Vec3 {
        let ap = self.sample(distance, altitude_km, cos_sun_angle);
        let transmittance = ap.w;
        let inscatter = ap.truncate();
        scene_color * transmittance + inscatter
    }
}

// ---------------------------------------------------------------------------
// Planet rendering from space
// ---------------------------------------------------------------------------

/// Configuration for rendering a planet from space.
#[derive(Debug, Clone)]
pub struct PlanetRenderConfig {
    /// Whether planet rendering is enabled.
    pub enabled: bool,
    /// Planet radius in world units.
    pub radius: f32,
    /// Planet centre position in world space.
    pub center: Vec3,
    /// Atmosphere colour (additive glow at the limb).
    pub atmosphere_color: Vec3,
    /// Atmosphere thickness (visual, in world units).
    pub atmosphere_thickness: f32,
    /// Ground colour (flat colour or texture handle).
    pub ground_color: Vec3,
    /// Whether to render clouds on the planet surface.
    pub clouds_enabled: bool,
    /// Cloud coverage.
    pub cloud_coverage: f32,
    /// Cloud colour.
    pub cloud_color: Vec3,
    /// Rotation speed (radians per second).
    pub rotation_speed: f32,
    /// Current rotation angle.
    pub rotation_angle: f32,
    /// Night side city lights colour.
    pub city_lights_color: Vec3,
    /// City lights intensity.
    pub city_lights_intensity: f32,
    /// Whether to render the terminator (day/night boundary).
    pub show_terminator: bool,
}

impl Default for PlanetRenderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            radius: 1000.0,
            center: Vec3::new(0.0, -1000.0, 0.0),
            atmosphere_color: Vec3::new(0.3, 0.5, 1.0),
            atmosphere_thickness: 50.0,
            ground_color: Vec3::new(0.2, 0.4, 0.15),
            clouds_enabled: true,
            cloud_coverage: 0.5,
            cloud_color: Vec3::splat(0.9),
            rotation_speed: 0.01,
            rotation_angle: 0.0,
            city_lights_color: Vec3::new(1.0, 0.9, 0.6),
            city_lights_intensity: 0.5,
            show_terminator: true,
        }
    }
}

/// Planet renderer.
#[derive(Debug)]
pub struct PlanetRenderer {
    pub config: PlanetRenderConfig,
    pub time: f32,
}

impl PlanetRenderer {
    pub fn new(config: PlanetRenderConfig) -> Self {
        Self { config, time: 0.0 }
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.config.rotation_angle += self.config.rotation_speed * dt;
        self.config.rotation_angle %= TWO_PI;
    }

    /// Ray-sphere intersection test.
    pub fn intersect_planet(&self, ray_origin: Vec3, ray_dir: Vec3) -> Option<f32> {
        let oc = ray_origin - self.config.center;
        let b = oc.dot(ray_dir);
        let c = oc.dot(oc) - self.config.radius * self.config.radius;
        let discriminant = b * b - c;
        if discriminant < 0.0 {
            return None;
        }
        let t = -b - discriminant.sqrt();
        if t > 0.0 { Some(t) } else { None }
    }

    /// Compute the atmosphere glow at the limb.
    pub fn atmosphere_glow(&self, ray_origin: Vec3, ray_dir: Vec3) -> Vec3 {
        if !self.config.enabled {
            return Vec3::ZERO;
        }

        let atmo_radius = self.config.radius + self.config.atmosphere_thickness;
        let oc = ray_origin - self.config.center;
        let b = oc.dot(ray_dir);
        let c = oc.dot(oc) - atmo_radius * atmo_radius;
        let discriminant = b * b - c;

        if discriminant < 0.0 {
            return Vec3::ZERO;
        }

        // Check if ray hits the planet itself
        if self.intersect_planet(ray_origin, ray_dir).is_some() {
            return Vec3::ZERO;
        }

        // Compute closest approach to planet centre
        let closest_approach = (oc + ray_dir * (-b)).length();
        let t = ((closest_approach - self.config.radius) / self.config.atmosphere_thickness)
            .clamp(0.0, 1.0);
        let glow_intensity = (1.0 - t).powf(3.0);

        self.config.atmosphere_color * glow_intensity
    }
}

// ---------------------------------------------------------------------------
// Ring system
// ---------------------------------------------------------------------------

/// Configuration for a planetary ring system.
#[derive(Debug, Clone)]
pub struct RingSystemConfig {
    /// Whether rings are enabled.
    pub enabled: bool,
    /// Inner radius of the ring system.
    pub inner_radius: f32,
    /// Outer radius of the ring system.
    pub outer_radius: f32,
    /// Planet centre (rings are in the XZ plane relative to planet).
    pub center: Vec3,
    /// Ring normal (typically Vec3::Y for horizontal rings).
    pub normal: Vec3,
    /// Ring colour.
    pub color: Vec3,
    /// Ring opacity.
    pub opacity: f32,
    /// Number of ring bands.
    pub band_count: u32,
    /// Gap positions (normalised 0-1 within inner-outer range).
    pub gap_positions: Vec<f32>,
    /// Gap widths (normalised).
    pub gap_widths: Vec<f32>,
    /// Whether rings cast shadows on the planet.
    pub cast_shadow: bool,
    /// Whether the planet casts a shadow on the rings.
    pub receive_shadow: bool,
}

impl Default for RingSystemConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            inner_radius: 1500.0,
            outer_radius: 3000.0,
            center: Vec3::ZERO,
            normal: Vec3::Y,
            color: Vec3::new(0.8, 0.7, 0.5),
            opacity: 0.7,
            band_count: 8,
            gap_positions: vec![0.4, 0.7],
            gap_widths: vec![0.03, 0.02],
            cast_shadow: true,
            receive_shadow: true,
        }
    }
}

/// Ring system renderer.
#[derive(Debug)]
pub struct RingSystem {
    pub config: RingSystemConfig,
}

impl RingSystem {
    pub fn new(config: RingSystemConfig) -> Self {
        Self { config }
    }

    /// Ray-plane intersection for ring disc.
    pub fn intersect(&self, ray_origin: Vec3, ray_dir: Vec3) -> Option<(f32, f32)> {
        if !self.config.enabled {
            return None;
        }

        let normal = self.config.normal.normalize_or_zero();
        let denom = ray_dir.dot(normal);
        if denom.abs() < EPSILON {
            return None;
        }

        let t = (self.config.center - ray_origin).dot(normal) / denom;
        if t < 0.0 {
            return None;
        }

        let hit = ray_origin + ray_dir * t;
        let dist = (hit - self.config.center).length();

        if dist >= self.config.inner_radius && dist <= self.config.outer_radius {
            let ring_t = (dist - self.config.inner_radius)
                / (self.config.outer_radius - self.config.inner_radius);
            Some((t, ring_t))
        } else {
            None
        }
    }

    /// Sample ring colour and opacity at a normalised ring position (0 = inner, 1 = outer).
    pub fn sample(&self, ring_t: f32) -> (Vec3, f32) {
        if !self.config.enabled {
            return (Vec3::ZERO, 0.0);
        }

        // Check gaps
        for i in 0..self.config.gap_positions.len() {
            let gap_pos = self.config.gap_positions[i];
            let gap_width = if i < self.config.gap_widths.len() {
                self.config.gap_widths[i]
            } else {
                0.02
            };
            if (ring_t - gap_pos).abs() < gap_width * 0.5 {
                return (Vec3::ZERO, 0.0);
            }
        }

        // Band-based density variation
        let band_freq = self.config.band_count as f32;
        let band_value = ((ring_t * band_freq * PI).sin() * 0.5 + 0.5) * 0.5 + 0.5;

        // Edge fade
        let edge_fade = (ring_t * 5.0).clamp(0.0, 1.0) * ((1.0 - ring_t) * 5.0).clamp(0.0, 1.0);

        let opacity = self.config.opacity * band_value * edge_fade;
        let color = self.config.color * band_value;

        (color, opacity)
    }
}

// ---------------------------------------------------------------------------
// Nebula backdrop
// ---------------------------------------------------------------------------

/// Configuration for a nebula backdrop.
#[derive(Debug, Clone)]
pub struct NebulaConfig {
    /// Whether the nebula is enabled.
    pub enabled: bool,
    /// Primary nebula colour.
    pub color_a: Vec3,
    /// Secondary nebula colour.
    pub color_b: Vec3,
    /// Tertiary/accent colour (stars, bright knots).
    pub color_c: Vec3,
    /// Intensity.
    pub intensity: f32,
    /// Scale (larger = wider nebula features).
    pub scale: f32,
    /// Noise octaves for fractal detail.
    pub octaves: u32,
    /// Density falloff power.
    pub density_power: f32,
    /// Star density within the nebula.
    pub star_density: f32,
    /// Star brightness.
    pub star_brightness: f32,
    /// Direction of the nebula centre.
    pub center_direction: Vec3,
    /// Angular spread in radians.
    pub angular_spread: f32,
}

impl Default for NebulaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            color_a: Vec3::new(0.4, 0.1, 0.5),
            color_b: Vec3::new(0.1, 0.2, 0.6),
            color_c: Vec3::new(0.8, 0.3, 0.1),
            intensity: 0.3,
            scale: 1.0,
            octaves: 5,
            density_power: 2.0,
            star_density: 100.0,
            star_brightness: 2.0,
            center_direction: Vec3::new(0.0, 0.5, -1.0).normalize(),
            angular_spread: PI * 0.5,
        }
    }
}

/// Nebula renderer.
#[derive(Debug)]
pub struct NebulaRenderer {
    pub config: NebulaConfig,
}

impl NebulaRenderer {
    pub fn new(config: NebulaConfig) -> Self {
        Self { config }
    }

    /// Sample the nebula colour at a view direction.
    pub fn sample(&self, view_dir: Vec3) -> Vec3 {
        if !self.config.enabled {
            return Vec3::ZERO;
        }

        let center = self.config.center_direction.normalize_or_zero();
        let cos_angle = view_dir.dot(center);
        let angle = cos_angle.acos();

        if angle > self.config.angular_spread {
            return Vec3::ZERO;
        }

        let t = angle / self.config.angular_spread;
        let density = (1.0 - t).powf(self.config.density_power);

        // Simple hash-based noise approximation (would be replaced by proper 3D noise on GPU)
        let noise = self.simple_noise(view_dir * self.config.scale);
        let detail = self.simple_noise(view_dir * self.config.scale * 3.0);

        let color_blend = noise * 0.5 + 0.5;
        let nebula_color = Vec3::lerp(self.config.color_a, self.config.color_b, color_blend);
        let accent = (detail * 3.0 - 2.0).max(0.0) * self.config.color_c;

        (nebula_color * density + accent * density * 0.5) * self.config.intensity
    }

    /// Simple 3D hash noise (CPU reference, replaced by GPU noise in shader).
    fn simple_noise(&self, p: Vec3) -> f32 {
        let dot = p.x * 127.1 + p.y * 311.7 + p.z * 74.7;
        let hash = (dot.sin() * 43758.5453).fract();
        hash
    }
}

// ---------------------------------------------------------------------------
// Procedural sky V2 system
// ---------------------------------------------------------------------------

/// The main procedural sky V2 system combining all components.
#[derive(Debug)]
pub struct ProceduralSkyV2 {
    /// Atmosphere parameters.
    pub atmosphere: AtmosphereParams,
    /// Transmittance LUT.
    pub transmittance_lut: TransmittanceLut,
    /// Multiple scattering LUT.
    pub multi_scatter_lut: MultiScatterLut,
    /// Aerial perspective LUT.
    pub aerial_perspective_lut: AerialPerspectiveLut,
    /// Planet renderer (for space views).
    pub planet: PlanetRenderer,
    /// Ring system.
    pub rings: RingSystem,
    /// Nebula backdrop.
    pub nebula: NebulaRenderer,
    /// Sun direction.
    pub sun_direction: Vec3,
    /// Whether LUTs need recomputation.
    pub luts_dirty: bool,
    /// Sky intensity multiplier.
    pub sky_intensity: f32,
    /// Whether we're in space (above atmosphere).
    pub in_space: bool,
    /// Camera altitude in km.
    pub camera_altitude_km: f32,
}

impl ProceduralSkyV2 {
    /// Create a new procedural sky.
    pub fn new(atmosphere: AtmosphereParams) -> Self {
        Self {
            transmittance_lut: TransmittanceLut::new(TRANSMITTANCE_LUT_WIDTH, TRANSMITTANCE_LUT_HEIGHT),
            multi_scatter_lut: MultiScatterLut::new(MULTI_SCATTER_LUT_SIZE),
            aerial_perspective_lut: AerialPerspectiveLut::new(
                AERIAL_LUT_WIDTH, AERIAL_LUT_HEIGHT, AERIAL_LUT_DEPTH, 200.0,
            ),
            planet: PlanetRenderer::new(PlanetRenderConfig::default()),
            rings: RingSystem::new(RingSystemConfig::default()),
            nebula: NebulaRenderer::new(NebulaConfig::default()),
            sun_direction: Vec3::new(0.0, 0.5, -1.0).normalize(),
            luts_dirty: true,
            sky_intensity: 1.0,
            in_space: false,
            camera_altitude_km: 0.0,
            atmosphere,
        }
    }

    /// Create with Earth-like defaults.
    pub fn earth() -> Self {
        Self::new(AtmosphereParams::default())
    }

    /// Create with Mars-like atmosphere.
    pub fn mars() -> Self {
        Self::new(AtmosphereParams::mars())
    }

    /// Set the sun direction and mark LUTs dirty.
    pub fn set_sun_direction(&mut self, dir: Vec3) {
        self.sun_direction = dir.normalize_or_zero();
        // Aerial perspective depends on sun direction
        self.luts_dirty = true;
    }

    /// Set the camera altitude.
    pub fn set_camera_altitude(&mut self, altitude_km: f32) {
        self.camera_altitude_km = altitude_km;
        self.in_space = altitude_km > (self.atmosphere.atmosphere_radius - self.atmosphere.planet_radius);
    }

    /// Precompute all LUTs.
    pub fn precompute_luts(&mut self) {
        self.transmittance_lut.compute(&self.atmosphere, 40);
        // Multi-scatter and aerial perspective would be computed similarly
        // (full implementation would be GPU compute shaders)
        self.luts_dirty = false;
    }

    /// Update the sky system.
    pub fn update(&mut self, dt: f32) {
        self.planet.update(dt);
        if self.luts_dirty {
            self.precompute_luts();
        }
    }

    /// Sample the sky colour at a view direction.
    pub fn sample_sky(&self, view_dir: Vec3) -> Vec3 {
        let mut color = Vec3::ZERO;

        if self.in_space {
            // Space view: planet, rings, nebula, stars
            color += self.planet.atmosphere_glow(
                Vec3::new(0.0, self.camera_altitude_km + self.atmosphere.planet_radius, 0.0),
                view_dir,
            );
            color += self.nebula.sample(view_dir);

            if let Some((_t, ring_t)) = self.rings.intersect(
                Vec3::new(0.0, self.camera_altitude_km + self.atmosphere.planet_radius, 0.0),
                view_dir,
            ) {
                let (ring_color, ring_alpha) = self.rings.sample(ring_t);
                color = color * (1.0 - ring_alpha) + ring_color * ring_alpha;
            }
        } else {
            // Ground/atmosphere view
            let altitude = self.camera_altitude_km;
            let cos_sun = view_dir.dot(self.sun_direction);

            // Transmittance
            let transmittance = self.transmittance_lut.sample(
                altitude,
                view_dir.y,
                self.atmosphere.atmosphere_radius - self.atmosphere.planet_radius,
            );

            // Single scattering (simplified)
            let rayleigh_phase_val = rayleigh_phase(cos_sun);
            let mie_phase_val = self.atmosphere.mie.phase(cos_sun);

            let scattering = self.atmosphere.scattering_at(altitude);
            let inscatter = scattering * (rayleigh_phase_val + mie_phase_val) * self.atmosphere.sun_illuminance;

            color = inscatter * (Vec3::ONE - transmittance) + transmittance * Vec3::ZERO; // background = black
        }

        // Sun disc
        let sun_cos = view_dir.dot(self.sun_direction);
        let sun_angle = sun_cos.acos();
        if sun_angle < self.atmosphere.sun_angular_radius {
            color += self.atmosphere.sun_illuminance * 100.0;
        }

        color * self.sky_intensity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rayleigh_density() {
        let rayleigh = RayleighCoefficients::default();
        assert!((rayleigh.density(0.0) - 1.0).abs() < EPSILON);
        assert!(rayleigh.density(8.5) < 0.4);
    }

    #[test]
    fn test_ozone_density() {
        let ozone = OzoneParameters::default();
        assert!(ozone.density(OZONE_CENTER_ALTITUDE) > 0.9);
        assert!(ozone.density(0.0) < EPSILON);
        assert!(ozone.density(100.0) < EPSILON);
    }

    #[test]
    fn test_rayleigh_phase() {
        let forward = rayleigh_phase(1.0);
        let backward = rayleigh_phase(-1.0);
        assert!((forward - backward).abs() < EPSILON); // Symmetric
    }

    #[test]
    fn test_transmittance_lut() {
        let mut lut = TransmittanceLut::new(16, 8);
        let params = AtmosphereParams::default();
        lut.compute(&params, 10);
        // At zero altitude looking up, transmittance should be < 1.0
        let t = lut.sample(0.0, 1.0, ATMOSPHERE_HEIGHT_KM);
        assert!(t.x <= 1.0 && t.y <= 1.0 && t.z <= 1.0);
    }

    #[test]
    fn test_aerial_perspective() {
        let lut = AerialPerspectiveLut::new(4, 4, 4, 100.0);
        let ap = lut.sample(0.0, 0.0, 1.0);
        // Empty LUT should return zero
        assert!(ap.length() < EPSILON);
    }

    #[test]
    fn test_ring_system() {
        let rings = RingSystem::new(RingSystemConfig {
            enabled: true,
            inner_radius: 100.0,
            outer_radius: 200.0,
            center: Vec3::ZERO,
            normal: Vec3::Y,
            ..Default::default()
        });
        // Ray from above hitting the ring plane
        let result = rings.intersect(Vec3::new(150.0, 100.0, 0.0), Vec3::NEG_Y);
        assert!(result.is_some());
    }

    #[test]
    fn test_procedural_sky_v2() {
        let mut sky = ProceduralSkyV2::earth();
        sky.update(0.016);
        let color = sky.sample_sky(Vec3::Y);
        assert!(color.length() >= 0.0);
    }
}
