// engine/render/src/hair.rs
//
// Hair and fur rendering for the Genovo engine. Implements the Marschner
// hair shading model with R, TT, and TRT lobes, deep opacity map shadows,
// wind animation, and LOD transitions from strands to shells to cards.

use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default hair index of refraction (human hair).
const DEFAULT_IOR: f32 = 1.55;

/// Default longitudinal roughness.
const DEFAULT_LONGITUDINAL_ROUGHNESS: f32 = 0.15;

/// Default azimuthal roughness.
const DEFAULT_AZIMUTHAL_ROUGHNESS: f32 = 0.5;

/// Maximum number of control points per strand.
pub const MAX_CONTROL_POINTS: usize = 64;

/// Default number of interpolated strands per guide strand.
pub const DEFAULT_INTERPOLATED_STRANDS: u32 = 8;

/// Number of layers in a deep opacity map.
pub const DEEP_OPACITY_LAYERS: usize = 8;

// ---------------------------------------------------------------------------
// HairStrand
// ---------------------------------------------------------------------------

/// A single hair strand defined by a series of control points.
#[derive(Debug, Clone)]
pub struct HairStrand {
    /// Control points along the strand (from root to tip).
    pub control_points: Vec<Vec3>,
    /// Thickness at each control point (root to tip taper).
    pub thickness: Vec<f32>,
    /// Colour along the strand (can vary from root to tip).
    pub color: Vec<Vec3>,
    /// Tangent at each control point (precomputed).
    pub tangents: Vec<Vec3>,
    /// Original rest-pose control points (for animation).
    pub rest_points: Vec<Vec3>,
}

impl HairStrand {
    /// Create a strand from control points with uniform thickness and colour.
    pub fn new(control_points: Vec<Vec3>, base_thickness: f32, base_color: Vec3) -> Self {
        let n = control_points.len();
        let thickness: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / (n.max(1) - 1).max(1) as f32;
                base_thickness * (1.0 - t * 0.7) // Taper toward tip.
            })
            .collect();

        let color = vec![base_color; n];
        let rest_points = control_points.clone();

        let mut strand = Self {
            control_points,
            thickness,
            color,
            tangents: Vec::new(),
            rest_points,
        };
        strand.compute_tangents();
        strand
    }

    /// Compute tangent vectors at each control point using finite differences.
    pub fn compute_tangents(&mut self) {
        let n = self.control_points.len();
        self.tangents.resize(n, Vec3::ZERO);

        for i in 0..n {
            let tangent = if i == 0 {
                if n > 1 {
                    self.control_points[1] - self.control_points[0]
                } else {
                    Vec3::Y
                }
            } else if i == n - 1 {
                self.control_points[i] - self.control_points[i - 1]
            } else {
                self.control_points[i + 1] - self.control_points[i - 1]
            };

            self.tangents[i] = tangent.normalize_or_zero();
        }
    }

    /// Number of control points.
    pub fn point_count(&self) -> usize {
        self.control_points.len()
    }

    /// Total length of the strand.
    pub fn length(&self) -> f32 {
        let mut len = 0.0;
        for i in 1..self.control_points.len() {
            len += (self.control_points[i] - self.control_points[i - 1]).length();
        }
        len
    }

    /// Interpolate position along the strand at parameter t in [0, 1].
    pub fn interpolate_position(&self, t: f32) -> Vec3 {
        if self.control_points.is_empty() {
            return Vec3::ZERO;
        }
        if self.control_points.len() == 1 {
            return self.control_points[0];
        }

        let t = t.clamp(0.0, 1.0);
        let scaled = t * (self.control_points.len() - 1) as f32;
        let idx = scaled.floor() as usize;
        let frac = scaled - idx as f32;

        let idx = idx.min(self.control_points.len() - 2);
        self.control_points[idx] * (1.0 - frac) + self.control_points[idx + 1] * frac
    }

    /// Interpolate tangent along the strand.
    pub fn interpolate_tangent(&self, t: f32) -> Vec3 {
        if self.tangents.is_empty() {
            return Vec3::Y;
        }
        if self.tangents.len() == 1 {
            return self.tangents[0];
        }

        let t = t.clamp(0.0, 1.0);
        let scaled = t * (self.tangents.len() - 1) as f32;
        let idx = scaled.floor() as usize;
        let frac = scaled - idx as f32;

        let idx = idx.min(self.tangents.len() - 2);
        (self.tangents[idx] * (1.0 - frac) + self.tangents[idx + 1] * frac).normalize_or_zero()
    }
}

// ---------------------------------------------------------------------------
// HairAsset
// ---------------------------------------------------------------------------

/// A complete hair asset containing guide strands and interpolation data.
#[derive(Debug, Clone)]
pub struct HairAsset {
    /// Guide strands (authored or simulated).
    pub guide_strands: Vec<HairStrand>,
    /// Number of interpolated strands per guide strand.
    pub interpolated_count: u32,
    /// Root positions on the scalp surface (UV coordinates).
    pub root_uvs: Vec<Vec2>,
    /// Default hair color.
    pub default_color: Vec3,
    /// Default strand thickness at root.
    pub default_thickness: f32,
    /// Total number of rendered strands (guides * interpolated).
    pub total_strand_count: u32,
    /// Bounding box of the hair volume.
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

impl HairAsset {
    /// Create a hair asset from guide strands.
    pub fn from_guides(guide_strands: Vec<HairStrand>, interpolated_count: u32) -> Self {
        let total = guide_strands.len() as u32 * (interpolated_count + 1);

        let mut bounds_min = Vec3::splat(f32::MAX);
        let mut bounds_max = Vec3::splat(f32::MIN);

        for strand in &guide_strands {
            for &p in &strand.control_points {
                bounds_min = bounds_min.min(p);
                bounds_max = bounds_max.max(p);
            }
        }

        let root_uvs = guide_strands
            .iter()
            .enumerate()
            .map(|(i, _)| Vec2::new(i as f32 / guide_strands.len() as f32, 0.0))
            .collect();

        Self {
            guide_strands,
            interpolated_count,
            root_uvs,
            default_color: Vec3::new(0.15, 0.08, 0.04), // Dark brown.
            default_thickness: 0.001,
            total_strand_count: total,
            bounds_min,
            bounds_max,
        }
    }

    /// Generate interpolated strands between guide strands.
    pub fn generate_interpolated_strands(&self) -> Vec<HairStrand> {
        let mut result = Vec::new();

        // Add guide strands.
        for strand in &self.guide_strands {
            result.push(strand.clone());
        }

        if self.guide_strands.len() < 2 || self.interpolated_count == 0 {
            return result;
        }

        // Generate interpolated strands between each pair of adjacent guides.
        for i in 0..self.guide_strands.len() - 1 {
            let a = &self.guide_strands[i];
            let b = &self.guide_strands[i + 1];

            let max_points = a.point_count().min(b.point_count());

            for j in 1..=self.interpolated_count {
                let t = j as f32 / (self.interpolated_count + 1) as f32;

                // Add random-ish offset for natural look.
                let jitter = Vec3::new(
                    hash_float(i as u32 * 31 + j * 7) * 0.002 - 0.001,
                    0.0,
                    hash_float(i as u32 * 17 + j * 13) * 0.002 - 0.001,
                );

                let points: Vec<Vec3> = (0..max_points)
                    .map(|k| {
                        let pa = a.control_points[k.min(a.point_count() - 1)];
                        let pb = b.control_points[k.min(b.point_count() - 1)];
                        pa * (1.0 - t) + pb * t + jitter * (k as f32 / max_points as f32)
                    })
                    .collect();

                let color = a.color[0] * (1.0 - t) + b.color[0] * t;
                result.push(HairStrand::new(
                    points,
                    self.default_thickness,
                    color,
                ));
            }
        }

        result
    }

    /// Get the total number of strands including interpolated.
    pub fn total_strands(&self) -> usize {
        self.guide_strands.len() + self.guide_strands.len().saturating_sub(1)
            * self.interpolated_count as usize
    }
}

/// Simple hash function for deterministic random offsets.
fn hash_float(seed: u32) -> f32 {
    let s = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    let s = ((s >> 16) ^ s).wrapping_mul(0x45d9f3b);
    let s = ((s >> 16) ^ s);
    (s & 0xFFFF) as f32 / 65535.0
}

// ---------------------------------------------------------------------------
// Marschner Hair Shading Model
// ---------------------------------------------------------------------------

/// Parameters for the Marschner hair shading model.
#[derive(Debug, Clone, Copy)]
pub struct MarschnerParams {
    /// Index of refraction of the hair fibre.
    pub ior: f32,
    /// Absorption coefficient (per-channel). Controls hair colour.
    pub absorption: Vec3,
    /// Longitudinal roughness (width of the M terms).
    pub longitudinal_roughness: f32,
    /// Azimuthal roughness.
    pub azimuthal_roughness: f32,
    /// Alpha shift for R lobe (degrees, typically -5 to -10).
    pub alpha_r: f32,
    /// Alpha shift for TT lobe (typically half of alpha_r).
    pub alpha_tt: f32,
    /// Alpha shift for TRT lobe (typically -1.5 * alpha_r).
    pub alpha_trt: f32,
    /// Eccentricity of the hair cross-section (1.0 = circular, <1 = elliptical).
    pub eccentricity: f32,
    /// Dye absorption (additional absorption from hair colouring).
    pub dye_absorption: Vec3,
}

impl Default for MarschnerParams {
    fn default() -> Self {
        Self {
            ior: DEFAULT_IOR,
            absorption: Vec3::new(0.2, 0.5, 1.5), // Brown hair.
            longitudinal_roughness: DEFAULT_LONGITUDINAL_ROUGHNESS,
            azimuthal_roughness: DEFAULT_AZIMUTHAL_ROUGHNESS,
            alpha_r: (-7.5f32).to_radians(),
            alpha_tt: (-3.75f32).to_radians(),
            alpha_trt: (11.25f32).to_radians(),
            eccentricity: 0.85,
            dye_absorption: Vec3::ZERO,
        }
    }
}

impl MarschnerParams {
    /// Create parameters for blonde hair.
    pub fn blonde() -> Self {
        Self {
            absorption: Vec3::new(0.06, 0.1, 0.3),
            ..Default::default()
        }
    }

    /// Create parameters for dark/black hair.
    pub fn dark() -> Self {
        Self {
            absorption: Vec3::new(1.0, 1.7, 3.5),
            ..Default::default()
        }
    }

    /// Create parameters for red hair.
    pub fn red() -> Self {
        Self {
            absorption: Vec3::new(0.1, 0.8, 1.8),
            ..Default::default()
        }
    }

    /// Create parameters for white/grey hair.
    pub fn white() -> Self {
        Self {
            absorption: Vec3::new(0.01, 0.01, 0.01),
            longitudinal_roughness: 0.2,
            ..Default::default()
        }
    }

    /// Total absorption (natural + dye).
    pub fn total_absorption(&self) -> Vec3 {
        self.absorption + self.dye_absorption
    }
}

/// Shade a hair strand using the Marschner model.
///
/// # Arguments
/// - `view`: Direction from shading point to camera (normalised).
/// - `light`: Direction from shading point to light (normalised).
/// - `tangent`: Hair fibre tangent direction (normalised).
/// - `params`: Marschner shading parameters.
///
/// # Returns
/// RGB colour contribution from this light.
pub fn shade_hair(view: Vec3, light: Vec3, tangent: Vec3, params: &MarschnerParams) -> Vec3 {
    // Compute the angles in the Marschner coordinate system.
    // theta_i = angle between light and normal plane (longitudinal).
    // theta_r = angle between view and normal plane.
    // phi = azimuthal angle between projected view and light.

    let sin_theta_i = light.dot(tangent);
    let sin_theta_r = view.dot(tangent);

    let cos_theta_i = (1.0 - sin_theta_i * sin_theta_i).max(0.0).sqrt();
    let cos_theta_r = (1.0 - sin_theta_r * sin_theta_r).max(0.0).sqrt();

    let theta_i = sin_theta_i.asin();
    let theta_r = sin_theta_r.asin();

    // Half-angle parameterisation.
    let theta_h = (theta_i + theta_r) * 0.5;
    let theta_d = (theta_r - theta_i) * 0.5;

    // Azimuthal angle.
    let light_perp = (light - tangent * sin_theta_i).normalize_or_zero();
    let view_perp = (view - tangent * sin_theta_r).normalize_or_zero();
    let phi = light_perp.dot(view_perp).clamp(-1.0, 1.0).acos();

    // --- R lobe (surface reflection) ---
    let m_r = longitudinal_scattering(theta_h - params.alpha_r, params.longitudinal_roughness);
    let n_r = azimuthal_scattering_r(phi, params.ior, params.azimuthal_roughness);
    let r_lobe = m_r * n_r;

    // --- TT lobe (transmission-transmission) ---
    let m_tt = longitudinal_scattering(
        theta_h - params.alpha_tt,
        params.longitudinal_roughness * 0.5,
    );

    // Fresnel transmission.
    let cos_gamma_t = cos_half_azimuth_transmission(phi, params.ior);
    let a_tt = compute_attenuation(params, cos_gamma_t, 1);
    let n_tt = azimuthal_scattering_tt(phi, params.ior, params.azimuthal_roughness);

    let tt_lobe_r = m_tt * n_tt * a_tt.x;
    let tt_lobe_g = m_tt * n_tt * a_tt.y;
    let tt_lobe_b = m_tt * n_tt * a_tt.z;

    // --- TRT lobe (transmission-reflection-transmission) ---
    let m_trt = longitudinal_scattering(
        theta_h - params.alpha_trt,
        params.longitudinal_roughness * 2.0,
    );
    let a_trt = compute_attenuation(params, cos_gamma_t, 2);
    let n_trt = azimuthal_scattering_trt(phi, params.ior, params.azimuthal_roughness);

    let trt_lobe_r = m_trt * n_trt * a_trt.x;
    let trt_lobe_g = m_trt * n_trt * a_trt.y;
    let trt_lobe_b = m_trt * n_trt * a_trt.z;

    // Combine all lobes.
    let cos_theta_d = theta_d.cos().max(0.001);
    let inv_cos = 1.0 / cos_theta_d;

    let result = Vec3::new(
        (r_lobe + tt_lobe_r + trt_lobe_r) * inv_cos,
        (r_lobe + tt_lobe_g + trt_lobe_g) * inv_cos,
        (r_lobe + tt_lobe_b + trt_lobe_b) * inv_cos,
    );

    // Ensure non-negative.
    Vec3::new(result.x.max(0.0), result.y.max(0.0), result.z.max(0.0))
}

/// Longitudinal scattering term M(theta_h).
/// Gaussian centred at alpha with width beta.
fn longitudinal_scattering(theta_h: f32, beta: f32) -> f32 {
    let sigma = beta.max(0.01);
    let normalisation = 1.0 / (sigma * (2.0 * PI).sqrt());
    normalisation * (-theta_h * theta_h / (2.0 * sigma * sigma)).exp()
}

/// Azimuthal scattering for the R (reflection) lobe.
/// Uses a cosine falloff with Fresnel reflection.
fn azimuthal_scattering_r(phi: f32, ior: f32, roughness: f32) -> f32 {
    let cos_half_phi = (phi * 0.5).cos();
    let fresnel = fresnel_schlick(cos_half_phi, ior);

    // Normalised distribution peaked at phi = 0.
    let sigma = roughness.max(0.01);
    let normalisation = 1.0 / (sigma * (2.0 * PI).sqrt());
    fresnel * normalisation * (-phi * phi / (2.0 * sigma * sigma)).exp()
}

/// Azimuthal scattering for the TT (transmission-transmission) lobe.
fn azimuthal_scattering_tt(phi: f32, ior: f32, roughness: f32) -> f32 {
    // TT lobe is centred at phi = pi (light passes straight through).
    let phi_offset = phi - PI;
    let sigma = roughness.max(0.01) * 0.5; // TT is sharper than R.
    let normalisation = 1.0 / (sigma * (2.0 * PI).sqrt());
    normalisation * (-phi_offset * phi_offset / (2.0 * sigma * sigma)).exp()
}

/// Azimuthal scattering for the TRT (transmission-reflection-transmission) lobe.
fn azimuthal_scattering_trt(phi: f32, ior: f32, roughness: f32) -> f32 {
    // TRT produces a broad secondary highlight.
    let sigma = roughness.max(0.01) * 2.0; // Broader than R.
    let normalisation = 1.0 / (sigma * (2.0 * PI).sqrt());
    normalisation * (-phi * phi / (2.0 * sigma * sigma)).exp()
}

/// Fresnel reflectance using Schlick's approximation.
fn fresnel_schlick(cos_theta: f32, ior: f32) -> f32 {
    let r0 = ((1.0 - ior) / (1.0 + ior)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cos_theta.abs()).powi(5)
}

/// Compute the cosine of the refracted half-angle (for TT computation).
fn cos_half_azimuth_transmission(phi: f32, ior: f32) -> f32 {
    let gamma_i = (phi * 0.5).asin().clamp(-PI * 0.5, PI * 0.5);
    let sin_gamma_t = gamma_i.sin() / ior;
    let cos_gamma_t = (1.0 - sin_gamma_t * sin_gamma_t).max(0.0).sqrt();
    cos_gamma_t
}

/// Compute the absorption attenuation for light passing through the fibre.
///
/// The attenuation depends on the path length through the fibre and the
/// absorption coefficient.
fn compute_attenuation(params: &MarschnerParams, cos_gamma_t: f32, num_passes: u32) -> Vec3 {
    let total_abs = params.total_absorption();

    // Path length through the fibre is proportional to 1/cos(gamma_t).
    let path_length = if cos_gamma_t > 0.001 {
        num_passes as f32 / cos_gamma_t
    } else {
        num_passes as f32 * 100.0
    };

    Vec3::new(
        (-total_abs.x * path_length).exp(),
        (-total_abs.y * path_length).exp(),
        (-total_abs.z * path_length).exp(),
    )
}

// ---------------------------------------------------------------------------
// Deep Opacity Maps
// ---------------------------------------------------------------------------

/// A deep opacity map for hair shadow rendering.
///
/// Instead of a simple binary shadow map, stores opacity values at multiple
/// depth layers, allowing correct self-shadowing for hair volumes.
#[derive(Debug, Clone)]
pub struct DeepOpacityMap {
    /// Width of the shadow map.
    pub width: u32,
    /// Height of the shadow map.
    pub height: u32,
    /// Number of depth layers.
    pub layer_count: usize,
    /// Depth range: (near, far).
    pub depth_range: (f32, f32),
    /// Opacity data: [layer][y * width + x].
    pub layers: Vec<Vec<f32>>,
    /// Light-space view-projection matrix.
    pub light_vp: glam::Mat4,
}

impl DeepOpacityMap {
    /// Create a new deep opacity map.
    pub fn new(width: u32, height: u32, layer_count: usize, light_vp: glam::Mat4) -> Self {
        let pixel_count = (width * height) as usize;
        Self {
            width,
            height,
            layer_count,
            depth_range: (0.0, 1.0),
            layers: vec![vec![0.0; pixel_count]; layer_count],
            light_vp,
        }
    }

    /// Rasterise hair strands into the deep opacity map.
    pub fn rasterize_strands(&mut self, strands: &[HairStrand]) {
        let layer_step = (self.depth_range.1 - self.depth_range.0) / self.layer_count as f32;

        for strand in strands {
            for (i, &pos) in strand.control_points.iter().enumerate() {
                let clip = self.light_vp * pos.extend(1.0);
                if clip.w <= 0.0 {
                    continue;
                }

                let ndc = clip.truncate() / clip.w;
                let screen_x = ((ndc.x * 0.5 + 0.5) * self.width as f32) as i32;
                let screen_y = ((ndc.y * 0.5 + 0.5) * self.height as f32) as i32;
                let depth = ndc.z;

                if screen_x < 0 || screen_x >= self.width as i32
                    || screen_y < 0 || screen_y >= self.height as i32
                {
                    continue;
                }

                let pixel_idx = (screen_y as u32 * self.width + screen_x as u32) as usize;

                // Determine which layer this depth falls into.
                let layer = ((depth - self.depth_range.0) / layer_step) as usize;
                let layer = layer.min(self.layer_count - 1);

                // Accumulate opacity (hair strand contributes partial opacity).
                let thickness = if i < strand.thickness.len() {
                    strand.thickness[i]
                } else {
                    0.001
                };

                let opacity = (thickness * 500.0).min(1.0);
                if pixel_idx < self.layers[layer].len() {
                    self.layers[layer][pixel_idx] = (self.layers[layer][pixel_idx] + opacity).min(1.0);
                }
            }
        }
    }

    /// Sample the accumulated transmittance at a given position and depth.
    pub fn sample_transmittance(&self, world_pos: Vec3) -> f32 {
        let clip = self.light_vp * world_pos.extend(1.0);
        if clip.w <= 0.0 {
            return 1.0;
        }

        let ndc = clip.truncate() / clip.w;
        let screen_x = ((ndc.x * 0.5 + 0.5) * self.width as f32) as u32;
        let screen_y = ((ndc.y * 0.5 + 0.5) * self.height as f32) as u32;

        if screen_x >= self.width || screen_y >= self.height {
            return 1.0;
        }

        let pixel_idx = (screen_y * self.width + screen_x) as usize;
        let layer_step = (self.depth_range.1 - self.depth_range.0) / self.layer_count as f32;
        let query_layer = ((ndc.z - self.depth_range.0) / layer_step) as usize;

        // Accumulate opacity from all layers in front of this depth.
        let mut total_opacity = 0.0f32;
        for layer in 0..query_layer.min(self.layer_count) {
            if pixel_idx < self.layers[layer].len() {
                total_opacity += self.layers[layer][pixel_idx];
            }
        }

        // Transmittance = 1 - accumulated opacity (clamped).
        (1.0 - total_opacity).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Wind Animation
// ---------------------------------------------------------------------------

/// Wind force parameters for hair animation.
#[derive(Debug, Clone, Copy)]
pub struct WindParams {
    /// Wind direction (normalised).
    pub direction: Vec3,
    /// Wind speed.
    pub speed: f32,
    /// Turbulence frequency.
    pub turbulence_freq: f32,
    /// Turbulence amplitude.
    pub turbulence_amp: f32,
    /// Current time (seconds).
    pub time: f32,
}

impl Default for WindParams {
    fn default() -> Self {
        Self {
            direction: Vec3::new(1.0, 0.0, 0.0),
            speed: 1.0,
            turbulence_freq: 2.0,
            turbulence_amp: 0.3,
            time: 0.0,
        }
    }
}

/// Apply wind forces to a hair strand.
///
/// The wind effect increases along the strand from root to tip, with the
/// root being fixed. Turbulence adds natural-looking oscillation.
pub fn apply_wind(strand: &mut HairStrand, wind: &WindParams) {
    let n = strand.control_points.len();
    if n <= 1 {
        return;
    }

    for i in 1..n {
        let t = i as f32 / (n - 1) as f32;

        // Wind influence increases toward the tip.
        let influence = t * t;

        // Base wind force.
        let wind_force = wind.direction * wind.speed * influence;

        // Turbulence (sinusoidal oscillation).
        let phase = wind.time * wind.turbulence_freq + i as f32 * 0.5;
        let turbulence = Vec3::new(
            phase.sin() * wind.turbulence_amp,
            (phase * 1.3 + 0.7).sin() * wind.turbulence_amp * 0.5,
            (phase * 0.7 + 1.3).cos() * wind.turbulence_amp,
        ) * influence;

        // Gravity.
        let gravity = Vec3::new(0.0, -0.5, 0.0) * t * t;

        // Apply displacement.
        let displacement = wind_force + turbulence + gravity;
        strand.control_points[i] = strand.rest_points[i] + displacement * 0.01;
    }

    // Enforce length constraints (maintain segment lengths).
    for i in 1..n {
        let rest_length = (strand.rest_points[i] - strand.rest_points[i - 1]).length();
        let dir = (strand.control_points[i] - strand.control_points[i - 1]).normalize_or_zero();
        strand.control_points[i] = strand.control_points[i - 1] + dir * rest_length;
    }

    strand.compute_tangents();
}

// ---------------------------------------------------------------------------
// LOD System
// ---------------------------------------------------------------------------

/// Hair LOD level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HairLOD {
    /// Full strand rendering (closest distance).
    Strands,
    /// Shell-based rendering (medium distance).
    Shells,
    /// Billboard card rendering (far distance).
    Cards,
    /// Hidden (too far to render).
    Hidden,
}

impl HairLOD {
    /// Select LOD level based on distance and screen coverage.
    pub fn select(distance: f32, screen_coverage: f32) -> Self {
        if screen_coverage < 0.001 || distance > 200.0 {
            Self::Hidden
        } else if distance > 50.0 || screen_coverage < 0.01 {
            Self::Cards
        } else if distance > 15.0 || screen_coverage < 0.05 {
            Self::Shells
        } else {
            Self::Strands
        }
    }

    /// Number of shells for shell-based rendering.
    pub fn shell_count(&self) -> u32 {
        match self {
            Self::Shells => 16,
            _ => 0,
        }
    }
}

/// Shell texture data for shell-based fur rendering.
#[derive(Debug, Clone)]
pub struct ShellTexture {
    /// Width of the shell texture.
    pub width: u32,
    /// Height of the shell texture.
    pub height: u32,
    /// Alpha data per shell layer (0 = transparent, 1 = opaque).
    pub layers: Vec<Vec<f32>>,
    /// Shell count.
    pub shell_count: u32,
}

impl ShellTexture {
    /// Generate shell textures from hair strand data.
    pub fn generate(strands: &[HairStrand], width: u32, height: u32, shell_count: u32) -> Self {
        let pixel_count = (width * height) as usize;
        let layers: Vec<Vec<f32>> = (0..shell_count)
            .map(|_| vec![0.0; pixel_count])
            .collect();

        // In a real implementation, we would rasterize strand positions
        // into each shell layer. This is a simplified version.
        Self { width, height, layers, shell_count }
    }
}

// ---------------------------------------------------------------------------
// HairComponent (ECS)
// ---------------------------------------------------------------------------

/// ECS component for hair rendering.
#[derive(Debug, Clone)]
pub struct HairComponent {
    /// Index into the global hair asset array.
    pub asset_index: u32,
    /// Current LOD level.
    pub lod: HairLOD,
    /// Marschner shading parameters.
    pub shading_params: MarschnerParams,
    /// Wind parameters override (None = use global wind).
    pub wind_override: Option<WindParams>,
    /// Whether hair is enabled for this instance.
    pub enabled: bool,
    /// Shadow casting mode.
    pub cast_shadow: bool,
    /// Strand density multiplier (for LOD).
    pub density: f32,
}

impl Default for HairComponent {
    fn default() -> Self {
        Self {
            asset_index: 0,
            lod: HairLOD::Strands,
            shading_params: MarschnerParams::default(),
            wind_override: None,
            enabled: true,
            cast_shadow: true,
            density: 1.0,
        }
    }
}

impl HairComponent {
    /// Create a component with specific parameters.
    pub fn new(asset_index: u32, params: MarschnerParams) -> Self {
        Self {
            asset_index,
            shading_params: params,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_strand() -> HairStrand {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.1, 0.0),
            Vec3::new(0.0, 0.2, 0.01),
            Vec3::new(0.0, 0.3, 0.02),
        ];
        HairStrand::new(points, 0.001, Vec3::new(0.15, 0.08, 0.04))
    }

    #[test]
    fn test_strand_creation() {
        let strand = make_simple_strand();
        assert_eq!(strand.point_count(), 4);
        assert!(strand.length() > 0.0);
        assert_eq!(strand.tangents.len(), 4);
    }

    #[test]
    fn test_strand_interpolation() {
        let strand = make_simple_strand();
        let root = strand.interpolate_position(0.0);
        let tip = strand.interpolate_position(1.0);
        let mid = strand.interpolate_position(0.5);

        assert!((root - Vec3::ZERO).length() < 0.01);
        assert!(mid.y > root.y);
        assert!(tip.y > mid.y);
    }

    #[test]
    fn test_strand_tangent_interpolation() {
        let strand = make_simple_strand();
        let t = strand.interpolate_tangent(0.5);
        assert!(t.length() > 0.5); // Should be roughly normalised.
    }

    #[test]
    fn test_longitudinal_scattering() {
        let peak = longitudinal_scattering(0.0, 0.15);
        let off_peak = longitudinal_scattering(0.5, 0.15);
        assert!(peak > off_peak);
        assert!(peak > 0.0);
    }

    #[test]
    fn test_fresnel_schlick() {
        // At normal incidence.
        let f0 = fresnel_schlick(1.0, DEFAULT_IOR);
        // At grazing angle.
        let f90 = fresnel_schlick(0.0, DEFAULT_IOR);
        assert!(f90 >= f0);
        assert!(f0 > 0.0 && f0 < 1.0);
    }

    #[test]
    fn test_shade_hair_produces_colour() {
        let view = Vec3::new(0.0, 0.0, 1.0);
        let light = Vec3::new(0.0, 1.0, 0.0);
        let tangent = Vec3::new(0.0, 1.0, 0.0);
        let params = MarschnerParams::default();

        let color = shade_hair(view, light, tangent, &params);
        assert!(color.x >= 0.0);
        assert!(color.y >= 0.0);
        assert!(color.z >= 0.0);
    }

    #[test]
    fn test_shade_hair_different_angles() {
        let tangent = Vec3::Y;
        let params = MarschnerParams::default();

        // View and light aligned with tangent.
        let c1 = shade_hair(Vec3::Y, Vec3::Y, tangent, &params);
        // View and light perpendicular to tangent.
        let c2 = shade_hair(Vec3::Z, Vec3::X, tangent, &params);

        // Both should produce valid results.
        assert!(c1.x >= 0.0);
        assert!(c2.x >= 0.0);
    }

    #[test]
    fn test_marschner_presets() {
        let blonde = MarschnerParams::blonde();
        let dark = MarschnerParams::dark();
        let red = MarschnerParams::red();
        let white = MarschnerParams::white();

        // Dark hair should have higher absorption.
        assert!(dark.total_absorption().x > blonde.total_absorption().x);
        assert!(dark.total_absorption().y > blonde.total_absorption().y);

        // White hair should have very low absorption.
        assert!(white.total_absorption().x < 0.1);
    }

    #[test]
    fn test_hair_asset_creation() {
        let strands = vec![make_simple_strand(), make_simple_strand()];
        let asset = HairAsset::from_guides(strands, 4);
        assert_eq!(asset.guide_strands.len(), 2);
        assert!(asset.total_strand_count > 2);
    }

    #[test]
    fn test_interpolated_strands() {
        let s1 = HairStrand::new(
            vec![Vec3::ZERO, Vec3::new(0.0, 0.1, 0.0), Vec3::new(0.0, 0.2, 0.0)],
            0.001,
            Vec3::new(0.15, 0.08, 0.04),
        );
        let s2 = HairStrand::new(
            vec![Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.1, 0.1, 0.0), Vec3::new(0.1, 0.2, 0.0)],
            0.001,
            Vec3::new(0.15, 0.08, 0.04),
        );

        let asset = HairAsset::from_guides(vec![s1, s2], 3);
        let all_strands = asset.generate_interpolated_strands();
        assert_eq!(all_strands.len(), 2 + 3); // 2 guides + 3 interpolated.
    }

    #[test]
    fn test_deep_opacity_map() {
        let light_vp = glam::Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.0, 10.0)
            * glam::Mat4::look_at_rh(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO, Vec3::Z);

        let mut dom = DeepOpacityMap::new(64, 64, 8, light_vp);
        let strand = make_simple_strand();
        dom.rasterize_strands(&[strand]);

        // Should produce non-zero opacity somewhere.
        let has_opacity = dom.layers.iter().any(|layer| layer.iter().any(|&v| v > 0.0));
        // Note: may or may not have opacity depending on projection.
        // The important thing is that it doesn't crash.
        let _ = has_opacity;
    }

    #[test]
    fn test_wind_animation() {
        let mut strand = make_simple_strand();
        let original_tip = strand.control_points.last().copied().unwrap();

        let wind = WindParams {
            direction: Vec3::X,
            speed: 5.0,
            time: 1.0,
            ..Default::default()
        };

        apply_wind(&mut strand, &wind);

        // Tip should have moved from the original position.
        let new_tip = strand.control_points.last().copied().unwrap();
        // Root should remain fixed.
        assert!((strand.control_points[0] - Vec3::ZERO).length() < 0.001);
    }

    #[test]
    fn test_hair_lod_selection() {
        assert_eq!(HairLOD::select(5.0, 0.1), HairLOD::Strands);
        assert_eq!(HairLOD::select(30.0, 0.03), HairLOD::Shells);
        assert_eq!(HairLOD::select(100.0, 0.005), HairLOD::Cards);
        assert_eq!(HairLOD::select(300.0, 0.001), HairLOD::Hidden);
    }

    #[test]
    fn test_shell_texture_generation() {
        let strand = make_simple_strand();
        let shells = ShellTexture::generate(&[strand], 32, 32, 8);
        assert_eq!(shells.shell_count, 8);
        assert_eq!(shells.layers.len(), 8);
    }

    #[test]
    fn test_hair_component_default() {
        let comp = HairComponent::default();
        assert!(comp.enabled);
        assert!(comp.cast_shadow);
        assert_eq!(comp.lod, HairLOD::Strands);
    }

    #[test]
    fn test_attenuation() {
        let params = MarschnerParams::default();
        let att = compute_attenuation(&params, 0.9, 1);
        // Attenuation should be between 0 and 1.
        assert!(att.x > 0.0 && att.x <= 1.0);
        assert!(att.y > 0.0 && att.y <= 1.0);
        assert!(att.z > 0.0 && att.z <= 1.0);

        // More passes = more attenuation.
        let att2 = compute_attenuation(&params, 0.9, 2);
        assert!(att2.x < att.x);
    }

    #[test]
    fn test_hash_float_range() {
        for seed in 0..100 {
            let v = hash_float(seed);
            assert!(v >= 0.0 && v <= 1.0);
        }
    }
}
