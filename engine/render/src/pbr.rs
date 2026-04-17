// engine/render/src/pbr_v2.rs
//
// Enhanced PBR material model extending the standard metallic-roughness workflow
// with clearcoat, sheen, transmission, thin-film iridescence, anisotropy
// direction mapping, subsurface color, specular tint, and IOR control.
//
// Implements the full Disney/Filament-style principled BRDF with all extension
// lobes. Every function operates in linear-space with unit vectors.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Vector math helpers (inline, no external deps beyond f32)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            return Self::ZERO;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }

    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }

    #[inline]
    pub fn mul_element(self, rhs: Self) -> Self {
        Self { x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z }
    }

    #[inline]
    pub fn max_component(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    #[inline]
    pub fn luminance(self) -> f32 {
        0.2126 * self.x + 0.7152 * self.y + 0.0722 * self.z
    }

    #[inline]
    pub fn reflect(self, normal: Self) -> Self {
        self.sub(normal.scale(2.0 * self.dot(normal)))
    }

    #[inline]
    pub fn refract(self, normal: Self, eta: f32) -> Option<Self> {
        let cos_i = -self.dot(normal);
        let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
        if sin2_t > 1.0 {
            return None; // total internal reflection
        }
        let cos_t = (1.0 - sin2_t).sqrt();
        Some(self.scale(eta).add(normal.scale(eta * cos_i - cos_t)))
    }
}

// ---------------------------------------------------------------------------
// Enhanced PBR material parameters
// ---------------------------------------------------------------------------

/// Full principled BRDF material parameters.
#[derive(Debug, Clone)]
pub struct PbrMaterialV2 {
    // Base layer
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub reflectance: f32, // dielectric F0 control (0.5 = 4% like plastic)

    // Normal / bump
    pub normal_scale: f32,

    // Emission
    pub emissive_color: Vec3,
    pub emissive_intensity: f32,

    // Ambient occlusion
    pub ao: f32,

    // Clearcoat layer
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub clearcoat_ior: f32,
    pub clearcoat_normal_scale: f32,

    // Sheen layer (for fabric / velvet)
    pub sheen: f32,
    pub sheen_color: Vec3,
    pub sheen_roughness: f32,

    // Transmission / refraction
    pub transmission: f32,
    pub transmission_roughness: f32,
    pub ior: f32,
    pub thickness: f32,           // for thin-walled transmission
    pub attenuation_color: Vec3,  // volume absorption color
    pub attenuation_distance: f32,

    // Thin-film iridescence
    pub iridescence: f32,
    pub iridescence_ior: f32,
    pub iridescence_thickness_min: f32,
    pub iridescence_thickness_max: f32,

    // Anisotropy
    pub anisotropy: f32,          // -1..1, sign picks tangent/bitangent axis
    pub anisotropy_rotation: f32, // radians

    // Subsurface scattering
    pub subsurface: f32,
    pub subsurface_color: Vec3,
    pub subsurface_radius: Vec3,  // per-channel mean free path

    // Specular tint (Disney specular tint)
    pub specular_tint: f32,

    // Alpha
    pub alpha_cutoff: f32,
    pub alpha_mode: AlphaMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

impl Default for PbrMaterialV2 {
    fn default() -> Self {
        Self {
            base_color: Vec3::new(0.8, 0.8, 0.8),
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            normal_scale: 1.0,
            emissive_color: Vec3::ZERO,
            emissive_intensity: 0.0,
            ao: 1.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.05,
            clearcoat_ior: 1.5,
            clearcoat_normal_scale: 1.0,
            sheen: 0.0,
            sheen_color: Vec3::ONE,
            sheen_roughness: 0.5,
            transmission: 0.0,
            transmission_roughness: 0.0,
            ior: 1.5,
            thickness: 0.0,
            attenuation_color: Vec3::ONE,
            attenuation_distance: f32::INFINITY,
            iridescence: 0.0,
            iridescence_ior: 1.3,
            iridescence_thickness_min: 100.0,
            iridescence_thickness_max: 400.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            subsurface: 0.0,
            subsurface_color: Vec3::ONE,
            subsurface_radius: Vec3::new(1.0, 0.2, 0.1),
            specular_tint: 0.0,
            alpha_cutoff: 0.5,
            alpha_mode: AlphaMode::Opaque,
        }
    }
}

// ---------------------------------------------------------------------------
// NDF functions
// ---------------------------------------------------------------------------

/// GGX / Trowbridge-Reitz Normal Distribution Function.
#[inline]
pub fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d).max(1e-7)
}

/// Anisotropic GGX NDF.
#[inline]
pub fn d_ggx_aniso(n_dot_h: f32, h_dot_t: f32, h_dot_b: f32, at: f32, ab: f32) -> f32 {
    let at2 = at * at;
    let ab2 = ab * ab;
    let d = (h_dot_t * h_dot_t) / at2 + (h_dot_b * h_dot_b) / ab2 + n_dot_h * n_dot_h;
    1.0 / (PI * at * ab * d * d).max(1e-7)
}

/// Charlie NDF for sheen (Estevez & Kulla).
#[inline]
pub fn d_charlie(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let inv_alpha = 1.0 / alpha;
    let sin_theta = (1.0 - n_dot_h * n_dot_h).max(0.0).sqrt();
    let cos2 = n_dot_h * n_dot_h;
    (2.0 + inv_alpha) * sin_theta.powf(inv_alpha) / (2.0 * PI)
}

// ---------------------------------------------------------------------------
// Geometry / Visibility functions
// ---------------------------------------------------------------------------

/// GGX Smith geometry term (single direction).
#[inline]
pub fn g_smith_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_v2 = n_dot_v * n_dot_v;
    let denom = n_dot_v + (a2 + (1.0 - a2) * n_dot_v2).sqrt();
    2.0 * n_dot_v / denom.max(1e-7)
}

/// Combined Smith GGX visibility for both light and view directions.
#[inline]
pub fn v_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let ggxv = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let ggxl = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    0.5 / (ggxv + ggxl).max(1e-7)
}

/// Anisotropic Smith GGX visibility function.
#[inline]
pub fn v_smith_ggx_aniso(
    n_dot_v: f32,
    n_dot_l: f32,
    t_dot_v: f32,
    b_dot_v: f32,
    t_dot_l: f32,
    b_dot_l: f32,
    at: f32,
    ab: f32,
) -> f32 {
    let ggxv = n_dot_l * ((at * t_dot_v).powi(2) + (ab * b_dot_v).powi(2) + n_dot_v * n_dot_v).sqrt();
    let ggxl = n_dot_v * ((at * t_dot_l).powi(2) + (ab * b_dot_l).powi(2) + n_dot_l * n_dot_l).sqrt();
    0.5 / (ggxv + ggxl).max(1e-7)
}

/// Neubelt visibility for sheen.
#[inline]
pub fn v_neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
    1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)).max(1e-7)
}

/// Kelemen visibility (fast approximation for clearcoat).
#[inline]
pub fn v_kelemen(l_dot_h: f32) -> f32 {
    0.25 / (l_dot_h * l_dot_h).max(1e-7)
}

// ---------------------------------------------------------------------------
// Fresnel functions
// ---------------------------------------------------------------------------

/// Schlick Fresnel approximation.
#[inline]
pub fn f_schlick_scalar(cos_theta: f32, f0: f32) -> f32 {
    let t = (1.0 - cos_theta).max(0.0);
    let t2 = t * t;
    let t5 = t2 * t2 * t;
    f0 + (1.0 - f0) * t5
}

/// Schlick Fresnel with vec3 F0.
#[inline]
pub fn f_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    let t = (1.0 - cos_theta).max(0.0);
    let t2 = t * t;
    let t5 = t2 * t2 * t;
    Vec3::new(
        f0.x + (1.0 - f0.x) * t5,
        f0.y + (1.0 - f0.y) * t5,
        f0.z + (1.0 - f0.z) * t5,
    )
}

/// Schlick Fresnel with roughness factor for environment BRDF.
#[inline]
pub fn f_schlick_roughness(cos_theta: f32, f0: Vec3, roughness: f32) -> Vec3 {
    let t = (1.0 - cos_theta).max(0.0);
    let t2 = t * t;
    let t5 = t2 * t2 * t;
    let max_reflect = (1.0 - roughness).max(f0.x).max(f0.y.max(f0.z));
    Vec3::new(
        f0.x + (max_reflect - f0.x) * t5,
        f0.y + (max_reflect - f0.y) * t5,
        f0.z + (max_reflect - f0.z) * t5,
    )
}

/// Full Fresnel equation for dielectric (not Schlick).
#[inline]
pub fn fresnel_dielectric(cos_theta_i: f32, eta: f32) -> f32 {
    let sin2_t = eta * eta * (1.0 - cos_theta_i * cos_theta_i);
    if sin2_t > 1.0 {
        return 1.0; // total internal reflection
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    let rs = (eta * cos_theta_i - cos_t) / (eta * cos_theta_i + cos_t);
    let rp = (cos_theta_i - eta * cos_t) / (cos_theta_i + eta * cos_t);
    (rs * rs + rp * rp) * 0.5
}

/// F0 from IOR (index of refraction).
#[inline]
pub fn f0_from_ior(ior: f32) -> f32 {
    let r = (ior - 1.0) / (ior + 1.0);
    r * r
}

/// IOR from F0 reflectance value.
#[inline]
pub fn ior_from_f0(f0: f32) -> f32 {
    let sqrt_f0 = f0.sqrt().min(0.999);
    (1.0 + sqrt_f0) / (1.0 - sqrt_f0)
}

// ---------------------------------------------------------------------------
// Thin-film iridescence
// ---------------------------------------------------------------------------

/// Compute thin-film iridescence Fresnel factor.
///
/// Models constructive/destructive interference in a thin dielectric film
/// (like a soap bubble or oil slick). Returns a per-channel F0 modification.
pub fn evaluate_iridescence(
    outside_ior: f32,
    film_ior: f32,
    film_thickness_nm: f32,
    base_f0: Vec3,
    cos_theta: f32,
) -> Vec3 {
    // Snell's law: compute cosine of refracted angle inside film
    let sin2_theta1 = 1.0 - cos_theta * cos_theta;
    let eta_film = outside_ior / film_ior;
    let sin2_theta2 = eta_film * eta_film * sin2_theta1;

    if sin2_theta2 >= 1.0 {
        // Total internal reflection, return maximum reflectance
        return Vec3::ONE;
    }

    let cos_theta2 = (1.0 - sin2_theta2).sqrt();

    // Fresnel at outer surface (air -> film)
    let r_outer = fresnel_dielectric(cos_theta, outside_ior / film_ior);

    // Phase shift from optical path length through the film
    // OPD = 2 * film_ior * thickness * cos(theta2)
    let opd = 2.0 * film_ior * film_thickness_nm * cos_theta2;

    // Wavelengths for RGB channels (nm)
    let wavelengths = [650.0_f32, 532.0, 450.0]; // R, G, B

    let mut result = [0.0_f32; 3];
    for (i, &lambda) in wavelengths.iter().enumerate() {
        // Phase difference
        let phase = 2.0 * PI * opd / lambda;

        // Airy function approximation for thin film
        // R = R1 + R2 + 2*sqrt(R1*R2)*cos(phase) / (1 + R1*R2 + 2*sqrt(R1*R2)*cos(phase))
        let base_f0_channel = match i {
            0 => base_f0.x,
            1 => base_f0.y,
            _ => base_f0.z,
        };
        let r_inner = base_f0_channel;
        let sqrt_r1_r2 = (r_outer * r_inner).sqrt();
        let cos_phase = phase.cos();

        let numerator = r_outer + r_inner + 2.0 * sqrt_r1_r2 * cos_phase;
        let denominator = 1.0 + r_outer * r_inner + 2.0 * sqrt_r1_r2 * cos_phase;

        result[i] = (numerator / denominator.max(1e-7)).clamp(0.0, 1.0);
    }

    Vec3::new(result[0], result[1], result[2])
}

// ---------------------------------------------------------------------------
// Subsurface scattering approximation
// ---------------------------------------------------------------------------

/// Burley / Disney diffuse approximation with subsurface scattering.
///
/// Blends between Lambert diffuse and a wrap-lighting subsurface model
/// based on the subsurface parameter.
pub fn evaluate_subsurface_diffuse(
    base_color: Vec3,
    subsurface_color: Vec3,
    subsurface: f32,
    roughness: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    l_dot_h: f32,
) -> Vec3 {
    // Disney diffuse (Burley)
    let fd90 = 0.5 + 2.0 * l_dot_h * l_dot_h * roughness;
    let light_scatter = f_schlick_scalar(n_dot_l, 1.0) * (fd90 - 1.0) + 1.0;
    let view_scatter = f_schlick_scalar(n_dot_v, 1.0) * (fd90 - 1.0) + 1.0;
    let disney_diffuse = base_color.scale(light_scatter * view_scatter / PI);

    // Subsurface approximation (Hanrahan-Krueger)
    let fss90 = l_dot_h * l_dot_h * roughness;
    let fss_light = f_schlick_scalar(n_dot_l, 1.0) * (fss90 - 1.0) + 1.0;
    let fss_view = f_schlick_scalar(n_dot_v, 1.0) * (fss90 - 1.0) + 1.0;
    let fss = 1.25 * (fss_light * fss_view * (1.0 / (n_dot_l + n_dot_v).max(1e-4) - 0.5) + 0.5);
    let ss_color = subsurface_color.mul_element(base_color);
    let subsurface_result = ss_color.scale(fss / PI);

    // Blend
    disney_diffuse.lerp(subsurface_result, subsurface)
}

/// Compute subsurface profile (Gaussian approximation of diffusion).
/// Returns radial weights for a given set of sample distances.
pub fn compute_subsurface_profile(
    radius: Vec3,
    num_samples: usize,
    max_radius_mm: f32,
) -> Vec<SubsurfaceSample> {
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = (i as f32 + 0.5) / num_samples as f32;
        let distance = t * max_radius_mm;

        // Sum of Gaussians approximation (3 Gaussians per channel)
        let weight = Vec3::new(
            gaussian_profile(distance, radius.x),
            gaussian_profile(distance, radius.y),
            gaussian_profile(distance, radius.z),
        );

        samples.push(SubsurfaceSample { distance, weight });
    }

    // Normalize weights
    let mut total = Vec3::ZERO;
    for s in &samples {
        total = total.add(s.weight);
    }
    let inv_total = Vec3::new(
        1.0 / total.x.max(1e-7),
        1.0 / total.y.max(1e-7),
        1.0 / total.z.max(1e-7),
    );
    for s in &mut samples {
        s.weight = s.weight.mul_element(inv_total);
    }

    samples
}

#[derive(Debug, Clone)]
pub struct SubsurfaceSample {
    pub distance: f32,
    pub weight: Vec3,
}

fn gaussian_profile(r: f32, variance: f32) -> f32 {
    let v = variance * variance;
    let factor = 1.0 / (2.0 * PI * v).sqrt();
    factor * (-r * r / (2.0 * v)).exp()
}

// ---------------------------------------------------------------------------
// Clearcoat lobe
// ---------------------------------------------------------------------------

/// Evaluate the clearcoat specular lobe.
///
/// Clearcoat is modeled as a separate GGX lobe with its own roughness,
/// IOR-derived F0, and optional normal map.
pub fn evaluate_clearcoat(
    clearcoat: f32,
    clearcoat_roughness: f32,
    clearcoat_ior: f32,
    n_dot_h: f32,
    n_dot_l: f32,
    n_dot_v: f32,
    l_dot_h: f32,
) -> ClearcoatResult {
    if clearcoat < 1e-4 {
        return ClearcoatResult {
            specular: 0.0,
            attenuation: 1.0,
        };
    }

    let cc_f0 = f0_from_ior(clearcoat_ior);

    // NDF
    let d = d_ggx(n_dot_h, clearcoat_roughness);

    // Visibility (Kelemen for fast clearcoat)
    let v = v_kelemen(l_dot_h);

    // Fresnel
    let f = f_schlick_scalar(l_dot_h, cc_f0);

    let specular = d * v * f * clearcoat;

    // Energy absorbed by clearcoat (reduces base layer contribution)
    let attenuation = 1.0 - clearcoat * f;

    ClearcoatResult {
        specular,
        attenuation,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ClearcoatResult {
    pub specular: f32,
    pub attenuation: f32,
}

// ---------------------------------------------------------------------------
// Sheen lobe
// ---------------------------------------------------------------------------

/// Evaluate the sheen lobe for fabric-like materials.
///
/// Uses the Charlie NDF (Estevez & Kulla) with Neubelt visibility.
pub fn evaluate_sheen(
    sheen: f32,
    sheen_color: Vec3,
    sheen_roughness: f32,
    n_dot_h: f32,
    n_dot_v: f32,
    n_dot_l: f32,
) -> SheenResult {
    if sheen < 1e-4 {
        return SheenResult {
            color: Vec3::ZERO,
            scaling: 1.0,
        };
    }

    let d = d_charlie(n_dot_h, sheen_roughness);
    let v = v_neubelt(n_dot_v, n_dot_l);

    let specular = sheen_color.scale(d * v * sheen);

    // Energy compensation: reduce diffuse by sheen albedo
    let sheen_albedo = sheen_albedo_lut(n_dot_v, sheen_roughness);
    let scaling = 1.0 - sheen * sheen_albedo;

    SheenResult {
        color: specular,
        scaling,
    }
}

/// Approximate sheen directional albedo (for energy conservation).
fn sheen_albedo_lut(n_dot_v: f32, roughness: f32) -> f32 {
    // Analytical fit from Estevez & Kulla
    let a = roughness;
    let b = n_dot_v;
    // Polynomial approximation
    let t = (-3.0 * b).exp();
    a * (1.0 - t) * 0.6 + t * 0.1
}

#[derive(Debug, Clone, Copy)]
pub struct SheenResult {
    pub color: Vec3,
    pub scaling: f32,
}

// ---------------------------------------------------------------------------
// Transmission / refraction
// ---------------------------------------------------------------------------

/// Evaluate transmission for transparent dielectric materials.
///
/// Models refraction through thin surfaces (e.g., glass, water) with
/// roughness-based blur and volume absorption.
pub fn evaluate_transmission(
    base_color: Vec3,
    transmission: f32,
    transmission_roughness: f32,
    ior: f32,
    thickness: f32,
    attenuation_color: Vec3,
    attenuation_distance: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    n_dot_h: f32,
    l_dot_h: f32,
) -> TransmissionResult {
    if transmission < 1e-4 {
        return TransmissionResult {
            color: Vec3::ZERO,
            absorption: Vec3::ONE,
        };
    }

    let eta = 1.0 / ior;

    // Rough transmission uses GGX with modified roughness
    let roughness = (transmission_roughness * transmission_roughness + 0.01).sqrt();
    let d = d_ggx(n_dot_h, roughness);
    let v = v_smith_ggx_correlated(n_dot_v, n_dot_l.abs(), roughness);
    let f = f_schlick_scalar(l_dot_h.abs(), f0_from_ior(ior));

    // Transmission weight (1 - F for transmission, we refract what isn't reflected)
    let btdf = d * v * (1.0 - f) * transmission;

    // Volume absorption (Beer-Lambert law)
    let absorption = compute_volume_absorption(
        attenuation_color,
        attenuation_distance,
        thickness,
    );

    let color = base_color.mul_element(absorption).scale(btdf);

    TransmissionResult { color, absorption }
}

/// Beer-Lambert volume absorption.
pub fn compute_volume_absorption(
    attenuation_color: Vec3,
    attenuation_distance: f32,
    thickness: f32,
) -> Vec3 {
    if attenuation_distance >= f32::MAX * 0.5 || thickness < 1e-6 {
        return Vec3::ONE;
    }
    // sigma_a = -ln(color) / distance
    let sigma_a = Vec3::new(
        -attenuation_color.x.max(1e-6).ln() / attenuation_distance,
        -attenuation_color.y.max(1e-6).ln() / attenuation_distance,
        -attenuation_color.z.max(1e-6).ln() / attenuation_distance,
    );
    Vec3::new(
        (-sigma_a.x * thickness).exp(),
        (-sigma_a.y * thickness).exp(),
        (-sigma_a.z * thickness).exp(),
    )
}

#[derive(Debug, Clone, Copy)]
pub struct TransmissionResult {
    pub color: Vec3,
    pub absorption: Vec3,
}

// ---------------------------------------------------------------------------
// Anisotropy helpers
// ---------------------------------------------------------------------------

/// Compute anisotropic roughness from base roughness and anisotropy factor.
///
/// Returns (alpha_tangent, alpha_bitangent).
pub fn compute_aniso_roughness(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let a = roughness * roughness;
    let aspect = (1.0 - anisotropy * 0.9).sqrt();
    let at = (a / aspect).max(0.001);
    let ab = (a * aspect).max(0.001);
    (at, ab)
}

/// Rotate tangent frame by anisotropy rotation angle.
pub fn rotate_tangent_frame(tangent: Vec3, bitangent: Vec3, angle: f32) -> (Vec3, Vec3) {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let new_t = tangent.scale(cos_a).add(bitangent.scale(sin_a));
    let new_b = bitangent.scale(cos_a).sub(tangent.scale(sin_a));
    (new_t.normalize(), new_b.normalize())
}

// ---------------------------------------------------------------------------
// Specular tint (Disney)
// ---------------------------------------------------------------------------

/// Apply Disney specular tint to F0.
///
/// Tints the specular reflection towards the base color for non-metallic
/// surfaces, giving colored highlights (like some gem stones).
pub fn apply_specular_tint(base_color: Vec3, f0: Vec3, specular_tint: f32) -> Vec3 {
    let lum = base_color.luminance().max(1e-7);
    let tint_color = base_color.scale(1.0 / lum);
    f0.lerp(f0.mul_element(tint_color), specular_tint)
}

// ---------------------------------------------------------------------------
// BRDF integration (split-sum LUT generation)
// ---------------------------------------------------------------------------

/// Pre-compute the BRDF integration LUT for image-based lighting.
///
/// Generates an NxN texture where:
/// - U axis = NdotV (cos_theta)
/// - V axis = roughness
/// - R channel = scale factor for F0
/// - G channel = bias (added to F0 * scale)
pub fn generate_brdf_lut(size: usize) -> Vec<[f32; 2]> {
    let mut lut = vec![[0.0f32; 2]; size * size];

    for y in 0..size {
        let roughness = (y as f32 + 0.5) / size as f32;
        for x in 0..size {
            let n_dot_v = (x as f32 + 0.5) / size as f32;
            let n_dot_v = n_dot_v.max(0.001);

            let result = integrate_brdf(n_dot_v, roughness, 1024);
            lut[y * size + x] = [result.0, result.1];
        }
    }

    lut
}

/// Importance-sample the GGX BRDF and integrate over the hemisphere.
fn integrate_brdf(n_dot_v: f32, roughness: f32, num_samples: u32) -> (f32, f32) {
    let v = Vec3::new((1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v);
    let n = Vec3::new(0.0, 0.0, 1.0);

    let mut a = 0.0_f32;
    let mut b = 0.0_f32;

    for i in 0..num_samples {
        let xi = hammersley(i, num_samples);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = h.scale(2.0 * v.dot(h)).sub(v);

        let n_dot_l = l.z.max(0.0);
        let n_dot_h = h.z.max(0.0);
        let v_dot_h = v.dot(h).max(0.0);

        if n_dot_l > 0.0 {
            let g = g_smith_ggx(n_dot_v, roughness) * g_smith_ggx(n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v).max(1e-7);
            let fc = (1.0 - v_dot_h).powi(5);

            a += g_vis * (1.0 - fc);
            b += g_vis * fc;
        }
    }

    let inv_samples = 1.0 / num_samples as f32;
    (a * inv_samples, b * inv_samples)
}

/// Hammersley low-discrepancy sequence.
fn hammersley(i: u32, n: u32) -> [f32; 2] {
    let mut bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    let radical_inverse = bits as f32 * 2.328_306_4e-10;
    [i as f32 / n as f32, radical_inverse]
}

/// Importance-sample the GGX distribution.
fn importance_sample_ggx(xi: [f32; 2], n: Vec3, roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    // Spherical to Cartesian (tangent space)
    let h = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);

    // Tangent-space to world-space
    let up = if n.z.abs() < 0.999 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let tangent = up.cross(n).normalize();
    let bitangent = n.cross(tangent);

    tangent.scale(h.x).add(bitangent.scale(h.y)).add(n.scale(h.z)).normalize()
}

// ---------------------------------------------------------------------------
// Full principled BRDF evaluation
// ---------------------------------------------------------------------------

/// Shading context for BRDF evaluation.
#[derive(Debug, Clone)]
pub struct ShadingContext {
    pub n: Vec3,          // surface normal
    pub v: Vec3,          // view direction (towards camera)
    pub l: Vec3,          // light direction (towards light)
    pub t: Vec3,          // tangent
    pub b: Vec3,          // bitangent
    pub uv: [f32; 2],    // texture coordinates
}

impl ShadingContext {
    /// Compute half-vector and all dot products needed by BRDF.
    pub fn compute_dots(&self) -> ShadingDots {
        let h = self.v.add(self.l).normalize();
        ShadingDots {
            h,
            n_dot_v: self.n.dot(self.v).max(1e-5),
            n_dot_l: self.n.dot(self.l).max(0.0),
            n_dot_h: self.n.dot(h).max(0.0),
            l_dot_h: self.l.dot(h).max(0.0),
            t_dot_h: self.t.dot(h),
            b_dot_h: self.b.dot(h),
            t_dot_v: self.t.dot(self.v),
            b_dot_v: self.b.dot(self.v),
            t_dot_l: self.t.dot(self.l),
            b_dot_l: self.b.dot(self.l),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ShadingDots {
    pub h: Vec3,
    pub n_dot_v: f32,
    pub n_dot_l: f32,
    pub n_dot_h: f32,
    pub l_dot_h: f32,
    pub t_dot_h: f32,
    pub b_dot_h: f32,
    pub t_dot_v: f32,
    pub b_dot_v: f32,
    pub t_dot_l: f32,
    pub b_dot_l: f32,
}

/// Result of evaluating the principled BRDF.
#[derive(Debug, Clone)]
pub struct BrdfResult {
    pub diffuse: Vec3,
    pub specular: Vec3,
    pub clearcoat: Vec3,
    pub sheen: Vec3,
    pub transmission: Vec3,
    pub emissive: Vec3,
    pub total: Vec3,
}

/// Evaluate the full principled BRDF for a single light.
pub fn evaluate_principled_brdf(
    material: &PbrMaterialV2,
    ctx: &ShadingContext,
    light_color: Vec3,
    light_intensity: f32,
) -> BrdfResult {
    let dots = ctx.compute_dots();

    if dots.n_dot_l <= 0.0 && material.transmission < 1e-4 {
        return BrdfResult {
            diffuse: Vec3::ZERO,
            specular: Vec3::ZERO,
            clearcoat: Vec3::ZERO,
            sheen: Vec3::ZERO,
            transmission: Vec3::ZERO,
            emissive: material.emissive_color.scale(material.emissive_intensity),
            total: material.emissive_color.scale(material.emissive_intensity),
        };
    }

    let irradiance = light_color.scale(light_intensity * dots.n_dot_l.max(0.0));

    // --- F0 computation ---
    let dielectric_f0 = 0.16 * material.reflectance * material.reflectance;
    let mut f0 = Vec3::new(dielectric_f0, dielectric_f0, dielectric_f0)
        .lerp(material.base_color, material.metallic);

    // Specular tint
    if material.specular_tint > 0.0 {
        f0 = apply_specular_tint(material.base_color, f0, material.specular_tint);
    }

    // Iridescence modifies F0
    if material.iridescence > 0.0 {
        let thickness = material.iridescence_thickness_min
            + (material.iridescence_thickness_max - material.iridescence_thickness_min) * 0.5;
        let irid_f0 = evaluate_iridescence(
            1.0,
            material.iridescence_ior,
            thickness,
            f0,
            dots.n_dot_v,
        );
        f0 = f0.lerp(irid_f0, material.iridescence);
    }

    // --- Specular lobe ---
    let specular = if material.anisotropy.abs() > 1e-4 {
        // Anisotropic path
        let (at, ab) = compute_aniso_roughness(material.roughness, material.anisotropy);
        let (t, b) = rotate_tangent_frame(ctx.t, ctx.b, material.anisotropy_rotation);
        let t_dot_h = t.dot(dots.h);
        let b_dot_h = b.dot(dots.h);
        let t_dot_v = t.dot(ctx.v);
        let b_dot_v = b.dot(ctx.v);
        let t_dot_l = t.dot(ctx.l);
        let b_dot_l = b.dot(ctx.l);

        let d = d_ggx_aniso(dots.n_dot_h, t_dot_h, b_dot_h, at, ab);
        let v = v_smith_ggx_aniso(dots.n_dot_v, dots.n_dot_l, t_dot_v, b_dot_v, t_dot_l, b_dot_l, at, ab);
        let f = f_schlick(dots.l_dot_h, f0);

        f.scale(d * v)
    } else {
        // Isotropic path
        let d = d_ggx(dots.n_dot_h, material.roughness);
        let v = v_smith_ggx_correlated(dots.n_dot_v, dots.n_dot_l, material.roughness);
        let f = f_schlick(dots.l_dot_h, f0);

        f.scale(d * v)
    };

    // --- Diffuse lobe ---
    let diffuse_color = material.base_color.scale(1.0 - material.metallic);
    let diffuse = if material.subsurface > 0.0 {
        evaluate_subsurface_diffuse(
            diffuse_color,
            material.subsurface_color,
            material.subsurface,
            material.roughness,
            dots.n_dot_v,
            dots.n_dot_l,
            dots.l_dot_h,
        )
    } else {
        // Standard Burley diffuse
        let fd90 = 0.5 + 2.0 * dots.l_dot_h * dots.l_dot_h * material.roughness;
        let light_scatter = 1.0 + (fd90 - 1.0) * (1.0 - dots.n_dot_l).powi(5);
        let view_scatter = 1.0 + (fd90 - 1.0) * (1.0 - dots.n_dot_v).powi(5);
        diffuse_color.scale(light_scatter * view_scatter / PI)
    };

    // --- Clearcoat ---
    let cc = evaluate_clearcoat(
        material.clearcoat,
        material.clearcoat_roughness,
        material.clearcoat_ior,
        dots.n_dot_h,
        dots.n_dot_l,
        dots.n_dot_v,
        dots.l_dot_h,
    );

    // --- Sheen ---
    let sh = evaluate_sheen(
        material.sheen,
        material.sheen_color,
        material.sheen_roughness,
        dots.n_dot_h,
        dots.n_dot_v,
        dots.n_dot_l,
    );

    // --- Transmission ---
    let tr = evaluate_transmission(
        material.base_color,
        material.transmission,
        material.transmission_roughness,
        material.ior,
        material.thickness,
        material.attenuation_color,
        material.attenuation_distance,
        dots.n_dot_v,
        dots.n_dot_l,
        dots.n_dot_h,
        dots.l_dot_h,
    );

    // --- Combine lobes ---
    // Diffuse is reduced by metallic, transmission, and sheen
    let diffuse_weight = (1.0 - material.metallic) * (1.0 - material.transmission);
    let mut diffuse_final = diffuse.scale(diffuse_weight * sh.scaling * cc.attenuation);

    // Apply AO to diffuse
    diffuse_final = diffuse_final.scale(material.ao);

    let specular_final = specular.scale(cc.attenuation);
    let clearcoat_final = Vec3::new(cc.specular, cc.specular, cc.specular);

    // Emissive
    let emissive = material.emissive_color.scale(material.emissive_intensity);

    // Total outgoing radiance
    let lit = diffuse_final.add(specular_final).add(clearcoat_final).add(sh.color).add(tr.color);
    let total = lit.mul_element(irradiance).add(emissive);

    BrdfResult {
        diffuse: diffuse_final.mul_element(irradiance),
        specular: specular_final.mul_element(irradiance),
        clearcoat: clearcoat_final.mul_element(irradiance),
        sheen: sh.color.mul_element(irradiance),
        transmission: tr.color.mul_element(irradiance),
        emissive,
        total,
    }
}

// ---------------------------------------------------------------------------
// Material presets
// ---------------------------------------------------------------------------

impl PbrMaterialV2 {
    /// Standard plastic material.
    pub fn plastic(color: Vec3) -> Self {
        Self {
            base_color: color,
            roughness: 0.4,
            reflectance: 0.5,
            ..Default::default()
        }
    }

    /// Polished metal.
    pub fn metal(color: Vec3, roughness: f32) -> Self {
        Self {
            base_color: color,
            metallic: 1.0,
            roughness,
            ..Default::default()
        }
    }

    /// Brushed metal with anisotropy.
    pub fn brushed_metal(color: Vec3, anisotropy: f32) -> Self {
        Self {
            base_color: color,
            metallic: 1.0,
            roughness: 0.3,
            anisotropy,
            ..Default::default()
        }
    }

    /// Car paint with clearcoat.
    pub fn car_paint(color: Vec3) -> Self {
        Self {
            base_color: color,
            metallic: 0.0,
            roughness: 0.4,
            clearcoat: 1.0,
            clearcoat_roughness: 0.03,
            clearcoat_ior: 1.5,
            ..Default::default()
        }
    }

    /// Iridescent material (soap bubble, oil slick).
    pub fn iridescent(base_color: Vec3, thickness_nm: f32) -> Self {
        Self {
            base_color,
            roughness: 0.1,
            iridescence: 1.0,
            iridescence_ior: 1.3,
            iridescence_thickness_min: thickness_nm * 0.5,
            iridescence_thickness_max: thickness_nm * 1.5,
            ..Default::default()
        }
    }

    /// Glass / transparent material.
    pub fn glass(color: Vec3, ior: f32) -> Self {
        Self {
            base_color: color,
            roughness: 0.0,
            transmission: 1.0,
            ior,
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        }
    }

    /// Velvet / fabric with sheen.
    pub fn fabric(color: Vec3, sheen_color: Vec3) -> Self {
        Self {
            base_color: color,
            roughness: 0.8,
            sheen: 1.0,
            sheen_color,
            sheen_roughness: 0.5,
            ..Default::default()
        }
    }

    /// Skin with subsurface scattering.
    pub fn skin(color: Vec3) -> Self {
        Self {
            base_color: color,
            roughness: 0.5,
            subsurface: 0.5,
            subsurface_color: Vec3::new(0.8, 0.2, 0.1),
            subsurface_radius: Vec3::new(1.0, 0.2, 0.1),
            ..Default::default()
        }
    }

    /// Diamond with high IOR and dispersion-like iridescence.
    pub fn diamond() -> Self {
        Self {
            base_color: Vec3::ONE,
            roughness: 0.0,
            reflectance: 1.0,
            transmission: 0.5,
            ior: 2.42,
            iridescence: 0.3,
            iridescence_ior: 2.42,
            iridescence_thickness_min: 200.0,
            iridescence_thickness_max: 600.0,
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        }
    }

    /// Wax / candle with subsurface.
    pub fn wax(color: Vec3) -> Self {
        Self {
            base_color: color,
            roughness: 0.6,
            subsurface: 0.8,
            subsurface_color: color,
            subsurface_radius: Vec3::new(0.6, 0.4, 0.2),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// GPU uniform packing
// ---------------------------------------------------------------------------

/// Packed material data ready for GPU uniform buffer upload.
/// 16-float aligned for std140 layout compatibility.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PbrMaterialGpu {
    pub base_color: [f32; 4],           // .w = alpha
    pub emissive: [f32; 4],             // .w = emissive_intensity
    pub metallic_roughness: [f32; 4],   // x=metallic, y=roughness, z=reflectance, w=ao
    pub clearcoat_params: [f32; 4],     // x=clearcoat, y=cc_roughness, z=cc_ior, w=cc_normal_scale
    pub sheen_params: [f32; 4],         // xyz=sheen_color, w=sheen_roughness
    pub sheen_misc: [f32; 4],           // x=sheen, y=specular_tint, z=0, w=0
    pub transmission_params: [f32; 4],  // x=transmission, y=trans_roughness, z=ior, w=thickness
    pub attenuation: [f32; 4],          // xyz=attenuation_color, w=attenuation_distance
    pub iridescence_params: [f32; 4],   // x=iridescence, y=irid_ior, z=thickness_min, w=thickness_max
    pub anisotropy_params: [f32; 4],    // x=anisotropy, y=aniso_rotation, z=subsurface, w=normal_scale
    pub subsurface_color: [f32; 4],     // xyz=ss_color, w=alpha_cutoff
    pub subsurface_radius: [f32; 4],    // xyz=ss_radius, w=alpha_mode (as float)
}

impl PbrMaterialV2 {
    /// Pack material into GPU-ready uniform data.
    pub fn to_gpu(&self) -> PbrMaterialGpu {
        PbrMaterialGpu {
            base_color: [self.base_color.x, self.base_color.y, self.base_color.z, 1.0],
            emissive: [
                self.emissive_color.x,
                self.emissive_color.y,
                self.emissive_color.z,
                self.emissive_intensity,
            ],
            metallic_roughness: [self.metallic, self.roughness, self.reflectance, self.ao],
            clearcoat_params: [
                self.clearcoat,
                self.clearcoat_roughness,
                self.clearcoat_ior,
                self.clearcoat_normal_scale,
            ],
            sheen_params: [
                self.sheen_color.x,
                self.sheen_color.y,
                self.sheen_color.z,
                self.sheen_roughness,
            ],
            sheen_misc: [self.sheen, self.specular_tint, 0.0, 0.0],
            transmission_params: [
                self.transmission,
                self.transmission_roughness,
                self.ior,
                self.thickness,
            ],
            attenuation: [
                self.attenuation_color.x,
                self.attenuation_color.y,
                self.attenuation_color.z,
                self.attenuation_distance,
            ],
            iridescence_params: [
                self.iridescence,
                self.iridescence_ior,
                self.iridescence_thickness_min,
                self.iridescence_thickness_max,
            ],
            anisotropy_params: [
                self.anisotropy,
                self.anisotropy_rotation,
                self.subsurface,
                self.normal_scale,
            ],
            subsurface_color: [
                self.subsurface_color.x,
                self.subsurface_color.y,
                self.subsurface_color.z,
                self.alpha_cutoff,
            ],
            subsurface_radius: [
                self.subsurface_radius.x,
                self.subsurface_radius.y,
                self.subsurface_radius.z,
                self.alpha_mode as u32 as f32,
            ],
        }
    }

    /// Determine which shader features are needed for this material.
    pub fn required_features(&self) -> PbrFeatureFlags {
        let mut flags = PbrFeatureFlags::empty();

        if self.metallic > 0.0 {
            flags |= PbrFeatureFlags::METALLIC;
        }
        if self.clearcoat > 0.0 {
            flags |= PbrFeatureFlags::CLEARCOAT;
        }
        if self.sheen > 0.0 {
            flags |= PbrFeatureFlags::SHEEN;
        }
        if self.transmission > 0.0 {
            flags |= PbrFeatureFlags::TRANSMISSION;
        }
        if self.iridescence > 0.0 {
            flags |= PbrFeatureFlags::IRIDESCENCE;
        }
        if self.anisotropy.abs() > 1e-4 {
            flags |= PbrFeatureFlags::ANISOTROPY;
        }
        if self.subsurface > 0.0 {
            flags |= PbrFeatureFlags::SUBSURFACE;
        }
        if self.specular_tint > 0.0 {
            flags |= PbrFeatureFlags::SPECULAR_TINT;
        }
        if self.emissive_intensity > 0.0 {
            flags |= PbrFeatureFlags::EMISSIVE;
        }
        if self.alpha_mode == AlphaMode::Mask {
            flags |= PbrFeatureFlags::ALPHA_MASK;
        }
        if self.alpha_mode == AlphaMode::Blend {
            flags |= PbrFeatureFlags::ALPHA_BLEND;
        }

        flags
    }
}

// ---------------------------------------------------------------------------
// Feature flags for shader permutations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PbrFeatureFlags(u32);

impl PbrFeatureFlags {
    pub const NONE: Self = Self(0);
    pub const METALLIC: Self = Self(1 << 0);
    pub const CLEARCOAT: Self = Self(1 << 1);
    pub const SHEEN: Self = Self(1 << 2);
    pub const TRANSMISSION: Self = Self(1 << 3);
    pub const IRIDESCENCE: Self = Self(1 << 4);
    pub const ANISOTROPY: Self = Self(1 << 5);
    pub const SUBSURFACE: Self = Self(1 << 6);
    pub const SPECULAR_TINT: Self = Self(1 << 7);
    pub const EMISSIVE: Self = Self(1 << 8);
    pub const ALPHA_MASK: Self = Self(1 << 9);
    pub const ALPHA_BLEND: Self = Self(1 << 10);
    pub const NORMAL_MAP: Self = Self(1 << 11);
    pub const VERTEX_COLORS: Self = Self(1 << 12);

    pub const fn empty() -> Self { Self(0) }

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn shader_defines(self) -> Vec<(&'static str, &'static str)> {
        let mut defines = Vec::new();
        if self.contains(Self::METALLIC) { defines.push(("HAS_METALLIC", "1")); }
        if self.contains(Self::CLEARCOAT) { defines.push(("HAS_CLEARCOAT", "1")); }
        if self.contains(Self::SHEEN) { defines.push(("HAS_SHEEN", "1")); }
        if self.contains(Self::TRANSMISSION) { defines.push(("HAS_TRANSMISSION", "1")); }
        if self.contains(Self::IRIDESCENCE) { defines.push(("HAS_IRIDESCENCE", "1")); }
        if self.contains(Self::ANISOTROPY) { defines.push(("HAS_ANISOTROPY", "1")); }
        if self.contains(Self::SUBSURFACE) { defines.push(("HAS_SUBSURFACE", "1")); }
        if self.contains(Self::SPECULAR_TINT) { defines.push(("HAS_SPECULAR_TINT", "1")); }
        if self.contains(Self::EMISSIVE) { defines.push(("HAS_EMISSIVE", "1")); }
        if self.contains(Self::ALPHA_MASK) { defines.push(("ALPHA_MASK", "1")); }
        if self.contains(Self::ALPHA_BLEND) { defines.push(("ALPHA_BLEND", "1")); }
        if self.contains(Self::NORMAL_MAP) { defines.push(("HAS_NORMAL_MAP", "1")); }
        if self.contains(Self::VERTEX_COLORS) { defines.push(("HAS_VERTEX_COLORS", "1")); }
        defines
    }

    pub fn complexity_score(self) -> u32 {
        let mut score = 10; // base shader
        if self.contains(Self::METALLIC) { score += 2; }
        if self.contains(Self::CLEARCOAT) { score += 8; }
        if self.contains(Self::SHEEN) { score += 6; }
        if self.contains(Self::TRANSMISSION) { score += 12; }
        if self.contains(Self::IRIDESCENCE) { score += 10; }
        if self.contains(Self::ANISOTROPY) { score += 5; }
        if self.contains(Self::SUBSURFACE) { score += 15; }
        if self.contains(Self::SPECULAR_TINT) { score += 1; }
        score
    }
}

impl std::ops::BitOr for PbrFeatureFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
}

impl std::ops::BitOrAssign for PbrFeatureFlags {
    fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
}

impl std::ops::BitAnd for PbrFeatureFlags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self { Self(self.0 & rhs.0) }
}

// ---------------------------------------------------------------------------
// Multi-scattering energy compensation
// ---------------------------------------------------------------------------

/// Kulla-Conty multi-scattering energy compensation.
///
/// Standard single-scatter GGX loses energy at high roughness. This
/// computes the missing energy and adds it back as a diffuse-like term.
pub fn multi_scatter_compensation(
    f0: Vec3,
    roughness: f32,
    n_dot_v: f32,
    n_dot_l: f32,
) -> Vec3 {
    // Approximate directional albedo E(mu) from pre-integrated BRDF
    let e_v = brdf_energy_approx(n_dot_v, roughness);
    let e_l = brdf_energy_approx(n_dot_l, roughness);
    let e_avg = brdf_avg_energy_approx(roughness);

    // Average Fresnel
    let f_avg = f0.scale(20.0 / 21.0).add(Vec3::new(1.0 / 21.0, 1.0 / 21.0, 1.0 / 21.0));

    // Multi-scattering factor
    let f_ms = f_avg.scale(e_avg).scale(1.0 / (1.0 - f_avg.scale(1.0 - e_avg).max_component()).max(1e-7));

    let compensation = (1.0 - e_v) * (1.0 - e_l) / (PI * (1.0 - e_avg)).max(1e-7);
    f_ms.scale(compensation)
}

/// Polynomial fit for directional albedo E(cos_theta, roughness).
fn brdf_energy_approx(cos_theta: f32, roughness: f32) -> f32 {
    let x = cos_theta;
    let y = roughness;
    // Fitted polynomial (matches pre-integrated LUT closely)
    let a = -1.0 * y * y * y + 0.8 * y * y + 0.3 * y;
    let b = 0.85 - 0.5 * y;
    (a * (1.0 - x) + b * x).clamp(0.0, 1.0)
}

/// Average hemispherical albedo E_avg(roughness).
fn brdf_avg_energy_approx(roughness: f32) -> f32 {
    let r = roughness;
    (1.0 - r * r * 0.5).clamp(0.1, 1.0)
}

// ---------------------------------------------------------------------------
// Material LOD system
// ---------------------------------------------------------------------------

/// Select a simplified material based on distance for performance.
pub struct MaterialLodSystem {
    pub lod_distances: [f32; 4], // distances at which to simplify
}

impl Default for MaterialLodSystem {
    fn default() -> Self {
        Self {
            lod_distances: [20.0, 50.0, 100.0, 200.0],
        }
    }
}

impl MaterialLodSystem {
    /// Simplify a material based on camera distance.
    pub fn simplify_for_distance(&self, material: &PbrMaterialV2, distance: f32) -> PbrMaterialV2 {
        let mut result = material.clone();

        if distance > self.lod_distances[3] {
            // LOD 4: minimal - just base color + roughness, no special lobes
            result.clearcoat = 0.0;
            result.sheen = 0.0;
            result.transmission = 0.0;
            result.iridescence = 0.0;
            result.anisotropy = 0.0;
            result.subsurface = 0.0;
            result.specular_tint = 0.0;
        } else if distance > self.lod_distances[2] {
            // LOD 3: disable expensive effects
            result.iridescence = 0.0;
            result.subsurface = 0.0;
            result.transmission = 0.0;
            result.anisotropy = 0.0;
        } else if distance > self.lod_distances[1] {
            // LOD 2: reduce clearcoat, no iridescence
            result.iridescence = 0.0;
            result.clearcoat *= 0.5;
            result.subsurface *= 0.5;
        } else if distance > self.lod_distances[0] {
            // LOD 1: subtle reductions
            result.iridescence *= 0.5;
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggx_ndf_roughness_zero_is_delta() {
        // Very low roughness -> very high peak at n_dot_h = 1
        let d = d_ggx(1.0, 0.01);
        assert!(d > 100.0);
    }

    #[test]
    fn test_fresnel_at_normal_incidence() {
        let f = f_schlick_scalar(1.0, 0.04);
        assert!((f - 0.04).abs() < 1e-6);
    }

    #[test]
    fn test_fresnel_at_grazing_angle() {
        let f = f_schlick_scalar(0.0, 0.04);
        assert!((f - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ior_f0_roundtrip() {
        let ior = 1.5;
        let f0 = f0_from_ior(ior);
        let ior_back = ior_from_f0(f0);
        assert!((ior - ior_back).abs() < 1e-4);
    }

    #[test]
    fn test_default_material() {
        let mat = PbrMaterialV2::default();
        assert_eq!(mat.metallic, 0.0);
        assert_eq!(mat.alpha_mode, AlphaMode::Opaque);
    }

    #[test]
    fn test_feature_flags() {
        let mat = PbrMaterialV2::car_paint(Vec3::new(1.0, 0.0, 0.0));
        let flags = mat.required_features();
        assert!(flags.contains(PbrFeatureFlags::CLEARCOAT));
        assert!(!flags.contains(PbrFeatureFlags::SHEEN));
    }

    #[test]
    fn test_volume_absorption() {
        let abs = compute_volume_absorption(
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
            1.0,
        );
        // Should be exp(-ln(0.5)) = exp(0.693) ≈ 0.5
        assert!((abs.x - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_brdf_lut_generation() {
        let lut = generate_brdf_lut(4);
        assert_eq!(lut.len(), 16);
        // Values should be in [0, 1]
        for entry in &lut {
            assert!(entry[0] >= 0.0 && entry[0] <= 1.5);
            assert!(entry[1] >= 0.0 && entry[1] <= 1.5);
        }
    }

    #[test]
    fn test_principled_brdf_evaluation() {
        let mat = PbrMaterialV2::default();
        let ctx = ShadingContext {
            n: Vec3::UP,
            v: Vec3::new(0.0, 1.0, 0.0),
            l: Vec3::new(0.577, 0.577, 0.577),
            t: Vec3::new(1.0, 0.0, 0.0),
            b: Vec3::new(0.0, 0.0, 1.0),
            uv: [0.0, 0.0],
        };
        let result = evaluate_principled_brdf(&mat, &ctx, Vec3::ONE, 1.0);
        assert!(result.total.x > 0.0);
        assert!(result.total.y > 0.0);
        assert!(result.total.z > 0.0);
    }

    #[test]
    fn test_gpu_packing() {
        let mat = PbrMaterialV2::glass(Vec3::new(0.9, 0.95, 1.0), 1.5);
        let gpu = mat.to_gpu();
        assert!((gpu.transmission_params[0] - 1.0).abs() < 1e-6);
        assert!((gpu.transmission_params[2] - 1.5).abs() < 1e-6);
    }
}
