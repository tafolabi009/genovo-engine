// engine/render/src/pbr/brdf.rs
//
// Cook-Torrance microfacet BRDF implementation for PBR rendering.
// Contains the Normal Distribution Function (NDF), geometry/masking-shadowing
// function, Fresnel equations, and pre-computed BRDF look-up table generation
// for image-based lighting via split-sum approximation.
//
// All functions operate in linear space and assume unit vectors where stated.

use glam::{Vec2, Vec3};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Normal Distribution Function (NDF)
// ---------------------------------------------------------------------------

/// Normal Distribution Functions describe how microfacet normals are
/// distributed around the macrosurface normal.
pub struct NormalDistributionFunction;

impl NormalDistributionFunction {
    /// GGX / Trowbridge-Reitz NDF.
    ///
    /// D(h) = alpha^2 / (pi * ((n.h)^2 * (alpha^2 - 1) + 1)^2)
    ///
    /// where alpha = roughness^2 (the remapping used by Disney/Unreal).
    ///
    /// # Arguments
    /// - `n_dot_h` — clamped dot product of normal and half-vector.
    /// - `roughness` — perceptual roughness (0..1).
    #[inline]
    pub fn ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h2 = n_dot_h * n_dot_h;
        let denom = n_dot_h2 * (alpha2 - 1.0) + 1.0;
        alpha2 / (PI * denom * denom).max(1e-7)
    }

    /// Anisotropic GGX NDF for materials with directional brushing.
    ///
    /// D(h) = 1 / (pi * alpha_t * alpha_b * ((h.t/alpha_t)^2 + (h.b/alpha_b)^2 + (h.n)^2)^2)
    ///
    /// # Arguments
    /// - `n_dot_h` — dot(N, H).
    /// - `h_dot_t` — dot(H, tangent).
    /// - `h_dot_b` — dot(H, bitangent).
    /// - `alpha_t` — roughness along tangent direction.
    /// - `alpha_b` — roughness along bitangent direction.
    #[inline]
    pub fn ggx_anisotropic(
        n_dot_h: f32,
        h_dot_t: f32,
        h_dot_b: f32,
        alpha_t: f32,
        alpha_b: f32,
    ) -> f32 {
        let at2 = alpha_t * alpha_t;
        let ab2 = alpha_b * alpha_b;
        let term = (h_dot_t * h_dot_t) / at2 + (h_dot_b * h_dot_b) / ab2 + n_dot_h * n_dot_h;
        1.0 / (PI * alpha_t * alpha_b * term * term).max(1e-7)
    }

    /// Beckmann NDF (for comparison / legacy pipelines).
    ///
    /// D(h) = exp(-(tan(theta_h))^2 / alpha^2) / (pi * alpha^2 * cos^4(theta_h))
    #[inline]
    pub fn beckmann(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let n_dot_h2 = n_dot_h * n_dot_h;
        let n_dot_h4 = n_dot_h2 * n_dot_h2;
        let tan2 = (1.0 - n_dot_h2) / n_dot_h2.max(1e-7);
        (-tan2 / alpha2).exp() / (PI * alpha2 * n_dot_h4).max(1e-7)
    }

    /// Blinn-Phong NDF (for legacy / comparison).
    #[inline]
    pub fn blinn_phong(n_dot_h: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        // Shininess exponent approximation from roughness.
        let shininess = 2.0 / alpha2.max(1e-4) - 2.0;
        (shininess + 2.0) / (2.0 * PI) * n_dot_h.max(0.0).powf(shininess)
    }
}

// ---------------------------------------------------------------------------
// Geometry / Masking-Shadowing Function
// ---------------------------------------------------------------------------

/// Geometry functions model self-shadowing and masking of microfacets.
pub struct GeometryFunction;

impl GeometryFunction {
    /// Smith G1 for GGX with the Schlick approximation (UE4 remapping).
    ///
    /// G1(v) = n.v / (n.v * (1 - k) + k)
    ///
    /// where k = (roughness + 1)^2 / 8  (for direct lighting)
    ///   or  k = roughness^2 / 2        (for IBL)
    ///
    /// # Arguments
    /// - `n_dot_v` — clamped dot product of normal and view/light direction.
    /// - `k` — remapped roughness parameter.
    #[inline]
    pub fn schlick_ggx_g1(n_dot_v: f32, k: f32) -> f32 {
        n_dot_v / (n_dot_v * (1.0 - k) + k).max(1e-7)
    }

    /// Smith G2 for GGX with the Schlick approximation (uncorrelated).
    ///
    /// G(N,V,L) = G1(N,V) * G1(N,L)
    ///
    /// Uses the direct-lighting k remapping: k = (roughness+1)^2 / 8.
    ///
    /// # Arguments
    /// - `n_dot_v` — dot(N, V), clamped to [0,1].
    /// - `n_dot_l` — dot(N, L), clamped to [0,1].
    /// - `roughness` — perceptual roughness.
    #[inline]
    pub fn smith_ggx(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let r = roughness + 1.0;
        let k = (r * r) / 8.0;
        Self::schlick_ggx_g1(n_dot_v, k) * Self::schlick_ggx_g1(n_dot_l, k)
    }

    /// Smith G2 for IBL (uses k = roughness^2 / 2).
    #[inline]
    pub fn smith_ggx_ibl(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let k = alpha / 2.0;
        Self::schlick_ggx_g1(n_dot_v, k) * Self::schlick_ggx_g1(n_dot_l, k)
    }

    /// Height-correlated Smith G2 (more physically accurate than uncorrelated).
    ///
    /// From Heitz 2014, "Understanding the Masking-Shadowing Function in
    /// Microfacet-Based BRDFs".
    #[inline]
    pub fn smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let alpha = roughness * roughness;
        let alpha2 = alpha * alpha;
        let lambda_v = n_dot_l * (n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2).sqrt();
        let lambda_l = n_dot_v * (n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2).sqrt();
        0.5 / (lambda_v + lambda_l).max(1e-7)
    }

    /// Anisotropic Smith G2 (correlated) for the anisotropic GGX NDF.
    #[inline]
    pub fn smith_ggx_anisotropic(
        n_dot_v: f32,
        n_dot_l: f32,
        v_dot_t: f32,
        v_dot_b: f32,
        l_dot_t: f32,
        l_dot_b: f32,
        alpha_t: f32,
        alpha_b: f32,
    ) -> f32 {
        let lambda_v = n_dot_l
            * ((v_dot_t * alpha_t).powi(2) + (v_dot_b * alpha_b).powi(2) + n_dot_v.powi(2))
                .sqrt();
        let lambda_l = n_dot_v
            * ((l_dot_t * alpha_t).powi(2) + (l_dot_b * alpha_b).powi(2) + n_dot_l.powi(2))
                .sqrt();
        0.5 / (lambda_v + lambda_l).max(1e-7)
    }

    /// The Kelemen approximation used for clearcoat geometry term.
    /// G = 0.25 / (L.H)^2  (cheaper than full Smith for a secondary lobe).
    #[inline]
    pub fn kelemen(l_dot_h: f32) -> f32 {
        0.25 / (l_dot_h * l_dot_h).max(1e-7)
    }

    /// Neubelt geometry term for sheen (Charlie distribution).
    #[inline]
    pub fn neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
        1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)).max(1e-7)
    }
}

// ---------------------------------------------------------------------------
// Fresnel
// ---------------------------------------------------------------------------

/// Fresnel equations model how reflectance varies with viewing angle.
pub struct FresnelFunction;

impl FresnelFunction {
    /// Schlick approximation of the Fresnel equation.
    ///
    /// F(v,h) = F0 + (1 - F0) * (1 - v.h)^5
    ///
    /// # Arguments
    /// - `cos_theta` — dot(V, H) or dot(N, V), clamped to [0,1].
    /// - `f0` — reflectance at normal incidence.
    #[inline]
    pub fn schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
        let one_minus = (1.0 - cos_theta).max(0.0);
        let factor = one_minus.powi(5);
        f0 + (Vec3::ONE - f0) * factor
    }

    /// Schlick approximation with roughness (used for IBL).
    /// Ensures rough surfaces do not produce brighter reflections at grazing
    /// angles than they would from direct lighting.
    ///
    /// F(v,h) = F0 + (max(1-roughness, F0) - F0) * (1 - v.h)^5
    #[inline]
    pub fn schlick_roughness(cos_theta: f32, f0: Vec3, roughness: f32) -> Vec3 {
        let one_minus = (1.0 - cos_theta).max(0.0);
        let factor = one_minus.powi(5);
        let one_minus_rough = Vec3::splat(1.0 - roughness);
        let max_f0 = Vec3::new(
            f0.x.max(one_minus_rough.x),
            f0.y.max(one_minus_rough.y),
            f0.z.max(one_minus_rough.z),
        );
        f0 + (max_f0 - f0) * factor
    }

    /// Exact Fresnel equation for dielectrics (real IOR only).
    ///
    /// Uses the standard derivation from Snell's law.
    ///
    /// # Arguments
    /// - `cos_theta_i` — cosine of the incidence angle.
    /// - `eta` — ratio of indices of refraction (n1/n2).
    #[inline]
    pub fn dielectric(cos_theta_i: f32, eta: f32) -> f32 {
        let cos_i = cos_theta_i.abs();
        let sin2_t = eta * eta * (1.0 - cos_i * cos_i);

        // Total internal reflection.
        if sin2_t > 1.0 {
            return 1.0;
        }

        let cos_t = (1.0 - sin2_t).sqrt();
        let r_perp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
        let r_para = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
        (r_perp * r_perp + r_para * r_para) * 0.5
    }

    /// Convert reflectance at normal incidence (F0) to an IOR value.
    /// F0 = ((n-1)/(n+1))^2  =>  n = (1+sqrt(F0)) / (1-sqrt(F0))
    #[inline]
    pub fn f0_to_ior(f0: f32) -> f32 {
        let s = f0.max(0.0).sqrt();
        (1.0 + s) / (1.0 - s).max(1e-7)
    }

    /// Convert IOR to reflectance at normal incidence (F0).
    #[inline]
    pub fn ior_to_f0(ior: f32) -> f32 {
        let r = (ior - 1.0) / (ior + 1.0);
        r * r
    }

    /// Compute F0 for a metallic-roughness workflow material.
    /// For dielectrics, F0 is derived from reflectance; for metals, F0
    /// is the albedo colour.
    #[inline]
    pub fn compute_f0(albedo: Vec3, metallic: f32, reflectance: f32) -> Vec3 {
        let dielectric_f0 = Vec3::splat(0.16 * reflectance * reflectance);
        dielectric_f0 * (1.0 - metallic) + albedo * metallic
    }
}

// ---------------------------------------------------------------------------
// Diffuse BRDF
// ---------------------------------------------------------------------------

/// Diffuse reflection models.
pub struct DiffuseBrdf;

impl DiffuseBrdf {
    /// Lambertian diffuse BRDF: f_d = albedo / pi.
    #[inline]
    pub fn lambertian(albedo: Vec3) -> Vec3 {
        albedo * (1.0 / PI)
    }

    /// Disney/Burley diffuse BRDF (roughness-dependent).
    ///
    /// Adds a retro-reflection term at grazing angles that increases with
    /// roughness.
    ///
    /// f_d = (albedo / pi) * (1 + (F_D90 - 1)(1 - NdotL)^5) * (1 + (F_D90 - 1)(1 - NdotV)^5)
    /// where F_D90 = 0.5 + 2 * roughness * (LdotH)^2
    #[inline]
    pub fn burley(albedo: Vec3, roughness: f32, n_dot_v: f32, n_dot_l: f32, l_dot_h: f32) -> Vec3 {
        let f_d90 = 0.5 + 2.0 * roughness * l_dot_h * l_dot_h;
        let light_scatter = 1.0 + (f_d90 - 1.0) * (1.0 - n_dot_l).max(0.0).powi(5);
        let view_scatter = 1.0 + (f_d90 - 1.0) * (1.0 - n_dot_v).max(0.0).powi(5);
        albedo * (light_scatter * view_scatter / PI)
    }

    /// Oren-Nayar diffuse BRDF (models rough diffuse surfaces like clay).
    #[inline]
    pub fn oren_nayar(
        albedo: Vec3,
        roughness: f32,
        n_dot_v: f32,
        n_dot_l: f32,
        l_dot_v: f32,
    ) -> Vec3 {
        let sigma2 = roughness * roughness;
        let a = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
        let b = 0.45 * sigma2 / (sigma2 + 0.09);

        let cos_phi_diff = (l_dot_v - n_dot_l * n_dot_v)
            / ((1.0 - n_dot_l * n_dot_l).max(0.0).sqrt()
                * (1.0 - n_dot_v * n_dot_v).max(0.0).sqrt())
            .max(1e-7);

        let theta_i = n_dot_l.clamp(0.0, 1.0).acos();
        let theta_r = n_dot_v.clamp(0.0, 1.0).acos();
        let alpha_angle = theta_i.max(theta_r);
        let beta_angle = theta_i.min(theta_r);

        let c = alpha_angle.sin() * beta_angle.tan();
        albedo * (1.0 / PI) * (a + b * cos_phi_diff.max(0.0) * c)
    }
}

// ---------------------------------------------------------------------------
// Cook-Torrance Specular BRDF
// ---------------------------------------------------------------------------

/// Complete Cook-Torrance microfacet specular BRDF evaluation.
pub struct CookTorranceBrdf;

impl CookTorranceBrdf {
    /// Evaluate the specular BRDF for a single light.
    ///
    /// f_s = D(h) * G(v,l) * F(v,h) / (4 * (N.V) * (N.L))
    ///
    /// # Arguments
    /// - `n` — macrosurface normal (unit).
    /// - `v` — view direction (unit, pointing away from surface).
    /// - `l` — light direction (unit, pointing away from surface).
    /// - `roughness` — perceptual roughness (0..1).
    /// - `f0` — reflectance at normal incidence.
    ///
    /// # Returns
    /// (specular_colour, fresnel) — the specular BRDF value and the Fresnel
    /// term (useful for energy conservation in the diffuse term).
    pub fn evaluate(n: Vec3, v: Vec3, l: Vec3, roughness: f32, f0: Vec3) -> (Vec3, Vec3) {
        let h = (v + l).normalize_or_zero();

        let n_dot_h = n.dot(h).max(0.0);
        let n_dot_v = n.dot(v).max(1e-4);
        let n_dot_l = n.dot(l).max(0.0);
        let v_dot_h = v.dot(h).max(0.0);

        // NDF
        let d = NormalDistributionFunction::ggx(n_dot_h, roughness);

        // Geometry
        let g = GeometryFunction::smith_ggx_correlated(n_dot_v, n_dot_l, roughness);

        // Fresnel
        let f = FresnelFunction::schlick(v_dot_h, f0);

        // Cook-Torrance denominator is folded into the correlated Smith G2
        // when using `smith_ggx_correlated`, so we only multiply D * G * F.
        // The correlated form already divides by 4*NdotV*NdotL internally
        // via the `0.5 / (lambda_v + lambda_l)` formulation.
        let specular = f * (d * g);

        (specular, f)
    }

    /// Evaluate the specular BRDF using the uncorrelated Smith G2 (the
    /// original Cook-Torrance formulation with the explicit 4*NdotV*NdotL
    /// denominator).
    pub fn evaluate_uncorrelated(
        n: Vec3,
        v: Vec3,
        l: Vec3,
        roughness: f32,
        f0: Vec3,
    ) -> (Vec3, Vec3) {
        let h = (v + l).normalize_or_zero();

        let n_dot_h = n.dot(h).max(0.0);
        let n_dot_v = n.dot(v).max(1e-4);
        let n_dot_l = n.dot(l).max(0.0);
        let v_dot_h = v.dot(h).max(0.0);

        let d = NormalDistributionFunction::ggx(n_dot_h, roughness);
        let g = GeometryFunction::smith_ggx(n_dot_v, n_dot_l, roughness);
        let f = FresnelFunction::schlick(v_dot_h, f0);

        let denom = 4.0 * n_dot_v * n_dot_l;
        let specular = f * (d * g / denom.max(1e-7));

        (specular, f)
    }

    /// Evaluate the full PBR lighting equation for a single directional light.
    ///
    /// Returns the combined diffuse + specular radiance contribution.
    ///
    /// # Arguments
    /// - `n` — macrosurface normal.
    /// - `v` — view direction.
    /// - `l` — light direction.
    /// - `light_color` — light radiance (linear RGB).
    /// - `albedo` — base colour (linear RGB).
    /// - `metallic` — metallic factor.
    /// - `roughness` — perceptual roughness.
    /// - `f0` — reflectance at normal incidence.
    pub fn evaluate_pbr(
        n: Vec3,
        v: Vec3,
        l: Vec3,
        light_color: Vec3,
        albedo: Vec3,
        metallic: f32,
        roughness: f32,
        f0: Vec3,
    ) -> Vec3 {
        let n_dot_l = n.dot(l).max(0.0);
        if n_dot_l <= 0.0 {
            return Vec3::ZERO;
        }

        let (specular, fresnel) = Self::evaluate(n, v, l, roughness, f0);

        // Energy conservation: diffuse is attenuated by the Fresnel term
        // and the metallic factor (metals have no diffuse).
        let k_s = fresnel;
        let k_d = (Vec3::ONE - k_s) * (1.0 - metallic);

        let diffuse = DiffuseBrdf::lambertian(albedo);

        (k_d * diffuse + specular) * light_color * n_dot_l
    }

    /// Full PBR with Burley diffuse instead of Lambertian.
    pub fn evaluate_pbr_burley(
        n: Vec3,
        v: Vec3,
        l: Vec3,
        light_color: Vec3,
        albedo: Vec3,
        metallic: f32,
        roughness: f32,
        f0: Vec3,
    ) -> Vec3 {
        let h = (v + l).normalize_or_zero();
        let n_dot_l = n.dot(l).max(0.0);
        let n_dot_v = n.dot(v).max(1e-4);
        let l_dot_h = l.dot(h).max(0.0);

        if n_dot_l <= 0.0 {
            return Vec3::ZERO;
        }

        let (specular, fresnel) = Self::evaluate(n, v, l, roughness, f0);

        let k_s = fresnel;
        let k_d = (Vec3::ONE - k_s) * (1.0 - metallic);

        let diffuse = DiffuseBrdf::burley(albedo, roughness, n_dot_v, n_dot_l, l_dot_h);

        (k_d * diffuse + specular) * light_color * n_dot_l
    }
}

// ---------------------------------------------------------------------------
// IBL (Image-Based Lighting)
// ---------------------------------------------------------------------------

/// Computes the specular IBL contribution using the split-sum approximation.
///
/// The rendering equation for environment lighting is split into two parts:
///   L_spec ≈ prefilteredColor * (F0 * brdf.x + brdf.y)
///
/// where `brdf` is looked up from a pre-computed BRDF LUT.
///
/// # Arguments
/// - `prefiltered_color` — the pre-filtered environment map sample at the
///   appropriate roughness mip level.
/// - `f0` — reflectance at normal incidence.
/// - `roughness` — perceptual roughness.
/// - `n_dot_v` — dot(N, V).
pub fn ibl_specular(prefiltered_color: Vec3, f0: Vec3, roughness: f32, n_dot_v: f32) -> Vec3 {
    let brdf = integrate_brdf(n_dot_v, roughness);
    let fresnel = FresnelFunction::schlick_roughness(n_dot_v, f0, roughness);
    prefiltered_color * (fresnel * brdf.x + Vec3::splat(brdf.y))
}

/// Computes the diffuse IBL contribution.
///
/// # Arguments
/// - `irradiance` — the irradiance map sample for the normal direction.
/// - `albedo` — base colour.
/// - `metallic` — metallic factor.
/// - `f0` — reflectance at normal incidence.
/// - `roughness` — perceptual roughness.
/// - `n_dot_v` — dot(N, V).
pub fn ibl_diffuse(
    irradiance: Vec3,
    albedo: Vec3,
    metallic: f32,
    f0: Vec3,
    roughness: f32,
    n_dot_v: f32,
) -> Vec3 {
    let fresnel = FresnelFunction::schlick_roughness(n_dot_v, f0, roughness);
    let k_d = (Vec3::ONE - fresnel) * (1.0 - metallic);
    k_d * irradiance * albedo
}

// ---------------------------------------------------------------------------
// BRDF LUT Generation (split-sum)
// ---------------------------------------------------------------------------

/// Pre-computed BRDF look-up table for the split-sum IBL approximation.
///
/// The LUT is a 2D texture where:
/// - U axis = NdotV (0..1)
/// - V axis = roughness (0..1)
/// - R channel = scale factor for F0
/// - G channel = bias (added to result)
pub struct BrdfLut {
    /// Width of the LUT texture (typically 512).
    pub width: u32,
    /// Height of the LUT texture (typically 512).
    pub height: u32,
    /// Pixel data in RG16Float format (2 f32 per pixel stored as [f32; 2]).
    pub data: Vec<[f32; 2]>,
}

impl BrdfLut {
    /// Generate the BRDF integration LUT.
    ///
    /// This performs importance-sampled integration of the split-sum
    /// approximation for each (NdotV, roughness) pair.
    ///
    /// # Arguments
    /// - `width` — texture width (typically 512).
    /// - `height` — texture height (typically 512).
    /// - `sample_count` — number of importance samples (typically 1024).
    pub fn generate(width: u32, height: u32, sample_count: u32) -> Self {
        let mut data = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            let roughness = (y as f32 + 0.5) / height as f32;
            for x in 0..width {
                let n_dot_v = (x as f32 + 0.5) / width as f32;
                let n_dot_v = n_dot_v.max(1e-4); // avoid division by zero

                let result = integrate_brdf_sampled(n_dot_v, roughness, sample_count);
                data.push([result.x, result.y]);
            }
        }

        Self {
            width,
            height,
            data,
        }
    }

    /// Generate with default settings (512x512, 1024 samples).
    pub fn generate_default() -> Self {
        Self::generate(512, 512, 1024)
    }

    /// Look up the BRDF integration result for a given (NdotV, roughness).
    pub fn sample(&self, n_dot_v: f32, roughness: f32) -> Vec2 {
        let u = (n_dot_v * self.width as f32).min(self.width as f32 - 1.0) as usize;
        let v = (roughness * self.height as f32).min(self.height as f32 - 1.0) as usize;
        let idx = v * self.width as usize + u;
        let val = self.data[idx.min(self.data.len() - 1)];
        Vec2::new(val[0], val[1])
    }

    /// Convert to a byte buffer suitable for GPU upload (RG32Float).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.data.len() * 8);
        for pixel in &self.data {
            bytes.extend_from_slice(bytemuck::bytes_of(&pixel[0]));
            bytes.extend_from_slice(bytemuck::bytes_of(&pixel[1]));
        }
        bytes
    }
}

// ---------------------------------------------------------------------------
// Importance Sampling
// ---------------------------------------------------------------------------

/// Hammersley quasi-random sequence point.
///
/// Generates the `i`-th point of an N-point Hammersley sequence in [0,1]^2.
/// Uses radical inverse (Van der Corput sequence) for the second dimension.
///
/// This produces a well-distributed low-discrepancy sequence suitable for
/// Monte Carlo integration.
#[inline]
pub fn hammersley(i: u32, n: u32) -> Vec2 {
    Vec2::new(i as f32 / n as f32, radical_inverse_vdc(i))
}

/// Radical inverse using base-2 (Van der Corput sequence).
///
/// Reverses the bits of `n` and maps the result to [0,1).
#[inline]
pub fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.328_306_4e-10 // 0x100000000 as f32
}

/// Generate an importance-sampled direction for GGX NDF.
///
/// Given a 2D random sample `xi` in [0,1]^2, produces a half-vector `H` in
/// tangent space that is distributed according to the GGX NDF for the given
/// roughness.
///
/// The tangent space has N = (0, 0, 1).
///
/// # Arguments
/// - `xi` — 2D random sample from Hammersley.
/// - `roughness` — perceptual roughness.
///
/// # Returns
/// Half-vector in tangent space.
pub fn importance_sample_ggx(xi: Vec2, roughness: f32) -> Vec3 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;

    // Spherical coordinates from the GGX distribution.
    let phi = 2.0 * PI * xi.x;
    let cos_theta = ((1.0 - xi.y) / (1.0 + (alpha2 - 1.0) * xi.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

    // Convert to Cartesian (tangent space, Z-up).
    Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta)
}

/// Importance-sample a direction for the Charlie sheen distribution.
pub fn importance_sample_charlie(xi: Vec2, roughness: f32) -> Vec3 {
    let alpha = roughness * roughness;
    let sin_theta = (2.0 * xi.y / (1.0 + xi.y)).sqrt().powf(alpha / (2.0 * alpha + 1.0));
    let cos_theta = (1.0 - sin_theta * sin_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * xi.x;

    Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta)
}

/// Integrate the BRDF for a given (NdotV, roughness) using importance sampling.
///
/// This is the core of the BRDF LUT generation. It evaluates:
///   integral of f(v,l) * cos(theta_l) dl
///
/// over the hemisphere, split into a scale factor for F0 and a bias term.
pub fn integrate_brdf(n_dot_v: f32, roughness: f32) -> Vec2 {
    integrate_brdf_sampled(n_dot_v, roughness, 1024)
}

/// Integrate the BRDF with a specified number of samples.
pub fn integrate_brdf_sampled(n_dot_v: f32, roughness: f32, sample_count: u32) -> Vec2 {
    // View vector in tangent space (N = Z-up).
    let v = Vec3::new(
        (1.0 - n_dot_v * n_dot_v).max(0.0).sqrt(), // sin(theta)
        0.0,
        n_dot_v, // cos(theta)
    );

    let _n = Vec3::Z;
    let mut a = 0.0_f32; // scale for F0
    let mut b = 0.0_f32; // bias

    for i in 0..sample_count {
        let xi = hammersley(i, sample_count);
        let h = importance_sample_ggx(xi, roughness);
        let l = (2.0 * v.dot(h) * h - v).normalize_or_zero();

        let n_dot_l = l.z.max(0.0);
        let n_dot_h = h.z.max(0.0);
        let v_dot_h = v.dot(h).max(0.0);

        if n_dot_l > 0.0 {
            let g = GeometryFunction::smith_ggx_ibl(n_dot_v, n_dot_l, roughness);
            // G_Vis = G * VdotH / (NdotH * NdotV)
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v).max(1e-7);
            let fc = (1.0 - v_dot_h).max(0.0).powi(5);

            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }

    let inv = 1.0 / sample_count as f32;
    Vec2::new(a * inv, b * inv)
}

// ---------------------------------------------------------------------------
// Prefiltered environment map helpers
// ---------------------------------------------------------------------------

/// Compute the mip level for pre-filtered environment map sampling.
///
/// Rougher surfaces sample higher (blurrier) mip levels.
///
/// # Arguments
/// - `roughness` — perceptual roughness.
/// - `max_mip` — the highest mip level in the environment map.
#[inline]
pub fn prefilter_mip_level(roughness: f32, max_mip: f32) -> f32 {
    roughness * max_mip
}

/// Compute the reflection direction for specular IBL sampling.
///
/// R = 2 * (N . V) * N - V
#[inline]
pub fn reflect_direction(n: Vec3, v: Vec3) -> Vec3 {
    2.0 * n.dot(v) * n - v
}

/// Convert a tangent-space vector to world space given a normal.
///
/// Builds a TBN matrix from the normal and transforms the tangent-space
/// vector into world space.
pub fn tangent_to_world(tangent_vec: Vec3, normal: Vec3) -> Vec3 {
    // Build an orthonormal basis around the normal.
    let up = if normal.y.abs() < 0.999 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent = up.cross(normal).normalize_or_zero();
    let bitangent = normal.cross(tangent);

    tangent * tangent_vec.x + bitangent * tangent_vec.y + normal * tangent_vec.z
}

/// Pre-filter an environment map direction for a given roughness using
/// importance sampling. Returns the accumulated radiance direction and weight.
///
/// This is a reference implementation for CPU-side prefiltering. In production
/// the prefiltering would run as a compute shader.
///
/// # Arguments
/// - `normal` — the normal direction for this texel.
/// - `roughness` — roughness for this mip level.
/// - `sample_count` — number of importance samples.
/// - `sample_env` — closure that samples the environment map given a direction.
pub fn prefilter_env_map(
    normal: Vec3,
    roughness: f32,
    sample_count: u32,
    sample_env: impl Fn(Vec3) -> Vec3,
) -> Vec3 {
    let n = normal;
    let v = normal; // assume V == N for prefiltering
    let mut total_color = Vec3::ZERO;
    let mut total_weight = 0.0_f32;

    for i in 0..sample_count {
        let xi = hammersley(i, sample_count);
        let h = importance_sample_ggx(xi, roughness);
        let h_world = tangent_to_world(h, n);
        let l = (2.0 * v.dot(h_world) * h_world - v).normalize_or_zero();

        let n_dot_l = n.dot(l).max(0.0);
        if n_dot_l > 0.0 {
            let color = sample_env(l);
            total_color += color * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    if total_weight > 0.0 {
        total_color / total_weight
    } else {
        Vec3::ZERO
    }
}

// ---------------------------------------------------------------------------
// Multi-scattering energy compensation
// ---------------------------------------------------------------------------

/// Compute the multi-scattering energy compensation factor.
///
/// Single-scattering BRDFs lose energy at high roughness because they only
/// model one bounce. This factor compensates by adding back the lost energy.
///
/// Based on "A Multiple-Scattering Microfacet Model for Real-Time Image Based
/// Lighting" (Fdez-Agüera 2019).
///
/// # Arguments
/// - `f0` — reflectance at normal incidence.
/// - `brdf_lut` — (scale, bias) from the BRDF LUT.
pub fn multiscatter_compensation(f0: Vec3, brdf_lut: Vec2) -> (Vec3, Vec3) {
    // Single-scattering energy fraction.
    let e_ss = f0 * brdf_lut.x + Vec3::splat(brdf_lut.y);
    // Average Fresnel.
    let f_avg = f0 + (Vec3::ONE - f0) / 21.0;
    // Multi-scattering scale.
    let f_ms = f_avg * e_ss / (Vec3::ONE - f_avg * (Vec3::ONE - e_ss));
    // Total energy scale for specular.
    let energy_scale = Vec3::ONE + f_ms * (Vec3::ONE / e_ss - Vec3::ONE);

    (energy_scale, f_ms)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    #[test]
    fn ggx_ndf_is_normalized_at_rough_1() {
        // For roughness = 1 and NdotH = 1, D should be 1/pi.
        let d = NormalDistributionFunction::ggx(1.0, 1.0);
        assert!((d - 1.0 / PI).abs() < 0.01);
    }

    #[test]
    fn ggx_ndf_peak_at_n_dot_h_1() {
        // D(1.0) should be the peak for any roughness.
        let d_peak = NormalDistributionFunction::ggx(1.0, 0.3);
        let d_off = NormalDistributionFunction::ggx(0.5, 0.3);
        assert!(d_peak > d_off);
    }

    #[test]
    fn fresnel_at_normal_incidence_equals_f0() {
        let f0 = Vec3::splat(0.04);
        let f = FresnelFunction::schlick(1.0, f0);
        assert!((f.x - f0.x).abs() < EPSILON);
    }

    #[test]
    fn fresnel_at_grazing_is_white() {
        let f0 = Vec3::splat(0.04);
        let f = FresnelFunction::schlick(0.0, f0);
        assert!((f.x - 1.0).abs() < EPSILON);
    }

    #[test]
    fn smith_geometry_one_at_smooth() {
        // For very low roughness and NdotV=NdotL=1, G should be ~1.
        let g = GeometryFunction::smith_ggx(1.0, 1.0, 0.01);
        assert!((g - 1.0).abs() < 0.05);
    }

    #[test]
    fn hammersley_first_point() {
        let p = hammersley(0, 16);
        assert!((p.x - 0.0).abs() < EPSILON);
        assert!((p.y - 0.0).abs() < EPSILON);
    }

    #[test]
    fn brdf_lut_integration_range() {
        let result = integrate_brdf(0.5, 0.5);
        assert!(result.x >= 0.0 && result.x <= 1.0);
        assert!(result.y >= 0.0 && result.y <= 1.0);
    }

    #[test]
    fn ior_roundtrip() {
        let ior = 1.5;
        let f0 = FresnelFunction::ior_to_f0(ior);
        let ior2 = FresnelFunction::f0_to_ior(f0);
        assert!((ior - ior2).abs() < EPSILON);
    }

    #[test]
    fn lambertian_energy_conservation() {
        // Lambertian diffuse for white albedo should integrate to 1 over the
        // hemisphere: integral of (1/pi) * cos(theta) dw = 1.
        let albedo = Vec3::ONE;
        let fd = DiffuseBrdf::lambertian(albedo);
        // fd = 1/pi ≈ 0.318
        assert!((fd.x - 1.0 / PI).abs() < EPSILON);
    }

    #[test]
    fn cook_torrance_non_negative() {
        let n = Vec3::Y;
        let v = Vec3::new(0.0, 1.0, 0.0);
        let l = Vec3::new(0.5, 0.5, 0.0).normalize();
        let f0 = Vec3::splat(0.04);
        let (spec, _) = CookTorranceBrdf::evaluate(n, v, l, 0.5, f0);
        assert!(spec.x >= 0.0);
        assert!(spec.y >= 0.0);
        assert!(spec.z >= 0.0);
    }

    #[test]
    fn multiscatter_compensation_increases_energy() {
        let f0 = Vec3::splat(0.04);
        let brdf_lut = Vec2::new(0.5, 0.1);
        let (energy_scale, _) = multiscatter_compensation(f0, brdf_lut);
        assert!(energy_scale.x >= 1.0);
    }
}
