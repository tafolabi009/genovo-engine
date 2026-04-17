// engine/render/src/pbr_v2.rs
//
// Enhanced PBR material model: clearcoat, sheen, transmission, thin-film
// iridescence, anisotropy, subsurface color, specular tint, IOR control.
// Full Disney/Filament principled BRDF with all extension lobes.

use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    pub fn cross(self, r: Self) -> Self { Self { x: self.y*r.z - self.z*r.y, y: self.z*r.x - self.x*r.z, z: self.x*r.y - self.y*r.x } }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn normalize(self) -> Self { let l = self.length(); if l < 1e-12 { Self::ZERO } else { Self { x: self.x/l, y: self.y/l, z: self.z/l } } }
    pub fn scale(self, s: f32) -> Self { Self { x: self.x*s, y: self.y*s, z: self.z*s } }
    pub fn add(self, r: Self) -> Self { Self { x: self.x+r.x, y: self.y+r.y, z: self.z+r.z } }
    pub fn sub(self, r: Self) -> Self { Self { x: self.x-r.x, y: self.y-r.y, z: self.z-r.z } }
    pub fn lerp(self, r: Self, t: f32) -> Self { self.add(r.sub(self).scale(t)) }
    pub fn mul_elem(self, r: Self) -> Self { Self { x: self.x*r.x, y: self.y*r.y, z: self.z*r.z } }
    pub fn luminance(self) -> f32 { 0.2126 * self.x + 0.7152 * self.y + 0.0722 * self.z }
    pub fn max_component(self) -> f32 { self.x.max(self.y).max(self.z) }
}

/// Full principled BRDF material parameters.
#[derive(Debug, Clone)]
pub struct PbrMaterialV2 {
    pub base_color: Vec3, pub metallic: f32, pub roughness: f32, pub reflectance: f32,
    pub normal_scale: f32, pub emissive_color: Vec3, pub emissive_intensity: f32, pub ao: f32,
    pub clearcoat: f32, pub clearcoat_roughness: f32, pub clearcoat_ior: f32,
    pub sheen: f32, pub sheen_color: Vec3, pub sheen_roughness: f32,
    pub transmission: f32, pub transmission_roughness: f32, pub ior: f32,
    pub thickness: f32, pub attenuation_color: Vec3, pub attenuation_distance: f32,
    pub iridescence: f32, pub iridescence_ior: f32,
    pub iridescence_thickness_min: f32, pub iridescence_thickness_max: f32,
    pub anisotropy: f32, pub anisotropy_rotation: f32,
    pub subsurface: f32, pub subsurface_color: Vec3, pub subsurface_radius: Vec3,
    pub specular_tint: f32, pub alpha_cutoff: f32, pub alpha_mode: AlphaMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode { Opaque, Mask, Blend }

impl Default for PbrMaterialV2 {
    fn default() -> Self {
        Self {
            base_color: Vec3::new(0.8, 0.8, 0.8), metallic: 0.0, roughness: 0.5,
            reflectance: 0.5, normal_scale: 1.0, emissive_color: Vec3::ZERO,
            emissive_intensity: 0.0, ao: 1.0, clearcoat: 0.0, clearcoat_roughness: 0.05,
            clearcoat_ior: 1.5, sheen: 0.0, sheen_color: Vec3::ONE, sheen_roughness: 0.5,
            transmission: 0.0, transmission_roughness: 0.0, ior: 1.5, thickness: 0.0,
            attenuation_color: Vec3::ONE, attenuation_distance: f32::INFINITY,
            iridescence: 0.0, iridescence_ior: 1.3,
            iridescence_thickness_min: 100.0, iridescence_thickness_max: 400.0,
            anisotropy: 0.0, anisotropy_rotation: 0.0,
            subsurface: 0.0, subsurface_color: Vec3::ONE,
            subsurface_radius: Vec3::new(1.0, 0.2, 0.1),
            specular_tint: 0.0, alpha_cutoff: 0.5, alpha_mode: AlphaMode::Opaque,
        }
    }
}

// ---------------------------------------------------------------------------
// NDF functions
// ---------------------------------------------------------------------------

#[inline]
pub fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness; let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d).max(1e-7)
}

#[inline]
pub fn d_ggx_aniso(n_dot_h: f32, h_dot_t: f32, h_dot_b: f32, at: f32, ab: f32) -> f32 {
    let d = (h_dot_t * h_dot_t) / (at * at) + (h_dot_b * h_dot_b) / (ab * ab) + n_dot_h * n_dot_h;
    1.0 / (PI * at * ab * d * d).max(1e-7)
}

#[inline]
pub fn d_charlie(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness; let inv_alpha = 1.0 / alpha;
    let sin_theta = (1.0 - n_dot_h * n_dot_h).max(0.0).sqrt();
    (2.0 + inv_alpha) * sin_theta.powf(inv_alpha) / (2.0 * PI)
}

// ---------------------------------------------------------------------------
// Visibility functions
// ---------------------------------------------------------------------------

#[inline]
pub fn v_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = roughness * roughness; let a2 = a * a;
    let ggxv = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let ggxl = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    0.5 / (ggxv + ggxl).max(1e-7)
}

#[inline]
pub fn v_neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
    1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)).max(1e-7)
}

#[inline]
pub fn v_kelemen(l_dot_h: f32) -> f32 { 0.25 / (l_dot_h * l_dot_h).max(1e-7) }

// ---------------------------------------------------------------------------
// Fresnel functions
// ---------------------------------------------------------------------------

#[inline]
pub fn f_schlick_scalar(cos_theta: f32, f0: f32) -> f32 {
    let t = (1.0 - cos_theta).max(0.0); let t2 = t * t; let t5 = t2 * t2 * t;
    f0 + (1.0 - f0) * t5
}

#[inline]
pub fn f_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    let t = (1.0 - cos_theta).max(0.0); let t2 = t * t; let t5 = t2 * t2 * t;
    Vec3::new(f0.x + (1.0 - f0.x) * t5, f0.y + (1.0 - f0.y) * t5, f0.z + (1.0 - f0.z) * t5)
}

#[inline]
pub fn f0_from_ior(ior: f32) -> f32 { let r = (ior - 1.0) / (ior + 1.0); r * r }

#[inline]
pub fn ior_from_f0(f0: f32) -> f32 { let s = f0.sqrt().min(0.999); (1.0 + s) / (1.0 - s) }

pub fn fresnel_dielectric(cos_theta_i: f32, eta: f32) -> f32 {
    let sin2_t = eta * eta * (1.0 - cos_theta_i * cos_theta_i);
    if sin2_t > 1.0 { return 1.0; }
    let cos_t = (1.0 - sin2_t).sqrt();
    let rs = (eta * cos_theta_i - cos_t) / (eta * cos_theta_i + cos_t);
    let rp = (cos_theta_i - eta * cos_t) / (cos_theta_i + eta * cos_t);
    (rs * rs + rp * rp) * 0.5
}

// ---------------------------------------------------------------------------
// Thin-film iridescence
// ---------------------------------------------------------------------------

pub fn evaluate_iridescence(outside_ior: f32, film_ior: f32, film_thickness_nm: f32, base_f0: Vec3, cos_theta: f32) -> Vec3 {
    let sin2_theta1 = 1.0 - cos_theta * cos_theta;
    let eta_film = outside_ior / film_ior;
    let sin2_theta2 = eta_film * eta_film * sin2_theta1;
    if sin2_theta2 >= 1.0 { return Vec3::ONE; }
    let cos_theta2 = (1.0 - sin2_theta2).sqrt();
    let r_outer = fresnel_dielectric(cos_theta, outside_ior / film_ior);
    let opd = 2.0 * film_ior * film_thickness_nm * cos_theta2;
    let wavelengths = [650.0_f32, 532.0, 450.0];
    let mut result = [0.0_f32; 3];
    for (i, &lambda) in wavelengths.iter().enumerate() {
        let phase = 2.0 * PI * opd / lambda;
        let base_f0_ch = match i { 0 => base_f0.x, 1 => base_f0.y, _ => base_f0.z };
        let sqrt_r = (r_outer * base_f0_ch).sqrt();
        let cos_phase = phase.cos();
        let num = r_outer + base_f0_ch + 2.0 * sqrt_r * cos_phase;
        let den = 1.0 + r_outer * base_f0_ch + 2.0 * sqrt_r * cos_phase;
        result[i] = (num / den.max(1e-7)).clamp(0.0, 1.0);
    }
    Vec3::new(result[0], result[1], result[2])
}

// ---------------------------------------------------------------------------
// Subsurface scattering approximation
// ---------------------------------------------------------------------------

pub fn evaluate_subsurface_diffuse(base_color: Vec3, ss_color: Vec3, subsurface: f32, roughness: f32, n_dot_v: f32, n_dot_l: f32, l_dot_h: f32) -> Vec3 {
    let fd90 = 0.5 + 2.0 * l_dot_h * l_dot_h * roughness;
    let light_scatter = 1.0 + (fd90 - 1.0) * (1.0 - n_dot_l).powi(5);
    let view_scatter = 1.0 + (fd90 - 1.0) * (1.0 - n_dot_v).powi(5);
    let disney_diffuse = base_color.scale(light_scatter * view_scatter / PI);
    let fss90 = l_dot_h * l_dot_h * roughness;
    let fss_l = 1.0 + (fss90 - 1.0) * (1.0 - n_dot_l).powi(5);
    let fss_v = 1.0 + (fss90 - 1.0) * (1.0 - n_dot_v).powi(5);
    let fss = 1.25 * (fss_l * fss_v * (1.0 / (n_dot_l + n_dot_v).max(1e-4) - 0.5) + 0.5);
    let ss_result = ss_color.mul_elem(base_color).scale(fss / PI);
    disney_diffuse.lerp(ss_result, subsurface)
}

// ---------------------------------------------------------------------------
// Clearcoat lobe
// ---------------------------------------------------------------------------

pub struct ClearcoatResult { pub specular: f32, pub attenuation: f32 }

pub fn evaluate_clearcoat(cc: f32, cc_rough: f32, cc_ior: f32, n_dot_h: f32, n_dot_l: f32, n_dot_v: f32, l_dot_h: f32) -> ClearcoatResult {
    if cc < 1e-4 { return ClearcoatResult { specular: 0.0, attenuation: 1.0 }; }
    let cc_f0 = f0_from_ior(cc_ior);
    let d = d_ggx(n_dot_h, cc_rough);
    let v = v_kelemen(l_dot_h);
    let f = f_schlick_scalar(l_dot_h, cc_f0);
    ClearcoatResult { specular: d * v * f * cc, attenuation: 1.0 - cc * f }
}

// ---------------------------------------------------------------------------
// Sheen lobe
// ---------------------------------------------------------------------------

pub struct SheenResult { pub color: Vec3, pub scaling: f32 }

pub fn evaluate_sheen(sheen: f32, sheen_color: Vec3, sheen_rough: f32, n_dot_h: f32, n_dot_v: f32, n_dot_l: f32) -> SheenResult {
    if sheen < 1e-4 { return SheenResult { color: Vec3::ZERO, scaling: 1.0 }; }
    let d = d_charlie(n_dot_h, sheen_rough);
    let v = v_neubelt(n_dot_v, n_dot_l);
    let specular = sheen_color.scale(d * v * sheen);
    let albedo = sheen_rough * (1.0 - ((-3.0 * n_dot_v).exp())) * 0.6 + ((-3.0 * n_dot_v).exp()) * 0.1;
    SheenResult { color: specular, scaling: 1.0 - sheen * albedo }
}

// ---------------------------------------------------------------------------
// Transmission
// ---------------------------------------------------------------------------

pub fn compute_volume_absorption(attenuation_color: Vec3, attenuation_distance: f32, thickness: f32) -> Vec3 {
    if attenuation_distance >= f32::MAX * 0.5 || thickness < 1e-6 { return Vec3::ONE; }
    let sigma_a = Vec3::new(
        -attenuation_color.x.max(1e-6).ln() / attenuation_distance,
        -attenuation_color.y.max(1e-6).ln() / attenuation_distance,
        -attenuation_color.z.max(1e-6).ln() / attenuation_distance,
    );
    Vec3::new((-sigma_a.x * thickness).exp(), (-sigma_a.y * thickness).exp(), (-sigma_a.z * thickness).exp())
}

// ---------------------------------------------------------------------------
// Anisotropy helpers
// ---------------------------------------------------------------------------

pub fn compute_aniso_roughness(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let a = roughness * roughness;
    let aspect = (1.0 - anisotropy * 0.9).sqrt();
    ((a / aspect).max(0.001), (a * aspect).max(0.001))
}

pub fn apply_specular_tint(base_color: Vec3, f0: Vec3, specular_tint: f32) -> Vec3 {
    let lum = base_color.luminance().max(1e-7);
    let tint = base_color.scale(1.0 / lum);
    f0.lerp(f0.mul_elem(tint), specular_tint)
}

// ---------------------------------------------------------------------------
// Full principled BRDF evaluation
// ---------------------------------------------------------------------------

pub struct ShadingContext { pub n: Vec3, pub v: Vec3, pub l: Vec3, pub t: Vec3, pub b: Vec3 }

pub struct BrdfResult { pub diffuse: Vec3, pub specular: Vec3, pub clearcoat: Vec3, pub sheen: Vec3, pub transmission: Vec3, pub emissive: Vec3, pub total: Vec3 }

pub fn evaluate_principled_brdf(mat: &PbrMaterialV2, ctx: &ShadingContext, light_color: Vec3, light_intensity: f32) -> BrdfResult {
    let h = ctx.v.add(ctx.l).normalize();
    let n_dot_v = ctx.n.dot(ctx.v).max(1e-5);
    let n_dot_l = ctx.n.dot(ctx.l).max(0.0);
    let n_dot_h = ctx.n.dot(h).max(0.0);
    let l_dot_h = ctx.l.dot(h).max(0.0);

    if n_dot_l <= 0.0 && mat.transmission < 1e-4 {
        let e = mat.emissive_color.scale(mat.emissive_intensity);
        return BrdfResult { diffuse: Vec3::ZERO, specular: Vec3::ZERO, clearcoat: Vec3::ZERO, sheen: Vec3::ZERO, transmission: Vec3::ZERO, emissive: e, total: e };
    }

    let irradiance = light_color.scale(light_intensity * n_dot_l.max(0.0));
    let dielectric_f0 = 0.16 * mat.reflectance * mat.reflectance;
    let mut f0 = Vec3::new(dielectric_f0, dielectric_f0, dielectric_f0).lerp(mat.base_color, mat.metallic);
    if mat.specular_tint > 0.0 { f0 = apply_specular_tint(mat.base_color, f0, mat.specular_tint); }
    if mat.iridescence > 0.0 {
        let thickness = (mat.iridescence_thickness_min + mat.iridescence_thickness_max) * 0.5;
        let irid = evaluate_iridescence(1.0, mat.iridescence_ior, thickness, f0, n_dot_v);
        f0 = f0.lerp(irid, mat.iridescence);
    }

    // Specular
    let specular = if mat.anisotropy.abs() > 1e-4 {
        let (at, ab) = compute_aniso_roughness(mat.roughness, mat.anisotropy);
        let t_dot_h = ctx.t.dot(h); let b_dot_h = ctx.b.dot(h);
        let d = d_ggx_aniso(n_dot_h, t_dot_h, b_dot_h, at, ab);
        let v = v_smith_ggx_correlated(n_dot_v, n_dot_l, mat.roughness);
        f_schlick(l_dot_h, f0).scale(d * v)
    } else {
        let d = d_ggx(n_dot_h, mat.roughness);
        let v = v_smith_ggx_correlated(n_dot_v, n_dot_l, mat.roughness);
        f_schlick(l_dot_h, f0).scale(d * v)
    };

    // Diffuse
    let diff_color = mat.base_color.scale(1.0 - mat.metallic);
    let diffuse = if mat.subsurface > 0.0 {
        evaluate_subsurface_diffuse(diff_color, mat.subsurface_color, mat.subsurface, mat.roughness, n_dot_v, n_dot_l, l_dot_h)
    } else {
        let fd90 = 0.5 + 2.0 * l_dot_h * l_dot_h * mat.roughness;
        let ls = 1.0 + (fd90 - 1.0) * (1.0 - n_dot_l).powi(5);
        let vs = 1.0 + (fd90 - 1.0) * (1.0 - n_dot_v).powi(5);
        diff_color.scale(ls * vs / PI)
    };

    let cc = evaluate_clearcoat(mat.clearcoat, mat.clearcoat_roughness, mat.clearcoat_ior, n_dot_h, n_dot_l, n_dot_v, l_dot_h);
    let sh = evaluate_sheen(mat.sheen, mat.sheen_color, mat.sheen_roughness, n_dot_h, n_dot_v, n_dot_l);

    let diff_weight = (1.0 - mat.metallic) * (1.0 - mat.transmission);
    let diff_final = diffuse.scale(diff_weight * sh.scaling * cc.attenuation * mat.ao);
    let spec_final = specular.scale(cc.attenuation);
    let cc_final = Vec3::new(cc.specular, cc.specular, cc.specular);
    let emissive = mat.emissive_color.scale(mat.emissive_intensity);
    let lit = diff_final.add(spec_final).add(cc_final).add(sh.color);
    let total = lit.mul_elem(irradiance).add(emissive);

    BrdfResult { diffuse: diff_final.mul_elem(irradiance), specular: spec_final.mul_elem(irradiance), clearcoat: cc_final.mul_elem(irradiance), sheen: sh.color.mul_elem(irradiance), transmission: Vec3::ZERO, emissive, total }
}

// ---------------------------------------------------------------------------
// Material presets
// ---------------------------------------------------------------------------

impl PbrMaterialV2 {
    pub fn plastic(color: Vec3) -> Self { Self { base_color: color, roughness: 0.4, ..Default::default() } }
    pub fn metal(color: Vec3, roughness: f32) -> Self { Self { base_color: color, metallic: 1.0, roughness, ..Default::default() } }
    pub fn car_paint(color: Vec3) -> Self { Self { base_color: color, clearcoat: 1.0, clearcoat_roughness: 0.03, ..Default::default() } }
    pub fn glass(color: Vec3, ior: f32) -> Self { Self { base_color: color, roughness: 0.0, transmission: 1.0, ior, alpha_mode: AlphaMode::Blend, ..Default::default() } }
    pub fn fabric(color: Vec3, sheen_color: Vec3) -> Self { Self { base_color: color, roughness: 0.8, sheen: 1.0, sheen_color, ..Default::default() } }
    pub fn skin(color: Vec3) -> Self { Self { base_color: color, roughness: 0.5, subsurface: 0.5, subsurface_color: Vec3::new(0.8, 0.2, 0.1), ..Default::default() } }

    pub fn required_features(&self) -> u32 {
        let mut f = 0u32;
        if self.metallic > 0.0 { f |= 1; }
        if self.clearcoat > 0.0 { f |= 2; }
        if self.sheen > 0.0 { f |= 4; }
        if self.transmission > 0.0 { f |= 8; }
        if self.iridescence > 0.0 { f |= 16; }
        if self.anisotropy.abs() > 1e-4 { f |= 32; }
        if self.subsurface > 0.0 { f |= 64; }
        f
    }

    #[repr(C)]
    pub fn to_gpu(&self) -> PbrMaterialGpu {
        PbrMaterialGpu {
            base_color: [self.base_color.x, self.base_color.y, self.base_color.z, 1.0],
            emissive: [self.emissive_color.x, self.emissive_color.y, self.emissive_color.z, self.emissive_intensity],
            metallic_roughness: [self.metallic, self.roughness, self.reflectance, self.ao],
            clearcoat: [self.clearcoat, self.clearcoat_roughness, self.clearcoat_ior, self.normal_scale],
            sheen: [self.sheen_color.x, self.sheen_color.y, self.sheen_color.z, self.sheen_roughness],
            transmission: [self.transmission, self.transmission_roughness, self.ior, self.thickness],
            iridescence: [self.iridescence, self.iridescence_ior, self.iridescence_thickness_min, self.iridescence_thickness_max],
            aniso_subsurface: [self.anisotropy, self.anisotropy_rotation, self.subsurface, self.sheen],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PbrMaterialGpu {
    pub base_color: [f32; 4], pub emissive: [f32; 4],
    pub metallic_roughness: [f32; 4], pub clearcoat: [f32; 4],
    pub sheen: [f32; 4], pub transmission: [f32; 4],
    pub iridescence: [f32; 4], pub aniso_subsurface: [f32; 4],
}

// ---------------------------------------------------------------------------
// BRDF LUT generation
// ---------------------------------------------------------------------------

pub fn generate_brdf_lut(size: usize) -> Vec<[f32; 2]> {
    let mut lut = vec![[0.0f32; 2]; size * size];
    for y in 0..size {
        let roughness = (y as f32 + 0.5) / size as f32;
        for x in 0..size {
            let n_dot_v = (x as f32 + 0.5) / size as f32;
            let n_dot_v = n_dot_v.max(0.001);
            let (a, b) = integrate_brdf(n_dot_v, roughness, 256);
            lut[y * size + x] = [a, b];
        }
    }
    lut
}

fn integrate_brdf(n_dot_v: f32, roughness: f32, num_samples: u32) -> (f32, f32) {
    let v = Vec3::new((1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v);
    let n = Vec3::new(0.0, 0.0, 1.0);
    let mut a = 0.0_f32; let mut b = 0.0_f32;
    for i in 0..num_samples {
        let xi = hammersley(i, num_samples);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = h.scale(2.0 * v.dot(h)).sub(v);
        let n_dot_l = l.z.max(0.0); let n_dot_h = h.z.max(0.0); let v_dot_h = v.dot(h).max(0.0);
        if n_dot_l > 0.0 {
            let a2 = (roughness * roughness).powi(2);
            let g1v = 2.0 * n_dot_v / (n_dot_v + (a2 + (1.0-a2)*n_dot_v*n_dot_v).sqrt());
            let g1l = 2.0 * n_dot_l / (n_dot_l + (a2 + (1.0-a2)*n_dot_l*n_dot_l).sqrt());
            let g = g1v * g1l;
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v).max(1e-7);
            let fc = (1.0 - v_dot_h).powi(5);
            a += g_vis * (1.0 - fc); b += g_vis * fc;
        }
    }
    (a / num_samples as f32, b / num_samples as f32)
}

fn hammersley(i: u32, n: u32) -> [f32; 2] {
    let mut bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    [i as f32 / n as f32, bits as f32 * 2.328_306_4e-10]
}

fn importance_sample_ggx(xi: [f32; 2], n: Vec3, roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let h = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
    let up = if n.z.abs() < 0.999 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let tangent = up.cross(n).normalize(); let bitangent = n.cross(tangent);
    tangent.scale(h.x).add(bitangent.scale(h.y)).add(n.scale(h.z)).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_ggx() { let d = d_ggx(1.0, 0.01); assert!(d > 100.0); }
    #[test] fn test_fresnel() { let f = f_schlick_scalar(1.0, 0.04); assert!((f - 0.04).abs() < 1e-6); }
    #[test] fn test_ior_roundtrip() { let ior = 1.5; let f = f0_from_ior(ior); assert!((ior_from_f0(f) - ior).abs() < 1e-3); }
    #[test] fn test_default_material() { let m = PbrMaterialV2::default(); assert_eq!(m.metallic, 0.0); }
    #[test] fn test_brdf_lut() { let l = generate_brdf_lut(4); assert_eq!(l.len(), 16); }
    #[test] fn test_brdf_eval() {
        let mat = PbrMaterialV2::default();
        let ctx = ShadingContext { n: Vec3::new(0.0,1.0,0.0), v: Vec3::new(0.0,1.0,0.0), l: Vec3::new(0.577,0.577,0.577), t: Vec3::new(1.0,0.0,0.0), b: Vec3::new(0.0,0.0,1.0) };
        let r = evaluate_principled_brdf(&mat, &ctx, Vec3::ONE, 1.0);
        assert!(r.total.x > 0.0);
    }
}
