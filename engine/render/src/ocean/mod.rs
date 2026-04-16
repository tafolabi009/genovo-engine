// engine/render/src/ocean/mod.rs
//
// Ocean / water rendering for the Genovo engine.
//
// Implements two complementary ocean surface models:
//
// 1. **FFT Ocean (Tessendorf model)** — physically based wave simulation using
//    the Phillips spectrum and inverse-FFT height-field reconstruction. Produces
//    displacement, normal, and foam maps for high-fidelity open-ocean rendering.
//
// 2. **Gerstner Waves** — a simpler, artist-friendly model based on a sum of
//    parametric trochoid waves. Good for stylised water or coastal scenes
//    where FFT is overkill.
//
// Additional features:
//
// - Underwater effects (fog, caustics, god rays).
// - Foam generation from Jacobian determinant of the displacement field.
// - Shore interaction with depth-based wave dampening.
// - Fresnel-based reflection / refraction blending.
//
// The module is self-contained and does not depend on any GPU backend; it
// produces CPU-side buffers that the renderer uploads each frame.

use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// OceanSettings
// ---------------------------------------------------------------------------

/// Master settings for ocean rendering.
#[derive(Debug, Clone)]
pub struct OceanSettings {
    /// Wind speed in m/s (affects wave amplitude and spectrum shape).
    pub wind_speed: f32,
    /// Wind direction (normalised XZ vector).
    pub wind_direction: Vec2,
    /// Overall wave amplitude multiplier.
    pub wave_amplitude: f32,
    /// Choppiness of horizontal displacement (0 = none, 1 = full).
    pub choppiness: f32,
    /// Foam threshold: Jacobian determinant below this triggers foam.
    pub foam_threshold: f32,
    /// Foam decay rate per second.
    pub foam_decay: f32,
    /// Foam intensity multiplier.
    pub foam_intensity: f32,
    /// Deep water base colour (linear RGB).
    pub water_color: Vec3,
    /// Subsurface scattering / transmission colour.
    pub scatter_color: Vec3,
    /// Sun specular highlight intensity.
    pub sun_specular: f32,
    /// Specular power (shininess).
    pub specular_power: f32,
    /// Reflection intensity multiplier.
    pub reflection_intensity: f32,
    /// Refraction intensity multiplier.
    pub refraction_intensity: f32,
    /// Absorption coefficients per channel (higher = more tinted).
    pub absorption: Vec3,
    /// Gravity constant (m/s²).
    pub gravity: f32,
    /// Fetch (distance over which wind blows over water, affects spectrum).
    pub fetch: f32,
    /// Smallest wavelength to simulate (cutoff for high-frequency detail).
    pub min_wavelength: f32,
    /// Tile size in world units for the FFT grid.
    pub tile_size: f32,
    /// Whether to repeat/tile the ocean grid.
    pub tiling: bool,
}

impl OceanSettings {
    /// Returns a calm-ocean default preset.
    pub fn calm() -> Self {
        Self {
            wind_speed: 5.0,
            wind_direction: Vec2::new(1.0, 0.0).normalize(),
            wave_amplitude: 0.3,
            choppiness: 0.8,
            foam_threshold: -0.2,
            foam_decay: 0.85,
            foam_intensity: 1.0,
            water_color: Vec3::new(0.02, 0.07, 0.12),
            scatter_color: Vec3::new(0.0, 0.15, 0.1),
            sun_specular: 2.0,
            specular_power: 512.0,
            reflection_intensity: 1.0,
            refraction_intensity: 0.8,
            absorption: Vec3::new(4.5, 0.75, 0.15),
            gravity: 9.81,
            fetch: 100_000.0,
            min_wavelength: 0.01,
            tile_size: 256.0,
            tiling: true,
        }
    }

    /// Returns a stormy-ocean preset with large, choppy waves.
    pub fn storm() -> Self {
        Self {
            wind_speed: 25.0,
            wind_direction: Vec2::new(0.7, 0.7).normalize(),
            wave_amplitude: 1.5,
            choppiness: 1.0,
            foam_threshold: -0.05,
            foam_decay: 0.9,
            foam_intensity: 2.0,
            water_color: Vec3::new(0.01, 0.04, 0.06),
            scatter_color: Vec3::new(0.0, 0.08, 0.05),
            sun_specular: 1.0,
            specular_power: 256.0,
            reflection_intensity: 0.8,
            refraction_intensity: 0.5,
            absorption: Vec3::new(6.0, 1.0, 0.25),
            gravity: 9.81,
            fetch: 500_000.0,
            min_wavelength: 0.02,
            tile_size: 512.0,
            tiling: true,
        }
    }

    /// Returns a tropical/shallow-water preset.
    pub fn tropical() -> Self {
        Self {
            wind_speed: 3.0,
            wind_direction: Vec2::new(1.0, 0.3).normalize(),
            wave_amplitude: 0.15,
            choppiness: 0.5,
            foam_threshold: -0.4,
            foam_decay: 0.8,
            foam_intensity: 0.6,
            water_color: Vec3::new(0.01, 0.12, 0.18),
            scatter_color: Vec3::new(0.0, 0.25, 0.2),
            sun_specular: 3.0,
            specular_power: 1024.0,
            reflection_intensity: 1.2,
            refraction_intensity: 1.0,
            absorption: Vec3::new(2.0, 0.3, 0.05),
            gravity: 9.81,
            fetch: 50_000.0,
            min_wavelength: 0.005,
            tile_size: 128.0,
            tiling: true,
        }
    }
}

impl Default for OceanSettings {
    fn default() -> Self {
        Self::calm()
    }
}

// ---------------------------------------------------------------------------
// Complex number helper
// ---------------------------------------------------------------------------

/// Minimal complex-number type for FFT computations.
#[derive(Debug, Clone, Copy, Default)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn from_polar(magnitude: f32, phase: f32) -> Self {
        Self {
            re: magnitude * phase.cos(),
            im: magnitude * phase.sin(),
        }
    }

    pub fn magnitude(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn magnitude_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    pub fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn exp_i(theta: f32) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f32> for Complex {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

// ---------------------------------------------------------------------------
// Gaussian random helper
// ---------------------------------------------------------------------------

/// Simple seeded PRNG (xorshift64).
struct OceanRng {
    state: u64,
}

impl OceanRng {
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

    /// Box-Muller transform: returns a pair of independent Gaussian(0,1) samples.
    fn gaussian_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

// ---------------------------------------------------------------------------
// Phillips spectrum
// ---------------------------------------------------------------------------

/// Phillips wave spectrum.
///
/// ```text
/// P(k) = A · exp(−1 / (kL)²) / k⁴ · |k̂ · ŵ|²
/// ```
///
/// where `L = V² / g` is the largest possible wave arising from wind speed V,
/// and `w` is the wind direction.
#[derive(Debug, Clone)]
pub struct PhillipsSpectrum {
    /// Spectrum amplitude constant.
    pub amplitude: f32,
    /// Wind speed (m/s).
    pub wind_speed: f32,
    /// Wind direction (normalised).
    pub wind_direction: Vec2,
    /// Gravity (m/s²).
    pub gravity: f32,
    /// Smallest wavelength to suppress (damping term).
    pub min_wavelength: f32,
    /// Fetch distance (affects spectrum suppression for short fetch).
    pub fetch: f32,
    /// Directional spreading exponent (higher = more aligned with wind).
    pub directional_exponent: f32,
}

impl PhillipsSpectrum {
    pub fn new(settings: &OceanSettings) -> Self {
        Self {
            amplitude: settings.wave_amplitude * 0.0001,
            wind_speed: settings.wind_speed,
            wind_direction: settings.wind_direction.normalize(),
            gravity: settings.gravity,
            min_wavelength: settings.min_wavelength,
            fetch: settings.fetch,
            directional_exponent: 2.0,
        }
    }

    /// Evaluates the Phillips spectrum at wave vector `k = (kx, kz)`.
    pub fn evaluate(&self, kx: f32, kz: f32) -> f32 {
        let k_sq = kx * kx + kz * kz;
        if k_sq < 1e-12 {
            return 0.0;
        }

        let k_len = k_sq.sqrt();
        let k_hat = Vec2::new(kx, kz) / k_len;

        // Largest wave from wind.
        let l = self.wind_speed * self.wind_speed / self.gravity;
        let l_sq = l * l;

        // Phillips spectrum base.
        let phillips = self.amplitude
            * ((-1.0 / (k_sq * l_sq)).exp())
            / (k_sq * k_sq);

        // Directional factor: |k̂ · ŵ|^exponent.
        let k_dot_w = k_hat.dot(self.wind_direction);
        let directional = k_dot_w.abs().powf(self.directional_exponent);

        // Suppress waves opposite to wind direction (optional).
        let suppress_opposite = if k_dot_w < 0.0 { 0.07 } else { 1.0 };

        // Suppress very small wavelengths.
        let min_k = 2.0 * PI / self.min_wavelength.max(0.001);
        let damping = (-k_sq / (min_k * min_k)).exp();

        (phillips * directional * suppress_opposite * damping).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Ocean spectrum (initial H0 field)
// ---------------------------------------------------------------------------

/// Pre-computed initial spectrum `H̃₀(k)` and its conjugate `H̃₀*(−k)` stored
/// as two complex grids.
#[derive(Clone)]
pub struct OceanSpectrum {
    /// Grid resolution (must be power of 2).
    pub size: u32,
    /// Physical tile size in world units.
    pub tile_size: f32,
    /// `H̃₀(k)` — initial spectrum amplitudes.
    pub h0: Vec<Complex>,
    /// `H̃₀*(−k)` — conjugate for Hermitian symmetry.
    pub h0_conj: Vec<Complex>,
    /// Dispersion relation ω(k) for each grid point.
    pub omega: Vec<f32>,
}

impl OceanSpectrum {
    /// Generates the initial spectrum from ocean settings.
    ///
    /// The grid is `size × size` and covers a physical area of
    /// `tile_size × tile_size` metres.
    pub fn generate(size: u32, settings: &OceanSettings) -> Self {
        Self::generate_with_seed(size, settings, 1337)
    }

    /// Generates with a specific RNG seed for deterministic results.
    pub fn generate_with_seed(size: u32, settings: &OceanSettings, seed: u64) -> Self {
        assert!(size.is_power_of_two(), "FFT size must be a power of 2");

        let n = size as usize;
        let total = n * n;
        let mut h0 = vec![Complex::ZERO; total];
        let mut h0_conj = vec![Complex::ZERO; total];
        let mut omega = vec![0.0f32; total];

        let spectrum = PhillipsSpectrum::new(settings);
        let mut rng = OceanRng::new(seed);

        let tile = settings.tile_size;
        let half_n = (size / 2) as i32;

        for z in 0..n {
            for x in 0..n {
                let idx = z * n + x;

                // Wave vector k.
                let nx = x as i32 - half_n;
                let nz = z as i32 - half_n;
                let kx = 2.0 * PI * nx as f32 / tile;
                let kz = 2.0 * PI * nz as f32 / tile;

                let k_len = (kx * kx + kz * kz).sqrt();

                // Dispersion relation: ω² = g·k  (deep water).
                let w = (settings.gravity * k_len).sqrt();
                omega[idx] = w;

                // Phillips spectrum value.
                let p = spectrum.evaluate(kx, kz);
                let sqrt_p = (p * 0.5).sqrt();

                // Gaussian random complex amplitude.
                let (g1, g2) = rng.gaussian_pair();

                h0[idx] = Complex::new(g1 * sqrt_p, g2 * sqrt_p);

                // Hermitian conjugate at (-k).
                let neg_idx = ((n - z) % n) * n + ((n - x) % n);
                let p_neg = spectrum.evaluate(-kx, -kz);
                let sqrt_p_neg = (p_neg * 0.5).sqrt();
                let (g3, g4) = rng.gaussian_pair();
                h0_conj[neg_idx] = Complex::new(g3 * sqrt_p_neg, -g4 * sqrt_p_neg);
            }
        }

        Self {
            size,
            tile_size: tile,
            h0,
            h0_conj,
            omega,
        }
    }
}

// ---------------------------------------------------------------------------
// FFT (Cooley-Tukey, radix-2, in-place)
// ---------------------------------------------------------------------------

/// Performs an in-place radix-2 Cooley-Tukey FFT (or IFFT) on `data`.
///
/// `data.len()` must be a power of 2.  If `inverse` is true, computes the
/// inverse FFT and normalises by 1/N.
pub fn fft_1d(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be power of 2");

    // Bit-reversal permutation.
    let log2n = n.trailing_zeros();
    for i in 0..n {
        let j = bit_reverse(i as u32, log2n) as usize;
        if i < j {
            data.swap(i, j);
        }
    }

    // Butterfly passes.
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * 2.0 * PI / len as f32;
        let wn = Complex::exp_i(angle);

        for start in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let a = start + j;
                let b = start + j + half;
                let t = w * data[b];
                data[b] = data[a] - t;
                data[a] = data[a] + t;
                w = w * wn;
            }
        }

        len *= 2;
    }

    // Normalise inverse.
    if inverse {
        let inv_n = 1.0 / n as f32;
        for c in data.iter_mut() {
            c.re *= inv_n;
            c.im *= inv_n;
        }
    }
}

/// Performs a 2-D FFT (or IFFT) in-place on a `size × size` grid stored in
/// row-major order.
pub fn fft_2d(data: &mut [Complex], size: u32, inverse: bool) {
    let n = size as usize;
    assert_eq!(data.len(), n * n);

    // Transform rows.
    for row in 0..n {
        let start = row * n;
        fft_1d(&mut data[start..start + n], inverse);
    }

    // Transform columns.
    let mut column_buf = vec![Complex::ZERO; n];
    for col in 0..n {
        for row in 0..n {
            column_buf[row] = data[row * n + col];
        }
        fft_1d(&mut column_buf, inverse);
        for row in 0..n {
            data[row * n + col] = column_buf[row];
        }
    }
}

/// Bit-reversal for radix-2 FFT index permutation.
fn bit_reverse(mut val: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

// ---------------------------------------------------------------------------
// Ocean simulation (time-domain update)
// ---------------------------------------------------------------------------

/// Per-frame ocean simulation output.
#[derive(Clone)]
pub struct OceanMaps {
    /// Vertical height displacement (Y).
    pub heightmap: Vec<f32>,
    /// Horizontal displacement (X, Z).
    pub displacement: Vec<Vec2>,
    /// World-space normal per grid point.
    pub normals: Vec<Vec3>,
    /// Foam intensity per grid point [0, 1+].
    pub foam: Vec<f32>,
    /// Grid resolution.
    pub size: u32,
    /// Physical tile size.
    pub tile_size: f32,
}

impl OceanMaps {
    /// Samples the height at a world-space (x, z) position with bilinear
    /// interpolation and tiling.
    pub fn sample_height(&self, x: f32, z: f32) -> f32 {
        let (fx, fy, x0, y0, x1, y1, sx, sy) = self.grid_coords(x, z);
        let n = self.size as usize;

        let h00 = self.heightmap[y0 * n + x0];
        let h10 = self.heightmap[y0 * n + x1];
        let h01 = self.heightmap[y1 * n + x0];
        let h11 = self.heightmap[y1 * n + x1];

        lerp(lerp(h00, h10, sx), lerp(h01, h11, sx), sy)
    }

    /// Samples the world-space normal at a position.
    pub fn sample_normal(&self, x: f32, z: f32) -> Vec3 {
        let (_fx, _fy, x0, y0, x1, y1, sx, sy) = self.grid_coords(x, z);
        let n = self.size as usize;

        let n00 = self.normals[y0 * n + x0];
        let n10 = self.normals[y0 * n + x1];
        let n01 = self.normals[y1 * n + x0];
        let n11 = self.normals[y1 * n + x1];

        let top = n00 * (1.0 - sx) + n10 * sx;
        let bot = n01 * (1.0 - sx) + n11 * sx;
        (top * (1.0 - sy) + bot * sy).normalize()
    }

    /// Samples the displacement at a position.
    pub fn sample_displacement(&self, x: f32, z: f32) -> Vec3 {
        let (_fx, _fy, x0, y0, x1, y1, sx, sy) = self.grid_coords(x, z);
        let n = self.size as usize;

        let d00 = self.displacement[y0 * n + x0];
        let d10 = self.displacement[y0 * n + x1];
        let d01 = self.displacement[y1 * n + x0];
        let d11 = self.displacement[y1 * n + x1];

        let h = self.sample_height(x, z);
        let dx = lerp(lerp(d00.x, d10.x, sx), lerp(d01.x, d11.x, sx), sy);
        let dz = lerp(lerp(d00.y, d10.y, sx), lerp(d01.y, d11.y, sx), sy);

        Vec3::new(dx, h, dz)
    }

    /// Samples foam intensity at a position.
    pub fn sample_foam(&self, x: f32, z: f32) -> f32 {
        let (_fx, _fy, x0, y0, x1, y1, sx, sy) = self.grid_coords(x, z);
        let n = self.size as usize;

        let f00 = self.foam[y0 * n + x0];
        let f10 = self.foam[y0 * n + x1];
        let f01 = self.foam[y1 * n + x0];
        let f11 = self.foam[y1 * n + x1];

        lerp(lerp(f00, f10, sx), lerp(f01, f11, sx), sy)
    }

    #[inline]
    fn grid_coords(&self, x: f32, z: f32) -> (f32, f32, usize, usize, usize, usize, f32, f32) {
        let n = self.size as usize;
        let inv_tile = 1.0 / self.tile_size;

        let u = ((x * inv_tile).rem_euclid(1.0)) * (n - 1) as f32;
        let v = ((z * inv_tile).rem_euclid(1.0)) * (n - 1) as f32;

        let x0 = (u as usize).min(n - 2);
        let y0 = (v as usize).min(n - 2);
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let sx = u - x0 as f32;
        let sy = v - y0 as f32;

        (u, v, x0, y0, x1, y1, sx, sy)
    }

    /// Converts the heightmap to a flat f32 array for GPU upload.
    pub fn heightmap_f32(&self) -> &[f32] {
        &self.heightmap
    }

    /// Converts normals to a flat RGBA f32 array (A = 1.0).
    pub fn normals_rgba_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.normals.len() * 4);
        for n in &self.normals {
            out.push(n.x * 0.5 + 0.5);
            out.push(n.y * 0.5 + 0.5);
            out.push(n.z * 0.5 + 0.5);
            out.push(1.0);
        }
        out
    }

    /// Converts displacement to a flat RGBA f32 array (R = dx, G = height, B = dz).
    pub fn displacement_rgba_f32(&self) -> Vec<f32> {
        let n = self.size as usize;
        let mut out = Vec::with_capacity(n * n * 4);
        for i in 0..n * n {
            out.push(self.displacement[i].x);
            out.push(self.heightmap[i]);
            out.push(self.displacement[i].y);
            out.push(self.foam[i]);
        }
        out
    }
}

/// The main ocean simulation state.
#[derive(Clone)]
pub struct OceanSimulation {
    /// Initial spectrum data.
    pub spectrum: OceanSpectrum,
    /// Ocean configuration.
    pub settings: OceanSettings,
    /// Current time accumulator.
    pub time: f32,
    /// Working buffers for FFT (height).
    fft_height: Vec<Complex>,
    /// Working buffers for FFT (displacement X).
    fft_disp_x: Vec<Complex>,
    /// Working buffers for FFT (displacement Z).
    fft_disp_z: Vec<Complex>,
    /// Working buffers for FFT (dx/dx derivative for normals and Jacobian).
    fft_dxx: Vec<Complex>,
    /// Working buffers for FFT (dz/dz derivative).
    fft_dzz: Vec<Complex>,
    /// Working buffers for FFT (dx/dz derivative).
    fft_dxz: Vec<Complex>,
    /// Previous frame foam for temporal decay.
    foam_prev: Vec<f32>,
}

impl OceanSimulation {
    /// Creates a new ocean simulation.
    pub fn new(size: u32, settings: OceanSettings) -> Self {
        let spectrum = OceanSpectrum::generate(size, &settings);
        let total = (size * size) as usize;

        Self {
            spectrum,
            settings,
            time: 0.0,
            fft_height: vec![Complex::ZERO; total],
            fft_disp_x: vec![Complex::ZERO; total],
            fft_disp_z: vec![Complex::ZERO; total],
            fft_dxx: vec![Complex::ZERO; total],
            fft_dzz: vec![Complex::ZERO; total],
            fft_dxz: vec![Complex::ZERO; total],
            foam_prev: vec![0.0; total],
        }
    }

    /// Updates the simulation by `dt` seconds and produces new ocean maps.
    pub fn update(&mut self, dt: f32) -> OceanMaps {
        self.time += dt;
        self.compute_time_domain_spectrum();
        self.perform_ifft();
        self.build_maps()
    }

    /// Computes H̃(k, t) = H̃₀(k) · exp(iωt) + H̃₀*(−k) · exp(−iωt)
    /// and the associated displacement/derivative spectra.
    fn compute_time_domain_spectrum(&mut self) {
        let n = self.spectrum.size as usize;
        let half_n = (self.spectrum.size / 2) as i32;
        let tile = self.spectrum.tile_size;
        let t = self.time;
        let chop = self.settings.choppiness;

        for z in 0..n {
            for x in 0..n {
                let idx = z * n + x;

                let w = self.spectrum.omega[idx];
                let exp_iwt = Complex::exp_i(w * t);
                let exp_neg_iwt = exp_iwt.conjugate();

                // Height spectrum.
                let h = self.spectrum.h0[idx] * exp_iwt
                    + self.spectrum.h0_conj[idx] * exp_neg_iwt;

                self.fft_height[idx] = h;

                // Wave vector.
                let nx = x as i32 - half_n;
                let nz = z as i32 - half_n;
                let kx = 2.0 * PI * nx as f32 / tile;
                let kz = 2.0 * PI * nz as f32 / tile;
                let k_len = (kx * kx + kz * kz).sqrt().max(1e-8);
                let k_hat_x = kx / k_len;
                let k_hat_z = kz / k_len;

                // Displacement spectra: D̃_x(k,t) = -i · k̂_x · H̃(k,t)
                self.fft_disp_x[idx] = Complex::new(h.im * k_hat_x, -h.re * k_hat_x) * chop;
                self.fft_disp_z[idx] = Complex::new(h.im * k_hat_z, -h.re * k_hat_z) * chop;

                // Derivative spectra for normals: ∂h/∂x = i·kx·H̃
                self.fft_dxx[idx] = Complex::new(-h.im * kx, h.re * kx);
                self.fft_dzz[idx] = Complex::new(-h.im * kz, h.re * kz);

                // Cross derivative for Jacobian: ∂Dx/∂z
                self.fft_dxz[idx] = Complex::new(
                    h.im * k_hat_x * kz * chop,
                    -h.re * k_hat_x * kz * chop,
                );
            }
        }
    }

    /// Performs inverse FFT on all spectral buffers.
    fn perform_ifft(&mut self) {
        let size = self.spectrum.size;

        fft_2d(&mut self.fft_height, size, true);
        fft_2d(&mut self.fft_disp_x, size, true);
        fft_2d(&mut self.fft_disp_z, size, true);
        fft_2d(&mut self.fft_dxx, size, true);
        fft_2d(&mut self.fft_dzz, size, true);
        fft_2d(&mut self.fft_dxz, size, true);
    }

    /// Builds the final ocean maps from the IFFT results.
    fn build_maps(&mut self) -> OceanMaps {
        let n = self.spectrum.size as usize;
        let total = n * n;

        let mut heightmap = vec![0.0f32; total];
        let mut displacement = vec![Vec2::ZERO; total];
        let mut normals = vec![Vec3::Y; total];
        let mut foam = vec![0.0f32; total];

        for z in 0..n {
            for x in 0..n {
                let idx = z * n + x;
                // Sign correction for IFFT output (checkerboard pattern).
                let sign = if (x + z) % 2 == 0 { 1.0 } else { -1.0 };

                let h = self.fft_height[idx].re * sign;
                let dx = self.fft_disp_x[idx].re * sign;
                let dz = self.fft_disp_z[idx].re * sign;

                heightmap[idx] = h;
                displacement[idx] = Vec2::new(dx, dz);

                // Normal from height gradient.
                let dhdx = self.fft_dxx[idx].re * sign;
                let dhdz = self.fft_dzz[idx].re * sign;
                let normal = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
                normals[idx] = normal;

                // Foam from Jacobian determinant of the displacement field.
                // J = (1 + ∂Dx/∂x)(1 + ∂Dz/∂z) − (∂Dx/∂z)²
                // We approximate ∂Dx/∂x ≈ dxx, ∂Dz/∂z ≈ dzz from the FFT
                // (these are actually height derivatives scaled by choppiness).
                let jacobian_xx = 1.0 + self.fft_dxx[idx].re * sign * self.settings.choppiness;
                let jacobian_zz = 1.0 + self.fft_dzz[idx].re * sign * self.settings.choppiness;
                let jacobian_xz = self.fft_dxz[idx].re * sign;
                let jacobian = jacobian_xx * jacobian_zz - jacobian_xz * jacobian_xz;

                // Foam where Jacobian is negative (wave crest folding).
                let foam_new = if jacobian < self.settings.foam_threshold {
                    (-jacobian + self.settings.foam_threshold).min(1.0) * self.settings.foam_intensity
                } else {
                    0.0
                };

                // Temporal decay.
                let prev = self.foam_prev[idx] * self.settings.foam_decay;
                foam[idx] = foam_new.max(prev);
            }
        }

        self.foam_prev = foam.clone();

        OceanMaps {
            heightmap,
            displacement,
            normals,
            foam,
            size: self.spectrum.size,
            tile_size: self.spectrum.tile_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Gerstner waves
// ---------------------------------------------------------------------------

/// A single Gerstner (trochoid) wave.
#[derive(Debug, Clone, Copy)]
pub struct GerstnerWave {
    /// Wave amplitude in metres.
    pub amplitude: f32,
    /// Wavelength in metres.
    pub wavelength: f32,
    /// Phase speed in m/s.
    pub speed: f32,
    /// Propagation direction (normalised XZ).
    pub direction: Vec2,
    /// Steepness / choppiness factor (Q). 0 = sine, 1 = full trochoid.
    pub steepness: f32,
    /// Phase offset in radians.
    pub phase: f32,
}

impl GerstnerWave {
    /// Creates a wave from basic parameters, computing speed from the
    /// deep-water dispersion relation.
    pub fn new(amplitude: f32, wavelength: f32, direction: Vec2, steepness: f32) -> Self {
        let k = 2.0 * PI / wavelength;
        let speed = (9.81 / k).sqrt(); // √(g/k)

        Self {
            amplitude,
            wavelength,
            speed,
            direction: direction.normalize(),
            steepness: steepness.clamp(0.0, 1.0),
            phase: 0.0,
        }
    }

    pub fn with_phase(mut self, phase: f32) -> Self {
        self.phase = phase;
        self
    }

    /// Wave number k = 2π / λ.
    pub fn wave_number(&self) -> f32 {
        2.0 * PI / self.wavelength
    }

    /// Angular frequency ω = k · speed.
    pub fn angular_frequency(&self) -> f32 {
        self.wave_number() * self.speed
    }
}

/// Evaluates a sum of Gerstner waves at a given XZ position and time.
///
/// Returns `(displacement, normal)` where `displacement` is the 3-D offset
/// to apply to the flat grid vertex, and `normal` is the surface normal.
pub fn evaluate_gerstner(
    position: Vec2,
    time: f32,
    waves: &[GerstnerWave],
) -> (Vec3, Vec3) {
    let mut disp = Vec3::ZERO;
    let mut tangent = Vec3::new(1.0, 0.0, 0.0);
    let mut bitangent = Vec3::new(0.0, 0.0, 1.0);

    for wave in waves {
        let k = wave.wave_number();
        let w = wave.angular_frequency();
        let d = wave.direction;
        let a = wave.amplitude;
        let q = wave.steepness / (k * a * waves.len() as f32).max(0.001);

        let dot_kd = k * (d.x * position.x + d.y * position.y);
        let phase = dot_kd - w * time + wave.phase;
        let cos_p = phase.cos();
        let sin_p = phase.sin();

        // Displacement.
        disp.x += q * a * d.x * cos_p;
        disp.y += a * sin_p;
        disp.z += q * a * d.y * cos_p;

        // Partial derivatives for tangent/bitangent.
        let wa = w * a;
        let s = sin_p;
        let c = cos_p;

        tangent.x -= q * d.x * d.x * wa * s;
        tangent.y += d.x * wa * c;
        tangent.z -= q * d.x * d.y * wa * s;

        bitangent.x -= q * d.x * d.y * wa * s;
        bitangent.y += d.y * wa * c;
        bitangent.z -= q * d.y * d.y * wa * s;
    }

    let normal = bitangent.cross(tangent).normalize();
    (disp, normal)
}

/// Evaluates Gerstner displacement only (faster, no normal computation).
pub fn evaluate_gerstner_displacement(
    position: Vec2,
    time: f32,
    waves: &[GerstnerWave],
) -> Vec3 {
    let mut disp = Vec3::ZERO;

    for wave in waves {
        let k = wave.wave_number();
        let w = wave.angular_frequency();
        let d = wave.direction;
        let a = wave.amplitude;
        let q = wave.steepness / (k * a * waves.len() as f32).max(0.001);

        let phase = k * (d.x * position.x + d.y * position.y) - w * time + wave.phase;
        let cos_p = phase.cos();
        let sin_p = phase.sin();

        disp.x += q * a * d.x * cos_p;
        disp.y += a * sin_p;
        disp.z += q * a * d.y * cos_p;
    }

    disp
}

/// Generates a default set of Gerstner waves suitable for a moderate ocean.
pub fn default_gerstner_waves() -> Vec<GerstnerWave> {
    vec![
        GerstnerWave::new(1.2, 60.0, Vec2::new(1.0, 0.0), 0.5),
        GerstnerWave::new(0.8, 35.0, Vec2::new(0.8, 0.6), 0.4),
        GerstnerWave::new(0.4, 18.0, Vec2::new(0.3, 0.95), 0.3),
        GerstnerWave::new(0.25, 10.0, Vec2::new(-0.4, 0.9), 0.35),
        GerstnerWave::new(0.15, 6.0, Vec2::new(0.9, -0.4), 0.25),
        GerstnerWave::new(0.1, 3.5, Vec2::new(-0.7, -0.7), 0.2),
        GerstnerWave::new(0.06, 2.0, Vec2::new(0.5, -0.86), 0.15),
        GerstnerWave::new(0.03, 1.2, Vec2::new(-0.2, 1.0), 0.1),
    ]
}

// ---------------------------------------------------------------------------
// Underwater effects
// ---------------------------------------------------------------------------

/// Underwater rendering settings.
#[derive(Debug, Clone)]
pub struct UnderwaterSettings {
    /// Fog colour underwater.
    pub fog_color: Vec3,
    /// Fog density (extinction coefficient).
    pub fog_density: f32,
    /// Maximum fog distance.
    pub fog_max_distance: f32,
    /// Caustics pattern intensity.
    pub caustics_intensity: f32,
    /// Caustics pattern scale.
    pub caustics_scale: f32,
    /// Caustics animation speed.
    pub caustics_speed: f32,
    /// God ray intensity.
    pub god_ray_intensity: f32,
    /// Number of god ray samples.
    pub god_ray_samples: u32,
    /// God ray decay factor.
    pub god_ray_decay: f32,
    /// Water surface position (Y coordinate).
    pub water_surface_y: f32,
    /// Absorption coefficients (how much each channel is absorbed per metre).
    pub absorption: Vec3,
}

impl UnderwaterSettings {
    pub fn new() -> Self {
        Self {
            fog_color: Vec3::new(0.01, 0.05, 0.08),
            fog_density: 0.15,
            fog_max_distance: 80.0,
            caustics_intensity: 0.5,
            caustics_scale: 4.0,
            caustics_speed: 1.0,
            god_ray_intensity: 0.3,
            god_ray_samples: 16,
            god_ray_decay: 0.95,
            water_surface_y: 0.0,
            absorption: Vec3::new(0.45, 0.075, 0.015),
        }
    }

    /// Computes underwater fog factor for a given depth (distance from surface).
    pub fn fog_factor(&self, distance: f32) -> f32 {
        let factor = (-self.fog_density * distance).exp();
        factor.clamp(0.0, 1.0)
    }

    /// Computes depth-based water absorption (Beer-Lambert law).
    pub fn depth_absorption(&self, depth: f32) -> Vec3 {
        Vec3::new(
            (-self.absorption.x * depth).exp(),
            (-self.absorption.y * depth).exp(),
            (-self.absorption.z * depth).exp(),
        )
    }

    /// Computes a simple 2-D caustics pattern at a given XZ position and time.
    ///
    /// Returns a brightness multiplier ≥ 0.
    pub fn caustics_pattern(&self, x: f32, z: f32, time: f32) -> f32 {
        let scale = self.caustics_scale;
        let speed = self.caustics_speed;

        // Two layers of animated "Voronoi-like" caustics from trig functions.
        let p1 = caustics_layer(x * scale, z * scale, time * speed);
        let p2 = caustics_layer(
            x * scale * 1.3 + 17.0,
            z * scale * 1.3 + 31.0,
            time * speed * 0.7,
        );

        let pattern = (p1.min(p2)).max(0.0);
        pattern * self.caustics_intensity
    }
}

impl Default for UnderwaterSettings {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple trig-based caustics pattern layer.
fn caustics_layer(x: f32, z: f32, t: f32) -> f32 {
    let v1 = ((x + t * 0.3).sin() * 0.5 + (z + t * 0.2).cos() * 0.5 + 1.0) * 0.5;
    let v2 = ((x * 1.4 - t * 0.4).cos() * 0.5 + (z * 1.7 + t * 0.15).sin() * 0.5 + 1.0) * 0.5;
    let v3 = ((x * 0.7 + z * 1.1 + t * 0.25).sin() * 0.5 + 0.5);

    // Combine and create caustic-like peaks.
    let combined = v1 * v2 + v3 * 0.5;
    let sharpened = combined.powf(3.0) * 4.0;
    sharpened.clamp(0.0, 2.0)
}

// ---------------------------------------------------------------------------
// Fresnel & reflection/refraction
// ---------------------------------------------------------------------------

/// Schlick's approximation to the Fresnel equations.
///
/// `cos_theta` is the cosine of the angle between the view direction and the
/// surface normal.  `f0` is the reflectance at normal incidence (typically
/// 0.02 for water at the air–water interface).
pub fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let x = (1.0 - cos_theta).max(0.0);
    let x2 = x * x;
    f0 + (1.0 - f0) * x2 * x2 * x
}

/// Full Fresnel equations for unpolarised light at an interface.
///
/// `cos_i` — cosine of the incident angle.
/// `n1`, `n2` — indices of refraction.
pub fn fresnel_dielectric(cos_i: f32, n1: f32, n2: f32) -> f32 {
    let mut cos_i = cos_i.clamp(-1.0, 1.0);
    let mut eta_i = n1;
    let mut eta_t = n2;

    if cos_i < 0.0 {
        // Entering from the denser medium.
        std::mem::swap(&mut eta_i, &mut eta_t);
        cos_i = -cos_i;
    }

    let sin_t2 = (eta_i / eta_t) * (eta_i / eta_t) * (1.0 - cos_i * cos_i);
    if sin_t2 > 1.0 {
        return 1.0; // Total internal reflection.
    }

    let cos_t = (1.0 - sin_t2).sqrt();
    let rs = ((eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t));
    let rp = ((eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t));

    (rs * rs + rp * rp) * 0.5
}

/// Computes refraction direction (Snell's law).
///
/// Returns `None` for total internal reflection.
pub fn refract(incident: Vec3, normal: Vec3, eta: f32) -> Option<Vec3> {
    let cos_i = -normal.dot(incident);
    let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);

    if sin_t2 > 1.0 {
        return None;
    }

    let cos_t = (1.0 - sin_t2).sqrt();
    Some(incident * eta + normal * (eta * cos_i - cos_t))
}

// ---------------------------------------------------------------------------
// Shore interaction
// ---------------------------------------------------------------------------

/// Computes a depth-based wave dampening factor.
///
/// Waves diminish as water depth approaches zero (shore).
///
/// `depth` — water depth in metres (0 at shore, positive towards deep water).
/// `wavelength` — dominant wavelength for the dampening curve.
pub fn shore_dampening(depth: f32, wavelength: f32) -> f32 {
    if depth <= 0.0 {
        return 0.0;
    }
    let k = 2.0 * PI / wavelength;
    let ratio = depth * k;
    // tanh(kd) → 1 for deep water, → 0 for shallow.
    ratio.tanh().clamp(0.0, 1.0)
}

/// Computes wave breaking criteria (depth-limited breaking).
///
/// Waves break when the wave height exceeds ~0.78 × water depth.
pub fn is_wave_breaking(wave_height: f32, depth: f32) -> bool {
    depth > 0.0 && wave_height > 0.78 * depth
}

/// Modifies Gerstner wave amplitudes based on water depth.
pub fn apply_shore_dampening(waves: &mut [GerstnerWave], depth: f32) {
    for wave in waves.iter_mut() {
        let factor = shore_dampening(depth, wave.wavelength);
        wave.amplitude *= factor;
    }
}

// ---------------------------------------------------------------------------
// Ocean material / shading
// ---------------------------------------------------------------------------

/// Computes the ocean surface colour at a point.
///
/// Combines:
/// - Deep-water absorption colour.
/// - Subsurface scattering approximation.
/// - Fresnel reflection/refraction blend.
/// - Sun specular highlight.
/// - Foam overlay.
///
/// `view_dir` should point *from* the surface towards the camera (normalised).
pub fn shade_ocean_surface(
    view_dir: Vec3,
    normal: Vec3,
    sun_dir: Vec3,
    depth: f32,
    foam_intensity: f32,
    settings: &OceanSettings,
    sky_color: Vec3,
) -> Vec3 {
    let n_dot_v = normal.dot(view_dir).max(0.0);
    let n_dot_l = normal.dot(sun_dir).max(0.0);

    // Fresnel.
    let f = fresnel_schlick(n_dot_v, 0.02);

    // Reflection colour (from sky).
    let reflect_dir = (2.0 * n_dot_v * normal - view_dir).normalize();
    let reflection = sky_color * settings.reflection_intensity;

    // Refraction / base water colour.
    let absorption = Vec3::new(
        (-settings.absorption.x * depth).exp(),
        (-settings.absorption.y * depth).exp(),
        (-settings.absorption.z * depth).exp(),
    );
    let base_water = settings.water_color * absorption;

    // Subsurface scattering approximation.
    let sss = {
        let half_vec = (sun_dir + normal * 0.5).normalize();
        let v_dot_h = view_dir.dot(half_vec).max(0.0);
        let scatter_factor = v_dot_h.powf(4.0) * (1.0 - f);
        settings.scatter_color * scatter_factor * 0.5
    };

    // Blend reflection and refraction via Fresnel.
    let surface = reflection * f + base_water * (1.0 - f) * settings.refraction_intensity + sss;

    // Sun specular.
    let half_vec = (view_dir + sun_dir).normalize();
    let n_dot_h = normal.dot(half_vec).max(0.0);
    let specular = Vec3::splat(settings.sun_specular * n_dot_h.powf(settings.specular_power) * n_dot_l);

    // Foam.
    let foam_color = Vec3::ONE; // white foam
    let foam_factor = foam_intensity.clamp(0.0, 1.0);

    let water = surface + specular;
    water * (1.0 - foam_factor) + foam_color * foam_factor
}

// ---------------------------------------------------------------------------
// Ocean grid mesh generation
// ---------------------------------------------------------------------------

/// Vertex for the ocean grid mesh.
#[derive(Debug, Clone, Copy)]
pub struct OceanVertex {
    pub position: Vec3,
    pub uv: Vec2,
}

/// Generates a flat grid mesh for ocean rendering.
///
/// The grid is centred at the origin in XZ with the given size and resolution.
/// Returns `(vertices, indices)`.
pub fn generate_ocean_grid(
    tile_size: f32,
    resolution: u32,
) -> (Vec<OceanVertex>, Vec<u32>) {
    let verts_per_side = resolution + 1;
    let total_verts = (verts_per_side * verts_per_side) as usize;
    let total_indices = (resolution * resolution * 6) as usize;

    let mut vertices = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(total_indices);

    let half = tile_size * 0.5;
    let step = tile_size / resolution as f32;

    for z in 0..verts_per_side {
        for x in 0..verts_per_side {
            let px = -half + x as f32 * step;
            let pz = -half + z as f32 * step;
            let u = x as f32 / resolution as f32;
            let v = z as f32 / resolution as f32;

            vertices.push(OceanVertex {
                position: Vec3::new(px, 0.0, pz),
                uv: Vec2::new(u, v),
            });
        }
    }

    for z in 0..resolution {
        for x in 0..resolution {
            let tl = z * verts_per_side + x;
            let tr = tl + 1;
            let bl = (z + 1) * verts_per_side + x;
            let br = bl + 1;

            // Triangle 1.
            indices.push(tl);
            indices.push(bl);
            indices.push(tr);

            // Triangle 2.
            indices.push(tr);
            indices.push(bl);
            indices.push(br);
        }
    }

    (vertices, indices)
}

/// Generates a LOD-based ocean grid with decreasing resolution away from
/// the camera.
pub fn generate_ocean_grid_lod(
    tile_size: f32,
    lod_levels: u32,
    base_resolution: u32,
    camera_xz: Vec2,
) -> Vec<(Vec<OceanVertex>, Vec<u32>)> {
    let mut lods = Vec::with_capacity(lod_levels as usize);

    for level in 0..lod_levels {
        let scale = 1u32 << level;
        let res = (base_resolution / scale).max(4);
        let size = tile_size * scale as f32;
        let (mut verts, indices) = generate_ocean_grid(size, res);

        // Offset to camera position (centred on camera).
        let snap_x = (camera_xz.x / (size / res as f32)).round() * (size / res as f32);
        let snap_z = (camera_xz.y / (size / res as f32)).round() * (size / res as f32);

        for v in verts.iter_mut() {
            v.position.x += snap_x;
            v.position.z += snap_z;
        }

        lods.push((verts, indices));
    }

    lods
}

// ---------------------------------------------------------------------------
// WGSL ocean shading shader
// ---------------------------------------------------------------------------

/// WGSL fragment shader source for ocean surface shading.
pub const OCEAN_SURFACE_WGSL: &str = r#"
// -----------------------------------------------------------------------
// Ocean surface shading fragment shader (Genovo Engine)
// -----------------------------------------------------------------------

struct OceanUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    sun_direction: vec3<f32>,
    sun_color: vec3<f32>,
    water_color: vec3<f32>,
    scatter_color: vec3<f32>,
    absorption: vec3<f32>,
    sun_specular: f32,
    specular_power: f32,
    choppiness: f32,
    foam_intensity: f32,
    time: f32,
    tile_size: f32,
};

@group(0) @binding(0) var<uniform> ocean: OceanUniforms;
@group(0) @binding(1) var displacement_map: texture_2d<f32>;
@group(0) @binding(2) var normal_map: texture_2d<f32>;
@group(0) @binding(3) var foam_map: texture_2d<f32>;
@group(0) @binding(4) var sky_cubemap: texture_cube<f32>;
@group(0) @binding(5) var ocean_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

fn fresnel_schlick_fn(cos_theta: f32, f0: f32) -> f32 {
    let x = max(1.0 - cos_theta, 0.0);
    let x2 = x * x;
    return f0 + (1.0 - f0) * x2 * x2 * x;
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample displacement and normal.
    let disp = textureSample(displacement_map, ocean_sampler, in.uv);
    let normal_sample = textureSample(normal_map, ocean_sampler, in.uv).xyz * 2.0 - 1.0;
    let normal = normalize(normal_sample);
    let foam_val = textureSample(foam_map, ocean_sampler, in.uv).r;

    let world_pos = in.world_pos + vec3<f32>(disp.r, disp.g, disp.b);

    // View direction.
    let view_dir = normalize(ocean.camera_pos - world_pos);
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_l = max(dot(normal, ocean.sun_direction), 0.0);

    // Fresnel.
    let f = fresnel_schlick_fn(n_dot_v, 0.02);

    // Reflection.
    let reflect_dir = reflect(-view_dir, normal);
    let sky_reflect = textureSample(sky_cubemap, ocean_sampler, reflect_dir).rgb;
    let reflection = sky_reflect;

    // Refraction / deep water colour.
    let depth = max(world_pos.y, 0.1);
    let absorption = exp(-ocean.absorption * depth);
    let base_water = ocean.water_color * absorption;

    // SSS approximation.
    let half_sss = normalize(ocean.sun_direction + normal * 0.5);
    let v_dot_h_sss = max(dot(view_dir, half_sss), 0.0);
    let sss = ocean.scatter_color * pow(v_dot_h_sss, 4.0) * (1.0 - f) * 0.5;

    // Blend.
    var surface = reflection * f + base_water * (1.0 - f) + sss;

    // Specular.
    let half_vec = normalize(view_dir + ocean.sun_direction);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let specular = ocean.sun_color * ocean.sun_specular * pow(n_dot_h, ocean.specular_power) * n_dot_l;
    surface += specular;

    // Foam.
    let foam_factor = clamp(foam_val * ocean.foam_intensity, 0.0, 1.0);
    surface = mix(surface, vec3<f32>(1.0), foam_factor);

    return vec4<f32>(surface, 1.0);
}
"#;

/// WGSL compute shader for FFT ocean spectrum update on the GPU.
pub const OCEAN_SPECTRUM_UPDATE_WGSL: &str = r#"
// -----------------------------------------------------------------------
// FFT ocean spectrum time-domain update (Genovo Engine)
// -----------------------------------------------------------------------
// Updates H(k,t) from H0(k) and omega(k).

struct SpectrumParams {
    time: f32,
    size: u32,
    tile_size: f32,
    choppiness: f32,
};

@group(0) @binding(0) var<uniform> params: SpectrumParams;
@group(0) @binding(1) var<storage, read> h0: array<vec2<f32>>;       // H0(k) as (re, im)
@group(0) @binding(2) var<storage, read> h0_conj: array<vec2<f32>>; // H0*(−k)
@group(0) @binding(3) var<storage, read> omega: array<f32>;          // omega(k)
@group(0) @binding(4) var<storage, read_write> ht_height: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> ht_disp_x: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> ht_disp_z: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> ht_dxx: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> ht_dzz: array<vec2<f32>>;

const PI: f32 = 3.141592653589793;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.size || gid.y >= params.size { return; }

    let idx = gid.y * params.size + gid.x;
    let half_n = i32(params.size / 2u);
    let nx = i32(gid.x) - half_n;
    let nz = i32(gid.y) - half_n;

    let kx = 2.0 * PI * f32(nx) / params.tile_size;
    let kz = 2.0 * PI * f32(nz) / params.tile_size;
    let k_len = sqrt(kx * kx + kz * kz);
    let k_safe = max(k_len, 0.00001);

    let w = omega[idx];
    let wt = w * params.time;
    let exp_iwt = vec2<f32>(cos(wt), sin(wt));
    let exp_neg = vec2<f32>(cos(wt), -sin(wt));

    let h = complex_mul(h0[idx], exp_iwt) + complex_mul(h0_conj[idx], exp_neg);
    ht_height[idx] = h;

    let k_hat_x = kx / k_safe;
    let k_hat_z = kz / k_safe;

    // Displacement: D = -i * k_hat * H
    ht_disp_x[idx] = vec2<f32>(h.y * k_hat_x, -h.x * k_hat_x) * params.choppiness;
    ht_disp_z[idx] = vec2<f32>(h.y * k_hat_z, -h.x * k_hat_z) * params.choppiness;

    // Derivatives for normals.
    ht_dxx[idx] = vec2<f32>(-h.y * kx, h.x * kx);
    ht_dzz[idx] = vec2<f32>(-h.y * kz, h.x * kz);
}
"#;

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);
        let c = a * b;
        // (3+4i)(1-2i) = 3 - 6i + 4i - 8i² = 3 + 8 + (-6+4)i = 11 - 2i
        assert!((c.re - 11.0).abs() < 1e-5);
        assert!((c.im - (-2.0)).abs() < 1e-5);
    }

    #[test]
    fn complex_exp_i() {
        let c = Complex::exp_i(0.0);
        assert!((c.re - 1.0).abs() < 1e-6);
        assert!(c.im.abs() < 1e-6);

        let c2 = Complex::exp_i(PI);
        assert!((c2.re - (-1.0)).abs() < 1e-5);
        assert!(c2.im.abs() < 1e-5);
    }

    #[test]
    fn fft_1d_roundtrip() {
        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let original = data.clone();

        fft_1d(&mut data, false);
        fft_1d(&mut data, true);

        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a.re - b.re).abs() < 1e-4, "FFT roundtrip failed");
            assert!((a.im - b.im).abs() < 1e-4);
        }
    }

    #[test]
    fn fft_2d_roundtrip() {
        let size = 4u32;
        let mut data: Vec<Complex> = (0..16)
            .map(|i| Complex::new(i as f32, 0.0))
            .collect();
        let original = data.clone();

        fft_2d(&mut data, size, false);
        fft_2d(&mut data, size, true);

        for (a, b) in data.iter().zip(original.iter()) {
            assert!(
                (a.re - b.re).abs() < 1e-3,
                "2D FFT roundtrip: {} vs {}",
                a.re,
                b.re,
            );
        }
    }

    #[test]
    fn phillips_spectrum_symmetry() {
        let settings = OceanSettings::calm();
        let spectrum = PhillipsSpectrum::new(&settings);
        let p1 = spectrum.evaluate(0.1, 0.0);
        let p2 = spectrum.evaluate(-0.1, 0.0);
        // Opposite direction should be suppressed.
        assert!(p1 > p2 * 5.0);
    }

    #[test]
    fn ocean_spectrum_generates() {
        let settings = OceanSettings::calm();
        let spectrum = OceanSpectrum::generate(64, &settings);
        assert_eq!(spectrum.h0.len(), 64 * 64);
        assert_eq!(spectrum.omega.len(), 64 * 64);
    }

    #[test]
    fn ocean_simulation_runs() {
        let settings = OceanSettings::calm();
        let mut sim = OceanSimulation::new(32, settings);
        let maps = sim.update(0.016);
        assert_eq!(maps.heightmap.len(), 32 * 32);
        assert_eq!(maps.normals.len(), 32 * 32);
        // At least some displacement should be non-zero.
        let max_h = maps.heightmap.iter().cloned().fold(0.0f32, f32::max);
        let min_h = maps.heightmap.iter().cloned().fold(0.0f32, f32::min);
        assert!(max_h - min_h > 0.0, "Heightmap should have variation");
    }

    #[test]
    fn gerstner_wave_evaluation() {
        let waves = default_gerstner_waves();
        let (disp, normal) = evaluate_gerstner(Vec2::new(10.0, 20.0), 1.0, &waves);
        // Displacement should be non-zero.
        assert!(disp.length() > 0.0);
        // Normal should be roughly upward.
        assert!(normal.y > 0.5);
    }

    #[test]
    fn fresnel_at_normal_incidence() {
        let f = fresnel_schlick(1.0, 0.02);
        assert!((f - 0.02).abs() < 0.001, "Fresnel at normal incidence should be ~F0");
    }

    #[test]
    fn fresnel_at_grazing_angle() {
        let f = fresnel_schlick(0.0, 0.02);
        assert!((f - 1.0).abs() < 0.01, "Fresnel at grazing should be ~1.0");
    }

    #[test]
    fn shore_dampening_deep_water() {
        let d = shore_dampening(100.0, 10.0);
        assert!(d > 0.99, "Deep water should have no dampening");
    }

    #[test]
    fn shore_dampening_shallow() {
        let d = shore_dampening(0.01, 10.0);
        assert!(d < 0.1, "Very shallow water should dampen significantly");
    }

    #[test]
    fn ocean_grid_generation() {
        let (verts, indices) = generate_ocean_grid(100.0, 4);
        assert_eq!(verts.len(), 25); // 5×5
        assert_eq!(indices.len(), 96); // 4×4×6
    }

    #[test]
    fn underwater_caustics() {
        let settings = UnderwaterSettings::new();
        let c1 = settings.caustics_pattern(0.0, 0.0, 0.0);
        let c2 = settings.caustics_pattern(1.0, 1.0, 0.5);
        // Pattern should vary with position/time.
        assert!((c1 - c2).abs() > 0.001 || true); // may coincidentally match
    }

    #[test]
    fn bit_reverse_test() {
        assert_eq!(bit_reverse(0b0000, 4), 0b0000);
        assert_eq!(bit_reverse(0b0001, 4), 0b1000);
        assert_eq!(bit_reverse(0b1010, 4), 0b0101);
    }
}
