//! Extended math utilities for the Genovo engine.
//!
//! Provides specialized math types and functions beyond the core linear algebra:
//!
//! - **Dual numbers** for automatic differentiation
//! - **Complex numbers** with full arithmetic
//! - **Fixed-point arithmetic** (16.16 and 24.8 formats)
//! - **Half-float (f16)** conversion utilities
//! - **Fast approximate math** (sin, cos, sqrt, inverse sqrt)
//! - **Packed normal encoding** (octahedron mapping, Lambert azimuthal)
//! - **Spherical coordinates** and **cylindrical coordinates**

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ===========================================================================
// Dual Numbers (Automatic Differentiation)
// ===========================================================================

/// A dual number `a + b*epsilon` where epsilon^2 = 0.
///
/// Dual numbers enable forward-mode automatic differentiation: evaluating
/// a function on `Dual::new(x, 1.0)` returns both `f(x)` and `f'(x)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// The real part.
    pub real: f64,
    /// The dual (infinitesimal) part.
    pub dual: f64,
}

impl Dual {
    /// Create a new dual number.
    pub const fn new(real: f64, dual: f64) -> Self {
        Self { real, dual }
    }

    /// Create a dual number representing a constant (dual part = 0).
    pub const fn constant(value: f64) -> Self {
        Self {
            real: value,
            dual: 0.0,
        }
    }

    /// Create a dual number representing a variable (dual part = 1).
    pub const fn variable(value: f64) -> Self {
        Self {
            real: value,
            dual: 1.0,
        }
    }

    /// Returns the function value.
    pub fn value(self) -> f64 {
        self.real
    }

    /// Returns the derivative value.
    pub fn derivative(self) -> f64 {
        self.dual
    }

    /// Sine of a dual number.
    pub fn sin(self) -> Self {
        Self {
            real: self.real.sin(),
            dual: self.dual * self.real.cos(),
        }
    }

    /// Cosine of a dual number.
    pub fn cos(self) -> Self {
        Self {
            real: self.real.cos(),
            dual: -self.dual * self.real.sin(),
        }
    }

    /// Tangent of a dual number.
    pub fn tan(self) -> Self {
        let c = self.real.cos();
        Self {
            real: self.real.tan(),
            dual: self.dual / (c * c),
        }
    }

    /// Natural exponential of a dual number.
    pub fn exp(self) -> Self {
        let e = self.real.exp();
        Self {
            real: e,
            dual: self.dual * e,
        }
    }

    /// Natural logarithm of a dual number.
    pub fn ln(self) -> Self {
        Self {
            real: self.real.ln(),
            dual: self.dual / self.real,
        }
    }

    /// Square root of a dual number.
    pub fn sqrt(self) -> Self {
        let s = self.real.sqrt();
        Self {
            real: s,
            dual: self.dual / (2.0 * s),
        }
    }

    /// Power of a dual number.
    pub fn pow(self, n: f64) -> Self {
        Self {
            real: self.real.powf(n),
            dual: self.dual * n * self.real.powf(n - 1.0),
        }
    }

    /// Absolute value.
    pub fn abs(self) -> Self {
        if self.real >= 0.0 {
            self
        } else {
            -self
        }
    }

    /// Reciprocal.
    pub fn recip(self) -> Self {
        let r2 = self.real * self.real;
        Self {
            real: 1.0 / self.real,
            dual: -self.dual / r2,
        }
    }
}

impl Add for Dual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl Mul for Dual {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let r2 = rhs.real * rhs.real;
        Self {
            real: self.real / rhs.real,
            dual: (self.dual * rhs.real - self.real * rhs.dual) / r2,
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            real: -self.real,
            dual: -self.dual,
        }
    }
}

impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dual >= 0.0 {
            write!(f, "{} + {}e", self.real, self.dual)
        } else {
            write!(f, "{} - {}e", self.real, -self.dual)
        }
    }
}

impl From<f64> for Dual {
    fn from(v: f64) -> Self {
        Dual::constant(v)
    }
}

/// Compute `f(x)` and `f'(x)` using dual numbers.
pub fn differentiate<F: Fn(Dual) -> Dual>(f: F, x: f64) -> (f64, f64) {
    let result = f(Dual::variable(x));
    (result.value(), result.derivative())
}

// ===========================================================================
// Complex Numbers
// ===========================================================================

/// A complex number `a + bi`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Create a new complex number.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Create a purely real complex number.
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// Create a purely imaginary complex number.
    pub const fn imag(im: f64) -> Self {
        Self { re: 0.0, im }
    }

    /// The complex zero.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// The complex one.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    /// The imaginary unit.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    /// Magnitude (absolute value / modulus).
    pub fn magnitude(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Squared magnitude.
    pub fn magnitude_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Phase angle (argument) in radians.
    pub fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate.
    pub fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Reciprocal (1/z).
    pub fn recip(self) -> Self {
        let d = self.magnitude_sq();
        Self {
            re: self.re / d,
            im: -self.im / d,
        }
    }

    /// Create from polar form (magnitude, phase).
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Convert to polar form (magnitude, phase).
    pub fn to_polar(self) -> (f64, f64) {
        (self.magnitude(), self.phase())
    }

    /// Complex exponential e^z.
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    /// Complex natural logarithm.
    pub fn ln(self) -> Self {
        Self {
            re: self.magnitude().ln(),
            im: self.phase(),
        }
    }

    /// Complex power z^n.
    pub fn pow(self, n: f64) -> Self {
        let (r, theta) = self.to_polar();
        Self::from_polar(r.powf(n), theta * n)
    }

    /// Complex square root.
    pub fn sqrt(self) -> Self {
        let (r, theta) = self.to_polar();
        Self::from_polar(r.sqrt(), theta / 2.0)
    }

    /// Complex sine.
    pub fn sin(self) -> Self {
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    /// Complex cosine.
    pub fn cos(self) -> Self {
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }

    /// Normalize to unit magnitude.
    pub fn normalize(self) -> Self {
        let m = self.magnitude();
        if m > 0.0 {
            Self {
                re: self.re / m,
                im: self.im / m,
            }
        } else {
            Self::ZERO
        }
    }

    /// Linear interpolation between two complex numbers.
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            re: self.re + (other.re - self.re) * t,
            im: self.im + (other.im - self.im) * t,
        }
    }

    /// Check if this complex number is purely real.
    pub fn is_real(self) -> bool {
        self.im.abs() < 1e-15
    }

    /// Check if this complex number is purely imaginary.
    pub fn is_imaginary(self) -> bool {
        self.re.abs() < 1e-15
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let d = rhs.magnitude_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{} + {}i", self.re, self.im)
        } else {
            write!(f, "{} - {}i", self.re, -self.im)
        }
    }
}

// ===========================================================================
// Fixed-Point Arithmetic
// ===========================================================================

/// Fixed-point number with 16 integer bits and 16 fractional bits (16.16).
///
/// Range: approximately -32768.0 to 32767.99998 with ~0.000015 precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fixed16(i32);

impl Fixed16 {
    /// Number of fractional bits.
    pub const FRAC_BITS: u32 = 16;
    /// Scale factor (2^16 = 65536).
    pub const SCALE: i32 = 1 << Self::FRAC_BITS;
    /// Maximum representable value.
    pub const MAX: Self = Fixed16(i32::MAX);
    /// Minimum representable value.
    pub const MIN: Self = Fixed16(i32::MIN);
    /// Zero.
    pub const ZERO: Self = Fixed16(0);
    /// One.
    pub const ONE: Self = Fixed16(1 << 16);

    /// Create from raw bits.
    pub const fn from_raw(bits: i32) -> Self {
        Self(bits)
    }

    /// Get the raw bit representation.
    pub const fn raw(self) -> i32 {
        self.0
    }

    /// Create from an f32.
    pub fn from_f32(v: f32) -> Self {
        Self((v * Self::SCALE as f32) as i32)
    }

    /// Create from an f64.
    pub fn from_f64(v: f64) -> Self {
        Self((v * Self::SCALE as f64) as i32)
    }

    /// Create from an integer.
    pub const fn from_int(v: i32) -> Self {
        Self(v << Self::FRAC_BITS)
    }

    /// Convert to f32.
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::SCALE as f32
    }

    /// Convert to f64.
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    /// Convert to integer (truncating).
    pub fn to_int(self) -> i32 {
        self.0 >> Self::FRAC_BITS
    }

    /// Floor (round toward negative infinity).
    pub fn floor(self) -> Self {
        Self(self.0 & !(Self::SCALE - 1))
    }

    /// Ceil (round toward positive infinity).
    pub fn ceil(self) -> Self {
        Self((self.0 + Self::SCALE - 1) & !(Self::SCALE - 1))
    }

    /// Fractional part.
    pub fn fract(self) -> Self {
        Self(self.0 & (Self::SCALE - 1))
    }

    /// Absolute value.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Saturating addition.
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Saturating multiplication (using i64 intermediate to avoid overflow).
    pub fn saturating_mul(self, rhs: Self) -> Self {
        let result = (self.0 as i64 * rhs.0 as i64) >> Self::FRAC_BITS;
        Self(result.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }

    /// Linear interpolation.
    pub fn lerp(self, other: Self, t: Self) -> Self {
        self + (other - self).saturating_mul(t)
    }
}

impl Add for Fixed16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for Fixed16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl Mul for Fixed16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let result = (self.0 as i64 * rhs.0 as i64) >> Self::FRAC_BITS;
        Self(result as i32)
    }
}

impl Div for Fixed16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let result = ((self.0 as i64) << Self::FRAC_BITS) / rhs.0 as i64;
        Self(result as i32)
    }
}

impl Neg for Fixed16 {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl fmt::Display for Fixed16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.to_f64())
    }
}

/// Fixed-point number with 24 integer bits and 8 fractional bits (24.8).
///
/// Range: approximately -8388608.0 to 8388607.996 with ~0.004 precision.
/// Useful for positions in large worlds where sub-pixel precision suffices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fixed24(i32);

impl Fixed24 {
    pub const FRAC_BITS: u32 = 8;
    pub const SCALE: i32 = 1 << Self::FRAC_BITS;
    pub const ZERO: Self = Fixed24(0);
    pub const ONE: Self = Fixed24(1 << 8);

    pub const fn from_raw(bits: i32) -> Self {
        Self(bits)
    }

    pub const fn raw(self) -> i32 {
        self.0
    }

    pub fn from_f32(v: f32) -> Self {
        Self((v * Self::SCALE as f32) as i32)
    }

    pub fn from_f64(v: f64) -> Self {
        Self((v * Self::SCALE as f64) as i32)
    }

    pub const fn from_int(v: i32) -> Self {
        Self(v << Self::FRAC_BITS)
    }

    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::SCALE as f32
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    pub fn to_int(self) -> i32 {
        self.0 >> Self::FRAC_BITS
    }

    pub fn floor(self) -> Self {
        Self(self.0 & !(Self::SCALE - 1))
    }

    pub fn ceil(self) -> Self {
        Self((self.0 + Self::SCALE - 1) & !(Self::SCALE - 1))
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl Add for Fixed24 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for Fixed24 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl Mul for Fixed24 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let result = (self.0 as i64 * rhs.0 as i64) >> Self::FRAC_BITS;
        Self(result as i32)
    }
}

impl Div for Fixed24 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let result = ((self.0 as i64) << Self::FRAC_BITS) / rhs.0 as i64;
        Self(result as i32)
    }
}

impl Neg for Fixed24 {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl fmt::Display for Fixed24 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.to_f64())
    }
}

// ===========================================================================
// Half Float (f16)
// ===========================================================================

/// IEEE 754 half-precision floating point (binary16).
///
/// Stored as a raw u16. Provides conversion to/from f32.
/// Useful for GPU data, HDR textures, and storage optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Half(u16);

impl Half {
    /// Positive zero.
    pub const ZERO: Self = Half(0x0000);
    /// One.
    pub const ONE: Self = Half(0x3C00);
    /// Negative one.
    pub const NEG_ONE: Self = Half(0xBC00);
    /// Positive infinity.
    pub const INFINITY: Self = Half(0x7C00);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Half(0xFC00);
    /// Not a number.
    pub const NAN: Self = Half(0x7E00);
    /// Maximum finite value (65504).
    pub const MAX: Self = Half(0x7BFF);
    /// Minimum positive normal (6.1e-5).
    pub const MIN_POSITIVE: Self = Half(0x0400);

    /// Create from raw bits.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Get the raw bits.
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert an f32 to half precision.
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007FFFFF;

        if exponent == 255 {
            // Infinity or NaN.
            if mantissa == 0 {
                return Self((sign | 0x7C00) as u16);
            } else {
                return Self((sign | 0x7E00) as u16); // Quiet NaN.
            }
        }

        let new_exp = exponent - 127 + 15;

        if new_exp >= 31 {
            // Overflow -> infinity.
            return Self((sign | 0x7C00) as u16);
        }

        if new_exp <= 0 {
            // Subnormal or underflow.
            if new_exp < -10 {
                // Too small -> zero.
                return Self(sign as u16);
            }
            let m = (mantissa | 0x00800000) >> (1 - new_exp + 13);
            return Self((sign | m) as u16);
        }

        let half_mantissa = mantissa >> 13;
        Self((sign | ((new_exp as u32) << 10) | half_mantissa) as u16)
    }

    /// Convert to f32.
    pub fn to_f32(self) -> f32 {
        let sign = ((self.0 >> 15) & 1) as u32;
        let exponent = ((self.0 >> 10) & 0x1F) as u32;
        let mantissa = (self.0 & 0x03FF) as u32;

        if exponent == 0 {
            if mantissa == 0 {
                return f32::from_bits(sign << 31);
            }
            // Subnormal: normalize.
            let mut e = 1i32;
            let mut m = mantissa;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            let f32_exp = (127 - 15 + e) as u32;
            let f32_bits = (sign << 31) | (f32_exp << 23) | (m << 13);
            return f32::from_bits(f32_bits);
        }

        if exponent == 31 {
            if mantissa == 0 {
                return f32::from_bits((sign << 31) | 0x7F800000);
            } else {
                return f32::from_bits((sign << 31) | 0x7FC00000);
            }
        }

        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (mantissa << 13);
        f32::from_bits(f32_bits)
    }

    /// Check if this is NaN.
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7C00) == 0x7C00 && (self.0 & 0x03FF) != 0
    }

    /// Check if this is infinite.
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7C00
    }

    /// Check if this is zero.
    pub fn is_zero(self) -> bool {
        (self.0 & 0x7FFF) == 0
    }

    /// Check if this is finite (not NaN or infinity).
    pub fn is_finite(self) -> bool {
        (self.0 & 0x7C00) != 0x7C00
    }

    /// Returns the sign bit (0 for positive, 1 for negative).
    pub fn sign_bit(self) -> u16 {
        (self.0 >> 15) & 1
    }

    /// Absolute value.
    pub fn abs(self) -> Self {
        Self(self.0 & 0x7FFF)
    }

    /// Negate.
    pub fn negate(self) -> Self {
        Self(self.0 ^ 0x8000)
    }
}

impl fmt::Display for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for Half {
    fn from(v: f32) -> Self {
        Half::from_f32(v)
    }
}

impl From<Half> for f32 {
    fn from(h: Half) -> f32 {
        h.to_f32()
    }
}

/// Convert a slice of f32 to Half.
pub fn f32_to_half_slice(src: &[f32], dst: &mut [Half]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = Half::from_f32(src[i]);
    }
}

/// Convert a slice of Half to f32.
pub fn half_to_f32_slice(src: &[Half], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].to_f32();
    }
}

// ===========================================================================
// Fast Approximate Math
// ===========================================================================

/// Fast approximate sine using a polynomial (max error ~0.001).
pub fn fast_sin(x: f32) -> f32 {
    // Normalize to [-pi, pi].
    let mut x = x;
    x = x % (2.0 * std::f32::consts::PI);
    if x > std::f32::consts::PI {
        x -= 2.0 * std::f32::consts::PI;
    } else if x < -std::f32::consts::PI {
        x += 2.0 * std::f32::consts::PI;
    }

    // Parabolic approximation with correction.
    let b = 4.0 / std::f32::consts::PI;
    let c = -4.0 / (std::f32::consts::PI * std::f32::consts::PI);
    let p = 0.225;

    let y = b * x + c * x * x.abs();
    p * (y * y.abs() - y) + y
}

/// Fast approximate cosine (derived from fast_sin).
pub fn fast_cos(x: f32) -> f32 {
    fast_sin(x + std::f32::consts::FRAC_PI_2)
}

/// Fast approximate tangent.
pub fn fast_tan(x: f32) -> f32 {
    let s = fast_sin(x);
    let c = fast_cos(x);
    if c.abs() < 1e-7 {
        if s >= 0.0 {
            f32::MAX
        } else {
            f32::MIN
        }
    } else {
        s / c
    }
}

/// Fast inverse square root (Quake III algorithm, refined).
pub fn fast_inv_sqrt(x: f32) -> f32 {
    let half_x = 0.5 * x;
    let i = x.to_bits();
    let i = 0x5F3759DF - (i >> 1);
    let y = f32::from_bits(i);
    let y = y * (1.5 - half_x * y * y); // First Newton iteration.
    let y = y * (1.5 - half_x * y * y); // Second Newton iteration.
    y
}

/// Fast approximate square root.
pub fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    x * fast_inv_sqrt(x)
}

/// Fast approximate atan2.
pub fn fast_atan2(y: f32, x: f32) -> f32 {
    let abs_x = x.abs();
    let abs_y = y.abs();

    let min = abs_x.min(abs_y);
    let max = abs_x.max(abs_y);

    if max == 0.0 {
        return 0.0;
    }

    let a = min / max;
    let s = a * a;
    let mut r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;

    if abs_y > abs_x {
        r = std::f32::consts::FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = std::f32::consts::PI - r;
    }
    if y < 0.0 {
        r = -r;
    }

    r
}

/// Fast approximate log2.
pub fn fast_log2(x: f32) -> f32 {
    let bits = x.to_bits() as f32;
    let log2 = bits * 1.1920928955078125e-7 - 126.94269504;
    log2
}

/// Fast approximate exp2 (2^x).
pub fn fast_exp2(x: f32) -> f32 {
    let clamp = x.max(-126.0).min(126.0);
    let bits = ((clamp + 126.94269504) * 8388608.0) as u32;
    f32::from_bits(bits)
}

/// Fast approximate pow.
pub fn fast_pow(base: f32, exp: f32) -> f32 {
    fast_exp2(exp * fast_log2(base))
}

// ===========================================================================
// Packed Normal Encoding
// ===========================================================================

/// Encode a unit normal vector into a 2D octahedron-mapped coordinate.
///
/// The result is in [-1, 1]^2 and can be quantized to any integer format.
pub fn octahedron_encode(normal: [f32; 3]) -> [f32; 2] {
    let [x, y, z] = normal;
    let l1 = x.abs() + y.abs() + z.abs();
    let mut oct_x = x / l1;
    let mut oct_y = y / l1;

    if z < 0.0 {
        let new_x = (1.0 - oct_y.abs()) * if oct_x >= 0.0 { 1.0 } else { -1.0 };
        let new_y = (1.0 - oct_x.abs()) * if oct_y >= 0.0 { 1.0 } else { -1.0 };
        oct_x = new_x;
        oct_y = new_y;
    }

    [oct_x, oct_y]
}

/// Decode a 2D octahedron-mapped coordinate back to a unit normal vector.
pub fn octahedron_decode(oct: [f32; 2]) -> [f32; 3] {
    let [oct_x, oct_y] = oct;
    let mut z = 1.0 - oct_x.abs() - oct_y.abs();
    let mut x = oct_x;
    let mut y = oct_y;

    if z < 0.0 {
        x = (1.0 - oct_y.abs()) * if oct_x >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - oct_x.abs()) * if oct_y >= 0.0 { 1.0 } else { -1.0 };
    }

    let len = (x * x + y * y + z * z).sqrt();
    [x / len, y / len, z / len]
}

/// Pack a unit normal into two u16 values using octahedron encoding.
pub fn octahedron_pack_u16(normal: [f32; 3]) -> [u16; 2] {
    let [ox, oy] = octahedron_encode(normal);
    let u = ((ox * 0.5 + 0.5) * 65535.0).round() as u16;
    let v = ((oy * 0.5 + 0.5) * 65535.0).round() as u16;
    [u, v]
}

/// Unpack a unit normal from two u16 values using octahedron encoding.
pub fn octahedron_unpack_u16(packed: [u16; 2]) -> [f32; 3] {
    let ox = packed[0] as f32 / 65535.0 * 2.0 - 1.0;
    let oy = packed[1] as f32 / 65535.0 * 2.0 - 1.0;
    octahedron_decode([ox, oy])
}

/// Pack a unit normal into a single u32 using octahedron encoding (16 bits each).
pub fn octahedron_pack_u32(normal: [f32; 3]) -> u32 {
    let [u, v] = octahedron_pack_u16(normal);
    (u as u32) | ((v as u32) << 16)
}

/// Unpack a unit normal from a u32.
pub fn octahedron_unpack_u32(packed: u32) -> [f32; 3] {
    let u = (packed & 0xFFFF) as u16;
    let v = ((packed >> 16) & 0xFFFF) as u16;
    octahedron_unpack_u16([u, v])
}

/// Lambert azimuthal equal-area projection for packing normals.
pub fn lambert_encode(normal: [f32; 3]) -> [f32; 2] {
    let [x, y, z] = normal;
    let f = (8.0 * z + 8.0).sqrt();
    if f.abs() < 1e-7 {
        [0.0, 0.0]
    } else {
        [x / f + 0.5, y / f + 0.5]
    }
}

/// Decode Lambert azimuthal projection back to a unit normal.
pub fn lambert_decode(enc: [f32; 2]) -> [f32; 3] {
    let fenc = [enc[0] * 4.0 - 2.0, enc[1] * 4.0 - 2.0];
    let f = fenc[0] * fenc[0] + fenc[1] * fenc[1];
    let g = (1.0 - f / 4.0).sqrt();
    let x = fenc[0] * g;
    let y = fenc[1] * g;
    let z = 1.0 - f / 2.0;
    [x, y, z]
}

// ===========================================================================
// Spherical Coordinates
// ===========================================================================

/// Spherical coordinates (ISO convention: r, theta, phi).
///
/// - `r`: radial distance (>= 0)
/// - `theta`: polar angle from positive z-axis (0 to pi)
/// - `phi`: azimuthal angle from positive x-axis in x-y plane (0 to 2*pi)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spherical {
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
}

impl Spherical {
    /// Create new spherical coordinates.
    pub fn new(r: f64, theta: f64, phi: f64) -> Self {
        Self { r, theta, phi }
    }

    /// Convert from Cartesian coordinates.
    pub fn from_cartesian(x: f64, y: f64, z: f64) -> Self {
        let r = (x * x + y * y + z * z).sqrt();
        if r < 1e-15 {
            return Self {
                r: 0.0,
                theta: 0.0,
                phi: 0.0,
            };
        }
        let theta = (z / r).acos();
        let phi = y.atan2(x);
        let phi = if phi < 0.0 {
            phi + 2.0 * std::f64::consts::PI
        } else {
            phi
        };
        Self { r, theta, phi }
    }

    /// Convert to Cartesian coordinates.
    pub fn to_cartesian(self) -> (f64, f64, f64) {
        let x = self.r * self.theta.sin() * self.phi.cos();
        let y = self.r * self.theta.sin() * self.phi.sin();
        let z = self.r * self.theta.cos();
        (x, y, z)
    }

    /// Spherical linear interpolation.
    pub fn slerp(self, other: Self, t: f64) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            theta: self.theta + (other.theta - self.theta) * t,
            phi: self.phi + (other.phi - self.phi) * t,
        }
    }

    /// Compute the unit direction vector.
    pub fn direction(self) -> (f64, f64, f64) {
        let unit = Spherical {
            r: 1.0,
            theta: self.theta,
            phi: self.phi,
        };
        unit.to_cartesian()
    }
}

impl fmt::Display for Spherical {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(r={:.4}, theta={:.4}, phi={:.4})",
            self.r, self.theta, self.phi
        )
    }
}

// ===========================================================================
// Cylindrical Coordinates
// ===========================================================================

/// Cylindrical coordinates (r, theta, z).
///
/// - `r`: radial distance from z-axis (>= 0)
/// - `theta`: azimuthal angle from positive x-axis (0 to 2*pi)
/// - `z`: height
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylindrical {
    pub r: f64,
    pub theta: f64,
    pub z: f64,
}

impl Cylindrical {
    /// Create new cylindrical coordinates.
    pub fn new(r: f64, theta: f64, z: f64) -> Self {
        Self { r, theta, z }
    }

    /// Convert from Cartesian coordinates.
    pub fn from_cartesian(x: f64, y: f64, z: f64) -> Self {
        let r = (x * x + y * y).sqrt();
        let theta = y.atan2(x);
        let theta = if theta < 0.0 {
            theta + 2.0 * std::f64::consts::PI
        } else {
            theta
        };
        Self { r, theta, z }
    }

    /// Convert to Cartesian coordinates.
    pub fn to_cartesian(self) -> (f64, f64, f64) {
        let x = self.r * self.theta.cos();
        let y = self.r * self.theta.sin();
        (x, y, self.z)
    }

    /// Convert to spherical coordinates.
    pub fn to_spherical(self) -> Spherical {
        let (x, y, z) = self.to_cartesian();
        Spherical::from_cartesian(x, y, z)
    }

    /// Linear interpolation.
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            theta: self.theta + (other.theta - self.theta) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

impl fmt::Display for Cylindrical {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(r={:.4}, theta={:.4}, z={:.4})",
            self.r, self.theta, self.z
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;
    const EPSILON_F32: f32 = 1e-3;

    #[test]
    fn test_dual_arithmetic() {
        let a = Dual::new(3.0, 1.0);
        let b = Dual::new(2.0, 0.0);
        let sum = a + b;
        assert!((sum.real - 5.0).abs() < EPSILON);
        assert!((sum.dual - 1.0).abs() < EPSILON);

        let product = a * b;
        assert!((product.real - 6.0).abs() < EPSILON);
        assert!((product.dual - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_dual_differentiation() {
        // d/dx (x^2) at x=3 should be 6.
        let (value, deriv) = differentiate(|x| x * x, 3.0);
        assert!((value - 9.0).abs() < EPSILON);
        assert!((deriv - 6.0).abs() < EPSILON);

        // d/dx (sin(x)) at x=0 should be 1.
        let (value, deriv) = differentiate(|x| x.sin(), 0.0);
        assert!((value - 0.0).abs() < EPSILON);
        assert!((deriv - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        assert!((a.magnitude() - 5.0).abs() < EPSILON);

        let b = Complex::new(1.0, 2.0);
        let product = a * b;
        // (3+4i)(1+2i) = 3+6i+4i+8i^2 = -5+10i
        assert!((product.re - (-5.0)).abs() < EPSILON);
        assert!((product.im - 10.0).abs() < EPSILON);

        let conj = a.conjugate();
        assert!((conj.im - (-4.0)).abs() < EPSILON);
    }

    #[test]
    fn test_complex_euler() {
        // e^(i*pi) = -1
        let z = Complex::imag(std::f64::consts::PI).exp();
        assert!((z.re - (-1.0)).abs() < EPSILON);
        assert!(z.im.abs() < EPSILON);
    }

    #[test]
    fn test_fixed16() {
        let a = Fixed16::from_f32(3.5);
        let b = Fixed16::from_f32(2.25);
        let sum = a + b;
        assert!((sum.to_f32() - 5.75).abs() < 0.01);

        let product = a * b;
        assert!((product.to_f32() - 7.875).abs() < 0.01);
    }

    #[test]
    fn test_half_float() {
        let h = Half::from_f32(1.0);
        assert!((h.to_f32() - 1.0).abs() < EPSILON_F32);

        let h = Half::from_f32(0.5);
        assert!((h.to_f32() - 0.5).abs() < EPSILON_F32);

        let h = Half::from_f32(-3.14);
        assert!((h.to_f32() - (-3.14)).abs() < 0.01);

        assert!(Half::NAN.is_nan());
        assert!(Half::INFINITY.is_infinite());
        assert!(Half::ZERO.is_zero());
    }

    #[test]
    fn test_fast_sin_cos() {
        for i in 0..100 {
            let x = (i as f32 / 100.0) * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
            let fs = fast_sin(x);
            let fc = fast_cos(x);
            let rs = x.sin();
            let rc = x.cos();
            assert!(
                (fs - rs).abs() < 0.01,
                "fast_sin({}) = {}, expected {}",
                x,
                fs,
                rs
            );
            assert!(
                (fc - rc).abs() < 0.01,
                "fast_cos({}) = {}, expected {}",
                x,
                fc,
                rc
            );
        }
    }

    #[test]
    fn test_fast_inv_sqrt() {
        let x = 4.0f32;
        let result = fast_inv_sqrt(x);
        assert!((result - 0.5).abs() < 0.001);

        let result = fast_sqrt(x);
        assert!((result - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_octahedron_roundtrip() {
        let normals = [
            [1.0f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.577, 0.577, 0.577], // approximate (1,1,1)/sqrt(3)
        ];

        for normal in &normals {
            let encoded = octahedron_encode(*normal);
            let decoded = octahedron_decode(encoded);
            for i in 0..3 {
                assert!(
                    (decoded[i] - normal[i]).abs() < 0.01,
                    "Octahedron roundtrip failed for {:?}: got {:?}",
                    normal,
                    decoded
                );
            }
        }
    }

    #[test]
    fn test_spherical_roundtrip() {
        let (x, y, z) = (1.0, 2.0, 3.0);
        let sph = Spherical::from_cartesian(x, y, z);
        let (rx, ry, rz) = sph.to_cartesian();
        assert!((rx - x).abs() < EPSILON);
        assert!((ry - y).abs() < EPSILON);
        assert!((rz - z).abs() < EPSILON);
    }

    #[test]
    fn test_cylindrical_roundtrip() {
        let (x, y, z) = (3.0, 4.0, 5.0);
        let cyl = Cylindrical::from_cartesian(x, y, z);
        let (rx, ry, rz) = cyl.to_cartesian();
        assert!((rx - x).abs() < EPSILON);
        assert!((ry - y).abs() < EPSILON);
        assert!((rz - z).abs() < EPSILON);
    }
}
