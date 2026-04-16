//! Color types and color space conversions for the Genovo engine.
//!
//! Provides a linear-space `Color` type with conversions to/from sRGB, HSV,
//! HSL, and hexadecimal representations. Includes perceptual operations like
//! luminance, contrast ratio, and gradient sampling.
//!
//! # Color Space Convention
//!
//! All `Color` values are stored in **linear space** (not gamma-encoded sRGB).
//! This is critical for correct lighting calculations in the renderer. Use
//! `from_srgb` / `to_srgb` to convert when interfacing with user-facing color
//! pickers or file formats that use sRGB.

use std::fmt;

// ===========================================================================
// Color
// ===========================================================================

/// A linear-space RGBA color with f32 components.
///
/// Components are stored in linear color space. The alpha channel uses
/// straight (non-premultiplied) alpha. Values are not clamped internally,
/// allowing HDR (values > 1.0) for physically based rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    /// Red component (linear space).
    pub r: f32,
    /// Green component (linear space).
    pub g: f32,
    /// Blue component (linear space).
    pub b: f32,
    /// Alpha (opacity). 0.0 = fully transparent, 1.0 = fully opaque.
    pub a: f32,
}

impl Color {
    // -- Named color constants (linear space approximations) -----------------
    // These are the sRGB values converted to linear space for accuracy.

    pub const TRANSPARENT: Color = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
    pub const BLACK: Color = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const WHITE: Color = Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const RED: Color = Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Color = Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Color = Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const YELLOW: Color = Color { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const CYAN: Color = Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const MAGENTA: Color = Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const ORANGE: Color = Color { r: 1.0, g: 0.376_47, b: 0.0, a: 1.0 }; // #FF6400 approx
    pub const PURPLE: Color = Color { r: 0.215_86, g: 0.0, b: 0.533_27, a: 1.0 }; // #800080 approx
    pub const GRAY: Color = Color { r: 0.215_86, g: 0.215_86, b: 0.215_86, a: 1.0 }; // #808080
    pub const DARK_GRAY: Color = Color { r: 0.051_269, g: 0.051_269, b: 0.051_269, a: 1.0 }; // #404040
    pub const LIGHT_GRAY: Color = Color { r: 0.502_886, g: 0.502_886, b: 0.502_886, a: 1.0 }; // #BFBFBF
    pub const BROWN: Color = Color { r: 0.376_47, g: 0.154_48, b: 0.023_153, a: 1.0 }; // #A0522D approx
    pub const PINK: Color = Color { r: 1.0, g: 0.527_12, b: 0.597_20, a: 1.0 }; // #FFC0CB approx
    pub const LIME: Color = Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const NAVY: Color = Color { r: 0.0, g: 0.0, b: 0.215_86, a: 1.0 }; // #000080
    pub const TEAL: Color = Color { r: 0.0, g: 0.215_86, b: 0.215_86, a: 1.0 }; // #008080
    pub const MAROON: Color = Color { r: 0.215_86, g: 0.0, b: 0.0, a: 1.0 }; // #800000
    pub const OLIVE: Color = Color { r: 0.215_86, g: 0.215_86, b: 0.0, a: 1.0 }; // #808000
    pub const AQUA: Color = Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const SILVER: Color = Color { r: 0.502_886, g: 0.502_886, b: 0.502_886, a: 1.0 }; // #C0C0C0
    pub const GOLD: Color = Color { r: 1.0, g: 0.679_54, b: 0.0, a: 1.0 }; // #FFD700 approx
    pub const SKY_BLUE: Color = Color { r: 0.244_97, g: 0.617_21, b: 0.830_77, a: 1.0 }; // #87CEEB approx
    pub const CORAL: Color = Color { r: 1.0, g: 0.212_23, b: 0.114_44, a: 1.0 }; // #FF7F50 approx
    pub const SALMON: Color = Color { r: 0.981_47, g: 0.367_65, b: 0.194_62, a: 1.0 }; // #FA8072 approx
    pub const INDIGO: Color = Color { r: 0.069_25, g: 0.0, b: 0.533_27, a: 1.0 }; // #4B0082 approx
    pub const TURQUOISE: Color = Color { r: 0.088_66, g: 0.745_40, b: 0.614_83, a: 1.0 }; // #40E0D0 approx
    pub const VIOLET: Color = Color { r: 0.577_58, g: 0.127_44, b: 0.905_20, a: 1.0 }; // #EE82EE approx
    pub const CRIMSON: Color = Color { r: 0.774_98, g: 0.013_70, b: 0.073_24, a: 1.0 }; // #DC143C approx
    pub const CHARTREUSE: Color = Color { r: 0.212_23, g: 1.0, b: 0.0, a: 1.0 }; // #7FFF00 approx

    // -- Constructors -------------------------------------------------------

    /// Create a new color with the given RGBA components in linear space.
    #[inline]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Create a new opaque color (alpha = 1.0).
    #[inline]
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Create a color from sRGB byte values (0-255).
    ///
    /// Converts from gamma-encoded sRGB to linear space.
    pub fn from_rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: srgb_to_linear(r as f32 / 255.0),
            g: srgb_to_linear(g as f32 / 255.0),
            b: srgb_to_linear(b as f32 / 255.0),
            a: a as f32 / 255.0,
        }
    }

    /// Convert to sRGB byte values (0-255).
    pub fn to_rgba8(&self) -> [u8; 4] {
        [
            (linear_to_srgb(self.r) * 255.0).round().clamp(0.0, 255.0) as u8,
            (linear_to_srgb(self.g) * 255.0).round().clamp(0.0, 255.0) as u8,
            (linear_to_srgb(self.b) * 255.0).round().clamp(0.0, 255.0) as u8,
            (self.a * 255.0).round().clamp(0.0, 255.0) as u8,
        ]
    }

    // -- sRGB conversions ---------------------------------------------------

    /// Create a color from sRGB float values (gamma-encoded, [0, 1]).
    ///
    /// This is the correct conversion to use when receiving colors from
    /// UI color pickers, CSS colors, or image files stored in sRGB.
    pub fn from_srgb(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: srgb_to_linear(r),
            g: srgb_to_linear(g),
            b: srgb_to_linear(b),
            a: 1.0,
        }
    }

    /// Create a color from sRGB float values with alpha.
    pub fn from_srgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r: srgb_to_linear(r),
            g: srgb_to_linear(g),
            b: srgb_to_linear(b),
            a,
        }
    }

    /// Convert to sRGB float values (gamma-encoded, [0, 1]).
    pub fn to_srgb(&self) -> (f32, f32, f32) {
        (
            linear_to_srgb(self.r),
            linear_to_srgb(self.g),
            linear_to_srgb(self.b),
        )
    }

    /// Convert to sRGB float values with alpha.
    pub fn to_srgba(&self) -> (f32, f32, f32, f32) {
        let (r, g, b) = self.to_srgb();
        (r, g, b, self.a)
    }

    // -- Hex conversions ----------------------------------------------------

    /// Parse a hex color string.
    ///
    /// Supports formats: `"#RGB"`, `"#RGBA"`, `"#RRGGBB"`, `"#RRGGBBAA"`,
    /// and the same without the `#` prefix.
    ///
    /// The input is treated as sRGB and converted to linear space.
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim().trim_start_matches('#');
        match hex.len() {
            3 => {
                // #RGB -> expand to #RRGGBB
                let r = u8::from_str_radix(&hex[0..1], 16).ok()?;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()?;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()?;
                Some(Self::from_rgba8(r * 17, g * 17, b * 17, 255))
            }
            4 => {
                // #RGBA
                let r = u8::from_str_radix(&hex[0..1], 16).ok()?;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()?;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()?;
                let a = u8::from_str_radix(&hex[3..4], 16).ok()?;
                Some(Self::from_rgba8(r * 17, g * 17, b * 17, a * 17))
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some(Self::from_rgba8(r, g, b, 255))
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
                Some(Self::from_rgba8(r, g, b, a))
            }
            _ => None,
        }
    }

    /// Convert to a hex string in sRGB space (e.g. "#FF00FF").
    pub fn to_hex(&self) -> String {
        let [r, g, b, _a] = self.to_rgba8();
        format!("#{:02X}{:02X}{:02X}", r, g, b)
    }

    /// Convert to a hex string with alpha (e.g. "#FF00FFCC").
    pub fn to_hex_with_alpha(&self) -> String {
        let [r, g, b, a] = self.to_rgba8();
        format!("#{:02X}{:02X}{:02X}{:02X}", r, g, b, a)
    }

    // -- HSV conversions ----------------------------------------------------

    /// Create a color from HSV values.
    ///
    /// * `h` — hue in degrees [0, 360)
    /// * `s` — saturation [0, 1]
    /// * `v` — value/brightness [0, 1]
    ///
    /// The resulting color is in linear space.
    pub fn from_hsv(h: f32, s: f32, v: f32) -> Self {
        let (r, g, b) = hsv_to_rgb(h, s, v);
        // The HSV conversion produces sRGB values, so convert to linear.
        Self::from_srgb(r, g, b)
    }

    /// Convert to HSV. Returns `(hue, saturation, value)` where hue is in
    /// degrees [0, 360), and saturation/value are in [0, 1].
    pub fn to_hsv(&self) -> (f32, f32, f32) {
        let (r, g, b) = self.to_srgb();
        rgb_to_hsv(r, g, b)
    }

    // -- HSL conversions ----------------------------------------------------

    /// Create a color from HSL values.
    ///
    /// * `h` — hue in degrees [0, 360)
    /// * `s` — saturation [0, 1]
    /// * `l` — lightness [0, 1]
    pub fn from_hsl(h: f32, s: f32, l: f32) -> Self {
        let (r, g, b) = hsl_to_rgb(h, s, l);
        Self::from_srgb(r, g, b)
    }

    /// Convert to HSL. Returns `(hue, saturation, lightness)`.
    pub fn to_hsl(&self) -> (f32, f32, f32) {
        let (r, g, b) = self.to_srgb();
        rgb_to_hsl(r, g, b)
    }

    // -- Interpolation ------------------------------------------------------

    /// Linearly interpolate between this color and `other` in linear space.
    ///
    /// This is the correct interpolation for lighting calculations but may
    /// produce desaturated midpoints for some color pairs. Use `lerp_hsv`
    /// for perceptually smoother color transitions.
    #[inline]
    pub fn lerp(&self, other: &Color, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        Color {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// Interpolate in HSV space for perceptually smoother transitions.
    ///
    /// Useful for rainbow effects, heat maps, and UI color animations
    /// where linear RGB interpolation would produce muddy midpoints.
    pub fn lerp_hsv(&self, other: &Color, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        let (h1, s1, v1) = self.to_hsv();
        let (h2, s2, v2) = other.to_hsv();

        // Interpolate hue along the shortest arc.
        let mut dh = h2 - h1;
        if dh > 180.0 {
            dh -= 360.0;
        } else if dh < -180.0 {
            dh += 360.0;
        }
        let h = (h1 + dh * t).rem_euclid(360.0);
        let s = s1 + (s2 - s1) * t;
        let v = v1 + (v2 - v1) * t;
        let a = self.a + (other.a - self.a) * t;

        let mut c = Color::from_hsv(h, s, v);
        c.a = a;
        c
    }

    // -- Perceptual operations ----------------------------------------------

    /// Perceptual luminance using the Rec. 709 coefficients.
    ///
    /// Computes the relative luminance of the color in linear space:
    /// `L = 0.2126 * R + 0.7152 * G + 0.0722 * B`
    ///
    /// This matches the WCAG definition of relative luminance.
    #[inline]
    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// WCAG contrast ratio between this color and another.
    ///
    /// Returns a value in [1, 21]. The WCAG 2.0 guidelines require:
    /// - Normal text: ratio >= 4.5:1
    /// - Large text: ratio >= 3:1
    /// - Enhanced: ratio >= 7:1
    pub fn contrast_ratio(&self, other: &Color) -> f32 {
        let l1 = self.luminance();
        let l2 = other.luminance();
        let lighter = l1.max(l2);
        let darker = l1.min(l2);
        (lighter + 0.05) / (darker + 0.05)
    }

    /// Lighten the color by the given amount.
    ///
    /// `amount` is in [0, 1] where 0 = no change, 1 = fully white.
    /// Operates in linear space by lerping toward white.
    pub fn lighten(&self, amount: f32) -> Color {
        self.lerp(&Color::WHITE, amount.clamp(0.0, 1.0))
    }

    /// Darken the color by the given amount.
    ///
    /// `amount` is in [0, 1] where 0 = no change, 1 = fully black.
    pub fn darken(&self, amount: f32) -> Color {
        self.lerp(&Color::BLACK, amount.clamp(0.0, 1.0))
    }

    /// Increase saturation by the given amount.
    ///
    /// Operates in HSV space. `amount` is in [0, 1].
    pub fn saturate(&self, amount: f32) -> Color {
        let (h, s, v) = self.to_hsv();
        let new_s = (s + amount).clamp(0.0, 1.0);
        let mut c = Color::from_hsv(h, new_s, v);
        c.a = self.a;
        c
    }

    /// Decrease saturation by the given amount.
    pub fn desaturate(&self, amount: f32) -> Color {
        let (h, s, v) = self.to_hsv();
        let new_s = (s - amount).clamp(0.0, 1.0);
        let mut c = Color::from_hsv(h, new_s, v);
        c.a = self.a;
        c
    }

    /// Return the complementary color (180 degrees opposite on the hue wheel).
    pub fn complement(&self) -> Color {
        let (h, s, v) = self.to_hsv();
        let new_h = (h + 180.0).rem_euclid(360.0);
        let mut c = Color::from_hsv(new_h, s, v);
        c.a = self.a;
        c
    }

    /// Convert to grayscale using perceptual luminance.
    pub fn grayscale(&self) -> Color {
        let l = self.luminance();
        Color {
            r: l,
            g: l,
            b: l,
            a: self.a,
        }
    }

    /// Invert the color (1 - component for each channel).
    pub fn invert(&self) -> Color {
        Color {
            r: 1.0 - self.r,
            g: 1.0 - self.g,
            b: 1.0 - self.b,
            a: self.a,
        }
    }

    /// Return the color with a new alpha value.
    #[inline]
    pub fn with_alpha(&self, a: f32) -> Color {
        Color { a, ..*self }
    }

    /// Clamp all components to [0, 1].
    pub fn clamped(&self) -> Color {
        Color {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }

    /// Convert to an array `[r, g, b, a]`.
    #[inline]
    pub fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Create from an array `[r, g, b, a]`.
    #[inline]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self {
            r: arr[0],
            g: arr[1],
            b: arr[2],
            a: arr[3],
        }
    }

    /// Multiply two colors component-wise (modulate).
    #[inline]
    pub fn multiply(&self, other: &Color) -> Color {
        Color {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
            a: self.a * other.a,
        }
    }

    /// Additive blend (screen).
    pub fn screen(&self, other: &Color) -> Color {
        Color {
            r: 1.0 - (1.0 - self.r) * (1.0 - other.r),
            g: 1.0 - (1.0 - self.g) * (1.0 - other.g),
            b: 1.0 - (1.0 - self.b) * (1.0 - other.b),
            a: self.a,
        }
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Color({:.3}, {:.3}, {:.3}, {:.3})",
            self.r, self.g, self.b, self.a
        )
    }
}

// ===========================================================================
// Gradient
// ===========================================================================

/// A color stop in a gradient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorStop {
    /// Position along the gradient, in [0, 1].
    pub position: f32,
    /// Color at this stop.
    pub color: Color,
}

impl ColorStop {
    /// Create a new color stop.
    pub fn new(position: f32, color: Color) -> Self {
        Self { position, color }
    }
}

/// A smooth color gradient defined by sorted color stops.
///
/// Supports linear interpolation between stops and can be sampled at any
/// position in [0, 1]. Useful for heat maps, terrain coloring, UI elements,
/// and particle color-over-lifetime curves.
#[derive(Debug, Clone)]
pub struct Gradient {
    /// Color stops, sorted by position.
    stops: Vec<ColorStop>,
}

impl Gradient {
    /// Create a gradient from a list of color stops.
    ///
    /// Stops are sorted by position. At least two stops are recommended.
    pub fn new(mut stops: Vec<ColorStop>) -> Self {
        stops.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap_or(std::cmp::Ordering::Equal));
        Self { stops }
    }

    /// Create a simple two-stop gradient from `start` to `end`.
    pub fn two_color(start: Color, end: Color) -> Self {
        Self::new(vec![
            ColorStop::new(0.0, start),
            ColorStop::new(1.0, end),
        ])
    }

    /// Create a three-stop gradient.
    pub fn three_color(start: Color, mid: Color, end: Color) -> Self {
        Self::new(vec![
            ColorStop::new(0.0, start),
            ColorStop::new(0.5, mid),
            ColorStop::new(1.0, end),
        ])
    }

    /// Sample the gradient at the given position `t` in [0, 1].
    ///
    /// Uses linear interpolation between the two enclosing stops.
    /// Values outside [0, 1] are clamped.
    pub fn sample(&self, t: f32) -> Color {
        if self.stops.is_empty() {
            return Color::BLACK;
        }
        if self.stops.len() == 1 {
            return self.stops[0].color;
        }

        let t = t.clamp(0.0, 1.0);

        // Before the first stop.
        if t <= self.stops[0].position {
            return self.stops[0].color;
        }

        // After the last stop.
        let last = self.stops.len() - 1;
        if t >= self.stops[last].position {
            return self.stops[last].color;
        }

        // Find the two enclosing stops.
        for i in 0..last {
            let s0 = &self.stops[i];
            let s1 = &self.stops[i + 1];
            if t >= s0.position && t <= s1.position {
                let range = s1.position - s0.position;
                if range < f32::EPSILON {
                    return s0.color;
                }
                let local_t = (t - s0.position) / range;
                return s0.color.lerp(&s1.color, local_t);
            }
        }

        self.stops[last].color
    }

    /// Add a color stop to the gradient.
    pub fn add_stop(&mut self, stop: ColorStop) {
        self.stops.push(stop);
        self.stops.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Number of color stops.
    pub fn stop_count(&self) -> usize {
        self.stops.len()
    }

    /// Get a reference to all stops.
    pub fn stops(&self) -> &[ColorStop] {
        &self.stops
    }

    /// Generate N evenly-spaced color samples from the gradient.
    pub fn sample_n(&self, count: usize) -> Vec<Color> {
        if count == 0 {
            return Vec::new();
        }
        if count == 1 {
            return vec![self.sample(0.5)];
        }
        (0..count)
            .map(|i| {
                let t = i as f32 / (count - 1) as f32;
                self.sample(t)
            })
            .collect()
    }

    /// Create a gradient from a predefined palette.
    pub fn from_colors(colors: &[Color]) -> Self {
        if colors.is_empty() {
            return Self::new(Vec::new());
        }
        let n = colors.len();
        let stops: Vec<ColorStop> = colors
            .iter()
            .enumerate()
            .map(|(i, &c)| ColorStop::new(if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 }, c))
            .collect();
        Self::new(stops)
    }
}

impl Default for Gradient {
    fn default() -> Self {
        Self::two_color(Color::BLACK, Color::WHITE)
    }
}

// ===========================================================================
// Color space conversion helpers
// ===========================================================================

/// Convert a single sRGB channel to linear.
///
/// Applies the inverse sRGB transfer function:
/// - For values <= 0.04045: linear = srgb / 12.92
/// - For values > 0.04045: linear = ((srgb + 0.055) / 1.055) ^ 2.4
#[inline]
pub fn srgb_to_linear(srgb: f32) -> f32 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert a single linear channel to sRGB.
///
/// Applies the sRGB transfer function:
/// - For values <= 0.0031308: srgb = linear * 12.92
/// - For values > 0.0031308: srgb = 1.055 * linear ^ (1/2.4) - 0.055
#[inline]
pub fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert HSV to RGB (all in [0, 1] range, hue in [0, 360)).
///
/// Uses the standard hexagonal model.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s < f32::EPSILON {
        return (v, v, v);
    }

    let h = h.rem_euclid(360.0);
    let h = h / 60.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        5 => (v, p, q),
        _ => (v, v, v),
    }
}

/// Convert RGB to HSV (hue in [0, 360), s and v in [0, 1]).
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;

    if delta < f32::EPSILON {
        return (0.0, 0.0, v);
    }

    let s = delta / max;

    let h = if (r - max).abs() < f32::EPSILON {
        (g - b) / delta
    } else if (g - max).abs() < f32::EPSILON {
        2.0 + (b - r) / delta
    } else {
        4.0 + (r - g) / delta
    };

    let h = (h * 60.0).rem_euclid(360.0);

    (h, s, v)
}

/// Convert HSL to RGB (hue in [0, 360), s and l in [0, 1]).
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s < f32::EPSILON {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h_norm = h / 360.0;

    let hue_to_rgb = |t: f32| -> f32 {
        let t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };

    (
        hue_to_rgb(h_norm + 1.0 / 3.0),
        hue_to_rgb(h_norm),
        hue_to_rgb(h_norm - 1.0 / 3.0),
    )
}

/// Convert RGB to HSL.
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if (max - min).abs() < f32::EPSILON {
        return (0.0, 0.0, l);
    }

    let delta = max - min;
    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    let h = if (r - max).abs() < f32::EPSILON {
        (g - b) / delta + if g < b { 6.0 } else { 0.0 }
    } else if (g - max).abs() < f32::EPSILON {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };

    let h = h * 60.0;
    (h, s, l)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_roundtrip() {
        for i in 0..=255 {
            let srgb = i as f32 / 255.0;
            let linear = srgb_to_linear(srgb);
            let back = linear_to_srgb(linear);
            assert!((srgb - back).abs() < 0.002, "Roundtrip failed for sRGB={srgb}: got {back}");
        }
    }

    #[test]
    fn hex_parse_6digit() {
        let c = Color::from_hex("#FF0000").unwrap();
        let [r, g, b, a] = c.to_rgba8();
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
        assert_eq!(a, 255);
    }

    #[test]
    fn hex_parse_without_hash() {
        let c = Color::from_hex("00FF00").unwrap();
        let [r, g, b, _] = c.to_rgba8();
        assert_eq!(r, 0);
        assert_eq!(g, 255);
        assert_eq!(b, 0);
    }

    #[test]
    fn hex_parse_3digit() {
        let c = Color::from_hex("#F00").unwrap();
        let [r, g, b, _] = c.to_rgba8();
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn hex_parse_8digit_with_alpha() {
        let c = Color::from_hex("#FF000080").unwrap();
        let [r, _g, _b, a] = c.to_rgba8();
        assert_eq!(r, 255);
        assert_eq!(a, 128);
    }

    #[test]
    fn hex_parse_invalid() {
        assert!(Color::from_hex("XYZ").is_none());
        assert!(Color::from_hex("#12345").is_none());
    }

    #[test]
    fn to_hex_roundtrip() {
        let c = Color::from_hex("#FF8040").unwrap();
        let hex = c.to_hex();
        assert_eq!(hex, "#FF8040");
    }

    #[test]
    fn hsv_roundtrip() {
        let original = Color::from_srgb(0.8, 0.3, 0.5);
        let (h, s, v) = original.to_hsv();
        let restored = Color::from_hsv(h, s, v);
        let (or, og, ob) = original.to_srgb();
        let (rr, rg, rb) = restored.to_srgb();
        assert!((or - rr).abs() < 0.02, "R mismatch: {or} vs {rr}");
        assert!((og - rg).abs() < 0.02, "G mismatch: {og} vs {rg}");
        assert!((ob - rb).abs() < 0.02, "B mismatch: {ob} vs {rb}");
    }

    #[test]
    fn hsl_roundtrip() {
        let original = Color::from_srgb(0.2, 0.7, 0.5);
        let (h, s, l) = original.to_hsl();
        let restored = Color::from_hsl(h, s, l);
        let (or, og, ob) = original.to_srgb();
        let (rr, rg, rb) = restored.to_srgb();
        assert!((or - rr).abs() < 0.02, "R: {or} vs {rr}");
        assert!((og - rg).abs() < 0.02, "G: {og} vs {rg}");
        assert!((ob - rb).abs() < 0.02, "B: {ob} vs {rb}");
    }

    #[test]
    fn luminance_black_white() {
        assert!((Color::BLACK.luminance() - 0.0).abs() < f32::EPSILON);
        assert!((Color::WHITE.luminance() - 1.0).abs() < 0.01);
    }

    #[test]
    fn contrast_ratio_black_white() {
        let ratio = Color::BLACK.contrast_ratio(&Color::WHITE);
        assert!((ratio - 21.0).abs() < 0.1, "Black/White contrast ratio should be ~21, got {ratio}");
    }

    #[test]
    fn contrast_ratio_same_color() {
        let ratio = Color::RED.contrast_ratio(&Color::RED);
        assert!((ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn lerp_endpoints() {
        let a = Color::RED;
        let b = Color::BLUE;
        let at_0 = a.lerp(&b, 0.0);
        let at_1 = a.lerp(&b, 1.0);
        assert!((at_0.r - a.r).abs() < f32::EPSILON);
        assert!((at_1.b - b.b).abs() < f32::EPSILON);
    }

    #[test]
    fn lerp_midpoint() {
        let a = Color::rgb(0.0, 0.0, 0.0);
        let b = Color::rgb(1.0, 1.0, 1.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.r - 0.5).abs() < f32::EPSILON);
        assert!((mid.g - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn lighten_darken() {
        let c = Color::from_srgb(0.5, 0.5, 0.5);
        let lighter = c.lighten(0.5);
        let darker = c.darken(0.5);
        assert!(lighter.luminance() > c.luminance());
        assert!(darker.luminance() < c.luminance());
    }

    #[test]
    fn complement_red_is_cyan() {
        let red = Color::from_hsv(0.0, 1.0, 1.0);
        let comp = red.complement();
        let (h, _s, _v) = comp.to_hsv();
        assert!((h - 180.0).abs() < 2.0, "Complement hue should be ~180, got {h}");
    }

    #[test]
    fn gradient_two_color() {
        let g = Gradient::two_color(Color::BLACK, Color::WHITE);
        let start = g.sample(0.0);
        let end = g.sample(1.0);
        let mid = g.sample(0.5);
        assert!((start.r - 0.0).abs() < f32::EPSILON);
        assert!((end.r - 1.0).abs() < f32::EPSILON);
        assert!((mid.r - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn gradient_three_color() {
        let g = Gradient::three_color(Color::RED, Color::GREEN, Color::BLUE);
        let at_0 = g.sample(0.0);
        let at_half = g.sample(0.5);
        let at_1 = g.sample(1.0);
        assert!((at_0.r - Color::RED.r).abs() < f32::EPSILON);
        assert!((at_half.g - Color::GREEN.g).abs() < f32::EPSILON);
        assert!((at_1.b - Color::BLUE.b).abs() < f32::EPSILON);
    }

    #[test]
    fn gradient_clamping() {
        let g = Gradient::two_color(Color::BLACK, Color::WHITE);
        let before = g.sample(-1.0);
        let after = g.sample(2.0);
        assert!((before.r - 0.0).abs() < f32::EPSILON);
        assert!((after.r - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn gradient_sample_n() {
        let g = Gradient::two_color(Color::BLACK, Color::WHITE);
        let samples = g.sample_n(5);
        assert_eq!(samples.len(), 5);
        assert!((samples[0].r - 0.0).abs() < f32::EPSILON);
        assert!((samples[4].r - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn color_grayscale() {
        let c = Color::rgb(0.5, 0.3, 0.8);
        let gray = c.grayscale();
        let l = c.luminance();
        assert!((gray.r - l).abs() < f32::EPSILON);
        assert!((gray.g - l).abs() < f32::EPSILON);
        assert!((gray.b - l).abs() < f32::EPSILON);
    }

    #[test]
    fn color_invert() {
        let c = Color::rgb(0.3, 0.6, 0.9);
        let inv = c.invert();
        assert!((inv.r - 0.7).abs() < f32::EPSILON);
        assert!((inv.g - 0.4).abs() < f32::EPSILON);
        assert!((inv.b - 0.1).abs() < 0.001);
    }

    #[test]
    fn color_multiply() {
        let a = Color::rgb(0.5, 1.0, 0.0);
        let b = Color::rgb(1.0, 0.5, 1.0);
        let m = a.multiply(&b);
        assert!((m.r - 0.5).abs() < f32::EPSILON);
        assert!((m.g - 0.5).abs() < f32::EPSILON);
        assert!((m.b - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn color_with_alpha() {
        let c = Color::RED.with_alpha(0.5);
        assert_eq!(c.r, Color::RED.r);
        assert!((c.a - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn named_colors_are_opaque() {
        let colors = [
            Color::RED, Color::GREEN, Color::BLUE, Color::WHITE, Color::BLACK,
            Color::YELLOW, Color::CYAN, Color::MAGENTA, Color::ORANGE, Color::PURPLE,
            Color::GRAY, Color::PINK, Color::LIME, Color::NAVY, Color::TEAL,
            Color::MAROON, Color::OLIVE, Color::GOLD, Color::SKY_BLUE, Color::CORAL,
            Color::SALMON, Color::INDIGO, Color::TURQUOISE, Color::VIOLET, Color::CRIMSON,
            Color::CHARTREUSE, Color::BROWN, Color::SILVER, Color::DARK_GRAY, Color::LIGHT_GRAY,
            Color::AQUA,
        ];
        for (i, c) in colors.iter().enumerate() {
            assert!((c.a - 1.0).abs() < f32::EPSILON, "Color at index {i} should be opaque");
        }
    }
}
