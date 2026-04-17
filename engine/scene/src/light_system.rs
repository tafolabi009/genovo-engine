//! Scene lighting system for the Genovo engine.
//!
//! Provides a complete lighting subsystem with support for multiple light types,
//! physical light units, colour temperature, shadow configuration, light layers,
//! and per-camera light culling.
//!
//! # Light types
//!
//! - **Ambient** -- uniform global illumination.
//! - **Directional** -- sun/moon-like infinite-distance light.
//! - **Point** -- omnidirectional local light with attenuation.
//! - **Spot** -- cone-shaped local light with inner/outer angles.
//! - **Area** (rect/disc) -- physically accurate area lights.
//!
//! # Physical units
//!
//! The system uses physically-based light intensity units:
//! - Lumens (lm) for point and spot lights.
//! - Candela (cd) for directional lights.
//! - Lux (lx) for ambient / environment lights.
//! Color temperature (Kelvin) can be converted to RGB.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

/// Linear RGB colour (pre-multiplied, HDR-safe).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl LinearColor {
    /// Pure white.
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
    };

    /// Pure black.
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };

    /// Creates a new linear colour.
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Creates a colour from sRGB (0..255).
    pub fn from_srgb(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: srgb_to_linear(r as f32 / 255.0),
            g: srgb_to_linear(g as f32 / 255.0),
            b: srgb_to_linear(b as f32 / 255.0),
        }
    }

    /// Multiplies the colour by a scalar.
    pub fn scaled(&self, s: f32) -> Self {
        Self {
            r: self.r * s,
            g: self.g * s,
            b: self.b * s,
        }
    }

    /// Component-wise multiply with another colour.
    pub fn multiply(&self, other: &Self) -> Self {
        Self {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }

    /// Linearly interpolates between two colours.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }

    /// Returns the luminance of the colour (Rec. 709 coefficients).
    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Returns the colour as a 3-element array.
    pub fn to_array(&self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// Returns the colour as a 4-element array with alpha = 1.
    pub fn to_array4(&self) -> [f32; 4] {
        [self.r, self.g, self.b, 1.0]
    }

    /// Clamps all channels to the 0..1 range (LDR).
    pub fn clamped(&self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }

    /// Checks if the colour is approximately black.
    pub fn is_black(&self, epsilon: f32) -> bool {
        self.r.abs() < epsilon && self.g.abs() < epsilon && self.b.abs() < epsilon
    }
}

impl Default for LinearColor {
    fn default() -> Self {
        Self::WHITE
    }
}

/// Converts an sRGB component to linear space.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts a linear component to sRGB space.
#[allow(dead_code)]
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

// ---------------------------------------------------------------------------
// Color temperature conversion
// ---------------------------------------------------------------------------

/// Converts a colour temperature in Kelvin to a linear RGB colour.
///
/// Uses the Tanner Helland approximation which is accurate for the range
/// 1000 K .. 40000 K. Commonly used temperatures:
/// - Candle: ~1800 K
/// - Incandescent bulb: ~2700 K
/// - Halogen: ~3200 K
/// - Daylight: ~5500 K
/// - Overcast sky: ~6500 K
/// - Clear blue sky: ~10000 K
pub fn color_temperature_to_rgb(kelvin: f32) -> LinearColor {
    let temp = (kelvin / 100.0).clamp(10.0, 400.0);

    let r;
    let g;
    let b;

    // Red channel.
    if temp <= 66.0 {
        r = 1.0;
    } else {
        let x = temp - 60.0;
        r = (329.698727446 * x.powf(-0.1332047592) / 255.0).clamp(0.0, 1.0);
    }

    // Green channel.
    if temp <= 66.0 {
        let x = temp;
        g = (99.4708025861 * x.ln() - 161.1195681661).clamp(0.0, 255.0) / 255.0;
    } else {
        let x = temp - 60.0;
        g = (288.1221695283 * x.powf(-0.0755148492) / 255.0).clamp(0.0, 1.0);
    }

    // Blue channel.
    if temp >= 66.0 {
        b = 1.0;
    } else if temp <= 19.0 {
        b = 0.0;
    } else {
        let x = temp - 10.0;
        b = (138.5177312231 * x.ln() - 305.0447927307).clamp(0.0, 255.0) / 255.0;
    }

    LinearColor::new(r, g, b)
}

/// Common colour temperature presets.
pub mod color_temperature {
    /// Candle flame (~1800 K).
    pub const CANDLE: f32 = 1800.0;
    /// Warm white incandescent bulb (~2700 K).
    pub const INCANDESCENT: f32 = 2700.0;
    /// Sunrise/sunset (~3000 K).
    pub const SUNRISE: f32 = 3000.0;
    /// Halogen lamp (~3200 K).
    pub const HALOGEN: f32 = 3200.0;
    /// Fluorescent white (~4000 K).
    pub const FLUORESCENT: f32 = 4000.0;
    /// Noon daylight (~5500 K).
    pub const DAYLIGHT: f32 = 5500.0;
    /// Overcast sky (~6500 K).
    pub const OVERCAST: f32 = 6500.0;
    /// Shade (~7500 K).
    pub const SHADE: f32 = 7500.0;
    /// Clear blue sky (~10000 K).
    pub const BLUE_SKY: f32 = 10000.0;
}

// ---------------------------------------------------------------------------
// Light intensity units
// ---------------------------------------------------------------------------

/// Physical light intensity with unit specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightIntensity {
    /// Luminous flux (lumens). Used for point and spot lights.
    Lumens(f32),
    /// Luminous intensity (candela). Used for directional lights.
    Candela(f32),
    /// Illuminance (lux). Used for ambient/environment light.
    Lux(f32),
    /// Unitless intensity multiplier (legacy / artistic mode).
    Unitless(f32),
}

impl LightIntensity {
    /// Returns the raw numeric value regardless of unit.
    pub fn value(&self) -> f32 {
        match self {
            Self::Lumens(v) | Self::Candela(v) | Self::Lux(v) | Self::Unitless(v) => *v,
        }
    }

    /// Converts lumens to candela (assuming isotropic point light).
    /// Candela = Lumens / (4 * pi)
    pub fn lumens_to_candela(lumens: f32) -> f32 {
        lumens / (4.0 * std::f32::consts::PI)
    }

    /// Converts candela to lumens (assuming isotropic point light).
    pub fn candela_to_lumens(candela: f32) -> f32 {
        candela * 4.0 * std::f32::consts::PI
    }

    /// Converts lumens to lux at a given distance.
    /// Lux = Lumens / (4 * pi * distance^2)
    pub fn lumens_to_lux(lumens: f32, distance: f32) -> f32 {
        if distance <= 0.0 {
            return f32::INFINITY;
        }
        lumens / (4.0 * std::f32::consts::PI * distance * distance)
    }

    /// Converts lux to lumens over a given area (m^2).
    pub fn lux_to_lumens(lux: f32, area: f32) -> f32 {
        lux * area
    }

    /// Whether this is zero intensity.
    pub fn is_zero(&self) -> bool {
        self.value().abs() < 1e-9
    }
}

impl Default for LightIntensity {
    fn default() -> Self {
        Self::Unitless(1.0)
    }
}

impl fmt::Display for LightIntensity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lumens(v) => write!(f, "{:.1} lm", v),
            Self::Candela(v) => write!(f, "{:.1} cd", v),
            Self::Lux(v) => write!(f, "{:.1} lx", v),
            Self::Unitless(v) => write!(f, "{:.2}", v),
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow configuration
// ---------------------------------------------------------------------------

/// Shadow map resolution presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowResolution {
    /// 256x256
    Low,
    /// 512x512
    Medium,
    /// 1024x1024
    High,
    /// 2048x2048
    VeryHigh,
    /// 4096x4096
    Ultra,
    /// Custom resolution.
    Custom(u32),
}

impl ShadowResolution {
    /// Returns the pixel dimension.
    pub fn pixels(&self) -> u32 {
        match self {
            Self::Low => 256,
            Self::Medium => 512,
            Self::High => 1024,
            Self::VeryHigh => 2048,
            Self::Ultra => 4096,
            Self::Custom(v) => *v,
        }
    }
}

impl Default for ShadowResolution {
    fn default() -> Self {
        Self::High
    }
}

/// Shadow filtering modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowFilter {
    /// Hard shadows (nearest sampling).
    Hard,
    /// PCF (percentage closer filtering) with N samples.
    PCF { samples: u32 },
    /// PCSS (percentage closer soft shadows).
    PCSS { search_samples: u32, filter_samples: u32 },
    /// VSM (variance shadow mapping).
    VSM,
    /// ESM (exponential shadow mapping).
    ESM { exponent: u32 },
}

impl Default for ShadowFilter {
    fn default() -> Self {
        Self::PCF { samples: 4 }
    }
}

/// Configuration for shadow casting on a light.
#[derive(Debug, Clone)]
pub struct ShadowConfig {
    /// Whether shadows are enabled for this light.
    pub enabled: bool,
    /// Shadow map resolution.
    pub resolution: ShadowResolution,
    /// Shadow filtering mode.
    pub filter: ShadowFilter,
    /// Near plane for the shadow camera.
    pub near: f32,
    /// Far plane / max shadow distance.
    pub far: f32,
    /// Shadow bias to prevent self-shadowing (depth bias).
    pub depth_bias: f32,
    /// Normal bias to prevent shadow acne.
    pub normal_bias: f32,
    /// Number of cascades for directional light CSM.
    pub cascade_count: u32,
    /// Cascade split lambda (0 = linear, 1 = logarithmic).
    pub cascade_lambda: f32,
    /// Shadow strength (0 = no shadow, 1 = full shadow).
    pub strength: f32,
    /// Maximum distance from the camera at which shadows are rendered.
    pub max_distance: f32,
    /// Fade distance before max_distance where shadows begin to fade.
    pub fade_distance: f32,
}

impl ShadowConfig {
    /// Creates a new shadow configuration with defaults.
    pub fn new() -> Self {
        Self {
            enabled: true,
            resolution: ShadowResolution::High,
            filter: ShadowFilter::PCF { samples: 4 },
            near: 0.1,
            far: 500.0,
            depth_bias: 0.005,
            normal_bias: 0.02,
            cascade_count: 4,
            cascade_lambda: 0.75,
            strength: 1.0,
            max_distance: 200.0,
            fade_distance: 20.0,
        }
    }

    /// Shadow config for a directional light (CSM).
    pub fn directional() -> Self {
        Self {
            cascade_count: 4,
            resolution: ShadowResolution::VeryHigh,
            max_distance: 500.0,
            ..Self::new()
        }
    }

    /// Shadow config for a point light (cubemap).
    pub fn point() -> Self {
        Self {
            cascade_count: 1,
            resolution: ShadowResolution::Medium,
            max_distance: 50.0,
            ..Self::new()
        }
    }

    /// Shadow config for a spot light.
    pub fn spot() -> Self {
        Self {
            cascade_count: 1,
            resolution: ShadowResolution::High,
            max_distance: 100.0,
            ..Self::new()
        }
    }

    /// Computes cascade split distances for CSM.
    pub fn compute_cascade_splits(&self, camera_near: f32) -> Vec<f32> {
        let mut splits = Vec::with_capacity(self.cascade_count as usize + 1);
        splits.push(camera_near);

        for i in 1..=self.cascade_count {
            let p = i as f32 / self.cascade_count as f32;

            // Logarithmic split.
            let log_split = camera_near * (self.far / camera_near).powf(p);
            // Linear split.
            let lin_split = camera_near + (self.far - camera_near) * p;
            // Blend based on lambda.
            let split = self.cascade_lambda * log_split
                + (1.0 - self.cascade_lambda) * lin_split;

            splits.push(split);
        }

        splits
    }

    /// Shadow fade factor at a given distance from the camera.
    pub fn shadow_fade(&self, distance: f32) -> f32 {
        if distance >= self.max_distance {
            return 0.0;
        }
        let fade_start = self.max_distance - self.fade_distance;
        if distance <= fade_start {
            return self.strength;
        }
        let t = (distance - fade_start) / self.fade_distance;
        self.strength * (1.0 - t.clamp(0.0, 1.0))
    }
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Light layers
// ---------------------------------------------------------------------------

/// A bitmask that determines which objects a light affects.
///
/// Both lights and renderable objects have layer masks. A light affects an
/// object only if `(light.layers & object.layers) != 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightLayerMask(pub u32);

impl LightLayerMask {
    /// All layers (default -- light affects everything).
    pub const ALL: Self = Self(0xFFFF_FFFF);
    /// No layers (light affects nothing).
    pub const NONE: Self = Self(0);

    /// Creates a mask with a single layer.
    pub fn layer(index: u8) -> Self {
        assert!(index < 32, "Layer index must be 0..31");
        Self(1 << index)
    }

    /// Creates a mask with multiple layers.
    pub fn layers(indices: &[u8]) -> Self {
        let mut mask = 0u32;
        for &idx in indices {
            assert!(idx < 32, "Layer index must be 0..31");
            mask |= 1 << idx;
        }
        Self(mask)
    }

    /// Whether a specific layer is set.
    pub fn has_layer(&self, index: u8) -> bool {
        (self.0 & (1 << index)) != 0
    }

    /// Whether this mask overlaps with another.
    pub fn overlaps(&self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// Adds a layer to the mask.
    pub fn add_layer(&mut self, index: u8) {
        self.0 |= 1 << index;
    }

    /// Removes a layer from the mask.
    pub fn remove_layer(&mut self, index: u8) {
        self.0 &= !(1 << index);
    }

    /// Toggles a layer in the mask.
    pub fn toggle_layer(&mut self, index: u8) {
        self.0 ^= 1 << index;
    }

    /// Returns the number of layers set.
    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }
}

impl Default for LightLayerMask {
    fn default() -> Self {
        Self::ALL
    }
}

// ---------------------------------------------------------------------------
// Vec3 helper (self-contained)
// ---------------------------------------------------------------------------

/// Minimal 3D vector for self-contained compilation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const UP: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const DOWN: Self = Self {
        x: 0.0,
        y: -1.0,
        z: 0.0,
    };
    pub const FORWARD: Self = Self {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-9 {
            return Self::ZERO;
        }
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn distance_squared(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::ZERO
    }
}

// ---------------------------------------------------------------------------
// Light types
// ---------------------------------------------------------------------------

/// The type of a light source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    /// Ambient light: uniform, directionless illumination.
    Ambient,
    /// Directional light: parallel rays from infinitely far away (sun).
    Directional,
    /// Point light: omnidirectional with distance attenuation.
    Point,
    /// Spot light: cone-shaped with distance attenuation.
    Spot,
    /// Rectangular area light.
    AreaRect,
    /// Disc-shaped area light.
    AreaDisc,
}

impl fmt::Display for LightType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ambient => write!(f, "Ambient"),
            Self::Directional => write!(f, "Directional"),
            Self::Point => write!(f, "Point"),
            Self::Spot => write!(f, "Spot"),
            Self::AreaRect => write!(f, "Area (Rect)"),
            Self::AreaDisc => write!(f, "Area (Disc)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Light component
// ---------------------------------------------------------------------------

/// A light component that can be attached to an entity.
#[derive(Debug, Clone)]
pub struct LightComponent {
    /// Unique id for this light instance.
    pub id: u64,
    /// The type of light.
    pub light_type: LightType,
    /// Light colour in linear space.
    pub color: LinearColor,
    /// Light intensity with physical units.
    pub intensity: LightIntensity,
    /// Optional colour temperature in Kelvin (overrides `color` when set).
    pub color_temperature: Option<f32>,
    /// Direction the light is pointing (for directional / spot lights).
    pub direction: Vec3,
    /// Position of the light in world space (updated by transform system).
    pub position: Vec3,
    /// Maximum range for point / spot lights. Beyond this distance the
    /// light has zero contribution.
    pub range: f32,
    /// Inner cone angle in radians (spot lights).
    pub inner_angle: f32,
    /// Outer cone angle in radians (spot lights).
    pub outer_angle: f32,
    /// Area light dimensions: (width, height) for rect, (radius, _) for disc.
    pub area_size: (f32, f32),
    /// Shadow configuration.
    pub shadow: ShadowConfig,
    /// Layer mask -- determines which objects this light affects.
    pub layers: LightLayerMask,
    /// Whether the light is enabled.
    pub enabled: bool,
    /// Whether the light affects specular highlights.
    pub affect_specular: bool,
    /// Whether the light is included in baked lightmaps.
    pub baked: bool,
    /// Cookie texture identifier (optional projected texture).
    pub cookie: Option<String>,
    /// Volumetric scattering intensity (0 = none).
    pub volumetric_intensity: f32,
    /// Indirect lighting multiplier (for GI).
    pub indirect_multiplier: f32,
    /// Priority for light culling (higher = more important).
    pub priority: i32,
}

impl LightComponent {
    /// Creates a new ambient light.
    pub fn ambient(color: LinearColor, intensity: f32) -> Self {
        Self {
            id: next_light_id(),
            light_type: LightType::Ambient,
            color,
            intensity: LightIntensity::Lux(intensity),
            color_temperature: None,
            direction: Vec3::DOWN,
            position: Vec3::ZERO,
            range: f32::INFINITY,
            inner_angle: 0.0,
            outer_angle: 0.0,
            area_size: (0.0, 0.0),
            shadow: ShadowConfig { enabled: false, ..ShadowConfig::new() },
            layers: LightLayerMask::ALL,
            enabled: true,
            affect_specular: false,
            baked: true,
            cookie: None,
            volumetric_intensity: 0.0,
            indirect_multiplier: 1.0,
            priority: 0,
        }
    }

    /// Creates a new directional light.
    pub fn directional(direction: Vec3, color: LinearColor, intensity: f32) -> Self {
        Self {
            id: next_light_id(),
            light_type: LightType::Directional,
            color,
            intensity: LightIntensity::Candela(intensity),
            color_temperature: None,
            direction: direction.normalized(),
            position: Vec3::ZERO,
            range: f32::INFINITY,
            inner_angle: 0.0,
            outer_angle: 0.0,
            area_size: (0.0, 0.0),
            shadow: ShadowConfig::directional(),
            layers: LightLayerMask::ALL,
            enabled: true,
            affect_specular: true,
            baked: false,
            cookie: None,
            volumetric_intensity: 0.0,
            indirect_multiplier: 1.0,
            priority: 100,
        }
    }

    /// Creates a new point light.
    pub fn point(position: Vec3, color: LinearColor, lumens: f32, range: f32) -> Self {
        Self {
            id: next_light_id(),
            light_type: LightType::Point,
            color,
            intensity: LightIntensity::Lumens(lumens),
            color_temperature: None,
            direction: Vec3::DOWN,
            position,
            range,
            inner_angle: 0.0,
            outer_angle: 0.0,
            area_size: (0.0, 0.0),
            shadow: ShadowConfig::point(),
            layers: LightLayerMask::ALL,
            enabled: true,
            affect_specular: true,
            baked: false,
            cookie: None,
            volumetric_intensity: 0.0,
            indirect_multiplier: 1.0,
            priority: 50,
        }
    }

    /// Creates a new spot light.
    pub fn spot(
        position: Vec3,
        direction: Vec3,
        color: LinearColor,
        lumens: f32,
        range: f32,
        inner_angle_deg: f32,
        outer_angle_deg: f32,
    ) -> Self {
        Self {
            id: next_light_id(),
            light_type: LightType::Spot,
            color,
            intensity: LightIntensity::Lumens(lumens),
            color_temperature: None,
            direction: direction.normalized(),
            position,
            range,
            inner_angle: inner_angle_deg.to_radians(),
            outer_angle: outer_angle_deg.to_radians(),
            area_size: (0.0, 0.0),
            shadow: ShadowConfig::spot(),
            layers: LightLayerMask::ALL,
            enabled: true,
            affect_specular: true,
            baked: false,
            cookie: None,
            volumetric_intensity: 0.0,
            indirect_multiplier: 1.0,
            priority: 50,
        }
    }

    /// Creates a rectangular area light.
    pub fn area_rect(
        position: Vec3,
        direction: Vec3,
        width: f32,
        height: f32,
        color: LinearColor,
        lumens: f32,
    ) -> Self {
        Self {
            id: next_light_id(),
            light_type: LightType::AreaRect,
            color,
            intensity: LightIntensity::Lumens(lumens),
            color_temperature: None,
            direction: direction.normalized(),
            position,
            range: 20.0,
            inner_angle: 0.0,
            outer_angle: std::f32::consts::PI,
            area_size: (width, height),
            shadow: ShadowConfig { enabled: false, ..ShadowConfig::new() },
            layers: LightLayerMask::ALL,
            enabled: true,
            affect_specular: true,
            baked: true,
            cookie: None,
            volumetric_intensity: 0.0,
            indirect_multiplier: 1.0,
            priority: 30,
        }
    }

    /// Creates a disc-shaped area light.
    pub fn area_disc(
        position: Vec3,
        direction: Vec3,
        radius: f32,
        color: LinearColor,
        lumens: f32,
    ) -> Self {
        Self {
            id: next_light_id(),
            light_type: LightType::AreaDisc,
            color,
            intensity: LightIntensity::Lumens(lumens),
            color_temperature: None,
            direction: direction.normalized(),
            position,
            range: 20.0,
            inner_angle: 0.0,
            outer_angle: std::f32::consts::PI,
            area_size: (radius, 0.0),
            shadow: ShadowConfig { enabled: false, ..ShadowConfig::new() },
            layers: LightLayerMask::ALL,
            enabled: true,
            affect_specular: true,
            baked: true,
            cookie: None,
            volumetric_intensity: 0.0,
            indirect_multiplier: 1.0,
            priority: 30,
        }
    }

    /// Returns the effective colour, accounting for colour temperature.
    pub fn effective_color(&self) -> LinearColor {
        match self.color_temperature {
            Some(kelvin) => {
                let temp_color = color_temperature_to_rgb(kelvin);
                self.color.multiply(&temp_color)
            }
            None => self.color,
        }
    }

    /// Sets the colour temperature and clears the manual colour.
    pub fn set_color_temperature(&mut self, kelvin: f32) {
        self.color_temperature = Some(kelvin.clamp(1000.0, 40000.0));
        self.color = LinearColor::WHITE;
    }

    /// Clears the colour temperature, using the manual colour instead.
    pub fn clear_color_temperature(&mut self) {
        self.color_temperature = None;
    }

    /// Computes distance attenuation for point / spot lights.
    ///
    /// Uses the UE4-style inverse-square falloff with a smooth cutoff at range.
    pub fn attenuation(&self, distance: f32) -> f32 {
        if distance >= self.range {
            return 0.0;
        }
        // Inverse-square falloff.
        let d2 = distance * distance + 1.0; // +1 avoids singularity at d=0.
        let falloff = 1.0 / d2;

        // Smooth cutoff near max range.
        let ratio = distance / self.range;
        let window = (1.0 - ratio * ratio).max(0.0);
        let window = window * window;

        falloff * window
    }

    /// Computes the spot light angular attenuation.
    ///
    /// Returns 1.0 inside the inner cone, 0.0 outside the outer cone,
    /// and smoothly interpolates between.
    pub fn spot_attenuation(&self, angle_to_axis: f32) -> f32 {
        if self.light_type != LightType::Spot {
            return 1.0;
        }

        let cos_angle = angle_to_axis.cos();
        let cos_inner = self.inner_angle.cos();
        let cos_outer = self.outer_angle.cos();

        if cos_angle >= cos_inner {
            1.0
        } else if cos_angle <= cos_outer {
            0.0
        } else {
            let t = (cos_angle - cos_outer) / (cos_inner - cos_outer);
            t * t
        }
    }

    /// Returns the effective intensity value for the light.
    pub fn effective_intensity(&self) -> f32 {
        self.intensity.value()
    }

    /// Whether the light produces any visible illumination.
    pub fn is_contributing(&self) -> bool {
        self.enabled && !self.intensity.is_zero() && !self.effective_color().is_black(1e-6)
    }

    /// Checks whether this light can affect an object with the given layer mask.
    pub fn affects_layer(&self, object_layers: LightLayerMask) -> bool {
        self.layers.overlaps(object_layers)
    }
}

/// Monotonically increasing light id generator.
fn next_light_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Light culling
// ---------------------------------------------------------------------------

/// Simplified frustum for light culling.
#[derive(Debug, Clone)]
pub struct CameraFrustum {
    /// Camera position.
    pub position: Vec3,
    /// Camera forward direction.
    pub forward: Vec3,
    /// Camera up direction.
    pub up: Vec3,
    /// Near plane distance.
    pub near: f32,
    /// Far plane distance.
    pub far: f32,
    /// Horizontal field of view in radians.
    pub fov_x: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
}

impl CameraFrustum {
    /// Creates a perspective frustum.
    pub fn perspective(
        position: Vec3,
        forward: Vec3,
        up: Vec3,
        fov_y_deg: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let fov_y = fov_y_deg.to_radians();
        let fov_x = 2.0 * ((fov_y * 0.5).tan() * aspect).atan();
        Self {
            position,
            forward: forward.normalized(),
            up: up.normalized(),
            near,
            far,
            fov_x,
            fov_y,
        }
    }

    /// Tests whether a sphere (position + radius) is inside the frustum.
    /// Uses a simplified cone test for performance.
    pub fn sphere_in_frustum(&self, center: Vec3, radius: f32) -> bool {
        let to_center = Vec3::new(
            center.x - self.position.x,
            center.y - self.position.y,
            center.z - self.position.z,
        );
        let dist_along_forward = to_center.dot(&self.forward);

        // Behind the camera.
        if dist_along_forward + radius < self.near {
            return false;
        }
        // Beyond far plane.
        if dist_along_forward - radius > self.far {
            return false;
        }

        // Cone test (simplified -- uses the wider FOV).
        let half_angle = self.fov_x.max(self.fov_y) * 0.5;
        let cone_radius = dist_along_forward * half_angle.tan();
        let perp_dist_sq = to_center.length_squared() - dist_along_forward * dist_along_forward;
        let max_dist = cone_radius + radius;

        perp_dist_sq <= max_dist * max_dist
    }
}

/// Result of culling a single light.
#[derive(Debug, Clone)]
pub struct CulledLight {
    /// The original light id.
    pub light_id: u64,
    /// The light type.
    pub light_type: LightType,
    /// Screen-space importance metric (used for sorting / budget).
    pub importance: f32,
    /// Whether the light casts shadows in the current view.
    pub cast_shadows: bool,
    /// Effective colour * intensity for the shader.
    pub final_color: [f32; 3],
    /// Position (for local lights).
    pub position: Vec3,
    /// Direction (for directional / spot).
    pub direction: Vec3,
    /// Range (for local lights).
    pub range: f32,
    /// Spot angles (inner, outer) in radians.
    pub spot_angles: (f32, f32),
}

/// Performs light culling against a camera frustum.
///
/// Returns a sorted list of lights visible from the given camera,
/// filtered by layer mask and frustum test. The list is sorted by
/// importance (most important lights first).
pub fn cull_lights(
    lights: &[LightComponent],
    frustum: &CameraFrustum,
    camera_layers: LightLayerMask,
    max_lights: usize,
) -> Vec<CulledLight> {
    let mut culled = Vec::new();

    for light in lights {
        // Skip disabled lights.
        if !light.is_contributing() {
            continue;
        }

        // Layer test.
        if !light.affects_layer(camera_layers) {
            continue;
        }

        // Frustum test (directional and ambient lights always pass).
        match light.light_type {
            LightType::Ambient | LightType::Directional => {}
            LightType::Point | LightType::AreaRect | LightType::AreaDisc => {
                if !frustum.sphere_in_frustum(light.position, light.range) {
                    continue;
                }
            }
            LightType::Spot => {
                // Use the spot's bounding sphere.
                if !frustum.sphere_in_frustum(light.position, light.range) {
                    continue;
                }
            }
        }

        // Compute importance metric.
        let importance = compute_light_importance(light, &frustum.position);

        let effective = light.effective_color().scaled(light.effective_intensity());

        culled.push(CulledLight {
            light_id: light.id,
            light_type: light.light_type,
            importance,
            cast_shadows: light.shadow.enabled,
            final_color: effective.to_array(),
            position: light.position,
            direction: light.direction,
            range: light.range,
            spot_angles: (light.inner_angle, light.outer_angle),
        });
    }

    // Sort by importance (descending).
    culled.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate to budget.
    culled.truncate(max_lights);

    culled
}

/// Computes a screen-space importance metric for a light.
///
/// Directional/ambient lights get high importance; local lights are rated
/// by intensity / distance^2.
fn compute_light_importance(light: &LightComponent, camera_pos: &Vec3) -> f32 {
    match light.light_type {
        LightType::Ambient => 1000.0 + light.priority as f32,
        LightType::Directional => 500.0 + light.priority as f32,
        _ => {
            let dist_sq = camera_pos.distance_squared(&light.position).max(1.0);
            let base = light.effective_intensity() / dist_sq;
            base + light.priority as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Light environment (scene-level ambient settings)
// ---------------------------------------------------------------------------

/// Scene-wide lighting environment settings.
#[derive(Debug, Clone)]
pub struct LightEnvironment {
    /// Ambient light color and intensity.
    pub ambient_color: LinearColor,
    /// Ambient intensity in lux.
    pub ambient_intensity: f32,
    /// Sky colour for gradient ambient (top hemisphere).
    pub sky_color: LinearColor,
    /// Ground colour for gradient ambient (bottom hemisphere).
    pub ground_color: LinearColor,
    /// Ambient mode: flat, gradient, or cubemap.
    pub ambient_mode: AmbientMode,
    /// Environment map / reflection cubemap identifier.
    pub environment_map: Option<String>,
    /// Environment map intensity multiplier.
    pub environment_intensity: f32,
    /// Fog colour.
    pub fog_color: LinearColor,
    /// Fog density.
    pub fog_density: f32,
    /// Fog start distance.
    pub fog_start: f32,
    /// Fog end distance.
    pub fog_end: f32,
    /// Whether fog is enabled.
    pub fog_enabled: bool,
    /// Exposure compensation (EV).
    pub exposure_compensation: f32,
}

/// How ambient light is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbientMode {
    /// Flat colour -- single colour applied uniformly.
    Flat,
    /// Gradient -- sky colour above, ground colour below, blended.
    Gradient,
    /// Cubemap -- sampled from an environment cubemap.
    Cubemap,
}

impl LightEnvironment {
    /// Creates a default outdoor lighting environment.
    pub fn outdoor_day() -> Self {
        Self {
            ambient_color: LinearColor::new(0.3, 0.35, 0.4),
            ambient_intensity: 1.0,
            sky_color: LinearColor::new(0.4, 0.5, 0.8),
            ground_color: LinearColor::new(0.2, 0.15, 0.1),
            ambient_mode: AmbientMode::Gradient,
            environment_map: None,
            environment_intensity: 1.0,
            fog_color: LinearColor::new(0.7, 0.75, 0.8),
            fog_density: 0.001,
            fog_start: 50.0,
            fog_end: 500.0,
            fog_enabled: true,
            exposure_compensation: 0.0,
        }
    }

    /// Creates a default indoor lighting environment.
    pub fn indoor() -> Self {
        Self {
            ambient_color: LinearColor::new(0.15, 0.15, 0.15),
            ambient_intensity: 0.5,
            sky_color: LinearColor::new(0.15, 0.15, 0.2),
            ground_color: LinearColor::new(0.1, 0.1, 0.08),
            ambient_mode: AmbientMode::Flat,
            environment_map: None,
            environment_intensity: 0.5,
            fog_color: LinearColor::BLACK,
            fog_density: 0.0,
            fog_start: 0.0,
            fog_end: 100.0,
            fog_enabled: false,
            exposure_compensation: 0.0,
        }
    }

    /// Creates a night-time environment.
    pub fn night() -> Self {
        Self {
            ambient_color: LinearColor::new(0.02, 0.03, 0.05),
            ambient_intensity: 0.1,
            sky_color: LinearColor::new(0.01, 0.02, 0.05),
            ground_color: LinearColor::new(0.01, 0.01, 0.01),
            ambient_mode: AmbientMode::Gradient,
            environment_map: None,
            environment_intensity: 0.1,
            fog_color: LinearColor::new(0.05, 0.05, 0.08),
            fog_density: 0.005,
            fog_start: 10.0,
            fog_end: 200.0,
            fog_enabled: true,
            exposure_compensation: 2.0,
        }
    }

    /// Computes the ambient light contribution for a surface normal.
    pub fn sample_ambient(&self, surface_normal: &Vec3) -> LinearColor {
        match self.ambient_mode {
            AmbientMode::Flat => self.ambient_color.scaled(self.ambient_intensity),
            AmbientMode::Gradient => {
                // Blend between ground and sky based on normal's Y component.
                let t = (surface_normal.y * 0.5 + 0.5).clamp(0.0, 1.0);
                let blended = self.ground_color.lerp(&self.sky_color, t);
                blended.scaled(self.ambient_intensity)
            }
            AmbientMode::Cubemap => {
                // In a real implementation, this would sample the cubemap.
                // Fallback to gradient.
                let t = (surface_normal.y * 0.5 + 0.5).clamp(0.0, 1.0);
                let blended = self.ground_color.lerp(&self.sky_color, t);
                blended.scaled(self.ambient_intensity * self.environment_intensity)
            }
        }
    }

    /// Computes fog factor at a given distance.
    pub fn fog_factor(&self, distance: f32) -> f32 {
        if !self.fog_enabled || self.fog_density <= 0.0 {
            return 0.0;
        }
        // Exponential fog.
        let exp_fog = (-self.fog_density * distance).exp();
        // Linear fog.
        let linear_fog = if self.fog_end > self.fog_start {
            1.0 - ((self.fog_end - distance) / (self.fog_end - self.fog_start)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Use the stronger fog.
        (1.0 - exp_fog).max(linear_fog).clamp(0.0, 1.0)
    }
}

impl Default for LightEnvironment {
    fn default() -> Self {
        Self::outdoor_day()
    }
}

// ---------------------------------------------------------------------------
// LightSystem -- per-frame update and management
// ---------------------------------------------------------------------------

/// The per-frame lighting system that manages all lights in the scene.
pub struct LightSystem {
    /// All registered lights.
    lights: Vec<LightComponent>,
    /// Light id -> index mapping for fast lookup.
    light_index: HashMap<u64, usize>,
    /// The scene-level environment settings.
    environment: LightEnvironment,
    /// Maximum number of lights to send to the GPU per frame.
    max_visible_lights: usize,
    /// Whether to sort lights by importance each frame.
    sort_by_importance: bool,
    /// Named light layer descriptions.
    layer_names: HashMap<u8, String>,
}

impl LightSystem {
    /// Creates a new light system.
    pub fn new() -> Self {
        Self {
            lights: Vec::new(),
            light_index: HashMap::new(),
            environment: LightEnvironment::default(),
            max_visible_lights: 256,
            sort_by_importance: true,
            layer_names: HashMap::new(),
        }
    }

    /// Adds a light to the system. Returns its id.
    pub fn add_light(&mut self, light: LightComponent) -> u64 {
        let id = light.id;
        let idx = self.lights.len();
        self.light_index.insert(id, idx);
        self.lights.push(light);
        id
    }

    /// Removes a light by id.
    pub fn remove_light(&mut self, id: u64) -> Option<LightComponent> {
        if let Some(&idx) = self.light_index.get(&id) {
            let removed = self.lights.swap_remove(idx);
            self.light_index.remove(&id);

            // Fix the index of the swapped element.
            if idx < self.lights.len() {
                let swapped_id = self.lights[idx].id;
                self.light_index.insert(swapped_id, idx);
            }

            Some(removed)
        } else {
            None
        }
    }

    /// Returns a reference to a light by id.
    pub fn get_light(&self, id: u64) -> Option<&LightComponent> {
        self.light_index
            .get(&id)
            .and_then(|&idx| self.lights.get(idx))
    }

    /// Returns a mutable reference to a light by id.
    pub fn get_light_mut(&mut self, id: u64) -> Option<&mut LightComponent> {
        self.light_index
            .get(&id)
            .copied()
            .and_then(move |idx| self.lights.get_mut(idx))
    }

    /// Returns all lights.
    pub fn lights(&self) -> &[LightComponent] {
        &self.lights
    }

    /// Returns the number of lights.
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    /// Returns a reference to the environment settings.
    pub fn environment(&self) -> &LightEnvironment {
        &self.environment
    }

    /// Returns a mutable reference to the environment settings.
    pub fn environment_mut(&mut self) -> &mut LightEnvironment {
        &mut self.environment
    }

    /// Sets the environment settings.
    pub fn set_environment(&mut self, env: LightEnvironment) {
        self.environment = env;
    }

    /// Sets the maximum number of visible lights per frame.
    pub fn set_max_visible_lights(&mut self, max: usize) {
        self.max_visible_lights = max;
    }

    /// Names a light layer for debugging / editor display.
    pub fn set_layer_name(&mut self, layer: u8, name: &str) {
        self.layer_names.insert(layer, name.to_string());
    }

    /// Returns the name of a light layer.
    pub fn layer_name(&self, layer: u8) -> Option<&str> {
        self.layer_names.get(&layer).map(|s| s.as_str())
    }

    /// Culls lights for a camera and returns the visible set.
    pub fn cull_for_camera(
        &self,
        frustum: &CameraFrustum,
        camera_layers: LightLayerMask,
    ) -> Vec<CulledLight> {
        cull_lights(&self.lights, frustum, camera_layers, self.max_visible_lights)
    }

    /// Returns all directional lights (useful for shadow setup).
    pub fn directional_lights(&self) -> Vec<&LightComponent> {
        self.lights
            .iter()
            .filter(|l| l.light_type == LightType::Directional && l.enabled)
            .collect()
    }

    /// Returns the primary directional light (highest priority).
    pub fn primary_directional(&self) -> Option<&LightComponent> {
        self.lights
            .iter()
            .filter(|l| l.light_type == LightType::Directional && l.enabled)
            .max_by_key(|l| l.priority)
    }

    /// Returns all shadow-casting lights.
    pub fn shadow_casters(&self) -> Vec<&LightComponent> {
        self.lights
            .iter()
            .filter(|l| l.enabled && l.shadow.enabled)
            .collect()
    }

    /// Removes all lights.
    pub fn clear(&mut self) {
        self.lights.clear();
        self.light_index.clear();
    }

    /// Returns lights of a specific type.
    pub fn lights_of_type(&self, light_type: LightType) -> Vec<&LightComponent> {
        self.lights
            .iter()
            .filter(|l| l.light_type == light_type)
            .collect()
    }

    /// Enables or disables all lights in a given layer.
    pub fn set_layer_enabled(&mut self, layer: u8, enabled: bool) {
        for light in &mut self.lights {
            if light.layers.has_layer(layer) {
                light.enabled = enabled;
            }
        }
    }
}

impl Default for LightSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_temperature_daylight() {
        let c = color_temperature_to_rgb(color_temperature::DAYLIGHT);
        // Daylight should be close to white.
        assert!(c.r > 0.8);
        assert!(c.g > 0.8);
        assert!(c.b > 0.8);
    }

    #[test]
    fn color_temperature_candle() {
        let c = color_temperature_to_rgb(color_temperature::CANDLE);
        // Candle should be warm (more red, less blue).
        assert!(c.r > c.b);
    }

    #[test]
    fn linear_color_operations() {
        let a = LinearColor::new(0.5, 0.5, 0.5);
        let b = LinearColor::new(1.0, 0.0, 0.0);

        let scaled = a.scaled(2.0);
        assert!((scaled.r - 1.0).abs() < 0.001);

        let lerped = a.lerp(&b, 0.5);
        assert!((lerped.r - 0.75).abs() < 0.001);

        let mult = a.multiply(&b);
        assert!((mult.r - 0.5).abs() < 0.001);
        assert!((mult.g).abs() < 0.001);
    }

    #[test]
    fn light_intensity_conversions() {
        let lumens = 800.0;
        let candela = LightIntensity::lumens_to_candela(lumens);
        let back = LightIntensity::candela_to_lumens(candela);
        assert!((back - lumens).abs() < 0.1);
    }

    #[test]
    fn shadow_cascade_splits() {
        let config = ShadowConfig::directional();
        let splits = config.compute_cascade_splits(0.1);
        assert_eq!(splits.len(), 5); // near + 4 cascades
        assert!((splits[0] - 0.1).abs() < 0.001);
        // Each split should be greater than the previous.
        for i in 1..splits.len() {
            assert!(splits[i] > splits[i - 1]);
        }
    }

    #[test]
    fn shadow_fade() {
        let config = ShadowConfig {
            max_distance: 100.0,
            fade_distance: 20.0,
            strength: 1.0,
            ..ShadowConfig::new()
        };
        assert!((config.shadow_fade(50.0) - 1.0).abs() < 0.001);
        assert!((config.shadow_fade(100.0)).abs() < 0.001);
        assert!((config.shadow_fade(90.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn light_layer_mask() {
        let mut mask = LightLayerMask::NONE;
        mask.add_layer(0);
        mask.add_layer(3);
        assert!(mask.has_layer(0));
        assert!(mask.has_layer(3));
        assert!(!mask.has_layer(1));
        assert_eq!(mask.count(), 2);

        let other = LightLayerMask::layer(3);
        assert!(mask.overlaps(other));
    }

    #[test]
    fn point_light_attenuation() {
        let light = LightComponent::point(
            Vec3::ZERO,
            LinearColor::WHITE,
            800.0,
            10.0,
        );

        // At distance 0, attenuation should be near max.
        let a0 = light.attenuation(0.0);
        assert!(a0 > 0.9);

        // At range boundary, attenuation should be 0.
        let a_max = light.attenuation(10.0);
        assert!(a_max.abs() < 0.001);

        // Should decrease with distance.
        let a5 = light.attenuation(5.0);
        assert!(a5 < a0);
        assert!(a5 > a_max);
    }

    #[test]
    fn spot_light_angular_attenuation() {
        let light = LightComponent::spot(
            Vec3::ZERO,
            Vec3::FORWARD,
            LinearColor::WHITE,
            800.0,
            20.0,
            15.0,
            30.0,
        );

        // Inside inner cone.
        assert!((light.spot_attenuation(0.0) - 1.0).abs() < 0.001);

        // Outside outer cone.
        let outside = light.spot_attenuation(std::f32::consts::FRAC_PI_2);
        assert!(outside.abs() < 0.001);
    }

    #[test]
    fn light_system_add_remove() {
        let mut sys = LightSystem::new();

        let id1 = sys.add_light(LightComponent::point(
            Vec3::new(1.0, 2.0, 3.0),
            LinearColor::WHITE,
            800.0,
            10.0,
        ));
        let id2 = sys.add_light(LightComponent::directional(
            Vec3::DOWN,
            LinearColor::WHITE,
            100.0,
        ));

        assert_eq!(sys.light_count(), 2);
        assert!(sys.get_light(id1).is_some());

        sys.remove_light(id1);
        assert_eq!(sys.light_count(), 1);
        assert!(sys.get_light(id1).is_none());
        assert!(sys.get_light(id2).is_some());
    }

    #[test]
    fn light_culling() {
        let mut sys = LightSystem::new();

        // Add a directional light.
        sys.add_light(LightComponent::directional(
            Vec3::DOWN,
            LinearColor::WHITE,
            100.0,
        ));

        // Add a point light in front of the camera.
        sys.add_light(LightComponent::point(
            Vec3::new(0.0, 0.0, -5.0),
            LinearColor::WHITE,
            800.0,
            10.0,
        ));

        // Add a point light behind the camera (should be culled).
        sys.add_light(LightComponent::point(
            Vec3::new(0.0, 0.0, 100.0),
            LinearColor::WHITE,
            100.0,
            5.0,
        ));

        let frustum = CameraFrustum::perspective(
            Vec3::ZERO,
            Vec3::FORWARD,
            Vec3::UP,
            60.0,
            16.0 / 9.0,
            0.1,
            50.0,
        );

        let culled = sys.cull_for_camera(&frustum, LightLayerMask::ALL);

        // Directional always passes, point in front passes, point behind culled.
        assert!(culled.len() >= 2);
    }

    #[test]
    fn environment_ambient_sampling() {
        let env = LightEnvironment::outdoor_day();

        let up_sample = env.sample_ambient(&Vec3::UP);
        let down_sample = env.sample_ambient(&Vec3::DOWN);

        // Sky should be bluer/brighter than ground.
        assert!(up_sample.b > down_sample.b);
    }

    #[test]
    fn environment_fog() {
        let env = LightEnvironment::outdoor_day();

        let near_fog = env.fog_factor(0.0);
        let far_fog = env.fog_factor(1000.0);

        assert!(near_fog < 0.1);
        assert!(far_fog > 0.5);
    }

    #[test]
    fn color_temperature_override() {
        let mut light = LightComponent::point(
            Vec3::ZERO,
            LinearColor::new(1.0, 0.0, 0.0), // Red
            800.0,
            10.0,
        );

        // Without temperature, should be red.
        let c1 = light.effective_color();
        assert!(c1.r > 0.5);
        assert!(c1.g < 0.1);

        // Set daylight temperature -- should be close to white.
        light.set_color_temperature(5500.0);
        let c2 = light.effective_color();
        assert!(c2.r > 0.5);
        assert!(c2.g > 0.5);
        assert!(c2.b > 0.5);
    }

    #[test]
    fn primary_directional_light() {
        let mut sys = LightSystem::new();

        let mut sun = LightComponent::directional(Vec3::DOWN, LinearColor::WHITE, 100.0);
        sun.priority = 100;
        sys.add_light(sun);

        let mut moon = LightComponent::directional(Vec3::DOWN, LinearColor::new(0.1, 0.1, 0.2), 10.0);
        moon.priority = 50;
        sys.add_light(moon);

        let primary = sys.primary_directional().unwrap();
        assert_eq!(primary.priority, 100);
    }

    #[test]
    fn light_contributing_check() {
        let mut light = LightComponent::point(Vec3::ZERO, LinearColor::WHITE, 800.0, 10.0);
        assert!(light.is_contributing());

        light.enabled = false;
        assert!(!light.is_contributing());

        light.enabled = true;
        light.intensity = LightIntensity::Lumens(0.0);
        assert!(!light.is_contributing());
    }

    #[test]
    fn layer_names() {
        let mut sys = LightSystem::new();
        sys.set_layer_name(0, "Default");
        sys.set_layer_name(1, "Characters");
        assert_eq!(sys.layer_name(0), Some("Default"));
        assert_eq!(sys.layer_name(1), Some("Characters"));
        assert_eq!(sys.layer_name(2), None);
    }

    #[test]
    fn linear_color_luminance() {
        let white = LinearColor::WHITE;
        assert!((white.luminance() - 1.0).abs() < 0.01);

        let black = LinearColor::BLACK;
        assert!(black.luminance().abs() < 0.001);
    }

    #[test]
    fn shadow_resolution_pixels() {
        assert_eq!(ShadowResolution::Low.pixels(), 256);
        assert_eq!(ShadowResolution::Ultra.pixels(), 4096);
        assert_eq!(ShadowResolution::Custom(512).pixels(), 512);
    }
}
