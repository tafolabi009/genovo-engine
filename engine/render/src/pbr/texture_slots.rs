// engine/render/src/pbr/texture_slots.rs
//
// Texture slot management for PBR materials. Defines the available texture
// slots, their bindings, UV transforms, and the set of all texture bindings
// for a material.

use glam::{Mat3, Vec2};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// TextureSlot
// ---------------------------------------------------------------------------

/// Enumerates the texture slots available in a PBR material.
///
/// Each slot corresponds to a specific role in the PBR shading model.
/// The numeric value determines the bind-group binding index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum TextureSlot {
    /// Base colour / albedo map (sRGB).
    Albedo = 0,
    /// Tangent-space normal map (linear, RG or RGB).
    Normal = 1,
    /// Packed metallic (B) + roughness (G) map (linear).
    MetallicRoughness = 2,
    /// Ambient occlusion map (linear, R channel).
    AO = 3,
    /// Emissive colour map (sRGB).
    Emissive = 4,
    /// Height / displacement map (linear, R channel).
    Height = 5,
    /// Detail normal map (tangent space, applied on top of primary normal).
    DetailNormal = 6,
    /// Detail albedo map (multiplied with primary albedo).
    DetailAlbedo = 7,
    /// Clearcoat normal map.
    ClearcoatNormal = 8,
    /// Anisotropy direction map.
    Anisotropy = 9,
    /// Subsurface scattering thickness map.
    SubsurfaceThickness = 10,
    /// Sheen colour map.
    Sheen = 11,
    /// Transmission map (for thin-surface transparency).
    Transmission = 12,
    /// Light map / baked lighting.
    Lightmap = 13,
}

impl TextureSlot {
    /// Total number of texture slots.
    pub const COUNT: usize = 14;

    /// All slots as an array for iteration.
    pub const ALL: [TextureSlot; Self::COUNT] = [
        Self::Albedo,
        Self::Normal,
        Self::MetallicRoughness,
        Self::AO,
        Self::Emissive,
        Self::Height,
        Self::DetailNormal,
        Self::DetailAlbedo,
        Self::ClearcoatNormal,
        Self::Anisotropy,
        Self::SubsurfaceThickness,
        Self::Sheen,
        Self::Transmission,
        Self::Lightmap,
    ];

    /// Returns the binding index for use in a bind-group layout.
    #[inline]
    pub fn binding_index(self) -> u32 {
        self as u32
    }

    /// Returns `true` if this slot stores sRGB-encoded data (needs
    /// linearisation on sample).
    pub fn is_srgb(self) -> bool {
        matches!(self, Self::Albedo | Self::Emissive | Self::DetailAlbedo)
    }

    /// Returns `true` if this slot stores linear data.
    pub fn is_linear(self) -> bool {
        !self.is_srgb()
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Albedo => "Albedo",
            Self::Normal => "Normal",
            Self::MetallicRoughness => "MetallicRoughness",
            Self::AO => "AO",
            Self::Emissive => "Emissive",
            Self::Height => "Height",
            Self::DetailNormal => "DetailNormal",
            Self::DetailAlbedo => "DetailAlbedo",
            Self::ClearcoatNormal => "ClearcoatNormal",
            Self::Anisotropy => "Anisotropy",
            Self::SubsurfaceThickness => "SubsurfaceThickness",
            Self::Sheen => "Sheen",
            Self::Transmission => "Transmission",
            Self::Lightmap => "Lightmap",
        }
    }

    /// Default texture format for this slot.
    pub fn default_format(self) -> TextureSlotFormat {
        match self {
            Self::Albedo | Self::Emissive | Self::DetailAlbedo | Self::Sheen => {
                TextureSlotFormat::Rgba8Srgb
            }
            Self::Normal | Self::DetailNormal | Self::ClearcoatNormal | Self::Anisotropy => {
                TextureSlotFormat::Rgba8Unorm
            }
            Self::MetallicRoughness | Self::AO | Self::SubsurfaceThickness | Self::Transmission => {
                TextureSlotFormat::Rgba8Unorm
            }
            Self::Height => TextureSlotFormat::R16Float,
            Self::Lightmap => TextureSlotFormat::Rgba16Float,
        }
    }

    /// Shader define name used for conditional compilation.
    pub fn feature_define(self) -> &'static str {
        match self {
            Self::Albedo => "HAS_ALBEDO_MAP",
            Self::Normal => "HAS_NORMAL_MAP",
            Self::MetallicRoughness => "HAS_METALLIC_ROUGHNESS_MAP",
            Self::AO => "HAS_AO_MAP",
            Self::Emissive => "HAS_EMISSIVE_MAP",
            Self::Height => "HAS_HEIGHT_MAP",
            Self::DetailNormal => "HAS_DETAIL_NORMAL_MAP",
            Self::DetailAlbedo => "HAS_DETAIL_ALBEDO_MAP",
            Self::ClearcoatNormal => "HAS_CLEARCOAT_NORMAL_MAP",
            Self::Anisotropy => "HAS_ANISOTROPY_MAP",
            Self::SubsurfaceThickness => "HAS_SUBSURFACE_MAP",
            Self::Sheen => "HAS_SHEEN_MAP",
            Self::Transmission => "HAS_TRANSMISSION_MAP",
            Self::Lightmap => "HAS_LIGHTMAP",
        }
    }
}

impl fmt::Display for TextureSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// TextureSlotFormat
// ---------------------------------------------------------------------------

/// Texture formats used by material texture slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSlotFormat {
    Rgba8Srgb,
    Rgba8Unorm,
    Rg8Unorm,
    R8Unorm,
    R16Float,
    Rg16Float,
    Rgba16Float,
    Bc1Srgb,
    Bc3Srgb,
    Bc5Unorm,
    Bc7Srgb,
    Bc7Unorm,
}

impl TextureSlotFormat {
    /// Bytes per pixel for uncompressed formats.
    pub fn bytes_per_pixel(self) -> Option<u32> {
        match self {
            Self::Rgba8Srgb | Self::Rgba8Unorm => Some(4),
            Self::Rg8Unorm => Some(2),
            Self::R8Unorm => Some(1),
            Self::R16Float => Some(2),
            Self::Rg16Float => Some(4),
            Self::Rgba16Float => Some(8),
            _ => None, // block-compressed
        }
    }

    /// Returns `true` for block-compressed formats.
    pub fn is_compressed(self) -> bool {
        matches!(
            self,
            Self::Bc1Srgb | Self::Bc3Srgb | Self::Bc5Unorm | Self::Bc7Srgb | Self::Bc7Unorm
        )
    }
}

// ---------------------------------------------------------------------------
// UvTransform
// ---------------------------------------------------------------------------

/// 2D UV coordinate transform (scale, rotation, offset).
///
/// Applied in the vertex/fragment shader to transform texture coordinates
/// before sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvTransform {
    /// Scale factors (default [1, 1]).
    pub scale: Vec2,
    /// Rotation in radians (counter-clockwise).
    pub rotation: f32,
    /// Translation offset.
    pub offset: Vec2,
}

impl Default for UvTransform {
    fn default() -> Self {
        Self {
            scale: Vec2::ONE,
            rotation: 0.0,
            offset: Vec2::ZERO,
        }
    }
}

impl UvTransform {
    /// Identity transform (no modification).
    pub const IDENTITY: Self = Self {
        scale: Vec2::ONE,
        rotation: 0.0,
        offset: Vec2::ZERO,
    };

    /// Create a simple tiling transform.
    pub fn tiled(tile_x: f32, tile_y: f32) -> Self {
        Self {
            scale: Vec2::new(tile_x, tile_y),
            rotation: 0.0,
            offset: Vec2::ZERO,
        }
    }

    /// Returns `true` if this is an identity transform.
    pub fn is_identity(&self) -> bool {
        (self.scale - Vec2::ONE).length_squared() < 1e-7
            && self.rotation.abs() < 1e-7
            && self.offset.length_squared() < 1e-7
    }

    /// Build a 3x3 affine matrix for this UV transform.
    ///
    /// The transform order is: scale -> rotate -> translate.
    pub fn to_matrix(&self) -> Mat3 {
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();

        // Scale * Rotate
        let m00 = self.scale.x * cos_r;
        let m01 = self.scale.x * sin_r;
        let m10 = -self.scale.y * sin_r;
        let m11 = self.scale.y * cos_r;

        Mat3::from_cols_array(&[
            m00,
            m01,
            0.0,
            m10,
            m11,
            0.0,
            self.offset.x,
            self.offset.y,
            1.0,
        ])
    }

    /// Transform a UV coordinate.
    pub fn apply(&self, uv: Vec2) -> Vec2 {
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        let scaled = uv * self.scale;
        let rotated = Vec2::new(
            scaled.x * cos_r - scaled.y * sin_r,
            scaled.x * sin_r + scaled.y * cos_r,
        );
        rotated + self.offset
    }
}

/// GPU-compatible UV transform. 16 bytes, suitable for a uniform buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UvTransformGpu {
    pub scale_offset: [f32; 4],   // xy = scale, zw = offset
    pub rotation_pad: [f32; 4],   // x = rotation, yzw = padding
}

impl From<&UvTransform> for UvTransformGpu {
    fn from(t: &UvTransform) -> Self {
        Self {
            scale_offset: [t.scale.x, t.scale.y, t.offset.x, t.offset.y],
            rotation_pad: [t.rotation, 0.0, 0.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// SamplerConfig
// ---------------------------------------------------------------------------

/// Sampler configuration for a texture binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerConfig {
    /// Minification filter.
    pub min_filter: FilterMode,
    /// Magnification filter.
    pub mag_filter: FilterMode,
    /// Mipmap filter.
    pub mipmap_filter: FilterMode,
    /// Address mode for U axis.
    pub address_u: AddressMode,
    /// Address mode for V axis.
    pub address_v: AddressMode,
    /// Address mode for W axis (for 3D textures).
    pub address_w: AddressMode,
    /// Maximum anisotropy level (1 = disabled, 16 = max quality).
    pub max_anisotropy: u8,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_u: AddressMode::Repeat,
            address_v: AddressMode::Repeat,
            address_w: AddressMode::Repeat,
            max_anisotropy: 16,
        }
    }
}

/// Texture filtering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterMode {
    Nearest,
    Linear,
}

/// Texture address / wrap mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressMode {
    Repeat,
    MirrorRepeat,
    ClampToEdge,
    ClampToBorder,
}

// ---------------------------------------------------------------------------
// TextureBinding
// ---------------------------------------------------------------------------

/// Describes a single texture binding within a material.
///
/// Associates a texture handle with its sampler configuration, UV channel,
/// and UV transform.
#[derive(Debug, Clone)]
pub struct TextureBinding {
    /// Handle to the texture resource. Uses u64 as a generic handle type
    /// that can map to any backend texture.
    pub texture_handle: u64,
    /// The sampler configuration for this texture.
    pub sampler: SamplerConfig,
    /// Which UV channel to use (0 = first, 1 = second, etc.).
    pub uv_channel: u32,
    /// UV coordinate transform.
    pub uv_transform: UvTransform,
    /// Optional texture swizzle (e.g. for single-channel AO in the R of
    /// a packed texture).
    pub swizzle: TextureSwizzle,
    /// Texture format.
    pub format: TextureSlotFormat,
}

impl TextureBinding {
    /// Create a new binding with default sampler and no UV transform.
    pub fn new(texture_handle: u64, format: TextureSlotFormat) -> Self {
        Self {
            texture_handle,
            sampler: SamplerConfig::default(),
            uv_channel: 0,
            uv_transform: UvTransform::IDENTITY,
            swizzle: TextureSwizzle::Identity,
            format,
        }
    }

    /// Set the UV channel.
    pub fn with_uv_channel(mut self, channel: u32) -> Self {
        self.uv_channel = channel;
        self
    }

    /// Set the UV transform.
    pub fn with_uv_transform(mut self, transform: UvTransform) -> Self {
        self.uv_transform = transform;
        self
    }

    /// Set the sampler configuration.
    pub fn with_sampler(mut self, sampler: SamplerConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set the swizzle.
    pub fn with_swizzle(mut self, swizzle: TextureSwizzle) -> Self {
        self.swizzle = swizzle;
        self
    }
}

/// Texture component swizzle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSwizzle {
    /// RGBA as-is.
    Identity,
    /// Use the R channel for all components (greyscale).
    RToAll,
    /// Use RG channels (e.g. for normal maps stored as RG8).
    RgOnly,
    /// Custom swizzle: each element selects a source channel (0=R,1=G,2=B,3=A).
    Custom([u8; 4]),
}

// ---------------------------------------------------------------------------
// MaterialTextureSet
// ---------------------------------------------------------------------------

/// Manages all texture bindings for a material.
///
/// Indexed by `TextureSlot`, this set stores at most one `TextureBinding`
/// per slot.
#[derive(Debug, Clone)]
pub struct MaterialTextureSet {
    bindings: HashMap<TextureSlot, TextureBinding>,
}

impl MaterialTextureSet {
    /// Create an empty texture set.
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Bind a texture to a slot.
    pub fn set(&mut self, slot: TextureSlot, binding: TextureBinding) {
        self.bindings.insert(slot, binding);
    }

    /// Remove a texture binding from a slot.
    pub fn remove(&mut self, slot: TextureSlot) -> Option<TextureBinding> {
        self.bindings.remove(&slot)
    }

    /// Returns `true` if the given slot has a texture bound.
    pub fn has_slot(&self, slot: TextureSlot) -> bool {
        self.bindings.contains_key(&slot)
    }

    /// Get the binding for a slot.
    pub fn get(&self, slot: TextureSlot) -> Option<&TextureBinding> {
        self.bindings.get(&slot)
    }

    /// Get a mutable reference to the binding for a slot.
    pub fn get_mut(&mut self, slot: TextureSlot) -> Option<&mut TextureBinding> {
        self.bindings.get_mut(&slot)
    }

    /// Iterate over all bound slots.
    pub fn iter(&self) -> impl Iterator<Item = (TextureSlot, &TextureBinding)> {
        self.bindings.iter().map(|(&slot, binding)| (slot, binding))
    }

    /// Number of bound texture slots.
    pub fn count(&self) -> usize {
        self.bindings.len()
    }

    /// Returns `true` if no textures are bound.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Clear all bindings.
    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    /// Collect the feature defines for all bound texture slots.
    pub fn feature_defines(&self) -> Vec<&'static str> {
        self.bindings
            .keys()
            .map(|slot| slot.feature_define())
            .collect()
    }
}

impl Default for MaterialTextureSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Mipmap generation helpers
// ---------------------------------------------------------------------------

/// Compute the number of mip levels for a texture of the given dimensions.
pub fn mip_level_count(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height) as f32;
    (max_dim.log2().floor() as u32) + 1
}

/// Compute the dimensions of a given mip level.
pub fn mip_dimensions(width: u32, height: u32, level: u32) -> (u32, u32) {
    let w = (width >> level).max(1);
    let h = (height >> level).max(1);
    (w, h)
}

/// Generate a mipmap chain by simple 2x2 box filtering.
///
/// # Arguments
/// - `src` — source pixel data (RGBA u8, row-major).
/// - `width` — source width.
/// - `height` — source height.
///
/// # Returns
/// A vector of mip levels, each containing the pixel data for that level.
/// Level 0 is the original data (cloned).
pub fn generate_mipmaps_rgba8(src: &[u8], width: u32, height: u32) -> Vec<Vec<u8>> {
    let levels = mip_level_count(width, height);
    let mut result = Vec::with_capacity(levels as usize);
    result.push(src.to_vec());

    let mut prev_w = width;
    let mut prev_h = height;
    let mut prev_data = src.to_vec();

    for _level in 1..levels {
        let new_w = (prev_w / 2).max(1);
        let new_h = (prev_h / 2).max(1);
        let mut new_data = vec![0u8; (new_w * new_h * 4) as usize];

        for y in 0..new_h {
            for x in 0..new_w {
                let sx = (x * 2).min(prev_w - 1);
                let sy = (y * 2).min(prev_h - 1);
                let sx1 = (sx + 1).min(prev_w - 1);
                let sy1 = (sy + 1).min(prev_h - 1);

                for c in 0..4 {
                    let idx00 = ((sy * prev_w + sx) * 4 + c) as usize;
                    let idx10 = ((sy * prev_w + sx1) * 4 + c) as usize;
                    let idx01 = ((sy1 * prev_w + sx) * 4 + c) as usize;
                    let idx11 = ((sy1 * prev_w + sx1) * 4 + c) as usize;

                    let p00 = prev_data.get(idx00).copied().unwrap_or(0) as u16;
                    let p10 = prev_data.get(idx10).copied().unwrap_or(0) as u16;
                    let p01 = prev_data.get(idx01).copied().unwrap_or(0) as u16;
                    let p11 = prev_data.get(idx11).copied().unwrap_or(0) as u16;

                    let avg = ((p00 + p10 + p01 + p11 + 2) / 4) as u8;
                    new_data[((y * new_w + x) * 4 + c) as usize] = avg;
                }
            }
        }

        result.push(new_data.clone());
        prev_w = new_w;
        prev_h = new_h;
        prev_data = new_data;
    }

    result
}

/// Generate a mipmap chain for a single-channel (R8) texture.
pub fn generate_mipmaps_r8(src: &[u8], width: u32, height: u32) -> Vec<Vec<u8>> {
    let levels = mip_level_count(width, height);
    let mut result = Vec::with_capacity(levels as usize);
    result.push(src.to_vec());

    let mut prev_w = width;
    let mut prev_h = height;
    let mut prev_data = src.to_vec();

    for _level in 1..levels {
        let new_w = (prev_w / 2).max(1);
        let new_h = (prev_h / 2).max(1);
        let mut new_data = vec![0u8; (new_w * new_h) as usize];

        for y in 0..new_h {
            for x in 0..new_w {
                let sx = (x * 2).min(prev_w - 1);
                let sy = (y * 2).min(prev_h - 1);
                let sx1 = (sx + 1).min(prev_w - 1);
                let sy1 = (sy + 1).min(prev_h - 1);

                let p00 = prev_data[(sy * prev_w + sx) as usize] as u16;
                let p10 = prev_data[(sy * prev_w + sx1) as usize] as u16;
                let p01 = prev_data[(sy1 * prev_w + sx) as usize] as u16;
                let p11 = prev_data[(sy1 * prev_w + sx1) as usize] as u16;

                let avg = ((p00 + p10 + p01 + p11 + 2) / 4) as u8;
                new_data[(y * new_w + x) as usize] = avg;
            }
        }

        result.push(new_data.clone());
        prev_w = new_w;
        prev_h = new_h;
        prev_data = new_data;
    }

    result
}

/// sRGB to linear conversion for a single channel value.
#[inline]
pub fn srgb_to_linear(srgb: f32) -> f32 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear to sRGB conversion for a single channel value.
#[inline]
pub fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_slot_binding_indices() {
        assert_eq!(TextureSlot::Albedo.binding_index(), 0);
        assert_eq!(TextureSlot::Normal.binding_index(), 1);
        assert_eq!(TextureSlot::MetallicRoughness.binding_index(), 2);
    }

    #[test]
    fn uv_transform_identity() {
        let t = UvTransform::IDENTITY;
        assert!(t.is_identity());
        let uv = t.apply(Vec2::new(0.5, 0.5));
        assert!((uv.x - 0.5).abs() < 1e-6);
        assert!((uv.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn uv_transform_tiling() {
        let t = UvTransform::tiled(2.0, 2.0);
        let uv = t.apply(Vec2::new(0.5, 0.5));
        assert!((uv.x - 1.0).abs() < 1e-6);
        assert!((uv.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mip_level_count_power_of_two() {
        assert_eq!(mip_level_count(256, 256), 9);
        assert_eq!(mip_level_count(1024, 1024), 11);
        assert_eq!(mip_level_count(1, 1), 1);
    }

    #[test]
    fn mip_dimensions_halve() {
        assert_eq!(mip_dimensions(256, 256, 0), (256, 256));
        assert_eq!(mip_dimensions(256, 256, 1), (128, 128));
        assert_eq!(mip_dimensions(256, 256, 8), (1, 1));
    }

    #[test]
    fn material_texture_set_operations() {
        let mut set = MaterialTextureSet::new();
        assert!(set.is_empty());

        set.set(
            TextureSlot::Albedo,
            TextureBinding::new(42, TextureSlotFormat::Rgba8Srgb),
        );
        assert!(set.has_slot(TextureSlot::Albedo));
        assert!(!set.has_slot(TextureSlot::Normal));
        assert_eq!(set.count(), 1);

        set.remove(TextureSlot::Albedo);
        assert!(!set.has_slot(TextureSlot::Albedo));
    }

    #[test]
    fn srgb_roundtrip() {
        for i in 0..=255 {
            let srgb = i as f32 / 255.0;
            let linear = srgb_to_linear(srgb);
            let back = linear_to_srgb(linear);
            assert!((srgb - back).abs() < 0.002, "Failed for srgb={srgb}");
        }
    }
}
