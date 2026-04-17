// engine/render/src/material_layering.rs
//
// Material layering system for the Genovo engine.
//
// Supports:
// - Height-based blending between material layers
// - Detail materials (add wear, scratches, grime overlays)
// - Decal materials (projected onto surfaces)
// - Material masks (control blending per-pixel)
// - Parallax occlusion mapping (steep parallax with silhouette)
// - Clearcoat layer (automotive paint, lacquered wood)
//
// Materials are composed of multiple layers that blend together using
// height maps, masks, and blend modes.

use glam::{Vec2, Vec3, Vec4};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;

/// Maximum number of layers in a layered material.
pub const MAX_MATERIAL_LAYERS: usize = 8;

/// Maximum number of decal projectors per surface.
pub const MAX_DECAL_PROJECTORS: usize = 16;

/// Default POM step count for parallax mapping.
pub const DEFAULT_POM_STEPS: u32 = 32;

/// Default POM refinement steps for binary search.
pub const DEFAULT_POM_REFINEMENT_STEPS: u32 = 8;

// ---------------------------------------------------------------------------
// Blend modes
// ---------------------------------------------------------------------------

/// How two material layers are blended together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Linear interpolation based on alpha.
    Lerp,
    /// Height-based blending with sharp transitions.
    HeightBlend,
    /// Additive blending (details are added on top).
    Additive,
    /// Multiplicative blending.
    Multiply,
    /// Overlay blend mode (Photoshop-style).
    Overlay,
    /// Minimum of both values.
    Min,
    /// Maximum of both values.
    Max,
    /// Replace (hard cut based on mask).
    Replace,
}

impl BlendMode {
    /// Blend two scalar values using this mode.
    pub fn blend_scalar(self, base: f32, overlay: f32, alpha: f32) -> f32 {
        match self {
            Self::Lerp => base + (overlay - base) * alpha,
            Self::HeightBlend => base + (overlay - base) * alpha, // Height blend is handled externally
            Self::Additive => base + overlay * alpha,
            Self::Multiply => base * (1.0 - alpha + overlay * alpha),
            Self::Overlay => {
                let result = if base < 0.5 {
                    2.0 * base * overlay
                } else {
                    1.0 - 2.0 * (1.0 - base) * (1.0 - overlay)
                };
                base + (result - base) * alpha
            }
            Self::Min => base.min(overlay),
            Self::Max => base.max(overlay),
            Self::Replace => if alpha > 0.5 { overlay } else { base },
        }
    }

    /// Blend two Vec3 values using this mode.
    pub fn blend_vec3(self, base: Vec3, overlay: Vec3, alpha: f32) -> Vec3 {
        Vec3::new(
            self.blend_scalar(base.x, overlay.x, alpha),
            self.blend_scalar(base.y, overlay.y, alpha),
            self.blend_scalar(base.z, overlay.z, alpha),
        )
    }
}

// ---------------------------------------------------------------------------
// Material properties (PBR)
// ---------------------------------------------------------------------------

/// Sampled PBR material properties at a surface point.
#[derive(Debug, Clone, Copy)]
pub struct MaterialSample {
    /// Base colour / albedo (linear RGB).
    pub albedo: Vec3,
    /// Normal (tangent space, typically from normal map).
    pub normal: Vec3,
    /// Metallic factor (0 = dielectric, 1 = metal).
    pub metallic: f32,
    /// Roughness (0 = mirror, 1 = fully rough).
    pub roughness: f32,
    /// Ambient occlusion.
    pub ao: f32,
    /// Emissive colour.
    pub emissive: Vec3,
    /// Height (for parallax mapping and height blending).
    pub height: f32,
    /// Opacity.
    pub opacity: f32,
}

impl Default for MaterialSample {
    fn default() -> Self {
        Self {
            albedo: Vec3::splat(0.5),
            normal: Vec3::new(0.0, 0.0, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            emissive: Vec3::ZERO,
            height: 0.5,
            opacity: 1.0,
        }
    }
}

impl MaterialSample {
    /// Create a simple diffuse material.
    pub fn diffuse(albedo: Vec3) -> Self {
        Self {
            albedo,
            ..Default::default()
        }
    }

    /// Create a metallic material.
    pub fn metal(albedo: Vec3, roughness: f32) -> Self {
        Self {
            albedo,
            metallic: 1.0,
            roughness,
            ..Default::default()
        }
    }

    /// Lerp between two material samples.
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            albedo: Vec3::lerp(a.albedo, b.albedo, t),
            normal: Vec3::lerp(a.normal, b.normal, t).normalize_or_zero(),
            metallic: a.metallic + (b.metallic - a.metallic) * t,
            roughness: a.roughness + (b.roughness - a.roughness) * t,
            ao: a.ao + (b.ao - a.ao) * t,
            emissive: Vec3::lerp(a.emissive, b.emissive, t),
            height: a.height + (b.height - a.height) * t,
            opacity: a.opacity + (b.opacity - a.opacity) * t,
        }
    }
}

// ---------------------------------------------------------------------------
// Material layer
// ---------------------------------------------------------------------------

/// A single layer in a layered material.
#[derive(Debug, Clone)]
pub struct MaterialLayer {
    /// Name of this layer.
    pub name: String,
    /// Whether this layer is enabled.
    pub enabled: bool,
    /// Blend mode for this layer.
    pub blend_mode: BlendMode,
    /// Global opacity of this layer.
    pub opacity: f32,
    /// UV tiling scale.
    pub tiling: Vec2,
    /// UV offset.
    pub offset: Vec2,
    /// UV rotation in radians.
    pub rotation: f32,
    /// Height map influence for height-based blending.
    pub height_blend_sharpness: f32,
    /// Height offset (shift the blend boundary).
    pub height_offset: f32,
    /// Material properties (would normally be texture handles).
    pub base_sample: MaterialSample,
    /// Mask channel index (-1 = no mask, 0-3 = RGBA channels of mask texture).
    pub mask_channel: i32,
    /// Whether to use triplanar mapping for this layer.
    pub triplanar: bool,
    /// Triplanar sharpness.
    pub triplanar_sharpness: f32,
}

impl Default for MaterialLayer {
    fn default() -> Self {
        Self {
            name: String::from("Layer"),
            enabled: true,
            blend_mode: BlendMode::Lerp,
            opacity: 1.0,
            tiling: Vec2::ONE,
            offset: Vec2::ZERO,
            rotation: 0.0,
            height_blend_sharpness: 10.0,
            height_offset: 0.0,
            base_sample: MaterialSample::default(),
            mask_channel: -1,
            triplanar: false,
            triplanar_sharpness: 8.0,
        }
    }
}

impl MaterialLayer {
    /// Create a new named layer.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Set the blend mode.
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Set opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set tiling.
    pub fn with_tiling(mut self, tiling: Vec2) -> Self {
        self.tiling = tiling;
        self
    }

    /// Set height blend sharpness.
    pub fn with_height_blend(mut self, sharpness: f32, offset: f32) -> Self {
        self.height_blend_sharpness = sharpness;
        self.height_offset = offset;
        self.blend_mode = BlendMode::HeightBlend;
        self
    }

    /// Compute the UV for this layer given base UV.
    pub fn compute_uv(&self, base_uv: Vec2) -> Vec2 {
        let centered = base_uv - Vec2::splat(0.5);
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        let rotated = Vec2::new(
            centered.x * cos_r - centered.y * sin_r,
            centered.x * sin_r + centered.y * cos_r,
        );
        (rotated + Vec2::splat(0.5)) * self.tiling + self.offset
    }
}

// ---------------------------------------------------------------------------
// Height-based blending
// ---------------------------------------------------------------------------

/// Compute height-based blend weight for two layers.
///
/// This creates a sharp, natural-looking transition where taller features
/// of one material poke through the other (e.g. rocks through snow).
pub fn height_blend(
    height_a: f32,
    height_b: f32,
    blend_factor: f32,
    sharpness: f32,
) -> f32 {
    let depth = sharpness * 0.1;
    let ha = height_a + (1.0 - blend_factor);
    let hb = height_b + blend_factor;
    let max_h = ha.max(hb);
    let wa = (ha - max_h + depth).max(0.0);
    let wb = (hb - max_h + depth).max(0.0);
    let total = wa + wb;
    if total > EPSILON {
        wb / total
    } else {
        blend_factor
    }
}

/// Blend an array of material layers using height-based blending and masks.
pub fn blend_layers(
    layers: &[MaterialLayer],
    base_uv: Vec2,
    mask_values: &[f32],
) -> MaterialSample {
    if layers.is_empty() {
        return MaterialSample::default();
    }

    let mut result = layers[0].base_sample;

    for i in 1..layers.len() {
        let layer = &layers[i];
        if !layer.enabled || layer.opacity < EPSILON {
            continue;
        }

        let mut alpha = layer.opacity;

        // Apply mask
        if layer.mask_channel >= 0 && (layer.mask_channel as usize) < mask_values.len() {
            alpha *= mask_values[layer.mask_channel as usize];
        }

        let layer_sample = &layer.base_sample;

        match layer.blend_mode {
            BlendMode::HeightBlend => {
                let blend = height_blend(
                    result.height + layer.height_offset,
                    layer_sample.height,
                    alpha,
                    layer.height_blend_sharpness,
                );
                result = MaterialSample::lerp(&result, layer_sample, blend);
            }
            mode => {
                result.albedo = mode.blend_vec3(result.albedo, layer_sample.albedo, alpha);
                result.metallic = mode.blend_scalar(result.metallic, layer_sample.metallic, alpha);
                result.roughness = mode.blend_scalar(result.roughness, layer_sample.roughness, alpha);
                result.ao = mode.blend_scalar(result.ao, layer_sample.ao, alpha);
                result.emissive = mode.blend_vec3(result.emissive, layer_sample.emissive, alpha);
                result.height = mode.blend_scalar(result.height, layer_sample.height, alpha);
                // Normal blending (reoriented normal blending for better results)
                result.normal = reoriented_normal_blend(result.normal, layer_sample.normal, alpha);
            }
        }
    }

    result
}

/// Reoriented normal map blending (Barr 2012).
fn reoriented_normal_blend(base: Vec3, detail: Vec3, weight: f32) -> Vec3 {
    if weight < EPSILON {
        return base;
    }
    let b = base * Vec3::new(1.0, 1.0, 1.0) + Vec3::new(0.0, 0.0, 1.0);
    let d = detail * Vec3::new(-1.0, -1.0, 1.0);
    let blended = Vec3::new(
        b.x * d.z + d.x * b.z,
        b.y * d.z + d.y * b.z,
        b.z * d.z - b.x * d.x - b.y * d.y,
    );
    let lerped = Vec3::lerp(base, blended.normalize_or_zero(), weight);
    lerped.normalize_or_zero()
}

// ---------------------------------------------------------------------------
// Detail materials
// ---------------------------------------------------------------------------

/// Type of detail overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetailType {
    /// Wear patterns (edges, convex areas).
    Wear,
    /// Scratches.
    Scratches,
    /// Dirt / grime.
    Dirt,
    /// Moss / organic growth.
    Moss,
    /// Rust.
    Rust,
    /// Snow accumulation.
    Snow,
    /// Water / wetness.
    Wetness,
    /// Custom detail type.
    Custom,
}

/// A detail material that adds surface variation.
#[derive(Debug, Clone)]
pub struct DetailMaterial {
    /// Type of detail.
    pub detail_type: DetailType,
    /// Whether this detail is enabled.
    pub enabled: bool,
    /// Detail material properties.
    pub sample: MaterialSample,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Overall intensity.
    pub intensity: f32,
    /// UV tiling for the detail.
    pub tiling: Vec2,
    /// Whether the detail is curvature-dependent (appears on edges).
    pub curvature_dependent: bool,
    /// Curvature threshold for wear effects.
    pub curvature_threshold: f32,
    /// Whether the detail is world-space projected.
    pub world_space: bool,
    /// World-space projection axis (for snow, wetness, etc.).
    pub projection_axis: Vec3,
    /// Projection threshold (dot product with projection axis).
    pub projection_threshold: f32,
}

impl DetailMaterial {
    /// Create a wear detail.
    pub fn wear() -> Self {
        Self {
            detail_type: DetailType::Wear,
            enabled: true,
            sample: MaterialSample {
                albedo: Vec3::splat(0.3),
                metallic: 0.8,
                roughness: 0.3,
                ..Default::default()
            },
            blend_mode: BlendMode::Lerp,
            intensity: 0.5,
            tiling: Vec2::splat(4.0),
            curvature_dependent: true,
            curvature_threshold: 0.3,
            world_space: false,
            projection_axis: Vec3::Y,
            projection_threshold: 0.5,
        }
    }

    /// Create a scratch detail.
    pub fn scratches() -> Self {
        Self {
            detail_type: DetailType::Scratches,
            enabled: true,
            sample: MaterialSample {
                albedo: Vec3::splat(0.4),
                metallic: 0.6,
                roughness: 0.8,
                ..Default::default()
            },
            blend_mode: BlendMode::Lerp,
            intensity: 0.3,
            tiling: Vec2::splat(8.0),
            curvature_dependent: false,
            curvature_threshold: 0.5,
            world_space: false,
            projection_axis: Vec3::Y,
            projection_threshold: 0.5,
        }
    }

    /// Create a snow accumulation detail.
    pub fn snow() -> Self {
        Self {
            detail_type: DetailType::Snow,
            enabled: true,
            sample: MaterialSample {
                albedo: Vec3::splat(0.95),
                metallic: 0.0,
                roughness: 0.7,
                ..Default::default()
            },
            blend_mode: BlendMode::Lerp,
            intensity: 1.0,
            tiling: Vec2::splat(2.0),
            curvature_dependent: false,
            curvature_threshold: 0.5,
            world_space: true,
            projection_axis: Vec3::Y,
            projection_threshold: 0.7,
        }
    }

    /// Create a dirt/grime detail.
    pub fn dirt() -> Self {
        Self {
            detail_type: DetailType::Dirt,
            enabled: true,
            sample: MaterialSample {
                albedo: Vec3::new(0.15, 0.12, 0.08),
                metallic: 0.0,
                roughness: 0.9,
                ao: 0.5,
                ..Default::default()
            },
            blend_mode: BlendMode::Multiply,
            intensity: 0.4,
            tiling: Vec2::splat(3.0),
            curvature_dependent: true,
            curvature_threshold: -0.2, // Accumulates in concavities
            world_space: false,
            projection_axis: Vec3::Y,
            projection_threshold: 0.5,
        }
    }

    /// Create a wetness detail.
    pub fn wetness() -> Self {
        Self {
            detail_type: DetailType::Wetness,
            enabled: true,
            sample: MaterialSample {
                albedo: Vec3::ZERO, // Darkens albedo
                metallic: 0.0,
                roughness: 0.1, // Very smooth when wet
                ..Default::default()
            },
            blend_mode: BlendMode::Multiply,
            intensity: 0.8,
            tiling: Vec2::ONE,
            curvature_dependent: false,
            curvature_threshold: 0.5,
            world_space: true,
            projection_axis: Vec3::Y,
            projection_threshold: 0.5,
        }
    }

    /// Evaluate the detail weight at a point.
    pub fn evaluate_weight(
        &self,
        surface_normal: Vec3,
        curvature: f32,
    ) -> f32 {
        if !self.enabled {
            return 0.0;
        }

        let mut weight = self.intensity;

        // Curvature-based weighting
        if self.curvature_dependent {
            let curvature_factor = if self.curvature_threshold >= 0.0 {
                // Convex areas (edges, wear)
                ((curvature - self.curvature_threshold) / (1.0 - self.curvature_threshold + EPSILON))
                    .clamp(0.0, 1.0)
            } else {
                // Concave areas (cavities, dirt)
                ((self.curvature_threshold - curvature) / (self.curvature_threshold.abs() + EPSILON))
                    .clamp(0.0, 1.0)
            };
            weight *= curvature_factor;
        }

        // World-space projection
        if self.world_space {
            let dot = surface_normal.dot(self.projection_axis);
            let proj_factor = ((dot - self.projection_threshold) / (1.0 - self.projection_threshold + EPSILON))
                .clamp(0.0, 1.0);
            weight *= proj_factor;
        }

        weight
    }
}

// ---------------------------------------------------------------------------
// Decal materials
// ---------------------------------------------------------------------------

/// A decal material that is projected onto surfaces.
#[derive(Debug, Clone)]
pub struct DecalMaterial {
    /// Name identifier.
    pub name: String,
    /// Whether the decal is active.
    pub enabled: bool,
    /// Decal material properties.
    pub sample: MaterialSample,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Which channels this decal affects.
    pub channels: DecalChannels,
    /// Projection matrix (object -> clip space of the decal projector).
    pub projection: glam::Mat4,
    /// Inverse projection for world -> decal UV.
    pub inverse_projection: glam::Mat4,
    /// World-space position of the projector.
    pub position: Vec3,
    /// World-space forward direction of the projector.
    pub direction: Vec3,
    /// Projection half-size.
    pub half_size: Vec3,
    /// Normal fade angle (in radians). Decal fades on surfaces not facing the projector.
    pub normal_fade_angle: f32,
    /// Distance fade start.
    pub fade_start: f32,
    /// Distance fade end.
    pub fade_end: f32,
    /// Sort order (higher = rendered on top).
    pub sort_order: i32,
}

/// Which material channels a decal affects.
#[derive(Debug, Clone, Copy)]
pub struct DecalChannels {
    pub albedo: bool,
    pub normal: bool,
    pub metallic: bool,
    pub roughness: bool,
    pub ao: bool,
    pub emissive: bool,
}

impl Default for DecalChannels {
    fn default() -> Self {
        Self {
            albedo: true,
            normal: true,
            metallic: false,
            roughness: false,
            ao: false,
            emissive: false,
        }
    }
}

impl DecalChannels {
    /// All channels affected.
    pub fn all() -> Self {
        Self {
            albedo: true,
            normal: true,
            metallic: true,
            roughness: true,
            ao: true,
            emissive: true,
        }
    }

    /// Only albedo.
    pub fn albedo_only() -> Self {
        Self {
            albedo: true,
            ..Self::default()
        }
    }
}

impl DecalMaterial {
    /// Create a new decal.
    pub fn new(name: &str, position: Vec3, direction: Vec3, half_size: Vec3) -> Self {
        Self {
            name: name.to_string(),
            enabled: true,
            sample: MaterialSample::default(),
            blend_mode: BlendMode::Lerp,
            channels: DecalChannels::default(),
            projection: glam::Mat4::IDENTITY,
            inverse_projection: glam::Mat4::IDENTITY,
            position,
            direction: direction.normalize_or_zero(),
            half_size,
            normal_fade_angle: 1.0, // ~57 degrees
            fade_start: 0.8,
            fade_end: 1.0,
            sort_order: 0,
        }
    }

    /// Project a world position into decal UV space.
    pub fn project(&self, world_pos: Vec3) -> Option<Vec3> {
        let local = world_pos - self.position;
        // Simplified axis-aligned projection
        let u = local.x / (self.half_size.x + EPSILON);
        let v = local.y / (self.half_size.y + EPSILON);
        let w = local.z / (self.half_size.z + EPSILON);

        if u.abs() <= 1.0 && v.abs() <= 1.0 && w.abs() <= 1.0 {
            Some(Vec3::new(
                (u + 1.0) * 0.5,
                (v + 1.0) * 0.5,
                (w + 1.0) * 0.5,
            ))
        } else {
            None
        }
    }

    /// Compute the fade factor based on normal angle and distance from projector.
    pub fn compute_fade(&self, surface_normal: Vec3, projected_depth: f32) -> f32 {
        let normal_dot = surface_normal.dot(-self.direction).max(0.0);
        let normal_fade = if self.normal_fade_angle > EPSILON {
            (normal_dot / self.normal_fade_angle).clamp(0.0, 1.0)
        } else {
            if normal_dot > 0.0 { 1.0 } else { 0.0 }
        };

        let depth_fade = if self.fade_end > self.fade_start {
            let d = projected_depth.abs();
            1.0 - ((d - self.fade_start) / (self.fade_end - self.fade_start)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        normal_fade * depth_fade
    }
}

// ---------------------------------------------------------------------------
// Material masks
// ---------------------------------------------------------------------------

/// A material mask controlling per-pixel blending between layers.
#[derive(Debug, Clone)]
pub struct MaterialMask {
    /// Name of the mask.
    pub name: String,
    /// Which channel to use (R, G, B, A).
    pub channel: MaskChannel,
    /// Invert the mask.
    pub invert: bool,
    /// Contrast adjustment.
    pub contrast: f32,
    /// Brightness adjustment.
    pub brightness: f32,
    /// UV tiling for the mask texture.
    pub tiling: Vec2,
    /// UV offset.
    pub offset: Vec2,
}

/// Which texture channel to use for masking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskChannel {
    Red,
    Green,
    Blue,
    Alpha,
    Luminance,
}

impl MaterialMask {
    /// Create a new mask.
    pub fn new(name: &str, channel: MaskChannel) -> Self {
        Self {
            name: name.to_string(),
            channel,
            invert: false,
            contrast: 1.0,
            brightness: 0.0,
            tiling: Vec2::ONE,
            offset: Vec2::ZERO,
        }
    }

    /// Process a raw mask value through contrast and brightness adjustments.
    pub fn process(&self, raw_value: f32) -> f32 {
        let mut v = raw_value;

        if self.invert {
            v = 1.0 - v;
        }

        // Contrast: remap around 0.5
        v = ((v - 0.5) * self.contrast + 0.5 + self.brightness).clamp(0.0, 1.0);

        v
    }

    /// Extract the appropriate channel from an RGBA value.
    pub fn extract(&self, rgba: Vec4) -> f32 {
        let raw = match self.channel {
            MaskChannel::Red => rgba.x,
            MaskChannel::Green => rgba.y,
            MaskChannel::Blue => rgba.z,
            MaskChannel::Alpha => rgba.w,
            MaskChannel::Luminance => rgba.x * 0.299 + rgba.y * 0.587 + rgba.z * 0.114,
        };
        self.process(raw)
    }
}

// ---------------------------------------------------------------------------
// Parallax Occlusion Mapping
// ---------------------------------------------------------------------------

/// Configuration for parallax occlusion mapping.
#[derive(Debug, Clone)]
pub struct ParallaxConfig {
    /// Whether POM is enabled.
    pub enabled: bool,
    /// Height scale (depth of the parallax effect in UV units).
    pub height_scale: f32,
    /// Number of marching steps.
    pub step_count: u32,
    /// Number of binary search refinement steps.
    pub refinement_steps: u32,
    /// Whether to compute self-shadowing.
    pub self_shadow: bool,
    /// Number of shadow steps.
    pub shadow_steps: u32,
    /// Whether to clip silhouette edges.
    pub clip_edges: bool,
    /// Minimum number of steps (varies with view angle).
    pub min_steps: u32,
    /// Maximum number of steps (varies with view angle).
    pub max_steps: u32,
    /// LOD bias for the height map.
    pub lod_bias: f32,
}

impl Default for ParallaxConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            height_scale: 0.05,
            step_count: DEFAULT_POM_STEPS,
            refinement_steps: DEFAULT_POM_REFINEMENT_STEPS,
            self_shadow: false,
            shadow_steps: 16,
            clip_edges: true,
            min_steps: 8,
            max_steps: 64,
            lod_bias: 0.0,
        }
    }
}

/// Parallax occlusion mapping calculator.
#[derive(Debug)]
pub struct ParallaxOcclusionMapper {
    pub config: ParallaxConfig,
}

impl ParallaxOcclusionMapper {
    pub fn new(config: ParallaxConfig) -> Self {
        Self { config }
    }

    /// Compute the adjusted UV using steep parallax mapping.
    ///
    /// `view_dir_ts`: view direction in tangent space.
    /// `height_fn`: function that returns height value (0-1) at a UV coordinate.
    pub fn compute_uv(
        &self,
        uv: Vec2,
        view_dir_ts: Vec3,
        height_fn: &dyn Fn(Vec2) -> f32,
    ) -> (Vec2, f32) {
        if !self.config.enabled || view_dir_ts.z.abs() < EPSILON {
            return (uv, 1.0);
        }

        // Compute step count based on view angle
        let angle_factor = view_dir_ts.z.abs();
        let step_count = (self.config.max_steps as f32
            + (self.config.min_steps as f32 - self.config.max_steps as f32) * angle_factor)
            as u32;
        let step_count = step_count.max(1);

        let layer_depth = 1.0 / step_count as f32;
        let delta_uv = Vec2::new(-view_dir_ts.x, -view_dir_ts.y) / view_dir_ts.z
            * self.config.height_scale
            / step_count as f32;

        let mut current_uv = uv;
        let mut current_depth = 0.0f32;
        let mut current_height = height_fn(current_uv);

        // Linear search
        while current_depth < current_height {
            current_uv += delta_uv;
            current_depth += layer_depth;
            current_height = height_fn(current_uv);
        }

        // Binary search refinement
        let mut prev_uv = current_uv - delta_uv;
        let mut step_uv = delta_uv;
        let mut step_depth = layer_depth;

        for _ in 0..self.config.refinement_steps {
            step_uv *= 0.5;
            step_depth *= 0.5;
            let mid_uv = prev_uv + step_uv;
            let mid_height = height_fn(mid_uv);
            let mid_depth = current_depth - step_depth;

            if mid_depth < mid_height {
                prev_uv = mid_uv;
            } else {
                current_uv = mid_uv;
                current_depth = mid_depth;
            }
        }

        // Self-shadowing
        let shadow = if self.config.self_shadow {
            self.compute_self_shadow(current_uv, current_depth, view_dir_ts, height_fn)
        } else {
            1.0
        };

        (current_uv, shadow)
    }

    /// Compute self-shadow factor for POM.
    fn compute_self_shadow(
        &self,
        uv: Vec2,
        depth: f32,
        light_dir_ts: Vec3,
        height_fn: &dyn Fn(Vec2) -> f32,
    ) -> f32 {
        if light_dir_ts.z <= EPSILON {
            return 0.0; // Light below surface
        }

        let step_count = self.config.shadow_steps;
        let layer_depth = depth / step_count as f32;
        let delta_uv = Vec2::new(light_dir_ts.x, light_dir_ts.y) / light_dir_ts.z
            * self.config.height_scale
            / step_count as f32;

        let mut current_uv = uv;
        let mut current_depth = depth;
        let mut shadow = 1.0f32;

        for _ in 0..step_count {
            current_uv += delta_uv;
            current_depth -= layer_depth;

            if current_depth < 0.0 {
                break;
            }

            let height = height_fn(current_uv);
            if height > current_depth {
                let occlude = (height - current_depth) * 10.0;
                shadow = shadow.min(1.0 - occlude.clamp(0.0, 1.0));
            }
        }

        shadow.max(0.2)
    }
}

// ---------------------------------------------------------------------------
// Clearcoat layer
// ---------------------------------------------------------------------------

/// Configuration for a clearcoat layer (e.g. automotive paint, lacquer).
#[derive(Debug, Clone)]
pub struct ClearcoatConfig {
    /// Whether clearcoat is enabled.
    pub enabled: bool,
    /// Clearcoat intensity (0 = no clearcoat, 1 = full).
    pub intensity: f32,
    /// Clearcoat roughness (typically very smooth, 0.0 - 0.2).
    pub roughness: f32,
    /// Index of refraction for the clearcoat layer.
    pub ior: f32,
    /// Normal map influence on the clearcoat layer (typically reduced).
    pub normal_influence: f32,
    /// Tint colour for the clearcoat (clear = white).
    pub tint: Vec3,
    /// Whether to use a separate normal map for the clearcoat.
    pub separate_normal: bool,
    /// Clearcoat normal (if separate_normal is true).
    pub clearcoat_normal: Vec3,
    /// Thickness of the clearcoat (affects Fresnel).
    pub thickness: f32,
}

impl Default for ClearcoatConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 1.0,
            roughness: 0.04,
            ior: 1.5,
            normal_influence: 0.5,
            tint: Vec3::ONE,
            separate_normal: false,
            clearcoat_normal: Vec3::new(0.0, 0.0, 1.0),
            thickness: 1.0,
        }
    }
}

impl ClearcoatConfig {
    /// Compute the Fresnel term at normal incidence (F0) from the IOR.
    pub fn f0(&self) -> f32 {
        let r = (self.ior - 1.0) / (self.ior + 1.0);
        r * r
    }

    /// Schlick Fresnel approximation.
    pub fn fresnel(&self, cos_theta: f32) -> f32 {
        let f0 = self.f0();
        f0 + (1.0 - f0) * (1.0 - cos_theta).max(0.0).powi(5)
    }

    /// Compute the clearcoat contribution to the final colour.
    pub fn evaluate(
        &self,
        base_color: Vec3,
        view_dir: Vec3,
        normal: Vec3,
        light_dir: Vec3,
    ) -> Vec3 {
        if !self.enabled || self.intensity < EPSILON {
            return base_color;
        }

        let n = if self.separate_normal {
            let blended = Vec3::lerp(normal, self.clearcoat_normal, self.normal_influence);
            blended.normalize_or_zero()
        } else {
            normal
        };

        let half_vec = (view_dir + light_dir).normalize_or_zero();
        let n_dot_h = n.dot(half_vec).max(0.0);
        let n_dot_v = n.dot(view_dir).max(0.0);

        let fresnel = self.fresnel(n_dot_v);

        // Simple GGX for clearcoat
        let a = self.roughness * self.roughness;
        let a2 = a * a;
        let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
        let ndf = a2 / (std::f32::consts::PI * d * d + EPSILON);

        let clearcoat_specular = self.tint * fresnel * ndf * self.intensity;

        // Energy conservation: clearcoat absorbs some light
        let energy_compensation = 1.0 - fresnel * self.intensity;
        base_color * energy_compensation + clearcoat_specular
    }
}

// ---------------------------------------------------------------------------
// Layered material (complete)
// ---------------------------------------------------------------------------

/// A complete layered material combining all features.
#[derive(Debug)]
pub struct LayeredMaterial {
    /// Name.
    pub name: String,
    /// Base layers.
    pub layers: Vec<MaterialLayer>,
    /// Detail materials.
    pub details: Vec<DetailMaterial>,
    /// Decal materials.
    pub decals: Vec<DecalMaterial>,
    /// Material masks.
    pub masks: Vec<MaterialMask>,
    /// Parallax occlusion mapping.
    pub parallax: ParallaxOcclusionMapper,
    /// Clearcoat layer.
    pub clearcoat: ClearcoatConfig,
    /// Global opacity.
    pub opacity: f32,
    /// Double-sided rendering.
    pub double_sided: bool,
}

impl LayeredMaterial {
    /// Create a new layered material with a single base layer.
    pub fn new(name: &str, base: MaterialSample) -> Self {
        let mut base_layer = MaterialLayer::new("Base");
        base_layer.base_sample = base;
        Self {
            name: name.to_string(),
            layers: vec![base_layer],
            details: Vec::new(),
            decals: Vec::new(),
            masks: Vec::new(),
            parallax: ParallaxOcclusionMapper::new(ParallaxConfig::default()),
            clearcoat: ClearcoatConfig::default(),
            opacity: 1.0,
            double_sided: false,
        }
    }

    /// Add a layer.
    pub fn add_layer(&mut self, layer: MaterialLayer) {
        if self.layers.len() < MAX_MATERIAL_LAYERS {
            self.layers.push(layer);
        }
    }

    /// Add a detail material.
    pub fn add_detail(&mut self, detail: DetailMaterial) {
        self.details.push(detail);
    }

    /// Add a decal.
    pub fn add_decal(&mut self, decal: DecalMaterial) {
        if self.decals.len() < MAX_DECAL_PROJECTORS {
            self.decals.push(decal);
        }
    }

    /// Add a mask.
    pub fn add_mask(&mut self, mask: MaterialMask) {
        self.masks.push(mask);
    }

    /// Enable clearcoat.
    pub fn set_clearcoat(&mut self, config: ClearcoatConfig) {
        self.clearcoat = config;
    }

    /// Enable parallax mapping.
    pub fn set_parallax(&mut self, config: ParallaxConfig) {
        self.parallax = ParallaxOcclusionMapper::new(config);
    }

    /// Total layer count (base + overlays).
    pub fn total_layer_count(&self) -> usize {
        self.layers.len() + self.details.len() + self.decals.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_modes() {
        assert!((BlendMode::Lerp.blend_scalar(0.0, 1.0, 0.5) - 0.5).abs() < EPSILON);
        assert!((BlendMode::Additive.blend_scalar(0.5, 0.5, 1.0) - 1.0).abs() < EPSILON);
        assert!((BlendMode::Multiply.blend_scalar(0.5, 0.5, 1.0) - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_height_blend() {
        let result = height_blend(1.0, 0.0, 0.5, 10.0);
        // Should favour the taller layer (height_a = 1.0)
        assert!(result < 0.5);
    }

    #[test]
    fn test_material_sample_lerp() {
        let a = MaterialSample::diffuse(Vec3::new(1.0, 0.0, 0.0));
        let b = MaterialSample::diffuse(Vec3::new(0.0, 0.0, 1.0));
        let mid = MaterialSample::lerp(&a, &b, 0.5);
        assert!((mid.albedo.x - 0.5).abs() < EPSILON);
        assert!((mid.albedo.z - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_clearcoat_fresnel() {
        let cc = ClearcoatConfig::default();
        let f0 = cc.f0();
        assert!(f0 > 0.0 && f0 < 1.0);
        let f_edge = cc.fresnel(0.0);
        assert!((f_edge - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_decal_projection() {
        let decal = DecalMaterial::new("test", Vec3::ZERO, Vec3::NEG_Z, Vec3::ONE);
        let inside = decal.project(Vec3::new(0.5, 0.5, 0.5));
        assert!(inside.is_some());
        let outside = decal.project(Vec3::new(5.0, 5.0, 5.0));
        assert!(outside.is_none());
    }

    #[test]
    fn test_mask_processing() {
        let mask = MaterialMask {
            name: "test".to_string(),
            channel: MaskChannel::Red,
            invert: true,
            contrast: 1.0,
            brightness: 0.0,
            tiling: Vec2::ONE,
            offset: Vec2::ZERO,
        };
        let result = mask.extract(Vec4::new(0.3, 0.5, 0.7, 1.0));
        assert!((result - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_detail_weight() {
        let snow = DetailMaterial::snow();
        let upward = snow.evaluate_weight(Vec3::Y, 0.0);
        assert!(upward > 0.0);
        let downward = snow.evaluate_weight(Vec3::NEG_Y, 0.0);
        assert!((downward - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_layered_material() {
        let mut mat = LayeredMaterial::new("test", MaterialSample::diffuse(Vec3::ONE));
        mat.add_layer(MaterialLayer::new("overlay"));
        mat.add_detail(DetailMaterial::wear());
        mat.add_decal(DecalMaterial::new("blood", Vec3::ZERO, Vec3::NEG_Z, Vec3::ONE));
        assert_eq!(mat.total_layer_count(), 4);
    }
}
