// engine/render/src/pbr/material.rs
//
// PBR material definitions, material instances with GPU resource bindings,
// and a material library for managing named materials at runtime.

use crate::pbr::texture_slots::{MaterialTextureSet, TextureSlot};
use glam::{Vec3, Vec4};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// MaterialHandle
// ---------------------------------------------------------------------------

/// Opaque handle to a material stored in the `MaterialLibrary`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialHandle(pub(crate) u64);

impl MaterialHandle {
    /// A sentinel handle that refers to no material.
    pub const INVALID: Self = Self(u64::MAX);

    /// Returns `true` if the handle is valid (not the sentinel).
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != u64::MAX
    }
}

impl Default for MaterialHandle {
    fn default() -> Self {
        Self::INVALID
    }
}

static NEXT_MATERIAL_ID: AtomicU64 = AtomicU64::new(1);

fn alloc_material_id() -> u64 {
    NEXT_MATERIAL_ID.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// AlphaMode
// ---------------------------------------------------------------------------

/// How transparency is handled for this material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlphaMode {
    /// Fully opaque. Alpha channel is ignored.
    Opaque,
    /// Alpha-tested: pixels below the cutoff are discarded.
    Mask { cutoff: f32 },
    /// Alpha-blended with the framebuffer using standard over compositing.
    Blend,
    /// Pre-multiplied alpha blending.
    Premultiplied,
    /// Additive blending (e.g. fire, glow effects).
    Additive,
}

impl Default for AlphaMode {
    fn default() -> Self {
        Self::Opaque
    }
}

impl AlphaMode {
    /// Returns the alpha cutoff, or 0.0 if not in mask mode.
    pub fn cutoff(self) -> f32 {
        match self {
            Self::Mask { cutoff } => cutoff,
            _ => 0.0,
        }
    }

    /// Returns `true` if blending is required (material is not fully opaque
    /// or alpha-tested).
    pub fn requires_blending(self) -> bool {
        matches!(self, Self::Blend | Self::Premultiplied | Self::Additive)
    }

    /// Returns a sort-order priority. Lower values are rendered first.
    /// Opaque geometry renders before transparent.
    pub fn render_order(self) -> u8 {
        match self {
            Self::Opaque => 0,
            Self::Mask { .. } => 1,
            Self::Premultiplied => 2,
            Self::Blend => 3,
            Self::Additive => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Dirty flags
// ---------------------------------------------------------------------------

bitflags::bitflags! {
    /// Tracks which parts of a material have changed since the last GPU upload.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MaterialDirtyFlags: u32 {
        const ALBEDO        = 0b0000_0001;
        const METALLIC      = 0b0000_0010;
        const ROUGHNESS     = 0b0000_0100;
        const NORMAL        = 0b0000_1000;
        const EMISSIVE      = 0b0001_0000;
        const ALPHA         = 0b0010_0000;
        const TEXTURES      = 0b0100_0000;
        const CLEARCOAT     = 0b1000_0000;
        const ANISOTROPY    = 0b0001_0000_0000;
        const SUBSURFACE    = 0b0010_0000_0000;
        const SHEEN         = 0b0100_0000_0000;
        const ALL           = 0xFFFF_FFFF;
    }
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// A PBR material definition. This is the CPU-side description that is used
/// to create `MaterialInstance`s with actual GPU resources.
///
/// Uses the metallic-roughness workflow (as in glTF 2.0):
/// - `albedo_color` — base colour (linear RGB + alpha).
/// - `metallic` — 0.0 = dielectric, 1.0 = metal.
/// - `roughness` — 0.0 = perfectly smooth, 1.0 = fully rough.
/// - `normal_scale` — strength of the normal map effect.
/// - `emissive_color` — self-illumination colour (linear RGB).
/// - `emissive_strength` — multiplier for `emissive_color`.
/// - `ao_strength` — ambient occlusion strength (0..1).
/// - `alpha_mode` — how transparency is handled.
/// - `double_sided` — disables backface culling when `true`.
#[derive(Debug, Clone)]
pub struct Material {
    /// Unique identifier.
    pub id: u64,
    /// Human-readable name (for debugging and the material library).
    pub name: String,

    // -- Core PBR parameters --------------------------------------------------
    /// Base colour in linear space. Alpha component is used for transparency.
    pub albedo_color: Vec4,
    /// Metallic factor (0.0 = dielectric, 1.0 = conductor).
    pub metallic: f32,
    /// Perceptual roughness (0.0 = mirror, 1.0 = fully diffuse).
    pub roughness: f32,
    /// Reflectance at normal incidence for dielectrics (default 0.5 = 4% F0).
    pub reflectance: f32,
    /// Strength of the normal map effect.
    pub normal_scale: f32,
    /// Self-illumination colour (linear RGB).
    pub emissive_color: Vec3,
    /// Multiplier for `emissive_color`. Allows HDR bloom.
    pub emissive_strength: f32,
    /// Ambient occlusion strength (0.0 = none, 1.0 = full).
    pub ao_strength: f32,

    // -- Clearcoat layer ------------------------------------------------------
    /// Clearcoat intensity (0.0 = disabled, 1.0 = full).
    pub clearcoat: f32,
    /// Clearcoat roughness.
    pub clearcoat_roughness: f32,

    // -- Anisotropy -----------------------------------------------------------
    /// Anisotropy strength (-1..1). 0.0 = isotropic.
    pub anisotropy: f32,
    /// Anisotropy rotation in radians.
    pub anisotropy_rotation: f32,

    // -- Subsurface scattering ------------------------------------------------
    /// Subsurface scattering strength (0.0 = disabled).
    pub subsurface: f32,
    /// Subsurface scattering colour tint.
    pub subsurface_color: Vec3,

    // -- Sheen (for fabrics) --------------------------------------------------
    /// Sheen colour.
    pub sheen_color: Vec3,
    /// Sheen roughness.
    pub sheen_roughness: f32,

    // -- Alpha / blending -----------------------------------------------------
    /// How transparency is handled.
    pub alpha_mode: AlphaMode,
    /// Disables backface culling when `true`.
    pub double_sided: bool,

    // -- Texture bindings (CPU description) -----------------------------------
    /// Which texture slots are assigned.
    pub textures: MaterialTextureSet,

    // -- Internal bookkeeping -------------------------------------------------
    /// Dirty flags tracking which parameters have changed.
    pub(crate) dirty: MaterialDirtyFlags,
    /// Version counter, incremented on every parameter change.
    pub(crate) version: u64,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            id: alloc_material_id(),
            name: String::from("Unnamed Material"),
            albedo_color: Vec4::ONE,
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            normal_scale: 1.0,
            emissive_color: Vec3::ZERO,
            emissive_strength: 1.0,
            ao_strength: 1.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            subsurface: 0.0,
            subsurface_color: Vec3::ONE,
            sheen_color: Vec3::ZERO,
            sheen_roughness: 0.5,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            textures: MaterialTextureSet::new(),
            dirty: MaterialDirtyFlags::ALL,
            version: 0,
        }
    }
}

impl Material {
    /// Create a new material with the given name and default PBR parameters.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    // -- Setters with dirty tracking ------------------------------------------

    /// Set the base albedo colour (linear RGBA).
    pub fn set_albedo_color(&mut self, color: Vec4) -> &mut Self {
        if self.albedo_color != color {
            self.albedo_color = color;
            self.mark_dirty(MaterialDirtyFlags::ALBEDO);
        }
        self
    }

    /// Set the metallic factor.
    pub fn set_metallic(&mut self, metallic: f32) -> &mut Self {
        let metallic = metallic.clamp(0.0, 1.0);
        if (self.metallic - metallic).abs() > f32::EPSILON {
            self.metallic = metallic;
            self.mark_dirty(MaterialDirtyFlags::METALLIC);
        }
        self
    }

    /// Set the roughness factor.
    pub fn set_roughness(&mut self, roughness: f32) -> &mut Self {
        let roughness = roughness.clamp(0.0, 1.0);
        if (self.roughness - roughness).abs() > f32::EPSILON {
            self.roughness = roughness;
            self.mark_dirty(MaterialDirtyFlags::ROUGHNESS);
        }
        self
    }

    /// Set the reflectance at normal incidence for dielectrics.
    /// A value of 0.5 corresponds to 4% reflectance (F0 = 0.04), which is
    /// typical for most non-metallic surfaces. The actual F0 is computed as
    /// `0.16 * reflectance * reflectance`.
    pub fn set_reflectance(&mut self, reflectance: f32) -> &mut Self {
        let reflectance = reflectance.clamp(0.0, 1.0);
        if (self.reflectance - reflectance).abs() > f32::EPSILON {
            self.reflectance = reflectance;
            self.mark_dirty(MaterialDirtyFlags::METALLIC);
        }
        self
    }

    /// Set the normal map strength.
    pub fn set_normal_scale(&mut self, scale: f32) -> &mut Self {
        if (self.normal_scale - scale).abs() > f32::EPSILON {
            self.normal_scale = scale;
            self.mark_dirty(MaterialDirtyFlags::NORMAL);
        }
        self
    }

    /// Set the emissive colour and strength.
    pub fn set_emissive(&mut self, color: Vec3, strength: f32) -> &mut Self {
        if self.emissive_color != color || (self.emissive_strength - strength).abs() > f32::EPSILON
        {
            self.emissive_color = color;
            self.emissive_strength = strength;
            self.mark_dirty(MaterialDirtyFlags::EMISSIVE);
        }
        self
    }

    /// Set the ambient occlusion strength (0.0 = no AO, 1.0 = full AO).
    pub fn set_ao_strength(&mut self, strength: f32) -> &mut Self {
        let strength = strength.clamp(0.0, 1.0);
        if (self.ao_strength - strength).abs() > f32::EPSILON {
            self.ao_strength = strength;
            self.mark_dirty(MaterialDirtyFlags::TEXTURES);
        }
        self
    }

    /// Set the alpha mode.
    pub fn set_alpha_mode(&mut self, mode: AlphaMode) -> &mut Self {
        if self.alpha_mode != mode {
            self.alpha_mode = mode;
            self.mark_dirty(MaterialDirtyFlags::ALPHA);
        }
        self
    }

    /// Set whether the material is double-sided (disables backface culling).
    pub fn set_double_sided(&mut self, double_sided: bool) -> &mut Self {
        if self.double_sided != double_sided {
            self.double_sided = double_sided;
            self.mark_dirty(MaterialDirtyFlags::ALPHA);
        }
        self
    }

    /// Set clearcoat parameters.
    pub fn set_clearcoat(&mut self, intensity: f32, roughness: f32) -> &mut Self {
        let intensity = intensity.clamp(0.0, 1.0);
        let roughness = roughness.clamp(0.0, 1.0);
        if (self.clearcoat - intensity).abs() > f32::EPSILON
            || (self.clearcoat_roughness - roughness).abs() > f32::EPSILON
        {
            self.clearcoat = intensity;
            self.clearcoat_roughness = roughness;
            self.mark_dirty(MaterialDirtyFlags::CLEARCOAT);
        }
        self
    }

    /// Set anisotropy parameters.
    pub fn set_anisotropy(&mut self, strength: f32, rotation: f32) -> &mut Self {
        let strength = strength.clamp(-1.0, 1.0);
        if (self.anisotropy - strength).abs() > f32::EPSILON
            || (self.anisotropy_rotation - rotation).abs() > f32::EPSILON
        {
            self.anisotropy = strength;
            self.anisotropy_rotation = rotation;
            self.mark_dirty(MaterialDirtyFlags::ANISOTROPY);
        }
        self
    }

    /// Set subsurface scattering parameters.
    pub fn set_subsurface(&mut self, strength: f32, color: Vec3) -> &mut Self {
        let strength = strength.clamp(0.0, 1.0);
        if (self.subsurface - strength).abs() > f32::EPSILON || self.subsurface_color != color {
            self.subsurface = strength;
            self.subsurface_color = color;
            self.mark_dirty(MaterialDirtyFlags::SUBSURFACE);
        }
        self
    }

    /// Set sheen parameters (for fabric-like materials).
    pub fn set_sheen(&mut self, color: Vec3, roughness: f32) -> &mut Self {
        let roughness = roughness.clamp(0.0, 1.0);
        if self.sheen_color != color || (self.sheen_roughness - roughness).abs() > f32::EPSILON {
            self.sheen_color = color;
            self.sheen_roughness = roughness;
            self.mark_dirty(MaterialDirtyFlags::SHEEN);
        }
        self
    }

    // -- Dirty flag helpers ---------------------------------------------------

    fn mark_dirty(&mut self, flags: MaterialDirtyFlags) {
        self.dirty |= flags;
        self.version += 1;
    }

    /// Returns `true` if any parameter has changed since the last
    /// `clear_dirty()` call.
    pub fn is_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    /// Returns the current dirty flags.
    pub fn dirty_flags(&self) -> MaterialDirtyFlags {
        self.dirty
    }

    /// Clear the dirty flags (call after uploading to GPU).
    pub fn clear_dirty(&mut self) {
        self.dirty = MaterialDirtyFlags::empty();
    }

    // -- Sorting key ----------------------------------------------------------

    /// Compute a 64-bit sorting key for render-queue ordering.
    ///
    /// The key layout (MSB to LSB):
    /// - Bits 62..63: alpha mode render order (2 bits)
    /// - Bits 32..61: material id (30 bits)
    /// - Bits 16..31: pipeline variant hash (16 bits)
    /// - Bits  0..15: reserved / distance
    ///
    /// This groups draw calls by transparency mode first, then by material to
    /// minimise GPU state changes.
    pub fn sort_key(&self) -> u64 {
        let alpha_order = (self.alpha_mode.render_order() as u64) << 62;
        let mat_id_bits = (self.id & 0x3FFF_FFFF) << 32;
        let pipeline_bits = (self.pipeline_variant_hash() as u64) << 16;
        alpha_order | mat_id_bits | pipeline_bits
    }

    /// Compute a hash representing the pipeline variant this material requires.
    /// Materials with the same variant hash can share a pipeline.
    fn pipeline_variant_hash(&self) -> u16 {
        let mut h: u16 = 0;
        if self.textures.has_slot(TextureSlot::Albedo) {
            h |= 1 << 0;
        }
        if self.textures.has_slot(TextureSlot::Normal) {
            h |= 1 << 1;
        }
        if self.textures.has_slot(TextureSlot::MetallicRoughness) {
            h |= 1 << 2;
        }
        if self.textures.has_slot(TextureSlot::AO) {
            h |= 1 << 3;
        }
        if self.textures.has_slot(TextureSlot::Emissive) {
            h |= 1 << 4;
        }
        if self.textures.has_slot(TextureSlot::Height) {
            h |= 1 << 5;
        }
        if self.double_sided {
            h |= 1 << 6;
        }
        match self.alpha_mode {
            AlphaMode::Opaque => {}
            AlphaMode::Mask { .. } => h |= 1 << 7,
            AlphaMode::Blend => h |= 1 << 8,
            AlphaMode::Premultiplied => h |= 3 << 7,
            AlphaMode::Additive => h |= 1 << 9,
        }
        if self.clearcoat > 0.0 {
            h |= 1 << 10;
        }
        if self.anisotropy.abs() > f32::EPSILON {
            h |= 1 << 11;
        }
        if self.subsurface > 0.0 {
            h |= 1 << 12;
        }
        if self.sheen_color.length_squared() > f32::EPSILON {
            h |= 1 << 13;
        }
        h
    }

    /// Compute the F0 (reflectance at normal incidence) for this material.
    /// For dielectrics this is derived from `reflectance`; for metals it is
    /// the albedo colour.
    pub fn compute_f0(&self) -> Vec3 {
        let dielectric_f0 = 0.16 * self.reflectance * self.reflectance;
        let dielectric = Vec3::splat(dielectric_f0);
        let metallic_f0 = self.albedo_color.truncate();
        dielectric * (1.0 - self.metallic) + metallic_f0 * self.metallic
    }

    /// Compute the diffuse colour. For metals this is black (metals have no
    /// diffuse reflection); for dielectrics it is the albedo colour.
    pub fn compute_diffuse_color(&self) -> Vec3 {
        self.albedo_color.truncate() * (1.0 - self.metallic)
    }
}

// ---------------------------------------------------------------------------
// GPU-compatible uniform struct
// ---------------------------------------------------------------------------

/// Material parameters packed for GPU upload. Must be kept in sync with the
/// material uniform layout in the PBR fragment shader.
///
/// Layout:  128 bytes, 16-byte aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    /// Base colour (linear RGBA).
    pub albedo_color: [f32; 4],
    /// Emissive colour (linear RGB) + emissive strength in W.
    pub emissive: [f32; 4],
    /// x=metallic, y=roughness, z=reflectance, w=normal_scale.
    pub metallic_roughness: [f32; 4],
    /// x=ao_strength, y=alpha_cutoff, z=clearcoat, w=clearcoat_roughness.
    pub ao_alpha_clearcoat: [f32; 4],
    /// x=anisotropy, y=anisotropy_rotation, z=subsurface, w=sheen_roughness.
    pub aniso_subsurface_sheen: [f32; 4],
    /// Subsurface colour (RGB) + padding.
    pub subsurface_color: [f32; 4],
    /// Sheen colour (RGB) + feature flags bitmask in W (as float bits).
    pub sheen_color_flags: [f32; 4],
    /// Reserved for future use / padding.
    pub _reserved: [f32; 4],
}

impl MaterialUniform {
    /// Build a `MaterialUniform` from a `Material`.
    pub fn from_material(mat: &Material) -> Self {
        let flags = mat.pipeline_variant_hash() as u32;
        Self {
            albedo_color: mat.albedo_color.to_array(),
            emissive: [
                mat.emissive_color.x,
                mat.emissive_color.y,
                mat.emissive_color.z,
                mat.emissive_strength,
            ],
            metallic_roughness: [mat.metallic, mat.roughness, mat.reflectance, mat.normal_scale],
            ao_alpha_clearcoat: [
                mat.ao_strength,
                mat.alpha_mode.cutoff(),
                mat.clearcoat,
                mat.clearcoat_roughness,
            ],
            aniso_subsurface_sheen: [
                mat.anisotropy,
                mat.anisotropy_rotation,
                mat.subsurface,
                mat.sheen_roughness,
            ],
            subsurface_color: [
                mat.subsurface_color.x,
                mat.subsurface_color.y,
                mat.subsurface_color.z,
                0.0,
            ],
            sheen_color_flags: [
                mat.sheen_color.x,
                mat.sheen_color.y,
                mat.sheen_color.z,
                f32::from_bits(flags),
            ],
            _reserved: [0.0; 4],
        }
    }
}

// ---------------------------------------------------------------------------
// MaterialInstance
// ---------------------------------------------------------------------------

/// A runtime material instance with GPU resource bindings.
///
/// Each `MaterialInstance` corresponds to a single `Material` but also holds
/// the actual GPU handles (uniform buffer, bind group, textures) needed to
/// render with it.
pub struct MaterialInstance {
    /// The source material definition (shared, immutable after creation unless
    /// parameters are mutated through the library).
    pub material: Arc<Material>,
    /// Version of the material at the time of the last GPU upload.
    pub uploaded_version: u64,
    /// The GPU uniform buffer containing `MaterialUniform`.
    pub uniform_buffer: Option<crate::interface::resource::BufferHandle>,
    /// The bind group combining the uniform buffer and all texture bindings.
    pub bind_group: Option<BindGroupHandle>,
    /// Cached sort key for the render queue.
    pub cached_sort_key: u64,
}

/// Placeholder bind-group handle until we integrate with the device trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindGroupHandle(pub u64);

impl MaterialInstance {
    /// Create a new instance for a given material. GPU resources are not
    /// allocated until `upload()` is called.
    pub fn new(material: Arc<Material>) -> Self {
        let sort_key = material.sort_key();
        Self {
            material,
            uploaded_version: 0,
            uniform_buffer: None,
            bind_group: None,
            cached_sort_key: sort_key,
        }
    }

    /// Returns `true` if the GPU resources need to be (re-)uploaded because
    /// the source material has been modified.
    pub fn needs_upload(&self) -> bool {
        self.uploaded_version != self.material.version || self.uniform_buffer.is_none()
    }

    /// Build the `MaterialUniform` for the current material state.
    pub fn build_uniform(&self) -> MaterialUniform {
        MaterialUniform::from_material(&self.material)
    }

    /// Mark the instance as up-to-date after a GPU upload.
    pub fn mark_uploaded(&mut self) {
        self.uploaded_version = self.material.version;
        self.cached_sort_key = self.material.sort_key();
    }
}

// ---------------------------------------------------------------------------
// StandardMaterial presets
// ---------------------------------------------------------------------------

/// Factory functions for common PBR material presets.
pub struct StandardMaterial;

impl StandardMaterial {
    /// Default lit material (white dielectric, roughness 0.5).
    pub fn default_lit() -> Material {
        Material::new("StandardLit")
    }

    /// A pure metal material.
    pub fn metal(name: &str, albedo: Vec3, roughness: f32) -> Material {
        let mut mat = Material::new(name);
        mat.set_albedo_color(albedo.extend(1.0));
        mat.set_metallic(1.0);
        mat.set_roughness(roughness);
        mat
    }

    /// Gold preset.
    pub fn gold() -> Material {
        Self::metal("Gold", Vec3::new(1.0, 0.766, 0.336), 0.3)
    }

    /// Silver preset.
    pub fn silver() -> Material {
        Self::metal("Silver", Vec3::new(0.972, 0.960, 0.915), 0.2)
    }

    /// Copper preset.
    pub fn copper() -> Material {
        Self::metal("Copper", Vec3::new(0.955, 0.638, 0.538), 0.35)
    }

    /// Iron preset.
    pub fn iron() -> Material {
        Self::metal("Iron", Vec3::new(0.56, 0.57, 0.58), 0.45)
    }

    /// A smooth dielectric (e.g. plastic, ceramic).
    pub fn smooth_dielectric(name: &str, color: Vec3) -> Material {
        let mut mat = Material::new(name);
        mat.set_albedo_color(color.extend(1.0));
        mat.set_metallic(0.0);
        mat.set_roughness(0.1);
        mat
    }

    /// A rough dielectric (e.g. concrete, stone).
    pub fn rough_dielectric(name: &str, color: Vec3) -> Material {
        let mut mat = Material::new(name);
        mat.set_albedo_color(color.extend(1.0));
        mat.set_metallic(0.0);
        mat.set_roughness(0.9);
        mat
    }

    /// An emissive material (e.g. neon, lava).
    pub fn emissive(name: &str, color: Vec3, strength: f32) -> Material {
        let mut mat = Material::new(name);
        mat.set_albedo_color(Vec4::new(0.0, 0.0, 0.0, 1.0));
        mat.set_emissive(color, strength);
        mat
    }

    /// A glass-like transparent material.
    pub fn glass(name: &str, tint: Vec3, opacity: f32) -> Material {
        let mut mat = Material::new(name);
        mat.set_albedo_color(tint.extend(opacity));
        mat.set_metallic(0.0);
        mat.set_roughness(0.05);
        mat.set_reflectance(0.9);
        mat.set_alpha_mode(AlphaMode::Blend);
        mat
    }

    /// A clearcoat material (e.g. car paint, lacquered wood).
    pub fn clearcoat(name: &str, base_color: Vec3, coat_intensity: f32) -> Material {
        let mut mat = Material::new(name);
        mat.set_albedo_color(base_color.extend(1.0));
        mat.set_metallic(0.0);
        mat.set_roughness(0.4);
        mat.set_clearcoat(coat_intensity, 0.1);
        mat
    }

    /// A fabric material with sheen.
    pub fn fabric(name: &str, color: Vec3, sheen: Vec3) -> Material {
        let mut mat = Material::new(name);
        mat.set_albedo_color(color.extend(1.0));
        mat.set_metallic(0.0);
        mat.set_roughness(0.8);
        mat.set_sheen(sheen, 0.5);
        mat
    }

    /// A subsurface-scattering material (e.g. skin, wax, marble).
    pub fn subsurface(name: &str, color: Vec3, sss_color: Vec3, strength: f32) -> Material {
        let mut mat = Material::new(name);
        mat.set_albedo_color(color.extend(1.0));
        mat.set_metallic(0.0);
        mat.set_roughness(0.5);
        mat.set_subsurface(strength, sss_color);
        mat
    }
}

// ---------------------------------------------------------------------------
// MaterialLibrary
// ---------------------------------------------------------------------------

/// Registry of materials by name and handle. Provides O(1) lookup by handle
/// and name-based lookup for editor workflows.
pub struct MaterialLibrary {
    /// Materials indexed by handle.
    materials: HashMap<MaterialHandle, Arc<Material>>,
    /// Name-to-handle index for editor convenience.
    name_index: HashMap<String, MaterialHandle>,
    /// Material instances (one per material that has been instantiated).
    instances: HashMap<MaterialHandle, MaterialInstance>,
    /// Default material handle (always present).
    default_handle: MaterialHandle,
}

impl MaterialLibrary {
    /// Create a new library with a default material pre-registered.
    pub fn new() -> Self {
        let default_mat = Arc::new(StandardMaterial::default_lit());
        let handle = MaterialHandle(default_mat.id);

        let mut materials = HashMap::new();
        let mut name_index = HashMap::new();
        materials.insert(handle, Arc::clone(&default_mat));
        name_index.insert(default_mat.name.clone(), handle);

        Self {
            materials,
            name_index,
            instances: HashMap::new(),
            default_handle: handle,
        }
    }

    /// Returns the handle of the default material.
    pub fn default_handle(&self) -> MaterialHandle {
        self.default_handle
    }

    /// Register a new material. Returns its handle.
    pub fn add(&mut self, material: Material) -> MaterialHandle {
        let handle = MaterialHandle(material.id);
        self.name_index.insert(material.name.clone(), handle);
        self.materials.insert(handle, Arc::new(material));
        handle
    }

    /// Remove a material by handle. Returns the removed material, if any.
    pub fn remove(&mut self, handle: MaterialHandle) -> Option<Arc<Material>> {
        if handle == self.default_handle {
            log::warn!("Cannot remove the default material");
            return None;
        }
        if let Some(mat) = self.materials.remove(&handle) {
            self.name_index.remove(&mat.name);
            self.instances.remove(&handle);
            Some(mat)
        } else {
            None
        }
    }

    /// Look up a material by handle.
    pub fn get(&self, handle: MaterialHandle) -> Option<&Arc<Material>> {
        self.materials.get(&handle)
    }

    /// Look up a material by name.
    pub fn get_by_name(&self, name: &str) -> Option<&Arc<Material>> {
        self.name_index
            .get(name)
            .and_then(|h| self.materials.get(h))
    }

    /// Look up a handle by name.
    pub fn handle_by_name(&self, name: &str) -> Option<MaterialHandle> {
        self.name_index.get(name).copied()
    }

    /// Get or create a `MaterialInstance` for the given handle.
    pub fn get_instance(&mut self, handle: MaterialHandle) -> Option<&mut MaterialInstance> {
        if !self.materials.contains_key(&handle) {
            return None;
        }
        self.instances.entry(handle).or_insert_with(|| {
            let mat = self.materials[&handle].clone();
            MaterialInstance::new(mat)
        });
        self.instances.get_mut(&handle)
    }

    /// Iterate over all materials.
    pub fn iter(&self) -> impl Iterator<Item = (MaterialHandle, &Arc<Material>)> {
        self.materials.iter().map(|(&h, m)| (h, m))
    }

    /// Number of registered materials.
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// Returns `true` if the library contains no materials (this should never
    /// happen because the default material is always present).
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Collect all material instances that need a GPU upload.
    pub fn dirty_instances(&mut self) -> Vec<MaterialHandle> {
        let mut result = Vec::new();
        for (&handle, instance) in &self.instances {
            if instance.needs_upload() {
                result.push(handle);
            }
        }
        result
    }

    /// Build sorted render batches. Returns handles sorted by material sort
    /// key to minimise GPU state changes.
    pub fn sorted_handles(&self) -> Vec<MaterialHandle> {
        let mut handles: Vec<(MaterialHandle, u64)> = self
            .materials
            .iter()
            .map(|(&h, m)| (h, m.sort_key()))
            .collect();
        handles.sort_by_key(|(_, key)| *key);
        handles.into_iter().map(|(h, _)| h).collect()
    }
}

impl Default for MaterialLibrary {
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
    fn default_material_is_opaque_dielectric() {
        let mat = Material::default();
        assert_eq!(mat.alpha_mode, AlphaMode::Opaque);
        assert!((mat.metallic - 0.0).abs() < f32::EPSILON);
        assert!((mat.roughness - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn dirty_tracking_works() {
        let mut mat = Material::default();
        mat.clear_dirty();
        assert!(!mat.is_dirty());

        mat.set_metallic(1.0);
        assert!(mat.is_dirty());
        assert!(mat.dirty_flags().contains(MaterialDirtyFlags::METALLIC));
    }

    #[test]
    fn sort_key_orders_opaque_before_blend() {
        let opaque = Material::new("Opaque");
        let mut blend = Material::new("Blend");
        blend.set_alpha_mode(AlphaMode::Blend);
        assert!(opaque.sort_key() < blend.sort_key());
    }

    #[test]
    fn material_library_basics() {
        let mut lib = MaterialLibrary::new();
        assert_eq!(lib.len(), 1); // default

        let gold = StandardMaterial::gold();
        let handle = lib.add(gold);
        assert_eq!(lib.len(), 2);
        assert!(lib.get(handle).is_some());
        assert!(lib.get_by_name("Gold").is_some());

        lib.remove(handle);
        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn f0_computation() {
        let mut mat = Material::default();
        // Default dielectric: F0 = 0.16 * 0.5^2 = 0.04
        let f0 = mat.compute_f0();
        assert!((f0.x - 0.04).abs() < 1e-5);

        // Pure metal: F0 = albedo colour
        mat.set_metallic(1.0);
        mat.set_albedo_color(Vec4::new(1.0, 0.766, 0.336, 1.0));
        let f0 = mat.compute_f0();
        assert!((f0.x - 1.0).abs() < 1e-5);
        assert!((f0.y - 0.766).abs() < 1e-3);
    }

    #[test]
    fn standard_presets_are_distinct() {
        let gold = StandardMaterial::gold();
        let silver = StandardMaterial::silver();
        assert_ne!(gold.id, silver.id);
        assert_ne!(gold.name, silver.name);
    }
}
