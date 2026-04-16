// engine/render/src/material_system.rs
//
// Runtime material system for the Genovo engine. Builds on the PBR material
// primitives in `pbr::material` and provides a general-purpose material
// abstraction that is shader-agnostic. Supports arbitrary uniform data,
// texture bindings, render state, shader permutations, hot-reload, draw-call
// batching via sort keys, and JSON serialisation.

use crate::interface::resource::{ShaderHandle, TextureHandle};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// TextureId & SamplerId
// ---------------------------------------------------------------------------

/// Lightweight identifier for a texture resource within the material system.
/// This is decoupled from the GPU-level `TextureHandle` so that materials can
/// be serialised and transferred before GPU resources are allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u64);

impl TextureId {
    pub const INVALID: Self = Self(u64::MAX);

    pub fn is_valid(self) -> bool {
        self.0 != u64::MAX
    }
}

impl Default for TextureId {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Lightweight identifier for a sampler resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerId(pub u64);

impl Default for SamplerId {
    fn default() -> Self {
        Self(0)
    }
}

// ---------------------------------------------------------------------------
// UniformValue
// ---------------------------------------------------------------------------

/// A dynamically-typed value that can be stored in a material's uniform data
/// map and uploaded to the GPU as part of a uniform/storage buffer.
#[derive(Debug, Clone, PartialEq)]
pub enum UniformValue {
    /// Single-precision floating-point scalar.
    Float(f32),
    /// Two-component float vector.
    Vec2([f32; 2]),
    /// Three-component float vector.
    Vec3([f32; 3]),
    /// Four-component float vector.
    Vec4([f32; 4]),
    /// 4x4 float matrix stored in column-major order (16 floats).
    Mat4([f32; 16]),
    /// Signed 32-bit integer.
    Int(i32),
    /// Boolean (uploaded as `u32` on the GPU: 0 or 1).
    Bool(bool),
    /// Linear RGBA colour (identical layout to Vec4, but semantically distinct).
    Color([f32; 4]),
    /// Reference to a texture resource.
    Texture(TextureId),
}

impl UniformValue {
    /// Size of this value in bytes when packed for GPU upload.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float(_) => 4,
            Self::Vec2(_) => 8,
            Self::Vec3(_) => 12,
            Self::Vec4(_) | Self::Color(_) => 16,
            Self::Mat4(_) => 64,
            Self::Int(_) => 4,
            Self::Bool(_) => 4, // uploaded as u32
            Self::Texture(_) => 0, // bound separately, not in the uniform buffer
        }
    }

    /// Write the value into a byte buffer at the given offset.
    /// Returns the number of bytes written.
    pub fn write_bytes(&self, dst: &mut [u8], offset: usize) -> usize {
        match self {
            Self::Float(v) => {
                dst[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
                4
            }
            Self::Vec2(v) => {
                for (i, f) in v.iter().enumerate() {
                    dst[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&f.to_le_bytes());
                }
                8
            }
            Self::Vec3(v) => {
                for (i, f) in v.iter().enumerate() {
                    dst[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&f.to_le_bytes());
                }
                12
            }
            Self::Vec4(v) | Self::Color(v) => {
                for (i, f) in v.iter().enumerate() {
                    dst[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&f.to_le_bytes());
                }
                16
            }
            Self::Mat4(v) => {
                for (i, f) in v.iter().enumerate() {
                    dst[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&f.to_le_bytes());
                }
                64
            }
            Self::Int(v) => {
                dst[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
                4
            }
            Self::Bool(v) => {
                let u: u32 = if *v { 1 } else { 0 };
                dst[offset..offset + 4].copy_from_slice(&u.to_le_bytes());
                4
            }
            Self::Texture(_) => 0,
        }
    }

    /// Attempt to read a value back from a byte slice.
    pub fn read_float(src: &[u8], offset: usize) -> f32 {
        let bytes: [u8; 4] = src[offset..offset + 4].try_into().unwrap_or([0; 4]);
        f32::from_le_bytes(bytes)
    }

    /// Return the WGSL type name for this value kind.
    pub fn wgsl_type_name(&self) -> &'static str {
        match self {
            Self::Float(_) => "f32",
            Self::Vec2(_) => "vec2<f32>",
            Self::Vec3(_) => "vec3<f32>",
            Self::Vec4(_) | Self::Color(_) => "vec4<f32>",
            Self::Mat4(_) => "mat4x4<f32>",
            Self::Int(_) => "i32",
            Self::Bool(_) => "u32",
            Self::Texture(_) => "texture_2d<f32>",
        }
    }

    /// Return a default/zero value for a given WGSL type name.
    pub fn default_for_type(type_name: &str) -> Option<Self> {
        match type_name {
            "f32" => Some(Self::Float(0.0)),
            "vec2<f32>" => Some(Self::Vec2([0.0; 2])),
            "vec3<f32>" => Some(Self::Vec3([0.0; 3])),
            "vec4<f32>" => Some(Self::Vec4([0.0; 4])),
            "mat4x4<f32>" => Some(Self::Mat4([
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ])),
            "i32" => Some(Self::Int(0)),
            "u32" => Some(Self::Bool(false)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// BlendMode, DepthMode, CullMode
// ---------------------------------------------------------------------------

/// Blend state for a material's render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// No blending; fragment replaces framebuffer value.
    Opaque,
    /// Standard alpha-over compositing.
    AlphaBlend,
    /// Pre-multiplied alpha blending.
    PremultipliedAlpha,
    /// Additive blending (src + dst).
    Additive,
    /// Multiplicative blending (src * dst).
    Multiply,
    /// Custom blend factors.
    Custom {
        src_color: BlendFactor,
        dst_color: BlendFactor,
        src_alpha: BlendFactor,
        dst_alpha: BlendFactor,
    },
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::Opaque
    }
}

/// Blend factors used in custom blend modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
}

/// Depth test configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthMode {
    /// Depth test enabled, depth write enabled.
    ReadWrite,
    /// Depth test enabled, depth write disabled.
    ReadOnly,
    /// Depth test disabled, depth write disabled.
    Disabled,
    /// Depth test disabled, depth write enabled (rarely used).
    WriteOnly,
}

impl Default for DepthMode {
    fn default() -> Self {
        Self::ReadWrite
    }
}

/// Depth comparison function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthCompare {
    Never,
    Less,
    LessEqual,
    Equal,
    GreaterEqual,
    Greater,
    NotEqual,
    Always,
}

impl Default for DepthCompare {
    fn default() -> Self {
        Self::LessEqual
    }
}

/// Face culling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullFace {
    /// No culling -- both sides are rendered.
    None,
    /// Cull back-facing triangles (default).
    Back,
    /// Cull front-facing triangles.
    Front,
}

impl Default for CullFace {
    fn default() -> Self {
        Self::Back
    }
}

/// Combined render state for a material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderState {
    pub blend: BlendMode,
    pub depth_mode: DepthMode,
    pub depth_compare: DepthCompare,
    pub cull: CullFace,
    pub depth_bias: f32,
    pub depth_bias_slope: f32,
    pub stencil_enabled: bool,
    pub color_write_mask: u8, // RGBA bits
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            blend: BlendMode::Opaque,
            depth_mode: DepthMode::ReadWrite,
            depth_compare: DepthCompare::LessEqual,
            cull: CullFace::Back,
            depth_bias: 0.0,
            depth_bias_slope: 0.0,
            stencil_enabled: false,
            color_write_mask: 0b1111, // RGBA all enabled
        }
    }
}

impl RenderState {
    /// State suitable for transparent objects.
    pub fn transparent() -> Self {
        Self {
            blend: BlendMode::AlphaBlend,
            depth_mode: DepthMode::ReadOnly,
            depth_compare: DepthCompare::LessEqual,
            cull: CullFace::Back,
            ..Default::default()
        }
    }

    /// State suitable for shadow casters.
    pub fn shadow_caster() -> Self {
        Self {
            blend: BlendMode::Opaque,
            depth_mode: DepthMode::ReadWrite,
            depth_compare: DepthCompare::LessEqual,
            cull: CullFace::Front, // front-face culling reduces shadow acne
            depth_bias: 2.0,
            depth_bias_slope: 2.0,
            ..Default::default()
        }
    }

    /// State for wireframe overlay (no depth write, additive blend).
    pub fn wireframe_overlay() -> Self {
        Self {
            blend: BlendMode::Additive,
            depth_mode: DepthMode::ReadOnly,
            depth_compare: DepthCompare::LessEqual,
            cull: CullFace::None,
            ..Default::default()
        }
    }

    /// State for depth pre-pass (no colour writes, depth write enabled).
    pub fn depth_prepass() -> Self {
        Self {
            blend: BlendMode::Opaque,
            depth_mode: DepthMode::ReadWrite,
            depth_compare: DepthCompare::LessEqual,
            cull: CullFace::Back,
            color_write_mask: 0b0000,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// MaterialPass
// ---------------------------------------------------------------------------

/// Identifies which render pass a material (or material instance) participates
/// in. A single material definition can produce draw calls in multiple passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MaterialPass {
    /// Main opaque geometry pass.
    Opaque,
    /// Transparent geometry pass (sorted back-to-front).
    Transparent,
    /// Shadow map rendering pass.
    ShadowCaster,
    /// Depth pre-pass (populates the depth buffer before the main pass).
    DepthPrepass,
    /// Deferred G-buffer pass.
    GBuffer,
    /// Velocity buffer pass (for motion blur / TAA).
    Velocity,
    /// Custom / user-defined pass identified by index.
    Custom(u16),
}

impl MaterialPass {
    /// Returns a numeric ordering value used in sort keys.
    pub fn order(&self) -> u8 {
        match self {
            Self::DepthPrepass => 0,
            Self::ShadowCaster => 1,
            Self::GBuffer => 2,
            Self::Opaque => 3,
            Self::Velocity => 4,
            Self::Transparent => 5,
            Self::Custom(n) => 6 + (*n as u8).min(249),
        }
    }
}

// ---------------------------------------------------------------------------
// TextureBinding
// ---------------------------------------------------------------------------

/// A binding of a texture + sampler to a named slot in a material.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureBinding {
    /// Slot name (e.g. "albedo_map", "normal_map").
    pub name: String,
    /// Texture resource identifier.
    pub texture: TextureId,
    /// Sampler identifier.
    pub sampler: SamplerId,
    /// Bind group index.
    pub group: u32,
    /// Texture binding index within the group.
    pub binding_texture: u32,
    /// Sampler binding index within the group.
    pub binding_sampler: u32,
    /// UV channel to use (0 or 1).
    pub uv_channel: u32,
    /// Tiling scale.
    pub tiling: [f32; 2],
    /// UV offset.
    pub offset: [f32; 2],
}

impl TextureBinding {
    pub fn new(name: impl Into<String>, texture: TextureId) -> Self {
        Self {
            name: name.into(),
            texture,
            sampler: SamplerId::default(),
            group: 1,
            binding_texture: 0,
            binding_sampler: 1,
            uv_channel: 0,
            tiling: [1.0, 1.0],
            offset: [0.0, 0.0],
        }
    }

    pub fn with_group_binding(mut self, group: u32, tex_binding: u32, sampler_binding: u32) -> Self {
        self.group = group;
        self.binding_texture = tex_binding;
        self.binding_sampler = sampler_binding;
        self
    }

    pub fn with_tiling(mut self, tiling: [f32; 2]) -> Self {
        self.tiling = tiling;
        self
    }

    pub fn with_offset(mut self, offset: [f32; 2]) -> Self {
        self.offset = offset;
        self
    }

    pub fn with_uv_channel(mut self, channel: u32) -> Self {
        self.uv_channel = channel;
        self
    }
}

// ---------------------------------------------------------------------------
// ShaderFeatureFlags
// ---------------------------------------------------------------------------

/// Bit flags representing shader features. Used to select shader permutations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderFeatureFlags {
    flags: Vec<String>,
}

impl ShaderFeatureFlags {
    pub fn new() -> Self {
        Self { flags: Vec::new() }
    }

    pub fn enable(&mut self, flag: impl Into<String>) -> &mut Self {
        let flag = flag.into();
        if !self.flags.contains(&flag) {
            self.flags.push(flag);
            self.flags.sort();
        }
        self
    }

    pub fn disable(&mut self, flag: &str) -> &mut Self {
        self.flags.retain(|f| f != flag);
        self
    }

    pub fn is_enabled(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }

    pub fn flags(&self) -> &[String] {
        &self.flags
    }

    /// Compute a stable hash for permutation cache keys.
    pub fn hash_u64(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.flags.hash(&mut hasher);
        hasher.finish()
    }

    /// Convert to preprocessor define pairs for the shader compiler.
    pub fn as_defines(&self) -> Vec<(String, String)> {
        self.flags.iter().map(|f| (f.clone(), "1".to_string())).collect()
    }
}

impl Default for ShaderFeatureFlags {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

static NEXT_MATERIAL_SYSTEM_ID: AtomicU64 = AtomicU64::new(1);

fn alloc_id() -> u64 {
    NEXT_MATERIAL_SYSTEM_ID.fetch_add(1, Ordering::Relaxed)
}

/// A runtime material definition. Unlike the PBR-specific `pbr::Material`,
/// this is shader-agnostic: it stores arbitrary named uniforms, texture
/// bindings, render state, and a reference to a shader (by handle or name).
#[derive(Debug, Clone)]
pub struct Material {
    /// Unique runtime identifier.
    pub id: u64,
    /// Human-readable name (for debugging, editor, and the material library).
    pub name: String,
    /// Reference to the shader program. This may be a raw handle or a symbolic
    /// name resolved at bind time.
    pub shader_handle: Option<ShaderHandle>,
    /// Symbolic shader name (resolved by the `MaterialManager`).
    pub shader_name: String,
    /// Arbitrary named uniform parameters.
    pub uniform_data: HashMap<String, UniformValue>,
    /// Texture slot bindings.
    pub texture_bindings: Vec<TextureBinding>,
    /// Render state (blend, depth, cull, etc.).
    pub render_state: RenderState,
    /// Sort key used for draw-call batching. Lower values are drawn first.
    pub sort_key: u64,
    /// Instance count for hardware instancing (0 = not instanced).
    pub instance_count: u32,
    /// Which render passes this material participates in.
    pub passes: Vec<MaterialPass>,
    /// Shader feature flags driving permutation selection.
    pub features: ShaderFeatureFlags,
    /// Version counter, bumped on every mutation.
    pub version: u64,
    /// Whether this material needs its sort key recalculated.
    dirty_sort_key: bool,
    /// Tags for filtering and grouping (e.g. "terrain", "vegetation").
    pub tags: Vec<String>,
    /// Priority within its pass (lower = earlier).
    pub priority: i32,
    /// Whether the material is visible (enables quick toggling).
    pub visible: bool,
}

impl Material {
    /// Create a new material with the given name and default state.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: alloc_id(),
            name: name.into(),
            shader_handle: None,
            shader_name: String::new(),
            uniform_data: HashMap::new(),
            texture_bindings: Vec::new(),
            render_state: RenderState::default(),
            sort_key: 0,
            instance_count: 0,
            passes: vec![MaterialPass::Opaque],
            features: ShaderFeatureFlags::new(),
            version: 0,
            dirty_sort_key: true,
            tags: Vec::new(),
            priority: 0,
            visible: true,
        }
    }

    /// Create a material bound to a specific shader by name.
    pub fn with_shader(name: impl Into<String>, shader_name: impl Into<String>) -> Self {
        let mut mat = Self::new(name);
        mat.shader_name = shader_name.into();
        mat
    }

    // -- Uniform data manipulation -------------------------------------------

    /// Set a uniform parameter value. Overwrites any existing value with the
    /// same name.
    pub fn set_uniform(&mut self, name: impl Into<String>, value: UniformValue) -> &mut Self {
        self.uniform_data.insert(name.into(), value);
        self.bump_version();
        self
    }

    /// Get a uniform parameter value by name.
    pub fn get_uniform(&self, name: &str) -> Option<&UniformValue> {
        self.uniform_data.get(name)
    }

    /// Remove a uniform parameter.
    pub fn remove_uniform(&mut self, name: &str) -> Option<UniformValue> {
        let v = self.uniform_data.remove(name);
        if v.is_some() {
            self.bump_version();
        }
        v
    }

    /// Set a float uniform.
    pub fn set_float(&mut self, name: impl Into<String>, value: f32) -> &mut Self {
        self.set_uniform(name, UniformValue::Float(value))
    }

    /// Set a vec2 uniform.
    pub fn set_vec2(&mut self, name: impl Into<String>, value: [f32; 2]) -> &mut Self {
        self.set_uniform(name, UniformValue::Vec2(value))
    }

    /// Set a vec3 uniform.
    pub fn set_vec3(&mut self, name: impl Into<String>, value: [f32; 3]) -> &mut Self {
        self.set_uniform(name, UniformValue::Vec3(value))
    }

    /// Set a vec4 uniform.
    pub fn set_vec4(&mut self, name: impl Into<String>, value: [f32; 4]) -> &mut Self {
        self.set_uniform(name, UniformValue::Vec4(value))
    }

    /// Set a colour uniform.
    pub fn set_color(&mut self, name: impl Into<String>, value: [f32; 4]) -> &mut Self {
        self.set_uniform(name, UniformValue::Color(value))
    }

    /// Set a mat4 uniform.
    pub fn set_mat4(&mut self, name: impl Into<String>, value: [f32; 16]) -> &mut Self {
        self.set_uniform(name, UniformValue::Mat4(value))
    }

    /// Set an int uniform.
    pub fn set_int(&mut self, name: impl Into<String>, value: i32) -> &mut Self {
        self.set_uniform(name, UniformValue::Int(value))
    }

    /// Set a bool uniform.
    pub fn set_bool(&mut self, name: impl Into<String>, value: bool) -> &mut Self {
        self.set_uniform(name, UniformValue::Bool(value))
    }

    // -- Texture bindings ----------------------------------------------------

    /// Add a texture binding.
    pub fn add_texture_binding(&mut self, binding: TextureBinding) -> &mut Self {
        // Replace existing binding with the same name.
        if let Some(existing) = self.texture_bindings.iter_mut().find(|b| b.name == binding.name) {
            *existing = binding;
        } else {
            self.texture_bindings.push(binding);
        }
        self.bump_version();
        self
    }

    /// Remove a texture binding by name.
    pub fn remove_texture_binding(&mut self, name: &str) -> &mut Self {
        self.texture_bindings.retain(|b| b.name != name);
        self.bump_version();
        self
    }

    /// Get a texture binding by name.
    pub fn get_texture_binding(&self, name: &str) -> Option<&TextureBinding> {
        self.texture_bindings.iter().find(|b| b.name == name)
    }

    // -- Render state --------------------------------------------------------

    /// Set the render state.
    pub fn set_render_state(&mut self, state: RenderState) -> &mut Self {
        self.render_state = state;
        self.dirty_sort_key = true;
        self.bump_version();
        self
    }

    /// Set the blend mode.
    pub fn set_blend_mode(&mut self, mode: BlendMode) -> &mut Self {
        self.render_state.blend = mode;
        self.dirty_sort_key = true;
        self.bump_version();
        self
    }

    /// Set the depth mode.
    pub fn set_depth_mode(&mut self, mode: DepthMode) -> &mut Self {
        self.render_state.depth_mode = mode;
        self.bump_version();
        self
    }

    /// Set the cull face mode.
    pub fn set_cull_face(&mut self, cull: CullFace) -> &mut Self {
        self.render_state.cull = cull;
        self.bump_version();
        self
    }

    // -- Passes --------------------------------------------------------------

    /// Set the render passes this material participates in.
    pub fn set_passes(&mut self, passes: Vec<MaterialPass>) -> &mut Self {
        self.passes = passes;
        self.dirty_sort_key = true;
        self.bump_version();
        self
    }

    /// Add a render pass.
    pub fn add_pass(&mut self, pass: MaterialPass) -> &mut Self {
        if !self.passes.contains(&pass) {
            self.passes.push(pass);
        }
        self
    }

    /// Check if the material participates in a given pass.
    pub fn has_pass(&self, pass: MaterialPass) -> bool {
        self.passes.contains(&pass)
    }

    // -- Features / permutations ---------------------------------------------

    /// Enable a shader feature flag.
    pub fn enable_feature(&mut self, flag: impl Into<String>) -> &mut Self {
        self.features.enable(flag);
        self.dirty_sort_key = true;
        self.bump_version();
        self
    }

    /// Disable a shader feature flag.
    pub fn disable_feature(&mut self, flag: &str) -> &mut Self {
        self.features.disable(flag);
        self.dirty_sort_key = true;
        self.bump_version();
        self
    }

    // -- Sort key ------------------------------------------------------------

    /// Compute the sort key for draw-call batching. The key layout packs:
    ///   [63..56] pass order            (8 bits)
    ///   [55..48] blend mode / opacity  (8 bits)
    ///   [47..32] shader hash           (16 bits)
    ///   [31..16] feature flags hash    (16 bits)
    ///   [15..0]  material id           (16 bits)
    ///
    /// This groups draw calls by pass, then by transparency, then by shader,
    /// then by material -- minimising GPU state changes.
    pub fn compute_sort_key(&mut self) -> u64 {
        if !self.dirty_sort_key {
            return self.sort_key;
        }

        let pass_order = self.passes.iter().map(|p| p.order()).min().unwrap_or(3) as u64;
        let blend_order: u64 = match self.render_state.blend {
            BlendMode::Opaque => 0,
            BlendMode::Multiply => 1,
            BlendMode::PremultipliedAlpha => 2,
            BlendMode::AlphaBlend => 3,
            BlendMode::Additive => 4,
            BlendMode::Custom { .. } => 5,
        };

        let shader_hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            self.shader_name.hash(&mut h);
            (h.finish() & 0xFFFF) as u64
        };

        let feature_hash = (self.features.hash_u64() & 0xFFFF) as u64;
        let mat_id = (self.id & 0xFFFF) as u64;

        self.sort_key = (pass_order << 56)
            | (blend_order << 48)
            | (shader_hash << 32)
            | (feature_hash << 16)
            | mat_id;

        self.dirty_sort_key = false;
        self.sort_key
    }

    // -- Uniform buffer packing ----------------------------------------------

    /// Compute the total size (in bytes) needed for the uniform buffer,
    /// respecting std140 alignment rules.
    pub fn uniform_buffer_size(&self) -> usize {
        let mut size = 0usize;
        let mut sorted_keys: Vec<&String> = self.uniform_data.keys().collect();
        sorted_keys.sort(); // Stable ordering

        for key in &sorted_keys {
            if let Some(val) = self.uniform_data.get(*key) {
                let val_size = val.byte_size();
                if val_size == 0 {
                    continue; // Textures are not in the buffer
                }
                // Align to the value's natural alignment (std140 rules).
                let alignment = match val {
                    UniformValue::Vec3(_) | UniformValue::Vec4(_) | UniformValue::Color(_) => 16,
                    UniformValue::Vec2(_) => 8,
                    UniformValue::Mat4(_) => 16,
                    _ => 4,
                };
                size = (size + alignment - 1) & !(alignment - 1);
                size += val_size;
            }
        }
        // Round up to 16-byte alignment for the full buffer.
        (size + 15) & !15
    }

    /// Pack all uniform values into a byte buffer.
    /// Returns the packed bytes ready for GPU upload.
    pub fn pack_uniforms(&self) -> Vec<u8> {
        let buf_size = self.uniform_buffer_size();
        let mut buffer = vec![0u8; buf_size];
        let mut offset = 0usize;

        let mut sorted_keys: Vec<&String> = self.uniform_data.keys().collect();
        sorted_keys.sort();

        for key in &sorted_keys {
            if let Some(val) = self.uniform_data.get(*key) {
                let val_size = val.byte_size();
                if val_size == 0 {
                    continue;
                }
                let alignment = match val {
                    UniformValue::Vec3(_) | UniformValue::Vec4(_) | UniformValue::Color(_) => 16,
                    UniformValue::Vec2(_) => 8,
                    UniformValue::Mat4(_) => 16,
                    _ => 4,
                };
                offset = (offset + alignment - 1) & !(alignment - 1);
                val.write_bytes(&mut buffer, offset);
                offset += val_size;
            }
        }

        buffer
    }

    // -- Versioning ----------------------------------------------------------

    fn bump_version(&mut self) {
        self.version += 1;
    }

    /// Current version number.
    pub fn version(&self) -> u64 {
        self.version
    }
}

// ---------------------------------------------------------------------------
// MaterialInstance
// ---------------------------------------------------------------------------

/// A per-object material instance that overrides specific parameters from a
/// base `Material`. This allows many objects to share a single material
/// definition while varying some parameters (e.g. colour tint, texture).
#[derive(Debug, Clone)]
pub struct MaterialInstance {
    /// The base material this instance derives from.
    pub base_material_id: u64,
    /// Unique instance identifier.
    pub instance_id: u64,
    /// Parameter overrides (only the overridden values are stored).
    pub overrides: HashMap<String, UniformValue>,
    /// Texture binding overrides.
    pub texture_overrides: Vec<TextureBinding>,
    /// Cached sort key (derived from the base material).
    pub sort_key: u64,
    /// Instance-specific render state override (None = use base).
    pub render_state_override: Option<RenderState>,
    /// Version of the base material when this instance was last synced.
    pub base_version: u64,
    /// Instance version, bumped on override changes.
    pub instance_version: u64,
    /// GPU uniform buffer handle (managed by MaterialManager).
    pub gpu_buffer: Option<u64>,
    /// Whether this instance is visible.
    pub visible: bool,
}

impl MaterialInstance {
    /// Create a new instance derived from a base material.
    pub fn new(base: &Material) -> Self {
        Self {
            base_material_id: base.id,
            instance_id: alloc_id(),
            overrides: HashMap::new(),
            texture_overrides: Vec::new(),
            sort_key: base.sort_key,
            render_state_override: None,
            base_version: base.version,
            instance_version: 0,
            gpu_buffer: None,
            visible: true,
        }
    }

    /// Set a parameter override.
    pub fn set_override(&mut self, name: impl Into<String>, value: UniformValue) -> &mut Self {
        self.overrides.insert(name.into(), value);
        self.instance_version += 1;
        self
    }

    /// Remove a parameter override (falls back to the base material value).
    pub fn remove_override(&mut self, name: &str) -> &mut Self {
        self.overrides.remove(name);
        self.instance_version += 1;
        self
    }

    /// Get the effective value of a parameter, checking overrides first, then
    /// falling back to the base material.
    pub fn get_effective_value<'a>(&'a self, name: &str, base: &'a Material) -> Option<&'a UniformValue> {
        self.overrides.get(name).or_else(|| base.uniform_data.get(name))
    }

    /// Set a render-state override.
    pub fn set_render_state(&mut self, state: RenderState) -> &mut Self {
        self.render_state_override = Some(state);
        self.instance_version += 1;
        self
    }

    /// Get the effective render state.
    pub fn effective_render_state(&self, base: &Material) -> RenderState {
        self.render_state_override.unwrap_or(base.render_state)
    }

    /// Add a texture binding override.
    pub fn override_texture(&mut self, binding: TextureBinding) -> &mut Self {
        if let Some(existing) = self.texture_overrides.iter_mut().find(|b| b.name == binding.name) {
            *existing = binding;
        } else {
            self.texture_overrides.push(binding);
        }
        self.instance_version += 1;
        self
    }

    /// Check if this instance needs a GPU update.
    pub fn needs_update(&self, base: &Material) -> bool {
        self.base_version != base.version || self.gpu_buffer.is_none()
    }

    /// Pack the effective uniforms (base + overrides) into bytes.
    pub fn pack_uniforms(&self, base: &Material) -> Vec<u8> {
        // Build merged uniform map.
        let mut merged = base.uniform_data.clone();
        for (k, v) in &self.overrides {
            merged.insert(k.clone(), v.clone());
        }

        let buf_size = {
            let mut size = 0usize;
            let mut keys: Vec<&String> = merged.keys().collect();
            keys.sort();
            for key in &keys {
                if let Some(val) = merged.get(*key) {
                    let val_size = val.byte_size();
                    if val_size == 0 { continue; }
                    let alignment = match val {
                        UniformValue::Vec3(_) | UniformValue::Vec4(_) | UniformValue::Color(_) => 16,
                        UniformValue::Vec2(_) => 8,
                        UniformValue::Mat4(_) => 16,
                        _ => 4,
                    };
                    size = (size + alignment - 1) & !(alignment - 1);
                    size += val_size;
                }
            }
            (size + 15) & !15
        };

        let mut buffer = vec![0u8; buf_size];
        let mut offset = 0usize;
        let mut keys: Vec<&String> = merged.keys().collect();
        keys.sort();
        for key in &keys {
            if let Some(val) = merged.get(*key) {
                let val_size = val.byte_size();
                if val_size == 0 { continue; }
                let alignment = match val {
                    UniformValue::Vec3(_) | UniformValue::Vec4(_) | UniformValue::Color(_) => 16,
                    UniformValue::Vec2(_) => 8,
                    UniformValue::Mat4(_) => 16,
                    _ => 4,
                };
                offset = (offset + alignment - 1) & !(alignment - 1);
                val.write_bytes(&mut buffer, offset);
                offset += val_size;
            }
        }
        buffer
    }
}

// ---------------------------------------------------------------------------
// MaterialSortEntry
// ---------------------------------------------------------------------------

/// Entry used during draw-call sorting and batching.
#[derive(Debug, Clone, Copy)]
pub struct MaterialSortEntry {
    /// The sort key (see `Material::compute_sort_key`).
    pub sort_key: u64,
    /// Material id.
    pub material_id: u64,
    /// Instance id (0 for base materials).
    pub instance_id: u64,
    /// Shader handle hash (for grouping by shader).
    pub shader_hash: u32,
    /// Mesh handle / id (for grouping by mesh within the same shader+material).
    pub mesh_id: u64,
    /// Distance from camera (used for transparent sorting).
    pub camera_distance: f32,
}

/// Sort a list of material sort entries for optimal batching.
///
/// The sorting strategy is:
///   1. Group by pass (opaque before transparent).
///   2. Within opaque: sort by shader -> material -> mesh (front-to-back for
///      depth early-out).
///   3. Within transparent: sort back-to-front by camera distance.
pub fn sort_material_entries(entries: &mut [MaterialSortEntry]) {
    entries.sort_unstable_by(|a, b| {
        // Primary: sort key (encodes pass, blend, shader, features, material).
        let key_cmp = a.sort_key.cmp(&b.sort_key);
        if key_cmp != std::cmp::Ordering::Equal {
            return key_cmp;
        }
        // Secondary: for entries with the same key, sort by shader hash.
        let shader_cmp = a.shader_hash.cmp(&b.shader_hash);
        if shader_cmp != std::cmp::Ordering::Equal {
            return shader_cmp;
        }
        // Tertiary: group by mesh.
        let mesh_cmp = a.mesh_id.cmp(&b.mesh_id);
        if mesh_cmp != std::cmp::Ordering::Equal {
            return mesh_cmp;
        }
        // For transparent objects with identical keys, sort by distance
        // (back-to-front: larger distance first).
        b.camera_distance
            .partial_cmp(&a.camera_distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Given a sorted list of entries, compute batch boundaries. Each batch is a
/// contiguous range of entries that share the same shader + material, allowing
/// them to be drawn with a single pipeline bind and potentially instanced.
pub fn compute_batches(entries: &[MaterialSortEntry]) -> Vec<MaterialBatch> {
    if entries.is_empty() {
        return Vec::new();
    }

    let mut batches = Vec::new();
    let mut batch_start = 0usize;
    let mut current_shader = entries[0].shader_hash;
    let mut current_material = entries[0].material_id;

    for i in 1..entries.len() {
        let entry = &entries[i];
        if entry.shader_hash != current_shader || entry.material_id != current_material {
            batches.push(MaterialBatch {
                start_index: batch_start,
                count: i - batch_start,
                shader_hash: current_shader,
                material_id: current_material,
            });
            batch_start = i;
            current_shader = entry.shader_hash;
            current_material = entry.material_id;
        }
    }

    // Final batch.
    batches.push(MaterialBatch {
        start_index: batch_start,
        count: entries.len() - batch_start,
        shader_hash: current_shader,
        material_id: current_material,
    });

    batches
}

/// A contiguous batch of draw calls sharing the same shader + material state.
#[derive(Debug, Clone)]
pub struct MaterialBatch {
    /// Index into the sorted entry array where this batch starts.
    pub start_index: usize,
    /// Number of entries in this batch.
    pub count: usize,
    /// Shader hash identifying the pipeline.
    pub shader_hash: u32,
    /// Material id for parameter binding.
    pub material_id: u64,
}

// ---------------------------------------------------------------------------
// Default material presets
// ---------------------------------------------------------------------------

/// Factory for standard built-in material presets.
pub struct DefaultMaterials;

impl DefaultMaterials {
    /// Standard PBR lit material ("DefaultLit").
    pub fn default_lit() -> Material {
        let mut mat = Material::with_shader("DefaultLit", "shaders/pbr");
        mat.set_color("albedo_color", [1.0, 1.0, 1.0, 1.0]);
        mat.set_float("metallic", 0.0);
        mat.set_float("roughness", 0.5);
        mat.set_float("reflectance", 0.5);
        mat.set_float("normal_scale", 1.0);
        mat.set_vec3("emissive_color", [0.0, 0.0, 0.0]);
        mat.set_float("emissive_strength", 1.0);
        mat.set_float("ao_strength", 1.0);
        mat.set_passes(vec![MaterialPass::Opaque, MaterialPass::ShadowCaster, MaterialPass::DepthPrepass]);
        mat.enable_feature("PBR_LIT");
        mat
    }

    /// Standard unlit material ("DefaultUnlit").
    pub fn default_unlit() -> Material {
        let mut mat = Material::with_shader("DefaultUnlit", "shaders/unlit");
        mat.set_color("base_color", [1.0, 1.0, 1.0, 1.0]);
        mat.set_passes(vec![MaterialPass::Opaque]);
        mat.set_render_state(RenderState::default());
        mat.enable_feature("UNLIT");
        mat
    }

    /// Wireframe debug material ("Wireframe").
    pub fn wireframe() -> Material {
        let mut mat = Material::with_shader("Wireframe", "shaders/wireframe");
        mat.set_color("wire_color", [0.0, 1.0, 0.0, 1.0]);
        mat.set_float("wire_thickness", 1.0);
        mat.set_passes(vec![MaterialPass::Opaque]);
        mat.set_render_state(RenderState::wireframe_overlay());
        mat.enable_feature("WIREFRAME");
        mat
    }

    /// Error material -- bright pink so missing materials are visible.
    pub fn error() -> Material {
        let mut mat = Material::with_shader("Error", "shaders/error");
        mat.set_color("error_color", [1.0, 0.0, 1.0, 1.0]);
        mat.set_passes(vec![MaterialPass::Opaque]);
        mat.enable_feature("UNLIT");
        mat
    }

    /// Shadow caster material (depth-only, no fragment output).
    pub fn shadow_caster() -> Material {
        let mut mat = Material::with_shader("ShadowCaster", "shaders/depth_only");
        mat.set_passes(vec![MaterialPass::ShadowCaster]);
        mat.set_render_state(RenderState::shadow_caster());
        mat
    }

    /// Depth pre-pass material.
    pub fn depth_prepass() -> Material {
        let mut mat = Material::with_shader("DepthPrepass", "shaders/depth_only");
        mat.set_passes(vec![MaterialPass::DepthPrepass]);
        mat.set_render_state(RenderState::depth_prepass());
        mat
    }

    /// Standard PBR material with all texture slots wired up.
    pub fn standard_pbr_textured() -> Material {
        let mut mat = Self::default_lit();
        mat.name = "StandardPBRTextured".to_string();
        mat.id = alloc_id();

        // Wire up standard PBR texture slots.
        mat.add_texture_binding(
            TextureBinding::new("albedo_map", TextureId::INVALID)
                .with_group_binding(1, 0, 1),
        );
        mat.add_texture_binding(
            TextureBinding::new("normal_map", TextureId::INVALID)
                .with_group_binding(1, 2, 3),
        );
        mat.add_texture_binding(
            TextureBinding::new("metallic_roughness_map", TextureId::INVALID)
                .with_group_binding(1, 4, 5),
        );
        mat.add_texture_binding(
            TextureBinding::new("ao_map", TextureId::INVALID)
                .with_group_binding(1, 6, 7),
        );
        mat.add_texture_binding(
            TextureBinding::new("emissive_map", TextureId::INVALID)
                .with_group_binding(1, 8, 9),
        );
        mat.add_texture_binding(
            TextureBinding::new("height_map", TextureId::INVALID)
                .with_group_binding(1, 10, 11),
        );

        mat.enable_feature("HAS_ALBEDO_MAP");
        mat.enable_feature("HAS_NORMAL_MAP");
        mat.enable_feature("HAS_METALLIC_ROUGHNESS_MAP");
        mat.enable_feature("HAS_AO_MAP");
        mat.enable_feature("HAS_EMISSIVE_MAP");

        mat
    }

    /// G-buffer material for deferred rendering.
    pub fn gbuffer() -> Material {
        let mut mat = Self::default_lit();
        mat.name = "GBuffer".to_string();
        mat.id = alloc_id();
        mat.shader_name = "shaders/gbuffer".to_string();
        mat.set_passes(vec![MaterialPass::GBuffer]);
        mat.enable_feature("GBUFFER_OUTPUT");
        mat
    }

    /// Transparent glass material.
    pub fn glass() -> Material {
        let mut mat = Material::with_shader("Glass", "shaders/pbr");
        mat.set_color("albedo_color", [0.9, 0.95, 1.0, 0.3]);
        mat.set_float("metallic", 0.0);
        mat.set_float("roughness", 0.05);
        mat.set_float("reflectance", 0.9);
        mat.set_blend_mode(BlendMode::AlphaBlend);
        mat.set_depth_mode(DepthMode::ReadOnly);
        mat.set_passes(vec![MaterialPass::Transparent]);
        mat.enable_feature("PBR_LIT");
        mat.enable_feature("TRANSPARENT");
        mat
    }
}

// ---------------------------------------------------------------------------
// MaterialManager
// ---------------------------------------------------------------------------

/// Central manager for all materials in the engine. Handles creation, caching,
/// lookup, hot-reload, and serialisation.
pub struct MaterialManager {
    /// All registered materials, keyed by id.
    materials: HashMap<u64, Arc<Material>>,
    /// Name-to-id index for editor and script access.
    name_index: HashMap<String, u64>,
    /// Material instances keyed by instance id.
    instances: HashMap<u64, MaterialInstance>,
    /// Material presets (named templates for quick creation).
    presets: HashMap<String, Material>,
    /// Shader variant cache: (shader_name, feature_hash) -> shader handle.
    shader_variant_cache: HashMap<(String, u64), ShaderHandle>,
    /// Hot-reload watchers: shader_name -> list of material ids using it.
    shader_to_materials: HashMap<String, Vec<u64>>,
    /// Default material id.
    default_material_id: u64,
    /// Error material id.
    error_material_id: u64,
    /// Statistics.
    stats: MaterialManagerStats,
}

/// Statistics tracked by the material manager.
#[derive(Debug, Clone, Default)]
pub struct MaterialManagerStats {
    pub total_materials: usize,
    pub total_instances: usize,
    pub total_shader_variants: usize,
    pub total_presets: usize,
    pub uniform_bytes_uploaded: u64,
    pub materials_created_this_frame: u32,
    pub materials_destroyed_this_frame: u32,
}

impl MaterialManager {
    /// Create a new material manager with default materials pre-registered.
    pub fn new() -> Self {
        let default_lit = DefaultMaterials::default_lit();
        let error_mat = DefaultMaterials::error();

        let default_id = default_lit.id;
        let error_id = error_mat.id;

        let mut mgr = Self {
            materials: HashMap::new(),
            name_index: HashMap::new(),
            instances: HashMap::new(),
            presets: HashMap::new(),
            shader_variant_cache: HashMap::new(),
            shader_to_materials: HashMap::new(),
            default_material_id: default_id,
            error_material_id: error_id,
            stats: MaterialManagerStats::default(),
        };

        mgr.register_material(default_lit);
        mgr.register_material(error_mat);

        // Register all default presets.
        mgr.register_preset("DefaultLit", DefaultMaterials::default_lit());
        mgr.register_preset("DefaultUnlit", DefaultMaterials::default_unlit());
        mgr.register_preset("Wireframe", DefaultMaterials::wireframe());
        mgr.register_preset("Error", DefaultMaterials::error());
        mgr.register_preset("ShadowCaster", DefaultMaterials::shadow_caster());
        mgr.register_preset("DepthPrepass", DefaultMaterials::depth_prepass());
        mgr.register_preset("StandardPBRTextured", DefaultMaterials::standard_pbr_textured());
        mgr.register_preset("GBuffer", DefaultMaterials::gbuffer());
        mgr.register_preset("Glass", DefaultMaterials::glass());

        mgr
    }

    /// Register a material and return its id.
    pub fn register_material(&mut self, material: Material) -> u64 {
        let id = material.id;
        let name = material.name.clone();
        let shader_name = material.shader_name.clone();

        self.name_index.insert(name, id);
        self.materials.insert(id, Arc::new(material));

        // Track shader -> material mapping for hot-reload.
        if !shader_name.is_empty() {
            self.shader_to_materials
                .entry(shader_name)
                .or_default()
                .push(id);
        }

        self.stats.total_materials = self.materials.len();
        self.stats.materials_created_this_frame += 1;
        id
    }

    /// Create a material from a preset template.
    pub fn create_from_preset(&mut self, preset_name: &str) -> Option<u64> {
        let preset = self.presets.get(preset_name)?.clone();
        let mut mat = preset;
        mat.id = alloc_id(); // Assign a fresh id.
        Some(self.register_material(mat))
    }

    /// Create a material from a preset with a custom name.
    pub fn create_from_preset_named(&mut self, preset_name: &str, name: impl Into<String>) -> Option<u64> {
        let preset = self.presets.get(preset_name)?.clone();
        let mut mat = preset;
        mat.id = alloc_id();
        mat.name = name.into();
        Some(self.register_material(mat))
    }

    /// Remove a material by id.
    pub fn remove_material(&mut self, id: u64) -> Option<Arc<Material>> {
        if id == self.default_material_id || id == self.error_material_id {
            return None; // Cannot remove built-in materials.
        }
        if let Some(mat) = self.materials.remove(&id) {
            self.name_index.remove(&mat.name);
            // Remove from shader tracking.
            if let Some(mats) = self.shader_to_materials.get_mut(&mat.shader_name) {
                mats.retain(|&mid| mid != id);
            }
            // Remove all instances of this material.
            self.instances.retain(|_, inst| inst.base_material_id != id);
            self.stats.total_materials = self.materials.len();
            self.stats.total_instances = self.instances.len();
            self.stats.materials_destroyed_this_frame += 1;
            Some(mat)
        } else {
            None
        }
    }

    /// Look up a material by id.
    pub fn get(&self, id: u64) -> Option<&Arc<Material>> {
        self.materials.get(&id)
    }

    /// Look up a material by name.
    pub fn get_by_name(&self, name: &str) -> Option<&Arc<Material>> {
        self.name_index.get(name).and_then(|id| self.materials.get(id))
    }

    /// Get the id of a material by name.
    pub fn id_by_name(&self, name: &str) -> Option<u64> {
        self.name_index.get(name).copied()
    }

    /// Get the default material.
    pub fn default_material(&self) -> &Arc<Material> {
        self.materials.get(&self.default_material_id).unwrap()
    }

    /// Get the error material.
    pub fn error_material(&self) -> &Arc<Material> {
        self.materials.get(&self.error_material_id).unwrap()
    }

    // -- Instances -----------------------------------------------------------

    /// Create a material instance from a base material id.
    pub fn create_instance(&mut self, base_id: u64) -> Option<u64> {
        let base = self.materials.get(&base_id)?;
        let instance = MaterialInstance::new(base);
        let inst_id = instance.instance_id;
        self.instances.insert(inst_id, instance);
        self.stats.total_instances = self.instances.len();
        Some(inst_id)
    }

    /// Get a mutable reference to an instance.
    pub fn get_instance_mut(&mut self, instance_id: u64) -> Option<&mut MaterialInstance> {
        self.instances.get_mut(&instance_id)
    }

    /// Get an immutable reference to an instance.
    pub fn get_instance(&self, instance_id: u64) -> Option<&MaterialInstance> {
        self.instances.get(&instance_id)
    }

    /// Remove an instance.
    pub fn remove_instance(&mut self, instance_id: u64) -> Option<MaterialInstance> {
        let inst = self.instances.remove(&instance_id);
        self.stats.total_instances = self.instances.len();
        inst
    }

    // -- Presets -------------------------------------------------------------

    /// Register a preset template.
    pub fn register_preset(&mut self, name: impl Into<String>, material: Material) {
        self.presets.insert(name.into(), material);
        self.stats.total_presets = self.presets.len();
    }

    /// Get a preset by name.
    pub fn get_preset(&self, name: &str) -> Option<&Material> {
        self.presets.get(name)
    }

    /// List all preset names.
    pub fn preset_names(&self) -> Vec<&str> {
        self.presets.keys().map(|s| s.as_str()).collect()
    }

    // -- Shader permutations -------------------------------------------------

    /// Look up or insert a shader variant in the cache.
    pub fn get_or_insert_shader_variant(
        &mut self,
        shader_name: &str,
        features: &ShaderFeatureFlags,
        create_fn: impl FnOnce() -> Option<ShaderHandle>,
    ) -> Option<ShaderHandle> {
        let key = (shader_name.to_string(), features.hash_u64());
        if let Some(&handle) = self.shader_variant_cache.get(&key) {
            return Some(handle);
        }
        let handle = create_fn()?;
        self.shader_variant_cache.insert(key, handle);
        self.stats.total_shader_variants = self.shader_variant_cache.len();
        Some(handle)
    }

    /// Invalidate all cached shader variants for a given shader name.
    /// Called during hot-reload when a shader source file changes.
    pub fn invalidate_shader_variants(&mut self, shader_name: &str) {
        self.shader_variant_cache
            .retain(|(name, _), _| name != shader_name);
        self.stats.total_shader_variants = self.shader_variant_cache.len();
    }

    // -- Hot-reload ----------------------------------------------------------

    /// Notify the manager that a shader has been modified. This invalidates
    /// cached shader variants and marks all materials using that shader as
    /// dirty so they will be re-bound next frame.
    pub fn on_shader_modified(&mut self, shader_name: &str) {
        self.invalidate_shader_variants(shader_name);

        // Mark all materials using this shader for re-upload.
        if let Some(mat_ids) = self.shader_to_materials.get(shader_name) {
            let ids: Vec<u64> = mat_ids.clone();
            for id in ids {
                if let Some(mat) = self.materials.get_mut(&id) {
                    // We need to get a mutable reference; since we store Arc,
                    // we clone-on-write.
                    let mut new_mat = (**mat).clone();
                    new_mat.shader_handle = None; // Force re-bind.
                    new_mat.bump_version();
                    *mat = Arc::new(new_mat);
                }
            }
        }
    }

    // -- Sorting & batching --------------------------------------------------

    /// Collect all visible materials and compute sort entries for draw-call
    /// batching. The returned entries are pre-sorted.
    pub fn build_sort_entries(
        &self,
        draw_list: &[(u64, u64, f32)], // (material_id, mesh_id, camera_distance)
    ) -> Vec<MaterialSortEntry> {
        let mut entries: Vec<MaterialSortEntry> = draw_list
            .iter()
            .filter_map(|&(mat_id, mesh_id, dist)| {
                let mat = self.materials.get(&mat_id)?;
                if !mat.visible {
                    return None;
                }
                let shader_hash = {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    mat.shader_name.hash(&mut h);
                    (h.finish() & 0xFFFF_FFFF) as u32
                };
                Some(MaterialSortEntry {
                    sort_key: mat.sort_key,
                    material_id: mat_id,
                    instance_id: 0,
                    shader_hash,
                    mesh_id,
                    camera_distance: dist,
                })
            })
            .collect();

        sort_material_entries(&mut entries);
        entries
    }

    // -- Statistics ----------------------------------------------------------

    /// Get a snapshot of manager statistics.
    pub fn stats(&self) -> &MaterialManagerStats {
        &self.stats
    }

    /// Reset per-frame statistics counters.
    pub fn reset_frame_stats(&mut self) {
        self.stats.materials_created_this_frame = 0;
        self.stats.materials_destroyed_this_frame = 0;
    }

    /// Total number of registered materials.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Total number of active instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Iterate over all materials.
    pub fn iter_materials(&self) -> impl Iterator<Item = &Arc<Material>> {
        self.materials.values()
    }
}

impl Default for MaterialManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

/// Serialize a `Material` to a JSON string.
///
/// This produces a human-readable JSON representation suitable for asset files
/// and editor save/load. GPU handles are not serialised.
pub fn material_to_json(material: &Material) -> String {
    let mut json = String::with_capacity(2048);
    json.push_str("{\n");
    json.push_str(&format!("  \"name\": \"{}\",\n", material.name));
    json.push_str(&format!("  \"shader\": \"{}\",\n", material.shader_name));

    // Uniforms.
    json.push_str("  \"uniforms\": {\n");
    let mut sorted_keys: Vec<&String> = material.uniform_data.keys().collect();
    sorted_keys.sort();
    for (i, key) in sorted_keys.iter().enumerate() {
        if let Some(val) = material.uniform_data.get(*key) {
            let val_str = match val {
                UniformValue::Float(v) => format!("{{ \"type\": \"float\", \"value\": {} }}", v),
                UniformValue::Vec2(v) => format!("{{ \"type\": \"vec2\", \"value\": [{}, {}] }}", v[0], v[1]),
                UniformValue::Vec3(v) => format!("{{ \"type\": \"vec3\", \"value\": [{}, {}, {}] }}", v[0], v[1], v[2]),
                UniformValue::Vec4(v) => format!("{{ \"type\": \"vec4\", \"value\": [{}, {}, {}, {}] }}", v[0], v[1], v[2], v[3]),
                UniformValue::Mat4(v) => {
                    let vals: Vec<String> = v.iter().map(|f| format!("{}", f)).collect();
                    format!("{{ \"type\": \"mat4\", \"value\": [{}] }}", vals.join(", "))
                }
                UniformValue::Int(v) => format!("{{ \"type\": \"int\", \"value\": {} }}", v),
                UniformValue::Bool(v) => format!("{{ \"type\": \"bool\", \"value\": {} }}", v),
                UniformValue::Color(v) => format!("{{ \"type\": \"color\", \"value\": [{}, {}, {}, {}] }}", v[0], v[1], v[2], v[3]),
                UniformValue::Texture(id) => format!("{{ \"type\": \"texture\", \"value\": {} }}", id.0),
            };
            let comma = if i + 1 < sorted_keys.len() { "," } else { "" };
            json.push_str(&format!("    \"{}\": {}{}\n", key, val_str, comma));
        }
    }
    json.push_str("  },\n");

    // Texture bindings.
    json.push_str("  \"texture_bindings\": [\n");
    for (i, binding) in material.texture_bindings.iter().enumerate() {
        let comma = if i + 1 < material.texture_bindings.len() { "," } else { "" };
        json.push_str(&format!(
            "    {{ \"name\": \"{}\", \"texture_id\": {}, \"group\": {}, \"binding_texture\": {}, \"binding_sampler\": {}, \"uv_channel\": {}, \"tiling\": [{}, {}], \"offset\": [{}, {}] }}{}\n",
            binding.name, binding.texture.0, binding.group,
            binding.binding_texture, binding.binding_sampler,
            binding.uv_channel,
            binding.tiling[0], binding.tiling[1],
            binding.offset[0], binding.offset[1],
            comma,
        ));
    }
    json.push_str("  ],\n");

    // Render state.
    let blend_str = match material.render_state.blend {
        BlendMode::Opaque => "opaque",
        BlendMode::AlphaBlend => "alpha_blend",
        BlendMode::PremultipliedAlpha => "premultiplied_alpha",
        BlendMode::Additive => "additive",
        BlendMode::Multiply => "multiply",
        BlendMode::Custom { .. } => "custom",
    };
    let depth_str = match material.render_state.depth_mode {
        DepthMode::ReadWrite => "read_write",
        DepthMode::ReadOnly => "read_only",
        DepthMode::Disabled => "disabled",
        DepthMode::WriteOnly => "write_only",
    };
    let cull_str = match material.render_state.cull {
        CullFace::None => "none",
        CullFace::Back => "back",
        CullFace::Front => "front",
    };
    json.push_str("  \"render_state\": {\n");
    json.push_str(&format!("    \"blend\": \"{}\",\n", blend_str));
    json.push_str(&format!("    \"depth\": \"{}\",\n", depth_str));
    json.push_str(&format!("    \"cull\": \"{}\"\n", cull_str));
    json.push_str("  },\n");

    // Passes.
    let pass_strs: Vec<&str> = material.passes.iter().map(|p| match p {
        MaterialPass::Opaque => "opaque",
        MaterialPass::Transparent => "transparent",
        MaterialPass::ShadowCaster => "shadow_caster",
        MaterialPass::DepthPrepass => "depth_prepass",
        MaterialPass::GBuffer => "gbuffer",
        MaterialPass::Velocity => "velocity",
        MaterialPass::Custom(_) => "custom",
    }).collect();
    json.push_str(&format!("  \"passes\": [{}],\n",
        pass_strs.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", ")));

    // Features.
    let feature_strs: Vec<String> = material.features.flags().iter()
        .map(|f| format!("\"{}\"", f))
        .collect();
    json.push_str(&format!("  \"features\": [{}],\n", feature_strs.join(", ")));

    // Tags.
    let tag_strs: Vec<String> = material.tags.iter()
        .map(|t| format!("\"{}\"", t))
        .collect();
    json.push_str(&format!("  \"tags\": [{}],\n", tag_strs.join(", ")));

    json.push_str(&format!("  \"priority\": {},\n", material.priority));
    json.push_str(&format!("  \"visible\": {}\n", material.visible));
    json.push_str("}");
    json
}

/// Parse a JSON string into a `Material`.
///
/// This is a simple hand-written parser for the format produced by
/// `material_to_json`. It does not depend on serde to keep dependencies
/// minimal.
pub fn material_from_json(json: &str) -> Result<Material, String> {
    let mut mat = Material::new("Unnamed");

    // Extract top-level string fields.
    if let Some(name) = extract_json_string(json, "name") {
        mat.name = name;
    }
    if let Some(shader) = extract_json_string(json, "shader") {
        mat.shader_name = shader;
    }

    // Extract uniforms block.
    if let Some(uniforms_block) = extract_json_object(json, "uniforms") {
        parse_uniforms_block(&uniforms_block, &mut mat);
    }

    // Extract render state.
    if let Some(rs_block) = extract_json_object(json, "render_state") {
        if let Some(blend) = extract_json_string(&rs_block, "blend") {
            mat.render_state.blend = match blend.as_str() {
                "opaque" => BlendMode::Opaque,
                "alpha_blend" => BlendMode::AlphaBlend,
                "premultiplied_alpha" => BlendMode::PremultipliedAlpha,
                "additive" => BlendMode::Additive,
                "multiply" => BlendMode::Multiply,
                _ => BlendMode::Opaque,
            };
        }
        if let Some(depth) = extract_json_string(&rs_block, "depth") {
            mat.render_state.depth_mode = match depth.as_str() {
                "read_write" => DepthMode::ReadWrite,
                "read_only" => DepthMode::ReadOnly,
                "disabled" => DepthMode::Disabled,
                "write_only" => DepthMode::WriteOnly,
                _ => DepthMode::ReadWrite,
            };
        }
        if let Some(cull) = extract_json_string(&rs_block, "cull") {
            mat.render_state.cull = match cull.as_str() {
                "none" => CullFace::None,
                "back" => CullFace::Back,
                "front" => CullFace::Front,
                _ => CullFace::Back,
            };
        }
    }

    // Extract passes.
    if let Some(passes_block) = extract_json_array(json, "passes") {
        mat.passes.clear();
        for pass_str in extract_string_array_elements(&passes_block) {
            match pass_str.as_str() {
                "opaque" => mat.passes.push(MaterialPass::Opaque),
                "transparent" => mat.passes.push(MaterialPass::Transparent),
                "shadow_caster" => mat.passes.push(MaterialPass::ShadowCaster),
                "depth_prepass" => mat.passes.push(MaterialPass::DepthPrepass),
                "gbuffer" => mat.passes.push(MaterialPass::GBuffer),
                "velocity" => mat.passes.push(MaterialPass::Velocity),
                _ => {}
            }
        }
    }

    // Extract features.
    if let Some(features_block) = extract_json_array(json, "features") {
        for feature in extract_string_array_elements(&features_block) {
            mat.features.enable(feature);
        }
    }

    // Extract tags.
    if let Some(tags_block) = extract_json_array(json, "tags") {
        mat.tags = extract_string_array_elements(&tags_block);
    }

    // Extract priority.
    if let Some(priority) = extract_json_number(json, "priority") {
        mat.priority = priority as i32;
    }

    // Extract visible.
    if let Some(visible) = extract_json_bool(json, "visible") {
        mat.visible = visible;
    }

    Ok(mat)
}

// ---------------------------------------------------------------------------
// JSON helper functions (minimal, no serde dependency)
// ---------------------------------------------------------------------------

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let idx = json.find(&pattern)?;
    let after = &json[idx + pattern.len()..];
    let after = after.trim_start();
    if !after.starts_with('"') {
        return None;
    }
    let after = &after[1..];
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

fn extract_json_object(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let idx = json.find(&pattern)?;
    let after = &json[idx + pattern.len()..];
    let after = after.trim_start();
    if !after.starts_with('{') {
        return None;
    }
    let mut depth = 0i32;
    let mut end = 0;
    for (i, ch) in after.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    if end > 0 {
        Some(after[..end].to_string())
    } else {
        None
    }
}

fn extract_json_array(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let idx = json.find(&pattern)?;
    let after = &json[idx + pattern.len()..];
    let after = after.trim_start();
    if !after.starts_with('[') {
        return None;
    }
    let mut depth = 0i32;
    let mut end = 0;
    for (i, ch) in after.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    end = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    if end > 0 {
        Some(after[..end].to_string())
    } else {
        None
    }
}

fn extract_string_array_elements(array_str: &str) -> Vec<String> {
    let mut results = Vec::new();
    let inner = array_str.trim();
    let inner = if inner.starts_with('[') && inner.ends_with(']') {
        &inner[1..inner.len() - 1]
    } else {
        inner
    };
    for part in inner.split(',') {
        let part = part.trim();
        if part.starts_with('"') && part.ends_with('"') {
            results.push(part[1..part.len() - 1].to_string());
        }
    }
    results
}

fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", key);
    let idx = json.find(&pattern)?;
    let after = &json[idx + pattern.len()..];
    let after = after.trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != '+')?;
    after[..end].parse::<f64>().ok()
}

fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let pattern = format!("\"{}\":", key);
    let idx = json.find(&pattern)?;
    let after = &json[idx + pattern.len()..];
    let after = after.trim_start();
    if after.starts_with("true") {
        Some(true)
    } else if after.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn parse_uniforms_block(block: &str, mat: &mut Material) {
    // Very simple parser: find "key": { "type": "...", "value": ... } patterns.
    let inner = block.trim();
    let inner = if inner.starts_with('{') && inner.ends_with('}') {
        &inner[1..inner.len() - 1]
    } else {
        inner
    };

    // Split by top-level commas outside nested braces.
    let entries = split_top_level(inner, ',');
    for entry in entries {
        let entry = entry.trim();
        // Find the key.
        if let Some(colon_pos) = entry.find(':') {
            let key_part = entry[..colon_pos].trim();
            let key = if key_part.starts_with('"') && key_part.ends_with('"') {
                key_part[1..key_part.len() - 1].to_string()
            } else {
                continue;
            };

            let val_part = entry[colon_pos + 1..].trim();
            if let Some(type_str) = extract_json_string(val_part, "type") {
                match type_str.as_str() {
                    "float" => {
                        if let Some(v) = extract_json_number(val_part, "value") {
                            mat.set_float(key, v as f32);
                        }
                    }
                    "int" => {
                        if let Some(v) = extract_json_number(val_part, "value") {
                            mat.set_int(key, v as i32);
                        }
                    }
                    "bool" => {
                        if let Some(v) = extract_json_bool(val_part, "value") {
                            mat.set_bool(key, v);
                        }
                    }
                    "vec2" => {
                        if let Some(arr) = extract_json_array(val_part, "value") {
                            let nums = parse_float_array(&arr);
                            if nums.len() >= 2 {
                                mat.set_vec2(key, [nums[0], nums[1]]);
                            }
                        }
                    }
                    "vec3" => {
                        if let Some(arr) = extract_json_array(val_part, "value") {
                            let nums = parse_float_array(&arr);
                            if nums.len() >= 3 {
                                mat.set_vec3(key, [nums[0], nums[1], nums[2]]);
                            }
                        }
                    }
                    "vec4" => {
                        if let Some(arr) = extract_json_array(val_part, "value") {
                            let nums = parse_float_array(&arr);
                            if nums.len() >= 4 {
                                mat.set_vec4(key, [nums[0], nums[1], nums[2], nums[3]]);
                            }
                        }
                    }
                    "color" => {
                        if let Some(arr) = extract_json_array(val_part, "value") {
                            let nums = parse_float_array(&arr);
                            if nums.len() >= 4 {
                                mat.set_color(key, [nums[0], nums[1], nums[2], nums[3]]);
                            }
                        }
                    }
                    "mat4" => {
                        if let Some(arr) = extract_json_array(val_part, "value") {
                            let nums = parse_float_array(&arr);
                            if nums.len() >= 16 {
                                let mut m = [0.0f32; 16];
                                m.copy_from_slice(&nums[..16]);
                                mat.set_mat4(key, m);
                            }
                        }
                    }
                    "texture" => {
                        if let Some(v) = extract_json_number(val_part, "value") {
                            mat.set_uniform(key, UniformValue::Texture(TextureId(v as u64)));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

fn split_top_level(s: &str, delimiter: char) -> Vec<String> {
    let mut results = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();

    for ch in s.chars() {
        match ch {
            '{' | '[' => {
                depth += 1;
                current.push(ch);
            }
            '}' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            c if c == delimiter && depth == 0 => {
                results.push(std::mem::take(&mut current));
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.trim().is_empty() {
        results.push(current);
    }
    results
}

fn parse_float_array(array_str: &str) -> Vec<f32> {
    let inner = array_str.trim();
    let inner = if inner.starts_with('[') && inner.ends_with(']') {
        &inner[1..inner.len() - 1]
    } else {
        inner
    };
    inner
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_value_byte_sizes() {
        assert_eq!(UniformValue::Float(1.0).byte_size(), 4);
        assert_eq!(UniformValue::Vec2([0.0; 2]).byte_size(), 8);
        assert_eq!(UniformValue::Vec3([0.0; 3]).byte_size(), 12);
        assert_eq!(UniformValue::Vec4([0.0; 4]).byte_size(), 16);
        assert_eq!(UniformValue::Mat4([0.0; 16]).byte_size(), 64);
        assert_eq!(UniformValue::Int(0).byte_size(), 4);
        assert_eq!(UniformValue::Bool(true).byte_size(), 4);
        assert_eq!(UniformValue::Texture(TextureId(0)).byte_size(), 0);
    }

    #[test]
    fn uniform_write_and_read() {
        let mut buf = vec![0u8; 16];
        UniformValue::Float(3.14).write_bytes(&mut buf, 0);
        let v = UniformValue::read_float(&buf, 0);
        assert!((v - 3.14).abs() < 1e-5);
    }

    #[test]
    fn material_sort_key_opaque_before_transparent() {
        let mut opaque = Material::with_shader("Opaque", "shaders/pbr");
        opaque.set_blend_mode(BlendMode::Opaque);
        let opaque_key = opaque.compute_sort_key();

        let mut transparent = Material::with_shader("Trans", "shaders/pbr");
        transparent.set_blend_mode(BlendMode::AlphaBlend);
        let trans_key = transparent.compute_sort_key();

        assert!(opaque_key < trans_key);
    }

    #[test]
    fn material_sort_entries_batching() {
        let mut entries = vec![
            MaterialSortEntry { sort_key: 100, material_id: 1, instance_id: 0, shader_hash: 10, mesh_id: 1, camera_distance: 5.0 },
            MaterialSortEntry { sort_key: 50, material_id: 2, instance_id: 0, shader_hash: 10, mesh_id: 2, camera_distance: 3.0 },
            MaterialSortEntry { sort_key: 100, material_id: 1, instance_id: 0, shader_hash: 10, mesh_id: 3, camera_distance: 7.0 },
            MaterialSortEntry { sort_key: 50, material_id: 2, instance_id: 0, shader_hash: 20, mesh_id: 4, camera_distance: 1.0 },
        ];
        sort_material_entries(&mut entries);

        // Entries with sort_key 50 should come first.
        assert_eq!(entries[0].sort_key, 50);
        assert_eq!(entries[1].sort_key, 50);

        let batches = compute_batches(&entries);
        assert!(batches.len() >= 2);
    }

    #[test]
    fn material_manager_basics() {
        let mut mgr = MaterialManager::new();
        assert!(mgr.material_count() >= 2); // default + error

        let mat = Material::with_shader("TestMat", "shaders/test");
        let id = mgr.register_material(mat);
        assert!(mgr.get(id).is_some());
        assert!(mgr.get_by_name("TestMat").is_some());

        mgr.remove_material(id);
        assert!(mgr.get(id).is_none());
    }

    #[test]
    fn material_instance_overrides() {
        let mut base = Material::with_shader("Base", "shaders/pbr");
        base.set_float("roughness", 0.5);
        base.set_color("albedo", [1.0, 1.0, 1.0, 1.0]);

        let mut inst = MaterialInstance::new(&base);
        inst.set_override("roughness", UniformValue::Float(0.8));

        assert_eq!(
            inst.get_effective_value("roughness", &base),
            Some(&UniformValue::Float(0.8))
        );
        assert_eq!(
            inst.get_effective_value("albedo", &base),
            Some(&UniformValue::Color([1.0, 1.0, 1.0, 1.0]))
        );
    }

    #[test]
    fn material_json_round_trip() {
        let mut mat = Material::with_shader("TestMat", "shaders/pbr");
        mat.set_float("roughness", 0.5);
        mat.set_vec3("emissive", [1.0, 0.5, 0.0]);
        mat.set_color("albedo", [0.8, 0.2, 0.1, 1.0]);
        mat.set_bool("double_sided", true);
        mat.set_int("layer", 3);
        mat.enable_feature("PBR_LIT");
        mat.tags = vec!["terrain".to_string()];

        let json = material_to_json(&mat);
        let parsed = material_from_json(&json).unwrap();

        assert_eq!(parsed.name, "TestMat");
        assert_eq!(parsed.shader_name, "shaders/pbr");
        assert!(parsed.features.is_enabled("PBR_LIT"));
    }

    #[test]
    fn uniform_buffer_packing() {
        let mut mat = Material::new("PackTest");
        mat.set_float("a_float", 1.0);
        mat.set_vec4("b_vec4", [2.0, 3.0, 4.0, 5.0]);

        let bytes = mat.pack_uniforms();
        assert!(!bytes.is_empty());
        // Buffer should be 16-byte aligned.
        assert_eq!(bytes.len() % 16, 0);
    }

    #[test]
    fn default_materials_have_correct_passes() {
        let lit = DefaultMaterials::default_lit();
        assert!(lit.has_pass(MaterialPass::Opaque));
        assert!(lit.has_pass(MaterialPass::ShadowCaster));

        let glass = DefaultMaterials::glass();
        assert!(glass.has_pass(MaterialPass::Transparent));

        let shadow = DefaultMaterials::shadow_caster();
        assert!(shadow.has_pass(MaterialPass::ShadowCaster));
    }

    #[test]
    fn create_from_preset() {
        let mut mgr = MaterialManager::new();
        let id = mgr.create_from_preset("Glass").unwrap();
        let mat = mgr.get(id).unwrap();
        assert!(mat.has_pass(MaterialPass::Transparent));
    }

    #[test]
    fn shader_feature_flags() {
        let mut flags = ShaderFeatureFlags::new();
        flags.enable("HAS_NORMAL_MAP");
        flags.enable("USE_SKINNING");
        assert!(flags.is_enabled("HAS_NORMAL_MAP"));
        assert!(!flags.is_enabled("HAS_AO_MAP"));

        flags.disable("HAS_NORMAL_MAP");
        assert!(!flags.is_enabled("HAS_NORMAL_MAP"));

        // Hash should be stable.
        let mut flags2 = ShaderFeatureFlags::new();
        flags2.enable("USE_SKINNING");
        assert_eq!(flags.hash_u64(), flags2.hash_u64());
    }

    #[test]
    fn render_state_presets() {
        let shadow = RenderState::shadow_caster();
        assert!(matches!(shadow.cull, CullFace::Front));
        assert!(shadow.depth_bias > 0.0);

        let prepass = RenderState::depth_prepass();
        assert_eq!(prepass.color_write_mask, 0b0000);

        let transparent = RenderState::transparent();
        assert!(matches!(transparent.blend, BlendMode::AlphaBlend));
        assert!(matches!(transparent.depth_mode, DepthMode::ReadOnly));
    }

    #[test]
    fn material_pass_ordering() {
        assert!(MaterialPass::DepthPrepass.order() < MaterialPass::Opaque.order());
        assert!(MaterialPass::Opaque.order() < MaterialPass::Transparent.order());
        assert!(MaterialPass::ShadowCaster.order() < MaterialPass::Opaque.order());
    }

    #[test]
    fn hot_reload_invalidates_variants() {
        let mut mgr = MaterialManager::new();
        let mat = Material::with_shader("ReloadTest", "shaders/pbr");
        let id = mgr.register_material(mat);

        // Simulate caching a shader variant.
        let features = ShaderFeatureFlags::new();
        let handle = ShaderHandle(42);
        mgr.shader_variant_cache.insert(
            ("shaders/pbr".to_string(), features.hash_u64()),
            handle,
        );
        assert_eq!(mgr.shader_variant_cache.len(), 1);

        // Trigger hot-reload.
        mgr.on_shader_modified("shaders/pbr");
        assert_eq!(mgr.shader_variant_cache.len(), 0);

        // The material should have been version-bumped.
        let reloaded = mgr.get(id).unwrap();
        assert!(reloaded.shader_handle.is_none());
    }
}
