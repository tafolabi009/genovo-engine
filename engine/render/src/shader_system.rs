// engine/render/src/shader_system.rs
//
// Shader management system: shader program objects, uniform binding, texture
// slot management, global shader parameters, shader warm-up, shader
// complexity metrics, and shader variant tracking.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderVariantKey(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UniformLocation(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSlot(pub u32);

/// Shader stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Geometry,
    TessControl,
    TessEval,
    Compute,
}

// ---------------------------------------------------------------------------
// Uniform types
// ---------------------------------------------------------------------------

/// Shader uniform value.
#[derive(Debug, Clone)]
pub enum UniformValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Int(i32),
    IVec2([i32; 2]),
    IVec3([i32; 3]),
    IVec4([i32; 4]),
    UInt(u32),
    Mat3([f32; 9]),
    Mat4([f32; 16]),
    FloatArray(Vec<f32>),
    Vec4Array(Vec<[f32; 4]>),
    Mat4Array(Vec<[f32; 16]>),
    Bool(bool),
}

impl UniformValue {
    /// Size in bytes for this uniform value.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Float(_) => 4,
            Self::Vec2(_) => 8,
            Self::Vec3(_) => 12,
            Self::Vec4(_) => 16,
            Self::Int(_) | Self::UInt(_) | Self::Bool(_) => 4,
            Self::IVec2(_) => 8,
            Self::IVec3(_) => 12,
            Self::IVec4(_) => 16,
            Self::Mat3(_) => 36,
            Self::Mat4(_) => 64,
            Self::FloatArray(v) => v.len() * 4,
            Self::Vec4Array(v) => v.len() * 16,
            Self::Mat4Array(v) => v.len() * 64,
        }
    }

    /// Get the uniform type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Float(_) => "float",
            Self::Vec2(_) => "vec2",
            Self::Vec3(_) => "vec3",
            Self::Vec4(_) => "vec4",
            Self::Int(_) => "int",
            Self::IVec2(_) => "ivec2",
            Self::IVec3(_) => "ivec3",
            Self::IVec4(_) => "ivec4",
            Self::UInt(_) => "uint",
            Self::Mat3(_) => "mat3",
            Self::Mat4(_) => "mat4",
            Self::FloatArray(_) => "float[]",
            Self::Vec4Array(_) => "vec4[]",
            Self::Mat4Array(_) => "mat4[]",
            Self::Bool(_) => "bool",
        }
    }
}

// ---------------------------------------------------------------------------
// Shader define
// ---------------------------------------------------------------------------

/// A preprocessor define for shader compilation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderDefine {
    pub name: String,
    pub value: Option<String>,
}

impl ShaderDefine {
    pub fn flag(name: &str) -> Self {
        Self { name: name.to_string(), value: None }
    }

    pub fn valued(name: &str, value: &str) -> Self {
        Self { name: name.to_string(), value: Some(value.to_string()) }
    }

    pub fn to_directive(&self) -> String {
        match &self.value {
            Some(val) => format!("#define {} {}", self.name, val),
            None => format!("#define {}", self.name),
        }
    }
}

// ---------------------------------------------------------------------------
// Shader source
// ---------------------------------------------------------------------------

/// Source code for a single shader stage.
#[derive(Debug, Clone)]
pub struct ShaderStageSource {
    pub stage: ShaderStage,
    pub source: String,
    pub entry_point: String,
    pub includes: Vec<String>,
}

/// A complete shader source with all stages.
#[derive(Debug, Clone)]
pub struct ShaderSource {
    pub name: String,
    pub stages: Vec<ShaderStageSource>,
    pub defines: Vec<ShaderDefine>,
    pub version: u32,
}

impl ShaderSource {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            stages: Vec::new(),
            defines: Vec::new(),
            version: 1,
        }
    }

    pub fn add_stage(&mut self, stage: ShaderStage, source: &str, entry_point: &str) {
        self.stages.push(ShaderStageSource {
            stage,
            source: source.to_string(),
            entry_point: entry_point.to_string(),
            includes: Vec::new(),
        });
    }

    pub fn add_define(&mut self, define: ShaderDefine) {
        self.defines.push(define);
    }

    /// Generate a hash key for this source+defines combination.
    pub fn variant_key(&self) -> ShaderVariantKey {
        let mut hasher = FnvHasher::new();
        hasher.write_str(&self.name);
        for d in &self.defines {
            hasher.write_str(&d.name);
            if let Some(ref v) = d.value {
                hasher.write_str(v);
            }
        }
        ShaderVariantKey(hasher.finish())
    }

    /// Preprocess the source: inject defines and resolve includes.
    pub fn preprocess(&self, stage: ShaderStage, include_resolver: &dyn Fn(&str) -> Option<String>) -> String {
        let stage_src = match self.stages.iter().find(|s| s.stage == stage) {
            Some(s) => &s.source,
            None => return String::new(),
        };

        let mut output = String::with_capacity(stage_src.len() + 1024);

        // Add defines
        for d in &self.defines {
            output.push_str(&d.to_directive());
            output.push('\n');
        }
        output.push('\n');

        // Resolve includes
        for line in stage_src.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("#include") {
                let path = trimmed
                    .strip_prefix("#include")
                    .unwrap_or("")
                    .trim()
                    .trim_matches('"')
                    .trim_matches('<')
                    .trim_matches('>');

                if let Some(included) = include_resolver(path) {
                    output.push_str(&format!("// begin include: {}\n", path));
                    output.push_str(&included);
                    output.push_str(&format!("\n// end include: {}\n", path));
                } else {
                    output.push_str(&format!("// ERROR: include not found: {}\n", path));
                }
            } else {
                output.push_str(line);
                output.push('\n');
            }
        }

        output
    }
}

/// Simple FNV-1a hasher for shader variant keys.
struct FnvHasher { hash: u64 }
impl FnvHasher {
    fn new() -> Self { Self { hash: 0xcbf29ce484222325 } }
    fn write_str(&mut self, s: &str) {
        for b in s.bytes() {
            self.hash ^= b as u64;
            self.hash = self.hash.wrapping_mul(0x100000001b3);
        }
    }
    fn finish(&self) -> u64 { self.hash }
}

// ---------------------------------------------------------------------------
// Shader program
// ---------------------------------------------------------------------------

/// A compiled shader program with uniform metadata.
#[derive(Debug, Clone)]
pub struct ShaderProgram {
    pub id: ShaderId,
    pub name: String,
    pub variant_key: ShaderVariantKey,
    pub uniforms: HashMap<String, UniformInfo>,
    pub texture_slots: HashMap<String, TextureSlotInfo>,
    pub uniform_blocks: Vec<UniformBlockInfo>,
    pub stages: Vec<ShaderStage>,
    pub is_compiled: bool,
    pub compile_time_ms: f32,
    pub complexity: ShaderComplexity,
}

#[derive(Debug, Clone)]
pub struct UniformInfo {
    pub location: UniformLocation,
    pub uniform_type: String,
    pub array_size: u32,
    pub dirty: bool,
    pub value: Option<UniformValue>,
}

#[derive(Debug, Clone)]
pub struct TextureSlotInfo {
    pub slot: TextureSlot,
    pub sampler_type: String,
    pub bound_texture: Option<u64>, // texture handle
}

#[derive(Debug, Clone)]
pub struct UniformBlockInfo {
    pub name: String,
    pub binding: u32,
    pub size_bytes: u32,
    pub members: Vec<UniformBlockMember>,
}

#[derive(Debug, Clone)]
pub struct UniformBlockMember {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub member_type: String,
}

impl ShaderProgram {
    pub fn new(id: ShaderId, name: &str, variant_key: ShaderVariantKey) -> Self {
        Self {
            id,
            name: name.to_string(),
            variant_key,
            uniforms: HashMap::new(),
            texture_slots: HashMap::new(),
            uniform_blocks: Vec::new(),
            stages: Vec::new(),
            is_compiled: false,
            compile_time_ms: 0.0,
            complexity: ShaderComplexity::default(),
        }
    }

    /// Set a uniform value.
    pub fn set_uniform(&mut self, name: &str, value: UniformValue) -> bool {
        if let Some(info) = self.uniforms.get_mut(name) {
            info.value = Some(value);
            info.dirty = true;
            true
        } else {
            false
        }
    }

    /// Bind a texture to a named slot.
    pub fn bind_texture(&mut self, name: &str, texture_handle: u64) -> bool {
        if let Some(slot) = self.texture_slots.get_mut(name) {
            slot.bound_texture = Some(texture_handle);
            true
        } else {
            false
        }
    }

    /// Get all dirty uniforms and mark them clean.
    pub fn flush_dirty_uniforms(&mut self) -> Vec<(UniformLocation, UniformValue)> {
        let mut dirty = Vec::new();
        for info in self.uniforms.values_mut() {
            if info.dirty {
                if let Some(ref val) = info.value {
                    dirty.push((info.location, val.clone()));
                }
                info.dirty = false;
            }
        }
        dirty
    }

    /// Register a uniform.
    pub fn register_uniform(&mut self, name: &str, location: u32, uniform_type: &str) {
        self.uniforms.insert(name.to_string(), UniformInfo {
            location: UniformLocation(location),
            uniform_type: uniform_type.to_string(),
            array_size: 1,
            dirty: false,
            value: None,
        });
    }

    /// Register a texture slot.
    pub fn register_texture(&mut self, name: &str, slot: u32, sampler_type: &str) {
        self.texture_slots.insert(name.to_string(), TextureSlotInfo {
            slot: TextureSlot(slot),
            sampler_type: sampler_type.to_string(),
            bound_texture: None,
        });
    }
}

// ---------------------------------------------------------------------------
// Shader complexity metrics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ShaderComplexity {
    pub alu_instructions: u32,
    pub texture_samples: u32,
    pub branch_instructions: u32,
    pub register_pressure: u32,
    pub estimated_cycles: u32,
    pub complexity_tier: ComplexityTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComplexityTier {
    #[default]
    Simple,
    Medium,
    Complex,
    VeryComplex,
}

impl ShaderComplexity {
    /// Estimate complexity from shader source code.
    pub fn estimate_from_source(source: &str) -> Self {
        let mut alu = 0u32;
        let mut tex = 0u32;
        let mut branches = 0u32;

        for line in source.lines() {
            let trimmed = line.trim();

            // Count math operations
            if trimmed.contains("sin(") || trimmed.contains("cos(") || trimmed.contains("tan(") {
                alu += 4; // trig is expensive
            }
            if trimmed.contains("pow(") || trimmed.contains("exp(") || trimmed.contains("log(") {
                alu += 3;
            }
            if trimmed.contains("sqrt(") || trimmed.contains("inversesqrt(") {
                alu += 2;
            }
            if trimmed.contains("normalize(") || trimmed.contains("length(") {
                alu += 2;
            }
            if trimmed.contains("dot(") || trimmed.contains("cross(") {
                alu += 1;
            }
            if trimmed.contains('+') || trimmed.contains('-') || trimmed.contains('*') || trimmed.contains('/') {
                alu += 1;
            }

            // Count texture samples
            if trimmed.contains("texture(") || trimmed.contains("textureLod(")
                || trimmed.contains("texelFetch(") || trimmed.contains("textureGrad(") {
                tex += 1;
            }
            if trimmed.contains("textureCube(") {
                tex += 1;
            }

            // Count branches
            if trimmed.starts_with("if ") || trimmed.starts_with("if(") {
                branches += 1;
            }
            if trimmed.contains("? ") || trimmed.contains("?:") {
                branches += 1;
            }
        }

        let estimated_cycles = alu + tex * 8 + branches * 4;
        let register_pressure = (alu / 4 + tex).min(128);

        let tier = if estimated_cycles < 50 {
            ComplexityTier::Simple
        } else if estimated_cycles < 150 {
            ComplexityTier::Medium
        } else if estimated_cycles < 400 {
            ComplexityTier::Complex
        } else {
            ComplexityTier::VeryComplex
        };

        Self {
            alu_instructions: alu,
            texture_samples: tex,
            branch_instructions: branches,
            register_pressure,
            estimated_cycles,
            complexity_tier: tier,
        }
    }
}

// ---------------------------------------------------------------------------
// Global shader parameters
// ---------------------------------------------------------------------------

/// Engine-wide shader parameters shared across all shaders.
pub struct GlobalShaderParameters {
    pub params: HashMap<String, UniformValue>,
    pub dirty: bool,
    pub version: u64,
}

impl GlobalShaderParameters {
    pub fn new() -> Self {
        let mut params = HashMap::new();

        // Default global parameters
        params.insert("u_Time".to_string(), UniformValue::Float(0.0));
        params.insert("u_DeltaTime".to_string(), UniformValue::Float(0.016));
        params.insert("u_FrameCount".to_string(), UniformValue::UInt(0));
        params.insert("u_ViewMatrix".to_string(), UniformValue::Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]));
        params.insert("u_ProjectionMatrix".to_string(), UniformValue::Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]));
        params.insert("u_ViewProjectionMatrix".to_string(), UniformValue::Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]));
        params.insert("u_CameraPosition".to_string(), UniformValue::Vec3([0.0, 0.0, 0.0]));
        params.insert("u_ScreenSize".to_string(), UniformValue::Vec2([1920.0, 1080.0]));
        params.insert("u_NearFar".to_string(), UniformValue::Vec2([0.1, 1000.0]));
        params.insert("u_AmbientColor".to_string(), UniformValue::Vec3([0.1, 0.1, 0.1]));
        params.insert("u_SunDirection".to_string(), UniformValue::Vec3([0.577, 0.577, 0.577]));
        params.insert("u_SunColor".to_string(), UniformValue::Vec3([1.0, 0.95, 0.85]));
        params.insert("u_SunIntensity".to_string(), UniformValue::Float(3.0));
        params.insert("u_FogColor".to_string(), UniformValue::Vec3([0.7, 0.8, 0.9]));
        params.insert("u_FogDensity".to_string(), UniformValue::Float(0.01));
        params.insert("u_ExposureScale".to_string(), UniformValue::Float(1.0));

        Self { params, dirty: true, version: 0 }
    }

    /// Set a global parameter.
    pub fn set(&mut self, name: &str, value: UniformValue) {
        self.params.insert(name.to_string(), value);
        self.dirty = true;
        self.version += 1;
    }

    /// Get a global parameter.
    pub fn get(&self, name: &str) -> Option<&UniformValue> {
        self.params.get(name)
    }

    /// Update per-frame parameters.
    pub fn update_frame(&mut self, time: f32, dt: f32, frame_count: u32) {
        self.set("u_Time", UniformValue::Float(time));
        self.set("u_DeltaTime", UniformValue::Float(dt));
        self.set("u_FrameCount", UniformValue::UInt(frame_count));
    }

    /// Update camera parameters.
    pub fn update_camera(&mut self, view: [f32; 16], proj: [f32; 16], vp: [f32; 16], pos: [f32; 3]) {
        self.set("u_ViewMatrix", UniformValue::Mat4(view));
        self.set("u_ProjectionMatrix", UniformValue::Mat4(proj));
        self.set("u_ViewProjectionMatrix", UniformValue::Mat4(vp));
        self.set("u_CameraPosition", UniformValue::Vec3(pos));
    }

    /// Apply globals to a shader program.
    pub fn apply_to(&self, program: &mut ShaderProgram) {
        for (name, value) in &self.params {
            program.set_uniform(name, value.clone());
        }
    }

    /// Total size in bytes.
    pub fn total_size_bytes(&self) -> usize {
        self.params.values().map(|v| v.size_bytes()).sum()
    }
}

// ---------------------------------------------------------------------------
// Shader library / manager
// ---------------------------------------------------------------------------

/// Manages all shaders in the engine.
pub struct ShaderManager {
    shaders: HashMap<ShaderId, ShaderProgram>,
    sources: HashMap<String, ShaderSource>,
    variant_cache: HashMap<ShaderVariantKey, ShaderId>,
    include_cache: HashMap<String, String>,
    next_id: u32,
    pub globals: GlobalShaderParameters,
    pub warmup_list: Vec<WarmupEntry>,
    pub stats: ShaderManagerStats,
}

#[derive(Debug, Clone)]
pub struct WarmupEntry {
    pub shader_name: String,
    pub defines: Vec<ShaderDefine>,
    pub priority: u32,
}

#[derive(Debug, Clone, Default)]
pub struct ShaderManagerStats {
    pub total_shaders: u32,
    pub total_variants: u32,
    pub total_compile_time_ms: f32,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_uniforms: u32,
    pub total_texture_slots: u32,
}

impl ShaderManager {
    pub fn new() -> Self {
        Self {
            shaders: HashMap::new(),
            sources: HashMap::new(),
            variant_cache: HashMap::new(),
            include_cache: HashMap::new(),
            next_id: 1,
            globals: GlobalShaderParameters::new(),
            warmup_list: Vec::new(),
            stats: ShaderManagerStats::default(),
        }
    }

    /// Register a shader source.
    pub fn register_source(&mut self, source: ShaderSource) {
        self.sources.insert(source.name.clone(), source);
    }

    /// Register an include file.
    pub fn register_include(&mut self, path: &str, content: &str) {
        self.include_cache.insert(path.to_string(), content.to_string());
    }

    /// Get or create a compiled shader variant.
    pub fn get_or_compile(&mut self, name: &str, defines: &[ShaderDefine]) -> Option<ShaderId> {
        let source = self.sources.get(name)?.clone();

        let mut full_source = source.clone();
        for d in defines {
            if !full_source.defines.contains(d) {
                full_source.defines.push(d.clone());
            }
        }

        let variant_key = full_source.variant_key();

        if let Some(&id) = self.variant_cache.get(&variant_key) {
            self.stats.cache_hits += 1;
            return Some(id);
        }

        self.stats.cache_misses += 1;

        let id = ShaderId(self.next_id);
        self.next_id += 1;

        let start_time = std::time::Instant::now();

        let mut program = ShaderProgram::new(id, name, variant_key);
        program.stages = full_source.stages.iter().map(|s| s.stage).collect();

        // Estimate complexity from vertex + fragment source
        for stage_src in &full_source.stages {
            let processed = full_source.preprocess(stage_src.stage, &|path| {
                self.include_cache.get(path).cloned()
            });

            if stage_src.stage == ShaderStage::Fragment {
                program.complexity = ShaderComplexity::estimate_from_source(&processed);
            }
        }

        program.is_compiled = true;
        program.compile_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        self.stats.total_compile_time_ms += program.compile_time_ms;
        self.stats.total_shaders += 1;
        self.stats.total_variants += 1;

        self.variant_cache.insert(variant_key, id);
        self.shaders.insert(id, program);

        Some(id)
    }

    /// Get a shader program by ID.
    pub fn get_shader(&self, id: ShaderId) -> Option<&ShaderProgram> {
        self.shaders.get(&id)
    }

    /// Get a mutable shader program by ID.
    pub fn get_shader_mut(&mut self, id: ShaderId) -> Option<&mut ShaderProgram> {
        self.shaders.get_mut(&id)
    }

    /// Warm up shaders by compiling them ahead of time.
    pub fn warm_up(&mut self) {
        let entries: Vec<_> = self.warmup_list.drain(..).collect();
        let mut sorted = entries;
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));

        for entry in sorted {
            self.get_or_compile(&entry.shader_name, &entry.defines);
        }
    }

    /// Add a shader to the warm-up list.
    pub fn add_warmup(&mut self, name: &str, defines: Vec<ShaderDefine>, priority: u32) {
        self.warmup_list.push(WarmupEntry {
            shader_name: name.to_string(),
            defines,
            priority,
        });
    }

    /// Remove all variants of a shader (for hot reload).
    pub fn invalidate_shader(&mut self, name: &str) {
        let ids_to_remove: Vec<ShaderId> = self.shaders.iter()
            .filter(|(_, prog)| prog.name == name)
            .map(|(&id, _)| id)
            .collect();

        for id in ids_to_remove {
            if let Some(prog) = self.shaders.remove(&id) {
                self.variant_cache.remove(&prog.variant_key);
                self.stats.total_shaders -= 1;
            }
        }
    }

    /// Get statistics.
    pub fn update_stats(&mut self) {
        self.stats.total_uniforms = self.shaders.values()
            .map(|s| s.uniforms.len() as u32)
            .sum();
        self.stats.total_texture_slots = self.shaders.values()
            .map(|s| s.texture_slots.len() as u32)
            .sum();
    }

    /// Get all shader names.
    pub fn shader_names(&self) -> Vec<&str> {
        self.sources.keys().map(|s| s.as_str()).collect()
    }

    /// Get all variants for a shader.
    pub fn variants_for(&self, name: &str) -> Vec<&ShaderProgram> {
        self.shaders.values()
            .filter(|p| p.name == name)
            .collect()
    }

    /// Find the most complex shader.
    pub fn most_complex_shader(&self) -> Option<(&str, u32)> {
        self.shaders.values()
            .max_by_key(|s| s.complexity.estimated_cycles)
            .map(|s| (s.name.as_str(), s.complexity.estimated_cycles))
    }
}

// ---------------------------------------------------------------------------
// Shader parameter block (UBO data)
// ---------------------------------------------------------------------------

/// A block of parameters for uploading to a uniform buffer.
#[derive(Debug, Clone)]
pub struct ParameterBlock {
    pub name: String,
    pub binding: u32,
    pub data: Vec<u8>,
    pub layout: Vec<ParameterLayout>,
    pub dirty: bool,
}

#[derive(Debug, Clone)]
pub struct ParameterLayout {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub param_type: String,
}

impl ParameterBlock {
    pub fn new(name: &str, binding: u32, size: usize) -> Self {
        Self {
            name: name.to_string(),
            binding,
            data: vec![0u8; size],
            layout: Vec::new(),
            dirty: true,
        }
    }

    /// Write a float at the given offset.
    pub fn write_float(&mut self, offset: usize, value: f32) {
        if offset + 4 <= self.data.len() {
            self.data[offset..offset+4].copy_from_slice(&value.to_le_bytes());
            self.dirty = true;
        }
    }

    /// Write a vec4 at the given offset.
    pub fn write_vec4(&mut self, offset: usize, value: [f32; 4]) {
        if offset + 16 <= self.data.len() {
            for (i, &v) in value.iter().enumerate() {
                self.data[offset + i*4..offset + i*4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            self.dirty = true;
        }
    }

    /// Write a mat4 at the given offset.
    pub fn write_mat4(&mut self, offset: usize, value: &[f32; 16]) {
        if offset + 64 <= self.data.len() {
            for (i, &v) in value.iter().enumerate() {
                self.data[offset + i*4..offset + i*4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            self.dirty = true;
        }
    }

    /// Read a float from the given offset.
    pub fn read_float(&self, offset: usize) -> f32 {
        if offset + 4 <= self.data.len() {
            f32::from_le_bytes([
                self.data[offset], self.data[offset+1],
                self.data[offset+2], self.data[offset+3],
            ])
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Material property sheet
// ---------------------------------------------------------------------------

/// Per-material properties that map to shader uniforms.
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    pub shader_id: ShaderId,
    pub floats: HashMap<String, f32>,
    pub vectors: HashMap<String, [f32; 4]>,
    pub matrices: HashMap<String, [f32; 16]>,
    pub textures: HashMap<String, u64>,
    pub ints: HashMap<String, i32>,
    pub render_queue: i32,
    pub double_sided: bool,
    pub wireframe: bool,
    pub dirty: bool,
}

impl MaterialProperties {
    pub fn new(shader_id: ShaderId) -> Self {
        Self {
            shader_id,
            floats: HashMap::new(),
            vectors: HashMap::new(),
            matrices: HashMap::new(),
            textures: HashMap::new(),
            ints: HashMap::new(),
            render_queue: 2000,
            double_sided: false,
            wireframe: false,
            dirty: true,
        }
    }

    pub fn set_float(&mut self, name: &str, value: f32) {
        self.floats.insert(name.to_string(), value);
        self.dirty = true;
    }

    pub fn set_vector(&mut self, name: &str, value: [f32; 4]) {
        self.vectors.insert(name.to_string(), value);
        self.dirty = true;
    }

    pub fn set_texture(&mut self, name: &str, handle: u64) {
        self.textures.insert(name.to_string(), handle);
        self.dirty = true;
    }

    /// Apply properties to a shader program.
    pub fn apply_to(&self, program: &mut ShaderProgram) {
        for (name, &val) in &self.floats {
            program.set_uniform(name, UniformValue::Float(val));
        }
        for (name, val) in &self.vectors {
            program.set_uniform(name, UniformValue::Vec4(*val));
        }
        for (name, val) in &self.matrices {
            program.set_uniform(name, UniformValue::Mat4(*val));
        }
        for (name, &handle) in &self.textures {
            program.bind_texture(name, handle);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_source_variant_key() {
        let mut src = ShaderSource::new("test_shader");
        src.add_define(ShaderDefine::flag("HAS_NORMAL_MAP"));
        let key1 = src.variant_key();

        src.add_define(ShaderDefine::flag("HAS_SHADOW"));
        let key2 = src.variant_key();

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_shader_complexity() {
        let source = r#"
            vec3 n = normalize(v_Normal);
            float NdotL = dot(n, lightDir);
            vec4 texColor = texture(u_Albedo, v_UV);
            float spec = pow(dot(reflect(-lightDir, n), viewDir), 32.0);
        "#;
        let complexity = ShaderComplexity::estimate_from_source(source);
        assert!(complexity.alu_instructions > 0);
        assert!(complexity.texture_samples > 0);
    }

    #[test]
    fn test_global_params() {
        let mut globals = GlobalShaderParameters::new();
        globals.update_frame(1.0, 0.016, 60);
        match globals.get("u_Time") {
            Some(UniformValue::Float(t)) => assert!(*t > 0.0),
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_shader_manager() {
        let mut mgr = ShaderManager::new();
        let mut src = ShaderSource::new("basic");
        src.add_stage(ShaderStage::Vertex, "void main() {}", "main");
        src.add_stage(ShaderStage::Fragment, "void main() {}", "main");
        mgr.register_source(src);

        let id = mgr.get_or_compile("basic", &[]);
        assert!(id.is_some());

        // Cache hit
        let id2 = mgr.get_or_compile("basic", &[]);
        assert_eq!(id, id2);
        assert_eq!(mgr.stats.cache_hits, 1);
    }

    #[test]
    fn test_parameter_block() {
        let mut block = ParameterBlock::new("globals", 0, 256);
        block.write_float(0, 3.14);
        let val = block.read_float(0);
        assert!((val - 3.14).abs() < 1e-5);
    }

    #[test]
    fn test_material_properties() {
        let props = MaterialProperties::new(ShaderId(1));
        assert_eq!(props.render_queue, 2000);
    }
}
