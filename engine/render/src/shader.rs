// engine/render/src/shader.rs
//
// Shader management utilities: loading, caching, permutation handling, and
// material-shader binding. Also provides built-in WGSL shader source strings
// for the wgpu backend.

use crate::interface::device::RenderDevice;
use crate::interface::resource::{ShaderDesc, ShaderHandle, ShaderStage};
use crate::RenderError;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Built-in WGSL shaders
// ---------------------------------------------------------------------------

/// Built-in triangle shader in WGSL. Contains both vertex and fragment stages.
///
/// The vertex shader outputs a coloured triangle using hardcoded positions.
/// No vertex buffers are required: the vertex index selects from an inline
/// array of positions and colours.
pub const BUILTIN_TRIANGLE_WGSL: &str = r#"
// ---- Built-in coloured triangle shader ----

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Hardcoded triangle positions in clip space.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.5),   // top
        vec2<f32>(-0.5, -0.5),   // bottom-left
        vec2<f32>( 0.5, -0.5),   // bottom-right
    );

    // Vertex colours: red, green, blue.
    var colors = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.color = colors[vertex_index];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

/// Built-in solid-colour shader. Useful for debug rendering and clear
/// operations implemented as full-screen triangles.
pub const BUILTIN_SOLID_COLOR_WGSL: &str = r#"
// ---- Built-in solid colour shader ----
// Draws a full-screen triangle with a uniform colour.

struct PushConstants {
    color: vec4<f32>,
};

// NOTE: push constants are not directly available in WGSL yet. We use
// a hardcoded colour here; the Renderer overrides it by creating a
// per-colour pipeline variant or binding a uniform buffer.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle (oversized, clipped to viewport).
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.1, 0.1, 0.1, 1.0);
}
"#;

/// Built-in vertex-colour shader. Reads position and colour from vertex
/// buffers, applies an MVP transform via push constants / uniform.
pub const BUILTIN_VERTEX_COLOR_WGSL: &str = r#"
// ---- Built-in vertex-colour mesh shader ----

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// Built-in textured quad shader. Reads position + UV, samples from a
/// texture binding.
pub const BUILTIN_TEXTURED_WGSL: &str = r#"
// ---- Built-in textured mesh shader ----

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.uv);
}
"#;

/// Built-in depth-only shader. Writes depth but no colour output.
/// Used for shadow map and depth pre-pass rendering.
pub const BUILTIN_DEPTH_ONLY_WGSL: &str = r#"
// ---- Built-in depth-only shader ----

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 1.0);
    return out;
}
"#;

// ---------------------------------------------------------------------------
// ShaderLibrary
// ---------------------------------------------------------------------------

/// Key used to look up a cached shader module.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ShaderCacheKey {
    /// Path or logical name of the shader source.
    name: String,
    /// Target pipeline stage.
    stage: ShaderStage,
    /// Active permutation defines (sorted for stable hashing).
    defines: Vec<(String, String)>,
}

/// A library that loads, compiles, and caches shader modules.
///
/// Shaders are identified by a logical name (typically a file path relative to
/// the asset root) combined with a pipeline stage and a set of preprocessor
/// defines forming a *permutation*.
pub struct ShaderLibrary {
    /// The render device used to compile shader modules.
    device: Arc<dyn RenderDevice>,
    /// Cache of previously compiled shader handles.
    cache: RwLock<HashMap<ShaderCacheKey, ShaderHandle>>,
    /// Root directory for shader source / bytecode files.
    root_path: PathBuf,
}

impl ShaderLibrary {
    /// Create a new shader library.
    ///
    /// `root_path` is the directory under which compiled shader bytecode
    /// (e.g. `.spv`, `.dxil`, `.metallib`) or WGSL source (`.wgsl`) can be
    /// found.
    pub fn new(device: Arc<dyn RenderDevice>, root_path: impl Into<PathBuf>) -> Self {
        Self {
            device,
            cache: RwLock::new(HashMap::new()),
            root_path: root_path.into(),
        }
    }

    /// Load (or retrieve from cache) a shader module.
    ///
    /// `name` is a logical identifier (e.g. `"shaders/pbr.vert"`). The
    /// library resolves this against [`root_path`](Self::root_path) and loads
    /// the pre-compiled bytecode or WGSL source.
    pub fn load(
        &self,
        name: &str,
        stage: ShaderStage,
        defines: &[(&str, &str)],
    ) -> std::result::Result<ShaderHandle, RenderError> {
        let sorted_defines: Vec<(String, String)> = {
            let mut d: Vec<_> = defines
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            d.sort();
            d
        };

        let key = ShaderCacheKey {
            name: name.to_string(),
            stage,
            defines: sorted_defines,
        };

        // Fast path: return cached handle.
        {
            let cache = self.cache.read();
            if let Some(&handle) = cache.get(&key) {
                return Ok(handle);
            }
        }

        // Slow path: load from disk and compile.
        let bytecode = self.load_bytecode(name, stage, defines)?;
        let handle = self.device.create_shader(&ShaderDesc {
            label: Some(name.to_string()),
            stage,
            bytecode,
            entry_point: String::from("main"),
        })?;

        self.cache.write().insert(key, handle);
        Ok(handle)
    }

    /// Load a WGSL shader from source string directly.
    ///
    /// This bypasses the filesystem and compiles the given WGSL source
    /// directly. Useful for built-in shaders.
    pub fn load_wgsl(
        &self,
        name: &str,
        stage: ShaderStage,
        wgsl_source: &str,
        entry_point: &str,
    ) -> std::result::Result<ShaderHandle, RenderError> {
        let key = ShaderCacheKey {
            name: name.to_string(),
            stage,
            defines: Vec::new(),
        };

        {
            let cache = self.cache.read();
            if let Some(&handle) = cache.get(&key) {
                return Ok(handle);
            }
        }

        let handle = self.device.create_shader(&ShaderDesc {
            label: Some(name.to_string()),
            stage,
            bytecode: wgsl_source.as_bytes().to_vec(),
            entry_point: entry_point.to_string(),
        })?;

        self.cache.write().insert(key, handle);
        Ok(handle)
    }

    /// Resolve a shader name to a file path and read the bytecode.
    fn load_bytecode(
        &self,
        name: &str,
        _stage: ShaderStage,
        _defines: &[(&str, &str)],
    ) -> std::result::Result<Vec<u8>, RenderError> {
        let path = self.root_path.join(name);

        // Try the exact path first.
        if path.exists() {
            return std::fs::read(&path).map_err(|e| {
                RenderError::ShaderLoad(format!(
                    "Failed to read shader {}: {}",
                    path.display(),
                    e
                ))
            });
        }

        // If the name ends with .spv, try looking for a .wgsl counterpart.
        if name.ends_with(".spv") {
            let wgsl_name = name.replace(".spv", ".wgsl");
            // Remove the stage suffix for .wgsl (e.g. "foo.vert.wgsl" -> "foo.wgsl")
            let wgsl_name_simple = wgsl_name
                .replace(".vert.wgsl", ".wgsl")
                .replace(".frag.wgsl", ".wgsl")
                .replace(".comp.wgsl", ".wgsl");

            let wgsl_path = self.root_path.join(&wgsl_name);
            if wgsl_path.exists() {
                return std::fs::read(&wgsl_path).map_err(|e| {
                    RenderError::ShaderLoad(format!(
                        "Failed to read WGSL shader {}: {}",
                        wgsl_path.display(),
                        e
                    ))
                });
            }

            let wgsl_path_simple = self.root_path.join(&wgsl_name_simple);
            if wgsl_path_simple.exists() {
                return std::fs::read(&wgsl_path_simple).map_err(|e| {
                    RenderError::ShaderLoad(format!(
                        "Failed to read WGSL shader {}: {}",
                        wgsl_path_simple.display(),
                        e
                    ))
                });
            }
        }

        // Also try with .wgsl extension directly.
        let wgsl_path = self.root_path.join(format!("{name}.wgsl"));
        if wgsl_path.exists() {
            return std::fs::read(&wgsl_path).map_err(|e| {
                RenderError::ShaderLoad(format!(
                    "Failed to read WGSL shader {}: {}",
                    wgsl_path.display(),
                    e
                ))
            });
        }

        Err(RenderError::ShaderLoad(format!(
            "Shader not found: {} (searched in {})",
            name,
            self.root_path.display()
        )))
    }

    /// Invalidate the entire cache (e.g. after a hot-reload).
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Number of cached shader modules.
    pub fn cached_count(&self) -> usize {
        self.cache.read().len()
    }
}

// ---------------------------------------------------------------------------
// ShaderPermutation
// ---------------------------------------------------------------------------

/// Represents a specific set of preprocessor defines that together select a
/// unique compiled variant of a shader.
///
/// Permutation management prevents the combinatorial explosion of
/// hand-written shader files by encoding feature toggles (e.g.
/// `HAS_NORMAL_MAP`, `USE_SKINNING`) as compile-time constants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderPermutation {
    /// Sorted list of (define_name, define_value) pairs.
    defines: Vec<(String, String)>,
}

impl ShaderPermutation {
    /// Create a new empty permutation (base variant).
    pub fn new() -> Self {
        Self {
            defines: Vec::new(),
        }
    }

    /// Add or update a define.
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<String>) -> &mut Self {
        let name = name.into();
        let value = value.into();
        if let Some(existing) = self.defines.iter_mut().find(|(n, _)| *n == name) {
            existing.1 = value;
        } else {
            self.defines.push((name, value));
            self.defines.sort_by(|a, b| a.0.cmp(&b.0));
        }
        self
    }

    /// Remove a define.
    pub fn unset(&mut self, name: &str) -> &mut Self {
        self.defines.retain(|(n, _)| n != name);
        self
    }

    /// Enable a boolean flag define (value = "1").
    pub fn enable(&mut self, name: impl Into<String>) -> &mut Self {
        self.set(name, "1")
    }

    /// Get the defines as a slice of (name, value) pairs.
    pub fn defines(&self) -> &[(String, String)] {
        &self.defines
    }

    /// Convert to the slice format expected by [`ShaderLibrary::load`].
    pub fn as_ref_pairs(&self) -> Vec<(&str, &str)> {
        self.defines
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }

    /// Compute a stable hash of this permutation for use in file-name
    /// suffixes and cache keys.
    pub fn hash_u64(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.defines.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for ShaderPermutation {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MaterialShader
// ---------------------------------------------------------------------------

/// Bundles the shader handles and permutation for a specific material type.
///
/// A `MaterialShader` is the link between the material system (which
/// describes *what* to render) and the pipeline system (which describes
/// *how* to render it).
#[derive(Debug, Clone)]
pub struct MaterialShader {
    /// Logical name of the shader program (e.g. `"shaders/pbr"`).
    pub shader_name: String,
    /// Active permutation for this material instance.
    pub permutation: ShaderPermutation,
    /// Cached vertex shader handle (populated after first load).
    pub vertex_shader: Option<ShaderHandle>,
    /// Cached fragment shader handle (populated after first load).
    pub fragment_shader: Option<ShaderHandle>,
}

impl MaterialShader {
    /// Create a new material-shader binding.
    pub fn new(shader_name: impl Into<String>) -> Self {
        Self {
            shader_name: shader_name.into(),
            permutation: ShaderPermutation::new(),
            vertex_shader: None,
            fragment_shader: None,
        }
    }

    /// Load (or reload) the shader handles from the library.
    ///
    /// This populates `vertex_shader` and `fragment_shader` using the
    /// current permutation. Tries WGSL first, then falls back to SPIR-V.
    pub fn load(&mut self, library: &ShaderLibrary) -> std::result::Result<(), RenderError> {
        let pairs = self.permutation.as_ref_pairs();

        // Try WGSL first, then SPIR-V.
        let vert_name_wgsl = format!("{}.wgsl", self.shader_name);
        let vert_name_spv = format!("{}.vert.spv", self.shader_name);

        let vert_result = library.load(&vert_name_wgsl, ShaderStage::Vertex, &pairs);
        self.vertex_shader = Some(match vert_result {
            Ok(h) => h,
            Err(_) => library.load(&vert_name_spv, ShaderStage::Vertex, &pairs)?,
        });

        let frag_name_wgsl = format!("{}.wgsl", self.shader_name);
        let frag_name_spv = format!("{}.frag.spv", self.shader_name);

        let frag_result = library.load(&frag_name_wgsl, ShaderStage::Fragment, &pairs);
        self.fragment_shader = Some(match frag_result {
            Ok(h) => h,
            Err(_) => library.load(&frag_name_spv, ShaderStage::Fragment, &pairs)?,
        });

        Ok(())
    }

    /// Whether both shader stages have been loaded.
    pub fn is_loaded(&self) -> bool {
        self.vertex_shader.is_some() && self.fragment_shader.is_some()
    }
}
