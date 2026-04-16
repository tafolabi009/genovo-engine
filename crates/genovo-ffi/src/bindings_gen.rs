//! # Binding Generator
//!
//! Programmatic generation of language bindings for the Genovo FFI layer.
//! Produces C header files, C++ wrapper classes, Python ctypes bindings,
//! and C# P/Invoke declarations.
//!
//! This module serves as an alternative to `cbindgen` when fine-grained control
//! over the output format is required, or when bindings for languages beyond C
//! are needed.

use std::fmt::Write;

// ============================================================================
// Function signature metadata
// ============================================================================

/// Describes a C function parameter.
#[derive(Debug, Clone)]
pub struct FfiParam {
    /// Parameter name.
    pub name: String,
    /// C type name (e.g., "float", "const char*", "uint32_t").
    pub c_type: String,
    /// Python ctypes type (e.g., "c_float", "c_char_p").
    pub py_type: String,
    /// C# type (e.g., "float", "string", "uint").
    pub cs_type: String,
    /// Whether this is an output parameter.
    pub is_out: bool,
}

/// Describes a C function exported by the FFI layer.
#[derive(Debug, Clone)]
pub struct FfiFunctionDesc {
    /// Function name (the symbol name).
    pub name: String,
    /// Return type in C.
    pub c_return: String,
    /// Return type in Python ctypes.
    pub py_return: String,
    /// Return type in C#.
    pub cs_return: String,
    /// Parameters.
    pub params: Vec<FfiParam>,
    /// Documentation comment.
    pub doc: String,
    /// Whether the function is safe to call from any thread.
    pub thread_safe: bool,
}

/// Describes a C struct exported by the FFI layer.
#[derive(Debug, Clone)]
pub struct FfiStructDesc {
    /// Struct name.
    pub name: String,
    /// Fields.
    pub fields: Vec<FfiParam>,
    /// Documentation comment.
    pub doc: String,
}

/// Describes an enum exported by the FFI layer.
#[derive(Debug, Clone)]
pub struct FfiEnumDesc {
    /// Enum name.
    pub name: String,
    /// Variants as (name, value) pairs.
    pub variants: Vec<(String, i32)>,
    /// Documentation comment.
    pub doc: String,
}

// ============================================================================
// Registry of all exported symbols
// ============================================================================

/// Collects all exported FFI symbols for binding generation.
fn get_exported_functions() -> Vec<FfiFunctionDesc> {
    vec![
        // -- Engine lifecycle --
        FfiFunctionDesc {
            name: "genovo_init".into(),
            c_return: "int32_t".into(),
            py_return: "c_int32".into(),
            cs_return: "int".into(),
            params: vec![],
            doc: "Initialize the Genovo engine. Must be called before any other function.".into(),
            thread_safe: false,
        },
        FfiFunctionDesc {
            name: "genovo_shutdown".into(),
            c_return: "int32_t".into(),
            py_return: "c_int32".into(),
            cs_return: "int".into(),
            params: vec![],
            doc: "Shut down the engine. All resources must be freed before calling this.".into(),
            thread_safe: false,
        },
        FfiFunctionDesc {
            name: "genovo_version".into(),
            c_return: "uint32_t".into(),
            py_return: "c_uint32".into(),
            cs_return: "uint".into(),
            params: vec![],
            doc: "Return the packed version number.".into(),
            thread_safe: true,
        },
        FfiFunctionDesc {
            name: "genovo_version_string".into(),
            c_return: "const char*".into(),
            py_return: "c_char_p".into(),
            cs_return: "IntPtr".into(),
            params: vec![],
            doc: "Return a static string describing the engine version.".into(),
            thread_safe: true,
        },
        FfiFunctionDesc {
            name: "genovo_get_last_error".into(),
            c_return: "const char*".into(),
            py_return: "c_char_p".into(),
            cs_return: "IntPtr".into(),
            params: vec![],
            doc: "Get the last error message on this thread.".into(),
            thread_safe: true,
        },
        FfiFunctionDesc {
            name: "genovo_clear_error".into(),
            c_return: "void".into(),
            py_return: "None".into(),
            cs_return: "void".into(),
            params: vec![],
            doc: "Clear the last error message on this thread.".into(),
            thread_safe: true,
        },
        // -- Engine instance --
        FfiFunctionDesc {
            name: "genovo_engine_create".into(),
            c_return: "void*".into(),
            py_return: "c_void_p".into(),
            cs_return: "IntPtr".into(),
            params: vec![FfiParam {
                name: "config".into(),
                c_type: "const FfiEngineConfig*".into(),
                py_type: "POINTER(FfiEngineConfig)".into(),
                cs_type: "ref FfiEngineConfig".into(),
                is_out: false,
            }],
            doc: "Create a new engine instance.".into(),
            thread_safe: false,
        },
        FfiFunctionDesc {
            name: "genovo_engine_destroy".into(),
            c_return: "int32_t".into(),
            py_return: "c_int32".into(),
            cs_return: "int".into(),
            params: vec![FfiParam {
                name: "engine".into(),
                c_type: "void*".into(),
                py_type: "c_void_p".into(),
                cs_type: "IntPtr".into(),
                is_out: false,
            }],
            doc: "Destroy an engine instance.".into(),
            thread_safe: false,
        },
        FfiFunctionDesc {
            name: "genovo_engine_update".into(),
            c_return: "int32_t".into(),
            py_return: "c_int32".into(),
            cs_return: "int".into(),
            params: vec![
                FfiParam {
                    name: "engine".into(),
                    c_type: "void*".into(),
                    py_type: "c_void_p".into(),
                    cs_type: "IntPtr".into(),
                    is_out: false,
                },
                FfiParam {
                    name: "dt".into(),
                    c_type: "float".into(),
                    py_type: "c_float".into(),
                    cs_type: "float".into(),
                    is_out: false,
                },
            ],
            doc: "Update the engine by dt seconds.".into(),
            thread_safe: false,
        },
        // -- Physics --
        FfiFunctionDesc {
            name: "genovo_physics_world_create".into(),
            c_return: "void*".into(),
            py_return: "c_void_p".into(),
            cs_return: "IntPtr".into(),
            params: vec![
                FfiParam {
                    name: "gravity_x".into(),
                    c_type: "float".into(),
                    py_type: "c_float".into(),
                    cs_type: "float".into(),
                    is_out: false,
                },
                FfiParam {
                    name: "gravity_y".into(),
                    c_type: "float".into(),
                    py_type: "c_float".into(),
                    cs_type: "float".into(),
                    is_out: false,
                },
                FfiParam {
                    name: "gravity_z".into(),
                    c_type: "float".into(),
                    py_type: "c_float".into(),
                    cs_type: "float".into(),
                    is_out: false,
                },
            ],
            doc: "Create a new physics world with the specified gravity.".into(),
            thread_safe: false,
        },
        FfiFunctionDesc {
            name: "genovo_physics_world_destroy".into(),
            c_return: "int32_t".into(),
            py_return: "c_int32".into(),
            cs_return: "int".into(),
            params: vec![FfiParam {
                name: "world".into(),
                c_type: "void*".into(),
                py_type: "c_void_p".into(),
                cs_type: "IntPtr".into(),
                is_out: false,
            }],
            doc: "Destroy a physics world.".into(),
            thread_safe: false,
        },
        FfiFunctionDesc {
            name: "genovo_physics_world_step".into(),
            c_return: "int32_t".into(),
            py_return: "c_int32".into(),
            cs_return: "int".into(),
            params: vec![
                FfiParam {
                    name: "world".into(),
                    c_type: "void*".into(),
                    py_type: "c_void_p".into(),
                    cs_type: "IntPtr".into(),
                    is_out: false,
                },
                FfiParam {
                    name: "dt".into(),
                    c_type: "float".into(),
                    py_type: "c_float".into(),
                    cs_type: "float".into(),
                    is_out: false,
                },
            ],
            doc: "Advance the physics simulation by dt seconds.".into(),
            thread_safe: false,
        },
        // -- Audio --
        FfiFunctionDesc {
            name: "genovo_audio_mixer_create".into(),
            c_return: "void*".into(),
            py_return: "c_void_p".into(),
            cs_return: "IntPtr".into(),
            params: vec![],
            doc: "Create a new software audio mixer.".into(),
            thread_safe: false,
        },
        FfiFunctionDesc {
            name: "genovo_audio_mixer_destroy".into(),
            c_return: "int32_t".into(),
            py_return: "c_int32".into(),
            cs_return: "int".into(),
            params: vec![FfiParam {
                name: "mixer".into(),
                c_type: "void*".into(),
                py_type: "c_void_p".into(),
                cs_type: "IntPtr".into(),
                is_out: false,
            }],
            doc: "Destroy a software audio mixer.".into(),
            thread_safe: false,
        },
        // -- Memory --
        FfiFunctionDesc {
            name: "genovo_alloc".into(),
            c_return: "void*".into(),
            py_return: "c_void_p".into(),
            cs_return: "IntPtr".into(),
            params: vec![
                FfiParam {
                    name: "size".into(),
                    c_type: "size_t".into(),
                    py_type: "c_size_t".into(),
                    cs_type: "UIntPtr".into(),
                    is_out: false,
                },
                FfiParam {
                    name: "align".into(),
                    c_type: "size_t".into(),
                    py_type: "c_size_t".into(),
                    cs_type: "UIntPtr".into(),
                    is_out: false,
                },
            ],
            doc: "Allocate aligned memory.".into(),
            thread_safe: true,
        },
        FfiFunctionDesc {
            name: "genovo_free".into(),
            c_return: "void".into(),
            py_return: "None".into(),
            cs_return: "void".into(),
            params: vec![
                FfiParam {
                    name: "ptr".into(),
                    c_type: "void*".into(),
                    py_type: "c_void_p".into(),
                    cs_type: "IntPtr".into(),
                    is_out: false,
                },
                FfiParam {
                    name: "size".into(),
                    c_type: "size_t".into(),
                    py_type: "c_size_t".into(),
                    cs_type: "UIntPtr".into(),
                    is_out: false,
                },
                FfiParam {
                    name: "align".into(),
                    c_type: "size_t".into(),
                    py_type: "c_size_t".into(),
                    cs_type: "UIntPtr".into(),
                    is_out: false,
                },
            ],
            doc: "Free memory allocated with genovo_alloc.".into(),
            thread_safe: true,
        },
    ]
}

fn get_exported_structs() -> Vec<FfiStructDesc> {
    vec![
        FfiStructDesc {
            name: "FfiVec3".into(),
            fields: vec![
                FfiParam { name: "x".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "y".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "z".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
            ],
            doc: "3-component floating-point vector.".into(),
        },
        FfiStructDesc {
            name: "FfiQuat".into(),
            fields: vec![
                FfiParam { name: "x".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "y".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "z".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "w".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
            ],
            doc: "Quaternion (x, y, z, w).".into(),
        },
        FfiStructDesc {
            name: "FfiTransform".into(),
            fields: vec![
                FfiParam { name: "position".into(), c_type: "FfiVec3".into(), py_type: "FfiVec3".into(), cs_type: "FfiVec3".into(), is_out: false },
                FfiParam { name: "rotation".into(), c_type: "FfiQuat".into(), py_type: "FfiQuat".into(), cs_type: "FfiQuat".into(), is_out: false },
                FfiParam { name: "scale".into(), c_type: "FfiVec3".into(), py_type: "FfiVec3".into(), cs_type: "FfiVec3".into(), is_out: false },
            ],
            doc: "Transform (position + rotation + scale).".into(),
        },
        FfiStructDesc {
            name: "FfiMat4".into(),
            fields: vec![
                FfiParam { name: "data".into(), c_type: "float[16]".into(), py_type: "c_float * 16".into(), cs_type: "fixed float[16]".into(), is_out: false },
            ],
            doc: "4x4 matrix (column-major, 16 floats).".into(),
        },
        FfiStructDesc {
            name: "FfiColor".into(),
            fields: vec![
                FfiParam { name: "r".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "g".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "b".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "a".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
            ],
            doc: "RGBA color (linear, 0..1 range).".into(),
        },
        FfiStructDesc {
            name: "FfiRect".into(),
            fields: vec![
                FfiParam { name: "x".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "y".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "width".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
                FfiParam { name: "height".into(), c_type: "float".into(), py_type: "c_float".into(), cs_type: "float".into(), is_out: false },
            ],
            doc: "2D rectangle.".into(),
        },
        FfiStructDesc {
            name: "FfiRay".into(),
            fields: vec![
                FfiParam { name: "origin".into(), c_type: "FfiVec3".into(), py_type: "FfiVec3".into(), cs_type: "FfiVec3".into(), is_out: false },
                FfiParam { name: "direction".into(), c_type: "FfiVec3".into(), py_type: "FfiVec3".into(), cs_type: "FfiVec3".into(), is_out: false },
            ],
            doc: "A ray in 3D space.".into(),
        },
    ]
}

fn get_exported_enums() -> Vec<FfiEnumDesc> {
    vec![
        FfiEnumDesc {
            name: "FfiAssetStatus".into(),
            variants: vec![
                ("NotLoaded".into(), 0),
                ("Loading".into(), 1),
                ("Loaded".into(), 2),
                ("Failed".into(), 3),
                ("Unloaded".into(), 4),
            ],
            doc: "Asset loading status.".into(),
        },
        FfiEnumDesc {
            name: "FfiEventType".into(),
            variants: vec![
                ("EngineInit".into(), 0),
                ("EngineShutdown".into(), 1),
                ("FrameBegin".into(), 2),
                ("FrameEnd".into(), 3),
                ("EntityCreated".into(), 4),
                ("EntityDestroyed".into(), 5),
                ("PhysicsStep".into(), 6),
                ("Collision".into(), 7),
                ("AssetLoaded".into(), 8),
                ("AssetUnloaded".into(), 9),
                ("SceneLoaded".into(), 10),
                ("WindowResize".into(), 11),
                ("WindowFocus".into(), 12),
                ("Input".into(), 13),
                ("Custom".into(), 100),
            ],
            doc: "Event types for the callback system.".into(),
        },
        FfiEnumDesc {
            name: "FfiErrorCategory".into(),
            variants: vec![
                ("None".into(), 0),
                ("InvalidArgument".into(), 1),
                ("NotFound".into(), 2),
                ("OutOfMemory".into(), 3),
                ("IoError".into(), 4),
                ("PermissionDenied".into(), 5),
                ("Timeout".into(), 6),
                ("Internal".into(), 7),
                ("NotSupported".into(), 8),
                ("Cancelled".into(), 9),
            ],
            doc: "Error category for extended error reporting.".into(),
        },
    ]
}

// ============================================================================
// C header generation
// ============================================================================

/// Generate a C header file containing all exported FFI symbols.
pub fn generate_c_header() -> String {
    let mut out = String::with_capacity(8192);

    writeln!(out, "/* ================================================================").unwrap();
    writeln!(out, " * genovo_bindings.h -- Auto-generated Genovo C API bindings").unwrap();
    writeln!(out, " * Generated by genovo-ffi bindings_gen module").unwrap();
    writeln!(out, " * DO NOT MODIFY MANUALLY").unwrap();
    writeln!(out, " * ================================================================ */").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#ifndef GENOVO_BINDINGS_H").unwrap();
    writeln!(out, "#define GENOVO_BINDINGS_H").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#include <stdint.h>").unwrap();
    writeln!(out, "#include <stddef.h>").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#ifdef __cplusplus").unwrap();
    writeln!(out, "extern \"C\" {{").unwrap();
    writeln!(out, "#endif").unwrap();
    writeln!(out).unwrap();

    // Error codes
    writeln!(out, "/* Error codes */").unwrap();
    writeln!(out, "#define FFI_OK                     0").unwrap();
    writeln!(out, "#define FFI_ERR_NULL_POINTER       -1").unwrap();
    writeln!(out, "#define FFI_ERR_INVALID_HANDLE     -2").unwrap();
    writeln!(out, "#define FFI_ERR_INTERNAL           -3").unwrap();
    writeln!(out, "#define FFI_ERR_INVALID_PARAMETER  -4").unwrap();
    writeln!(out, "#define FFI_ERR_OUT_OF_MEMORY      -5").unwrap();
    writeln!(out, "#define FFI_ERR_NOT_IMPLEMENTED    -6").unwrap();
    writeln!(out, "#define FFI_ERR_PANIC              -7").unwrap();
    writeln!(out).unwrap();

    // Enums
    for e in &get_exported_enums() {
        writeln!(out, "/* {} */", e.doc).unwrap();
        writeln!(out, "typedef enum {{").unwrap();
        for (i, (name, value)) in e.variants.iter().enumerate() {
            let comma = if i + 1 < e.variants.len() { "," } else { "" };
            writeln!(out, "    {}_{} = {}{}", e.name.to_uppercase(), name.to_uppercase(), value, comma).unwrap();
        }
        writeln!(out, "}} {};", e.name).unwrap();
        writeln!(out).unwrap();
    }

    // Structs
    for s in &get_exported_structs() {
        writeln!(out, "/* {} */", s.doc).unwrap();
        writeln!(out, "typedef struct {{").unwrap();
        for f in &s.fields {
            if f.c_type.contains('[') {
                // Array field: e.g., "float[16]" -> "float data[16];"
                let bracket_pos = f.c_type.find('[').unwrap();
                let base_type = &f.c_type[..bracket_pos];
                let array_part = &f.c_type[bracket_pos..];
                writeln!(out, "    {} {}{};", base_type, f.name, array_part).unwrap();
            } else {
                writeln!(out, "    {} {};", f.c_type, f.name).unwrap();
            }
        }
        writeln!(out, "}} {};", s.name).unwrap();
        writeln!(out).unwrap();
    }

    // Functions
    writeln!(out, "/* ---- Functions ---- */").unwrap();
    writeln!(out).unwrap();
    for func in &get_exported_functions() {
        writeln!(out, "/* {} */", func.doc).unwrap();
        let params = if func.params.is_empty() {
            "void".to_string()
        } else {
            func.params
                .iter()
                .map(|p| format!("{} {}", p.c_type, p.name))
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(out, "{} {}({});", func.c_return, func.name, params).unwrap();
        writeln!(out).unwrap();
    }

    writeln!(out, "#ifdef __cplusplus").unwrap();
    writeln!(out, "}} /* extern \"C\" */").unwrap();
    writeln!(out, "#endif").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#endif /* GENOVO_BINDINGS_H */").unwrap();

    out
}

// ============================================================================
// C++ wrapper generation
// ============================================================================

/// Generate C++ wrapper classes that provide RAII and exception-safe access
/// to the C API.
pub fn generate_cpp_wrapper() -> String {
    let mut out = String::with_capacity(8192);

    writeln!(out, "/* ================================================================").unwrap();
    writeln!(out, " * genovo.hpp -- Auto-generated C++ wrapper for the Genovo engine").unwrap();
    writeln!(out, " * Generated by genovo-ffi bindings_gen module").unwrap();
    writeln!(out, " * DO NOT MODIFY MANUALLY").unwrap();
    writeln!(out, " * ================================================================ */").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#pragma once").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#include \"genovo_bindings.h\"").unwrap();
    writeln!(out, "#include <stdexcept>").unwrap();
    writeln!(out, "#include <string>").unwrap();
    writeln!(out, "#include <memory>").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "namespace genovo {{").unwrap();
    writeln!(out).unwrap();

    // Helper: check result
    writeln!(out, "inline void check_result(int32_t result) {{").unwrap();
    writeln!(out, "    if (result != FFI_OK) {{").unwrap();
    writeln!(out, "        const char* err = genovo_get_last_error();").unwrap();
    writeln!(out, "        std::string msg = err ? err : \"Unknown error\";").unwrap();
    writeln!(out, "        throw std::runtime_error(\"Genovo error (\" + std::to_string(result) + \"): \" + msg);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // Vec3 wrapper
    writeln!(out, "struct Vec3 : public FfiVec3 {{").unwrap();
    writeln!(out, "    Vec3() : FfiVec3{{0.0f, 0.0f, 0.0f}} {{}}").unwrap();
    writeln!(out, "    Vec3(float x, float y, float z) : FfiVec3{{x, y, z}} {{}}").unwrap();
    writeln!(out, "    Vec3(const FfiVec3& v) : FfiVec3(v) {{}}").unwrap();
    writeln!(out, "    float length() const {{ return std::sqrt(x*x + y*y + z*z); }}").unwrap();
    writeln!(out, "    Vec3 normalized() const {{").unwrap();
    writeln!(out, "        float l = length();").unwrap();
    writeln!(out, "        return l > 1e-6f ? Vec3(x/l, y/l, z/l) : Vec3();").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    float dot(const Vec3& o) const {{ return x*o.x + y*o.y + z*o.z; }}").unwrap();
    writeln!(out, "    Vec3 cross(const Vec3& o) const {{").unwrap();
    writeln!(out, "        return Vec3(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x);").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    Vec3 operator+(const Vec3& o) const {{ return Vec3(x+o.x, y+o.y, z+o.z); }}").unwrap();
    writeln!(out, "    Vec3 operator-(const Vec3& o) const {{ return Vec3(x-o.x, y-o.y, z-o.z); }}").unwrap();
    writeln!(out, "    Vec3 operator*(float s) const {{ return Vec3(x*s, y*s, z*s); }}").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();

    // PhysicsWorld wrapper
    writeln!(out, "class PhysicsWorld {{").unwrap();
    writeln!(out, "public:").unwrap();
    writeln!(out, "    explicit PhysicsWorld(Vec3 gravity = Vec3(0, -9.81f, 0)) {{").unwrap();
    writeln!(out, "        handle_ = genovo_physics_world_create(gravity.x, gravity.y, gravity.z);").unwrap();
    writeln!(out, "        if (!handle_) throw std::runtime_error(\"Failed to create physics world\");").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "    ~PhysicsWorld() {{ if (handle_) genovo_physics_world_destroy(handle_); }}").unwrap();
    writeln!(out, "    PhysicsWorld(const PhysicsWorld&) = delete;").unwrap();
    writeln!(out, "    PhysicsWorld& operator=(const PhysicsWorld&) = delete;").unwrap();
    writeln!(out, "    PhysicsWorld(PhysicsWorld&& o) noexcept : handle_(o.handle_) {{ o.handle_ = nullptr; }}").unwrap();
    writeln!(out, "    void step(float dt) {{ check_result(genovo_physics_world_step(handle_, dt)); }}").unwrap();
    writeln!(out, "    void set_gravity(Vec3 g) {{ check_result(genovo_physics_world_set_gravity(handle_, g.x, g.y, g.z)); }}").unwrap();
    writeln!(out, "    void* raw() {{ return handle_; }}").unwrap();
    writeln!(out, "private:").unwrap();
    writeln!(out, "    void* handle_;").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();

    // AudioMixer wrapper
    writeln!(out, "class AudioMixer {{").unwrap();
    writeln!(out, "public:").unwrap();
    writeln!(out, "    AudioMixer() {{ handle_ = genovo_audio_mixer_create(); }}").unwrap();
    writeln!(out, "    ~AudioMixer() {{ if (handle_) genovo_audio_mixer_destroy(handle_); }}").unwrap();
    writeln!(out, "    AudioMixer(const AudioMixer&) = delete;").unwrap();
    writeln!(out, "    AudioMixer& operator=(const AudioMixer&) = delete;").unwrap();
    writeln!(out, "    void update(float dt) {{ check_result(genovo_audio_mixer_update(handle_, dt)); }}").unwrap();
    writeln!(out, "    void set_master_volume(float vol) {{ check_result(genovo_audio_set_master_volume(handle_, vol)); }}").unwrap();
    writeln!(out, "    void stop_all() {{ check_result(genovo_audio_stop_all(handle_)); }}").unwrap();
    writeln!(out, "    void* raw() {{ return handle_; }}").unwrap();
    writeln!(out, "private:").unwrap();
    writeln!(out, "    void* handle_;").unwrap();
    writeln!(out, "}};").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "}} // namespace genovo").unwrap();

    out
}

// ============================================================================
// Python bindings generation
// ============================================================================

/// Generate Python ctypes bindings for the Genovo FFI layer.
pub fn generate_python_bindings() -> String {
    let mut out = String::with_capacity(8192);

    writeln!(out, "\"\"\"").unwrap();
    writeln!(out, "genovo.py -- Auto-generated Python ctypes bindings for the Genovo engine").unwrap();
    writeln!(out, "Generated by genovo-ffi bindings_gen module").unwrap();
    writeln!(out, "DO NOT MODIFY MANUALLY").unwrap();
    writeln!(out, "\"\"\"").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "import ctypes").unwrap();
    writeln!(out, "import ctypes.util").unwrap();
    writeln!(out, "import os").unwrap();
    writeln!(out, "import sys").unwrap();
    writeln!(out, "from ctypes import (").unwrap();
    writeln!(out, "    c_char_p, c_float, c_int32, c_uint32, c_uint64,").unwrap();
    writeln!(out, "    c_void_p, c_size_t, c_uint8, c_double,").unwrap();
    writeln!(out, "    Structure, POINTER, CFUNCTYPE,").unwrap();
    writeln!(out, ")").unwrap();
    writeln!(out).unwrap();

    // Error codes
    writeln!(out, "# Error codes").unwrap();
    writeln!(out, "FFI_OK = 0").unwrap();
    writeln!(out, "FFI_ERR_NULL_POINTER = -1").unwrap();
    writeln!(out, "FFI_ERR_INVALID_HANDLE = -2").unwrap();
    writeln!(out, "FFI_ERR_INTERNAL = -3").unwrap();
    writeln!(out, "FFI_ERR_INVALID_PARAMETER = -4").unwrap();
    writeln!(out, "FFI_ERR_OUT_OF_MEMORY = -5").unwrap();
    writeln!(out, "FFI_ERR_NOT_IMPLEMENTED = -6").unwrap();
    writeln!(out, "FFI_ERR_PANIC = -7").unwrap();
    writeln!(out).unwrap();

    // Enums as integer constants
    for e in &get_exported_enums() {
        writeln!(out, "# {} -- {}", e.name, e.doc).unwrap();
        for (name, value) in &e.variants {
            writeln!(out, "{}_{} = {}", e.name.to_uppercase(), name.to_uppercase(), value).unwrap();
        }
        writeln!(out).unwrap();
    }

    // Structs
    for s in &get_exported_structs() {
        writeln!(out, "# {}", s.doc).unwrap();
        writeln!(out, "class {}(Structure):", s.name).unwrap();
        writeln!(out, "    _fields_ = [").unwrap();
        for f in &s.fields {
            writeln!(out, "        (\"{}\", {}),", f.name, f.py_type).unwrap();
        }
        writeln!(out, "    ]").unwrap();
        writeln!(out).unwrap();
    }

    // Library loading
    writeln!(out, "def _load_library():").unwrap();
    writeln!(out, "    \"\"\"Load the genovo shared library.\"\"\"").unwrap();
    writeln!(out, "    if sys.platform == 'win32':").unwrap();
    writeln!(out, "        lib_name = 'genovo_ffi.dll'").unwrap();
    writeln!(out, "    elif sys.platform == 'darwin':").unwrap();
    writeln!(out, "        lib_name = 'libgenovo_ffi.dylib'").unwrap();
    writeln!(out, "    else:").unwrap();
    writeln!(out, "        lib_name = 'libgenovo_ffi.so'").unwrap();
    writeln!(out, "    # Try current directory first, then system paths").unwrap();
    writeln!(out, "    search_paths = [").unwrap();
    writeln!(out, "        os.path.join(os.path.dirname(__file__), lib_name),").unwrap();
    writeln!(out, "        lib_name,").unwrap();
    writeln!(out, "    ]").unwrap();
    writeln!(out, "    for path in search_paths:").unwrap();
    writeln!(out, "        try:").unwrap();
    writeln!(out, "            return ctypes.CDLL(path)").unwrap();
    writeln!(out, "        except OSError:").unwrap();
    writeln!(out, "            continue").unwrap();
    writeln!(out, "    raise OSError(f'Could not find {{lib_name}}')").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "_lib = _load_library()").unwrap();
    writeln!(out).unwrap();

    // Function declarations
    writeln!(out, "# ---- Function declarations ----").unwrap();
    writeln!(out).unwrap();
    for func in &get_exported_functions() {
        writeln!(out, "# {}", func.doc).unwrap();
        let py_ret = if func.py_return == "None" {
            "None".to_string()
        } else {
            func.py_return.clone()
        };
        writeln!(out, "_lib.{}.restype = {}", func.name, py_ret).unwrap();
        if func.params.is_empty() {
            writeln!(out, "_lib.{}.argtypes = []", func.name).unwrap();
        } else {
            let arg_types: Vec<String> = func.params.iter().map(|p| p.py_type.clone()).collect();
            writeln!(out, "_lib.{}.argtypes = [{}]", func.name, arg_types.join(", ")).unwrap();
        }
        writeln!(out).unwrap();
    }

    // Helper function
    writeln!(out, "def check_result(result):").unwrap();
    writeln!(out, "    \"\"\"Check an FFI result code and raise on error.\"\"\"").unwrap();
    writeln!(out, "    if result != FFI_OK:").unwrap();
    writeln!(out, "        err = _lib.genovo_get_last_error()").unwrap();
    writeln!(out, "        msg = err.decode('utf-8') if err else 'Unknown error'").unwrap();
    writeln!(out, "        raise RuntimeError(f'Genovo error ({{result}}): {{msg}}')").unwrap();
    writeln!(out).unwrap();

    // High-level Python class
    writeln!(out, "class Engine:").unwrap();
    writeln!(out, "    \"\"\"High-level Python wrapper for the Genovo engine.\"\"\"").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    def __init__(self):").unwrap();
    writeln!(out, "        check_result(_lib.genovo_init())").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    def shutdown(self):").unwrap();
    writeln!(out, "        check_result(_lib.genovo_shutdown())").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    @staticmethod").unwrap();
    writeln!(out, "    def version():").unwrap();
    writeln!(out, "        return _lib.genovo_version()").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    @staticmethod").unwrap();
    writeln!(out, "    def version_string():").unwrap();
    writeln!(out, "        ptr = _lib.genovo_version_string()").unwrap();
    writeln!(out, "        return ptr.decode('utf-8') if ptr else ''").unwrap();
    writeln!(out).unwrap();

    out
}

// ============================================================================
// C# P/Invoke generation
// ============================================================================

/// Generate C# P/Invoke declarations for the Genovo FFI layer.
pub fn generate_csharp_bindings() -> String {
    let mut out = String::with_capacity(8192);

    writeln!(out, "// ================================================================").unwrap();
    writeln!(out, "// Genovo.Interop.cs -- Auto-generated C# P/Invoke bindings").unwrap();
    writeln!(out, "// Generated by genovo-ffi bindings_gen module").unwrap();
    writeln!(out, "// DO NOT MODIFY MANUALLY").unwrap();
    writeln!(out, "// ================================================================").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "using System;").unwrap();
    writeln!(out, "using System.Runtime.InteropServices;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "namespace Genovo.Interop").unwrap();
    writeln!(out, "{{").unwrap();

    // Error codes
    writeln!(out, "    public static class ErrorCodes").unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        public const int OK = 0;").unwrap();
    writeln!(out, "        public const int ERR_NULL_POINTER = -1;").unwrap();
    writeln!(out, "        public const int ERR_INVALID_HANDLE = -2;").unwrap();
    writeln!(out, "        public const int ERR_INTERNAL = -3;").unwrap();
    writeln!(out, "        public const int ERR_INVALID_PARAMETER = -4;").unwrap();
    writeln!(out, "        public const int ERR_OUT_OF_MEMORY = -5;").unwrap();
    writeln!(out, "        public const int ERR_NOT_IMPLEMENTED = -6;").unwrap();
    writeln!(out, "        public const int ERR_PANIC = -7;").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();

    // Enums
    for e in &get_exported_enums() {
        writeln!(out, "    /// <summary>{}</summary>", e.doc).unwrap();
        writeln!(out, "    public enum {}", e.name).unwrap();
        writeln!(out, "    {{").unwrap();
        for (name, value) in &e.variants {
            writeln!(out, "        {} = {},", name, value).unwrap();
        }
        writeln!(out, "    }}").unwrap();
        writeln!(out).unwrap();
    }

    // Structs
    for s in &get_exported_structs() {
        writeln!(out, "    /// <summary>{}</summary>", s.doc).unwrap();
        writeln!(out, "    [StructLayout(LayoutKind.Sequential)]").unwrap();
        writeln!(out, "    public struct {}", s.name).unwrap();
        writeln!(out, "    {{").unwrap();
        for f in &s.fields {
            if f.cs_type.starts_with("fixed") {
                // Fixed-size buffer
                let inner = f.cs_type.replace("fixed ", "").replace('[', " ").replace(']', "");
                let parts: Vec<&str> = inner.split_whitespace().collect();
                writeln!(out, "        public unsafe fixed {} {}[{}];", parts[0], f.name, parts[1]).unwrap();
            } else {
                writeln!(out, "        public {} {};", f.cs_type, f.name).unwrap();
            }
        }
        writeln!(out, "    }}").unwrap();
        writeln!(out).unwrap();
    }

    // Native methods
    let dll_name = "genovo_ffi";
    writeln!(out, "    public static class NativeMethods").unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        private const string DllName = \"{}\";", dll_name).unwrap();
    writeln!(out).unwrap();

    for func in &get_exported_functions() {
        writeln!(out, "        /// <summary>{}</summary>", func.doc).unwrap();
        writeln!(out, "        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]").unwrap();

        let params = func
            .params
            .iter()
            .map(|p| format!("{} {}", p.cs_type, p.name))
            .collect::<Vec<_>>()
            .join(", ");

        writeln!(
            out,
            "        public static extern {} {}({});",
            func.cs_return, func.name, params
        )
        .unwrap();
        writeln!(out).unwrap();
    }

    writeln!(out, "    }}").unwrap();

    // High-level wrapper
    writeln!(out).unwrap();
    writeln!(out, "    /// <summary>High-level C# wrapper for the Genovo engine.</summary>").unwrap();
    writeln!(out, "    public class Engine : IDisposable").unwrap();
    writeln!(out, "    {{").unwrap();
    writeln!(out, "        private bool _disposed;").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        public Engine()").unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            int result = NativeMethods.genovo_init();").unwrap();
    writeln!(out, "            if (result != ErrorCodes.OK)").unwrap();
    writeln!(out, "                throw new InvalidOperationException($\"Failed to initialize engine: {{result}}\");").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        public static uint Version => NativeMethods.genovo_version();").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "        public void Dispose()").unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            if (!_disposed)").unwrap();
    writeln!(out, "            {{").unwrap();
    writeln!(out, "                NativeMethods.genovo_shutdown();").unwrap();
    writeln!(out, "                _disposed = true;").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();

    writeln!(out, "}}").unwrap();

    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_c_header_not_empty() {
        let header = generate_c_header();
        assert!(!header.is_empty());
        assert!(header.contains("#ifndef GENOVO_BINDINGS_H"));
        assert!(header.contains("#endif"));
        assert!(header.contains("genovo_init"));
        assert!(header.contains("FfiVec3"));
        assert!(header.contains("FFI_OK"));
    }

    #[test]
    fn test_generate_c_header_has_structs() {
        let header = generate_c_header();
        assert!(header.contains("typedef struct"));
        assert!(header.contains("FfiQuat"));
        assert!(header.contains("FfiTransform"));
    }

    #[test]
    fn test_generate_c_header_has_enums() {
        let header = generate_c_header();
        assert!(header.contains("typedef enum"));
        assert!(header.contains("FfiAssetStatus"));
    }

    #[test]
    fn test_generate_c_header_has_extern_c() {
        let header = generate_c_header();
        assert!(header.contains("extern \"C\""));
    }

    #[test]
    fn test_generate_cpp_wrapper_not_empty() {
        let wrapper = generate_cpp_wrapper();
        assert!(!wrapper.is_empty());
        assert!(wrapper.contains("#pragma once"));
        assert!(wrapper.contains("namespace genovo"));
        assert!(wrapper.contains("class PhysicsWorld"));
        assert!(wrapper.contains("class AudioMixer"));
        assert!(wrapper.contains("check_result"));
    }

    #[test]
    fn test_generate_cpp_wrapper_has_raii() {
        let wrapper = generate_cpp_wrapper();
        assert!(wrapper.contains("~PhysicsWorld()"));
        assert!(wrapper.contains("~AudioMixer()"));
        assert!(wrapper.contains("= delete"));
    }

    #[test]
    fn test_generate_python_bindings_not_empty() {
        let bindings = generate_python_bindings();
        assert!(!bindings.is_empty());
        assert!(bindings.contains("import ctypes"));
        assert!(bindings.contains("class FfiVec3(Structure):"));
        assert!(bindings.contains("_lib.genovo_init"));
        assert!(bindings.contains("class Engine:"));
        assert!(bindings.contains("FFI_OK = 0"));
    }

    #[test]
    fn test_generate_python_bindings_has_struct_fields() {
        let bindings = generate_python_bindings();
        assert!(bindings.contains("_fields_"));
        assert!(bindings.contains("c_float"));
    }

    #[test]
    fn test_generate_csharp_bindings_not_empty() {
        let bindings = generate_csharp_bindings();
        assert!(!bindings.is_empty());
        assert!(bindings.contains("namespace Genovo.Interop"));
        assert!(bindings.contains("[DllImport"));
        assert!(bindings.contains("public struct FfiVec3"));
        assert!(bindings.contains("class Engine : IDisposable"));
        assert!(bindings.contains("ErrorCodes"));
    }

    #[test]
    fn test_generate_csharp_bindings_has_enums() {
        let bindings = generate_csharp_bindings();
        assert!(bindings.contains("public enum FfiAssetStatus"));
        assert!(bindings.contains("public enum FfiEventType"));
    }

    #[test]
    fn test_exported_functions_list() {
        let funcs = get_exported_functions();
        assert!(!funcs.is_empty());
        let names: Vec<&str> = funcs.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"genovo_init"));
        assert!(names.contains(&"genovo_shutdown"));
        assert!(names.contains(&"genovo_version"));
    }

    #[test]
    fn test_exported_structs_list() {
        let structs = get_exported_structs();
        assert!(!structs.is_empty());
        let names: Vec<&str> = structs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"FfiVec3"));
        assert!(names.contains(&"FfiQuat"));
        assert!(names.contains(&"FfiTransform"));
        assert!(names.contains(&"FfiMat4"));
        assert!(names.contains(&"FfiColor"));
    }

    #[test]
    fn test_exported_enums_list() {
        let enums = get_exported_enums();
        assert!(!enums.is_empty());
        let names: Vec<&str> = enums.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"FfiAssetStatus"));
        assert!(names.contains(&"FfiEventType"));
    }
}
