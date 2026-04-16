//! Build script for genovo-ffi.
//!
//! Uses cbindgen to generate a C header file (`genovo.h`) from the Rust FFI
//! functions defined in `src/lib.rs`, and configures platform-specific library
//! search paths for optional C++ dependencies (PhysX, FMOD, Wwise).
//!
//! The generated header is placed at `include/genovo.h` relative to this
//! crate's manifest directory so that C/C++ consumers can include it directly.

fn main() {
    let crate_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo");
    let crate_path = std::path::Path::new(&crate_dir);
    let output_dir = crate_path.join("include");

    // -----------------------------------------------------------------------
    // Create the include directory if it does not exist
    // -----------------------------------------------------------------------
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        println!(
            "cargo:warning=Failed to create include directory: {}. \
             Header generation will be skipped.",
            e
        );
    }

    // -----------------------------------------------------------------------
    // Load cbindgen configuration
    // -----------------------------------------------------------------------
    let cbindgen_toml = crate_path.join("cbindgen.toml");
    let config = if cbindgen_toml.exists() {
        match cbindgen::Config::from_file(&cbindgen_toml) {
            Ok(cfg) => {
                println!(
                    "cargo:warning=Loaded cbindgen config from {}",
                    cbindgen_toml.display()
                );
                cfg
            }
            Err(e) => {
                println!(
                    "cargo:warning=Failed to parse cbindgen.toml ({}), using defaults",
                    e
                );
                cbindgen::Config::default()
            }
        }
    } else {
        println!("cargo:warning=cbindgen.toml not found, using defaults");
        create_default_config()
    };

    // -----------------------------------------------------------------------
    // Generate the C header
    // -----------------------------------------------------------------------
    let header_path = output_dir.join("genovo.h");

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&header_path);
            println!(
                "cargo:warning=Generated C header at {}",
                header_path.display()
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=cbindgen failed to generate header: {}. \
                 This is non-fatal; the library will still build but the \
                 header file will not be updated.",
                e
            );
        }
    }

    // -----------------------------------------------------------------------
    // Platform-specific C++ library linking (opt-in via environment vars)
    // -----------------------------------------------------------------------
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // PhysX SDK (optional -- set GENOVO_PHYSX_DIR to enable)
    if let Ok(physx_dir) = std::env::var("GENOVO_PHYSX_DIR") {
        configure_physx_linking(&target_os, &physx_dir);
    }

    // FMOD Studio SDK (optional -- set GENOVO_FMOD_DIR to enable)
    if let Ok(fmod_dir) = std::env::var("GENOVO_FMOD_DIR") {
        configure_fmod_linking(&target_os, &fmod_dir);
    }

    // Wwise SDK (optional -- set GENOVO_WWISE_DIR to enable)
    if let Ok(wwise_dir) = std::env::var("GENOVO_WWISE_DIR") {
        configure_wwise_linking(&target_os, &wwise_dir);
    }

    // -----------------------------------------------------------------------
    // Re-run triggers
    // -----------------------------------------------------------------------
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/physx_bridge.rs");
    println!("cargo:rerun-if-changed=src/audio_bridge.rs");
    println!("cargo:rerun-if-changed=src/engine_ffi.rs");
    println!("cargo:rerun-if-changed=src/cpp_bridge.rs");
    println!("cargo:rerun-if-changed=src/bindings_gen.rs");
    println!("cargo:rerun-if-env-changed=GENOVO_PHYSX_DIR");
    println!("cargo:rerun-if-env-changed=GENOVO_FMOD_DIR");
    println!("cargo:rerun-if-env-changed=GENOVO_WWISE_DIR");
}

// ===========================================================================
// Default cbindgen config (fallback when cbindgen.toml is missing)
// ===========================================================================

fn create_default_config() -> cbindgen::Config {
    let mut config = cbindgen::Config::default();
    config.language = cbindgen::Language::C;
    config.include_guard = Some("GENOVO_H".to_string());
    config.cpp_compat = true;
    config.include_version = true;
    config.autogen_warning = Some(
        "/* WARNING: This file is auto-generated by cbindgen. Do not modify manually. */"
            .to_string(),
    );
    config.header = Some(
        r#"/* ========================================================================
 * genovo.h - Auto-generated C API for the Genovo game engine.
 * ======================================================================== */"#
            .to_string(),
    );
    config.trailer = Some("/* End of genovo.h */".to_string());

    // Enum style: prefix variants with the enum name in SCREAMING_SNAKE_CASE.
    config.enumeration.rename_variants =
        cbindgen::RenameRule::ScreamingSnakeCase;
    config.enumeration.prefix_with_name = true;

    config
}

// ===========================================================================
// PhysX linking configuration
// ===========================================================================

fn configure_physx_linking(target_os: &str, physx_dir: &str) {
    println!("cargo:warning=Configuring PhysX linking from {}", physx_dir);

    let lib_subdir = match target_os {
        "windows" => "bin/win.x86_64.vc143/release",
        "linux" => "bin/linux.clang/release",
        "macos" => "bin/mac.x86_64/release",
        other => {
            println!(
                "cargo:warning=PhysX: unsupported target OS '{}'",
                other
            );
            return;
        }
    };

    let lib_path = format!("{}/{}", physx_dir, lib_subdir);
    println!("cargo:rustc-link-search=native={}", lib_path);

    let libs = [
        "PhysX",
        "PhysXCommon",
        "PhysXFoundation",
        "PhysXCooking",
        "PhysXExtensions",
        "PhysXCharacterKinematic",
    ];

    for lib in &libs {
        let lib_name = if target_os == "windows" {
            format!("{}_64", lib)
        } else {
            lib.to_string()
        };
        println!("cargo:rustc-link-lib=static={}", lib_name);
    }

    // PhysX include path (for any cc-compiled C++ bridge files in the future).
    println!("cargo:include={}/include", physx_dir);
}

// ===========================================================================
// FMOD linking configuration
// ===========================================================================

fn configure_fmod_linking(target_os: &str, fmod_dir: &str) {
    println!("cargo:warning=Configuring FMOD linking from {}", fmod_dir);

    match target_os {
        "windows" => {
            println!(
                "cargo:rustc-link-search=native={}/api/core/lib/x64",
                fmod_dir
            );
            println!(
                "cargo:rustc-link-search=native={}/api/studio/lib/x64",
                fmod_dir
            );
            println!("cargo:rustc-link-lib=dylib=fmod");
            println!("cargo:rustc-link-lib=dylib=fmodstudio");
        }
        "linux" => {
            println!(
                "cargo:rustc-link-search=native={}/api/core/lib/x86_64",
                fmod_dir
            );
            println!(
                "cargo:rustc-link-search=native={}/api/studio/lib/x86_64",
                fmod_dir
            );
            println!("cargo:rustc-link-lib=dylib=fmod");
            println!("cargo:rustc-link-lib=dylib=fmodstudio");
        }
        "macos" => {
            println!(
                "cargo:rustc-link-search=native={}/api/core/lib",
                fmod_dir
            );
            println!(
                "cargo:rustc-link-search=native={}/api/studio/lib",
                fmod_dir
            );
            println!("cargo:rustc-link-lib=dylib=fmod");
            println!("cargo:rustc-link-lib=dylib=fmodstudio");
        }
        other => {
            println!(
                "cargo:warning=FMOD: unsupported target OS '{}'",
                other
            );
        }
    }
}

// ===========================================================================
// Wwise linking configuration
// ===========================================================================

fn configure_wwise_linking(target_os: &str, wwise_dir: &str) {
    println!(
        "cargo:warning=Configuring Wwise linking from {}",
        wwise_dir
    );

    let profile = if cfg!(debug_assertions) {
        "Profile"
    } else {
        "Release"
    };

    match target_os {
        "windows" => {
            let lib_path = format!(
                "{}/SDK/x64_{}/lib",
                wwise_dir, profile
            );
            println!("cargo:rustc-link-search=native={}", lib_path);
            let wwise_libs = [
                "AkSoundEngine",
                "AkMusicEngine",
                "AkStreamMgr",
                "AkMemoryMgr",
                "AkSpatialAudio",
            ];
            for lib in &wwise_libs {
                println!("cargo:rustc-link-lib=static={}", lib);
            }
        }
        "linux" | "macos" => {
            let platform = if target_os == "linux" {
                "Linux_x64"
            } else {
                "Mac"
            };
            let lib_path = format!(
                "{}/SDK/{}_{}/lib",
                wwise_dir, platform, profile
            );
            println!("cargo:rustc-link-search=native={}", lib_path);
            println!("cargo:rustc-link-lib=static=AkSoundEngine");
        }
        other => {
            println!(
                "cargo:warning=Wwise: unsupported target OS '{}'",
                other
            );
        }
    }
}
