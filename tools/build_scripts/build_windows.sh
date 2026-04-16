#!/usr/bin/env bash
# =============================================================================
# Genovo Engine - Windows Build Script
# =============================================================================
# Prerequisites:
#   - Rust toolchain (rustup, cargo) with stable-x86_64-pc-windows-msvc
#   - CMake 3.20+
#   - Visual Studio 2022 Build Tools (for MSVC linker)
#   - Vulkan SDK (optional, for shader compilation)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_TYPE="${1:-release}"

echo "=== Genovo Engine - Windows Build ==="
echo "  Project root: $PROJECT_ROOT"
echo "  Build type:   $BUILD_TYPE"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Build Rust crates
# ---------------------------------------------------------------------------
echo "--- Building Rust workspace ---"

cd "$PROJECT_ROOT"

if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release
elif [ "$BUILD_TYPE" = "release-with-debug" ]; then
    cargo build --profile release-with-debug
else
    cargo build
fi

echo "Rust build complete."
echo ""

# ---------------------------------------------------------------------------
# Step 2: Build C++ FFI layer (if CMake project exists)
# ---------------------------------------------------------------------------
FFI_CMAKE="$PROJECT_ROOT/crates/genovo-ffi/CMakeLists.txt"

if [ -f "$FFI_CMAKE" ]; then
    echo "--- Building C++ FFI layer ---"

    FFI_BUILD_DIR="$PROJECT_ROOT/target/cmake-build-${BUILD_TYPE}"
    mkdir -p "$FFI_BUILD_DIR"

    CMAKE_BUILD_TYPE="Release"
    if [ "$BUILD_TYPE" = "debug" ]; then
        CMAKE_BUILD_TYPE="Debug"
    elif [ "$BUILD_TYPE" = "release-with-debug" ]; then
        CMAKE_BUILD_TYPE="RelWithDebInfo"
    fi

    cmake -S "$PROJECT_ROOT/crates/genovo-ffi" \
          -B "$FFI_BUILD_DIR" \
          -G "Visual Studio 17 2022" \
          -A x64 \
          -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
          -DCARGO_TARGET_DIR="$PROJECT_ROOT/target" \
          -DCARGO_PROFILE="$BUILD_TYPE"

    cmake --build "$FFI_BUILD_DIR" --config "$CMAKE_BUILD_TYPE" --parallel

    echo "C++ FFI build complete."
else
    echo "Skipping C++ FFI build (no CMakeLists.txt found)."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Compile shaders (if Vulkan SDK is available)
# ---------------------------------------------------------------------------
if command -v glslangValidator &> /dev/null; then
    echo "--- Compiling shaders ---"
    # TODO(BUILD): Integrate shader compilation into build pipeline
    # python "$PROJECT_ROOT/tools/shader_compiler/compile.py" \
    #     --input "$PROJECT_ROOT/assets/shaders" \
    #     --output "$PROJECT_ROOT/target/shaders" \
    #     --target vulkan
    echo "Shader compilation: TODO"
else
    echo "Skipping shader compilation (glslangValidator not found)."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Cook assets
# ---------------------------------------------------------------------------
# TODO(BUILD): Integrate asset cooking into build pipeline
# python "$PROJECT_ROOT/tools/asset_cooker/cook.py" \
#     --input "$PROJECT_ROOT/assets" \
#     --output "$PROJECT_ROOT/target/cooked" \
#     --platform windows
echo "Asset cooking: TODO"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Run tests
# ---------------------------------------------------------------------------
echo "--- Running tests ---"
cargo test --workspace
echo ""

echo "=== Windows build complete ==="
